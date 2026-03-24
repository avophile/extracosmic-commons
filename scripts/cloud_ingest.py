#!/usr/bin/env python3
"""Cloud ingestion script for Extracosmic Commons.

Run this on a GPU cloud machine (RunPod, Lambda, etc.) to embed the full
corpus much faster than on a Mac CPU. Usage:

    # On cloud machine:
    pip install -e ".[dev]"
    python scripts/cloud_ingest.py --corpus-dir /path/to/corpus --data-dir data

    # Then download data/ back to your Mac:
    scp -r cloud:/path/to/extracosmic-commons/data/ ./data/

The script ingests:
1. Lecture transcripts (.md files)
2. Bilingual JSON (clean_text.json)
3. Zotero export directories
4. Scholarly workbench metadata
5. Standalone PDF files/directories
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from extracosmic_commons.database import Database
from extracosmic_commons.embeddings import EmbeddingPipeline
from extracosmic_commons.index import FAISSIndex
from extracosmic_commons.ingest.bilingual import BilingualIngester
from extracosmic_commons.ingest.pdf import PDFIngester
from extracosmic_commons.ingest.transcript import TranscriptIngester
from extracosmic_commons.ingest.workbench import WorkbenchImporter
from extracosmic_commons.ingest.zotero import ZoteroImporter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def ingest_transcripts(paths: list[Path], db, embedder, index):
    """Ingest transcript markdown files."""
    ingester = TranscriptIngester()
    for p in paths:
        try:
            t0 = time.time()
            source = ingester.ingest(p, db, embedder, index)
            chunks = db.get_chunks_by_source(source.id)
            logger.info(f"Transcript: {source.title} — {len(chunks)} chunks in {time.time()-t0:.1f}s")
        except Exception as e:
            logger.error(f"Failed transcript {p.name}: {e}")


def ingest_bilingual(path: Path, db, embedder, index):
    """Ingest bilingual JSON."""
    ingester = BilingualIngester()
    try:
        t0 = time.time()
        de_src, en_src = ingester.ingest(path, db, embedder, index)
        de_n = len(db.get_chunks_by_source(de_src.id))
        en_n = len(db.get_chunks_by_source(en_src.id))
        logger.info(f"Bilingual: {de_n} DE + {en_n} EN chunks in {time.time()-t0:.1f}s")
    except Exception as e:
        logger.error(f"Failed bilingual {path.name}: {e}")


def ingest_zotero(paths: list[Path], db, embedder, index):
    """Ingest Zotero export directories."""
    importer = ZoteroImporter()
    for p in paths:
        t0 = time.time()
        sources = importer.import_collection(
            p, db, embedder, index,
            progress_callback=lambda d, t, title: logger.info(f"  [{d}/{t}] {title[:60]}") if d % 10 == 0 else None,
        )
        logger.info(f"Zotero {p.name}: {len(sources)} sources in {time.time()-t0:.1f}s")


def ingest_workbench(path: Path, db, embedder, index):
    """Ingest scholarly workbench."""
    importer = WorkbenchImporter()
    t0 = time.time()
    sources = importer.import_all(
        path, db, embedder, index,
        progress_callback=lambda d, t, title: logger.info(f"  [{d}/{t}] {title[:60]}") if d % 10 == 0 else None,
    )
    logger.info(f"Workbench: {len(sources)} sources in {time.time()-t0:.1f}s")


def ingest_pdfs(paths: list[Path], db, embedder, index):
    """Ingest standalone PDF files."""
    ingester = PDFIngester()
    for p in paths:
        try:
            t0 = time.time()
            source = ingester.ingest(p, db, embedder, index)
            chunks = db.get_chunks_by_source(source.id)
            logger.info(f"PDF: {source.title[:60]} — {len(chunks)} chunks in {time.time()-t0:.1f}s")
        except Exception as e:
            logger.error(f"Failed PDF {p.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Cloud corpus ingestion for Extracosmic Commons")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Output data directory")
    parser.add_argument("--corpus-dir", type=Path, help="Root directory containing all corpus files")
    parser.add_argument("--transcripts", nargs="*", type=Path, help="Transcript .md files")
    parser.add_argument("--bilingual", type=Path, help="clean_text.json path")
    parser.add_argument("--zotero", nargs="*", type=Path, help="Zotero export directories")
    parser.add_argument("--workbench", type=Path, help="Scholarly workbench root")
    parser.add_argument("--pdfs", nargs="*", type=Path, help="PDF files or directories")
    args = parser.parse_args()

    # Init components
    db = Database(args.data_dir / "extracosmic.db")
    embedder = EmbeddingPipeline()
    index_path = args.data_dir / "faiss_index"
    if (index_path / "index.faiss").exists():
        index = FAISSIndex(index_path=index_path)
    else:
        index = FAISSIndex(dimension=1024)

    logger.info(f"Starting ingestion. Device: {embedder.model.device}")
    stats_before = db.get_stats()
    logger.info(f"Before: {stats_before['sources']} sources, {stats_before['chunks']} chunks")

    # Auto-discover corpus if --corpus-dir is given
    if args.corpus_dir:
        corpus = args.corpus_dir

        # Transcripts
        transcripts = []
        for pattern in ["*Complete_Transcript*.md", "*Master_Transcript*.md"]:
            transcripts.extend(corpus.rglob(pattern))
        if transcripts:
            ingest_transcripts(transcripts, db, embedder, index)

        # Bilingual
        bilingual = list(corpus.rglob("clean_text.json"))
        for b in bilingual:
            ingest_bilingual(b, db, embedder, index)

        # Zotero collections
        zotero_dirs = [d for d in corpus.iterdir() if d.is_dir() and "texts" in d.name.lower()]
        if zotero_dirs:
            ingest_zotero(zotero_dirs, db, embedder, index)

    # Explicit arguments
    if args.transcripts:
        ingest_transcripts(args.transcripts, db, embedder, index)

    if args.bilingual:
        ingest_bilingual(args.bilingual, db, embedder, index)

    if args.zotero:
        ingest_zotero(args.zotero, db, embedder, index)

    if args.workbench:
        ingest_workbench(args.workbench, db, embedder, index)

    if args.pdfs:
        all_pdfs = []
        for p in args.pdfs:
            if p.is_dir():
                all_pdfs.extend(sorted(p.rglob("*.pdf")))
            elif p.suffix.lower() == ".pdf":
                all_pdfs.append(p)
        if all_pdfs:
            ingest_pdfs(all_pdfs, db, embedder, index)

    # Save and report
    index.save(index_path)
    stats_after = db.get_stats()
    logger.info(f"After: {stats_after['sources']} sources, {stats_after['chunks']} chunks")
    logger.info(f"FAISS index: {index.size} vectors")
    logger.info(f"Data saved to {args.data_dir}/")
    db.close()


if __name__ == "__main__":
    main()
