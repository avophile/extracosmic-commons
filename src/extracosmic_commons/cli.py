"""Command-line interface for Extracosmic Commons.

Usage:
    ec ingest <file>                    # Ingest a single file
    ec ingest <directory> [--recursive] # Ingest all files in a directory
    ec import-zotero <path>             # Import a Zotero export directory
    ec import-workbench <path>          # Import from scholarly workbench
    ec search <query> [options]         # Hybrid semantic + keyword search
    ec build-bm25                       # Build BM25 keyword index
    ec tag-structural                   # Tag structural refs on all chunks
    ec cite <source-id>                 # Print citation in various formats
    ec enrich                           # Enrich source metadata via web APIs
    ec stats                            # Corpus statistics
"""

from __future__ import annotations

import csv
import io
import logging
import os
import sys
from pathlib import Path

import click

from .database import Database
from .embeddings import EmbeddingPipeline
from .index import FAISSIndex


def _get_data_dir() -> Path:
    """Resolve the data directory from env or default."""
    return Path(os.environ.get("EC_DATA_DIR", "data"))


def _init_components(data_dir: Path | None = None):
    """Initialize database, embedder, and index."""
    dd = data_dir or _get_data_dir()
    db = Database(dd / "extracosmic.db")
    embedder = EmbeddingPipeline()
    index_path = dd / "faiss_index"
    if (index_path / "index.faiss").exists():
        index = FAISSIndex(index_path=index_path)
    else:
        index = FAISSIndex(dimension=1024)
    return db, embedder, index, dd


def _init_bm25(dd: Path):
    """Load BM25 index if it exists."""
    from .bm25 import BM25Index

    bm25_path = dd / "bm25_index"
    if (bm25_path / "bm25_corpus.pkl").exists():
        return BM25Index(index_path=bm25_path)
    return None


@click.group()
@click.option("--data-dir", type=click.Path(), envvar="EC_DATA_DIR", default=None)
@click.pass_context
def main(ctx, data_dir):
    """Extracosmic Commons — scholarly research platform."""
    ctx.ensure_object(dict)
    if data_dir:
        ctx.obj["data_dir"] = Path(data_dir)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--recursive", "-r", is_flag=True, help="Recurse into subdirectories")
@click.pass_context
def ingest(ctx, path, recursive):
    """Ingest a file or directory."""
    from .ingest.bilingual import BilingualIngester
    from .ingest.pdf import PDFIngester
    from .ingest.transcript import TranscriptIngester

    data_dir = ctx.obj.get("data_dir")
    db, embedder, index, dd = _init_components(data_dir)

    target = Path(path)

    if target.is_file():
        files = [target]
    elif target.is_dir():
        pattern = "**/*" if recursive else "*"
        files = sorted(target.glob(pattern))
        files = [f for f in files if f.is_file()]
    else:
        click.echo(f"Error: {path} is not a file or directory", err=True)
        return

    supported = {".md", ".pdf", ".json"}
    files = [f for f in files if f.suffix.lower() in supported]

    if not files:
        click.echo("No supported files found (.md, .pdf, .json)")
        return

    transcript_ingester = TranscriptIngester()
    pdf_ingester = PDFIngester()
    bilingual_ingester = BilingualIngester()

    total_chunks = 0
    for f in files:
        try:
            if f.suffix.lower() == ".md":
                source = transcript_ingester.ingest(f, db, embedder, index)
                chunks = db.get_chunks_by_source(source.id)
                click.echo(f"  Ingested transcript: {source.title} ({len(chunks)} chunks)")
                total_chunks += len(chunks)
            elif f.suffix.lower() == ".pdf":
                source = pdf_ingester.ingest(f, db, embedder, index)
                chunks = db.get_chunks_by_source(source.id)
                click.echo(f"  Ingested PDF: {source.title} ({len(chunks)} chunks)")
                total_chunks += len(chunks)
            elif f.suffix.lower() == ".json" and "clean_text" in f.name:
                de_src, en_src = bilingual_ingester.ingest(f, db, embedder, index)
                de_chunks = db.get_chunks_by_source(de_src.id)
                en_chunks = db.get_chunks_by_source(en_src.id)
                n = len(de_chunks) + len(en_chunks)
                click.echo(f"  Ingested bilingual: {de_src.title} + {en_src.title} ({n} chunks)")
                total_chunks += n
        except Exception as e:
            click.echo(f"  Failed: {f.name}: {e}", err=True)

    index.save(dd / "faiss_index")
    click.echo(f"\nDone. {len(files)} files, {total_chunks} chunks total.")


@main.command("import-zotero")
@click.argument("path", type=click.Path(exists=True))
@click.pass_context
def import_zotero(ctx, path):
    """Import a Zotero export directory."""
    from .ingest.zotero import ZoteroImporter

    data_dir = ctx.obj.get("data_dir")
    db, embedder, index, dd = _init_components(data_dir)

    importer = ZoteroImporter()

    def progress(done, total, title):
        if done < total:
            click.echo(f"  [{done}/{total}] {title}")
        else:
            click.echo(f"  [{done}/{total}] Complete")

    sources = importer.import_collection(
        Path(path), db, embedder, index, progress_callback=progress,
    )

    index.save(dd / "faiss_index")
    click.echo(f"\nImported {len(sources)} documents from Zotero collection.")


@main.command("import-workbench")
@click.argument("path", type=click.Path(exists=True))
@click.pass_context
def import_workbench(ctx, path):
    """Import documents from scholarly workbench."""
    from .ingest.workbench import WorkbenchImporter

    data_dir = ctx.obj.get("data_dir")
    db, embedder, index, dd = _init_components(data_dir)

    importer = WorkbenchImporter()

    def progress(done, total, title):
        if done < total:
            click.echo(f"  [{done}/{total}] {title}")
        else:
            click.echo(f"  [{done}/{total}] Complete")

    sources = importer.import_all(
        Path(path), db, embedder, index, progress_callback=progress,
    )

    index.save(dd / "faiss_index")
    click.echo(f"\nImported {len(sources)} documents from scholarly workbench.")


@main.command()
@click.argument("query")
@click.option("-k", "--top-k", default=10, help="Number of results")
@click.option("--lecturer", help="Filter by lecturer name")
@click.option("--language", help="Filter by language code (en, de)")
@click.option("--type", "source_type", help="Filter by source type (pdf, transcript)")
@click.option("--bilingual", is_flag=True, help="Show bilingual pairs")
@click.option("--ref", multiple=True, help="Structural ref filter (key=value)")
@click.pass_context
def search(ctx, query, top_k, lecturer, language, source_type, bilingual, ref):
    """Search the corpus with hybrid semantic + keyword matching."""
    from .citations import CitationFormatter
    from .search import SearchEngine

    data_dir = ctx.obj.get("data_dir")
    db, embedder, index, dd = _init_components(data_dir)
    bm25 = _init_bm25(dd)
    engine = SearchEngine(db, embedder, index, bm25=bm25)
    formatter = CitationFormatter()

    filters = {}
    if lecturer:
        filters["lecturer"] = lecturer
    if language:
        filters["language"] = language
    if source_type:
        filters["source_type"] = source_type
    if ref:
        structural = {}
        for r in ref:
            if "=" in r:
                k, v = r.split("=", 1)
                structural[k] = v
        if structural:
            filters["structural_ref"] = structural

    results = engine.search(query, top_k=top_k, bilingual=bilingual, **filters)

    if not results:
        click.echo("No results found.")
        return

    hybrid_label = " (hybrid)" if bm25 else " (semantic)"

    for i, r in enumerate(results, 1):
        click.echo(f"\n{'─' * 60}")
        click.echo(f"  [{i}] Score: {r.score:.4f}{hybrid_label}")
        click.echo(f"  Source: {r.source.title}")
        if r.source.author:
            click.echo(f"  Author: {', '.join(r.source.author)}")

        # Location info
        if r.chunk.youtube_url:
            click.echo(f"  Timestamp: [{r.chunk.youtube_timestamp}]({r.chunk.youtube_url})")
        if r.chunk.pdf_page:
            click.echo(f"  Page: {r.chunk.pdf_page}")
        if r.chunk.lecturer:
            click.echo(f"  Lecturer: {r.chunk.lecturer}, Lecture {r.chunk.lecture_number}")
        if r.chunk.structural_ref:
            ref_str = ", ".join(f"{k}={v}" for k, v in r.chunk.structural_ref.items())
            click.echo(f"  Ref: {ref_str}")

        # Citation
        cite = formatter.chicago(r.source)
        click.echo(f"  Cite: {cite}")

        # Text snippet
        text = r.chunk.text[:300]
        if len(r.chunk.text) > 300:
            text += "..."
        click.echo(f"\n  {text}")

        # Bilingual pair
        if r.paired_chunk:
            lang = r.paired_chunk.language.upper()
            paired_text = r.paired_chunk.text[:200]
            if len(r.paired_chunk.text) > 200:
                paired_text += "..."
            click.echo(f"\n  [{lang}] {paired_text}")

    click.echo(f"\n{'─' * 60}")
    click.echo(f"{len(results)} results for: {query}")


@main.command("build-bm25")
@click.pass_context
def build_bm25(ctx):
    """Build the BM25 keyword index from all chunks."""
    from .bm25 import BM25Index

    data_dir = ctx.obj.get("data_dir")
    dd = data_dir or _get_data_dir()
    db = Database(dd / "extracosmic.db")

    click.echo("Reading chunks from database...")
    rows = db.conn.execute("SELECT id, text FROM chunks").fetchall()
    chunk_ids = [r["id"] for r in rows]
    texts = [r["text"] for r in rows]

    click.echo(f"Building BM25 index from {len(chunk_ids)} chunks...")
    bm25 = BM25Index()
    bm25.build(chunk_ids, texts)

    save_path = dd / "bm25_index"
    bm25.save(save_path)
    click.echo(f"BM25 index saved to {save_path} ({bm25.size} documents)")
    db.close()


@main.command("tag-structural")
@click.option("--source-id", help="Tag only a specific source")
@click.option("--dry-run", is_flag=True, help="Show what would be tagged")
@click.option("--source-type", help="Filter by source type")
@click.pass_context
def tag_structural(ctx, source_id, dry_run, source_type):
    """Tag structural references on chunks (headings, §, chapters, etc.)."""
    from .structural import StructuralTagger

    data_dir = ctx.obj.get("data_dir")
    dd = data_dir or _get_data_dir()
    db = Database(dd / "extracosmic.db")
    logging.basicConfig(level=logging.INFO)

    tagger = StructuralTagger()

    if source_id:
        n = tagger.tag_source(source_id, db)
        click.echo(f"Tagged {n} chunks for source {source_id}")
    else:
        n = tagger.tag_corpus(db, source_type=source_type, dry_run=dry_run)
        action = "Would tag" if dry_run else "Tagged"
        click.echo(f"\n{action} {n} chunks across corpus.")

    db.close()


@main.command()
@click.argument("source_id", required=False)
@click.option("--format", "fmt", default="chicago", type=click.Choice(["chicago", "bibtex", "ris", "csv"]))
@click.option("--all", "all_sources", is_flag=True, help="Export all sources")
@click.option("--search", "search_term", help="Filter sources by title/author")
@click.pass_context
def cite(ctx, source_id, fmt, all_sources, search_term):
    """Print citations for sources."""
    from .citations import CitationFormatter

    data_dir = ctx.obj.get("data_dir")
    dd = data_dir or _get_data_dir()
    db = Database(dd / "extracosmic.db")
    formatter = CitationFormatter()

    if source_id:
        source = db.get_source(source_id)
        if not source:
            click.echo(f"Source not found: {source_id}", err=True)
            return
        sources = [source]
    elif all_sources or search_term:
        sources = db.get_all_sources()
        if search_term:
            term = search_term.lower()
            sources = [
                s for s in sources
                if term in s.title.lower()
                or any(term in a.lower() for a in s.author)
            ]
    else:
        click.echo("Provide a source ID, --all, or --search <term>", err=True)
        return

    if not sources:
        click.echo("No matching sources found.")
        return

    if fmt == "chicago":
        for s in sources:
            click.echo(formatter.chicago(s))
    elif fmt == "bibtex":
        for s in sources:
            click.echo(formatter.bibtex(s))
            click.echo()
    elif fmt == "ris":
        for s in sources:
            click.echo(formatter.ris(s))
            click.echo()
    elif fmt == "csv":
        if sources:
            rows = [formatter.csv_row(s) for s in sources]
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
            click.echo(output.getvalue())

    if len(sources) > 1:
        click.echo(f"\n{len(sources)} citations exported.", err=True)

    db.close()


@main.command()
@click.option("--source-id", help="Enrich a specific source")
@click.option("--dry-run", is_flag=True, help="Show what would be enriched")
@click.pass_context
def enrich(ctx, source_id, dry_run):
    """Enrich source metadata via DOI/ISBN extraction and web APIs."""
    from .metadata import MetadataEnricher

    data_dir = ctx.obj.get("data_dir")
    dd = data_dir or _get_data_dir()
    db = Database(dd / "extracosmic.db")
    logging.basicConfig(level=logging.INFO)

    enricher = MetadataEnricher()

    if source_id:
        source = db.get_source(source_id)
        if not source:
            click.echo(f"Source not found: {source_id}", err=True)
            return
        new_fields = enricher.enrich_source(source, db, force=True)
        if new_fields:
            click.echo(f"Enriched with: {list(new_fields.keys())}")
        else:
            click.echo("No new metadata found.")
    else:
        def progress(done, total, title):
            if done < total and done % 10 == 0:
                click.echo(f"  [{done}/{total}] {title}")

        n = enricher.enrich_corpus(db, dry_run=dry_run, progress_callback=progress)
        action = "Would enrich" if dry_run else "Enriched"
        click.echo(f"\n{action} {n} sources.")

    db.close()


@main.command()
@click.pass_context
def stats(ctx):
    """Show corpus statistics."""
    data_dir = ctx.obj.get("data_dir")
    dd = data_dir or _get_data_dir()
    db = Database(dd / "extracosmic.db")

    s = db.get_stats()

    click.echo("\n  Extracosmic Commons — Corpus Statistics\n")
    click.echo(f"  Sources:   {s['sources']}")
    click.echo(f"  Chunks:    {s['chunks']}")
    click.echo(f"  Analyses:  {s['analyses']}")

    if s["sources_by_type"]:
        click.echo("\n  By source type:")
        for t, c in sorted(s["sources_by_type"].items()):
            click.echo(f"    {t}: {c}")

    if s["chunks_by_language"]:
        click.echo("\n  By language:")
        for lang, c in sorted(s["chunks_by_language"].items()):
            click.echo(f"    {lang}: {c}")

    if s["chunks_by_lecturer"]:
        click.echo("\n  By lecturer:")
        for lect, c in sorted(s["chunks_by_lecturer"].items()):
            click.echo(f"    {lect}: {c}")

    # Index sizes
    index_path = dd / "faiss_index" / "index.faiss"
    if index_path.exists():
        size_mb = index_path.stat().st_size / (1024 * 1024)
        click.echo(f"\n  FAISS index: {size_mb:.1f} MB")

    bm25_path = dd / "bm25_index" / "bm25_corpus.pkl"
    if bm25_path.exists():
        size_mb = bm25_path.stat().st_size / (1024 * 1024)
        click.echo(f"  BM25 index: {size_mb:.1f} MB")

    # Structural tagging coverage
    tagged = db.conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE structural_ref IS NOT NULL"
    ).fetchone()[0]
    total = s["chunks"]
    if total > 0:
        pct = tagged / total * 100
        click.echo(f"\n  Structural tags: {tagged}/{total} ({pct:.1f}%)")

    click.echo()
    db.close()
