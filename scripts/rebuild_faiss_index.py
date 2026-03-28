#!/usr/bin/env python3
"""Rebuild the FAISS index from all chunks in the SQLite database.

Purpose:
    After re-ingesting conversations (e.g., Wu batch re-ingestion with
    corrected diarization), the FAISS index becomes stale — it contains
    vectors keyed to old chunk IDs that no longer exist in the database.
    This script reads every chunk from SQLite, embeds them with BGE-M3,
    and writes a fresh FAISS index that's perfectly in sync with the DB.

Usage (RunPod — fast, GPU-accelerated):
    # Upload current DB to RunPod first:
    #   rsync -avz ~/Documents/Extracosmic_Commons/data/extracosmic.db \
    #     $RUNPOD:/workspace/extracosmic-commons/data/
    #
    # Then on RunPod:
    python scripts/rebuild_faiss_index.py --data-dir data
    #
    # Download rebuilt index back to Mac:
    #   rsync -avz $RUNPOD:/workspace/extracosmic-commons/data/faiss_index/ \
    #     ~/Documents/Extracosmic_Commons/data/faiss_index/

Usage (macOS — slower, subprocess isolation):
    python scripts/rebuild_faiss_index.py --data-dir data

Architecture:
    1. Reads all chunk IDs and texts from SQLite (15,772 chunks expected)
    2. Embeds in batches of 64 using BGE-M3 (1024-dim, L2-normalized)
    3. Builds a new FAISSIndexFlatIP from scratch
    4. Saves index.faiss + chunk_ids.json to data/faiss_index/
    5. Also rebuilds the BM25 keyword index for consistency

The old index is backed up to faiss_index.bak/ before overwriting.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def rebuild_faiss(data_dir: Path) -> dict:
    """Rebuild FAISS index from all chunks in the database.

    Returns a summary dict with counts and timing.
    """
    from extracosmic_commons.database import Database
    from extracosmic_commons.embeddings import EmbeddingPipeline
    from extracosmic_commons.index import FAISSIndex

    db_path = data_dir / "extracosmic.db"
    index_dir = data_dir / "faiss_index"
    backup_dir = data_dir / "faiss_index.bak"

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    # --- Step 1: Read all chunks from SQLite ---
    log.info("Reading chunks from database...")
    db = Database(db_path)
    rows = db.conn.execute("SELECT id, text FROM chunks ORDER BY source_id, rowid").fetchall()
    chunk_ids = [r["id"] for r in rows]
    texts = [r["text"] for r in rows]
    log.info(f"Found {len(chunk_ids)} chunks to embed")

    if not chunk_ids:
        log.warning("No chunks found in database — nothing to index")
        db.close()
        return {"chunks": 0, "elapsed_s": 0}

    # --- Step 2: Embed all texts with BGE-M3 ---
    log.info("Initializing BGE-M3 embedding pipeline...")
    embedder = EmbeddingPipeline()

    t0 = time.time()
    batch_size = 64  # Tuned for A40 48GB; macOS subprocess uses smaller batches internally

    def progress(done, total):
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        log.info(f"  Embedded {done}/{total} ({done/total*100:.1f}%) — {rate:.0f} chunks/s, ETA {eta:.0f}s")

    log.info(f"Embedding {len(texts)} chunks (batch_size={batch_size})...")
    embeddings = embedder.embed_batch(texts, batch_size=batch_size, progress_callback=progress)
    embed_time = time.time() - t0
    log.info(f"Embedding complete: {len(texts)} chunks in {embed_time:.1f}s ({len(texts)/embed_time:.0f} chunks/s)")

    # --- Step 3: Build new FAISS index ---
    log.info("Building FAISS index...")
    index = FAISSIndex(dimension=embedder.dimension)
    index.add_batch(chunk_ids, embeddings)
    log.info(f"FAISS index built: {index.size} vectors, {embedder.dimension} dimensions")

    # --- Step 4: Backup old index and save new one ---
    if index_dir.exists():
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.move(str(index_dir), str(backup_dir))
        log.info(f"Old index backed up to {backup_dir}")

    index.save(index_dir)
    index_size_mb = (index_dir / "index.faiss").stat().st_size / (1024 * 1024)
    log.info(f"New index saved to {index_dir} ({index_size_mb:.1f} MB)")

    # --- Step 5: Rebuild BM25 index too ---
    log.info("Rebuilding BM25 keyword index...")
    from extracosmic_commons.bm25 import BM25Index

    bm25 = BM25Index()
    bm25.build(chunk_ids, texts)
    bm25_dir = data_dir / "bm25_index"
    bm25.save(bm25_dir)
    log.info(f"BM25 index saved: {bm25.size} documents")

    # --- Step 6: Verify ---
    log.info("Verifying index consistency...")
    with open(index_dir / "chunk_ids.json") as f:
        saved_ids = json.load(f)

    db_ids = set(r["id"] for r in db.conn.execute("SELECT id FROM chunks").fetchall())
    faiss_ids = set(saved_ids)

    missing = db_ids - faiss_ids
    stale = faiss_ids - db_ids

    if missing:
        log.error(f"VERIFICATION FAILED: {len(missing)} DB chunks missing from FAISS")
    elif stale:
        log.error(f"VERIFICATION FAILED: {len(stale)} stale IDs in FAISS")
    else:
        log.info(f"VERIFIED: FAISS index perfectly matches DB ({len(db_ids)} chunks)")

    db.close()

    total_time = time.time() - t0
    return {
        "chunks": len(chunk_ids),
        "embed_time_s": round(embed_time, 1),
        "total_time_s": round(total_time, 1),
        "index_size_mb": round(index_size_mb, 1),
        "verified": len(missing) == 0 and len(stale) == 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Rebuild FAISS index from all DB chunks")
    parser.add_argument("--data-dir", default="data", help="Path to data directory (default: data)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    log.info(f"Rebuilding FAISS index from {data_dir / 'extracosmic.db'}")

    result = rebuild_faiss(data_dir)

    print(f"\n{'=' * 50}")
    print(f"  FAISS Index Rebuild Complete")
    print(f"  Chunks embedded:  {result['chunks']}")
    print(f"  Embedding time:   {result['embed_time_s']}s")
    print(f"  Total time:       {result['total_time_s']}s")
    print(f"  Index size:       {result['index_size_mb']} MB")
    print(f"  Verified:         {'✅ PASS' if result['verified'] else '❌ FAIL'}")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
