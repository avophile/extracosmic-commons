#!/usr/bin/env python3
"""Batch PDF ingestion for RunPod — reads a dedup manifest and processes
all unique PDFs through the Extracosmic Commons pipeline.

Designed for GPU-accelerated ingestion on RunPod A40. Reads the manifest
produced by dedup_scan.py, then for each unique PDF:
1. Checks if already ingested (checkpoint or DB dedup)
2. Extracts text via pypdf (CPU)
3. Creates Source + Chunks in SQLite
4. Embeds with BGE-M3 (GPU)
5. Adds embeddings to FAISS index

Supports:
- Tier filtering (--tier 1|2|3) for staged ingestion
- Checkpoint/resume (saves progress every N PDFs)
- Error logging (failed PDFs recorded, ingestion continues)
- Progress reporting with ETA

Usage:
    # Ingest tier 1 (Hegel primary texts)
    python3 scripts/batch_ingest.py \\
        --manifest data/dedup_manifest.json \\
        --tier 1 \\
        --checkpoint-every 100

    # Resume after interruption (skips already-completed hashes)
    python3 scripts/batch_ingest.py \\
        --manifest data/dedup_manifest.json \\
        --resume

    # Ingest all tiers
    python3 scripts/batch_ingest.py \\
        --manifest data/dedup_manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Add the project source to path so we can import the pipeline
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

from extracosmic_commons.ingest.pdf import PDFIngester


def load_manifest(path: Path | str) -> dict[str, dict]:
    """Load the dedup manifest from a JSON file.

    Args:
        path: Path to the manifest JSON file produced by dedup_scan.py.

    Returns:
        Dict mapping hash → manifest entry.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return json.loads(path.read_text())


def filter_by_tier(manifest: dict[str, dict], tier: int | None) -> dict[str, dict]:
    """Filter manifest entries by prioritization tier.

    Args:
        manifest: The full dedup manifest.
        tier: Tier number (1, 2, or 3) to keep. None returns all entries.

    Returns:
        Filtered manifest dict.
    """
    if tier is None:
        return manifest
    return {h: e for h, e in manifest.items() if e["tier"] == tier}


def load_checkpoint(path: Path | str) -> dict:
    """Load ingestion checkpoint from disk.

    Returns a dict with 'completed_hashes' and 'failed_hashes' lists.
    If the file doesn't exist, returns an empty default.

    Args:
        path: Path to the checkpoint JSON file.

    Returns:
        Checkpoint dict with completed_hashes and failed_hashes lists.
    """
    path = Path(path)
    if not path.exists():
        return {"completed_hashes": [], "failed_hashes": []}
    return json.loads(path.read_text())


def save_checkpoint(path: Path | str, data: dict) -> None:
    """Save ingestion checkpoint to disk.

    Args:
        path: Path to write the checkpoint JSON file.
        data: Checkpoint dict with completed_hashes and failed_hashes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


class BatchIngestionRunner:
    """Orchestrates batch PDF ingestion from a dedup manifest.

    Processes each manifest entry through PDFIngester, with checkpoint
    support for resume-after-interruption and error logging for failed
    PDFs. Designed to run on RunPod with GPU acceleration.

    Args:
        manifest: Dedup manifest dict (hash → entry).
        db: Database instance for source/chunk storage.
        embedder: EmbeddingPipeline for BGE-M3 embeddings.
        index: FAISSIndex for vector storage.
        checkpoint_path: Where to save progress checkpoints.
        errors_path: Where to log ingestion errors.
        checkpoint_every: Save checkpoint every N successfully ingested PDFs.
        progress_callback: Optional callback(done, total, title) for progress.
    """

    def __init__(
        self,
        manifest: dict[str, dict],
        db: Any,
        embedder: Any,
        index: Any,
        checkpoint_path: Path | str = "data/ingestion_checkpoint.json",
        errors_path: Path | str = "data/ingestion_errors.json",
        checkpoint_every: int = 100,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ):
        self.manifest = manifest
        self.db = db
        self.embedder = embedder
        self.index = index
        self.checkpoint_path = Path(checkpoint_path)
        self.errors_path = Path(errors_path)
        self.checkpoint_every = checkpoint_every
        self.progress_callback = progress_callback
        self.checkpoint = load_checkpoint(self.checkpoint_path)

    def _title_from_path(self, path: str) -> str:
        """Extract a human-readable title from a PDF path.

        Strips the .pdf extension and cleans up common filename patterns.
        Workbench hash-named files get a generic title since the real
        title comes from metadata (handled separately).

        Args:
            path: Filesystem path to the PDF.

        Returns:
            Cleaned-up title string.
        """
        stem = Path(path).stem
        # Workbench hash-named files — return the hash as-is (the caller
        # or WorkbenchImporter can override with metadata)
        if len(stem) == 64 and all(c in "0123456789abcdef" for c in stem):
            return f"Workbench document {stem[:12]}..."
        return stem

    def run(self, tier: int | None = None) -> dict[str, int]:
        """Execute the batch ingestion.

        Iterates over all manifest entries (optionally filtered by tier),
        skipping those already in the checkpoint or database. For each
        remaining PDF, runs PDFIngester.ingest() and records the result.

        Args:
            tier: Optional tier filter (applied on top of manifest already
                  being filtered). Usually the manifest is pre-filtered.

        Returns:
            Stats dict with keys: total, ingested, skipped_checkpoint,
            skipped_db, failed.
        """
        entries = list(self.manifest.values())
        total = len(entries)
        completed_set = set(self.checkpoint.get("completed_hashes", []))
        failed_set = set(self.checkpoint.get("failed_hashes", []))
        errors_log: list[dict] = []

        stats = {
            "total": total,
            "ingested": 0,
            "skipped_checkpoint": 0,
            "skipped_db": 0,
            "failed": 0,
        }

        ingester = PDFIngester()
        start_time = time.time()

        for i, entry in enumerate(entries):
            file_hash = entry["hash"]
            pdf_path = Path(entry["path"])
            title = self._title_from_path(entry["path"])

            # Skip if already in checkpoint
            if file_hash in completed_set or file_hash in failed_set:
                stats["skipped_checkpoint"] += 1
                if self.progress_callback:
                    self.progress_callback(i + 1, total, f"[skip] {title[:50]}")
                continue

            # Skip if already in database (by title match)
            if self.db.source_exists(title):
                completed_set.add(file_hash)
                stats["skipped_db"] += 1
                if self.progress_callback:
                    self.progress_callback(i + 1, total, f"[dedup] {title[:50]}")
                continue

            # Attempt ingestion
            try:
                source = ingester.ingest(
                    path=pdf_path,
                    db=self.db,
                    embedder=self.embedder,
                    index=self.index,
                    title=title,
                    metadata={
                        "collection": entry["collection"],
                        "tier": entry["tier"],
                        "file_hash": file_hash,
                    },
                )
                completed_set.add(file_hash)
                stats["ingested"] += 1
                logger.info(f"[{i+1}/{total}] Ingested: {title[:60]} (source={source.id[:8]})")

            except Exception as e:
                failed_set.add(file_hash)
                stats["failed"] += 1
                error_entry = {
                    "hash": file_hash,
                    "path": str(pdf_path),
                    "title": title,
                    "error": str(e),
                    "collection": entry["collection"],
                }
                errors_log.append(error_entry)
                logger.warning(f"[{i+1}/{total}] Failed: {title[:60]} — {e}")

            # Progress callback
            if self.progress_callback:
                self.progress_callback(i + 1, total, title[:50])

            # Periodic checkpoint
            if (stats["ingested"] + stats["failed"]) % self.checkpoint_every == 0:
                self._save_state(completed_set, failed_set, errors_log)

        # Final checkpoint and error log
        self._save_state(completed_set, failed_set, errors_log)

        # Report
        elapsed = time.time() - start_time
        stats["elapsed_seconds"] = round(elapsed, 1)
        stats["pdfs_per_minute"] = round(stats["ingested"] / (elapsed / 60), 1) if elapsed > 0 else 0

        return stats

    def _save_state(self, completed: set, failed: set, errors: list) -> None:
        """Persist checkpoint and error log to disk.

        Called periodically during ingestion and once at the end.

        Args:
            completed: Set of successfully ingested file hashes.
            failed: Set of failed file hashes.
            errors: List of error detail dicts.
        """
        save_checkpoint(self.checkpoint_path, {
            "completed_hashes": sorted(completed),
            "failed_hashes": sorted(failed),
        })
        if errors:
            self.errors_path.parent.mkdir(parents=True, exist_ok=True)
            self.errors_path.write_text(json.dumps(errors, indent=2))


def main():
    """CLI entry point for batch ingestion."""
    parser = argparse.ArgumentParser(
        description="Batch-ingest PDFs from a dedup manifest into Extracosmic Commons.",
    )
    parser.add_argument("--manifest", required=True, help="Path to dedup_manifest.json")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], help="Process only this tier")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="Save checkpoint every N PDFs (default: 100)")
    parser.add_argument("--data-dir", default="data", help="Data directory for DB/indexes")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load manifest
    manifest = load_manifest(args.manifest)
    if args.tier:
        manifest = filter_by_tier(manifest, args.tier)
        logger.info(f"Filtered to tier {args.tier}: {len(manifest)} entries")

    if not manifest:
        print("No entries to process.")
        return

    # Initialize pipeline components
    from extracosmic_commons.database import Database
    from extracosmic_commons.embeddings import EmbeddingPipeline
    from extracosmic_commons.index import FAISSIndex

    data_dir = Path(args.data_dir)
    db = Database(data_dir / "extracosmic.db")
    embedder = EmbeddingPipeline()

    index_path = data_dir / "faiss_index"
    if (index_path / "index.faiss").exists():
        index = FAISSIndex(index_path=index_path)
    else:
        index = FAISSIndex(dimension=1024)

    # Checkpoint paths
    checkpoint_path = data_dir / "ingestion_checkpoint.json"
    errors_path = data_dir / "ingestion_errors.json"

    def progress(done, total, title):
        pct = done / total * 100 if total else 0
        print(f"\r  [{done}/{total}] {pct:.0f}% — {title:<55}", end="", flush=True)

    runner = BatchIngestionRunner(
        manifest=manifest,
        db=db,
        embedder=embedder,
        index=index,
        checkpoint_path=checkpoint_path,
        errors_path=errors_path,
        checkpoint_every=args.checkpoint_every,
        progress_callback=progress,
    )

    print(f"\nStarting batch ingestion: {len(manifest)} PDFs")
    print(f"  Checkpoint every: {args.checkpoint_every}")
    print(f"  Resume mode: {args.resume}")
    print()

    stats = runner.run()

    # Save FAISS index
    index.save(data_dir / "faiss_index")

    print(f"\n\n=== Batch Ingestion Complete ===")
    print(f"  Total in manifest: {stats['total']}")
    print(f"  Ingested:          {stats['ingested']}")
    print(f"  Skipped (checkpoint): {stats.get('skipped_checkpoint', 0)}")
    print(f"  Skipped (DB dedup):   {stats.get('skipped_db', 0)}")
    print(f"  Failed:            {stats['failed']}")
    print(f"  Elapsed:           {stats.get('elapsed_seconds', 0):.0f}s")
    print(f"  Rate:              {stats.get('pdfs_per_minute', 0):.1f} PDFs/min")

    if stats["failed"] > 0:
        print(f"\n  Error log: {errors_path}")


if __name__ == "__main__":
    main()
