#!/usr/bin/env python3
"""Deduplication scanner for PDF collections.

Scans multiple PDF collections, computes SHA-256 hashes to find exact
duplicates, and builds a dedup manifest that assigns each unique PDF
to a prioritization tier. The manifest is consumed by batch_ingest.py
and package_for_runpod.py to avoid re-ingesting duplicates.

Three-layer dedup strategy:
1. SHA-256 file hash — catches exact byte-identical copies
2. Collection-tier preference — when duplicates span tiers, the
   higher-priority (lower-numbered) tier wins
3. Database check — batch_ingest.py skips files already in SQLite
   (this script handles layers 1 and 2 only)

Usage:
    python3 scripts/dedup_scan.py \\
        --collections \\
            "hegel_collection:/path/to/The Hegel Collection" \\
            "hegel_texts:/path/to/HegelTranscripts/Hegel texts" \\
            "workbench:/path/to/scholarly-workbench-integrated" \\
            "zotero:/path/to/___Zotero" \\
            "zotero_rag:/path/to/zotero-rag/Random RAG docs" \\
        --output data/dedup_manifest.json

    # Dry run — print stats without writing manifest
    python3 scripts/dedup_scan.py --collections ... --dry-run
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier assignments: lower number = higher priority.
# When a PDF exists in multiple collections, the collection with the
# lowest tier number wins the manifest entry.
#
# Tier 1: Hegel primary & secondary texts (best for citation cross-ref)
# Tier 2: Scholarly commentary & workbench imports
# Tier 3: Broader Zotero collections (most overlap with tiers 1-2)
# ---------------------------------------------------------------------------
COLLECTION_TIERS: dict[str, int] = {
    "hegel_collection": 1,
    "hegel_texts": 2,
    "workbench": 2,
    "zotero": 3,
    "zotero_rag": 3,
    "pdf_import": 2,
}


def hash_file(path: Path | str) -> str:
    """Compute SHA-256 hash of a file's contents.

    Reads the file in 64KB chunks to handle large PDFs without loading
    the entire file into memory.

    Args:
        path: Filesystem path to the file.

    Returns:
        64-character lowercase hex digest string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)  # 64KB chunks
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def scan_collections(collections: dict[str, str]) -> list[dict]:
    """Scan all collections and return a list of file entries.

    For each collection, recursively finds all .pdf files and records
    their path, hash, size, and collection name.

    Args:
        collections: Mapping of collection_name → base directory path.
            For the workbench collection, PDFs are found under
            backend/user_files/default_user/uploads/.

    Returns:
        List of dicts, each with keys: path, hash, size_bytes, collection.
    """
    results = []

    for name, base_path in collections.items():
        base = Path(base_path)
        if not base.exists():
            logger.warning(f"Collection path does not exist: {base}")
            continue

        # Workbench stores PDFs in a nested uploads/ directory
        if name == "workbench":
            search_path = base / "backend" / "user_files" / "default_user" / "uploads"
            if not search_path.exists():
                logger.warning(f"Workbench uploads dir not found: {search_path}")
                continue
        else:
            search_path = base

        # Find all PDFs recursively
        pdfs = sorted(search_path.rglob("*.pdf"))
        logger.info(f"  {name}: found {len(pdfs)} PDFs in {search_path}")

        for pdf_path in pdfs:
            try:
                file_hash = hash_file(pdf_path)
                size = pdf_path.stat().st_size
                results.append({
                    "path": str(pdf_path),
                    "hash": file_hash,
                    "size_bytes": size,
                    "collection": name,
                })
            except (OSError, PermissionError) as e:
                logger.warning(f"  Could not hash {pdf_path.name}: {e}")

    return results


def find_duplicates(all_files: list[dict]) -> dict[str, list[dict]]:
    """Group files by hash and return only groups with 2+ entries.

    Identifies exact duplicates (byte-identical files) across collections.

    Args:
        all_files: Output from scan_collections().

    Returns:
        Dict mapping hash → list of file entries for hashes that appear
        more than once.
    """
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for entry in all_files:
        by_hash[entry["hash"]].append(entry)

    return {h: entries for h, entries in by_hash.items() if len(entries) >= 2}


def assign_tiers(entry: dict) -> int:
    """Determine the prioritization tier for a file entry.

    Uses COLLECTION_TIERS mapping; defaults to tier 3 for unknown
    collections.

    Args:
        entry: A file entry dict with a 'collection' key.

    Returns:
        Integer tier (1, 2, or 3).
    """
    return COLLECTION_TIERS.get(entry["collection"], 3)


def build_manifest(all_files: list[dict]) -> dict[str, dict]:
    """Build the dedup manifest: one entry per unique file hash.

    When duplicates exist across collections, the entry from the
    highest-priority tier (lowest number) wins. If two entries share
    the same tier, the first one found (alphabetically by path) wins.

    Args:
        all_files: Output from scan_collections().

    Returns:
        Dict mapping hash → manifest entry with keys:
        path, collection, tier, size_bytes, hash.
    """
    # Group all files by hash
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for entry in all_files:
        by_hash[entry["hash"]].append(entry)

    manifest: dict[str, dict] = {}

    for file_hash, entries in by_hash.items():
        # Sort entries by tier (ascending) then by path (alphabetical)
        # so the highest-priority collection wins
        sorted_entries = sorted(
            entries,
            key=lambda e: (assign_tiers(e), e["path"]),
        )
        winner = sorted_entries[0]

        manifest[file_hash] = {
            "hash": file_hash,
            "path": winner["path"],
            "collection": winner["collection"],
            "tier": assign_tiers(winner),
            "size_bytes": winner["size_bytes"],
        }

    return manifest


def print_report(all_files: list[dict], manifest: dict[str, dict],
                 duplicates: dict[str, list[dict]]) -> None:
    """Print a human-readable dedup report to stdout.

    Shows per-collection counts, duplicate details, tier breakdown,
    and total savings from deduplication.
    """
    # Per-collection counts
    from collections import Counter
    coll_counts = Counter(e["collection"] for e in all_files)
    coll_sizes = defaultdict(int)
    for e in all_files:
        coll_sizes[e["collection"]] += e["size_bytes"]

    print("\n=== PDF Collection Scan ===\n")
    for name in sorted(coll_counts):
        size_mb = coll_sizes[name] / (1024 * 1024)
        print(f"  {name}: {coll_counts[name]} PDFs ({size_mb:.1f} MB)")
    print(f"\n  Total files scanned: {len(all_files)}")

    # Duplicates
    if duplicates:
        print(f"\n=== Duplicates Found: {len(duplicates)} groups ===\n")
        for file_hash, entries in duplicates.items():
            names = [Path(e["path"]).name for e in entries]
            colls = [e["collection"] for e in entries]
            print(f"  Hash: {file_hash[:12]}...")
            for name, coll in zip(names, colls):
                print(f"    [{coll}] {name}")
    else:
        print("\n  No duplicates found.")

    # Manifest summary
    unique_count = len(manifest)
    dup_count = len(all_files) - unique_count
    unique_size = sum(e["size_bytes"] for e in manifest.values())
    total_size = sum(e["size_bytes"] for e in all_files)
    saved = total_size - unique_size

    print(f"\n=== Dedup Manifest ===\n")
    print(f"  Unique files: {unique_count}")
    print(f"  Duplicates removed: {dup_count}")
    print(f"  Total size (all files): {total_size / (1024*1024):.1f} MB")
    print(f"  Unique size (manifest): {unique_size / (1024*1024):.1f} MB")
    print(f"  Space saved: {saved / (1024*1024):.1f} MB")

    # Tier breakdown
    tier_counts = defaultdict(int)
    for e in manifest.values():
        tier_counts[e["tier"]] += 1
    print(f"\n  By tier:")
    for tier in sorted(tier_counts):
        print(f"    Tier {tier}: {tier_counts[tier]} files")
    print()


def main():
    """CLI entry point: scan collections and write manifest."""
    parser = argparse.ArgumentParser(
        description="Scan PDF collections and build a dedup manifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--collections", nargs="+", required=True,
        help="Collection specs as 'name:path' (e.g., 'hegel_collection:/path/to/dir')",
    )
    parser.add_argument(
        "--output", "-o", default="data/dedup_manifest.json",
        help="Output path for the JSON manifest (default: data/dedup_manifest.json)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print report only, don't write manifest file",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Parse collection specs
    collections = {}
    for spec in args.collections:
        if ":" not in spec:
            print(f"Error: collection spec must be 'name:path', got: {spec}",
                  file=sys.stderr)
            sys.exit(1)
        name, path = spec.split(":", 1)
        collections[name] = path

    # Scan
    logger.info("Scanning collections...")
    all_files = scan_collections(collections)

    if not all_files:
        print("No PDF files found in any collection.")
        sys.exit(0)

    # Dedup
    duplicates = find_duplicates(all_files)
    manifest = build_manifest(all_files)

    # Report
    print_report(all_files, manifest, duplicates)

    # Write manifest
    if not args.dry_run:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(manifest, indent=2))
        print(f"Manifest written to {out_path} ({len(manifest)} entries)")


if __name__ == "__main__":
    main()
