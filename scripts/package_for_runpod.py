#!/usr/bin/env python3
"""Package unique PDFs from a dedup manifest for RunPod transfer.

Reads the manifest produced by dedup_scan.py, copies unique PDFs into a
flat directory structure with human-readable filenames, and creates a
tar.gz archive. The archive includes the manifest (with updated paths)
so batch_ingest.py on RunPod can find everything it needs.

Workbench hash-named files are renamed to their human-readable titles
using the metadata.json sidecar file.

Usage:
    python3 scripts/package_for_runpod.py \\
        --manifest data/dedup_manifest.json \\
        --output /tmp/runpod_ingestion_package.tar.gz
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_workbench_title(path: str) -> str:
    """Resolve a human-readable title for a PDF file.

    For workbench hash-named files (64-char hex stems), looks up the
    title in the adjacent metadata.json. For regular files, returns
    the filename stem.

    Args:
        path: Filesystem path to the PDF.

    Returns:
        Human-readable title string.
    """
    p = Path(path)
    stem = p.stem

    # Check if this is a hash-named workbench file
    # (64 hex chars, which is SHA-256)
    if len(stem) >= 12 and re.match(r'^[0-9a-f]+$', stem):
        # Look for metadata.json in the parent directory tree
        # Workbench layout: .../uploads/{hash}.pdf with metadata.json
        # in the same parent or grandparent directory
        for meta_dir in [p.parent, p.parent.parent]:
            meta_path = meta_dir / "metadata.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    if stem in meta and "title" in meta[stem]:
                        return meta[stem]["title"]
                except (json.JSONDecodeError, KeyError):
                    pass
        # Fallback: return the hash
        return stem

    return stem


def collect_files(manifest: dict[str, dict]) -> list[dict]:
    """Collect file entries from the manifest, resolving destinations.

    For each manifest entry, determines the source path and a unique
    destination filename for the package. Skips entries whose source
    files don't exist on disk.

    Args:
        manifest: Dedup manifest dict (hash → entry).

    Returns:
        List of dicts with keys: src (absolute path), dest (filename),
        hash, collection, tier.
    """
    files = []
    used_names: set[str] = set()

    for file_hash, entry in manifest.items():
        src_path = Path(entry["path"])
        if not src_path.exists():
            logger.warning(f"Source file missing, skipping: {src_path}")
            continue

        # Determine destination filename
        title = resolve_workbench_title(str(src_path))
        # Sanitize: replace problematic chars, keep it filesystem-safe
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', title)
        safe_name = safe_name.strip('. ')[:200]  # max 200 chars
        dest_name = f"{safe_name}.pdf"

        # Handle collisions by appending hash prefix
        if dest_name in used_names:
            dest_name = f"{safe_name}_{file_hash[:8]}.pdf"

        used_names.add(dest_name)

        files.append({
            "src": str(src_path),
            "dest": dest_name,
            "hash": file_hash,
            "collection": entry["collection"],
            "tier": entry["tier"],
        })

    return files


def create_package(
    manifest: dict[str, dict],
    output_path: Path | str,
) -> Path:
    """Create a tar.gz archive of unique PDFs with updated manifest.

    Copies all PDFs from the manifest into a flat directory inside a
    temporary staging area, writes an updated manifest with relative
    paths, and compresses everything into a tar.gz archive.

    Args:
        manifest: Dedup manifest dict (hash → entry).
        output_path: Where to write the .tar.gz file.

    Returns:
        Path to the created archive.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = collect_files(manifest)

    with tempfile.TemporaryDirectory() as staging_dir:
        staging = Path(staging_dir) / "ingestion_package"
        pdfs_dir = staging / "pdfs"
        pdfs_dir.mkdir(parents=True)

        # Copy PDFs to staging
        updated_manifest: dict[str, dict] = {}
        for f in files:
            dest_path = pdfs_dir / f["dest"]
            shutil.copy2(f["src"], dest_path)

            # Update manifest entry with relative path
            updated_manifest[f["hash"]] = {
                "hash": f["hash"],
                "path": f"pdfs/{f['dest']}",
                "collection": f["collection"],
                "tier": f["tier"],
                "size_bytes": dest_path.stat().st_size,
            }

        # Write updated manifest
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(updated_manifest, indent=2))

        # Create tar.gz
        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(staging, arcname="ingestion_package")

    logger.info(
        f"Package created: {output_path} "
        f"({len(files)} PDFs, {output_path.stat().st_size / (1024*1024):.1f} MB)"
    )
    return output_path


def main():
    """CLI entry point for creating the RunPod transfer package."""
    parser = argparse.ArgumentParser(
        description="Package unique PDFs from a dedup manifest for RunPod transfer.",
    )
    parser.add_argument("--manifest", required=True, help="Path to dedup_manifest.json")
    parser.add_argument("--output", "-o", required=True, help="Output .tar.gz path")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3],
                        help="Package only a specific tier")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Load manifest
    manifest = json.loads(Path(args.manifest).read_text())

    if args.tier:
        manifest = {h: e for h, e in manifest.items() if e["tier"] == args.tier}
        logger.info(f"Filtered to tier {args.tier}: {len(manifest)} entries")

    if not manifest:
        print("No entries to package.")
        return

    # Create package
    output = create_package(manifest, args.output)

    total_size = sum(e["size_bytes"] for e in manifest.values())
    archive_size = output.stat().st_size
    ratio = archive_size / total_size * 100 if total_size else 0

    print(f"\n=== RunPod Package Created ===")
    print(f"  PDFs: {len(manifest)}")
    print(f"  Original size: {total_size / (1024*1024):.1f} MB")
    print(f"  Archive size:  {archive_size / (1024*1024):.1f} MB ({ratio:.0f}%)")
    print(f"  Output: {output}")
    print(f"\n  Transfer with: runpodctl send {output}")


if __name__ == "__main__":
    main()
