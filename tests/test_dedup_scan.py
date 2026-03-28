"""Tests for scripts/dedup_scan.py — PDF deduplication scanner.

TDD tests for the three-layer dedup strategy:
1. SHA-256 file hash for exact duplicates
2. Title similarity matching for near-duplicates
3. Manifest generation with tier assignments

Tests use temporary directories with small PDF-like files to simulate
the collection layout without requiring actual PDFs.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Import will work once the script is written
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from dedup_scan import (
    hash_file,
    scan_collections,
    build_manifest,
    assign_tiers,
    find_duplicates,
    COLLECTION_TIERS,
)


@pytest.fixture
def collection_dirs(tmp_path):
    """Create a temporary collection layout with some duplicate files.

    Layout:
        tmp_path/
        ├── hegel_collection/
        │   ├── Hegel.SoL.DiGiovanni.pdf      (unique)
        │   └── Houlgate.OpeningOfLogic.pdf    (duplicate of hegel_texts copy)
        ├── hegel_texts/
        │   ├── Houlgate.OpeningOfLogic.pdf    (duplicate of hegel_collection copy)
        │   ├── Pippin.Persistence.pdf         (unique)
        │   └── Smith.Slavery.pdf              (unique)
        ├── workbench/
        │   └── backend/user_files/default_user/
        │       ├── metadata.json
        │       └── uploads/
        │           └── abc123.pdf             (same content as Pippin)
        └── zotero/
            └── Munoz.Disidentifications.pdf   (unique)
    """
    # Create collection directories
    hegel_coll = tmp_path / "hegel_collection"
    hegel_texts = tmp_path / "hegel_texts"
    workbench = tmp_path / "workbench" / "backend" / "user_files" / "default_user" / "uploads"
    zotero = tmp_path / "zotero"

    for d in [hegel_coll, hegel_texts, workbench, zotero]:
        d.mkdir(parents=True)

    # Create PDF files with known content
    # Unique files
    (hegel_coll / "Hegel.SoL.DiGiovanni.pdf").write_bytes(b"digiovanni content here")
    (hegel_texts / "Smith.Slavery.pdf").write_bytes(b"smith slavery content")
    (zotero / "Munoz.Disidentifications.pdf").write_bytes(b"munoz content here")

    # Duplicate pair: same content in hegel_collection and hegel_texts
    dup_content = b"houlgate opening of logic content bytes"
    (hegel_coll / "Houlgate.OpeningOfLogic.pdf").write_bytes(dup_content)
    (hegel_texts / "Houlgate.OpeningOfLogic.pdf").write_bytes(dup_content)

    # Another duplicate: workbench hash-named file = same as Pippin
    pippin_content = b"pippin persistence of subjectivity"
    (hegel_texts / "Pippin.Persistence.pdf").write_bytes(pippin_content)
    (workbench / "abc123.pdf").write_bytes(pippin_content)

    # Workbench metadata.json
    meta = {
        "abc123": {
            "title": "The Persistence of Subjectivity",
            "authors": ["Robert Pippin"],
            "year": 2005,
        }
    }
    meta_path = workbench.parent / "metadata.json"
    meta_path.write_text(json.dumps(meta))

    return {
        "hegel_collection": str(hegel_coll),
        "hegel_texts": str(hegel_texts),
        "workbench": str(tmp_path / "workbench"),
        "zotero": str(zotero),
    }


class TestHashFile:
    """Tests for the SHA-256 hashing function."""

    def test_hash_returns_hex_string(self, tmp_path):
        """hash_file returns a 64-char hex digest."""
        f = tmp_path / "test.pdf"
        f.write_bytes(b"test content")
        result = hash_file(f)
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex length

    def test_identical_files_same_hash(self, tmp_path):
        """Two files with identical content produce the same hash."""
        content = b"identical content"
        f1 = tmp_path / "a.pdf"
        f2 = tmp_path / "b.pdf"
        f1.write_bytes(content)
        f2.write_bytes(content)
        assert hash_file(f1) == hash_file(f2)

    def test_different_files_different_hash(self, tmp_path):
        """Files with different content produce different hashes."""
        f1 = tmp_path / "a.pdf"
        f2 = tmp_path / "b.pdf"
        f1.write_bytes(b"content a")
        f2.write_bytes(b"content b")
        assert hash_file(f1) != hash_file(f2)

    def test_hash_matches_hashlib(self, tmp_path):
        """hash_file result matches direct hashlib.sha256 computation."""
        content = b"verify against hashlib"
        f = tmp_path / "check.pdf"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert hash_file(f) == expected


class TestScanCollections:
    """Tests for scanning multiple collection directories."""

    def test_finds_all_pdfs(self, collection_dirs):
        """scan_collections finds all PDF files across all collections."""
        results = scan_collections(collection_dirs)
        # 7 total PDF files (including duplicates)
        assert len(results) == 7

    def test_records_collection_name(self, collection_dirs):
        """Each entry records which collection it came from."""
        results = scan_collections(collection_dirs)
        collections_found = {r["collection"] for r in results}
        assert collections_found == {"hegel_collection", "hegel_texts", "workbench", "zotero"}

    def test_records_file_path(self, collection_dirs):
        """Each entry has a valid file path."""
        results = scan_collections(collection_dirs)
        for r in results:
            assert Path(r["path"]).exists()

    def test_records_file_hash(self, collection_dirs):
        """Each entry has a SHA-256 hash."""
        results = scan_collections(collection_dirs)
        for r in results:
            assert len(r["hash"]) == 64

    def test_records_file_size(self, collection_dirs):
        """Each entry records file size in bytes."""
        results = scan_collections(collection_dirs)
        for r in results:
            assert r["size_bytes"] > 0


class TestFindDuplicates:
    """Tests for identifying duplicate files across collections."""

    def test_finds_duplicate_pairs(self, collection_dirs):
        """find_duplicates identifies files with matching hashes."""
        all_files = scan_collections(collection_dirs)
        dupes = find_duplicates(all_files)
        # Should find 2 duplicate groups (Houlgate pair + Pippin/workbench pair)
        assert len(dupes) == 2

    def test_duplicate_groups_have_multiple_paths(self, collection_dirs):
        """Each duplicate group has 2+ file entries."""
        all_files = scan_collections(collection_dirs)
        dupes = find_duplicates(all_files)
        for hash_val, entries in dupes.items():
            assert len(entries) >= 2

    def test_unique_files_not_in_duplicates(self, collection_dirs):
        """Files that appear only once are not in the duplicates dict."""
        all_files = scan_collections(collection_dirs)
        dupes = find_duplicates(all_files)
        dup_paths = set()
        for entries in dupes.values():
            for e in entries:
                dup_paths.add(e["path"])
        # DiGiovanni, Smith, Munoz are unique — should not appear
        for path_str in dup_paths:
            name = Path(path_str).name
            assert name not in ("Hegel.SoL.DiGiovanni.pdf", "Smith.Slavery.pdf",
                                "Munoz.Disidentifications.pdf")


class TestAssignTiers:
    """Tests for tier assignment based on collection membership."""

    def test_hegel_collection_is_tier1(self, collection_dirs):
        """Files from The Hegel Collection get tier 1."""
        all_files = scan_collections(collection_dirs)
        manifest = build_manifest(all_files)
        hegel_coll_entries = [e for e in manifest.values()
                              if e["collection"] == "hegel_collection"]
        for entry in hegel_coll_entries:
            assert entry["tier"] == 1

    def test_zotero_is_tier3(self, collection_dirs):
        """Files from Zotero collections get tier 3."""
        all_files = scan_collections(collection_dirs)
        manifest = build_manifest(all_files)
        zotero_entries = [e for e in manifest.values()
                          if e["collection"] == "zotero"]
        for entry in zotero_entries:
            assert entry["tier"] == 3

    def test_duplicates_prefer_higher_tier(self, collection_dirs):
        """When a file exists in multiple collections, the higher-tier
        (lower number) collection wins the manifest entry."""
        all_files = scan_collections(collection_dirs)
        manifest = build_manifest(all_files)
        # Houlgate exists in hegel_collection (tier 1) and hegel_texts (tier 2)
        # The manifest should keep the tier 1 version
        houlgate_entries = [e for e in manifest.values()
                            if "Houlgate" in Path(e["path"]).name or "houlgate" in Path(e["path"]).name.lower()]
        # Should be exactly 1 entry after dedup
        assert len(houlgate_entries) == 1
        assert houlgate_entries[0]["tier"] == 1


class TestBuildManifest:
    """Tests for the complete manifest generation."""

    def test_manifest_has_unique_entries_only(self, collection_dirs):
        """Manifest contains exactly one entry per unique file hash."""
        all_files = scan_collections(collection_dirs)
        manifest = build_manifest(all_files)
        # 7 files - 2 duplicates = 5 unique
        assert len(manifest) == 5

    def test_manifest_keyed_by_hash(self, collection_dirs):
        """Manifest is a dict keyed by SHA-256 hash."""
        all_files = scan_collections(collection_dirs)
        manifest = build_manifest(all_files)
        for key in manifest:
            assert len(key) == 64  # SHA-256 hex

    def test_manifest_entry_has_required_fields(self, collection_dirs):
        """Each manifest entry has path, collection, tier, size_bytes."""
        all_files = scan_collections(collection_dirs)
        manifest = build_manifest(all_files)
        required = {"path", "collection", "tier", "size_bytes", "hash"}
        for entry in manifest.values():
            assert required.issubset(entry.keys())

    def test_manifest_json_serializable(self, collection_dirs):
        """Manifest can be serialized to JSON."""
        all_files = scan_collections(collection_dirs)
        manifest = build_manifest(all_files)
        json_str = json.dumps(manifest, indent=2)
        loaded = json.loads(json_str)
        assert len(loaded) == len(manifest)

    def test_total_size_calculation(self, collection_dirs):
        """Manifest entries' sizes sum to a reasonable total."""
        all_files = scan_collections(collection_dirs)
        manifest = build_manifest(all_files)
        total = sum(e["size_bytes"] for e in manifest.values())
        assert total > 0


class TestCollectionTiers:
    """Tests for the tier configuration constants."""

    def test_all_known_collections_have_tiers(self):
        """Every expected collection name maps to a tier."""
        expected = {"hegel_collection", "hegel_texts", "workbench", "zotero", "zotero_rag"}
        assert expected.issubset(set(COLLECTION_TIERS.keys()))

    def test_tiers_are_1_2_or_3(self):
        """All tier values are 1, 2, or 3."""
        for tier in COLLECTION_TIERS.values():
            assert tier in (1, 2, 3)
