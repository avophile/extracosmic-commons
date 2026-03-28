"""Tests for scripts/package_for_runpod.py — RunPod transfer packager.

TDD tests for the script that reads a dedup manifest, copies unique PDFs
into a flat directory structure, and creates a tar.gz archive for
efficient transfer to RunPod via runpodctl.
"""

from __future__ import annotations

import json
import tarfile
import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from package_for_runpod import (
    collect_files,
    resolve_workbench_title,
    create_package,
)


@pytest.fixture
def manifest_with_files(tmp_path):
    """Create a manifest and corresponding fake PDF files.

    Simulates 4 unique PDFs across 3 collections, including one
    workbench hash-named file with a metadata.json sidecar.
    """
    # Create source PDFs
    coll_dir = tmp_path / "sources"
    hegel = coll_dir / "hegel_collection"
    texts = coll_dir / "hegel_texts"
    wb_uploads = coll_dir / "workbench" / "backend" / "user_files" / "default_user" / "uploads"

    for d in [hegel, texts, wb_uploads]:
        d.mkdir(parents=True)

    (hegel / "Hegel.SoL.DiGiovanni.pdf").write_bytes(b"%PDF digiovanni")
    (texts / "Pippin.Persistence.pdf").write_bytes(b"%PDF pippin")
    (texts / "Smith.Slavery.pdf").write_bytes(b"%PDF smith")
    (wb_uploads / "abc123def456.pdf").write_bytes(b"%PDF workbench doc")

    # Workbench metadata
    meta = {
        "abc123def456": {
            "title": "The Persistence of Subjectivity",
            "authors": ["Robert Pippin"],
            "year": 2005,
        }
    }
    (wb_uploads.parent / "metadata.json").write_text(json.dumps(meta))

    manifest = {
        "hash1": {
            "hash": "hash1",
            "path": str(hegel / "Hegel.SoL.DiGiovanni.pdf"),
            "collection": "hegel_collection",
            "tier": 1,
            "size_bytes": 15,
        },
        "hash2": {
            "hash": "hash2",
            "path": str(texts / "Pippin.Persistence.pdf"),
            "collection": "hegel_texts",
            "tier": 2,
            "size_bytes": 12,
        },
        "hash3": {
            "hash": "hash3",
            "path": str(texts / "Smith.Slavery.pdf"),
            "collection": "hegel_texts",
            "tier": 2,
            "size_bytes": 11,
        },
        "hash4": {
            "hash": "hash4",
            "path": str(wb_uploads / "abc123def456.pdf"),
            "collection": "workbench",
            "tier": 2,
            "size_bytes": 18,
        },
    }

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return manifest_path, manifest, coll_dir


class TestCollectFiles:
    """Tests for collecting and organizing files from the manifest."""

    def test_collects_all_files(self, manifest_with_files):
        """collect_files returns entries for every manifest item."""
        _, manifest, _ = manifest_with_files
        files = collect_files(manifest)
        assert len(files) == 4

    def test_entries_have_source_and_dest(self, manifest_with_files):
        """Each collected file has a source path and destination filename."""
        _, manifest, _ = manifest_with_files
        files = collect_files(manifest)
        for f in files:
            assert "src" in f
            assert "dest" in f
            assert Path(f["src"]).exists()

    def test_destination_filenames_are_unique(self, manifest_with_files):
        """All destination filenames are unique (no collisions)."""
        _, manifest, _ = manifest_with_files
        files = collect_files(manifest)
        dests = [f["dest"] for f in files]
        assert len(dests) == len(set(dests))

    def test_destination_preserves_pdf_extension(self, manifest_with_files):
        """All destination filenames end with .pdf."""
        _, manifest, _ = manifest_with_files
        files = collect_files(manifest)
        for f in files:
            assert f["dest"].endswith(".pdf")

    def test_skips_missing_source_files(self, manifest_with_files):
        """Files that don't exist on disk are skipped with a warning."""
        _, manifest, _ = manifest_with_files
        # Add a phantom entry
        manifest["phantom"] = {
            "hash": "phantom",
            "path": "/nonexistent/phantom.pdf",
            "collection": "zotero",
            "tier": 3,
            "size_bytes": 0,
        }
        files = collect_files(manifest)
        # Should still have 4 files (the phantom is skipped)
        assert len(files) == 4


class TestResolveWorkbenchTitle:
    """Tests for resolving human-readable titles for hash-named workbench files."""

    def test_resolves_title_from_metadata(self, manifest_with_files):
        """Workbench hash file gets its title from metadata.json."""
        _, manifest, _ = manifest_with_files
        wb_entry = manifest["hash4"]
        title = resolve_workbench_title(wb_entry["path"])
        assert title == "The Persistence of Subjectivity"

    def test_returns_hash_if_no_metadata(self, tmp_path):
        """Returns the hash stem if metadata.json is missing."""
        fake_pdf = tmp_path / "deadbeef1234.pdf"
        fake_pdf.write_bytes(b"pdf content")
        title = resolve_workbench_title(str(fake_pdf))
        assert "deadbeef1234" in title

    def test_returns_filename_for_named_pdfs(self, tmp_path):
        """Non-hash-named files return their stem as the title."""
        f = tmp_path / "Hegel.SoL.pdf"
        f.write_bytes(b"pdf")
        title = resolve_workbench_title(str(f))
        assert title == "Hegel.SoL"


class TestCreatePackage:
    """Tests for creating the tar.gz archive."""

    def test_creates_tar_gz(self, manifest_with_files, tmp_path):
        """create_package produces a .tar.gz file."""
        _, manifest, _ = manifest_with_files
        output = tmp_path / "output" / "package.tar.gz"
        create_package(manifest, output)
        assert output.exists()
        assert output.suffix == ".gz"

    def test_tar_contains_all_pdfs(self, manifest_with_files, tmp_path):
        """The tar archive contains one entry per manifest item."""
        _, manifest, _ = manifest_with_files
        output = tmp_path / "output" / "package.tar.gz"
        create_package(manifest, output)

        with tarfile.open(output, "r:gz") as tar:
            members = tar.getnames()
            pdf_members = [m for m in members if m.endswith(".pdf")]
            assert len(pdf_members) == 4

    def test_tar_contains_manifest_copy(self, manifest_with_files, tmp_path):
        """The tar archive includes a copy of the manifest for reference."""
        _, manifest, _ = manifest_with_files
        output = tmp_path / "output" / "package.tar.gz"
        create_package(manifest, output)

        with tarfile.open(output, "r:gz") as tar:
            members = tar.getnames()
            assert any("manifest" in m for m in members)

    def test_tar_pdfs_are_extractable(self, manifest_with_files, tmp_path):
        """PDFs in the tar can be extracted and have correct content."""
        _, manifest, _ = manifest_with_files
        output = tmp_path / "output" / "package.tar.gz"
        create_package(manifest, output)

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        with tarfile.open(output, "r:gz") as tar:
            tar.extractall(extract_dir, filter="data")

        # Should find PDF files in the extracted directory
        extracted_pdfs = list(extract_dir.rglob("*.pdf"))
        assert len(extracted_pdfs) == 4
        for pdf in extracted_pdfs:
            assert pdf.stat().st_size > 0

    def test_updates_manifest_paths(self, manifest_with_files, tmp_path):
        """The manifest inside the tar has paths updated to the package layout."""
        _, manifest, _ = manifest_with_files
        output = tmp_path / "output" / "package.tar.gz"
        create_package(manifest, output)

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        with tarfile.open(output, "r:gz") as tar:
            tar.extractall(extract_dir, filter="data")

        manifest_files = list(extract_dir.rglob("manifest.json"))
        assert len(manifest_files) == 1
        packed_manifest = json.loads(manifest_files[0].read_text())
        for entry in packed_manifest.values():
            # Paths should be relative to the package, not absolute
            assert not entry["path"].startswith("/")
