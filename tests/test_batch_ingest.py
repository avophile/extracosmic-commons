"""Tests for scripts/batch_ingest.py — RunPod batch PDF ingestion.

TDD tests for the batch ingestion pipeline that reads a dedup manifest,
processes PDFs through the existing PDFIngester, tracks progress via
checkpoints, and handles errors gracefully.

Uses mock PDFIngester/EmbeddingPipeline/FAISSIndex to avoid requiring
actual GPU or model downloads during testing.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from batch_ingest import (
    load_manifest,
    filter_by_tier,
    load_checkpoint,
    save_checkpoint,
    BatchIngestionRunner,
)


@pytest.fixture
def manifest_file(tmp_path):
    """Create a temporary manifest file with 5 entries across 3 tiers."""
    # Create fake PDFs
    pdfs_dir = tmp_path / "pdfs"
    pdfs_dir.mkdir()

    manifest = {}
    entries = [
        ("aaa111", "Hegel.SoL.pdf", "hegel_collection", 1, 1000),
        ("bbb222", "Houlgate.Opening.pdf", "hegel_collection", 1, 2000),
        ("ccc333", "Pippin.Persistence.pdf", "hegel_texts", 2, 3000),
        ("ddd444", "Smith.Slavery.pdf", "hegel_texts", 2, 1500),
        ("eee555", "Munoz.Disidentifications.pdf", "zotero", 3, 2500),
    ]

    for hash_val, name, coll, tier, size in entries:
        pdf_path = pdfs_dir / name
        # Write minimal PDF-like content (not real PDF, but enough for testing)
        pdf_path.write_bytes(b"%PDF-1.4 fake content for " + name.encode())
        manifest[hash_val] = {
            "hash": hash_val,
            "path": str(pdf_path),
            "collection": coll,
            "tier": tier,
            "size_bytes": size,
        }

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path, manifest


@pytest.fixture
def checkpoint_path(tmp_path):
    """Path for checkpoint file."""
    return tmp_path / "checkpoint.json"


@pytest.fixture
def errors_path(tmp_path):
    """Path for errors log file."""
    return tmp_path / "errors.json"


class TestLoadManifest:
    """Tests for loading and validating the dedup manifest."""

    def test_loads_valid_manifest(self, manifest_file):
        """load_manifest returns a dict from a valid JSON file."""
        path, expected = manifest_file
        result = load_manifest(path)
        assert len(result) == 5

    def test_manifest_entries_have_required_keys(self, manifest_file):
        """Each manifest entry has path, collection, tier, size_bytes, hash."""
        path, _ = manifest_file
        result = load_manifest(path)
        required = {"path", "collection", "tier", "size_bytes", "hash"}
        for entry in result.values():
            assert required.issubset(entry.keys())

    def test_missing_manifest_raises(self, tmp_path):
        """load_manifest raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_manifest(tmp_path / "nonexistent.json")


class TestFilterByTier:
    """Tests for tier-based filtering of manifest entries."""

    def test_filter_tier1(self, manifest_file):
        """filter_by_tier(1) returns only tier 1 entries."""
        _, manifest = manifest_file
        tier1 = filter_by_tier(manifest, tier=1)
        assert len(tier1) == 2
        for entry in tier1.values():
            assert entry["tier"] == 1

    def test_filter_tier2(self, manifest_file):
        """filter_by_tier(2) returns only tier 2 entries."""
        _, manifest = manifest_file
        tier2 = filter_by_tier(manifest, tier=2)
        assert len(tier2) == 2

    def test_filter_tier3(self, manifest_file):
        """filter_by_tier(3) returns only tier 3 entries."""
        _, manifest = manifest_file
        tier3 = filter_by_tier(manifest, tier=3)
        assert len(tier3) == 1

    def test_filter_none_returns_all(self, manifest_file):
        """filter_by_tier(None) returns all entries."""
        _, manifest = manifest_file
        all_entries = filter_by_tier(manifest, tier=None)
        assert len(all_entries) == 5


class TestCheckpoints:
    """Tests for checkpoint save/load for resumable ingestion."""

    def test_save_and_load_checkpoint(self, checkpoint_path):
        """Checkpoint round-trips through JSON correctly."""
        data = {"completed_hashes": ["aaa111", "bbb222"], "failed_hashes": ["ccc333"]}
        save_checkpoint(checkpoint_path, data)
        loaded = load_checkpoint(checkpoint_path)
        assert loaded == data

    def test_load_missing_checkpoint_returns_empty(self, checkpoint_path):
        """Loading a nonexistent checkpoint returns a default empty dict."""
        result = load_checkpoint(checkpoint_path)
        assert result == {"completed_hashes": [], "failed_hashes": []}

    def test_checkpoint_preserves_completed_list(self, checkpoint_path):
        """Completed hashes list grows as items are added."""
        data = {"completed_hashes": ["aaa111"], "failed_hashes": []}
        save_checkpoint(checkpoint_path, data)
        loaded = load_checkpoint(checkpoint_path)
        assert "aaa111" in loaded["completed_hashes"]


class TestBatchIngestionRunner:
    """Tests for the main batch ingestion runner.

    Uses mocked PDFIngester, Database, EmbeddingPipeline, and FAISSIndex
    to test the orchestration logic without actual PDF processing or GPU.
    """

    @pytest.fixture
    def runner(self, manifest_file, checkpoint_path, errors_path):
        """Create a BatchIngestionRunner with mocked components."""
        manifest_path, manifest = manifest_file

        # Mock the heavy dependencies
        mock_db = MagicMock()
        mock_db.source_exists.return_value = False  # no dedup hits
        mock_embedder = MagicMock()
        mock_index = MagicMock()

        runner = BatchIngestionRunner(
            manifest=manifest,
            db=mock_db,
            embedder=mock_embedder,
            index=mock_index,
            checkpoint_path=checkpoint_path,
            errors_path=errors_path,
            checkpoint_every=2,
        )
        return runner

    def test_runner_processes_all_entries(self, runner):
        """Runner attempts to process every manifest entry."""
        with patch("batch_ingest.PDFIngester") as MockIngester:
            mock_instance = MockIngester.return_value
            mock_source = MagicMock()
            mock_source.id = "src-1"
            mock_instance.ingest.return_value = mock_source

            stats = runner.run()
            assert stats["total"] == 5
            assert stats["ingested"] + stats["skipped_checkpoint"] + stats["skipped_db"] + stats["failed"] == 5

    def test_runner_skips_already_completed(self, runner, checkpoint_path):
        """Runner skips hashes listed in the checkpoint file."""
        # Pre-populate checkpoint with 2 completed hashes
        save_checkpoint(checkpoint_path, {
            "completed_hashes": ["aaa111", "bbb222"],
            "failed_hashes": [],
        })
        runner.checkpoint = load_checkpoint(checkpoint_path)

        with patch("batch_ingest.PDFIngester") as MockIngester:
            mock_instance = MockIngester.return_value
            mock_source = MagicMock()
            mock_source.id = "src-1"
            mock_instance.ingest.return_value = mock_source

            stats = runner.run()
            # 2 were in checkpoint, so only 3 should be attempted
            assert stats["skipped_checkpoint"] == 2

    def test_runner_skips_db_duplicates(self, runner):
        """Runner skips files that already exist in the database."""
        runner.db.source_exists.return_value = True  # everything is a "duplicate"

        with patch("batch_ingest.PDFIngester") as MockIngester:
            stats = runner.run()
            assert stats["skipped_db"] == 5
            # PDFIngester.ingest should never be called
            MockIngester.return_value.ingest.assert_not_called()

    def test_runner_handles_ingestion_errors(self, runner):
        """Runner logs errors and continues when a PDF fails to ingest."""
        with patch("batch_ingest.PDFIngester") as MockIngester:
            mock_instance = MockIngester.return_value
            mock_instance.ingest.side_effect = Exception("corrupt PDF")

            stats = runner.run()
            assert stats["failed"] == 5  # all fail
            assert stats["ingested"] == 0

    def test_runner_writes_checkpoint(self, runner, checkpoint_path):
        """Runner saves checkpoints at the configured interval."""
        with patch("batch_ingest.PDFIngester") as MockIngester:
            mock_instance = MockIngester.return_value
            mock_source = MagicMock()
            mock_source.id = "src-1"
            mock_instance.ingest.return_value = mock_source

            runner.run()
            # With checkpoint_every=2 and 5 files, checkpoint should exist
            assert checkpoint_path.exists()
            cp = load_checkpoint(checkpoint_path)
            assert len(cp["completed_hashes"]) == 5

    def test_runner_writes_errors_log(self, runner, errors_path):
        """Runner writes failed PDFs to the errors log file."""
        with patch("batch_ingest.PDFIngester") as MockIngester:
            mock_instance = MockIngester.return_value
            mock_instance.ingest.side_effect = Exception("bad pdf")

            runner.run()
            assert errors_path.exists()
            errors = json.loads(errors_path.read_text())
            assert len(errors) == 5
            assert "error" in errors[0]

    def test_runner_filters_by_tier(self, manifest_file, checkpoint_path, errors_path):
        """Runner can be configured to process only a specific tier."""
        manifest_path, manifest = manifest_file
        tier1_only = filter_by_tier(manifest, tier=1)

        mock_db = MagicMock()
        mock_db.source_exists.return_value = False

        runner = BatchIngestionRunner(
            manifest=tier1_only,
            db=mock_db,
            embedder=MagicMock(),
            index=MagicMock(),
            checkpoint_path=checkpoint_path,
            errors_path=errors_path,
        )

        with patch("batch_ingest.PDFIngester") as MockIngester:
            mock_instance = MockIngester.return_value
            mock_source = MagicMock()
            mock_source.id = "src-1"
            mock_instance.ingest.return_value = mock_source

            stats = runner.run()
            assert stats["total"] == 2  # only tier 1

    def test_runner_progress_callback(self, runner):
        """Runner calls progress_callback with correct arguments."""
        progress_calls = []

        def track_progress(done, total, title):
            progress_calls.append((done, total, title))

        runner.progress_callback = track_progress

        with patch("batch_ingest.PDFIngester") as MockIngester:
            mock_instance = MockIngester.return_value
            mock_source = MagicMock()
            mock_source.id = "src-1"
            mock_instance.ingest.return_value = mock_source

            runner.run()
            assert len(progress_calls) > 0
            # Last call should have done == total
            last = progress_calls[-1]
            assert last[0] == last[1]
