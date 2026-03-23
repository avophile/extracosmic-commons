"""Tests for scholarly workbench importer."""

import json

import pytest

from extracosmic_commons.ingest.workbench import WorkbenchImporter

SAMPLE_METADATA = {
    "abc123hash": {
        "title": "Science of Logic",
        "authors": ["Hegel", "di Giovanni"],
        "year": 2010,
        "publisher": "Cambridge UP",
        "pages": 841,
        "chicago_citation": "Hegel. Science of Logic. Cambridge, 2010.",
        "status": "analyzed",
    },
    "def456hash": {
        "title": "Critique of Pure Reason",
        "authors": ["Kant"],
        "year": 1998,
        "publisher": "Cambridge UP",
        "pages": 841,
        "chicago_citation": "Kant. Critique of Pure Reason. Cambridge, 1998.",
    },
}


@pytest.fixture
def workbench_dir(tmp_path):
    """Create a minimal scholarly workbench directory structure."""
    user_dir = tmp_path / "backend" / "user_files" / "default_user"
    uploads_dir = user_dir / "uploads"
    uploads_dir.mkdir(parents=True)

    # Write metadata
    (user_dir / "metadata.json").write_text(json.dumps(SAMPLE_METADATA))

    # Create fake PDFs
    (uploads_dir / "abc123hash.pdf").write_bytes(b"%PDF-1.4 fake")
    (uploads_dir / "def456hash.pdf").write_bytes(b"%PDF-1.4 fake")

    return tmp_path


class TestWorkbenchImporter:
    def test_scan_returns_items(self, workbench_dir):
        importer = WorkbenchImporter()
        items = importer.scan_workbench(workbench_dir)
        assert len(items) == 2

    def test_scan_extracts_metadata(self, workbench_dir):
        importer = WorkbenchImporter()
        items = importer.scan_workbench(workbench_dir)

        titles = {it["title"] for it in items}
        assert "Science of Logic" in titles
        assert "Critique of Pure Reason" in titles

    def test_scan_resolves_pdf_paths(self, workbench_dir):
        importer = WorkbenchImporter()
        items = importer.scan_workbench(workbench_dir)

        for item in items:
            assert item["pdf_path"].exists()

    def test_scan_missing_metadata(self, tmp_path):
        importer = WorkbenchImporter()
        items = importer.scan_workbench(tmp_path)
        assert items == []
