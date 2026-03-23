"""Scholarly Workbench metadata.json importer.

Imports documents already processed in the scholarly-workbench-integrated
project. The workbench stores PDFs as hash-named files with rich metadata
in a JSON sidecar file.

Structure:
    scholarly-workbench-integrated/
    └── backend/user_files/default_user/
        ├── metadata.json          # Hash → {title, authors, year, doi, ...}
        └── uploads/
            ├── {sha256_hash_1}.pdf
            ├── {sha256_hash_2}.pdf
            └── ...
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

from ..database import Database
from ..embeddings import EmbeddingPipeline
from ..index import FAISSIndex
from ..models import Source
from .pdf import PDFIngester

logger = logging.getLogger(__name__)


class WorkbenchImporter:
    """Imports documents from the scholarly workbench."""

    def __init__(self):
        self._pdf_ingester = PDFIngester()

    def scan_workbench(self, workbench_path: Path) -> list[dict]:
        """Read metadata.json and return list of importable documents."""
        meta_path = workbench_path / "backend" / "user_files" / "default_user" / "metadata.json"
        if not meta_path.exists():
            logger.warning(f"metadata.json not found at {meta_path}")
            return []

        data = json.loads(meta_path.read_text(encoding="utf-8"))
        uploads_dir = meta_path.parent / "uploads"

        items = []
        for file_hash, meta in data.items():
            pdf_path = uploads_dir / f"{file_hash}.pdf"
            if not pdf_path.exists():
                continue

            items.append({
                "pdf_path": pdf_path,
                "title": meta.get("title", file_hash),
                "authors": meta.get("authors", []),
                "year": meta.get("year"),
                "publisher": meta.get("publisher"),
                "doi": meta.get("abstract"),  # workbench stores DOI in abstract field sometimes
                "isbn": None,
                "chicago_citation": meta.get("chicago_citation"),
                "pages": meta.get("pages"),
            })

        return items

    def import_all(
        self,
        workbench_path: Path,
        db: Database,
        embedder: EmbeddingPipeline,
        index: FAISSIndex,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[Source]:
        """Import all documents from the scholarly workbench."""
        items = self.scan_workbench(workbench_path)
        sources = []
        skipped = 0
        failed = []
        total = len(items)

        for i, item in enumerate(items):
            if progress_callback:
                progress_callback(i, total, item["title"][:60])

            # Dedup
            if db.source_exists(item["title"], item["authors"] or None):
                skipped += 1
                continue

            metadata = {}
            for key in ("year", "publisher", "doi", "isbn", "chicago_citation", "pages"):
                if item.get(key):
                    metadata[key] = item[key]

            try:
                source = self._pdf_ingester.ingest(
                    path=item["pdf_path"],
                    db=db,
                    embedder=embedder,
                    index=index,
                    title=item["title"],
                    author=item["authors"],
                    metadata=metadata,
                )
                sources.append(source)
            except Exception as e:
                logger.warning(f"Failed to ingest {item['title']}: {e}")
                failed.append((item["title"], str(e)))

        if progress_callback:
            progress_callback(total, total, "Done")

        logger.info(
            f"Workbench import: {len(sources)} imported, {skipped} skipped (dedup), "
            f"{len(failed)} failed out of {total} items"
        )

        return sources
