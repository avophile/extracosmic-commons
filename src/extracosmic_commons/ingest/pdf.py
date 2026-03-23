"""PDF text extraction and page-based chunking.

Basic pypdf extraction for Phase 0. Docling structure-aware chunking
will replace this in Phase 5, but page-based chunking is sufficient
to get the corpus indexed and searchable now.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pypdf import PdfReader

from ..database import Database
from ..embeddings import EmbeddingPipeline
from ..index import FAISSIndex
from ..models import Chunk, Source, SourceType

logger = logging.getLogger(__name__)

# Pages with less text than this are likely blanks, images, or title pages
MIN_PAGE_TEXT_CHARS = 50

# Target chunk size — split pages longer than this at paragraph boundaries
MAX_CHUNK_CHARS = 1500


def extract_text(path: Path) -> list[tuple[int, str]]:
    """Extract text from a PDF, returning (page_number, text) pairs.

    Skips pages with less than MIN_PAGE_TEXT_CHARS characters.
    Page numbers are 1-indexed.
    """
    reader = PdfReader(str(path))
    pages = []

    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            logger.warning(f"Failed to extract page {i} from {path.name}: {e}")
            continue

        text = text.strip()
        if len(text) >= MIN_PAGE_TEXT_CHARS:
            pages.append((i, text))

    return pages


def _split_page_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split a page's text at paragraph boundaries if it exceeds max_chars."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split('\n\n')
    if len(paragraphs) <= 1:
        # No paragraph breaks — split at newlines
        paragraphs = text.split('\n')

    result = []
    current = ""

    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_chars:
            result.append(current.strip())
            current = para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        result.append(current.strip())

    return [r for r in result if len(r) >= MIN_PAGE_TEXT_CHARS]


class PDFIngester:
    """Ingests PDF files as page-based chunks."""

    def parse_pdf(
        self,
        path: Path,
        title: str | None = None,
        author: list[str] | None = None,
        language: list[str] | None = None,
        metadata: dict | None = None,
    ) -> tuple[Source, list[Chunk]]:
        """Extract text and create Source + Chunks from a PDF.

        Metadata can be supplied by the caller (from Zotero RDF, workbench
        metadata.json, or filename parsing). If not supplied, we use the
        filename as the title.
        """
        pages = extract_text(path)

        source = Source(
            title=title or path.stem,
            type=SourceType.PDF,
            author=author or [],
            language=language or ["en"],
            source_path=str(path),
            metadata=metadata or {},
        )

        chunks = []
        for page_num, page_text in pages:
            segments = _split_page_text(page_text)
            for seg_idx, seg_text in enumerate(segments):
                chunks.append(Chunk(
                    source_id=source.id,
                    text=seg_text,
                    language=(language or ["en"])[0],
                    pdf_page=page_num,
                    paragraph_index=len(chunks),
                    chunk_method="page",
                ))

        return source, chunks

    def ingest(
        self,
        path: Path,
        db: Database,
        embedder: EmbeddingPipeline,
        index: FAISSIndex,
        title: str | None = None,
        author: list[str] | None = None,
        language: list[str] | None = None,
        metadata: dict | None = None,
    ) -> Source:
        """Parse, embed, and store a PDF."""
        source, chunks = self.parse_pdf(
            path, title=title, author=author, language=language, metadata=metadata
        )

        if not chunks:
            logger.warning(f"No text extracted from {path.name}")
            # Still create the source record for tracking
            db.insert_source(source)
            return source

        texts = [c.text for c in chunks]
        embeddings = embedder.embed_batch(texts)

        db.insert_source(source)
        db.insert_chunks_batch(chunks)
        index.add_batch([c.id for c in chunks], embeddings)

        return source
