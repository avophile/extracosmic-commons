"""Lecture transcript ingestion pipeline.

Parses markdown transcripts from the HegelTranscripts project. These
transcripts use a consistent format:

    **[HH:MM:SS](youtube_url)** Transcript text continues here...

Each timestamp marker becomes a chunk boundary. Header metadata (lecturer,
lecture count, playlist URL) is extracted from the markdown frontmatter.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..database import Database
from ..embeddings import EmbeddingPipeline
from ..index import FAISSIndex
from ..models import Chunk, Source, SourceType

# Matches both formats:
#   **[00:29:16](https://...)** — Thompson/Houlgate (with URL)
#   **[00:29:16]**             — Radnik (no URL)
TIMESTAMP_PATTERN = re.compile(
    r'\*\*\[(\d{1,2}:\d{2}:\d{2})\](?:\((https?://[^)]+)\))?\*\*\s*'
)

# Matches lecture/session headers:
#   ## Session 1a — Title        (Thompson)
#   ## Lecture 1 — Title          (Houlgate)
#   ## Being 5                    (Radnik)
#   ## Essence 3 — Shine (Part 2) (Radnik)
LECTURE_HEADER_PATTERN = re.compile(
    r'^##\s+(?:Session|Lecture|Being|Essence)\s+(\d+\w?)\s*(?:[—–-]\s*(.+))?',
    re.MULTILINE,
)


def _extract_header_metadata(text: str) -> dict:
    """Extract metadata from the markdown header block."""
    metadata = {}

    # Lecturer name
    m = re.search(r'\*\*Lecturer:\*\*\s*(.+)', text)
    if m:
        metadata["lecturer"] = m.group(1).strip()

    # Total lectures/sessions
    m = re.search(r'\*\*Total (?:Sessions|Lectures):\*\*\s*(\d+)', text)
    if m:
        metadata["lecture_count"] = int(m.group(1))

    # Source playlist URL
    m = re.search(r'\*\*Source:\*\*\s*\[.*?\]\((https?://[^)]+)\)', text)
    if m:
        metadata["playlist_url"] = m.group(1)

    # Scope
    m = re.search(r'\*\*Scope:\*\*\s*(.+)', text)
    if m:
        metadata["scope"] = m.group(1).strip()

    return metadata


def _split_long_text(text: str, max_chars: int, min_chars: int) -> list[str]:
    """Split a long text into segments of at most max_chars.

    Tries paragraph boundaries first, then sentence boundaries, then
    word boundaries as a last resort. Lecture transcripts often lack
    punctuation and paragraph breaks, so the word-boundary fallback
    is essential.
    """
    if len(text) <= max_chars:
        return [text] if len(text) >= min_chars else []

    segments = []
    # First try paragraph splits
    paragraphs = text.split('\n\n')

    for para in paragraphs:
        # If a single paragraph exceeds max_chars, split it further
        if len(para) > max_chars:
            subsegments = _split_at_word_boundaries(para, max_chars)
            segments.extend(s for s in subsegments if len(s) >= min_chars)
        else:
            # Try to merge small paragraphs
            if segments and len(segments[-1]) + len(para) + 2 <= max_chars:
                segments[-1] = segments[-1] + "\n\n" + para
            elif len(para) >= min_chars:
                segments.append(para)

    return segments


def _split_at_word_boundaries(text: str, max_chars: int) -> list[str]:
    """Split text at word boundaries to stay under max_chars per segment.

    For unpunctuated lecture transcripts where sentence splitting fails.
    """
    words = text.split()
    segments = []
    current = ""

    for word in words:
        if current and len(current) + 1 + len(word) > max_chars:
            segments.append(current)
            current = word
        else:
            current = current + " " + word if current else word

    if current:
        segments.append(current)

    return segments


def _split_into_chunks(
    text: str,
    source_id: str,
    lecturer: str | None,
    max_chunk_chars: int = 2000,
    min_chunk_chars: int = 100,
) -> list[Chunk]:
    """Split transcript text into chunks at timestamp boundaries."""
    chunks = []
    current_lecture_num: int | None = None

    # Find all timestamp positions
    matches = list(TIMESTAMP_PATTERN.finditer(text))
    if not matches:
        return chunks

    # Also track lecture headers for lecture_number assignment
    lecture_headers = {m.start(): m.group(1) for m in LECTURE_HEADER_PATTERN.finditer(text)}

    for i, match in enumerate(matches):
        timestamp = match.group(1)
        youtube_url = match.group(2)

        # Check if there's a lecture header before this timestamp
        for header_pos, lecture_id in sorted(lecture_headers.items()):
            if header_pos < match.start():
                # Extract numeric part
                num_match = re.match(r'(\d+)', lecture_id)
                if num_match:
                    current_lecture_num = int(num_match.group(1))

        # Extract text between this timestamp and the next (or end of file)
        start = match.end()
        if i + 1 < len(matches):
            # Find the start of the next timestamp line — go back to include any
            # lecture header that might precede it
            end = matches[i + 1].start()
        else:
            end = len(text)

        chunk_text = text[start:end].strip()

        # Remove any lecture headers from the chunk text
        chunk_text = LECTURE_HEADER_PATTERN.sub('', chunk_text).strip()

        if len(chunk_text) < min_chunk_chars:
            continue

        # Split long chunks at paragraph or sentence boundaries
        if len(chunk_text) > max_chunk_chars:
            segments = _split_long_text(chunk_text, max_chunk_chars, min_chunk_chars)
            for seg in segments:
                chunks.append(Chunk(
                    source_id=source_id,
                    text=seg,
                    language="en",
                    youtube_timestamp=timestamp,
                    youtube_url=youtube_url,
                    paragraph_index=len(chunks),
                    lecturer=lecturer,
                    lecture_number=current_lecture_num,
                    chunk_method="timestamp",
                ))
        else:
            chunks.append(Chunk(
                source_id=source_id,
                text=chunk_text,
                language="en",
                youtube_timestamp=timestamp,
                youtube_url=youtube_url,
                paragraph_index=len(chunks),
                lecturer=lecturer,
                lecture_number=current_lecture_num,
                chunk_method="timestamp",
            ))

    return chunks


class TranscriptIngester:
    """Ingests lecture transcript markdown files."""

    def parse_transcript(self, path: Path) -> tuple[Source, list[Chunk]]:
        """Parse a transcript markdown file into Source + Chunks.

        Returns (source, chunks) without embedding or persisting.
        """
        text = path.read_text(encoding="utf-8")
        metadata = _extract_header_metadata(text)

        lecturer = metadata.get("lecturer", path.stem)

        source = Source(
            title=f"{lecturer} Lectures on Hegel's Science of Logic",
            type=SourceType.TRANSCRIPT,
            author=[lecturer],
            source_path=str(path),
            source_url=metadata.get("playlist_url"),
            metadata=metadata,
        )

        chunks = _split_into_chunks(
            text,
            source_id=source.id,
            lecturer=lecturer,
        )

        return source, chunks

    def ingest(
        self,
        path: Path,
        db: Database,
        embedder: EmbeddingPipeline,
        index: FAISSIndex,
    ) -> Source:
        """Parse, embed, and store a transcript.

        Full pipeline: parse → embed batch → store in DB + FAISS.
        """
        source, chunks = self.parse_transcript(path)

        if not chunks:
            raise ValueError(f"No chunks extracted from {path}")

        # Embed all chunks
        texts = [c.text for c in chunks]
        embeddings = embedder.embed_batch(texts)

        # Store in database
        db.insert_source(source)
        db.insert_chunks_batch(chunks)

        # Add to FAISS index
        chunk_ids = [c.id for c in chunks]
        index.add_batch(chunk_ids, embeddings)

        return source
