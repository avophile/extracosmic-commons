"""Critical edition PDF ingester with character-level extraction.

Ported from the hegel-bilingual project's build_json_v3.py. Uses pdfplumber
character-level positioning for precise text reconstruction, preserving:
- Correct word spacing (gap > 1.5pt = space)
- Paragraph indentation detection
- Font-based heading classification
- Margin reference extraction (GW pages, Akademie refs, Bekker numbers, etc.)

This ingester is used for scholarly critical editions where basic pypdf
extraction loses margin annotations and font metadata. Regular PDFs still
use the faster pypdf-based PDFIngester.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pdfplumber

from ..database import Database
from ..edition_profiles import EditionProfile
from ..embeddings import EmbeddingPipeline
from ..index import FAISSIndex
from ..models import Chunk, Source, SourceType

logger = logging.getLogger(__name__)


# ── Character-level extraction (ported from hegel-bilingual) ──


def _build_line_from_chars(chars: list[dict], space_threshold: float = 1.5) -> str:
    """Build a text line from pdfplumber character objects.

    Inserts spaces based on horizontal gaps between characters.
    Word boundaries typically have gaps of ~2.8-3.9pt; ligature pairs ~0pt.
    The threshold of 1.5pt catches all word boundaries without splitting
    ligatures.
    """
    if not chars:
        return ""
    parts = [chars[0].get("text", "")]
    for i in range(1, len(chars)):
        prev_x1 = chars[i - 1].get("x1", 0)
        curr_x0 = chars[i].get("x0", 0)
        gap = curr_x0 - prev_x1
        if gap > space_threshold:
            parts.append(" ")
        parts.append(chars[i].get("text", ""))
    return "".join(parts)


def extract_page_text(
    page,
    space_threshold: float = 1.5,
    line_threshold: float = 3.0,
    indent_threshold: float = 8.0,
) -> tuple[str, list[tuple[str, float, list[dict]]]]:
    """Extract text from a pdfplumber page using character-level positioning.

    Returns:
        (full_text, line_data)
        - full_text: reconstructed text with paragraph breaks as \\n\\n
        - line_data: list of (line_text, first_char_x0, line_chars) for font analysis
    """
    chars = page.chars
    if not chars:
        return "", []

    chars_sorted = sorted(chars, key=lambda c: (round(c["top"], 1), c["x0"]))

    line_data: list[tuple[str, float, list[dict]]] = []
    current_line_chars: list[dict] = []
    current_y: float | None = None

    for char in chars_sorted:
        text = char.get("text", "")
        if not text:
            continue

        y = round(char["top"], 1)

        if current_y is not None and abs(y - current_y) > line_threshold:
            if current_line_chars:
                lt = _build_line_from_chars(current_line_chars, space_threshold)
                fx = current_line_chars[0]["x0"]
                line_data.append((lt, fx, list(current_line_chars)))
            current_line_chars = []
            current_y = y

        if current_y is None:
            current_y = y

        current_line_chars.append(char)

    if current_line_chars:
        lt = _build_line_from_chars(current_line_chars, space_threshold)
        fx = current_line_chars[0]["x0"]
        line_data.append((lt, fx, list(current_line_chars)))

    if len(line_data) < 3:
        return "\n".join(t for t, _, _ in line_data), line_data

    # Determine page margin from most common first-char x-position
    body_x0s = [round(x0 / 2) * 2 for t, x0, _ in line_data if len(t) > 25]
    if not body_x0s:
        return "\n".join(t for t, _, _ in line_data), line_data

    margin_x = Counter(body_x0s).most_common(1)[0][0]

    # Build text with paragraph breaks at indented lines
    output_lines = []
    for text, x0, _ in line_data:
        rx0 = round(x0 / 2) * 2
        is_indented = rx0 > margin_x + indent_threshold and len(text) > 25
        if is_indented:
            output_lines.append("")  # paragraph break marker
        output_lines.append(text)

    return "\n".join(output_lines), line_data


# ── Font-based heading classification ──


def _is_small_caps(fontname: str) -> bool:
    return "SC" in fontname or "SmallCaps" in fontname.lower()


def _is_italic(fontname: str) -> bool:
    return "Italic" in fontname or "Oblique" in fontname


def _is_bold(fontname: str) -> bool:
    return "Bold" in fontname or "Black" in fontname


def classify_line_by_font(line_chars: list[dict]) -> str:
    """Classify a line as 'heading' or 'body' using font metadata.

    Ported from hegel-bilingual build_json_v3.py. Uses font name, size,
    and position to detect headings without fragile regex matching.
    """
    if not line_chars:
        return "body"

    fonts = []
    sizes = []
    for c in line_chars:
        if c.get("text", "").strip():
            fonts.append(c.get("fontname", ""))
            sizes.append(float(c.get("size", 11)))

    if not fonts:
        return "body"

    avg_size = sum(sizes) / len(sizes)
    n_sc = sum(1 for f in fonts if _is_small_caps(f))
    n_italic = sum(1 for f in fonts if _is_italic(f))
    n_bold = sum(1 for f in fonts if _is_bold(f))
    n_total = len(fonts)

    # Small caps → always heading
    if n_sc > n_total * 0.5:
        return "heading"

    # Large italic (>12pt) → section/chapter title
    if avg_size > 12.0 and n_italic > n_total * 0.5:
        return "heading"

    # Bold-only lines are often margin refs (GW numbers), not headings
    if n_bold > n_total * 0.5:
        return "body"

    # Short italic at body size → sub-heading (Remark, Note, etc.)
    text = "".join(c.get("text", "") for c in line_chars)
    text_len = len(text.replace(" ", ""))
    if n_italic > n_total * 0.8 and avg_size < 12.5 and text_len < 50:
        return "heading"

    return "body"


# ── Reference extraction ──


def extract_margin_refs(
    text: str, line_data: list, profile: EditionProfile
) -> list[str]:
    """Extract edition reference markers from page text.

    Uses the profile's margin/header/inline patterns to find references
    like GW page numbers, Akademie refs, Bekker numbers, etc.
    """
    refs = []

    if profile.margin_ref_pattern:
        for m in profile.margin_ref_pattern.finditer(text):
            refs.append(m.group(1))

    if profile.header_ref_pattern:
        # Check only first few lines (running headers)
        for line_text, _, _ in line_data[:3]:
            for m in profile.header_ref_pattern.finditer(line_text):
                refs.append(m.group(1))

    if profile.inline_ref_pattern:
        for m in profile.inline_ref_pattern.finditer(text):
            refs.append(m.group(1))

    return refs


# ── Critical Edition Ingester ──


class CriticalEditionIngester:
    """Ingest PDFs with character-level extraction and edition-aware page references."""

    def __init__(self, profile: EditionProfile):
        self.profile = profile

    def _compute_translation_page(self, pdf_page: int) -> int:
        """Map physical PDF page to printed translation page."""
        return pdf_page + self.profile.page_offset

    def parse_pdf(
        self,
        path: Path,
        title: str | None = None,
        author: list[str] | None = None,
        extra_metadata: dict | None = None,
    ) -> tuple[Source, list[Chunk]]:
        """Extract text with character-level precision and edition-aware references.

        Each chunk gets structural_ref populated with:
        - translation_page (printed page)
        - edition-specific reference (gw_page, akademie_ref, etc.)
        - heading info from font-based classification
        """
        profile_meta = self.profile.to_source_metadata()
        if extra_metadata:
            profile_meta.update(extra_metadata)

        source = Source(
            title=title or path.stem,
            type=SourceType.PDF,
            author=author or [],
            language=[self.profile.original_language if self.profile.is_original_language else "en"],
            source_path=str(path),
            metadata=profile_meta,
        )

        chunks = []
        current_heading: str | None = None
        current_section_refs: dict[str, Any] = {}

        with pdfplumber.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text, line_data = extract_page_text(page)
                if not text or len(text.strip()) < 50:
                    continue

                # Extract edition references from this page
                refs = extract_margin_refs(text, line_data, self.profile)
                translation_page = self._compute_translation_page(page_num)

                # Classify lines for heading detection
                for line_text, _, line_chars in line_data:
                    if classify_line_by_font(line_chars) == "heading":
                        heading_text = line_text.strip()
                        if len(heading_text) > 3 and len(heading_text) < 200:
                            current_heading = heading_text

                # Build structural_ref for this page's chunks
                structural_ref: dict[str, Any] = {
                    "translation_page": translation_page,
                }
                if refs:
                    # Use profile's ref_label as the key
                    ref_key = f"{self.profile.ref_label.lower()}_page"
                    structural_ref[ref_key] = refs[0]  # primary ref for this page
                    if len(refs) > 1:
                        structural_ref[f"{ref_key}_range"] = f"{refs[0]}-{refs[-1]}"
                if current_heading:
                    structural_ref["heading"] = current_heading
                structural_ref.update(current_section_refs)

                # Split page text into paragraph chunks
                paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                for para_idx, para_text in enumerate(paragraphs):
                    if len(para_text) < 30:  # skip very short fragments
                        continue

                    chunks.append(Chunk(
                        source_id=source.id,
                        text=para_text,
                        language=source.language[0],
                        structural_ref=dict(structural_ref),
                        pdf_page=page_num,
                        paragraph_index=len(chunks),
                        chunk_method="critical_edition",
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
        extra_metadata: dict | None = None,
    ) -> Source:
        """Full pipeline: extract → chunk with refs → embed → store."""
        source, chunks = self.parse_pdf(
            path, title=title, author=author, extra_metadata=extra_metadata
        )

        if not chunks:
            logger.warning(f"No chunks extracted from {path.name}")
            db.insert_source(source)
            return source

        texts = [c.text for c in chunks]
        embeddings = embedder.embed_batch(texts)

        db.insert_source(source)
        db.insert_chunks_batch(chunks)
        index.add_batch([c.id for c in chunks], embeddings)

        return source
