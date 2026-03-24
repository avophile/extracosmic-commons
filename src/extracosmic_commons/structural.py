"""Structural tagger for Extracosmic Commons.

Extracts structural metadata from chunk text and populates the structural_ref
JSON field. Works across all source types — Hegel's § system, Kant's CPR
divisions, generic chapter/section headings, abstract/introduction/conclusion
structures, and table of contents hierarchies.

Two-pass approach:
1. Heading detection — which chunks ARE structural headings
2. Section propagation — which section does each body chunk BELONG to
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from .database import Database
from .models import Chunk

logger = logging.getLogger(__name__)

# --- Heading detection patterns ---

# Hegel-specific
HEGEL_SECTION_RE = re.compile(r'§\s*(\d+)', re.IGNORECASE)
HEGEL_GW_PAGE_RE = re.compile(r'(?:GW|Gesammelte\s+Werke)\s*(\d+)[.:]\s*(\d+)', re.IGNORECASE)
HEGEL_GW_SHORT_RE = re.compile(r'\((\d{1,2}):(\d+)\)')
HEGEL_DOCTRINE_RE = re.compile(
    r'(?:Doctrine|Lehre)\s+(?:of|vom|von der|des)\s+(Being|Essence|(?:the\s+)?Concept|Notion|Sein|Wesen|Begriff)',
    re.IGNORECASE,
)
HEGEL_NAMED_SECTIONS = {
    "Determinate Being": "Determinate Being",
    "Dasein": "Determinate Being",
    "Being-for-self": "Being-for-self",
    "Fürsichsein": "Being-for-self",
    "Quality": "Quality",
    "Qualität": "Quality",
    "Quantity": "Quantity",
    "Quantität": "Quantity",
    "Measure": "Measure",
    "Maß": "Measure",
    "Becoming": "Becoming",
    "Werden": "Becoming",
}

# Generic academic structure
CHAPTER_RE = re.compile(
    r'^(?:CHAPTER|Chapter|Chapitre|Kapitel)\s+(\d+|[IVXLC]+)[.:\s—–-]\s*(.+)?',
    re.MULTILINE,
)
SECTION_NUMBERED_RE = re.compile(
    r'^(?:Section|SECTION|Abschnitt)\s+(\d+(?:\.\d+)*|[IVXLC]+)[.:\s—–-]\s*(.+)?',
    re.MULTILINE,
)
PART_RE = re.compile(
    r'^(?:PART|Part|Teil)\s+(\d+|[IVXLC]+)[.:\s—–-]\s*(.+)?',
    re.MULTILINE,
)
NUMBERED_HEADING_RE = re.compile(
    r'^(\d+(?:\.\d+)*)\s+([A-Z][A-Za-z\s,:\-—]+)$',
    re.MULTILINE,
)

# Divisions common in academic texts
DIVISION_KEYWORDS = [
    "Abstract", "Introduction", "Conclusion", "Summary", "Preface",
    "Foreword", "Epilogue", "Prologue", "Acknowledgments", "Acknowledgements",
    "Appendix", "Bibliography", "References", "Notes", "Index",
    "Methodology", "Results", "Discussion", "Literature Review",
    "Einleitung", "Einführung", "Schluss", "Zusammenfassung",  # German
]
DIVISION_RE = re.compile(
    r'^(' + '|'.join(re.escape(d) for d in DIVISION_KEYWORDS) + r')\s*$',
    re.MULTILINE | re.IGNORECASE,
)

# Hegel subsections
HEGEL_SUBSECTION_RE = re.compile(
    r'^(Remark|Note|Addition|Anmerkung|Zusatz|Beweis)\b',
    re.MULTILINE | re.IGNORECASE,
)

# ALL-CAPS headings (common in PDF extraction)
ALLCAPS_HEADING_RE = re.compile(r'^([A-Z][A-Z\s,:\-—]{3,80})$', re.MULTILINE)

# Lecture transcript verbal references
SPOKEN_SECTION_RE = re.compile(
    r'(?:section|paragraph|§)\s*(\d+)', re.IGNORECASE
)
SPOKEN_DOCTRINE_RE = re.compile(
    r'(?:doctrine of|now in)\s+(being|essence|(?:the\s+)?concept|notion)',
    re.IGNORECASE,
)

# Heading hierarchy levels (for nesting)
HEADING_LEVELS = {
    "part": 1,
    "division": 2,
    "chapter": 3,
    "section": 4,
    "subsection": 5,
    "heading": 6,
}


@dataclass
class HeadingMatch:
    """A detected heading within a chunk."""

    level: str  # part, division, chapter, section, subsection, heading
    key: str  # the structural_ref key to set
    value: str  # the value to assign
    confidence: float = 1.0  # 0-1, how certain we are this is a real heading


def detect_headings(text: str) -> list[HeadingMatch]:
    """Detect structural headings in a chunk's text.

    Returns all detected headings, from most to least specific.
    """
    matches = []

    # Hegel § references
    for m in HEGEL_SECTION_RE.finditer(text):
        matches.append(HeadingMatch("section", "section", f"§{m.group(1)}"))

    # GW page references
    for m in HEGEL_GW_PAGE_RE.finditer(text):
        matches.append(HeadingMatch("section", "gw_page", f"{m.group(1)}.{m.group(2)}"))
    for m in HEGEL_GW_SHORT_RE.finditer(text):
        vol, page = m.group(1), m.group(2)
        if 20 <= int(vol) <= 22:  # GW volumes for Hegel's Logic
            matches.append(HeadingMatch("section", "gw_page", f"{vol}.{page}"))

    # Hegel doctrine
    for m in HEGEL_DOCTRINE_RE.finditer(text):
        doctrine = m.group(1).strip()
        # Normalize German/English
        for norm in ("Being", "Essence", "Concept"):
            if norm.lower() in doctrine.lower() or doctrine.lower() in ("sein", "wesen", "begriff"):
                matches.append(HeadingMatch("division", "doctrine", norm))
                break

    # Hegel named sections
    for keyword, normalized in HEGEL_NAMED_SECTIONS.items():
        if keyword.lower() in text.lower():
            matches.append(HeadingMatch("chapter", "chapter", normalized, confidence=0.7))

    # Hegel subsections (Remark, Note, Addition)
    for m in HEGEL_SUBSECTION_RE.finditer(text):
        matches.append(HeadingMatch("subsection", "subsection", m.group(1).capitalize()))

    # Part headings
    for m in PART_RE.finditer(text):
        title = m.group(2).strip() if m.group(2) else ""
        value = f"{m.group(1)}. {title}".strip().rstrip(".")
        matches.append(HeadingMatch("part", "part", value))

    # Chapter headings
    for m in CHAPTER_RE.finditer(text):
        title = m.group(2).strip() if m.group(2) else ""
        value = f"{m.group(1)}. {title}".strip().rstrip(".")
        matches.append(HeadingMatch("chapter", "chapter", value))

    # Numbered section headings (Section 1.2 or SECTION III)
    for m in SECTION_NUMBERED_RE.finditer(text):
        title = m.group(2).strip() if m.group(2) else ""
        value = f"{m.group(1)}. {title}".strip().rstrip(".")
        matches.append(HeadingMatch("section", "section", value))

    # Numbered headings without "Chapter"/"Section" prefix (1.2.3 Title)
    for m in NUMBERED_HEADING_RE.finditer(text):
        num = m.group(1)
        title = m.group(2).strip()
        if len(title) < 80:  # Avoid matching body text
            level = "chapter" if "." not in num else "section"
            matches.append(HeadingMatch(level, level, f"{num} {title}", confidence=0.8))

    # Division keywords (Introduction, Conclusion, etc.)
    for m in DIVISION_RE.finditer(text):
        matches.append(HeadingMatch("division", "division", m.group(1).capitalize()))

    # ALL-CAPS headings
    for m in ALLCAPS_HEADING_RE.finditer(text):
        heading = m.group(1).strip()
        # Filter out common false positives
        if len(heading) > 3 and heading.count(" ") < 10:
            matches.append(HeadingMatch("heading", "heading", heading.title(), confidence=0.6))

    # Spoken references in lecture transcripts
    for m in SPOKEN_SECTION_RE.finditer(text):
        matches.append(HeadingMatch("section", "section", f"§{m.group(1)}", confidence=0.8))
    for m in SPOKEN_DOCTRINE_RE.finditer(text):
        doctrine = m.group(1).strip().capitalize()
        if doctrine == "The concept" or doctrine == "Notion":
            doctrine = "Concept"
        matches.append(HeadingMatch("division", "doctrine", doctrine, confidence=0.7))

    return matches


def _is_heading_chunk(chunk: Chunk, matches: list[HeadingMatch]) -> bool:
    """Determine if a chunk is primarily a heading (vs body text with a reference).

    A chunk is a heading if it's short and its main content is the heading itself.
    Body text that happens to mention § numbers is not a heading chunk.
    """
    if not matches:
        return False
    # Short chunks with high-confidence matches are headings
    if len(chunk.text) < 200 and any(m.confidence >= 0.8 for m in matches):
        return True
    # Very short chunks are almost always headings
    if len(chunk.text) < 50:
        return True
    return False


def _matches_to_ref(matches: list[HeadingMatch]) -> dict[str, Any]:
    """Convert heading matches to a structural_ref dict.

    Keeps the highest-confidence match for each key.
    """
    ref: dict[str, Any] = {}
    for m in sorted(matches, key=lambda x: x.confidence, reverse=True):
        if m.key not in ref:
            ref[m.key] = m.value
    return ref


class StructuralTagger:
    """Extracts structural references from chunk text and source context.

    Two-pass approach:
    1. Detect headings in each chunk
    2. Propagate heading context to subsequent body chunks
    """

    def tag_source(self, source_id: str, db: Database) -> int:
        """Tag all chunks for a single source. Returns count of chunks updated."""
        chunks = db.get_chunks_by_source(source_id)
        if not chunks:
            return 0

        # Sort by paragraph_index (or pdf_page as fallback)
        chunks.sort(key=lambda c: (c.pdf_page or 0, c.paragraph_index or 0))

        # Pass 1: Detect headings in each chunk
        chunk_matches: dict[str, list[HeadingMatch]] = {}
        for chunk in chunks:
            matches = detect_headings(chunk.text)
            if matches:
                chunk_matches[chunk.id] = matches

        # Pass 2: Propagate heading context
        current_context: dict[str, Any] = {}
        updated = 0

        for chunk in chunks:
            matches = chunk_matches.get(chunk.id, [])

            if matches:
                ref = _matches_to_ref(matches)

                if _is_heading_chunk(chunk, matches):
                    # This IS a heading — update context
                    # A new chapter resets section/subsection
                    for m in matches:
                        level_num = HEADING_LEVELS.get(m.level, 99)
                        # Clear more specific levels when a broader level changes
                        keys_to_clear = [
                            k for k, v_level in [
                                ("subsection", 5), ("heading", 6),
                                ("section", 4), ("chapter", 3),
                            ]
                            if v_level >= level_num and k in current_context
                        ]
                        for k in keys_to_clear:
                            del current_context[k]
                    current_context.update(ref)
                else:
                    # Body text with references — merge with current context
                    ref = {**current_context, **ref}

                # Update chunk
                chunk.structural_ref = ref
                db.conn.execute(
                    "UPDATE chunks SET structural_ref = ? WHERE id = ?",
                    (json.dumps(ref), chunk.id),
                )
                updated += 1
            elif current_context:
                # No headings detected — inherit from current context
                if chunk.structural_ref is None:
                    chunk.structural_ref = dict(current_context)
                    db.conn.execute(
                        "UPDATE chunks SET structural_ref = ? WHERE id = ?",
                        (json.dumps(current_context), chunk.id),
                    )
                    updated += 1

        db.conn.commit()
        return updated

    def tag_corpus(
        self,
        db: Database,
        source_type: str | None = None,
        dry_run: bool = False,
    ) -> int:
        """Tag all sources in the database.

        Args:
            db: Database instance.
            source_type: Optional filter by source type.
            dry_run: If True, count what would be tagged without writing.

        Returns:
            Total number of chunks updated (or would-be-updated in dry_run).
        """
        sources = db.get_all_sources()
        if source_type:
            sources = [s for s in sources if s.type.value == source_type]

        total_updated = 0
        for source in sources:
            if dry_run:
                chunks = db.get_chunks_by_source(source.id)
                untagged = sum(1 for c in chunks if c.structural_ref is None)
                if untagged:
                    logger.info(f"  Would tag {untagged} chunks in: {source.title}")
                    total_updated += untagged
            else:
                n = self.tag_source(source.id, db)
                if n:
                    logger.info(f"  Tagged {n} chunks in: {source.title}")
                total_updated += n

        return total_updated
