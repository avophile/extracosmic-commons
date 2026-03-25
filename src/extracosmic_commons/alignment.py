"""Cross-translation alignment engine for Extracosmic Commons.

Assigns canonical section IDs and aligns paragraphs across editions of
the same work. The original language text defines the canonical paragraph
structure; translations are aligned to it.

Two operations:
1. build_canonical_sections — assign canonical_section IDs from heading hierarchy
2. align_editions — group translation ¶s under original-language ¶s
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .database import Database
from .edition_profiles import get_profiles_for_work
from .models import Chunk, Source

logger = logging.getLogger(__name__)


def _normalize_heading(heading: str) -> str:
    """Normalize a heading for cross-edition matching.

    Strip numbering, case, and minor wording differences so that
    "Chapter 1. Being" and "Chapter 1: Being" match.
    """
    h = heading.strip().lower()
    # Remove leading numbering (Chapter 1., Section I., etc.)
    h = re.sub(r'^(?:chapter|section|part|abschnitt|kapitel|teil)\s*[\divxlc]+[.:—–\-\s]*', '', h)
    # Remove leading letter labels (a., b., c., A., B., C.)
    h = re.sub(r'^[a-cα-γ]\.\s*', '', h)
    # Normalize whitespace
    h = re.sub(r'\s+', ' ', h).strip()
    return h


def _heading_to_canonical_id(work_id: str, heading: str, context: dict) -> str:
    """Generate a canonical section ID from a heading and its context.

    Example: "hegel-sol:being.quality.being-for-self"
    """
    normalized = _normalize_heading(heading)
    # Create a slug
    slug = re.sub(r'[^a-z0-9]+', '-', normalized).strip('-')
    if not slug:
        slug = "untitled"

    # Build path from context (doctrine, part, etc.)
    parts = [work_id]
    for key in ("doctrine", "part", "division"):
        val = context.get(key)
        if val:
            parts.append(re.sub(r'[^a-z0-9]+', '-', val.lower()).strip('-'))
    parts.append(slug)

    return ":".join(parts[:2]) + "." + ".".join(parts[2:]) if len(parts) > 2 else ":".join(parts)


def build_canonical_sections(work_id: str, db: Database) -> int:
    """Assign canonical_section IDs to all chunks of a work.

    Walks each source's chunks in order. When a heading is found in
    structural_ref, generates a canonical ID and propagates it to
    subsequent body chunks until the next heading.

    Returns count of chunks updated.
    """
    sources = db.get_sources_by_work_id(work_id)
    if not sources:
        logger.warning(f"No sources found for work_id={work_id}")
        return 0

    updated = 0

    for source in sources:
        chunks = db.get_chunks_by_source(source.id)
        if not chunks:
            continue

        chunks.sort(key=lambda c: (c.pdf_page or 0, c.paragraph_index or 0))

        current_canonical: str | None = None
        current_context: dict[str, str] = {}

        for chunk in chunks:
            ref = chunk.structural_ref or {}
            heading = ref.get("heading")

            # Track context from broader structural elements
            for key in ("doctrine", "part", "division"):
                if key in ref:
                    current_context[key] = ref[key]

            # If chunk has a heading, generate canonical ID
            if heading and len(heading) > 2:
                current_canonical = _heading_to_canonical_id(
                    work_id, heading, current_context
                )

            # Assign canonical_section if we have one
            if current_canonical and ref.get("canonical_section") != current_canonical:
                new_ref = {**ref, "canonical_section": current_canonical}
                db.conn.execute(
                    "UPDATE chunks SET structural_ref = ? WHERE id = ?",
                    (json.dumps(new_ref), chunk.id),
                )
                updated += 1

        db.conn.commit()
        if updated:
            logger.info(f"  Assigned canonical sections: {source.title[:50]} ({updated} chunks)")

    return updated


def align_editions(work_id: str, db: Database) -> int:
    """Align paragraphs across editions using GW page refs as anchors.

    For each source pair sharing a GW page, groups translation ¶s under
    the corresponding original-language ¶. Updates structural_ref with
    canonical_para and alignment_confidence.

    Returns count of chunks aligned.
    """
    sources = db.get_sources_by_work_id(work_id)
    if len(sources) < 2:
        logger.info(f"Only {len(sources)} source(s) for {work_id}, nothing to align")
        return 0

    # Find the original-language source(s)
    originals = [s for s in sources if s.metadata.get("is_original_language")]
    translations = [s for s in sources if not s.metadata.get("is_original_language")]

    if not translations:
        logger.info("No translations found to align")
        return 0

    updated = 0

    # For each translation, align its chunks to original or to other translations
    for trans_source in translations:
        trans_chunks = db.get_chunks_by_source(trans_source.id)
        if not trans_chunks:
            continue

        trans_chunks.sort(key=lambda c: (c.pdf_page or 0, c.paragraph_index or 0))

        # Build a map of gw_page → chunks from this translation
        for chunk in trans_chunks:
            ref = chunk.structural_ref or {}
            gw_page = ref.get("gw_page")

            if gw_page and not ref.get("canonical_para"):
                # Assign canonical_para based on GW page
                canonical_para = f"{work_id}:gw.{gw_page}"
                new_ref = {
                    **ref,
                    "canonical_para": canonical_para,
                    "alignment_confidence": 0.9,
                }
                db.conn.execute(
                    "UPDATE chunks SET structural_ref = ? WHERE id = ?",
                    (json.dumps(new_ref), chunk.id),
                )
                updated += 1

    # Also update original chunks with canonical_para
    for orig_source in originals:
        orig_chunks = db.get_chunks_by_source(orig_source.id)
        for chunk in orig_chunks:
            ref = chunk.structural_ref or {}
            gw_page = ref.get("gw_page")
            if gw_page and not ref.get("canonical_para"):
                canonical_para = f"{work_id}:gw.{gw_page}"
                new_ref = {
                    **ref,
                    "canonical_para": canonical_para,
                    "alignment_confidence": 1.0,
                }
                db.conn.execute(
                    "UPDATE chunks SET structural_ref = ? WHERE id = ?",
                    (json.dumps(new_ref), chunk.id),
                )
                updated += 1

    db.conn.commit()
    logger.info(f"Aligned {updated} chunks for {work_id}")
    return updated


def align_work(work_id: str, db: Database) -> dict[str, int]:
    """Full alignment pipeline for a work: canonical sections + edition alignment.

    Returns stats dict.
    """
    sections = build_canonical_sections(work_id, db)
    aligned = align_editions(work_id, db)
    return {
        "canonical_sections_assigned": sections,
        "paragraphs_aligned": aligned,
    }
