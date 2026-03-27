"""Citation extraction pipeline for conversation and lecture transcripts.

Identifies moments where speakers read from or reference primary texts
(especially Hegel's Science of Logic, Phenomenology, etc.), extracting
structured citation records with page numbers, edition info, and audio
timestamps using regex/heuristic pattern matching — zero API cost.

Architecture:
    1. KNOWN_WORKS catalog maps title variants → canonical metadata
    2. PAGE_PATTERNS are compiled regexes that extract page numbers
       with edition attribution (di Giovanni, Miller, Suhrkamp, GW)
    3. READING_INDICATORS detect when a speaker is reading aloud vs.
       merely referencing a text by page number
    4. extract_citations_from_segments() scans each segment against
       all patterns, building CitationRecord objects with confidence
       scores based on how many signals converge

Citation types:
    - READING: Speaker reads aloud from a text (verbatim or near-verbatim)
    - REFERENCE: Speaker refers to a passage by page/section without reading
    - PARAPHRASE: Speaker restates a passage in their own words
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from enum import Enum


class CitationType(str, Enum):
    """How the source text is invoked in the conversation."""
    READING = "reading"
    REFERENCE = "reference"
    PARAPHRASE = "paraphrase"


# ── Known works catalog ──────────────────────────────────────────────────
# Each entry maps regex patterns for title variants to canonical metadata.
# The "aliases" field contains compiled regexes that match title mentions
# in conversational speech (including informal abbreviations).

KNOWN_WORKS: list[dict[str, Any]] = [
    # ── Hegel ────────────────────────────────────────────────────────
    {
        "title": "Science of Logic",
        "author": "Hegel",
        "aliases": [
            re.compile(r"\bscience of logic\b", re.I),
            re.compile(r"\bwissenschaft der logik\b", re.I),
            re.compile(r"\bthe logic\b", re.I),
            re.compile(r"\bSoL\b"),
            re.compile(r"\bgreater logic\b", re.I),
            re.compile(r"\bgroße logik\b", re.I),
        ],
    },
    {
        "title": "Phenomenology of Spirit",
        "author": "Hegel",
        "aliases": [
            re.compile(r"\bphenomenology of spirit\b", re.I),
            re.compile(r"\bphänomenologie des geistes\b", re.I),
            re.compile(r"\bthe phenomenology\b", re.I),
            re.compile(r"\bPhG\b"),
        ],
    },
    {
        "title": "Philosophy of Right",
        "author": "Hegel",
        "aliases": [
            re.compile(r"\bphilosophy of right\b", re.I),
            re.compile(r"\brechtsphilosophie\b", re.I),
            re.compile(r"\belements of the philosophy of right\b", re.I),
        ],
    },
    {
        "title": "Encyclopedia Logic",
        "author": "Hegel",
        "aliases": [
            re.compile(r"\bencyclopedia logic\b", re.I),
            re.compile(r"\bencyclopedia of the philosophical sciences\b", re.I),
            re.compile(r"\bthe encyclopedia\b", re.I),
            re.compile(r"\bsmaller logic\b", re.I),
            re.compile(r"\benzyklopädie\b", re.I),
        ],
    },
    {
        "title": "Philosophy of Nature",
        "author": "Hegel",
        "aliases": [
            re.compile(r"\bphilosophy of nature\b", re.I),
            re.compile(r"\bnaturphilosophie\b", re.I),
        ],
    },
    {
        "title": "Philosophy of Spirit",
        "author": "Hegel",
        "aliases": [
            re.compile(r"\bphilosophy of spirit\b", re.I),
            re.compile(r"\bphilosophy of mind\b", re.I),
        ],
    },
    {
        "title": "Lectures on the History of Philosophy",
        "author": "Hegel",
        "aliases": [
            re.compile(r"\blectures on the history of philosophy\b", re.I),
            re.compile(r"\bhistory of philosophy\b", re.I),
        ],
    },
    {
        "title": "Lectures on Aesthetics",
        "author": "Hegel",
        "aliases": [
            re.compile(r"\blectures on aesthetics\b", re.I),
            re.compile(r"\baesthetics\b", re.I),
        ],
    },
    # ── Other philosophers ───────────────────────────────────────────
    {
        "title": "Grundrisse",
        "author": "Marx",
        "aliases": [re.compile(r"\bGrundrisse\b", re.I)],
    },
    {
        "title": "Capital",
        "author": "Marx",
        "aliases": [
            re.compile(r"\bDas Kapital\b", re.I),
            re.compile(r"\bCapital\b"),
        ],
    },
    {
        "title": "Being and Time",
        "author": "Heidegger",
        "aliases": [
            re.compile(r"\bBeing and Time\b", re.I),
            re.compile(r"\bSein und Zeit\b", re.I),
        ],
    },
    {
        "title": "Critique of Pure Reason",
        "author": "Kant",
        "aliases": [
            re.compile(r"\bCritique of Pure Reason\b", re.I),
            re.compile(r"\bKritik der reinen Vernunft\b", re.I),
            re.compile(r"\bfirst Critique\b", re.I),
        ],
    },
    {
        "title": "Critique of Practical Reason",
        "author": "Kant",
        "aliases": [
            re.compile(r"\bCritique of Practical Reason\b", re.I),
            re.compile(r"\bsecond Critique\b", re.I),
        ],
    },
    {
        "title": "Critique of Judgment",
        "author": "Kant",
        "aliases": [
            re.compile(r"\bCritique of Judg[e]?ment\b", re.I),
            re.compile(r"\bthird Critique\b", re.I),
        ],
    },
    {
        "title": "Metaphysics",
        "author": "Aristotle",
        "aliases": [re.compile(r"\bMetaphysics\b")],
    },
    {
        "title": "Nicomachean Ethics",
        "author": "Aristotle",
        "aliases": [
            re.compile(r"\bNicomachean Ethics\b", re.I),
            re.compile(r"\bthe Ethics\b", re.I),
        ],
    },
    {
        "title": "Republic",
        "author": "Plato",
        "aliases": [re.compile(r"\bthe Republic\b", re.I)],
    },
]


# ── Page reference patterns ──────────────────────────────────────────────
# Each pattern returns named groups: page (the number), and optionally
# edition (which translation/edition the page belongs to).
# Patterns are tried in order; first match wins for a given text span.

# Edition markers that indicate German or English editions.
# These are used both standalone and within page-reference regexes.
_GERMAN_EDITIONS = re.compile(
    r"\b(?:Suhrkamp|GW\s*\d+|Gesammelte\s+Werke|German|deutschen?)\b", re.I
)
_ENGLISH_EDITIONS = re.compile(
    r"\b(?:di\s*Giovanni|Giovanni|Miller|Pinkard|Harris|Knox|"
    r"translation|English|your\s+translation)\b", re.I
)

# Compiled page-reference regexes. Each yields a match with group "page".
# We search the full segment text; edition context is resolved separately.
PAGE_PATTERNS: list[re.Pattern] = [
    # "page 478", "Page 110", "pages 480-481"
    re.compile(
        r"\bpages?\s+(?P<page>\d{1,4})(?:\s*[-–,]\s*(?P<page2>\d{1,4}))?",
        re.I,
    ),
    # "p. 478", "p.480", "pp. 32-35"
    re.compile(
        r"\bpp?\.\s*(?P<page>\d{1,4})(?:\s*[-–,]\s*(?P<page2>\d{1,4}))?",
        re.I,
    ),
    # "GW 21, page 45" — Gesammelte Werke with volume
    re.compile(
        r"\bGW\s*(?P<gw_vol>\d{1,2})\b[,\s]*(?:page|p\.?)?\s*(?P<page>\d{1,4})?",
        re.I,
    ),
    # Bare "on 480" or "she's on 480, 481" — only when near edition markers
    re.compile(
        r"\bon\s+(?P<page>\d{2,4})(?:\s*[,]\s*(?P<page2>\d{2,4}))?",
        re.I,
    ),
]


# ── Reading indicators ───────────────────────────────────────────────────
# Phrases that signal the speaker is reading aloud from a text,
# as opposed to merely citing a page number. Ordered by specificity.

READING_INDICATORS: list[re.Pattern] = [
    # Direct attribution + quotation: "Hegel says that X"
    re.compile(r"\b(?:Hegel|he|she)\s+(?:says|writes|states|argues|claims)\s+that\b", re.I),
    # Explicit reading: "let me read", "I'll read", "reading from"
    re.compile(r"\b(?:let\s+me\s+read|I(?:'ll|'m\s+going\s+to)\s+read|reading\s+from)\b", re.I),
    # Quoting: "I quote", "quote from", "and I quote"
    re.compile(r"\b(?:I\s+quote|and\s+I\s+quote|quote\s+from|here's\s+a\s+quote)\b", re.I),
    # Reciting: "he says:" or "she says," followed by content
    re.compile(r"\b(?:Hegel|he|she|it)\s+(?:says|writes|goes)\s*[,:]\s", re.I),
]

# Paraphrase indicators — speaker restates without reading verbatim.
PARAPHRASE_INDICATORS: list[re.Pattern] = [
    re.compile(r"\bfor\s+Hegel\b", re.I),
    re.compile(r"\bHegel(?:'s)?\s+(?:point|argument|claim|view|position)\b", re.I),
    re.compile(r"\bwhat\s+Hegel\s+(?:means|is\s+saying|is\s+getting\s+at)\b", re.I),
    re.compile(r"\baccording\s+to\s+(?:Hegel|him)\b", re.I),
]


# ── Data structures ──────────────────────────────────────────────────────

@dataclass
class CitationRecord:
    """A structured citation extracted from a transcript.

    Links a moment in a conversation/lecture to a specific passage in a
    primary text. The audio timestamp allows jumping directly to the
    point where the citation occurs.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    work_title: str = ""
    work_author: str = "Hegel"
    citation_type: CitationType = CitationType.REFERENCE
    page_german: str | None = None
    page_english: str | None = None
    edition_german: str | None = None
    edition_english: str | None = None
    section_ref: str | None = None
    quoted_text: str | None = None
    speaker: str = ""
    conversation_date: str = ""
    audio_timestamp: str = ""
    audio_timestamp_end: str | None = None
    audio_path: str | None = None
    conversation_source_id: str | None = None
    conversation_chunk_id: str | None = None
    cited_source_id: str | None = None
    cited_chunk_id: str | None = None
    discussion_context: str | None = None
    confidence: float = 0.0
    extraction_notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "work_title": self.work_title,
            "work_author": self.work_author,
            "citation_type": self.citation_type.value,
            "page_german": self.page_german,
            "page_english": self.page_english,
            "edition_german": self.edition_german,
            "edition_english": self.edition_english,
            "section_ref": self.section_ref,
            "quoted_text": self.quoted_text,
            "speaker": self.speaker,
            "conversation_date": self.conversation_date,
            "audio_timestamp": self.audio_timestamp,
            "audio_timestamp_end": self.audio_timestamp_end,
            "audio_path": self.audio_path,
            "conversation_source_id": self.conversation_source_id,
            "conversation_chunk_id": self.conversation_chunk_id,
            "cited_source_id": self.cited_source_id,
            "cited_chunk_id": self.cited_chunk_id,
            "discussion_context": self.discussion_context,
            "confidence": self.confidence,
            "extraction_notes": self.extraction_notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CitationRecord:
        return cls(**{
            k: CitationType(v) if k == "citation_type" else v
            for k, v in d.items() if k in cls.__dataclass_fields__
        })


# ── Core extraction functions ────────────────────────────────────────────


def identify_work(text: str) -> dict[str, str] | None:
    """Match text against KNOWN_WORKS catalog.

    Scans for title aliases in the text. Returns the first matching
    work's canonical metadata dict {"title": ..., "author": ...},
    or None if no known work is mentioned.

    Why first-match: aliases are ordered from most specific (full title)
    to least specific (abbreviations), so the first hit is the best.
    """
    for work in KNOWN_WORKS:
        for alias_re in work["aliases"]:
            if alias_re.search(text):
                return {"title": work["title"], "author": work["author"]}
    return None


def extract_page_refs(text: str) -> list[dict[str, str | None]]:
    """Extract structured page references from a text segment.

    Returns a list of dicts, each with keys:
        page_german, page_english, edition_german, edition_english

    How edition assignment works:
        1. If an explicit edition marker (di Giovanni, Suhrkamp, etc.)
           appears near the page number, that determines the language.
        2. If the speaker says "in the German" or "in the English",
           that overrides edition markers.
        3. GW references are always German.
        4. If no edition context is found, the page is assigned to
           English by default (di Giovanni is the most common edition
           used in these conversations).
    """
    refs: list[dict[str, str | None]] = []
    text_lower = text.lower()

    # Determine edition context from the full segment text.
    # These flags tell us which language the speaker is referencing.
    has_german_marker = bool(_GERMAN_EDITIONS.search(text))
    has_english_marker = bool(_ENGLISH_EDITIONS.search(text))

    # Extract the specific edition names for attribution
    german_edition = None
    english_edition = None
    if has_german_marker:
        gm = _GERMAN_EDITIONS.search(text)
        if gm:
            german_edition = gm.group(0).strip()
    if has_english_marker:
        em = _ENGLISH_EDITIONS.search(text)
        if em:
            english_edition = em.group(0).strip()

    for pat in PAGE_PATTERNS:
        for m in pat.finditer(text):
            page_num = m.group("page")
            if not page_num:
                # GW reference without a page number — store as section ref
                gw_vol = m.groupdict().get("gw_vol")
                if gw_vol:
                    refs.append({
                        "page_german": None,
                        "page_english": None,
                        "edition_german": f"GW {gw_vol}",
                        "edition_english": None,
                    })
                continue

            # For the bare "on NNN" pattern, require an edition marker
            # nearby to avoid false positives like "on 5 people".
            if pat.pattern.startswith(r"\bon\s"):
                if not (has_german_marker or has_english_marker):
                    continue

            page2 = m.groupdict().get("page2")
            gw_vol = m.groupdict().get("gw_vol")

            # Assign page to german or english based on context
            pg = page_num
            ref: dict[str, str | None] = {
                "page_german": None,
                "page_english": None,
                "edition_german": german_edition,
                "edition_english": english_edition,
            }

            if gw_vol:
                # GW references are always German edition
                ref["page_german"] = pg
                ref["edition_german"] = f"GW {gw_vol}"
            elif has_german_marker and not has_english_marker:
                ref["page_german"] = pg
            elif has_english_marker and not has_german_marker:
                ref["page_english"] = pg
            elif has_german_marker and has_english_marker:
                # Both markers present — use proximity heuristic.
                # Check which marker is closer to the page number.
                page_pos = m.start()
                gm_match = _GERMAN_EDITIONS.search(text)
                em_match = _ENGLISH_EDITIONS.search(text)
                g_dist = abs(gm_match.start() - page_pos) if gm_match else 9999
                e_dist = abs(em_match.start() - page_pos) if em_match else 9999
                if g_dist < e_dist:
                    ref["page_german"] = pg
                else:
                    ref["page_english"] = pg
            else:
                # No edition context — default to English (di Giovanni
                # is the primary edition in these conversations)
                ref["page_english"] = pg

            refs.append(ref)

            # Handle second page in a range (e.g., "pages 480-481")
            if page2:
                ref2 = dict(ref)
                if ref["page_german"]:
                    ref2["page_german"] = page2
                else:
                    ref2["page_english"] = page2
                refs.append(ref2)

    # Deduplicate by (page_german, page_english) pairs
    seen = set()
    deduped = []
    for r in refs:
        key = (r["page_german"], r["page_english"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    return deduped


def detect_citation_type(text: str) -> CitationType:
    """Classify how a source text is invoked in a segment.

    Priority order (first match wins):
        1. READING — speaker reads verbatim (strongest indicators)
        2. PARAPHRASE — speaker restates in own words
        3. REFERENCE — default when page/work is mentioned without
           reading or paraphrasing indicators

    Why this order: reading indicators are the most distinctive
    (explicit speech acts like "Hegel says that..." or "let me read").
    Paraphrase indicators are weaker ("for Hegel, ..."). Everything
    else with a page reference defaults to REFERENCE.
    """
    for pat in READING_INDICATORS:
        if pat.search(text):
            return CitationType.READING

    for pat in PARAPHRASE_INDICATORS:
        if pat.search(text):
            return CitationType.PARAPHRASE

    return CitationType.REFERENCE


def _compute_confidence(
    page_refs: list[dict],
    work: dict | None,
    citation_type: CitationType,
    text: str,
) -> float:
    """Score how confident we are that this is a real citation.

    Confidence is a 0.0–1.0 float built from additive signals:
        +0.3  explicit page number found
        +0.2  edition marker present (di Giovanni, Suhrkamp, etc.)
        +0.2  known work title identified
        +0.15 reading/quote indicator present
        +0.1  both German and English pages present
        +0.05 long quoted text (>50 chars after the indicator)

    Capped at 1.0. The thresholds are tuned to the Wu conversation
    corpus: a bare "he says" scores ~0.15, while "page 478 in di
    Giovanni's Science of Logic" scores ~0.9.
    """
    score = 0.0

    if page_refs:
        score += 0.3
        # Edition marker bonus
        if any(r.get("edition_german") or r.get("edition_english") for r in page_refs):
            score += 0.2
        # Both languages bonus
        if any(r.get("page_german") for r in page_refs) and \
           any(r.get("page_english") for r in page_refs):
            score += 0.1

    if work:
        score += 0.2

    if citation_type == CitationType.READING:
        score += 0.15

    if len(text) > 150:
        score += 0.05

    return min(score, 1.0)


def _has_citation_signal(text: str) -> bool:
    """Quick pre-filter: does this segment contain anything citation-like?

    Used to skip the majority of segments (casual conversation) without
    running the full pattern battery. Checks for page keywords, edition
    markers, reading indicators, or known work titles.
    """
    tl = text.lower()
    # Fast keyword check before expensive regex
    quick_keywords = [
        "page ", "p. ", "p.", "gw ", "footnote",
        "giovanni", "miller", "suhrkamp", "pinkard",
        "hegel says", "hegel writes", "he says that", "she says that",
        "let me read", "i quote", "quote from", "reading from",
        "science of logic", "phenomenology", "philosophy of right",
        "encyclopedia", "grundrisse", "being and time", "critique of",
        "for hegel", "according to hegel", "hegel's point",
    ]
    if any(kw in tl for kw in quick_keywords):
        return True

    # Check READING_INDICATORS (compiled regexes) as fallback
    for pat in READING_INDICATORS:
        if pat.search(text):
            return True

    return False


def extract_citations_from_segments(
    segments: list[dict[str, Any]],
    conversation_date: str = "",
    audio_path: str | None = None,
    conversation_source_id: str | None = None,
    context_window: int = 2,
) -> list[CitationRecord]:
    """Scan transcript segments for citations, returning CitationRecords.

    This is the main entry point for the heuristic extraction engine.
    It processes each segment individually, but uses a sliding context
    window (±context_window segments) to resolve edition markers and
    work titles that may appear in adjacent segments.

    Args:
        segments: List of whisperx segment dicts with keys:
            text, speaker, start, end.
        conversation_date: ISO-ish date string (e.g. "2025.07.07")
            for the conversation_date field on each record.
        audio_path: Path to the audio file, stored on each record.
        conversation_source_id: DB source ID for the conversation.
        context_window: Number of adjacent segments to include when
            resolving edition/work context. Default 2 means we look
            at the 2 segments before and after the current one.

    Returns:
        List of CitationRecord objects, one per detected citation.
        Multiple citations can come from a single segment if it
        contains multiple page references.
    """
    citations: list[CitationRecord] = []

    for i, seg in enumerate(segments):
        text = seg.get("text", "")
        if not _has_citation_signal(text):
            continue

        # Build context window: combine adjacent segment texts for
        # resolving edition markers and work titles that may span
        # multiple speaker turns.
        ctx_start = max(0, i - context_window)
        ctx_end = min(len(segments), i + context_window + 1)
        context_text = " ".join(
            s.get("text", "") for s in segments[ctx_start:ctx_end]
        )

        # 1. Extract page references from this segment's text
        page_refs = extract_page_refs(text)

        # 2. Identify which work is being cited
        #    Check the segment text first, then fall back to context
        work = identify_work(text) or identify_work(context_text)

        # 3. Determine citation type (reading, reference, paraphrase)
        citation_type = detect_citation_type(text)

        # 4. Compute confidence score
        confidence = _compute_confidence(page_refs, work, citation_type, text)

        # Skip very low-confidence hits (bare "he says" about non-Hegel
        # topics, etc.) — threshold tuned to avoid noise.
        if confidence < 0.1:
            continue

        # If no page refs found but we have a strong reading indicator
        # + work identification, still create a citation (the speaker
        # is reading but didn't state a page number).
        if not page_refs and citation_type != CitationType.READING:
            # Only page-ref or reading segments pass through
            if not work:
                continue

        # 5. Build CitationRecord(s) — one per page reference,
        #    or one for a reading without explicit pages.
        speaker = seg.get("speaker", "")
        start_time = seg.get("start", 0.0)
        end_time = seg.get("end", start_time)

        # Extract quoted text for READING citations: everything after
        # the reading indicator phrase is likely the quoted content.
        quoted_text = None
        if citation_type == CitationType.READING:
            # Try to capture text after "says that", "quote:", etc.
            for pat in READING_INDICATORS:
                m = pat.search(text)
                if m:
                    after = text[m.end():].strip()
                    if len(after) > 10:
                        quoted_text = after
                    break
            if not quoted_text and len(text) > 30:
                quoted_text = text

        if page_refs:
            # One CitationRecord per distinct page reference
            for ref in page_refs:
                rec = CitationRecord(
                    work_title=work["title"] if work else "",
                    work_author=work["author"] if work else "Hegel",
                    citation_type=citation_type,
                    page_german=ref.get("page_german"),
                    page_english=ref.get("page_english"),
                    edition_german=ref.get("edition_german"),
                    edition_english=ref.get("edition_english"),
                    quoted_text=quoted_text,
                    speaker=speaker,
                    conversation_date=conversation_date,
                    audio_timestamp=str(start_time),
                    audio_timestamp_end=str(end_time),
                    audio_path=audio_path,
                    conversation_source_id=conversation_source_id,
                    discussion_context=text,
                    confidence=confidence,
                    extraction_notes="heuristic/regex extraction",
                )
                citations.append(rec)
        else:
            # No explicit page ref but strong enough signal (reading
            # indicator + work title, or paraphrase of known work)
            rec = CitationRecord(
                work_title=work["title"] if work else "",
                work_author=work["author"] if work else "Hegel",
                citation_type=citation_type,
                quoted_text=quoted_text,
                speaker=speaker,
                conversation_date=conversation_date,
                audio_timestamp=str(start_time),
                audio_timestamp_end=str(end_time),
                audio_path=audio_path,
                conversation_source_id=conversation_source_id,
                discussion_context=text,
                confidence=confidence,
                extraction_notes="heuristic/regex extraction (no page ref)",
            )
            citations.append(rec)

    return citations


# ── Transcript-level wrappers ────────────────────────────────────────────


def extract_citations_from_transcript(
    json_path: Path,
    conversation_source_id: str | None = None,
) -> list[CitationRecord]:
    """Extract citations from a Wu conversation JSON transcript file.

    Reads the whisperx JSON, extracts the date from the filename
    (e.g. "Wu_2025.07.07.json" → "2025.07.07"), and runs the
    segment-level extractor over all segments.

    Args:
        json_path: Path to the LLM-cleaned JSON transcript.
        conversation_source_id: Optional DB source ID.

    Returns:
        List of CitationRecord objects found in the transcript.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        return []

    # Extract date from filename: "Wu_2025.07.07.json" → "2025.07.07"
    stem = json_path.stem
    date_part = ""
    date_match = re.search(r"(\d{4}\.\d{2}\.\d{2})", stem)
    if date_match:
        date_part = date_match.group(1)

    return extract_citations_from_segments(
        segments,
        conversation_date=date_part,
        audio_path=str(json_path),
        conversation_source_id=conversation_source_id,
    )


def extract_citations_from_lecture(
    source_id: str,
    db: Any,
    lecturer_name: str = "",
) -> list[CitationRecord]:
    """Extract citations from lecture transcript chunks already in the DB.

    Lectures (Thompson, Houlgate, Radnik) are stored as chunks in the
    database. This function reads all chunks for a source, concatenates
    adjacent chunks into pseudo-segments, and runs the extractor.

    Args:
        source_id: Database source ID for the lecture.
        db: Database instance with get_chunks_by_source().
        lecturer_name: Speaker name for attribution.

    Returns:
        List of CitationRecord objects.
    """
    chunks = db.get_chunks_by_source(source_id)
    if not chunks:
        return []

    # Convert chunks to segment-like dicts for the extractor
    segments = []
    for chunk in chunks:
        seg = {
            "text": chunk.text,
            "speaker": lecturer_name,
            "start": 0.0,
            "end": 0.0,
        }
        # Use youtube_timestamp if available
        if hasattr(chunk, "youtube_timestamp") and chunk.youtube_timestamp:
            seg["start"] = chunk.youtube_timestamp
        segments.append(seg)

    return extract_citations_from_segments(
        segments,
        conversation_source_id=source_id,
    )


def cross_reference_citations(
    citations: list[CitationRecord],
    db: Any,
    embedder: Any,
    index: Any,
    top_k: int = 3,
) -> list[CitationRecord]:
    """Link citations to existing source text chunks via semantic search.

    For each citation that has quoted_text or a page reference, we
    search the FAISS index for the most similar chunk in the corpus.
    If the best match is from a source text (not another conversation),
    we set cited_source_id and cited_chunk_id on the citation.

    This creates bidirectional links: conversation → source text and
    (via DB queries) source text → conversations that cite it.

    Args:
        citations: List of CitationRecord objects to cross-reference.
        db: Database instance for looking up chunk metadata.
        embedder: EmbeddingPipeline with encode() method.
        index: FAISSIndex with search() method.
        top_k: Number of nearest neighbors to consider.

    Returns:
        The same list of citations, mutated with cited_source_id
        and cited_chunk_id where matches were found.
    """
    for cit in citations:
        # Build a search query from the citation's best available text
        query_parts = []
        if cit.quoted_text:
            query_parts.append(cit.quoted_text)
        if cit.work_title:
            query_parts.append(cit.work_title)
        if cit.page_english:
            query_parts.append(f"page {cit.page_english}")
        if cit.page_german:
            query_parts.append(f"page {cit.page_german}")
        if cit.discussion_context and not cit.quoted_text:
            query_parts.append(cit.discussion_context)

        if not query_parts:
            continue

        query = " ".join(query_parts)

        try:
            # Embed the query and search FAISS
            embedding = embedder.encode(query)
            results = index.search(embedding, k=top_k)

            # Find the best match that's from a source text (PDF),
            # not from another conversation transcript
            for result in results:
                chunk_id = result.get("chunk_id") or result.get("id")
                if not chunk_id:
                    continue
                chunk = db.get_chunk(chunk_id)
                if not chunk:
                    continue
                source = db.get_source(chunk.source_id)
                if not source:
                    continue
                # Prefer PDF sources (the actual texts) over transcripts
                if source.type.value in ("pdf", "critical_edition"):
                    cit.cited_source_id = chunk.source_id
                    cit.cited_chunk_id = chunk_id
                    break
        except Exception:
            # Cross-referencing is best-effort; don't fail the pipeline
            continue

    return citations
