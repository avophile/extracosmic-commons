"""Tests for the regex/heuristic citation extraction engine.

TDD: These tests define the expected behavior of the pattern-matching
extractor before implementation. Each test covers a distinct pattern
type observed in real Wu conversation transcripts.
"""

import json
import pytest
from pathlib import Path

from extracosmic_commons.ingest.citation_extractor import (
    CitationRecord,
    CitationType,
    KNOWN_WORKS,
    PAGE_PATTERNS,
    READING_INDICATORS,
    extract_citations_from_segments,
    identify_work,
    extract_page_refs,
    detect_citation_type,
)


# ── Fixtures: synthetic transcript segments ──────────────────────────────

def _seg(text, speaker="SPEAKER_01", start=100.0, end=110.0):
    """Helper to build a minimal segment dict."""
    return {"text": text, "speaker": speaker, "start": start, "end": end}


# ── Test: Page reference extraction ──────────────────────────────────────

class TestExtractPageRefs:
    """extract_page_refs() should pull structured page numbers from text."""

    def test_di_giovanni_page(self):
        """'page 478' near 'Giovanni' → english page with di Giovanni edition."""
        refs = extract_page_refs(
            "There's a footnote there from Giovanni which says here on page 478"
        )
        assert any(r["page_english"] == "478" for r in refs)
        assert any("Giovanni" in (r.get("edition_english") or "") for r in refs)

    def test_german_page_explicit(self):
        """'In the German, it's page 110' → german page."""
        refs = extract_page_refs(
            "In the German, it's page 110."
        )
        assert any(r["page_german"] == "110" for r in refs)

    def test_english_page_explicit(self):
        """'page 32 in your translation' → english page."""
        refs = extract_page_refs(
            "I'm in your translation, page 32."
        )
        assert any(r["page_english"] == "32" for r in refs)

    def test_gw_reference(self):
        """'GW 21' → Gesammelte Werke volume reference."""
        refs = extract_page_refs("GW 21, page 45")
        assert any(r.get("page_german") == "45" for r in refs)
        assert any("GW 21" in (r.get("edition_german") or "") for r in refs)

    def test_p_dot_abbreviation(self):
        """'p. 480' should parse as a page reference."""
        refs = extract_page_refs("she's on p. 480 in Di Giovanni")
        assert any(r["page_english"] == "480" for r in refs)

    def test_page_range(self):
        """'pages 480-481' or '480, 481' should capture the range."""
        refs = extract_page_refs("So she's on 480, 481 in Di Giovanni.")
        assert len(refs) >= 1
        # Should capture at least one page number
        pages = [r.get("page_english") for r in refs]
        assert "480" in pages or "481" in pages

    def test_suhrkamp_edition(self):
        """'Suhrkamp page 234' → German edition page."""
        refs = extract_page_refs("Suhrkamp page 234")
        assert any(r["page_german"] == "234" for r in refs)
        assert any("Suhrkamp" in (r.get("edition_german") or "") for r in refs)

    def test_miller_translation(self):
        """'Miller translation, page 59' → English page, Miller edition."""
        refs = extract_page_refs("In the Miller translation, page 59")
        assert any(r["page_english"] == "59" for r in refs)
        assert any("Miller" in (r.get("edition_english") or "") for r in refs)

    def test_no_false_positives(self):
        """Ordinary numbers shouldn't be mistaken for page refs."""
        refs = extract_page_refs("We had about 5 people in the room")
        assert len(refs) == 0


# ── Test: Work identification ────────────────────────────────────────────

class TestIdentifyWork:
    """identify_work() should match text against KNOWN_WORKS catalog."""

    def test_science_of_logic(self):
        work = identify_work("This is from the Science of Logic, page 478.")
        assert work is not None
        assert "Logic" in work["title"]
        assert work["author"] == "Hegel"

    def test_phenomenology_of_spirit(self):
        work = identify_work("In the Phenomenology of Spirit, paragraph 32")
        assert work is not None
        assert "Phenomenology" in work["title"]

    def test_philosophy_of_right(self):
        work = identify_work("The Philosophy of Right says something different")
        assert work is not None
        assert "Right" in work["title"]

    def test_encyclopedia(self):
        work = identify_work("In the Encyclopedia Logic he makes this point")
        assert work is not None

    def test_grundrisse(self):
        work = identify_work("passage from the Grundrisse")
        assert work is not None
        assert work["author"] == "Marx"

    def test_being_and_time(self):
        work = identify_work("in Being and Time, right, this being-for-itself")
        assert work is not None
        assert work["author"] == "Heidegger"

    def test_critique_of_pure_reason(self):
        work = identify_work("Kant argues in the Critique of Pure Reason")
        assert work is not None
        assert work["author"] == "Kant"

    def test_no_match(self):
        """Ordinary text with no work title should return None."""
        work = identify_work("I had a great time at the barbecue last weekend")
        assert work is None

    def test_abbreviation_sol(self):
        """'the Logic' or 'SoL' as shorthand for Science of Logic."""
        work = identify_work("In the Logic, this is the key move")
        assert work is not None
        assert "Logic" in work["title"]


# ── Test: Citation type detection ────────────────────────────────────────

class TestDetectCitationType:
    """detect_citation_type() classifies how the source is invoked."""

    def test_reading_hegel_says(self):
        """'Hegel says ...' followed by quoted-sounding text → READING."""
        ct = detect_citation_type(
            "Hegel says that one soul must bathe in this ether of the one substance"
        )
        assert ct == CitationType.READING

    def test_reading_let_me_read(self):
        """'Let me read this to you' → READING."""
        ct = detect_citation_type("Let me read this to you some years ago.")
        assert ct == CitationType.READING

    def test_reading_i_quote(self):
        """'I quote' → READING."""
        ct = detect_citation_type("this is too long, I quote.")
        assert ct == CitationType.READING

    def test_reference_page_only(self):
        """Page reference without reading indicator → REFERENCE."""
        ct = detect_citation_type("What page are you on in Giovanni?")
        assert ct == CitationType.REFERENCE

    def test_reference_footnote(self):
        """'footnote on page X' → REFERENCE."""
        ct = detect_citation_type(
            "There's a footnote there from Giovanni which says here on page 478"
        )
        # This has "says" but in the context of a footnote attribution,
        # which is more of a reference than a reading.
        assert ct in (CitationType.REFERENCE, CitationType.READING)

    def test_paraphrase_for_hegel(self):
        """'For Hegel, X is Y' → PARAPHRASE (restating, not reading)."""
        ct = detect_citation_type(
            "For Hegel, the absolute idea is the unity of the theoretical and practical idea"
        )
        assert ct == CitationType.PARAPHRASE


# ── Test: Full segment extraction pipeline ───────────────────────────────

class TestExtractCitationsFromSegments:
    """extract_citations_from_segments() processes a list of segments
    and returns CitationRecord objects for segments containing citations."""

    def test_explicit_page_reference(self):
        """A segment with 'page X in Di Giovanni' should yield a citation."""
        segs = [
            _seg("There's a footnote from Giovanni on page 478, "
                 "Zufälligkeit, later Hegel used Accidentalität.",
                 speaker="SPEAKER_01", start=4827.5, end=4870.0),
        ]
        citations = extract_citations_from_segments(
            segs, conversation_date="2025.07.07"
        )
        assert len(citations) >= 1
        c = citations[0]
        assert c.page_english == "478"
        assert c.work_author == "Hegel"
        assert c.conversation_date == "2025.07.07"
        assert c.audio_timestamp == "4827.5"
        assert c.confidence > 0

    def test_german_page_reference(self):
        """'In the German, it's page 110' should produce a german page."""
        segs = [
            _seg("In the German, it's page 110.", start=2581.0, end=2590.0),
        ]
        citations = extract_citations_from_segments(
            segs, conversation_date="2025.06.23"
        )
        assert len(citations) >= 1
        assert citations[0].page_german == "110"

    def test_reading_with_attribution(self):
        """'Hegel says that X' should produce a READING citation."""
        segs = [
            _seg("Hegel says that one soul must bathe in this ether of "
                 "the one substance in which everything one has held as "
                 "true comes to naught",
                 speaker="SPEAKER_02", start=105.7, end=120.0),
        ]
        citations = extract_citations_from_segments(
            segs, conversation_date="2025.07.12"
        )
        assert len(citations) >= 1
        c = citations[0]
        assert c.citation_type == CitationType.READING
        assert c.speaker == "SPEAKER_02"
        assert "bathe" in (c.quoted_text or "")

    def test_no_citation_segments_ignored(self):
        """Ordinary conversation segments should produce no citations."""
        segs = [
            _seg("Hey, how's it going?"),
            _seg("Yeah, I had a good week."),
            _seg("Let's get into it then."),
        ]
        citations = extract_citations_from_segments(segs)
        assert len(citations) == 0

    def test_adjacent_segments_merged(self):
        """A page ref in one segment followed by reading in the next
        should be linkable (both captured, context window covers both)."""
        segs = [
            _seg("What page is that?", speaker="SPEAKER_04",
                 start=4607.8, end=4610.0),
            _seg("She's on page 480 in Di Giovanni.",
                 speaker="SPEAKER_01", start=4870.0, end=4875.0),
            _seg("So the one and the many ones has one type of relationship.",
                 speaker="SPEAKER_01", start=5064.0, end=5070.0),
        ]
        citations = extract_citations_from_segments(
            segs, conversation_date="2025.07.07"
        )
        # At least one citation for the page 480 ref
        assert any(c.page_english == "480" for c in citations)

    def test_multiple_citations_in_conversation(self):
        """Multiple distinct page refs in a transcript yield multiple records."""
        segs = [
            _seg("page 478 in Giovanni", start=100.0),
            _seg("some discussion about being", start=200.0),
            _seg("she's on 480, 481 in Di Giovanni", start=300.0),
            _seg("In the German it's page 110", start=400.0),
        ]
        citations = extract_citations_from_segments(segs)
        assert len(citations) >= 3

    def test_work_title_in_context(self):
        """When a work title appears near a citation, it should be captured."""
        segs = [
            _seg("In the Science of Logic, page 478 in di Giovanni",
                 start=100.0),
        ]
        citations = extract_citations_from_segments(segs)
        assert len(citations) >= 1
        assert "Logic" in citations[0].work_title

    def test_confidence_higher_for_explicit_refs(self):
        """Explicit page + edition should have higher confidence than
        a vague 'he says' without page numbers."""
        segs_explicit = [
            _seg("page 478 in di Giovanni translation", start=100.0),
        ]
        segs_vague = [
            _seg("he says that being is nothing", start=200.0),
        ]
        c_explicit = extract_citations_from_segments(segs_explicit)
        c_vague = extract_citations_from_segments(segs_vague)
        if c_explicit and c_vague:
            assert c_explicit[0].confidence > c_vague[0].confidence
