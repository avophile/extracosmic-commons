"""Tests for critical edition PDF ingester."""

import pytest

from extracosmic_commons.edition_profiles import HEGEL_SOL_DI_GIOVANNI
from extracosmic_commons.ingest.critical_edition import (
    CriticalEditionIngester,
    classify_line_by_font,
    extract_margin_refs,
    extract_page_text,
)
from extracosmic_commons.search import _diversify_results, SearchResult
from extracosmic_commons.models import Chunk, Source, SourceType


class TestCharacterExtraction:
    """Test the character-level text extraction helpers."""

    def test_build_line_from_chars(self):
        from extracosmic_commons.ingest.critical_edition import _build_line_from_chars

        chars = [
            {"text": "H", "x0": 0, "x1": 8},
            {"text": "e", "x0": 8, "x1": 14},
            {"text": "l", "x0": 14, "x1": 18},
            {"text": "l", "x0": 18, "x1": 22},
            {"text": "o", "x0": 22, "x1": 28},
            {"text": "W", "x0": 32, "x1": 40},  # 4pt gap → space
            {"text": "o", "x0": 40, "x1": 46},
            {"text": "r", "x0": 46, "x1": 50},
            {"text": "l", "x0": 50, "x1": 54},
            {"text": "d", "x0": 54, "x1": 60},
        ]
        result = _build_line_from_chars(chars, space_threshold=1.5)
        assert result == "Hello World"


class TestFontClassification:
    def test_small_caps_is_heading(self):
        chars = [
            {"text": "S", "fontname": "TimesNewRomanSC", "size": 11},
            {"text": "E", "fontname": "TimesNewRomanSC", "size": 11},
            {"text": "C", "fontname": "TimesNewRomanSC", "size": 11},
        ]
        assert classify_line_by_font(chars) == "heading"

    def test_large_italic_is_heading(self):
        chars = [
            {"text": "C", "fontname": "TimesItalic", "size": 14},
            {"text": "h", "fontname": "TimesItalic", "size": 14},
        ]
        assert classify_line_by_font(chars) == "heading"

    def test_bold_is_body(self):
        """Bold lines are typically GW margin refs, not headings."""
        chars = [
            {"text": "2", "fontname": "TimesBold", "size": 11},
            {"text": "1", "fontname": "TimesBold", "size": 11},
            {"text": ".", "fontname": "TimesBold", "size": 11},
            {"text": "6", "fontname": "TimesBold", "size": 11},
            {"text": "8", "fontname": "TimesBold", "size": 11},
        ]
        assert classify_line_by_font(chars) == "body"

    def test_regular_body_text(self):
        chars = [
            {"text": "T", "fontname": "TimesNewRoman", "size": 11},
            {"text": "h", "fontname": "TimesNewRoman", "size": 11},
            {"text": "e", "fontname": "TimesNewRoman", "size": 11},
        ]
        assert classify_line_by_font(chars) == "body"

    def test_empty_is_body(self):
        assert classify_line_by_font([]) == "body"


class TestMarginRefExtraction:
    def test_gw_ref_extraction(self):
        refs = extract_margin_refs(
            "Some text 21.364 more text 21.365",
            [],
            HEGEL_SOL_DI_GIOVANNI,
        )
        assert "21.364" in refs
        assert "21.365" in refs

    def test_no_refs(self):
        refs = extract_margin_refs(
            "No references here",
            [],
            HEGEL_SOL_DI_GIOVANNI,
        )
        assert refs == []


class TestCriticalEditionIngester:
    def test_page_offset(self):
        import copy

        profile = copy.copy(HEGEL_SOL_DI_GIOVANNI)
        profile.page_offset = 281
        ingester = CriticalEditionIngester(profile)
        assert ingester._compute_translation_page(37) == 318


class TestSourceDiversity:
    def _make_result(self, source_type, source_id, score, chunk_id="c"):
        source = Source(title=f"Source {source_id}", type=source_type, id=source_id)
        chunk = Chunk(source_id=source_id, text="text", id=f"{chunk_id}-{source_id}-{score}")
        return SearchResult(chunk=chunk, score=score, source=source)

    def test_diversify_includes_all_types(self):
        results = [
            self._make_result(SourceType.PDF, "pdf1", 0.95, "c1"),
            self._make_result(SourceType.PDF, "pdf1", 0.93, "c2"),
            self._make_result(SourceType.PDF, "pdf1", 0.91, "c3"),
            self._make_result(SourceType.PDF, "pdf2", 0.89, "c4"),
            self._make_result(SourceType.TRANSCRIPT, "t1", 0.80, "c5"),
            self._make_result(SourceType.TRANSCRIPT, "t2", 0.75, "c6"),
            self._make_result(SourceType.BILINGUAL_PAIR, "b1", 0.70, "c7"),
        ]
        diversified = _diversify_results(results, top_k=5)
        types = {r.source.type for r in diversified}
        assert SourceType.TRANSCRIPT in types
        assert SourceType.PDF in types
        assert SourceType.BILINGUAL_PAIR in types

    def test_diversify_preserves_score_order_within_type(self):
        results = [
            self._make_result(SourceType.PDF, "pdf1", 0.95, "c1"),
            self._make_result(SourceType.PDF, "pdf2", 0.85, "c2"),
            self._make_result(SourceType.TRANSCRIPT, "t1", 0.80, "c3"),
        ]
        diversified = _diversify_results(results, top_k=3)
        pdf_results = [r for r in diversified if r.source.type == SourceType.PDF]
        if len(pdf_results) >= 2:
            assert pdf_results[0].score >= pdf_results[1].score

    def test_diversify_no_change_when_few_results(self):
        results = [
            self._make_result(SourceType.PDF, "pdf1", 0.95, "c1"),
            self._make_result(SourceType.TRANSCRIPT, "t1", 0.80, "c2"),
        ]
        diversified = _diversify_results(results, top_k=5)
        assert len(diversified) == 2
