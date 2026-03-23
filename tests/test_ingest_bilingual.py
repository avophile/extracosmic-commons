"""Tests for bilingual JSON ingestion."""

import json

import pytest

from extracosmic_commons.ingest.bilingual import BilingualIngester
from extracosmic_commons.models import SourceType

SAMPLE_BILINGUAL_JSON = {
    "metadata": {
        "title_de": "Erster Abschnitt. Bestimmtheit. (Qualität.)",
        "title_en": "Section I. Determinateness (Quality)",
        "gw_range": "21.68–172",
        "translator": "di Giovanni",
        "edition": "Cambridge UP, 2010",
    },
    "de_paragraphs": [
        {"type": "heading", "text": "Erstes Kapitel. Sein", "gw21_page": 68, "level": 2},
        {"type": "body", "text": "Sein, reines Sein – ohne alle weitere Bestimmung.", "gw21_page": 68},
        {"type": "body", "text": "Es ist das reine Unbestimmte und Leere.", "gw21_page": 69},
    ],
    "en_paragraphs": [
        {"type": "heading", "text": "Chapter 1. Being"},
        {"type": "body", "text": "Being, pure being – without further determination."},
        {"type": "body", "text": "It is the pure indeterminate and empty."},
    ],
    "en_footnotes": {"1": "This is a footnote."},
    "fn_para_map": {"1": ["1"]},
}


@pytest.fixture
def sample_bilingual_json(tmp_path):
    p = tmp_path / "clean_text.json"
    p.write_text(json.dumps(SAMPLE_BILINGUAL_JSON))
    return p


class TestBilingualIngester:
    def test_parse_creates_two_sources(self, sample_bilingual_json):
        ingester = BilingualIngester()
        de_src, en_src, chunks = ingester.parse_bilingual_json(sample_bilingual_json)

        assert de_src.type == SourceType.BILINGUAL_PAIR
        assert en_src.type == SourceType.BILINGUAL_PAIR
        assert de_src.language == ["de"]
        assert en_src.language == ["en"]

    def test_parse_creates_paired_chunks(self, sample_bilingual_json):
        ingester = BilingualIngester()
        de_src, en_src, chunks = ingester.parse_bilingual_json(sample_bilingual_json)

        de_chunks = [c for c in chunks if c.language == "de"]
        en_chunks = [c for c in chunks if c.language == "en"]

        assert len(de_chunks) == 3
        assert len(en_chunks) == 3

        # Paired IDs should cross-reference
        assert de_chunks[0].paired_chunk_id == en_chunks[0].id
        assert en_chunks[0].paired_chunk_id == de_chunks[0].id

    def test_german_chunks_have_gw_page(self, sample_bilingual_json):
        ingester = BilingualIngester()
        _, _, chunks = ingester.parse_bilingual_json(sample_bilingual_json)

        de_body = [c for c in chunks if c.language == "de" and "Sein, reines" in c.text]
        assert len(de_body) == 1
        assert de_body[0].structural_ref is not None
        assert de_body[0].structural_ref["gw_page"] == "21.68"

    def test_headings_have_structural_ref(self, sample_bilingual_json):
        ingester = BilingualIngester()
        _, _, chunks = ingester.parse_bilingual_json(sample_bilingual_json)

        de_heading = [c for c in chunks if c.language == "de" and "Kapitel" in c.text]
        assert len(de_heading) == 1
        assert de_heading[0].structural_ref is not None
        assert de_heading[0].structural_ref["level"] == 2

    def test_chunk_method(self, sample_bilingual_json):
        ingester = BilingualIngester()
        _, _, chunks = ingester.parse_bilingual_json(sample_bilingual_json)
        assert all(c.chunk_method == "bilingual_alignment" for c in chunks)

    def test_metadata_preserved(self, sample_bilingual_json):
        ingester = BilingualIngester()
        de_src, en_src, _ = ingester.parse_bilingual_json(sample_bilingual_json)

        assert "di Giovanni" in en_src.author
        assert "21.68" in de_src.edition
