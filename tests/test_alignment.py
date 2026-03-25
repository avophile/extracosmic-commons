"""Tests for cross-translation alignment engine."""

import pytest

from extracosmic_commons.alignment import (
    _normalize_heading,
    _heading_to_canonical_id,
    build_canonical_sections,
    align_editions,
    align_work,
)
from extracosmic_commons.database import Database
from extracosmic_commons.models import Chunk, Source, SourceType


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    yield d
    d.close()


class TestHeadingNormalization:
    def test_strip_chapter_prefix(self):
        assert _normalize_heading("Chapter 1. Being") == "being"

    def test_strip_section_prefix(self):
        assert _normalize_heading("Section I: Quality") == "quality"

    def test_strip_letter_label(self):
        assert _normalize_heading("a. Unity of Being and Nothing") == "unity of being and nothing"

    def test_case_insensitive(self):
        assert _normalize_heading("THE DOCTRINE OF BEING") == "the doctrine of being"

    def test_german_prefix(self):
        assert _normalize_heading("Kapitel 1: Sein") == "sein"

    def test_plain_heading(self):
        assert _normalize_heading("Being-for-self") == "being-for-self"


class TestCanonicalIdGeneration:
    def test_basic_id(self):
        cid = _heading_to_canonical_id("hegel-sol", "Being", {})
        assert "hegel-sol" in cid
        assert "being" in cid

    def test_id_with_context(self):
        cid = _heading_to_canonical_id("hegel-sol", "Quality", {"doctrine": "Being"})
        assert "being" in cid
        assert "quality" in cid

    def test_id_deterministic(self):
        a = _heading_to_canonical_id("w", "Heading", {"doctrine": "X"})
        b = _heading_to_canonical_id("w", "Heading", {"doctrine": "X"})
        assert a == b


class TestCanonicalSections:
    def test_build_assigns_sections(self, db):
        source = Source(
            title="Test Edition",
            type=SourceType.PDF,
            metadata={"work_id": "test-work"},
        )
        db.insert_source(source)

        chunks = [
            Chunk(
                source_id=source.id,
                text="Introduction to the work",
                structural_ref={"heading": "Introduction"},
                paragraph_index=0,
                pdf_page=1,
            ),
            Chunk(
                source_id=source.id,
                text="Body text under introduction",
                paragraph_index=1,
                pdf_page=2,
            ),
            Chunk(
                source_id=source.id,
                text="Chapter 1 heading",
                structural_ref={"heading": "Chapter 1: Being"},
                paragraph_index=2,
                pdf_page=5,
            ),
            Chunk(
                source_id=source.id,
                text="Body under chapter 1",
                paragraph_index=3,
                pdf_page=6,
            ),
        ]
        db.insert_chunks_batch(chunks)

        updated = build_canonical_sections("test-work", db)
        assert updated > 0

        # Check that chunks got canonical_section
        tagged = db.get_chunks_by_source(source.id)
        intro_chunk = next(c for c in tagged if c.paragraph_index == 0)
        assert intro_chunk.structural_ref is not None
        assert "canonical_section" in intro_chunk.structural_ref


class TestEditionAlignment:
    def test_align_by_gw_page(self, db):
        """Chunks from different editions with same gw_page get aligned."""
        src_de = Source(
            title="GW 21",
            type=SourceType.PDF,
            metadata={"work_id": "test-sol", "is_original_language": True},
        )
        src_en = Source(
            title="di Giovanni",
            type=SourceType.PDF,
            metadata={"work_id": "test-sol", "is_original_language": False},
        )
        db.insert_source(src_de)
        db.insert_source(src_en)

        de_chunk = Chunk(
            source_id=src_de.id,
            text="Sein, reines Sein",
            language="de",
            structural_ref={"gw_page": "21.68"},
            paragraph_index=0,
        )
        en_chunk = Chunk(
            source_id=src_en.id,
            text="Being, pure being",
            language="en",
            structural_ref={"gw_page": "21.68"},
            paragraph_index=0,
        )
        db.insert_chunks_batch([de_chunk, en_chunk])

        aligned = align_editions("test-sol", db)
        assert aligned >= 2

        # Both should now have canonical_para
        de_updated = db.get_chunks_by_ids([de_chunk.id])[0]
        en_updated = db.get_chunks_by_ids([en_chunk.id])[0]
        assert de_updated.structural_ref.get("canonical_para") is not None
        assert en_updated.structural_ref.get("canonical_para") == de_updated.structural_ref["canonical_para"]

    def test_align_single_source_noop(self, db):
        """Work with only one source has nothing to align."""
        source = Source(
            title="Only One",
            type=SourceType.PDF,
            metadata={"work_id": "solo-work"},
        )
        db.insert_source(source)
        db.insert_chunk(Chunk(source_id=source.id, text="text"))

        aligned = align_editions("solo-work", db)
        assert aligned == 0


class TestAlignWork:
    def test_full_pipeline(self, db):
        src = Source(
            title="Edition A",
            type=SourceType.PDF,
            metadata={"work_id": "full-test", "is_original_language": True},
        )
        db.insert_source(src)
        db.insert_chunk(Chunk(
            source_id=src.id,
            text="Heading text",
            structural_ref={"heading": "Introduction", "gw_page": "21.1"},
            paragraph_index=0,
            pdf_page=1,
        ))

        stats = align_work("full-test", db)
        assert stats["canonical_sections_assigned"] >= 0
        assert stats["paragraphs_aligned"] >= 0
