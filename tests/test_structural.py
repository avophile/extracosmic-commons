"""Tests for structural tagger."""

import pytest

from extracosmic_commons.database import Database
from extracosmic_commons.models import Chunk, Source, SourceType
from extracosmic_commons.structural import StructuralTagger, detect_headings


class TestHeadingDetection:
    def test_hegel_section(self):
        matches = detect_headings("§ 132 is where Hegel discusses Aufheben")
        refs = {m.key: m.value for m in matches}
        assert refs.get("section") == "§132"

    def test_hegel_section_no_space(self):
        matches = detect_headings("In §132, Hegel argues that...")
        refs = {m.key: m.value for m in matches}
        assert refs.get("section") == "§132"

    def test_hegel_doctrine(self):
        matches = detect_headings("The Doctrine of Being is the first part")
        refs = {m.key: m.value for m in matches}
        assert refs.get("doctrine") == "Being"

    def test_hegel_gw_page(self):
        matches = detect_headings("See GW 21.97 for the passage")
        refs = {m.key: m.value for m in matches}
        assert refs.get("gw_page") == "21.97"

    def test_chapter_heading(self):
        matches = detect_headings("Chapter 3: The Dialectic of Quantity")
        refs = {m.key: m.value for m in matches}
        assert "chapter" in refs
        assert "3" in refs["chapter"]
        assert "Quantity" in refs["chapter"]

    def test_chapter_roman(self):
        matches = detect_headings("CHAPTER IV — Measure")
        refs = {m.key: m.value for m in matches}
        assert "chapter" in refs
        assert "IV" in refs["chapter"]

    def test_section_numbered(self):
        matches = detect_headings("Section 2.3 The Problem of Predication")
        refs = {m.key: m.value for m in matches}
        assert "section" in refs

    def test_introduction(self):
        matches = detect_headings("Introduction")
        refs = {m.key: m.value for m in matches}
        assert refs.get("division") == "Introduction"

    def test_conclusion(self):
        matches = detect_headings("Conclusion")
        refs = {m.key: m.value for m in matches}
        assert refs.get("division") == "Conclusion"

    def test_abstract(self):
        matches = detect_headings("Abstract")
        refs = {m.key: m.value for m in matches}
        assert refs.get("division") == "Abstract"

    def test_allcaps_heading(self):
        matches = detect_headings("THE PROBLEM OF BEING")
        refs = {m.key: m.value for m in matches}
        assert any(m.key == "heading" for m in matches)

    def test_numbered_heading(self):
        matches = detect_headings("1.2 Methodology and Sources")
        refs = {m.key: m.value for m in matches}
        assert "section" in refs

    def test_spoken_section_reference(self):
        matches = detect_headings("so if you look at section 132 Hegel says")
        refs = {m.key: m.value for m in matches}
        assert refs.get("section") == "§132"

    def test_spoken_doctrine(self):
        matches = detect_headings("we're now in the doctrine of essence")
        refs = {m.key: m.value for m in matches}
        assert refs.get("doctrine") == "Essence"

    def test_remark_subsection(self):
        matches = detect_headings("Remark: On the meaning of Aufheben")
        refs = {m.key: m.value for m in matches}
        assert refs.get("subsection") == "Remark"

    def test_no_structural_content(self):
        matches = detect_headings("This is ordinary body text about philosophy.")
        assert len(matches) == 0

    def test_part_heading(self):
        matches = detect_headings("Part II: The Doctrine of Essence")
        refs = {m.key: m.value for m in matches}
        assert "part" in refs
        assert "II" in refs["part"]

    def test_german_einleitung(self):
        matches = detect_headings("Einleitung")
        refs = {m.key: m.value for m in matches}
        assert refs.get("division") == "Einleitung"


class TestSectionPropagation:
    @pytest.fixture
    def db(self, tmp_path):
        d = Database(tmp_path / "test.db")
        yield d
        d.close()

    def test_heading_propagates_to_body(self, db):
        """Body chunks after a heading should inherit its structural_ref."""
        source = Source(title="Test Book", type=SourceType.PDF)
        db.insert_source(source)

        chunks = [
            Chunk(source_id=source.id, text="Introduction", paragraph_index=0, pdf_page=1),
            Chunk(source_id=source.id, text="This is body text about the topic.", paragraph_index=1, pdf_page=2),
            Chunk(source_id=source.id, text="More body text continues here.", paragraph_index=2, pdf_page=3),
            Chunk(source_id=source.id, text="Chapter 1: Being and Nothing", paragraph_index=3, pdf_page=4),
            Chunk(source_id=source.id, text="In this chapter we explore being.", paragraph_index=4, pdf_page=5),
        ]
        db.insert_chunks_batch(chunks)

        tagger = StructuralTagger()
        updated = tagger.tag_source(source.id, db)
        assert updated > 0

        # Check that body chunks inherited the heading
        tagged = db.get_chunks_by_source(source.id)
        tagged.sort(key=lambda c: c.paragraph_index or 0)

        # Chunk 0: "Introduction" → division: Introduction
        assert tagged[0].structural_ref is not None
        assert tagged[0].structural_ref.get("division") == "Introduction"

        # Chunk 1-2: body → should inherit Introduction
        assert tagged[1].structural_ref is not None
        assert tagged[1].structural_ref.get("division") == "Introduction"

        # Chunk 3: "Chapter 1: Being and Nothing" → chapter
        assert tagged[3].structural_ref is not None
        assert "chapter" in tagged[3].structural_ref

        # Chunk 4: body after chapter → should inherit chapter
        assert tagged[4].structural_ref is not None
        assert "chapter" in tagged[4].structural_ref

    def test_chapter_resets_section(self, db):
        """A new chapter should reset the section context."""
        source = Source(title="Test", type=SourceType.PDF)
        db.insert_source(source)

        chunks = [
            Chunk(source_id=source.id, text="Section 1.1 First Topic", paragraph_index=0, pdf_page=1),
            Chunk(source_id=source.id, text="Body under 1.1", paragraph_index=1, pdf_page=2),
            Chunk(source_id=source.id, text="Chapter 2: New Chapter", paragraph_index=2, pdf_page=10),
            Chunk(source_id=source.id, text="Body under chapter 2", paragraph_index=3, pdf_page=11),
        ]
        db.insert_chunks_batch(chunks)

        tagger = StructuralTagger()
        tagger.tag_source(source.id, db)

        tagged = db.get_chunks_by_source(source.id)
        tagged.sort(key=lambda c: c.paragraph_index or 0)

        # Chunk 3 (under Chapter 2) should have chapter but NOT section 1.1
        ref = tagged[3].structural_ref
        assert ref is not None
        assert "chapter" in ref
        assert ref.get("section") is None or "1.1" not in str(ref.get("section", ""))

    def test_unstructured_source(self, db):
        """Source with no structural content should leave chunks untagged."""
        source = Source(title="Plain Text", type=SourceType.PDF)
        db.insert_source(source)

        chunks = [
            Chunk(source_id=source.id, text="This is just ordinary text about nothing structural.", paragraph_index=0),
            Chunk(source_id=source.id, text="Another paragraph of plain content without any headings.", paragraph_index=1),
        ]
        db.insert_chunks_batch(chunks)

        tagger = StructuralTagger()
        updated = tagger.tag_source(source.id, db)
        assert updated == 0
