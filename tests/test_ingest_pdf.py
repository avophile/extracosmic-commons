"""Tests for PDF ingestion pipeline."""

import pytest

from extracosmic_commons.ingest.pdf import PDFIngester, _split_page_text


class TestSplitPageText:
    def test_short_text_not_split(self):
        result = _split_page_text("Short text", max_chars=1500)
        assert len(result) == 1
        assert result[0] == "Short text"

    def test_long_text_split_at_paragraphs(self):
        text = ("Paragraph one. " * 50) + "\n\n" + ("Paragraph two. " * 50)
        result = _split_page_text(text, max_chars=500)
        assert len(result) >= 2

    def test_very_short_segments_filtered(self):
        """When splitting produces segments below MIN_PAGE_TEXT_CHARS, they're dropped."""
        # Two paragraphs: one tiny, one substantial
        text = "Hi\n\n" + ("Substantial paragraph content. " * 10)
        result = _split_page_text(text, max_chars=200)
        # The tiny "Hi" segment should be filtered out
        assert all(len(r) >= 50 for r in result)


class TestPDFIngester:
    def test_parse_creates_source_and_chunks(self, tmp_path):
        """Test with a minimal PDF. We create one using pypdf."""
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        # pypdf blank pages have no text, so we test the empty case
        pdf_path = tmp_path / "test.pdf"
        with open(pdf_path, "wb") as f:
            writer.write(f)

        ingester = PDFIngester()
        source, chunks = ingester.parse_pdf(
            pdf_path, title="Test PDF", author=["Author"]
        )
        assert source.title == "Test PDF"
        assert source.author == ["Author"]
        # Blank page → no chunks (below MIN_PAGE_TEXT_CHARS)
        assert len(chunks) == 0

    def test_parse_with_metadata(self, tmp_path):
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        pdf_path = tmp_path / "meta.pdf"
        with open(pdf_path, "wb") as f:
            writer.write(f)

        ingester = PDFIngester()
        source, _ = ingester.parse_pdf(
            pdf_path,
            title="Hegel on Being",
            author=["Houlgate"],
            metadata={"year": 2022, "isbn": "978-x"},
        )
        assert source.metadata["year"] == 2022
        assert source.metadata["isbn"] == "978-x"
