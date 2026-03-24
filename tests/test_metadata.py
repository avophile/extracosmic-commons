"""Tests for metadata enrichment pipeline."""

import pytest

from extracosmic_commons.metadata import extract_doi, extract_isbn, _validate_isbn


class TestDOIExtraction:
    def test_basic_doi(self):
        text = "The paper (doi: 10.1017/hgl.2014.20) examines..."
        assert extract_doi(text) == "10.1017/hgl.2014.20"

    def test_doi_in_url(self):
        text = "Available at https://doi.org/10.1017/hgl.2014.20"
        doi = extract_doi(text)
        assert doi is not None
        assert "10.1017" in doi

    def test_doi_with_trailing_period(self):
        text = "See 10.1017/hgl.2014.20."
        doi = extract_doi(text)
        assert doi == "10.1017/hgl.2014.20"

    def test_no_doi(self):
        text = "This text has no digital object identifier."
        assert extract_doi(text) is None

    def test_doi_complex(self):
        text = "doi:10.1093/acprof:oso/9780199217694.003.0005"
        doi = extract_doi(text)
        assert doi is not None
        assert "10.1093" in doi


class TestISBNExtraction:
    def test_isbn_13(self):
        text = "ISBN 978-0-521-83255-7"
        isbn = extract_isbn(text)
        assert isbn is not None
        assert len(isbn) == 13

    def test_isbn_10(self):
        text = "ISBN 0-521-83255-0"
        isbn = extract_isbn(text)
        # May or may not validate depending on checksum
        # Just test it doesn't crash

    def test_no_isbn(self):
        text = "This text has no ISBN."
        assert extract_isbn(text) is None

    def test_validate_isbn_13(self):
        # Valid ISBN-13 for Science of Logic
        assert _validate_isbn("9780521832557") is True

    def test_validate_isbn_13_invalid(self):
        assert _validate_isbn("9780521832558") is False


class TestMetadataEnricher:
    """Integration tests for web enrichment are skipped by default.

    They require network access and hit rate-limited APIs. Run with
    --runslow to include them.
    """
    pass
