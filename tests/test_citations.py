"""Tests for citation formatting and export."""

import pytest

from extracosmic_commons.citations import CitationFormatter
from extracosmic_commons.models import Source, SourceType


@pytest.fixture
def formatter():
    return CitationFormatter()


@pytest.fixture
def book_source():
    return Source(
        title="Hegel on Being",
        type=SourceType.PDF,
        author=["Houlgate, Stephen"],
        metadata={
            "year": "2022",
            "publisher": "Bloomsbury Academic",
            "isbn": "978-1350055391",
        },
    )


@pytest.fixture
def article_source():
    return Source(
        title="Hegel on the Category of Quantity",
        type=SourceType.PDF,
        author=["Houlgate, Stephen"],
        metadata={
            "year": "2014",
            "journal": "Hegel Bulletin",
            "volume": "35",
            "issue": "2",
            "pages": "402-426",
            "doi": "10.1017/hgl.2014.20",
            "zotero_item_type": "journalArticle",
        },
    )


@pytest.fixture
def transcript_source():
    return Source(
        title="Science of Logic Lectures",
        type=SourceType.TRANSCRIPT,
        author=["Thompson, Kevin"],
        source_url="https://youtube.com/playlist?list=PLZDJ",
        metadata={"year": "2024"},
    )


@pytest.fixture
def preformatted_source():
    return Source(
        title="Critique of Pure Reason",
        type=SourceType.PDF,
        author=["Kant, Immanuel"],
        metadata={
            "year": "1998",
            "chicago_citation": "Kant, Immanuel. Critique of Pure Reason. Cambridge: Cambridge UP, 1998.",
        },
    )


class TestChicago:
    def test_book(self, formatter, book_source):
        cite = formatter.chicago(book_source)
        assert "Houlgate" in cite
        assert "Hegel on Being" in cite
        assert "2022" in cite
        assert "Bloomsbury" in cite

    def test_article(self, formatter, article_source):
        cite = formatter.chicago(article_source)
        assert "Houlgate" in cite
        assert "Quantity" in cite
        assert "Hegel Bulletin" in cite
        assert "35" in cite
        assert "402-426" in cite

    def test_transcript(self, formatter, transcript_source):
        cite = formatter.chicago(transcript_source)
        assert "Thompson" in cite
        assert "Lecture" in cite
        assert "youtube.com" in cite

    def test_preformatted(self, formatter, preformatted_source):
        """If chicago_citation exists in metadata, use it directly."""
        cite = formatter.chicago(preformatted_source)
        assert "Cambridge" in cite
        assert "1998" in cite


class TestBibTeX:
    def test_book(self, formatter, book_source):
        bib = formatter.bibtex(book_source)
        assert bib.startswith("@book{")
        assert "author = {Houlgate, Stephen}" in bib
        assert "title = {Hegel on Being}" in bib
        assert "year = {2022}" in bib
        assert "publisher = {Bloomsbury Academic}" in bib

    def test_article(self, formatter, article_source):
        bib = formatter.bibtex(article_source)
        assert bib.startswith("@article{")
        assert "journal = {Hegel Bulletin}" in bib
        assert "volume = {35}" in bib
        assert "doi = {10.1017/hgl.2014.20}" in bib


class TestRIS:
    def test_book(self, formatter, book_source):
        ris = formatter.ris(book_source)
        assert "TY  - BOOK" in ris
        assert "AU  - Houlgate, Stephen" in ris
        assert "TI  - Hegel on Being" in ris
        assert "ER  - " in ris

    def test_article(self, formatter, article_source):
        ris = formatter.ris(article_source)
        assert "TY  - JOUR" in ris
        assert "JO  - Hegel Bulletin" in ris


class TestCSV:
    def test_csv_row(self, formatter, book_source):
        row = formatter.csv_row(book_source)
        assert row["title"] == "Hegel on Being"
        assert row["author"] == "Houlgate, Stephen"
        assert row["year"] == "2022"
        assert row["publisher"] == "Bloomsbury Academic"
