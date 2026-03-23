"""Tests for Zotero RDF importer."""

import pytest

from extracosmic_commons.ingest.zotero import ZoteroItem, parse_rdf

# Minimal Zotero RDF for testing
SAMPLE_RDF = """<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF
 xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
 xmlns:z="http://www.zotero.org/namespaces/export#"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:bib="http://purl.org/net/biblio#"
 xmlns:foaf="http://xmlns.com/foaf/0.1/"
 xmlns:link="http://purl.org/rss/1.0/modules/link/"
 xmlns:prism="http://prismstandard.org/namespaces/1.2/basic/">
    <rdf:Description rdf:about="#item_1">
        <z:itemType>book</z:itemType>
        <dc:title>Hegel on Being</dc:title>
        <bib:authors>
            <rdf:Seq>
                <rdf:li>
                    <foaf:Person>
                        <foaf:surname>Houlgate</foaf:surname>
                        <foaf:givenName>Stephen</foaf:givenName>
                    </foaf:Person>
                </rdf:li>
            </rdf:Seq>
        </bib:authors>
        <dc:date>2022</dc:date>
        <dc:publisher>Bloomsbury Academic</dc:publisher>
        <dc:identifier>978-1350055391</dc:identifier>
        <link:link rdf:resource="files/208678/Houlgate - 2022 - Hegel on being.pdf"/>
    </rdf:Description>
    <rdf:Description rdf:about="#item_2">
        <z:itemType>journalArticle</z:itemType>
        <dc:title>Hegel on the Category of Quantity</dc:title>
        <bib:authors>
            <rdf:Seq>
                <rdf:li>
                    <foaf:Person>
                        <foaf:surname>Houlgate</foaf:surname>
                        <foaf:givenName>Stephen</foaf:givenName>
                    </foaf:Person>
                </rdf:li>
            </rdf:Seq>
        </bib:authors>
        <dc:date>2014</dc:date>
        <prism:volume>55</prism:volume>
        <bib:pages>402-426</bib:pages>
    </rdf:Description>
    <rdf:Description rdf:about="#item_att">
        <z:itemType>attachment</z:itemType>
        <dc:title>PDF</dc:title>
    </rdf:Description>
</rdf:RDF>
"""


@pytest.fixture
def zotero_collection(tmp_path):
    """Create a minimal Zotero export directory structure."""
    coll = tmp_path / "Houlgate texts" / "1954 Houlgate, Stephen"
    coll.mkdir(parents=True)

    # Write RDF
    rdf_path = coll / "Houlgate.rdf"
    rdf_path.write_text(SAMPLE_RDF)

    # Create a fake PDF
    files_dir = coll / "files" / "208678"
    files_dir.mkdir(parents=True)
    pdf = files_dir / "Houlgate - 2022 - Hegel on being.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake pdf content")

    return tmp_path / "Houlgate texts"


class TestParseRDF:
    def test_parses_items(self, zotero_collection):
        rdf_files = list(zotero_collection.rglob("*.rdf"))
        items = parse_rdf(rdf_files[0], rdf_files[0].parent)

        # Should find 2 items (attachment is skipped)
        assert len(items) == 2

    def test_extracts_book_metadata(self, zotero_collection):
        rdf_files = list(zotero_collection.rglob("*.rdf"))
        items = parse_rdf(rdf_files[0], rdf_files[0].parent)

        book = next(it for it in items if it.title == "Hegel on Being")
        assert book.item_type == "book"
        assert book.authors == ["Houlgate, Stephen"]
        assert book.year == "2022"
        assert book.publisher == "Bloomsbury Academic"

    def test_extracts_article_metadata(self, zotero_collection):
        rdf_files = list(zotero_collection.rglob("*.rdf"))
        items = parse_rdf(rdf_files[0], rdf_files[0].parent)

        article = next(it for it in items if "Quantity" in it.title)
        assert article.item_type == "journalArticle"
        assert article.volume == "55"
        assert article.pages == "402-426"

    def test_resolves_pdf_paths(self, zotero_collection):
        rdf_files = list(zotero_collection.rglob("*.rdf"))
        items = parse_rdf(rdf_files[0], rdf_files[0].parent)

        book = next(it for it in items if it.title == "Hegel on Being")
        assert book.pdf_paths is not None
        assert len(book.pdf_paths) == 1
        assert book.pdf_paths[0].name == "Houlgate - 2022 - Hegel on being.pdf"

    def test_no_pdf_for_article(self, zotero_collection):
        rdf_files = list(zotero_collection.rglob("*.rdf"))
        items = parse_rdf(rdf_files[0], rdf_files[0].parent)

        article = next(it for it in items if "Quantity" in it.title)
        assert article.pdf_paths is None

    def test_skips_attachments(self, zotero_collection):
        rdf_files = list(zotero_collection.rglob("*.rdf"))
        items = parse_rdf(rdf_files[0], rdf_files[0].parent)

        types = [it.item_type for it in items]
        assert "attachment" not in types
