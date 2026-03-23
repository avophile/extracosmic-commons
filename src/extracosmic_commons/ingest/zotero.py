"""Zotero RDF export importer.

Parses Zotero export directories (RDF metadata + PDFs in files/{id}/)
to extract rich bibliographic metadata and ingest the associated PDFs.

The HegelTranscripts project contains four Zotero collections:
- Hegel texts (1,042 PDFs)
- Radnik texts (55 PDFs)
- Thompson texts (46 PDFs)
- Houlgate texts (30 PDFs)

Each collection is a directory containing:
- An .rdf file with Zotero metadata (title, authors, date, publisher, etc.)
- A files/ subdirectory with numbered folders, each containing one PDF
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..database import Database
from ..embeddings import EmbeddingPipeline
from ..index import FAISSIndex
from ..models import Source
from .pdf import PDFIngester

logger = logging.getLogger(__name__)

# XML namespaces used in Zotero RDF exports
NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "z": "http://www.zotero.org/namespaces/export#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
    "bib": "http://purl.org/net/biblio#",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "link": "http://purl.org/rss/1.0/modules/link/",
    "prism": "http://prismstandard.org/namespaces/1.2/basic/",
}


@dataclass
class ZoteroItem:
    """Parsed metadata from a Zotero RDF item."""

    item_type: str
    title: str
    authors: list[str]
    year: str | None = None
    publisher: str | None = None
    doi: str | None = None
    isbn: str | None = None
    journal: str | None = None
    volume: str | None = None
    pages: str | None = None
    abstract: str | None = None
    pdf_paths: list[Path] | None = None
    rdf_id: str | None = None


def parse_rdf(rdf_path: Path, collection_root: Path) -> list[ZoteroItem]:
    """Parse a Zotero RDF file and resolve PDF paths.

    Args:
        rdf_path: Path to the .rdf file.
        collection_root: Parent directory containing the files/ subdirectory.
    """
    try:
        tree = ET.parse(str(rdf_path))
    except ET.ParseError as e:
        logger.error(f"Failed to parse RDF {rdf_path}: {e}")
        return []

    root = tree.getroot()
    items = []

    # Find all items (Description elements with z:itemType)
    for desc in root.findall(".//rdf:Description", NS):
        item_type_el = desc.find("z:itemType", NS)
        if item_type_el is None:
            continue

        item_type = item_type_el.text or "unknown"

        # Skip attachment entries — they're the PDF file records, not items
        if item_type == "attachment":
            continue

        # Title
        title_el = desc.find("dc:title", NS)
        title = title_el.text.strip() if title_el is not None and title_el.text else ""
        if not title:
            # Try to get title from parent element (isPartOf)
            part_of = desc.find("dcterms:isPartOf", NS)
            if part_of is not None:
                book = part_of.find(".//dc:title", NS)
                if book is not None and book.text:
                    title = book.text.strip()

        if not title:
            continue

        # Authors
        authors = []
        for person in desc.findall(".//bib:authors//foaf:Person", NS):
            surname = person.find("foaf:surname", NS)
            given = person.find("foaf:givenName", NS)
            if surname is not None and surname.text:
                name = surname.text
                if given is not None and given.text:
                    name = f"{surname.text}, {given.text}"
                authors.append(name)

        # Date / year
        date_el = desc.find("dc:date", NS)
        year = None
        if date_el is not None and date_el.text:
            # Extract just the year from dates like "2010-03-15" or "2010"
            year = date_el.text.strip()[:4]

        # Publisher
        publisher_el = desc.find("dc:publisher", NS)
        publisher = publisher_el.text.strip() if publisher_el is not None and publisher_el.text else None

        # Identifiers
        doi = isbn = None
        for ident in desc.findall("dc:identifier", NS):
            if ident.text:
                text = ident.text.strip()
                if text.startswith("10.") or "doi" in text.lower():
                    doi = text
                elif "isbn" in text.lower() or len(text.replace("-", "")) in (10, 13):
                    isbn = text

        # Journal
        journal = None
        part_of = desc.find("dcterms:isPartOf", NS)
        if part_of is not None:
            journal_el = part_of.find(".//dc:title", NS)
            if journal_el is not None and journal_el.text:
                journal = journal_el.text.strip()

        # Volume / pages
        volume_el = desc.find("prism:volume", NS)
        volume = volume_el.text.strip() if volume_el is not None and volume_el.text else None

        pages_el = desc.find("bib:pages", NS)
        pages = pages_el.text.strip() if pages_el is not None and pages_el.text else None

        # Abstract
        abstract_el = desc.find("dcterms:abstract", NS)
        abstract = abstract_el.text.strip() if abstract_el is not None and abstract_el.text else None

        # RDF ID for linking to file attachments
        rdf_id = desc.get(f"{{{NS['rdf']}}}about", "")

        # Resolve PDF paths from link elements
        pdf_paths = []
        for link_el in desc.findall("link:link", NS):
            href = link_el.get(f"{{{NS['rdf']}}}resource", "")
            if href:
                # href is like "files/208678/Houlgate - 2022 - Hegel on being.pdf"
                pdf_path = collection_root / href
                if pdf_path.exists() and pdf_path.suffix.lower() == ".pdf":
                    pdf_paths.append(pdf_path)

        # If no links found, try to find PDFs by scanning the rdf_id's attachments
        if not pdf_paths and rdf_id:
            # Look for attachment items that reference this item
            for att_desc in root.findall(".//rdf:Description", NS):
                att_type = att_desc.find("z:itemType", NS)
                if att_type is not None and att_type.text == "attachment":
                    for link_el in att_desc.findall("link:link", NS):
                        href = link_el.get(f"{{{NS['rdf']}}}resource", "")
                        if href:
                            pdf_path = collection_root / href
                            if pdf_path.exists() and pdf_path.suffix.lower() == ".pdf":
                                pdf_paths.append(pdf_path)

        items.append(ZoteroItem(
            item_type=item_type,
            title=title,
            authors=authors,
            year=year,
            publisher=publisher,
            doi=doi,
            isbn=isbn,
            journal=journal,
            volume=volume,
            pages=pages,
            abstract=abstract,
            pdf_paths=pdf_paths if pdf_paths else None,
            rdf_id=rdf_id,
        ))

    # If RDF parsing didn't find PDF links, scan the files/ directory directly
    files_dir = collection_root / "files"
    if files_dir.exists():
        # Build a set of already-found PDFs
        found_pdfs = set()
        for item in items:
            if item.pdf_paths:
                for p in item.pdf_paths:
                    found_pdfs.add(p)

        # Find unlinked PDFs
        unlinked_pdfs = []
        for subdir in sorted(files_dir.iterdir()):
            if subdir.is_dir():
                for pdf_file in subdir.glob("*.pdf"):
                    if pdf_file not in found_pdfs:
                        unlinked_pdfs.append(pdf_file)

        # Try to match unlinked PDFs to items by filename
        for pdf_path in unlinked_pdfs:
            matched = False
            pdf_stem = pdf_path.stem.lower()
            for item in items:
                if item.pdf_paths is None:
                    # Check if any author name appears in the PDF filename
                    for author in item.authors:
                        if author.split(",")[0].lower() in pdf_stem:
                            item.pdf_paths = [pdf_path]
                            matched = True
                            break
                if matched:
                    break

    return items


class ZoteroImporter:
    """Imports a Zotero export directory into the Extracosmic Commons."""

    def __init__(self):
        self._pdf_ingester = PDFIngester()

    def scan_collection(self, collection_path: Path) -> list[ZoteroItem]:
        """Scan a Zotero export directory and return parsed items."""
        # Find the .rdf file
        rdf_files = list(collection_path.rglob("*.rdf"))
        if not rdf_files:
            logger.warning(f"No .rdf file found in {collection_path}")
            return []

        rdf_path = rdf_files[0]
        collection_root = rdf_path.parent
        return parse_rdf(rdf_path, collection_root)

    def import_collection(
        self,
        collection_path: Path,
        db: Database,
        embedder: EmbeddingPipeline,
        index: FAISSIndex,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[Source]:
        """Import all PDFs from a Zotero collection.

        Args:
            collection_path: Root directory of the Zotero export.
            db: Database instance.
            embedder: Embedding pipeline.
            index: FAISS index.
            progress_callback: Optional fn(completed, total, current_title).

        Returns:
            List of created Source records.
        """
        items = self.scan_collection(collection_path)
        items_with_pdfs = [it for it in items if it.pdf_paths]

        sources = []
        failed = []
        skipped = 0
        total = len(items_with_pdfs)

        for i, item in enumerate(items_with_pdfs):
            if progress_callback:
                progress_callback(i, total, item.title[:60])

            # Dedup check
            if db.source_exists(item.title, item.authors or None):
                skipped += 1
                continue

            # Take the first PDF
            pdf_path = item.pdf_paths[0]

            metadata = {}
            if item.year:
                metadata["year"] = item.year
            if item.publisher:
                metadata["publisher"] = item.publisher
            if item.doi:
                metadata["doi"] = item.doi
            if item.isbn:
                metadata["isbn"] = item.isbn
            if item.journal:
                metadata["journal"] = item.journal
            if item.volume:
                metadata["volume"] = item.volume
            if item.pages:
                metadata["pages"] = item.pages
            if item.abstract:
                metadata["abstract"] = item.abstract[:500]  # Truncate long abstracts
            metadata["zotero_item_type"] = item.item_type

            try:
                source = self._pdf_ingester.ingest(
                    path=pdf_path,
                    db=db,
                    embedder=embedder,
                    index=index,
                    title=item.title,
                    author=item.authors,
                    metadata=metadata,
                )
                sources.append(source)
            except Exception as e:
                logger.warning(f"Failed to ingest {item.title}: {e}")
                failed.append((item.title, str(e)))

        if progress_callback:
            progress_callback(total, total, "Done")

        logger.info(
            f"Zotero import: {len(sources)} imported, {skipped} skipped (dedup), "
            f"{len(failed)} failed out of {total} items with PDFs"
        )

        return sources
