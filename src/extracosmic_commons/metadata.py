"""Metadata enrichment pipeline for Extracosmic Commons.

Extracts DOI/ISBN identifiers from PDF text and enriches source metadata
via CrossRef and OpenLibrary APIs. Ported from scholarly workbench
file_ingestion_agent.py (identifier extraction) and citation_enhancer.py
(web enrichment).
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from .database import Database
from .models import Source

logger = logging.getLogger(__name__)

# DOI pattern: 10.NNNN/anything (from file_ingestion_agent.py)
DOI_RE = re.compile(r'(10\.\d{4,9}/[^\s\]\)>]+)', re.IGNORECASE)

# ISBN patterns (10 or 13 digit, with optional hyphens in any position)
ISBN_13_RE = re.compile(r'(?:ISBN(?:-1[03])?[:\s]*)?(97[89][\d-]{10,15}\d)')
ISBN_10_RE = re.compile(r'(?:ISBN(?:-10)?[:\s]*)?([\dX]{10})')
ISBN_CLEAN_RE = re.compile(r'[-\s]')


def _clean_doi(doi: str) -> str:
    """Strip trailing punctuation from extracted DOI."""
    return doi.rstrip('.,;:)}]>"\'')


def _validate_isbn(isbn: str) -> bool:
    """Validate ISBN checksum (Luhn algorithm for ISBN-13, modular for ISBN-10)."""
    digits = ISBN_CLEAN_RE.sub('', isbn)

    if len(digits) == 13:
        total = sum(
            int(d) * (1 if i % 2 == 0 else 3)
            for i, d in enumerate(digits)
        )
        return total % 10 == 0
    elif len(digits) == 10:
        total = sum(
            (10 if d == 'X' else int(d)) * (10 - i)
            for i, d in enumerate(digits)
        )
        return total % 11 == 0
    return False


def extract_doi(text: str) -> str | None:
    """Extract a DOI from text."""
    m = DOI_RE.search(text)
    if m:
        return _clean_doi(m.group(1))
    return None


def extract_isbn(text: str) -> str | None:
    """Extract a valid ISBN from text."""
    for pattern in (ISBN_13_RE, ISBN_10_RE):
        m = pattern.search(text)
        if m:
            isbn = m.group(1)
            if _validate_isbn(isbn):
                return ISBN_CLEAN_RE.sub('', isbn)
    return None


class MetadataEnricher:
    """Extract identifiers from PDF text and enrich via web APIs."""

    def __init__(self, rate_limit_seconds: float = 1.0):
        self._rate_limit = rate_limit_seconds
        self._last_request_time = 0.0

    def _rate_limited_get(self, url: str) -> dict | None:
        """HTTP GET with rate limiting and error handling."""
        import httpx

        # Respect rate limit
        elapsed = time.time() - self._last_request_time
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)

        try:
            self._last_request_time = time.time()
            resp = httpx.get(
                url,
                timeout=15.0,
                headers={"User-Agent": "ExtracosmiCommons/0.1 (scholarly research platform)"},
                follow_redirects=True,
            )
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.debug(f"HTTP {resp.status_code} for {url}")
                return None
        except Exception as e:
            logger.debug(f"Request failed for {url}: {e}")
            return None

    def enrich_from_doi(self, doi: str) -> dict[str, Any]:
        """Lookup metadata via CrossRef API."""
        url = f"https://api.crossref.org/works/{doi}"
        data = self._rate_limited_get(url)
        if not data or "message" not in data:
            return {}

        msg = data["message"]
        result: dict[str, Any] = {"doi": doi}

        # Title
        titles = msg.get("title", [])
        if titles:
            result["title"] = titles[0]

        # Authors
        authors = []
        for a in msg.get("author", []):
            family = a.get("family", "")
            given = a.get("given", "")
            if family and given:
                authors.append(f"{family}, {given}")
            elif family:
                authors.append(family)
        if authors:
            result["authors"] = authors

        # Year
        issued = msg.get("issued", {}).get("date-parts", [[]])
        if issued and issued[0]:
            result["year"] = str(issued[0][0])

        # Publisher
        if msg.get("publisher"):
            result["publisher"] = msg["publisher"]

        # Journal
        container = msg.get("container-title", [])
        if container:
            result["journal"] = container[0]

        # Volume, issue, pages
        if msg.get("volume"):
            result["volume"] = msg["volume"]
        if msg.get("issue"):
            result["issue"] = msg["issue"]
        if msg.get("page"):
            result["pages"] = msg["page"]

        return result

    def enrich_from_isbn(self, isbn: str) -> dict[str, Any]:
        """Lookup metadata via OpenLibrary API."""
        url = f"https://openlibrary.org/isbn/{isbn}.json"
        data = self._rate_limited_get(url)
        if not data:
            return {}

        result: dict[str, Any] = {"isbn": isbn}

        if data.get("title"):
            result["title"] = data["title"]

        if data.get("publish_date"):
            # Extract year from various date formats
            year_match = re.search(r'(\d{4})', data["publish_date"])
            if year_match:
                result["year"] = year_match.group(1)

        if data.get("publishers"):
            result["publisher"] = data["publishers"][0]

        if data.get("number_of_pages"):
            result["page_count"] = data["number_of_pages"]

        return result

    def enrich_source(
        self,
        source: Source,
        db: Database,
        force: bool = False,
    ) -> dict[str, Any]:
        """Extract identifiers from a source's chunks and enrich via web APIs.

        Args:
            source: Source to enrich.
            db: Database to read chunks from.
            force: If True, re-enrich even if metadata already has doi/isbn.

        Returns:
            Dict of newly discovered metadata fields.
        """
        meta = source.metadata
        new_fields: dict[str, Any] = {}

        # Skip if already enriched (unless forced)
        if not force and meta.get("doi") and meta.get("year"):
            return {}

        # Get first and last chunks to search for identifiers
        chunks = db.get_chunks_by_source(source.id)
        if not chunks:
            return {}

        # Sort by page/paragraph
        chunks.sort(key=lambda c: (c.pdf_page or 0, c.paragraph_index or 0))

        # Search first 3 chunks for DOI (title pages, abstracts)
        doi = meta.get("doi")
        if not doi:
            first_text = " ".join(c.text for c in chunks[:3])
            doi = extract_doi(first_text)
            if doi:
                new_fields["doi"] = doi

        # Search last 5 chunks for ISBN (copyright pages)
        isbn = meta.get("isbn")
        if not isbn:
            last_text = " ".join(c.text for c in chunks[-5:])
            isbn = extract_isbn(last_text)
            if isbn:
                new_fields["isbn"] = isbn

        # Enrich via web APIs
        if doi and not meta.get("year"):
            web_meta = self.enrich_from_doi(doi)
            for key in ("year", "publisher", "journal", "volume", "issue", "pages"):
                if key not in meta and key in web_meta:
                    new_fields[key] = web_meta[key]

        elif isbn and not meta.get("year"):
            web_meta = self.enrich_from_isbn(isbn)
            for key in ("year", "publisher", "page_count"):
                if key not in meta and key in web_meta:
                    new_fields[key] = web_meta[key]

        # Update source metadata in database
        if new_fields:
            updated_meta = {**meta, **new_fields}
            db.conn.execute(
                "UPDATE sources SET metadata = ? WHERE id = ?",
                (json.dumps(updated_meta), source.id),
            )
            db.conn.commit()
            source.metadata = updated_meta

        return new_fields

    def enrich_corpus(
        self,
        db: Database,
        dry_run: bool = False,
        progress_callback=None,
    ) -> int:
        """Enrich all sources missing key metadata.

        Returns count of sources enriched.
        """
        sources = db.get_all_sources()
        # Focus on sources missing year or DOI
        candidates = [
            s for s in sources
            if not s.metadata.get("year") or not s.metadata.get("doi")
        ]

        enriched = 0
        total = len(candidates)

        for i, source in enumerate(candidates):
            if progress_callback:
                progress_callback(i, total, source.title[:60])

            if dry_run:
                chunks = db.get_chunks_by_source(source.id)
                if chunks:
                    first_text = " ".join(c.text for c in chunks[:3])
                    doi = extract_doi(first_text)
                    if doi:
                        logger.info(f"  Would enrich: {source.title} (DOI: {doi})")
                        enriched += 1
            else:
                new_fields = self.enrich_source(source, db)
                if new_fields:
                    logger.info(f"  Enriched: {source.title} (+{list(new_fields.keys())})")
                    enriched += 1

        if progress_callback:
            progress_callback(total, total, "Done")

        return enriched
