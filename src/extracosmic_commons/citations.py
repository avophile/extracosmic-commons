"""Citation formatting and export for Extracosmic Commons.

Formats Source metadata as citations in Chicago 18th edition (humanities
standard) and exports to BibTeX, RIS, and CSV for reference manager
interoperability.

Ported from scholarly workbench citation_exporter.py, adapted for the
Extracosmic Commons Source data model.
"""

from __future__ import annotations

import re
from typing import Any

from .models import Source, SourceType


def _format_authors_chicago(authors: list[str]) -> str:
    """Format author list for Chicago note-bibliography style.

    First author: Last, First. Subsequent: First Last.
    """
    if not authors:
        return ""
    if len(authors) == 1:
        return authors[0]
    if len(authors) == 2:
        return f"{authors[0]} and {authors[1]}"
    # 3+ authors: first author, ..., and last author
    return ", ".join(authors[:-1]) + f", and {authors[-1]}"


def _format_authors_bibtex(authors: list[str]) -> str:
    """Format author list for BibTeX: 'Last, First and Last, First'."""
    return " and ".join(authors)


def _clean_doi(doi: str) -> str:
    """Normalize DOI to bare identifier (no URL prefix)."""
    doi = doi.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:", "DOI:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi.strip()


def _bibtex_key(source: Source) -> str:
    """Generate a BibTeX citation key: author_year_firstword."""
    author = source.author[0].split(",")[0].split()[-1] if source.author else "unknown"
    year = source.metadata.get("year", "nd")
    first_word = re.sub(r'[^a-zA-Z]', '', source.title.split()[0]) if source.title else "untitled"
    return f"{author}_{year}_{first_word}".lower()


def _is_article(source: Source) -> bool:
    """Determine if a source is a journal article (vs book/other)."""
    meta = source.metadata
    if meta.get("journal"):
        return True
    if meta.get("zotero_item_type") in ("journalArticle", "article"):
        return True
    return False


class CitationFormatter:
    """Formats Source metadata as citations in various styles."""

    def chicago(self, source: Source) -> str:
        """Chicago 18th edition note-bibliography format.

        If the source already has a chicago_citation in metadata (from
        scholarly workbench import), returns that directly.
        """
        meta = source.metadata

        # Use pre-existing citation if available
        if meta.get("chicago_citation"):
            return meta["chicago_citation"]

        authors = _format_authors_chicago(source.author)
        year = meta.get("year", "n.d.")
        title = source.title

        if source.type == SourceType.TRANSCRIPT:
            # Lecture format
            parts = [authors + "."] if authors else []
            parts.append(f'"{title}."')
            parts.append(f"Lecture, {year}.")
            if source.source_url:
                parts.append(source.source_url)
            return " ".join(parts)

        if _is_article(source):
            # Journal article
            journal = meta.get("journal", "")
            volume = meta.get("volume", "")
            issue = meta.get("issue", "")
            pages = meta.get("pages", "")

            parts = [authors + "."] if authors else []
            parts.append(f'"{title}."')
            if journal:
                journal_part = f"*{journal}*"
                if volume:
                    journal_part += f" {volume}"
                if issue:
                    journal_part += f", no. {issue}"
                journal_part += f" ({year})"
                if pages:
                    journal_part += f": {pages}"
                parts.append(journal_part + ".")
            else:
                parts.append(f"{year}.")
            doi = meta.get("doi")
            if doi:
                parts.append(f"https://doi.org/{_clean_doi(doi)}")
            return " ".join(parts)

        # Default: book format
        publisher = meta.get("publisher", "")
        parts = [authors + "."] if authors else []
        parts.append(f"*{title}*.")
        if source.edition:
            parts.append(f"{source.edition}.")
        if publisher:
            parts.append(f"{publisher}, {year}.")
        else:
            parts.append(f"{year}.")
        doi = meta.get("doi")
        if doi:
            parts.append(f"https://doi.org/{_clean_doi(doi)}")
        return " ".join(parts)

    def bibtex(self, source: Source) -> str:
        """BibTeX entry."""
        meta = source.metadata
        key = _bibtex_key(source)
        entry_type = "article" if _is_article(source) else "book"

        lines = [f"@{entry_type}{{{key},"]
        if source.author:
            lines.append(f"  author = {{{_format_authors_bibtex(source.author)}}},")
        lines.append(f"  title = {{{source.title}}},")
        year = meta.get("year", "")
        if year:
            lines.append(f"  year = {{{year}}},")
        if entry_type == "article":
            if meta.get("journal"):
                lines.append(f"  journal = {{{meta['journal']}}},")
            if meta.get("volume"):
                lines.append(f"  volume = {{{meta['volume']}}},")
            if meta.get("issue"):
                lines.append(f"  number = {{{meta['issue']}}},")
            if meta.get("pages"):
                lines.append(f"  pages = {{{meta['pages']}}},")
        else:
            if meta.get("publisher"):
                lines.append(f"  publisher = {{{meta['publisher']}}},")
            if meta.get("isbn"):
                lines.append(f"  isbn = {{{meta['isbn']}}},")
        if meta.get("doi"):
            lines.append(f"  doi = {{{_clean_doi(meta['doi'])}}},")
        if source.source_url:
            lines.append(f"  url = {{{source.source_url}}},")
        lines.append("}")
        return "\n".join(lines)

    def ris(self, source: Source) -> str:
        """RIS format entry."""
        meta = source.metadata
        entry_type = "JOUR" if _is_article(source) else "BOOK"

        lines = [f"TY  - {entry_type}"]
        for author in source.author:
            lines.append(f"AU  - {author}")
        lines.append(f"TI  - {source.title}")
        year = meta.get("year", "")
        if year:
            lines.append(f"PY  - {year}")
        if _is_article(source):
            if meta.get("journal"):
                lines.append(f"JO  - {meta['journal']}")
            if meta.get("volume"):
                lines.append(f"VL  - {meta['volume']}")
            if meta.get("issue"):
                lines.append(f"IS  - {meta['issue']}")
            if meta.get("pages"):
                lines.append(f"SP  - {meta['pages']}")
        else:
            if meta.get("publisher"):
                lines.append(f"PB  - {meta['publisher']}")
            if meta.get("isbn"):
                lines.append(f"SN  - {meta['isbn']}")
        if meta.get("doi"):
            lines.append(f"DO  - {_clean_doi(meta['doi'])}")
        if source.source_url:
            lines.append(f"UR  - {source.source_url}")
        lines.append("ER  - ")
        return "\n".join(lines)

    def csv_row(self, source: Source) -> dict[str, Any]:
        """Dict suitable for csv.DictWriter."""
        meta = source.metadata
        return {
            "title": source.title,
            "author": "; ".join(source.author),
            "year": meta.get("year", ""),
            "type": source.type.value,
            "publisher": meta.get("publisher", ""),
            "journal": meta.get("journal", ""),
            "volume": meta.get("volume", ""),
            "issue": meta.get("issue", ""),
            "pages": meta.get("pages", ""),
            "doi": meta.get("doi", ""),
            "isbn": meta.get("isbn", ""),
            "url": source.source_url or "",
        }
