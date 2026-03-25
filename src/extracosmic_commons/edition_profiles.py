"""Edition profiles for critical edition texts.

An EditionProfile describes how a specific edition of a scholarly work
encodes its reference system — where margin annotations appear, what
regex patterns match them, how PDF pages map to printed pages, and how
section headings map to canonical IDs shared across translations.

This is the generalization mechanism: adding support for a new author,
work, or translation requires only defining a new EditionProfile, not
changing any code.

Examples:
    Hegel GW 21 (German): GW page refs in running headers
    Hegel di Giovanni (English): GW page refs as bold margin text
    Kant Guyer/Wood: A/B edition refs in margins
    Aristotle: Bekker numbers in margins
    Plato: Stephanus numbers in margins
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EditionProfile:
    """How a specific edition encodes its reference system."""

    # Identity
    name: str
    work_id: str
    edition_id: str

    # Reference system
    reference_system: str  # "gw", "akademie", "bekker", "stephanus", "paragraph"
    ref_label: str  # Display label: "GW", "Ak", "Bekker", "§"

    # Patterns for finding edition references in extracted text
    margin_ref_pattern: re.Pattern | None = None
    header_ref_pattern: re.Pattern | None = None
    inline_ref_pattern: re.Pattern | None = None

    # PDF page → printed page mapping
    # translation_page = pdf_page + page_offset
    page_offset: int = 0

    # Work metadata
    original_language: str = "de"
    is_original_language: bool = False
    original_date: str | None = None
    edition_date: str | None = None
    translator: str | None = None
    publisher: str | None = None

    # Version info (for works with multiple versions, e.g., SoL 1812 vs 1832)
    version_id: str | None = None
    version_label: str | None = None
    gw_volume: str | None = None

    # Canonical section map: heading text → canonical section ID
    # Used to normalize section headings across translations
    section_map: dict[str, str] | None = None

    # Cross-edition § concordance: this_ed_§ → {other_ed: other_§}
    # For works where § numbers shift between editions (e.g., Encyclopedia)
    section_concordance: dict[str, dict[str, str]] | None = None

    # Font-based heading detection rules (for pdfplumber extraction)
    # Each rule: {"font_type": "small_caps"|"italic"|"bold", "min_size": float, ...}
    heading_font_rules: list[dict[str, Any]] | None = None

    def to_source_metadata(self) -> dict[str, Any]:
        """Generate Source.metadata dict from this profile."""
        meta: dict[str, Any] = {
            "work_id": self.work_id,
            "edition_id": self.edition_id,
            "edition_label": f"{self.translator or self.name} {self.edition_date or ''}".strip(),
            "reference_system": self.reference_system,
            "is_original_language": self.is_original_language,
            "original_language": self.original_language,
        }
        if self.original_date:
            meta["original_date"] = self.original_date
        if self.edition_date:
            meta["edition_date"] = self.edition_date
            meta["year"] = self.edition_date
        if self.translator:
            meta["translator"] = self.translator
        if self.publisher:
            meta["publisher"] = self.publisher
        if self.version_id:
            meta["version_id"] = self.version_id
        if self.version_label:
            meta["version_label"] = self.version_label
        if self.gw_volume:
            meta["gw_volume"] = self.gw_volume
        return meta


# ── Built-in profiles ──

HEGEL_SOL_DI_GIOVANNI = EditionProfile(
    name="Hegel, Science of Logic (di Giovanni 2010)",
    work_id="hegel-sol",
    edition_id="di-giovanni-2010",
    reference_system="gw",
    ref_label="GW",
    margin_ref_pattern=re.compile(r'\b(21\.\d{2,3})\b'),
    original_language="de",
    is_original_language=False,
    original_date="1812",
    edition_date="2010",
    translator="di Giovanni, George",
    publisher="Cambridge UP",
)

HEGEL_SOL_MILLER = EditionProfile(
    name="Hegel, Science of Logic (Miller 1969)",
    work_id="hegel-sol",
    edition_id="miller-1969",
    reference_system="gw",
    ref_label="GW",
    margin_ref_pattern=re.compile(r'\b(21\.\d{2,3})\b'),
    original_language="de",
    is_original_language=False,
    original_date="1812",
    edition_date="1969",
    translator="Miller, A.V.",
    publisher="Allen & Unwin",
)

HEGEL_SOL_GW21 = EditionProfile(
    name="Hegel, Wissenschaft der Logik (GW 21, 1832)",
    work_id="hegel-sol",
    edition_id="gw-21",
    reference_system="gw",
    ref_label="GW",
    header_ref_pattern=re.compile(r'^(\d{2,3})\s+(?:LOGIK|LEHRE|BESTIMMTHEIT|FÜRSICHSEYN)'),
    original_language="de",
    is_original_language=True,
    original_date="1832",
    edition_date="1832",
    version_id="sol-being-1832",
    version_label="Book One: Being (revised 1832)",
    gw_volume="21",
)

HEGEL_SOL_GW11 = EditionProfile(
    name="Hegel, Wissenschaft der Logik (GW 11, 1812)",
    work_id="hegel-sol",
    edition_id="gw-11",
    reference_system="gw",
    ref_label="GW",
    original_language="de",
    is_original_language=True,
    original_date="1812",
    edition_date="1812",
    version_id="sol-being-1812",
    version_label="Book One: Being (original 1812)",
    gw_volume="11",
)

HEGEL_PHENOMENOLOGY_INWOOD = EditionProfile(
    name="Hegel, Phenomenology of Spirit (Inwood 2018)",
    work_id="hegel-phenomenology",
    edition_id="inwood-2018",
    reference_system="paragraph",
    ref_label="§",
    margin_ref_pattern=re.compile(r'§\s*(\d+)'),
    original_language="de",
    is_original_language=False,
    original_date="1807",
    edition_date="2018",
    translator="Inwood, Michael",
    publisher="Oxford UP",
)

HEGEL_PHENOMENOLOGY_MILLER = EditionProfile(
    name="Hegel, Phenomenology of Spirit (Miller 1977)",
    work_id="hegel-phenomenology",
    edition_id="miller-1977",
    reference_system="paragraph",
    ref_label="§",
    margin_ref_pattern=re.compile(r'§\s*(\d+)'),
    original_language="de",
    is_original_language=False,
    original_date="1807",
    edition_date="1977",
    translator="Miller, A.V.",
    publisher="Oxford UP",
)

HEGEL_PHENOMENOLOGY_PINKARD = EditionProfile(
    name="Hegel, Phenomenology of Spirit (Pinkard 2018)",
    work_id="hegel-phenomenology",
    edition_id="pinkard-2018",
    reference_system="paragraph",
    ref_label="§",
    margin_ref_pattern=re.compile(r'§\s*(\d+)'),
    original_language="de",
    is_original_language=False,
    original_date="1807",
    edition_date="2018",
    translator="Pinkard, Terry",
    publisher="Cambridge UP",
)

HEGEL_ENCYCLOPEDIA_GW20 = EditionProfile(
    name="Hegel, Enzyklopädie (GW 20, 3rd ed 1830)",
    work_id="hegel-encyclopedia",
    edition_id="gw-20",
    reference_system="paragraph",
    ref_label="§",
    margin_ref_pattern=re.compile(r'§\.?\s*(\d+)'),
    original_language="de",
    is_original_language=True,
    original_date="1817",
    edition_date="1830",
    version_id="encyclopedia-3rd-1830",
    version_label="3rd edition (1830)",
    gw_volume="20",
)

KANT_CPR_GUYER_WOOD = EditionProfile(
    name="Kant, Critique of Pure Reason (Guyer/Wood 1998)",
    work_id="kant-cpr",
    edition_id="guyer-wood-1998",
    reference_system="akademie",
    ref_label="Ak",
    margin_ref_pattern=re.compile(r'\b([AB]\s*\d{1,4})\b'),
    original_language="de",
    is_original_language=False,
    original_date="1781",
    edition_date="1998",
    translator="Guyer, Paul; Wood, Allen",
    publisher="Cambridge UP",
)


# ── Profile registry ──

_BUILTIN_PROFILES: dict[str, EditionProfile] = {
    p.edition_id: p
    for p in [
        HEGEL_SOL_DI_GIOVANNI,
        HEGEL_SOL_MILLER,
        HEGEL_SOL_GW21,
        HEGEL_SOL_GW11,
        HEGEL_PHENOMENOLOGY_INWOOD,
        HEGEL_PHENOMENOLOGY_MILLER,
        HEGEL_PHENOMENOLOGY_PINKARD,
        HEGEL_ENCYCLOPEDIA_GW20,
        KANT_CPR_GUYER_WOOD,
    ]
}

_CUSTOM_PROFILES: dict[str, EditionProfile] = {}


def get_profile(edition_id: str) -> EditionProfile | None:
    """Look up a profile by edition_id."""
    return _CUSTOM_PROFILES.get(edition_id) or _BUILTIN_PROFILES.get(edition_id)


def list_profiles() -> list[EditionProfile]:
    """List all available profiles (built-in + custom)."""
    all_profiles = {**_BUILTIN_PROFILES, **_CUSTOM_PROFILES}
    return sorted(all_profiles.values(), key=lambda p: (p.work_id, p.edition_id))


def register_profile(profile: EditionProfile) -> None:
    """Register a custom edition profile."""
    _CUSTOM_PROFILES[profile.edition_id] = profile


def get_profiles_for_work(work_id: str) -> list[EditionProfile]:
    """Get all profiles for a given work (for cross-translation lookup)."""
    all_profiles = {**_BUILTIN_PROFILES, **_CUSTOM_PROFILES}
    return [p for p in all_profiles.values() if p.work_id == work_id]
