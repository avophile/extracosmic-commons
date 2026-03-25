# Critical Edition Ingester — Multi-Edition, Multi-Translation Architecture

## Context

Two problems exposed by the "nodal lines of measures" search:

1. **Wrong PDFs in corpus**: Zotero RDF parser mislinked SoL section metadata to unrelated PDFs. The actual di Giovanni section PDFs are in `~/Documents/The Hegel Collection/`.

2. **No multi-edition page references**: Scholarly texts need both translation page numbers AND critical edition references (GW 21.364, Ak A235/B294, Bekker 1072a).

A deeper architectural issue: the system must handle **multiple translations of the same work**. The Phenomenology of Spirit exists in translations by Inwood (2018), Pinkard (2018), Miller (1977), and Fuss & Dobbins (2019). Their paragraph breaks don't align, but their ToC structures are comparable. A scholar needs to search one translation and see the corresponding passage from the others.

The hegel-bilingual project already solved the hardest part — character-level PDF extraction with GW reference parsing — for Section I (Quality). We port and generalize that approach.

## Data Model: Work → Edition → Source → Chunk

```
Work (abstract entity)
  "Hegel, Phänomenologie des Geistes" (1807)
  ├── Edition: Inwood 2018 (preferred)
  │   └── Sources: full PDF, section PDFs
  ├── Edition: Miller 1977
  ├── Edition: Pinkard 2018
  ├── Edition: Fuss & Dobbins 2019
  └── Edition: German original (Suhrkamp)

Work
  "Hegel, Wissenschaft der Logik"
  │
  │  NOTE: The SoL has a complex textual history. Hegel revised only
  │  Book One (Being) in 1832. Books Two and Three were never revised.
  │  Di Giovanni translates the revised Being (GW 21) + original
  │  Essence & Concept (GW 11, GW 12). The GW has BOTH versions
  │  of Book One.
  │
  ├── Version: Book One — Being (1812 original)
  │   └── Edition: GW 11 (German critical edition, 1812 text)
  │       └── Source: GW11 PDF
  │
  ├── Version: Book One — Being (1832 revision)
  │   ├── Edition: GW 21 (German critical edition, 1832 text)
  │   │   └── Source: GW21 PDF
  │   └── Edition: di Giovanni 2010 (Book One only)
  │       └── Sources: section PDFs (Quality, Quantity, Measure)
  │
  ├── Version: Book Two — Essence (1813, never revised)
  │   ├── Edition: GW 11 (German)
  │   └── Edition: di Giovanni 2010 (Book Two portion)
  │       └── Sources: section PDFs (Reflection, Appearance, Actuality)
  │
  ├── Version: Book Three — Concept (1816, never revised)
  │   ├── Edition: GW 12 (German)
  │   └── Edition: di Giovanni 2010 (Book Three portion)
  │       └── Sources: section PDFs (Subjectivity, Objectivity, Idea)
  │
  └── Edition: Miller 1969 (complete translation)
      └── Source: The Science of Logic (Miller).pdf
```

**Implementation**: Works and Editions live in Source.metadata — no new tables needed:

```python
# Simple case: Phenomenology of Spirit (one version, multiple translations)
source.metadata = {
    "work_id": "hegel-phenomenology",
    "work_title": "Phänomenologie des Geistes",
    "edition_id": "inwood-2018",
    "edition_label": "Inwood 2018",
    "translator": "Inwood, Michael",
    "original_date": "1807",         # when the work was first published
    "edition_date": "2018",          # when this edition/translation was published
    "year": "2018",                  # kept for backward compat (= edition_date)
    "publisher": "Oxford UP",
    "is_original_language": False,
    "original_language": "de",
    "reference_system": "paragraph",
}

# Complex case: SoL di Giovanni — a COMPOSITE translation spanning
# multiple versions and GW volumes
source.metadata = {
    "work_id": "hegel-sol",
    "work_title": "Wissenschaft der Logik",
    "edition_id": "di-giovanni-2010",
    "edition_label": "di Giovanni 2010",
    "translator": "di Giovanni, George",
    "original_date": "1812",         # first publication of the work
    "edition_date": "2010",
    "publisher": "Cambridge UP",
    "is_original_language": False,
    "original_language": "de",
    "reference_system": "gw",
    # Version info — which version of the original this edition translates
    "version_id": "sol-being-1832",  # this section PDF covers the 1832 revision
    "version_label": "Book One: Being (revised 1832)",
    "gw_volume": "21",              # GW 21 for the 1832 Being
}

# The 1812 original of Book One (German only, no English translation)
source.metadata = {
    "work_id": "hegel-sol",
    "work_title": "Wissenschaft der Logik",
    "edition_id": "gw-11",
    "edition_label": "GW 11 (1812)",
    "original_date": "1812",
    "edition_date": "1812",
    "is_original_language": True,
    "original_language": "de",
    "reference_system": "gw",
    "version_id": "sol-being-1812",  # the ORIGINAL 1812 version
    "version_label": "Book One: Being (original 1812)",
    "gw_volume": "11",
}
```

The Zotero convention `"1807 (2018)"` maps to `original_date="1807"`, `edition_date="2018"`. Citations use both: "Hegel. *Phenomenology of Spirit* (1807). Trans. Michael Inwood. Oxford UP, 2018."

**Version comparison**: When `version_id` differs but `work_id` matches, the system can compare versions — e.g., showing how Hegel revised the Being section between 1812 (GW 11) and 1832 (GW 21). Cross-version alignment uses the same layered approach as cross-translation alignment, with section headings as the guaranteed anchor.

**Cross-edition § mapping**: Some works have § numbers that shift between editions. Hegel's Encyclopedia had three editions (1817, 1827, 1830) with different § numbering — the GW critical apparatus records these variants (e.g., `§. 244.] O₃: §. 224. O₂: §. 244`). The EditionProfile can include an optional **§ concordance** — a mapping between edition-specific § numbers:

```python
# In the EditionProfile for Encyclopedia
section_concordance = {
    # 3rd ed §  →  { 1st ed §, 2nd ed § }
    "§244": {"O1": "§224", "O2": "§244"},
    "§132": {"O1": "§119", "O2": "§132"},
    ...
}
```

This concordance can be:
- Built manually from the critical apparatus footnotes
- Extracted semi-automatically from the GW edition's variant notes (the `O₁`, `O₂`, `O₃` apparatus at the bottom of each page)
- Used at query time: searching for "§132" in the 3rd edition also finds the corresponding passage in the 1st edition under "§119"

This pattern generalizes to any work with shifting reference numbers across editions (e.g., Kant's A/B edition numbering in the CPR, which is already handled by the dual `A235/B294` reference system).

## Cross-Translation Alignment (Layered)

**Core principle: the original language text defines the canonical paragraph structure.** Translations are aligned to the original, not to each other.

For German philosophical texts (Hegel, Kant, Fichte, Schelling), original paragraphs are typically longer than translation paragraphs — translators routinely split one German ¶ into 2-4 English ¶s, but never combine. So alignment is many-to-one:

```
DE ¶1  →  EN(Inwood) ¶1, ¶2       EN(Miller) ¶1
DE ¶2  →  EN(Inwood) ¶3            EN(Miller) ¶2, ¶3
DE ¶3  →  EN(Inwood) ¶4, ¶5, ¶6   EN(Miller) ¶4, ¶5
```

For French → English (Malabou, Derrida, Foucault), paragraphs are roughly 1:1.

For ancient texts (Aristotle, Plato), standardized reference systems (Bekker, Stephanus) provide alignment anchors regardless of paragraph structure.

**Layer 1 — Section/chapter alignment (guaranteed)**:

All translations of the same work share a ToC structure. We normalize section headings to a **canonical section ID** derived from the original's structure:

```python
chunk.structural_ref = {
    "canonical_section": "phenomenology.IV.A",   # shared across all editions
    "heading": "Self-Sufficiency and Non-Self-Sufficiency of Self-Consciousness",
    "chapter": "IV",
}
```

**Layer 2 — Original-anchored paragraph alignment (high confidence for German, varies for others)**:

Within a shared section, each original-language paragraph gets a canonical paragraph ID. Translation paragraphs are grouped under their original parent:

```python
# German original ¶ (the canonical unit)
de_chunk.structural_ref = {
    "canonical_section": "sol.being.quality.I.A",
    "canonical_para": "sol.gw21.68.p1",     # first paragraph on GW page 68
    "gw_page": "21.68",
}

# di Giovanni translation — two ¶s that correspond to this one German ¶
en_chunk_1.structural_ref = {
    "canonical_section": "sol.being.quality.I.A",
    "canonical_para": "sol.gw21.68.p1",     # same parent
    "translation_page": 59,
    "gw_page": "21.68",
    "para_group_index": 0,                   # first of 2 translation ¶s for this parent
    "alignment_confidence": 0.9,
}
en_chunk_2.structural_ref = {
    "canonical_section": "sol.being.quality.I.A",
    "canonical_para": "sol.gw21.68.p1",     # same parent
    "translation_page": 59,
    "gw_page": "21.68",
    "para_group_index": 1,                   # second of 2
    "alignment_confidence": 0.9,
}
```

**How paragraph alignment works:**

1. Within a canonical section, sort original-language ¶s by page + position → assign canonical_para IDs
2. Sort translation ¶s by page + position
3. Use GW page references (or other edition markers) as anchors to match translation ¶s to original ¶s
4. Between anchors, group translation ¶s proportionally by character count
5. Confidence is high when GW anchors are available (di Giovanni), lower when using proportional estimation only

**Search behavior**: When a result comes from Inwood, the system offers "See also: Miller §174-175, Pinkard p.104-105" by finding chunks from other editions with the same `canonical_para` (or failing that, same `canonical_section` + proportional position).

## Edition Profiles

```python
@dataclass
class EditionProfile:
    """How a specific edition encodes its reference system."""
    name: str                          # "Hegel, Science of Logic (di Giovanni 2010)"
    work_id: str                       # "hegel-sol"
    edition_id: str                    # "di-giovanni-2010"
    reference_system: str              # "gw" | "akademie" | "bekker" | "stephanus" | "paragraph"
    ref_label: str                     # "GW" | "Ak" | "Bekker"

    # Patterns for finding edition references in extracted text
    margin_ref_pattern: re.Pattern | None    # Bold margin annotations
    header_ref_pattern: re.Pattern | None    # Running header refs
    inline_ref_pattern: re.Pattern | None    # Inline refs

    # PDF page → printed page mapping
    page_offset: int = 0

    # Font-based heading rules (optional, for pdfplumber extraction)
    heading_font_rules: list | None = None

    # Canonical section map: maps section headings to canonical IDs
    section_map: dict | None = None
```

**Pre-defined profiles** (shipped with the package):

```python
HEGEL_SOL_DI_GIOVANNI = EditionProfile(
    work_id="hegel-sol",
    edition_id="di-giovanni-2010",
    reference_system="gw",
    ref_label="GW",
    margin_ref_pattern=re.compile(r'\b(21\.\d{2,3})\b'),
    section_map={...}   # from hegel_v2_spec.md SECTION_MAP
)

HEGEL_SOL_MILLER = EditionProfile(
    work_id="hegel-sol",
    edition_id="miller-1969",
    reference_system="gw",
    ref_label="GW",
    margin_ref_pattern=re.compile(r'\b(21\.\d{2,3})\b'),
)

HEGEL_PHENOMENOLOGY_INWOOD = EditionProfile(
    work_id="hegel-phenomenology",
    edition_id="inwood-2018",
    reference_system="paragraph",
    ref_label="§",
    margin_ref_pattern=re.compile(r'§\s*(\d+)'),
)

KANT_CPR_GUYER_WOOD = EditionProfile(
    work_id="kant-cpr",
    edition_id="guyer-wood-1998",
    reference_system="akademie",
    ref_label="Ak",
    margin_ref_pattern=re.compile(r'\b([AB]\s*\d{1,4})\b'),
)
```

Users can define custom profiles for any edition.

## Files to Create/Modify

```
src/extracosmic_commons/
├── edition_profiles.py         # NEW — EditionProfile dataclass + built-in profiles
├── ingest/
│   └── critical_edition.py     # NEW — character-level extraction + edition-aware chunking
├── models.py                   # MODIFY — document structural_ref conventions
├── database.py                 # MODIFY — add update/delete methods, work_id queries
├── search.py                   # MODIFY — cross-translation "see also" in results
├── cli.py                      # MODIFY — add ec ingest-edition, ec list-profiles
tests/
├── test_critical_edition.py    # NEW
├── test_edition_profiles.py    # NEW
```

## Step-by-Step Implementation

### Step 1: pdfplumber Dependency
Add `pdfplumber>=0.11` to pyproject.toml.

### Step 2: Edition Profile Model
**File:** `edition_profiles.py`
- `EditionProfile` dataclass with all fields above
- Built-in profiles for: Hegel SoL (di Giovanni, Miller), Hegel GW21, Hegel Phenomenology (Inwood, Miller, Pinkard), Kant CPR (Guyer/Wood)
- `get_profile(edition_id)`, `list_profiles()`, `detect_profile(source)` helpers
- `register_profile(profile)` for user-defined profiles

### Step 3: Character-Level PDF Extractor
**File:** `ingest/critical_edition.py`

Port from `hegel-bilingual/build_json_v3.py`:
- `_extract_page_from_chars(page)` — character-level text with indent detection
- `_build_line_from_chars(chars, threshold)` — gap-based spacing
- `_classify_line_by_font(line_chars)` — heading detection from font metadata
- `_extract_margin_refs(page, profile)` — find edition references using profile patterns

**New:**
- `_compute_translation_page(pdf_page, profile)` — apply page_offset
- `_assign_canonical_section(heading_text, profile)` — map headings to canonical IDs
- `_assign_canonical_para(chunk, section_chunks, profile)` — assign original-anchored paragraph ID using GW/edition refs as anchors, proportional grouping between anchors
- `_group_translation_paras(translation_chunks, original_chunks)` — group multiple translation ¶s under their original-language parent

**CriticalEditionIngester class:**
```python
def parse_pdf(self, path, profile, **metadata) -> tuple[Source, list[Chunk]]:
    """Extract with character-level precision, edition-aware references."""

def ingest(self, path, profile, db, embedder, index, **metadata) -> Source:
    """Full pipeline: extract → chunk with refs → embed → store."""

def align_to_original(self, translation_source_id, original_source_id, db) -> int:
    """Post-ingestion: align translation chunks to original-language paragraphs.
    Groups translation ¶s under their original parent using edition refs as anchors.
    Returns count of chunks aligned."""
```

Each chunk gets:
```python
# Translation chunk (e.g., di Giovanni):
structural_ref = {
    "canonical_section": "sol.being.measure.nodal-lines",
    "canonical_para": "sol.gw21.364.p1",
    "heading": "b. Nodal Lines of Measure-Relations",
    "translation_page": 318,
    "gw_page": "21.364",
    "para_group_index": 0,
    "alignment_confidence": 0.9,
}

# Original-language chunk (e.g., GW21):
structural_ref = {
    "canonical_section": "sol.being.measure.nodal-lines",
    "canonical_para": "sol.gw21.364.p1",
    "heading": "b. Knotenlinie von Maaßverhältnissen",
    "gw_page": "21.364",
}
```

### Step 4: Database Additions
**File:** `database.py`

```python
def update_chunk_structural_ref(self, chunk_id, structural_ref) -> None
def delete_source_and_chunks(self, source_id) -> int
def get_sources_by_work_id(self, work_id) -> list[Source]
def get_chunks_by_canonical_section(self, canonical_section) -> list[Chunk]
```

### Step 5: Source Diversity in Search Results
**File:** `search.py`

Current problem: with 460K chunks, top-10 results are dominated by whichever source mentions the query terms most densely (e.g., Houlgate's reader's guide for "nodal lines"). Lecture transcripts and primary texts are pushed out even though they contain the topic.

**Fix: source-type-aware result diversification.** After scoring, ensure the top-k results include representation from each source type present in the candidate set:

```python
def _diversify_results(results: list[SearchResult], top_k: int) -> list[SearchResult]:
    """Ensure results include different source types and editions.

    Strategy: round-robin across source types (transcript, pdf, bilingual),
    then across distinct sources within each type, always picking the
    highest-scoring chunk from each. Falls back to pure score ranking
    when diversity slots are exhausted.
    """
```

For `ec search "nodal lines" -k 10`, the diversified results might be:
- Top result from transcripts (Thompson discussing nodal lines)
- Top result from primary texts (di Giovanni SoL)
- Top result from secondary literature (Houlgate's guide)
- Top result from Radnik transcript
- Remaining 6 by pure score

This ensures a scholar always sees what the lecturers say alongside what the texts say.

### Step 6: Cross-Translation Search
**File:** `search.py`

Add to `SearchResult`:
```python
@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    source: Source
    paired_chunk: Chunk | None = None
    cross_translations: list[CrossRef] | None = None  # NEW

@dataclass
class CrossRef:
    """A corresponding passage in another translation of the same work."""
    edition_label: str       # "Miller 1977"
    chunk: Chunk
    source: Source
    confidence: float        # alignment confidence
```

When `bilingual=True` or a new `--compare` flag is set, search finds matching chunks from other editions of the same work using `canonical_section` + `alignment_position`.

### Step 6: CLI
```
ec ingest-edition <pdf> --profile di-giovanni-2010 --page-offset 282
ec ingest-edition <dir> --profile di-giovanni-2010 --recursive
ec list-profiles
ec search "nodal lines" --compare    # show cross-translation matches
ec fix-sources --remove-mismatched   # clean up wrong Zotero links
```

### Step 7: Re-ingest Science of Logic

1. Remove mismatched Zotero SoL sources
2. Ingest from `~/Documents/The Hegel Collection/`:
   - Full text: `Hegel.SoL.Complete.1812.DiGiovanni.2010.pdf` (offset TBD)
   - 13 section PDFs with their respective offsets
   - All with `work_id="hegel-sol"`, `edition_id="di-giovanni-2010"`
3. Also ingest the Miller translation with `edition_id="miller-1969"`
4. Verify cross-translation alignment

### Step 8: Tests & Verification

- Character-level extraction produces correct text with paragraph breaks
- GW references extracted from di Giovanni margin text
- Edition profiles detect correct reference patterns
- Cross-translation search finds matching passages
- `ec search "nodal lines of measures"` → Hegel primary text with `translation_page=318, gw_page=21.364`

---

## Key Design Decisions

1. **Work/Edition in metadata, not new tables**: Keeps the schema simple. `work_id` and `edition_id` in Source.metadata are sufficient for grouping and querying. A dedicated Works table can come later if needed.

2. **Canonical section IDs as the alignment backbone**: `sol.being.measure.nodal-lines` is the shared identifier across all translations. Paragraph-level alignment is a confidence-scored estimate on top.

3. **EditionProfile is the generalization**: Adding a new author/work/translation is just defining a new profile with the right regex patterns and section map. No code changes needed.

4. **Character-level extraction only for critical editions**: Regular PDFs still use pypdf (fast, good enough). pdfplumber is only invoked when an EditionProfile is specified — it's slower but captures margin annotations and font metadata.

5. **Layered alignment**: Section-level is guaranteed (from ToC structure). Paragraph-level is proportional estimate with explicit confidence. Users see the reliable layer by default.
