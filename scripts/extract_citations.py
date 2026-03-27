#!/usr/bin/env python3
"""Extract structured citation records from all transcripts in Extracosmic Commons.

Processes both Wu conversation transcripts (from JSON files) and lecture
transcripts (Thompson, Houlgate, Radnik — from DB chunks), identifying
every moment where a speaker reads from or references a primary text.

Uses regex/heuristic pattern matching (zero API cost). Extracted citations
are stored in the citations table and optionally cross-referenced against
existing source text chunks via FAISS semantic search.

Usage:
    python scripts/extract_citations.py                    # All sources
    python scripts/extract_citations.py --conversations    # Wu conversations only
    python scripts/extract_citations.py --lectures         # Lectures only
    python scripts/extract_citations.py --file Wu_2025.07.07  # Single file
    python scripts/extract_citations.py --dry-run          # Count matches without storing
"""

import sys
import argparse
import glob
import json
from pathlib import Path
from datetime import datetime

# Add project src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from extracosmic_commons.database import Database
from extracosmic_commons.ingest.citation_extractor import (
    CitationRecord,
    extract_citations_from_transcript,
    extract_citations_from_lecture,
    cross_reference_citations,
)

# ── Paths ────────────────────────────────────────────────────────────────
TRANSCRIPTS_DIR = Path("/Volumes/External SSD/podcast-pipeline/01-transcripts-llm-cleaned")
DATA_DIR = Path(__file__).parent.parent / "data"

# ── Lecture sources to extract from ──────────────────────────────────────
# Maps source title substring → lecturer name for speaker attribution.
LECTURE_SOURCES = {
    "Kevin Thompson": "Thompson",
    "Stephen Houlgate": "Houlgate",
    "Radnik": "Radnik",
}


def process_conversations(db, embedder=None, index=None,
                          file_filter=None, dry_run=False):
    """Extract citations from Wu conversation JSON transcripts.

    Scans each transcript's segments for page references, reading
    indicators, and work title mentions using regex/heuristic patterns.
    Zero API cost — all processing is local.

    Args:
        db: Database instance.
        embedder: EmbeddingPipeline for cross-referencing (optional).
        index: FAISSIndex for cross-referencing (optional).
        file_filter: If set, only process files matching this basename.
        dry_run: If True, extract but don't store in database.
    """
    json_files = sorted(glob.glob(str(TRANSCRIPTS_DIR / "Wu*.json")))
    print(f"Found {len(json_files)} Wu conversation transcripts")

    if file_filter:
        json_files = [f for f in json_files if file_filter in Path(f).stem]
        print(f"  Filtered to {len(json_files)} matching '{file_filter}'")

    total_citations = 0
    for jp in json_files:
        basename = Path(jp).stem
        print(f"\n--- {basename} ---")

        # Load transcript to report segment count
        with open(jp, "r") as f:
            data = json.load(f)
        seg_count = len(data.get("segments", []))

        # Extract citations using heuristic engine
        citations = extract_citations_from_transcript(json_path=Path(jp))
        print(f"  {seg_count} segments → {len(citations)} citations found")

        if dry_run:
            # Print sample citations for review
            for c in citations[:5]:
                pg = c.page_english or c.page_german or "no page"
                print(f"    [{c.citation_type.value}] {c.work_title or '?'} "
                      f"p.{pg} @{float(c.audio_timestamp):.0f}s "
                      f"conf={c.confidence:.2f}")
            if len(citations) > 5:
                print(f"    ... and {len(citations) - 5} more")
            continue

        # Cross-reference against corpus if embedder available
        if embedder and index and citations:
            citations = cross_reference_citations(
                citations, db, embedder, index
            )

        # Store in database
        if citations:
            dicts = [c.to_dict() for c in citations]
            n = db.insert_citations_batch(dicts)
            print(f"  Stored {n} citations in database")
            total_citations += n

    print(f"\nConversations total: {total_citations} citations extracted")


def process_lectures(db, embedder=None, index=None, dry_run=False):
    """Extract citations from lecture transcripts already in the database.

    Finds Thompson, Houlgate, and Radnik sources by title matching,
    then runs heuristic extraction on their chunks.
    """
    all_sources = db.get_all_sources()
    total_citations = 0

    for source in all_sources:
        lecturer_name = None
        for title_substr, name in LECTURE_SOURCES.items():
            if title_substr in source.title:
                lecturer_name = name
                break

        if not lecturer_name:
            continue

        chunks = db.get_chunks_by_source(source.id)
        print(f"\n--- {source.title} ({lecturer_name}) ---")
        print(f"  {len(chunks)} chunks")

        citations = extract_citations_from_lecture(
            source_id=source.id,
            db=db,
            lecturer_name=lecturer_name,
        )
        print(f"  → {len(citations)} citations found")

        if dry_run:
            for c in citations[:5]:
                pg = c.page_english or c.page_german or "no page"
                print(f"    [{c.citation_type.value}] {c.work_title or '?'} "
                      f"p.{pg} conf={c.confidence:.2f}")
            continue

        # Cross-reference against corpus
        if embedder and index and citations:
            citations = cross_reference_citations(
                citations, db, embedder, index
            )

        if citations:
            dicts = [c.to_dict() for c in citations]
            n = db.insert_citations_batch(dicts)
            print(f"  Stored {n} citations in database")
            total_citations += n

    print(f"\nLectures total: {total_citations} citations extracted")


def main():
    parser = argparse.ArgumentParser(
        description="Extract citations from transcripts (regex/heuristic, zero API cost)"
    )
    parser.add_argument("--conversations", action="store_true",
                        help="Process Wu conversations only")
    parser.add_argument("--lectures", action="store_true",
                        help="Process lectures only")
    parser.add_argument("--file", type=str, default=None,
                        help="Process a single file matching this basename")
    parser.add_argument("--dry-run", action="store_true",
                        help="Extract and display without storing in database")
    parser.add_argument("--no-crossref", action="store_true",
                        help="Skip corpus cross-referencing (faster, no embeddings)")
    args = parser.parse_args()

    print("=" * 60)
    print("Extracosmic Commons — Citation Extraction (Heuristic)")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"  Cross-referencing: {'OFF' if args.no_crossref else 'ON'}")
    print("=" * 60)

    db = Database(DATA_DIR / "extracosmic.db")

    # Load embedder/index for cross-referencing (unless disabled)
    embedder = None
    index = None
    if not args.no_crossref and not args.dry_run:
        try:
            from extracosmic_commons.embeddings import EmbeddingPipeline
            from extracosmic_commons.index import FAISSIndex
            print("Loading embedding model for cross-referencing...")
            embedder = EmbeddingPipeline()
            index_path = DATA_DIR / "faiss_index"
            if (index_path / "index.faiss").exists():
                index = FAISSIndex(index_path=index_path)
                print("  Embedder and FAISS index loaded.")
            else:
                print("  WARNING: No FAISS index found. Skipping cross-referencing.")
                embedder = None
        except Exception as e:
            print(f"  WARNING: Could not load embedder: {e}")
            print("  Skipping cross-referencing.")

    # Determine what to process
    do_conversations = not args.lectures  # default: both
    do_lectures = not args.conversations  # default: both

    if do_conversations:
        process_conversations(
            db, embedder, index,
            file_filter=args.file, dry_run=args.dry_run,
        )

    if do_lectures and not args.file:
        process_lectures(
            db, embedder, index, dry_run=args.dry_run,
        )

    # Final stats
    if not args.dry_run:
        stats = db.get_citation_stats()
        print(f"\n{'='*60}")
        print(f"Citation corpus: {stats['total_citations']} total")
        for work, count in stats.get("by_work", {}).items():
            print(f"  {work}: {count}")
        print(f"  Cross-referenced to source text: {stats['cross_referenced']}")
        print(f"{'='*60}")

    # Notification to Drafts
    drafts_dir = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/Drafts"
    if drafts_dir.exists() and not args.dry_run:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{timestamp}] Citation extraction complete."
        if not args.dry_run:
            stats = db.get_citation_stats()
            msg += f" {stats['total_citations']} citations in corpus."
        fname = f"citation_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(drafts_dir / fname, "w") as f:
            f.write(msg)
        print(f"  Notification saved to Drafts: {fname}")


if __name__ == "__main__":
    main()
