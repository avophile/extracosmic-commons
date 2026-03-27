#!/usr/bin/env python3
"""Ingest all Wu conversation transcripts into Extracosmic Commons.

Reads LLM-cleaned JSON transcripts from the podcast-pipeline and ingests
them into the Extracosmic Commons database with FAISS embeddings, linking
each chunk to the original audio file with precise timestamps.

Usage:
    python scripts/ingest_wu_conversations.py [--dry-run]
"""

import sys
import os
import glob
import json
import argparse
from pathlib import Path

# Add the project src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from extracosmic_commons.database import Database
from extracosmic_commons.embeddings import EmbeddingPipeline
from extracosmic_commons.index import FAISSIndex
from extracosmic_commons.ingest.conversation import ConversationIngester


# ── Paths ────────────────────────────────────────────────────────────────
TRANSCRIPTS_DIR = Path("/Volumes/External SSD/podcast-pipeline/01-transcripts-llm-cleaned")
AUDIO_DIR = Path("/Volumes/External SSD/podcast-pipeline/00-raw")
DATA_DIR = Path(__file__).parent.parent / "data"

# ── Transcript filename → Audio filename mapping ─────────────────────────
# Built by comparing the two directory listings. The transcript filenames
# use underscores and ISO-ish dates (Wu_2025.08.03), while the audio
# filenames use spaces and US-format dates (Wu 8.3.2025).
AUDIO_MAP = {
    "Wu_2025.06.23":        "Wu 6.23.2025.m4a",
    "Wu_2025.07.07":        "Wu 7.7.2025.m4a",
    "Wu_2025.07.12":        "Wu 7.12.2025.m4a",
    "Wu_2025.07.19":        "Wu 7.19.2025.m4a",
    "Wu_2025.08.03":        "Wu 8.3.2025.m4a",
    "Wu_2025.08.03_2":      "Wu 8.3.2025 2.m4a",
    "Wu_2025.08.10":        "Wu 8.10.2025.m4a",
    "Wu_2025.08.17":        "Wu 8.17.2025.m4a",
    "Wu_2025.08.17_2":      "Wu 8.17.2025 2.m4a",
    "Wu_2025.08.24":        "Wu 8.24.2025.m4a",
    "Wu_2025.08.24_A":      "Wu 8.24.2025 A.m4a",
    "Wu_2025.08.31":        "Wu 8.31.2025.m4a",
    "Wu_2025.09.05_HeideggerK": "Wu 9.5.2025 Heidegger Kant Strauss.m4a",
    "Wu_2025.09.07":        "Wu 9.7.2025.m4a",
    "Wu_2025.09.15_1":      "Wu 2025.9.15 1.m4a",
    "Wu_2025.09.15_2":      "Wu 2025.9.15 2.m4a",
    "Wu_2025.09.20":        "Wu 2025.9.20.m4a",
    "Wu_2025.09.27":        "Wu 2025.9.27.m4a",
    "Wu_2025.10.04_1":      "Wu 2025.10.4 1.m4a",
    "Wu_2025.10.04_2":      "Wu 2025.10.4 2.m4a",
    "Wu_2025.10.11":        "Wu 2025.10.11.m4a",
    "Wu_2025.10.17":        "Wu 2025.10.17.m4a",
    "Wu_2025.10.25":        "Wu 2025.10.25.m4a",
    "Wu_2025.10.28":        "Wu 2025.10.28.m4a",
    "Wu_2025.10.29":        "Wu 2025.10.29.m4a",
    "Wu_2025.11.06":        "Wu 2025.11.6.m4a",
    "Wu_2025.11.13":        "Wu 2025.11.13.m4a",
    "Wu_2025.11.20":        "Wu 2025.11.20.m4a",
    "Wu_2025.12.06":        "Wu 2025.12.6.m4a",
    "Wu_2025.12.14":        "Wu 2025.12.14.m4a",
    "Wu_2025.12.29":        "Wu 2025.12.29.m4a",
    "Wu_2026.01.10":        "Wu 2026.1.10.m4a",
    "Wu_2026.03.01":        "Wu 2026.03.01.m4a",
    "Wu_2026.03.14":        "Wu 2026.03.14.m4a",
    "Wu_2026.03.15":        "Wu 2026.03.15.m4a",
    "Wu+Tony_2025.09.14":   "Wu and Tone 9.14.25.m4a",
    "Wu+Tony_2026.02.07":   "Wu and Tone 2026.02.07.m4a",
}


def main():
    parser = argparse.ArgumentParser(description="Ingest Wu conversations into Extracosmic Commons")
    parser.add_argument("--dry-run", action="store_true", help="Parse and report without writing to DB")
    args = parser.parse_args()

    # Find all Wu transcript JSONs
    json_files = sorted(glob.glob(str(TRANSCRIPTS_DIR / "Wu*.json")))
    print(f"Found {len(json_files)} Wu transcript files")

    if not json_files:
        print("No files found!")
        return

    # Initialize components (unless dry run)
    db = None
    embedder = None
    index = None

    if not args.dry_run:
        print("Loading database and embedding model...")
        db = Database(DATA_DIR / "extracosmic.db")
        embedder = EmbeddingPipeline()

        index_path = DATA_DIR / "faiss_index"
        if (index_path / "index.faiss").exists():
            index = FAISSIndex(index_path=index_path)
        else:
            index = FAISSIndex(dimension=1024)
        print("Components loaded.\n")

    ingester = ConversationIngester()
    total_chunks = 0
    total_sources = 0
    errors = []

    for jp in json_files:
        basename = Path(jp).stem  # e.g. "Wu_2026.03.01"

        # Look up audio file
        audio_filename = AUDIO_MAP.get(basename)
        if audio_filename:
            audio_path = str(AUDIO_DIR / audio_filename)
            if not Path(audio_path).exists():
                print(f"  WARNING: Audio file not found: {audio_path}")
                audio_path = None
        else:
            print(f"  WARNING: No audio mapping for {basename}")
            audio_path = None

        if args.dry_run:
            # Parse only — don't embed or store
            source, chunks = ingester.parse_conversation(Path(jp), audio_path)
            speakers = set(c.lecturer for c in chunks)
            print(f"  {basename}: {len(chunks)} chunks, speakers={speakers}, audio={'YES' if audio_path else 'NO'}")
            total_chunks += len(chunks)
            total_sources += 1
        else:
            # Check if already ingested (by title match)
            date_str = basename.replace("Wu_", "").replace("Wu+Tony_", "")
            existing_title = f"Wu Conversation — {date_str}"
            if "Tony" in basename:
                existing_title = f"Wu & Tony Conversation — {date_str}"

            if db.source_exists(existing_title):
                print(f"  SKIP (already ingested): {basename}")
                continue

            try:
                source = ingester.ingest(
                    json_path=Path(jp),
                    audio_path=audio_path,
                    db=db,
                    embedder=embedder,
                    index=index,
                )
                n_chunks = len(db.get_chunks_by_source(source.id))
                total_chunks += n_chunks
                total_sources += 1
                print(f"  OK: {basename} — {n_chunks} chunks, audio={'YES' if audio_path else 'NO'}")
            except Exception as e:
                errors.append((basename, str(e)))
                print(f"  ERROR: {basename} — {e}")

    # Save FAISS index if we wrote anything
    if not args.dry_run and index and total_sources > 0:
        index_path = DATA_DIR / "faiss_index"
        index_path.mkdir(parents=True, exist_ok=True)
        index.save(index_path)
        print(f"\nFAISS index saved to {index_path}")

    print(f"\n{'='*50}")
    print(f"{'DRY RUN ' if args.dry_run else ''}COMPLETE")
    print(f"  Sources ingested: {total_sources}")
    print(f"  Total chunks: {total_chunks}")
    if errors:
        print(f"  Errors: {len(errors)}")
        for name, err in errors:
            print(f"    {name}: {err}")


if __name__ == "__main__":
    main()
