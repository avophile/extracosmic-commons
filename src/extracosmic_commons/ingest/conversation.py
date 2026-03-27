"""Conversation transcript ingestion pipeline.

Parses JSON transcripts from the podcast-pipeline project. These are
diarized conversation recordings (Wu philosophical discussions) with
speaker labels and precise timestamps, enabling search results to link
directly to the exact point in the original audio file.

Input format (LLM-cleaned JSON from podcast-pipeline):
    {
        "text": "full transcript text...",
        "segments": [
            {"start": 0.251, "end": 8.164, "text": "...", "speaker": "SPEAKER_01"},
            ...
        ],
        "language": "en",
        "duration": 13051.2,
        "speakers": ["SPEAKER_00", "SPEAKER_01", ...],
        "llm_cleaned": true,
        "terminology_corrected": true
    }

Each chunk groups consecutive segments by the same speaker (up to a max
duration/length), preserving natural conversational turns. This means
search results return coherent speech turns rather than arbitrary text
windows.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..database import Database
from ..embeddings import EmbeddingPipeline
from ..index import FAISSIndex
from ..models import Chunk, Source, SourceType


# Default speaker name mapping for Wu conversations.
# SPEAKER_00 is typically Douglas, SPEAKER_01/02 are typically Wu.
# Can be overridden per-file via speaker_map parameter.
DEFAULT_SPEAKER_MAP = {
    "SPEAKER_00": "Douglas",
    "SPEAKER_01": "Wu",
    "SPEAKER_02": "Wu",
    "SPEAKER_03": "Wu",
    "SPEAKER_04": "Wu",
    "SPEAKER_05": "Wu",
    "SPEAKER_06": "Wu",
    "UNKNOWN": "Unknown",
}

# For Wu+Tony files, Tony is typically a third speaker
TONY_SPEAKER_MAP = {
    "SPEAKER_00": "Douglas",
    "SPEAKER_01": "Tony",
    "SPEAKER_02": "Wu",
    "SPEAKER_03": "Wu",
    "SPEAKER_04": "Wu",
    "UNKNOWN": "Unknown",
}


def _seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds (float) to HH:MM:SS format for media linking."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _extract_date_from_filename(filename: str) -> str | None:
    """Extract date string from filename like Wu_2026.03.01.json -> 2026.03.01."""
    m = re.search(r'(\d{4}\.\d{2}\.\d{2})', filename)
    return m.group(1) if m else None


def _group_segments_by_speaker_turns(
    segments: list[dict],
    speaker_map: dict[str, str],
    max_turn_chars: int = 2000,
    min_turn_chars: int = 50,
) -> list[dict[str, Any]]:
    """Group consecutive segments by the same speaker into conversational turns.

    This preserves natural speech flow — a single turn is one person talking
    until another person speaks. Long monologues are split at sentence boundaries
    to keep chunks under max_turn_chars.

    Returns list of dicts with keys: speaker, speaker_raw, text, start, end.
    """
    if not segments:
        return []

    turns = []
    current_speaker_raw = segments[0].get("speaker", "UNKNOWN")
    current_speaker = speaker_map.get(current_speaker_raw, current_speaker_raw)
    current_text = segments[0].get("text", "").strip()
    current_start = segments[0].get("start", 0)
    current_end = segments[0].get("end", 0)

    for seg in segments[1:]:
        seg_speaker_raw = seg.get("speaker", "UNKNOWN")
        seg_speaker = speaker_map.get(seg_speaker_raw, seg_speaker_raw)
        seg_text = seg.get("text", "").strip()
        seg_end = seg.get("end", 0)

        # Same speaker and not too long — append to current turn
        if seg_speaker == current_speaker and len(current_text) + len(seg_text) < max_turn_chars:
            current_text += " " + seg_text
            current_end = seg_end
        else:
            # Flush current turn
            if len(current_text) >= min_turn_chars:
                turns.append({
                    "speaker": current_speaker,
                    "speaker_raw": current_speaker_raw,
                    "text": current_text.strip(),
                    "start": current_start,
                    "end": current_end,
                })
            # Start new turn
            current_speaker_raw = seg_speaker_raw
            current_speaker = seg_speaker
            current_text = seg_text
            current_start = seg.get("start", 0)
            current_end = seg_end

    # Flush final turn
    if len(current_text) >= min_turn_chars:
        turns.append({
            "speaker": current_speaker,
            "speaker_raw": current_speaker_raw,
            "text": current_text.strip(),
            "start": current_start,
            "end": current_end,
        })

    return turns


def _turns_to_chunks(
    turns: list[dict],
    source_id: str,
    audio_path: str | None = None,
) -> list[Chunk]:
    """Convert speaker turns into Chunk objects with timestamp metadata.

    Each chunk stores:
    - text: the speaker's words for this turn
    - lecturer: the speaker name (Douglas, Wu, Tony, etc.)
    - youtube_timestamp: HH:MM:SS format for media seeking
    - youtube_url: not used (local audio), but could store file:// URI
    - paragraph_index: sequential position in the conversation
    """
    chunks = []
    for i, turn in enumerate(turns):
        timestamp = _seconds_to_timestamp(turn["start"])

        chunk = Chunk(
            source_id=source_id,
            text=turn["text"],
            language="en",
            youtube_timestamp=timestamp,
            youtube_url=audio_path,  # Store audio file path here for open-video API
            paragraph_index=i,
            lecturer=turn["speaker"],
            chunk_method="speaker_turn",
            structural_ref={
                "speaker": turn["speaker"],
                "speaker_raw": turn["speaker_raw"],
                "start_seconds": turn["start"],
                "end_seconds": turn["end"],
            },
        )
        chunks.append(chunk)

    return chunks


class ConversationIngester:
    """Ingests diarized conversation transcripts from the podcast-pipeline.

    Reads LLM-cleaned JSON files, groups segments into speaker turns,
    and creates searchable chunks linked to audio timestamps.
    """

    def __init__(self, speaker_map: dict[str, str] | None = None):
        """Initialize with optional custom speaker mapping.

        Args:
            speaker_map: Override the default SPEAKER_XX -> name mapping.
                         If None, uses DEFAULT_SPEAKER_MAP (or TONY_SPEAKER_MAP
                         for files containing 'Tony' in the name).
        """
        self.speaker_map = speaker_map

    def _get_speaker_map(self, filename: str) -> dict[str, str]:
        """Get the appropriate speaker map for a given file."""
        if self.speaker_map:
            return self.speaker_map
        if "Tony" in filename or "Tone" in filename:
            return TONY_SPEAKER_MAP
        return DEFAULT_SPEAKER_MAP

    def parse_conversation(
        self,
        json_path: Path,
        audio_path: str | None = None,
    ) -> tuple[Source, list[Chunk]]:
        """Parse a conversation JSON file into Source + Chunks.

        Args:
            json_path: Path to the LLM-cleaned JSON transcript.
            audio_path: Path to the original .m4a audio file on disk.
                        Used for timestamp-linked playback from search results.

        Returns:
            (source, chunks) without embedding or persisting.
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        filename = json_path.stem  # e.g. "Wu_2026.03.01"
        date_str = _extract_date_from_filename(filename)
        segments = data.get("segments", [])
        duration = data.get("duration", 0)
        speakers_raw = data.get("speakers", [])

        speaker_map = self._get_speaker_map(filename)
        speakers_named = sorted(set(
            speaker_map.get(s, s) for s in speakers_raw
        ))

        # Build descriptive title
        participants = " & ".join(speakers_named) if speakers_named else "Unknown"
        title = f"Wu Conversation — {date_str}" if date_str else f"Wu Conversation — {filename}"
        if "Tony" in filename or "Tone" in filename:
            title = f"Wu & Tony Conversation — {date_str}"

        source = Source(
            title=title,
            type=SourceType.TRANSCRIPT,
            author=speakers_named,
            source_path=audio_path or str(json_path),
            metadata={
                "conversation_date": date_str,
                "duration_seconds": duration,
                "duration_minutes": round(duration / 60, 1) if duration else None,
                "segment_count": len(segments),
                "speakers_raw": speakers_raw,
                "speakers_named": speakers_named,
                "llm_cleaned": data.get("llm_cleaned", False),
                "terminology_corrected": data.get("terminology_corrected", False),
                "transcript_json": str(json_path),
                "source_type": "conversation",
            },
        )

        # Group segments into speaker turns, then convert to chunks
        turns = _group_segments_by_speaker_turns(segments, speaker_map)
        chunks = _turns_to_chunks(turns, source.id, audio_path)

        return source, chunks

    def ingest(
        self,
        json_path: Path,
        audio_path: str | None,
        db: Database,
        embedder: EmbeddingPipeline,
        index: FAISSIndex,
    ) -> Source:
        """Parse, embed, and store a conversation transcript.

        Full pipeline: parse → embed batch → store in DB + FAISS.
        """
        source, chunks = self.parse_conversation(json_path, audio_path)

        if not chunks:
            raise ValueError(f"No chunks extracted from {json_path}")

        # Embed all chunks
        texts = [c.text for c in chunks]
        embeddings = embedder.embed_batch(texts)

        # Store in database
        db.insert_source(source)
        db.insert_chunks_batch(chunks)

        # Add to FAISS index
        chunk_ids = [c.id for c in chunks]
        index.add_batch(chunk_ids, embeddings)

        return source
