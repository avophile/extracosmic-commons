"""Tests for transcript ingestion pipeline."""

import pytest

from extracosmic_commons.ingest.transcript import (
    TranscriptIngester,
    _extract_header_metadata,
    _split_into_chunks,
)
from extracosmic_commons.models import SourceType

SAMPLE_TRANSCRIPT = """# Hegel's Science of Logic — Complete Lecture Transcripts

**Lecturer:** Kevin Thompson, DePaul University
**Course Period:** September 2024 – March 2025
**Source:** [YouTube Playlist](https://www.youtube.com/playlist?list=PLZDJ2sJwRHVPamkOKvadG8hKK6jGhZw5S)
**Total Sessions:** 40 lectures (~56 hours)

---

<a id="session-1a"></a>
## Session 1a — Historical Background & Orientation

[Watch on YouTube](https://youtube.com/watch?v=abc) | *Timestamps directly linked*

**[00:29:16](https://youtube.com/watch?v=abc&t=1756)** You can't understand any of it without at least now he may be deluded, right? Let's always remember Hegel's not right because Hegel says it. You'll say that a lot when you read this book.

**[00:29:34](https://youtube.com/watch?v=abc&t=1774)** There are people all over the world, believe it or not, that are great fans of the Logic. They think it's the most important book in the history of philosophy. And there are others who think it's the worst book ever written.

<a id="session-2a"></a>
## Session 2a — Being, Nothing, Becoming

**[00:00:13](https://youtube.com/watch?v=def&t=13)** So today we're going to start with the very beginning of the Science of Logic. Hegel begins with pure being.
"""


@pytest.fixture
def sample_transcript(tmp_path):
    p = tmp_path / "thompson_transcript.md"
    p.write_text(SAMPLE_TRANSCRIPT)
    return p


class TestHeaderMetadata:
    def test_extract_lecturer(self):
        meta = _extract_header_metadata(SAMPLE_TRANSCRIPT)
        assert meta["lecturer"] == "Kevin Thompson, DePaul University"

    def test_extract_session_count(self):
        meta = _extract_header_metadata(SAMPLE_TRANSCRIPT)
        assert meta["lecture_count"] == 40

    def test_extract_playlist_url(self):
        meta = _extract_header_metadata(SAMPLE_TRANSCRIPT)
        assert "youtube.com/playlist" in meta["playlist_url"]


class TestChunkSplitting:
    def test_basic_splitting(self):
        chunks = _split_into_chunks(SAMPLE_TRANSCRIPT, source_id="test", lecturer="Thompson")
        assert len(chunks) == 3  # Three timestamp markers

    def test_chunk_has_timestamp(self):
        chunks = _split_into_chunks(SAMPLE_TRANSCRIPT, source_id="test", lecturer="Thompson")
        assert chunks[0].youtube_timestamp == "00:29:16"
        assert chunks[1].youtube_timestamp == "00:29:34"

    def test_chunk_has_url(self):
        chunks = _split_into_chunks(SAMPLE_TRANSCRIPT, source_id="test", lecturer="Thompson")
        assert "youtube.com" in chunks[0].youtube_url

    def test_chunk_has_lecturer(self):
        chunks = _split_into_chunks(SAMPLE_TRANSCRIPT, source_id="test", lecturer="Thompson")
        assert chunks[0].lecturer == "Thompson"

    def test_chunk_has_lecture_number(self):
        chunks = _split_into_chunks(SAMPLE_TRANSCRIPT, source_id="test", lecturer="Thompson")
        assert chunks[0].lecture_number == 1  # Session 1a
        assert chunks[2].lecture_number == 2  # Session 2a

    def test_chunk_method(self):
        chunks = _split_into_chunks(SAMPLE_TRANSCRIPT, source_id="test", lecturer="Thompson")
        assert all(c.chunk_method == "timestamp" for c in chunks)

    def test_short_chunks_skipped(self):
        """Chunks shorter than min_chunk_chars are skipped."""
        short = '**[00:00:01](http://x.com)** Hi.\n\n**[00:00:02](http://x.com)** Yes.\n'
        chunks = _split_into_chunks(short, source_id="test", lecturer="T", min_chunk_chars=100)
        assert len(chunks) == 0


class TestTranscriptIngester:
    def test_parse_transcript(self, sample_transcript):
        ingester = TranscriptIngester()
        source, chunks = ingester.parse_transcript(sample_transcript)

        assert source.type == SourceType.TRANSCRIPT
        assert "Thompson" in source.author[0]
        assert len(chunks) == 3
        assert all(c.source_id == source.id for c in chunks)
