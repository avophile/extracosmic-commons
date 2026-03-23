"""Tests for core data model."""

import json
from datetime import datetime, timezone

from extracosmic_commons.models import (
    Analysis,
    Chunk,
    SharingStatus,
    Source,
    SourceType,
)


class TestSource:
    def test_create_minimal(self):
        s = Source(title="Test", type=SourceType.PDF)
        assert s.title == "Test"
        assert s.type == SourceType.PDF
        assert len(s.id) == 36  # UUID4 format
        assert s.language == ["en"]
        assert s.sharing_status == SharingStatus.LOCAL_ONLY

    def test_create_with_metadata(self):
        s = Source(
            title="Science of Logic",
            type=SourceType.PDF,
            author=["Hegel", "di Giovanni"],
            language=["de", "en"],
            edition="Cambridge, 2010",
            metadata={"isbn": "978-0521832557", "year": 2010},
        )
        assert s.author == ["Hegel", "di Giovanni"]
        assert s.metadata["isbn"] == "978-0521832557"

    def test_roundtrip_dict(self):
        s = Source(
            title="Test Source",
            type=SourceType.TRANSCRIPT,
            author=["Thompson"],
            metadata={"lectures": 40},
        )
        d = s.to_dict()
        s2 = Source.from_dict(d)
        assert s2.id == s.id
        assert s2.title == s.title
        assert s2.type == s.type
        assert s2.author == s.author
        assert s2.metadata == s.metadata

    def test_enum_values(self):
        assert SourceType.PDF.value == "pdf"
        assert SourceType.TRANSCRIPT.value == "transcript"
        assert SourceType.BILINGUAL_PAIR.value == "bilingual_pair"
        assert SharingStatus.LOCAL_ONLY.value == "local_only"
        assert SharingStatus.SHARED_FULL.value == "shared_full"

    def test_serialized_json_fields(self):
        """Verify JSON fields serialize to strings for SQLite."""
        s = Source(title="X", type=SourceType.PDF, author=["A", "B"])
        d = s.to_dict()
        assert isinstance(d["author"], str)
        assert json.loads(d["author"]) == ["A", "B"]


class TestChunk:
    def test_create_minimal(self):
        c = Chunk(source_id="src-1", text="Hello world")
        assert c.source_id == "src-1"
        assert c.text == "Hello world"
        assert c.language == "en"
        assert c.structural_ref is None
        assert c.embedding is None

    def test_create_with_structural_ref(self):
        c = Chunk(
            source_id="src-1",
            text="Sein und Nichts",
            language="de",
            structural_ref={
                "section": "§132",
                "doctrine": "Being",
                "chapter": "Determinate Being",
                "gw_page": "21.97",
            },
        )
        assert c.structural_ref["section"] == "§132"
        assert c.structural_ref["doctrine"] == "Being"

    def test_structural_ref_different_schemas(self):
        """Different works can have different structural hierarchies."""
        hegel = Chunk(
            source_id="s1",
            text="...",
            structural_ref={"section": "§132", "doctrine": "Being"},
        )
        kant = Chunk(
            source_id="s2",
            text="...",
            structural_ref={"division": "Transcendental Analytic", "section": "§15"},
        )
        article = Chunk(
            source_id="s3",
            text="...",
            structural_ref={"section": "3.2", "heading": "Methodology"},
        )
        assert hegel.structural_ref["doctrine"] == "Being"
        assert kant.structural_ref["division"] == "Transcendental Analytic"
        assert article.structural_ref["heading"] == "Methodology"

    def test_create_lecture_chunk(self):
        c = Chunk(
            source_id="src-1",
            text="Thompson explains...",
            youtube_timestamp="01:23:45",
            youtube_url="https://youtube.com/watch?v=abc&t=5025",
            lecturer="Thompson",
            lecture_number=15,
            chunk_method="timestamp",
        )
        assert c.lecturer == "Thompson"
        assert c.youtube_timestamp == "01:23:45"

    def test_bilingual_pairing(self):
        de = Chunk(source_id="s1", text="Das Sein", language="de", id="de-1")
        en = Chunk(
            source_id="s2",
            text="Being",
            language="en",
            id="en-1",
            paired_chunk_id="de-1",
        )
        assert en.paired_chunk_id == de.id

    def test_roundtrip_dict(self):
        c = Chunk(
            source_id="src-1",
            text="Test text",
            structural_ref={"section": "§1", "doctrine": "Being"},
            pdf_page=42,
            chunk_method="page",
        )
        d = c.to_dict()
        c2 = Chunk.from_dict(d)
        assert c2.id == c.id
        assert c2.structural_ref == c.structural_ref
        assert c2.pdf_page == 42

    def test_dict_excludes_embedding(self):
        """Embedding is stored in FAISS, not serialized to dict."""
        import numpy as np

        c = Chunk(source_id="s1", text="t", embedding=np.zeros(1024))
        d = c.to_dict()
        assert "embedding" not in d

    def test_null_structural_ref_roundtrip(self):
        c = Chunk(source_id="s1", text="t", structural_ref=None)
        d = c.to_dict()
        assert d["structural_ref"] is None
        c2 = Chunk.from_dict(d)
        assert c2.structural_ref is None


class TestAnalysis:
    def test_create_minimal(self):
        a = Analysis(title="Note", content="# My analysis\n\nSome text.")
        assert a.title == "Note"
        assert a.linked_chunks == []
        assert a.tags == []

    def test_create_with_links(self):
        a = Analysis(
            title="Comparison",
            content="Thompson and Houlgate differ on...",
            linked_chunks=["chunk-1", "chunk-2"],
            linked_analyses=["analysis-1"],
            tags=["Being", "comparison"],
        )
        assert len(a.linked_chunks) == 2
        assert "comparison" in a.tags

    def test_roundtrip_dict(self):
        a = Analysis(
            title="Test",
            content="Content",
            tags=["a", "b"],
            linked_chunks=["c1"],
        )
        d = a.to_dict()
        a2 = Analysis.from_dict(d)
        assert a2.id == a.id
        assert a2.tags == a.tags
        assert a2.linked_chunks == a.linked_chunks

    def test_timestamps(self):
        a = Analysis(title="T", content="C")
        assert isinstance(a.created_at, datetime)
        assert a.created_at.tzinfo == timezone.utc
