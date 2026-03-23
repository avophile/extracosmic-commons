"""Tests for SQLite database layer."""

import pytest

from extracosmic_commons.database import Database
from extracosmic_commons.models import (
    Analysis,
    Chunk,
    Source,
    SourceType,
)


@pytest.fixture
def db(tmp_path):
    """Ephemeral database for each test."""
    d = Database(tmp_path / "test.db")
    yield d
    d.close()


@pytest.fixture
def sample_source():
    return Source(
        title="Science of Logic",
        type=SourceType.PDF,
        author=["Hegel", "di Giovanni"],
        language=["de", "en"],
        metadata={"year": 2010, "isbn": "978-0521832557"},
    )


@pytest.fixture
def sample_transcript_source():
    return Source(
        title="Thompson Lectures on SoL",
        type=SourceType.TRANSCRIPT,
        author=["Thompson"],
    )


class TestSchema:
    def test_init_schema_idempotent(self, db):
        """Calling init_schema twice doesn't raise."""
        db.init_schema()
        db.init_schema()

    def test_wal_mode(self, db):
        mode = db.conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"


class TestSourceCRUD:
    def test_insert_and_get(self, db, sample_source):
        db.insert_source(sample_source)
        fetched = db.get_source(sample_source.id)
        assert fetched is not None
        assert fetched.title == "Science of Logic"
        assert fetched.author == ["Hegel", "di Giovanni"]
        assert fetched.metadata["isbn"] == "978-0521832557"

    def test_get_nonexistent(self, db):
        assert db.get_source("nonexistent-id") is None

    def test_get_all_sources(self, db, sample_source, sample_transcript_source):
        db.insert_source(sample_source)
        db.insert_source(sample_transcript_source)
        all_sources = db.get_all_sources()
        assert len(all_sources) == 2

    def test_source_exists_by_title(self, db, sample_source):
        assert db.source_exists("Science of Logic") is False
        db.insert_source(sample_source)
        assert db.source_exists("Science of Logic") is True

    def test_source_exists_by_title_and_author(self, db, sample_source):
        db.insert_source(sample_source)
        assert db.source_exists("Science of Logic", ["Hegel", "di Giovanni"]) is True
        assert db.source_exists("Science of Logic", ["Miller"]) is False


class TestChunkCRUD:
    def test_insert_and_get_by_source(self, db, sample_source):
        db.insert_source(sample_source)
        chunks = [
            Chunk(source_id=sample_source.id, text="First chunk", paragraph_index=0),
            Chunk(source_id=sample_source.id, text="Second chunk", paragraph_index=1),
        ]
        db.insert_chunks_batch(chunks)
        fetched = db.get_chunks_by_source(sample_source.id)
        assert len(fetched) == 2
        assert fetched[0].text == "First chunk"
        assert fetched[1].text == "Second chunk"

    def test_insert_single(self, db, sample_source):
        db.insert_source(sample_source)
        c = Chunk(source_id=sample_source.id, text="Single chunk")
        db.insert_chunk(c)
        fetched = db.get_chunks_by_source(sample_source.id)
        assert len(fetched) == 1

    def test_get_by_ids(self, db, sample_source):
        db.insert_source(sample_source)
        c1 = Chunk(source_id=sample_source.id, text="A")
        c2 = Chunk(source_id=sample_source.id, text="B")
        c3 = Chunk(source_id=sample_source.id, text="C")
        db.insert_chunks_batch([c1, c2, c3])

        # Fetch in specific order
        fetched = db.get_chunks_by_ids([c3.id, c1.id])
        assert len(fetched) == 2
        assert fetched[0].id == c3.id
        assert fetched[1].id == c1.id

    def test_get_by_ids_empty(self, db):
        assert db.get_chunks_by_ids([]) == []

    def test_structural_ref_query(self, db, sample_source):
        db.insert_source(sample_source)
        c1 = Chunk(
            source_id=sample_source.id,
            text="Being passage",
            structural_ref={"doctrine": "Being", "section": "§132"},
        )
        c2 = Chunk(
            source_id=sample_source.id,
            text="Essence passage",
            structural_ref={"doctrine": "Essence", "section": "§200"},
        )
        c3 = Chunk(
            source_id=sample_source.id,
            text="No ref",
            structural_ref=None,
        )
        db.insert_chunks_batch([c1, c2, c3])

        being_chunks = db.get_chunks_by_structural_ref(doctrine="Being")
        assert len(being_chunks) == 1
        assert being_chunks[0].text == "Being passage"

        section_chunks = db.get_chunks_by_structural_ref(section="§200")
        assert len(section_chunks) == 1
        assert section_chunks[0].structural_ref["doctrine"] == "Essence"

    def test_structural_ref_multi_filter(self, db, sample_source):
        db.insert_source(sample_source)
        c = Chunk(
            source_id=sample_source.id,
            text="Specific",
            structural_ref={"doctrine": "Being", "section": "§132"},
        )
        db.insert_chunk(c)

        # Both filters match
        results = db.get_chunks_by_structural_ref(doctrine="Being", section="§132")
        assert len(results) == 1

        # One filter doesn't match
        results = db.get_chunks_by_structural_ref(doctrine="Being", section="§999")
        assert len(results) == 0

    def test_search_metadata_by_lecturer(self, db, sample_transcript_source):
        db.insert_source(sample_transcript_source)
        c = Chunk(
            source_id=sample_transcript_source.id,
            text="Thompson says...",
            lecturer="Thompson",
        )
        db.insert_chunk(c)

        results = db.search_metadata(lecturer="Thompson")
        assert len(results) == 1
        assert results[0].lecturer == "Thompson"

        results = db.search_metadata(lecturer="Houlgate")
        assert len(results) == 0

    def test_search_metadata_by_language(self, db, sample_source):
        db.insert_source(sample_source)
        c_de = Chunk(source_id=sample_source.id, text="Sein", language="de")
        c_en = Chunk(source_id=sample_source.id, text="Being", language="en")
        db.insert_chunks_batch([c_de, c_en])

        results = db.search_metadata(language="de")
        assert len(results) == 1
        assert results[0].text == "Sein"

    def test_search_metadata_by_source_type(self, db, sample_source, sample_transcript_source):
        db.insert_source(sample_source)
        db.insert_source(sample_transcript_source)
        db.insert_chunk(Chunk(source_id=sample_source.id, text="PDF chunk"))
        db.insert_chunk(Chunk(source_id=sample_transcript_source.id, text="Transcript chunk"))

        results = db.search_metadata(source_type="transcript")
        assert len(results) == 1
        assert results[0].text == "Transcript chunk"


class TestAnalysisCRUD:
    def test_insert_and_get(self, db):
        a = Analysis(
            title="My Note",
            content="# Analysis\n\nSome thoughts.",
            tags=["Being", "comparison"],
            linked_chunks=["chunk-1"],
        )
        db.insert_analysis(a)
        fetched = db.get_analysis(a.id)
        assert fetched is not None
        assert fetched.title == "My Note"
        assert fetched.tags == ["Being", "comparison"]
        assert fetched.linked_chunks == ["chunk-1"]

    def test_get_nonexistent(self, db):
        assert db.get_analysis("nonexistent") is None


class TestStats:
    def test_empty_stats(self, db):
        stats = db.get_stats()
        assert stats["sources"] == 0
        assert stats["chunks"] == 0
        assert stats["analyses"] == 0

    def test_populated_stats(self, db, sample_source, sample_transcript_source):
        db.insert_source(sample_source)
        db.insert_source(sample_transcript_source)
        db.insert_chunks_batch([
            Chunk(source_id=sample_source.id, text="A", language="de"),
            Chunk(source_id=sample_source.id, text="B", language="en"),
            Chunk(
                source_id=sample_transcript_source.id,
                text="C",
                lecturer="Thompson",
            ),
        ])
        db.insert_analysis(Analysis(title="T", content="C"))

        stats = db.get_stats()
        assert stats["sources"] == 2
        assert stats["chunks"] == 3
        assert stats["analyses"] == 1
        assert stats["sources_by_type"]["pdf"] == 1
        assert stats["sources_by_type"]["transcript"] == 1
        assert stats["chunks_by_language"]["de"] == 1
        assert stats["chunks_by_language"]["en"] == 2
        assert stats["chunks_by_lecturer"]["Thompson"] == 1
