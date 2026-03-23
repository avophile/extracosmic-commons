"""Tests for the search engine."""

import numpy as np
import pytest

from extracosmic_commons.database import Database
from extracosmic_commons.embeddings import EmbeddingPipeline
from extracosmic_commons.index import FAISSIndex
from extracosmic_commons.models import Chunk, Source, SourceType
from extracosmic_commons.search import SearchEngine


@pytest.fixture
def embedder():
    return EmbeddingPipeline(model_name="sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture
def populated_system(tmp_path, embedder):
    """A fully populated system with sources, chunks, and index."""
    db = Database(tmp_path / "test.db")
    index = FAISSIndex(dimension=embedder.dimension)

    # Create sources
    transcript_src = Source(
        title="Thompson SoL Lectures",
        type=SourceType.TRANSCRIPT,
        author=["Thompson"],
    )
    pdf_src = Source(
        title="Science of Logic",
        type=SourceType.PDF,
        author=["Hegel"],
    )
    db.insert_source(transcript_src)
    db.insert_source(pdf_src)

    # Create chunks with meaningful text
    chunks = [
        Chunk(
            source_id=transcript_src.id,
            text="The transition from Being to Nothing is the fundamental move in Hegel's Logic. Pure being and pure nothing are the same.",
            lecturer="Thompson",
            lecture_number=1,
            chunk_method="timestamp",
        ),
        Chunk(
            source_id=transcript_src.id,
            text="Aufheben means to cancel, preserve, and raise up simultaneously. It is the key dialectical movement.",
            lecturer="Thompson",
            lecture_number=5,
            chunk_method="timestamp",
        ),
        Chunk(
            source_id=pdf_src.id,
            text="Being, pure being – without further determination. In its indeterminate immediacy it is equal only to itself.",
            language="en",
            pdf_page=68,
            chunk_method="page",
        ),
        Chunk(
            source_id=pdf_src.id,
            text="Sein, reines Sein – ohne alle weitere Bestimmung.",
            language="de",
            pdf_page=68,
            chunk_method="page",
            structural_ref={"doctrine": "Being", "gw_page": "21.68"},
        ),
    ]

    # Set up bilingual pairing
    chunks[2].paired_chunk_id = chunks[3].id
    chunks[3].paired_chunk_id = chunks[2].id

    # Embed and index
    texts = [c.text for c in chunks]
    embeddings = embedder.embed_batch(texts)

    db.insert_chunks_batch(chunks)
    index.add_batch([c.id for c in chunks], embeddings)

    return db, index, embedder, chunks


class TestSearchEngine:
    def test_basic_search(self, populated_system):
        db, index, embedder, _ = populated_system
        engine = SearchEngine(db, embedder, index)

        results = engine.search("Being and Nothing")
        assert len(results) > 0
        # Most relevant should be about Being/Nothing
        assert "Being" in results[0].chunk.text or "Sein" in results[0].chunk.text

    def test_search_returns_source(self, populated_system):
        db, index, embedder, _ = populated_system
        engine = SearchEngine(db, embedder, index)

        results = engine.search("dialectical movement Aufheben")
        assert len(results) > 0
        assert results[0].source is not None
        assert results[0].source.title is not None

    def test_filter_by_lecturer(self, populated_system):
        db, index, embedder, _ = populated_system
        engine = SearchEngine(db, embedder, index)

        results = engine.search("Being", lecturer="Thompson")
        assert all(r.chunk.lecturer == "Thompson" for r in results)

    def test_filter_by_language(self, populated_system):
        db, index, embedder, _ = populated_system
        engine = SearchEngine(db, embedder, index)

        results = engine.search("Sein", language="de")
        assert all(r.chunk.language == "de" for r in results)

    def test_filter_by_source_type(self, populated_system):
        db, index, embedder, _ = populated_system
        engine = SearchEngine(db, embedder, index)

        results = engine.search("Being", source_type="transcript")
        for r in results:
            assert r.source.type == SourceType.TRANSCRIPT

    def test_bilingual_pair_resolution(self, populated_system):
        db, index, embedder, _ = populated_system
        engine = SearchEngine(db, embedder, index)

        results = engine.search("pure being indeterminate", bilingual=True)
        # Find the result that has a paired chunk
        paired_results = [r for r in results if r.paired_chunk is not None]
        assert len(paired_results) > 0

    def test_top_k_limits_results(self, populated_system):
        db, index, embedder, _ = populated_system
        engine = SearchEngine(db, embedder, index)

        results = engine.search("Hegel", top_k=2)
        assert len(results) <= 2

    def test_empty_corpus_returns_empty(self, tmp_path, embedder):
        db = Database(tmp_path / "empty.db")
        index = FAISSIndex(dimension=embedder.dimension)
        engine = SearchEngine(db, embedder, index)

        results = engine.search("anything")
        assert results == []

    def test_structural_ref_filter(self, populated_system):
        db, index, embedder, _ = populated_system
        engine = SearchEngine(db, embedder, index)

        results = engine.search(
            "Sein", structural_ref={"doctrine": "Being"}
        )
        for r in results:
            assert r.chunk.structural_ref is not None
            assert r.chunk.structural_ref["doctrine"] == "Being"
