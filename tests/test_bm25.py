"""Tests for BM25 index manager."""

import pytest

from extracosmic_commons.bm25 import BM25Index


@pytest.fixture
def sample_corpus():
    return {
        "c1": "Hegel discusses the transition from Being to Nothing in the opening of the Logic",
        "c2": "Aufheben means to cancel, preserve, and raise up simultaneously",
        "c3": "Kant's Critique of Pure Reason examines the conditions for knowledge",
        "c4": "The dialectical movement of Aufheben is central to Hegel's method",
        "c5": "Being, pure being – without further determination",
    }


@pytest.fixture
def index(sample_corpus):
    idx = BM25Index()
    idx.build(list(sample_corpus.keys()), list(sample_corpus.values()))
    return idx


class TestBM25Index:
    def test_build(self, index, sample_corpus):
        assert index.size == len(sample_corpus)

    def test_search_exact_term(self, index):
        """Exact term match should rank highest."""
        results = index.search("Aufheben")
        assert len(results) > 0
        # Chunks mentioning "Aufheben" should be top results
        top_ids = [cid for cid, _ in results[:2]]
        assert "c2" in top_ids or "c4" in top_ids

    def test_search_multi_term(self, index):
        results = index.search("Being Nothing transition")
        assert len(results) > 0
        # c1 mentions all three terms
        assert results[0][0] == "c1"

    def test_search_no_match(self, index):
        results = index.search("xyzzyx nonexistent")
        assert len(results) == 0  # All scores should be 0

    def test_search_top_k(self, index):
        results = index.search("Hegel", top_k=2)
        assert len(results) <= 2

    def test_search_empty_index(self):
        idx = BM25Index()
        assert idx.search("anything") == []

    def test_save_and_load(self, index, tmp_path):
        save_path = tmp_path / "bm25"
        index.save(save_path)

        loaded = BM25Index(index_path=save_path)
        assert loaded.size == index.size

        # Search should work on loaded index
        results = loaded.search("Aufheben")
        assert len(results) > 0

    def test_scores_positive(self, index):
        """All returned scores should be positive."""
        results = index.search("Being")
        for _, score in results:
            assert score > 0

    def test_build_mismatch_raises(self):
        idx = BM25Index()
        with pytest.raises(AssertionError):
            idx.build(["a", "b"], ["only one text"])
