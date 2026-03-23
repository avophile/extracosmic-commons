"""Tests for FAISS index manager."""

import numpy as np
import pytest

from extracosmic_commons.index import FAISSIndex


@pytest.fixture
def index():
    """Empty FAISS index with small dimension for fast tests."""
    return FAISSIndex(dimension=64)


def random_vec(dim=64):
    """Generate a random L2-normalized vector."""
    v = np.random.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


class TestFAISSIndex:
    def test_empty_index(self, index):
        assert index.size == 0

    def test_add_single(self, index):
        index.add("chunk-1", random_vec())
        assert index.size == 1

    def test_add_batch(self, index):
        ids = [f"chunk-{i}" for i in range(10)]
        vecs = np.vstack([random_vec() for _ in range(10)])
        index.add_batch(ids, vecs)
        assert index.size == 10

    def test_search_empty(self, index):
        results = index.search(random_vec())
        assert results == []

    def test_search_returns_nearest(self, index):
        # Add a known vector and a random one
        target = random_vec()
        index.add("target", target)
        index.add("other", random_vec())

        # Search for the target itself — should be top result
        results = index.search(target, top_k=2)
        assert len(results) == 2
        assert results[0][0] == "target"
        assert results[0][1] > results[1][1]  # Higher score

    def test_search_top_k(self, index):
        for i in range(20):
            index.add(f"chunk-{i}", random_vec())

        results = index.search(random_vec(), top_k=5)
        assert len(results) == 5

    def test_search_top_k_exceeds_size(self, index):
        index.add("only-one", random_vec())
        results = index.search(random_vec(), top_k=10)
        assert len(results) == 1

    def test_search_score_range(self, index):
        """Cosine similarity via inner product should be in [-1, 1]."""
        for i in range(5):
            index.add(f"c-{i}", random_vec())

        results = index.search(random_vec(), top_k=5)
        for _, score in results:
            assert -1.0 <= score <= 1.0 + 1e-6

    def test_save_and_load(self, index, tmp_path):
        ids = ["a", "b", "c"]
        vecs = np.vstack([random_vec() for _ in range(3)])
        index.add_batch(ids, vecs)

        save_path = tmp_path / "faiss_index"
        index.save(save_path)

        # Load into a new index
        loaded = FAISSIndex(index_path=save_path)
        assert loaded.size == 3

        # Search should work on loaded index
        results = loaded.search(vecs[0], top_k=1)
        assert results[0][0] == "a"

    def test_save_creates_directory(self, index, tmp_path):
        index.add("x", random_vec())
        nested = tmp_path / "a" / "b" / "index"
        index.save(nested)
        assert (nested / "index.faiss").exists()
        assert (nested / "chunk_ids.json").exists()

    def test_add_batch_empty(self, index):
        """Adding empty batch is a no-op."""
        index.add_batch([], np.zeros((0, 64), dtype=np.float32))
        assert index.size == 0

    def test_batch_id_vector_mismatch_raises(self, index):
        with pytest.raises(AssertionError):
            index.add_batch(["a", "b"], np.vstack([random_vec()]))
