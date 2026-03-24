"""Tests for BGE-M3 embedding pipeline.

Uses a small model fixture for fast CI. The full BGE-M3 model test is
marked slow and only runs when --runslow is passed.
"""

import platform

import numpy as np
import pytest

from extracosmic_commons.embeddings import EmbeddingPipeline


@pytest.fixture
def embedder():
    """Use a small, fast model for unit tests instead of the 2.3GB BGE-M3."""
    return EmbeddingPipeline(model_name="sentence-transformers/all-MiniLM-L6-v2")


class TestEmbeddingPipeline:
    def test_embed_returns_vector(self, embedder):
        vec = embedder.embed("Hello world")
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32
        assert vec.ndim == 1

    def test_embed_dimension(self, embedder):
        vec = embedder.embed("Test")
        assert vec.shape[0] == embedder.dimension

    def test_embed_normalized(self, embedder):
        """Vectors should be L2-normalized (norm ≈ 1.0)."""
        vec = embedder.embed("Test normalization")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 0.01

    def test_embed_empty_string(self, embedder):
        """Empty strings return zero vector."""
        vec = embedder.embed("")
        assert np.allclose(vec, 0.0)

    def test_embed_deterministic(self, embedder):
        """Same input produces same output."""
        v1 = embedder.embed("Deterministic test")
        v2 = embedder.embed("Deterministic test")
        assert np.allclose(v1, v2)

    def test_embed_different_texts_differ(self, embedder):
        """Different texts produce different vectors."""
        v1 = embedder.embed("Hegel's Science of Logic")
        v2 = embedder.embed("Python programming tutorial")
        # Cosine similarity should be low
        sim = np.dot(v1, v2)
        assert sim < 0.9  # Not identical

    def test_embed_batch(self, embedder):
        texts = ["First text", "Second text", "Third text"]
        vecs = embedder.embed_batch(texts)
        assert vecs.shape == (3, embedder.dimension)
        assert vecs.dtype == np.float32

    def test_embed_batch_normalized(self, embedder):
        texts = ["A", "B", "C"]
        vecs = embedder.embed_batch(texts)
        norms = np.linalg.norm(vecs, axis=1)
        assert np.allclose(norms, 1.0, atol=0.01)

    def test_embed_batch_empty(self, embedder):
        vecs = embedder.embed_batch([])
        assert vecs.shape[0] == 0

    def test_embed_batch_consistent_with_single(self, embedder):
        """Batch embedding should produce same results as single."""
        text = "Consistency check"
        single = embedder.embed(text)
        batch = embedder.embed_batch([text])
        assert np.allclose(single, batch[0], atol=1e-5)

    @pytest.mark.skipif(
        platform.system() == "Darwin",
        reason="Progress callback not supported in macOS subprocess mode",
    )
    def test_embed_batch_with_progress(self, embedder):
        """Progress callback is called (non-macOS only)."""
        calls = []
        texts = ["A", "B", "C", "D", "E"]
        embedder.embed_batch(
            texts,
            batch_size=2,
            progress_callback=lambda done, total: calls.append((done, total)),
        )
        assert len(calls) > 0
        assert calls[-1] == (5, 5)  # Last call reports completion

    def test_lazy_loading(self):
        """Model isn't loaded until first use."""
        pipeline = EmbeddingPipeline()
        assert pipeline._model is None
        # Don't actually trigger loading (would download 2.3GB)
