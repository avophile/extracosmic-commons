"""FAISS vector index manager for Extracosmic Commons.

Manages a flat inner-product FAISS index for semantic search. Vectors
are L2-normalized at embedding time, so inner product equals cosine
similarity.

Ported from the scholarly workbench's indexer_faiss.py pattern, adapted
for the Extracosmic Commons data model (chunk IDs as primary keys rather
than positional metadata).
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


class FAISSIndex:
    """FAISS index with chunk ID mapping.

    Maintains a parallel list of chunk IDs so that FAISS search results
    can be mapped back to Chunk records in the SQLite database.
    """

    def __init__(self, dimension: int = 1024, index_path: Path | None = None):
        """Create a new index or load from disk.

        Args:
            dimension: Vector dimension (1024 for BGE-M3).
            index_path: If provided and exists, loads the index from disk.
        """
        self.dimension = dimension
        self._chunk_ids: list[str] = []

        if index_path and (index_path / "index.faiss").exists():
            self.load(index_path)
        else:
            self._index = faiss.IndexFlatIP(dimension)

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self._index.ntotal

    def add(self, chunk_id: str, embedding: np.ndarray) -> None:
        """Add a single vector with its chunk ID."""
        vec = embedding.reshape(1, -1).astype(np.float32)
        self._index.add(vec)
        self._chunk_ids.append(chunk_id)

    def add_batch(self, chunk_ids: list[str], embeddings: np.ndarray) -> None:
        """Add multiple vectors with their chunk IDs.

        Args:
            chunk_ids: List of chunk ID strings.
            embeddings: (N, dimension) float32 array, L2-normalized.
        """
        if len(chunk_ids) == 0:
            return
        vecs = embeddings.astype(np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        assert vecs.shape[0] == len(chunk_ids), (
            f"Mismatch: {vecs.shape[0]} vectors vs {len(chunk_ids)} IDs"
        )
        self._index.add(vecs)
        self._chunk_ids.extend(chunk_ids)

    def search(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Search for nearest neighbors.

        Args:
            query_embedding: 1D float32 vector, L2-normalized.
            top_k: Number of results to return.

        Returns:
            List of (chunk_id, score) tuples, sorted by descending score.
        """
        if self.size == 0:
            return []

        vec = query_embedding.reshape(1, -1).astype(np.float32)
        k = min(top_k, self.size)
        scores, indices = self._index.search(vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for insufficient results
                continue
            results.append((self._chunk_ids[idx], float(score)))

        return results

    def save(self, path: Path) -> None:
        """Persist index and ID mapping to disk."""
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "index.faiss"))
        with open(path / "chunk_ids.json", "w") as f:
            json.dump(self._chunk_ids, f)

    def load(self, path: Path) -> None:
        """Load index and ID mapping from disk."""
        self._index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "chunk_ids.json") as f:
            self._chunk_ids = json.load(f)
        self.dimension = self._index.d
