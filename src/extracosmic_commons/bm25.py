"""BM25 keyword index for hybrid search.

Complements FAISS semantic search with exact term matching. BM25 excels at
finding specific terminology (Aufheben, Fürsichsein, §132) that semantic
embeddings may conflate with related but distinct concepts.

Ported from scholarly workbench query_hybrid.py, adapted for the Extracosmic
Commons data model. Uses rank_bm25.BM25Okapi with simple whitespace tokenization.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25: lowercase + split on whitespace.

    Simple but effective for philosophical terminology. Preserves German
    compound words and technical terms. Can upgrade to spaCy/NLTK later
    if needed.
    """
    return text.lower().split()


class BM25Index:
    """BM25 keyword index with chunk ID mapping.

    Maintains a parallel list of chunk IDs so that BM25 search results
    can be mapped back to Chunk records in the SQLite database.
    """

    def __init__(self, index_path: Path | None = None):
        """Create a new index or load from disk."""
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._tokenized_corpus: list[list[str]] = []

        if index_path and (index_path / "bm25_corpus.pkl").exists():
            self.load(index_path)

    @property
    def size(self) -> int:
        """Number of documents in the index."""
        return len(self._chunk_ids)

    def build(self, chunk_ids: list[str], texts: list[str]) -> None:
        """Build the BM25 index from chunk texts.

        Args:
            chunk_ids: List of chunk ID strings.
            texts: List of text strings, parallel to chunk_ids.
        """
        assert len(chunk_ids) == len(texts), (
            f"Mismatch: {len(chunk_ids)} IDs vs {len(texts)} texts"
        )
        self._chunk_ids = list(chunk_ids)
        self._tokenized_corpus = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info(f"Built BM25 index with {len(chunk_ids)} documents")

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """Search for documents matching the query terms.

        Returns (chunk_id, score) pairs sorted by descending BM25 score.
        """
        if self._bm25 is None or self.size == 0:
            return []

        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices by score
        k = min(top_k, self.size)
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:  # Skip zero-score results
                results.append((self._chunk_ids[idx], score))

        return results

    def save(self, path: Path) -> None:
        """Persist the tokenized corpus and chunk IDs.

        The BM25Okapi object itself is rebuilt from the tokenized corpus
        on load — this is fast (<1s for 460K docs) and avoids pickle
        compatibility issues with the rank_bm25 library.
        """
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "bm25_corpus.pkl", "wb") as f:
            pickle.dump(self._tokenized_corpus, f)
        with open(path / "bm25_chunk_ids.json", "w") as f:
            json.dump(self._chunk_ids, f)
        logger.info(f"Saved BM25 index ({self.size} docs) to {path}")

    def load(self, path: Path) -> None:
        """Load the tokenized corpus and rebuild BM25Okapi."""
        with open(path / "bm25_corpus.pkl", "rb") as f:
            self._tokenized_corpus = pickle.load(f)
        with open(path / "bm25_chunk_ids.json") as f:
            self._chunk_ids = json.load(f)
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info(f"Loaded BM25 index ({self.size} docs) from {path}")
