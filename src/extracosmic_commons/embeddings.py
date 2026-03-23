"""BGE-M3 embedding pipeline for Extracosmic Commons.

Provides multilingual text embeddings using BAAI/bge-m3 (1024-dim, 100+ languages).
The model is loaded lazily on first use to avoid slow import-time initialization.

BGE-M3 was chosen over bge-base-en-v1.5 (used in the scholarly workbench) because
this platform requires multilingual support: German Hegel texts, French philosophy
(Malabou, Derrida, Foucault), and English translations must all inhabit the same
vector space.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


class EmbeddingPipeline:
    """Lazy-loaded multilingual embedding pipeline.

    The SentenceTransformer model is only loaded when the first embedding
    is requested, not at construction time. This keeps imports fast and
    allows tests to mock the model.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazy-load the SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """Embedding dimension. BGE-M3 produces 1024-dim vectors."""
        return self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Returns a 1024-dim float32 vector, L2-normalized for cosine
        similarity via inner product in FAISS.
        """
        if not text or not text.strip():
            return np.zeros(self.dimension, dtype=np.float32)

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embedding.astype(np.float32)

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Embed a batch of texts.

        Returns an (N, 1024) float32 array, L2-normalized.

        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts per encoding batch.
            progress_callback: Optional fn(completed, total) called after each batch.
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        # Replace empty strings with a space to avoid model issues
        cleaned = [t if t and t.strip() else " " for t in texts]

        if progress_callback is None:
            embeddings = self.model.encode(
                cleaned,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        else:
            # Process in batches with progress reporting
            all_embeddings = []
            for i in range(0, len(cleaned), batch_size):
                batch = cleaned[i : i + batch_size]
                batch_emb = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                all_embeddings.append(batch_emb)
                progress_callback(min(i + batch_size, len(cleaned)), len(cleaned))
            embeddings = np.vstack(all_embeddings)

        return embeddings.astype(np.float32)
