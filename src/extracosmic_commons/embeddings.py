"""BGE-M3 embedding pipeline for Extracosmic Commons.

Provides multilingual text embeddings using BAAI/bge-m3 (1024-dim, 100+ languages).
The model is loaded lazily on first use to avoid slow import-time initialization.

On macOS, FAISS and PyTorch both link libomp, causing segfaults when both are
loaded in the same process. The EmbeddingPipeline detects this and automatically
routes embedding calls through a subprocess that only loads PyTorch. On Linux/cloud
machines, everything runs in-process for maximum speed.
"""

from __future__ import annotations

import platform
from typing import Callable

import numpy as np

# Detect macOS libomp conflict at module level
_MACOS = platform.system() == "Darwin"


class EmbeddingPipeline:
    """Multilingual embedding pipeline with automatic macOS subprocess isolation.

    On macOS: embeddings are computed in a subprocess to avoid the FAISS+PyTorch
    libomp segfault. Slower but reliable.

    On Linux/cloud: embeddings are computed in-process with GPU acceleration.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self._model = None
        self._dimension: int | None = None

    @property
    def model(self):
        """Lazy-load the SentenceTransformer model (in-process, non-macOS only)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """Embedding dimension. BGE-M3 produces 1024-dim vectors."""
        if self._dimension is not None:
            return self._dimension
        if _MACOS:
            # Avoid loading the model just to get the dimension
            if "bge-m3" in self.model_name:
                self._dimension = 1024
            elif "MiniLM-L6" in self.model_name:
                self._dimension = 384
            else:
                self._dimension = self.model.get_sentence_embedding_dimension()
        else:
            self._dimension = self.model.get_sentence_embedding_dimension()
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Returns a float32 vector, L2-normalized for cosine similarity.
        """
        if not text or not text.strip():
            return np.zeros(self.dimension, dtype=np.float32)

        if _MACOS:
            from .embed_subprocess import embed_texts_subprocess

            return embed_texts_subprocess([text], model_name=self.model_name)[0]

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

        Returns an (N, dimension) float32 array, L2-normalized.
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        # Replace empty strings with a space to avoid model issues
        cleaned = [t if t and t.strip() else " " for t in texts]

        if _MACOS:
            from .embed_subprocess import embed_texts_subprocess

            return embed_texts_subprocess(
                cleaned, model_name=self.model_name, batch_size=batch_size
            )

        # Auto-reduce batch size for long texts to prevent OOM
        avg_len = sum(len(t) for t in cleaned) / len(cleaned)
        if avg_len > 4000:
            batch_size = min(batch_size, 4)
        elif avg_len > 1000:
            batch_size = min(batch_size, 8)

        if progress_callback is None:
            embeddings = self.model.encode(
                cleaned,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        else:
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
