"""Subprocess-based embedding to avoid FAISS + PyTorch libomp conflict on macOS.

On macOS, FAISS and PyTorch both link libomp, causing segfaults when both
are loaded in the same process. This module provides embedding via a
subprocess that only loads PyTorch (no FAISS), communicating via temp files.

On Linux/cloud machines this overhead is unnecessary — the EmbeddingPipeline
can be used directly.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def _needs_subprocess() -> bool:
    """Check if we need subprocess isolation (macOS only)."""
    return platform.system() == "Darwin"


def embed_texts_subprocess(
    texts: list[str],
    model_name: str = "BAAI/bge-m3",
    batch_size: int = 8,
) -> np.ndarray:
    """Embed texts in a subprocess to avoid libomp conflict.

    Writes texts to a temp file, runs a subprocess that loads only
    SentenceTransformer (no FAISS), and reads back the embeddings.
    """
    if not texts:
        return np.zeros((0, 1024), dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        texts_path = Path(tmpdir) / "texts.json"
        emb_path = Path(tmpdir) / "embeddings.npy"

        # Write texts
        texts_path.write_text(json.dumps(texts))

        # Run embedding in subprocess (no faiss imported)
        script = f"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import numpy as np
from sentence_transformers import SentenceTransformer

texts = json.loads(open('{texts_path}').read())
model = SentenceTransformer('{model_name}', device='cpu')

avg_len = sum(len(t) for t in texts) / max(len(texts), 1)
bs = min({batch_size}, 4 if avg_len > 4000 else 8 if avg_len > 1000 else {batch_size})

embeddings = model.encode(texts, batch_size=bs, normalize_embeddings=True, show_progress_bar=False)
np.save('{emb_path}', embeddings.astype(np.float32))
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Embedding subprocess failed: {result.stderr}")

        return np.load(str(emb_path))


def embed_query_subprocess(
    query: str,
    model_name: str = "BAAI/bge-m3",
) -> np.ndarray:
    """Embed a single query in a subprocess."""
    result = embed_texts_subprocess([query], model_name=model_name)
    return result[0]
