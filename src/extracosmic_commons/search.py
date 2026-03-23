"""Semantic search engine for Extracosmic Commons.

Combines FAISS vector search with SQLite metadata filtering.
Phase 0 is semantic-only; BM25 hybrid search is Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .database import Database
from .embeddings import EmbeddingPipeline
from .index import FAISSIndex
from .models import Chunk, Source


@dataclass
class SearchResult:
    """A single search result with chunk, score, and source metadata."""

    chunk: Chunk
    score: float
    source: Source
    paired_chunk: Chunk | None = None


class SearchEngine:
    """Semantic search over the Extracosmic Commons corpus.

    Embeds the query with BGE-M3, searches FAISS for nearest neighbors,
    then enriches results with metadata from SQLite. Supports post-retrieval
    filtering and automatic bilingual pair resolution.
    """

    def __init__(
        self,
        db: Database,
        embedder: EmbeddingPipeline,
        index: FAISSIndex,
    ):
        self.db = db
        self.embedder = embedder
        self.index = index

    def search(
        self,
        query: str,
        top_k: int = 10,
        bilingual: bool = False,
        **filters: Any,
    ) -> list[SearchResult]:
        """Search the corpus.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results.
            bilingual: If True, auto-fetch paired chunks for bilingual results.
            **filters: Post-retrieval filters. Supported:
                - lecturer: str
                - language: str
                - source_type: str (SourceType value)
                - structural_ref: dict of path→value filters

        Returns:
            List of SearchResult, sorted by descending score.
        """
        # Embed query
        query_vec = self.embedder.embed(query)

        # Over-fetch from FAISS to account for post-filtering
        fetch_k = top_k * 3 if filters else top_k
        faiss_results = self.index.search(query_vec, top_k=fetch_k)

        if not faiss_results:
            return []

        # Fetch chunks from database
        chunk_ids = [cid for cid, _ in faiss_results]
        scores = {cid: score for cid, score in faiss_results}
        chunks = self.db.get_chunks_by_ids(chunk_ids)

        # Build source cache
        source_cache: dict[str, Source] = {}

        # Apply filters
        results = []
        for chunk in chunks:
            # Post-retrieval filters
            if "lecturer" in filters and chunk.lecturer != filters["lecturer"]:
                continue
            if "language" in filters and chunk.language != filters["language"]:
                continue
            if "source_type" in filters:
                if chunk.source_id not in source_cache:
                    src = self.db.get_source(chunk.source_id)
                    if src:
                        source_cache[chunk.source_id] = src
                src = source_cache.get(chunk.source_id)
                if src and src.type.value != filters["source_type"]:
                    continue
            if "structural_ref" in filters and isinstance(filters["structural_ref"], dict):
                if chunk.structural_ref is None:
                    continue
                match = all(
                    chunk.structural_ref.get(k) == v
                    for k, v in filters["structural_ref"].items()
                )
                if not match:
                    continue

            # Fetch source if not cached
            if chunk.source_id not in source_cache:
                src = self.db.get_source(chunk.source_id)
                if src:
                    source_cache[chunk.source_id] = src

            source = source_cache.get(chunk.source_id)
            if source is None:
                continue

            # Bilingual pair resolution
            paired = None
            if bilingual and chunk.paired_chunk_id:
                paired_chunks = self.db.get_chunks_by_ids([chunk.paired_chunk_id])
                paired = paired_chunks[0] if paired_chunks else None

            results.append(SearchResult(
                chunk=chunk,
                score=scores.get(chunk.id, 0.0),
                source=source,
                paired_chunk=paired,
            ))

        # Sort by score and trim to top_k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
