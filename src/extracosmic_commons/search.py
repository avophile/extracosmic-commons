"""Hybrid search engine for Extracosmic Commons.

Combines FAISS semantic search with BM25 keyword matching using normalized
score fusion. The merge formula (ported from scholarly workbench):

    hybrid_score = (semantic / max_semantic) * semantic_weight
                 + (bm25 / max_bm25) * keyword_weight

BM25 is optional — if no BM25 index is provided, falls back to semantic-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .bm25 import BM25Index
from .database import Database
from .embeddings import EmbeddingPipeline
from .index import FAISSIndex
from .models import Chunk, Source


@dataclass
class CrossRef:
    """A corresponding passage in another translation/edition of the same work."""

    edition_label: str
    chunk: Chunk
    source: Source
    confidence: float


@dataclass
class SearchResult:
    """A single search result with chunk, score, and source metadata."""

    chunk: Chunk
    score: float
    source: Source
    paired_chunk: Chunk | None = None
    cross_translations: list[CrossRef] | None = None


def _diversify_results(results: list[SearchResult], top_k: int) -> list[SearchResult]:
    """Ensure results include different source types and distinct sources.

    Round-robin across source types (transcript, pdf, bilingual_pair),
    picking the highest-scoring unseen chunk from each type. Ensures a
    scholar sees what lecturers say alongside what the texts say, rather
    than getting 10 results all from one commentary.
    """
    if len(results) <= top_k:
        return results

    # Group by source type
    by_type: dict[str, list[SearchResult]] = {}
    for r in results:
        t = r.source.type.value
        by_type.setdefault(t, []).append(r)

    # Also track distinct sources within each type
    selected: list[SearchResult] = []
    seen_source_ids: set[str] = set()

    # Phase 1: one top result per source type
    for source_type in sorted(by_type.keys()):
        for r in by_type[source_type]:
            if r.source.id not in seen_source_ids:
                selected.append(r)
                seen_source_ids.add(r.source.id)
                break

    # Phase 2: one top result per distinct source (across all types)
    for r in results:
        if len(selected) >= top_k:
            break
        if r.source.id not in seen_source_ids:
            selected.append(r)
            seen_source_ids.add(r.source.id)

    # Phase 3: fill remaining slots by score
    selected_ids = {r.chunk.id for r in selected}
    for r in results:
        if len(selected) >= top_k:
            break
        if r.chunk.id not in selected_ids:
            selected.append(r)
            selected_ids.add(r.chunk.id)

    return selected[:top_k]


class SearchEngine:
    """Hybrid semantic + keyword search over the Extracosmic Commons corpus.

    When a BM25 index is provided, search results combine semantic similarity
    (FAISS) with exact term matching (BM25) using configurable weights.
    """

    def __init__(
        self,
        db: Database,
        embedder: EmbeddingPipeline,
        index: FAISSIndex,
        bm25: BM25Index | None = None,
        semantic_weight: float = 0.7,
    ):
        self.db = db
        self.embedder = embedder
        self.index = index
        self.bm25 = bm25
        self.semantic_weight = semantic_weight
        self.keyword_weight = 1.0 - semantic_weight

    def _hybrid_scores(
        self, query: str, query_vec, fetch_k: int
    ) -> dict[str, float]:
        """Compute hybrid scores by fusing FAISS and BM25 results.

        Returns a dict of chunk_id → hybrid_score.
        """
        # FAISS semantic search
        faiss_results = self.index.search(query_vec, top_k=fetch_k)
        semantic_scores = {cid: score for cid, score in faiss_results}

        if self.bm25 is None or self.bm25.size == 0:
            return semantic_scores

        # BM25 keyword search
        bm25_results = self.bm25.search(query, top_k=fetch_k)
        bm25_scores = {cid: score for cid, score in bm25_results}

        # Union all candidate IDs
        all_ids = set(semantic_scores.keys()) | set(bm25_scores.keys())

        # Normalize each distribution independently
        max_semantic = max(semantic_scores.values()) if semantic_scores else 1.0
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0

        if max_semantic == 0:
            max_semantic = 1.0
        if max_bm25 == 0:
            max_bm25 = 1.0

        hybrid = {}
        for cid in all_ids:
            sem = semantic_scores.get(cid, 0.0) / max_semantic
            kw = bm25_scores.get(cid, 0.0) / max_bm25
            hybrid[cid] = sem * self.semantic_weight + kw * self.keyword_weight

        return hybrid

    def search(
        self,
        query: str,
        top_k: int = 10,
        bilingual: bool = False,
        diversify: bool = True,
        **filters: Any,
    ) -> list[SearchResult]:
        """Search the corpus.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results.
            bilingual: If True, auto-fetch paired chunks for bilingual results.
            diversify: If True (default), ensure results include different source
                types and distinct sources, not just the highest-scoring chunks
                from one dominant source.
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

        # Over-fetch to account for post-filtering
        fetch_k = max(top_k * 3, 100) if filters else max(top_k, 100)

        # Get hybrid scores (or semantic-only if no BM25)
        scores = self._hybrid_scores(query, query_vec, fetch_k)

        if not scores:
            return []

        # Fetch chunks from database — sorted by score descending
        sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
        # Only fetch what we might need
        chunk_ids = sorted_ids[: fetch_k]
        chunks = self.db.get_chunks_by_ids(chunk_ids)

        # Build source cache
        source_cache: dict[str, Source] = {}

        # Apply filters and build results
        results = []
        for chunk in chunks:
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

            # Fetch source
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

        results.sort(key=lambda r: r.score, reverse=True)

        if diversify and len(results) > top_k:
            results = _diversify_results(results, top_k)

        return results[:top_k]
