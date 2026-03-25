"""FastAPI web viewer for Extracosmic Commons.

A minimal web interface for searching, browsing, and comparing editions.
Starts with `ec serve` and runs at localhost:8000.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Extracosmic Commons", version="0.2.0")

# Templates directory
_TEMPLATE_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))


def _get_components():
    """Lazy-initialize database, embedder, index, search engine."""
    from ..bm25 import BM25Index
    from ..database import Database
    from ..embeddings import EmbeddingPipeline
    from ..index import FAISSIndex
    from ..search import SearchEngine

    data_dir = Path(os.environ.get("EC_DATA_DIR", "data"))
    db = Database(data_dir / "extracosmic.db")
    embedder = EmbeddingPipeline()

    index_path = data_dir / "faiss_index"
    index = FAISSIndex(index_path=index_path) if (index_path / "index.faiss").exists() else FAISSIndex(dimension=1024)

    bm25_path = data_dir / "bm25_index"
    bm25 = BM25Index(index_path=bm25_path) if (bm25_path / "bm25_corpus.pkl").exists() else None

    engine = SearchEngine(db, embedder, index, bm25=bm25)
    return db, engine


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Search form and corpus stats."""
    db, _ = _get_components()
    stats = db.get_stats()
    db.close()
    return templates.TemplateResponse(request, "search.html", {
        "stats": stats,
        "results": None,
        "query": "",
    })


@app.get("/search", response_class=HTMLResponse)
async def search_view(
    request: Request,
    q: str = Query("", description="Search query"),
    k: int = Query(10, description="Number of results"),
    compare: bool = Query(False, description="Show cross-translation comparisons"),
    bilingual: bool = Query(False, description="Show bilingual pairs"),
    lecturer: str | None = Query(None),
    language: str | None = Query(None),
    source_type: str | None = Query(None),
):
    """Search results page."""
    if not q.strip():
        return templates.TemplateResponse("search.html", {
            "request": request,
            "stats": None,
            "results": None,
            "query": "",
        })

    db, engine = _get_components()

    filters = {}
    if lecturer:
        filters["lecturer"] = lecturer
    if language:
        filters["language"] = language
    if source_type:
        filters["source_type"] = source_type

    results = engine.search(
        q, top_k=k, bilingual=bilingual, compare=compare, **filters
    )

    db.close()

    return templates.TemplateResponse(request, "search.html", {
        "stats": None,
        "results": results,
        "query": q,
        "compare": compare,
        "bilingual": bilingual,
    })


@app.get("/api/search")
async def api_search(
    q: str = Query(..., description="Search query"),
    k: int = Query(10),
    compare: bool = Query(False),
    bilingual: bool = Query(False),
):
    """JSON API for search."""
    db, engine = _get_components()
    results = engine.search(q, top_k=k, bilingual=bilingual, compare=compare)
    db.close()

    return [
        {
            "score": r.score,
            "text": r.chunk.text[:500],
            "source_title": r.source.title,
            "author": r.source.author,
            "structural_ref": r.chunk.structural_ref,
            "pdf_page": r.chunk.pdf_page,
            "lecturer": r.chunk.lecturer,
            "cross_translations": [
                {
                    "edition": xref.edition_label,
                    "text": xref.chunk.text[:300],
                    "confidence": xref.confidence,
                }
                for xref in (r.cross_translations or [])
            ] if r.cross_translations else None,
        }
        for r in results
    ]


@app.get("/api/stats")
async def api_stats():
    """JSON API for corpus stats."""
    db, _ = _get_components()
    stats = db.get_stats()
    db.close()
    return stats
