"""FastAPI web viewer for Extracosmic Commons.

A minimal web interface for searching, browsing, and comparing editions.
Starts with `ec serve` and runs at localhost:8000.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Extracosmic Commons", version="0.2.0")

_TEMPLATE_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))


_cached_components = None


@app.on_event("startup")
async def preload_components():
    """Preload indexes on startup so the first request doesn't timeout."""
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Preloading indexes...")
    _get_components()
    logger.info("Indexes loaded, ready to serve.")


def _get_components():
    """Lazy-initialize and cache database, embedder, index, search engine.

    Components are cached after first initialization to avoid reloading
    the 680MB BM25 index and embedding model on every request.
    """
    global _cached_components
    if _cached_components is not None:
        return _cached_components

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
    _cached_components = (db, engine)
    return _cached_components


def _format_citation(source) -> str:
    """Generate a Chicago citation for a source.

    Filters out raw SPEAKER_XX labels from the author list before
    formatting, since whisperx sometimes over-segments speakers.
    """
    from ..citations import CitationFormatter
    import copy

    # Clean raw speaker IDs from author list for citation display
    if hasattr(source, 'author') and isinstance(source.author, list):
        clean = [a for a in source.author if not a.startswith('SPEAKER_') and a != 'UNKNOWN']
        if clean != source.author:
            source = copy.copy(source)
            source.author = clean if clean else source.author

    formatter = CitationFormatter()
    return formatter.chicago(source)


# Path mapping: RunPod ingestion paths → local filesystem paths
_PATH_MAP = [
    ("/workspace/corpus/zotero_hegel/Hegel texts/",
     os.path.expanduser("~/Documents/HegelTranscripts/Hegel texts/")),
    ("/workspace/corpus/zotero_houlgate/Houlgate texts/",
     os.path.expanduser("~/Documents/HegelTranscripts/Houlgate texts/")),
    ("/workspace/corpus/zotero_thompson/Thompson texts/",
     os.path.expanduser("~/Documents/HegelTranscripts/Thompson texts/")),
    ("/workspace/corpus/zotero_radnik/Radnik texts/",
     os.path.expanduser("~/Documents/HegelTranscripts/Radnik texts/")),
    ("/workspace/corpus/workbench/",
     os.path.expanduser("~/Documents/scholarly-workbench-integrated/")),
    ("/workspace/corpus/desktop_hegel/Hegel/",
     os.path.expanduser("~/Desktop/Hegel/")),
    ("/workspace/corpus/transcripts/",
     os.path.expanduser("~/Documents/HegelTranscripts/")),
    ("/workspace/corpus/houlgate_transcripts/",
     os.path.expanduser("~/Documents/HegelTranscripts/houlgate/")),
    ("/workspace/corpus/",
     os.path.expanduser("~/Documents/")),
    # Podcast pipeline audio files on External SSD
    ("/Volumes/External SSD/podcast-pipeline/",
     "/Volumes/External SSD/podcast-pipeline/"),
]


def _map_path(runpod_path: str | None) -> str | None:
    """Translate a RunPod ingestion path to the local filesystem."""
    if not runpod_path:
        return None
    for prefix, local in _PATH_MAP:
        if runpod_path.startswith(prefix):
            return local + runpod_path[len(prefix):]
    return runpod_path  # Return as-is if no mapping found


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Search form and corpus stats."""
    db, _ = _get_components()
    stats = db.get_stats()
    return templates.TemplateResponse(request, "search.html", {
        "stats": stats,
        "results": None,
        "query": "",
        "compare": False,
        "bilingual": False,
        "result_data_json": "[]",
        "citations": [],
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
        return templates.TemplateResponse(request, "search.html", {
            "stats": None,
            "results": None,
            "query": "",
            "compare": False,
            "bilingual": False,
            "result_data_json": "[]",
            "citations": [],
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

    # Build citation and data for each result (for JS clipboard/context)
    citations = []
    result_data = []
    for r in results:
        cite = _format_citation(r.source)
        citations.append(cite)
        result_data.append({
            "chunk_id": r.chunk.id,
            "text": r.chunk.text,
            "citation": cite,
            "pdf_page": r.chunk.pdf_page,
            "source_path": r.source.source_path,
            "source_title": r.source.title,
            "chunk_method": r.chunk.chunk_method,
            "youtube_url": r.chunk.youtube_url,
            "youtube_timestamp": r.chunk.youtube_timestamp,
            "lecturer": r.chunk.lecturer if r.chunk.lecturer and not r.chunk.lecturer.startswith("SPEAKER_") else "Wu",
        })



    # Attach citation data to each result for inline display.
    # For each conversation chunk, look up any citations it contains.
    chunk_citations = {}
    for r in results:
        if r.chunk.chunk_method == "speaker_turn":
            cites = db.conn.execute(
                "SELECT work_title, citation_type, page_english, page_german, "
                "speaker, confidence FROM citations WHERE conversation_chunk_id = ?",
                (r.chunk.id,),
            ).fetchall()
            if cites:
                chunk_citations[r.chunk.id] = [dict(c) for c in cites]
        # Also check reverse: if this is a primary text chunk, who cites it?
        elif r.chunk.chunk_method in ("paragraph", "bilingual_aligned"):
            rev = db.conn.execute(
                "SELECT speaker, conversation_date, work_title "
                "FROM citations WHERE cited_chunk_id = ? LIMIT 5",
                (r.chunk.id,),
            ).fetchall()
            if rev:
                chunk_citations[r.chunk.id] = [
                    {**dict(c), "_reverse": True} for c in rev
                ]

    # Clean author lists: remove raw SPEAKER_XX labels from display
    for r in results:
        if hasattr(r.source, 'author') and isinstance(r.source.author, list):
            clean = [a for a in r.source.author if not a.startswith('SPEAKER_') and a != 'UNKNOWN']
            r.source.clean_authors = ', '.join(clean) if clean else ', '.join(r.source.author)
        else:
            r.source.clean_authors = ', '.join(r.source.author) if r.source.author else ''

    return templates.TemplateResponse(request, "search.html", {
        "stats": None,
        "results": results,
        "query": q,
        "compare": compare,
        "bilingual": bilingual,
        "result_data_json": json.dumps(result_data),
        "citations": citations,
        "chunk_citations": chunk_citations,
    })


@app.get("/api/context/{chunk_id}")
async def api_context(chunk_id: str):
    """Get surrounding context for a chunk (before + current + after)."""
    db, _ = _get_components()

    # Find the chunk
    chunks = db.get_chunks_by_ids([chunk_id])
    if not chunks:
    
        return {"error": "Chunk not found"}

    chunk = chunks[0]

    # Get all chunks from the same source, sorted by position
    source_chunks = db.get_chunks_by_source(chunk.source_id)
    source_chunks.sort(key=lambda c: (c.pdf_page or 0, c.paragraph_index or 0))

    # Find this chunk's position
    idx = None
    for i, c in enumerate(source_chunks):
        if c.id == chunk_id:
            idx = i
            break

    if idx is None:
    
        return {"before": [], "current": chunk.text, "after": []}

    # Get 3 chunks before and 3 after
    before = [c.text for c in source_chunks[max(0, idx - 3):idx]]
    after = [c.text for c in source_chunks[idx + 1:idx + 4]]

    return {
        "before": before,
        "current": chunk.text,
        "after": after,
    }


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

    output = []
    for r in results:
        output.append({
            "score": r.score,
            "chunk_id": r.chunk.id,
            "text": r.chunk.text[:500],
            "source_title": r.source.title,
            "author": r.source.author,
            "structural_ref": r.chunk.structural_ref,
            "pdf_page": r.chunk.pdf_page,
            "lecturer": r.chunk.lecturer if r.chunk.lecturer and not r.chunk.lecturer.startswith("SPEAKER_") else "Wu",
            "citation": _format_citation(r.source),
            "cross_translations": [
                {
                    "edition": xref.edition_label,
                    "text": xref.chunk.text[:300],
                    "confidence": xref.confidence,
                }
                for xref in (r.cross_translations or [])
            ] if r.cross_translations else None,
        })

    return output


@app.get("/api/open-pdf")
async def api_open_pdf(path: str = Query(...), page: int = Query(1)):
    """Open a PDF at a specific page, or a video file."""
    import subprocess

    # Map RunPod paths to local paths
    local_path = _map_path(path)
    filepath = Path(local_path)
    if not filepath.exists():
        return {"error": f"File not found: {local_path} (original: {path})"}

    # Use macOS 'open' command — works without Automation permissions
    subprocess.Popen(["open", str(filepath)])

    return {"ok": True, "path": str(filepath), "page": page}


@app.get("/api/open-video")
async def api_open_video(path: str = Query(...), timestamp: str = Query("00:00:00")):
    """Open a video file and seek to the given timestamp."""
    import subprocess

    local_path = _map_path(path)
    filepath = Path(local_path)
    if not filepath.exists():
        return {"error": f"File not found: {local_path} (original: {path})"}

    # Parse timestamp HH:MM:SS to seconds
    parts = timestamp.split(":")
    seconds = 0
    if len(parts) == 3:
        seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:
        seconds = int(parts[0]) * 60 + int(parts[1])

    # Open in QuickTime and seek via AppleScript
    script = f'''
        tell application "QuickTime Player"
            open POSIX file "{filepath}"
            activate
            delay 1
            tell document 1
                set current time to {seconds}
                play
            end tell
        end tell
    '''
    subprocess.Popen(["osascript", "-e", script])
    return {"ok": True, "path": str(filepath), "timestamp": timestamp, "seconds": seconds}



# ===========================================================================
# Citation Integration (Phase 2)
# ===========================================================================


@app.get("/api/citations/chunk/{chunk_id}")
async def api_citations_for_chunk(chunk_id: str):
    """Get citations extracted from a specific conversation chunk.

    Returns all citation records where conversation_chunk_id matches,
    i.e. "what texts are cited in this chunk of conversation?"
    Used to show inline citation badges on search results.
    """
    db, _ = _get_components()
    rows = db.conn.execute(
        "SELECT * FROM citations WHERE conversation_chunk_id = ? ORDER BY confidence DESC",
        (chunk_id,),
    ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/citations/reverse/{chunk_id}")
async def api_reverse_citations(chunk_id: str):
    """Reverse citation lookup: which conversations cite this text chunk?

    Returns all citation records where cited_chunk_id matches,
    i.e. "where in the Wu conversations is this Hegel passage discussed?"
    Used to show a 'Discussed in...' section on primary text results.
    """
    db, _ = _get_components()
    return db.get_citations_for_chunk(chunk_id)


@app.get("/api/citations")
async def api_citations_browse(
    work: str | None = Query(None, description="Filter by work title"),
    speaker: str | None = Query(None, description="Filter by speaker"),
    type: str | None = Query(None, description="Filter by citation type (reading/reference/paraphrase)"),
    page: str | None = Query(None, description="Filter by page number"),
    limit: int = Query(100, description="Max results"),
    offset: int = Query(0, description="Pagination offset"),
):
    """Browse all citations with optional filters and facets.

    Returns a paginated list of citations plus facet counts for building
    filter dropdowns in the citation browser UI.
    """
    db, _ = _get_components()

    # Build dynamic WHERE clause from filters
    conditions = []
    params: list = []

    if work:
        conditions.append("work_title = ?")
        params.append(work)
    if speaker:
        conditions.append("speaker = ?")
        params.append(speaker)
    if type:
        conditions.append("citation_type = ?")
        params.append(type)
    if page:
        conditions.append("(page_german = ? OR page_english = ?)")
        params.extend([page, page])

    where = " WHERE " + " AND ".join(conditions) if conditions else ""

    # Get total count for pagination
    total = db.conn.execute(
        f"SELECT COUNT(*) FROM citations{where}", params
    ).fetchone()[0]

    # Get the filtered, paginated citations
    rows = db.conn.execute(
        f"SELECT * FROM citations{where} ORDER BY conversation_date, audio_timestamp LIMIT ? OFFSET ?",
        params + [limit, offset],
    ).fetchall()

    # Build facets (counts for filter dropdowns — always computed on full corpus)
    works = db.conn.execute(
        "SELECT work_title, COUNT(*) as cnt FROM citations GROUP BY work_title ORDER BY cnt DESC"
    ).fetchall()
    speakers = db.conn.execute(
        "SELECT speaker, COUNT(*) as cnt FROM citations GROUP BY speaker ORDER BY cnt DESC"
    ).fetchall()
    types = db.conn.execute(
        "SELECT citation_type, COUNT(*) as cnt FROM citations GROUP BY citation_type ORDER BY cnt DESC"
    ).fetchall()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "citations": [dict(r) for r in rows],
        "facets": {
            "works": {r["work_title"]: r["cnt"] for r in works},
            "speakers": {r["speaker"]: r["cnt"] for r in speakers},
            "types": {r["citation_type"]: r["cnt"] for r in types},
        },
    }


@app.get("/api/citation-stats")
async def api_citation_stats():
    """Citation corpus statistics — counts by type, work, and speaker."""
    db, _ = _get_components()
    return db.get_citation_stats()


@app.get("/citations", response_class=HTMLResponse)
async def citations_page(
    request: Request,
    work: str | None = Query(None),
    speaker: str | None = Query(None),
    type: str | None = Query(None),
):
    """Citation browser page — HTML view of all citations with filters."""
    db, _ = _get_components()

    # Reuse the API logic for filtered results
    conditions = []
    params: list = []
    if work:
        conditions.append("work_title = ?")
        params.append(work)
    if speaker:
        conditions.append("speaker = ?")
        params.append(speaker)
    if type:
        conditions.append("citation_type = ?")
        params.append(type)

    where = " WHERE " + " AND ".join(conditions) if conditions else ""
    total = db.conn.execute(
        f"SELECT COUNT(*) FROM citations{where}", params
    ).fetchone()[0]

    rows = db.conn.execute(
        f"SELECT * FROM citations{where} ORDER BY work_title, page_english, conversation_date LIMIT 200",
        params,
    ).fetchall()
    citations = [dict(r) for r in rows]

    # Facets for filter dropdowns
    works = db.conn.execute(
        "SELECT DISTINCT work_title FROM citations ORDER BY work_title"
    ).fetchall()
    speakers = db.conn.execute(
        "SELECT DISTINCT speaker FROM citations ORDER BY speaker"
    ).fetchall()
    types = db.conn.execute(
        "SELECT DISTINCT citation_type FROM citations ORDER BY citation_type"
    ).fetchall()

    return templates.TemplateResponse(request, "citations.html", {
        "citations": citations,
        "total": total,
        "works": [r["work_title"] for r in works],
        "speakers": [r["speaker"] for r in speakers],
        "types": [r["citation_type"] for r in types],
        "active_work": work or "",
        "active_speaker": speaker or "",
        "active_type": type or "",
    })

@app.get("/api/stats")
async def api_stats():
    """JSON API for corpus stats."""
    db, _ = _get_components()
    stats = db.get_stats()
    return stats
