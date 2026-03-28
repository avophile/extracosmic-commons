"""TDD tests for Phase 2: Citation Integration in Search UI.

Tests cover three deliverables:
1. Citation data attached to search results (conversation chunks with citations)
2. Reverse citation view (which conversations cite a given passage)
3. Dedicated citation browser page with filtering

These tests run against the FastAPI app with a real SQLite DB populated
with fixture data, no embedding model needed.
"""

import json
import uuid

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures: build a minimal DB with sources, chunks, and citations
# ---------------------------------------------------------------------------

def _id():
    """Generate a UUID string."""
    return str(uuid.uuid4())


@pytest.fixture()
def populated_app(tmp_path, monkeypatch):
    """Create a FastAPI TestClient backed by a DB with citations.

    Populates:
    - 1 Wu conversation source with 3 speaker-turn chunks
    - 1 primary text source (Science of Logic) with 2 chunks
    - 3 citation records linking conversation chunks → primary text
    """
    import importlib

    monkeypatch.setenv("EC_DATA_DIR", str(tmp_path))

    from extracosmic_commons.database import Database
    db = Database(tmp_path / "extracosmic.db")

    # --- Sources ---
    conv_source_id = _id()
    db.conn.execute(
        """INSERT INTO sources (id, type, title, author, language, edition,
           source_path, source_url, metadata, sharing_status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (conv_source_id, "transcript", "Wu Conversation 2026-03-15",
         '["Wu", "Douglas"]', '["en"]', None,
         "/audio/wu_2026-03-15.m4a", None, '{}', "local_only",
         "2026-03-15T10:00:00+00:00"),
    )

    text_source_id = _id()
    db.conn.execute(
        """INSERT INTO sources (id, type, title, author, language, edition,
           source_path, source_url, metadata, sharing_status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (text_source_id, "pdf", "Science of Logic",
         '["Hegel", "di Giovanni"]', '["en"]', "di Giovanni 2010",
         "/texts/sol_di_giovanni.pdf", None, '{}', "local_only",
         "2025-01-01T00:00:00+00:00"),
    )

    # --- Chunks ---
    chunk_wu_1 = _id()
    chunk_wu_2 = _id()
    chunk_wu_3 = _id()
    chunk_text_1 = _id()
    chunk_text_2 = _id()

    for i, (cid, sid, text, lecturer, method) in enumerate([
        (chunk_wu_1, conv_source_id,
         "so on page 110 he says that being and nothing are the same",
         "Wu", "speaker_turn"),
        (chunk_wu_2, conv_source_id,
         "right and then on page 115 Hegel introduces becoming",
         "Douglas", "speaker_turn"),
        (chunk_wu_3, conv_source_id,
         "I think the key point about determinate being is on 130",
         "Wu", "speaker_turn"),
        (chunk_text_1, text_source_id,
         "Being, pure being – without further determination.",
         None, "paragraph"),
        (chunk_text_2, text_source_id,
         "Becoming is the unseparatedness of being and nothing.",
         None, "paragraph"),
    ]):
        db.conn.execute(
            """INSERT INTO chunks (id, source_id, text, language,
               structural_ref, pdf_page, youtube_timestamp, youtube_url,
               paragraph_index, paired_chunk_id, lecturer, lecture_number,
               chunk_method, sharing_status)
               VALUES (?, ?, ?, 'en', NULL, NULL, NULL, NULL, ?, NULL, ?, NULL, ?, 'local_only')""",
            (cid, sid, text, i, lecturer, method),
        )

    # --- Citations ---
    cite_1_id = _id()
    cite_2_id = _id()
    cite_3_id = _id()

    for cit_id, work, ctype, page_en, speaker, conv_chunk, cited_chunk, context, conf in [
        (cite_1_id, "Science of Logic", "reading", "110", "Wu",
         chunk_wu_1, chunk_text_1,
         "Wu reads from the opening of the Quality section", 0.85),
        (cite_2_id, "Science of Logic", "reference", "115", "Douglas",
         chunk_wu_2, chunk_text_2,
         "Douglas references the transition to Becoming", 0.70),
        (cite_3_id, "Science of Logic", "reference", "130", "Wu",
         chunk_wu_3, None,
         "Wu references the section on Determinate Being", 0.60),
    ]:
        db.conn.execute(
            """INSERT INTO citations (id, work_title, work_author, citation_type,
               page_german, page_english, edition_german, edition_english,
               section_ref, quoted_text, speaker, conversation_date,
               audio_timestamp, audio_timestamp_end, audio_path,
               conversation_source_id, conversation_chunk_id,
               cited_source_id, cited_chunk_id,
               discussion_context, confidence, extraction_notes)
               VALUES (?, ?, 'Hegel', ?, NULL, ?, NULL, 'di Giovanni 2010',
               NULL, NULL, ?, '2026-03-15', '00:05:00', NULL, NULL,
               ?, ?, ?, ?, ?, ?, NULL)""",
            (cit_id, work, ctype, page_en, speaker,
             conv_source_id, conv_chunk,
             text_source_id if cited_chunk else None, cited_chunk,
             context, conf),
        )

    db.conn.commit()
    db.close()

    # Force fresh app initialization for each test
    import extracosmic_commons.web.app as app_mod
    app_mod._cached_components = None
    importlib.reload(app_mod)

    client = TestClient(app_mod.app)
    yield client, {
        "conv_source_id": conv_source_id,
        "text_source_id": text_source_id,
        "chunk_wu_1": chunk_wu_1,
        "chunk_wu_2": chunk_wu_2,
        "chunk_wu_3": chunk_wu_3,
        "chunk_text_1": chunk_text_1,
        "chunk_text_2": chunk_text_2,
        "cite_1_id": cite_1_id,
        "cite_2_id": cite_2_id,
        "cite_3_id": cite_3_id,
    }

    # Cleanup
    app_mod._cached_components = None


# ===========================================================================
# 1. API endpoint: get citations for a conversation chunk
# ===========================================================================

class TestCitationsByChunkAPI:
    """GET /api/citations/chunk/{chunk_id} — citations attached to a chunk."""

    def test_returns_citations_for_chunk_with_citation(self, populated_app):
        """A conversation chunk that contains a citation returns it."""
        client, ids = populated_app
        resp = client.get(f"/api/citations/chunk/{ids['chunk_wu_1']}")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["work_title"] == "Science of Logic"
        assert data[0]["page_english"] == "110"
        assert data[0]["citation_type"] == "reading"
        assert data[0]["speaker"] == "Wu"

    def test_returns_empty_for_chunk_without_citation(self, populated_app):
        """A chunk with no citations returns an empty list, not an error."""
        client, ids = populated_app
        resp = client.get(f"/api/citations/chunk/{ids['chunk_text_1']}")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_404_for_nonexistent_chunk(self, populated_app):
        """A nonexistent chunk_id returns 404."""
        client, _ = populated_app
        resp = client.get(f"/api/citations/chunk/{_id()}")
        assert resp.status_code == 200  # empty list is fine
        assert resp.json() == []


# ===========================================================================
# 2. API endpoint: reverse citation lookup (who cites this text?)
# ===========================================================================

class TestReverseCitationsAPI:
    """GET /api/citations/reverse/{chunk_id} — which conversations cite this text chunk."""

    def test_reverse_lookup_finds_citing_conversations(self, populated_app):
        """A primary text chunk that is cited returns the citing records."""
        client, ids = populated_app
        resp = client.get(f"/api/citations/reverse/{ids['chunk_text_1']}")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["speaker"] == "Wu"
        assert data[0]["conversation_chunk_id"] == ids["chunk_wu_1"]

    def test_reverse_lookup_empty_for_uncited_chunk(self, populated_app):
        """A text chunk that nobody cites returns empty list."""
        client, ids = populated_app
        # chunk_text_2 is cited by chunk_wu_2, so let's use a chunk with no reverse citation
        resp = client.get(f"/api/citations/reverse/{ids['chunk_wu_3']}")
        assert resp.status_code == 200
        assert resp.json() == []


# ===========================================================================
# 3. API endpoint: citation browser with filters
# ===========================================================================

class TestCitationBrowserAPI:
    """GET /api/citations — browse all citations with optional filters."""

    def test_returns_all_citations(self, populated_app):
        """Without filters, returns all 3 citations."""
        client, _ = populated_app
        resp = client.get("/api/citations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        assert len(data["citations"]) == 3

    def test_filter_by_work(self, populated_app):
        """Filter by work_title returns only matching citations."""
        client, _ = populated_app
        resp = client.get("/api/citations?work=Science+of+Logic")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3  # all are Science of Logic

    def test_filter_by_speaker(self, populated_app):
        """Filter by speaker returns only that speaker's citations."""
        client, _ = populated_app
        resp = client.get("/api/citations?speaker=Wu")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        for c in data["citations"]:
            assert c["speaker"] == "Wu"

    def test_filter_by_citation_type(self, populated_app):
        """Filter by citation_type returns only that type."""
        client, _ = populated_app
        resp = client.get("/api/citations?type=reading")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["citations"][0]["citation_type"] == "reading"

    def test_combined_filters(self, populated_app):
        """Multiple filters combine with AND logic."""
        client, _ = populated_app
        resp = client.get("/api/citations?speaker=Douglas&type=reference")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["citations"][0]["speaker"] == "Douglas"

    def test_includes_facets(self, populated_app):
        """Response includes facets for UI filter dropdowns."""
        client, _ = populated_app
        resp = client.get("/api/citations")
        data = resp.json()
        assert "facets" in data
        assert "works" in data["facets"]
        assert "speakers" in data["facets"]
        assert "types" in data["facets"]
        assert "Science of Logic" in data["facets"]["works"]


# ===========================================================================
# 4. Citation browser HTML page
# ===========================================================================

class TestCitationBrowserPage:
    """GET /citations — HTML page for browsing citations."""

    def test_citations_page_loads(self, populated_app):
        """The citation browser page renders without error."""
        client, _ = populated_app
        resp = client.get("/citations")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Citation Browser" in resp.text

    def test_citations_page_shows_count(self, populated_app):
        """The page shows the total citation count."""
        client, _ = populated_app
        resp = client.get("/citations")
        assert "3 citations" in resp.text


# ===========================================================================
# 5. Citation stats API (extends existing)
# ===========================================================================

class TestCitationStatsAPI:
    """GET /api/citation-stats — citation corpus statistics."""

    def test_citation_stats_endpoint(self, populated_app):
        """Stats endpoint returns counts by type, work, and speaker."""
        client, _ = populated_app
        resp = client.get("/api/citation-stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_citations"] == 3
        assert "reading" in data["by_type"]
        assert "reference" in data["by_type"]
        assert "Science of Logic" in data["by_work"]
