"""SQLite database layer for Extracosmic Commons.

Provides structured storage for Sources, Chunks, and Analyses. The database
complements the FAISS vector index: FAISS handles similarity search over
embeddings, while SQLite handles metadata queries, structural lookups, and
provenance tracking.

The structural_ref field on chunks is stored as JSON text and queried via
SQLite's json_extract() function, allowing different works to have different
organizational hierarchies without schema changes.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .models import Analysis, Chunk, Source


class Database:
    """SQLite-backed metadata store for Extracosmic Commons.

    Wraps a SQLite connection with typed CRUD operations for the core
    data model. Uses WAL mode for concurrent read/write access.
    """

    def __init__(self, db_path: str | Path = "data/extracosmic.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.init_schema()

    def init_schema(self) -> None:
        """Create tables if they don't exist. Idempotent."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS sources (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                title TEXT NOT NULL,
                author TEXT NOT NULL DEFAULT '[]',
                language TEXT NOT NULL DEFAULT '["en"]',
                edition TEXT,
                source_path TEXT,
                source_url TEXT,
                metadata TEXT NOT NULL DEFAULT '{}',
                sharing_status TEXT NOT NULL DEFAULT 'local_only',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL REFERENCES sources(id),
                text TEXT NOT NULL,
                language TEXT NOT NULL DEFAULT 'en',
                structural_ref TEXT,
                pdf_page INTEGER,
                youtube_timestamp TEXT,
                youtube_url TEXT,
                paragraph_index INTEGER,
                paired_chunk_id TEXT,
                lecturer TEXT,
                lecture_number INTEGER,
                chunk_method TEXT NOT NULL DEFAULT 'unknown',
                sharing_status TEXT NOT NULL DEFAULT 'local_only'
            );

            CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                linked_chunks TEXT NOT NULL DEFAULT '[]',
                linked_analyses TEXT NOT NULL DEFAULT '[]',
                tags TEXT NOT NULL DEFAULT '[]',
                sharing_status TEXT NOT NULL DEFAULT 'local_only',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            -- Indexes for common query patterns
            CREATE INDEX IF NOT EXISTS idx_chunks_source_id ON chunks(source_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_language ON chunks(language);
            CREATE INDEX IF NOT EXISTS idx_chunks_lecturer ON chunks(lecturer);
            CREATE INDEX IF NOT EXISTS idx_chunks_pdf_page ON chunks(pdf_page);
            CREATE INDEX IF NOT EXISTS idx_sources_type ON sources(type);
        """)
        self.conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    # --- Source CRUD ---

    def insert_source(self, source: Source) -> str:
        """Insert a Source record. Returns the source ID."""
        d = source.to_dict()
        self.conn.execute(
            """INSERT INTO sources
               (id, type, title, author, language, edition, source_path,
                source_url, metadata, sharing_status, created_at)
               VALUES (:id, :type, :title, :author, :language, :edition,
                       :source_path, :source_url, :metadata, :sharing_status, :created_at)""",
            d,
        )
        self.conn.commit()
        return source.id

    def get_source(self, source_id: str) -> Source | None:
        """Fetch a Source by ID. Returns None if not found."""
        row = self.conn.execute(
            "SELECT * FROM sources WHERE id = ?", (source_id,)
        ).fetchone()
        if row is None:
            return None
        return Source.from_dict(dict(row))

    def get_all_sources(self) -> list[Source]:
        """Fetch all Sources."""
        rows = self.conn.execute("SELECT * FROM sources ORDER BY created_at").fetchall()
        return [Source.from_dict(dict(r)) for r in rows]

    def source_exists(self, title: str, author: list[str] | None = None) -> bool:
        """Check if a source with this title (and optionally author) already exists.

        Used for deduplication during import.
        """
        if author:
            author_json = json.dumps(sorted(author))
            # Check for exact title match with any author overlap
            rows = self.conn.execute(
                "SELECT author FROM sources WHERE title = ?", (title,)
            ).fetchall()
            for row in rows:
                existing = sorted(json.loads(row["author"]))
                if json.dumps(existing) == author_json:
                    return True
            return False
        else:
            row = self.conn.execute(
                "SELECT 1 FROM sources WHERE title = ? LIMIT 1", (title,)
            ).fetchone()
            return row is not None

    # --- Chunk CRUD ---

    def insert_chunk(self, chunk: Chunk) -> str:
        """Insert a single Chunk. Returns the chunk ID."""
        d = chunk.to_dict()
        self.conn.execute(
            """INSERT INTO chunks
               (id, source_id, text, language, structural_ref, pdf_page,
                youtube_timestamp, youtube_url, paragraph_index, paired_chunk_id,
                lecturer, lecture_number, chunk_method, sharing_status)
               VALUES (:id, :source_id, :text, :language, :structural_ref, :pdf_page,
                       :youtube_timestamp, :youtube_url, :paragraph_index, :paired_chunk_id,
                       :lecturer, :lecture_number, :chunk_method, :sharing_status)""",
            d,
        )
        self.conn.commit()
        return chunk.id

    def insert_chunks_batch(self, chunks: list[Chunk]) -> list[str]:
        """Insert multiple Chunks in a single transaction. Returns chunk IDs."""
        ids = []
        dicts = [c.to_dict() for c in chunks]
        self.conn.executemany(
            """INSERT INTO chunks
               (id, source_id, text, language, structural_ref, pdf_page,
                youtube_timestamp, youtube_url, paragraph_index, paired_chunk_id,
                lecturer, lecture_number, chunk_method, sharing_status)
               VALUES (:id, :source_id, :text, :language, :structural_ref, :pdf_page,
                       :youtube_timestamp, :youtube_url, :paragraph_index, :paired_chunk_id,
                       :lecturer, :lecture_number, :chunk_method, :sharing_status)""",
            dicts,
        )
        self.conn.commit()
        return [c.id for c in chunks]

    def get_chunks_by_source(self, source_id: str) -> list[Chunk]:
        """Fetch all Chunks for a given Source."""
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE source_id = ? ORDER BY paragraph_index",
            (source_id,),
        ).fetchall()
        return [Chunk.from_dict(dict(r)) for r in rows]

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        """Fetch Chunks by a list of IDs. Preserves input order."""
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self.conn.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders})", chunk_ids
        ).fetchall()
        # Preserve input order
        by_id = {dict(r)["id"]: Chunk.from_dict(dict(r)) for r in rows}
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def get_chunks_by_structural_ref(self, **path_filters: str) -> list[Chunk]:
        """Query chunks by structural_ref JSON paths.

        Example: get_chunks_by_structural_ref(doctrine="Being", section="§132")
        generates: WHERE json_extract(structural_ref, '$.doctrine') = 'Being'
                   AND json_extract(structural_ref, '$.section') = '§132'
        """
        if not path_filters:
            return []

        conditions = []
        params = []
        for key, value in path_filters.items():
            conditions.append(f"json_extract(structural_ref, '$.{key}') = ?")
            params.append(value)

        where = " AND ".join(conditions)
        rows = self.conn.execute(
            f"SELECT * FROM chunks WHERE {where} ORDER BY paragraph_index", params
        ).fetchall()
        return [Chunk.from_dict(dict(r)) for r in rows]

    def search_metadata(self, **filters: Any) -> list[Chunk]:
        """Query chunks by metadata filters.

        Supported filters: lecturer, language, source_type (joins sources),
        chunk_method, source_id.
        """
        conditions = []
        params: list[Any] = []
        join_sources = False

        for key, value in filters.items():
            if value is None:
                continue
            if key == "lecturer":
                conditions.append("c.lecturer = ?")
                params.append(value)
            elif key == "language":
                conditions.append("c.language = ?")
                params.append(value)
            elif key == "chunk_method":
                conditions.append("c.chunk_method = ?")
                params.append(value)
            elif key == "source_id":
                conditions.append("c.source_id = ?")
                params.append(value)
            elif key == "source_type":
                join_sources = True
                conditions.append("s.type = ?")
                params.append(value)

        if join_sources:
            query = "SELECT c.* FROM chunks c JOIN sources s ON c.source_id = s.id"
        else:
            query = "SELECT c.* FROM chunks c"

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY c.paragraph_index"
        rows = self.conn.execute(query, params).fetchall()
        return [Chunk.from_dict(dict(r)) for r in rows]

    # --- Chunk updates ---

    def update_chunk_structural_ref(self, chunk_id: str, structural_ref: dict | None) -> None:
        """Update a chunk's structural_ref in place."""
        import json

        ref_json = json.dumps(structural_ref) if structural_ref else None
        self.conn.execute(
            "UPDATE chunks SET structural_ref = ? WHERE id = ?",
            (ref_json, chunk_id),
        )
        self.conn.commit()

    def delete_source_and_chunks(self, source_id: str) -> int:
        """Remove a source and all its chunks. Returns chunk count deleted."""
        count = self.conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_id = ?", (source_id,)
        ).fetchone()[0]
        self.conn.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
        self.conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))
        self.conn.commit()
        return count

    # --- Work/Edition queries ---

    def get_sources_by_work_id(self, work_id: str) -> list[Source]:
        """Get all sources for a work (across editions/versions)."""
        rows = self.conn.execute(
            "SELECT * FROM sources WHERE json_extract(metadata, '$.work_id') = ?",
            (work_id,),
        ).fetchall()
        return [Source.from_dict(dict(r)) for r in rows]

    def get_chunks_by_canonical_section(self, canonical_section: str) -> list[Chunk]:
        """Get chunks across all editions that share a canonical section ID."""
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE json_extract(structural_ref, '$.canonical_section') = ? ORDER BY paragraph_index",
            (canonical_section,),
        ).fetchall()
        return [Chunk.from_dict(dict(r)) for r in rows]

    # --- Analysis CRUD ---

    def insert_analysis(self, analysis: Analysis) -> str:
        """Insert an Analysis record. Returns the analysis ID."""
        d = analysis.to_dict()
        self.conn.execute(
            """INSERT INTO analyses
               (id, title, content, linked_chunks, linked_analyses,
                tags, sharing_status, created_at, updated_at)
               VALUES (:id, :title, :content, :linked_chunks, :linked_analyses,
                       :tags, :sharing_status, :created_at, :updated_at)""",
            d,
        )
        self.conn.commit()
        return analysis.id

    def get_analysis(self, analysis_id: str) -> Analysis | None:
        """Fetch an Analysis by ID."""
        row = self.conn.execute(
            "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
        ).fetchone()
        if row is None:
            return None
        return Analysis.from_dict(dict(row))

    # --- Statistics ---

    def get_stats(self) -> dict[str, Any]:
        """Return corpus statistics."""
        source_count = self.conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        chunk_count = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        analysis_count = self.conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]

        # Counts by source type
        type_rows = self.conn.execute(
            "SELECT type, COUNT(*) as cnt FROM sources GROUP BY type"
        ).fetchall()
        sources_by_type = {r["type"]: r["cnt"] for r in type_rows}

        # Counts by language
        lang_rows = self.conn.execute(
            "SELECT language, COUNT(*) as cnt FROM chunks GROUP BY language"
        ).fetchall()
        chunks_by_language = {r["language"]: r["cnt"] for r in lang_rows}

        # Lecturer stats
        lect_rows = self.conn.execute(
            "SELECT lecturer, COUNT(*) as cnt FROM chunks WHERE lecturer IS NOT NULL GROUP BY lecturer"
        ).fetchall()
        chunks_by_lecturer = {r["lecturer"]: r["cnt"] for r in lect_rows}

        return {
            "sources": source_count,
            "chunks": chunk_count,
            "analyses": analysis_count,
            "sources_by_type": sources_by_type,
            "chunks_by_language": chunks_by_language,
            "chunks_by_lecturer": chunks_by_lecturer,
        }
