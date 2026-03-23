"""Core data model for Extracosmic Commons.

Three entities form the foundation:
- Source: a document, transcript, or text in the system
- Chunk: an embedded unit of text with structural metadata
- Analysis: a research note linked to source chunks

The structural_ref field on Chunk is a flexible JSON object that accommodates
different organizational hierarchies across works (Hegel §, Kant CPR divisions,
generic chapter/section). This avoids hardcoding any single author's structure
into the schema while enabling rich structural queries via SQLite json_extract().
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np


class SourceType(str, Enum):
    """Type of source document."""

    PDF = "pdf"
    TRANSCRIPT = "transcript"
    BILINGUAL_PAIR = "bilingual_pair"
    NOTE = "note"
    ARTICLE = "article"


class SharingStatus(str, Enum):
    """Sharing level for the cooperative tier."""

    LOCAL_ONLY = "local_only"
    SHARED_EMBEDDINGS = "shared_embeddings"
    SHARED_FULL = "shared_full"


def _new_id() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _now() -> datetime:
    """Current UTC timestamp."""
    return datetime.now(timezone.utc)


@dataclass
class Source:
    """A document, transcript, or text in the system.

    Represents the top-level container for ingested content. Each Source
    produces one or more Chunks when ingested.
    """

    title: str
    type: SourceType
    id: str = field(default_factory=_new_id)
    author: list[str] = field(default_factory=list)
    language: list[str] = field(default_factory=lambda: ["en"])
    edition: str | None = None
    source_path: str | None = None
    source_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    sharing_status: SharingStatus = SharingStatus.LOCAL_ONLY
    created_at: datetime = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary suitable for SQLite storage."""
        return {
            "id": self.id,
            "type": self.type.value,
            "title": self.title,
            "author": json.dumps(self.author),
            "language": json.dumps(self.language),
            "edition": self.edition,
            "source_path": self.source_path,
            "source_url": self.source_url,
            "metadata": json.dumps(self.metadata),
            "sharing_status": self.sharing_status.value,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Source:
        """Deserialize from a dictionary (SQLite row)."""
        return cls(
            id=d["id"],
            type=SourceType(d["type"]),
            title=d["title"],
            author=json.loads(d["author"]) if isinstance(d["author"], str) else d["author"],
            language=json.loads(d["language"]) if isinstance(d["language"], str) else d["language"],
            edition=d.get("edition"),
            source_path=d.get("source_path"),
            source_url=d.get("source_url"),
            metadata=json.loads(d["metadata"]) if isinstance(d.get("metadata"), str) else d.get("metadata", {}),
            sharing_status=SharingStatus(d["sharing_status"]),
            created_at=datetime.fromisoformat(d["created_at"]),
        )


@dataclass
class Chunk:
    """An embedded unit of text with rich structural metadata.

    Chunks are the atomic unit of retrieval. Each chunk belongs to exactly
    one Source and may be paired with a chunk in another language for
    bilingual retrieval.

    The structural_ref field is a flexible JSON object — different works
    have different hierarchies:
        Hegel SoL: {"section": "§132", "doctrine": "Being", "chapter": "..."}
        Kant CPR:  {"division": "Transcendental Analytic", "section": "§15"}
        Article:   {"section": "3.2", "heading": "Methodology"}
        None:      for chunks not yet structurally tagged
    """

    source_id: str
    text: str
    id: str = field(default_factory=_new_id)
    language: str = "en"

    # Structural metadata — flexible JSON, not schema-bound
    structural_ref: dict[str, Any] | None = None

    # Location within source
    pdf_page: int | None = None
    youtube_timestamp: str | None = None
    youtube_url: str | None = None
    paragraph_index: int | None = None

    # Bilingual pairing
    paired_chunk_id: str | None = None

    # Lecture-specific
    lecturer: str | None = None
    lecture_number: int | None = None

    # Provenance
    chunk_method: str = "unknown"
    sharing_status: SharingStatus = SharingStatus.LOCAL_ONLY

    # Embedding — stored in FAISS, not SQLite
    embedding: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for SQLite storage. Excludes embedding."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "text": self.text,
            "language": self.language,
            "structural_ref": json.dumps(self.structural_ref) if self.structural_ref else None,
            "pdf_page": self.pdf_page,
            "youtube_timestamp": self.youtube_timestamp,
            "youtube_url": self.youtube_url,
            "paragraph_index": self.paragraph_index,
            "paired_chunk_id": self.paired_chunk_id,
            "lecturer": self.lecturer,
            "lecture_number": self.lecture_number,
            "chunk_method": self.chunk_method,
            "sharing_status": self.sharing_status.value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chunk:
        """Deserialize from a dictionary (SQLite row)."""
        structural_ref = d.get("structural_ref")
        if isinstance(structural_ref, str):
            structural_ref = json.loads(structural_ref)

        return cls(
            id=d["id"],
            source_id=d["source_id"],
            text=d["text"],
            language=d.get("language", "en"),
            structural_ref=structural_ref,
            pdf_page=d.get("pdf_page"),
            youtube_timestamp=d.get("youtube_timestamp"),
            youtube_url=d.get("youtube_url"),
            paragraph_index=d.get("paragraph_index"),
            paired_chunk_id=d.get("paired_chunk_id"),
            lecturer=d.get("lecturer"),
            lecture_number=d.get("lecture_number"),
            chunk_method=d.get("chunk_method", "unknown"),
            sharing_status=SharingStatus(d.get("sharing_status", "local_only")),
        )


@dataclass
class Analysis:
    """A research note linked to source chunks.

    Analyses are the user's own work — interpretive notes, arguments,
    comparisons — grounded in specific chunks from the corpus. Wiki-style
    linking between analyses enables building a web of interpretation.
    """

    title: str
    content: str  # markdown
    id: str = field(default_factory=_new_id)
    linked_chunks: list[str] = field(default_factory=list)
    linked_analyses: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    sharing_status: SharingStatus = SharingStatus.LOCAL_ONLY
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for SQLite storage."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "linked_chunks": json.dumps(self.linked_chunks),
            "linked_analyses": json.dumps(self.linked_analyses),
            "tags": json.dumps(self.tags),
            "sharing_status": self.sharing_status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Analysis:
        """Deserialize from a dictionary (SQLite row)."""
        return cls(
            id=d["id"],
            title=d["title"],
            content=d["content"],
            linked_chunks=json.loads(d["linked_chunks"]) if isinstance(d["linked_chunks"], str) else d["linked_chunks"],
            linked_analyses=json.loads(d["linked_analyses"]) if isinstance(d["linked_analyses"], str) else d["linked_analyses"],
            tags=json.loads(d["tags"]) if isinstance(d["tags"], str) else d["tags"],
            sharing_status=SharingStatus(d["sharing_status"]),
            created_at=datetime.fromisoformat(d["created_at"]),
            updated_at=datetime.fromisoformat(d["updated_at"]),
        )
