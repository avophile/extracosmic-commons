"""Bilingual JSON ingestion from hegel-bilingual project.

Ingests the clean_text.json intermediate format, which contains aligned
German/English paragraphs from Hegel's Science of Logic with structural
metadata (GW21 page numbers, heading levels, section names).

Each German paragraph becomes a Chunk with language='de', and each English
paragraph becomes a Chunk with language='en'. Paired chunks are linked by
paired_chunk_id for bilingual retrieval.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..database import Database
from ..embeddings import EmbeddingPipeline
from ..index import FAISSIndex
from ..models import Chunk, Source, SourceType


def _heading_to_structural_ref(para: dict) -> dict | None:
    """Convert a heading paragraph to a structural_ref dict."""
    if para.get("type") != "heading":
        return None
    return {
        "section": para.get("text", "").strip(),
        "level": para.get("level", 0),
        "doctrine": "Being",  # Section I is within the Doctrine of Being
    }


def _body_to_structural_ref(para: dict) -> dict | None:
    """Create structural_ref for body paragraphs from gw_page."""
    gw = para.get("gw21_page")
    if gw:
        return {
            "gw_page": f"21.{gw}",
            "doctrine": "Being",
        }
    return None


class BilingualIngester:
    """Ingests bilingual Hegel JSON (clean_text.json format)."""

    def parse_bilingual_json(
        self, path: Path
    ) -> tuple[Source, Source, list[Chunk]]:
        """Parse clean_text.json into two Sources and paired Chunks.

        Returns (german_source, english_source, all_chunks).
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data.get("metadata", {})

        de_source = Source(
            title=meta.get("title_de", "Hegel — Wissenschaft der Logik (German)"),
            type=SourceType.BILINGUAL_PAIR,
            author=["Hegel"],
            language=["de"],
            edition=f"GW21 {meta.get('gw_range', '')}",
            source_path=str(path),
            metadata=meta,
        )

        en_source = Source(
            title=meta.get("title_en", "Hegel — Science of Logic (English)"),
            type=SourceType.BILINGUAL_PAIR,
            author=["Hegel", meta.get("translator", "di Giovanni")],
            language=["en"],
            edition=meta.get("edition", ""),
            source_path=str(path),
            metadata=meta,
        )

        de_paragraphs = data.get("de_paragraphs", [])
        en_paragraphs = data.get("en_paragraphs", [])

        all_chunks = []

        # Create German chunks
        de_chunks = []
        for i, para in enumerate(de_paragraphs):
            text = para.get("text", "").strip()
            if not text:
                continue

            structural_ref = (
                _heading_to_structural_ref(para) or _body_to_structural_ref(para)
            )

            de_chunks.append(Chunk(
                source_id=de_source.id,
                text=text,
                language="de",
                structural_ref=structural_ref,
                paragraph_index=i,
                chunk_method="bilingual_alignment",
            ))

        # Create English chunks
        en_chunks = []
        for i, para in enumerate(en_paragraphs):
            text = para.get("text", "").strip()
            if not text:
                continue

            en_chunks.append(Chunk(
                source_id=en_source.id,
                text=text,
                language="en",
                paragraph_index=i,
                chunk_method="bilingual_alignment",
            ))

        # Pair chunks by position (within the shorter list)
        pair_count = min(len(de_chunks), len(en_chunks))
        for i in range(pair_count):
            de_chunks[i].paired_chunk_id = en_chunks[i].id
            en_chunks[i].paired_chunk_id = de_chunks[i].id

        all_chunks.extend(de_chunks)
        all_chunks.extend(en_chunks)

        return de_source, en_source, all_chunks

    def ingest(
        self,
        path: Path,
        db: Database,
        embedder: EmbeddingPipeline,
        index: FAISSIndex,
    ) -> tuple[Source, Source]:
        """Parse, embed, and store bilingual content."""
        de_source, en_source, chunks = self.parse_bilingual_json(path)

        if not chunks:
            raise ValueError(f"No chunks extracted from {path}")

        texts = [c.text for c in chunks]
        embeddings = embedder.embed_batch(texts)

        db.insert_source(de_source)
        db.insert_source(en_source)
        db.insert_chunks_batch(chunks)
        index.add_batch([c.id for c in chunks], embeddings)

        return de_source, en_source
