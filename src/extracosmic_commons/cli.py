"""Command-line interface for Extracosmic Commons.

Usage:
    ec ingest <file>                    # Ingest a single file
    ec ingest <directory> [--recursive] # Ingest all files in a directory
    ec import-zotero <path>             # Import a Zotero export directory
    ec import-workbench <path>          # Import from scholarly workbench
    ec search <query> [options]         # Semantic search
    ec stats                            # Corpus statistics
"""

from __future__ import annotations

import os
from pathlib import Path

import click

from .database import Database
from .embeddings import EmbeddingPipeline
from .index import FAISSIndex


def _get_data_dir() -> Path:
    """Resolve the data directory from env or default."""
    return Path(os.environ.get("EC_DATA_DIR", "data"))


def _init_components(data_dir: Path | None = None):
    """Initialize database, embedder, and index."""
    dd = data_dir or _get_data_dir()
    db = Database(dd / "extracosmic.db")
    embedder = EmbeddingPipeline()
    index_path = dd / "faiss_index"
    if (index_path / "index.faiss").exists():
        index = FAISSIndex(index_path=index_path)
    else:
        index = FAISSIndex(dimension=1024)  # BGE-M3 dimension
    return db, embedder, index, dd


@click.group()
@click.option("--data-dir", type=click.Path(), envvar="EC_DATA_DIR", default=None)
@click.pass_context
def main(ctx, data_dir):
    """Extracosmic Commons — scholarly research platform."""
    ctx.ensure_object(dict)
    if data_dir:
        ctx.obj["data_dir"] = Path(data_dir)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--recursive", "-r", is_flag=True, help="Recurse into subdirectories")
@click.pass_context
def ingest(ctx, path, recursive):
    """Ingest a file or directory."""
    from .ingest.bilingual import BilingualIngester
    from .ingest.pdf import PDFIngester
    from .ingest.transcript import TranscriptIngester

    data_dir = ctx.obj.get("data_dir")
    db, embedder, index, dd = _init_components(data_dir)

    target = Path(path)

    if target.is_file():
        files = [target]
    elif target.is_dir():
        pattern = "**/*" if recursive else "*"
        files = sorted(target.glob(pattern))
        files = [f for f in files if f.is_file()]
    else:
        click.echo(f"Error: {path} is not a file or directory", err=True)
        return

    # Filter to supported extensions
    supported = {".md", ".pdf", ".json"}
    files = [f for f in files if f.suffix.lower() in supported]

    if not files:
        click.echo("No supported files found (.md, .pdf, .json)")
        return

    transcript_ingester = TranscriptIngester()
    pdf_ingester = PDFIngester()
    bilingual_ingester = BilingualIngester()

    total_chunks = 0
    for f in files:
        try:
            if f.suffix.lower() == ".md":
                source = transcript_ingester.ingest(f, db, embedder, index)
                chunks = db.get_chunks_by_source(source.id)
                click.echo(f"  Ingested transcript: {source.title} ({len(chunks)} chunks)")
                total_chunks += len(chunks)
            elif f.suffix.lower() == ".pdf":
                source = pdf_ingester.ingest(f, db, embedder, index)
                chunks = db.get_chunks_by_source(source.id)
                click.echo(f"  Ingested PDF: {source.title} ({len(chunks)} chunks)")
                total_chunks += len(chunks)
            elif f.suffix.lower() == ".json" and "clean_text" in f.name:
                de_src, en_src = bilingual_ingester.ingest(f, db, embedder, index)
                de_chunks = db.get_chunks_by_source(de_src.id)
                en_chunks = db.get_chunks_by_source(en_src.id)
                n = len(de_chunks) + len(en_chunks)
                click.echo(f"  Ingested bilingual: {de_src.title} + {en_src.title} ({n} chunks)")
                total_chunks += n
        except Exception as e:
            click.echo(f"  Failed: {f.name}: {e}", err=True)

    # Save index
    index.save(dd / "faiss_index")
    click.echo(f"\nDone. {len(files)} files, {total_chunks} chunks total.")


@main.command("import-zotero")
@click.argument("path", type=click.Path(exists=True))
@click.pass_context
def import_zotero(ctx, path):
    """Import a Zotero export directory."""
    from .ingest.zotero import ZoteroImporter

    data_dir = ctx.obj.get("data_dir")
    db, embedder, index, dd = _init_components(data_dir)

    importer = ZoteroImporter()

    def progress(done, total, title):
        if done < total:
            click.echo(f"  [{done}/{total}] {title}")
        else:
            click.echo(f"  [{done}/{total}] Complete")

    sources = importer.import_collection(
        Path(path), db, embedder, index, progress_callback=progress,
    )

    index.save(dd / "faiss_index")
    click.echo(f"\nImported {len(sources)} documents from Zotero collection.")


@main.command("import-workbench")
@click.argument("path", type=click.Path(exists=True))
@click.pass_context
def import_workbench(ctx, path):
    """Import documents from scholarly workbench."""
    from .ingest.workbench import WorkbenchImporter

    data_dir = ctx.obj.get("data_dir")
    db, embedder, index, dd = _init_components(data_dir)

    importer = WorkbenchImporter()

    def progress(done, total, title):
        if done < total:
            click.echo(f"  [{done}/{total}] {title}")
        else:
            click.echo(f"  [{done}/{total}] Complete")

    sources = importer.import_all(
        Path(path), db, embedder, index, progress_callback=progress,
    )

    index.save(dd / "faiss_index")
    click.echo(f"\nImported {len(sources)} documents from scholarly workbench.")


@main.command()
@click.argument("query")
@click.option("-k", "--top-k", default=10, help="Number of results")
@click.option("--lecturer", help="Filter by lecturer name")
@click.option("--language", help="Filter by language code (en, de)")
@click.option("--type", "source_type", help="Filter by source type (pdf, transcript)")
@click.option("--bilingual", is_flag=True, help="Show bilingual pairs")
@click.option("--ref", multiple=True, help="Structural ref filter (key=value)")
@click.pass_context
def search(ctx, query, top_k, lecturer, language, source_type, bilingual, ref):
    """Search the corpus."""
    from .search import SearchEngine

    data_dir = ctx.obj.get("data_dir")
    db, embedder, index, dd = _init_components(data_dir)
    engine = SearchEngine(db, embedder, index)

    filters = {}
    if lecturer:
        filters["lecturer"] = lecturer
    if language:
        filters["language"] = language
    if source_type:
        filters["source_type"] = source_type
    if ref:
        structural = {}
        for r in ref:
            if "=" in r:
                k, v = r.split("=", 1)
                structural[k] = v
        if structural:
            filters["structural_ref"] = structural

    results = engine.search(query, top_k=top_k, bilingual=bilingual, **filters)

    if not results:
        click.echo("No results found.")
        return

    for i, r in enumerate(results, 1):
        click.echo(f"\n{'─' * 60}")
        click.echo(f"  [{i}] Score: {r.score:.4f}")
        click.echo(f"  Source: {r.source.title}")
        if r.source.author:
            click.echo(f"  Author: {', '.join(r.source.author)}")

        # Location info
        if r.chunk.youtube_url:
            click.echo(f"  Timestamp: [{r.chunk.youtube_timestamp}]({r.chunk.youtube_url})")
        if r.chunk.pdf_page:
            click.echo(f"  Page: {r.chunk.pdf_page}")
        if r.chunk.lecturer:
            click.echo(f"  Lecturer: {r.chunk.lecturer}, Lecture {r.chunk.lecture_number}")
        if r.chunk.structural_ref:
            ref_str = ", ".join(f"{k}={v}" for k, v in r.chunk.structural_ref.items())
            click.echo(f"  Ref: {ref_str}")

        # Text snippet (truncated)
        text = r.chunk.text[:300]
        if len(r.chunk.text) > 300:
            text += "..."
        click.echo(f"\n  {text}")

        # Bilingual pair
        if r.paired_chunk:
            lang = r.paired_chunk.language.upper()
            paired_text = r.paired_chunk.text[:200]
            if len(r.paired_chunk.text) > 200:
                paired_text += "..."
            click.echo(f"\n  [{lang}] {paired_text}")

    click.echo(f"\n{'─' * 60}")
    click.echo(f"{len(results)} results for: {query}")


@main.command()
@click.pass_context
def stats(ctx):
    """Show corpus statistics."""
    data_dir = ctx.obj.get("data_dir")
    dd = data_dir or _get_data_dir()
    db = Database(dd / "extracosmic.db")

    s = db.get_stats()

    click.echo("\n  Extracosmic Commons — Corpus Statistics\n")
    click.echo(f"  Sources:   {s['sources']}")
    click.echo(f"  Chunks:    {s['chunks']}")
    click.echo(f"  Analyses:  {s['analyses']}")

    if s["sources_by_type"]:
        click.echo("\n  By source type:")
        for t, c in sorted(s["sources_by_type"].items()):
            click.echo(f"    {t}: {c}")

    if s["chunks_by_language"]:
        click.echo("\n  By language:")
        for l, c in sorted(s["chunks_by_language"].items()):
            click.echo(f"    {l}: {c}")

    if s["chunks_by_lecturer"]:
        click.echo("\n  By lecturer:")
        for l, c in sorted(s["chunks_by_lecturer"].items()):
            click.echo(f"    {l}: {c}")

    # FAISS index size
    index_path = dd / "faiss_index" / "index.faiss"
    if index_path.exists():
        size_mb = index_path.stat().st_size / (1024 * 1024)
        click.echo(f"\n  FAISS index: {size_mb:.1f} MB")

    click.echo()
    db.close()
