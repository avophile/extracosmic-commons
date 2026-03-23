# Extracosmic Commons

Privacy-first, cooperatively owned scholarly research platform. Citation-aware, multilingual RAG for philosophical and academic texts.

## Quick Start

```bash
pip install -e ".[dev]"
pytest
```

## CLI

```bash
ec ingest <file>           # Ingest a transcript, PDF, or bilingual JSON
ec import-zotero <path>    # Import a Zotero export directory
ec search <query>          # Semantic search across the corpus
ec stats                   # Show corpus statistics
```
