#!/usr/bin/env python3
"""Optimized batch ingestion for Wu conversation transcripts.

Key optimization: instead of calling the embedding subprocess once per file
(which reloads the 1.5GB BGE-M3 model each time), this script:
1. Parses ALL remaining transcripts into chunks (fast, no model needed)
2. Collects ALL chunk texts across all files
3. Embeds everything in a SINGLE subprocess call (one model load)
4. Batch-inserts all sources, chunks, and FAISS vectors

This reduces 27 model loads (~13 min overhead) to 1 model load (~30 sec).
"""

import sys
import os
import glob
import json
import time
import tempfile
import subprocess
import numpy as np
from pathlib import Path

# Add the project src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from extracosmic_commons.database import Database
from extracosmic_commons.index import FAISSIndex
from extracosmic_commons.ingest.conversation import ConversationIngester

# ── Paths ────────────────────────────────────────────────────────────────
TRANSCRIPTS_DIR = Path("/Volumes/External SSD/podcast-pipeline/01-transcripts-llm-cleaned")
AUDIO_DIR = Path("/Volumes/External SSD/podcast-pipeline/00-raw")
DATA_DIR = Path(__file__).parent.parent / "data"

# ── Transcript filename → Audio filename mapping ─────────────────────────
AUDIO_MAP = {
    "Wu_2025.06.23":        "Wu 6.23.2025.m4a",
    "Wu_2025.07.07":        "Wu 7.7.2025.m4a",
    "Wu_2025.07.12":        "Wu 7.12.2025.m4a",
    "Wu_2025.07.19":        "Wu 7.19.2025.m4a",
    "Wu_2025.08.03":        "Wu 8.3.2025.m4a",
    "Wu_2025.08.03_2":      "Wu 8.3.2025 2.m4a",
    "Wu_2025.08.10":        "Wu 8.10.2025.m4a",
    "Wu_2025.08.17":        "Wu 8.17.2025.m4a",
    "Wu_2025.08.17_2":      "Wu 8.17.2025 2.m4a",
    "Wu_2025.08.24":        "Wu 8.24.2025.m4a",
    "Wu_2025.08.24_A":      "Wu 8.24.2025 A.m4a",
    "Wu_2025.08.31":        "Wu 8.31.2025.m4a",
    "Wu_2025.09.05_HeideggerK": "Wu 9.5.2025 Heidegger Kant Strauss.m4a",
    "Wu_2025.09.07":        "Wu 9.7.2025.m4a",
    "Wu_2025.09.15_1":      "Wu 2025.9.15 1.m4a",
    "Wu_2025.09.15_2":      "Wu 2025.9.15 2.m4a",
    "Wu_2025.09.20":        "Wu 2025.9.20.m4a",
    "Wu_2025.09.27":        "Wu 2025.9.27.m4a",
    "Wu_2025.10.04_1":      "Wu 2025.10.4 1.m4a",
    "Wu_2025.10.04_2":      "Wu 2025.10.4 2.m4a",
    "Wu_2025.10.11":        "Wu 2025.10.11.m4a",
    "Wu_2025.10.17":        "Wu 2025.10.17.m4a",
    "Wu_2025.10.25":        "Wu 2025.10.25.m4a",
    "Wu_2025.10.28":        "Wu 2025.10.28.m4a",
    "Wu_2025.10.29":        "Wu 2025.10.29.m4a",
    "Wu_2025.11.06":        "Wu 2025.11.6.m4a",
    "Wu_2025.11.13":        "Wu 2025.11.13.m4a",
    "Wu_2025.11.20":        "Wu 2025.11.20.m4a",
    "Wu_2025.12.06":        "Wu 2025.12.6.m4a",
    "Wu_2025.12.14":        "Wu 2025.12.14.m4a",
    "Wu_2025.12.29":        "Wu 2025.12.29.m4a",
    "Wu_2026.01.10":        "Wu 2026.1.10.m4a",
    "Wu_2026.03.01":        "Wu 2026.03.01.m4a",
    "Wu_2026.03.14":        "Wu 2026.03.14.m4a",
    "Wu_2026.03.15":        "Wu 2026.03.15.m4a",
    "Wu+Tony_2025.09.14":   "Wu and Tone 9.14.25.m4a",
    "Wu+Tony_2026.02.07":   "Wu and Tone 2026.02.07.m4a",
}


def embed_all_texts_subprocess(texts: list[str], model_name: str = "BAAI/bge-m3", batch_size: int = 32) -> np.ndarray:
    """Embed ALL texts in a single subprocess call.
    
    This loads the model once and embeds everything, avoiding the per-file
    model reload overhead that makes the default pipeline so slow on macOS.
    """
    if not texts:
        return np.zeros((0, 1024), dtype=np.float32)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        texts_path = Path(tmpdir) / "texts.json"
        emb_path = Path(tmpdir) / "embeddings.npy"
        
        # Write all texts at once
        with open(texts_path, 'w', encoding='utf-8') as f:
            json.dump(texts, f)
        
        script = f"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer

print(f"Loading model...", flush=True)
t0 = time.time()
model = SentenceTransformer('{model_name}', device='mps')
print(f"Model loaded in {{time.time()-t0:.1f}}s", flush=True)

texts = json.loads(open('{texts_path}', encoding='utf-8').read())
print(f"Embedding {{len(texts)}} texts with batch_size={batch_size}...", flush=True)

t1 = time.time()
embeddings = model.encode(
    texts,
    batch_size={batch_size},
    normalize_embeddings=True,
    show_progress_bar=True,
    device='mps',
)
elapsed = time.time() - t1
print(f"Embedded {{len(texts)}} texts in {{elapsed:.1f}}s ({{len(texts)/elapsed:.1f}} texts/sec)", flush=True)

np.save('{emb_path}', embeddings.astype(np.float32))
print("Saved.", flush=True)
"""
        print(f"Starting embedding subprocess for {len(texts)} texts...")
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min max
        )
        
        # Print subprocess output
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                print(f"  [embed] {line}")
        
        if result.returncode != 0:
            print(f"  [embed] STDERR: {result.stderr[-500:]}")
            raise RuntimeError(f"Embedding subprocess failed: {result.stderr[-200:]}")
        
        return np.load(str(emb_path))


def main():
    t_start = time.time()
    
    # Find all Wu transcript JSONs
    json_files = sorted(glob.glob(str(TRANSCRIPTS_DIR / "Wu*.json")))
    print(f"Found {len(json_files)} Wu transcript files\n")
    
    if not json_files:
        print("No files found!")
        return
    
    # ── Phase 1: Parse all files, skip already-ingested ──────────────────
    print("Phase 1: Parsing transcripts and checking for already-ingested files...")
    db = Database(DATA_DIR / "extracosmic.db")
    ingester = ConversationIngester()
    
    # Collect all (source, chunks, basename) tuples for files that need ingestion
    pending = []
    skipped = 0
    
    for jp in json_files:
        basename = Path(jp).stem
        
        # Check if already ingested
        date_str = basename.replace("Wu_", "").replace("Wu+Tony_", "")
        existing_title = f"Wu Conversation — {date_str}" if "Tony" not in basename else f"Wu & Tony Conversation — {date_str}"
        
        if db.source_exists(existing_title):
            print(f"  SKIP: {basename}")
            skipped += 1
            continue
        
        # Look up audio file
        audio_filename = AUDIO_MAP.get(basename)
        audio_path = None
        if audio_filename:
            ap = str(AUDIO_DIR / audio_filename)
            if Path(ap).exists():
                audio_path = ap
            else:
                print(f"  WARNING: Audio not found: {ap}")
        else:
            print(f"  WARNING: No audio mapping for {basename}")
        
        # Parse into source + chunks
        source, chunks = ingester.parse_conversation(Path(jp), audio_path)
        pending.append((basename, source, chunks))
        print(f"  PARSE: {basename} — {len(chunks)} chunks")
    
    if not pending:
        print(f"\nAll files already ingested! ({skipped} skipped)")
        return
    
    # Flatten all chunk texts for batch embedding
    all_texts = []
    chunk_boundaries = []  # (start_idx, end_idx) for each file's chunks
    for basename, source, chunks in pending:
        start = len(all_texts)
        all_texts.extend(c.text for c in chunks)
        chunk_boundaries.append((start, len(all_texts)))
    
    total_chunks = len(all_texts)
    print(f"\nPhase 1 complete: {len(pending)} files to ingest, {total_chunks} chunks total, {skipped} already done\n")
    
    # ── Phase 2: Embed ALL chunks in one subprocess call ─────────────────
    print(f"Phase 2: Embedding {total_chunks} chunks in a single model load...")
    t_embed = time.time()
    
    # Use MPS (Apple Silicon GPU) with larger batch size
    all_embeddings = embed_all_texts_subprocess(all_texts, batch_size=64)
    
    embed_time = time.time() - t_embed
    print(f"\nPhase 2 complete: {total_chunks} chunks embedded in {embed_time:.1f}s\n")
    
    # ── Phase 3: Store everything in DB + FAISS ──────────────────────────
    print("Phase 3: Storing in database and FAISS index...")
    
    index_path = DATA_DIR / "faiss_index"
    if (index_path / "index.faiss").exists():
        index = FAISSIndex(index_path=index_path)
    else:
        index = FAISSIndex(dimension=1024)
    
    for i, (basename, source, chunks) in enumerate(pending):
        start_idx, end_idx = chunk_boundaries[i]
        file_embeddings = all_embeddings[start_idx:end_idx]
        
        # Store source
        db.insert_source(source)
        
        # Store chunks
        db.insert_chunks_batch(chunks)
        
        # Add to FAISS
        chunk_ids = [c.id for c in chunks]
        index.add_batch(chunk_ids, file_embeddings)
        
        print(f"  STORED: {basename} — {len(chunks)} chunks")
    
    # Save FAISS index
    index_path.mkdir(parents=True, exist_ok=True)
    index.save(index_path)
    print(f"\nFAISS index saved to {index_path}")
    
    total_time = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"COMPLETE in {total_time:.1f}s")
    print(f"  Files ingested: {len(pending)}")
    print(f"  Chunks embedded+stored: {total_chunks}")
    print(f"  Files skipped: {skipped}")
    print(f"  Embedding time: {embed_time:.1f}s ({total_chunks/embed_time:.1f} chunks/sec)")


if __name__ == "__main__":
    main()
