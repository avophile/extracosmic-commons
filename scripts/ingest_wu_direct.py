#!/usr/bin/env python3
"""Direct ingestion — no subprocess, no libomp conflict.

This standalone script loads SentenceTransformer directly. Since FAISS
is only used for index.add_batch (numpy arrays), we can safely import
both in the same process by setting KMP_DUPLICATE_LIB_OK=TRUE.

This avoids the 30min subprocess timeout issue entirely.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys
import glob
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from extracosmic_commons.database import Database
from extracosmic_commons.index import FAISSIndex
from extracosmic_commons.ingest.conversation import ConversationIngester

TRANSCRIPTS_DIR = Path("/Volumes/External SSD/podcast-pipeline/01-transcripts-llm-cleaned")
AUDIO_DIR = Path("/Volumes/External SSD/podcast-pipeline/00-raw")
DATA_DIR = Path(__file__).parent.parent / "data"

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


def main():
    t_start = time.time()
    
    json_files = sorted(glob.glob(str(TRANSCRIPTS_DIR / "Wu*.json")))
    print(f"Found {len(json_files)} Wu transcript files\n")
    
    # ── Phase 1: Parse ──────────────────────────────────────────────────
    print("Phase 1: Parsing transcripts...")
    db = Database(DATA_DIR / "extracosmic.db")
    ingester = ConversationIngester()
    
    pending = []
    skipped = 0
    
    for jp in json_files:
        basename = Path(jp).stem
        date_str = basename.replace("Wu_", "").replace("Wu+Tony_", "")
        existing_title = f"Wu Conversation — {date_str}" if "Tony" not in basename else f"Wu & Tony Conversation — {date_str}"
        
        if db.source_exists(existing_title):
            print(f"  SKIP: {basename}")
            skipped += 1
            continue
        
        audio_filename = AUDIO_MAP.get(basename)
        audio_path = None
        if audio_filename:
            ap = str(AUDIO_DIR / audio_filename)
            if Path(ap).exists():
                audio_path = ap
        
        source, chunks = ingester.parse_conversation(Path(jp), audio_path)
        pending.append((basename, source, chunks))
        print(f"  PARSE: {basename} — {len(chunks)} chunks")
    
    if not pending:
        print(f"\nAll files already ingested! ({skipped} skipped)")
        return
    
    all_texts = []
    chunk_boundaries = []
    for basename, source, chunks in pending:
        start = len(all_texts)
        all_texts.extend(c.text for c in chunks)
        chunk_boundaries.append((start, len(all_texts)))
    
    total_chunks = len(all_texts)
    print(f"\nPhase 1 done: {len(pending)} files, {total_chunks} chunks, {skipped} skipped\n")
    
    # ── Phase 2: Load model and embed directly (no subprocess) ──────────
    print("Phase 2: Loading BGE-M3 model...")
    from sentence_transformers import SentenceTransformer
    
    t_load = time.time()
    model = SentenceTransformer('BAAI/bge-m3', device='cpu')
    print(f"Model loaded in {time.time()-t_load:.1f}s\n")
    
    print(f"Embedding {total_chunks} chunks...")
    t_embed = time.time()
    
    # Process in batches with progress reporting
    batch_size = 16
    all_embeddings = []
    for i in range(0, total_chunks, batch_size):
        batch = all_texts[i:i+batch_size]
        emb = model.encode(batch, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(emb)
        done = min(i + batch_size, total_chunks)
        elapsed = time.time() - t_embed
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total_chunks - done) / rate if rate > 0 else 0
        print(f"  {done}/{total_chunks} ({rate:.1f}/sec, ETA {eta:.0f}s)")
    
    all_embeddings = np.vstack(all_embeddings).astype(np.float32)
    embed_time = time.time() - t_embed
    print(f"\nEmbedding done: {embed_time:.1f}s ({total_chunks/embed_time:.1f} chunks/sec)\n")
    
    # ── Phase 3: Store ──────────────────────────────────────────────────
    print("Phase 3: Storing in DB + FAISS...")
    index_path = DATA_DIR / "faiss_index"
    if (index_path / "index.faiss").exists():
        index = FAISSIndex(index_path=index_path)
    else:
        index = FAISSIndex(dimension=1024)
    
    for i, (basename, source, chunks) in enumerate(pending):
        start_idx, end_idx = chunk_boundaries[i]
        file_embeddings = all_embeddings[start_idx:end_idx]
        db.insert_source(source)
        db.insert_chunks_batch(chunks)
        chunk_ids = [c.id for c in chunks]
        index.add_batch(chunk_ids, file_embeddings)
        print(f"  STORED: {basename} — {len(chunks)} chunks")
    
    index_path.mkdir(parents=True, exist_ok=True)
    index.save(index_path)
    
    total_time = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"COMPLETE in {total_time:.1f}s")
    print(f"  Files: {len(pending)} ingested, {skipped} skipped")
    print(f"  Chunks: {total_chunks}")
    print(f"  Embed: {embed_time:.1f}s ({total_chunks/embed_time:.1f}/sec)")


if __name__ == "__main__":
    main()
