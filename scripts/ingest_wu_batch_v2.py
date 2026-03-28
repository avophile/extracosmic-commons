#!/usr/bin/env python3
"""Optimized batch ingestion v2 — fixes venv Python path and uses CPU.

Key fixes over v1:
- Uses explicit venv Python path for subprocess (not sys.executable)
- Uses CPU device (MPS hangs on large batches on M2)
- Smaller batch_size=16 for stability
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

# The venv python — hardcoded to avoid sys.executable resolving to system python
VENV_PYTHON = "/Users/douglaslocklin/Documents/Extracosmic_Commons/.venv/bin/python3"

# Add the project src to path
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


def embed_all_texts_subprocess(texts, batch_size=16):
    """Embed ALL texts in a single subprocess using the VENV python."""
    if not texts:
        return np.zeros((0, 1024), dtype=np.float32)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        texts_path = Path(tmpdir) / "texts.json"
        emb_path = Path(tmpdir) / "embeddings.npy"
        
        with open(texts_path, 'w', encoding='utf-8') as f:
            json.dump(texts, f)
        
        # Write the embedding script to a file (avoids quoting issues)
        script_path = Path(tmpdir) / "embed_script.py"
        script_path.write_text(f'''
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import time
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading BGE-M3 model on CPU...", flush=True)
t0 = time.time()
model = SentenceTransformer('BAAI/bge-m3', device='cpu')
print(f"Model loaded in {{time.time()-t0:.1f}}s", flush=True)

with open('{texts_path}', 'r', encoding='utf-8') as f:
    texts = json.load(f)
print(f"Embedding {{len(texts)}} texts (batch_size={batch_size})...", flush=True)

t1 = time.time()
# Process in sub-batches and report progress
all_emb = []
bs = {batch_size}
for i in range(0, len(texts), bs):
    batch = texts[i:i+bs]
    emb = model.encode(batch, batch_size=bs, normalize_embeddings=True, show_progress_bar=False)
    all_emb.append(emb)
    done = min(i+bs, len(texts))
    elapsed = time.time() - t1
    rate = done / elapsed if elapsed > 0 else 0
    eta = (len(texts) - done) / rate if rate > 0 else 0
    print(f"  {{done}}/{{len(texts)}} ({{rate:.1f}} chunks/sec, ETA {{eta:.0f}}s)", flush=True)

embeddings = np.vstack(all_emb)
elapsed = time.time() - t1
print(f"Done: {{len(texts)}} texts in {{elapsed:.1f}}s ({{len(texts)/elapsed:.1f}} chunks/sec)", flush=True)

np.save('{emb_path}', embeddings.astype(np.float32))
print("Saved embeddings.", flush=True)
''')
        
        print(f"Starting embedding subprocess for {len(texts)} texts...")
        print(f"Using: {VENV_PYTHON}")
        
        result = subprocess.run(
            [VENV_PYTHON, str(script_path)],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                print(f"  [embed] {line}")
        
        if result.returncode != 0:
            print(f"  [embed] STDERR: {result.stderr[-1000:]}")
            raise RuntimeError(f"Embedding subprocess failed (rc={result.returncode})")
        
        return np.load(str(emb_path))


def main():
    t_start = time.time()
    
    json_files = sorted(glob.glob(str(TRANSCRIPTS_DIR / "Wu*.json")))
    print(f"Found {len(json_files)} Wu transcript files\n")
    
    if not json_files:
        print("No files found!")
        return
    
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
    
    print(f"Phase 2: Embedding {total_chunks} chunks (single model load, CPU)...")
    t_embed = time.time()
    all_embeddings = embed_all_texts_subprocess(all_texts, batch_size=16)
    embed_time = time.time() - t_embed
    print(f"\nPhase 2 done: {embed_time:.1f}s\n")
    
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
    print(f"\nFAISS index saved.")
    
    total_time = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"COMPLETE in {total_time:.1f}s")
    print(f"  Files ingested: {len(pending)}")
    print(f"  Chunks: {total_chunks}")
    print(f"  Skipped: {skipped}")
    print(f"  Embed time: {embed_time:.1f}s ({total_chunks/embed_time:.1f} chunks/sec)")


if __name__ == "__main__":
    main()
