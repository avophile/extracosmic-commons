#!/bin/bash
# ============================================================
# RunPod Setup for Extracosmic Commons Corpus Ingestion
# ============================================================
#
# STEP 1: On your Mac, upload corpus to RunPod
# ─────────────────────────────────────────────
# First, set your RunPod SSH alias (replace with your pod IP):
#   export RUNPOD="root@YOUR_POD_IP"
#
# Upload the PDFs and metadata:
#   rsync -avz --include='*/' --include='*.pdf' --include='*.rdf' --exclude='*' \
#     ~/Documents/HegelTranscripts/ $RUNPOD:/workspace/corpus/HegelTranscripts/
#
#   rsync -avz ~/Documents/scholarly-workbench-integrated/backend/user_files/ \
#     $RUNPOD:/workspace/corpus/workbench/backend/user_files/
#
#   rsync -avz ~/Desktop/Hegel/ $RUNPOD:/workspace/corpus/Desktop_Hegel/
#
# Upload transcripts + bilingual:
#   rsync -avz ~/Documents/HegelTranscripts/*_Transcript*.md \
#     $RUNPOD:/workspace/corpus/transcripts/
#   rsync -avz ~/Documents/HegelTranscripts/houlgate/*_Transcript*.md \
#     $RUNPOD:/workspace/corpus/transcripts/
#   rsync -avz ~/Documents/HegelTranscripts/radnik/*_Transcript*.md \
#     $RUNPOD:/workspace/corpus/transcripts/
#   rsync -avz ~/Documents/hegel-bilingual/clean_text.json \
#     $RUNPOD:/workspace/corpus/
#
# Upload existing local data (so we don't re-embed):
#   rsync -avz ~/Documents/Extracosmic_Commons/data/ \
#     $RUNPOD:/workspace/extracosmic-commons/data/
#
# STEP 2: On RunPod, run this script
# ───────────────────────────────────
#   bash /workspace/extracosmic-commons/scripts/runpod_setup.sh
#
# STEP 3: Run ingestion
# ─────────────────────
#   (see commands printed at the end of this script)
#
# STEP 4: Download results back to Mac
# ─────────────────────────────────────
#   rsync -avz $RUNPOD:/workspace/extracosmic-commons/data/ \
#     ~/Documents/Extracosmic_Commons/data/
# ============================================================

set -e

echo "=== Extracosmic Commons — RunPod Setup ==="
echo ""

# Clone repo
cd /workspace
if [ ! -d extracosmic-commons ]; then
    git clone https://github.com/avophile/extracosmic-commons.git
else
    cd extracosmic-commons && git pull && cd /workspace
fi
cd extracosmic-commons

# Install
pip install -e ".[dev]"

# Quick test
python -c "from extracosmic_commons.embeddings import EmbeddingPipeline; e = EmbeddingPipeline(); v = e.embed('test'); print(f'Embedding works: dim={v.shape[0]}, device={e.model.device}')"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Run the full ingestion with:"
echo ""
echo "python scripts/cloud_ingest.py --data-dir data \\"
echo "  --transcripts /workspace/corpus/transcripts/*.md \\"
echo "  --bilingual /workspace/corpus/clean_text.json \\"
echo "  --zotero '/workspace/corpus/HegelTranscripts/Hegel texts' \\"
echo "           '/workspace/corpus/HegelTranscripts/Radnik texts' \\"
echo "           '/workspace/corpus/HegelTranscripts/Thompson texts' \\"
echo "           '/workspace/corpus/HegelTranscripts/Houlgate texts' \\"
echo "  --workbench /workspace/corpus/workbench \\"
echo "  --pdfs /workspace/corpus/Desktop_Hegel"
echo ""
echo "When done, download data/ back to your Mac:"
echo "  rsync -avz \$RUNPOD:/workspace/extracosmic-commons/data/ ~/Documents/Extracosmic_Commons/data/"
echo ""
