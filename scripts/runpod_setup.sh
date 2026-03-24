#!/bin/bash
# RunPod setup script for Extracosmic Commons corpus ingestion
#
# Usage:
#   1. Spin up a RunPod GPU pod (A40 recommended, cheapest that works well)
#   2. Upload your corpus (see below)
#   3. Run this script
#   4. Run the ingestion
#   5. Download data/ back to your Mac
#
# Upload corpus to RunPod:
#   rsync -avz --include='*/' --include='*.pdf' --include='*.rdf' --exclude='*' \
#     ~/Documents/HegelTranscripts/ runpod:/workspace/corpus/HegelTranscripts/
#
#   rsync -avz ~/Documents/scholarly-workbench-integrated/backend/user_files/ \
#     runpod:/workspace/corpus/workbench/backend/user_files/
#
#   rsync -avz ~/Desktop/Hegel/ runpod:/workspace/corpus/Desktop_Hegel/

set -e

echo "=== Extracosmic Commons — RunPod Setup ==="

# Clone repo
cd /workspace
if [ ! -d extracosmic-commons ]; then
    git clone https://github.com/avophile/extracosmic-commons.git
fi
cd extracosmic-commons

# Install
pip install -e ".[dev]"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Now run the ingestion:"
echo ""
echo "  python scripts/cloud_ingest.py \\"
echo "    --data-dir data \\"
echo "    --zotero '/workspace/corpus/HegelTranscripts/Hegel texts' \\"
echo "             '/workspace/corpus/HegelTranscripts/Radnik texts' \\"
echo "             '/workspace/corpus/HegelTranscripts/Thompson texts' \\"
echo "             '/workspace/corpus/HegelTranscripts/Houlgate texts' \\"
echo "    --workbench /workspace/corpus/workbench \\"
echo "    --pdfs /workspace/corpus/Desktop_Hegel"
echo ""
echo "When done, download data/ back to your Mac:"
echo ""
echo "  rsync -avz runpod:/workspace/extracosmic-commons/data/ ~/Documents/Extracosmic_Commons/data/"
echo ""
