"""Extracosmic Commons — privacy-first, cooperatively owned scholarly research platform."""

import os
import platform

# macOS: must set these BEFORE torch/faiss are imported to avoid:
# 1. libomp duplicate library crash (FAISS + PyTorch both link libomp)
# 2. Tokenizer fork segfault (tokenizers multiprocessing + libomp conflict)
if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__version__ = "0.1.0"
