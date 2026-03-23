"""Shared test fixtures for Extracosmic Commons."""

import os

import pytest

# macOS: FAISS and PyTorch both link libomp, causing a duplicate library crash.
# This env var tells OpenMP to tolerate the duplicate.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Ephemeral data directory for tests that need file persistence."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir
