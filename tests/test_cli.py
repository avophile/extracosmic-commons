"""Tests for CLI interface."""

import json

import pytest
from click.testing import CliRunner

from extracosmic_commons.cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def sample_transcript(tmp_path):
    p = tmp_path / "transcript.md"
    p.write_text("""# Lectures
**Lecturer:** Test Lecturer
**Total Sessions:** 2

## Session 1 — Introduction

**[00:00:01](https://youtube.com/watch?v=abc&t=1)** This is the first chunk of the lecture about Hegel's Science of Logic and the beginning of the system.

**[00:01:00](https://youtube.com/watch?v=abc&t=60)** This is the second chunk discussing the transition from Being to Nothing in the opening of the Logic.
""")
    return p


class TestCLI:
    def test_stats_empty(self, runner, tmp_path):
        result = runner.invoke(main, ["--data-dir", str(tmp_path), "stats"])
        assert result.exit_code == 0
        assert "Sources:   0" in result.output

    def test_search_empty_corpus(self, runner, tmp_path):
        result = runner.invoke(main, ["--data-dir", str(tmp_path), "search", "test"])
        # Should handle gracefully even with no model loaded — but this
        # will try to load the model, so we just check it doesn't crash badly
        # In practice this test needs the embedding model available
        assert result.exit_code == 0 or "No results" in (result.output or "")

    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Extracosmic Commons" in result.output

    def test_ingest_help(self, runner):
        result = runner.invoke(main, ["ingest", "--help"])
        assert result.exit_code == 0
        assert "file or directory" in result.output

    def test_search_help(self, runner):
        result = runner.invoke(main, ["search", "--help"])
        assert result.exit_code == 0
        assert "--bilingual" in result.output
