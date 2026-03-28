"""
Tests for the drafts.py auto-tagging and Drafts app integration module.

Covers all four tag categories (Project, Status, Context, Claude-authored),
URL construction, and the send_to_drafts function (with subprocess mocked).
"""

import urllib.parse
from unittest.mock import patch, MagicMock

import pytest

from extracosmic_commons.drafts import (
    infer_project_tags,
    infer_status_tags,
    infer_context_tags,
    infer_all_tags,
    build_drafts_url,
    send_to_drafts,
    send_pipeline_report,
)


# -----------------------------------------------------------------------
# Project tag inference
# -----------------------------------------------------------------------

class TestInferProjectTags:
    """Tests for project tag inference from content keywords."""

    def test_extracosmic_keywords(self):
        """FAISS and search UI keywords should tag as Extracosmic Commons."""
        assert "Extracosmic Commons" in infer_project_tags("Rebuilt the FAISS index")
        assert "Extracosmic Commons" in infer_project_tags("Updated the search UI")

    def test_pipeline_keywords(self):
        """Pipeline and overnight keywords should tag as Pipeline."""
        assert "Pipeline" in infer_project_tags("Overnight pipeline completed")
        assert "Pipeline" in infer_project_tags("Stage 4 re-ingestion done")

    def test_groq_keywords(self):
        """Groq and LLM cleanup keywords should tag as Groq."""
        assert "Groq" in infer_project_tags("Switched to llama-3.1-8b-instant on Groq")
        assert "Groq" in infer_project_tags("LLM cleanup finished")

    def test_runpod_keywords(self):
        """RunPod and GPU keywords should tag as RunPod."""
        assert "RunPod" in infer_project_tags("RunPod GPU diarization complete")
        assert "RunPod" in infer_project_tags("whisperx transcription on A6000")

    def test_citation_keywords(self):
        """Citation keywords should tag as Citations."""
        assert "Citations" in infer_project_tags("841 citations extracted")
        assert "Citations" in infer_project_tags("Cross-reference pass complete")

    def test_multiple_projects(self):
        """A single text can match multiple project tags."""
        tags = infer_project_tags("Pipeline re-ingested into FAISS index")
        assert "Pipeline" in tags
        assert "Extracosmic Commons" in tags

    def test_no_match(self):
        """Text with no project keywords should return empty list."""
        assert infer_project_tags("Hello world") == []


# -----------------------------------------------------------------------
# Status tag inference
# -----------------------------------------------------------------------

class TestInferStatusTags:
    """Tests for status tag inference from content keywords."""

    def test_success_keywords(self):
        """Success/complete/finished should tag as Success."""
        assert "Success" in infer_status_tags("Pipeline completed successfully")
        assert "Success" in infer_status_tags("All 37 files done with zero errors")

    def test_zero_errors_not_tagged_error(self):
        """'zero errors' should NOT trigger the Error tag — it's a success."""
        assert "Error" not in infer_status_tags("Finished with zero errors")
        assert "Error" not in infer_status_tags("no error found")

    def test_error_keywords(self):
        """Error/fail/exception should tag as Error."""
        assert "Error" in infer_status_tags("CUDA OOM error on file 25")
        assert "Error" in infer_status_tags("Traceback in stage 3")

    def test_progress_keywords(self):
        """Progress/running/processing should tag as Progress."""
        assert "Progress" in infer_status_tags("Currently processing stage 2 of 6")
        assert "Progress" in infer_status_tags("Pipeline running on RunPod")

    def test_cost_keywords(self):
        """Cost/invoice/billing should tag as Cost."""
        assert "Cost" in infer_status_tags("Groq invoice was $8.89")
        assert "Cost" in infer_status_tags("Billing on pay-as-you-go tier")

    def test_warning_keywords(self):
        """Warning/caution should tag as Warning."""
        assert "Warning" in infer_status_tags("Unexpected cost on Groq account")

    def test_no_match(self):
        """Text with no status keywords should return empty list."""
        assert infer_status_tags("The weather is nice") == []


# -----------------------------------------------------------------------
# Content tag inference
# -----------------------------------------------------------------------

class TestInferContextTags:
    """Tests for specific, compound context tag inference."""

    def test_citation_extractor_compound(self):
        """'citation extractor' should produce the compound context tag."""
        assert "citation extractor" in infer_context_tags("Refactored the citation extractor")

    def test_groq_llm_cleanup_compound(self):
        """'Groq LLM cleanup' should produce the compound context tag."""
        assert "Groq LLM cleanup" in infer_context_tags("Processed through Groq LLM cleanup")

    def test_cuda_oom_compound(self):
        """'CUDA OOM' and 'CUDA out-of-memory' should produce the compound tag."""
        assert "CUDA OOM" in infer_context_tags("Hit a CUDA OOM error")
        assert "CUDA OOM" in infer_context_tags("CUDA out-of-memory on file 25")

    def test_specific_model_names(self):
        """Specific model names should be extracted as tags."""
        assert "llama-3.1-8b-instant" in infer_context_tags("Using llama-3.1-8b-instant")
        assert "llama-3.3-70b-versatile" in infer_context_tags("Switched from llama-3.3-70b-versatile")

    def test_specific_tools(self):
        """Specific tools like whisperx, diarization, FAISS should tag."""
        assert "whisperx" in infer_context_tags("whisperx transcription complete")
        assert "diarization" in infer_context_tags("Re-diarization of all 37 files")
        assert "FAISS" in infer_context_tags("Rebuilt the FAISS index")

    def test_testing_tag(self):
        """Test/pytest keywords should still tag as Testing."""
        assert "Testing" in infer_context_tags("All 32 pytest tests passing")

    def test_git_tag(self):
        """Git keywords should tag as Git."""
        assert "Git" in infer_context_tags("Committed and pushed to main")

    def test_multiple_context_tags(self):
        """A single text can match multiple specific context tags."""
        tags = infer_context_tags("whisperx diarization on RunPod with CUDA OOM")
        assert "whisperx" in tags
        assert "diarization" in tags
        assert "CUDA OOM" in tags

    def test_no_match(self):
        """Text with no context keywords should return empty list."""
        assert infer_context_tags("The weather is nice") == []


# -----------------------------------------------------------------------
# Combined tag inference (infer_all_tags)
# -----------------------------------------------------------------------

class TestInferAllTags:
    """Tests for the combined tag inference across all categories."""

    def test_claude_authored_always_present(self):
        """Claude-authored tag must always be the last tag, even for empty text."""
        tags = infer_all_tags("")
        assert tags[-1] == "Claude-authored"

    def test_claude_authored_on_plain_text(self):
        """Even text with no keyword matches should get Claude-authored."""
        tags = infer_all_tags("Just a plain message")
        assert tags == ["Claude-authored"]

    def test_all_categories_combined(self):
        """A rich text should produce tags from all three inferred categories + Claude-authored."""
        text = "Pipeline completed successfully: Groq LLM cleanup using llama-3.1-8b-instant"
        tags = infer_all_tags(text)
        # Project tags
        assert "Pipeline" in tags
        assert "Groq" in tags
        # Status tags
        assert "Success" in tags
        # Context tags
        assert "Groq LLM cleanup" in tags
        assert "llama-3.1-8b-instant" in tags
        # Always present
        assert "Claude-authored" in tags
        assert tags[-1] == "Claude-authored"

    def test_deduplication(self):
        """Tags should not appear more than once even if matched by multiple rules."""
        text = "Testing the pytest suite passed"
        tags = infer_all_tags(text)
        testing_count = tags.count("Testing")
        assert testing_count == 1, f"'Testing' appeared {testing_count} times, expected 1"


# -----------------------------------------------------------------------
# URL construction
# -----------------------------------------------------------------------

class TestBuildDraftsUrl:
    """Tests for Drafts URL scheme construction."""

    def test_url_starts_with_scheme(self):
        """URL must start with the Drafts x-callback-url scheme."""
        url = build_drafts_url("hello", tags=["TestTag"])
        assert url.startswith("drafts://x-callback-url/create?")

    def test_text_is_encoded(self):
        """The text parameter should be URL-encoded in the query string."""
        url = build_drafts_url("hello world", tags=["T"])
        assert "text=hello%20world" in url

    def test_tags_appear_as_repeated_params(self):
        """Each tag should appear as a separate &tag= parameter."""
        url = build_drafts_url("hi", tags=["Alpha", "Beta", "Claude-authored"])
        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        assert "Alpha" in parsed["tag"]
        assert "Beta" in parsed["tag"]
        assert "Claude-authored" in parsed["tag"]

    def test_claude_authored_added_to_explicit_tags(self):
        """Even with explicit tags, Claude-authored should be appended if missing."""
        url = build_drafts_url("hi", tags=["CustomTag"])
        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        assert "Claude-authored" in parsed["tag"]

    def test_auto_infer_when_no_tags(self):
        """When tags=None, tags should be auto-inferred from text."""
        url = build_drafts_url("Pipeline completed with zero errors")
        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        assert "Pipeline" in parsed["tag"]
        assert "Success" in parsed["tag"]
        assert "Claude-authored" in parsed["tag"]

    def test_folder_param(self):
        """The folder parameter should appear in the URL."""
        url = build_drafts_url("hi", tags=["T"], folder="Archive")
        assert "folder=Archive" in url


# -----------------------------------------------------------------------
# send_to_drafts (mocked subprocess)
# -----------------------------------------------------------------------

class TestSendToDrafts:
    """Tests for the send_to_drafts function with mocked subprocess."""

    @patch("extracosmic_commons.drafts.subprocess.run")
    def test_calls_open_with_url(self, mock_run):
        """send_to_drafts should call `open` with a drafts:// URL."""
        mock_run.return_value = MagicMock(returncode=0)
        result = send_to_drafts("Test message", timestamp=False)
        assert result is True
        args = mock_run.call_args[0][0]
        assert args[0] == "open"
        assert args[1].startswith("drafts://x-callback-url/create?")

    @patch("extracosmic_commons.drafts.subprocess.run")
    def test_returns_false_on_failure(self, mock_run):
        """send_to_drafts should return False when open fails."""
        mock_run.return_value = MagicMock(returncode=1)
        result = send_to_drafts("Test message", timestamp=False)
        assert result is False

    @patch("extracosmic_commons.drafts.subprocess.run")
    def test_timestamp_prepended(self, mock_run):
        """When timestamp=True, a 'Created:' line should be in the URL text."""
        mock_run.return_value = MagicMock(returncode=0)
        send_to_drafts("Test message", timestamp=True)
        url = mock_run.call_args[0][0][1]
        decoded = urllib.parse.unquote(url)
        assert "Created:" in decoded

    @patch("extracosmic_commons.drafts.subprocess.run")
    def test_no_timestamp_when_disabled(self, mock_run):
        """When timestamp=False, no 'Created:' line should appear."""
        mock_run.return_value = MagicMock(returncode=0)
        send_to_drafts("Test message", timestamp=False)
        url = mock_run.call_args[0][0][1]
        decoded = urllib.parse.unquote(url)
        assert "Created:" not in decoded

    @patch("extracosmic_commons.drafts.subprocess.run")
    def test_claude_authored_always_in_url(self, mock_run):
        """Every draft must have the Claude-authored tag in the URL."""
        mock_run.return_value = MagicMock(returncode=0)
        send_to_drafts("Just a note", timestamp=False)
        url = mock_run.call_args[0][0][1]
        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        assert "Claude-authored" in parsed["tag"]

    @patch("extracosmic_commons.drafts.subprocess.run")
    def test_handles_timeout(self, mock_run):
        """send_to_drafts should handle subprocess timeout gracefully."""
        from subprocess import TimeoutExpired
        mock_run.side_effect = TimeoutExpired(cmd="open", timeout=10)
        result = send_to_drafts("Test", timestamp=False)
        assert result is False


# -----------------------------------------------------------------------
# send_pipeline_report convenience wrapper
# -----------------------------------------------------------------------

class TestSendPipelineReport:
    """Tests for the pipeline report convenience wrapper."""

    @patch("extracosmic_commons.drafts.subprocess.run")
    def test_sends_report_with_auto_tags(self, mock_run):
        """Pipeline report should be sent with auto-inferred tags."""
        mock_run.return_value = MagicMock(returncode=0)
        report = "Overnight pipeline completed: 37 files, 841 citations, zero errors"
        result = send_pipeline_report(report)
        assert result is True
        url = mock_run.call_args[0][0][1]
        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        assert "Pipeline" in parsed["tag"]
        assert "Success" in parsed["tag"]
        assert "Claude-authored" in parsed["tag"]
