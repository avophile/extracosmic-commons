"""
drafts.py — Auto-tagged draft creation for the Drafts app (Agile Tortoise).

Sends text to the Drafts app via its URL scheme with automatically inferred tags
across four categories:

1. **Project tags** — Which project this relates to (e.g., "Extracosmic Commons",
   "Pipeline", "Groq"). Inferred from content keywords.
2. **Status tags** — What kind of update this is (e.g., "Success", "Error",
   "Progress", "Complete"). Inferred from content keywords.
3. **Content tags** — What the content is about (e.g., "Citation", "Transcript",
   "Search", "Hegel"). Inferred from content keywords.
4. **Claude-authored** — A fixed tag applied to every draft created by this module,
   marking it as machine-generated.

URL scheme format:
    drafts://x-callback-url/create?text=TEXT&tag=TAG1&tag=TAG2&folder=inbox
"""

import re
import subprocess
import urllib.parse
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Tag inference rules
# ---------------------------------------------------------------------------

# PROJECT_TAG_RULES: keyword patterns → project tag.
# Each tuple is (compiled_regex, tag_string). First match wins per rule,
# but multiple rules can match (a draft can belong to multiple projects).
PROJECT_TAG_RULES = [
    (re.compile(r"extracosmic|faiss|bge-m3|search ui|web viewer", re.I), "Extracosmic Commons"),
    (re.compile(r"pipeline|overnight|stage \d|ingestion|re-ingest", re.I), "Pipeline"),
    (re.compile(r"groq|llm cleanup|free.?tier", re.I), "Groq"),
    (re.compile(r"runpod|gpu|whisperx|diariz", re.I), "RunPod"),
    (re.compile(r"citation|cross.?reference", re.I), "Citations"),
]

# STATUS_TAG_RULES: keyword patterns → status tag.
# These indicate the nature of the update (success, failure, progress, etc.).
STATUS_TAG_RULES = [
    (re.compile(r"success|completed|finished|done|pass(?:ed|ing)?|zero errors", re.I), "Success"),
    (re.compile(r"(?<!zero )(?<!no )error|fail|exception|traceback|crash", re.I), "Error"),
    (re.compile(r"progress|running|processing|stage \d of", re.I), "Progress"),
    (re.compile(r"warning|caution|attention|unexpected", re.I), "Warning"),
    (re.compile(r"cost|invoice|billing|\$\d", re.I), "Cost"),
]

# CONTEXT_TAG_RULES: specific, compound concepts extracted from content.
# These are precise, descriptive tags — the kind you'd actually search for.
# Ordered from most specific to least; multiple can match.
CONTEXT_TAG_RULES = [
    # Compound tool/concept tags (most specific first)
    (re.compile(r"citation extractor", re.I), "citation extractor"),
    (re.compile(r"groq llm cleanup|llm cleanup.*groq|groq.*cleanup", re.I), "Groq LLM cleanup"),
    (re.compile(r"cuda oom|cuda out.of.memory|gpu oom", re.I), "CUDA OOM"),
    (re.compile(r"search ui|web viewer|search interface", re.I), "search UI"),
    (re.compile(r"overnight pipeline", re.I), "overnight pipeline"),
    (re.compile(r"speaker.turn chunk", re.I), "speaker-turn chunking"),
    # Specific model names
    (re.compile(r"llama-3\.1-8b-instant", re.I), "llama-3.1-8b-instant"),
    (re.compile(r"llama-3\.3-70b-versatile", re.I), "llama-3.3-70b-versatile"),
    (re.compile(r"llama-4-scout", re.I), "llama-4-scout"),
    # Specific tools/technologies
    (re.compile(r"whisperx", re.I), "whisperx"),
    (re.compile(r"diariz(?:ation|ed|ing)", re.I), "diarization"),
    (re.compile(r"pyannote", re.I), "pyannote"),
    (re.compile(r"faiss", re.I), "FAISS"),
    (re.compile(r"bge-m3", re.I), "BGE-M3"),
    (re.compile(r"bm25", re.I), "BM25"),
    # Broader topic tags (only when no more specific tag covers it)
    (re.compile(r"test|pytest|assert|tdd", re.I), "Testing"),
    (re.compile(r"git|commit|push|branch|merge", re.I), "Git"),
]


def infer_project_tags(text: str) -> list[str]:
    """Scan text against PROJECT_TAG_RULES and return matching project tags.

    Each rule's regex is tested against the full text. Multiple rules can match,
    so a single draft can be tagged with multiple projects (e.g., both
    "Extracosmic Commons" and "Pipeline" if the text mentions both).

    Args:
        text: The draft content to scan.

    Returns:
        A deduplicated list of project tag strings.
    """
    tags = []
    for pattern, tag in PROJECT_TAG_RULES:
        if pattern.search(text):
            tags.append(tag)
    return list(dict.fromkeys(tags))  # deduplicate while preserving order


def infer_status_tags(text: str) -> list[str]:
    """Scan text against STATUS_TAG_RULES and return matching status tags.

    Status tags indicate the nature of the update — success, error, progress, etc.
    Multiple can match (e.g., a report might mention both successes and warnings).

    Args:
        text: The draft content to scan.

    Returns:
        A deduplicated list of status tag strings.
    """
    tags = []
    for pattern, tag in STATUS_TAG_RULES:
        if pattern.search(text):
            tags.append(tag)
    return list(dict.fromkeys(tags))


def infer_context_tags(text: str) -> list[str]:
    """Scan text against CONTEXT_TAG_RULES and return matching context tags.

    Context tags are specific, compound concepts extracted from the content —
    things like "citation extractor", "CUDA OOM", "llama-3.1-8b-instant".
    These are the precise, descriptive tags you'd actually search for.

    Args:
        text: The draft content to scan.

    Returns:
        A deduplicated list of context tag strings.
    """
    tags = []
    for pattern, tag in CONTEXT_TAG_RULES:
        if pattern.search(text):
            tags.append(tag)
    return list(dict.fromkeys(tags))


def infer_all_tags(text: str) -> list[str]:
    """Infer all tags for a draft across all four categories.

    Combines project, status, and content tags from keyword scanning,
    then appends the fixed "Claude-authored" tag. This is the main
    entry point for tag inference — call this to get the complete tag list.

    Args:
        text: The draft content to scan.

    Returns:
        A deduplicated list of all tags, with "Claude-authored" always last.
    """
    tags = []
    tags.extend(infer_project_tags(text))
    tags.extend(infer_status_tags(text))
    tags.extend(infer_context_tags(text))
    # Deduplicate (some rules overlap, e.g., "Error" in both status and content)
    tags = list(dict.fromkeys(tags))
    # Always add Claude-authored as the final tag
    tags.append("Claude-authored")
    return tags


def build_drafts_url(
    text: str,
    tags: Optional[list[str]] = None,
    folder: str = "Inbox",
) -> str:
    """Build a Drafts app URL scheme string for creating a new draft.

    Constructs a URL of the form:
        drafts://x-callback-url/create?text=...&tag=...&tag=...&folder=...

    If no tags are provided, they are automatically inferred from the text
    using infer_all_tags(). The "Claude-authored" tag is always included.

    Args:
        text: The body text of the draft.
        tags: Optional explicit tag list. If None, tags are auto-inferred.
        folder: Drafts folder to create in. Defaults to "Inbox".

    Returns:
        A fully encoded Drafts URL scheme string.
    """
    if tags is None:
        tags = infer_all_tags(text)
    elif "Claude-authored" not in tags:
        # Ensure Claude-authored is always present even with explicit tags
        tags = list(tags) + ["Claude-authored"]

    # Build query parameters. Drafts expects repeated &tag= params.
    params = [("text", text)]
    for tag in tags:
        params.append(("tag", tag))
    params.append(("folder", folder))

    query_string = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    return f"drafts://x-callback-url/create?{query_string}"


def send_to_drafts(
    text: str,
    tags: Optional[list[str]] = None,
    folder: str = "Inbox",
    timestamp: bool = True,
) -> bool:
    """Send text to the Drafts app as a new draft with auto-generated tags.

    This is the main public function. It infers tags from the text content
    across all four categories (Project, Status, Content, Claude-authored),
    builds the URL scheme string, and opens it via macOS `open` command,
    which triggers Drafts to create the new draft.

    If timestamp=True (default), a timestamp line is prepended to the text
    in the format "Created: 2026-03-28 14:30:00".

    Args:
        text: The body text of the draft.
        tags: Optional explicit tag list. If None, tags are auto-inferred
              from the text content. "Claude-authored" is always added.
        folder: Drafts folder to create in. Defaults to "Inbox".
        timestamp: Whether to prepend a creation timestamp. Defaults to True.

    Returns:
        True if the `open` command succeeded, False otherwise.
    """
    if timestamp:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = f"Created: {now}\n\n{text}"

    url = build_drafts_url(text, tags=tags, folder=folder)

    try:
        result = subprocess.run(
            ["open", url],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"Failed to send to Drafts: {e}")
        return False


def send_pipeline_report(
    report: str,
    folder: str = "Inbox",
) -> bool:
    """Convenience wrapper for sending pipeline completion reports to Drafts.

    Automatically tags the report based on its content, always includes
    "Claude-authored", and places it in the specified folder. This is
    intended as a drop-in replacement for the plain file-write notification
    in overnight_pipeline.py Stage 6.

    Args:
        report: The pipeline completion report text.
        folder: Drafts folder. Defaults to "Inbox".

    Returns:
        True if the draft was created successfully, False otherwise.
    """
    return send_to_drafts(report, folder=folder)
