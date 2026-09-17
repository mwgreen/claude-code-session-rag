"""
Parse Claude Code JSONL transcripts into indexable chunks.

Transcript entries (one JSON object per line) that matter:
  type "user"       - a prompt. `message.content` is a string (or a list of blocks;
                      tool_result blocks are ignored, text blocks are kept).
  type "assistant"  - `message.content` is a list of text / tool_use / thinking blocks.
  type "summary"    - compaction summary (short title).
  type "ai-title"   - the session title Claude Code generates.
Everything else (system, attachment, progress, mode, ...) is skipped.

Turn assembly: a user prompt opens a turn; following assistant text blocks are
appended; tool_use blocks are summarised into an "Actions:" list (file paths,
commands) so "which file did we change for X" is searchable. Long turns are
split into several chunks that each repeat the user prompt, instead of being
truncated.

Noise: slash-command echoes, local command output, task notifications and
injected <system-reminder> blocks are stripped; prompts that are only noise are
not indexed.

Incremental reading: the caller passes the byte offset it has indexed up to.
Only complete lines (terminated by a newline) are consumed, so a line that is
still being written is picked up on the next pass instead of being lost.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

MAX_CHUNK_CHARS = 6000
USER_HEADER_CHARS = 1500
MAX_CHUNKS_PER_TURN = 8
MAX_ACTIONS_CHARS = 1200
MIN_CHUNK_CHARS = 20

_SKIP_TYPES = frozenset({
    "file-history-snapshot", "progress", "system", "queue-operation", "mode",
    "permission-mode", "atis-latch", "attachment", "last-prompt",
})

_NOISE_TAGS = (
    "system-reminder", "command-name", "command-message", "command-args",
    "local-command-stdout", "local-command-stderr", "local-command-caveat",
    "task-notification", "bash-input", "bash-stdout", "bash-stderr",
    "ide_opened_file", "ide_selection",
)
_TAG_ALT = "|".join(re.escape(t) for t in _NOISE_TAGS)
_TAG_BLOCK_RE = re.compile(rf"<({_TAG_ALT})\b[^>]*>.*?</\1\s*>", re.S)
_TAG_STRAY_RE = re.compile(rf"</?({_TAG_ALT})\b[^>]*>")
_SENTINEL_RE = re.compile(r"^\s*<<[^<>]+>>\s*$")
_MULTI_NL_RE = re.compile(r"\n{3,}")

_PATH_TOOLS = frozenset({"Edit", "MultiEdit", "Write", "Read", "NotebookEdit"})


def clean_user_text(text: str) -> str:
    """Strip Claude Code's injected/echoed markup from a user prompt."""
    if not text:
        return ""
    cleaned = _TAG_BLOCK_RE.sub(" ", text)
    cleaned = _TAG_STRAY_RE.sub(" ", cleaned)
    if _SENTINEL_RE.match(cleaned):
        return ""
    cleaned = _MULTI_NL_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def describe_tool_use(block: Dict) -> Optional[str]:
    """One-line summary of a tool_use block, or None if it is not worth indexing."""
    name = str(block.get("name") or "")
    inp = block.get("input") or {}
    if not isinstance(inp, dict):
        inp = {}
    if not name:
        return None
    if name in _PATH_TOOLS:
        path = inp.get("file_path") or inp.get("notebook_path") or ""
        return f"{name} {path}".strip()
    if name == "Bash":
        cmd = " ".join(str(inp.get("command", "")).split())
        return f"Bash: {cmd[:160]}" if cmd else None
    if name in ("Grep", "Glob"):
        return f"{name} {inp.get('pattern', '')}".strip()
    if name in ("Agent", "Task"):
        desc = str(inp.get("description") or "").strip()
        return f"{name}: {desc[:160]}" if desc else name
    if name == "Skill":
        return f"Skill {inp.get('skill', '')}".strip()
    if name == "WebFetch":
        return f"WebFetch {inp.get('url', '')}".strip()
    if name == "WebSearch":
        return f"WebSearch {inp.get('query', '')}".strip()
    if name in ("TodoWrite", "AskUserQuestion", "ToolSearch", "ExitPlanMode", "EnterPlanMode"):
        return None
    return name


def _format_actions(actions: List[str]) -> str:
    if not actions:
        return ""
    seen = set()
    unique: List[str] = []
    for a in actions:
        if a not in seen:
            seen.add(a)
            unique.append(a)
    lines = []
    used = len("Actions:\n")
    for i, a in enumerate(unique):
        line = f"- {a}"
        if used + len(line) + 1 > MAX_ACTIONS_CHARS:
            lines.append(f"- ... (+{len(unique) - i} more)")
            break
        lines.append(line)
        used += len(line) + 1
    return "Actions:\n" + "\n".join(lines)


def _split_text(text: str, budget: int) -> List[str]:
    """Split on paragraph/line/word boundaries so pieces stay <= budget chars."""
    pieces: List[str] = []
    budget = max(200, budget)
    while len(text) > budget:
        lo = budget // 2
        cut = text.rfind("\n\n", lo, budget)
        if cut == -1:
            cut = text.rfind("\n", lo, budget)
        if cut == -1:
            cut = text.rfind(" ", lo, budget)
        if cut == -1:
            cut = budget
        pieces.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        pieces.append(text)
    return pieces


@dataclass
class _PendingTurn:
    user_text: str
    start_byte: int
    timestamp: str = ""
    git_branch: str = ""
    assistant_texts: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)


def _make_chunk(text: str, session_id: str, transcript_file: str, turn_index: int,
                timestamp: str, git_branch: str, chunk_type: str) -> Dict:
    content_hash = hashlib.sha256(f"{turn_index}:{text}".encode("utf-8")).hexdigest()[:16]
    return {
        "text": text,
        "doc_id": f"{session_id}::{content_hash}",
        "session_id": session_id,
        "transcript_file": transcript_file,
        "turn_index": turn_index,
        "timestamp": timestamp,
        "git_branch": git_branch,
        "chunk_type": chunk_type,
    }


def build_turn_chunks(pending: _PendingTurn, session_id: str, transcript_file: str,
                      chunk_type: str, max_chunk_chars: int = MAX_CHUNK_CHARS) -> List[Dict]:
    """Turn one user/assistant exchange into one or more chunks."""
    user = pending.user_text.strip()
    assistant = "\n\n".join(t for t in pending.assistant_texts if t).strip()
    actions = _format_actions(pending.actions)

    parts = [f"User: {user}"]
    if assistant:
        parts.append(f"Assistant: {assistant}")
    if actions:
        parts.append(actions)
    full = "\n\n".join(parts)
    if len(full.strip()) < MIN_CHUNK_CHARS:
        return []

    common = dict(session_id=session_id, transcript_file=transcript_file,
                  timestamp=pending.timestamp, git_branch=pending.git_branch,
                  chunk_type=chunk_type)
    if len(full) <= max_chunk_chars:
        return [_make_chunk(full, turn_index=pending.start_byte, **common)]

    # Long turn: every chunk repeats (the start of) the prompt for context.
    header = f"User: {user[:USER_HEADER_CHARS]}" + (" …" if len(user) > USER_HEADER_CHARS else "")
    body_parts = []
    if len(user) > USER_HEADER_CHARS:
        body_parts.append("User (continued): " + user[USER_HEADER_CHARS:])
    if assistant:
        body_parts.append(assistant)
    if actions:
        body_parts.append(actions)
    body = "\n\n".join(body_parts)
    budget = max_chunk_chars - len(header) - 40
    pieces = _split_text(body, budget)
    if len(pieces) > MAX_CHUNKS_PER_TURN:
        pieces = pieces[:MAX_CHUNKS_PER_TURN]
        pieces[-1] = pieces[-1][: max(0, budget - 20)] + "\n\n[truncated]"
    n = len(pieces)
    chunks = []
    for i, piece in enumerate(pieces):
        text = f"{header}\n\nAssistant (part {i + 1}/{n}): {piece}"
        chunks.append(_make_chunk(text, turn_index=pending.start_byte + i, **common))
    return chunks


@dataclass
class ParseResult:
    turns: List[Dict]
    new_offset: int
    cwd: Optional[str] = None
    git_branch: str = ""


def _user_text_from_content(content) -> str:
    if isinstance(content, str):
        return clean_user_text(content)
    if isinstance(content, list):
        texts = [b.get("text", "") for b in content
                 if isinstance(b, dict) and b.get("type") == "text"]
        return clean_user_text("\n\n".join(t for t in texts if t))
    return ""


def parse_transcript(
    transcript_path: str,
    session_id: str,
    start_offset: int = 0,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
    chunk_type: str = "turn",
    include_actions: bool = True,
) -> ParseResult:
    """Parse new complete lines of a transcript starting at `start_offset`.

    Returns the chunks plus the byte offset just past the last complete line.
    `chunk_type` is "turn" for a session's main transcript and e.g. "subagent"
    for nested subagent transcripts.
    """
    file_size = os.path.getsize(transcript_path)
    if start_offset > file_size:
        start_offset = 0  # file was truncated or rewritten
    if start_offset >= file_size:
        return ParseResult([], file_size)

    with open(transcript_path, "rb") as f:
        f.seek(start_offset)
        data = f.read()

    last_nl = data.rfind(b"\n")
    if last_nl == -1:
        return ParseResult([], start_offset)  # partial line still being written
    data = data[: last_nl + 1]
    new_offset = start_offset + last_nl + 1

    transcript_file = os.path.basename(transcript_path)
    turns: List[Dict] = []
    seen_summaries: set = set()
    pending: Optional[_PendingTurn] = None
    timestamp = ""
    git_branch = ""
    cwd: Optional[str] = None

    def flush():
        nonlocal pending
        if pending is not None:
            turns.extend(build_turn_chunks(pending, session_id, transcript_file,
                                           chunk_type, max_chunk_chars))
            pending = None

    pos = start_offset
    for raw in data.split(b"\n"):
        line_start = pos
        pos += len(raw) + 1
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(entry, dict):
            continue

        entry_type = entry.get("type", "")
        if entry.get("gitBranch"):
            git_branch = entry["gitBranch"]
        if entry.get("timestamp"):
            timestamp = entry["timestamp"]
        if cwd is None and entry.get("cwd"):
            cwd = entry["cwd"]

        if entry_type in _SKIP_TYPES:
            continue

        if entry_type in ("summary", "ai-title"):
            if entry_type == "summary":
                flush()
                title = str(entry.get("summary") or "").strip()
                label = "Session Summary"
            else:
                title = str(entry.get("aiTitle") or "").strip()
                label = "Session Title"
            if title:
                text = f"{label}: {title}"
                key = hashlib.sha256(f"summary:{text}".encode("utf-8")).hexdigest()[:16]
                doc_id = f"{session_id}::{key}"
                if doc_id not in seen_summaries:
                    seen_summaries.add(doc_id)
                    chunk = _make_chunk(text, session_id, transcript_file, line_start,
                                        timestamp, git_branch, "summary")
                    chunk["doc_id"] = doc_id  # same title in the same session indexes once
                    turns.append(chunk)
            continue

        if entry_type == "user":
            if entry.get("isMeta"):
                continue
            message = entry.get("message") or {}
            text = _user_text_from_content(message.get("content"))
            if not text:
                continue
            flush()
            pending = _PendingTurn(user_text=text, start_byte=line_start,
                                   timestamp=timestamp, git_branch=git_branch)
            continue

        if entry_type == "assistant":
            if pending is None:
                continue  # assistant text without a prompt (resumed mid-turn) is skipped
            message = entry.get("message") or {}
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text = str(block.get("text") or "").strip()
                    if text:
                        pending.assistant_texts.append(text)
                elif btype == "tool_use" and include_actions:
                    desc = describe_tool_use(block)
                    if desc:
                        pending.actions.append(desc)
            # Keep the turn's timestamp at the latest assistant activity.
            if timestamp:
                pending.timestamp = timestamp

    flush()
    return ParseResult(turns, new_offset, cwd, git_branch)
