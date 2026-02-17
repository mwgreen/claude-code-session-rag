"""
Parse Claude Code JSONL transcripts into conversation turns for indexing.

Transcript format (one JSON object per line):
- type: "user"     → user message (content is str or list of tool_result)
- type: "assistant" → assistant message (content is list of text/tool_use/thinking blocks)
- type: "summary"  → compaction summary (short title string)
- type: "system", "file-history-snapshot", "progress", "queue-operation" → skip

Turn assembly:
1. A user entry with plain string content starts a new turn
2. Subsequent assistant text blocks are accumulated
3. Combined: "User: {user_text}\\n\\nAssistant: {assistant_text}"
4. Summary entries become standalone chunks

Incremental reading:
- index_state.json tracks last_byte_offset per transcript file
- On each invocation, seek to offset, read only new lines
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple


def parse_transcript(
    transcript_path: str,
    session_id: str,
    start_offset: int = 0,
    max_turn_chars: int = 8000,
) -> Tuple[List[Dict], int]:
    """Parse a Claude Code JSONL transcript into indexable turns.

    Args:
        transcript_path: Path to the .jsonl file
        session_id: Session UUID for this transcript
        start_offset: Byte offset to resume reading from
        max_turn_chars: Maximum characters per turn text (truncate longer)

    Returns:
        (turns, new_offset) where turns is a list of dicts ready for rag_engine.add_turns()
        and new_offset is the byte position after the last line read.
    """
    turns = []
    current_user_text = None
    current_assistant_texts = []
    current_timestamp = ""
    current_git_branch = ""
    turn_index = 0

    # Count existing turns from before this parse to continue turn_index
    # (For simplicity, we start at 0 within each incremental parse batch.
    #  The doc_id uses session_id::turn_index which is unique per session.)

    file_size = os.path.getsize(transcript_path)
    if start_offset >= file_size:
        return [], file_size

    transcript_file = os.path.basename(transcript_path)

    with open(transcript_path, "r", encoding="utf-8") as f:
        f.seek(start_offset)
        current_offset = start_offset

        for line in f:
            line_bytes = len(line.encode("utf-8"))
            current_offset += line_bytes

            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = entry.get("type", "")

            # Track git branch from any entry that has it
            entry_branch = entry.get("gitBranch", "")
            if entry_branch:
                current_git_branch = entry_branch

            # Track timestamp from any entry that has it
            entry_ts = entry.get("timestamp", "")
            if entry_ts:
                current_timestamp = entry_ts

            # Skip non-conversation entries
            if entry_type in ("file-history-snapshot", "progress", "system", "queue-operation"):
                continue

            # Handle summary entries (compaction summaries)
            if entry_type == "summary":
                # Flush any pending turn first
                if current_user_text is not None:
                    turn = _build_turn(
                        current_user_text, current_assistant_texts,
                        session_id, transcript_file, turn_index,
                        current_timestamp, current_git_branch,
                        max_turn_chars, "turn",
                    )
                    if turn:
                        turns.append(turn)
                        turn_index += 1
                    current_user_text = None
                    current_assistant_texts = []

                summary_text = entry.get("summary", "")
                if summary_text:
                    turns.append({
                        "text": f"Session Summary: {summary_text}",
                        "doc_id": f"{session_id}::summary::{turn_index}",
                        "session_id": session_id,
                        "transcript_file": transcript_file,
                        "turn_index": turn_index,
                        "timestamp": current_timestamp,
                        "git_branch": current_git_branch,
                        "chunk_type": "summary",
                    })
                    turn_index += 1
                continue

            # Handle user messages
            if entry_type == "user":
                message = entry.get("message", {})
                content = message.get("content", "")

                # Skip tool_result messages (content is a list)
                if isinstance(content, list):
                    continue

                # Skip isMeta messages
                if entry.get("isMeta"):
                    continue

                # Skip empty content
                if not isinstance(content, str) or not content.strip():
                    continue

                # Flush previous turn if we have one
                if current_user_text is not None:
                    turn = _build_turn(
                        current_user_text, current_assistant_texts,
                        session_id, transcript_file, turn_index,
                        current_timestamp, current_git_branch,
                        max_turn_chars, "turn",
                    )
                    if turn:
                        turns.append(turn)
                        turn_index += 1

                # Start new turn
                current_user_text = content.strip()
                current_assistant_texts = []
                continue

            # Handle assistant messages
            if entry_type == "assistant":
                message = entry.get("message", {})
                content = message.get("content", [])

                if not isinstance(content, list):
                    continue

                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text = block.get("text", "").strip()
                        if text:
                            current_assistant_texts.append(text)

        # Flush final pending turn
        if current_user_text is not None:
            turn = _build_turn(
                current_user_text, current_assistant_texts,
                session_id, transcript_file, turn_index,
                current_timestamp, current_git_branch,
                max_turn_chars, "turn",
            )
            if turn:
                turns.append(turn)

    return turns, current_offset


def _build_turn(
    user_text: str,
    assistant_texts: List[str],
    session_id: str,
    transcript_file: str,
    turn_index: int,
    timestamp: str,
    git_branch: str,
    max_chars: int,
    chunk_type: str,
) -> Optional[Dict]:
    """Build a turn dict from user + assistant text."""
    parts = [f"User: {user_text}"]
    if assistant_texts:
        combined_assistant = "\n\n".join(assistant_texts)
        parts.append(f"Assistant: {combined_assistant}")

    text = "\n\n".join(parts)

    # Truncate if too long
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[truncated]"

    # Skip very short turns (likely just whitespace or newlines)
    if len(text.strip()) < 20:
        return None

    return {
        "text": text,
        "doc_id": f"{session_id}::{turn_index}",
        "session_id": session_id,
        "transcript_file": transcript_file,
        "turn_index": turn_index,
        "timestamp": timestamp,
        "git_branch": git_branch,
        "chunk_type": chunk_type,
    }


# --- Index state management ---

def load_index_state(project_root: str) -> Dict:
    """Load index state tracking file."""
    state_path = Path(project_root) / ".session-rag" / "index_state.json"
    if state_path.exists():
        try:
            with open(state_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_index_state(project_root: str, state: Dict):
    """Save index state tracking file."""
    state_dir = Path(project_root) / ".session-rag"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "index_state.json"
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def get_transcript_offset(state: Dict, transcript_path: str) -> int:
    """Get the last indexed byte offset for a transcript file."""
    return state.get("transcripts", {}).get(transcript_path, {}).get("last_byte_offset", 0)


def set_transcript_offset(state: Dict, transcript_path: str, offset: int):
    """Update the byte offset for a transcript file."""
    if "transcripts" not in state:
        state["transcripts"] = {}
    if transcript_path not in state["transcripts"]:
        state["transcripts"][transcript_path] = {}
    state["transcripts"][transcript_path]["last_byte_offset"] = offset
