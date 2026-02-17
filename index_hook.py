#!/usr/bin/env python3
"""
Hook entry point for Claude Code Stop/PreCompact hooks.

Reads hook input JSON from stdin, derives project root, and POSTs
to the session-rag server to index new transcript turns.

Exit codes:
  0 = success (or server not running — silently skip)
  1 = error (logged but doesn't block Claude)
  Never exits 2 — that would block Claude from stopping.
"""

import json
import os
import subprocess
import sys

import httpx

SERVER_URL = os.getenv("SESSION_RAG_URL", "http://127.0.0.1:7102")


def get_project_root(cwd: str) -> str:
    """Derive project root from cwd using git."""
    try:
        result = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return cwd


def main():
    # Read hook input from stdin
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            sys.exit(0)
        hook_input = json.loads(raw)
    except (json.JSONDecodeError, IOError):
        sys.exit(0)

    transcript_path = hook_input.get("transcript_path", "")
    session_id = hook_input.get("session_id", "")
    cwd = hook_input.get("cwd", "")

    if not transcript_path or not session_id:
        sys.exit(0)

    # Skip if this is a stop hook that was triggered by a previous stop hook
    if hook_input.get("stop_hook_active"):
        sys.exit(0)

    project_root = get_project_root(cwd) if cwd else ""
    if not project_root:
        sys.exit(0)

    # Fire-and-forget POST to server.
    # We don't wait for the response — indexing happens in the background.
    # If it fails, byte offsets don't advance and the next call retries.
    try:
        with httpx.Client() as client:
            client.post(
                f"{SERVER_URL}/index",
                json={
                    "transcript_path": transcript_path,
                    "session_id": session_id,
                    "cwd": cwd,
                },
                headers={"X-Project-Root": project_root},
                timeout=1.0,  # Just enough to send the request
            )
    except (httpx.ConnectError, httpx.ReadTimeout):
        # Server not running or still processing — either way, fine.
        pass
    except Exception:
        pass


if __name__ == "__main__":
    main()
