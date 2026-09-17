"""
MCP tool definitions for session-rag.

The project a request belongs to arrives in the X-Project-Root header (set by
the headersHelper script when Claude Code connects) and is kept in a
ContextVar for the duration of the request. If that project has nothing
indexed, searches fall back to all projects and say so.
"""

from __future__ import annotations

import contextvars
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mcp import types
from mcp.server import Server

import rag_engine

logger = logging.getLogger("session-rag.tools")

ALL_PROJECTS = "*"
_SCOPE_CACHE_TTL = 120.0

_current_project_root: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_project_root", default=None
)
_scope_cache: Dict[str, Tuple[float, int]] = {}


def set_current_project_root(root: Optional[str]) -> None:
    root = (root or "").strip()
    _current_project_root.set(root.rstrip("/") or None if root else None)


def get_current_project_root() -> Optional[str]:
    return _current_project_root.get()


def get_db_path() -> str:
    return rag_engine.DEFAULT_DB_PATH


def invalidate_scope_cache() -> None:
    _scope_cache.clear()


async def resolve_scope(requested: Optional[str]) -> Tuple[Optional[str], str]:
    """Decide which project_root filter to use.

    requested: None -> current project (from header); "*" -> all projects;
    any other string -> that project. Returns (filter_root, human note).
    """
    if requested == ALL_PROJECTS:
        return None, "all projects"
    root = (requested or get_current_project_root() or "").rstrip("/")
    if not root:
        return None, "all projects (no project root in request)"
    now = time.time()
    cached = _scope_cache.get(root)
    if cached is None or now - cached[0] > _SCOPE_CACHE_TTL:
        try:
            n = await rag_engine.run(rag_engine.count, rag_engine.project_filter(root, True),
                                     db_path=get_db_path())
        except Exception as exc:
            logger.warning("Scope check failed for %s: %s", root, exc)
            n = 1  # assume it has data rather than widening silently
        _scope_cache[root] = (now, n)
        cached = _scope_cache[root]
    if cached[1] == 0:
        return None, f"all projects ({Path(root).name} has no indexed turns yet)"
    return root, f"project {Path(root).name} ({root})"


# --- Formatting -------------------------------------------------------------

def _fmt_score(r: Dict) -> str:
    parts = []
    if r.get("score") is not None:
        parts.append(f"score {r['score']:.2f}")
    if r.get("similarity") is not None:
        parts.append(f"sim {r['similarity']:.2f}")
    return " · ".join(parts)


def format_results(results: List[Dict], scope_note: str = "") -> str:
    lines: List[str] = []
    if scope_note:
        lines += [f"Scope: {scope_note}", ""]
    if not results:
        lines.append("No results found.")
        return "\n".join(lines)
    for i, r in enumerate(results, 1):
        header = [f"**Result {i}**"]
        ts = (r.get("timestamp") or "")[:19]
        if ts:
            header.append(f"({ts})")
        if r.get("git_branch"):
            header.append(f"[{r['git_branch']}]")
        if r.get("project_root"):
            header.append(f"project:{Path(r['project_root']).name}")
        if r.get("session_id"):
            header.append(f"session:{r['session_id']}")
        lines.append(" ".join(header))
        meta = [f"Turn: {r.get('turn_index', 0)}", f"Type: {r.get('chunk_type', 'turn')}"]
        if r.get("transcript_file") and r.get("chunk_type") == "subagent":
            meta.append(f"File: {r['transcript_file']}")
        score = _fmt_score(r)
        if score:
            meta.append(score)
        lines += [f"*{' | '.join(meta)}*", "", r.get("content", ""), "", "---", ""]
    return "\n".join(lines)


def format_turns(results: List[Dict]) -> str:
    if not results:
        return "No turns found."
    lines: List[str] = []
    for r in results:
        header = [f"**Turn {r.get('turn_index', 0)}**"]
        ts = (r.get("timestamp") or "")[:19]
        if ts:
            header.append(f"({ts})")
        if r.get("git_branch"):
            header.append(f"[{r['git_branch']}]")
        lines += [" ".join(header), f"*Type: {r.get('chunk_type', 'turn')}*", "",
                  r.get("content", ""), "", "---", ""]
    return "\n".join(lines)


def format_stats(stats: Dict, scope_note: str, db_path: str) -> str:
    lines = [f"Scope: {scope_note}", "",
             f"**Total Chunks:** {stats['total_turns']}",
             f"**Sessions:** {stats['sessions']}",
             f"**Model:** {rag_engine.get_model_name()} ({rag_engine.model_spec().dim}d)"]
    if stats.get("projects") and len(stats["projects"]) > 1:
        lines.append(f"**Projects:** {len(stats['projects'])}")
    if stats.get("branches"):
        shown = stats["branches"][:25]
        more = f" (+{len(stats['branches']) - 25} more)" if len(stats["branches"]) > 25 else ""
        lines.append(f"**Branches:** {', '.join(shown)}{more}")
    if stats.get("by_type"):
        lines.append("\n### By Type")
        for t, c in sorted(stats["by_type"].items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- {t}: {c}")
    lines.append(f"\n**Index Location:** {db_path}")
    return "\n".join(lines)


# --- Tool registration -------------------------------------------------------

_PROJECT_ROOT_PROP = {
    "type": "string",
    "description": ("Project to search: omit for the current project, '*' for all projects, "
                    "or an absolute project path."),
}


def register_tools(server: Server) -> None:
    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="search_session",
                description=(
                    "Search past Claude Code conversations (this project by default) for "
                    "decisions, code snippets, error messages and reasoning. Hybrid semantic + "
                    "keyword search with a recency boost. Pass session_id (the CLAUDE_SESSION_ID "
                    "environment variable) to limit results to the current session."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string",
                                  "description": "Natural language or keyword query (e.g. 'approval workflow decision', 'pymilvus unix socket error')"},
                        "n": {"type": "integer", "description": "Number of results (default 5, max 50)", "default": 5},
                        "session_id": {"type": "string",
                                       "description": "Limit to one session. Use the CLAUDE_SESSION_ID env var for the current session."},
                        "project_root": _PROJECT_ROOT_PROP,
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="search_all_sessions",
                description=(
                    "Search past conversations without recency bias. Defaults to the current "
                    "project; pass project_root='*' for every project. Optional git branch filter."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language or keyword query"},
                        "n": {"type": "integer", "description": "Number of results (default 10, max 50)", "default": 10},
                        "git_branch": {"type": "string", "description": "Only results from this git branch"},
                        "project_root": _PROJECT_ROOT_PROP,
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="get_turns",
                description=(
                    "Show the conversation turns around a search hit. Pass the session_id and "
                    "turn index from a result (and its File value for subagent results)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session ID from a search result"},
                        "turn_index": {"type": "integer", "description": "Turn index from a search result"},
                        "context": {"type": "integer", "description": "Turns before and after (default 2, max 20)", "default": 2},
                        "transcript_file": {"type": "string", "description": "Transcript file name shown on subagent results"},
                    },
                    "required": ["session_id", "turn_index"],
                },
            ),
            types.Tool(
                name="get_session_stats",
                description="Index statistics (chunks, sessions, branches) for the current project or, with project_root='*', everything.",
                inputSchema={
                    "type": "object",
                    "properties": {"project_root": _PROJECT_ROOT_PROP},
                    "required": [],
                },
            ),
            types.Tool(
                name="cleanup_sessions",
                description="Delete indexed data by age (days), session ID or git branch.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "max_age_days": {"type": "integer", "description": "Delete chunks older than this many days"},
                        "session_id": {"type": "string", "description": "Delete all chunks of this session"},
                        "git_branch": {"type": "string", "description": "Delete all chunks from this git branch"},
                    },
                    "required": [],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        arguments = arguments or {}
        db = get_db_path()
        try:
            if name in ("search_session", "search_all_sessions"):
                scope_root, note = await resolve_scope(arguments.get("project_root"))
                default_n = 5 if name == "search_session" else 10
                n = int(arguments.get("n") or default_n)
                session_id = arguments.get("session_id") if name == "search_session" else None
                if session_id:
                    note = f"session {session_id[:8]}… within {note}"
                results = await rag_engine.run(
                    rag_engine.search, arguments["query"], n,
                    session_id=session_id,
                    git_branch=arguments.get("git_branch") if name == "search_all_sessions" else None,
                    project_root=scope_root,
                    recency_boost=(name == "search_session"),
                    db_path=db,
                )
                return [types.TextContent(type="text", text=format_results(results, note))]

            if name == "get_turns":
                results = await rag_engine.run(
                    rag_engine.get_turns, arguments["session_id"], int(arguments["turn_index"]),
                    context=int(arguments.get("context") or 2), db_path=db,
                    transcript_file=arguments.get("transcript_file"),
                )
                return [types.TextContent(type="text", text=format_turns(results))]

            if name == "get_session_stats":
                scope_root, note = await resolve_scope(arguments.get("project_root"))
                stats = await rag_engine.run(rag_engine.get_stats, project_root=scope_root, db_path=db)
                return [types.TextContent(type="text", text=format_stats(stats, note, db))]

            if name == "cleanup_sessions":
                max_age = arguments.get("max_age_days")
                sid = arguments.get("session_id")
                branch = arguments.get("git_branch")
                if not any([max_age, sid, branch]):
                    return [types.TextContent(type="text",
                                              text="Specify at least one of: max_age_days, session_id, git_branch")]
                parts = []
                if max_age:
                    c = await rag_engine.run(rag_engine.delete_older_than, int(max_age), db_path=db)
                    parts.append(f"Deleted {c} chunks older than {max_age} days")
                if sid:
                    c = await rag_engine.run(rag_engine.delete_by_session, sid, db_path=db)
                    parts.append(f"Deleted {c} chunks for session {sid[:12]}")
                if branch:
                    c = await rag_engine.run(rag_engine.delete_by_branch, branch, db_path=db)
                    parts.append(f"Deleted {c} chunks for branch '{branch}'")
                invalidate_scope_cache()
                stats = await rag_engine.run(rag_engine.get_stats, db_path=db)
                parts.append(f"\nRemaining: {stats['total_turns']} chunks across {stats['sessions']} sessions")
                return [types.TextContent(type="text", text="\n".join(parts))]

            raise ValueError(f"Unknown tool: {name}")
        except Exception as exc:
            logger.exception("Tool %s failed", name)
            return [types.TextContent(type="text", text=f"Error executing {name}: {exc}")]
