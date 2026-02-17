"""
MCP tool definitions and project context for session-rag.
"""

import contextvars
from pathlib import Path
from mcp.server import Server
from mcp import types

import rag_engine


# --- Project context ---

_current_project_root: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_project_root", default=None
)


class ProjectNotConfiguredError(Exception):
    pass


_PROJECT_ERROR_MSG = (
    "No project configured. Add to your .mcp.json:\n"
    '  "headers": {"X-Project-Root": "/path/to/your/project"}'
)


def set_current_project_root(root: str | None):
    _current_project_root.set(root)


def get_current_project_root() -> str:
    root = _current_project_root.get()
    if root is None:
        raise ProjectNotConfiguredError(_PROJECT_ERROR_MSG)
    return root


def get_db_path() -> str:
    root = get_current_project_root()
    return str(Path(root) / ".session-rag" / "milvus.db")


# --- Formatting helpers ---

def format_results(results: list[dict]) -> str:
    """Format search results as markdown."""
    if not results:
        return "No results found."

    output = []
    for i, r in enumerate(results, 1):
        # Header with metadata
        session_short = r.get("session_id", "")[:8]
        branch = r.get("git_branch", "")
        ts = r.get("timestamp", "")[:19]  # trim to readable
        chunk_type = r.get("chunk_type", "turn")
        similarity = 1 - r.get("distance", 0)

        header_parts = [f"**Result {i}**"]
        if ts:
            header_parts.append(f"({ts})")
        if branch:
            header_parts.append(f"[{branch}]")
        if session_short:
            header_parts.append(f"session:{session_short}")

        output.append(" ".join(header_parts))

        meta = f"*Type: {chunk_type} | Relevance: {similarity:.2f}*"
        output.append(meta)
        output.append("")
        output.append(r.get("content", ""))
        output.append("")
        output.append("---")
        output.append("")

    return "\n".join(output)


def format_stats(stats: dict, db_path: str) -> str:
    """Format index statistics."""
    lines = [
        f"**Total Turns:** {stats['total_turns']}",
        f"**Sessions:** {stats['sessions']}",
    ]

    if stats.get("branches"):
        lines.append(f"**Branches:** {', '.join(stats['branches'])}")

    if stats.get("by_type"):
        lines.append("\n### By Type")
        for t, count in sorted(stats["by_type"].items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- {t}: {count}")

    lines.append(f"\n**Index Location:** {db_path}")
    return "\n".join(lines)


# --- Tool registration ---

def register_tools(server: Server):
    """Register session-rag MCP tools."""

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="search_session",
                description=(
                    "Search conversation history for past discussions, decisions, "
                    "code snippets, and error messages. Results are boosted by recency "
                    "so recent conversations rank higher."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query (e.g., 'approval workflow decision', 'error in deploy script')",
                        },
                        "n": {
                            "type": "integer",
                            "description": "Number of results to return (default: 5)",
                            "default": 5,
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Claude session ID to filter to. Usually auto-resolved; only pass if auto-resolution fails.",
                        },
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="search_all_sessions",
                description=(
                    "Search across ALL past conversation sessions. Pure semantic search "
                    "without recency bias. Optionally filter by git branch."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query",
                        },
                        "n": {
                            "type": "integer",
                            "description": "Number of results to return (default: 10)",
                            "default": 10,
                        },
                        "git_branch": {
                            "type": "string",
                            "description": "Filter by git branch name (e.g., 'develop', 'feature/my-feature')",
                        },
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="get_session_stats",
                description="Get session index statistics (turn count, session count, branches)",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
            types.Tool(
                name="cleanup_sessions",
                description=(
                    "Delete old session data from the index. "
                    "Can delete by age (days), specific session ID, or git branch."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "max_age_days": {
                            "type": "integer",
                            "description": "Delete turns older than this many days",
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Delete all turns for this session ID",
                        },
                        "git_branch": {
                            "type": "string",
                            "description": "Delete all turns for this git branch",
                        },
                    },
                    "required": [],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        try:
            db = get_db_path()
        except ProjectNotConfiguredError as e:
            return [types.TextContent(type="text", text=str(e))]

        try:
            if name == "search_session":
                session_id = arguments.get("session_id")
                results = rag_engine.search(
                    arguments["query"],
                    arguments.get("n", 5),
                    session_id=session_id,
                    recency_boost=True,
                    db_path=db,
                )
                return [types.TextContent(type="text", text=format_results(results))]

            elif name == "search_all_sessions":
                results = rag_engine.search(
                    arguments["query"],
                    arguments.get("n", 10),
                    git_branch=arguments.get("git_branch"),
                    recency_boost=False,
                    db_path=db,
                )
                return [types.TextContent(type="text", text=format_results(results))]

            elif name == "get_session_stats":
                stats = rag_engine.get_stats(db_path=db)
                return [types.TextContent(type="text", text=format_stats(stats, db))]

            elif name == "cleanup_sessions":
                max_age = arguments.get("max_age_days")
                sid = arguments.get("session_id")
                branch = arguments.get("git_branch")

                if not any([max_age, sid, branch]):
                    return [types.TextContent(
                        type="text",
                        text="Specify at least one of: max_age_days, session_id, git_branch",
                    )]

                parts = []
                if max_age:
                    count = rag_engine.delete_older_than(max_age, db_path=db)
                    parts.append(f"Deleted {count} turns older than {max_age} days")
                if sid:
                    count = rag_engine.delete_by_session(sid, db_path=db)
                    parts.append(f"Deleted {count} turns for session {sid[:12]}")
                if branch:
                    count = rag_engine.delete_by_branch(branch, db_path=db)
                    parts.append(f"Deleted {count} turns for branch '{branch}'")

                stats = rag_engine.get_stats(db_path=db)
                parts.append(f"\nRemaining: {stats['total_turns']} turns across {stats['sessions']} sessions")
                return [types.TextContent(type="text", text="\n".join(parts))]

            else:
                raise ValueError(f"Unknown tool: {name}")

        except Exception as e:
            return [types.TextContent(type="text", text=f"Error executing {name}: {str(e)}")]
