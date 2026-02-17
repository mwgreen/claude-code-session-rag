#!/usr/bin/env python3
"""
Persistent HTTP server for session-rag MCP system.

Runs as a long-lived process serving MCP via StreamableHTTP.
Projects are identified by the X-Project-Root header in each request.
The DB for each project lives at {project_root}/.session-rag/milvus.db.

Start: ./session-rag-server.sh
Health: curl http://127.0.0.1:7102/health
"""

import contextlib
import json
import os
import signal
import sys
import time
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

import rag_engine
import transcript_parser
from tools import register_tools, set_current_project_root


# --- Configuration ---

HOST = os.getenv("SESSION_RAG_HOST", "127.0.0.1")
PORT = int(os.getenv("SESSION_RAG_PORT", "7102"))
AUTO_EXPIRE_DAYS = int(os.getenv("SESSION_RAG_EXPIRE_DAYS", "30"))
_EXPIRE_CHECK_INTERVAL = 86400  # Check once per day

_SERVER_DIR = Path.home() / ".session-rag"
PID_FILE = _SERVER_DIR / "server.pid"
LOG_FILE = _SERVER_DIR / "server.log"


# --- Project middleware ---

class ProjectMiddleware:
    """ASGI middleware that extracts X-Project-Root header and sets ContextVar."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            project_root = headers.get(b"x-project-root", b"").decode("utf-8").strip()
            set_current_project_root(project_root if project_root else None)

        await self.app(scope, receive, send)


# --- Health endpoint ---

async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "server": "session-rag", "port": PORT})


# --- Index endpoint (called by hooks) ---

async def index_endpoint(request: Request) -> JSONResponse:
    """Index new turns from a transcript file.

    Expected JSON body:
        {
            "transcript_path": "/path/to/session.jsonl",
            "session_id": "uuid",
            "cwd": "/path/to/project"   (optional, fallback for project root)
        }

    Project root comes from X-Project-Root header (preferred) or cwd in body.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    transcript_path = body.get("transcript_path", "")
    session_id = body.get("session_id", "")

    if not transcript_path or not session_id:
        return JSONResponse(
            {"error": "transcript_path and session_id are required"},
            status_code=400,
        )

    if not os.path.exists(transcript_path):
        return JSONResponse(
            {"error": f"Transcript not found: {transcript_path}"},
            status_code=404,
        )

    # Determine project root from header or body
    headers = dict(request.scope.get("headers", []))
    project_root = headers.get(b"x-project-root", b"").decode("utf-8").strip()
    if not project_root:
        project_root = body.get("cwd", "")
    if not project_root:
        return JSONResponse(
            {"error": "Project root required (X-Project-Root header or cwd in body)"},
            status_code=400,
        )

    db_path = str(Path(project_root) / ".session-rag" / "milvus.db")

    # Load incremental state
    state = transcript_parser.load_index_state(project_root)
    offset = transcript_parser.get_transcript_offset(state, transcript_path)

    # Parse new turns
    turns, new_offset = transcript_parser.parse_transcript(
        transcript_path, session_id, start_offset=offset
    )

    if not turns:
        # Update offset even if no turns (e.g., only tool_result messages)
        transcript_parser.set_transcript_offset(state, transcript_path, new_offset)
        transcript_parser.save_index_state(project_root, state)
        return JSONResponse({"indexed": 0, "message": "No new turns to index"})

    # Index turns
    count = await rag_engine.add_turns_async(turns, db_path=db_path)

    # Save state
    transcript_parser.set_transcript_offset(state, transcript_path, new_offset)
    transcript_parser.save_index_state(project_root, state)

    print(f"[index] Indexed {count} turns from {os.path.basename(transcript_path)} "
          f"(session {session_id[:8]})", file=sys.stderr)

    # Auto-expiry: prune old turns once per day
    expired = 0
    if AUTO_EXPIRE_DAYS > 0:
        last_expire = state.get("last_expire_check", 0)
        now = time.time()
        if now - last_expire > _EXPIRE_CHECK_INTERVAL:
            expired = rag_engine.delete_older_than(AUTO_EXPIRE_DAYS, db_path=db_path)
            state["last_expire_check"] = now
            transcript_parser.save_index_state(project_root, state)
            if expired > 0:
                print(f"[expire] Pruned {expired} turns older than {AUTO_EXPIRE_DAYS} days",
                      file=sys.stderr)

    return JSONResponse({"indexed": count, "expired": expired, "session_id": session_id})


# --- Lifespan ---

@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    """Server lifecycle: PID file, model preload, server mode init."""
    _SERVER_DIR.mkdir(parents=True, exist_ok=True)

    PID_FILE.write_text(str(os.getpid()))
    print(f"[HTTP] PID {os.getpid()} written to {PID_FILE}", file=sys.stderr)

    # Pre-load embedding model
    try:
        print("[HTTP] Pre-loading Nomic model...", file=sys.stderr)
        rag_engine.get_model()
        print("[HTTP] Nomic model loaded.", file=sys.stderr)
    except Exception as e:
        print(f"[HTTP] Warning: Could not pre-load model: {e}", file=sys.stderr)

    rag_engine.init_server_mode()

    async with session_manager.run():
        print(f"[HTTP] Server ready on http://{HOST}:{PORT}", file=sys.stderr)
        try:
            yield
        finally:
            pass

    rag_engine.close_server_mode()
    if PID_FILE.exists():
        PID_FILE.unlink()
    print("[HTTP] Server stopped.", file=sys.stderr)


# --- MCP server setup ---

mcp_server = Server("session-rag")
register_tools(mcp_server)

session_manager = StreamableHTTPSessionManager(
    app=mcp_server,
    stateless=True,
    json_response=True,
)


# --- Starlette app ---

app = Starlette(
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/index", index_endpoint, methods=["POST"]),
        Mount("/mcp", app=ProjectMiddleware(session_manager.handle_request)),
    ],
    lifespan=lifespan,
)


# --- Signal handling ---

def _handle_signal(signum, frame):
    print(f"[HTTP] Received signal {signum}, shutting down...", file=sys.stderr)
    raise SystemExit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _handle_signal)

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="warning",
    )
