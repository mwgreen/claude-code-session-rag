#!/usr/bin/env python3
"""
Persistent HTTP server for session-rag.

Serves MCP over StreamableHTTP (stateless, JSON responses) plus a few plain
endpoints used by the hooks and the launcher script:

  GET  /health   200 when model + index are ready, 503 otherwise
  GET  /status   indexer / watcher / engine details
  POST /index    hook: index a transcript now  {"transcript_path", "session_id", "cwd"}
  POST /watch    hook: register a project and backfill its transcripts

Projects are identified by the X-Project-Root header. All projects share one
index at ~/.session-rag/.

Start with ./session-rag-server.sh (it also runs a watchdog).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys
import traceback
from pathlib import Path

import uvicorn
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

import file_watcher
import indexer as indexer_mod
import rag_engine
from index_state import STATE_DIR
from tools import invalidate_scope_cache, register_tools, set_current_project_root

logging.basicConfig(
    level=os.getenv("SESSION_RAG_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr,
)
for noisy in ("mcp", "httpx", "uvicorn.access"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("session-rag.server")

HOST = os.getenv("SESSION_RAG_HOST", "127.0.0.1")
PORT = int(os.getenv("SESSION_RAG_PORT", "7102"))
AUTO_EXPIRE_DAYS = int(os.getenv("SESSION_RAG_EXPIRE_DAYS", "365"))
EXPIRE_CHECK_SECONDS = 3600
DB_PATH = str(STATE_DIR / "milvus.db")
PID_FILE = STATE_DIR / "server.pid"

_ready = {"model": False, "index": False, "error": None}
_background: list = []


def _header_project_root(scope_or_request) -> str:
    headers = scope_or_request.get("headers", []) if isinstance(scope_or_request, dict) \
        else scope_or_request.scope.get("headers", [])
    for key, value in headers:
        if key == b"x-project-root":
            return value.decode("utf-8", "replace").strip().rstrip("/")
    return ""


class ProjectMiddleware:
    """Puts the X-Project-Root header into the tools' ContextVar."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            set_current_project_root(_header_project_root(scope) or None)
        try:
            await self.app(scope, receive, send)
        except Exception as exc:
            logger.error("ASGI handler error: %s\n%s", exc, traceback.format_exc())
            if scope["type"] == "http":
                body = json.dumps({"error": "internal_server_error", "detail": str(exc)}).encode()
                try:
                    await send({"type": "http.response.start", "status": 500, "headers": [
                        [b"content-type", b"application/json"],
                        [b"content-length", str(len(body)).encode()],
                    ]})
                    await send({"type": "http.response.body", "body": body})
                except Exception:
                    pass  # response already started


# --- Plain endpoints ----------------------------------------------------------

def _status_payload() -> dict:
    idx = indexer_mod.get_indexer()
    return {
        "status": "ok" if _ready["model"] and _ready["index"] else "starting",
        "server": "session-rag",
        "port": PORT,
        "model_name": rag_engine.get_model_name(),
        "model_id": rag_engine.model_spec().model_id,
        "embed_dim": rag_engine.model_spec().dim,
        "model_loaded": _ready["model"],
        "milvus": _ready["index"],
        "error": _ready["error"],
        "indexer": idx.status() if idx else None,
        "watcher": file_watcher.get_watcher_status(),
    }


async def health(request: Request) -> JSONResponse:
    payload = _status_payload()
    code = 200 if payload["status"] == "ok" else 503
    return JSONResponse(payload, status_code=code)


async def status(request: Request) -> JSONResponse:
    return JSONResponse(_status_payload())


async def index_endpoint(request: Request) -> JSONResponse:
    """Hook entry point: queue a transcript for immediate indexing."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    transcript_path = str(body.get("transcript_path") or "")
    session_id = str(body.get("session_id") or "")
    if not transcript_path or not session_id:
        return JSONResponse({"error": "transcript_path and session_id are required"}, status_code=400)
    if not os.path.exists(transcript_path):
        return JSONResponse({"error": f"Transcript not found: {transcript_path}"}, status_code=404)
    project_root = _header_project_root(request) or str(body.get("cwd") or "").rstrip("/")
    idx = indexer_mod.get_indexer()
    if idx is None:
        return JSONResponse({"error": "indexer not running"}, status_code=503)
    if project_root:
        idx.register_project(project_root)
        invalidate_scope_cache()
    idx.enqueue(transcript_path, project_root=project_root, session_id=session_id, immediate=True)
    return JSONResponse({"queued": True, "session_id": session_id, "project_root": project_root})


async def watch_endpoint(request: Request) -> JSONResponse:
    """Hook entry point: register the project and backfill its transcripts."""
    project_root = _header_project_root(request)
    if not project_root:
        try:
            body = await request.json()
            project_root = str(body.get("project_root") or body.get("cwd") or "").rstrip("/")
        except Exception:
            pass
    if not project_root:
        return JSONResponse({"error": "Project root required (X-Project-Root header or project_root in body)"},
                            status_code=400)
    idx = indexer_mod.get_indexer()
    if idx is None:
        return JSONResponse({"error": "indexer not running"}, status_code=503)
    idx.register_project(project_root)
    invalidate_scope_cache()
    slug = idx.slug_map.slug_for(project_root)
    queued = await idx.schedule_backfill(slug_filter=slug)
    return JSONResponse({"watching": project_root, "backfill_queued": queued})


# --- Background tasks -----------------------------------------------------------

async def _expiry_loop():
    while True:
        try:
            idx = indexer_mod.get_indexer()
            if idx is not None:
                deleted = await idx.expire_old(AUTO_EXPIRE_DAYS)
                if deleted:
                    invalidate_scope_cache()
        except Exception as exc:
            logger.warning("Expiry check failed: %s", exc)
        await asyncio.sleep(EXPIRE_CHECK_SECONDS)


async def _initial_backfill():
    idx = indexer_mod.get_indexer()
    if idx is not None:
        await idx.schedule_backfill()


# --- Lifespan ---------------------------------------------------------------------

@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Starting (PID %d)", os.getpid())

    try:
        rag_engine.init_server_mode(db_path=DB_PATH)   # fails fast on a model/index mismatch
        _ready["index"] = True
    except Exception as exc:
        _ready["error"] = str(exc)
        logger.error("Index initialisation failed: %s", exc)
        raise
    # Only the process that owns the database claims the PID file.
    PID_FILE.write_text(str(os.getpid()))

    try:
        logger.info("Loading embedding model %s ...", rag_engine.get_model_name())
        await rag_engine.run(rag_engine.load_model)
        _ready["model"] = True
    except Exception as exc:
        _ready["error"] = str(exc)
        logger.error("Could not load embedding model: %s", exc)
        raise

    try:
        n = await rag_engine.run(rag_engine.backfill_fts, DB_PATH)
        if n:
            logger.info("FTS backfill: %d records", n)
    except Exception as exc:
        logger.warning("FTS backfill failed: %s", exc)

    idx = await indexer_mod.start_indexer(DB_PATH, file_watcher.DEBOUNCE_SECONDS)
    file_watcher.start_watcher(idx)
    _background.append(asyncio.create_task(_initial_backfill(), name="initial-backfill"))
    if AUTO_EXPIRE_DAYS > 0:
        _background.append(asyncio.create_task(_expiry_loop(), name="expiry"))

    async with session_manager.run():
        logger.info("Server ready on http://%s:%d (model=%s)", HOST, PORT, rag_engine.get_model_name())
        try:
            yield
        finally:
            logger.info("Shutting down ...")

    for task in _background:
        task.cancel()
    for task in _background:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
    file_watcher.stop_watcher()
    await indexer_mod.stop_indexer()
    rag_engine.close_server_mode()
    if PID_FILE.exists() and PID_FILE.read_text().strip() == str(os.getpid()):
        PID_FILE.unlink()
    logger.info("Server stopped.")


# --- MCP wiring ---------------------------------------------------------------------

mcp_server = Server(
    "session-rag",
    instructions=(
        "Session-RAG provides hybrid semantic + keyword search over past Claude Code "
        "conversations. Results default to the current project. To restrict "
        "search_session to the current session, pass session_id from the "
        "CLAUDE_SESSION_ID environment variable. Use search_all_sessions with "
        "project_root='*' to search every project. Use get_turns to read the "
        "conversation around a hit."
    ),
)
register_tools(mcp_server)
session_manager = StreamableHTTPSessionManager(app=mcp_server, stateless=True, json_response=True)

app = Starlette(
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/status", status, methods=["GET"]),
        Route("/index", index_endpoint, methods=["POST"]),
        Route("/watch", watch_endpoint, methods=["POST"]),
        Mount("/mcp", app=ProjectMiddleware(session_manager.handle_request)),
    ],
    lifespan=lifespan,
)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
