# session-rag

Semantic + keyword search over your Claude Code session transcripts. Recovers what
context compaction throws away: decisions, code snippets, error messages, and the
reasoning behind them, across every project on the machine.

## How it works

- **Embedding model**: EmbeddingGemma-300M by default, run locally with MLX on Apple
  Silicon. Qwen3-Embedding-0.6B is available as an alternative (see *Embedding models*).
- **Vector store**: Milvus Lite, one global database at `~/.session-rag/milvus.db`.
- **Keyword index**: SQLite FTS5 mirror of every chunk (BM25). Search is hybrid: both
  engines run, results are merged with Reciprocal Rank Fusion.
- **Indexing**: one sequential pipeline fed by a file watcher on `~/.claude/projects/`,
  by the Stop/PreCompact hooks, and by a backfill on every session start.
- **Server**: HTTP MCP server on `127.0.0.1:7102`. Started at login by a launchd agent
  (recommended) or on demand by a SessionStart hook; a health watchdog restarts it if it
  hangs.
- **Memory**: about 600 MB for the model plus the index.

## Quick start

```bash
./setup.sh          # venv, dependencies, model download, hooks, global MCP entry
# then restart Claude Code (or: ./session-rag-server.sh start)
```

`setup.sh` is idempotent. Run it again after pulling changes to refresh hooks, the
header helper and (if installed) the launchd agent.

```bash
./session-rag-server.sh install-launchd   # recommended: start at login, restart on crash
```

## MCP tools

| Tool | What it does |
|------|--------------|
| `search_session` | Hybrid search with a recency boost. Defaults to the current project; pass `session_id` (the `CLAUDE_SESSION_ID` env var) to stay inside the current session. |
| `search_all_sessions` | Hybrid search without recency bias. Current project by default, `project_root="*"` for everything, optional `git_branch` filter. |
| `get_turns` | The conversation around a hit (`session_id` + `turn_index` from a result; add `transcript_file` for subagent hits). |
| `get_session_stats` | Chunk/session/branch counts for the current project, or all with `project_root="*"`. |
| `cleanup_sessions` | Delete by age, session id or branch. |

Every search result shows `score` (rank-fusion score; 1.0 = top of both engines, up to
1.3 with the recency boost) and `sim` (cosine similarity, 1.0 = identical). Every response starts with a `Scope:` line
so you can see which project filter was applied. If the current project has nothing
indexed yet, the search widens to all projects and says so.

## What gets indexed

Each user prompt plus the assistant's text replies becomes one chunk:

```
User: <prompt>

Assistant: <reply text>

Actions:
- Edit /path/to/file.py
- Bash: pytest -q tests/
```

- Tool calls are summarised into the `Actions` list (file paths, commands, search
  patterns) so "which file did we change for X" is searchable. Tool *results*,
  thinking blocks and file contents are not indexed.
- Long turns are split into ~6000-character chunks that each repeat the prompt,
  instead of being truncated.
- Session titles and compaction summaries are indexed as `summary` chunks.
- Subagent transcripts (`<session>/subagents/*.jsonl`) are indexed as `subagent`
  chunks under the parent session id.
- Slash-command echoes, local command output, task notifications and injected
  `<system-reminder>` blocks are stripped and never indexed.

Transcripts are read incrementally by byte offset; only complete lines are consumed,
so a line still being written is picked up on the next pass.

## Architecture

```
Claude Code session
  ├─ SessionStart hook ─► session-rag-server.sh start   (idempotent, lock-protected)
  │                    └► session_start_hook.sh ─► POST /watch  (register project, backfill)
  ├─ Stop / PreCompact ─► index_hook.py ─► POST /index          (queue transcript now)
  └─ MCP tools ─────────► http://127.0.0.1:7102/mcp/  (X-Project-Root header per session)

session-rag server (http_server.py)
  ├─ file_watcher.py   FSEvents on ~/.claude/projects/**.jsonl ─┐
  ├─ indexer.py        single queue → parse new bytes → embed → insert
  │                    (offsets in index_state.json, written atomically)
  ├─ transcript_parser.py   JSONL → chunks
  ├─ embedder.py            MLX models (EmbeddingGemma / Qwen3)
  └─ rag_engine.py          Milvus Lite + FTS5, hybrid search, one worker thread
```

All engine work (embedding, Milvus, SQLite) runs on one worker thread, which keeps the
event loop responsive and avoids cross-thread SQLite use.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SESSION_RAG_MODEL` | from `~/.session-rag/config.json`, else `embeddinggemma` | `embeddinggemma`, `qwen3`, `qwen3-4bit` or `modernbert` |
| `SESSION_RAG_PORT` | `7102` | HTTP port |
| `SESSION_RAG_EXPIRE_DAYS` | `365` | Auto-prune chunks older than this (`0` disables) |
| `SESSION_RAG_WATCH` | `true` | Enable the file watcher |
| `SESSION_RAG_WATCH_DEBOUNCE` | `2.0` | Seconds of quiet before a changed transcript is indexed |
| `SESSION_RAG_LOG_LEVEL` | `INFO` | Server log level |
| `SESSION_RAG_START_CONFIRM` | `15` | Seconds `start` waits for `/health` before handing readiness to the watchdog |
| `SESSION_RAG_STARTUP_GRACE` | `180` | Seconds the watchdog leaves a live but not yet healthy server alone |
| `SESSION_RAG_CA_BUNDLE` | | Extra CA certificate (PEM) for model downloads behind a TLS proxy. `NODE_EXTRA_CA_CERTS` is honoured too. |

The model choice is normally stored in `~/.session-rag/config.json` (written by
`cleanup.py migrate-model`), so the launchd agent, the hooks and the CLI always agree.
The other variables are read from the server's environment: the launchd agent's plist
(`install-launchd` regenerates it) or the shell that runs `session-rag-server.sh`.

## Embedding models

| Name | Model | Dims | Context | Notes |
|------|-------|------|---------|-------|
| `embeddinggemma` | `mlx-community/embeddinggemma-300m-bf16` | 768 | 2048 tokens | Default. Fastest; best measured retrieval on session data. |
| `qwen3` | `mlx-community/Qwen3-Embedding-0.6B-8bit` | 1024 | 8192 tokens (32k max) | Long context, strong on code benchmarks, ~4x slower to index. |
| `qwen3-4bit` | `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` | 1024 | 8192 tokens | Smaller/faster Qwen3, slight quality loss. |
| `modernbert` | `nomic-ai/modernbert-embed-base` | 768 | 8192 tokens | Legacy option. |

EmbeddingGemma is a bidirectional encoder; `embedder.py` runs the reference forward pass
(bidirectional attention, padding masked, sliding window on local layers) because the
`mlx-embeddings` 0.0.5 implementation leaks padding tokens into batched embeddings.

Measured on this machine's own index (6,476 chunks, 94 session-title queries, hit rate of
a chunk from the right session):

| Model | Hit@1 | Hit@5 | Hit@10 | Index speed |
|-------|-------|-------|--------|-------------|
| EmbeddingGemma | 0.76 | 0.90 | 0.95 | ~60 chunks/s |
| Qwen3-Embedding-0.6B (8-bit) | 0.74 | 0.88 | 0.93 | ~14 chunks/s |

### Switching models

Vectors from different models are incompatible, so the index must be re-embedded. The
FTS mirror holds the text of every chunk, so nothing is lost even for transcripts that
Claude Code has since deleted:

```bash
./download-model.sh qwen3                     # once
./session-rag-server.sh stop
./venv/bin/python cleanup.py migrate-model --to qwen3   # re-embeds and records the model in config.json
./session-rag-server.sh start
```

The server stamps `~/.session-rag/model_identity.json` and refuses to start if the
configured model does not match the index.

## Data management

Stop the server first; Milvus Lite allows one process per database.

```bash
./session-rag-server.sh stop
./venv/bin/python cleanup.py list [--project /path]     # sessions
./venv/bin/python cleanup.py stats [--project /path]    # counts, model
./venv/bin/python cleanup.py expire --days 60           # delete old chunks
./venv/bin/python cleanup.py delete --session <id>      # or --branch <name>
./venv/bin/python cleanup.py prune-noise                # drop command-echo chunks from old indexes
./venv/bin/python cleanup.py migrate-model --to qwen3   # re-embed with another model
./venv/bin/python cleanup.py reindex                    # drop + re-read all transcripts on disk
./venv/bin/python cleanup.py reset                      # delete everything
./venv/bin/python cleanup.py models                     # list models / download state
./session-rag-server.sh start
```

## Server management

```bash
./session-rag-server.sh start     # idempotent; safe from concurrent sessions
./session-rag-server.sh stop      # stays stopped (unloads the launchd agent if installed)
./session-rag-server.sh status    # health + indexer stats
./session-rag-server.sh restart
./session-rag-server.sh logs      # tail -f the log
./session-rag-server.sh install-launchd     # launchd owns the process: start at login, restart on exit
./session-rag-server.sh uninstall-launchd
curl http://127.0.0.1:7102/health # 200 when ready, 503 while starting
curl http://127.0.0.1:7102/status
```

`start` returns within `SESSION_RAG_START_CONFIRM` seconds even on a cold boot; if the
model is still loading, the watchdog takes over and reports readiness in the log. The
watchdog restarts a server whose `/health` stays down, but never one that is still
starting, and backs off on repeated restarts.

With the launchd agent installed, `start`/`stop`/`restart` go through `launchctl`, so the
hooks, the watchdog and launchd can never start competing server processes (which is
what produced the "address already in use" / "Open local milvus failed" errors before).

Logs: `~/.session-rag/server.log` (rotated at 10 MB). PID: `~/.session-rag/server.pid`.

## Development

```bash
./venv/bin/python -m unittest discover -s tests -v
```

## Files

```
http_server.py          HTTP MCP server, /health /status /index /watch
indexer.py              single indexing pipeline (queue, offsets, backfill, project roots)
file_watcher.py         FSEvents → indexer
transcript_parser.py    JSONL transcripts → chunks
embedder.py             model registry + MLX embedding backends (+ download helper)
rag_engine.py           Milvus Lite + FTS5 storage, hybrid search
fts_hybrid.py           FTS5 sidecar, query building, RRF
index_state.py          atomic JSON state (offsets, slug map)
tools.py                MCP tool definitions
cleanup.py              maintenance CLI
index_hook.py           Stop/PreCompact hook
session_start_hook.sh   SessionStart hook
session-rag-server.sh   server lifecycle + watchdog
download-model.sh       model download
setup.sh                installer
tests/                  unit tests
```

Runtime files:

```
~/.claude.json                                 global MCP server entry (user scope)
~/.claude/settings.json                        hooks
~/.claude/mcp-helpers/session-rag-headers.sh   sends X-Project-Root per session
~/.session-rag/milvus.db                       vectors + metadata
~/.session-rag/fts.db                          keyword index / text mirror
~/.session-rag/index_state.json                per-transcript byte offsets
~/.session-rag/slug_map.json                   project slug → root path
~/.session-rag/model_identity.json             model the index was built with
~/.session-rag/config.json                     model choice shared by server + CLI
~/.session-rag/server.log, server.pid, watchdog.pid
~/Library/LaunchAgents/com.mattgreen.session-rag.plist   launchd agent (optional)
```

## Troubleshooting

- **Model download fails with a certificate error** (corporate TLS proxy): point
  `SESSION_RAG_CA_BUNDLE` at the proxy's CA PEM (or rely on `NODE_EXTRA_CA_CERTS`) and rerun
  `./download-model.sh`.
- **"Model mismatch" at startup**: `SESSION_RAG_MODEL` differs from the model the index was
  built with. Run `cleanup.py migrate-model --to <model>` or unset the variable.
- **"Open local milvus failed"**: another process holds the database (a second server,
  or `cleanup.py`). `./session-rag-server.sh status` shows the owner.
- **Results say "all projects (X has no indexed turns yet)"**: the project header did not
  match anything indexed. Chunks are tagged with the git repo root of the session's
  working directory; `search_all_sessions(project_root="*")` searches everything.
