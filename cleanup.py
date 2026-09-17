#!/usr/bin/env python3
"""
Maintenance CLI for the session-rag index (~/.session-rag/).

Stop the server first (./session-rag-server.sh stop): Milvus Lite allows one
process per database file.

Commands:
  list            [--project <root>]        List indexed sessions
  stats           [--project <root>]        Index statistics
  expire          [--days N]                Delete chunks older than N days (default 365)
  delete          --session <id> | --branch <name>
  prune-noise                               Remove chunks that are only Claude Code command echoes
  migrate-model   --to <model> [--keep-noise]
                                            Re-embed the whole index with another model
                                            (text comes from the FTS mirror; nothing is lost)
  reindex         [--yes]                   Drop the index and re-read every transcript on disk
  reset           [--yes]                   Delete everything
  models                                    List supported embedding models
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import urllib.request
from pathlib import Path

import embedder
import rag_engine
from index_state import STATE_DIR, IndexState

PID_FILE = STATE_DIR / "server.pid"
PORT = int(os.getenv("SESSION_RAG_PORT", "7102"))


def _server_running() -> bool:
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/status", timeout=2)
        return True
    except Exception:
        return True  # process alive even if HTTP is not answering


def require_server_stopped() -> None:
    if _server_running():
        print("The session-rag server is running and holds the database.\n"
              "Stop it first:  ./session-rag-server.sh stop", file=sys.stderr)
        sys.exit(2)


def _confirm(prompt: str, yes: bool) -> bool:
    if yes:
        return True
    return input(f"{prompt} [y/N] ").strip().lower() == "y"


def cmd_list(args):
    sessions = rag_engine.list_sessions(project_root=args.project)
    if not sessions:
        print("No sessions indexed.")
        return
    print(f"{'Session ID':<40} {'Chunks':>6} {'Project':<28} {'Branch':<22} {'Last':>19}")
    print("-" * 120)
    for s in sessions:
        branches = ", ".join(s["branches"]) if s["branches"] else "(none)"
        project = Path(s["project_root"]).name if s.get("project_root") else "(unknown)"
        print(f"{s['session_id']:<40} {s['turns']:>6} {project[:28]:<28} {branches[:22]:<22} {s['max_ts'][:19]:>19}")
    print(f"\nTotal: {len(sessions)} sessions, {sum(s['turns'] for s in sessions)} chunks")


def cmd_stats(args):
    stats = rag_engine.get_stats(project_root=args.project)
    identity = rag_engine.read_identity()
    if args.project:
        print(f"Project:      {args.project}")
    print(f"Total chunks: {stats['total_turns']}")
    print(f"Sessions:     {stats['sessions']}")
    print(f"Projects:     {len(stats.get('projects', []))}")
    print(f"Model:        {identity.get('model_name', '?')} ({identity.get('model_id', '?')}, {identity.get('embed_dim', '?')}d)")
    if stats.get("branches"):
        print(f"Branches:     {', '.join(stats['branches'][:30])}")
    if stats.get("by_type"):
        print("\nBy type:")
        for t, c in sorted(stats["by_type"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {t}: {c}")
    print(f"\nDB location:  {rag_engine.DEFAULT_DB_PATH}")


def cmd_expire(args):
    before = rag_engine.get_stats()
    print(f"Current index: {before['total_turns']} chunks across {before['sessions']} sessions")
    count = rag_engine.delete_older_than(args.days)
    after = rag_engine.get_stats()
    print(f"Deleted {count} chunks older than {args.days} days. "
          f"Remaining: {after['total_turns']} chunks across {after['sessions']} sessions")


def cmd_delete(args):
    if args.session:
        print(f"Deleted {rag_engine.delete_by_session(args.session)} chunks for session {args.session}")
    elif args.branch:
        print(f"Deleted {rag_engine.delete_by_branch(args.branch)} chunks for branch '{args.branch}'")
    else:
        print("Error: specify --session or --branch", file=sys.stderr)
        sys.exit(1)


def cmd_prune_noise(args):
    print(f"Removed {rag_engine.delete_noise()} noise chunks.")


def cmd_migrate_model(args):
    spec = rag_engine.use_model(args.to)
    if not spec.is_downloaded():
        print(f"Model '{spec.name}' is not downloaded. Run: ./download-model.sh {spec.name}", file=sys.stderr)
        sys.exit(1)
    identity = rag_engine.read_identity()
    print(f"Current index model: {identity.get('model_name', '?')} -> new model: {spec.name} ({spec.model_id})")
    n = rag_engine.reembed_all(progress=print, skip_noise=not args.keep_noise)
    embedder.write_config(model=spec.name)
    env_model = os.environ.get("SESSION_RAG_MODEL")
    print(f"\nMigration complete ({n} chunks). Model '{spec.name}' recorded in {embedder.CONFIG_PATH}.")
    if env_model and env_model.lower() != spec.name:
        print(f"Note: SESSION_RAG_MODEL={env_model} is set in this shell and overrides the config file; "
              f"unset it or set it to {spec.name}.")
    print("Start the server again: ./session-rag-server.sh start")


def cmd_reindex(args):
    state = IndexState()
    known = list(state.transcripts)
    missing = [p for p in known if not os.path.exists(p)]
    print(f"{len(known)} transcripts tracked, {len(missing)} no longer exist on disk.")
    if missing:
        print("Chunks from deleted transcripts cannot be recovered by re-reading; use "
              "'migrate-model' instead if you only want new vectors.")
    if not _confirm("Drop the index and re-read every transcript at next server start?", args.yes):
        print("Cancelled.")
        return
    rag_engine.clear_collection()
    state.reset_all_offsets()
    state.save(force=True)
    print("Index cleared and offsets reset. Start the server to rebuild: ./session-rag-server.sh start")


def cmd_reset(args):
    if not _confirm("This deletes ALL indexed data for all projects. Continue?", args.yes):
        print("Cancelled.")
        return
    rag_engine.clear_collection()
    state = IndexState()
    state.reset_all_offsets()
    state.save(force=True)
    print("Reset complete.")


def cmd_models(args):
    active = embedder.resolve_model_name()
    identity = rag_engine.read_identity()
    source = ("SESSION_RAG_MODEL" if os.environ.get("SESSION_RAG_MODEL")
              else "config.json" if embedder.read_config().get("model") else "default")
    for spec in embedder.REGISTRY.values():
        flags = []
        if spec.name == active:
            flags.append(f"active ({source})")
        if spec.name == identity.get("model_name"):
            flags.append("index built with this")
        state = "downloaded" if spec.is_downloaded() else "not downloaded"
        print(f"{spec.name:<14} {spec.model_id:<46} {spec.dim:>5}d  {state}"
              + (f"  [{'; '.join(flags)}]" if flags else ""))
        print(f"               {spec.description}")


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Manage the session-rag index",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="List indexed sessions"); p.add_argument("--project")
    p = sub.add_parser("stats", help="Show index statistics"); p.add_argument("--project")
    p = sub.add_parser("expire", help="Delete chunks older than N days"); p.add_argument("--days", type=int, default=365)
    p = sub.add_parser("delete", help="Delete by session or branch"); p.add_argument("--session"); p.add_argument("--branch")
    sub.add_parser("prune-noise", help="Remove command-echo noise chunks")
    p = sub.add_parser("migrate-model", help="Re-embed the index with another model")
    p.add_argument("--to", required=True, help=f"Target model: {', '.join(embedder.REGISTRY)}")
    p.add_argument("--keep-noise", action="store_true", help="Keep noise chunks instead of dropping them")
    p = sub.add_parser("reindex", help="Drop the index and re-read all transcripts"); p.add_argument("--yes", "-y", action="store_true")
    p = sub.add_parser("reset", help="Delete all data"); p.add_argument("--yes", "-y", action="store_true")
    sub.add_parser("models", help="List embedding models")

    args = parser.parse_args()
    if args.command != "models":
        require_server_stopped()
    {
        "list": cmd_list, "stats": cmd_stats, "expire": cmd_expire, "delete": cmd_delete,
        "prune-noise": cmd_prune_noise, "migrate-model": cmd_migrate_model,
        "reindex": cmd_reindex, "reset": cmd_reset, "models": cmd_models,
    }[args.command](args)


if __name__ == "__main__":
    main()
