#!/usr/bin/env python3
"""
CLI tool to manage and clean up session-rag data.

All data lives in a single global DB at ~/.session-rag/milvus.db.
Use --project to filter to a specific project.

Usage:
    cleanup.py list        [--project <root>]                # List all sessions
    cleanup.py expire      [--days N]                        # Delete turns older than N days (default: 365)
    cleanup.py delete      --session <id>                    # Delete a specific session
    cleanup.py delete      --branch <name>                   # Delete all turns for a branch
    cleanup.py reset                                         # Drop everything and start fresh
    cleanup.py stats       [--project <root>]                # Show index statistics
"""

import argparse
import sys
from pathlib import Path

import rag_engine


def get_db_path() -> str:
    """Global DB path — all projects share one index."""
    return str(Path.home() / ".session-rag" / "milvus.db")


def cmd_list(args):
    db = get_db_path()
    project = getattr(args, 'project', None)
    sessions = rag_engine.list_sessions(project_root=project, db_path=db)

    if not sessions:
        print("No sessions indexed.")
        return

    print(f"{'Session ID':<40} {'Turns':>6} {'Branch':<30} {'First':>20} {'Last':>20}")
    print("-" * 120)
    for s in sessions:
        branches = ", ".join(s["branches"]) if s["branches"] else "(none)"
        first = s["min_ts"][:19] if s["min_ts"] else ""
        last = s["max_ts"][:19] if s["max_ts"] else ""
        print(f"{s['session_id']:<40} {s['turns']:>6} {branches:<30} {first:>20} {last:>20}")

    print(f"\nTotal: {len(sessions)} sessions, {sum(s['turns'] for s in sessions)} turns")


def cmd_expire(args):
    db = get_db_path()
    days = args.days

    # Show what would be deleted
    stats_before = rag_engine.get_stats(db_path=db)
    print(f"Current index: {stats_before['total_turns']} turns across {stats_before['sessions']} sessions")
    print(f"Deleting turns older than {days} days...")

    count = rag_engine.delete_older_than(days, db_path=db)
    print(f"Deleted {count} turns.")

    stats_after = rag_engine.get_stats(db_path=db)
    print(f"Remaining: {stats_after['total_turns']} turns across {stats_after['sessions']} sessions")


def cmd_delete(args):
    db = get_db_path()

    if args.session:
        count = rag_engine.delete_by_session(args.session, db_path=db)
        print(f"Deleted {count} turns for session {args.session}")
    elif args.branch:
        count = rag_engine.delete_by_branch(args.branch, db_path=db)
        print(f"Deleted {count} turns for branch '{args.branch}'")
    else:
        print("Error: specify --session or --branch", file=sys.stderr)
        sys.exit(1)


def cmd_reset(args):
    db = get_db_path()

    if not args.yes:
        answer = input("This will delete ALL indexed data (all projects). Continue? [y/N] ")
        if answer.lower() != "y":
            print("Cancelled.")
            return

    rag_engine.clear_collection(db_path=db)

    # Clean up WAL/SHM files left by SQLite FTS
    from fts_hybrid import FTSIndex
    fts_path = Path(FTSIndex.db_path(db))
    for suffix in ("-wal", "-shm"):
        f = Path(str(fts_path) + suffix)
        if f.exists():
            f.unlink()

    print("Reset complete. All session data deleted.")


def cmd_stats(args):
    db = get_db_path()
    project = getattr(args, 'project', None)
    stats = rag_engine.get_stats(project_root=project, db_path=db)

    if project:
        print(f"Project:      {project}")
    print(f"Total turns:  {stats['total_turns']}")
    print(f"Sessions:     {stats['sessions']}")

    if stats.get("branches"):
        print(f"Branches:     {', '.join(stats['branches'])}")

    if stats.get("by_type"):
        print("\nBy type:")
        for t, count in sorted(stats["by_type"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {t}: {count}")

    print(f"\nDB location:  {db}")


def main():
    parser = argparse.ArgumentParser(
        description="Manage session-rag index data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = subparsers.add_parser("list", help="List all indexed sessions")
    p_list.add_argument("--project", help="Filter to a specific project root")

    # expire
    p_expire = subparsers.add_parser("expire", help="Delete turns older than N days")
    p_expire.add_argument("--days", type=int, default=365, help="Max age in days (default: 365)")

    # delete
    p_delete = subparsers.add_parser("delete", help="Delete by session or branch")
    p_delete.add_argument("--session", help="Session ID to delete")
    p_delete.add_argument("--branch", help="Git branch to delete")

    # reset
    p_reset = subparsers.add_parser("reset", help="Delete all data (full reset)")
    p_reset.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    # stats
    p_stats = subparsers.add_parser("stats", help="Show index statistics")
    p_stats.add_argument("--project", help="Filter to a specific project root")

    args = parser.parse_args()

    commands = {
        "list": cmd_list,
        "expire": cmd_expire,
        "delete": cmd_delete,
        "reset": cmd_reset,
        "stats": cmd_stats,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
