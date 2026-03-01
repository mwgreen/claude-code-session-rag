"""
Global file watcher for automatic transcript indexing.

Uses macOS FSEvents (via watchdog) to monitor ~/.claude/projects/ for all
transcript changes across every project. A single Observer handles everything.

Pipeline: FSEvents -> watchdog thread -> asyncio.Queue -> debounce -> incremental parse + index

Slug→root mapping is persisted at ~/.session-rag/slug_map.json, populated by
/watch and /index hooks. Unknown slugs get project_root="" until a hook registers them.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Set

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import rag_engine
import transcript_parser


# --- Configuration ---

_watcher_config = {
    'enabled': os.getenv('SESSION_RAG_WATCH', 'true').lower() in ('true', '1', 'yes'),
    'debounce_seconds': float(os.getenv('SESSION_RAG_WATCH_DEBOUNCE', '2.0')),
}

_CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"
_SLUG_MAP_PATH = Path.home() / ".session-rag" / "slug_map.json"


def _log(msg: str):
    print(f"[watcher] {msg}", file=sys.stderr)


# --- Slug <-> project root mapping ---

_slug_map: Dict[str, str] = {}  # slug -> project_root


def _load_slug_map():
    """Load persisted slug→root mapping."""
    global _slug_map
    if _SLUG_MAP_PATH.exists():
        try:
            with open(_SLUG_MAP_PATH) as f:
                _slug_map = json.load(f)
        except (json.JSONDecodeError, IOError):
            _slug_map = {}


def _save_slug_map():
    """Persist slug→root mapping."""
    _SLUG_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_SLUG_MAP_PATH, "w") as f:
        json.dump(_slug_map, f, indent=2)


def _project_root_to_slug(project_root: str) -> str:
    """Convert a project root path to its Claude Code slug.

    Claude Code slugs: absolute path with / and . replaced by -.
    e.g., /Users/matt/project -> -Users-matt-project
    """
    return project_root.replace("/", "-").replace(".", "-")


def _slug_from_transcript_path(transcript_path: str) -> Optional[str]:
    """Extract the slug directory name from a transcript path.

    Transcript paths are like:
      ~/.claude/projects/{slug}/{session_id}.jsonl
    """
    try:
        path = Path(transcript_path)
        projects_str = str(_CLAUDE_PROJECTS)
        path_str = str(path.parent)
        if path_str.startswith(projects_str):
            relative = path_str[len(projects_str):].strip("/")
            # The slug is the first path component
            return relative.split("/")[0] if relative else None
    except Exception:
        pass
    return None


def register_project(project_root: str):
    """Register a project root, recording its slug→root mapping.

    Called by /watch and /index hooks when they know the real project root.
    """
    slug = _project_root_to_slug(project_root)
    if slug not in _slug_map or _slug_map[slug] != project_root:
        _slug_map[slug] = project_root
        _save_slug_map()
        _log(f"Registered slug {slug} -> {project_root}")


def _get_project_root_for_slug(slug: str) -> str:
    """Look up the project root for a slug. Returns "" if unknown."""
    return _slug_map.get(slug, "")


# --- TranscriptChangeHandler ---

class TranscriptChangeHandler(FileSystemEventHandler):
    """Watchdog event handler that filters for .jsonl transcript files."""

    def __init__(self, change_queue: asyncio.Queue,
                 loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.change_queue = change_queue
        self.loop = loop

    def _should_handle(self, path: str) -> bool:
        return path.endswith('.jsonl')

    def _enqueue(self, path: str):
        try:
            self.loop.call_soon_threadsafe(
                self.change_queue.put_nowait, path
            )
        except RuntimeError:
            pass  # Loop closed during shutdown

    def on_modified(self, event):
        if not event.is_directory and self._should_handle(event.src_path):
            self._enqueue(event.src_path)

    def on_created(self, event):
        if not event.is_directory and self._should_handle(event.src_path):
            self._enqueue(event.src_path)


# --- GlobalTranscriptWatcher ---

class GlobalTranscriptWatcher:
    """Watches ~/.claude/projects/ recursively and indexes new turns from all projects."""

    def __init__(self, db_path: str, debounce_seconds: float = 2.0):
        self.db_path = db_path
        self.debounce_seconds = debounce_seconds

        self._observer: Optional[Observer] = None
        self._change_queue: asyncio.Queue = asyncio.Queue()
        self._drain_task: Optional[asyncio.Task] = None
        self._pending_files: Set[str] = set()
        self._debounce_handle: Optional[asyncio.TimerHandle] = None
        self._processing = False
        self._stopped = False
        self.stats = {
            'turns_indexed': 0,
            'files_processed': 0,
            'batches_processed': 0,
            'errors': 0,
        }

    async def start(self):
        """Start watching ~/.claude/projects/ recursively."""
        if not _CLAUDE_PROJECTS.is_dir():
            _log(f"No Claude projects directory at {_CLAUDE_PROJECTS}")
            return

        loop = asyncio.get_event_loop()
        handler = TranscriptChangeHandler(self._change_queue, loop)

        self._observer = Observer()
        self._observer.schedule(handler, str(_CLAUDE_PROJECTS), recursive=True)
        self._observer.daemon = True
        self._observer.start()

        self._drain_task = asyncio.create_task(self._drain_queue())

        # Count total transcript files across all project slug dirs
        n_files = sum(1 for _ in _CLAUDE_PROJECTS.rglob("*.jsonl"))
        n_slugs = sum(1 for d in _CLAUDE_PROJECTS.iterdir() if d.is_dir())
        _log(f"Watching {_CLAUDE_PROJECTS} ({n_slugs} project dirs, "
             f"{n_files} transcripts, debounce={self.debounce_seconds}s)")

    async def stop(self):
        """Stop the watcher and cancel pending work."""
        self._stopped = True

        if self._debounce_handle is not None:
            self._debounce_handle.cancel()
            self._debounce_handle = None

        if self._drain_task is not None:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass
            self._drain_task = None

        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

        _log(f"Stopped (indexed={self.stats['turns_indexed']}, "
             f"files={self.stats['files_processed']}, "
             f"batches={self.stats['batches_processed']})")

    async def _drain_queue(self):
        """Continuously read events from the queue into the pending set."""
        try:
            while True:
                path = await self._change_queue.get()
                self._pending_files.add(path)
                self._reset_debounce()
        except asyncio.CancelledError:
            pass

    def _reset_debounce(self):
        if self._debounce_handle is not None:
            self._debounce_handle.cancel()

        loop = asyncio.get_event_loop()
        self._debounce_handle = loop.call_later(
            self.debounce_seconds,
            lambda: asyncio.ensure_future(self._trigger_processing())
        )

    async def _trigger_processing(self):
        if self._stopped:
            return
        if self._processing:
            self._reset_debounce()
            return
        await self._process_batch()

    async def _process_batch(self):
        """Process accumulated transcript changes."""
        if not self._pending_files:
            return

        self._processing = True

        # Snapshot and clear
        batch = set(self._pending_files)
        self._pending_files.clear()

        total_indexed = 0
        errors = 0

        try:
            state = transcript_parser.load_index_state()

            for transcript_path in batch:
                if not Path(transcript_path).exists():
                    continue

                slug = _slug_from_transcript_path(transcript_path)
                project_root = _get_project_root_for_slug(slug) if slug else ""
                session_id = Path(transcript_path).stem

                offset = transcript_parser.get_transcript_offset(
                    state, transcript_path)

                try:
                    turns, new_offset = transcript_parser.parse_transcript(
                        transcript_path, session_id, start_offset=offset
                    )

                    if turns:
                        for t in turns:
                            t["project_root"] = project_root
                        count = await rag_engine.add_turns_async(
                            turns, db_path=self.db_path)
                        total_indexed += count
                        _log(f"Indexed {count} turns from "
                             f"{Path(transcript_path).name} "
                             f"(session {session_id[:8]})")

                    transcript_parser.set_transcript_offset(
                        state, transcript_path, new_offset,
                        project_root=project_root)

                except Exception as e:
                    _log(f"Error indexing {Path(transcript_path).name}: {e}")
                    errors += 1

            transcript_parser.save_index_state(state)

            self.stats['turns_indexed'] += total_indexed
            self.stats['files_processed'] += len(batch)
            self.stats['batches_processed'] += 1
            self.stats['errors'] += errors

        except Exception as e:
            _log(f"Batch processing error: {e}")
            self.stats['errors'] += 1
        finally:
            self._processing = False
            if self._pending_files:
                self._reset_debounce()

    async def backfill(self, slug_filter: Optional[str] = None):
        """Index all unindexed or partially-indexed transcript files.

        If slug_filter is provided, only backfill that slug's directory.
        Otherwise, backfill ALL project slug directories.
        """
        if not _CLAUDE_PROJECTS.is_dir():
            return 0

        state = transcript_parser.load_index_state()

        # Collect all transcript files to check
        if slug_filter:
            slug_dir = _CLAUDE_PROJECTS / slug_filter
            if slug_dir.is_dir():
                transcripts = list(slug_dir.glob("*.jsonl"))
            else:
                return 0
        else:
            transcripts = list(_CLAUDE_PROJECTS.rglob("*.jsonl"))

        needs_indexing = []
        for t in transcripts:
            path = str(t)
            offset = transcript_parser.get_transcript_offset(state, path)
            try:
                file_size = t.stat().st_size
            except OSError:
                continue
            if offset < file_size:
                slug = _slug_from_transcript_path(path)
                project_root = _get_project_root_for_slug(slug) if slug else ""
                needs_indexing.append((path, t.stem, offset, project_root))

        if not needs_indexing:
            scope = slug_filter or "all projects"
            _log(f"Backfill ({scope}): all {len(transcripts)} transcripts up to date")
            return 0

        scope = slug_filter or "all projects"
        _log(f"Backfill ({scope}): {len(needs_indexing)} of {len(transcripts)} "
             f"transcripts need indexing...")

        total_indexed = 0
        for transcript_path, session_id, offset, project_root in needs_indexing:
            try:
                turns, new_offset = transcript_parser.parse_transcript(
                    transcript_path, session_id, start_offset=offset
                )

                if turns:
                    for t in turns:
                        t["project_root"] = project_root
                    count = await rag_engine.add_turns_async(
                        turns, db_path=self.db_path)
                    total_indexed += count

                transcript_parser.set_transcript_offset(
                    state, transcript_path, new_offset,
                    project_root=project_root)

            except Exception as e:
                _log(f"Backfill error {Path(transcript_path).name}: {e}")

            # Yield control periodically
            await asyncio.sleep(0)

        transcript_parser.save_index_state(state)
        _log(f"Backfill ({scope}) complete: {total_indexed} turns indexed from "
             f"{len(needs_indexing)} transcripts")
        return total_indexed


# --- Global watcher singleton ---

_global_watcher: Optional[GlobalTranscriptWatcher] = None


async def start_global_watcher(db_path: str) -> Optional[GlobalTranscriptWatcher]:
    """Start the single global watcher. Called once at server startup."""
    global _global_watcher
    if not _watcher_config['enabled']:
        return None
    if _global_watcher is not None:
        return _global_watcher

    _load_slug_map()

    watcher = GlobalTranscriptWatcher(
        db_path=db_path,
        debounce_seconds=_watcher_config['debounce_seconds'],
    )
    await watcher.start()
    _global_watcher = watcher
    return watcher


async def stop_global_watcher():
    """Stop the global watcher. Called during server shutdown."""
    global _global_watcher
    if _global_watcher is not None:
        await _global_watcher.stop()
        _global_watcher = None


def get_watcher_status() -> Dict:
    """Return status of the global watcher."""
    if _global_watcher is None:
        return {}
    return {
        'global': {
            'watching': str(_CLAUDE_PROJECTS),
            'pending': len(_global_watcher._pending_files),
            'processing': _global_watcher._processing,
            'stats': dict(_global_watcher.stats),
            'registered_projects': len(_slug_map),
        }
    }


def get_global_watcher() -> Optional[GlobalTranscriptWatcher]:
    """Get the global watcher instance (for backfill triggers etc)."""
    return _global_watcher
