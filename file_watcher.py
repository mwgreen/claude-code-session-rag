"""
File watcher for automatic transcript indexing.

Uses macOS FSEvents (via watchdog) to detect .jsonl transcript changes and
index new turns in real time. When a transcript file grows (new turns written
by Claude Code), the watcher debounces briefly then reads from the last known
byte offset — nothing is lost, just batched.

Pipeline: FSEvents -> watchdog thread -> asyncio.Queue -> debounce -> incremental parse + index

The server owns index_state.json exclusively — no more race conditions from
concurrent hook processes.
"""

import asyncio
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


def _log(msg: str):
    print(f"[watcher] {msg}", file=sys.stderr)


# --- TranscriptChangeHandler ---

class TranscriptChangeHandler(FileSystemEventHandler):
    """Watchdog event handler that filters for .jsonl transcript files."""

    def __init__(self, change_queue: asyncio.Queue,
                 loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.change_queue = change_queue
        self.loop = loop

    def _should_handle(self, path: str) -> bool:
        """Only handle .jsonl files (Claude Code transcripts)."""
        return path.endswith('.jsonl')

    def _enqueue(self, path: str):
        """Thread-safe enqueue to asyncio loop."""
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


# --- TranscriptWatcher ---

class TranscriptWatcher:
    """Watches Claude Code transcript directories and indexes new turns."""

    def __init__(self, project_root: str, db_path: str,
                 debounce_seconds: float = 2.0):
        self.project_root = project_root
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

    def _get_transcript_dir(self) -> Optional[str]:
        """Derive the Claude Code transcript directory for this project.

        Claude stores transcripts at:
          ~/.claude/projects/{slug}/{session_id}.jsonl
        where slug is the project root absolute path with / and . replaced by -.
        """
        claude_projects = Path.home() / ".claude" / "projects"
        if not claude_projects.is_dir():
            return None

        # Claude Code slug: replace / and . with -
        slug = self.project_root.replace("/", "-").replace(".", "-")
        transcript_dir = claude_projects / slug
        if transcript_dir.is_dir():
            return str(transcript_dir)

        # Also try without leading dash
        slug_no_lead = slug.lstrip("-")
        transcript_dir = claude_projects / slug_no_lead
        if transcript_dir.is_dir():
            return str(transcript_dir)

        # Fallback: scan directories for one that contains this project path
        # (handles unknown slug transformations)
        normalized = self.project_root.rstrip("/").lower()
        for d in claude_projects.iterdir():
            if not d.is_dir():
                continue
            # Reconstruct path from slug: leading dash = /, internal dashes = / or .
            # Check if slug matches by comparing normalized forms
            candidate = d.name.replace("-", "/").lower()
            if candidate == normalized or candidate == "/" + normalized:
                return str(d)

        return None

    async def start(self):
        """Start watching the transcript directory."""
        transcript_dir = self._get_transcript_dir()
        if not transcript_dir:
            _log(f"No transcript directory found for {self.project_root}")
            return

        loop = asyncio.get_event_loop()
        handler = TranscriptChangeHandler(self._change_queue, loop)

        self._observer = Observer()
        self._observer.schedule(handler, transcript_dir, recursive=False)
        self._observer.daemon = True
        self._observer.start()

        self._drain_task = asyncio.create_task(self._drain_queue())

        n_files = len(list(Path(transcript_dir).glob("*.jsonl")))
        _log(f"Watching {transcript_dir} ({n_files} transcripts, "
             f"debounce={self.debounce_seconds}s)")

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
        """Reset the debounce timer. Called each time a new event arrives."""
        if self._debounce_handle is not None:
            self._debounce_handle.cancel()

        loop = asyncio.get_event_loop()
        self._debounce_handle = loop.call_later(
            self.debounce_seconds,
            lambda: asyncio.ensure_future(self._trigger_processing())
        )

    async def _trigger_processing(self):
        """Called when debounce timer fires."""
        if self._stopped:
            return

        # Don't start a new batch while one is processing
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
            # Load state once for the whole batch
            state = transcript_parser.load_index_state(self.project_root)

            for transcript_path in batch:
                if not Path(transcript_path).exists():
                    continue

                # Extract session_id from filename (uuid.jsonl)
                session_id = Path(transcript_path).stem

                offset = transcript_parser.get_transcript_offset(
                    state, transcript_path)

                try:
                    turns, new_offset = transcript_parser.parse_transcript(
                        transcript_path, session_id, start_offset=offset
                    )

                    if turns:
                        loop = asyncio.get_event_loop()
                        count = await rag_engine.add_turns_async(
                            turns, db_path=self.db_path)
                        total_indexed += count
                        _log(f"Indexed {count} turns from "
                             f"{Path(transcript_path).name} "
                             f"(session {session_id})")

                    # Always advance offset (even if no turns — skip tool_results etc)
                    transcript_parser.set_transcript_offset(
                        state, transcript_path, new_offset)

                except Exception as e:
                    _log(f"Error indexing {Path(transcript_path).name}: {e}")
                    errors += 1

            # Save state once after the whole batch
            transcript_parser.save_index_state(self.project_root, state)

            self.stats['turns_indexed'] += total_indexed
            self.stats['files_processed'] += len(batch)
            self.stats['batches_processed'] += 1
            self.stats['errors'] += errors

        except Exception as e:
            _log(f"Batch processing error: {e}")
            self.stats['errors'] += 1
        finally:
            self._processing = False
            # If more changes accumulated during processing, trigger again
            if self._pending_files:
                self._reset_debounce()

    async def backfill(self):
        """Index all unindexed or partially-indexed transcript files.

        Called on startup to catch sessions that were missed (server wasn't
        running, race conditions, etc).
        """
        transcript_dir = self._get_transcript_dir()
        if not transcript_dir:
            return 0

        state = transcript_parser.load_index_state(self.project_root)
        transcripts = list(Path(transcript_dir).glob("*.jsonl"))

        needs_indexing = []
        for t in transcripts:
            path = str(t)
            offset = transcript_parser.get_transcript_offset(state, path)
            file_size = t.stat().st_size
            if offset < file_size:
                needs_indexing.append((path, t.stem, offset))

        if not needs_indexing:
            _log(f"Backfill: all {len(transcripts)} transcripts up to date")
            return 0

        _log(f"Backfill: {len(needs_indexing)} of {len(transcripts)} "
             f"transcripts need indexing...")

        total_indexed = 0
        for transcript_path, session_id, offset in needs_indexing:
            try:
                turns, new_offset = transcript_parser.parse_transcript(
                    transcript_path, session_id, start_offset=offset
                )

                if turns:
                    count = await rag_engine.add_turns_async(
                        turns, db_path=self.db_path)
                    total_indexed += count

                transcript_parser.set_transcript_offset(
                    state, transcript_path, new_offset)

            except Exception as e:
                _log(f"Backfill error {Path(transcript_path).name}: {e}")

            # Yield control periodically
            await asyncio.sleep(0)

        transcript_parser.save_index_state(self.project_root, state)
        _log(f"Backfill complete: {total_indexed} turns indexed from "
             f"{len(needs_indexing)} transcripts")
        return total_indexed


# --- Watcher Manager ---

_watchers: Dict[str, TranscriptWatcher] = {}


async def ensure_watcher(project_root: str, db_path: str) -> Optional[TranscriptWatcher]:
    """Ensure a watcher exists for the given project. Creates one if needed.
    Returns None if watching is disabled."""
    if not _watcher_config['enabled']:
        return None
    if project_root in _watchers:
        return _watchers[project_root]

    watcher = TranscriptWatcher(
        project_root=project_root,
        db_path=db_path,
        debounce_seconds=_watcher_config['debounce_seconds'],
    )
    await watcher.start()
    _watchers[project_root] = watcher
    return watcher


async def stop_all_watchers():
    """Stop all active watchers. Called during server shutdown."""
    for watcher in list(_watchers.values()):
        await watcher.stop()
    _watchers.clear()


def get_watcher_status() -> Dict:
    """Return status of all active watchers."""
    return {
        root: {
            'pending': len(w._pending_files),
            'processing': w._processing,
            'stats': dict(w.stats),
        }
        for root, w in _watchers.items()
    }
