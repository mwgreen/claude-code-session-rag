"""
Filesystem watcher: turns FSEvents on ~/.claude/projects/ into Indexer.enqueue() calls.

One recursive watchdog Observer covers every project. All parsing/indexing
happens in the Indexer; this module only forwards paths.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from indexer import CLAUDE_PROJECTS, Indexer

logger = logging.getLogger("session-rag.watcher")

WATCH_ENABLED = os.getenv("SESSION_RAG_WATCH", "true").lower() in ("true", "1", "yes")
DEBOUNCE_SECONDS = float(os.getenv("SESSION_RAG_WATCH_DEBOUNCE", "2.0"))


class _Handler(FileSystemEventHandler):
    def __init__(self, indexer: Indexer, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self._indexer = indexer
        self._loop = loop

    def _forward(self, path) -> None:
        path = os.fsdecode(path) if isinstance(path, bytes) else str(path)
        if not path.endswith(".jsonl"):
            return
        try:
            self._loop.call_soon_threadsafe(self._indexer.enqueue, path)
        except RuntimeError:
            pass  # event loop closed during shutdown

    def on_modified(self, event):
        if not event.is_directory:
            self._forward(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self._forward(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self._forward(event.dest_path)


class TranscriptWatcher:
    def __init__(self, indexer: Indexer, root: Path = CLAUDE_PROJECTS):
        self._indexer = indexer
        self._root = root
        self._observer: Optional[Observer] = None

    def start(self) -> bool:
        if not self._root.is_dir():
            logger.warning("No Claude projects directory at %s; watcher not started", self._root)
            return False
        loop = asyncio.get_running_loop()
        self._observer = Observer()
        self._observer.schedule(_Handler(self._indexer, loop), str(self._root), recursive=True)
        self._observer.daemon = True
        self._observer.start()
        n_slugs = sum(1 for d in self._root.iterdir() if d.is_dir())
        logger.info("Watching %s (%d project dirs, debounce=%.1fs)", self._root, n_slugs,
                    self._indexer.debounce_seconds)
        return True

    def stop(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

    def status(self) -> Dict:
        return {
            "watching": str(self._root),
            "alive": bool(self._observer is not None and self._observer.is_alive()),
        }


_watcher: Optional[TranscriptWatcher] = None


def start_watcher(indexer: Indexer) -> Optional[TranscriptWatcher]:
    global _watcher
    if not WATCH_ENABLED:
        logger.info("File watcher disabled (SESSION_RAG_WATCH=false)")
        return None
    if _watcher is None:
        _watcher = TranscriptWatcher(indexer)
        if not _watcher.start():
            _watcher = None
    return _watcher


def stop_watcher() -> None:
    global _watcher
    if _watcher is not None:
        _watcher.stop()
        _watcher = None


def get_watcher_status() -> Dict:
    return _watcher.status() if _watcher is not None else {"watching": None, "alive": False}
