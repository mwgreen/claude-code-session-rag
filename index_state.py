"""
Persistent indexing state for session-rag.

Two small JSON files under ~/.session-rag/:
  index_state.json  - per-transcript byte offsets (how far each file has been indexed)
  slug_map.json     - Claude Code project slug -> project root path

Both are written atomically (temp file + rename) so a crash mid-write can never
leave a truncated file behind. A truncated index_state.json would otherwise
reset every offset and trigger a full re-index of all transcripts.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger("session-rag.state")

STATE_DIR = Path.home() / ".session-rag"
STATE_PATH = STATE_DIR / "index_state.json"
SLUG_MAP_PATH = STATE_DIR / "slug_map.json"


def atomic_write_json(path: Path, data: Any, indent: Optional[int] = 1) -> None:
    """Write JSON to `path` atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Could not read %s (%s); starting from empty state", path, exc)
        return default


class IndexState:
    """Byte-offset bookkeeping for every transcript file."""

    def __init__(self, path: Union[str, Path] = STATE_PATH):
        self.path = Path(path)
        data = read_json(self.path, {})
        if not isinstance(data, dict):
            data = {}
        self._data: Dict[str, Any] = data
        self._data.setdefault("transcripts", {})
        self._dirty = False

    # -- offsets --
    @property
    def transcripts(self) -> Dict[str, Dict[str, Any]]:
        return self._data["transcripts"]

    def get_offset(self, transcript_path: str) -> int:
        return int(self.transcripts.get(transcript_path, {}).get("last_byte_offset", 0))

    def get_project_root(self, transcript_path: str) -> str:
        return self.transcripts.get(transcript_path, {}).get("project_root", "") or ""

    def set_offset(self, transcript_path: str, offset: int, project_root: str = "") -> None:
        entry = self.transcripts.setdefault(transcript_path, {})
        if entry.get("last_byte_offset") != offset:
            entry["last_byte_offset"] = offset
            self._dirty = True
        if project_root and entry.get("project_root") != project_root:
            entry["project_root"] = project_root
            self._dirty = True

    def forget(self, transcript_path: str) -> None:
        if self.transcripts.pop(transcript_path, None) is not None:
            self._dirty = True

    def reset_all_offsets(self) -> int:
        n = len(self.transcripts)
        self._data["transcripts"] = {}
        self._dirty = True
        return n

    # -- misc keys (e.g. last_expire_check) --
    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        if self._data.get(key) != value:
            self._data[key] = value
            self._dirty = True

    # -- persistence --
    @property
    def dirty(self) -> bool:
        return self._dirty

    def save(self, force: bool = False) -> bool:
        if not (self._dirty or force):
            return False
        atomic_write_json(self.path, self._data)
        self._dirty = False
        return True


class SlugMap:
    """Claude Code stores transcripts under ~/.claude/projects/<slug>/ where the slug
    is the launch directory with '/' and '.' replaced by '-'. The mapping back to a
    real path is ambiguous, so we record it whenever we learn it (from hooks or from
    the `cwd` field inside transcripts)."""

    def __init__(self, path: Union[str, Path] = SLUG_MAP_PATH):
        self.path = Path(path)
        data = read_json(self.path, {})
        self._map: Dict[str, str] = data if isinstance(data, dict) else {}

    @staticmethod
    def slug_for(project_root: str) -> str:
        return project_root.replace("/", "-").replace(".", "-")

    def get(self, slug: str) -> str:
        return self._map.get(slug, "")

    def __len__(self) -> int:
        return len(self._map)

    def known_roots(self):
        return set(self._map.values())

    def register(self, project_root: str, slug: Optional[str] = None) -> bool:
        """Record slug -> root. Returns True if the mapping changed."""
        if not project_root:
            return False
        slug = slug or self.slug_for(project_root)
        if self._map.get(slug) == project_root:
            return False
        self._map[slug] = project_root
        try:
            atomic_write_json(self.path, self._map, indent=2)
        except OSError as exc:
            logger.warning("Could not persist slug map: %s", exc)
        logger.info("Registered project %s -> %s", slug, project_root)
        return True
