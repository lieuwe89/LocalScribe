"""Cheap gate in front of the library disk walk.

``LibraryDB.sync_dirs`` globs ``*.json`` in each library dir (non-recursive)
and ``stat()``s every file. Running that on *every* ``GET /transcripts`` and
``/sync/*`` request is wasteful — the UI search bar queries per keystroke and
devices poll the sync endpoints on an interval.

A directory's mtime changes whenever an entry is added, removed, or renamed
within it. All transcript writes go through a temp file + atomic rename
(``writer._atomic_write`` and the route handlers), which is exactly such an
entry change. So an *unchanged* directory mtime means an unchanged ``*.json``
set, and we can skip the walk entirely with no staleness — only stat the
directories, not their contents.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterable, Protocol


class _SyncableDB(Protocol):
    def sync_dirs(self, dirs: Iterable[Path]) -> dict: ...


class LibraryReconciler:
    def __init__(self, db: _SyncableDB) -> None:
        self._db = db
        self._lock = threading.Lock()
        self._seen: dict[str, float] = {}

    def reconcile(self, dirs: Iterable[Path]) -> None:
        """Run ``sync_dirs`` only if a directory's mtime changed.

        Synchronous when a change is detected (the caller needs the fresh
        result), a handful of cheap ``stat`` calls otherwise.
        """
        paths = [Path(d) for d in dirs]
        with self._lock:
            current: dict[str, float] = {}
            changed = False
            for d in paths:
                try:
                    mtime = d.stat().st_mtime
                except OSError:
                    # Unstattable (missing/permission) — skip; if it was seen
                    # before, the set-difference check below flags the change.
                    continue
                key = str(d)
                current[key] = mtime
                if self._seen.get(key) != mtime:
                    changed = True
            # A directory that disappeared (seen before, unstattable now) is
            # also a change: sync_dirs prunes its rows.
            if not changed and set(current) != set(self._seen):
                changed = True
            if not changed:
                return
            self._db.sync_dirs(paths)
            self._seen = current
