"""Tests for the directory-mtime-gated library reconciler."""

from __future__ import annotations

import time
from pathlib import Path

from speechtotext.api.reconcile import LibraryReconciler


class FakeDB:
    def __init__(self) -> None:
        self.calls = 0

    def sync_dirs(self, dirs) -> None:
        self.calls += 1


def test_first_touch_syncs(tmp_path: Path) -> None:
    db = FakeDB()
    rec = LibraryReconciler(db)
    rec.reconcile([tmp_path])
    assert db.calls == 1


def test_unchanged_dir_skips_walk(tmp_path: Path) -> None:
    # The expensive walk must not run when the directory hasn't changed —
    # this is the search-as-you-type / poll storm the fix targets.
    db = FakeDB()
    rec = LibraryReconciler(db)
    rec.reconcile([tmp_path])
    rec.reconcile([tmp_path])
    assert db.calls == 1


def test_new_file_triggers_sync(tmp_path: Path) -> None:
    db = FakeDB()
    rec = LibraryReconciler(db)
    rec.reconcile([tmp_path])
    time.sleep(0.02)
    (tmp_path / "x.json").write_text("{}")  # adding an entry bumps dir mtime
    rec.reconcile([tmp_path])
    assert db.calls == 2


def test_new_dir_in_set_triggers_sync(tmp_path: Path) -> None:
    db = FakeDB()
    rec = LibraryReconciler(db)
    rec.reconcile([tmp_path])
    sub = tmp_path / "sub"
    sub.mkdir()
    rec.reconcile([tmp_path, sub])
    assert db.calls == 2
