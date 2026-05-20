"""Tests for ``speechtotext.api.storage`` backends."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from speechtotext.api.storage import (
    ConcurrencyError,
    ETAG_ABSENT,
    FsBackend,
    NotFoundError,
    ObjectMeta,
    StorageBackend,
    StorageError,
)


@pytest.fixture
def backend(tmp_path: Path) -> FsBackend:
    return FsBackend(tmp_path / "library")


# ── Protocol ────────────────────────────────────────────────────────────────


class TestProtocol:
    def test_fs_backend_satisfies_protocol(self, backend: FsBackend) -> None:
        assert isinstance(backend, StorageBackend)


# ── Put / Get ───────────────────────────────────────────────────────────────


class TestPutGet:
    def test_roundtrip(self, backend: FsBackend) -> None:
        etag = backend.put("a.json", b"hello")
        assert etag
        assert backend.get("a.json") == b"hello"

    def test_get_missing_raises_notfound(self, backend: FsBackend) -> None:
        with pytest.raises(NotFoundError):
            backend.get("missing.json")

    def test_put_overwrites(self, backend: FsBackend) -> None:
        backend.put("a.json", b"v1")
        backend.put("a.json", b"v2")
        assert backend.get("a.json") == b"v2"

    def test_nested_key_creates_directories(self, backend: FsBackend) -> None:
        backend.put("nested/deep/key.json", b"x")
        assert backend.get("nested/deep/key.json") == b"x"

    def test_unicode_payload_roundtrip(self, backend: FsBackend) -> None:
        # Transcript JSON contains Dutch / non-ASCII regularly.
        payload = '{"text": "hoi — café"}'.encode("utf-8")
        backend.put("a.json", payload)
        assert backend.get("a.json") == payload


# ── ETag semantics ──────────────────────────────────────────────────────────


class TestEtag:
    def test_put_returns_etag(self, backend: FsBackend) -> None:
        etag = backend.put("a", b"v1")
        assert isinstance(etag, str)
        assert etag

    def test_etag_changes_on_modify(self, backend: FsBackend) -> None:
        etag1 = backend.put("a", b"v1")
        # Filesystem mtime resolution varies; force a measurable gap.
        time.sleep(0.02)
        etag2 = backend.put("a", b"v2-longer")
        assert etag1 != etag2

    def test_head_etag_matches_put(self, backend: FsBackend) -> None:
        etag = backend.put("a", b"v")
        head = backend.head("a")
        assert head is not None
        assert head.etag == etag

    def test_put_with_etag_absent_when_missing_succeeds(
        self, backend: FsBackend
    ) -> None:
        backend.put("a", b"v1", etag=ETAG_ABSENT)
        assert backend.get("a") == b"v1"

    def test_put_with_etag_absent_when_present_raises(
        self, backend: FsBackend
    ) -> None:
        backend.put("a", b"v1")
        with pytest.raises(ConcurrencyError):
            backend.put("a", b"v2", etag=ETAG_ABSENT)

    def test_put_with_matching_etag_succeeds(self, backend: FsBackend) -> None:
        etag = backend.put("a", b"v1")
        backend.put("a", b"v2", etag=etag)
        assert backend.get("a") == b"v2"

    def test_put_with_stale_etag_raises(self, backend: FsBackend) -> None:
        backend.put("a", b"v1")
        with pytest.raises(ConcurrencyError):
            backend.put("a", b"v2", etag="not-the-current-etag")


# ── Delete ──────────────────────────────────────────────────────────────────


class TestDelete:
    def test_delete_existing(self, backend: FsBackend) -> None:
        backend.put("a", b"v")
        backend.delete("a")
        assert not backend.exists("a")

    def test_delete_missing_is_noop(self, backend: FsBackend) -> None:
        # Idempotent — no exception expected.
        backend.delete("never-existed")


# ── List ────────────────────────────────────────────────────────────────────


class TestList:
    def test_list_empty_root(self, backend: FsBackend) -> None:
        assert list(backend.list()) == []

    def test_list_returns_files(self, backend: FsBackend) -> None:
        backend.put("a.json", b"x")
        backend.put("b.json", b"y")
        keys = {m.key for m in backend.list()}
        assert keys == {"a.json", "b.json"}

    def test_list_filters_by_prefix(self, backend: FsBackend) -> None:
        backend.put("transcripts/a.json", b"x")
        backend.put("transcripts/b.json", b"y")
        backend.put("audio/c.mp3", b"z")
        keys = {m.key for m in backend.list("transcripts/")}
        assert keys == {"transcripts/a.json", "transcripts/b.json"}

    def test_list_skips_tmp_files(
        self, backend: FsBackend, tmp_path: Path
    ) -> None:
        # Simulate an aborted atomic write leaving a .tmp behind.
        backend.put("a.json", b"x")
        stray_tmp = tmp_path / "library" / "leftover.json.tmp"
        stray_tmp.write_bytes(b"junk")
        keys = {m.key for m in backend.list()}
        assert "leftover.json.tmp" not in keys
        assert "a.json" in keys

    def test_list_meta_fields(self, backend: FsBackend) -> None:
        backend.put("a.json", b"hello")
        items = list(backend.list())
        assert len(items) == 1
        m = items[0]
        assert isinstance(m, ObjectMeta)
        assert m.key == "a.json"
        assert m.size == 5
        assert m.etag
        assert m.mtime > 0


# ── Head / Exists ───────────────────────────────────────────────────────────


class TestHeadExists:
    def test_head_existing(self, backend: FsBackend) -> None:
        backend.put("a.json", b"hello")
        head = backend.head("a.json")
        assert head is not None
        assert head.key == "a.json"
        assert head.size == 5

    def test_head_missing(self, backend: FsBackend) -> None:
        assert backend.head("missing") is None

    def test_exists_true(self, backend: FsBackend) -> None:
        backend.put("a", b"x")
        assert backend.exists("a")

    def test_exists_false_for_missing(self, backend: FsBackend) -> None:
        assert not backend.exists("never")


# ── Atomicity ───────────────────────────────────────────────────────────────


class TestAtomicity:
    def test_no_tmp_file_after_successful_put(
        self, backend: FsBackend, tmp_path: Path
    ) -> None:
        backend.put("a.json", b"hello")
        leftovers = list((tmp_path / "library").rglob("*.tmp"))
        assert leftovers == []

    def test_tmp_file_cleaned_on_replace_failure(
        self, backend: FsBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(src: object, dst: object) -> None:  # type: ignore[override]
            raise OSError("disk full simulation")

        # Patch the live module's `os` binding directly. Using a string
        # path here would walk `speechtotext.api.storage` via parent
        # attribute lookup, which is unreliable across tests that wipe
        # sys.modules (see api/conftest.py for context).
        import speechtotext.api.storage as _storage
        monkeypatch.setattr(_storage.os, "replace", boom)
        with pytest.raises(StorageError):
            backend.put("a.json", b"hello")
        # No orphaned tmp file should remain.
        leftovers = list((tmp_path / "library").rglob("*.tmp"))
        assert leftovers == []


# ── Key validation ──────────────────────────────────────────────────────────


class TestKeyValidation:
    def test_empty_key_rejected(self, backend: FsBackend) -> None:
        with pytest.raises(ValueError):
            backend.put("", b"x")

    def test_absolute_key_rejected(self, backend: FsBackend) -> None:
        with pytest.raises(ValueError):
            backend.put("/etc/passwd", b"x")

    def test_dotdot_key_rejected(self, backend: FsBackend) -> None:
        with pytest.raises(ValueError):
            backend.put("../escape", b"x")

    def test_backslash_key_rejected(self, backend: FsBackend) -> None:
        with pytest.raises(ValueError):
            backend.put("foo\\bar", b"x")

    def test_dotdot_nested_rejected(self, backend: FsBackend) -> None:
        with pytest.raises(ValueError):
            backend.put("foo/../bar", b"x")

    def test_invalid_key_on_get(self, backend: FsBackend) -> None:
        with pytest.raises(ValueError):
            backend.get("")
