"""Storage backend abstraction for transcript data.

The hub reads and writes transcript artifacts (`.json` and `.txt` files
produced by `speechtotext.writer`, plus the original audio files) through
this interface. Two implementations ship:

- ``FsBackend`` — local filesystem. Default for single-machine desktop use;
  preserves the historical layout where transcript sidecars sit next to the
  source audio file.
- ``S3Backend`` — S3-compatible object storage (AWS S3, Cloudflare R2,
  Backblaze B2, self-hosted MinIO / Garage). Required for multi-device sync.

The interface is intentionally small so future backends (WebDAV, etc.) can
be added with minimal API surface. All methods address objects by *key*: an
opaque string that the backend interprets — for ``FsBackend`` the key maps
to a path under the configured root; for ``S3Backend`` the key is the S3
object key under a configured bucket.

Concurrency model: callers may write the same key from multiple processes
or devices. Implementations expose optimistic concurrency via ETags. A
caller that wishes to avoid overwriting a concurrent change passes the
``etag`` it last read; if the stored ETag has moved, ``put`` raises
``ConcurrencyError`` instead of overwriting.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable


class StorageError(Exception):
    """Base for storage-layer errors."""


class NotFoundError(StorageError):
    """Raised when a key does not exist."""


class ConcurrencyError(StorageError):
    """Raised when an etag-conditioned put loses the race."""


@dataclass(frozen=True)
class ObjectMeta:
    """Metadata for a stored object.

    Attributes:
        key: The opaque object identifier (path-like for FsBackend,
            S3 key for S3Backend).
        size: Object size in bytes.
        mtime: Last-modified timestamp as a Unix epoch float. For
            backends that don't preserve sub-second precision, this is
            still expressed in seconds.
        etag: Opaque entity tag. Stable while the object's content
            does not change. Two reads of the same unchanged object
            return equal etags; any modification produces a different
            etag. The exact algorithm is backend-defined and must not
            be parsed by callers.
    """

    key: str
    size: int
    mtime: float
    etag: str


@runtime_checkable
class StorageBackend(Protocol):
    """Abstract storage backend for transcript artifacts.

    Implementations must be safe to call from multiple threads. Long-lived
    instances may cache connections or sessions; ``close()`` (if present
    on the implementation) releases them.
    """

    def get(self, key: str) -> bytes:
        """Read the bytes stored at ``key``.

        Raises:
            NotFoundError: if no object exists at ``key``.
            StorageError: for backend-specific failures.
        """
        ...

    def put(self, key: str, data: bytes, *, etag: str | None = None) -> str:
        """Write ``data`` to ``key`` and return the new etag.

        Args:
            key: Object key to write.
            data: Raw bytes payload.
            etag: If provided, the put is conditional: the existing
                object's etag must match this value, otherwise
                ``ConcurrencyError`` is raised. To require that the
                object *not* exist, pass the sentinel value ``""``
                (the empty string, exposed as ``ETAG_ABSENT``).

        Returns:
            The etag of the newly stored object. Pass this back to
            subsequent conditional ``put`` calls to chain updates.

        Raises:
            ConcurrencyError: if an ``etag`` was supplied and does not
                match the current state.
            StorageError: for backend-specific failures.
        """
        ...

    def delete(self, key: str) -> None:
        """Remove the object at ``key``.

        Deleting a missing key is a no-op (idempotent). Backend
        failures other than "not found" raise ``StorageError``.
        """
        ...

    def list(self, prefix: str = "") -> Iterator[ObjectMeta]:
        """Iterate objects whose key begins with ``prefix``.

        Order is unspecified. Implementations stream results
        lazily where the backend permits.
        """
        ...

    def head(self, key: str) -> ObjectMeta | None:
        """Return metadata for ``key`` or ``None`` if missing.

        Cheaper than ``get`` because it does not transfer the body.
        """
        ...

    def exists(self, key: str) -> bool:
        """Return ``True`` iff an object exists at ``key``."""
        ...


# Sentinel used as the ``etag`` argument to ``put`` to require that the
# target key does not currently exist. Spelled as a constant so call sites
# read clearly and type-checkers don't flag a bare empty string.
ETAG_ABSENT = ""


class FsBackend:
    """Filesystem-backed :class:`StorageBackend`.

    Keys are interpreted as POSIX-style paths relative to ``root``. Writes
    are atomic via tmp-then-rename. ETags are derived from ``(st_mtime_ns,
    st_size)`` so unchanged files yield stable etags across reads.

    Thread-safety is at the filesystem-syscall level: ``os.replace`` and
    ``unlink`` are atomic, so concurrent readers will never observe a
    partially-written file. Conditional puts (``etag=`` argument) check
    the current state then write; concurrent writers may both see the same
    "current" state before either's rename lands, in which case the second
    rename wins. For strict serializability across writers, use a single
    writer process or switch to :class:`S3Backend`, which has true
    conditional writes.
    """

    _TMP_SUFFIX = ".tmp"

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _key_to_path(self, key: str) -> Path:
        if not key:
            raise ValueError("key cannot be empty")
        if key.startswith("/"):
            raise ValueError(f"key must be a relative path, not absolute: {key!r}")
        if "\\" in key:
            raise ValueError(f"key must use POSIX separators, not backslash: {key!r}")
        parts = Path(key).parts
        if ".." in parts:
            raise ValueError(f"key cannot contain '..': {key!r}")
        return self.root.joinpath(*parts)

    @staticmethod
    def _etag_from_stat(stat: os.stat_result) -> str:
        # mtime_ns + size is collision-free in practice for a single-writer
        # workflow; two writes of equal size in the same nanosecond are
        # extraordinarily unlikely. If we ever need cryptographic stability
        # across machines, this can be swapped for a content hash without
        # affecting the public interface.
        return f"{stat.st_mtime_ns}-{stat.st_size}"

    def _current_etag(self, path: Path) -> str | None:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        return self._etag_from_stat(stat)

    def get(self, key: str) -> bytes:
        path = self._key_to_path(key)
        try:
            return path.read_bytes()
        except FileNotFoundError as exc:
            raise NotFoundError(key) from exc
        except OSError as exc:
            raise StorageError(f"failed to read {key!r}: {exc}") from exc

    def put(self, key: str, data: bytes, *, etag: str | None = None) -> str:
        path = self._key_to_path(key)
        if etag is not None:
            current = self._current_etag(path)
            if etag == ETAG_ABSENT:
                if current is not None:
                    raise ConcurrencyError(
                        f"object already exists at {key!r} (etag={current!r})"
                    )
            else:
                if current != etag:
                    raise ConcurrencyError(
                        f"etag mismatch at {key!r}: have {current!r}, "
                        f"expected {etag!r}"
                    )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + self._TMP_SUFFIX)
        try:
            tmp.write_bytes(data)
            os.replace(tmp, path)
        except OSError as exc:
            # Clean up an orphaned tmp file so list() / disk usage stay clean.
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
            raise StorageError(f"failed to write {key!r}: {exc}") from exc
        return self._etag_from_stat(path.stat())

    def delete(self, key: str) -> None:
        path = self._key_to_path(key)
        try:
            path.unlink()
        except FileNotFoundError:
            return  # idempotent: deleting a missing key is fine
        except OSError as exc:
            raise StorageError(f"failed to delete {key!r}: {exc}") from exc

    def list(self, prefix: str = "") -> Iterator[ObjectMeta]:
        if not self.root.exists():
            return
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            if path.name.endswith(self._TMP_SUFFIX):
                # Skip in-flight atomic-write tmp files.
                continue
            rel = path.relative_to(self.root).as_posix()
            if not rel.startswith(prefix):
                continue
            try:
                stat = path.stat()
            except FileNotFoundError:
                # Raced with a concurrent delete; just skip.
                continue
            yield ObjectMeta(
                key=rel,
                size=stat.st_size,
                mtime=stat.st_mtime,
                etag=self._etag_from_stat(stat),
            )

    def head(self, key: str) -> ObjectMeta | None:
        path = self._key_to_path(key)
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        if not path.is_file():
            return None
        return ObjectMeta(
            key=key,
            size=stat.st_size,
            mtime=stat.st_mtime,
            etag=self._etag_from_stat(stat),
        )

    def exists(self, key: str) -> bool:
        path = self._key_to_path(key)
        return path.is_file()
