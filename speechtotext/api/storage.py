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


def _validate_key(key: str) -> None:
    """Reject keys that would escape the backend's storage root.

    Used by every backend so that the call sites can rely on the same
    set of rejections regardless of which implementation is wired in.
    Allowed: POSIX-style relative paths with forward slashes. Rejected:
    empty, absolute (``/...``), backslash-containing, or any path with
    a ``..`` component.
    """
    if not key:
        raise ValueError("key cannot be empty")
    if key.startswith("/"):
        raise ValueError(f"key must be a relative path, not absolute: {key!r}")
    if "\\" in key:
        raise ValueError(f"key must use POSIX separators, not backslash: {key!r}")
    if ".." in key.split("/"):
        raise ValueError(f"key cannot contain '..': {key!r}")


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
        _validate_key(key)
        return self.root.joinpath(*Path(key).parts)

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


class S3Backend:
    """S3-compatible :class:`StorageBackend`.

    Works against any provider implementing the S3 API: AWS S3,
    Cloudflare R2, Backblaze B2, iDrive E2, Wasabi, MinIO, Garage, etc.
    Constructor takes the endpoint URL so a single class covers all of
    them; pass ``endpoint_url=None`` for AWS's default.

    boto3 is imported lazily so the rest of the API stack can import
    ``storage`` even when boto3 is not installed (it lives in the
    optional ``[api]`` extra).

    Conditional puts (``etag=`` argument) are implemented as a
    head-then-put pair under the hood. There is therefore a small
    race window where two writers can both see the same "current"
    state before either's put lands; for single-writer workloads or
    use cases where the loser's put is acceptable this is fine. AWS S3
    has true conditional writes via ``If-Match`` (added 2024) but
    boto3's exposed surface and other S3-compatibles' support is
    still uneven; revisit when standardised across providers.
    """

    def __init__(
        self,
        *,
        bucket: str,
        access_key: str,
        secret_key: str,
        region: str = "us-east-1",
        endpoint_url: str | None = None,
    ) -> None:
        try:
            import boto3  # type: ignore[import-not-found]
            from botocore.config import Config  # type: ignore[import-not-found]
            from botocore.exceptions import ClientError  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - dep guard
            raise StorageError(
                "S3Backend requires boto3 — install via `pip install '.[api]'`"
            ) from exc

        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=Config(signature_version="s3v4"),
        )
        self._ClientError = ClientError

    @staticmethod
    def _normalize_etag(raw: str) -> str:
        # S3 wraps etags in double quotes in HTTP headers; boto3 leaves
        # them in. Strip so etags compare equal across head/get/list/put.
        return raw.strip('"')

    def _error_code(self, exc: Exception) -> str:
        return exc.response.get("Error", {}).get("Code", "") if hasattr(exc, "response") else ""  # type: ignore[attr-defined]

    def _is_not_found(self, exc: Exception) -> bool:
        return self._error_code(exc) in ("NoSuchKey", "404", "NoSuchBucket")

    def _current_etag(self, key: str) -> str | None:
        try:
            resp = self._client.head_object(Bucket=self._bucket, Key=key)
        except self._ClientError as exc:
            if self._is_not_found(exc):
                return None
            raise StorageError(f"failed to head {key!r}: {exc}") from exc
        return self._normalize_etag(resp["ETag"])

    def get(self, key: str) -> bytes:
        _validate_key(key)
        try:
            resp = self._client.get_object(Bucket=self._bucket, Key=key)
        except self._ClientError as exc:
            if self._is_not_found(exc):
                raise NotFoundError(key) from exc
            raise StorageError(f"failed to get {key!r}: {exc}") from exc
        return resp["Body"].read()

    def put(self, key: str, data: bytes, *, etag: str | None = None) -> str:
        _validate_key(key)
        if etag is not None:
            current = self._current_etag(key)
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
        try:
            resp = self._client.put_object(
                Bucket=self._bucket, Key=key, Body=data
            )
        except self._ClientError as exc:
            raise StorageError(f"failed to put {key!r}: {exc}") from exc
        return self._normalize_etag(resp["ETag"])

    def delete(self, key: str) -> None:
        _validate_key(key)
        try:
            self._client.delete_object(Bucket=self._bucket, Key=key)
        except self._ClientError as exc:
            # S3 delete is idempotent for missing keys at the protocol
            # level, so we only surface other failures.
            if self._is_not_found(exc):
                return
            raise StorageError(f"failed to delete {key!r}: {exc}") from exc

    def list(self, prefix: str = "") -> Iterator[ObjectMeta]:
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                yield ObjectMeta(
                    key=obj["Key"],
                    size=int(obj["Size"]),
                    mtime=obj["LastModified"].timestamp(),
                    etag=self._normalize_etag(obj["ETag"]),
                )

    def head(self, key: str) -> ObjectMeta | None:
        _validate_key(key)
        try:
            resp = self._client.head_object(Bucket=self._bucket, Key=key)
        except self._ClientError as exc:
            if self._is_not_found(exc):
                return None
            raise StorageError(f"failed to head {key!r}: {exc}") from exc
        return ObjectMeta(
            key=key,
            size=int(resp["ContentLength"]),
            mtime=resp["LastModified"].timestamp(),
            etag=self._normalize_etag(resp["ETag"]),
        )

    def exists(self, key: str) -> bool:
        return self.head(key) is not None
