"""Tests for ``speechtotext.api.storage.S3Backend`` using moto's S3 mock.

These tests skip if ``moto`` or ``boto3`` is unavailable, so the wider
test suite still runs in environments without the optional ``[dev]`` /
``[api]`` deps installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")

from moto import mock_aws  # noqa: E402

from speechtotext.api.storage import (  # noqa: E402
    ConcurrencyError,
    ETAG_ABSENT,
    NotFoundError,
    ObjectMeta,
    S3Backend,
    StorageBackend,
)

if TYPE_CHECKING:
    pass


_BUCKET = "locallexis-test-bucket"


@pytest.fixture
def s3_backend():
    """Yield a S3Backend wired to a moto-mocked S3 region.

    The mock starts inside the fixture so each test gets a fresh,
    isolated bucket. Boto3 is constructed inside the mock context so
    the client picks up moto's intercept.
    """
    with mock_aws():
        # Create the bucket up-front so the backend can use it.
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=_BUCKET)
        yield S3Backend(
            bucket=_BUCKET,
            access_key="test-access",
            secret_key="test-secret",
            region="us-east-1",
            endpoint_url=None,
        )


# ── Protocol ────────────────────────────────────────────────────────────────


class TestProtocol:
    def test_s3_backend_satisfies_protocol(self, s3_backend: S3Backend) -> None:
        assert isinstance(s3_backend, StorageBackend)


# ── Put / Get ───────────────────────────────────────────────────────────────


class TestPutGet:
    def test_roundtrip(self, s3_backend: S3Backend) -> None:
        etag = s3_backend.put("a.json", b"hello")
        assert etag
        assert s3_backend.get("a.json") == b"hello"

    def test_get_missing_raises_notfound(self, s3_backend: S3Backend) -> None:
        with pytest.raises(NotFoundError):
            s3_backend.get("missing.json")

    def test_put_overwrites(self, s3_backend: S3Backend) -> None:
        s3_backend.put("a.json", b"v1")
        s3_backend.put("a.json", b"v2")
        assert s3_backend.get("a.json") == b"v2"

    def test_nested_key(self, s3_backend: S3Backend) -> None:
        s3_backend.put("nested/deep/key.json", b"x")
        assert s3_backend.get("nested/deep/key.json") == b"x"

    def test_unicode_payload(self, s3_backend: S3Backend) -> None:
        payload = '{"text": "hoi — café"}'.encode("utf-8")
        s3_backend.put("a.json", payload)
        assert s3_backend.get("a.json") == payload


# ── ETag semantics ──────────────────────────────────────────────────────────


class TestEtag:
    def test_put_returns_etag(self, s3_backend: S3Backend) -> None:
        etag = s3_backend.put("a", b"v1")
        assert isinstance(etag, str)
        assert etag
        # S3 etags are MD5 hex (32 chars) for non-multipart uploads;
        # we strip the surrounding quotes when normalizing.
        assert '"' not in etag

    def test_etag_changes_on_modify(self, s3_backend: S3Backend) -> None:
        etag1 = s3_backend.put("a", b"v1")
        etag2 = s3_backend.put("a", b"v2-different-content")
        assert etag1 != etag2

    def test_head_etag_matches_put(self, s3_backend: S3Backend) -> None:
        etag = s3_backend.put("a", b"v")
        head = s3_backend.head("a")
        assert head is not None
        assert head.etag == etag

    def test_list_etag_matches_put(self, s3_backend: S3Backend) -> None:
        etag = s3_backend.put("a", b"v")
        items = list(s3_backend.list())
        assert len(items) == 1
        assert items[0].etag == etag

    def test_put_with_etag_absent_when_missing_succeeds(
        self, s3_backend: S3Backend
    ) -> None:
        s3_backend.put("a", b"v1", etag=ETAG_ABSENT)
        assert s3_backend.get("a") == b"v1"

    def test_put_with_etag_absent_when_present_raises(
        self, s3_backend: S3Backend
    ) -> None:
        s3_backend.put("a", b"v1")
        with pytest.raises(ConcurrencyError):
            s3_backend.put("a", b"v2", etag=ETAG_ABSENT)

    def test_put_with_matching_etag_succeeds(self, s3_backend: S3Backend) -> None:
        etag = s3_backend.put("a", b"v1")
        s3_backend.put("a", b"v2", etag=etag)
        assert s3_backend.get("a") == b"v2"

    def test_put_with_stale_etag_raises(self, s3_backend: S3Backend) -> None:
        s3_backend.put("a", b"v1")
        with pytest.raises(ConcurrencyError):
            s3_backend.put("a", b"v2", etag="not-the-current-etag")


# ── Delete ──────────────────────────────────────────────────────────────────


class TestDelete:
    def test_delete_existing(self, s3_backend: S3Backend) -> None:
        s3_backend.put("a", b"v")
        s3_backend.delete("a")
        assert not s3_backend.exists("a")

    def test_delete_missing_is_noop(self, s3_backend: S3Backend) -> None:
        # S3 delete is idempotent for missing keys.
        s3_backend.delete("never-existed")


# ── List ────────────────────────────────────────────────────────────────────


class TestList:
    def test_list_empty_bucket(self, s3_backend: S3Backend) -> None:
        assert list(s3_backend.list()) == []

    def test_list_returns_keys(self, s3_backend: S3Backend) -> None:
        s3_backend.put("a.json", b"x")
        s3_backend.put("b.json", b"y")
        keys = {m.key for m in s3_backend.list()}
        assert keys == {"a.json", "b.json"}

    def test_list_filters_by_prefix(self, s3_backend: S3Backend) -> None:
        s3_backend.put("transcripts/a.json", b"x")
        s3_backend.put("transcripts/b.json", b"y")
        s3_backend.put("audio/c.mp3", b"z")
        keys = {m.key for m in s3_backend.list("transcripts/")}
        assert keys == {"transcripts/a.json", "transcripts/b.json"}

    def test_list_meta_fields(self, s3_backend: S3Backend) -> None:
        s3_backend.put("a.json", b"hello")
        items = list(s3_backend.list())
        assert len(items) == 1
        m = items[0]
        assert isinstance(m, ObjectMeta)
        assert m.key == "a.json"
        assert m.size == 5
        assert m.etag
        assert m.mtime > 0

    def test_list_paginates(self, s3_backend: S3Backend) -> None:
        # Force more than one page from the listing paginator (default
        # page size is 1000; we put 5 and expect them all back).
        for i in range(5):
            s3_backend.put(f"key-{i}.json", str(i).encode())
        items = sorted(s3_backend.list(), key=lambda m: m.key)
        assert len(items) == 5
        assert [m.key for m in items] == [f"key-{i}.json" for i in range(5)]


# ── Head / Exists ───────────────────────────────────────────────────────────


class TestHeadExists:
    def test_head_existing(self, s3_backend: S3Backend) -> None:
        s3_backend.put("a.json", b"hello")
        head = s3_backend.head("a.json")
        assert head is not None
        assert head.key == "a.json"
        assert head.size == 5

    def test_head_missing(self, s3_backend: S3Backend) -> None:
        assert s3_backend.head("missing") is None

    def test_exists_true(self, s3_backend: S3Backend) -> None:
        s3_backend.put("a", b"x")
        assert s3_backend.exists("a")

    def test_exists_false(self, s3_backend: S3Backend) -> None:
        assert not s3_backend.exists("never")


# ── Key validation (delegated to module-level _validate_key) ────────────────


class TestKeyValidation:
    def test_empty_key_rejected(self, s3_backend: S3Backend) -> None:
        with pytest.raises(ValueError):
            s3_backend.put("", b"x")

    def test_absolute_key_rejected(self, s3_backend: S3Backend) -> None:
        with pytest.raises(ValueError):
            s3_backend.put("/etc/passwd", b"x")

    def test_dotdot_rejected(self, s3_backend: S3Backend) -> None:
        with pytest.raises(ValueError):
            s3_backend.put("foo/../bar", b"x")
