"""Replay-protection + query-signing tests for device-signed requests."""

from __future__ import annotations

import base64
import time

import pytest
from fastapi.testclient import TestClient
from nacl.signing import SigningKey

from speechtotext.api.app import create_app
from tests.api._signing import signed_headers


@pytest.fixture
def app(tmp_path):
    app = create_app(library_db_path=tmp_path / "library.db")
    app.state.library_dirs.add(tmp_path)
    return app


def _pair(client: TestClient):
    token = client.post("/pair/tokens").json()["token"]
    sk = SigningKey.generate()
    r = client.post(
        "/pair",
        json={
            "token": token,
            "device_pubkey_b64": base64.b64encode(bytes(sk.verify_key)).decode("ascii"),
            "device_name": "d",
        },
    )
    return sk, r.json()["device_id"]


def test_valid_signed_request_succeeds(app):
    c = TestClient(app)
    sk, dev = _pair(c)
    h = signed_headers(sk, dev, "GET", "/sync/snapshot")
    assert c.get("/sync/snapshot", headers=h).status_code == 200


def test_replayed_nonce_rejected(app):
    c = TestClient(app)
    sk, dev = _pair(c)
    h = signed_headers(sk, dev, "GET", "/sync/snapshot")
    assert c.get("/sync/snapshot", headers=h).status_code == 200
    # Identical headers → identical (device, nonce) → replay.
    r = c.get("/sync/snapshot", headers=h)
    assert r.status_code == 401
    assert "replay" in r.text.lower()


def test_stale_timestamp_rejected(app):
    c = TestClient(app)
    sk, dev = _pair(c)
    h = signed_headers(
        sk, dev, "GET", "/sync/snapshot", timestamp=int(time.time()) - 100_000
    )
    assert c.get("/sync/snapshot", headers=h).status_code == 401


def test_missing_timestamp_rejected(app):
    c = TestClient(app)
    sk, dev = _pair(c)
    h = signed_headers(sk, dev, "GET", "/sync/snapshot")
    del h["X-Timestamp"]
    assert c.get("/sync/snapshot", headers=h).status_code == 401


def test_missing_nonce_rejected(app):
    c = TestClient(app)
    sk, dev = _pair(c)
    h = signed_headers(sk, dev, "GET", "/sync/snapshot")
    del h["X-Nonce"]
    assert c.get("/sync/snapshot", headers=h).status_code == 401


def test_query_string_is_signed(app):
    c = TestClient(app)
    sk, dev = _pair(c)
    # Sign for limit=1 but send limit=999 → signature must not verify.
    h = signed_headers(sk, dev, "GET", "/sync/snapshot?limit=1")
    assert c.get("/sync/snapshot?limit=999", headers=h).status_code == 401
