"""Tests for the pairing token store and pairing endpoints."""

from __future__ import annotations

import base64
import time

import pytest

from speechtotext.api.pairing import (
    PairingTokenStore,
    TOKEN_TTL_SECONDS,
    new_device_id,
)


# ── Token store ────────────────────────────────────────────────────────────


class TestPairingTokenStore:
    def test_mint_returns_unique_tokens(self) -> None:
        store = PairingTokenStore()
        a = store.mint()
        b = store.mint()
        assert a.token != b.token
        assert len(a.token) == 32  # 16 bytes hex

    def test_consume_returns_minted(self) -> None:
        store = PairingTokenStore()
        minted = store.mint()
        consumed = store.consume(minted.token)
        assert consumed.token == minted.token

    def test_consume_unknown_raises(self) -> None:
        store = PairingTokenStore()
        with pytest.raises(ValueError, match="unknown"):
            store.consume("not-a-real-token")

    def test_consume_twice_raises(self) -> None:
        store = PairingTokenStore()
        tok = store.mint()
        store.consume(tok.token)
        with pytest.raises(ValueError, match="consumed"):
            store.consume(tok.token)

    def test_expired_token_raises(self) -> None:
        store = PairingTokenStore()
        tok = store.mint()
        future = tok.created_at + TOKEN_TTL_SECONDS + 1
        with pytest.raises(ValueError, match="expired"):
            store.consume(tok.token, now=future)

    def test_expired_token_cannot_be_consumed_later(self) -> None:
        store = PairingTokenStore()
        tok = store.mint()
        future = tok.created_at + TOKEN_TTL_SECONDS + 1
        with pytest.raises(ValueError, match="expired"):
            store.consume(tok.token, now=future)
        # Even rolling the clock back, the token is now in 'consumed'.
        with pytest.raises(ValueError, match="consumed"):
            store.consume(tok.token, now=tok.created_at)

    def test_purge_expired_drops_old_tokens(self) -> None:
        store = PairingTokenStore()
        old = store.mint()
        fresh = store.mint()
        # Force the old token's clock backwards by manipulating internal
        # state — the public surface only lets time move forward.
        store._tokens[old.token] = type(old)(
            token=old.token,
            created_at=old.created_at - TOKEN_TTL_SECONDS - 10,
        )
        purged = store.purge_expired()
        assert purged == 1
        # Fresh token survives.
        assert store.consume(fresh.token).token == fresh.token

    def test_purge_drops_consumed_entries_after_2x_ttl(self) -> None:
        """_consumed must not grow forever — entries past replay window
        get forgotten so long-running hubs stay bounded."""
        store = PairingTokenStore()
        tok = store.mint()
        store.consume(tok.token)
        assert tok.token in store._consumed
        # Replay window = 2 * TTL. Past that, the token cannot legitimately
        # be re-submitted (it would already fail the TTL check), so we
        # can drop it from the consumed set.
        future = tok.created_at + 2 * TOKEN_TTL_SECONDS + 1
        store.purge_expired(now=future)
        assert tok.token not in store._consumed

    def test_consumed_token_still_rejected_within_replay_window(self) -> None:
        """Defensive check: consumed tokens stay flagged through 2*TTL."""
        store = PairingTokenStore()
        tok = store.mint()
        store.consume(tok.token)
        # 1.5 TTL later — purge should not have dropped this entry yet.
        future = tok.created_at + TOKEN_TTL_SECONDS + 1
        store.purge_expired(now=future)
        with pytest.raises(ValueError, match="consumed"):
            store.consume(tok.token, now=future)

    def test_mint_opportunistically_purges_stale_state(self) -> None:
        """Each mint sweeps the store so no background timer is needed."""
        store = PairingTokenStore()
        old = store.mint()
        # Backdate the old token so it's well past 2*TTL.
        store._tokens[old.token] = type(old)(
            token=old.token,
            created_at=old.created_at - 3 * TOKEN_TTL_SECONDS,
        )
        assert old.token in store._tokens
        # A fresh mint should sweep the backdated entry away.
        store.mint()
        assert old.token not in store._tokens

    def test_consumed_set_stays_bounded_across_many_mints(self) -> None:
        """100 mint+consume cycles followed by a clock-jump shouldn't
        leave 100 entries in _consumed."""
        store = PairingTokenStore()
        anchor = time.time()
        # Mint and consume 50 tokens at t0.
        first_tokens = []
        for _ in range(50):
            t = store.mint()
            store.consume(t.token)
            first_tokens.append(t.token)
        assert len(store._consumed) == 50
        # Jump clock past 2*TTL relative to those tokens and mint again
        # so purge runs.
        future = anchor + 2 * TOKEN_TTL_SECONDS + 1
        # Backdate the consumed timestamps to anchor.
        store._consumed = {t: anchor for t in store._consumed}
        # Mint with the future clock so purge sees them as stale.
        # We have to drive purge_expired explicitly here because mint()
        # uses time.time() and we can't easily monkeypatch from inside.
        store.purge_expired(now=future)
        assert len(store._consumed) == 0


def test_new_device_id_format() -> None:
    did = new_device_id()
    assert did.startswith("dev-")
    assert len(did) == len("dev-") + 12  # 6 bytes hex


# ── Endpoints (FastAPI TestClient) ─────────────────────────────────────────


pytest.importorskip("nacl")
from nacl.public import PrivateKey, SealedBox  # noqa: E402
from nacl.signing import SigningKey  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from speechtotext.api.app import create_app  # noqa: E402


@pytest.fixture
def app(tmp_path):
    return create_app(library_db_path=tmp_path / "library.db")


class TestMintTokenEndpoint:
    def test_returns_token_and_metadata(self, app) -> None:
        client = TestClient(app)
        r = client.post("/pair/tokens")
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["token"]) == 32
        assert body["expires_at"] > time.time()
        assert body["workspace_id"].startswith("ws_")
        assert body["ttl_seconds"] == TOKEN_TTL_SECONDS

    def test_two_mints_return_distinct_tokens(self, app) -> None:
        client = TestClient(app)
        a = client.post("/pair/tokens").json()["token"]
        b = client.post("/pair/tokens").json()["token"]
        assert a != b


class TestPairEndpoint:
    def _device_signing_key(self) -> SigningKey:
        return SigningKey.generate()

    def _b64(self, raw: bytes) -> str:
        return base64.b64encode(raw).decode("ascii")

    def test_pair_with_valid_token_succeeds(self, app) -> None:
        client = TestClient(app)
        token = client.post("/pair/tokens").json()["token"]

        signing_key = self._device_signing_key()
        verify_key_bytes = bytes(signing_key.verify_key)

        r = client.post(
            "/pair",
            json={
                "token": token,
                "device_pubkey_b64": self._b64(verify_key_bytes),
                "device_name": "test-iPad",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["device_id"].startswith("dev-")
        assert body["workspace_id"].startswith("ws_")
        assert body["lamport_observed"] >= 0

        # Verify the sealedbox actually decrypts on the device side.
        sealed = base64.b64decode(body["workspace_key_sealed_b64"])
        curve_private = signing_key.to_curve25519_private_key()
        opened = SealedBox(curve_private).decrypt(sealed)
        assert len(opened) == 32  # W is 32 bytes

    def test_token_is_single_use(self, app) -> None:
        client = TestClient(app)
        token = client.post("/pair/tokens").json()["token"]
        sk = self._device_signing_key()
        body = {
            "token": token,
            "device_pubkey_b64": self._b64(bytes(sk.verify_key)),
            "device_name": "device-A",
        }
        r1 = client.post("/pair", json=body)
        assert r1.status_code == 200
        # Same token, second device — must fail.
        sk2 = self._device_signing_key()
        body["device_pubkey_b64"] = self._b64(bytes(sk2.verify_key))
        body["device_name"] = "device-B"
        r2 = client.post("/pair", json=body)
        assert r2.status_code == 401

    def test_unknown_token_rejected(self, app) -> None:
        client = TestClient(app)
        sk = self._device_signing_key()
        r = client.post(
            "/pair",
            json={
                "token": "00" * 16,
                "device_pubkey_b64": self._b64(bytes(sk.verify_key)),
                "device_name": "test",
            },
        )
        assert r.status_code == 401

    def test_bad_pubkey_base64_rejected(self, app) -> None:
        client = TestClient(app)
        token = client.post("/pair/tokens").json()["token"]
        r = client.post(
            "/pair",
            json={
                "token": token,
                "device_pubkey_b64": "not---valid===base64!!",
                "device_name": "test",
            },
        )
        assert r.status_code == 400

    def test_short_pubkey_rejected(self, app) -> None:
        client = TestClient(app)
        token = client.post("/pair/tokens").json()["token"]
        r = client.post(
            "/pair",
            json={
                "token": token,
                "device_pubkey_b64": self._b64(b"\x00" * 16),
                "device_name": "test",
            },
        )
        assert r.status_code == 400
        assert "32 bytes" in r.text

    def test_empty_device_name_rejected(self, app) -> None:
        client = TestClient(app)
        token = client.post("/pair/tokens").json()["token"]
        sk = self._device_signing_key()
        r = client.post(
            "/pair",
            json={
                "token": token,
                "device_pubkey_b64": self._b64(bytes(sk.verify_key)),
                "device_name": "",
            },
        )
        assert r.status_code == 422  # pydantic validation

    def test_sealed_box_uses_workspace_key(self, app) -> None:
        """Sealing returns the same key that secrets_store returns on
        the hub side — sanity check that pairing isn't returning a
        random different key by mistake."""
        from speechtotext.api.secrets_store import get_workspace_key

        client = TestClient(app)
        token = client.post("/pair/tokens").json()["token"]
        sk = self._device_signing_key()
        r = client.post(
            "/pair",
            json={
                "token": token,
                "device_pubkey_b64": self._b64(bytes(sk.verify_key)),
                "device_name": "test",
            },
        )
        sealed = base64.b64decode(r.json()["workspace_key_sealed_b64"])
        curve_private = sk.to_curve25519_private_key()
        decrypted = SealedBox(curve_private).decrypt(sealed)
        assert decrypted == get_workspace_key()


# ── /devices/paired endpoint ─────────────────────────────────────────────


class TestListPairedDevices:
    def _b64(self, raw: bytes) -> str:
        return base64.b64encode(raw).decode("ascii")

    def test_empty_initially(self, app) -> None:
        client = TestClient(app)
        r = client.get("/devices/paired")
        assert r.status_code == 200
        assert r.json() == {"devices": []}

    def test_lists_paired_devices(self, app) -> None:
        client = TestClient(app)
        # Pair two devices.
        for name in ("ipad-A", "ipad-B"):
            token = client.post("/pair/tokens").json()["token"]
            sk = SigningKey.generate()
            client.post(
                "/pair",
                json={
                    "token": token,
                    "device_pubkey_b64": self._b64(bytes(sk.verify_key)),
                    "device_name": name,
                },
            )
        r = client.get("/devices/paired")
        assert r.status_code == 200
        body = r.json()
        names = {d["name"] for d in body["devices"]}
        assert names == {"ipad-A", "ipad-B"}

    def test_pubkey_not_exposed(self, app) -> None:
        """Pubkey must never appear in the listing — it's an internal
        verification detail, not user-facing data."""
        client = TestClient(app)
        token = client.post("/pair/tokens").json()["token"]
        sk = SigningKey.generate()
        client.post(
            "/pair",
            json={
                "token": token,
                "device_pubkey_b64": self._b64(bytes(sk.verify_key)),
                "device_name": "iPad",
            },
        )
        r = client.get("/devices/paired")
        body = r.json()
        for dev in body["devices"]:
            assert "pubkey_b64" not in dev
            assert "pubkey" not in dev

    def test_response_shape(self, app) -> None:
        client = TestClient(app)
        token = client.post("/pair/tokens").json()["token"]
        sk = SigningKey.generate()
        client.post(
            "/pair",
            json={
                "token": token,
                "device_pubkey_b64": self._b64(bytes(sk.verify_key)),
                "device_name": "device-name-X",
            },
        )
        r = client.get("/devices/paired")
        dev = r.json()["devices"][0]
        assert set(dev.keys()) == {
            "device_id", "name", "paired_at", "last_seen"
        }
        assert dev["device_id"].startswith("dev-")
        assert dev["name"] == "device-name-X"
        assert dev["last_seen"] is None  # never seen until a signed call


class TestUnpairDevice:
    def _b64(self, raw: bytes) -> str:
        return base64.b64encode(raw).decode("ascii")

    def _pair(self, client: TestClient, name: str) -> str:
        token = client.post("/pair/tokens").json()["token"]
        sk = SigningKey.generate()
        r = client.post(
            "/pair",
            json={
                "token": token,
                "device_pubkey_b64": self._b64(bytes(sk.verify_key)),
                "device_name": name,
            },
        )
        return r.json()["device_id"]

    def test_unpair_removes_device(self, app) -> None:
        client = TestClient(app)
        dev_id = self._pair(client, "ipad-A")
        r = client.delete(f"/devices/paired/{dev_id}")
        assert r.status_code == 200
        assert r.json() == {"ok": True}
        listing = client.get("/devices/paired").json()["devices"]
        assert all(d["device_id"] != dev_id for d in listing)

    def test_unpair_unknown_is_noop(self, app) -> None:
        client = TestClient(app)
        r = client.delete("/devices/paired/dev-nonexistent")
        assert r.status_code == 200
        assert r.json() == {"ok": True}

    def test_unpair_only_targets_one(self, app) -> None:
        client = TestClient(app)
        keep = self._pair(client, "ipad-keep")
        drop = self._pair(client, "ipad-drop")
        client.delete(f"/devices/paired/{drop}")
        listing = client.get("/devices/paired").json()["devices"]
        ids = {d["device_id"] for d in listing}
        assert keep in ids and drop not in ids
