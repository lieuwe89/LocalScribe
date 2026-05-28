import tomllib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from speechtotext.api.app import create_app


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    cfg_path = tmp_path / "config.toml"
    monkeypatch.setattr("speechtotext.api.routes_config.DEFAULT_CONFIG_PATH", cfg_path)
    app = create_app()
    return TestClient(app), cfg_path


def test_config_get_returns_defaults_and_hf_flag(client):
    c, _ = client
    r = c.get("/config")
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "auto"
    assert body["hf_token_set"] is False


def test_config_patch_writes_toml(client):
    c, cfg_path = client
    r = c.patch("/config", json={"asr_model": "small", "hf_token": "hf_xxx"})
    assert r.status_code == 200
    raw = tomllib.loads(cfg_path.read_text())
    assert raw["asr_model"] == "small"
    assert raw["hf_token"] == "hf_xxx"


def test_config_patch_rejects_unknown_top_level_key(client):
    c, _ = client
    r = c.patch("/config", json={"asr_model": "small", "totally_bogus": "yes"})
    assert r.status_code == 422


def test_config_patch_rejects_unknown_watch_key(client):
    c, _ = client
    r = c.patch("/config", json={"watch": {"recursive": True, "stranger": 1}})
    assert r.status_code == 422


def test_config_patch_rejects_float_debounce(client):
    # debounce_seconds is integer seconds; floats should be rejected, not
    # silently truncated when load_config calls int() on them later.
    c, _ = client
    r = c.patch("/config", json={"watch": {"debounce_seconds": 2.5}})
    assert r.status_code == 422


def test_config_patch_rejects_invalid_backend(client):
    c, _ = client
    r = c.patch("/config", json={"backend": "tpu"})
    assert r.status_code == 422


def test_config_patch_rejects_invalid_extension(client):
    c, _ = client
    r = c.patch("/config", json={"watch": {"extensions": ["mp3", "../etc"]}})
    assert r.status_code == 422


def test_config_patch_escapes_quotes_and_newlines(client):
    # Hostile values should round-trip safely through TOML — no broken
    # files, no injection of new TOML keys via embedded newlines.
    c, cfg_path = client
    payload = 'hf_"abc"\nbackend = "cuda"\n'
    r = c.patch("/config", json={"hf_token": payload})
    assert r.status_code == 200
    raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    assert raw["hf_token"] == payload
    # Backend was never set; the embedded "backend = cuda" must not leak.
    assert raw.get("backend") != "cuda"


def test_config_patch_partial_watch_preserves_siblings(client):
    c, cfg_path = client
    r1 = c.patch(
        "/config",
        json={"watch": {"recursive": True, "extensions": ["mp3", "wav"]}},
    )
    assert r1.status_code == 200
    r2 = c.patch("/config", json={"watch": {"debounce_seconds": 7}})
    assert r2.status_code == 200
    raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    assert raw["watch"]["debounce_seconds"] == 7
    assert raw["watch"]["recursive"] is True
    assert raw["watch"]["extensions"] == ["mp3", "wav"]


def test_config_patch_writes_atomically(client):
    # The temp file must not survive on a successful write; it must be
    # renamed onto the target path, leaving exactly one config file.
    c, cfg_path = client
    r = c.patch("/config", json={"asr_model": "small"})
    assert r.status_code == 200
    siblings = list(cfg_path.parent.iterdir())
    assert siblings == [cfg_path]


def test_concurrent_config_patches_dont_lose_updates(client, monkeypatch):
    # Two clients PATCHing different keys at once must not clobber each
    # other. Without a write lock both read the same on-disk state, each
    # writes only its own key, and the second write wins — losing the
    # first key. A slow serializer widens the race so the test is reliable.
    import threading
    import time
    import tomllib
    from concurrent.futures import ThreadPoolExecutor

    import speechtotext.api.routes_config as rc

    c, cfg_path = client
    real_dump = rc._dump_toml

    def slow_dump(d):
        time.sleep(0.05)
        return real_dump(d)

    monkeypatch.setattr(rc, "_dump_toml", slow_dump)

    ready = threading.Event()

    def run(body):
        ready.wait(timeout=2.0)
        return c.patch("/config", json=body)

    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(run, {"asr_model": "modelA"})
        f2 = ex.submit(run, {"backend": "cpu"})
        ready.set()
        r1 = f1.result(timeout=10)
        r2 = f2.result(timeout=10)

    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text
    raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    assert raw.get("asr_model") == "modelA", raw
    assert raw.get("backend") == "cpu", raw
