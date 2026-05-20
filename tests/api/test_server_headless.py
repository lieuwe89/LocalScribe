"""Tests for the headless server entry point (``server.headless``).

The function is small but the env-var contract is part of the public
deploy interface — Docker images, systemd units, and operators rely
on it. Cover the happy path and the only realistic misconfiguration
(non-integer port).
"""

from __future__ import annotations

import pytest

from speechtotext.api import server


@pytest.fixture
def fake_uvicorn(monkeypatch):
    """Replace ``uvicorn.run`` so the test doesn't actually start a server."""
    calls: list[dict] = []

    def fake_run(app, **kwargs):
        calls.append({"app": app, **kwargs})

    monkeypatch.setattr(server.uvicorn, "run", fake_run)
    return calls


class TestHeadless:
    def test_defaults_to_0_0_0_0_port_8765(
        self, fake_uvicorn, monkeypatch
    ) -> None:
        monkeypatch.delenv("LOCALLEXIS_HOST", raising=False)
        monkeypatch.delenv("LOCALLEXIS_PORT", raising=False)
        server.headless()
        assert len(fake_uvicorn) == 1
        call = fake_uvicorn[0]
        assert call["host"] == "0.0.0.0"
        assert call["port"] == 8765

    def test_env_overrides_host(self, fake_uvicorn, monkeypatch) -> None:
        monkeypatch.setenv("LOCALLEXIS_HOST", "127.0.0.1")
        monkeypatch.delenv("LOCALLEXIS_PORT", raising=False)
        server.headless()
        assert fake_uvicorn[0]["host"] == "127.0.0.1"

    def test_env_overrides_port(self, fake_uvicorn, monkeypatch) -> None:
        monkeypatch.delenv("LOCALLEXIS_HOST", raising=False)
        monkeypatch.setenv("LOCALLEXIS_PORT", "9999")
        server.headless()
        assert fake_uvicorn[0]["port"] == 9999

    def test_non_integer_port_exits_clearly(
        self, fake_uvicorn, monkeypatch
    ) -> None:
        monkeypatch.setenv("LOCALLEXIS_PORT", "not-a-number")
        with pytest.raises(SystemExit) as exc_info:
            server.headless()
        # Operator-readable message, not a bare traceback.
        assert "LOCALLEXIS_PORT" in str(exc_info.value)

    def test_no_handshake_to_stdout(
        self, fake_uvicorn, monkeypatch, capsys
    ) -> None:
        monkeypatch.delenv("LOCALLEXIS_HOST", raising=False)
        monkeypatch.delenv("LOCALLEXIS_PORT", raising=False)
        server.headless()
        captured = capsys.readouterr()
        assert "locallexis" not in captured.out
        assert "locallexis" not in captured.err


class TestRunBackwardsCompat:
    """The Tauri-spawned ``run`` entry must keep printing the handshake
    and binding 127.0.0.1; the sidecar lifecycle depends on it."""

    def test_run_still_prints_handshake(
        self, fake_uvicorn, capsys
    ) -> None:
        server.run(port=12345)
        captured = capsys.readouterr()
        assert '"port": 12345' in captured.out
        assert fake_uvicorn[0]["host"] == "127.0.0.1"
        assert fake_uvicorn[0]["port"] == 12345
