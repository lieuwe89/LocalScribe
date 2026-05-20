from __future__ import annotations

import json
import os
import socket
import sys

import uvicorn

from speechtotext.api.app import create_app


def pick_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def run(host: str = "127.0.0.1", port: int | None = None, print_handshake: bool = True) -> None:
    """Tauri-spawned sidecar entry: localhost, random port, JSON handshake on stdout."""
    p = port or pick_port()
    if print_handshake:
        sys.stdout.write(json.dumps({"locallexis": {"host": host, "port": p}}) + "\n")
        sys.stdout.flush()
    uvicorn.run(create_app(), host=host, port=p, log_level="warning")


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def headless() -> None:
    """Standalone hub entry: binds 0.0.0.0 on a stable port.

    Used by the headless deployment (Docker image, systemd unit, NAS /
    VPS install) and by the Tauri shell when "hub mode" is on.

    Defaults:

    - host: ``0.0.0.0``  (override via ``LOCALLEXIS_HOST``)
    - port: ``8765``     (override via ``LOCALLEXIS_PORT``)
    - tls:  off          (enable via ``LOCALLEXIS_TLS_ENABLED=1``)

    When ``LOCALLEXIS_TLS_ENABLED`` is truthy, the sidecar serves
    HTTPS using the self-signed cert at
    ``<app-data>/hub-cert.pem`` (auto-generated on first call). Mobile
    clients are expected to pin the cert's SPKI fingerprint via the
    pairing QR rather than rely on hostname matching.

    No stdout handshake — the Tauri shell uses ``run`` for the
    localhost sidecar lifecycle that needs the handshake; ``headless``
    is the LAN/server entry.
    """
    host = os.environ.get("LOCALLEXIS_HOST", "0.0.0.0")
    try:
        port = int(os.environ.get("LOCALLEXIS_PORT", "8765"))
    except ValueError as exc:
        raise SystemExit(
            f"LOCALLEXIS_PORT must be an integer, got "
            f"{os.environ.get('LOCALLEXIS_PORT')!r}: {exc}"
        )

    kwargs: dict = {"host": host, "port": port, "log_level": "info"}

    if _env_truthy("LOCALLEXIS_TLS_ENABLED"):
        # Lazy import so non-TLS callers don't pay the cryptography
        # import cost (and tests can monkeypatch the resolver).
        from speechtotext.api.tls import get_or_create_tls

        cert_path, key_path = get_or_create_tls()
        kwargs["ssl_certfile"] = str(cert_path)
        kwargs["ssl_keyfile"] = str(key_path)

    uvicorn.run(create_app(), **kwargs)
