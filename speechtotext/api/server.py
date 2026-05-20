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


def headless() -> None:
    """Standalone hub entry: binds 0.0.0.0 on a stable port.

    Used by the headless deployment (Docker image, systemd unit, NAS /
    VPS install). Defaults:

    - host: ``0.0.0.0`` (override via ``LOCALLEXIS_HOST``)
    - port: ``8765``    (override via ``LOCALLEXIS_PORT``)

    No stdout handshake — the Tauri shell is the only caller that
    needs it. Log level is ``info`` so operators see request logs.

    Why a separate entry point at all? The Tauri-spawned ``run`` binds
    ``127.0.0.1`` and picks a random port, which is correct for a
    single-machine desktop install but useless on a server where the
    phone must reach the hub over the LAN. ``headless`` is the
    server-mode equivalent.
    """
    host = os.environ.get("LOCALLEXIS_HOST", "0.0.0.0")
    try:
        port = int(os.environ.get("LOCALLEXIS_PORT", "8765"))
    except ValueError as exc:
        raise SystemExit(
            f"LOCALLEXIS_PORT must be an integer, got "
            f"{os.environ.get('LOCALLEXIS_PORT')!r}: {exc}"
        )
    uvicorn.run(create_app(), host=host, port=port, log_level="info")
