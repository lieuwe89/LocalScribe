"""Workspace identity for the hub.

A hub maintains two long-lived identifiers across reboots:

- ``workspace_id`` — identifies this user's workspace globally. All paired
  devices and synced transcripts belong to one ``workspace_id``. Future
  multi-user setups (v2) introduce additional workspaces with their own
  ids; v1 keeps a single workspace per hub.
- ``device_id`` — identifies the hub itself among the paired devices, so
  ops the hub originates (e.g. user-driven relabels on the desktop app)
  carry a stable origin in the CRDT history.

Both are stable random values generated on first boot and persisted to a
small JSON file under the platform-appropriate app-data directory
(matching :func:`speechtotext.api.library_db.default_app_data_dir`).
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path

from speechtotext.api.library_db import default_app_data_dir

_FILENAME = "workspace.json"


class CorruptedWorkspaceError(RuntimeError):
    """Raised when workspace.json exists but is unreadable/unparseable.

    Distinguishes a corrupt identity file (fatal — refuse to regenerate, a
    new workspace_id would break every paired device) from a missing one
    (fine — generate a fresh identity).
    """


def workspace_file_path(config_dir: Path | None = None) -> Path:
    """Resolve the path to the workspace identity file.

    Pass ``config_dir`` in tests to redirect away from the user's real
    app-data directory.
    """
    base = Path(config_dir) if config_dir else default_app_data_dir()
    return base / _FILENAME


def _new_workspace_id() -> str:
    # 16 hex chars (~64 bits of randomness): plenty to avoid collisions
    # across the small number of workspaces a single user owns.
    return f"ws_{secrets.token_hex(8)}"


def _new_device_id() -> str:
    return f"hub-{secrets.token_hex(6)}"


def _load(config_dir: Path | None) -> dict:
    path = workspace_file_path(config_dir)
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CorruptedWorkspaceError(f"failed to read {path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CorruptedWorkspaceError(
            f"{path} exists but is not valid JSON: {exc}. Refusing to "
            "regenerate — a new workspace_id would break every paired "
            "device. Restore the file from backup, or delete it to start "
            "a fresh workspace."
        ) from exc
    if not isinstance(data, dict):
        raise CorruptedWorkspaceError(
            f"{path} does not contain a JSON object; refusing to "
            "regenerate. Restore from backup, or delete it to reset."
        )
    return data


def _save(data: dict, config_dir: Path | None) -> None:
    path = workspace_file_path(config_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(tmp, path)


def _ensure(config_dir: Path | None) -> dict:
    """Load or create the workspace identity record.

    Both ``workspace_id`` and ``device_id`` are generated together on
    first call so the file is always in one of two states: missing, or
    complete. Partial writes are recovered on the next call.
    """
    data = _load(config_dir)
    changed = False
    if "workspace_id" not in data:
        data["workspace_id"] = _new_workspace_id()
        changed = True
    if "device_id" not in data:
        data["device_id"] = _new_device_id()
        changed = True
    if changed:
        _save(data, config_dir)
    return data


def get_workspace_id(config_dir: Path | None = None) -> str:
    """Return the persistent workspace_id, generating on first call."""
    return _ensure(config_dir)["workspace_id"]


def get_device_id(config_dir: Path | None = None) -> str:
    """Return the persistent hub device_id, generating on first call."""
    return _ensure(config_dir)["device_id"]


# ── Lamport counter ────────────────────────────────────────────────────────
#
# The hub maintains a workspace-scoped Lamport counter. Every applied PATCH
# op advances it. It is persisted alongside workspace_id / device_id so a
# hub restart preserves the global ordering across the fleet.
#
# Persistence is "write on every advance" — fine at the PATCH rate this
# product sees (sparse human edits, not high-frequency machine writes).
# An in-memory cache layered on top is a future optimisation, not v1.


def get_lamport(config_dir: Path | None = None) -> int:
    """Return the current hub Lamport counter (0 on first call)."""
    data = _ensure(config_dir)
    try:
        return int(data.get("lamport_counter", 0))
    except (TypeError, ValueError):
        return 0


def bump_lamport_to(new_value: int, config_dir: Path | None = None) -> int:
    """Advance the persisted Lamport counter to ``new_value``.

    No-op when ``new_value`` <= current counter (Lamport must only ever
    move forward). Returns the resulting counter value.
    """
    data = _ensure(config_dir)
    try:
        current = int(data.get("lamport_counter", 0))
    except (TypeError, ValueError):
        current = 0
    if new_value > current:
        data["lamport_counter"] = new_value
        _save(data, config_dir)
        return new_value
    return current
