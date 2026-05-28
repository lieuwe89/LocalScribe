"""Hub-side storage of the workspace symmetric key ``W``.

``W`` is a 32-byte random symmetric key that the pairing protocol
sealed-boxes to a new device's public key, so the key itself never
travels in plaintext over the wire. ``W`` is also used (in v2 work) for
client-side encryption layered on top of S3 storage so the storage
provider sees ciphertext only.

Persistence
-----------

Stored at ``<app-data-dir>/secrets.bin`` with mode ``0600``. Separated
from non-secret metadata (``workspace.json``) to make the "what to
back up" story explicit: lose ``secrets.bin`` and you lose the
ability to onboard new devices into the existing workspace, and any
encrypted-storage payloads with it. Users back this file up — see
the recovery section of the architecture plan.

OS-keychain integration (so ``secrets.bin`` itself is encrypted at
rest) is a v2 hardening. v1 relies on the file-permission gate plus
the user's disk-encryption.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

from speechtotext.api.library_db import default_app_data_dir

_FILENAME = "secrets.bin"
_KEY_BYTES = 32


class CorruptedSecretError(RuntimeError):
    """Raised when secrets.bin exists but is unreadable/truncated.

    Distinguishes a corrupt key file (fatal — refuse to overwrite) from a
    missing one (fine — generate a fresh key).
    """


def secrets_file_path(config_dir: Path | None = None) -> Path:
    """Path of the secrets file. Pass ``config_dir`` in tests to redirect."""
    base = Path(config_dir) if config_dir else default_app_data_dir()
    return base / _FILENAME


def _generate_and_write(path: Path) -> bytes:
    key = secrets.token_bytes(_KEY_BYTES)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Write into the tmp file with mode 0600 BEFORE the atomic rename,
    # so the key bytes are never world-readable on disk even briefly.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, key)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    return key


def get_workspace_key(config_dir: Path | None = None) -> bytes:
    """Return the 32-byte workspace symmetric key W, generating on first call.

    Subsequent calls return the same key, even across process restarts.
    If the secrets file exists but is shorter than 32 bytes (truncated or
    corrupt) this raises :class:`CorruptedSecretError` rather than
    regenerating: silently minting a new key would permanently destroy
    the ability to decrypt existing encrypted-storage payloads, so we
    fail loudly and leave the damaged file in place for recovery.
    """
    path = secrets_file_path(config_dir)
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"failed to read {path}: {exc}") from exc
        if len(existing) >= _KEY_BYTES:
            return existing[:_KEY_BYTES]
        raise CorruptedSecretError(
            f"{path} exists but is {len(existing)} bytes "
            f"(expected >= {_KEY_BYTES}); refusing to overwrite the "
            "workspace key. Restore secrets.bin from backup, or delete it "
            "to start a fresh workspace — existing encrypted backups and "
            "paired devices will be unrecoverable if you do."
        )
    return _generate_and_write(path)
