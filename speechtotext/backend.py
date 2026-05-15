from __future__ import annotations

import os
from typing import Literal

from speechtotext.config import Config

ResolvedBackend = Literal["cpu", "cuda", "mps"]
_VALID = frozenset({"auto", "cpu", "cuda", "mps"})


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _mps_available() -> bool:
    try:
        import torch

        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        return False


def resolve_backend(cli_flag: str | None, config: Config) -> ResolvedBackend:
    chosen = cli_flag or os.environ.get("STT_BACKEND") or config.backend
    if chosen not in _VALID:
        raise ValueError(f"invalid backend {chosen!r}")
    if chosen != "auto":
        return chosen  # type: ignore[return-value]
    if _cuda_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"
