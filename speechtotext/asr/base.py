from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from speechtotext.models import Segment


@runtime_checkable
class ASRBackend(Protocol):
    def transcribe(self, wav_path: Path, language: str | None) -> list[Segment]: ...
