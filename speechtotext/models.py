from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

Stage = Literal["ingest", "vad", "asr", "diarize", "merge", "write"]
_VALID_STAGES: frozenset[str] = frozenset(
    {"ingest", "vad", "asr", "diarize", "merge", "write"}
)


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    language: str | None = None


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker_id: str


@dataclass(frozen=True)
class LabeledSegment:
    start: float
    end: float
    text: str
    speaker_id: str


@dataclass
class Transcript:
    audio_path: Path
    duration_seconds: float
    language: str
    speakers: dict[str, str]
    segments: list[LabeledSegment]
    models: dict[str, str]
    created_at: datetime


@dataclass
class ProgressEvent:
    stage: Stage
    pct: float
    message: str = ""

    def __post_init__(self) -> None:
        if self.stage not in _VALID_STAGES:
            raise ValueError(f"invalid stage: {self.stage!r}")
        if not 0.0 <= self.pct <= 1.0:
            raise ValueError(f"pct out of range [0,1]: {self.pct}")
