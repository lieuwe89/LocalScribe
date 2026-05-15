from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from speechtotext.models import LabeledSegment, Transcript
from speechtotext.writer import _atomic_write, format_txt


def _load(json_path: Path) -> Transcript:
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    return Transcript(
        audio_path=Path(raw["audio_path"]),
        duration_seconds=raw["duration_seconds"],
        language=raw["language"],
        speakers=dict(raw["speakers"]),
        segments=[
            LabeledSegment(
                start=s["start"], end=s["end"], text=s["text"], speaker_id=s["speaker"]
            )
            for s in raw["segments"]
        ],
        models=dict(raw.get("models", {})),
        created_at=datetime.fromisoformat(raw["created_at"]),
    )


def relabel(json_path: Path, mapping: dict[str, str]) -> None:
    transcript = _load(json_path)
    unknown = set(mapping) - set(transcript.speakers)
    if unknown:
        raise KeyError(
            f"unknown speaker ids: {sorted(unknown)}. valid: {sorted(transcript.speakers)}"
        )
    transcript.speakers.update(mapping)

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    raw["speakers"] = dict(transcript.speakers)
    _atomic_write(json_path, json.dumps(raw, indent=2, ensure_ascii=False))

    txt_path = json_path.with_suffix(".txt")
    _atomic_write(txt_path, format_txt(transcript))
