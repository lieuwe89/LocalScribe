from datetime import datetime, timezone
from pathlib import Path

import pytest

from speechtotext.models import (
    LabeledSegment,
    ProgressEvent,
    Segment,
    SpeakerTurn,
    Transcript,
)


def test_segment_construction():
    s = Segment(start=0.0, end=1.5, text="hallo", language="nl")
    assert s.start == 0.0
    assert s.end == 1.5
    assert s.text == "hallo"
    assert s.language == "nl"


def test_speaker_turn_construction():
    t = SpeakerTurn(start=0.0, end=3.0, speaker_id="SPEAKER_00")
    assert t.speaker_id == "SPEAKER_00"


def test_labeled_segment_construction():
    ls = LabeledSegment(start=0.0, end=1.0, text="hi", speaker_id="SPEAKER_00")
    assert ls.speaker_id == "SPEAKER_00"


def test_transcript_construction():
    tr = Transcript(
        audio_path=Path("/tmp/x.mp3"),
        duration_seconds=42.0,
        language="en",
        speakers={"SPEAKER_00": "Speaker 1"},
        segments=[LabeledSegment(0.0, 1.0, "hi", "SPEAKER_00")],
        models={"asr": "faster-whisper:large-v3"},
        created_at=datetime(2026, 5, 15, tzinfo=timezone.utc),
    )
    assert tr.duration_seconds == 42.0
    assert len(tr.segments) == 1


def test_progress_event_construction():
    e = ProgressEvent(stage="asr", pct=0.5, message="halfway")
    assert e.stage == "asr"
    assert e.pct == 0.5


def test_progress_event_invalid_stage_raises():
    with pytest.raises(ValueError):
        ProgressEvent(stage="not-a-stage", pct=0.5, message="")  # type: ignore[arg-type]


def test_progress_event_invalid_pct_raises():
    with pytest.raises(ValueError, match="pct out of range"):
        ProgressEvent(stage="asr", pct=1.5, message="")
    with pytest.raises(ValueError, match="pct out of range"):
        ProgressEvent(stage="asr", pct=-0.1, message="")
