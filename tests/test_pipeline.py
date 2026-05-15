from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from speechtotext.config import Config
from speechtotext.models import LabeledSegment, ProgressEvent, Segment, SpeakerTurn
from speechtotext.pipeline import Pipeline


class FakeASR:
    def transcribe(self, wav_path: Path, language: str | None):
        return [Segment(0.0, 1.0, "hi", "en"), Segment(1.0, 2.0, "there", "en")]


class FakeDiarizer:
    def diarize(self, wav_path: Path, num_speakers: int | None):
        return [
            SpeakerTurn(0.0, 1.0, "SPEAKER_00"),
            SpeakerTurn(1.0, 2.0, "SPEAKER_01"),
        ]


@pytest.fixture
def fake_pipeline(tmp_path: Path) -> Pipeline:
    cfg = Config(backend="cpu", hf_token="hf_test")
    p = Pipeline(
        config=cfg,
        asr=FakeASR(),
        diarizer=FakeDiarizer(),
        resolved_backend="cpu",
    )
    return p


def test_run_produces_transcript_with_labels(fake_pipeline: Pipeline, tmp_path: Path):
    audio = tmp_path / "input.mp3"
    audio.write_bytes(b"fake")
    wav = tmp_path / "normalized.wav"
    wav.write_bytes(b"fake")

    with patch(
        "speechtotext.pipeline.normalize_to_wav", return_value=wav
    ):
        result = fake_pipeline.run(audio, language=None, num_speakers=None)

    assert result.language == "en"
    assert result.audio_path == audio
    assert len(result.segments) == 2
    assert result.segments[0].speaker_id == "SPEAKER_00"
    assert result.segments[1].speaker_id == "SPEAKER_01"
    assert result.models["backend"] == "cpu"


def test_run_emits_progress_events(fake_pipeline: Pipeline, tmp_path: Path):
    audio = tmp_path / "input.mp3"
    audio.write_bytes(b"fake")
    wav = tmp_path / "normalized.wav"
    wav.write_bytes(b"fake")

    events: list[ProgressEvent] = []
    with patch("speechtotext.pipeline.normalize_to_wav", return_value=wav):
        fake_pipeline.run(audio, language=None, on_progress=events.append)

    stages = [e.stage for e in events]
    assert stages[0] == "ingest"
    assert "asr" in stages
    assert "diarize" in stages
    assert stages[-1] == "merge"  # write happens outside pipeline.run


def test_run_cleans_up_temp_wav(fake_pipeline: Pipeline, tmp_path: Path):
    audio = tmp_path / "input.mp3"
    audio.write_bytes(b"fake")
    wav = tmp_path / "normalized.wav"
    wav.write_bytes(b"fake")
    with patch("speechtotext.pipeline.normalize_to_wav", return_value=wav):
        fake_pipeline.run(audio, language=None)
    assert not wav.exists()
