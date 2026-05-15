import json
from pathlib import Path

import pytest

from speechtotext.relabel import relabel


@pytest.fixture
def sidecar(tmp_path: Path) -> Path:
    audio = tmp_path / "rec.mp3"
    audio.write_bytes(b"fake")
    js = audio.with_suffix(".json")
    js.write_text(
        json.dumps(
            {
                "version": 1,
                "audio_path": str(audio),
                "duration_seconds": 4.0,
                "language": "en",
                "speakers": {
                    "SPEAKER_00": "Speaker 1",
                    "SPEAKER_01": "Speaker 2",
                },
                "segments": [
                    {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "text": "hello"},
                    {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01", "text": "hi"},
                ],
                "models": {"asr": "faster-whisper:large-v3"},
                "created_at": "2026-05-15T12:00:00+00:00",
            }
        )
    )
    audio.with_suffix(".txt").write_text("[00:00:00] Speaker 1: hello\n")
    return js


def test_relabel_renames_in_json(sidecar: Path):
    relabel(sidecar, {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"})
    data = json.loads(sidecar.read_text())
    assert data["speakers"]["SPEAKER_00"] == "Alice"
    assert data["speakers"]["SPEAKER_01"] == "Bob"


def test_relabel_regenerates_txt(sidecar: Path):
    relabel(sidecar, {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"})
    txt = sidecar.with_suffix(".txt").read_text()
    assert "Alice: hello" in txt
    assert "Bob: hi" in txt


def test_invalid_speaker_id_raises(sidecar: Path):
    with pytest.raises(KeyError, match="SPEAKER_99"):
        relabel(sidecar, {"SPEAKER_99": "Ghost"})
