import shutil
import threading
import time
from pathlib import Path

from speechtotext.ingest.watch import WatchQueue, should_process


def test_should_skip_when_sidecar_newer(tmp_path: Path):
    audio = tmp_path / "rec.mp3"
    audio.write_bytes(b"fake")
    sidecar = audio.with_suffix(".json")
    time.sleep(0.05)
    sidecar.write_text("{}")
    assert should_process(audio, overwrite=False) is False


def test_should_process_when_no_sidecar(tmp_path: Path):
    audio = tmp_path / "rec.mp3"
    audio.write_bytes(b"fake")
    assert should_process(audio, overwrite=False) is True


def test_should_process_when_overwrite(tmp_path: Path):
    audio = tmp_path / "rec.mp3"
    audio.write_bytes(b"fake")
    audio.with_suffix(".json").write_text("{}")
    assert should_process(audio, overwrite=True) is True


def test_should_process_when_audio_newer_than_sidecar(tmp_path: Path):
    audio = tmp_path / "rec.mp3"
    audio.write_bytes(b"fake")
    sidecar = audio.with_suffix(".json")
    sidecar.write_text("{}")
    time.sleep(0.05)
    audio.write_bytes(b"newer")
    assert should_process(audio, overwrite=False) is True


def test_queue_debounces_rapid_writes(tmp_path: Path):
    q = WatchQueue(debounce_seconds=0.2)
    f = tmp_path / "a.mp3"
    f.write_bytes(b"x")
    q.enqueue(f)
    q.enqueue(f)
    q.enqueue(f)
    ready_initially = q.drain_ready()
    assert ready_initially == []
    time.sleep(0.25)
    ready_after = q.drain_ready()
    assert ready_after == [f]
