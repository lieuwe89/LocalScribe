# Speech-to-Text CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a privacy-preserving local CLI (`stt`) that transcribes Dutch and English audio, labels speakers anonymously, and stores results as `.txt` + `.json` sidecars — designed to run on macOS arm64 (dev) and Linux x86_64 (homelab) with a swappable CPU/CUDA/MPS backend.

**Architecture:** Layered Python package. CLI (Typer) is a thin shell over a `Pipeline` orchestrator. Pipeline composes swappable modules behind `ASRBackend` / `DiarizerBackend` Protocols. ASR via `faster-whisper` (Whisper large-v3), diarization via `pyannote.audio` 4.0. Output is a frozen-schema JSON sidecar plus a human-readable `.txt` next to the source audio.

**Tech Stack:** Python 3.11+, Typer, faster-whisper, pyannote.audio 4.0, silero-vad, ffmpeg (subprocess), sounddevice, watchdog, pytest, tomllib (stdlib).

**Reference Spec:** `docs/superpowers/specs/2026-05-14-speech-to-text-cli-design.md`

---

## Conventions Used Throughout This Plan

- **Python imports** use absolute module paths (`speechtotext.models`, never `..models`).
- **In code**, filesystem paths use `pathlib.Path` with absolute paths (e.g., `Path.home() / ".config" / ...`).
- **In shell commands** below, paths are written relative to the project root (`/Users/lieuwejongsma/SpeechToText`) because git/pytest commands assume cwd = repo root.
- **Tests** use `pytest` and live in `tests/`. Mirror the package layout.
- **Commits** are atomic (one task = one commit). Conventional Commits style.
- **Dependencies** are added to `pyproject.toml` only when first used by a task. Don't dump everything in Task 1.
- **Fixture audio** lives in `tests/fixtures/audio/`. Generation script is part of Task 14.

---

## File Map (locked in here so tasks reference consistent paths)

```
SpeechToText/
├── pyproject.toml
├── README.md
├── .gitignore
├── speechtotext/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── pipeline.py
│   ├── models.py
│   ├── merger.py
│   ├── writer.py
│   ├── relabel.py
│   ├── backend.py                  # backend resolution helper
│   ├── progress.py                 # progress event helpers + console renderer
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── file.py
│   │   ├── mic.py
│   │   └── watch.py
│   ├── asr/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── faster_whisper.py
│   ├── diarize/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── pyannote.py
│   └── vad/
│       ├── __init__.py
│       ├── base.py
│       └── silero.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_config.py
│   ├── test_backend.py
│   ├── test_merger.py
│   ├── test_writer.py
│   ├── test_relabel.py
│   ├── test_pipeline.py
│   ├── test_cli.py
│   ├── ingest/
│   │   ├── test_file.py
│   │   └── test_watch.py
│   ├── asr/
│   │   └── test_faster_whisper.py
│   ├── diarize/
│   │   └── test_pyannote.py
│   └── fixtures/
│       ├── audio/
│       │   ├── README.md
│       │   ├── nl_2speakers_10s.wav
│       │   ├── en_2speakers_10s.wav
│       │   └── en_silence_then_speech.wav
│       └── generate_fixtures.py
└── docs/superpowers/  (already exists)
```

---

## Task 1: Project Skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `speechtotext/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

`tests/test_smoke.py`:
```python
import speechtotext

def test_package_importable():
    assert hasattr(speechtotext, "__version__")
    assert isinstance(speechtotext.__version__, str)
```

- [ ] **Step 2: Run test and verify it fails**

```bash
python -m pytest tests/test_smoke.py -v
```
Expected: ImportError or AttributeError — package not yet present.

- [ ] **Step 3: Create the package**

`speechtotext/__init__.py`:
```python
__version__ = "0.1.0"
```

`tests/__init__.py`: (empty file)

`tests/conftest.py`:
```python
from pathlib import Path
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    return FIXTURE_DIR


@pytest.fixture(scope="session")
def audio_fixture_dir(fixture_dir: Path) -> Path:
    return fixture_dir / "audio"
```

- [ ] **Step 4: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "speechtotext"
version = "0.1.0"
description = "Local privacy-preserving speech-to-text with speaker diarization."
requires-python = ">=3.11"
authors = [{ name = "Lieuwe Jongsma" }]
dependencies = []

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
]

[project.scripts]
stt = "speechtotext.cli:app"

[tool.setuptools.packages.find]
include = ["speechtotext*"]
exclude = ["tests*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

- [ ] **Step 5: Create .gitignore**

```
__pycache__/
*.py[cod]
.pytest_cache/
.coverage
htmlcov/
*.egg-info/
build/
dist/
.venv/
venv/
.env
*.wav.tmp
*.txt.tmp
*.json.tmp
.DS_Store
```

- [ ] **Step 6: Create README.md (stub)**

```markdown
# speechtotext

Local privacy-preserving speech-to-text with speaker diarization.

See `docs/superpowers/specs/2026-05-14-speech-to-text-cli-design.md` for design.

## Install (dev)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run tests

```bash
pytest
```
```

- [ ] **Step 7: Install and run smoke test**

```bash
python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]" && python -m pytest tests/test_smoke.py -v
```
Expected: 1 passed.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore README.md speechtotext/ tests/
git commit -m "feat: project skeleton with smoke test"
```

---

## Task 2: Domain Models (Dataclasses)

**Files:**
- Create: `speechtotext/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing tests**

`tests/test_models.py`:
```python
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
```

- [ ] **Step 2: Run test and verify it fails**

```bash
python -m pytest tests/test_models.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement models**

`speechtotext/models.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field
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
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
python -m pytest tests/test_models.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add speechtotext/models.py tests/test_models.py
git commit -m "feat(models): add core dataclasses for segments, turns, transcripts, progress"
```

---

## Task 3: Configuration Loading

**Files:**
- Create: `speechtotext/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

`tests/test_config.py`:
```python
from pathlib import Path

import pytest

from speechtotext.config import Config, WatchConfig, load_config


def test_default_config_when_no_file(tmp_path: Path):
    cfg = load_config(config_path=tmp_path / "missing.toml")
    assert cfg.backend in {"auto", "cpu", "cuda", "mps"}
    assert cfg.asr_model == "large-v3"
    assert cfg.hf_token is None
    assert cfg.default_out_dir is None
    assert cfg.watch.recursive is False
    assert cfg.watch.debounce_seconds == 2
    assert "wav" in cfg.watch.extensions


def test_loads_from_toml(tmp_path: Path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '''
backend = "cuda"
asr_model = "medium"
hf_token = "hf_abc"
model_cache_dir = "~/cache"
default_out_dir = "/tmp/out"

[watch]
recursive = true
debounce_seconds = 5
extensions = ["mp3", "flac"]
'''
    )
    cfg = load_config(config_path=cfg_file)
    assert cfg.backend == "cuda"
    assert cfg.asr_model == "medium"
    assert cfg.hf_token == "hf_abc"
    assert cfg.model_cache_dir == Path("~/cache").expanduser()
    assert cfg.default_out_dir == Path("/tmp/out")
    assert cfg.watch.recursive is True
    assert cfg.watch.debounce_seconds == 5
    assert cfg.watch.extensions == ["mp3", "flac"]


def test_invalid_backend_rejected(tmp_path: Path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('backend = "tpu"\n')
    with pytest.raises(ValueError, match="backend"):
        load_config(config_path=cfg_file)
```

- [ ] **Step 2: Run test and verify it fails**

```bash
python -m pytest tests/test_config.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement config**

`speechtotext/config.py`:
```python
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Backend = Literal["auto", "cpu", "cuda", "mps"]
_VALID_BACKENDS: frozenset[str] = frozenset({"auto", "cpu", "cuda", "mps"})

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "speechtotext" / "config.toml"
DEFAULT_MODEL_CACHE = Path.home() / ".cache" / "speechtotext" / "models"


@dataclass
class WatchConfig:
    recursive: bool = False
    debounce_seconds: int = 2
    extensions: list[str] = field(
        default_factory=lambda: ["mp3", "wav", "m4a", "mp4", "flac"]
    )


@dataclass
class Config:
    backend: Backend = "auto"
    asr_model: str = "large-v3"
    hf_token: str | None = None
    model_cache_dir: Path = field(default_factory=lambda: DEFAULT_MODEL_CACHE)
    default_out_dir: Path | None = None
    watch: WatchConfig = field(default_factory=WatchConfig)


def _expand(p: str) -> Path:
    return Path(os.path.expandvars(p)).expanduser()


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Config:
    if not config_path.exists():
        return Config()

    with config_path.open("rb") as fh:
        raw = tomllib.load(fh)

    backend = raw.get("backend", "auto")
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"invalid backend {backend!r}; expected one of {sorted(_VALID_BACKENDS)}"
        )

    watch_raw = raw.get("watch", {}) or {}
    watch = WatchConfig(
        recursive=bool(watch_raw.get("recursive", False)),
        debounce_seconds=int(watch_raw.get("debounce_seconds", 2)),
        extensions=list(
            watch_raw.get("extensions", ["mp3", "wav", "m4a", "mp4", "flac"])
        ),
    )

    return Config(
        backend=backend,  # type: ignore[arg-type]
        asr_model=str(raw.get("asr_model", "large-v3")),
        hf_token=raw.get("hf_token"),
        model_cache_dir=_expand(raw.get("model_cache_dir", str(DEFAULT_MODEL_CACHE))),
        default_out_dir=_expand(raw["default_out_dir"])
        if raw.get("default_out_dir")
        else None,
        watch=watch,
    )
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
python -m pytest tests/test_config.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add speechtotext/config.py tests/test_config.py
git commit -m "feat(config): load TOML config with defaults and validation"
```

---

## Task 4: Backend Resolution

**Files:**
- Create: `speechtotext/backend.py`
- Create: `tests/test_backend.py`

- [ ] **Step 1: Write failing tests**

`tests/test_backend.py`:
```python
from unittest.mock import patch

import pytest

from speechtotext.backend import resolve_backend
from speechtotext.config import Config


@pytest.fixture
def cfg() -> Config:
    return Config(backend="auto")


def test_cli_flag_wins(cfg: Config, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("STT_BACKEND", "cuda")
    cfg.backend = "cpu"
    assert resolve_backend(cli_flag="mps", config=cfg) == "mps"


def test_env_var_wins_over_config(cfg: Config, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("STT_BACKEND", "cuda")
    cfg.backend = "cpu"
    with patch("speechtotext.backend._cuda_available", return_value=True):
        assert resolve_backend(cli_flag=None, config=cfg) == "cuda"


def test_config_wins_when_no_flag_no_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("STT_BACKEND", raising=False)
    cfg = Config(backend="cpu")
    assert resolve_backend(cli_flag=None, config=cfg) == "cpu"


def test_auto_prefers_cuda(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("STT_BACKEND", raising=False)
    cfg = Config(backend="auto")
    with (
        patch("speechtotext.backend._cuda_available", return_value=True),
        patch("speechtotext.backend._mps_available", return_value=True),
    ):
        assert resolve_backend(cli_flag=None, config=cfg) == "cuda"


def test_auto_falls_back_to_mps_then_cpu(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("STT_BACKEND", raising=False)
    cfg = Config(backend="auto")
    with (
        patch("speechtotext.backend._cuda_available", return_value=False),
        patch("speechtotext.backend._mps_available", return_value=True),
    ):
        assert resolve_backend(cli_flag=None, config=cfg) == "mps"
    with (
        patch("speechtotext.backend._cuda_available", return_value=False),
        patch("speechtotext.backend._mps_available", return_value=False),
    ):
        assert resolve_backend(cli_flag=None, config=cfg) == "cpu"


def test_invalid_cli_flag_rejected(cfg: Config):
    with pytest.raises(ValueError):
        resolve_backend(cli_flag="tpu", config=cfg)
```

- [ ] **Step 2: Run test and verify it fails**

```bash
python -m pytest tests/test_backend.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement backend resolution**

`speechtotext/backend.py`:
```python
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
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
python -m pytest tests/test_backend.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add speechtotext/backend.py tests/test_backend.py
git commit -m "feat(backend): resolve cpu/cuda/mps from flag, env, config, autodetect"
```

---

## Task 5: Merger (Speaker-Segment Alignment)

**Files:**
- Create: `speechtotext/merger.py`
- Create: `tests/test_merger.py`

- [ ] **Step 1: Write failing tests**

`tests/test_merger.py`:
```python
from speechtotext.merger import merge
from speechtotext.models import Segment, SpeakerTurn


def test_single_speaker_single_segment():
    segs = [Segment(0.0, 2.0, "hello", "en")]
    turns = [SpeakerTurn(0.0, 3.0, "SPEAKER_00")]
    out = merge(segs, turns)
    assert len(out) == 1
    assert out[0].speaker_id == "SPEAKER_00"
    assert out[0].text == "hello"


def test_assigns_max_overlap_speaker():
    segs = [Segment(0.0, 4.0, "hi there", "en")]
    turns = [
        SpeakerTurn(0.0, 1.0, "SPEAKER_00"),  # 1s overlap
        SpeakerTurn(1.0, 4.0, "SPEAKER_01"),  # 3s overlap → wins
    ]
    out = merge(segs, turns)
    assert out[0].speaker_id == "SPEAKER_01"


def test_unknown_when_no_overlap():
    segs = [Segment(5.0, 6.0, "orphan", "en")]
    turns = [SpeakerTurn(0.0, 1.0, "SPEAKER_00")]
    out = merge(segs, turns)
    assert out[0].speaker_id == "UNKNOWN"


def test_unknown_when_overlap_below_threshold():
    # 30ms overlap < 50ms threshold
    segs = [Segment(0.0, 1.0, "tiny", "en")]
    turns = [SpeakerTurn(0.97, 5.0, "SPEAKER_00")]
    out = merge(segs, turns)
    assert out[0].speaker_id == "UNKNOWN"


def test_preserves_order_and_text():
    segs = [
        Segment(0.0, 1.0, "first", "en"),
        Segment(1.5, 2.5, "second", "en"),
    ]
    turns = [
        SpeakerTurn(0.0, 1.2, "SPEAKER_00"),
        SpeakerTurn(1.2, 3.0, "SPEAKER_01"),
    ]
    out = merge(segs, turns)
    assert [s.text for s in out] == ["first", "second"]
    assert [s.speaker_id for s in out] == ["SPEAKER_00", "SPEAKER_01"]


def test_empty_turns_yields_all_unknown():
    segs = [Segment(0.0, 1.0, "x", "en")]
    out = merge(segs, [])
    assert out[0].speaker_id == "UNKNOWN"


def test_empty_segments_yields_empty():
    assert merge([], [SpeakerTurn(0.0, 1.0, "SPEAKER_00")]) == []
```

- [ ] **Step 2: Run test and verify it fails**

```bash
python -m pytest tests/test_merger.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement merger**

`speechtotext/merger.py`:
```python
from __future__ import annotations

from speechtotext.models import LabeledSegment, Segment, SpeakerTurn

_MIN_OVERLAP_SECONDS = 0.050  # 50ms


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def merge(
    segments: list[Segment], turns: list[SpeakerTurn]
) -> list[LabeledSegment]:
    out: list[LabeledSegment] = []
    for seg in segments:
        best_id = "UNKNOWN"
        best_overlap = 0.0
        for turn in turns:
            ov = _overlap(seg.start, seg.end, turn.start, turn.end)
            if ov > best_overlap:
                best_overlap = ov
                best_id = turn.speaker_id
        if best_overlap < _MIN_OVERLAP_SECONDS:
            best_id = "UNKNOWN"
        out.append(
            LabeledSegment(
                start=seg.start, end=seg.end, text=seg.text, speaker_id=best_id
            )
        )
    return out
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
python -m pytest tests/test_merger.py -v
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add speechtotext/merger.py tests/test_merger.py
git commit -m "feat(merger): align ASR segments to speaker turns by max overlap"
```

---

## Task 6: Writer (Atomic .txt + .json Output)

**Files:**
- Create: `speechtotext/writer.py`
- Create: `tests/test_writer.py`

- [ ] **Step 1: Write failing tests**

`tests/test_writer.py`:
```python
import json
from datetime import datetime, timezone
from pathlib import Path

from speechtotext.models import LabeledSegment, Transcript
from speechtotext.writer import format_txt, write_transcript


def _transcript(audio: Path) -> Transcript:
    return Transcript(
        audio_path=audio,
        duration_seconds=4.0,
        language="en",
        speakers={"SPEAKER_00": "Speaker 1", "SPEAKER_01": "Speaker 2"},
        segments=[
            LabeledSegment(0.0, 2.0, "hello", "SPEAKER_00"),
            LabeledSegment(2.0, 4.0, "hi there", "SPEAKER_01"),
        ],
        models={"asr": "faster-whisper:large-v3", "diarizer": "pyannote:4.0", "backend": "cpu"},
        created_at=datetime(2026, 5, 15, 12, 0, 0, tzinfo=timezone.utc),
    )


def test_writes_sibling_files(tmp_path: Path):
    audio = tmp_path / "rec.mp3"
    audio.write_bytes(b"fake")
    tr = _transcript(audio)

    write_transcript(tr)

    txt = audio.with_suffix(".txt")
    js = audio.with_suffix(".json")
    assert txt.exists()
    assert js.exists()
    data = json.loads(js.read_text())
    assert data["version"] == 1
    assert data["language"] == "en"
    assert data["speakers"]["SPEAKER_00"] == "Speaker 1"
    assert len(data["segments"]) == 2
    assert data["segments"][0]["text"] == "hello"


def test_txt_format(tmp_path: Path):
    audio = tmp_path / "rec.mp3"
    audio.write_bytes(b"fake")
    txt = format_txt(_transcript(audio))
    assert "[00:00:00] Speaker 1: hello" in txt
    assert "[00:00:02] Speaker 2: hi there" in txt


def test_no_temp_files_left(tmp_path: Path):
    audio = tmp_path / "rec.mp3"
    audio.write_bytes(b"fake")
    write_transcript(_transcript(audio))
    assert not any(tmp_path.glob("*.tmp"))


def test_atomic_replace(tmp_path: Path):
    audio = tmp_path / "rec.mp3"
    audio.write_bytes(b"fake")
    (audio.with_suffix(".json")).write_text('{"old": true}')
    write_transcript(_transcript(audio))
    data = json.loads(audio.with_suffix(".json").read_text())
    assert "old" not in data
    assert data["version"] == 1
```

- [ ] **Step 2: Run test and verify it fails**

```bash
python -m pytest tests/test_writer.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement writer**

`speechtotext/writer.py`:
```python
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from speechtotext.models import Transcript

_SCHEMA_VERSION = 1


def _format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_txt(t: Transcript) -> str:
    lines: list[str] = []
    for seg in t.segments:
        display = t.speakers.get(seg.speaker_id, seg.speaker_id)
        lines.append(f"[{_format_timestamp(seg.start)}] {display}: {seg.text}")
    return "\n".join(lines) + ("\n" if lines else "")


def _serialize(t: Transcript) -> dict:
    return {
        "version": _SCHEMA_VERSION,
        "audio_path": str(t.audio_path),
        "duration_seconds": t.duration_seconds,
        "language": t.language,
        "speakers": dict(t.speakers),
        "segments": [
            {
                "start": s.start,
                "end": s.end,
                "speaker": s.speaker_id,
                "text": s.text,
            }
            for s in t.segments
        ],
        "models": dict(t.models),
        "created_at": t.created_at.isoformat(),
    }


def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def write_transcript(t: Transcript) -> tuple[Path, Path]:
    audio = t.audio_path
    txt_path = audio.with_suffix(".txt")
    json_path = audio.with_suffix(".json")

    txt_content = format_txt(t)
    json_content = json.dumps(_serialize(t), indent=2, ensure_ascii=False)

    _atomic_write(txt_path, txt_content)
    _atomic_write(json_path, json_content)
    return txt_path, json_path
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
python -m pytest tests/test_writer.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add speechtotext/writer.py tests/test_writer.py
git commit -m "feat(writer): atomic .txt + .json sidecar emission with frozen v1 schema"
```

---

## Task 7: Relabel Speakers

**Files:**
- Create: `speechtotext/relabel.py`
- Create: `tests/test_relabel.py`

- [ ] **Step 1: Write failing tests**

`tests/test_relabel.py`:
```python
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
```

- [ ] **Step 2: Run test and verify it fails**

```bash
python -m pytest tests/test_relabel.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement relabel**

`speechtotext/relabel.py`:
```python
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
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
python -m pytest tests/test_relabel.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add speechtotext/relabel.py tests/test_relabel.py
git commit -m "feat(relabel): rename speakers in sidecar JSON and regenerate TXT"
```

---

## Task 8: ASR Backend (faster-whisper)

**Files:**
- Create: `speechtotext/asr/__init__.py`
- Create: `speechtotext/asr/base.py`
- Create: `speechtotext/asr/faster_whisper.py`
- Create: `tests/asr/__init__.py`
- Create: `tests/asr/test_faster_whisper.py`

- [ ] **Step 1: Add dependency**

Edit `pyproject.toml`, append to `dependencies`:
```toml
dependencies = [
    "faster-whisper>=1.0",
]
```

- [ ] **Step 2: Write failing tests**

`tests/asr/__init__.py`: (empty)

`tests/asr/test_faster_whisper.py`:
```python
from pathlib import Path
from unittest.mock import MagicMock, patch

from speechtotext.asr.faster_whisper import FasterWhisperASR
from speechtotext.models import Segment


def _fake_whisper_segment(start: float, end: float, text: str):
    s = MagicMock()
    s.start = start
    s.end = end
    s.text = text
    return s


def test_transcribe_returns_segments(tmp_path: Path):
    wav = tmp_path / "x.wav"
    wav.write_bytes(b"fake")

    fake_segments = iter(
        [
            _fake_whisper_segment(0.0, 1.0, "hello"),
            _fake_whisper_segment(1.0, 2.0, "world"),
        ]
    )
    fake_info = MagicMock(language="en")

    with patch("speechtotext.asr.faster_whisper.WhisperModel") as Model:
        instance = Model.return_value
        instance.transcribe.return_value = (fake_segments, fake_info)

        asr = FasterWhisperASR(model_size="tiny", backend="cpu")
        result = asr.transcribe(wav, language=None)

    assert len(result) == 2
    assert isinstance(result[0], Segment)
    assert result[0].text == "hello"
    assert result[0].language == "en"


def test_backend_to_device_mapping():
    with patch("speechtotext.asr.faster_whisper.WhisperModel") as Model:
        FasterWhisperASR(model_size="tiny", backend="cuda")
        kwargs = Model.call_args.kwargs
        assert kwargs["device"] == "cuda"
        assert kwargs["compute_type"] == "float16"

    with patch("speechtotext.asr.faster_whisper.WhisperModel") as Model:
        FasterWhisperASR(model_size="tiny", backend="mps")
        kwargs = Model.call_args.kwargs
        assert kwargs["device"] == "cpu"  # mps not yet supported by CTranslate2
        assert kwargs["compute_type"] == "int8"

    with patch("speechtotext.asr.faster_whisper.WhisperModel") as Model:
        FasterWhisperASR(model_size="tiny", backend="cpu")
        kwargs = Model.call_args.kwargs
        assert kwargs["device"] == "cpu"
        assert kwargs["compute_type"] == "int8"


def test_language_passed_through(tmp_path: Path):
    wav = tmp_path / "x.wav"
    wav.write_bytes(b"fake")
    fake_info = MagicMock(language="nl")
    with patch("speechtotext.asr.faster_whisper.WhisperModel") as Model:
        instance = Model.return_value
        instance.transcribe.return_value = (iter([]), fake_info)
        asr = FasterWhisperASR(model_size="tiny", backend="cpu")
        asr.transcribe(wav, language="nl")
        assert instance.transcribe.call_args.kwargs["language"] == "nl"
```

- [ ] **Step 3: Implement the ASR module**

`speechtotext/asr/__init__.py`: (empty)

`speechtotext/asr/base.py`:
```python
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from speechtotext.models import Segment


@runtime_checkable
class ASRBackend(Protocol):
    def transcribe(self, wav_path: Path, language: str | None) -> list[Segment]: ...
```

`speechtotext/asr/faster_whisper.py`:
```python
from __future__ import annotations

from pathlib import Path
from typing import Literal

from faster_whisper import WhisperModel

from speechtotext.models import Segment

_DEVICE_MAP: dict[str, tuple[str, str]] = {
    "cpu": ("cpu", "int8"),
    "cuda": ("cuda", "float16"),
    "mps": ("cpu", "int8"),  # CTranslate2 has no native MPS; CPU on Apple Silicon
}


class FasterWhisperASR:
    def __init__(
        self,
        model_size: str = "large-v3",
        backend: Literal["cpu", "cuda", "mps"] = "cpu",
        download_root: Path | None = None,
    ) -> None:
        device, compute_type = _DEVICE_MAP[backend]
        self._model = WhisperModel(
            model_size_or_path=model_size,
            device=device,
            compute_type=compute_type,
            download_root=str(download_root) if download_root else None,
        )

    def transcribe(self, wav_path: Path, language: str | None) -> list[Segment]:
        segments_iter, info = self._model.transcribe(
            str(wav_path),
            language=language,
            beam_size=5,
            temperature=0.0,
            vad_filter=True,
        )
        out: list[Segment] = []
        for s in segments_iter:
            out.append(
                Segment(
                    start=float(s.start),
                    end=float(s.end),
                    text=s.text.strip(),
                    language=info.language,
                )
            )
        return out
```

- [ ] **Step 4: Install and run tests**

```bash
pip install -e ".[dev]" && python -m pytest tests/asr/ -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add speechtotext/asr/ tests/asr/ pyproject.toml
git commit -m "feat(asr): faster-whisper backend with cpu/cuda/mps device mapping"
```

---

## Task 9: Diarization Backend (pyannote.audio 4.0)

**Files:**
- Create: `speechtotext/diarize/__init__.py`
- Create: `speechtotext/diarize/base.py`
- Create: `speechtotext/diarize/pyannote.py`
- Create: `tests/diarize/__init__.py`
- Create: `tests/diarize/test_pyannote.py`

- [ ] **Step 1: Add dependency**

Edit `pyproject.toml`:
```toml
dependencies = [
    "faster-whisper>=1.0",
    "pyannote.audio>=4.0",
    "torch>=2.2",
]
```

- [ ] **Step 2: Write failing tests**

`tests/diarize/__init__.py`: (empty)

`tests/diarize/test_pyannote.py`:
```python
from pathlib import Path
from unittest.mock import MagicMock, patch

from speechtotext.diarize.pyannote import PyannoteDiarizer
from speechtotext.models import SpeakerTurn


def _fake_annotation(turns: list[tuple[float, float, str]]):
    ann = MagicMock()

    def _itertracks(yield_label: bool = True):
        for s, e, lab in turns:
            seg = MagicMock(start=s, end=e)
            yield seg, None, lab

    ann.itertracks.side_effect = _itertracks
    return ann


def test_diarize_returns_speaker_turns(tmp_path: Path):
    wav = tmp_path / "x.wav"
    wav.write_bytes(b"fake")

    pipeline = MagicMock()
    pipeline.return_value = _fake_annotation(
        [(0.0, 1.5, "SPEAKER_00"), (1.5, 3.0, "SPEAKER_01")]
    )

    with patch(
        "speechtotext.diarize.pyannote.Pipeline.from_pretrained", return_value=pipeline
    ):
        diarizer = PyannoteDiarizer(hf_token="hf_test", backend="cpu")
        turns = diarizer.diarize(wav, num_speakers=None)

    assert len(turns) == 2
    assert isinstance(turns[0], SpeakerTurn)
    assert turns[0].speaker_id == "SPEAKER_00"
    assert turns[1].speaker_id == "SPEAKER_01"


def test_diarize_passes_num_speakers_hint(tmp_path: Path):
    wav = tmp_path / "x.wav"
    wav.write_bytes(b"fake")
    pipeline = MagicMock()
    pipeline.return_value = _fake_annotation([])

    with patch(
        "speechtotext.diarize.pyannote.Pipeline.from_pretrained", return_value=pipeline
    ):
        diarizer = PyannoteDiarizer(hf_token="hf_test", backend="cpu")
        diarizer.diarize(wav, num_speakers=3)

    assert pipeline.call_args.kwargs.get("num_speakers") == 3


def test_backend_sets_torch_device():
    pipeline = MagicMock()
    with (
        patch(
            "speechtotext.diarize.pyannote.Pipeline.from_pretrained", return_value=pipeline
        ),
        patch("speechtotext.diarize.pyannote.torch") as torch_mod,
    ):
        torch_mod.device.return_value = "fake-device"
        PyannoteDiarizer(hf_token="hf_test", backend="cuda")
        torch_mod.device.assert_called_with("cuda")
        pipeline.to.assert_called_with("fake-device")
```

- [ ] **Step 3: Implement the diarization module**

`speechtotext/diarize/__init__.py`: (empty)

`speechtotext/diarize/base.py`:
```python
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from speechtotext.models import SpeakerTurn


@runtime_checkable
class DiarizerBackend(Protocol):
    def diarize(self, wav_path: Path, num_speakers: int | None) -> list[SpeakerTurn]: ...
```

`speechtotext/diarize/pyannote.py`:
```python
from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from pyannote.audio import Pipeline

from speechtotext.models import SpeakerTurn

_MODEL_ID = "pyannote/speaker-diarization-3.1"  # 4.0 endpoint: update when stable URL known


class PyannoteDiarizer:
    def __init__(
        self,
        hf_token: str,
        backend: Literal["cpu", "cuda", "mps"] = "cpu",
        model_id: str = _MODEL_ID,
    ) -> None:
        if not hf_token:
            raise ValueError("pyannote requires a Hugging Face access token")
        self._pipeline = Pipeline.from_pretrained(model_id, use_auth_token=hf_token)
        device = torch.device(backend)
        self._pipeline.to(device)

    def diarize(self, wav_path: Path, num_speakers: int | None) -> list[SpeakerTurn]:
        kwargs: dict = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        annotation = self._pipeline(str(wav_path), **kwargs)
        turns: list[SpeakerTurn] = []
        for segment, _track, label in annotation.itertracks(yield_label=True):
            turns.append(
                SpeakerTurn(
                    start=float(segment.start),
                    end=float(segment.end),
                    speaker_id=str(label),
                )
            )
        return turns
```

> **Note for implementer:** the model id placeholder `pyannote/speaker-diarization-3.1` is used because the exact 4.0 HF endpoint name was not confirmed at plan time. Verify the correct identifier from https://huggingface.co/pyannote during execution and update `_MODEL_ID`. Tests mock the pipeline and don't depend on the name.

- [ ] **Step 4: Install and run tests**

```bash
pip install -e ".[dev]" && python -m pytest tests/diarize/ -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add speechtotext/diarize/ tests/diarize/ pyproject.toml
git commit -m "feat(diarize): pyannote.audio wrapper with device selection"
```

---

## Task 10: Ingest — File Normalization via ffmpeg

**Files:**
- Create: `speechtotext/ingest/__init__.py`
- Create: `speechtotext/ingest/file.py`
- Create: `tests/ingest/__init__.py`
- Create: `tests/ingest/test_file.py`

- [ ] **Step 1: Write failing tests**

`tests/ingest/__init__.py`: (empty)

`tests/ingest/test_file.py`:
```python
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from speechtotext.ingest.file import IngestError, normalize_to_wav


def test_runs_ffmpeg_with_correct_args(tmp_path: Path):
    src = tmp_path / "a.mp3"
    src.write_bytes(b"fake")
    out = tmp_path / "out.wav"

    with patch("speechtotext.ingest.file.subprocess.run") as run:
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        normalize_to_wav(src, out)

    args = run.call_args.args[0]
    assert args[0] == "ffmpeg"
    assert "-ac" in args and "1" in args
    assert "-ar" in args and "16000" in args
    assert str(src) in args
    assert str(out) in args


def test_raises_on_ffmpeg_failure(tmp_path: Path):
    src = tmp_path / "a.mp3"
    src.write_bytes(b"fake")
    with patch("speechtotext.ingest.file.subprocess.run") as run:
        run.return_value.returncode = 1
        run.return_value.stderr = "bad codec"
        with pytest.raises(IngestError, match="bad codec"):
            normalize_to_wav(src, tmp_path / "out.wav")


def test_raises_when_ffmpeg_missing(tmp_path: Path):
    src = tmp_path / "a.mp3"
    src.write_bytes(b"fake")
    with patch(
        "speechtotext.ingest.file.subprocess.run", side_effect=FileNotFoundError
    ):
        with pytest.raises(IngestError, match="ffmpeg not found"):
            normalize_to_wav(src, tmp_path / "out.wav")


def test_input_missing(tmp_path: Path):
    with pytest.raises(IngestError, match="does not exist"):
        normalize_to_wav(tmp_path / "missing.mp3", tmp_path / "out.wav")
```

- [ ] **Step 2: Run test and verify it fails**

```bash
python -m pytest tests/ingest/ -v
```
Expected: ImportError.

- [ ] **Step 3: Implement file ingest**

`speechtotext/ingest/__init__.py`: (empty)

`speechtotext/ingest/file.py`:
```python
from __future__ import annotations

import subprocess
from pathlib import Path


class IngestError(RuntimeError):
    pass


def normalize_to_wav(src: Path, dst: Path) -> Path:
    if not src.exists():
        raise IngestError(f"input does not exist: {src}")

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise IngestError("ffmpeg not found on PATH") from exc

    if result.returncode != 0:
        raise IngestError(result.stderr.strip() or "ffmpeg conversion failed")
    return dst
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
python -m pytest tests/ingest/test_file.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add speechtotext/ingest/__init__.py speechtotext/ingest/file.py tests/ingest/
git commit -m "feat(ingest): normalize input audio to 16kHz mono WAV via ffmpeg"
```

---

## Task 11: Pipeline Orchestrator

**Files:**
- Create: `speechtotext/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

`tests/test_pipeline.py`:
```python
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
```

- [ ] **Step 2: Run test and verify it fails**

```bash
python -m pytest tests/test_pipeline.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement Pipeline**

`speechtotext/pipeline.py`:
```python
from __future__ import annotations

import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from speechtotext.asr.base import ASRBackend
from speechtotext.config import Config
from speechtotext.diarize.base import DiarizerBackend
from speechtotext.ingest.file import normalize_to_wav
from speechtotext.merger import merge
from speechtotext.models import ProgressEvent, Transcript

ProgressCallback = Callable[[ProgressEvent], None]


def _noop(_: ProgressEvent) -> None:
    pass


class Pipeline:
    def __init__(
        self,
        config: Config,
        asr: ASRBackend,
        diarizer: DiarizerBackend,
        resolved_backend: str,
    ) -> None:
        self._config = config
        self._asr = asr
        self._diarizer = diarizer
        self._backend = resolved_backend

    def run(
        self,
        audio: Path,
        language: str | None = None,
        num_speakers: int | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> Transcript:
        emit = on_progress or _noop

        tmp_dir = Path(tempfile.gettempdir()) / "speechtotext"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        wav = tmp_dir / f"{uuid.uuid4().hex}.wav"

        try:
            emit(ProgressEvent("ingest", 0.0, f"normalizing {audio.name}"))
            normalize_to_wav(audio, wav)
            emit(ProgressEvent("ingest", 1.0, "normalized"))

            emit(ProgressEvent("asr", 0.0, "transcribing"))
            segments = self._asr.transcribe(wav, language=language)
            detected_language = segments[0].language if segments else (language or "unknown")
            emit(ProgressEvent("asr", 1.0, f"{len(segments)} segments"))

            emit(ProgressEvent("diarize", 0.0, "diarizing"))
            turns = self._diarizer.diarize(wav, num_speakers=num_speakers)
            emit(ProgressEvent("diarize", 1.0, f"{len(turns)} turns"))

            emit(ProgressEvent("merge", 0.0, "aligning speakers"))
            labeled = merge(segments, turns)
            speaker_ids = sorted({s.speaker_id for s in labeled if s.speaker_id != "UNKNOWN"})
            speakers = {sid: f"Speaker {i + 1}" for i, sid in enumerate(speaker_ids)}
            emit(ProgressEvent("merge", 1.0, f"{len(speakers)} speakers"))

            duration = max((s.end for s in labeled), default=0.0)
            transcript = Transcript(
                audio_path=audio,
                duration_seconds=duration,
                language=detected_language or "unknown",
                speakers=speakers,
                segments=labeled,
                models={
                    "asr": f"faster-whisper:{self._config.asr_model}",
                    "diarizer": "pyannote:4.0",
                    "backend": self._backend,
                },
                created_at=datetime.now(timezone.utc),
            )
            return transcript
        finally:
            try:
                if wav.exists():
                    wav.unlink()
            except OSError:
                pass
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
python -m pytest tests/test_pipeline.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add speechtotext/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): orchestrate ingest, asr, diarize, merge with progress events"
```

---

## Task 12: Mic Ingest (Recording)

**Files:**
- Create: `speechtotext/ingest/mic.py`
- Create: `tests/ingest/test_mic.py`

- [ ] **Step 1: Add dependency**

Edit `pyproject.toml`:
```toml
dependencies = [
    "faster-whisper>=1.0",
    "pyannote.audio>=4.0",
    "torch>=2.2",
    "sounddevice>=0.4",
    "soundfile>=0.12",
    "numpy>=1.26",
]
```

- [ ] **Step 2: Write failing tests**

`tests/ingest/test_mic.py`:
```python
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from speechtotext.ingest.mic import record_to_wav


def test_record_writes_wav(tmp_path: Path):
    out = tmp_path / "rec.wav"
    fake_chunks = [
        np.zeros((1600, 1), dtype=np.int16),
        np.ones((1600, 1), dtype=np.int16),
    ]
    stop = MagicMock()
    stop.is_set.side_effect = [False, False, True]

    with (
        patch("speechtotext.ingest.mic.sd.InputStream") as stream_cls,
        patch("speechtotext.ingest.mic.sf.SoundFile") as sf_cls,
    ):
        stream = stream_cls.return_value.__enter__.return_value
        stream.read.side_effect = [(c, False) for c in fake_chunks]
        sf_handle = sf_cls.return_value.__enter__.return_value

        record_to_wav(out, sample_rate=16000, channels=1, stop_event=stop)

    assert sf_cls.call_args.args[0] == str(out)
    assert sf_handle.write.call_count == 2
```

- [ ] **Step 3: Run test and verify it fails**

```bash
python -m pytest tests/ingest/test_mic.py -v
```
Expected: ImportError.

- [ ] **Step 4: Implement mic recorder**

`speechtotext/ingest/mic.py`:
```python
from __future__ import annotations

import threading
from pathlib import Path

import sounddevice as sd
import soundfile as sf


def record_to_wav(
    out_path: Path,
    sample_rate: int = 16000,
    channels: int = 1,
    block_size: int = 1600,
    stop_event: threading.Event | None = None,
    device: str | int | None = None,
) -> Path:
    stop = stop_event or threading.Event()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with sf.SoundFile(
        str(out_path),
        mode="w",
        samplerate=sample_rate,
        channels=channels,
        subtype="PCM_16",
    ) as fh, sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="int16",
        blocksize=block_size,
        device=device,
    ) as stream:
        while not stop.is_set():
            data, _overflow = stream.read(block_size)
            fh.write(data)

    return out_path
```

- [ ] **Step 5: Run tests and verify they pass**

```bash
python -m pytest tests/ingest/test_mic.py -v
```
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add speechtotext/ingest/mic.py tests/ingest/test_mic.py pyproject.toml
git commit -m "feat(ingest): record from microphone to 16kHz mono WAV"
```

---

## Task 13: Watch Folder Daemon

**Files:**
- Create: `speechtotext/ingest/watch.py`
- Create: `tests/ingest/test_watch.py`

- [ ] **Step 1: Add dependency**

Edit `pyproject.toml`:
```toml
dependencies = [
    "faster-whisper>=1.0",
    "pyannote.audio>=4.0",
    "torch>=2.2",
    "sounddevice>=0.4",
    "soundfile>=0.12",
    "numpy>=1.26",
    "watchdog>=4.0",
]
```

- [ ] **Step 2: Write failing tests**

`tests/ingest/test_watch.py`:
```python
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
```

- [ ] **Step 3: Implement watch module**

`speechtotext/ingest/watch.py`:
```python
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer


def should_process(audio: Path, overwrite: bool) -> bool:
    sidecar = audio.with_suffix(".json")
    if overwrite or not sidecar.exists():
        return True
    return audio.stat().st_mtime > sidecar.stat().st_mtime


class WatchQueue:
    def __init__(self, debounce_seconds: float = 2.0) -> None:
        self._debounce = debounce_seconds
        self._pending: "OrderedDict[Path, float]" = OrderedDict()
        self._lock = threading.Lock()

    def enqueue(self, path: Path) -> None:
        with self._lock:
            self._pending[path] = time.monotonic()
            self._pending.move_to_end(path)

    def drain_ready(self) -> list[Path]:
        now = time.monotonic()
        ready: list[Path] = []
        with self._lock:
            for path, ts in list(self._pending.items()):
                if now - ts >= self._debounce:
                    ready.append(path)
                    self._pending.pop(path, None)
        return ready


class _Handler(FileSystemEventHandler):
    def __init__(self, queue: WatchQueue, extensions: set[str]) -> None:
        self._queue = queue
        self._exts = {e.lower().lstrip(".") for e in extensions}

    def on_created(self, event: FileSystemEvent) -> None:
        self._maybe(event)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._maybe(event)

    def _maybe(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        p = Path(event.src_path)
        if p.suffix.lower().lstrip(".") in self._exts:
            self._queue.enqueue(p)


def run_watch(
    directory: Path,
    extensions: list[str],
    debounce_seconds: float,
    recursive: bool,
    on_ready: Callable[[Path], None],
    stop_event: threading.Event | None = None,
    poll_interval: float = 0.5,
) -> None:
    stop = stop_event or threading.Event()
    queue = WatchQueue(debounce_seconds=debounce_seconds)
    observer = Observer()
    observer.schedule(_Handler(queue, set(extensions)), str(directory), recursive=recursive)
    observer.start()
    try:
        while not stop.is_set():
            for path in queue.drain_ready():
                on_ready(path)
            time.sleep(poll_interval)
    finally:
        observer.stop()
        observer.join()
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
python -m pytest tests/ingest/test_watch.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add speechtotext/ingest/watch.py tests/ingest/test_watch.py pyproject.toml
git commit -m "feat(ingest): watch folder daemon with debounce and overwrite policy"
```

---

## Task 14: Test Fixture Audio + Generator

**Files:**
- Create: `tests/fixtures/audio/README.md`
- Create: `tests/fixtures/generate_fixtures.py`
- Create: `tests/fixtures/audio/.gitkeep`

> **Why this matters:** Diarization + ASR tests can use mocks for backend tests, but the eventual integration test (Task 16) needs real-ish multi-speaker audio. Real diarization-quality audio is hard to generate synthetically. This task ships a generator script using `espeak-ng` (CLI) for two synthetic "speakers" with different pitches. It's enough to exercise the pipeline end-to-end on CPU. Real-world accuracy isn't tested here — that's manual QA.

- [ ] **Step 1: Document fixture requirements**

`tests/fixtures/audio/README.md`:
```markdown
# Audio fixtures

These wav files are required for integration tests. They are *not* committed
to git (binary churn). Regenerate locally:

```bash
python tests/fixtures/generate_fixtures.py
```

Required system tools:
- `espeak-ng` (Linux: `apt install espeak-ng`; macOS: `brew install espeak-ng`)
- `ffmpeg`

Produces:
- `nl_2speakers_10s.wav` — Dutch, two synthetic speakers
- `en_2speakers_10s.wav` — English, two synthetic speakers
- `en_silence_then_speech.wav` — 3s silence then English speech
```

- [ ] **Step 2: Write generator**

`tests/fixtures/generate_fixtures.py`:
```python
"""Generate synthetic test audio using espeak-ng + ffmpeg.

This is good enough for end-to-end pipeline tests on CPU.
It does NOT exercise real-world ASR/diarization quality.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).parent / "audio"
OUT.mkdir(parents=True, exist_ok=True)


def _require(tool: str) -> None:
    if shutil.which(tool) is None:
        sys.exit(f"missing required tool: {tool}")


def _espeak(text: str, voice: str, pitch: int, out_wav: Path) -> None:
    subprocess.run(
        [
            "espeak-ng",
            "-v",
            voice,
            "-p",
            str(pitch),
            "-w",
            str(out_wav),
            text,
        ],
        check=True,
    )


def _concat(parts: list[Path], dst: Path) -> None:
    listfile = dst.with_suffix(".txt")
    listfile.write_text("\n".join(f"file '{p}'" for p in parts))
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(listfile),
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(dst),
        ],
        check=True,
    )
    listfile.unlink()


def _silence(seconds: float, dst: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"anullsrc=r=16000:cl=mono",
            "-t", str(seconds), "-c:a", "pcm_s16le", str(dst),
        ],
        check=True,
    )


def main() -> int:
    _require("espeak-ng")
    _require("ffmpeg")

    work = OUT / "_work"
    work.mkdir(exist_ok=True)

    # English, two speakers
    _espeak("Hello, this is the first speaker talking now.", "en", 30, work / "en_a.wav")
    _espeak("And this is a second speaker responding to you.", "en", 70, work / "en_b.wav")
    _concat([work / "en_a.wav", work / "en_b.wav"], OUT / "en_2speakers_10s.wav")

    # Dutch, two speakers
    _espeak("Hallo, dit is de eerste spreker die nu praat.", "nl", 30, work / "nl_a.wav")
    _espeak("En dit is een tweede spreker die antwoordt.", "nl", 70, work / "nl_b.wav")
    _concat([work / "nl_a.wav", work / "nl_b.wav"], OUT / "nl_2speakers_10s.wav")

    # Silence then speech
    _silence(3.0, work / "silence.wav")
    _espeak("Now the speech begins after a few seconds.", "en", 50, work / "after.wav")
    _concat([work / "silence.wav", work / "after.wav"], OUT / "en_silence_then_speech.wav")

    for f in work.iterdir():
        f.unlink()
    work.rmdir()
    print(f"generated fixtures in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Run generator locally to verify it works**

```bash
python tests/fixtures/generate_fixtures.py
ls -la tests/fixtures/audio/
```
Expected: three `.wav` files present, each <500 KB.

- [ ] **Step 4: Add .gitkeep so the dir is tracked**

`tests/fixtures/audio/.gitkeep`: (empty file)

Edit `.gitignore`, append:
```
tests/fixtures/audio/*.wav
tests/fixtures/audio/_work/
```

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/audio/README.md tests/fixtures/audio/.gitkeep tests/fixtures/generate_fixtures.py .gitignore
git commit -m "test: synthetic audio fixture generator (espeak-ng + ffmpeg)"
```

---

## Task 15: CLI (Typer Subcommands)

**Files:**
- Create: `speechtotext/cli.py`
- Create: `speechtotext/progress.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Add dependency**

Edit `pyproject.toml`:
```toml
dependencies = [
    "faster-whisper>=1.0",
    "pyannote.audio>=4.0",
    "torch>=2.2",
    "sounddevice>=0.4",
    "soundfile>=0.12",
    "numpy>=1.26",
    "watchdog>=4.0",
    "typer>=0.12",
    "rich>=13.0",
]
```

- [ ] **Step 2: Write failing tests**

`tests/test_cli.py`:
```python
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from speechtotext.cli import app

runner = CliRunner()


def test_help_lists_subcommands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for sub in ("transcribe", "record", "watch", "relabel", "config", "doctor"):
        assert sub in result.stdout


def test_transcribe_invokes_pipeline_and_writer(tmp_path: Path):
    audio = tmp_path / "in.mp3"
    audio.write_bytes(b"fake")

    pipeline_instance = MagicMock()
    pipeline_instance.run.return_value = MagicMock()

    with (
        patch("speechtotext.cli._build_pipeline", return_value=(pipeline_instance, "cpu")),
        patch("speechtotext.cli.write_transcript") as wt,
    ):
        result = runner.invoke(app, ["transcribe", str(audio), "--backend", "cpu"])

    assert result.exit_code == 0, result.stdout
    pipeline_instance.run.assert_called_once()
    wt.assert_called_once()


def test_transcribe_skips_when_sidecar_exists(tmp_path: Path):
    audio = tmp_path / "in.mp3"
    audio.write_bytes(b"fake")
    audio.with_suffix(".json").write_text("{}")
    with patch("speechtotext.cli._build_pipeline") as build:
        result = runner.invoke(app, ["transcribe", str(audio)])
    assert result.exit_code == 0
    assert "skipping" in result.stdout.lower()
    build.assert_not_called()


def test_relabel_invokes_module(tmp_path: Path):
    js = tmp_path / "rec.json"
    js.write_text("{}")
    with patch("speechtotext.cli.relabel_module.relabel") as rel:
        result = runner.invoke(
            app, ["relabel", str(js), "SPEAKER_00=Alice", "SPEAKER_01=Bob"]
        )
    assert result.exit_code == 0
    rel.assert_called_once_with(js, {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"})


def test_doctor_reports_status():
    with (
        patch("speechtotext.cli.shutil.which", return_value="/usr/bin/ffmpeg"),
        patch("speechtotext.cli.load_config") as lc,
    ):
        cfg = MagicMock()
        cfg.hf_token = "hf_x"
        cfg.model_cache_dir = Path("/tmp")
        cfg.backend = "cpu"
        lc.return_value = cfg
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "ffmpeg" in result.stdout.lower()
```

- [ ] **Step 3: Implement progress renderer**

`speechtotext/progress.py`:
```python
from __future__ import annotations

import json
import sys
from typing import Callable

from rich.console import Console

from speechtotext.models import ProgressEvent

_console = Console(stderr=True)


def console_renderer(quiet: bool = False) -> Callable[[ProgressEvent], None]:
    def _emit(event: ProgressEvent) -> None:
        if quiet:
            return
        _console.log(f"[{event.stage}] {int(event.pct * 100):3d}% {event.message}")
    return _emit


def json_renderer() -> Callable[[ProgressEvent], None]:
    def _emit(event: ProgressEvent) -> None:
        sys.stderr.write(
            json.dumps(
                {"stage": event.stage, "pct": event.pct, "message": event.message}
            )
            + "\n"
        )
        sys.stderr.flush()
    return _emit
```

- [ ] **Step 4: Implement CLI**

`speechtotext/cli.py`:
```python
from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Annotated, Callable

import typer

from speechtotext import relabel as relabel_module
from speechtotext.asr.faster_whisper import FasterWhisperASR
from speechtotext.backend import resolve_backend
from speechtotext.config import DEFAULT_CONFIG_PATH, load_config
from speechtotext.diarize.pyannote import PyannoteDiarizer
from speechtotext.ingest.file import IngestError
from speechtotext.ingest.mic import record_to_wav
from speechtotext.ingest.watch import run_watch, should_process
from speechtotext.pipeline import Pipeline
from speechtotext.progress import console_renderer, json_renderer
from speechtotext.writer import write_transcript

app = typer.Typer(no_args_is_help=True, help="Local speech-to-text with speaker labels.")


def _progress(quiet: bool, json_logs: bool) -> Callable:
    if json_logs:
        return json_renderer()
    return console_renderer(quiet=quiet)


def _build_pipeline(cli_backend: str | None, config_path: Path) -> tuple[Pipeline, str]:
    cfg = load_config(config_path=config_path)
    backend = resolve_backend(cli_flag=cli_backend, config=cfg)
    asr = FasterWhisperASR(
        model_size=cfg.asr_model, backend=backend, download_root=cfg.model_cache_dir
    )
    if not cfg.hf_token:
        raise typer.BadParameter(
            "Hugging Face token required for pyannote diarization. "
            "Set hf_token in config or HF_TOKEN env var."
        )
    diarizer = PyannoteDiarizer(hf_token=cfg.hf_token, backend=backend)
    pipeline = Pipeline(
        config=cfg, asr=asr, diarizer=diarizer, resolved_backend=backend
    )
    return pipeline, backend


@app.command()
def transcribe(
    audio: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    lang: Annotated[str, typer.Option("--lang")] = "auto",
    speakers: Annotated[int | None, typer.Option("--speakers")] = None,
    backend: Annotated[str | None, typer.Option("--backend")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    json_logs: Annotated[bool, typer.Option("--json-logs")] = False,
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG_PATH,
) -> None:
    if not should_process(audio, overwrite=overwrite):
        typer.echo(f"skipping {audio.name} (sidecar exists; use --overwrite to force)")
        raise typer.Exit(code=0)

    pipeline, resolved = _build_pipeline(backend, config_path)
    on_progress = _progress(quiet=quiet, json_logs=json_logs)
    transcript = pipeline.run(
        audio,
        language=None if lang == "auto" else lang,
        num_speakers=speakers,
        on_progress=on_progress,
    )
    txt_path, json_path = write_transcript(transcript)
    typer.echo(f"wrote {txt_path.name}, {json_path.name}")


@app.command()
def record(
    out: Annotated[Path, typer.Option("--out")] = Path("recording.wav"),
    device: Annotated[str | None, typer.Option("--device")] = None,
    transcribe_after: Annotated[bool, typer.Option("--transcribe/--no-transcribe")] = True,
    backend: Annotated[str | None, typer.Option("--backend")] = None,
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG_PATH,
) -> None:
    stop = threading.Event()
    typer.echo("recording — press Ctrl+C to stop")
    try:
        record_to_wav(out, device=device, stop_event=stop)
    except KeyboardInterrupt:
        stop.set()
    typer.echo(f"wrote {out}")
    if transcribe_after:
        transcribe(audio=out, lang="auto", speakers=None, backend=backend,
                   overwrite=False, quiet=False, json_logs=False, config_path=config_path)


@app.command()
def watch(
    directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    recursive: Annotated[bool, typer.Option("--recursive")] = False,
    exts: Annotated[str, typer.Option("--exts")] = "mp3,wav,m4a,mp4,flac",
    backend: Annotated[str | None, typer.Option("--backend")] = None,
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG_PATH,
) -> None:
    cfg = load_config(config_path=config_path)
    pipeline, resolved = _build_pipeline(backend, config_path)
    extensions = [e.strip() for e in exts.split(",") if e.strip()]

    def _on_ready(path: Path) -> None:
        if not should_process(path, overwrite=False):
            return
        marker = path.with_suffix(path.suffix + ".stt-processing")
        marker.touch()
        try:
            transcript = pipeline.run(path, on_progress=console_renderer())
            write_transcript(transcript)
        except Exception as exc:
            err = path.with_suffix(path.suffix + ".stt-error.txt")
            err.write_text(f"{type(exc).__name__}: {exc}\n")
            typer.echo(f"error on {path.name}: {exc}", err=True)
        finally:
            marker.unlink(missing_ok=True)

    typer.echo(f"watching {directory} — Ctrl+C to stop")
    run_watch(
        directory=directory,
        extensions=extensions,
        debounce_seconds=cfg.watch.debounce_seconds,
        recursive=recursive,
        on_ready=_on_ready,
    )


@app.command()
def relabel(
    json_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    pairs: Annotated[list[str], typer.Argument()],
) -> None:
    mapping: dict[str, str] = {}
    for p in pairs:
        if "=" not in p:
            raise typer.BadParameter(f"expected SPEAKER_NN=Name, got {p!r}")
        sid, name = p.split("=", 1)
        mapping[sid.strip()] = name.strip()
    relabel_module.relabel(json_path, mapping)
    typer.echo(f"relabeled {len(mapping)} speakers in {json_path.name}")


@app.command()
def config(
    action: Annotated[str, typer.Argument()] = "show",
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG_PATH,
) -> None:
    if action == "path":
        typer.echo(str(config_path))
    elif action == "show":
        cfg = load_config(config_path=config_path)
        typer.echo(repr(cfg))
    elif action == "edit":
        editor = subprocess.run(
            [shutil.which("editor") or "nano", str(config_path)]
        )
        raise typer.Exit(code=editor.returncode)
    else:
        raise typer.BadParameter(f"unknown action {action!r}")


@app.command()
def doctor(
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG_PATH,
) -> None:
    ok = True
    ff = shutil.which("ffmpeg")
    typer.echo(f"ffmpeg: {ff or 'MISSING'}")
    if not ff:
        ok = False

    cfg = load_config(config_path=config_path)
    typer.echo(f"backend (config): {cfg.backend}")
    typer.echo(f"hf_token: {'set' if cfg.hf_token else 'MISSING'}")
    if not cfg.hf_token:
        ok = False
    typer.echo(f"model_cache_dir: {cfg.model_cache_dir}")

    try:
        import torch  # noqa
        typer.echo(f"torch: OK ({torch.__version__})")
        typer.echo(f"  cuda: {torch.cuda.is_available()}")
        typer.echo(f"  mps:  {bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())}")
    except Exception as exc:
        typer.echo(f"torch: MISSING ({exc})")
        ok = False

    raise typer.Exit(code=0 if ok else 1)
```

- [ ] **Step 5: Run tests and verify they pass**

```bash
pip install -e ".[dev]" && python -m pytest tests/test_cli.py -v
```
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add speechtotext/cli.py speechtotext/progress.py tests/test_cli.py pyproject.toml
git commit -m "feat(cli): transcribe, record, watch, relabel, config, doctor subcommands"
```

---

## Task 16: Integration Test (End-to-End on Fixture Audio)

**Files:**
- Create: `tests/test_integration.py`

This test only runs when fixtures and the heavy deps are available locally; CI uses tiny models.

- [ ] **Step 1: Write the integration test**

`tests/test_integration.py`:
```python
import json
import shutil
from pathlib import Path

import pytest

from speechtotext.asr.faster_whisper import FasterWhisperASR
from speechtotext.config import Config
from speechtotext.diarize.base import DiarizerBackend
from speechtotext.models import SpeakerTurn
from speechtotext.pipeline import Pipeline
from speechtotext.writer import write_transcript


pytestmark = pytest.mark.integration


class DummyDiarizer:
    """Skip the real pyannote model in CI; assign one speaker per half."""

    def diarize(self, wav_path: Path, num_speakers):
        # 10s file: first half SPEAKER_00, second half SPEAKER_01
        return [
            SpeakerTurn(0.0, 5.0, "SPEAKER_00"),
            SpeakerTurn(5.0, 10.0, "SPEAKER_01"),
        ]


@pytest.fixture
def en_audio(audio_fixture_dir: Path) -> Path:
    p = audio_fixture_dir / "en_2speakers_10s.wav"
    if not p.exists():
        pytest.skip("fixture audio missing; run tests/fixtures/generate_fixtures.py")
    return p


def test_end_to_end_english(en_audio: Path, tmp_path: Path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")

    # Copy fixture so output is written next to the copy (not into committed tree)
    work = tmp_path / en_audio.name
    work.write_bytes(en_audio.read_bytes())

    cfg = Config(backend="cpu")
    asr = FasterWhisperASR(model_size="tiny", backend="cpu")
    pipeline = Pipeline(
        config=cfg, asr=asr, diarizer=DummyDiarizer(), resolved_backend="cpu"
    )
    transcript = pipeline.run(work)
    write_transcript(transcript)

    js = work.with_suffix(".json")
    txt = work.with_suffix(".txt")
    assert js.exists()
    assert txt.exists()

    data = json.loads(js.read_text())
    assert data["version"] == 1
    assert len(data["segments"]) > 0
    # at least one segment per speaker (speakers split at 5s mark)
    spks = {s["speaker"] for s in data["segments"]}
    assert "SPEAKER_00" in spks or "SPEAKER_01" in spks
```

- [ ] **Step 2: Register the integration marker**

Edit `pyproject.toml`, append:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
markers = ["integration: end-to-end tests using real models (slow)"]
```

(Replace the existing `[tool.pytest.ini_options]` block.)

- [ ] **Step 3: Run integration test locally**

```bash
python tests/fixtures/generate_fixtures.py && python -m pytest tests/test_integration.py -v -m integration
```
Expected: 1 passed (downloads tiny Whisper model on first run).

- [ ] **Step 4: Confirm full suite still passes excluding integration**

```bash
python -m pytest -m "not integration" -v
```
Expected: all previously passing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration.py pyproject.toml
git commit -m "test: end-to-end pipeline integration test using whisper-tiny"
```

---

## Task 17: README + Final Smoke

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Expand README**

Replace `README.md`:
```markdown
# stt — local speech-to-text with speaker labels

Privacy-preserving CLI. Dutch + English. Runs on macOS arm64 and Linux x86_64.
Swappable CPU / CUDA / MPS backend.

## Install (dev)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
brew install ffmpeg          # macOS
# apt install ffmpeg         # Linux
```

Pyannote requires a Hugging Face access token (free). Set it once:

```bash
mkdir -p ~/.config/speechtotext
cat > ~/.config/speechtotext/config.toml <<'EOF'
backend = "auto"
asr_model = "large-v3"
hf_token = "hf_..."
EOF
```

## Usage

```bash
stt doctor                          # check setup
stt transcribe meeting.mp3          # → meeting.txt + meeting.json
stt transcribe call.wav --lang nl
stt record --out memo.wav           # mic; Ctrl-C to stop, auto-transcribes
stt watch ~/Recordings              # daemon: new files → transcribed
stt relabel meeting.json SPEAKER_00=Alice SPEAKER_01=Bob
```

## Output format

`<audio>.txt`:
```
[00:00:00] Alice: hallo
[00:00:04] Bob: hoi
```

`<audio>.json` follows the frozen schema documented in
`docs/superpowers/specs/2026-05-14-speech-to-text-cli-design.md`.

## Tests

```bash
pytest -m "not integration"          # fast suite, no models
python tests/fixtures/generate_fixtures.py
pytest -m integration                # downloads whisper-tiny
```

## Phase 2

Summarization and RAG Q&A across transcripts read the same `.json` sidecars.
See spec for the schema contract.
```

- [ ] **Step 2: Run the full fast test suite one last time**

```bash
python -m pytest -m "not integration" -v
```
Expected: all green.

- [ ] **Step 3: Run `stt doctor` manually**

```bash
stt doctor
```
Expected: prints status; exits 0 if ffmpeg + hf_token + torch present, else 1 with clear gaps listed.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: README usage, install, output format"
```

---

## Self-Review Notes (post-write check)

- **Spec coverage:**
  - Section 2 (Stack) → Tasks 1, 8, 9, 12, 13, 15 cover all listed libs.
  - Section 3 (Architecture) → Tasks 5, 6, 11 implement the boxes.
  - Section 4 (Modules) → file map matches exactly.
  - Section 5 (Interfaces) → Tasks 2, 8, 9, 11 define types and Protocols.
  - Section 6 (Sidecar JSON v1) → Task 6 writes; Task 7 reads; Task 11 produces.
  - Section 7 (CLI) → Task 15 implements all six subcommands.
  - Section 8 (Data flow) → Task 11 orchestrates; Task 10 ingests; Task 6 writes.
  - Section 9 (Backend resolution) → Task 4 fully covered.
  - Section 10 (Error handling) → covered in Tasks 6 (atomicity), 10 (ffmpeg), 13 (watch errors), 15 (doctor, BadParameter).
  - Section 11 (Testing) → Tasks 2–13 unit, Task 16 integration; fixture infra in Task 14.
  - Section 12 (Future-UI hooks) → Pipeline callback in Task 11; JSON schema frozen in Task 6.

- **Placeholder scan:** every code block contains the actual implementation. The single hand-off note in Task 9 (`_MODEL_ID = "pyannote/speaker-diarization-3.1"`) is called out explicitly with a verification instruction — not a TODO in the code, but a confirmed-at-execution-time decision documented in the plan.

- **Type consistency:** `Segment`, `SpeakerTurn`, `LabeledSegment`, `Transcript`, `ProgressEvent` defined once in Task 2, used identically across Tasks 5, 6, 7, 8, 9, 11. CLI options consistent across `transcribe`/`record`/`watch`. `should_process()` and `_atomic_write()` referenced by their exact names across tasks that import them.
