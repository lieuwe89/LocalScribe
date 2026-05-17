from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class IngestError(RuntimeError):
    pass


# Common ffmpeg install locations on macOS/Linux/Windows. GUI-launched apps on
# macOS get a stripped PATH that excludes Homebrew, so a `which ffmpeg` lookup
# fails even when ffmpeg is installed.
_FFMPEG_FALLBACK_PATHS: tuple[str, ...] = (
    "/opt/homebrew/bin/ffmpeg",       # Apple Silicon Homebrew
    "/usr/local/bin/ffmpeg",          # Intel Homebrew + many manual installs
    "/opt/local/bin/ffmpeg",          # MacPorts
    "/usr/bin/ffmpeg",                # Linux distros
    "C:/Program Files/ffmpeg/bin/ffmpeg.exe",
)


def _resolve_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    for candidate in _FFMPEG_FALLBACK_PATHS:
        if Path(candidate).is_file():
            return candidate
    raise IngestError(
        "ffmpeg not found on PATH. Install via `brew install ffmpeg` (macOS), "
        "your distro's package manager (Linux), or download from "
        "https://ffmpeg.org and add to PATH (Windows)."
    )


def normalize_to_wav(src: Path, dst: Path) -> Path:
    if not src.exists():
        raise IngestError(f"input does not exist: {src}")

    ffmpeg = _resolve_ffmpeg()
    cmd = [
        ffmpeg,
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
        raise IngestError(f"ffmpeg not found at {ffmpeg}") from exc

    if result.returncode != 0:
        raise IngestError(result.stderr.strip() or "ffmpeg conversion failed")
    return dst
