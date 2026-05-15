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
