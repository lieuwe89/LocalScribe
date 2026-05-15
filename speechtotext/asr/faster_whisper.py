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
