from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from speechtotext.api import runner

router = APIRouter()


class TranscribeRequest(BaseModel):
    path: str
    language: str | None = None
    num_speakers: int | None = None
    backend: str | None = None


@router.post("/jobs/transcribe", status_code=202)
def post_transcribe(req: TranscribeRequest, request: Request) -> dict:
    audio = Path(req.path)
    if not audio.exists() or audio.is_dir():
        raise HTTPException(status_code=404, detail=f"audio not found: {audio}")
    registry = request.app.state.jobs
    job_id = registry.create(kind="transcribe", audio_path=str(audio))
    runner.run_transcribe_job(
        registry, job_id, audio,
        language=req.language,
        num_speakers=req.num_speakers,
        backend=req.backend,
    )
    return {"job_id": job_id}


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> dict:
    try:
        rec = request.app.state.jobs.get(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
    return {
        "id": rec.id,
        "kind": rec.kind,
        "status": rec.status.value,
        "stage": rec.stage,
        "percent": rec.percent,
        "error": rec.error,
        "transcript_id": rec.transcript_id,
        "audio_path": rec.audio_path,
        "paths": rec.paths,
    }
