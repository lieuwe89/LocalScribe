from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from speechtotext.api.jobs import JobRegistry
from speechtotext.api.routes_jobs import router as jobs_router
from speechtotext.api.routes_transcripts import router as transcripts_router
from speechtotext.config import load_config


def create_app() -> FastAPI:
    app = FastAPI(title="LocalScribe", version="0.1.0")
    app.state.jobs = JobRegistry()
    app.state.library_dirs: set[Path] = set()

    try:
        _cfg = load_config()
        if _cfg.default_out_dir:
            app.state.library_dirs.add(_cfg.default_out_dir)
    except Exception:
        pass

    app.state.jobs.set_on_complete_dir(app.state.library_dirs.add)

    app.include_router(jobs_router)
    app.include_router(transcripts_router)

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    return app
