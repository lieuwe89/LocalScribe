from __future__ import annotations

from fastapi import FastAPI

from speechtotext.api.jobs import JobRegistry
from speechtotext.api.routes_jobs import router as jobs_router


def create_app() -> FastAPI:
    app = FastAPI(title="LocalScribe", version="0.1.0")
    app.state.jobs = JobRegistry()
    app.include_router(jobs_router)

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    return app
