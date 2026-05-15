from __future__ import annotations

from fastapi import FastAPI

from speechtotext.api.jobs import JobRegistry


def create_app() -> FastAPI:
    app = FastAPI(title="LocalScribe", version="0.1.0")
    app.state.jobs = JobRegistry()

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    return app
