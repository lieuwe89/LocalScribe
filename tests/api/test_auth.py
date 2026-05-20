"""Bearer-token auth tests for the sidecar HTTP API.

The Tauri launcher sets LOCALLEXIS_API_TOKEN before spawning the sidecar;
when it is set, every request must carry Authorization: Bearer <token>.
When it is unset (e.g. `stt serve` from the CLI), the API stays anonymous
so standalone use still works.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from speechtotext.api.app import create_app


TOKEN = "test-token-7e9f4a"


@pytest.fixture
def auth_client(monkeypatch):
    monkeypatch.setenv("LOCALLEXIS_API_TOKEN", TOKEN)
    return TestClient(create_app())


@pytest.fixture
def anon_client(monkeypatch):
    monkeypatch.delenv("LOCALLEXIS_API_TOKEN", raising=False)
    return TestClient(create_app())


def test_health_requires_bearer_when_token_env_set(auth_client):
    r = auth_client.get("/health")
    assert r.status_code == 401


def test_health_accepts_matching_bearer(auth_client):
    r = auth_client.get("/health", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_health_rejects_wrong_token(auth_client):
    r = auth_client.get("/health", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_health_rejects_non_bearer_scheme(auth_client):
    r = auth_client.get("/health", headers={"Authorization": f"Basic {TOKEN}"})
    assert r.status_code == 401


def test_health_rejects_missing_authorization_header(auth_client):
    r = auth_client.get("/health")
    assert r.status_code == 401


def test_protected_route_also_requires_bearer(auth_client):
    # Pick a route that doesn't touch state-heavy fixtures.
    r = auth_client.get("/transcripts")
    assert r.status_code == 401


def test_options_preflight_skips_auth(auth_client):
    # CORS preflight must succeed without a bearer header so the browser
    # can learn the allowed methods before issuing the real request.
    r = auth_client.options(
        "/health",
        headers={
            "Origin": "tauri://localhost",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r.status_code in (200, 204)


def test_health_anonymous_when_env_unset(anon_client):
    r = anon_client.get("/health")
    assert r.status_code == 200


def test_protected_route_anonymous_when_env_unset(anon_client):
    r = anon_client.get("/transcripts")
    assert r.status_code == 200
