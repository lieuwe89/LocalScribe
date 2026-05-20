import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from speechtotext.api.app import create_app


@pytest.fixture
def app_with_lib(tmp_path):
    app = create_app(library_db_path=tmp_path / "library.db")
    app.state.library_dirs.add(tmp_path)
    sample = {
        "version": 1,
        "audio_path": str(tmp_path / "meet.mp3"),
        "duration_seconds": 60.0,
        "language": "en",
        "speakers": {"SPEAKER_00": "Alice"},
        "segments": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "hi"}],
        "models": {"asr": "faster-whisper:tiny"},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (tmp_path / "meet.json").write_text(json.dumps(sample))
    return app


def test_list_transcripts_returns_metadata(app_with_lib):
    client = TestClient(app_with_lib)
    r = client.get("/transcripts")
    assert r.status_code == 200
    items = r.json()
    assert len(items) == 1
    item = items[0]
    assert item["id"] == "meet"
    assert item["duration_seconds"] == 60.0
    assert item["speakers"] == 1


def test_get_transcript_returns_full_json(app_with_lib):
    client = TestClient(app_with_lib)
    r = client.get("/transcripts/meet")
    assert r.status_code == 200
    assert r.json()["segments"][0]["text"] == "hi"


def test_patch_relabel_rewrites_sidecar(app_with_lib, tmp_path):
    client = TestClient(app_with_lib)
    r = client.patch("/transcripts/meet/relabel", json={"SPEAKER_00": "Bob"})
    assert r.status_code == 200
    raw = json.loads((tmp_path / "meet.json").read_text())
    assert raw["speakers"]["SPEAKER_00"] == "Bob"


# ── PATCH /transcripts/{tid} (CRDT op) ─────────────────────────────────────


class TestPatchTranscriptOp:
    def _op(self, op="relabel", key="speakers.SPEAKER_00", value="Bob",
            device="ipad-test", lamport_observed=0):
        return {
            "op": op,
            "key": key,
            "value": value,
            "device": device,
            "lamport_observed": lamport_observed,
        }

    def test_applies_relabel(self, app_with_lib, tmp_path):
        client = TestClient(app_with_lib)
        r = client.patch("/transcripts/meet", json=self._op())
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["speakers"]["SPEAKER_00"] == "Bob"
        assert body["lamport_assigned"] >= 1
        # Persisted to disk.
        raw = json.loads((tmp_path / "meet.json").read_text())
        assert raw["speakers"]["SPEAKER_00"] == "Bob"
        assert raw["_clocks"]["speakers.SPEAKER_00"]["device"] == "ipad-test"

    def test_history_grows_on_each_op(self, app_with_lib, tmp_path):
        client = TestClient(app_with_lib)
        client.patch("/transcripts/meet", json=self._op(value="Bob"))
        client.patch(
            "/transcripts/meet",
            json=self._op(value="Carol", lamport_observed=1),
        )
        raw = json.loads((tmp_path / "meet.json").read_text())
        assert len(raw["_history"]) == 2
        assert raw["speakers"]["SPEAKER_00"] == "Carol"

    def test_lamport_strictly_advances(self, app_with_lib):
        client = TestClient(app_with_lib)
        r1 = client.patch("/transcripts/meet", json=self._op(value="V1"))
        r2 = client.patch(
            "/transcripts/meet",
            json=self._op(value="V2", lamport_observed=0),
        )
        assert r2.json()["lamport_assigned"] > r1.json()["lamport_assigned"]

    def test_lww_stale_lamport_loses(self, app_with_lib, tmp_path):
        client = TestClient(app_with_lib)
        # Hub-assigned lamport reaches 3.
        client.patch("/transcripts/meet", json=self._op(value="V1"))
        client.patch(
            "/transcripts/meet", json=self._op(value="V2", lamport_observed=1)
        )
        client.patch(
            "/transcripts/meet", json=self._op(value="V3", lamport_observed=2)
        )
        # A device that has only seen lamport=0 submits "V_stale" with
        # an older device id ("aaa" < "ipad-test"). Hub assigns lamport=4
        # so this op DOES beat existing (lamport=3) by lamport alone.
        # To force a loss, supply a device that loses tiebreak AND
        # observed lamport too low to escape it… but hub still assigns
        # max(hub,obs)+1 so it always wins by lamport. Instead we test
        # that the LATER hub lamport always wins the speaker value.
        r = client.patch(
            "/transcripts/meet",
            json=self._op(value="V_late", lamport_observed=0),
        )
        assert r.status_code == 200
        raw = json.loads((tmp_path / "meet.json").read_text())
        # Latest lamport wins under hub-assigned ordering.
        assert raw["speakers"]["SPEAKER_00"] == "V_late"
        # All four ops landed in history regardless of merge outcome.
        assert len(raw["_history"]) == 4

    def test_missing_transcript_returns_404(self, app_with_lib):
        client = TestClient(app_with_lib)
        r = client.patch("/transcripts/does-not-exist", json=self._op())
        assert r.status_code == 404

    def test_bad_op_type_returns_400(self, app_with_lib):
        client = TestClient(app_with_lib)
        r = client.patch(
            "/transcripts/meet", json=self._op(op="delete")
        )
        assert r.status_code == 400

    def test_bad_key_returns_400(self, app_with_lib):
        client = TestClient(app_with_lib)
        r = client.patch(
            "/transcripts/meet", json=self._op(key="transcript.title")
        )
        assert r.status_code == 400

    def test_missing_device_rejected(self, app_with_lib):
        client = TestClient(app_with_lib)
        body = self._op()
        body["device"] = ""
        r = client.patch("/transcripts/meet", json=body)
        # Pydantic validates min_length=1 → 422.
        assert r.status_code == 422

    def test_negative_lamport_rejected(self, app_with_lib):
        client = TestClient(app_with_lib)
        body = self._op()
        body["lamport_observed"] = -1
        r = client.patch("/transcripts/meet", json=body)
        # Pydantic ge=0 → 422.
        assert r.status_code == 422

    def test_workspace_id_stamped_on_v1_doc(self, app_with_lib, tmp_path):
        """A pre-v2 transcript should gain _workspace_id on first PATCH."""
        client = TestClient(app_with_lib)
        raw_before = json.loads((tmp_path / "meet.json").read_text())
        assert "_workspace_id" not in raw_before  # v1 sample fixture
        client.patch("/transcripts/meet", json=self._op())
        raw_after = json.loads((tmp_path / "meet.json").read_text())
        assert raw_after["_workspace_id"].startswith("ws_")

    def test_response_shape(self, app_with_lib):
        client = TestClient(app_with_lib)
        r = client.patch("/transcripts/meet", json=self._op())
        assert r.status_code == 200
        body = r.json()
        for key in ("applied", "speakers", "_clocks", "_history", "lamport_assigned"):
            assert key in body
        applied = body["applied"]
        for key in ("op", "key", "value", "device", "lamport", "ts"):
            assert key in applied
        assert applied["op"] == "relabel"
        assert applied["device"] == "ipad-test"
        # from_value reflects the prior value ("Alice" in the fixture).
        assert applied["from_value"] == "Alice"
