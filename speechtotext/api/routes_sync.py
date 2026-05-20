"""Multi-device sync endpoints — snapshot + delta.

Wire shape::

    GET /sync/snapshot
        -> { workspace_id, cursor, transcripts: [...] }

    GET /sync/since/{cursor}
        -> { workspace_id, cursor, transcripts: [...] }

``cursor`` is a numeric value: the largest ``json_mtime`` (unix epoch
seconds, float, sub-second precision where the filesystem supports it)
present in the response. A device reads ``cursor``, stores it locally,
and passes the same value back on the next ``/sync/since`` call to
receive only what changed in between.

Why mtime, not Lamport?
-----------------------

The workspace-wide Lamport counter tracks per-field CRDT op ordering,
but a freshly-transcribed transcript (no edits yet) has all clocks at
0. A Lamport-based cursor would miss those. ``json_mtime`` advances
every time the writer atomically replaces a sidecar, which captures
both new transcripts and CRDT-modified ones uniformly. The two
concepts live side-by-side: clocks order edits *within* a transcript;
mtime orders *which transcripts changed*.

Authorisation
-------------

Both endpoints require a paired-device signature
(:func:`speechtotext.api.auth.verify_device_signature`). Same shape
as ``PATCH /transcripts/{id}``: ``X-Device-Id`` + ``X-Signature-B64``
over METHOD + "\\n" + PATH + "\\n" + body.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from speechtotext.api.auth import verify_device_signature
from speechtotext.api.workspace import get_workspace_id

router = APIRouter()


class SyncResponse(BaseModel):
    workspace_id: str
    cursor: float = Field(
        description="Largest json_mtime present in this response. "
        "Pass back to /sync/since to fetch deltas after.",
    )
    transcripts: list[dict[str, Any]] = Field(
        description="Full transcript JSON docs. Empty if nothing changed."
    )


def _build_delta(request: Request, *, since: float) -> SyncResponse:
    db = request.app.state.library_db
    # Cheap drift check before responding so the response reflects
    # what's actually on disk right now. sync_dirs is fast when
    # nothing has changed (mtime+size short-circuit).
    db.sync_dirs(list(request.app.state.library_dirs))

    rows = db.list_since(since)

    new_cursor = since
    docs: list[dict[str, Any]] = []
    for row in rows:
        path = Path(row["json_path"])
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # Index references a file that's gone or corrupt; skip.
            continue
        docs.append(doc)
        # Advance cursor monotonically. Rows are returned mtime-ASC so
        # the last value wins, but use max() defensively.
        new_cursor = max(new_cursor, float(row["json_mtime"]))

    # If nothing changed but the library has content, surface the
    # library's current max as the cursor so the device's local
    # cursor doesn't lag forever on idle workspaces.
    if not docs:
        max_seen = db.max_mtime()
        new_cursor = max(new_cursor, max_seen)

    return SyncResponse(
        workspace_id=get_workspace_id(),
        cursor=new_cursor,
        transcripts=docs,
    )


@router.get("/sync/snapshot", response_model=SyncResponse)
def sync_snapshot(
    request: Request,
    device_id: str = Depends(verify_device_signature),
) -> SyncResponse:
    """Return every transcript currently in the workspace.

    Used by new devices on first connect; the returned cursor is the
    starting point for subsequent ``/sync/since`` polls.
    """
    return _build_delta(request, since=0.0)


@router.get("/sync/since/{cursor}", response_model=SyncResponse)
def sync_since(
    cursor: float,
    request: Request,
    device_id: str = Depends(verify_device_signature),
) -> SyncResponse:
    """Return transcripts whose mtime is strictly greater than ``cursor``."""
    return _build_delta(request, since=cursor)
