import asyncio
import pytest

from speechtotext.api.events import StageEvent, LineEvent, CompleteEvent, ErrorEvent
from speechtotext.api.jobs import JobRegistry, JobStatus


@pytest.mark.asyncio
async def test_create_job_returns_unique_id():
    reg = JobRegistry()
    a = reg.create(kind="transcribe", audio_path="/tmp/a.mp3")
    b = reg.create(kind="transcribe", audio_path="/tmp/b.mp3")
    assert a != b
    assert reg.get(a).status == JobStatus.pending


@pytest.mark.asyncio
async def test_publish_and_subscribe():
    reg = JobRegistry()
    job_id = reg.create(kind="transcribe", audio_path="/tmp/x.mp3")
    received: list = []

    async def consumer():
        async for ev in reg.subscribe(job_id):
            received.append(ev)
            if isinstance(ev, CompleteEvent):
                break

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)  # let consumer attach
    await reg.publish(job_id, StageEvent(stage="asr", percent=50))
    await reg.publish(job_id, LineEvent(speaker="SPEAKER_00", ts=1.0, text="hi"))
    await reg.publish(job_id, CompleteEvent(transcript_id="x", paths={"txt": "/tmp/x.txt", "json": "/tmp/x.json"}))
    await asyncio.wait_for(task, timeout=1.0)

    assert [type(e).__name__ for e in received] == ["StageEvent", "LineEvent", "CompleteEvent"]
    assert reg.get(job_id).status == JobStatus.complete


@pytest.mark.asyncio
async def test_error_event_marks_job_failed():
    reg = JobRegistry()
    job_id = reg.create(kind="transcribe", audio_path="/tmp/x.mp3")
    await reg.publish(job_id, ErrorEvent(message="boom"))
    assert reg.get(job_id).status == JobStatus.failed
    assert reg.get(job_id).error == "boom"


def test_registry_prunes_old_terminal_jobs(monkeypatch):
    # A desktop app runs for days; without eviction the jobs dict grows
    # forever. Completed/failed jobs past the cap are dropped oldest-first.
    import speechtotext.api.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "MAX_JOBS", 5)
    reg = JobRegistry()
    old = [reg.create(kind="transcribe") for _ in range(5)]
    for jid in old:
        reg.get(jid).status = JobStatus.complete
    new = [reg.create(kind="transcribe") for _ in range(5)]

    assert len(reg.all()) <= 5
    # Oldest terminal jobs evicted; the fresh (non-terminal) ones survive.
    assert all(reg._jobs.get(jid) is None for jid in old)
    assert all(reg._jobs.get(jid) is not None for jid in new)


def test_registry_keeps_jobs_with_active_subscribers(monkeypatch):
    # A terminal job that still has a live SSE subscriber must not be
    # pruned out from under the stream.
    import speechtotext.api.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "MAX_JOBS", 2)
    reg = JobRegistry()
    keep = reg.create(kind="transcribe")
    reg.get(keep).status = JobStatus.complete
    reg._queues[keep].append(asyncio.Queue())  # simulate an attached stream
    for _ in range(5):
        jid = reg.create(kind="transcribe")
        reg.get(jid).status = JobStatus.complete
    assert reg._jobs.get(keep) is not None
