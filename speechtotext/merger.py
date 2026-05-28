from __future__ import annotations

from speechtotext.models import LabeledSegment, Segment, SpeakerTurn

_MIN_OVERLAP_SECONDS = 0.050  # 50ms


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def merge(
    segments: list[Segment], turns: list[SpeakerTurn]
) -> list[LabeledSegment]:
    """Assign each ASR segment the speaker of its max-overlapping turn.

    Linear-time (O(N + M) for sorted input, plus the sort) two-pointer
    sweep: process segments in start order and advance a shared pointer
    past turns that can no longer overlap. Output preserves the caller's
    original segment order regardless of input ordering.
    """
    if not turns:
        return [
            LabeledSegment(
                start=s.start, end=s.end, text=s.text, speaker_id="UNKNOWN"
            )
            for s in segments
        ]

    turns_sorted = sorted(turns, key=lambda t: t.start)
    n = len(turns_sorted)
    # Visit segments in start order, but write results back at their
    # original index so the returned list matches the input order.
    order = sorted(range(len(segments)), key=lambda i: segments[i].start)
    result: list[LabeledSegment] = [None] * len(segments)  # type: ignore[list-item]

    j = 0
    for i in order:
        seg = segments[i]
        # Turns ending at/before this segment's start can't overlap it.
        # Since segments are visited in start order, they can't overlap any
        # later segment either, so advance the shared pointer permanently.
        while j < n and turns_sorted[j].end <= seg.start:
            j += 1
        best_id = "UNKNOWN"
        best_overlap = 0.0
        k = j
        while k < n and turns_sorted[k].start < seg.end:
            ov = _overlap(
                seg.start, seg.end, turns_sorted[k].start, turns_sorted[k].end
            )
            if ov > best_overlap:
                best_overlap = ov
                best_id = turns_sorted[k].speaker_id
            k += 1
        if best_overlap < _MIN_OVERLAP_SECONDS:
            best_id = "UNKNOWN"
        result[i] = LabeledSegment(
            start=seg.start, end=seg.end, text=seg.text, speaker_id=best_id
        )
    return result
