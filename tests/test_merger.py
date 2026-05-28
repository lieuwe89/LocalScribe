from speechtotext.merger import merge
from speechtotext.models import Segment, SpeakerTurn


def test_single_speaker_single_segment():
    segs = [Segment(0.0, 2.0, "hello", "en")]
    turns = [SpeakerTurn(0.0, 3.0, "SPEAKER_00")]
    out = merge(segs, turns)
    assert len(out) == 1
    assert out[0].speaker_id == "SPEAKER_00"
    assert out[0].text == "hello"


def test_assigns_max_overlap_speaker():
    segs = [Segment(0.0, 4.0, "hi there", "en")]
    turns = [
        SpeakerTurn(0.0, 1.0, "SPEAKER_00"),  # 1s overlap
        SpeakerTurn(1.0, 4.0, "SPEAKER_01"),  # 3s overlap → wins
    ]
    out = merge(segs, turns)
    assert out[0].speaker_id == "SPEAKER_01"


def test_unknown_when_no_overlap():
    segs = [Segment(5.0, 6.0, "orphan", "en")]
    turns = [SpeakerTurn(0.0, 1.0, "SPEAKER_00")]
    out = merge(segs, turns)
    assert out[0].speaker_id == "UNKNOWN"


def test_unknown_when_overlap_below_threshold():
    # 30ms overlap < 50ms threshold
    segs = [Segment(0.0, 1.0, "tiny", "en")]
    turns = [SpeakerTurn(0.97, 5.0, "SPEAKER_00")]
    out = merge(segs, turns)
    assert out[0].speaker_id == "UNKNOWN"


def test_preserves_order_and_text():
    segs = [
        Segment(0.0, 1.0, "first", "en"),
        Segment(1.5, 2.5, "second", "en"),
    ]
    turns = [
        SpeakerTurn(0.0, 1.2, "SPEAKER_00"),
        SpeakerTurn(1.2, 3.0, "SPEAKER_01"),
    ]
    out = merge(segs, turns)
    assert [s.text for s in out] == ["first", "second"]
    assert [s.speaker_id for s in out] == ["SPEAKER_00", "SPEAKER_01"]


def test_empty_turns_yields_all_unknown():
    segs = [Segment(0.0, 1.0, "x", "en")]
    out = merge(segs, [])
    assert out[0].speaker_id == "UNKNOWN"


def test_empty_segments_yields_empty():
    assert merge([], [SpeakerTurn(0.0, 1.0, "SPEAKER_00")]) == []


def test_handles_unsorted_segments_and_spanning_turn():
    # Guards the linear two-pointer sweep: segments are given out of time
    # order and a single long turn spans all of them. Output order must
    # match input order, and every segment must resolve to the turn.
    segs = [
        Segment(5.0, 6.0, "third", "en"),
        Segment(0.0, 1.0, "first", "en"),
        Segment(2.0, 3.0, "second", "en"),
    ]
    turns = [SpeakerTurn(0.0, 10.0, "SPEAKER_00")]
    out = merge(segs, turns)
    assert [s.text for s in out] == ["third", "first", "second"]
    assert all(s.speaker_id == "SPEAKER_00" for s in out)


def test_picks_max_overlap_across_many_turns():
    # A segment overlapping several turns must pick the maximum-overlap
    # one even when it is not the first or last turn by start time.
    segs = [Segment(0.0, 10.0, "long", "en")]
    turns = [
        SpeakerTurn(0.0, 1.0, "S0"),   # 1s
        SpeakerTurn(1.0, 7.0, "S1"),   # 6s → wins
        SpeakerTurn(7.0, 9.0, "S2"),   # 2s
    ]
    out = merge(segs, turns)
    assert out[0].speaker_id == "S1"
