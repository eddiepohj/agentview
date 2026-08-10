from datetime import datetime, timezone
import pytest
from agentview.events import Event, KINDS, merge, parse_iso


def test_parse_iso_handles_z_suffix_as_utc():
    assert parse_iso("2026-06-25T13:16:50Z") == datetime(
        2026, 6, 25, 13, 16, 50, tzinfo=timezone.utc)


def test_parse_iso_assumes_utc_when_offset_absent():
    assert parse_iso("2026-06-25T13:16:50").tzinfo == timezone.utc


def test_parse_iso_returns_none_for_junk():
    assert parse_iso(None) is None
    assert parse_iso("not a date") is None


def test_known_kinds_include_the_tiered_runner_events():
    for k in ("step.dispatch", "agent.spawn", "gate.open",
              "md.review", "fable.rule", "doc.write"):
        assert k in KINDS


def test_event_rejects_unknown_kind():
    with pytest.raises(ValueError):
        Event(ts=parse_iso("2026-06-25T13:16:50Z"), kind="nope.nope",
              role="step-runner", step=None, payload={},
              artifact_path=None, source="test")


def test_merge_sorts_by_timestamp_and_is_stable():
    a = Event(parse_iso("2026-06-25T10:00:00Z"), "run.start",
              "step-runner", None, {"n": 1}, None, "a")
    b = Event(parse_iso("2026-06-25T09:00:00Z"), "run.start",
              "step-runner", None, {"n": 2}, None, "b")
    c = Event(parse_iso("2026-06-25T10:00:00Z"), "run.start",
              "step-runner", None, {"n": 3}, None, "c")
    assert [e.payload["n"] for e in merge([a, c], [b])] == [2, 1, 3]


# --- Item A: an explicit offset must survive, not be relabelled --------------
#
# `parse_iso` feeds every ledger `updated`, and those windows bucket every
# per-step figure. The two tests above cannot see the one mutation that
# matters: `return d.replace(tzinfo=timezone.utc)` unconditionally still
# parses "Z" correctly and still yields `tzinfo == timezone.utc` for a naive
# stamp, so both stay green while every offset-bearing timestamp silently
# moves by its offset. Nothing in the corpus carries an offset today, which is
# exactly why nothing would notice if it started to.
#
# The assertion has to be on the resulting *instant*. Asserting on `.tzinfo`
# cannot discriminate: the mutant sets it to UTC too.


def test_parse_iso_preserves_a_positive_offset_as_a_distinct_instant():
    """13:16:50+05:00 is 08:16:50Z. Relabelling rather than converting moves
    the instant five hours later."""
    assert parse_iso("2026-06-25T13:16:50+05:00") == datetime(
        2026, 6, 25, 8, 16, 50, tzinfo=timezone.utc)


def test_parse_iso_preserves_a_negative_offset_as_a_distinct_instant():
    """The symmetric case, so a mutant that merely negated the offset -- and
    would satisfy the test above only by luck of sign -- is caught too."""
    assert parse_iso("2026-06-25T13:16:50-05:00") == datetime(
        2026, 6, 25, 18, 16, 50, tzinfo=timezone.utc)


def test_an_offset_stamp_and_its_utc_equivalent_order_identically():
    """The property that actually matters downstream: `merge` and
    `step_windows` sort on these values, so two spellings of one instant must
    compare equal and a genuinely later instant must sort after both."""
    offset = parse_iso("2026-06-25T13:16:50+05:00")
    same = parse_iso("2026-06-25T08:16:50Z")
    later = parse_iso("2026-06-25T09:00:00Z")
    assert offset == same
    assert offset < later
