import json

from agentview.layouts import BY_MARKER
from agentview.sources.gatepin import reviewer_pin

_TIERED, _PLAN = BY_MARKER["_tieredrunner"], BY_MARKER["_planrunner"]


def _pins(tmp_path, data):
    p = tmp_path / "gate-models.json"
    p.write_text(json.dumps(data))
    return p


def test_the_pinned_model_wins_over_any_default(tmp_path):
    p = _pins(tmp_path, {
        "cmag-other": {"model": "gpt-5.6-sol", "effort": "high"},
        "cmag-demo-run": {"model": "gpt-5.4-mini", "effort": "medium",
                          "since": "2026-07-27T09:01:25Z"}})
    got = reviewer_pin(_TIERED, "demo-run", p)
    assert got["model"] == "gpt-5.4-mini"
    assert got["effort"] == "medium"


def test_a_slug_with_no_pin_returns_none(tmp_path):
    p = _pins(tmp_path, {"cmag-other": {"model": "gpt-5.6-sol"},
                         "cmag-third": {"model": "gpt-5.4-mini"}})
    assert reviewer_pin(_TIERED, "demo-run", p) is None


def test_a_layout_with_no_md_tier_never_reads_the_file(tmp_path):
    assert reviewer_pin(_PLAN, "main", tmp_path / "missing.json") is None


def test_a_missing_or_malformed_file_is_not_an_error(tmp_path):
    assert reviewer_pin(_TIERED, "demo-run", tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert reviewer_pin(_TIERED, "demo-run", bad) is None


def test_a_pin_missing_its_model_is_treated_as_absent(tmp_path):
    p = _pins(tmp_path, {"cmag-demo-run": {"effort": "high"},
                         "cmag-other": {"model": "gpt-5.6-sol"}})
    assert reviewer_pin(_TIERED, "demo-run", p) is None


def test_no_pin_file_means_no_implicit_private_machine_read():
    assert reviewer_pin(_TIERED, "demo-run") is None
