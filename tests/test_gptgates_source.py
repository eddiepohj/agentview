import json
from agentview.sources.gptgates import read_gpt_rounds, gpt_events


def _rounds(tmp_path):
    d = tmp_path / "gates" / "gpt"
    d.mkdir(parents=True)
    (d / "b-r1.json").write_text(json.dumps({
        "gate": "b", "round": 1, "thread": "cmag-plan",
        "ts": "2026-06-25T13:00:00Z", "outcome": "continue",
        "report": {"risks": [{"id": "R1", "severity": "high",
                              "title": "audio ordering"}]},
        "classification": {"blocking": ["R1"], "advisory": []},
        "tokens": 900}))
    (d / "b-r2.json").write_text(json.dumps({
        "gate": "b", "round": 2, "thread": "cmag-plan",
        "ts": "2026-06-25T13:10:00Z", "outcome": "pass",
        "report": {"risks": []},
        "classification": {"blocking": [], "advisory": []},
        "tokens": 400}))
    return d


def test_rounds_are_read_in_order(tmp_path):
    rows = read_gpt_rounds(_rounds(tmp_path))
    assert [r["round"] for r in rows] == [1, 2]


def test_each_round_emits_md_review(tmp_path):
    evs = gpt_events(_rounds(tmp_path))
    reviews = [e for e in evs if e.kind == "md.review"]
    assert len(reviews) == 2
    assert reviews[0].payload["blocking"] == ["R1"]
    assert reviews[0].role == "md"


def test_escalate_outcome_emits_md_escalate(tmp_path):
    d = _rounds(tmp_path)
    (d / "a-r3.json").write_text(json.dumps({
        "gate": "a", "round": 3, "thread": "cmag-plan",
        "ts": "2026-06-25T14:00:00Z", "outcome": "escalate",
        "report": {"risks": [{"id": "R9", "severity": "critical"}]},
        "classification": {"blocking": ["R9"], "advisory": []}, "tokens": 100}))
    kinds = [e.kind for e in gpt_events(d)]
    assert "md.escalate" in kinds


def test_missing_directory_yields_nothing(tmp_path):
    assert gpt_events(tmp_path / "nope") == []


def test_malformed_round_file_becomes_an_anomaly(tmp_path):
    d = _rounds(tmp_path)
    (d / "b-r3.json").write_text("{broken")
    anomalies = [e for e in gpt_events(d) if e.kind == "anomaly"]
    assert anomalies and anomalies[0].payload["reason"] == "malformed-gate-round"


def test_valid_json_array_becomes_an_anomaly_not_a_crash(tmp_path):
    """A round file can contain syntactically valid JSON that is not an
    object -- e.g. a bare array. json.loads succeeds, so the
    except (OSError, json.JSONDecodeError) clause never fires; the next
    line, data.setdefault(...), would raise AttributeError on a list. This
    must be treated as malformed, not allowed to raise."""
    d = tmp_path / "gates" / "gpt"
    d.mkdir(parents=True)
    (d / "a-r1.json").write_text(json.dumps([1, 2, 3]))
    rows = read_gpt_rounds(d)  # must not raise
    assert rows == [{"_malformed": True, "path": str(d / "a-r1.json")}]
    anomalies = [e for e in gpt_events(d) if e.kind == "anomaly"]
    assert anomalies and anomalies[0].payload["reason"] == "malformed-gate-round"


def test_valid_json_string_becomes_an_anomaly_not_a_crash(tmp_path):
    """Same hazard as the array case, but for a bare JSON string scalar:
    json.loads("\"oops\"") succeeds and returns a str, which also has no
    .setdefault method."""
    d = tmp_path / "gates" / "gpt"
    d.mkdir(parents=True)
    (d / "a-r1.json").write_text(json.dumps("oops"))
    rows = read_gpt_rounds(d)  # must not raise
    assert rows == [{"_malformed": True, "path": str(d / "a-r1.json")}]
    anomalies = [e for e in gpt_events(d) if e.kind == "anomaly"]
    assert anomalies and anomalies[0].payload["reason"] == "malformed-gate-round"


def test_rounds_are_sorted_by_gate_then_round_not_glob_order(tmp_path):
    """a-r2.json and a-r10.json genuinely disagree between lexical filename
    order and (gate, round) order: "a-r10.json" < "a-r2.json" lexically
    (because '1' < '2'), but round 2 must come before round 10. A prior
    version of this test used single-digit rounds on different single-letter
    gates, where lexical filename order always coincided with (gate, round)
    order regardless of write order (sorted(d.glob(...)) sorts by name, not
    by creation time) -- so it passed even with the final
    `sorted(rows, key=...)` line deleted. This fixture does not have that
    problem: only the explicit (gate, round) sort produces [2, 10]."""
    d = tmp_path / "gates" / "gpt"
    d.mkdir(parents=True)
    (d / "a-r2.json").write_text(json.dumps({
        "gate": "a", "round": 2, "thread": "t",
        "ts": "2026-06-25T13:00:00Z", "outcome": "pass",
        "report": {"risks": []},
        "classification": {"blocking": [], "advisory": []}, "tokens": 1}))
    (d / "a-r10.json").write_text(json.dumps({
        "gate": "a", "round": 10, "thread": "t",
        "ts": "2026-06-25T14:00:00Z", "outcome": "pass",
        "report": {"risks": []},
        "classification": {"blocking": [], "advisory": []}, "tokens": 1}))
    rows = read_gpt_rounds(d)
    assert [r["round"] for r in rows] == [2, 10]


def test_round_ten_sorts_after_round_nine_numerically(tmp_path):
    """Round numbers must sort numerically, not lexically: '10' < '9' as
    strings but 9 < 10 as ints. A version that sorted on the raw filename
    string (or on r.get("round") without int conversion from a
    string-typed body value) would put r10 before r9."""
    d = tmp_path / "gates" / "gpt"
    d.mkdir(parents=True)
    for n in (9, 10):
        (d / f"0-r{n}.json").write_text(json.dumps({
            "gate": "0", "round": n, "thread": "t",
            "ts": "2026-06-25T13:00:00Z", "outcome": "pass",
            "report": {"risks": []},
            "classification": {"blocking": [], "advisory": []}, "tokens": 1}))
    rows = read_gpt_rounds(d)
    assert [r["round"] for r in rows] == [9, 10]


def test_rounds_nested_per_step_are_found_with_their_step_id(tmp_path):
    # Two steps, so an implementation reading only the first directory fails.
    #
    # Deviation from the plan's own literal assertion (found during Step 9's
    # implementation, not part of Gate A's reviewed corrections): each round
    # file yields *two* events carrying its step (an `md.review` and a
    # `doc.write` -- unconditional per round, confirmed by
    # `test_each_round_emits_md_review` and the "doc.write for every round"
    # behaviour this step's own brief documents), so
    # `sorted(e.step for e in events)` is `["S4", "S4", "S6", "S6"]`, not
    # `["S4", "S6"]` as the plan's text has it. `set()` restores the assertion
    # to what the comment above says it is testing -- both step directories
    # were actually walked -- without weakening it: an implementation that
    # reads only the first directory still produces `{"S4"}` (or `{"S6"}`)
    # and still fails this.
    for step, gate in (("S4", "b"), ("S6", "b")):
        d = tmp_path / "md" / step
        d.mkdir(parents=True)
        (d / f"{gate}-r1.json").write_text(json.dumps(
            {"gate": gate, "outcome": "pass",
             "report": {"risks": []}, "classification": {"blocking": []}}))
    events = gpt_events(tmp_path / "md", nesting="per-step")
    assert sorted(set(e.step for e in events)) == ["S4", "S6"]


def test_body_gate_and_round_win_over_filename_via_setdefault(tmp_path):
    """The brief's implementation uses data.setdefault("gate", ...) and
    data.setdefault("round", ...): setdefault only fills a key when it is
    ABSENT, so when the JSON body already carries gate/round, those values
    win even if they disagree with the filename. Filed under a-r1.json but
    the body claims gate "b" round 2 - the body should win."""
    d = tmp_path / "gates" / "gpt"
    d.mkdir(parents=True)
    (d / "a-r1.json").write_text(json.dumps({
        "gate": "b", "round": 2, "thread": "t",
        "ts": "2026-06-25T13:00:00Z", "outcome": "pass",
        "report": {"risks": []},
        "classification": {"blocking": [], "advisory": []}, "tokens": 1}))
    rows = read_gpt_rounds(d)
    assert len(rows) == 1
    assert rows[0]["gate"] == "b"
    assert rows[0]["round"] == 2
