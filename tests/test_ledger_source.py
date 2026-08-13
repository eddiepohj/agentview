import json
from datetime import datetime, timezone
from agentview.discovery import RunRef, find_runs
from agentview.layouts import BY_MARKER
from agentview.model import build_run, _shared_turn_totals
from agentview.sources.ledger import read_ledger, ledger_events, step_windows


def _plan_ledger(tmp_path):
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps({
        "version": 1, "created": "2026-06-25T12:00:00Z",
        "updated": "2026-06-25T14:00:00Z",
        "steps": [
            {"id": "C.1", "owner": "builder", "status": "complete",
             "depends_on": [], "track": "main", "long_running": False,
             "wake": None, "deliverable": "out.md", "note": "did it",
             "tier": "sonnet", "tokens": 1000, "duration_sec": 30,
             "attempts": 2, "updated": "2026-06-25T13:00:00Z"},
            {"id": "C.2", "owner": "builder", "status": "suspended",
             "depends_on": ["C.1"], "track": "soak", "long_running": True,
             "wake": "time>=2026-06-26T06:00:00Z", "deliverable": None,
             "note": None, "tier": None, "tokens": None,
             "duration_sec": None, "attempts": None,
             "updated": "2026-06-25T14:00:00Z"},
        ]}))
    return p


def test_detects_plan_runner_schema(tmp_path):
    assert read_ledger(_plan_ledger(tmp_path))["schema"] == "plan-runner"


def test_detects_light_runner_schema(tmp_path):
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps([{"id": "1", "status": "complete"}]))
    got = read_ledger(p)
    assert got["schema"] == "light-runner"
    assert got["steps"][0]["id"] == "1"


def test_complete_step_emits_step_complete_with_observability(tmp_path):
    evs = ledger_events(_plan_ledger(tmp_path))
    done = [e for e in evs if e.kind == "step.complete"]
    assert done[0].step == "C.1"
    assert done[0].payload["attempts"] == 2
    assert done[0].payload["tier"] == "sonnet"


def test_suspended_step_emits_suspend_with_wake(tmp_path):
    evs = ledger_events(_plan_ledger(tmp_path))
    susp = [e for e in evs if e.kind == "step.suspend"]
    assert susp[0].payload["wake"].startswith("time>=")


def test_step_windows_are_ordered_by_updated(tmp_path):
    w = step_windows(_plan_ledger(tmp_path))
    assert [sid for sid, _ in w] == ["C.1", "C.2"]
    assert w[0][1] < w[1][1]


def test_missing_observability_fields_do_not_crash(tmp_path):
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps({"steps": [{"id": "X", "status": "complete"}]}))
    assert ledger_events(p)[0].payload["attempts"] is None


def test_step_windows_sorts_steps_not_already_in_source_order(tmp_path):
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps({"steps": [
        {"id": "A", "status": "complete", "updated": "2026-06-25T15:00:00Z"},
        {"id": "B", "status": "complete", "updated": "2026-06-25T12:00:00Z"},
        {"id": "C", "status": "complete", "updated": "2026-06-25T13:30:00Z"},
    ]}))
    w = step_windows(p)
    assert [sid for sid, _ in w] == ["B", "C", "A"]
    assert w[0][1] < w[1][1] < w[2][1]


def test_doc_write_emitted_only_for_step_with_deliverable(tmp_path):
    evs = ledger_events(_plan_ledger(tmp_path))
    doc_writes = [e for e in evs if e.kind == "doc.write"]
    # Only C.1 has a deliverable ("out.md"); C.2's deliverable is None.
    assert len(doc_writes) == 1
    assert doc_writes[0].artifact_path == "out.md"
    assert doc_writes[0].step == "C.1"


def test_unparseable_updated_emits_anomaly_and_is_absent_from_step_windows(tmp_path):
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps({"steps": [
        {"id": "M", "status": "complete"},
        {"id": "N", "status": "complete", "updated": "not-a-timestamp"},
        {"id": "OK", "status": "complete", "updated": "2026-06-25T13:00:00Z"},
    ]}))

    evs = ledger_events(p)
    anomalies = {e.step: e for e in evs if e.kind == "anomaly"}
    assert set(anomalies) == {"M", "N"}
    assert anomalies["M"].payload["reason"] == "unparseable-step-updated"
    assert anomalies["M"].payload["updated"] is None
    assert anomalies["N"].payload["reason"] == "unparseable-step-updated"
    assert anomalies["N"].payload["updated"] == "not-a-timestamp"
    assert anomalies["M"].artifact_path == str(p)

    w = step_windows(p)
    assert [sid for sid, _ in w] == ["OK"]


# --- Step 4: dispatches, sessions, fable_rulings, gate_class, answered_by ----


def _run_from(tmp_path, data):
    """Write `data` as a current-schema ledger under a real max-runner layout
    and build the Run from it, so these tests exercise `find_runs` and
    `build_run` end to end rather than only `read_ledger` in isolation."""
    sd = tmp_path / "_maxrunner" / "mr" / "state"
    sd.mkdir(parents=True)
    (sd / "ledger.json").write_text(json.dumps(data))
    return build_run(find_runs(tmp_path)[0], sessions=[])


_V2 = {
    "version": 1,
    "steps": [
        {"id": "S1", "status": "complete", "updated": "2026-07-27T09:00:00Z",
         "tier": "sonnet", "tokens": 41320, "duration_sec": 88.0,
         "attempts": 2, "gate_class": "technical", "answered_by": "director"},
        {"id": "S2", "status": "awaiting-human",
         "updated": "2026-07-27T09:30:00Z", "tokens": None,
         "gate_class": "design", "answered_by": None},
    ],
    "dispatches": [
        {"label": "gate0", "kind": "director", "step": None, "tier": "opus",
         "tokens": 500, "duration_sec": 12.0},
        {"label": "S1 verify", "kind": "verifier", "step": "S1",
         "tier": "haiku", "tokens": 90, "duration_sec": 3.0},
    ],
    "fable_rulings": [{"step": "S1", "ruling": "claude-correct", "note": "n"}],
    "sessions": [{"session": "abc", "closed": ["S1"], "waivers": [],
                  "trigger": "loop"}],
}


def test_the_v2_fields_reach_the_run(tmp_path):
    run = _run_from(tmp_path, _V2)          # helper already in this file
    assert [d["kind"] for d in run.dispatches] == ["director", "verifier"]
    assert [s["trigger"] for s in run.sessions] == ["loop"]
    assert [r["step"] for r in run.fable_rulings] == ["S1"]
    assert run.dispatches[0] == {"label": "gate0", "kind": "director",
                                 "step": None, "tier": "opus", "tokens": 500,
                                 "duration_sec": 12.0}
    assert run.sessions[0] == {"session": "abc", "closed": ["S1"],
                               "waivers": [], "trigger": "loop"}
    assert run.fable_rulings[0] == {"step": "S1", "ruling": "claude-correct",
                                    "note": "n"}
    by_id = {s.id: s for s in run.steps}
    assert by_id["S1"].gate_class == "technical"
    assert by_id["S1"].answered_by == "director"
    assert by_id["S1"].ledger_tokens == 41320
    assert by_id["S2"].gate_class == "design"
    assert by_id["S2"].answered_by is None


def test_a_v1_ledger_still_loads(tmp_path):
    run = _run_from(tmp_path, {"version": 1, "steps": [
        {"id": "1", "status": "complete", "updated": "2026-07-27T09:00:00Z"}]})
    assert run.dispatches == []
    assert run.sessions == []
    assert run.fable_rulings == []
    assert run.steps[0].gate_class is None


def test_malformed_collections_are_dropped_not_fatal(tmp_path):
    run = _run_from(tmp_path, {"version": 1, "steps": [],
                               "dispatches": "not a list",
                               "fable_rulings": [None, {"step": "S1"}]})
    assert run.dispatches == []
    assert run.fable_rulings == [{"step": "S1"}]


def test_malformed_step_tokens_sanitize_to_none_not_a_raise(tmp_path):
    """A ledger is a file agentview does not own: a step's own `tokens` may be
    any JSON value. `Step.ledger_tokens` must be `None` rather than the raw
    garbage, and building the run must not raise."""
    run = _run_from(tmp_path, {"version": 1, "steps": [
        {"id": "S1", "status": "complete",
         "updated": "2026-07-27T09:00:00Z", "tokens": "oops"}]})
    assert run.steps[0].ledger_tokens is None


def test_malformed_dispatch_and_session_values_sanitize_without_raising(
        tmp_path):
    run = _run_from(tmp_path, {"version": 1, "steps": [],
                               "dispatches": [
                                   {"label": "x", "kind": "director",
                                    "step": None, "tier": "opus",
                                    "tokens": [1, 2], "duration_sec": 1}],
                               "sessions": [
                                   {"session": "abc", "closed": "S1",
                                    "waivers": [], "trigger": 7},
                                   {"session": 7, "closed": ["S1"],
                                    "waivers": "oops", "trigger": "loop"}]})
    assert run.dispatches[0]["tokens"] is None
    assert run.sessions[0]["closed"] == []
    assert run.sessions[0]["trigger"] is None
    assert run.sessions[1]["session"] is None
    assert run.sessions[1]["waivers"] == []


def test_session_paths_and_ledger_sessions_are_not_conflated(
        tmp_path, write_jsonl, assistant):
    """`Run.session_paths` (the renamed field, real transcript paths) and
    `Run.sessions` (the ledger's own dispatch-session rows, `list[dict]`)
    must not be conflated by the rename: `_shared_turn_totals` reads only
    `session_paths` -- if it read `sessions` instead it would try
    `set(...)` on unhashable dicts and blow up immediately, rather than
    quietly under- or over-counting."""
    sd = tmp_path / "proj" / "_planrunner" / "state"
    sd.mkdir(parents=True)
    ledger = sd / "ledger.json"
    ledger.write_text(json.dumps({
        "created": "2026-07-27T09:00:00Z",
        "steps": [{"id": "S1", "status": "complete",
                   "updated": "2026-07-27T10:00:00Z"}],
        "sessions": [{"session": "abc", "closed": ["S1"], "waivers": [],
                      "trigger": "loop"}],
    }))
    ref = RunRef(project=tmp_path / "proj", state_dir=sd, ledger=ledger,
                build_log=None, gpt_dir=None, slug="main",
                layout=BY_MARKER["_planrunner"])
    session = write_jsonl("s.jsonl", [
        assistant("2026-07-27T09:30:00Z", out=100),
        assistant("2026-07-27T09:45:00Z", out=200)])

    run = build_run(ref, [session])

    assert run.session_paths == [session]
    assert run.sessions == [{"session": "abc", "closed": ["S1"],
                             "waivers": [], "trigger": "loop"}]

    lo = datetime(2026, 7, 27, 9, 0, tzinfo=timezone.utc)
    hi = datetime(2026, 7, 27, 10, 0, tzinfo=timezone.utc)
    assert _shared_turn_totals(run, run, lo, hi) == (2, 300)
