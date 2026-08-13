import json
import types
from pathlib import Path
from agentview.events import Event, parse_iso
from agentview.metrics import (tokens_by_model_family, gate_ratio, fable_count,
                             vocab_drift, waiver_audit, summary,
                             self_report_drift)
from agentview.model import Run, Step, build_run
from agentview.discovery import RunRef, find_runs
from agentview.layouts import BY_MARKER
from agentview.sources.transcript import AgentRun


def _run(events=(), agents=(), orchestrator_tokens=0, tmp_path=None):
    ref = RunRef(project=tmp_path or Path("/x"), state_dir=Path("/x"),
                 ledger=Path("/x/ledger.json"), build_log=None,
                 gpt_dir=None, slug="main", layout=BY_MARKER["_planrunner"])
    run = Run(ref=ref, steps=[], agents=list(agents), events=list(events),
              docs=[], session_paths=[], anomalies=[e for e in events
                                               if e.kind == "anomaly"])
    run.orchestrator_output_tokens = orchestrator_tokens
    return run


def _run_from(tmp_path, data):
    """Write `data` as a current-schema ledger under a real max-runner layout
    and build the Run from it, so gate/fable tests exercise `find_runs` and
    `build_run` end to end. Private copy of test_ledger_source.py's helper of
    the same name -- kept local rather than imported across test modules
    (fragile discovery-order coupling)."""
    sd = tmp_path / "_maxrunner" / "mr" / "state"
    sd.mkdir(parents=True)
    (sd / "ledger.json").write_text(json.dumps(data))
    return build_run(find_runs(tmp_path)[0], sessions=[])


def test_gate_ratio_flags_both_extremes(tmp_path):
    # All four gates answered by a non-human: ratio 0.0 is still an alarm.
    run = _run_from(tmp_path, {"version": 1, "steps": [
        {"id": f"S{i}", "status": "complete", "gate_class": "technical",
         "answered_by": "director"} for i in range(4)]})
    got = gate_ratio(run)
    assert got["opened"] == 4 and got["by_human"] == 0
    assert got["alarm"] is True


def test_gate_ratio_is_calm_in_the_middle(tmp_path):
    run = _run_from(tmp_path, {"version": 1, "steps": [
        {"id": "S0", "status": "complete", "gate_class": "technical",
         "answered_by": "human"},
        {"id": "S1", "status": "complete", "gate_class": "technical",
         "answered_by": "director"},
        {"id": "S2", "status": "complete", "gate_class": "technical",
         "answered_by": None},
        {"id": "S3", "status": "complete", "gate_class": "technical",
         "answered_by": None}]})
    assert gate_ratio(run)["alarm"] is False


def test_fable_count_trips_the_circuit_breaker(tmp_path):
    run = _run_from(tmp_path, {"version": 1, "steps": [], "fable_rulings": [
        {"step": f"S{i}", "ruling": "gpt-correct"} for i in range(3)]})
    got = fable_count(run, threshold=3)
    assert got["count"] == 3 and got["tripped"] is True


def test_vocab_drift_counts_off_vocabulary_statuses():
    evs = [Event(None, "anomaly", "step-runner", "A",
                 {"reason": "off-vocabulary-status", "status": "milestone"},
                 None, "buildlog")]
    got = vocab_drift(_run(events=evs))
    assert got["count"] == 1 and got["statuses"] == {"milestone": 1}


def test_waiver_audit_flags_records_without_a_named_human(tmp_path):
    d = tmp_path / "_planrunner" / "state" / "changes"
    d.mkdir(parents=True)
    (d / "chg-1.md").write_text("# Change: chg-1\nKind: waiver\n"
                                "Authorised by: user (test-user)\n")
    (d / "chg-2.md").write_text("# Change: chg-2\nKind: waiver\n")
    flagged = waiver_audit(tmp_path)
    assert [f["path"].endswith("chg-2.md") for f in flagged] == [True]


# --- Strengthened coverage beyond the brief -----------------------------
#
# The two tests below close gaps the brief's minimum suite leaves open,
# per the task's own instructions to check each test against "what one-line
# change to the implementation would still pass this, and would that be a
# real bug?"


def test_waiver_audit_ignores_a_change_record_that_is_not_a_waiver(tmp_path):
    # The brief's fixture has exactly one non-waiver record, but never
    # asserts on it directly by content -- only implicitly, via the
    # flagged list containing just chg-2. A bug that flagged *every*
    # change record regardless of Kind would still pass the brief's test
    # if there happened to be only one waiver-less-authoriser record, since
    # the non-waiver "chg-0" here has no authoriser line either -- if Kind
    # filtering were broken, this would get flagged too.
    d = tmp_path / "_planrunner" / "state" / "changes"
    d.mkdir(parents=True)
    (d / "chg-0.md").write_text("# Change: chg-0\nKind: scope-change\n")
    (d / "chg-1.md").write_text("# Change: chg-1\nKind: waiver\n"
                                "Authorised by: user (test-user)\n")
    flagged = waiver_audit(tmp_path)
    assert flagged == []


def test_waiver_audit_flags_an_authoriser_line_present_but_empty(tmp_path):
    # Distinguishes "no 'Authorised by:' line at all" (brief's chg-2 case,
    # which fails the `not m` branch) from "the line exists but names no
    # one" (which must fail the separate `not m.group("who").strip()`
    # branch). A regex-only check (`_AUTH.search(text) is None`) without
    # the `.strip()` emptiness check would incorrectly treat this record as
    # having a named authoriser, since the line is present and matches.
    d = tmp_path / "_planrunner" / "state" / "changes"
    d.mkdir(parents=True)
    (d / "chg-3.md").write_text("# Change: chg-3\nKind: waiver\n"
                                "Authorised by:   \n")
    flagged = waiver_audit(tmp_path)
    assert [f["path"].endswith("chg-3.md") for f in flagged] == [True]


# --- Fix round 1 additions ----------------------------------------------
#
# summary() is one of the six required interfaces but nothing called it;
# a mis-wired key (e.g. "gates" bound to fable_count(run) instead of
# gate_ratio(run)) or a shape that made waiver_audit(run.ref.project)
# raise would have shipped undetected. fable_count was only ever tested
# exactly at the threshold, so an unconditionally-tripping breaker or an
# off-by-one threshold would have passed the whole suite.


def test_summary_aggregates_every_metric_under_its_own_key(tmp_path):
    agents = [AgentRun("a1", "general-purpose", "claude-sonnet-4-6", "d",
                       1, 5_000, 10.0)]
    evs = [
        Event(parse_iso("2026-06-25T10:00:00Z"), "gate.open", "director",
              "S0", {}, None, "t"),
        Event(parse_iso("2026-06-25T10:01:00Z"), "gate.answer", "director",
              "S0", {"answered_by": "human"}, None, "t"),
        Event(parse_iso("2026-06-25T10:02:00Z"), "fable.rule", "fable",
              None, {"ruling": "gpt-correct"}, None, "t"),
        Event(None, "anomaly", "step-runner", "A",
              {"reason": "off-vocabulary-status", "status": "milestone"},
              None, "buildlog"),
    ]
    run = _run(events=evs, agents=agents, orchestrator_tokens=1_000,
              tmp_path=tmp_path)
    s = summary(run)

    # Every required key is present, and each maps to exactly what its own
    # dedicated function returns for the same Run -- this catches a
    # mis-wired key (e.g. "gates": fable_count(run)) without duplicating
    # any metric logic here. The four events above are chosen so that
    # tokens_by_model_family, gate_ratio, fable_count, and vocab_drift each
    # return visibly distinct dicts, so a swap between any two keys is
    # guaranteed to fail one of these equality checks.
    assert set(s.keys()) == {"tokens_by_model_family", "gates", "fable",
                             "vocab_drift", "waivers", "steps", "agents",
                             "anomalies"}
    assert s["tokens_by_model_family"] == tokens_by_model_family(run)
    assert s["gates"] == gate_ratio(run)
    assert s["fable"] == fable_count(run)
    assert s["vocab_drift"] == vocab_drift(run)
    assert s["waivers"] == waiver_audit(run.ref.project)
    assert s["steps"] == len(run.steps) == 0
    assert s["agents"] == len(run.agents) == 1
    assert s["anomalies"] == len(run.anomalies) == 1


def test_summary_on_an_empty_run_does_not_raise_and_zeros_the_counts(tmp_path):
    run = _run(events=[], agents=[], orchestrator_tokens=0, tmp_path=tmp_path)
    s = summary(run)
    assert s["steps"] == 0
    assert s["agents"] == 0
    assert s["anomalies"] == 0
    assert s["tokens_by_model_family"] == {"orchestrator": 0}
    assert s["gates"]["opened"] == 0
    assert s["gates"]["alarm"] is False
    assert s["fable"]["count"] == 0
    assert s["fable"]["tripped"] is False
    assert s["vocab_drift"]["count"] == 0
    assert s["waivers"] == []


def test_fable_count_does_not_trip_one_below_threshold(tmp_path):
    run = _run_from(tmp_path, {"version": 1, "steps": [], "fable_rulings": [
        {"step": f"S{i}", "ruling": "gpt-correct"} for i in range(2)]})
    got = fable_count(run, threshold=3)
    assert got["count"] == 2 and got["tripped"] is False


def test_fable_count_trips_one_above_threshold(tmp_path):
    run = _run_from(tmp_path, {"version": 1, "steps": [], "fable_rulings": [
        {"step": f"S{i}", "ruling": "gpt-correct"} for i in range(4)]})
    got = fable_count(run, threshold=3)
    assert got["count"] == 4 and got["tripped"] is True


# --- Step 5: fable_count and gate_ratio read the ledger, not unemitted -----
# events ---------------------------------------------------------------------
#
# v1 counted `fable.rule`/`gate.open`/`gate.answer` events that nothing ever
# emits. Every gate/fable test above this section was rewritten in place to
# ledger-shaped fixtures (via `_run_from`); the tests below are new. The
# ungated step in the second test carries an `answered_by` with no
# `gate_class` (Gate A round-1 R2), so counting answers across every step
# instead of only gated ones is caught by the same assertions.


def test_fable_count_reads_the_ledger_not_an_unemitted_event(tmp_path):
    # Two rulings, so `1 if rulings else 0` fails here.
    run = _run_from(tmp_path, {"version": 1, "steps": [], "fable_rulings": [
        {"step": "S4", "ruling": "claude-correct"},
        {"step": "S6", "ruling": "gpt-correct"}]})
    assert fable_count(run)["count"] == 2
    assert fable_count(run, threshold=2)["tripped"] is True


def test_gate_ratio_reads_gate_class_and_answered_by(tmp_path):
    # Three gates, two answered, one of them by the human: every denominator
    # and numerator differs, so a swapped pair cannot pass.
    run = _run_from(tmp_path, {"version": 1, "steps": [
        {"id": "1", "status": "complete", "gate_class": "technical",
         "answered_by": "director"},
        {"id": "2", "status": "complete", "gate_class": "design",
         "answered_by": "human"},
        {"id": "3", "status": "awaiting-human", "gate_class": "design",
         "answered_by": None},
        {"id": "4", "status": "complete", "answered_by": "human"}]})
    got = gate_ratio(run)
    assert got["opened"] == 3
    assert got["answered"] == 2
    assert got["by_human"] == 1
    assert abs(got["ratio"] - 1 / 3) < 1e-9
    assert got["alarm"] is False


def test_a_run_that_opened_no_gate_has_no_ratio(tmp_path):
    run = _run_from(tmp_path, {"version": 1, "steps": [
        {"id": "1", "status": "complete"}]})
    got = gate_ratio(run)
    assert got["opened"] == 0
    assert got["ratio"] is None
    assert got["alarm"] is False


def test_every_gate_answered_by_the_human_is_an_alarm(tmp_path):
    run = _run_from(tmp_path, {"version": 1, "steps": [
        {"id": "1", "status": "complete", "gate_class": "design",
         "answered_by": "human"},
        {"id": "2", "status": "complete", "gate_class": "design",
         "answered_by": "human"}]})
    assert gate_ratio(run)["alarm"] is True


def test_legacy_events_are_ignored_when_ledger_fields_are_present(tmp_path):
    # A partial migration that kept reading run.events on top of the new
    # ledger fields would double-count against these events; this proves
    # only the ledger-derived figures come back, not a sum of both.
    evs = [Event(parse_iso("2026-06-25T10:00:00Z"), "fable.rule", "fable",
                 None, {"ruling": "gpt-correct"}, None, "t")
           for _ in range(5)]
    evs += [Event(parse_iso("2026-06-25T10:01:00Z"), "gate.open", "director",
                  f"E{i}", {}, None, "t") for i in range(5)]
    evs += [Event(parse_iso("2026-06-25T10:02:00Z"), "gate.answer", "director",
                  f"E{i}", {"answered_by": "human"}, None, "t")
            for i in range(5)]
    ref = RunRef(project=tmp_path, state_dir=tmp_path,
                 ledger=tmp_path / "ledger.json", build_log=None,
                 gpt_dir=None, slug="main", layout=BY_MARKER["_planrunner"])
    steps = [Step(id="S1", status="complete", gate_class="technical",
                  answered_by="director"),
             Step(id="S2", status="complete", gate_class="design",
                  answered_by="human")]
    run = Run(ref=ref, steps=steps, agents=[], events=evs, docs=[],
              session_paths=[], anomalies=[],
              fable_rulings=[{"step": "S4", "ruling": "claude-correct"}])
    assert fable_count(run)["count"] == 1
    got = gate_ratio(run)
    assert got["opened"] == 2
    assert got["answered"] == 2
    assert got["by_human"] == 1


def test_fable_count_stays_zero_when_ledger_is_empty_even_with_legacy_events(
        tmp_path):
    # The test above pairs a NON-empty `fable_rulings` with legacy events, so
    # `rulings = getattr(run, "fable_rulings", []) or []` only proves the
    # ledger value wins when both sides of the `or` are truthy -- a fallback
    # to `[e for e in run.events if e.kind == "fable.rule"]` would short-
    # circuit past undetected. This run has genuinely zero rulings (a clean
    # run) and several stale `fable.rule` events, which is the realistic,
    # security-relevant case a buggy fallback would get wrong.
    evs = [Event(parse_iso("2026-06-25T10:00:00Z"), "fable.rule", "fable",
                 None, {"ruling": "gpt-correct"}, None, "t")
           for _ in range(5)]
    ref = RunRef(project=tmp_path, state_dir=tmp_path,
                 ledger=tmp_path / "ledger.json", build_log=None,
                 gpt_dir=None, slug="main", layout=BY_MARKER["_planrunner"])
    run = Run(ref=ref, steps=[], agents=[], events=evs, docs=[],
              session_paths=[], anomalies=[], fable_rulings=[])
    assert fable_count(run)["count"] == 0
    assert fable_count(run)["tripped"] is False


def test_gate_ratio_stays_at_zero_opened_when_no_step_is_gated_even_with_legacy_events(
        tmp_path):
    # Analogous to the fable test above: the existing gate coverage always
    # pairs gated steps with legacy events, so a fallback to
    # `[s for s in run.steps if getattr(s, "gate_class", None)] or [legacy
    # gate.open/gate.answer events]` would survive undetected. Here no step
    # carries a `gate_class` -- a run that never opened a gate -- while
    # several stale `gate.open`/`gate.answer` events are present.
    evs = [Event(parse_iso("2026-06-25T10:01:00Z"), "gate.open", "director",
                 f"E{i}", {}, None, "t") for i in range(5)]
    evs += [Event(parse_iso("2026-06-25T10:02:00Z"), "gate.answer", "director",
                  f"E{i}", {"answered_by": "human"}, None, "t")
            for i in range(5)]
    ref = RunRef(project=tmp_path, state_dir=tmp_path,
                 ledger=tmp_path / "ledger.json", build_log=None,
                 gpt_dir=None, slug="main", layout=BY_MARKER["_planrunner"])
    steps = [Step(id="S1", status="complete"), Step(id="S2", status="complete")]
    run = Run(ref=ref, steps=steps, agents=[], events=evs, docs=[],
              session_paths=[], anomalies=[], fable_rulings=[])
    got = gate_ratio(run)
    assert got["opened"] == 0
    assert got["ratio"] is None


# --- Step 10: self_report_drift -- the ledger's claim beside the derived ----
# figure -----------------------------------------------------------------
#
# `_step`/`_fake` below are new local helpers, written for this step. The
# plan's own line-1947 comment ("this file's existing test doubles, used
# above since Step 5") is stale/incorrect: neither existed anywhere in this
# repo before this step (recon confirmed with `grep -r "def _step\|def
# _fake" tests/`, and the step-runner independently confirmed by reading
# this file). Gate A's approach record for Step 10 resolves this as a
# correction to the plan's own text, not a scope change.


def _step(id, ledger_tokens=None, derived=None, turns=None):
    """A bare `Step` carrying only the fields `self_report_drift` reads.
    `turns` is always passed through explicitly, including when `None` --
    `Step.turns`'s dataclass default is `field(default_factory=list)` (an
    empty list), so omitting the kwarg here would silently collapse "not
    mentioned" into the same value as `turns=[]` ("no turns observed"),
    destroying exactly the unrecorded/no-attribution distinction this
    step's acceptance criteria require."""
    return Step(id=id, status="complete", ledger_tokens=ledger_tokens,
                output_tokens=derived or 0, turns=turns)


def _fake(steps):
    """`self_report_drift` reads only `run.steps`; a `SimpleNamespace` avoids
    fabricating a `RunRef` and the several other fields a real `Run` requires
    but this metric never touches."""
    return types.SimpleNamespace(steps=steps)


def test_drift_reports_the_ledger_and_the_derived_figure_side_by_side():
    run = _fake(steps=[_step("S1", ledger_tokens=41320, derived=58004),
                       _step("S2", ledger_tokens=1000, derived=1020)])
    rows = {r["step"]: r for r in self_report_drift(run)["steps"]}
    assert rows["S1"]["ledger"] == 41320
    assert rows["S1"]["derived"] == 58004
    assert rows["S1"]["delta"] == 16684
    assert rows["S1"]["divergent"] is True
    # 2% and only 20 tokens: real but not worth a warning.
    assert rows["S2"]["divergent"] is False


def test_an_unrecorded_field_is_not_a_zero():
    run = _fake(steps=[_step("S1", ledger_tokens=None, derived=58004),
                       _step("S2", ledger_tokens=0, derived=0)])
    rows = {r["step"]: r for r in self_report_drift(run)["steps"]}
    assert rows["S1"]["ledger"] is None
    assert rows["S1"]["state"] == "unrecorded"
    # A recorded zero is a claim, and it agrees with the derived zero.
    assert rows["S2"]["state"] == "agrees"


def test_drift_changes_neither_side():
    run = _fake(steps=[_step("S1", ledger_tokens=41320, derived=58004)])
    self_report_drift(run)
    assert run.steps[0].output_tokens == 58004
    assert run.steps[0].ledger_tokens == 41320


def test_a_step_with_nothing_on_either_side_is_not_divergent():
    run = _fake(steps=[_step("S1", ledger_tokens=None, derived=0)])
    row = self_report_drift(run)["steps"][0]
    assert row["divergent"] is False
    assert row["state"] == "unrecorded"


def test_the_summary_counts_only_material_divergence():
    run = _fake(steps=[_step("S1", ledger_tokens=41320, derived=58004),
                       _step("S2", ledger_tokens=1000, derived=1020),
                       _step("S3", ledger_tokens=None, derived=900)])
    got = self_report_drift(run)
    assert got["divergent"] == 1
    assert got["unrecorded"] == 1


def test_a_claimed_zero_against_a_real_derived_figure_is_infinite_not_none():
    # The material case R7 named: a recorded zero that a nonzero derived
    # figure flatly contradicts must not render identically to "no data".
    run = _fake(steps=[_step("S1", ledger_tokens=0, derived=50000)])
    row = self_report_drift(run)["steps"][0]
    assert row["divergent"] is True
    assert row["ratio"] == "infinite"


def test_a_claimed_zero_that_agrees_with_a_derived_zero_is_ratio_one():
    run = _fake(steps=[_step("S1", ledger_tokens=0, derived=0)])
    row = self_report_drift(run)["steps"][0]
    assert row["state"] == "agrees"
    assert row["ratio"] == 1.0


def test_a_claimed_zero_against_a_small_derived_figure_below_the_floor_is_still_infinite():
    # Gate B R1: a claimed zero contradicted by ANY nonzero derived figure is
    # a total, material divergence -- not subject to the floor/rel thresholds
    # that exist for comparing two nonzero magnitudes. derived=500 is below
    # the 1000 floor, so the generic `abs(delta) >= floor` formula alone
    # would (wrongly) call this "agrees" while still rendering ratio
    # "infinite" -- a self-contradictory row.
    run = _fake(steps=[_step("S1", ledger_tokens=0, derived=500)])
    row = self_report_drift(run)["steps"][0]
    assert row["divergent"] is True
    assert row["state"] == "diverges"
    assert row["ratio"] == "infinite"


def test_a_claimed_zero_against_a_derived_figure_exactly_at_the_floor_is_infinite():
    run = _fake(steps=[_step("S1", ledger_tokens=0, derived=1000)])
    row = self_report_drift(run)["steps"][0]
    assert row["divergent"] is True
    assert row["state"] == "diverges"
    assert row["ratio"] == "infinite"


def test_a_claimed_zero_with_real_nonempty_attribution_summing_to_zero_agrees():
    # `test_a_claimed_zero_that_agrees_with_a_derived_zero_is_ratio_one` above
    # never passes `turns`, so it never proves a step with genuine, nonempty
    # attribution (turns present, summing to zero output tokens) is reported
    # as "agrees" rather than accidentally landing in "no-attribution".
    run = _fake(steps=[_step("S1", ledger_tokens=0, derived=0, turns=[100])])
    row = self_report_drift(run)["steps"][0]
    assert row["state"] == "agrees"
    assert row["ratio"] == 1.0


def test_zero_turns_attributed_is_not_a_measured_zero():
    # A step with a ledger figure but no turns found is a different finding
    # from a step whose turns summed to zero -- the derived side has its own
    # "unrecorded" analogue.
    run = _fake(steps=[_step("S1", ledger_tokens=41320, derived=0, turns=[])])
    row = self_report_drift(run)["steps"][0]
    assert row["state"] == "no-attribution"
    assert row["divergent"] is False


# --- Step 10 corrective pass -- skeptic gaps 1-3 -----------------------------
#
# The mandatory skeptic found that the tests above never pin the exact
# `ratio` value for an ordinary (non-zero-claimed) step, never exercise a
# negative `delta` (the real `demo-run` corpus has two: S3 and S5), and never
# hit the `>=` divergence-threshold boundary exactly -- three one-line
# mutations to the implementation that would still pass all 75 pre-existing
# tests. These four tests close those gaps against the shipped, unmodified
# `self_report_drift`.


def test_ratio_is_derived_over_claimed_not_claimed_over_derived():
    # 30000/20000 == 1.5; the swapped form (claimed/derived) is 0.6666...,
    # so the two are unambiguous at a glance and this pins the direction.
    run = _fake(steps=[_step("S1", ledger_tokens=20000, derived=30000)])
    row = self_report_drift(run)["steps"][0]
    assert row["ratio"] == 1.5


def test_a_negative_delta_where_the_ledger_over_claims_still_diverges():
    # The ledger claims more than agentview derives (delta < 0) -- the real
    # demo-run corpus hits this at S3 (delta -4887) and S5 (delta -4555).
    # Dropping `abs()` from the divergence check would misread this 20%
    # over-claim as non-divergent, since a bare `delta >= floor and delta >=
    # rel * biggest` is false for any negative delta.
    run = _fake(steps=[_step("S1", ledger_tokens=50000, derived=40000)])
    row = self_report_drift(run)["steps"][0]
    assert row["delta"] == -10000
    assert row["divergent"] is True
    assert row["state"] == "diverges"


def test_divergence_at_exactly_the_floor_is_inclusive():
    # delta == 1000 exactly, while the relative clause (0.10 * 6000 == 600)
    # is satisfied with margin -- the floor clause alone is at its own
    # boundary. Changing its `>=` to `>` would flip this step to "agrees".
    run = _fake(steps=[_step("S1", ledger_tokens=5000, derived=6000)])
    row = self_report_drift(run)["steps"][0]
    assert row["delta"] == 1000
    assert row["divergent"] is True
    assert row["state"] == "diverges"


def test_divergence_at_exactly_the_relative_threshold_is_inclusive():
    # delta == 2000, biggest == 20000, rel * biggest == 0.10 * 20000 == 2000
    # exactly -- the relative clause is at its own boundary, while the floor
    # clause (2000 >= 1000) is satisfied with margin. Changing the relative
    # clause's `>=` to `>` would flip this step to "agrees".
    run = _fake(steps=[_step("S1", ledger_tokens=18000, derived=20000)])
    row = self_report_drift(run)["steps"][0]
    assert row["delta"] == 2000
    assert row["divergent"] is True
    assert row["state"] == "diverges"
