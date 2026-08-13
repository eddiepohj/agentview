import json
from datetime import datetime, timezone
from agentview.discovery import RunRef
from agentview.layouts import BY_MARKER
from agentview.metrics import tokens_by_model_family, self_report_drift
from agentview.model import (build_run, assign_turns_to_steps,
                           cross_run_anomalies)
from agentview.sources.transcript import Turn


def _utc(h, m=0):
    return datetime(2026, 6, 25, h, m, tzinfo=timezone.utc)


def test_turns_bucket_into_the_step_window_that_closes_after_them():
    windows = [("A", _utc(11)), ("B", _utc(13))]
    turns = [Turn(1, _utc(10), "m", 10, 100),
             Turn(2, _utc(12), "m", 20, 200),
             Turn(3, _utc(12, 30), "m", 30, 300)]
    got = assign_turns_to_steps(windows, turns)
    assert got["A"] == [1]
    assert got["B"] == [2, 3]


def test_turns_after_the_last_window_are_unassigned():
    windows = [("A", _utc(11))]
    turns = [Turn(1, _utc(12), "m", 10, 100)]
    assert assign_turns_to_steps(windows, turns) == {"A": []}


def _run(tmp_path):
    sd = tmp_path / "proj" / "_planrunner" / "state"
    sd.mkdir(parents=True)
    ledger = sd / "ledger.json"
    ledger.write_text(json.dumps({"steps": [
        {"id": "A", "status": "complete", "owner": "builder", "tier": "sonnet",
         "attempts": 1, "deliverable": "a.md", "depends_on": [],
         "updated": "2026-06-25T11:00:00Z"},
        {"id": "B", "status": "halted", "owner": "builder", "tier": "opus",
         "attempts": 2, "deliverable": None, "depends_on": ["A"],
         "updated": "2026-06-25T13:00:00Z"}]}))
    log = tmp_path / "proj" / "BUILD-LOG.md"
    log.write_text("25/06/2026 11:00 | A | complete | did it\n"
                   "25/06/2026 13:00 | B | milestone | drifted\n")
    return RunRef(project=tmp_path / "proj", state_dir=sd, ledger=ledger,
                  build_log=log, gpt_dir=None, slug="main", layout=BY_MARKER["_planrunner"])


def test_build_run_assembles_steps_in_dependency_input_order(tmp_path):
    run = build_run(_run(tmp_path), sessions=[])
    assert [s.id for s in run.steps] == ["A", "B"]
    assert run.steps[1].status == "halted"
    assert run.steps[1].attempts == 2


def test_step_ledger_tier_is_read_from_the_ledgers_own_tier_key(tmp_path):
    """Step 3.7 renamed the Python attribute `Step.tier` -> `Step.ledger_tier`;
    the ledger's own JSON key stays `"tier"` and is still read via
    `raw.get("tier")`. `_run`'s fixture ledger sets step A's `"tier"` to
    `"sonnet"` -- confirm that value actually survives onto the built
    `Step.ledger_tier` attribute, not just the untouched dict literal."""
    run = build_run(_run(tmp_path), sessions=[])
    by_id = {s.id: s for s in run.steps}
    assert by_id["A"].ledger_tier == "sonnet"
    assert by_id["B"].ledger_tier == "opus"


def test_step_ledger_tier_is_none_when_the_ledger_has_no_tier_key(tmp_path):
    """A raw ledger row with no `"tier"` key at all -- not merely an empty
    string -- must leave `Step.ledger_tier` as `None`, the `raw.get` default."""
    sd = tmp_path / "proj" / "_planrunner" / "state"
    sd.mkdir(parents=True)
    ledger = sd / "ledger.json"
    ledger.write_text(json.dumps({"steps": [
        {"id": "A", "status": "complete", "owner": "builder",
         "attempts": 1, "deliverable": None, "depends_on": [],
         "updated": "2026-06-25T11:00:00Z"}]}))
    ref = RunRef(project=tmp_path / "proj", state_dir=sd, ledger=ledger,
                build_log=None, gpt_dir=None, slug="main",
                layout=BY_MARKER["_planrunner"])
    run = build_run(ref, sessions=[])
    assert run.steps[0].ledger_tier is None


def test_build_run_collects_documents(tmp_path):
    run = build_run(_run(tmp_path), sessions=[])
    assert any(d.endswith("a.md") for d in run.docs)


def test_build_run_surfaces_off_vocabulary_statuses_as_anomalies(tmp_path):
    run = build_run(_run(tmp_path), sessions=[])
    reasons = [a.payload.get("reason") for a in run.anomalies]
    assert "off-vocabulary-status" in reasons


def test_events_are_time_ordered(tmp_path):
    run = build_run(_run(tmp_path), sessions=[])
    stamped = [e.ts for e in run.events if e.ts]
    assert stamped == sorted(stamped)


# --- Step 3, R6: orchestrator_models via a real build_run, not just the ------
# hand-built fixture in test_roles.py.


def test_build_run_sets_orchestrator_models_first_seen_deduplicated(
        tmp_path, write_jsonl, assistant):
    """First-seen order, de-duplicated, from `all_turns`. Sonnet appears
    first, then opus, then sonnet again -- alphabetical order would read
    [opus, sonnet], so this distinguishes first-seen from sorted."""
    session = write_jsonl("s.jsonl", [
        assistant("2026-06-25T11:15:00Z", model="claude-sonnet-4-6"),
        assistant("2026-06-25T11:30:00Z", model="claude-opus-4-8"),
        assistant("2026-06-25T11:45:00Z", model="claude-sonnet-4-6"),
    ])
    run = build_run(_run(tmp_path), sessions=[session])
    assert run.orchestrator_models == ["claude-sonnet-4-6", "claude-opus-4-8"]


def test_orchestrator_models_are_chronological_first_seen_not_session_iteration_order(
        tmp_path, write_jsonl, assistant):
    """`_renumber_within_run` computes each turn's chronological `.index` by
    sorting a *local* list, but never reorders `all_turns` itself. Session A's
    single turn is chronologically LATER (11:45) than session B's (11:15), yet
    A is iterated -- and so extended into `all_turns` -- first. A comprehension
    that walks `all_turns` in its original per-session order would see A's
    model (opus) before B's (sonnet), even though sonnet ran first in time. It
    is deliberately session A that is iterated first while holding the later
    timestamp, so session-iteration order and chronological order disagree."""
    session_a = write_jsonl("session-a.jsonl", [
        assistant("2026-06-25T11:45:00Z", model="claude-opus-4-8")])
    session_b = write_jsonl("session-b.jsonl", [
        assistant("2026-06-25T11:15:00Z", model="claude-sonnet-4-6")])

    run = build_run(_run(tmp_path), sessions=[session_a, session_b])

    assert run.orchestrator_models == ["claude-sonnet-4-6", "claude-opus-4-8"]


# --- Strengthened coverage beyond the brief -----------------------------
#
# The four tests below close gaps the brief's minimum suite leaves open:
# step ordering that a coincidental alphabetical/topological sort could
# also satisfy, turn-to-step assignment that a "dump everything in the
# first/last bucket" bug could also satisfy, output_tokens summing turns
# it shouldn't, and a merged event stream that could secretly come from
# only one source.


def _ledger_with_steps(tmp_path, steps, created=None):
    """A ledger for one run. `created` is the run's opening timestamp, which
    every real plan-runner ledger carries; it is what bounds the first step's
    turn window from below."""
    sd = tmp_path / "proj" / "_planrunner" / "state"
    sd.mkdir(parents=True, exist_ok=True)
    ledger = sd / "ledger.json"
    body = {"steps": steps}
    if created:
        body["created"] = created
    ledger.write_text(json.dumps(body))
    return RunRef(project=tmp_path / "proj", state_dir=sd, ledger=ledger,
                  build_log=None, gpt_dir=None, slug="main", layout=BY_MARKER["_planrunner"])


def test_build_run_preserves_ledger_input_order_not_alphabetical_or_topological(tmp_path):
    # Input order is B, C, A. Alphabetical order would be A, B, C.
    # Topological order (B depends_on A) would put A before B.
    # Only "preserve the ledger array's literal order" produces B, C, A.
    steps = [
        {"id": "B", "status": "complete", "owner": "builder", "tier": "sonnet",
         "attempts": 1, "deliverable": None, "depends_on": ["A"],
         "updated": "2026-06-25T11:00:00Z"},
        {"id": "C", "status": "complete", "owner": "builder", "tier": "sonnet",
         "attempts": 1, "deliverable": None, "depends_on": [],
         "updated": "2026-06-25T12:00:00Z"},
        {"id": "A", "status": "complete", "owner": "builder", "tier": "sonnet",
         "attempts": 1, "deliverable": None, "depends_on": [],
         "updated": "2026-06-25T13:00:00Z"},
    ]
    run = build_run(_ledger_with_steps(tmp_path, steps), sessions=[])
    assert [s.id for s in run.steps] == ["B", "C", "A"]


def _session_with_turns(tmp_path, write_jsonl, assistant, name="session.jsonl"):
    records = [
        assistant("2026-06-25T10:30:00Z", out=100),  # -> window A (ends 11:00)
        assistant("2026-06-25T12:00:00Z", out=200),  # -> window B (ends 13:00)
        assistant("2026-06-25T14:00:00Z", out=300),  # -> window C (ends 15:00)
    ]
    return write_jsonl(name, records)


def test_build_run_buckets_turns_and_sums_output_tokens_to_the_correct_step(
        tmp_path, write_jsonl, assistant):
    steps = [
        {"id": "A", "status": "complete", "owner": "builder", "tier": "sonnet",
         "attempts": 1, "deliverable": None, "depends_on": [],
         "updated": "2026-06-25T11:00:00Z"},
        {"id": "B", "status": "complete", "owner": "builder", "tier": "sonnet",
         "attempts": 1, "deliverable": None, "depends_on": [],
         "updated": "2026-06-25T13:00:00Z"},
        {"id": "C", "status": "complete", "owner": "builder", "tier": "sonnet",
         "attempts": 1, "deliverable": None, "depends_on": [],
         "updated": "2026-06-25T15:00:00Z"},
    ]
    # The run opens at 10:00, before the first turn at 10:30, so every turn
    # here is inside the run's span and the bucketing is what is under test.
    ref = _ledger_with_steps(tmp_path, steps, created="2026-06-25T10:00:00Z")
    session = _session_with_turns(tmp_path, write_jsonl, assistant)
    run = build_run(ref, sessions=[session])
    by_id = {s.id: s for s in run.steps}

    # Correct bucketing: each turn lands in exactly one specific step, not
    # merely "some" step.
    assert by_id["A"].turns == [1]
    assert by_id["B"].turns == [2]
    assert by_id["C"].turns == [3]

    # output_tokens sums only the turns bucketed to that step. If the
    # implementation summed every turn into every step, all three would
    # read 600 instead of their own turn's tokens.
    assert by_id["A"].output_tokens == 100
    assert by_id["B"].output_tokens == 200
    assert by_id["C"].output_tokens == 300


def test_build_run_events_include_more_than_one_source_at_once(
        tmp_path, write_jsonl, assistant):
    run = build_run(_run(tmp_path), sessions=[
        _session_with_turns(tmp_path, write_jsonl, assistant)])
    sources = {e.source for e in run.events}
    # ledger + BUILD-LOG + transcript must all be represented in the same
    # merged stream, proving `merge` was actually fed all the streams
    # rather than, say, only the ledger's.
    assert {"ledger", "buildlog", "transcript"} <= sources
    assert [e for e in run.events if e.source == "ledger"]
    assert [e for e in run.events if e.source == "buildlog"]
    assert [e for e in run.events if e.source == "transcript"]


# --- Fix round 2: a run's span, not a session's, bounds its turns ------------
#
# One session can hold two complete runs back to back in a single context
# (the real corpus has exactly this: format-v2 then soak). Selecting the
# session is therefore only half the attribution; the ledger span decides
# which of its turns and agents belong to which run.


def _two_runs_one_session(tmp_path, write_jsonl, assistant):
    """Runs A (10:00-12:00) and B (13:00-15:00) sharing one transcript.

    Turn tokens are chosen so no sum is reachable two ways: A's turns total
    150, B's total 700, everything together 1150. A test asserting on those
    figures can tell "the right turns" from "some turns"."""
    base = tmp_path / "proj" / "_planrunner"
    refs = []
    for name, created, updated in (("state-a", "2026-06-25T10:00:00Z",
                                    "2026-06-25T12:00:00Z"),
                                   ("state-b", "2026-06-25T13:00:00Z",
                                    "2026-06-25T15:00:00Z")):
        sd = base / name
        sd.mkdir(parents=True)
        (sd / "ledger.json").write_text(json.dumps({
            "created": created,
            "steps": [{"id": f"{name}-S1", "status": "complete",
                       "updated": updated}]}))
        refs.append(RunRef(project=tmp_path / "proj", state_dir=sd,
                           ledger=sd / "ledger.json", build_log=None,
                           gpt_dir=None, slug=name, layout=BY_MARKER["_planrunner"]))

    session = write_jsonl("shared.jsonl", [
        assistant("2026-06-25T09:00:00Z", out=300),   # before both runs
        assistant("2026-06-25T11:00:00Z", out=50),    # run A
        assistant("2026-06-25T11:30:00Z", out=100),   # run A
        assistant("2026-06-25T14:00:00Z", out=700),   # run B
    ])
    return refs[0], refs[1], session


def test_one_session_holding_two_runs_is_split_by_ledger_span(
        tmp_path, write_jsonl, assistant):
    """The format-v2/soak case in miniature. Both runs are handed the same
    transcript; each must report only its own turns.

    Without the span filter both report 1150 -- identical figures for two
    different runs, which is the defect this fix exists to remove."""
    a, b, session = _two_runs_one_session(tmp_path, write_jsonl, assistant)

    run_a = build_run(a, sessions=[session])
    run_b = build_run(b, sessions=[session])

    assert run_a.orchestrator_output_tokens == 150
    assert run_b.orchestrator_output_tokens == 700
    assert run_a.orchestrator_output_tokens != run_b.orchestrator_output_tokens


def test_turns_before_a_run_begins_are_not_charged_to_its_first_step(
        tmp_path, write_jsonl, assistant):
    """The 300-token turn at 09:00 predates run A entirely. Bounding step
    windows only from above hands it to A's first step, which is how two
    runs came to report the same `638 turns / 1,070,164 tokens` for their
    respective first steps."""
    a, _b, session = _two_runs_one_session(tmp_path, write_jsonl, assistant)
    step = build_run(a, sessions=[session]).steps[0]

    assert step.output_tokens == 150
    assert len(step.turns) == 2


def test_assign_turns_to_steps_bounds_the_first_window_from_below():
    """`assign_turns_to_steps` must be correct called directly, not only via
    `build_run`. The 09:00 turn is inside window A from above but before the
    run started at 10:00.

    Drop the lower bound and A collects both turns."""
    windows = [("A", _utc(11)), ("B", _utc(13))]
    turns = [Turn(1, _utc(9), "m", 10, 100),
             Turn(2, _utc(10, 30), "m", 20, 200)]
    got = assign_turns_to_steps(windows, turns, _utc(10))
    assert got["A"] == [2]
    assert got["B"] == []


def test_assign_turns_to_steps_without_a_start_is_unbounded_below():
    """Omitting `start` must preserve the previous behaviour, so callers
    that cannot determine a run start are not silently emptied out."""
    windows = [("A", _utc(11))]
    turns = [Turn(1, _utc(9), "m", 10, 100)]
    assert assign_turns_to_steps(windows, turns) == {"A": [1]}


def _agent(sub, agent_id, ts, out):
    (sub / f"agent-{agent_id}.meta.json").write_text(json.dumps(
        {"agentType": "Explore", "description": agent_id,
         "toolUseId": f"toolu_{agent_id}", "spawnDepth": 1}))
    (sub / f"agent-{agent_id}.jsonl").write_text(json.dumps(
        {"type": "assistant", "timestamp": ts,
         "message": {"model": "claude-sonnet-4-6",
                     "usage": {"output_tokens": out, "input_tokens": 1,
                               "cache_read_input_tokens": 0,
                               "cache_creation_input_tokens": 0},
                     "content": []}}) + "\n")


def test_agents_are_attributed_by_span_too(tmp_path, write_jsonl, assistant):
    """`tokens_by_model_family` reads `run.agents`, so an unfiltered agent pool makes
    every run in a project report the same worker spend -- the corpus showed
    the identical `31 agents` for three different runs. Each run must see
    only the agents spawned inside its own span."""
    a, b, session = _two_runs_one_session(tmp_path, write_jsonl, assistant)
    sub = session.parent / session.stem / "subagents"
    sub.mkdir(parents=True)
    _agent(sub, "early", "2026-06-25T09:10:00Z", 11)   # before both runs
    _agent(sub, "in-a", "2026-06-25T11:10:00Z", 22)    # run A
    _agent(sub, "in-b", "2026-06-25T14:10:00Z", 33)    # run B

    run_a, run_b = build_run(a, [session]), build_run(b, [session])
    assert [x.agent_id for x in run_a.agents] == ["in-a"]
    assert [x.agent_id for x in run_b.agents] == ["in-b"]
    assert tokens_by_model_family(run_a)["sonnet"] == 22
    assert tokens_by_model_family(run_b)["sonnet"] == 33


# --- Fix wave, item 3 (ESC-2): Turn.index collided across sessions -----------
#
# `read_turns` numbers from 1 per session, which is right for its own
# contract. `build_run` then keyed `by_index` on that across *all* of a run's
# sessions, so two sessions' turn 1 collided and the later silently replaced
# the earlier. A run that suspends and resumes spans two sessions -- which is
# plan-runner's normal mode -- so this is not hypothetical.


def _two_session_turn_index_collision_run(tmp_path, write_jsonl, assistant):
    """One run, two sessions, two steps.

    Session A's turns fall in step S1's window, session B's in S2's. Both
    sessions number their turns 1 and 2. The token values are chosen so that
    every wrong answer is a different number from every right one:
    S1 = 1000+300 = 1300, S2 = 2000+70 = 2070, run total 3370. Under the
    collision `by_index` holds only session B's turns, so *both* steps read
    2070 and S1 reports session B's spend as its own."""
    sd = tmp_path / "proj" / "_planrunner" / "state"
    sd.mkdir(parents=True)
    ledger = sd / "ledger.json"
    ledger.write_text(json.dumps({
        "created": "2026-06-25T10:00:00Z",
        "steps": [
            {"id": "S1", "status": "complete", "updated":
             "2026-06-25T11:59:00Z"},
            {"id": "S2", "status": "complete", "updated":
             "2026-06-25T13:00:00Z"}]}))
    ref = RunRef(project=tmp_path / "proj", state_dir=sd, ledger=ledger,
                 build_log=None, gpt_dir=None, slug="main", layout=BY_MARKER["_planrunner"])
    first = write_jsonl("before-suspend.jsonl", [
        assistant("2026-06-25T11:00:00Z", out=1000),
        assistant("2026-06-25T11:30:00Z", out=300)])
    second = write_jsonl("after-resume.jsonl", [
        assistant("2026-06-25T12:00:00Z", out=2000),
        assistant("2026-06-25T12:30:00Z", out=70)])
    return ref, [first, second]


def test_two_sessions_do_not_collide_on_turn_index(tmp_path, write_jsonl,
                                                   assistant):
    ref, sessions = _two_session_turn_index_collision_run(
        tmp_path, write_jsonl, assistant)
    by_id = {s.id: s for s in build_run(ref, sessions).steps}

    assert by_id["S1"].output_tokens == 1300
    assert by_id["S2"].output_tokens == 2070


def test_a_runs_turn_indices_are_unique_across_its_sessions(tmp_path,
                                                            write_jsonl,
                                                            assistant):
    """The property behind the figures above, asserted directly. Summing to
    the right totals could in principle happen with duplicate indices and a
    compensating error; the indices themselves must be distinct."""
    ref, sessions = _two_session_turn_index_collision_run(
        tmp_path, write_jsonl, assistant)
    steps = build_run(ref, sessions).steps
    every = [i for s in steps for i in s.turns]

    assert sorted(every) == [1, 2, 3, 4]
    assert len(set(every)) == len(every)


def test_a_single_session_run_is_unaffected_by_renumbering(tmp_path,
                                                           write_jsonl,
                                                           assistant):
    """The renumbering must be a no-op for the single-session case, which is
    every run that never suspended."""
    sd = tmp_path / "proj" / "_planrunner" / "state"
    sd.mkdir(parents=True)
    ledger = sd / "ledger.json"
    ledger.write_text(json.dumps({
        "created": "2026-06-25T10:00:00Z",
        "steps": [{"id": "S1", "status": "complete",
                   "updated": "2026-06-25T13:00:00Z"}]}))
    ref = RunRef(project=tmp_path / "proj", state_dir=sd, ledger=ledger,
                 build_log=None, gpt_dir=None, slug="main", layout=BY_MARKER["_planrunner"])
    session = write_jsonl("only.jsonl", [
        assistant("2026-06-25T11:00:00Z", out=1000),
        assistant("2026-06-25T11:30:00Z", out=300),
        assistant("2026-06-25T12:00:00Z", out=70)])

    step = build_run(ref, [session]).steps[0]
    assert step.turns == [1, 2, 3]
    assert step.output_tokens == 1370


# --- Fix wave, item 4 (C1): overlapping runs double-count in silence ---------
#
# The attribution arithmetic is deliberately unchanged. agentview validates
# runs and hunts blind spots; when a number cannot be clean it says so
# instead of quietly producing a tidier one.


def _ref_with_span(tmp_path, name, created, updated, slug=None):
    sd = tmp_path / "proj" / "_planrunner" / name
    sd.mkdir(parents=True)
    (sd / "ledger.json").write_text(json.dumps({
        "created": created,
        "steps": [{"id": f"{name}-S1", "status": "complete",
                   "updated": updated}]}))
    return RunRef(project=tmp_path / "proj", state_dir=sd,
                  ledger=sd / "ledger.json", build_log=None, gpt_dir=None,
                  slug=slug or name, layout=BY_MARKER["_planrunner"])


def _reasons(events):
    return [e.payload.get("reason") for e in events]


def test_overlapping_runs_over_a_shared_session_report_the_shared_counts(
        tmp_path, write_jsonl, assistant):
    """The corpus case in miniature: the operator started the next run while
    still closing out the last one's final steps, one session at a time.

    Only the two turns inside *both* spans are double-counted; the turn
    before the overlap and the turn after it belong to exactly one run each,
    so a naive "all shared-session turns" count would say 4 turns / 1,000
    tokens instead of 2 / 500."""
    a = _ref_with_span(tmp_path, "state-a", "2026-06-25T10:00:00Z",
                       "2026-06-25T13:00:00Z")
    b = _ref_with_span(tmp_path, "state-b", "2026-06-25T12:00:00Z",
                       "2026-06-25T15:00:00Z")
    session = write_jsonl("shared.jsonl", [
        assistant("2026-06-25T11:00:00Z", out=100),   # run A only
        assistant("2026-06-25T12:30:00Z", out=200),   # both
        assistant("2026-06-25T12:45:00Z", out=300),   # both
        assistant("2026-06-25T14:00:00Z", out=400)])  # run B only

    runs = [build_run(a, [session]), build_run(b, [session])]
    events = cross_run_anomalies(runs)

    assert _reasons(events) == ["run-span-overlap", "run-span-overlap"]
    assert {e.payload["slug"] for e in events} == {"state-a", "state-b"}
    assert {e.payload["other_slug"] for e in events} == {"state-a", "state-b"}
    for e in events:
        assert e.payload["overlap_seconds"] == 3600.0
        assert e.payload["shared_turns"] == 2
        assert e.payload["shared_output_tokens"] == 500


def test_non_overlapping_runs_emit_no_overlap_anomaly(tmp_path, write_jsonl,
                                                      assistant):
    a = _ref_with_span(tmp_path, "state-a", "2026-06-25T10:00:00Z",
                       "2026-06-25T11:00:00Z")
    b = _ref_with_span(tmp_path, "state-b", "2026-06-25T12:00:00Z",
                       "2026-06-25T13:00:00Z")
    session = write_jsonl("shared.jsonl", [
        assistant("2026-06-25T10:30:00Z", out=100),
        assistant("2026-06-25T12:30:00Z", out=200)])

    runs = [build_run(a, [session]), build_run(b, [session])]
    assert cross_run_anomalies(runs) == []


def test_spans_that_merely_touch_are_not_an_overlap(tmp_path):
    """One run ending exactly as the next begins shares no instant of work.
    A `>=` comparison would report a zero-second overlap on every clean
    hand-off and make the warning meaningless."""
    a = _ref_with_span(tmp_path, "state-a", "2026-06-25T10:00:00Z",
                       "2026-06-25T12:00:00Z")
    b = _ref_with_span(tmp_path, "state-b", "2026-06-25T12:00:00Z",
                       "2026-06-25T14:00:00Z")
    assert cross_run_anomalies([build_run(a, []), build_run(b, [])]) == []


def test_a_single_run_never_produces_a_cross_run_anomaly(tmp_path):
    """`build_run` sees one run and cannot know about siblings, which is why
    this lives outside it. One run in hand means nothing to compare."""
    a = _ref_with_span(tmp_path, "state-a", "2026-06-25T10:00:00Z",
                       "2026-06-25T13:00:00Z")
    run = build_run(a, [])
    assert cross_run_anomalies([run]) == []
    assert "run-span-overlap" not in _reasons(run.anomalies)


def _run_with_steps_and_session(tmp_path, updates, stamps, write_jsonl,
                                assistant, created="2026-06-25T10:00:00Z"):
    sd = tmp_path / "proj" / "_planrunner" / "state"
    sd.mkdir(parents=True)
    (sd / "ledger.json").write_text(json.dumps({
        "created": created,
        "steps": [{"id": sid, "status": "complete", "updated": u}
                  for sid, u in updates]}))
    ref = RunRef(project=tmp_path / "proj", state_dir=sd,
                 ledger=sd / "ledger.json", build_log=None, gpt_dir=None,
                 slug="main", layout=BY_MARKER["_planrunner"])
    session = write_jsonl("s.jsonl", [assistant(ts, out=100) for ts in stamps])
    return build_run(ref, [session])


def test_coverage_starting_long_after_the_span_start_is_reported(
        tmp_path, write_jsonl, assistant):
    """`main` in the real corpus: its span opens at 14:56 on 06-24 but the
    transcript covering its first 21 hours is absent from ~/.claude/projects,
    so every figure it reports is missing that work. The number is left
    alone; the gap is stated."""
    run = _run_with_steps_and_session(
        tmp_path, [("A", "2026-06-25T15:00:00Z")],
        ["2026-06-25T14:00:00Z"], write_jsonl, assistant)
    late = [a for a in run.anomalies
            if a.payload.get("reason") == "coverage-starts-late"]

    assert len(late) == 1
    assert late[0].payload["gap_seconds"] == 4 * 3600
    assert late[0].payload["span_start"].startswith("2026-06-25T10:00")
    assert late[0].payload["first_turn_ts"].startswith("2026-06-25T14:00")


def test_coverage_that_begins_promptly_is_not_reported(tmp_path, write_jsonl,
                                                       assistant):
    """Healthy runs in the corpus start 5-23 seconds after their span opens.
    A threshold that flagged those would make the anomaly noise."""
    run = _run_with_steps_and_session(
        tmp_path, [("A", "2026-06-25T15:00:00Z")],
        ["2026-06-25T10:00:20Z"], write_jsonl, assistant)
    assert "coverage-starts-late" not in _reasons(run.anomalies)


def test_every_turn_landing_in_one_step_is_reported(tmp_path, write_jsonl,
                                                    assistant):
    """`main` again: 213 turns in step F.3 while its other 24 steps read 0.
    Per-step figures that concentrate like that are not a split at all."""
    run = _run_with_steps_and_session(
        tmp_path,
        [("A", "2026-06-25T11:00:00Z"), ("B", "2026-06-25T12:00:00Z"),
         ("C", "2026-06-25T13:00:00Z")],
        ["2026-06-25T10:00:10Z", "2026-06-25T10:00:20Z",
         "2026-06-25T10:00:30Z"], write_jsonl, assistant)
    hit = [a for a in run.anomalies
           if a.payload.get("reason") == "all-turns-in-one-step"]

    assert len(hit) == 1
    assert hit[0].payload["step"] == "A"
    assert hit[0].payload["turns"] == 3


def test_turns_spread_over_several_steps_are_not_reported(tmp_path,
                                                          write_jsonl,
                                                          assistant):
    run = _run_with_steps_and_session(
        tmp_path,
        [("A", "2026-06-25T11:00:00Z"), ("B", "2026-06-25T12:00:00Z"),
         ("C", "2026-06-25T13:00:00Z")],
        ["2026-06-25T10:00:10Z", "2026-06-25T11:30:00Z"],
        write_jsonl, assistant)
    assert "all-turns-in-one-step" not in _reasons(run.anomalies)


def test_a_one_step_run_is_not_reported_as_concentrated(tmp_path, write_jsonl,
                                                        assistant):
    """A run with a single step has nowhere else for its turns to go; saying
    they all landed in one step would be true and useless."""
    run = _run_with_steps_and_session(
        tmp_path, [("A", "2026-06-25T15:00:00Z")],
        ["2026-06-25T10:00:10Z"], write_jsonl, assistant)
    assert "all-turns-in-one-step" not in _reasons(run.anomalies)


# --- Fix wave, item 5 (C4): the span model made the live pane blind ----------


def _in_progress_run(tmp_path):
    sd = tmp_path / "proj" / "_planrunner" / "state"
    sd.mkdir(parents=True)
    (sd / "ledger.json").write_text(json.dumps({
        "created": "2026-06-25T10:00:00Z",
        "steps": [
            {"id": "A", "status": "complete", "updated":
             "2026-06-25T11:00:00Z"},
            {"id": "B", "status": "in-progress", "updated":
             "2026-06-25T12:00:00Z"}]}))
    return RunRef(project=tmp_path / "proj", state_dir=sd,
                  ledger=sd / "ledger.json", build_log=None, gpt_dir=None,
                  slug="main", layout=BY_MARKER["_planrunner"])


def _in_flight_session(write_jsonl, assistant):
    return write_jsonl("live.jsonl", [
        assistant("2026-06-25T10:30:00Z", out=300),     # inside the ledger span
        assistant("2026-06-25T13:00:00Z", out=12000)])  # after the last write


def test_live_true_counts_turns_after_the_last_ledger_write(tmp_path,
                                                            write_jsonl,
                                                            assistant):
    """`run_span` closes at the latest step `updated`, which is the last time
    the orchestrator wrote state -- not the moment work stopped. Everything
    happening right now is after it, so the pane showed 300 output tokens
    where the session held 12,300."""
    ref = _in_progress_run(tmp_path)
    run = build_run(ref, [_in_flight_session(write_jsonl, assistant)],
                    live=True)
    assert run.orchestrator_output_tokens == 12300


def test_live_false_still_excludes_them(tmp_path, write_jsonl, assistant):
    """Reports keep the retrospective span exactly as it was; only the live
    pane asks for the open upper bound."""
    ref = _in_progress_run(tmp_path)
    run = build_run(ref, [_in_flight_session(write_jsonl, assistant)])
    assert run.orchestrator_output_tokens == 300


def test_the_in_flight_step_receives_the_turns_produced_since_its_own_update(
        tmp_path, write_jsonl, assistant):
    """Widening the run's span is not enough on its own: every step window
    closes at that step's `updated` too, so the turns still fell off the end
    of the window list and the step running right now showed zero."""
    ref = _in_progress_run(tmp_path)
    session = _in_flight_session(write_jsonl, assistant)

    live = {s.id: s for s in build_run(ref, [session], live=True).steps}
    assert live["B"].output_tokens == 12000
    assert live["A"].output_tokens == 300

    retro = {s.id: s for s in build_run(ref, [session]).steps}
    assert retro["B"].turns == []


def test_a_ledger_with_no_parseable_updated_says_its_span_is_unbounded(
        tmp_path, write_jsonl, assistant):
    """A freshly started run has no completed step yet, so nothing bounds it
    from above. It renders, and it says why its figures are open-ended."""
    sd = tmp_path / "proj" / "_planrunner" / "state"
    sd.mkdir(parents=True)
    (sd / "ledger.json").write_text(json.dumps({
        "created": "2026-06-25T10:00:00Z",
        "steps": [{"id": "A", "status": "in-progress", "updated": None}]}))
    ref = RunRef(project=tmp_path / "proj", state_dir=sd,
                 ledger=sd / "ledger.json", build_log=None, gpt_dir=None,
                 slug="main", layout=BY_MARKER["_planrunner"])
    session = write_jsonl("s.jsonl", [assistant("2026-06-25T10:30:00Z",
                                                out=777)])

    run = build_run(ref, [session])
    assert "unbounded-span" in _reasons(run.anomalies)
    assert run.orchestrator_output_tokens == 777


def test_live_mode_names_the_turns_that_the_report_will_not_count(
        tmp_path, write_jsonl, assistant):
    """Deviation from the brief, stated as a test. Opening the upper bound is
    correct for the pane, but it makes the live figure exceed the report's --
    by 8x on one real run -- and a bare "provisional" label does not explain
    that. The pane names the turns responsible."""
    ref = _in_progress_run(tmp_path)
    run = build_run(ref, [_in_flight_session(write_jsonl, assistant)],
                    live=True)
    open_span = [a for a in run.anomalies
                 if a.payload.get("reason") == "live-span-open"]

    assert len(open_span) == 1
    assert open_span[0].payload["turns"] == 1
    assert open_span[0].payload["output_tokens"] == 12000
    assert open_span[0].payload["since"].startswith("2026-06-25T12:00")


# A test asserting that a *retrospective* run lacks `live-span-open` was
# written here and then removed: it cannot fail. With `live=False` the run's
# turns are already filtered to `[start, end]`, so none can be later than the
# last ledger write and the warning is unreachable by construction. Deleting
# the `if live` guard in `build_run` leaves such a test green -- verified by
# mutation. `test_live_false_still_excludes_them` above pins the property
# that actually carries the weight: the retrospective span excludes those
# turns from the figures entirely.


def test_a_live_run_with_no_turns_past_the_last_write_stays_quiet(
        tmp_path, write_jsonl, assistant):
    """Nothing to explain means nothing is said, or the warning stops
    carrying information."""
    ref = _in_progress_run(tmp_path)
    session = write_jsonl("live.jsonl", [
        assistant("2026-06-25T10:30:00Z", out=300)])
    run = build_run(ref, [session], live=True)
    assert "live-span-open" not in _reasons(run.anomalies)


# --- Residual R1: `unbounded-span` must reach the live pane too --------------
#
# `_live_windows` appends a synthetic open-ended window for the in-flight
# step, and the unboundedness check was reading the list *after* that. A fresh
# run -- no parseable `updated` anywhere, which is exactly the case this
# anomaly exists for -- therefore reported it retrospectively and went silent
# in the pane, the one surface item 5 is about. Span-unboundedness is a
# property of the ledger's own windows, not of whether a synthetic one was
# appended.


def _fresh_run(tmp_path):
    """A run that has started and completed nothing: no step carries a
    parseable `updated`, so nothing bounds its span from above."""
    sd = tmp_path / "proj" / "_planrunner" / "state"
    sd.mkdir(parents=True)
    (sd / "ledger.json").write_text(json.dumps({
        "created": "2026-06-25T10:00:00Z",
        "steps": [{"id": "A", "status": "in-progress", "updated": None},
                  {"id": "B", "status": "pending", "updated": None}]}))
    return RunRef(project=tmp_path / "proj", state_dir=sd,
                  ledger=sd / "ledger.json", build_log=None, gpt_dir=None,
                  slug="main", layout=BY_MARKER["_planrunner"])


def test_a_fresh_run_reports_an_unbounded_span_in_live_mode_too(
        tmp_path, write_jsonl, assistant):
    """The fresh run has an `in-progress` step, so `_live_windows` gives it a
    synthetic window and the live window list is non-empty while the ledger's
    is not. Both modes must still say the span is unbounded."""
    ref = _fresh_run(tmp_path)
    session = write_jsonl("s.jsonl", [assistant("2026-06-25T10:30:00Z",
                                                out=777)])

    live = build_run(ref, [session], live=True)
    retro = build_run(ref, [session])

    assert "unbounded-span" in _reasons(live.anomalies)
    assert "unbounded-span" in _reasons(retro.anomalies)
    # The synthetic window is still doing its job: deviation B is unchanged.
    assert {s.id: s.output_tokens for s in live.steps}["A"] == 777


def test_a_run_with_a_normal_span_reports_it_in_neither_mode(
        tmp_path, write_jsonl, assistant):
    """The counterpart. A run whose steps carry real timestamps is bounded,
    live or not, and must not grow the warning -- otherwise the fix would be
    "always fire", which passes the test above for the wrong reason."""
    ref = _in_progress_run(tmp_path)
    session = _in_flight_session(write_jsonl, assistant)

    assert "unbounded-span" not in _reasons(build_run(ref, [session],
                                                      live=True).anomalies)
    assert "unbounded-span" not in _reasons(build_run(ref,
                                                      [session]).anomalies)


# --- Item B: a ledger `deliverable` is free text, not a path ----------------
#
# On the real corpus only 7/21, 4/10, 2/6 and 0/6 entries name a file; the
# rest are prose like "E.4 skeptic verdict (BUILD-LOG) + E.3 reply evidence".
# The split is computed here rather than in the surfaces because
# `render_frame` must stay a pure function of the in-memory Run.


def _mixed_deliverables_run(tmp_path):
    """Four deliverables: one absolute path to a real file, one relative to
    the project root, and two prose entries -- the shape the corpus actually
    has."""
    proj = tmp_path / "proj"
    sd = proj / "_planrunner" / "state"
    sd.mkdir(parents=True)
    absolute = tmp_path / "outside" / "spec.md"
    absolute.parent.mkdir()
    absolute.write_text("real\n")
    (proj / "notes").mkdir()
    (proj / "notes" / "design.md").write_text("also real\n")

    deliverables = [
        str(absolute),
        "notes/design.md",
        "E.4 skeptic verdict (BUILD-LOG) + E.3 reply evidence",
        "launchd job com.example installed + loaded",
    ]
    (sd / "ledger.json").write_text(json.dumps({"steps": [
        {"id": f"S{i}", "status": "complete", "deliverable": d,
         "updated": f"2026-06-25T{11 + i:02d}:00:00Z"}
        for i, d in enumerate(deliverables)]}))
    ref = RunRef(project=proj, state_dir=sd, ledger=sd / "ledger.json",
                 build_log=None, gpt_dir=None, slug="main", layout=BY_MARKER["_planrunner"])
    return build_run(ref, sessions=[]), str(absolute)


def test_deliverables_are_split_into_real_paths_and_prose(tmp_path):
    """Both resolution routes must work -- as given, and relative to the
    project root, which is the form the corpus uses. Checking only one of
    them silently demotes half the real files to prose."""
    run, absolute = _mixed_deliverables_run(tmp_path)

    assert sorted(run.doc_paths) == sorted([absolute, "notes/design.md"])
    assert run.doc_notes == [
        "E.4 skeptic verdict (BUILD-LOG) + E.3 reply evidence",
        "launchd job com.example installed + loaded",
    ]
    # Nothing is discarded: `docs` still holds every entry verbatim.
    assert len(run.docs) == 4
    assert set(run.doc_paths) | set(run.doc_notes) == set(run.docs)


def test_a_deliverable_is_classified_by_existence_not_by_looking_path_like(
        tmp_path):
    """"Ends in .md" is the tempting shortcut and it is wrong in both
    directions: a named file that was never written is not a document, and
    the corpus's prose entries cite `.md` filenames inside sentences."""
    proj = tmp_path / "proj"
    sd = proj / "_planrunner" / "state"
    sd.mkdir(parents=True)
    (sd / "ledger.json").write_text(json.dumps({"steps": [
        {"id": "S1", "status": "complete", "deliverable": "never-written.md",
         "updated": "2026-06-25T11:00:00Z"},
        {"id": "S2", "status": "complete",
         "deliverable": "_planrunner/E2.md + concepts/theory.md (filed)",
         "updated": "2026-06-25T12:00:00Z"}]}))
    ref = RunRef(project=proj, state_dir=sd, ledger=sd / "ledger.json",
                 build_log=None, gpt_dir=None, slug="main", layout=BY_MARKER["_planrunner"])

    run = build_run(ref, sessions=[])
    assert run.doc_paths == []
    assert len(run.doc_notes) == 2


def test_prose_that_cannot_be_a_filename_does_not_raise(tmp_path):
    """Over-long components and embedded NULs make `is_file()` raise rather
    than answer False. A deliverable is free text and may contain anything."""
    proj = tmp_path / "proj"
    sd = proj / "_planrunner" / "state"
    sd.mkdir(parents=True)
    (sd / "ledger.json").write_text(json.dumps({"steps": [
        {"id": "S1", "status": "complete", "deliverable": "x" * 5000,
         "updated": "2026-06-25T11:00:00Z"},
        {"id": "S2", "status": "complete", "deliverable": "has\x00null",
         "updated": "2026-06-25T12:00:00Z"}]}))
    ref = RunRef(project=proj, state_dir=sd, ledger=sd / "ledger.json",
                 build_log=None, gpt_dir=None, slug="main", layout=BY_MARKER["_planrunner"])

    run = build_run(ref, sessions=[])  # must not raise
    assert run.doc_paths == []
    assert len(run.doc_notes) == 2


# --- Step 6: turn renumbering reaches `parent_turn`; steps carry their -------
# agents.
#
# `parent_turn` is read from transcript metadata with the *session's* own
# numbering (`read_agents` builds `parent_by_tool_use` from that session's
# `read_turns` alone). `_renumber_within_run` then replaces every turn's
# `.index` with a run-local one. An agent from a run's second (or later)
# session therefore needs its `parent_turn` remapped onto the same run-local
# scale, or it silently points at the *first* session's turn of the same
# number. `Step.agents` is the first consumer that reads `parent_turn` and
# `Step.turns` together, which is why both had to land on that scale first.


def _two_session_run(tmp_path, assistant):
    """One run, one step, two sessions. Session one contributes turns 1 and
    2, so session two's only turn -- the one that dispatches agent a1 -- is
    run-local turn 3 only after `_renumber_within_run` runs. Session two's own
    numbering also calls that turn "1", so an implementation that forgets to
    remap `parent_turn` reports 1 -- indistinguishable from correct in a
    one-session fixture, which is why this one uses two."""
    sd = tmp_path / "proj" / "_planrunner" / "state"
    sd.mkdir(parents=True)
    ledger = sd / "ledger.json"
    ledger.write_text(json.dumps({
        "created": "2026-06-25T09:00:00Z",
        "steps": [{"id": "S1", "status": "complete",
                   "updated": "2026-06-25T10:20:00Z"}]}))
    ref = RunRef(project=tmp_path / "proj", state_dir=sd, ledger=ledger,
                 build_log=None, gpt_dir=None, slug="main",
                 layout=BY_MARKER["_planrunner"])

    # Both sessions name the run's own state dir in a Bash tool_use, the
    # signal a real session-filter keys on -- this fixture hands `build_run`
    # its sessions directly, but a fixture that could not survive a real
    # filter would prove nothing about the corpus.
    bash_tool = {"type": "tool_use", "id": "bash-1", "name": "Bash",
                "input": {"command": f'ledger.py --state-dir "{sd}"'}}
    s1 = tmp_path / "s1.jsonl"
    s1.write_text("\n".join(json.dumps(r) for r in [
        assistant("2026-06-25T10:00:00Z", tools=[bash_tool]),
        assistant("2026-06-25T10:05:00Z", tools=[bash_tool])]) + "\n")

    agent_tool = {"type": "tool_use", "id": "tu-1", "name": "Agent",
                 "input": {"description": "a1"}}
    s2 = tmp_path / "s2.jsonl"
    s2.write_text(json.dumps(
        assistant("2026-06-25T10:10:00Z", tools=[bash_tool, agent_tool])) + "\n")

    sub = tmp_path / "s2" / "subagents"
    sub.mkdir(parents=True)
    (sub / "agent-a1.meta.json").write_text(json.dumps(
        {"toolUseId": "tu-1", "agentType": "implementer"}))
    (sub / "agent-a1.jsonl").write_text(json.dumps({
        "type": "assistant", "timestamp": "2026-06-25T10:11:00Z",
        "message": {"model": "claude-haiku-4-5-20251001",
                    "usage": {"output_tokens": 50, "input_tokens": 1,
                              "cache_read_input_tokens": 0,
                              "cache_creation_input_tokens": 0},
                    "content": []}}) + "\n")

    return build_run(ref, sessions=[s1, s2])


def test_parent_turn_is_renumbered_with_the_run_not_the_session(
        tmp_path, assistant):
    # Session one contributes turns 1 and 2, so session two's only turn is
    # run-local turn 3. An implementation that forgets to remap reports 1 --
    # indistinguishable from correct in a one-session fixture.
    run = _two_session_run(tmp_path, assistant)
    assert [a.parent_turn for a in run.agents] == [3]


def test_steps_carry_the_agents_dispatched_from_their_turns(
        tmp_path, assistant):
    run = _two_session_run(tmp_path, assistant)
    assert [a.agent_id for a in run.steps[0].agents] == ["a1"]
    assert run.steps[0].agents[0].model == "claude-haiku-4-5-20251001"


# --- Gate A fix 1: an agent dispatched outside the run's span -----------------


def test_a_round_present_on_disk_and_in_the_transcript_merges_to_one_event(
        tmp_path):
    """Gate A's conceded R1 fix: the plan's own new tests exercise
    `gpt_events`/`gate_call_events` in isolation only -- none go through
    `build_run`'s merge/split wiring. This one does, on a `_maxrunner`-style
    per-step layout, covering the escalating-round, backfilled-duration,
    surviving-companion-event and surviving-anomaly cases together."""
    proj = tmp_path / "proj"
    sd = proj / "_maxrunner" / "myslug" / "state"
    sd.mkdir(parents=True)
    ledger = sd / "ledger.json"
    # `created` bounds the run's span from below (08:00), so the transcript
    # call's send timestamp (08:10:31) falls inside `[start, end]` and is
    # not dropped by `call_events`' span filter.
    ledger.write_text(json.dumps({
        "created": "2026-07-27T08:00:00Z",
        "steps": [{"id": "S1", "status": "complete",
                   "updated": "2026-07-27T09:00:00Z"}]}))
    gpt_dir = sd / "md"
    step_dir = gpt_dir / "S1"
    step_dir.mkdir(parents=True)
    # A well-formed, escalating round -- yields md.review + md.escalate +
    # doc.write on the disk side.
    (step_dir / "0-r1.json").write_text(json.dumps({
        "gate": "0", "round": 1, "ts": "2026-07-27T08:10:00Z",
        "outcome": "escalate",
        "report": {"risks": [{"id": "R9", "severity": "critical",
                              "claim": "danger"}]},
        "classification": {"blocking": ["R9"], "advisory": []},
        "tokens": 500}))
    # A second, unparseable round file -- yields an anomaly.
    (step_dir / "0-r2.json").write_text("{not valid json")

    ref = RunRef(project=proj, state_dir=sd, ledger=ledger, build_log=None,
                gpt_dir=gpt_dir, slug="myslug", layout=BY_MARKER["_maxrunner"])

    # The MD's `gate.py review` call for the same (step, gate, round) as the
    # escalating disk round, correlated by tool_use_id, with a real
    # send/return timestamp pair. Its own verdict body deliberately
    # disagrees with the disk's (clean here, escalating on disk) so a test
    # failure would reveal the transcript's data leaking through instead of
    # the disk's winning.
    bash_tool = {"type": "tool_use", "id": "t1", "name": "Bash",
                "input": {"command":
                          "python3 ~/reviewer/gate.py review --gate 0 "
                          "--thread cmag-myslug --step S1 --round 1"}}
    session = tmp_path / "session.jsonl"
    session.write_text("\n".join(json.dumps(r) for r in [
        {"type": "assistant", "timestamp": "2026-07-27T08:10:31Z",
         "cwd": "/proj", "sessionId": "s1", "version": "2.1.169",
         "gitBranch": "main",
         "message": {"model": "claude-opus-4-8",
                     "usage": {"input_tokens": 2, "output_tokens": 10,
                               "cache_creation_input_tokens": 10,
                               "cache_read_input_tokens": 1000},
                     "content": [bash_tool]}},
        {"type": "user", "timestamp": "2026-07-27T08:11:59Z",
         "message": {"content": [
             {"type": "tool_result", "tool_use_id": "t1",
              "content": json.dumps({"report": {"risks": []}}),
              "is_error": False}]}},
    ]) + "\n")

    run = build_run(ref, [session])

    # 1. Exactly one md.review-kind event for (S1, "0", 1) -- not two --
    #    and its outcome/blocking come from disk, not the transcript's
    #    (differing) verdict.
    reviews = [e for e in run.events if e.kind == "md.review"
              and e.payload.get("gate") == "0"
              and e.payload.get("round") == 1]
    assert len(reviews) == 1
    review = reviews[0]
    assert review.payload["outcome"] == "escalate"
    assert review.payload["blocking"] == ["R9"]

    # 2. duration_sec is backfilled from the transcript call (disk never
    #    sets it, so a non-None value proves the merge enrichment ran).
    assert review.payload["duration_sec"] == 88.0

    # 3. The disk's companion md.escalate event survives, untouched.
    escalates = [e for e in run.events if e.kind == "md.escalate"
                and e.payload.get("gate") == "0"
                and e.payload.get("round") == 1]
    assert len(escalates) == 1

    # 4. The disk's doc.write event for that round survives.
    doc_writes = [e for e in run.events if e.kind == "doc.write"
                 and e.artifact_path == str(step_dir / "0-r1.json")]
    assert len(doc_writes) == 1

    # 5. The disk's anomaly event (malformed second file) survives.
    anomalies = [e for e in run.events if e.kind == "anomaly"
                and e.artifact_path == str(step_dir / "0-r2.json")]
    assert len(anomalies) == 1


def test_a_gate_call_outside_the_runs_span_is_excluded_from_run_events(
        tmp_path):
    """`build_run`'s own top-of-function comment: a session may span several
    runs back to back, so its turns and agents are cut down to this run's
    ledger span before being kept -- `call_events` must be too, or a session
    shared between two runs leaks one run's gate calls into its sibling's MD
    counts/timing."""
    proj = tmp_path / "proj"
    sd = proj / "_maxrunner" / "myslug" / "state"
    sd.mkdir(parents=True)
    ledger = sd / "ledger.json"
    ledger.write_text(json.dumps({
        "created": "2026-07-27T10:00:00Z",
        "steps": [{"id": "S1", "status": "complete",
                   "updated": "2026-07-27T11:00:00Z"}]}))
    ref = RunRef(project=proj, state_dir=sd, ledger=ledger, build_log=None,
                gpt_dir=None, slug="myslug", layout=BY_MARKER["_maxrunner"])

    # This call's send timestamp (08:10) sits well before the run's span
    # (10:00-11:00) -- e.g. a sibling run's own gate call, in a transcript
    # this run's session selection happens to share.
    bash_tool = {"type": "tool_use", "id": "t1", "name": "Bash",
                "input": {"command":
                          "python3 ~/reviewer/gate.py review --gate 0 "
                          "--thread cmag-other --round 1"}}
    session = tmp_path / "session.jsonl"
    session.write_text("\n".join(json.dumps(r) for r in [
        {"type": "assistant", "timestamp": "2026-07-27T08:10:00Z",
         "cwd": "/proj", "sessionId": "s1", "version": "2.1.169",
         "gitBranch": "main",
         "message": {"model": "claude-opus-4-8",
                     "usage": {"input_tokens": 2, "output_tokens": 10,
                               "cache_creation_input_tokens": 10,
                               "cache_read_input_tokens": 1000},
                     "content": [bash_tool]}},
        {"type": "user", "timestamp": "2026-07-27T08:11:00Z",
         "message": {"content": [
             {"type": "tool_result", "tool_use_id": "t1",
              "content": json.dumps({"report": {"risks": []}}),
              "is_error": False}]}},
    ]) + "\n")

    run = build_run(ref, [session])
    assert [e for e in run.events if e.kind.startswith("md.")] == []


def test_an_agent_dispatched_outside_the_runs_span_reports_no_parent_turn(
        tmp_path, assistant):
    """The dispatching turn (08:00) precedes the run's span (`created`
    09:00), so `build_run` never keeps it in `all_turns` -- the remap lookup
    (`pre.get((session, parent_turn))`) misses and the agent must report
    `parent_turn is None`, not a stale session-local index. The agent's own
    activity (10:05) sits inside the span, so the agent itself is not
    filtered out of `run.agents` -- only its parent attribution is unknown,
    and it must not be attached to any step."""
    sd = tmp_path / "proj" / "_planrunner" / "state"
    sd.mkdir(parents=True)
    ledger = sd / "ledger.json"
    ledger.write_text(json.dumps({
        "created": "2026-06-25T09:00:00Z",
        "steps": [{"id": "S1", "status": "complete",
                   "updated": "2026-06-25T11:00:00Z"}]}))
    ref = RunRef(project=tmp_path / "proj", state_dir=sd, ledger=ledger,
                 build_log=None, gpt_dir=None, slug="main",
                 layout=BY_MARKER["_planrunner"])

    agent_tool = {"type": "tool_use", "id": "tu-2", "name": "Agent",
                 "input": {"description": "a2"}}
    session = tmp_path / "before.jsonl"
    session.write_text("\n".join(json.dumps(r) for r in [
        assistant("2026-06-25T08:00:00Z", tools=[agent_tool]),
        assistant("2026-06-25T10:00:00Z")]) + "\n")

    sub = tmp_path / "before" / "subagents"
    sub.mkdir(parents=True)
    (sub / "agent-a2.meta.json").write_text(json.dumps(
        {"toolUseId": "tu-2", "agentType": "implementer"}))
    (sub / "agent-a2.jsonl").write_text(json.dumps({
        "type": "assistant", "timestamp": "2026-06-25T10:05:00Z",
        "message": {"model": "claude-haiku-4-5-20251001",
                    "usage": {"output_tokens": 40, "input_tokens": 1,
                              "cache_read_input_tokens": 0,
                              "cache_creation_input_tokens": 0},
                    "content": []}}) + "\n")

    run = build_run(ref, sessions=[session])

    assert [a.parent_turn for a in run.agents] == [None]
    assert all(a.agent_id != "a2" for s in run.steps for a in s.agents)


# --- Gate A fix 3: `Step.agents` is exclusive per step, not shared ------------


def _two_steps_two_agents_run(tmp_path, assistant):
    """Two ledger steps, each with its own dispatching turn and agent, so an
    attach-to-every-step (or always-first-step) implementation of
    `Step.agents` is distinguishable from the real per-step grouping: step S1
    must hold only a1 and S2 only a2 -- never both, never neither."""
    sd = tmp_path / "proj" / "_planrunner" / "state"
    sd.mkdir(parents=True)
    ledger = sd / "ledger.json"
    ledger.write_text(json.dumps({
        "created": "2026-06-25T09:00:00Z",
        "steps": [
            {"id": "S1", "status": "complete",
             "updated": "2026-06-25T10:30:00Z"},
            {"id": "S2", "status": "complete",
             "updated": "2026-06-25T12:00:00Z"}]}))
    ref = RunRef(project=tmp_path / "proj", state_dir=sd, ledger=ledger,
                 build_log=None, gpt_dir=None, slug="main",
                 layout=BY_MARKER["_planrunner"])

    tu1 = {"type": "tool_use", "id": "tu-1", "name": "Agent",
          "input": {"description": "a1"}}
    tu2 = {"type": "tool_use", "id": "tu-2", "name": "Agent",
          "input": {"description": "a2"}}
    session = tmp_path / "sess.jsonl"
    session.write_text("\n".join(json.dumps(r) for r in [
        assistant("2026-06-25T10:00:00Z", tools=[tu1]),
        assistant("2026-06-25T11:00:00Z", tools=[tu2])]) + "\n")

    sub = tmp_path / "sess" / "subagents"
    sub.mkdir(parents=True)
    for agent_id, tool_use_id, ts in (
            ("a1", "tu-1", "2026-06-25T10:05:00Z"),
            ("a2", "tu-2", "2026-06-25T11:05:00Z")):
        (sub / f"agent-{agent_id}.meta.json").write_text(json.dumps(
            {"toolUseId": tool_use_id, "agentType": "implementer"}))
        (sub / f"agent-{agent_id}.jsonl").write_text(json.dumps({
            "type": "assistant", "timestamp": ts,
            "message": {"model": "claude-haiku-4-5-20251001",
                        "usage": {"output_tokens": 30, "input_tokens": 1,
                                  "cache_read_input_tokens": 0,
                                  "cache_creation_input_tokens": 0},
                        "content": []}}) + "\n")

    return build_run(ref, sessions=[session])


def test_step_agents_are_exclusive_per_step_not_attached_to_every_step(
        tmp_path, assistant):
    """The plan's own test only asserts `run.steps[0].agents`, which an
    attach-to-every-step (or always-first-step) implementation would also
    satisfy. Two steps, two agents, one dispatched from each: exact
    membership, checked per step, not just step 0 in isolation."""
    run = _two_steps_two_agents_run(tmp_path, assistant)
    by_id = {s.id: [a.agent_id for a in s.agents] for s in run.steps}
    assert by_id["S1"] == ["a1"]
    assert by_id["S2"] == ["a2"]


# --- Gate B round 2, R1: an agent dispatched in-span but reporting late ------
#
# The original filter kept an agent only when its *own* `ts_start` fell
# inside `[start, end]`. An agent's own first output can land just after the
# run's `end` even though the turn that spawned it sits squarely inside the
# span -- that agent was dropped from `run.agents` before parent-turn remap
# ever ran, so it could never reach `Step.agents`, silently breaking "holds
# the agents dispatched from that step's turns" for the step whose turn
# actually dispatched it.


def test_an_agent_reporting_after_the_span_is_kept_by_its_dispatching_turn(
        tmp_path, assistant):
    """The dispatching turn (10:00) is inside the run's span (09:00-11:00),
    but agent a3's own first recorded output (11:05) lands just after `end`.
    The old ts_start-only filter would drop it from `run.agents` entirely;
    it must instead survive via the dispatching-turn-in-`pre` branch, keep a
    real (non-None) `parent_turn`, and land in the step whose turns include
    that dispatching turn."""
    sd = tmp_path / "proj" / "_planrunner" / "state"
    sd.mkdir(parents=True)
    ledger = sd / "ledger.json"
    ledger.write_text(json.dumps({
        "created": "2026-06-25T09:00:00Z",
        "steps": [{"id": "S1", "status": "complete",
                   "updated": "2026-06-25T11:00:00Z"}]}))
    ref = RunRef(project=tmp_path / "proj", state_dir=sd, ledger=ledger,
                 build_log=None, gpt_dir=None, slug="main",
                 layout=BY_MARKER["_planrunner"])

    agent_tool = {"type": "tool_use", "id": "tu-3", "name": "Agent",
                 "input": {"description": "a3"}}
    session = tmp_path / "late-report.jsonl"
    session.write_text(json.dumps(
        assistant("2026-06-25T10:00:00Z", tools=[agent_tool])) + "\n")

    sub = tmp_path / "late-report" / "subagents"
    sub.mkdir(parents=True)
    (sub / "agent-a3.meta.json").write_text(json.dumps(
        {"toolUseId": "tu-3", "agentType": "implementer"}))
    (sub / "agent-a3.jsonl").write_text(json.dumps({
        "type": "assistant", "timestamp": "2026-06-25T11:05:00Z",
        "message": {"model": "claude-haiku-4-5-20251001",
                    "usage": {"output_tokens": 25, "input_tokens": 1,
                              "cache_read_input_tokens": 0,
                              "cache_creation_input_tokens": 0},
                    "content": []}}) + "\n")

    run = build_run(ref, sessions=[session])

    assert [a.agent_id for a in run.agents] == ["a3"]
    a3 = run.agents[0]
    assert a3.parent_turn is not None
    dispatching_step = next(s for s in run.steps
                            if a3.parent_turn in s.turns)
    assert dispatching_step.id == "S1"
    assert [a.agent_id for a in dispatching_step.agents] == ["a3"]


# --- Step 10, Gate A R1: self_report_drift off the real build_run pipeline --
# not a hand-built double ------------------------------------------------
#
# Every other test of self_report_drift (tests/test_metrics.py) constructs
# Step.ledger_tokens/turns/output_tokens directly via the _step/_fake
# doubles. Nothing exercises those fields reaching self_report_drift through
# a real ledger + real transcript, where ledger_tokens comes from
# ledger.py's _num() and turns/output_tokens come from assign_turns_to_steps
# and by_index summation. This is that one integration test, the fix Gate A
# adopted for blocking risk R1.


def test_self_report_drift_reads_the_real_pipeline_not_a_double(
        tmp_path, write_jsonl, assistant):
    """Three steps, each carrying a `tokens` figure the ledger itself wrote.
    A's one turn matches its claim exactly (agrees); B's one turn flatly
    contradicts its claim (diverges); C's window catches no turn at all, so
    its `turns` bucket is a real, explicit empty list (no-attribution) --
    the same distinction the doubles test, reached here through
    assign_turns_to_steps and the by_index sum instead of a hand-built
    Step(...)."""
    steps = [
        {"id": "A", "status": "complete", "owner": "builder", "tier": "sonnet",
         "attempts": 1, "deliverable": None, "depends_on": [],
         "updated": "2026-06-25T11:00:00Z", "tokens": 100},
        {"id": "B", "status": "complete", "owner": "builder", "tier": "sonnet",
         "attempts": 1, "deliverable": None, "depends_on": [],
         "updated": "2026-06-25T13:00:00Z", "tokens": 50},
        {"id": "C", "status": "complete", "owner": "builder", "tier": "sonnet",
         "attempts": 1, "deliverable": None, "depends_on": [],
         "updated": "2026-06-25T15:00:00Z", "tokens": 9000},
    ]
    ref = _ledger_with_steps(tmp_path, steps, created="2026-06-25T10:00:00Z")
    session = write_jsonl("session.jsonl", [
        assistant("2026-06-25T10:30:00Z", out=100),  # -> window A: matches
        assistant("2026-06-25T12:00:00Z", out=5000),  # -> window B: contradicts
        # No turn falls in (13:00, 15:00]: C's turns bucket stays empty.
    ])
    run = build_run(ref, sessions=[session])
    got = {r["step"]: r for r in self_report_drift(run)["steps"]}

    assert got["A"]["ledger"] == 100
    assert got["A"]["derived"] == 100
    assert got["A"]["state"] == "agrees"
    assert got["A"]["divergent"] is False

    assert got["B"]["ledger"] == 50
    assert got["B"]["derived"] == 5000
    assert got["B"]["state"] == "diverges"
    assert got["B"]["divergent"] is True

    assert got["C"]["ledger"] == 9000
    assert got["C"]["state"] == "no-attribution"
    assert got["C"]["divergent"] is False
