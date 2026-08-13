import io
import pytest
import json
from agentview.discovery import find_runs
from agentview.model import build_run
from agentview.surfaces import live
from agentview.surfaces.live import render_frame, replay, watch
from agentview.surfaces.viewstate import ViewState, reduce


def _row_for(frame, label):
    """The one frame line whose content starts with `label`. Raises rather than
    returning None: `"x" in None` errors confusingly, and two matches mean the
    frame is not what the test thinks it is."""
    rows = [l for l in frame.splitlines() if l.strip().startswith(label)]
    assert len(rows) == 1, f"expected one {label!r} row, got {len(rows)}"
    return rows[0]


def _run(tmp_path):
    sd = tmp_path / "proj" / "_planrunner" / "state"
    sd.mkdir(parents=True)
    (sd / "ledger.json").write_text(json.dumps({"steps": [
        {"id": "A", "status": "complete", "owner": "b", "tier": "sonnet",
         "attempts": 2, "deliverable": "a.md", "depends_on": [],
         "updated": "2026-06-25T11:00:00Z"},
        {"id": "B", "status": "in-progress", "owner": "b", "tier": "opus",
         "attempts": None, "deliverable": None, "depends_on": ["A"],
         "updated": "2026-06-25T12:00:00Z"},
        {"id": "C", "status": "suspended", "owner": "b", "tier": None,
         "attempts": None, "deliverable": None, "depends_on": ["B"],
         "wake": "human:voice-memo-sent",
         "updated": "2026-06-25T12:30:00Z"}]}))
    return build_run(find_runs(tmp_path)[0], sessions=[])


def test_frame_contains_every_tier_row(tmp_path):
    """`_run` builds a `_planrunner` run, whose layout declares only three
    rows (human, orchestrator, workers) -- rows now come from
    `run.ref.layout.tiers`, not a fixed six-row list, so the other three
    tiers must be absent, not merely unchecked."""
    frame = render_frame(_run(tmp_path))
    for tier in ("human", "orchestrator", "workers"):
        assert tier in frame
    for absent in ("fable", "gpt md", "director"):
        assert absent not in frame


def test_frame_shows_step_progress(tmp_path):
    assert "1/3" in render_frame(_run(tmp_path))


def test_frame_marks_the_in_progress_step(tmp_path):
    frame = render_frame(_run(tmp_path))
    line = [l for l in frame.splitlines() if l.strip().startswith("B")][0]
    assert "in-progress" in line


def test_frame_shows_the_wake_condition_of_a_suspended_step(tmp_path):
    assert "human:voice-memo-sent" in render_frame(_run(tmp_path))


def test_frame_shows_attempt_counts(tmp_path):
    line = [l for l in render_frame(_run(tmp_path)).splitlines()
            if l.strip().startswith("A")][0]
    assert "x2" in line


def test_frame_respects_width(tmp_path):
    for line in render_frame(_run(tmp_path), width=60).splitlines():
        assert len(line) <= 60


def test_watch_stops_after_the_requested_iterations(tmp_path):
    _run(tmp_path)
    ref = find_runs(tmp_path)[0]
    buf = io.StringIO()
    watch(ref, sessions=[], interval=0, iterations=2, stream=buf)
    assert buf.getvalue().count(live.CLEAR) == 2


# --- Step 16: raw mode / read_key wiring -- q exits via ViewState.quit ------

def test_watch_exits_via_view_quit_on_q_without_hanging(tmp_path, monkeypatch):
    """Drives the reducer/loop wiring directly: fake `term.raw_mode` as
    interactive and feed `term.read_key` a single "q", with no `iterations`
    bound at all -- if `q` did not break the loop through `view.quit`, this
    test would hang forever rather than merely fail, which is exactly the
    risk the plan calls out. `term.raw_mode`/`term.read_key` are faked at
    the module level (not via a real fd) so this stays a fast unit test, not
    one depending on the real terminal the test process happens to run
    under; the real TTY path is proven separately by the pty-backed
    integration test in `tests/test_term.py`."""
    import contextlib

    import agentview.surfaces.live as live_mod

    _run(tmp_path)
    ref = find_runs(tmp_path)[0]

    @contextlib.contextmanager
    def fake_raw_mode(*a, **k):
        yield True

    keys = iter(["q"])
    monkeypatch.setattr(live_mod.term, "raw_mode", fake_raw_mode)
    monkeypatch.setattr(live_mod.term, "read_key",
                        lambda *a, **k: next(keys, None))

    buf = io.StringIO()
    watch(ref, sessions=[], interval=0, stream=buf)  # no `iterations` bound
    assert buf.getvalue().count(live.CLEAR) == 1


# --- Strengthening beyond the brief ---

def _run_with_tokens(tmp_path, done_count, total_count):
    """Build a run with `done_count` complete steps out of `total_count`,
    chosen so the "done/total" figure cannot be confused with any other
    count in the frame (agent count, tier count, tokens-by-tier count)."""
    sd = tmp_path / "proj" / "_planrunner" / "state"
    sd.mkdir(parents=True)
    steps = []
    for i in range(total_count):
        status = "complete" if i < done_count else "in-progress"
        steps.append({
            "id": f"S{i}", "status": status, "owner": "b", "tier": "sonnet",
            "attempts": 1, "deliverable": None, "depends_on": [],
            "updated": f"2026-06-25T{11 + i:02d}:00:00Z",
        })
    (sd / "ledger.json").write_text(json.dumps({"steps": steps}))
    return build_run(find_runs(tmp_path)[0], sessions=[])


def test_frame_step_progress_counts_only_done_statuses_not_total_minus_one(tmp_path):
    """A fixture where done=1 of 3 could pass even if the implementation
    used e.g. `len(steps) - 2` or counted "in-progress" steps instead of
    "complete" ones, since both would coincidentally also yield 1. Use a
    5-step run with 3 done so the only correct computation is counting
    steps whose status is in the "complete" family."""
    run = _run_with_tokens(tmp_path, done_count=3, total_count=5)
    frame = render_frame(run)
    assert "3/5" in frame
    # Confirm no other plausible-but-wrong count would have matched:
    # neither "in-progress count" (2), nor "total - 1" (4), nor "0" appear
    # as the progress figure.
    assert "2/5" not in frame
    assert "4/5" not in frame


def test_frame_clips_a_line_that_genuinely_exceeds_width_including_the_rule(tmp_path):
    """The brief's width test (width=60) never actually forces clipping,
    since none of the default fixture's lines reach 60 chars -- an
    implementation that silently skipped _clip entirely would still pass
    it. Force a long project name and a long owner/step id so at least
    one content line must be truncated, and pin both the exact clipped
    length and the ellipsis marker. Also confirm the separator rule
    itself is exactly `width` characters (not longer), since the rule is
    built directly from `width` and a regression there would slip past a
    <= check alone if rule generation used the wrong variable."""
    long_name = "a-very-long-project-directory-name-that-will-not-fit-in-width"
    sd = tmp_path / long_name / "_planrunner" / "state"
    sd.mkdir(parents=True)
    (sd / "ledger.json").write_text(json.dumps({"steps": [
        {"id": "STEP-WITH-A-VERY-LONG-IDENTIFIER", "status": "in-progress",
         "owner": "an-extremely-long-owner-name-field-value", "tier": "opus",
         "attempts": 1, "deliverable": None, "depends_on": [],
         "updated": "2026-06-25T11:00:00Z"}]}))
    run = build_run(find_runs(tmp_path)[0], sessions=[])

    width = 40
    frame = render_frame(run, width=width)
    lines = frame.splitlines()

    rule_lines = [l for l in lines if l and set(l) == {"─"}]
    assert rule_lines, "expected at least one separator rule line"
    for rule in rule_lines:
        assert len(rule) == width

    header = lines[0]
    assert len(header) == width
    assert header.endswith("…")

    step_line = [l for l in lines if "STEP-WITH-A-VERY-LONG-IDENTIFIER" in l
                 or l.strip().startswith("STEP-WITH")][0]
    assert len(step_line) <= width


def test_frame_token_footer_shows_per_tier_figures_not_a_single_total(tmp_path):
    """The brief's fixture has sessions=[] so run.agents is empty and
    tokens_by_tier collapses to a single {"orchestrator": 0} entry --
    an implementation that printed one summed total instead of a
    per-tier breakdown would still pass every brief test. Build a run
    with agents at two different tiers plus non-zero orchestrator
    output, then confirm each tier's own figure is present individually
    (not just their sum) and that the orchestrator figure specifically
    is shown."""
    sd = tmp_path / "proj" / "_planrunner" / "state"
    sd.mkdir(parents=True)
    (sd / "ledger.json").write_text(json.dumps({"steps": [
        {"id": "A", "status": "complete", "owner": "b", "tier": "sonnet",
         "attempts": 1, "deliverable": None, "depends_on": [],
         "updated": "2026-06-25T11:00:00Z"}]}))
    ref = find_runs(tmp_path)[0]

    class FakeAgent:
        def __init__(self, model, output_tokens):
            self.model = model
            self.output_tokens = output_tokens

    run = build_run(ref, sessions=[])
    run.agents.append(FakeAgent("claude-sonnet-4", 111))
    run.agents.append(FakeAgent("claude-opus-4", 222))
    run.orchestrator_output_tokens = 4242

    frame = render_frame(run)
    assert "sonnet 111" in frame
    assert "opus 222" in frame
    assert "orchestrator 4,242" in frame
    # The sum of the two agent tiers must not appear as a fused total --
    # that would indicate the footer collapsed distinct tiers together.
    assert "333" not in frame


def test_watch_rereads_the_run_from_disk_on_each_iteration(tmp_path, monkeypatch):
    """watch() must call build_run fresh every iteration, not render a
    model captured once outside the loop. Mutate the ledger on disk
    between iterations (via a patched time.sleep, since interval=0 gives
    the loop no natural pause point) and confirm the two emitted frames
    reflect the two different on-disk states."""
    sd = tmp_path / "proj" / "_planrunner" / "state"
    sd.mkdir(parents=True)
    ledger_path = sd / "ledger.json"
    ledger_path.write_text(json.dumps({"steps": [
        {"id": "A", "status": "in-progress", "owner": "b", "tier": "sonnet",
         "attempts": 1, "deliverable": None, "depends_on": [],
         "updated": "2026-06-25T11:00:00Z"}]}))
    ref = find_runs(tmp_path)[0]

    calls = {"n": 0}

    def fake_sleep(_seconds):
        calls["n"] += 1
        ledger_path.write_text(json.dumps({"steps": [
            {"id": "A", "status": "complete", "owner": "b", "tier": "sonnet",
             "attempts": 1, "deliverable": None, "depends_on": [],
             "updated": "2026-06-25T13:00:00Z"}]}))

    import agentview.surfaces.live as live_mod
    monkeypatch.setattr(live_mod.time, "sleep", fake_sleep)

    buf = io.StringIO()
    watch(ref, sessions=[], interval=0, iterations=2, stream=buf)

    frames = buf.getvalue().split(live.CLEAR)
    frames = [f for f in frames if f]
    assert len(frames) == 2
    assert "in-progress" in frames[0]
    assert "complete" in frames[1]
    assert "in-progress" not in frames[1]


# --- Fix round 1: render_frame must not touch disk (purity) ---

def test_frame_renders_when_the_project_path_does_not_exist(tmp_path):
    """render_frame must be a pure function of the in-memory Run -- it
    must not do any filesystem access of its own (e.g. via
    metrics.summary -> waiver_audit -> Path.rglob). Point ref.project at
    a path that was never created and confirm rendering still succeeds
    instead of raising from a disk scan."""
    run = _run(tmp_path)
    run.ref.project = tmp_path / "this-directory-was-never-created"
    frame = render_frame(run)  # must not raise
    assert "plan-runner" in frame


@pytest.mark.parametrize("marker,expected", [
    ("_planrunner", "plan-runner"),
    ("_lightrunner", "light-runner"),
    ("_maxrunner", "max-runner"),
    ("_tieredrunner", "max-runner"),   # legacy marker, same runner
])
def test_the_header_names_the_runner_that_wrote_the_run(fake_run, marker,
                                                        expected):
    """The header label read as a literal `"tiered-runner"` for every layout,
    so `watch` on a plan-runner run announced the wrong runner. `watch` and
    `replay` accept all four layouts, and the pane tests only ever built the
    tiered fixture, so nothing could catch it. Parametrised over every marker
    precisely so a literal cannot pass again."""
    from agentview.layouts import BY_MARKER
    frame = render_frame(fake_run(layout=BY_MARKER[marker]))
    header = frame.splitlines()[0]
    assert header.startswith(expected)
    wrong = {"plan-runner", "light-runner", "max-runner"} - {expected}
    assert not any(w in header for w in wrong)


def test_render_frame_is_pure_even_if_disk_changes_between_calls(tmp_path):
    """The purity property stated as an assertion: two calls to
    render_frame on the same Run object must return byte-identical
    output, even if a new changes/*.md waiver file appears on disk
    between the two calls. A render_frame that (re)computed
    metrics.summary(run) -- which runs waiver_audit(run.ref.project),
    scanning changes/*.md on every call -- would be sensitive to this
    and could render differently, or at minimum perform unnecessary
    disk I/O, on the second call."""
    run = _run(tmp_path)
    frame_before = render_frame(run)

    changes_dir = run.ref.project / "changes"
    changes_dir.mkdir(parents=True, exist_ok=True)
    (changes_dir / "w.md").write_text("Kind: waiver\n")

    frame_after = render_frame(run)
    assert frame_after == frame_before


def test_render_frame_never_calls_waiver_audit(tmp_path, monkeypatch):
    """Direct instrumentation of the purity claim, since neither test
    above actually discriminates a regression to `s = summary(run)`:
    Path.rglob on a missing/odd path swallows OSError silently (so the
    "doesn't raise" test above passes either way), and render_frame
    never surfaces s["waivers"] in its output (so the byte-identical
    test above passes either way too). Patch waiver_audit itself -- the
    one call that performs disk I/O inside metrics.summary -- to raise
    if invoked, and confirm render_frame does not trigger it."""
    import agentview.metrics as metrics_mod

    def _boom(_project):
        raise AssertionError("waiver_audit must not be called by render_frame")

    monkeypatch.setattr(metrics_mod, "waiver_audit", _boom)
    run = _run(tmp_path)
    render_frame(run)  # must not raise


# --- Fix round 1: document list must not truncate silently ---

def test_frame_shows_a_remainder_count_when_more_than_eight_documents(tmp_path):
    """The document list silently dropped anything past the eighth entry. A
    viewer whose purpose is to show what a run wrote must say so instead
    of hiding it -- assert the exact remainder count for a 10-document
    run (2 hidden).

    The ten deliverables are now written to disk. The pane lists only
    entries that resolve to a real file, so a fixture of ten names with no
    files behind them would exercise the prose branch and never reach the
    truncation logic this test exists to pin."""
    proj = tmp_path / "proj"
    sd = proj / "_planrunner" / "state"
    sd.mkdir(parents=True)
    steps = [{
        "id": f"S{i}", "status": "complete", "owner": "b", "tier": "sonnet",
        "attempts": 1, "deliverable": f"doc{i}.md", "depends_on": [],
        "updated": f"2026-06-25T{11 + i:02d}:00:00Z",
    } for i in range(10)]
    (sd / "ledger.json").write_text(json.dumps({"steps": steps}))
    for i in range(10):
        (proj / f"doc{i}.md").write_text(f"deliverable {i}\n")
    run = build_run(find_runs(tmp_path)[0], sessions=[])

    assert len(run.docs) == 10
    assert len(run.doc_paths) == 10 and not run.doc_notes
    frame = render_frame(run)
    assert "… and 2 more" in frame


# --- Fix wave, item 2 (ESC-1): `replay` wrote to a stale stdout --------------
#
# `stream: TextIO = sys.stdout` binds the stdout object at *import*. Task 13
# worked around it for `watch` by passing `stream=sys.stdout` from cli.py,
# but cli.py called `live.replay(...)` with no stream, so replay kept the
# stale reference: under `redirect_stdout`, its frame leaked to the real
# stdout and the buffer captured 0 characters. The general fix is a `None`
# default resolved in the body, for both functions.

def test_replay_output_is_captured_by_redirect_stdout(tmp_path):
    """The proof of the escape, as a test. A stale import-time default sends
    the frame past the redirect and leaves the buffer empty."""
    import contextlib
    run = _run(tmp_path)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        replay(run)
    assert "plan-runner" in buf.getvalue()


def test_replay_progresses_through_distinct_stages(tmp_path):
    run = _run(tmp_path)
    buf = io.StringIO()
    replay(run, stream=buf)
    frames = [frame for frame in buf.getvalue().split(live.CLEAR) if frame]
    assert len(frames) == len(run.steps) + 1
    assert len(set(frames)) > 1
    assert frames[-1].strip() == render_frame(run).strip()


def test_replay_rejects_a_nonpositive_increment(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        replay(_run(tmp_path), stream=io.StringIO(), step=0)


def test_watch_output_is_captured_by_redirect_stdout(tmp_path):
    """`watch` is only correct today because cli.py passes `stream=sys.stdout`
    explicitly. Called without that argument -- as any other caller would --
    it must still honour the current stdout."""
    import contextlib
    _run(tmp_path)
    ref = find_runs(tmp_path)[0]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        watch(ref, sessions=[], interval=0, iterations=1)
    assert "plan-runner" in buf.getvalue()


# --- Fix wave, item 5 (C4): the live pane must show movement -----------------

def _live_project(tmp_path, created, step_updated, status="in-progress"):
    sd = tmp_path / "proj" / "_planrunner" / "state"
    sd.mkdir(parents=True)
    (sd / "ledger.json").write_text(json.dumps({
        "created": created,
        "steps": [{"id": "A", "status": status, "owner": "b", "tier": "opus",
                   "attempts": 1, "deliverable": None, "depends_on": [],
                   "updated": step_updated}]}))
    return find_runs(tmp_path)[0]


def _driver_session(projects_root, name, state_dir, stamps, out=500):
    d = projects_root / "-Users-example"
    d.mkdir(parents=True, exist_ok=True)
    records = []
    for i, ts in enumerate(stamps, 1):
        records.append({
            "type": "assistant", "timestamp": ts, "cwd": "/anywhere",
            "sessionId": name,
            "message": {"model": "claude-opus-4-8",
                        "usage": {"input_tokens": 1, "output_tokens": out,
                                  "cache_creation_input_tokens": 0,
                                  "cache_read_input_tokens": 0},
                        "content": [{"type": "tool_use", "id": f"t{i}",
                                     "name": "Bash",
                                     "input": {"command":
                                               f'SD={state_dir}; run "$SD"'}}]}})
    p = d / f"{name}.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return p


def test_watch_picks_up_a_session_created_between_iterations(tmp_path,
                                                             monkeypatch):
    """cli.py resolved sessions once, before the loop, so a session started
    while the pane was already open could never appear -- and the session's
    turns land *after* the ledger's last write, so the retrospective span
    would drop them even if it did. Both faults are exercised at once: the
    session file appears between redraws and its turns are later than the
    last `updated`."""
    ref = _live_project(tmp_path, "2026-06-25T10:00:00Z",
                        "2026-06-25T11:00:00Z")
    projects_root = tmp_path / "projects"
    projects_root.mkdir()

    def fake_sleep(_seconds):
        _driver_session(projects_root, "driver", ref.state_dir,
                        ["2026-06-25T12:00:00Z"], out=500)

    import agentview.surfaces.live as live_mod
    monkeypatch.setattr(live_mod.time, "sleep", fake_sleep)

    buf = io.StringIO()
    watch(ref, interval=0, iterations=2, stream=buf,
          projects_root=projects_root)

    frames = [f for f in buf.getvalue().split(live.CLEAR) if f]
    assert len(frames) == 2
    assert "orchestrator 0" in frames[0]
    assert "orchestrator 500" in frames[1]


def test_frame_labels_its_figures_provisional(tmp_path):
    """The live pane is not an accuracy instrument; nobody must mistake a
    live figure for the authoritative one."""
    assert "provisional" in render_frame(_run(tmp_path))


def test_frame_lifts_accuracy_anomalies_above_the_tier_rows(tmp_path):
    """Anomalies that say "this number is not clean" are not buried: they
    are printed before the tier table, with a `!` marker."""
    from agentview.events import Event
    run = _run(tmp_path)
    run.anomalies.append(Event(
        None, "anomaly", "unknown", None,
        {"reason": "run-span-overlap", "slug": "main",
         "other_slug": "format-v2", "overlap_seconds": 6247.0},
        None, "agentview"))

    lines = render_frame(run, width=200).splitlines()
    warn = [i for i, l in enumerate(lines) if "format-v2" in l]
    tiers = [i for i, l in enumerate(lines) if l.strip().startswith("role ")]
    assert warn, "the overlap warning is not shown at all"
    assert warn[0] < tiers[0], "the warning is below the tier table"


def test_frame_shows_no_warning_block_when_there_is_nothing_to_warn_about(
        tmp_path):
    """The counterpart: a clean run must not grow a permanent scare banner,
    or the marker stops meaning anything."""
    frame = render_frame(_run(tmp_path))
    assert " ! " not in frame


def test_the_pane_warns_that_a_fresh_runs_span_is_unbounded(tmp_path):
    """R1, at the surface it was missing from. A freshly started run showed
    figures with nothing explaining why they were provisional, because the
    anomaly was computed against the live window list rather than the
    ledger's."""
    sd = tmp_path / "proj" / "_planrunner" / "state"
    sd.mkdir(parents=True)
    (sd / "ledger.json").write_text(json.dumps({
        "created": "2026-06-25T10:00:00Z",
        "steps": [{"id": "A", "status": "in-progress", "owner": "b",
                   "tier": "opus", "attempts": 1, "deliverable": None,
                   "depends_on": [], "updated": None}]}))
    ref = find_runs(tmp_path)[0]

    buf = io.StringIO()
    watch(ref, sessions=[], interval=0, iterations=1, stream=buf)
    frame = buf.getvalue()

    assert " ! " in frame, "the fresh-run frame carries no warning line at all"
    # Asserted against the head of the message, not its tail: the pane clips
    # to 80 columns and "no upper bound" falls past the cut. That truncation
    # is parked under its own ruling and is not this fix's business.
    assert "no parseable step timestamp" in frame


# --- Item B: the pane must not call prose a document path -------------------

def _mixed_deliverables(tmp_path):
    """One absolute path to a real file, one relative to the project root,
    and two prose entries -- the shape the real corpus has."""
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
        {"id": f"S{i}", "status": "complete", "owner": "b", "tier": "sonnet",
         "attempts": 1, "deliverable": d, "depends_on": [],
         "updated": f"2026-06-25T{11 + i:02d}:00:00Z"}
        for i, d in enumerate(deliverables)]}))
    return build_run(find_runs(tmp_path)[0], sessions=[]), str(absolute)


def test_pane_lists_only_real_files_and_counts_the_prose(tmp_path):
    """The pane showed every ledger `deliverable` under a heading called
    "documents". On the real corpus most of them are sentences."""
    run, absolute = _mixed_deliverables(tmp_path)
    frame = render_frame(run, width=200)

    assert " documents" in frame
    assert "notes/design.md" in frame
    assert absolute in frame
    assert "2 ledger deliverables are prose, not paths" in frame
    # The prose itself is not listed among the documents.
    assert "E.4 skeptic verdict" not in frame
    assert "launchd job com.example" not in frame


def test_pane_never_claims_absolute_paths_when_an_entry_is_unresolvable(
        tmp_path):
    """The claim was unconditional. With a mixed list it is false twice over
    -- some entries are not paths at all, and the resolvable ones are
    project-relative."""
    run, _ = _mixed_deliverables(tmp_path)
    assert "are absolute" not in render_frame(run, width=200)


def test_pane_does_not_claim_absolute_for_resolvable_relative_paths(tmp_path):
    """The corpus's resolvable deliverables are project-relative
    (`_planrunner/E1-outbound.md`). Every entry naming a real file is not
    enough; they must actually be absolute."""
    proj = tmp_path / "proj"
    sd = proj / "_planrunner" / "state"
    sd.mkdir(parents=True)
    (proj / "notes").mkdir()
    (proj / "notes" / "design.md").write_text("real\n")
    (sd / "ledger.json").write_text(json.dumps({"steps": [
        {"id": "S1", "status": "complete", "owner": "b", "tier": "sonnet",
         "attempts": 1, "deliverable": "notes/design.md", "depends_on": [],
         "updated": "2026-06-25T11:00:00Z"}]}))
    run = build_run(find_runs(tmp_path)[0], sessions=[])

    assert run.doc_paths == ["notes/design.md"]
    assert not run.doc_notes
    assert "are absolute" not in render_frame(run, width=200)


def test_pane_does_claim_absolute_when_every_entry_genuinely_is(tmp_path):
    """The counterpart. Without this, deleting the descriptive line
    altogether would satisfy every assertion above."""
    proj = tmp_path / "proj"
    sd = proj / "_planrunner" / "state"
    sd.mkdir(parents=True)
    doc = tmp_path / "outside" / "spec.md"
    doc.parent.mkdir()
    doc.write_text("real\n")
    (sd / "ledger.json").write_text(json.dumps({"steps": [
        {"id": "S1", "status": "complete", "owner": "b", "tier": "sonnet",
         "attempts": 1, "deliverable": str(doc), "depends_on": [],
         "updated": "2026-06-25T11:00:00Z"}]}))
    run = build_run(find_runs(tmp_path)[0], sessions=[])

    assert "document paths above are absolute" in render_frame(run, width=200)


def test_pane_reports_prose_even_when_no_deliverable_resolves(tmp_path):
    """secondary-project's case: 0 of 6 entries name a file. The pane
    must not simply fall silent about what the run recorded."""
    proj = tmp_path / "proj"
    sd = proj / "_planrunner" / "state"
    sd.mkdir(parents=True)
    (sd / "ledger.json").write_text(json.dumps({"steps": [
        {"id": "S1", "status": "complete", "owner": "b", "tier": "sonnet",
         "attempts": 1, "deliverable": "hardened verify: (a)(b)(c) pass",
         "depends_on": [], "updated": "2026-06-25T11:00:00Z"}]}))
    run = build_run(find_runs(tmp_path)[0], sessions=[])
    frame = render_frame(run, width=200)

    assert " documents" not in frame
    assert "1 ledger deliverable is prose, not paths" in frame
    assert "are absolute" not in frame


def test_the_redraw_sequence_clears_the_scrollback():
    """Regression guard for the pane growing the scrollback each tick.

    A bare "\x1b[2J" pushes the cleared frame into scrollback, so the buffer
    grows every redraw and the reader ends up above the current frame. The 3J
    ("erase saved lines") is what keeps the pane redrawing in place.
    """
    assert live.CLEAR.endswith("\x1b[3J")
    assert live.CLEAR.startswith("\x1b[H")


# --- Step 11: warnings wrap; document paths middle-clip ----------------------

def test_a_long_warning_keeps_its_explanation(unbounded_run):
    frame = render_frame(unbounded_run, width=80)
    assert "no parseable step timestamp" in frame
    assert "sessions were selected by reference alone" in frame
    assert all(len(l) <= 80 for l in frame.splitlines())


def test_a_wrapped_warning_is_indented_under_its_marker(unbounded_run):
    lines = render_frame(unbounded_run, width=80).splitlines()
    first = next(i for i, l in enumerate(lines) if l.startswith(" ! "))
    assert lines[first + 1].startswith("   ")
    assert not lines[first + 1].startswith(" ! ")


def test_a_long_document_path_keeps_its_filename(run_with_long_doc_path):
    frame = render_frame(run_with_long_doc_path, width=80)
    assert "E4-outbound.md" in frame
    assert "…" in frame
    assert all(len(l) <= 80 for l in frame.splitlines())


# --- Step 12: the `gpt md` row reads the event stream -----------------------

def test_the_md_row_reports_the_latest_round(md_round_run):
    row = _row_for(render_frame(md_round_run, width=100), "gpt md")
    assert "gate b, round 2/3" in row
    assert "blocking: 1 high" in row


def test_the_md_row_reports_the_chronologically_latest_round_not_the_highest(
        md_round_chronology_run):
    row = _row_for(render_frame(md_round_chronology_run, width=100), "gpt md")
    assert "gate b, round 1/3" in row


def test_the_md_row_says_no_rounds_only_when_a_readable_dir_is_empty(
        empty_gpt_dir_run):
    row = _row_for(render_frame(empty_gpt_dir_run, width=100), "gpt md")
    assert "no rounds recorded" in row
    assert "unobservable" not in row


def test_the_md_row_is_unobservable_with_no_events_and_no_directory(blind_run):
    row = _row_for(render_frame(blind_run, width=100), "gpt md")
    assert "unobservable" in row
    assert "no rounds recorded" not in row


def test_the_md_row_shows_escalated_on_a_tied_round_review_then_escalate(
        md_tied_round_review_first_run):
    """`gptgates.py` emits `md.review` immediately followed by `md.escalate`
    for the same round file, so both events share one (ts, round, gate).
    `md.review` is appended first; without a kind tiebreak, `max()` keeps
    the first maximal element on a tie and the escalation never surfaces."""
    row = _row_for(render_frame(md_tied_round_review_first_run, width=100),
                   "gpt md")
    assert "escalated" in row


def test_the_md_row_shows_escalated_on_a_tied_round_escalate_then_review(
        md_tied_round_escalate_first_run):
    """The reverse input order. Input order is an accident of emission, not
    a contract -- the tiebreak must pick `md.escalate` regardless of which
    of the pair the caller happened to list first."""
    row = _row_for(render_frame(md_tied_round_escalate_first_run, width=100),
                   "gpt md")
    assert "escalated" in row


# --- Step 13: the roles table -- layout-driven rows, models with provenance --

def test_observed_models_render_bare(two_model_run):
    frame = render_frame(two_model_run, width=100)
    assert _row_for(frame, "step-runner").split()[1] == "claude-opus-5"
    assert "declared" not in _row_for(frame, "step-runner")
    workers = _row_for(frame, "workers")
    assert "haiku-4-5-20251001" in workers and "sonnet-4-6" in workers


def test_inferred_models_are_marked_inferred(inferred_roles_run):
    frame = render_frame(inferred_roles_run, width=100)
    assert "· inferred" in _row_for(frame, "director")
    assert "· inferred" in _row_for(frame, "fable")


def test_declared_is_only_the_fallback(two_model_run):
    assert "opus · declared" in _row_for(
        render_frame(two_model_run, width=100), "director")


def test_the_md_row_prefers_the_pin_over_the_default(pinned_md_run):
    row = _row_for(render_frame(pinned_md_run, width=100), "gpt md")
    assert "gpt-5.4-mini · medium" in row
    assert "gpt-5.6-sol" not in row


def test_the_director_row_shows_its_real_dispatch_count(inferred_roles_run):
    row = _row_for(render_frame(inferred_roles_run, width=100), "director")
    assert "2 dispatches" in row
    assert "unobservable" not in row


def test_the_fable_row_names_what_was_ruled(inferred_roles_run):
    row = _row_for(render_frame(inferred_roles_run, width=100), "fable")
    assert "1 ruling" in row and "S4" in row


def test_rows_come_from_the_layout_not_a_fixed_list(planrunner_run):
    frame = render_frame(planrunner_run, width=100)
    assert "orchestrator" in frame
    for absent in ("director", "fable", "gpt md"):
        assert absent not in frame, absent


def test_unobservable_appears_only_where_it_is_true(inferred_roles_run,
                                                    blind_run):
    assert "unobservable" not in render_frame(inferred_roles_run, width=100)
    assert "unobservable" in render_frame(blind_run, width=100)


def test_every_line_fits_eighty_columns(inferred_roles_run):
    assert all(len(l) <= 80
               for l in render_frame(inferred_roles_run, width=80).splitlines())


# --- Gate B round 1 (Step 13): undisplayed roles must agree between the
# workers count and the workers model cell; a dispatched dedicated-role
# agent must beat "unobservable" even with no richer state computed yet ----

def test_an_undisplayed_specialist_roles_model_appears_in_the_workers_row(
        undisplayed_role_run):
    """R1: `_role_states` folds a `skeptic` agent into the worker count
    (`role not in displayed`), but `_model_cell`'s `else` branch required an
    exact `role == tier.role` match, so the skeptic's model never appeared
    in the same row as its count."""
    row = _row_for(render_frame(undisplayed_role_run, width=100), "workers")
    assert "2 dispatched" in row
    assert "sonnet-4-6" in row
    assert "haiku-4-5-20251001" in row


def test_a_dispatched_fable_agent_with_no_ruling_yet_is_not_unobservable(
        fable_dispatched_no_ruling_run):
    """R2: `_role_states` only gave `fable` a richer state when
    `fable_rulings` was non-empty. A fable agent dispatched with no ruling
    yet fell through to the tail loop's "unobservable", contradicting
    `run.agents`, which proves the role was dispatched."""
    row = _row_for(render_frame(fable_dispatched_no_ruling_run, width=100),
                   "fable")
    assert "1 dispatched" in row
    assert "unobservable" not in row


# --- Step 14: ledger hygiene -- prose deliverables and self-report drift ----

def test_prose_deliverables_are_reported_as_ledger_hygiene(prose_run):
    frame = render_frame(prose_run, width=80)
    assert "ledger hygiene" in frame
    assert "full text is in the report" in frame
    assert "notes " not in frame


def test_one_prose_deliverable_reads_as_singular(one_prose_run):
    assert "1 ledger deliverable is prose" in render_frame(one_prose_run,
                                                            width=80)


def test_two_prose_deliverables_read_as_plural(prose_run):
    assert "2 ledger deliverables are prose" in render_frame(prose_run,
                                                              width=80)


def test_self_report_drift_is_surfaced(drift_run):
    frame = render_frame(drift_run, width=80)
    assert "ledger hygiene" in frame
    assert "1 step's token figure diverges" in frame
    assert "1 unrecorded" in frame


def test_a_clean_run_shows_no_hygiene_section(clean_run):
    assert "ledger hygiene" not in render_frame(clean_run, width=80)


# --- Step 17: selection marker, step detail, document tray, help overlay ----

def test_the_selected_step_is_marked_distinctly_from_the_active_one(
        two_step_run):
    frame = render_frame(two_step_run, ViewState(selected=1), width=80)
    assert "●" in _row_for(frame, two_step_run.steps[0].id)
    selected = _row_for(frame, two_step_run.steps[1].id)
    assert ">" in selected and "●" not in selected


def test_expanding_shows_agents_full_model_ids_and_the_ledger_comparison(
        two_step_run):
    frame = render_frame(two_step_run, ViewState(selected=0, expanded=True),
                         width=100)
    assert "claude-haiku-4-5-20251001" in frame   # full id, not shortened
    assert "attempts" in frame
    assert "ledger 880" in frame and "derived 900" in frame


def test_the_tray_lists_every_document(many_docs_run):
    closed = render_frame(many_docs_run, ViewState(tray_open=False), width=100)
    opened = render_frame(many_docs_run, ViewState(tray_open=True), width=100)
    assert "and 4 more" in closed and "and 4 more" not in opened
    assert opened.count("doc-") == 12


def test_help_advertises_exactly_the_keys_the_reducer_handles(two_step_run):
    from agentview.surfaces.live import HELP_KEYS
    frame = render_frame(two_step_run, ViewState(help_open=True), width=100)
    for key in HELP_KEYS:
        assert key in frame or key in ("\r",)
        # `k` at selected=0 correctly clamps to a state equal to the input --
        # that is the reducer working, not the key doing nothing. Prove
        # liveness from whichever of the two starting states a key can move:
        # `j` moves from selected=0, `k` moves from selected=1.
        assert (reduce(ViewState(), key, two_step_run) != ViewState() or
                reduce(ViewState(selected=1), key, two_step_run) !=
                ViewState(selected=1)), \
            f"{key!r} is advertised but does nothing from either state"


def test_selecting_past_the_step_list_clamps_to_the_last_step(two_step_run):
    """`ViewState(selected=5, expanded=True)` against a 2-step fixture must
    not raise, and must mark the *last* step selected rather than indexing
    past the list -- `render_frame` derives a locally clamped selection
    rather than trusting `view.selected` directly, since a live-watched run
    is rebuilt fresh each redraw and can shrink between frames."""
    frame = render_frame(two_step_run, ViewState(selected=5, expanded=True),
                         width=100)  # must not raise
    row = _row_for(frame, two_step_run.steps[1].id)
    assert ">" in row


# --- Gate B round 1 (R1/R2): unrecorded attempts / no-attribution turns must
# not read as a real value -----------------------------------------------

def test_an_unrecorded_attempt_count_reads_as_unrecorded_not_one(
        two_step_run):
    """E.2 (`two_step_run`'s second step) has `attempts=None` -- expanding it
    must not fabricate `attempts 1` via `st.attempts or 1`'s truthiness."""
    frame = render_frame(two_step_run, ViewState(selected=1, expanded=True),
                         width=100)
    assert "attempts unrecorded" in frame
    assert "attempts 1" not in frame


def test_a_real_positive_attempt_count_renders_the_number(two_step_run):
    """E.1 has a real `attempts=2` -- the detail line must show it verbatim,
    not merely something truthy."""
    frame = render_frame(two_step_run, ViewState(selected=0, expanded=True),
                         width=100)
    assert "attempts 2" in frame


def test_no_turns_reads_as_no_attribution_not_a_measured_zero(two_step_run):
    """E.2 has empty `turns` -- expanding it must not print `derived 0`,
    which would be indistinguishable from turns that were found and really
    summed to zero."""
    frame = render_frame(two_step_run, ViewState(selected=1, expanded=True),
                         width=100)
    assert "turns 0" in frame
    assert "derived no attribution" in frame
    assert "derived 0" not in frame


def test_turns_that_genuinely_sum_to_zero_still_show_the_real_zero(
        zero_output_step_run):
    """A step with a real (non-empty) `turns` list whose output tokens
    genuinely sum to zero must still show the literal `0`, not be
    suppressed into the same `no attribution` text an empty `turns` gets."""
    frame = render_frame(zero_output_step_run,
                         ViewState(selected=0, expanded=True), width=100)
    assert "turns 1" in frame
    assert "derived 0" in frame
    assert "no attribution" not in frame


def test_an_explicit_zero_attempt_count_renders_as_zero_not_one(
        zero_output_step_run):
    """`attempts=0` is a real, explicit value distinct from an unrecorded
    (`None`) count -- `st.attempts or 1`'s truthiness would coerce it to 1,
    identically to an unrecorded step. It must render `0` verbatim."""
    frame = render_frame(zero_output_step_run,
                         ViewState(selected=0, expanded=True), width=100)
    assert "attempts 0" in frame
    assert "attempts 1" not in frame
    assert "attempts unrecorded" not in frame
