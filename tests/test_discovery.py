import json
from agentview.discovery import (RunRef, find_runs, find_sessions,
                               run_sessions, sessions_for_runs,
                               _needles, _scan_session)
from agentview.layouts import BY_MARKER


def _project(tmp_path, state_dirs, logs):
    proj = tmp_path / "proj"
    for sd in state_dirs:
        d = proj / "_planrunner" / sd
        d.mkdir(parents=True)
        (d / "ledger.json").write_text(json.dumps({"steps": []}))
    for lg in logs:
        (proj / lg).write_text("")
    return proj


def test_each_state_dir_is_a_separate_run(tmp_path):
    _project(tmp_path, ["state", "state-format-v2", "state-soak"], ["BUILD-LOG.md"])
    runs = find_runs(tmp_path)
    assert sorted(r.slug for r in runs) == ["format-v2", "main", "soak"]


def test_state_dir_without_a_ledger_is_ignored(tmp_path):
    proj = _project(tmp_path, ["state"], ["BUILD-LOG.md"])
    (proj / "_planrunner" / "spike").mkdir()
    assert [r.slug for r in find_runs(tmp_path)] == ["main"]


def test_build_log_pairs_by_suffix(tmp_path):
    _project(tmp_path, ["state", "state-soak"],
             ["BUILD-LOG.md", "BUILD-LOG-soak.md"])
    by_slug = {r.slug: r for r in find_runs(tmp_path)}
    assert by_slug["soak"].build_log.name == "BUILD-LOG-soak.md"
    assert by_slug["main"].build_log.name == "BUILD-LOG.md"


def test_missing_build_log_is_none(tmp_path):
    _project(tmp_path, ["state-orphan"], [])
    assert find_runs(tmp_path)[0].build_log is None


def test_light_runner_ledger_is_discovered(tmp_path):
    proj = tmp_path / "lite"
    (proj / "_lightrunner").mkdir(parents=True)
    (proj / "_lightrunner" / "ledger.json").write_text("[]")
    assert [r.slug for r in find_runs(tmp_path)] == ["main"]


# --- Strengthening tests beyond the brief ---


def test_build_log_falls_back_to_plain_when_suffixed_absent(tmp_path):
    """A state-<suffix> dir with only BUILD-LOG.md present (no suffixed
    variant) must resolve to the plain log, not None. This proves the
    fallback half of the pairing rule, not just the suffix-match half."""
    _project(tmp_path, ["state-soak"], ["BUILD-LOG.md"])
    runs = find_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0].build_log.name == "BUILD-LOG.md"


def test_build_log_prefers_suffixed_when_both_exist(tmp_path):
    """When both BUILD-LOG.md and BUILD-LOG-<suffix>.md exist for the same
    suffixed state dir, the suffixed one must win. Without this, an
    implementation that always returns the plain log (ignoring suffix
    matching entirely) would still pass test_build_log_pairs_by_suffix
    if that test only checked one slug at a time in isolation -- here we
    pin down the single-run case directly."""
    _project(tmp_path, ["state-soak"], ["BUILD-LOG.md", "BUILD-LOG-soak.md"])
    runs = find_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0].build_log.name == "BUILD-LOG-soak.md"


def test_prune_set_blocks_descent_into_node_modules(tmp_path):
    """A ledger.json sitting under a pruned directory name (node_modules)
    must never surface as a run. Without PRUNE filtering, os.walk would
    happily descend into vendored/dependency trees and produce spurious
    runs from third-party fixtures or copied state."""
    proj = tmp_path / "hasdeps"
    real = proj / "_planrunner" / "state"
    real.mkdir(parents=True)
    (real / "ledger.json").write_text(json.dumps({"steps": []}))

    junk = proj / "node_modules" / "somepkg" / "_planrunner" / "state"
    junk.mkdir(parents=True)
    (junk / "ledger.json").write_text(json.dumps({"steps": []}))

    runs = find_runs(tmp_path)
    assert [r.slug for r in runs] == ["main"]
    assert all("node_modules" not in str(r.project) for r in runs)


def test_nested_project_is_discovered(tmp_path):
    """A _planrunner directory living inside a subdirectory of another
    project (e.g. a vendored or nested sub-project) must still be found.
    This proves find_runs walks recursively rather than only checking
    root/_planrunner directly."""
    proj = tmp_path / "outer" / "packages" / "inner"
    d = proj / "_planrunner" / "state"
    d.mkdir(parents=True)
    (d / "ledger.json").write_text(json.dumps({"steps": []}))
    (proj / "BUILD-LOG.md").write_text("")

    runs = find_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0].slug == "main"
    assert runs[0].project == proj


# --- Fix round 1: slug collisions within a project must not merge runs ---


def test_colliding_slugs_are_disambiguated_by_state_dir_name(tmp_path):
    """`_planrunner/soak/` (slug "soak" via the identity fallback) and
    `_planrunner/state-soak/` (slug "soak" via the state-<suffix> stripping
    rule) collide. A slug-keyed consumer (e.g. `by_slug = {r.slug: r for r
    in find_runs(...)}`, exactly as used elsewhere in this file) would
    silently drop one run if both kept slug "soak" -- precisely the merge
    this task exists to prevent. Both must survive with distinct slugs,
    each equal to its own state dir's name verbatim."""
    proj = tmp_path / "proj"
    for sd in ("soak", "state-soak"):
        d = proj / "_planrunner" / sd
        d.mkdir(parents=True)
        (d / "ledger.json").write_text(json.dumps({"steps": []}))

    runs = find_runs(tmp_path)
    assert len(runs) == 2
    slugs = {r.slug for r in runs}
    assert slugs == {"soak", "state-soak"}
    by_slug = {r.slug: r for r in runs}
    assert len(by_slug) == 2  # neither run was dropped by the collision


def test_non_colliding_slugs_are_unaffected_by_collision_handling(tmp_path):
    """A project with state, state-format-v2 and state-soak has no slug
    collisions, so the collision-handling added in this fix round must not
    change their slugs: `state` must still yield "main" and
    `state-format-v2` must still yield "format-v2" plain, not their raw
    directory names."""
    _project(tmp_path, ["state", "state-format-v2", "state-soak"], ["BUILD-LOG.md"])
    runs = find_runs(tmp_path)
    assert sorted(r.slug for r in runs) == ["format-v2", "main", "soak"]


def test_combined_real_project_fixture(tmp_path):
    """Reproduces the real project the task brief was written from:
    primary-project/_planrunner/ with state, state-format-v2, state-soak and
    spike (spike has no ledger.json and must be skipped) side by side, plus
    three separate BUILD-LOG variants. Exactly three runs must be found,
    spike must be absent, and each run must pair with its own BUILD-LOG by
    suffix rather than all falling back to the plain log."""
    proj = _project(
        tmp_path,
        ["state", "state-format-v2", "state-soak"],
        ["BUILD-LOG.md", "BUILD-LOG-format-v2.md", "BUILD-LOG-soak.md"],
    )
    (proj / "_planrunner" / "spike").mkdir()

    runs = find_runs(tmp_path)
    assert sorted(r.slug for r in runs) == ["format-v2", "main", "soak"]
    assert all(r.slug != "spike" for r in runs)

    by_slug = {r.slug: r for r in runs}
    assert by_slug["main"].build_log.name == "BUILD-LOG.md"
    assert by_slug["format-v2"].build_log.name == "BUILD-LOG-format-v2.md"
    assert by_slug["soak"].build_log.name == "BUILD-LOG-soak.md"


# --- Fix round 2: run-scoped session attribution -----------------------------
#
# `find_sessions` maps a project to one encoded transcript directory. A run
# needs a different unit entirely: the sessions that actually drove *it*.
# The helpers below build synthetic transcripts under a fake projects root so
# every dimension of the two-filter rule (reference AND time overlap) can be
# exercised in isolation.


def _run(project, state_dir_name, created, updates):
    """A RunRef whose ledger opens at `created` and whose steps close at
    each timestamp in `updates`."""
    sd = project / "_planrunner" / state_dir_name
    sd.mkdir(parents=True, exist_ok=True)
    ledger = sd / "ledger.json"
    ledger.write_text(json.dumps({
        "created": created,
        "steps": [{"id": f"S{i}", "status": "complete", "updated": u}
                  for i, u in enumerate(updates, 1)]}))
    return RunRef(project=project, state_dir=sd, ledger=ledger,
                  build_log=None, gpt_dir=None, slug=state_dir_name,
                  layout=BY_MARKER["_planrunner"])


def _session(projects_root, encoded_dir, name, turns, cwd="/anywhere"):
    """A transcript under `projects_root/encoded_dir`. Each turn is
    `(timestamp, command_or_None)`; a command is placed in a Bash tool_use
    input, exactly where a real runner invocation appears. `cwd` defaults to
    a value that belongs to no real project, which is fine for every fixture
    that references a state dir by its absolute form (cwd-independent by
    design); a fixture exercising a *relative*-only reference must pass the
    project the session was actually driven from, since that is now the
    signal that tells two projects sharing the same relative state-dir name
    apart."""
    d = projects_root / encoded_dir
    d.mkdir(parents=True, exist_ok=True)
    records = []
    for i, (ts, command) in enumerate(turns, 1):
        content = []
        if command is not None:
            content.append({"type": "tool_use", "id": f"toolu_{name}_{i}",
                            "name": "Bash", "input": {"command": command}})
        records.append({
            "type": "assistant", "timestamp": ts, "cwd": cwd,
            "sessionId": name,
            "message": {"model": "claude-opus-4-8",
                        "usage": {"input_tokens": 1, "output_tokens": 100,
                                  "cache_creation_input_tokens": 0,
                                  "cache_read_input_tokens": 0},
                        "content": content}})
    p = d / f"{name}.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return p


def test_session_in_an_unrelated_encoded_directory_is_still_found(tmp_path):
    """The voice-imessage case. The run lives in `<tmp>/voice` but the work
    was driven from a session whose cwd -- and therefore whose encoded
    transcript directory -- is the home directory. `find_sessions` returns
    nothing for that project; the run-scoped finder must still find it via
    the state-dir reference plus the time overlap.

    An implementation that kept mapping project -> encoded directory would
    return [] here, which is exactly the zero-everywhere symptom."""
    projects_root = tmp_path / "projects"
    project = tmp_path / "voice"
    ref = _run(project, "state", "2026-07-12T20:30:00Z",
               ["2026-07-13T07:00:00Z"])

    sess = _session(projects_root, "-Users-example", "driver", [
        ("2026-07-12T21:00:00Z", f'SD={ref.state_dir}; "$L" --state-dir "$SD"'),
        ("2026-07-13T06:00:00Z", None)])

    # The encoded directory for this project does not even exist, so the old
    # project-keyed lookup has nothing to return.
    assert find_sessions(str(project), projects_root) == []
    assert run_sessions(ref, projects_root) == [sess]


def test_session_referencing_the_state_dir_but_outside_the_span_is_excluded(
        tmp_path):
    """Over-selection is the failure mode string matching alone produces: a
    later analysis session quotes the same state dir while having nothing to
    do with the run. Two sessions reference the identical path; only the one
    whose turns straddle the ledger span may be selected.

    Drop the overlap filter and both come back."""
    projects_root = tmp_path / "projects"
    project = tmp_path / "voice"
    ref = _run(project, "state", "2026-07-12T20:30:00Z",
               ["2026-07-13T07:00:00Z"])
    cmd = f'SD={ref.state_dir}; "$L" --state-dir "$SD"'

    during = _session(projects_root, "-Users-example", "during",
                      [("2026-07-12T21:00:00Z", cmd)])
    _session(projects_root, "-Users-example", "much-later",
             [("2026-07-26T18:00:00Z", cmd)])

    assert run_sessions(ref, projects_root) == [during]


def test_session_overlapping_in_time_but_never_naming_the_run_is_excluded(
        tmp_path):
    """The converse filter. A session running concurrently with the run --
    on entirely unrelated work -- must not be attributed to it. Drop the
    reference filter and every contemporaneous session is swept in, which is
    the "one project's whole spend on every run" symptom in another guise."""
    projects_root = tmp_path / "projects"
    project = tmp_path / "voice"
    ref = _run(project, "state", "2026-07-12T20:30:00Z",
               ["2026-07-13T07:00:00Z"])

    _session(projects_root, "-Users-example", "unrelated",
             [("2026-07-12T21:00:00Z", "git status"),
              ("2026-07-12T22:00:00Z", None)])

    assert run_sessions(ref, projects_root) == []


def test_a_state_dir_is_not_matched_by_a_longer_sibling_name(tmp_path):
    """`_planrunner/state` is a literal prefix of `_planrunner/state-soak`,
    in both the absolute and the project-relative form. A session that only
    ever names `state-soak` therefore contains the string `state` too. Plain
    `needle in blob` would hand the soak session to the `main` run as well,
    silently re-merging two runs whose separation is the whole point.

    Both runs share a span here, so time overlap cannot do this work: only
    the name-boundary rule can."""
    projects_root = tmp_path / "projects"
    project = tmp_path / "wiki"
    main = _run(project, "state", "2026-06-26T06:00:00Z",
                ["2026-06-26T08:00:00Z"])
    soak = _run(project, "state-soak", "2026-06-26T06:00:00Z",
                ["2026-06-26T08:00:00Z"])

    sess = _session(projects_root, "-Users-example", "soak-only", [
        ("2026-06-26T07:00:00Z",
         f'SD={soak.state_dir}; "$L" --state-dir "$SD" show')])

    found = sessions_for_runs([main, soak], projects_root)
    assert found[soak.state_dir] == [sess]
    assert found[main.state_dir] == []


def test_the_project_relative_path_form_is_matched(tmp_path):
    """Real invocations use both the absolute path and the path relative to
    the project root. A session that only ever writes the relative form must
    still be attributed; matching the absolute path alone returns []."""
    projects_root = tmp_path / "projects"
    project = tmp_path / "wiki"
    ref = _run(project, "state-soak", "2026-06-26T06:00:00Z",
               ["2026-06-26T08:00:00Z"])

    # A relative-only reference also needs the session's own cwd to agree
    # with the project it names -- give it the project it was really driven
    # from, exactly as a real transcript would.
    sess = _session(projects_root, "-Users-example", "rel", [
        ("2026-06-26T07:00:00Z",
         'SD=_planrunner/state-soak; "$L" --state-dir "$SD" show')],
        cwd=str(project))

    assert str(ref.state_dir) not in sess.read_text()  # absolute form absent
    assert run_sessions(ref, projects_root) == [sess]


def test_a_path_mentioned_only_in_prose_is_not_a_reference(tmp_path):
    """The reference signal is a *tool input*, not any occurrence in the
    transcript. A session that merely discusses the state dir in assistant
    text -- a review or a post-mortem -- did not drive the run. Matching the
    raw JSONL line instead of the tool_use inputs would select this."""
    projects_root = tmp_path / "projects"
    project = tmp_path / "wiki"
    ref = _run(project, "state-soak", "2026-06-26T06:00:00Z",
               ["2026-06-26T08:00:00Z"])

    d = projects_root / "-Users-example"
    d.mkdir(parents=True)
    rec = {"type": "assistant", "timestamp": "2026-06-26T07:00:00Z",
           "sessionId": "prose",
           "message": {"model": "claude-opus-4-8",
                       "usage": {"input_tokens": 1, "output_tokens": 100,
                                 "cache_creation_input_tokens": 0,
                                 "cache_read_input_tokens": 0},
                       "content": [{"type": "text",
                                    "text": f"I looked at {ref.state_dir} "
                                            "and it seems fine."}]}}
    (d / "prose.jsonl").write_text(json.dumps(rec) + "\n")

    assert str(ref.state_dir) in (d / "prose.jsonl").read_text()
    assert run_sessions(ref, projects_root) == []


def test_every_transcript_is_read_once_regardless_of_run_count(tmp_path,
                                                               monkeypatch):
    """Scanning every project directory is the price of correct attribution;
    re-scanning it per run would multiply it by the run count. Three runs
    over three transcripts must still be three file reads.

    A per-run implementation (`[run_sessions(r, root) for r in refs]`) reads
    nine and fails this."""
    projects_root = tmp_path / "projects"
    project = tmp_path / "wiki"
    refs = [_run(project, f"state-{n}", "2026-06-26T06:00:00Z",
                 ["2026-06-26T08:00:00Z"]) for n in ("a", "b", "c")]
    for n in ("s1", "s2", "s3"):
        _session(projects_root, "-Users-example", n,
                 [("2026-06-26T07:00:00Z", "git status")])

    import agentview.discovery as disc
    reads = []
    real = disc._scan_session
    monkeypatch.setattr(disc, "_scan_session",
                        lambda p, n: (reads.append(p), real(p, n))[1])

    disc.sessions_for_runs(refs, projects_root)
    assert len(reads) == 3
    assert len(set(reads)) == 3


def test_missing_projects_root_yields_empty_lists_for_every_run(tmp_path):
    """A machine with no transcripts at all must degrade to "no sessions",
    not raise."""
    ref = _run(tmp_path / "proj", "state", "2026-06-26T06:00:00Z",
               ["2026-06-26T08:00:00Z"])
    assert sessions_for_runs([ref], tmp_path / "absent") == {ref.state_dir: []}


# --- Fix wave, item 5 (C4): an unbounded ledger must not select zero --------


def test_a_ledger_with_no_parseable_updated_still_selects_its_sessions(
        tmp_path):
    """`run_span` returned `end=None` and `_overlaps` treated a missing bound
    as fatal, so a freshly started run -- no completed step yet, therefore no
    parseable `updated` anywhere -- was handed zero sessions and rendered
    completely empty. A missing bound is *open*, not fatal; the reference
    signal still has to hold, so this is not a return to "everything"."""
    projects_root = tmp_path / "projects"
    project = tmp_path / "fresh"
    sd = project / "_planrunner" / "state"
    sd.mkdir(parents=True)
    (sd / "ledger.json").write_text(json.dumps({
        "created": "2026-07-12T20:30:00Z",
        "steps": [{"id": "S1", "status": "in-progress", "updated": None}]}))
    ref = RunRef(project=project, state_dir=sd, ledger=sd / "ledger.json",
                 build_log=None, gpt_dir=None, slug="main",
                 layout=BY_MARKER["_planrunner"])

    driver = _session(projects_root, "-Users-example", "driver", [
        ("2026-07-12T20:31:00Z", f'SD={sd}; "$L" --state-dir "$SD" set')])
    _session(projects_root, "-Users-example", "unrelated", [
        ("2026-07-12T20:31:00Z", "git status")])

    assert run_sessions(ref, projects_root) == [driver]


def test_a_live_run_picks_up_a_session_that_starts_after_the_last_write(
        tmp_path):
    """The live pane's case. The ledger's last `updated` is the last time the
    orchestrator wrote state; a session opened after it is the one working
    right now. Under the closed span it is filtered out, so the pane could
    never see the session it exists to watch."""
    projects_root = tmp_path / "projects"
    project = tmp_path / "proj"
    ref = _run(project, "state", "2026-07-12T20:30:00Z",
               ["2026-07-12T21:00:00Z"])
    later = _session(projects_root, "-Users-example", "resumed", [
        ("2026-07-12T22:00:00Z", f'SD={ref.state_dir}; "$L" --state-dir "$SD"')])

    assert run_sessions(ref, projects_root) == []
    assert run_sessions(ref, projects_root, live=True) == [later]


# --- Fix wave, item 6: the session filter is correct; pin what it excludes ---
#
# Eight sessions in the real corpus name the voice-imessage state dir and
# exactly one survives the reference-AND-overlap filter. Inspecting all eight
# showed the filter is right, not too tight: one genuinely drove the run
# (`SD=.../secondary-project/_planrunner/state; "$L" --state-dir "$SD"
# init/set`), three were later analysis sessions quoting the path, and three
# were plan-runner sessions driving *primary-project's own* `_planrunner/state`
# -- matched only because the project-relative needle `_planrunner/state` is
# identical for every project. That last class is the sharp one and was not
# pinned by any existing test.


def test_a_session_driving_another_projects_run_is_not_swept_in_by_the_relative_form(
        tmp_path):
    """Two projects, each with a `_planrunner/state`. The relative needle
    `_planrunner/state` cannot tell them apart -- it is byte-identical for
    both -- so the reference signal alone hands wiki's orchestrator session
    to the voice run as well. Only the span separates them.

    This is the mirror image of the double-counting finding: loosening the
    filter here would make a run report another project's entire spend."""
    projects_root = tmp_path / "projects"
    voice = _run(tmp_path / "voice", "state", "2026-07-12T20:30:00Z",
                 ["2026-07-13T07:00:00Z"])
    wiki = _run(tmp_path / "wiki", "state", "2026-06-25T12:00:00Z",
                ["2026-06-27T08:00:00Z"])

    # This session's own cwd is wiki's project root -- exactly what the cwd
    # guard added for the relative-only match needs to agree with, since the
    # reference text alone (below) cannot distinguish the two projects.
    driver = _session(projects_root, "-Users-example", "wiki-driver", [
        ("2026-06-25T13:00:00Z",
         'cd ~/wiki && "$L" --state-dir _planrunner/state set --id F.3')],
        cwd=str(tmp_path / "wiki"))

    # The reference signal on its own matches both runs...
    scan = _scan_session(driver, {str(voice.state_dir): _needles(voice),
                                  str(wiki.state_dir): _needles(wiki)})
    assert scan.references == {str(voice.state_dir), str(wiki.state_dir)}

    # ...and only the span keeps them apart.
    found = sessions_for_runs([voice, wiki], projects_root)
    assert found[wiki.state_dir] == [driver]
    assert found[voice.state_dir] == []


# --- max-runner discovery ----------------------------------------------------
#
# max-runner's layout inverts plan-runner's: `<project>/_maxrunner/<slug>/state/
# ledger.json`. The slug is the *parent* of the state directory, and the state
# directory is always literally named "state" -- the reverse of plan-runner,
# where the state directory itself carries the slug (via `state-<suffix>`
# stripping, or the directory's own name). A naive addition of "_maxrunner" to
# the marker-name tuple, still deriving slug from the state dir's own name,
# would read "state" every time and report every max-runner run as "main".
# Every test below uses at least two differently-named slug directories so a
# single-slug fixture (which "main" x1 could accidentally satisfy) can't hide
# that bug.


def _maxrunner_project(tmp_path, slugs, build_logs=()):
    proj = tmp_path / "proj"
    for slug in slugs:
        d = proj / "_maxrunner" / slug / "state"
        d.mkdir(parents=True)
        (d / "ledger.json").write_text(json.dumps({"steps": []}))
    for lg in build_logs:
        (proj / lg).write_text("")
    return proj


def test_maxrunner_slug_is_the_parent_directory_not_the_state_dir_name(tmp_path):
    """Two slugs, both with a state dir literally named "state". If slug were
    derived the plan-runner way (from the state dir's own name, "state" ->
    "main"), both would collapse to "main" -- one slug instead of two, and the
    wrong one. Only reading the parent directory's name survives this."""
    proj = _maxrunner_project(tmp_path, ["alpha", "beta"])
    runs = find_runs(tmp_path)
    assert {r.slug for r in runs} == {"alpha", "beta"}
    assert all(r.state_dir.name == "state" for r in runs)
    assert all(r.project == proj for r in runs)


def test_maxrunner_state_dir_is_slug_dir_over_state_not_slug_dir_itself(tmp_path):
    """Pins the exact path shape: `_maxrunner/<slug>/state`, not
    `_maxrunner/<slug>` itself (which is what a lightrunner-style "the marker
    child is the run" reading would produce)."""
    proj = _maxrunner_project(tmp_path, ["alpha", "beta"])
    by_slug = {r.slug: r for r in find_runs(tmp_path)}
    assert by_slug["alpha"].state_dir == proj / "_maxrunner" / "alpha" / "state"
    assert by_slug["beta"].state_dir == proj / "_maxrunner" / "beta" / "state"
    assert by_slug["alpha"].ledger == \
        proj / "_maxrunner" / "alpha" / "state" / "ledger.json"


def test_maxrunner_scaffolded_but_empty_run_is_ignored_without_crashing(tmp_path):
    """The exact shape of the live demo-run run when checked: four empty
    directories under state/ and no ledger.json anywhere. Must not crash and
    must not be reported as a run."""
    proj = tmp_path / "proj"
    slug_dir = proj / "_maxrunner" / "demo-run"
    for sub in ("briefs", "recon", "reports"):
        (slug_dir / sub).mkdir(parents=True)
    for sub in ("gates", "records", "changes", "tracks", "sweeps", "md"):
        (slug_dir / "state" / sub).mkdir(parents=True)
    assert find_runs(tmp_path) == []


def test_maxrunner_run_is_picked_up_as_soon_as_its_ledger_appears(tmp_path):
    """Same scaffold as above, but this time a ledger.json lands in state/ --
    the run must now be discovered, proving pickup needs no other signal."""
    proj = tmp_path / "proj"
    slug_dir = proj / "_maxrunner" / "demo-run"
    for sub in ("gates", "records"):
        (slug_dir / "state" / sub).mkdir(parents=True)
    (slug_dir / "state" / "ledger.json").write_text(json.dumps({"steps": []}))
    runs = find_runs(tmp_path)
    assert [r.slug for r in runs] == ["demo-run"]


def test_maxrunner_build_log_is_the_unsuffixed_project_root_log(tmp_path):
    """max-runner never pairs a suffixed BUILD-LOG-<slug>.md the way
    plan-runner does -- every run under one project's `_maxrunner/` shares the
    single unsuffixed BUILD-LOG.md at the project root. Both slugs here must
    resolve to the *same* file object, and a suffixed file that happens to
    exist (mimicking someone assuming the plan-runner convention carries over)
    must be ignored rather than preferred."""
    proj = _maxrunner_project(tmp_path, ["alpha", "beta"],
                              build_logs=["BUILD-LOG.md", "BUILD-LOG-alpha.md"])
    by_slug = {r.slug: r for r in find_runs(tmp_path)}
    assert by_slug["alpha"].build_log == proj / "BUILD-LOG.md"
    assert by_slug["beta"].build_log == proj / "BUILD-LOG.md"


def test_maxrunner_missing_build_log_is_none(tmp_path):
    proj = _maxrunner_project(tmp_path, ["alpha"])
    assert find_runs(tmp_path)[0].build_log is None


def test_maxrunner_gpt_dir_is_none_even_when_gates_gpt_and_md_both_exist(
        tmp_path):
    """The MD tier's gate-round JSON lands under `state/md/`: the skill passes
    `--emit-dir _maxrunner/<slug>/state/md --step <id>` on every Gate 0/A/B
    round, and gate.py's `write_round` drops `<emit_dir>/<step>/<gate>-r<n>.json`
    there. `state/gates/gpt/` -- plan-runner's own shape -- is a false cognate
    for max-runner and must never be read, even when populated (accidental
    reuse). Populate *both* `state/gates/gpt/` and `state/md/` with a
    plausible round file and confirm only `state/md/` is wired: `gpt_dir`
    must resolve there, not to `None` and not to `gates/gpt`."""
    proj = _maxrunner_project(tmp_path, ["alpha"])
    sd = proj / "_maxrunner" / "alpha" / "state"
    (sd / "gates" / "gpt").mkdir(parents=True)
    (sd / "gates" / "gpt" / "0-r1.json").write_text(json.dumps(
        {"report": {"risks": []}, "classification": {"blocking": []},
         "outcome": "pass"}))
    (sd / "md").mkdir(parents=True)
    (sd / "md" / "0-r1.json").write_text(json.dumps(
        {"report": {"risks": []}, "classification": {"blocking": []},
         "outcome": "pass"}))
    gpt_dir = find_runs(tmp_path)[0].gpt_dir
    assert gpt_dir is not None
    assert gpt_dir.name == "md"


def test_maxrunner_and_planrunner_runs_coexist_without_merging(tmp_path):
    """A project running both skills side by side (plausible during a
    migration) must yield two distinct runs, each with its own correctly
    shaped state dir -- neither marker's handling may leak into the other's."""
    proj = tmp_path / "proj"
    pr = proj / "_planrunner" / "state"
    pr.mkdir(parents=True)
    (pr / "ledger.json").write_text(json.dumps({"steps": []}))
    mr = proj / "_maxrunner" / "mr-x" / "state"
    mr.mkdir(parents=True)
    (mr / "ledger.json").write_text(json.dumps({"steps": []}))

    runs = find_runs(tmp_path)
    assert {r.slug for r in runs} == {"main", "mr-x"}
    by_slug = {r.slug: r for r in runs}
    assert by_slug["main"].state_dir == pr
    assert by_slug["mr-x"].state_dir == mr


def test_maxrunner_out_inside_the_project_is_refused(tmp_path):
    """The read-only --out guard (surfaces/report.py) must refuse a max-runner
    project's own tree exactly as it refuses plan-runner's -- it keys off
    `RunRef.project`, computed generically, so this pins that the new
    discovery path still yields the right project root for the guard to
    check."""
    from agentview.cli import main as cli_main

    proj = _maxrunner_project(tmp_path, ["demo-run"])
    out = proj / "_maxrunner" / "demo-run" / "state" / "agentview-report.html"
    no_sessions = ["--projects-root", "/nonexistent-projects-root-for-tests"]

    rc = cli_main(["report", str(proj), "--out", str(out), *no_sessions])
    assert rc != 0
    assert not out.exists()


def test_a_relative_needle_does_not_cross_match_another_project(
        tmp_path, assistant):
    for proj in ("a", "b"):
        sd = tmp_path / "projects" / proj / "_planrunner" / "state"
        sd.mkdir(parents=True)
        # `created` gives the span an earlier bound than its one `updated`
        # stamp, so the 11:30 session below -- deliberately before the step
        # completed at 12:00 -- falls inside [start, end] instead of being
        # excluded by span filtering, which the acceptance criterion requires
        # to be a non-factor here (spans overlap for both a and b regardless).
        (sd / "ledger.json").write_text(json.dumps({
            "created": "2026-07-20T11:00:00Z",
            "steps": [{"id": "1", "status": "complete",
                       "updated": "2026-07-20T12:00:00Z"}]}))

    root = tmp_path / "sessions"
    (root / "proj-a").mkdir(parents=True)
    rec = assistant("2026-07-20T11:30:00Z", tools=[
        {"type": "tool_use", "id": "t1", "name": "Bash",
         "input": {"command": "ledger.py --state-dir _planrunner/state"}}])
    rec["cwd"] = str(tmp_path / "projects" / "a")
    (root / "proj-a" / "s1.jsonl").write_text(json.dumps(rec) + "\n")

    refs = find_runs(tmp_path / "projects")
    by_name = {r.project.name: r for r in refs}
    got = sessions_for_runs(list(refs), root)
    assert len(got[by_name["a"].state_dir]) == 1
    assert got[by_name["b"].state_dir] == []


def test_an_absolute_needle_matches_from_any_cwd(tmp_path, assistant):
    sd = tmp_path / "projects" / "a" / "_planrunner" / "state"
    sd.mkdir(parents=True)
    # Same `created` widening as the cross-match test above, so the 11:30
    # session falls inside the run's span rather than being excluded by span
    # filtering for reasons unrelated to the cwd guard under test.
    (sd / "ledger.json").write_text(json.dumps({
        "created": "2026-07-20T11:00:00Z",
        "steps": [{"id": "1", "status": "complete",
                   "updated": "2026-07-20T12:00:00Z"}]}))
    root = tmp_path / "sessions"
    (root / "elsewhere").mkdir(parents=True)
    rec = assistant("2026-07-20T11:30:00Z", tools=[
        {"type": "tool_use", "id": "t1", "name": "Bash",
         "input": {"command": f"ledger.py --state-dir {sd}"}}])
    rec["cwd"] = "/somewhere/else/entirely"
    (root / "elsewhere" / "s1.jsonl").write_text(json.dumps(rec) + "\n")
    refs = find_runs(tmp_path / "projects")
    assert len(sessions_for_runs(list(refs), root)[refs[0].state_dir]) == 1


# --- Gate B round 1 fix round: R1/R2/R4 -------------------------------------
#
# An absolute needle substring-contains its own relative form
# (`/proj/_planrunner/state` also names `_planrunner/state`), so a single
# session can register both an absolute hit (no cwd guard) and a
# relative-only hit (cwd guard) for the very same ref. The three tests below
# pin, respectively: a scan must contribute at most one session per ref no
# matter how many of its needle-forms matched (R1); a relative-only hit must
# be judged against the cwd of the record it actually occurred in, not
# whichever record's cwd a session-wide variable happened to capture first
# (R2); and a non-string `cwd` in a malformed transcript record must degrade
# to "no match" rather than crash the `.startswith` guard (R4).


def test_an_absolute_reference_is_not_double_counted_when_cwd_also_matches(
        tmp_path):
    """The session below names the run only via the ABSOLUTE state-dir path,
    and its own cwd also happens to equal that ref's project -- entirely
    plausible, since the invocation really was made from there. Before the
    fix, this session matched twice: once via the absolute key (no cwd guard
    needed) and once via the shared relative key (cwd guard passes, because
    the cwd genuinely is inside this project), appending the same path to
    the run's session list twice and double-counting every turn and token in
    it. One session driving one run must produce exactly one entry."""
    projects_root = tmp_path / "projects"
    project = tmp_path / "proj"
    ref = _run(project, "state", "2026-07-20T11:00:00Z",
               ["2026-07-20T12:00:00Z"])

    sess = _session(projects_root, "-Users-example", "abs-and-matching-cwd", [
        ("2026-07-20T11:30:00Z", f'"$L" --state-dir {ref.state_dir}')],
        cwd=str(project))

    got = run_sessions(ref, projects_root)
    assert got == [sess]
    assert len(got) == 1


def test_a_relative_hit_is_judged_by_its_own_records_cwd_not_the_sessions_first(
        tmp_path):
    """A single transcript can span several runner invocations from
    different project directories (see the module docstring). The session
    below has two assistant records: the first belongs to project A and
    names nothing; the actual relative-form invocation of `_planrunner/
    state` happens in the SECOND record, whose cwd correctly names project
    B. A session-wide 'first cwd' would judge that later match against A's
    cwd instead -- wrongly admitting A and wrongly rejecting B. Only the
    cwd of the record the match actually occurred in may decide it."""
    projects_root = tmp_path / "projects"
    proj_a = tmp_path / "a"
    proj_b = tmp_path / "b"
    ref_a = _run(proj_a, "state", "2026-07-20T10:00:00Z",
                ["2026-07-20T13:00:00Z"])
    ref_b = _run(proj_b, "state", "2026-07-20T10:00:00Z",
                ["2026-07-20T13:00:00Z"])

    def _rec(ts, cwd, command):
        content = []
        if command is not None:
            content.append({"type": "tool_use", "id": f"toolu_{ts}",
                            "name": "Bash", "input": {"command": command}})
        return {"type": "assistant", "timestamp": ts, "cwd": cwd,
                "sessionId": "spans-two-projects",
                "message": {"model": "claude-opus-4-8",
                            "usage": {"input_tokens": 1, "output_tokens": 100,
                                      "cache_creation_input_tokens": 0,
                                      "cache_read_input_tokens": 0},
                            "content": content}}

    d = projects_root / "-Users-example"
    d.mkdir(parents=True)
    records = [
        _rec("2026-07-20T11:00:00Z", str(proj_a), "git status"),
        _rec("2026-07-20T12:00:00Z", str(proj_b),
             '"$L" --state-dir _planrunner/state show'),
    ]
    sess = d / "spans-two-projects.jsonl"
    sess.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    found = sessions_for_runs([ref_a, ref_b], projects_root)
    assert found[ref_b.state_dir] == [sess]
    assert found[ref_a.state_dir] == []


def test_a_relative_hit_is_judged_by_every_records_cwd_not_just_the_last(
        tmp_path):
    """R2, the sharper case: ONE session invokes the shared relative form
    `_planrunner/state` TWICE at different points in time -- once from inside
    project A's directory, later once from inside project B's directory, with
    both projects holding a state dir named `_planrunner/state`. The relative
    needle cannot tell A and B apart; only each record's own cwd can, and
    BOTH cwds were genuinely observed for this key within this scan.

    A `rel_cwd` shaped as one value per key (`dict[str, str | None]`)
    overwrites the earlier (A) cwd with the later (B) one, so when project
    A's guard is checked it wrongly sees B's cwd and rejects a legitimate
    match -- the earlier attribution is silently lost even though nothing
    about A's invocation was wrong. Both A and B must receive this session."""
    projects_root = tmp_path / "projects"
    proj_a = tmp_path / "a"
    proj_b = tmp_path / "b"
    ref_a = _run(proj_a, "state", "2026-07-20T10:00:00Z",
                ["2026-07-20T13:00:00Z"])
    ref_b = _run(proj_b, "state", "2026-07-20T10:00:00Z",
                ["2026-07-20T13:00:00Z"])

    def _rec(ts, cwd):
        return {"type": "assistant", "timestamp": ts, "cwd": cwd,
                "sessionId": "spans-two-projects-twice",
                "message": {"model": "claude-opus-4-8",
                            "usage": {"input_tokens": 1, "output_tokens": 100,
                                      "cache_creation_input_tokens": 0,
                                      "cache_read_input_tokens": 0},
                            "content": [
                                {"type": "tool_use", "id": f"toolu_{ts}",
                                 "name": "Bash",
                                 "input": {"command":
                                           '"$L" --state-dir '
                                           '_planrunner/state show'}}]}}

    d = projects_root / "-Users-example"
    d.mkdir(parents=True)
    records = [
        _rec("2026-07-20T11:00:00Z", str(proj_a)),   # matched while in A
        _rec("2026-07-20T12:00:00Z", str(proj_b)),   # matched later, while in B
    ]
    sess = d / "spans-two-projects-twice.jsonl"
    sess.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    found = sessions_for_runs([ref_a, ref_b], projects_root)
    assert found[ref_a.state_dir] == [sess]  # the earlier A cwd is not lost
    assert found[ref_b.state_dir] == [sess]  # nor does keeping it cost B


def test_a_non_string_cwd_does_not_raise_and_is_simply_not_a_relative_match(
        tmp_path):
    """A malformed transcript can carry a non-string `cwd` -- a number here,
    but a list or object are the same defect. `sessions_for_runs` reads the
    cwd back and calls `.startswith` on it for a relative-only match; without
    a type check at capture time that raises `AttributeError` and aborts
    discovery for every run from one bad external record. Discovery must
    degrade leniently instead: the malformed cwd is simply unavailable, so
    the relative-only reference it would have gated fails to match."""
    projects_root = tmp_path / "projects"
    project = tmp_path / "proj"
    ref = _run(project, "state", "2026-07-20T11:00:00Z",
               ["2026-07-20T12:00:00Z"])

    d = projects_root / "-Users-example"
    d.mkdir(parents=True)
    rec = {"type": "assistant", "timestamp": "2026-07-20T11:30:00Z",
           "cwd": 12345, "sessionId": "malformed-cwd",
           "message": {"model": "claude-opus-4-8",
                       "usage": {"input_tokens": 1, "output_tokens": 100,
                                 "cache_creation_input_tokens": 0,
                                 "cache_read_input_tokens": 0},
                       "content": [{"type": "tool_use", "id": "t1",
                                    "name": "Bash",
                                    "input": {"command":
                                              '"$L" --state-dir '
                                              '_planrunner/state show'}}]}}
    (d / "malformed-cwd.jsonl").write_text(json.dumps(rec) + "\n")

    got = sessions_for_runs([ref], projects_root)  # must not raise
    assert got == {ref.state_dir: []}
