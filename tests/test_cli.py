import json
from agentview.cli import main

# Session attribution now scans every project directory under the projects
# root. Point the CLI at a directory that does not exist so these tests
# depend only on their own tmp_path fixtures and never on whatever
# transcripts happen to live on the machine running them.
NO_SESSIONS = ["--projects-root", "/nonexistent-projects-root-for-tests"]


def _project(tmp_path):
    sd = tmp_path / "proj" / "_planrunner" / "state"
    sd.mkdir(parents=True)
    (sd / "ledger.json").write_text(json.dumps({"steps": [
        {"id": "A", "status": "complete", "owner": "b", "tier": "sonnet",
         "attempts": 1, "deliverable": "a.md", "depends_on": [],
         "updated": "2026-06-25T11:00:00Z"}]}))
    return tmp_path


def test_report_writes_the_requested_file(tmp_path):
    root = _project(tmp_path)
    out = tmp_path / "r.html"
    assert main(["report", str(root), "--out", str(out), *NO_SESSIONS]) == 0
    assert out.exists() and "agentview report" in out.read_text()


def test_report_on_a_root_with_no_runs_exits_nonzero(tmp_path):
    assert main(["report", str(tmp_path), "--out", str(tmp_path / "r.html"),
                 *NO_SESSIONS]) == 1


def test_watch_honours_iteration_bound(tmp_path, capsys):
    root = _project(tmp_path)
    assert main(["watch", str(root), "--iterations", "1", "--interval", "0",
                 *NO_SESSIONS]) == 0
    assert "plan-runner" in capsys.readouterr().out


def test_unknown_subcommand_is_rejected(tmp_path):
    import pytest
    with pytest.raises(SystemExit):
        main(["frobnicate"])


# --- Tests added beyond the brief -----------------------------------------
#
# The brief only proves that `report` exits non-zero when no run is found.
# `watch` and `replay` share the same "no runs found" branch in cli.py but
# were never exercised by a test that could tell a return of 1 from a crash
# or a return of 0. See task-13-report.md for the mutation evidence.

def test_watch_exits_nonzero_when_no_run_found(tmp_path):
    assert main(["watch", str(tmp_path), "--iterations", "1", "--interval", "0",
                 *NO_SESSIONS]) == 1


def test_replay_exits_nonzero_when_no_run_found(tmp_path):
    assert main(["replay", str(tmp_path), *NO_SESSIONS]) == 1


def _multi_run_project(tmp_path):
    """Two runs under one project, with distinguishable step ids and
    deliverables so a test can prove --slug picked the right one rather
    than merely picking *a* run.

    Both deliverables are written to disk: the pane lists only entries that
    resolve to a real file, so bare names with nothing behind them would be
    reported as a prose count and neither marker would appear in the frame,
    costing this test the half of its evidence that comes from deliverables."""
    base = tmp_path / "proj" / "_planrunner"
    (tmp_path / "proj").mkdir(parents=True, exist_ok=True)
    (tmp_path / "proj" / "main-only.md").write_text("main deliverable\n")
    (tmp_path / "proj" / "soak-only.md").write_text("soak deliverable\n")
    main_sd = base / "state"
    main_sd.mkdir(parents=True)
    main_sd.joinpath("ledger.json").write_text(json.dumps({"steps": [
        {"id": "STEPMAIN", "status": "complete", "owner": "b", "tier": "sonnet",
         "attempts": 1, "deliverable": "main-only.md", "depends_on": [],
         "updated": "2026-06-25T11:00:00Z"}]}))
    soak_sd = base / "state-soak"
    soak_sd.mkdir(parents=True)
    soak_sd.joinpath("ledger.json").write_text(json.dumps({"steps": [
        {"id": "STEPSOAK", "status": "complete", "owner": "b", "tier": "sonnet",
         "attempts": 1, "deliverable": "soak-only.md", "depends_on": [],
         "updated": "2026-06-25T11:00:00Z"}]}))
    return tmp_path


def test_slug_selects_the_correct_run_among_several(tmp_path, capsys):
    root = _multi_run_project(tmp_path)
    assert main(["watch", str(root), "--slug", "soak",
                 "--iterations", "1", "--interval", "0", *NO_SESSIONS]) == 0
    out = capsys.readouterr().out
    assert "run: soak" in out
    assert "run: main" not in out
    assert "STEPSOAK" in out
    assert "STEPMAIN" not in out
    assert "soak-only.md" in out
    assert "main-only.md" not in out


def _single_run_project(root_dir, step_id, deliverable):
    """A standalone project (its own root) with one run inside it."""
    sd = root_dir / "proj" / "_planrunner" / "state"
    sd.mkdir(parents=True)
    sd.joinpath("ledger.json").write_text(json.dumps({"steps": [
        {"id": step_id, "status": "complete", "owner": "b", "tier": "sonnet",
         "attempts": 1, "deliverable": deliverable, "depends_on": [],
         "updated": "2026-06-25T11:00:00Z"}]}))


def test_report_across_multiple_roots_finds_the_one_with_a_run(tmp_path):
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    has_run_root = tmp_path / "has-run"
    _single_run_project(has_run_root, "ONLYSTEP", "only.md")

    out = tmp_path / "r.html"
    assert main(["report", str(empty_root), str(has_run_root),
                 "--out", str(out), *NO_SESSIONS]) == 0
    text = out.read_text()
    assert "agentview report" in text
    assert "ONLYSTEP" in text


def test_report_across_multiple_roots_includes_every_root(tmp_path):
    """Two roots, each with its own distinct run. A flattening bug that
    silently drops one root (e.g. `roots[0]` instead of iterating all of
    `args.roots`) would still exit 0 and still render a report -- it would
    just be missing content. Only asserting on both runs' markers being
    present catches that."""
    root_a = tmp_path / "root-a"
    _single_run_project(root_a, "STEPFROMA", "a-only.md")
    root_b = tmp_path / "root-b"
    _single_run_project(root_b, "STEPFROMB", "b-only.md")

    out = tmp_path / "r.html"
    assert main(["report", str(root_a), str(root_b), "--out", str(out),
                 *NO_SESSIONS]) == 0
    text = out.read_text()
    assert "STEPFROMA" in text
    assert "STEPFROMB" in text
    assert "a-only.md" in text
    assert "b-only.md" in text


# --- Fix wave: CLI wiring ----------------------------------------------------

def test_report_exits_nonzero_when_out_is_inside_a_run_project(tmp_path):
    """The read-only violation must surface as a failed command, not a
    warning on a successful one. Behaviour of the guard itself is pinned in
    test_readonly.py; this pins that the CLI propagates it."""
    root = _project(tmp_path)
    out = root / "proj" / "_planrunner" / "state" / "r.html"
    assert main(["report", str(root), "--out", str(out), *NO_SESSIONS]) != 0
    assert not out.exists()


def test_replay_writes_to_the_current_stdout(tmp_path, capsys):
    """cli.py called `live.replay(...)` with no `stream`, so replay used the
    default bound at import time. `capsys` reads the current stdout, which
    is what a frame written to a stale reference misses."""
    root = _project(tmp_path)
    assert main(["replay", str(root), *NO_SESSIONS]) == 0
    assert "plan-runner" in capsys.readouterr().out
