import hashlib
import json
import os
import time

import pytest

from agentview.cli import main
from agentview.discovery import find_runs
from agentview.model import build_run
from agentview.surfaces.report import ReadOnlyViolation, write_report

NO_SESSIONS = ["--projects-root", "/nonexistent-projects-root-for-tests"]


def _snapshot(root):
    return {str(p): (p.stat().st_mtime_ns, p.stat().st_size)
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_no_subcommand_modifies_the_run_directory(tmp_path):
    proj = tmp_path / "proj"
    sd = proj / "_planrunner" / "state"
    sd.mkdir(parents=True)
    (sd / "ledger.json").write_text(json.dumps({"steps": [
        {"id": "A", "status": "complete", "owner": "b", "tier": "sonnet",
         "attempts": 1, "deliverable": "a.md", "depends_on": [],
         "updated": "2026-06-25T11:00:00Z"}]}))
    (proj / "BUILD-LOG.md").write_text("25/06/2026 11:00 | A | complete | ok\n")

    before = _snapshot(proj)
    out = tmp_path / "elsewhere" / "r.html"
    no_sessions = ["--projects-root", "/nonexistent-projects-root-for-tests"]
    main(["report", str(proj), "--out", str(out), *no_sessions])
    main(["watch", str(proj), "--iterations", "1", "--interval", "0",
          *no_sessions])
    main(["replay", str(proj), *no_sessions])
    assert _snapshot(proj) == before
    assert out.exists()


# --- Tests added beyond the brief -----------------------------------------
#
# The brief's test proves that *this implementation, today* does not touch
# the project directory. It does not prove the snapshot-equality assertion
# is capable of catching a violation at all -- a comparison that always
# passes regardless of what happens on disk would make the test above
# worthless. These two tests pin down the guard mechanism itself. Separately
# (see task-13-report.md) a deliberate stray write was injected into
# cli.py's report path, the brief's test was confirmed to fail, and the
# write was reverted.

def test_readonly_snapshot_would_catch_a_stray_write(tmp_path):
    proj = tmp_path / "proj"
    sd = proj / "_planrunner" / "state"
    sd.mkdir(parents=True)
    (sd / "ledger.json").write_text(json.dumps({"steps": [
        {"id": "A", "status": "complete", "owner": "b", "tier": "sonnet",
         "attempts": 1, "deliverable": "a.md", "depends_on": [],
         "updated": "2026-06-25T11:00:00Z"}]}))

    before = _snapshot(proj)
    # Simulate exactly the kind of bug this guard exists to catch: some
    # code path writing into the run directory instead of only --out.
    (sd / "unexpected.txt").write_text("a stray write")
    assert _snapshot(proj) != before


def test_snapshot_detects_in_place_content_change_of_same_size(tmp_path):
    """A same-size overwrite leaves the set of paths identical. If the
    snapshot only recorded which paths exist, it would be blind to this.
    It must also see mtime/size so an in-place rewrite is detected."""
    f = tmp_path / "a.txt"
    f.write_text("aaaa")
    before = _snapshot(tmp_path)

    time.sleep(0.01)
    f.write_text("bbbb")  # same byte length, different content
    after = _snapshot(tmp_path)

    assert set(after) == set(before)  # path set alone would see no change
    assert after != before  # but the snapshot (mtime, size) does


# --- Fix wave, item 1 (C2): the read-only guarantee must be enforced ---------
#
# The tests above only ever write to `tmp/elsewhere`, a path that is outside
# the project by construction, so they cannot observe an unguarded writer.
# `agentview report <proj> --out <proj>/_planrunner/state/agentview-report.html`
# duly created a file inside the run's own state directory. The spec's §3 is
# unconditional, so the check has to be on the writer, not on the caller's
# good manners.


def _project_with_run(tmp_path):
    proj = tmp_path / "proj"
    sd = proj / "_planrunner" / "state"
    sd.mkdir(parents=True)
    (sd / "ledger.json").write_text(json.dumps({"steps": [
        {"id": "A", "status": "complete", "owner": "b", "tier": "sonnet",
         "attempts": 1, "deliverable": "a.md", "depends_on": [],
         "updated": "2026-06-25T11:00:00Z"}]}))
    return proj


def test_out_inside_a_run_project_is_refused_and_nothing_is_written(
        tmp_path, capsys):
    """The exact command the reviewer ran. Refusal means a clear error and a
    non-zero exit -- not a silent relocation, which would be its own lie
    about where the report went."""
    proj = _project_with_run(tmp_path)
    before = _snapshot(proj)
    out = proj / "_planrunner" / "state" / "agentview-report.html"

    rc = main(["report", str(proj), "--out", str(out), *NO_SESSIONS])

    assert rc != 0
    assert not out.exists()
    assert _snapshot(proj) == before
    assert "refusing" in capsys.readouterr().err


def test_out_reaching_a_run_project_via_dotdot_is_refused(tmp_path):
    """`..` traversal must not smuggle a path past the guard: the check
    resolves both sides before comparing. A guard written against the
    unresolved string would let this through."""
    proj = _project_with_run(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    sneaky = outside / ".." / "proj" / "sneaked-in.html"

    before = _snapshot(proj)
    rc = main(["report", str(proj), "--out", str(sneaky), *NO_SESSIONS])

    assert rc != 0
    assert not (proj / "sneaked-in.html").exists()
    assert _snapshot(proj) == before


def test_a_legitimate_out_outside_every_run_project_still_writes(tmp_path):
    """The guard must refuse the violation and only the violation. A guard
    that refused everything would pass both tests above."""
    proj = _project_with_run(tmp_path)
    out = tmp_path / "reports" / "r.html"

    assert main(["report", str(proj), "--out", str(out), *NO_SESSIONS]) == 0
    assert out.exists()
    assert "agentview report" in out.read_text()


def test_write_report_itself_raises_rather_than_relying_on_the_cli(tmp_path):
    """The guarantee belongs to the writer. Any future caller of
    `write_report` -- not just `cli.main` -- must hit it, so the refusal is
    pinned at the function that opens the file."""
    proj = _project_with_run(tmp_path)
    run = build_run(find_runs(proj)[0], sessions=[])
    out = proj / "report.html"

    with pytest.raises(ReadOnlyViolation):
        write_report([run], out)
    assert not out.exists()


def test_the_guard_covers_every_run_in_the_report_not_only_the_first(tmp_path):
    """`report` takes several roots. Checking only `runs[0]`'s project would
    pass every test above while still writing into the second project."""
    first = _project_with_run(tmp_path / "one")
    second = _project_with_run(tmp_path / "two")
    before = _snapshot(second)

    out = second / "_planrunner" / "state" / "r.html"
    rc = main(["report", str(first), str(second), "--out", str(out),
               *NO_SESSIONS])

    assert rc != 0
    assert not out.exists()
    assert _snapshot(second) == before


# --- Step 18.2: symlink, relative-path, and whole-tree snapshot coverage ----
#
# The dotdot case (`test_out_reaching_a_run_project_via_dotdot_is_refused`
# above) already existed before this step. What was actually missing per the
# brief's three numbered requirements is: a symlink whose target resolves
# inside a discovered run directory; a bare relative `--out` path; and a
# filesystem snapshot test that checks a whole realistic tree, not one named
# path.


def test_out_via_a_symlink_resolving_inside_a_run_project_is_refused(
        tmp_path):
    """A symlink is another way to reach the same forbidden target while the
    unresolved string looks harmless. `link` here names a path entirely
    outside every run project; only once the filesystem is consulted does it
    turn out to point straight back into one. A guard that resolved `..` but
    stopped short of following symlinks would miss exactly this."""
    proj = _project_with_run(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    link = outside / "proj_link"
    link.symlink_to(proj)

    before = _snapshot(proj)
    out = link / "sneaked-via-symlink.html"

    rc = main(["report", str(proj), "--out", str(out), *NO_SESSIONS])

    assert rc != 0
    assert not (proj / "sneaked-via-symlink.html").exists()
    assert _snapshot(proj) == before


def test_out_as_a_relative_path_reaching_a_run_project_is_refused(
        tmp_path, monkeypatch):
    """`--out` need not be absolute at all. A guard that only ever resolved
    (or only ever compared) absolute strings could miss a bare relative path
    whose resolution -- relative to the process's own cwd -- lands inside a
    run project."""
    proj = _project_with_run(tmp_path)
    monkeypatch.chdir(tmp_path)
    before = _snapshot(proj)

    rc = main(["report", str(proj), "--out", "proj/relative-in.html",
               *NO_SESSIONS])

    assert rc != 0
    assert not (proj / "relative-in.html").exists()
    assert _snapshot(proj) == before


def _full_snapshot(root):
    """Existence, mtime/size, and content hash of every path under `root`,
    files *and* directories alike -- not just the files `_snapshot` tracks.
    A directory created or removed with nothing written inside it would be
    invisible to a files-only snapshot; this one also catches that. The
    sha256 of each file's bytes is included alongside mtime_ns/size so a
    same-size content change with a restored mtime is still caught.
    Permission bits (mode) are recorded for every entry -- a chmod changes
    nothing else in the tuple -- and a symlink's own target path (via
    os.readlink) is recorded too, so re-pointing a symlink is caught even
    when the link's own mtime/size/mode are unchanged."""
    out = {}
    for p in sorted(root.rglob("*")):
        mode = p.lstat().st_mode
        if p.is_symlink():
            target = os.readlink(p)
            try:
                digest = hashlib.sha256(p.read_bytes()).hexdigest()
            except OSError:
                digest = None
            out[str(p)] = (
                "symlink", p.lstat().st_mtime_ns, p.lstat().st_size, digest,
                mode, target)
        elif p.is_file():
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            out[str(p)] = (
                "file", p.lstat().st_mtime_ns, p.lstat().st_size, digest,
                mode, None)
        else:
            out[str(p)] = ("dir", None, None, None, mode, None)
    return out


def test_filesystem_snapshot_of_a_realistic_tree_is_byte_for_byte_unchanged(
        tmp_path):
    """Every test above names one path and checks that one path. This one
    builds a scratch tree shaped like a real machine -- a discovered run
    directory plus a sibling directory that merely looks like a home
    directory, holding files no run of agentview should ever touch -- snapshots
    every path in the whole tree, drives `report` and `watch` against it with
    a legitimate `--out` outside the tree, and asserts nothing in the tree
    moved at all: not one path added, removed, or changed in mtime/size,
    anywhere -- not only at the one path each narrower test above names. This
    is what actually establishes "agentview never creates, modifies, or deletes
    a file outside --out" as a tested property of the whole tree, rather than
    an assertion about one literal path."""
    scratch = tmp_path / "scratch"

    proj = scratch / "decoy-project"
    sd = proj / "_planrunner" / "state"
    sd.mkdir(parents=True)
    (sd / "ledger.json").write_text(json.dumps({"steps": [
        {"id": "A", "status": "complete", "owner": "b", "tier": "sonnet",
         "attempts": 1, "deliverable": "a.md", "depends_on": [],
         "updated": "2026-06-25T11:00:00Z"}]}))
    (proj / "BUILD-LOG.md").write_text(
        "25/06/2026 11:00 | A | complete | ok\n")

    # A decoy "home-like" directory: a sibling of the discovered run, not a
    # run itself, holding files that merely look sensitive. Nothing here
    # should move either.
    home_like = scratch / "home-like"
    (home_like / ".ssh").mkdir(parents=True)
    (home_like / ".ssh" / "known_hosts").write_text("not a real key\n")
    (home_like / ".bash_history").write_text("ls\ncd ..\n")
    (home_like / "Documents").mkdir(parents=True)
    (home_like / "Documents" / "notes.txt").write_text("unrelated notes\n")

    before = _full_snapshot(scratch)

    out = tmp_path / "outside-the-tree" / "report.html"
    rc = main(["report", str(scratch), "--out", str(out), *NO_SESSIONS])
    assert rc == 0
    assert out.exists()

    main(["watch", str(scratch), "--iterations", "1", "--interval", "0",
          *NO_SESSIONS])

    assert _full_snapshot(scratch) == before
