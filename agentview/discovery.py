# Copyright 2026 Edvard Pohjavirta
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for the full text.

"""Locate runs on disk. A project may hold several; never merge them."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from .events import parse_iso
from .layouts import BY_MARKER, RunnerLayout, slug_for, state_dirs
from .paths import session_dir
from .sources.ledger import run_span

PRUNE = {".git", "node_modules", ".venv", "venv", "__pycache__",
         ".mypy_cache", ".pytest_cache", "site-packages", "dist", "build"}


@dataclass
class RunRef:
    project: Path
    state_dir: Path
    ledger: Path
    build_log: Path | None
    gpt_dir: Path | None
    slug: str
    layout: RunnerLayout


def _build_log_for(project: Path, slug: str, lay: RunnerLayout) -> Path | None:
    candidates = []
    if lay.build_log == "suffixed" and slug != "main":
        candidates.append(project / f"BUILD-LOG-{slug}.md")
    candidates.append(project / "BUILD-LOG.md")
    return next((c for c in candidates if c.exists()), None)


def find_runs(root: Path) -> list[RunRef]:
    runs: list[RunRef] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in PRUNE]
        p = Path(dirpath)
        lay = BY_MARKER.get(p.name)
        if lay is None:
            continue
        project = p.parent

        batch = [(sd, sd / "ledger.json", slug_for(sd, lay))
                 for sd in state_dirs(p, lay)]
        batch = [b for b in batch if b[1].exists()]

        # Two differently-named state dirs can map to one slug. Collapsing them
        # would merge distinct runs, which this module must never do.
        counts: dict[str, int] = {}
        for _, _, slug in batch:
            counts[slug] = counts.get(slug, 0) + 1

        for sd, ledger, slug in batch:
            if counts[slug] > 1:
                slug = sd.name
            gpt = sd / lay.gpt_subpath if lay.gpt_subpath else None
            runs.append(RunRef(
                project=project, state_dir=sd, ledger=ledger,
                build_log=_build_log_for(project, slug, lay),
                gpt_dir=gpt if gpt and gpt.is_dir() else None,
                slug=slug, layout=lay))
    return runs


def find_sessions(cwd: str, projects_root: Path) -> list[Path]:
    d = session_dir(cwd, projects_root)
    return sorted(d.glob("*.jsonl")) if d.is_dir() else []


# --- Run-scoped session attribution -----------------------------------------
#
# `find_sessions` maps a project directory to the one transcript directory
# whose name encodes that path. That is the wrong unit for a *run*: a project
# accumulates dozens of unrelated sessions, one session can contain several
# runs back to back, and a run is frequently driven from a session whose cwd
# is not the project at all. Scoping a run's sessions therefore needs two
# independent signals, and both are required:
#
#   reference -- the run's state dir is named in the session's tool inputs;
#   overlap   -- the session's turns straddle the run's ledger span.
#
# Reference alone over-selects badly: later sessions that merely discuss a
# state dir (analysis, review, this very fix) quote the same path. Overlap
# alone cannot separate two runs driven from different projects at the same
# time. Together they pick out the sessions that actually drove the run.

# Characters that can continue a filesystem name. `proj/state` must not be
# reported as a mention of `proj/state-soak`.
_NAME_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")


@dataclass
class SessionScan:
    """One transcript read once: its turn span and which runs it names.

    `rel_cwd` holds every `cwd` seen on an assistant record where a
    relative-only key was actually matched -- one transcript can span several
    runner invocations from different project directories (see the module
    docstring), so neither a single session-wide cwd nor "the last one seen"
    can judge every relative hit: the same shared relative key
    (`_planrunner/state`) can be matched twice in one session, once while the
    session's cwd is inside project A and later once while it is inside
    project B, and both attributions must survive -- a dict of one value per
    key would let the later write silently clobber the earlier one. A key
    absent from `rel_cwd`, or present with an empty list, was never matched
    relative-only with a usable cwd (missing, or not a string -- see
    `_scan_session`)."""
    path: Path
    ts_start: datetime | None
    ts_end: datetime | None
    references: frozenset[str]
    rel_references: frozenset[str]
    rel_cwd: dict[str, list[str]]


def _mentions(blob: str, needle: str) -> bool:
    """True when `needle` names a path in `blob` rather than merely prefixing
    a longer name. A trailing `/`, quote or end-of-string ends the name."""
    i = blob.find(needle)
    while i != -1:
        end = i + len(needle)
        if blob[end:end + 1] not in _NAME_CHARS:
            return True
        i = blob.find(needle, i + 1)
    return False


def _needles(ref: RunRef) -> tuple[list[str], list[str]]:
    """The path forms a real invocation may use for this run's state dir,
    as `(absolute_forms, relative_forms)`.

    Both the absolute path and the path relative to the project root: the
    runner is driven through shell variables (`SD=...; "$L" --state-dir
    "$SD"`), so only the path string itself is reliable -- never a
    `ledger.py --state-dir` adjacency -- and either form may be the one that
    was assigned. They are returned separately because the relative form is
    byte-identical across every project sharing the same state-dir name
    (e.g. `_planrunner/state`) and therefore cannot stand alone as a match --
    callers must additionally confirm the session's own `cwd` agrees with the
    specific project being checked.
    """
    absolute = [str(ref.state_dir)]
    relative: list[str] = []
    try:
        relative.append(str(ref.state_dir.relative_to(ref.project)))
    except ValueError:
        pass
    return absolute, relative


def _scan_session(path: Path,
                  needles: dict[str, tuple[list[str], list[str]]]
                  ) -> SessionScan:
    """Read one transcript once, checking each `key`'s absolute forms first
    and its relative forms only as a fallback.

    `references` is every key any form of which was mentioned (absolute or
    relative combined) -- unchanged from before the cwd guard existed, so a
    caller that only wants "was this run named at all" keeps working exactly
    as it did. `rel_references` is the subset that was named *only* via its
    relative form -- never confirmed by an absolute mention -- which is the
    set a caller must additionally gate on the session's own `cwd` before
    trusting, since the relative form alone cannot tell projects apart.
    """
    ts_start: datetime | None = None
    ts_end: datetime | None = None
    hits: set[str] = set()
    rel_hits: set[str] = set()
    rel_cwd: dict[str, list[str]] = {}
    pending = {k: v for k, v in needles.items()}
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            if rec.get("type") != "assistant":
                continue
            msg = rec.get("message")
            msg = msg if isinstance(msg, dict) else {}
            if not (msg.get("usage") or {}):
                continue  # matches read_turns: only usage-bearing turns count
            # A malformed transcript can carry a non-string `cwd` (a number,
            # list or object); treat anything but a real string as
            # unavailable rather than storing it, so the guard below --
            # which calls `.startswith` on it -- can never see one.
            raw_cwd = rec.get("cwd")
            record_cwd = raw_cwd if isinstance(raw_cwd, str) else None
            ts = parse_iso(rec.get("timestamp"))
            if ts is not None:
                ts_start = ts if ts_start is None or ts < ts_start else ts_start
                ts_end = ts if ts_end is None or ts > ts_end else ts_end
            if not pending:
                continue
            for c in msg.get("content") or []:
                if not (isinstance(c, dict) and c.get("type") == "tool_use"):
                    continue
                blob = json.dumps(c.get("input"), default=str)
                for key in list(pending):
                    absolute, relative = pending[key]
                    if any(_mentions(blob, n) for n in absolute):
                        hits.add(key)
                        rel_hits.discard(key)
                        # An absolute hit means this key is no longer
                        # relative-only for this scan; drop the whole
                        # accumulated list, not one entry from it.
                        rel_cwd.pop(key, None)
                        del pending[key]
                    elif any(_mentions(blob, n) for n in relative):
                        # Judged against *this* record's cwd, not whichever
                        # record's cwd happened to be captured first -- a
                        # later match in a session that has moved to a
                        # different project must not inherit an earlier,
                        # unrelated invocation's directory. Appended, not
                        # overwritten: the same shared relative key can be
                        # matched from more than one project's cwd within a
                        # single session, and an earlier valid cwd must
                        # survive a later one rather than being clobbered by
                        # it. Only a real string cwd is ever appended (a
                        # malformed, non-string cwd was already normalised to
                        # `None` above).
                        rel_hits.add(key)
                        if record_cwd is not None:
                            rel_cwd.setdefault(key, []).append(record_cwd)
    return SessionScan(path, ts_start, ts_end, frozenset(hits | rel_hits),
                       frozenset(rel_hits), rel_cwd)


def _overlaps(scan: SessionScan, start: datetime | None,
              end: datetime | None) -> bool:
    """Does the session's turn span meet the run's `[start, end]`?

    Either bound may be `None`, and a missing bound is *open*, not fatal.
    Two callers need that. A live run has no upper bound at all, so a session
    started after the ledger's last write is still the session driving it
    right now. And a ledger whose steps carry no parseable `updated` yields
    no upper bound either; refusing every session there returned zero
    sessions and a run that rendered completely empty. An open bound
    constrains nothing; the reference signal still has to hold.
    """
    if scan.ts_start is None or scan.ts_end is None:
        return False
    if start is not None and scan.ts_end < start:
        return False
    if end is not None and scan.ts_start > end:
        return False
    return True


def sessions_for_runs(refs: Sequence[RunRef], projects_root: Path,
                      live: bool = False) -> dict[Path, list[Path]]:
    """Sessions that drove each run, keyed by the run's state dir.

    Every transcript under `projects_root` is read exactly once, whatever the
    number of runs, so adding runs costs no extra I/O.

    `live=True` drops the upper bound of every run's span, so a session
    started while the pane is watching is picked up instead of being filtered
    out for beginning after the ledger's last write.
    """
    out: dict[Path, list[Path]] = {ref.state_dir: [] for ref in refs}
    if not refs or not projects_root.is_dir():
        return out

    # Keyed by the literal needle text, not by ref identity: the absolute
    # form is unique per ref, but the relative form (`_planrunner/state`) is
    # byte-identical for every project holding it, so `by_key` must map that
    # one key to every ref it could name -- collapsing it to a single ref
    # here would silently drop all but whichever ref happened to be written
    # last, before the cwd guard below ever got a chance to run.
    by_key: dict[str, list[RunRef]] = {}
    needles: dict[str, tuple[list[str], list[str]]] = {}
    spans = {str(ref.state_dir): run_span(ref.ledger, live=live) for ref in refs}
    for ref in refs:
        absolute, relative = _needles(ref)
        for form in absolute:
            by_key.setdefault(form, []).append(ref)
            needles.setdefault(form, ([], []))[0].append(form)
        for form in relative:
            by_key.setdefault(form, []).append(ref)
            needles.setdefault(form, ([], []))[1].append(form)

    for scan in _scan_all(projects_root, needles):
        # An absolute reference substring-contains its own relative form
        # (`/proj/a/_planrunner/state` names `_planrunner/state` too), so
        # `scan.references` can carry both the absolute key and the shared
        # relative key for the very same ref. A scan may therefore appear
        # under more than one key for one ref -- `appended` makes sure it
        # still contributes at most one session to that ref's list, no
        # matter how many of its needle-forms matched.
        appended: set[Path] = set()
        for key in sorted(scan.references):
            is_relative_only = key in scan.rel_references
            for ref in by_key.get(key, []):
                if ref.state_dir in appended:
                    continue
                if is_relative_only:
                    # Relative-only match: require that AT LEAST ONE of the
                    # cwds recorded for THIS key be inside THIS candidate's
                    # project -- not any other record's cwd from the same
                    # session, and not only the most recent one. `_planrunner/
                    # state` names every project that holds it, and one
                    # session can match it once per project it was driven
                    # from at different times; each candidate is checked in
                    # turn against the full set of recorded cwds, so an
                    # earlier attribution can never be lost to a later one.
                    proj = str(ref.project)
                    cwds = scan.rel_cwd.get(key, [])
                    if not any(cwd == proj or cwd.startswith(proj + os.sep)
                              for cwd in cwds):
                        continue
                start, end = spans[str(ref.state_dir)]
                if _overlaps(scan, start, end):
                    out[ref.state_dir].append(scan.path)
                    appended.add(ref.state_dir)
    return out


def _scan_all(projects_root: Path,
              needles: dict[str, list[str]]) -> Iterable[SessionScan]:
    for proj in sorted(projects_root.iterdir()):
        if not proj.is_dir():
            continue
        for sess in sorted(proj.glob("*.jsonl")):
            yield _scan_session(sess, needles)


def run_sessions(ref: RunRef, projects_root: Path,
                 live: bool = False) -> list[Path]:
    """The sessions that drove a single run. See `sessions_for_runs`."""
    return sessions_for_runs([ref], projects_root, live=live)[ref.state_dir]
