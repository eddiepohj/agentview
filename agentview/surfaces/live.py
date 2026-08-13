"""Live terminal pane. Plain ANSI full-redraw; render_frame is pure."""
from __future__ import annotations

import sys
import textwrap
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, TextIO

from ..discovery import RunRef, run_sessions
from ..events import ACCURACY_REASONS, describe_anomaly
from ..metrics import (fable_count, gate_ratio, self_report_drift,
                       tokens_by_model_family, vocab_drift)
from ..model import build_run, run_span_overlap_events
from . import term
from .viewstate import ViewState, reduce

# Home, erase the screen, then drop the scrollback (\x1b[3J) — the same
# sequence modern `clear` emits. Without the 3J, every redraw pushes a
# frame into scrollback, the buffer grows each tick, and the reader ends
# up above the current frame and has to scroll down to it.
CLEAR = "\x1b[H\x1b[2J\x1b[3J"
_DONE = {"complete", "complete-retry"}

# Sentinel for events carrying no parseable `ts` (`parse_iso` returns None),
# so `_md_state`'s sort key never compares `None` against a real `datetime`.
_MIN_TS = datetime.min.replace(tzinfo=timezone.utc)

_MODEL_W = 30

HELP_KEYS = ("j", "k", "d", "?", "\r", "q")
_HELP = (" keys   j/k select step · enter expand · d documents · ? help · "
         "q quit")


def _clip(s: str, width: int) -> str:
    return s if len(s) <= width else s[: max(0, width - 1)] + "…"


def _wrap(s: str, width: int, indent: str = "   ") -> list[str]:
    """Wrap one logical line, indenting continuations.

    v1 clipped, so a warning's count survived and the clause explaining it did
    not -- observed four times, twice forcing a test to assert on a truncated
    phrase. For a tool whose bar is never presenting something as clean when it
    is not, a warning that loses its explanation is not cosmetic.
    """
    out = textwrap.wrap(s, width=width, subsequent_indent=indent,
                        break_long_words=False, break_on_hyphens=False)
    return [_clip(l, width) for l in out] or [""]


def _middle_clip(s: str, width: int) -> str:
    """Clip from the middle, keeping the tail -- for paths, where the filename
    is what the reader wants and the head is boilerplate."""
    if len(s) <= width:
        return s
    keep_tail = max(0, width - 1 - (width // 3))
    head = width - 1 - keep_tail
    return s[:head] + "…" + s[len(s) - keep_tail:]


def _local_summary(run: Any) -> dict[str, Any]:
    """The subset of metrics.summary() that render_frame actually reads.

    Deliberately does not call metrics.summary(run) -- that also runs
    waiver_audit(run.ref.project), which touches disk (rglob + read_text).
    render_frame must stay a pure function of the in-memory Run, so it
    calls only the metric functions with no filesystem access."""
    return {"tokens_by_model_family": tokens_by_model_family(run),
            "gates": gate_ratio(run),
            "fable": fable_count(run),
            "vocab_drift": vocab_drift(run)}


def _md_state(run: Any) -> str:
    """What the MD tier is doing, read from the events it actually emits.

    v1 rendered a static test of whether a directory exists, so a layout that
    wrote no rounds was frozen on "no rounds recorded" -- indistinguishable
    from a tier that ran and found nothing.
    """
    rounds = [e for e in run.events if e.kind in ("md.review", "md.escalate")]
    if rounds:
        # Latest by `Event.ts`, never by round number alone -- rounds are
        # scoped to a gate/step, so a later gate starting at round 1 would
        # otherwise lose to an older gate at round 3. `(round, gate)` is
        # only a tiebreak for events sharing one timestamp (e.g. a
        # md.review/md.escalate pair emitted from the same round file).
        # `gptgates.py` emits that pair with an identical (ts, round, gate)
        # when a round escalates, and always appends `md.review` first --
        # `max()` keeps the first maximal element on a tie, so without a
        # fourth, kind-priority element `md.review` would always win and
        # `latest.kind == "md.escalate"` below could never fire. `True >
        # False` ranks `md.escalate` above `md.review` on an exact tie,
        # independent of which of the pair happened to be appended first.
        latest = max(rounds, key=lambda e: (e.ts or _MIN_TS,
                                            e.payload.get("round") or 0,
                                            e.payload.get("gate") or "",
                                            e.kind == "md.escalate"))
        p = latest.payload
        line = f"● gate {p.get('gate', '?')}, round {p.get('round', '?')}"
        if p.get("rounds_total"):
            line += f"/{p['rounds_total']}"
        blocking = p.get("blocking") or []
        if blocking:
            # `gptgates.py` (Step 9) emits `blocking` as bare risk-ID
            # strings, not severity-tagged dicts -- only the test fixture's
            # shape carries a "severity" key. Real risk data lives in
            # `p["risks"]`, keyed by id; fall back to "unknown" for an id
            # that fails to resolve rather than crash on `str.get`.
            by_id = {r.get("id"): r.get("severity", "unknown")
                     for r in (p.get("risks") or []) if isinstance(r, dict)}
            worst = Counter(
                b.get("severity", "unknown") if isinstance(b, dict)
                else by_id.get(b, "unknown")
                for b in blocking)
            line += "   blocking: " + ", ".join(
                f"{n} {sev}" for sev, n in sorted(worst.items()))
        if latest.kind == "md.escalate":
            line += "   escalated"
        return line
    if run.ref.gpt_dir is None:
        # A reviewer pin is itself evidence the MD tier ran (Step 13's
        # `_model_cell`: "a pinned reviewer model is itself an observation --
        # gate.py wrote the pin as a byproduct of actually running the gate,
        # not a guess"). A run carrying one is not channel-less just because
        # this particular run recorded no round events and no gate directory.
        if getattr(run, "reviewer_pin", None):
            return "idle    no rounds recorded"
        return "unobservable — no rounds and no gate directory"
    return "idle    no rounds recorded"


def _short_model(model: str) -> str:
    """Drop the vendor prefix in this column only. Two full worker IDs overflow
    25 columns and middle-clip into one unreadable hybrid
    (`claude-h…claude-sonnet-4-6`). Full IDs stay in the expanded step view."""
    return model[len("claude-"):] if model.startswith("claude-") else model


def _model_cell(run: Any, tier: Any) -> str:
    """One model cell, marked with how agentview knows it.

    Four grades in falling confidence, applied uniformly to every tier
    including `md`: observed renders bare (for `md`, a pinned reviewer model
    is itself an observation -- gate.py wrote the pin as a byproduct of
    actually running the gate, not a guess); inferred renders `· inferred`
    (the model is real, the role match is a naming convention the skill never
    guaranteed); declared renders `· declared` (a contract, not a
    measurement); otherwise `unknown`. Provenance is tracked **per
    contributing model**, not per cell — a role with one observed and one
    inferred agent must not blanket-mark an observed model as inferred, and an
    observed sighting of a model always wins over an inferred one for that
    same model.
    """
    if tier.role == "human":
        return "—"          # a human runs no model; nothing is unknown here
    if tier.role == "step-runner":
        # The step-runner is the orchestrating session itself, never a
        # dispatched agent -- `assign_agent_roles` can never produce a
        # `run.agents` entry with role "step-runner" (the skill dispatches
        # only the eight worker categories). Its models live in
        # `run.orchestrator_models`, read directly off the transcript, so
        # they are an observation, exactly like an agent's own `a.model`.
        models = sorted(set(getattr(run, "orchestrator_models", None) or []))
        if models:
            return ", ".join(models)
    elif tier.role == "md":
        pin = getattr(run, "reviewer_pin", None)
        if pin:
            effort = f" · {pin['effort']}" if pin.get("effort") else ""
            return f"{pin['model']}{effort}"          # observed: gate.py's own pin
    else:
        displayed = {t.role for t in run.ref.layout.tiers}
        by_model: dict[str, str] = {}
        for a in run.agents:
            role = getattr(a, "role", "worker")
            role = role if role in displayed else "worker"
            if role != tier.role or not a.model:
                continue
            src = getattr(a, "role_source", "observed")
            if a.model not in by_model or (by_model[a.model] == "inferred"
                                           and src == "observed"):
                by_model[a.model] = src
        if by_model:
            return ", ".join(
                _short_model(m) + (" · inferred" if src == "inferred" else "")
                for m, src in sorted(by_model.items()))
    if tier.declared_model:
        return f"{tier.declared_model} · declared"
    return "unknown"


def _role_states(run: Any, s: dict[str, Any]) -> dict[str, str]:
    """One state cell per role.

    `unobservable` is the fallback, not a per-role label. Each of the three
    tiers the design first applied it to turned out to leave traces: fable
    writes the ledger's rulings, the director dispatches like any agent, and
    the MD's whole exchange sits in the caller's transcript.
    """
    gates, active = s["gates"], [st for st in run.steps
                                 if st.status in ("in-progress", "recon")]
    by_role: dict[str, list[Any]] = {}
    for a in run.agents:
        by_role.setdefault(getattr(a, "role", "worker"), []).append(a)

    # A role resolves to one of the skill's eight categories even when the
    # layout displays no dedicated row for it (e.g. `skeptic`, `verifier` on
    # max-runner, which declares rows only for human/fable/md/director/
    # step-runner/worker) -- such agents still count as workers for display,
    # while `AgentRun.role` keeps recording the real category.
    displayed = {t.role for t in run.ref.layout.tiers}
    worker_count = sum(len(agents) for role, agents in by_role.items()
                       if role == "worker" or role not in displayed)

    opened = gates["opened"]
    human_state = (f"idle    gates: {gates['by_human']} answered of {opened}"
                   if opened else "idle    no gates opened")
    states = {
        "human": human_state,
        "step-runner": ("● " + active[0].id if active else "idle")
                       + f"    output "
                         f"{s['tokens_by_model_family'].get('orchestrator', 0):,}",
        "worker": f"{worker_count} dispatched",
        "md": _md_state(run),
    }
    rulings = s["fable"]["rulings"]
    if rulings:
        noun = "ruling" if len(rulings) == 1 else "rulings"
        last = rulings[-1]
        states["fable"] = (f"{len(rulings)} {noun}: {last.get('step', '?')} "
                           f"{last.get('ruling', '')}").rstrip()
    directors = by_role.get("director", [])
    if directors:
        noun = "dispatch" if len(directors) == 1 else "dispatches"
        sources = {getattr(a, "role_source", "observed") for a in directors}
        # Unconditional "inferred" was itself a false-provenance bug (R8): a
        # future run with ledger/transcript-observed directors must not be
        # branded inferred just because today's corpus always is.
        suffix = (" · inferred" if sources == {"inferred"} else
                 "" if sources == {"observed"} else " · mixed")
        states["director"] = f"{len(directors)} {noun}{suffix}"

    # A dedicated role dispatched with no richer state computed above (e.g.
    # a `fable` agent with no recorded ruling yet) still has real dispatch
    # evidence in `by_role` -- that must beat `unobservable`, the same way
    # `worker_count` already beats it for the worker row.
    for role, agents in by_role.items():
        key = role if role in displayed else "worker"
        states.setdefault(key, f"{len(agents)} dispatched")

    for tier in run.ref.layout.tiers:
        states.setdefault(tier.role, "unobservable — no channel for this tier")
    return states


def _step_detail(run: Any, st: Any, width: int) -> list[str]:
    """The selected step, expanded. Reads only the in-memory Run.

    Attempts and the derived token figure each get the same treatment
    `self_report_drift` (metrics.py) already gives the ledger-vs-derived
    comparison: an absent value renders as its own word, never folded into a
    default that reads identically to a real, distinct value. `st.attempts
    or 1` and an unconditional `derived {st.output_tokens}` both did exactly
    that folding -- an unrecorded attempt count printed as a real single
    attempt, and a step with no attributed turns printed as a measured zero.
    Checking `is None` (attempts) and `len(st.turns) == 0` (turns) rather
    than truthiness also means a real, explicit `0` attempts value -- if the
    ledger ever records one -- prints as `0`, not `1`.
    """
    claimed = getattr(st, "ledger_tokens", None)
    no_attribution = len(st.turns) == 0
    derived = "no attribution" if no_attribution else f"{st.output_tokens:,}"
    tokens = (f"ledger {claimed:,}   derived {derived}"
              if claimed is not None
              else f"ledger unrecorded   derived {derived}")
    attempts = "unrecorded" if st.attempts is None else str(st.attempts)
    out = [_clip(f"    attempts {attempts}   turns {len(st.turns)}   "
                 f"{tokens}", width)]
    for a in st.agents or []:
        out.append(_middle_clip(
            f"    ↳ {a.role}  {a.model or 'unknown'}  "
            f"{a.output_tokens:,} out", width))
    if not st.agents:
        out.append(_clip("    no agents attributed to this step", width))
    return out


def _replay_snapshot(run: Any, stage: int) -> Any:
    """A pure view of the first ``stage`` completed ledger steps."""
    visible_steps = list(run.steps[:max(0, stage)])
    visible_ids = {st.id for st in visible_steps}
    agents = []
    seen = set()
    for st in visible_steps:
        for agent in st.agents:
            if id(agent) not in seen:
                agents.append(agent)
                seen.add(id(agent))
    events = [e for e in run.events if e.step is None or e.step in visible_ids]
    rulings = [r for r in run.fable_rulings
               if r.get("step") in visible_ids or r.get("step") is None]
    return replace(run, steps=visible_steps, agents=agents, events=events,
                   anomalies=[e for e in run.anomalies
                              if e.step is None or e.step in visible_ids],
                   orchestrator_output_tokens=sum(st.output_tokens
                                                  for st in visible_steps),
                   fable_rulings=rulings)


def render_frame(run: Any, view: Any = None, now_turn: int | None = None,
                 width: int = 80) -> str:
    if now_turn is not None:
        run = _replay_snapshot(run, now_turn)
    view = view if view is not None else ViewState()
    s = _local_summary(run)
    done = sum(1 for st in run.steps if st.status in _DONE)
    rule = "─" * width
    lines = [
        _clip(f"{run.ref.layout.display}  {run.ref.project.name}"
              f"  run: {run.ref.slug}"
              f"    step {done}/{len(run.steps)}", width),
        rule]

    warnings = [describe_anomaly(a) for a in run.anomalies
                if a.payload.get("reason") in ACCURACY_REASONS]
    for w in warnings:
        lines.extend(_wrap(f" ! {w}", width))
    if warnings:
        lines.append(rule)

    lines.append(_clip(f" {'role':<13} {'model':<{_MODEL_W}} state", width))
    states = _role_states(run, s)
    for tier in run.ref.layout.tiers:
        model = _middle_clip(_model_cell(run, tier), _MODEL_W)
        lines.append(_clip(f" {tier.label:<13} {model:<{_MODEL_W}} "
                           f"{states.get(tier.role, 'idle')}", width))

    lines += [rule, _clip(" ledger", width)]
    # A locally clamped selection, derived here rather than trusted from
    # `view.selected` directly -- `ViewState` persists across redraws, but a
    # live-watched run is rebuilt fresh each redraw and can shrink or reorder
    # its step list between frames. Deriving keeps `reduce` pure while making
    # every render safe against a stale index.
    selected = min(view.selected, max(0, len(run.steps) - 1))
    for i, st in enumerate(run.steps):
        mark = "●" if st.status in ("in-progress", "recon") else " "
        sel = ">" if i == selected else " "
        attempts = f" x{st.attempts}" if (st.attempts or 0) > 1 else ""
        wake = ""
        for e in run.events:
            if e.step == st.id and e.kind == "step.suspend" and e.payload.get("wake"):
                wake = f"  wake: {e.payload['wake']}"
                break
        lines.append(_clip(f" {st.id:<8} {st.status or '?':<14}{attempts}"
                           f"{sel}{mark}{wake}", width))
        if view.expanded and i == selected:
            lines.extend(_step_detail(run, st, width))

    # Only entries that resolve to a real file are listed under "documents".
    # Closed keeps v1's eight-plus-count behaviour; open (`view.tray_open`)
    # lists every document and drops the "and N more" line, since there is
    # nothing left uncounted.
    if run.doc_paths:
        lines.append(rule)
        lines.append(_clip(" documents", width))
        shown = run.doc_paths if view.tray_open else run.doc_paths[:8]
        for d in shown:
            lines.append(f"   {_middle_clip(d, width - 3)}")
        if not view.tray_open and len(run.doc_paths) > 8:
            lines.append(_clip(f"   … and {len(run.doc_paths) - 8} more",
                               width))

    # Everything ledger hygiene is not the run's work, only how the runner
    # kept its books: entries that are prose rather than a resolvable path
    # (reported as a count -- never discarded, the report lists them in
    # full), and the ledger's self-reported token spend beside what the
    # transcript derives. The section is silent when both signals are quiet.
    drift = self_report_drift(run)
    if run.doc_notes or drift["divergent"] or drift["unrecorded"]:
        lines.append(rule)
        lines.append(_clip(" ledger hygiene", width))
    if run.doc_notes:
        n = len(run.doc_notes)
        noun = "deliverable is" if n == 1 else "deliverables are"
        lines.extend(_wrap(
            f"   {n} ledger {noun} prose, not paths — full text is in the "
            "report. This describes how the runner writes its ledger, not "
            "the run's work.", width))
    if drift["divergent"] or drift["unrecorded"]:
        d, u = drift["divergent"], drift["unrecorded"]
        noun = "step's token figure diverges" if d == 1 else \
               "steps' token figures diverge"
        lines.extend(_wrap(
            f"   {d} {noun} from the derived total, {u} unrecorded — the "
            "ledger's own claim, shown beside what the transcript supports. "
            "Neither figure is adjusted.", width))

    tiers = ", ".join(f"{k} {v:,}" for k, v in sorted(s["tokens_by_model_family"].items()))
    lines += [rule, _clip(f" tokens  {tiers}", width),
              _clip(" tokens  provisional — this pane shows movement, not a "
                    "settled count", width)]
    if s["vocab_drift"]["count"]:
        lines.append(_clip(f" drift   {s['vocab_drift']['count']} "
                           f"off-vocabulary build-log statuses", width))
    # The claim is made only when it is true of everything the pane showed:
    # every entry resolved to a file (nothing was relegated to the note
    # count) *and* every one of them is genuinely absolute. It used to be
    # unconditional while most entries were prose and the resolvable ones
    # were project-relative. A pure string test -- no disk access, so
    # render_frame stays a pure function of the Run.
    exit_line = " ctrl-c to exit · ? help"
    if (run.doc_paths and not run.doc_notes
            and all(Path(d).is_absolute() for d in run.doc_paths)):
        exit_line += " · document paths above are absolute"
    lines.append(_clip(exit_line, width))
    if view.help_open:
        lines.append(_clip(_HELP, width))
    return "\n".join(lines)


def watch(ref: Any, sessions: list[Path] | None = None, interval: float = 1.0,
          iterations: int | None = None, stream: TextIO | None = None,
          projects_root: Path | None = None, pin_path: Path | None = None,
          siblings: Sequence[RunRef] = ()) -> None:
    """Redraw the live pane until `iterations` is reached, or forever.

    `stream` defaults to `sys.stdout` *at call time*, not at import time; a
    module-level default binds whatever object stdout happened to be when
    agentview was imported, which leaked frames past `redirect_stdout`.

    When `projects_root` is given the run's sessions are re-resolved on every
    redraw. Resolving once before the loop meant a session started while the
    pane was already open could never appear, which is precisely the case a
    live pane exists for.

    `term.raw_mode` wraps the *whole* loop, entered once before the first
    frame and exited once after the last, so its `finally` restores the
    terminal on every exit path -- a normal `break`, an exception, or a
    `KeyboardInterrupt` raised mid-loop. Off a TTY it yields `False` and the
    loop falls back to `time.sleep`, exactly as before Step 16 -- no raw mode
    attempted, no input read. On a TTY, `term.read_key` replaces the sleep as
    the loop's one pause point, and `q` (via `ViewState.quit`) breaks it.
    """
    stream = sys.stdout if stream is None else stream
    n = 0
    view = ViewState()
    with term.raw_mode() as interactive:
        while iterations is None or n < iterations:
            sess = (run_sessions(ref, projects_root, live=True)
                    if projects_root is not None else list(sessions or []))
            run = build_run(ref, sess, live=True, pin_path=pin_path)
            run.anomalies.extend(run_span_overlap_events(ref, siblings))
            stream.write(CLEAR + render_frame(run, view) + "\n")
            stream.flush()
            n += 1
            if iterations is not None and n >= iterations:
                break
            if interactive:
                key = term.read_key(timeout=interval)
                if key is not None:
                    view = reduce(view, key, run)
                    if view.quit:
                        break
            else:
                time.sleep(interval)


def replay(run: Any, stream: TextIO | None = None, step: int = 1) -> None:
    """Redraw the frame once per `step` steps, for testing the pane offline.

    `stream` resolves `sys.stdout` in the body for the same reason `watch`
    does; the import-time default is what let `replay` write past a caller's
    `redirect_stdout`."""
    stream = sys.stdout if stream is None else stream
    if step <= 0:
        raise ValueError("step must be positive")
    for i in range(0, len(run.steps) + 1, step):
        stream.write(CLEAR + render_frame(run, now_turn=i) + "\n")
        stream.flush()
