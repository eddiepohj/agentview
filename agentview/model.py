"""Assemble a Run from the event streams of every source."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

from .discovery import RunRef
from .events import Event, merge, parse_iso
from .sources import (buildlog, gatecalls, gatepin, gptgates,
                      ledger as ledger_src, transcript)

_MIN_TS = datetime.min.replace(tzinfo=timezone.utc)
_MAX_TS = datetime.max.replace(tzinfo=timezone.utc)

# How far after a run's span start the first transcript turn may sit before
# the coverage is called into question. Healthy runs in the real corpus start
# within 5-23 seconds; the run whose first 21 hours of transcript are missing
# from ~/.claude/projects starts 77,571 seconds late. Five minutes separates
# those two populations by two orders of magnitude either way.
_COVERAGE_GAP_SEC = 300.0

_ACTIVE = ("in-progress", "recon")


@dataclass
class Step:
    id: str
    status: str | None
    owner: str | None = None
    ledger_tier: str | None = None
    attempts: int | None = None
    deliverable: str | None = None
    depends_on: list[str] = field(default_factory=list)
    turns: list[int] = field(default_factory=list)
    output_tokens: int = 0
    agents: list[Any] = field(default_factory=list)
    gate_class: str | None = None
    answered_by: str | None = None
    ledger_tokens: int | float | None = None


@dataclass
class Run:
    ref: RunRef
    steps: list[Step]
    agents: list[Any]
    events: list[Event]
    docs: list[str]
    session_paths: list[Path]
    anomalies: list[Event]
    orchestrator_output_tokens: int = 0
    orchestrator_models: list[str] = field(default_factory=list)
    # `docs` is every ledger `deliverable`, verbatim, and stays that way. But
    # that field is free text: on the real corpus only 7/21, 4/10, 2/6 and
    # 0/6 entries name a file, the rest being prose like "E.4 skeptic verdict
    # (BUILD-LOG) + E.3 reply evidence". Presenting those as paths -- in a
    # <code> tag, under a heading called "documents", beside the claim that
    # they are absolute -- states something false in a surface whose whole
    # purpose is not to. The split is computed here, once, because the
    # surfaces must not touch the disk: `render_frame` is a pure function of
    # the in-memory Run and there are tests pinning that.
    doc_paths: list[str] = field(default_factory=list)   # names a real file
    doc_notes: list[str] = field(default_factory=list)   # prose, verbatim
    # The ledger's own `dispatches`/`sessions`/`fable_rulings` rows, carried
    # through verbatim (well-formed members only -- see `_sanitize` in
    # `sources/ledger.py`). `sessions` here is the ledger's dispatch-session
    # bookkeeping (`{"session", "closed", "waivers", "trigger"}`), not to be
    # confused with `session_paths` above, which is real transcript paths.
    sessions: list[dict[str, Any]] = field(default_factory=list)
    dispatches: list[dict[str, Any]] = field(default_factory=list)
    fable_rulings: list[dict[str, Any]] = field(default_factory=list)
    reviewer_pin: dict | None = None


def assign_turns_to_steps(windows: list[tuple[str, datetime]],
                          turns: list[transcript.Turn],
                          start: datetime | None = None) -> dict[str, list[int]]:
    """Bucket turns into the step window that closes after them.

    Each window is bounded below as well as above. For windows in ascending
    order -- which is what `step_windows` returns -- every window after the
    first is already bounded below by its predecessor's close, because the
    scan below stops at the *first* window that closes at or after the turn.
    The one bound that must be stated is the first window's, and that is
    `start`: the run's beginning. Without it the first step swallows every
    turn that ever preceded the run, which is how two runs sharing a session
    came to report identical first-step figures.

    Turns later than every window remain unassigned, as before; there is
    deliberately no nearest-neighbour fallback.
    """
    out: dict[str, list[int]] = {sid: [] for sid, _ in windows}
    if not windows:
        return out
    for t in turns:
        if t.ts is None:
            continue
        for i, (sid, end) in enumerate(windows):
            if t.ts > end:
                continue
            if i == 0 and start is not None and t.ts < start:
                break  # before the run began: this turn belongs to no step
            out[sid].append(t.index)
            break
    return out


def _within(ts: datetime | None, start: datetime | None,
            end: datetime | None) -> bool:
    """Is `ts` inside the run's span? An unknown span filters nothing."""
    if start is None and end is None:
        return True
    if ts is None:
        return False
    return ((start is None or ts >= start) and (end is None or ts <= end))


def _names_a_file(raw: str, project: Path) -> bool:
    """Does a ledger `deliverable` name a file that exists?

    Tried both as given -- a deliverable may already be absolute -- and
    relative to the run's project root, which is the form the real corpus
    uses (`_planrunner/E1-outbound.md`). Most entries answer False: the field
    is free text and holds prose at least as often as it holds a path.

    The entry is *classified*, never rewritten. Substituting the resolved
    absolute path would be more useful in principle, but the pane clips to 80
    columns, and a resolved path is long enough that the filename -- the part
    a reader is looking for -- falls off the end. Showing what the ledger
    actually says is the honest option and survives the clip.
    """
    for candidate in (Path(raw), project / raw):
        try:
            if candidate.is_file():
                return True
        except (OSError, ValueError):
            # Prose makes poor filenames: over-long components and embedded
            # NULs raise rather than answering False.
            continue
    return False


def _split_deliverables(docs: list[str],
                        project: Path) -> tuple[list[str], list[str]]:
    """Partition `docs` into entries naming a real file, and everything else.

    Nothing is discarded -- the prose entries are real ledger content that a
    reader may want. They are simply not paths, and are returned separately
    so no surface can present them as such.
    """
    paths = [d for d in docs if _names_a_file(d, project)]
    notes = [d for d in docs if d not in set(paths)]
    return paths, notes


def _renumber_within_run(turns: list[transcript.Turn]) -> None:
    """Make `Turn.index` unique across a run's sessions, in time order.

    `read_turns` numbers from 1 per session, which is right for its own
    contract. A run that suspends and resumes -- plan-runner's normal mode --
    spans two sessions, so two turns both call themselves 1. `build_run`
    then keyed `by_index` on that, and the later turn silently replaced the
    earlier one: a two-session step reported 6,000 output tokens against a
    true 3,300. Renumber the run's own local list; the sessions' numbering
    is untouched.
    """
    dated = sorted((t for t in turns if t.ts is not None), key=lambda t: t.ts)
    undated = [t for t in turns if t.ts is None]
    for i, t in enumerate(dated + undated, 1):
        t.index = i


def _live_windows(windows: list[tuple[str, datetime]],
                  raw_steps: list[dict[str, Any]]) -> list[tuple[str, datetime]]:
    """Give the in-flight step an open-ended window for live rendering.

    Every step's window closes at its own `updated`, so turns produced after
    the orchestrator last wrote state fall off the end of the list and the
    step that is running right now shows zero turns -- the pane going blind
    at exactly the moment it is supposed to show movement. Appending an
    unbounded window for the active step keeps `assign_turns_to_steps`
    unchanged: it already stops at the first window closing at or after a
    turn, and the duplicate step id collapses into the same bucket.
    """
    active = [s for s in raw_steps if s.get("status") in _ACTIVE]
    if not active:
        return windows
    latest = max(active, key=lambda s: parse_iso(s.get("updated")) or _MIN_TS)
    return windows + [(latest.get("id"), _MAX_TS)]


def assign_agent_roles(agents, layout) -> None:
    """Resolve each dispatched agent's role, best source first.

    1. the transcript's `agentType` -- the skill now requires one of the eight
       as `subagent_type` and forbids `general-purpose`
    2. the dispatch description's prefix -- the only signal for runs made
       before the taxonomy landed, where all 25 dispatches are general-purpose

    No ledger-first rank: `dispatches[].kind` names a step and a category, not
    a specific agent (Step 4.1 confirmed the real schema carries no per-agent
    identity), so it cannot be joined to one `AgentRun` and is not a
    resolution source here.

    (1) is an observation. (2) is a naming convention the skill never
    guaranteed, so it is marked `inferred` and must not read as a reading.
    Validated against the skill's closed taxonomy (`ROLES`), not against
    `layout.tiers` -- a category like `skeptic` is a real, observed role even
    on a layout whose display groups it under a shared `workers` row.
    """
    from .events import ROLES
    valid = set(ROLES) - {"human", "unknown"}
    prefixes = [(p, t.role) for t in layout.tiers for p in t.infer_prefixes]

    for a in agents:
        declared = getattr(a, "agent_type", None)
        if declared and declared != "general-purpose" and declared in valid:
            a.role, a.role_source = declared, "observed"
            continue
        desc = a.description or ""
        for prefix, role in prefixes:
            if desc.startswith(prefix):
                a.role, a.role_source = role, "inferred"
                break


def _merge_gate_events(disk_events, call_events):
    """One event per (step, gate, round) -- a run using --emit-dir carries the
    same round in both the on-disk file and the caller's transcript, and
    concatenating both streams would double-count every md.* metric."""
    def key(e):
        return (e.step, e.payload.get("gate"), e.payload.get("round"),
                e.payload.get("thread") if e.step is None else None)

    by_key: dict[Any, Event] = {}
    for e in disk_events:            # disk first: its verdict/blocking wins
        by_key[key(e)] = e
    for e in call_events:
        k = key(e)
        if k in by_key:
            if by_key[k].payload.get("duration_sec") is None:
                by_key[k].payload["duration_sec"] = e.payload.get("duration_sec")
        else:
            by_key[k] = e
    return list(by_key.values())


def build_run(ref: RunRef, sessions: list[Path], live: bool = False,
              pin_path: Path | None = None) -> Run:
    # A session may span several runs back to back in one context, so
    # selecting the session is only half the job: its turns and agents must
    # also be cut down to this run's ledger span.
    start, end = ledger_src.run_span(ref.ledger, live=live)
    data = ledger_src.read_ledger(ref.ledger)
    streams: list[list[Event]] = [ledger_src.ledger_events(ref.ledger)]
    if ref.build_log:
        streams.append(buildlog.buildlog_events(ref.build_log))
    disk_all = (gptgates.gpt_events(ref.gpt_dir, nesting=ref.layout.gpt_nesting)
                if ref.gpt_dir else [])
    disk_gate = [e for e in disk_all if e.kind == "md.review"]
    disk_other = [e for e in disk_all if e.kind != "md.review"]
    call_events = [e for s in sessions for e in gatecalls.gate_call_events(s)
                  if _within(e.ts, start, end)]
    streams.append(_merge_gate_events(disk_gate, call_events))
    streams.append(disk_other)

    all_turns: list[transcript.Turn] = []
    agents: list[transcript.AgentRun] = []
    pre: dict[tuple[Path, int], transcript.Turn] = {}
    for s in sessions:
        streams.append(transcript.turn_events(s))
        streams.append(transcript.agent_events(s))
        kept = [t for t in transcript.read_turns(s) if _within(t.ts, start, end)]
        for t in kept:
            pre[(s, t.index)] = t
        all_turns.extend(kept)
        # An agent is kept if its own activity is in-span (the original
        # rule), *or* the turn that dispatched it is -- an agent's own first
        # output can land just after the run's `end` even though the turn
        # that spawned it is squarely inside the span, and dropping it here
        # would silently violate `Step.agents`' contract for that step.
        raw_agents = transcript.read_agents(s)
        session_agents = [a for a in raw_agents
                          if _within(a.ts_start, start, end)
                          or (a.parent_turn is not None
                              and (s, a.parent_turn) in pre)]
        for a in session_agents:
            a.session = s
        agents.extend(session_agents)
    _renumber_within_run(all_turns)

    # `parent_turn` was read with the session's own numbering, which
    # `_renumber_within_run` has just replaced. Remap it, or an agent from the
    # second session points at the first session's turn of the same number. A
    # dispatching turn outside the span was never kept, so its agent gets None
    # rather than a stale index.
    for a in agents:
        if a.parent_turn is not None:
            t = pre.get((a.session, a.parent_turn))
            a.parent_turn = t.index if t is not None else None

    assign_agent_roles(agents, ref.layout)

    # An agent belongs to the step whose turns dispatched it -- the first
    # consumer to read `parent_turn` and `Step.turns` together, which is why
    # they had to be put on the same scale first.
    agents_by_turn: dict[int, list[Any]] = {}
    for a in agents:
        if a.parent_turn is not None:
            agents_by_turn.setdefault(a.parent_turn, []).append(a)

    # `ledger_windows` is what the ledger itself states; `windows` may carry a
    # synthetic open-ended one for the in-flight step. Span-unboundedness is a
    # property of the former only -- asking the latter would mean a fresh run
    # stopped reporting it in live mode, which is the one surface it is for.
    ledger_windows = ledger_src.step_windows(ref.ledger)
    hard_end = ledger_windows[-1][1] if ledger_windows else None
    windows = (_live_windows(ledger_windows, data["steps"]) if live
               else ledger_windows)
    buckets = assign_turns_to_steps(windows, all_turns, start)
    by_index = {t.index: t for t in all_turns}
    streams.append(_attribution_anomalies(ref, start, all_turns,
                                          ledger_windows, buckets,
                                          len(sessions),
                                          hard_end if live else None))
    events = merge(*streams)

    steps: list[Step] = []
    for raw in data["steps"]:
        sid = raw.get("id")
        turn_ids = buckets.get(sid, [])
        steps.append(Step(
            id=sid, status=raw.get("status"), owner=raw.get("owner"),
            ledger_tier=raw.get("tier"), attempts=raw.get("attempts"),
            deliverable=raw.get("deliverable"),
            depends_on=raw.get("depends_on") or [],
            turns=turn_ids,
            output_tokens=sum(by_index[i].output_tokens for i in turn_ids
                              if i in by_index),
            agents=[a for i in turn_ids for a in agents_by_turn.get(i, [])],
            gate_class=ledger_src._str(raw.get("gate_class")),
            answered_by=ledger_src._str(raw.get("answered_by")),
            ledger_tokens=ledger_src._num(raw.get("tokens")),
        ))

    docs = sorted({e.artifact_path for e in events
                   if e.kind == "doc.write" and e.artifact_path})
    doc_paths, doc_notes = _split_deliverables(docs, ref.project)
    anomalies = [e for e in events if e.kind == "anomaly"]
    return Run(ref=ref, steps=steps, agents=agents, events=events,
               docs=docs, session_paths=sessions, anomalies=anomalies,
               orchestrator_output_tokens=sum(t.output_tokens
                                              for t in all_turns),
               orchestrator_models=list(dict.fromkeys(
                   t.model for t in sorted(all_turns, key=lambda t: t.index)
                   if t.model)),
               doc_paths=doc_paths, doc_notes=doc_notes,
               sessions=data["sessions"], dispatches=data["dispatches"],
               fable_rulings=data["fable_rulings"],
               reviewer_pin=gatepin.reviewer_pin(ref.layout, ref.slug, pin_path))


# --- Honesty about the numbers ----------------------------------------------
#
# agentview validates runs and hunts blind spots; it is not a billing system.
# The attribution arithmetic below is deliberately *not* adjusted by any of
# these. When a figure cannot be clean, the tool says so rather than quietly
# producing a tidier number.


def _anomaly(reason: str, ref: RunRef, payload: dict[str, Any],
             ts: datetime | None = None, step: str | None = None) -> Event:
    return Event(ts=ts, kind="anomaly", role="unknown", step=step,
                 payload={"reason": reason, "slug": ref.slug,
                          "state_dir": str(ref.state_dir), **payload},
                 artifact_path=str(ref.ledger), source="agentview")


def _attribution_anomalies(ref: RunRef, start: datetime | None,
                           turns: list[transcript.Turn],
                           ledger_windows: list[tuple[str, datetime]],
                           buckets: dict[str, list[int]],
                           n_sessions: int,
                           live_since: datetime | None = None) -> list[Event]:
    """The single-run tells that a figure is not what it looks like.

    `ledger_windows` must be the windows the ledger states, never the live
    list: a synthetic window for the in-flight step says nothing about
    whether the run's span has an upper bound.
    """
    out: list[Event] = []

    # Live mode drops the span's upper bound, so the pane legitimately counts
    # turns the report never will. On a run that is not actually in flight
    # that difference can be large -- 4.5M against the report's 542K for one
    # corpus run -- and a lone "provisional" label does not explain an eight-
    # fold gap. Say which turns account for it.
    if live_since is not None:
        after = [t for t in turns if t.ts is not None and t.ts > live_since]
        if after:
            out.append(_anomaly("live-span-open", ref, {
                "since": live_since.isoformat(), "turns": len(after),
                "output_tokens": sum(t.output_tokens for t in after)},
                ts=live_since))

    if not ledger_windows:
        out.append(_anomaly("unbounded-span", ref, {
            "span_start": start.isoformat() if start else None,
            "sessions": n_sessions}))

    dated = [t.ts for t in turns if t.ts is not None]
    if start is not None and dated:
        first = min(dated)
        gap = (first - start).total_seconds()
        if gap > _COVERAGE_GAP_SEC:
            out.append(_anomaly("coverage-starts-late", ref, {
                "span_start": start.isoformat(),
                "first_turn_ts": first.isoformat(),
                "gap_seconds": gap}, ts=start))

    filled = [(sid, ids) for sid, ids in buckets.items() if ids]
    if len(buckets) > 1 and len(filled) == 1:
        sid, ids = filled[0]
        out.append(_anomaly("all-turns-in-one-step", ref,
                            {"step": sid, "turns": len(ids)}, step=sid))
    return out


def _shared_turn_totals(a: Run, b: Run, lo: datetime,
                        hi: datetime) -> tuple[int, int]:
    """Turns counted by both runs: same transcript, inside both spans."""
    turns = tokens = 0
    for path in sorted(set(a.session_paths) & set(b.session_paths)):
        for t in transcript.read_turns(path):
            if t.ts is not None and lo <= t.ts <= hi:
                turns += 1
                tokens += t.output_tokens
    return turns, tokens


def cross_run_anomalies(runs: Sequence[Run]) -> list[Event]:
    """Overlap warnings that need sibling runs to see.

    Two runs' ledger spans can overlap even when the operator worked strictly
    one session at a time -- starting the next plan while still closing out
    the last one's final steps. Every turn inside both spans of a shared
    transcript is then counted by both runs. `build_run` sees a single run
    and cannot know this, so it is computed here over the whole set.

    Emitted once per run per overlapping pair, so a surface can select a
    given run's warnings by `payload["state_dir"]`.
    """
    spans = [(r, *ledger_src.run_span(r.ref.ledger)) for r in runs]
    out: list[Event] = []
    for (a, a0, a1), (b, b0, b1) in combinations(spans, 2):
        if a0 is None or a1 is None or b0 is None or b1 is None:
            continue
        lo, hi = max(a0, b0), min(a1, b1)
        if hi <= lo:
            continue
        shared_turns, shared_tokens = _shared_turn_totals(a, b, lo, hi)
        payload = {"overlap_seconds": (hi - lo).total_seconds(),
                   "shared_turns": shared_turns,
                   "shared_output_tokens": shared_tokens}
        for one, other in ((a, b), (b, a)):
            out.append(_anomaly("run-span-overlap", one.ref,
                                {**payload, "other_slug": other.ref.slug},
                                ts=lo))
    return out


def run_span_overlap_events(ref: RunRef,
                            siblings: Sequence[RunRef]) -> list[Event]:
    """The cheap, span-only form of `cross_run_anomalies`, for the live pane.

    Reads ledgers only -- no transcripts -- because the pane recomputes this
    on every redraw. It therefore cannot state the shared turn counts, and
    `describe_anomaly` says "may be counted in both" instead of a figure.
    """
    start, end = ledger_src.run_span(ref.ledger)
    if start is None or end is None:
        return []
    out: list[Event] = []
    for other in siblings:
        if other.state_dir == ref.state_dir:
            continue
        o0, o1 = ledger_src.run_span(other.ledger)
        if o0 is None or o1 is None:
            continue
        lo, hi = max(start, o0), min(end, o1)
        if hi <= lo:
            continue
        out.append(_anomaly("run-span-overlap", ref,
                            {"other_slug": other.slug,
                             "overlap_seconds": (hi - lo).total_seconds()},
                            ts=lo))
    return out
