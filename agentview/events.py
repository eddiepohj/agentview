# Copyright 2026 Edvard Pohjavirta
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for the full text.

"""Normalized event stream shared by every agentview source and surface."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

KINDS = frozenset({
    "run.start", "run.exit",
    "step.dispatch", "step.verify", "step.complete", "step.retry",
    "step.halt", "step.suspend", "step.resume",
    "agent.spawn", "agent.return",
    "recon.probe",
    "gate.open", "gate.answer",
    "md.review", "md.defense", "md.escalate",
    "fable.rule",
    "doc.write",
    "anomaly",
})

# The skill's taxonomy is closed: "there is no ninth, and there is no
# catch-all" (references/agent-taxonomy.md). So this list can be exhaustive
# rather than open-ended, and an unknown role is a real error.
ROLES = ("human", "step-runner", "md",
         "director", "implementer", "observer", "verifier", "skeptic",
         "recon", "defender", "fable",
         "worker", "unknown")


def parse_iso(s: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp, treating a missing offset as UTC."""
    if not s or not isinstance(s, str):
        return None
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class Event:
    ts: datetime | None
    kind: str
    role: str
    step: str | None
    payload: dict[str, Any] = field(default_factory=dict)
    artifact_path: str | None = None
    source: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown event kind: {self.kind!r}")
        if self.role not in ROLES:
            raise ValueError(f"unknown event role: {self.role!r}")


# Anomalies that state "this number is not clean". They are not buried in a
# table with the rest; both surfaces lift them to the top. `anomaly` remains
# a single event kind -- these are `payload["reason"]` values, not new kinds.
ACCURACY_REASONS = (
    "run-span-overlap",
    "coverage-starts-late",
    "all-turns-in-one-step",
    "unbounded-span",
    "live-span-open",
)


def _hms(seconds: float | None) -> str:
    if seconds is None:
        return "an unknown time"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{sec:02d}s" if m else f"{sec}s"


def describe_anomaly(e: Event) -> str:
    """One plain line saying what is wrong with a figure. Shared by surfaces
    so the report and the live pane never disagree about what happened."""
    p = e.payload
    reason = p.get("reason")
    if reason == "run-span-overlap":
        line = (f"run span overlaps sibling run '{p.get('other_slug')}' by "
                f"{_hms(p.get('overlap_seconds'))}")
        shared = p.get("shared_turns")
        if shared is None:
            return line + " — token figures may be counted in both runs"
        return (line + f" — {shared} turns / "
                f"{p.get('shared_output_tokens', 0):,} output tokens are "
                "counted in both runs")
    if reason == "coverage-starts-late":
        return (f"transcript coverage begins {_hms(p.get('gap_seconds'))} "
                f"after the run's span started ({p.get('span_start')}); the "
                "work before that is missing from every figure")
    if reason == "all-turns-in-one-step":
        return (f"all {p.get('turns')} attributed turns landed in step "
                f"{p.get('step')}; every other step reads zero, so the "
                "per-step split is not trustworthy")
    if reason == "live-span-open":
        return (f"{p.get('turns')} turns / {p.get('output_tokens', 0):,} "
                "output tokens here were produced after the run's last "
                f"ledger write ({p.get('since')}); the live span has no "
                "upper bound, so this figure exceeds the report's")
    if reason == "unbounded-span":
        return ("the ledger carries no parseable step timestamp, so the "
                "run's span has no upper bound; sessions were selected by "
                "reference alone and may include unrelated work")
    return f"{reason}: {p}"


_FAR_PAST = datetime.min.replace(tzinfo=timezone.utc)


def merge(*streams: Iterable[Event]) -> list[Event]:
    """Merge event streams into one list ordered by timestamp, stably."""
    out: list[Event] = []
    for s in streams:
        out.extend(s)
    return sorted(out, key=lambda e: e.ts or _FAR_PAST)
