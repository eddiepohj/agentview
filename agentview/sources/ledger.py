"""Run-state ledger reader. Handles plan-runner and light-runner schemas."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..events import Event, parse_iso

def _dicts(value: Any) -> list[dict[str, Any]]:
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def _num(value: Any) -> int | float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _strs(value: Any) -> list[str]:
    return [v for v in value if isinstance(v, str)] if isinstance(value, list) else []


def _sanitize(entries: list[dict[str, Any]], numeric: tuple[str, ...],
             text: tuple[str, ...], text_lists: tuple[str, ...] = ()
             ) -> list[dict[str, Any]]:
    out = []
    for e in entries:
        row = dict(e)
        for k in numeric:
            if k in row:
                row[k] = _num(row[k])
        for k in text:
            if k in row:
                row[k] = _str(row[k])
        for k in text_lists:
            if k in row:
                row[k] = _strs(row[k])
        out.append(row)
    return out


_STATUS_KIND = {
    "complete": "step.complete",
    "complete-retry": "step.complete",
    "halted": "step.halt",
    "suspended": "step.suspend",
    "in-progress": "step.dispatch",
    "recon": "step.dispatch",
    "awaiting-human": "gate.open",
}


def read_ledger(path: Path) -> dict[str, Any]:
    empty = {"steps": [], "meta": {}, "dispatches": [], "sessions": [],
             "fable_rulings": [], "issues": []}
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return {"schema": "unknown", **empty,
                "issues": [{"reason": "unreadable-ledger",
                            "detail": str(exc)}]}
    if isinstance(raw, list):
        steps = _dicts(raw)
        issues = ([{"reason": "malformed-ledger-steps",
                    "count": len(raw) - len(steps)}]
                  if len(steps) != len(raw) else [])
        return {"schema": "light-runner", **empty, "steps": steps,
                "issues": issues}
    if not isinstance(raw, dict):
        return {"schema": "unknown", **empty,
                "issues": [{"reason": "malformed-ledger-root"}]}
    raw_steps = raw.get("steps", [])
    steps = _dicts(raw_steps)
    issues = ([] if isinstance(raw_steps, list) and len(steps) == len(raw_steps)
              else [{"reason": "malformed-ledger-steps"}])
    return {"schema": "plan-runner", "steps": steps,
            "meta": {k: v for k, v in raw.items() if k != "steps"},
            "dispatches": _sanitize(_dicts(raw.get("dispatches")),
                                    numeric=("tokens", "duration_sec"),
                                    text=("label", "kind", "step", "tier")),
            "sessions": _sanitize(_dicts(raw.get("sessions")), numeric=(),
                                  text=("session", "trigger"),
                                  text_lists=("closed", "waivers")),
            "fable_rulings": _sanitize(_dicts(raw.get("fable_rulings")),
                                       numeric=(), text=("step", "ruling",
                                                          "note")),
            "issues": issues}


def step_windows(path: Path) -> list[tuple[str, datetime]]:
    """(step_id, updated) pairs in UTC order. The correlation primitive."""
    out = []
    for s in read_ledger(path)["steps"]:
        ts = parse_iso(s.get("updated"))
        if ts:
            out.append((s.get("id"), ts))
    return sorted(out, key=lambda x: x[1])


def run_span(path: Path,
             live: bool = False) -> tuple[datetime | None, datetime | None]:
    """The run's wall-clock span `[start, end]`, derived from the ledger.

    `start` is the ledger's top-level ``created`` when present (it predates
    the first step's completion), otherwise the earliest step ``updated``.
    `end` is the latest step ``updated``. Either may be ``None`` when the
    ledger carries no usable timestamps at all, in which case callers must
    treat the span as unknown rather than empty.

    `live=True` returns an *open* upper bound (`end=None`). A ledger's last
    `updated` is the last time the orchestrator wrote state, not the moment
    the run stopped working: everything happening right now is, by
    definition, after it. Retrospective callers (the report) keep the closed
    span; the live pane, whose job is showing movement, asks for the open
    one.
    """
    data = read_ledger(path)
    stamps = sorted(ts for ts in
                    (parse_iso(s.get("updated")) for s in data["steps"]) if ts)
    created = parse_iso(data["meta"].get("created"))
    start = created or (stamps[0] if stamps else None)
    if live:
        return start, None
    return start, (stamps[-1] if stamps else None)


def ledger_events(path: Path) -> list[Event]:
    data = read_ledger(path)
    events: list[Event] = []
    for issue in data["issues"]:
        events.append(Event(
            ts=None, kind="anomaly", role="step-runner", step=None,
            payload=issue, artifact_path=str(path), source="ledger"))
    for s in data["steps"]:
        updated_raw = s.get("updated")
        ts = parse_iso(updated_raw)
        kind = _STATUS_KIND.get(s.get("status"), "step.verify")
        events.append(Event(
            ts=ts,
            kind=kind,
            role="step-runner",
            step=s.get("id"),
            payload={"status": s.get("status"), "owner": s.get("owner"),
                     "tier": s.get("tier"), "tokens": s.get("tokens"),
                     "duration_sec": s.get("duration_sec"),
                     "attempts": s.get("attempts"), "wake": s.get("wake"),
                     "track": s.get("track"), "note": s.get("note"),
                     "depends_on": s.get("depends_on") or [],
                     "schema": data["schema"]},
            artifact_path=s.get("deliverable"),
            source="ledger",
        ))
        deliverable = _str(s.get("deliverable"))
        if deliverable:
            events.append(Event(ts, "doc.write",
                                "step-runner", s.get("id"),
                                {"kind": "deliverable"}, deliverable,
                                "ledger"))
        if ts is None:
            events.append(Event(
                ts=None, kind="anomaly", role="step-runner", step=s.get("id"),
                payload={"reason": "unparseable-step-updated",
                         "updated": updated_raw},
                artifact_path=str(path), source="ledger",
            ))
    return events
