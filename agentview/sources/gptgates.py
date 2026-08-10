"""Reader for gate.py's per-round JSON drops under <state-dir>/gates/gpt/."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..events import Event, parse_iso

_NAME = re.compile(r"^(?P<gate>[0ab])-r(?P<round>\d+)\.json$")


def read_gpt_rounds(gpt_dir: Path, nesting: str = "flat") -> list[dict[str, Any]]:
    d = Path(gpt_dir)
    if not d.is_dir():
        return []
    pattern = "*/*.json" if nesting == "per-step" else "*.json"
    rows: list[dict[str, Any]] = []
    for p in sorted(d.glob(pattern)):
        m = _NAME.match(p.name)
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            row: dict[str, Any] = {"_malformed": True, "path": str(p)}
            if nesting == "per-step":
                row["step"] = p.parent.name
            rows.append(row)
            continue
        if not isinstance(data, dict):
            row = {"_malformed": True, "path": str(p)}
            if nesting == "per-step":
                row["step"] = p.parent.name
            rows.append(row)
            continue
        data.setdefault("gate", m.group("gate") if m else None)
        data.setdefault("round", int(m.group("round")) if m else None)
        if nesting == "per-step":
            data.setdefault("step", p.parent.name)
        data["path"] = str(p)
        rows.append(data)
    return sorted(rows, key=lambda r: (r.get("gate") or "", r.get("round") or 0))


def gpt_events(gpt_dir: Path, nesting: str = "flat") -> list[Event]:
    events: list[Event] = []
    for r in read_gpt_rounds(gpt_dir, nesting=nesting):
        step = r.get("step")
        if r.get("_malformed"):
            events.append(Event(None, "anomaly", "md", step,
                                {"reason": "malformed-gate-round"},
                                r["path"], "gptgates"))
            continue
        ts = parse_iso(r.get("ts"))
        classification = r.get("classification")
        classification = classification if isinstance(classification, dict) else {}
        report = r.get("report")
        report = report if isinstance(report, dict) else {}
        blocking = classification.get("blocking")
        blocking = blocking if isinstance(blocking, list) else []
        advisory = classification.get("advisory")
        advisory = advisory if isinstance(advisory, list) else []
        risks = report.get("risks")
        risks = risks if isinstance(risks, list) else []
        events.append(Event(ts, "md.review", "md", step,
                            {"gate": r.get("gate"), "round": r.get("round"),
                             "outcome": r.get("outcome"), "blocking": blocking,
                             "advisory": advisory,
                             "tokens": r.get("tokens"),
                             "risks": risks},
                            r["path"], "gptgates"))
        if r.get("outcome") == "escalate":
            events.append(Event(ts, "md.escalate", "md", step,
                                {"gate": r.get("gate"), "round": r.get("round"),
                                 "surviving": blocking}, r["path"], "gptgates"))
        events.append(Event(ts, "doc.write", "md", step, {"kind": "gpt-round"},
                            r["path"], "gptgates"))
    return events
