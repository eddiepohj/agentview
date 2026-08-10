"""BUILD-LOG.md reader. Lenient by design: drift is reported, not dropped."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..events import Event

VOCAB = frozenset({
    "complete", "complete-retry", "halted",
    "red-verified", "suspended", "resumed", "gate-open", "gate-cleared",
    "informational", "follow-up", "divergence", "waiver",
})

_LINE = re.compile(r"^(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2})\s*\|(.*)$")


def read_buildlog(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n, line in enumerate(Path(path).read_text(errors="replace").splitlines(), 1):
        m = _LINE.match(line)
        if not m:
            continue
        fields = [f.strip() for f in m.group(3).split("|", 2)]
        if len(fields) < 3:
            continue
        step, status, summary = fields[0], fields[1], fields[2]
        rows.append({"line_no": n, "date": m.group(1), "time": m.group(2),
                     "step": step, "status": status, "summary": summary,
                     "off_vocab": status not in VOCAB})
    return rows


def buildlog_events(path: Path) -> list[Event]:
    """Events with ts=None: BUILD-LOG time is local and must not be correlated."""
    events = []
    for row in read_buildlog(path):
        events.append(Event(
            ts=None,
            kind="anomaly" if row["off_vocab"] else "step.verify",
            role="step-runner",
            step=row["step"] or None,
            payload={"status": row["status"], "summary": row["summary"],
                     "local_time": f"{row['date']} {row['time']}",
                     "line_no": row["line_no"],
                     **({"reason": "off-vocabulary-status"} if row["off_vocab"] else {})},
            artifact_path=str(path),
            source="buildlog",
        ))
    return events
