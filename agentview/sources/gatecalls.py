"""The MD's gate calls, read from the transcript of the session that made them.

The MD is a shell call outside the agent tree, so it has no transcript of its
own -- which is why v1 recorded it as unobservable. That looked at the wrong
session. The call is *synchronous*: the step-runner blocks on it, so both halves
land in the step-runner's transcript -- the Bash tool call going out, the tool
result coming back. demo-run carries 40, and the result body is gate.py's own
JSON, the same shape `gptgates.py` parses off disk. No `--emit-dir` needed, and
no skill change: this works for runs that predate the emitter as well as after.
"""
from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from ..events import Event, parse_iso

_SUBCOMMANDS = {"review", "reevaluate"}
_BLOCKING_SEVERITIES = {"critical", "high"}
_FLAGS = {"--gate", "--round", "--thread", "--step"}


def _records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def _parse_command(command: str) -> dict[str, str] | None:
    """The parsed flags of a `gate.py review|reevaluate` call, or None if
    `command` carries neither subcommand. Flags may appear in any order, so
    they are scanned for by name rather than by position.

    The subcommand must immediately follow a `gate.py` token -- not merely
    appear anywhere in the tokenized command. `shlex.split` does not
    understand heredocs, so a `cat > f << 'EOF' ... EOF` command whose body
    happens to prose-mention "review" (e.g. dispatching a Gate A review
    write-up) tokenizes that word too; requiring adjacency to the script
    name is what tells a real invocation from that.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    subcommand = None
    for i, tok in enumerate(tokens):
        if (tok == "gate.py" or tok.endswith("/gate.py")) \
                and i + 1 < len(tokens) and tokens[i + 1] in _SUBCOMMANDS:
            subcommand = tokens[i + 1]
            break
    if subcommand is None:
        return None
    parsed = {"subcommand": subcommand}
    for i, tok in enumerate(tokens):
        if tok in _FLAGS and i + 1 < len(tokens):
            parsed[tok[2:]] = tokens[i + 1]
    return parsed


def _classify(body: Any) -> tuple[str | None, list[dict[str, Any]]]:
    """`(outcome, blocking)` from a gate.py verdict body. A non-JSON,
    truncated, or structurally invalid body (missing/wrong-shaped
    `report`/`report.risks`) drops the risks and leaves outcome None --
    never raises, and is never mistaken for a genuine "reviewed, zero
    risks found" pass. Only a well-shaped body earns `"pass"`; `gptgates.py`
    applies this same discipline to disk-sourced rounds
    (`if not isinstance(data, dict): ... malformed`) and this holds
    transcript-sourced ones to the same standard."""
    if not isinstance(body, str) or not body:
        return None, []
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None, []
    if not isinstance(data, dict):
        return None, []
    report = data.get("report")
    if not isinstance(report, dict):
        return None, []
    risks = report.get("risks")
    if not isinstance(risks, list):
        return None, []
    blocking = [r for r in risks if isinstance(r, dict)
                and r.get("severity") in _BLOCKING_SEVERITIES]
    return ("escalate" if blocking else "pass"), blocking


def gate_call_events(session_path: Path) -> list[Event]:
    records = _records(session_path)

    # tool_use_id -> (return timestamp, reply body, is_error). Built first
    # so calls are correlated to their own result, never positionally.
    results: dict[str, tuple[str | None, Any, bool]] = {}
    for r in records:
        if r.get("type") != "user":
            continue
        message = r.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for c in content:
            if (isinstance(c, dict) and c.get("type") == "tool_result"
                    and c.get("tool_use_id")):
                results[c["tool_use_id"]] = (r.get("timestamp"), c.get("content"),
                                             bool(c.get("is_error", False)))

    events: list[Event] = []
    for r in records:
        if r.get("type") != "assistant":
            continue
        message = r.get("message")
        content = message.get("content") if isinstance(message, dict) else []
        for c in content or []:
            if not (isinstance(c, dict) and c.get("type") == "tool_use"
                    and c.get("name") == "Bash"):
                continue
            command = (c.get("input") or {}).get("command")
            if not isinstance(command, str):
                continue
            parsed = _parse_command(command)
            if parsed is None:
                continue

            send_ts = parse_iso(r.get("timestamp"))
            result_ts, body, is_error = results.get(c.get("id"),
                                                     (None, None, False))
            return_ts = parse_iso(result_ts)
            duration = ((return_ts - send_ts).total_seconds()
                        if send_ts is not None and return_ts is not None
                        else None)
            # An errored Bash call (gate.py crashed, timed out, ...) must
            # never be synthesized as a clean "pass" just because its body
            # happens to be JSON without report/risks keys -- that would
            # report a gate that never ran as one that passed.
            outcome, blocking = (None, []) if is_error else _classify(body)

            round_raw = parsed.get("round")
            round_val: int | str | None = round_raw
            if isinstance(round_raw, str) and round_raw.isdigit():
                round_val = int(round_raw)

            payload = {
                "gate": parsed.get("gate"),
                "round": round_val,
                "thread": parsed.get("thread"),
                "subcommand": parsed.get("subcommand"),
                "outcome": outcome,
                "blocking": blocking,
                "duration_sec": duration,
            }
            kind = "md.escalate" if outcome == "escalate" else "md.review"
            events.append(Event(send_ts, kind, "md", parsed.get("step"),
                                payload, None, "gatecalls"))
    return events
