"""Session transcript reader: turns, token usage, tool calls, anomalies."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..events import Event, parse_iso


@dataclass
class Turn:
    index: int
    ts: datetime | None
    model: str | None
    output_tokens: int
    context_tokens: int
    tool_uses: list[dict[str, Any]] = field(default_factory=list)
    errors: int = 0


def _records(path: Path) -> tuple[list[dict], int]:
    records, bad = [], 0
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(record, dict):
                records.append(record)
            else:
                bad += 1
    return records, bad


def session_meta(path: Path) -> dict[str, Any]:
    records, _ = _records(path)
    meta = {"cwd": None, "session_id": None, "version": None, "git_branch": None}
    for r in records:
        meta["cwd"] = r.get("cwd") or meta["cwd"]
        meta["session_id"] = r.get("sessionId") or meta["session_id"]
        meta["version"] = r.get("version") or meta["version"]
        meta["git_branch"] = r.get("gitBranch") or meta["git_branch"]
    return meta


def read_turns(path: Path) -> list[Turn]:
    records, _ = _records(path)
    turns: list[Turn] = []
    for r in records:
        if r.get("type") == "user":
            message = r.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, list) and turns:
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "tool_result" \
                            and c.get("is_error"):
                        turns[-1].errors += 1
            continue
        if r.get("type") != "assistant":
            continue
        msg = r.get("message")
        msg = msg if isinstance(msg, dict) else {}
        usage = msg.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        if not usage:
            continue
        tools = [c for c in (msg.get("content") or [])
                 if isinstance(c, dict) and c.get("type") == "tool_use"]
        turns.append(Turn(
            index=len(turns) + 1,
            ts=parse_iso(r.get("timestamp")),
            model=msg.get("model"),
            output_tokens=_tokens(usage.get("output_tokens")),
            context_tokens=(_tokens(usage.get("input_tokens"))
                            + _tokens(usage.get("cache_creation_input_tokens"))
                            + _tokens(usage.get("cache_read_input_tokens"))),
            tool_uses=tools,
        ))
    return turns


def _tokens(value: Any) -> int:
    """A malformed usage field is unknown to the transcript, not fatal."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def turn_events(path: Path) -> list[Event]:
    records, bad = _records(path)
    turns = read_turns(path)
    meta = session_meta(path)
    events: list[Event] = []
    if turns:
        events.append(Event(turns[0].ts, "run.start", "step-runner", None,
                            {"session_id": meta["session_id"], "cwd": meta["cwd"],
                             "turns": len(turns)}, str(path), "transcript"))
    if bad:
        events.append(Event(turns[0].ts if turns else None, "anomaly", "unknown",
                            None, {"reason": "malformed-jsonl", "count": bad},
                            str(path), "transcript"))
    unconsumed: dict[str, int] = {}
    for r in records:
        t = r.get("type")
        if t not in ("user", "assistant"):
            unconsumed[t] = unconsumed.get(t, 0) + 1
    if unconsumed:
        events.append(Event(turns[0].ts if turns else None, "anomaly", "unknown",
                            None, {"reason": "unconsumed-record-types",
                                   "types": unconsumed}, str(path), "transcript"))
    if turns:
        events.append(Event(turns[-1].ts, "run.exit", "step-runner", None,
                            {"session_id": meta["session_id"]}, str(path),
                            "transcript"))
    return events


@dataclass
class AgentRun:
    agent_id: str
    agent_type: str | None
    model: str | None
    description: str | None
    parent_turn: int | None
    output_tokens: int
    duration_sec: float | None
    spawn_depth: int | None = None
    ts_start: datetime | None = None
    ts_end: datetime | None = None
    session: Path | None = None
    role: str = "worker"
    role_source: str = "observed"


def _subagents_dir(session_path: Path) -> Path:
    return session_path.parent / session_path.stem / "subagents"


def _iter_agent_meta(sub: Path):
    """Yield (agent_id, meta_path, meta, ok) for every agent-*.meta.json in sub.

    `ok` is False when the meta file could not be read/parsed; `meta` is `{}`
    in that case so callers can still produce a degraded (orphaned) AgentRun.
    """
    for meta_path in sorted(sub.glob("agent-*.meta.json")):
        agent_id = meta_path.name[len("agent-"):-len(".meta.json")]
        try:
            meta = json.loads(meta_path.read_text())
            ok = isinstance(meta, dict)
            if not ok:
                meta = {}
        except (OSError, json.JSONDecodeError):
            meta, ok = {}, False
        yield agent_id, meta_path, meta, ok


def read_agents(session_path: Path) -> list[AgentRun]:
    sub = _subagents_dir(session_path)
    if not sub.is_dir():
        return []
    parent_by_tool_use: dict[str, int] = {}
    for turn in read_turns(session_path):
        for tu in turn.tool_uses:
            if tu.get("name") == "Agent" and tu.get("id"):
                parent_by_tool_use[tu["id"]] = turn.index

    agents: list[AgentRun] = []
    for agent_id, _meta_path, meta, _ok in _iter_agent_meta(sub):
        jsonl = sub / f"agent-{agent_id}.jsonl"
        out, model, stamps = 0, None, []
        if jsonl.exists():
            records, _ = _records(jsonl)
            for r in records:
                if r.get("type") != "assistant":
                    continue
                msg = r.get("message") or {}
                usage = msg.get("usage") or {}
                if not usage:
                    continue
                out += usage.get("output_tokens") or 0
                model = model or msg.get("model")
                ts = parse_iso(r.get("timestamp"))
                if ts:
                    stamps.append(ts)
        dur = (max(stamps) - min(stamps)).total_seconds() if len(stamps) > 1 else None
        agents.append(AgentRun(
            agent_id=agent_id,
            agent_type=meta.get("agentType"),
            model=model,
            description=meta.get("description"),
            parent_turn=parent_by_tool_use.get(meta.get("toolUseId")),
            output_tokens=out,
            duration_sec=dur,
            spawn_depth=meta.get("spawnDepth"),
            ts_start=min(stamps) if stamps else None,
            ts_end=max(stamps) if stamps else None,
        ))
    return agents


def agent_events(session_path: Path) -> list[Event]:
    events: list[Event] = []
    sub = _subagents_dir(session_path)
    if sub.is_dir():
        for agent_id, meta_path, _meta, ok in _iter_agent_meta(sub):
            if not ok:
                events.append(Event(None, "anomaly", "worker", None,
                                    {"reason": "unreadable-agent-meta",
                                     "agent_id": agent_id},
                                    str(meta_path), "transcript"))
    for a in read_agents(session_path):
        base = {"agent_id": a.agent_id, "agent_type": a.agent_type,
                "model": a.model, "description": a.description,
                "parent_turn": a.parent_turn}
        events.append(Event(a.ts_start, "agent.spawn", "worker", None, base,
                            None, "transcript"))
        events.append(Event(a.ts_end, "agent.return", "worker", None,
                            {**base, "output_tokens": a.output_tokens,
                             "duration_sec": a.duration_sec}, None, "transcript"))
    return events
