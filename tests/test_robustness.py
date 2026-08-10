"""Malformed external artifacts degrade to diagnostics instead of crashes."""
import json

from agentview.discovery import find_runs
from agentview.model import build_run
from agentview.sources.gatecalls import gate_call_events
from agentview.sources.gptgates import gpt_events
from agentview.sources.ledger import ledger_events, read_ledger
from agentview.sources.transcript import read_agents, read_turns, turn_events


def _state(tmp_path, payload):
    state = tmp_path / "project" / "_planrunner" / "state"
    state.mkdir(parents=True)
    (state / "ledger.json").write_text(json.dumps(payload))
    return state


def test_wrong_shaped_ledger_degrades_to_an_anomaly(tmp_path):
    state = _state(tmp_path, "not a ledger")
    data = read_ledger(state / "ledger.json")
    assert data["steps"] == []
    assert data["issues"] == [{"reason": "malformed-ledger-root"}]
    events = ledger_events(state / "ledger.json")
    assert events[0].payload["reason"] == "malformed-ledger-root"
    assert build_run(find_runs(tmp_path)[0], []).steps == []


def test_non_mapping_ledger_steps_are_ignored(tmp_path):
    state = _state(tmp_path, [None, {"id": "A", "status": "complete"}])
    data = read_ledger(state / "ledger.json")
    assert [s["id"] for s in data["steps"]] == ["A"]
    assert data["issues"][0]["reason"] == "malformed-ledger-steps"


def test_wrong_shaped_gate_sections_do_not_crash(tmp_path):
    gates = tmp_path / "gates"
    gates.mkdir()
    (gates / "a-r1.json").write_text(json.dumps({
        "classification": "wrong", "report": ["wrong"]}))
    events = gpt_events(gates)
    review = next(e for e in events if e.kind == "md.review")
    assert review.payload["blocking"] == []
    assert review.payload["risks"] == []


def test_non_mapping_jsonl_records_become_malformed_diagnostics(tmp_path):
    session = tmp_path / "session.jsonl"
    session.write_text("[]\n" + json.dumps({
        "type": "assistant", "timestamp": "2026-01-01T00:00:00Z",
        "message": {"usage": {"output_tokens": "wrong"}, "content": []}})
                       + "\n")
    assert read_turns(session)[0].output_tokens == 0
    assert any(e.payload.get("reason") == "malformed-jsonl"
               for e in turn_events(session))
    assert gate_call_events(session) == []


def test_wrong_shaped_jsonl_message_does_not_crash_readers(tmp_path):
    session = tmp_path / "session.jsonl"
    session.write_text(json.dumps({"type": "user", "message": []}) + "\n"
                       + json.dumps({"type": "assistant", "message": []}) + "\n")
    assert read_turns(session) == []
    assert gate_call_events(session) == []


def test_non_mapping_agent_metadata_degrades_to_orphaned_agent(tmp_path):
    session = tmp_path / "session.jsonl"
    session.write_text("")
    subagents = tmp_path / "session" / "subagents"
    subagents.mkdir(parents=True)
    (subagents / "agent-a.meta.json").write_text("[]")
    agents = read_agents(session)
    assert len(agents) == 1
    assert agents[0].agent_type is None
