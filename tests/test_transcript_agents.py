import json
from agentview.sources.transcript import read_agents, agent_events


def _make_session(tmp_path, assistant):
    sess = tmp_path / "s1.jsonl"
    sess.write_text("\n".join(json.dumps(r) for r in [
        assistant("2026-06-25T10:00:00Z"),
        assistant("2026-06-25T10:05:00Z", tools=[{
            "type": "tool_use", "id": "toolu_A", "name": "Agent",
            "input": {"description": "Wiki scan", "subagent_type": "Explore",
                      "prompt": "..."}}]),
    ]) + "\n")
    sub = tmp_path / "s1" / "subagents"
    sub.mkdir(parents=True)
    (sub / "agent-x1.meta.json").write_text(json.dumps(
        {"agentType": "Explore", "description": "Wiki scan",
         "toolUseId": "toolu_A", "spawnDepth": 1}))
    (sub / "agent-x1.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"type": "assistant", "timestamp": "2026-06-25T10:05:10Z",
         "message": {"model": "claude-sonnet-4-6",
                     "usage": {"output_tokens": 300, "input_tokens": 1,
                               "cache_read_input_tokens": 5,
                               "cache_creation_input_tokens": 0},
                     "content": []}},
        {"type": "assistant", "timestamp": "2026-06-25T10:06:10Z",
         "message": {"model": "claude-sonnet-4-6",
                     "usage": {"output_tokens": 200, "input_tokens": 1,
                               "cache_read_input_tokens": 5,
                               "cache_creation_input_tokens": 0},
                     "content": []}},
    ]) + "\n")
    return sess


def test_agent_links_to_parent_turn_via_tool_use_id(tmp_path, assistant):
    sess = _make_session(tmp_path, assistant)
    agents = read_agents(sess)
    assert len(agents) == 1
    assert agents[0].parent_turn == 2
    assert agents[0].agent_type == "Explore"


def test_agent_model_comes_from_the_subagent_transcript(tmp_path, assistant):
    sess = _make_session(tmp_path, assistant)
    assert read_agents(sess)[0].model == "claude-sonnet-4-6"


def test_agent_tokens_and_duration_are_summed(tmp_path, assistant):
    sess = _make_session(tmp_path, assistant)
    a = read_agents(sess)[0]
    assert a.output_tokens == 500
    assert a.duration_sec == 60.0


def test_orphan_agent_without_matching_tool_use_has_no_parent(tmp_path, assistant):
    sess = _make_session(tmp_path, assistant)
    sub = tmp_path / "s1" / "subagents"
    (sub / "agent-x2.meta.json").write_text(json.dumps(
        {"agentType": "general-purpose", "description": "orphan",
         "toolUseId": "toolu_MISSING"}))
    (sub / "agent-x2.jsonl").write_text(json.dumps(
        {"type": "assistant", "timestamp": "2026-06-25T10:07:00Z",
         "message": {"model": "claude-haiku-4-5", "usage": {"output_tokens": 10},
                     "content": []}}) + "\n")
    parents = {a.agent_id: a.parent_turn for a in read_agents(sess)}
    assert parents["x2"] is None


def test_agent_events_emit_spawn_and_return(tmp_path, assistant):
    sess = _make_session(tmp_path, assistant)
    kinds = [e.kind for e in agent_events(sess)]
    assert kinds.count("agent.spawn") == 1
    assert kinds.count("agent.return") == 1


def test_missing_subagents_directory_yields_no_agents(tmp_path, assistant, write_jsonl):
    p = write_jsonl("lonely.jsonl", [assistant("2026-06-25T10:00:00Z")])
    assert read_agents(p) == []


def test_agent_link_uses_tool_use_id_not_position(tmp_path, assistant):
    # Two Agent tool_use blocks on different turns; the single subagent's
    # meta.json references the *second* one (toolu_B, turn 4). An
    # implementation that matched positionally (e.g. by first Agent call
    # found) would wrongly return turn 2.
    sess = tmp_path / "s2.jsonl"
    sess.write_text("\n".join(json.dumps(r) for r in [
        assistant("2026-06-25T09:00:00Z"),
        assistant("2026-06-25T09:01:00Z", tools=[{
            "type": "tool_use", "id": "toolu_A", "name": "Agent",
            "input": {"description": "first", "subagent_type": "Explore",
                      "prompt": "..."}}]),
        assistant("2026-06-25T09:02:00Z"),
        assistant("2026-06-25T09:03:00Z", tools=[{
            "type": "tool_use", "id": "toolu_B", "name": "Agent",
            "input": {"description": "second", "subagent_type": "Explore",
                      "prompt": "..."}}]),
    ]) + "\n")
    sub = tmp_path / "s2" / "subagents"
    sub.mkdir(parents=True)
    (sub / "agent-y1.meta.json").write_text(json.dumps(
        {"agentType": "Explore", "description": "second",
         "toolUseId": "toolu_B"}))
    (sub / "agent-y1.jsonl").write_text(json.dumps(
        {"type": "assistant", "timestamp": "2026-06-25T09:03:10Z",
         "message": {"model": "claude-sonnet-4-6",
                     "usage": {"output_tokens": 50}, "content": []}}) + "\n")
    agents = read_agents(sess)
    assert len(agents) == 1
    assert agents[0].parent_turn == 4


def test_malformed_agent_meta_yields_orphan_and_anomaly_event(tmp_path, assistant):
    sess = _make_session(tmp_path, assistant)
    sub = tmp_path / "s1" / "subagents"
    (sub / "agent-bad.meta.json").write_text("{not valid json")
    (sub / "agent-bad.jsonl").write_text(json.dumps(
        {"type": "assistant", "timestamp": "2026-06-25T10:08:00Z",
         "message": {"model": "claude-haiku-4-5",
                     "usage": {"output_tokens": 5}, "content": []}}) + "\n")

    agents = {a.agent_id: a for a in read_agents(sess)}
    assert "bad" in agents
    assert agents["bad"].parent_turn is None
    assert agents["bad"].agent_type is None

    anomalies = [e for e in agent_events(sess) if e.kind == "anomaly"]
    assert len(anomalies) == 1
    assert anomalies[0].payload["reason"] == "unreadable-agent-meta"
    assert anomalies[0].payload["agent_id"] == "bad"


def test_agent_meta_without_matching_transcript_has_zero_tokens(tmp_path, assistant):
    sess = _make_session(tmp_path, assistant)
    sub = tmp_path / "s1" / "subagents"
    (sub / "agent-crashed.meta.json").write_text(json.dumps(
        {"agentType": "Explore", "description": "crashed before writing",
         "toolUseId": "toolu_A"}))
    # No agent-crashed.jsonl — the subagent crashed before writing anything.

    agents = {a.agent_id: a for a in read_agents(sess)}
    a = agents["crashed"]
    assert a.output_tokens == 0
    assert a.model is None
    assert a.duration_sec is None
