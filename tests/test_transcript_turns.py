from agentview.sources.transcript import read_turns, session_meta, turn_events


def test_reads_only_assistant_turns_with_usage(write_jsonl, assistant):
    p = write_jsonl("s.jsonl", [
        {"type": "user", "message": {"content": "hi"}},
        assistant("2026-06-25T10:00:00Z", out=50),
        {"type": "assistant", "timestamp": "2026-06-25T10:01:00Z",
         "message": {"model": "m", "content": []}},
    ])
    turns = read_turns(p)
    assert len(turns) == 1
    assert turns[0].index == 1
    assert turns[0].output_tokens == 50


def test_context_tokens_sums_all_three_input_fields(write_jsonl, assistant):
    p = write_jsonl("s.jsonl", [assistant("2026-06-25T10:00:00Z", cache_read=5000)])
    assert read_turns(p)[0].context_tokens == 2 + 10 + 5000


def test_tool_errors_attach_to_the_preceding_turn(write_jsonl, assistant, tool_result):
    p = write_jsonl("s.jsonl", [
        assistant("2026-06-25T10:00:00Z",
                  tools=[{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]),
        tool_result("t1", is_error=True),
    ])
    assert read_turns(p)[0].errors == 1


def test_tool_error_attaches_to_middle_turn_not_first_or_last(write_jsonl, assistant, tool_result):
    p = write_jsonl("s.jsonl", [
        assistant("2026-06-25T10:00:00Z"),
        assistant("2026-06-25T10:01:00Z",
                  tools=[{"type": "tool_use", "id": "t2", "name": "Bash", "input": {}}]),
        tool_result("t2", is_error=True),
        assistant("2026-06-25T10:02:00Z"),
    ])
    turns = read_turns(p)
    assert len(turns) == 3
    assert turns[0].errors == 0
    assert turns[1].errors == 1
    assert turns[2].errors == 0


def test_unconsumed_record_types_are_aggregated_into_one_anomaly(write_jsonl, assistant):
    p = write_jsonl("s.jsonl", [
        {"type": "system", "subtype": "compact_boundary"},
        assistant("2026-06-25T10:00:00Z"),
        {"type": "summary", "summary": "..."},
        {"type": "system", "subtype": "other"},
    ])
    anomalies = [e for e in turn_events(p) if e.kind == "anomaly"
                 and e.payload.get("reason") == "unconsumed-record-types"]
    assert len(anomalies) == 1
    assert anomalies[0].payload["types"] == {"system": 2, "summary": 1}


def test_no_unconsumed_record_type_anomaly_when_only_user_and_assistant(write_jsonl, assistant):
    p = write_jsonl("s.jsonl", [
        {"type": "user", "message": {"content": "hi"}},
        assistant("2026-06-25T10:00:00Z"),
    ])
    anomalies = [e for e in turn_events(p) if e.kind == "anomaly"
                 and e.payload.get("reason") == "unconsumed-record-types"]
    assert anomalies == []


def test_malformed_lines_are_skipped_and_counted(write_jsonl, assistant):
    p = write_jsonl("s.jsonl", [assistant("2026-06-25T10:00:00Z")])
    p.write_text(p.read_text() + "{not json\n")
    turns = read_turns(p)
    assert len(turns) == 1
    anomalies = [e for e in turn_events(p) if e.kind == "anomaly"]
    assert anomalies and anomalies[0].payload["reason"] == "malformed-jsonl"


def test_session_meta_extracts_cwd_and_id(write_jsonl, assistant):
    p = write_jsonl("s.jsonl", [assistant("2026-06-25T10:00:00Z")])
    m = session_meta(p)
    assert m["cwd"] == "/proj" and m["session_id"] == "s1"


def test_run_start_and_exit_events_bracket_the_session(write_jsonl, assistant):
    p = write_jsonl("s.jsonl", [
        assistant("2026-06-25T10:00:00Z"),
        assistant("2026-06-25T11:00:00Z"),
    ])
    kinds = [e.kind for e in turn_events(p)]
    assert kinds[0] == "run.start" and kinds[-1] == "run.exit"
