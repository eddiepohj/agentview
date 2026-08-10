import json

from agentview.sources.gatecalls import gate_call_events


def _session(tmp_path, records):
    p = tmp_path / "session.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return p


def _call(ts, tool_id, command):
    return {
        "type": "assistant", "timestamp": ts, "cwd": "/proj",
        "sessionId": "s1", "version": "2.1.169", "gitBranch": "main",
        "message": {
            "model": "claude-opus-4-8",
            "usage": {"input_tokens": 2, "output_tokens": 10,
                      "cache_creation_input_tokens": 10,
                      "cache_read_input_tokens": 1000},
            "content": [{"type": "tool_use", "id": tool_id, "name": "Bash",
                        "input": {"command": command}}],
        },
    }


def _result(ts, tool_id, body, is_error=False):
    return {"type": "user", "timestamp": ts, "message": {"content": [
        {"type": "tool_result", "tool_use_id": tool_id,
         "content": body, "is_error": is_error}]}}


_VERDICT = json.dumps({"report": {"risks": [
    {"id": "R1", "severity": "critical", "claim": "eval() is RCE"},
    {"id": "R2", "severity": "low", "claim": "nit"}]}})


def test_a_gate_call_yields_gate_round_and_latency(tmp_path):
    p = _session(tmp_path, [
        _call("2026-07-27T08:10:31Z", "t1",
              "python3 ~/reviewer/gate.py review --gate 0 "
              "--thread cmag-demo-run --artifact PLAN.md --round 1"),
        _result("2026-07-27T08:11:59Z", "t1", _VERDICT)])
    (e,) = gate_call_events(p)
    assert e.kind == "md.escalate"        # a critical risk survives
    assert e.role == "md"
    assert e.payload["gate"] == "0"
    assert e.payload["round"] == 1
    assert e.payload["thread"] == "cmag-demo-run"
    assert e.payload["duration_sec"] == 88.0
    assert e.payload["blocking"] == [{"id": "R1", "severity": "critical",
                                      "claim": "eval() is RCE"}]


def test_two_calls_are_correlated_to_their_own_results(tmp_path):
    # Results returned out of order: a positional pairing swaps them.
    p = _session(tmp_path, [
        _call("2026-07-27T08:10:00Z", "t1",
              "gate.py review --gate a --thread cmag-x --round 1"),
        _call("2026-07-27T08:10:05Z", "t2",
              "gate.py review --gate b --thread cmag-x --round 3"),
        _result("2026-07-27T08:10:30Z", "t2", _VERDICT),
        _result("2026-07-27T08:11:00Z", "t1", _VERDICT)])
    by_gate = {e.payload["gate"]: e for e in gate_call_events(p)}
    assert by_gate["a"].payload["duration_sec"] == 60.0
    assert by_gate["b"].payload["duration_sec"] == 25.0
    assert by_gate["b"].payload["round"] == 3


def test_a_clean_review_passes(tmp_path):
    verdict = json.dumps({"report": {"risks": [
        {"id": "R1", "severity": "low", "claim": "nit"}]}})
    p = _session(tmp_path, [
        _call("2026-07-27T08:00:00Z", "t1",
              "gate.py review --gate a --thread cmag-x --round 1"),
        _result("2026-07-27T08:00:10Z", "t1", verdict)])
    (e,) = gate_call_events(p)
    assert e.kind == "md.review"
    assert e.payload["outcome"] == "pass"
    assert e.payload["blocking"] == []


def test_a_well_formed_zero_risk_body_is_a_genuine_pass(tmp_path):
    """The legitimate clean-pass case, pinned on its own so tightening
    `_classify`'s shape checks (below) cannot accidentally swallow it too:
    a genuinely well-shaped `report.risks: []` body is a real "reviewed,
    zero risks found" verdict, not a malformed one."""
    p = _session(tmp_path, [
        _call("2026-07-27T08:00:00Z", "t1",
              "gate.py review --gate a --thread cmag-x --round 1"),
        _result("2026-07-27T08:00:10Z", "t1",
                json.dumps({"report": {"risks": []}}))])
    (e,) = gate_call_events(p)
    assert e.payload["outcome"] == "pass"
    assert e.payload["blocking"] == []


def test_a_structurally_invalid_body_is_not_mistaken_for_a_pass(tmp_path):
    """`{}` or `{"error": "gate failed"}` -- well-formed JSON, is_error is
    False (gate.py exited cleanly), but neither carries a real
    `report.risks` list. Silently falling back to an empty risks list for
    a missing/wrong-shaped `report` is indistinguishable from "genuinely
    reviewed, zero risks found" -- `gptgates.py`'s own disk reader already
    treats a wrong-shaped body as malformed rather than as data
    (`if not isinstance(data, dict): ... malformed`); `_classify` must hold
    the transcript-sourced case to the same standard, not default it to a
    clean pass."""
    for body in ("{}", json.dumps({"error": "gate failed"})):
        p = _session(tmp_path, [
            _call("2026-07-27T08:00:00Z", "t1",
                  "gate.py review --gate a --thread cmag-x --round 1"),
            _result("2026-07-27T08:00:10Z", "t1", body)])
        (e,) = gate_call_events(p)
        assert e.payload["outcome"] is None, body
        assert e.payload["blocking"] == []


def test_an_unparseable_result_keeps_timing_and_drops_risks(tmp_path):
    p = _session(tmp_path, [
        _call("2026-07-27T08:00:00Z", "t1",
              "gate.py review --gate a --thread cmag-x --round 1"),
        _result("2026-07-27T08:00:10Z", "t1", "Traceback: boom")])
    (e,) = gate_call_events(p)
    assert e.payload["duration_sec"] == 10.0
    assert e.payload["outcome"] is None
    assert e.payload["blocking"] == []


def test_a_call_with_no_result_is_still_in_flight(tmp_path):
    p = _session(tmp_path, [
        _call("2026-07-27T08:00:00Z", "t1",
              "gate.py review --gate a --thread cmag-x --round 1")])
    (e,) = gate_call_events(p)
    assert e.payload["duration_sec"] is None


def test_a_bash_call_that_merely_mentions_gate_py_is_ignored(tmp_path):
    p = _session(tmp_path, [
        _call("2026-07-27T08:00:00Z", "t1",
              "ls ~/reviewer/gate.py && echo EXISTS")])
    assert gate_call_events(p) == []


def test_an_errored_call_is_not_reported_as_a_clean_pass(tmp_path):
    # A body that would otherwise classify as a clean pass (no report/risks
    # keys -> `_classify` sees no blocking risks -> "pass") must not survive
    # when the Bash call itself errored (gate.py crashed, timed out, ...).
    # Reporting a gate that never ran as a pass is a metrics-integrity bug,
    # not a degraded-but-honest reading -- unlike the unparseable-body case,
    # which correctly keeps outcome=None too, this is about not being
    # accidentally *right*-shaped JSON fooling `_classify`.
    p = _session(tmp_path, [
        _call("2026-07-27T08:00:00Z", "t1",
              "gate.py review --gate a --thread cmag-x --round 1"),
        _result("2026-07-27T08:00:10Z", "t1", json.dumps({"error": "boom"}),
                is_error=True)])
    (e,) = gate_call_events(p)
    assert e.payload["outcome"] is None
    assert e.payload["blocking"] == []
    assert e.payload["duration_sec"] == 10.0


def test_a_heredoc_body_prose_mentioning_review_is_not_a_gate_call(tmp_path):
    # `shlex.split` does not understand heredocs, so a `cat > f << 'EOF'`
    # command whose body happens to prose-mention "review" (dispatching a
    # Gate A review write-up, in the real corpus) tokenizes that word too.
    # Requiring the subcommand to immediately follow a `gate.py` token is
    # what tells this apart from a real invocation; a bare "does 'review'
    # appear anywhere in the tokens" match would not.
    p = _session(tmp_path, [
        _call("2026-07-27T08:00:00Z", "t1",
              "cat > /tmp/gate0-defenses-r1.json << 'EOF'\n"
              "## Step S2 approach for Gate A review\n"
              "Plan to reevaluate the renderer after Gate A responds.\n"
              "EOF")])
    assert gate_call_events(p) == []
