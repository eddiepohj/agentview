import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentview.events import Event, parse_iso
from agentview.layouts import BY_MARKER


def pytest_addoption(parser):
    parser.addoption("--run-corpus", action="store_true", default=False,
                     help="run opt-in checks against a separately supplied corpus")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-corpus"):
        return
    skip = pytest.mark.skip(reason="private corpus checks require --run-corpus")
    for item in items:
        if "corpus" in item.keywords:
            item.add_marker(skip)


def _assistant(ts, out=100, cache_read=1000, model="claude-opus-4-8", tools=None):
    return {
        "type": "assistant", "timestamp": ts, "cwd": "/proj",
        "sessionId": "s1", "version": "2.1.169", "gitBranch": "main",
        "message": {
            "model": model,
            "usage": {"input_tokens": 2, "output_tokens": out,
                      "cache_creation_input_tokens": 10,
                      "cache_read_input_tokens": cache_read},
            "content": tools or [],
        },
    }


def _tool_result(tool_use_id, is_error=False):
    return {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": tool_use_id,
         "content": "ok", "is_error": is_error}]}}


@pytest.fixture
def write_jsonl(tmp_path):
    def _write(name, records):
        p = tmp_path / name
        p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        return p
    return _write


@pytest.fixture
def assistant():
    return _assistant


@pytest.fixture
def tool_result():
    return _tool_result


def _step(sid, status="complete", attempts=None, turns=(), out=0, agents=(),
          ledger_tokens=None, gate_class=None, answered_by=None):
    return SimpleNamespace(id=sid, status=status, owner="implementer",
                           ledger_tier="sonnet", attempts=attempts,
                           deliverable=None, depends_on=[], turns=list(turns),
                           output_tokens=out, agents=list(agents),
                           ledger_tokens=ledger_tokens, gate_class=gate_class,
                           answered_by=answered_by)


def _agent(agent_id, model, out=100, role="worker", role_source="observed"):
    return SimpleNamespace(agent_id=agent_id, agent_type="implementer",
                           model=model, description="work", parent_turn=1,
                           output_tokens=out, duration_sec=1.0, spawn_depth=1,
                           session=None, role=role, role_source=role_source)


@pytest.fixture
def fake_run():
    """A Run stand-in for pane tests. render_frame reads only these fields."""
    def _make(**over):
        layout = over.pop("layout", BY_MARKER["_tieredrunner"])
        ref = SimpleNamespace(
            project=Path("/proj"), slug="demo-run", layout=layout,
            state_dir=Path("/proj/_tieredrunner/demo-run/state"),
            ledger=Path("/proj/_tieredrunner/demo-run/state/ledger.json"),
            build_log=None, gpt_dir=over.pop("gpt_dir", None))
        base = dict(ref=ref, steps=[], agents=[], events=[], docs=[],
                    sessions=[], anomalies=[], orchestrator_output_tokens=0,
                    orchestrator_models=[], doc_paths=[], doc_notes=[],
                    dispatches=[], fable_rulings=[], reviewer_pin=None)
        base.update(over)
        return SimpleNamespace(**base)
    return _make


def _anomaly(reason, **payload):
    return Event(ts=None, kind="anomaly", role="unknown", step=None,
                 payload={"reason": reason, **payload}, artifact_path=None,
                 source="test")


def _md(kind, gate, rnd, total, blocking=()):
    return Event(ts=parse_iso(f"2026-07-20T1{rnd}:00:00Z"), kind=kind,
                 role="md", step=None,
                 payload={"gate": gate, "round": rnd, "rounds_total": total,
                          "blocking": list(blocking)},
                 artifact_path=None, source="test")


@pytest.fixture
def unbounded_run(fake_run):
    return fake_run(anomalies=[_anomaly("unbounded-span", span_start=None,
                                        sessions=1)])


@pytest.fixture
def run_with_long_doc_path(fake_run):
    p = ("/Users/example/projects/sample-project/_tieredrunner/demo-run/"
         "reports/deliverables/E4-outbound.md")
    assert len(p) > 90
    return fake_run(doc_paths=[p])


@pytest.fixture
def md_round_run(fake_run):
    # TWO rounds with different gates: a `rounds[0]` bug would pass one.
    return fake_run(events=[_md("md.review", "a", 1, 3),
                            _md("md.review", "b", 2, 3,
                                [{"severity": "high"}])])


@pytest.fixture
def md_round_chronology_run(fake_run):
    # Real demo-run shape: S4 reaches gate b, round 3 first; S7 (a later
    # step, chronologically later ts) only reaches gate b, round 1. A
    # `(round, gate)` selection wrongly picks S4's round 3 over S7's round 1.
    return fake_run(events=[
        Event(ts=parse_iso("2026-07-20T10:00:00Z"), kind="md.review",
              role="md", step="S4",
              payload={"gate": "b", "round": 3, "rounds_total": 3,
                       "blocking": []},
              artifact_path=None, source="test"),
        Event(ts=parse_iso("2026-07-20T14:00:00Z"), kind="md.review",
              role="md", step="S7",
              payload={"gate": "b", "round": 1, "rounds_total": 3,
                       "blocking": []},
              artifact_path=None, source="test"),
    ])


@pytest.fixture
def md_tied_round_review_first_run(fake_run):
    # `md.review` then `md.escalate` for the identical (ts, round, gate) --
    # the exact shape `gptgates.py` emits from one round file when a round
    # escalates. `md.escalate` must still win the tie.
    return fake_run(events=[_md("md.review", "b", 2, 3),
                            _md("md.escalate", "b", 2, 3)])


@pytest.fixture
def md_tied_round_escalate_first_run(fake_run):
    # Same tie, reversed input order -- the fix must not depend on which of
    # the pair happens to be appended first.
    return fake_run(events=[_md("md.escalate", "b", 2, 3),
                            _md("md.review", "b", 2, 3)])


@pytest.fixture
def empty_gpt_dir_run(fake_run, tmp_path):
    d = tmp_path / "md"
    d.mkdir()
    return fake_run(gpt_dir=d)


@pytest.fixture
def two_model_run(fake_run):
    return fake_run(agents=[_agent("a1", "claude-haiku-4-5-20251001", 1200),
                            _agent("a2", "claude-sonnet-4-6", 340)],
                    orchestrator_models=["claude-opus-5"],
                    orchestrator_output_tokens=98_000)


@pytest.fixture
def inferred_roles_run(fake_run):
    # TWO directors, so a row printing "1 dispatch" unconditionally fails.
    return fake_run(
        agents=[_agent("d1", "claude-opus-5", role="director",
                       role_source="inferred"),
                _agent("d2", "claude-opus-5", role="director",
                       role_source="inferred"),
                _agent("f1", "claude-fable-5", role="fable",
                       role_source="inferred"),
                _agent("w1", "claude-sonnet-4-6")],
        orchestrator_models=["claude-opus-5"],
        reviewer_pin={"model": "gpt-5.4-mini", "effort": "medium"},
        fable_rulings=[{"step": "S4", "ruling": "claude-correct"}])


@pytest.fixture
def pinned_md_run(two_model_run):
    two_model_run.reviewer_pin = {"model": "gpt-5.4-mini", "effort": "medium"}
    return two_model_run


@pytest.fixture
def planrunner_run(fake_run):
    return fake_run(layout=BY_MARKER["_planrunner"],
                    orchestrator_models=["claude-opus-5"])


@pytest.fixture
def undisplayed_role_run(fake_run):
    """The `_tieredrunner` layout has no dedicated row for `skeptic` --
    `_role_states`'s worker count already folds such an agent in via `role
    not in displayed`; `_model_cell`'s `else` branch must apply the same
    fold so the skeptic's model is not invisible in the row whose count
    includes it (R1)."""
    return fake_run(
        agents=[_agent("w1", "claude-sonnet-4-6"),
                _agent("s1", "claude-haiku-4-5-20251001", role="skeptic")])


@pytest.fixture
def fable_dispatched_no_ruling_run(fake_run):
    """A `fable`-role agent with no recorded ledger ruling yet -- unlike
    every other fable fixture in this file, `fable_rulings` is empty here.
    `run.agents` still proves the role was dispatched, so the row must not
    fall through to the tail loop's `unobservable` (R2)."""
    return fake_run(
        agents=[_agent("f1", "claude-fable-5", role="fable")],
        fable_rulings=[])


@pytest.fixture
def blind_run(fake_run):
    """A tier with no channel at all: no agents, no pin, no rulings, no gate
    calls. `unobservable` must still render here."""
    return fake_run(orchestrator_models=["claude-opus-5"])


@pytest.fixture
def prose_run(fake_run):
    return fake_run(doc_notes=["E.4 skeptic verdict (BUILD-LOG)",
                               "E.3 reply evidence"])


@pytest.fixture
def one_prose_run(fake_run):
    return fake_run(doc_notes=["E.4 skeptic verdict (BUILD-LOG)"])


@pytest.fixture
def clean_run(fake_run):
    # A genuinely clean *ledger*, not merely an absent one: S1's claimed and
    # derived token figures agree exactly, so `self_report_drift` returns a
    # non-empty `steps` list with `divergent`/`unrecorded` both 0. Without a
    # step here, `drift["steps"]` stays `[]` regardless of any bug in the
    # divergent/unrecorded gating, and Step 14's mutation note (render
    # whenever `drift["steps"]` is non-empty) could never be caught by this
    # fixture.
    return fake_run(doc_paths=["/proj/a.md"],
                    steps=[_step("S1", ledger_tokens=500, out=500,
                                 turns=[1])])


@pytest.fixture
def drift_run(fake_run):
    # S1 needs a non-empty `turns` so `self_report_drift` derives a real
    # figure for it (an explicit empty `turns=[]`, `_step`'s default, reads
    # as "no attribution found", not a comparable zero -- see
    # `agentview/metrics.py:124-129`). Without this, S1 can never register as
    # "diverges" no matter how far `out` is from `ledger_tokens`.
    return fake_run(steps=[_step("S1", ledger_tokens=41320, out=58004,
                                 turns=[1]),
                           _step("S2", ledger_tokens=None, out=900)])


@pytest.fixture
def two_step_run(fake_run):
    a = _agent("a1", "claude-haiku-4-5-20251001", 1200)
    return fake_run(
        steps=[_step("E.1", "in-progress", attempts=2, turns=[1, 2], out=900,
                     agents=[a], ledger_tokens=880),
               _step("E.2", "pending")],
        agents=[a], orchestrator_models=["claude-opus-5"])


@pytest.fixture
def many_docs_run(fake_run):
    # Twelve, so an unconditional [:8] cannot pass the open-tray assertion.
    return fake_run(doc_paths=[f"/proj/doc-{i}.md" for i in range(12)])


@pytest.fixture
def zero_output_step_run(fake_run):
    # `turns=[1]` is a real, non-empty attribution that happens to sum to
    # zero output tokens -- distinct from an empty `turns`, which means no
    # attribution was found at all (see `agentview/metrics.py:124-129`, the
    # same distinction `self_report_drift` draws for the ledger comparison).
    # `attempts=0` is an explicit, real value distinct from an unrecorded
    # (`None`) attempt count -- `_step_detail` must not fold either zero
    # into a truthiness default (R1/R2, Gate B round 1).
    return fake_run(steps=[_step("S1", attempts=0, turns=[1], out=0,
                                 ledger_tokens=0)])
