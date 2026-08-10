import pytest

from agentview.events import Event, ROLES, parse_iso
from agentview.metrics import model_family, models_by_role, tokens_by_model_family


class _Agent:
    def __init__(self, model, out, role="worker"):
        self.model, self.output_tokens, self.role = model, out, role


class _Run:
    def __init__(self, agents, orch_tokens, orch_models):
        self.agents = agents
        self.orchestrator_output_tokens = orch_tokens
        self.orchestrator_models = orch_models


def test_roles_cover_the_skills_eight_categories():
    for cat in ("director", "implementer", "observer", "verifier", "skeptic",
                "recon", "defender", "fable"):
        assert cat in ROLES, cat
    for extra in ("human", "step-runner", "md", "unknown", "worker"):
        assert extra in ROLES, extra
    assert len(ROLES) == 13, "closed vocabulary: eight categories plus five, no more"


def test_event_rejects_an_unknown_role():
    for bad_role in ("general-purpose", "nope"):
        with pytest.raises(ValueError):
            Event(ts=parse_iso("2026-07-20T10:00:00Z"), kind="run.start",
                  role=bad_role, step=None, payload={},
                  artifact_path=None, source="test")


def test_event_accepts_every_declared_role():
    for role in ROLES:
        Event(ts=parse_iso("2026-07-20T10:00:00Z"), kind="run.start",
              role=role, step=None, payload={}, artifact_path=None,
              source="test")


def test_model_family_reads_the_family_out_of_a_model_id():
    assert model_family("claude-haiku-4-5-20251001") == "haiku"
    assert model_family("claude-sonnet-4-6") == "sonnet"
    assert model_family("claude-opus-5") == "opus"
    assert model_family(None) == "unknown"
    # R7/R9: the fable branch, and a nonempty-but-unrecognized model id --
    # this is the direct replacement for the old, now-removed per-tier
    # helper test in test_metrics.py, which asserted exactly this
    # "gpt-4o" case.
    assert model_family("claude-fable-1") == "fable"
    assert model_family("gpt-4o") == "unknown"


def test_token_totals_key_on_model_family_and_are_unchanged():
    # Two families, three agents, different counts: an implementation that
    # assigned instead of accumulating, or grouped under one key, differs here.
    run = _Run([_Agent("claude-haiku-4-5-20251001", 1200),
                _Agent("claude-sonnet-4-6", 340),
                _Agent("claude-haiku-4-5-20251001", 60)],
               98_000, ["claude-opus-5"])
    assert tokens_by_model_family(run) == {
        "orchestrator": 98_000, "haiku": 1260, "sonnet": 340}


def test_models_by_role_groups_on_the_agents_role():
    # R5: a repeated model within one role (two haiku "worker" agents) proves
    # the per-role collection actually de-duplicates rather than just
    # collecting one entry per agent.
    run = _Run([_Agent("claude-haiku-4-5-20251001", 1),
                _Agent("claude-sonnet-4-6", 1),
                _Agent("claude-haiku-4-5-20251001", 1),
                _Agent("claude-opus-5", 1, role="director")],
               1, ["claude-opus-5"])
    got = models_by_role(run)
    assert got["worker"] == ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"]
    assert got["director"] == ["claude-opus-5"]
    assert got["step-runner"] == ["claude-opus-5"]
    # R10: every key `models_by_role` returns is a member of the same closed
    # vocabulary `Event` validates against -- no unlisted role leaks in.
    assert set(got) == set(ROLES)


def test_models_by_role_dedupes_and_sorts_the_step_runners_own_models():
    # `models_by_role`'s "step-runner" entry comes from `run.orchestrator_models`
    # (not `run.agents`), via `sorted(set(...))`. Every other test in this file
    # supplies that list with exactly one model, so a `sorted(set(...))` that
    # happened to degrade into a bare passthrough would still pass them. A
    # duplicate, unsorted list is the one fixture shape that tells `sorted` and
    # `set` apart from an identity function.
    run = _Run([], 0, ["claude-opus-5", "claude-haiku-4-5-20251001",
                       "claude-opus-5"])
    assert models_by_role(run)["step-runner"] == [
        "claude-haiku-4-5-20251001", "claude-opus-5"]


def test_models_by_role_is_empty_for_a_role_with_no_agents():
    run = _Run([], 0, [])
    for role in ("director", "fable", "md", "human"):
        assert models_by_role(run)[role] == [], role
