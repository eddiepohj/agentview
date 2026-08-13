from types import SimpleNamespace

from agentview.layouts import BY_MARKER
from agentview.model import assign_agent_roles

_TIERED = BY_MARKER["_maxrunner"]
_PLAN = BY_MARKER["_planrunner"]


def _a(desc, agent_type="general-purpose"):
    return SimpleNamespace(description=desc, agent_type=agent_type,
                           role="worker", role_source="observed")


def test_a_typed_agent_is_observed_not_inferred():
    agents = [_a("anything at all", agent_type="verifier")]
    assign_agent_roles(agents, _TIERED)
    assert agents[0].role == "verifier"
    assert agents[0].role_source == "observed"


def test_general_purpose_is_not_a_role():
    # It is the catch-all the taxonomy forbids; treating it as a role would
    # classify every agent in the existing corpus as a general-purpose tier.
    agents = [_a("Implement S7 quotd entrypoint")]
    assign_agent_roles(agents, _TIERED)
    assert agents[0].role == "worker"


def test_the_director_is_recognised_by_its_dispatch_description():
    agents = [_a("Director answers S3 technical gate"),
              _a("S2 adversarial skeptic"),
              _a("Fable adjudicates S4 eval() deadlock")]
    assign_agent_roles(agents, _TIERED)
    assert [a.role for a in agents] == ["director", "worker", "fable"]
    assert [a.role_source for a in agents] == ["inferred", "observed",
                                               "inferred"]


def test_a_prefix_matches_only_at_the_start():
    agents = [_a("Verify what the Director asked for")]
    assign_agent_roles(agents, _TIERED)
    assert agents[0].role == "worker"


def test_a_layout_without_prefixes_infers_nothing():
    agents = [_a("Director answers S3 technical gate")]
    assign_agent_roles(agents, _PLAN)
    assert agents[0].role == "worker"


def test_a_missing_description_is_not_an_error():
    agents = [_a(None)]
    assign_agent_roles(agents, _TIERED)
    assert agents[0].role == "worker"


def test_an_agent_type_outside_the_closed_taxonomy_is_not_observed():
    # Distinct from "general-purpose": this is a value that is truthy, is
    # not the catch-all, and still is not one of the skill's eight roles.
    # `declared in valid` must reject it; a mutation weakening that check to
    # `bool(declared)` would wrongly treat it as an observed role.
    agents = [_a("Implement something", agent_type="not-a-real-role")]
    assign_agent_roles(agents, _TIERED)
    assert agents[0].role == "worker"
    assert agents[0].role_source == "observed"


def test_a_valid_agent_type_wins_over_a_conflicting_description_prefix():
    # The resolution order is agentType -> description prefix, not the
    # reverse: an agent typed "verifier" but *described* like a director must
    # still resolve to the observed type, never the inferred prefix match.
    agents = [_a("Director answers S3 technical gate", agent_type="verifier")]
    assign_agent_roles(agents, _TIERED)
    assert agents[0].role == "verifier"
    assert agents[0].role_source == "observed"
