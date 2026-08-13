from agentview.discovery import find_runs
from agentview.layouts import BY_MARKER, LAYOUTS, RunnerLayout, TierSpec


def _ledger(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"steps": []}')


def test_maxrunner_slug_is_the_parent_of_the_state_dir(tmp_path):
    # The direction already got wrong once: max-runner names the run one level
    # ABOVE the state dir, plan-runner names it AT the state dir.
    _ledger(tmp_path / "proj/_maxrunner/demo-run/state/ledger.json")
    _ledger(tmp_path / "proj/_maxrunner/soak-2/state/ledger.json")
    runs = find_runs(tmp_path)
    assert sorted(r.slug for r in runs) == ["demo-run", "soak-2"]
    assert all(r.state_dir.name == "state" for r in runs)


def test_planrunner_slug_is_the_state_dir_itself(tmp_path):
    _ledger(tmp_path / "proj/_planrunner/state/ledger.json")
    _ledger(tmp_path / "proj/_planrunner/state-format-v2/ledger.json")
    assert sorted(r.slug for r in find_runs(tmp_path)) == ["format-v2", "main"]


def test_lightrunner_state_dir_is_the_marker_itself(tmp_path):
    _ledger(tmp_path / "proj/_lightrunner/ledger.json")
    runs = find_runs(tmp_path)
    assert [r.slug for r in runs] == ["main"]
    assert runs[0].state_dir.name == "_lightrunner"


def test_colliding_slugs_fall_back_to_the_state_dir_name(tmp_path):
    _ledger(tmp_path / "proj/_planrunner/soak/ledger.json")
    _ledger(tmp_path / "proj/_planrunner/state-soak/ledger.json")
    assert sorted(r.slug for r in find_runs(tmp_path)) == ["soak", "state-soak"]


def test_maxrunner_takes_the_unsuffixed_build_log(tmp_path):
    _ledger(tmp_path / "proj/_maxrunner/demo-run/state/ledger.json")
    (tmp_path / "proj/BUILD-LOG.md").write_text("x")
    (tmp_path / "proj/BUILD-LOG-demo-run.md").write_text("wrong one")
    assert find_runs(tmp_path)[0].build_log.name == "BUILD-LOG.md"


def test_planrunner_prefers_the_suffixed_build_log(tmp_path):
    _ledger(tmp_path / "proj/_planrunner/state-soak/ledger.json")
    (tmp_path / "proj/BUILD-LOG.md").write_text("x")
    (tmp_path / "proj/BUILD-LOG-soak.md").write_text("right one")
    assert find_runs(tmp_path)[0].build_log.name == "BUILD-LOG-soak.md"


def test_maxrunner_gate_rounds_are_nested_per_step(tmp_path):
    # gate.py write_round drops <emit_dir>/<step>/<gate>-r<n>.json, and the
    # skill passes --emit-dir _maxrunner/<slug>/state/md --step <id>.
    _ledger(tmp_path / "proj/_maxrunner/demo-run/state/ledger.json")
    (tmp_path / "proj/_maxrunner/demo-run/state/md/S4").mkdir(parents=True)
    ref = find_runs(tmp_path)[0]
    assert ref.gpt_dir.name == "md"
    assert ref.layout.gpt_nesting == "per-step"


def test_planrunner_gate_rounds_are_flat(tmp_path):
    _ledger(tmp_path / "proj/_planrunner/state/ledger.json")
    (tmp_path / "proj/_planrunner/state/gates/gpt").mkdir(parents=True)
    ref = find_runs(tmp_path)[0]
    assert ref.gpt_dir.name == "gpt"
    assert ref.layout.gpt_nesting == "flat"


def test_tiered_layouts_declare_the_tiered_roles():
    by = {lay.marker: {t.role for t in lay.tiers} for lay in LAYOUTS}
    assert {"director", "fable", "md"} <= by["_maxrunner"]
    assert {"director", "fable", "md"} <= by["_tieredrunner"]
    assert not ({"director", "fable", "md"} & by["_planrunner"])
    assert not ({"director", "fable", "md"} & by["_lightrunner"])


def test_legacy_tieredrunner_runs_remain_readable(tmp_path):
    """`_tieredrunner` is the marker max-runner wrote while it was briefly
    published under that name. Runs made then must still be discoverable."""
    _ledger(tmp_path / "proj/_tieredrunner/legacy-run/state/ledger.json")
    runs = find_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0].slug == "legacy-run"
    assert runs[0].layout.marker == "_tieredrunner"


def test_the_legacy_marker_reports_the_runner_that_wrote_it(tmp_path):
    """A legacy marker names the same runner, so it must display the same
    name. Asserting on `display` and not on `marker` is the point: a reader
    looking at an old run should see "max-runner", not the retired spelling."""
    assert BY_MARKER["_tieredrunner"].display == BY_MARKER["_maxrunner"].display


def test_every_layout_declares_a_display_name(tmp_path):
    """Surfaces render `layout.display`. A layout added without one would
    render a blank runner name rather than fail, so pin it here."""
    for lay in LAYOUTS:
        assert lay.display, lay.marker


def test_every_layout_has_a_human_a_runner_and_workers():
    for lay in LAYOUTS:
        roles = {t.role for t in lay.tiers}
        assert {"human", "step-runner", "worker"} <= roles, lay.marker


def test_only_a_runner_with_an_md_tier_declares_a_gate_thread():
    for lay in LAYOUTS:
        assert bool(lay.gate_thread) is any(t.role == "md" for t in lay.tiers)
    assert BY_MARKER["_maxrunner"].gate_thread.format(slug="demo-run") == \
        "cmag-demo-run"


def test_declared_models_match_the_skill_taxonomy():
    tiers = {t.role: t for t in BY_MARKER["_maxrunner"].tiers}
    assert tiers["director"].declared_model == "opus"
    assert tiers["fable"].declared_model == "claude-fable-5"
    # Observed from the transcript, so no declared value to fall back to.
    assert tiers["step-runner"].declared_model is None
    assert tiers["worker"].declared_model is None


def test_a_layout_is_data_not_code():
    for lay in LAYOUTS:
        assert isinstance(lay, RunnerLayout)
        assert lay.depth in ("marker", "child", "grandchild")
        assert lay.slug_from in ("fixed-main", "state-dir", "parent-of-state")
        assert lay.build_log in ("suffixed", "root-only")
        assert lay.gpt_nesting in ("flat", "per-step")
        assert all(isinstance(t, TierSpec) for t in lay.tiers)
