import dataclasses

import pytest

from agentview.surfaces.viewstate import ViewState, reduce


class _Run:
    def __init__(self, n):
        self.steps = [type("S", (), {"id": f"E.{i}"})() for i in range(n)]


def test_reduce_does_not_mutate_its_input():
    v = ViewState()
    out = reduce(v, "j", _Run(3))
    assert v.selected == 0 and out.selected == 1 and out is not v


def test_selection_clamps_at_the_last_step():
    # Five presses against four steps: modulo would wrap to 0 and pass a
    # three-press test.
    v = ViewState()
    for _ in range(5):
        v = reduce(v, "j", _Run(4))
    assert v.selected == 3


def test_selection_clamps_at_the_first_step():
    v = ViewState(selected=1)
    for _ in range(3):
        v = reduce(v, "k", _Run(4))
    assert v.selected == 0


def test_d_toggles_the_tray_both_ways():
    run = _Run(2)
    v = reduce(ViewState(), "d", run)
    assert v.tray_open is True
    assert reduce(v, "d", run).tray_open is False


def test_enter_expands_and_question_mark_helps():
    assert reduce(ViewState(), "\r", _Run(2)).expanded is True
    assert reduce(ViewState(), "?", _Run(1)).help_open is True


def test_an_unknown_key_changes_nothing():
    v = ViewState(selected=1, tray_open=True)
    assert reduce(v, "Z", _Run(4)) == v


def test_viewstate_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        ViewState().selected = 2
