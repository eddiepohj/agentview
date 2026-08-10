"""What the reader is looking at. Pure: no I/O, no clock, no terminal.

Keeping the pane's state in one frozen value is what lets `render_frame` stay a
pure function under interactivity. Every syscall lives in `term.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class ViewState:
    selected: int = 0
    scroll: int = 0
    tray_open: bool = False
    help_open: bool = False
    expanded: bool = False
    quit: bool = False


_UP, _DOWN = ("k", "\x1b[A"), ("j", "\x1b[B")


def reduce(view: ViewState, key: str, run: Any) -> ViewState:
    """Apply one keypress. Returns a new ViewState; never mutates the input.
    An unrecognised key returns an equal state rather than raising: the reader
    is holding a keyboard, not calling an API."""
    last = max(0, len(getattr(run, "steps", []) or []) - 1)
    if key in _DOWN:
        return replace(view, selected=min(view.selected + 1, last))
    if key in _UP:
        return replace(view, selected=max(view.selected - 1, 0))
    if key == "d":
        return replace(view, tray_open=not view.tray_open)
    if key == "?":
        return replace(view, help_open=not view.help_open)
    if key in ("\r", "\n"):
        return replace(view, expanded=not view.expanded)
    if key == "q":
        return replace(view, quit=True)
    return view
