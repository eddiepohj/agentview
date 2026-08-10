"""The reviewer `gate.py` pinned for a run, read from its own state file.

`gate.py` resolves a reviewer once per thread and records it, so a run's early
and late turns are never judged by different models. The pin is written as a
byproduct of the gate working -- nothing was added to any skill to produce it.

Reading the pin rather than the default is load-bearing: DEFAULT_MODEL is
`gpt-5.6-sol`, but `cmag-demo-run` is pinned to `gpt-5.4-mini` at `medium`,
having been in flight when the reviewer changed. A pane showing the default
would name a reviewer that run never used.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..layouts import RunnerLayout

def reviewer_pin(layout: RunnerLayout, slug: str,
                 path: Path | None = None) -> dict[str, Any] | None:
    """The pinned reviewer for this run, or None.

    A pin is deliberately opt-in: reading an unrelated private directory by
    default makes a local viewer non-portable and surprises a fresh install.
    ``path`` therefore has to be supplied explicitly by the caller.
    """
    if not layout.gate_thread or path is None:
        return None
    try:
        pins = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(pins, dict):
        return None
    pin = pins.get(layout.gate_thread.format(slug=slug))
    if not isinstance(pin, dict) or not pin.get("model"):
        return None
    return {"model": pin["model"], "effort": pin.get("effort"),
            "since": pin.get("since")}
