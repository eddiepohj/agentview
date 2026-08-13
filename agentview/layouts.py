# Copyright 2026 Edvard Pohjavirta
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for the full text.

"""Runner layouts and their tiers, declared as data.

Every runner-specific fact lives in this table and nowhere else. v1 spread the
same knowledge across three call sites in `discovery.py`; adding tiered-runner
meant editing all three, including getting "is the slug the state dir or its
parent" right by hand. Here that is one field with one fixture behind it.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TierSpec:
    role: str
    """A value from `events.ROLES`."""
    label: str
    """Display only. No code compares against this string."""
    declared_model: str | None = None
    """What `references/agent-taxonomy.md` says this tier runs on. Used only as
    the last fallback, and rendered `· declared` so a contract never reads as a
    measurement."""
    infer_prefixes: tuple[str, ...] = ()
    """Dispatch-description prefixes identifying this tier in runs made before
    the taxonomy landed, where every dispatch is `general-purpose`. Anything
    matched this way is marked `· inferred`."""


@dataclass(frozen=True)
class RunnerLayout:
    marker: str
    depth: str               # "marker" | "child" | "grandchild"
    state_dir_name: str
    slug_from: str           # "fixed-main" | "state-dir" | "parent-of-state"
    strip_prefix: str | None
    main_names: tuple[str, ...]
    build_log: str           # "suffixed" | "root-only"
    gpt_subpath: str | None
    gpt_nesting: str         # "flat" | "per-step"
    tiers: tuple[TierSpec, ...]
    gate_thread: str | None


_HUMAN = TierSpec("human", "human")
_WORKERS = TierSpec("worker", "workers")
_PLAIN_TIERS = (_HUMAN, TierSpec("step-runner", "orchestrator"), _WORKERS)
_TIERED_TIERS = (
    _HUMAN,
    TierSpec("fable", "fable", "claude-fable-5", ("Fable",)),
    TierSpec("md", "gpt md", "gpt-5.6-sol"),
    TierSpec("director", "director", "opus", ("Director",)),
    TierSpec("step-runner", "step-runner"),
    _WORKERS,
)

LAYOUTS: tuple[RunnerLayout, ...] = (
    RunnerLayout(
        marker="_planrunner", depth="child", state_dir_name="",
        slug_from="state-dir", strip_prefix="state-", main_names=("state",),
        build_log="suffixed", gpt_subpath="gates/gpt", gpt_nesting="flat",
        tiers=_PLAIN_TIERS, gate_thread=None,
    ),
    RunnerLayout(
        marker="_lightrunner", depth="marker", state_dir_name="",
        slug_from="fixed-main", strip_prefix=None, main_names=(),
        build_log="suffixed", gpt_subpath="gates/gpt", gpt_nesting="flat",
        tiers=_PLAIN_TIERS, gate_thread=None,
    ),
    RunnerLayout(
        # Gate B passes `--emit-dir _tieredrunner/<slug>/state/md --step <id>` on
        # every round, and gate.py's write_round drops
        # <emit_dir>/<step>/<gate>-r<n>.json -- hence "md", nested per step.
        # v1 recorded None here because no invocation then passed --emit-dir.
        marker="_tieredrunner", depth="grandchild", state_dir_name="state",
        slug_from="parent-of-state", strip_prefix=None, main_names=(),
        build_log="root-only", gpt_subpath="md", gpt_nesting="per-step",
        tiers=_TIERED_TIERS,
        gate_thread="cmag-{slug}",
    ),
    RunnerLayout(
        # Read compatibility for runs created before Max Runner was renamed.
        # MaxView never writes either layout, so retaining this marker cannot
        # create new legacy state.
        marker="_maxrunner", depth="grandchild", state_dir_name="state",
        slug_from="parent-of-state", strip_prefix=None, main_names=(),
        build_log="root-only", gpt_subpath="md", gpt_nesting="per-step",
        tiers=_TIERED_TIERS,
        gate_thread="cmag-{slug}",
    ),
)

BY_MARKER = {lay.marker: lay for lay in LAYOUTS}


def state_dirs(marker_dir, lay: RunnerLayout) -> list:
    """The candidate state directories under one marker directory."""
    if lay.depth == "marker":
        return [marker_dir]
    if lay.depth == "child":
        return [d for d in sorted(marker_dir.iterdir()) if d.is_dir()]
    return [child / lay.state_dir_name
            for child in sorted(marker_dir.iterdir()) if child.is_dir()]


def slug_for(state_dir, lay: RunnerLayout) -> str:
    """The run's slug, derived from its state dir under this layout."""
    if lay.slug_from == "fixed-main":
        return "main"
    if lay.slug_from == "parent-of-state":
        return state_dir.parent.name
    name = state_dir.name
    if name in lay.main_names:
        return "main"
    if lay.strip_prefix and name.startswith(lay.strip_prefix):
        return name[len(lay.strip_prefix):]
    return name
