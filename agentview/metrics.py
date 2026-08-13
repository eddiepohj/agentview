# Copyright 2026 Edvard Pohjavirta
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for the full text.

"""Metrics derived by the viewer, never self-reported by a skill."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from .events import ROLES


def model_family(model: str | None) -> str:
    """The model family a participant ran on -- a different axis from its role.
    A run has both, and one word meaning both is what v1 got wrong."""
    m = (model or "").lower()
    for needle in ("haiku", "sonnet", "opus", "fable"):
        if needle in m:
            return needle
    return "unknown"


def tokens_by_model_family(run: Any) -> dict[str, int]:
    """Output tokens by model family, orchestrator included. Renamed from
    the old per-tier token function; the arithmetic is unchanged."""
    out: Counter[str] = Counter()
    out["orchestrator"] = getattr(run, "orchestrator_output_tokens", 0)
    for a in run.agents:
        out[model_family(a.model)] += a.output_tokens or 0
    return dict(out)


def models_by_role(run: Any) -> dict[str, list[str]]:
    """Distinct models observed per role, sorted. Reads only the in-memory Run,
    so `render_frame` stays pure."""
    out: dict[str, list[str]] = {r: [] for r in ROLES}
    by_role: dict[str, set[str]] = {}
    for a in run.agents:
        if a.model:
            by_role.setdefault(getattr(a, "role", "worker"), set()).add(a.model)
    for role, models in by_role.items():
        out[role] = sorted(models)
    out["step-runner"] = sorted(set(getattr(run, "orchestrator_models", [])))
    return out


def fable_count(run: Any, threshold: int = 3) -> dict[str, Any]:
    """FM-9 circuit breaker. Reads the ledger's `fable_rulings`, which
    `ledger.py fable-record` writes.

    v1 counted `fable.rule` events -- a kind the event model defines and
    nothing emits -- so it reported 0 rulings for a run whose ledger recorded
    one, and the breaker could never trip.
    """
    rulings = getattr(run, "fable_rulings", []) or []
    return {"count": len(rulings), "threshold": threshold,
            "tripped": len(rulings) >= threshold, "rulings": rulings}


def gate_ratio(run: Any) -> dict[str, Any]:
    """FM-2 tell: human-answered / gates-opened. Both 0.0 and 1.0 are alarms.

    Reads per-step `gate_class` and `answered_by`. v1 counted `gate.open` /
    `gate.answer` events, neither of which is emitted, so it returned
    `ratio=0.0, alarm=True` for every run with an awaiting-human step.

    `opened == 0` gives `ratio=None`, not 0.0: a run that opened no gate has no
    ratio, and calling that zero would alarm on every gateless run.
    """
    gated = [s for s in run.steps if getattr(s, "gate_class", None)]
    answered = [s for s in gated if getattr(s, "answered_by", None)]
    by_human = [s for s in answered if s.answered_by == "human"]
    ratio = (len(by_human) / len(gated)) if gated else None
    return {"opened": len(gated), "answered": len(answered),
            "by_human": len(by_human), "ratio": ratio,
            "alarm": bool(gated) and ratio in (0.0, 1.0)}


def vocab_drift(run: Any) -> dict[str, Any]:
    statuses = Counter(
        e.payload.get("status") for e in run.events
        if e.kind == "anomaly"
        and e.payload.get("reason") == "off-vocabulary-status")
    return {"count": sum(statuses.values()), "statuses": dict(statuses)}


_AUTH = re.compile(r"^Authorised by:\s*(?P<who>.+)$", re.M)


def waiver_audit(project: Path) -> list[dict[str, Any]]:
    """FM-6 audit: waivers must name a human authoriser."""
    flagged = []
    for p in sorted(Path(project).rglob("changes/*.md")):
        text = p.read_text(errors="replace")
        if "Kind: waiver" not in text:
            continue
        m = _AUTH.search(text)
        if not m or not m.group("who").strip():
            flagged.append({"path": str(p), "reason": "no named authoriser"})
    return flagged


def self_report_drift(run: Any, rel: float = 0.10,
                      floor: int = 1000) -> dict[str, Any]:
    """The ledger's self-reported spend beside agentview's derived spend.

    This project began because plan-runner, asked to self-report its own
    observability, recorded `attempts` for 1 step in 25. max-runner has since
    made those fields mandatory at every close -- which does not make them
    true, it makes them checkable. Nothing else in the toolchain can check
    them: the ledger cannot audit itself, and a session cannot report on
    itself.

    Names the discrepancy; corrects neither side. An absent field is
    `unrecorded`, never folded into zero -- "the runner did not write this" and
    "the runner wrote 0" are different findings. The identical conflation on
    the *derived* side -- no turns found vs. turns summing to zero -- gets its
    own state, `no-attribution`, rather than reusing `or 0`. And `ratio` never
    silently equates a claimed-zero-vs-real-total mismatch with "nothing to
    compare": that case renders the string `"infinite"`, not `None`.
    """
    rows = []
    for st in run.steps:
        claimed = getattr(st, "ledger_tokens", None)
        # `turns` is only trusted as a real "nothing found" signal when the
        # object actually carries the field as an explicit (possibly empty)
        # list -- a minimal test double that omits it entirely is read as
        # "not applicable here", not as a false no-attribution finding.
        turns = getattr(st, "turns", None)
        no_attribution = turns is not None and len(turns) == 0
        derived = None if no_attribution else (st.output_tokens or 0)
        if claimed is None:
            state, divergent, delta, ratio = "unrecorded", False, None, None
        elif derived is None:
            state, divergent, delta, ratio = "no-attribution", False, None, None
        else:
            delta = derived - claimed
            # A claimed zero contradicted by any nonzero derived figure is a
            # total, material divergence -- unconditional, not subject to the
            # floor/rel thresholds below (those exist only for comparing two
            # nonzero magnitudes). Exact agreement (0 vs 0) stays non-
            # divergent.
            if claimed == 0 and derived == 0:
                divergent, ratio = False, 1.0
            elif claimed == 0:
                divergent, ratio = True, "infinite"
            else:
                biggest = max(abs(claimed), abs(derived))
                divergent = abs(delta) >= floor and abs(delta) >= rel * biggest
                ratio = derived / claimed
            state = "diverges" if divergent else "agrees"
        rows.append({"step": st.id, "ledger": claimed, "derived": derived,
                     "delta": delta, "state": state, "divergent": divergent,
                     "ratio": ratio})
    return {"steps": rows,
            "divergent": sum(1 for r in rows if r["divergent"]),
            "unrecorded": sum(1 for r in rows if r["state"] == "unrecorded"),
            "rel": rel, "floor": floor}


def summary(run: Any) -> dict[str, Any]:
    return {"tokens_by_model_family": tokens_by_model_family(run),
            "gates": gate_ratio(run),
            "fable": fable_count(run),
            "vocab_drift": vocab_drift(run),
            "waivers": waiver_audit(run.ref.project),
            "steps": len(run.steps),
            "agents": len(run.agents),
            "anomalies": len(run.anomalies)}
