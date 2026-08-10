"""Static single-file HTML report. Self-contained, no JS, no external assets."""
from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from ..events import ACCURACY_REASONS, Event, describe_anomaly
from ..metrics import summary
from ..model import cross_run_anomalies

_CSS = """body{font-family:ui-sans-serif,system-ui,sans-serif;margin:2rem;
line-height:1.5;color:#111}table{border-collapse:collapse;margin:1rem 0;
width:100%}th,td{border:1px solid #ddd;padding:.4rem .6rem;text-align:left;
font-size:14px}th{background:#f5f5f5;font-weight:500}h1{font-size:22px}
h2{font-size:18px;margin-top:2rem}code{font-family:ui-monospace,monospace}
.warn{color:#a33}
.alert{border:2px solid #a33;background:#fff4f4;padding:.8rem 1rem;
margin:1rem 0;border-radius:4px}
.alert .alert-title{margin:0;color:#a33;font-size:17px;font-weight:600}
.alert ul{margin:.4rem 0 0 1.1rem;padding:0}
.alert li{font-size:14px;margin:.2rem 0}"""


class ReadOnlyViolation(Exception):
    """Raised when an `--out` path would land inside a run's project.

    The spec's read-only guarantee is unconditional: agentview does not write
    into any run's project directory, ever. `write_report` refused nothing
    before this, and `--out <proj>/_planrunner/state/agentview-report.html`
    duly created a file inside the run's own state directory.
    """


def _refuse_writes_into_a_run(runs: list[Any], out: Path) -> Path:
    """Resolve `out` and refuse it if it lies inside any run's project root.

    Both sides are resolved before comparison, so `..` traversal and symlinks
    cannot smuggle a path past the check. Refusal is an exception, not a
    silent relocation: a report written somewhere the caller did not ask for
    is its own kind of dishonesty.
    """
    target = Path(out).resolve()
    for run in runs:
        project = Path(run.ref.project).resolve()
        if target.is_relative_to(project):
            raise ReadOnlyViolation(
                f"refusing to write {target}: it is inside the project "
                f"directory {project} of run '{run.ref.slug}'. agentview never "
                "writes into a run's project directory; choose an --out path "
                "outside it.")
    return target


def _alert_block(events: list[Event], qualify: bool = False) -> str:
    """A bordered, top-of-section warning box. `qualify` names the run each
    warning belongs to, for the report-wide block where context is absent."""
    if not events:
        return ""
    items = "".join(
        "<li>" + escape((f"{e.payload.get('slug')}: " if qualify else "")
                        + describe_anomaly(e)) + "</li>" for e in events)
    # Deliberately not an <h2>: that element is the report's per-run
    # landmark, and this box must not be mistaken for a run section.
    return ('<div class="alert"><p class="alert-title">these figures are '
            "not clean</p>"
            "<p>agentview surfaces the uncertainty rather than resolving it. "
            "The attribution below is unchanged; read it with these in "
            f"mind.</p><ul>{items}</ul></div>")


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape('' if c is None else str(c))}</td>"
                         for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _display_path(value: Any, project: Path, redact_paths: bool) -> str:
    """Return a display-safe path without exposing a machine's directory tree."""
    raw = str(value)
    if not redact_paths:
        return raw
    try:
        candidate = Path(raw)
        if candidate.is_absolute():
            try:
                return str(candidate.relative_to(project))
            except ValueError:
                return f"external/{candidate.name or 'path'}"
    except (OSError, ValueError):
        return "redacted-path"
    return raw


def _run_section(run: Any, cross: list[Event] | None = None,
                 redact_paths: bool = False) -> str:
    s = summary(run)
    project = Path(run.ref.project)
    project_name = project.name if redact_paths else str(project)
    parts = [f"<h2>{escape(project_name)} — {escape(run.ref.slug)}</h2>"]
    own = [a for a in run.anomalies
           if a.payload.get("reason") in ACCURACY_REASONS]
    mine = [e for e in (cross or [])
            if e.payload.get("state_dir") == str(run.ref.state_dir)]
    parts.append(_alert_block(mine + own))
    parts.append(_table(
        ["step", "status", "owner", "ledger tier", "attempts", "turns",
         "output tokens"],
        [[st.id, st.status, st.owner, st.ledger_tier, st.attempts,
          len(st.turns), st.output_tokens] for st in run.steps]))
    parts.append("<h3>tokens by model family</h3>")
    parts.append(_table(["model family", "output tokens"],
                        sorted(s["tokens_by_model_family"].items())))
    if s["vocab_drift"]["count"]:
        parts.append('<h3 class="warn">build-log vocabulary drift</h3>')
        parts.append(_table(["status", "count"],
                            sorted(s["vocab_drift"]["statuses"].items())))
    # `<code>` and a heading called "documents" assert "this is a file". Most
    # ledger `deliverable` values are prose, so they get their own section and
    # no code formatting -- listed in full, because they are real ledger
    # content, just not paths.
    if run.doc_paths:
        parts.append("<h3>documents</h3><ul>" + "".join(
            f"<li><code>{escape(_display_path(d, project, redact_paths))}</code></li>"
            for d in run.doc_paths) + "</ul>")
    if run.doc_notes:
        parts.append("<h3>ledger deliverables that are not paths</h3><ul>"
                     + "".join(f"<li>{escape(d)}</li>"
                               for d in run.doc_notes) + "</ul>")
    if s["waivers"]:
        parts.append('<h3 class="warn">waivers without a named authoriser</h3>')
        parts.append(_table(["path"], [[_display_path(w["path"], project,
                                                        redact_paths)]
                                      for w in s["waivers"]]))
    return "\n".join(parts)


def render(runs: list[Any], redact_paths: bool = False) -> str:
    # Overlap can only be seen with every run in hand, so it is computed
    # here, where the whole set is available, and never inside build_run.
    cross = cross_run_anomalies(runs)
    body = "\n".join(_run_section(r, cross, redact_paths) for r in runs)
    return ("<!doctype html>\n<html><head><meta charset=\"utf-8\">"
            "<title>agentview report</title>"
            f"<style>{_CSS}</style></head><body>"
            f"<h1>agentview report</h1>{_alert_block(cross, qualify=True)}"
            f"{body}</body></html>")


def write_report(runs: list[Any], out: Path, redact_paths: bool = False) -> Path:
    """Render `runs` to `out`. Refuses any `out` inside a run's project."""
    out = _refuse_writes_into_a_run(runs, out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(runs, redact_paths=redact_paths))
    return out
