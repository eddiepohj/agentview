# AgentView

AgentView is a read-only command-line viewer for agentic runs. It reconstructs
what happened from artifacts a runner already writes: a ledger, optional
session transcripts, build logs, and optional reviewer-round files. It does
not run agents, modify a discovered run, call a network service, or require a
server.

## Install

AgentView supports Python 3.10 or newer on macOS and Linux.

```bash
git clone https://github.com/eddiepohj/agentview.git
cd agentview
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
```

## Try the included synthetic example

```bash
agentview report examples/demo-project \
  --projects-root /tmp/empty-agentview-transcripts \
  --out /tmp/agentview-report.html \
  --redact-paths
```

The report is a single self-contained HTML file. `--redact-paths` is the safe
choice for any report that may be shared: it replaces absolute project and
document locations with relative names or a non-location label.

## Commands

```bash
# Static report for one or more project roots.
agentview report /path/to/project --out /tmp/agentview.html

# Live terminal view. Use --slug when a root has multiple runs.
agentview watch /path/to/project --slug main

# Re-render a completed run step by step.
agentview replay /path/to/project --slug main
```

By default AgentView looks for Claude Code transcript files under
`~/.claude/projects`. Override that location with `--projects-root`, or point
it at an empty directory for ledger-only reports. A reviewer model pin is
never read implicitly; supply it only when needed with `--pin-file PATH`.

## What AgentView reads

| Artifact | Required | Use |
|---|---:|---|
| Runner `ledger.json` | Yes | Discovers runs, steps, status, and declared deliverables. |
| Claude Code JSONL transcripts | No | Attributes turns, token output, and dispatched agents. |
| `BUILD-LOG*.md` | No | Surfaces status-vocabulary drift. |
| Reviewer-round JSON | No | Shows review rounds and escalations. |
| Reviewer pin JSON passed with `--pin-file` | No | Displays the model recorded for a reviewer thread. |

Supported layouts are `_planrunner`, `_lightrunner`, and `_tieredrunner`.
Legacy `_maxrunner` runs remain readable. A missing optional artifact produces a
degraded report rather than an error. Malformed external records are treated as
diagnostics where possible; AgentView does not modify them.

## Privacy and safety

- AgentView performs no network requests.
- It writes only the explicit `report --out` file, and refuses an output path
  inside a discovered run project.
- Reports can contain artifact text and paths. Review them before sharing and
  use `--redact-paths` for public output.
- Transcript access can expose sensitive local work. Use `--projects-root` to
  keep the scan scoped to data you intend to inspect.

## Development

```bash
python -m pytest
python -m agentview report examples/demo-project \
  --projects-root /tmp/empty-agentview-transcripts \
  --out /tmp/agentview.html --redact-paths
```

The default test suite uses synthetic fixtures only. Separately maintained
private-corpus checks, if present in a development checkout, are opt-in:

```bash
python -m pytest --run-corpus
```

## In the works

These roadmap items are exploratory and are not part of the current release contract:

- **A shared runner interoperability contract.** Read and validate the same versioned run
  manifest and normalized event stream emitted by Light Runner, Plan Runner, and Tiered
  Runner. This would reduce layout-specific discovery, expose reviewer provenance and
  schema drift explicitly, and make new adversarial-model adapters visible without
  AgentView-specific parsing changes.
- **Portable run-health feedback.** Add an explicit `agentview doctor --json` diagnostic
  export that a user can inspect and attach to a later Tiered Runner director sweep. The
  handoff would surface missing evidence, attribution gaps, and compatibility drift while
  preserving AgentView's read-only, non-orchestrating boundary.

## License

Licensed under the Apache License 2.0. See `LICENSE` and `NOTICE`.
