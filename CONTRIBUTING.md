# Contributing

Please keep contributions read-only with respect to inspected run artifacts.

Before opening a change:

1. Create synthetic fixtures; do not add personal paths, private transcripts,
   generated reports, credentials, or local runner state.
2. Run `python -m pytest`.
3. Run a report against `examples/demo-project` with `--redact-paths`.
4. Check `git status` and confirm no generated output or local settings are
   staged.
