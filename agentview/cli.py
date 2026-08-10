"""agentview command line. Read-only: writes only to an explicit --out path."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .discovery import find_runs, run_sessions, sessions_for_runs
from .model import build_run
from .paths import projects_root
from .surfaces import live, report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agentview")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("report", help="render a static HTML report")
    r.add_argument("roots", nargs="+")
    r.add_argument("--out", required=True)
    r.add_argument("--projects-root", default=None)
    r.add_argument("--pin-file", default=None,
                   help="optional reviewer pin JSON; never read implicitly")
    r.add_argument("--redact-paths", action="store_true",
                   help="hide absolute paths in the generated report")

    w = sub.add_parser("watch", help="live terminal pane")
    w.add_argument("root")
    w.add_argument("--slug", default=None)
    w.add_argument("--interval", type=float, default=1.0)
    w.add_argument("--iterations", type=int, default=None)
    w.add_argument("--projects-root", default=None)
    w.add_argument("--pin-file", default=None,
                   help="optional reviewer pin JSON; never read implicitly")

    rp = sub.add_parser("replay", help="redraw a finished run frame by frame")
    rp.add_argument("root")
    rp.add_argument("--slug", default=None)
    rp.add_argument("--projects-root", default=None)
    rp.add_argument("--pin-file", default=None,
                    help="optional reviewer pin JSON; never read implicitly")

    args = p.parse_args(argv)
    proot = Path(args.projects_root) if args.projects_root else projects_root()
    pin_path = Path(args.pin_file) if args.pin_file else None

    if args.cmd == "report":
        refs = [ref for root in args.roots for ref in find_runs(Path(root))]
        if not refs:
            print("no runs found", file=sys.stderr)
            return 1
        # One scan of the transcript corpus for every run in the report.
        by_run = sessions_for_runs(refs, proot)
        runs = [build_run(ref, by_run[ref.state_dir], pin_path=pin_path)
                for ref in refs]
        try:
            out = report.write_report(runs, Path(args.out),
                                      redact_paths=args.redact_paths)
        except report.ReadOnlyViolation as exc:
            print(f"agentview: {exc}", file=sys.stderr)
            return 2
        print(out)
        return 0

    all_refs = find_runs(Path(args.root))
    refs = [x for x in all_refs if x.slug == args.slug] if args.slug else all_refs
    if not refs:
        print("no runs found", file=sys.stderr)
        return 1
    ref = refs[0]

    if args.cmd == "watch":
        # Sessions are resolved *inside* the loop, so one started while the
        # pane is open is picked up; siblings let the pane warn about a
        # sibling run whose span overlaps this one's.
        live.watch(ref, interval=args.interval, iterations=args.iterations,
                   projects_root=proot, siblings=all_refs, pin_path=pin_path)
        return 0

    live.replay(build_run(ref, run_sessions(ref, proot), pin_path=pin_path))
    return 0
