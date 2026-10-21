"""Command line entrypoint: the parser, its usage checks, and dispatch.

Run with `python -m orchestrator <spec.yaml>`.

Running a spec is the primary path. Recorded sessions are browsed with a small
set of mutually exclusive flags (`--sessions`, `--session`, `--session-spec`,
`--session-task`) rather than subcommands: a subparser cannot coexist with the
bare `<spec.yaml>` positional without ambiguity (a spec file named `show`
would suddenly mean something else), and every existing invocation must keep
parsing exactly as before.

What each invocation does lives in `commands`; this module stops at a parsed,
checked `argparse.Namespace` and a loaded spec.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import sessions as store
from . import spec as spec_mod
from .commands import browse_sessions, run_spec

# Options that only make sense when running a spec. Combining them with a
# browse flag is almost certainly a mistake, so it is rejected, not ignored.
_RUN_ONLY = (
    "project_dir",
    "max_parallel",
    "dry_run",
    "verbose",
    "quiet",
    "json_report",
    "cleanup_worktrees",
    "live",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orchestrator",
        description="Run a DAG of Claude tasks in parallel, with clarifying questions intact.",
    )
    # Optional only so the browse flags can stand alone; a run still requires
    # it, checked after parsing with argparse's usual message.
    parser.add_argument("spec", type=Path, nargs="?", help="path to the YAML spec")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=None,
        help="override the spec's project_dir (the agents' place of work)",
    )
    parser.add_argument(
        "-j", "--max-parallel", type=int, default=4, help="max tasks running at once (default: 4)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="validate the spec and print the plan, then exit"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="stream assistant text from each task"
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress progress logs (the summary is still printed)",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=None,
        help="also write the run report as JSON to this path",
    )
    parser.add_argument(
        "--cleanup-worktrees",
        action="store_true",
        help="remove worktrees created by this run when it finishes",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="show a live tabbed view of every task's output while the run goes "
        "(needs a terminal; progress logs are printed when it closes)",
    )

    sessions = parser.add_argument_group(
        "recorded sessions",
        "Every run is recorded. These flags read the record instead of running a spec; "
        "RUN_ID may be any unique prefix of a run id.",
    )
    browse_flags = sessions.add_mutually_exclusive_group()
    browse_flags.add_argument(
        "--sessions", action="store_true", help="list recorded sessions, newest first"
    )
    browse_flags.add_argument("--session", metavar="RUN_ID", help="show one session's summary")
    browse_flags.add_argument(
        "--session-spec", metavar="RUN_ID", help="print the spec a session ran, verbatim"
    )
    browse_flags.add_argument(
        "--session-task",
        nargs=2,
        metavar=("RUN_ID", "TASK_ID"),
        help="print everything recorded for one task, output in full",
    )
    sessions.add_argument(
        "--session-dir",
        type=Path,
        default=None,
        help=f"where sessions are stored (default: ${store.ENV_VAR}, "
        "else ~/.local/share/claude-orchestrator/sessions); also used when recording a run",
    )
    return parser


def _check_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> bool:
    """Reject invalid combinations via `parser.error`. Returns True when browsing."""
    if args.live and args.quiet:
        parser.error("--live and --quiet cannot be used together: --live shows everything")

    browsing = (
        args.sessions
        or args.session is not None
        or args.session_spec is not None
        or args.session_task is not None
    )
    if browsing:
        if args.spec is not None:
            parser.error("a spec cannot be combined with the session browsing flags")
        clashing = [d for d in _RUN_ONLY if getattr(args, d) != parser.get_default(d)]
        if clashing:
            opts = ", ".join("--" + d.replace("_", "-") for d in clashing)
            parser.error(f"{opts}: only valid when running a spec, not with session browsing")
        return True

    if args.spec is None:
        parser.error("the following arguments are required: spec")

    if args.max_parallel < 1:
        parser.error("--max-parallel must be at least 1")
    return False


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if _check_args(parser, args):
        return browse_sessions(args)

    try:
        spec = spec_mod.load(args.spec, project_dir=args.project_dir)
    except FileNotFoundError:
        print(f"spec not found: {args.spec}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - surface validation errors readably
        print(f"invalid spec: {exc}", file=sys.stderr)
        return 2

    return run_spec(spec, args)


if __name__ == "__main__":
    raise SystemExit(main())
