"""Command line entrypoint: `python -m orchestrator <spec.yaml>`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anyio

from . import graph, report
from . import spec as spec_mod
from .models import MODEL_BY_DIFFICULTY
from .worktree import WorktreeError


def plan_text(spec: spec_mod.Spec) -> str:
    """Show the execution waves without spending anything."""
    from graphlib import TopologicalSorter

    sorter = TopologicalSorter({t.id: set(t.depends_on) for t in spec.tasks})
    sorter.prepare()
    by_id = spec.by_id

    lines = [f"{len(spec.tasks)} task(s) across these waves (tasks in a wave run in parallel):"]
    wave = 0
    while sorter.is_active():
        ready = sorter.get_ready()
        wave += 1
        lines.append(f"\n  wave {wave}:")
        for tid in ready:
            t = by_id[tid]
            d = spec.difficulty_for(t)
            lines.append(
                f"    {tid:<20} agent={t.agent:<16} difficulty={d:<6} "
                f"model={MODEL_BY_DIFFICULTY[d]}"
                + ("  [worktree]" if t.isolate == "worktree" else "")
            )
        sorter.done(*ready)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="orchestrator",
        description="Run a DAG of Claude tasks in parallel, with clarifying questions intact.",
    )
    parser.add_argument("spec", type=Path, help="path to the YAML spec")
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
    args = parser.parse_args(argv)

    if args.max_parallel < 1:
        parser.error("--max-parallel must be at least 1")

    try:
        spec = spec_mod.load(args.spec, project_dir=args.project_dir)
    except FileNotFoundError:
        print(f"spec not found: {args.spec}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - surface validation errors readably
        print(f"invalid spec: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(plan_text(spec))
        return 0

    results = anyio.run(graph.run, spec, args.max_parallel, args.verbose, args.quiet)

    print(report.render(spec, results))
    if args.json_report:
        report.write_json(args.json_report, spec, results)
        print(f"json report: {args.json_report}")

    if args.cleanup_worktrees:
        from . import worktree as wt

        # Each isolated task records its own worktree path, branch and main
        # repo, so cleanup needs nothing threaded out of the run context.
        for r in results.values():
            if r.branch and r.cwd and r.worktree_repo:
                try:
                    wt.remove(
                        wt.Worktree(path=Path(r.cwd), branch=r.branch, repo=Path(r.worktree_repo))
                    )
                    print(f"removed worktree {r.cwd}")
                except WorktreeError as exc:
                    print(f"could not remove worktree {r.cwd}: {exc}", file=sys.stderr)

    failed = [r for r in results.values() if r.status == "failed"]
    skipped = [r for r in results.values() if r.status == "skipped"]
    return 1 if (failed or skipped) else 0


if __name__ == "__main__":
    raise SystemExit(main())
