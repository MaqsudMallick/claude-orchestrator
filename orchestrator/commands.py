"""What each command does, once the command line has been parsed and checked.

`cli` owns argument parsing and the usage errors argparse reports; this module
owns the behaviour behind a valid invocation: running a spec (including the
live view and everything printed or written afterwards) and browsing recorded
sessions. Keeping the seam here means the behaviour can be driven from a plain
`argparse.Namespace` without going through `sys.argv`, and the parser can grow
flags without the run path being in the diff.
"""

from __future__ import annotations

import argparse
import functools
import io
import os
import sys
import threading
from pathlib import Path
from typing import TextIO

import anyio

from . import browse, graph, live, report
from . import session_store as store
from . import spec as spec_mod
from .models import MODEL_BY_DIFFICULTY
from .runner import TaskResult
from .spec import Spec
from .stream import StreamHub
from .worktree import WorktreeError

# Exit code for a lookup that found nothing (unknown or ambiguous run id,
# unknown task, unreadable record). Matches the "spec not found" exit so 1
# keeps its meaning of "the run happened but tasks failed".
EXIT_NOT_FOUND = 2

LIVE_FALLBACK = "--live needs an interactive terminal; falling back to normal output"


def plan_text(spec: spec_mod.Spec) -> str:
    """Show the execution waves without spending anything.

    The grouping comes from `Spec.waves()`, the same one the live view is
    given, so the plan printed here is the plan the run shows.
    """
    by_id = spec.by_id

    lines = [f"{len(spec.tasks)} task(s) across these waves (tasks in a wave run in parallel):"]
    for wave, ids in enumerate(spec.waves(), start=1):
        lines.append(f"\n  wave {wave}:")
        for tid in ids:
            t = by_id[tid]
            d = spec.difficulty_for(t)
            lines.append(
                f"    {tid:<20} agent={t.agent:<16} difficulty={d:<6} "
                f"model={MODEL_BY_DIFFICULTY[d]}"
                + ("  [worktree]" if t.isolate == "worktree" else "")
            )
    return "\n".join(lines)


def _write_stdout(text: str) -> None:
    """Write browse output, tolerating a reader that leaves early.

    Task output can be long and is naturally piped into `head` or `less`; when
    the reader exits, Python would otherwise print a BrokenPipeError traceback
    on top of a perfectly good result. Pointing stdout at devnull afterwards
    stops the interpreter's final flush from raising again.
    """
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except BrokenPipeError:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())


def _with_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


def browse_sessions(args: argparse.Namespace) -> int:
    """Handle the session browsing flags. Returns the exit code.

    Lookups go through `session_store`, which resolves unique run id prefixes;
    failures are rendered by `browse.render_error` with whatever context helps
    the user pick again (recent sessions, the matching ids, the session's task
    ids) and reported on stderr with EXIT_NOT_FOUND.
    """
    root = store.session_root(args.session_dir)

    if args.sessions:
        try:
            listing = store.list_sessions(root)
        except OSError as exc:
            print(f"could not read sessions in {root}: {exc}", file=sys.stderr)
            return EXIT_NOT_FOUND
        _write_stdout(_with_newline(browse.render_session_list(listing, root=root)))
        return 0

    run_id: str
    task_id: str | None = None
    if args.session_task is not None:
        run_id, task_id = args.session_task
    else:
        run_id = args.session if args.session is not None else args.session_spec

    session: store.Session | None = None
    try:
        session = store.load_session(run_id, root)
        if task_id is not None:
            text = _with_newline(browse.render_task(session, task_id))
        elif args.session_spec is not None:
            # Verbatim, with no newline added, so redirecting it to a file
            # reproduces the spec the run used.
            text = browse.render_spec(session)
        else:
            text = _with_newline(browse.render_session(session))
    except store.SessionStoreError as exc:
        # Listing is only needed to suggest alternatives for a bad run id; it
        # is skipped for task errors, where the session itself is the context.
        sessions = None if session is not None else _safe_list(root)
        print(browse.render_error(exc, sessions=sessions, session=session), file=sys.stderr)
        return EXIT_NOT_FOUND
    except OSError as exc:
        print(f"could not read session {run_id!r}: {exc}", file=sys.stderr)
        return EXIT_NOT_FOUND

    _write_stdout(text)
    return 0


def _safe_list(root: Path) -> list[store.SessionSummary] | None:
    """Sessions for error suggestions; None (no suggestions) if unreadable."""
    try:
        return store.list_sessions(root)
    except OSError:
        return None


class _HeldStderr(io.StringIO):
    """Stands in for stderr while the live view owns the screen.

    Progress logs (and `--verbose` text) go to stderr from the event loop and
    from worker threads; written straight to the terminal they would scribble
    over the curses screen. Held instead, they are replayed in order once the
    terminal is back, so a live run ends with the same log a plain run prints.
    After `release` writes pass straight through, which covers the case where
    the user closes the view with `q` and the run keeps going.
    """

    def __init__(self, real: TextIO) -> None:
        super().__init__()
        self._real = real
        self._held: list[str] = []
        self._released = False
        # Logs arrive from worker threads too (session recording).
        self._lock = threading.Lock()

    def write(self, s: str) -> int:
        with self._lock:
            if not self._released:
                self._held.append(s)
                return len(s)
        return self._real.write(s)

    def flush(self) -> None:
        with self._lock:
            if not self._released:
                return
        self._real.flush()

    def release(self) -> None:
        """Replay everything held, then pass writes through. Idempotent."""
        with self._lock:
            if self._released:
                return
            self._released = True
            held, self._held = self._held, []
        self._real.write("".join(held))
        self._real.flush()


async def _run_live(
    spec: spec_mod.Spec,
    max_parallel: int,
    verbose: bool,
    spec_path: Path | None,
    session_root: Path | None,
) -> dict[str, TaskResult]:
    """Run the DAG with the live viewer alongside it.

    The viewer is only an observer. If it fails, its error is caught here, so
    the task group does not cancel the run; the run carries on and the error is
    reported on stderr after the replayed logs. `q` only closes the view; the run continues with its
    logs going to stderr as usual (Ctrl-C still aborts). When the run ends, for
    any reason, the viewer is stopped and has restored the terminal before held
    logs are replayed and before control returns to print the summary.
    """
    hub = StreamHub()
    stop = anyio.Event()
    real_stderr = sys.stderr
    held = _HeldStderr(real_stderr)
    viewer_error: Exception | None = None

    async def viewer() -> None:
        nonlocal viewer_error
        try:
            await live.run_viewer(hub, stop)
        except Exception as exc:  # noqa: BLE001 - the view must never fail the run
            viewer_error = exc
        finally:
            held.release()

    results: dict[str, TaskResult] = {}
    sys.stderr = held
    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(viewer)
            try:
                # quiet=False: --live rejects --quiet, and the logs are held, not lost.
                results = await graph.run(
                    spec,
                    max_parallel,
                    verbose,
                    False,
                    spec_path=spec_path,
                    session_root=session_root,
                    hub=hub,
                )
            finally:
                stop.set()
    finally:
        held.release()
        sys.stderr = real_stderr
    if viewer_error is not None:
        print(
            f"live view stopped early: {type(viewer_error).__name__}: {viewer_error}",
            file=sys.stderr,
        )
    return results


def run_spec(spec: Spec, args: argparse.Namespace) -> int:
    """Run (or, with `--dry-run`, plan) a loaded spec. Returns the exit code."""
    if args.dry_run:
        print(plan_text(spec))
        return 0

    # anyio.run only forwards positionals, hence the partial. spec_path lets
    # the session record keep the spec byte for byte instead of a rebuild, but
    # only when the file is what actually ran: with --project-dir the verbatim
    # file names a different project, and `--session-spec` output fed back in
    # would silently run against it. The rebuild carries the resolved
    # project_dir, so it is the faithful record in that case.
    spec_path = args.spec if args.project_dir is None else None
    use_live = args.live
    if use_live and not live.supported():
        print(LIVE_FALLBACK, file=sys.stderr)
        use_live = False
    results: dict[str, TaskResult]
    if use_live:
        results = anyio.run(
            _run_live, spec, args.max_parallel, args.verbose, spec_path, args.session_dir
        )
    else:
        # No hub at all here: the plain path is exactly the pre-`--live` one.
        run = functools.partial(graph.run, spec_path=spec_path, session_root=args.session_dir)
        results = anyio.run(run, spec, args.max_parallel, args.verbose, args.quiet)

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
