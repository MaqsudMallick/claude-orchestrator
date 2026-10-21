"""DAG execution on LangGraph.

Each task is a node; each dependency is an edge. LangGraph waits for all of a
node's inbound edges before running it and runs everything else in a superstep
concurrently, which gives the "independent tasks run in parallel" behaviour
without a hand-written scheduler. A semaphore caps real concurrency.

Failure policy: a task whose upstream failed is marked `skipped` and does not
run, while unrelated branches carry on to completion.

Every run is recorded through `session_store`: the session opens before the
first task starts, each task (skipped ones included) is appended the moment its
node finishes, and the session is finalised when the run ends, also when it
ends by an exception or Ctrl-C. Recording is an observer in the same sense as
review: it must never be able to fail the run it observes, so every store call
goes through `RunContext`, which logs failures and carries on.

A run can also feed a `StreamHub` for a live view (`RunContext.hub`). With no
hub, nothing here or in the runner touches one. With a hub, every task is
registered up front so the view lists them in spec order from the start; the
runner reports the tasks that run, and the skip is reported here, where it is
decided.
"""

from __future__ import annotations

import sys
import uuid
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Protocol, TypedDict

import anyio
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..sessions import store as session_store
from ..spec import Spec, Task, model_for
from ..stream import StreamHub
from .result import TaskResult
from .runner import run_task
from .worktree import Worktree


def merge_results(
    left: dict[str, TaskResult], right: dict[str, TaskResult]
) -> dict[str, TaskResult]:
    """Reducer: parallel branches each contribute their own task's result."""
    merged = dict(left)
    merged.update(right)
    return merged


class RunState(TypedDict):
    results: Annotated[dict[str, TaskResult], merge_results]


@dataclass
class RunContext:
    run_id: str
    ask_lock: Any
    sem: Any
    verbose: bool = False
    quiet: bool = False
    worktrees: list[Worktree] = field(default_factory=list)
    # None when recording is off for this run, or could not be started.
    session: session_store.SessionWriter | None = None
    # Why the session could not be started, for the end-of-run line.
    session_error: str = ""
    # Task ids whose record reached disk, so finalising can retry the rest.
    recorded: set[str] = field(default_factory=set)
    # Finished tasks still without a record after the retries at finalise.
    unrecorded: list[str] = field(default_factory=list)
    summary: session_store.SessionSummary | None = None
    # Live output for a viewer. None is the headless path: no publishing at all.
    hub: StreamHub | None = None

    def log(self, msg: str) -> None:
        if not self.quiet:
            print(msg, file=sys.stderr, flush=True)

    def _record_failed(self, what: str, exc: BaseException) -> str:
        detail = f"{type(exc).__name__}: {exc}"
        self.log(f"session {self.run_id}: could not record {what}: {detail}")
        return detail

    def open_session(self, spec: Spec, spec_path: Path | None, root: Path | str | None) -> None:
        """Start the session record. On failure the run goes ahead unrecorded."""
        try:
            self.session = session_store.begin_session(
                spec, self.run_id, spec_path=spec_path, root=root
            )
        except Exception as exc:  # noqa: BLE001 - recording must never kill the run
            self.session_error = self._record_failed("session start", exc)

    def record(self, result: TaskResult) -> None:
        """Append one task's record. Never raises: a full disk or a store bug
        costs the record, not the run. Called from a worker thread."""
        if self.session is None:
            return
        try:
            self.session.record_task(result)
        except Exception as exc:  # noqa: BLE001 - recording must never kill the run
            self._record_failed(f"task {result.task_id}", exc)
        else:
            self.recorded.add(result.task_id)

    def close_session(self, results: Mapping[str, TaskResult], error: str | None) -> None:
        """Retry any task whose record was lost, then write the `end` line.

        The retries are done one task at a time here rather than by passing
        `results` to `finalize`. With `finalize(results)`, one unrecordable task
        would raise before the `end` line was written and leave a run that
        actually finished looking interrupted."""
        if self.session is None:
            return
        for task_id, result in results.items():
            if task_id not in self.recorded:
                self.record(result)
        self.unrecorded = [t for t in results if t not in self.recorded]
        try:
            self.summary = self.session.finalize(error=error)
        except Exception as exc:  # noqa: BLE001 - recording must never kill the run
            self._record_failed("end of run", exc)

    def announce_session(self) -> None:
        """Say where the record is. Printed even with --quiet, because the
        summary is still printed under --quiet and this line is the way back
        into the run afterwards. Reports what is actually missing, not how many
        writes failed, since a failed write may have been recovered by a retry."""
        if self.session is None:
            msg = f"run {self.run_id}: not recorded ({self.session_error})"
        else:
            msg = f"run {self.run_id}: session recorded at {self.session.path}"
            gaps = []
            if self.unrecorded:
                gaps.append("no record for " + ", ".join(self.unrecorded))
            if self.summary is None:
                gaps.append("end of run not written")
            if gaps:
                msg += f"  (incomplete: {'; '.join(gaps)})"
        print(msg, file=sys.stderr, flush=True)


class NodeFn(Protocol):
    """A task node. Declared as a Protocol rather than a Callable alias so
    mypy can bind it to LangGraph's own node protocol at add_node()."""

    def __call__(self, state: RunState) -> Awaitable[dict[str, dict[str, TaskResult]]]: ...


def make_node(task: Task, spec: Spec, ctx: RunContext) -> NodeFn:
    async def node(state: RunState) -> dict[str, dict[str, TaskResult]]:
        results = state["results"]

        blockers = []
        for dep in task.depends_on:
            upstream = results.get(dep)
            if upstream is None:
                blockers.append(f"{dep} (no result)")
            elif not upstream.ok:
                blockers.append(f"{dep} ({upstream.status})")

        if blockers:
            reason = "upstream not ok: " + ", ".join(blockers)
            ctx.log(f"[{task.id}] skipped  {reason}")
            if ctx.hub is not None:
                ctx.hub.set_status(task.id, "skipped")
            result = TaskResult(
                task_id=task.id,
                status="skipped",
                difficulty=spec.difficulty_for(task),
                reason=reason,
            )
        else:
            async with ctx.sem:
                result = await run_task(task, spec, results, ctx)

        # Written now, not at the end, so an interrupted run keeps every task
        # that finished. The write fsyncs, so it runs off the event loop to
        # avoid stalling tasks that are still streaming. ctx.record never raises.
        await anyio.to_thread.run_sync(ctx.record, result)
        return {"results": {task.id: result}}

    return node


def build(spec: Spec, ctx: RunContext) -> CompiledStateGraph[RunState, None, RunState, RunState]:
    graph = StateGraph(RunState)
    for task in spec.tasks:
        graph.add_node(task.id, make_node(task, spec, ctx))

    has_dependents = {d for t in spec.tasks for d in t.depends_on}
    for task in spec.tasks:
        if task.depends_on:
            for dep in task.depends_on:
                graph.add_edge(dep, task.id)
        else:
            graph.add_edge(START, task.id)
        if task.id not in has_dependents:
            graph.add_edge(task.id, END)

    return graph.compile()


async def run(
    spec: Spec,
    max_parallel: int,
    verbose: bool,
    quiet: bool,
    *,
    spec_path: Path | None = None,
    run_id: str | None = None,
    session_root: Path | str | None = None,
    hub: StreamHub | None = None,
) -> dict[str, TaskResult]:
    """Execute the DAG and record it as a session.

    `spec_path` lets the record store the spec file byte for byte; without it
    the store falls back to a rebuild of the parsed spec. `run_id` lets a
    programmatic caller choose the id up front so it can load the session by
    that id afterwards (`session_store.load_session(run_id)`); it must match
    [A-Za-z0-9_]+ and be unused, or the run goes ahead unrecorded.
    `session_root` overrides where sessions are stored, as in session_store.
    `hub` receives live output for a viewer; leave it None for a headless run.
    """
    ctx = RunContext(
        run_id=run_id or uuid.uuid4().hex[:8],
        ask_lock=anyio.Lock(),
        sem=anyio.Semaphore(max_parallel),
        verbose=verbose,
        quiet=quiet,
        hub=hub,
    )
    if hub is not None:
        # The wave comes from the same grouping `--dry-run` prints, so the live
        # view and the plan agree on which tasks run together.
        wave_of = {tid: i for i, wave in enumerate(spec.waves()) for tid in wave}
        for task in spec.tasks:
            hub.register(
                task.id,
                task.agent,
                model_for(spec.difficulty_for(task)),
                wave=wave_of[task.id],
                depends_on=task.depends_on,
            )
    ctx.log(f"run {ctx.run_id}: {len(spec.tasks)} task(s), max_parallel={max_parallel}")
    ctx.open_session(spec, spec_path, session_root)

    results: dict[str, TaskResult] = {}
    error: str | None = None
    try:
        app = build(spec, ctx)
        final = await app.ainvoke(
            {"results": {}},
            config={"recursion_limit": max(25, len(spec.tasks) * 2 + 5)},
        )
        results = final["results"]
    except BaseException as exc:
        # Ctrl-C arrives as cancellation, whose message is usually empty.
        error = (
            f"{type(exc).__name__}: {exc}" if str(exc) else f"interrupted ({type(exc).__name__})"
        )
        raise
    finally:
        # Synchronous on purpose: this also runs while being cancelled, when
        # awaiting anything would be cancelled in turn. Tasks that finished
        # before an interruption are already on disk; this adds the end line.
        ctx.close_session(results, error)
        if ctx.worktrees:
            ctx.log("\nworktrees created (left in place):")
            for w in ctx.worktrees:
                ctx.log(f"  {w.branch}  ->  {w.path}")
        ctx.announce_session()

    return results
