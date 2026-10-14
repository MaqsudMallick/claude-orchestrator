"""DAG execution on LangGraph.

Each task is a node; each dependency is an edge. LangGraph waits for all of a
node's inbound edges before running it and runs everything else in a superstep
concurrently, which gives the "independent tasks run in parallel" behaviour
without a hand-written scheduler. A semaphore caps real concurrency.

Failure policy: a task whose upstream failed is marked `skipped` and does not
run, while unrelated branches carry on to completion.
"""

from __future__ import annotations

import sys
import uuid
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Annotated, Any, Protocol, TypedDict

import anyio
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .runner import TaskResult, run_task
from .spec import Spec, Task
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

    def log(self, msg: str) -> None:
        if not self.quiet:
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
            return {
                "results": {
                    task.id: TaskResult(
                        task_id=task.id,
                        status="skipped",
                        difficulty=spec.difficulty_for(task),
                        reason=reason,
                    )
                }
            }

        async with ctx.sem:
            result = await run_task(task, spec, results, ctx)
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


async def run(spec: Spec, max_parallel: int, verbose: bool, quiet: bool) -> dict[str, TaskResult]:
    ctx = RunContext(
        run_id=uuid.uuid4().hex[:8],
        ask_lock=anyio.Lock(),
        sem=anyio.Semaphore(max_parallel),
        verbose=verbose,
        quiet=quiet,
    )
    ctx.log(f"run {ctx.run_id}: {len(spec.tasks)} task(s), max_parallel={max_parallel}")

    app = build(spec, ctx)
    final = await app.ainvoke(
        {"results": {}},
        config={"recursion_limit": max(25, len(spec.tasks) * 2 + 5)},
    )
    results: dict[str, TaskResult] = final["results"]

    if ctx.worktrees:
        ctx.log("\nworktrees created (left in place):")
        for w in ctx.worktrees:
            ctx.log(f"  {w.branch}  ->  {w.path}")

    return results
