"""Running a single task as its own ClaudeSDKClient session.

Each task is a full session rather than a Task-tool subagent, because subagents
cannot reach AskUserQuestion. The agent's persona and toolset become session
options; the difficulty alias picks the model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)

from . import worktree as wt
from .models import model_for
from .permissions import make_callback
from .review import run_reviews
from .spec import Spec, Task


@dataclass
class TaskResult:
    task_id: str
    status: str  # ok | failed | skipped
    text: str = ""
    model: str = ""
    difficulty: str = ""
    cost_usd: float = 0.0
    duration_s: float = 0.0
    num_turns: int = 0
    stop_reason: str | None = None
    subtype: str | None = None
    reason: str = ""  # why it failed or was skipped
    cwd: str = ""
    branch: str | None = None
    worktree_repo: str | None = None
    session_id: str | None = None
    review_verdict: str = ""
    review_cost_usd: float = 0.0
    reviews: list[Any] = field(default_factory=list)
    questions: list[dict[str, Any]] = field(default_factory=list)
    denials: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def total_cost_usd(self) -> float:
        return self.cost_usd + self.review_cost_usd


def build_prompt(task: Task, spec: Spec, results: dict[str, TaskResult]) -> str:
    """Prepend upstream results so a dependent task can see what it builds on."""
    cap = spec.defaults.max_context_chars
    chunks: list[str] = []
    for dep in task.depends_on:
        r = results.get(dep)
        if r is None or not r.text:
            continue
        body = r.text
        if len(body) > cap:
            body = body[:cap] + f"\n...[truncated, {len(r.text) - cap} more chars]"
        chunks.append(f'<upstream_result task="{dep}">\n{body}\n</upstream_result>')

    if not chunks:
        return task.prompt
    return (
        "Results from the tasks this one depends on:\n\n"
        + "\n\n".join(chunks)
        + "\n\nYour task:\n"
        + task.prompt
    )


async def run_task(task: Task, spec: Spec, results: dict[str, TaskResult], ctx: Any) -> TaskResult:
    difficulty = spec.difficulty_for(task)
    model = model_for(difficulty)
    started = time.monotonic()

    result = TaskResult(task_id=task.id, status="failed", model=model, difficulty=difficulty)

    base_cwd = spec.resolve_cwd(task)
    created: wt.Worktree | None = None
    try:
        if task.isolate == "worktree":
            created = wt.create(base_cwd, task.id, ctx.run_id)
            ctx.worktrees.append(created)
            cwd = created.path
            result.branch = created.branch
            result.worktree_repo = str(created.repo)
        else:
            cwd = base_cwd
    except wt.WorktreeError as exc:
        result.reason = f"worktree setup failed: {exc}"
        result.duration_s = time.monotonic() - started
        return result

    result.cwd = str(cwd)
    if not cwd.exists():
        result.reason = f"cwd does not exist: {cwd}"
        result.duration_s = time.monotonic() - started
        return result

    roots = [
        cwd.resolve(),
        *spec.allowed_write_roots(),
    ]

    agent = spec.agents[task.agent]
    tools = None
    if agent.tools is not None:
        # Restricting the toolset hides AskUserQuestion unless it is re-added,
        # which would silently disable clarifying questions for this agent.
        tools = list(dict.fromkeys([*agent.tools, "AskUserQuestion"]))

    options = ClaudeAgentOptions(
        model=model,
        system_prompt=agent.system_prompt,
        tools=tools,
        cwd=str(cwd),
        permission_mode="default",
        can_use_tool=make_callback(
            task_id=task.id,
            agent=task.agent,
            roots=roots,
            lock=ctx.ask_lock,
            allow_push=spec.defaults.allow_push,
            on_question=lambda tid, qs, ans: result.questions.append(
                {"questions": qs, "answers": ans}
            ),
            on_deny=lambda tid, reason: result.denials.append(reason),
        ),
        max_turns=spec.max_turns_for(task),
        effort=spec.effort_for(task),
        max_budget_usd=task.max_budget_usd,
    )

    prompt = build_prompt(task, spec, results)
    ctx.log(
        f"[{task.id}] start  agent={task.agent} difficulty={difficulty} model={model} cwd={cwd}"
    )

    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage) and ctx.verbose:
                    for block in msg.content:
                        if isinstance(block, TextBlock) and block.text.strip():
                            ctx.log(f"[{task.id}] {block.text.strip()[:300]}")
                if isinstance(msg, ResultMessage):
                    result.text = msg.result or ""
                    result.cost_usd = msg.total_cost_usd or 0.0
                    result.num_turns = msg.num_turns
                    result.stop_reason = msg.stop_reason
                    result.session_id = msg.session_id
                    result.subtype = msg.subtype
                    if msg.is_error:
                        # is_error also covers non-crash cutoffs such as
                        # max_turns, so carry the detail through to the report.
                        result.status = "failed"
                        detail = ", ".join(msg.errors or []) or msg.subtype or "unknown"
                        result.reason = f"{detail} (stop_reason={msg.stop_reason})"
                    else:
                        result.status = "ok"
    except Exception as exc:  # noqa: BLE001 - one task must not kill the run
        result.status = "failed"
        result.reason = f"{type(exc).__name__}: {exc}"

    ctx.log(
        f"[{task.id}] {result.status}  {time.monotonic() - started:.1f}s  "
        f"${result.cost_usd:.4f}" + (f"  ({result.reason})" if result.reason else "")
    )

    if result.ok:
        outcome = await run_reviews(task, spec, result, cwd, ctx)
        result.reviews = outcome.rounds
        result.review_cost_usd = outcome.cost_usd
        result.review_verdict = outcome.verdict if outcome.rounds else ""
        # Running out of review rounds is not a failure: the work is accepted
        # as-is and the graph moves on. The verdict is kept for the report so
        # the outstanding findings are still visible.

    result.duration_s = time.monotonic() - started
    return result
