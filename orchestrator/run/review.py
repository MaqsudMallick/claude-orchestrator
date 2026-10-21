"""Automatic review of completed work.

A reviewer is a separate session from the implementer - its own context, its own
persona - but deliberately the same model, so review quality tracks the
difficulty tier the task was given. It runs Claude Code's bundled `code-review`
skill.

When the reviewer asks for changes, the fix round *resumes the implementer's own
session* rather than starting a fresh one. The implementer already knows what it
built and why; re-explaining that through a new prompt loses the reasoning and
costs more.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, ResultMessage

from ..gate import make_callback
from ..spec import model_for

if TYPE_CHECKING:
    from ..spec import Spec, Task
    from .result import TaskResult

APPROVED = "approved"
CHANGES_REQUESTED = "changes_requested"
UNPARSED = "unparsed"
ERROR = "error"

# The reviewer ends with this marker so the verdict is machine-readable.
VERDICT_RE = re.compile(r"VERDICT:\s*(APPROVED|CHANGES_REQUESTED)", re.IGNORECASE)

REVIEWER_SYSTEM_PROMPT = """\
You are reviewing work another engineer just finished. You are a second pair of
eyes, not the author - you did not write this and have no stake in defending it.

Use the `code-review` skill to do the review.

You must not modify any file. Reviewing and fixing are separate jobs here; the
original engineer applies the fixes, because they hold the context for why the
code is shaped the way it is.

Judge the work against what the task actually asked for, which is quoted for you.
Report real defects: things that are wrong, broken, unsafe, or missing relative
to the request. Do not report style preferences or speculative improvements - a
reviewer that cries wolf costs a fix round for nothing.

End your reply with exactly one line, and nothing after it:

VERDICT: APPROVED              (if you found nothing worth fixing)
VERDICT: CHANGES_REQUESTED     (if you found defects that should be fixed now)
"""


@dataclass
class ReviewRound:
    round: int
    verdict: str
    findings: str = ""
    cost_usd: float = 0.0
    duration_s: float = 0.0
    fix_attempted: bool = False
    fix_ok: bool = False
    fix_cost_usd: float = 0.0
    error: str = ""

    @property
    def total_cost_usd(self) -> float:
        return self.cost_usd + self.fix_cost_usd


@dataclass
class ReviewOutcome:
    verdict: str = APPROVED
    rounds: list[ReviewRound] = field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        return sum(r.total_cost_usd for r in self.rounds)


def _is_git_repo(cwd: Path) -> bool:
    proc = subprocess.run(
        ["git", "-C", str(cwd), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def _review_prompt(task: Task, result: TaskResult, cwd: Path) -> str:
    scope = (
        "This directory is a git repository. Review the uncommitted changes "
        "(`git status`, `git diff`, and any untracked files the task created)."
        if _is_git_repo(cwd)
        else "This directory is not a git repository, so there is no diff. "
        "Review the files present that are relevant to the task below."
    )
    return f"""{scope}

The task the engineer was given:
<task>
{task.prompt.strip()}
</task>

What they reported doing:
<report>
{(result.text or "(no summary given)").strip()[:4000]}
</report>

Review their work against that task."""


def _parse_verdict(text: str) -> str:
    matches = VERDICT_RE.findall(text or "")
    if not matches:
        return UNPARSED
    return CHANGES_REQUESTED if matches[-1].upper() == "CHANGES_REQUESTED" else APPROVED


async def _run_session(options: ClaudeAgentOptions, prompt: str) -> ResultMessage | None:
    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for msg in client.receive_response():
            if isinstance(msg, ResultMessage):
                return msg
    return None


async def _review_once(
    task: Task, spec: Spec, result: TaskResult, cwd: Path, model: str, n: int, ctx: Any
) -> ReviewRound:
    started = time.monotonic()
    rnd = ReviewRound(round=n, verdict=ERROR)

    options = ClaudeAgentOptions(
        model=model,
        system_prompt=REVIEWER_SYSTEM_PROMPT,
        cwd=str(cwd),
        skills=["code-review"],
        setting_sources=["user", "project"],
        # Reviewing and fixing are separate roles; keep the reviewer read-only.
        disallowed_tools=["Write", "Edit", "NotebookEdit"],
        permission_mode="default",
        can_use_tool=make_callback(
            task_id=f"{task.id}:review{n}",
            agent=f"reviewer({model})",
            roots=[cwd.resolve()],
            lock=ctx.ask_lock,
            allow_push=False,
            on_question=lambda tid, qs, ans: None,
            on_deny=lambda tid, reason: result.denials.append(reason),
        ),
        max_turns=spec.max_turns_for(task) or 40,
    )

    msg = await _run_session(options, _review_prompt(task, result, cwd))
    rnd.duration_s = time.monotonic() - started
    if msg is None:
        rnd.error = "reviewer produced no result"
        return rnd

    rnd.cost_usd = msg.total_cost_usd or 0.0
    rnd.findings = msg.result or ""
    if msg.is_error:
        rnd.error = f"reviewer failed (stop_reason={msg.stop_reason})"
        return rnd

    rnd.verdict = _parse_verdict(rnd.findings)
    return rnd


async def _apply_fix(
    task: Task, spec: Spec, result: TaskResult, rnd: ReviewRound, model: str, cwd: Path, ctx: Any
) -> None:
    """Resume the implementer's session so it fixes with its original context."""
    if not result.session_id:
        rnd.error = "cannot fix: implementer session id not recorded"
        return

    agent = spec.agents[task.agent]
    tools = None
    if agent.tools is not None:
        tools = list(dict.fromkeys([*agent.tools, "AskUserQuestion"]))

    options = ClaudeAgentOptions(
        model=model,
        system_prompt=agent.system_prompt,
        tools=tools,
        cwd=str(cwd),
        resume=result.session_id,
        permission_mode="default",
        can_use_tool=make_callback(
            task_id=f"{task.id}:fix{rnd.round}",
            agent=task.agent,
            roots=[cwd.resolve()],
            lock=ctx.ask_lock,
            allow_push=spec.defaults.allow_push,
            on_question=lambda tid, qs, ans: result.questions.append(
                {"questions": qs, "answers": ans}
            ),
            on_deny=lambda tid, reason: result.denials.append(reason),
        ),
        max_turns=spec.max_turns_for(task),
    )

    prompt = (
        "A reviewer looked at the work you just finished and asked for changes.\n\n"
        f"<review>\n{rnd.findings.strip()}\n</review>\n\n"
        "Address the findings that are genuine defects. If you disagree with one, "
        "say so and explain why rather than changing the code to match it. "
        "Finish by summarising what you changed."
    )

    rnd.fix_attempted = True
    msg = await _run_session(options, prompt)
    if msg is None:
        rnd.error = "fix produced no result"
        return
    rnd.fix_cost_usd = msg.total_cost_usd or 0.0
    rnd.fix_ok = not msg.is_error
    if msg.is_error:
        rnd.error = f"fix failed (stop_reason={msg.stop_reason})"
    elif msg.result:
        # The fix supersedes the original summary for downstream consumers.
        result.text = msg.result
    if msg.session_id:
        result.session_id = msg.session_id


async def run_reviews(
    task: Task, spec: Spec, result: TaskResult, cwd: Path, ctx: Any
) -> ReviewOutcome:
    """Review/fix loop. Never raises: a broken reviewer must not fail the task."""
    cfg = spec.review_for(task)
    outcome = ReviewOutcome()
    if not cfg.enabled:
        return outcome

    model = model_for(spec.difficulty_for(task))

    for n in range(1, cfg.max_reviews + 1):
        try:
            rnd = await _review_once(task, spec, result, cwd, model, n, ctx)
        except Exception as exc:  # noqa: BLE001 - review must never kill the task
            rnd = ReviewRound(round=n, verdict=ERROR, error=f"{type(exc).__name__}: {exc}")

        outcome.rounds.append(rnd)
        ctx.log(
            f"[{task.id}] review {n}/{cfg.max_reviews}: {rnd.verdict}"
            f"  ${rnd.cost_usd:.4f}" + (f"  ({rnd.error})" if rnd.error else "")
        )

        if rnd.verdict in (APPROVED, UNPARSED, ERROR):
            # Only an explicit CHANGES_REQUESTED justifies spending a fix round.
            outcome.verdict = rnd.verdict
            return outcome

        outcome.verdict = CHANGES_REQUESTED
        if n == cfg.max_reviews:
            # Out of rounds; leave the verdict as changes_requested.
            return outcome

        try:
            await _apply_fix(task, spec, result, rnd, model, cwd, ctx)
        except Exception as exc:  # noqa: BLE001
            rnd.error = f"{type(exc).__name__}: {exc}"
        ctx.log(f"[{task.id}] fix {n}: {'ok' if rnd.fix_ok else 'failed'}  ${rnd.fix_cost_usd:.4f}")
        if not rnd.fix_ok:
            return outcome

    return outcome
