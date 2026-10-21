"""Tool gating.

Every tool call lands here. AskUserQuestion is routed to the human; everything
else is allowed unless it trips a deny rule. Auto-allowing without deny rules
would be exactly as permissive as bypassPermissions, which is the thing this
design is avoiding. The rules themselves live in `policy`; this module only
wires them, and the question route, into the SDK callback.

While a human answers a question, the live view (if any) must let go of the
terminal, or the prompt and the view draw over each other. That handover is the
hub's `suspended()`; with no hub, a `nullcontext` stands in so the prompt path
is the same either way.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager, ExitStack, nullcontext
from pathlib import Path
from typing import Any

import anyio
from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

from ..stream import StreamHub
from .policy import check_denied
from .questions import ask_human

PermissionResult = PermissionResultAllow | PermissionResultDeny


def make_callback(
    task_id: str,
    agent: str,
    roots: list[Path],
    lock: Any,
    allow_push: bool,
    on_question: Callable[[str, list[dict[str, Any]], dict[str, str]], None],
    on_deny: Callable[[str, str], None],
    hub: StreamHub | None = None,
) -> Callable[[str, dict[str, Any], Any], Awaitable[PermissionResult]]:
    """Build the per-task can_use_tool callback.

    `hub` defaults to None so callers that predate the live view keep working;
    they simply do not release the terminal around their questions.
    """

    async def can_use_tool(
        tool_name: str, input_data: dict[str, Any], context: Any
    ) -> PermissionResult:
        if tool_name == "AskUserQuestion":
            questions = input_data.get("questions") or []
            release: AbstractContextManager[None] = (
                hub.suspended() if hub is not None else nullcontext()
            )
            with ExitStack() as stack:
                # Entering can block until the viewer lets go of the terminal,
                # so it runs on a worker thread rather than stalling every other
                # task's stream on the event loop. Registering it on the stack
                # from that thread means the suspension is still ended if this
                # task is cancelled while waiting for the answer.
                #
                # The suspension is taken before ask_human's lock, not inside
                # it: a question queued behind another keeps the view released
                # across the gap, instead of letting it redraw for an instant
                # between two prompts.
                await anyio.to_thread.run_sync(stack.enter_context, release)
                answers = await ask_human(lock, task_id, agent, questions)
            on_question(task_id, questions, answers)
            # The tool's contract: echo the questions back alongside the answers.
            return PermissionResultAllow(updated_input={"questions": questions, "answers": answers})

        reason = check_denied(tool_name, input_data, roots, allow_push)
        if reason:
            on_deny(task_id, reason)
            return PermissionResultDeny(message=reason)

        return PermissionResultAllow(updated_input=input_data)

    return can_use_tool
