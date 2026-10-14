"""Tool gating.

Every tool call lands here. AskUserQuestion is routed to the human; everything
else is allowed unless it trips a deny rule. Auto-allowing without deny rules
would be exactly as permissive as bypassPermissions, which is the thing this
design is avoiding.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

from .questions import ask_human

PermissionResult = PermissionResultAllow | PermissionResultDeny

# Matched against the raw Bash command string.
DANGEROUS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bgit\s+push\b.*(--force|-f\b)"), "force push"),
    (re.compile(r"\bgit\s+reset\s+--hard\b"), "hard reset"),
    (re.compile(r"\bgit\s+clean\s+-[a-zA-Z]*f"), "destructive git clean"),
    (re.compile(r"\bsudo\b"), "sudo"),
    (re.compile(r"\b(mkfs|fdisk|shutdown|reboot|halt)\b"), "system-level command"),
    (re.compile(r"\bdd\s+.*\bof=/dev/"), "raw device write"),
    (re.compile(r"\bcurl\b[^|]*\|\s*(ba)?sh"), "pipe-to-shell"),
    (re.compile(r"\bwget\b[^|]*\|\s*(ba)?sh"), "pipe-to-shell"),
    (re.compile(r":\(\)\s*\{.*\};\s*:"), "fork bomb"),
]

RM_CALL = re.compile(r"\brm\b((?:\s+-[-\w]+)*)")


def _is_recursive_force_rm(cmd: str) -> bool:
    """True only when an `rm` carries both a recursive and a force flag.

    Parsing the flags beats pattern-matching the whole command: an earlier
    regex here matched any `rm` followed by a word ending in two of [rf], so
    `rm stuff` read as `rm -rf`, and unrelated commands that merely contained
    the text were blocked outright.
    """
    for flags in RM_CALL.findall(cmd):
        short = "".join(f.lstrip("-") for f in flags.split() if not f.startswith("--"))
        long_flags = {f.lstrip("-") for f in flags.split() if f.startswith("--")}
        recursive = "r" in short.lower() or {"recursive"} & long_flags
        force = "f" in short or {"force"} & long_flags
        if recursive and force:
            return True
    return False


PUSH = re.compile(r"\bgit\s+push\b")

# Redirect targets that are not real writes.
SAFE_REDIRECT_TARGETS = {
    "/dev/null",
    "/dev/stdout",
    "/dev/stderr",
    "/dev/tty",
}

# Tools whose file argument must stay inside an allowed root.
PATH_TOOLS = {
    "Write": "file_path",
    "Edit": "file_path",
    "NotebookEdit": "notebook_path",
}


def _inside(path: Path, roots: list[Path]) -> bool:
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        return False
    return any(resolved == r or r in resolved.parents for r in roots)


def check_denied(
    tool_name: str, input_data: dict[str, Any], roots: list[Path], allow_push: bool
) -> str | None:
    """Return a denial reason, or None to allow."""
    if tool_name == "Bash":
        cmd = str(input_data.get("command", ""))
        if _is_recursive_force_rm(cmd):
            return f"blocked by orchestrator deny rule (recursive force delete): {cmd[:200]}"
        for pattern, label in DANGEROUS:
            if pattern.search(cmd):
                return f"blocked by orchestrator deny rule ({label}): {cmd[:200]}"
        if PUSH.search(cmd) and not allow_push:
            return (
                "blocked by orchestrator deny rule (git push): pushes can trigger "
                "deploys. Set defaults.allow_push: true in the spec to permit it."
            )
        # A redirect into a path outside the sandbox is a write in disguise -
        # except the standard null/tty sinks, which write nowhere real and show
        # up in ordinary commands like `cmd > /dev/null 2>&1`.
        for target in re.findall(r">>?\s*([^\s;|&]+)", cmd):
            if target in SAFE_REDIRECT_TARGETS:
                continue
            if target.startswith("/") and not _inside(Path(target), roots):
                return f"blocked: shell redirect writes outside the task sandbox: {target}"
        return None

    field = PATH_TOOLS.get(tool_name)
    if field:
        raw = input_data.get(field)
        if raw and not _inside(Path(str(raw)), roots):
            allowed = ", ".join(str(r) for r in roots)
            return f"blocked: {raw} is outside the task sandbox ({allowed})"
    return None


def make_callback(
    task_id: str,
    agent: str,
    roots: list[Path],
    lock: Any,
    allow_push: bool,
    on_question: Callable[[str, list[dict[str, Any]], dict[str, str]], None],
    on_deny: Callable[[str, str], None],
) -> Callable[[str, dict[str, Any], Any], Awaitable[PermissionResult]]:
    """Build the per-task can_use_tool callback."""

    async def can_use_tool(
        tool_name: str, input_data: dict[str, Any], context: Any
    ) -> PermissionResult:
        if tool_name == "AskUserQuestion":
            questions = input_data.get("questions") or []
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
