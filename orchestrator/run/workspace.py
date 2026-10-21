"""Deciding and provisioning the directory a task runs in.

Where a task runs is decided before its session exists, and everything after
depends on it: the session's cwd, the permission callback's write sandbox, and
the review that inspects the result. Doing it here keeps the git worktree
plumbing and the sandbox rules out of the runner, which only needs the answer.

`prepare` does not record anything on the run context. Registering a created
worktree for later cleanup is the caller's job, so this module stays free of
run-level state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..spec import Spec, Task
from . import worktree as wt
from .worktree import Worktree


class WorkspaceError(RuntimeError):
    """The task's workspace could not be prepared; the message is the failure reason.

    `cwd` and `worktree` carry whatever was resolved or created before the
    failure, so the caller can still report and register them.
    """

    def __init__(
        self, message: str, *, cwd: Path | None = None, worktree: Worktree | None = None
    ) -> None:
        super().__init__(message)
        self.cwd = cwd
        self.worktree = worktree


@dataclass(frozen=True)
class Workspace:
    cwd: Path
    roots: list[Path]  # the write sandbox
    worktree: Worktree | None


def prepare(task: Task, spec: Spec, run_id: str) -> Workspace:
    """Resolve the task's cwd, creating a worktree for it if it is isolated."""
    base_cwd = spec.resolve_cwd(task)
    created: Worktree | None = None
    try:
        if task.isolate == "worktree":
            created = wt.create(base_cwd, task.id, run_id)
            cwd = created.path
        else:
            cwd = base_cwd
    except wt.WorktreeError as exc:
        raise WorkspaceError(f"worktree setup failed: {exc}") from exc

    if not cwd.exists():
        raise WorkspaceError(f"cwd does not exist: {cwd}", cwd=cwd, worktree=created)

    roots = [
        cwd.resolve(),
        *spec.allowed_write_roots(),
    ]
    return Workspace(cwd=cwd, roots=roots, worktree=created)
