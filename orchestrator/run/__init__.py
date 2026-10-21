"""Executing a run: the task DAG, each task's session, review, and its working directory.

`graph`, `runner`, `review`, `workspace` and `worktree` are part of the surface as
modules, as they were before the split; their public names are re-exported too.
"""

from . import graph, review, runner, workspace, worktree
from .graph import NodeFn, RunContext, RunState, build, make_node, merge_results, run
from .result import TaskResult
from .review import (
    APPROVED,
    CHANGES_REQUESTED,
    ERROR,
    REVIEWER_SYSTEM_PROMPT,
    UNPARSED,
    VERDICT_RE,
    ReviewOutcome,
    ReviewRound,
    run_reviews,
)
from .runner import build_options, build_prompt, execute_session, review_and_finalise, run_task
from .workspace import Workspace, WorkspaceError, prepare
from .worktree import Worktree, WorktreeError, create, remove, repo_root

__all__ = [
    "graph",
    "review",
    "runner",
    "workspace",
    "worktree",
    "TaskResult",
    "NodeFn",
    "RunContext",
    "RunState",
    "build",
    "make_node",
    "merge_results",
    "run",
    "APPROVED",
    "CHANGES_REQUESTED",
    "ERROR",
    "REVIEWER_SYSTEM_PROMPT",
    "UNPARSED",
    "VERDICT_RE",
    "ReviewOutcome",
    "ReviewRound",
    "run_reviews",
    "build_options",
    "build_prompt",
    "execute_session",
    "review_and_finalise",
    "run_task",
    "Workspace",
    "WorkspaceError",
    "prepare",
    "Worktree",
    "WorktreeError",
    "create",
    "remove",
    "repo_root",
]
