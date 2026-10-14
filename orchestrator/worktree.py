"""Opt-in git worktree isolation.

A task with `isolate: worktree` runs in its own worktree + branch, so parallel
tasks can touch the same paths without fighting. Worktrees are deliberately
left on disk when the run ends: removing them would throw away the work.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


class WorktreeError(RuntimeError):
    pass


def _git(args: list[str], cwd: Path) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise WorktreeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def repo_root(path: Path) -> Path:
    return Path(_git(["rev-parse", "--show-toplevel"], path))


@dataclass
class Worktree:
    path: Path
    branch: str
    repo: Path


def create(source_cwd: Path, task_id: str, run_id: str) -> Worktree:
    """Create `<repo>/../.orchestrator-worktrees/<run_id>/<task_id>` on a new branch."""
    if not source_cwd.exists():
        raise WorktreeError(f"cwd does not exist: {source_cwd}")
    root = repo_root(source_cwd)

    base = root.parent / ".orchestrator-worktrees" / run_id
    base.mkdir(parents=True, exist_ok=True)
    target = base / task_id
    branch = f"orch/{run_id}/{task_id}"

    if target.exists():
        raise WorktreeError(f"worktree path already exists: {target}")
    _git(["worktree", "add", "-b", branch, str(target)], root)
    return Worktree(path=target, branch=branch, repo=root)


def remove(wt: Worktree) -> None:
    """Only called on --cleanup-worktrees."""
    _git(["worktree", "remove", "--force", str(wt.path)], wt.repo)
    _git(["branch", "-D", wt.branch], wt.repo)
