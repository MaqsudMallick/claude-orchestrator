"""`TaskResult`: the outcome of one task, shared by the runner, the graph and the report."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
