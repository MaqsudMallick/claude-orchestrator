"""Spec loading and validation.

YAML in, a validated `Spec` out. Everything that can be caught before spending
a token is caught here: unknown agents, dangling dependencies, duplicate ids,
reserved ids and dependency cycles.
"""

from __future__ import annotations

from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import DIFFICULTIES

Difficulty = Literal["easy", "medium", "hard"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]

# LangGraph reserves these node names.
RESERVED_IDS = {"START", "END", "__start__", "__end__"}


class ReviewConfig(BaseModel):
    """Spec-level review settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    # Each round is a reviewer session plus, if it asks for changes, a fix
    # session - so this is the main lever on what review costs.
    max_reviews: int = Field(default=1, ge=1, le=10)


class ReviewOverride(BaseModel):
    """Per-task review settings. Unset fields fall back to defaults.review,
    so a task can change one field without silently resetting the others."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    max_reviews: int | None = Field(default=None, ge=1, le=10)


class Agent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_prompt: str
    description: str = ""
    # None means "every tool the CLI offers". A list restricts the toolset;
    # AskUserQuestion is re-added automatically so the agent can still ask.
    tools: list[str] | None = None


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    agent: str
    prompt: str
    # None means "fall back to defaults.difficulty".
    difficulty: Difficulty | None = None
    depends_on: list[str] = Field(default_factory=list)
    # Relative to the spec file's directory unless absolute.
    cwd: str | None = None
    isolate: Literal["none", "worktree"] = "none"
    effort: Effort | None = None
    max_turns: int | None = None
    max_budget_usd: float | None = None
    review: ReviewOverride | None = None

    @model_validator(mode="after")
    def _check(self) -> Task:
        if self.id in RESERVED_IDS:
            raise ValueError(f"task id {self.id!r} is reserved")
        if self.id in self.depends_on:
            raise ValueError(f"task {self.id!r} depends on itself")
        dupes = {d for d in self.depends_on if self.depends_on.count(d) > 1}
        if dupes:
            raise ValueError(f"task {self.id!r} repeats dependencies: {sorted(dupes)}")
        return self


class Defaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cwd: str = "."
    difficulty: Difficulty = "medium"
    effort: Effort | None = None
    max_turns: int | None = None
    # Upstream result text injected into a dependent's prompt is clipped here.
    max_context_chars: int = 4000
    # Extra directories tasks may write to, beyond their own cwd.
    allowed_write_dirs: list[str] = Field(default_factory=list)
    # `git push` is denied by default; flip only if you mean it.
    allow_push: bool = False
    review: ReviewConfig = Field(default_factory=ReviewConfig)


class Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    # The agents' place of work. Relative paths resolve against the spec file's
    # own directory; every task cwd then resolves under this.
    project_dir: str = "."
    defaults: Defaults = Field(default_factory=Defaults)
    agents: dict[str, Agent]
    tasks: Annotated[list[Task], Field(min_length=1)]

    # Set by load(); the directory relative paths resolve against.
    base_dir: Path = Field(default=Path("."), exclude=True)

    @model_validator(mode="after")
    def _check_graph(self) -> Spec:
        ids = [t.id for t in self.tasks]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate task ids: {dupes}")

        known = set(ids)
        for t in self.tasks:
            if t.agent not in self.agents:
                raise ValueError(
                    f"task {t.id!r} uses unknown agent {t.agent!r}; "
                    f"defined agents: {sorted(self.agents)}"
                )
            missing = [d for d in t.depends_on if d not in known]
            if missing:
                raise ValueError(f"task {t.id!r} depends on unknown task(s): {missing}")

        # Cycle check up front, so the failure names the cycle instead of
        # surfacing as a LangGraph runtime error.
        sorter = TopologicalSorter({t.id: set(t.depends_on) for t in self.tasks})
        try:
            sorter.prepare()
        except CycleError as exc:
            raise ValueError(f"dependency cycle: {' -> '.join(exc.args[1])}") from None
        return self

    @property
    def by_id(self) -> dict[str, Task]:
        return {t.id: t for t in self.tasks}

    @property
    def project_root(self) -> Path:
        """The directory the agents work in."""
        p = Path(self.project_dir).expanduser()
        return p.resolve() if p.is_absolute() else (self.base_dir / p).resolve()

    def resolve_cwd(self, task: Task) -> Path:
        """A task's working directory, resolved under the project root."""
        raw = task.cwd if task.cwd is not None else self.defaults.cwd
        p = Path(raw).expanduser()
        return p.resolve() if p.is_absolute() else (self.project_root / p).resolve()

    def allowed_write_roots(self) -> list[Path]:
        """Extra writable roots, also resolved under the project root."""
        out = []
        for d in self.defaults.allowed_write_dirs:
            p = Path(d).expanduser()
            out.append(p.resolve() if p.is_absolute() else (self.project_root / p).resolve())
        return out

    def difficulty_for(self, task: Task) -> Difficulty:
        return task.difficulty if task.difficulty is not None else self.defaults.difficulty

    def review_for(self, task: Task) -> ReviewConfig:
        base = self.defaults.review
        if task.review is None:
            return base
        return ReviewConfig(
            enabled=base.enabled if task.review.enabled is None else task.review.enabled,
            max_reviews=(
                base.max_reviews if task.review.max_reviews is None else task.review.max_reviews
            ),
        )

    def effort_for(self, task: Task) -> Effort | None:
        return task.effort if task.effort is not None else self.defaults.effort

    def max_turns_for(self, task: Task) -> int | None:
        return task.max_turns if task.max_turns is not None else self.defaults.max_turns


def load(path: str | Path, project_dir: str | Path | None = None) -> Spec:
    """Parse and validate a spec file. Accepts .yaml/.yml (JSON is valid YAML).

    `project_dir` overrides the spec's own setting, so one spec can be pointed
    at different checkouts without editing it.
    """
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{p}: expected a mapping at the top level, got {type(raw).__name__}")
    spec = Spec.model_validate(raw)
    spec.base_dir = p.parent
    if project_dir is not None:
        # An override from the command line is relative to the caller's cwd,
        # not to the spec file.
        spec.project_dir = str(Path(project_dir).expanduser().resolve())
    root = spec.project_root
    if not root.is_dir():
        # Caught here rather than at run time: every task would fail the same
        # way, and a typo in the project path should not cost a single token.
        raise ValueError(f"project_dir does not exist or is not a directory: {root}")
    return spec


__all__ = [
    "Agent",
    "Defaults",
    "ReviewConfig",
    "ReviewOverride",
    "Spec",
    "Task",
    "load",
    "DIFFICULTIES",
]
