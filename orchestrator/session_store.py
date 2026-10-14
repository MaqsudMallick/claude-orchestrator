"""Durable, append-only records of orchestrator runs.

A run should leave enough behind that someone can come back a week later and
see what it did: the spec it ran, each task's prompt and full output, the
questions a human answered, review verdicts, denied tool calls and what it all
cost.

On-disk layout, one directory per run:

    <root>/<YYYYmmddTHHMMSSZ>-<run_id>/
        spec.yaml      the spec exactly as supplied, byte for byte
        events.jsonl   one JSON object per line: a `begin` record, then one
                       `task` record per task as it completes, then `end`

Why this shape:

- **Append-only JSON Lines.** Each task is written (and fsynced) the moment it
  completes, so a run that crashes or is interrupted still leaves every task it
  finished plus the plan of what it intended. A record without an `end` line
  reads back as an unfinished session rather than as nothing. A write torn by a
  crash can only damage the last line, and a torn line fails to parse instead of
  silently reading as a shorter string, which is why this is JSON and not a
  YAML stream.
- **The spec is its own file, not a field.** Storing it raw means no escaping
  ever touches it: anything a YAML file can contain survives exactly, and
  `cat spec.yaml` shows what ran. Its sha256 is kept in the `begin` record.
- **Awkward characters.** Everything else is JSON-escaped (quotes, newlines,
  control characters). Non-ASCII is written as-is so the file stays readable;
  lone surrogates, the one thing UTF-8 cannot encode, are written as JSON
  `\\uXXXX` escapes. Readers split lines on `\\n` only, never on Unicode line
  separators that may legitimately appear inside output text.
- **Plain text, readable without this module:** `jq . events.jsonl`, or
  `jq -r 'select(.kind=="task") | .text'` for task outputs.

Policy, as agreed with the user:

- Records live in a user-level directory (see `session_root`), not the project:
  they survive `git clean` and cover runs from every project. Override with the
  ORCHESTRATOR_SESSION_DIR environment variable or an explicit `root=` argument;
  the spec has no field for this.
- Nothing is truncated: prompts, outputs, answers and review findings are kept
  in full, because a reader cannot recover what was cut.
- Nothing is redacted. Instead the session directories are created 0700 and
  files 0600, since a record is as sensitive as the terminal scrollback of the
  run that produced it.

Nothing here swallows its own errors; callers that must not fail because of
recording (the run loop) are expected to catch `OSError` and
`SessionStoreError` themselves.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import re
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from .models import model_for

if TYPE_CHECKING:
    # Type-only: the run loop imports this module, and runner -> session_store
    # -> runner would otherwise be an import cycle.
    from .runner import TaskResult
    from .spec import Spec

FORMAT_VERSION = 1
ENV_VAR = "ORCHESTRATOR_SESSION_DIR"
SPEC_FILENAME = "spec.yaml"
EVENTS_FILENAME = "events.jsonl"

# Run ids become directory names, so they are restricted to a filename-safe set.
# Always used with fullmatch: `^...$` with match() would accept "abc\n".
_RUN_ID_RE = re.compile(r"[A-Za-z0-9_]+")
_DIR_TS_FORMAT = "%Y%m%dT%H%M%SZ"


# --------------------------------------------------------------------- errors


class SessionStoreError(Exception):
    """Base class for every error this module raises on purpose."""


class SessionNotFoundError(SessionStoreError, LookupError):
    """No session matches the given run id or prefix."""


class AmbiguousSessionError(SessionStoreError, LookupError):
    """A run id prefix matches more than one session."""

    def __init__(self, prefix: str, matches: list[str]) -> None:
        self.prefix = prefix
        self.matches = matches
        super().__init__(
            f"run id prefix {prefix!r} is ambiguous; matches: {', '.join(sorted(matches))}"
        )


class TaskNotFoundError(SessionStoreError, LookupError):
    """The session has no record for that task id.

    `planned` distinguishes a typo (False) from a task that was in the spec but
    never finished, e.g. because the run was interrupted (True).
    """

    def __init__(self, run_id: str, task_id: str, planned: bool) -> None:
        self.run_id = run_id
        self.task_id = task_id
        self.planned = planned
        why = (
            "was planned but has no record (the run was interrupted before it finished)"
            if planned
            else "is not a task in this session"
        )
        super().__init__(f"task {task_id!r} {why} (run {run_id})")


# --------------------------------------------------------- records (on disk)


class _Record(BaseModel):
    # Unknown fields are ignored rather than rejected, so an older reader can
    # still open a record written by a newer version that added fields.
    model_config = ConfigDict(extra="ignore")


class TaskPlan(_Record):
    """A task as the spec declared it, captured at the start of the run.

    Present for every task, including ones that never ran, so an interrupted
    session still shows what it set out to do.
    """

    id: str
    agent: str
    difficulty: str
    model: str
    depends_on: list[str] = Field(default_factory=list)
    prompt: str = ""
    isolate: str = "none"


class SessionHeader(_Record):
    """The `begin` line: written once, before any task runs."""

    kind: Literal["begin"] = "begin"
    format: int = FORMAT_VERSION
    run_id: str
    started_at: datetime
    # Filename of the spec as run, e.g. "session-tracking.yaml".
    spec_name: str
    # Absolute path the spec was read from, if known. It may since have changed
    # or been deleted; spec.yaml in the session directory is the durable copy.
    spec_path: str | None = None
    spec_sha256: str
    # False when the original file was unavailable and spec.yaml was rebuilt
    # from the validated model (same meaning, comments and layout lost).
    spec_verbatim: bool = True
    project_root: str = ""
    tasks: list[TaskPlan] = Field(default_factory=list)


class QuestionOption(_Record):
    label: str = ""
    description: str = ""


class QuestionAnswer(_Record):
    """One clarifying question and the answer the human gave.

    `exchange` numbers the AskUserQuestion calls within a task from 1, so
    questions that were asked together share an exchange number.
    """

    exchange: int
    question: str
    header: str = ""
    options: list[QuestionOption] = Field(default_factory=list)
    multi_select: bool = False
    answer: str = ""
    # False when the human's answer was not preserved: no answer under this
    # question's key, or a later question in the same exchange had identical
    # text and overwrote it before it reached the store.
    answer_recorded: bool = True


class ReviewRecord(_Record):
    """One review round. Mirrors review.ReviewRound."""

    round: int
    verdict: str  # approved | changes_requested | unparsed | error
    findings: str = ""
    cost_usd: float = 0.0
    duration_s: float = 0.0
    fix_attempted: bool = False
    fix_ok: bool = False
    fix_cost_usd: float = 0.0
    error: str = ""


class TaskRecord(_Record):
    """A `task` line: everything known about one task once it finished."""

    kind: Literal["task"] = "task"
    recorded_at: datetime
    task_id: str
    agent: str = ""
    difficulty: str = ""
    model: str = ""
    status: str  # ok | failed | skipped
    reason: str = ""
    duration_s: float = 0.0
    # Implementation session only; review_cost_usd covers review and fix rounds.
    cost_usd: float = 0.0
    review_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    num_turns: int = 0
    stop_reason: str | None = None
    subtype: str | None = None
    session_id: str | None = None
    cwd: str = ""
    branch: str | None = None
    worktree_repo: str | None = None
    # The prompt as written in the spec. Upstream results that were prepended
    # at run time are the `text` of the dependencies' own records.
    prompt: str = ""
    # The task's final output. After a fix round this is the fix's summary,
    # which is what downstream tasks were given.
    text: str = ""
    review_verdict: str = ""
    reviews: list[ReviewRecord] = Field(default_factory=list)
    questions: list[QuestionAnswer] = Field(default_factory=list)
    denials: list[str] = Field(default_factory=list)


class SessionEnd(_Record):
    """The `end` line. Absent if the run never finished."""

    kind: Literal["end"] = "end"
    finished_at: datetime
    duration_s: float
    total_cost_usd: float
    counts: dict[str, int] = Field(default_factory=dict)
    # Set when the run was finalised after aborting, e.g. on KeyboardInterrupt.
    error: str | None = None


_Event = Annotated[SessionHeader | TaskRecord | SessionEnd, Field(discriminator="kind")]
_EVENT_ADAPTER: TypeAdapter[SessionHeader | TaskRecord | SessionEnd] = TypeAdapter(_Event)


# ------------------------------------------------------ loaded / derived views


def _status_counts(statuses: Iterable[str], not_run: int) -> dict[str, int]:
    """ok/failed/skipped/not_run always present, plus any other status seen."""
    counts = {"ok": 0, "failed": 0, "skipped": 0}
    for s in statuses:
        counts[s] = counts.get(s, 0) + 1
    counts["not_run"] = not_run
    return counts


@dataclass(frozen=True)
class SessionSummary:
    """One row of `list_sessions`: the run-level facts, without task bodies."""

    run_id: str
    started_at: datetime
    path: Path
    spec_name: str
    finished: bool
    # None while unfinished: the run's true length is unknown.
    duration_s: float | None
    total_cost_usd: float
    counts: dict[str, int]
    task_count: int


class Session(_Record):
    """A session read back from disk, complete or partial."""

    path: Path
    header: SessionHeader
    # task_id -> record, in completion order. If a task was recorded twice the
    # later record wins.
    tasks: dict[str, TaskRecord] = Field(default_factory=dict)
    end: SessionEnd | None = None
    # The stored spec, decoded as UTF-8. Read `spec_file` for the exact bytes.
    spec_text: str = ""
    # Problems tolerated while reading, e.g. a line torn by a crash.
    warnings: list[str] = Field(default_factory=list)

    @property
    def run_id(self) -> str:
        return self.header.run_id

    @property
    def started_at(self) -> datetime:
        return self.header.started_at

    @property
    def finished(self) -> bool:
        return self.end is not None

    @property
    def spec_file(self) -> Path:
        return self.path / SPEC_FILENAME

    @property
    def duration_s(self) -> float | None:
        return self.end.duration_s if self.end is not None else None

    @property
    def last_activity_at(self) -> datetime:
        """When the record was last written to: useful for unfinished runs."""
        stamps = [self.header.started_at, *(t.recorded_at for t in self.tasks.values())]
        if self.end is not None:
            stamps.append(self.end.finished_at)
        return max(stamps)

    @property
    def total_cost_usd(self) -> float:
        return sum(t.total_cost_usd for t in self.tasks.values())

    @property
    def counts(self) -> dict[str, int]:
        """Recomputed from the task records, so it is right for partial runs too."""
        planned = {p.id for p in self.header.tasks}
        return _status_counts(
            (t.status for t in self.tasks.values()), len(planned - set(self.tasks))
        )

    def ordered_task_ids(self) -> list[str]:
        """Planned task ids in dependency order, ties broken by spec order.

        Tasks recorded but absent from the plan (should not happen) come last.
        """
        plan = self.header.tasks
        done: set[str] = set()
        order: list[str] = []
        remaining = list(plan)
        while remaining:
            ready = [p for p in remaining if all(d in done for d in p.depends_on)]
            if not ready:
                # Cannot happen for a validated spec; never loop forever on a
                # hand-edited record.
                ready = remaining
            for p in ready:
                order.append(p.id)
                done.add(p.id)
            remaining = [p for p in remaining if p.id not in done]
        order.extend(tid for tid in self.tasks if tid not in done)
        return order

    def plan_for(self, task_id: str) -> TaskPlan | None:
        return next((p for p in self.header.tasks if p.id == task_id), None)

    def task(self, task_id: str) -> TaskRecord:
        """A task's record, or TaskNotFoundError."""
        try:
            return self.tasks[task_id]
        except KeyError:
            raise TaskNotFoundError(
                self.run_id, task_id, planned=self.plan_for(task_id) is not None
            ) from None

    def summary(self) -> SessionSummary:
        return SessionSummary(
            run_id=self.run_id,
            started_at=self.started_at,
            path=self.path,
            spec_name=self.header.spec_name,
            finished=self.finished,
            duration_s=self.duration_s,
            total_cost_usd=self.total_cost_usd,
            counts=self.counts,
            task_count=len(self.header.tasks),
        )


# ------------------------------------------------------------------ location


def session_root(root: Path | str | None = None) -> Path:
    """Where sessions live: `root` if given, else $ORCHESTRATOR_SESSION_DIR,
    else $XDG_DATA_HOME/claude-orchestrator/sessions, else
    ~/.local/share/claude-orchestrator/sessions. Does not create it."""
    if root is not None:
        return Path(root).expanduser()
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME")
    # The XDG spec says a relative value is invalid and must be ignored.
    base = Path(xdg) if xdg and Path(xdg).is_absolute() else Path.home() / ".local" / "share"
    return base / "claude-orchestrator" / "sessions"


def _dir_name(run_id: str, started_at: datetime) -> str:
    return f"{started_at.astimezone(UTC).strftime(_DIR_TS_FORMAT)}-{run_id}"


def _run_id_of(dir_name: str) -> str | None:
    stamp, sep, run_id = dir_name.partition("-")
    if not sep or not run_id or not _RUN_ID_RE.fullmatch(run_id):
        return None
    try:
        datetime.strptime(stamp, _DIR_TS_FORMAT)
    except ValueError:
        return None
    return run_id


def _session_dirs(root: Path) -> dict[str, list[Path]]:
    """run_id -> session directories under root (normally one each)."""
    found: dict[str, list[Path]] = {}
    if not root.is_dir():
        return found
    for child in root.iterdir():
        run_id = _run_id_of(child.name)
        if run_id is not None and child.is_dir():
            found.setdefault(run_id, []).append(child)
    return found


def resolve_session(run_id: str, root: Path | str | None = None) -> Path:
    """The directory of the session whose run id is `run_id` or starts with it.

    An exact match wins over prefix matches. Raises SessionNotFoundError or
    AmbiguousSessionError.
    """
    base = session_root(root)
    if not run_id:
        raise SessionNotFoundError("empty run id")
    dirs = _session_dirs(base)
    if run_id in dirs:
        candidates = dirs[run_id]
        matches = [run_id]
    else:
        matches = [rid for rid in dirs if rid.startswith(run_id)]
        if not matches:
            raise SessionNotFoundError(f"no session matches run id {run_id!r} in {base}")
        if len(matches) > 1:
            raise AmbiguousSessionError(run_id, matches)
        candidates = dirs[matches[0]]
    if len(candidates) > 1:
        # Same 8-hex run id at two different times: astronomically unlikely,
        # but refuse to guess.
        raise AmbiguousSessionError(run_id, [c.name for c in candidates])
    return candidates[0]


# -------------------------------------------------------------------- writing


def _encode_line(record: BaseModel) -> bytes:
    text = json.dumps(record.model_dump(mode="json"), ensure_ascii=False)
    # A lone surrogate inside a string (e.g. undecodable tool output) cannot be
    # encoded as UTF-8; backslashreplace turns it into a `\udcXX` escape, which
    # is valid JSON and decodes back to the same string.
    return (text + "\n").encode("utf-8", "backslashreplace")


def _write_private(path: Path, data: bytes, *, append: bool) -> None:
    """Write `data` whole or not at all, as far as this process can arrange.

    The run loop carries on after a failed write (e.g. ENOSPC), so a fragment
    left behind would be glued onto the next record and cost both lines on
    load. Two guards: a failed write truncates back to where it started, and if
    that truncation also failed, the next append first terminates the stray
    fragment with a newline so only the fragment is lost.
    """
    flags = os.O_RDWR | os.O_CREAT | (os.O_APPEND if append else os.O_EXCL)
    fd = os.open(path, flags, 0o600)
    try:
        start = os.fstat(fd).st_size
        if start and os.pread(fd, 1, start - 1) != b"\n":
            data = b"\n" + data
        try:
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.fsync(fd)
        except BaseException:
            with contextlib.suppress(OSError):
                os.ftruncate(fd, start)
            raise
    finally:
        os.close(fd)


def _reconstruct_spec(spec: Spec) -> bytes:
    body = yaml.safe_dump(
        spec.model_dump(mode="json"), sort_keys=False, allow_unicode=True, width=100
    )
    note = (
        "# Reconstructed by claude-orchestrator from the validated spec: the original\n"
        "# file was not available when this run started. Comments and layout are lost.\n"
    )
    return (note + body).encode("utf-8")


def _questions(exchanges: list[dict[str, Any]]) -> list[QuestionAnswer]:
    """Flatten TaskResult.questions ({"questions": [...], "answers": {...}} per
    AskUserQuestion call) into one record per question.

    Answers are keyed by question text (the tool's own contract), using exactly
    the key questions._render writes, including its "(no question text)"
    fallback. Two questions with the same text in one exchange share a key, so
    _render has already overwritten the earlier answer by the time it gets
    here: only the last of them is given the answer, and the earlier ones are
    marked answer_recorded=False rather than shown someone else's answer.
    """
    out: list[QuestionAnswer] = []
    for n, exchange in enumerate(exchanges, 1):
        asked = [
            q if isinstance(q, dict) else {"question": str(q)}
            for q in exchange.get("questions") or []
        ]
        answers = exchange.get("answers")
        if not isinstance(answers, dict):
            answers = {}
        keys = [q.get("question", "(no question text)") for q in asked]
        for i, (q, key) in enumerate(zip(asked, keys, strict=True)):
            shadowed = key in keys[i + 1 :]
            recorded = not shadowed and key in answers
            text = str(key)
            options = [
                QuestionOption(
                    label=str(o.get("label", "")), description=str(o.get("description") or "")
                )
                if isinstance(o, dict)
                else QuestionOption(label=str(o))
                for o in q.get("options") or []
            ]
            out.append(
                QuestionAnswer(
                    exchange=n,
                    question=text,
                    header=str(q.get("header") or ""),
                    options=options,
                    multi_select=bool(q.get("multiSelect")),
                    answer=str(answers[key]) if recorded else "",
                    answer_recorded=recorded,
                )
            )
    return out


def _reviews(rounds: list[Any]) -> list[ReviewRecord]:
    out: list[ReviewRecord] = []
    for r in rounds:
        if dataclasses.is_dataclass(r) and not isinstance(r, type):
            out.append(ReviewRecord.model_validate(dataclasses.asdict(r)))
        elif isinstance(r, dict):
            out.append(ReviewRecord.model_validate(r))
    return out


@dataclass
class SessionWriter:
    """Handle for a run in progress. Obtain one from `begin_session`.

    Safe to call from concurrently completing tasks: each record is a single
    locked append.
    """

    header: SessionHeader
    path: Path
    _t0: float = field(default_factory=time.monotonic, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _recorded: dict[str, TaskRecord] = field(default_factory=dict, repr=False)
    _end: SessionEnd | None = field(default=None, repr=False)

    @property
    def run_id(self) -> str:
        return self.header.run_id

    @property
    def started_at(self) -> datetime:
        return self.header.started_at

    @property
    def events_file(self) -> Path:
        return self.path / EVENTS_FILENAME

    @property
    def spec_file(self) -> Path:
        return self.path / SPEC_FILENAME

    @property
    def finished(self) -> bool:
        return self._end is not None

    def record_task(self, result: TaskResult) -> TaskRecord:
        """Append one task's record and fsync it. Call as each task completes,
        skipped tasks included. Recording a task id again appends a newer
        record, which wins on load."""
        plan = next((p for p in self.header.tasks if p.id == result.task_id), None)
        record = TaskRecord(
            recorded_at=datetime.now(UTC),
            task_id=result.task_id,
            agent=plan.agent if plan else "",
            difficulty=result.difficulty or (plan.difficulty if plan else ""),
            model=result.model or (plan.model if plan else ""),
            status=result.status,
            reason=result.reason,
            duration_s=result.duration_s,
            cost_usd=result.cost_usd,
            review_cost_usd=result.review_cost_usd,
            total_cost_usd=result.total_cost_usd,
            num_turns=result.num_turns,
            stop_reason=result.stop_reason,
            subtype=result.subtype,
            session_id=result.session_id,
            cwd=result.cwd,
            branch=result.branch,
            worktree_repo=result.worktree_repo,
            prompt=plan.prompt if plan else "",
            text=result.text,
            review_verdict=result.review_verdict,
            reviews=_reviews(result.reviews),
            questions=_questions(result.questions),
            denials=[str(d) for d in result.denials],
        )
        with self._lock:
            if self._end is not None:
                raise SessionStoreError(f"session {self.run_id} is already finalised")
            _write_private(self.events_file, _encode_line(record), append=True)
            # Same ordering rule as the loader, so sums over tasks agree exactly.
            self._recorded.pop(record.task_id, None)
            self._recorded[record.task_id] = record
        return record

    def _as_session(self) -> Session:
        return Session(
            path=self.path, header=self.header, tasks=dict(self._recorded), end=self._end
        )

    def finalize(
        self, results: Mapping[str, TaskResult] | None = None, *, error: str | None = None
    ) -> SessionSummary:
        """Write the `end` record and close the session.

        Any result in `results` whose task was never passed to `record_task` is
        recorded first, so passing the run's final results is a safety net
        against a missed task. `error` marks a run finalised after aborting.
        Raises SessionStoreError if already finalised.
        """
        for task_id, result in (results or {}).items():
            if task_id not in self._recorded:
                self.record_task(result)
        with self._lock:
            if self._end is not None:
                raise SessionStoreError(f"session {self.run_id} is already finalised")
            # Totals come from the same Session code the readers use, so the
            # summary printed at the end of a run matches list_sessions later.
            so_far = self._as_session()
            end = SessionEnd(
                finished_at=datetime.now(UTC),
                duration_s=time.monotonic() - self._t0,
                total_cost_usd=so_far.total_cost_usd,
                counts=so_far.counts,
                error=error,
            )
            _write_private(self.events_file, _encode_line(end), append=True)
            self._end = end
            return self._as_session().summary()


def begin_session(
    spec: Spec,
    run_id: str,
    *,
    spec_path: Path | str | None = None,
    spec_bytes: bytes | None = None,
    root: Path | str | None = None,
) -> SessionWriter:
    """Create the session directory, store the spec and write the `begin` line.

    The stored spec is, in order of preference: `spec_bytes` as given; the
    current contents of `spec_path`; or, if neither is available, a YAML
    rebuild of `spec` (flagged `spec_verbatim=False`). Pass `spec_path` even
    with `spec_bytes` so the record names the file.

    Raises ValueError for a run id that is not [A-Za-z0-9_]+, SessionStoreError
    if this run id already has a session, OSError if the root is unwritable.
    """
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"run id must match [A-Za-z0-9_]+, got {run_id!r}")

    started_at = datetime.now(UTC)
    t0 = time.monotonic()

    resolved_path = Path(spec_path).expanduser().resolve() if spec_path is not None else None
    verbatim = True
    if spec_bytes is None and resolved_path is not None:
        try:
            spec_bytes = resolved_path.read_bytes()
        except OSError:
            spec_bytes = None
    if spec_bytes is None:
        spec_bytes = _reconstruct_spec(spec)
        verbatim = False

    header = SessionHeader(
        run_id=run_id,
        started_at=started_at,
        spec_name=resolved_path.name if resolved_path is not None else "(unnamed spec)",
        spec_path=str(resolved_path) if resolved_path is not None else None,
        spec_sha256=hashlib.sha256(spec_bytes).hexdigest(),
        spec_verbatim=verbatim,
        project_root=str(spec.project_root),
        tasks=[
            TaskPlan(
                id=t.id,
                agent=t.agent,
                difficulty=spec.difficulty_for(t),
                model=model_for(spec.difficulty_for(t)),
                depends_on=list(t.depends_on),
                prompt=t.prompt,
                isolate=t.isolate,
            )
            for t in spec.tasks
        ],
    )

    base = session_root(root)
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    if run_id in _session_dirs(base):
        raise SessionStoreError(f"a session for run id {run_id!r} already exists in {base}")
    path = base / _dir_name(run_id, started_at)
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        raise SessionStoreError(f"session directory already exists: {path}") from None
    # mkdir's mode is filtered by the umask; make the privacy explicit.
    path.chmod(0o700)

    # Spec first, then the begin line: a directory with an events file always
    # has its spec beside it.
    _write_private(path / SPEC_FILENAME, spec_bytes, append=False)
    _write_private(path / EVENTS_FILENAME, _encode_line(header), append=False)
    return SessionWriter(header=header, path=path, _t0=t0)


# -------------------------------------------------------------------- reading


def _read_events(
    path: Path,
) -> tuple[SessionHeader, dict[str, TaskRecord], SessionEnd | None, list[str]]:
    events_file = path / EVENTS_FILENAME
    try:
        data = events_file.read_bytes()
    except FileNotFoundError:
        raise SessionStoreError(f"session has no {EVENTS_FILENAME}: {path}") from None

    # Split on b"\n" only: str.splitlines() would also split on U+2028 and
    # friends, which ensure_ascii=False leaves unescaped inside strings.
    lines = data.split(b"\n")
    header: SessionHeader | None = None
    tasks: dict[str, TaskRecord] = {}
    end: SessionEnd | None = None
    warnings: list[str] = []

    for lineno, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        is_last = lineno == len(lines)  # no trailing newline: possibly torn
        try:
            event = _EVENT_ADAPTER.validate_python(json.loads(raw.decode("utf-8", "replace")))
        except (json.JSONDecodeError, ValidationError) as exc:
            what = "incomplete final line (write interrupted)" if is_last else "unreadable line"
            warnings.append(f"{EVENTS_FILENAME}:{lineno}: {what}: {type(exc).__name__}")
            continue
        if isinstance(event, SessionHeader):
            if header is None:
                header = event
            else:
                warnings.append(f"{EVENTS_FILENAME}:{lineno}: duplicate begin record ignored")
        elif isinstance(event, TaskRecord):
            tasks.pop(event.task_id, None)  # keep completion order for the latest record
            tasks[event.task_id] = event
        else:
            end = event

    if header is None:
        raise SessionStoreError(f"session has no readable begin record: {path}")
    return header, tasks, end, warnings


def _load_dir(path: Path, *, with_spec: bool) -> Session:
    header, tasks, end, warnings = _read_events(path)
    spec_text = ""
    if with_spec:
        try:
            spec_text = (path / SPEC_FILENAME).read_bytes().decode("utf-8", "replace")
        except FileNotFoundError:
            warnings.append(f"{SPEC_FILENAME} is missing")
    return Session(
        path=path, header=header, tasks=tasks, end=end, spec_text=spec_text, warnings=warnings
    )


def list_sessions(root: Path | str | None = None) -> list[SessionSummary]:
    """Every readable session under the root, most recently started first.

    An empty or missing root gives []. Directories without a readable begin
    record (e.g. a crash during `begin_session`) are skipped.
    """
    out: list[SessionSummary] = []
    for paths in _session_dirs(session_root(root)).values():
        for path in paths:
            try:
                out.append(_load_dir(path, with_spec=False).summary())
            except (SessionStoreError, OSError):
                continue
    out.sort(key=lambda s: (s.started_at, s.run_id), reverse=True)
    return out


def load_session(run_id: str, root: Path | str | None = None) -> Session:
    """A full session, including the spec text, by run id or unique prefix.

    Raises SessionNotFoundError, AmbiguousSessionError, or SessionStoreError
    if the session exists but has no readable begin record.
    """
    return _load_dir(resolve_session(run_id, root), with_spec=True)


def load_task(run_id: str, task_id: str, root: Path | str | None = None) -> TaskRecord:
    """One task's record from a session, by run id (or unique prefix).

    Raises what load_session raises, plus TaskNotFoundError.
    """
    return load_session(run_id, root).task(task_id)


__all__ = [
    "ENV_VAR",
    "EVENTS_FILENAME",
    "FORMAT_VERSION",
    "SPEC_FILENAME",
    "AmbiguousSessionError",
    "QuestionAnswer",
    "QuestionOption",
    "ReviewRecord",
    "Session",
    "SessionEnd",
    "SessionHeader",
    "SessionNotFoundError",
    "SessionStoreError",
    "SessionSummary",
    "SessionWriter",
    "TaskNotFoundError",
    "TaskPlan",
    "TaskRecord",
    "begin_session",
    "list_sessions",
    "load_session",
    "load_task",
    "resolve_session",
    "session_root",
]
