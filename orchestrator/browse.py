"""Render recorded sessions as terminal text.

This is the read-only half of session browsing: everything here takes data
already loaded by `session_store` and returns a string. Nothing parses
arguments, touches the filesystem or prints, so the CLI decides where text goes
and tests can call these directly with hand-built records.

Conventions shared by every renderer:

- Times are shown in the viewer's local timezone by default, because "when did
  I run this" is a wall-clock question. Every renderer that shows a time takes
  `tz=` so tests (and anyone comparing across machines) can pin it.
- Costs use four decimals and a `$`, as the end-of-run report does, so the
  numbers here can be compared with what was printed when the run finished.
- Long free text (prompts, outputs, review findings, the spec) is reproduced
  verbatim, never wrapped or truncated: the store keeps it in full precisely so
  a reader can see it whole. Only table cells are squeezed onto one line.

Lookup failures (no such run, ambiguous prefix, unknown task) are raised by
`session_store` as `SessionStoreError` subclasses; `render_error` turns them
into a message worth showing a user, with suggestions where we have the data.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, tzinfo
from pathlib import Path

from .session_store import (
    SPEC_FILENAME,
    AmbiguousSessionError,
    QuestionAnswer,
    ReviewRecord,
    Session,
    SessionNotFoundError,
    SessionStoreError,
    SessionSummary,
    TaskNotFoundError,
    TaskPlan,
    TaskRecord,
)

RULE_WIDTH = 72

# Order statuses appear in count summaries; anything else seen follows, sorted.
_STATUS_ORDER = ("ok", "failed", "skipped", "not_run")

_VERDICT_SHORT = {
    "approved": "approved",
    "changes_requested": "CHANGES",
    "unparsed": "unparsed",
    "error": "ERROR",
}

# How many recent sessions to suggest when a run id matches nothing.
_SUGGEST_RECENT = 5


# ------------------------------------------------------------------- helpers


def _one_line(text: str) -> str:
    """Collapse whitespace (newlines included) so a value fits one table cell."""
    return " ".join(text.split())


def _table(header: Sequence[str], rows: Sequence[Sequence[str]], *, indent: str = "") -> list[str]:
    """Left-aligned columns, two spaces apart, trailing blanks stripped."""
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(cells: Sequence[str]) -> str:
        joined = "  ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True))
        return (indent + joined).rstrip()

    return [line(header), *(line(r) for r in rows)]


def _section(title: str) -> str:
    head = f"== {title} "
    return head + "=" * max(3, RULE_WIDTH - len(head))


def _fields(pairs: Sequence[tuple[str, str]], *, indent: str = "  ") -> list[str]:
    """Aligned `label  value` lines; continuation lines of a value line up too."""
    pairs = [(k, v) for k, v in pairs if v]
    if not pairs:
        return []
    width = max(len(k) for k, _ in pairs)
    pad = indent + " " * (width + 2)
    out: list[str] = []
    for key, value in pairs:
        first, *rest = value.split("\n")
        out.append(f"{indent}{key.ljust(width)}  {first}".rstrip())
        out.extend((pad + r).rstrip() for r in rest)
    return out


def _when(moment: datetime, tz: tzinfo | None) -> str:
    """Wall-clock time in `tz`, or the system's local zone when tz is None."""
    local = moment.astimezone(tz)
    return local.strftime("%Y-%m-%d %H:%M:%S %z").rstrip()


def _when_short(moment: datetime, tz: tzinfo | None) -> str:
    return moment.astimezone(tz).strftime("%Y-%m-%d %H:%M")


def _cost(usd: float) -> str:
    return f"${usd:.4f}"


def _duration(seconds: float) -> str:
    """Human-scale duration: `42.1s`, `3m 07s`, `1h 02m`."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    whole = round(seconds)
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {secs:02d}s"


def _counts(counts: dict[str, int]) -> str:
    """`3 ok, 1 failed, 2 not run`; zero counts omitted, `-` if nothing at all."""
    keys = [k for k in _STATUS_ORDER if k in counts]
    keys += sorted(k for k in counts if k not in _STATUS_ORDER)
    parts = [f"{counts[k]} {k.replace('_', ' ')}" for k in keys if counts[k]]
    return ", ".join(parts) or "-"


def _review_cell(record: TaskRecord | None) -> str:
    """Verdict plus rounds taken, e.g. `approved(2)`, matching the run report."""
    if record is None or not record.reviews:
        return "-"
    verdict = record.review_verdict or record.reviews[-1].verdict
    return f"{_VERDICT_SHORT.get(verdict, verdict)}({len(record.reviews)})"


def _not_run_note(session: Session) -> str:
    if session.finished:
        return "not run: the run finished without a record for this task"
    return "not run: the run did not finish, and stopped before this task completed"


# ------------------------------------------------------------ session list


def render_session_list(
    sessions: Sequence[SessionSummary],
    *,
    root: Path | None = None,
    tz: tzinfo | None = None,
) -> str:
    """A table of past sessions, in the order given (list_sessions: newest first).

    `root` is only used to say where we looked when there is nothing to show,
    which is the first thing a user needs to know if they expected sessions:
    usually a different ORCHESTRATOR_SESSION_DIR, or a run that predates
    recording.
    """
    if not sessions:
        where = f" in {root}" if root is not None else ""
        return f"No sessions recorded yet{where}."

    rows = [
        (
            s.run_id,
            _when_short(s.started_at, tz),
            _duration(s.duration_s) if s.duration_s is not None else "unfinished",
            _one_line(s.spec_name),
            _counts(s.counts),
            _cost(s.total_cost_usd),
        )
        for s in sessions
    ]
    lines = _table(("RUN ID", "STARTED", "DURATION", "SPEC", "TASKS", "COST"), rows)
    noun = "session" if len(sessions) == 1 else "sessions"
    total = sum(s.total_cost_usd for s in sessions)
    lines += ["", f"{len(sessions)} {noun}, {_cost(total)} in total"]
    return "\n".join(lines)


# --------------------------------------------------------- session summary


def render_session(session: Session, *, tz: tzinfo | None = None) -> str:
    """Run-level facts and totals, then one row per task in dependency order.

    Planned tasks without a record appear as `not run`, so an interrupted run
    shows what it never got to. Reasons for failures go below the table rather
    than in it, because they are free text and would wreck the column layout.
    """
    header = session.header
    tasks = session.tasks

    if session.end is not None:
        assert session.duration_s is not None
        ended = f"{_when(session.end.finished_at, tz)} ({_duration(session.duration_s)})"
    else:
        ended = f"unfinished (last activity {_when(session.last_activity_at, tz)})"

    spec_line = header.spec_path or header.spec_name
    if not header.spec_verbatim:
        spec_line += "\n(original file was unavailable; the stored copy is a rebuild)"

    review_total = sum(t.review_cost_usd for t in tasks.values())
    cost_line = _cost(session.total_cost_usd)
    if review_total:
        cost_line += f" (of which review {_cost(review_total)})"

    questions = sum(len(t.questions) for t in tasks.values())
    denials = sum(len(t.denials) for t in tasks.values())

    lines = [f"Run {session.run_id}  {header.spec_name}", ""]
    lines += _fields(
        [
            ("started", _when(header.started_at, tz)),
            ("finished", ended),
            ("aborted", (session.end.error or "") if session.end is not None else ""),
            ("spec", spec_line),
            ("project", header.project_root),
            ("record", str(session.path)),
            ("tasks", f"{_counts(session.counts)} ({len(header.tasks)} planned)"),
            ("cost", cost_line),
            ("questions", f"{questions} asked" if questions else ""),
            ("denials", f"{denials} tool call(s) denied" if denials else ""),
        ]
    )

    rows: list[tuple[str, ...]] = []
    for tid in session.ordered_task_ids():
        record = tasks.get(tid)
        plan = session.plan_for(tid)
        if record is None:
            rows.append((tid, "not run", plan.model if plan else "", "-", "", "", _deps(plan)))
            continue
        rows.append(
            (
                tid,
                record.status,
                record.model or (plan.model if plan else ""),
                _review_cell(record),
                _duration(record.duration_s),
                _cost(record.total_cost_usd),
                _deps(plan),
            )
        )
    lines += ["", *_table(("TASK", "STATUS", "MODEL", "REVIEW", "TIME", "COST", "AFTER"), rows)]

    problems = [t for t in tasks.values() if t.status != "ok"]
    if problems:
        lines += ["", "not ok:"]
        for t in problems:
            lines.append(f"  {t.task_id}: {t.status} - {_one_line(t.reason) or '(no detail)'}")

    open_findings = [t for t in tasks.values() if t.review_verdict == "changes_requested"]
    if open_findings:
        lines += ["", "review rounds ran out; work accepted with findings open:"]
        lines += [f"  {t.task_id} (after {len(t.reviews)} round(s))" for t in open_findings]

    if session.warnings:
        lines += ["", "warnings while reading the record:"]
        lines += [f"  {w}" for w in session.warnings]
    return "\n".join(lines)


def _deps(plan: TaskPlan | None) -> str:
    return ", ".join(plan.depends_on) if plan and plan.depends_on else ""


# ----------------------------------------------------------------- the spec


def render_spec(session: Session) -> str:
    """The spec the session ran, exactly as stored: no header, no reformatting.

    Returned unmodified (including whether it ends in a newline), so the caller
    should write it without adding one, e.g. `sys.stdout.write(...)`, and the
    result can be redirected to a file and run again. The text is the stored
    bytes decoded as UTF-8; anyone needing the raw bytes of a file that is not
    valid UTF-8 should copy `session.spec_file` instead.

    If the original file was unavailable at run time, the stored copy is a
    rebuild, and it says so itself in a leading comment.

    Raises SessionStoreError when the session has no stored spec. A real spec
    is never empty (it must declare tasks), so empty text means spec.yaml is
    missing or was emptied after the run.
    """
    if not session.spec_text:
        raise SessionStoreError(
            f"session {session.run_id} has no stored spec ({session.path / SPEC_FILENAME} "
            "is missing or empty)"
        )
    return session.spec_text


# ---------------------------------------------------------------- one task


def render_task(session: Session, task_id: str, *, tz: tzinfo | None = None) -> str:
    """Everything recorded about one task, free text in full.

    Takes the whole session rather than a TaskRecord so it can show the task's
    place in the plan (dependencies, isolation) and so a task that was planned
    but never recorded, as in an interrupted run, still renders: its plan and
    prompt, marked as not run. That is more useful than an error to someone
    asking "what was this task going to do?".

    Raises TaskNotFoundError (planned=False) only when the id is not in the
    session at all; pass the session to `render_error` to list valid ids.
    """
    plan = session.plan_for(task_id)
    record = session.tasks.get(task_id)
    if record is None and plan is None:
        raise TaskNotFoundError(session.run_id, task_id, planned=False)

    lines = [f"Task {task_id}  (run {session.run_id}, {session.header.spec_name})", ""]
    if record is None:
        assert plan is not None
        lines += _fields(
            [
                ("status", _not_run_note(session)),
                ("agent", plan.agent),
                ("model", f"{plan.model} ({plan.difficulty})"),
                ("after", _deps(plan)),
                ("isolate", plan.isolate if plan.isolate != "none" else ""),
            ]
        )
        lines += ["", _section("Prompt"), plan.prompt or "(empty prompt)"]
        return "\n".join(lines)

    status = record.status
    if record.reason:
        status += f" - {record.reason}"
    model = record.model or (plan.model if plan else "")
    if record.difficulty:
        model += f" ({record.difficulty})"
    run_detail = f"{_duration(record.duration_s)}, {record.num_turns} turn(s)"
    if record.stop_reason:
        run_detail += f", stop reason {record.stop_reason}"
    if record.subtype:
        run_detail += f", result {record.subtype}"
    cost = _cost(record.total_cost_usd)
    if record.review_cost_usd:
        cost += (
            f" (implementation {_cost(record.cost_usd)}, "
            f"review and fixes {_cost(record.review_cost_usd)})"
        )
    branch = record.branch or ""
    if branch and record.worktree_repo:
        branch += f" (worktree of {record.worktree_repo})"

    lines += _fields(
        [
            ("status", status),
            ("agent", record.agent or (plan.agent if plan else "")),
            ("model", model),
            ("after", _deps(plan)),
            ("isolate", plan.isolate if plan and plan.isolate != "none" else ""),
            ("ran", run_detail),
            ("cost", cost),
            ("review", _review_cell(record) if record.reviews else ""),
            ("recorded", _when(record.recorded_at, tz)),
            ("cwd", record.cwd),
            ("branch", branch),
            ("session", record.session_id or ""),
        ]
    )

    prompt = record.prompt or (plan.prompt if plan else "")
    lines += ["", _section("Prompt"), prompt or "(empty prompt)"]
    lines += ["", _section("Output"), record.text or "(no output)"]

    if record.questions:
        lines += ["", _section(f"Questions ({len(record.questions)})")]
        lines += _render_questions(record.questions)

    if record.reviews:
        noun = "round" if len(record.reviews) == 1 else "rounds"
        lines += ["", _section(f"Review ({len(record.reviews)} {noun})")]
        for n, review in enumerate(record.reviews):
            lines += ([""] if n else []) + _render_review(review)

    if record.denials:
        lines += ["", _section(f"Denied tool calls ({len(record.denials)})")]
        lines += [f"- {d}" for d in record.denials]

    return "\n".join(lines)


def _render_questions(questions: Sequence[QuestionAnswer]) -> list[str]:
    """Questions grouped by the AskUserQuestion call that asked them."""
    out: list[str] = []
    exchange: int | None = None
    for i, q in enumerate(questions):
        if q.exchange != exchange:
            if exchange is not None:
                out.append("")
            exchange = q.exchange
            out.append(f"Exchange {exchange}:")
        title = f"[{q.header}] {q.question}" if q.header else q.question
        out.append(f"  Q: {_indent_rest(title, '     ')}")
        if q.options:
            kind = "choose any" if q.multi_select else "choose one"
            out.append(f"     options ({kind}):")
            for o in q.options:
                desc = f" - {_one_line(o.description)}" if o.description else ""
                out.append(f"       * {_one_line(o.label)}{desc}")
        if q.answer_recorded:
            out.append(f"  A: {_indent_rest(q.answer or '(empty answer)', '     ')}")
        elif any(
            later.exchange == q.exchange and later.question == q.question
            for later in questions[i + 1 :]
        ):
            # The answer map is keyed by question text, so it was overwritten
            # before it was recorded (see session_store._questions).
            out.append("  A: (lost: a later question in this exchange had the same text)")
        else:
            out.append("  A: (no answer recorded)")
    return out


def _render_review(review: ReviewRecord) -> list[str]:
    head = (
        f"Round {review.round}: {review.verdict}  "
        f"{_cost(review.cost_usd)}  {_duration(review.duration_s)}"
    )
    out = [head]
    if review.fix_attempted:
        outcome = "succeeded" if review.fix_ok else "failed"
        out.append(f"  fix {outcome} ({_cost(review.fix_cost_usd)})")
    if review.error:
        out.append(f"  error: {_indent_rest(review.error, '         ')}")
    if review.findings:
        # Verbatim: findings are usually markdown the reviewer laid out itself.
        out += ["  findings:", review.findings]
    return out


def _indent_rest(text: str, pad: str) -> str:
    """Indent every line after the first, for values that follow a label."""
    return text.replace("\n", "\n" + pad)


# ------------------------------------------------------------------- errors


def render_error(
    exc: SessionStoreError,
    *,
    sessions: Sequence[SessionSummary] | None = None,
    session: Session | None = None,
    tz: tzinfo | None = None,
) -> str:
    """A user-facing message for a failed lookup, with suggestions if possible.

    The store's own messages are accurate but bare. Given more context this
    adds what the user needs next:

    - `sessions` (from list_sessions): for an unknown run id, say there are no
      sessions at all, or list the most recent ones; for an ambiguous prefix,
      show each match with its date and spec so the right one is recognisable.
    - `session`: for an unknown task id, list the tasks that do exist.

    Any other SessionStoreError is returned as its message.
    """
    if isinstance(exc, AmbiguousSessionError):
        lines = [
            f"Run id {exc.prefix!r} matches {len(exc.matches)} sessions; "
            "use more characters of the run id."
        ]
        known = {s.run_id: s for s in sessions or ()}
        matched = [known[m] for m in exc.matches if m in known]
        if matched:
            matched.sort(key=lambda s: (s.started_at, s.run_id), reverse=True)
            rows = [
                (s.run_id, _when_short(s.started_at, tz), _one_line(s.spec_name)) for s in matched
            ]
            lines += _table(("RUN ID", "STARTED", "SPEC"), rows, indent="  ")
        else:
            lines += [f"  {m}" for m in sorted(exc.matches)]
        return "\n".join(lines)

    if isinstance(exc, SessionNotFoundError):
        if sessions is not None and not sessions:
            return f"{exc}\nNo sessions have been recorded yet."
        lines = [str(exc)]
        if sessions:
            recent = list(sessions)[:_SUGGEST_RECENT]
            lines.append("Most recent sessions:")
            rows = [
                (s.run_id, _when_short(s.started_at, tz), _one_line(s.spec_name)) for s in recent
            ]
            lines += _table(("RUN ID", "STARTED", "SPEC"), rows, indent="  ")
        return "\n".join(lines)

    if isinstance(exc, TaskNotFoundError):
        lines = [str(exc)]
        if session is not None:
            ids = session.ordered_task_ids()
            if ids:
                lines.append(f"Tasks in this session: {', '.join(ids)}")
            else:
                lines.append("This session has no tasks.")
        return "\n".join(lines)

    return str(exc)


__all__ = [
    "render_error",
    "render_session",
    "render_session_list",
    "render_spec",
    "render_task",
]
