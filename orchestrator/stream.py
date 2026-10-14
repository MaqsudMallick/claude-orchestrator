"""Live output capture for running tasks: the layer between publishers and a viewer.

Tasks publish what they are doing (assistant text, tool calls, status changes)
as `StreamUpdate`s; a viewer reads per-task `TaskStream` snapshots and draws
them. This module only captures and normalizes. It never renders or prints, so
it works the same with a curses viewer, a plain-text one, or none at all.

Threading model:

- Publishers run on the asyncio event loop; the viewer reads from a worker
  thread. All shared state sits behind one `threading.Lock` (an `anyio.Lock`
  cannot be held across threads), and every read hands back a copy, so a reader
  never sees a buffer mid-update.
- The lock is only ever held for short, non-blocking bookkeeping, so taking it
  on the event loop does not stall other tasks.

Text normalization:

- Assistant text arrives as token fragments, not lines. Fragments accumulate in
  `pending`; each newline moves a completed line into the task's ring buffer.
  `\\r\\n` and lone `\\r` count as newlines.
- Non-text events (`tool`, `status`, `note`) are whole events, not fragments.
  Any in-progress text line is flushed first so the buffer stays in
  chronological order, then the event's text is appended as its own line(s).
  Lines are stored verbatim with no kind marker: deciding how a tool call looks
  is the viewer's job.
- The ring buffer holds `scrollback` lines per task, and `pending` is
  hard-wrapped at `MAX_PENDING` characters, so output that never emits a
  newline is bounded too.

Terminal handover (`StreamHub.suspended`): see that method for the handshake.
The short version is that a viewer must `attach_viewer` before it takes the
terminal; with no viewer attached, suspending never waits.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Literal

EventKind = Literal["text", "tool", "status", "note"]

# A single unterminated line longer than this is wrapped into the buffer, so a
# task streaming without newlines cannot grow `pending` without limit.
MAX_PENDING = 4096

# How long `suspended()` waits for an attached viewer to confirm it released the
# terminal. A viewer that has hung or died must not wedge a question forever;
# after this the prompt goes ahead and at worst the screen is briefly garbled.
SUSPEND_ACK_TIMEOUT = 5.0


@dataclass(frozen=True)
class StreamUpdate:
    task_id: str
    kind: EventKind
    text: str
    at: float  # time.monotonic()


@dataclass
class TaskStream:
    task_id: str
    agent: str
    model: str
    status: str  # pending | running | ok | failed | skipped
    lines: list[str]  # snapshot of the ring buffer, oldest first
    pending: str  # trailing partial line, not yet newline-terminated
    updated_at: float
    unseen: int  # updates since this task was last marked seen


class _State:
    """Mutable per-task state. Only touched with the hub's lock held."""

    __slots__ = (
        "after_cr",
        "agent",
        "lines",
        "model",
        "pending",
        "status",
        "unseen",
        "updated_at",
    )

    def __init__(self, agent: str, model: str, scrollback: int) -> None:
        self.after_cr = False
        self.agent = agent
        self.model = model
        self.status = "pending"
        self.lines: deque[str] = deque(maxlen=scrollback)
        self.pending = ""
        self.updated_at = time.monotonic()
        self.unseen = 0

    def feed(self, text: str) -> None:
        """Accumulate a text fragment, moving each completed line to the buffer."""
        if not text:
            return
        # A "\r\n" split across two fragments is one newline, not two.
        if self.after_cr and text.startswith("\n"):
            text = text[1:]
        self.after_cr = text.endswith("\r")
        parts = _normalize_newlines(text).split("\n")
        # Every part but the last was terminated by a newline in this fragment.
        for part in parts[:-1]:
            self.lines.append(self.pending + part)
            self.pending = ""
        self.pending += parts[-1]
        while len(self.pending) > MAX_PENDING:
            self.lines.append(self.pending[:MAX_PENDING])
            self.pending = self.pending[MAX_PENDING:]

    def flush(self) -> None:
        self.after_cr = False
        if self.pending:
            self.lines.append(self.pending)
            self.pending = ""

    def snapshot(self, task_id: str) -> TaskStream:
        return TaskStream(
            task_id=task_id,
            agent=self.agent,
            model=self.model,
            status=self.status,
            lines=list(self.lines),
            pending=self.pending,
            updated_at=self.updated_at,
            unseen=self.unseen,
        )


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


class StreamHub:
    """Thread-safe store of every task's live output, plus the terminal handover.

    Unknown task ids are tolerated on the write side: `publish` and `set_status`
    register them on the fly (agent and model empty), because dropping output or
    raising into a running task is worse than an unlabelled pane. `mark_seen`
    ignores them. Only `snapshot` raises `KeyError`, since a reader asking for a
    task that does not exist is a bug in the reader.
    """

    def __init__(self, scrollback: int = 500) -> None:
        if scrollback < 1:
            raise ValueError(f"scrollback must be at least 1, got {scrollback}")
        self._scrollback = scrollback
        self._lock = threading.Lock()
        # Shares the lock, so handover state and task state are never out of step.
        self._cond = threading.Condition(self._lock)
        # dicts keep insertion order, which is the registration order.
        self._tasks: dict[str, _State] = {}
        # Handover state; see `suspended`.
        self._viewers = 0
        self._suspend_depth = 0
        self._released = False

    # -- publishing (event loop) ---------------------------------------------

    def register(self, task_id: str, agent: str, model: str) -> None:
        """Declare a task. Re-registering updates its labels and keeps its output."""
        with self._lock:
            state = self._tasks.get(task_id)
            if state is None:
                self._tasks[task_id] = _State(agent, model, self._scrollback)
            else:
                state.agent = agent
                state.model = model

    def publish(self, update: StreamUpdate) -> None:
        with self._lock:
            state = self._ensure(update.task_id)
            if update.kind == "text":
                state.feed(update.text)
            else:
                state.flush()
                state.feed(update.text)
                state.flush()
            state.updated_at = update.at
            state.unseen += 1

    def set_status(self, task_id: str, status: str) -> None:
        """Record a lifecycle change. Counts as an update for `unseen`."""
        with self._lock:
            state = self._ensure(task_id)
            if state.status == status:
                return
            state.status = status
            state.updated_at = time.monotonic()
            state.unseen += 1

    def _ensure(self, task_id: str) -> _State:
        state = self._tasks.get(task_id)
        if state is None:
            state = self._tasks[task_id] = _State("", "", self._scrollback)
        return state

    # -- reading (viewer thread) ---------------------------------------------

    def mark_seen(self, task_id: str) -> None:
        with self._lock:
            state = self._tasks.get(task_id)
            if state is not None:
                state.unseen = 0

    def task_ids(self) -> list[str]:
        """Task ids in registration order."""
        with self._lock:
            return list(self._tasks)

    def snapshot(self, task_id: str) -> TaskStream:
        """A copy of one task's stream. Raises KeyError for an unknown id."""
        with self._lock:
            return self._tasks[task_id].snapshot(task_id)

    def snapshots(self) -> list[TaskStream]:
        """Copies of every task's stream in registration order, taken atomically."""
        with self._lock:
            return [state.snapshot(tid) for tid, state in self._tasks.items()]

    # -- terminal handover ---------------------------------------------------

    def suspended(self) -> AbstractContextManager[None]:
        """Take the terminal away from the viewer for the duration of the block.

        Used around a human prompt (AskUserQuestion) so the prompt and the
        viewer do not both draw on the screen.

        The hub tracks two separate things: whether a suspension is active
        (some caller is inside this block), and whether the viewer has actually
        let go of the terminal. The second changes only when the viewer says so
        (`acknowledge_suspend` / `reclaim_terminal`), never when a suspension
        ends. That separation is what lets back-to-back prompts through without
        a fresh acknowledgement: a viewer that is still released stays released.

        Prompt side (this method):

        1. Entering marks a suspension active (`suspend_requested()` turns True).
        2. It returns at once if no viewer is attached (the normal case) or the
           viewer is already released. Otherwise it blocks until the viewer
           calls `acknowledge_suspend()`, the viewer detaches, or
           `SUSPEND_ACK_TIMEOUT` passes.
        3. Leaving ends the suspension once the outermost block exits (nesting
           is allowed) and never blocks.

        Entering can block, so call it from a worker thread (for example inside
        the function handed to `anyio.to_thread.run_sync`), not on the event
        loop, or publishers stall while the viewer catches up.

        Viewer side:

        - `attach_viewer()` before taking the terminal; `detach_viewer()` after
          restoring it for good (this also releases any waiting prompt).
        - Draw only while you hold the terminal. Poll `suspend_requested()`;
          when it is True, restore the terminal (e.g. `curses.endwin()`) and
          then call `acknowledge_suspend()`. A suspension may already be active
          at attach time, so check before the first draw.
        - While released, call `reclaim_terminal()` (blocking, or with a timeout
          to keep polling) until it returns True, then take the terminal back
          and redraw everything. It succeeds only when no suspension is active,
          and checks and reclaims atomically, so a prompt that starts in the
          gap keeps the terminal and the viewer simply stays released.
        """
        return self._suspended()

    @contextmanager
    def _suspended(self) -> Iterator[None]:
        with self._cond:
            self._suspend_depth += 1
            self._cond.notify_all()
            self._cond.wait_for(
                lambda: self._released or self._viewers == 0, timeout=SUSPEND_ACK_TIMEOUT
            )
        try:
            yield
        finally:
            with self._cond:
                self._suspend_depth -= 1
                self._cond.notify_all()

    def attach_viewer(self) -> None:
        """Announce that a viewer owns the terminal and will honour suspensions."""
        with self._cond:
            self._viewers += 1

    def detach_viewer(self) -> None:
        """Announce that a viewer has given the terminal back for good."""
        with self._cond:
            if self._viewers > 0:
                self._viewers -= 1
            if self._viewers == 0:
                self._released = False
            self._cond.notify_all()

    def suspend_requested(self) -> bool:
        """True while some caller is inside `suspended()`. Viewers poll this."""
        with self._cond:
            return self._suspend_depth > 0

    def acknowledge_suspend(self) -> None:
        """Viewer confirms it has released the terminal. Ignored if not suspended.

        Ignoring an acknowledgement outside a suspension matters: otherwise a
        stray one would leave the hub believing the viewer is released while it
        is still drawing, and the next prompt would not wait for it.
        """
        with self._cond:
            if self._suspend_depth > 0:
                self._released = True
                self._cond.notify_all()

    def reclaim_terminal(self, timeout: float | None = None) -> bool:
        """Wait up to `timeout` for no active suspension, then hand the terminal back.

        Returns True if the viewer now holds the terminal again and may redraw;
        False on timeout, in which case it is still released. `timeout=0` polls.
        """
        with self._cond:
            if not self._cond.wait_for(lambda: self._suspend_depth == 0, timeout=timeout):
                return False
            self._released = False
            return True
