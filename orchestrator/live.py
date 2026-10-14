"""Terminal viewer for a running DAG: tabs of tasks over the selected task's live output.

This is the drawing half of `stream`. It only reads from a `StreamHub` (through
`snapshots`, plus the bookkeeping calls `mark_seen` and the handover protocol)
and never publishes, so turning it off changes nothing about a run.

Layout, top to bottom:

- a tab bar of every task in registration order: number, id, status glyph, and
  `*` when the task has output the user has not looked at yet, with a separator
  wherever the wave changes between neighbouring tabs (see `_draw_tabs`);
- an info row for the selected task (agent, model, status, follow mode, and for
  a task that has not started, the dependencies it is still waiting for);
- the selected task's scrollback plus its in-progress `pending` line, hard
  wrapped to the window width, newest at the bottom;
- the key legend.

Threading. curses calls block (`getch` waits up to its timeout) and are not
async-aware, so the whole curses session runs on one worker thread via
`anyio.to_thread.run_sync`. The event loop keeps publishing meanwhile. The
thread talks back to the loop only through anyio's `from_thread` helpers: it
peeks at `stop` with `from_thread.run_sync(stop.is_set)` (an anyio `Event` is
not safe to touch directly from another thread) and calls
`from_thread.check_cancelled()` each tick, so cancelling `run_viewer` unwinds
the thread within one poll interval instead of hanging on it.

Terminal handover. Question prompts take the terminal through
`StreamHub.suspended()`; this module is the viewer side of that handshake, see
`_Viewer._hand_over`. Every path out of the curses session, including an
exception or cancellation, goes through `_Viewer.run`'s `finally`, which puts
the tty back in shell mode before the hub is told the viewer has detached.
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING

import anyio
import anyio.from_thread
import anyio.to_thread

from .stream import StreamHub, TaskStream

if TYPE_CHECKING:
    import curses

# getch timeout in ms: how often the loop wakes to redraw, check `stop` and
# notice a suspension request when no key is pressed.
POLL_MS = 100

# How long one `reclaim_terminal` wait lasts while released. Between waits the
# thread checks `stop` and cancellation, so this bounds how long quitting takes
# while a prompt holds the terminal.
_RECLAIM_POLL = POLL_MS / 1000

_UNICODE_GLYPHS = {"pending": "·", "running": "▶", "ok": "✓", "failed": "✗", "skipped": "–"}
_ASCII_GLYPHS = {"pending": ".", "running": ">", "ok": "+", "failed": "x", "skipped": "-"}

# Colour pair numbers, allocated in `_Viewer._setup` when the terminal has colours.
_PAIR_OK = 1
_PAIR_FAILED = 2
_PAIR_RUNNING = 3
_PAIR_ACCENT = 4


def _legend(follow: bool, unicode: bool, show_tools: bool) -> str:
    """Return the key legend shown on the bottom row.

    The `f` entry names what pressing `f` will do next, not the current mode:
    the current mode is already shown on the info row, and a legend that said
    "follow" while following would read as an instruction to do what is already
    happening. The label is padded to the width of the longer word so `q quit`
    stays put when the mode flips. The `t` entry follows the same rule for the
    same reasons: it says what `t` will do, and is padded to its longer label.
    Kept free of `self` and curses so it can be tested without a terminal.
    """
    arrows = "←/→" if unicode else "<-/->"
    toggle = ("unfollow" if follow else "follow").ljust(len("unfollow"))
    tools = ("hide tools" if show_tools else "tools").ljust(len("hide tools"))
    return f"{arrows} or 1-9 select   f {toggle}   t {tools}   q quit"


# Drawn between two neighbouring tabs from different waves.
_WAVE_SEP = "│"
_WAVE_SEP_ASCII = "|"

# Statuses after which a task no longer holds up its dependents. A failed
# dependency is "finished" too: its dependents are about to be skipped, and
# listing it as something still awaited would suggest the wait could end well.
_FINISHED = frozenset({"ok", "failed", "skipped"})


def supported() -> bool:
    """True when a curses viewer can run here: stdout is a TTY and curses imports.

    `find_spec` rather than an import, so asking does not load curses (or fail
    noisily on a Python built without it, such as stock Windows).
    """
    try:
        isatty = sys.stdout.isatty()
    except (AttributeError, ValueError):  # replaced or closed stdout
        return False
    return isatty and importlib.util.find_spec("curses") is not None


async def run_viewer(hub: StreamHub, stop: anyio.Event) -> None:
    """Show the live viewer until `stop` is set, the user presses `q`, or we are cancelled.

    Pressing `q` sets `stop` (on the event loop, where anyio events belong), so
    whoever started the viewer learns the user is done with it. The terminal is
    restored before this returns or raises.
    """
    user_quit = await anyio.to_thread.run_sync(_thread_main, hub, stop)
    if user_quit:
        stop.set()


def _thread_main(hub: StreamHub, stop: anyio.Event) -> bool:
    """Worker-thread body. Returns True if the user quit with `q`."""
    hub.attach_viewer()
    try:
        return _Viewer(hub, stop).run()
    finally:
        # Only after the terminal is restored: detaching releases any prompt
        # still waiting on us, and it must find a clean screen.
        hub.detach_viewer()


def _should_exit(stop: anyio.Event) -> bool:
    """Raise if `run_viewer` was cancelled; otherwise report whether `stop` is set."""
    anyio.from_thread.check_cancelled()
    return anyio.from_thread.run_sync(stop.is_set)


class _Viewer:
    """One curses session. All methods run on the worker thread."""

    def __init__(self, hub: StreamHub, stop: anyio.Event) -> None:
        self.hub = hub
        self.stop = stop
        self.selected: str | None = None
        self.follow = False
        # Tool-call lines are hidden by default: they are mostly noise next to
        # the model's own text. `t` reveals them (see `_draw_body`).
        self.show_tools = False
        self.scr: curses.window | None = None
        self.unicode = (sys.stdout.encoding or "").lower().replace("-", "") == "utf8"
        self.glyphs = _UNICODE_GLYPHS if self.unicode else _ASCII_GLYPHS

    # -- lifecycle -----------------------------------------------------------

    def run(self) -> bool:
        import curses

        # A prompt may already hold the terminal when we attach. `initscr` can
        # emit terminal init sequences (alternate screen and so on), so do not
        # even start curses until the prompt is done.
        if self.hub.suspend_requested():
            self.hub.acknowledge_suspend()
            if not self._wait_reclaim():
                return False
        try:
            self.scr = curses.initscr()
            self._setup()
            return self._loop()
        finally:
            self._restore()

    def _setup(self) -> None:
        import curses

        assert self.scr is not None
        curses.noecho()
        curses.cbreak()
        self.scr.keypad(True)
        self.scr.timeout(POLL_MS)
        # Some terminals cannot hide the cursor; that is cosmetic, not fatal.
        with contextlib.suppress(curses.error):
            curses.curs_set(0)
        if curses.has_colors():
            curses.start_color()
            background = -1
            try:
                curses.use_default_colors()
            except curses.error:
                background = curses.COLOR_BLACK
            curses.init_pair(_PAIR_OK, curses.COLOR_GREEN, background)
            curses.init_pair(_PAIR_FAILED, curses.COLOR_RED, background)
            curses.init_pair(_PAIR_RUNNING, curses.COLOR_YELLOW, background)
            curses.init_pair(_PAIR_ACCENT, curses.COLOR_CYAN, background)

    def _restore(self) -> None:
        """Return the tty to shell mode. Safe to call in any state, including twice.

        Mirrors `curses.wrapper`'s cleanup, but skips it when we are already
        out of curses (released for a prompt, or `initscr` never succeeded):
        poking modes then would scribble on the prompt's terminal settings.
        """
        import curses

        if self.scr is None or curses.isendwin():
            return
        try:
            self.scr.keypad(False)
            curses.nocbreak()
            curses.echo()
            with contextlib.suppress(curses.error):
                curses.curs_set(1)
        finally:
            curses.endwin()

    # -- main loop -----------------------------------------------------------

    def _loop(self) -> bool:
        import curses

        assert self.scr is not None
        while True:
            if _should_exit(self.stop):
                return False
            if self.hub.suspend_requested() and not self._hand_over():
                return False
            self._draw()
            key = self.scr.getch()  # waits up to POLL_MS
            if key == -1 or key == curses.KEY_RESIZE:
                continue  # the next pass redraws at the new size
            if key in (ord("q"), ord("Q")):
                return True
            self._on_key(key)

    def _hand_over(self) -> bool:
        """Give the terminal to a prompt and take it back once the prompt is done.

        Viewer side of `StreamHub.suspended`: leave curses first, then
        acknowledge, so the prompt only starts once the screen is in shell
        mode. We take the terminal back only when `reclaim_terminal` says so,
        never merely because `suspend_requested()` went False, since a second
        prompt may start straight after the first and must not be drawn over.

        Returns False if we were told to stop while released; the terminal is
        then already restored.
        """
        import curses

        assert self.scr is not None
        curses.endwin()
        self.hub.acknowledge_suspend()
        if not self._wait_reclaim():
            return False
        # Back into program mode with a full repaint: the prompt drew over the
        # whole screen and curses' idea of what is on it is stale. Keys typed
        # into the prompt may also still be queued; they were not meant for us.
        self.scr.clearok(True)
        self.scr.refresh()
        curses.flushinp()
        return True

    def _wait_reclaim(self) -> bool:
        """Block until the hub hands the terminal back. False if told to stop first."""
        while not self.hub.reclaim_terminal(timeout=_RECLAIM_POLL):
            if _should_exit(self.stop):
                return False
        return True

    def _on_key(self, key: int) -> None:
        import curses

        if key in (ord("f"), ord("F")):
            self.follow = not self.follow
            return
        if key in (ord("t"), ord("T")):
            # Changes what is drawn, not which task is watched, so unlike a
            # manual selection this leaves follow mode alone.
            self.show_tools = not self.show_tools
            return
        ids = self.hub.task_ids()
        if not ids:
            return
        if ord("1") <= key <= ord("9"):
            index = key - ord("1")
            if index >= len(ids):
                return
        elif key in (curses.KEY_LEFT, curses.KEY_RIGHT):
            step = -1 if key == curses.KEY_LEFT else 1
            current = ids.index(self.selected) if self.selected in ids else 0
            index = (current + step) % len(ids)
        else:
            return
        self.selected = ids[index]
        # Choosing a task by hand means the user wants to stay on it; follow
        # would otherwise yank them away on the next redraw.
        self.follow = False

    # -- drawing -------------------------------------------------------------

    def _draw(self) -> None:
        assert self.scr is not None
        snaps = self.hub.snapshots()
        current = self._pick(snaps)
        if current is not None:
            self.hub.mark_seen(current.task_id)
        rows, cols = self.scr.getmaxyx()
        self.scr.erase()
        if rows >= 1:
            self._draw_tabs(snaps, cols)
        if rows >= 2:
            self._draw_info(current, snaps, 1, cols)
        if rows >= 4:
            self._draw_body(current, 2, rows - 3, cols)
        if rows >= 3:
            legend = _legend(self.follow, self.unicode, self.show_tools)
            self._put(rows - 1, 0, legend, cols, self._dim())
        self.scr.refresh()

    def _pick(self, snaps: Sequence[TaskStream]) -> TaskStream | None:
        """Resolve the selected task, applying follow mode. Updates `self.selected`."""
        if not snaps:
            return None
        if self.follow:
            chosen = max(snaps, key=lambda s: s.updated_at)
        else:
            chosen = next((s for s in snaps if s.task_id == self.selected), snaps[0])
        self.selected = chosen.task_id
        return chosen

    def _draw_tabs(self, snaps: Sequence[TaskStream], cols: int) -> None:
        """Draw the tab bar, with a separator wherever the wave changes.

        Tabs stay in registration order and numbered by it, because that is the
        order `_on_key` indexes: the tab labelled 3 is always the one `3`
        selects, and left/right walk the bar left to right. So a wave forms one
        visible group exactly when its tasks are contiguous in the spec (the
        usual case); a spec that interleaves waves gets a separator at each
        change rather than a reordered bar whose numbers no longer match keys.

        Separators are shown only when the whole bar fits, separators included.
        Once it does not, something is going to be cut off, and every column a
        separator takes is one a task id could have used: an id you cannot read
        costs more than a grouping you cannot see. Without separators the bar
        scrolls exactly as it always has.
        """
        import curses

        if not snaps:
            self._put(
                0,
                0,
                " waiting for tasks…" if self.unicode else " waiting for tasks...",
                cols,
                self._dim(),
            )
            return
        labels: list[str] = []
        for number, snap in enumerate(snaps, start=1):
            prefix = f"{number}:" if number <= 9 else ""
            unseen = "*" if snap.unseen and snap.task_id != self.selected else " "
            labels.append(f" {prefix}{snap.task_id} {self._glyph(snap.status)}{unseen}")
        chosen = next(i for i, s in enumerate(snaps) if s.task_id == self.selected)
        sep = _WAVE_SEP if self.unicode else _WAVE_SEP_ASCII
        # Indexes of tabs that start a new wave and so get a separator before them.
        wave_starts = {i for i in range(1, len(snaps)) if snaps[i].wave != snaps[i - 1].wave}
        if sum(len(t) for t in labels) + len(sep) * len(wave_starts) > cols:
            wave_starts = set()
        # Scroll the bar so the selected tab is always visible. With separators
        # on, the whole bar fits, so this never scrolls and they need no room.
        first = 0
        while first < chosen and sum(len(t) for t in labels[first : chosen + 1]) + 1 > cols:
            first += 1
        x = 0
        if first > 0:
            x = self._put(0, x, "<", cols, self._dim())
        for i in range(first, len(snaps)):
            if x >= cols:
                break
            if i in wave_starts:
                x = self._put(0, x, sep, cols, self._dim())
            snap, label = snaps[i], labels[i]
            base = curses.A_REVERSE | curses.A_BOLD if i == chosen else curses.A_NORMAL
            # Draw the glyph in its status colour, the rest in the tab style.
            glyph_at = len(label) - 2
            x = self._put(0, x, label[:glyph_at], cols, base)
            x = self._put(0, x, label[glyph_at], cols, base | self._status_attr(snap.status))
            x = self._put(0, x, label[glyph_at + 1 :], cols, base | curses.A_BOLD)

    def _draw_info(
        self, current: TaskStream | None, snaps: Sequence[TaskStream], row: int, cols: int
    ) -> None:
        if current is None:
            return
        sep = " · " if self.unicode else " | "
        parts = [current.task_id]
        if current.agent or current.model:
            parts.append(" ".join(p for p in (current.agent, current.model) if p))
        parts.append(current.status)
        waiting = _waiting_for(current, snaps)
        if waiting:
            parts.append("waiting for: " + ", ".join(waiting))
        if self.follow:
            parts.append("follow")
        x = self._put(row, 0, " " + sep.join(parts) + " ", cols, self._accent())
        self._put(row, x, "─" * cols if self.unicode else "-" * cols, cols, self._dim())

    def _draw_body(self, current: TaskStream | None, top: int, height: int, cols: int) -> None:
        if current is None or height <= 0 or cols <= 0:
            return
        # Filter here, at draw time, rather than in the hub: the buffer keeps
        # every line, so toggling `t` reveals tool lines already received, not
        # just future ones. Only "tool" is hidden; "note" lines carry run-level
        # events such as a failure's reason and must always stay visible.
        source = [line.text for line in current.lines if self.show_tools or line.kind != "tool"]
        # `pending` is always open model text: tool events are flushed whole.
        if current.pending:
            source.append(current.pending)
        # Wrap bottom-up and stop once the window is full: the scrollback can be
        # hundreds of lines and only the newest screenful is ever visible.
        rows: list[str] = []
        for line in reversed(source):
            rows[:0] = _wrap(_sanitize(line), cols)
            if len(rows) >= height:
                break
        for offset, text in enumerate(rows[-height:]):
            self._put(top + offset, 0, text, cols, 0)

    # -- helpers -------------------------------------------------------------

    def _put(self, row: int, x: int, text: str, cols: int, attr: int) -> int:
        """addstr clipped to the window; returns the column after the text.

        curses raises when a write ends in the bottom-right cell (the cursor
        cannot advance past it), and on some terminals for odd widths, so
        drawing errors are swallowed: a missing cell beats a crashed viewer.
        """
        import curses

        assert self.scr is not None
        room = cols - x
        if room <= 0 or not text:
            return x
        text = text[:room]
        with contextlib.suppress(curses.error):
            self.scr.addstr(row, x, text, attr)
        return x + len(text)

    def _glyph(self, status: str) -> str:
        return self.glyphs.get(status, "?")

    def _status_attr(self, status: str) -> int:
        pair = {"ok": _PAIR_OK, "failed": _PAIR_FAILED, "running": _PAIR_RUNNING}.get(status)
        return self._color(pair) if pair is not None else 0

    def _accent(self) -> int:
        import curses

        return self._color(_PAIR_ACCENT) | curses.A_BOLD

    def _dim(self) -> int:
        import curses

        return curses.A_DIM

    @staticmethod
    def _color(pair: int) -> int:
        import curses

        return curses.color_pair(pair) if curses.has_colors() else 0


def _waiting_for(task: TaskStream, snaps: Sequence[TaskStream]) -> list[str]:
    """Dependencies of a not-yet-started `task` that have not finished, in spec order.

    Empty once the task has started (what it waited for no longer matters) and
    when every dependency is done; a task that is still pending at that point is
    waiting for a free slot, not for another task. A dependency the hub has
    never heard of counts as unfinished: nothing says it has run.
    """
    if task.status != "pending" or not task.depends_on:
        return []
    status = {s.task_id: s.status for s in snaps}
    return [dep for dep in task.depends_on if status.get(dep) not in _FINISHED]


def _sanitize(line: str) -> str:
    """Make a captured line safe to addstr: expand tabs, neutralize control chars.

    Task output can contain ANSI escapes or other control bytes. Passed through,
    they would move the real cursor behind curses' back and corrupt the screen.
    """
    line = line.expandtabs(4)
    if line.isprintable():
        return line
    return "".join(c if c.isprintable() else "?" for c in line)


def _wrap(line: str, width: int) -> list[str]:
    """Hard-wrap at exactly `width` columns, keeping whitespace; "" stays one row.

    Output is mostly code and logs, where reflowing on word boundaries would
    misrepresent indentation, so this cuts mid-word like a terminal does.
    Width is counted in code points, so wide (CJK) characters overflow a little.
    """
    if not line:
        return [""]
    return [line[i : i + width] for i in range(0, len(line), width)]
