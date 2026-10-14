"""Terminal viewer for a running DAG: the run's PERT chart over the selected task's output.

This is the drawing half of `stream`. It only reads from a `StreamHub` (through
`snapshots`, plus the bookkeeping calls `mark_seen` and the handover protocol)
and never publishes, so turning it off changes nothing about a run.

Layout, top to bottom:

- the run drawn as a PERT chart (see `_pert_lines`): every task with its
  number, id, status glyph, `*` when it has output the user has not looked at
  yet, and brackets around the selected one, grouped into one column per wave.
  It is the header, so it is always there, but its height is budgeted (see
  `_chart_budget` and `_header_lines`) and `g` collapses it to one line per wave;
- an info row for the selected task (agent, model, status, its wave and the
  tasks sharing it, follow mode, and for a task that has not started, the
  dependencies it is still waiting for);
- the selected task's scrollback plus its in-progress `pending` line, newest at
  the bottom, in whatever rows remain: rendered as markdown through
  `richtext.render` (see `_Viewer._draw_body`), or with `m` the raw text hard
  wrapped to the window width;
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
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anyio
import anyio.from_thread
import anyio.to_thread

from . import richtext
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

# The one table of status glyphs, as (unicode, ascii), read through
# `_status_glyph`. Pending is `▶`, the task queued to play next; running is `■`,
# the one occupying its slot now. The two must stay distinct at a glance, so a
# task that is actually doing something stands out from one that is merely waiting.
#
# Every glyph must be exactly one column wide: the chart pads columns by `len`,
# so a two-cell glyph shifts everything after it. That is why running is U+25A0
# BLACK SQUARE and not the real pause glyph U+23F8: that one is missing from many
# terminal fonts and often drawn two columns wide, which breaks the chart's
# alignment. `■` is deliberate, not a placeholder awaiting a "proper" pause sign.
_GLYPHS: dict[str, tuple[str, str]] = {
    "pending": ("▶", ">"),
    "running": ("■", "#"),
    "ok": ("✓", "+"),
    "failed": ("✗", "x"),
    "skipped": ("–", "-"),
}
_UNICODE_GLYPHS = {status: pair[0] for status, pair in _GLYPHS.items()}
_ASCII_GLYPHS = {status: pair[1] for status, pair in _GLYPHS.items()}


def _status_glyph(status: str, unicode: bool) -> str:
    """The glyph for `status`, or `?` for a status this module does not know.

    The fallback is a single character too, so an unexpected status costs
    nothing worse than a question mark in otherwise aligned columns.
    """
    return (_UNICODE_GLYPHS if unicode else _ASCII_GLYPHS).get(status, "?")


def _locale_is_utf8() -> bool:
    """Whether the C library's locale - the one ncurses encodes with - is UTF-8."""
    import locale

    try:
        codeset = locale.nl_langinfo(locale.CODESET)
    except (AttributeError, ValueError):  # no nl_langinfo on this platform
        return True
    return codeset.lower().replace("-", "") == "utf8"


# Colour pair number, allocated in `_Viewer._setup` when the terminal has colours.
_PAIR_ACCENT = 1

# Pairs for span colours are allocated on demand after the accent pair, at most
# this many. `richtext` only ever names the eight basic colours, so eight covers
# every colour it can produce; the cap is what keeps a future richer palette
# from walking off the end of the terminal's pair table (often 64, sometimes 8).
# Past the cap, or past `COLOR_PAIRS`, a span is drawn without colour.
_MAX_SPAN_PAIRS = 8


def _legend(
    follow: bool,
    unicode: bool,
    show_tools: bool,
    expanded: bool = True,
    markdown: bool = True,
) -> str:
    """Return the key legend shown on the bottom row.

    The `f` entry names what pressing `f` will do next, not the current mode:
    the current mode is already shown on the info row, and a legend that said
    "follow" while following would read as an instruction to do what is already
    happening. The label is padded to the width of the longer word so `q quit`
    stays put when the mode flips. The `t` entry follows the same rule for the
    same reasons: it says what `t` will do, and is padded to its longer label.
    So does `g`, which collapses an expanded chart and expands a collapsed one,
    and `m`, which offers `raw` while the body is rendered as markdown and
    `render` while it shows raw text. `expanded` and `markdown` default to True
    because that is how a viewer starts.

    The line must fit 80 columns. At its widest (ASCII arrows, every label
    padded) it is exactly 80, so a new entry needs shorter wording elsewhere.
    Kept free of `self` and curses so it can be tested without a terminal.
    """
    arrows = "←/→" if unicode else "<-/->"
    toggle = ("unfollow" if follow else "follow").ljust(len("unfollow"))
    tools = ("hide tools" if show_tools else "tools").ljust(len("hide tools"))
    chart = ("collapse" if expanded else "expand").ljust(len("collapse"))
    body = ("raw" if markdown else "render").ljust(len("render"))
    return f"{arrows} or 1-9 select   f {toggle}   t {tools}   g {chart}   m {body}   q quit"


# The chart's height budget, see `_chart_budget`. It gets a third of the rows it
# shares with the body, but never fewer than `_CHART_FLOOR`, and never so many
# that the body drops below `_MIN_BODY` while the terminal has room for both.
_CHART_FLOOR = 3
_MIN_BODY = 3


def _chart_budget(rows: int) -> int:
    """How many rows the chart may use on a terminal `rows` tall.

    The chart shares the screen with the body; the info row and the legend are
    fixed. A third of the shared rows, at least three so the chart is never a
    sliver, keeps a wave of eight tasks from pushing the body off a short
    terminal. Where three would eat the body, the body wins down to one chart
    row, since output is what the user is there to read and the header can
    always shrink to a summary line. On a terminal too short for any body the
    chart still gets its one row: it is the header, and it shows the selection.
    """
    # Info and legend are only drawn once there is a row for them past the chart's.
    fixed = min(max(rows - 1, 0), 2)
    shared = rows - fixed
    if shared <= 0:
        return 0
    share = max(_CHART_FLOOR, shared // 3)
    return max(1, min(share, shared - _MIN_BODY))


def _header_lines(
    snaps: Sequence[TaskStream],
    selected: str,
    width: int,
    height: int,
    unicode: bool,
    expanded: bool,
) -> list[str]:
    """The chart as drawn in the header: at most `height` lines, none wider than `width`.

    Expanded, this is the full `_pert_lines` chart when it fits the height;
    collapsed, or when the full chart is too tall, it is `_pert_lines`' compact
    one-line-per-wave form; when even that is too tall, one summary line. Each
    step down keeps the selection visible. Width degradation stays inside
    `_pert_lines`, which only ever narrows a chart into fewer lines.
    """
    if height <= 0 or width <= 0:
        return []
    if not snaps:
        return [(" waiting for tasks…" if unicode else " waiting for tasks...")[:width]]

    def fits(lines: list[str]) -> bool:
        # Too wide for even one line per wave, `_pert_lines` says so in a line of
        # its own. That fits any height but shows neither the selection nor the
        # run, so it counts as not fitting: the summary line is built to be cut.
        too_narrow = len(lines) == 1 and any(t.startswith(lines[0]) for t in _TOO_NARROW)
        return len(lines) <= height and not too_narrow

    if expanded:
        full = _pert_lines(snaps, selected, width, unicode)
        if fits(full):
            return full
    compact = _pert_lines(snaps, selected, width, unicode, compact=True)
    if fits(compact):
        return compact
    return [_summary_line(snaps, selected, width, unicode)]


def _summary_line(snaps: Sequence[TaskStream], selected: str, width: int, unicode: bool) -> str:
    """The whole run on one line, for a header with room for nothing more.

    The selected task comes first, numbered as its key, because the line is cut
    at `width` and the selection is the one thing the header must not lose.
    Then the run's size and a count per status, in the chart's glyphs, and how
    many other tasks have output waiting to be looked at.
    """
    number = next((n for n, s in enumerate(snaps, start=1) if s.task_id == selected), None)
    parts: list[str] = []
    if number is not None:
        status = snaps[number - 1].status
        parts.append(f"[{number} {selected} {_status_glyph(status, unicode)}]")
    waves = len({s.wave for s in snaps})
    parts.append(f"{len(snaps)} tasks, {waves} wave{'s' if waves != 1 else ''}")
    counts: dict[str, int] = {}
    for snap in snaps:
        counts[snap.status] = counts.get(snap.status, 0) + 1
    # In `_GLYPHS` order rather than first-seen, so the counts do not trade places mid-run.
    rank = {st: i for i, st in enumerate(_GLYPHS)}
    ordered = sorted(counts.items(), key=lambda item: rank.get(item[0], len(rank)))
    parts.append(" ".join(f"{_status_glyph(st, unicode)}{n}" for st, n in ordered))
    unseen = sum(1 for s in snaps if s.unseen and s.task_id != selected)
    if unseen:
        parts.append(f"*{unseen} unseen")
    sep = " · " if unicode else " | "
    return sep.join(parts)[:width]


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
        # Whether the header chart is drawn in full (still within its height
        # budget) or collapsed to one line per wave. `g` flips it.
        self.chart_expanded = True
        # Whether the body is rendered as markdown or shown raw. `m` flips it.
        self.markdown = True
        # One-entry render cache, see `_rendered`.
        self._render_key: tuple[str, int] | None = None
        self._render_rows: list[list[richtext.Span]] = []
        # Span colour name -> pair number, or None for a colour that could not
        # get a pair, so a full table is not retried on every redraw.
        self._span_pairs: dict[str, int | None] = {}
        self._background = -1
        self.scr: curses.window | None = None
        self.unicode = (sys.stdout.encoding or "").lower().replace("-", "") == "utf8"

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
        # stdout's encoding is not what curses writes in. Under `LC_ALL=C`,
        # Python's UTF-8 mode reports utf-8 while ncurses works in the C
        # library's ASCII locale and turns every non-ASCII glyph into a blank -
        # running and pending then look identical. Trust the locale curses uses.
        if self.unicode and not _locale_is_utf8():
            self.unicode = False
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
            curses.init_pair(_PAIR_ACCENT, curses.COLOR_CYAN, background)
            # Span pairs are made later, on demand, against the same background.
            self._background = background

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
        if key in (ord("g"), ord("G")):
            # Like `t`: a different view of the same task, so neither the
            # selection nor follow mode moves.
            self.chart_expanded = not self.chart_expanded
            return
        if key in (ord("m"), ord("M")):
            # Same family again: how the body is drawn, not what is selected.
            self.markdown = not self.markdown
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
        header = _header_lines(
            snaps,
            self.selected or "",
            cols,
            _chart_budget(rows),
            self.unicode,
            self.chart_expanded,
        )
        for offset, text in enumerate(header):
            self._put(offset, 0, text, cols, 0)
        info = len(header)
        if info < rows:
            self._draw_info(current, snaps, info, cols)
        if rows >= 3:
            # The body takes whatever the header and info row leave above the legend.
            self._draw_body(current, info + 1, rows - info - 2, cols)
            legend = _legend(
                self.follow, self.unicode, self.show_tools, self.chart_expanded, self.markdown
            )
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
        # `wave` is 0-based on the stream; people count waves from 1.
        parts.append(f"wave {current.wave + 1} of {max(s.wave for s in snaps) + 1}")
        alongside = _runs_with(current, snaps)
        if alongside:
            parts.append("runs with: " + ", ".join(alongside))
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
        if not self.markdown:
            self._draw_raw(source, top, height, cols)
            return
        # Bottom-up as in the raw path, but collecting source lines rather than
        # rows, and until twice the window: markdown joins a paragraph's soft
        # line breaks and drops fence lines, so a document that hard-wraps to
        # exactly `height` rows usually renders shorter. The slack is cheap
        # (linear in what is collected) and also gives the parser some context
        # above the first visible row.
        budget = 2 * height
        used = 0
        start = len(source)
        while start > 0 and used < budget:
            start -= 1
            source[start] = _sanitize(source[start])
            used += len(_wrap(source[start], cols))
        visible = source[start:]
        # Cut off inside a code fence, the slice would render the fenced text as
        # prose and its closing fence as an opener that swallows everything
        # after. Re-open the fence instead, with its original info string.
        opener = _open_fence(source[:start])
        if opener is not None:
            visible.insert(0, _sanitize(opener))
        rendered = self._rendered("\n".join(visible), cols)
        for offset, spans in enumerate(rendered[-height:]):
            x = 0
            for span in spans:
                if x >= cols:
                    break  # the rest of the row starts past the edge
                x = self._put(top + offset, x, span.text, cols, self._span_attr(span))

    def _draw_raw(self, source: list[str], top: int, height: int, cols: int) -> None:
        """The body as plain text, hard wrapped: what `m` switches to.

        Wraps bottom-up and stops once the window is full: the scrollback can be
        hundreds of lines and only the newest screenful is ever visible.
        """
        rows: list[str] = []
        for line in reversed(source):
            rows[:0] = _wrap(_sanitize(line), cols)
            if len(rows) >= height:
                break
        for offset, text in enumerate(rows[-height:]):
            self._put(top + offset, 0, text, cols, 0)

    def _rendered(self, document: str, width: int) -> list[list[richtext.Span]]:
        """`richtext.render(document, width)`, reused while neither argument changes.

        A render is a CommonMark parse plus Pygments highlighting, and the body
        is redrawn ten times a second. Between output updates the document is
        identical, so one entry is enough: a key-press or an idle tick redraws
        from the cache. The key is the whole document, not just its last line,
        since a scroll, a toggle of `t` or a new selection changes it elsewhere.
        """
        key = (document, width)
        if key != self._render_key:
            self._render_rows = richtext.render(document, width)
            self._render_key = key
        return self._render_rows

    def _span_attr(self, span: richtext.Span) -> int:
        import curses

        attr = 0
        if span.bold:
            attr |= curses.A_BOLD
        if span.dim:
            attr |= curses.A_DIM
        if span.underline:
            attr |= curses.A_UNDERLINE
        if span.reverse:
            attr |= curses.A_REVERSE
        if span.colour is not None:
            attr |= self._span_colour(span.colour)
        return attr

    def _span_colour(self, name: str) -> int:
        """The colour attribute for a span colour name, allocating its pair on first use.

        Pairs come after the accent pair, against the same background, one per
        name. Once `_MAX_SPAN_PAIRS` are taken, or the terminal has no pairs
        left, or the name is not a curses colour, the answer is 0: the span is
        drawn without colour, which loses nothing but the tint.
        """
        if name not in self._span_pairs:
            self._span_pairs[name] = self._allocate_pair(name)
        pair = self._span_pairs[name]
        return 0 if pair is None else self._color(pair)

    def _allocate_pair(self, name: str) -> int | None:
        import curses

        if not curses.has_colors():
            return None
        foreground = getattr(curses, f"COLOR_{name.upper()}", None)
        if not isinstance(foreground, int):
            return None
        taken = sum(1 for pair in self._span_pairs.values() if pair is not None)
        number = _PAIR_ACCENT + 1 + taken
        if taken >= _MAX_SPAN_PAIRS or number >= curses.COLOR_PAIRS:
            return None
        try:
            curses.init_pair(number, foreground, self._background)
        except curses.error:
            return None
        return number

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


def _runs_with(current: TaskStream, snaps: Sequence[TaskStream]) -> list[str]:
    """Ids of the other tasks in `current`'s wave, in registration order.

    These are the tasks the DAG allows to run at the same time as `current`:
    the chart's column says so too, but not once it has collapsed to a summary
    line, and the info row should not depend on the header's shape. It says
    nothing about whether they are running right now: that is the glyphs' job,
    and a wave whose concurrency is capped may run its members one at a time.
    """
    return [s.task_id for s in snaps if s.wave == current.wave and s.task_id != current.task_id]


def _sanitize(line: str) -> str:
    """Make a captured line safe to addstr: expand tabs, neutralize control chars.

    Task output can contain ANSI escapes or other control bytes. Passed through,
    they would move the real cursor behind curses' back and corrupt the screen.
    """
    line = line.expandtabs(4)
    if line.isprintable():
        return line
    return "".join(c if c.isprintable() else "?" for c in line)


def _open_fence(lines: Sequence[str]) -> str | None:
    """The opening line of the code fence still open after `lines`, or None.

    A simplified CommonMark fence: up to three spaces, then three or more
    backticks or tildes; it is closed by a line of at least as many of the same
    character and nothing else. Indented code and fences inside lists are not
    tracked, and a miss only costs the rendering of the window's top rows.
    Linear in `lines`, which is bounded by the hub's scrollback.
    """
    opener: str | None = None
    marker = ""
    for line in lines:
        stripped = line.lstrip(" ")
        if len(line) - len(stripped) > 3 or stripped[:1] not in ("`", "~"):
            continue
        char = stripped[0]
        run = len(stripped) - len(stripped.lstrip(char))
        if run < 3:
            continue
        if opener is None:
            # A backtick fence's info string may not contain backticks.
            if char == "`" and "`" in stripped[run:]:
                continue
            opener, marker = line, stripped[:run]
        elif char == marker[0] and run >= len(marker) and not stripped[run:].strip():
            opener, marker = None, ""
    return opener


def _wrap(line: str, width: int) -> list[str]:
    """Hard-wrap at exactly `width` columns, keeping whitespace; "" stays one row.

    Output is mostly code and logs, where reflowing on word boundaries would
    misrepresent indentation, so this cuts mid-word like a terminal does.
    Width is counted in code points, so wide (CJK) characters overflow a little.
    """
    if not line:
        return [""]
    return [line[i : i + width] for i in range(0, len(line), width)]


# -- PERT chart ----------------------------------------------------------------

# Box-drawing junction for a lane cell, keyed by which of its four sides carry
# a line: (up, down, left, right). Every combination a lane can produce is
# listed; a lane cell always has at least two sides.
_JUNCTIONS = {
    (False, False, True, True): "─",
    (True, True, False, False): "│",
    (False, True, True, False): "┐",
    (True, False, True, False): "┘",
    (True, True, True, False): "┤",
    (False, True, False, True): "┌",
    (True, False, False, True): "└",
    (True, True, False, True): "├",
    (False, True, True, True): "┬",
    (True, False, True, True): "┴",
    (True, True, True, True): "┼",
}

# Width of the blank gap between columns when no connectors are drawn in it.
_BARE_GAP = 2

_TOO_NARROW = ("terminal too narrow for the chart", "too narrow for the chart", "too narrow")


@dataclass(frozen=True)
class _Block:
    """One complete-bipartite bundle of edges between two adjacent columns.

    Every row in `sources` (left column) feeds every row in `targets` (right
    column), so the whole bundle can share one vertical trunk without the
    drawing claiming an edge that does not exist.
    """

    sources: frozenset[int]
    targets: frozenset[int]
    lane: int
    top: int
    bottom: int


def _route(feeds: Sequence[frozenset[int]]) -> list[_Block] | None:
    """Lay out the edges between two adjacent columns, or None if they will not draw cleanly.

    `feeds[r]` is the set of left-column rows that right-column row `r`
    depends on. Targets with identical sources share a trunk; trunks whose row
    spans overlap get separate lanes, left to right. That covers fan-in,
    fan-out, several side-by-side chains and the common "two groups feeding two
    tasks" shape.

    It gives up (returns None) rather than draw something misleading when a
    source feeds two different bundles, since joined lines would then imply
    edges that do not exist, or when a horizontal line would have to cross a
    trunk. The caller annotates instead. Anything smarter is general edge
    routing, which is deliberately out of scope.
    """
    by_sources: dict[frozenset[int], set[int]] = {}
    for row, sources in enumerate(feeds):
        if sources:
            by_sources.setdefault(sources, set()).add(row)
    claimed: set[int] = set()
    for sources in by_sources:
        if claimed & sources:
            return None
        claimed |= sources
    bundles = sorted(
        ((s, frozenset(t)) for s, t in by_sources.items()), key=lambda b: min(b[0] | b[1])
    )
    lane_ends: list[int] = []
    blocks: list[_Block] = []
    for sources, targets in bundles:
        rows = sources | targets
        top, bottom = min(rows), max(rows)
        lane = next((i for i, end in enumerate(lane_ends) if end < top), len(lane_ends))
        if lane == len(lane_ends):
            lane_ends.append(bottom)
        else:
            lane_ends[lane] = bottom
        blocks.append(_Block(sources, targets, lane, top, bottom))
    # A source's line runs from the left edge to its lane, crossing every lane
    # left of it; a target's runs from its lane to the right edge. Either may
    # only cross a lane at a row where that lane is empty.
    for block in blocks:
        for row in block.sources:
            if any(o.lane < block.lane and o.top <= row <= o.bottom for o in blocks):
                return None
        for row in block.targets:
            if any(o.lane > block.lane and o.top <= row <= o.bottom for o in blocks):
                return None
    return blocks


def _draw_gap(blocks: Sequence[_Block], height: int, unicode: bool) -> list[str]:
    """Render routed blocks as `height` equal-width strings, one per row."""
    lanes = 1 + max(b.lane for b in blocks)
    width = 2 * lanes + 2  # lead-in, lanes one apart, two-cell lead-out
    h = "─" if unicode else "-"
    out: list[str] = []
    for row in range(height):
        cells = [" "] * width
        for b in blocks:
            x = 1 + 2 * b.lane
            if row in b.sources:
                cells[:x] = [h] * x
            if row in b.targets:
                cells[x + 1 :] = [h] * (width - x - 1)
        for b in blocks:
            if b.top <= row <= b.bottom:
                sides = (row > b.top, row < b.bottom, row in b.sources, row in b.targets)
                if unicode:
                    cells[1 + 2 * b.lane] = _JUNCTIONS.get(sides, "┼")
                else:
                    straight = {(False, False, True, True): "-", (True, True, False, False): "|"}
                    cells[1 + 2 * b.lane] = straight.get(sides, "+")
        out.append("".join(cells))
    return out


def _pert_lines(
    snaps: Sequence[TaskStream],
    selected: str,
    width: int,
    unicode: bool,
    compact: bool = False,
) -> list[str]:
    """The run drawn as a PERT chart: plain text lines, none longer than `width`.

    One column per wave, left to right; within a column, tasks in registration
    order. Each node is `<number> <id> <status glyph>`, numbered in
    registration order as the 1-9 keys and arrows select, and the selected one
    is wrapped in brackets so it stands out without colour. Any other node with
    output the user has not looked at gets a `*` in the cell its closing
    bracket would use, so marking it never moves anything either. Each column
    has a `wave N` header and a rule beneath it.

    Edges are drawn only between adjacent columns, with box-drawing characters
    (`-`, `|`, `+` in ASCII), and only where `_route` can do so without
    implying an edge that does not exist. A dependency that skips a column is
    *not* routed: the dependent is annotated with the node numbers of what it
    waits on, as in `3 verify ▶ ←1`. So is every dependent between two columns
    whose edges `_route` declines to draw. This is a decision, not an
    oversight: general edge routing in a terminal costs far more than it is
    worth, and a number you can match against another node is unambiguous.

    When the chart does not fit, it degrades in this order, each step taken
    only if the previous one is still too wide: drop the connectors, drop the
    wave headers, list each wave on one line, and finally say plainly that the
    terminal is too narrow. Every step is a complete drawing; nothing is ever
    clipped mid-shape. `compact` starts at the one-line-per-wave step, which is
    how the header collapses; height is otherwise the caller's problem.

    Pure: no curses, no colour, no state, so tests and bug reports see exactly
    what the screen shows.
    """
    if not snaps or width <= 0:
        return []
    waves = sorted({s.wave for s in snaps})
    column_of = {wave: c for c, wave in enumerate(waves)}
    columns: list[list[int]] = [[] for _ in waves]
    for i, snap in enumerate(snaps):
        columns[column_of[snap.wave]].append(i)
    col = [column_of[s.wave] for s in snaps]
    row = {i: r for members in columns for r, i in enumerate(members)}
    index_of = {s.task_id: i for i, s in enumerate(snaps)}
    # Dependencies we know about and that sit in an earlier column. Anything
    # else (unknown ids, a malformed same-wave edge) cannot be drawn either way.
    deps = [
        [index_of[d] for d in s.depends_on if d in index_of and col[index_of[d]] < col[i]]
        for i, s in enumerate(snaps)
    ]

    # Tab numbers to print beside a node instead of drawing its edges: always
    # for skipped columns, and for adjacent ones `_route` will not draw.
    notes = [{d for d in deps[i] if col[d] < col[i] - 1} for i in range(len(snaps))]
    gaps: list[list[_Block] | None] = []
    for c in range(len(columns) - 1):
        feeds = [frozenset(row[d] for d in deps[i] if col[d] == c) for i in columns[c + 1]]
        blocks = _route(feeds)
        if blocks is None:
            for i in columns[c + 1]:
                notes[i].update(d for d in deps[i] if col[d] == c)
        gaps.append(blocks or None)

    arrow = "←" if unicode else "<-"
    texts: dict[int, str] = {}
    for members in columns:
        num_w = max(len(str(i + 1)) for i in members)
        id_w = max(len(snaps[i].task_id) for i in members)
        for i in members:
            text = f"{str(i + 1).rjust(num_w)} {snaps[i].task_id.ljust(id_w)} "
            text += _status_glyph(snaps[i].status, unicode)
            if notes[i]:
                text += f" {arrow}" + ",".join(str(d + 1) for d in sorted(notes[i]))
            texts[i] = text

    # (connectors, headers) for each full-detail attempt; compact skips them all.
    grids: tuple[tuple[bool, bool], ...] = (
        () if compact else ((True, True), (False, True), (False, False))
    )
    for connectors, headers in grids:
        grid = _grid(columns, texts, gaps, waves, selected, snaps, connectors, headers, unicode)
        if max(len(line) for line in grid) <= width:
            return grid

    def entry(i: int) -> str:
        task_id = snaps[i].task_id
        if task_id == selected:
            return f"[{task_id}]"
        return f"{task_id}*" if snaps[i].unseen else task_id

    listing = [
        f"wave {wave + 1}: " + ", ".join(entry(i) for i in members)
        for wave, members in zip(waves, columns, strict=True)
    ]
    if max(len(line) for line in listing) <= width:
        return listing
    return [next((t for t in _TOO_NARROW if len(t) <= width), _TOO_NARROW[-1][:width])]


def _grid(
    columns: Sequence[Sequence[int]],
    texts: dict[int, str],
    gaps: Sequence[Sequence[_Block] | None],
    waves: Sequence[int],
    selected: str,
    snaps: Sequence[TaskStream],
    connectors: bool,
    headers: bool,
    unicode: bool,
) -> list[str]:
    """One full-detail rendering of the chart for `_pert_lines`, at any width.

    Each node sits in a cell one wider than its text on both sides; the spare
    cells hold the selection brackets, or for an unselected node with unseen
    output a `*` on the right, so marking a task never moves anything.
    A node with an outgoing connector has the rest of its cell filled with rule
    so the line visibly starts at the node rather than floating in the gap.
    """
    h = "─" if unicode else "-"
    height = max(len(members) for members in columns)
    labels = [f"wave {wave + 1}" for wave in waves]
    text_w = [
        max([len(texts[i]) for i in members] + ([len(label)] if headers else []))
        for members, label in zip(columns, labels, strict=True)
    ]
    drawn: list[list[str]] = []
    feeding: list[set[int]] = []
    for gap in gaps:
        if connectors and gap:
            drawn.append(_draw_gap(gap, height, unicode))
            feeding.append({r for b in gap for r in b.sources})
        else:
            drawn.append([" " * _BARE_GAP] * height)
            feeding.append(set())

    lines: list[str] = []
    if headers:
        for row_labels in (labels, [h * len(label) for label in labels]):
            parts: list[str] = []
            for c, label in enumerate(row_labels):
                parts.append(" " + label.ljust(text_w[c]) + " ")
                if c < len(drawn):
                    parts.append(" " * len(drawn[c][0]))
            lines.append("".join(parts).rstrip())
    for r in range(height):
        parts = []
        for c, members in enumerate(columns):
            if r < len(members):
                i = members[r]
                if snaps[i].task_id == selected:
                    left, right = "[", "]"
                else:
                    left, right = " ", "*" if snaps[i].unseen else " "
                fill = h if c < len(feeding) and r in feeding[c] else " "
                parts.append(left + texts[i] + right + fill * (text_w[c] - len(texts[i])))
            else:
                parts.append(" " * (text_w[c] + 2))
            if c < len(drawn):
                parts.append(drawn[c][r])
        lines.append("".join(parts).rstrip())
    return lines
