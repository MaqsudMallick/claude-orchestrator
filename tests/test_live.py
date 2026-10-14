"""The live viewer's pure helpers: the key legend, line sanitising and wrapping,
and the "waiting for" and "runs with" lists on the info row.

The curses session itself (`_Viewer`) is not driven here: it needs a real
terminal, and this suite runs without one. Apart from a few checks on
`_Viewer`'s key-toggled state (constructor and `_on_key` only, which never open
a screen), everything below is a plain function of its arguments, so these tests
need neither a terminal nor a curses screen.

`live` is imported as a module and helpers are looked up on it per test, so a
helper that is missing fails only the tests that use it rather than the whole
file at collection time.
"""

import re
import sys
from collections.abc import Callable

import anyio
import pytest
from orchestrator import live
from orchestrator.stream import StreamHub, TaskStream

BOTH = [pytest.param(True, id="unicode"), pytest.param(False, id="ascii")]
FOLLOW = [pytest.param(True, id="follow-on"), pytest.param(False, id="follow-off")]
TOOLS = [pytest.param(True, id="tools-shown"), pytest.param(False, id="tools-hidden")]


def _has_word(text: str, phrase: str) -> bool:
    """`phrase` appears as whole words, so `f follow` does not match inside `f unfollow`
    and `q` does not match inside some other word."""
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None


# -- _legend -----------------------------------------------------------------


@pytest.mark.parametrize("show_tools", TOOLS)
@pytest.mark.parametrize("unicode", BOTH)
def test_legend_with_follow_on_offers_unfollow(unicode: bool, show_tools: bool) -> None:
    text = live._legend(follow=True, unicode=unicode, show_tools=show_tools)
    assert _has_word(text, "f unfollow")
    # The bug being fixed: the legend kept offering to turn follow on when it
    # already was. "f follow" as its own phrase must be gone, not merely joined.
    assert not _has_word(text, "f follow")


@pytest.mark.parametrize("show_tools", TOOLS)
@pytest.mark.parametrize("unicode", BOTH)
def test_legend_with_follow_off_offers_follow_and_not_unfollow(
    unicode: bool, show_tools: bool
) -> None:
    text = live._legend(follow=False, unicode=unicode, show_tools=show_tools)
    assert _has_word(text, "f follow")
    # `"follow" in text` is true in both states; only the negative tells them apart.
    assert "unfollow" not in text


@pytest.mark.parametrize("show_tools", TOOLS)
@pytest.mark.parametrize("unicode", BOTH)
def test_legend_differs_between_follow_states(unicode: bool, show_tools: bool) -> None:
    assert live._legend(follow=True, unicode=unicode, show_tools=show_tools) != live._legend(
        follow=False, unicode=unicode, show_tools=show_tools
    )


@pytest.mark.parametrize("show_tools", TOOLS)
@pytest.mark.parametrize("follow", FOLLOW)
def test_legend_ascii_variant_is_pure_ascii(follow: bool, show_tools: bool) -> None:
    text = live._legend(follow=follow, unicode=False, show_tools=show_tools)
    assert text.isascii(), [c for c in text if not c.isascii()]
    assert "<-/->" in text
    # And printable: a control byte would be just as bad on a dumb terminal.
    assert text.isprintable()


@pytest.mark.parametrize("show_tools", TOOLS)
@pytest.mark.parametrize("follow", FOLLOW)
def test_legend_unicode_variant_uses_arrow_glyphs(follow: bool, show_tools: bool) -> None:
    text = live._legend(follow=follow, unicode=True, show_tools=show_tools)
    assert "←/→" in text
    assert "<-/->" not in text
    assert text.isprintable()


@pytest.mark.parametrize("show_tools", TOOLS)
@pytest.mark.parametrize("unicode", BOTH)
@pytest.mark.parametrize("follow", FOLLOW)
def test_legend_mentions_select_keys_and_quit(
    follow: bool, unicode: bool, show_tools: bool
) -> None:
    text = live._legend(follow=follow, unicode=unicode, show_tools=show_tools)
    assert "1-9" in text
    assert "select" in text
    assert _has_word(text, "q")
    assert "quit" in text
    # One line: it is drawn into the single bottom row.
    assert "\n" not in text


@pytest.mark.parametrize("show_tools", TOOLS)
@pytest.mark.parametrize("unicode", BOTH)
@pytest.mark.parametrize("follow", FOLLOW)
def test_legend_is_deterministic(follow: bool, unicode: bool, show_tools: bool) -> None:
    assert live._legend(follow=follow, unicode=unicode, show_tools=show_tools) == live._legend(
        follow=follow, unicode=unicode, show_tools=show_tools
    )


def test_legend_does_not_depend_on_call_order() -> None:
    """No hidden state: the answer for one combination is unaffected by earlier calls."""
    combos = [(f, u, t) for f in (True, False) for u in (True, False) for t in (True, False)]
    first = {c: live._legend(*c) for c in combos}
    second = {c: live._legend(*c) for c in reversed(combos)}
    assert first == second
    assert len(set(first.values())) == 8  # every combination is distinguishable


def test_legend_needs_no_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Callable with curses unimportable and stdout not a TTY.

    A `None` entry in `sys.modules` makes `import curses` raise ImportError, so
    any attempt to reach for the terminal from inside `_legend` fails loudly.
    """
    monkeypatch.setitem(sys.modules, "curses", None)

    class _NotATty:
        encoding = "ascii"

        def isatty(self) -> bool:
            return False

        def write(self, _: str) -> int:
            raise AssertionError("_legend wrote to stdout")

    monkeypatch.setattr(sys, "stdout", _NotATty())
    assert _has_word(live._legend(follow=True, unicode=True, show_tools=True), "f unfollow")
    assert _has_word(live._legend(follow=False, unicode=False, show_tools=False), "f follow")
    assert _has_word(live._legend(follow=False, unicode=False, show_tools=False), "t tools")


# -- _legend: the tools toggle -------------------------------------------------


@pytest.mark.parametrize("unicode", BOTH)
@pytest.mark.parametrize("follow", FOLLOW)
def test_legend_with_tools_hidden_offers_tools_and_not_hide(follow: bool, unicode: bool) -> None:
    text = live._legend(follow=follow, unicode=unicode, show_tools=False)
    assert _has_word(text, "t tools")
    # Like `f`, the entry names what `t` will do. With tools already hidden,
    # offering to hide them would be wrong, so "hide" must not appear at all.
    assert "hide" not in text


@pytest.mark.parametrize("unicode", BOTH)
@pytest.mark.parametrize("follow", FOLLOW)
def test_legend_with_tools_shown_offers_hide_tools(follow: bool, unicode: bool) -> None:
    text = live._legend(follow=follow, unicode=unicode, show_tools=True)
    assert _has_word(text, "t hide tools")
    # "t tools" as its own phrase must be gone, not merely sit beside "hide".
    assert not _has_word(text, "t tools")


@pytest.mark.parametrize("unicode", BOTH)
@pytest.mark.parametrize("follow", FOLLOW)
def test_legend_differs_between_tools_states(follow: bool, unicode: bool) -> None:
    assert live._legend(follow=follow, unicode=unicode, show_tools=True) != live._legend(
        follow=follow, unicode=unicode, show_tools=False
    )


@pytest.mark.parametrize("unicode", BOTH)
@pytest.mark.parametrize("follow", FOLLOW)
def test_legend_tools_toggle_does_not_change_the_follow_entry(follow: bool, unicode: bool) -> None:
    """The two toggles are independent: flipping `t` leaves the `f` entry's wording alone."""
    phrase = "f unfollow" if follow else "f follow"
    for show_tools in (True, False):
        assert _has_word(
            live._legend(follow=follow, unicode=unicode, show_tools=show_tools), phrase
        )


# -- _Viewer state: tools hidden by default, `t` toggles ----------------------
#
# Only the constructor and `_on_key` are used: neither touches the terminal
# (`_on_key` imports the curses module for its key constants but never calls
# `initscr`). Drawing is not exercised; see the note on filtering below.


def _viewer() -> live._Viewer:
    return live._Viewer(StreamHub(), anyio.Event())


def test_viewer_hides_tool_lines_by_default() -> None:
    # The whole user-visible point of the change: a fresh viewer shows the
    # model's text without tool-call noise until the user asks for it.
    assert _viewer().show_tools is False


def test_fresh_viewer_legend_offers_to_show_tools() -> None:
    """What the user actually sees on first draw, built the way `_draw` builds it."""
    v = _viewer()
    text = live._legend(v.follow, v.unicode, v.show_tools)
    assert _has_word(text, "t tools")
    assert "hide" not in text


@pytest.mark.parametrize("key", ["t", "T"])
def test_t_toggles_show_tools_on_and_off(key: str) -> None:
    v = _viewer()
    v._on_key(ord(key))
    assert v.show_tools is True
    v._on_key(ord(key))
    assert v.show_tools is False


@pytest.mark.parametrize("key", ["t", "T"])
@pytest.mark.parametrize("follow", FOLLOW)
def test_t_leaves_follow_mode_alone(key: str, follow: bool) -> None:
    hub = StreamHub()
    hub.register("a", agent="", model="")
    hub.register("b", agent="", model="")
    v = live._Viewer(hub, anyio.Event())
    v.follow = follow
    v.selected = "b"
    v._on_key(ord(key))
    assert v.show_tools is True
    # Unlike picking a task by hand, revealing tools does not mean "stay here".
    assert v.follow is follow
    assert v.selected == "b"


def test_f_does_not_touch_show_tools() -> None:
    v = _viewer()
    v._on_key(ord("t"))
    v._on_key(ord("f"))
    assert v.show_tools is True
    assert v.follow is True


# Filtering itself (tool lines skipped unless `show_tools`, note lines always
# kept) is done inline in `_Viewer._draw_body`, which writes straight to the
# curses window through `_put`. There is no pure helper to call, so it is not
# tested here rather than tested through a fake screen.


# -- _sanitize ---------------------------------------------------------------


def test_sanitize_leaves_printable_text_untouched() -> None:
    line = "def f(x):  return x  # ünïcödé ✓ 漢字"
    assert live._sanitize(line) == line


def test_sanitize_expands_tabs_to_four_column_stops() -> None:
    assert live._sanitize("\tx") == "    x"
    assert live._sanitize("ab\tc") == "ab  c"  # to the next stop, not four more


def test_sanitize_neutralises_ansi_escapes() -> None:
    out = live._sanitize("\x1b[31mred\x1b[0m")
    assert "\x1b" not in out
    assert out.isprintable()
    # Visible text survives and each control character becomes one cell, so
    # nothing shifts the columns of what follows.
    assert out == "?[31mred?[0m"


@pytest.mark.parametrize("ctrl", ["\r", "\n", "\x00", "\x07", "\x08", "\x7f", "\x9b"])
def test_sanitize_replaces_each_control_char_with_one_placeholder(ctrl: str) -> None:
    out = live._sanitize(f"a{ctrl}b")
    assert out == "a?b"


def test_sanitize_output_is_always_printable() -> None:
    every_low_byte = "".join(chr(i) for i in range(256))
    assert live._sanitize(every_low_byte).isprintable()


def test_sanitize_empty() -> None:
    assert live._sanitize("") == ""


# -- _wrap -------------------------------------------------------------------


def test_wrap_empty_line_is_one_empty_row() -> None:
    # A blank line in the output must still take up a row on screen.
    assert live._wrap("", 10) == [""]


def test_wrap_short_line_is_unchanged() -> None:
    assert live._wrap("hello", 10) == ["hello"]


def test_wrap_exact_width_is_one_row() -> None:
    assert live._wrap("abcde", 5) == ["abcde"]


def test_wrap_cuts_mid_word_at_exact_width() -> None:
    assert live._wrap("abcdefghij", 4) == ["abcd", "efgh", "ij"]


def test_wrap_preserves_whitespace_and_indentation() -> None:
    line = "    if x:   y"
    rows = live._wrap(line, 5)
    assert rows == ["    i", "f x: ", "  y"]
    assert "".join(rows) == line


@pytest.mark.parametrize("width", [1, 2, 3, 7, 80])
def test_wrap_rows_fit_and_rejoin_to_the_original(width: int) -> None:
    line = "The quick brown fox\tjumps over the lazy dog, 0123456789."
    rows = live._wrap(line, width)
    assert all(len(r) <= width for r in rows)
    assert all(len(r) == width for r in rows[:-1])  # only the last row is short
    assert "".join(rows) == line


# -- _waiting_for ------------------------------------------------------------


def _task(
    task_id: str,
    status: str = "pending",
    depends_on: tuple[str, ...] = (),
    wave: int = 0,
) -> TaskStream:
    return TaskStream(
        task_id=task_id,
        agent="",
        model="",
        status=status,
        lines=[],
        pending="",
        updated_at=0.0,
        unseen=0,
        wave=wave,
        depends_on=depends_on,
    )


def test_waiting_for_lists_unfinished_dependencies_in_spec_order() -> None:
    task = _task("c", depends_on=("b", "a"))
    snaps = [_task("a", "running"), _task("b", "pending"), task]
    assert live._waiting_for(task, snaps) == ["b", "a"]


@pytest.mark.parametrize("done", ["ok", "failed", "skipped"])
def test_waiting_for_treats_every_terminal_status_as_finished(done: str) -> None:
    # A failed dependency is not something to keep waiting on: its dependents
    # are about to be skipped.
    task = _task("b", depends_on=("a",))
    assert live._waiting_for(task, [_task("a", done), task]) == []


def test_waiting_for_drops_only_the_finished_dependencies() -> None:
    task = _task("d", depends_on=("a", "b", "c"))
    snaps = [_task("a", "ok"), _task("b", "running"), _task("c", "failed"), task]
    assert live._waiting_for(task, snaps) == ["b"]


def test_waiting_for_counts_an_unknown_dependency_as_unfinished() -> None:
    task = _task("b", depends_on=("ghost",))
    assert live._waiting_for(task, [task]) == ["ghost"]


@pytest.mark.parametrize("status", ["running", "ok", "failed", "skipped"])
def test_waiting_for_is_empty_once_the_task_has_started(status: str) -> None:
    task = _task("b", status, depends_on=("a",))
    assert live._waiting_for(task, [_task("a", "pending"), task]) == []


def test_waiting_for_is_empty_without_dependencies() -> None:
    task = _task("a")
    assert live._waiting_for(task, [task]) == []


# -- _runs_with --------------------------------------------------------------
#
# `snaps` is in bar order (the order tabs are drawn), so every expected list
# below is written in the order its tasks appear in `snaps`. Ids are chosen so
# that bar order is never alphabetical order, so a sorted result would fail.


def test_runs_with_lists_both_wave_mates_in_bar_order() -> None:
    snaps = [_task("zeta", wave=1), _task("alpha", wave=1), _task("mid", wave=1)]
    assert live._runs_with(snaps[1], snaps) == ["zeta", "mid"]
    assert live._runs_with(snaps[0], snaps) == ["alpha", "mid"]
    assert live._runs_with(snaps[2], snaps) == ["zeta", "alpha"]


def test_runs_with_is_empty_for_a_task_alone_in_its_wave() -> None:
    snaps = [_task("a", wave=0), _task("solo", wave=1), _task("c", wave=2), _task("d", wave=2)]
    assert live._runs_with(snaps[1], snaps) == []


def test_runs_with_is_empty_when_the_task_is_the_only_one() -> None:
    task = _task("only", wave=0)
    assert live._runs_with(task, [task]) == []


@pytest.mark.parametrize("position", [0, 1, 2, 3])
def test_runs_with_never_lists_the_task_itself(position: int) -> None:
    # The off-by-one this guards against would put "runs with: <itself>" on
    # screen. Checked from every slot, including the first and last, where an
    # index-based slice is most likely to slip.
    snaps = [_task(t, wave=3) for t in ("d", "b", "a", "c")]
    current = snaps[position]
    result = live._runs_with(current, snaps)
    assert current.task_id not in result
    assert result == [s.task_id for s in snaps if s is not current]


def test_runs_with_excludes_other_waves_on_both_sides_of_the_bar() -> None:
    # Neighbouring tabs from waves 0 and 2 sit right against wave 1's edges.
    snaps = [
        _task("w0", wave=0),
        _task("first", wave=1),
        _task("second", wave=1),
        _task("w2", wave=2),
    ]
    assert live._runs_with(snaps[1], snaps) == ["second"]
    assert live._runs_with(snaps[2], snaps) == ["first"]


def test_runs_with_skips_an_adjacent_task_from_another_wave_when_waves_interleave() -> None:
    # A spec that interleaves waves puts a different-wave task between two
    # wave-mates. It is adjacent to both and must still be left out, while the
    # non-adjacent wave-mate is found.
    snaps = [_task("x", wave=0), _task("y", wave=1), _task("z", wave=0)]
    assert live._runs_with(snaps[0], snaps) == ["z"]
    assert live._runs_with(snaps[2], snaps) == ["x"]
    assert live._runs_with(snaps[1], snaps) == []


def test_runs_with_ignores_status_and_dependencies() -> None:
    # Wave membership alone decides it: a finished or pending wave-mate, or one
    # that depends on nothing in common, still ran in the same wave.
    snaps = [
        _task("a", "ok", wave=1),
        _task("b", "running", wave=1, depends_on=("root",)),
        _task("c", "failed", wave=1),
        _task("d", "skipped", wave=1),
    ]
    assert live._runs_with(snaps[1], snaps) == ["a", "c", "d"]


# -- _pert_lines -------------------------------------------------------------
#
# The chart is a pure function of the snapshots, so everything below reads the
# plain text it returns. Ids are whole words that never prefix one another or
# any word the chart itself prints ("wave", "terminal", "narrow", ...), so a
# whole-word search finds each node exactly and a truncated id is detectable.
# Bar order is deliberately not alphabetical within a wave.

# Wide enough that nothing should need to degrade.
WIDE = 200

_BOX_CORNERS_AND_VERTICALS = set("┐┘┌└┴┬├┤┼│╮╯╭╰╴╵╶╷")


def _two_waves() -> list[TaskStream]:
    """Fan-in: [[loader, tests], [verify]], the layout drawn in specs/pert-chart.yaml."""
    return [
        _task("loader", "running", wave=0),
        _task("tests", "ok", wave=0),
        _task("verify", "pending", depends_on=("loader", "tests"), wave=1),
    ]


def _three_waves() -> list[TaskStream]:
    """Fan-out and fan-in, plus one dependency (fetch -> report) that skips a column."""
    return [
        _task("fetch", "ok", wave=0),
        _task("parse", "failed", wave=0),
        _task("index", "running", wave=0),
        _task("merge", "skipped", depends_on=("fetch", "parse"), wave=1),
        _task("audit", "pending", depends_on=("index",), wave=1),
        _task("report", "pending", depends_on=("merge", "audit", "fetch"), wave=2),
    ]


Graph = Callable[[], list[TaskStream]]
GRAPHS = [pytest.param(_two_waves, id="two-wave"), pytest.param(_three_waves, id="three-wave")]


def _find(lines: list[str], word: str) -> list[tuple[int, int]]:
    """(row, column) of every whole-word occurrence of `word` in the chart."""
    pat = re.compile(rf"(?<![\w-]){re.escape(word)}(?![\w-])")
    return [(r, m.start()) for r, line in enumerate(lines) for m in pat.finditer(line)]


def _at(lines: list[str], word: str) -> tuple[int, int]:
    found = _find(lines, word)
    assert len(found) == 1, (word, found, lines)
    return found[0]


def _show(lines: list[str]) -> str:
    """Assertion message: the chart as it would appear on screen."""
    return "\n" + "\n".join(lines)


def _check_plain(lines: list[str], width: int, unicode: bool) -> None:
    """Every chart, at any width, is plain printable text that fits."""
    assert isinstance(lines, list)
    for line in lines:
        assert isinstance(line, str)
        assert len(line) <= width, (width, line)
        # Printable rules out ANSI escapes and colour codes: the shape has to be
        # carried by the characters themselves.
        assert line.isprintable(), repr(line)
        if not unicode:
            assert line.isascii(), [c for c in line if not c.isascii()]


@pytest.mark.parametrize("unicode", BOTH)
@pytest.mark.parametrize("graph", GRAPHS)
def test_pert_every_task_appears_exactly_once(graph: Graph, unicode: bool) -> None:
    snaps = graph()
    lines = live._pert_lines(snaps, snaps[0].task_id, WIDE, unicode)
    _check_plain(lines, WIDE, unicode)
    for s in snaps:
        assert len(_find(lines, s.task_id)) == 1, (s.task_id, _show(lines))


@pytest.mark.parametrize("unicode", BOTH)
@pytest.mark.parametrize("graph", GRAPHS)
def test_pert_same_wave_shares_a_column_and_later_waves_sit_to_the_right(
    graph: Graph, unicode: bool
) -> None:
    snaps = graph()
    # Repeated for every possible selection: a selection mark that pushed its
    # task's id over by a bracket would break the column it sits in.
    for selected in snaps:
        lines = live._pert_lines(snaps, selected.task_id, WIDE, unicode)
        column: dict[int, int] = {}
        for s in snaps:
            _, col = _at(lines, s.task_id)
            column.setdefault(s.wave, col)
            assert col == column[s.wave], (s.task_id, selected.task_id, _show(lines))
        waves = sorted(column)
        assert waves == list(range(max(s.wave for s in snaps) + 1))
        for earlier, later in zip(waves, waves[1:], strict=False):
            assert column[earlier] < column[later], _show(lines)


@pytest.mark.parametrize("unicode", BOTH)
def test_pert_stacks_a_wave_in_bar_order_with_bar_tab_numbers(unicode: bool) -> None:
    # Bar order interleaves the waves, so a wave's stacking order and its tab
    # numbers are not the same thing as its position in the column: `zulu` is
    # second in wave 1's column but is tab 3, the key that selects it.
    snaps = [
        _task("yankee", "running", wave=0),
        _task("solo", "pending", depends_on=("yankee",), wave=1),
        _task("zulu", "ok", wave=0),
        _task("xray", "pending", wave=0),
    ]
    lines = live._pert_lines(snaps, "solo", WIDE, unicode)
    rows = [_at(lines, t)[0] for t in ("yankee", "zulu", "xray")]
    assert rows == sorted(rows) and len(set(rows)) == 3, _show(lines)
    for number, s in enumerate(snaps, start=1):
        row, col = _at(lines, s.task_id)
        # The tab number sits just before the id ("3 zulu"), allowing for a
        # separator such as ":" or ".", and is not part of a longer number.
        before = lines[row][:col]
        assert re.search(rf"(?<!\d){number}\W{{0,3}}$", before), (s.task_id, _show(lines))


@pytest.mark.parametrize("graph", GRAPHS)
def test_pert_each_node_shows_its_status_glyph(graph: Graph) -> None:
    # Unicode only: its glyphs cannot be mistaken for connectors, whereas the
    # ASCII "-" and "." could come from the drawing around a node.
    snaps = graph()
    lines = live._pert_lines(snaps, snaps[0].task_id, WIDE, True)
    for s in snaps:
        row, col = _at(lines, s.task_id)
        after = lines[row][col + len(s.task_id) :]
        # Up to the next node on the same row, if any.
        next_cols = [c for o in snaps for r, c in _find(lines, o.task_id) if r == row and c > col]
        segment = after[: min(next_cols) - col - len(s.task_id)] if next_cols else after
        assert live._UNICODE_GLYPHS[s.status] in segment, (s.task_id, _show(lines))


@pytest.mark.parametrize("graph", GRAPHS)
def test_pert_ascii_variant_is_pure_ascii(graph: Graph) -> None:
    snaps = graph()
    for width in (WIDE, 80, 60, 40, 20, 10):
        for selected in snaps:
            lines = live._pert_lines(snaps, selected.task_id, width, False)
            text = "\n".join(lines)
            assert text.isascii(), (width, [c for c in text if not c.isascii()])


@pytest.mark.parametrize("unicode", BOTH)
@pytest.mark.parametrize("graph", GRAPHS)
def test_pert_selected_task_is_marked_in_plain_text(graph: Graph, unicode: bool) -> None:
    snaps = graph()
    charts = {s.task_id: live._pert_lines(snaps, s.task_id, WIDE, unicode) for s in snaps}
    for lines in charts.values():
        _check_plain(lines, WIDE, unicode)  # no reverse video or colour codes to lean on
    # Each selection reads differently...
    assert len({tuple(c) for c in charts.values()}) == len(snaps)
    # ...and the difference is on the selected task's own row: compared with a
    # chart where some task on another row is selected, that row has changed.
    for t in snaps:
        for u in snaps:
            row_t, _ = _at(charts[t.task_id], t.task_id)
            if u is t or _at(charts[t.task_id], u.task_id)[0] == row_t:
                continue
            assert _at(charts[u.task_id], t.task_id)[0] == row_t  # nodes do not move
            assert charts[t.task_id][row_t] != charts[u.task_id][row_t], (
                t.task_id,
                u.task_id,
                _show(charts[t.task_id]),
            )


@pytest.mark.parametrize("unicode", BOTH)
def test_pert_draws_connectors_for_fan_in(unicode: bool) -> None:
    """Positive control for the single-task test: the same check does find edges."""
    # No `ok` task: its ASCII glyph is "+", which would pass this on its own.
    snaps = [
        _task("loader", "running", wave=0),
        _task("tests", "pending", wave=0),
        _task("verify", "pending", depends_on=("loader", "tests"), wave=1),
    ]
    lines = live._pert_lines(snaps, "verify", WIDE, unicode)
    text = "\n".join(lines)
    if unicode:
        assert _BOX_CORNERS_AND_VERTICALS & set(text), _show(lines)
    else:
        assert "+" in text or "|" in text, _show(lines)


@pytest.mark.parametrize("status", ["pending", "running", "ok", "failed"])
@pytest.mark.parametrize("unicode", BOTH)
def test_pert_single_task_renders_without_connectors(unicode: bool, status: str) -> None:
    snaps = [_task("solo", status, wave=0)]
    lines = live._pert_lines(snaps, "solo", WIDE, unicode)
    _check_plain(lines, WIDE, unicode)
    row, col = _at(lines, "solo")
    assert re.search(r"(?<!\d)1\W{0,3}$", lines[row][:col]), _show(lines)
    glyph = (live._UNICODE_GLYPHS if unicode else live._ASCII_GLYPHS)[status]
    after = lines[row][col + len("solo") :]
    assert glyph in after, _show(lines)
    # The ASCII glyph for `ok` is "+", a connector character too, so the node's
    # own glyph is taken out before looking for edges.
    rest = [*lines[:row], after.replace(glyph, "", 1), *lines[row + 1 :]]
    text = "\n".join(rest)
    if unicode:
        assert not _BOX_CORNERS_AND_VERTICALS & set(text), _show(lines)
        # No edge leaving the node: a wave header's rule may use box drawing,
        # but the node's own row should not.
        assert not any("\u2500" <= c <= "\u257f" for c in lines[row]), _show(lines)
    else:
        assert "+" not in text and "|" not in text, _show(lines)
        assert "-" not in after.replace(glyph, "", 1), _show(lines)


@pytest.mark.parametrize("unicode", BOTH)
@pytest.mark.parametrize("graph", GRAPHS)
def test_pert_fits_width_and_degrades_honestly_at_every_width(graph: Graph, unicode: bool) -> None:
    """From 1 column up to roomy: each line fits, and the chart either still names
    every task exactly once or gives up with one honest line. What it may not do
    is silently lose a task or cut an id in half."""
    snaps = graph()
    ids = [s.task_id for s in snaps]
    for width in range(1, 121):
        lines = live._pert_lines(snaps, ids[-1], width, unicode)
        _check_plain(lines, width, unicode)
        counts = {i: len(_find(lines, i)) for i in ids}
        if all(n == 1 for n in counts.values()):
            # Readable, not garbled: no word is the stub of a truncated id.
            words = set(re.findall(r"\w+", "\n".join(lines)))
            stubs = {w for w in words for i in ids if len(w) >= 2 and w != i and i.startswith(w)}
            assert not stubs, (width, stubs, _show(lines))
            continue
        assert len(lines) == 1, (width, counts, _show(lines))
        if width >= 40:
            assert "narrow" in lines[0].lower(), (width, lines)


@pytest.mark.parametrize("unicode", BOTH)
def test_pert_full_chart_has_wave_headers_in_order(unicode: bool) -> None:
    lines = live._pert_lines(_three_waves(), "fetch", WIDE, unicode)
    headers = [_find(lines, f"wave {n}") for n in (1, 2, 3)]
    assert all(len(h) == 1 for h in headers), _show(lines)
    (r1, c1), (r2, c2), (r3, c3) = (h[0] for h in headers)
    assert r1 == r2 == r3 and c1 < c2 < c3, _show(lines)


def test_pert_needs_no_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pure layout: works with curses unimportable."""
    monkeypatch.setitem(sys.modules, "curses", None)
    snaps = _two_waves()
    first = live._pert_lines(snaps, "tests", 60, True)
    assert first == live._pert_lines(snaps, "tests", 60, True)
