"""The live viewer's pure helpers: the key legend, line sanitising and wrapping,
and the "waiting for" list on the info row.

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


def _task(task_id: str, status: str = "pending", depends_on: tuple[str, ...] = ()) -> TaskStream:
    return TaskStream(
        task_id=task_id,
        agent="",
        model="",
        status=status,
        lines=[],
        pending="",
        updated_at=0.0,
        unseen=0,
        wave=0,
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
