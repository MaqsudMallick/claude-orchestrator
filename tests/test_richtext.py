"""Markdown -> styled, wrapped lines for the curses body (`orchestrator.richtext`).

These pin the contract the screen layout relies on, not rich's taste:

- every line fits in `width` terminal cells, so nothing spills into the next pane;
- the spans of a line join up to that line's visible text, with no character
  (or inter-word space) lost at span boundaries;
- markdown markers are interpreted, not printed, except inside code blocks;
- half-typed markdown (the normal state while text streams in) never raises and
  never swallows what follows;
- `colour` is only ever a basic curses colour name.

Deliberately not asserted: which exact attributes rich gives a heading, code
block or quote. Those are rich's choices and may change between versions; only
"styled somehow" is checked where styling matters.
"""

import unicodedata
from dataclasses import fields

import pytest
from orchestrator.richtext import Span, render
from rich.cells import cell_len

BASIC_COLOURS = {"black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"}

WIDTHS = [5, 8, 12, 20, 40, 80]

PROSE = (
    "The orchestrator runs each task in its own worktree and streams the agent's "
    "output into this pane while the run is still going, so the text keeps growing."
)

KITCHEN_SINK = """\
# Release notes

Some **bold** words, some *italic* ones, `inline code` and a [link](https://example.com/a/b).

## Changes

- first item that is long enough to need wrapping on a narrow screen
- second item
  1. nested numbered
  2. another one

> A quoted paragraph that also runs long enough to wrap around.

```python
def f(x):
    return x ** 2  # **not bold**
```

| col a | col b |
|-------|-------|
| 1     | two   |

---

Final paragraph with a verylongunbrokenidentifierthatcannotbewrappedatspaces in it.
"""

DOCS = {
    "prose": PROSE,
    "kitchen-sink": KITCHEN_SINK,
    "long-url": "see https://example.com/" + "segment/" * 12 + "end for details",
    "wide-chars": "日本語のテキストは各文字が二つのセルを使います。 emoji 🎉🎉🎉 too",
    "code-only": "```\n" + "x = 1  # a long comment line that goes on and on\n" * 3 + "```",
    "unterminated": "ok\n\ntail **open and `also open and [a link",
}


def line_text(line: list[Span]) -> str:
    return "".join(span.text for span in line)


def all_text(lines: list[list[Span]]) -> str:
    return "\n".join(line_text(line) for line in lines)


def non_ws(s: str) -> str:
    return "".join(s.split())


def spans_containing(lines: list[list[Span]], needle: str) -> list[Span]:
    return [span for line in lines for span in line if needle in span.text]


def assert_well_formed(lines) -> None:
    assert isinstance(lines, list)
    for line in lines:
        assert isinstance(line, list)
        for span in line:
            assert isinstance(span, Span)
            assert isinstance(span.text, str)


# --- width -----------------------------------------------------------------


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("doc", DOCS.values(), ids=DOCS.keys())
def test_no_line_exceeds_width(doc, width):
    lines = render(doc, width)
    assert_well_formed(lines)
    for i, line in enumerate(lines):
        # Terminal cells, not code points: a CJK character or emoji takes two.
        assert cell_len(line_text(line)) <= width, f"line {i} too wide: {line_text(line)!r}"


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("doc", DOCS.values(), ids=DOCS.keys())
def test_spans_carry_no_control_characters(doc, width):
    # A newline or tab inside a span would move the curses cursor behind the
    # layout's back, making the width check above meaningless.
    for line in render(doc, width):
        for span in line:
            bad = [c for c in span.text if unicodedata.category(c) == "Cc"]
            assert not bad, f"control chars {bad!r} in {span.text!r}"


def test_tabs_do_not_break_width():
    doc = "a\tb\tc\n\n```\n\tindented\twith\ttabs\n```"
    for width in (8, 20):
        for line in render(doc, width):
            assert "\t" not in line_text(line)
            assert cell_len(line_text(line)) <= width


def test_narrower_width_never_gives_fewer_lines():
    counts = [len(render(PROSE, w)) for w in (80, 40, 20, 12)]
    assert counts == sorted(counts)
    assert counts[-1] > counts[0], "prose at width 12 should wrap more than at 80"


# --- round trip ------------------------------------------------------------


@pytest.mark.parametrize("width", WIDTHS)
def test_round_trip_plain_prose_keeps_every_character(width):
    # Wrapping may only add or remove whitespace; every visible character of
    # the source must come out, once, in order.
    assert non_ws(all_text(render(PROSE, width))) == non_ws(PROSE)


@pytest.mark.parametrize("width", [12, 20, 40, 80])
def test_round_trip_prose_lines_rejoin_to_source_words(width):
    # Every word here is shorter than the width, so lines break only at
    # spaces: stripping each line and joining with single spaces must give back
    # the source exactly. Catches a space lost or doubled between spans.
    lines = [line_text(line).strip() for line in render(PROSE, width)]
    assert " ".join(t for t in lines if t) == " ".join(PROSE.split())


@pytest.mark.parametrize("width", WIDTHS)
def test_round_trip_inline_markup_loses_only_the_markers(width):
    src = "alpha **bravo charlie** delta *echo* foxtrot `golf hotel` india"
    expected = non_ws(src.replace("**", "").replace("*", "").replace("`", ""))
    assert non_ws(all_text(render(src, width))) == expected


def test_round_trip_keeps_spaces_at_span_boundaries():
    lines = render("say **hello** to `code` now", 80)
    texts = [line_text(line).strip() for line in lines if line_text(line).strip()]
    assert texts == ["say hello to code now"]


def test_line_text_is_the_concatenation_of_its_spans():
    # Spans of one line must not overlap or repeat: total span length equals
    # the visible length of the line.
    for line in render(KITCHEN_SINK, 40):
        assert sum(len(s.text) for s in line) == len(line_text(line))
        assert "\n" not in line_text(line)


# --- inline styling ----------------------------------------------------------


def test_bold_is_flagged_and_markers_are_dropped():
    lines = render("say **hello** now", 40)
    assert "**" not in all_text(lines)
    hello = spans_containing(lines, "hello")
    assert hello and all(s.bold for s in hello)
    say = spans_containing(lines, "say")
    assert say and not any(s.bold for s in say), "plain text must not be bold"


def test_bold_survives_wrapping():
    lines = render("**one two three four five six**", 8)
    assert "**" not in all_text(lines)
    worded = [s for line in lines for s in line if s.text.strip()]
    assert worded and all(s.bold for s in worded)
    assert non_ws(all_text(lines)) == "onetwothreefourfivesix"


def test_heading_is_styled_not_hashed():
    lines = render("# Title here\n\nbody text", 40)
    text = all_text(lines)
    assert "#" not in text
    assert "Title here" in text
    title = spans_containing(lines, "Title")
    assert title
    assert any(s.bold or s.underline or s.reverse or s.dim or s.colour for s in title)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_heading_levels_drop_hashes(level):
    lines = render("#" * level + " Section name", 40)
    assert "#" not in all_text(lines)
    assert "Section name" in all_text(lines)


# --- code blocks -------------------------------------------------------------


CODE = """\
before

```
x = **not bold**
# not a heading
- not a list
```

after
"""


def test_fenced_code_block_survives_literally():
    lines = render(CODE, 40)
    texts = [line_text(line).strip() for line in lines]
    for src in ("x = **not bold**", "# not a heading", "- not a list"):
        assert src in texts, f"code line {src!r} was altered: {texts!r}"
    assert "```" not in all_text(lines)
    assert "before" in all_text(lines) and "after" in all_text(lines)


@pytest.mark.parametrize("width", [5, 12])
def test_code_block_keeps_its_characters_when_narrow(width):
    lines = render(CODE, width)
    out = non_ws(all_text(lines))
    for src in ("x = **not bold**", "# not a heading", "- not a list"):
        assert non_ws(src) in out


# --- partial / degenerate input ---------------------------------------------


def test_unterminated_bold_keeps_the_tail():
    lines = render("ok\n\ntail **open", 40)
    assert_well_formed(lines)
    text = all_text(lines)
    assert "ok" in text
    assert "tail" in text
    assert "open" in text


@pytest.mark.parametrize(
    "partial",
    [
        "tail **open",
        "tail *open",
        "tail `open",
        "tail [open",
        "tail [open](http://exa",
        "tail ~~open",
        "tail\n\n```python\nopen code",
        "tail\n\n# ",
        "tail\n\n- ",
        "tail\n\n> ",
        "tail\n\n| a | b",
        "tail **",
    ],
)
@pytest.mark.parametrize("width", [8, 40])
def test_partial_markdown_never_raises_or_eats_text(partial, width):
    lines = render(partial, width)
    assert_well_formed(lines)
    out = non_ws(all_text(lines))
    assert "tail" in out
    if "open" in partial:
        assert "open" in out


def test_every_prefix_of_a_document_renders():
    # Simulates the body being redrawn as each character streams in.
    for i in range(len(KITCHEN_SINK) + 1):
        lines = render(KITCHEN_SINK[:i], 30)
        assert_well_formed(lines)
        for line in lines:
            assert cell_len(line_text(line)) <= 30


@pytest.mark.parametrize("blank", ["", " ", "\n", "\n\n\n", "   \n\t\n  "])
def test_empty_and_whitespace_input_is_sane(blank):
    lines = render(blank, 40)
    assert_well_formed(lines)
    assert non_ws(all_text(lines)) == ""
    for line in lines:
        assert cell_len(line_text(line)) <= 40


@pytest.mark.parametrize("width", [5, 40])
def test_single_word(width):
    lines = render("hello", width)
    assert_well_formed(lines)
    assert non_ws(all_text(lines)) == "hello"
    assert all(cell_len(line_text(line)) <= width for line in lines)


def test_single_word_fits_on_one_line():
    lines = [line_text(line).strip() for line in render("hello", 40)]
    assert [t for t in lines if t] == ["hello"]


# --- the Span type -------------------------------------------------------------


@pytest.mark.parametrize("width", [8, 40, 80])
def test_colour_is_none_or_a_basic_curses_name(width):
    # Code blocks go through rich's syntax highlighter, which produces
    # truecolour styles: they must be mapped down, not passed through.
    for line in render(KITCHEN_SINK, width):
        for span in line:
            assert span.colour is None or (
                type(span.colour) is str and span.colour in BASIC_COLOURS
            ), f"bad colour {span.colour!r} on {span.text!r}"


def test_style_flags_are_plain_bools():
    for line in render(KITCHEN_SINK, 40):
        for span in line:
            for f in fields(Span):
                if f.name in ("text", "colour"):
                    continue
                assert type(getattr(span, f.name)) is bool, (f.name, span)


def test_span_defaults_and_immutability():
    s = Span("x")
    assert (s.bold, s.dim, s.underline, s.reverse, s.colour) == (False, False, False, False, None)
    with pytest.raises(AttributeError):
        s.text = "y"  # type: ignore[misc]
    assert Span("x", bold=True) == Span("x", bold=True)
    assert Span("x", bold=True) != Span("x")
