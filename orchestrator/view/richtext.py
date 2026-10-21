"""Markdown to styled spans, for the live body pane.

This module decides *what* is styled; ``live.py`` decides how a style looks on
screen. Nothing here touches curses, so the whole pipeline can be exercised
offline: text in, a list of visual lines of :class:`Span` out.

rich does the parsing, layout and wrapping. What it emits is a stream of
``Segment`` objects carrying full ``Style`` objects; we flatten that into the
handful of attributes a curses colour-pair table can honour.

The body is rendered while the model is still typing, so the input is usually
a *partial* document. rich, via markdown-it, already treats unterminated
emphasis (``a **bold``) and unclosed code spans as literal text, and an open
code fence simply runs to the end of the buffer, which is what the finished
document will look like anyway. The one case it gets wrong is raw HTML: rich
renders nothing for HTML blocks, and markdown-it extends an HTML block to the
next blank line, so a half-typed ``<div`` would hide everything after it. We
therefore reparse with HTML disabled so tags show up as the literal text the
model wrote.
"""

from __future__ import annotations

import io
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

from markdown_it import MarkdownIt
from rich.cells import cell_len
from rich.color import Color, ColorType
from rich.console import Console
from rich.markdown import Markdown
from rich.segment import Segment
from rich.style import Style

_BASIC_COLOURS: tuple[str, ...] = (
    "black",
    "red",
    "green",
    "yellow",
    "blue",
    "magenta",
    "cyan",
    "white",
)

# Same parser rich builds internally, minus raw HTML (see the module docstring).
_PARSER = MarkdownIt("commonmark", {"html": False}).enable("strikethrough").enable("table")


@dataclass(frozen=True)
class Span:
    """A run of text sharing one style, in terms curses can draw."""

    text: str
    bold: bool = False
    dim: bool = False
    underline: bool = False
    reverse: bool = False
    colour: str | None = None  # one of _BASIC_COLOURS, else None


def render(text: str, width: int) -> list[list[Span]]:
    """Render ``text`` as markdown, wrapped to ``width`` cells.

    Returns one list of spans per visual line. Never raises: if rich fails on
    some input, the display drops to plain text, one clipped span per input
    line, rather than taking the viewer down with it.
    """
    if width < 1:
        return []
    text = _sanitise(text)
    try:
        return _render_rich(text, width)
    except Exception:
        return _render_plain(text, width)


def _render_rich(text: str, width: int) -> list[list[Span]]:
    # A private file keeps rich from probing or writing to the real terminal,
    # which curses owns. The colour system is fixed so the result does not
    # depend on $TERM or NO_COLOR in whatever environment runs this.
    console = Console(
        width=width,
        file=io.StringIO(),
        force_terminal=True,
        color_system="truecolor",
        no_color=False,
        highlight=False,
        legacy_windows=False,
    )
    # hyperlinks=False makes rich print the URL after the link text; curses
    # cannot draw OSC 8 links, so otherwise the URL would be lost.
    markdown = Markdown(text, hyperlinks=False)
    markdown.parsed = _PARSER.parse(text)
    return [_finish_line(line, width) for line in _split_lines(console.render(markdown))]


def _split_lines(segments: Iterable[Segment]) -> list[list[Span]]:
    """Cut the segment stream into visual lines.

    rich usually emits a bare ``"\\n"`` segment between lines, but some
    renderables (code blocks, notably) fold the newline into a padding
    segment, so newlines are split out of every segment's text.
    """
    lines: list[list[Span]] = []
    current: list[Span] = []
    for segment in segments:
        if segment.control:
            continue
        parts = segment.text.split("\n")
        for index, part in enumerate(parts):
            if index:
                lines.append(current)
                current = []
            if part:
                current.append(_to_span(part, segment.style))
    if current:
        lines.append(current)
    return lines


def _to_span(text: str, style: Style | None) -> Span:
    if style is None:
        return Span(text)
    return Span(
        text,
        bold=bool(style.bold),
        dim=bool(style.dim),
        # Terminals render italic unreliably and curses has no portable
        # attribute for it, so it is shown as underline instead.
        underline=bool(style.underline or style.italic),
        reverse=bool(style.reverse),
        colour=_basic_colour(style.color),
    )


def _basic_colour(color: Color | None) -> str | None:
    """Reduce a rich colour to one of the eight ANSI names, or None.

    curses allocates colour pairs from a small fixed table, and syntax
    themes emit dozens of truecolour values. Approximating those to the
    nearest basic colour would recolour code arbitrarily, so anything that
    is not already one of the eight is left at the terminal default.
    """
    if color is None:
        return None
    if color.type in (ColorType.STANDARD, ColorType.EIGHT_BIT):
        number = color.number
        if number is not None and 0 <= number < len(_BASIC_COLOURS):
            return _BASIC_COLOURS[number]
    return None


def _finish_line(spans: list[Span], width: int) -> list[Span]:
    return _clip(_strip_padding(spans), width)


def _strip_padding(spans: list[Span]) -> list[Span]:
    """Drop the trailing spaces rich pads every line out to ``width`` with.

    Only padding that would be invisible is removed: spaces under underline
    or reverse are drawn, so those spans are kept as they are.
    """
    out = list(spans)
    while out:
        last = out[-1]
        if last.underline or last.reverse:
            break
        stripped = last.text.rstrip(" ")
        if stripped:
            out[-1] = _with_text(last, stripped)
            break
        out.pop()
    return out


def _clip(spans: list[Span], width: int) -> list[Span]:
    """Cut a line to at most ``width`` cells.

    rich is given the width and should honour it, but a line that overruns
    would wrap or raise inside curses, so it is checked rather than trusted.
    Measuring cells rather than characters also keeps ``len()`` within bounds,
    since no character is narrower than one cell.
    """
    out: list[Span] = []
    used = 0
    for span in spans:
        size = cell_len(span.text)
        if used + size <= width:
            out.append(span)
            used += size
            continue
        kept: list[str] = []
        for char in span.text:
            char_size = cell_len(char)
            if used + char_size > width:
                break
            kept.append(char)
            used += char_size
        if kept:
            out.append(_with_text(span, "".join(kept)))
        break
    return out


def _with_text(span: Span, text: str) -> Span:
    return Span(text, span.bold, span.dim, span.underline, span.reverse, span.colour)


def _sanitise(text: str) -> str:
    """Remove characters that would corrupt a curses screen.

    Escape sequences and carriage returns in model output would move the
    cursor or recolour the terminal behind curses' back. Tabs become spaces
    so cell widths are knowable; newlines are structure and are kept.
    """
    text = text.replace("\r\n", "\n").replace("\t", "    ")
    return "".join(char for char in text if char == "\n" or unicodedata.category(char) != "Cc")


def _render_plain(text: str, width: int) -> list[list[Span]]:
    return [_clip([Span(line)], width) if line else [] for line in text.split("\n")]
