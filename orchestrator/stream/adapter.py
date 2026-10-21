"""Translating a task's session output into `StreamHub` updates.

The hub speaks in `StreamUpdate`s; the Claude SDK speaks in raw API stream
events whose shape (`content_block_delta` wrapping a `text_delta`, and so on)
is an SDK detail. Keeping that unwrapping here means the runner sequences a
session without knowing what a stream event looks like, and a change in the
SDK's event shapes is fixed in one place.
"""

from __future__ import annotations

import time
from typing import Any

from .hub import EventKind, StreamHub, StreamUpdate


def _publish(hub: StreamHub, task_id: str, kind: EventKind, text: str) -> None:
    hub.publish(StreamUpdate(task_id=task_id, kind=kind, text=text, at=time.monotonic()))


def note(hub: StreamHub, task_id: str, text: str) -> None:
    """Publish a note: a review round, a failure reason, anything not from the model."""
    _publish(hub, task_id, "note", text)


def tool(hub: StreamHub, task_id: str, tool_name: str) -> None:
    """Publish that the task called `tool_name`."""
    _publish(hub, task_id, "tool", tool_name)


def stream_event(hub: StreamHub, task_id: str, event: dict[str, Any], line_open: bool) -> bool:
    """Publish one raw API stream event's text, returning the new `line_open`.

    Only text deltas carry output worth showing; tool input deltas are partial
    JSON and the tool itself is published from the finished `ToolUseBlock`.

    The hub only starts a new line on a newline or a non-text event, so two text
    blocks in a row with no tool call between them would run together on one
    line. Ending an open line when a block stops keeps them apart without
    adding a blank line after text that already ended in a newline.
    """
    kind = event.get("type")
    if kind == "content_block_delta":
        delta = event.get("delta") or {}
        if delta.get("type") == "text_delta":
            text = str(delta.get("text", ""))
            if text:
                _publish(hub, task_id, "text", text)
                line_open = not text.endswith(("\n", "\r"))
    elif kind == "content_block_stop" and line_open:
        _publish(hub, task_id, "text", "\n")
        line_open = False
    return line_open
