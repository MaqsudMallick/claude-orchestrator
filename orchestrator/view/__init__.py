"""What the user sees: the live viewer, markdown rendering, the run report, session browsing.

`live`, `richtext`, `report` and `browse` are part of the surface as modules, as they
were before the split. Their public names are re-exported too, except `render`:
`report.render` and `richtext.render` share the name, so each is reached through
its module.
"""

from . import browse, live, report, richtext
from .browse import render_error, render_session, render_session_list, render_spec, render_task
from .live import POLL_MS, run_viewer, supported
from .report import write_json
from .richtext import Span

__all__ = [
    "browse",
    "live",
    "report",
    "richtext",
    "render_error",
    "render_session",
    "render_session_list",
    "render_spec",
    "render_task",
    "POLL_MS",
    "run_viewer",
    "supported",
    "write_json",
    "Span",
]
