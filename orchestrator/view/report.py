"""Run reporting: per-task outcome and what it cost."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..run import TaskResult
from ..spec import MODEL_BY_DIFFICULTY, Spec


def _row(cells: tuple[str, ...], widths: tuple[int, ...]) -> str:
    return "  ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True)).rstrip()


def _review_cell(r: TaskResult) -> str:
    """Verdict plus how many rounds it took, e.g. `approved(2)`."""
    if not r.reviews:
        return "-"
    short = {
        "approved": "approved",
        "changes_requested": "CHANGES",
        "unparsed": "unparsed",
        "error": "ERROR",
    }.get(r.review_verdict, r.review_verdict)
    return f"{short}({len(r.reviews)})"


def render(spec: Spec, results: dict[str, TaskResult]) -> str:
    order = [t.id for t in spec.tasks]
    rows = [
        (
            tid,
            results[tid].status if tid in results else "not-run",
            results[tid].difficulty if tid in results else "",
            results[tid].model if tid in results else "",
            _review_cell(results[tid]) if tid in results else "",
            f"{results[tid].duration_s:.1f}s" if tid in results else "",
            f"${results[tid].total_cost_usd:.4f}" if tid in results else "",
        )
        for tid in order
    ]
    header = ("TASK", "STATUS", "DIFF", "MODEL", "REVIEW", "TIME", "COST")
    widths = tuple(max(len(header[i]), max((len(r[i]) for r in rows), default=0)) for i in range(7))

    lines = ["", "=" * 72, "RUN SUMMARY", "=" * 72, _row(header, widths), "-" * 72]
    lines.extend(_row(r, widths) for r in rows)
    lines.append("-" * 72)

    total = sum(r.total_cost_usd for r in results.values())
    review_total = sum(r.review_cost_usd for r in results.values())
    by_model: dict[str, float] = {}
    for r in results.values():
        if r.model:
            by_model[r.model] = by_model.get(r.model, 0.0) + r.total_cost_usd

    counts: dict[str, int] = {}
    for r in results.values():
        counts[r.status] = counts.get(r.status, 0) + 1
    lines.append("  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    if by_model:
        lines.append("")
        lines.append("cost by model:")
        for model, cost in sorted(by_model.items(), key=lambda kv: -kv[1]):
            alias = next((a for a, m in MODEL_BY_DIFFICULTY.items() if m == model), "?")
            lines.append(f"  {model:<22} ({alias:<6}) ${cost:.4f}")
    lines.append("")
    if review_total:
        lines.append(
            f"  of which review: ${review_total:.4f} ({review_total / total * 100:.0f}% of spend)"
        )
    lines.append(f"TOTAL COST: ${total:.4f}")

    problems = [r for r in results.values() if not r.ok]
    if problems:
        lines.append("")
        lines.append("not ok:")
        for r in problems:
            lines.append(f"  {r.task_id}: {r.status} - {r.reason or '(no detail)'}")

    unresolved = [r for r in results.values() if r.review_verdict == "changes_requested"]
    if unresolved:
        lines.append("")
        lines.append("review rounds ran out; work accepted as-is with findings open:")
        for r in unresolved:
            lines.append(f"  {r.task_id} (after {len(r.reviews)} round(s))")

    questions = sum(len(r.questions) for r in results.values())
    denials = sum(len(r.denials) for r in results.values())
    if questions or denials:
        lines.append("")
        lines.append(f"clarifying questions answered: {questions}   tool calls denied: {denials}")
    lines.append("=" * 72)
    return "\n".join(lines)


def write_json(path: Path, spec: Spec, results: dict[str, TaskResult]) -> None:
    payload = {
        "total_cost_usd": round(sum(r.total_cost_usd for r in results.values()), 6),
        "review_cost_usd": round(sum(r.review_cost_usd for r in results.values()), 6),
        "tasks": {tid: asdict(r) for tid, r in results.items()},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
