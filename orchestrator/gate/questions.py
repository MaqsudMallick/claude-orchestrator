"""The human answer channel for AskUserQuestion.

Parallel tasks share one terminal, so every prompt is serialized behind a lock
and labelled with the task that asked. Reading stdin happens on a worker thread
so it never blocks the event loop running the other tasks.
"""

from __future__ import annotations

import sys
from typing import Any

import anyio

SEP = "=" * 72


def _render(task_id: str, agent: str, questions: list[dict[str, Any]]) -> dict[str, str]:
    """Blocking terminal prompt. Runs on a worker thread."""
    answers: dict[str, str] = {}
    print(f"\n{SEP}\n  QUESTION from task '{task_id}' (agent: {agent})\n{SEP}", flush=True)

    for qi, q in enumerate(questions, 1):
        text = q.get("question", "(no question text)")
        header = q.get("header") or ""
        options = q.get("options") or []
        multi = bool(q.get("multiSelect"))

        label = f"[{header}] " if header else ""
        print(f"\n({qi}/{len(questions)}) {label}{text}", flush=True)
        for oi, opt in enumerate(options, 1):
            desc = opt.get("description") or ""
            print(f"   {oi}. {opt.get('label', '')}", flush=True)
            if desc:
                print(f"      {desc}", flush=True)
        hint = "numbers separated by commas" if multi else "a number"
        print(f"   (enter {hint}, or type free text)", flush=True)

        answers[text] = _read_one(options, multi)

    print(f"\n{SEP}\n", flush=True)
    return answers


def _read_one(options: list[dict[str, Any]], multi: bool) -> str:
    while True:
        try:
            raw = input("   > ").strip()
        except EOFError:
            # No human available (piped stdin, CI). Fall back to the first
            # option rather than hanging the whole run forever.
            fallback = options[0].get("label", "") if options else ""
            print(f"   [stdin closed; defaulting to {fallback!r}]", file=sys.stderr, flush=True)
            return fallback

        if not raw:
            print("   (an answer is required)", flush=True)
            continue

        picks = [p.strip() for p in raw.split(",")] if multi else [raw]
        if all(p.isdigit() for p in picks) and options:
            idxs = [int(p) for p in picks]
            if all(1 <= i <= len(options) for i in idxs):
                chosen = [options[i - 1].get("label", "") for i in idxs]
                # Multi-select is returned as one comma-joined string: the
                # free-text path accepts any string, so this is always valid.
                return ", ".join(chosen)
            print(f"   (pick between 1 and {len(options)})", flush=True)
            continue

        # Anything non-numeric is treated as the "Other" free-text answer.
        return raw


async def ask_human(
    lock: anyio.Lock, task_id: str, agent: str, questions: list[dict[str, Any]]
) -> dict[str, str]:
    async with lock:
        return await anyio.to_thread.run_sync(_render, task_id, agent, questions)
