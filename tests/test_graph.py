"""DAG behaviour with the Claude session stubbed out: ordering, parallelism,
and the skip-dependents failure policy."""

import textwrap

import anyio
import pytest
from orchestrator import spec as spec_mod
from orchestrator.run import TaskResult, graph

SPEC = """
agents:
  a: {system_prompt: x}
tasks:
  - {id: schema, agent: a, prompt: p}
  - {id: api,    agent: a, prompt: p, depends_on: [schema]}
  - {id: docs,   agent: a, prompt: p, depends_on: [schema]}
  - {id: tests,  agent: a, prompt: p, depends_on: [api]}
"""


@pytest.fixture
def spec(tmp_path):
    p = tmp_path / "spec.yaml"
    p.write_text(textwrap.dedent(SPEC))
    return spec_mod.load(p)


def fake_runner(fail: set[str], log: list, concurrency: dict):
    async def run_task(task, spec, results, ctx):
        log.append(("start", task.id))
        concurrency["now"] = concurrency.get("now", 0) + 1
        concurrency["peak"] = max(concurrency.get("peak", 0), concurrency["now"])
        await anyio.sleep(0.05)
        concurrency["now"] -= 1
        log.append(("end", task.id))
        status = "failed" if task.id in fail else "ok"
        return TaskResult(
            task_id=task.id,
            status=status,
            text=f"{task.id} output",
            cost_usd=0.01,
            reason="boom" if status == "failed" else "",
        )

    return run_task


def test_runs_in_dependency_order_and_parallelises_siblings(spec, monkeypatch):
    log, conc = [], {}
    monkeypatch.setattr(graph, "run_task", fake_runner(set(), log, conc))

    results = anyio.run(graph.run, spec, 4, False, True)

    assert {k: v.status for k, v in results.items()} == {
        "schema": "ok",
        "api": "ok",
        "docs": "ok",
        "tests": "ok",
    }
    order = [tid for kind, tid in log if kind == "start"]
    assert order[0] == "schema"
    assert order.index("tests") > order.index("api")
    # api and docs are independent, so they overlap
    assert conc["peak"] >= 2


def test_failure_skips_dependents_but_lets_siblings_finish(spec, monkeypatch):
    log, conc = [], {}
    monkeypatch.setattr(graph, "run_task", fake_runner({"api"}, log, conc))

    results = anyio.run(graph.run, spec, 4, False, True)

    assert results["schema"].status == "ok"
    assert results["api"].status == "failed"
    # tests depended on the failed task...
    assert results["tests"].status == "skipped"
    assert "api" in results["tests"].reason
    # ...while the unrelated branch still completed
    assert results["docs"].status == "ok"
    assert ("start", "tests") not in log


def test_root_failure_skips_everything_downstream(spec, monkeypatch):
    log, conc = [], {}
    monkeypatch.setattr(graph, "run_task", fake_runner({"schema"}, log, conc))

    results = anyio.run(graph.run, spec, 4, False, True)

    assert results["schema"].status == "failed"
    for tid in ("api", "docs", "tests"):
        assert results[tid].status == "skipped", tid
    assert [tid for kind, tid in log if kind == "start"] == ["schema"]


def test_max_parallel_is_respected(spec, monkeypatch):
    log, conc = [], {}
    monkeypatch.setattr(graph, "run_task", fake_runner(set(), log, conc))

    anyio.run(graph.run, spec, 1, False, True)
    assert conc["peak"] == 1


def test_upstream_text_is_injected_into_dependent_prompt(spec):
    from orchestrator.run.runner import build_prompt

    results = {"schema": TaskResult(task_id="schema", status="ok", text="TABLE links(...)")}
    api = spec.by_id["api"]
    prompt = build_prompt(api, spec, results)
    assert "TABLE links(...)" in prompt
    assert 'task="schema"' in prompt
    assert api.prompt.strip() in prompt


def test_upstream_text_is_truncated(spec):
    from orchestrator.run.runner import build_prompt

    spec.defaults.max_context_chars = 20
    results = {"schema": TaskResult(task_id="schema", status="ok", text="x" * 500)}
    prompt = build_prompt(spec.by_id["api"], spec, results)
    assert "truncated" in prompt
    assert len(prompt) < 400
