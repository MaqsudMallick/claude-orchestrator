"""Review loop behaviour with the reviewer/implementer sessions stubbed out."""

import textwrap

import anyio
import pytest
from claude_agent_sdk import ResultMessage
from orchestrator import spec as spec_mod
from orchestrator.run import TaskResult
from orchestrator.run import review as rv
from pydantic import ValidationError

SPEC = """
defaults:
  review:
    enabled: true
    max_reviews: 3
agents:
  a: {system_prompt: x}
tasks:
  - {id: one, agent: a, prompt: build the thing}
"""


def load(tmp_path, body=SPEC):
    p = tmp_path / "spec.yaml"
    p.write_text(textwrap.dedent(body))
    return spec_mod.load(p)


class Ctx:
    def __init__(self):
        self.ask_lock = anyio.Lock()
        self.logs = []

    def log(self, msg):
        self.logs.append(msg)


def msg(text, *, is_error=False, cost=0.01, sid="s1"):
    return ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=is_error,
        num_turns=1,
        session_id=sid,
        total_cost_usd=cost,
        result=text,
    )


def ok_result():
    return TaskResult(task_id="one", status="ok", text="built it", session_id="s1")


# --- verdict parsing -------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("looks good\nVERDICT: APPROVED", rv.APPROVED),
        ("problems\nVERDICT: CHANGES_REQUESTED", rv.CHANGES_REQUESTED),
        ("verdict: approved", rv.APPROVED),
        ("no marker at all", rv.UNPARSED),
        # the final marker wins, so quoting the options above doesn't confuse it
        ("VERDICT: APPROVED ... actually\nVERDICT: CHANGES_REQUESTED", rv.CHANGES_REQUESTED),
    ],
)
def test_verdict_parsing(text, expected):
    assert rv._parse_verdict(text) == expected


# --- config merging --------------------------------------------------------


def test_review_is_off_by_default(tmp_path):
    s = load(
        tmp_path,
        """
    agents:
      a: {system_prompt: x}
    tasks:
      - {id: one, agent: a, prompt: p}
    """,
    )
    assert s.review_for(s.by_id["one"]).enabled is False


def test_task_override_of_one_field_keeps_the_others(tmp_path):
    s = load(
        tmp_path,
        """
    defaults:
      review: {enabled: true, max_reviews: 3}
    agents:
      a: {system_prompt: x}
    tasks:
      - {id: keep, agent: a, prompt: p}
      - {id: fewer, agent: a, prompt: p, review: {max_reviews: 1}}
      - {id: disabled, agent: a, prompt: p, review: {enabled: false}}
    """,
    )
    # regression: a partial override must not silently reset the other fields
    fewer = s.review_for(s.by_id["fewer"])
    assert (fewer.enabled, fewer.max_reviews) == (True, 1)
    off = s.review_for(s.by_id["disabled"])
    assert (off.enabled, off.max_reviews) == (False, 3)
    assert s.review_for(s.by_id["keep"]).max_reviews == 3


def test_max_reviews_is_bounded(tmp_path):
    for bad in (0, 99):
        with pytest.raises(ValidationError, match="max_reviews"):
            load(
                tmp_path,
                f"""
            defaults:
              review: {{enabled: true, max_reviews: {bad}}}
            agents:
              a: {{system_prompt: x}}
            tasks:
              - {{id: one, agent: a, prompt: p}}
            """,
            )


# --- the loop --------------------------------------------------------------


def run_loop(spec, result, monkeypatch, replies):
    """`replies` is consumed one per session (review, fix, review, ...)."""
    calls = []

    async def fake_session(options, prompt):
        calls.append(prompt)
        return replies.pop(0)

    monkeypatch.setattr(rv, "_run_session", fake_session)
    monkeypatch.setattr(rv, "_is_git_repo", lambda cwd: False)
    outcome = anyio.run(rv.run_reviews, spec.by_id["one"], spec, result, spec.base_dir, Ctx())
    return outcome, calls


def test_disabled_review_runs_nothing(tmp_path, monkeypatch):
    s = load(
        tmp_path,
        """
    defaults:
      review: {enabled: false}
    agents:
      a: {system_prompt: x}
    tasks:
      - {id: one, agent: a, prompt: p}
    """,
    )
    outcome, calls = run_loop(s, ok_result(), monkeypatch, [])
    assert outcome.rounds == [] and calls == []


def test_approved_first_time_costs_one_session(tmp_path, monkeypatch):
    s = load(tmp_path)
    outcome, calls = run_loop(s, ok_result(), monkeypatch, [msg("fine\nVERDICT: APPROVED")])
    assert outcome.verdict == rv.APPROVED
    assert len(outcome.rounds) == 1
    assert len(calls) == 1  # no fix session when nothing was requested


def test_changes_then_approved(tmp_path, monkeypatch):
    s = load(tmp_path)
    result = ok_result()
    outcome, calls = run_loop(
        s,
        result,
        monkeypatch,
        [
            msg("bug on line 3\nVERDICT: CHANGES_REQUESTED"),
            msg("fixed the off-by-one", cost=0.02),
            msg("all good now\nVERDICT: APPROVED"),
        ],
    )
    assert outcome.verdict == rv.APPROVED
    assert len(outcome.rounds) == 2
    assert outcome.rounds[0].fix_attempted and outcome.rounds[0].fix_ok
    # the fix summary replaces the original, so dependents inherit the fixed state
    assert result.text == "fixed the off-by-one"
    assert "bug on line 3" in calls[1]


def test_exhausting_max_reviews_reports_unresolved(tmp_path, monkeypatch):
    s = load(
        tmp_path,
        """
    defaults:
      review: {enabled: true, max_reviews: 2}
    agents:
      a: {system_prompt: x}
    tasks:
      - {id: one, agent: a, prompt: p}
    """,
    )
    outcome, _ = run_loop(
        s,
        ok_result(),
        monkeypatch,
        [
            msg("still broken\nVERDICT: CHANGES_REQUESTED"),
            msg("tried again"),
            msg("still broken\nVERDICT: CHANGES_REQUESTED"),
        ],
    )
    assert outcome.verdict == rv.CHANGES_REQUESTED
    assert len(outcome.rounds) == 2


def test_unparsed_verdict_does_not_burn_a_fix_round(tmp_path, monkeypatch):
    s = load(tmp_path)
    outcome, calls = run_loop(s, ok_result(), monkeypatch, [msg("I have opinions but no marker")])
    assert outcome.verdict == rv.UNPARSED
    assert len(calls) == 1


def test_reviewer_crash_is_not_fatal(tmp_path, monkeypatch):
    s = load(tmp_path)

    async def boom(options, prompt):
        raise RuntimeError("transport died")

    monkeypatch.setattr(rv, "_run_session", boom)
    monkeypatch.setattr(rv, "_is_git_repo", lambda cwd: False)
    outcome = anyio.run(rv.run_reviews, s.by_id["one"], s, ok_result(), tmp_path, Ctx())
    assert outcome.verdict == rv.ERROR
    assert "transport died" in outcome.rounds[0].error


def test_review_cost_is_summed_across_rounds(tmp_path, monkeypatch):
    s = load(tmp_path)
    outcome, _ = run_loop(
        s,
        ok_result(),
        monkeypatch,
        [
            msg("nope\nVERDICT: CHANGES_REQUESTED", cost=0.05),
            msg("fixed", cost=0.10),
            msg("ok\nVERDICT: APPROVED", cost=0.05),
        ],
    )
    assert outcome.cost_usd == pytest.approx(0.20)


# --- reporting -------------------------------------------------------------


def test_report_shows_review_column_and_folds_review_into_cost(tmp_path):
    """Regression: the review column was silently dropped once by a bad patch,
    leaving per-task cost excluding review while the total included it."""
    from orchestrator.view import report

    s = load(tmp_path)
    result = TaskResult(
        task_id="one",
        status="ok",
        model="claude-haiku-5-5",
        difficulty="easy",
        cost_usd=0.0045,
        review_cost_usd=0.0097,
        review_verdict=rv.APPROVED,
        reviews=[rv.ReviewRound(round=1, verdict=rv.APPROVED)],
    )
    out = report.render(s, {"one": result})

    assert "REVIEW" in out
    assert "approved(1)" in out
    # per-task cost must include review, or the table contradicts the total
    assert "$0.0142" in out
    assert "of which review" in out


def test_report_flags_unresolved_findings(tmp_path):
    from orchestrator.view import report

    s = load(tmp_path)
    result = TaskResult(
        task_id="one",
        status="ok",
        review_verdict=rv.CHANGES_REQUESTED,
        reviews=[
            rv.ReviewRound(round=1, verdict=rv.CHANGES_REQUESTED),
            rv.ReviewRound(round=2, verdict=rv.CHANGES_REQUESTED),
        ],
    )
    out = report.render(s, {"one": result})
    assert "CHANGES(2)" in out
    assert "accepted as-is" in out
    # running out of rounds must not fail the task - dependents still run
    assert result.status == "ok"


def test_fail_on_unresolved_is_no_longer_a_field(tmp_path):
    """Review never fails a task: running out of rounds accepts the work and
    moves on, so the old opt-in gate is gone rather than merely defaulted off."""
    with pytest.raises(ValidationError, match="fail_on_unresolved"):
        load(
            tmp_path,
            """
        defaults:
          review: {enabled: true, fail_on_unresolved: true}
        agents:
          a: {system_prompt: x}
        tasks:
          - {id: one, agent: a, prompt: p}
        """,
        )


def test_exhausting_rounds_leaves_verdict_visible_but_does_not_fail(tmp_path, monkeypatch):
    s = load(
        tmp_path,
        """
    defaults:
      review: {enabled: true, max_reviews: 1}
    agents:
      a: {system_prompt: x}
    tasks:
      - {id: one, agent: a, prompt: p}
    """,
    )
    result = ok_result()
    outcome, calls = run_loop(
        s,
        result,
        monkeypatch,
        [
            msg("not happy\nVERDICT: CHANGES_REQUESTED"),
        ],
    )
    assert outcome.verdict == rv.CHANGES_REQUESTED
    assert len(calls) == 1  # no fix round on the final round
    assert result.status == "ok"  # accepted, not failed
