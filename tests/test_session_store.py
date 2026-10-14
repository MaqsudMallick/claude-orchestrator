"""Session tracking: what a run leaves on disk, and reading it back.

Offline like the rest of the suite. The round-trip test stubs the Claude
session one level lower than test_graph.py does, at `runner.ClaudeSDKClient`,
so the real runner, permission callback and question prompt all run and the
recorded questions are the ones the stub actually asked through them. Every
test stores sessions under tmp_path, and a guard fixture fails any test in
this file that writes to the default session directory instead.
"""

import atexit
import builtins
import os
import shutil
import tempfile
import textwrap
from datetime import UTC, datetime

import anyio
import pytest
from claude_agent_sdk import ResultMessage, ToolPermissionContext
from orchestrator import browse, graph, runner
from orchestrator import review as rv
from orchestrator import session_store as ss
from orchestrator import spec as spec_mod
from orchestrator.runner import TaskResult

# STOPGAP, belongs in tests/conftest.py. graph.run records a session by
# default, and other test modules (test_graph.py) call it without a storage
# path, so a plain `pytest` run would write into the user's real session
# directory. pytest imports every test module before running any test, so
# setting this at import covers the whole collected run. It does NOT cover a
# run that skips this module, e.g. `pytest tests/test_graph.py`. Overridden
# unconditionally: a value already set may point at the user's real store.
_SUITE_SESSION_DIR = tempfile.mkdtemp(prefix="orchestrator-test-sessions-")
os.environ[ss.ENV_VAR] = _SUITE_SESSION_DIR
atexit.register(shutil.rmtree, _SUITE_SESSION_DIR, ignore_errors=True)


@pytest.fixture(autouse=True)
def default_root_stays_empty(tmp_path, monkeypatch):
    """Point the default session root at a sandbox and require it to stay empty:
    every test passes an explicit root, so anything here means a code path
    ignored it and would have written into the user's real directory."""
    default = tmp_path / "default-session-root"
    monkeypatch.setenv(ss.ENV_VAR, str(default))
    yield
    assert not default.exists() or not any(default.iterdir())


@pytest.fixture
def root(tmp_path):
    return tmp_path / "sessions"


def write_spec(tmp_path, body, name="spec.yaml"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return p


TWO_TASKS = """
agents:
  a: {system_prompt: x}
tasks:
  - {id: schema, agent: a, prompt: do schema}
  - {id: api,    agent: a, prompt: do api, depends_on: [schema]}
"""


@pytest.fixture
def spec(tmp_path):
    return spec_mod.load(write_spec(tmp_path, TWO_TASKS))


def stub_tasks(monkeypatch, behaviour):
    """Replace run_task in the graph, as test_graph.py does. `behaviour` maps
    task id -> a TaskResult, or an exception to raise."""

    async def run_task(task, spec, results, ctx):
        outcome = behaviour.get(task.id) or TaskResult(task_id=task.id, status="ok", text="done")
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(graph, "run_task", run_task)


# --- a finished run round-trips --------------------------------------------

# Output text with what is awkward for a line-oriented JSON store: newlines,
# quotes, a backslash, a tab, a NUL, U+2028 (which str.splitlines() splits on),
# a lone surrogate (undecodable tool output; UTF-8 cannot encode it), non-ASCII
# and a line that is itself valid JSON.
SCHEMA_OUTPUT = (
    'CREATE TABLE "links" (\n\tid int\n);\r\n'
    "path C:\\new\\table\x00\u2028next para\u2029 caf\u00e9 \U0001f600 \udcff\n"
    '{"kind": "end", "finished_at": "2020-01-01T00:00:00Z"}\n'
)
API_OUTPUT = "api done"


class FakeClient:
    """Stands in for ClaudeSDKClient. Each task's script is a list of tool
    calls sent through the real can_use_tool callback, then the final result."""

    scripts: dict = {}

    def __init__(self, options):
        self.options = options

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def query(self, prompt):
        (self.script,) = [s for marker, s in self.scripts.items() if prompt.endswith(marker)]

    async def receive_response(self):
        for tool, tool_input in self.script["tools"]:
            await self.options.can_use_tool(tool, tool_input, ToolPermissionContext())
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=3,
            session_id="sess-" + self.script["text"][:3],
            total_cost_usd=self.script["cost"],
            result=self.script["text"],
        )


def test_finished_run_reads_back_what_was_recorded(tmp_path, root, monkeypatch, capsys):
    spec = spec_mod.load(
        write_spec(
            tmp_path,
            """
            agents:
              a: {system_prompt: x}
            tasks:
              - {id: schema, agent: a, prompt: do schema, review: {enabled: true}}
              - {id: api,    agent: a, prompt: do api, depends_on: [schema]}
            """,
        )
    )
    ask = "AskUserQuestion"
    FakeClient.scripts = {
        "do schema": {
            "tools": [
                # Two questions in one exchange, then a second exchange.
                (
                    ask,
                    {
                        "questions": [
                            {
                                "question": "Which database?",
                                "header": "DB",
                                "options": [
                                    {"label": "Postgres", "description": "server"},
                                    {"label": "SQLite", "description": "file"},
                                ],
                            },
                            {"question": "Anything else?", "options": []},
                        ]
                    },
                ),
                (ask, {"questions": [{"question": "Proceed?", "options": []}]}),
                ("Bash", {"command": "sudo rm /etc/hosts"}),
            ],
            "text": SCHEMA_OUTPUT,
            "cost": 0.0123,
        },
        "do api": {"tools": [], "text": API_OUTPUT, "cost": 0.5},
    }
    monkeypatch.setattr(runner, "ClaudeSDKClient", FakeClient)
    # The human at the terminal: picks option 2, then types free text twice.
    replies = iter(["2", 'keep it "simple"\u2028please', "yes"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(replies))

    # The reviewer for `schema`, stubbed as in test_review.py.
    async def reviewer(options, prompt):
        return ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="r",
            total_cost_usd=0.002,
            result="looks fine\nVERDICT: APPROVED",
        )

    monkeypatch.setattr(rv, "_run_session", reviewer)
    monkeypatch.setattr(rv, "_is_git_repo", lambda cwd: False)

    async def go():
        return await graph.run(spec, 2, False, True, run_id="rt1", session_root=root)

    results = anyio.run(go)
    assert {t: r.status for t, r in results.items()} == {"schema": "ok", "api": "ok"}

    session = ss.load_session("rt1", root)
    assert session.end is not None and session.end.error is None
    assert session.warnings == []
    assert list(session.tasks) == ["schema", "api"]

    # Everything the run returned is what the record holds, field for field.
    for tid, result in results.items():
        rec = session.tasks[tid]
        assert rec.text == result.text
        assert (rec.status, rec.model, rec.session_id) == (
            result.status,
            result.model,
            result.session_id,
        )
        assert (rec.cost_usd, rec.review_cost_usd, rec.total_cost_usd) == (
            result.cost_usd,
            result.review_cost_usd,
            result.total_cost_usd,
        )

    schema = session.tasks["schema"]
    # Output survives exactly, awkward characters and all.
    assert schema.text == SCHEMA_OUTPUT
    assert session.tasks["api"].text == API_OUTPUT
    # The record keeps the spec's prompt, not the upstream-prefixed one.
    assert session.tasks["api"].prompt == "do api"

    # Questions: what was asked, grouped by exchange, with the human's answers.
    assert [(q.exchange, q.question, q.answer, q.answer_recorded) for q in schema.questions] == [
        (1, "Which database?", "SQLite", True),
        (1, "Anything else?", 'keep it "simple"\u2028please', True),
        (2, "Proceed?", "yes", True),
    ]
    first = schema.questions[0]
    assert first.header == "DB"
    assert [(o.label, o.description) for o in first.options] == [
        ("Postgres", "server"),
        ("SQLite", "file"),
    ]
    assert len(schema.denials) == 1 and "sudo" in schema.denials[0]

    # Costs: review spend is kept separately and folded into the totals.
    assert schema.cost_usd == pytest.approx(0.0123)
    assert schema.review_cost_usd == pytest.approx(0.002)
    assert [(r.round, r.verdict) for r in schema.reviews] == [(1, rv.APPROVED)]
    assert session.total_cost_usd == pytest.approx(0.5143)
    assert session.end.total_cost_usd == pytest.approx(0.5143)
    [summary] = ss.list_sessions(root)
    assert summary.total_cost_usd == pytest.approx(0.5143)
    assert summary.finished
    assert summary.counts == {"ok": 2, "failed": 0, "skipped": 0, "not_run": 0}

    # The end-of-run line tells the user where the record is, even under --quiet.
    assert f"session recorded at {session.path}" in capsys.readouterr().err


def test_skipped_and_failed_tasks_are_recorded_with_their_reason(spec, root, monkeypatch):
    stub_tasks(
        monkeypatch,
        {"schema": TaskResult(task_id="schema", status="failed", reason="max_turns")},
    )
    anyio.run(lambda: graph.run(spec, 2, False, True, run_id="sk", session_root=root))

    session = ss.load_session("sk", root)
    assert session.tasks["schema"].status == "failed"
    assert session.tasks["schema"].reason == "max_turns"
    api = session.tasks["api"]
    assert api.status == "skipped"
    assert "schema" in api.reason
    assert session.counts == {"ok": 0, "failed": 1, "skipped": 1, "not_run": 0}


# --- interrupted runs ------------------------------------------------------


def test_run_killed_mid_write_reads_back_as_a_partial_session(spec, root):
    """The process dies (SIGKILL, power loss) after one task, part-way through
    writing the next: no `end` line, and a torn final line."""
    writer = ss.begin_session(spec, "crash1", root=root)
    writer.record_task(TaskResult(task_id="schema", status="ok", text="the schema", cost_usd=0.25))
    with writer.events_file.open("ab") as fh:
        fh.write(b'{"kind": "task", "task_id": "api", "text": "half writ')

    session = ss.load_session("crash1", root)
    assert not session.finished
    assert session.duration_s is None
    assert list(session.tasks) == ["schema"]
    assert session.tasks["schema"].text == "the schema"
    assert session.total_cost_usd == pytest.approx(0.25)
    assert session.counts == {"ok": 1, "failed": 0, "skipped": 0, "not_run": 1}
    # The torn line is reported, not fatal.
    assert len(session.warnings) == 1
    # The plan still says what the run meant to do.
    assert [p.id for p in session.header.tasks] == ["schema", "api"]

    # Asking for the task that never finished is a clear error, distinguishable
    # from a typo.
    with pytest.raises(ss.TaskNotFoundError) as info:
        ss.load_task("crash1", "api", root)
    assert info.value.planned is True

    [summary] = ss.list_sessions(root)
    assert (summary.run_id, summary.finished, summary.duration_s) == ("crash1", False, None)


def test_interrupted_graph_run_keeps_finished_tasks_and_says_why(tmp_path, root, monkeypatch):
    spec = spec_mod.load(
        write_spec(
            tmp_path,
            """
            agents:
              a: {system_prompt: x}
            tasks:
              - {id: schema, agent: a, prompt: p}
              - {id: api,    agent: a, prompt: p, depends_on: [schema]}
              - {id: docs,   agent: a, prompt: p, depends_on: [api]}
            """,
        )
    )
    stub_tasks(
        monkeypatch,
        {
            "schema": TaskResult(task_id="schema", status="ok", text="schema out", cost_usd=0.1),
            "api": KeyboardInterrupt(),
        },
    )
    with pytest.raises(KeyboardInterrupt):
        anyio.run(lambda: graph.run(spec, 2, False, True, run_id="ctrlc", session_root=root))

    session = ss.load_session("ctrlc", root)
    assert list(session.tasks) == ["schema"]
    assert session.tasks["schema"].text == "schema out"
    assert session.end is not None and session.end.error
    assert session.counts["not_run"] == 2
    with pytest.raises(ss.TaskNotFoundError):
        session.task("docs")


# --- the stored spec -------------------------------------------------------


def test_stored_spec_is_byte_for_byte_what_was_supplied(tmp_path, root, monkeypatch):
    # A BOM, CRLF line endings, comments, quotes, backslashes, a tab, U+2028,
    # non-ASCII, a block scalar with trailing spaces, and no final newline.
    original = (
        '\ufeff# Spec with "quotes", a back\\slash and caf\u00e9 \U0001f600\r\n'
        "agents:\r\n"
        "  a: {system_prompt: 'it''s \"x\"'}   # trailing comment\r\n"
        "tasks:\r\n"
        '  - {id: one, agent: a, prompt: "tab\\there \\u2028 and \\\\ \\"q\\""}\r\n'
        "  - id: two\r\n"
        "    agent: a\r\n"
        "    prompt: |\r\n"
        "      line one   \r\n"
        '      {"kind": "end"}\r\n'
        "\r\n"
        "# end, no newline"
    ).encode()
    path = tmp_path / "awkward.yaml"
    path.write_bytes(original)
    spec = spec_mod.load(path)
    stub_tasks(monkeypatch, {})

    anyio.run(
        lambda: graph.run(spec, 2, False, True, run_id="bytes", spec_path=path, session_root=root)
    )
    # Editing the spec after the run must not change what the record says ran.
    path.write_bytes(b"agents: {}\n")

    session = ss.load_session("bytes", root)
    assert session.spec_file.read_bytes() == original
    assert session.header.spec_verbatim is True
    assert session.header.spec_name == "awkward.yaml"
    assert browse.render_spec(session).encode() == original
    # The parsed prompt, which goes through JSON in events.jsonl, survives too.
    assert session.header.tasks[0].prompt == 'tab\there \u2028 and \\ "q"'
    assert session.tasks["two"].prompt == 'line one   \n{"kind": "end"}\n'


# --- listing ---------------------------------------------------------------


def test_sessions_list_most_recent_first(spec, root, monkeypatch):
    """Ordered by start time: not by creation order, run id, or finish state."""
    starts = {
        "ccc": datetime(2026, 3, 1, 12, 0, tzinfo=UTC),
        "bbb": datetime(2025, 12, 31, 23, 59, 59, tzinfo=UTC),
        "aaa": datetime(2026, 3, 1, 12, 0, 1, tzinfo=UTC),
    }
    pending = list(starts.values())

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return pending.pop(0)

    monkeypatch.setattr(ss, "datetime", Clock)
    writers = [ss.begin_session(spec, rid, root=root) for rid in starts]
    monkeypatch.setattr(ss, "datetime", datetime)  # real clock again for finalize
    writers[1].finalize()  # finishing a run does not move it up the list

    listed = ss.list_sessions(root)
    assert [s.run_id for s in listed] == ["aaa", "ccc", "bbb"]
    assert [s.started_at for s in listed] == sorted(starts.values(), reverse=True)


def test_listing_an_empty_or_missing_root_is_empty(root):
    assert ss.list_sessions(root) == []
    root.mkdir()
    assert ss.list_sessions(root) == []


# --- looking sessions and tasks up -----------------------------------------


@pytest.fixture
def three_sessions(spec, root):
    for rid in ("abc", "abc123", "abd456"):
        ss.begin_session(spec, rid, root=root).finalize()
    return root


@pytest.mark.parametrize(
    "query,expected",
    [
        ("abd", "abd456"),
        ("abc1", "abc123"),
        ("abc123", "abc123"),
        # An exact id wins even though it is also a prefix of another.
        ("abc", "abc"),
    ],
)
def test_run_id_prefix_resolves(three_sessions, query, expected):
    assert ss.load_session(query, three_sessions).run_id == expected


def test_ambiguous_prefix_is_an_error_naming_the_matches(three_sessions):
    with pytest.raises(ss.AmbiguousSessionError) as info:
        ss.load_session("ab", three_sessions)
    assert sorted(info.value.matches) == ["abc", "abc123", "abd456"]

    message = browse.render_error(info.value, sessions=ss.list_sessions(three_sessions))
    for rid in ("abc", "abc123", "abd456"):
        assert rid in message


@pytest.mark.parametrize("query", ["zzz", "", "../abc", "abc123x", "ABC"])
def test_unknown_run_id_is_an_error_not_a_crash(three_sessions, query):
    with pytest.raises(ss.SessionNotFoundError):
        ss.load_session(query, three_sessions)


def test_unknown_run_id_in_an_empty_store_is_an_error(root):
    with pytest.raises(ss.SessionNotFoundError) as info:
        ss.load_session("abc", root)
    assert "No sessions" in browse.render_error(info.value, sessions=ss.list_sessions(root))


def test_unknown_task_id_is_an_error_not_a_crash(spec, root):
    writer = ss.begin_session(spec, "tasks1", root=root)
    writer.record_task(TaskResult(task_id="schema", status="ok", text="x"))
    writer.finalize()

    with pytest.raises(ss.TaskNotFoundError) as info:
        ss.load_task("tasks1", "nope", root)
    # A typo, not a task the run never reached.
    assert info.value.planned is False
    # The CLI catches SessionStoreError; this must be one.
    assert isinstance(info.value, ss.SessionStoreError)

    session = ss.load_session("tasks1", root)
    message = browse.render_error(info.value, session=session)
    assert "schema" in message and "api" in message


# --- recording must never fail the run -------------------------------------


def _always(exc):
    def raiser(*args, **kwargs):
        raise exc

    return raiser


@pytest.mark.parametrize(
    "break_store",
    [
        pytest.param(
            lambda mp, tmp: mp.setattr(ss, "begin_session", _always(RuntimeError("store bug"))),
            id="begin-raises",
        ),
        pytest.param(
            lambda mp, tmp: mp.setattr(
                ss.SessionWriter, "record_task", _always(OSError(28, "No space left on device"))
            ),
            id="every-task-write-fails",
        ),
        pytest.param(
            lambda mp, tmp: mp.setattr(
                ss.SessionWriter, "finalize", _always(ss.SessionStoreError("already finalised"))
            ),
            id="finalize-raises",
        ),
    ],
)
def test_a_failing_store_does_not_fail_the_run(
    spec, root, tmp_path, monkeypatch, capsys, break_store
):
    break_store(monkeypatch, tmp_path)
    stub_tasks(monkeypatch, {})

    results = anyio.run(lambda: graph.run(spec, 2, False, True, run_id="brk", session_root=root))

    assert {t: r.status for t, r in results.items()} == {"schema": "ok", "api": "ok"}
    # ...and the user is told the record is missing or incomplete.
    err = capsys.readouterr().err
    assert "run brk:" in err
    assert "not recorded" in err or "incomplete" in err


def test_an_unwritable_session_root_does_not_fail_the_run(spec, tmp_path, monkeypatch, capsys):
    not_a_dir = tmp_path / "sessions"
    not_a_dir.write_text("a file where the session directory should be")
    stub_tasks(monkeypatch, {})

    results = anyio.run(
        lambda: graph.run(spec, 2, False, True, run_id="ro", session_root=not_a_dir)
    )

    assert all(r.ok for r in results.values())
    assert "not recorded" in capsys.readouterr().err


def test_a_transient_write_failure_is_recovered_before_the_run_ends(spec, root, monkeypatch):
    real = ss.SessionWriter.record_task
    failed = []

    def flaky(self, result):
        if result.task_id == "schema" and not failed:
            failed.append(result.task_id)
            raise OSError(28, "No space left on device")
        return real(self, result)

    monkeypatch.setattr(ss.SessionWriter, "record_task", flaky)
    stub_tasks(monkeypatch, {})

    anyio.run(lambda: graph.run(spec, 2, False, True, run_id="flaky", session_root=root))

    assert failed == ["schema"]
    session = ss.load_session("flaky", root)
    assert session.finished
    assert set(session.tasks) == {"schema", "api"}
