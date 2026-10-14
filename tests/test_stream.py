"""The streaming layer: StreamHub's buffering, snapshots, thread safety and
terminal handover, plus the runner's use of it end to end.

No curses here. The hub is where the logic lives; the viewer only draws what a
snapshot hands it, and is left to manual testing. The end-to-end tests stub the
Claude session at `runner.ClaudeSDKClient`, as test_session_store.py does, so
the real runner, graph and permission callback run offline.

Anything that could deadlock runs on a daemon thread joined with a timeout, so
a regression fails the test instead of hanging the suite.
"""

import builtins
import sys
import textwrap
import threading
import time

import anyio
import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    StreamEvent,
    ToolPermissionContext,
    ToolUseBlock,
)
from orchestrator import graph, runner, stream
from orchestrator import spec as spec_mod
from orchestrator.stream import Line, StreamHub, StreamUpdate


def text(hub, task_id, *fragments):
    for f in fragments:
        hub.publish(StreamUpdate(task_id=task_id, kind="text", text=f, at=time.monotonic()))


def texts(snap):
    """The text of each buffered line, for tests that care about content, not kind."""
    return [ln.text for ln in snap.lines]


def run_in_thread(fn, timeout=2.0):
    """Run fn on a daemon thread; fail (rather than hang) if it does not return."""
    out = {}

    def target():
        try:
            out["value"] = fn()
        except BaseException as exc:  # surfaced in the test thread below
            out["error"] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), f"{fn} did not return within {timeout}s"
    if "error" in out:
        raise out["error"]
    return out.get("value")


@pytest.fixture
def fast_switching():
    """Switch threads far more often than the default 5ms, so races show up."""
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    yield
    sys.setswitchinterval(old)


# --- text accumulation -----------------------------------------------------


def test_fragments_accumulate_into_lines():
    hub = StreamHub()
    text(hub, "t", "He", "llo\nwor", "ld\n")
    snap = hub.snapshot("t")
    assert texts(snap) == ["Hello", "world"]
    assert snap.pending == ""


def test_trailing_fragment_stays_pending_and_is_not_lost():
    hub = StreamHub()
    text(hub, "t", "done\nhalf a li", "ne")
    snap = hub.snapshot("t")
    assert texts(snap) == ["done"]
    assert snap.pending == "half a line"

    # It is completed by the next newline, not dropped or duplicated.
    text(hub, "t", " and the rest\n")
    snap = hub.snapshot("t")
    assert texts(snap) == ["done", "half a line and the rest"]
    assert snap.pending == ""


def test_non_text_event_flushes_pending_text_first():
    # A tool call mid-line must not be glued onto, or lose, the open text line.
    hub = StreamHub()
    text(hub, "t", "let me look")
    hub.publish(StreamUpdate(task_id="t", kind="tool", text="Read", at=time.monotonic()))
    text(hub, "t", "found it\n")
    assert texts(hub.snapshot("t")) == ["let me look", "Read", "found it"]


def test_text_fragments_produce_text_lines():
    hub = StreamHub()
    text(hub, "t", "He", "llo\nwor", "ld")
    hub.publish(StreamUpdate(task_id="t", kind="tool", text="Read", at=time.monotonic()))
    # Both the newline-completed line and the one closed by the tool's flush.
    assert hub.snapshot("t").lines[:2] == [Line("text", "Hello"), Line("text", "world")]


def test_tool_update_produces_a_tool_line():
    hub = StreamHub()
    text(hub, "t", "let me look")
    hub.publish(StreamUpdate(task_id="t", kind="tool", text="Read", at=time.monotonic()))
    text(hub, "t", "found it\n")
    # The open text closed by the tool stays "text"; only the tool's own line is "tool".
    assert hub.snapshot("t").lines == [
        Line("text", "let me look"),
        Line("tool", "Read"),
        Line("text", "found it"),
    ]


def test_note_update_produces_note_lines():
    hub = StreamHub()
    hub.publish(StreamUpdate(task_id="t", kind="note", text="one\ntwo", at=time.monotonic()))
    # Every line of a multi-line event carries its kind, including the last,
    # which is completed by the trailing flush rather than a newline.
    assert hub.snapshot("t").lines == [Line("note", "one"), Line("note", "two")]


def test_wrapped_long_event_keeps_its_kind():
    hub = StreamHub()
    long = "y" * (stream.MAX_PENDING + 5)
    hub.publish(StreamUpdate(task_id="t", kind="tool", text=long, at=time.monotonic()))
    snap = hub.snapshot("t")
    assert {ln.kind for ln in snap.lines} == {"tool"}
    assert "".join(texts(snap)) == long
    assert snap.pending == ""


def test_hidden_kinds_still_occupy_the_ring_buffer():
    # Deliberate: tool lines are kept even though a viewer may hide them, so
    # toggling them on can show them retroactively. They count against scrollback.
    hub = StreamHub(scrollback=3)
    text(hub, "t", "a\n")
    for name in ("Read", "Grep", "Edit"):
        hub.publish(StreamUpdate(task_id="t", kind="tool", text=name, at=time.monotonic()))
    assert hub.snapshot("t").lines == [
        Line("tool", "Read"),
        Line("tool", "Grep"),
        Line("tool", "Edit"),
    ]


def test_crlf_split_across_fragments_is_one_newline():
    hub = StreamHub()
    text(hub, "t", "a\r", "\nb\r\n", "c\rd\n")
    assert texts(hub.snapshot("t")) == ["a", "b", "c", "d"]


def test_output_without_newlines_is_bounded():
    hub = StreamHub()
    text(hub, "t", "x" * (stream.MAX_PENDING * 2 + 10))
    snap = hub.snapshot("t")
    assert len(snap.pending) <= stream.MAX_PENDING
    # Wrapped, not truncated: every character is still there.
    assert "".join(texts(snap)) + snap.pending == "x" * (stream.MAX_PENDING * 2 + 10)


# --- ring buffer -----------------------------------------------------------


def test_ring_buffer_keeps_the_most_recent_lines():
    hub = StreamHub(scrollback=3)
    text(hub, "t", *(f"line {i}\n" for i in range(10)))
    assert texts(hub.snapshot("t")) == ["line 7", "line 8", "line 9"]


def test_ring_buffer_bound_is_per_task():
    hub = StreamHub(scrollback=2)
    text(hub, "a", "a1\na2\na3\n")
    text(hub, "b", "b1\n")
    assert texts(hub.snapshot("a")) == ["a2", "a3"]
    assert texts(hub.snapshot("b")) == ["b1"]


def test_scrollback_must_be_positive():
    with pytest.raises(ValueError):
        StreamHub(scrollback=0)


# --- unseen / mark_seen ----------------------------------------------------


def test_unseen_counts_updates_and_mark_seen_resets_it():
    hub = StreamHub()
    hub.register("t", "a", "m")
    assert hub.snapshot("t").unseen == 0

    text(hub, "t", "one", " two\n")
    hub.set_status("t", "running")
    assert hub.snapshot("t").unseen == 3

    hub.mark_seen("t")
    assert hub.snapshot("t").unseen == 0
    # Output is untouched by marking seen.
    assert texts(hub.snapshot("t")) == ["one two"]

    text(hub, "t", "more\n")
    assert hub.snapshot("t").unseen == 1


def test_repeating_the_same_status_is_not_an_update():
    # runner.run_task re-registers and re-sets status; that must not light up
    # a task the viewer has already looked at.
    hub = StreamHub()
    hub.set_status("t", "running")
    hub.mark_seen("t")
    hub.set_status("t", "running")
    assert hub.snapshot("t").unseen == 0


def test_mark_seen_on_unknown_task_is_ignored():
    hub = StreamHub()
    hub.mark_seen("nope")
    assert hub.task_ids() == []


# --- registration ----------------------------------------------------------


def test_task_ids_preserve_registration_order():
    hub = StreamHub()
    for tid in ("schema", "api", "docs", "tests"):
        hub.register(tid, "a", "m")
    # Re-registering and publishing do not reorder.
    hub.register("schema", "b", "m2")
    text(hub, "docs", "x\n")
    assert hub.task_ids() == ["schema", "api", "docs", "tests"]
    assert [s.task_id for s in hub.snapshots()] == ["schema", "api", "docs", "tests"]


def test_reregistering_updates_labels_and_keeps_output():
    hub = StreamHub()
    hub.register("t", "old", "m1")
    text(hub, "t", "kept\n")
    hub.register("t", "new", "m2")
    snap = hub.snapshot("t")
    assert (snap.agent, snap.model, texts(snap)) == ("new", "m2", ["kept"])


def test_first_registration_sets_wave_and_depends_on():
    hub = StreamHub()
    hub.register("api", "a", "m", wave=1, depends_on=["schema"])
    snap = hub.snapshot("api")
    assert (snap.wave, snap.depends_on) == (1, ("schema",))
    assert isinstance(snap.depends_on, tuple)


def test_first_registration_without_graph_shape_defaults_to_wave_zero_no_deps():
    hub = StreamHub()
    hub.register("t", "a", "m")
    text(hub, "late", "x\n")  # implicit registration gets the same defaults
    for tid in ("t", "late"):
        snap = hub.snapshot(tid)
        assert (snap.wave, snap.depends_on) == (0, ())


def test_three_argument_reregistration_keeps_wave_and_depends_on():
    # graph.run pre-registers with the spec; runner re-registers with labels only
    # as the task starts. That second call must not reset the graph shape.
    hub = StreamHub()
    hub.register("api", "a", "m", wave=2, depends_on=["schema", "auth"])
    hub.register("api", "b", "m2")
    snap = hub.snapshot("api")
    assert (snap.agent, snap.model) == ("b", "m2")
    assert (snap.wave, snap.depends_on) == (2, ("schema", "auth"))


def test_explicit_values_on_reregistration_update_the_graph_shape():
    hub = StreamHub()
    hub.register("api", "a", "m", wave=2, depends_on=["schema"])
    hub.register("api", "a", "m", wave=3, depends_on=["docs"])
    snap = hub.snapshot("api")
    assert (snap.wave, snap.depends_on) == (3, ("docs",))

    # An explicit empty sequence is a value, not "leave as is".
    hub.register("api", "a", "m", wave=0, depends_on=[])
    snap = hub.snapshot("api")
    assert (snap.wave, snap.depends_on) == (0, ())


def test_register_copies_the_callers_depends_on():
    hub = StreamHub()
    deps = ["schema"]
    hub.register("api", "a", "m", depends_on=deps)
    deps.append("injected")
    assert hub.snapshot("api").depends_on == ("schema",)


def test_publish_to_unregistered_task_registers_it_rather_than_dropping():
    hub = StreamHub()
    text(hub, "late", "hi\n")
    assert hub.task_ids() == ["late"]
    assert texts(hub.snapshot("late")) == ["hi"]


# --- snapshots -------------------------------------------------------------


def test_snapshot_is_a_copy():
    hub = StreamHub()
    text(hub, "t", "a\nb\n")
    snap = hub.snapshot("t")
    snap.lines.append(Line("text", "injected"))
    snap.lines[0] = Line("tool", "changed")
    snap.unseen = 99
    snap.status = "ok"

    again = hub.snapshot("t")
    assert texts(again) == ["a", "b"]
    assert again.unseen == 1  # one publish, however many lines it carried
    assert again.status == "pending"

    # And the hub moving on does not change a snapshot already taken.
    text(hub, "t", "c\n")
    assert texts(again) == ["a", "b"]


def test_snapshot_depends_on_cannot_affect_the_hub():
    hub = StreamHub()
    hub.register("api", "a", "m", wave=1, depends_on=["schema"])
    snap = hub.snapshot("api")
    assert isinstance(snap.depends_on, tuple)
    with pytest.raises(AttributeError):
        snap.depends_on.append("injected")  # type: ignore[attr-defined]
    # Rebinding the field on the copy leaves the hub alone too.
    snap.depends_on = ("other",)
    snap.wave = 9

    again = hub.snapshot("api")
    assert (again.wave, again.depends_on) == (1, ("schema",))


def test_snapshot_of_unknown_task_raises_keyerror():
    hub = StreamHub()
    hub.register("known", "a", "m")
    with pytest.raises(KeyError):
        hub.snapshot("unknown")


def test_snapshot_taken_mid_publish_is_self_consistent(fast_switching):
    """A writer streams numbered lines in small fragments while a reader
    snapshots continuously. Every snapshot must look like some instant of the
    stream: complete lines contiguous and in order, `pending` a prefix of the
    next line, and `unseen` equal to the fragments that produced exactly that."""
    hub = StreamHub(scrollback=10_000)
    n_lines = 400
    hub.register("t", "a", "m")

    def line(i):
        return f"line {i:04d} " + "z" * (i % 7)

    fragments = []
    for i in range(n_lines):
        s = line(i) + "\n"
        fragments += [s[k : k + 3] for k in range(0, len(s), 3)]

    done = threading.Event()
    problems = []

    def writer():
        text(hub, "t", *fragments)
        done.set()

    def reader():
        checked = 0
        while not done.is_set() or checked == 0:
            snap = hub.snapshot("t")
            n = len(snap.lines)
            if texts(snap) != [line(i) for i in range(n)]:
                problems.append(("lines", n, texts(snap)[-3:]))
            if n < n_lines and not (line(n) + "\n").startswith(snap.pending):
                problems.append(("pending", n, snap.pending))
            consumed = sum(len(f) for f in fragments[: snap.unseen])
            produced = sum(len(line(i)) + 1 for i in range(n)) + len(snap.pending)
            if consumed != produced:
                problems.append(("unseen", snap.unseen, consumed, produced))
            checked += 1
        return checked

    r = threading.Thread(target=lambda: problems.append(("checked", reader())), daemon=True)
    r.start()
    run_in_thread(writer, timeout=10)
    r.join(10)
    assert not r.is_alive()

    checks = [p for p in problems if p[0] == "checked"]
    assert [p for p in problems if p[0] != "checked"] == []
    assert checks and checks[0][1] > 0
    assert texts(hub.snapshot("t")) == [line(i) for i in range(n_lines)]


# --- thread safety ---------------------------------------------------------


def test_concurrent_whole_line_publishes_lose_nothing_and_do_not_interleave(fast_switching):
    """Many threads publish complete lines to one task at once. Every line must
    arrive exactly once and intact, and each thread's lines in its own order."""
    hub = StreamHub(scrollback=100_000)
    n_threads, per_thread = 8, 300
    start = threading.Barrier(n_threads)

    def worker(w):
        start.wait()
        for i in range(per_thread):
            text(hub, "shared", f"w{w}-{i:04d}-" + "abcdefghij" * 3 + "\n")

    threads = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
        assert not t.is_alive()

    snap = hub.snapshot("shared")
    assert snap.pending == ""
    assert snap.unseen == n_threads * per_thread
    expected = {
        f"w{w}-{i:04d}-" + "abcdefghij" * 3 for w in range(n_threads) for i in range(per_thread)
    }
    assert len(snap.lines) == len(expected)
    assert set(texts(snap)) == expected
    for w in range(n_threads):
        mine = [ln for ln in texts(snap) if ln.startswith(f"w{w}-")]
        assert mine == sorted(mine)


def test_concurrent_fragment_streams_to_separate_tasks_stay_separate(fast_switching):
    """The real shape of a run: several tasks each stream token fragments at
    once. No task's partial line may pick up another task's fragments."""
    hub = StreamHub(scrollback=100_000)
    n_threads, per_thread = 8, 200
    start = threading.Barrier(n_threads)

    def worker(w):
        start.wait()
        for i in range(per_thread):
            s = f"task{w} line {i:04d}\n"
            text(hub, f"t{w}", *(s[k : k + 2] for k in range(0, len(s), 2)))

    threads = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
        assert not t.is_alive()

    assert sorted(hub.task_ids()) == sorted(f"t{w}" for w in range(n_threads))
    for w in range(n_threads):
        snap = hub.snapshot(f"t{w}")
        assert texts(snap) == [f"task{w} line {i:04d}" for i in range(per_thread)]
        assert snap.pending == ""


# --- terminal handover -----------------------------------------------------


@pytest.fixture
def long_ack_timeout(monkeypatch):
    """Make any accidental wait obvious: far longer than the join timeouts."""
    monkeypatch.setattr(stream, "SUSPEND_ACK_TIMEOUT", 30.0)


def test_suspended_does_not_block_without_a_viewer(long_ack_timeout):
    hub = StreamHub()

    def prompt():
        started = time.monotonic()
        with hub.suspended():
            assert hub.suspend_requested()
        return time.monotonic() - started

    assert run_in_thread(prompt) < 0.5
    assert not hub.suspend_requested()


def test_suspended_does_not_block_nested_or_after_a_viewer_has_left(long_ack_timeout):
    hub = StreamHub()
    hub.attach_viewer()
    hub.detach_viewer()

    def prompts():
        for _ in range(20):
            with hub.suspended(), hub.suspended():
                pass
        return hub.suspend_requested()

    assert run_in_thread(prompts) is False


def test_suspended_waits_for_an_attached_viewer_to_acknowledge(long_ack_timeout):
    hub = StreamHub()
    hub.attach_viewer()
    entered = threading.Event()

    def prompt():
        with hub.suspended():
            entered.set()

    t = threading.Thread(target=prompt, daemon=True)
    t.start()
    # The viewer still holds the terminal, so the prompt must not draw yet.
    assert not entered.wait(0.2)
    assert hub.suspend_requested()

    hub.acknowledge_suspend()
    assert entered.wait(2)
    t.join(2)
    assert hub.reclaim_terminal(timeout=1) is True
    hub.detach_viewer()


def test_detaching_viewer_releases_a_waiting_prompt(long_ack_timeout):
    # A viewer that crashes out must not leave a question stuck.
    hub = StreamHub()
    hub.attach_viewer()
    t = threading.Thread(target=lambda: hub.suspended().__enter__(), daemon=True)
    t.start()
    time.sleep(0.05)
    hub.detach_viewer()
    t.join(2)
    assert not t.is_alive()


def test_back_to_back_prompts_need_only_one_acknowledgement(long_ack_timeout):
    # Regression: ending a suspension used to mark the viewer as holding the
    # terminal again, so the next prompt waited out the whole ack timeout.
    hub = StreamHub()
    hub.attach_viewer()

    def viewer_ack_once():
        while not hub.suspend_requested():
            time.sleep(0.001)
        hub.acknowledge_suspend()

    v = threading.Thread(target=viewer_ack_once, daemon=True)
    v.start()

    def burst():
        for _ in range(3):
            with hub.suspended():
                pass

    run_in_thread(burst)
    v.join(2)
    hub.detach_viewer()


def test_stray_acknowledgement_does_not_skip_the_next_wait(long_ack_timeout):
    hub = StreamHub()
    hub.attach_viewer()
    hub.acknowledge_suspend()  # not suspended: must be ignored

    entered = threading.Event()
    t = threading.Thread(target=lambda: (hub.suspended().__enter__(), entered.set()), daemon=True)
    t.start()
    assert not entered.wait(0.2)
    hub.detach_viewer()
    t.join(2)


def test_reclaim_terminal_fails_while_a_suspension_is_active():
    hub = StreamHub()
    with hub.suspended():
        assert hub.reclaim_terminal(timeout=0) is False
    assert hub.reclaim_terminal(timeout=0) is True


# --- the runner, with and without a hub ------------------------------------

SPEC = """
agents:
  a: {system_prompt: x}
tasks:
  - {id: one,    agent: a, prompt: do one}
  - {id: broken, agent: a, prompt: do broken, cwd: does-not-exist}
  - {id: after,  agent: a, prompt: do after, depends_on: [broken]}
"""


def delta(t):
    return StreamEvent(
        uuid="u",
        session_id="s",
        event={"type": "content_block_delta", "delta": {"type": "text_delta", "text": t}},
    )


def block_stop():
    return StreamEvent(uuid="u", session_id="s", event={"type": "content_block_stop"})


class FakeClient:
    """Stands in for ClaudeSDKClient: streams text as deltas, calls a tool, asks
    one question through the real permission callback, then finishes."""

    seen_options: list = []

    def __init__(self, options):
        self.options = options
        FakeClient.seen_options.append(options)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def query(self, prompt):
        pass

    async def receive_response(self):
        # The real client only sends deltas when partial messages are on.
        if self.options.include_partial_messages:
            for t in ("Hel", "lo\nwor", "ld"):
                yield delta(t)
            yield block_stop()
        yield AssistantMessage(
            content=[ToolUseBlock(id="1", name="Read", input={"file_path": "x"})],
            model="m",
        )
        await self.options.can_use_tool(
            "AskUserQuestion",
            {"questions": [{"question": "Proceed?", "options": []}]},
            ToolPermissionContext(),
        )
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="sess",
            total_cost_usd=0.01,
            result="Hello\nworld",
        )


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    p = tmp_path / "spec.yaml"
    p.write_text(textwrap.dedent(SPEC))
    spec = spec_mod.load(p)
    FakeClient.seen_options = []
    monkeypatch.setattr(runner, "ClaudeSDKClient", FakeClient)
    monkeypatch.setattr(builtins, "input", lambda prompt="": "yes")
    return spec, tmp_path / "sessions"


def run_graph(spec, root, hub):
    async def go():
        with anyio.fail_after(10):  # a handover deadlock fails instead of hanging
            return await graph.run(spec, 2, False, True, session_root=root, hub=hub)

    return anyio.run(go)


def test_run_without_a_hub_behaves_as_before(stubbed, long_ack_timeout):
    spec, root = stubbed
    results = run_graph(spec, root, hub=None)

    assert {t: r.status for t, r in results.items()} == {
        "one": "ok",
        "broken": "failed",
        "after": "skipped",
    }
    one = results["one"]
    assert one.text == "Hello\nworld"
    assert one.questions == [
        {"questions": [{"question": "Proceed?", "options": []}], "answers": {"Proceed?": "yes"}}
    ]
    assert "cwd does not exist" in results["broken"].reason
    # Headless runs do not ask the SDK for token-level streaming.
    [opts] = FakeClient.seen_options
    assert opts.include_partial_messages is False


def test_run_with_a_hub_streams_each_task_into_it(stubbed, long_ack_timeout):
    spec, root = stubbed
    hub = StreamHub()
    results = run_graph(spec, root, hub=hub)

    # Same outcome as headless: the hub only observes.
    assert {t: r.status for t, r in results.items()} == {
        "one": "ok",
        "broken": "failed",
        "after": "skipped",
    }
    assert results["one"].text == "Hello\nworld"
    [opts] = FakeClient.seen_options
    assert opts.include_partial_messages is True

    assert hub.task_ids() == ["one", "broken", "after"]
    one = hub.snapshot("one")
    assert (one.agent, one.status) == ("a", "ok")
    assert one.model
    # The open line "world" is closed at block stop, before the tool line.
    assert one.lines == [Line("text", "Hello"), Line("text", "world"), Line("tool", "Read")]
    assert one.pending == ""

    broken = hub.snapshot("broken")
    assert broken.status == "failed"
    assert any("cwd does not exist" in ln for ln in texts(broken))
    assert hub.snapshot("after").status == "skipped"
    # The question ran with no viewer attached and did not leave a suspension open.
    assert not hub.suspend_requested()
