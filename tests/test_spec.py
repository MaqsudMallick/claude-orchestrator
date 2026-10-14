import re
import textwrap
from pathlib import Path

import pytest
from orchestrator import spec as spec_mod
from orchestrator.models import MODEL_BY_DIFFICULTY, model_for
from pydantic import ValidationError

BASE = """
agents:
  a:
    system_prompt: hello
tasks:
  - id: one
    agent: a
    prompt: do a thing
"""


def write(tmp_path, body):
    p = tmp_path / "spec.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_loads_minimal_spec(tmp_path):
    s = spec_mod.load(write(tmp_path, BASE))
    assert [t.id for t in s.tasks] == ["one"]
    # unset on the task; resolved through defaults
    assert s.tasks[0].difficulty is None
    assert s.difficulty_for(s.tasks[0]) == "medium"
    assert s.base_dir == tmp_path


def test_difficulty_aliases_are_the_only_model_selector():
    assert model_for("easy") == "claude-haiku-5-5"
    assert model_for("medium") == "claude-opus-5-5"
    assert model_for("hard") == "claude-fable-5-1"
    with pytest.raises(ValueError, match="unknown difficulty"):
        model_for("claude-opus-5-5")


def test_raw_model_id_is_rejected_as_difficulty(tmp_path):
    body = BASE + "    difficulty: claude-opus-5-5\n"
    with pytest.raises(ValidationError, match="difficulty"):
        spec_mod.load(write(tmp_path, body))


def test_unknown_agent_is_caught(tmp_path):
    body = """
    agents:
      a: {system_prompt: x}
    tasks:
      - id: one
        agent: nope
        prompt: p
    """
    with pytest.raises(ValueError, match="unknown agent"):
        spec_mod.load(write(tmp_path, body))


def test_unknown_dependency_is_caught(tmp_path):
    body = """
    agents:
      a: {system_prompt: x}
    tasks:
      - id: one
        agent: a
        prompt: p
        depends_on: [ghost]
    """
    with pytest.raises(ValueError, match="unknown task"):
        spec_mod.load(write(tmp_path, body))


def test_duplicate_ids_are_caught(tmp_path):
    body = """
    agents:
      a: {system_prompt: x}
    tasks:
      - {id: one, agent: a, prompt: p}
      - {id: one, agent: a, prompt: q}
    """
    with pytest.raises(ValueError, match="duplicate task ids"):
        spec_mod.load(write(tmp_path, body))


def test_cycle_is_caught_before_running(tmp_path):
    body = """
    agents:
      a: {system_prompt: x}
    tasks:
      - {id: one, agent: a, prompt: p, depends_on: [two]}
      - {id: two, agent: a, prompt: q, depends_on: [one]}
    """
    with pytest.raises(ValueError, match="dependency cycle"):
        spec_mod.load(write(tmp_path, body))


def test_reserved_id_is_rejected(tmp_path):
    body = """
    agents:
      a: {system_prompt: x}
    tasks:
      - {id: END, agent: a, prompt: p}
    """
    with pytest.raises(ValueError, match="reserved"):
        spec_mod.load(write(tmp_path, body))


def test_unknown_field_is_rejected(tmp_path):
    body = BASE + "    modle: typo\n"
    with pytest.raises(ValidationError, match="modle"):
        spec_mod.load(write(tmp_path, body))


def test_every_difficulty_maps_to_a_model():
    from orchestrator.spec import Task

    for d in MODEL_BY_DIFFICULTY:
        t = Task(id="t", agent="a", prompt="p", difficulty=d)
        assert model_for(t.difficulty)


def test_defaults_difficulty_applies_to_tasks_that_omit_it(tmp_path):
    body = """
    defaults:
      difficulty: easy
    agents:
      a: {system_prompt: x}
    tasks:
      - {id: cheap, agent: a, prompt: p}
      - {id: pricey, agent: a, prompt: p, difficulty: hard}
    """
    s = spec_mod.load(write(tmp_path, body))
    # regression: defaults.difficulty used to be shadowed by a task-level
    # default of "medium", silently running every task on the wrong model.
    assert s.difficulty_for(s.by_id["cheap"]) == "easy"
    assert model_for(s.difficulty_for(s.by_id["cheap"])) == "claude-haiku-5-5"
    assert s.difficulty_for(s.by_id["pricey"]) == "hard"


def test_waves_linear_chain_is_one_id_per_wave(tmp_path):
    body = """
    agents:
      a: {system_prompt: x}
    tasks:
      - {id: one, agent: a, prompt: p}
      - {id: two, agent: a, prompt: p, depends_on: [one]}
      - {id: three, agent: a, prompt: p, depends_on: [two]}
    """
    s = spec_mod.load(write(tmp_path, body))
    assert s.waves() == [["one"], ["two"], ["three"]]


def test_waves_independent_tasks_share_a_wave(tmp_path):
    body = """
    agents:
      a: {system_prompt: x}
    tasks:
      - {id: one, agent: a, prompt: p}
      - {id: two, agent: a, prompt: p}
      - {id: three, agent: a, prompt: p}
    """
    s = spec_mod.load(write(tmp_path, body))
    assert s.waves() == [["one", "two", "three"]]


def test_waves_diamond(tmp_path):
    body = """
    agents:
      a: {system_prompt: x}
    tasks:
      - {id: schema, agent: a, prompt: p}
      - {id: api, agent: a, prompt: p, depends_on: [schema]}
      - {id: docs, agent: a, prompt: p, depends_on: [schema]}
      - {id: tests, agent: a, prompt: p, depends_on: [api, docs]}
    """
    s = spec_mod.load(write(tmp_path, body))
    assert s.waves() == [["schema"], ["api", "docs"], ["tests"]]


def test_waves_order_within_a_wave_follows_spec_order(tmp_path):
    # `late` is listed first and pulls `x` into the sorter before `first`, so
    # the sorter alone would yield ["x", "first"]. Spec order puts `first` ahead.
    body = """
    agents:
      a: {system_prompt: x}
    tasks:
      - {id: late, agent: a, prompt: p, depends_on: [x]}
      - {id: first, agent: a, prompt: p}
      - {id: x, agent: a, prompt: p}
    """
    s = spec_mod.load(write(tmp_path, body))
    assert s.waves() == [["first", "x"], ["late"]]
    # Not alphabetical either: reversing the listing reverses the wave.
    body_rev = """
    agents:
      a: {system_prompt: x}
    tasks:
      - {id: x, agent: a, prompt: p}
      - {id: first, agent: a, prompt: p}
      - {id: late, agent: a, prompt: p, depends_on: [x]}
    """
    s = spec_mod.load(write(tmp_path, body_rev))
    assert s.waves() == [["x", "first"], ["late"]]


# --------------------------------------------------------------------------
# Strict key checking: a key written twice in one mapping is rejected at load.
#
# PyYAML's SafeLoader accepts duplicates and keeps the last value, so before
# this change every spec below loaded without complaint and silently dropped
# the first value. Each "duplicate" spec here is otherwise valid: if the check
# regressed, load() would succeed and the test would fail on "DID NOT RAISE",
# rather than passing on some unrelated validation error.
#
# The exception type is not part of the contract (the CLI reports any exception
# as "invalid spec: ..."), so these assert on the message instead: it must name
# the file, the line and the key, because that is what makes it fixable.
# --------------------------------------------------------------------------

SPEC_NAME = "strict-keys-spec.yaml"


def write_named(tmp_path, body, name=SPEC_NAME):
    """Write a dedented spec under a distinctive file name, so a message that
    names "the file" can be told apart from one that merely says "spec"."""
    p = tmp_path / name
    text = textwrap.dedent(body)
    p.write_text(text)
    return p, text


def lines_of(text, needle):
    """1-based line numbers of every line whose stripped form starts with `needle`."""
    return [i for i, line in enumerate(text.splitlines(), 1) if line.strip().startswith(needle)]


def assert_names_duplicate(exc_info, path, text, key, needle):
    """The message names the file, the key and the line of the duplicate.

    Which of the two occurrences counts as "the line" is left open: either
    points a reader at the problem. The path is stripped before looking for the
    number, so digits in pytest's temp directory cannot satisfy the check.
    """
    msg = str(exc_info.value)
    assert path.name in msg, msg
    assert key in msg, msg
    occurrences = lines_of(text, needle)
    assert len(occurrences) == 2, occurrences  # guard the fixture itself
    rest = msg.replace(str(path), "").replace(path.name, "")
    assert any(re.search(rf"(?<!\d){n}(?!\d)", rest) for n in occurrences), (
        f"expected line {occurrences} in: {msg}"
    )


def test_duplicate_prompt_in_task_is_rejected_and_named(tmp_path):
    body = """
    agents:
      a: {system_prompt: x}
    tasks:
      - id: one
        agent: a
        prompt: FIRST
        prompt: SECOND
    """
    path, text = write_named(tmp_path, body)
    with pytest.raises(Exception) as exc_info:
        spec_mod.load(path)
    assert_names_duplicate(exc_info, path, text, "prompt", "prompt:")


def test_duplicate_top_level_agents_is_rejected(tmp_path):
    # Both blocks define `a`, so last-wins would load cleanly; only the
    # duplicate check can reject this.
    body = """
    agents:
      a: {system_prompt: first}
    agents:
      a: {system_prompt: second}
    tasks:
      - {id: one, agent: a, prompt: p}
    """
    path, text = write_named(tmp_path, body)
    with pytest.raises(Exception) as exc_info:
        spec_mod.load(path)
    assert_names_duplicate(exc_info, path, text, "agents", "agents:")


def test_duplicate_agent_name_inside_agents_is_rejected(tmp_path):
    body = """
    agents:
      dev:
        system_prompt: careful reviewer
      dev:
        system_prompt: reckless cowboy
    tasks:
      - {id: one, agent: dev, prompt: p}
    """
    path, text = write_named(tmp_path, body)
    with pytest.raises(Exception) as exc_info:
        spec_mod.load(path)
    assert_names_duplicate(exc_info, path, text, "dev", "dev:")


def test_duplicate_in_flow_mapping_is_rejected(tmp_path):
    # Same mistake in `{...}` style, which the existing specs use heavily.
    body = """
    agents:
      a: {system_prompt: x}
    tasks:
      - {id: one, agent: a, prompt: FIRST, prompt: SECOND}
    """
    path, _ = write_named(tmp_path, body)
    with pytest.raises(Exception, match="prompt"):
        spec_mod.load(path)


def test_duplicate_written_with_different_quoting_is_rejected(tmp_path):
    # `prompt` and `'prompt'` construct the same key, so the document would
    # still lose the first value; the check must compare what the keys mean.
    body = """
    agents:
      a: {system_prompt: x}
    tasks:
      - id: one
        agent: a
        prompt: FIRST
        'prompt': SECOND
    """
    path, _ = write_named(tmp_path, body)
    with pytest.raises(Exception, match="prompt"):
        spec_mod.load(path)


def test_merge_key_override_still_loads_and_override_wins(tmp_path):
    # After flattening, `description` appears twice in `reviewer` and `prompt`
    # twice in `two`. Written, each mapping holds `<<` plus distinct keys, so
    # this is valid YAML and must load. A check run after the merge breaks it.
    body = """
    agents:
      base: &base_agent
        system_prompt: shared system prompt
        description: from base
        tools: [Read, Grep]
      reviewer:
        <<: *base_agent
        description: overridden locally
    tasks:
      - &base_task
        id: one
        agent: base
        prompt: base prompt
        difficulty: easy
      - <<: *base_task
        id: two
        prompt: overridden prompt
        depends_on: [one]
    """
    path, _ = write_named(tmp_path, body)
    s = spec_mod.load(path)

    reviewer = s.agents["reviewer"]
    assert reviewer.description == "overridden locally"
    # Merged keys that were not overridden still come through.
    assert reviewer.system_prompt == "shared system prompt"
    assert reviewer.tools == ["Read", "Grep"]
    # The anchor's own mapping is untouched by the override.
    assert s.agents["base"].description == "from base"

    two = s.by_id["two"]
    assert two.prompt == "overridden prompt"
    assert two.agent == "base"
    assert two.difficulty == "easy"
    assert s.by_id["one"].prompt == "base prompt"
    assert s.waves() == [["one"], ["two"]]


def test_merge_of_several_anchors_sharing_a_key_still_loads(tmp_path):
    # `<<: [*a, *b]` where both define `description` is legal YAML (earlier
    # entries win); neither the merge sequence nor the local key is a duplicate.
    body = """
    agents:
      a: &a {system_prompt: from a, description: desc a}
      b: &b {system_prompt: from b, description: desc b, tools: [Read]}
      c:
        <<: [*a, *b]
        tools: [Write]
    tasks:
      - {id: one, agent: c, prompt: p}
    """
    path, _ = write_named(tmp_path, body)
    s = spec_mod.load(path)
    c = s.agents["c"]
    assert c.system_prompt == "from a"
    assert c.description == "desc a"
    assert c.tools == ["Write"]


def test_spec_without_duplicates_loads_unchanged(tmp_path):
    # The same key name in sibling mappings is not a duplicate: two agents each
    # with `system_prompt`, two tasks each with `prompt`. A check that tracked
    # keys across the whole document instead of per mapping would reject this.
    body = """
    defaults:
      difficulty: easy
      max_turns: 7
    agents:
      writer: {system_prompt: write things, description: writes}
      checker: {system_prompt: check things, description: checks}
    tasks:
      - id: draft
        agent: writer
        prompt: draft the doc
      - id: check
        agent: checker
        prompt: check the doc
        difficulty: hard
        depends_on: [draft]
    """
    path, _ = write_named(tmp_path, body)
    s = spec_mod.load(path)
    assert sorted(s.agents) == ["checker", "writer"]
    assert s.agents["writer"].system_prompt == "write things"
    assert s.agents["checker"].system_prompt == "check things"
    assert s.by_id["draft"].prompt == "draft the doc"
    assert s.by_id["check"].prompt == "check the doc"
    assert s.difficulty_for(s.by_id["draft"]) == "easy"
    assert s.difficulty_for(s.by_id["check"]) == "hard"
    assert s.max_turns_for(s.by_id["draft"]) == 7
    assert s.waves() == [["draft"], ["check"]]
    assert s.base_dir == tmp_path


def test_shipped_example_spec_still_loads():
    # A real, hand-written spec with block scalars, comments and flow lists.
    example = Path(__file__).resolve().parent.parent / "examples" / "example.yaml"
    s = spec_mod.load(example)
    assert s.waves() == [["schema"], ["api", "docs"], ["tests"]]
    assert "db-architect" in s.agents


def test_python_object_tag_is_still_refused(tmp_path):
    # The loader must stay a SafeLoader: a spec file must never run code.
    # If the tag were honoured, os.mkdir would create this directory.
    target = tmp_path / "pwned"
    body = (
        "agents: !!python/object/apply:os.mkdir\n"
        f"  - {str(target)!r}\n"
        "tasks:\n"
        "  - {id: one, agent: a, prompt: p}\n"
    )
    path = tmp_path / SPEC_NAME
    path.write_text(body)
    with pytest.raises(Exception, match="python/object/apply"):
        spec_mod.load(path)
    assert not target.exists()
