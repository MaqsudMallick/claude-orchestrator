import textwrap

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
