"""project_dir: the agents' place of work, and what resolves under it."""

import textwrap
from pathlib import Path

import pytest
from orchestrator import spec as spec_mod

BODY = """
project_dir: {pd}
defaults:
  cwd: {cwd}
  allowed_write_dirs: [shared]
agents:
  a: {{system_prompt: x}}
tasks:
  - {{id: one, agent: a, prompt: p}}
  - {{id: two, agent: a, prompt: p, cwd: sub}}
  - {{id: abs, agent: a, prompt: p, cwd: /tmp}}
"""


def write(tmp_path, pd=".", cwd="."):
    p = tmp_path / "spec.yaml"
    p.write_text(textwrap.dedent(BODY.format(pd=pd, cwd=cwd)))
    return p


def test_defaults_to_the_spec_directory(tmp_path):
    s = spec_mod.load(write(tmp_path))
    assert s.project_root == tmp_path.resolve()


def test_relative_project_dir_resolves_against_the_spec_file(tmp_path):
    (tmp_path / "repo").mkdir()
    s = spec_mod.load(write(tmp_path, pd="./repo"))
    assert s.project_root == (tmp_path / "repo").resolve()


def test_absolute_project_dir_is_used_as_is(tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir()
    s = spec_mod.load(write(tmp_path, pd=str(target)))
    assert s.project_root == target.resolve()


def test_task_cwds_resolve_under_the_project_dir(tmp_path):
    (tmp_path / "repo").mkdir()
    s = spec_mod.load(write(tmp_path, pd="./repo"))
    root = (tmp_path / "repo").resolve()
    assert s.resolve_cwd(s.by_id["one"]) == root
    assert s.resolve_cwd(s.by_id["two"]) == root / "sub"
    # an absolute task cwd opts out of the project root deliberately
    assert s.resolve_cwd(s.by_id["abs"]) == Path("/tmp").resolve()


def test_allowed_write_dirs_resolve_under_the_project_dir(tmp_path):
    (tmp_path / "repo").mkdir()
    s = spec_mod.load(write(tmp_path, pd="./repo"))
    assert s.allowed_write_roots() == [(tmp_path / "repo" / "shared").resolve()]


def test_missing_project_dir_fails_at_load_not_at_run(tmp_path):
    """Every task would fail identically, so a typo here should cost nothing."""
    with pytest.raises(ValueError, match="project_dir does not exist"):
        spec_mod.load(write(tmp_path, pd="./not-there"))


def test_project_dir_pointing_at_a_file_is_rejected(tmp_path):
    (tmp_path / "afile").write_text("x")
    with pytest.raises(ValueError, match="not a directory"):
        spec_mod.load(write(tmp_path, pd="./afile"))


def test_override_beats_the_spec(tmp_path):
    (tmp_path / "repo").mkdir()
    other = tmp_path / "other"
    other.mkdir()
    s = spec_mod.load(write(tmp_path, pd="./repo"), project_dir=other)
    assert s.project_root == other.resolve()
    assert s.resolve_cwd(s.by_id["two"]) == (other / "sub").resolve()


def test_override_of_a_missing_dir_still_fails(tmp_path):
    (tmp_path / "repo").mkdir()
    with pytest.raises(ValueError, match="project_dir does not exist"):
        spec_mod.load(write(tmp_path, pd="./repo"), project_dir=tmp_path / "ghost")


def test_sibling_task_dirs_are_separate_write_sandboxes(tmp_path):
    """Documented behaviour: a task's sandbox is its own cwd, not the whole
    project, so sibling tasks cannot write over each other."""
    from orchestrator.permissions import check_denied

    (tmp_path / "repo" / "services" / "api").mkdir(parents=True)
    (tmp_path / "repo" / "services" / "web").mkdir(parents=True)
    s = spec_mod.load(write(tmp_path, pd="./repo"))
    api_root = [s.project_root / "services" / "api"]

    inside = {"file_path": str(s.project_root / "services" / "api" / "h.py")}
    sibling = {"file_path": str(s.project_root / "services" / "web" / "h.js")}
    assert check_denied("Write", inside, api_root, False) is None
    assert check_denied("Write", sibling, api_root, False)


def test_allowed_write_dirs_widens_the_sandbox_across_the_project(tmp_path):
    from orchestrator.permissions import check_denied

    (tmp_path / "repo" / "shared").mkdir(parents=True)
    s = spec_mod.load(write(tmp_path, pd="./repo"))
    roots = [s.project_root / "services", *s.allowed_write_roots()]
    shared = {"file_path": str(s.project_root / "shared" / "types.py")}
    assert check_denied("Write", shared, roots, False) is None
