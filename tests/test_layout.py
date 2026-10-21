"""The package layout of `orchestrator/` is a deliberate architecture, enforced here.

`orchestrator/` is split into packages by responsibility, and those packages are
layered: a package may import only from the strictly lower layers listed in
`ALLOWED` below. Without something checking the direction, a split like this
drifts back into a flat pile where everything imports everything.

A failure here means one of two things:

  * the import is wrong - move the code, invert the dependency, or pass the
    value in from a higher layer; or
  * the architecture has been consciously changed - in which case update
    `LAYER` / `ALLOWED` / `ENTRY_MODULES` in this file, on purpose and in the
    same change, so the new design is written down.

It never means "add an exception so CI goes green". Type-only imports under
`if TYPE_CHECKING:` and imports inside functions count too: they are still a
dependency in the wrong direction, just a deferred one.

Everything is read with `ast`; nothing under `orchestrator/` is imported, so a
broken import shows up as a readable failure rather than an ImportError.
"""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent / "orchestrator"
TOP = "orchestrator"

# Layer numbers are used for the direction in failure messages; ALLOWED is the
# rule itself. They are not the same thing: `view` is L4 but may not import
# `gate` (L2), so "anything lower" would be too permissive.
LAYER: dict[str, int] = {
    "spec": 0,
    "stream": 1,
    "gate": 2,
    "sessions": 2,
    "run": 3,
    "view": 4,
}
ALLOWED: dict[str, frozenset[str]] = {
    "spec": frozenset(),
    "stream": frozenset(),
    "gate": frozenset({"stream"}),
    "sessions": frozenset({"spec"}),
    "run": frozenset({"spec", "stream", "gate", "sessions"}),
    "view": frozenset({"spec", "stream", "sessions", "run"}),
}
PACKAGES = tuple(LAYER)

# The entry layer (L5): the only modules allowed directly in `orchestrator/`,
# and allowed to import anything.
ENTRY_MODULES = frozenset({"__init__", "__main__", "cli", "commands"})
ENTRY_LAYER = 5


# --------------------------------------------------------------- scanning --


@dataclass(frozen=True)
class Module:
    name: str  # dotted, e.g. "orchestrator.gate.policy"
    path: Path
    rel: str  # e.g. "orchestrator/gate/policy.py", for messages
    base: Path  # the directory containing `orchestrator/`
    is_package_init: bool

    @property
    def unit(self) -> str:
        """The architectural unit: a package name, or an entry module's stem."""
        return unit_of(self.name)


@dataclass(frozen=True)
class Import:
    importer: Module
    target: str  # dotted, fully resolved, always under `orchestrator`
    lineno: int
    type_only: bool

    def describe(self) -> str:
        kind = " (inside `if TYPE_CHECKING:`)" if self.type_only else ""
        where = f"{self.importer.rel}:{self.lineno}"
        return f"{where}: `{self.importer.name}` imports `{self.target}`{kind}"


def unit_of(dotted: str) -> str:
    """Map `orchestrator.run.graph` -> `run`, `orchestrator.cli` -> `cli`, and
    bare `orchestrator` -> `__init__`: the top-level package is an entry module."""
    parts = dotted.split(".")
    assert parts[0] == TOP, dotted
    return parts[1] if len(parts) > 1 else "__init__"


def layer_of(unit: str) -> int:
    return LAYER.get(unit, ENTRY_LAYER)


def iter_modules(root: Path) -> list[Module]:
    mods = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root.parent).with_suffix("")
        parts = list(rel.parts)
        is_init = parts[-1] == "__init__"
        if is_init:
            parts.pop()
        rel_path = path.relative_to(root.parent).as_posix()
        mods.append(Module(".".join(parts), path, rel_path, root.parent, is_init))
    return mods


def _is_type_checking_guard(node: ast.If) -> bool:
    test = node.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _resolve_relative(mod: Module, level: int, lineno: int) -> str:
    # A package's own __init__ *is* the package; a plain module lives in its parent.
    package = mod.name.split(".") if mod.is_package_init else mod.name.split(".")[:-1]
    if level - 1 >= len(package):
        raise AssertionError(
            f"{mod.path}:{lineno}: relative import with {level} dots escapes the "
            f"`{TOP}` package from `{mod.name}`"
        )
    return ".".join(package[: len(package) - (level - 1)])


def collect_imports(mod: Module) -> list[Import]:
    """Every import in `mod` that resolves to something under `orchestrator`,
    wherever it appears: module level, inside functions, or type-only."""
    tree = ast.parse(mod.path.read_text(encoding="utf-8"), filename=str(mod.path))
    out: list[Import] = []

    def add(target: str, node: ast.stmt, type_only: bool) -> None:
        if target == TOP or target.startswith(TOP + "."):
            out.append(Import(mod, target, node.lineno, type_only))

    def visit(node: ast.AST, type_only: bool) -> None:
        if isinstance(node, ast.If) and _is_type_checking_guard(node):
            for child in node.body:
                visit(child, True)
            for child in node.orelse:
                visit(child, type_only)
            return
        if isinstance(node, ast.Import):
            for alias in node.names:
                add(alias.name, node, type_only)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = _resolve_relative(mod, node.level, node.lineno)
                full = f"{base}.{node.module}" if node.module else base
            else:
                full = node.module or ""
            # `from pkg import x` imports submodule `pkg.x` when one exists on
            # disk, and otherwise just reads attribute `x` of `pkg`.
            pkg_dir = mod.base.joinpath(*full.split("."))
            for alias in node.names:
                sub = pkg_dir / alias.name
                is_module = sub.is_dir() or sub.with_suffix(".py").is_file()
                add(f"{full}.{alias.name}" if is_module else full, node, type_only)
        for child in ast.iter_child_nodes(node):
            visit(child, type_only)

    visit(tree, False)
    return out


def violations(root: Path) -> list[str]:
    problems = []
    for mod in iter_modules(root):
        src = mod.unit
        if src not in LAYER:
            continue  # entry modules may import anything
        for imp in collect_imports(mod):
            dst = unit_of(imp.target)
            if dst == src or dst in ALLOWED[src]:
                continue
            s, d = layer_of(src), layer_of(dst)
            if dst not in LAYER and dst not in ENTRY_MODULES:
                why = f"unclassified: {src} (L{s}) -> {dst}, which has no place in the layering"
            elif d > s:
                why = f"upward: {src} (L{s}) -> {dst} (L{d})"
            elif d == s:
                why = f"sideways: {src} (L{s}) -> {dst} (L{d}), same layer"
            else:
                allowed = ", ".join(sorted(ALLOWED[src])) or "nothing internal"
                why = (
                    f"not permitted: {src} (L{s}) -> {dst} (L{d}); {src} may import only: {allowed}"
                )
            problems.append(f"{imp.describe()}\n      {why}")
    return problems


def package_graph(root: Path) -> dict[str, set[str]]:
    """Unit -> units it imports, ignoring imports within the same unit."""
    graph: dict[str, set[str]] = {}
    for mod in iter_modules(root):
        deps = graph.setdefault(mod.unit, set())
        for imp in collect_imports(mod):
            dst = unit_of(imp.target)
            if dst != mod.unit:
                deps.add(dst)
    return graph


def find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """Strongly connected components of size > 1, each as a sorted list (Tarjan)."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    sccs: list[list[str]] = []

    def strong(v: str) -> None:
        index[v] = low[v] = len(index)
        stack.append(v)
        on_stack.add(v)
        for w in sorted(graph.get(v, ())):
            if w not in index:
                strong(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1:
                sccs.append(sorted(comp))

    for v in sorted(graph):
        if v not in index:
            strong(v)
    return sccs


def public_surface(init: Path) -> list[str]:
    """What a package's `__init__.py` offers: `__all__` if declared, otherwise
    the public names it binds by import, def, class or assignment."""
    tree = ast.parse(init.read_text(encoding="utf-8"), filename=str(init))
    bound: set[str] = set()
    declared: list[str] | None = None
    for node in tree.body:
        if isinstance(node, ast.ImportFrom | ast.Import):
            for alias in node.names:
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            bound.add(node.name)
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    bound.add(t.id)
                    if t.id == "__all__" and node.value is not None:
                        declared = [
                            e.value
                            for e in getattr(node.value, "elts", [])
                            if isinstance(e, ast.Constant) and isinstance(e.value, str)
                        ]
    if declared is not None:
        missing = [n for n in declared if n not in bound]
        assert not missing, (
            f"{init}: __all__ lists names the module never binds: {missing}. "
            f"`from {init.parent.name} import *` would raise."
        )
        return declared
    return sorted(n for n in bound if not n.startswith("_"))


# ------------------------------------------------------- the real layout --


def test_layer_table_is_itself_layered():
    """Guards the rule, not the code: every permitted edge must point strictly
    down, or the table would allow the very cycles it exists to prevent."""
    assert set(ALLOWED) == set(LAYER)
    bad = [
        f"{src} (L{LAYER[src]}) -> {dst} (L{LAYER[dst]})"
        for src, deps in ALLOWED.items()
        for dst in deps
        if LAYER[dst] >= LAYER[src]
    ]
    assert not bad, "ALLOWED permits non-downward edges: " + "; ".join(bad)


@pytest.mark.parametrize("package", PACKAGES)
def test_package_exists_with_init(package):
    pkg = ROOT / package
    assert pkg.is_dir(), f"expected package directory orchestrator/{package}/ is missing"
    assert (pkg / "__init__.py").is_file(), (
        f"orchestrator/{package}/ has no __init__.py, so it is a namespace package "
        f"with no public surface"
    )


def test_only_entry_modules_at_top_level():
    stray = sorted(p.name for p in ROOT.glob("*.py") if p.stem not in ENTRY_MODULES)
    assert not stray, (
        f"modules sitting directly in orchestrator/: {stray}. Only "
        f"{sorted(m + '.py' for m in ENTRY_MODULES)} belong there; put the rest "
        f"in one of the packages {list(PACKAGES)}."
    )
    missing = sorted(m for m in ENTRY_MODULES if not (ROOT / f"{m}.py").is_file())
    assert not missing, f"entry modules missing from orchestrator/: {missing}"


def test_no_unclassified_packages():
    """A new directory of code outside the layer table would escape the
    layering check entirely, so it has to be placed in the table first."""
    dirs = {p.parent.relative_to(ROOT).parts[0] for p in ROOT.rglob("*.py") if p.parent != ROOT}
    unknown = sorted(dirs - set(PACKAGES))
    assert not unknown, (
        f"orchestrator/ contains packages with no place in the layering: {unknown}. "
        f"Add them to LAYER and ALLOWED in {Path(__file__).name} deliberately."
    )


@pytest.mark.parametrize("package", PACKAGES)
def test_package_exports_public_surface(package):
    init = ROOT / package / "__init__.py"
    if not init.is_file():
        pytest.fail(f"orchestrator/{package}/__init__.py is missing")
    surface = public_surface(init)
    assert surface, (
        f"orchestrator/{package}/__init__.py exports nothing. Each package's __init__ "
        f"is its public surface: re-export what callers use, so they import "
        f"`from orchestrator.{package} import X` rather than reaching into its modules."
    )


def test_scan_is_not_vacuous():
    """If resolution silently broke, every check below would pass on an empty
    list. `run` is defined by depending on lower layers, so it must show some."""
    graph = package_graph(ROOT)
    for package in PACKAGES:
        assert any(m.unit == package for m in iter_modules(ROOT)), f"no modules in {package}"
    assert graph.get("run"), "found no internal imports from `run`; the scanner is broken"


def test_imports_respect_layering():
    problems = violations(ROOT)
    assert not problems, (
        f"{len(problems)} import(s) break the package layering:\n\n  "
        + "\n  ".join(problems)
        + "\n\nA package may import only from the layers listed in ALLOWED "
        f"({Path(__file__).name}). Fix the import, or - if the architecture is "
        "meant to change - change ALLOWED on purpose."
    )


def test_no_import_cycles_between_packages():
    graph = package_graph(ROOT)
    cycles = find_cycles(graph)

    def explain(comp: list[str]) -> str:
        edges = [f"{a} -> {b}" for a in comp for b in sorted(graph[a]) if b in comp]
        return f"{{{', '.join(comp)}}}: " + ", ".join(edges)

    assert not cycles, "import cycles between packages:\n  " + "\n  ".join(map(explain, cycles))


# -------------------------------------------- the scanner, on fake trees --
# These pin down the resolution rules so a green layering test means something.


def make_tree(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / TOP
    for rel, src in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(src))
    return root


def targets(root: Path, rel: str) -> set[str]:
    mod = next(m for m in iter_modules(root) if m.path == root / rel)
    return {i.target for i in collect_imports(mod)}


def test_resolves_relative_and_absolute_forms(tmp_path):
    root = make_tree(
        tmp_path,
        {
            "__init__.py": "",
            "stream/__init__.py": "",
            "stream/hub.py": "",
            "run/__init__.py": "",
            "gate/__init__.py": "from .policy import P\nfrom . import callback, VERSION\n",
            "gate/callback.py": "",
            "gate/questions.py": "",
            "gate/policy.py": """\
                import os
                from os import path
                from .. import stream
                from ..stream import StreamHub
                from ..stream.hub import X
                from . import questions
                from orchestrator.spec import Spec
                import orchestrator.sessions.store
                from orchestrator import run, __version__
            """,
        },
    )
    assert targets(root, "gate/policy.py") == {
        "orchestrator.stream",  # `from .. import stream` and `from ..stream import`
        "orchestrator.stream.hub",
        "orchestrator.gate.questions",
        "orchestrator.spec",
        "orchestrator.sessions.store",
        "orchestrator.run",  # a subpackage on disk, so a real submodule import
        "orchestrator",  # `__version__` is only an attribute of the top package
    }
    # In an __init__, `.` is the package itself, not its parent; `VERSION` is
    # not a module on disk, so it resolves to the package.
    assert targets(root, "gate/__init__.py") == {
        "orchestrator.gate.policy",
        "orchestrator.gate.callback",
        "orchestrator.gate",
    }


def test_finds_deferred_and_type_only_imports(tmp_path):
    root = make_tree(
        tmp_path,
        {
            "sessions/store.py": """\
                from typing import TYPE_CHECKING
                if TYPE_CHECKING:
                    from ..run import TaskResult
                def later():
                    from ..view import live
            """,
        },
    )
    mod = iter_modules(root)[0]
    found = {(i.target, i.type_only) for i in collect_imports(mod)}
    assert found == {("orchestrator.run", True), ("orchestrator.view", False)}


def test_relative_import_escaping_package_is_reported(tmp_path):
    root = make_tree(tmp_path, {"spec/schema.py": "from ... import elsewhere\n"})
    with pytest.raises(AssertionError, match="escapes"):
        collect_imports(iter_modules(root)[0])


def test_violation_message_names_both_ends_and_direction(tmp_path):
    root = make_tree(
        tmp_path,
        {
            "__init__.py": "",
            "cli.py": "from .view import live\nfrom .gate import x\n",  # entry: anything
            "run/graph.py": "from ..sessions import store\n",  # allowed, downward
            "run/runner.py": "from . import graph\n",  # same package
            "sessions/store.py": "\n\nfrom ..run.result import TaskResult\n",  # upward
            "gate/policy.py": "from ..sessions import S\n",  # sideways, same layer
            "view/live.py": "from ..gate import G\n",  # downward but not permitted
            "spec/schema.py": "from orchestrator import __version__\n",  # up to entry
        },
    )
    problems = violations(root)
    joined = "\n".join(problems)
    assert len(problems) == 4, joined
    assert (
        "orchestrator/sessions/store.py:3: `orchestrator.sessions.store` imports "
        "`orchestrator.run.result`\n      upward: sessions (L2) -> run (L3)"
    ) in joined
    assert "sideways: gate (L2) -> sessions (L2)" in joined
    assert (
        "not permitted: view (L4) -> gate (L2); view may import only: run, sessions, spec, "
        in joined
    )
    assert "upward: spec (L0) -> __init__ (L5)" in joined


def test_cycle_detection(tmp_path):
    root = make_tree(
        tmp_path,
        {
            "run/graph.py": "from ..sessions import store\n",
            "sessions/store.py": "from ..run import TaskResult\n",
            "view/live.py": "from ..run import graph\n",
        },
    )
    assert find_cycles(package_graph(root)) == [["run", "sessions"]]
    assert find_cycles({"a": {"b"}, "b": {"c"}, "c": set()}) == []


def test_public_surface_rules(tmp_path):
    empty = tmp_path / "a.py"
    empty.write_text('"""Docstring only."""\nimport os as _os\n')
    assert public_surface(empty) == []

    implicit = tmp_path / "b.py"
    implicit.write_text("from .hub import StreamHub, TaskStream\n")
    assert public_surface(implicit) == ["StreamHub", "TaskStream"]

    lying = tmp_path / "c.py"
    lying.write_text('from .hub import StreamHub\n__all__ = ["StreamHub", "Gone"]\n')
    with pytest.raises(AssertionError, match="Gone"):
        public_surface(lying)
