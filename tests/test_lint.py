"""§22.6 static checks (AST-based) over ``src/alife``.

Each rule is a function returning violations; ``test_rules_are_not_vacuous`` feeds known-bad snippets through
them. Allow-lists name ``(module, function)`` pairs; a package that needs an exception adds it here, with
a reason.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "alife"

AT_MODULES = {"ops", "render"}  # rule 1: `.at[` only here (render: uint32 compositing)
SETITEM_ALLOW = {("ops", "set_rows")}  # rule 2: array subscript assignment
SIM_MODULES = {
    "spatial",
    "fields",
    "sensing",
    "brain",
    "actions",
    "bonds",
    "physics",
    "energy",
    "lifecycle",
    "step",
    "invariants",
}  # rule 5
SETITEM_MODULES = SIM_MODULES | {"ops", "rng", "state"}  # rule 2 scope: modules that handle arrays in-graph
HOST_ALLOW = {("invariants", "full_check")}  # rule 5: the host-side exhaustive check reads numpy by design
KEYLESS_RANDOM = {"key", "split"}  # take the key (or seed) positionally by design
SAFE_ROOTS = {"cfg", "d", "math", "layout", "dims", "self"}  # float()/int() of config-derived Python values


def _modules() -> Iterator[tuple[str, ast.Module]]:
    for path in sorted(SRC.glob("*.py")):
        yield path.stem, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _functions(tree: ast.AST) -> Iterator[tuple[str, ast.AST]]:
    """(enclosing top-level function or method name, node) for every node; module/class level is ''."""

    def walk(node: ast.AST, owner: str) -> Iterator[tuple[str, ast.AST]]:
        for child in ast.iter_child_nodes(node):
            name = (
                child.name
                if not owner and isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
                else owner
            )
            yield name, child
            yield from walk(child, name)

    yield from walk(tree, "")


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _root(node: ast.AST) -> str | None:
    while isinstance(node, ast.Attribute | ast.Subscript | ast.Call):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


# --- rules ----------------------------------------------------------------------------------------------


def rule_at(mod: str, tree: ast.AST) -> list[str]:
    """1. `.at[` only in ops/render; in ops every scattering helper checks integer dtypes."""
    out = []
    users: dict[str, int] = {}
    for fn, node in _functions(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "at"
        ):
            if mod not in AT_MODULES:
                out.append(f"{mod}.{fn or '<module>'}:{node.lineno}: `.at[` outside ops/render")
            users[fn] = node.lineno
    if mod == "ops":
        for fn, lineno in users.items():
            body = next(n for f, n in _functions(tree) if isinstance(n, ast.FunctionDef) and n.name == fn)
            checks = [
                c
                for c in ast.walk(body)
                if isinstance(c, ast.Call)
                and isinstance(c.func, ast.Name)
                and c.func.id.startswith("_require_")
            ]
            if not checks:
                out.append(f"ops.{fn}:{lineno}: scatter without an integer dtype check")
    return out


def rule_setitem(mod: str, tree: ast.AST) -> list[str]:
    """2. Subscript assignment to arrays only in ops.set_rows."""
    if mod not in SETITEM_MODULES:
        return []
    out = []
    for fn, node in _functions(tree):
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AugAssign | ast.AnnAssign):
            targets = [node.target]
        for t in targets:
            for sub in ast.walk(t):
                if isinstance(sub, ast.Subscript) and (mod, fn) not in SETITEM_ALLOW:
                    out.append(f"{mod}.{fn or '<module>'}:{node.lineno}: subscript assignment")
    return out


def rule_random(mod: str, tree: ast.AST) -> list[str]:
    """3. Every mx.random.<fn>( passes key=; mx.random.seed is never called."""
    out = []
    for fn, node in _functions(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        if not (name.startswith("mx.random.") or name.startswith("mlx.core.random.")):
            continue
        leaf = name.rsplit(".", 1)[1]
        if leaf == "seed":
            out.append(f"{mod}.{fn}:{node.lineno}: mx.random.seed")
        elif leaf not in KEYLESS_RANDOM and not any(k.arg == "key" for k in node.keywords):
            out.append(f"{mod}.{fn}:{node.lineno}: {name}( without key=")
    return out


def rule_async_eval(mod: str, tree: ast.AST) -> list[str]:
    """4. mx.async_eval is never used."""
    return [
        f"{mod}:{node.lineno}: async_eval"
        for node in ast.walk(tree)
        if (isinstance(node, ast.Attribute) and node.attr == "async_eval")
        or (isinstance(node, ast.Name) and node.id == "async_eval")
    ]


def rule_host_reads(mod: str, tree: ast.AST) -> list[str]:
    """5. No .item(), .tolist(), np.asarray/np.array, mx.eval or float()/int() of arrays in sim modules."""
    if mod not in SIM_MODULES:
        return []
    out = []
    for fn, node in _functions(tree):
        if (mod, fn) in HOST_ALLOW or not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        where = f"{mod}.{fn or '<module>'}:{node.lineno}"
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("item", "tolist"):
            out.append(f"{where}: .{node.func.attr}()")
        elif name in ("np.asarray", "np.array", "numpy.asarray", "numpy.array", "mx.eval"):
            out.append(f"{where}: {name}(")
        elif isinstance(node.func, ast.Name) and node.func.id in ("float", "int") and node.args:
            for sub in ast.walk(node.args[0]):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                    continue  # builtins such as round(); their arguments are still walked
                if isinstance(sub, ast.Call | ast.Subscript | ast.Attribute) and _root(sub) not in SAFE_ROOTS:
                    out.append(f"{where}: {node.func.id}() of a possible array")
                    break
    return out


RULES = (rule_at, rule_setitem, rule_random, rule_async_eval, rule_host_reads)


@pytest.mark.parametrize("rule", RULES, ids=lambda r: r.__name__)
def test_source_tree(rule) -> None:
    mods = list(_modules())
    assert {"ops", "state", "step", "bonds"} <= {m for m, _ in mods}
    violations = [v for mod, tree in mods for v in rule(mod, tree)]
    assert violations == []


def test_rules_are_not_vacuous() -> None:
    def bad(rule, mod: str, src: str) -> list[str]:
        return rule(mod, ast.parse(src))

    assert bad(rule_at, "energy", "def f(x, i):\n    return x.at[i].add(1)\n")
    assert bad(rule_at, "ops", "def f(x, i):\n    return x.at[i].add(1)\n")
    assert not bad(rule_at, "ops", "def f(x, i):\n    _require_int(i, 'i')\n    return x.at[i].add(1)\n")
    assert bad(rule_setitem, "bonds", "def f(x):\n    x[0] = 1\n")
    assert bad(rule_setitem, "ops", "def place(x):\n    x[0] += 1\n")
    assert not bad(rule_setitem, "ops", "def set_rows(x):\n    x[0] = 1\n")
    assert not bad(rule_setitem, "config", "def f(d):\n    d['a'] = 1\n")
    assert bad(rule_random, "brain", "def f():\n    return mx.random.normal((3,))\n")
    assert bad(rule_random, "brain", "def f():\n    mx.random.seed(0)\n")
    assert not bad(
        rule_random, "brain", "def f(k):\n    return mx.random.normal((3,), key=k), mx.random.split(k)\n"
    )
    assert bad(rule_async_eval, "runner", "def f(s):\n    mx.async_eval(s)\n")
    for snippet in ("x.item()", "np.asarray(x)", "float(x.sum())", "int(s.step)", "mx.eval(x)", "x.tolist()"):
        assert bad(rule_host_reads, "physics", f"def f(x, s):\n    return {snippet}\n"), snippet
    for snippet in ("float(cfg.physics.drag)", "int(d.n)", "int(round(2.5))", "float(n)"):
        assert not bad(rule_host_reads, "physics", f"def f(cfg, d, n):\n    return {snippet}\n"), snippet
    assert not bad(rule_host_reads, "invariants", "def full_check(s):\n    return s.energy.item()\n")
    assert not bad(rule_host_reads, "metrics", "def f(s):\n    return s.energy.item()\n")


# --- project and CI configuration --------------------------------------------------------------------------

ROOT = SRC.parents[1]


def test_python_range_matches_mlx_wheels() -> None:
    """Every Python minor that ``requires-python`` admits has an mlx wheel in uv.lock (mlx has no sdist),
    and ``.python-version`` pins one of them, so ``uv sync`` cannot pick an uninstallable interpreter."""
    import re
    import tomllib

    spec = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["requires-python"]
    lo = re.search(r">=\s*3\.(\d+)", spec)
    hi = re.search(r"<\s*3\.(\d+)", spec)
    assert lo and hi, f"requires-python {spec!r} needs a lower and an upper bound"
    minors = range(int(lo[1]), int(hi[1]))
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    tags = {
        int(m[1])
        for pkg in lock["package"]
        if pkg["name"] == "mlx"
        for w in pkg.get("wheels", [])
        if (m := re.search(r"-cp3(\d+)-", w["url"]))
    }
    assert set(minors) <= tags, f"mlx wheels cover 3.{sorted(tags)}, requires-python admits 3.{list(minors)}"
    pinned = (ROOT / ".python-version").read_text().strip()
    assert re.fullmatch(r"3\.(\d+)", pinned) and int(pinned.split(".")[1]) in minors, pinned


def test_ci_installs_extras_and_caches_mlx_jit() -> None:
    """The fast-suite job installs the viewer/video extras (else the pygame loop test always skips) and both
    jobs keep MLX's CPU JIT cache (``$TMPDIR/mlx``) between runs (§22.0 budget)."""
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    test_job, nightly = ci.split("\n  nightly:")
    assert "--extra viewer" in test_job and "--extra video" in test_job
    for job in (test_job, nightly):
        assert "TMPDIR=$RUNNER_TEMP" in job and "path: ${{ runner.temp }}/mlx" in job
