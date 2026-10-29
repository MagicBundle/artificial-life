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

import geometry_allowlist

SRC = Path(__file__).resolve().parents[1] / "src" / "alife"

AT_MODULES = {"ops", "render"}  # rule 1: `.at[` only here (render: uint32 compositing)
SETITEM_ALLOW = {("ops", "set_rows")}  # rule 2: array subscript assignment
SIM_MODULES = {
    "spatial",
    "knn_metal",  # the fused Metal kernel (§4): no host reads or setitem, except the host-side self_check
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
    "groups",  # the generic pure-MLX grouping primitives (§29.3)
    "game.observe",  # the pure-MLX observer update and rules (§28): no host reads, like the simulation
    "game.rules",
    "game.social",  # the pure-MLX social state and its update (§29)
    "game.tracers",  # the tracer rig and the dumps' device arrays (§32): observer-only, pure MLX
    "weather",  # the W0 weather model (STORMS §3.2): pure MLX, wired into the step in W1
    "sphere",  # the device geometry of the sphere world (§46): pure MLX (its host twin, geom, is numpy)
}  # rule 5
DIRECTOR_MODULES = {f"director.{p.stem}" for p in (SRC / "director").glob("*.py")}  # host-side, scanned too
GAME_MODULES = {
    f"game.{p.stem}" for p in (SRC / "game").glob("*.py")
}  # the games vertical (§28), scanned too
SETITEM_MODULES = (
    SIM_MODULES
    | {"ops", "rng", "state", "stream", "streamrec", "storms", "play"}
    | DIRECTOR_MODULES
    | GAME_MODULES
)  # rule 2
CLOCK_MODULES = {
    "play"
}  # rule 7: real time only through the injected Clock (its RealClock is the one reader)
CLOCK_CLASS = "RealClock"
HOST_ALLOW = {  # rule 5: host-side checks that read arrays by design
    ("invariants", "full_check"),
    ("knn_metal", "self_check"),  # the once-per-process startup comparison of the kernel with the MLX search
    ("knn_metal", "_check_scene"),  # builds that comparison's scene in numpy
}
METAL_KERNEL_MODULES = {"knn_metal"}  # rule 6: the only module that builds a custom Metal kernel
KEYLESS_RANDOM = {"key", "split"}  # take the key (or seed) positionally by design
SAFE_ROOTS = {"cfg", "d", "math", "layout", "dims", "self"}  # float()/int() of config-derived Python values


def _modules() -> Iterator[tuple[str, ast.Module]]:
    """Every module of ``alife`` and of its packages ``director``, ``game`` and ``analysis`` (named
    ``<package>.<stem>``)."""
    for path in sorted(SRC.glob("*.py")):
        yield path.stem, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for package in ("director", "game", "analysis"):
        for path in sorted((SRC / package).glob("*.py")):
            yield f"{package}.{path.stem}", ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


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


def rule_metal_kernel(mod: str, tree: ast.AST) -> list[str]:
    """6. ``mx.fast.metal_kernel`` is called only in ``knn_metal``, the one reviewed file of kernel
    sources (each must be bitwise equal to the pure-MLX path it replaces, DESIGN §4 and F21)."""
    if mod in METAL_KERNEL_MODULES:
        return []
    out = []
    for fn, node in _functions(tree):
        hit = (
            (isinstance(node, ast.Attribute) and node.attr == "metal_kernel")
            or (isinstance(node, ast.Name) and node.id == "metal_kernel")
            or (isinstance(node, ast.Constant) and node.value == "metal_kernel")  # getattr(mx.fast, "...")
            or (isinstance(node, ast.ImportFrom) and any(a.name == "metal_kernel" for a in node.names))
        )
        if hit:
            out.append(f"{mod}.{fn or '<module>'}:{node.lineno}: metal_kernel outside knn_metal")
    return out


def rule_clock(mod: str, tree: ast.AST) -> list[str]:
    """7. The play loop reads and waits on real time only through the injected ``Clock``: ``time.sleep``,
    ``time.monotonic``, ``time.perf_counter`` and ``time.time`` appear only inside ``RealClock``, so the
    scheduler tests run on a fake clock with no real waiting."""
    if mod not in CLOCK_MODULES:
        return []
    real = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == CLOCK_CLASS]
    allowed = {id(sub) for c in real for sub in ast.walk(c)}
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and id(node) not in allowed:
            name = _dotted(node.func)
            if name in (
                "time.sleep",
                "time.monotonic",
                "time.perf_counter",
                "time.time",
                "sleep",
                "monotonic",
            ):
                out.append(f"{mod}:{node.lineno}: {name}( outside the Clock")
    return out


RULES = (rule_at, rule_setitem, rule_random, rule_async_eval, rule_host_reads, rule_metal_kernel, rule_clock)


@pytest.mark.parametrize("rule", RULES, ids=lambda r: r.__name__)
def test_source_tree(rule) -> None:
    mods = list(_modules())
    assert {"ops", "state", "step", "bonds", "director.rules", "director.actions"} <= {m for m, _ in mods}
    assert {"game.observe", "game.rules", "game.observer"} <= {m for m, _ in mods}
    # the world report (DESIGN §31): templates, lexicon and report are host-side game modules, scanned too
    assert {"game.report", "game.templates", "game.lexicon"} <= {m for m, _ in mods}
    # the play loop's host modules (DESIGN §33.2): the chronicle and the observer colour modes
    assert {"game.chronicle", "game.colors"} <= {m for m, _ in mods}
    # the generic modules of §29, a device half and a host half, and the game layer built on them
    assert {"groups", "tracking", "game.social", "game.society"} <= {m for m, _ in mods}
    assert {"spatial", "knn_metal"} <= {m for m, _ in mods}
    assert {"geom", "sphere"} <= {m for m, _ in mods}  # the sphere world's host and device geometry (§46)
    assert {"storms", "streamrec", "play"} <= {m for m, _ in mods}
    assert {"analysis.resim", "analysis.labels"} <= {
        m for m, _ in mods
    }  # rule 6 is repo-wide: analysis is scanned
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
    assert bad(rule_setitem, "director.rules", "def f(d):\n    d['a'] = 1\n")  # the director scans clean too
    assert bad(rule_setitem, "game.observe", "def f(d):\n    d['a'] = 1\n")  # and so do the game modules
    assert bad(rule_setitem, "game.observer", "def f(d):\n    d['a'] = 1\n")  # the host observer too
    assert bad(rule_setitem, "game.rules", "def f(d):\n    d['a'] = 1\n")
    assert bad(rule_setitem, "streamrec", "def f(d):\n    d['a'] = 1\n")  # the play-record encoders (§33.1)
    assert bad(rule_setitem, "storms", "def f(d):\n    d['a'] = 1\n")  # the storm tracker and layer (§33.3)
    assert bad(rule_setitem, "play", "def f(d):\n    d['a'] = 1\n")  # the play loop's host module (§33.2)
    assert bad(rule_setitem, "groups", "def f(d):\n    d['a'] = 1\n")  # the generic device module is scanned
    assert not bad(rule_setitem, "tracking", "def f(d):\n    d['a'] = 1\n")  # the host module is not
    assert bad(rule_setitem, "game.society", "def f(d):\n    d['a'] = 1\n")  # the host game module is scanned
    assert bad(rule_setitem, "game.social", "def f(x):\n    x[0] += 1\n")
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
    assert bad(rule_host_reads, "game.rules", "def f(x):\n    return int(Intent.NONE)\n")  # use .value
    assert bad(rule_host_reads, "game.observe", "def f(x):\n    return int(ActFlag.BUDDED)\n")  # use .value
    assert not bad(rule_host_reads, "game.observe", "def f(x):\n    return ActFlag.BUDDED.value\n")
    assert bad(rule_host_reads, "sphere", "def f(x):\n    return float(x.sum())\n")  # the device geometry
    assert bad(rule_setitem, "sphere", "def f(x):\n    x[0] = 1\n")
    assert not bad(
        rule_host_reads, "geom", "def f(x):\n    return float(x.sum())\n"
    )  # its host twin is numpy
    assert bad(rule_host_reads, "groups", "def f(x):\n    return int(x.sum())\n")
    assert bad(rule_host_reads, "groups", "def f(x):\n    return x.item()\n")
    assert bad(rule_host_reads, "game.social", "def f(sr):\n    return int(sr.slots)\n")  # sr is no safe root
    assert bad(rule_host_reads, "game.social", "def f(x):\n    return int(SocialKind.A)\n")  # use .value
    assert not bad(rule_host_reads, "game.social", "def f(sr):\n    return sr.slots + 1\n")
    assert not bad(rule_host_reads, "game.society", "def f(x):\n    return int(x.sum())\n")  # host numpy
    assert not bad(
        rule_host_reads, "groups", "def f(rate):\n    return min(int(rate * 2.0**64), 2**64 - 1)\n"
    )
    # tracking is host numpy by design and not a simulation module
    assert not bad(rule_host_reads, "tracking", "def f(x):\n    return int(x.sum())\n")
    assert bad(
        rule_host_reads, "weather", "def f(x):\n    return float(x.sum())\n"
    )  # the W0 weather model too
    assert not bad(rule_host_reads, "invariants", "def full_check(s):\n    return s.energy.item()\n")
    assert not bad(rule_host_reads, "knn_metal", "def self_check(s):\n    return np.array(s)\n")
    assert bad(
        rule_host_reads, "knn_metal", "def scan_select(s):\n    return np.array(s)\n"
    )  # only self_check
    assert bad(rule_setitem, "knn_metal", "def f(x):\n    x[0] = 1\n")
    for snippet in (
        "mx.fast.metal_kernel(name='k')",
        "mlx.core.fast.metal_kernel(name='k')",
        "getattr(mx.fast, 'metal_kernel')",
        "metal_kernel(name='k')",
    ):
        assert bad(rule_metal_kernel, "spatial", f"def f():\n    return {snippet}\n"), snippet
    assert bad(rule_metal_kernel, "ops", "from mlx.core.fast import metal_kernel\n")
    assert not bad(rule_metal_kernel, "knn_metal", "def f():\n    return mx.fast.metal_kernel(name='k')\n")
    assert not bad(rule_metal_kernel, "spatial", "def f():\n    return knn_metal.scan_select(1)\n")
    assert not bad(rule_host_reads, "metrics", "def f(s):\n    return s.energy.item()\n")
    # rule 7: the play loop's clock
    assert bad(rule_clock, "play", "def f():\n    time.sleep(1)\n")
    assert bad(rule_clock, "play", "class Scheduler:\n    def poll(self):\n        return time.monotonic()\n")
    assert not bad(
        rule_clock, "play", "class RealClock:\n    def monotonic(self):\n        return time.monotonic()\n"
    )
    assert not bad(rule_clock, "stream", "def f():\n    time.sleep(1)\n")


# --- the geometry lint (DESIGN §46.12, sphere series S0)
# ------------------------------------------------------
#
# A sphere world's state is Cartesian, so `pos[:, 2]` (the Cartesian z, not the height), `vel[:, 2]`,
# `pos[:, :2]`,
# `ops.wrap`/`ops.min_image` (the identity on a sphere) and the box lattice (`world.size`, `dims.fg`,
# ...) keep
# type-checking and return plausible numbers in every consumer that has not been converted to the geometry
# accessors (`alife.geom.geometry`, `alife.sphere.geometry`). This rule counts those raw uses per file
# and pattern
# and compares them with `tests/geometry_allowlist.py`, the audit of the consumers as data. It is a
# ratchet: a new
# raw use fails until it is allowlisted with a reason a reviewer can see, and an allowlisted count can
# only go down.

GEOMETRY_PATTERNS = (
    "z_index",  # pos / vel (or a name ending in pos / vel) indexed [..., 2] or [..., 2:]
    "xy_slice",  # ... indexed [..., :2]
    "wrap",  # ops.wrap, ops.wrap_xy_clamp_z, ops.min_image
    "world.size",
    "world.height",
    "world.field_cell",
    "lattice",  # .fg .fz .g_hash .cell_hash .g_hash_z .cell_hash_z .n_cells .field_shape
)
WRAP_FUNCTIONS = {"wrap", "wrap_xy_clamp_z", "min_image"}
WORLD_ATTRS = {"size", "height", "field_cell"}
LATTICE_ATTRS = {"fg", "fz", "g_hash", "cell_hash", "g_hash_z", "cell_hash_z", "n_cells", "field_shape"}


def _leaf(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    return node.attr if isinstance(node, ast.Attribute) else None


def _index_kind(sl: ast.AST) -> str | None:
    """'z_index' / 'xy_slice' for a subscript whose last index is 2, 2: or :2 (``[..., 2]``, ``[:, :2]``)."""
    if not isinstance(sl, ast.Tuple) or len(sl.elts) < 2:
        return None
    last = sl.elts[-1]
    if isinstance(last, ast.Constant) and last.value == 2:
        return "z_index"
    if isinstance(last, ast.Slice):
        lo, up = last.lower, last.upper
        if isinstance(up, ast.Constant) and up.value == 2 and (lo is None or getattr(lo, "value", None) == 0):
            return "xy_slice"
        if isinstance(lo, ast.Constant) and lo.value == 2:
            return "z_index"
    return None


def _world_aliases(tree: ast.AST) -> set[str]:
    """Names bound to a ``.world`` config section in the module (``w = cfg.world``)."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt, val = node.targets[0], node.value
        pairs = (
            list(zip(tgt.elts, val.elts, strict=True))
            if isinstance(tgt, ast.Tuple) and isinstance(val, ast.Tuple) and len(tgt.elts) == len(val.elts)
            else [(tgt, val)]
        )
        out |= {
            t.id
            for t, v in pairs
            if isinstance(t, ast.Name) and isinstance(v, ast.Attribute) and v.attr == "world"
        }
    return out


def geometry_hits(tree: ast.AST) -> dict[str, int]:
    """Raw uses of the box geometry in one module, by pattern (``GEOMETRY_PATTERNS``)."""
    aliases = _world_aliases(tree) | {"world"}
    bare = {  # `from alife.ops import min_image` style imports
        (a.asname or a.name)
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[-1] == "ops"
        for a in n.names
        if a.name in WRAP_FUNCTIONS
    }
    hits = dict.fromkeys(GEOMETRY_PATTERNS, 0)
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            name = _leaf(node.value)
            kind = _index_kind(node.slice)
            if kind and name and (name.endswith("pos") or name.endswith("vel")):
                hits[kind] += 1
        elif isinstance(node, ast.Attribute):
            base = _leaf(node.value)
            if node.attr in WRAP_FUNCTIONS and base == "ops":
                hits["wrap"] += 1
            elif node.attr in WORLD_ATTRS and base in aliases:
                hits[f"world.{node.attr}"] += 1
            elif node.attr in LATTICE_ATTRS:
                hits["lattice"] += 1
        elif isinstance(node, ast.Name) and node.id in bare and isinstance(node.ctx, ast.Load):
            hits["wrap"] += 1
    return {k: v for k, v in hits.items() if v}


def geometry_scan() -> dict[tuple[str, str], int]:
    """(file relative to src/alife, pattern) -> count, for every module."""
    out: dict[tuple[str, str], int] = {}
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        for pattern, n in geometry_hits(
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        ).items():
            out[(rel, pattern)] = n
    return out


def test_no_new_raw_box_geometry_outside_the_allowlist() -> None:
    """The ratchet: a raw `pos[:, 2]`, `ops.min_image`, `world.size`, `dims.fg`, ... beyond what
    ``tests/geometry_allowlist.py`` records fails. Convert the use to the geometry accessors, or (a
    box-only tool, a
    pair expression, a definition) add an allowlist line with its reason."""
    allowed = {(e.file, e.pattern): e.count for e in geometry_allowlist.ALLOWLIST}
    new = {key: (n, allowed.get(key, 0)) for key, n in geometry_scan().items() if n > allowed.get(key, 0)}
    assert not new, "new raw geometry uses (file, pattern): (found, allowed): " + repr(new)


def test_the_geometry_allowlist_is_exact_and_dated() -> None:
    """The allowlist is the audit of DESIGN §46.12 as data: no stale count (lower it on a conversion),
    no duplicate, only known reasons, and an entry 'converted in S<k>' is deleted by the PR S<k>."""
    scan = geometry_scan()
    seen: set[tuple[str, str]] = set()
    for e in geometry_allowlist.ALLOWLIST:
        key = (e.file, e.pattern)
        assert key not in seen, f"duplicate allowlist entry {key}"
        seen.add(key)
        assert e.pattern in GEOMETRY_PATTERNS, e
        assert e.reason in geometry_allowlist.REASONS, e
        assert (SRC / e.file).is_file(), f"{e.file} does not exist"
        assert scan.get(key, 0) == e.count, (
            f"{key}: the allowlist says {e.count}, the source has {scan.get(key, 0)}"
        )
        if e.reason == "converted":
            assert e.until is not None and e.until > geometry_allowlist.CURRENT_PR, (
                f"{key} is 'converted in S{e.until}': S{geometry_allowlist.CURRENT_PR} must have deleted it"
            )
        else:
            assert e.until is None, e
    assert {k for k in scan} <= seen  # every raw use is covered (the ratchet test names the offenders)


def test_geometry_lint_is_not_vacuous() -> None:
    def hits(src: str) -> dict[str, int]:
        return geometry_hits(ast.parse(src))

    assert hits("z = s.pos[:, 2]\n") == {"z_index": 1}
    assert hits("z = vel[..., 2]\n") == {"z_index": 1}
    assert hits("z = new_pos[:, 2:]\n") == {"z_index": 1}
    assert hits("xy = patch_pos[:, :2]\n") == {"xy_slice": 1}
    assert hits("xy = pos[..., :2]\n") == {"xy_slice": 1}
    assert hits("a = ops.min_image(d, L)\nb = ops.wrap(x, L)\nc = ops.wrap_xy_clamp_z(p, L, 0, 1)\n") == {
        "wrap": 3
    }
    assert hits("from alife.ops import min_image\na = min_image(d, L)\n") == {"wrap": 1}
    assert hits("L = cfg.world.size\nH = world.height\nf = cfg.world.field_cell\n") == {
        "world.size": 1,
        "world.height": 1,
        "world.field_cell": 1,
    }
    assert hits("w = cfg.world\nL = w.size\n") == {"world.size": 1}
    assert hits("w, p = cfg.world, cfg.physics\nL = w.height\nq = p.size\n") == {"world.height": 1}
    assert hits("n = d.fg * dims(cfg).fz + x.g_hash + y.cell_hash\n") == {"lattice": 4}
    # what is not a raw use
    assert not hits("z = pos[:, 0]\ny = pos[:, 1]\nx = other[:, 2]\nq = pos[2]\nr = pos[:2]\n")
    assert not hits("L = cfg.sensing.radius\nz = box.height(pos)\nd = box.displacement(a, b)\n")
    assert not hits("w = cfg.sensing\nL = w.size\n")  # an alias of another section


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
