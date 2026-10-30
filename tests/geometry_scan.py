"""The scanner of the geometry lint (DESIGN §46.12; sphere series S0, hardened after the independent review).

A sphere world's state is Cartesian, so `pos[:, 2]` (the Cartesian z, not the height), `vel[:, 2]`,
`pos[:, :2]`, `ops.min_image` / `ops.wrap` (the identity on a sphere), the host torus helpers and the box
lattice (`world.size`, `dims.fg`, ...) keep type-checking and return plausible numbers in every consumer that
has not been converted to the geometry accessors. ``geometry_hits`` counts those raw uses per pattern in one
module from its AST; ``tests/test_lint.py`` compares the counts per file with ``tests/geometry_allowlist.py``.

What is detected (no regular expression on names decides alone):

* ``z_index`` / ``xy_slice``: a subscript whose last index is the constant ``2``, ``2:`` (z) or ``:2`` /
  ``0:2`` (xy), on ANY array (the plumbing of the brain inputs is allowlisted like the rest); and, for a
  *position-like* base, the indices ``-1``, a bare ``[2]``, ``take(x, 2)``, ``split(x, 3)`` and
  ``x, y, z = p.T``. Position-like means: an identifier with a ``pos``/``vel``/``position``/``velocity``/...
  token, a dict access with such a string key (``a["pos"]``), or a name ASSIGNED from a position-like
  expression (taint, to a fixpoint over the module: ``p = s.pos`` then ``p[:, 2]``), through subscripts,
  attributes and calls (``pos[m][:, 2]``, ``np.asarray(pos)[..., 2]``).
* ``wrap``: calls or references of ``ops.wrap``, ``ops.wrap_xy_clamp_z`` and ``ops.min_image`` through any
  alias of the ``ops`` module (``from alife import ops as O``) or imported by name (with ``as``); the host
  torus helpers ``min_image`` (any base), ``min_image_np``, ``circular_mean``, ``_wrap`` (director); and the
  manual forms ``d - L * round(d / L)`` and ``x % size`` / ``x % L`` / ``x % length``.
* ``world.size`` / ``world.height`` / ``world.field_cell``: attribute reads on ``world``, on a name bound to a
  ``....world`` section (``w = cfg.world``, tuple targets included) or on a parameter annotated
  ``WorldConfig``.
* ``lattice``: ``.fg .fz .g_hash .cell_hash .g_hash_z .cell_hash_z .n_cells .field_shape``.

Not detectable: an arbitrary re-implementation under other names. The rotation-equivariance test of S2
(spec 2.9, item 3) remains the backstop for those.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "alife"

GEOMETRY_PATTERNS = (
    "z_index",  # a [..., 2] / [..., 2:] index (any array), and the position-like forms listed above
    "xy_slice",  # a [..., :2] slice
    "wrap",  # ops.wrap, wrap_xy_clamp_z, min_image and the host torus helpers, manual forms
    "world.size",
    "world.height",
    "world.field_cell",
    "lattice",  # .fg .fz .g_hash .cell_hash .g_hash_z .cell_hash_z .n_cells .field_shape
)
OPS_FUNCTIONS = {"wrap", "wrap_xy_clamp_z", "min_image"}  # attributes of the ops module (or imported from it)
WRAP_ANY_BASE = {"min_image", "wrap_xy_clamp_z", "min_image_np", "circular_mean", "_wrap"}  # unique names
WORLD_ATTRS = {"size", "height", "field_cell"}
LATTICE_ATTRS = {"fg", "fz", "g_hash", "cell_hash", "g_hash_z", "cell_hash_z", "n_cells", "field_shape"}
POSITION_TOKENS = {
    "pos",
    "poss",
    "position",
    "positions",
    "vel",
    "vels",
    "velocity",
    "velocities",
    "xyz",
    "coords",
    "centroid",
}
MOD_BY_SIZE = {"size", "L", "length", "side"}  # `x % size`: a torus wrap written by hand


def _tokens(name: str) -> set[str]:
    return {t.lower() for t in re.split(r"[_\d]+", name) if t}


def _ident(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    return node.attr if isinstance(node, ast.Attribute) else None


def _const(node: ast.AST | None) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
    ):
        return -node.operand.value
    return None


def _last_index_kind(sl: ast.AST, poslike: bool, direct: bool = True) -> str | None:
    """'z_index' / 'xy_slice' for a subscript whose last index selects the third component or xy.

    ``direct``: the base is position-like by its own name (not only by taint). A bare ``x[-1]`` on a name that
    is merely assigned from a position-like expression is the last item of a list (``records[-1]``,
    ``ckpts[-1]``, ``shape[-1]``), not a z; it counts only on a directly position-like base, or as a tuple
    index (``p[:, -1]``, ``p[..., -1]``), which is a component of an array.
    """
    elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
    last = elts[-1]
    tuple_form = isinstance(sl, ast.Tuple) and len(elts) >= 2
    c = _const(last)
    if c == 2 and (tuple_form or poslike):
        return "z_index"
    if c == -1 and poslike and (tuple_form or direct):
        return "z_index"
    if isinstance(last, ast.Slice) and (tuple_form or poslike):
        lo, up = _const(last.lower), _const(last.upper)
        if last.upper is not None and up == 2 and (last.lower is None or lo == 0):
            return "xy_slice"
        if last.lower is not None and lo in (2, -1) and last.upper is None:
            return "z_index"
    return None


class _Scanner:
    def __init__(self, tree: ast.AST):
        self.tree = tree
        self.tainted: set[str] = set()
        self.world_aliases: set[str] = {"world"}
        self.ops_aliases: set[str] = {"ops"}
        self.ops_names: set[str] = set()  # names imported from the ops module
        self._collect()

    # -- pre-passes ------------------------------------------------------------------------------------
    def _collect(self) -> None:
        for n in ast.walk(self.tree):
            if isinstance(n, ast.ImportFrom):
                mod = (n.module or "").split(".")[-1]
                for a in n.names:
                    if a.name == "ops":
                        self.ops_aliases.add(a.asname or a.name)
                    elif mod == "ops" and a.name in OPS_FUNCTIONS:
                        self.ops_names.add(a.asname or a.name)
            elif isinstance(n, ast.Import):
                for a in n.names:
                    if a.name.split(".")[-1] == "ops":
                        self.ops_aliases.add(a.asname or a.name.split(".")[-1])
            elif isinstance(n, ast.arg) and n.annotation is not None:
                ann = ast.unparse(n.annotation)
                if ann.split(".")[-1].strip("'\"") == "WorldConfig":
                    self.world_aliases.add(n.arg)
            elif isinstance(n, ast.Assign):
                for t, v in self._pairs(n.targets, n.value):
                    if isinstance(t, ast.Name) and isinstance(v, ast.Attribute) and v.attr == "world":
                        self.world_aliases.add(t.id)
        for _ in range(8):  # taint to a fixpoint
            before = len(self.tainted)
            for n in ast.walk(self.tree):
                targets: list[ast.AST] = []
                value: ast.AST | None = None
                if isinstance(n, ast.Assign):
                    targets, value = list(n.targets), n.value
                elif isinstance(n, ast.AnnAssign | ast.AugAssign | ast.NamedExpr):
                    targets, value = [n.target], n.value
                elif isinstance(n, ast.For | ast.comprehension):
                    targets, value = [n.target], n.iter
                if value is not None and self.poslike(value):
                    for t in targets:
                        self.tainted |= {x.id for x in ast.walk(t) if isinstance(x, ast.Name)}
            if len(self.tainted) == before:
                break

    @staticmethod
    def _pairs(targets: list[ast.AST], value: ast.AST) -> list[tuple[ast.AST, ast.AST]]:
        tgt = targets[0] if len(targets) == 1 else None
        if isinstance(tgt, ast.Tuple) and isinstance(value, ast.Tuple) and len(tgt.elts) == len(value.elts):
            return list(zip(tgt.elts, value.elts, strict=True))
        return [(t, value) for t in targets]

    def poslike(self, node: ast.AST) -> bool:
        """Does this expression denote (or contain, through subscripts, attributes and calls) a position-like
        value?"""
        if isinstance(node, ast.Name):
            return node.id in self.tainted or bool(_tokens(node.id) & POSITION_TOKENS)
        if isinstance(node, ast.Attribute):
            return bool(_tokens(node.attr) & POSITION_TOKENS) or self.poslike(node.value)
        if isinstance(node, ast.Subscript):
            key = _const(node.slice)
            if isinstance(key, str) and _tokens(key) & POSITION_TOKENS:
                return True
            return self.poslike(node.value)
        if isinstance(node, ast.Call):
            return any(self.poslike(a) for a in node.args) or self.poslike(node.func)
        if isinstance(node, ast.BinOp):
            return self.poslike(node.left) or self.poslike(node.right)
        if isinstance(node, ast.UnaryOp):
            return self.poslike(node.operand)
        if isinstance(node, ast.IfExp):
            return self.poslike(node.body) or self.poslike(node.orelse)
        if isinstance(node, ast.Tuple | ast.List):
            return any(self.poslike(e) for e in node.elts)
        return False

    def _direct(self, node: ast.AST) -> bool:
        """Is the base position-like by a token of its own name (a name, an attribute, a dict key), through
        subscripts and calls, without counting taint? ``.shape`` is a tuple of sizes, never a position."""
        if isinstance(node, ast.Name):
            return bool(_tokens(node.id) & POSITION_TOKENS)
        if isinstance(node, ast.Attribute):
            return node.attr != "shape" and (
                bool(_tokens(node.attr) & POSITION_TOKENS) or self._direct(node.value)
            )
        if isinstance(node, ast.Subscript):
            key = _const(node.slice)
            if isinstance(key, str) and _tokens(key) & POSITION_TOKENS:
                return True
            return self._direct(node.value)
        if isinstance(node, ast.Call):
            return any(self._direct(a) for a in node.args) or self._direct(node.func)
        return self.poslike(node) and not isinstance(node, ast.Name)

    # -- the scan --------------------------------------------------------------------------------------
    def hits(self) -> dict[str, int]:
        h = dict.fromkeys(GEOMETRY_PATTERNS, 0)
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Subscript):
                kind = _last_index_kind(n.slice, self.poslike(n.value), self._direct(n.value))
                if kind:
                    h[kind] += 1
            elif isinstance(n, ast.Call):
                self._call(n, h)
            elif isinstance(n, ast.Assign):
                # x, y, z = p.T
                if (
                    len(n.targets) == 1
                    and isinstance(n.targets[0], ast.Tuple)
                    and len(n.targets[0].elts) == 3
                    and isinstance(n.value, ast.Attribute)
                    and n.value.attr == "T"
                    and self.poslike(n.value.value)
                ):
                    h["z_index"] += 1
            elif isinstance(n, ast.Attribute):
                self._attribute(n, h)
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                if n.id in WRAP_ANY_BASE or n.id in self.ops_names:
                    h["wrap"] += 1
            elif isinstance(n, ast.BinOp):
                self._binop(n, h)
        return {k: v for k, v in h.items() if v}

    def _call(self, n: ast.Call, h: dict[str, int]) -> None:
        name = _ident(n.func)
        if name in ("take", "take_along_axis") and len(n.args) >= 2 and self.poslike(n.args[0]):
            if _const(n.args[1]) in (2, -1):
                h["z_index"] += 1
        elif name == "split" and n.args and self.poslike(n.args[0]):
            h["z_index"] += 1

    def _attribute(self, n: ast.Attribute, h: dict[str, int]) -> None:
        base = _ident(n.value)
        if n.attr in WRAP_ANY_BASE or (n.attr in OPS_FUNCTIONS and base in self.ops_aliases):
            h["wrap"] += 1
        elif n.attr in WORLD_ATTRS and base in self.world_aliases:
            h[f"world.{n.attr}"] += 1
        elif n.attr in LATTICE_ATTRS:
            h["lattice"] += 1

    def _binop(self, n: ast.BinOp, h: dict[str, int]) -> None:
        if isinstance(n.op, ast.Mod) and (_ident(n.right) in MOD_BY_SIZE):
            h["wrap"] += 1
        elif isinstance(n.op, ast.Mult):  # L * round(d / L), either order
            for a, b in ((n.left, n.right), (n.right, n.left)):
                if (
                    isinstance(b, ast.Call)
                    and _ident(b.func) in ("round", "rint")
                    and b.args
                    and isinstance(b.args[0], ast.BinOp)
                    and isinstance(b.args[0].op, ast.Div)
                    and ast.dump(b.args[0].right) == ast.dump(a)
                ):
                    h["wrap"] += 1
                    return


def geometry_hits(tree: ast.AST) -> dict[str, int]:
    """Raw uses of the box geometry in one module, by pattern (``GEOMETRY_PATTERNS``)."""
    return _Scanner(tree).hits()


# The modules that DEFINE the geometry (the config keys, the torus primitives, the two accessor modules, and
# the sphere's field tables and kernels, S1): their raw
# reads are the point of them, so they are not counted (a count there would break on every legitimate edit).
DEFINITION_FILES = frozenset(
    {
        "config.py", "ops.py", "geom.py", "sphere.py", "fieldtab.py", "fields_sph.py",
        "game/place.py",  # S5: the game layer's host geometry dispatch (centroid, height, distance, text)
        "storms_sph.py",  # S5: the storm detector and tracker of the sphere (its cells and its poles)
    }
)  # fmt: skip


def geometry_scan() -> dict[tuple[str, str], int]:
    """(file relative to src/alife, pattern) -> count, for every module except ``DEFINITION_FILES``."""
    out: dict[tuple[str, str], int] = {}
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel in DEFINITION_FILES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for pattern, n in geometry_hits(tree).items():
            out[(rel, pattern)] = n
    return out


if __name__ == "__main__":  # `python tests/geometry_scan.py`: the current counts, as allowlist lines to paste
    order = {p: i for i, p in enumerate(GEOMETRY_PATTERNS)}
    for (f, p), n in sorted(geometry_scan().items(), key=lambda kv: (kv[0][0], order[kv[0][1]])):
        print(f'    Allow("{f}", "{p}", {n}, "<reason>"),')
