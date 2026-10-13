"""§27.1 the genome layout per dims: ``layout.for_dims``, the frozen 3D tail and a dims guard over ``src``."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from alife import layout as L

SRC = Path(__file__).resolve().parents[1] / "src" / "alife"
DIMS_DEPENDENT = {"D_IN", "D_OUT", "GENOME_LAYOUT_VERSION"}  # via dims(cfg).lay or genome_layout_version
ONLY_3D = (
    "NBR_DZ",
    "IN_VEL_UP",
    "IN_RES_GRAD_UP",
    "IN_CHEM_GRAD_UP",
    "IN_NBR_DZ",
    "IN_BOND_DZ",
    "IN_HEIGHT",
    "IN_PITCH",
    "IN_GROUNDED",
    "OUT_PITCH",
    "OUT_BUOYANCY",
    "OUT_Q_BOND_DZ",
    "OUT_Q_ATTACK_DZ",
)


def _tiles(blocks: tuple[tuple[str, int, int], ...], total: int) -> None:
    covered = [0] * total
    for _, start, width in blocks:
        assert width >= 1
        for i in range(start, start + width):
            assert 0 <= i < total
            covered[i] += 1
    assert covered == [1] * total, "blocks must be disjoint and tile [0, D)"
    assert [s for _, s, _ in blocks] == sorted(s for _, s, _ in blocks)
    assert len({name for name, _, _ in blocks}) == len(blocks)


def test_for_dims_2_equals_module_constants() -> None:
    lay = L.for_dims(2)
    assert lay.nd == 2 and lay is L.for_dims(2)
    for f in dataclasses.fields(lay):
        value = getattr(lay, f.name)
        if f.name != "nd" and value is not None:
            assert value == getattr(L, f.name), f.name


def test_3d_only_indices_are_none_in_2d() -> None:
    lay2, lay3 = L.for_dims(2), L.for_dims(3)
    assert {f.name for f in dataclasses.fields(lay2) if getattr(lay2, f.name) is None} == set(ONLY_3D)
    assert all(isinstance(getattr(lay3, name), int) for name in ONLY_3D)
    assert not any(hasattr(L, name) for name in ONLY_3D)  # no module constant to read by mistake in 2D


def test_3d_sizes_and_documented_indices() -> None:
    lay = L.for_dims(3)
    assert (lay.nd, lay.D_IN, lay.D_OUT, lay.NF, lay.F, lay.NBR_DZ) == (3, 84, 39, 8, 7, 7)
    assert lay.F == lay.NF - 1 and lay.NBR_FEATURE_NAMES[lay.NBR_DZ] == "dz"
    assert (lay.IN_VEL_UP, lay.IN_RES_GRAD_UP, lay.IN_CHEM_GRAD_UP, lay.IN_NBR_DZ) == (69, 70, 71, 74)
    assert (lay.IN_BOND_DZ, lay.IN_HEIGHT, lay.IN_PITCH, lay.IN_GROUNDED) == (77, 81, 82, 83)
    assert (lay.OUT_PITCH, lay.OUT_BUOYANCY, lay.OUT_Q_BOND_DZ, lay.OUT_Q_ATTACK_DZ) == (35, 36, 37, 38)
    names = lay.INPUT_NAMES
    assert names[lay.IN_HEIGHT] == "height" and names[lay.IN_CHEM_GRAD_UP + 2] == "chem2_grad_up"
    assert names[lay.IN_NBR_DZ + 2] == "nbr2_dz" and names[lay.IN_BOND_DZ + 3] == "bond3_dz"
    assert lay.OUTPUT_NAMES[lay.OUT_BUOYANCY] == "buoyancy"
    # Parameters per agent: (84 + 32)·32 + 33·39 at the default H = 32.
    assert (lay.D_IN + 32) * 32 + 33 * lay.D_OUT == 4999


@pytest.mark.parametrize("nd", [2, 3])
def test_blocks_tile_inputs_and_outputs(nd: int) -> None:
    lay = L.for_dims(nd)
    _tiles(lay.INPUT_BLOCKS, lay.D_IN)
    _tiles(lay.OUTPUT_BLOCKS, lay.D_OUT)
    assert len(lay.INPUT_NAMES) == len(set(lay.INPUT_NAMES)) == lay.D_IN
    assert len(lay.OUTPUT_NAMES) == len(set(lay.OUTPUT_NAMES)) == lay.D_OUT
    assert len(lay.NBR_FEATURE_NAMES) == lay.NF
    # Every name inside a block starts with the block name's first word (the test_layout rule).
    for blocks, names in ((lay.INPUT_BLOCKS, lay.INPUT_NAMES), (lay.OUTPUT_BLOCKS, lay.OUTPUT_NAMES)):
        for name, start, width in blocks:
            for i in range(start, start + width):
                assert names[i].startswith(name.split("_")[0]), (name, names[i])


def test_3d_prefix_is_2d() -> None:
    """Append-only: every 2D index keeps its meaning in 3D, so the 3D tables start with the 2D ones."""
    lay2, lay3 = L.for_dims(2), L.for_dims(3)
    assert lay3.INPUT_NAMES[: lay2.D_IN] == lay2.INPUT_NAMES
    assert lay3.OUTPUT_NAMES[: lay2.D_OUT] == lay2.OUTPUT_NAMES
    assert lay3.INPUT_BLOCKS[: len(lay2.INPUT_BLOCKS)] == lay2.INPUT_BLOCKS
    assert lay3.OUTPUT_BLOCKS[: len(lay2.OUTPUT_BLOCKS)] == lay2.OUTPUT_BLOCKS
    assert lay3.NBR_FEATURE_NAMES[: lay2.NF] == lay2.NBR_FEATURE_NAMES
    # The 2D query blocks keep their place; the 7th components are in the tail.
    assert lay3.OUT_Q_BOND_DZ >= lay2.D_OUT and lay3.OUT_Q_ATTACK_DZ >= lay2.D_OUT


def test_for_dims_rejects_other_dims() -> None:
    for nd in (1, 4):
        with pytest.raises(ValueError, match="dims"):
            L.for_dims(nd)
        with pytest.raises(ValueError, match="dims"):
            L.genome_layout_version(nd)


def test_genome_layout_version_reads_global(monkeypatch: pytest.MonkeyPatch) -> None:
    assert (L.genome_layout_version(2), L.genome_layout_version(3)) == (1, 2)
    assert L.GENOME_LAYOUT_VERSION_3D == 2
    monkeypatch.setattr(L, "GENOME_LAYOUT_VERSION", 7)
    assert L.genome_layout_version(2) == 7 and L.genome_layout_version(3) == 2


# --- guard: dims-dependent constants are read through dims(cfg) -------------------------------------------


def _dims_constant_reads(tree: ast.AST) -> list[str]:
    """Lines reading ``layout.D_IN``, ``D_OUT`` or ``GENOME_LAYOUT_VERSION`` directly, under any alias."""
    aliases = {"alife.layout"}
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module in ("alife", None):  # `from alife import layout`, `from . import layout`
                aliases |= {a.asname or a.name for a in node.names if a.name == "layout"}
            elif node.module in ("alife.layout", "layout"):
                bad = [a.name for a in node.names if a.name in DIMS_DEPENDENT]
                out += [f"{node.lineno}: from layout import {name}" for name in bad]
        elif isinstance(node, ast.Import):
            aliases |= {a.asname for a in node.names if a.name == "alife.layout" and a.asname}
    reads = (n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in DIMS_DEPENDENT)
    out += [f"{n.lineno}: {ast.unparse(n)}" for n in reads if ast.unparse(n.value) in aliases]
    return out


def test_no_dims_dependent_module_constants_in_src() -> None:
    """Outside layout.py, the totals and the genome version come from ``dims(cfg).lay`` and
    ``layout.genome_layout_version(nd)``: a direct read would silently be the 2D value in a 3D world."""
    files = sorted(p for p in SRC.rglob("*.py") if p.name != "layout.py")
    assert len(files) > 30 and any(p.parent.name == "director" for p in files)
    found = [
        f"{p.relative_to(SRC)}:{v}"
        for p in files
        for v in _dims_constant_reads(ast.parse(p.read_text(encoding="utf-8")))
    ]
    assert found == []


def test_dims_guard_is_not_vacuous() -> None:
    for src in (
        "from alife import layout\nx = layout.D_IN\n",
        "from alife import layout as lo\nx = lo.D_OUT\n",
        "import alife.layout as lo\nx = lo.GENOME_LAYOUT_VERSION\n",
        "from alife.layout import D_IN\n",
        "from . import layout\nx = layout.D_OUT\n",
    ):
        assert _dims_constant_reads(ast.parse(src)), src
    for src in ("x = d.lay.D_IN\n", "from alife import layout\nx = layout.NS\n", "x = cfg.D_OUT\n"):
        assert not _dims_constant_reads(ast.parse(src)), src
