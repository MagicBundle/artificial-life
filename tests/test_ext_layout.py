"""The layout composition of the feature extensions (DESIGN §43.2): offsets cumulative over the enabled
features in TAIL_ORDER, the tiling of the input and output vectors over every subset of the five features in
2D and 3D, the version integer and the ``layout_features`` string, ``Layout.__getattr__``.

Run against the stub registry (``tests/ext_stub.py``) with the real block widths of the plan.
"""

from __future__ import annotations

import itertools

import pytest

import ext_stub as stub
from alife import ext, layout
from alife.config import dims, load_config

SUBSETS = [c for r in range(6) for c in itertools.combinations(ext.EXTENSIONS, r)]
TOTALS = {  # (nd, subset) -> (D_IN, D_OUT): the table of section 1.2 of the plan
    (2, ()): (69, 35), (3, ()): (84, 39),
    (2, ext.EXTENSIONS): (97, 44), (3, ext.EXTENSIONS): (113, 48),
    (2, ("structures",)): (70, 36), (3, ("structures",)): (85, 40),
    (2, ("bonds",)): (82, 37), (3, ("bonds",)): (97, 41),
    (2, ("comm",)): (75, 39), (3, ("comm",)): (91, 43),
    (2, ("niche",)): (70, 36), (3, ("niche",)): (85, 40),
    (2, ("arms",)): (76, 36), (3, ("arms",)): (91, 40),
}  # fmt: skip


@pytest.fixture(autouse=True)
def _registry(monkeypatch: pytest.MonkeyPatch) -> None:
    stub.install(monkeypatch)


def _cfg(nd: int, names):
    preset = "tiny" if nd == 2 else "tiny3d"
    return stub.stub_cfg(load_config(preset=preset, overrides=["brain.weight_dtype=float32"]), *names)


@pytest.mark.parametrize("nd", [2, 3])
@pytest.mark.parametrize("names", SUBSETS, ids=lambda s: "+".join(s) or "none")
def test_blocks_tile_the_vectors_for_every_subset(nd: int, names: tuple[str, ...]) -> None:
    lay = dims(_cfg(nd, names)).lay
    for blocks, total in ((lay.INPUT_BLOCKS, lay.D_IN), (lay.OUTPUT_BLOCKS, lay.D_OUT)):
        pos = 0
        for _, start, width in blocks:
            assert start == pos and width > 0
            pos += width
        assert pos == total
    assert len(lay.INPUT_NAMES) == lay.D_IN and len(lay.OUTPUT_NAMES) == lay.D_OUT
    assert len(set(lay.INPUT_NAMES)) == lay.D_IN and len(set(lay.OUTPUT_NAMES)) == lay.D_OUT
    if (nd, names) in TOTALS:
        assert TOTALS[nd, names] == (lay.D_IN, lay.D_OUT)
    # the descriptor columns move only with arms
    base_nf = 7 if nd == 2 else 8
    assert ((base_nf + 1, base_nf) if "arms" in names else (base_nf, base_nf - 1)) == (lay.NF, lay.F)


def test_feature_off_is_todays_layout_object() -> None:
    for nd in (2, 3):
        assert dims(_cfg(nd, ())).lay is layout.for_dims(nd) is layout.for_dims(nd, ())
        assert ext.layout_features(_cfg(nd, ())) == "" and ext.mask(_cfg(nd, ())) == 0


def test_all_on_offsets_3d_and_2d() -> None:
    lay3, lay2 = dims(_cfg(3, ext.EXTENSIONS)).lay, dims(_cfg(2, ext.EXTENSIONS)).lay
    # inputs: material 84, bonds 85..97, voice 98..104, farm 105, arms 106..112
    assert (lay3.IN_MATERIAL, lay3.IN_ROLE, lay3.IN_BOND_ROLE, lay3.IN_BOND_KIN, lay3.IN_BOND_FLOW) == (
        84, 85, 86, 90, 94,
    )  # fmt: skip
    assert (lay3.IN_VOICE, lay3.IN_VOICE_BEARING, lay3.IN_FARM, lay3.IN_SELF_ATK, lay3.IN_NBR_SIG) == (
        98, 102, 105, 106, 110,
    )  # fmt: skip
    assert (lay3.OUT_BUILD, lay3.OUT_ROLE_A, lay3.OUT_ROLE_B, lay3.OUT_VOICE, lay3.OUT_FARM_PUT) == (
        39, 40, 41, 42, 46,
    )  # fmt: skip
    assert lay3.OUT_Q_ATTACK_SIG == 47
    assert (lay2.IN_MATERIAL, lay2.IN_ROLE, lay2.IN_VOICE, lay2.IN_VOICE_BEARING, lay2.IN_FARM) == (
        69, 70, 83, 87, 89,
    )  # fmt: skip
    assert (lay2.OUT_BUILD, lay2.OUT_ROLE_A, lay2.OUT_VOICE, lay2.OUT_FARM_PUT, lay2.OUT_Q_ATTACK_SIG) == (
        35, 36, 38, 42, 43,
    )  # fmt: skip


def test_offsets_are_cumulative_over_enabled_features_only() -> None:
    lay = dims(_cfg(2, ("comm", "niche"))).lay
    assert lay.IN_VOICE == 69 and lay.IN_FARM == 69 + 6 and lay.OUT_VOICE == 35 and lay.OUT_FARM_PUT == 39
    assert (
        lay.IN_MATERIAL is None and lay.OUT_BUILD is None and lay.IN_SELF_ATK is None
    )  # off: None, never -1


def test_unregistered_names_are_attribute_errors() -> None:
    lay = layout.for_dims(2)
    with pytest.raises(AttributeError):
        lay.IN_NOT_A_BLOCK  # noqa: B018
    with pytest.raises(AttributeError):
        lay.D_SOMETHING  # noqa: B018
    assert lay.IN_VEL_UP is None  # a real 3D-only field, unchanged


def test_silent_blocks_are_the_declared_ones() -> None:
    lay = dims(_cfg(3, ext.EXTENSIONS)).lay
    assert lay.silent_in_rows() == ((105, 1), (106, 1), (107, 1), (108, 1), (109, 1), (110, 3))
    assert lay.silent_out_cols() == ((46, 1), (47, 1))


def test_version_integer_and_features_string() -> None:
    cfg = _cfg(3, ext.EXTENSIONS)
    assert ext.mask(cfg) == 0b11111
    assert layout.genome_layout_version(3, ext.mask(cfg)) == 2 + 4 * 31 == 126
    assert layout.genome_layout_version(2, 0) == layout.genome_layout_version(2) == 1
    assert ext.layout_features(cfg) == "arms.perc@1,structures@1,bonds@1,comm4@1,niche.farm@1"
    assert ext.mask(_cfg(2, ("comm",))) == 8 and layout.genome_layout_version(2, 8) == 33
    # decodes uniquely for bases 1..3
    for base, mask in itertools.product((1, 2, 3), (0, 1, 5, 31)):
        v = base + 4 * mask
        assert (v % 4, v // 4) == (base, mask)


def test_tail_order_puts_arms_last_and_state_order_is_extensions_order() -> None:
    assert ext.TAIL_ORDER == ("structures", "bonds", "comm", "niche", "arms")
    assert ext.EXTENSIONS == ("arms", "structures", "bonds", "comm", "niche")
    blocks = ext.layout_blocks(_cfg(2, ext.EXTENSIONS), 2)
    assert [owner for owner, *_ in blocks] == list(ext.TAIL_ORDER)


def test_dims_plumbing_of_the_voice_width(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(2, ("comm",))
    assert dims(cfg).voice_w == 0 and dims(cfg).msg_w == layout.MSG
    stub.install(monkeypatch, voice_w=4)
    cfg2 = stub.stub_cfg(
        load_config(preset="tiny", overrides=["brain.weight_dtype=float32", "run.seed=9"]), "comm"
    )
    assert (dims(cfg2).voice_w, dims(cfg2).msg_w, dims(cfg2).features) == (4, layout.MSG + 4, ("comm",))
    assert dims(load_config(preset="tiny")).msg_w == layout.MSG


def test_params_follow_the_composed_layout() -> None:
    d = dims(_cfg(2, ext.EXTENSIONS))
    assert d.w1_rows == d.lay.D_IN + d.h == 97 + d.h and d.params == d.w1_rows * d.h + d.w2_rows * 44
