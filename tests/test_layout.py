"""§2.1, §5, §7 layout constants: sizes, block tiling and names."""

from __future__ import annotations

from alife import layout as L


def _tiles(blocks: tuple[tuple[str, int, int], ...], total: int) -> None:
    covered = [0] * total
    for _, start, width in blocks:
        assert width >= 1
        for i in range(start, start + width):
            assert 0 <= i < total
            covered[i] += 1
    assert covered == [1] * total, "blocks must be disjoint and tile [0, D)"
    starts = [s for _, s, _ in blocks]
    assert starts == sorted(starts)
    assert len({name for name, _, _ in blocks}) == len(blocks)


def test_sizes() -> None:
    assert L.D_IN == 69 and L.D_OUT == 35
    assert (L.B, L.NS, L.NF, L.F, L.MSG, L.NCH, L.T) == (4, 3, 7, 6, 2, 3, 4)
    assert L.F == L.NF - 1
    assert L.GENOME_LAYOUT_VERSION == 1


def test_input_blocks_tile() -> None:
    _tiles(L.INPUT_BLOCKS, L.D_IN)
    assert L.IN_NBR + L.NS * L.NF == L.IN_BOND
    assert L.IN_BOND + L.B * L.BOND_W == L.D_IN


def test_output_blocks_tile() -> None:
    _tiles(L.OUTPUT_BLOCKS, L.D_OUT)
    assert L.OUT_Q_ATTACK + L.F == L.D_OUT


def test_documented_indices() -> None:
    assert (L.IN_BIAS, L.IN_ENERGY, L.IN_AGE, L.IN_VEL, L.IN_NBONDS) == (0, 1, 2, 3, 5)
    assert (L.IN_HURT, L.IN_FED, L.IN_GIFT, L.IN_NOISE, L.IN_CLOCK) == (6, 7, 8, 9, 10)
    assert (L.IN_RES, L.IN_RES_GRAD, L.IN_CHEM, L.IN_CHEM_GRAD) == (12, 13, 15, 18)
    assert (L.IN_CROWD, L.IN_KIN_MEAN, L.IN_MSG_MEAN, L.IN_NBR, L.IN_BOND) == (24, 25, 26, 28, 49)
    assert (L.OUT_THRUST, L.OUT_TURN, L.OUT_MSG, L.OUT_EMIT, L.OUT_HARVEST) == (0, 1, 2, 4, 7)
    assert (L.OUT_GRIP, L.OUT_CONTRACT, L.OUT_BOND, L.OUT_RELEASE, L.OUT_SHARE) == (8, 9, 10, 11, 15)
    assert (L.OUT_ATTACK, L.OUT_DEFEND, L.OUT_REPRODUCE, L.OUT_CHILD_BOND) == (19, 20, 21, 22)
    assert (L.OUT_Q_BOND, L.OUT_Q_ATTACK) == (23, 29)
    nbr = (L.NBR_PRESENT, L.NBR_DX, L.NBR_DY, L.NBR_KIN, L.NBR_RELE, L.NBR_MSG0, L.NBR_BONDED)
    assert nbr == tuple(range(L.NF))
    bond = (L.BOND_PRESENT, L.BOND_DX, L.BOND_DY, L.BOND_MSG0, L.BOND_RELE)
    assert bond == tuple(range(L.BOND_W))


def test_names() -> None:
    assert len(L.INPUT_NAMES) == L.D_IN and len(set(L.INPUT_NAMES)) == L.D_IN
    assert len(L.OUTPUT_NAMES) == L.D_OUT and len(set(L.OUTPUT_NAMES)) == L.D_OUT
    assert len(L.NBR_FEATURE_NAMES) == L.NF and len(L.BOND_FEATURE_NAMES) == L.BOND_W
    assert L.INPUT_NAMES[L.IN_NOISE] == "noise"
    assert L.INPUT_NAMES[L.IN_KIN_MEAN] == "kin_mean"
    assert L.INPUT_NAMES[L.IN_NBR + 2 * L.NF + L.NBR_BONDED] == "nbr2_bonded"
    assert L.INPUT_NAMES[L.IN_BOND + 3 * L.BOND_W + L.BOND_RELE] == "bond3_rele"
    assert L.INPUT_NAMES[L.IN_CHEM_GRAD + 2 * 2 + 1] == "chem2_grad_lat"
    assert L.OUTPUT_NAMES[L.OUT_CHILD_BOND] == "child_bond"
    assert L.OUTPUT_NAMES[L.OUT_Q_ATTACK + 5] == "q_attack5"
    # Every name inside a block starts with the block name's first word.
    for blocks, names in ((L.INPUT_BLOCKS, L.INPUT_NAMES), (L.OUTPUT_BLOCKS, L.OUTPUT_NAMES)):
        for name, start, width in blocks:
            for i in range(start, start + width):
                assert names[i].startswith(name.split("_")[0]), (name, names[i])


def test_constants() -> None:
    assert (L.LR_SCALE, L.AGE_SCALE, L.RES_SAT, L.RES_GRAD_GAIN) == (2.0, 1000.0, 2.0, 4.0)
    assert L.EPS_LEN == 1e-6 and L.C_REF_MARGIN == 50
