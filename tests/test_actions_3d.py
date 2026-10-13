"""§27.2 the 3D action decode: the seven-wide queries in the order of φ, ``Motion3`` (pitch change and
buoyancy fraction), masking, and the 2D decode as the 3D one's prefix."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from alife import actions, layout
from alife.actions import Actions, Motion3
from alife.config import Config, dims
from alife.layout import B
from alife.spatial import Neighbors
from helpers import as_3d

N_ = 16


@pytest.fixture(scope="module")
def cfg3(micro_cfg: Config) -> Config:
    return as_3d(micro_cfg)


def _empty(cfg: Config, n: int = N_):
    """(y, desc, idx, valid, dist, partner, alive) at the config's widths: no neighbours, logits 0, alive."""
    d = dims(cfg)
    y = np.zeros((n, d.lay.D_OUT), np.float32)
    desc = np.zeros((n, d.k, d.lay.NF), np.float32)
    idx = np.tile(np.arange(n, dtype=np.int32)[:, None], (1, d.k))
    valid = np.zeros((n, d.k), bool)
    dist = np.full((n, d.k), cfg.sensing.radius, np.float32)
    partner = np.full((n, B), -1, np.int32)
    return y, desc, idx, valid, dist, partner, np.ones(n, bool)


def _nbr(cfg: Config, idx, valid, dist) -> Neighbors:
    d = dims(cfg)
    zero = mx.array(0, mx.int32)
    off = mx.zeros((idx.shape[0], d.k, d.nd))
    return Neighbors(mx.array(idx), mx.array(valid), off, mx.array(dist), zero, zero)


def _decode(cfg: Config, y, desc, idx, valid, dist, partner, alive) -> Actions:
    nbr = _nbr(cfg, idx, valid, dist)
    act = actions.decode(mx.array(y), mx.array(desc), nbr, mx.array(partner), mx.array(alive), cfg)
    mx.eval(act)
    return act


def _np(t: Actions | Motion3) -> dict[str, np.ndarray]:
    return {f: np.array(v) for f, v in t._asdict().items()}


# --- Motion3 ------------------------------------------------------------------------------------------------


def test_decode_motion3_ranges_at_extreme_logits(cfg3: Config) -> None:
    lay, tmax = dims(cfg3).lay, cfg3.physics.turn_max
    y = np.zeros((3, lay.D_OUT), np.float32)
    y[0], y[1] = 60.0, -60.0
    mo = actions.decode_motion3(mx.array(y), mx.ones((3,), dtype=mx.bool_), cfg3)
    assert isinstance(mo, Motion3)
    for f in Motion3._fields:
        assert getattr(mo, f).shape == (3,) and getattr(mo, f).dtype == mx.float32, f
    m = _np(mo)
    assert (
        m["pitch"][0] == pytest.approx(tmax) and m["pitch"][1] == pytest.approx(-tmax) and m["pitch"][2] == 0
    )
    assert np.all(np.abs(m["pitch"]) <= tmax * (1 + 1e-7))
    # max(tanh, 0): saturates at 1 and is exactly 0 (free) for every non-positive logit.
    assert m["buoy"][0] == pytest.approx(1.0) and m["buoy"][1] == 0.0 and m["buoy"][2] == 0.0
    assert np.all((m["buoy"] >= 0) & (m["buoy"] <= 1))


def test_decode_motion3_reads_only_its_columns(cfg3: Config) -> None:
    lay, tmax = dims(cfg3).lay, cfg3.physics.turn_max
    y = np.random.default_rng(0).normal(0, 2, (N_, lay.D_OUT)).astype(np.float32)
    m = _np(actions.decode_motion3(mx.array(y), mx.ones((N_,), dtype=mx.bool_), cfg3))
    np.testing.assert_allclose(m["pitch"], tmax * np.tanh(y[:, lay.OUT_PITCH]), atol=1e-6)
    np.testing.assert_allclose(m["buoy"], np.maximum(np.tanh(y[:, lay.OUT_BUOYANCY]), 0.0), atol=1e-6)
    assert (m["buoy"] == 0).any() and (m["buoy"] > 0).any()
    y2 = y.copy()
    y2[:, : layout.D_OUT] = 0.0  # the 2D outputs do not feed Motion3
    y2[:, lay.OUT_Q_BOND_DZ] = y2[:, lay.OUT_Q_ATTACK_DZ] = 0.0
    m2 = _np(actions.decode_motion3(mx.array(y2), mx.ones((N_,), dtype=mx.bool_), cfg3))
    assert all(np.array_equal(m[f], m2[f]) for f in Motion3._fields)


def test_decode_motion3_is_masked_and_3d_only(cfg3: Config, micro_cfg: Config) -> None:
    y = np.full((6, dims(cfg3).lay.D_OUT), 2.0, np.float32)
    keep = np.array([True, False, True, False, False, True])
    m = _np(actions.decode_motion3(mx.array(y), mx.array(keep), cfg3))
    for f in Motion3._fields:
        assert np.all(m[f][keep] > 0) and np.all(m[f][~keep] == 0.0), f
    full = actions.decode_motion3(mx.array(y), mx.ones((6,), dtype=mx.bool_), cfg3)
    masked = actions.mask(full, mx.array(keep))
    assert type(masked) is Motion3  # mask keeps the type (Actions stays Actions)
    assert all(np.array_equal(_np(masked)[f], m[f]) for f in Motion3._fields)
    with pytest.raises(AssertionError, match="3D only"):
        actions.decode_motion3(mx.zeros((6, layout.D_OUT)), mx.ones((6,), dtype=mx.bool_), micro_cfg)


# --- the 3D queries ---------------------------------------------------------------------------------------


def test_query_dz_column_scores_nbr_dz(cfg3: Config) -> None:
    """``*_DZ`` multiplies the descriptor's last feature (``NBR_DZ``, φ's seventh entry), and the bond and
    attack queries do not share it."""
    lay = dims(cfg3).lay
    for q_bond_dz, q_att_dz, want_b, want_a in [(1.0, -1.0, 1, 0), (-1.0, 1.0, 0, 1)]:
        y, desc, idx, valid, dist, partner, alive = _empty(cfg3, 3)
        y[:, layout.OUT_BOND] = 1.0  # every row wants to bond, so both neighbours can accept
        y[0, layout.OUT_ATTACK] = 1.0
        for k, (j, dz) in enumerate([(1, -0.5), (2, 0.5)]):  # nearest below, the other above
            idx[0, k], valid[0, k], dist[0, k] = j, True, 1.0 + 0.5 * k
            desc[0, k, layout.NBR_PRESENT], desc[0, k, lay.NBR_DZ] = 1.0, dz
        y[0, lay.OUT_Q_BOND_DZ], y[0, lay.OUT_Q_ATTACK_DZ] = q_bond_dz, q_att_dz
        a = _np(_decode(cfg3, y, desc, idx, valid, dist, partner, alive))
        assert (a["bond_k"][0], a["bond_target"][0]) == (want_b, 1 + want_b)
        assert (a["attack_k"][0], a["attack_target"][0]) == (want_a, 1 + want_a) and a["attack"][0]
    # The 2D block still scores φ's first six entries: dx·1 + dz·3 picks the neighbour above.
    desc[0, :2, layout.NBR_DX] = [0.5, -0.5]
    y[0, layout.OUT_Q_BOND : layout.OUT_Q_BOND + layout.F] = [1.0, 0, 0, 0, 0, 0]
    y[0, lay.OUT_Q_BOND_DZ] = 0.0
    assert _np(_decode(cfg3, y, desc, idx, valid, dist, partner, alive))["bond_k"][0] == 0
    y[0, lay.OUT_Q_BOND_DZ] = 3.0
    assert _np(_decode(cfg3, y, desc, idx, valid, dist, partner, alive))["bond_k"][0] == 1


def test_2d_outputs_match_the_2d_decode_when_dz_is_zero(micro_cfg: Config, cfg3: Config) -> None:
    """With every ``NBR_DZ`` 0 the 3D decode of outputs 0–34 is the 2D decode, whatever the DZ queries say."""
    rs = np.random.default_rng(2)
    k = dims(cfg3).k
    y, desc, idx, valid, dist, partner, alive = _empty(cfg3)
    y[:] = rs.normal(0, 2, y.shape)
    y[:, layout.OUT_ATTACK] += 1.0
    valid[:] = rs.random((N_, k)) < 0.7
    idx[:] = np.where(valid, rs.integers(0, N_, (N_, k)), idx)
    dist[:] = np.where(valid, rs.uniform(0.2, cfg3.sensing.radius, (N_, k)), dist)
    desc[..., : layout.NF] = rs.uniform(-1, 1, (N_, k, layout.NF))
    desc[..., layout.NBR_PRESENT] = valid
    desc[..., layout.NBR_BONDED] = rs.random((N_, k)) < 0.2
    desc[~valid] = 0.0
    partner[:] = np.where(rs.random((N_, B)) < 0.3, rs.integers(0, N_, (N_, B)), -1)
    alive[:] = rs.random(N_) < 0.9
    a3 = _np(_decode(cfg3, y, desc, idx, valid, dist, partner, alive))
    a2 = _np(
        _decode(micro_cfg, y[:, : layout.D_OUT], desc[..., : layout.NF], idx, valid, dist, partner, alive)
    )
    for f in Actions._fields:
        assert np.array_equal(a3[f], a2[f]), f
    assert (a3["bond_k"] >= 0).any() and (a3["attack_k"] >= 0).any()  # targeting really ran


def test_decode_3d_compiles(cfg3: Config) -> None:
    rs = np.random.default_rng(3)
    y, desc, idx, valid, dist, partner, alive = _empty(cfg3)
    y[:] = rs.normal(0, 2, y.shape)
    for i in range(N_):
        for k in range(3):
            idx[i, k], valid[i, k], dist[i, k] = (i + k + 1) % N_, True, 0.5 + k
            desc[i, k, 1:] = rs.uniform(-1, 1, dims(cfg3).lay.F)
            desc[i, k, layout.NBR_PRESENT], desc[i, k, layout.NBR_BONDED] = 1.0, 0.0
    eager = _decode(cfg3, y, desc, idx, valid, dist, partner, alive)
    nbr = _nbr(cfg3, idx, valid, dist)
    fn = mx.compile(lambda y_, d_, n_, p_, a_: actions.decode(y_, d_, n_, p_, a_, cfg3))
    comp = fn(mx.array(y), mx.array(desc), nbr, mx.array(partner), mx.array(alive))
    for f in Actions._fields:
        assert np.allclose(np.array(getattr(eager, f)), np.array(getattr(comp, f)), atol=1e-6), f
    mo = actions.decode_motion3(mx.array(y), mx.array(alive), cfg3)
    mo_c = mx.compile(lambda y_, a_: actions.decode_motion3(y_, a_, cfg3))(mx.array(y), mx.array(alive))
    for f in Motion3._fields:
        assert np.allclose(np.array(getattr(mo, f)), np.array(getattr(mo_c, f)), atol=1e-6), f
