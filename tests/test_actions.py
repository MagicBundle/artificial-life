"""§7 actions: decode ranges and thresholds, query targeting (eligibility, argmax, tie rules) and masking."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from alife import actions, layout
from alife.actions import Actions
from alife.config import Config, apply_overrides, dims
from alife.layout import D_OUT, NF, B
from alife.spatial import Neighbors

N_ = 16  # rows used by the unit scenes (decode is shape-agnostic in N)


def _empty(cfg: Config, n: int = N_):
    """(y, desc, nbr, partner, alive) with no neighbours, empty bonds, every row alive, logits 0."""
    k = dims(cfg).k
    y = np.zeros((n, D_OUT), np.float32)
    desc = np.zeros((n, k, NF), np.float32)
    idx = np.tile(np.arange(n, dtype=np.int32)[:, None], (1, k))
    valid = np.zeros((n, k), bool)
    dist = np.full((n, k), cfg.sensing.radius, np.float32)
    partner = np.full((n, B), -1, np.int32)
    alive = np.ones(n, bool)
    return y, desc, idx, valid, dist, partner, alive


def _set_nbr(desc, idx, valid, dist, i: int, k: int, j: int, d: float, phi=(0.0,) * 6, bonded: bool = False):
    """Put neighbour ``j`` at lane ``k`` of row ``i`` with targeting descriptor ``phi`` (bonded flag last)."""
    idx[i, k], valid[i, k], dist[i, k] = j, True, d
    desc[i, k, layout.NBR_PRESENT] = 1.0
    desc[i, k, 1:] = phi
    desc[i, k, layout.NBR_BONDED] = 1.0 if bonded else 0.0


def _decode(cfg: Config, y, desc, idx, valid, dist, partner, alive) -> Actions:
    k = dims(cfg).k
    nbr = Neighbors(
        mx.array(idx),
        mx.array(valid),
        mx.zeros((idx.shape[0], k, 2)),
        mx.array(dist),
        mx.array(0, mx.int32),
        mx.array(0, mx.int32),
    )
    act = actions.decode(mx.array(y), mx.array(desc), nbr, mx.array(partner), mx.array(alive), cfg)
    mx.eval(act)
    return act


def _np(act: Actions) -> dict[str, np.ndarray]:
    return {f: np.array(v) for f, v in act._asdict().items()}


def _check_types(act: Actions, n: int) -> None:
    shapes = {"msg": (n, layout.MSG), "emit": (n, layout.NCH), "release": (n, B), "share": (n, B)}
    bools = {"bond_want", "release", "attack", "reproduce", "child_bond"}
    ints = {"bond_k", "bond_target", "attack_k", "attack_target"}
    for f, v in act._asdict().items():
        assert v.shape == shapes.get(f, (n,)), f
        want = mx.bool_ if f in bools else mx.int32 if f in ints else mx.float32
        assert v.dtype == want, (f, v.dtype)


def test_decode_ranges_at_extreme_logits(micro_cfg: Config) -> None:
    cfg = micro_cfg
    y, desc, idx, valid, dist, partner, alive = _empty(cfg, 3)
    y[0], y[1], y[2] = 60.0, -60.0, 0.0
    act = _decode(cfg, y, desc, idx, valid, dist, partner, alive)
    _check_types(act, 3)
    a = _np(act)
    tmax = cfg.physics.turn_max
    for f in ("thrust", "contract"):
        assert a[f][0] == pytest.approx(1.0) and a[f][1] == pytest.approx(-1.0) and a[f][2] == 0.0
        assert np.all(np.abs(a[f]) <= 1.0)
    assert (
        a["turn"][0] == pytest.approx(tmax) and a["turn"][1] == pytest.approx(-tmax) and a["turn"][2] == 0.0
    )
    assert np.all(np.abs(a["turn"]) <= tmax * (1 + 1e-7))
    assert np.allclose(a["msg"][0], 1.0) and np.allclose(a["msg"][1], -1.0) and np.all(a["msg"][2] == 0)
    for f in ("emit", "share"):  # max(tanh, 0) ∈ [0, 1)
        assert np.allclose(a[f][0], 1.0) and np.all(a[f][1] == 0.0) and np.all(a[f][2] == 0.0)
        assert np.all((a[f] >= 0) & (a[f] <= 1))
    for f in ("harvest", "grip", "defend"):  # sigmoid
        assert a[f][0] == pytest.approx(1.0) and a[f][1] == pytest.approx(0.0, abs=1e-12)
        assert a[f][2] == 0.5 and np.all((a[f] >= 0) & (a[f] <= 1))
    for f in ("bond_want", "reproduce", "child_bond"):  # y > 0, strict
        assert a[f].tolist() == [True, False, False]
    assert a["release"][0].all() and not a["release"][1:].any()
    # No neighbours: nothing to target, so attack is cancelled even at a huge logit.
    assert not a["attack"].any()
    for f in ("bond_k", "bond_target", "attack_k", "attack_target"):
        assert np.all(a[f] == -1)


def test_decode_thresholds_are_strict(micro_cfg: Config) -> None:
    cfg = micro_cfg
    rt, at = cfg.bonds.release_threshold, cfg.combat.threshold
    y, desc, idx, valid, dist, partner, alive = _empty(cfg, 4)
    eps = 1e-4
    for i, (bond, rel, att) in enumerate(
        [(0.0, rt, at), (eps, rt + eps, at + eps), (-eps, rt - eps, at - eps)]
    ):
        y[i, layout.OUT_BOND] = y[i, layout.OUT_REPRODUCE] = y[i, layout.OUT_CHILD_BOND] = bond
        y[i, layout.OUT_RELEASE : layout.OUT_RELEASE + B] = rel
        y[i, layout.OUT_ATTACK] = att
        _set_nbr(desc, idx, valid, dist, i, 0, 3, 1.0)  # an attackable neighbour
    a = _np(_decode(cfg, y, desc, idx, valid, dist, partner, alive))
    for f in ("bond_want", "reproduce", "child_bond"):
        assert a[f][:3].tolist() == [False, True, False]
    assert a["release"][:3].all(1).tolist() == [False, True, False]
    assert a["attack"][:3].tolist() == [False, True, False]
    assert a["attack_target"][:3].tolist() == [-1, 3, -1]
    # Per-slot release: only the slots above the threshold.
    y2, *rest = _empty(cfg, 1)
    y2[0, layout.OUT_RELEASE : layout.OUT_RELEASE + B] = (rt + 1, rt - 1, rt + 0.5, rt)
    assert _np(_decode(cfg, y2, *rest))["release"][0].tolist() == [True, False, True, False]


def test_attack_target_is_argmax_over_eligible(micro_cfg: Config) -> None:
    cfg = micro_cfg
    y, desc, idx, valid, dist, partner, alive = _empty(cfg)
    rng = cfg.combat.range
    q = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], np.float32)  # score = NBR_DX
    y[0, layout.OUT_ATTACK] = 5.0
    y[0, layout.OUT_Q_ATTACK : layout.OUT_Q_ATTACK + 6] = q
    _set_nbr(desc, idx, valid, dist, 0, 0, 4, 1.0, phi=(0.1, 0, 0, 0, 0, 0))
    _set_nbr(desc, idx, valid, dist, 0, 1, 5, 1.5, phi=(0.9, 0, 0, 0, 0, 0), bonded=True)  # best, but bonded
    _set_nbr(desc, idx, valid, dist, 0, 2, 6, 2.0, phi=(0.5, 0, 0, 0, 0, 0))  # best eligible
    _set_nbr(desc, idx, valid, dist, 0, 3, 7, rng, phi=(0.95, 0, 0, 0, 0, 0))  # out of range (dist == range)
    valid[0, 4], desc[0, 4, 1] = False, 5.0  # an invalid lane with a huge score is ignored
    a = _np(_decode(cfg, y, desc, idx, valid, dist, partner, alive))
    assert a["attack"][0] and a["attack_k"][0] == 2 and a["attack_target"][0] == 6
    # attack_bonded lets the bonded neighbour win.
    cfg_b = apply_overrides(cfg, ["combat.attack_bonded=true"])
    a = _np(_decode(cfg_b, y, desc, idx, valid, dist, partner, alive))
    assert a["attack_k"][0] == 1 and a["attack_target"][0] == 5
    # A negative query flips the preference to the lowest score among the eligible.
    y[0, layout.OUT_Q_ATTACK] = -1.0
    a = _np(_decode(cfg, y, desc, idx, valid, dist, partner, alive))
    assert a["attack_k"][0] == 0 and a["attack_target"][0] == 4


def test_targeting_ties_go_to_the_nearest(micro_cfg: Config) -> None:
    cfg = micro_cfg
    y, desc, idx, valid, dist, partner, alive = _empty(cfg)
    y[0, layout.OUT_ATTACK] = 1.0
    y[0, layout.OUT_BOND] = 1.0
    y[3:6, layout.OUT_BOND] = 1.0  # the neighbours are willing and have free slots
    for k, j in enumerate((3, 4, 5)):
        _set_nbr(desc, idx, valid, dist, 0, k, j, 1.0 + 0.5 * k, phi=(0.0, 0.3, 0.0, 0.0, 0.0, 0.0))
    # q = 0 makes every score 0: all tie.
    a = _np(_decode(cfg, y, desc, idx, valid, dist, partner, alive))
    assert a["attack_k"][0] == 0 and a["attack_target"][0] == 3
    assert a["bond_k"][0] == 0 and a["bond_target"][0] == 3
    # Make the nearest ineligible for attack (bonded): the tie then goes to the next lowest k.
    desc[0, 0, layout.NBR_BONDED] = 1.0
    a = _np(_decode(cfg, y, desc, idx, valid, dist, partner, alive))
    assert a["attack_k"][0] == 1 and a["bond_k"][0] == 1


def test_bond_target_requires_consenting_partner(micro_cfg: Config) -> None:
    cfg = micro_cfg
    fr = cfg.bonds.form_range
    y, desc, idx, valid, dist, partner, alive = _empty(cfg)
    q = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)  # score = NBR_KIN
    y[0, layout.OUT_BOND] = 1.0
    y[0, layout.OUT_Q_BOND : layout.OUT_Q_BOND + 6] = q
    # k=0: unwilling (bond logit ≤ 0); k=1: willing but full; k=2: dead; k=3: out of form range;
    # k=4: already my partner; k=5: willing, free, alive, in range → the only eligible one, though lowest kin.
    for k, (j, d, kin) in enumerate(
        [(3, 1.0, 0.9), (4, 1.0, 0.9), (5, 1.0, 0.9), (6, fr, 0.9), (7, 1.0, 0.9)]
    ):
        _set_nbr(desc, idx, valid, dist, 0, k, j, d, phi=(0.0, 0.0, kin, 0.0, 0.0, 0.0), bonded=(j == 7))
    _set_nbr(desc, idx, valid, dist, 0, 5, 8, 2.5, phi=(0.0, 0.0, 0.1, 0.0, 0.0, 0.0))
    y[[4, 5, 6, 7, 8], layout.OUT_BOND] = 1.0
    y[3, layout.OUT_BOND] = 0.0
    partner[4] = (9, 10, 11, 12)
    alive[5] = False
    partner[0, 1], partner[7, 0] = 7, 0
    a = _np(_decode(cfg, y, desc, idx, valid, dist, partner, alive))
    assert a["bond_want"][0] and a["bond_k"][0] == 5 and a["bond_target"][0] == 8
    # Unwilling myself → no target even with eligible neighbours.
    y[0, layout.OUT_BOND] = -1.0
    a = _np(_decode(cfg, y, desc, idx, valid, dist, partner, alive))
    assert not a["bond_want"][0] and a["bond_k"][0] == -1 and a["bond_target"][0] == -1
    # A full row of my own makes nobody pick me.
    y[0, layout.OUT_BOND] = 1.0
    y[8, layout.OUT_BOND] = 1.0
    d2, i2, v2, dist2 = desc.copy(), idx.copy(), valid.copy(), dist.copy()
    _set_nbr(d2, i2, v2, dist2, 8, 0, 0, 1.0)
    partner[0] = (7, 13, 14, 15)
    a = _np(_decode(cfg, y, d2, i2, v2, dist2, partner, alive))
    assert a["bond_target"][8] == -1


def test_dead_rows_are_masked(micro_cfg: Config) -> None:
    cfg = micro_cfg
    y, desc, idx, valid, dist, partner, alive = _empty(cfg, 4)
    y[:] = 9.0
    for i in range(4):
        _set_nbr(desc, idx, valid, dist, i, 0, (i + 1) % 4, 1.0)
    alive[1] = False
    a = _np(_decode(cfg, y, desc, idx, valid, dist, partner, alive))
    for f, v in a.items():
        row = v[1]
        if v.dtype == bool:
            assert not np.any(row), f
        elif v.dtype == np.int32:
            assert np.all(row == -1), f
        else:
            assert np.all(row == 0.0), f
    # Row 0's only neighbour is the dead row 1: no attack and no bond target (guard on neighbour liveness).
    assert a["attack_target"][0] == -1 and a["bond_target"][0] == -1
    assert a["attack_target"][2] == 3 and a["attack"][2]


def test_mask_cancels_everything(micro_cfg: Config) -> None:
    cfg = micro_cfg
    y, desc, idx, valid, dist, partner, alive = _empty(cfg, 6)
    rs = np.random.default_rng(0)
    y[:] = rs.normal(0, 3, y.shape)
    y[:, layout.OUT_ATTACK] = 5.0
    y[:, layout.OUT_BOND] = 5.0
    for i in range(6):
        _set_nbr(desc, idx, valid, dist, i, 0, (i + 1) % 6, 1.0, phi=tuple(rs.normal(0, 1, 6)))
    act = _decode(cfg, y, desc, idx, valid, dist, partner, alive)
    keep = np.array([True, False, True, False, False, True])
    m = actions.mask(act, mx.array(keep))
    _check_types(m, 6)
    full, out = _np(act), _np(m)
    for f in Actions._fields:
        assert np.array_equal(out[f][keep], full[f][keep]), f
        dropped = out[f][~keep]
        want = False if dropped.dtype == bool else -1 if dropped.dtype == np.int32 else 0.0
        assert np.all(dropped == want), f
    assert full["attack"].all() and full["bond_k"].min() == 0  # the masked rows really had actions
    # mask(·, all True) is the identity; mask is idempotent.
    same = _np(actions.mask(act, mx.ones((6,), dtype=mx.bool_)))
    assert all(np.array_equal(same[f], full[f]) for f in Actions._fields)
    twice = _np(actions.mask(m, mx.array(keep)))
    assert all(np.array_equal(twice[f], out[f]) for f in Actions._fields)


def test_decode_compiles(micro_cfg: Config) -> None:
    cfg = micro_cfg
    y, desc, idx, valid, dist, partner, alive = _empty(cfg)
    rs = np.random.default_rng(1)
    y[:] = rs.normal(0, 2, y.shape)
    for i in range(N_):
        for k in range(3):
            _set_nbr(desc, idx, valid, dist, i, k, (i + k + 1) % N_, 0.5 + k, phi=tuple(rs.uniform(-1, 1, 6)))
    eager = _decode(cfg, y, desc, idx, valid, dist, partner, alive)
    k = dims(cfg).k
    nbr = Neighbors(
        mx.array(idx),
        mx.array(valid),
        mx.zeros((N_, k, 2)),
        mx.array(dist),
        mx.array(0, mx.int32),
        mx.array(0, mx.int32),
    )
    fn = mx.compile(lambda y_, d_, n_, p_, a_: actions.decode(y_, d_, n_, p_, a_, cfg))
    comp = fn(mx.array(y), mx.array(desc), nbr, mx.array(partner), mx.array(alive))
    for f in Actions._fields:
        assert np.allclose(np.array(getattr(eager, f)), np.array(getattr(comp, f)), atol=1e-6), f
