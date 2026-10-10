"""§10.3–§10.7 and §11 energy: costs and solvency, harvest, combat, share, cap; exact per-function ledgers."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from alife import actions, energy
from alife.actions import Actions
from alife.config import Config, apply_overrides, dims
from alife.layout import MSG, NCH, B


def _actions(n: int, **kw: np.ndarray | float) -> Actions:
    """All-zero (inactive) actions with the given fields overridden."""
    z = np.zeros(n, np.float32)
    base = {
        "thrust": z, "turn": z, "msg": np.zeros((n, MSG), np.float32), "emit": np.zeros((n, NCH), np.float32),
        "harvest": z, "grip": z, "contract": z, "bond_want": np.zeros(n, bool),
        "release": np.zeros((n, B), bool), "share": np.zeros((n, B), np.float32), "attack": np.zeros(n, bool),
        "defend": z, "reproduce": np.zeros(n, bool), "child_bond": np.zeros(n, bool),
        "bond_k": np.full(n, -1, np.int32), "bond_target": np.full(n, -1, np.int32),
        "attack_k": np.full(n, -1, np.int32), "attack_target": np.full(n, -1, np.int32),
    }  # fmt: skip
    for k, v in kw.items():
        base[k] = np.broadcast_to(np.asarray(v, base[k].dtype), base[k].shape).copy()
    return Actions(**{k: mx.array(v) for k, v in base.items()})


def _random_actions(rs: np.random.Generator, n: int, cfg: Config) -> Actions:
    tm = cfg.physics.turn_max
    return _actions(
        n,
        thrust=np.tanh(rs.normal(0, 2, n)),
        turn=tm * np.tanh(rs.normal(0, 2, n)),
        msg=np.tanh(rs.normal(0, 2, (n, MSG))),
        emit=np.maximum(np.tanh(rs.normal(0, 2, (n, NCH))), 0),
        harvest=1 / (1 + np.exp(-rs.normal(0, 2, n))),
        grip=1 / (1 + np.exp(-rs.normal(0, 2, n))),
        contract=np.tanh(rs.normal(0, 2, n)),
        defend=1 / (1 + np.exp(-rs.normal(0, 2, n))),
        attack=rs.random(n) < 0.3,
        share=np.maximum(np.tanh(rs.normal(0, 1, (n, B))), 0),
    )


def _np(a: mx.array) -> np.ndarray:
    return np.array(a)


def _pair_bonds(n: int, pairs: list[tuple[int, int, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    partner = np.full((n, B), -1, np.int32)
    rev = np.full((n, B), -1, np.int32)
    for i, si, j, sj in pairs:
        partner[i, si], rev[i, si], partner[j, sj], rev[j, sj] = j, sj, i, si
    return partner, rev


# --- costs (S8) -------------------------------------------------------------------------------------------


def _ref_costs(e, act: Actions, nb, bcost, alive, cfg: Config):
    """float32 numpy reference of §10.3."""
    ec, f32 = cfg.energy, np.float32
    a = {k: np.array(v) for k, v in act._asdict().items()}

    def fl(x):
        return np.floor(x.astype(f32)).astype(np.int32)

    basal = ec.basal_cost + bcost + ec.bond_upkeep * nb
    action = (
        fl(f32(ec.move_cost) * (a["thrust"] * a["thrust"]))
        + fl(f32(ec.turn_cost) * np.abs(a["turn"]) / f32(cfg.physics.turn_max))
        + fl(f32(ec.grip_cost) * a["grip"])
        + fl(f32(ec.contract_cost) * np.abs(a["contract"]) * nb.astype(f32))
        + fl(f32(ec.msg_cost) * np.abs(a["msg"]).sum(1))
        + fl(f32(ec.emit_cost) * a["emit"].sum(1))
        + fl(f32(ec.harvest_cost) * a["harvest"])
        + fl(f32(ec.defend_cost) * a["defend"])
        + ec.attack_cost * a["attack"].astype(np.int32)
    )
    basal, action = np.where(alive, basal, 0), np.where(alive, action, 0)
    solvent = alive & (e >= basal + action)
    pb = np.minimum(basal, e)
    pa = np.minimum(action, e - pb)
    return e - pb - pa, solvent, pb, pa


def test_pay_costs_matches_reference_and_closes(micro_cfg: Config) -> None:
    rs = np.random.default_rng(0)
    n = 200
    for _ in range(10):
        alive = rs.random(n) < 0.8
        e = np.where(alive, rs.integers(0, 600, n), 0).astype(np.int32)
        nb = np.where(alive, rs.integers(0, B + 1, n), 0).astype(np.int32)
        bcost = np.where(alive, rs.integers(0, 200, n), 0).astype(np.int32)
        act = actions.mask(_random_actions(rs, n, micro_cfg), mx.array(alive))
        c = energy.pay_costs(mx.array(e), act, mx.array(nb), mx.array(bcost), mx.array(alive), micro_cfg)
        got = [_np(x) for x in c]
        for g, r in zip(got, _ref_costs(e, act, nb, bcost, alive, micro_cfg), strict=True):
            assert np.array_equal(g, r)
        e2, solvent, pb, pa = got
        assert c.energy.dtype == mx.int32 and c.paid_basal.dtype == mx.int32 and c.solvent.dtype == mx.bool_
        assert np.array_equal(e - e2, pb + pa)  # ledger: HEAT_BASAL + HEAT_ACTION
        assert np.all(pb + pa <= e) and np.all(e2 >= 0) and np.all(pb >= 0) and np.all(pa >= 0)
        assert np.all(e2[alive & ~solvent] == 0) and not solvent[~alive].any()
        assert np.all(pb[~alive] == 0) and np.all(pa[~alive] == 0)
        assert 0 < (alive & ~solvent).sum() < alive.sum()


def test_pay_costs_exact_terms_and_insolvency(micro_cfg: Config) -> None:
    ec = micro_cfg.energy
    n = 4
    act = _actions(n, thrust=0.5, harvest=0.55, attack=np.array([True, True, True, False]))
    nb = np.array([2, 2, 2, 0], np.int32)
    bcost = np.array([37, 37, 37, 0], np.int32)
    basal = ec.basal_cost + 37 + 2 * ec.bond_upkeep  # 147
    action = 25 + 5 + ec.attack_cost  # floor(100·0.25) + floor(10·0.55) + 50 = 80
    e = np.array([basal + action, basal + action - 1, 100, 0], np.int32)
    alive = np.array([True, True, True, False])
    c = energy.pay_costs(mx.array(e), act, mx.array(nb), mx.array(bcost), mx.array(alive), micro_cfg)
    e2, solvent, pb, pa = (_np(x) for x in c)
    assert solvent.tolist() == [True, False, False, False]
    assert e2.tolist() == [0, 0, 0, 0]
    assert pb.tolist() == [basal, basal, 100, 0] and pa.tolist() == [action, action - 1, 0, 0]
    # Insolvency cancels every action of that agent (actions.mask with keep = solvent).
    masked = actions.mask(act, c.solvent)
    assert _np(masked.attack).tolist() == [True, False, False, False]
    assert _np(masked.thrust).tolist() == [0.5, 0.0, 0.0, 0.0]


# --- harvest (S10) ----------------------------------------------------------------------------------------


def _ref_take(demand, target, active, supply):
    d = np.where(active, demand, 0).astype(np.int64)
    t = np.where(active, target, 0)
    tot = np.bincount(t, weights=d, minlength=len(supply)).astype(np.int64)
    q = supply.astype(np.float32) * np.float32(1 << 15) / np.maximum(tot, 1).astype(np.float32)
    ratio = np.floor(np.minimum(q, np.float32((1 << 15) + 1))).astype(np.int64) - 1
    ratio = np.where(tot <= supply, 1 << 15, np.clip(ratio, 0, 1 << 15))
    take = (d * ratio[t]) >> 15
    return take, np.bincount(t, weights=take, minlength=len(supply)).astype(np.int64)


def test_harvest_proportional_exact_and_closes(micro_cfg: Config) -> None:
    ec, fg = micro_cfg.energy, dims(micro_cfg).fg
    rs = np.random.default_rng(1)
    n = 120
    for trial in range(12):
        solvent = rs.random(n) < 0.85
        e = rs.integers(1, ec.e_max + 1, n).astype(np.int32)
        e[:5] = ec.e_max  # full agents demand nothing (never negative)
        fcell = rs.integers(0, 6 if trial % 2 else fg * fg, n).astype(np.int32)  # crowded or spread out
        res = rs.integers(0, 1500, (fg, fg)).astype(np.int32)
        res.reshape(-1)[0] = 0
        h = rs.random(n).astype(np.float32)
        act = _actions(n, harvest=h)
        e2, r2, got = energy.harvest(
            mx.array(e), mx.array(res), mx.array(fcell), act, mx.array(solvent), micro_cfg
        )
        e2, r2, got = _np(e2), _np(r2), _np(got)
        demand = np.clip(
            np.minimum(np.floor(h * np.float32(ec.harvest_max)), ec.e_max - e), 0, ec.harvest_max
        )
        demand = np.where(solvent, demand, 0)
        take, taken = _ref_take(demand, fcell, solvent, res.reshape(-1))
        assert np.array_equal(got, take) and np.array_equal(res.reshape(-1) - r2.reshape(-1), taken)
        assert np.array_equal(e2 - e, got) and got.sum() == (res - r2).sum()  # ledger: FLOW_HARVEST
        assert np.all(got >= 0) and np.all(got <= demand) and np.all(r2 >= 0) and np.all(e2 <= ec.e_max)
        assert np.all(got[~solvent] == 0) and np.all(got[:5] == 0)
        assert r2.dtype == np.int32 and r2.shape == (fg, fg)


def test_harvest_contested_and_uncontested_cells(micro_cfg: Config) -> None:
    fg = dims(micro_cfg).fg
    n = 4
    res = np.zeros((fg, fg), np.int32)
    res.reshape(-1)[[3, 7]] = (300, 10_000)
    fcell = np.array([3, 3, 7, 7], np.int32)
    h = np.array([1.0, 0.5, 1.0, 0.25], np.float32)  # demands 400, 200, 400, 100
    e = np.full(n, 1000, np.int32)
    _e2, r2, got = energy.harvest(
        mx.array(e),
        mx.array(res),
        mx.array(fcell),
        _actions(n, harvest=h),
        mx.ones(n, dtype=mx.bool_),
        micro_cfg,
    )
    got = _np(got)
    assert got[2:].tolist() == [400, 100]  # uncontested: served exactly
    assert 0 <= 300 - got[:2].sum() <= 2 + 1 and got[0] == pytest.approx(2 * got[1], abs=1)  # proportional
    assert _np(r2).reshape(-1)[[3, 7]].tolist() == [300 - got[:2].sum(), 9500]


# --- combat (S11) -----------------------------------------------------------------------------------------


def _combat(cfg, e, res, fcell, act, partner):
    cb = energy.combat(mx.array(e), mx.array(res), mx.array(fcell), act, mx.array(partner), cfg)
    return energy.Combat(*(_np(x) for x in cb))


def _check_combat_ledger(e, res, cb) -> None:
    assert cb.lost.sum() == cb.gain.sum() + cb.spill.sum()  # FLOW_STOLEN = ATTACK_GAIN + SPILL
    assert e.sum() + res.sum() == cb.energy.sum() + cb.resource.sum()
    assert np.array_equal(cb.energy, e - cb.lost + cb.gain)
    assert np.all(cb.energy >= 0) and np.all(cb.lost <= e)


def test_combat_mutual_attack(micro_cfg: Config) -> None:
    fg = dims(micro_cfg).fg
    n = 4
    e = np.array([10_000, 10_000, 500, 0], np.int32)
    res = np.zeros((fg, fg), np.int32)
    fcell = np.array([5, 9, 0, 0], np.int32)
    act = _actions(n, attack=np.array([1, 1, 0, 0], bool), attack_target=np.array([1, 0, -1, -1]))
    cb = _combat(micro_cfg, e, res, fcell, act, np.full((n, B), -1, np.int32))
    gain = (4000 * 179) >> 8  # 2796
    assert cb.lost.tolist() == [4000, 4000, 0, 0] and cb.gain.tolist() == [gain, gain, 0, 0]
    assert cb.spill.tolist() == [4000 - gain, 4000 - gain, 0, 0]
    assert cb.energy.tolist() == [10_000 - 4000 + gain] * 2 + [500, 0]
    # Each attacker's spill lands in its victim's cell.
    assert cb.resource.reshape(-1)[9] == 4000 - gain and cb.resource.reshape(-1)[5] == 4000 - gain
    assert cb.attack.tolist() == [True, True, False, False] and cb.hit.tolist() == [True, True, False, False]
    assert int(cb.max_attackers) == 1
    _check_combat_ledger(e, res, cb)


def test_combat_three_attackers_on_poor_victim_and_defend(micro_cfg: Config) -> None:
    fg = dims(micro_cfg).fg
    n = 6
    e = np.array([3000, 50, 50, 50, 8000, 50], np.int32)
    res = np.full((fg, fg), 7, np.int32)
    fcell = np.array([17, 0, 1, 2, 30, 3], np.int32)
    act = _actions(
        n,
        attack=np.array([0, 1, 1, 1, 0, 1], bool),
        attack_target=np.array([-1, 0, 0, 0, -1, 4]),
        defend=np.array([0.0, 0, 0, 0, 1.0, 0], np.float32),
    )
    cb = _combat(micro_cfg, e, res, fcell, act, np.full((n, B), -1, np.int32))
    # Three demands of 4000 against 3000 q: ratio floor(3000·2^15/12000) − 1 = 8191, grant 999 each.
    grant = (4000 * 8191) >> 15
    assert cb.lost[0] == 3 * grant and 0 <= 3000 - cb.lost[0] <= 3 + 1
    assert cb.gain[1:4].tolist() == [(grant * 179) >> 8] * 3
    # Defend 1.0 cuts the demand to floor(4000·0.25) = 1000.
    assert cb.lost[4] == 1000 and cb.gain[5] == (1000 * 179) >> 8
    flat = cb.resource.reshape(-1)
    assert flat[17] == 7 + cb.spill[1:4].sum() and flat[30] == 7 + cb.spill[5]
    assert int(cb.max_attackers) == 3 and cb.hit.tolist() == [False, True, True, True, False, True]
    _check_combat_ledger(e, res, cb)


def test_combat_drained_victim_and_zero_grant(micro_cfg: Config) -> None:
    fg = dims(micro_cfg).fg
    n = 3
    e = np.array([0, 100, 100], np.int32)
    res = np.zeros((fg, fg), np.int32)
    act = _actions(n, attack=np.array([0, 1, 1], bool), attack_target=np.array([-1, 0, 0]))
    cb = _combat(micro_cfg, e, res, np.zeros(n, np.int32), act, np.full((n, B), -1, np.int32))
    assert cb.attack.tolist() == [False, True, True] and not cb.hit.any() and cb.lost.sum() == 0
    _check_combat_ledger(e, res, cb)


def test_combat_drops_attack_on_new_partner(micro_cfg: Config) -> None:
    fg = dims(micro_cfg).fg
    n = 4
    e = np.full(n, 10_000, np.int32)
    res = np.zeros((fg, fg), np.int32)
    partner, _ = _pair_bonds(n, [(0, 2, 1, 0)])  # 0–1 formed at S9 this step
    act = _actions(n, attack=np.array([1, 0, 1, 0], bool), attack_target=np.array([1, -1, 3, -1]))
    cb = _combat(micro_cfg, e, res, np.zeros(n, np.int32), act, partner)
    assert cb.attack.tolist() == [False, False, True, False] and cb.lost[1] == 0 and cb.gain[0] == 0
    assert cb.lost[3] == 4000
    _check_combat_ledger(e, res, cb)
    cfg2 = apply_overrides(micro_cfg, ["combat.attack_bonded=true"])
    cb2 = _combat(cfg2, e, res, np.zeros(n, np.int32), act, partner)
    assert cb2.attack.tolist() == [True, False, True, False] and cb2.lost[1] == 4000
    _check_combat_ledger(e, res, cb2)


def test_combat_random_ledger(micro_cfg: Config) -> None:
    fg = dims(micro_cfg).fg
    rs = np.random.default_rng(2)
    n = 150
    for _ in range(10):
        e = rs.integers(0, 20_000, n).astype(np.int32)
        res = rs.integers(0, 5000, (fg, fg)).astype(np.int32)
        fcell = rs.integers(0, fg * fg, n).astype(np.int32)
        tgt = rs.integers(0, 12, n)  # many attackers per victim
        tgt = np.where(tgt == np.arange(n), -1, tgt)
        att = (rs.random(n) < 0.5) & (tgt >= 0)
        tgt = np.where(att, tgt, -1)
        act = _actions(n, attack=att, attack_target=tgt, defend=rs.random(n))
        partner, _ = _pair_bonds(n, [(int(i), 0, int(tgt[i]), 1) for i in np.flatnonzero(att)[:1]])
        cb = _combat(micro_cfg, e, res, fcell, act, partner)
        _check_combat_ledger(e, res, cb)
        assert np.all(cb.spill >= 0) and np.all(cb.gain >= 0) and np.all(cb.spill[~cb.attack] == 0)
        assert not cb.attack[np.flatnonzero(att)[0]]
        assert int(cb.max_attackers) == np.bincount(tgt[cb.attack], minlength=1).max()


# --- share (S12) ------------------------------------------------------------------------------------------


def _share(cfg, e, partner, rev, partner0, share, solvent):
    n = e.shape[0]
    sh = energy.share(
        mx.array(e), mx.array(partner), mx.array(rev), mx.array(partner0), _actions(n, share=share),
        mx.array(solvent), cfg,
    )  # fmt: skip
    return energy.Share(*(_np(x) for x in sh))


def _check_share_ledger(e, sh) -> None:
    assert e.sum() == sh.energy.sum() + sh.heat.sum()  # HEAT_SHARE is the only loss
    assert sh.out.sum() == sh.inflow.sum() + sh.heat.sum()
    assert np.array_equal(sh.energy, e - sh.out + sh.inflow)
    assert np.all(sh.out <= e) and np.all(sh.energy >= 0)


def test_share_chain_exact(micro_cfg: Config) -> None:
    n = 4
    partner, rev = _pair_bonds(n, [(0, 0, 1, 1), (1, 0, 2, 3)])  # A–B, B–C
    share = np.zeros((n, B), np.float32)
    share[0, 0] = 0.5  # A → B
    share[1, 0] = 0.5  # B → C
    e = np.array([5000, 5000, 5000, 0], np.int32)
    sh = _share(micro_cfg, e, partner, rev, partner, share, np.ones(n, bool))
    g = (500 * 230) >> 8  # 449: q8(0.9) = 230
    assert sh.out.tolist() == [500, 500, 0, 0] and sh.inflow.tolist() == [0, g, g, 0]
    assert sh.heat.tolist() == [0, 500 - g, 500 - g, 0]
    assert sh.energy.tolist() == [4500, 5000 - 500 + g, 5000 + g, 0]
    _check_share_ledger(e, sh)


def test_share_capped_by_energy_over_b(micro_cfg: Config) -> None:
    n = 5
    partner, rev = _pair_bonds(n, [(0, 0, 1, 0), (0, 1, 2, 0), (0, 2, 3, 0), (0, 3, 4, 0)])
    share = np.zeros((n, B), np.float32)
    share[0] = 0.999
    e = np.array([7, 0, 0, 0, 0], np.int32)
    sh = _share(micro_cfg, e, partner, rev, partner, share, np.ones(n, bool))
    assert sh.out[0] == 4 * (7 // B) and sh.energy[0] == 7 - 4
    _check_share_ledger(e, sh)


def test_share_gating(micro_cfg: Config) -> None:
    n = 6
    # Start of step: 0–1 (slot 0 / slot 0) and 2–3. At S9 0–1 broke and 0's slot 0 was refilled with 4;
    # 2–3 is unchanged; 5 bonded with 2 in 2's previously empty slot 1.
    partner0, _ = _pair_bonds(n, [(0, 0, 1, 0), (2, 0, 3, 0)])
    partner, rev = _pair_bonds(n, [(0, 0, 4, 0), (2, 0, 3, 0), (2, 1, 5, 0)])
    share = np.full((n, B), 0.9, np.float32)
    e = np.full(n, 10_000, np.int32)
    sh = _share(micro_cfg, e, partner, rev, partner0, share, np.ones(n, bool))
    g = (900 * 230) >> 8
    assert sh.out[0] == 0 and sh.out[4] == 0  # refilled slot: neither end sends
    assert sh.out[2] == 900 and sh.out[3] == 900  # unchanged bond sends both ways; new slot 2→5 does not
    assert sh.out[5] == 0 and sh.inflow[5] == 0 and sh.inflow[2] == g and sh.inflow[3] == g
    _check_share_ledger(e, sh)
    # An insolvent sender sends nothing but still receives.
    solvent = np.ones(n, bool)
    solvent[3] = False
    sh = _share(micro_cfg, e, partner, rev, partner0, share, solvent)
    assert sh.out[3] == 0 and sh.inflow[3] == g and sh.inflow[2] == 0
    _check_share_ledger(e, sh)


def test_share_random_ledger(micro_cfg: Config) -> None:
    rs = np.random.default_rng(3)
    n = 120
    for _ in range(10):
        pairs, free = [], np.ones((n, B), bool)
        for _ in range(150):
            i, j = rs.choice(n, 2, replace=False)
            if free[i].any() and free[j].any() and not any({i, j} == {a, c} for a, _, c, _ in pairs):
                si, sj = int(np.argmax(free[i])), int(np.argmax(free[j]))
                free[i, si] = free[j, sj] = False
                pairs.append((int(i), si, int(j), sj))
        partner, rev = _pair_bonds(n, pairs)
        partner0 = np.where(rs.random((n, B)) < 0.1, -1, partner)  # some slots are new this step
        e = rs.integers(0, 3000, n).astype(np.int32)
        share = np.maximum(np.tanh(rs.normal(0, 1, (n, B))), 0).astype(np.float32)
        sh = _share(micro_cfg, e, partner, rev, partner0, share, rs.random(n) < 0.9)
        _check_share_ledger(e, sh)
        assert sh.out.sum() > 0


# --- cap (S13) --------------------------------------------------------------------------------------------


def test_cap_energy(micro_cfg: Config) -> None:
    em = micro_cfg.energy.e_max
    e = np.array([0, 5, em, em + 1, em + 12_345], np.int32)
    e2, over = energy.cap_energy(mx.array(e), micro_cfg)
    e2, over = _np(e2), _np(over)
    assert e2.tolist() == [0, 5, em, em, em] and over.tolist() == [0, 0, 0, 1, 12_345]
    assert e.sum() == e2.sum() + over.sum() and over.dtype == np.int32


def test_energy_functions_compile(micro_cfg: Config) -> None:
    """The whole S8–S13 chain compiles and matches eager evaluation exactly (integer outputs)."""
    rs = np.random.default_rng(4)
    n, fg = dims(micro_cfg).n, dims(micro_cfg).fg
    alive = rs.random(n) < 0.9
    partner, rev = _pair_bonds(n, [(0, 0, 1, 0), (2, 0, 3, 0), (4, 1, 5, 2)])
    act0 = _random_actions(rs, n, micro_cfg)
    tgt = np.where(np.asarray(act0.attack), (np.arange(n) + 7) % n, -1).astype(np.int32)
    act0 = act0._replace(attack_target=mx.array(tgt))
    e = rs.integers(0, 50_000, n).astype(np.int32)
    res = rs.integers(0, 8000, (fg, fg)).astype(np.int32)
    fcell = rs.integers(0, fg * fg, n).astype(np.int32)

    def chain(e, act, partner, rev, alive, res, fcell):
        c = energy.pay_costs(
            e, act, (partner >= 0).sum(1).astype(mx.int32), mx.zeros_like(e), alive, micro_cfg
        )
        act = actions.mask(act, c.solvent)
        E, R, got = energy.harvest(c.energy, res, fcell, act, c.solvent, micro_cfg)
        cb = energy.combat(E, R, fcell, act, partner, micro_cfg)
        sh = energy.share(cb.energy, partner, rev, partner, act, c.solvent, micro_cfg)
        return c, got, cb, sh, energy.cap_energy(sh.energy, micro_cfg)

    args = (
        mx.array(e),
        act0,
        mx.array(partner),
        mx.array(rev),
        mx.array(alive),
        mx.array(res),
        mx.array(fcell),
    )
    eager, comp = chain(*args), mx.compile(chain)(*args)
    for a, b in zip(tree_flatten(eager), tree_flatten(comp), strict=True):
        assert np.array_equal(np.array(a[1]), np.array(b[1]))
