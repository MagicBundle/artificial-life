"""One combat signature and the protection floor (DESIGN §43.6): the all-None call is the legacy combat, null
protections change nothing, the floor bounds the product, the stack keeps the integer pipeline exact, pierce
lifts only the environmental protections."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from alife import energy as en
from alife import ops
from alife.config import Config, apply_overrides, dims
from test_energy import _actions

N_TRIALS = 6


def _legacy_combat(energy, resource, fcell, act, partner, cfg):
    """A frozen copy of ``energy.combat`` before PR 0a (the reference of test 1)."""
    cc = cfg.combat
    n = energy.shape[0]
    tgt = act.attack_target
    attack = act.attack & (tgt >= 0)
    if not cc.attack_bonded:
        attack = attack & ~((partner == tgt[:, None]) & (partner >= 0)).any(axis=1)
    t0 = mx.maximum(tgt, 0)
    want = ops.floor_i32(cc.attack_max * (1.0 - cc.defend_efficiency * act.defend[t0]))
    want = mx.where(attack, mx.clip(want, 0, cc.attack_max), 0)
    grant, taken = ops.proportional_take(want, t0, attack, energy)
    demand = ops.scatter_add_i32(n, t0, want, attack)
    residue = mx.where(demand >= energy, energy - taken, 0)
    lost = taken + residue
    gain = ops.mul_q8(grant, ops.q8(cc.efficiency))
    spill = grant - gain
    dep = ops.scatter_add_i32(resource.size, fcell[t0], spill, attack) + ops.scatter_add_i32(
        resource.size, fcell, residue
    )
    return {
        "energy": energy - lost + gain, "resource": resource + dep.reshape(resource.shape), "lost": lost,
        "gain": gain, "spill": spill, "residue": residue, "attack": attack, "hit": attack & (grant > 0),
    }  # fmt: skip


def _batch(cfg: Config, seed: int):
    rs = np.random.default_rng(seed)
    d = dims(cfg)
    n = d.n
    e = rs.integers(0, 12_000, n).astype(np.int32)
    res = rs.integers(0, 500, d.field_shape).astype(np.int32)
    fcell = rs.integers(0, res.size, n).astype(np.int32)
    attack = rs.random(n) < 0.6
    tgt = rs.integers(0, n, n).astype(np.int32)
    defend = rs.random(n).astype(np.float32)
    act = _actions(n, attack=attack, attack_target=tgt, defend=defend)
    partner = np.full((n, 4), -1, np.int32)
    return mx.array(e), mx.array(res), mx.array(fcell), act, mx.array(partner), rs


def _fields(cb) -> dict[str, np.ndarray]:
    names = ("energy", "resource", "lost", "gain", "spill", "residue", "attack", "hit")
    return {k: np.array(getattr(cb, k)) for k in names}


def _same(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> None:
    for k in a:
        assert np.array_equal(a[k], b[k]), k


def test_all_none_call_equals_the_legacy_combat(micro_cfg: Config) -> None:
    for seed in range(N_TRIALS):
        e, res, fcell, act, partner, _ = _batch(micro_cfg, seed)
        cb = en.combat(e, res, fcell, act, partner, micro_cfg)
        ref = _legacy_combat(e, res, fcell, act, partner, micro_cfg)
        _same(_fields(cb), {k: np.array(v) for k, v in ref.items()})
        assert cb.prot is None and cb.capped is None and cb.shielded is None and cb.q is None
        assert int(cb.max_attackers) >= 1 and np.array(cb.hit).any()  # the batch exercises real attacks
    # the floor alone (no factor supplied) traces nothing either
    floor_cfg = apply_overrides(micro_cfg, ["combat.protect_floor=0.2"])
    e, res, fcell, act, partner, _ = _batch(micro_cfg, 0)
    _same(
        _fields(en.combat(e, res, fcell, act, partner, floor_cfg)),
        _fields(en.combat(e, res, fcell, act, partner, micro_cfg)),
    )


def test_null_protections_change_nothing(micro_cfg: Config) -> None:
    n = dims(micro_cfg).n
    zeros, ones = mx.zeros((n,), mx.float32), mx.ones((n,), mx.float32)
    for floor in ("0.0", "0.3"):
        cfg = apply_overrides(micro_cfg, [f"combat.protect_floor={floor}"])
        e, res, fcell, act, partner, _ = _batch(micro_cfg, 1)
        want = _fields(en.combat(e, res, fcell, act, partner, micro_cfg))
        for kw in ({"shelter": zeros}, {"guard": zeros}, {"shelter": zeros, "guard": zeros}):
            cb = en.combat(e, res, fcell, act, partner, cfg, **kw)
            _same(_fields(cb), want)
            assert np.array(cb.prot)[np.array(cb.attack)].tolist() == [1.0] * int(np.array(cb.attack).sum())
            assert not np.array(cb.capped).any()
        t0 = mx.maximum(act.attack_target, 0)
        p, capped = en.protection(t0, 0.3, arms_p=ones)  # a unit armour factor
        assert np.array(p).tolist() == [1.0] * n and not np.array(capped).any()


def test_floor_bounds_the_product(micro_cfg: Config) -> None:
    f = 0.25
    cfg = apply_overrides(micro_cfg, [f"combat.protect_floor={f}"])
    cc = cfg.combat
    for seed in range(N_TRIALS):
        e, res, fcell, act, partner, rs = _batch(cfg, seed)
        n = dims(cfg).n
        shelter = mx.array(rs.random(n).astype(np.float32) * 0.9)
        guard = mx.array(rs.random(n).astype(np.float32) * 0.9)
        cb = en.combat(e, res, fcell, act, partner, cfg, shelter=shelter, guard=guard)
        t0 = np.maximum(np.array(act.attack_target), 0)
        sh, gu = np.array(shelter)[t0], np.array(guard)[t0]
        prod = (np.float32(1) - sh) * (np.float32(1) - gu)
        attack = np.array(cb.attack)
        assert np.allclose(np.array(cb.prot)[attack], np.maximum(prod, np.float32(f))[attack], rtol=1e-6)
        assert np.array_equal(np.array(cb.capped), attack & (prod < np.float32(f)))
        assert (
            attack.any() and (prod[attack] < f).any() and (prod[attack] >= f).any()
        )  # both sides are exercised
        # the demand is at least the floor share of attack_max before ``defend``
        defend = np.array(act.defend)[t0]
        floor_demand = np.floor(
            np.float32(cc.attack_max * f) * (np.float32(1) - np.float32(cc.defend_efficiency) * defend)
        )
        dem = np.array(
            ops.floor_i32(cc.attack_max * cb.prot * (1.0 - cc.defend_efficiency * act.defend[mx.array(t0)]))
        )
        assert (dem[attack] >= floor_demand[attack] - 1).all()
        # PROT_CAPPED counts exactly the capped rows
        assert int(np.array(cb.capped).sum()) == int((attack & (prod < np.float32(f))).sum())
    # f = 0 traces no cap: the capped output is None, the product is the plain one
    p, capped = en.protection(mx.zeros((4,), mx.int32), 0.0, shelter=mx.full((4,), 0.99, mx.float32))
    assert capped is None and np.allclose(np.array(p), 0.01)


def test_stack_commutes_and_the_integer_pipeline_stays_exact(micro_cfg: Config) -> None:
    cfg = apply_overrides(micro_cfg, ["combat.protect_floor=0.1"])
    n = dims(cfg).n
    for seed in range(N_TRIALS):
        e, res, fcell, act, partner, rs = _batch(cfg, seed)
        a = mx.array(rs.random(n).astype(np.float32) * 0.8)
        b = mx.array(rs.random(n).astype(np.float32) * 0.8)
        c1 = en.combat(e, res, fcell, act, partner, cfg, shelter=a, guard=b)
        c2 = en.combat(e, res, fcell, act, partner, cfg, shelter=b, guard=a)
        np.testing.assert_allclose(
            np.array(c1.prot)[np.array(c1.attack)], np.array(c2.prot)[np.array(c2.attack)], rtol=1e-6
        )
        for cb in (c1, c2):
            f = _fields(cb)
            assert f["lost"].sum() == f["gain"].sum() + f["spill"].sum() + f["residue"].sum()  # FLOW_STOLEN
            assert np.array(e).sum() + np.array(res).sum() == f["energy"].sum() + f["resource"].sum()
            assert (f["energy"] >= 0).all()
        assert c1.shielded is not None and np.array(c1.shielded).sum() > 0


def test_pierce_lifts_only_the_environmental_protections(micro_cfg: Config) -> None:
    n = 32
    t0 = mx.arange(n, dtype=mx.int32)
    rs = np.random.default_rng(3)
    shelter = mx.array(rs.random(n).astype(np.float32) * 0.9)
    guard = mx.array(rs.random(n).astype(np.float32) * 0.9)
    armour = mx.array(rs.random(n).astype(np.float32) * 0.5 + 0.1)
    one = mx.ones((n,), mx.float32)
    p_full, _ = en.protection(t0, 0.05, arms_p=armour, shelter=shelter, guard=guard, pierce=one)
    p_arm, _ = en.protection(t0, 0.05, arms_p=armour)
    np.testing.assert_allclose(np.array(p_full), np.maximum(np.array(p_arm), 0.05), rtol=1e-6)  # armour stays
    p_none, _ = en.protection(t0, 0.05, arms_p=armour, shelter=shelter, guard=guard)
    p_half, _ = en.protection(t0, 0.05, arms_p=armour, shelter=shelter, guard=guard, pierce=0.5 * one)
    assert (np.array(p_half) >= np.array(p_none) - 1e-7).all() and (
        np.array(p_half) <= np.array(p_full) + 1e-7
    ).all()
    # a piercing attacker is still bounded below by the floor
    assert (np.array(en.protection(t0, 0.3, arms_p=armour * 0.1, pierce=one)[0]) >= 0.3).all()


def test_the_arms_argument_evaluates_the_contest(micro_cfg: Config) -> None:
    """A1: ``arms=`` is the arms contest factor (the contest itself is tested in ``test_arms``)."""
    cfg = apply_overrides(micro_cfg, ["arms.enabled=true"])
    e, res, fcell, act, partner, _ = _batch(cfg, 0)
    z = mx.zeros((dims(cfg).n, 6), mx.float32)
    got = en.combat(e, res, fcell, act, partner, cfg, arms=z)
    _same(_fields(got), _fields(en.combat(e, res, fcell, act, partner, cfg)))


def test_protection_floor_keeps_attack_profitable_at_the_recommended_value() -> None:
    """With the defaults a full-defend, maximally protected victim still pays an attacker more than the
    attack costs at f = 0.10 (70 > 50) and not at the arms spec's 0.02 (14 < 50)."""
    cfg = Config()
    cc, ec = cfg.combat, cfg.energy
    worst = lambda f: cc.efficiency * cc.attack_max * (1 - cc.defend_efficiency) * f  # noqa: E731
    assert worst(0.10) > ec.attack_cost > worst(0.02)
