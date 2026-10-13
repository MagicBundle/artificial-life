"""§27.7 the 3D energy costs: ``energy.pay_costs3`` (the pitch turn cost, buoyancy use and the cap's upkeep,
solvency and the payment order, the exact per-function ledger) and the two cost keys ``energy.buoyancy_cost``
and ``energy.buoyancy_cap_cost``."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from alife import actions, config, energy
from alife.actions import Actions, Motion3
from alife.config import (
    POST_V01_KEYS,
    THREE_D_KEYS,
    UNHASHED_KEYS,
    Config,
    ConfigError,
    config_from_dict,
    config_hash,
    config_to_dict,
    load_config,
)
from alife.layout import MSG, NCH, B
from helpers import ON_GPU, as_3d

TINY_HASH_V01 = "d25e0eaee7349ae366e3e4e60ebab107cf7a89fcf49be88a12d81099ff2561d5"  # test_features.py
COST_KEYS = (("energy", "buoyancy_cost"), ("energy", "buoyancy_cap_cost"))
TINY3D = ["world.dims=3", "world.height=24", "fields.chem_diffusion=0.16,0.1,0.16", "fields.benthic_lambda=8"]


@pytest.fixture(scope="module")
def cfg3(micro_cfg: Config) -> Config:
    """``micro`` as a 3D world with the actuator on and both buoyancy costs non-zero."""
    return as_3d(
        micro_cfg,
        physics={"buoyancy_max": 0.05},
        mutation={"buoy_init": 0.25, "buoy_sigma": 0.02},
        energy={"buoyancy_cost": 120, "buoyancy_cap_cost": 30},
    )


@pytest.fixture(scope="module")
def cfg3_off(micro_cfg: Config) -> Config:
    """``micro`` as a 3D world without the actuator (``buoyancy_max = 0``; the costs are then 0)."""
    return as_3d(micro_cfg)


@pytest.fixture
def enable_3d(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "DIMS_SUPPORTED", (2, 3))


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
    )


def _motion(pitch: np.ndarray | float, buoy: np.ndarray | float, n: int) -> Motion3:
    f32 = np.float32
    return Motion3(
        mx.array(np.broadcast_to(f32(pitch), n).copy()), mx.array(np.broadcast_to(f32(buoy), n).copy())
    )


def _ref_costs3(e, act: Actions, mo: Motion3, nb, bcost, cap, alive, cfg: Config):
    """float32 numpy reference of §27.7: §10.3 with the pitch in the turn term, plus the buoyancy due."""
    ec, f32 = cfg.energy, np.float32
    a = {k: np.array(v) for k, v in act._asdict().items()}
    pitch, buoy = np.array(mo.pitch), np.array(mo.buoy)

    def fl(x):
        return np.floor(np.asarray(x, f32)).astype(np.int32)

    basal = ec.basal_cost + bcost + ec.bond_upkeep * nb
    action = (
        fl(f32(ec.move_cost) * (a["thrust"] * a["thrust"]))
        + fl(f32(ec.turn_cost) * (np.abs(a["turn"]) + np.abs(pitch)) / f32(cfg.physics.turn_max))
        + fl(f32(ec.grip_cost) * a["grip"])
        + fl(f32(ec.contract_cost) * np.abs(a["contract"]) * nb.astype(f32))
        + fl(f32(ec.msg_cost) * np.abs(a["msg"]).sum(1))
        + fl(f32(ec.emit_cost) * a["emit"].sum(1))
        + fl(f32(ec.harvest_cost) * a["harvest"])
        + fl(f32(ec.defend_cost) * a["defend"])
        + ec.attack_cost * a["attack"].astype(np.int32)
    )
    lift = cap * buoy
    buoy_due = fl(f32(ec.buoyancy_cost) * (lift * lift)) + fl(f32(ec.buoyancy_cap_cost) * cap)
    basal, action, buoy_due = (np.where(alive, x, 0) for x in (basal, action, buoy_due))
    solvent = alive & (e >= basal + action + buoy_due)
    pb = np.minimum(basal, e)
    pa = np.minimum(action, e - pb)
    py = np.minimum(buoy_due, e - pb - pa)
    return e - pb - pa - py, solvent, pb, pa, py


def _pay(e, act, mo, nb, bcost, cap, alive, cfg) -> energy.Costs3:
    return energy.pay_costs3(
        mx.array(e), act, mo, mx.array(nb), mx.array(bcost), mx.array(cap), mx.array(alive), cfg
    )


# --- pay_costs3 ---------------------------------------------------------------------------------------------


def test_costs3_is_costs_plus_the_buoyancy_payment() -> None:
    assert energy.Costs._fields == ("energy", "solvent", "paid_basal", "paid_action")  # unchanged
    assert energy.Costs3._fields == (*energy.Costs._fields, "paid_buoy")


def test_pay_costs3_matches_reference_and_closes(cfg3: Config) -> None:
    rs = np.random.default_rng(3)
    n, tm = 200, cfg3.physics.turn_max
    for _ in range(10):
        alive = rs.random(n) < 0.8
        e = np.where(alive, rs.integers(0, 700, n), 0).astype(np.int32)
        nb = np.where(alive, rs.integers(0, B + 1, n), 0).astype(np.int32)
        bcost = np.where(alive, rs.integers(0, 200, n), 0).astype(np.int32)
        cap = rs.uniform(0, 1, n).astype(np.float32)  # stale on dead rows, as in the state
        keep = mx.array(alive)
        act = actions.mask(_random_actions(rs, n, cfg3), keep)
        mo = actions.mask(
            _motion(tm * np.tanh(rs.normal(0, 2, n)), np.maximum(np.tanh(rs.normal(0, 1, n)), 0), n), keep
        )
        c = _pay(e, act, mo, nb, bcost, cap, alive, cfg3)
        got = [np.array(x) for x in c]
        for g, r in zip(got, _ref_costs3(e, act, mo, nb, bcost, cap, alive, cfg3), strict=True):
            assert np.array_equal(g, r)
        e2, solvent, pb, pa, py = got
        assert all(x.dtype == mx.int32 for x in (c.energy, c.paid_basal, c.paid_action, c.paid_buoy))
        assert c.solvent.dtype == mx.bool_
        assert np.array_equal(e - e2, pb + pa + py)  # ledger: HEAT_BASAL + HEAT_ACTION + HEAT_BUOYANCY
        assert np.all(e2 >= 0) and np.all(pb >= 0) and np.all(pa >= 0) and np.all(py >= 0)
        assert np.all(e2[alive & ~solvent] == 0) and not solvent[~alive].any()
        assert not (pb[~alive].any() or pa[~alive].any() or py[~alive].any())  # dead rows pay 0
        assert 0 < (alive & ~solvent).sum() < alive.sum()
        ok = alive & solvent  # every solvent agent pays at least its cap's upkeep, floor(30·cap)
        assert np.all(py[ok] >= np.floor(np.float32(30) * cap[ok])) and py[ok].max() > 30


def test_buoyancy_due_exact_terms_and_payment_order(cfg3: Config) -> None:
    """``floor(120·(cap·b)²) + floor(30·cap)``, paid after basal and action; solvency on the total."""
    ec = cfg3.energy
    n = 6
    cap = np.array([0.5, 0.5, 0.5, 0.5, 1.0, 0.5], np.float32)
    buoy = np.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.5], np.float32)
    basal = ec.basal_cost  # no bonds, no brain cost, no actions
    buoy_due = 7 + 15  # floor(120·(0.5·0.5)²) + floor(30·0.5)
    e = np.array([basal + buoy_due, basal + buoy_due - 1, basal, 40, basal + 30, 0], np.int32)
    alive = np.array([True, True, True, True, True, False])
    mo = actions.mask(_motion(0.0, buoy, n), mx.array(alive))
    zeros = np.zeros(n, np.int32)
    c = _pay(e, _actions(n), mo, zeros, zeros, cap, alive, cfg3)
    e2, solvent, pb, pa, py = (np.array(x) for x in c)
    assert solvent.tolist() == [True, False, False, False, True, False]
    assert pb.tolist() == [basal, basal, basal, 40, basal, 0]
    assert pa.tolist() == [0] * n
    assert py.tolist() == [buoy_due, buoy_due - 1, 0, 0, 30, 0]  # buoyancy is paid last
    assert e2.tolist() == [0, 0, 0, 0, 0, 0]
    # Insolvency cancels the agent's 3D actions too (actions.mask with keep = solvent).
    assert np.array(actions.mask(mo, c.solvent).buoy).tolist() == [0.5, 0, 0, 0, 0, 0]


def test_cap_upkeep_is_paid_without_lift(cfg3: Config) -> None:
    n = 5
    cap = np.array([0.0, 0.25, 0.5, 1.0, 1.0], np.float32)
    alive = np.array([True, True, True, True, False])
    e = np.full(n, 10_000, np.int32) * alive
    zeros = np.zeros(n, np.int32)
    c = _pay(e, _actions(n), _motion(0.0, 0.0, n), zeros, zeros, cap, alive, cfg3)
    assert np.array(c.paid_buoy).tolist() == [0, 7, 15, 30, 0]  # floor(30·cap); a dead row pays nothing
    assert np.array(c.solvent).tolist() == [True, True, True, True, False]


def test_pitch_turn_cost(cfg3: Config) -> None:
    """``floor(turn_cost·(|turn| + |Δφ|)/turn_max)``: a pitch change costs like a yaw change, and they add."""
    n = 5
    turn = np.array([0.25, 0.0, 0.0, 0.25, -0.25], np.float32)
    pitch = np.array([0.0, 0.25, -0.25, 0.25, 0.25], np.float32)
    e = np.full(n, 10_000, np.int32)
    zeros = np.zeros(n, np.int32)
    caps = np.zeros(n, np.float32)
    c = _pay(e, _actions(n, turn=turn), _motion(pitch, 0.0, n), zeros, zeros, caps, np.ones(n, bool), cfg3)
    assert (cfg3.energy.turn_cost, cfg3.physics.turn_max) == (10, 0.3)
    assert np.array(c.paid_action).tolist() == [8, 8, 8, 16, 16]  # floor(10·0.25/0.3), floor(10·0.5/0.3)


@pytest.mark.parametrize("actuator", [True, False])
def test_level_and_capless_is_pay_costs_bitwise(cfg3: Config, cfg3_off: Config, actuator: bool) -> None:
    """With Δφ = 0 and buoy_cap = 0 (or no actuator) every shared output equals the 2D ``pay_costs``."""
    cfg = cfg3 if actuator else cfg3_off
    rs = np.random.default_rng(5)
    n = 300
    alive = rs.random(n) < 0.8
    e = np.where(alive, rs.integers(0, 700, n), 0).astype(np.int32)
    nb = np.where(alive, rs.integers(0, B + 1, n), 0).astype(np.int32)
    bcost = np.where(alive, rs.integers(0, 200, n), 0).astype(np.int32)
    act = actions.mask(_random_actions(rs, n, cfg), mx.array(alive))
    cap = np.zeros(n, np.float32) if actuator else rs.uniform(0, 1, n).astype(np.float32)  # stale: never read
    mo = _motion(0.0, np.maximum(np.tanh(rs.normal(0, 1, n)), 0), n)
    c3 = _pay(e, act, mo, nb, bcost, cap, alive, cfg)
    c2 = energy.pay_costs(mx.array(e), act, mx.array(nb), mx.array(bcost), mx.array(alive), cfg)
    for a, b in zip(c3[:4], c2, strict=True):
        assert np.array_equal(np.array(a), np.array(b))
    assert not np.array(c3.paid_buoy).any()
    assert int((alive & ~np.array(c2.solvent)).sum()) > 0


@pytest.mark.parametrize("actuator", [True, False])
def test_pay_costs3_compiles(cfg3: Config, cfg3_off: Config, actuator: bool) -> None:
    """Every output, including a ``paid_buoy`` that is 0 by construction without the actuator, comes out of
    ``mx.compile`` (MLX 0.32 fails on a constant output beside its users, I14)."""
    cfg = cfg3 if actuator else cfg3_off
    rs = np.random.default_rng(9)
    n = 128
    alive = rs.random(n) < 0.8
    e = mx.array(np.where(alive, rs.integers(0, 700, n), 0).astype(np.int32))
    act = actions.mask(_random_actions(rs, n, cfg), mx.array(alive))
    mo = _motion(rs.uniform(-0.3, 0.3, n), rs.uniform(0, 1, n), n)
    nb, bc = mx.zeros((n,), mx.int32), mx.full((n,), 37, mx.int32)
    cap, al = mx.array(rs.uniform(0, 1, n).astype(np.float32)), mx.array(alive)

    def fn(e, act, mo, nb, bc, cap, al):
        return energy.pay_costs3(e, act, mo, nb, bc, cap, al, cfg)

    got = mx.compile(fn)(e, act, mo, nb, bc, cap, al)
    want = fn(e, act, mo, nb, bc, cap, al)
    mx.eval(got, want)
    for a, b in zip(got, want, strict=True):
        if not ON_GPU:  # fused kernels may round the float terms differently on Metal
            assert np.array_equal(np.array(a), np.array(b))
    assert np.array_equal(np.array(e) - np.array(got.energy), sum(np.array(x) for x in got[2:]))


# --- the cost keys ------------------------------------------------------------------------------------------


def test_cost_keys_default_off_and_listed() -> None:
    assert THREE_D_KEYS[-2:] == COST_KEYS
    defaults = config_to_dict(Config())
    for sec, key in COST_KEYS:
        assert defaults[sec][key] == 0 and isinstance(defaults[sec][key], int), (sec, key)
        assert (sec, key) in POST_V01_KEYS and (sec, key) not in UNHASHED_KEYS, (sec, key)


@pytest.mark.parametrize(("sec", "key"), COST_KEYS)
def test_cost_key_hash_moves_only_off_default(sec: str, key: str) -> None:
    base = config_to_dict(load_config(preset="tiny"))
    assert config_hash(config_from_dict(base)) == TINY_HASH_V01
    assert config_hash(config_from_dict({**base, sec: {**base[sec], key: 5}})) != TINY_HASH_V01
    old = {**base, sec: {k: v for k, v in base[sec].items() if k != key}}  # a config.json from before PR 3
    assert config_hash(config_from_dict(old)) == TINY_HASH_V01


@pytest.mark.parametrize(("sec", "key"), COST_KEYS)
def test_cost_keys_rejected_in_2d(sec: str, key: str) -> None:
    with pytest.raises(ConfigError) as ei:
        load_config(preset="tiny", overrides=[f"{sec}.{key}=5"])
    msg = str(ei.value)
    assert msg.startswith("1 config problem(s)") and f"{sec}.{key} is a 3D key" in msg, msg


@pytest.mark.usefixtures("enable_3d")
def test_cost_keys_validate_in_3d() -> None:
    on = [*TINY3D, "physics.buoyancy_max=0.04", "mutation.buoy_init=0.5"]
    cfg = load_config(
        preset="tiny", overrides=[*on, "energy.buoyancy_cost=65535", "energy.buoyancy_cap_cost=3"]
    )
    assert (cfg.energy.buoyancy_cost, cfg.energy.buoyancy_cap_cost) == (65535, 3)
    load_config(preset="tiny", overrides=TINY3D)  # both 0 without the actuator


ACTUATOR = ["physics.buoyancy_max=0.04"]
BAD_COSTS: list[tuple[list[str], str]] = [
    ([*ACTUATOR, "energy.buoyancy_cost=-1"], "energy.buoyancy_cost must be ≥ 0, got -1"),
    ([*ACTUATOR, "energy.buoyancy_cap_cost=-2"], "energy.buoyancy_cap_cost must be ≥ 0, got -2"),
    ([*ACTUATOR, "energy.buoyancy_cost=65536"], "energy.buoyancy_cost 65536 must be < 2^16"),
    ([*ACTUATOR, "energy.buoyancy_cap_cost=70000"], "energy.buoyancy_cap_cost 70000 must be < 2^16"),
    (["energy.buoyancy_cost=5"], "must be 0 when physics.buoyancy_max = 0"),
    (["energy.buoyancy_cap_cost=5"], "must be 0 when physics.buoyancy_max = 0"),
]


@pytest.mark.usefixtures("enable_3d")
@pytest.mark.parametrize(("extra", "fragment"), BAD_COSTS)
def test_each_cost_rule_rejects(extra: list[str], fragment: str) -> None:
    with pytest.raises(ConfigError) as ei:
        load_config(preset="tiny", overrides=[*TINY3D, *extra])
    msg = str(ei.value)
    assert fragment in msg and msg.startswith("1 config problem(s)"), msg
