"""§2 state: init_state schema and values, totals, (de)serialisation, digests and host reads."""

from __future__ import annotations

import random

import mlx.core as mx
import numpy as np
import pytest

from alife import bonds, layout, ops
from alife.config import Config, dims, load_config
from alife.state import (
    AGENT_FIELDS,
    ERROR_MASK,
    N_COUNTERS,
    N_LEDGER,
    R_HARD,
    CounterIdx,
    DeathCause,
    Diag,
    LedgerIdx,
    State,
    _initial_state,
    age,
    init_state,
    state_digest,
    state_from_arrays,
    state_to_arrays,
    to_numpy,
    total_energy,
)
from helpers import make_scene


def _expected_schema(cfg: Config) -> dict[str, tuple[tuple[int, ...], mx.Dtype]]:
    d = dims(cfg)
    n, m, h, fg, w = d.n, d.m, d.h, d.fg, d.wdtype
    i32, f32, i64 = mx.int32, mx.float32, mx.int64
    return {
        "alive": ((n,), mx.bool_),
        "uid": ((n,), i64),
        "parent_uid": ((n,), i64),
        "clade": ((n,), i32),
        "generation": ((n,), i32),
        "birth_step": ((n,), i32),
        "energy": ((n,), i32),
        "pos": ((n, 2), f32),
        "vel": ((n, 2), f32),
        "heading": ((n,), f32),
        "h": ((n, h), f32),
        "msg": ((n, layout.MSG), f32),
        "last_loss": ((n,), i32),
        "last_harvest": ((n,), i32),
        "last_inflow": ((n,), i32),
        "partner": ((n, layout.B), i32),
        "rev": ((n, layout.B), i32),
        "tag": ((n, layout.T), f32),
        "mut_sigma": ((n,), f32),
        "mut_rate": ((n,), f32),
        "offspring_frac": ((n,), f32),
        "brain_cost": ((n,), i32),
        "W1": ((n, d.w1_rows, h), w),
        "W2": ((n, d.w2_rows, layout.D_OUT), w),
        "pend_slot": ((m,), i32),
        "pend_W1": ((m, d.w1_rows, h), w),
        "pend_W2": ((m, d.w2_rows, layout.D_OUT), w),
        "resource": ((fg, fg), i32),
        "chem": ((layout.NCH, fg, fg), f32),
        "patch_pos": ((d.pp, 2), f32),
        "patch_vel": ((d.pp, 2), f32),
        "key": ((2,), mx.uint32),
        "step": ((), i32),
        "next_uid": ((), i64),
        "total": ((), i64),
        "total0": ((), i64),
        "ledger_cum": ((N_LEDGER,), i64),
        "diag": ((), i32),
    }


def test_field_order() -> None:
    assert State._fields[:6] == ("alive", "uid", "parent_uid", "clade", "generation", "birth_step")
    assert State._fields[-7:] == ("key", "step", "next_uid", "total", "total0", "ledger_cum", "diag")
    assert len(State._fields) == 38
    assert AGENT_FIELDS[-1] == "W2" and len(AGENT_FIELDS) == 24


@pytest.mark.parametrize("preset", ["tiny", "dev", "64k", "256k"])
@pytest.mark.parametrize("overlays", [(), ("fast",)])
def test_schema_every_preset(preset: str, overlays: tuple[str, ...]) -> None:
    """Shapes and dtypes from the lazy graph (large presets are not evaluated)."""
    cfg = load_config(preset=preset, overlays=overlays)
    s = _initial_state(cfg)
    want = _expected_schema(cfg)
    assert set(want) == set(State._fields)
    for name, a in zip(State._fields, s, strict=True):
        assert (tuple(a.shape), a.dtype) == want[name], name


def _check_init_values(cfg: Config, s: State) -> None:
    d = dims(cfg)
    n, m, n0, L = d.n, d.m, cfg.population.n_init, cfg.world.size
    a = {k: to_numpy(v) for k, v in s._asdict().items()}
    fo, de = np.arange(n) < n0, np.arange(n) >= n0
    assert np.array_equal(a["alive"], fo)
    assert np.array_equal(a["uid"][fo], np.arange(n0)) and np.all(a["uid"][de] == -1)
    assert np.all(a["parent_uid"] == -1)
    assert np.array_equal(a["clade"][fo], np.arange(n0)) and np.all(a["clade"][de] == 0)
    assert np.all(a["generation"] == 0) and np.all(a["birth_step"] == -1)
    assert np.all(a["energy"][fo] == cfg.energy.e_init) and np.all(a["energy"][de] == 0)
    assert np.all(a["pos"] >= 0) and np.all(a["pos"] < L) and np.all(a["pos"][de] == 0)
    assert np.all(a["heading"] >= -np.pi) and np.all(a["heading"] < np.pi) and np.all(a["heading"][de] == 0)
    for k in ("vel", "h", "msg", "last_loss", "last_harvest", "last_inflow"):
        assert np.all(a[k] == 0), k
    assert np.all(a["partner"] == -1) and np.all(a["rev"] == -1)
    assert np.all(np.abs(a["tag"]) <= 1) and np.all(a["tag"][de] == 0) and np.any(a["tag"][fo] != 0)
    mu = cfg.mutation
    assert np.all(a["mut_sigma"] == np.float32(mu.sigma_init))
    assert np.all(a["mut_rate"] == np.float32(mu.rate_init))
    assert np.all(a["offspring_frac"] == np.float32(mu.frac_init))
    assert np.all(a["W1"][de] == 0) and np.all(a["W2"][de] == 0) and np.all(a["brain_cost"][de] == 0)
    assert np.all(np.abs(a["W1"]) <= cfg.brain.weight_clip) and np.any(a["W1"][fo] != 0)
    l1 = np.abs(a["W1"][fo].astype(np.float32)).sum((1, 2)) + np.abs(a["W2"][fo].astype(np.float32)).sum(
        (1, 2)
    )
    assert np.allclose(a["brain_cost"][fo], np.floor(cfg.energy.brain_cost_per_l1 * l1), atol=1)
    assert np.array_equal(a["pend_slot"], np.arange(n - m, n)) and np.all(~a["alive"][a["pend_slot"]])
    assert np.all(a["pend_W1"] == 0) and np.all(a["pend_W2"] == 0)
    assert np.all(a["resource"] >= 0) and np.all(a["resource"] <= cfg.fields.resource_cap)
    assert np.all(a["chem"] == 0)
    assert np.all(a["patch_pos"] >= 0) and np.all(a["patch_pos"] < L)
    assert np.allclose(np.linalg.norm(a["patch_vel"], axis=1), cfg.fields.patch_drift, rtol=1e-5)
    assert a["step"] == 0 and a["next_uid"] == n0 and a["diag"] == 0 and np.all(a["ledger_cum"] == 0)
    ref_total = (
        a["energy"].astype(np.int64).sum() + n0 * cfg.energy.structure + a["resource"].astype(np.int64).sum()
    )
    assert a["total"] == a["total0"] == ref_total


def test_init_values_tiny(tiny_cfg: Config) -> None:
    _check_init_values(tiny_cfg, init_state(tiny_cfg))


def test_init_values_micro_and_f16(micro_cfg: Config, tiny_cfg_f16: Config) -> None:
    _check_init_values(micro_cfg, init_state(micro_cfg))
    s = init_state(tiny_cfg_f16)
    assert s.W1.dtype == mx.float16
    _check_init_values(tiny_cfg_f16, s)


def test_init_deterministic_and_seeded(tiny_cfg: Config) -> None:
    a, b = init_state(tiny_cfg), init_state(tiny_cfg)
    assert state_digest(a) == state_digest(b)
    other = load_config(preset="tiny", overrides=["brain.weight_dtype=float32", "run.seed=1"])
    assert state_digest(init_state(other)) != state_digest(a)


def test_total_energy_exact_beyond_int32() -> None:
    n = 262_144
    alive = mx.ones((n,), dtype=mx.bool_)
    energy = mx.full((n,), 1_000_000 - 1, dtype=mx.int32)
    resource = mx.full((512, 512), R_HARD, dtype=mx.int32)
    got = total_energy(energy, alive, resource, Config())
    want = n * (1_000_000 - 1) + n * 10_000 + 512 * 512 * R_HARD
    assert got.dtype == mx.int64 and got.shape == () and int(np.array(got)) == want
    assert n * 10_000 > 2**31


def test_age() -> None:
    a = age(mx.array(5, dtype=mx.int32), mx.array([-1, 0, 4], dtype=mx.int32))
    assert a.dtype == mx.int32 and np.array(a).tolist() == [5, 4, 0]


def test_arrays_round_trip(micro_cfg: Config) -> None:
    s = init_state(micro_cfg)
    d = state_to_arrays(s)
    assert list(d) == list(State._fields)
    items = list(d.items())
    random.Random(0).shuffle(items)
    s2 = state_from_arrays(dict(items))
    assert state_digest(s2) == state_digest(s)
    assert all(x is y for x, y in zip(s, s2, strict=True))
    missing = dict(d)
    del missing["diag"]
    with pytest.raises(ValueError, match="diag"):
        state_from_arrays(missing)
    with pytest.raises(ValueError, match="bogus"):
        state_from_arrays({**d, "bogus": mx.array(0)})


def test_digest_sensitive(micro_cfg: Config) -> None:
    s = init_state(micro_cfg)
    base = state_digest(s)
    assert len(base) == 64 and base == state_digest(s)
    assert state_digest(s._replace(diag=mx.array(1, dtype=mx.int32))) != base
    assert state_digest(s._replace(energy=s.energy.astype(mx.int64))) != base  # dtype participates
    e = np.array(s.energy)
    e[3] += 1
    assert state_digest(s._replace(energy=mx.array(e))) != base


def test_to_numpy_bf16(tiny_cfg_bf16: Config) -> None:
    x = mx.array([1.5, -2.25, 3.0], dtype=mx.bfloat16)
    y = to_numpy(x)
    assert y.dtype == np.float32 and y.tolist() == [1.5, -2.25, 3.0]
    s = init_state(tiny_cfg_bf16)
    assert s.W1.dtype == mx.bfloat16
    assert len(state_digest(s)) == 64
    z = to_numpy(mx.array([1, 2], dtype=mx.int32))
    z[0] = 9  # an owned copy
    assert z.dtype == np.int32


def test_enums() -> None:
    assert N_LEDGER == 16 and [e.value for e in LedgerIdx] == list(range(16))
    assert LedgerIdx.FLOW_STRUCTURE == 14 and LedgerIdx.RESERVED == 15 and LedgerIdx.HEAT_OVERFLOW == 6
    assert N_COUNTERS == 26 and [c.value for c in CounterIdx] == list(range(26))
    assert CounterIdx.BOND_PROPOSERS == 25 and CounterIdx.V_CLIPS == 20 and CounterIdx.KIN_BOND_SUM == 23
    assert (DeathCause.STARVED, DeathCause.KILLED, DeathCause.AGED) == (0, 1, 2)
    assert Diag.LEDGER == 1 and Diag.POS_RANGE == 256 and Diag.INT_HEADROOM == 64
    assert Diag.CELL_OVERFLOW_WARN == 1 << 16 and Diag.VCLIP_WARN == 1 << 17
    errors = [f for f in Diag if f & ERROR_MASK]
    assert len(errors) == 9 and not (Diag.CELL_OVERFLOW_WARN & ERROR_MASK)
    assert R_HARD == (1 << 24) - 1


def test_exact_sum_used_for_totals(micro_cfg: Config) -> None:
    s = init_state(micro_cfg)
    t = total_energy(s.energy, s.alive, s.resource, micro_cfg)
    manual = ops.exact_sum(s.energy) + ops.exact_sum(s.resource) + 8 * micro_cfg.energy.structure
    assert int(np.array(t)) == int(np.array(manual)) == int(np.array(s.total))


def test_make_scene(micro_cfg: Config) -> None:
    d = dims(micro_cfg)
    w1 = np.full((d.w1_rows, d.h), 0.5, np.float32)
    s = make_scene(
        micro_cfg,
        agents=[
            {"pos": (1.0, 2.0), "energy": 1234, "heading": 0.5},
            {"pos": (3.0, 2.0), "W1": w1, "tag": (0.1, 0.2, 0.3, 0.4)},
            {"row": 10, "pos": (40.0, 40.0), "uid": 77, "brain_cost": 5},
        ],
        bonds=[(0, 1), (0, 2, 10, 3)],
        resource=100,
        step=3,
    )
    a = {k: to_numpy(v) for k, v in s._asdict().items()}
    assert np.flatnonzero(a["alive"]).tolist() == [0, 1, 10]
    assert a["uid"][[0, 1, 10]].tolist() == [0, 1, 77] and a["next_uid"] == 78
    assert a["energy"][[0, 1, 10]].tolist() == [1234, micro_cfg.energy.e_init, micro_cfg.energy.e_init]
    assert a["pos"][1].tolist() == [3.0, 2.0] and a["heading"][0] == np.float32(0.5)
    assert np.allclose(a["tag"][1], [0.1, 0.2, 0.3, 0.4])
    assert a["brain_cost"][[0, 1, 10]].tolist() == [0, int(0.1 * 0.5 * w1.size), 5]
    assert (a["partner"][0, 0], a["rev"][0, 0], a["partner"][1, 0], a["rev"][1, 0]) == (1, 0, 0, 0)
    assert (a["partner"][0, 2], a["rev"][0, 2], a["partner"][10, 3], a["rev"][10, 3]) == (10, 3, 0, 2)
    assert bool(np.array(bonds.symmetric_ok(s.partner, s.rev, s.alive)))
    assert np.all(a["resource"] == 100) and a["step"] == 3
    assert not a["alive"][a["pend_slot"]].any() and len(set(a["pend_slot"].tolist())) == d.m
    assert a["total"] == a["total0"] == a["energy"].sum() + 3 * micro_cfg.energy.structure + 100 * d.fg**2
    assert np.all(a["uid"][~a["alive"]] == -1) and np.all(a["energy"][~a["alive"]] == 0)
    for name, arr in zip(State._fields, s, strict=True):
        assert arr.dtype == getattr(init_state(micro_cfg), name).dtype, name
    with pytest.raises(KeyError):
        make_scene(micro_cfg, agents=[{"pos": (0.0, 0.0), "colour": 1}])
