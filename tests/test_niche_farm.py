"""Farming (DESIGN §41): the ledger and the soil identity close, the cost is in HEAT_ACTION, the bonus is
booked in INJECT_REGEN inside the resource cap, insolvency, the controls, the dithered put (E17) and decay."""

from __future__ import annotations

import dataclasses

import mlx.core as mx
import numpy as np
import pytest

import helpers
from alife import fields, invariants, niche, ops
from alife.config import dims, load_config
from alife.state import Diag, LedgerIdx, init_state, to_numpy, total_energy
from alife.step import make_step
from niche_helpers import cfg_for, farmers, np_soil, run

STEPS = 400
STEPS_SPHERE = 80  # an eager sphere step costs about 0.1 s


@pytest.mark.parametrize("nd", [2, 3, "s"])
def test_the_ledger_and_the_soil_identity_close_every_step(nd: int | str) -> None:
    cfg = cfg_for(nd)
    cap = cfg.niche.farm_cap
    peak = yielded = 0
    for t, (prev, s, out, row) in enumerate(run(cfg, farmers(cfg), STEPS_SPHERE if nd == "s" else STEPS)):
        led = to_numpy(out.ledger)
        assert helpers.ledger_net(led, cfg) == int(to_numpy(s.total) - to_numpy(prev.total)), t
        assert led[LedgerIdx.FLOW_STOLEN] == led[LedgerIdx.FLOW_ATTACK_GAIN] + led[LedgerIdx.FLOW_SPILL]
        f0, f1 = np_soil(prev), np_soil(s)
        assert f1.sum() == f0.sum() + row["SOIL_PUT"] - row["SOIL_DECAYED"], t
        assert row["SOIL_TOTAL"] == f1.sum(), t
        assert f1.min() >= 0 and f1.max() <= cap, t
        assert row["FARM_YIELD"] >= 0 and row["SOIL_WASTED"] >= 0, t
        assert int(to_numpy(s.diag)) & 0xFFFF == 0, (t, Diag(int(to_numpy(s.diag))))
        if t % 20 == 0:
            assert invariants.full_check(s, cfg, prev, out) == [], t
        peak, yielded = max(peak, row["SOIL_TOTAL"]), yielded + row["FARM_YIELD"]
    assert peak > 0 and yielded > 0  # soil appeared and it paid


def test_a_full_cell_refuses_units_and_the_identity_still_closes() -> None:
    """Zero-weight farmers stay in their cells and fill them: with a tiny cap the units beyond it are
    ``SOIL_WASTED`` (paid for, not accepted) and ``units paid = SOIL_PUT + SOIL_WASTED``."""
    cfg = cfg_for(2, "niche.farm_cap=64")
    wasted = 0
    for t, (prev, s, out, row) in enumerate(run(cfg, farmers(cfg, zero_weights=True), 60)):
        f0, f1 = np_soil(prev), np_soil(s)
        assert f1.sum() == f0.sum() + row["SOIL_PUT"] - row["SOIL_DECAYED"], t
        assert f1.max() <= 64, t
        led = to_numpy(out.ledger)
        assert helpers.ledger_net(led, cfg) == int(to_numpy(s.total) - to_numpy(prev.total)), t
        wasted += row["SOIL_WASTED"]
    assert wasted > 0


@pytest.mark.parametrize("nd", [2, 3, "s"])
def test_the_put_cost_is_booked_in_heat_action(nd: int | str) -> None:
    """The same state, one step, ``farm_cost`` 30 against 0: the actions are the same (the brain does not see
    the cost), so ``HEAT_ACTION`` differs by exactly ``30·(SOIL_PUT + SOIL_WASTED)`` when all are solvent."""
    on, off = cfg_for(nd), cfg_for(nd, "niche.farm_cost=0")
    s = farmers(on)
    (_, _, o_on, r_on), (_, _, o_off, r_off) = next(run(on, s, 1)), next(run(off, s, 1))
    paid = r_on["SOIL_PUT"] + r_on["SOIL_WASTED"]
    assert paid == r_off["SOIL_PUT"] + r_off["SOIL_WASTED"] > 0
    d = to_numpy(o_on.ledger)[LedgerIdx.HEAT_ACTION] - to_numpy(o_off.ledger)[LedgerIdx.HEAT_ACTION]
    assert d == 30 * paid
    for i in LedgerIdx:  # every other term is unchanged by the cost (it only moves energy to heat)
        if i != LedgerIdx.HEAT_ACTION:
            assert to_numpy(o_on.ledger)[i] == to_numpy(o_off.ledger)[i], i


def test_an_insolvent_farmer_pays_what_it_holds_and_puts_nothing() -> None:
    cfg = cfg_for(2)
    s = farmers(cfg, zero_weights=True)
    d = dims(cfg)
    alive = to_numpy(s.alive)
    e = to_numpy(s.energy).copy()
    poor = alive & (np.arange(d.n) % 2 == 0)
    e[poor] = 3  # below the basal cost: insolvent, dies at S14
    e = mx.array(e)
    tot = total_energy(e, s.alive, s.resource, cfg)
    s = s._replace(energy=e, total=tot, total0=tot)
    mx.eval(s)
    base = next(run(cfg, farmers(cfg, zero_weights=True), 1))[3]
    _, s1, out, row = next(run(cfg, s, 1))
    assert 0 < row["FARMERS"] <= int((alive & ~poor).sum())
    assert row["SOIL_PUT"] < base["SOIL_PUT"]
    led = to_numpy(out.ledger)
    assert helpers.ledger_net(led, cfg) == int(to_numpy(s1.total) - to_numpy(s.total))
    assert invariants.full_check(s1, cfg, s, out) == []


def test_the_ledger_has_no_new_term_with_farming_on() -> None:
    cfg = cfg_for(2)
    s, out = make_step(cfg)(init_state(cfg))
    assert out.ledger.shape == (16,) and s.ledger_cum.shape == (16,)


# --- the bonus ----------------------------------------------------------------------------------------------


def test_the_bonus_is_monotone_bounded_and_zero_without_soil() -> None:
    cfg = cfg_for(2)
    soil = mx.arange(0, cfg.niche.farm_cap + 1, dtype=mx.int32)
    b = np.asarray(niche.bonus_of(soil, cfg))
    assert b[0] == 0 and (np.diff(b) >= 0).all()
    assert b[-1] <= cfg.niche.farm_bonus_frac * cfg.fields.regen_max + 1
    assert niche.bonus_q16(cfg) == round(0.5 * 320 / 4095 * 65536)
    ab = cfg_for(2, "niche.control=ablate")
    assert not np.asarray(niche.bonus_of(soil, ab)).any()


@pytest.mark.parametrize("nd", [2, 3, "s"])
def test_update_resource_farm_with_no_bonus_is_update_resource(nd: int | str) -> None:
    cfg = cfg_for(nd)
    s = init_state(cfg)
    zero = mx.zeros_like(s.resource)
    for step in (0, 7):
        st = mx.array(step, dtype=mx.int32)
        r0, i0 = fields.update_resource(s.resource, s.patch_pos, cfg, st)
        r1, i1, j1 = fields.update_resource_farm(s.resource, s.patch_pos, zero, cfg, st)
        assert np.array_equal(np.asarray(r0), np.asarray(r1))
        assert np.array_equal(np.asarray(i0), np.asarray(i1)) and np.array_equal(
            np.asarray(i0), np.asarray(j1)
        )


@pytest.mark.parametrize("nd", [2, 3, "s"])
def test_the_resource_cap_binds_whatever_the_bonus(nd: int | str) -> None:
    cfg = cfg_for(nd)
    cap = cfg.fields.resource_cap
    rs = np.random.default_rng(2)
    s = init_state(cfg)
    shape = tuple(s.resource.shape)
    res = mx.array(rs.integers(0, cap + 1, shape).astype(np.int32))
    huge = mx.array(rs.integers(0, 10 * cap, shape).astype(np.int32))
    r1, inc, inc0 = fields.update_resource_farm(res, s.patch_pos, huge, cfg, mx.array(3, dtype=mx.int32))
    inc, inc0, res_np = np.asarray(inc), np.asarray(inc0), np.asarray(res)
    assert (inc >= 0).all() and (inc <= cap - res_np).all() and (inc >= inc0).all()
    assert int(np.asarray(r1).sum()) == int(res_np.sum()) + int(inc.sum())  # diffusion conserves the total


@pytest.mark.parametrize("nd", [2, "s"])
def test_the_bonus_is_booked_in_inject_regen_and_equals_the_yield(nd: int | str) -> None:
    """A state with soil, one step with the bonus against the ``ablate`` control (same actions, same costs,
    same harvest: the bonus acts only at S17): INJECT_REGEN differs by exactly FARM_YIELD."""
    cfg, ab = cfg_for(nd), cfg_for(nd, "niche.control=ablate")
    s = farmers(cfg)
    soil = np.zeros(tuple(s.soil.shape), np.int32)
    soil[::2, ::3] = 3000
    s = s._replace(soil=mx.array(soil))
    (_, _, o_on, r_on), (_, _, o_ab, r_ab) = next(run(cfg, s, 1)), next(run(ab, s, 1))
    assert r_ab["FARM_YIELD"] == 0 and r_on["FARM_YIELD"] > 0
    diff = to_numpy(o_on.ledger)[LedgerIdx.INJECT_REGEN] - to_numpy(o_ab.ledger)[LedgerIdx.INJECT_REGEN]
    assert diff == r_on["FARM_YIELD"]
    assert r_on["SOIL_PUT"] == r_ab["SOIL_PUT"]  # the control builds the same soil


@pytest.mark.parametrize("nd", [2, "s"])
def test_the_bonus_acts_from_the_next_step(nd: int | str) -> None:
    """The soil written at a step is read for the bonus of the next one (the start-of-step soil)."""
    cfg = cfg_for(nd)
    s = init_state(cfg)  # nobody farms (silent) and the soil is empty
    assert not np_soil(s).any()
    soil = np.zeros(tuple(s.soil.shape), np.int32)
    soil[:] = 1000
    s2 = s._replace(soil=mx.array(soil))
    assert next(run(cfg, s2, 1))[3]["FARM_YIELD"] > 0
    assert next(run(cfg, s, 1))[3]["FARM_YIELD"] == 0


# --- controls -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("nd", [2, "s"])
def test_mute_puts_nothing_and_ablate_pays_for_a_useless_soil(nd: int | str) -> None:
    mute, ab = cfg_for(nd, "niche.control=mute"), cfg_for(nd, "niche.control=ablate")
    r = next(run(mute, farmers(mute), 3))[3]
    assert r["FARMERS"] == 0 and r["SOIL_PUT"] == 0 and r["FARM_YIELD"] == 0
    runs = list(run(ab, farmers(ab), 30))
    assert sum(r[3]["FARM_YIELD"] for r in runs) == 0 and np_soil(runs[-1][1]).sum() > 0


# --- E17: the dithered put ----------------------------------------------------------------------------------


def _y(cfg, p: float) -> mx.array:
    """A brain output whose farm_put column decodes to the put ``p`` (all other columns 0)."""
    d = dims(cfg)
    y = np.zeros((20000, d.lay.D_OUT), np.float32)
    y[:, d.lay.OUT_FARM_PUT] = np.arctanh(np.float32(p)) if p < 1.0 else 20.0
    return mx.array(y)


def test_e17_a_sub_unit_put_buys_a_unit_with_the_matching_probability() -> None:
    cfg = cfg_for(2)
    rate = cfg.niche.farm_rate
    uid = mx.arange(20000, dtype=mx.int32)
    for p in (0.01, 0.03, 0.0625, 0.1, 0.3, 0.77):
        got = []
        for step in range(20):
            u = np.asarray(niche.units_of(_y(cfg, p), uid, mx.array(step, dtype=mx.int32), cfg))
            got.append(u.mean())
        mean = float(np.mean(got))
        assert abs(mean - p * rate) < 0.01 + 0.02 * p * rate, (p, mean, p * rate)  # unbiased
    u0 = np.asarray(niche.units_of(_y(cfg, 0.0), uid, mx.array(5, dtype=mx.int32), cfg))
    assert not u0.any()  # silence is free
    u1 = np.asarray(niche.units_of(_y(cfg, 1.0), uid, mx.array(5, dtype=mx.int32), cfg))
    assert (u1 == rate).all()  # full output is exactly farm_rate
    neg = np.asarray(niche.units_of(-_y(cfg, 0.5), uid, mx.array(5, dtype=mx.int32), cfg))
    assert not neg.any()


def test_e17_the_cost_of_sub_unit_puts_is_exact_on_average() -> None:
    """Zero-weight farmers with a small output (bias 0.02, put about 0.02): the units paid per row per step
    average ``rate·put`` and the ``HEAT_ACTION`` of the farm part is ``farm_cost`` times the paid units."""
    cfg = cfg_for(2)
    on, off = cfg, cfg_for(2, "niche.farm_cost=0")
    bias = 0.02
    put = float(np.tanh(np.float32(bias)))
    tot_units, tot_alive, d_heat = 0, 0, 0
    start_on, start_off = farmers(on, bias, zero_weights=True), farmers(off, bias, zero_weights=True)
    for (_, s_on, o_on, r_on), (_, _, o_off, _) in zip(
        run(on, start_on, 80), run(off, start_off, 80), strict=True
    ):
        tot_units += r_on["SOIL_PUT"] + r_on["SOIL_WASTED"]
        tot_alive += int(to_numpy(s_on.alive).sum())
        d_heat += int(
            to_numpy(o_on.ledger)[LedgerIdx.HEAT_ACTION] - to_numpy(o_off.ledger)[LedgerIdx.HEAT_ACTION]
        )
    assert tot_units > 0
    assert abs(tot_units / tot_alive - put * cfg.niche.farm_rate) < 0.05 * put * cfg.niche.farm_rate
    # the farm cost equals farm_cost x the units paid (the two trajectories are the same while the states are)
    assert d_heat == 30 * tot_units


# --- the dithered decay (E17 spirit, E18) -------------------------------------------------------------------


def test_the_dithered_decay_is_unbiased_where_the_floor_and_ceil_are_not() -> None:
    n, q16, cap = 20000, 16, 4095
    cells = mx.arange(n, dtype=mx.int32)
    x = mx.full((n,), cap, dtype=mx.int32)
    assert int(((x * q16) >> 16).sum()) == 0  # a plain floor decay never touches 4095 at q16 = 16
    for steps in (256, 1024):
        x = mx.full((n,), cap, dtype=mx.int32)
        for t in range(steps):
            x = x - ops.dithered_decay(x, q16, cells, mx.array(t, dtype=mx.int32), niche.DECAY_SALT)
        want = cap * np.exp(-q16 * steps / 65536)
        assert abs(float(np.asarray(x).mean()) - want) < 0.01 * cap, (
            steps,
            float(np.asarray(x).mean()),
            want,
        )
    big = mx.arange(0, cap + 1, dtype=mx.int32)
    for q in (1, 16, 33, 40000, 65535):
        for t in (0, 1, 99):
            loss = np.asarray(ops.dithered_decay(big, q, big, mx.array(t, dtype=mx.int32), niche.DECAY_SALT))
            assert (loss >= 0).all() and (loss <= np.arange(cap + 1)).all() and loss[0] == 0, (q, t)


def test_the_decay_is_deterministic_and_depends_on_the_step() -> None:
    x = mx.full((1000,), 500, dtype=mx.int32)
    c = mx.arange(1000, dtype=mx.int32)
    a = np.asarray(ops.dithered_decay(x, 100, c, mx.array(4, dtype=mx.int32), niche.DECAY_SALT))
    b = np.asarray(ops.dithered_decay(x, 100, c, mx.array(4, dtype=mx.int32), niche.DECAY_SALT))
    c2 = np.asarray(ops.dithered_decay(x, 100, c, mx.array(5, dtype=mx.int32), niche.DECAY_SALT))
    assert np.array_equal(a, b) and not np.array_equal(a, c2)


def test_e18_the_put_and_decay_dithers_are_independent_streams() -> None:
    from test_ext_salts import independence_report

    n = 20000
    ids = mx.arange(n, dtype=mx.int32)
    step = mx.array(11, dtype=mx.int32)
    put = np.asarray(ops.hash32(ids, step, niche.SALT))
    dec = np.asarray(ops.hash32(ids, step, niche.DECAY_SALT))
    rep = independence_report(put, dec)
    assert abs(rep["z"]) < 4.5 and rep["p_mi"] > 0.01 and rep["bit_dev"] == 0 and rep["bit_rot"] <= 2, rep
    assert niche.SALT != niche.DECAY_SALT
    # farming consumes no RNG stream: the layout version is the same as a farming-off world's
    from alife import rng

    assert rng.layout_version(cfg_for(2)) == rng.layout_version(load_config(preset="tiny"))


# --- the soil is a state field, not energy ------------------------------------------------------------------


def test_soil_is_a_nonconserved_field_on_the_food_lattice() -> None:
    from alife import ext

    cfg = cfg_for(3)
    spec = next(sp for sp in ext.REGISTRY if sp.name == "niche")
    (f,) = spec.state_fields(cfg, 3)
    assert f.name == "soil" and not f.per_agent and not f.conserved and f.dtype == mx.int32
    assert tuple(init_state(cfg).soil.shape) == dims(cfg).field_shape
    assert not np.asarray(init_state(cfg).soil).any()


def test_the_config_replace_helper_changes_only_one_key() -> None:  # guards the helpers of this module
    a = cfg_for(2)
    b = dataclasses.replace(a, niche=dataclasses.replace(a.niche, farm_cost=31))
    assert a.niche.farm_cost == 30 and b.niche.farm_cost == 31


# --- invariants ---------------------------------------------------------------------------------------------


def test_the_invariants_check_the_soil_range_on_the_host_and_the_device() -> None:
    cfg = cfg_for(2)
    s = init_state(cfg)
    assert invariants.full_check(s, cfg) == []
    for bad_value in (-1, cfg.niche.farm_cap + 1):
        soil = np.array(s.soil)
        soil[0, 0] = bad_value
        bad = s._replace(soil=mx.array(soil))
        assert any("soil: values outside" in p for p in invariants.full_check(bad, cfg)), bad_value
    # the device check: a step from a corrupt (negative; a high one is clipped by the update) soil flags
    # FIELD_RANGE
    soil = np.array(s.soil)
    soil[1, 1] = -50
    s1, _ = make_step(cfg)(s._replace(soil=mx.array(soil)))
    mx.eval(s1)
    assert int(to_numpy(s1.diag)) & Diag.FIELD_RANGE
