"""Built structures and the energy ledger (DESIGN §38.7): material holds no energy, so the 16-term ledger
closes with structures on and the build cost is booked in ``HEAT_ACTION``; the material has its own exact
identity ``Σmat' = Σmat + BUILD_UNITS - MAT_WASTED - MAT_DECAYED``."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import helpers
from alife import invariants
from alife.config import dims
from alife.state import Diag, LedgerIdx, init_state, to_numpy, total_energy
from alife.step import make_step
from structures_helpers import builders, cfg_for, np_mat, run, units_for

STEPS = 1500


@pytest.mark.parametrize("nd", [2, 3])
def test_the_ledger_and_the_material_identity_close_every_step(nd: int) -> None:
    cfg = cfg_for(nd)
    cap = cfg.structures.cap
    peak = 0
    for t, (prev, s, out, row) in enumerate(run(cfg, builders(cfg), STEPS)):
        led = to_numpy(out.ledger)
        assert helpers.ledger_net(led, cfg) == int(to_numpy(s.total) - to_numpy(prev.total)), t
        assert led[LedgerIdx.FLOW_STOLEN] == led[LedgerIdx.FLOW_ATTACK_GAIN] + led[LedgerIdx.FLOW_SPILL]
        m0, m1 = np_mat(prev), np_mat(s)
        assert m1.sum() == m0.sum() + row["BUILD_UNITS"] - row["MAT_WASTED"] - row["MAT_DECAYED"], t
        assert row["MAT_TOTAL"] == m1.sum() and row["MAT_CELLS"] == (m1 > 0).sum(), t
        assert m1.min() >= 0 and m1.max() <= cap, t
        assert row["APERTURE_VIOLATIONS"] == 0 and row["TRAPPED"] == 0, t
        assert int(to_numpy(s.diag)) & 0xFFFF == 0, (t, Diag(int(to_numpy(s.diag))))
        if t % 100 == 0:
            assert invariants.full_check(s, cfg, prev, out) == [], t
        peak = max(peak, row["MAT_TOTAL"])
    assert peak > 0  # material appeared


@pytest.mark.parametrize("nd", [2, 3])
def test_the_build_cost_is_booked_in_heat_action(nd: int) -> None:
    """The same state, one step, ``build_cost`` 2 against 0: the actions are the same (the brain does not see
    the cost), so ``HEAT_ACTION`` differs by exactly ``2·BUILD_UNITS`` when everyone is solvent."""
    on, off = cfg_for(nd), cfg_for(nd, "structures.build_cost=0")
    s = builders(on)
    (_, _, o_on, r_on), (_, _, o_off, r_off) = next(run(on, s, 1)), next(run(off, s, 1))
    assert r_on["BUILD_UNITS"] == r_off["BUILD_UNITS"] > 0
    d = to_numpy(o_on.ledger)[LedgerIdx.HEAT_ACTION] - to_numpy(o_off.ledger)[LedgerIdx.HEAT_ACTION]
    assert d == 2 * r_on["BUILD_UNITS"]
    # every other term of the ledger is unchanged by the cost (it only moves energy to heat)
    for i in LedgerIdx:
        if i != LedgerIdx.HEAT_ACTION:
            assert to_numpy(o_on.ledger)[i] == to_numpy(o_off.ledger)[i], i


def test_an_insolvent_builder_pays_what_it_holds_and_deposits_nothing() -> None:
    cfg = cfg_for(2)
    s = builders(cfg, zero_weights=True)
    d = dims(cfg)
    alive = to_numpy(s.alive)
    e = to_numpy(s.energy).copy()
    poor = alive & (np.arange(d.n) % 2 == 0)
    e[poor] = 3  # below the basal cost: insolvent, dies at S14
    e = mx.array(e)
    tot = total_energy(e, s.alive, s.resource, cfg)
    s = s._replace(energy=e, total=tot, total0=tot)
    mx.eval(s)
    _, s1, out, row = next(run(cfg, s, 1))
    assert row["BUILD_UNITS"] == units_for(cfg) * int((alive & ~poor).sum())
    led = to_numpy(out.ledger)
    assert helpers.ledger_net(led, cfg) == int(to_numpy(s1.total) - to_numpy(s.total))
    assert invariants.full_check(s1, cfg, s, out) == []


def test_the_ledger_has_no_new_term_with_structures_on() -> None:
    cfg = cfg_for(2)
    s, out = make_step(cfg)(init_state(cfg))
    assert out.ledger.shape == (16,) and s.ledger_cum.shape == (16,)
