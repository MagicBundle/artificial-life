"""The ledger policy of the extension mechanism (DESIGN §43.4): no 17th term, and an extension action due
goes through ``pay_costs`` into ``HEAT_ACTION`` so the ledger closes."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import ext_stub as stub
from alife import energy as en
from alife import invariants
from alife.config import dims, load_config
from alife.state import N_LEDGER, Diag, LedgerIdx, init_state, ledger_names, to_numpy
from alife.step import make_step

STEPS = 120


@pytest.fixture(autouse=True)
def _registry(monkeypatch: pytest.MonkeyPatch) -> None:
    stub.install(monkeypatch)


def test_no_17th_term() -> None:
    """A later addition must be a deliberate, reviewed change (it would also change every checkpoint)."""
    assert N_LEDGER == 16 == len(LedgerIdx) == len(ledger_names(2)) == len(ledger_names(3))
    assert LedgerIdx.RESERVED == 15
    cfg = load_config(preset="tiny", overrides=["brain.weight_dtype=float32"])
    s, out = make_step(cfg)(init_state(cfg))
    assert out.ledger.shape == (16,) and s.ledger_cum.shape == (16,)


def _extra_due_costs(cfg, extra):
    d = dims(cfg)
    e = mx.full((d.n,), 1000, mx.int32)
    from test_energy import _actions  # the all-zero actions helper of the unit tests

    act = _actions(d.n)
    alive = mx.ones((d.n,), mx.bool_)
    c0 = en.pay_costs(e, act, mx.zeros((d.n,), mx.int32), mx.zeros((d.n,), mx.int32), alive, cfg)
    c1 = en.pay_costs(e, act, mx.zeros((d.n,), mx.int32), mx.zeros((d.n,), mx.int32), alive, cfg, extra)
    return c0, c1


def test_extra_due_joins_the_action_due_in_pay_costs() -> None:
    cfg = load_config(preset="tiny")
    extra = mx.arange(dims(cfg).n, dtype=mx.int32) % 7
    c0, c1 = _extra_due_costs(cfg, extra)
    assert np.array_equal(np.array(c1.paid_action - c0.paid_action), np.array(extra))
    assert np.array_equal(np.array(c1.paid_basal), np.array(c0.paid_basal))
    assert np.array_equal(np.array(c1.energy), np.array(c0.energy - extra))
    # a due above the energy is paid up to the energy and the agent is insolvent
    big = mx.full((dims(cfg).n,), 5000, mx.int32)
    _, c2 = _extra_due_costs(cfg, big)
    assert not np.any(np.array(c2.solvent)) and not np.any(np.array(c2.energy))


@pytest.mark.parametrize("nd", [2, 3])
def test_the_ledger_closes_with_a_stub_due_and_every_stub_on(nd: int) -> None:
    base = load_config(preset="tiny" if nd == 2 else "tiny3d", overrides=["brain.weight_dtype=float32"])
    cfg_on, cfg_off = stub.stub_cfg(base, *stub.WITHOUT_ARMS), base
    fn = make_step(cfg_on)
    s = init_state(cfg_on)
    prev, heat = s, 0
    for t in range(STEPS):
        s, out = fn(s)
        mx.eval(s, out)
        assert int(to_numpy(s.diag)) & int(Diag.LEDGER) == 0 and int(to_numpy(s.diag)) == 0, f"step {t}"
        heat += int(to_numpy(out.ledger)[LedgerIdx.HEAT_ACTION])
        if t % 30 == 0:
            assert invariants.full_check(s, cfg_on, prev, out) == [], f"step {t}"
        prev = s
    ledger_cum = to_numpy(s.ledger_cum)
    assert heat == int(ledger_cum[LedgerIdx.HEAT_ACTION]) > 0
    # the comm-style stub pays 2 quanta per living agent-step through HEAT_ACTION, on top of the legacy action
    # costs: the first step of the founders books at least 2 per founder
    _, first = make_step(cfg_on)(init_state(cfg_on))
    assert int(to_numpy(first.ledger)[LedgerIdx.HEAT_ACTION]) >= 2 * cfg_on.population.n_init
    assert cfg_off.population.n_init == cfg_on.population.n_init
