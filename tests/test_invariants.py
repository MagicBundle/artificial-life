"""§22.2–§22.3: ``device_check`` and ``full_check`` on healthy states, and deliberate corruption of each
invariant.

Healthy end-of-step states come from a lifecycle-only pseudo-step (S0 flush, S14 deaths, S16 births and the
matching ledger), so the relational checks (``prev``/``out``) are exercised without the full step.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from alife import invariants, lifecycle
from alife.config import Config, apply_overrides, dims
from alife.rng import Stream, step_keys
from alife.state import N_COUNTERS, R_HARD, Diag, LedgerIdx, State, age, init_state, to_numpy, total_energy
from alife.step import StepOut
from helpers import make_scene


def _np(a: mx.array) -> np.ndarray:
    return to_numpy(a)


def _pseudo_step(s: State, cfg: Config, kill: np.ndarray) -> tuple[State, StepOut, mx.array]:
    """S0 + S14 + S16 + bookkeeping: rows in ``kill`` die insolvent; eligible survivors request a birth."""
    d, S = dims(cfg), cfg.energy.structure
    W1, W2 = lifecycle.flush_pending(s.W1, s.W2, s.pend_slot, s.pend_W1, s.pend_W2)
    s = s._replace(W1=W1, W2=W2)
    _, ks = step_keys(s.key)
    a = age(s.step, s.birth_step)
    fcell = mx.array((np.arange(d.n) * 7 % (d.fg * d.fg)).astype(np.int32))
    solvent = s.alive & ~mx.array(kill)
    dt = lifecycle.deaths(
        s.alive, solvent, s.energy, mx.zeros_like(s.energy), a, s.uid, s.generation, s.clade, s.partner,
        s.rev, s.resource, fcell, cfg,
    )  # fmt: skip
    sv = dt.survivor
    s1 = s._replace(
        alive=sv, energy=dt.energy, uid=dt.uid, partner=dt.partner, rev=dt.rev, resource=dt.resource,
        vel=mx.where(sv[:, None], s.vel, 0.0), h=mx.where(sv[:, None], s.h, 0.0),
        msg=mx.where(sv[:, None], s.msg, 0.0), last_loss=mx.where(sv, s.last_loss, 0),
        last_harvest=mx.where(sv, s.last_harvest, 0), last_inflow=mx.where(sv, s.last_inflow, 0),
    )  # fmt: skip
    req = sv & (dt.energy >= cfg.repro.min_energy) & (a >= cfg.repro.min_age)
    lanes = lifecycle.allocate_lanes(req, sv, ks[Stream.BIRTH_PRIO], cfg)
    rp = lifecycle.reproduce(s1, lanes, sv, ks, cfg)
    n_birth, n_imm = int(_np(lanes.n_birth)), int(_np(lanes.n_imm))
    ledger = np.zeros(len(LedgerIdx), np.int64)
    ledger[LedgerIdx.INJECT_IMMIGRATION] = n_imm * (cfg.energy.e_init + S)
    ledger[LedgerIdx.HEAT_BIRTH] = n_birth * cfg.repro.birth_cost
    ledger[LedgerIdx.HEAT_OVERFLOW] = _np(dt.r_overflow).astype(np.int64).sum()
    ledger[LedgerIdx.FLOW_CORPSE] = _np(dt.corpse).astype(np.int64).sum()
    ledger[LedgerIdx.FLOW_ENDOWMENT] = _np(rp.endow).astype(np.int64).sum()
    ledger[LedgerIdx.FLOW_STRUCTURE] = n_birth * S
    led = mx.array(ledger)
    s2 = rp.state
    s2 = s2._replace(
        step=s.step + 1,
        key=step_keys(s.key)[0],
        total=total_energy(s2.energy, s2.alive, s2.resource, cfg),
        ledger_cum=s.ledger_cum + led,
    )
    out = StepOut(
        counters=mx.zeros((N_COUNTERS,), dtype=mx.int32), ledger=led, births=rp.records, deaths=dt.records
    )
    mx.eval(s2, out)
    return s2, out, led


def _scene(cfg: Config) -> State:
    rs = np.random.default_rng(11)
    L = cfg.world.size
    agents = [
        dict(
            pos=rs.uniform(0, L, 2),
            heading=rs.uniform(-3, 3),
            energy=int(rs.integers(20_000, 120_000)),
            birth_step=-30,
            tag=rs.uniform(-1, 1, 4),
            h=rs.uniform(-1, 1, dims(cfg).h),
            msg=rs.uniform(-1, 1, 2),
            vel=(0.1, -0.2),
            last_harvest=7,
        )
        for _ in range(24)
    ]
    return make_scene(cfg, agents, bonds=[(0, 1), (1, 2), (2, 3), (5, 6), (7, 8), (0, 9)], step=5)


def _healthy(cfg: Config, steps: int = 4) -> list[tuple[State, State, StepOut, mx.array]]:
    """(prev, state, out, ledger) for each pseudo-step from the scene."""
    s = _scene(cfg)
    rs = np.random.default_rng(12)
    hist = []
    for _ in range(steps):
        alive = _np(s.alive)
        kill = alive & (rs.uniform(size=alive.size) < 0.15)
        s1, out, led = _pseudo_step(s, cfg, kill)
        hist.append((s, s1, out, led))
        s = s1
    return hist


def _device(s: State, cfg: Config, ledger: mx.array, prev_total: mx.array, **kw: int) -> int:
    args = {k: mx.array(kw.get(k, 0), dtype=mx.int32) for k in ("max_cell_occ", "max_attackers",
                                                                  "overflow_cells", "v_clips")}  # fmt: skip
    return int(_np(invariants.device_check(s, ledger, prev_total, cfg=cfg, **args)))


@pytest.fixture(scope="module")
def history(micro_cfg: Config) -> list[tuple[State, State, StepOut, mx.array]]:
    return _healthy(micro_cfg)


# --- healthy states ----------------------------------------------------------------------------------------


def test_init_states_are_clean(tiny_cfg: Config, tiny_cfg_bf16: Config, micro_cfg: Config) -> None:
    for cfg in (tiny_cfg, tiny_cfg_bf16, micro_cfg):
        s = init_state(cfg)
        assert invariants.full_check(s, cfg) == []
        zero = mx.zeros((len(LedgerIdx),), dtype=mx.int64)
        assert _device(s, cfg, zero, s.total) == 0
        assert invariants.device_check(s, zero, s.total, *(mx.array(0),) * 4, cfg).dtype == mx.int32


def test_pseudo_steps_are_clean(micro_cfg: Config, history) -> None:
    births = deaths = 0
    for prev, s, out, led in history:
        assert invariants.full_check(s, micro_cfg, prev, out) == []
        assert _device(s, micro_cfg, led, prev.total) == 0
        births += int(_np(out.births.valid).sum())
        deaths += int(_np(out.deaths.valid).sum())
    assert births > 0 and deaths > 0  # the history is not vacuous
    assert int(_np(history[-1][1].step)) >= 6  # the step ≥ 1 pending clause is live


def test_immigration_pseudo_steps_are_clean(micro_cfg: Config) -> None:
    cfg = apply_overrides(micro_cfg, ["population.immigration_floor=50", "repro.min_energy=100000"])
    hist = _healthy(cfg, steps=2)
    imm = sum(int((_np(o.births.parent_uid)[_np(o.births.valid)] == -1).sum()) for _, _, o, _ in hist)
    assert imm > 0
    for prev, s, out, led in hist:
        assert invariants.full_check(s, cfg, prev, out) == []
        assert _device(s, cfg, led, prev.total) == 0


def test_device_check_flags_and_warnings(micro_cfg: Config, history) -> None:
    prev, s, _, led = history[-1]
    d = dims(micro_cfg)
    assert _device(s, micro_cfg, led, prev.total, max_cell_occ=d.occ_limit) == Diag.INT_HEADROOM
    assert _device(s, micro_cfg, led, prev.total, max_cell_occ=d.occ_limit - 1) == 0
    assert _device(s, micro_cfg, led, prev.total, max_attackers=1 << 15) == Diag.INT_HEADROOM
    assert _device(s, micro_cfg, led, prev.total, overflow_cells=3) == Diag.CELL_OVERFLOW_WARN
    assert _device(s, micro_cfg, led, prev.total, v_clips=1) == Diag.VCLIP_WARN
    theft = _np(led).copy()
    theft[LedgerIdx.FLOW_STOLEN], theft[LedgerIdx.FLOW_ATTACK_GAIN], theft[LedgerIdx.FLOW_SPILL] = 10, 7, 2
    assert _device(s, micro_cfg, mx.array(theft), prev.total) == Diag.LEDGER
    theft[LedgerIdx.FLOW_SPILL] = 3
    assert _device(s, micro_cfg, mx.array(theft), prev.total) == 0
    off = _np(led).copy()
    off[LedgerIdx.HEAT_BASAL] += 1
    assert _device(s, micro_cfg, mx.array(off), prev.total) == Diag.LEDGER


# --- corruptions ------------------------------------------------------------------------------------------


def _set(s: State, name: str, fn: Callable[[np.ndarray], Any]) -> State:
    a = _np(getattr(s, name)).copy()
    fn(a)
    return s._replace(**{name: mx.array(a).astype(getattr(s, name).dtype)})


def _rows(s: State) -> tuple[int, int, int, int]:
    """(an alive row, a dead row, a bonded alive row, its first occupied slot)."""
    alive, partner = _np(s.alive), _np(s.partner)
    bonded = np.flatnonzero(alive & (partner >= 0).any(1))[0]
    return np.flatnonzero(alive)[0], np.flatnonzero(~alive)[0], bonded, int(np.argmax(partner[bonded] >= 0))


def _corrupt_rev(s: State) -> State:
    _, _, i, b = _rows(s)
    return _set(s, "rev", lambda a: a.__setitem__((i, b), (a[i, b] + 1) % 4))


def _dead_bond(s: State) -> State:
    i, j, _, _ = _rows(s)
    s = _set(s, "partner", lambda a: a.__setitem__((j, 0), i))
    return _set(s, "rev", lambda a: a.__setitem__((j, 0), 0))


def _over_capacity(s: State) -> State:
    d_alive = _np(s.alive)
    need = (d_alive.size - 8 + 1) - int(d_alive.sum())  # micro: N_CAP = N − M = 56
    rows = np.flatnonzero(~d_alive)[:need]
    return _set(s, "alive", lambda a: a.__setitem__(rows, True))


def _at(name: str, which: int, value: Any, *, col: Any = None) -> Callable[[State], State]:
    """Set ``name[row]`` (``which``: 0 alive row, 1 dead row) or ``name[row, col]`` to ``value``."""

    def f(s: State) -> State:
        row = _rows(s)[which]
        idx = row if col is None else (row, col)
        return _set(s, name, lambda a: a.__setitem__(idx, value))

    return f


def _field(name: str, idx: Any, value: Any = None, delta: int = 0) -> Callable[[State], State]:
    def f(s: State) -> State:
        def g(a: np.ndarray) -> None:
            a[idx] = a[idx] + delta if value is None else value

        return _set(s, name, g)

    return f


L_MICRO = 48.0
BOTH = [  # (id, corruption, Diag bit, full_check message fragment)
    ("asym_rev", _corrupt_rev, Diag.BOND_ASYM, "bonds: asymmetric"),
    ("dead_energy", _at("energy", 1, 5), Diag.ENERGY_RANGE, "dead row: energy"),
    ("dead_uid", _at("uid", 1, 12345), Diag.DEAD_DIRTY, "dead row: uid"),
    ("dead_bond", _dead_bond, Diag.DEAD_DIRTY, "dead row: bond"),
    ("leaked_quantum", _field("resource", (0, 0), delta=1), Diag.LEDGER, "ledger: recomputed TOTAL"),
    ("nan_pos", _at("pos", 0, np.nan, col=0), Diag.NONFINITE, "pos:"),
    ("pos_ge_L", _at("pos", 0, L_MICRO, col=1), Diag.POS_RANGE, "pos:"),
    ("negative_resource", _field("resource", (1, 1), value=-1), Diag.FIELD_RANGE, "resource:"),
    ("over_capacity", _over_capacity, Diag.CAPACITY, "capacity: n_alive"),
    ("alive_zero_energy", _at("energy", 0, 0), Diag.ENERGY_RANGE, "energy: alive"),
    ("alive_over_emax", _at("energy", 0, 2_000_000), Diag.ENERGY_RANGE, "energy: alive"),
    ("nan_vel", _at("vel", 0, np.nan, col=1), Diag.NONFINITE, "vel:"),
    ("nan_h", _at("h", 0, np.nan, col=0), Diag.NONFINITE, "h:"),
    ("nan_chem", _field("chem", (2, 3, 4), value=np.nan), Diag.NONFINITE, "chem:"),
    ("resource_over_hard", _field("resource", (2, 2), value=R_HARD + 1), Diag.FIELD_RANGE, "resource:"),
]


@pytest.mark.parametrize(("corrupt", "bit", "msg"), [c[1:] for c in BOTH], ids=[c[0] for c in BOTH])
def test_both_checks_report(micro_cfg: Config, history, corrupt, bit: Diag, msg: str) -> None:
    _, s, _, _ = history[-1]
    bad = corrupt(s)
    zero = mx.zeros((len(LedgerIdx),), dtype=mx.int64)
    bits = _device(bad, micro_cfg, zero, s.total)  # healthy TOTAL and a zero ledger (§22.2)
    assert bits & bit, f"{Diag(bits)!r} lacks {bit!r}"
    problems = invariants.full_check(bad, micro_cfg)
    assert any(msg in p for p in problems), problems


def _dup_uid(s: State) -> State:
    rows = np.flatnonzero(_np(s.alive))[:2]
    return _set(s, "uid", lambda a: a.__setitem__(rows[1], a[rows[0]]))


def _pend_dup(s: State) -> State:
    return _set(s, "pend_slot", lambda a: a.__setitem__(1, a[0]))


def _pend_drop_newborn(s: State) -> State:
    alive, ps = _np(s.alive), _np(s.pend_slot)
    spare = np.setdiff1d(np.flatnonzero(~alive), ps)[0]
    lane = int(np.flatnonzero(alive[ps])[0])  # a lane holding a newborn
    return _set(s, "pend_slot", lambda a: a.__setitem__(lane, spare))


def _pend_established(s: State) -> State:
    alive, bstep, ps = _np(s.alive), _np(s.birth_step), _np(s.pend_slot)
    old = np.flatnonzero(alive & (bstep < _np(s.step) - 1))[0]
    lane = int(np.flatnonzero(~alive[ps])[0])  # a padding lane
    return _set(s, "pend_slot", lambda a: a.__setitem__(lane, old))


FULL_ONLY = [
    ("duplicate_uid", _dup_uid, "duplicate uid"),
    ("sigma_high", _at("mut_sigma", 0, 0.31), "mut_sigma"),
    ("rate_low", _at("mut_rate", 1, 0.0005), "mut_rate"),
    ("frac_high", _at("offspring_frac", 0, 0.95), "offspring_frac"),
    ("h_range", _at("h", 0, 1.5, col=2), "h: outside"),
    ("heading_pi", _at("heading", 0, np.float32(math.pi)), "heading"),
    ("heading_big", _at("heading", 1, 4.0), "heading"),
    ("tag_range", _at("tag", 1, -1.01, col=3), "tag"),
    ("msg_range", _at("msg", 0, 1.2, col=1), "msg: outside"),
    ("chem_over_max", _field("chem", (0, 1, 1), value=16.5), "chem"),
    ("pend_duplicate", _pend_dup, "pend_slot entries are not distinct"),
    ("pend_missing_newborn", _pend_drop_newborn, "newborn row missing"),
    ("pend_established", _pend_established, "established row"),
    ("dead_vel", _at("vel", 1, 0.1, col=0), "dead row: vel"),
    ("dead_h", _at("h", 1, 0.5, col=0), "dead row: h"),
    ("dead_msg", _at("msg", 1, -0.5, col=1), "dead row: msg"),
    ("dead_last_loss", _at("last_loss", 1, 3), "dead row: last_loss"),
    ("dead_last_harvest", _at("last_harvest", 1, 3), "dead row: last_harvest"),
    ("dead_last_inflow", _at("last_inflow", 1, 3), "dead row: last_inflow"),
    ("future_birth", _at("birth_step", 0, 10**6), "age:"),
    ("clade_range", _at("clade", 1, 999), "clade"),
    ("generation_negative", _at("generation", 0, -1), "generation"),
    ("weight_clip", _field("W1", (3, 0, 0), value=4.5), "W1"),
    ("parent_uid", _at("parent_uid", 0, 10**9), "parent_uid"),
    ("speed", _at("vel", 0, 0.6, col=0), "vel: speed"),
    ("patch_pos", _field("patch_pos", (0, 0), value=-0.5), "patch_pos"),
    ("ledger_cum", _field("ledger_cum", LedgerIdx.HEAT_BASAL, delta=1), "global closure"),
    ("state_total", _field("total", (), delta=1), "state.total"),
    ("layout", lambda s: s._replace(energy=s.energy.astype(mx.int64)), "layout: energy"),
]


@pytest.mark.parametrize(("corrupt", "msg"), [c[1:] for c in FULL_ONLY], ids=[c[0] for c in FULL_ONLY])
def test_full_check_reports(micro_cfg: Config, history, corrupt, msg: str) -> None:
    _, s, _, _ = history[-1]
    problems = invariants.full_check(corrupt(s), micro_cfg)
    assert any(msg in p for p in problems), problems


def _resurrect(s: State, out: StepOut) -> State:
    dead_uid = _np(out.deaths.uid)[_np(out.deaths.valid)][0]
    row = np.flatnonzero(_np(s.alive))[0]
    return _set(s, "uid", lambda a: a.__setitem__(row, dead_uid))


def _newborn_record(out: StepOut) -> StepOut:
    uid = _np(out.births.uid).copy()
    uid[0] += 1000
    return out._replace(births=out.births._replace(uid=mx.array(uid)))


STEPWISE = [  # corruptions that only the relational (prev/out) checks can see
    ("resurrection", lambda s, o: (_resurrect(s, o), o), "resurrection"),
    ("next_uid_jump", lambda s, o: (_field("next_uid", (), delta=1)(s), o), "next_uid advanced"),
    ("record_uid", lambda s, o: (s, _newborn_record(o)), "births: newborn row uid differs"),
    ("step_skip", lambda s, o: (_field("step", (), delta=1)(s), o), "does not follow"),
    (
        "per_step_ledger",
        lambda s, o: (_field("ledger_cum", LedgerIdx.FLOW_CORPSE, delta=-1)(s), o),
        "out.ledger",
    ),
    (
        "theft_identity",
        lambda s, o: (_field("ledger_cum", LedgerIdx.FLOW_STOLEN, delta=4)(s), o),
        "FLOW_STOLEN",
    ),
    (
        "unrecorded_newborn",
        lambda s, o: (s, o._replace(births=o.births._replace(valid=mx.zeros_like(o.births.valid)))),
        "not a newborn",
    ),
]


@pytest.mark.parametrize(("corrupt", "msg"), [c[1:] for c in STEPWISE], ids=[c[0] for c in STEPWISE])
def test_full_check_stepwise_reports(micro_cfg: Config, history, corrupt, msg: str) -> None:
    prev, s, out, _ = history[-1]
    assert int(_np(out.births.valid).sum()) > 0 and int(_np(out.deaths.valid).sum()) > 0
    bad_s, bad_out = corrupt(s, out)
    problems = invariants.full_check(bad_s, micro_cfg, prev, bad_out)
    assert any(msg in p for p in problems), problems
