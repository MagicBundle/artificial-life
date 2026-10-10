"""Whole-step invariants (§22.2): every compiled step passes ``full_check`` and leaves no error diag bit.

Also checks the step's counters and ledger against the state transition (bond conservation, death and birth
counts, count × constant ledger terms), runs the §22.2 stress variants, and mutation-tests both checks on a
real trajectory state. Owned by WP4.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import mlx.core as mx
import numpy as np
import pytest

from alife import invariants, layout
from alife.config import Config, apply_overrides, dims, load_config
from alife.state import (
    ERROR_MASK,
    N_COUNTERS,
    CounterIdx,
    DeathCause,
    Diag,
    LedgerIdx,
    State,
    init_state,
    to_numpy,
)
from alife.step import StepOut, make_step
from helpers import set_output_bias

STEPS = 200
STEPS_SLOW = 300
STRESS_STEPS = 60
STRESS_STEPS_SLOW = 150
C = CounterIdx


def _np(a: mx.array) -> np.ndarray:
    return to_numpy(a)


class Run(NamedTuple):
    prev: State
    state: State
    out: StepOut
    totals: np.ndarray  # (N_COUNTERS,) int64 sums over the run
    peak_alive: int
    ledger: np.ndarray  # (16,) int64 sums over the run


def _check_counters(prev: State, s: State, out: StepOut, cfg: Config) -> None:
    """The step's counters and ledger agree with the state transition (§19.1, §10.8, §16.3)."""
    d, S = dims(cfg), cfg.energy.structure
    c = _np(out.counters).astype(np.int64)
    led = _np(out.ledger)
    assert c.shape == (N_COUNTERS,) and out.counters.dtype == mx.int32 and out.ledger.dtype == mx.int64
    alive0, alive1 = _np(prev.alive), _np(s.alive)
    n0, n1 = int(alive0.sum()), int(alive1.sum())
    bv = _np(out.births.valid)
    assert c[C.N_ALIVE] == n1
    assert c[C.BIRTHS] + c[C.IMMIGRANTS] == bv.sum()
    assert (
        0 <= c[C.BIRTHS] <= c[C.BIRTH_REQUESTS] and c[C.BIRTHS_BLOCKED] == c[C.BIRTH_REQUESTS] - c[C.BIRTHS]
    )
    n_dead = n0 + int(bv.sum()) - n1
    assert c[C.DEATHS_STARVED] + c[C.DEATHS_KILLED] + c[C.DEATHS_AGED] == n_dead
    dv = _np(out.deaths.valid)
    assert dv.sum() == min(n_dead, d.dl) and c[C.DEATHS_UNLOGGED] == max(n_dead - d.dl, 0)
    causes = _np(out.deaths.cause)[dv]
    for cause, idx in ((DeathCause.STARVED, C.DEATHS_STARVED), (DeathCause.KILLED, C.DEATHS_KILLED)):
        assert (causes == cause).sum() <= c[idx]
    assert c[C.INSOLVENT] <= c[C.DEATHS_STARVED]  # insolvent agents always die STARVED (§10.3)
    # Bond pairs are conserved: BONDS_TOTAL(t+1) = BONDS_TOTAL(t) + formed + child − broken.
    pairs0 = int((_np(prev.partner) >= 0).sum()) // 2
    assert c[C.BONDS_TOTAL] == int((_np(s.partner) >= 0).sum()) // 2
    assert c[C.BONDS_TOTAL] == pairs0 + c[C.BONDS_FORMED] + c[C.CHILD_BONDS] - c[C.BONDS_BROKEN]
    assert 0 <= c[C.CHILD_BONDS] <= c[C.BIRTHS]
    assert c[C.BONDS_FORMED] <= c[C.BOND_PROPOSERS] <= 2 * c[C.BONDS_FORMED]
    assert 0 <= c[C.ATTACK_HITS] <= c[C.ATTACKS] <= n0
    for s_idx, b_idx, k_idx in (
        (C.KIN_ATTACK_SUM, C.KIN_ATTACK_BASE, C.ATTACKS),
        (C.KIN_BOND_SUM, C.KIN_BOND_BASE, C.BOND_PROPOSERS),
    ):
        assert 0 <= c[s_idx] <= 256 * c[k_idx] and 0 <= c[b_idx] <= 256 * c[k_idx]
    assert 0 <= c[C.SHARERS] <= n0 and 0 <= c[C.EMITTERS] <= n0
    assert c[C.CELL_OVERFLOW] >= 0 and c[C.MAX_OCCUPANCY] >= 0 and 0 <= c[C.V_CLIPS] <= n0
    # Count × constant ledger terms (§10.8) and flows that need a matching counter.
    assert led[LedgerIdx.HEAT_BIRTH] == c[C.BIRTHS] * cfg.repro.birth_cost
    assert led[LedgerIdx.FLOW_STRUCTURE] == c[C.BIRTHS] * S
    assert led[LedgerIdx.INJECT_IMMIGRATION] == c[C.IMMIGRANTS] * (cfg.energy.e_init + S)
    assert (c[C.SHARERS] > 0) == (led[LedgerIdx.FLOW_SHARED] > 0)
    # A hit steals something; theft needs an attacker (a drained victim's residue can flow with zero grants).
    assert c[C.ATTACK_HITS] == 0 or led[LedgerIdx.FLOW_STOLEN] > 0
    assert led[LedgerIdx.FLOW_STOLEN] == 0 or c[C.ATTACKS] > 0


def _run(cfg: Config, s: State, steps: int) -> Run:
    """Run ``steps`` compiled steps, checking every invariant after each one (§22.2)."""
    fn = make_step(cfg, compile=True)
    dead_uids: set[int] = set()
    totals = np.zeros(N_COUNTERS, np.int64)
    ledger = np.zeros(len(LedgerIdx), np.int64)
    peak = int(_np(s.alive).sum())
    prev, out = s, None
    for i in range(steps):
        prev = s
        s, out = fn(s)
        mx.eval(s, out)
        problems = invariants.full_check(s, cfg, prev, out)
        assert problems == [], f"step {i}: {problems}"
        diag = int(_np(s.diag))
        assert diag & ERROR_MASK == 0, f"step {i}: {Diag(diag & ERROR_MASK)!r}"
        alive_uids = set(_np(s.uid)[_np(s.alive)].tolist())
        assert not (alive_uids & dead_uids), f"step {i}: a uid from an earlier death record is alive"
        dead_uids |= set(_np(out.deaths.uid)[_np(out.deaths.valid)].tolist())
        _check_counters(prev, s, out, cfg)
        totals += _np(out.counters)
        ledger += _np(out.ledger)
        peak = max(peak, int(_np(out.counters)[C.N_ALIVE]))
    assert out is not None
    assert int(_np(s.step)) == int(_np(prev.step)) + 1
    return Run(prev, s, out, totals, peak, ledger)


# --- the main run ------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trajectory(tiny_cfg: Config) -> Run:
    return _run(tiny_cfg, init_state(tiny_cfg), STEPS)


def test_every_step_is_clean(tiny_cfg: Config, trajectory: Run) -> None:
    t, s = trajectory.totals, trajectory.state
    assert int(_np(s.step)) == STEPS
    # The run exercised the paths the checks are about.
    for idx in (C.BIRTHS, C.DEATHS_STARVED, C.BONDS_FORMED, C.BONDS_BROKEN, C.CHILD_BONDS, C.ATTACK_HITS):
        assert t[idx] > 0, idx.name
    assert t[C.SHARERS] > 0 and t[C.EMITTERS] > 0 and trajectory.ledger[LedgerIdx.FLOW_HARVEST] > 0
    assert _np(s.alive).sum() > 0
    # The cumulative ledger is the sum of the per-step ledgers.
    assert np.array_equal(_np(s.ledger_cum), trajectory.ledger)


@pytest.mark.slow
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_every_step_is_clean_long(dtype: str) -> None:
    cfg = load_config(preset="tiny", overrides=[f"brain.weight_dtype={dtype}"])
    run = _run(cfg, init_state(cfg), STEPS_SLOW)
    assert run.totals[C.BIRTHS] > 0 and _np(run.state.alive).sum() > 0


# --- stress variants (§22.2) -------------------------------------------------------------------------------


def _base(dtype: str) -> Config:
    return load_config(preset="tiny", overrides=[f"brain.weight_dtype={dtype}"])


def _dense_start(cfg: Config) -> State:
    """Every founder inside a 12 × 12 square at the world centre: many hash cells overflow C."""
    s = init_state(cfg)
    n0, L = cfg.population.n_init, cfg.world.size
    rs = np.random.default_rng(3)
    pos = _np(s.pos)
    pos[:n0] = (L / 2 - 6.0 + rs.uniform(0.0, 12.0, (n0, 2))).astype(np.float32)
    s = s._replace(pos=mx.array(pos))
    mx.eval(s)
    return s


Variant = tuple[Config, State, Callable[[Run, Config], None]]


def _variant(name: str, dtype: str) -> Variant:
    base = _base(dtype)

    def none(run: Run, cfg: Config) -> None:
        assert run.totals[C.N_ALIVE] > 0

    if name == "attack":
        s = set_output_bias(init_state(base), base, {layout.OUT_ATTACK: 4.0})

        def attacked(run: Run, cfg: Config) -> None:
            assert run.totals[C.ATTACK_HITS] > 50

        return base, s, attacked
    if name == "reproduce":
        cfg = apply_overrides(base, ["population.n_init=960", "repro.min_age=0"])
        s = set_output_bias(init_state(cfg), cfg, {layout.OUT_REPRODUCE: 4.0})

        def pressure(run: Run, cfg: Config) -> None:
            assert run.peak_alive == dims(cfg).n_cap and run.totals[C.BIRTHS_BLOCKED] > 0

        return cfg, s, pressure
    if name == "dense":

        def overflowed(run: Run, cfg: Config) -> None:
            assert run.totals[C.CELL_OVERFLOW] > 0 and int(_np(run.state.diag)) & Diag.CELL_OVERFLOW_WARN

        return base, _dense_start(base), overflowed
    if name == "immigration":
        cfg = apply_overrides(base, ["population.immigration_floor=600"])

        def immigrated(run: Run, cfg: Config) -> None:
            assert run.totals[C.IMMIGRANTS] > 0
            assert (_np(run.state.clade)[_np(run.state.alive)] == cfg.population.n_init).any()

        return cfg, init_state(cfg), immigrated
    if name == "max_age":
        cfg = apply_overrides(base, ["population.max_age=30"])

        def aged(run: Run, cfg: Config) -> None:
            assert run.totals[C.DEATHS_AGED] > 0 and run.totals[C.DEATHS_UNLOGGED] > 0

        return cfg, init_state(cfg), aged
    if name == "fast":
        cfg = load_config(preset="tiny", overlays=["fast"], overrides=[f"brain.weight_dtype={dtype}"])
        return cfg, init_state(cfg), none
    if name == "f16":
        cfg = _base("float16")
        return cfg, init_state(cfg), none
    raise ValueError(name)


VARIANTS = ("attack", "reproduce", "dense", "immigration", "max_age", "fast", "f16")


@pytest.mark.parametrize("name", VARIANTS)
def test_stress_variant(name: str) -> None:
    cfg, s, expect = _variant(name, "float32")
    run = _run(cfg, s, STRESS_STEPS)
    expect(run, cfg)


@pytest.mark.slow
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("name", VARIANTS)
def test_stress_variant_long(name: str, dtype: str) -> None:
    cfg, s, expect = _variant(name, dtype)
    run = _run(cfg, s, STRESS_STEPS_SLOW)
    expect(run, cfg)


# --- mutation testing of the checks (§22.2) ----------------------------------------------------------------


def _corrupt(s: State, fn: Callable[[dict[str, np.ndarray]], None]) -> State:
    a = {name: _np(getattr(s, name)) for name in State._fields}
    fn(a)
    out = State(**{name: mx.array(a[name]).astype(getattr(s, name).dtype) for name in State._fields})
    mx.eval(out)
    return out


def _rows(s: State) -> dict[str, int]:
    alive, partner = _np(s.alive), _np(s.partner)
    bonded = np.flatnonzero(alive & (partner >= 0).any(1))
    alive_rows = np.flatnonzero(alive)
    return {
        "alive": int(alive_rows[0]),
        "alive2": int(alive_rows[1]),
        "dead": int(np.flatnonzero(~alive)[0]),
        "bonded": int(bonded[0]),
        "slot": int(np.argmax(partner[bonded[0]] >= 0)),
    }


def _set(name: str, idx: tuple, value: float | int) -> Callable[[dict[str, np.ndarray]], None]:
    def fn(a: dict[str, np.ndarray]) -> None:
        a[name][idx] = value

    return fn


def _device_cases(r: dict[str, int], cfg: Config) -> dict[str, tuple[Callable, Diag]]:
    i, b, dead, al = r["bonded"], r["slot"], r["dead"], r["alive"]

    def asym(a: dict[str, np.ndarray]) -> None:
        a["rev"][i, b] = (a["rev"][i, b] + 1) % layout.B

    def dead_bond(a: dict[str, np.ndarray]) -> None:
        a["partner"][dead, 0], a["rev"][dead, 0] = al, 0

    def over_cap(a: dict[str, np.ndarray]) -> None:
        a["alive"][:] = True

    return {
        "asymmetric_rev": (asym, Diag.BOND_ASYM),
        "dead_row_energy": (_set("energy", (dead,), 5), Diag.ENERGY_RANGE),
        "dead_row_uid": (_set("uid", (dead,), 10**6), Diag.DEAD_DIRTY),
        "dead_row_bond": (dead_bond, Diag.DEAD_DIRTY),
        "leaked_quantum": (lambda a: a["resource"].__setitem__((0, 0), a["resource"][0, 0] + 1), Diag.LEDGER),
        "nan_pos": (_set("pos", (al, 0), np.nan), Diag.NONFINITE),
        "pos_eq_L": (_set("pos", (al, 1), cfg.world.size), Diag.POS_RANGE),
        "negative_resource": (_set("resource", (1, 1), -1), Diag.FIELD_RANGE),
        "over_capacity": (over_cap, Diag.CAPACITY),
    }


def _host_only_cases(r: dict[str, int], cfg: Config) -> dict[str, Callable]:
    al, al2, dead = r["alive"], r["alive2"], r["dead"]
    mu = cfg.mutation

    def dup_uid(a: dict[str, np.ndarray]) -> None:
        a["uid"][al2] = a["uid"][al]

    def dup_pending(a: dict[str, np.ndarray]) -> None:
        a["pend_slot"][1] = a["pend_slot"][0]

    return {
        "duplicate_uid": dup_uid,
        "sigma_out_of_range": _set("mut_sigma", (al,), mu.sigma_max * 2),
        "rate_out_of_range": _set("mut_rate", (al,), 0.0),
        "frac_out_of_range": _set("offspring_frac", (al,), 0.99),
        "h_out_of_range": _set("h", (al, 0), 1.5),
        "heading_out_of_range": _set("heading", (al,), 3.5),
        "tag_out_of_range": _set("tag", (al, 0), -1.5),
        "chem_above_max": _set("chem", (0, 0, 0), cfg.fields.chem_max + 1.0),
        "pending_duplicate": dup_pending,
        "dead_row_vel": _set("vel", (dead, 0), 0.1),
        "dead_row_h": _set("h", (dead, 0), 0.1),
        "dead_row_msg": _set("msg", (dead, 1), -0.1),
        "dead_row_last_loss": _set("last_loss", (dead,), 3),
        "dead_row_last_inflow": _set("last_inflow", (dead,), 3),
    }


def _device_bits(s: State, healthy_total: mx.array, cfg: Config) -> int:
    zero = mx.zeros((len(LedgerIdx),), dtype=mx.int64)
    z = mx.array(0, dtype=mx.int32)
    return int(_np(invariants.device_check(s, zero, healthy_total, z, z, z, z, cfg)))


def test_checks_catch_corruptions_of_a_real_state(tiny_cfg: Config, trajectory: Run) -> None:
    cfg, s = tiny_cfg, trajectory.state
    assert invariants.full_check(s, cfg) == [] and _device_bits(s, s.total, cfg) == 0
    r = _rows(s)
    for name, (fn, bit) in _device_cases(r, cfg).items():
        bad = _corrupt(s, fn)
        assert _device_bits(bad, s.total, cfg) & bit, name
        assert invariants.full_check(bad, cfg) != [], name
    for name, fn in _host_only_cases(r, cfg).items():
        bad = _corrupt(s, fn)
        assert invariants.full_check(bad, cfg) != [], name


def test_step_checks_catch_a_bad_transition(tiny_cfg: Config, trajectory: Run) -> None:
    """``full_check(s, cfg, prev, out)`` sees corruptions only the transition reveals."""
    cfg, prev, s, out = tiny_cfg, trajectory.prev, trajectory.state, trajectory.out
    assert invariants.full_check(s, cfg, prev, out) == []
    leak = out._replace(ledger=out.ledger + mx.array(np.eye(len(LedgerIdx), dtype=np.int64)[0]))
    assert invariants.full_check(s, cfg, prev, leak) != []
    survivor_uid = sorted(
        set(_np(s.uid)[_np(s.alive)].tolist()) & set(_np(prev.uid)[_np(prev.alive)].tolist())
    )[0]
    dl = dims(cfg).dl
    zombie = out.deaths._replace(
        uid=mx.array(np.full(dl, survivor_uid, np.int64)), valid=mx.array(np.ones(dl, bool))
    )
    assert invariants.full_check(s, cfg, prev, out._replace(deaths=zombie)) != []
    stale = s._replace(step=s.step + 1)
    assert invariants.full_check(stale, cfg, prev, out) != []


def test_bf16_steps_are_clean(tiny_cfg_bf16: Config) -> None:
    """bfloat16 weights (host reads widen through ``to_numpy``, F17) pass every check for a few steps."""
    run = _run(tiny_cfg_bf16, init_state(tiny_cfg_bf16), 5)
    assert run.state.W1.dtype == mx.bfloat16 and run.state.pend_W1.dtype == mx.bfloat16


# --- the step's device_check inputs (§10.9, §22.3) ---------------------------------------------------------


def test_step_feeds_device_check_its_own_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """An eager step hands ``device_check`` the real max cell occupancy, attackers per victim, overflow cells
    and speed clips, and the resulting warning bits reach ``s.diag``."""
    from alife import energy

    cfg = apply_overrides(_base("float32"), ["physics.v_max=0.05"])  # full thrust (0.1/drag) clips
    s = set_output_bias(_dense_start(cfg), cfg, {layout.OUT_ATTACK: 4.0, layout.OUT_THRUST: 4.0})
    seen: dict[str, tuple] = {}
    device_check, combat = invariants.device_check, energy.combat

    def spy_check(*args: mx.array) -> mx.array:
        seen["check"] = args
        return device_check(*args)

    def spy_combat(*args: mx.array) -> energy.Combat:
        seen["combat"] = (args, combat(*args))
        return seen["combat"][1]

    monkeypatch.setattr(invariants, "device_check", spy_check)
    monkeypatch.setattr(energy, "combat", spy_combat)
    s1, out = make_step(cfg, compile=False)(s)
    mx.eval(s1, out)
    occ, attackers, overflow, clips = (int(_np(x)) for x in seen["check"][3:7])

    alive = _np(s.alive)
    cell = np.floor(_np(s.pos)[alive] / cfg.world.field_cell).astype(np.int64)
    assert occ == np.unique(cell, axis=0, return_counts=True)[1].max() > 1
    (_, _, _, act, _, _), cb = seen["combat"]
    assert attackers == np.bincount(_np(act.attack_target)[_np(cb.attack)]).max() > 1
    counters = _np(out.counters)
    assert overflow == counters[C.CELL_OVERFLOW] > 0
    assert clips == counters[C.V_CLIPS] > 0
    diag = int(_np(s1.diag))
    assert diag & Diag.VCLIP_WARN and diag & Diag.CELL_OVERFLOW_WARN and not diag & ERROR_MASK


def test_crowded_field_cell_sets_int_headroom() -> None:
    """A start-of-step field cell holding ``occ_limit`` agents trips ``INT_HEADROOM`` in a real step."""
    cfg = apply_overrides(_base("float32"), ["population.n_max=2048", "population.n_init=1100"])
    d = dims(cfg)
    assert d.occ_limit <= cfg.population.n_init
    s = init_state(cfg)
    pos = _np(s.pos)
    rs = np.random.default_rng(5)
    pos[: cfg.population.n_init] = 60.0 + rs.uniform(0.0, cfg.world.field_cell, (cfg.population.n_init, 2))
    s = s._replace(pos=mx.array(pos.astype(np.float32)))
    s1, out = make_step(cfg, compile=False)(s)
    mx.eval(s1, out)
    assert int(_np(s1.diag)) & Diag.INT_HEADROOM
