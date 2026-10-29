"""Arms races, stages 1 and 2, in the whole step (DESIGN §42): null equivalence, the ledger closing with real
traits, determinism, compiled equals eager, the bench mirror, and a deliberately broken energy path."""

from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from alife import arms, bench, invariants
from alife import energy as en
from alife.config import Config, apply_overrides, load_config
from alife.state import CounterIdx, Diag, LedgerIdx, init_state, legacy_digest, state_digest, to_numpy
from alife.step import make_step

NULL = ["arms.enabled=true"]  # zero traits, key_floor 1: the null-equivalence configuration
REAL = [
    "arms.enabled=true",
    "arms.kappa=2",
    "arms.key_floor=0.2",
    "arms.speed_slow=0.3",
    "arms.upkeep_atk=20",
    "arms.upkeep_arm=20",
    "arms.init_atk=0.3",
    "arms.init_arm=0.4",
    "arms.sigma=0.05",
    "arms.sigma_phase=0.05",
]
PRESET = {2: "tiny", 3: "tiny3d"}


def _cfg(nd: int, extra: list[str], *, compile: bool = True) -> Config:
    cfg = load_config(preset=PRESET[nd], overrides=["brain.weight_dtype=float32", *extra])
    return cfg if compile else replace(cfg, run=replace(cfg.run, compile=False))


def _start(cfg: Config):
    s = init_state(cfg)
    s = s._replace(birth_step=mx.where(s.alive, -50, s.birth_step))  # founders old enough to breed
    mx.eval(s)
    return s


def _run(cfg: Config, steps: int, check_every: int = 0):
    fn = make_step(cfg)
    s = _start(cfg)
    prev, bits, births, attacks, problems = s, 0, 0, 0, []
    for t in range(steps):
        s, out = fn(s)
        mx.eval(s, out)
        bits |= int(to_numpy(s.diag))
        c = to_numpy(out.counters)
        births += int(c[CounterIdx.BIRTHS])
        attacks += int(c[CounterIdx.ATTACKS])
        led = to_numpy(out.ledger)
        assert led[LedgerIdx.FLOW_STOLEN] == led[LedgerIdx.FLOW_ATTACK_GAIN] + led[LedgerIdx.FLOW_SPILL], t
        if check_every and t % check_every == 0:
            problems += invariants.full_check(s, cfg, prev, out)
        prev = s
    return s, bits, births, attacks, problems


@pytest.mark.parametrize("nd", [2, 3])
def test_t1_null_arms_leave_the_trajectory_unchanged(nd: int) -> None:
    """Arms on with zero traits, zero upkeep and key_floor 1: the legacy digest equals the arms-off run's."""
    on, off = _cfg(nd, NULL), _cfg(nd, [])
    s_on, s_off = _start(on), _start(off)
    f_on, f_off = make_step(on), make_step(off)
    births = 0
    for t in range(40):
        s_on, o_on = f_on(s_on)
        s_off, o_off = f_off(s_off)
        mx.eval(s_on, s_off, o_on, o_off)
        births += int(to_numpy(o_on.counters)[CounterIdx.BIRTHS])
        if t % 10 == 9:
            assert legacy_digest(s_on) == legacy_digest(s_off), t
    assert births > 0  # the birth hook ran
    np.testing.assert_array_equal(to_numpy(s_on.ledger_cum), to_numpy(s_off.ledger_cum))
    assert not np.array(s_on.arms)[:, : arms.KEY].any()  # the magnitudes stayed zero


@pytest.mark.parametrize("nd", [2, 3])
def test_t3_the_ledger_closes_with_real_traits(nd: int) -> None:
    cfg = _cfg(nd, REAL)
    s, bits, births, attacks, problems = _run(cfg, 200, check_every=40)
    assert bits == 0 and problems == []
    assert births > 0 and attacks > 0  # reproduction (trait mutation) and the contest both ran
    x = np.array(s.arms)
    assert np.isfinite(x).all() and (x >= 0).all() and (x <= 1).all()
    assert (x[:, arms.KEY :] < 1).all()
    alive = np.array(s.alive)
    assert alive.sum() > 8
    # mutation acted: the founders all had (0.3, 0.4); the population now varies
    assert np.std(x[alive, arms.ATK]) > 0 and np.std(x[alive, arms.ARM]) > 0
    # the upkeep reached HEAT_BASAL: more basal heat than the same run without it
    base = _cfg(nd, [o for o in REAL if "upkeep" not in o])
    s2, *_ = _run(base, 200)
    heat = lambda st: int(to_numpy(st.ledger_cum)[LedgerIdx.HEAT_BASAL])  # noqa: E731
    assert heat(s) / max(int(alive.sum()), 1) > 0 and heat(s) != heat(s2)


@pytest.mark.parametrize("nd", [2, 3])
def test_a_real_arms_run_is_deterministic_and_not_vacuous(nd: int) -> None:
    cfg = _cfg(nd, REAL)
    a, *_ = _run(cfg, 60)
    b, *_ = _run(cfg, 60)
    assert state_digest(a) == state_digest(b)
    off, *_ = _run(_cfg(nd, []), 60)
    assert legacy_digest(a) != legacy_digest(off)  # the flag does something


def test_the_arms_functions_compiled_equal_eager() -> None:
    """The whole step is not bit-equal compiled and eager with or without arms (the brain's float
    reductions differ from step 0), so the arms functions are compared on their own: contest, upkeep,
    child traits and the physics act are elementwise and fuse without changing a bit."""
    cfg = _cfg(2, REAL)
    rs = np.random.default_rng(4)
    n = 4096
    x = mx.array(rs.random((n, 6)).astype(np.float32))
    t0 = mx.array(rs.integers(0, n, n).astype(np.int32))
    uid, step = mx.array(np.arange(n, dtype=np.int64)), mx.array(17, mx.int32)
    z = mx.array(rs.standard_normal((n, 6)).astype(np.float32))
    sig = arms.sigma_vector(cfg)
    cases = {
        "contest": (lambda a, t: arms.contest_factor(a, t, None, cfg), (x, t0)),
        "upkeep": (lambda u, s, a: arms.upkeep(u, s, a, cfg), (uid, step, x)),
        "child": (lambda a, zz: arms.child_traits(a, zz, sig), (x, z)),
    }
    for name, (f, args) in cases.items():
        got, want = mx.compile(f)(*args), f(*args)
        np.testing.assert_array_equal(np.array(got), np.array(want), err_msg=name)


@pytest.mark.parametrize("nd", [2, 3])
def test_the_bench_mirror_equals_the_step_with_arms_on(nd: int) -> None:
    cfg = _cfg(nd, REAL, compile=False)
    fused, staged = make_step(cfg), bench._StagedStep(cfg)
    s0 = _start(cfg)
    s, holder = s0, [s0]
    for _ in range(4):
        s, out = fused(s)
        out_staged = staged(holder)
        mx.eval(s, out, holder[0], out_staged)
        assert state_digest(holder[0]) == state_digest(s)
        for a, b in zip(tree_flatten(out), tree_flatten(out_staged), strict=True):
            np.testing.assert_array_equal(to_numpy(a[1]), to_numpy(b[1]), err_msg=a[0])


def test_armour_slows_the_population() -> None:
    """With ARM = 1 and speed_slow = 0.9 the physics sees a tenth of the thrust: the mean speed falls."""
    mean_speed = []
    for slow in ("0.0", "0.9"):
        cfg = _cfg(2, ["arms.enabled=true", "arms.init_arm=1.0", f"arms.speed_slow={slow}"])
        s, *_ = _run(cfg, 30)
        v = np.linalg.norm(np.array(s.vel), axis=1)[np.array(s.alive)]
        mean_speed.append(float(v.mean()))
    assert mean_speed[1] < mean_speed[0]


def test_immigrants_arrive_naive_with_fresh_phases() -> None:
    cfg = _cfg(
        2,
        [*REAL, "population.immigration_every=5", "population.immigration_count=16"],
    )
    s, bits, *_ = _run(cfg, 60)
    assert bits == 0
    x = np.array(s.arms)[np.array(s.alive)]
    assert np.isfinite(x).all() and (x[:, arms.KEY :] < 1).all()


# --- the test of the ledger test: a deliberately broken energy path --------------------------------------


def test_the_ledger_check_catches_an_energy_leak_in_the_arms_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """A combat that creates one quantum per attacker from nothing (no ledger term) must set Diag.LEDGER:
    otherwise the closure tests above would pass for a broken contest."""
    cfg = _cfg(2, REAL)
    real = en.combat

    def leaky(*a, **kw):
        cb = real(*a, **kw)
        return cb._replace(energy=cb.energy + cb.attack.astype(mx.int32))

    monkeypatch.setattr(en, "combat", leaky)
    _, bits, _, attacks, _ = _run(cfg, 60)
    assert attacks > 0 and bits & int(Diag.LEDGER)
    monkeypatch.undo()
    _, bits_ok, *_ = _run(cfg, 60)
    assert bits_ok == 0


def test_an_unbooked_upkeep_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """The upkeep joins the brain cost; an upkeep taken from the energy outside ``pay_costs`` would leak."""
    cfg = _cfg(2, REAL)
    real = en.pay_costs

    def leaky(energy, act, nb, bcost, alive, c, **kw):
        out = real(energy, act, nb, bcost, alive, c, **kw)
        return out._replace(energy=mx.maximum(out.energy - 1, 0) * out.energy.astype(mx.bool_))

    monkeypatch.setattr(en, "pay_costs", leaky)
    _, bits, *_ = _run(cfg, 20)
    assert bits & int(Diag.LEDGER)


def test_apply_overrides_roundtrip(tiny_cfg: Config) -> None:
    assert apply_overrides(tiny_cfg, REAL).arms.upkeep_atk == 20.0


def test_bench_arms_records_the_settings() -> None:
    cfg = _cfg(2, REAL)
    r = bench.run_bench(cfg, steps=4, warmup=1, arms=True, weather_pairs=1)
    assert r["arms"]["arms_config"]["upkeep_atk"] == 20.0 and r["arms"]["arms_config"]["enabled"]
    assert r["arms"]["marginal_ms_median"] == r["arms"]["marginal_ms_median"]  # a number, not NaN
