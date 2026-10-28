"""Arms races, stages 1 and 2 (DESIGN §42): the contest, the costs and the inheritance as functions.

T1 null equivalence, T2 contest properties, T3 ledger closure of the combat, and the owner-approved edge
cases E11 (0/0 at the clamp bounds), E12 (phases wrapped after millions of mutations) and E13 (sub-threshold
upkeep is not free).
"""

from __future__ import annotations

import dataclasses
import itertools

import mlx.core as mx
import numpy as np
import pytest

from alife import arms
from alife import energy as en
from alife.config import Config, apply_overrides, dims
from test_ext_protection import _batch, _fields, _legacy_combat, _same

ON = ["arms.enabled=true"]
MAGS = [0.0, 1e-9, 0.3, 1.0]


def _cfg(base: Config, *extra: str) -> Config:
    return apply_overrides(base, [*ON, *extra])


def _arms(n: int, rs: np.random.Generator, *, mags: bool = True) -> mx.array:
    x = np.zeros((n, arms.N_TRAITS), np.float32)
    if mags:
        x[:, arms.ATK] = rs.random(n)
        x[:, arms.ARM] = rs.random(n)
    x[:, arms.KEY] = rs.random(n)
    x[:, arms.LOCK] = rs.random(n)
    return mx.array(x)


# --- T1: null equivalence ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra", [[], ["arms.key_floor=0.2"], ["arms.kappa=0"], ["arms.kappa=0", "arms.key_floor=0.2"]]
)
def test_t1_zero_traits_equal_the_legacy_combat(micro_cfg: Config, extra: list[str]) -> None:
    """Zero magnitudes and equal keys and locks: the factor is exactly 1.0, so every Combat field equals the
    legacy combat's (the full graph with arms off is the legacy one by construction)."""
    cfg = _cfg(micro_cfg, *extra)
    n = dims(cfg).n
    z = mx.zeros((n, arms.N_TRAITS), mx.float32)
    for seed in range(5):
        e, res, fcell, act, partner, _ = _batch(cfg, seed)
        cb = en.combat(e, res, fcell, act, partner, cfg, arms=z)
        ref = _legacy_combat(e, res, fcell, act, partner, cfg)
        _same(_fields(cb), {k: np.array(v) for k, v in ref.items()})
        assert np.array(cb.hit).any()


def test_t1_the_off_call_is_untouched(micro_cfg: Config) -> None:
    assert arms.combat_kw(None, micro_cfg) == {}
    assert arms.basal_cost(dataclasses.make_dataclass("S", ["brain_cost"])(5), micro_cfg) == 5
    act = object()
    assert arms.physics_act(act, None, micro_cfg) is act


# --- T2: contest properties ------------------------------------------------------------------------------


def _grid_factor(cfg: Config, atk: np.ndarray, armour: np.ndarray, key: np.ndarray, lock: np.ndarray):
    """The MLX factor for attackers (atk, key) against victims (armour, lock), one pair per row."""
    n = len(atk)
    x = np.zeros((2 * n, arms.N_TRAITS), np.float32)
    x[:n, arms.ATK], x[:n, arms.KEY] = atk, key
    x[n:, arms.ARM], x[n:, arms.LOCK] = armour, lock
    t0 = mx.array(np.concatenate([np.arange(n, 2 * n)] * 2), mx.int32)  # rows n.. are victims only
    p = arms.contest_factor(mx.array(x), t0, None, cfg)
    return None if p is None else np.array(p)[:n]


def test_t2_monotone_bounded_and_the_mirror_agrees(micro_cfg: Config) -> None:
    cfg = _cfg(micro_cfg, "arms.key_floor=0.2")
    rs = np.random.default_rng(0)
    n = 4000
    atk, armour, key, lock = (rs.random(n).astype(np.float32) for _ in range(4))
    p = _grid_factor(cfg, atk, armour, key, lock)
    assert p.dtype == np.float32 and (p >= 0).all() and (p <= 1).all()
    np.testing.assert_allclose(p, arms.contest_np(atk, armour, key, lock, cfg), rtol=2e-6, atol=2e-6)
    a = cfg.arms
    r_max = a.kappa / (a.kappa + a.a0)
    assert (p >= (1 - r_max) * a.key_floor - 1e-6).all()  # no immunity: bounded away from 0
    # monotone: P falls in the armour, rises in the attack strength (key and lock aligned)
    ones = np.ones(50, np.float32)
    grid = np.linspace(0, 1, 50, dtype=np.float32)
    zero = np.zeros(50, np.float32)
    assert (np.diff(_grid_factor(cfg, zero + 0.5, grid, zero, zero)) <= 1e-7).all()
    assert (np.diff(_grid_factor(cfg, grid, 0.5 * ones, zero, zero)) >= -1e-7).all()
    # the armour is futile at atk = 1 less than at atk = 0: r_max(ATK) = kappa / (kappa + a0 + ATK)
    assert _grid_factor(cfg, ones, ones, zero, zero)[0] == pytest.approx(1 - 2 / 4, abs=1e-5)
    assert _grid_factor(cfg, zero, ones, zero, zero)[0] == pytest.approx(1 - 2 / 3, abs=1e-5)


def test_t2_key_lock_match(micro_cfg: Config) -> None:
    cfg = _cfg(micro_cfg, "arms.key_floor=0.2", "arms.kappa=0")  # the key axis alone
    z = np.zeros(3, np.float32)
    key = np.array([0.3, 0.3, 0.3], np.float32)
    lock = np.array([0.3, 0.8, 0.55], np.float32)  # aligned, half a turn, a quarter turn
    p = _grid_factor(cfg, z, z, key, lock)
    assert p[0] == pytest.approx(1.0, abs=1e-6)
    assert p[1] == pytest.approx(0.2, abs=1e-6)
    assert p[2] == pytest.approx(0.2 + 0.8 * 0.25, abs=1e-6)  # c = 0.5
    # the circle: a lock one turn away (the unwrapped difference) matches the same
    q = _grid_factor(cfg, z, z, key, lock - 1.0)
    np.testing.assert_allclose(q, p, atol=2e-6)
    # key_floor = 1 and kappa = 0: statically off
    assert _grid_factor(_cfg(micro_cfg, "arms.kappa=0"), z, z, key, lock) is None


def test_t2_the_integer_bounds_of_the_contest(micro_cfg: Config) -> None:
    """``m <= 1`` so the demand stays below attack_max < 2^16 and ``grant·q < 2^24`` (DESIGN §10.9)."""
    cfg = _cfg(micro_cfg, "arms.key_floor=0.2")
    n = dims(cfg).n
    for seed in range(5):
        e, res, fcell, act, partner, rs = _batch(cfg, seed)
        a = _arms(n, rs)
        cb = en.combat(e, res, fcell, act, partner, cfg, arms=a)
        base = en.combat(e, res, fcell, act, partner, cfg)
        assert int(np.array(cb.lost).max()) <= cfg.combat.attack_max * n
        assert cfg.combat.attack_max < 2**16 and cfg.combat.attack_max * 256 < 2**24
        assert (np.array(cb.prot)[np.array(cb.attack)] <= 1.0).all()
        # no larger a theft than the unprotected one, victim by victim in total
        assert int(np.array(cb.lost).sum()) <= int(np.array(base.lost).sum())


# --- E11: 0/0 at the clamp bounds -------------------------------------------------------------------------


@pytest.mark.parametrize("a0", [1.0, 1e-3, 0.0])
@pytest.mark.parametrize("kappa", [0.0, 2.0, 9.0])
@pytest.mark.parametrize("floor", [1.0, 0.0])
def test_e11_attack_and_armour_at_every_bound(
    micro_cfg: Config, kappa: float, a0: float, floor: float
) -> None:
    """ATK and ARM at 0, 1 and a hair outside [0, 1] (clamped), a0 down to 0 (the validated minimum is > 0,
    but the epsilon keeps 0/0 out even then): finite, in [0, 1], never NaN."""
    cfg = _cfg(micro_cfg)
    cfg = dataclasses.replace(cfg, arms=dataclasses.replace(cfg.arms, kappa=kappa, a0=a0, key_floor=floor))
    bounds = np.array([-1e-7, 0.0, 1e-30, 1.0, 1.0 + 1e-6], np.float32)
    atk, armour = (x.ravel().astype(np.float32) for x in np.meshgrid(bounds, bounds))
    z = np.zeros_like(atk)
    p = _grid_factor(cfg, atk, armour, z + 0.25, z + 0.75)
    if kappa == 0.0 and floor == 1.0:
        assert p is None
        return
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    p_np = arms.contest_np(atk, armour, z + 0.25, z + 0.75, cfg)
    assert np.isfinite(p_np).all()
    both0 = (np.clip(atk, 0, 1) == 0) & (np.clip(armour, 0, 1) == 0)
    if floor == 1.0:
        assert (p[both0] == 1.0).all()  # 0 / eps = 0: no repulsion, not NaN


def test_e11_combat_with_extreme_traits_conserves(micro_cfg: Config) -> None:
    cfg = _cfg(micro_cfg, "arms.key_floor=0.0")
    n = dims(cfg).n
    for vals in itertools.product((0.0, 1.0), repeat=2):
        x = np.zeros((n, arms.N_TRAITS), np.float32)
        x[:, arms.ATK], x[:, arms.ARM] = vals
        e, res, fcell, act, partner, _ = _batch(cfg, 3)
        cb = en.combat(e, res, fcell, act, partner, cfg, arms=mx.array(x))
        assert np.isfinite(np.array(cb.prot)).all()
        assert int(np.array(cb.lost).sum()) == int(
            np.array(cb.gain).sum() + np.array(cb.spill).sum() + np.array(cb.residue).sum()
        )


# --- T3: the ledger of the combat -------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
def test_t3_the_combat_conserves_energy(micro_cfg: Config, seed: int) -> None:
    cfg = _cfg(micro_cfg, "arms.key_floor=0.2", "combat.protect_floor=0.1")
    n = dims(cfg).n
    e, res, fcell, act, partner, rs = _batch(cfg, seed)
    cb = en.combat(e, res, fcell, act, partner, cfg, arms=_arms(n, rs))
    total0 = int(np.array(e).sum()) + int(np.array(res).sum())
    total1 = int(np.array(cb.energy).sum()) + int(np.array(cb.resource).sum())
    assert total1 == total0
    lost, gain, spill, residue = (
        np.array(getattr(cb, k)).astype(np.int64) for k in ("lost", "gain", "spill", "residue")
    )
    assert lost.sum() == gain.sum() + spill.sum() + residue.sum()
    assert (np.array(cb.energy) >= 0).all() and (np.array(cb.resource) >= 0).all()


# --- the costs --------------------------------------------------------------------------------------------


def _upkeep(cfg: Config, atk: np.ndarray, arm: np.ndarray, uid: np.ndarray, step: int) -> np.ndarray:
    x = np.zeros((len(atk), arms.N_TRAITS), np.float32)
    x[:, arms.ATK], x[:, arms.ARM] = atk, arm
    return np.array(arms.upkeep(mx.array(uid.astype(np.int64)), mx.array(step, mx.int32), mx.array(x), cfg))


def test_e13_a_sub_unit_upkeep_is_paid_on_average(micro_cfg: Config) -> None:
    """``floor(0.03)`` is 0, a free ride; the dither pays 0.03 on average (and 0 for a zero trait)."""
    cfg = _cfg(micro_cfg, "arms.upkeep_atk=3")
    n = 200_000
    uid = np.arange(n)
    atk = np.full(n, 0.1, np.float32)  # x = 3 * 0.01 = 0.03
    up = _upkeep(cfg, atk, np.zeros(n, np.float32), uid, step=7)
    assert set(np.unique(up)) <= {0, 1} and up.sum() > 0
    assert abs(up.mean() - 0.03) < 2e-3  # standard error 3.8e-4
    # the same agent across steps pays the same mean (the dither moves with the step)
    steps = np.array([_upkeep(cfg, atk[:1], np.zeros(1, np.float32), uid[:1], t)[0] for t in range(20_000)])
    assert abs(steps.mean() - 0.03) < 4e-3
    # a zero trait pays exactly nothing, whatever the uid and the step
    zero = _upkeep(cfg, np.zeros(n, np.float32), np.zeros(n, np.float32), uid, step=123)
    assert zero.max() == 0


def test_e13_the_mean_upkeep_is_exact_for_mixed_traits(micro_cfg: Config) -> None:
    cfg = _cfg(micro_cfg, "arms.upkeep_atk=20", "arms.upkeep_arm=15")
    rs = np.random.default_rng(5)
    n = 300_000
    atk, arm = rs.random(n).astype(np.float32), rs.random(n).astype(np.float32)
    x = 20 * atk.astype(np.float64) ** 2 + 15 * arm.astype(np.float64) ** 2
    up = _upkeep(cfg, atk, arm, np.arange(n), step=11)
    assert np.abs(up - x).max() < 1.0 + 1e-3  # floor(x) or floor(x) + 1
    assert abs((up - x).mean()) < 0.01
    np.testing.assert_array_equal(up, _upkeep(cfg, atk, arm, np.arange(n), step=11))  # deterministic


def test_the_founder_upkeep_matches_the_mean(micro_cfg: Config) -> None:
    cfg = _cfg(
        micro_cfg, "arms.upkeep_atk=20", "arms.upkeep_arm=15", "arms.init_atk=0.3", "arms.init_arm=0.5"
    )
    assert arms.founder_upkeep(cfg) == pytest.approx(20 * 0.09 + 15 * 0.25)
    assert arms.founder_upkeep(micro_cfg) == 0.0


def test_the_speed_factor_only_slows(micro_cfg: Config) -> None:
    from test_energy import _actions

    cfg = _cfg(micro_cfg, "arms.speed_slow=0.3")
    n = dims(cfg).n
    act = _actions(n, thrust=np.ones(n, np.float32))
    x = np.zeros((n, arms.N_TRAITS), np.float32)
    x[:, arms.ARM] = np.linspace(0, 1, n)
    s = dataclasses.make_dataclass("S", ["arms"])(mx.array(x))
    out = np.array(arms.physics_act(act, s, cfg).thrust)
    assert (out <= 1.0).all() and out.min() == pytest.approx(0.7, abs=1e-6) and out[0] == 1.0
    assert arms.physics_act(act, s, micro_cfg) is act  # off


# --- inheritance ------------------------------------------------------------------------------------------


def test_child_traits_clip_wrap_and_determinism(micro_cfg: Config) -> None:
    cfg = _cfg(micro_cfg, "arms.sigma=0.5", "arms.sigma_phase=0.5")
    sig = arms.sigma_vector(cfg)
    assert np.array(sig).tolist() == [0.5, 0.5, 0.0, 0.0, 0.5, 0.5]
    rs = np.random.default_rng(1)
    parent = mx.array(rs.random((5000, 6)).astype(np.float32))
    z = mx.array(rs.standard_normal((5000, 6)).astype(np.float32))
    c = np.array(arms.child_traits(parent, z, sig))
    assert (c[:, arms.KEY :] >= 0).all() and (c[:, arms.KEY :] < 1).all()  # phases wrapped
    assert (c[:, :2] >= 0).all() and (c[:, :2] <= 1).all()  # magnitudes clipped
    assert (c[:, 2:4] == np.array(parent)[:, 2:4]).all()  # toxin and display do not mutate yet
    assert (c[:, :2] == 0).any() and (c[:, :2] == 1).any()  # mass at the bounds
    np.testing.assert_array_equal(c, np.array(arms.child_traits(parent, z, sig)))


def test_e12_phases_stay_wrapped_after_millions_of_mutations(micro_cfg: Config) -> None:
    """4M x 2 phase mutations, each child the parent of the next: no phase reaches 1.0 or leaves [0, 1), no
    NaN, and the angle never accumulates beyond one turn (so ``cos(2π(KEY − LOCK))`` keeps its precision)."""
    cfg = _cfg(micro_cfg, "arms.sigma=0.03", "arms.sigma_phase=0.05")
    sig = arms.sigma_vector(cfg)
    lanes, iters = 2048, 2000
    x = mx.array(np.random.default_rng(2).random((lanes, 6)).astype(np.float32))
    key = mx.random.key(9)
    lo, hi = 1.0, 0.0
    for _ in range(iters):
        key, k = mx.random.split(key)
        x = arms.child_traits(x, mx.random.normal((lanes, 6), key=k), sig)
        ph = x[:, arms.KEY :]
        lo, hi = min(lo, float(ph.min())), max(hi, float(ph.max()))
    xs = np.array(x)
    assert np.isfinite(xs).all() and lo >= 0.0 and hi < 1.0
    assert (xs[:, :2] >= 0).all() and (xs[:, :2] <= 1).all()
    # the wrap itself: a tiny negative phase must not round to 1.0
    tiny = mx.array([[0, 0, 0, 0, -1e-9, 1.0 + 1e-7]], mx.float32)
    out = np.array(arms.child_traits(tiny, mx.zeros((1, 6)), sig))
    assert out[0, arms.KEY] == 0.0 and 0.0 <= out[0, arms.LOCK] < 1.0
    # the match of a phase with itself is exactly aligned after the wrap
    p = np.array([out[0, arms.KEY]], np.float32)
    assert arms.contest_np(np.zeros(1), np.zeros(1), p, p, cfg)[0] == pytest.approx(1.0)
