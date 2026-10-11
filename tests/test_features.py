"""Optional levers added after v0.1: periodic immigration (§12.7), seasons (§13.1) and their config rules.

Defaults must leave every earlier trajectory, config hash and checkpoint untouched (§17, §18).
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from alife import fields, invariants, lifecycle
from alife.config import (
    POST_V01_KEYS,
    Config,
    apply_overrides,
    config_from_dict,
    config_hash,
    config_to_dict,
    dims,
    load_config,
    validate,
)
from alife.state import CounterIdx, LedgerIdx, State, init_state, state_digest, to_numpy
from alife.step import make_step

C = CounterIdx

# config_hash of the unmodified ``tiny`` preset, computed before the post-v0.1 keys existed.
TINY_HASH_V01 = "d25e0eaee7349ae366e3e4e60ebab107cf7a89fcf49be88a12d81099ff2561d5"


def _np(a: mx.array) -> np.ndarray:
    return to_numpy(a)


# --- config: defaults, validation, hash -------------------------------------------------------------------


def test_tiny_config_hash_is_pinned_across_the_new_keys() -> None:
    assert config_hash(load_config(preset="tiny")) == TINY_HASH_V01


def test_old_config_dict_loads_with_defaults_and_same_hash() -> None:
    cfg = load_config(preset="tiny")
    old = config_to_dict(cfg)
    for sec, key in POST_V01_KEYS:  # a config.json written before the keys existed
        del old[sec][key]
    back = config_from_dict(old)
    validate(back)
    assert back == cfg
    assert config_hash(back) == TINY_HASH_V01
    assert (back.population.immigration_every, back.population.immigration_count) == (0, 0)
    assert (back.fields.season_period, back.fields.season_amplitude) == (0, 0.0)
    assert back.fields.season_mode == "global"


def test_hash_tracks_new_keys_only_off_default() -> None:
    cfg = load_config(preset="tiny")
    for ov in (
        ["population.immigration_every=10", "population.immigration_count=3"],
        ["population.immigration_count=3"],
        ["fields.season_period=50"],
        ["fields.season_period=50", "fields.season_amplitude=0.3"],
        ["fields.season_mode=per_patch"],
    ):
        assert config_hash(apply_overrides(cfg, ov)) != config_hash(cfg), ov
    # An explicit default is the same as an absent key.
    assert config_hash(apply_overrides(cfg, ["fields.season_mode=global"])) == config_hash(cfg)


def test_valid_new_settings_pass() -> None:
    cfg = load_config(
        preset="tiny",
        overrides=[
            "population.immigration_every=5",
            "population.immigration_count=32",
            "fields.season_period=20",
            "fields.season_amplitude=1.0",
            "fields.season_mode=per_patch",
        ],
    )
    assert cfg.population.immigration_count == 32 and cfg.fields.season_mode == "per_patch"


# --- periodic immigration ---------------------------------------------------------------------------------


def _lanes(cfg: Config, n_surv: int, n_req: int, step: int | None) -> lifecycle.Lanes:
    n = dims(cfg).n
    sv = np.zeros(n, bool)
    sv[:n_surv] = True
    req = np.zeros(n, bool)
    req[:n_req] = True
    st = None if step is None else mx.array(step, dtype=mx.int32)
    return lifecycle.allocate_lanes(mx.array(req), mx.array(sv), mx.random.key(3), cfg, st)


def test_periodic_lanes_schedule_and_caps(micro_cfg: Config) -> None:
    d = dims(micro_cfg)  # N = 64, M = 8, N_CAP = 56
    cfg = apply_overrides(micro_cfg, ["population.immigration_every=3", "population.immigration_count=5"])
    for step in range(12):  # on schedule exactly when (step + 1) % 3 == 0
        want = 5 if (step + 1) % 3 == 0 else 0
        ln = _lanes(cfg, n_surv=10, n_req=0, step=step)
        assert int(_np(ln.n_imm)) == want, step
        assert int((_np(ln.kind) == lifecycle.LaneKind.IMMIGRANT).sum()) == want
    # Lane cap: births take lanes first, immigrants get what is left of M = 8.
    ln = _lanes(cfg, n_surv=10, n_req=6, step=2)
    assert (int(_np(ln.n_birth)), int(_np(ln.n_imm))) == (6, 2)
    # Capacity cap: only N_CAP − n_surv − births rows are free.
    ln = _lanes(cfg, n_surv=d.n_cap - 2, n_req=0, step=2)
    assert int(_np(ln.n_imm)) == 2
    ln = _lanes(cfg, n_surv=d.n_cap - 3, n_req=2, step=2)
    assert (int(_np(ln.n_birth)), int(_np(ln.n_imm))) == (2, 1)
    ln = _lanes(cfg, n_surv=d.n_cap, n_req=4, step=2)
    assert (int(_np(ln.n_birth)), int(_np(ln.n_imm))) == (0, 0)
    with pytest.raises(ValueError, match="step"):
        _lanes(cfg, n_surv=10, n_req=0, step=None)


def test_periodic_combines_with_floor_by_max(micro_cfg: Config) -> None:
    base = ["population.immigration_floor=40", "population.immigrants_per_step=2"]
    cfg = apply_overrides(
        micro_cfg, [*base, "population.immigration_every=3", "population.immigration_count=5"]
    )
    assert int(_np(_lanes(cfg, 8, 0, step=2).n_imm)) == 5  # periodic demand 5 > floor demand 2
    assert int(_np(_lanes(cfg, 8, 0, step=0).n_imm)) == 2  # off schedule: the floor rule alone
    assert int(_np(_lanes(cfg, 39, 0, step=0).n_imm)) == 1  # floor demand 40 − 39
    assert int(_np(_lanes(cfg, 39, 0, step=2).n_imm)) == 5  # max, not the sum
    hi = apply_overrides(cfg, ["population.immigrants_per_step=7", "population.immigration_count=2"])
    assert int(_np(_lanes(hi, 8, 0, step=2).n_imm)) == 7  # floor demand 7 > periodic 2
    assert int(_np(_lanes(hi, 40, 0, step=2).n_imm)) == 2  # floor met: the periodic rule remains


def _run(cfg: Config, steps: int) -> tuple[State, list[tuple[State, State, object]]]:
    fn = make_step(cfg, compile=True)
    s = init_state(cfg)
    hist = []
    for i in range(steps):
        prev = s
        s, out = fn(s)
        mx.eval(s, out)
        problems = invariants.full_check(s, cfg, prev, out)
        assert problems == [], f"step {i}: {problems}"
        hist.append((prev, s, out))
    return s, hist


def test_periodic_immigration_whole_step(tiny_cfg: Config) -> None:
    cfg = apply_overrides(
        tiny_cfg,
        ["population.immigration_every=6", "population.immigration_count=4", "brain.weight_dtype=float32"],
    )
    d, S = dims(cfg), cfg.energy.structure
    s, hist = _run(cfg, 25)
    total = 0
    for i, (_prev, _s, out) in enumerate(hist):
        c, led = _np(out.counters), _np(out.ledger)
        # Birth lanes come first, so immigrants get what is left of the M lanes (births start near step 20).
        want = min(4, d.m - int(c[C.BIRTHS])) if (i + 1) % 6 == 0 else 0
        assert c[C.IMMIGRANTS] == want, i
        assert led[LedgerIdx.INJECT_IMMIGRATION] == want * (cfg.energy.e_init + S)
        valid = _np(out.births.valid)
        imm = valid & (_np(out.births.parent_uid) == -1)
        assert imm.sum() == want
        assert (_np(out.births.clade)[imm] == cfg.population.n_init).all()
        assert (_np(out.births.generation)[imm] == 0).all()
        total += want
    assert total >= 12 and int(_np(s.alive).sum()) <= d.n_cap  # steps 5, 11, 17 are always full
    # Deterministic: a second run reproduces the digest.
    s2, _ = _run(cfg, 25)
    assert state_digest(s2) == state_digest(s)


def test_periodic_immigration_off_by_default_matches_no_keys(tiny_cfg: Config) -> None:
    assert tiny_cfg.population.immigration_every == 0 and tiny_cfg.population.immigration_count == 0


# --- seasons ----------------------------------------------------------------------------------------------


def _patch_pos(cfg: Config) -> mx.array:
    k = mx.random.split(mx.random.key(11), 2)
    pos, _ = fields.init_patches(k[0], k[1], cfg)
    return pos


def _gauss_sum(patch_pos: np.ndarray, cfg: Config, weights: np.ndarray) -> np.ndarray:
    """float64 reference: Σ_p w_p · torus Gaussian_p on the cell grid, indexed [y, x]."""
    fg, L, fc = dims(cfg).fg, cfg.world.size, cfg.world.field_cell
    c = (np.arange(fg) + 0.5) * fc
    yy, xx = np.meshgrid(c, c, indexing="ij")
    total = np.zeros((fg, fg))
    for (px, py), w in zip(patch_pos.astype(np.float64), weights, strict=True):
        dx, dy = xx - px, yy - py
        dx -= L * np.round(dx / L)
        dy -= L * np.round(dy / L)
        total += w * np.exp(-(dx * dx + dy * dy) / (2 * cfg.fields.patch_sigma**2))
    return total


def _seasonal(cfg: Config, mode: str, period: int = 40, amp: float = 0.5, **extra: str) -> Config:
    ov = [f"fields.season_period={period}", f"fields.season_amplitude={amp}", f"fields.season_mode={mode}"]
    return apply_overrides(cfg, ov + [f"{k}={v}" for k, v in extra.items()])


def test_season_weights_phases_and_mean(tiny_cfg: Config) -> None:
    pp, period, amp = dims(tiny_cfg).pp, 40, 0.5
    glob, per = _seasonal(tiny_cfg, "global", period, amp), _seasonal(tiny_cfg, "per_patch", period, amp)
    ts = np.arange(period)
    wg = np.stack([_np(fields.season_weights(mx.array(int(t), mx.int32), glob)) for t in ts])
    wp = np.stack([_np(fields.season_weights(mx.array(int(t), mx.int32), per)) for t in ts])
    p_idx = np.arange(pp)
    ref_g = 1 + amp * np.sin(2 * math.pi * ts[:, None] / period + 0 * p_idx)
    ref_p = 1 + amp * np.sin(2 * math.pi * ts[:, None] / period + 2 * math.pi * p_idx / pp)
    assert wg.shape == wp.shape == (period, pp)
    assert np.allclose(wg, ref_g, atol=1e-5) and np.allclose(wp, ref_p, atol=1e-5)
    assert np.allclose(wg, wg[:, :1])  # global: every patch in phase
    assert np.ptp(wp[1]) > 0.1 and not np.allclose(wp[:, 0], wp[:, 1])  # per_patch: phases differ
    assert np.allclose(wg.mean(axis=0), 1.0, atol=1e-5) and np.allclose(wp.mean(axis=0), 1.0, atol=1e-5)
    assert wg.min() >= 1 - amp - 1e-6 and wg.max() <= 1 + amp + 1e-6
    # Periodic in t, also far out (t mod P is taken in integers).
    far = _np(fields.season_weights(mx.array(2_000_000_000 + 7, mx.int32), per))
    ref = _np(fields.season_weights(mx.array((2_000_000_000 + 7) % period, mx.int32), per))
    assert np.array_equal(far, ref)


def test_regen_oscillates_with_period_and_amplitude(tiny_cfg: Config) -> None:
    period, amp = 40, 0.5
    cfg = apply_overrides(
        _seasonal(tiny_cfg, "global", period, amp), ["fields.n_patches=1", "fields.fertility_base=0.05"]
    )
    pos = _patch_pos(cfg)
    g = _gauss_sum(_np(pos), cfg, np.ones(1))
    base = cfg.fields.fertility_base
    series = []
    for t in range(2 * period + 1):
        fert = _np(fields.fertility(pos, cfg, mx.array(t, mx.int32)))
        w = 1 + amp * math.sin(2 * math.pi * t / period)
        assert np.allclose(fert, np.clip(base + w * g, 0, 1), atol=2e-6), t
        rate = _np(fields.regen_rate(pos, cfg, mx.array(t, mx.int32)))
        assert rate.dtype == np.int32
        assert np.abs(rate - np.floor(cfg.fields.regen_max * np.clip(base + w * g, 0, 1))).max() <= 1
        series.append(int(rate.astype(np.int64).sum()))
    s = np.array(series)
    assert np.array_equal(s[:period], s[period : 2 * period])  # exact period
    assert s.argmax() in (period // 4, period + period // 4) or s[period // 4] >= s.max() - 2
    assert s[period // 4] > s[0] > s[3 * period // 4]  # peak, mid, trough
    assert s[period // 4] > 1.2 * s[3 * period // 4]  # amplitude is visible
    # Time average: step=None (carrying-capacity estimate) is the unweighted w = 1 field, i.e. no seasons.
    off = apply_overrides(cfg, ["fields.season_period=0", "fields.season_amplitude=0.0"])
    assert np.array_equal(_np(fields.fertility(pos, cfg)), _np(fields.fertility(pos, off)))
    assert np.array_equal(_np(fields.regen_rate(pos, cfg)), _np(fields.regen_rate(pos, off)))
    # w(0) = 1 with a global phase of zero: step 0 equals the unweighted field.
    assert np.array_equal(
        _np(fields.fertility(pos, cfg, mx.array(0, mx.int32))), _np(fields.fertility(pos, off))
    )


def test_per_patch_phases_differ_in_the_field(tiny_cfg: Config) -> None:
    period = 40
    glob, per = _seasonal(tiny_cfg, "global", period, 0.8), _seasonal(tiny_cfg, "per_patch", period, 0.8)
    pos = _patch_pos(glob)
    t = mx.array(period // 4, mx.int32)  # global peak
    fg_, fp_ = _np(fields.fertility(pos, glob, t)), _np(fields.fertility(pos, per, t))
    assert not np.array_equal(fg_, fp_)
    w = _np(fields.season_weights(t, per))
    ref = np.clip(per.fields.fertility_base + _gauss_sum(_np(pos), per, w), 0, 1)
    assert np.allclose(fp_, ref, atol=2e-6)


def test_seasons_off_is_bit_identical(tiny_cfg: Config) -> None:
    pos = _patch_pos(tiny_cfg)
    t = mx.array(123, mx.int32)
    base = _np(fields.fertility(pos, tiny_cfg))
    assert np.array_equal(base, _np(fields.fertility(pos, tiny_cfg, t)))
    # Amplitude 0 (period set) and period 0 (amplitude set via a direct replace) are both "off".
    zero_amp = apply_overrides(tiny_cfg, ["fields.season_period=50"])
    assert np.array_equal(base, _np(fields.fertility(pos, zero_amp, t)))
    res = mx.full((dims(tiny_cfg).fg,) * 2, 100, mx.int32)
    r0, i0 = fields.update_resource(res, pos, tiny_cfg)
    r1, i1 = fields.update_resource(res, pos, tiny_cfg, t)
    assert np.array_equal(_np(r0), _np(r1)) and np.array_equal(_np(i0), _np(i1))


def test_update_resource_is_integer_and_conserving_with_seasons(tiny_cfg: Config) -> None:
    cfg = _seasonal(tiny_cfg, "per_patch", 40, 0.9)
    pos, d = _patch_pos(cfg), dims(cfg)
    res = mx.array(
        np.random.default_rng(1).integers(0, cfg.fields.resource_cap, (d.fg, d.fg)).astype(np.int32)
    )
    seen = set()
    for t in (0, 10, 20, 30):
        out, inc = fields.update_resource(res, pos, cfg, mx.array(t, mx.int32))
        assert out.dtype == inc.dtype == mx.int32
        assert int(_np(out).astype(np.int64).sum()) == int(_np(res).astype(np.int64).sum()) + int(
            _np(inc).astype(np.int64).sum()
        )
        seen.add(int(_np(inc).astype(np.int64).sum()))
    assert len(seen) > 1


def test_seasons_whole_step_ledger_closes(tiny_cfg: Config) -> None:
    cfg = _seasonal(tiny_cfg, "per_patch", 16, 0.8)
    s, hist = _run(cfg, 40)  # full_check each step includes the exact energy ledger
    regen = np.array([int(_np(out.ledger)[LedgerIdx.INJECT_REGEN]) for _, _, out in hist])
    assert regen.min() != regen.max()
    cum = _np(s.ledger_cum)
    assert int(cum[LedgerIdx.INJECT_REGEN]) == int(regen.sum())
    s_off, _ = _run(tiny_cfg, 40)
    assert state_digest(s_off) != state_digest(s)


def test_both_features_on_pass_checks_and_are_deterministic(tiny_cfg: Config) -> None:
    cfg = apply_overrides(
        _seasonal(tiny_cfg, "global", 12, 1.0),
        [
            "population.immigration_every=4",
            "population.immigration_count=3",
            "population.immigration_floor=40",
        ],
    )
    s, hist = _run(cfg, 30)
    imm = [int(_np(out.counters)[C.IMMIGRANTS]) for _, _, out in hist]
    births = [int(_np(out.counters)[C.BIRTHS]) for _, _, out in hist]
    m = dims(cfg).m
    assert imm == [min(3, m - b) if (i + 1) % 4 == 0 else 0 for i, b in enumerate(births)]
    assert sum(imm) >= 15
    s2, _ = _run(cfg, 30)
    assert state_digest(s2) == state_digest(s)
