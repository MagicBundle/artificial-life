"""Season-aware extinction risk (rules version 2; DESIGN §26.5, Review-log row I43).

Under seasons a population that follows the capacity falls by a large fraction every period without a crash.
Rules version 1 read that fall as a crash (the evaluation's arms C fired 34 and 38 extinction rescues on it);
version 2 models the fall the seasons force and sets the flag on the excess only. The tests cover the model
against a per-step reference (with and without a regen pulse), the engine's capacity table against the
director's own ``K_est``, the tracker's flag on synthetic series (a seasonal one that no longer fires, a crash
on a trough that does, also on top of a regen pulse, no seasons identical to version 1), the rule's reason and
journal note, and the director's end of it (the table is computed only when it is needed, a resumed version-1
run keeps its rules).
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from alife import metrics, runner
from alife.config import Config, apply_overrides, is_3d
from alife.director import Director, log, prompt, rules, seasons
from alife.director.rules import RULE_PARAMS
from alife.director.vitals import Tracker, Vitals
from alife.runio import RunDir, load_checkpoint, read_jsonl
from alife.state import init_state, state_digest
from test_director_rules import ctx, vitals

N = seasons.TABLE_SIZE
R, W = 1000, RULE_PARAMS.risk_fall_window
V1 = dataclasses.replace(RULE_PARAMS, rules_version=1)


def sine_shape(a: float) -> seasons.CapacityShape:
    """A table that is exactly a sine of amplitude ``a`` (what a lattice without base fertility and clipping
    would give), so the reference below can be written in closed form."""
    return seasons.CapacityShape(a, tuple(1 + a * math.sin(2 * math.pi * j / N) for j in range(N)))


# --- the model against a per-step reference ---------------------------------------------------------------


def reference_forced_fall(a: float, period: int, t: int) -> float:
    """The fall a sine capacity forces at boundary ``t``, step by step: the vital sign's own formula
    (lowest per-step value of the current window over the highest window mean of the fall window)."""
    s = np.arange(t - W - R + 1, t + 1)
    c = 1 + a * np.sin(2 * np.pi * s / period)
    ends = range(t, t - W, -R)  # the windows (h - R, h] with h > t - W
    means = [c[(s > h - R) & (s <= h)].mean() for h in ends]
    return 1 - c[(s > t - R) & (s <= t)].min() / max(means)


def model_forced_fall(a: float, period: int, t: int) -> float:
    windows = [((a, period), h - R, h) for h in range(t - W + R, t + 1, R)]
    return seasons.forced_fall(sine_shape(a), windows)


@pytest.mark.parametrize(
    ("a", "period"), [(0.4, 45_000), (0.55, 12_000), (0.5, 3_056), (0.3, 8_621), (0.4, 400_000)]
)
def test_the_forced_fall_matches_the_per_step_reference_at_every_phase(a: float, period: int) -> None:
    worst = 0.0
    for t in range(120_000, 120_000 + period + 1, max(R, period // 40) // R * R):
        ref, got = reference_forced_fall(a, period, t), model_forced_fall(a, period, t)
        worst = max(worst, abs(ref - got))
    assert worst < 1e-2  # the 16 samples of a window and the table's linear interpolation (32 phases)


def test_the_forced_fall_is_the_full_swing_at_a_trough_and_zero_at_a_peak() -> None:
    a, period = 0.4, 45_000
    trough = (
        3 * period // 4 + 2 * period
    )  # sin = -1; a long period, so the fall window holds the peak before it
    peak = period // 4 + 2 * period  # the window before it holds the trough, not a higher value than the peak
    assert model_forced_fall(a, period, trough) == pytest.approx(2 * a / (1 + a), abs=6e-3)
    assert model_forced_fall(a, period, peak) == pytest.approx(0.0, abs=6e-3)


Pulse = tuple[int, int, float]  # (on, off, regen multiplier) over (on, off]; multiples of R, like boundaries


def pulse_windows(a: float, period: int, t: int, pulse: Pulse | None) -> list[tuple]:
    """The rule windows of the fall window ending at ``t``, each with the multiplier it ran with."""
    on, off, m = pulse or (0, 0, 1.0)
    return [
        ((a, period), h - R, h, m if on <= h - R and h <= off else 1.0) for h in range(t - W + R, t + 1, R)
    ]


def reference_levers_fall(a: float, period: int, t: int, pulse: Pulse) -> tuple[float, float]:
    """Step by step: the fall that the seasons and the pulse together force, and the seasons' alone."""
    s = np.arange(t - W - R + 1, t + 1)
    on, off, m = pulse
    base = 1 + a * np.sin(2 * np.pi * s / period)
    out = []
    for c in (base * np.where((s > on) & (s <= off), m, 1.0), base):
        means = [c[(s > h - R) & (s <= h)].mean() for h in range(t, t - W, -R)]
        out.append(1 - c[(s > t - R) & (s <= t)].min() / max(means))
    return out[0], out[1]


@pytest.mark.parametrize(
    ("pulse", "differs"),
    [
        ((162_000, 174_000, 1.4), "levers"),  # begun after a peak, running through the next trough
        ((150_000, 162_000, 1.4), "seasons"),  # over before the trough: the drop back is no season's
        ((160_000, 172_000, 0.6), "seasons"),  # a pulse below 1 (a damper's, or a random draw's)
        ((100_000, 200_000, 2.0), None),  # over the whole fall window: the same ratios either way
    ],
)
def test_the_forced_fall_is_the_lower_of_what_seasons_and_pulse_force_and_the_seasons_alone(
    pulse: Pulse, differs: str | None
) -> None:
    a, period = 0.4, 45_000
    lower = {"levers": False, "seasons": False}
    for t in range(150_000, 150_000 + period + 1, 3_000):
        together, alone = reference_levers_fall(a, period, t, pulse)
        got = seasons.forced_fall(sine_shape(a), pulse_windows(a, period, t, pulse))
        assert got == pytest.approx(min(together, alone), abs=1e-2), t
        lower["levers"] |= together < alone - 0.05  # the pulse and the seasons force less than the seasons
        lower["seasons"] |= alone < together - 0.05  # the seasons alone force less: the pulse's drop is extra
    assert [k for k, v in lower.items() if v] == ([differs] if differs else [])


def test_a_pulse_begun_after_the_peak_narrows_the_exemption_and_one_that_ended_leaves_it_alone() -> None:
    a, period = 0.4, 45_000
    t = 169_000  # just after the trough at 168 750
    plain = seasons.forced_fall(sine_shape(a), pulse_windows(a, period, t, None))
    assert plain == pytest.approx(2 * a / (1 + a), abs=6e-3)  # the full swing

    def forced(pulse: Pulse) -> float:
        return seasons.forced_fall(sine_shape(a), pulse_windows(a, period, t, pulse))

    # x1.4 from 162 000: the capacity of the last windows is 1.4 times what the seasons give, and the
    # peak before it was not boosted, so the seasons did not force the whole of the fall to the trough
    assert forced((162_000, 174_000, 1.4)) == pytest.approx(1 - 1.4 * (1 - plain), abs=0.03)
    assert forced((162_000, 174_000, 1.4)) < plain - 0.15
    # a pulse over before the trough dropped the capacity back, which no season forced: seasons alone count
    assert forced((130_000, 166_000, 1.4)) == pytest.approx(plain, abs=1e-12)
    # a pulse below 1 begun after the peak makes the fall bigger than the seasons force: not exempt either
    assert forced((160_000, 172_000, 0.6)) == pytest.approx(plain, abs=1e-12)
    # while one that held the peak down means the population had less to fall from: a smaller exemption
    assert forced((130_000, 166_000, 0.6)) < plain - 0.2
    assert forced((100_000, 200_000, 1.4)) == pytest.approx(plain, abs=1e-12)  # over all of it: no change
    assert forced((164_000, 175_000, 1.0)) == plain  # a multiplier of 1 is no pulse, exactly


def test_the_regeneration_multiplier_is_read_from_a_rows_env_and_scales_a_windows_capacity() -> None:
    assert seasons.regen_mult_of({"regen_mult": 1.4}) == 1.4
    for absent in (None, {}, {"regen_mult": None}, {"regen_mult": 0.0}, {"regen_mult": -1.0}):
        assert seasons.regen_mult_of(absent) == 1.0  # nothing recorded, or nothing the lever range allows
    for odd in (float("nan"), float("inf"), True, "1.4"):
        assert seasons.regen_mult_of({"regen_mult": odd}) == 1.0
    shape = sine_shape(0.4)
    mean, low = seasons.window_stats(shape, (0.4, 45_000), 30_000.0, 31_000.0)
    mean2, low2 = seasons.window_stats(shape, (0.4, 45_000), 30_000.0, 31_000.0, 1.5)
    assert (mean2, low2) == pytest.approx((1.5 * mean, 1.5 * low))
    assert seasons.window_stats(shape, None, 0.0, 1_000.0, 1.5) == pytest.approx((1.5, 1.5))


def test_the_capacity_is_the_table_read_periodically_and_scaled_by_the_amplitude() -> None:
    shape = sine_shape(0.5)
    steps = np.array([0.0, 123.0, 4_999.5, 10_000.0, 31_337.0])
    full = seasons.capacity(shape, (0.5, 20_000), steps)
    assert full == pytest.approx(1 + 0.5 * np.sin(2 * np.pi * steps / 20_000), abs=3e-3)
    half = seasons.capacity(shape, (0.25, 20_000), steps)  # another amplitude scales the deviation from 1
    assert half - 1 == pytest.approx((full - 1) / 2, abs=1e-12)
    assert seasons.capacity(shape, None, steps) == pytest.approx(np.ones(5))  # no seasons: nothing forced
    wrapped = seasons.capacity(shape, (0.5, 20_000), steps + 20_000)
    assert wrapped == pytest.approx(full, abs=1e-9)


def test_the_excess_divides_the_forced_ratio_out_of_the_observed_one() -> None:
    assert seasons.excess_fall(0.45, 0.0) == 0.45  # nothing forced: the fall itself, exactly
    assert seasons.excess_fall(0.1 + 0.2, 0.0) == 0.1 + 0.2
    assert seasons.excess_fall(0.42, 0.42) == pytest.approx(0.0, abs=1e-12)  # all of it is the seasons'
    assert seasons.excess_fall(0.3, 0.42) == 0.0  # the population did better than the model: never negative
    assert seasons.excess_fall(0.75, 0.50) == pytest.approx(0.5)  # a fall to 25% where 50% is forced
    assert seasons.excess_fall(0.9, 0.75) == pytest.approx(0.6)  # to a tenth where a quarter is forced
    assert seasons.excess_fall(0.5, 1.0) == 0.0  # the seasons force the capacity to zero: nothing to judge


def test_a_regime_is_read_from_a_rows_env() -> None:
    assert seasons.regime_of({"season_amp": 0.6000000238418579, "season_period": 45_000}) == (
        0.6000000238418579,
        45_000,
    )
    assert seasons.regime_of({"season_amp": 0.0, "season_period": 45_000}) is None
    assert seasons.regime_of({"season_amp": 0.5, "season_period": 0}) is None
    assert seasons.regime_of({}) is None and seasons.regime_of(None) is None


# --- the engine's capacity table --------------------------------------------------------------------------


def seasons_cfg(base: Config, *extra: str) -> Config:
    return apply_overrides(base, ["director.enabled=true", "run.director=live", *extra])


def test_the_table_is_the_engines_own_capacity_at_equally_spaced_phases(tiny_cfg: Config) -> None:
    """``table[j]`` is ``K_est`` at the step of phase ``j / n`` over ``K_est`` without seasons, read through
    ``metrics.carrying_capacity`` (the value the director computes at every boundary)."""
    cfg = seasons_cfg(tiny_cfg)
    s = init_state(cfg)
    amp, period, n = 0.6, 3_200, 32
    table = metrics.seasonal_capacity_table(s, cfg, s.env, amp, n)
    on = s.env._replace(
        season_amp=mx.array(amp, dtype=mx.float32), season_period=mx.array(period, dtype=mx.int32)
    )
    base = metrics.carrying_capacity(s, cfg, s.step, s.env)
    assert table.shape == (n,) and table[0] == pytest.approx(1.0)
    for j in range(n):  # the same phase through a different period: equal up to a float32 ulp of the sine
        step = mx.array(j * period // n, dtype=mx.int32)
        assert table[j] == pytest.approx(metrics.carrying_capacity(s, cfg, step, on) / base, rel=1e-4)
    # not a sine of the weight: the base fertility and the clip damp the swing (0.6 would be the plain sine's)
    swing = (table.max() - table.min()) / (table.max() + table.min())
    assert 0.2 < swing < 0.5 and 0.9 < table.mean() < 1.0
    # the table does not depend on the period the state's env carries, nor on regen_mult (a constant factor)
    other = s.env._replace(season_period=mx.array(777, dtype=mx.int32), regen_mult=mx.array(1.5, mx.float32))
    assert metrics.seasonal_capacity_table(s, cfg, other, amp, n) == pytest.approx(table, rel=5e-3)
    # a smaller amplitude gives a smaller swing, of about the same proportion of the amplitude
    half = metrics.seasonal_capacity_table(s, cfg, s.env, amp / 2, n)
    assert (half.max() - half.min()) < 0.6 * (table.max() - table.min())


def test_per_patch_seasons_cancel_in_the_total(tiny_cfg: Config) -> None:
    """With ``season_mode = per_patch`` the patches' sines have phases ``2 pi p / PP``: the lattice's total
    swings far less than with the global mode, which the table (and so the exemption) reflects."""
    glob = seasons_cfg(tiny_cfg)
    per = seasons_cfg(tiny_cfg, "fields.season_mode=per_patch")
    s, sp = init_state(glob), init_state(per)
    g = metrics.seasonal_capacity_table(s, glob, s.env, 0.6, 32)
    p = metrics.seasonal_capacity_table(sp, per, sp.env, 0.6, 32)
    assert (p.max() - p.min()) < 0.35 * (g.max() - g.min())


def test_the_3d_table_reads_the_floor_layer(tiny3d_cfg: Config) -> None:
    cfg = seasons_cfg(tiny3d_cfg)
    s = init_state(cfg)
    amp, period, n = 0.6, 3_200, 32
    table = metrics.seasonal_capacity_table(s, cfg, s.env, amp, n)
    on = s.env._replace(
        season_amp=mx.array(amp, dtype=mx.float32), season_period=mx.array(period, dtype=mx.int32)
    )
    base = metrics.floor_carrying_capacity(s, cfg, s.step, s.env)
    for j in range(n):
        step = mx.array(j * period // n, dtype=mx.int32)
        assert table[j] == pytest.approx(metrics.floor_carrying_capacity(s, cfg, step, on) / base, rel=1e-4)


def test_the_table_size_must_divide_the_probe_period(tiny_cfg: Config) -> None:
    cfg = seasons_cfg(tiny_cfg)
    s = init_state(cfg)
    for bad in (1, 3, 33):
        with pytest.raises(ValueError, match="divides"):
            metrics.seasonal_capacity_table(s, cfg, s.env, 0.5, bad)


def test_a_world_without_regeneration_has_nothing_for_the_seasons_to_modulate(tiny_cfg: Config) -> None:
    cfg = seasons_cfg(tiny_cfg)
    s = init_state(cfg)
    dead = s.env._replace(regen_mult=mx.array(0.0, dtype=mx.float32))
    assert metrics.seasonal_capacity_table(s, cfg, dead, 0.5, 32) == pytest.approx(np.ones(32))


# --- the tracker's flag on synthetic series ---------------------------------------------------------------


def seasonal_rows(
    a: float,
    period: int,
    steps: int,
    *,
    n0: float = 1000.0,
    crash: tuple[int, int, float] | None = None,
    pulse: tuple[int, int, float] | None = None,
    start: int = 0,
    every: int = 100,
) -> list[dict[str, Any]]:
    """Metrics rows of a population that follows the sine capacity ``n0·(1 + a sin(2 pi s / period))`` exactly
    (``crash``: from step ``lo`` to ``hi`` it is multiplied by ``f``), with the regime in each row's
    ``env``. ``pulse = (on, off, m)`` is a regen pulse of multiplier ``m`` over ``(on, off]``: the capacity,
    and so the population, is multiplied by ``m`` there, and every row's ``env`` records the ``regen_mult``
    its interval ran with (``on`` and ``off`` are multiples of ``every``, as a director's boundaries are)."""
    rows = []
    for end in range(start + every, steps + 1, every):
        s = np.arange(end - every + 1, end + 1)
        n = n0 * (1 + a * np.sin(2 * np.pi * s / period))
        if pulse:
            on, off, m = pulse
            n = np.where((s > on) & (s <= off), n * m, n)
        if crash:
            lo, hi, f = crash
            n = np.where((s > lo) & (s <= hi), n * f, n)
        env = {"season_amp": a, "season_period": period}
        if pulse:
            env["regen_mult"] = pulse[2] if pulse[0] <= end - every and end <= pulse[1] else 1.0
        rows.append(
            {
                "step": end,
                "steps": every,
                "n_alive": float(n.mean()),
                "n_alive_min": float(n.min()),
                "env": env,
            }
        )
    return rows


def vitals_series(
    rows: list[dict[str, Any]], params: rules.RuleParams, a: float, k_est: float = 900.0
) -> list[Vitals]:
    """The vitals at every rule boundary, as a live director of ``params`` would compute them (it hands the
    tracker the capacity shape only when seasons are in force in the fall window)."""
    tr, out = Tracker(params), []
    for r in rows:
        tr.observe(r)
        if r["step"] % R == 0:
            bd = tr.close(r["step"])
            shape = sine_shape(a) if params.season_aware and tr.season_amp(bd) > 0 else None
            out.append(tr.vitals(bd, k_est, 10**6, None, shape))
    return out


@pytest.mark.parametrize(("a", "period"), [(0.4, 45_000), (0.55, 12_000), (0.5, 3_056), (0.4, 15_000)])
def test_a_seasonal_population_without_a_crash_no_longer_sets_the_flag(a: float, period: int) -> None:
    rows = seasonal_rows(a, period, 400_000)
    old, new = vitals_series(rows, V1, a), vitals_series(rows, RULE_PARAMS, a)
    assert sum(v.extinction_risk for v in old) > 150  # version 1 reads every trough as a crash
    assert max(v.fall for v in old if v.fall is not None) > 0.5
    assert not any(v.extinction_risk for v in new)
    assert max(v.fall_excess for v in new if v.fall_excess is not None) < 0.05
    assert [v.fall for v in new] == [v.fall for v in old]  # ``fall`` stays the observed value in both
    # what the seasons force is logged next to it, and the compact form gains exactly those two keys
    late = new[-1]
    assert late.fall_seasonal == pytest.approx(late.fall, abs=0.05)
    keys = list(late.compact())
    assert keys[keys.index("fall") + 1 : keys.index("fall") + 3] == ["fall_seasonal", "fall_excess"]


def test_a_crash_on_a_trough_still_sets_the_flag_on_the_excess() -> None:
    a, period = 0.4, 45_000
    lo = 3 * period // 4 + 3 * period  # the trough of the fourth cycle
    for f, excess in ((0.6, 0.4), (0.5, 0.5), (0.3, 0.7)):
        rows = seasonal_rows(a, period, 400_000, crash=(lo, lo + 3_000, f))
        new = vitals_series(rows, RULE_PARAMS, a)
        hit = [v for v in new if v.extinction_risk]
        assert hit and all(lo < v.step <= lo + 3_500 for v in hit), f
        assert max(v.fall_excess for v in hit) == pytest.approx(excess, abs=0.02)
        assert all(v.fall > v.fall_excess for v in hit)  # the seasons' part is taken off
    # a crash of 20% is below the 30% threshold of a world without seasons too: not a rescue
    mild = vitals_series(seasonal_rows(a, period, 400_000, crash=(lo, lo + 3_000, 0.8)), RULE_PARAMS, a)
    assert not any(v.extinction_risk for v in mild)


@pytest.mark.parametrize("on", [159_000, 162_000, 165_000])  # 9 750, 6 750 and 3 750 steps before the trough
@pytest.mark.parametrize("f", [0.6, 0.5])
def test_a_crash_on_a_trough_inside_a_regen_pulse_begun_after_the_peak_still_sets_the_flag(
    on: int, f: float
) -> None:
    """A boom (a x1.4 pulse of the model, of a persona, of a random draw) begun after the last peak raises the
    capacity the population follows, so the seasons' share of the fall to the trough is smaller than the
    seasons alone would force. The exemption must follow: a crash of 40% or 50% on top of that is a crash."""
    a, period = 0.4, 45_000
    trough = 3 * period // 4 + 3 * period  # 168 750
    rows = seasonal_rows(a, period, 200_000, crash=(166_000, 171_000, f), pulse=(on, on + 12_000, 1.4))
    old, new = vitals_series(rows, V1, a), vitals_series(rows, RULE_PARAMS, a)
    in_crash = [(o, n) for o, n in zip(old, new, strict=True) if 166_000 < n.step <= 171_000]
    assert len(in_crash) == 5 and trough in range(166_000, 171_000)
    assert all(o.extinction_risk and n.extinction_risk for o, n in in_crash)
    assert max(n.fall_excess for _, n in in_crash) == pytest.approx(1 - f, abs=0.03)
    # what the seasons and the pulse force is less than the seasons alone force (about 0.57 at a trough)
    assert all(n.fall_seasonal < 0.5 for _, n in in_crash if n.step >= 168_000)
    # without the crash the same pulse and trough raise nothing
    calm = vitals_series(seasonal_rows(a, period, 168_000, pulse=(on, on + 12_000, 1.4)), RULE_PARAMS, a)
    assert not any(v.extinction_risk for v in calm)


def test_the_end_of_a_pulse_is_still_a_fall_the_seasons_did_not_force() -> None:
    """A x2 pulse that ends just after a peak drops the capacity to half, which no season forced: the flag is
    set as in version 1 (DESIGN §26.5 limit (ii)), the exemption not being widened by the pulse's own drop."""
    a, period = 0.4, 45_000
    peak = 146_250  # the pulse ends at the boundary after it
    pulse = (120_000, 147_000, 2.0)
    rows = seasonal_rows(a, period, 160_000, pulse=pulse)
    old, new = (next(v for v in vitals_series(rows, p, a) if v.step == 148_000) for p in (V1, RULE_PARAMS))
    assert peak < pulse[1] < 148_000
    assert old.fall > 0.4 and old.extinction_risk
    assert new.fall_seasonal < 0.05 and new.fall_excess == pytest.approx(new.fall, abs=0.05)
    assert new.extinction_risk


def test_a_pulse_over_the_whole_fall_window_changes_nothing() -> None:
    a, period = 0.4, 45_000
    plain = vitals_series(seasonal_rows(a, period, 300_000), RULE_PARAMS, a)
    boosted = vitals_series(seasonal_rows(a, period, 300_000, pulse=(0, 300_000, 1.4)), RULE_PARAMS, a)
    assert not any(v.extinction_risk for v in boosted)
    assert [v.fall_seasonal for v in boosted] == pytest.approx([v.fall_seasonal for v in plain], abs=1e-9)
    assert [v.fall_excess for v in boosted] == pytest.approx([v.fall_excess for v in plain], abs=1e-9)


def test_the_same_observed_fall_is_exempt_at_a_trough_and_flagged_at_a_peak() -> None:
    """A fall of about 0.45 within the fall window: at a trough of amplitude 0.4 the seasons force more than
    that; at a peak they force nothing and a fall of the same size is a crash. The model is phase aware."""
    a, period = 0.4, 40_000
    trough, peak = 3 * period // 4 + period, period // 4 + 2 * period  # 70 000 and 90 000: rule boundaries
    at_trough = vitals_series(seasonal_rows(a, period, trough), RULE_PARAMS, a)[-1]
    crash = (peak - 1_000, peak, 0.55)  # the last window of the cycle's rise is at 55% of what it was
    at_peak = vitals_series(seasonal_rows(a, period, peak, crash=crash), RULE_PARAMS, a)[-1]
    assert at_trough.step == trough and at_peak.step == peak
    assert at_trough.fall > 0.3 and not at_trough.extinction_risk
    assert at_peak.fall > 0.3 and at_peak.extinction_risk
    assert at_peak.fall_seasonal < 0.05 < at_trough.fall_seasonal


def test_a_regime_switched_on_inside_the_window_forces_its_first_fall() -> None:
    """Seasons switched on at a phase where the weight is 1 (so the capacity is continuous) fall to their
    trough a quarter of a period later; that fall is the new regime's, though earlier windows ran without."""
    a, period = 0.4, 20_000
    t_on = (
        100_000 + period // 2
    )  # sin(11 pi) = 0, and the weight then falls to the trough at t_on + period / 4
    flat = {"season_amp": 0.0, "season_period": 0}
    rows = [
        {"step": s, "steps": 100, "n_alive": 1000.0, "n_alive_min": 1000, "env": flat}
        for s in range(100, t_on + 1, 100)
    ]
    rows += seasonal_rows(a, period, t_on + period, start=t_on)
    t = t_on + period // 4
    new = next(v for v in vitals_series(rows, RULE_PARAMS, a) if v.step == t)
    old = next(v for v in vitals_series(rows, V1, a) if v.step == t)
    assert old.fall == new.fall and new.fall == pytest.approx(a, abs=0.02)  # 1 -> 1 - a
    assert old.extinction_risk and not new.extinction_risk
    assert new.fall_seasonal == pytest.approx(new.fall, abs=0.02)


def test_without_seasons_the_vitals_are_those_of_version_1() -> None:
    """The byte-identical claim at the tracker: rows of a world without seasons give the same ``Vitals`` under
    both versions, whether or not a shape is passed, and the compact log keys are the build-step-2 ones."""
    rng = np.random.default_rng(3)
    n = 1000 * np.exp(rng.normal(0, 0.03, 200))
    n[60:90] *= 0.55  # a crash and a recovery
    rows = [
        {
            "step": 100 * (i + 1),
            "steps": 100,
            "n_alive": float(x),
            "n_alive_min": float(x * 0.97),
            "env": {"season_amp": 0.0, "season_period": 0},
        }
        for i, x in enumerate(n)
    ]
    a, b = (vitals_series(rows, p, 0.5) for p in (V1, RULE_PARAMS))
    assert a == b and any(v.extinction_risk for v in a) and not all(v.extinction_risk for v in a)
    assert all(v.fall_seasonal is None and v.fall_excess is None for v in b)
    assert [list(v.compact()) for v in b][-1] == [
        "n_alive",
        "k_est",
        "n_over_k",
        "fall",
        "extinction_risk",
        "top_clade_share",
        "novelty",
        "coverage",
        "new_cells_200k",
        "b_shift",
    ]
    # a shape handed to a tracker whose windows ran without seasons forces nothing: the same decisions
    tr = Tracker(RULE_PARAMS)
    flags = []
    for r in rows:
        tr.observe(r)
        if r["step"] % R == 0:
            v = tr.vitals(tr.close(r["step"]), 900.0, 10**6, None, sine_shape(0.5))
            flags.append((v.extinction_risk, v.fall_seasonal, v.fall_excess == v.fall))
    assert flags == [(v.extinction_risk, 0.0, True) for v in a]


def test_a_tracker_of_version_1_ignores_a_shape_it_is_given() -> None:
    tr = Tracker(V1)
    for r in seasonal_rows(0.4, 45_000, 100_000):
        tr.observe(r)
        if r["step"] % R == 0:
            v = tr.vitals(tr.close(r["step"]), 900.0, 10**6, None, sine_shape(0.4))
            assert v.fall_seasonal is None and v.fall_excess is None


def test_the_window_regime_comes_from_the_rows_and_survives_a_rebuild() -> None:
    rows = seasonal_rows(0.6000000238418579, 45_000, 3_000)
    tr = Tracker(RULE_PARAMS)
    for r in rows:
        tr.observe(r)
        if r["step"] % R == 0:
            tr.close(r["step"])
    assert [h.regime for h in tr.history] == [(0.6000000238418579, 45_000)] * 3
    assert [h.regen_mult for h in tr.history] == [1.0] * 3  # the rows record no regen_mult: neutral
    assert tr.season_amp(tr.history[-1]) == 0.6000000238418579
    plain = Tracker(RULE_PARAMS)
    plain.observe({"step": 100, "steps": 100, "n_alive": 5.0})  # a row without ``env``: no seasons
    assert plain.close(100).regime is None and plain.season_amp(plain.history[-1]) == 0.0
    boosted = Tracker(RULE_PARAMS)  # the window's last row speaks for it (the interval's multiplier)
    boosted.observe({"step": 50, "steps": 50, "n_alive": 5.0, "env": {"regen_mult": 1.0}})
    boosted.observe({"step": 100, "steps": 50, "n_alive": 5.0, "env": {"regen_mult": 1.5}})
    bd = boosted.close(100)
    assert bd.regen_mult == 1.5 and bd.regime is None


def test_season_amp_looks_back_only_over_the_fall_window() -> None:
    tr = Tracker(RULE_PARAMS)
    rows = seasonal_rows(0.5, 10_000, 60_000)[:100]  # seasons for the first 10 000 steps
    rows += [
        {"step": s, "steps": 100, "n_alive": 10.0, "n_alive_min": 10, "env": {}}
        for s in range(10_100, 70_001, 100)
    ]
    seen = {}
    for r in rows:
        tr.observe(r)
        if r["step"] % R == 0:
            seen[r["step"]] = tr.season_amp(tr.close(r["step"]))
    assert seen[10_000] == 0.5 and seen[59_000] == 0.5  # the last seasonal window, 10 000, is within 50 000
    assert seen[60_000] == 0.0  # at 60 000 it has left the window (h > t - 50 000)


# --- the rule: reason and journal note --------------------------------------------------------------------


def seasonal_vitals(t: int, fall: float, forced: float) -> Vitals:
    excess = seasons.excess_fall(fall, forced)
    risk = excess > RULE_PARAMS.risk_fall
    return vitals(t, fall=fall, fall_seasonal=forced, fall_excess=excess, extinction_risk=risk)


def test_an_exempt_fall_makes_no_rescue_and_leaves_a_note_saying_by_how_much() -> None:
    dec = rules.decide(ctx(v=seasonal_vitals(300_000, 0.45, 0.42)))
    assert dec.proposals == ()
    (note,) = dec.notes
    assert note.startswith(
        "seasonal_exemption (fall 0.45 within 50,000 steps; the environment forces 0.42, excess "
    )
    assert note.endswith("): no extinction rescue")
    assert note.split(" (", 1)[0] == rules.SEASONAL_EXEMPTION  # the key the journal dedupes on
    assert "excess 0.052" in note


def test_a_rescue_on_the_excess_names_the_seasons_share_in_its_reason() -> None:
    dec = rules.decide(ctx(v=seasonal_vitals(300_000, 0.82, 0.42)))
    assert {p.action for p in dec.proposals} == {"regen_pulse", "immigration_pulse"}
    reasons = {p.reason for p in dec.proposals}
    assert reasons == {"n/K_est 0.9, fall 0.82 (the environment forces 0.42, excess 0.69) (override)"}
    assert dec.notes == ()


def test_without_the_split_the_reasons_and_notes_are_those_of_version_1() -> None:
    crash = rules.decide(ctx(vit={"fall": 0.5, "extinction_risk": True}))
    assert {p.reason for p in crash.proposals} == {"n/K_est 0.9, fall 0.5 (override)"}
    calm = rules.decide(ctx(vit={"fall": 0.5}))  # a fall above the threshold the flag did not read: no note
    assert calm.proposals == () and calm.notes == ()


def test_a_fall_that_never_crossed_the_threshold_leaves_no_note_even_with_the_split() -> None:
    dec = rules.decide(ctx(v=seasonal_vitals(300_000, 0.2, 0.15)))
    assert dec == rules.Decision((), ())


def test_a_low_ratio_still_rescues_inside_an_exempt_fall() -> None:
    """The n/K criterion is unchanged: seasons do not exempt a population far under its capacity."""
    v = dataclasses.replace(seasonal_vitals(300_000, 0.45, 0.42), n_over_k=0.1, extinction_risk=True)
    dec = rules.decide(ctx(v=v))
    assert {p.action for p in dec.proposals} == {"regen_pulse", "immigration_pulse"}
    assert "(the environment forces 0.42, excess 0.052)" in next(iter(dec.proposals)).reason


def test_the_exemption_is_noted_once_per_run_of_boundaries(tiny_cfg: Config, tmp_path: Path) -> None:
    cfg = seasons_cfg(tiny_cfg, "director.rule_every=100")
    rd = RunDir.create(tmp_path / "run", cfg)
    d = Director(rd, cfg, "live", 0)
    note = (
        "seasonal_exemption (fall 0.45 within 50,000 steps; the environment forces 0.42, excess 0.052): no x"
    )
    d._journal(100, [], [note])
    d._journal(200, [], [note])  # a persisting condition is noted once
    d._journal(300, [], [])
    d._journal(400, [], [note])  # and again when it comes back
    lines = rd.read_jsonl("journal.jsonl")
    assert [ln["step"] for ln in lines] == [100, 400]


def test_the_prompt_names_the_seasons_share_only_when_there_is_one() -> None:
    plain = prompt._vitals_text(ctx(), RULE_PARAMS)
    seasonal = prompt._vitals_text(ctx(v=seasonal_vitals(300_000, 0.45, 0.42)), RULE_PARAMS)
    assert "environment forces" not in plain
    assert "fall within 50k steps 0.01; extinction-risk flag no" in plain
    want = (
        "fall within 50k steps 0.45 (the environment forces 0.42 of it; the flag reads the excess 0.0517); "
    )
    assert want + "extinction-risk flag no" in seasonal
    note = rules.decide(ctx(v=seasonal_vitals(300_000, 0.45, 0.42))).notes
    assert "- not taken: seasonal_exemption (fall 0.45" in prompt._taken_text([], note)


# --- the director: when the table is computed, and which rules a run lives by -----------------------------


def live_director(cfg: Config, tmp_path: Path, name: str = "run") -> tuple[Director, RunDir]:
    rd = RunDir.create(tmp_path / name, cfg)
    return Director(rd, cfg, "live", 0), rd


def feed(d: Director, rows: list[dict[str, Any]], until: int) -> None:
    for r in rows:
        if r["step"] > until:
            break
        d.observe(r)
        if r["step"] < until and d.due(r["step"]):
            d.tracker.close(r["step"])


def seasonal_state(cfg: Config, amp: float, period: int) -> Any:
    s = init_state(cfg)
    return s._replace(
        env_season_amp=mx.array(amp, dtype=mx.float32), env_season_period=mx.array(period, dtype=mx.int32)
    )


@pytest.fixture(params=["tiny_cfg", "tiny3d_cfg"])
def cfg_1000(request: pytest.FixtureRequest) -> Config:
    """``tiny`` and ``tiny3d`` with a rule window of 1 000 steps."""
    return seasons_cfg(
        request.getfixturevalue(request.param),
        "director.rule_every=1000",
        "run.metrics_every=100",
        "run.heavy_metrics_every=1000",
        "run.checkpoint_every=1000",
    )


def test_the_director_judges_a_seasonal_population_by_its_excess(
    cfg_1000: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through ``Director._vitals`` with the real engine table: a population that follows the lattice's own
    seasonal capacity is not at risk at a trough, one that crashes on top of it is, and under rules version 1
    the same trough is."""
    amp, period = 0.6, 20_000
    state = seasonal_state(cfg_1000, amp, period)
    table = metrics.seasonal_capacity_table(state, cfg_1000, state.env, amp, N)
    shape = seasons.CapacityShape(amp, tuple(float(x) for x in table))
    t = 3 * period // 4 + 2 * period  # a trough, after two full cycles
    # the capacity the risk test divides by: the one K_est in 2D, the floor layer's in 3D (§27.10)
    k = metrics.floor_carrying_capacity(state, cfg_1000, state.step, state.env)

    def rows(f: float) -> list[dict[str, Any]]:
        out = []
        for end in range(100, t + 1, 100):
            s = np.arange(end - 99, end + 1)
            n = 0.8 * k * seasons.capacity(shape, (amp, period), s) * np.where(s > t - 2_000, f, 1.0)
            env = {"season_amp": amp, "season_period": period}
            out.append(
                {
                    "step": end,
                    "steps": 100,
                    "n_alive": float(n.mean()),
                    "n_alive_min": float(n.min()),
                    "env": env,
                }
            )
        return out

    calls: list[float] = []
    real = metrics.seasonal_capacity_table
    monkeypatch.setattr(
        metrics, "seasonal_capacity_table", lambda *a, **kw: (calls.append(a[3]), real(*a, **kw))[1]
    )
    d, _ = live_director(cfg_1000, tmp_path, "new")
    feed(d, rows(1.0), t)
    v = d._vitals(state, t)
    assert calls == [pytest.approx(amp)]
    assert v.fall > 0.3 and v.fall_seasonal == pytest.approx(v.fall, abs=0.03) and not v.extinction_risk
    assert v.compact()["fall_seasonal"] == round(v.fall_seasonal, 4)
    d, _ = live_director(cfg_1000, tmp_path, "crash")
    feed(d, rows(0.5), t)
    crashed = d._vitals(state, t)
    assert crashed.extinction_risk and crashed.fall_excess == pytest.approx(0.5, abs=0.03)
    # version 1: no table, the observed fall sets the flag at the same trough
    calls.clear()
    rd = RunDir.create(tmp_path / "old", cfg_1000)
    rd.append_jsonl("director.jsonl", [log.make_header(cfg_1000, 0, 2, rules_version=1)])
    old = Director(rd, cfg_1000, "live", 0)
    assert old.params.rules_version == 1
    feed(old, rows(1.0), t)
    v1 = old._vitals(state, t)
    assert (
        calls == [] and v1.fall_seasonal is None and v1.extinction_risk and v1.fall == pytest.approx(v.fall)
    )


def test_no_table_is_computed_without_seasons(
    cfg_1000: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*a: Any, **kw: Any) -> Any:
        raise AssertionError("the capacity table is only for runs under seasons")

    monkeypatch.setattr(metrics, "seasonal_capacity_table", boom)
    d, _ = live_director(cfg_1000, tmp_path)
    state = init_state(cfg_1000)
    rows = [
        {
            "step": s,
            "steps": 100,
            "n_alive": 100.0,
            "n_alive_min": 90,
            "env": {"season_amp": 0.0, "season_period": 0},
        }
        for s in range(100, 5_001, 100)
    ]
    feed(d, rows, 5_000)
    v = d._vitals(state, 5_000)
    assert v.fall_seasonal is None and v.fall_excess is None
    floor = (
        ["k_floor", "n_over_k_floor"] if is_3d(cfg_1000) else []
    )  # the keys of PR 4 in 3D, of step 2 in 2D
    assert list(v.compact()) == [
        "n_alive",
        "k_est",
        "n_over_k",
        *floor,
        "fall",
        "extinction_risk",
        "top_clade_share",
        "novelty",
        "coverage",
        "new_cells_200k",
        "b_shift",
    ]


def test_without_seasons_a_real_run_is_version_1_byte_for_byte(tiny_cfg: Config, tmp_path: Path) -> None:
    """The same world run live under the real rules of version 1 and of version 2 (a rescue fires at the
    first boundary: 20 founders against a capacity of hundreds): the decision log, the journal, the archive
    file, the metrics rows and the final state are identical, the log's header saying 1 and 2 apart."""
    cfg = apply_overrides(
        tiny_cfg,
        [
            "director.enabled=true",
            "run.director=live",
            "director.rule_every=50",
            "run.metrics_every=10",
            "run.heavy_metrics_every=50",
            "run.checkpoint_every=100",
            "run.diversity_sample=64",
            "population.n_init=20",
        ],
    )
    runs = {}
    for name, params in (("v1", V1), ("v2", RULE_PARAMS)):
        rd = RunDir.create(tmp_path / name, cfg)
        res = runner.run(
            rd, cfg, init_state(cfg), 200, quiet=True, director=Director(rd, cfg, "live", 0, params)
        )
        assert res.status == "done"
        runs[name] = rd
    old, new = (read_jsonl(runs[n].path / "director.jsonl") for n in ("v1", "v2"))
    assert (old[0]["rules_version"], new[0]["rules_version"]) == (1, 2)
    assert {**old[0], "rules_version": 2} == new[0] and old[1:] == new[1:]
    assert any(e.get("rule") == "extinction_rescue" for e in new[1:])
    assert all("fall_seasonal" not in e["vitals"] for e in new[1:])
    for name in ("journal.jsonl", "director_archive.json"):
        assert (runs["v1"].path / name).read_bytes() == (runs["v2"].path / name).read_bytes(), name
    perf = {"wall_s", "steps_per_s", "peak_mem_gb"}
    rows = [[{k: v for k, v in r.items() if k not in perf} for r in runs[n].metrics_rows()] for n in runs]
    assert rows[0] == rows[1]
    digests = [state_digest(load_checkpoint(runs[n].checkpoint_at(200), cfg)) for n in runs]
    assert digests[0] == digests[1]
