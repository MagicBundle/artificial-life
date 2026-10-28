"""Throughput, stages, donation health, determinism and brain-variant benchmarks (§15, §21). WP6.

``run_bench`` is the instrument for the §15 performance model. Every run follows the runner's discipline
(§15.4): one blocking ``mx.eval`` per step, and no Python reference to a previous ``State`` survives a step
call, so weight donation is measured as production sees it. The report is JSON-compatible (str, int, float,
bool, None, lists and dicts only); ``format_report`` renders it for a terminal.

- **Throughput:** compile time (first call minus a steady step), ms/step statistics, steps/s, peak memory.
- **Donation health:** after warm-up, the peak memory over ``DONATION_STEPS`` steps above the resident
  baseline; more than ``DONATION_LIMIT`` × the weight bytes means ``W1``/``W2`` are being copied (F1).
- **Stages** (``stages``): the step re-wired as one compiled function per sub-step group, each evaluated on
  its own. The wiring mirrors ``step.step`` exactly (the tests check bitwise equality in eager mode); the
  total is inflated by the extra syncs.
- **Determinism** (``determinism``): two runs of ``steps`` steps from ``init_state`` compare state digests.
- **Brain variants** (``brain_variants``): ``brain.forward`` timed for every ``impl`` × ``weight_dtype``.
- **Observers** (``observe``, §28, §29.12): the step with the game observers (``make_step(observe=True)``,
  ``Observer.update`` and ``Observer.take_records``, evaluations and social snapshots included) against the
  plain step, as blocks (control, observed, observed, control) of the runner's own loop, with a check that
  the trajectories are bitwise equal; and the parts of one social snapshot (``observe.social``).
- **Weather** (``weather``, W1): the step with the engine weather layer on against the same config with it
  off, as blocks (off, on, on, off). The trajectories differ by construction, so none is compared. The
  marginal cost per step is the number STORMS §8 W1 asks for; ``stages`` adds the rows ``wind`` and
  ``weather`` when the layer is on.
- **Play** (``run_play_bench``, G3a, DESIGN §33.5): the play loop of ``alife play`` run headless, once per
  director mode and rate, with the full record set into a draining reader thread (a pipe), and the cost of its
  parts; the gate of DESIGN §33.5.3 is read from it. A separate report: ``alife bench --play``. G3b adds
  the ``player`` director mode, scripted powers (``--play-powers``: a click and the chapter-end step itself,
  with the click-to-applied latency and a digest checked against ``resume --director replay``), and a line
  for the chapter, report and alert work (DESIGN §34.8).
- **Arms** (``arms``, A1, DESIGN §42): the step with the arms layer on against the same config with it off,
  the same (off, on, on, off) blocks; the trajectories differ by construction. The gate of the arms spec
  (§7): below 3 % overhead on 64k3d before the first long run.
- **Weather-org** (``weather_org``, W2): the same blocks with the weather on in both arms, the "off" arm
  having ``seed_gain = shade_gain = 0``: the marginal cost of the organism couplings (the seeding signal and
  the shade count, DESIGN §32).
"""

from __future__ import annotations

import gc
import json
import math
import platform
import statistics
import struct
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, Final, NamedTuple

import mlx.core as mx
import numpy as np

from alife import (
    actions,
    arms,
    bonds,
    brain,
    ext,
    fields,
    invariants,
    lifecycle,
    metrics,
    physics,
    sensing,
    spatial,
)
from alife import energy as en
from alife import step as step_mod
from alife import weather as wthr
from alife.actions import Actions
from alife.config import (
    BRAIN_IMPLS,
    WEIGHT_DTYPES,
    ArmsConfig,
    Config,
    ConfigError,
    WeatherConfig,
    apply_overrides,
    config_hash,
    dims,
)
from alife.rng import Stream, step_keys
from alife.state import (
    ERROR_MASK,
    CounterIdx,
    Diag,
    State,
    age,
    init_state,
    state_digest,
    storm_counter_index,
    to_numpy,
)

DONATION_STEPS: Final = 10
DONATION_LIMIT: Final = 0.25  # healthy when a copying step peaks > 0.25 × bytes(W1 + W2) higher (§15.4)
_SHOW_PROBLEMS: Final = 5  # full_check messages quoted in the report

# Sub-step groups timed by ``stages`` (§3, §15.2), in pipeline order, with their sub-step labels.
STAGES: Final[tuple[tuple[str, str], ...]] = (
    ("flush", "S0"),
    ("neighbors", "S1-S2"),
    ("sense", "S3-S5"),
    ("brain", "S6"),
    ("decode", "S7"),
    ("interact", "S8-S14"),
    ("physics", "S15"),
    ("reproduce", "S16"),
    ("fields", "S17"),
    ("finish", "S18"),
)

# The two rows the weather layer adds (W1): the wind sampling at S3w and the weather update with the rain
# remap at S17w, in pipeline order.
_WIND_STAGE: Final = ("wind", "S3w")
_WEATHER_STAGE: Final = ("weather", "S17w")


def stages_for(cfg: Config) -> tuple[tuple[str, str], ...]:
    """``STAGES`` for a weather-off config (so its report is unchanged), with the ``wind`` row after ``sense``
    and the ``weather`` row before ``fields`` when the weather layer is on."""
    if not cfg.weather.enabled:
        return STAGES
    out: list[tuple[str, str]] = []
    for st in STAGES:
        if st[0] == "fields":
            out.append(_WEATHER_STAGE)
        out.append(st)
        if st[0] == "sense":
            out.append(_WIND_STAGE)
    return tuple(out)


_GB: Final = float(1 << 30)
WEATHER_PAIRS: Final = 2  # off/on block pairs, in the order off on on off (the leading arm alternates)


def run_bench(
    cfg: Config,
    *,
    steps: int = 200,
    warmup: int = 20,
    stages: bool = False,
    determinism: bool = False,
    brain_variants: bool = False,
    observe: bool = False,
    weather: bool = False,
    weather_pairs: int = WEATHER_PAIRS,
    weather_org: bool = False,
    arms: bool = False,
) -> dict[str, Any]:
    """Run the benchmark suite and return a JSON-compatible report.

    ``warmup`` steps (at least the compile call) run before anything is timed; ``steps`` steps are timed.
    The optional parts reuse ``steps`` and ``warmup`` (determinism: 2 × ``steps`` steps; brain variants:
    ``steps`` timed forward calls per variant; observers and weather: ``2 * pairs`` blocks of ``steps // 2``
    timed steps; ``weather`` needs ``cfg.weather.enabled``, ``weather_org`` also a ``seed_gain`` or
    ``shade_gain`` above 0).
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")
    if weather and not cfg.weather.enabled:
        raise ConfigError("bench --weather needs weather.enabled = true (for example --overlay storm)")
    if weather_org and not (
        cfg.weather.enabled and (cfg.weather.seed_gain > 0 or cfg.weather.shade_gain > 0)
    ):
        raise ConfigError(
            "bench --weather-org needs weather.enabled = true and weather.seed_gain or "
            "weather.shade_gain above 0 (for example --overlay storm --set weather.seed_gain=0.03 "
            "--set weather.shade_gain=0.02)"
        )
    if arms and not cfg.arms.enabled:
        raise ConfigError("bench --arms needs arms.enabled = true (for example --set arms.enabled=true)")
    if weather_pairs < 1:
        raise ValueError(f"weather_pairs must be >= 1, got {weather_pairs}")
    report: dict[str, Any] = {**_environment(cfg), "steps": steps, "warmup": max(warmup, 1)}
    fn = step_mod.make_step(cfg)
    report.update(_throughput(fn, cfg, steps, warmup))
    if stages:
        report["stages"] = _bench_stages(cfg, steps, warmup, report["ms_per_step"])
    if determinism:
        report["determinism"] = _determinism(fn, cfg, steps)
    if brain_variants:
        report["brain_variants"] = _brain_variants(cfg, steps, warmup)
    if observe:
        report["observe"] = _bench_observe(cfg, steps, warmup)
    if weather:
        report["weather"] = _bench_weather(cfg, steps, warmup, weather_pairs)
    if weather_org:
        report["weather_org"] = _bench_weather(cfg, steps, warmup, weather_pairs, org=True)
    if arms:
        report["arms"] = _bench_weather(
            cfg, steps, warmup, weather_pairs, off=replace(cfg, arms=ArmsConfig())
        )
    return report


# --- environment ----------------------------------------------------------------------------------------


def _environment(cfg: Config) -> dict[str, Any]:
    import alife

    d = dims(cfg)
    return {
        "alife_version": alife.__version__,
        "mlx_version": mx.__version__,
        "python": platform.python_version(),
        "platform": f"{sys.platform}-{platform.machine()}",
        "device": str(mx.default_device()),
        "knn_impl": spatial.knn_impl(cfg),  # "metal" (the fused kernel) or "mlx" (§4)
        "config_hash": config_hash(cfg),
        "config": {
            "n_max": d.n,
            "n_init": cfg.population.n_init,
            "max_births_per_step": d.m,
            "world_size": cfg.world.size,
            "field_grid": d.fg,
            **({"dims": d.nd, "world_height": cfg.world.height, "field_grid_z": d.fz} if d.d3 else {}),
            "k_neighbors": d.k,
            "cell_capacity": d.c,
            "hidden": d.h,
            "weight_dtype": cfg.brain.weight_dtype,
            "brain_impl": cfg.brain.impl,
            "compile": cfg.run.compile,
            "runtime_checks": cfg.run.runtime_checks,
            "seed": cfg.run.seed,
        },
    }


def _nbytes(*arrays: mx.array) -> int:
    return int(sum(a.nbytes for a in arrays))


def _diag_report(diag: int) -> dict[str, Any]:
    return {
        "diag": diag,
        "diag_flags": [f.name for f in Diag if diag & f.value],
        "diag_ok": not diag & ERROR_MASK,
    }


# --- throughput and donation health -----------------------------------------------------------------------


def _throughput(fn: Callable[[State], Any], cfg: Config, steps: int, warmup: int) -> dict[str, Any]:
    """Compile, warm up, measure donation health, then time ``steps`` steps (one sync eval each)."""
    s = init_state(cfg)
    t0 = time.perf_counter()
    s, out = fn(s)
    mx.eval(s, out)
    first = time.perf_counter() - t0
    for _ in range(max(warmup, 1) - 1):
        s, out = fn(s)
        mx.eval(s, out)

    # Donation health (§15.4), stepped inline: a helper taking ``s`` would keep the caller's reference to
    # the old state alive during its first step and so force the very weight copy it is looking for.
    weights = _nbytes(s.W1, s.W2)
    gc.collect()
    baseline = mx.get_active_memory()
    mx.reset_peak_memory()
    for _ in range(DONATION_STEPS):
        s, out = fn(s)
        mx.eval(s, out)
    peak = mx.get_peak_memory()
    # Calibration: one step that deliberately keeps its input state alive, i.e. the known-copy case.
    gc.collect()
    baseline_ref = mx.get_active_memory()
    mx.reset_peak_memory()
    held = s
    s, out = fn(s)
    mx.eval(s, out)
    peak_ref = mx.get_peak_memory()
    del held
    donation = _donation(peak - baseline, peak_ref - baseline_ref, weights)

    acc = mx.zeros_like(out.counters)
    n_alive_start = int(to_numpy(s.alive.sum()))  # the counters below cover the timed steps only
    times = []
    mx.reset_peak_memory()  # steady-state peak: excludes compile and the calibration step's copy
    t_all = time.perf_counter()
    for _ in range(steps):
        t0 = time.perf_counter()
        s, out = fn(s)
        acc = acc + out.counters
        mx.eval(s, acc)
        times.append(time.perf_counter() - t0)
    wall = time.perf_counter() - t_all
    del out
    peak = mx.get_peak_memory()

    c = to_numpy(acc)
    med = statistics.median(times)
    problems = invariants.full_check(s, cfg)
    return {
        "compile_s": max(first - med, 0.0),
        "first_step_s": first,
        "ms_per_step": 1e3 * wall / steps,
        "ms_per_step_median": 1e3 * med,
        "ms_per_step_p90": 1e3 * float(np.quantile(times, 0.9)),
        "ms_per_step_min": 1e3 * min(times),
        "ms_per_step_max": 1e3 * max(times),
        "steps_per_s": steps / wall,
        "peak_mem_gb": peak / _GB,
        "state_gb": _nbytes(*s) / _GB,
        "weights_gb": _nbytes(s.W1, s.W2) / _GB,
        "n_alive_start": n_alive_start,
        "n_alive_end": int(to_numpy(s.alive.sum())),
        "births": int(c[CounterIdx.BIRTHS]),
        "immigrants": int(c[CounterIdx.IMMIGRANTS]),
        "deaths": int(
            c[CounterIdx.DEATHS_STARVED]
            + c[CounterIdx.DEATHS_KILLED]
            + c[CounterIdx.DEATHS_AGED]
            + c[CounterIdx.DEATHS_CATASTROPHE]
            + (c[storm_counter_index(dims(cfg).nd)] if cfg.weather.enabled else 0)
        ),
        **_diag_report(int(to_numpy(s.diag))),
        "invariants_ok": not problems,
        "invariant_problems": problems[:_SHOW_PROBLEMS],
        "donation": donation,
    }


def _donation(excess: int, excess_copy: int, weights: int) -> dict[str, Any]:
    """The donation-health verdict (§15.4).

    ``excess`` is the peak above the resident baseline over ``DONATION_STEPS`` normal steps, ``excess_copy``
    the same for one step that holds a reference to its input (which forces the weight copy, F1). Donation
    is healthy when the normal steps stay more than ``DONATION_LIMIT`` × the weight bytes below the copying
    step. The step's own transients (neighbour lists, mutation randoms) cancel in the difference: they are
    ≈ 0.3× f32 or 0.6× fp16 weight bytes on CPU, so an absolute bound would misread them as a copy.
    """
    excess, excess_copy = max(excess, 0), max(excess_copy, 0)
    saved = (excess_copy - excess) / weights if weights else 0.0
    ok = saved > DONATION_LIMIT
    return {
        "ok": ok,
        "verdict": "ok: weights updated in place"
        if ok
        else f"WARNING: steps peak within {saved:.2f}x the weight bytes of a step that copies them (needs "
        f"> {DONATION_LIMIT}); W1/W2 are probably being copied every step (F1, §15.4)",
        "steps": DONATION_STEPS,
        "excess_gb": excess / _GB,
        "excess_copy_gb": excess_copy / _GB,
        "weights_gb": weights / _GB,
        "excess_over_weights": excess / weights if weights else 0.0,
        "saved_over_weights": saved,
        "limit": DONATION_LIMIT,
    }


# --- determinism ----------------------------------------------------------------------------------------


def _determinism(fn: Callable[[State], Any], cfg: Config, steps: int) -> dict[str, Any]:
    """Two runs of ``steps`` steps from ``init_state(cfg)``; their SHA-256 state digests must match."""
    digests = []
    for _ in range(2):
        s = init_state(cfg)
        t0 = time.perf_counter()
        for _ in range(steps):
            s, out = fn(s)
            mx.eval(s, out)
        wall = time.perf_counter() - t0
        digests.append((state_digest(s), wall))
        del s, out
    (da, wa), (db, wb) = digests
    return {"steps": steps, "digest_a": da, "digest_b": db, "match": da == db, "wall_s": [wa, wb]}


# --- observers ------------------------------------------------------------------------------------------

OBSERVE_BLOCKS: Final = 2  # control/observed block pairs, in the order C O O C (the leading arm alternates)
SOCIAL_REPS: Final = 10  # timed repetitions of each part of a social snapshot (observe.social)


def _median_ms(fn: Callable[[], Any], reps: int) -> float:
    """Median wall time of ``fn()`` over ``reps`` calls, in ms (``fn`` evaluates what it builds)."""
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return 1e3 * statistics.median(times)


def _bench_observe(cfg: Config, steps: int, warmup: int) -> dict[str, Any]:
    """The cost of the game observers (§28, §29) as the runner pays it, against the plain step.

    Two arms, each following the runner's loop with one blocking eval per step: the *control* is
    ``make_step(cfg)`` and ``mx.eval(state, out)``; the *observed* arm is ``make_step(cfg, observe=True)``
    plus ``Observer.update`` (the labels' evaluation every ``rules.every`` steps and the social snapshot every
    ``social.snapshot_every`` steps included), ``mx.eval(state, out.base(), *obs.arrays())`` and
    ``obs.take_records()`` (the records are dropped: the runner's append and fsync, about 1 ms per snapshot,
    are not timed), so that a snapshot settles at its own step. They run as blocks in the order control,
    observed, observed, control (the leading arm alternates in each pair, so that a drift linear in time, such
    as a thermal one, cancels from the difference of the means; in the order C O C O it would not), each
    restarting from ``init_state(cfg)`` with the same warm-up and the same number of timed steps. The rules
    and the social rules are ``observer.DEFAULT_RULES`` and ``observer.DEFAULT_SOCIAL``, read at call time.

    Nothing that compiles is timed: the observer's closures compile again for each block, so the warm-up
    covers the first evaluation step and the first snapshot step (the first trace of the snapshot closure
    with real ``Acts``, its settle and its relabel): ``warm = max(warmup, 1, rules.every, S + 1)`` with
    ``S = social.snapshot_every``. The timed steps of a block are ``ceil(max(steps // 2, 1) / S) * S``, so
    that they hold a whole number of snapshots, ``steps_per_block / S`` of them.

    The median step times give ``overhead_ms_median``; the evaluation and snapshot steps are slower than the
    others, which a median hides, so ``overhead_ms_mean`` and ``observed_steps_per_s`` are from the means.
    ``snapshot_step_ms_mean`` (steps that are multiples of ``S``: the snapshot's device part and its settle)
    and ``other_step_ms_mean`` split the observed mean. ``trajectory_match`` compares the state digests at the
    end of the first two blocks: the observers must not change the trajectory (§28.2).

    ``social`` times, on a state stepped ``warm`` steps with a fresh observer, ``SOCIAL_REPS`` repetitions of
    each part of one snapshot: ``metrics.components`` alone (``social.colony_iters`` rounds), the snapshot
    closure (device part), ``Society.settle`` on a copy of the society (host part) and the heavy metrics
    (``metrics.snapshot_metrics(heavy=True)``, which the runner computes at the same step when the game
    cadence coincides, §29.9). On ``init_state`` there are few bonds: ``analysis.social`` on an evolved
    checkpoint is the realistic number.
    """
    from alife.game import observer as observer_mod  # local: only this part needs the game package
    from alife.game.observer import Observer

    rules, sr = observer_mod.DEFAULT_RULES, observer_mod.DEFAULT_SOCIAL
    every = sr.snapshot_every
    warm = max(warmup, 1, rules.every, every + 1)
    per_block = -(-max(steps // 2, 1) // every) * every
    control_fn = step_mod.make_step(cfg)
    observed_fn = step_mod.make_step(cfg, observe=True)
    times: dict[str, list[float]] = {"control": [], "observed": [], "snapshot": [], "other": []}
    digests: dict[str, str] = {}

    def block(arm: str) -> None:
        s = init_state(cfg)
        obs = Observer.fresh(s, cfg) if arm == "observed" else None
        for i in range(warm + per_block):
            t0 = time.perf_counter()
            if obs is None:
                s, out = control_fn(s)
                mx.eval(s, out)
            else:
                s, out = observed_fn(s)
                obs.update(s, out.acts, i + 1)
                mx.eval(s, out.base(), *obs.arrays())
                obs.take_records()
            if i >= warm:
                dt = time.perf_counter() - t0
                times[arm].append(dt)
                if obs is not None:
                    times["snapshot" if (i + 1) % every == 0 else "other"].append(dt)
        if arm not in digests:  # the first block of each arm
            digests[arm] = state_digest(s)
        del s, out, obs

    for pair in range(OBSERVE_BLOCKS):  # C O, then O C: the mean block time of both arms is the same
        first, second = ("control", "observed") if pair % 2 == 0 else ("observed", "control")
        block(first)
        block(second)
    ctl, obs_t = times["control"], times["observed"]
    ctl_med, obs_med = statistics.median(ctl), statistics.median(obs_t)
    ctl_mean, obs_mean = statistics.fmean(ctl), statistics.fmean(obs_t)
    snap_t, other_t = times["snapshot"], times["other"]

    return {
        "warmup": warm,
        "block_pairs": OBSERVE_BLOCKS,
        "steps_per_block": per_block,
        "control_ms_median": 1e3 * ctl_med,
        "observed_ms_median": 1e3 * obs_med,
        "overhead_ms_median": 1e3 * (obs_med - ctl_med),
        "control_ms_per_step": 1e3 * ctl_mean,
        "observed_ms_per_step": 1e3 * obs_mean,
        "overhead_ms_mean": 1e3 * (obs_mean - ctl_mean),
        "observed_steps_per_s": 1.0 / obs_mean,
        "snapshot_step_ms_mean": 1e3 * statistics.fmean(snap_t) if snap_t else None,
        "other_step_ms_mean": 1e3 * statistics.fmean(other_t) if other_t else None,
        "trajectory_match": digests["control"] == digests["observed"],
        "social": _bench_social(cfg, observed_fn, warm),
    }


def _bench_social(cfg: Config, observed_fn: Callable[[State], Any], warm: int) -> dict[str, Any]:
    """``SOCIAL_REPS`` timed repetitions of each part of one social snapshot (``observe.social``), on a state
    stepped ``warm`` steps with a fresh observer: ``metrics.components`` alone, the snapshot closure (device),
    ``Society.settle`` on a copy of the society (host) and the heavy metrics."""
    from alife.game import observer as observer_mod
    from alife.game.observer import Observer
    from alife.game.social import soc_in
    from alife.game.society import Society, host_snap

    sr = observer_mod.DEFAULT_SOCIAL
    s = init_state(cfg)
    obs = Observer.fresh(s, cfg)
    for i in range(warm):
        s, out = observed_fn(s)
        obs.update(s, out.acts, i + 1)
        mx.eval(s, out.base(), *obs.arrays())
        obs.take_records()
    si, so, acts = soc_in(s), obs.so, out.acts
    mx.eval(*so)  # the last settle's relabel is lazy

    def comps(partner: mx.array, alive: mx.array) -> tuple[mx.array, mx.array]:
        return metrics.components(partner, alive, sr.colony_iters)

    comp_fn = mx.compile(comps) if cfg.run.compile else comps
    mx.eval(comp_fn(s.partner, s.alive))  # compile, untimed
    components_ms = _median_ms(lambda: mx.eval(comp_fn(s.partner, s.alive)), SOCIAL_REPS)
    arch = obs.o.label_arch[:, 0]
    snap = obs._snap_fn(so.kin, so, si, acts, arch)
    mx.eval(*snap)
    device_ms = _median_ms(lambda: mx.eval(*obs._snap_fn(so.kin, so, si, acts, arch)), SOCIAL_REPS)
    h = host_snap(snap)
    arrays, step_next = obs.society.arrays(), obs.society.snapshot_step + sr.snapshot_every
    settle_times = []
    for _ in range(SOCIAL_REPS):
        copy, reason = Society.from_arrays(arrays, cfg, sr, step=obs.step)
        if copy is None:
            raise RuntimeError(f"the society does not round-trip through its sidecar form: {reason}")
        t0 = time.perf_counter()
        copy.settle(h, step_next)
        settle_times.append(time.perf_counter() - t0)
    host_ms = 1e3 * statistics.median(settle_times)
    metrics.snapshot_metrics(s, cfg, heavy=True)  # compile, untimed
    heavy_ms = _median_ms(lambda: metrics.snapshot_metrics(s, cfg, heavy=True), SOCIAL_REPS)
    return {
        "snapshot_every": sr.snapshot_every,
        "reps": SOCIAL_REPS,
        "components_ms_median": components_ms,
        "snapshot_device_ms_median": device_ms,
        "snapshot_host_ms_median": host_ms,
        "heavy_ms_median": heavy_ms,
        "amortised_ms_per_step": (device_ms + host_ms) / sr.snapshot_every,
    }


def _bench_weather(
    cfg: Config,
    steps: int,
    warmup: int,
    pairs: int = WEATHER_PAIRS,
    *,
    org: bool = False,
    off: Config | None = None,
) -> dict[str, Any]:
    """The marginal cost of the engine weather layer: the compiled step with ``cfg`` (weather on) against the
    same config with the layer off (``replace(cfg, weather=WeatherConfig())``), STORMS §8 W1. With ``org``
    the reference keeps the weather on and sets ``seed_gain = shade_gain = 0``: the marginal cost of the
    organism couplings (DESIGN §32).

    Each arm follows the runner's loop (one blocking eval per step). The blocks run off, on, on, off for two
    pairs (the leading arm alternates in each pair, so that a drift linear in time, such as a thermal one,
    cancels from the difference of the means), each from ``init_state`` with the same warm-up and
    ``max(steps // 2, 1)`` timed steps; each arm's step compiles once. The trajectories differ by
    construction (the wind moves agents), so none is compared: the cost is of the step the config describes.
    ``marginal_ms_median`` is the difference of the median step times over all blocks of each arm,
    ``marginal_ms_mean`` of the means; ``pair_marginal_ms`` the same median difference within each pair.
    """
    if off is not None:  # the reference of another layer (arms): ``off`` is the config with it off
        cfg_off = off
    elif org:
        cfg_off = replace(cfg, weather=replace(cfg.weather, seed_gain=0.0, shade_gain=0.0))
    else:
        cfg_off = replace(cfg, weather=WeatherConfig())
    fns = {"off": step_mod.make_step(cfg_off), "on": step_mod.make_step(cfg)}
    cfgs = {"off": cfg_off, "on": cfg}
    warm, per_block = max(warmup, 1), max(steps // 2, 1)
    times: dict[str, list[float]] = {"off": [], "on": []}
    pair_times: list[dict[str, list[float]]] = []

    def block(arm: str, into: list[float]) -> None:
        s = init_state(cfgs[arm])
        for i in range(warm + per_block):
            t0 = time.perf_counter()
            s, out = fns[arm](s)
            mx.eval(s, out)
            if i >= warm:
                into.append(time.perf_counter() - t0)
        del s, out

    for pair in range(pairs):  # off on, then on off, ...: the mean block time of both arms is the same
        cur: dict[str, list[float]] = {"off": [], "on": []}
        for arm in ("off", "on") if pair % 2 == 0 else ("on", "off"):
            block(arm, cur[arm])
        pair_times.append(cur)
        for arm in cur:
            times[arm].extend(cur[arm])
    off_med, on_med = statistics.median(times["off"]), statistics.median(times["on"])
    off_mean, on_mean = statistics.fmean(times["off"]), statistics.fmean(times["on"])
    return {
        "warmup": warm,
        "block_pairs": pairs,
        "steps_per_block": per_block,
        "off_ms_median": 1e3 * off_med,
        "on_ms_median": 1e3 * on_med,
        "marginal_ms_median": 1e3 * (on_med - off_med),
        "off_ms_mean": 1e3 * off_mean,
        "on_ms_mean": 1e3 * on_mean,
        "marginal_ms_mean": 1e3 * (on_mean - off_mean),
        "pair_marginal_ms": [
            1e3 * (statistics.median(p["on"]) - statistics.median(p["off"])) for p in pair_times
        ],
        "steps_per_s_on": 1.0 / on_mean,
    }


# --- brain variants -------------------------------------------------------------------------------------


def _brain_variants(cfg: Config, reps: int, warmup: int) -> list[dict[str, Any]]:
    """``brain.forward`` on all N rows for each ``impl`` × ``weight_dtype`` (§15.3 lever 2, §23.3).

    Weights are the founder weights of ``init_state(cfg)`` cast to each dtype; inputs are uniform in
    ``[−1, 1]`` from a fixed bench-local key (the simulation key is never used).
    """
    s = init_state(cfg)
    d = dims(cfg)
    kx, kh = mx.random.split(mx.random.key(0), 2)
    x = mx.random.uniform(-1.0, 1.0, (d.n, d.lay.D_IN), key=kx)
    h = mx.random.uniform(-1.0, 1.0, (d.n, d.h), key=kh)
    W1f, W2f = s.W1.astype(mx.float32), s.W2.astype(mx.float32)
    del s
    mx.eval(x, h, W1f, W2f)
    rows = []
    for impl in BRAIN_IMPLS:
        for wd in WEIGHT_DTYPES:
            vcfg = replace(cfg, brain=replace(cfg.brain, impl=impl, weight_dtype=wd))
            row: dict[str, Any] = {"impl": impl, "weight_dtype": wd}
            try:
                row.update(_time_forward(vcfg, W1f, W2f, x, h, reps, warmup))
            except (RuntimeError, ValueError) as e:  # a backend gap is a finding, not a crash
                row["error"] = f"{type(e).__name__}: {e}"
            rows.append(row)
    return rows


def _time_forward(
    vcfg: Config, W1f: mx.array, W2f: mx.array, x: mx.array, h: mx.array, reps: int, warmup: int
) -> dict[str, Any]:
    wd = dims(vcfg).wdtype
    W1, W2 = W1f.astype(wd), W2f.astype(wd)
    mx.eval(W1, W2)

    def fwd(W1: mx.array, W2: mx.array, x: mx.array, h: mx.array) -> tuple[mx.array, mx.array]:
        return brain.forward(W1, W2, x, h, vcfg)

    fn = mx.compile(fwd) if vcfg.run.compile else fwd
    for _ in range(max(warmup, 1)):
        mx.eval(fn(W1, W2, x, h))
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn(W1, W2, x, h))
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    weight_bytes = _nbytes(W1, W2)
    return {
        "ms": 1e3 * statistics.fmean(times),
        "ms_median": 1e3 * med,
        "weights_gb": weight_bytes / _GB,
        "weight_gb_per_s": weight_bytes / med / 1e9 if med > 0 else None,
    }


# --- stages ---------------------------------------------------------------------------------------------


class _Sense(NamedTuple):
    age: mx.array
    geom: bonds.BondGeom
    fcell: mx.array
    desc: mx.array
    x: mx.array
    grounded: mx.array | None = None  # 3D only: physics.grounded of the start-of-step pos
    ctx: ext.ExtCtx | None = None  # the extension hooks' view (§43)


class _Interact(NamedTuple):
    costs: en.Costs | en.Costs3  # Costs3 in 3D
    act: Actions  # masked by solvency
    partner: mx.array  # post-S9
    rev: mx.array
    formed: mx.array
    as_prop: mx.array
    broken: mx.array
    slots_s9: mx.array
    got: mx.array
    cb: en.Combat
    sh: en.Share
    e_over: mx.array
    dt: lifecycle.Deaths
    mo: actions.Motion3 | None = None  # 3D only, masked by solvency
    xdec: dict[str, Any] | None = None  # the feature decodes, masked by solvency (§43)


def _stage_fns(cfg: Config) -> dict[str, Callable[..., Any]]:
    """One function per sub-step group of ``stages_for(cfg)``; together they are exactly ``step.step``,
    including its static 3D branches (§27.8) and, with ``weather.enabled``, the wind (S3w) and the weather
    update (S17w), which call the same ``weather`` functions as the step (§30)."""
    d3 = dims(cfg).d3
    wthr_on = cfg.weather.enabled  # static

    def flush(
        W1: mx.array, W2: mx.array, slot: mx.array, pW1: mx.array, pW2: mx.array
    ) -> tuple[mx.array, ...]:
        return lifecycle.flush_pending(W1, W2, slot, pW1, pW2)

    def neighbors(key: mx.array, pos: mx.array, alive: mx.array) -> tuple[Any, ...]:
        key_next, ks = step_keys(key)
        return key_next, ks, spatial.find_neighbors(pos, alive, cfg)

    def sense(s: State, nbr: spatial.Neighbors, ks: mx.array) -> _Sense:
        a = age(s.step, s.birth_step)
        geom = bonds.bond_geometry(s.pos, s.partner, cfg)
        fcell = fields.cell_index(s.pos, cfg)
        if d3:  # static
            gr = physics.grounded(s.pos, cfg)
            fobs = fields.sample_fields(s.resource, s.chem, fcell, s.heading, cfg, pitch=s.pitch)
            desc = sensing.neighbor_descriptors(s, nbr, cfg)
            ectx = ext.ExtCtx(nbr, desc, geom, fcell)
            x = sensing.build_inputs(
                s,
                a,
                nbr,
                desc,
                geom,
                fobs,
                ks[Stream.NOISE],
                cfg,
                grounded=gr,
                ext_in=ext.step_inputs(s, cfg, ectx),
            )
            return _Sense(a, geom, fcell, desc, x, gr, ectx)
        fobs = fields.sample_fields(s.resource, s.chem, fcell, s.heading, cfg)
        desc = sensing.neighbor_descriptors(s, nbr, cfg)
        ectx = ext.ExtCtx(nbr, desc, geom, fcell)
        x = sensing.build_inputs(
            s, a, nbr, desc, geom, fobs, ks[Stream.NOISE], cfg, ext_in=ext.step_inputs(s, cfg, ectx)
        )
        return _Sense(a, geom, fcell, desc, x, None, ectx)

    def think(W1: mx.array, W2: mx.array, x: mx.array, h: mx.array) -> tuple[mx.array, mx.array]:
        return brain.forward(W1, W2, x, h, cfg)

    def decode(
        y: mx.array, desc: mx.array, nbr: spatial.Neighbors, s: State
    ) -> Actions | tuple[Actions, actions.Motion3]:
        if d3:  # static: Actions and Motion3
            act = actions.decode(y, desc, nbr, s.partner, s.alive, cfg)
            return act, actions.decode_motion3(y, s.alive, cfg)
        return actions.decode(y, desc, nbr, s.partner, s.alive, cfg)

    def kw1(ks: mx.array) -> mx.array:
        """The hazard key of the step's weather stream split (``step.step``'s ``kw[1]``)."""
        return mx.random.split(ks[Stream.WEATHER], 2)[1]

    def wind_force(s: State) -> wthr.AgentForce:
        """S3w: the stored weather sampled at the start-of-step positions."""
        return wthr.agent_wind(wthr.unpack(s.wx), s.pos, s.buoy_cap if d3 else None, s.alive, cfg)

    def interact(
        s: State,
        act: Actions,
        se: _Sense,
        ks: mx.array,
        mo: actions.Motion3 | None,
        wind: wthr.AgentForce | None,
        y: mx.array,
    ) -> _Interact:
        # the mirror of ``step.step``: the feature dues, passed only when a feature has one (DESIGN §43.4)
        extra_due = ext.sum_due(spec.due(s, act, y, cfg) for spec in ext.active(cfg) if spec.due is not None)
        due_kw = {} if extra_due is None else {"extra_due": extra_due}
        bcost = arms.basal_cost(s, cfg)  # the arms upkeep joins the brain cost (DESIGN §42.3)
        if d3:  # static
            nb0 = bonds.n_bonds(s.partner)
            c = en.pay_costs3(s.energy, act, mo, nb0, bcost, s.buoy_cap, s.alive, cfg, **due_kw)
            mo = actions.mask(mo, c.solvent)
        else:
            c = en.pay_costs(s.energy, act, bonds.n_bonds(s.partner), bcost, s.alive, cfg, **due_kw)
        xdec = ext.mask_all(ext.decode_all(s, y, se.ctx, cfg), c.solvent, cfg)
        act = actions.mask(act, c.solvent)
        flag = act.release | (se.geom.length > cfg.bonds.break_length)
        if wind is not None and wind.breaks is not None:  # static: storm_break_u > 0
            flag = flag | wind.breaks[:, None]
        partner, rev, broken = bonds.break_bonds(s.partner, s.rev, flag & se.geom.valid)
        partner, rev, formed, as_prop = bonds.form_bonds(
            partner, rev, s.alive & c.solvent & act.bond_want, act.bond_target, ks[Stream.BOND_PRIO]
        )
        slots_s9 = (partner >= 0).astype(mx.int32).sum()
        E, resource, got = en.harvest(c.energy, s.resource, se.fcell, act, c.solvent, cfg)
        cb = en.combat(E, resource, se.fcell, act, partner, cfg, **arms.combat_kw(s, cfg))
        sh = en.share(cb.energy, partner, rev, s.partner, act, c.solvent, cfg)
        E, e_over = en.cap_energy(sh.energy, cfg)
        env = s.env if cfg.director.enabled else None  # static
        cat_mask = (
            lifecycle.catastrophe_mask(s.pos, s.alive, env, ks[Stream.CATASTROPHE], cfg)
            if env is not None
            else None
        )
        storm = lifecycle.storm_mask(wind.speed, s.alive, kw1(ks), cfg) if wind is not None else None
        dt = lifecycle.deaths(
            s.alive,
            c.solvent,
            E,
            cb.lost,
            se.age,
            s.uid,
            s.generation,
            s.clade,
            partner,
            rev,
            cb.resource,
            se.fcell,
            cfg,
            cat_mask=cat_mask,
            storm_mask=storm,
        )
        return _Interact(
            c, act, partner, rev, formed, as_prop, broken, slots_s9, got, cb, sh, e_over, dt, mo, xdec
        )

    def move(
        s: State,
        act: Actions,
        nbr: spatial.Neighbors,
        dt: lifecycle.Deaths,
        mo: actions.Motion3 | None,
        gr: mx.array | None,
        wind: wthr.AgentForce | None,
    ) -> tuple[mx.array, ...]:
        wind_f = wind.force if wind is not None else None  # static
        act_p = arms.physics_act(act, s, cfg)  # the armour slows the thrust (DESIGN §42.3)
        if d3:  # static: Integrated3 (a NamedTuple; its first three are pos, vel, heading as in 2D)
            sv = dt.survivor
            F = physics.forces3(
                s.pos, s.heading, s.pitch, act_p, mo, s.buoy_cap, nbr, sv, dt.partner, cfg, wind=wind_f
            )
            return physics.integrate3(s.pos, s.heading, s.pitch, F, act.grip, act.turn, mo.pitch, gr, sv, cfg)
        F = physics.forces(s.pos, s.heading, act_p, nbr, dt.survivor, dt.partner, cfg, wind=wind_f)
        return physics.integrate(s.pos, s.heading, F, act.grip, act.turn, dt.survivor, cfg)

    def reproduce(
        s: State, it: _Interact, h_new: mx.array, moved: tuple[mx.array, ...], a: mx.array, ks: mx.array
    ) -> tuple[tuple[mx.array, ...], lifecycle.Repro]:
        dt, sv, act = it.dt, it.dt.survivor, it.act
        pos, vel, heading = moved[:3]
        s1 = s._replace(
            alive=sv,
            energy=dt.energy,
            uid=dt.uid,
            partner=dt.partner,
            rev=dt.rev,
            resource=dt.resource,
            pos=pos,
            vel=vel,
            heading=heading,
            h=mx.where(sv[:, None], h_new, 0.0),
            msg=mx.where(sv[:, None], act.msg, 0.0),
            last_loss=mx.where(sv, it.cb.lost, 0),
            last_harvest=mx.where(sv, it.got, 0),
            last_inflow=mx.where(sv, it.sh.inflow, 0),
        )
        if d3:  # static
            s1 = s1._replace(pitch=moved.pitch)
        req = sv & act.reproduce & (dt.energy >= cfg.repro.min_energy) & (a >= cfg.repro.min_age)
        imm_request = s.env_imm_request if cfg.director.enabled else None  # static
        lanes = lifecycle.allocate_lanes(req, sv, ks[Stream.BIRTH_PRIO], cfg, s.step, imm_request=imm_request)
        rp = lifecycle.reproduce(s1, lanes, act.child_bond, ks, cfg)
        # The lane counts leave as one stacked array: with immigration off ``n_imm`` is a constant, and
        # MLX 0.32 mx.compile fails (``_Map_base::at``) when it is an output next to arrays that use it.
        counts = mx.stack([lanes.n_req, lanes.n_birth, lanes.n_imm])
        return (lanes.parent, lanes.slot, lanes.kind, counts), rp

    def weather_update(
        s: State, ks: mx.array, wind: wthr.AgentForce
    ) -> tuple[mx.array, wthr.RainIn | None, mx.array]:
        """S17w: advance the stored weather one step; returns the new packed ``wx``, the rain for the
        resource update (None when ``rain_gain == 0``) and the ``WEATHER_WARN`` input."""
        ws = wthr.unpack(s.wx)
        qs, t_mult = wthr.organism_forcing(s.chem, s.pos, s.alive, ws, cfg)  # S17: the start-of-step state
        ws2, wd = wthr.advance(
            ws, s.step, mx.random.split(ks[Stream.WEATHER], 2)[0], cfg, qs=qs, t_mult=t_mult
        )
        return wthr.pack(ws2), wthr.rain_for(ws2, cfg), wthr.health(wd, wind)

    def field_update(
        s2: State, it: _Interact, fcell: mx.array, rain: wthr.RainIn | None = None
    ) -> tuple[mx.array, ...]:
        dep = fields.deposit_chem(it.act.emit, fcell, it.costs.solvent, cfg)
        chem = fields.update_chem(s2.chem, dep, cfg)
        env = s2.env if cfg.director.enabled else None  # static
        resource, regen_inc = fields.update_resource(s2.resource, s2.patch_pos, cfg, s2.step, env, rain)
        return chem, resource, regen_inc, fields.drift_patches(s2.patch_pos, s2.patch_vel, cfg, env)

    def finish(s2: State, s: State, flows: Any) -> tuple[State, Any]:
        return step_mod._finish(s2, s, cfg, flows)

    fns = {
        "flush": flush,
        "neighbors": neighbors,
        "sense": sense,
        "brain": think,
        "decode": decode,
        "interact": interact,
        "physics": move,
        "reproduce": reproduce,
        "fields": field_update,
        "finish": finish,
    }
    if wthr_on:  # static
        fns["wind"] = wind_force
        fns["weather"] = weather_update
    return fns


class _StagedStep:
    """``step.step`` as separately compiled (when ``cfg.run.compile``) and separately evaluated stages."""

    def __init__(self, cfg: Config, *, compile: bool | None = None) -> None:
        use = cfg.run.compile if compile is None else compile
        self.cfg = cfg
        self.fns = {k: (mx.compile(f) if use else f) for k, f in _stage_fns(cfg).items()}
        self.stages = stages_for(cfg)
        self.times: dict[str, list[float]] = {name: [] for name, _ in self.stages}

    def _run(self, name: str, *args: Any) -> Any:
        t0 = time.perf_counter()
        res = self.fns[name](*args)
        mx.eval(res)
        self.times[name].append(time.perf_counter() - t0)
        return res

    def __call__(self, holder: list[State]) -> Any:
        """Advance the state held in ``holder`` (a one-element list) by one step; returns the ``StepOut``.

        The state is handed over in a list so that no reference to the old weights survives into the S0
        evaluation, which can then write in place as in the fused step (F1).
        """
        s = holder.pop()
        t0 = time.perf_counter()
        W1, W2 = self.fns["flush"](s.W1, s.W2, s.pend_slot, s.pend_W1, s.pend_W2)
        s = s._replace(W1=W1, W2=W2)
        mx.eval(W1, W2)
        self.times["flush"].append(time.perf_counter() - t0)
        key_next, ks, nbr = self._run("neighbors", s.key, s.pos, s.alive)
        se = self._run("sense", s, nbr, ks)
        h_new, y = self._run("brain", s.W1, s.W2, se.x, s.h)
        d3 = dims(self.cfg).d3
        dec = self._run("decode", y, se.desc, nbr, s)
        act, mo = dec if d3 else (dec, None)
        wthr_on = self.cfg.weather.enabled
        wind = self._run("wind", s) if wthr_on else None
        it = self._run("interact", s, act, se, ks, mo, wind, y)
        moved = self._run("physics", s, it.act, nbr, it.dt, it.mo, se.grounded, wind)
        (parent, slot, kind, counts), rp = self._run("reproduce", s, it, h_new, moved, se.age, ks)
        lanes = lifecycle.Lanes(parent, slot, kind, counts[0], counts[1], counts[2])
        wx2, rain, weather_bad = self._run("weather", s, ks, wind) if wthr_on else (None, None, None)
        chem, resource, regen_inc, patch_pos = self._run("fields", rp.state, it, se.fcell, rain)
        s2 = rp.state._replace(
            chem=chem, resource=resource, patch_pos=patch_pos, key=key_next, step=s.step + 1
        )
        if wthr_on:  # static
            s2 = s2._replace(wx=wx2)
        if self.cfg.director.enabled:  # static
            s2 = step_mod.consume_env(s2, s.env, lanes)
        flows = step_mod._Flows(
            costs=it.costs,
            got=it.got,
            cb=it.cb,
            sh=it.sh,
            e_over=it.e_over,
            dt=it.dt,
            lanes=lanes,
            rp=rp,
            regen_inc=regen_inc,
            nbr=nbr,
            v_clips=moved.v_clips if d3 else moved[3],
            fcell=se.fcell,
            desc=se.desc,
            x=se.x,
            act=it.act,
            as_prop=it.as_prop,
            formed=it.formed,
            broken=it.broken,
            slots_s9=it.slots_s9,
            weather_bad=weather_bad,
        )
        if it.xdec:  # static
            flows = flows._replace(ext=step_mod.ExtFlows(**it.xdec))
        if d3:  # static: the Counter3D inputs
            flows = flows._replace(z_floor=moved.z_floor, z_surface=moved.z_surface, grounded=se.grounded)
        s_new, out = self._run("finish", s2, s, flows)
        holder.append(s_new)
        return out

    def reset_times(self) -> None:
        for v in self.times.values():
            v.clear()


def _bench_stages(cfg: Config, steps: int, warmup: int, fused_ms: float) -> dict[str, Any]:
    """Per-group mean ms over ``steps`` staged steps after ``warmup`` (at least one, which compiles)."""
    staged = _StagedStep(cfg)
    holder = [init_state(cfg)]
    for _ in range(max(warmup, 1)):
        staged(holder)
    staged.reset_times()
    for _ in range(steps):
        staged(holder)
    ms = {name: 1e3 * statistics.fmean(staged.times[name]) for name, _ in staged.stages}
    total = math.fsum(ms.values())
    return {
        "labels": dict(staged.stages),
        "ms": ms,
        "frac": {name: v / total if total > 0 else 0.0 for name, v in ms.items()},
        "total_ms": total,
        "fused_ms": fused_ms,
        "inflation": total / fused_ms if fused_ms > 0 else None,
    }


# --- text report ----------------------------------------------------------------------------------------


# --- play (G3a, DESIGN §33.5) ----------------------------------------------------------------------------

PLAY_RATES: Final = ("max", "1")  # the rates of the gate: unthrottled and the base rate
PLAY_DIRECTORS: Final = ("off", "rules", "player")
PLAY_WARMUP: Final = 300
PLAY_STEPS: Final = 2000
# The gate of DESIGN §33.5.3 (the coordinator's Mac day): at 1x, achieved >= 0.98x, no late steps, p99 step
# <= 66 ms; at max, achieved >= 2.5x. The client's draw time (p95 <= 25 ms) is the Swift bench's, not here.
PLAY_GATE_1X: Final = {"achieved_x": 0.98, "late_steps": 0, "step_ms_p99": 66.0}
PLAY_GATE_MAX: Final = {"achieved_x": 2.5}
# The powers of DESIGN §34.8 (G3b): a click is applied within half a second (the limit of C6 is two) and the
# report of a chapter is on its way within half a second of the chapter's end.
PLAY_GATE_POWERS: Final = {"power_applied_p95_s": 0.5, "report_ms": 500.0}
PLAY_POWER_LIMIT_S: Final = 2.0
PLAY_POWER_IDS: Final = (9001, 9002)  # the scripted requests' ids


def _rate_value(rate: str) -> float:
    if rate == "max":
        return math.inf
    value = float(rate)
    if not 0.05 <= value <= 16:
        raise ValueError(f"a play rate is max or a multiplier in [0.05, 16], not {rate!r}")
    return value


def _stat(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


_RUN_DIRECTOR: Final = {"rules": "live", "player": "player"}  # the bench's names for `run.director`


def _play_items(director: str) -> list[str]:
    """What ``alife play`` applies on top of a world's config for a director mode (``cli._play_items``)."""
    from alife import play

    return [
        *(f"run.{k}={v}" for k, v in play.PLAY_CADENCE.items()),
        "game.observe=true",
        f"run.director={_RUN_DIRECTOR.get(director, 'off')}",
        *(["director.llm_every=0"] if director == "rules" else []),
    ]


class ScriptedControl:
    """Control lines handed over when the world reaches a step: the bench's "clicks" (``--play-powers``). The
    pacer's ``before_step`` sets ``now`` to the step it is at, and ``poll`` gives the lines whose step has
    been
    reached, in order, noting when each left (the click of a power, whose latency is read against its ``PLAN``
    ``applied``)."""

    def __init__(self, script: list[tuple[int, str]]) -> None:
        self.pending = sorted(script, key=lambda item: item[0])
        self.now = -1
        self.clicked: dict[int, float] = {}  # request id -> perf_counter when its line was handed over

    def poll(self, timeout: float | None = 0) -> list[str]:
        out: list[str] = []
        while self.pending and self.pending[0][0] <= self.now:
            _, line = self.pending.pop(0)
            out.append(line)
            for token in line.split():
                if token.startswith("id=") and token[3:].isdigit():
                    self.clicked[int(token[3:])] = time.perf_counter()
        if not out and timeout:
            threading.Event().wait(timeout)
        return out

    def close(self) -> None:
        pass


def power_script(cfg: Config, *, start: int, warmup: int, total: int, chapter: int) -> list[tuple[int, str]]:
    """The two scripted powers of the gate (DESIGN §34.8): a Food x1.3 half way between the end of the warm-up
    and the next chapter's end, and a Strike at the chapter-end step itself (rule P: a power at a step with a
    checkpoint). The arm must hold that chapter end after its warm-up, with a step to spare to see the strike
    applied."""
    from alife.director import actions as dir_actions

    end = (start // chapter + 1) * chapter
    measured = start + warmup
    if not measured < end < start + total:
        raise ConfigError(
            f"--play-powers needs a chapter end of {chapter} steps inside the measured part of the arm: the "
            f"arm runs steps {start + 1} to {start + total} (the warm-up ends at {measured}) and the next "
            f"chapter ends at {end}. Start from a checkpoint nearer to one (--play-resume RUN:STEP) or "
            "lengthen the arm "
            "(--steps)"
        )
    low, _ = dir_actions.radius_bounds(cfg, dir_actions.GAME_SPECS)
    middle = float(cfg.world.size) / 2
    food, strike = PLAY_POWER_IDS
    return [
        ((measured + end) // 2, f"power food id={food} mult=1.3 steps=3000"),
        (end, f"power strike id={strike} x={middle:g} y={middle:g} radius={low:g} kill=0.45"),
    ]


class _Timing:
    """The pacer's measurements of one arm: wall time per step and per part, after the warm-up."""

    def __init__(self, warmup: int) -> None:
        self.warmup = warmup
        self.start: int | None = None
        self.recording = False
        self.work: list[
            float
        ] = []  # ms from the exit of before_step to the next entry, whatever the step did
        self.work_typical: list[float] = []  # (the same, for a step that took no snapshot and no metrics row)
        self.work_snapshot: list[float] = []
        self.work_boundary: list[float] = []
        self.loop_typical: list[float] = []  # ms of the runner's step: before_step exit -> after_step entry
        self.loop_snapshot: list[float] = []
        self.parts: dict[str, list[float]] = {}
        self.first_at: float | None = None
        self.last_at = 0.0
        self.bytes_at_first: dict[str, int] = {}
        self.frames_at_first = 0
        self.steps = 0

    def add(self, key: str, ms: float) -> None:
        if self.recording:
            self.parts.setdefault(key, []).append(ms)


def _bench_pacer_class() -> type:
    """``PlayPacer`` with a stopwatch on its parts (built lazily: the play module is heavy)."""
    from alife import play

    class BenchPacer(play.PlayPacer):  # type: ignore[misc]
        timing: _Timing
        _prev_exit: float | None = None
        _snap = False
        _bound = False
        _write_s = 0.0
        _saved_ms = 0.0
        _save_at = 0.0
        _after_exit = 0.0

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            # an arm has nobody to press `run`: the world must not wait at a chapter end or for an alert (the
            # stall of a chapter's save is then a hitch, as in any unattended run)
            self.autopause.apply(play.AutopauseLine("all", None, False))
            self.chapter_ends = 0
            self.report_ms: list[float] = []  # ms from the chapter's end to its report on the stream
            self.plans: list[tuple[int, str]] = []  # (request id, status) of the PLAN records written
            self.applied_at: dict[int, float] = {}
            self._chapter_t0: float | None = None
            self._loop = threading.current_thread()
            # the alert engine's work, whichever hook calls it: a part like the others
            for name in ("on_record", "on_tick", "on_row", "on_chapter", "on_territory", "engine_check"):
                method = getattr(self.alerts, name)
                setattr(self.alerts, name, self._timed("alerts", method))

        def _timed(self, key: str, fn: Callable[..., Any]) -> Callable[..., Any]:
            return lambda *a, **k: self._lap(key, fn, *a, **k)

        def _lap(self, key: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.timing.add(key, 1e3 * (time.perf_counter() - t0))

        def _write(self, tag: str, data: bytes) -> None:
            t0 = time.perf_counter()
            super()._write(tag, data)
            now = time.perf_counter()
            on_loop = threading.current_thread() is self._loop  # (a report is posted by a worker)
            if on_loop:
                self._write_s += now - t0
            if tag == "PLAN":
                (size,) = struct.unpack_from("<I", data, 24)  # after the tag, the length and the head
                obj = json.loads(data[28 : 28 + size])
                self.plans.append((int(obj["id"]), str(obj["status"])))
                if obj["status"] == "applied":
                    self.applied_at.setdefault(int(obj["id"]), now)
            elif tag == "RPRT" and self._chapter_t0 is not None:
                (flags,) = struct.unpack_from("<H", data, 22)
                if not flags & 1:  # not the backlog sent when the world was opened
                    self.report_ms.append(1e3 * (now - self._chapter_t0))
                    self._chapter_t0 = None

        def _send_frame(self, state: State) -> None:
            self._lap("frma", super()._send_frame, state)

        def _tick_storms(self, state: State, t: int) -> None:
            self._lap("tick_storms", super()._tick_storms, state, t)

        def _send_labels(self, state: State) -> None:
            self._lap("labels", super()._send_labels, state)

        def _send_tables(self, state: State, *, labels: bool = True) -> None:
            self._lap("tables", super()._send_tables, state, labels=labels)

        def _flush_events(self, *, backlog: bool = False) -> None:
            self._lap("events", super()._flush_events, backlog=backlog)

        def before_step(self, state: State, t: int) -> None:
            tm, now = self.timing, time.perf_counter()
            if tm.start is None:
                tm.start = t
            if self._prev_exit is not None and tm.recording:
                ms = 1e3 * (now - self._prev_exit)
                tm.work.append(ms)
                (
                    tm.work_boundary if self._bound else tm.work_snapshot if self._snap else tm.work_typical
                ).append(ms)
            was = tm.recording
            tm.recording = t - tm.start >= tm.warmup
            if tm.recording and not was:
                tm.first_at = now
                tm.bytes_at_first = dict(self.bytes_by_tag)
                tm.frames_at_first = self._frames_written
            if tm.recording:
                tm.last_at = now
                tm.steps = t - tm.start - tm.warmup
            if isinstance(self.control, ScriptedControl):
                self.control.now = t
            super().before_step(state, t)
            self._prev_exit = time.perf_counter()

        def after_step(self, state: State, out: Any, t: int, social: list[dict[str, Any]]) -> None:
            entry = time.perf_counter()
            self._snap = bool(social)
            self._bound = t % play.PLAY_CADENCE["metrics_every"] == 0
            tm = self.timing
            if tm.recording:
                (tm.loop_snapshot if self._snap else tm.loop_typical).append(1e3 * (entry - self._prev_exit))
            self._write_s = 0.0
            super().after_step(state, out, t, social)
            tm.add("write", 1e3 * self._write_s)
            self._after_exit = time.perf_counter()

        def before_checkpoint(self, state: State, t: int) -> None:
            self._save_at = time.perf_counter()
            if self.chapters.is_end(t):
                self.chapter_ends += 1
                self._chapter_t0 = self._save_at  # the report job starts in this call
            super().before_checkpoint(state, t)

        def after_checkpoint(self, t: int) -> None:
            super().after_checkpoint(t)
            self._saved_ms = 1e3 * (time.perf_counter() - self._save_at)
            self.timing.add("save", self._saved_ms)

        def after_boundary(self, state: State, row: dict[str, Any], t: int) -> None:
            super().after_boundary(state, row, t)
            ms = 1e3 * (time.perf_counter() - self._after_exit) - self._saved_ms
            self._saved_ms = 0.0
            self.timing.add("boundary", max(ms, 0.0))

    return BenchPacer


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p99": None, "max": None}
    a = np.asarray(values)
    return {"p50": float(np.quantile(a, 0.5)), "p99": float(np.quantile(a, 0.99)), "max": float(a.max())}


def _plain_engine_ms(cfg: Config, state: State, steps: int = 100, warmup: int = 30) -> float:
    """Median ms of the compiled step without observers, from ``state`` (the reference the observer's cost
    is read against)."""
    fn = step_mod.make_step(replace(cfg, game=replace(cfg.game, observe=False)))
    times = []
    for i in range(warmup + steps):
        t0 = time.perf_counter()
        state, out = fn(state)
        mx.eval(state, out)
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    return 1e3 * statistics.median(times)


def _play_arm(
    cfg: Config, template: Path, workdir: Path, *, name: str, director: str, rate: str, steps: int,
    warmup: int, frame_hz: int, base_rate: float, chapter: int | None = None, powers: bool = False,
) -> dict[str, Any]:  # fmt: skip
    """One arm: the world of ``template`` played ``warmup + steps`` steps under the bench pacer, its stream
    drained by a reader thread through a pipe. With ``powers`` (a ``player`` arm) the two scripted powers of
    the gate are sent as a player would send them, and the arm's log of them is kept for the replay check."""
    import os
    import threading

    from alife import play, runner, stream
    from alife.director import make_director
    from alife.runio import RunDir, load_checkpoint

    cfg_arm = apply_overrides(cfg, _play_items(director))
    state = load_checkpoint(template, cfg_arm, check_hash=False)
    start = int(to_numpy(state.step))
    total = warmup + steps
    rd = RunDir.create(workdir / name, cfg_arm, overwrite=True)
    rd.update_meta(start_step=start)
    r_fd, w_fd = os.pipe()
    drained = [0]

    def drain() -> None:
        with os.fdopen(r_fd, "rb", buffering=0) as reader:
            while chunk := reader.read(1 << 20):
                drained[0] += len(chunk)

    thread = threading.Thread(target=drain, name="bench-reader", daemon=True)
    thread.start()
    sink = os.fdopen(w_fd, "wb", buffering=0)
    timing = _Timing(warmup)
    chapter = chapter or play.CHAPTER
    options = play.PlayOptions(
        base_rate=base_rate, rate=_rate_value(rate), paused=False, frame_hz=frame_hz, source="preset",
        run=str(rd.path), fresh=True, until=start + total, chapter=chapter,
    )  # fmt: skip
    script = power_script(cfg_arm, start=start, warmup=warmup, total=total, chapter=chapter) if powers else []
    control = ScriptedControl(script) if script else stream.NoControl()
    pacer = _bench_pacer_class()(sink, control, options)
    pacer.timing = timing
    t_all = time.perf_counter()
    try:
        res = runner.run(
            rd, cfg_arm, state, total, quiet=True,
            director=make_director(rd, cfg_arm, start, game_specs=director == "player"), pacer=pacer,
        )  # fmt: skip
    finally:
        pacer.close()  # the report worker finishes before the sink goes
        sink.close()
        thread.join(timeout=60)
    wall = time.perf_counter() - t_all
    del state
    if res.status != "done" or pacer.failure is not None:
        raise RuntimeError(f"play arm {name} ended {res.status} ({pacer.failure!r}) at {timing.steps} steps")
    seconds = max(timing.last_at - (timing.first_at or timing.last_at), 1e-9)
    measured = max(timing.steps, 1)
    steps_per_s = measured / seconds
    after = pacer.bytes_by_tag
    delta = {k: after.get(k, 0) - timing.bytes_at_first.get(k, 0) for k in after}
    alert_ms = timing.parts.get("alerts", [])
    applied = sorted(i for i, status in pacer.plans if status == "applied")
    loop_typical = _stat(timing.loop_typical)
    loop_snapshot = _stat(timing.loop_snapshot)
    engine = None  # filled by the caller (one reference measurement per run)
    parts = {k: _stat(v) for k, v in timing.parts.items()}
    arm = {
        "name": name,
        "director": director,
        "rate": rate,
        "steps_measured": measured,
        "seconds": seconds,
        "steps_per_s": steps_per_s,
        "achieved_x": steps_per_s / base_rate,
        "step_ms": _quantiles(timing.work),
        "step_counts": {
            "typical": len(timing.work_typical),
            "snapshot": len(timing.work_snapshot),
            "boundary": len(timing.work_boundary),
        },
        "part_calls": {k: len(v) for k, v in sorted(timing.parts.items())},
        "step_ms_typical": _stat(timing.work_typical),
        "step_ms_snapshot": _stat(timing.work_snapshot),
        "step_ms_boundary": _stat(timing.work_boundary),
        "components_ms": {
            "engine": engine,
            "loop_typical": loop_typical,
            "loop_snapshot": loop_snapshot,
            "observer": None,
            "settle": None if loop_snapshot is None or loop_typical is None else loop_snapshot - loop_typical,
            "frma": parts.get("frma"),
            "tick": parts.get("tick_storms"),
            "labels": parts.get("labels"),
            "tables": parts.get("tables"),
            "events": parts.get("events"),
            "write": parts.get("write"),
            "boundary": parts.get("boundary"),
            "save": parts.get("save"),
        },
        "late_steps": pacer.sched.late_steps,
        "save_stall_steps": pacer.sched.save_stall_steps,
        "frames": pacer._frames_written - timing.frames_at_first,
        "bytes_per_s": {k: v / seconds for k, v in sorted(delta.items())},
        "mb_per_s": sum(delta.values()) / seconds / 1e6,
        "drained_bytes": drained[0],
        "wall_s": wall,
        "n_alive_end": int(to_numpy(res.state.alive.sum())),
        "digest": state_digest(res.state),
        "chapters": {"length": chapter, "ends": pacer.chapter_ends, "report_ms": list(pacer.report_ms)},
        "alerts": {"calls": len(alert_ms), "ms_total": sum(alert_ms)},
        "powers": None,
        "run": str(rd.path),
        "start": start,
        "total": total,
    }
    if script:
        assert isinstance(control, ScriptedControl)
        latency = [
            pacer.applied_at[i] - control.clicked[i]
            for i in PLAY_POWER_IDS
            if i in pacer.applied_at and i in control.clicked
        ]
        arm["powers"] = {
            "script": [{"step": step, "line": line} for step, line in script],
            "plans": [{"id": i, "status": status} for i, status in pacer.plans],
            "applied": applied,
            "refused": sorted({i for i, status in pacer.plans if status == "refused"}),
            "latency_s": latency,
            "all_applied": applied == sorted(PLAY_POWER_IDS),
        }
    return arm


def _plain_digest(cfg: Config, template: Path, workdir: Path, *, director: str, total: int) -> str:
    """The digest of the same world run by ``runner.run`` with no pacer."""
    from alife import runner
    from alife.director import make_director
    from alife.runio import RunDir, load_checkpoint

    cfg_arm = apply_overrides(cfg, _play_items(director))
    state = load_checkpoint(template, cfg_arm, check_hash=False)
    start = int(to_numpy(state.step))
    rd = RunDir.create(workdir / f"plain_{director}", cfg_arm, overwrite=True)
    rd.update_meta(start_step=start)
    played = make_director(rd, cfg_arm, start, game_specs=director == "player")
    res = runner.run(rd, cfg_arm, state, total, quiet=True, director=played)
    del state
    return state_digest(res.state)


def _replay_digest(cfg: Config, template: Path, workdir: Path, arm: dict[str, Any]) -> str:
    """The digest of the world a played arm's own log of the player's powers makes when it is replayed from
    the arm's first step with no pacer (what ``resume --director replay`` does): a played world with powers is
    equal to this one, not to the plain run (DESIGN §34.5)."""
    import shutil

    from alife import runner
    from alife.director import make_director
    from alife.runio import DIRECTOR_LOG_FILE, RunDir, load_checkpoint

    items = [i for i in _play_items("player") if not i.startswith("run.director=")] + ["run.director=replay"]
    cfg_replay = apply_overrides(cfg, items)
    state = load_checkpoint(template, cfg_replay, check_hash=False)
    start = int(to_numpy(state.step))
    rd = RunDir.create(workdir / f"replay_{arm['name']}", cfg_replay, overwrite=True)
    rd.update_meta(start_step=start)
    shutil.copyfile(Path(arm["run"]) / DIRECTOR_LOG_FILE, rd.path / DIRECTOR_LOG_FILE)
    replayed = make_director(rd, cfg_replay, start, game_specs=True)
    res = runner.run(rd, cfg_replay, state, arm["total"], quiet=True, director=replayed)
    del state
    return state_digest(res.state)


def _play_gate(arm: dict[str, Any]) -> dict[str, Any]:
    """The gate of DESIGN §33.5.3 read from an arm. Misses are reported with the component table, never
    cured by changing a parameter."""
    if arm["rate"] == "max":
        want = PLAY_GATE_MAX
        got = {"achieved_x": arm["achieved_x"]}
    elif float(arm["rate"]) == 1.0:
        want = PLAY_GATE_1X
        got = {
            "achieved_x": arm["achieved_x"],
            "late_steps": arm["late_steps"],
            "step_ms_p99": arm["step_ms"]["p99"],
        }
    else:
        return {"applies": False}
    checks = {
        k: (got[k] is not None and (got[k] <= v if k in ("late_steps", "step_ms_p99") else got[k] >= v))
        for k, v in want.items()
    }
    # G3b (DESIGN §34.8): the scripted powers are applied within half a second of the click (p95 of the
    # arm's clicks; the limit of C6 is two) and a chapter's report is on the stream within half a second of
    # its end
    latency = (arm.get("powers") or {}).get("latency_s") or []
    reports = (arm.get("chapters") or {}).get("report_ms") or []
    if arm.get("powers") is not None:
        p95 = float(np.quantile(np.asarray(latency), 0.95)) if latency else None
        want = {**want, "power_applied_p95_s": PLAY_GATE_POWERS["power_applied_p95_s"]}
        got = {**got, "power_applied_p95_s": p95}
        checks["power_applied_p95_s"] = (
            p95 is not None
            and p95 <= PLAY_GATE_POWERS["power_applied_p95_s"]
            and arm["powers"]["all_applied"]
        )
    if reports:
        want = {**want, "report_ms": PLAY_GATE_POWERS["report_ms"]}
        got = {**got, "report_ms": max(reports)}
        checks["report_ms"] = max(reports) <= PLAY_GATE_POWERS["report_ms"]
    return {
        "applies": True, "want": want, "got": got, "pass": all(checks.values()), "checks": checks,
        "five_minutes": arm["seconds"] >= 300,
    }  # fmt: skip


def run_play_bench(
    cfg: Config,
    template: Path,
    workdir: Path,
    *,
    steps: int = PLAY_STEPS,
    warmup: int = PLAY_WARMUP,
    rates: tuple[str, ...] = PLAY_RATES,
    directors: tuple[str, ...] = ("off", "rules"),
    frame_hz: int | None = None,
    base_rate: float | None = None,
    chapter: int | None = None,
    powers: bool = False,
) -> dict[str, Any]:
    """The play loop of ``alife play`` run headless on the world saved in ``template`` (a state file), for
    each director mode and rate: ``warmup`` steps, then ``steps`` measured, with the full record set drained
    by a reader thread. ``cfg`` is the world's config (the planet's: weather on, ``director.enabled`` for
    the rule tier); the cadence of play and ``game.observe`` are applied per arm.

    Returns the ``play`` section of the report. Per arm: the achieved rate, ``step_ms`` p50/p99/max, the
    cost of the parts, ``late_steps``, ``save_stall_steps``, bytes by tag and the final digest. Per director
    mode: the digest of a run without a pacer, which every arm must equal (observing, pacing and presenting
    are not part of the trajectory).

    G3b: the ``player`` mode is the rule tier's neighbour (it decides nothing). With ``powers`` its arms send
    the two scripted powers of the gate (a click, and a strike at the chapter-end step itself; ``chapter`` is
    the chapter length, ``play.CHAPTER`` by default), and a trajectory with powers is no longer the plain
    run's: its
    digest is the one the arm's own log makes when it is replayed (``replay_digest``), and the latency from
    the click to the ``PLAN`` ``applied`` is in the arm.
    """
    from alife import play

    if steps < 1 or warmup < 0:
        raise ValueError(f"play bench needs steps >= 1 and warmup >= 0, got {steps} and {warmup}")
    if not directors or not rates:
        raise ValueError("play bench needs at least one rate and one director mode")
    if ("rules" in directors or "player" in directors) and not cfg.director.enabled:
        raise ConfigError(
            "bench --play with the rule tier or the player needs director.enabled = true (its compiled step "
            "carries the levers): pass --set director.enabled=true"
        )
    if powers and "player" not in directors:
        raise ConfigError("--play-powers sends the powers of a player: it needs the player director mode")
    workdir.mkdir(parents=True, exist_ok=True)
    frame_hz = frame_hz or play.FRAME_HZ
    base_rate = base_rate or play.BASE_RATE
    total = warmup + steps
    from alife.runio import load_checkpoint

    state = load_checkpoint(template, apply_overrides(cfg, _play_items("off")), check_hash=False)
    engine_ms = _plain_engine_ms(cfg, state)
    del state
    plain = {d: _plain_digest(cfg, template, workdir, director=d, total=total) for d in directors}
    arms = []
    replayed: dict[str, str] = {}
    for director in directors:
        for rate in rates:
            arm = _play_arm(
                cfg, template, workdir, name=f"{director}_{rate}", director=director, rate=rate, steps=steps,
                warmup=warmup, frame_hz=frame_hz, base_rate=base_rate, chapter=chapter,
                powers=powers and director == "player",
            )  # fmt: skip
            comp = arm["components_ms"]
            comp["engine"] = engine_ms
            if comp["loop_typical"] is not None:
                comp["observer"] = max(comp["loop_typical"] - engine_ms, 0.0)
            if arm["powers"] is not None:  # a trajectory with the player's powers is its log's, replayed
                replayed[arm["name"]] = _replay_digest(cfg, template, workdir, arm)
                arm["expected"] = "replay"
                arm["digest_equal"] = arm["digest"] == replayed[arm["name"]]
            else:
                arm["expected"] = "plain"
                arm["digest_equal"] = arm["digest"] == plain[director]
            arm["gate"] = _play_gate(arm)
            arms.append(arm)
    return {
        "steps": steps, "warmup": warmup, "base_rate": base_rate, "frame_hz": frame_hz,
        "cadence": dict(play.PLAY_CADENCE), "engine_ms": engine_ms, "powers": powers,
        "plain_digest": plain, "replay_digest": replayed, "arms": arms,
        "digests_equal": all(a["digest_equal"] for a in arms),
    }  # fmt: skip


def _format_play(pl: dict[str, Any]) -> list[str]:
    """The lines of a ``play`` report: one per arm, the parts of the step, and the gate."""

    def ms(v: float | None) -> str:
        return "-" if v is None else f"{v:.2f}"

    lines = [
        f"play loop ({pl['warmup']} warm-up + {pl['steps']} steps per arm, base rate {pl['base_rate']:g}, "
        f"engine alone {pl['engine_ms']:.2f} ms/step); digests "
        + (
            "equal to the plain run"
            + (" (a run with powers: to its log replayed)" if pl.get("powers") else "")
            if pl["digests_equal"]
            else "DIFFER from the plain run"
            + (" or from the replay of an arm's log" if pl.get("powers") else "")
        )
    ]
    for a in pl["arms"]:
        q, c = a["step_ms"], a["components_ms"]
        gate = a["gate"]
        verdict = "" if not gate["applies"] else (" gate PASS" if gate["pass"] else " gate MISS")
        lines.append(
            f"  director {a['director']:<5} rate {a['rate']:<3} "
            f"{a['steps_per_s']:6.1f} steps/s = {a['achieved_x']:.2f}x; "
            f"step ms p50 {ms(q['p50'])} p99 {ms(q['p99'])} max {ms(q['max'])}; late {a['late_steps']}, "
            f"save stall {a['save_stall_steps']}; {a['mb_per_s']:.1f} MB/s{verdict}"
        )
        chapters, alerts, scripted = a.get("chapters"), a.get("alerts"), a.get("powers")
        if chapters and (chapters["ends"] or (alerts or {}).get("calls")):
            extras = [f"{chapters['ends']} chapter end(s) of {chapters['length']} steps"]
            if chapters["report_ms"]:
                extras.append(
                    "report on the stream after "
                    + ", ".join(f"{v:.0f}" for v in chapters["report_ms"])
                    + " ms"
                )
            if alerts:
                extras.append(f"alert engine {alerts['ms_total']:.1f} ms in {alerts['calls']} calls")
            lines.append("      chapter, report, alerts: " + "; ".join(extras))
        if scripted:
            latency = ", ".join(f"{1e3 * v:.0f}" for v in scripted["latency_s"]) or "-"
            lines.append(
                f"      powers: applied {scripted['applied']} of {list(PLAY_POWER_IDS)}, "
                f"click to applied {latency} ms"
                + (f", refused {scripted['refused']}" if scripted["refused"] else "")
                + f"; digest {'equal' if a['digest_equal'] else 'DIFFERS'} to the {a['expected']} run"
            )
        lines.append(
            "      parts (ms): "
            + ", ".join(
                f"{k} {ms(c[k])}"
                for k in (
                    "loop_typical",
                    "observer",
                    "settle",
                    "frma",
                    "tick",
                    "tables",
                    "write",
                    "boundary",
                    "save",
                )
            )
        )
    return lines


def format_report(report: dict[str, Any]) -> str:
    """A short human-readable summary of a ``run_bench`` report."""
    r = report
    c = r["config"]
    height = f"height={c['world_height']:g} " if "world_height" in c else ""  # 3D only; H is the hidden size
    lines = [
        f"alife {r['alife_version']}  mlx {r['mlx_version']}  {r['platform']}  {r['device']}",
        f"N={c['n_max']} M={c['max_births_per_step']} n_init={c['n_init']} L={c['world_size']:g} {height}"
        f"K={c['k_neighbors']} C={c['cell_capacity']} H={c['hidden']} {c['weight_dtype']}/{c['brain_impl']} "
        f"compile={c['compile']} checks={c['runtime_checks']}",
        f"compile {r['compile_s']:.2f} s   {r['ms_per_step']:.2f} ms/step (median "
        f"{r['ms_per_step_median']:.2f}, p90 {r['ms_per_step_p90']:.2f})   {r['steps_per_s']:.1f} steps/s   "
        f"peak {r['peak_mem_gb']:.3f} GB",
        f"population {r['n_alive_start']} -> {r['n_alive_end']}   "
        f"births {r['births']}  deaths {r['deaths']}   "
        f"diag {'OK' if r['diag_ok'] else 'ERROR'} {','.join(r['diag_flags'])}   "
        f"invariants {'OK' if r['invariants_ok'] else 'VIOLATED'}",
        f"donation health: {r['donation']['verdict']} (excess {r['donation']['excess_gb'] * 1e3:.1f} MB, "
        f"{r['donation']['excess_over_weights']:.2f}x weights)",
    ]
    if "stages" in r:
        st = r["stages"]
        lines.append(f"stages (total {st['total_ms']:.2f} ms vs fused {st['fused_ms']:.2f} ms):")
        lines += [
            f"  {label:>7} {name:<10} {st['ms'][name]:8.3f} ms  {100 * st['frac'][name]:5.1f}%"
            for name, label in st["labels"].items()
        ]
    if "determinism" in r:
        dt = r["determinism"]
        verdict = "OK" if dt["match"] else "MISMATCH"
        lines.append(f"determinism ({dt['steps']} steps x 2): {verdict} {dt['digest_a'][:16]}")
    if "brain_variants" in r:
        lines.append("brain variants (forward, all N rows):")
        for v in r["brain_variants"]:
            if "error" in v:
                lines.append(f"  {v['impl']:<7} {v['weight_dtype']:<9} error: {v['error']}")
            else:
                bw = v["weight_gb_per_s"]
                lines.append(
                    f"  {v['impl']:<7} {v['weight_dtype']:<9} {v['ms_median']:8.3f} ms"
                    + (f"  {bw:7.1f} GB/s" if bw is not None else "")
                )
    if "observe" in r:
        ob = r["observe"]
        lines.append(
            f"observers ({ob['block_pairs']} block pairs of {ob['steps_per_block']} steps): "
            f"control {ob['control_ms_median']:.2f} ms, observed {ob['observed_ms_median']:.2f} ms median "
            f"(overhead {ob['overhead_ms_median']:+.2f} ms median, {ob['overhead_ms_mean']:+.2f} ms mean); "
            f"{ob['observed_steps_per_s']:.1f} steps/s observed; trajectory "
            f"{'identical' if ob['trajectory_match'] else 'DIFFERS'}"
        )
        if "social" in ob:
            so = ob["social"]

            def ms(v: float | None) -> str:
                return "n/a" if v is None else f"{v:.2f} ms"

            lines.append(
                f"social (a snapshot every {so['snapshot_every']} steps): snapshot step "
                f"{ms(ob['snapshot_step_ms_mean'])}, other steps {ms(ob['other_step_ms_mean'])}; "
                f"one snapshot {so['snapshot_device_ms_median']:.2f} ms device + "
                f"{so['snapshot_host_ms_median']:.2f} ms host (components "
                f"{so['components_ms_median']:.2f} ms; heavy metrics {so['heavy_ms_median']:.2f} ms) = "
                f"{so['amortised_ms_per_step']:.3f} ms/step amortised"
            )
    if "play" in r:
        lines += _format_play(r["play"])
    for key, label, unit in (
        ("weather", "weather", "weather"),
        ("weather_org", "weather-org (seeding and shade)", "the couplings"),
    ):
        if key in r:
            wt = r[key]
            lines.append(
                f"{label} ({wt['block_pairs']} block pairs of {wt['steps_per_block']} steps): "
                f"off {wt['off_ms_median']:.2f} ms, on {wt['on_ms_median']:.2f} ms median "
                f"(marginal {wt['marginal_ms_median']:+.2f} ms median, "
                f"{wt['marginal_ms_mean']:+.2f} ms mean); "
                f"{wt['steps_per_s_on']:.1f} steps/s with {unit}"
            )
    return "\n".join(lines)
