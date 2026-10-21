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
"""

from __future__ import annotations

import gc
import math
import platform
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Final, NamedTuple

import mlx.core as mx
import numpy as np

from alife import actions, bonds, brain, fields, invariants, lifecycle, metrics, physics, sensing, spatial
from alife import energy as en
from alife import step as step_mod
from alife import weather as wthr
from alife.actions import Actions
from alife.config import BRAIN_IMPLS, WEIGHT_DTYPES, Config, ConfigError, WeatherConfig, config_hash, dims
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
) -> dict[str, Any]:
    """Run the benchmark suite and return a JSON-compatible report.

    ``warmup`` steps (at least the compile call) run before anything is timed; ``steps`` steps are timed.
    The optional parts reuse ``steps`` and ``warmup`` (determinism: 2 × ``steps`` steps; brain variants:
    ``steps`` timed forward calls per variant; observers and weather: ``2 * pairs`` blocks of ``steps // 2``
    timed steps; ``weather`` needs ``cfg.weather.enabled``).
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")
    if weather and not cfg.weather.enabled:
        raise ConfigError("bench --weather needs weather.enabled = true (for example --overlay storm)")
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


def _bench_weather(cfg: Config, steps: int, warmup: int, pairs: int = WEATHER_PAIRS) -> dict[str, Any]:
    """The marginal cost of the engine weather layer: the compiled step with ``cfg`` (weather on) against the
    same config with the layer off (``replace(cfg, weather=WeatherConfig())``), STORMS §8 W1.

    Each arm follows the runner's loop (one blocking eval per step). The blocks run off, on, on, off for two
    pairs (the leading arm alternates in each pair, so that a drift linear in time, such as a thermal one,
    cancels from the difference of the means), each from ``init_state`` with the same warm-up and
    ``max(steps // 2, 1)`` timed steps; each arm's step compiles once. The trajectories differ by
    construction (the wind moves agents), so none is compared: the cost is of the step the config describes.
    ``marginal_ms_median`` is the difference of the median step times over all blocks of each arm,
    ``marginal_ms_mean`` of the means; ``pair_marginal_ms`` the same median difference within each pair.
    """
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
            x = sensing.build_inputs(s, a, nbr, desc, geom, fobs, ks[Stream.NOISE], cfg, grounded=gr)
            return _Sense(a, geom, fcell, desc, x, gr)
        fobs = fields.sample_fields(s.resource, s.chem, fcell, s.heading, cfg)
        desc = sensing.neighbor_descriptors(s, nbr, cfg)
        x = sensing.build_inputs(s, a, nbr, desc, geom, fobs, ks[Stream.NOISE], cfg)
        return _Sense(a, geom, fcell, desc, x)

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
    ) -> _Interact:
        if d3:  # static
            nb0 = bonds.n_bonds(s.partner)
            c = en.pay_costs3(s.energy, act, mo, nb0, s.brain_cost, s.buoy_cap, s.alive, cfg)
            mo = actions.mask(mo, c.solvent)
        else:
            c = en.pay_costs(s.energy, act, bonds.n_bonds(s.partner), s.brain_cost, s.alive, cfg)
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
        cb = en.combat(E, resource, se.fcell, act, partner, cfg)
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
        return _Interact(c, act, partner, rev, formed, as_prop, broken, slots_s9, got, cb, sh, e_over, dt, mo)

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
        if d3:  # static: Integrated3 (a NamedTuple; its first three are pos, vel, heading as in 2D)
            sv = dt.survivor
            F = physics.forces3(
                s.pos, s.heading, s.pitch, act, mo, s.buoy_cap, nbr, sv, dt.partner, cfg, wind=wind_f
            )
            return physics.integrate3(s.pos, s.heading, s.pitch, F, act.grip, act.turn, mo.pitch, gr, sv, cfg)
        F = physics.forces(s.pos, s.heading, act, nbr, dt.survivor, dt.partner, cfg, wind=wind_f)
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
        ws2, wd = wthr.advance(wthr.unpack(s.wx), s.step, mx.random.split(ks[Stream.WEATHER], 2)[0], cfg)
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
        it = self._run("interact", s, act, se, ks, mo, wind)
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
    if "weather" in r:
        wt = r["weather"]
        lines.append(
            f"weather ({wt['block_pairs']} block pairs of {wt['steps_per_block']} steps): "
            f"off {wt['off_ms_median']:.2f} ms, on {wt['on_ms_median']:.2f} ms median "
            f"(marginal {wt['marginal_ms_median']:+.2f} ms median, {wt['marginal_ms_mean']:+.2f} ms mean); "
            f"{wt['steps_per_s_on']:.1f} steps/s with weather"
        )
    return "\n".join(lines)
