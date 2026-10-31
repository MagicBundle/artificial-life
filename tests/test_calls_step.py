"""Event-triggered calls in the whole step (DESIGN §47, step K1): the ledger closes with calls on (no new
term) at three call costs in the six worlds, the counters and the metrics, determinism, resume and observer
purity, the compiled step against the eager one, and the bench mirror. Eager unless a test says otherwise; a
compiled sphere is in ``test_calls_sphere_compiled``."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

import helpers
from alife import bench, calls, invariants, layout, metrics
from alife.config import dims
from alife.ext.calls import event_names
from alife.runio import load_checkpoint, write_state_file
from alife.state import (
    CounterIdx,
    Diag,
    LedgerIdx,
    counter_blocks,
    counter_index,
    init_state,
    n_counters,
    state_digest,
    to_numpy,
)
from alife.step import ActFlag, Acts, ExtActs, make_step
from calls_helpers import WEATHER, WORLDS, calls_cfg, patch_wind

STEPS = {"box2": 80, "box2w": 60, "box3": 60, "box3w": 40, "sph": 25, "sphw": 20}


def forced(cfg, s):
    """The founders attack, harvest on half of the rows and call on every voice channel: windows open often
    and the voice has something to say."""
    d = dims(cfg)
    s = helpers.set_output_bias(
        s,
        cfg,
        {layout.OUT_ATTACK: 1.0, layout.OUT_HARVEST: 2.0}
        | {d.lay.OUT_VOICE + c: 2.0 for c in range(d.voice_w)},
    )
    w2 = np.array(s.W2)
    w2[1::2, d.h, layout.OUT_HARVEST] = -4.0
    return s._replace(W2=mx.array(w2))


def wind_pattern(monkeypatch: pytest.MonkeyPatch, world: str) -> None:
    """Every third row sits in air above the storm threshold, in the weather worlds (their own storms are far
    from the founders)."""
    if world in WEATHER:
        patch_wind(monkeypatch, lambda n: np.where(np.arange(n) % 3 == 0, 0.07, 0.0))


def run(cfg, steps: int, s=None, *, observe: bool = False, compile: bool | None = None):
    fn = make_step(cfg, observe=observe, compile=compile)
    s = forced(cfg, init_state(cfg)) if s is None else s
    for _ in range(steps):
        prev = s
        s, out = fn(s)
        mx.eval(s, out)
        yield prev, s, out


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("cost", [0.0, 32.0, 512.0])
def test_the_ledger_closes_with_calls_on_and_the_counters_are_sound(
    world: str, cost: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No 17th term: the gated voice is HEAT_ACTION through ``extra_due``. TOTAL is conserved every step, the
    host check passes, ``OPEN`` counts the open windows of the start of the step exactly, ``SPOKE`` is bounded
    by it, and calls are only paid when made (cost 0 pays nothing)."""
    wind_pattern(monkeypatch, world)
    wx = world in WEATHER
    cfg = calls_cfg(world, f"comm.cost={cost}", events=7 if wx else 3, cost=cost)
    names = event_names(cfg.calls.events)
    idx = {k: counter_index(cfg, k) for k in ("COMM_DUE", "COMM_SPEAKERS")}
    for n in names:
        idx[f"o{n}"], idx[f"s{n}"] = (counter_index(cfg, f"CALLS_{k}_{n.upper()}") for k in ("OPEN", "SPOKE"))
    opened = dict.fromkeys(names, 0)
    spoke = due = 0
    for t, (prev, s, out) in enumerate(run(cfg, STEPS[world], compile=False)):  # (compiled: below)
        led, c = to_numpy(out.ledger), to_numpy(out.counters)
        assert led.shape == (16,) and helpers.ledger_net(led, cfg) == int(
            to_numpy(s.total) - to_numpy(prev.total)
        )
        assert c[idx["COMM_DUE"]] <= led[LedgerIdx.HEAT_ACTION]
        assert int(to_numpy(s.diag)) & 0xFFFF == 0, (t, Diag(int(to_numpy(s.diag))))
        win = (to_numpy(prev.evt) > 0) & to_numpy(prev.alive)[:, None]
        for k, n in enumerate(names):
            assert c[idx[f"o{n}"]] == win[:, k].sum() and c[idx[f"s{n}"]] <= c[idx[f"o{n}"]]
            opened[n] += int(c[idx[f"o{n}"]])
        assert c[idx["COMM_SPEAKERS"]] <= sum(c[idx[f"s{n}"]] for n in names)
        spoke, due = spoke + int(c[idx["COMM_SPEAKERS"]]), due + int(c[idx["COMM_DUE"]])
        if t % 10 == 0:
            assert invariants.full_check(s, cfg, prev, out) == [], t
    assert all(v > 0 for v in opened.values()), opened  # every event opened windows
    assert (due == 0) == (cost == 0.0) or spoke == 0
    assert spoke > 0


def test_the_counter_block_is_two_per_event_after_comm() -> None:
    cfg = calls_cfg("box2w", events=13)
    assert counter_blocks(cfg) == (
        ("comm", ("COMM_SPEAKERS", "COMM_DUE")),
        (
            "calls",
            (
                "CALLS_OPEN_HURT",
                "CALLS_SPOKE_HURT",
                "CALLS_OPEN_STORM",
                "CALLS_SPOKE_STORM",
                "CALLS_OPEN_LOST",
                "CALLS_SPOKE_LOST",
            ),
        ),
    )
    assert n_counters(2, True, counter_blocks(cfg)) == 28 + 2 + 6
    off = replace(cfg, calls=replace(cfg.calls, trigger=False))  # enabled alone: no counters, no field
    assert counter_blocks(off) == (("comm", ("COMM_SPEAKERS", "COMM_DUE")),)


@pytest.mark.parametrize("world", ["box2", "box3w", "sph"])
def test_determinism_resume_and_observer_purity(
    world: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wind_pattern(monkeypatch, world)
    cfg = calls_cfg(world, events=7 if world in WEATHER else 3)
    last = lambda g: list(g)[-1][1]  # noqa: E731
    a = last(run(cfg, 30))
    assert state_digest(a) == state_digest(last(run(cfg, 30)))
    half = last(run(cfg, 15))
    assert to_numpy(half.evt).any()  # the resume carries live windows
    path = tmp_path / "half.safetensors"
    write_state_file(path, half, cfg)
    back = load_checkpoint(path, cfg)
    assert np.array_equal(to_numpy(back.evt), to_numpy(half.evt)) and back.evt.dtype == mx.uint8
    assert state_digest(last(run(cfg, 15, back))) == state_digest(a)  # K12: the windows survive a resume
    plain, obs = list(run(cfg, 12)), list(run(cfg, 12, observe=True))
    assert state_digest(plain[-1][1]) == state_digest(obs[-1][1])
    for (prev, _, o1), (_, _, o2) in zip(plain, obs, strict=True):
        assert np.array_equal(to_numpy(o1.counters), to_numpy(o2.counters))
        x = o2.acts.ext
        assert (
            isinstance(x, ExtActs) and isinstance(x.calls, calls.CallsActs) and x.calls.open.dtype == mx.uint8
        )
        assert np.array_equal(
            to_numpy(x.calls.open),
            ((to_numpy(prev.evt) > 0) & to_numpy(prev.alive)[:, None]).astype(np.uint8),
        )
        spoken = int(((to_numpy(o2.acts.flags) & ActFlag.SPOKE.value) > 0).sum())
        assert spoken == int(to_numpy(o2.counters)[counter_index(cfg, "COMM_SPEAKERS")])
        assert isinstance(o2.acts, Acts)


@pytest.mark.parametrize("world", ["box2", "box2w", "box3"])
def test_compiled_and_eager_steps_agree(world: str) -> None:
    """One step from a state with windows planted open: every counter, the ledger and every integer field
    agree; floats to 1e-5 (bitwise equality is not promised, §14.2)."""
    cfg = calls_cfg(world, events=7 if world in WEATHER else 3)
    d = dims(cfg)
    s = forced(cfg, init_state(cfg))
    rs = np.random.default_rng(4)
    s = s._replace(evt=mx.array((rs.random((d.n, d.voice_w)) < 0.5).astype(np.uint8) * 3) * s.alive[:, None])
    s = s._replace(evt=s.evt.astype(mx.uint8))
    a, oa = make_step(cfg, compile=False)(s)
    b, ob = make_step(cfg, compile=True)(s)
    mx.eval(a, oa, b, ob)
    assert np.array_equal(to_numpy(oa.counters), to_numpy(ob.counters))
    assert np.array_equal(to_numpy(oa.ledger), to_numpy(ob.ledger))
    for name in type(s)._fields:
        x, y = to_numpy(getattr(a, name)), to_numpy(getattr(b, name))
        if x.dtype.kind == "f":
            np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-5, err_msg=name)
        else:
            assert np.array_equal(x, y), name
    assert to_numpy(oa.counters)[counter_index(cfg, "COMM_SPEAKERS")] > 0


@pytest.mark.parametrize("world", ["box2", "box3w", "sph", "sphw"])
def test_the_staged_pipeline_equals_the_step_with_calls_on(
    world: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    wind_pattern(monkeypatch, world)
    cfg = calls_cfg(world, events=7 if world in WEATHER else 3)
    cfg = replace(cfg, run=replace(cfg.run, compile=False))
    s0 = forced(cfg, init_state(cfg))
    fused, staged = make_step(cfg), bench._StagedStep(cfg)
    s, holder = s0, [s0]
    for i in range(5):
        s, out = fused(s)
        out_staged = staged(holder)
        mx.eval(s, out, holder[0], out_staged)
        assert state_digest(holder[0]) == state_digest(s), i
        for a, b in zip(tree_flatten(out), tree_flatten(out_staged), strict=True):
            assert a[0] == b[0]
            np.testing.assert_array_equal(to_numpy(a[1]), to_numpy(b[1]), err_msg=a[0])
    assert to_numpy(s.evt).any()


# --- metrics -------------------------------------------------------------------------------------------


def test_interval_row_and_snapshot_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    wind_pattern(monkeypatch, "box2w")
    cfg = calls_cfg("box2w", events=13, cost=8.0)
    trace = list(run(cfg, 30, compile=False))
    s = trace[-1][1]
    cs, ls = [to_numpy(o.counters) for *_, o in trace], [to_numpy(o.ledger) for *_, o in trace]
    blocks = counter_blocks(cfg)
    row = metrics.interval_row(30, np.stack(cs), np.stack(ls), {}, {}, nd=2, weather=True, blocks=blocks)
    alive_steps = int(np.stack(cs)[:, CounterIdx.N_ALIVE].sum())
    for n in ("hurt", "storm", "lost"):
        o = int(np.stack(cs)[:, counter_index(cfg, f"CALLS_OPEN_{n.upper()}")].sum())
        sp = int(np.stack(cs)[:, counter_index(cfg, f"CALLS_SPOKE_{n.upper()}")].sum())
        assert row[f"calls_open_{n}"] == pytest.approx(o / alive_steps)
        assert row[f"calls_rate_{n}"] == (pytest.approx(sp / o) if o else None)
        assert row[f"calls_rate_{n}"] is None or 0.0 <= row[f"calls_rate_{n}"] <= 1.0
    assert row["calls_open_storm"] > 0.1  # a third of the rows are in the storm air
    assert row["calls_deaths_storm_rate"] == pytest.approx(row["deaths_storm"] / alive_steps)
    snap = metrics.snapshot_metrics(s, cfg, heavy=False)
    assert 0.2 < snap["calls_win_storm"] <= 1.0 and set(snap) >= {"calls_win_hurt", "calls_win_lost"}
    assert snap["calls_exposure"] == pytest.approx(snap["weather"]["storm_exposure"])
    dry = calls_cfg("box2", events=3)
    assert "calls_exposure" not in metrics.snapshot_metrics(init_state(dry), dry, heavy=False)


# --- bench ---------------------------------------------------------------------------------------------


def test_bench_calls_flag_needs_the_feature_and_reports_the_marginal_cost() -> None:
    from alife.config import ConfigError, load_config

    with pytest.raises(ConfigError, match="bench --calls"):
        bench.run_bench(load_config(preset="tiny"), steps=1, warmup=1, calls_arm=True)
    cfg = calls_cfg("box2", events=3)
    r = bench.run_bench(cfg, steps=4, warmup=1, calls_arm=True, weather_pairs=1)
    c = r["calls"]
    assert c["calls_config"]["trigger"] is True and c["calls_config"]["events"] == 3
    assert {"off_ms_median", "on_ms_median", "marginal_ms_median"} <= set(c)


# --- the runner ----------------------------------------------------------------------------------------


def test_a_run_writes_the_calls_metrics_and_resumes(tmp_path: Path) -> None:
    """`alife run` with the calls on: every metrics row closes the ledger and carries the calls keys, the
    checkpoint holds `evt`, and `resume` continues the run."""
    from alife import cli
    from alife.runio import RunDir

    out = tmp_path / "run"
    sets = [
        "brain.weight_dtype=float32",
        "comm.enabled=true",
        "comm.width=2",
        "comm.cost=16",
        "calls.enabled=true",
        "calls.trigger=true",
        "calls.events=3",
        "run.metrics_every=10",
        "run.heavy_metrics_every=20",
        "run.checkpoint_every=20",
    ]
    args = [x for s in sets for x in ("--set", s)]
    assert cli.main(["run", "--preset", "tiny", *args, "--steps", "40", "--out", str(out), "--quiet"]) == 0
    rd = RunDir.open(out)
    rows = rd.metrics_rows()
    assert rows and all(r["ledger_error"] == 0 and r["diag"] & 0xFFFF == 0 for r in rows)
    keys = ("calls_open_hurt", "calls_rate_hurt", "calls_open_food", "calls_rate_food", "calls_win_hurt")
    assert all(all(k in r for r in rows) for k in keys), [k for k in keys if not all(k in r for r in rows)]
    assert any((r["calls_open_hurt"] or 0) > 0 for r in rows)
    cfg = rd.config()
    assert cfg.calls.trigger and load_checkpoint(rd.latest_checkpoint(), cfg).evt.shape[1] == 2
    assert cli.main(["resume", str(out), "--until", "60", "--quiet"]) == 0
    assert RunDir.open(out).metrics_rows()[-1]["step"] >= 60
