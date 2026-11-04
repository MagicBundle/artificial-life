"""The diel oracle pilot, PR D1b (DESIGN §49.7, docs/DIEL.md §7.0): assignment and strata, the scripted lift
against a host reference, the override as the only change, inheritance (immigrants unlabelled), the ledger
with a 20 % oracle, the harvest counters, the energy rows, the bench mirror, checkpoint, resume and fork."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from alife import actions, bench, cli, diel, diel_oracle, ext, invariants, metrics, sphere
from alife.config import ConfigError, load_config
from alife.runio import RunDir, load_checkpoint
from alife.state import Diag, LedgerIdx, counter_index, init_state, state_digest, to_numpy
from alife.step import make_step

F32 = "brain.weight_dtype=float32"
ON = ["diel.period=400", "diel.oracle=20"]


def load(*over: str, preset: str = "tiny3d"):
    return load_config(preset=preset, overrides=[F32, *over])


# --- config, registry -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("over", "msg"),
    [
        (["diel.period=400", "diel.oracle=20", "diel.control=mean"], "needs diel.control = none"),
        (
            ["diel.period=400", "diel.oracle=20", "physics.buoyancy_max=0.01"],
            "buoyancy_max > physics.gravity",
        ),
    ],
)
def test_validation(over: list[str], msg: str) -> None:
    with pytest.raises(ConfigError, match=msg.replace(".", r"\.").replace(">", ">")):
        load(*over)


def test_terminator_is_refused_for_the_oracle() -> None:
    with pytest.raises(ConfigError, match=r"needs diel\.mode = global"):
        load("diel.period=400", "diel.oracle=20", "diel.mode=terminator", preset="tiny3s")
    assert load(*ON, preset="tiny3s").diel.oracle == 20


def test_the_field_and_the_counters_exist_only_with_the_oracle() -> None:
    on, cyc = load(*ON), load("diel.period=400")
    assert [f.name for f in ext.state_fields(on, 3)] == ["diel_ctl", "diel_orc"]
    assert [f.name for f in ext.state_fields(cyc, 3)] == ["diel_ctl"]
    orc = ext.state_fields(on, 3)[1]
    assert orc.per_agent and orc.on_birth == "inherit" and not orc.conserved
    assert len(diel_oracle.COUNTERS) == 24 and diel_oracle.COUNTERS[0] == "DIEL_O1_PHOTIC_DAY"
    idx = [counter_index(on, n) for n in diel_oracle.COUNTERS]
    assert idx == list(range(idx[0], idx[0] + 24))
    with pytest.raises(KeyError):
        counter_index(cyc, "DIEL_O1_PHOTIC_DAY")
    assert ext.layout_features(on) == "" and ext.mask(on) == 0  # no brain footprint


# --- assignment -----------------------------------------------------------------------------------


def test_assignment_strata_hash_and_cap() -> None:
    cfg = load(*ON)
    s = init_state(cfg)
    orc, alive = to_numpy(s.diel_orc).astype(int), to_numpy(s.alive)
    uid = to_numpy(s.uid)
    z = to_numpy(sphere.geometry(cfg).height(s.pos)).astype(np.float64)
    band = np.clip(np.floor(4 * z / cfg.world.height), 0, 3)
    want_s = np.where(band == 0, 1, np.where(band == 3, 3, 2))
    pick = ((uid.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(1 << 32)) / 2.0**32 < 0.2
    np.testing.assert_array_equal(orc[alive], np.where(pick, want_s, -want_s)[alive])
    assert (orc[~alive] == 0).all()
    share = (orc[alive] > 0).mean()
    assert 0.12 < share < 0.28, share
    cap = to_numpy(s.buoy_cap)
    assert (cap[alive & (orc > 0)] == 1.0).all()
    np.testing.assert_array_equal(
        cap[orc < 0], to_numpy(init_state(load("diel.period=400")).buoy_cap)[orc < 0]
    )


def test_stratum_of_band_edges() -> None:
    cfg = load(*ON)
    h = cfg.world.height
    z = np.array([0.0, h / 4 - 1e-4, h / 4, h / 2, 3 * h / 4 - 1e-4, 3 * h / 4, h])
    np.testing.assert_array_equal(diel_oracle.stratum_of(z, cfg), [1, 1, 2, 2, 2, 3, 3])


def test_the_schedule_constants() -> None:
    cfg = load_config(preset="mini3d", overlays=["diel"], overrides=[F32, "diel.oracle=20"])
    sc = diel_oracle.schedule(cfg)
    ph, c, h = cfg.physics, cfg.diel, cfg.world.height
    v_rise, v_sink = (ph.buoyancy_max - ph.gravity) / ph.drag, ph.gravity / ph.drag
    assert sc.home[1:] == (h / 8, h / 2, 7 * h / 8)
    for k in (1, 2, 3):
        assert math.isclose(sc.up[k], min(7 * h / 8 - sc.home[k], 0.8 * v_rise * c.period * c.day_frac))
        assert math.isclose(sc.down[k], min(sc.home[k] - 1, 0.8 * v_sink * c.period * (1 - c.day_frac)))
    assert sc.up[3] == 0.0 and math.isclose(sc.hold, ph.gravity / ph.buoyancy_max)


# --- the plan and the override --------------------------------------------------------------------


def _host_buoy(orc: np.ndarray, z: np.ndarray, day: bool, cfg) -> np.ndarray:
    sc = diel_oracle.schedule(cfg)
    k = np.abs(orc)
    home = np.array(sc.home)[k]
    tgt = home + np.array(sc.up)[k] if day else home - np.array(sc.down)[k]
    if day:
        return np.where(z < tgt, 1.0, sc.hold)
    return np.where(z > tgt, 0.0, sc.hold)


@pytest.mark.parametrize("t", [0, 50, 150, 200, 250, 399])
def test_the_plan_against_a_host_reference(t: int) -> None:
    cfg = load(*ON)
    s = init_state(cfg)
    n = s.alive.shape[0]
    rng = np.random.default_rng(t)
    z = rng.uniform(0.5, cfg.world.height - 0.5, n).astype(np.float32)
    pos = to_numpy(s.pos).copy()
    pos[:, 2] = z
    orc = rng.integers(-3, 4, n).astype(np.int8)
    s = s._replace(pos=mx.array(pos), diel_orc=mx.array(orc), step=mx.array(t, dtype=mx.int32))
    p = diel_oracle.plan(s, cfg)
    _, sv = diel.light(mx.array(t, dtype=mx.int32), s.diel_ctl[1], cfg)
    day = bool(to_numpy(sv) > 0.5)
    assert bool(to_numpy(p.day)) == day
    assert day == (math.cos(2 * math.pi * t / 400) > math.cos(math.pi * 0.5))  # the sun is up
    ref = _host_buoy(orc.astype(int), z.astype(np.float64), day, cfg)
    on = to_numpy(p.on)
    np.testing.assert_allclose(to_numpy(p.buoy)[on], ref[on], atol=1e-7)
    np.testing.assert_array_equal(on, to_numpy(s.alive) & (orc > 0))
    # the override replaces the lift of the oracle rows only
    mo = actions.Motion3(pitch=mx.zeros((n,)), buoy=mx.full((n,), 0.37))
    out = to_numpy(diel_oracle.apply(mo, p).buoy)
    np.testing.assert_allclose(out[on], ref[on], atol=1e-7)
    assert np.all(out[~on] == np.float32(0.37))
    assert np.all(to_numpy(p.photic) == (z >= cfg.world.height - 2 * cfg.fields.photic_lambda))


def test_with_no_oracle_row_the_trajectory_equals_the_cycle_alone() -> None:
    """The oracle is only the override and the cap: a 0 % assignment (every row a comparison row) runs
    bit-identically to the cycle-on world without the oracle (same RNG, same everything else)."""
    on, cyc = load("diel.period=400", "diel.oracle=1"), load("diel.period=400")
    s_on, s_cyc = init_state(on), init_state(cyc)
    s_on = s_on._replace(diel_orc=-mx.abs(s_on.diel_orc), buoy_cap=s_cyc.buoy_cap)
    f_on, f_cyc = make_step(on), make_step(cyc)
    for _ in range(30):
        s_on, _ = f_on(s_on)
        s_cyc, _ = f_cyc(s_cyc)
    mx.eval(s_on, s_cyc)
    for name in type(s_cyc)._fields:
        np.testing.assert_array_equal(
            to_numpy(getattr(s_on, name)), to_numpy(getattr(s_cyc, name)), err_msg=name
        )


# --- the step: ledger, counters, inheritance ------------------------------------------------------


def test_the_ledger_closes_and_the_counters_book_the_harvest() -> None:
    cfg = load(*ON, "population.immigration_floor=0")
    fn = make_step(cfg)
    s = init_state(cfg)
    prev = s
    idx = {n: counter_index(cfg, n) for n in diel_oracle.COUNTERS}
    day_harvest = night_harvest = 0
    for t in range(800):
        s, out = fn(s)
        mx.eval(s, out)
        assert int(to_numpy(s.diag)) & int(Diag.LEDGER) == 0 and int(to_numpy(s.diag)) == 0, f"step {t}"
        cs = to_numpy(out.counters).astype(np.int64)
        harvest = int(to_numpy(out.ledger)[LedgerIdx.FLOW_HARVEST])
        all_q = sum(int(cs[idx[n]]) for n in diel_oracle.COUNTERS if "_ALL_" in n)
        photic_q = sum(int(cs[idx[n]]) for n in diel_oracle.COUNTERS if "_PHOTIC_" in n)
        unlabelled = int(to_numpy(mx.where(prev.alive & (prev.diel_orc == 0), 1, 0).sum()))
        if unlabelled == 0:  # every harvester is labelled: the counters split the whole harvest
            assert all_q == harvest, f"step {t}"
        assert 0 <= photic_q <= all_q
        day = bool(to_numpy(diel.light(prev.step, prev.diel_ctl[1], cfg)[1]) > 0.5)
        other = "_NIGHT" if day else "_DAY"
        assert all(int(cs[idx[n]]) == 0 for n in diel_oracle.COUNTERS if n.endswith(other)), f"step {t}"
        if day:
            day_harvest += all_q
        else:
            night_harvest += all_q
        if t % 100 == 0:
            assert invariants.full_check(s, cfg, prev, out) == [], f"step {t}"
        prev = s
    assert day_harvest > 0 and night_harvest > 0
    assert int(to_numpy(s.alive.sum())) > 0


def test_children_inherit_and_immigrants_are_unlabelled() -> None:
    cfg = load(*ON, "population.immigration_every=5", "population.immigration_count=4")
    fn = make_step(cfg)
    s = init_state(cfg)
    s = s._replace(birth_step=mx.where(s.alive, -400, s.birth_step))  # founders old enough to breed
    seen_child = seen_imm = 0
    for _ in range(120):
        prev = s
        s, out = fn(s)
        mx.eval(s, out)
        born = to_numpy(s.alive & (s.birth_step == prev.step))
        if not born.any():
            continue
        orc, par = to_numpy(s.diel_orc), to_numpy(s.parent_uid)
        alive_prev = to_numpy(prev.alive)
        lookup = {
            int(u): int(o)
            for u, o, a in zip(to_numpy(prev.uid), to_numpy(prev.diel_orc), alive_prev, strict=True)
            if a
        }
        for i in np.flatnonzero(born):
            if int(par[i]) >= 0:  # a birth: the parent's label
                assert orc[i] == lookup[int(par[i])], i
                seen_child += 1
            else:  # an immigrant (parent_uid -1): unlabelled
                assert orc[i] == 0, i
                seen_imm += 1
    assert seen_child > 0 and seen_imm > 0


# --- metrics, bench, files ------------------------------------------------------------------------


def test_the_energy_rows() -> None:
    cfg = load(*ON)
    s = init_state(cfg)
    s = s._replace(step=mx.array(401, dtype=mx.int32))  # founders aged 400: inside the window
    row = metrics.snapshot_metrics(s, cfg, heavy=False)
    orc, alive, e = to_numpy(s.diel_orc), to_numpy(s.alive), to_numpy(s.energy).astype(np.int64)
    want_n = [int((alive & (orc == g * k)).sum()) for g in (1, -1) for k in (1, 2, 3)]
    assert row["diel_orc_pop"] == want_n == row["diel_orc_n"]
    for j, (g, k) in enumerate((g, k) for g in (1, -1) for k in (1, 2, 3)):
        m = alive & (orc == g * k)
        assert row["diel_orc_energy"][j] == (float(e[m].mean()) if m.any() else None)
    old = metrics.snapshot_metrics(s._replace(step=mx.array(900, dtype=mx.int32)), cfg, heavy=False)
    assert old["diel_orc_n"] == [0] * 6 and old["diel_orc_energy"] == [None] * 6
    assert "diel_orc_pop" not in metrics.snapshot_metrics(
        init_state(load("diel.period=400")), load("diel.period=400"), heavy=False
    )


def test_the_bench_mirror_equals_the_step() -> None:
    cfg = load(*ON)
    cfg = replace(cfg, run=replace(cfg.run, compile=False))
    fused, staged = make_step(cfg), bench._StagedStep(cfg)
    s0 = init_state(cfg)
    s0 = s0._replace(birth_step=mx.where(s0.alive, -400, s0.birth_step), step=mx.array(150, dtype=mx.int32))
    mx.eval(s0)
    s, holder = s0, [s0]
    for _ in range(4):
        s, out = fused(s)
        out_staged = staged(holder)
        mx.eval(s, out, holder[0], out_staged)
        assert state_digest(holder[0]) == state_digest(s)
        for a, b in zip(tree_flatten(out), tree_flatten(out_staged), strict=True):
            np.testing.assert_array_equal(to_numpy(a[1]), to_numpy(b[1]), err_msg=a[0])


def _save(cfg, s, path: Path) -> RunDir:
    rd = RunDir.create(path, cfg)
    rd.save_checkpoint(s, cfg)
    rd.wait_saves()
    return rd


def _fork(src: Path, out: Path, *over: str, capsys) -> RunDir:
    args = ["run", "--init-from", str(src), "--steps", "10", "--out", str(out), "--quiet"]
    for o in over:
        args += ["--set", o]
    assert cli.main(args) == 0, capsys.readouterr().err
    return RunDir.open(out)


def test_checkpoint_resume_and_the_pilot_fork(tmp_path: Path, capsys) -> None:
    # the warm-up: cycle on, no oracle
    warm_cfg = load("diel.period=400")
    fn, s = make_step(warm_cfg), init_state(warm_cfg)
    for _ in range(30):
        s, _ = fn(s)
    mx.eval(s)
    warm = _save(warm_cfg, s, tmp_path / "warm")
    # the pilot's start: the fork assigns from the living rows of the source
    pilot = _fork(warm.path, tmp_path / "pilot", F32, "diel.oracle=20", capsys=capsys)
    pcfg = pilot.config()
    first = load_checkpoint(pilot.checkpoints()[0][1], pcfg)
    expect = diel_oracle.assign(load_checkpoint(warm.checkpoints()[-1][1], pcfg, check_hash=False), pcfg)
    np.testing.assert_array_equal(to_numpy(first.diel_orc), to_numpy(expect.diel_orc))
    # a fork of the pilot that keeps the oracle keeps the labels (the lineages continue)
    last = load_checkpoint(pilot.checkpoints()[-1][1], pcfg)
    cont = _fork(pilot.path, tmp_path / "cont", F32, capsys=capsys)
    np.testing.assert_array_equal(
        to_numpy(load_checkpoint(cont.checkpoints()[0][1], cont.config()).diel_orc), to_numpy(last.diel_orc)
    )
    # resume is bitwise
    a = last
    f = make_step(pcfg)
    for _ in range(5):
        a, _ = f(a)
    rc = cli.main(["resume", str(pilot.path), "--steps", "5", "--quiet"])
    assert rc == 0, capsys.readouterr().err
    resumed = load_checkpoint(RunDir.open(pilot.path).checkpoints()[-1][1], pcfg)
    mx.eval(a)
    assert state_digest(resumed) == state_digest(a)


def test_a_run_writes_the_interval_keys(tmp_path: Path, capsys) -> None:
    out = tmp_path / "run"
    rc = cli.main(
        ["run", "--preset", "tiny3d", "--set", F32, *[a for o in ON for a in ("--set", o)], "--steps", "40",
         "--out", str(out), "--quiet"]
    )  # fmt: skip
    assert rc == 0, capsys.readouterr().err
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines() if line.strip()]
    last = rows[-1]
    keys = [k for k in last if k.startswith("diel_harv_")]
    assert len(keys) == 24 and "diel_harv_o1_photic_day" in keys
    assert sum(last[k] for k in keys if "_all_" in k) > 0
    assert len(last["diel_orc_energy"]) == 6


@pytest.mark.parametrize("pct", ["10", "40", "0"])
def test_a_fork_cannot_change_an_assigned_oracle(tmp_path: Path, capsys, pct: str) -> None:
    """The source's oracle rows hold cap 1 with their own cap gone: relabelling or dropping them would leave
    comparison rows at cap 1 (review finding), so such a fork is refused before anything is written."""
    cfg = load(*ON)
    fn, s = make_step(cfg), init_state(cfg)
    for _ in range(5):
        s, _ = fn(s)
    mx.eval(s)
    src = _save(cfg, s, tmp_path / "src")
    out = tmp_path / "fork"
    args = ["run", "--init-from", str(src.path), "--set", F32, "--set", f"diel.oracle={pct}", "--steps", "5"]
    rc = cli.main([*args, "--out", str(out), "--quiet"])
    assert rc != 0 and "diel.oracle must equal the source's 20" in capsys.readouterr().err
    assert not out.exists()
