"""The sky-light input, PR D2 (DESIGN §49.4, docs/DIEL.md §4.2, §4.3): the layout, the log formula against
its goldens and a float64 reference, the matched control and the mute fork sensing the day mean, the silent
row, the migration, the ledger and the bench mirror."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from alife import bench, cli, diel, ext, invariants, layout, sphere
from alife.config import ConfigError, dims, load_config
from alife.runio import RunDir, load_checkpoint
from alife.state import Diag, init_state, state_digest, to_numpy
from alife.step import make_step

F32 = "brain.weight_dtype=float32"
SENSE = ["diel.period=400", "diel.sense=true"]


def load(*over: str, preset: str = "tiny3d"):
    return load_config(preset=preset, overrides=[F32, *over])


def _ctx(s):
    return ext.ExtCtx(None, None, None, mx.zeros((s.pos.shape[0],), dtype=mx.int32))


def test_the_layout() -> None:
    base, on = load("diel.period=400"), load(*SENSE)
    assert dims(on).lay.D_IN == dims(base).lay.D_IN + 1 == 85
    assert dims(on).lay.IN_LIGHT == 84 and dims(base).lay.IN_LIGHT is None
    assert dims(on).lay.D_OUT == dims(base).lay.D_OUT
    assert ext.layout_features(on) == "diel1@1" and ext.mask(on) == 64
    v0 = layout.genome_layout_version(3, 0)
    assert layout.genome_layout_version(3, ext.mask(on)) == v0 + 4 * 64
    assert ext.layout_blocks(on, 3) == (("diel", (("light", 1, True),), (), ()),)


def test_validation() -> None:
    for over, msg in (
        (["diel.period=400", "diel.extinction=35"], "needs diel.sense"),
        (["diel.period=400", "diel.silent_init=false"], "needs diel.sense"),
        (["diel.period=400", "diel.sense_control=mute"], "needs diel.sense"),
        (["diel.period=400", "diel.sense=true", "diel.extinction=-1"], "must be >= 0"),
    ):
        with pytest.raises(ConfigError, match=msg):
            load(*over)
    assert load(*SENSE, "diel.extinction=35", "diel.sense_control=mute", "diel.silent_init=false").diel.sense


def test_the_goldens_of_the_log_formula() -> None:
    """DESIGN §49.4's hand-computed values, to 1e-3 (0.0065 for 4.5e-5 is 0.00638 to four places)."""
    for i, want in ((0.0, 0.0), (1.0, 1.0), (4.5e-5, 0.0065), (8.3e-4, 0.0875)):
        assert abs(float(diel.sense_scale(i)) - want) < 1e-3, (i, want)
    assert math.isclose(math.log(1 / diel.SENSE_EPS), 6.9078, abs_tol=1e-4)


@pytest.mark.parametrize("preset", ["tiny3d", "tiny3s"])
def test_the_device_input_follows_the_host_reference(preset: str) -> None:
    cfg = load(*SENSE, "diel.night_level=0.05", "diel.twilight=0.5", preset=preset)
    s = init_state(cfg)
    for t in range(0, 400, 9):
        st = s._replace(step=mx.array(t, dtype=mx.int32))
        x = to_numpy(diel.sense_input(st, cfg, _ctx(st)))
        assert x.shape == (s.pos.shape[0], 1) and x.dtype == np.float32
        lt = diel.smooth_light(diel.sun_sine(2 * math.pi * t / 400, cfg), cfg.diel.day_frac, cfg)
        np.testing.assert_allclose(x[:, 0], diel.sense_scale(lt), atol=2e-6)


def test_the_input_reads_the_ramped_day_fraction() -> None:
    """While Long Night ramps (target 0.3, effective 0.5) the input is the light at the ramped value, the one
    the step's food regrows under (review finding: the at-rest tests could not tell the two apart)."""
    cfg = load(*SENSE)
    s = init_state(cfg)
    s = s._replace(diel_ctl=mx.array([0.3, 0.5], dtype=mx.float32))
    day = 0.5 - 1.0 / 800  # one ramp step toward the target
    for t in (90, 100, 110, 290, 300):  # around dusk and dawn, where the day fraction moves the light
        st = s._replace(step=mx.array(t, dtype=mx.int32))
        x = to_numpy(diel.sense_input(st, cfg, _ctx(st)))[:, 0]
        ref = diel.sense_scale(diel.smooth_light(diel.sun_sine(2 * math.pi * t / 400, cfg), day, cfg))
        np.testing.assert_allclose(x, ref, atol=2e-5)
        at_rest = diel.sense_scale(diel.smooth_light(diel.sun_sine(2 * math.pi * t / 400, cfg), 0.5, cfg))
        if t in (100, 300):
            assert abs(float(at_rest) - float(ref)) > 1e-4  # the test can tell the ramped value from the rest


def test_the_terminator_reads_the_agents_pixel() -> None:
    cfg = load(*SENSE, "diel.mode=terminator", preset="tiny3s")
    s = init_state(cfg)
    npix = sphere.geometry(cfg).npix
    fcell = mx.array(np.arange(s.pos.shape[0]) * 7 % (npix * 3), dtype=mx.int32)
    ctx = ext.ExtCtx(None, None, None, fcell)
    for t in (0, 100, 233):
        st = s._replace(step=mx.array(t, dtype=mx.int32))
        x = to_numpy(diel.sense_input(st, cfg, ctx))[:, 0]
        h = diel.sun_sine(2 * math.pi * t / 400, cfg)  # (Npix,)
        ref = diel.sense_scale(diel.smooth_light(h, cfg.diel.day_frac, cfg))[to_numpy(fcell) % npix]
        np.testing.assert_allclose(x, ref, atol=2e-6)


def test_the_extinction_arm_attenuates_with_depth() -> None:
    cfg = load(*SENSE, "diel.extinction=35")
    s = init_state(cfg)
    z = np.linspace(0.5, cfg.world.height - 0.5, s.pos.shape[0]).astype(np.float32)
    pos = to_numpy(s.pos).copy()
    pos[:, 2] = z
    st = s._replace(pos=mx.array(pos), step=mx.array(0, dtype=mx.int32))  # noon: L = 1
    x = to_numpy(diel.sense_input(st, cfg, _ctx(st)))[:, 0]
    ref = diel.sense_scale(np.exp(-(cfg.world.height - z.astype(np.float64)) / 35.0))
    np.testing.assert_allclose(x, ref, atol=2e-6)
    assert np.all(np.diff(x) >= 0)


@pytest.mark.parametrize("over", [["diel.control=mean"], ["diel.sense_control=mute"]])
def test_the_control_and_the_mute_fork_sense_the_day_mean(over: list[str]) -> None:
    cfg = load(*SENSE, *over, "diel.night_level=0.1")
    s = init_state(cfg)
    want = float(diel.sense_scale(diel.l_bar_host(cfg)))
    lt = diel.smooth_light(diel.sun_sine(2 * math.pi * np.arange(720) / 720, cfg), cfg.diel.day_frac, cfg)
    assert math.isclose(float(diel.l_bar_host(cfg)), float(lt.mean()), rel_tol=1e-12)
    for t in (0, 100, 200, 399):
        st = s._replace(step=mx.array(t, dtype=mx.int32))
        x = to_numpy(diel.sense_input(st, cfg, _ctx(st)))[:, 0]
        np.testing.assert_allclose(x, want, atol=2e-6)


def test_the_terminator_day_mean_is_per_pixel() -> None:
    cfg = load(*SENSE, "diel.mode=terminator", "diel.control=mean", preset="tiny3s")
    lb = diel.l_bar_host(cfg)
    assert lb.shape == (sphere.geometry(cfg).npix,)
    u = np.asarray(diel.geom.pixel_centres(cfg.world.nside))
    polar, equator = int(np.argmax(np.abs(u[:, 2]))), int(np.argmin(np.abs(u[:, 2])))
    assert lb[polar] != lb[equator]


def test_the_row_is_silent_for_founders_and_the_ledger_closes() -> None:
    cfg = load(*SENSE, "population.immigration_every=7", "population.immigration_count=3")
    s = init_state(cfg)
    li = dims(cfg).lay.IN_LIGHT
    assert not to_numpy(s.W1)[:, li].any()
    fn, prev = make_step(cfg), s
    for t in range(800):  # two days
        s, out = fn(s)
        mx.eval(s, out)
        assert int(to_numpy(s.diag)) & int(Diag.LEDGER) == 0 and int(to_numpy(s.diag)) == 0, f"step {t}"
        if t % 100 == 0:
            assert invariants.full_check(s, cfg, prev, out) == [], f"step {t}"
        prev = s
    immigrants = to_numpy(s.alive & (s.parent_uid < 0) & (s.birth_step > 0))
    assert immigrants.any() and not to_numpy(s.W1)[immigrants][:, li].any()  # immigrants start silent too


def _run(cfg, steps: int, s=None):
    fn = make_step(cfg)
    s = init_state(cfg) if s is None else s
    for _ in range(steps):
        s, _ = fn(s)
    mx.eval(s)
    return s


def test_the_migration_adds_a_silent_row_and_changes_no_behaviour(tmp_path: Path, capsys) -> None:
    """``alife migrate --enable diel`` on a diel-off evolved source adds the cycle and the silent light row.
    The zero row leaves every decision unchanged: with no births (whose mutation draws follow the W1 shape)
    the migrated world runs exactly as a fork of the same source with the cycle and no sense. (A diel world
    without the sense cannot gain it by migration: an enabled feature never changes its layout, §43.8.)"""
    src_cfg = load("repro.min_age=100000")
    s = _run(src_cfg, 20)
    rd = RunDir.create(tmp_path / "src", src_cfg)
    rd.save_checkpoint(s, src_cfg)
    rd.wait_saves()
    out = tmp_path / "migrated"
    rc = cli.main(
        [
            "migrate",
            str(rd.path),
            "--enable",
            "diel",
            "--set",
            "diel.period=400",
            "--out",
            str(out),
            "--quiet",
        ]
    )
    assert rc == 0, capsys.readouterr().err
    run = RunDir.open(out)
    cfg = run.config()
    assert cfg.diel.sense and cfg.diel.period == 400 and run.meta()["layout_features"] == "diel1@1"
    s1 = load_checkpoint(run.checkpoints()[-1][1], cfg)
    li = dims(cfg).lay.IN_LIGHT
    assert not to_numpy(s1.W1)[:, li].any()
    fork = tmp_path / "fork"
    args = ["run", "--init-from", str(rd.path), "--set", F32, "--set", "diel.period=400", "--steps", "30"]
    assert cli.main([*args, "--out", str(fork), "--quiet"]) == 0, capsys.readouterr().err
    frun = RunDir.open(fork)
    b = load_checkpoint(frun.checkpoints()[-1][1], frun.config())
    a = _run(cfg, 30, s1)
    for name in ("pos", "vel", "energy", "alive", "resource", "h", "buoy_cap", "pitch", "uid", "diel_ctl"):
        np.testing.assert_array_equal(to_numpy(getattr(a, name)), to_numpy(getattr(b, name)), err_msg=name)


def test_a_mute_fork_runs_and_senses_the_day_mean(tmp_path: Path, capsys) -> None:
    cfg = load(*SENSE)
    s = _run(cfg, 20)
    rd = RunDir.create(tmp_path / "src", cfg)
    rd.save_checkpoint(s, cfg)
    rd.wait_saves()
    out = tmp_path / "mute"
    args = ["run", "--init-from", str(rd.path), "--set", F32, "--set", "diel.sense_control=mute"]
    assert cli.main([*args, "--steps", "10", "--out", str(out), "--quiet"]) == 0, capsys.readouterr().err
    run = RunDir.open(out)
    mcfg = run.config()
    assert mcfg.diel.sense_control == "mute"
    last = load_checkpoint(run.checkpoints()[-1][1], mcfg)
    x = to_numpy(diel.sense_input(last, mcfg, _ctx(last)))[:, 0]
    np.testing.assert_allclose(x, float(diel.sense_scale(diel.l_bar_host(mcfg))), atol=2e-6)


def test_the_bench_mirror_equals_the_step() -> None:
    cfg = load(*SENSE)
    cfg = replace(cfg, run=replace(cfg.run, compile=False))
    fused, staged = make_step(cfg), bench._StagedStep(cfg)
    s0 = init_state(cfg)
    s0 = s0._replace(birth_step=mx.where(s0.alive, -400, s0.birth_step), step=mx.array(90, dtype=mx.int32))
    mx.eval(s0)
    s, holder = s0, [s0]
    for _ in range(4):
        s, out = fused(s)
        out_staged = staged(holder)
        mx.eval(s, out, holder[0], out_staged)
        assert state_digest(holder[0]) == state_digest(s)
        for a, b in zip(tree_flatten(out), tree_flatten(out_staged), strict=True):
            np.testing.assert_array_equal(to_numpy(a[1]), to_numpy(b[1]), err_msg=a[0])
