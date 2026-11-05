"""The diel cycle, PR D1 (DESIGN §49.2, §49.3, §49.5, §49.8): the ``[diel]`` section, its validation and
registry row, the light factor and the profile it modulates, the day mean ``V_bar``, the ``diel_ctl`` ramp,
the ledger over two days, the bench mirror, checkpoints and the migration. With ``diel.period = 0`` nothing
changes; that is the identity matrix's gate (``tests/identity_matrix.py``), not this file's."""

from __future__ import annotations

import dataclasses
import math
import platform
import re
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from alife import bench, cli, config, diel, ext, fields, invariants, metrics
from alife.config import ConfigError, DielConfig, config_hash, config_to_dict, dims, load_config
from alife.ext.diel import SPEC, day_frac_min
from alife.runio import RunDir, load_checkpoint
from alife.state import Diag, LedgerIdx, init_state, state_digest, to_numpy
from alife.step import make_step

F32 = "brain.weight_dtype=float32"
ON = "diel.period=400"  # tiny3d / tiny3s: metrics_every 10, 40 rows per day


def load(*over: str, preset: str = "tiny3d"):
    return load_config(preset=preset, overrides=[F32, *over])


# --- the section, the registry row, validation -------------------------------------------------------


def test_every_key_is_off_by_default_hashed_and_omitted_from_config_json() -> None:
    d = DielConfig()
    assert (d.period, d.mode, d.day_frac, d.twilight, d.night_level) == (0, "global", 0.5, 0.25, 0.0)
    assert (d.control, d.sense, d.extinction, d.sense_control, d.oracle, d.silent_init) == (
        "none",
        False,
        0.0,
        "none",
        0,
        True,
    )
    assert d.oracle_schedule == 1  # D1c (§49.14)
    keys = {("diel", f.name) for f in dataclasses.fields(DielConfig)}
    assert keys == set(ext.lever_keys()["diel"]) == set(SPEC.lever_keys) and len(keys) == 12
    assert keys <= set(config.POST_V01_KEYS) and keys <= set(config.SERIAL_OMIT_KEYS)
    assert not keys & (set(config.UNHASHED_KEYS) | set(config.OBSERVER_KEYS))
    off, on = load(), load(ON)
    assert "diel" not in config_to_dict(off, omit_defaults=True)
    assert config_to_dict(on, omit_defaults=True)["diel"] == {"period": 400}
    assert len({config_hash(off), config_hash(on), config_hash(load(ON, "diel.night_level=0.1"))}) == 3


def test_the_registry_row_has_no_footprint() -> None:
    assert ext.check_registry() == []
    assert SPEC.name == "diel" and SPEC.index == 6 and ext.EXTENSIONS[-1] == "diel"
    assert (
        ext.TAIL_ORDER.index("diel") == ext.TAIL_ORDER.index("niche") + 1 == ext.TAIL_ORDER.index("arms") - 1
    )
    on = load(ON)
    assert ext.active(on) == (SPEC,) and ext.layout_active(on) == () and ext.mask(on) == 0
    assert ext.layout_features(on) == "" and ext.layout_blocks(on, 3) == ()
    assert [f.name for f in ext.state_fields(on, 3)] == ["diel_ctl"] and ext.state_fields(load(), 3) == ()
    assert set(config.CLI_ONLY_FEATURE_KEYS["diel"]) == set(SPEC.lever_keys)  # no GUI levers until D6
    assert SPEC.overlay == "diel" and "diel" in config.WORLD_OVERLAYS
    assert "diel" in config.SPHERE_READY and config.SPHERE_FEATURE_PR["diel"] == "D1"


def test_the_overlay_is_the_claim_world_without_the_sense() -> None:
    cfg = load_config(preset="mini3d", overlays=["diel"])
    assert cfg.world.height == 40.0 and cfg.diel.period == 4000 and cfg.diel.mode == "global"
    assert not cfg.diel.sense and cfg.diel.period // cfg.run.metrics_every == 40


@pytest.mark.parametrize(
    ("over", "msg"),
    [
        (["diel.night_level=0.2"], "inert while diel.period is 0"),
        (["diel.mode=terminator"], "inert while diel.period is 0"),
        (["diel.control=mean"], "inert while diel.period is 0"),
        (["diel.sense=true"], "inert while diel.period is 0"),
        (["diel.period=399"], "in [400, 2^20]"),
        (["diel.period=2097152"], "in [400, 2^20]"),
        (["diel.period=405"], "multiple of run.metrics_every"),
        (["diel.period=400", "run.metrics_every=40"], "at least 12 metrics rows"),
        (["diel.period=400", "diel.mode=terminator"], "needs world.geometry = sphere"),
        (["diel.period=400", "diel.mode=polar"], "diel.mode 'polar'"),
        (["diel.period=400", "fields.photic_amp=0"], "fields.photic_amp > 0"),
        (["diel.period=400", "diel.twilight=0"], "diel.twilight 0.0 must be in (0, 1]"),
        (["diel.period=400", "diel.twilight=1.5"], "must be in (0, 1]"),
        (["diel.period=400", "diel.day_frac=0.95"], "diel.day_frac 0.95"),
        (["diel.period=400", "diel.day_frac=0.1"], "noon must reach full light"),
        (["diel.period=400", "diel.night_level=1"], "diel.night_level 1.0 must be in [0, 1)"),
        (["diel.period=400", "diel.control=half"], "diel.control 'half'"),
        (["diel.period=400", "diel.extinction=35"], "acts on the light input only: it needs diel.sense"),
        (["diel.period=400", "diel.silent_init=false"], "acts on the light input only: it needs diel.sense"),
        (["diel.period=400", "diel.sense_control=mute"], "needs diel.sense"),
        (
            ["diel.period=400", "diel.oracle=20", "physics.buoyancy_max=0"],
            "diel.oracle needs physics.buoyancy_max",
        ),
        (["diel.period=400", "diel.oracle=101"], "in [0, 100]"),
        (["diel.period=400", "diel.oracle=20", "diel.control=mean"], "needs diel.control = none"),
        (["diel.period=400", "fields.season_period=3000"], ">= 8 x diel.period"),
    ],
)
def test_validation_refuses(over: list[str], msg: str) -> None:
    with pytest.raises(ConfigError, match=re.escape(msg)):
        load(*over)


def test_validation_refuses_2d_and_the_live_director() -> None:
    with pytest.raises(ConfigError, match=r"world\.dims = 3"):
        load_config(preset="tiny", overrides=[ON])
    base = load(ON)
    with pytest.raises(ConfigError, match="refuses --director live"):
        config.validate(
            replace(
                base, director=replace(base.director, enabled=True), run=replace(base.run, director="live")
            )
        )


@pytest.mark.parametrize(
    "over",
    [
        [ON],
        [ON, "diel.control=mean", "diel.night_level=0.25", "diel.twilight=1", "diel.day_frac=0.34"],
        [ON, "fields.season_period=3200"],
        [
            "diel.period=1048576",
            "run.metrics_every=1024",
            "run.checkpoint_every=1024",
            "run.heavy_metrics_every=1024",
        ],
    ],
)
def test_validation_accepts(over: list[str]) -> None:
    assert load(*over).diel.period > 0


def test_day_frac_floor_is_the_full_noon_bound() -> None:
    for tw in (0.1, 0.25, 0.5, 1.0):
        lo = day_frac_min(tw)
        assert math.isclose(math.cos(math.pi * lo) + tw / 2, 1.0, abs_tol=1e-12)
    assert load(ON, "diel.twilight=1", f"diel.day_frac={day_frac_min(1.0)}").diel.twilight == 1.0


# --- the light factor and the profile ------------------------------------------------------------------


def _profile(cfg, t: int, day: float | None = None) -> np.ndarray:
    d = None if day is None else mx.array(day, dtype=mx.float32)
    return to_numpy(diel.profile(cfg, mx.array(t, dtype=mx.int32), d, fields.vertical_profile(cfg)))


@pytest.mark.parametrize("preset", ["tiny3d", "tiny3s"])
@pytest.mark.parametrize("night", ["0.0", "0.3"])
def test_full_light_is_the_static_profile_bit_for_bit(preset: str, night: str) -> None:
    """At noon (``t = 0 mod P``, global mode) ``S = 1`` and the profile is today's, element for element."""
    cfg = load(ON, f"diel.night_level={night}", preset=preset)
    static = to_numpy(fields.vertical_profile(cfg))
    for t in (0, 400, 4000):
        got = _profile(cfg, t)
        assert got.dtype == static.dtype and got.shape == static.shape
        assert got.tobytes() == static.tobytes()
    # and the whole fertility at noon equals the cycle-off fertility bit for bit
    off = load(preset=preset)
    s = init_state(off)
    t0 = mx.array(0, dtype=mx.int32)
    a = to_numpy(fields.fertility(s.patch_pos, cfg, t0))
    b = to_numpy(fields.fertility(s.patch_pos, off, t0))
    assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize("preset", ["tiny3d", "tiny3s"])
def test_the_device_light_follows_the_host_reference(preset: str) -> None:
    cfg = load(ON, "diel.night_level=0.1", "diel.twilight=0.5", preset=preset)
    c = cfg.diel
    for t in range(0, 400, 7):
        lt, s = diel.light(mx.array(t + 800, dtype=mx.int32), None, cfg)
        ref = diel.smooth_light(diel.sun_sine(2 * math.pi * t / c.period, cfg), c.day_frac, cfg)
        np.testing.assert_allclose(to_numpy(lt), ref, atol=2e-6)
        assert 0.0 <= float(to_numpy(s)) <= 1.0
    # noon full, midnight dark (n), the day fraction of the equator is day_frac (h > h0 for half the cycle)
    assert float(to_numpy(diel.light(mx.array(0, dtype=mx.int32), None, cfg)[0])) == 1.0
    assert math.isclose(
        float(to_numpy(diel.light(mx.array(200, dtype=mx.int32), None, cfg)[0])), 0.1, abs_tol=1e-6
    )
    h = diel.sun_sine(2 * math.pi * np.arange(4000) / 4000, cfg)
    assert abs((h > math.cos(math.pi * c.day_frac)).mean() - c.day_frac) < 1e-3
    # V stays in [B, 1] and V[0] = 1 at every phase (the floor layer never sees the cycle)
    b, _ = diel.split(cfg)
    for t in range(0, 400, 25):
        v = _profile(cfg, t).reshape(dims(cfg).fz)
        assert v[0] == 1.0 and np.all(v <= 1.0) and np.all(v >= b.astype(np.float32) - 1e-7)


def test_terminator_geometry_subsolar_longitude_and_poles() -> None:
    """``h = cos(lat)·cos(lon + th)``: the sun is overhead at longitude ``−th``, the pixels' daily swing is
    their ``cos(lat)``, so the light is the more constant the nearer the pole."""
    cfg = load(ON, "diel.mode=terminator", preset="tiny3s")
    u = np.asarray(diel.geom.pixel_centres(cfg.world.nside))
    lat = np.arcsin(np.clip(u[:, 2], -1, 1))
    lon = np.arctan2(u[:, 1], u[:, 0])
    for t in (0, 50, 133, 300):
        th = 2 * math.pi * t / 400
        h = to_numpy(diel.light(mx.array(t, dtype=mx.int32), None, cfg)[0])
        ref = diel.smooth_light(np.cos(lat) * np.cos(lon + th), cfg.diel.day_frac, cfg)
        np.testing.assert_allclose(h, ref, atol=2e-6)
        sub = int(np.argmax(np.cos(lat) * np.cos(lon + th)))
        assert abs(math.remainder(lon[sub] + th, 2 * math.pi)) < 0.5  # the subsolar pixel is near lon = -th
    hs = diel.sun_sine(2 * math.pi * np.arange(400) / 400, cfg)  # (400, Npix)
    np.testing.assert_allclose(hs.max(axis=0), np.cos(lat), atol=2e-3)
    np.testing.assert_allclose(hs.min(axis=0), -np.cos(lat), atol=2e-3)
    polar, equatorial = int(np.argmax(np.abs(lat))), int(np.argmin(np.abs(lat)))
    assert np.ptp(hs[:, polar]) < np.ptp(hs[:, equatorial])
    # the profile is (FZ, Npix) and the global sphere's (FZ, 1)
    assert _profile(cfg, 77).shape == (dims(cfg).fz, dims(cfg).npix)
    assert _profile(load(ON, preset="tiny3s"), 77).shape == (dims(cfg).fz, 1)


# --- V_bar, the control, the time-average ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("preset", "over"),
    [("tiny3d", []), ("tiny3s", []), ("tiny3s", ["diel.mode=terminator", "diel.night_level=0.2"])],
)
def test_v_bar_is_the_day_mean(preset: str, over: list[str]) -> None:
    """``V_bar`` is the mean of the cycling profile over the day: against the device profile of every step of
    a 720-step day (the table's own phases), to float32 rounding."""
    cfg = load("diel.period=720", *over, preset=preset)
    vb = diel.v_bar_host(cfg)
    acc = sum(_profile(cfg, t).astype(np.float64) for t in range(720)) / 720
    np.testing.assert_allclose(acc.reshape(vb.shape), vb, atol=2e-6)
    assert np.all(vb[0] == 1.0)
    b, p = diel.split(cfg)
    static = np.minimum(1.0, b + p)
    assert np.all(vb <= static.reshape(-1, *([1] * (vb.ndim - 1))) + 1e-12)
    # the device table, the matched control and the time-average (step None) are the same array
    dev = to_numpy(diel.v_bar(cfg))
    np.testing.assert_array_equal(dev.reshape(vb.shape), vb.astype(np.float32))
    mean = load("diel.period=720", "diel.control=mean", *over, preset=preset)
    for t in (0, 100, 360):
        np.testing.assert_array_equal(_profile(mean, t), to_numpy(diel.v_bar(mean)))
    s = init_state(cfg)
    fert0 = to_numpy(fields.fertility(s.patch_pos, cfg))
    fert_on = to_numpy(fields.fertility(s.patch_pos, mean, mx.array(5, dtype=mx.int32)))
    np.testing.assert_array_equal(fert0, fert_on)  # the initial resource is built on V_bar


# --- the step: the ramp, the regeneration, the ledger ------------------------------------------------------


def test_the_ramp_moves_by_at_most_one_half_period_per_step() -> None:
    cfg = load(ON)
    ctl = mx.array([0.3, 0.5], dtype=mx.float32)
    seen = []
    for _ in range(200):
        ctl = diel.ramp(ctl, cfg)
        seen.append(float(to_numpy(ctl[1])))
    steps = -np.diff([0.5, *seen])
    assert np.all(steps <= 1 / 800 + 1e-7) and math.isclose(seen[-1], 0.3, abs_tol=1e-6)
    assert math.isclose(seen[0], 0.5 - 1 / 800, abs_tol=1e-7)
    still = mx.array([0.5, 0.5], dtype=mx.float32)
    assert to_numpy(diel.ramp(still, cfg)).tobytes() == to_numpy(still).tobytes()


def _zero_resource(s, cfg):
    """No stored food, so ``resource_cap`` cannot bind within two days: ``INJECT_REGEN`` is then the whole
    regeneration."""
    r = mx.zeros_like(s.resource)
    return s._replace(resource=r, total=s.total - s.resource.astype(s.total.dtype).sum())


@pytest.mark.parametrize("preset", ["tiny3d", "tiny3s"])
def test_regeneration_over_a_day_against_the_probe_and_a_float64_reference(preset: str) -> None:
    """Each step's ``INJECT_REGEN`` equals ``Σ regen_rate`` of the stored state (start-of-step patches, the
    state's ``diel_ctl``) exactly; that probe is within one quantum per cell of the host float64
    ``floor(regen_max · fert · V)``, and the day's total within 1e-4."""
    cfg = load(ON, "fields.resource_cap=16000000", "diel.night_level=0.1", preset=preset)
    fn = make_step(cfg)
    s = _zero_resource(init_state(cfg), cfg)
    s = s._replace(diel_ctl=mx.array([0.42, 0.5], dtype=mx.float32))  # a ramp in progress: diel_ctl[1] moves
    b, p = diel.split(cfg)
    fz = dims(cfg).fz
    dev_total = ref_total = 0
    for _ in range(400):
        prev = s
        s, out = fn(s)
        mx.eval(s, out)
        booked = int(to_numpy(out.ledger)[LedgerIdx.INJECT_REGEN])
        rate = to_numpy(fields.regen_rate(prev.patch_pos, cfg, prev.step, diel=s.diel_ctl[1])).astype(
            np.int64
        )
        assert booked == int(rate.sum())
        fert2 = to_numpy(fields.fertility(prev.patch_pos, load(preset=preset), prev.step))[0].astype(
            np.float64
        )
        th = 2 * math.pi * (int(to_numpy(prev.step)) % 400) / 400
        lt = diel.smooth_light(diel.sun_sine(th, cfg), float(to_numpy(s.diel_ctl[1])), cfg)
        v = np.minimum(1.0, b + lt * p).reshape((fz,) + (1,) * (fert2.ndim))
        ref = np.floor(cfg.fields.regen_max * fert2[None] * v).astype(np.int64)
        assert np.abs(rate.reshape(ref.shape) - ref).max() <= 1
        dev_total += int(rate.sum())
        ref_total += int(ref.sum())
    assert abs(dev_total - ref_total) <= 1e-4 * ref_total
    assert math.isclose(float(to_numpy(s.diel_ctl[1])), 0.42, abs_tol=1e-6)  # the ramp ran (0.08 in 64 steps)


@pytest.mark.parametrize(
    ("preset", "over"),
    [
        ("tiny3d", []),
        ("tiny3s", []),
        pytest.param("tiny3s", ["diel.mode=terminator"], marks=pytest.mark.slow),  # ~100 s each on the CPU
        pytest.param("tiny3d", ["diel.control=mean"], marks=pytest.mark.slow),
    ],
)
def test_the_ledger_closes_over_two_days(preset: str, over: list[str]) -> None:
    """Two full days (800 steps), every step: no diag bit (the device ledger check), and the host invariants
    every 50 steps; a Long Night target written mid-run ramps the effective value with no ledger effect."""
    cfg = load(ON, *over, preset=preset)
    fn = make_step(cfg)
    s = init_state(cfg)
    prev = s
    for t in range(800):
        if t == 300:  # what Long Night's apply writes (D5b): the target only
            s = s._replace(diel_ctl=mx.stack([mx.array(0.3, dtype=mx.float32), s.diel_ctl[1]]))
        s, out = fn(s)
        mx.eval(s, out)
        assert int(to_numpy(s.diag)) & int(Diag.LEDGER) == 0 and int(to_numpy(s.diag)) == 0, f"step {t}"
        if t % 50 == 0:
            assert invariants.full_check(s, cfg, prev, out) == [], f"step {t}"
        prev = s
    assert int(to_numpy(s.alive.sum())) > 0
    np.testing.assert_allclose(to_numpy(s.diel_ctl), [0.3, 0.3], atol=1e-6)  # 0.2 in 160 steps of 1/800


def test_the_probes_reproduce_the_step() -> None:
    """The regeneration probe of ``metrics.carrying_capacity`` uses ``V_bar`` with no step and, with the
    director's live step, the light of that step at the state's day fraction."""
    cfg = load(ON)
    s = init_state(cfg)
    mean = metrics._layer_regen(s, cfg, None, None).sum()
    noon = metrics._layer_regen(s, cfg, mx.array(0, dtype=mx.int32), None).sum()
    night = metrics._layer_regen(s, cfg, mx.array(200, dtype=mx.int32), None).sum()
    assert night < mean < noon
    off = metrics._layer_regen(s, load(), mx.array(0, dtype=mx.int32), None)
    np.testing.assert_array_equal(metrics._layer_regen(s, cfg, mx.array(0, dtype=mx.int32), None), off)


def test_the_bench_mirror_equals_the_step() -> None:
    cfg = load(ON, "diel.night_level=0.1")
    cfg = replace(cfg, run=replace(cfg.run, compile=False))
    fused, staged = make_step(cfg), bench._StagedStep(cfg)
    s0 = init_state(cfg)
    s0 = s0._replace(diel_ctl=mx.array([0.35, 0.5], dtype=mx.float32))  # a ramp in progress
    mx.eval(s0)
    s, holder = s0, [s0]
    for _ in range(5):
        s, out = fused(s)
        out_staged = staged(holder)
        mx.eval(s, out, holder[0], out_staged)
        assert state_digest(holder[0]) == state_digest(s)
        for a, b in zip(tree_flatten(out), tree_flatten(out_staged), strict=True):
            np.testing.assert_array_equal(to_numpy(a[1]), to_numpy(b[1]), err_msg=a[0])
    assert float(to_numpy(s.diel_ctl[1])) < 0.5


# sin, cos and exp differ between CPU and Metal, so the diel golden is pinned per platform (the seasons' rule,
# DESIGN §49.2): the state digest after 50 compiled float32 steps with diel.period 400 and night_level 0.1.
GOLDEN = {
    "mlx": "0.32.3",
    "Darwin-arm64-cpu": {
        "tiny3d": "5b7810d785d208e3",
        "tiny3s": "ab5e469d0f11b7ef",
        "tiny3s-terminator": "2d4a3925d2cb5ade",
    },
}


@pytest.mark.parametrize("case", ["tiny3d", "tiny3s", "tiny3s-terminator"])
def test_the_per_platform_golden(case: str) -> None:
    preset, _, mode = case.partition("-")
    cfg = load(ON, "diel.night_level=0.1", *([f"diel.mode={mode}"] if mode else []), preset=preset)
    fn, s = make_step(cfg), init_state(cfg)
    for _ in range(50):
        s, _ = fn(s)
    mx.eval(s)
    key = f"{platform.system()}-{platform.machine()}-{'gpu' if mx.default_device().type == mx.gpu else 'cpu'}"
    pins = GOLDEN.get(key)
    if mx.__version__ != GOLDEN["mlx"] or not pins:
        pytest.skip(f"pinned for MLX {GOLDEN['mlx']} on Darwin-arm64-cpu")
    assert state_digest(s).startswith(pins[case]), state_digest(s)


# --- checkpoints, resume, the migration ---------------------------------------------------------------


def test_checkpoint_round_trip_and_resume_are_bitwise(tmp_path: Path) -> None:
    cfg = load(ON)
    fn = make_step(cfg)
    s = init_state(cfg)
    s = s._replace(diel_ctl=mx.array([0.3, 0.45], dtype=mx.float32))
    for _ in range(20):
        s, _ = fn(s)
    mx.eval(s)
    rd = RunDir.create(tmp_path / "run", cfg)
    rd.save_checkpoint(s, cfg)
    rd.wait_saves()
    back = load_checkpoint(rd.checkpoints()[-1][1], cfg)
    assert state_digest(back) == state_digest(s)
    assert to_numpy(back.diel_ctl).tobytes() == to_numpy(s.diel_ctl).tobytes()
    a, b = s, back
    for _ in range(10):
        a, _ = fn(a)
        b, _ = fn(b)
    mx.eval(a, b)
    assert state_digest(a) == state_digest(b)


def test_migrate_enable_diel_completes_the_field_to_rest(tmp_path: Path, capsys) -> None:
    src_cfg = load()
    fn, s = make_step(src_cfg), init_state(src_cfg)
    for _ in range(20):
        s, _ = fn(s)
    mx.eval(s)
    rd = RunDir.create(tmp_path / "src", src_cfg)
    rd.save_checkpoint(s, src_cfg)
    rd.wait_saves()
    out = tmp_path / "out"
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
    assert cfg.diel.period == 400 and run.meta()["migrated_from"]["added"] == ["diel"]
    # D2: the migration also adds the silent light input (tests/test_diel_sense.py covers it in full)
    assert cfg.diel.sense and run.meta()["layout_features"] == "diel1@1"
    s1 = load_checkpoint(run.checkpoints()[-1][1], cfg)
    np.testing.assert_array_equal(to_numpy(s1.diel_ctl), [0.5, 0.5])
    li = dims(cfg).lay.IN_LIGHT
    w1 = to_numpy(s1.W1)
    assert not w1[:, li].any()  # the new row is zero
    np.testing.assert_array_equal(np.delete(w1, li, axis=1), to_numpy(s.W1))  # the old rows are unchanged
    s2, o2 = make_step(cfg)(s1)
    mx.eval(s2, o2)
    assert int(to_numpy(s2.diag)) == 0


def _fork(src: Path, out: Path, *over: str, capsys) -> RunDir:
    args = ["run", "--init-from", str(src), "--steps", "10", "--out", str(out), "--quiet"]
    for o in over:
        args += ["--set", o]
    rc = cli.main(args)
    assert rc == 0, capsys.readouterr().err
    return RunDir.open(out)


def test_a_fork_restarts_the_day_fraction_at_its_own_rest(tmp_path: Path, capsys) -> None:
    """A fork resets ``diel_ctl`` to the new config's ``(day_frac, day_frac)``, as it resets the env levers: a
    source's Long Night target and a changed ``day_frac`` do not leave a stale clock (review finding)."""
    cfg = load(ON)
    fn, s = make_step(cfg), init_state(cfg)
    s = s._replace(diel_ctl=mx.array([0.3, 0.41], dtype=mx.float32))  # Long Night had acted
    for _ in range(20):
        s, _ = fn(s)
    mx.eval(s)
    rd = RunDir.create(tmp_path / "src", cfg)
    rd.save_checkpoint(s, cfg)
    rd.wait_saves()
    same = _fork(rd.path, tmp_path / "same", F32, capsys=capsys)
    np.testing.assert_allclose(
        to_numpy(load_checkpoint(same.checkpoints()[0][1], same.config()).diel_ctl), [0.5, 0.5]
    )
    moved = _fork(rd.path, tmp_path / "moved", F32, "diel.day_frac=0.6", capsys=capsys)
    first = load_checkpoint(moved.checkpoints()[0][1], moved.config())
    np.testing.assert_allclose(to_numpy(first.diel_ctl), [0.6, 0.6], atol=1e-7)


def test_a_fork_can_turn_the_cycle_on(tmp_path: Path, capsys) -> None:
    off = load()
    fn, s = make_step(off), init_state(off)
    for _ in range(20):
        s, _ = fn(s)
    mx.eval(s)
    rd = RunDir.create(tmp_path / "src", off)
    rd.save_checkpoint(s, off)
    rd.wait_saves()
    run = _fork(rd.path, tmp_path / "on", F32, ON, capsys=capsys)
    assert run.config().diel.period == 400
    last = load_checkpoint(run.checkpoints()[-1][1], run.config())
    np.testing.assert_allclose(to_numpy(last.diel_ctl), [0.5, 0.5])
    assert invariants.full_check(last, run.config()) == []
