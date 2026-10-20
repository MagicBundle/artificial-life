"""W1 build step 1 (DESIGN §30): ``State.wx``, the rest state and its fill, the frozen legacy digests, RNG
layout version 3, and the weather-off identities that need no stepping (hashes, counter widths, metrics
key sets).

CPU and fast, except the read-only check on ``runs/64k3d_s0`` (``slow``, skipped when the run is absent).
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from alife import layout, metrics, rng
from alife import weather as W
from alife.config import Config, config_hash, config_to_dict, dims, load_config
from alife.runio import FORMAT_VERSION, CheckpointError, load_checkpoint
from alife.state import (
    V0_FIELDS,
    V0_FIELDS_3D,
    State,
    State3,
    _digest,
    expected_layout,
    init_state,
    legacy_digest,
    legacy_state_digest,
    legacy_state_digest_3d,
    n_counters,
    state_digest,
    state_from_arrays,
    state_to_arrays,
    to_numpy,
)
from alife.step import make_step

RUNS = Path("/Users/jeromeverony/artificial-life/runs")


def _np(a: mx.array) -> np.ndarray:
    return np.asarray(a)


@pytest.fixture(scope="module")
def storm3d() -> Config:
    return load_config(
        preset="64k3d",
        overlays=["storm"],
        overrides=["population.n_max=256", "population.n_init=64", "population.max_births_per_step=32"],
    )


@pytest.fixture(scope="module")
def storm2d() -> Config:
    return load_config(
        preset="tiny",
        overrides=[
            "world.size=320",
            "weather.enabled=true",
            "weather.u_max=0.2",
            "brain.weight_dtype=float32",
        ],
    )


# --- the packed field -------------------------------------------------------------------------------------


def test_pack_unpack_round_trip() -> None:
    assert W.N_WX == len(W.WeatherState._fields) == 9
    cfg = load_config(preset="tiny")
    rest = W.unpack(init_state(cfg).wx)
    assert isinstance(rest, W.WeatherState) and all(a.shape == (160, 160) for a in rest)
    wx = W.pack(rest)
    assert wx.shape == (9, 160, 160) and wx.dtype == mx.float32
    assert all(np.array_equal(_np(a), _np(b)) for a, b in zip(W.unpack(wx), rest, strict=True))
    marks = W.WeatherState(
        *(mx.full((4, 4), float(i)) for i in range(9))
    )  # slice i is field i, in field order
    assert [float(_np(a)[0, 0]) for a in W.unpack(W.pack(marks))] == list(range(9))


def test_wx_is_the_48th_state_field_in_2d_and_3d(micro_cfg: Config, micro3d_cfg: Config) -> None:
    assert State._fields[47] == "wx" and State3._fields[47] == "wx"
    assert State3._fields[:48] == State._fields and State3._fields[48:] == ("pitch", "buoy_cap")
    for cfg in (micro_cfg, micro3d_cfg):
        spec = expected_layout(cfg)
        assert spec["wx"] == ((9, 160, 160), mx.float32) and list(spec).index("wx") == 47
        assert list(spec) == list(type(init_state(cfg))._fields)
        s = init_state(cfg)
        assert tuple(s.wx.shape) == (9, dims(cfg).wn, dims(cfg).wn) and s.wx.dtype == mx.float32


def test_dims_wn_follows_cells(storm3d: Config) -> None:
    c = replace(storm3d, weather=replace(storm3d.weather, cells=80))
    assert dims(c).wn == 80 and expected_layout(c)["wx"][0] == (9, 80, 80)
    assert init_state(c).wx.shape == (9, 80, 80)  # the rest state of an 80-cell weather (dx = 4)


# --- the rest state and the fill --------------------------------------------------------------------------


def test_the_rest_state_values(storm3d: Config) -> None:
    p = W.params_from_config(storm3d)
    wx = W.rest_array(storm3d)
    ws = W.unpack(wx)
    assert np.all(_np(ws.h1) == 32.0) and np.all(_np(ws.h2) == 48.0)
    for a in (ws.u1, ws.v1, ws.u2, ws.v2, ws.P, ws.Dc):
        assert np.all(_np(a) == 0.0)
    ts = W.warm_pools(p, mx.array(0, dtype=mx.int32))
    qe = np.minimum(1.0 * (p.q_bg + (p.q_pool - p.q_bg) * _np(ts)), 1.0)
    np.testing.assert_allclose(_np(ws.Q), qe, rtol=1e-6)
    assert _np(ws.Q).max() == 1.0 and _np(ws.Q).min() == pytest.approx(
        0.70, abs=1e-6
    )  # pools saturate at Q_s


def test_rest_array_under_the_pools_of_its_step(storm3d: Config) -> None:
    """A fork that turns weather on gets the rest state under the pools of the step it resumes at."""
    p = W.params_from_config(storm3d)
    at0, at5k = W.rest_array(storm3d, 0), W.rest_array(storm3d, 5000)
    assert not np.array_equal(_np(at0), _np(at5k))
    for step in (
        5000,
        mx.array(5000, dtype=mx.int32),
        mx.array(5000),
    ):  # Python int, int32 and default int arrays
        got = W.rest_array(storm3d, step)
        assert np.array_equal(_np(got), _np(at5k))
    qe = np.minimum(p.q_bg + (p.q_pool - p.q_bg) * _np(W.warm_pools(p, mx.array(5000, dtype=mx.int32))), 1.0)
    np.testing.assert_allclose(_np(W.unpack(at5k).Q), qe, rtol=1e-6)
    assert np.array_equal(
        _np(at5k)[[0, 1, 2, 3, 4, 5, 7, 8]], _np(at0)[[0, 1, 2, 3, 4, 5, 7, 8]]
    )  # only Q moves
    # the pools drift west by 0.009 per step, so the rest Q map moves with them
    assert float(np.abs(_np(at5k)[6] - _np(at0)[6]).max()) > 1e-3


def test_a_weather_off_config_always_gets_the_step_0_array(
    storm3d: Config, micro_cfg: Config, micro3d_cfg: Config
) -> None:
    """So a filled array is bitwise the one a native weather-off state carries."""
    for cfg in (micro_cfg, micro3d_cfg):
        native = init_state(cfg).wx
        assert np.array_equal(_np(W.rest_array(cfg, 5000)), _np(native))
        assert np.array_equal(_np(W.rest_array(cfg, mx.array(250_000, dtype=mx.int32))), _np(native))
    # the same parameters, switched on, at step 0, are that array too (the confirmed set at L = 320)
    assert np.array_equal(_np(W.rest_array(storm3d, 0)), _np(init_state(micro3d_cfg).wx))
    assert np.array_equal(_np(init_state(storm3d).wx), _np(W.rest_array(storm3d)))


def _file_meta(cfg: Config, s: State, *, rng_version: str = "3") -> dict[str, str]:
    return {
        "format_version": str(FORMAT_VERSION),
        "rng_layout_version": rng_version,
        "genome_layout_version": str(layout.genome_layout_version(cfg.world.dims)),
        **({"dims": str(cfg.world.dims)} if cfg.world.dims == 3 else {}),
        "step": str(int(to_numpy(s.step))),
        "config_hash": config_hash(cfg),
        "config": json.dumps(config_to_dict(cfg), sort_keys=True),
    }


def _drop(s: State | State3, *names: str) -> dict[str, mx.array]:
    return {k: v for k, v in state_to_arrays(s).items() if k not in names}


def test_a_3d_file_without_wx_loads_with_the_rest_state(micro3d_cfg: Config, tmp_path: Path) -> None:
    s = init_state(micro3d_cfg)
    path = tmp_path / "old3d.safetensors"
    mx.save_safetensors(str(path), _drop(s, "wx"), _file_meta(micro3d_cfg, s, rng_version="2"))
    back = load_checkpoint(path, micro3d_cfg)  # the config hash matches: a pre-W1 run resumes
    assert isinstance(back, State3) and np.array_equal(_np(back.wx), _np(W.rest_array(micro3d_cfg)))
    assert legacy_state_digest_3d(back) == legacy_state_digest_3d(s)
    assert all(np.array_equal(_np(getattr(back, f)), _np(getattr(s, f))) for f in V0_FIELDS_3D)


def test_a_2d_file_lacking_env_and_wx_loads_with_both_filled(micro_cfg: Config, tmp_path: Path) -> None:
    s = init_state(micro_cfg)
    env = tuple(f for f in State._fields if f.startswith("env_"))
    path = tmp_path / "old2d.safetensors"
    mx.save_safetensors(str(path), _drop(s, "wx", *env), _file_meta(micro_cfg, s, rng_version="1"))
    back = load_checkpoint(path, micro_cfg)
    assert legacy_state_digest(back) == legacy_state_digest(s) and state_digest(back) == state_digest(s)
    assert np.array_equal(_np(back.wx), _np(s.wx))
    # the director-era file (env present, no wx) loads as well
    path2 = tmp_path / "v2.safetensors"
    mx.save_safetensors(str(path2), _drop(s, "wx"), _file_meta(micro_cfg, s, rng_version="2"))
    assert state_digest(load_checkpoint(path2, micro_cfg)) == state_digest(s)


def test_the_fill_uses_the_files_own_step(storm3d: Config, micro3d_cfg: Config) -> None:
    """A weather-on config filling a file at step 5000 gets that step's rest state; weather off, step 0."""
    s = init_state(storm3d)._replace(step=mx.array(5000, dtype=mx.int32))
    got = state_from_arrays(_drop(s, "wx"), storm3d)
    assert np.array_equal(_np(got.wx), _np(W.rest_array(storm3d, 5000)))
    assert not np.array_equal(_np(got.wx), _np(W.rest_array(storm3d, 0)))
    off = init_state(micro3d_cfg)._replace(step=mx.array(5000, dtype=mx.int32))
    assert np.array_equal(
        _np(state_from_arrays(_drop(off, "wx"), micro3d_cfg).wx), _np(init_state(micro3d_cfg).wx)
    )


def test_the_strict_checks_stay(micro_cfg: Config, micro3d_cfg: Config) -> None:
    s2, s3 = init_state(micro_cfg), init_state(micro3d_cfg)
    with pytest.raises(ValueError, match="wx"):  # without cfg nothing is filled
        state_from_arrays(_drop(s3, "wx"))
    with pytest.raises(ValueError, match="bogus"):  # an unknown extra key still raises
        state_from_arrays({**state_to_arrays(s3), "bogus": mx.array(0)}, micro3d_cfg)
    with pytest.raises(ValueError, match="bogus"):
        state_from_arrays({**_drop(s2, "wx"), "bogus": mx.array(0)}, micro_cfg)
    with pytest.raises(ValueError, match="diag"):  # a missing non-fillable key
        state_from_arrays(_drop(s3, "wx", "diag"), micro3d_cfg)
    env = [f for f in State._fields if f.startswith("env_")]
    with pytest.raises(ValueError, match="env_cat_radius"):  # 3D: only wx is fillable, env_* is not
        state_from_arrays(_drop(s3, "wx", "env_cat_radius"), micro3d_cfg)
    with pytest.raises(ValueError, match="step"):  # the fill reads the file's step
        state_from_arrays(_drop(s3, "wx", "step"), micro3d_cfg)
    assert len(env) == 9
    assert state_digest(state_from_arrays(state_to_arrays(s3), micro3d_cfg)) == state_digest(s3)


def test_a_wrong_shaped_wx_is_refused_at_load(micro_cfg: Config, tmp_path: Path) -> None:
    s = init_state(micro_cfg)
    arrays = state_to_arrays(s) | {"wx": mx.zeros((9, 80, 80), dtype=mx.float32)}
    path = tmp_path / "bad.safetensors"
    mx.save_safetensors(str(path), arrays, _file_meta(micro_cfg, s))
    with pytest.raises(CheckpointError, match="wx"):
        load_checkpoint(path, micro_cfg)


@pytest.mark.slow
def test_the_64k3d_s0_checkpoint_loads_read_only() -> None:
    run = RUNS / "64k3d_s0"
    ckpt = run / "checkpoints" / "step_0000250000.safetensors"
    if not ckpt.is_file():
        pytest.skip(f"no {ckpt}")
    from alife.config import config_from_dict
    from alife.runio import checkpoint_metadata

    cfg = config_from_dict(json.loads((run / "config.json").read_text(encoding="utf-8")))
    meta = checkpoint_metadata(ckpt)
    assert config_hash(cfg) == meta["config_hash"]  # the run's hash is unchanged by the weather section
    s = load_checkpoint(ckpt, cfg)
    assert np.array_equal(
        _np(s.wx), _np(W.rest_array(cfg))
    )  # the pre-W1 file carries the (step-0) rest state
    arrays = mx.load(str(ckpt), format="safetensors")
    assert "wx" not in arrays
    from_file = _digest(State3(**{**{k: arrays[k] for k in V0_FIELDS_3D}, "wx": s.wx}), V0_FIELDS_3D)
    assert legacy_state_digest_3d(s) == from_file
    on = replace(cfg, weather=replace(cfg.weather, enabled=True))
    forked = load_checkpoint(ckpt, on, check_hash=False)
    assert np.array_equal(_np(forked.wx), _np(W.rest_array(on, 250_000)))
    assert legacy_state_digest_3d(forked) == legacy_state_digest_3d(s)


# --- the frozen legacy digests ----------------------------------------------------------------------------


def test_the_legacy_digests_do_not_see_wx(micro_cfg: Config, micro3d_cfg: Config) -> None:
    for cfg, legacy in ((micro_cfg, legacy_state_digest), (micro3d_cfg, legacy_state_digest_3d)):
        s = init_state(cfg)
        moved = s._replace(wx=s.wx + 1.0)
        assert legacy(s) == legacy(moved) == legacy_digest(s)
        assert state_digest(s) != state_digest(moved)  # state_digest covers every field, wx included
    s3 = init_state(micro3d_cfg)
    assert legacy_state_digest_3d(s3) != legacy_state_digest_3d(s3._replace(step=s3.step + 1))
    assert legacy_state_digest_3d(s3) != legacy_state_digest_3d(
        s3._replace(env_regen_mult=s3.env_regen_mult * 2)
    )


def test_the_legacy_3d_digest_is_the_digest_of_the_49_pre_wx_fields(micro3d_cfg: Config) -> None:
    s = init_state(micro3d_cfg)
    h = hashlib.sha256()
    for name in V0_FIELDS_3D:
        a = getattr(s, name)
        h.update(f"{name}|{a.dtype}|{tuple(a.shape)}|".encode())
        h.update(np.ascontiguousarray(to_numpy(a)).tobytes())
    assert legacy_state_digest_3d(s) == h.hexdigest()
    # the same value as state_digest restricted to State3 minus wx, which is what the pre-W1 state_digest was
    assert legacy_state_digest_3d(s) == _digest(s, tuple(f for f in State3._fields if f != "wx"))
    assert state_digest(s) != legacy_state_digest_3d(s)


def test_the_legacy_digests_reject_the_other_dimension(micro_cfg: Config, micro3d_cfg: Config) -> None:
    with pytest.raises(TypeError):
        legacy_state_digest(init_state(micro3d_cfg))
    with pytest.raises(TypeError):
        legacy_state_digest_3d(init_state(micro_cfg))
    assert legacy_digest(init_state(micro_cfg)) == legacy_state_digest(init_state(micro_cfg))


def test_the_legacy_v0_lists_are_literals_not_slices() -> None:
    assert len(V0_FIELDS) == 38 and len(V0_FIELDS_3D) == 49
    assert V0_FIELDS_3D[:38] == V0_FIELDS and V0_FIELDS_3D[-2:] == ("pitch", "buoy_cap")


_PRE_W1_TINY3D_50 = "556de6c6216dd530795fb91c2455c9e9d6f700bf0269c2e60acda212cff6d2fd"


@pytest.mark.golden
def test_legacy_3d_digest_equals_the_pre_w1_digest(tiny3d_cfg: Config) -> None:
    """``tiny3d`` f32, 50 compiled steps: the new ``legacy_state_digest_3d`` equals the ``state_digest`` of
    the commit before the weather layer (4337484, recorded with ``digest3d.py`` on this Mac CPU). Pinned
    to that platform; ``GOLDEN_3D`` is the Linux pin, the digest matrix the cross-code check elsewhere."""
    dev = mx.default_device()
    if (mx.__version__, platform.system(), platform.machine(), dev.type == mx.cpu) != (
        "0.32.3",
        "Darwin",
        "arm64",
        True,
    ):
        pytest.skip("the pre-W1 tiny3d digest was recorded on MLX 0.32.3, macOS arm64, CPU")
    fn, s = make_step(tiny3d_cfg), init_state(tiny3d_cfg)
    for _ in range(50):
        s, out = fn(s)
        mx.eval(s, out)
    assert legacy_state_digest_3d(s) == _PRE_W1_TINY3D_50


# --- weather off is the old engine: hashes, widths, metrics -------------------------------------------


def test_counter_widths_are_unchanged() -> None:
    assert n_counters(2) == 27 and n_counters(3) == 30


@pytest.mark.parametrize(
    ("preset", "snap_sha", "row_sha", "n_snap", "n_row"),
    [
        (
            "tiny",
            "73757fb9e9bfbcbd67f209ea9ac6021643cd69319a2621cbab9fbfb89b9da690",
            "85f92b033cc3e12737b781768880433b0bab6e7ecba272d88bb96e6f47bfd4af",
            35,
            68,
        ),
        (
            "tiny3d",
            "14601aeaa0012fc6dbcc176d4c9fcbc51d6d61ea2a87120ccba676b417477f73",
            "b3cee64923875d19c7eb5fa216b7f1a981c78d643208cce4f248b2d9c2ea8b4c",
            49,
            86,
        ),
    ],
)
def test_the_weather_off_metrics_key_sets_are_unchanged(
    preset: str, snap_sha: str, row_sha: str, n_snap: int, n_row: int
) -> None:
    """The key lists (sorted, JSON) of ``snapshot_metrics`` and ``interval_row`` at 4337484: no weather block,
    no ``deaths_storm``, no ``rain_*`` with weather off."""
    cfg = load_config(preset=preset, overrides=["brain.weight_dtype=float32"])
    nd = cfg.world.dims
    snap = metrics.snapshot_metrics(init_state(cfg), cfg, heavy=True)
    row = metrics.interval_row(
        10, np.zeros((2, n_counters(nd)), np.int64), np.zeros((2, 16), np.int64), snap, {}, nd=nd
    )
    for keys, sha, n in ((sorted(snap), snap_sha, n_snap), (sorted(row), row_sha, n_row)):
        assert len(keys) == n and hashlib.sha256(json.dumps(keys).encode()).hexdigest() == sha
        assert not [k for k in keys if "weather" in k or "storm" in k or k.startswith("rain")]


# --- RNG layout version 3 -----------------------------------------------------------------------------------


def test_the_weather_stream_takes_reserved_index_9() -> None:
    assert rng.Stream.WEATHER == 9 and rng.NUM_STREAMS == 16 and rng.RNG_LAYOUT_VERSION == 3
    assert set(rng.RNG_COMPATIBLE_VERSIONS) == {1, 2, 3}
    assert max(s.value for s in rng.Stream) == 9  # 10-15 stay reserved
    k = mx.random.key(7)
    nxt, ks = rng.step_keys(k)
    nxt2, ks2 = rng.step_keys(k)
    assert np.array_equal(_np(ks), _np(ks2)) and np.array_equal(_np(nxt), _np(nxt2))
    # the weather key is split two ways statically (disturbance key, hazard key): distinct, deterministic
    kw = mx.random.split(ks[rng.Stream.WEATHER], 2)
    assert not np.array_equal(_np(kw[0]), _np(kw[1]))
    assert not any(np.array_equal(_np(ks[rng.Stream.WEATHER]), _np(ks[i])) for i in range(16) if i != 9)


@pytest.mark.parametrize(
    ("version", "ok"), [("1", True), ("2", True), ("3", True), ("4", False), ("0", False)]
)
def test_3d_checkpoints_follow_the_compatible_versions(
    micro3d_cfg: Config, tmp_path: Path, version: str, ok: bool
) -> None:
    s = init_state(micro3d_cfg)
    p = tmp_path / f"v{version}.safetensors"
    mx.save_safetensors(str(p), state_to_arrays(s), _file_meta(micro3d_cfg, s, rng_version=version))
    if ok:
        assert state_digest(load_checkpoint(p, micro3d_cfg)) == state_digest(s)
    else:
        with pytest.raises(CheckpointError, match="rng_layout_version"):
            load_checkpoint(p, micro3d_cfg)


def test_checkpoints_are_written_at_version_3(micro_cfg: Config, tmp_path: Path) -> None:
    from alife.runio import checkpoint_metadata, write_state_file

    p = tmp_path / "w.safetensors"
    s = init_state(micro_cfg)
    write_state_file(p, s, micro_cfg)
    meta = checkpoint_metadata(p)
    assert meta["rng_layout_version"] == "3"
    assert np.array_equal(_np(load_checkpoint(p, micro_cfg).wx), _np(s.wx))
