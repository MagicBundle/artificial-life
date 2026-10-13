"""§27.5 the static 3D lattice: the z-major cell index, the vertical food profile, and the 3D fertility,
regeneration and initial field shapes; the 2D bodies refuse 3D inputs instead of dropping z."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from alife import fields
from alife.config import Config, apply_overrides, dims
from alife.state import ENV_FIELDS, Env, default_env
from helpers import as_3d


def _np(a: mx.array) -> np.ndarray:
    return np.array(a)


@pytest.fixture(scope="module")
def cfg3(micro_cfg: Config) -> Config:
    """``micro`` in 3D: L 48, field cell 4 (FG 12), height 24 (FZ 6), benthic_lambda 8."""
    return as_3d(micro_cfg)


def _patches(cfg: Config, seed: int = 0) -> mx.array:
    k = mx.random.split(mx.random.key(seed), 2)
    return fields.init_patches(k[0], k[1], cfg)[0]


# --- the cell index ---------------------------------------------------------------------------------------


def test_cell_index_3d_is_z_major(cfg3: Config) -> None:
    d, fc, L = dims(cfg3), cfg3.world.field_cell, cfg3.world.size
    rng = np.random.default_rng(0)
    pos = np.stack(
        [rng.uniform(0, L, 500), rng.uniform(0, L, 500), rng.uniform(d.z_lo, d.z_hi, 500)], axis=1
    ).astype(np.float32)
    got = _np(fields.cell_index(mx.array(pos), cfg3))
    q = np.floor(pos / np.float32(fc)).astype(np.int64)
    cx, cy, cz = q[:, 0] % d.fg, q[:, 1] % d.fg, np.clip(q[:, 2], 0, d.fz - 1)
    assert got.dtype == np.int32 and np.array_equal(got, (cz * d.fg + cy) * d.fg + cx)
    assert got.min() >= 0 and got.max() < d.n_cells and len(np.unique(cz)) == d.fz


def test_cell_index_floor_layer_is_the_2d_index(micro_cfg: Config, cfg3: Config) -> None:
    d = dims(cfg3)
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, cfg3.world.size, (200, 2)).astype(np.float32)
    flat2 = _np(fields.cell_index(mx.array(xy), micro_cfg))
    for z in (d.z_lo, 3.99, 9.0, d.z_hi):
        pos = np.concatenate([xy, np.full((200, 1), z, np.float32)], axis=1)
        flat3 = _np(fields.cell_index(mx.array(pos), cfg3))
        assert np.array_equal(flat3 % (d.fg * d.fg), flat2), z  # the floor cell below is fcell % FG²
        assert np.all(flat3 // (d.fg * d.fg) == math.floor(z / cfg3.world.field_cell)), z


def test_cell_index_clamps_z_and_wraps_xy(cfg3: Config) -> None:
    d, L, height = dims(cfg3), cfg3.world.size, cfg3.world.height
    top = float(np.nextafter(np.float32(height), np.float32(0)))
    seam = float(np.nextafter(np.float32(L), np.float32(0)))
    pos = mx.array([[0.0, 0.0, -0.5], [0.0, 0.0, top], [0.0, 0.0, height], [L, seam, 0.0]], dtype=mx.float32)
    got = _np(fields.cell_index(pos, cfg3))
    fg2 = d.fg * d.fg
    assert got[0] == 0 and got[1] == got[2] == (d.fz - 1) * fg2  # z is clamped, never wrapped
    assert 0 <= got[3] < fg2  # x = L wraps to cell 0 (the % FG seam guard)


def test_a_grounded_agent_is_in_the_floor_layer(cfg3: Config) -> None:
    """``r + floor_eps`` 1e-8 below the field cell: ``z_ground`` rounds down, not to the nearest float32 (the
    cell boundary), so the highest grounded z is still in layer 0 (§27.1)."""
    fc = cfg3.world.field_cell
    cfg = as_3d(cfg3, physics={"floor_eps": fc - 1.0 - 1e-8})
    d = dims(cfg)
    r_eps = cfg.physics.radius + cfg.physics.floor_eps
    assert r_eps < fc and float(np.float32(r_eps)) == fc  # the nearest float32 is the cell boundary
    assert d.z_lo <= d.z_ground < fc
    pos = mx.array([[1.0, 1.0, d.z_lo], [1.0, 1.0, d.z_ground]], dtype=mx.float32)
    assert np.all(_np(fields.cell_index(pos, cfg)) // (d.fg * d.fg) == 0)


def test_2d_bodies_refuse_3d_inputs(micro_cfg: Config, cfg3: Config) -> None:
    """The 2D cell index would drop z and 2D sampling would fold the layer into y: both refuse (static)."""
    pos3 = mx.zeros((4, 3), dtype=mx.float32)
    with pytest.raises(AssertionError, match="3D positions"):
        fields.cell_index(pos3, micro_cfg)
    d = dims(cfg3)
    res3, chem3 = mx.zeros(d.field_shape, dtype=mx.int32), mx.zeros((3, *d.field_shape), dtype=mx.float32)
    fcell = mx.zeros((4,), dtype=mx.int32)
    with pytest.raises(AssertionError, match="3D lattice"):
        fields.sample_fields(res3, chem3, fcell, mx.zeros((4,)), micro_cfg)


# --- the vertical profile and fertility -------------------------------------------------------------------


def test_vertical_profile_benthic(cfg3: Config) -> None:
    v = _np(fields.vertical_profile(cfg3))
    d, fc, lam = dims(cfg3), cfg3.world.field_cell, cfg3.fields.benthic_lambda
    assert v.shape == (d.fz, 1, 1) and v.dtype == np.float32
    want = np.float32([math.exp(-k * fc / lam) for k in range(d.fz)])
    assert np.array_equal(v.reshape(-1), want) and v[0, 0, 0] == 1.0
    assert np.all(np.diff(v.reshape(-1)) < 0)


def test_vertical_profile_photic(cfg3: Config) -> None:
    cfg = as_3d(cfg3, fields={"photic_amp": 0.6, "photic_lambda": 6.0})
    v = _np(fields.vertical_profile(cfg)).reshape(-1)
    d, fc = dims(cfg), cfg.world.field_cell
    benthic = [math.exp(-k * fc / 8.0) for k in range(d.fz)]
    want = [min(1.0, b + 0.6 * math.exp(-(d.fz - 1 - k) * fc / 6.0)) for k, b in enumerate(benthic)]
    assert np.array_equal(v, np.float32(want))
    assert v[0] == 1.0 and np.all(v <= 1.0) and v[-1] > v[-2]  # a second band under the surface
    # photic_amp = 0 is exactly the benthic profile, whatever photic_lambda holds.
    off = as_3d(cfg3, fields={"photic_amp": 0.0, "photic_lambda": 6.0})
    assert np.array_equal(_np(fields.vertical_profile(off)), _np(fields.vertical_profile(cfg3)))


def test_fertility_3d_floor_layer_is_2d_bitwise(micro_cfg: Config, cfg3: Config) -> None:
    v = _np(fields.vertical_profile(cfg3)).reshape(-1)
    seasons = {"season_period": 50, "season_amplitude": 0.5, "season_mode": "per_patch"}
    on2 = apply_overrides(micro_cfg, [f"fields.{k}={x}" for k, x in seasons.items()])
    env = Env(*(default_env(cfg3)[name] for name in ENV_FIELDS))
    step = mx.array(17, dtype=mx.int32)
    cases = [  # (2D config, its 3D twin, step, env): plain, seasonal, director env with halved gains
        (micro_cfg, cfg3, None, None),
        (on2, as_3d(on2), step, None),
        (micro_cfg, cfg3, step, env._replace(patch_gain=env.patch_gain * 0.5)),
    ]
    for seed, (c2, c3, t, e) in enumerate(cases):
        pp = _patches(c2, seed)
        f2, f3 = _np(fields.fertility(pp, c2, t, e)), _np(fields.fertility(pp, c3, t, e))
        assert f2.shape == (12, 12) and f3.shape == (6, 12, 12) and f3.dtype == np.float32
        assert np.array_equal(f3[0], f2), seed  # V[0] = 1 exactly
        assert np.array_equal(f3, f2[None] * v[:, None, None]), seed  # one float32 product per layer


def test_regen_rate_3d(micro_cfg: Config, cfg3: Config) -> None:
    pp = _patches(micro_cfg)
    r2, r3 = _np(fields.regen_rate(pp, micro_cfg)), _np(fields.regen_rate(pp, cfg3))
    assert r3.shape == dims(cfg3).field_shape and r3.dtype == np.int32
    assert np.array_equal(r3[0], r2)
    assert np.all(r3[1:] <= r3[:-1])  # the benthic band: never more food higher up
