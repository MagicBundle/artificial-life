"""§27.5 the static 3D lattice: the z-major cell index, the vertical food profile, and the 3D fertility,
regeneration and initial field shapes; the 2D bodies refuse 3D inputs instead of dropping z. §27.6 the field
dynamics: no-flux chemical and resource diffusion in z, the chemical deposit over the volume, and the 3D field
samples (mirrored z-gradients rotated into (fwd, lat, up))."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from alife import fields, layout
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


# --- field dynamics and sampling (§27.6) ------------------------------------------------------------------

NCH = layout.NCH


def _ref_chem_3d(chem: np.ndarray, dep: np.ndarray, cfg: Config) -> np.ndarray:
    """float64 reference: x and y rolled, z padded with its edge layers (the mirrored no-flux ghost)."""
    fi = cfg.fields
    c = chem.astype(np.float64) + dep.reshape(chem.shape) * fi.chem_deposit_scale
    cz = np.pad(c, ((0, 0), (1, 1), (0, 0), (0, 0)), mode="edge")
    lap = sum(np.roll(c, s, axis=a) for s in (1, -1) for a in (2, 3)) + cz[:, :-2] + cz[:, 2:] - 6 * c
    dd = np.array(fi.chem_diffusion)[:, None, None, None]
    keep = 1 - np.array(fi.chem_decay)[:, None, None, None]
    return np.clip(keep * (c + dd * lap), 0, fi.chem_max)


def test_update_chem_3d_matches_reference_and_clamps(cfg3: Config) -> None:
    d = dims(cfg3)
    rng = np.random.default_rng(3)
    chem = rng.uniform(0, 1.2 * cfg3.fields.chem_max, (NCH, *d.field_shape)).astype(np.float32)
    dep = rng.integers(0, 3000, NCH * d.n_cells).astype(np.int32)
    out = fields.update_chem(mx.array(chem), mx.array(dep), cfg3)
    assert out.dtype == mx.float32 and out.shape == (NCH, d.fz, d.fg, d.fg)
    o = _np(out)
    np.testing.assert_allclose(o, _ref_chem_3d(chem, dep, cfg3), atol=1e-5)
    assert o.min() >= 0 and o.max() <= cfg3.fields.chem_max


def test_update_chem_3d_has_no_flux_through_the_floor_or_the_surface(cfg3: Config) -> None:
    d = dims(cfg3)
    zero_dep = mx.zeros(NCH * d.n_cells, mx.int32)
    lam = cfg3.fields.chem_decay
    # A uniform field decays by exactly (1 − λ_c) per step in every layer, the floor and the surface included:
    # the mirrored stencil vanishes there, so the boundary layers lose nothing to the outside.
    c = mx.full((NCH, *d.field_shape), 3.0, mx.float32)
    for t in range(1, 4):
        c = fields.update_chem(c, zero_dep, cfg3)
        for ch in range(NCH):
            assert np.allclose(_np(c[ch]), 3.0 * (1 - lam[ch]) ** t, rtol=1e-6, atol=0)
    # Without decay, diffusion alone conserves each channel's mass over many steps.
    nodecay = as_3d(cfg3, fields={"chem_decay": [0.0, 0.0, 0.0]})
    c0 = np.random.default_rng(4).uniform(0, 4, (NCH, *d.field_shape)).astype(np.float32)
    c = mx.array(c0)
    for _ in range(50):
        c = fields.update_chem(c, zero_dep, nodecay)
    np.testing.assert_allclose(_np(c).sum(axis=(1, 2, 3)), c0.sum(axis=(1, 2, 3)), rtol=1e-5)
    assert _np(c).std() < c0.std()
    # z is not periodic: a source on the floor reaches the layer above, never the surface layer.
    src = np.zeros((NCH, *d.field_shape), np.float32)
    src[:, 0, 5, 5] = 8.0
    one = _np(fields.update_chem(mx.array(src), zero_dep, nodecay))
    assert np.all(one[:, 1, 5, 5] > 0) and not one[:, -1].any()
    np.testing.assert_allclose(one.sum(axis=(1, 2, 3)), 8.0, rtol=1e-6)


def _ref_diffuse_3d(r: np.ndarray, d_q8: int) -> np.ndarray:
    q = (r.astype(np.int64) * d_q8) >> 8
    qz = np.pad(q, ((1, 1), (0, 0), (0, 0)), mode="edge")
    return r - 6 * q + sum(np.roll(q, s, axis=a) for s in (1, -1) for a in (1, 2)) + qz[:-2] + qz[2:]


def test_update_resource_3d_regen_and_conservation(cfg3: Config) -> None:
    d, cap = dims(cfg3), cfg3.fields.resource_cap
    pp = _patches(cfg3, 5)
    rate = _np(fields.regen_rate(pp, cfg3)).astype(np.int64)
    rng = np.random.default_rng(5)
    res = rng.integers(0, cap + 1, d.field_shape)
    res[0, 0, :4] = [0, cap, cap - 1, 3 * cap]  # empty, at cap, one below, above cap (corpses on the floor)
    res[-1, 1, :2] = [(1 << 24) - 1, cap - 400]  # the surface layer
    out, inc = fields.update_resource(mx.array(res.astype(np.int32)), pp, cfg3)
    assert out.dtype == inc.dtype == mx.int32 and out.shape == inc.shape == d.field_shape
    o, i = _np(out).astype(np.int64), _np(inc).astype(np.int64)
    assert np.array_equal(i, np.clip(np.minimum(rate, cap - res), 0, None))
    assert o.sum() == res.sum() + i.sum(), "INJECT_REGEN = Σ inc; the 3D diffusion conserves exactly"
    assert np.array_equal(o, _ref_diffuse_3d(res + i, cfg3.fields.resource_diffusion_q8))
    assert o.min() >= 0


def _ref_sample_3d(res: np.ndarray, chem: np.ndarray, fcell: np.ndarray, yaw: np.ndarray, pitch: np.ndarray,
                   cfg: Config) -> dict[str, np.ndarray]:  # fmt: skip
    """float64 reference of the 3D §5 field inputs: explicit neighbour lookups (x, y periodic, z clamped with
    the divisor kept at 2) and the (fwd, lat, up) projections of §27.2."""
    d, cap, half = dims(cfg), cfg.fields.resource_cap, cfg.sensing.chem_half_sat
    fg, fz = d.fg, d.fz
    cz, cy, cx = fcell // (fg * fg), (fcell // fg) % fg, fcell % fg
    zp, zm = np.minimum(cz + 1, fz - 1), np.maximum(cz - 1, 0)
    th, ph = yaw.astype(np.float64), pitch.astype(np.float64)
    axes = (
        np.stack([np.cos(ph) * np.cos(th), np.cos(ph) * np.sin(th), np.sin(ph)], -1),
        np.stack([-np.sin(th), np.cos(th), np.zeros_like(th)], -1),
        np.stack([-np.sin(ph) * np.cos(th), -np.sin(ph) * np.sin(th), np.cos(ph)], -1),
    )

    def grad(f: np.ndarray) -> np.ndarray:  # (N,3) world-frame central differences
        f = f.astype(np.float64)
        return np.stack(
            [
                f[cz, cy, (cx + 1) % fg] - f[cz, cy, (cx - 1) % fg],
                f[cz, (cy + 1) % fg, cx] - f[cz, (cy - 1) % fg, cx],
                f[zp, cy, cx] - f[zm, cy, cx],
            ],
            -1,
        )

    def body(g: np.ndarray) -> np.ndarray:
        return np.stack([(g * a).sum(-1) for a in axes], -1)

    out = {"res": np.minimum(res[cz, cy, cx] / cap, layout.RES_SAT) / layout.RES_SAT}
    out["res_grad"] = np.clip(layout.RES_GRAD_GAIN * body(grad(res) / (2 * cap)), -1, 1)
    cv = chem[:, cz, cy, cx].T.astype(np.float64)
    out["chem"] = cv / (cv + half)
    out["chem_grad"] = np.stack([np.tanh(body(grad(chem[c]) / 2) / half) for c in range(NCH)], axis=1)
    return out


def _sample_3d(res: np.ndarray, chem: np.ndarray, pos: np.ndarray, yaw: np.ndarray, pitch: np.ndarray,
               cfg: Config):  # fmt: skip
    fcell = fields.cell_index(mx.array(pos.astype(np.float32)), cfg)
    obs = fields.sample_fields(
        mx.array(res.astype(np.int32)),
        mx.array(chem.astype(np.float32)),
        fcell,
        mx.array(yaw.astype(np.float32)),
        cfg,
        pitch=mx.array(pitch.astype(np.float32)),
    )
    return _np(fcell), {k: _np(v) for k, v in obs._asdict().items()}


def test_sample_fields_3d_matches_reference(cfg3: Config) -> None:
    d, L, cap = dims(cfg3), cfg3.world.size, cfg3.fields.resource_cap
    rng = np.random.default_rng(1)
    n = d.n
    res = rng.integers(0, 3 * cap, d.field_shape)
    res[0, 0, :4] = [0, 2 * cap, 3 * cap, (1 << 24) - 1]  # saturation and the hard cap
    chem = rng.uniform(0, cfg3.fields.chem_max, (NCH, *d.field_shape))
    pos = np.stack([rng.uniform(0, L, n), rng.uniform(0, L, n), rng.uniform(d.z_lo, d.z_hi, n)], 1)
    pos[:8, 2] = [d.z_lo, d.z_lo, 1.5, d.z_hi, d.z_hi, 20.5, 12.0, 4.0]  # the floor and surface layers
    pos[:4, :2] = [[0.5, 0.5], [L - 0.5, 2.0], [2.0, L - 0.5], [L - 0.5, L - 0.5]]  # at the x/y seams
    yaw = rng.uniform(-math.pi, math.pi, n)
    pitch = rng.uniform(-layout.PITCH_MAX, layout.PITCH_MAX, n)
    fcell, obs = _sample_3d(res, chem, pos, yaw, pitch, cfg3)
    assert set(fcell[:6] // (d.fg * d.fg)) == {0, d.fz - 1}
    ref = _ref_sample_3d(res, chem, fcell, yaw, pitch, cfg3)
    shapes = {"res": (n,), "res_grad": (n, 3), "chem": (n, NCH), "chem_grad": (n, NCH, 3)}
    for k, shape in shapes.items():
        assert obs[k].shape == shape and obs[k].dtype == np.float32, k
        np.testing.assert_allclose(obs[k], ref[k], atol=2e-6, err_msg=k)
        assert np.all(np.isfinite(obs[k])) and np.all(np.abs(obs[k]) <= 1), k


def test_sample_fields_3d_vertical_ramp(cfg3: Config) -> None:
    """R and chem grow with height. Level, the up component is positive: the full central difference in an
    interior layer and half of it (the mirror) in the floor and surface layers. Nose up by φ, the same
    gradient splits into fwd = g·sin φ and up = g·cos φ."""
    d, fc, cap = dims(cfg3), cfg3.world.field_cell, cfg3.fields.resource_cap
    k = np.arange(d.fz)
    res = np.broadcast_to(100 * k[:, None, None], d.field_shape).copy()  # dR/dz = 100 per layer
    chem = np.stack([np.broadcast_to(0.1 * (c + 1) * k[:, None, None], d.field_shape) for c in range(NCH)])
    n = d.n
    pos = np.tile([[5.5 * fc, 6.5 * fc, 2.5 * fc]], (n, 1))  # an interior layer
    pos[1, 2], pos[2, 2] = d.z_lo, d.z_hi  # the floor and the surface layers
    pitch = np.zeros(n)
    pitch[3] = math.pi / 4
    _, obs = _sample_3d(res, chem, pos, np.zeros(n), pitch, cfg3)
    g = layout.RES_GRAD_GAIN * 200 / (2 * cap)  # interior: (R[k+1] − R[k−1]) / (2·cap), with the gain
    np.testing.assert_allclose(obs["res_grad"][0], [0.0, 0.0, g], atol=1e-6)
    for row in (1, 2):
        np.testing.assert_allclose(obs["res_grad"][row], [0.0, 0.0, g / 2], atol=1e-6)
    s45 = math.sin(math.pi / 4)
    np.testing.assert_allclose(obs["res_grad"][3], [g * s45, 0.0, g * s45], atol=1e-6)
    for c in range(NCH):
        gz = 0.1 * (c + 1)  # interior central difference / 2
        np.testing.assert_allclose(obs["chem_grad"][0, c], np.tanh([0.0, 0.0, gz]), atol=1e-6)
        np.testing.assert_allclose(obs["chem_grad"][1, c], np.tanh([0.0, 0.0, gz / 2]), atol=1e-6)
        np.testing.assert_allclose(obs["chem_grad"][3, c], np.tanh([gz * s45, 0.0, gz * s45]), atol=1e-6)


def test_sample_fields_3d_needs_the_pitch(cfg3: Config) -> None:
    d = dims(cfg3)
    res, chem = mx.zeros(d.field_shape, dtype=mx.int32), mx.zeros((NCH, *d.field_shape), dtype=mx.float32)
    with pytest.raises(AssertionError, match="pitch"):
        fields.sample_fields(res, chem, mx.zeros((4,), dtype=mx.int32), mx.zeros((4,)), cfg3)


def test_deposit_chem_3d_exact(cfg3: Config) -> None:
    """Flat index ``c·n_cells + fcell`` over the whole volume: the top layer's cells (beyond FG², where a 2D
    channel stride would put them in the next channel) land in their own channel."""
    d = dims(cfg3)
    rng = np.random.default_rng(2)
    emit = rng.uniform(0, 1, (d.n, NCH)).astype(np.float32)
    fcell = rng.integers(0, d.n_cells, d.n).astype(np.int32)
    fcell[:3] = [0, d.n_cells - 1, d.fg * d.fg]  # first cell, last cell, first cell of layer 1
    emit[1] = 0.5
    active = rng.random(d.n) < 0.7
    active[:3] = True
    dep = fields.deposit_chem(mx.array(emit), mx.array(fcell), mx.array(active), cfg3)
    assert dep.dtype == mx.int32 and dep.shape == (NCH * d.n_cells,)
    quanta = np.floor(emit.astype(np.float64) * 256).astype(np.int64)
    ref = np.zeros(NCH * d.n_cells, np.int64)
    for c in range(NCH):
        np.add.at(ref, c * d.n_cells + fcell[active], quanta[active, c])
    assert np.array_equal(_np(dep), ref)
    grid = _np(dep).reshape(NCH, *d.field_shape)  # the layout update_chem reads
    assert np.array_equal(grid[:, -1, -1, -1], ref.reshape(NCH, -1)[:, -1]) and grid[2, -1, -1, -1] > 0
