"""§13 fields: patches and fertility (WP0 hand-over), cell index, sampling, chemicals, resource, drift."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from alife import fields, layout
from alife.config import Config, apply_overrides, dims


def _np(a: mx.array) -> np.ndarray:
    return np.array(a)


def _fert_reference(patch_pos: np.ndarray, cfg: Config) -> np.ndarray:
    """Direct 2-D torus Gaussian sum per cell (not separable), float64."""
    fg, L, fc = dims(cfg).fg, cfg.world.size, cfg.world.field_cell
    c = (np.arange(fg) + 0.5) * fc
    yy, xx = np.meshgrid(c, c, indexing="ij")  # [y, x]
    total = np.zeros((fg, fg))
    for px, py in patch_pos.astype(np.float64):
        dx = xx - px
        dy = yy - py
        dx -= L * np.round(dx / L)
        dy -= L * np.round(dy / L)
        total += np.exp(-(dx * dx + dy * dy) / (2 * cfg.fields.patch_sigma**2))
    return np.clip(cfg.fields.fertility_base + total, 0, 1)


def _keys(seed: int) -> tuple[mx.array, mx.array]:
    k = mx.random.split(mx.random.key(seed), 2)
    return k[0], k[1]


def test_init_patches(tiny_cfg: Config) -> None:
    d = dims(tiny_cfg)
    pos, vel = fields.init_patches(*_keys(0), tiny_cfg)
    assert pos.shape == (d.pp, 2) and vel.shape == (d.pp, 2)
    assert pos.dtype == mx.float32 and vel.dtype == mx.float32
    p, v = _np(pos), _np(vel)
    assert np.all(p >= 0) and np.all(p < tiny_cfg.world.size)
    assert np.allclose(np.linalg.norm(v, axis=1), tiny_cfg.fields.patch_drift, rtol=1e-5)
    pos2, vel2 = fields.init_patches(*_keys(0), tiny_cfg)
    assert np.array_equal(_np(pos2), p) and np.array_equal(_np(vel2), v)
    static = apply_overrides(tiny_cfg, ["fields.patch_drift=0"])
    assert np.all(_np(fields.init_patches(*_keys(0), static)[1]) == 0)


def test_init_patches_directions_uniform() -> None:
    cfg = apply_overrides(Config(), ["fields.n_patches=4000"])
    _, vel = fields.init_patches(*_keys(1), cfg)
    ang = np.arctan2(_np(vel)[:, 1], _np(vel)[:, 0])
    hist, _ = np.histogram(ang, bins=8, range=(-math.pi, math.pi))
    assert hist.min() > 0.8 * 500 and hist.max() < 1.2 * 500


def test_fertility_matches_reference(tiny_cfg: Config, micro_cfg: Config) -> None:
    for cfg in (tiny_cfg, micro_cfg):
        for seed in range(3):
            pos, _ = fields.init_patches(*_keys(seed), cfg)
            fert = fields.fertility(pos, cfg)
            assert fert.shape == (dims(cfg).fg,) * 2 and fert.dtype == mx.float32
            ref = _fert_reference(_np(pos), cfg)
            assert np.max(np.abs(_np(fert) - ref)) < 1e-5, "separable form equals the 2-D torus Gaussian"
            assert _np(fert).min() >= 0 and _np(fert).max() <= 1


def test_fertility_seam_and_peak(micro_cfg: Config) -> None:
    """A patch at the seam wraps symmetrically; the cell under a patch centre is the peak."""
    L, fg = micro_cfg.world.size, dims(micro_cfg).fg
    cfg = apply_overrides(micro_cfg, ["fields.n_patches=1", "fields.fertility_base=0"])
    fert = _np(fields.fertility(mx.array([[0.0, 0.0]]), cfg))
    assert np.allclose(fert, fert[::-1, ::-1], atol=1e-6)  # symmetric about the corner
    assert np.allclose(fert, fert.T, atol=1e-6)
    centre = mx.array([[(5 + 0.5) * cfg.world.field_cell, (7 + 0.5) * cfg.world.field_cell]])
    f2 = _np(fields.fertility(centre, cfg))
    assert np.unravel_index(np.argmax(f2), f2.shape) == (7, 5) and f2[7, 5] == 1.0
    assert fg == round(L / cfg.world.field_cell)


def test_regen_rate(tiny_cfg: Config) -> None:
    pos, _ = fields.init_patches(*_keys(4), tiny_cfg)
    rate = fields.regen_rate(pos, tiny_cfg)
    assert rate.dtype == mx.int32 and rate.shape == (dims(tiny_cfg).fg,) * 2
    r = _np(rate)
    assert r.min() >= 0 and r.max() <= tiny_cfg.fields.regen_max
    ref = np.floor(tiny_cfg.fields.regen_max * _np(fields.fertility(pos, tiny_cfg)))
    assert np.array_equal(r, ref.astype(np.int32))
    base = math.floor(tiny_cfg.fields.regen_max * tiny_cfg.fields.fertility_base)
    assert r.min() >= base


# --- WP1: cell index, sampling, chemicals, resource and patch drift ---------------------------------------


def _ref_cell(pos: np.ndarray, cfg: Config) -> np.ndarray:
    fg = dims(cfg).fg
    c = np.floor(pos.astype(np.float32) / np.float32(cfg.world.field_cell)).astype(np.int64) % fg
    return (c[:, 1] * fg + c[:, 0]).astype(np.int32)


def _ref_sample(
    res: np.ndarray, chem: np.ndarray, fcell: np.ndarray, heading: np.ndarray, cfg: Config
) -> dict[str, np.ndarray]:
    """float64 reference of the §5 field inputs, with explicit periodic neighbour lookups."""
    fg, cap, half = dims(cfg).fg, cfg.fields.resource_cap, cfg.sensing.chem_half_sat
    cy, cx = fcell // fg, fcell % fg
    th = heading.astype(np.float64)
    cos, sin = np.cos(th), np.sin(th)

    def body(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        return np.stack([gx * cos + gy * sin, gy * cos - gx * sin], axis=-1)

    def grad(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f = f.astype(np.float64)
        return f[cy, (cx + 1) % fg] - f[cy, (cx - 1) % fg], f[(cy + 1) % fg, cx] - f[(cy - 1) % fg, cx]

    out = {"res": np.minimum(res[cy, cx] / cap, layout.RES_SAT) / layout.RES_SAT}
    gx, gy = grad(res)
    out["res_grad"] = np.clip(layout.RES_GRAD_GAIN * body(gx / (2 * cap), gy / (2 * cap)), -1, 1)
    cv = chem[:, cy, cx].T.astype(np.float64)
    out["chem"] = cv / (cv + half)
    cg = []
    for c in range(layout.NCH):
        gx, gy = grad(chem[c])
        cg.append(np.tanh(body(gx / 2, gy / 2) / half))
    out["chem_grad"] = np.stack(cg, axis=1)
    return out


def _sample(res: np.ndarray, chem: np.ndarray, pos: np.ndarray, heading: np.ndarray, cfg: Config):
    fcell = fields.cell_index(mx.array(pos.astype(np.float32)), cfg)
    obs = fields.sample_fields(
        mx.array(res.astype(np.int32)),
        mx.array(chem.astype(np.float32)),
        fcell,
        mx.array(heading.astype(np.float32)),
        cfg,
    )
    return _np(fcell), {k: _np(v) for k, v in obs._asdict().items()}


def _cell_centre(cx: int, cy: int, cfg: Config) -> list[float]:
    return [(cx + 0.5) * cfg.world.field_cell, (cy + 0.5) * cfg.world.field_cell]


def test_cell_index(tiny_cfg: Config) -> None:
    d, L, fc = dims(tiny_cfg), tiny_cfg.world.size, tiny_cfg.world.field_cell
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, L, (d.n, 2)).astype(np.float32)
    below_l = np.nextafter(np.float32(L), np.float32(0))
    pos[:6] = [[0, 0], [below_l, below_l], [fc, 0], [fc - 1e-4, 2 * fc], [0, below_l], [L - fc, fc]]
    got = fields.cell_index(mx.array(pos), tiny_cfg)
    assert got.dtype == mx.int32 and got.shape == (d.n,)
    g = _np(got)
    assert np.array_equal(g, _ref_cell(pos, tiny_cfg))
    assert g[:6].tolist() == [0, d.fg * d.fg - 1, 1, 2 * d.fg, (d.fg - 1) * d.fg, d.fg + d.fg - 1]
    assert g.min() >= 0 and g.max() < d.fg * d.fg


def test_sample_fields_matches_reference(tiny_cfg: Config) -> None:
    d, L = dims(tiny_cfg), tiny_cfg.world.size
    rng = np.random.default_rng(1)
    cap = tiny_cfg.fields.resource_cap
    res = rng.integers(0, 3 * cap, (d.fg, d.fg))
    res[0, :4] = [0, 2 * cap, 3 * cap, (1 << 24) - 1]  # saturation and the hard cap
    chem = rng.uniform(0, tiny_cfg.fields.chem_max, (layout.NCH, d.fg, d.fg))
    pos = rng.uniform(0, L, (d.n, 2))
    pos[:4] = [_cell_centre(x, 0, tiny_cfg) for x in range(4)]
    heading = rng.uniform(-math.pi, math.pi, d.n)
    fcell, obs = _sample(res, chem, pos, heading, tiny_cfg)
    ref = _ref_sample(res, chem, fcell, heading, tiny_cfg)
    shapes = {
        "res": (d.n,),
        "res_grad": (d.n, 2),
        "chem": (d.n, layout.NCH),
        "chem_grad": (d.n, layout.NCH, 2),
    }
    for k, shape in shapes.items():
        assert obs[k].shape == shape and obs[k].dtype == np.float32, k
        np.testing.assert_allclose(obs[k], ref[k], atol=2e-6, err_msg=k)
        assert np.all(np.isfinite(obs[k])) and np.all(np.abs(obs[k]) <= 1), k
    assert obs["res"][:4].tolist() == [0.0, 1.0, 1.0, 1.0]


@pytest.mark.parametrize("heading", [0.0, math.pi / 2, -math.pi / 2, math.pi / 3, -math.pi])
def test_planted_ramp_gradient_sign_and_rotation(micro_cfg: Config, heading: float) -> None:
    """R and chem grow along +x (resource) or +y (chem); body-frame gradients rotate with the heading."""
    d = dims(micro_cfg)
    xs = np.arange(d.fg)
    res = np.broadcast_to(100 * xs[None, :], (d.fg, d.fg)).copy()  # dR/dx = 100 per cell
    chem = np.zeros((layout.NCH, d.fg, d.fg))
    for c in range(layout.NCH):
        chem[c] = 0.1 * (c + 1) * xs[:, None]  # d chem_c / dy = 0.1·(c+1) per cell
    pos = np.zeros((d.n, 2))
    pos[:] = _cell_centre(5, 6, micro_cfg)  # interior cell: no ramp discontinuity at the seam
    _, obs = _sample(res, chem, pos, np.full(d.n, heading), micro_cfg)
    g = layout.RES_GRAD_GAIN * 200 / (2 * micro_cfg.fields.resource_cap)  # world-frame +x component
    exp_res = np.clip([g * math.cos(heading), -g * math.sin(heading)], -1, 1)
    np.testing.assert_allclose(obs["res_grad"][0], exp_res, atol=1e-6)
    for c in range(layout.NCH):
        gy = 0.1 * (c + 1)  # central difference / 2
        exp = np.tanh([gy * math.sin(heading), gy * math.cos(heading)])
        np.testing.assert_allclose(obs["chem_grad"][0, c], exp, atol=1e-6)
    if heading == 0.0:  # facing +x: the resource ramp is straight ahead, the chem ramp to the left (+lat)
        assert obs["res_grad"][0, 0] > 0 and obs["res_grad"][0, 1] == 0
        assert np.all(obs["chem_grad"][0, :, 0] == 0) and np.all(obs["chem_grad"][0, :, 1] > 0)


def test_gradient_wraps_at_seam(micro_cfg: Config) -> None:
    d = dims(micro_cfg)
    res = np.zeros((d.fg, d.fg), np.int64)
    res[3, d.fg - 1] = 4000  # left of cell (0, 3) across the x seam
    res[0, 7] = 2000  # above cell (7, fg−1) across the y seam
    pos = np.zeros((d.n, 2))
    pos[0], pos[1] = _cell_centre(0, 3, micro_cfg), _cell_centre(7, d.fg - 1, micro_cfg)
    _, obs = _sample(res, np.zeros((layout.NCH, d.fg, d.fg)), pos, np.zeros(d.n), micro_cfg)
    cap = micro_cfg.fields.resource_cap
    np.testing.assert_allclose(obs["res_grad"][0], [-4 * 4000 / (2 * cap), 0], atol=1e-6)
    np.testing.assert_allclose(obs["res_grad"][1], [0, 4 * 2000 / (2 * cap)], atol=1e-6)


def test_deposit_chem_exact(tiny_cfg: Config) -> None:
    d = dims(tiny_cfg)
    fg2 = d.fg * d.fg
    rng = np.random.default_rng(2)
    emit = rng.uniform(0, 1, (d.n, layout.NCH)).astype(np.float32)
    emit[:3] = [[0.0, 0.999999, 0.5], [1.0 / 256, 255.5 / 256, 0.0], [0.0, 0.0, 0.0]]
    fcell = rng.integers(0, 6, d.n).astype(np.int32)  # heavy duplication in a few cells
    active = rng.random(d.n) < 0.7
    active[:3] = True
    dep = fields.deposit_chem(mx.array(emit), mx.array(fcell), mx.array(active), tiny_cfg)
    assert dep.dtype == mx.int32 and dep.shape == (layout.NCH * fg2,)
    quanta = np.floor(emit.astype(np.float64) * 256).astype(np.int64)
    assert quanta[:2].tolist() == [[0, 255, 128], [1, 255, 0]]
    ref = np.zeros(layout.NCH * fg2, np.int64)
    for c in range(layout.NCH):
        np.add.at(ref, c * fg2 + fcell[active], quanta[active, c])
    assert np.array_equal(_np(dep), ref)
    assert _np(dep).sum() == quanta[active].sum()
    none = fields.deposit_chem(mx.array(emit), mx.array(fcell), mx.zeros(d.n, mx.bool_), tiny_cfg)
    assert not _np(none).any()


def _ref_chem(chem: np.ndarray, dep: np.ndarray, cfg: Config) -> np.ndarray:
    fi = cfg.fields
    c = chem.astype(np.float64) + dep.reshape(chem.shape) * fi.chem_deposit_scale
    lap = sum(np.roll(c, s, axis=a) for s in (1, -1) for a in (1, 2)) - 4 * c
    dd = np.array(fi.chem_diffusion)[:, None, None]
    keep = 1 - np.array(fi.chem_decay)[:, None, None]
    return np.clip(keep * (c + dd * lap), 0, fi.chem_max)


def test_update_chem_matches_reference_and_clamps(tiny_cfg: Config) -> None:
    d = dims(tiny_cfg)
    rng = np.random.default_rng(3)
    chem = rng.uniform(0, 1.2 * tiny_cfg.fields.chem_max, (layout.NCH, d.fg, d.fg)).astype(np.float32)
    dep = rng.integers(0, 3000, layout.NCH * d.fg * d.fg).astype(np.int32)
    out = fields.update_chem(mx.array(chem), mx.array(dep), tiny_cfg)
    assert out.dtype == mx.float32 and out.shape == chem.shape
    o = _np(out)
    np.testing.assert_allclose(o, _ref_chem(chem, dep, tiny_cfg), atol=1e-5)
    assert o.min() >= 0 and o.max() <= tiny_cfg.fields.chem_max


def test_chem_decay_and_conservation(micro_cfg: Config) -> None:
    d = dims(micro_cfg)
    zero_dep = mx.zeros(layout.NCH * d.fg * d.fg, mx.int32)
    lam = micro_cfg.fields.chem_decay
    # A uniform field decays by exactly (1 − λ_c) per step: the stencil vanishes.
    c = mx.full((layout.NCH, d.fg, d.fg), 3.0, mx.float32)
    for t in range(1, 6):
        c = fields.update_chem(c, zero_dep, micro_cfg)
        for ch in range(layout.NCH):
            assert np.allclose(_np(c[ch]), 3.0 * (1 - lam[ch]) ** t, rtol=1e-6, atol=0)
    # A random field without deposits loses exactly the fraction λ_c of its mass (diffusion conserves).
    rng = np.random.default_rng(4)
    c0 = rng.uniform(0, 4, (layout.NCH, d.fg, d.fg)).astype(np.float32)
    c1 = _np(fields.update_chem(mx.array(c0), zero_dep, micro_cfg))
    for ch in range(layout.NCH):
        assert np.isclose(c1[ch].sum(), (1 - lam[ch]) * c0[ch].sum(), rtol=1e-5)
    # With λ = 0, diffusion alone conserves mass over many steps and spreads a point source.
    nodecay = apply_overrides(micro_cfg, ["fields.chem_decay=0,0,0"])
    c = mx.array(c0)
    for _ in range(50):
        c = fields.update_chem(c, zero_dep, nodecay)
    np.testing.assert_allclose(_np(c).sum(axis=(1, 2)), c0.sum(axis=(1, 2)), rtol=1e-5)
    assert _np(c).std() < c0.std()


def test_full_emitter_adds_one(micro_cfg: Config) -> None:
    """``chem_deposit_scale = 1/256``: 256 quanta (a full-strength emitter) add 1.0 before the stencil."""
    d = dims(micro_cfg)
    still = apply_overrides(micro_cfg, ["fields.chem_decay=0,0,0", "fields.chem_diffusion=0,0,0"])
    emit = mx.zeros((d.n, layout.NCH)).at[0].add(mx.array([255.9 / 256, 0.0, 0.5]))
    fcell = mx.full((d.n,), 17, mx.int32)
    active = mx.arange(d.n) == 0
    dep = fields.deposit_chem(emit, fcell, active, still)
    out = _np(fields.update_chem(mx.zeros((layout.NCH, d.fg, d.fg)), dep, still))
    assert out[0, 1, 5] == 255 / 256 and out[2, 1, 5] == 0.5 and out.sum() == 255 / 256 + 0.5
    full = _np(
        fields.update_chem(mx.zeros((layout.NCH, d.fg, d.fg)), mx.full(dep.shape, 256, mx.int32), still)
    )
    assert np.all(full == 1.0)


def test_update_resource_regen_cap_and_conservation(tiny_cfg: Config) -> None:
    d = dims(tiny_cfg)
    cap = tiny_cfg.fields.resource_cap
    pos, _ = fields.init_patches(*_keys(5), tiny_cfg)
    rate = _np(fields.regen_rate(pos, tiny_cfg)).astype(np.int64)
    rng = np.random.default_rng(5)
    res = rng.integers(0, cap + 1, (d.fg, d.fg))
    res[0, :4] = [0, cap, cap - 1, 3 * cap]  # empty, at cap, one below, above cap (corpses)
    res[1, :2] = [(1 << 24) - 1, cap - 400]
    out, inc = fields.update_resource(mx.array(res.astype(np.int32)), pos, tiny_cfg)
    assert out.dtype == mx.int32 and inc.dtype == mx.int32
    assert out.shape == inc.shape == (d.fg, d.fg)
    o, i = _np(out).astype(np.int64), _np(inc).astype(np.int64)
    assert np.array_equal(i, np.clip(np.minimum(rate, cap - res), 0, None))
    assert i[0, 1] == 0 and i[0, 2] == min(1, rate[0, 2]) and i[0, 3] == 0 and i[1, 0] == 0
    assert np.all(i <= rate) and np.all(res + i <= np.maximum(res, cap))
    assert o.sum() == res.sum() + i.sum(), "INJECT_REGEN = Σ inc; diffusion conserves exactly"
    assert o.min() >= 0
    # Integer diffusion matches the explicit flux stencil.
    r1 = res + i
    q = (r1 * tiny_cfg.fields.resource_diffusion_q8) >> 8
    ref = r1 - 4 * q + sum(np.roll(q, s, axis=a) for s in (1, -1) for a in (0, 1))
    assert np.array_equal(o, ref)
    # Without diffusion the field is just R + inc; a full field does not regenerate.
    nodiff = apply_overrides(tiny_cfg, ["fields.resource_diffusion_q8=0"])
    o2, _ = fields.update_resource(mx.array(res.astype(np.int32)), pos, nodiff)
    assert np.array_equal(_np(o2), r1)
    _, inc_full = fields.update_resource(mx.full((d.fg, d.fg), cap, mx.int32), pos, tiny_cfg)
    assert not _np(inc_full).any()


def test_resource_reaches_cap_over_time(micro_cfg: Config) -> None:
    d = dims(micro_cfg)
    cfg = apply_overrides(micro_cfg, ["fields.resource_diffusion_q8=0"])
    pos, _ = fields.init_patches(*_keys(6), cfg)
    r = mx.zeros((d.fg, d.fg), mx.int32)
    injected = 0
    steps = -(-cfg.fields.resource_cap // math.floor(cfg.fields.regen_max * cfg.fields.fertility_base)) + 1
    for _ in range(steps):
        r, inc = fields.update_resource(r, pos, cfg)
        injected += int(_np(inc).astype(np.int64).sum())
    assert np.all(_np(r) == cfg.fields.resource_cap)
    assert injected == cfg.fields.resource_cap * d.fg * d.fg


def test_drift_patches_moves_and_wraps(micro_cfg: Config) -> None:
    L = micro_cfg.world.size
    below_l = float(np.nextafter(np.float32(L), np.float32(0)))
    pos = mx.array([[1.0, 2.0], [L - 0.002, 5.0], [0.001, 0.001], [below_l, 0.0], [10.0, 10.0]])
    vel = mx.array([[0.003, -0.004], [0.005, 0.0], [-0.003, -0.004], [1e-6, 0.0], [-1e-9, 0.0]])
    out = fields.drift_patches(pos, vel, micro_cfg)
    assert out.dtype == mx.float32 and out.shape == (5, 2)
    o = _np(out)
    assert np.all(o >= 0) and np.all(o < L)
    np.testing.assert_allclose(o[0], [1.003, 1.996], atol=1e-6)
    np.testing.assert_allclose(o[1], [0.003, 5.0], atol=1e-4)
    np.testing.assert_allclose(o[2], [L - 0.002, L - 0.003], atol=1e-4)
    # Many steps: displacement accumulates as n·vel (mod L).
    p, v = fields.init_patches(*_keys(8), micro_cfg)
    p0 = _np(p).astype(np.float64)
    for _ in range(200):
        p = fields.drift_patches(p, v, micro_cfg)
    disp = _np(p) - p0
    disp -= L * np.round(disp / L)
    np.testing.assert_allclose(disp, 200 * _np(v), atol=1e-3)  # f32 rounding of positions near L
    assert np.all(_np(p) >= 0) and np.all(_np(p) < L)


@pytest.mark.parametrize("d_q8", [1, 3, 4, 17, 63, 64])
def test_resource_diffusion_respects_hard_cap(micro_cfg: Config, d_q8: int) -> None:
    """Fields within a few hundred quanta of R_HARD stay ≤ R_HARD (R_HARD ≡ −1 mod 256) and ≥ 0."""
    d = dims(micro_cfg)
    cfg = apply_overrides(micro_cfg, [f"fields.resource_diffusion_q8={d_q8}"])
    r_hard = (1 << 24) - 1
    rng = np.random.default_rng(d_q8)
    pos, _ = fields.init_patches(*_keys(9), cfg)
    for _ in range(5):
        res = rng.integers(r_hard - 600, r_hard + 1, (d.fg, d.fg))
        res[rng.random((d.fg, d.fg)) < 0.3] = r_hard
        out, inc = fields.update_resource(mx.array(res.astype(np.int32)), pos, cfg)
        o = _np(out).astype(np.int64)
        assert not _np(inc).any()  # above resource_cap: no regeneration
        assert o.max() <= r_hard and o.min() >= 0
        assert o.sum() == res.sum()
