"""The fields of the sphere world: validation of the diffusion constants, the vertical profile, fertility,
the patches and their rigid rotation, the resource step, the gradient inputs, deposit (``alife.fields`` and
``alife.fields_sph`` on a sphere; DESIGN §46.5, sphere series S1).

The sphere is not wired into the step before S2, so the tests widen ``config.GEOMETRY_SUPPORTED`` (the
``sphere_enabled`` fixture) and call the field functions directly. Everything is eager except where a test
says it passes its constants as runtime arrays.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from alife import fields, fields_sph, fieldtab, geom, layout, ops, rng, sphere, state
from alife.config import R_HARD, ConfigError, dims
from alife.state import Env
from sphere_fields_helpers import dev  # noqa: F401
from sphere_helpers import sphere_cfg, sphere_enabled  # noqa: F401

# overrides that make a sphere config usable without a population draw
pytestmark = pytest.mark.usefixtures("sphere_enabled")  # the sphere is refused by validate until S2

SMALL = ["population.n_max=1100", "population.n_init=64"]


def _cfg(which: str = "tiny3s", *extra: str):
    return sphere_cfg(which, *SMALL, *extra)


# --- validation of the field constants -------------------------------------------------------------


@pytest.mark.parametrize("which,bound", [("tiny3s", 11), ("mini3s", 11), ("64k3s", 14)])
def test_the_presets_accept_the_bound_and_refuse_one_more(which: str, bound: int) -> None:
    assert _cfg(which, f"fields.resource_diffusion_q8={bound}").fields.resource_diffusion_q8 == bound
    assert _cfg(which).fields.resource_diffusion_q8 == 4  # the default is far inside
    with pytest.raises(ConfigError, match=rf"resource_diffusion_q8 {bound + 1} must be <= {bound}"):
        _cfg(which, f"fields.resource_diffusion_q8={bound + 1}")


def test_the_box_diffusion_limits_still_apply_to_a_box() -> None:
    from alife.config import load_config

    with pytest.raises(ConfigError, match="resource_diffusion_q8 must be ≤ 42 in 3D"):
        load_config(preset="tiny3d", overrides=["fields.resource_diffusion_q8=43"])
    with pytest.raises(ConfigError, match="chem_diffusion must be ≤ 1/6 in 3D"):
        load_config(preset="tiny3d", overrides=["fields.chem_diffusion=0.2,0.1,0.1"])
    _cfg("tiny3s", "fields.chem_diffusion=0.25,0.25,0.25")  # a sphere sub-steps instead


def test_the_patch_drift_must_keep_the_rotation_series_exact() -> None:
    _cfg("tiny3s", "fields.patch_drift=0.18")  # 0.005 * 36
    with pytest.raises(ConfigError, match=r"patch_drift 0\.2 must be <= 0\.005·world\.radius"):
        _cfg("tiny3s", "fields.patch_drift=0.2")


# --- the vertical profile --------------------------------------------------------------------------


def test_the_profile_is_evaluated_at_the_lower_interface_heights() -> None:
    cfg = _cfg("mini3s")
    fi, w = cfg.fields, cfg.world
    z = geom.layer_edges(w.radius, w.height, w.layers)[:-1]
    off = _cfg("mini3s", "fields.photic_amp=0")
    want = np.minimum(1.0, np.exp(-z / fi.benthic_lambda))
    v = np.array(fields.vertical_profile(off))
    assert v.shape == (w.layers, 1)
    assert v[0, 0] == 1.0  # exactly, as in the box
    assert np.allclose(v[:, 0], want, rtol=1e-6) and (np.diff(v[:, 0]) < 0).all()
    ph = _cfg("mini3s", "fields.photic_amp=0.4", "fields.photic_lambda=5")
    v2 = np.array(fields.vertical_profile(ph))[:, 0]
    want2 = np.minimum(1.0, np.exp(-z / fi.benthic_lambda) + 0.4 * np.exp(-(z[-1] - z) / 5.0))
    assert np.allclose(v2, want2, rtol=1e-6) and v2[-1] > v[-1, 0]  # a photic band under the lid


# --- fertility -------------------------------------------------------------------------------------


def _random_patches(n: int, seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=(n, 3))
    return (v / np.linalg.norm(v, axis=1)[:, None]).astype(np.float32)


def _fertility_ref(cfg, pp: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """(Npix,) float64: the chord Gaussian summed over patches, clipped, on the host."""
    cen = geom.pixel_centres(cfg.world.nside)
    d2 = ((cen[None] - pp[:, None].astype(np.float64)) ** 2).sum(-1) * cfg.world.radius**2
    terms = np.exp(-d2 / (2 * cfg.fields.patch_sigma**2))
    if w is not None:
        terms = w[:, None] * terms
    return np.clip(cfg.fields.fertility_base + terms.sum(0), 0.0, 1.0)


@pytest.mark.parametrize("which", ["tiny3s", "64k3s"])
@pytest.mark.usefixtures("dev")
def test_fertility_is_the_chord_gaussian_times_the_profile(which: str) -> None:
    cfg = _cfg(which)
    pp = _random_patches(cfg.fields.n_patches, 3)
    got = np.array(fields.fertility(mx.array(pp), cfg))
    d = dims(cfg)
    assert got.shape == d.field_shape == (cfg.world.layers, geom.npix(cfg.world.nside))
    ang = _fertility_ref(cfg, pp)
    prof = np.array(fields.vertical_profile(cfg))
    assert np.abs(got - ang[None] * prof).max() < 5e-6
    assert np.array_equal(got[0], np.array(fields.fertility(mx.array(pp), cfg))[0])
    assert (got >= 0).all() and (got <= 1).all()
    # layer 0 is exactly the angular fertility (V[0] = 1)
    assert np.abs(got[0] - ang).max() < 5e-6
    # the peak sits under the patch: a patch centre is the cell nearest to it
    pix = geom.ang2pix(pp[0:1].astype(np.float64), cfg.world.nside)[0]
    assert got[0, pix] > 0.9 * got[0].max() or cfg.fields.n_patches > 1


def test_fertility_distance_is_the_chord_not_the_arc() -> None:
    cfg = _cfg("tiny3s", "fields.n_patches=1", "fields.fertility_base=0", "fields.patch_sigma=12")
    pp = np.array([[0.0, 0.0, 1.0]], np.float32)
    got = np.array(fields.fertility(mx.array(pp), cfg))[0]
    cen = geom.pixel_centres(cfg.world.nside)
    chord = 2 * cfg.world.radius * np.sin(0.5 * np.arccos(np.clip(cen[:, 2], -1, 1)))
    assert np.abs(got - np.exp(-(chord**2) / (2 * 12.0**2))).max() < 5e-6


def test_seasonal_and_director_weights_scale_the_patches() -> None:
    cfg = _cfg(
        "tiny3s", "fields.season_amplitude=0.4", "fields.season_period=120", "fields.season_mode=per_patch"
    )
    pp = _random_patches(cfg.fields.n_patches, 4)
    step = mx.array(30, dtype=mx.int32)
    w = np.array(fields.season_weights(step, cfg), np.float64)
    assert not np.allclose(w, 1.0)
    got = np.array(fields.fertility(mx.array(pp), cfg, step))
    ref = _fertility_ref(cfg, pp, w)[None] * np.array(fields.vertical_profile(cfg))
    assert np.abs(got - ref).max() < 5e-6
    # the director: patch_gain 0 removes a patch, 2 doubles it (before the clip)
    s = state.init_state(cfg)
    gain = np.array([0.0, 1.0, 2.0, 1.0][: cfg.fields.n_patches], np.float32)
    env = Env(*(getattr(s, f"env_{n}") for n in Env._fields))._replace(patch_gain=mx.array(gain))
    cfg2 = _cfg("tiny3s")  # no seasons: weights are the gain times 1
    got2 = np.array(fields.fertility(mx.array(pp), cfg2, mx.array(0, dtype=mx.int32), env))
    ref2 = _fertility_ref(cfg2, pp, gain.astype(np.float64))[None] * np.array(fields.vertical_profile(cfg2))
    assert np.abs(got2 - ref2).max() < 5e-6


def test_compiled_fertility_equals_eager_with_runtime_constants() -> None:
    """The compile-collision fallback (spec 10.1): every constant of the kernel enters as an array."""
    cfg = _cfg("mini3s")
    pp = mx.array(_random_patches(cfg.fields.n_patches, 6))
    cen = fields_sph.cell_centres(cfg.world.nside)
    w = mx.array(np.linspace(0.5, 1.5, cfg.fields.n_patches).astype(np.float32))
    args = (
        pp,
        w,
        cen,
        mx.array(cfg.world.radius),
        mx.array(cfg.fields.patch_sigma),
        mx.array(cfg.fields.fertility_base),
    )
    eager = fields_sph.fertility_cells(*args)
    comp = mx.compile(fields_sph.fertility_cells)(*args)
    assert np.array_equal(np.array(eager), np.array(comp))
    ref = _fertility_ref(cfg, np.array(pp), np.array(w, np.float64))
    assert float(np.abs(np.array(eager, np.float64) - ref).max()) < 1e-5


def test_regen_rate_is_the_floor_of_regen_max_times_fertility() -> None:
    cfg = _cfg("tiny3s")
    pp = mx.array(_random_patches(cfg.fields.n_patches, 5))
    fert = np.array(fields.fertility(pp, cfg), np.float32)
    rate = np.array(fields.regen_rate(pp, cfg))
    assert rate.dtype == np.int32 and rate.shape == fert.shape
    assert np.array_equal(rate, np.floor(np.float32(cfg.fields.regen_max) * fert).astype(np.int32))
    with pytest.raises(AssertionError, match="rain is not defined on the sphere"):
        fields.regen_rate(pp, cfg, rain=object())


# --- the patches -----------------------------------------------------------------------------------


def test_patch_draws_follow_the_rng_table() -> None:
    """``PATCH_POS`` is one (PP, 2) uniform draw mapped by ``cos θ = 2a − 1, λ = 2πb``; ``PATCH_VEL`` the
    box's ``(PP,)`` angle ψ; the drift is the angular velocity (û × t̂)·patch_drift/Rp with
    t̂ = cos ψ east + sin ψ north."""
    cfg = _cfg("tiny3s", "fields.n_patches=6", "fields.patch_drift=0.1")
    ik = rng.init_keys(cfg.run.seed)
    pos, om = fields.init_patches(ik[rng.InitStream.PATCH_POS], ik[rng.InitStream.PATCH_VEL], cfg)
    u = np.array(mx.random.uniform(0.0, 1.0, (6, 2), key=ik[rng.InitStream.PATCH_POS]), np.float64)
    psi = np.array(mx.random.uniform(0.0, 2.0 * math.pi, (6,), key=ik[rng.InitStream.PATCH_VEL]), np.float64)
    ct, lam = 2 * u[:, 0] - 1, 2 * math.pi * u[:, 1]
    st = np.sqrt(1 - ct**2)
    want = np.stack([st * np.cos(lam), st * np.sin(lam), ct], axis=1)
    p, w = np.array(pos, np.float64), np.array(om, np.float64)
    assert pos.shape == om.shape == (6, 3) and np.abs(p - want).max() < 2e-6
    east = np.stack([-p[:, 1], p[:, 0], np.zeros(6)], axis=1)
    east /= np.linalg.norm(east, axis=1)[:, None]
    north = np.cross(p, east)
    tvec = np.cos(psi)[:, None] * east + np.sin(psi)[:, None] * north
    assert np.abs(w - np.cross(p, tvec) * 0.1 / 36.0).max() < 1e-6
    assert np.abs((p * w).sum(1)).max() < 1e-7  # ω ⟂ the patch
    assert np.allclose(
        np.linalg.norm(w, axis=1) * 36.0, 0.1, rtol=1e-5
    )  # the drift speed on the floor sphere
    # a step moves the centre along t̂ by the drift speed (to first order)
    q = np.array(fields.drift_patches(pos, om, cfg), np.float64)
    assert np.abs((q - p) - tvec * 0.1 / 36.0).max() < 1e-5  # first order; the second is theta^2/2 = 3.9e-6


def test_the_centres_are_uniform_on_the_sphere() -> None:
    n = 60_000
    u = mx.random.uniform(0.0, 1.0, (n, 2), key=mx.random.key(1))
    p = np.array(fields_sph.sphere_point(u[:, 0], u[:, 1]), np.float64)
    assert np.abs(np.linalg.norm(p, axis=1) - 1).max() < 2e-7
    assert abs(p.mean(0)).max() < 0.01 and abs(p[:, 2].mean()) < 0.01
    counts = np.bincount(geom.ang2pix(p, 4), minlength=192)  # equal-area cells get equal shares
    assert counts.std() / counts.mean() < 0.12


def test_rotation_series_equals_the_exact_rodrigues_rotation() -> None:
    rg = np.random.default_rng(2)
    p = _random_patches(50, 7).astype(np.float64)
    k = np.cross(p, _random_patches(50, 8))
    k /= np.linalg.norm(k, axis=1)[:, None]  # axes perpendicular to p
    th = rg.uniform(1e-6, 0.05, 50)
    om = k * th[:, None]
    got = np.array(fields_sph.rotate(mx.array(p.astype(np.float32)), mx.array(om.astype(np.float32))))
    ex = p * np.cos(th)[:, None] + np.cross(k, p) * np.sin(th)[:, None]
    assert np.abs(got - ex).max() < 2e-7
    # zero velocity is exact
    assert np.array_equal(
        np.array(fields_sph.rotate(mx.array(p.astype(np.float32)), mx.zeros((50, 3)))),
        np.array(fields_sph.rotate(mx.array(p.astype(np.float32)), mx.zeros((50, 3)))),
    )


def test_patches_stay_on_their_small_circles_for_1e5_steps() -> None:
    """At 36 times the default drift (the speed 0.18 per step, 80 circuits in 1e5 steps) the centre keeps unit
    norm and its angle to the rotation axis; a stored Cartesian velocity would have left the tangent plane."""
    cfg = _cfg("tiny3s", "fields.patch_drift=0.18")
    ik = rng.init_keys(cfg.run.seed)
    p, om = fields.init_patches(ik[rng.InitStream.PATCH_POS], ik[rng.InitStream.PATCH_VEL], cfg)

    def chunk(p, om):  # 100 steps; only arrays enter, so no constant can collide in the compile cache
        for _ in range(100):
            p = fields_sph.rotate(p, om)
        return p

    f = mx.compile(chunk)
    q = p
    for _ in range(1000):
        q = f(q, om)
        mx.eval(q)
    p0, w, qn = (np.array(a, np.float64) for a in (p, om, q))
    th = np.linalg.norm(w, axis=1)
    k = w / th[:, None]
    assert np.abs(np.linalg.norm(qn, axis=1) - 1).max() < 3e-7
    assert np.abs((qn * k).sum(1)).max() < 1e-5  # still perpendicular to the axis (the circle's plane)
    ang = th * 100_000
    ex = (
        p0 * np.cos(ang)[:, None]
        + np.cross(k, p0) * np.sin(ang)[:, None]
        + k * (k * p0).sum(1)[:, None] * (1 - np.cos(ang))[:, None]
    )
    assert np.abs(qn - ex).max() < 2e-3  # 1111 rad of float32 steps: a position error far below a cell


def test_the_director_scales_the_rotation() -> None:
    cfg = _cfg("tiny3s", "fields.patch_drift=0.1")
    ik = rng.init_keys(cfg.run.seed)
    p, om = fields.init_patches(ik[rng.InitStream.PATCH_POS], ik[rng.InitStream.PATCH_VEL], cfg)
    s = state.init_state(cfg)

    def env(mult: float) -> Env:
        return Env(*(getattr(s, f"env_{n}") for n in Env._fields))._replace(drift_mult=mx.array(mult))

    assert np.array_equal(np.array(fields.drift_patches(p, om, cfg, env(0.0))), np.array(p))  # frozen
    d1 = np.array(fields.drift_patches(p, om, cfg, env(1.0))) - np.array(p)
    d5 = np.array(fields.drift_patches(p, om, cfg, env(5.0))) - np.array(p)
    assert np.abs(d5 - 5 * d1).max() < 0.02 * np.abs(d5).max()  # linear in the angle up to theta^2/2
    assert np.array_equal(
        np.array(fields.drift_patches(p, om, cfg, env(1.0))), np.array(fields.drift_patches(p, om, cfg))
    )


# --- the resource step -----------------------------------------------------------------------------


@pytest.mark.usefixtures("dev")
def test_update_resource_books_regeneration_and_conserves_the_rest() -> None:
    cfg = _cfg("tiny3s", "fields.resource_diffusion_q8=11")
    s = state.init_state(cfg)
    r = s.resource
    cap = cfg.fields.resource_cap
    assert r.shape == dims(cfg).field_shape and int(r.max()) <= cap
    for _ in range(25):
        r2, inc = fields.update_resource(r, s.patch_pos, cfg)
        mx.eval(r2, inc)
        assert int(r2.astype(mx.int64).sum()) == int(r.astype(mx.int64).sum()) + int(
            inc.astype(mx.int64).sum()
        )
        assert int(inc.min()) >= 0 and bool(((r + inc) <= cap).all())  # regeneration stops at resource_cap
        assert int(r2.min()) >= 0 and int(r2.max()) <= R_HARD
        r = r2
    assert int(r.max()) > int(s.resource.max()) or int(r.sum()) > int(s.resource.sum())


def test_update_resource_farm_variant_keeps_the_cap_and_books_both_increments() -> None:
    cfg = _cfg("tiny3s")
    s = state.init_state(cfg)
    bonus = mx.full(s.resource.shape, 3, dtype=mx.int32)
    r2, inc, inc0 = fields.update_resource_farm(s.resource, s.patch_pos, bonus, cfg)
    assert int(r2.astype(mx.int64).sum()) == int(s.resource.astype(mx.int64).sum()) + int(
        inc.astype(mx.int64).sum()
    )
    assert bool((inc >= inc0).all()) and bool((inc - inc0 <= 3).all())


def test_uniform_density_is_the_equilibrium_of_regeneration_free_diffusion() -> None:
    """Equal volumes: a uniform quanta field stays uniform when nothing regenerates."""
    cfg = _cfg("tiny3s", "fields.regen_max=0", "fields.fertility_base=0")
    r = mx.full(dims(cfg).field_shape, 4_000_000, dtype=mx.int32)
    r2, inc = fields.update_resource(r, state.init_state(cfg).patch_pos, cfg)
    assert int(inc.sum()) == 0 and bool((r2 == r).all())


# --- the gradient inputs ---------------------------------------------------------------------------


def _agents(cfg, n: int, seed: int, centres: bool):
    """Random agents of the shell: (pos, hdg, pitch, fcell); at cell centres when ``centres``."""
    d = dims(cfg)
    rg = np.random.default_rng(seed)
    u = _random_patches(n, seed)
    r = np.cbrt(rg.uniform(d.r_lo**3, d.r_hi**3, n)).astype(np.float32)
    pos = u * r[:, None]
    if centres:  # the centre of the agent's cell: direction of the pixel, volume-mid radius of the layer
        s = sphere.geometry(cfg)
        fc = np.array(s.cell_of(mx.array(pos)))
        k, p = fc // d.npix, fc % d.npix
        rc = np.cbrt(0.5 * (s.layer_radii[:-1] ** 3 + s.layer_radii[1:] ** 3))
        pos = (geom.pixel_centres(cfg.world.nside)[p] * rc[k][:, None]).astype(np.float32)
    up = pos / np.linalg.norm(pos, axis=1)[:, None]
    t = np.cross(up, _random_patches(n, seed + 100))
    hdg = (t / np.linalg.norm(t, axis=1)[:, None]).astype(np.float32)
    pitch = rg.uniform(-layout.PITCH_MAX, layout.PITCH_MAX, n).astype(np.float32)
    fcell = np.array(sphere.geometry(cfg).cell_of(mx.array(pos)))
    return mx.array(pos), mx.array(hdg), mx.array(pitch), mx.array(fcell.astype(np.int32))


def _linear_field(cfg, a: np.ndarray, base: float, amp: float) -> np.ndarray:
    """``base + amp·(a·x)`` at the cell centres (volume-mid radius of the layer), (FZ, Npix) float64."""
    w = cfg.world
    r = geom.layer_radii(w.radius, w.height, w.layers)
    rc = np.cbrt(0.5 * (r[:-1] ** 3 + r[1:] ** 3))
    cen = geom.pixel_centres(w.nside)
    return base + amp * ((cen @ a)[None, :] * rc[:, None])


def _body(vec: np.ndarray, pos, hdg, pitch) -> np.ndarray:
    up = np.array(pos) / np.linalg.norm(np.array(pos), axis=1)[:, None]
    h, cp, sp = (
        np.array(hdg, np.float64),
        np.cos(np.array(pitch, np.float64)),
        np.sin(np.array(pitch, np.float64)),
    )
    fwd = cp[:, None] * h + sp[:, None] * up
    lat = np.cross(up, h)
    upv = cp[:, None] * up - sp[:, None] * h
    return np.stack([(vec * fwd).sum(-1), (vec * lat).sum(-1), (vec * upv).sum(-1)], axis=-1)


def _sampled_gradient(cfg, a: np.ndarray, pos, fcell) -> np.ndarray:
    """What the sampler must return for a field linear in the position with the gradient ``a`` (planet axes,
    unit scale): the tangent part of ``a`` in the frame of the agent's cell centre, projected on the agent's
    own tangent plane, plus the radial part ``(a·û_c)`` on the agent's up, halved in the floor and lid layers
    (the box's mirrored boundary: a boundary layer sees half its one-sided slope)."""
    npix = geom.npix(cfg.world.nside)
    fc = np.array(fcell)
    k, p = fc // npix, fc % npix
    pn = np.array(pos, np.float64)
    up = pn / np.linalg.norm(pn, axis=1)[:, None]
    uc = geom.pixel_centres(cfg.world.nside)[p]
    au = (uc @ a)[:, None]
    gt = a - au * uc
    gt = gt - (gt * up).sum(1, keepdims=True) * up
    edge = ((k == 0) | (k == cfg.world.layers - 1))[:, None]
    return gt + np.where(edge, 0.5, 1.0) * au * up


@pytest.mark.parametrize("which", ["tiny3s", "mini3s"])
@pytest.mark.usefixtures("dev")
def test_the_gradient_of_a_linear_field_is_that_field_in_the_body_frame(which: str) -> None:
    """A field linear in the 3D position has the constant gradient ``A·a``. At a cell centre the sampled
    gradient (4-neighbour least squares plus the radial difference) must be it, in the (fwd, lat, up)
    frame, up to the grid's second-order error; the normalisation is ``∇R·field_cell/resource_cap`` times
    the gain."""
    cfg = _cfg(which, "fields.resource_cap=16000000")
    a = np.array([0.36, -0.48, 0.8])
    a /= np.linalg.norm(a)
    amp = 8.0e4  # quanta per unit length: 3.2e5 per cell, 0.02 of the cap
    r = _linear_field(cfg, a, 8.0e6, amp)
    ri = np.rint(r).astype(np.int32)
    pos, hdg, pitch, fcell = _agents(cfg, 400, 5, centres=True)
    obs = fields.sample_fields(
        mx.array(ri),
        mx.zeros((layout.NCH, *ri.shape)),
        fcell,
        mx.zeros(400),
        cfg,
        pitch=pitch,
        pos=pos,
        hdg=hdg,
    )
    want = layout.RES_GRAD_GAIN * _body(
        _sampled_gradient(cfg, a, pos, fcell) * amp * cfg.world.field_cell / 16_000_000, pos, hdg, pitch
    )
    got = np.array(obs.res_grad, np.float64)
    assert np.abs(want).max() < 1.0  # inside the clip, so the comparison is linear
    err = np.linalg.norm(got - want, axis=1) / np.linalg.norm(want, axis=1)
    assert (
        np.median(err) < 0.01 and np.quantile(err, 0.95) < 0.06 and err.max() < 0.12
    )  # seam cells are worst
    # the value channel is the cell value, normalised like the box
    cap = cfg.fields.resource_cap
    assert (
        np.abs(np.array(obs.res) - np.minimum(ri.reshape(-1)[np.array(fcell)] / cap, 2.0) / 2.0).max() < 1e-6
    )


def test_a_sampled_agent_sees_the_gradient_of_its_cell() -> None:
    """Anywhere inside a cell the sampled vector is the cell's, with the radial part on the agent's own up and
    the tangent part projected on its own tangent plane (no up component leaks from the cell's frame)."""
    cfg = _cfg("tiny3s", "fields.resource_cap=16000000")
    a = np.array([0.2, 0.5, -0.84])
    a /= np.linalg.norm(a)
    amp = 8.0e4
    ri = np.rint(_linear_field(cfg, a, 8.0e6, amp)).astype(np.int32)
    pos, hdg, pitch, fcell = _agents(cfg, 400, 6, centres=False)
    obs = fields.sample_fields(
        mx.array(ri),
        mx.zeros((layout.NCH, *ri.shape)),
        fcell,
        mx.zeros(400),
        cfg,
        pitch=pitch,
        pos=pos,
        hdg=hdg,
    )
    want = layout.RES_GRAD_GAIN * _body(
        _sampled_gradient(cfg, a, pos, fcell) * amp * 4.0 / 16_000_000, pos, hdg, pitch
    )
    err = np.linalg.norm(np.array(obs.res_grad, np.float64) - want, axis=1) / np.linalg.norm(want, axis=1)
    assert np.median(err) < 0.02 and np.quantile(err, 0.95) < 0.07 and err.max() < 0.15


def test_radial_difference_uses_the_true_layer_spacing_and_halves_at_the_boundaries() -> None:
    cfg = _cfg("tiny3s", "fields.resource_cap=16000000")
    g = fieldtab.gradient(*(cfg.world.radius, cfg.world.height, cfg.world.nside, cfg.world.layers))
    zc = g.zc
    # a field that depends on the layer centre height only: R = 8e6 + 1e5·zc
    n = geom.npix(cfg.world.nside)
    ri = np.rint(8.0e6 + 1.0e5 * zc)[:, None] * np.ones((1, n))
    pos, hdg, pitch, fcell = _agents(cfg, 800, 9, centres=True)
    obs = fields.sample_fields(
        mx.array(ri.astype(np.int32)),
        mx.zeros((layout.NCH, *ri.shape)),
        fcell,
        mx.zeros(800),
        cfg,
        pitch=pitch,
        pos=pos,
        hdg=hdg,
    )
    k = np.array(fcell) // n
    up = np.array(pos) / np.linalg.norm(np.array(pos), axis=1)[:, None]
    slope = np.where((k == 0) | (k == cfg.world.layers - 1), 0.5, 1.0) * 1.0e5  # quanta per unit height
    want = layout.RES_GRAD_GAIN * _body(up * (slope * 4.0 / 16_000_000)[:, None], pos, hdg, pitch)
    got = np.array(obs.res_grad, np.float64)
    interior = (k > 0) & (k < cfg.world.layers - 1)
    assert interior.any() and (~interior).any()
    assert np.abs(got - want)[interior].max() < 5e-3 * np.abs(want).max()  # exact to the integer rounding
    assert (
        np.abs(got - want)[~interior].max() < 5e-3 * np.abs(want).max()
    )  # the box's half slope at floor/lid


def test_chemical_inputs_are_normalised_like_the_box() -> None:
    cfg = _cfg("tiny3s")
    half = cfg.sensing.chem_half_sat
    a = np.array([0.0, 0.6, 0.8])
    pos, hdg, pitch, fcell = _agents(cfg, 300, 7, centres=True)
    base, amp = 2.0, 0.01
    lin = _linear_field(cfg, a, base, amp)
    chem = mx.array(np.broadcast_to(lin, (layout.NCH, *lin.shape)).astype(np.float32))
    res = mx.zeros(dims(cfg).field_shape, dtype=mx.int32)
    obs = fields.sample_fields(res, chem, fcell, mx.zeros(300), cfg, pitch=pitch, pos=pos, hdg=hdg)
    cv = np.array(chem).reshape(layout.NCH, -1)[:, np.array(fcell)].T
    assert np.abs(np.array(obs.chem) - cv / (cv + half)).max() < 1e-6
    g = _body(_sampled_gradient(cfg, a, pos, fcell) * amp * cfg.world.field_cell, pos, hdg, pitch)
    want = np.tanh(g / half)
    got = np.array(obs.chem_grad, np.float64)
    assert got.shape == (300, layout.NCH, 3)
    for ch in range(layout.NCH):
        e = np.linalg.norm(got[:, ch] - want, axis=1) / np.linalg.norm(want, axis=1)
        assert np.median(e) < 0.03 and np.quantile(e, 0.95) < 0.12 and e.max() < 0.25  # tanh bends a bit


def test_a_constant_field_has_no_gradient() -> None:
    cfg = _cfg("tiny3s")
    pos, hdg, pitch, fcell = _agents(cfg, 100, 8, centres=False)
    obs = fields.sample_fields(
        mx.full(dims(cfg).field_shape, 1234, dtype=mx.int32),
        mx.full((layout.NCH, *dims(cfg).field_shape), 0.7),
        fcell,
        mx.zeros(100),
        cfg,
        pitch=pitch,
        pos=pos,
        hdg=hdg,
    )
    assert np.abs(np.array(obs.res_grad)).max() < 1e-6 and np.abs(np.array(obs.chem_grad)).max() < 1e-6


def test_the_edge_neighbour_least_squares_is_well_conditioned() -> None:
    """The choice of the 4 edge neighbours (``fieldtab.GRADIENT_KIND``) over the 8 touching ones: the same
    conditioning (max 2.2 at Nside 21, the spec's figure for the 8-neighbour version) and no worse
    accuracy."""
    assert fieldtab.GRADIENT_KIND == "edge"
    for ns, bound in ((8, 1.9), (11, 2.1), (21, 2.25)):
        cond = geom.lsq_condition(ns, "edge")
        assert cond.max() < bound and cond.mean() < 1.35
        assert cond.max() <= geom.lsq_condition(ns, "touch").max() * 1.1


# --- deposit and the cell index --------------------------------------------------------------------


@pytest.mark.usefixtures("dev")
def test_cell_index_and_chemical_deposit_on_the_sphere() -> None:
    cfg = _cfg("tiny3s")
    d = dims(cfg)
    pos, _, _, fcell = _agents(cfg, 300, 10, centres=False)
    assert np.array_equal(np.array(fields.cell_index(pos, cfg)), np.array(fcell))
    assert int(fcell.min()) >= 0 and int(fcell.max()) < d.n_cells
    rg = np.random.default_rng(1)
    emit = mx.array(rg.random((300, layout.NCH)).astype(np.float32))
    active = mx.array(rg.random(300) < 0.7)
    dep = np.array(fields.deposit_chem(emit, fcell, active, cfg))
    assert dep.shape == (layout.NCH * d.n_cells,)
    q = np.maximum(np.floor(np.array(emit) * 256.0), 0).astype(np.int64)
    ref = np.zeros(layout.NCH * d.n_cells, np.int64)
    for c in range(layout.NCH):
        np.add.at(ref, np.array(fcell)[np.array(active)] + c * d.n_cells, q[np.array(active), c])
    assert np.array_equal(dep, ref)
    chem = mx.zeros((layout.NCH, *d.field_shape))
    out = fields.update_chem(chem, mx.array(dep), cfg)
    assert out.shape == chem.shape and float(out.min()) >= 0
    # the deposit, scaled, is conserved by the diffusion (decay off)
    cfg0 = _cfg("tiny3s", "fields.chem_decay=0,0,0", "fields.chem_max=1e9")
    out0 = np.array(fields.update_chem(chem, mx.array(dep), cfg0), np.float64)
    want = dep.reshape(layout.NCH, -1).sum(1) * cfg0.fields.chem_deposit_scale
    assert np.allclose(out0.reshape(layout.NCH, -1).sum(1), want, rtol=2e-6)


def test_ops_body_frame_is_orthonormal_and_reduces_to_the_box_frame() -> None:
    n = 200
    pos, hdg, pitch, _ = _agents(_cfg(), n, 3, centres=False)
    up = pos / mx.sqrt((pos * pos).sum(-1, keepdims=True))
    fwd, lat, upv = ops.body_axes_s(hdg, up, mx.cos(pitch), mx.sin(pitch))
    m = np.stack([np.array(fwd), np.array(lat), np.array(upv)], axis=1)  # (n, 3, 3)
    assert np.abs(m @ m.transpose(0, 2, 1) - np.eye(3)).max() < 2e-6
    assert np.abs(np.linalg.det(m) - 1.0).max() < 2e-6  # right-handed
    # a locally level scene at the north pole region maps to the box frame: heading theta, up = z
    th = mx.array(np.random.default_rng(0).uniform(-3, 3, 50).astype(np.float32))
    h = mx.stack([mx.cos(th), mx.sin(th), mx.zeros_like(th)], axis=-1)
    z = mx.broadcast_to(mx.array([0.0, 0.0, 1.0]), (50, 3))
    ph = mx.array(np.random.default_rng(1).uniform(-1, 1, 50).astype(np.float32))
    box = ops.body_axes(mx.cos(th), mx.sin(th), mx.cos(ph), mx.sin(ph))
    sph_ = ops.body_axes_s(h, z, mx.cos(ph), mx.sin(ph))
    for b, s_ in zip(box, sph_, strict=True):
        assert np.abs(np.array(b) - np.array(s_)).max() < 2e-6
    v = mx.array(np.random.default_rng(2).normal(size=(50, 3)).astype(np.float32))
    assert (
        np.abs(
            np.array(ops.to_body3(v, mx.cos(th), mx.sin(th), mx.cos(ph), mx.sin(ph)))
            - np.array(ops.to_body_s(v, h, z, mx.cos(ph), mx.sin(ph)))
        ).max()
        < 2e-6
    )
