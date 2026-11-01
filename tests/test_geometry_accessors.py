"""The geometry accessors (DESIGN §46.12; sphere series S0): ``geom.geometry(cfg)`` on the host and
``sphere.geometry(cfg)`` on the device. The ``Box`` methods are today's expressions written once, so
converting a consumer to them is bit-identical; the ``Sphere`` methods agree with the float64 host. Also: the
claim that ``min_image`` is the identity on the pair offsets of a sphere world (by value)."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from alife import fields, geom, ops, sphere
from alife.config import load_config
from sphere_helpers import sphere_cfg, sphere_enabled  # noqa: F401  (a fixture)

DEVICES = [mx.cpu, *([mx.gpu] if mx.metal.is_available() else [])]


@pytest.fixture(params=DEVICES, ids=lambda d: str(d).split(".")[-1].split(",")[0].strip(")"))
def dev(request):
    with mx.stream(request.param):
        yield request.param


def _box_points(cfg, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = cfg.world
    top = w.height if w.dims == 3 else 0.0
    lo = [0.0, 0.0, 0.0][: w.dims]
    hi = [w.size, w.size, top][: w.dims]
    return rng.uniform(lo, hi, size=(n, w.dims)).astype(np.float32)


@pytest.mark.parametrize("preset", ["tiny", "tiny3d"])
def test_the_device_box_is_todays_expressions_bit_for_bit(dev, preset: str) -> None:
    cfg = load_config(preset=preset)
    box = sphere.geometry(cfg)
    assert box.name == "box"
    a, b = (mx.array(_box_points(cfg, 4000, s)) for s in (1, 2))
    assert np.array_equal(np.array(box.cell_of(a)), np.array(fields.cell_index(a, cfg)))
    assert np.array_equal(np.array(box.displacement(a, b)), np.array(ops.min_image(b - a, cfg.world.size)))
    d = ops.min_image(b - a, cfg.world.size)
    assert np.array_equal(np.array(box.distance(a, b)), np.array(mx.sqrt((d * d).sum(axis=-1))))
    assert np.array_equal(np.array(box.arc(a, b)), np.array(box.distance(a, b)))
    if cfg.world.dims == 3:
        v = mx.array(_box_points(cfg, 4000, 3)) - 5.0
        assert np.array_equal(np.array(box.height(a)), np.array(a[:, 2]))
        assert np.array_equal(np.array(box.radial(v, a)), np.array(v[:, 2]))
        assert np.array_equal(np.array(box.up(a))[:, 2], np.ones(4000, np.float32))
        t = np.array(box.tangent(v, a))
        assert np.array_equal(t[:, :2], np.array(v[:, :2])) and not t[:, 2].any()
    assert np.array_equal(np.array(box.bearing(a[:, :2], a)), np.array(mx.arctan2(a[:, 1], a[:, 0])))


@pytest.mark.parametrize("preset", ["tiny", "tiny3d"])
def test_the_host_box_agrees_with_the_device_box(preset: str) -> None:
    cfg = load_config(preset=preset)
    hb, db = geom.geometry(cfg), sphere.geometry(cfg)
    a, b = (_box_points(cfg, 3000, s) for s in (4, 5))
    assert np.array_equal(hb.cell_of(a), np.array(db.cell_of(mx.array(a))))
    assert np.abs(hb.displacement(a, b) - np.array(db.displacement(mx.array(a), mx.array(b)))).max() < 1e-4
    # the torus: the displacement is the short way round, the centroid the circular mean
    nd, size = cfg.world.dims, cfg.world.size
    p0, p1 = np.array([1.0, 1.0, 2.0][:nd]), np.array([size - 1.0, 1.0, 4.0][:nd])
    assert hb.displacement(p0, p1)[0] == pytest.approx(-2.0)
    c = hb.centroid(np.stack([p0, p1]))
    assert min(c[0], size - c[0]) == pytest.approx(0.0, abs=1e-9) and c[1] == pytest.approx(1.0)


@pytest.mark.usefixtures("sphere_enabled")
@pytest.mark.parametrize("which", ["tiny3s", "mini3s"])
def test_the_device_sphere_agrees_with_the_host_sphere(dev, which: str) -> None:
    cfg = sphere_cfg(which)
    hs, ds = geom.geometry(cfg), sphere.geometry(cfg)
    assert hs.name == ds.name == "sphere"
    rp, h = cfg.world.radius, cfg.world.height
    rng = np.random.default_rng(8)
    n = 5000

    def shell(seed):
        r = np.random.default_rng(seed)
        u = r.normal(size=(n, 3))
        u /= np.linalg.norm(u, axis=1)[:, None]
        return (u * r.uniform(rp + 1, rp + h - 1, (n, 1))).astype(np.float32)

    a, b = shell(1), shell(2)
    ma, mb = mx.array(a), mx.array(b)
    v = rng.normal(size=(n, 3)).astype(np.float32)
    mv = mx.array(v)
    tol = dict(atol=2e-5)
    assert np.allclose(hs.height(a), np.array(ds.height(ma)), **tol)
    assert np.allclose(hs.up(a), np.array(ds.up(ma)), **tol)
    assert np.allclose(hs.radial(v, a), np.array(ds.radial(mv, ma)), **tol)
    assert np.allclose(hs.tangent(v, a), np.array(ds.tangent(mv, ma)), **tol)
    assert np.array_equal(hs.displacement(a, b), np.array(ds.displacement(ma, mb)))  # b − a, no min_image
    assert np.allclose(hs.distance(a, b), np.array(ds.distance(ma, mb)), atol=1e-4)
    assert np.allclose(hs.arc(a, b), np.array(ds.arc(ma, mb)), atol=1e-4)
    # the geometry of the accessors: radial + tangent = the vector, tangent ⟂ up
    t, r_ = hs.tangent(v, a), hs.radial(v, a)
    assert (
        np.allclose(t + r_[:, None] * hs.up(a), v, atol=1e-6) and np.abs((t * hs.up(a)).sum(1)).max() < 1e-6
    )
    b_same = b * (np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1))[:, None]  # at the radius of a
    assert (hs.arc(a, b_same) >= hs.distance(a, b_same) - 1e-9).all()  # the arc is longer than its chord
    # cells: the same except float32 ties
    hc, dc = hs.cell_of(a), np.array(ds.cell_of(ma))
    assert (hc != dc).mean() < 1e-3 and dc.max() < cfg.world.layers * 12 * cfg.world.nside**2
    ang = np.array(ds.bearing(mv, ma))
    assert (
        np.allclose(hs.bearing(v, a), ang, atol=2e-4)
        or (np.abs(hs.bearing(v, a) - ang) % (2 * math.pi) < 2e-4).all()
    )
    # the accessors have no preferred axis: height and arc are invariant under a rotation of the scene
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    ra, rb = (a @ q).astype(np.float32), (b @ q).astype(np.float32)
    assert np.allclose(hs.height(ra), hs.height(a), atol=1e-4) and np.allclose(
        hs.arc(ra, rb), hs.arc(a, b), atol=2e-4
    )
    assert np.allclose(np.array(ds.height(mx.array(ra))), np.array(ds.height(ma)), atol=1e-4)


def test_the_host_sphere_centroid() -> None:
    hs = geom.Sphere(radius=40.0, height=10.0, nside=8, layers=6)
    d = np.array([[0.6, 0.0, 0.8]])
    assert np.allclose(
        hs.centroid(np.concatenate([d * 42, d * 46])), d[0] * 44
    )  # same direction: mean radius
    pts = np.array([[42.0, 0, 0], [0, 42.0, 0]])
    c = hs.centroid(pts)
    assert np.allclose(c / np.linalg.norm(c), [math.sqrt(0.5), math.sqrt(0.5), 0]) and np.linalg.norm(
        c
    ) == pytest.approx(42)
    anti = hs.centroid(np.array([[42.0, 0, 0], [-42.0, 0, 0]]))  # cancelling sum: the first point's direction
    assert np.allclose(anti, [42.0, 0, 0])
    w = hs.centroid(pts, weights=np.array([3.0, 1.0]))
    assert w[0] > w[1] > 0


def test_the_host_sphere_cell_of_is_layer_major_like_the_box() -> None:
    hs = geom.Sphere(radius=36.0, height=16.0, nside=8, layers=8)
    u = geom.pixel_centres(8)[[0, 100, 767]]
    r = geom.layer_radii(36.0, 16.0, 8)
    for k in (0, 3, 7):
        pos = u * ((r[k] + r[k + 1]) / 2)
        assert np.array_equal(hs.cell_of(pos), k * 768 + np.array([0, 100, 767]))


@pytest.mark.parametrize("which", ["tiny3s", "64k3s"])
@pytest.mark.usefixtures("sphere_enabled")
def test_min_image_is_the_identity_on_the_pair_offsets_of_a_sphere_world(dev, which: str) -> None:
    """With world.size > 4(Rp + H), every offset between two bodies of the shell has |d| <= 2(Rp + H) < L/2,
    so ``d − L·round(d/L)`` returns d: bitwise except a signed zero (−0.0 becomes +0.0, review L8), so
    compared by value, over the whole range of offsets, the extremes ±2(Rp+H) and the exact zero included."""
    cfg = sphere_cfg(which)
    L, top = cfg.world.size, cfg.world.radius + cfg.world.height
    assert 4 * top < L
    rng = np.random.default_rng(9)
    u = rng.normal(size=(300_000, 3)).astype(np.float64)
    u /= np.linalg.norm(u, axis=1)[:, None]
    pos = (u * rng.uniform(cfg.world.radius, top, (300_000, 1))).astype(np.float32)
    d = pos[: len(pos) // 2] - pos[len(pos) // 2 :]
    extremes = np.array(
        [[2 * top, 0, 0], [-2 * top, 0, 0], [0, 0, 0], [-0.0, 0.0, 0.0], [0, 2 * top, -2 * top]], np.float32
    )
    d = np.concatenate([d, extremes])
    assert np.abs(d).max() <= 2 * top * math.sqrt(2) and np.abs(d).max() < L / 2
    out = np.array(ops.min_image(mx.array(d), L))
    assert (out == d).all()  # by value
    # and the accessor says so
    sph = sphere.geometry(cfg)
    a, b = mx.array(pos[:1000]), mx.array(pos[1000:2000])
    assert (np.array(sph.displacement(a, b)) == pos[1000:2000] - pos[:1000]).all()
