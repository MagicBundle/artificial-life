"""The device lookup of the sphere world (``alife.sphere``): float32 MLX against the float64 host reference of
``alife.geom``, on the CPU and on Metal (DESIGN §46.1; sphere series S0).

``ang2pix`` is allowed to differ from the float64 value only by landing in an adjacent pixel when a point is
within float32 rounding of a pixel boundary; the tests count those ties and require them to be adjacent. All
tests are eager except one, which passes every constant as a runtime ``mx.array`` (the compile-collision
fallback of the spec, section 10.1).
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from alife import geom, sphere

DEVICES = [mx.cpu, *([mx.gpu] if mx.metal.is_available() else [])]


@pytest.fixture(params=DEVICES, ids=lambda d: str(d).split(".")[-1].split(",")[0].strip(")"))
def dev(request):
    """Run the test body on this device (a stream context), whatever the process default is."""
    with mx.stream(request.param):
        yield request.param


def _random_shell(m: int, seed: int, lo: float = 36.0, hi: float = 130.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(m, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    return (v * rng.uniform(lo, hi, size=(m, 1))).astype(np.float32)


def _adjacent(nside: int, got: np.ndarray, ref: np.ndarray) -> bool:
    touch = geom.tables(nside).touch
    return all(int(g) == int(r) or int(g) in touch[int(r)] for g, r in zip(got, ref, strict=True))


def _check(nside: int, pts32: np.ndarray, max_rate: float) -> int:
    got = np.array(sphere.ang2pix(mx.array(pts32), nside))
    ref = geom.ang2pix(pts32.astype(np.float64), nside)
    bad = got != ref
    assert got.min() >= 0 and got.max() < geom.npix(nside)  # never out of range
    assert _adjacent(nside, got[bad], ref[bad])  # a tie is an adjacent pixel, never a far one
    assert bad.mean() <= max_rate
    return int(bad.sum())


@pytest.mark.parametrize("nside", [2, 5, 8, 21, 64])
def test_float32_ang2pix_matches_float64_except_adjacent_ties(dev, nside: int) -> None:
    n_bad = _check(nside, _random_shell(1_000_000, nside), max_rate=3e-5)
    assert n_bad < 40


@pytest.mark.parametrize("nside", [2, 3, 5, 8, 11, 21, 33, 64])
def test_every_pixel_centre_round_trips_in_float32(dev, nside: int) -> None:
    c = geom.pixel_centres(nside)
    for radius in (36.5, 100.0):
        got = np.array(sphere.ang2pix(mx.array((c * radius).astype(np.float32)), nside))
        assert np.array_equal(got, np.arange(geom.npix(nside)))
    # and the device centres are the host centres
    dc = np.array(sphere.pix2vec(mx.arange(geom.npix(nside)), nside))
    assert np.abs(dc - c).max() < 3e-6
    assert np.array_equal(np.array(sphere.ang2pix(mx.array(dc), nside)), np.arange(geom.npix(nside)))


def _adversarial_points(nside: int) -> np.ndarray:
    """Poles, every pixel corner (the 8 trivalent vertices, the 4 equatorial vertices, all ring crossings), 31
    points along every pixel edge, and rings of points around the poles at 1e-6 to 1e-2 rad."""
    n = geom.npix(nside)
    ix, iy, face = geom.pix2xyf(nside, np.arange(n))
    s = np.linspace(0, 1, 32)
    pts = [np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])]
    for k in range(4):
        (dx0, dy0), (dx1, dy1) = geom.CORNERS[k], geom.CORNERS[(k + 1) % 4]
        x = (ix[:, None] + dx0 + (dx1 - dx0) * s[None]) / nside
        y = (iy[:, None] + dy0 + (dy1 - dy0) * s[None]) / nside
        pts.append(geom.xyf2vec(x, y, face[:, None]).reshape(-1, 3))
    for r in (1e-6, 1e-4, 1e-2):
        for z in (1.0, -1.0):
            t = np.linspace(0, 2 * math.pi, 64, endpoint=False)
            pts.append(np.stack([r * np.cos(t), r * np.sin(t), z * np.sqrt(1 - r * r) * np.ones_like(t)], 1))
    # the exact z = ±2/3 and z = 0 rings
    for z in (2 / 3, -2 / 3, 0.0):
        t = np.linspace(0, 2 * math.pi, 200, endpoint=False)
        sz = math.sqrt(1 - z * z)
        pts.append(np.stack([sz * np.cos(t), sz * np.sin(t), z * np.ones_like(t)], 1))
    return np.concatenate(pts)


@pytest.mark.parametrize("nside", [2, 4, 5, 8, 21])
def test_poles_vertices_and_ring_boundaries(dev, nside: int) -> None:
    pts = _adversarial_points(nside)
    for radius in (36.5, 98.7, 129.9):
        _check(nside, (pts * radius).astype(np.float32), max_rate=1.0)  # ties are adjacent; none out of range
    # the poles themselves are the first pixel and the last
    got = np.array(sphere.ang2pix(mx.array(np.array([[0, 0, 40.0], [0, 0, -40.0]], dtype=np.float32)), nside))
    assert got[0] < 4 and got[1] >= geom.npix(nside) - 4
    # a point is in the pixel it lies in, within a few ulp of the boundary: the pixel centre is at most one
    # circumscribed radius away (HEALPix cells are at most ~1.3 sqrt(A) across)
    back = np.array(sphere.ang2pix(mx.array((pts * 50).astype(np.float32)), nside))
    cen = geom.pixel_centres(nside)[back]
    ang = np.arccos(np.clip((cen * pts).sum(1), -1, 1))
    assert ang.max() < 1.4 * geom.pixel_side(nside)


def test_ang2pix_is_independent_of_the_radius_and_handles_a_dead_row(dev) -> None:
    v = _random_shell(5000, 3, 1.0, 1.0)
    a = np.array(sphere.ang2pix(mx.array(v * 40), 21))
    b = np.array(sphere.ang2pix(mx.array(v * 120), 21))
    assert (a != b).mean() < 1e-3
    dead = np.array(sphere.ang2pix(mx.array(np.array([[0, 0, 37.0]], dtype=np.float32)), 21))
    assert 0 <= dead[0] < 4  # the engine's dead row (0, 0, r_lo) is a valid pole pixel


@pytest.mark.parametrize("rp,h,fz", [(36.0, 16.0, 8), (48.0, 24.0, 12), (90.0, 40.0, 18)])
def test_layer_index_matches_the_host_and_clamps(dev, rp: float, h: float, fz: int) -> None:
    rt = rp + h
    dv = (rt**3 - rp**3) / fz
    pts = _random_shell(500_000, 7, rp - 2.0, rt + 2.0)
    got = np.array(sphere.layer_index(mx.array(pts), rp**3, dv, fz))
    ref = geom.layer_of(np.linalg.norm(pts.astype(np.float64), axis=1), rp, h, fz)
    assert got.min() >= 0 and got.max() <= fz - 1
    assert np.abs(got - ref).max() <= 1 and (got != ref).mean() < 1e-5  # a flip is one layer at an interface
    r = geom.layer_radii(rp, h, fz)
    for k in range(1, fz):  # a body 1e-4 relative beyond an interface is in the layer above, below it below
        up = np.array([[r[k] * (1 + 1e-4), 0, 0]], dtype=np.float32)
        dn = np.array([[0, r[k] * (1 - 1e-4), 0]], dtype=np.float32)
        assert np.array(sphere.layer_index(mx.array(up), rp**3, dv, fz))[0] == k
        assert np.array(sphere.layer_index(mx.array(dn), rp**3, dv, fz))[0] == k - 1


def test_hash_and_field_cells(dev) -> None:
    rp, h, fz, nf = 48.0, 24.0, 12, 11
    nh = geom.derive_nh(rp, 1.0, 6.0)
    gz, th = geom.hash_layers(h, 6.0)
    pts = _random_shell(20_000, 5, rp + 1.0, rp + h - 1.0)
    alive = mx.array(np.arange(len(pts)) % 5 != 0)
    cid = np.array(sphere.hash_cell(mx.array(pts), alive, rp, th, gz, nh))
    sentinel = geom.npix(nh) * gz
    assert (cid[~np.array(alive)] == sentinel).all() and cid[np.array(alive)].max() < sentinel
    z = np.linalg.norm(pts.astype(np.float64), axis=1) - rp
    ref = np.clip(np.floor(z / th), 0, gz - 1).astype(np.int64) * geom.npix(nh) + geom.ang2pix(
        pts.astype(np.float64), nh
    )
    live = np.array(alive)
    assert (cid[live] != ref[live]).mean() < 1e-3  # equal up to float32 ties at layer or pixel boundaries
    rt = rp + h
    fc = np.array(sphere.field_cell(mx.array(pts), rp**3, (rt**3 - rp**3) / fz, fz, nf))
    layer = fc // geom.npix(nf)
    assert np.array_equal(fc % geom.npix(nf), np.array(sphere.ang2pix(mx.array(pts), nf)))
    assert layer.min() >= 0 and layer.max() < fz
    # the floor cell of a column is the cell index modulo Npix (corpses sink there)
    assert (fc % geom.npix(nf) < geom.npix(nf)).all()


def test_compiled_lookup_with_runtime_constants_equals_eager(dev) -> None:
    """The compile-collision fallback: Nside, Rp³, ΔV and FZ are passed as ``mx.array`` inputs, so no scalar
    constant is printed into the compiled graph (DESIGN F21 item 6). Three different configurations run in a
    permuted order in one process; each equals the eager Python-constant result bit for bit."""

    @mx.compile
    def lookup(pos, nside, rp3, dv, layers):
        return sphere.field_cell(pos, rp3, dv, layers, nside)

    cases = [(8, 36.0, 16.0, 8), (21, 90.0, 40.0, 18), (11, 48.0, 24.0, 12)]
    for nside, rp, h, fz in [cases[1], cases[0], cases[2], cases[1], cases[0]]:
        pts = mx.array(_random_shell(30_000, nside, rp, rp + h))
        dv = (rp + h) ** 3 - rp**3
        dv /= fz
        got = lookup(
            pts,
            mx.array(nside, mx.int32),
            mx.array(rp**3, mx.float32),
            mx.array(dv, mx.float32),
            mx.array(fz, mx.int32),
        )
        want = sphere.field_cell(pts, rp**3, dv, fz, nside)
        assert np.array_equal(np.array(got), np.array(want))
