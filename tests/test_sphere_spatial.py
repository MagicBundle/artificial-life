"""The neighbour search of the sphere world (DESIGN §46.3; sphere series S2).

``spatial._find_neighbors_sph``: the hash cell is (radial layer, HEALPix pixel of Nh) and a row scans the 5
x 25
stencil of the exact reach-2 table. It must equal brute-force kNN whenever no hash cell holds more than C
agents (F13), at the poles, at the trivalent vertices, along the ring seams and at the hash layer edges,
with the
box search's selection and tie rules, bitwise antisymmetric chord offsets, and the same counters; the fused
Metal kernel returns the MLX path's arrays bit for bit. Eager throughout (nothing here compiles a constant
that
could collide, DESIGN F21 item 6).
"""

from __future__ import annotations

import math
import warnings

import mlx.core as mx
import numpy as np
import pytest

from alife import geom, knn_metal, spatial
from alife.config import Config, dims, load_config
from sphere_fields_helpers import dev  # noqa: F401  (a fixture)
from sphere_helpers import sphere_enabled  # noqa: F401  (a fixture)

K = 8
GPU = mx.metal.is_available()
VERTEX = np.array([math.sqrt(5.0) / 3.0, 0.0, 2.0 / 3.0])  # a trivalent vertex of the HEALPix grid (z = 2/3)

WORLDS = {
    "tiny3s": ["world.radius=36", "world.height=16", "world.nside=8", "world.layers=8"],
    "mini3s": ["world.radius=48", "world.height=24", "world.nside=11", "world.layers=12"],
}


def cfg_for(which: str, n: int, *extra: str) -> Config:
    return load_config(
        preset="tiny3s",
        overrides=[
            *WORLDS[which],
            f"population.n_max={n}",
            "population.n_init=8",
            "population.max_births_per_step=32",
            f"sensing.k_neighbors={K}",
            "brain.weight_dtype=float32",
            *extra,
        ],
    )


def shell_points(
    rng: np.random.Generator, n: int, rp: float, h: float, centre=None, spread=0.1
) -> np.ndarray:
    """``n`` float32 points in the shell (a margin of 1 from both bounds); uniform over the sphere, or
    normally
    scattered around the direction ``centre`` (angular spread ``spread`` rad)."""
    if centre is None:
        z = rng.uniform(-1, 1, n)
        lam = rng.uniform(0, 2 * math.pi, n)
        st = np.sqrt(1 - z * z)
        u = np.stack([st * np.cos(lam), st * np.sin(lam), z], axis=1)
    else:
        u = np.asarray(centre, np.float64) + rng.normal(0, spread, (n, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
    rho = np.cbrt(rng.uniform((rp + 1) ** 3, (rp + h - 1) ** 3, n))
    return (u * rho[:, None]).astype(np.float32)


def brute_force(
    pos: np.ndarray, alive: np.ndarray, radius: float, k: int = K
) -> tuple[np.ndarray, np.ndarray]:
    """``(idx (N,K) int, valid (N,K))`` by brute force in float32 with the engine's expression for d²: the K
    nearest alive agents strictly inside ``radius``, nearest first, ties by the lower row (no ties occur in
    the
    random scenes)."""
    n = len(pos)
    idx = np.tile(np.arange(n)[:, None], (1, k))
    valid = np.zeros((n, k), bool)
    rows = np.flatnonzero(alive)
    r2 = np.float32(radius) * np.float32(radius)
    for i in rows:
        d = pos[rows] - pos[i]
        d2 = (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) + d[:, 2] * d[:, 2]
        m = (d2 < r2) & (rows != i)
        cand, dd = rows[m], d2[m]
        o = np.lexsort((cand, dd))[:k]
        idx[i, : len(o)] = cand[o]
        valid[i, : len(o)] = True
    return idx, valid


def scene(cfg: Config, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform points plus clusters at both poles, at the 8 trivalent vertices of the grid, on the equator
    seams and on the layer edges of the hash grid: positions where a table-driven stencil could be wrong."""
    d = dims(cfg)
    rng = np.random.default_rng(seed)
    rp, h = cfg.world.radius, cfg.world.height
    n = d.n
    parts = [shell_points(rng, n // 4, rp, h)]
    centres = [(0, 0, 1), (0, 0, -1), (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)]
    centres += [
        (sx * VERTEX[0], sy * VERTEX[0], sz * VERTEX[2])
        for sx, sy, sz in [
            (1, 0, 1),
            (-1, 0, 1),
            (0, 1, 1),
            (0, -1, 1),
            (1, 0, -1),
            (-1, 0, -1),
            (0, 1, -1),
            (0, -1, -1),
        ]
    ]
    per = max((n - len(parts[0])) // len(centres), 1)
    for c in centres:
        parts.append(shell_points(rng, per, rp, h, centre=c, spread=0.07))
    pts = np.concatenate(parts)[:n]
    if len(pts) < n:
        pts = np.concatenate([pts, shell_points(rng, n - len(pts), rp, h)])
    # snap a tenth of the points onto the hash layer interfaces (z = k·T_h, the radial seams of the stencil)
    sel = rng.choice(n, n // 10, replace=False)
    th = h / (math.floor(2 * h / cfg.sensing.radius))
    z = rng.integers(1, math.floor(2 * h / cfg.sensing.radius), len(sel)) * th
    u = pts[sel] / np.linalg.norm(pts[sel], axis=1, keepdims=True)
    pts[sel] = (u * (rp + z)[:, None]).astype(np.float32)
    alive = np.ones(n, bool)
    alive[rng.choice(n, n // 12, replace=False)] = False
    return pts.astype(np.float32), alive


@pytest.mark.usefixtures("dev")
@pytest.mark.parametrize("which", ["tiny3s", "mini3s"])
def test_search_equals_brute_force_at_poles_vertices_seams_and_layer_edges(which: str) -> None:
    cfg = cfg_for(which, 1024 if which == "tiny3s" else 2048)
    pos, alive = scene(cfg, seed=3)
    nb = spatial.find_neighbors(mx.array(pos), mx.array(alive), cfg, impl="mlx")
    mx.eval(nb)
    assert int(nb.overflow_cells) == 0 and int(nb.hidden_agents) == 0  # the premise of F13
    idx, valid = brute_force(pos, alive, cfg.sensing.radius)
    assert np.array_equal(np.array(nb.valid), valid)
    assert np.array_equal(np.where(valid, np.array(nb.idx), -1), np.where(valid, idx, -1))
    assert valid.sum() > 4 * len(
        pos
    )  # the scene is dense enough to be a test: >4 neighbours per row on average
    off = np.array(nb.off)
    d = pos[idx] - pos[:, None, :]
    assert np.array_equal(off[valid], d[valid])  # the offsets are the plain float32 differences
    assert np.array_equal(np.array(nb.dist)[~valid], np.full((~valid).sum(), np.float32(cfg.sensing.radius)))


@pytest.mark.usefixtures("dev")
def test_offsets_are_bitwise_antisymmetric_and_dead_rows_are_invisible() -> None:
    cfg = cfg_for("tiny3s", 1024)
    pos, alive = scene(cfg, seed=5)
    nb = spatial.find_neighbors(mx.array(pos), mx.array(alive), cfg, impl="mlx")
    idx, valid, off, dist = (np.array(a) for a in (nb.idx, nb.valid, nb.off, nb.dist))
    assert not valid[~alive].any() and alive[idx[valid]].all()  # dead rows neither see nor are seen
    pairs = 0
    for i, j in zip(*np.nonzero(valid), strict=True):
        other = idx[i, j]
        hit = np.flatnonzero(valid[other] & (idx[other] == i))
        if hit.size:  # mutual: the two chords are exact negatives of one another, the distances equal
            assert np.array_equal(off[other, hit[0]], -off[i, j]) and dist[other, hit[0]] == dist[i, j]
            pairs += 1
    assert pairs > 1000
    assert (dist[valid] < np.float32(cfg.sensing.radius)).all()  # strictly inside R


@pytest.mark.usefixtures("dev")
def test_min_image_is_the_identity_on_every_offset_of_the_scene() -> None:
    """``world.size > 4(Rp + H)`` (validate) makes ``ops.min_image`` bitwise the identity on pair offsets (by
    value: -0.0 and +0.0 compare equal), the premise of leaving every pair expression unchanged (§46.3)."""
    from alife import ops

    cfg = cfg_for("tiny3s", 1024)
    pos, alive = scene(cfg, seed=6)
    nb = spatial.find_neighbors(mx.array(pos), mx.array(alive), cfg, impl="mlx")
    off = np.array(nb.off)
    assert np.array_equal(np.array(ops.min_image(nb.off, cfg.world.size)), off)


@pytest.mark.usefixtures("dev")
def test_overflow_is_reported_and_the_visible_part_is_still_exact() -> None:
    """Forty agents on one spot with C = 32: the cell is over capacity (reported), only the first C of the run
    are visible, every reported neighbour is a true one, and the counters are the independent occupancy."""
    cfg = cfg_for("tiny3s", 1024)
    d = dims(cfg)
    pos, alive = scene(cfg, seed=7)
    pos[:40] = (VERTEX * 44.0).astype(np.float32)  # 40 identical positions at a trivalent vertex, mid shell
    alive[:40] = True
    nb = spatial.find_neighbors(mx.array(pos), mx.array(alive), cfg, impl="mlx")
    occ = np.array(spatial.hash_occupancy(mx.array(pos), mx.array(alive), cfg))
    assert occ.shape == (dims(cfg).g_hash_z, d.npix_h)
    assert int(nb.max_occupancy) == int(occ.max()) >= 40
    assert int(nb.overflow_cells) == int((occ > d.c).sum()) >= 1
    assert int(nb.hidden_agents) == int(np.maximum(occ - d.c, 0).sum()) >= 8
    idx, valid = np.array(nb.idx), np.array(nb.valid)
    r2 = np.float32(cfg.sensing.radius) ** 2
    dd = pos[idx] - pos[:, None, :]
    d2 = (dd[..., 0] * dd[..., 0] + dd[..., 1] * dd[..., 1]) + dd[..., 2] * dd[..., 2]
    assert (d2[valid] < r2).all() and alive[idx[valid]].all()


def test_the_hash_stencil_is_the_exact_table_with_the_pixel_first() -> None:
    for nh in (8, 11):
        tab = np.array(spatial.hash_stencil(nh))
        ref = geom.tables(nh).stencil2
        assert tab.shape == (12 * nh * nh, 25) and np.array_equal(tab, ref)
        assert np.array_equal(tab[:, 0], np.arange(len(tab)))
        assert (tab >= 0).sum(axis=1).min() >= 21 and (tab >= 0).sum(axis=1).max() == 25


# --- the Metal kernel -------------------------------------------------------------------------------------


def _same(a: spatial.Neighbors, b: spatial.Neighbors) -> list[str]:
    return [
        f
        for f in spatial.Neighbors._fields
        if getattr(a, f).dtype != getattr(b, f).dtype
        or getattr(a, f).shape != getattr(b, f).shape
        or np.array(getattr(a, f)).tobytes() != np.array(getattr(b, f)).tobytes()
    ]


@pytest.mark.skipif(not (GPU and knn_metal.available()), reason="needs the Metal GPU and the verified MLX")
@pytest.mark.parametrize("which", ["tiny3s", "mini3s"])
def test_the_kernel_returns_the_mlx_path_arrays_bit_for_bit(which: str) -> None:
    """Every field of ``Neighbors`` (ids, validity, chord offsets, distances and the three counters) is equal
    byte for byte, on the clustered scene and on one with an overfull cell."""
    cfg = cfg_for(which, 1024 if which == "tiny3s" else 2048)
    with mx.stream(mx.gpu):
        for seed, overfull in ((11, False), (12, True)):
            pos, alive = scene(cfg, seed)
            if overfull:
                pos[:45] = (VERTEX * (cfg.world.radius + 8.0)).astype(np.float32)
                alive[:45] = True
            p, a = mx.array(pos), mx.array(alive)
            ref = spatial.find_neighbors(p, a, cfg, impl="mlx")
            ker = spatial.find_neighbors(p, a, cfg, impl="metal")
            mx.eval(ref, ker)
            assert _same(ref, ker) == [], (which, seed)
            assert (int(ref.overflow_cells) > 0) == overfull


@pytest.mark.skipif(not (GPU and knn_metal.available()), reason="needs the Metal GPU and the verified MLX")
def test_the_kernel_self_check_passes_once_and_a_mismatch_disables_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(knn_metal, "_CHECKED", frozenset())
    monkeypatch.setattr(knn_metal, "_MISMATCH", False)
    calls: list[int] = []
    real = knn_metal.scan_select_sph

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(knn_metal, "scan_select_sph", spy)
    with mx.stream(mx.gpu):
        assert knn_metal.self_check(sphere=True) and len(calls) == 1
        assert knn_metal.self_check(sphere=True) and len(calls) == 1  # memoised: the second call runs nothing
        assert "sph" in knn_metal._CHECKED and not knn_metal._MISMATCH

        monkeypatch.setattr(knn_metal, "_CHECKED", frozenset())

        def broken(*a, **k):
            idx, valid, off, dist = real(*a, **k)
            return idx, valid, off, dist + 1e-3  # a wrong distance

        monkeypatch.setattr(knn_metal, "scan_select_sph", broken)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert not knn_metal.self_check(sphere=True)
        assert any("differs from the MLX search" in str(x.message) for x in w)
        assert knn_metal._MISMATCH and not knn_metal.available()
    monkeypatch.setattr(knn_metal, "_MISMATCH", False)


@pytest.mark.skipif(not (GPU and knn_metal.available()), reason="needs the Metal GPU and the verified MLX")
def test_the_self_check_scene_has_ties_overflow_and_the_special_points() -> None:
    cfg, pos, alive = knn_metal._check_scene_sph()
    d = dims(cfg)
    p, a = mx.array(pos), mx.array(alive)
    nb = spatial.find_neighbors(p, a, cfg, impl="mlx")
    assert int(nb.overflow_cells) > 0 and int(nb.valid.sum()) > 5000
    dist = np.array(nb.dist)[np.array(nb.valid)]
    assert len(np.unique(dist)) < len(dist) // 3  # many equal distances: the tie rule is exercised
    assert np.isnan(pos).any() and (~alive).any()
    assert d.nside_h == 8 and d.q == 125 * d.c
    rad = np.linalg.norm(pos[alive], axis=1)
    assert (
        rad.min() > cfg.world.radius and rad.max() < cfg.world.radius + cfg.world.height
    )  # inside the shell
    assert (np.abs(pos[alive][:, 2]) > 0.9 * rad).any()  # points near the poles are in it


def test_the_kernel_validates_its_inputs() -> None:
    n = 16
    spos = mx.zeros((n, 3))
    start = mx.zeros((12 * 64 * 5 + 1,), dtype=mx.int32)
    order = mx.arange(n, dtype=mx.int32)
    pix = mx.zeros((n,), dtype=mx.int32)
    nbr = mx.zeros((12 * 64, 25), dtype=mx.int32)
    alive = mx.ones((n,), dtype=mx.bool_)
    kw = dict(npix_h=12 * 64, gz=5, c=32, k=8, n=n, R=6.0)
    with pytest.raises(ValueError, match="start must be int32"):
        knn_metal.scan_select_sph(spos, start[:-1], order, pix, pix, nbr, alive, **kw)
    with pytest.raises(ValueError, match="supports K"):
        knn_metal.scan_select_sph(spos, start, order, pix, pix, nbr, alive, **{**kw, "k": 65})
    with pytest.raises(ValueError, match="nbr must be int32"):
        knn_metal.scan_select_sph(spos, start, order, pix, pix, nbr[:5], alive, **kw)
    with pytest.raises(ValueError, match="alive must be bool"):
        knn_metal.scan_select_sph(spos, start, order, pix, pix, nbr, pix, **kw)
