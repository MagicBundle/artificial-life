"""§27.2 the 3D neighbour search: brute-force equality with x, y periodic and z bounded, the floor, the
surface and the x/y seams, thin slabs (G_z = 1, 2), the top hash layer, overflow over G²·G_z cells and
compile."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from alife import spatial
from alife.config import Config, apply_overrides, dims
from helpers import as_3d

AMBIG = 1e-4  # distance tolerance treated as a float tie (boundary of R, or K-th vs (K+1)-th)


def _np(a: mx.array) -> np.ndarray:
    return np.array(a)


def _run(pos: np.ndarray, alive: np.ndarray, cfg: Config) -> dict[str, np.ndarray]:
    nb = spatial.find_neighbors(mx.array(pos.astype(np.float32)), mx.array(alive), cfg)
    mx.eval(nb)
    return {k: _np(v) for k, v in nb._asdict().items()}


def _pairs3(pos: np.ndarray, L: float) -> tuple[np.ndarray, np.ndarray]:
    """float64 offsets ``off[i, j] = pos[j] − pos[i]``: minimum image on x and y, a plain difference on z."""
    p = pos.astype(np.float64)
    off = p[None, :, :] - p[:, None, :]
    off[..., :2] -= L * np.round(off[..., :2] / L)
    return off, np.sqrt((off * off).sum(-1))


def _hash3(pos: np.ndarray, cfg: Config) -> np.ndarray:
    """The z-major hash cell of each position, in float32 like the search (z clamped, never wrapped)."""
    d = dims(cfg)
    p = pos.astype(np.float32)
    cxy = np.floor(p[:, :2] / np.float32(d.cell_hash)).astype(np.int64) % d.g_hash
    cz = np.clip(np.floor(p[:, 2] / np.float32(d.cell_hash_z)).astype(np.int64), 0, d.g_hash_z - 1)
    return (cz * d.g_hash + cxy[:, 1]) * d.g_hash + cxy[:, 0]


def _cell_counts3(pos: np.ndarray, alive: np.ndarray, cfg: Config) -> np.ndarray:
    d = dims(cfg)
    return np.bincount(_hash3(pos, cfg)[alive], minlength=d.g_hash**2 * d.g_hash_z)


def _check3(nb: dict[str, np.ndarray], pos: np.ndarray, alive: np.ndarray, cfg: Config, exact: bool) -> int:
    """Structural checks on every row; if ``exact``, equality with brute-force kNN. Returns rows compared."""
    d = dims(cfg)
    n, k, L, R = d.n, d.k, cfg.world.size, cfg.sensing.radius
    idx, valid, off, dist = nb["idx"], nb["valid"], nb["off"], nb["dist"]
    assert idx.shape == (n, k) and idx.dtype == np.int32
    assert valid.shape == (n, k) and valid.dtype == np.bool_
    assert off.shape == (n, k, 3) and off.dtype == np.float32
    assert dist.shape == (n, k) and dist.dtype == np.float32
    assert nb["overflow_cells"].shape == () and nb["overflow_cells"].dtype == np.int32
    assert nb["max_occupancy"].shape == () and nb["max_occupancy"].dtype == np.int32

    rows = np.arange(n)[:, None]
    assert np.all(np.where(valid, True, idx == rows))
    assert np.all(np.where(valid, True, dist == np.float32(R)))
    assert np.all(np.where(valid[..., None], True, off == 0))
    assert np.all(valid[:, 1:] <= valid[:, :-1])
    assert np.all((dist[:, 1:] >= dist[:, :-1])[valid[:, 1:]])
    assert not valid[~alive].any(), "dead rows list nobody"
    assert np.all(np.where(valid, idx != rows, True))
    assert np.all(alive[idx[valid]])
    roff, rdist = _pairs3(pos, L)
    i_v, k_v = np.nonzero(valid)
    j_v = idx[i_v, k_v]
    assert np.all(np.abs(dist[i_v, k_v] - rdist[i_v, j_v]) < AMBIG)
    assert np.all(np.abs(off[i_v, k_v] - roff[i_v, j_v]) < AMBIG)
    assert np.all(rdist[i_v, j_v] < R + AMBIG)
    p32 = pos.astype(np.float32)
    assert np.array_equal(off[i_v, k_v, 2], p32[j_v, 2] - p32[i_v, 2]), "z is a plain float32 difference"
    for i in range(n):
        ids = idx[i][valid[i]]
        assert len(np.unique(ids)) == len(ids), f"row {i}: duplicates"

    if not exact:
        return 0
    compared = 0
    for i in np.flatnonzero(alive):
        cand = np.flatnonzero(alive & (np.arange(n) != i) & (rdist[i] < R))
        near = np.abs(rdist[i][alive & (np.arange(n) != i)] - R) < AMBIG
        ref = cand[np.argsort(rdist[i, cand], kind="stable")]
        if near.any() or (len(ref) > k and rdist[i, ref[k]] - rdist[i, ref[k - 1]] < AMBIG):
            continue  # a genuine float tie at a selection boundary: either answer is correct
        compared += 1
        got = idx[i][valid[i]]
        assert len(got) == min(k, len(ref)), f"row {i}: {len(got)} listed, expected {min(k, len(ref))}"
        assert set(got.tolist()) == set(ref[:k].tolist()), f"row {i}"
    return compared


def _dense3(micro_cfg: Config, **sections: dict[str, float]) -> Config:
    """``micro`` in 3D with N = 1024: L = 48 (G_h = 8) and height 24 (G_z = 4) unless ``sections`` override
    them (``as_3d``), so most agents have more than K neighbours in range."""
    base = apply_overrides(micro_cfg, ["population.n_max=1024", "population.max_births_per_step=32"])
    return as_3d(base, **sections)


def _scene3(rng: np.random.Generator, cfg: Config, dead_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Uniform in ``[0, L)² × [z_lo, z_hi]`` with ``dead_frac`` dead rows."""
    d, L = dims(cfg), cfg.world.size
    xy = rng.uniform(0.0, L, (d.n, 2)).astype(np.float32)
    xy = np.where(xy >= L, np.float32(0.0), xy)
    z = rng.uniform(d.z_lo, d.z_hi, (d.n, 1)).astype(np.float32)
    return np.concatenate([xy, z], axis=1), rng.random(d.n) >= dead_frac


def _overflow_free(rng: np.random.Generator, cfg: Config, dead_frac: float, tries: int = 12):
    for _ in range(tries):
        pos, alive = _scene3(rng, cfg, dead_frac)
        if _cell_counts3(pos, alive, cfg).max() <= dims(cfg).c:
            yield pos, alive


def test_matches_brute_force_random(tiny_cfg: Config, micro_cfg: Config) -> None:
    """Random volume scenes with 20 % dead rows; exact kNN equality when overflow-free."""
    for cfg in (as_3d(tiny_cfg), _dense3(micro_cfg)):
        d = dims(cfg)
        assert d.g_hash_z == 4 and d.q == 27 * d.c
        done = 0
        for pos, alive in _overflow_free(np.random.default_rng(1), cfg, 0.2):
            nb = _run(pos, alive, cfg)
            assert nb["overflow_cells"] == 0
            assert nb["max_occupancy"] == _cell_counts3(pos, alive, cfg).max()
            rows = _check3(nb, pos, alive, cfg, exact=True)
            assert rows >= 0.95 * alive.sum()
            done += 1
            if done == 3:
                break
        assert done == 3
    # The dense scene really exercises selection: many rows have K valid entries.
    assert (nb["valid"].sum(1) == dims(cfg).k).mean() > 0.3


def test_floor_surface_and_xy_seams(micro_cfg: Config) -> None:
    """Agents within R of the x/y edges and of the floor and the surface, including the eight x/y corners at
    z = z_lo and z = z_hi (0 and the largest float32 below L)."""
    cfg = _dense3(micro_cfg)
    d, L, R = dims(cfg), cfg.world.size, cfg.sensing.radius
    rng = np.random.default_rng(7)
    below_l = np.nextafter(np.float32(L), np.float32(0.0))
    corners = [[0.0, 0.0], [below_l, below_l], [0.0, below_l], [below_l, 0.0]]
    done = 0
    for _ in range(6):
        n_used = 64  # spread over the corner hash cells: a few per cell, so no overflow
        band = rng.uniform(0.0, R, (n_used, 3))
        side = rng.random((n_used, 3)) < 0.5
        xy = np.where(side[:, :2], band[:, :2], L - band[:, :2])
        z = np.where(side[:, 2], d.z_lo + band[:, 2], d.z_hi - band[:, 2])
        p = np.concatenate([xy, z[:, None]], axis=1).astype(np.float32)
        p[:4] = [[*c, d.z_lo] for c in corners]
        p[4:8] = [[*c, d.z_hi] for c in corners]
        p[:, :2] = np.where(p[:, :2] >= L, below_l, p[:, :2])
        pos = np.zeros((d.n, 3), np.float32)
        pos[:, 2] = d.z_lo
        pos[:n_used] = p
        alive = np.zeros(d.n, bool)
        alive[:n_used] = rng.random(n_used) >= 0.2
        alive[:8] = True
        if _cell_counts3(pos, alive, cfg).max() > d.c:
            continue
        nb = _run(pos, alive, cfg)
        assert nb["overflow_cells"] == 0
        _check3(nb, pos, alive, cfg, exact=True)
        # Each corner quartet is mutually adjacent across both x/y seams; floor and surface quartets are not.
        for i in range(8):
            listed = set(nb["idx"][i][nb["valid"][i]].tolist())
            mine = set(range(0, 4)) if i < 4 else set(range(4, 8))
            assert mine - {i} <= listed and not (set(range(8)) - mine) & listed, i
        done += 1
    assert done >= 3


@pytest.mark.parametrize("height", [8.0, 12.0])
def test_thin_slabs_match_brute_force(micro_cfg: Config, height: float) -> None:
    """G_z = 1 and 2: out-of-range stencil layers are empty, so no hash layer is visited twice."""
    cfg = _dense3(micro_cfg, world={"height": height})
    d = dims(cfg)
    assert d.g_hash_z == {8.0: 1, 12.0: 2}[height]
    done = 0
    for pos, alive in _overflow_free(np.random.default_rng(int(height)), cfg, 0.5):
        nb = _run(pos, alive, cfg)
        assert nb["overflow_cells"] == 0
        assert _check3(nb, pos, alive, cfg, exact=True) >= 0.95 * alive.sum()
        done += 1
        if done == 2:
            break
    assert done == 2
    assert (nb["valid"].sum(1) == d.k).mean() > 0.3  # dense enough to exercise selection


def test_z_is_not_periodic(micro_cfg: Config) -> None:
    """The same x/y at the floor and under the surface are 22 apart, not 2 (24 − 22): never neighbours."""
    cfg = _dense3(micro_cfg)
    d = dims(cfg)
    pos = np.zeros((d.n, 3), np.float32)
    pos[:, 2] = d.z_lo
    alive = np.zeros(d.n, bool)
    pos[:4] = [
        [10.0, 10.0, d.z_lo],
        [10.0, 10.0, d.z_hi],
        [10.5, 10.0, d.z_lo + 2.0],
        [10.0, 10.5, d.z_hi - 3.0],
    ]
    alive[:4] = True
    nb = _run(pos, alive, cfg)
    _check3(nb, pos, alive, cfg, exact=True)
    assert nb["idx"][0][nb["valid"][0]].tolist() == [2]
    assert nb["idx"][1][nb["valid"][1]].tolist() == [3]
    np.testing.assert_allclose(nb["off"][0, 0], [0.5, 0.0, 2.0], atol=1e-6)  # neighbour above: dz > 0
    np.testing.assert_allclose(nb["off"][1, 0], [0.0, 0.5, -3.0], atol=1e-6)  # neighbour below: dz < 0


def test_radius_is_strict_along_z(micro_cfg: Config) -> None:
    cfg = _dense3(micro_cfg)
    R = cfg.sensing.radius
    pos = np.zeros((dims(cfg).n, 3), np.float32)
    pos[:, 2] = dims(cfg).z_lo
    alive = np.zeros(dims(cfg).n, bool)
    pos[:3] = [[10.0, 10.0, 5.0], [10.0, 10.0, 5.0 + R], [10.0, 10.0, 5.0 + R - 0.01]]
    alive[:3] = True
    nb = _run(pos, alive, cfg)
    assert nb["idx"][0][nb["valid"][0]].tolist() == [2]
    np.testing.assert_allclose(nb["dist"][0, 0], R - 0.01, atol=1e-6)


@pytest.mark.parametrize(("height", "radius"), [(24.0, 6.0), (28.0, 4.5)])
def test_top_of_the_world_maps_to_the_top_layer(micro_cfg: Config, height: float, radius: float) -> None:
    """z = height (and, at height 28 with R 4.5, z = nextafter(height, 0)) divides to G_z in float32: the
    layer is clamped to G_z − 1, so those agents stay in a real cell and visible."""
    cfg = _dense3(micro_cfg, world={"height": height, "size": 64.0}, sensing={"radius": radius})
    d = dims(cfg)
    top = float(np.nextafter(np.float32(height), np.float32(0)))
    zs = np.float32([top, height, height - 3.0])
    assert np.floor(np.float32(height) / np.float32(d.cell_hash_z)) == d.g_hash_z
    if height == 28.0:  # the rounding case: the largest float32 below the height still divides to G_z
        assert np.floor(np.float32(top) / np.float32(d.cell_hash_z)) == d.g_hash_z
    pos = np.zeros((d.n, 3), np.float32)
    alive = np.zeros(d.n, bool)
    pos[:3] = [[20.0, 20.0, z] for z in zs]
    alive[:3] = True
    assert _hash3(pos[:3], cfg).tolist() == [_hash3(pos[2:3], cfg)[0]] * 3  # one top-layer cell
    nb = _run(pos, alive, cfg)
    assert nb["max_occupancy"] == 3 and nb["overflow_cells"] == 0
    for i in range(3):
        assert set(nb["idx"][i][nb["valid"][i]].tolist()) == {0, 1, 2} - {i}, i


def test_overflow_is_detected_and_yields_valid_subset(micro_cfg: Config) -> None:
    cfg = _dense3(micro_cfg)
    d = dims(cfg)
    rng = np.random.default_rng(3)
    pos, alive = _scene3(rng, cfg, 0.9)
    crowd = np.arange(20)
    pos[crowd, :2] = rng.uniform(13.0, 15.0, (20, 2))  # one hash cell (2, 2, 1), all within R
    pos[crowd, 2] = rng.uniform(7.0, 9.0, 20)
    alive[crowd] = True
    counts = _cell_counts3(pos, alive, cfg)
    assert counts.shape == (d.g_hash**2 * d.g_hash_z,)
    nb = _run(pos, alive, cfg)
    assert nb["overflow_cells"] == (counts > d.c).sum() >= 1
    assert nb["max_occupancy"] == counts.max() >= 20
    _check3(nb, pos, alive, cfg, exact=False)
    assert np.all(nb["valid"][crowd].sum(1) == d.k)


def test_a_column_does_not_overflow(micro_cfg: Config) -> None:
    """Twenty agents over one x/y hash cell but spread over the four hash layers: no 3D cell is overfull, so
    the result is exact (a 2D hash of the same x/y would overflow C = 16)."""
    cfg = _dense3(micro_cfg)
    d = dims(cfg)
    rng = np.random.default_rng(4)
    pos = np.zeros((d.n, 3), np.float32)
    pos[:, 2] = d.z_lo
    alive = np.zeros(d.n, bool)
    col = np.arange(20)
    pos[col, :2] = rng.uniform(13.0, 17.0, (20, 2))
    pos[col, 2] = np.repeat(np.float32([3.0, 9.0, 15.0, 21.0]), 5)
    alive[col] = True
    nb = _run(pos, alive, cfg)
    assert nb["overflow_cells"] == 0 and nb["max_occupancy"] == 5
    _check3(nb, pos, alive, cfg, exact=True)


def test_empty_world_and_single_agent(micro_cfg: Config) -> None:
    cfg = _dense3(micro_cfg)
    d, R = dims(cfg), cfg.sensing.radius
    pos, _ = _scene3(np.random.default_rng(0), cfg, 0.0)
    for n_alive in (0, 1):
        alive = np.zeros(d.n, bool)
        alive[:n_alive] = True
        nb = _run(pos, alive, cfg)
        assert not nb["valid"].any()
        assert np.array_equal(nb["idx"], np.broadcast_to(np.arange(d.n)[:, None], (d.n, d.k)))
        assert np.all(nb["dist"] == np.float32(R)) and np.all(nb["off"] == 0)
        assert nb["overflow_cells"] == 0 and nb["max_occupancy"] == n_alive


def test_dispatch_and_the_2d_guard(micro_cfg: Config) -> None:
    """A 3D config takes the 3D search; the 2D search refuses (N,3) positions instead of ignoring z."""
    cfg = _dense3(micro_cfg)
    pos, alive = _scene3(np.random.default_rng(6), cfg, 0.5)
    assert _run(pos, alive, cfg)["off"].shape == (dims(cfg).n, dims(cfg).k, 3)
    with pytest.raises(AssertionError, match="3D positions"):
        spatial.find_neighbors(mx.array(pos[:64]), mx.array(alive[:64]), micro_cfg)


def test_deterministic_and_compiled_matches_eager(tiny_cfg: Config) -> None:
    cfg = as_3d(tiny_cfg)
    pos, alive = _scene3(np.random.default_rng(11), cfg, 0.3)
    a = _run(pos, alive, cfg)
    b = _run(pos, alive, cfg)
    for key in a:
        assert np.array_equal(a[key], b[key]), key
    fn = mx.compile(lambda p, al: spatial.find_neighbors(p, al, cfg))
    c = fn(mx.array(pos), mx.array(alive))
    mx.eval(c)
    assert np.array_equal(_np(c.idx), a["idx"]) and np.array_equal(_np(c.valid), a["valid"])
    assert c.overflow_cells.item() == a["overflow_cells"] and c.max_occupancy.item() == a["max_occupancy"]
    np.testing.assert_allclose(_np(c.dist), a["dist"], atol=1e-5)
    np.testing.assert_allclose(_np(c.off), a["off"], atol=1e-5)
