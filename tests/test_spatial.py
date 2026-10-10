"""§4 fixed-K neighbour search: brute-force equality, overflow, seams, padding and determinism."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from alife import spatial
from alife.config import Config, apply_overrides, dims

AMBIG = 1e-4  # distance tolerance treated as a float tie (boundary of R, or K-th vs (K+1)-th)


def _np(a: mx.array) -> np.ndarray:
    return np.array(a)


def _run(pos: np.ndarray, alive: np.ndarray, cfg: Config) -> dict[str, np.ndarray]:
    nb = spatial.find_neighbors(mx.array(pos.astype(np.float32)), mx.array(alive), cfg)
    mx.eval(nb)
    return {k: _np(v) for k, v in nb._asdict().items()}


def _pairs(pos: np.ndarray, L: float) -> tuple[np.ndarray, np.ndarray]:
    """float64 minimum-image offsets ``off[i, j] = pos[j] − pos[i]`` and distances."""
    p = pos.astype(np.float64)
    off = p[None, :, :] - p[:, None, :]
    off -= L * np.round(off / L)
    return off, np.sqrt((off * off).sum(-1))


def _cell_counts(pos: np.ndarray, alive: np.ndarray, cfg: Config) -> np.ndarray:
    d = dims(cfg)
    c = np.floor(pos.astype(np.float32) / np.float32(d.cell_hash)).astype(np.int64) % d.g_hash
    return np.bincount((c[:, 1] * d.g_hash + c[:, 0])[alive], minlength=d.g_hash**2)


def _check(nb: dict[str, np.ndarray], pos: np.ndarray, alive: np.ndarray, cfg: Config, exact: bool) -> int:
    """Structural checks on every row; if ``exact``, equality with brute-force kNN. Returns rows compared."""
    d = dims(cfg)
    n, k, L, R = d.n, d.k, cfg.world.size, cfg.sensing.radius
    idx, valid, off, dist = nb["idx"], nb["valid"], nb["off"], nb["dist"]
    assert idx.shape == (n, k) and idx.dtype == np.int32
    assert valid.shape == (n, k) and valid.dtype == np.bool_
    assert off.shape == (n, k, 2) and off.dtype == np.float32
    assert dist.shape == (n, k) and dist.dtype == np.float32
    assert nb["overflow_cells"].shape == () and nb["overflow_cells"].dtype == np.int32
    assert nb["max_occupancy"].shape == () and nb["max_occupancy"].dtype == np.int32

    rows = np.arange(n)[:, None]
    # Padding: own index, dist R, zero offset; valid entries form a prefix (sorted, nearest first).
    assert np.all(np.where(valid, True, idx == rows))
    assert np.all(np.where(valid, True, dist == np.float32(R)))
    assert np.all(np.where(valid[..., None], True, off == 0))
    assert np.all(valid[:, 1:] <= valid[:, :-1])
    assert np.all((dist[:, 1:] >= dist[:, :-1])[valid[:, 1:]])
    assert not valid[~alive].any(), "dead rows list nobody"
    # Every listed neighbour is alive, not self, strictly within R, with the right offset and distance.
    assert np.all(np.where(valid, idx != rows, True))
    assert np.all(alive[idx[valid]])
    roff, rdist = _pairs(pos, L)
    i_v, k_v = np.nonzero(valid)
    j_v = idx[i_v, k_v]
    assert np.all(np.abs(dist[i_v, k_v] - rdist[i_v, j_v]) < AMBIG)
    assert np.all(np.abs(off[i_v, k_v] - roff[i_v, j_v]) < AMBIG)
    assert np.all(rdist[i_v, j_v] < R + AMBIG)
    for i in range(n):
        ids = idx[i][valid[i]]
        assert len(np.unique(ids)) == len(ids), "no duplicates"

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


def _dense_cfg(micro_cfg: Config) -> Config:
    """L = 48 (G_h = 8, 64 hash cells) with N = 512, so most agents have more than K neighbours in range."""
    return apply_overrides(micro_cfg, ["population.n_max=512", "population.max_births_per_step=32"])


def _scene(rng: np.random.Generator, n: int, L: float, dead_frac: float) -> tuple[np.ndarray, np.ndarray]:
    pos = rng.uniform(0.0, L, (n, 2)).astype(np.float32)
    pos = np.where(pos >= L, np.float32(0.0), pos)
    return pos, rng.random(n) >= dead_frac


def test_matches_brute_force_random(tiny_cfg: Config, micro_cfg: Config) -> None:
    """Random torus scenes with 20 % dead rows; exact kNN equality when overflow-free."""
    for cfg in (tiny_cfg, _dense_cfg(micro_cfg)):
        d = dims(cfg)
        rng = np.random.default_rng(1)
        done = 0
        for _ in range(12):
            pos, alive = _scene(rng, d.n, cfg.world.size, 0.2)
            if _cell_counts(pos, alive, cfg).max() > d.c:
                continue  # this test is about the overflow-free regime
            nb = _run(pos, alive, cfg)
            assert nb["overflow_cells"] == 0
            assert nb["max_occupancy"] == _cell_counts(pos, alive, cfg).max()
            rows = _check(nb, pos, alive, cfg, exact=True)
            assert rows >= 0.95 * alive.sum()
            done += 1
            if done == 3:
                break
        assert done == 3
    # The dense scene really exercises selection: many rows have K valid entries.
    assert (nb["valid"].sum(1) == dims(cfg).k).mean() > 0.5


def test_edges_corners_and_seams(micro_cfg: Config) -> None:
    """Agents within R of the edges and corners (including 0 and the largest float32 below L)."""
    cfg = _dense_cfg(micro_cfg)
    d, L, R = dims(cfg), cfg.world.size, cfg.sensing.radius
    rng = np.random.default_rng(7)
    below_l = np.nextafter(np.float32(L), np.float32(0.0))
    for _ in range(3):
        n_used = 160
        band = rng.uniform(0.0, R, (n_used, 2))
        side = rng.random((n_used, 2)) < 0.5
        p = np.where(side, band, L - band).astype(np.float32)
        p[:4] = [[0.0, 0.0], [below_l, below_l], [0.0, below_l], [below_l, 0.0]]
        p = np.where(p >= L, below_l, p)
        pos = np.zeros((d.n, 2), np.float32)
        pos[:n_used] = p
        alive = np.zeros(d.n, bool)
        alive[:n_used] = rng.random(n_used) >= 0.2
        alive[:4] = True
        if _cell_counts(pos, alive, cfg).max() > d.c:
            continue
        nb = _run(pos, alive, cfg)
        assert nb["overflow_cells"] == 0
        _check(nb, pos, alive, cfg, exact=True)
        # The four corner agents are mutual neighbours at (nearly) zero distance across both seams.
        for i in range(4):
            listed = set(nb["idx"][i][nb["valid"][i]].tolist())
            assert {0, 1, 2, 3} - {i} <= listed


def test_seam_offsets_are_minimum_image(micro_cfg: Config) -> None:
    L = micro_cfg.world.size
    pos = np.zeros((dims(micro_cfg).n, 2), np.float32)
    alive = np.zeros(dims(micro_cfg).n, bool)
    pos[0], pos[1] = (0.5, 10.0), (L - 0.5, 10.0)  # x seam
    pos[2], pos[3] = (20.0, 0.25), (20.0, L - 0.75)  # y seam
    alive[:4] = True
    nb = _run(pos, alive, micro_cfg)
    assert nb["valid"][:4, 0].all() and not nb["valid"][:4, 1].any()
    assert nb["idx"][:4, 0].tolist() == [1, 0, 3, 2]
    np.testing.assert_allclose(nb["off"][0, 0], [-1.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(nb["off"][1, 0], [1.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(nb["off"][2, 0], [0.0, -1.0], atol=1e-5)
    np.testing.assert_allclose(nb["off"][3, 0], [0.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(nb["dist"][:4, 0], 1.0, atol=1e-5)


def test_empty_world_and_single_agent(micro_cfg: Config) -> None:
    d, R = dims(micro_cfg), micro_cfg.sensing.radius
    pos = np.random.default_rng(0).uniform(0, micro_cfg.world.size, (d.n, 2)).astype(np.float32)
    for n_alive in (0, 1):
        alive = np.zeros(d.n, bool)
        alive[:n_alive] = True
        nb = _run(pos, alive, micro_cfg)
        assert not nb["valid"].any()
        assert np.array_equal(nb["idx"], np.broadcast_to(np.arange(d.n)[:, None], (d.n, d.k)))
        assert np.all(nb["dist"] == np.float32(R)) and np.all(nb["off"] == 0)
        assert nb["overflow_cells"] == 0 and nb["max_occupancy"] == n_alive


def test_k_greater_than_available_and_dead_excluded(micro_cfg: Config) -> None:
    """Three agents in range of each other plus a dead agent on top of them: each lists exactly two."""
    d = dims(micro_cfg)
    pos = np.zeros((d.n, 2), np.float32)
    alive = np.zeros(d.n, bool)
    pos[:4] = [[10.0, 10.0], [11.0, 10.0], [10.0, 12.0], [10.5, 10.5]]
    pos[4] = [30.0, 30.0]  # alive, isolated
    alive[[0, 1, 2, 4]] = True
    nb = _run(pos, alive, micro_cfg)
    _check(nb, pos, alive, micro_cfg, exact=True)
    assert nb["valid"].sum(1)[:5].tolist() == [2, 2, 2, 0, 0]
    assert 3 not in nb["idx"][nb["valid"]]
    assert nb["idx"][0, :2].tolist() == [1, 2]  # sorted by distance: 1.0 then 2.0


def test_radius_is_strict(micro_cfg: Config) -> None:
    R = micro_cfg.sensing.radius
    pos = np.zeros((dims(micro_cfg).n, 2), np.float32)
    alive = np.zeros(dims(micro_cfg).n, bool)
    pos[:3] = [[10.0, 10.0], [10.0 + R, 10.0], [10.0, 10.0 + R - 0.01]]
    alive[:3] = True
    nb = _run(pos, alive, micro_cfg)
    assert nb["idx"][0][nb["valid"][0]].tolist() == [2]


def test_overflow_is_detected_and_yields_valid_subset(micro_cfg: Config) -> None:
    cfg = _dense_cfg(micro_cfg)
    d = dims(cfg)
    rng = np.random.default_rng(3)
    pos, alive = _scene(rng, d.n, cfg.world.size, 0.9)
    crowd = np.arange(20)
    pos[crowd] = rng.uniform(13.0, 15.0, (20, 2)).astype(np.float32)  # one hash cell, all within R
    alive[crowd] = True
    counts = _cell_counts(pos, alive, cfg)
    nb = _run(pos, alive, cfg)
    assert nb["overflow_cells"] == (counts > d.c).sum() >= 1
    assert nb["max_occupancy"] == counts.max() >= 20
    _check(nb, pos, alive, cfg, exact=False)
    # Crowd members see at least C − 1 others in their own cell, so their lists are full.
    assert np.all(nb["valid"][crowd].sum(1) == d.k)


def test_deterministic_and_compiled_matches_eager(tiny_cfg: Config) -> None:
    d = dims(tiny_cfg)
    pos, alive = _scene(np.random.default_rng(11), d.n, tiny_cfg.world.size, 0.3)
    a = _run(pos, alive, tiny_cfg)
    b = _run(pos, alive, tiny_cfg)
    for key in a:
        assert np.array_equal(a[key], b[key]), key
    fn = mx.compile(lambda p, al: spatial.find_neighbors(p, al, tiny_cfg))
    c = fn(mx.array(pos), mx.array(alive))
    mx.eval(c)
    assert np.array_equal(_np(c.idx), a["idx"]) and np.array_equal(_np(c.valid), a["valid"])
    np.testing.assert_allclose(_np(c.dist), a["dist"], atol=1e-5)
    np.testing.assert_allclose(_np(c.off), a["off"], atol=1e-5)


@pytest.mark.parametrize("overlay", [["sensing.k_neighbors=6", "sensing.cell_capacity=12"]])
def test_fast_overlay_shapes(micro_cfg: Config, overlay: list[str]) -> None:
    cfg = apply_overrides(_dense_cfg(micro_cfg), overlay)
    d = dims(cfg)
    rng = np.random.default_rng(5)
    for _ in range(10):
        pos, alive = _scene(rng, d.n, cfg.world.size, 0.5)
        if _cell_counts(pos, alive, cfg).max() <= d.c:
            break
    nb = _run(pos, alive, cfg)
    assert nb["idx"].shape == (d.n, 6)
    _check(nb, pos, alive, cfg, exact=bool(nb["overflow_cells"] == 0))
