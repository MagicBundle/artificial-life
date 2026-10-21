"""§4 "Fused Metal path", F21, I48: the fused Metal neighbour search against the pure-MLX search.

The GPU tests (skipped off a Metal GPU, so on the Linux CI) compare ``find_neighbors(impl="metal")`` with
``impl="mlx"`` bit for bit, floats as uint32 patterns, eager and under ``mx.compile``, on every field of
``Neighbors``: crafted scenes with exact ties, the R² boundary, the seams, the floor and the surface, a
randomised geometry fuzz, out-of-contract inputs, and the whole compiled step (the search of the step's own
S2 inputs compared every step, and the state digests of a kernel step and an MLX step in lockstep). They run
under every MLX version: the autouse fixture adds the running version to ``VERIFIED_MLX``, because the point
of these tests is to decide whether to add it for real (DESIGN §4, F21: re-run them on every MLX bump).
The CPU tests (selection rules, guards, static asserts) run everywhere.
"""

from __future__ import annotations

import dataclasses
import warnings
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from alife import cli, knn_metal, spatial
from alife import step as step_mod
from alife.config import Config, config_from_dict, config_to_dict, dims, load_config
from alife.runio import RunDir
from alife.state import init_state, state_digest, to_numpy
from helpers import MICRO3D, ON_GPU

GPU = ON_GPU and knn_metal._capable()
gpu = pytest.mark.skipif(not GPU, reason="needs the Metal GPU as default device and mx.fast.metal_kernel")
FIELDS = spatial.Neighbors._fields


@pytest.fixture(autouse=True)
def _running_mlx_is_verified(monkeypatch: pytest.MonkeyPatch) -> None:
    """The GPU tests are what verifies an MLX version, so they must run on one that is not listed yet. They
    also start from a clean ``ALIFE_KNN`` (every test that needs a value sets it)."""
    monkeypatch.delenv("ALIFE_KNN", raising=False)
    monkeypatch.setattr(knn_metal, "VERIFIED_MLX", knn_metal.VERIFIED_MLX | {mx.__version__})


# --- helpers --------------------------------------------------------------------------------------


def _cfg(preset: str, **over: Any) -> Config:
    """A config from a preset with ``section__key`` overrides, *not* validated (crafted geometry)."""
    base = config_to_dict(load_config(preset=preset))
    for k, v in over.items():
        sec, key = k.split("__")
        base[sec][key] = v
    return config_from_dict(base)


def _with_k(cfg: Config, k: int) -> Config:
    return dataclasses.replace(cfg, sensing=dataclasses.replace(cfg.sensing, k_neighbors=k))


def _bits(a: mx.array) -> np.ndarray:
    x = np.array(a)
    return x.view(np.uint32) if x.dtype == np.float32 else x


def _diff(a: spatial.Neighbors, b: spatial.Neighbors) -> dict[str, int]:
    """Mismatching elements per field (floats as bit patterns: the sign of zero and NaN payloads count);
    -1 where the dtype or shape differs."""
    out = {}
    for f in FIELDS:
        x, y = _bits(getattr(a, f)), _bits(getattr(b, f))
        out[f] = -1 if (x.shape != y.shape or x.dtype != y.dtype) else int((x != y).sum())
    return out


def _search(cfg: Config, pos: mx.array, alive: mx.array, impl: str, compiled: bool) -> spatial.Neighbors:
    def f(p: mx.array, a: mx.array) -> tuple[mx.array, ...]:
        return tuple(spatial.find_neighbors(p, a, cfg, impl=impl))

    out = spatial.Neighbors(*(mx.compile(f) if compiled else f)(pos, alive))
    mx.eval(out)
    return out


def _assert_same(cfg: Config, pos: np.ndarray, alive: np.ndarray) -> spatial.Neighbors:
    """The kernel equals the MLX search: eager against eager, and compiled against the compiled MLX search
    (the form the step runs in). Not against the eager MLX search: MLX 0.32.3's compile cache can hand a
    compiled function the fused elementwise kernel of another one whose scalar constants print alike when
    concatenated (cell size 12.0 with G = 1, then 1.0 with G = 21, in ``floor(p / c) % G``), which makes a
    compiled function differ from its own eager result, depending on the order of the tests (DESIGN F21 item
    6). Both searches share that preamble, so they stay equal to each other. Returns the MLX reference."""
    p, a = mx.array(pos.astype(np.float32)), mx.array(alive)
    ref = _search(cfg, p, a, "mlx", False)
    for compiled in (False, True):
        want = _search(cfg, p, a, "mlx", True) if compiled else ref
        ker = _search(cfg, p, a, "metal", compiled)
        assert not any(_diff(ker, want).values()), (compiled, _diff(ker, want))
    return ref


def _ties_in_k(ref: spatial.Neighbors) -> int:
    """Rows whose K list holds an exactly repeated finite distance."""
    dv = np.where(np.array(ref.valid), np.array(ref.dist), np.nan)
    return int((np.diff(np.sort(dv, 1), axis=1) == 0).any(1).sum()) if dv.shape[1] > 1 else 0


def _ties_at_k(cfg: Config, pos: np.ndarray, alive: np.ndarray) -> int:
    """Rows tied at the K boundary: with K + 1 neighbours, ``dist[K-1] == dist[K]``, both valid."""
    k = dims(cfg).k
    if k >= dims(cfg).q:
        return 0
    r1 = spatial.find_neighbors(
        mx.array(pos.astype(np.float32)), mx.array(alive), _with_k(cfg, k + 1), impl="mlx"
    )
    d, v = np.array(r1.dist), np.array(r1.valid)
    return int((v[:, k - 1] & v[:, k] & (d[:, k - 1] == d[:, k])).sum())


def _d2_matrix(cfg: Config, pos: np.ndarray, alive: np.ndarray) -> np.ndarray:
    """Float32 brute-force d² between alive agents (min image in x and y, the diagonal at inf; n small)."""
    L = np.float32(cfg.world.size)
    p = pos[alive].astype(np.float32)
    d = p[None, :, :] - p[:, None, :]
    d[..., :2] = d[..., :2] - L * np.rint(d[..., :2] / L)
    d2 = d[..., 0] * d[..., 0]
    for ax in range(1, p.shape[1]):
        d2 = d2 + d[..., ax] * d[..., ax]
    np.fill_diagonal(d2, np.inf)
    return d2


def _pairs_at_r(cfg: Config, pos: np.ndarray, alive: np.ndarray) -> int:
    """Ordered pairs of alive agents at d² exactly float32(R·R), by float32 brute force (n small)."""
    return int((_d2_matrix(cfg, pos, alive) == np.float32(cfg.sensing.radius**2)).sum())


def _pairs_between_rr(cfg: Config, pos: np.ndarray, alive: np.ndarray) -> int:
    """Ordered pairs whose d² is inside float32(R·R) (the float64 product rounded once) but not inside
    float32(R)·float32(R), or the other way round: the pairs that tell the two roundings of the constant
    apart."""
    R = cfg.sensing.radius
    a, b = np.float32(R * R), np.float32(R) * np.float32(R)
    d2 = _d2_matrix(cfg, pos, alive)
    return int(((d2 < a) != (d2 < b)).sum())


# --- crafted scenes -------------------------------------------------------------------------------


def _cfg3(
    n: int = 1024, L: float = 128.0, H: float = 24.0, R: float = 6.0, C: int = 16, K: int = 8
) -> Config:
    return _cfg(
        "tiny3d",
        population__n_max=n,
        population__n_init=min(256, n),
        population__max_births_per_step=min(32, n // 4),
        world__size=L,
        world__height=H,
        sensing__radius=R,
        sensing__cell_capacity=C,
        sensing__k_neighbors=K,
    )


def _cfg2(n: int = 1024, L: float = 128.0, R: float = 6.0, C: int = 16, K: int = 8) -> Config:
    return _cfg(
        "tiny",
        population__n_max=n,
        population__n_init=min(256, n),
        population__max_births_per_step=min(32, n // 4),
        world__size=L,
        sensing__radius=R,
        sensing__cell_capacity=C,
        sensing__k_neighbors=K,
    )


def _pad(points: Any, n: int, nd: int) -> tuple[np.ndarray, np.ndarray]:
    p = np.zeros((n, nd), np.float32)
    a = np.zeros(n, bool)
    pts = np.asarray(points, np.float32).reshape(-1, nd)
    p[: len(pts)] = pts
    a[: len(pts)] = True
    return p, a


_C0 = np.array([64.5, 64.5, 12.0], np.float32)
_Scene = tuple[Config, np.ndarray, np.ndarray]


def _rng(name: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(name.encode()))


def _scene_symmetric_shells() -> _Scene:
    pts = [_C0]
    for ax in range(3):
        for s in (-2.0, 2.0):
            q = _C0.copy()
            q[ax] += s
            pts.append(q)
    edges = [(1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0), (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1)]
    edges += [(0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)]
    pts += [_C0 + np.array(e, np.float32) * 1.5 for e in edges]
    pts += [
        _C0 + np.array([dx, dy, dz], np.float32) * 1.25 for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)
    ]
    return (_cfg3(), *_pad(np.array(pts), 1024, 3))


def _scene_identical_x20() -> _Scene:
    return (_cfg3(), *_pad(np.repeat(_C0[None], 20, 0), 1024, 3))


def _scene_r2_boundary() -> _Scene:
    R = np.float32(6.0)
    pts = [_C0]
    for ax in range(3):
        for s in (
            R,
            -R,
            np.nextafter(R, np.float32(0)),
            -np.nextafter(R, np.float32(0)),
            np.nextafter(R, np.float32(10)),
        ):
            q = _C0.copy()
            q[ax] = np.float32(q[ax] + s)
            pts.append(q)
    pts.append(_C0 + np.array([3.6, 4.8, 0.0], np.float32))  # a 3-4-5 triangle scaled to exactly R
    pts.append(_C0 + np.array([0.0, 3.6, 4.8], np.float32))
    return (_cfg3(), *_pad(np.array(pts, np.float32), 1024, 3))


def _scene_xy_seams() -> _Scene:
    L = 128.0
    edge = float(np.nextafter(np.float32(L), np.float32(0)))
    xs = (0.0, 0.3, L - 0.3, edge, 5.9, L - 5.9)
    ys = (0.0, 0.2, L - 0.2, edge, 6.1, L - 6.1)
    return (_cfg3(), *_pad(np.array([[x, y, 12.0] for x in xs for y in ys], np.float32), 1024, 3))


def _scene_floor_surface() -> _Scene:
    cfg = _cfg3()
    d, H = dims(cfg), 24.0
    top = float(np.nextafter(np.float32(H), np.float32(0)))
    zs = (d.z_lo, d.z_hi, 0.0, H, top, 5.9, 6.1, 18.0, 17.9)
    pts = [[40.0 + 0.9 * k, 40.0 + 0.5 * k, z] for z in zs for k in range(6)]
    return (cfg, *_pad(np.array(pts, np.float32), 1024, 3))


def _scene_thin_slab(height: float) -> Callable[[], _Scene]:
    def build() -> _Scene:
        cfg = _cfg3(H=height)
        rng = _rng(f"slab{height}")
        pts = np.stack([rng.random(900) * 128, rng.random(900) * 32, rng.random(900) * height], 1)
        return (cfg, *_pad(pts, 1024, 3))

    return build


def _scene_dead_garbage() -> _Scene:
    rng = _rng("dead")
    m = 1024
    pts = np.stack([rng.random(m) * 128, rng.random(m) * 30, rng.random(m) * 24], 1).astype(np.float32)
    alive = rng.random(m) < 0.6
    pts[~alive & (rng.random(m) < 0.3)] = 0.0
    pts[~alive & (rng.random(m) < 0.3)] = 1e9
    pts[~alive & (rng.random(m) < 0.3)] = np.nan
    return (_cfg3(), pts, alive)


def _scene_cell_over_c() -> _Scene:
    cfg, rng = _cfg3(), _rng("overc")
    pts = _C0 + (rng.random((40, 3)) * 5.5).astype(np.float32) - 2.75
    pts = np.concatenate([pts, _C0 + (rng.random((60, 3)) * 16 - 8).astype(np.float32)])
    pts[:, 2] = np.clip(pts[:, 2], dims(cfg).z_lo, dims(cfg).z_hi)
    return (cfg, *_pad(pts, 1024, 3))


def _scene_sparse() -> _Scene:
    rng = _rng("sparse")
    pts = np.stack([rng.random(60) * 128, rng.random(60) * 128, rng.random(60) * 24], 1)
    return (_cfg3(), *_pad(pts, 1024, 3))


def _scene_empty(nd: int) -> Callable[[], _Scene]:
    return lambda: ((_cfg3() if nd == 3 else _cfg2()), np.zeros((1024, nd), np.float32), np.zeros(1024, bool))


def _scene_single() -> _Scene:
    return (_cfg3(), *_pad(_C0[None], 1024, 3))


def _scene_lattice(nd: int, c: int, k: int, scale: float = 1.0) -> Callable[[], _Scene]:
    def build() -> _Scene:
        rng = _rng(f"lat{nd}{c}{k}")
        cfg = (_cfg3 if nd == 3 else _cfg2)(C=c, K=k)
        cols = [rng.integers(0, 256, 1024) * 0.5, rng.integers(0, 256, 1024) * 0.5]
        if nd == 3:
            cols.append(rng.integers(2, 46, 1024) * 0.5)
        pts = (np.stack(cols, 1) * scale).astype(np.float32)
        pts[:, :2] = pts[:, :2] % 128.0
        return (cfg, pts, rng.random(1024) < 0.9)

    return build


def _scene_dense_clump() -> _Scene:
    rng = _rng("clump")
    cols = [rng.integers(0, 32, 1024) * 0.5 + 60, rng.integers(0, 32, 1024) * 0.5 + 60]
    cols.append(rng.integers(2, 34, 1024) * 0.5)
    return (_cfg3(C=48), np.stack(cols, 1).astype(np.float32), np.ones(1024, bool))


def _scene_random(dens: int, c: int) -> Callable[[], _Scene]:
    def build() -> _Scene:
        rng = _rng(f"rnd{dens}{c}")
        pts = np.stack(
            [rng.random(1024) * 128, rng.random(1024) * 128 * (512 / dens), rng.random(1024) * 24], 1
        )
        return (_cfg3(C=c), pts.astype(np.float32), rng.random(1024) < 0.95)

    return build


def _scene_g2() -> _Scene:
    """An unvalidated G = 2 world: each stencil cell is visited twice (duplicate candidates)."""
    rng = _rng("g2")
    cfg = _cfg3(L=12.0, H=12.0)
    return (cfg, *_pad(rng.random((200, 3)) * 12, 1024, 3))


def _scene_k64() -> _Scene:
    """K at the kernel's bound, on a lattice clump with C = 48 (all 27·48 candidates populated, ties)."""
    rng = _rng("k64")
    cols = [rng.integers(0, 24, 1024) * 0.5 + 60, rng.integers(0, 24, 1024) * 0.5 + 60]
    cols.append(rng.integers(2, 26, 1024) * 0.5)
    return (_cfg3(C=48, K=knn_metal.K_MAX), np.stack(cols, 1).astype(np.float32), np.ones(1024, bool))


def _scene_ring2d() -> _Scene:
    pts = [[64.5, 64.5]]
    for s in (-2.0, 2.0):
        pts += [[64.5 + s, 64.5], [64.5, 64.5 + s]]
    for dx in (-1, 1):
        for dy in (-1, 1):
            pts += [[64.5 + dx * 2**0.5, 64.5 + dy * 2**0.5], [64.5 + dx * 1.2, 64.5 + dy * 1.6]]
    return (_cfg2(), *_pad(np.array(pts, np.float32), 1024, 2))


def _scene_identical_2d() -> _Scene:
    return (_cfg2(), *_pad(np.repeat(np.array([[10.0, 10.0]], np.float32), 30, 0), 1024, 2))


def _scene_seams_2d() -> _Scene:
    edge = float(np.nextafter(np.float32(128), np.float32(0)))
    xs = (0.0, 0.3, 128 - 0.3, edge, 6.0)
    ys = (0.0, 0.2, 128 - 0.2, edge, 6.0)
    return (_cfg2(), *_pad(np.array([[x, y] for x in xs for y in ys], np.float32), 1024, 2))


def _straddle(R: float, x0: float = 20.0) -> tuple[float, float] | None:
    """``(x0, x1)`` (float32 values) whose x gap has d² between float32(R·R) (the double product rounded
    once, as MLX casts ``R * R``) and float32(R)·float32(R) (the constant computed in float32), or None if
    the two agree."""
    a, b = np.float32(R * R), np.float32(R) * np.float32(R)
    if a == b:
        return None
    c = np.float32(np.float32(x0) + np.float32(R))
    for k in range(-40, 41):
        p1 = c
        for _ in range(abs(k)):
            p1 = np.nextafter(p1, np.float32(1e9) if k > 0 else np.float32(0))
        d2 = (p1 - np.float32(x0)) ** 2
        if (d2 < a) != (d2 < b):
            return float(np.float32(x0)), float(p1)
    return None


def _scene_rr_straddle(nd: int) -> Callable[[], _Scene]:
    """R = 7.91 in L = 100.3: float32(R·R) = 62.5681 and float32(R)·float32(R) differ by one ulp, and the pair
    at x = 20.0 and 27.90999984741211 (d² = 62.568096) sits between them: a kernel that squares the float32
    radius instead of using the once-rounded double product misses the pair."""

    def build() -> _Scene:
        sc = _straddle(7.91)
        assert sc is not None
        pts = [[sc[0], 40.0, 10.0][:nd], [sc[1], 40.0, 10.0][:nd]]
        cfg = _cfg3(L=100.3, H=24.0, R=7.91) if nd == 3 else _cfg2(L=100.3, R=7.91)
        return (cfg, *_pad(np.array(pts, np.float32), 1024, nd))

    return build


def _scene_half_l(nd: int, L: float, R: float) -> Callable[[], _Scene]:
    """Pairs exactly L/2 apart along x, along y and along both, with R > L/2 (an unvalidated world: the
    validated ones have G_h >= 3, so |d| = L/2 >= 1.5·R is masked anyway). The minimum image of a difference
    of exactly L/2 is where rint (to even) and round (away from zero) part, and where x·(1/L) can differ
    from x/L."""

    def build() -> _Scene:
        half = np.float32(L) / np.float32(2)
        pts = []
        for base in (0.0, 1.0):
            a = np.array([base, base, 10.0][:nd], np.float32)
            for step in ((half, 0.0), (0.0, half), (half, half)):
                b = a.copy()
                b[:2] = a[:2] + np.array(step, np.float32)
                pts += [b]
            pts += [a]
        cfg = _cfg3(L=L, H=12.0, R=R) if nd == 3 else _cfg2(L=L, R=R)
        return (cfg, *_pad(np.array(pts, np.float32), 1024, nd))

    return build


def _scene_recip(nd: int) -> Callable[[], _Scene]:
    """L = 31.373 and a pair at x = 15.686501 (float32), one ulp off L/2, with R > L/2: ``x/L`` and
    ``x·(1/L)``
    round differently here, which is also where ``math_mode: fast`` (a reciprocal multiply) parts from
    MLX (DESIGN F21 item 5)."""

    def build() -> _Scene:
        L = 31.373
        pts = [[0.0, 1.0, 10.0][:nd], [15.686500549316406, 1.0, 10.0][:nd]]
        cfg = _cfg3(L=L, H=20.0, R=L / 2 + 0.7) if nd == 3 else _cfg2(L=L, R=L / 2 + 0.7)
        return (cfg, *_pad(np.array(pts, np.float32), 1024, nd))

    return build


def _scene_dead_nan_2d() -> _Scene:
    rng = _rng("dead2")
    pts = np.stack([rng.random(1024) * 128, rng.random(1024) * 128], 1).astype(np.float32)
    alive = rng.random(1024) < 0.7
    pts[~alive] = np.nan
    return (_cfg2(), pts, alive)


# name -> (builder, what the scene must exhibit: ties inside K, ties at the K boundary, pairs exactly at R, an
# overflowed cell). The expectations keep a scene from becoming vacuous (a lattice that stopped tying, ...).
TIE_K, TIE_BOUND, AT_R, OVERFLOW = "tie_in_k", "tie_at_k_boundary", "pairs_at_R", "overflow"
BETWEEN_RR = "pairs_between_the_roundings_of_R2"
SCENES: dict[str, tuple[Callable[[], _Scene], frozenset[str]]] = {
    "3d_symmetric_shells": (_scene_symmetric_shells, frozenset({TIE_K, TIE_BOUND})),
    "3d_identical_x20": (_scene_identical_x20, frozenset({TIE_K, OVERFLOW})),
    "3d_r2_boundary": (_scene_r2_boundary, frozenset({AT_R})),
    "3d_xy_seams_corners": (_scene_xy_seams, frozenset()),
    "3d_floor_surface": (_scene_floor_surface, frozenset()),
    **{f"3d_slab_H{h:g}": (_scene_thin_slab(h), frozenset()) for h in (6.0, 11.9, 12.0, 13.0)},
    "3d_dead_rows_garbage": (_scene_dead_garbage, frozenset()),
    "3d_cell_over_c": (_scene_cell_over_c, frozenset({OVERFLOW})),
    "3d_sparse_fewer_than_k": (_scene_sparse, frozenset()),
    "3d_empty_world": (_scene_empty(3), frozenset()),
    "3d_single_agent": (_scene_single, frozenset()),
    **{
        f"3d_lattice_C{c}_K{k}": (_scene_lattice(3, c, k), frozenset({TIE_K} if k > 1 else {TIE_BOUND}))
        for c in (4, 8, 16, 48)
        for k in (1, 6, 8)
    },
    "3d_lattice_dense_clump_C48": (_scene_dense_clump, frozenset({TIE_K, TIE_BOUND})),
    "3d_random_1024_C16": (_scene_random(1024, 16), frozenset()),
    "3d_random_1024_C48": (_scene_random(1024, 48), frozenset()),
    "3d_random_512_C8": (_scene_random(512, 8), frozenset()),
    "3d_g2_unvalidated": (_scene_g2, frozenset()),
    "3d_k_at_bound_64": (_scene_k64, frozenset({TIE_K, TIE_BOUND})),
    "2d_symmetric_ring": (_scene_ring2d, frozenset({TIE_K})),
    "2d_identical_x30": (_scene_identical_2d, frozenset({TIE_K, OVERFLOW})),
    "2d_seams_r_boundary": (_scene_seams_2d, frozenset({AT_R})),
    **{
        f"2d_lattice_C{c}_K{k}": (_scene_lattice(2, c, k), frozenset({TIE_K} if k > 1 else {TIE_BOUND}))
        for c in (4, 16)
        for k in (1, 8)
    },
    "2d_empty": (_scene_empty(2), frozenset()),
    "2d_dead_nan": (_scene_dead_nan_2d, frozenset()),
    "3d_rr_straddle_R7.91": (_scene_rr_straddle(3), frozenset({BETWEEN_RR})),
    "2d_rr_straddle_R7.91": (_scene_rr_straddle(2), frozenset({BETWEEN_RR})),
    **{
        f"{nd}d_half_L{L:g}_R{R:g}": (_scene_half_l(nd, L, R), frozenset())
        for nd in (3, 2)
        for L, R in ((12.0, 6.5), (12.6, 7.0), (10.0, 5.5), (7.7, 4.0))
    },
    "3d_recip_L31.373": (_scene_recip(3), frozenset()),
    "2d_recip_L31.373": (_scene_recip(2), frozenset()),
}


def _scene_stats(cfg: Config, pos: np.ndarray, alive: np.ndarray, ref: spatial.Neighbors) -> dict[str, int]:
    return {
        TIE_K: _ties_in_k(ref),
        TIE_BOUND: _ties_at_k(cfg, pos, alive),
        AT_R: _pairs_at_r(cfg, pos, alive),
        OVERFLOW: int(ref.overflow_cells.item()),
        BETWEEN_RR: _pairs_between_rr(cfg, pos, alive),
    }


_SEEN: dict[str, set[str]] = {}


@gpu
@pytest.mark.parametrize("name", list(SCENES))
def test_crafted_scenes_bitwise(name: str) -> None:
    build, expect = SCENES[name]
    cfg, pos, alive = build()
    ref = _assert_same(cfg, pos, alive)
    stats = _scene_stats(cfg, pos, alive, ref)
    missing = {e for e in expect if stats[e] == 0}
    assert not missing, f"{name} no longer exhibits {sorted(missing)}: {stats}"
    _SEEN[name] = {k for k, v in stats.items() if v}


@gpu
def test_crafted_scenes_cover_every_hard_case() -> None:
    """Non-vacuity of the suite as a whole: ties inside K, ties at the K boundary, candidates exactly at R
    and an overflowed cell all occur, in 2D and in 3D (runs after the parametrised scenes, on what
    they saw)."""
    for name in SCENES:
        if name not in _SEEN:
            test_crafted_scenes_bitwise(name)
    for prefix in ("2d", "3d"):
        seen = set().union(*(s for n, s in _SEEN.items() if n.startswith(prefix)))
        assert {TIE_K, AT_R, OVERFLOW} <= seen, (prefix, seen)
    assert TIE_BOUND in set().union(*_SEEN.values())


@gpu
@pytest.mark.slow
@pytest.mark.parametrize(
    "name", ["64k3d_random_uniform", "64k3d_lattice_floor_pile", "64k_uniform", "64k_lattice"]
)
def test_full_size_scenes_bitwise(name: str) -> None:
    rng = _rng(name)
    n = 65536
    if name.startswith("64k3d"):
        cfg = load_config(preset="64k3d")
        if name.endswith("uniform"):
            pts = np.stack([rng.random(n) * 320, rng.random(n) * 320, rng.random(n) * 80], 1)
        else:
            pts = np.stack(
                [rng.integers(0, 640, n) * 0.5, rng.integers(0, 640, n) * 0.5, rng.integers(2, 20, n) * 0.5],
                1,
            )
    else:
        cfg = load_config(preset="64k")
        if name.endswith("uniform"):
            pts = np.stack([rng.random(n) * 1024, rng.random(n) * 1024], 1)
        else:
            pts = np.stack([rng.integers(0, 2048, n) * 0.5, rng.integers(0, 2048, n) * 0.5], 1)
    ref = _assert_same(cfg, pts.astype(np.float32), rng.random(n) < 0.6)
    if name.endswith("lattice") or name.endswith("pile"):
        assert _ties_in_k(ref) > 1000


# --- randomised geometries ------------------------------------------------------------------------


def _positions(
    rng: np.random.Generator, nd: int, n: int, L: float, H: float, d: Any, kind: str
) -> np.ndarray:
    lo, hi = (d.z_lo, d.z_hi) if nd == 3 else (0.0, 0.0)
    if kind == "lattice":
        h = float(rng.choice([0.25, 0.5, 1.0, 1.5, 2.0, d.cell_hash / 2, d.cell_hash / 3]))
        p = rng.integers(0, max(1, int(L / h)), size=(n, nd)).astype(np.float64) * h
        p[:, :2] %= L
    elif kind == "clump":
        c = rng.random(nd) * L
        p = c + (rng.random((n, nd)) - 0.5) * rng.uniform(1.0, 3 * d.cell_hash)
        p[:, :2] %= L
    elif kind == "dupes":
        base = rng.random((max(1, n // int(rng.integers(2, 30))), nd)) * L
        p = base[rng.integers(0, len(base), size=n)]
        p[:, :2] %= L
    elif kind == "cell_edges":
        p = rng.integers(0, d.g_hash + 1, size=(n, nd)).astype(np.float64) * d.cell_hash
        if nd == 3:
            p[:, 2] = rng.integers(0, d.g_hash_z + 1, size=n) * d.cell_hash_z
        p += rng.choice([-1e-3, 0.0, 0.0, 1e-3], size=(n, nd))
        p[:, :2] %= L
    elif kind == "seams":
        p = rng.random((n, nd)) * L
        near = rng.random((n, 2)) * 2 * d.cell_hash
        p[:, :2] = np.where(rng.random((n, 2)) < 0.5, near, L - near) % L
    else:
        p = rng.random((n, nd)) * L
    p = p.astype(np.float32)
    p[:, :2] = np.where(p[:, :2] >= np.float32(L), np.float32(0.0), p[:, :2])
    if nd == 3:
        p[:, 2] = np.clip(p[:, 2] * np.float32(H / L) if kind in ("uniform", "seams") else p[:, 2], lo, hi)
    return p


def _random_cases(seed: int, count: int) -> list[tuple[Config, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    cases = []
    while len(cases) < count:
        nd = int(rng.choice([2, 3]))
        R = float(
            rng.choice([1.0, 2.0, 3.0, 4.5, 6.0, 6.5, 5.3, 7.91, 0.7, 6.1])
        )  # 5.3 and 7.91: R·R not dyadic
        L = float(R * rng.integers(1 if nd == 3 else 3, 24) + rng.choice([0.0, 0.0, 0.37, 1.9, 0.123]))
        H = (
            max(R, float(min(L / 2, R * rng.integers(1, 6) + rng.choice([0.0, 0.0, 0.41]))))
            if nd == 3
            else 0.0
        )
        C = int(rng.choice([1, 2, 3, 4, 8, 16, 24, 32, 48, 64]))
        K = int(min(3**nd * C, rng.choice([1, 2, 4, 6, 8, 12, 16])))
        n = int(rng.choice([16, 64, 256, 1024, 2048, 4096]))
        kind = str(rng.choice(["lattice", "clump", "dupes", "cell_edges", "seams", "uniform"]))
        over: dict[str, Any] = dict(
            population__n_max=n,
            population__n_init=min(8, n),
            population__max_births_per_step=max(1, min(8, n // 4)),
            world__size=L,
            sensing__radius=R,
            sensing__cell_capacity=C,
            sensing__k_neighbors=K,
        )
        if nd == 3:
            over["world__height"] = H
        cfg = _cfg("tiny3d" if nd == 3 else "tiny", **over)
        d = dims(cfg)
        if d.g_hash < 1 or (nd == 3 and d.g_hash_z < 1):
            continue
        pos = _positions(rng, nd, n, L, H, d, kind)
        alive = rng.random(n) < rng.choice([1.0, 0.95, 0.7, 0.3])
        pos[~alive & (rng.random(n) < 0.5)] = rng.choice([np.nan, 0.0, 1e9, -5.0])
        cases.append((cfg, pos, alive))
    return cases


@gpu
@pytest.mark.parametrize(
    ("seed", "count"),
    [
        (101, 30),
        pytest.param(202, 200, marks=pytest.mark.slow),
        pytest.param(303, 200, marks=pytest.mark.slow),
    ],
)
def test_random_geometries_bitwise(seed: int, count: int) -> None:
    tied_k = tied_bound = 0
    for cfg, pos, alive in _random_cases(seed, count):
        ref = _assert_same(cfg, pos, alive)
        tied_k += _ties_in_k(ref)
        tied_bound += _ties_at_k(cfg, pos, alive)
    assert tied_k > 0 and tied_bound > 0, "the generators stopped producing ties"


@gpu
def test_r_squared_constant_is_the_once_rounded_product() -> None:
    """``d² < float32(R·R)`` with the double product rounded once, as MLX casts the Python scalar. For a
    non-dyadic R the constant differs by an ulp from float32(R)·float32(R) about half the time, and a pair
    whose d² falls between the two is in the MLX search and not in a kernel that squares the float32 radius.
    3-decimal radii, 3D and 2D, each with such a pair (found per radius)."""
    rng = np.random.default_rng(7)
    done = 0
    for R in np.round(rng.uniform(2.0, 9.0, 400), 3):
        sc = _straddle(float(R))
        if sc is None:
            continue
        nd = 3 if done % 2 else 2
        L = float(np.round(R * 13 + 0.3, 3))
        cfg = _cfg3(L=L, H=24.0, R=float(R)) if nd == 3 else _cfg2(L=L, R=float(R))
        pts = [[sc[0], 40.0, 10.0][:nd], [sc[1], 40.0, 10.0][:nd]]
        pos, alive = _pad(np.array(pts, np.float32), 1024, nd)
        ref = _assert_same(cfg, pos, alive)
        assert _pairs_between_rr(cfg, pos, alive) == 2  # the two ordered pairs of the one pair
        inside = _d2_matrix(cfg, pos, alive)[0, 1] < np.float32(float(R) * float(R))
        assert int(np.array(ref.valid).sum()) == 2 * int(
            inside
        )  # in both lists or in neither, by the double product
        done += 1
        if done == 24:
            break
    assert done == 24


@gpu
def test_out_of_contract_inputs_bitwise() -> None:
    """Alive rows outside the engine's invariants (unwrapped x or y, z out of range, NaN or inf, K = Q,
    rows permuted): the two searches agree on all of them, so the kernel is not a second definition
    of the contract."""
    rng = np.random.default_rng(11)
    n = 2048
    for nd, preset in ((3, "tiny3d"), (2, "tiny")):
        over = dict(
            population__n_max=n,
            population__n_init=8,
            population__max_births_per_step=8,
            sensing__cell_capacity=16,
        )
        cfg = _cfg(preset, **over)
        d, L = dims(cfg), cfg.world.size
        base = rng.random((n, nd)) * L
        if nd == 3:
            base[:, 2] = np.clip(base[:, 2] * (cfg.world.height / L), d.z_lo, d.z_hi)
        alive = np.ones(n, bool)
        p = base.copy()
        sel = rng.random(n) < 0.15
        p[sel, 0] = rng.choice([L, -1e-8, -0.5, L + 0.5, 2 * L - 1e-3, -L + 1e-3], size=sel.sum())
        sel = rng.random(n) < 0.15
        p[sel, 1] = rng.choice([L, -1e-8, -0.5, L + 0.5], size=sel.sum())
        _assert_same(cfg, p, alive)
        if nd == 3:
            p = base.copy()
            sel = rng.random(n) < 0.3
            h = cfg.world.height
            p[sel, 2] = rng.choice([-0.3, -1e-6, 0.0, h, h + 0.7, d.z_hi + 1e-3], size=sel.sum())
            _assert_same(cfg, p, alive)
            p = base.copy()
            p[:, 2] = 0.0  # every agent on the floor plane
            _assert_same(cfg, p, alive)
        p = base.copy()
        p[rng.random(n) < 0.05, 0] = np.nan
        p[rng.random(n) < 0.05, 1] = np.inf
        _assert_same(cfg, p, alive)
        cfg_kq = _cfg(preset, **{**over, "sensing__cell_capacity": 2, "sensing__k_neighbors": 3**nd * 2})
        _assert_same(cfg_kq, base, alive)
        # a real state with its rows permuted
        c = _cfg(preset, **{**over, "population__n_init": 256})
        s = init_state(c)
        perm = rng.permutation(c.population.n_max)
        _assert_same(c, to_numpy(s.pos)[perm], to_numpy(s.alive)[perm])


@gpu
def test_kernel_is_run_to_run_deterministic() -> None:
    cfg, pos, alive = _scene_lattice(3, 16, 8)()
    p, a = mx.array(pos), mx.array(alive)
    first = None
    for _ in range(20):
        out = spatial.find_neighbors(p, a, cfg, impl="metal")
        mx.eval(out)
        raw = [np.array(x).tobytes() for x in out]
        first = first or raw
        assert raw == first


@gpu
def test_metal_argpartition_is_stable() -> None:
    """The fact the kernel's tie order stands on (F16, F21): on Metal, ``argpartition`` is a stable merge
    sort, so the step's top-K pattern equals the stable top-K and equal distances keep the candidate
    order. A new MLX that changes this fails here first."""
    rng = np.random.default_rng(0)
    for q in (9, 27, 144, 432, 1296, 4200):
        a = rng.integers(0, 4, size=(512, q)).astype(np.float32)
        a[a == 3] = np.inf
        k = 8
        x = mx.array(a)

        def pattern(x: mx.array, k: int = k) -> mx.array:
            part = mx.argpartition(x, kth=k - 1, axis=1)[:, :k].astype(mx.int32)
            o = mx.argsort(mx.take_along_axis(x, part, 1), axis=1).astype(mx.int32)
            return mx.take_along_axis(part, o, 1)

        want = np.argsort(a, axis=1, kind="stable")[:, :k]
        for fn in (pattern, mx.compile(pattern)):
            assert np.array_equal(np.array(fn(x)), want), q


@gpu
def test_min_image_formula_matches_ops_on_sampled_floats() -> None:
    """The kernel's ``d − L·rint(d/L)`` against ``ops.min_image`` through the kernel itself: agents placed
    so that their x differences sweep ± L (the exhaustive sweep of every float32 is the slow test
    below)."""
    cfg = _cfg3(n=4096)
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 128, 4096).astype(np.float32)
    pos = np.stack([x, np.full(4096, 64.0, np.float32), np.full(4096, 12.0, np.float32)], 1)
    pos[:, 1] += rng.uniform(-1e-3, 1e-3, 4096).astype(np.float32)
    _assert_same(cfg, pos, np.ones(4096, bool))


@gpu
@pytest.mark.parametrize("L", [48.0, 128.0, 160.0, 320.0, 1024.0])
def test_min_image_exhaustive(L: float) -> None:
    """Every float32 ``d`` in (−L, L): the kernel's minimum-image statements (one op per statement under
    ``fp contract(off)``) equal ``ops.min_image``, compiled and eager, bitwise. The statements are cut out of
    ``knn_metal._SRC`` itself and run in a probe kernel built here (tests may call ``mx.fast.metal_kernel``;
    ``src`` may not)."""
    import re

    from alife import ops

    m = re.search(r"float ax = [^\n]*\n(.*?)const float dx = ax - mx_;", knn_metal._SRC, re.S)
    assert m and "rint" in m[1] and "__LF__" in m[1], "the minimum-image statements moved in knn_metal._SRC"
    # the y statements are the x statements with the letters swapped (so the x probe covers the y axis too)
    my = re.search(r"float ay = [^\n]*\n(.*?)const float dy = ay - my_;", knn_metal._SRC, re.S)
    assert my and "rint" in my[1], "the y minimum-image statements moved in knn_metal._SRC"
    swapped = my[1].replace("qy", "qx").replace("my_", "mx_").replace("ay", "ax")
    assert swapped == m[1], "the y minimum image is not the x one with the letters swapped"
    pragma = "#pragma METAL fp contract(off)" if "#pragma METAL fp contract(off)" in knn_metal._SRC else ""
    src = (
        f"{pragma}\nuint i = thread_position_in_grid.x;\nfloat ax = D[i];\n{m[1]}out[i] = ax - mx_;\n"
    ).replace("__LF__", knn_metal._f32_lit(L))
    kern = mx.fast.metal_kernel(
        name=f"mi_exh_{zlib.crc32(src.encode())}", input_names=["D"], output_names=["out"], source=src
    )
    comp = mx.compile(lambda d: ops.min_image(d, L))
    top = int(
        np.array([L], np.float32).view(np.uint32)[0]
    )  # the bit patterns [0, top) are the floats in [0, L)
    chunk, tested = 1 << 26, 0
    for sign in (0, 0x80000000):
        for lo in range(0, top, chunk):
            hi = min(top, lo + chunk)
            d = (mx.arange(lo, hi, dtype=mx.uint32) + mx.array(sign, dtype=mx.uint32)).view(mx.float32)
            ko = kern(
                inputs=[d],
                grid=(hi - lo, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(hi - lo,)],
                output_dtypes=[mx.float32],
            )[0]
            bad = (ko.view(mx.uint32) != comp(d).view(mx.uint32)).sum() + (
                ko.view(mx.uint32) != ops.min_image(d, L).view(mx.uint32)
            ).sum()
            assert int(bad.item()) == 0, (L, lo)
            tested += hi - lo
    assert tested > 2 * 10**9


# --- the kernel inside the compiled step ----------------------------------------------------------


def _lattice_start(cfg: Config, spacing: float) -> Any:
    """``init_state`` with its alive agents on a lattice of ``spacing`` at rest: exact distance ties from
    step 0."""
    s = init_state(cfg)
    d = dims(cfg)
    idx = np.flatnonzero(to_numpy(s.alive))
    pos = to_numpy(s.pos).copy()
    side = int(np.ceil(len(idx) ** (1 / d.nd)))
    grid = np.stack(np.meshgrid(*[np.arange(side)] * d.nd, indexing="ij"), -1).reshape(-1, d.nd)[: len(idx)]
    off = np.array([cfg.world.size / 2] * 2 + ([d.z_lo + 0.5] if d.nd == 3 else []), np.float32)[: d.nd]
    pos[idx] = grid.astype(np.float32) * np.float32(spacing) + off
    if d.nd == 3:
        pos[idx, 2] = np.clip(pos[idx, 2], d.z_lo, d.z_hi)
    return s._replace(pos=mx.array(pos), vel=mx.zeros_like(s.vel))


def _spy_scan_select(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    real = knn_metal.scan_select

    def spy(*args: Any, **kw: Any) -> Any:
        calls.append(1)
        return real(*args, **kw)

    monkeypatch.setattr(knn_metal, "scan_select", spy)
    return calls


@gpu
@pytest.mark.parametrize(
    ("preset", "feeds", "lattice"),
    [
        ("tiny3d", "mlx", 1.5),
        ("tiny3d", "metal", 1.5),
        ("tiny3d", "mlx", 0.0),
        ("tiny", "mlx", 1.5),
        ("tiny", "metal", 1.5),  # 2D is under auto too (I52): the kernel feeds the step
    ],
)
def test_search_inside_the_compiled_step_bitwise(
    monkeypatch: pytest.MonkeyPatch, preset: str, feeds: str, lattice: float
) -> None:
    """F4: the real ``step.step`` is traced with ``find_neighbors`` replaced by a wrapper that runs both
    searches on the step's own S2 inputs (so the inputs are the step's real ones), feeds one of them
    to the rest of the step and returns both as extra outputs of the same compiled function. Every
    field is compared bitwise every step; a second, kernel-only compiled step runs in lockstep from
    the same state and its digest is compared every 10 steps. The lattice start puts exact ties
    inside the real step."""
    cfg = load_config(preset=preset, overrides=["brain.weight_dtype=float32"])
    s0 = _lattice_start(cfg, lattice) if lattice else init_state(cfg)
    monkeypatch.setenv("ALIFE_KNN", "metal")
    orig = spatial.find_neighbors
    captured: list[tuple[spatial.Neighbors, spatial.Neighbors]] = []

    def wrapper(pos: mx.array, alive: mx.array, c: Config, **kw: Any) -> spatial.Neighbors:
        ref, ker = orig(pos, alive, c, impl="mlx"), orig(pos, alive, c, impl="metal")
        captured.append((ref, ker))
        return ref if feeds == "mlx" else ker

    def dual(s: Any) -> Any:
        captured.clear()
        s2, out = step_mod.step(s, cfg)
        ref, ker = captured[0]
        return s2, out, tuple(ref), tuple(ker)

    fdual = mx.compile(dual)
    monkeypatch.setattr(spatial, "find_neighbors", wrapper)
    sa, outa, ref, ker = fdual(s0)  # traced here, under the wrapper
    monkeypatch.setattr(spatial, "find_neighbors", orig)
    calls = _spy_scan_select(monkeypatch)
    fker = step_mod.make_step(cfg)
    sb = s0
    tied = 0
    for i in range(1, 61):
        if i > 1:
            sa, outa, ref, ker = fdual(sa)
        sb, outb = fker(sb)
        mx.eval(sa, outa, ref, ker, sb, outb)
        r, k = spatial.Neighbors(*ref), spatial.Neighbors(*ker)
        assert not any(_diff(k, r).values()), (i, _diff(k, r))
        tied += _ties_in_k(r)
        assert np.array_equal(to_numpy(outa.counters), to_numpy(outb.counters)), i
        assert np.array_equal(to_numpy(outa.ledger), to_numpy(outb.ledger)), i
        if i % 10 == 0:
            assert state_digest(sa) == state_digest(sb), i
    assert calls, "the kernel-only step did not trace the Metal kernel"
    if lattice:
        assert tied > 0, "the lattice start produced no distance ties"


@gpu
@pytest.mark.parametrize("preset", ["tiny", "tiny3d", "micro3d"])
def test_compiled_step_bitwise(monkeypatch: pytest.MonkeyPatch, preset: str) -> None:
    """Two fresh ``make_step`` closures, one traced with ``ALIFE_KNN=metal`` and one with ``mlx``, 60 steps
    from ``init_state``: equal counters and ledgers every step, equal ``state_digest`` every 10. The
    metal run really traced the kernel (a spy on ``scan_select``) and the mlx run did not."""
    if preset == "micro3d":
        cfg = load_config(preset="tiny3d", overrides=list(MICRO3D))
    else:
        cfg = load_config(preset=preset, overrides=["brain.weight_dtype=float32"])
    calls = _spy_scan_select(monkeypatch)
    runs = {}
    for which in ("metal", "mlx"):
        monkeypatch.setenv("ALIFE_KNN", which)
        calls.clear()
        fn, s = step_mod.make_step(cfg), init_state(cfg)
        trace = []
        for i in range(1, 61):
            s, out = fn(s)
            mx.eval(s, out)
            trace.append((to_numpy(out.counters).tobytes(), to_numpy(out.ledger).tobytes()))
            if i % 10 == 0:
                trace.append(state_digest(s))
        runs[which] = (trace, len(calls))
    assert runs["metal"][1] > 0 and runs["mlx"][1] == 0
    assert runs["metal"][0] == runs["mlx"][0]


@gpu
@pytest.mark.parametrize("preset", ["tiny", "tiny3d"])
def test_auto_traces_the_kernel_in_2d_and_3d(monkeypatch: pytest.MonkeyPatch, preset: str) -> None:
    """``auto`` (the default) traces the kernel in both dimensions (I52), ``ALIFE_KNN=mlx`` neither, and the
    provenance records say which (``meta.json`` and the bench environment)."""
    from alife.bench import _environment

    cfg = load_config(preset=preset, overrides=["brain.weight_dtype=float32"])
    calls = _spy_scan_select(monkeypatch)
    for env, expect in ((None, True), ("auto", True), ("mlx", False)):
        if env is None:
            monkeypatch.delenv("ALIFE_KNN", raising=False)
        else:
            monkeypatch.setenv("ALIFE_KNN", env)
        calls.clear()
        step_mod.make_step(cfg)(init_state(cfg))
        assert bool(calls) is expect, (preset, env)
        assert _environment(cfg)["knn_impl"] == ("metal" if expect else "mlx"), (preset, env)
        assert spatial.prepare_knn(cfg) == ("metal" if expect else "mlx"), (preset, env)


@gpu
@pytest.mark.slow
def test_64k3d_step_bitwise(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``64k3d`` preset from ``init_state`` (C = 48), 200 compiled steps with each search: equal
    counters, ledgers and (every 50 steps) ``state_digest``."""
    cfg = load_config(preset="64k3d")
    runs = {}
    for which in ("metal", "mlx"):
        monkeypatch.setenv("ALIFE_KNN", which)
        fn, s = step_mod.make_step(cfg), init_state(cfg)
        trace = []
        for i in range(1, 201):
            s, out = fn(s)
            mx.eval(s, out)
            trace.append((to_numpy(out.counters).tobytes(), to_numpy(out.ledger).tobytes()))
            if i % 50 == 0:
                trace.append(state_digest(s))
        runs[which] = trace
        del fn, s
    assert runs["metal"] == runs["mlx"]


# --- the guards (version set, self-check, K bound) ------------------------------------------------


@gpu
@pytest.mark.parametrize("preset", ["tiny", "tiny3d"])
def test_unverified_mlx_version_falls_back(monkeypatch: pytest.MonkeyPatch, preset: str) -> None:
    cfg = load_config(preset=preset)
    assert knn_metal.available() and spatial.knn_impl(cfg) == "metal"
    monkeypatch.setattr(knn_metal, "VERIFIED_MLX", frozenset({"0.0.0"}))
    assert not knn_metal.available()
    assert spatial.knn_impl(cfg) == "mlx" and spatial.knn_impl(cfg, "auto") == "mlx"
    with pytest.raises(RuntimeError, match="verified MLX"):
        spatial.knn_impl(cfg, "metal")


@gpu
def test_self_check_passes_and_is_memoised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(knn_metal, "_CHECKED", frozenset())
    calls = _spy_scan_select(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert knn_metal.self_check(3) and knn_metal.self_check(2)
        n = len(calls)
        assert n == 2 and knn_metal.self_check(3) and len(calls) == n  # the second call runs nothing
    assert knn_metal.available() and not knn_metal._MISMATCH


@gpu
@pytest.mark.parametrize("preset", ["tiny", "tiny3d"])
@pytest.mark.parametrize("fault", ["wrong_values", "raises"])
def test_self_check_mismatch_disables_the_kernel(
    monkeypatch: pytest.MonkeyPatch, fault: str, preset: str
) -> None:
    cfg = load_config(preset=preset)
    monkeypatch.setattr(knn_metal, "_CHECKED", frozenset())
    monkeypatch.setattr(knn_metal, "_MISMATCH", False)
    real = knn_metal.scan_select

    def broken(*args: Any, **kw: Any) -> Any:
        if fault == "raises":
            raise ValueError("kernel build failed")
        idx, valid, off, dist = real(*args, **kw)
        return idx, valid, off, dist + 1e-6  # one ulp-scale difference in dist

    monkeypatch.setattr(knn_metal, "scan_select", broken)
    with pytest.warns(RuntimeWarning, match="falling back to the pure-MLX search"):
        assert spatial.prepare_knn(cfg) == "mlx"
    assert not knn_metal.available()
    assert spatial.knn_impl(cfg) == "mlx"
    with pytest.raises(RuntimeError):
        spatial.knn_impl(cfg, "metal")  # an explicit request is an error, not a silent fallback
    monkeypatch.setattr(knn_metal, "scan_select", real)  # and the run still works, on the MLX path
    fn, s = step_mod.make_step(cfg), init_state(cfg)
    fn(s)


def test_self_check_scene_has_ties_and_overflow() -> None:
    """The scene the startup self-check compares is not trivial (CPU: the reference alone): exact ties,
    overfull cells, pairs at, just inside and just outside R, and dead rows both NaN and finite (a finite dead
    row on top of a live agent is what a kernel that ignored ``alive`` would list)."""
    for nd in (3, 2):
        cfg, pos, alive = knn_metal._check_scene(nd)
        assert pos.shape == (2048, nd) and pos.dtype == np.float32 and alive.dtype == bool
        ref = spatial.find_neighbors(mx.array(pos), mx.array(alive), cfg, impl="mlx")
        mx.eval(ref)
        assert _ties_in_k(ref) > 100 and int(ref.overflow_cells.item()) > 0
        assert int(np.array(ref.valid).sum()) > 2000 and not np.array(ref.valid)[~alive].any()
        assert _pairs_at_r(cfg, pos, alive) >= 2 * nd  # exactly R, along every axis
        d2 = _d2_matrix(cfg, pos, alive)
        r2 = np.float32(cfg.sensing.radius**2)
        assert (d2 < r2).any(axis=1).sum() > 100 and ((d2 > r2) & (d2 < r2 * np.float32(1.0001))).any()
        dead = pos[~alive]
        assert np.isnan(dead).all(axis=1).any() and np.isfinite(dead).all(axis=1).any()


# --- selection rules (CPU) ------------------------------------------------------------------------


@pytest.fixture
def pretend_metal(monkeypatch: pytest.MonkeyPatch) -> None:
    """The selection logic as if on a verified Metal GPU (no kernel is built or run)."""
    monkeypatch.setattr(knn_metal, "_capable", lambda: True)
    monkeypatch.setattr(spatial, "_on_gpu", lambda: True)
    monkeypatch.setattr(knn_metal, "_MISMATCH", False)
    monkeypatch.delenv("ALIFE_KNN", raising=False)


def test_selection_rules(pretend_metal: None, monkeypatch: pytest.MonkeyPatch) -> None:
    c3, c2 = load_config(preset="tiny3d"), load_config(preset="tiny")
    assert knn_metal.available()
    assert spatial.knn_impl(c3) == "metal"  # auto: the kernel in 3D
    assert spatial.knn_impl(c2) == "metal"  # and in 2D (the owner's decision of 2026-10-21, I52)
    assert spatial.knn_impl(c2, "metal") == "metal" and spatial.knn_impl(c3, "metal") == "metal"
    assert spatial.knn_impl(c3, "mlx") == "mlx" and spatial.knn_impl(c2, "mlx") == "mlx"
    monkeypatch.setenv("ALIFE_KNN", "mlx")
    assert spatial.knn_impl(c3) == "mlx" and spatial.knn_impl(c2) == "mlx"  # forces the reference path
    assert spatial.knn_impl(c3, "metal") == "metal"  # impl overrides the environment
    assert spatial.knn_impl(c2, "metal") == "metal"
    monkeypatch.setenv("ALIFE_KNN", "metal")
    assert spatial.knn_impl(c2) == "metal" and spatial.knn_impl(c3, "mlx") == "mlx"
    assert spatial.knn_impl(c2, "mlx") == "mlx"
    monkeypatch.setenv("ALIFE_KNN", "auto")
    assert spatial.knn_impl(c3) == "metal" and spatial.knn_impl(c2) == "metal"


@pytest.mark.parametrize("nd", [2, 3])
def test_k_bound(pretend_metal: None, nd: int) -> None:
    mk = _cfg3 if nd == 3 else _cfg2
    ok = mk(C=48, K=knn_metal.K_MAX)
    big = mk(C=48, K=knn_metal.K_MAX + 1)
    assert spatial.knn_impl(ok) == "metal"
    assert spatial.knn_impl(big) == "mlx"  # auto falls back above the bound
    assert spatial.knn_impl(big, "mlx") == "mlx"
    with pytest.raises(RuntimeError, match="k_neighbors <= 64"):
        spatial.knn_impl(big, "metal")


def test_unverified_mlx_says_so_once(pretend_metal: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Metal is there but this MLX is not in ``VERIFIED_MLX``: ``auto`` takes the MLX path (about 5 times
    slower in 3D, 1.5 times in 2D) and ``prepare_knn`` says so, once, in either dimension. It stays silent
    where the kernel was not going to be used anyway."""
    monkeypatch.setattr(knn_metal, "VERIFIED_MLX", frozenset({"0.0.0"}))
    monkeypatch.setattr(knn_metal, "_NOTED", False)
    c3, c2 = load_config(preset="tiny3d"), load_config(preset="tiny")
    quiet: list[Callable[[], str]] = [
        lambda: spatial.prepare_knn(_cfg3(C=48, K=knn_metal.K_MAX + 1)),  # K above the bound
        lambda: spatial.prepare_knn(_cfg2(C=16, K=knn_metal.K_MAX + 1)),  # and in 2D
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for call in quiet:
            assert call() == "mlx"
        monkeypatch.setenv("ALIFE_KNN", "mlx")
        assert spatial.prepare_knn(c3) == "mlx" and spatial.prepare_knn(c2) == "mlx"  # asked for
        monkeypatch.delenv("ALIFE_KNN")
        monkeypatch.setattr(spatial, "_on_gpu", lambda: False)
        assert spatial.prepare_knn(c3) == "mlx" and spatial.prepare_knn(c2) == "mlx"  # the CPU is the default
        monkeypatch.setattr(spatial, "_on_gpu", lambda: True)
        monkeypatch.setattr(knn_metal, "_capable", lambda: False)
        assert spatial.prepare_knn(c3) == "mlx" and spatial.prepare_knn(c2) == "mlx"  # no Metal at all
        monkeypatch.setattr(knn_metal, "_capable", lambda: True)
    assert not knn_metal._NOTED
    for first, later in ((c2, c3), (c3, c2)):  # either dimension warns, and once per process for both
        monkeypatch.setattr(knn_metal, "_NOTED", False)
        with pytest.warns(
            RuntimeWarning, match=r"MLX .* not in knn_metal.VERIFIED_MLX.*ALIFE_KNN=mlx"
        ) as rec:
            assert spatial.prepare_knn(first) == "mlx"
        assert len(rec) == 1 and rec[0].filename == __file__  # attributed to the caller of prepare_knn
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert spatial.prepare_knn(later) == "mlx" and spatial.prepare_knn(first) == "mlx"  # once
    monkeypatch.setenv("ALIFE_KNN", "metal")
    for c in (c3, c2):
        with pytest.raises(RuntimeError, match="verified MLX"):
            spatial.prepare_knn(c)  # asked for: an error, not a warning


def test_invalid_selection_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(preset="tiny3d")
    with pytest.raises(ValueError, match="ALIFE_KNN"):
        spatial.knn_impl(cfg, "gpu")
    monkeypatch.setenv("ALIFE_KNN", "fast")
    with pytest.raises(ValueError, match="'fast'"):
        spatial.knn_impl(cfg)
    with pytest.raises(ValueError):
        spatial.find_neighbors(mx.zeros((1024, 3)), mx.zeros(1024, dtype=mx.bool_), cfg)


def test_cpu_default_device_selects_the_mlx_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(knn_metal, "_capable", lambda: True)
    monkeypatch.setattr(spatial, "_on_gpu", lambda: False)
    monkeypatch.delenv("ALIFE_KNN", raising=False)
    for preset in ("tiny3d", "tiny"):
        cfg = load_config(preset=preset)
        assert spatial.knn_impl(cfg) == "mlx" and spatial.prepare_knn(cfg) == "mlx"
        with pytest.raises(RuntimeError):
            spatial.knn_impl(cfg, "metal")  # ALIFE_KNN=metal on the CPU raises
    monkeypatch.setattr(knn_metal, "_capable", lambda: False)
    monkeypatch.setattr(spatial, "_on_gpu", lambda: True)
    assert spatial.knn_impl(cfg) == "mlx"  # no Metal backend


def test_available_is_a_bool_and_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(knn_metal.available(), bool)
    monkeypatch.setattr(knn_metal, "_capable", lambda: True)
    assert knn_metal.available()  # the autouse fixture lists the running version
    monkeypatch.setattr(knn_metal, "VERIFIED_MLX", frozenset())
    assert knn_metal.available() is False  # an unlisted MLX version
    monkeypatch.setattr(knn_metal, "VERIFIED_MLX", frozenset({mx.__version__}))
    monkeypatch.setattr(knn_metal, "_MISMATCH", True)
    assert knn_metal.available() is False  # a failed self-check
    assert frozenset() != knn_metal.VERIFIED_MLX


def test_shipped_verified_versions() -> None:
    """The set the code ships: the pinned MLX range's first release, the one these tests were run on."""
    import tomllib

    pin = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    assert any("mlx>=0.32.3,<0.33" in d for d in pin["project"]["dependencies"])
    assert "0.32.3" in knn_metal.VERIFIED_MLX
    assert knn_metal.K_MAX == 64


def test_prepare_knn_runs_the_self_check_once_per_use(monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[int] = []
    monkeypatch.setattr(knn_metal, "_capable", lambda: True)
    monkeypatch.setattr(spatial, "_on_gpu", lambda: True)
    monkeypatch.setattr(knn_metal, "_MISMATCH", False)
    monkeypatch.setattr(knn_metal, "self_check", lambda nd=3: ran.append(nd) or True)
    monkeypatch.delenv("ALIFE_KNN", raising=False)
    assert spatial.prepare_knn(load_config(preset="tiny3d")) == "metal" and ran == [3]
    assert spatial.prepare_knn(load_config(preset="tiny")) == "metal" and ran == [3, 2]  # 2D, auto: checked
    monkeypatch.setenv("ALIFE_KNN", "mlx")
    assert spatial.prepare_knn(load_config(preset="tiny3d")) == "mlx" and ran == [3, 2]
    assert spatial.prepare_knn(load_config(preset="tiny")) == "mlx" and ran == [3, 2]  # forced: no check
    monkeypatch.setenv("ALIFE_KNN", "metal")
    assert spatial.prepare_knn(load_config(preset="tiny")) == "metal" and ran == [3, 2, 2]


def test_self_check_returns_false_where_the_kernel_cannot_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(spatial, "_on_gpu", lambda: False)
    monkeypatch.setattr(knn_metal, "_CHECKED", frozenset())
    assert knn_metal.self_check(3) is False and frozenset() == knn_metal._CHECKED


def test_scan_select_checks_its_inputs() -> None:
    """The kernel does no bounds checking, so ``scan_select`` raises ``ValueError`` (and not ``assert``, which
    ``python -O`` removes) on any shape, dtype or size the kernel would index out of bounds with. The checks
    run before anything is built, so this needs no GPU."""
    for nd in (3, 2):
        n = 64
        cfg = _cfg3(n=n) if nd == 3 else _cfg2(n=n)
        d = dims(cfg)
        args: dict[str, Any] = dict(
            spos=mx.zeros((n, nd)),
            start=mx.zeros(d.g_hash**2 * d.g_hash_z + 1, dtype=mx.int32),
            order=mx.arange(n, dtype=mx.int32),
            cxy=mx.zeros((n, 2), dtype=mx.int32),
            cz=mx.zeros(n if nd == 3 else 1, dtype=mx.int32),
            alive=mx.ones(n, dtype=mx.bool_),
        )
        static = dict(nd=nd, g=d.g_hash, gz=d.g_hash_z, c=d.c, k=d.k, n=n)

        def check(**over: Any) -> None:
            knn_metal._validate(**{**args, **static, **over})  # noqa: B023 (called within the iteration)

        check()  # the consistent arguments pass
        bad: list[tuple[str, dict[str, Any]]] = [
            ("bool alive", dict(alive=mx.ones(n, dtype=mx.int32))),
            ("float32 positions", dict(spos=mx.zeros((n, nd), dtype=mx.float16))),
            ("not", dict(spos=mx.zeros((n - 1, nd)))),  # a short array would be read out of bounds
            ("not", dict(spos=mx.zeros((n, 5 - nd)))),  # the other dimensionality's data
            ("alive", dict(alive=mx.ones(n + 1, dtype=mx.bool_))),
            ("order", dict(order=mx.arange(n - 1, dtype=mx.int32))),
            ("order", dict(order=mx.arange(n, dtype=mx.float32))),  # a float order would be read as bits
            ("start", dict(start=mx.zeros(10, dtype=mx.int32))),  # a start of the wrong length
            ("start", dict(start=args["start"].astype(mx.float32))),
            ("cxy", dict(cxy=mx.zeros((5, 2), dtype=mx.int32))),
            ("cxy", dict(cxy=mx.zeros((n, 3), dtype=mx.int32))),
            ("cxy", dict(cxy=mx.zeros((n, 2)))),
            ("cz", dict(cz=mx.zeros(n + 3 if nd == 3 else n, dtype=mx.int32))),  # (N,) in 3D, (1,) in 2D
            ("cz", dict(cz=mx.zeros(n if nd == 3 else 1))),  # a float layer index
            ("K in", dict(k=knn_metal.K_MAX + 1)),
            ("K in", dict(k=0)),
            ("nd must", dict(nd=4)),
            ("positive", dict(n=0)),
            ("positive", dict(c=0)),
            ("overflow", dict(n=17_000_000, k=64)),  # n·K·nd >= 2**31 in 2D and 3D, checked before any shape
        ]
        for match, over in bad:
            with pytest.raises(ValueError, match=match):
                check(**over)
    cfg = _cfg3(n=64)
    d = dims(cfg)
    ok = dict(
        spos=mx.zeros((64, 3)),
        start=mx.zeros(d.g_hash**2 * d.g_hash_z + 1, dtype=mx.int32),
        order=mx.arange(64, dtype=mx.int32),
        cxy=mx.zeros((64, 2), dtype=mx.int32),
        cz=mx.zeros(64, dtype=mx.int32),
        alive=mx.ones(64, dtype=mx.bool_),
    )
    with pytest.raises(
        ValueError, match="bool alive"
    ):  # the public function raises too, before building a kernel
        knn_metal.scan_select(
            **{**ok, "alive": mx.ones(64, dtype=mx.int32)},
            nd=3,
            g=d.g_hash,
            gz=d.g_hash_z,
            c=d.c,
            k=d.k,
            n=64,
            L=128.0,
            R=6.0,
        )


def test_kernel_compile_options_are_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    """``math_mode: safe`` is explicit: ``fast`` multiplies by the reciprocal in ``d / L`` and parts from MLX
    at a difference of one ulp from L/2 (``test_crafted_scenes_bitwise[2d_recip_L31.373]`` sees it on the
    GPU; this pins the option on any machine). Also the pragma, the header and the row-contiguous inputs."""
    seen: dict[str, Any] = {}

    def spy(**kw: Any) -> Any:
        seen.update(kw)
        return lambda **_: None

    monkeypatch.setattr(mx.fast, "metal_kernel", spy, raising=False)
    knn_metal._kernel.cache_clear()
    try:
        knn_metal._kernel(128.0, 6.0, 36.0)
    finally:
        knn_metal._kernel.cache_clear()
    assert seen["compile_options"] == {"math_mode": "safe"}
    assert seen["ensure_row_contiguous"] is True and seen["header"] == knn_metal._HEADER
    assert seen["source"].count("#pragma METAL fp contract(off)") == 1
    assert "fast::" not in seen["source"] and "metal::fast" not in seen["source"]


def test_hex_literals_are_the_float32_values() -> None:
    for x in (320.0, 128.0, 6.0, 5.3, 0.1, 6.000001, 4.123456789):
        lit = knn_metal._f32_lit(x)
        assert lit.endswith("f") and float.fromhex(lit[:-1]) == float(np.float32(x))
    r = 5.3
    assert float.fromhex(knn_metal._f32_lit(r * r)[:-1]) == float(np.float32(r * r))  # R·R rounded once


def test_the_comparison_sees_what_it_must() -> None:
    """The helper that every GPU test relies on: it counts a sign of zero, a one-ulp change, a NaN against a
    number and a dtype change, and accepts NaN against the same NaN."""
    z = mx.zeros((4, 8), dtype=mx.float32)
    base = spatial.Neighbors(
        mx.zeros((4, 8), dtype=mx.int32),
        mx.ones((4, 8), dtype=mx.bool_),
        mx.zeros((4, 8, 3)),
        z,
        mx.array(0, dtype=mx.int32),
        mx.array(1, dtype=mx.int32),
    )
    assert not any(_diff(base, base).values())
    neg0 = base._replace(
        dist=mx.array(np.where(np.arange(8) == 0, -0.0, 0.0).astype(np.float32) * np.ones((4, 1), np.float32))
    )
    assert _diff(neg0, base)["dist"] == 4
    one_ulp = np.zeros((4, 8), np.float32)
    one_ulp[1, 2] = np.nextafter(np.float32(0), np.float32(1))
    assert _diff(base._replace(dist=mx.array(one_ulp)), base)["dist"] == 1
    nan = base._replace(dist=mx.full((4, 8), float("nan")))
    assert _diff(nan, nan)["dist"] == 0 and _diff(nan, base)["dist"] == 32
    assert _diff(base._replace(idx=base.idx.astype(mx.int64)), base)["idx"] == -1


# --- provenance and the CLI -----------------------------------------------------------------------


def test_meta_and_bench_record_the_selected_search(tmp_path: Path) -> None:
    from alife.bench import _environment

    for preset in ("tiny", "tiny3d"):
        cfg = load_config(preset=preset, overrides=["brain.weight_dtype=float32"])
        want = spatial.knn_impl(cfg)
        assert RunDir.create(tmp_path / preset, cfg).meta()["knn_impl"] == want
        assert _environment(cfg)["knn_impl"] == want
    assert spatial.knn_impl(load_config(preset="tiny"), "mlx") == "mlx"


def test_cli_refuses_an_unusable_selection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``ALIFE_KNN=metal`` where the kernel cannot run, and an unknown value, are config errors (exit 2)
    that leave no run directory behind."""
    args = ["run", "--preset", "tiny", "--set", "brain.weight_dtype=float32", "--steps", "1", "--quiet"]
    monkeypatch.setenv("ALIFE_KNN", "bogus")
    assert cli.main([*args, "--out", str(tmp_path / "a")]) == 2 and not (tmp_path / "a").exists()
    if not ON_GPU:
        monkeypatch.setenv("ALIFE_KNN", "metal")
        assert cli.main([*args, "--out", str(tmp_path / "b")]) == 2 and not (tmp_path / "b").exists()
    monkeypatch.setenv("ALIFE_KNN", "mlx")
    assert cli.main([*args, "--out", str(tmp_path / "c")]) == 0
    assert RunDir.open(tmp_path / "c").meta()["knn_impl"] == "mlx"
