"""The fused Metal neighbour search (§4 "Fused Metal path", §27.2, F21, I48).

One ``mx.fast.metal_kernel`` runs the candidate scan, the strict ``R²`` mask and the K-best selection
of ``spatial.find_neighbors`` and writes the (N, K) outputs: one thread per sorted rank, which writes
only the row ``order[r]`` (a permutation), so there is no contested write and no atomic. It returns
the same ``idx``, ``valid``, ``off`` and ``dist``, bit for bit, as the pure-MLX search on the Metal GPU
of a verified MLX (``VERIFIED_MLX``):

- a stable insertion by ``(d², q)`` with ``q = t·C + j`` (stencil cell, then slot in the run), which
  is the order of Metal's stable ``ArgPartition`` (F16, F21);
- one float operation per statement under ``fp contract(off)``: no fused multiply-add (F21);
- ``d − L·rint(d/L)`` for the minimum image, ``d² < float32(R·R)``, ``metal::precise::sqrt`` (F21).

This is the only module that calls ``mx.fast.metal_kernel`` (lint rule 6). It imports only
``mlx.core`` and the standard library at module level, so ``spatial`` can import it. The pure-MLX
path stays the reference, the CPU path and the fallback; ``spatial.knn_impl`` selects between them
when the step is traced.
"""

from __future__ import annotations

import hashlib
import struct
import warnings
from functools import lru_cache
from typing import Any

import mlx.core as mx

# The MLX versions on which the kernel's bitwise identity was measured (tests/test_spatial_metal.py,
# DESIGN F21). The identity rests on MLX internals (a stable Metal sort, no a*b+c contraction in MLX's
# own kernels), so a new MLX is added here by the change that re-runs those tests, and not before.
VERIFIED_MLX: frozenset[str] = frozenset({"0.32.3"})
K_MAX = 64  # the kernel keeps 5 K-element private arrays per thread: above this it loses to the MLX path (§4)
_THREADGROUP = (
    64  # 32 to 256 were within ±3% in the probe (docs/bench/README.md: not reproducible from the repo)
)

_I32 = mx.int32
_MISMATCH = False  # set by ``self_check``: the kernel disagreed with the MLX path on this machine
_CHECKED: frozenset[int] = frozenset()  # the world dimensionalities ``self_check`` has run for
_NOTED = False  # ``note_unverified`` has warned in this process


def _capable() -> bool:
    """This MLX build has a usable Metal backend and ``mx.fast.metal_kernel`` (capability flags only)."""
    return bool(getattr(mx, "metal", None) and mx.metal.is_available() and hasattr(mx.fast, "metal_kernel"))


def available() -> bool:
    """The Metal kernel can run here and is verified for this MLX: Metal, ``mx.fast.metal_kernel``, a
    version in ``VERIFIED_MLX`` and no ``self_check`` mismatch. Reads only capability flags, never an
    array. Whether the *default device* is the GPU is the caller's check (``spatial.knn_impl``)."""
    return bool(not _MISMATCH and _capable() and mx.__version__ in VERIFIED_MLX)


def unverified_mlx() -> bool:
    """Metal and ``mx.fast.metal_kernel`` are there but this MLX version is not in ``VERIFIED_MLX``: the
    kernel is off and, under ``auto``, the pure-MLX search runs (5 times slower at 64k3d, 1.5 at 64k)."""
    return _capable() and mx.__version__ not in VERIFIED_MLX


def note_unverified() -> bool:
    """Warn once per process (``RuntimeWarning``) that the kernel is off because this MLX is unverified.
    True if it warned. Host side; ``spatial.prepare_knn`` calls it when ``auto`` would take the kernel."""
    global _NOTED
    if _NOTED or not unverified_mlx():
        return False
    _NOTED = True
    warnings.warn(
        f"MLX {mx.__version__} is not in knn_metal.VERIFIED_MLX {sorted(VERIFIED_MLX)}: the fused Metal "
        "neighbour search is off and the pure-MLX search runs (about 5 times slower in 3D, 1.5 times in 2D). "
        "Re-run tests/test_spatial_metal.py on this MLX and list it, or set ALIFE_KNN=mlx to silence this",
        RuntimeWarning,
        stacklevel=3,
    )
    return True


def _f32_lit(x: float) -> str:
    """An exact Metal float literal for float32(x): the hex float of the rounded value."""
    f = struct.unpack("<f", struct.pack("<f", x))[0]
    return float(f).hex() + "f"


_HEADER = """
inline int py_mod(int a, int g) { int r = a % g; if (r != 0 && ((r < 0) != (g < 0))) { r += g; } return r; }
"""

# __LF__ = float32(L), __RRF__ = float32(R·R) (the float64 product rounded once, as MLX casts the
# Python scalar in ``d2 < R * R``) and __RF__ = float32(R) are exact hex literals, filled in by
# ``_kernel``.
_SRC = """
#pragma METAL fp contract(off)
const int r = int(thread_position_in_grid.x);
if (r >= NT) { return; }
const int row = order[r];
float bd[K]; int bs[K]; float bx[K]; float by[K]; float bz[K];
for (int k = 0; k < K; ++k) { bd[k] = INFINITY; bs[k] = 0; bx[k] = 0.0f; by[k] = 0.0f; bz[k] = 0.0f; }
if (alive[row]) {
  const float px = spos[ND * r + 0];
  const float py = spos[ND * r + 1];
  const float pz = (ND == 3) ? spos[ND * r + 2] : 0.0f;
  const int cx = cxy[2 * row + 0];
  const int cy = cxy[2 * row + 1];
  const int czr = (ND == 3) ? cz[row] : 0;
  const int NS = (ND == 3) ? 27 : 9;
  for (int t = 0; t < NS; ++t) {                      // _DX27/_DY27/_DZ27 (2D: _DX9/_DY9) order
    const int ddx = (t % 3) - 1;
    const int ddy = ((t / 3) % 3) - 1;
    const int ddz = (ND == 3) ? (t / 9) - 1 : 0;
    const int ncz = czr + ddz;
    if (ND == 3 && (ncz < 0 || ncz >= GZ)) { continue; }   // an out-of-range layer has an empty run
    const int ncx = py_mod(cx + ddx, G);
    const int ncy = py_mod(cy + ddy, G);
    const int ncid = (ND == 3) ? ((ncz * G + ncy) * G + ncx) : (ncy * G + ncx);
    const int s = start[ncid];
    const int e = min(start[ncid + 1], s + C);             // the first C entries of the run
    for (int sl = s; sl < e; ++sl) {                      // q = t*C + (sl - s), increasing
      if (sl == r) { continue; }                          // slot != inv[i]
      float ax = spos[ND * sl + 0] - px;                  // min_image: d - L*rint(d/L), one op per statement
      float qx = ax / __LF__;
      qx = metal::rint(qx);
      float mx_ = __LF__ * qx;
      const float dx = ax - mx_;
      float ay = spos[ND * sl + 1] - py;
      float qy = ay / __LF__;
      qy = metal::rint(qy);
      float my_ = __LF__ * qy;
      const float dy = ay - my_;
      float d2;
      float dz = 0.0f;
      const float xx = dx * dx;
      const float yy = dy * dy;
      d2 = xx + yy;                                       // 2D: (off*off).sum(-1) of two terms
      if (ND == 3) {
        dz = spos[ND * sl + 2] - pz;                      // no min image in z (2*height <= L)
        const float zz = dz * dz;
        d2 = d2 + zz;                                     // (dx*dx + dy*dy) + dz*dz, left to right
      }
      if (!(d2 < __RRF__)) { continue; }                  // the strict R^2 mask
      if (!(d2 < bd[K - 1])) { continue; }                // a tie with the K-th loses: it has the larger q
      int p = K - 1;
      while (p > 0 && d2 < bd[p - 1]) {                   // strict: an equal earlier candidate stays ahead
        bd[p] = bd[p - 1]; bs[p] = bs[p - 1]; bx[p] = bx[p - 1]; by[p] = by[p - 1]; bz[p] = bz[p - 1];
        --p;
      }
      bd[p] = d2; bs[p] = sl; bx[p] = dx; by[p] = dy; bz[p] = dz;
    }
  }
}
for (int k = 0; k < K; ++k) {
  const bool v = bd[k] < INFINITY;
  const int o = row * K + k;
  idx[o] = v ? order[bs[k]] : row;                        // invalid: the agent's own index
  valid[o] = v;
  off[ND * o + 0] = v ? bx[k] : 0.0f;
  off[ND * o + 1] = v ? by[k] : 0.0f;
  if (ND == 3) { off[ND * o + 2] = v ? bz[k] : 0.0f; }
  dist[o] = v ? metal::precise::sqrt(bd[k]) : __RF__;    // MLX's Sqrt is metal::precise::sqrt
}
"""


@lru_cache(maxsize=32)
def _kernel(L: float, R: float, rr: float) -> Any:
    """The compiled kernel for a world size and radius (their float32 values are literals in the source). The
    name carries a hash of the source, so the library cache never depends on MLX's own keying."""
    src = _SRC.replace("__LF__", _f32_lit(L)).replace("__RRF__", _f32_lit(rr)).replace("__RF__", _f32_lit(R))
    assert "__" not in src, "unfilled placeholder in the Metal source"
    name = "alife_knn_" + hashlib.sha1((_HEADER + src).encode()).hexdigest()[:12]
    return mx.fast.metal_kernel(
        name=name,
        input_names=["spos", "start", "order", "cxy", "cz", "alive"],
        output_names=["idx", "valid", "off", "dist"],
        source=src,
        header=_HEADER,
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def _validate(
    spos: mx.array,
    start: mx.array,
    order: mx.array,
    cxy: mx.array,
    cz: mx.array,
    alive: mx.array,
    *,
    nd: int,
    g: int,
    gz: int,
    c: int,
    k: int,
    n: int,
) -> None:
    """The kernel does no bounds checking, so every size and dtype it indexes with is checked here, with
    ``ValueError`` (not ``assert``: ``python -O`` removes asserts and the kernel would read out of bounds)."""

    def need(ok: bool, what: str) -> None:
        if not ok:
            raise ValueError(f"the Metal neighbour search: {what}")

    need(nd in (2, 3), f"nd must be 2 or 3, got {nd}")
    need(min(g, gz, c, n) >= 1, f"g, gz, c and n must be positive, got {(g, gz, c, n)}")
    need(1 <= k <= K_MAX, f"supports K in [1, {K_MAX}], got {k}")
    need(n * k * nd < 2**31 and g * g * gz < 2**31 - 1, "the sizes overflow the kernel's int32 indexing")
    need(alive.dtype == mx.bool_, "needs a bool alive mask")
    need(spos.dtype == mx.float32, "needs float32 positions")
    need(spos.shape == (n, nd), f"spos {spos.shape} is not (N={n}, nd={nd})")
    need(alive.shape == (n,), f"alive {alive.shape} is not (N={n},)")
    need(
        order.shape == (n,) and order.dtype == _I32,
        f"order must be int32 ({n},), got {order.dtype} {order.shape}",
    )
    need(
        start.shape == (g * g * gz + 1,) and start.dtype == _I32,
        f"start must be int32 ({g * g * gz + 1},), got {start.dtype} {start.shape}",
    )
    need(
        cxy.shape == (n, 2) and cxy.dtype == _I32, f"cxy must be int32 ({n}, 2), got {cxy.dtype} {cxy.shape}"
    )
    cz_shape = (n,) if nd == 3 else (1,)
    need(cz.shape == cz_shape and cz.dtype == _I32, f"cz must be int32 {cz_shape}, got {cz.dtype} {cz.shape}")


def scan_select(
    spos: mx.array,
    start: mx.array,
    order: mx.array,
    cxy: mx.array,
    cz: mx.array,
    alive: mx.array,
    *,
    nd: int,
    g: int,
    gz: int,
    c: int,
    k: int,
    n: int,
    L: float,
    R: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """``(idx, valid, off, dist)`` of ``spatial.find_neighbors``, from the sorted data of its preamble.

    ``spos`` (N, nd) float32 is ``pos[order]``; ``start`` (G²·G_z + 1,) the run starts of the sorted
    hash cells; ``order`` (N,) the sort permutation (dead rows last); ``cxy`` (N, 2) int32 the x/y
    cells; ``cz`` (N,) int32 the clamped z layer (a (1,) dummy in 2D); ``alive`` (N,) bool. Static:
    every size is a template constant. The shapes, dtypes and sizes are checked (``ValueError``) because the
    kernel does no bounds checking.
    """
    _validate(spos, start, order, cxy, cz, alive, nd=nd, g=g, gz=gz, c=c, k=k, n=n)
    kern = _kernel(float(L), float(R), float(R * R))
    grid = -(-n // _THREADGROUP) * _THREADGROUP
    idx, valid, off, dist = kern(
        inputs=[spos, start, order, cxy, cz, alive],
        template=[("ND", nd), ("G", g), ("GZ", gz), ("C", c), ("K", k), ("NT", n)],
        grid=(grid, 1, 1),
        threadgroup=(_THREADGROUP, 1, 1),
        output_shapes=[(n, k), (n, k), (n, k, nd), (n, k)],
        output_dtypes=[_I32, mx.bool_, mx.float32, mx.float32],
    )
    return idx, valid, off, dist


# --- the startup self-check (host side: called by the CLI entry points before ``make_step``) ---------


def _check_scene(nd: int) -> tuple[Any, Any, Any]:
    """``(cfg, pos, alive)`` of a 2 048-agent scene with exact ties: stacked lattices (identical positions and
    equal distances, overfull cells), pairs at, just inside and just outside ``R`` along every axis, scattered
    agents, and dead rows (half with NaN positions, half with a finite one on top of a live agent)."""
    import numpy as np

    from alife.config import load_config

    n = 2048
    over = [
        f"population.n_max={n}",
        "population.n_init=8",
        "population.max_births_per_step=32",
        "sensing.cell_capacity=16",
        "sensing.k_neighbors=8",
        "brain.weight_dtype=float32",
    ]
    cfg = load_config(preset="tiny3d" if nd == 3 else "tiny", overrides=over)
    rng = np.random.default_rng(20261020)

    def lattice(side: int) -> Any:
        axis = np.arange(side, dtype=np.float64)
        return np.stack(np.meshgrid(*([axis] * nd), indexing="ij"), -1).reshape(-1, nd)

    stacked = np.repeat(lattice(6), 3, axis=0) * 1.0 + 20.0  # spacing 1, three agents per site
    sparse = lattice(8) * 3.0 + 70.0  # spacing 3, one agent per site
    R = float(cfg.sensing.radius)
    r32 = np.float32(R)
    gaps = [r32, np.nextafter(r32, np.float32(0)), np.nextafter(r32, np.float32(2 * R))]  # R, R-ulp, R+ulp
    edge: list[
        Any
    ] = []  # pairs exactly at, just inside and just outside R along every axis (1.5 + gap < 8: exact)
    for i, gap in enumerate(gaps):
        for ax in range(nd):
            o = [1.5 if j == ax else (4.5 if j == 2 else 1.5 + 3 * (nd * i + ax)) for j in range(nd)]
            edge += [o, [v + float(gap) if j == ax else v for j, v in enumerate(o)]]
    edge = np.asarray(edge, np.float64)
    scatter = rng.random((n - len(stacked) - len(sparse) - len(edge), nd)) * cfg.world.size
    pos = np.concatenate([stacked, sparse, edge, scatter])[:n]
    if nd == 3:
        z = np.clip(pos[:, 2] % cfg.world.height, 1.0, cfg.world.height - 1.0)
        pos = np.concatenate([pos[:, :2], z[:, None]], axis=1)
    alive = np.arange(n) % 17 != 0
    dead = ~alive
    # half of the dead rows keep a finite position, on a live agent's site (a kernel that ignored ``alive``
    # would list them); the other half are NaN
    pos = np.where(dead[:, None] & (np.arange(n)[:, None] % 2 == 0), np.nan, pos).astype(np.float32)
    return cfg, pos, alive


def self_check(nd: int = 3) -> bool:
    """Run the kernel and the pure-MLX search once on a scene with exact ties (eagerly, about 0.2 s
    with the first Metal compile) and compare every output bit for bit. True when the kernel is usable
    and matches.

    Where the kernel cannot run (CPU default device, no Metal, an MLX outside ``VERIFIED_MLX``) it
    returns False without running anything. On a mismatch, or if the kernel fails to build or run, it
    warns and disables the kernel for the process (``available()`` turns False), so the pure-MLX path
    is used. It runs at most once per process and ``nd``. Called by the CLI entry points before
    ``make_step``; a library caller that skips it gets the kernel unchecked (the verified-version
    guard still applies). Host side: it reads arrays.
    """
    global _MISMATCH, _CHECKED
    import numpy as np

    from alife import spatial  # not at module level: spatial imports this module

    if not available() or not spatial._on_gpu():
        return False
    if nd in _CHECKED:
        return True

    try:
        cfg, pos, alive = _check_scene(nd)
        p, a = mx.array(pos), mx.array(alive)
        ref = spatial.find_neighbors(p, a, cfg, impl="mlx")
        ker = spatial.find_neighbors(p, a, cfg, impl="metal")
        mx.eval(ref, ker)
        bad = [
            f
            for f in spatial.Neighbors._fields
            if getattr(ref, f).dtype != getattr(ker, f).dtype
            or getattr(ref, f).shape != getattr(ker, f).shape
            or np.array(getattr(ref, f)).tobytes() != np.array(getattr(ker, f)).tobytes()
        ]
        problem = f"differs from the MLX search in {bad}" if bad else ""
    except Exception as e:  # a kernel that cannot build or run is a reason to fall back
        problem = f"failed with {type(e).__name__}: {e}"
    _CHECKED = _CHECKED | {nd}
    if problem:
        _MISMATCH = True
        warnings.warn(
            f"the fused Metal neighbour search {problem} on this machine (MLX {mx.__version__}); "
            "falling back to the pure-MLX search",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True
