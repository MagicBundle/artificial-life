"""The compile-collision self-check of a sphere step (DESIGN §46.3, spec 10.1 fallback; F21 item 6).

MLX 0.32.3's compile cache can hand a compiled function another one's fused kernel when their scalar constants
print alike (``floor(x·c).astype(int32) % g`` with ``(c, g)`` = ``(12.0, 1)`` and ``(1.0, 21)``), and the CPU
kernel cache is machine-wide. The sphere's position-to-cell functions are exactly that pattern, with several
distinct constants per Nside, ``Rp``, ``FZ``, ``G_z`` and ``Npix``. Until the guard of the branch
``fix/mlx-compile-collision`` is merged, ``step.make_step`` calls ``require_compile_safe`` for a compiled
sphere step: it compiles the lookups with this configuration's own constants, runs them on a small fixed set
of points and compares them with the eager values; on a mismatch it raises (``RuntimeError``) instead of
running a step that would place agents in the wrong cells. This is a host-side module (it reads arrays) and
runs once per constant set and process.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from alife import sphere
from alife.config import Config, dims

_DONE: set[tuple[float, ...]] = set()
_N_RANDOM = 384


def _points(cfg: Config) -> np.ndarray:
    """A fixed, config-derived set of float32 positions in the shell: uniform points plus the poles, the
    equator and points a milliradian from the trivalent vertices (never exactly on a pixel boundary: a
    boundary tie may differ between a fused and an unfused float32 evaluation without any collision)."""
    d = dims(cfg)  # r_lo, r_hi: the float32 bounds of the shell
    rng = np.random.default_rng(20261030)
    z = rng.uniform(-1.0, 1.0, _N_RANDOM)
    lam = rng.uniform(0.0, 2.0 * math.pi, _N_RANDOM)
    st = np.sqrt(1.0 - z * z)
    u = np.stack([st * np.cos(lam), st * np.sin(lam), z], axis=1)
    k = np.arange(8)
    vz = np.where(k < 4, 2.0 / 3.0, -2.0 / 3.0)
    vl = (k % 4) * (math.pi / 2.0) + 1e-3
    vs = np.sqrt(1.0 - vz * vz)
    vert = np.stack([vs * np.cos(vl), vs * np.sin(vl), vz], axis=1)
    poles = np.array([[1e-4, 0.0, 1.0], [0.0, 1e-4, -1.0]])
    u = np.concatenate([u, vert, poles])
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    rho = np.cbrt(rng.uniform(float(d.r_lo) ** 3, float(d.r_hi) ** 3, len(u)))
    return (u * rho[:, None]).astype(np.float32)


def constants(cfg: Config) -> tuple[float, ...]:
    """The scalar constants the position-to-cell functions bake into a compiled step."""
    sp = sphere.geometry(cfg)
    base = (
        float(sp.radius),
        float(sp.height_max),
        float(sp.nside),
        float(sp.layers),
        float(sp.nside_h),
        float(sp.gz),
        float(sp.th),
    )
    out = (*base, float(cfg.weather.cells)) if cfg.weather.enabled else base  # the weather's Nside (S4b)
    if cfg.calls.far_range > 0.0:  # the column map of the far hearing (DESIGN §47.7): Ncol
        out = (*out, float(sp.npix))
    return out


def _lookups(cfg: Config):  # -> Callable[[mx.array], tuple[mx.array, ...]]
    sp = sphere.geometry(cfg)
    nw = cfg.weather.cells if cfg.weather.enabled else 0
    far = cfg.calls.far_range > 0.0

    def f(pos: mx.array) -> tuple[mx.array, ...]:
        alive = mx.ones((pos.shape[0],), dtype=mx.bool_)
        out = (
            sphere.ang2pix(pos, sp.nside),
            sphere.ang2pix(pos, sp.nside_h),
            sphere.layer_index(pos, sp.rp3, sp.dv, sp.layers),
            sphere.hash_layer(pos, sp.radius, sp.th, sp.gz),
            sphere.field_cell(pos, sp.rp3, sp.dv, sp.layers, sp.nside),
            sphere.hash_cell(pos, alive, sp.radius, sp.th, sp.gz, sp.nside_h),
        )
        # the weather's cell lookup (the wind at the agents and the fan's departure points, S4b)
        out = (*out, sphere.ang2pix(pos, nw)) if nw else out
        if far:  # the column map of the far hearing (its column count is a runtime array: nothing baked)
            from alife import calls

            cell = sphere.field_cell(pos, sp.rp3, sp.dv, sp.layers, sp.nside)
            out = (*out, calls.column_of(cell, cfg, calls.runtime(cfg)))
        return out

    return f


def require_compile_safe(cfg: Config) -> None:
    """Raise ``RuntimeError`` if the compiled position-to-cell lookups of ``cfg`` differ from the eager ones.

    Cheap (a few ms after the first compile) and memoised per constant set. A mismatch means the process's
    compile cache handed this configuration another one's kernel: run one configuration per process with its
    own ``TMPDIR`` (spec 10.1), or merge the guard of ``fix/mlx-compile-collision``."""
    key = constants(cfg)
    if key in _DONE:
        return
    pos = mx.array(_points(cfg))
    f = _lookups(cfg)
    want = f(pos)
    got = mx.compile(f)(pos)
    mx.eval(want, got)
    bad = [
        i
        for i, (a, b) in enumerate(zip(want, got, strict=True))
        if not np.array_equal(np.array(a), np.array(b))
    ]
    if bad:
        names = (
            "ang2pix(Nf)",
            "ang2pix(Nh)",
            "layer_index",
            "hash_layer",
            "field_cell",
            "hash_cell",
            "ang2pix(Nw)",
            *(("calls.column_of",) if cfg.calls.far_range > 0.0 else ()),
        )
        raise RuntimeError(
            "the compiled sphere lookups differ from the eager ones for this configuration "
            f"({', '.join(names[i] for i in bad)}): MLX's compile cache returned another configuration's "
            "kernel (F21 item 6, the compile collision). Run one sphere configuration per process, each "
            "with its own TMPDIR, or merge the guard of fix/mlx-compile-collision"
        )
    _DONE.add(key)


_DONE_CELLS: set[int] = set()


def require_cells_compile_safe(nsides: tuple[int, ...]) -> None:
    """The observer's version of ``require_compile_safe`` (DESIGN §46.7): the territory and footprint maps
    (``groups.sphere_cell`` at the social rules' Nsides) are compiled inside the observer's folds with their
    Nside as a constant, the same pattern as the step's position-to-cell lookups. Raise ``RuntimeError`` if
    the compiled lookup of any of ``nsides`` differs from the eager one on a fixed set of directions
    (memoised per Nside and process)."""
    todo = [n for n in sorted(set(nsides)) if n not in _DONE_CELLS]
    if not todo:
        return
    rng = np.random.default_rng(20261030)
    v = rng.normal(size=(512, 3))
    v = np.concatenate([v, [[1e-4, 0.0, 1.0], [0.0, 1e-4, -1.0], [1.0, 0.0, 0.0]]])
    pos = mx.array((v / np.linalg.norm(v, axis=1, keepdims=True) * 60.0).astype(np.float32))
    for n in todo:
        want = sphere.ang2pix(pos, n)
        got = mx.compile(lambda p, n=n: sphere.ang2pix(p, n))(pos)
        mx.eval(want, got)
        if not np.array_equal(np.array(want), np.array(got)):
            raise RuntimeError(
                f"the compiled territory lookup ang2pix(Nside {n}) differs from the eager one: MLX's compile "
                "cache returned another configuration's kernel (F21 item 6, the compile collision). Run one "
                "sphere configuration per process, each with its own TMPDIR, or merge the guard of "
                "fix/mlx-compile-collision"
            )
        _DONE_CELLS.add(n)
