"""Helpers of the sphere fields tests (sphere series S1): device fixture, independent numpy mirrors.

``mirror_diffuse`` is the integer diffusion written as an explicit list of directed pairs with Python-int
semantics (numpy int64), independent of the gather form of ``ops.int_diffuse_sph``; ``torus_tables``
builds the neighbour tables of a periodic box so that the table-driven kernel can be compared with the
box stencil.
"""

from __future__ import annotations

from collections.abc import Iterator

import mlx.core as mx
import numpy as np
import pytest

DEVICES = [mx.cpu, *([mx.gpu] if mx.metal.is_available() else [])]


@pytest.fixture(params=DEVICES, ids=lambda d: str(d).split(".")[-1].split(",")[0].strip(")"))
def dev(request) -> Iterator[object]:
    """Run the test body on this device (a stream context), whatever the process default is."""
    with mx.stream(request.param):
        yield request.param


def mirror_diffuse(r: np.ndarray, nbr: np.ndarray, a_t: np.ndarray, a_z: np.ndarray) -> np.ndarray:
    """The integer flux diffusion by an explicit list of directed pairs ``(i -> j, a_ij)``, each with the
    coefficient of the *sending* cell's own table (so the symmetry of the tables is exercised, not assumed):
    ``q = (R_i·a_ij) >> 16`` out of ``i`` into ``j``, accumulated with ``bincount`` (a scatter, where the
    device kernel gathers). ``r`` (FZ, Npix) int, returns int64."""
    fz, npix = r.shape
    r64 = r.astype(np.int64).reshape(-1)
    cell = np.arange(fz * npix).reshape(fz, npix)
    src, dst, a = [], [], []
    for j in range(4):  # tangential: (k, p) -> (k, nbr[p, j])
        src.append(cell.reshape(-1))
        dst.append((np.arange(fz)[:, None] * npix + nbr[None, :, j]).reshape(-1))
        a.append(a_t[:, :, j].astype(np.int64).reshape(-1))
    if fz > 1:  # radial: (k, p) -> (k+1, p) with a_z[k], and (k+1, p) -> (k, p) with the same a_z[k]
        az = np.broadcast_to(np.asarray(a_z, np.int64).reshape(-1, 1), (fz - 1, npix))
        src.append(cell[:-1].reshape(-1))
        dst.append(cell[1:].reshape(-1))
        a.append(az.reshape(-1))
        src.append(cell[1:].reshape(-1))
        dst.append(cell[:-1].reshape(-1))
        a.append(az.reshape(-1))
    src, dst, a = np.concatenate(src), np.concatenate(dst), np.concatenate(a)
    q = (r64[src] * a) >> 16
    out = np.zeros(r64.size, np.int64)
    inn = np.zeros(r64.size, np.int64)
    np.add.at(out, src, q)
    np.add.at(inn, dst, q)
    return (r64 - out + inn).reshape(fz, npix)


def torus_tables(fg: int, fz: int, d_q8: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The neighbour table ``(x+1, x-1, y+1, y-1)`` of an ``fg x fg`` torus and the coefficients ``256·d_q8``
    (the weight 256 of the box) for ``fz`` layers: ``(nbr (fg², 4), a_t (fz, fg², 4), a_z (fz-1, 1))``."""
    cy, cx = np.divmod(np.arange(fg * fg), fg)
    nbr = np.stack(
        [
            cy * fg + (cx + 1) % fg,
            cy * fg + (cx - 1) % fg,
            ((cy + 1) % fg) * fg + cx,
            ((cy - 1) % fg) * fg + cx,
        ],
        axis=1,
    ).astype(np.int32)
    a_t = np.full((fz, fg * fg, 4), 256 * d_q8, np.int32)
    a_z = np.full((max(fz - 1, 0), 1), 256 * d_q8, np.int32)
    return nbr, a_t, a_z
