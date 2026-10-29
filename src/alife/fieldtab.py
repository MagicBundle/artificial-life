"""Host tables of the sphere world's fields (numpy, float64 then integer; DESIGN §46.5; sphere series S1).

Everything the field kernels of ``alife.fields_sph`` need that is not a function of the state:

* the **conductance tables** of the shell's finite-volume cells: the tangential pair weight across each of the
  4 edges of an angular cell (``edge length · thickness / centre distance``, per layer) and the radial pair
  weight between two layers (``face area / layer spacing``), all divided by the (equal) cell volume and
  normalised so that the mean tangential weight of the floor layer is 256, then rounded to integers in 1/256
  units. They are **symmetric**: the pair weight is one number per undirected pair, which is what makes the
  integer flux antisymmetric and the total exactly conserved;
* the row sums (``Σ_j w_ij``) that bound the stable diffusion constants: ``d_q8`` for the integer resource and
  the number of sub-steps of the explicit chemical update;
* the vertical food profile at the layers' lower interface heights;
* the gradient tables: the tangent least-squares weights of the 4 edge neighbours (per unit length at the
  layer's radius) and the radial spacings with the box's mirrored boundaries.

The device twins (MLX) are in ``alife.fields_sph``; ``ops.int_diffuse_sph`` and ``ops.check_diffusion_tables``
are the integer kernel and its exhaustive table check. No function here is traced.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from alife import geom

if TYPE_CHECKING:
    from alife.config import Config

SUM_LIMIT = 0.9  # the validated bound of Σ_j α_ij, α_ij = d_q8·w_ij / 2^16 (DESIGN §46.5)
Q16 = 1 << 16


@dataclass(frozen=True)
class Conductance:
    """The pair weights of one shell (all arrays read-only).

    ``wt[k, p, j]`` is the weight (1/256 units) of the pair (cell ``(k, p)``, its edge neighbour ``nbr[p, j]``
    in layer ``k``); ``wt[k, p, j] == wt[k, nbr[p, j], back[p, j]]``. ``wz[k]`` is the weight of the pair of
    layers ``k`` and ``k+1`` at every pixel. ``scale`` is the factor that turned a conductance per unit
    volume into 1/256 units (the mean tangential conductance of the floor layer is ``256 / scale``).
    """

    nside: int
    layers: int
    nbr: np.ndarray  # (Npix, 4) int32: the edge neighbours
    back: np.ndarray  # (Npix, 4) int32: the slot of p in nbr[nbr[p, j]]
    wt: np.ndarray  # (FZ, Npix, 4) int32
    wz: np.ndarray  # (FZ-1,) int32
    thickness: np.ndarray  # (FZ,) float64
    cell_volume: float
    scale: float

    @property
    def rowsum(self) -> np.ndarray:
        """(FZ, Npix) int64: the sum of the (up to 6) pair weights of each cell."""
        s = self.wt.sum(axis=2, dtype=np.int64)
        up = np.concatenate([self.wz, [0]]).astype(np.int64)  # layer k <-> k+1; none above the top layer
        dn = np.concatenate([[0], self.wz]).astype(np.int64)
        return s + (up + dn)[:, None]

    @property
    def rowsum_max(self) -> int:
        return int(self.rowsum.max())

    def d_q8_max(self, limit: float = SUM_LIMIT) -> int:
        """The largest ``d_q8`` with ``Σ_j α_ij <= limit`` in every cell: ``floor(limit·2^16/rowsum_max)``."""
        return math.floor(limit * Q16 / self.rowsum_max + 1e-12)

    def chem_substeps(self, diffusion_max: float) -> int:
        """Sub-steps of the explicit chemical update: ``ceil(D·max_i Σ_j C_ij)`` with ``C_ij = w_ij/256`` (at
        least 1), so that every sub-step keeps all coefficients of the update non-negative."""
        return max(1, math.ceil(diffusion_max * self.rowsum_max / 256.0 - 1e-9))


def shell(cfg: Config) -> tuple[float, float, int, int]:
    """``(Rp, H, Nside, FZ)`` of a sphere config: the arguments of ``conductance`` and ``gradient``."""
    w = cfg.world
    return w.radius, w.height, w.nside, w.layers


def _corner_vectors(nside: int) -> np.ndarray:
    """(Npix, 4, 3) unit vectors of the corners of each pixel (corner k joins edge k-1 and edge k)."""
    ix, iy, face = geom.pix2xyf(nside, np.arange(geom.npix(nside)))
    out = np.empty((len(ix), 4, 3))
    for k, (dx, dy) in enumerate(geom.CORNERS):
        out[:, k] = geom.xyf2vec((ix + dx) / nside, (iy + dy) / nside, face)
    return out


def _angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), (a * b).sum(-1))


@functools.cache
def edge_ratio(nside: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(nbr, back, ratio)`` of an Nside: the edge neighbours (Npix, 4), the slot of the pixel in its
    neighbour's table, and the ratio ``L/D`` of the shared edge length to the distance between the two
    centres (both unit-sphere angles), made exactly symmetric (the minimum of the two directed values)."""
    tb = geom.tables(nside)
    nbr = tb.edge
    n = tb.npix
    back = np.empty_like(nbr)
    for j in range(4):  # the slot of p in the table of q = nbr[:, j]
        hit = nbr[nbr[:, j]] == np.arange(n)[:, None]
        if not (hit.sum(axis=1) == 1).all():
            raise AssertionError("an edge neighbour is not mutual")
        back[:, j] = hit.argmax(axis=1)
    cor = _corner_vectors(nside)
    length = np.stack([_angle(cor[:, j], cor[:, (j + 1) % 4]) for j in range(4)], axis=1)
    cen = geom.pixel_centres(nside)
    dist = np.stack([_angle(cen, cen[nbr[:, j]]) for j in range(4)], axis=1)
    ratio = length / dist
    ratio = np.minimum(ratio, np.take_along_axis(ratio[nbr], back[..., None], axis=2)[..., 0])
    # nbr[p, j] = q, ratio[q][back[p, j]] is the reverse directed value
    for a in (nbr, back, ratio):
        a.flags.writeable = False
    return nbr, back, ratio


@functools.cache
def conductance(radius: float, height: float, nside: int, layers: int) -> Conductance:
    """The pair weights of the shell ``(Rp, H, Nside, FZ)`` (module docstring). Layers are the equal-volume
    layers of ``geom.layer_radii``; the tangential conductance of layer ``k`` is ``(L/D)·t_k / V`` with
    ``t_k`` the thickness (the ratio ``L/D`` of the unit sphere does not depend on the radius), the radial
    one at the interface ``r_{k+1}`` is ``(4π/Npix)·r_{k+1}² / (rc_{k+1} − rc_k) / V`` with ``rc`` the volume
    midpoint radius of a layer."""
    nbr, back, ratio = edge_ratio(nside)
    n = geom.npix(nside)
    r = geom.layer_radii(radius, height, layers)
    t = np.diff(r)
    vol = geom.layer_volume(radius, height, layers, nside)
    ct = ratio[None, :, :] * t[:, None, None] / vol  # (FZ, Npix, 4)
    rc = np.cbrt(0.5 * (r[:-1] ** 3 + r[1:] ** 3))
    cz = (4.0 * math.pi / n) * r[1:-1] ** 2 / np.diff(rc) / vol  # (FZ-1,)
    scale = 256.0 / ct[0].mean()
    wt = np.rint(ct * scale).astype(np.int32)
    wz = np.rint(cz * scale).astype(np.int32)
    if wt.min() < 1 or (wz.size and wz.min() < 1):
        raise AssertionError("a pair weight rounds to zero")
    back_w = np.take_along_axis(wt[:, nbr, :], back[None, :, :, None], axis=3)[..., 0]
    if not np.array_equal(back_w, wt):
        raise AssertionError("the tangential pair weights are not symmetric")
    for a in (wt, wz, t):
        a.flags.writeable = False
    return Conductance(
        nside=nside,
        layers=layers,
        nbr=nbr,
        back=back,
        wt=wt,
        wz=wz,
        thickness=t,
        cell_volume=vol,
        scale=scale,
    )


def vertical_profile(
    radius: float, height: float, layers: int, benthic_lambda: float, photic_amp: float, photic_lambda: float
) -> np.ndarray:
    """(FZ,) the food profile ``V`` at the lower interface heights ``z_k`` (DESIGN §46.5): ``min(1,
    exp(−z_k/λ_b) + photic_amp·exp(−(z_{FZ−1} − z_k)/λ_p))``, the photic term only when ``photic_amp > 0``.
    ``V[0] = 1`` exactly (``z_0 = 0``), as in the box (where ``z_k = k·field_cell``)."""
    z = geom.layer_edges(radius, height, layers)[:-1]
    v = np.exp(-z / benthic_lambda)
    if photic_amp > 0:
        v = v + photic_amp * np.exp(-(z[-1] - z) / photic_lambda)
    return np.minimum(1.0, v)


@dataclass(frozen=True)
class Gradient:
    """Tables of the field gradient (DESIGN §46.5), all float64 (cast on the device)."""

    kind: str
    w: np.ndarray  # (Npix, k, 3) tangent weights per radian (geom.lsq_weights)
    inv_r: np.ndarray  # (FZ,) 1 / (volume-midpoint radius of the layer): per radian -> per unit length
    dz2: np.ndarray  # (FZ,) the radial divisor z_{k+1} − z_{k−1} of layer centres, mirrored at floor and lid
    zc: np.ndarray  # (FZ,) heights of the layer centres


GRADIENT_KIND = "edge"  # the 4 edge neighbours: DESIGN §46.5 (choice recorded there)


@functools.cache
def gradient(radius: float, height: float, nside: int, layers: int, kind: str = GRADIENT_KIND) -> Gradient:
    """The gradient tables of the shell. The radial divisor of layer ``k`` is ``zc_{k+1} − zc_{k−1}`` with the
    box's mirrored boundary: a missing neighbour layer is the layer itself, and the divisor is the one of the
    mirrored interior stencil (``2·(zc_1 − zc_0)`` at the floor), so a boundary layer sees half its one-sided
    slope, as the box's divisor 2 does (DESIGN §27.6)."""
    r = geom.layer_radii(radius, height, layers)
    rc = np.cbrt(0.5 * (r[:-1] ** 3 + r[1:] ** 3))
    zc = rc - radius
    ext = np.concatenate([[2 * zc[0] - zc[1]], zc, [2 * zc[-1] - zc[-2]]]) if layers > 1 else None
    dz2 = (ext[2:] - ext[:-2]) if ext is not None else np.full(layers, np.inf)
    w = geom.lsq_weights(nside, kind)
    w.flags.writeable = False
    return Gradient(kind=kind, w=w, inv_r=1.0 / rc, dz2=dz2, zc=zc)
