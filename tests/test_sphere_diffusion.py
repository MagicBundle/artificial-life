"""The conductance tables and the diffusion of the sphere world's fields (``alife.fieldtab``,
``ops.int_diffuse_sph``, ``ops.check_diffusion_tables``, the chemical update of ``alife.fields_sph``;
DESIGN §46.5, sphere series S1).

The integer resource diffusion is exact: the tests compare the table-driven int64 kernel with an
independent numpy mirror (an explicit list of directed pairs) bit for bit on random, saturated and
near-``R_HARD`` fields, run the corner cases of the spec (a floor cell at ``R_HARD`` with empty
neighbours, one at 10^6 in the floor of ``tiny3s`` where an int32 product wraps, uniform ``R_HARD``),
check the cap bound against an exhaustive enumeration, and show that with the weight 256 the kernel is
the box stencil. The chemicals are checked against a float64 reference, for mass conservation, for the
need of the sub-steps, and by the spectral-harmonic gate that the spec restates for the geometric
Laplacian. Everything integer runs on the CPU and on Metal; the compiled test passes every table as a
runtime array (the compile-collision fallback of the spec, 10.1).
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from alife import fields_sph, fieldtab, geom, ops
from alife.config import R_HARD
from sphere_fields_helpers import dev, mirror_diffuse, torus_tables  # noqa: F401

# (name, Rp, H, Nf, FZ, the largest d_q8 with sum(alpha) <= 0.9, rowsum_max of the spec / of this build)
PRESETS = [
    ("tiny3s", 36.0, 16.0, 8, 8, 11, 5233),
    ("mini3s", 48.0, 24.0, 11, 12, 11, 5329),
    ("64k3s", 90.0, 40.0, 21, 18, 14, 4189),
]
IDS = [p[0] for p in PRESETS]


def _cond(name: str) -> fieldtab.Conductance:
    _, rp, h, ns, fz, _, _ = next(p for p in PRESETS if p[0] == name)
    return fieldtab.conductance(rp, h, ns, fz)


def _arrays(c: fieldtab.Conductance, d_q8: int):
    return (
        mx.array(c.nbr, dtype=mx.int32),
        mx.array(c.wt * d_q8, dtype=mx.int32),
        mx.array((c.wz * d_q8).reshape(-1, 1), dtype=mx.int32),
    )


def _diffuse(r: np.ndarray, c: fieldtab.Conductance, d_q8: int) -> np.ndarray:
    nbr, a_t, a_z = _arrays(c, d_q8)
    out = ops.int_diffuse_sph(mx.array(r, dtype=mx.int32), nbr, a_t, a_z)
    mx.eval(out)
    return np.array(out)


# --- the tables ------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", IDS)
def test_pair_weights_are_symmetric_positive_and_normalised(name: str) -> None:
    c = _cond(name)
    n = geom.npix(c.nside)
    assert c.wt.shape == (c.layers, n, 4) and c.wz.shape == (c.layers - 1,)
    assert c.wt.min() >= 1 and c.wz.min() >= 1
    # the tangential weight of a pair is one number: seen from either cell
    for k in (0, c.layers // 2, c.layers - 1):
        for j in range(4):
            q, jb = c.nbr[:, j], c.back[:, j]
            assert np.array_equal(c.wt[k, :, j], c.wt[k, q, jb])
    assert np.array_equal(c.nbr[c.nbr[:, 0], c.back[:, 0]], np.arange(n))  # mutual
    assert abs(c.wt[0].mean() - 256.0) < 0.5  # normalised: the mean tangential weight of the floor is 256
    # the cells are thinner than wide: vertical exchange beats horizontal (decision D13), strongest aloft
    assert c.wz[0] > c.wt[0].mean() and c.wz[-1] > 4 * c.wt[-1].mean()
    assert (np.diff(c.thickness) < 0).all()  # equal-volume layers get thinner upward


@pytest.mark.parametrize("name,rowsum", [(p[0], p[6]) for p in PRESETS], ids=IDS)
def test_row_sums_and_the_stable_diffusion_constants_match_the_spec(name: str, rowsum: int) -> None:
    c = _cond(name)
    assert abs(c.rowsum_max - rowsum) <= 2  # the spec's 5233 / 5329 / 4189 (integer rounding of the weights)
    want = next(p[5] for p in PRESETS if p[0] == name)
    assert c.d_q8_max() == want  # 11, 11, 14
    assert c.d_q8_max() * c.rowsum_max / 65536 <= 0.9 < (c.d_q8_max() + 1) * c.rowsum_max / 65536
    assert 4 * c.rowsum_max / 65536 <= 0.9  # the default d_q8 = 4 is far inside (sum alpha <= 0.33)


def test_the_64k3s_floor_weights_match_the_spec() -> None:
    c = _cond("64k3s")
    assert (c.wt[0].min(), c.wt[0].max()) == (149, 437)  # "tangential 149 to 437 at the floor"
    assert c.wz[0] == 510  # "radial 510 ... "
    assert 123 <= c.wt[-1].mean() <= 136  # "123 to 136 (mean at the lid)"


@pytest.mark.parametrize("name,want", [("tiny3s", 4), ("mini3s", 4), ("64k3s", 3)])
def test_chemical_substeps_from_the_row_sums(name: str, want: int) -> None:
    c = _cond(name)
    assert c.chem_substeps(0.16) == want  # the spec's table says 4 for 64k3s; ceil(0.16 * 16.4) is 3
    assert c.chem_substeps(0.0) == 1
    # every sub-step keeps all coefficients non-negative
    n = c.chem_substeps(0.25)
    assert 0.25 / n * c.rowsum_max / 256 <= 1.0 and 0.25 / (n - 1) * c.rowsum_max / 256 > 1.0


# --- the integer kernel against the mirror ---------------------------------------------------------


def _fields(c: fieldtab.Conductance, kind: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (c.layers, geom.npix(c.nside))
    if kind == "random":
        return rng.integers(0, R_HARD + 1, shape)
    if kind == "saturated":
        r = np.full(shape, R_HARD)
        r[rng.random(shape) < 0.05] = 0
        return r
    if kind == "near_cap":
        return R_HARD - rng.integers(0, 70, shape)
    if kind == "sparse":
        r = np.zeros(shape, np.int64)
        hit = rng.random(shape) < 0.01
        r[hit] = rng.integers(0, 1_000_000, hit.sum())
        return r
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["random", "saturated", "near_cap", "sparse"])
@pytest.mark.parametrize("name", IDS)
@pytest.mark.usefixtures("dev")
def test_the_kernel_equals_the_mirror_and_conserves_exactly(name: str, kind: str) -> None:
    c = _cond(name)
    d_max = c.d_q8_max()
    for d_q8, seed in ((1, 1), (4, 2), (d_max, 3)):
        r = _fields(c, kind, seed).astype(np.int64)
        got = _diffuse(r, c, d_q8)
        ref = mirror_diffuse(r, c.nbr, c.wt * d_q8, c.wz * d_q8)
        assert np.array_equal(got, ref)
        assert got.astype(np.int64).sum() == r.sum()  # exact conservation
        assert got.min() >= 0 and got.max() <= R_HARD  # the cap bound, on this field
        r2 = got.copy()  # a second step from the result
        assert np.array_equal(_diffuse(r2, c, d_q8), mirror_diffuse(r2, c.nbr, c.wt * d_q8, c.wz * d_q8))


def test_the_kernel_moves_content_between_neighbours_only() -> None:
    c = _cond("tiny3s")
    n = geom.npix(c.nside)
    r = np.zeros((c.layers, n), np.int64)
    k, p = 3, 100
    r[k, p] = 1_000_000
    out = _diffuse(r, c, 4)
    moved = np.argwhere(out != r)
    allowed = {(k, p), (k - 1, p), (k + 1, p), *((k, int(q)) for q in c.nbr[p])}
    assert {tuple(int(v) for v in m) for m in moved} <= allowed and len(moved) == 7
    for j in range(4):  # (R * a) >> 16 to each edge neighbour, the radial neighbours likewise
        assert out[k, c.nbr[p, j]] == (1_000_000 * 4 * int(c.wt[k, p, j])) >> 16
    assert out[k + 1, p] == (1_000_000 * 4 * int(c.wz[k])) >> 16
    assert out[k - 1, p] == (1_000_000 * 4 * int(c.wz[k - 1])) >> 16


# --- the corner tests of the spec ------------------------------------------------------------------


@pytest.mark.parametrize("name", IDS)
@pytest.mark.usefixtures("dev")
def test_a_floor_cell_at_r_hard_with_empty_neighbours(name: str) -> None:
    c = _cond(name)
    n = geom.npix(c.nside)
    d_q8 = c.d_q8_max()
    for pix in (0, n // 2, n - 1):
        r = np.zeros((c.layers, n), np.int64)
        r[0, pix] = R_HARD
        out = _diffuse(r, c, d_q8)
        assert out.astype(np.int64).sum() == R_HARD
        assert out.min() >= 0 and out.max() <= R_HARD
        sent = sum((R_HARD * d_q8 * int(c.wt[0, pix, j])) >> 16 for j in range(4))
        sent += (R_HARD * d_q8 * int(c.wz[0])) >> 16
        assert out[0, pix] == R_HARD - sent  # what stays is what did not leave
        assert np.array_equal(out, mirror_diffuse(r, c.nbr, c.wt * d_q8, c.wz * d_q8))


@pytest.mark.usefixtures("dev")
def test_a_corpse_of_1e6_on_the_floor_of_tiny3s_does_not_wrap_int32() -> None:
    c = _cond("tiny3s")
    n = geom.npix(c.nside)
    d_q8 = c.d_q8_max()
    r = np.zeros((c.layers, n), np.int64)
    r[0, 17] = 1_000_000
    # the box's int32 form of the product wraps for this cell: the radial weight alone is 846
    with np.errstate(over="ignore"):
        naive = np.int32(1_000_000) * np.int32(d_q8 * int(c.wz[0]))
    assert int(naive) != 1_000_000 * d_q8 * int(c.wz[0]) and 1_000_000 * d_q8 * int(c.wz[0]) > 2**31
    out = _diffuse(r, c, d_q8)
    assert np.array_equal(out, mirror_diffuse(r, c.nbr, c.wt * d_q8, c.wz * d_q8))
    assert out.min() >= 0 and out.astype(np.int64).sum() == 1_000_000
    assert out[1, 17] == (1_000_000 * d_q8 * int(c.wz[0])) >> 16 > 0  # the upward flux is the true one


@pytest.mark.parametrize("name", IDS)
@pytest.mark.usefixtures("dev")
def test_a_uniform_field_does_not_move(name: str) -> None:
    c = _cond(name)
    n = geom.npix(c.nside)
    for v in (R_HARD, 12345, 1, 0):
        r = np.full((c.layers, n), v, np.int64)
        assert np.array_equal(
            _diffuse(r, c, c.d_q8_max()), r
        )  # uniform quanta = uniform density: equilibrium


@pytest.mark.parametrize("name", IDS)
@pytest.mark.usefixtures("dev")
def test_a_single_deficit_next_to_r_hard_never_overshoots_the_cap(name: str) -> None:
    """The worst neighbourhood of the cap bound on the real tables: every cell at ``R_HARD`` except one that
    is ``delta`` below it. Floors step together in that window, but the result never exceeds ``R_HARD``."""
    c = _cond(name)
    n = geom.npix(c.nside)
    rs = c.rowsum
    cells = {tuple(int(v) for v in np.unravel_index(rs.argmax(), rs.shape)): 0}
    cells[tuple(int(v) for v in np.unravel_index(rs.argmin(), rs.shape))] = 0
    cells.update({(0, 5): 0, (c.layers - 1, n - 3): 0, (c.layers // 2, n // 3): 0})
    d_q8 = c.d_q8_max()
    for k, p in cells:
        base = np.full((c.layers, n), R_HARD, np.int64)
        batch = []
        for delta in range(0, 80):
            r = base.copy()
            r[k, p] = R_HARD - delta
            batch.append(_diffuse(r, c, d_q8))
        got = np.array([b[k, p] for b in batch])
        assert got.max() <= R_HARD and got.min() >= 0
        assert all(b.max() <= R_HARD for b in batch)


# --- the cap bound against the exhaustive enumeration ----------------------------------------------


@pytest.mark.parametrize("name", IDS)
def test_check_diffusion_tables_passes_at_the_largest_stable_constant(name: str) -> None:
    c = _cond(name)
    d_q8 = c.d_q8_max()
    res = ops.check_diffusion_tables(c.wt * d_q8, c.wz * d_q8, R_HARD)
    assert (
        res.min_slack == 0
    )  # the cap is met with equality (a uniform R_HARD field is a fixed point), never exceeded
    assert 0.8 < res.row_sum_max <= 0.9
    assert res.window >= 50 and res.rows > 100
    for d in (1, 4, d_q8 - 1):
        ops.check_diffusion_tables(c.wt * d, c.wz * d, R_HARD)


@pytest.mark.parametrize("name", IDS)
def test_the_exhaustive_check_agrees_with_an_independent_enumeration(name: str) -> None:
    """Python-int enumeration of R_i over a window wider than the proof needs, for a sample of the distinct
    rows of the tables, against the all-``R_HARD`` neighbourhood."""
    c = _cond(name)
    d_q8 = c.d_q8_max()
    up = np.concatenate([c.wz, [0]]) * d_q8
    dn = np.concatenate([[0], c.wz]) * d_q8
    rows = np.unique(
        np.concatenate(
            [
                (c.wt * d_q8).reshape(c.layers, -1, 4),
                np.broadcast_to(up[:, None, None], (c.layers, c.wt.shape[1], 1)),
                np.broadcast_to(dn[:, None, None], (c.layers, c.wt.shape[1], 1)),
            ],
            axis=2,
        ).reshape(-1, 6),
        axis=0,
    )
    pick = np.random.default_rng(7).choice(len(rows), size=min(250, len(rows)), replace=False)
    for row in rows[pick].tolist():
        inflow = sum((R_HARD * a) >> 16 for a in row)
        for delta in range(0, 140):
            ri = R_HARD - delta
            new = ri - sum((ri * a) >> 16 for a in row) + inflow
            assert 0 <= new <= R_HARD, (row, delta, new)


def test_a_row_sum_above_the_limit_is_refused_and_would_go_negative() -> None:
    c = _cond("tiny3s")
    d_q8 = c.d_q8_max() + 1  # one past the bound: sum(alpha) = 0.958
    with pytest.raises(ValueError, match=r"sums to 0\.95"):
        ops.check_diffusion_tables(c.wt * d_q8, c.wz * d_q8, R_HARD)
    # a row with sum(alpha) > 1: the kernel empties a cell below zero (the reviewer-style counterexample)
    a_t = np.full((2, 1, 4), 17000, np.int32)  # 4 * 17000 / 65536 = 1.04
    a_z = np.zeros((1, 1), np.int32)
    with pytest.raises(ValueError, match=r"sums to 1\.0"):
        ops.check_diffusion_tables(a_t, a_z.reshape(-1), R_HARD)
    out_only = (np.int64(1_000_000) * a_t[0, 0].astype(np.int64)) >> 16
    assert 1_000_000 - out_only.sum() < 0  # a cell would send away more than it holds


def test_the_enumeration_is_not_vacuous() -> None:
    """With a cap whose binary pattern does not align the floors (R_HARD = 2^24 - 1 does, which is why the
    real tables meet the cap with equality), a row that satisfies the sum rule can still overshoot inside the
    window: the check finds it and names the row."""
    a = np.array([10661, 10823, 10661, 1130, 15346])  # four tangential and one radial pair: 0.80 per step
    assert a.sum() / 65536 < 0.9
    with pytest.raises(ValueError, match="above R_HARD = 1000"):
        ops.check_diffusion_tables(np.tile(a[:4].reshape(1, 1, 4), (2, 1, 1)), a[4:5], r_hard=1000)


# --- the box stencil is the special case w = 256 ---------------------------------------------------


@pytest.mark.parametrize("d_q8", [1, 4, 17, 42])
@pytest.mark.usefixtures("dev")
def test_weight_256_on_a_torus_reproduces_the_box_stencil_bit_for_bit(d_q8: int) -> None:
    fg, fz = 8, 5
    nbr, a_t, a_z = torus_tables(fg, fz, d_q8)
    rng = np.random.default_rng(d_q8)
    for hi in (R_HARD, 1000, 100):
        r = rng.integers(0, hi + 1, (fz, fg, fg)).astype(np.int32)
        want = ops.int_diffuse_3d(mx.array(r), d_q8)
        got = ops.int_diffuse_sph(
            mx.array(r.reshape(fz, -1)), mx.array(nbr), mx.array(a_t), mx.array(a_z)
        ).reshape(fz, fg, fg)
        assert np.array_equal(np.array(got), np.array(want))
    one = rng.integers(0, R_HARD + 1, (fg, fg)).astype(np.int32)  # one layer: the 2D stencil
    n1, t1, z1 = torus_tables(fg, 1, d_q8)
    got1 = ops.int_diffuse_sph(mx.array(one.reshape(1, -1)), mx.array(n1), mx.array(t1), mx.array(z1))
    assert np.array_equal(np.array(got1).reshape(fg, fg), np.array(ops.int_diffuse(mx.array(one), d_q8)))


# --- equilibrium -----------------------------------------------------------------------------------


def test_diffusion_relaxes_to_uniform_quanta_and_keeps_the_total() -> None:
    """Equal volumes make uniform quanta the equilibrium: a random field with a 3 % spread relaxes to a field
    whose cells differ by a few quanta only, with every quantum conserved."""
    c = _cond("tiny3s")
    d_q8 = c.d_q8_max()
    nbr, a_t, a_z = _arrays(c, d_q8)
    rng = np.random.default_rng(11)
    mean = 5_000_000
    r0 = (mean * (1.0 + 0.03 * rng.standard_normal((c.layers, geom.npix(c.nside))))).astype(np.int32)

    def many(r, nbr, a_t, a_z):  # constants as runtime inputs (compile-collision fallback)
        for _ in range(50):
            r = ops.int_diffuse_sph(r, nbr, a_t, a_z)
        return r

    f = mx.compile(many)
    r = mx.array(r0)
    for _ in range(100):  # 5 000 steps
        r = f(r, nbr, a_t, a_z)
    mx.eval(r)
    out = np.array(r).astype(np.int64)
    assert out.sum() == r0.astype(np.int64).sum()
    assert np.abs(out - out.mean()).max() < 0.0005 * mean  # from a 3 % spread (a 100 fold reduction at least)
    assert np.abs(r0 - r0.mean()).max() > 0.05 * mean


# --- the spherical-harmonic gate (spec 3.1) --------------------------------------------------------


def _tangential_operator(c: fieldtab.Conductance, k: int) -> np.ndarray:
    n = geom.npix(c.nside)
    m = np.zeros((n, n))
    rows = np.arange(n)
    for j in range(4):
        w = c.wt[k, :, j] / 256.0
        np.add.at(m, (rows, c.nbr[:, j]), w)
        m[rows, rows] -= w
    return m


def _multiplets(ev: np.ndarray, kappa: float, lmax: int = 6):
    i = 1  # ev[0] is the constant mode
    for ell in range(1, lmax + 1):
        blk = ev[i : i + 2 * ell + 1]
        i += 2 * ell + 1
        yield ell, blk / (ell * (ell + 1) * kappa)


@pytest.mark.parametrize("nside,fz,rp,h", [(8, 8, 36.0, 16.0), (12, 12, 48.0, 24.0)])
def test_the_geometric_laplacian_has_the_spherical_harmonic_spectrum(nside, fz, rp, h) -> None:
    """Multiplet means within 3 % of l(l+1) for l = 1..6 (4 % for l = 6 at Nside 8, where the spec's 3.3 % is
    quoted) and the spread inside a multiplet within +-25 %: the grid anisotropy is a documented effect."""
    c = fieldtab.conductance(rp, h, nside, fz)
    n = geom.npix(nside)
    for k in (0, fz - 1):
        m = _tangential_operator(c, k)
        assert np.abs(m - m.T).max() == 0.0  # symmetric: one number per pair
        ev = np.sort(-np.linalg.eigvalsh(m))
        kappa = (
            c.thickness[k] * (4 * math.pi / n) / (c.cell_volume * 256.0 / c.scale)
        )  # a unit-sphere Laplacian
        assert abs(ev[0]) < 1e-9
        for ell, rel in _multiplets(ev, kappa):
            tol = 0.04 if (nside == 8 and ell == 6) else 0.03
            assert abs(rel.mean() - 1.0) < tol, (nside, k, ell, rel.mean())
            assert rel.min() > 0.75 and rel.max() < 1.25, (nside, k, ell, rel.min(), rel.max())


# --- the chemicals ---------------------------------------------------------------------------------


def _chem_tables(c: fieldtab.Conductance, d_max: float) -> fields_sph.ChemTables:
    return fields_sph.ChemTables(
        nbr=mx.array(c.nbr, dtype=mx.int32),
        c_t=mx.array(c.wt / 256.0, dtype=mx.float32),
        c_z=mx.array((c.wz / 256.0).reshape(-1, 1), dtype=mx.float32),
        n_sub=c.chem_substeps(d_max),
    )


def _chem_ref(c: fieldtab.Conductance, chem, dep_q, diff, keep, scale, cmax, n_sub):
    """The chemical update in float64 numpy on the mirror of the pair list."""
    fz = c.layers
    x = chem.astype(np.float64) + dep_q.astype(np.float64) * scale
    for _ in range(n_sub):
        lap = np.zeros_like(x)
        for j in range(4):
            lap += (c.wt[None, :, :, j] / 256.0) * (x[:, :, c.nbr[:, j]] - x)
        if fz > 1:
            cz = (c.wz / 256.0).reshape(1, -1, 1)
            flux = cz * (x[:, 1:] - x[:, :-1])
            lap[:, :-1] += flux
            lap[:, 1:] -= flux
        x = x + (diff.reshape(-1, 1, 1) / n_sub) * lap
    return np.clip(keep.reshape(-1, 1, 1) * x, 0.0, cmax)


def _chem_inputs(c, seed):
    rng = np.random.default_rng(seed)
    shape = (3, c.layers, geom.npix(c.nside))
    chem = rng.random(shape) * 2.0
    dep = rng.integers(0, 30, shape) * (rng.random(shape) < 0.02)
    return chem.astype(np.float32), dep.astype(np.int32)


@pytest.mark.parametrize("name", IDS)
@pytest.mark.usefixtures("dev")
def test_chemical_update_matches_a_float64_reference(name: str) -> None:
    c = _cond(name)
    diff = np.array([0.16, 0.10, 0.16], np.float32)
    keep = np.array([1 - 0.01, 1 - 0.002, 1 - 0.01], np.float32)
    t = _chem_tables(c, 0.16)
    chem, dep = _chem_inputs(c, 5)
    got = fields_sph.update_chem_arrays(
        mx.array(chem),
        mx.array(dep),
        t,
        mx.array(diff).reshape(3, 1, 1),
        mx.array(keep).reshape(3, 1, 1),
        0.01,
        50.0,
    )
    ref = _chem_ref(c, chem, dep, diff, keep, 0.01, 50.0, t.n_sub)
    assert np.abs(np.array(got) - ref).max() < 2e-5 * max(1.0, ref.max())


@pytest.mark.parametrize("name", IDS)
@pytest.mark.usefixtures("dev")
def test_chemical_mass_is_conserved_without_decay_and_stays_non_negative(name: str) -> None:
    c = _cond(name)
    diff = mx.array([0.25, 0.1, 0.16]).reshape(3, 1, 1)  # the largest allowed D_c needs the most sub-steps
    t = _chem_tables(c, 0.25)
    chem, _ = _chem_inputs(c, 6)
    x = mx.array(chem)
    zero = mx.zeros(chem.shape, dtype=mx.int32)
    ones = mx.ones((3, 1, 1))
    total0 = chem.astype(np.float64).sum(axis=(1, 2))
    for _ in range(40):
        x = fields_sph.update_chem_arrays(x, zero, t, diff, ones, 0.0, 1e9)
    mx.eval(x)
    out = np.array(x)
    assert out.min() >= 0.0
    assert (
        np.abs(out.astype(np.float64).sum(axis=(1, 2)) / total0 - 1.0).max() < 5e-6
    )  # float32 rounding only
    assert out.std() < chem.std()  # and it diffuses


def test_the_substeps_are_needed_for_stability() -> None:
    """The box's one-shot update with D_c = 0.16 is unstable on the thin outer layers (row sum 20 against 6):
    a single explicit step per call blows a smooth field up, the sub-stepped one does not."""
    c = _cond("tiny3s")
    diff = mx.array([0.16, 0.16, 0.16]).reshape(3, 1, 1)
    one = mx.ones((3, 1, 1))
    zero = mx.zeros((3, c.layers, geom.npix(c.nside)), dtype=mx.int32)
    chem, _ = _chem_inputs(c, 8)
    sub, naive = _chem_tables(c, 0.16), _chem_tables(c, 0.16)._replace(n_sub=1)
    a, b = mx.array(chem), mx.array(chem)
    for _ in range(30):
        a = fields_sph.update_chem_arrays(a, zero, sub, diff, one, 0.0, 1e30)
        b = fields_sph.update_chem_arrays(b, zero, naive, diff, one, 0.0, 1e30)
    mx.eval(a, b)
    assert float(np.array(a).max()) <= chem.max() and np.array(a).min() >= 0.0
    assert float(np.abs(np.array(b)).max()) > 1e3 * chem.max()  # unstable without the sub-steps
    assert sub.n_sub == 4


def test_the_substepped_update_has_the_same_spherical_harmonic_spectrum() -> None:
    """The gate of the integer tables, applied to what the chemical update does in one step: the per-step
    decay rate of each mode of a one-layer shell, ``-ln(1 - (D/n) lambda)*n``, is l(l+1) times the same
    constant."""
    c = fieldtab.conductance(
        48.0, 24.0, 12, 1
    )  # one layer: no radial coupling, the tangential operator alone
    n = geom.npix(12)
    m = _tangential_operator(c, 0)
    ev = np.sort(-np.linalg.eigvalsh(m))
    kappa = c.thickness[0] * (4 * math.pi / n) / (c.cell_volume * 256.0 / c.scale)
    d = 0.16
    n_sub = 4  # forced: a one-layer shell alone would need only one
    rate = -n_sub * np.log1p(-(d / n_sub) * ev) / d  # per-step decay over D, in the units of lambda
    for _ell, rel in _multiplets(rate, kappa):
        assert abs(rel.mean() - 1.0) < 0.03 and rel.min() > 0.75 and rel.max() < 1.25


def test_the_chemical_update_runs_on_a_mode_of_the_operator_as_predicted() -> None:
    """Apply ``update_chem_arrays`` to one eigenvector of the one-layer tangential operator: the field is
    multiplied by ``(1 - (D/n) lambda)^n`` (decay and deposit off)."""
    c = fieldtab.conductance(48.0, 24.0, 12, 1)
    m = _tangential_operator(c, 0)
    w, v = np.linalg.eigh(m)
    idx = int(np.argsort(-w)[5])  # a low mode (index 5 from the top of -lambda)
    vec = v[:, idx].astype(np.float32)
    lam = -w[idx]
    t = fields_sph.ChemTables(
        mx.array(c.nbr, dtype=mx.int32), mx.array(c.wt / 256.0, dtype=mx.float32), mx.zeros((0, 1)), 4
    )
    d = 0.16
    x = mx.array(np.tile(1.0 + 5.0 * vec, (3, 1, 1)))  # a positive field: the update clips at zero
    out = fields_sph.update_chem_arrays(
        x, mx.zeros(x.shape, dtype=mx.int32), t, mx.full((3, 1, 1), d), mx.ones((3, 1, 1)), 0.0, 1e9
    )
    want = (1.0 - d / 4 * lam) ** 4
    assert np.abs(np.array(out)[0, 0] - 1.0 - want * 5.0 * vec).max() < 1e-6


# --- compile = eager -------------------------------------------------------------------------------


def test_compiled_kernels_equal_eager_with_runtime_tables() -> None:
    c = _cond("tiny3s")
    d_q8 = 7
    nbr, a_t, a_z = _arrays(c, d_q8)
    r = mx.array(_fields(c, "random", 21).astype(np.int32))
    eager = ops.int_diffuse_sph(r, nbr, a_t, a_z)
    comp = mx.compile(ops.int_diffuse_sph)(r, nbr, a_t, a_z)
    assert np.array_equal(np.array(eager), np.array(comp))

    chem, dep = _chem_inputs(c, 9)
    diff, keep = mx.array([0.16, 0.1, 0.16]).reshape(3, 1, 1), mx.array([0.99, 0.998, 0.99]).reshape(3, 1, 1)

    def upd(x, dp, nb, ct, cz, df, kp, scale, cmax):
        return fields_sph.update_chem_arrays(x, dp, fields_sph.ChemTables(nb, ct, cz, 4), df, kp, scale, cmax)

    args = (
        mx.array(chem),
        mx.array(dep),
        mx.array(c.nbr, dtype=mx.int32),
        mx.array(c.wt / 256.0, dtype=mx.float32),
        mx.array((c.wz / 256.0).reshape(-1, 1), dtype=mx.float32),
        diff,
        keep,
        mx.array(0.01),
        mx.array(50.0),
    )
    e, k = upd(*args), mx.compile(upd)(*args)
    mx.eval(e, k)
    assert np.abs(np.array(e) - np.array(k)).max() < 1e-5
