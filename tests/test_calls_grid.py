"""The column voice grid of the far hearing (DESIGN §47.7, calls step K2): the table against brute force, the
exact integer deposit, the whole pipeline against a float64 reference in a 2D box, a 3D box and a sphere, the
accuracy tables against the exact column sum (dense and sparse callers, the inner edge, the polar caps and the
belt), the edge cases K7 to K10, the saturation law, the ramp at the near range, the signs of the bearing and
the rotation-equivariance harness. Eager, on the CPU; nothing here compiles (``test_calls_far_step`` does)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import calls_grid_helpers as G
import hp_oracle
from alife import calls, calltab, comm, fields
from alife.config import ConfigError, dims, load_config
from calls_helpers import calls_cfg
from sphere_rotation import random_rotation, rot

# --- the table ---------------------------------------------------------------------------------------------


def _brute(kind: str, n: int, scale: float, hear: float, far: float, atten: float):
    """Rows ``[(p', w, off)]`` per column by brute force (the reference of ``calls_grid_helpers``,
    parametrised by the geometry instead of a config)."""
    if kind == "sphere":
        c = hp_oracle.pix2vec(n) * scale
        cell = scale * math.sqrt(math.pi / 3.0) / n
        rows = []
        for p in range(len(c)):
            dv = c - c[p]
            d = np.sqrt((dv * dv).sum(axis=1))
            rows.append(
                [
                    (q, G.kernel(float(d[q]), hear, far, cell, atten), dv[q] / far)
                    for q in range(len(c))
                    if hear <= d[q] <= far
                ]
            )
        return rows
    return G.box_rows(n, scale, hear, far, atten)


@pytest.mark.parametrize(
    ("kind", "n", "scale", "far", "atten"),
    [
        (
            "sphere",
            4,
            24.0,
            17.0,
            1.0,
        ),  # 192 columns: every pole, seam and trivalent-vertex column is in a row
        ("sphere", 7, 36.0, 21.5, 1.0),
        ("sphere", 12, 60.0, 23.0, 2.0),  # a non-unit atten_pow
        ("box", 16, 64.0, 15.0, 1.0),  # 16 x 16 columns of side 4: rows near the edges wrap the torus
        ("box", 20, 100.0, 24.0, 1.5),
    ],
)
def test_the_table_equals_brute_force(kind: str, n: int, scale: float, far: float, atten: float) -> None:
    """Every entry of every column (index, weight, offset), the count and the padding, against a double loop
    over the centres of an independent HEALPix (or over the nine torus images)."""
    hear = 6.0
    tab = calltab.build(kind, n, scale, hear, far, atten)
    rows = _brute(kind, n, scale, hear, far, atten)
    assert tab.ncol == len(rows) and tab.s == max(len(r) for r in rows)
    nt = 3 if kind == "sphere" else 2
    assert tab.idx.shape == (tab.ncol, tab.s) and tab.off.shape == (tab.ncol, tab.s, nt)
    assert tab.idx.dtype == np.int32 and tab.w.dtype == np.float32 and tab.off.dtype == np.float32
    for p, row in enumerate(rows):
        k = len(row)
        assert tab.count[p] == k
        assert list(tab.idx[p, :k]) == [q for q, _, _ in row], p  # ascending, the same set
        np.testing.assert_allclose(tab.w[p, :k], [w for _, w, _ in row], rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(tab.off[p, :k], np.array([o for *_, o in row]), atol=1e-6)
        assert not tab.w[p, k:].any() and not tab.idx[p, k:].any() and not tab.off[p, k:].any()  # K8
    # symmetry: q in the row of p with weight w <=> p in the row of q with the same weight and the opposite
    # offset
    for p in range(0, tab.ncol, max(tab.ncol // 40, 1)):
        for s in range(tab.count[p]):
            q = tab.idx[p, s]
            back = list(tab.idx[q, : tab.count[q]]).index(p)
            assert tab.w[q, back] == tab.w[p, s] and np.allclose(tab.off[q, back], -tab.off[p, s], atol=1e-6)
    assert tab.w.max() <= 1.0 and np.abs(tab.off).max() <= 1.0 + 1e-6
    for a in (tab.idx, tab.w, tab.off):
        assert not a.flags.writeable


def test_the_sphere_columns_include_the_poles_the_seams_and_the_trivalent_vertices() -> None:
    """The columns the generic comparison above must cover, named: the polar pixels of both caps, the pixels
    at the eight trivalent vertices (valence 3) and the pixels on the facet seams all have rows equal to the
    brute force (the chord table has no singularity there)."""
    n = 6
    tab = calltab.build("sphere", n, 30.0, 6.0, 20.0, 1.0)
    rows = _brute("sphere", n, 30.0, 6.0, 20.0, 1.0)
    c = hp_oracle.pix2vec(n)
    poles = list(np.flatnonzero(np.abs(c[:, 2]) > 0.95))
    _touch, _edge, valence, vid = hp_oracle.neighbours(n)
    tri = [p for p in range(tab.ncol) if (valence[vid[p]] == 3).any()]
    phi = np.arctan2(c[:, 1], c[:, 0]) * 2.0 / np.pi
    seam = list(np.flatnonzero(np.abs(phi - np.round(phi)) < 0.06))
    assert len(poles) >= 8 and len(tri) >= 8 and len(seam) >= 8
    for p in {*poles, *tri, *seam}:
        assert tab.count[p] == len(rows[p]) and list(tab.idx[p, : tab.count[p]]) == [
            q for q, _, _ in rows[p]
        ], p
        np.testing.assert_allclose(tab.w[p, : tab.count[p]], [w for _, w, _ in rows[p]], rtol=1e-6, atol=1e-9)
    # a polar column hears columns all around the pole: the tangent part of its offsets nearly cancels
    p = poles[0]
    tangent = tab.off[p, : tab.count[p], :2].sum(axis=0)
    assert np.linalg.norm(tangent) < 0.06 * tab.count[p]


def test_the_known_entry_counts() -> None:
    """S = 50 (64k3s, far 18), 92 (64k3s, 24) and 49 (mini3s, 18), the inner edge at 6 (spec 3.2)."""
    assert calltab.build("sphere", 21, 90.0, 6.0, 18.0, 1.0).s == 50
    assert calltab.build("sphere", 21, 90.0, 6.0, 24.0, 1.0).s == 92
    assert calltab.build("sphere", 11, 48.0, 6.0, 18.0, 1.0).s == 49
    assert calltab.column_side("sphere", 21, 90.0) == pytest.approx(4.386, abs=1e-3)
    assert calltab.column_side("box", 32, 128.0) == 4.0


def test_the_ranges_are_inclusive_and_the_weight_vanishes_at_both_edges() -> None:
    """A column exactly at ``hear_range`` or ``far_range`` is an entry (the range is closed) with weight 0:
    the far kernel starts at 0 where the near kernel ends, and ends at 0."""
    tab = calltab.build("box", 32, 128.0, 8.0, 20.0, 1.0)  # (2, 0) columns are at 8; (3, 4) columns at 20
    p = 100
    row = {
        int(q): float(w) for q, w in zip(tab.idx[p, : tab.count[p]], tab.w[p, : tab.count[p]], strict=True)
    }
    assert row[p + 2] == 0.0 and row[p + 3 + 4 * 32] == 0.0  # distances 8.0 and 20.0 exactly
    assert row[p + 3] > 0 and row[p + 2 * 32 + 2] > 0  # 12.0 and 11.3 are inside both
    assert p not in row and p + 1 not in row  # inside the near range: the near block hears it, not this one


# --- the deposit -------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def box2() -> object:
    return calls_cfg("box2", "calls.far_range=18", "calls.far_xref=2.0")


def test_the_deposit_is_exact_and_independent_of_the_order_of_the_rows(box2) -> None:
    cfg = box2
    d = dims(cfg)
    ns = G.scene(cfg, 700, 3)
    rs = np.random.default_rng(1)
    voice = (rs.random((d.n, d.voice_w)) * (rs.random((d.n, d.voice_w)) < 0.5)).astype(np.float32)
    voice[5, 0], voice[6, 0] = np.float32(1.0 - 2.0**-24), np.float32(0.0)  # the largest call; silence
    rt = calls.runtime(cfg)
    col = G.columns(cfg, ns)
    ncol = calltab.ncol_of(*calltab.geometry_key(cfg)[:2])
    dep = np.array(calls.deposit(mx.array(voice), col, ns.alive, rt, ncol))
    assert dep.dtype == np.int32 and dep.shape == (d.voice_w, ncol)
    want = np.zeros((d.voice_w, ncol), np.int64)
    alive, cn = np.array(ns.alive), np.array(col)
    for i in np.flatnonzero(alive):
        for c in range(d.voice_w):
            want[c, cn[i]] += min(255, math.floor(float(voice[i, c]) * 256.0))
    assert np.array_equal(dep, want)
    assert min(255, math.floor((1.0 - 2.0**-24) * 256.0)) == 255  # the cap is the call below 1
    for seed in range(3):  # any permutation of the rows gives the same deposit
        perm = np.random.default_rng(seed).permutation(d.n)
        dp = np.array(
            calls.deposit(mx.array(voice[perm]), mx.array(cn[perm]), mx.array(alive[perm]), rt, ncol)
        )
        assert np.array_equal(dp, dep)
    # a dead row deposits nothing (K6), whatever its stored voice says
    dead = np.flatnonzero(~alive)[:50]
    loud = voice.copy()
    loud[dead] = 0.9
    assert np.array_equal(np.array(calls.deposit(mx.array(loud), col, ns.alive, rt, ncol)), dep)


def test_the_deposit_quantum_is_capped_at_255(box2) -> None:
    """``min(255, floor(256 sig))``: the engine's call is below 1 (at most 255), but the deposit is a pure
    function and holds its own cap for any stored value (1.0 and above): a column's sum stays in int32."""
    cfg = box2
    d = dims(cfg)
    n = d.n
    alive = np.zeros(n, bool)
    alive[:3] = True
    voice = np.zeros((n, d.voice_w), np.float32)
    voice[0], voice[1], voice[2] = 1.0, 3.5, 0.5
    col = mx.zeros((n,), mx.int32)  # all three in column 0
    ncol = calltab.ncol_of(*calltab.geometry_key(cfg)[:2])
    dep = np.array(calls.deposit(mx.array(voice), col, mx.array(alive), calls.runtime(cfg), ncol))
    assert dep[0, 0] == 255 + 255 + 128 and dep.sum() == dep[:, 0].sum()


# --- the pipeline against float64 --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("world", "far"), [("box2", "18"), ("box3", "20"), ("box2w", "16.5"), ("sph", "20"), ("sphw", "22")]
)
def test_the_pipeline_equals_the_float64_reference(world: str, far: str) -> None:
    """``hear_far`` on a random scene (400 alive rows, 30 % calling, random headings): the heard intensity and
    the per-channel bearing in the hearer's frame equal a Python-loop float64 reference built from the brute
    force table, to float32 rounding (3e-7 relative in the spec; here an absolute 1e-6)."""
    cfg = calls_cfg(world, f"calls.far_range={far}", "calls.far_xref=2.5")
    rows, _ = G.ref_table(cfg)
    ns = G.scene(cfg, 400, 11)
    d = dims(cfg)
    rs = np.random.default_rng(12)
    voice = (rs.random((d.n, d.voice_w)) * (rs.random((d.n, d.voice_w)) < 0.3)).astype(np.float32)
    h, b = G.hear(cfg, ns, voice)
    hr, br = G.ref_hear(cfg, ns, voice, rows)
    assert h.shape == (d.n, d.voice_w) and b.shape == (d.n, 2 * d.voice_w)
    np.testing.assert_allclose(h, hr, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(b, br, atol=1e-6)
    alive = np.array(ns.alive)
    assert (h[alive] > 0).any() and (np.abs(b[alive]) > 0.05).any()  # non-trivial scene
    assert not h[~alive].any() and not b[~alive].any()  # a dead row hears nothing
    assert h.max() < 1.0 and np.abs(b).max() <= 1.0 + 1e-6


def test_the_pipeline_is_deterministic_and_independent_of_the_row_order(box2) -> None:
    """Rerun and a random permutation of the rows give identical bits (spec 3.3: integer deposit, a gather, a
    multiply and one fixed-order sum)."""
    cfg = box2
    d = dims(cfg)
    ns = G.scene(cfg, 600, 4)
    rs = np.random.default_rng(5)
    voice = (rs.random((d.n, d.voice_w)) * (rs.random((d.n, d.voice_w)) < 0.4)).astype(np.float32)
    h, b = G.hear(cfg, ns, voice)
    h2, b2 = G.hear(cfg, ns, voice)
    assert h.tobytes() == h2.tobytes() and b.tobytes() == b2.tobytes()
    perm = np.random.default_rng(9).permutation(d.n)
    ns_p = SimpleNamespace(**{**vars(ns), "alive": ns.alive[mx.array(perm)], "pos": ns.pos[mx.array(perm)],
                              "heading": ns.heading[mx.array(perm)]})  # fmt: skip
    hp, bp = G.hear(cfg, ns_p, voice[perm])
    assert hp.tobytes() == h[perm].tobytes() and bp.tobytes() == b[perm].tobytes()


# --- accuracy against the exact column sum -----------------------------------------------------------------


def _mini3s(far: float = 18.0):
    return load_config(
        preset="mini3s",
        overrides=[
            "brain.weight_dtype=float32",
            "comm.enabled=true",
            "comm.width=2",
            "calls.enabled=true",
            f"calls.far_range={far}",
        ],
    )


def _accuracy(cfg, n_callers: int, seed: int = 7, hearers: int = 3000, patch: float | None = None):
    """Grid against the exact sum for ``n_callers`` callers (uniform, or in a patch of sigma ``patch`` units)
    and ``hearers`` hearers: ``(median symmetric relative error, p90, share inaudible in the grid, share
    phantom in the grid, median bearing error in degrees, scene, hearer rows, x grid, x exact)`` over the
    hearers that are audible in either."""
    d = dims(cfg)
    rs = np.random.default_rng(seed)
    ns = G.scene(cfg, n_callers + hearers, seed)
    if patch is not None:  # callers and hearers near one centre
        u = rs.normal(size=(1, 3))
        u /= np.linalg.norm(u)
        pos = np.array(ns.pos)
        alive = np.flatnonzero(np.array(ns.alive))
        near = u + rs.normal(size=(len(alive), 3)) * (patch / cfg.world.radius)
        near /= np.linalg.norm(near, axis=1, keepdims=True)
        pos[alive] = near * np.linalg.norm(pos[alive], axis=1, keepdims=True)
        ns.pos = mx.array(pos.astype(np.float32))
    alive_idx = rs.permutation(np.flatnonzero(np.array(ns.alive)))
    call_rows, hear_rows = alive_idx[:n_callers], alive_idx[n_callers:]
    voice = np.zeros((d.n, d.voice_w), np.float32)
    voice[call_rows] = rs.uniform(0.15, 0.999, (n_callers, d.voice_w)).astype(np.float32)
    h, b = G.hear(cfg, ns, voice)
    xg = -cfg.calls.far_xref * np.log1p(-h[hear_rows].astype(np.float64))
    xe, be = G.exact_sum(cfg, ns, voice, hear_rows)
    m = (xe[:, 0] > 0) | (xg[:, 0] > 0)
    rel = np.abs(xg[m, 0] - xe[m, 0]) / np.maximum(np.maximum(xg[m, 0], xe[m, 0]), 1e-12)
    inaud = float(((xe[:, 0] > 0) & (xg[:, 0] == 0)).sum() / max((xe[:, 0] > 0).sum(), 1))
    phantom = float(((xe[:, 0] == 0) & (xg[:, 0] > 0)).sum() / max((xg[:, 0] > 0).sum(), 1))
    ex = np.zeros((len(hear_rows), 2))
    for k, i in enumerate(hear_rows):
        f, la = G.frame(cfg, ns, i)
        v = be[k, 0]
        up = np.array(ns.pos[i], np.float64)
        v = v - (v @ (up / np.linalg.norm(up))) * (up / np.linalg.norm(up))
        ex[k] = (v @ f, v @ la)
    bg = b[hear_rows][:, :2]
    both = (xe[:, 0] > 0) & (xg[:, 0] > 0)
    cos = (ex[both] * bg[both]).sum(1) / np.maximum(
        np.linalg.norm(ex[both], axis=1) * np.linalg.norm(bg[both], axis=1), 1e-12
    )
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return (
        float(np.median(rel)),
        float(np.quantile(rel, 0.9)),
        inaud,
        phantom,
        float(np.median(ang)),
        ns,
        hear_rows,
        xg,
        xe,
    )


# The measured accuracy of the grid (mini3s, far_range 18, DESIGN §47.7), with the bound each test holds it to
# (about 1.4 times the figure, which is deterministic for these seeds). callers: (median symmetric error,
# audibility flips in either direction, median bearing error in degrees).
ACCURACY = {
    3: (0.36, 0.20, 40.0),
    10: (0.36, 0.20, 40.0),
    100: (0.19, 0.05, 25.0),
    600: (0.09, 0.01, 20.0),
    2000: (0.05, 0.01, 20.0),
}


@pytest.mark.parametrize("n_callers", sorted(ACCURACY))
def test_the_grid_against_the_exact_sum_uniform_callers(n_callers: int) -> None:
    """Dense scenes are good (5 to 9 % median error at 600 to 2 000 callers), few callers are poor (about a
    third of the intensity and 10 to 15 % of the audibilities wrong at 3 to 10): the figures of the spec,
    with the ramp, as bounds. Not a claim of accuracy for a lone call (DESIGN §47.7)."""
    med_bound, flip_bound, ang_bound = ACCURACY[n_callers]
    med, p90, inaud, phantom, ang, *_ = _accuracy(_mini3s(), n_callers)
    assert med <= med_bound * 1.4, (med, p90)
    assert max(inaud, phantom) <= flip_bound * 1.4, (inaud, phantom)
    assert ang <= ang_bound, ang
    if n_callers <= 10:  # the documented weakness is real: this is not a vacuous bound
        assert med > 0.15 and max(inaud, phantom) > 0.03


def test_a_patch_of_callers_is_the_favourable_case() -> None:
    """3 to 10 callers in a patch of sigma 6 (a storm patch): hearers near it, 22 to 25 % median error in the
    spec, audibility flips only a few percent."""
    med, _p90, inaud, phantom, _ang, *_ = _accuracy(_mini3s(), 10, patch=6.0, hearers=3000)
    assert med <= 0.40 and max(inaud, phantom) <= 0.12


def test_the_inner_edge_shifts_by_about_one_column() -> None:
    """Pairs within 4.5 units of the near range land on the wrong side of it in about 17 % of the cases, never
    far from the edge: both ends are quantised, the edge moves by up to a column side (4.4), not half."""
    cfg = _mini3s()
    d, w = dims(cfg), cfg.world
    rs = np.random.default_rng(3)
    n = 4000
    # a caller at a random place and a hearer at distance r from it along a random great circle
    rows = np.arange(2 * n)
    u = rs.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    t = rs.normal(size=(n, 3))
    t -= (t * u).sum(axis=1, keepdims=True) * u
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    r = rs.uniform(cfg.sensing.radius - 6.0, cfg.sensing.radius + 6.0, n)  # the chord on the floor sphere
    ang = 2.0 * np.arcsin(np.minimum(r / (2.0 * w.radius), 1.0))
    v = u * np.cos(ang)[:, None] + t * np.sin(ang)[:, None]
    from alife.fields import cell_index

    pos_c, pos_h = u * (w.radius + 3.0), v * (w.radius + 3.0)
    pc = np.array(
        calls.column_of(cell_index(mx.array(pos_c.astype(np.float32)), cfg), cfg, calls.runtime(cfg))
    )
    ph = np.array(
        calls.column_of(cell_index(mx.array(pos_h.astype(np.float32)), cfg), cfg, calls.runtime(cfg))
    )
    tab = calltab.table_for(cfg)
    audible = np.array(
        [
            (tab.w[a, : tab.count[a]][tab.idx[a, : tab.count[a]] == b] > 0).any()
            for a, b in zip(ph, pc, strict=True)
        ]
    )
    exact = r > cfg.sensing.radius  # the far kernel is 0 inside the near range
    wrong = audible != exact
    near_edge = np.abs(r - cfg.sensing.radius) < 4.5
    assert 0.08 <= wrong[near_edge].mean() <= 0.30, wrong[near_edge].mean()
    assert wrong[np.abs(r - cfg.sensing.radius) > 5.9].mean() < 0.15
    del rows, d


def test_the_error_is_the_same_in_the_polar_caps_and_the_belt() -> None:
    """Equal-area pixels of unequal shape: the dense-scene error of the hearers in the caps (|z| > 2/3) and in
    the belt are both small and within a factor 2 of each other (spec 3.3, region)."""
    cfg = _mini3s()
    *_, ns, hear_rows, xg, xe = _accuracy(cfg, 600, seed=5)
    pos = np.array(ns.pos)[hear_rows]
    z = np.abs(pos[:, 2] / np.linalg.norm(pos, axis=1))
    errs = {}
    for name, sel in (("belt", z <= 2 / 3), ("caps", z > 2 / 3)):
        m = sel & ((xe[:, 0] > 0) | (xg[:, 0] > 0))
        assert m.sum() > 300, name
        errs[name] = float(
            np.median(np.abs(xg[m, 0] - xe[m, 0]) / np.maximum(np.maximum(xg[m, 0], xe[m, 0]), 1e-12))
        )
    assert max(errs.values()) <= 0.14 and 0.5 <= errs["caps"] / errs["belt"] <= 2.0, errs


def test_the_box_grid_is_as_accurate_as_the_sphere_grid() -> None:
    cfg = calls_cfg("box2", "calls.far_range=18")
    med, _p90, inaud, phantom, ang, *_ = _exact_box(cfg, 600)
    assert med <= 0.15 and max(inaud, phantom) <= 0.05 and ang <= 30.0, (med, inaud, phantom, ang)


def _exact_box(cfg, n_callers: int):
    d = dims(cfg)
    rs = np.random.default_rng(2)
    ns = G.scene(cfg, n_callers + 400, 2)
    idx = rs.permutation(np.flatnonzero(np.array(ns.alive)))
    call_rows, hear_rows = idx[:n_callers], idx[n_callers:]
    voice = np.zeros((d.n, d.voice_w), np.float32)
    voice[call_rows] = rs.uniform(0.15, 0.999, (n_callers, d.voice_w)).astype(np.float32)
    h, b = G.hear(cfg, ns, voice)
    xg = -cfg.calls.far_xref * np.log1p(-h[hear_rows].astype(np.float64))
    xe, be = G.exact_sum(cfg, ns, voice, hear_rows)
    m = (xe[:, 0] > 0) | (xg[:, 0] > 0)
    rel = np.abs(xg[m, 0] - xe[m, 0]) / np.maximum(np.maximum(xg[m, 0], xe[m, 0]), 1e-12)
    inaud = float(((xe[:, 0] > 0) & (xg[:, 0] == 0)).sum() / max((xe[:, 0] > 0).sum(), 1))
    phantom = float(((xe[:, 0] == 0) & (xg[:, 0] > 0)).sum() / max((xg[:, 0] > 0).sum(), 1))
    ex = np.zeros((len(hear_rows), 2))
    for k, i in enumerate(hear_rows):
        f, la = G.frame(cfg, ns, i)
        ex[k] = (be[k, 0] @ f[:2], be[k, 0] @ la[:2])
    bg = b[hear_rows][:, :2]
    both = (xe[:, 0] > 0) & (xg[:, 0] > 0)
    cos = (ex[both] * bg[both]).sum(1) / np.maximum(
        np.linalg.norm(ex[both], axis=1) * np.linalg.norm(bg[both], axis=1), 1e-12
    )
    return (
        float(np.median(rel)),
        0.0,
        inaud,
        phantom,
        float(np.median(np.degrees(np.arccos(np.clip(cos, -1, 1))))),
        ns,
    )


# --- edge cases K7 to K10 ----------------------------------------------------------------------------------


def test_k9_and_k10_a_lone_hearer_hears_exactly_nothing(box2) -> None:
    """One alive row, silent world, or a world with no row alive: the heard intensity and the bearing are
    exactly 0, with no NaN from the empty sum."""
    cfg = box2
    d = dims(cfg)
    for n_alive in (0, 1):
        ns = G.scene(cfg, n_alive, 6)
        voice = np.zeros((d.n, d.voice_w), np.float32)
        voice[np.flatnonzero(np.array(ns.alive))] = 0.9  # a lone caller does not hear itself
        h, b = G.hear(cfg, ns, voice)
        assert not h.any() and not b.any() and np.isfinite(h).all() and np.isfinite(b).all(), n_alive
    ns = G.scene(cfg, 40, 6)  # callers present but every hearer is dead except the callers' own columns
    h, b = G.hear(cfg, ns, np.zeros((d.n, d.voice_w), np.float32))
    assert not h.any() and not b.any()


def test_a_caller_in_the_near_range_is_not_heard_by_the_far_block(box2) -> None:
    """Inner edge: a caller 3 units away is the near block's, not this one's (its column is within the near
    range); one 12 units away is the far block's."""
    cfg = box2
    d = dims(cfg)

    def two(dist: float):
        n = d.n
        alive = np.zeros(n, bool)
        alive[:2] = True
        pos = np.zeros((n, 2), np.float32)
        pos[0], pos[1] = (64.0, 64.0), (64.0 + dist, 64.0)
        ns = SimpleNamespace(alive=mx.array(alive), pos=mx.array(pos), heading=mx.zeros((n,), mx.float32))
        voice = np.zeros((n, d.voice_w), np.float32)
        voice[1] = 0.9
        return G.hear(cfg, ns, voice)

    h, _ = two(3.0)
    assert not h[0].any()
    h, b = two(12.0)
    assert h[0].all() and b[0, 0] > 0.3  # ahead (heading 0, the caller is at +x)


def test_k7_deaf_hears_nothing_and_decoy_deposits_the_donors_voice_at_the_senders_column(box2) -> None:
    from dataclasses import replace

    cfg = box2
    d = dims(cfg)
    ns = G.scene(cfg, 500, 8)
    rs = np.random.default_rng(10)
    voice = (rs.random((d.n, d.voice_w)) * (rs.random((d.n, d.voice_w)) < 0.4)).astype(np.float32)
    msg = np.zeros((d.n, d.msg_w), np.float32)
    msg[:, 2:] = voice
    s = SimpleNamespace(**{**vars(ns), "msg": mx.array(msg), "step": mx.array(7, mx.int32)})
    ctx = SimpleNamespace(fcell=fields.cell_index(ns.pos, cfg))
    base_h, _base_b = calls.ext_inputs(s, cfg, ctx)
    assert np.array(base_h).any()
    deaf = replace(cfg, comm=replace(cfg.comm, control="deaf"))
    dh, db = calls.ext_inputs(s, deaf, ctx)
    assert not np.array(dh).any() and not np.array(db).any()  # K7: both blocks, exactly 0
    decoy = replace(cfg, comm=replace(cfg.comm, control="decoy"))
    xh, xb = calls.ext_inputs(s, decoy, ctx)
    src = np.array(comm.decoy_index(s.step, s.alive))
    alive = np.array(ns.alive)
    assert (
        sorted(src[alive]) == sorted(np.flatnonzero(alive)) and (src[alive] != np.flatnonzero(alive)).all()
    )  # a bijection
    eh, eb = G.hear(cfg, ns, voice[src])  # the donors' voices, each deposited at the sender's own column
    assert np.array(xh).tobytes() == eh.tobytes() and np.array(xb).tobytes() == eb.tobytes()
    assert np.array(xh).tobytes() != np.array(base_h).tobytes()
    # nofar and reflex_near zero the far block with the same layout
    nofar = replace(cfg, calls=replace(cfg.calls, control="nofar"))
    zh, zb = calls.ext_inputs(s, nofar, ctx)
    assert np.array(zh).shape == (d.n, d.voice_w) and np.array(zb).shape == (d.n, 2 * d.voice_w)
    assert not np.array(zh).any() and not np.array(zb).any()


# --- the law of the heard intensity ------------------------------------------------------------------------


def test_the_saturation_law_and_the_reference_gain() -> None:
    """``heard = 1 - exp(-x / far_xref)``: 0 for silence, monotone and below 1 in the loudness, ``x =
    far_xref`` gives ``1 - 1/e``, and a larger reference lowers the intensity of the same scene (the gain of
    the pilot)."""
    base = calls_cfg("box2", "calls.far_range=18")
    d = dims(base)
    ns = G.scene(base, 600, 2)
    rs = np.random.default_rng(4)
    voice = (rs.random((d.n, d.voice_w)) * (rs.random((d.n, d.voice_w)) < 0.4)).astype(np.float32)
    alive = np.array(ns.alive)
    xs = []
    for xref in (1.0, 2.0, 4.0):
        cfg = calls_cfg("box2", "calls.far_range=18", f"calls.far_xref={xref}")
        h, _ = G.hear(cfg, ns, voice)
        x = -xref * np.log1p(-h.astype(np.float64))  # recovers the loudness sum
        xs.append(x[alive])
        assert (h[alive] >= 0).all() and (h < 1).all()
    np.testing.assert_allclose(xs[0], xs[1], rtol=2e-5, atol=1e-6)  # the same loudness x, whatever the gain
    np.testing.assert_allclose(xs[0], xs[2], rtol=2e-5, atol=1e-6)
    h1, _ = G.hear(calls_cfg("box2", "calls.far_range=18", "calls.far_xref=1.0"), ns, voice)
    h4, _ = G.hear(calls_cfg("box2", "calls.far_range=18", "calls.far_xref=4.0"), ns, voice)
    assert (h4[alive] <= h1[alive]).all() and (h4[alive] < h1[alive]).any()
    big = xs[0].max()  # a loud scene saturates with xref 1 and not with xref >> x
    assert big > 1.0 and (1 - math.exp(-big)) > 0.63
    assert np.isclose(1 - math.exp(-1.0), 0.6321205588)


def test_the_loudness_is_continuous_across_the_near_range() -> None:
    """One caller moving away from the hearer: the near block (``comm.hear``'s weight) falls to 0 at the near
    range and the far weight rises from 0 there over one column side, so neither jumps. (Not monotone in
    total: the far kernel then falls to 0 at far_range, spec 3.2.)"""
    hear, far, cell, atten = 6.0, 18.0, 4.0, 1.0
    d = np.linspace(0.0, 20.0, 401)
    near = np.clip(1.0 - d / hear, 0.0, 1.0) ** atten
    farw = np.array([G.kernel(x, hear, far, cell, atten) for x in d])
    tot = near + farw
    assert np.abs(np.diff(tot)).max() < 0.06  # no jump anywhere (a step of 0.05 in distance)
    i = int(np.searchsorted(d, hear))
    assert near[i] == 0.0 and farw[i] == 0.0  # both vanish at the near range
    ramp = np.array([G.kernel(x, hear, far, cell, atten) / max(1 - x / far, 1e-9) for x in d])
    assert (
        ramp[np.searchsorted(d, hear + cell)] == pytest.approx(1.0)
        and 0 < ramp[np.searchsorted(d, hear + cell / 2)] < 1
    )
    assert farw[np.searchsorted(d, far)] == 0.0 and farw[np.searchsorted(d, far + 0.5)] == 0.0
    tab = calltab.build("box", 32, 128.0, hear, far, atten)
    nz = tab.w[tab.w > 0]
    assert nz.min() < 0.1 * nz.max()  # the entries nearest the edge are the faint ones


# --- the signs of the bearing ------------------------------------------------------------------------------


def _plant_box(cfg, heading: float, direction: float, dist: float = 12.0):
    d = dims(cfg)
    n = d.n
    alive = np.zeros(n, bool)
    alive[:2] = True
    pos = np.zeros((n, d.nd), np.float32)
    pos[0, :2] = (64.0, 64.0)
    pos[1, :2] = (64.0 + dist * math.cos(direction), 64.0 + dist * math.sin(direction))
    if d.d3:
        pos[:2, 2] = d.z_lo
    hd = np.zeros(n, np.float32)
    hd[0] = heading
    ns = SimpleNamespace(alive=mx.array(alive), pos=mx.array(pos), heading=mx.array(hd))
    voice = np.zeros((n, d.voice_w), np.float32)
    voice[1, 0] = 0.9
    return ns, voice


@pytest.mark.parametrize("world", ["box2", "box3"])
@pytest.mark.parametrize("heading", [0.0, 1.0, -2.2, 3.0])
@pytest.mark.parametrize("rel", [0.0, 0.7, math.pi / 2, 2.5, math.pi, -0.7, -math.pi / 2])
def test_the_bearing_signs_in_a_box(world: str, heading: float, rel: float) -> None:
    """A caller at angle ``rel`` from the hearer's heading (positive = to the left, +y when the heading is +x)
    gives (forward, lateral) = (cos rel, sin rel) times the mean offset, within the column quantisation."""
    cfg = calls_cfg(world, "calls.far_range=18")
    ns, voice = _plant_box(cfg, heading, heading + rel)
    h, b = G.hear(cfg, ns, voice)
    assert h[0, 0] > 0 and h[0, 1] == 0  # channel 0 only
    fwd, lat = b[0, 0], b[0, 1]
    got = math.atan2(lat, fwd)
    assert abs((got - rel + math.pi) % (2 * math.pi) - math.pi) < math.radians(25), (got, rel)
    assert b[0, 2] == 0 and b[0, 3] == 0  # the silent channel has no bearing
    assert math.hypot(fwd, lat) > 0.3  # a pointing vector, shorter than 1 (unit = far_range)


@pytest.mark.parametrize("rel", [0.0, 0.9, math.pi / 2, 2.4, math.pi, -1.3])
def test_the_bearing_signs_on_a_sphere(rel: float) -> None:
    """The same on the sphere: the caller is placed along the great circle that leaves the hearer in the
    direction ``cos(rel)·hdg + sin(rel)·(up × hdg)``; ``lat`` is positive to the left (``up × hdg``)."""
    cfg = calls_cfg("sph", "calls.far_range=20")
    d = dims(cfg)
    rs = np.random.default_rng(3)
    for trial in range(6):
        u = rs.normal(size=3)
        u /= np.linalg.norm(u)
        t = rs.normal(size=3)
        t -= (t @ u) * u
        hdg = t / np.linalg.norm(t)
        lat = np.cross(u, hdg)
        dirn = math.cos(rel) * hdg + math.sin(rel) * lat
        rad = cfg.world.radius + 3.0
        ang = 12.0 / cfg.world.radius
        uc = u * math.cos(ang) + dirn * math.sin(ang)
        n = d.n
        alive = np.zeros(n, bool)
        alive[:2] = True
        pos = np.zeros((n, 3), np.float32)
        pos[0], pos[1] = u * rad, uc * rad
        pos[2:] = (0.0, 0.0, cfg.world.radius + 3.0)
        hd = np.zeros((n, 3), np.float32)
        hd[:] = (1.0, 0.0, 0.0)
        hd[0] = hdg
        ns = SimpleNamespace(
            alive=mx.array(alive), pos=mx.array(pos), hdg=mx.array(hd), heading=mx.zeros((n,), mx.float32)
        )
        voice = np.zeros((n, d.voice_w), np.float32)
        voice[1, 0] = 0.9
        h, b = G.hear(cfg, ns, voice)
        assert h[0, 0] > 0
        got = math.atan2(b[0, 1], b[0, 0])
        assert abs((got - rel + math.pi) % (2 * math.pi) - math.pi) < math.radians(25), (trial, got, rel)


# --- the rotation-equivariance harness ---------------------------------------------------------------------

# The pixel grid is not rotation invariant: rotating the world moves every caller and hearer to another
# pixel. The exact reference IS equivariant (to the float32 rounding of the rotated positions); the grid is
# held to this explicit tolerance entry, which a consumer reading the Cartesian z (a raw ``pos[:, 2]``) or a
# box lattice would exceed by far (the negative control below). Dense scene: 600 callers, 3 000 hearers.
EQUIVARIANCE_TOL = {
    "exact_rel": 1e-5,  # the exact sum of the rotated scene equals the original's
    "heard_rel_median": 0.20,  # the grid's x before and after: measured 0.11
    "heard_rel_p90": 0.45,  # measured 0.27
    "bearing_deg_median": 40.0,  # body-frame bearing before and after: measured 22
}


def _rotated_scene():
    cfg = _mini3s()
    d = dims(cfg)
    rs = np.random.default_rng(5)
    n_callers, nh = 600, 3000
    ns = G.scene(cfg, n_callers + nh, 5)
    idx = rs.permutation(np.flatnonzero(np.array(ns.alive)))
    call_rows, hear_rows = idx[:n_callers], idx[n_callers:]
    voice = np.zeros((d.n, d.voice_w), np.float32)
    voice[call_rows] = rs.uniform(0.15, 0.999, (n_callers, d.voice_w)).astype(np.float32)
    rot_m = random_rotation(3)
    ns2 = SimpleNamespace(**{**vars(ns), "pos": rot(ns.pos, rot_m), "hdg": rot(ns.hdg, rot_m)})
    return cfg, ns, ns2, voice, hear_rows


def test_the_exact_reference_is_rotation_equivariant_and_the_grid_within_its_tolerance() -> None:
    tol = EQUIVARIANCE_TOL
    cfg, ns, ns2, voice, hear_rows = _rotated_scene()
    x0, _ = G.exact_sum(cfg, ns, voice, hear_rows)
    x1, _ = G.exact_sum(cfg, ns2, voice, hear_rows)
    assert (np.abs(x0 - x1) / np.maximum(x0, 1e-9)).max() < tol["exact_rel"]
    h0, b0 = G.hear(cfg, ns, voice)
    h1, b1 = G.hear(cfg, ns2, voice)
    g0 = -np.log1p(-h0[hear_rows].astype(np.float64))[:, 0]
    g1 = -np.log1p(-h1[hear_rows].astype(np.float64))[:, 0]
    m = (g0 > 0) & (g1 > 0)
    rel = np.abs(g0[m] - g1[m]) / np.maximum(g0[m], g1[m])
    assert np.median(rel) < tol["heard_rel_median"] and np.quantile(rel, 0.9) < tol["heard_rel_p90"]
    a, c = b0[hear_rows][m][:, :2], b1[hear_rows][m][:, :2]
    cos = (a * c).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(c, axis=1), 1e-12)
    assert np.median(np.degrees(np.arccos(np.clip(cos, -1, 1)))) < tol["bearing_deg_median"]


def test_a_raw_cartesian_z_would_break_the_equivariance_comparison() -> None:
    """The negative control: columns taken from the Cartesian z (what a raw ``pos[:, 2]`` consumer would do)
    do not follow the rotated world, so their heard intensity is far from the equivariant exact sum of the
    rotated scene, while the real grid is within its tolerance of it. The comparison above is not vacuous."""
    cfg, _ns, ns2, voice, hear_rows = _rotated_scene()
    d = dims(cfg)
    z = np.array(ns2.pos)[:, 2]
    col = mx.array(
        np.clip(((z - z.min()) / (np.ptp(z) + 1e-9) * (d.npix - 1)).astype(np.int32), 0, d.npix - 1)
    )
    h, _ = calls.hear_far(ns2, mx.array(voice), col, calls.runtime(cfg), cfg)
    raw = -cfg.calls.far_xref * np.log1p(-np.array(h, np.float64)[hear_rows, 0])
    exact, _ = G.exact_sum(cfg, ns2, voice, hear_rows)
    good = -cfg.calls.far_xref * np.log1p(-G.hear(cfg, ns2, voice)[0].astype(np.float64)[hear_rows, 0])

    def med(x: np.ndarray) -> float:
        m = (x > 0) | (exact[:, 0] > 0)
        return float(np.median(np.abs(x[m] - exact[m, 0]) / np.maximum(np.maximum(x[m], exact[m, 0]), 1e-12)))

    assert med(good) < EQUIVARIANCE_TOL["heard_rel_median"] and med(raw) > 2.0 * med(good)


# --- configuration -----------------------------------------------------------------------------------------


def test_far_range_validation() -> None:
    base = ["brain.weight_dtype=float32", "comm.enabled=true", "comm.width=2", "calls.enabled=true"]
    cases = [
        ("calls.far_range=17", "must be in"),  # below 4 columns (16 < 4 x 4.4 on mini3s: 17.86)
        ("calls.far_range=5", "must be in"),  # below the sensing radius too
        ("calls.far_range=25", "must be in"),  # above 4 sensing radii
        ("calls.far_range=-3", "must be 0 or a finite"),
        ("calls.far_range=nan", "finite"),
    ]
    for item, msg in cases:
        with pytest.raises(ConfigError, match=msg):
            load_config(preset="mini3s", overrides=[*base, item])
    for items, msg in (
        (["comm.range=4", "calls.far_range=18"], r"comm.range == 0"),
        (["comm.kin_exp=1", "calls.far_range=18"], r"comm.kin_exp == 0"),
    ):
        with pytest.raises(ConfigError, match=msg):
            load_config(preset="mini3s", overrides=[*base, *items])
    ok = load_config(preset="mini3s", overrides=[*base, "calls.far_range=17.9"])
    assert ok.calls.far_range == 17.9
    with pytest.raises(
        ConfigError, match=r"max 512|table"
    ):  # a range whose table is too large for a tiny column
        load_config(
            preset="tiny", overrides=[*base, "calls.far_range=24", "world.field_cell=0.5", "world.size=128"]
        )
