"""Built structures on the shell (DESIGN §38 on §46.5; sphere series S3): the colouring, the neighbourhood,
the update rule against an explicit numpy reference, the anti-trapping invariant on the sphere graph, the
sensor and the effects, and the eager step with the ledger and the material identity.

The compiled step with structures on is in ``test_sphere_features_compiled`` (cold child processes).
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from alife import config, fields, fields_sph, geom, invariants, physics, structures
from alife.config import ConfigError, dims
from alife.state import init_state, to_numpy
from helpers import set_output_bias
from sphere_feature_helpers import (
    adjacency_brute,
    cfg_of,
    counters,
    eager_steps,
    neighbour_index,
    ref_nopen,
    ref_phase,
    ref_update,
)

NSIDES = (2, 3, 4, 5, 8, 11)


# --- the colouring ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("nside", NSIDES)
def test_the_minimised_colouring_is_proper_on_the_distance_two_graph(nside: int) -> None:
    colour, k = geom.min_colouring(nside)
    assert geom.check_colouring(nside, colour)
    assert colour.min() == 0 and colour.max() == k - 1 and len(np.unique(colour)) == k
    assert 9 <= k <= geom.greedy_colouring(nside)[1]  # never worse than the greedy, never below the clique
    # brute force: any two pixels of one colour are more than two touching steps apart
    adj = adjacency_brute(nside)
    for p in range(len(adj)):
        near = set(adj[p]).union(*(adj[q] for q in adj[p])) - {p}
        assert all(colour[q] != colour[p] for q in near), p


@pytest.mark.parametrize(("nside", "limit"), [(8, 11), (11, 11), (21, 12)])
def test_the_colouring_is_small_at_the_preset_resolutions(nside: int, limit: int) -> None:
    """The greedy of S0 gives 15, 16 and 15 colours (the ring order alone 15, 16, 18: spec 2.6); 45 to 54
    phases become 33 to 36."""
    colour, k = geom.min_colouring(nside)
    assert k <= limit
    assert geom.check_colouring(nside, colour)


# The colouring decides which cell may close at which step, so it is part of the trajectory of every
# structures-on sphere run (and of resuming one), although the config hash does not contain it. Pinned here:
# sha256 of the int32 colour array (first 16 hex digits) and the colour count. A change of the seed, the
# iteration cap, the passes or NumPy's frozen RandomState stream changes them: re-pin DELIBERATELY (and the
# compiled structures digest of test_sphere_trajectory with it), as for the other digests (DESIGN §46.5).
COLOURING_PINS = {8: ("287d940ad91fb452", 11), 11: ("cf664d74586ccdc9", 11), 21: ("7446acc15afac274", 11)}


@pytest.mark.parametrize("nside", sorted(COLOURING_PINS))
def test_the_colouring_is_pinned(nside: int) -> None:
    import hashlib

    colour, k = geom.min_colouring(nside)
    assert (hashlib.sha256(colour.tobytes()).hexdigest()[:16], k) == COLOURING_PINS[nside]


def test_the_colouring_is_deterministic() -> None:
    a, ka = geom.min_colouring(8)
    geom.min_colouring.cache_clear()
    b, kb = geom.min_colouring(8)
    assert ka == kb and np.array_equal(a, b)


def test_check_colouring_rejects_a_wrong_colouring() -> None:
    colour, _ = geom.min_colouring(4)
    bad = colour.copy()
    p = int(geom.tables(4).touch[0, 0])
    bad[p] = bad[0]  # two touching pixels of one colour
    assert not geom.check_colouring(4, bad)


@pytest.mark.parametrize(("nside", "layers"), [(4, 5), (8, 8), (5, 2)])
def test_the_cells_of_one_phase_have_disjoint_neighbourhoods(nside: int, layers: int) -> None:
    """The point of the colouring: two cells that close in one step share no neighbour, so a cell loses at
    most one open neighbour per step. Counted by the explicit neighbourhood list, not by the colour graph."""
    idx = neighbour_index(nside, layers)
    phase, nph = ref_phase(layers, nside)
    flat = phase.reshape(-1)
    for ph in range(nph):
        members = np.flatnonzero(flat == ph)
        hits = np.zeros(layers * geom.npix(nside), np.int64)
        for c in members:
            hits[c] += 1  # the cell itself belongs to its box
            hits[idx[c][idx[c] >= 0]] += 1
        assert hits.max() <= 1, (ph, int(hits.max()))
    assert len(np.unique(flat)) == nph  # every phase is used


# --- the neighbourhood -----------------------------------------------------------------------------------


@pytest.mark.parametrize("nside", [3, 4, 5])
def test_the_touching_table_is_the_corner_sharing_relation(nside: int) -> None:
    adj = adjacency_brute(nside)
    touch = geom.tables(nside).touch
    for p in range(len(adj)):
        assert {int(q) for q in touch[p] if q >= 0} == adj[p]


def test_neighbour_counts_are_26_23_17_and_15() -> None:
    nside, layers = 8, 8
    n = fields_sph.n_neighbours(nside, layers)
    vals, counts = np.unique(n, return_counts=True)
    seven = int((geom.tables(nside).n_touch == 7).sum())
    assert seven == 24
    assert dict(zip(vals.tolist(), counts.tolist(), strict=True)) == {
        15: 2 * seven,
        17: 2 * (geom.npix(nside) - seven),
        23: (layers - 2) * seven,
        26: (layers - 2) * (geom.npix(nside) - seven),
    }
    idx = neighbour_index(nside, layers)  # the explicit list agrees
    assert np.array_equal((idx >= 0).sum(axis=1).reshape(n.shape), n)


@pytest.mark.parametrize(("nside", "layers"), [(4, 3), (8, 8)])
def test_nopen_equals_the_explicit_reference(nside: int, layers: int) -> None:
    cfg = _with_shell(cfg_of("structures.enabled=true", "structures.aperture=0.25"), nside, layers)
    thr = fields.mat_thr(cfg)
    rs = np.random.default_rng(3)
    for _ in range(6):
        m = (rs.random((layers, geom.npix(nside))) < 0.4) * rs.integers(thr, 4096, (layers, geom.npix(nside)))
        got = np.asarray(fields.mat_nopen(mx.array(m.astype(np.int32)), cfg))
        assert np.array_equal(got, ref_nopen(m, thr, nside))
        assert np.array_equal(structures.nopen_np(m, cfg), ref_nopen(m, thr, nside))


def _with_shell(cfg, nside: int, layers: int):
    """``cfg`` with another Nside and layer count; not validated (a unit-test shell)."""
    from dataclasses import replace

    return replace(cfg, world=replace(cfg.world, nside=nside, layers=layers))


# --- the update rule against the reference ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("aperture", "open_frac", "nside", "layers"),
    [(0.25, 0.5, 4, 5), (0.5, 0.3, 3, 4), (0.65, 0.4, 4, 3), (0.1, 0.8, 5, 2), (0.0, 0.5, 4, 4)],
)
def test_update_mat_equals_the_reference_bit_for_bit(aperture, open_frac, nside, layers) -> None:
    cfg = _with_shell(
        cfg_of(
            "structures.enabled=true", f"structures.aperture={aperture}", f"structures.open_frac={open_frac}"
        ),
        nside,
        layers,
    )
    shape = (layers, geom.npix(nside))
    rs = np.random.default_rng(5)
    for t in range(40):
        m = rs.integers(0, cfg.structures.cap + 1, shape) * (rs.random(shape) < 0.6)
        dep = rs.integers(0, 3000, shape) * (rs.random(shape) < 0.5)
        up = fields.update_mat(
            mx.array(m.astype(np.int32)), mx.array(dep.astype(np.int32)), mx.array(t, mx.int32), cfg
        )
        want, wasted, decayed, nopen = ref_update(m, dep, t, cfg)
        assert np.array_equal(np.asarray(up.mat), want), t
        assert int(up.wasted.item()) == wasted and int(up.decayed.item()) == decayed, t
        assert np.array_equal(np.asarray(up.nopen), nopen), t
        assert np.array_equal(structures.nopen_np(m, cfg), nopen), t


def test_the_mat_identity_holds_on_the_shell() -> None:
    cfg = _with_shell(cfg_of("structures.enabled=true", "structures.aperture=0.25"), 4, 5)
    shape = (5, geom.npix(4))
    rs = np.random.default_rng(8)
    m = mx.zeros(shape, mx.int32)
    for t in range(50):
        dep = mx.array((rs.integers(0, 2000, shape) * (rs.random(shape) < 0.3)).astype(np.int32))
        up = fields.update_mat(m, dep, mx.array(t, mx.int32), cfg)
        assert int(up.mat.sum().item()) == int(m.sum().item()) + int(dep.sum().item()) - int(
            up.wasted.item()
        ) - int(up.decayed.item())
        m = up.mat


# --- the anti-trapping invariant on the sphere graph -----------------------------------------------------


def _kmap(cfg) -> np.ndarray:
    return fields_sph.aperture_k(
        cfg.structures.aperture, fields_sph.n_neighbours(cfg.world.nside, cfg.world.layers)
    ).astype(np.int64)


CASES = [
    # (aperture, open_frac, deposit mode)
    (0.25, 0.5, "full"),
    (0.25, 0.5, "random"),
    (0.4, 0.3, "phase"),
    (0.375, 0.7, "half"),
    (0.65, 0.5, "boundary"),
    (0.65, 0.5, "full"),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: "-".join(map(str, c)))
def test_no_deposit_pattern_breaks_the_aperture_on_the_shell(case) -> None:
    """Whatever is deposited, every cell keeps ``ceil(aperture·n_nbr)`` open neighbours (the explicit
    reference, over 24 seven-neighbour pixels, the floor and lid layers and the 8 trivalent vertices); walls
    do form; the rule restricts the shape, it does not forbid closing."""
    aperture, open_frac, mode = case
    nside, layers = 4, 5
    cfg = _with_shell(
        cfg_of(
            "structures.enabled=true", f"structures.aperture={aperture}", f"structures.open_frac={open_frac}"
        ),
        nside,
        layers,
    )
    shape = (layers, geom.npix(nside))
    thr, kmap = fields.mat_thr(cfg), _kmap(cfg)
    phase, nph = ref_phase(layers, nside)
    rs = np.random.default_rng(11)
    m = np.zeros(shape, np.int64)
    closed_max = 0.0
    for t in range(500):
        if mode == "full":
            dep = np.full(shape, cfg.structures.cap)
        elif mode == "random":
            dep = rs.integers(0, 4096, shape) * (rs.random(shape) < 0.5)
        elif mode == "half":
            dep = np.where(rs.random(shape) < 0.9, 768, 0)
        elif mode == "phase":  # the most damaging: everything of the active phase at once
            dep = np.where(phase == t % nph, cfg.structures.cap, 0)
        else:  # boundary: only the floor and the lid layer
            dep = np.zeros(shape, np.int64)
            dep[0] = dep[-1] = cfg.structures.cap
        m = np.asarray(
            fields.update_mat(
                mx.array(m.astype(np.int32)), mx.array(dep.astype(np.int32)), mx.array(t, mx.int32), cfg
            ).mat
        ).astype(np.int64)
        nopen = ref_nopen(m, thr, nside)
        assert (nopen >= kmap).all(), (t, int((nopen - kmap).min()))
        closed_max = max(closed_max, float((m >= thr).mean()))
    assert closed_max > 0.0


def test_the_gate_refuses_a_closing_that_would_starve_a_neighbour_at_a_seven_neighbour_pixel() -> None:
    """Hand-made pattern at a seven-neighbour pixel (23 neighbours, 12 must stay open at aperture 0.5): with
    12 open neighbours a further closing next to it is refused in every phase of the cycle; with 13 it is
    allowed in exactly one phase, the target's own."""
    nside, layers = 4, 3
    cfg = _with_shell(
        cfg_of("structures.enabled=true", "structures.aperture=0.5", "structures.open_frac=0.5"),
        nside,
        layers,
    )
    npix = geom.npix(nside)
    tb = geom.tables(nside)
    p7 = int(np.flatnonzero(tb.n_touch == 7)[0])  # a seven-neighbour pixel
    thr, kmap = fields.mat_thr(cfg), _kmap(cfg)
    assert kmap[1, p7] == math.ceil(0.5 * 23 - 1e-9) == 12  # 12 of its 23 neighbours must stay open
    assert kmap[0, p7] == math.ceil(0.5 * 15 - 1e-9) == 8  # 8 of 15 on the floor layer
    nbrs = [int(i) for i in neighbour_index(nside, layers)[npix + p7] if i >= 0]
    assert len(nbrs) == 23
    n_phase = fields_sph.mat_tabs(cfg).n_phase

    def phases_that_close(n_open: int) -> int:
        m = np.zeros((layers, npix), np.int64)
        for i in nbrs[: len(nbrs) - n_open]:  # by hand: R2 does not apply to a hand-set material
            m.reshape(-1)[i] = cfg.structures.cap
        assert ref_nopen(m, thr, nside)[1, p7] == n_open
        target = nbrs[-1]  # an open neighbour of (1, p7)
        dep = np.zeros_like(m)
        dep.reshape(-1)[target] = cfg.structures.cap
        n = 0
        for t in range(n_phase):  # one closing cycle: every phase comes round once
            up = fields.update_mat(
                mx.array(m.astype(np.int32)), mx.array(dep.astype(np.int32)), mx.array(t, mx.int32), cfg
            )
            new = np.asarray(up.mat).astype(np.int64)
            n += int(new.reshape(-1)[target] >= thr)
            assert ref_nopen(new, thr, nside)[1, p7] >= n_open - 1 or n_open > 12
        return n

    assert phases_that_close(12) == 0
    assert phases_that_close(13) == 1


# --- the sensor and the effects --------------------------------------------------------------------------


def test_the_cover_is_the_maximum_over_the_six_face_neighbours() -> None:
    cfg = cfg_of(
        "structures.enabled=true",
        "structures.reach=0.5",
        "structures.shield_gain=0.5",
        "structures.barrier_gain=0.0",
        "structures.exit_ease=0.0",
        "combat.protect_floor=0.2",
    )
    nside, layers = cfg.world.nside, cfg.world.layers
    npix = geom.npix(nside)
    rs = np.random.default_rng(9)
    m = rs.integers(0, cfg.structures.cap + 1, (layers, npix)) * (rs.random((layers, npix)) < 0.3)
    assert "cover" in fields.mat_columns(cfg)
    tab = np.asarray(fields.mat_table(mx.array(m.astype(np.int32)), cfg))
    assert tab.shape == (layers * npix, 2)
    edge = geom.tables(nside).edge
    cap = cfg.structures.cap
    want = np.zeros((layers, npix))
    for k in range(layers):
        for p in range(npix):
            face = [m[k, q] for q in edge[p]] + [m[min(k + 1, layers - 1), p], m[max(k - 1, 0), p]]
            want[k, p] = max(m[k, p], 0.5 * max(face))
    np.testing.assert_allclose(tab[:, 1], (want / cap).reshape(-1), rtol=1e-6)
    np.testing.assert_allclose(tab[:, 0], (m / cap).reshape(-1), rtol=1e-6)


def test_the_material_gradient_points_up_the_slope() -> None:
    """A material that grows along a fixed direction ``n`` has a gradient whose tangent part is that
    direction's tangent projection (non-circular: the field is built from Cartesian cell centres)."""
    cfg = cfg_of(
        "structures.enabled=true",
        "structures.barrier_gain=1.0",
        "structures.exit_ease=0.5",
        "structures.shield_gain=0.0",
    )
    nside, layers = cfg.world.nside, cfg.world.layers
    assert fields.mat_columns(cfg) == ("here", "grad")
    cen = geom.pixel_centres(nside)
    n = np.array([0.3, -0.5, 0.8])
    n /= np.linalg.norm(n)
    m = np.tile(np.round(1500.0 * (1.0 + cen @ n)).astype(np.int64), (layers, 1))  # 0 .. 3000
    tab = np.asarray(fields.mat_table(mx.array(m.astype(np.int32)), cfg))
    g = tab[:, 1:4].reshape(layers, -1, 3)
    # the exact tangent gradient of the field: 1500·n projected onto the tangent plane, over r, times
    # field_cell / cap (the 'per cell' units of the box)
    k = 3
    r = geom.layer_radii(cfg.world.radius, cfg.world.height, layers)
    rc = np.cbrt(0.5 * (r[k] ** 3 + r[k + 1] ** 3))
    want = 1500.0 * (n[None, :] - (cen @ n)[:, None] * cen)
    want *= cfg.world.field_cell / cfg.structures.cap / rc
    got = g[k]
    cos = (got * want).sum(1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(want, axis=1) + 1e-30)
    assert np.median(cos) > 0.99 and np.percentile(cos, 5) > 0.9
    ratio = np.linalg.norm(got, axis=1) / (np.linalg.norm(want, axis=1) + 1e-30)
    assert 0.85 < np.median(ratio) < 1.15
    # a field that depends on the layer only has a purely radial gradient along the up of the cell
    m2 = np.tile((np.arange(layers) * 200)[:, None], (1, geom.npix(nside)))
    g2 = np.asarray(fields.mat_table(mx.array(m2.astype(np.int32)), cfg))[:, 1:4].reshape(layers, -1, 3)
    up = cen
    radial = (g2[3] * up).sum(1)
    assert radial.min() > 0
    tang = g2[3] - radial[:, None] * up
    assert np.linalg.norm(tang, axis=1).max() < 1e-4 * radial.min()


def test_the_barrier_slows_an_agent_in_the_material_by_the_factor() -> None:
    """From one state with the whole shell at the cap, a step with ``barrier_gain = 2`` moves every free agent
    by about a third of the plain step (overdamped: the drag divides the force; R1 bounds the factor by
    1 + gain)."""
    base = ["structures.enabled=true", "structures.aperture=0.25"]
    plain, walled = cfg_of(*base, "structures.barrier_gain=0.0"), cfg_of(*base, "structures.barrier_gain=2.0")
    s = init_state(plain)
    s = s._replace(mat=mx.full(s.mat.shape, plain.structures.cap, mx.int32))  # the same inputs for both
    assert plain.structures.cap == 4096
    from alife.step import make_step

    a, _ = make_step(plain, compile=False)(s)
    b, _ = make_step(walled, compile=False)(s)
    mx.eval(a, b)
    alive = to_numpy(s.alive)
    p0 = to_numpy(s.pos).astype(np.float64)
    da = np.linalg.norm(to_numpy(a.pos).astype(np.float64) - p0, axis=1)[alive]
    db = np.linalg.norm(to_numpy(b.pos).astype(np.float64) - p0, axis=1)[alive]
    keep = da > 1e-3
    ratio = np.median(db[keep] / da[keep])
    assert 1.0 / 3.0 * 0.9 < ratio < 1.0 / 3.0 * 1.1, ratio
    assert (db[keep] <= da[keep] * 1.05).all()


# --- configuration ---------------------------------------------------------------------------------------


def test_structures_are_ready_on_a_sphere_and_validate() -> None:
    assert "structures" in config.SPHERE_READY
    cfg = cfg_of("structures.enabled=true", "structures.aperture=0.4")
    assert cfg.structures.enabled and dims(cfg).sph
    assert fields_sph.mat_tabs(cfg).n_phase == 3 * geom.min_colouring(cfg.world.nside)[1]
    with pytest.raises(ConfigError, match=r"needs 19 open neighbours"):
        cfg_of("structures.enabled=true", "structures.aperture=0.7")
    with pytest.raises(ConfigError, match="inert"):
        cfg_of("structures.cap=1000")


def test_the_cell_count_of_the_validators_is_the_shells() -> None:
    n_cells = config.field_cells(cfg_of())
    assert n_cells == 8 * 768 == dims(cfg_of()).n_cells


# --- the eager step ---------------------------------------------------------------------------------------


def _builders(cfg, bias: float = 4.0):
    s = init_state(cfg)
    return set_output_bias(s, cfg, {dims(cfg).lay.OUT_BUILD: bias})


def test_forced_builders_never_trap_anyone_and_the_ledger_and_the_material_close() -> None:
    cfg = cfg_of("structures.enabled=true", "structures.open_frac=0.02", "structures.build_cost=2")
    s0 = _builders(cfg)
    closed = built = 0
    mat_prev = int(s0.mat.sum().item())
    for i, (prev, s, out) in enumerate(eager_steps(cfg, s0, 70)):
        c = counters(
            cfg,
            out,
            "BUILD_UNITS",
            "MAT_WASTED",
            "MAT_DECAYED",
            "MAT_TOTAL",
            "TRAPPED",
            "APERTURE_VIOLATIONS",
            "MAT_CLOSED",
        )
        assert c["TRAPPED"] == 0 and c["APERTURE_VIOLATIONS"] == 0, i
        assert c["MAT_TOTAL"] == mat_prev + c["BUILD_UNITS"] - c["MAT_WASTED"] - c["MAT_DECAYED"], i
        assert c["MAT_TOTAL"] == int(s.mat.sum().item())
        mat_prev = c["MAT_TOTAL"]
        closed, built = max(closed, c["MAT_CLOSED"]), built + c["BUILD_UNITS"]
        if i % 10 == 9:  # the 16-term ledger and every host check (incl. the aperture) every ten steps
            assert invariants.full_check(s, cfg, prev, out) == [], i
        assert int(to_numpy(s.diag)) & 0xFFFF == 0
    assert built > 0 and closed > 0  # walls did form


def test_the_build_cost_is_booked_as_action_heat() -> None:
    from alife.state import LedgerIdx

    cheap = cfg_of("structures.enabled=true", "structures.build_cost=0")
    dear = cfg_of("structures.enabled=true", "structures.build_cost=40")
    fn = []
    for cfg in (cheap, dear):
        s = _builders(cfg)
        (_, _, out), *_ = eager_steps(cfg, s, 1)
        fn.append((to_numpy(out.ledger), counters(cfg, out, "BUILD_UNITS")["BUILD_UNITS"]))
    (l0, u0), (l1, u1) = fn
    assert u0 == u1 > 0
    assert int(l1[LedgerIdx.HEAT_ACTION]) - int(l0[LedgerIdx.HEAT_ACTION]) == 40 * u1


def test_structures_off_is_a_dry_sphere_and_the_hash_is_unchanged() -> None:
    off = cfg_of()
    assert not off.structures.enabled
    assert config.config_hash(off) == config.config_hash(cfg_of("structures.enabled=false"))
    s = init_state(off)
    assert not hasattr(s, "mat")
    on = cfg_of("structures.enabled=true")
    assert config.config_hash(on) != config.config_hash(off)
    assert to_numpy(init_state(on).mat).shape == (8, 768)


def test_the_barrier_drag_is_bounded_by_the_gain_on_the_shell() -> None:
    cfg = cfg_of("structures.enabled=true", "structures.barrier_gain=3.0", "structures.exit_ease=0.0")
    here = mx.array(np.linspace(0, 1, 11), mx.float32)
    dm = np.asarray(physics.barrier_drag(mx.zeros((11, 3)), here, None, cfg))
    assert dm.min() >= 1.0 and dm.max() <= 4.0 + 1e-6


# --- the watchdogs of R2: the counter and the host check use the per-cell requirement (S3 review M2) -------


def _margin_material(cfg, layer: int, pix: int) -> tuple[np.ndarray, int]:
    """A hand-made material (R2 does not apply) that leaves the cell ``(layer, pix)`` with exactly ``k_c``
    open neighbours: its neighbours in the other layers are closed first, then the touching pixels of its own
    layer. Returns ``(m, k_c)``."""
    nside, layers = cfg.world.nside, cfg.world.layers
    npix = geom.npix(nside)
    thr = fields.mat_thr(cfg)
    kc = int(_kmap(cfg)[layer, pix])
    cell = layer * npix + pix
    nb = [int(i) for i in neighbour_index(nside, layers)[cell] if i >= 0]
    nb.sort(key=lambda i: (abs(i // npix - layer) == 0, i))  # the other layers first, then its own layer
    want_open = kc
    m = np.zeros((layers, npix), np.int64)
    for i in nb[: len(nb) - want_open]:
        m.reshape(-1)[i] = cfg.structures.cap
    assert ref_nopen(m, thr, nside).reshape(-1)[cell] == want_open
    return m, kc


def _drive(cfg, m: np.ndarray, cell: int = 0, alive: bool = True):
    """``structures.update`` on ``m`` with one inert row standing in ``cell``: returns the flows."""
    n = 4
    units = structures.actions.Build(mx.zeros((n,), mx.int32))
    fcell = mx.full((n,), cell, mx.int32)
    alive0 = mx.array([alive] + [False] * (n - 1))
    _, fl = structures.update(
        mx.array(m.astype(np.int32)), units, fcell, alive0, None, mx.array(3, mx.int32), cfg
    )
    return fl


@pytest.mark.parametrize("kind", ["floor", "seven-neighbour floor", "seven-neighbour interior", "interior"])
def test_the_violation_counter_and_the_host_check_sit_at_the_per_cell_margin(kind: str) -> None:
    """A cell at exactly ``nopen == k_c`` (the margin; below the scalar ``k = ceil(aperture·26)`` on the
    boundary layers and at the seven-neighbour pixels) is not a violation; one open neighbour fewer is
    exactly one, in the engine's counter and in the host check. A scalar threshold in either fails the
    first half."""
    cfg = cfg_of("structures.enabled=true", "structures.aperture=0.25")
    nside, layers = cfg.world.nside, cfg.world.layers
    npix = geom.npix(nside)
    seven = int(np.flatnonzero(geom.tables(nside).n_touch == 7)[0])
    six = int(np.flatnonzero(geom.tables(nside).n_touch == 8)[40])
    layer, pix = {
        "floor": (0, six),
        "seven-neighbour floor": (0, seven),
        "seven-neighbour interior": (3, seven),
        "interior": (3, six),
    }[kind]
    kc, k_scalar = int(_kmap(cfg)[layer, pix]), fields.mat_k(cfg)
    assert (kc < k_scalar) == (kind != "interior")  # the margin cells have fewer than the interior's 7
    thr, cell = fields.mat_thr(cfg), layer * npix + pix
    m, _ = _margin_material(cfg, layer, pix)
    ok = ref_nopen(m, thr, nside)
    others = (ok < _kmap(cfg)).reshape(-1)
    if others.any():  # the hand-made pattern must be valid elsewhere: close fewer of the far cells instead
        pytest.skip(f"the pattern violates {int(others.sum())} other cell(s) at {kind}")
    assert int(_drive(cfg, m).violations.item()) == 0 and structures.check({"mat": m}, cfg) == []
    # one open neighbour fewer: close one that leaves exactly this cell below its requirement
    open_nb = [int(i) for i in neighbour_index(nside, layers)[cell] if i >= 0 and m.reshape(-1)[i] < thr]
    for i in open_nb:
        m2 = m.copy()
        m2.reshape(-1)[i] = cfg.structures.cap
        bad = int((ref_nopen(m2, thr, nside) < _kmap(cfg)).sum())
        if bad == 1:
            break
    else:
        pytest.fail("no single closing leaves exactly one violation")
    assert int(_drive(cfg, m2).violations.item()) == 1
    problems = structures.check({"mat": m2}, cfg)
    assert len(problems) == 1 and "1 cell(s)" in problems[0]


def test_trapped_counts_an_agent_in_a_cell_with_no_open_neighbour_only() -> None:
    cfg = cfg_of("structures.enabled=true", "structures.aperture=0.0", "structures.barrier_gain=0.0")
    nside, layers = cfg.world.nside, cfg.world.layers
    npix = geom.npix(nside)
    layer, pix = 3, 100
    cell = layer * npix + pix
    nb = [int(i) for i in neighbour_index(nside, layers)[cell] if i >= 0]
    for n_open, want in ((0, 1), (1, 0), (2, 0)):
        m = np.zeros((layers, npix), np.int64)
        for i in nb[: len(nb) - n_open]:
            m.reshape(-1)[i] = cfg.structures.cap
        assert ref_nopen(m, fields.mat_thr(cfg), nside).reshape(-1)[cell] == n_open
        assert int(_drive(cfg, m, cell).trapped.item()) == want, n_open
    assert int(_drive(cfg, m, cell, alive=False).trapped.item()) == 0  # nobody there
