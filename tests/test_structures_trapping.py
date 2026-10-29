"""Anti-trapping geometry of built structures (DESIGN §38.4b): R1 bounded drag, R2 the aperture invariant,
R3 the easy exit, with the edge cases E6 (wind pinning), E8 (colouring on odd lattice sizes) and E9 (boundary
rows with 17 neighbours)."""

from __future__ import annotations

import itertools
from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from alife import fields, invariants, physics, structures
from alife import weather as W
from alife.config import ConfigError, apply_overrides, dims, load_config
from alife.spatial import Neighbors
from alife.state import to_numpy
from structures_helpers import builders, cfg_for, run
from weather_helpers import STRONG, spun_ws, storm2d_config, zero_actions

# --- a numpy reference of the update rule (the aperture probe of the spec, with this engine's colouring) ---


def _offsets(nd: int):
    return [o for o in itertools.product((-1, 0, 1), repeat=nd) if any(o)]


def _shift(a: np.ndarray, o, torus_axes):
    out, ex = a, np.ones(a.shape, bool)
    for ax, dlt in enumerate(o):
        if dlt == 0:
            continue
        out, ex = np.roll(out, -dlt, axis=ax), np.roll(ex, -dlt, axis=ax)
        if ax not in torus_axes:
            sl = [slice(None)] * a.ndim
            sl[ax] = slice(a.shape[ax] - 1, None) if dlt == 1 else slice(0, 1)
            ex[tuple(sl)] = False
    return out, ex


def ref_nopen(m, thr, nd3):
    torus = {1, 2} if nd3 else {0, 1}
    op = (m < thr).astype(np.int64)
    n = np.zeros(m.shape, np.int64)
    for o in _offsets(m.ndim):
        v, ex = _shift(op, o, torus)
        n += v * ex
    return n


def ref_boxmin(a, nd3):
    torus = {1, 2} if nd3 else {0, 1}
    r = a.copy()
    for o in _offsets(a.ndim):
        v, ex = _shift(a, o, torus)
        r = np.minimum(r, np.where(ex, v, r))
    return r


def ref_colour(shape, fg, nd3):
    px = next(p for p in range(3, fg + 1) if fg % p == 0)
    idx = np.indices(shape)
    if nd3:
        return idx[2] % px + px * (idx[1] % px) + px * px * (idx[0] % 3), 3 * px * px
    return idx[1] % px + px * (idx[0] % px), px * px


def ref_update(m, dep, t, cfg):
    sc = cfg.structures
    d3 = m.ndim == 3
    thr, k = fields.mat_thr(cfg), fields.mat_k(cfg)
    n = ref_nopen(m, thr, d3)
    pre = m + dep
    m1 = np.minimum(pre, sc.cap)
    if k > 0:
        slack = ref_boxmin(n, d3) >= k + 1
        code, nph = ref_colour(m.shape, m.shape[-1], d3)
        active = code == (t % nph)
        want = (m < thr) & (m1 >= thr)
        m1 = np.where(want & ~(active & slack), thr - 1, m1)
    loss = (m1 * sc.decay_q12 + 4095) >> 12
    return m1 - loss, int((pre - m1).sum()), int(loss.sum()), n


def with_world(cfg, fg: int, nd: int, fz: int = 6, **struct):
    """``cfg_for`` re-sized to a lattice side ``fg`` (field cell 4); not validated (a unit-test lattice)."""
    base = cfg_for(nd)
    w = replace(base.world, size=4.0 * fg, **({"height": 4.0 * fz} if nd == 3 else {}))
    return replace(base, world=w, structures=replace(base.structures, **struct))


# --- 3. the numpy reference equals the engine, bit for bit ----------------------------------------


@pytest.mark.parametrize(
    ("nd", "fg", "fz", "aperture", "open_frac"),
    [
        (2, 12, 1, 0.25, 0.5),
        (2, 9, 1, 0.5, 0.3),
        (2, 10, 1, 0.1, 0.8),
        (3, 12, 6, 0.25, 0.5),
        (3, 15, 5, 0.65, 0.4),
    ],
)
def test_the_reference_equals_update_mat_bit_for_bit(nd, fg, fz, aperture, open_frac) -> None:
    cfg = with_world(cfg_for(nd), fg, nd, fz, aperture=aperture, open_frac=open_frac)
    d = dims(cfg)
    rs = np.random.default_rng(5)
    for t in range(60):
        m = rs.integers(0, cfg.structures.cap + 1, d.field_shape) * (rs.random(d.field_shape) < 0.6)
        dep = rs.integers(0, 3000, d.field_shape) * (rs.random(d.field_shape) < 0.5)
        up = fields.update_mat(
            mx.array(m.astype(np.int32)), mx.array(dep.astype(np.int32)), mx.array(t, mx.int32), cfg
        )
        want, wasted, decayed, nopen = ref_update(m, dep, t, cfg)
        assert np.array_equal(np.asarray(up.mat), want), t
        assert int(up.wasted.item()) == wasted and int(up.decayed.item()) == decayed, t
        assert np.array_equal(np.asarray(up.nopen), nopen), t
        assert np.array_equal(structures.nopen_np(m, cfg), nopen), t


# --- 2. the invariant under adversarial deposits (property test) ----------------------------------


def _valid_state(rs, cfg, p_closed=0.35):
    """A random material with ``nopen >= k`` everywhere (closed cells are opened until it holds)."""
    d, thr, k = dims(cfg), fields.mat_thr(cfg), fields.mat_k(cfg)
    m = np.where(
        rs.random(d.field_shape) < p_closed, rs.integers(thr, cfg.structures.cap + 1, d.field_shape), 0
    )
    for _ in range(50):
        bad = ref_nopen(m, thr, d.d3) < k
        if not bad.any():
            return m
        near = np.zeros(m.shape, bool)
        for o in [(0,) * m.ndim, *_offsets(m.ndim)]:
            v, ex = _shift(bad, o, {1, 2} if d.d3 else {0, 1})
            near |= v & ex
        m = np.where(near, 0, m)
    raise AssertionError("could not build a valid random state")


CASES = [
    # (nd, fg, fz, aperture, open_frac, deposit mode, start)
    (2, 12, 1, 0.25, 0.5, "full", "open"),
    (2, 12, 1, 0.25, 0.5, "random", "valid"),
    (2, 9, 1, 0.5, 0.3, "colour", "open"),
    (2, 10, 1, 0.375, 0.7, "half", "valid"),
    (2, 15, 1, 0.125, 0.5, "full", "valid"),
    (3, 12, 6, 0.25, 0.5, "full", "open"),
    (3, 12, 6, 0.25, 0.5, "colour", "valid"),
    (3, 15, 5, 0.4, 0.6, "random", "open"),
    (
        3,
        12,
        6,
        0.65,
        0.5,
        "boundary",
        "open",
    ),  # k = 17: the floor and surface cells have exactly 17 neighbours
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: "-".join(map(str, c)))
def test_no_deposit_pattern_breaks_the_aperture(case) -> None:
    nd, fg, fz, aperture, open_frac, mode, start = case
    cfg = with_world(cfg_for(nd), fg, nd, fz, aperture=aperture, open_frac=open_frac)
    d, thr, k = dims(cfg), fields.mat_thr(cfg), fields.mat_k(cfg)
    rs = np.random.default_rng(11)
    m = _valid_state(rs, cfg) if start == "valid" else np.zeros(d.field_shape, np.int64)
    fn = mx.compile(
        lambda m, dep, t: fields.update_mat(m, dep, t, cfg).mat
    )  # compiled: the rolls are built once
    code, nph = ref_colour(d.field_shape, fg, d.d3)
    closed_max = 0.0
    for t in range(2000):
        if mode == "full":
            dep = np.full(d.field_shape, cfg.structures.cap)
        elif mode == "random":
            dep = rs.integers(0, 4096, d.field_shape) * (rs.random(d.field_shape) < 0.5)
        elif mode == "half":
            dep = np.where(rs.random(d.field_shape) < 0.9, 768, 0)
        elif mode == "colour":  # the most damaging: everything of the active colour at once
            dep = np.where(code == t % nph, cfg.structures.cap, 0)
        else:  # boundary: only the floor and the surface layer
            dep = np.zeros(d.field_shape, np.int64)
            dep[0] = dep[-1] = cfg.structures.cap
        m_new = np.asarray(
            fn(mx.array(m.astype(np.int32)), mx.array(dep.astype(np.int32)), mx.array(t, mx.int32))
        )
        m = m_new.astype(np.int64)
        nopen = ref_nopen(m, thr, d.d3)
        assert nopen.min() >= k, (t, int(nopen.min()), k)
        closed_max = max(closed_max, float((m >= thr).mean()))
    if mode == "boundary":  # k = 17: a floor or surface cell must keep all 17 neighbours open, so none closes
        assert closed_max == 0.0
    else:
        assert closed_max > 0.0  # the rule restricts the shape; it does not forbid closing


def test_the_invariant_also_holds_when_decay_is_off_the_table() -> None:
    """decay_q12 = 1 (the slowest allowed): material accumulates for 2000 steps, still no cell loses k."""
    cfg = with_world(cfg_for(2), 12, 2, decay_q12=1)
    d, thr, k = dims(cfg), fields.mat_thr(cfg), fields.mat_k(cfg)
    m = mx.zeros(d.field_shape, mx.int32)
    dep = mx.full(d.field_shape, cfg.structures.cap, mx.int32)
    fn = mx.compile(lambda m, t: fields.update_mat(m, dep, t, cfg).mat)
    for t in range(2000):
        m = fn(m, mx.array(t, mx.int32))
        assert ref_nopen(np.asarray(m).astype(np.int64), thr, False).min() >= k, t


# --- 1. by construction in the engine (forced builders), with and without wind (E6) ---------------


@pytest.mark.parametrize("nd", [2, 3])
def test_forced_builders_never_trap_anyone(nd: int) -> None:
    cfg = cfg_for(nd)
    t_closed = 0
    for t, (prev, s, out, row) in enumerate(run(cfg, builders(cfg), 1500)):  # noqa: B007
        assert row["TRAPPED"] == 0 and row["APERTURE_VIOLATIONS"] == 0, t
        t_closed = max(t_closed, row["MAT_CLOSED"])
    assert t_closed > 0  # walls did form
    assert invariants.full_check(s, cfg, prev, out) == []


def test_forced_builders_in_a_storm_never_trap_anyone() -> None:
    """E6: strong storm wind (hazard and bond break active) pushes the agents around while they build."""
    cfg = storm2d_config("structures.enabled=true", *STRONG, "weather.storm_p=0.0005")
    ws = spun_ws(cfg)
    s0 = builders(cfg)._replace(wx=W.pack(ws))
    t_closed = 0
    for t, (_, s, _, row) in enumerate(run(cfg, s0, 500)):  # noqa: B007
        assert row["TRAPPED"] == 0 and row["APERTURE_VIOLATIONS"] == 0, t
        t_closed = max(t_closed, row["MAT_CLOSED"])
    assert t_closed > 0 and int(to_numpy(s.diag)) & 0xFFFF == 0


# --- 4. escape time from a sealed pocket (physics level) ------------------------------------------


def _ring(cfg, thickness: int, centre=(16, 16)):
    """A 3x3 open pocket at ``centre`` inside a ring of CLOSED cells of the given thickness (hand-set, so R2
    does not apply)."""
    d = dims(cfg)
    m = np.zeros(d.field_shape, np.int32)
    cy, cx = centre
    for y in range(d.fg):
        for x in range(d.fg):
            r = max(abs(x - cx), abs(y - cy))
            if 2 <= r <= 1 + thickness:
                m[y, x] = cfg.structures.cap
    return mx.array(m)


def _alone(n: int, k: int, radius: float) -> Neighbors:
    """Nobody has a neighbour (2D): every slot invalid, pointing at the row itself."""
    zero = mx.array(0, mx.int32)
    idx = mx.broadcast_to(mx.arange(n, dtype=mx.int32)[:, None], (n, k))
    return Neighbors(
        idx, mx.zeros((n, k), mx.bool_), mx.zeros((n, k, 2), mx.float32), mx.full((n, k), radius), zero, zero
    )


def _walk(cfg, mat, pos0, thrust, wind=0.0, decay=False, steps=4000, stop_x=None):
    """Steps until the agent's x passes ``stop_x`` (or None): one agent through the step's physics."""
    n = 1
    cfg1 = cfg
    pos = mx.array([pos0], mx.float32)
    nbr = _alone(n, cfg.sensing.k_neighbors, cfg.sensing.radius)
    act = zero_actions(n)._replace(thrust=mx.array([thrust], mx.float32))
    surv = mx.ones((n,), mx.bool_)
    z = mx.zeros((n,))
    wind_f = mx.array([[wind, 0.0]], mx.float32) if wind else None
    for t in range(steps):
        fcell = fields.cell_index(pos, cfg1)
        sn = structures.sense(mat, fcell, cfg1)
        F = physics.forces(pos, z, act, nbr, surv, mx.full((n, 4), -1, mx.int32), cfg1, wind=wind_f)
        dm = structures.drag_mult(F, sn, cfg1)
        pos, _, _, _ = physics.integrate(pos, z, F, z, z, surv, cfg1, drag_mult=dm)
        if decay:
            mat = fields.update_mat(mat, mx.zeros_like(mat), mx.array(t, mx.int32), cfg1).mat
        if float(pos[0, 0].item()) > stop_x:
            return t + 1
    return None


@pytest.mark.parametrize("thickness", [1, 3])
def test_a_sealed_pocket_is_left_in_the_analytic_time(thickness: int) -> None:
    cfg = apply_overrides(cfg_for(2), ["structures.exit_ease=0", "structures.aperture=0"])
    fc, bg, drag = cfg.world.field_cell, cfg.structures.barrier_gain, cfg.physics.drag
    thrust = 1.0
    v_free = cfg.physics.thrust_max * thrust / drag
    centre = (16, 16)
    x0 = (centre[1] + 0.5) * fc
    x_out = (centre[1] + 2 + thickness) * fc + 0.01  # first x beyond the ring
    # the pocket is 3 cells wide: the agent starts mid-cell and has 1.5 cells to its edge
    expected = (1.5 * fc + thickness * fc * (1 + bg)) / v_free
    got = _walk(cfg, _ring(cfg, thickness, centre), (x0, (centre[0] + 0.5) * fc), thrust, stop_x=x_out)
    assert got is not None and abs(got - expected) <= 0.1 * expected, (got, expected)


def test_decay_shortens_the_escape() -> None:
    cfg = apply_overrides(
        cfg_for(2), ["structures.exit_ease=0", "structures.aperture=0", "structures.decay_q12=64"]
    )
    fc = cfg.world.field_cell
    kw = dict(pos0=(16.5 * fc, 16.5 * fc), thrust=1.0, stop_x=(16 + 2 + 3) * fc + 0.01)
    ring = _ring(cfg, 3)
    plain, decaying = _walk(cfg, ring, **kw), _walk(cfg, ring, decay=True, **kw)
    assert plain is not None and decaying is not None and decaying < plain


def test_a_single_closed_cell_is_crossed_within_200_steps_at_full_barrier() -> None:
    cfg = apply_overrides(cfg_for(2), ["structures.barrier_gain=3", "structures.aperture=0"])
    fc = cfg.world.field_cell
    m = np.zeros(dims(cfg).field_shape, np.int32)
    m[16, 17] = cfg.structures.cap
    for ease in (0.0, 0.75):
        c = apply_overrides(cfg, [f"structures.exit_ease={ease}"])
        got = _walk(c, mx.array(m), (17 * fc - 0.5, 16.5 * fc), 1.0, stop_x=18 * fc + 0.01)
        assert got is not None and got <= 200, (ease, got)


def test_the_easy_exit_never_makes_leaving_slower() -> None:
    plain = apply_overrides(cfg_for(2), ["structures.exit_ease=0", "structures.aperture=0"])
    ease = apply_overrides(plain, ["structures.exit_ease=0.75"])
    fc = plain.cfg if False else plain.world.field_cell
    ring = _ring(plain, 3)
    for x0 in ((16 + 2.5) * fc, (16 + 3.5) * fc, (16 + 4.5) * fc):  # inside the wall, pushed outward (+x)
        a = _walk(plain, ring, (x0, 16.5 * fc), 1.0, stop_x=(16 + 5) * fc + 0.01)
        b = _walk(ease, ring, (x0, 16.5 * fc), 1.0, stop_x=(16 + 5) * fc + 0.01)
        assert a is not None and b is not None and b <= a, (x0, a, b)
    # a turned-back organism inside the wall (pushed toward the pocket) leaves no later than the plain drag
    # (the thrust is negative, so the stop is a crossing to the left; mirror the walk by x -> -x)
    mir = np.asarray(ring)[:, ::-1].copy()
    for x0 in ((16 + 0.5) * fc + 0.0,):
        a = _walk(plain, mx.array(mir), (x0, 16.5 * fc), 1.0, stop_x=(16 + 2) * fc)
        b = _walk(ease, mx.array(mir), (x0, 16.5 * fc), 1.0, stop_x=(16 + 2) * fc)
        assert a is not None and b is not None and b <= a


def test_wind_that_pushes_into_the_wall_cannot_pin_an_agent() -> None:
    """E6: the easy exit applies to the ACTUAL push (thrust plus wind), not to the intended motion.

    Thrust tries to go back (-x), the wind overpowers it (+x): the net push crosses the wall and, at the
    outer surface, is eased as an exit; a rule that read the thrust alone would not ease it.
    """
    cfg = apply_overrides(cfg_for(2), ["structures.aperture=0"])
    fc = cfg.world.field_cell
    ring = _ring(cfg, 3)  # a thick wall: a one-cell membrane has no central-difference gradient
    thrust, wind = -0.2, 0.05  # force = 0.1*(-0.2) + 0.05 = +0.03 in x
    start = (16.5 * fc, 16.5 * fc)
    stop = (16 + 5) * fc + 0.01
    net = _walk(cfg, ring, start, thrust, wind=wind, stop_x=stop)
    plain = _walk(
        apply_overrides(cfg, ["structures.exit_ease=0"]), ring, start, thrust, wind=wind, stop_x=stop
    )
    assert net is not None and plain is not None and net < plain
    # the same drag function at the outer surface (the gradient points back into the wall, -x):
    sn = structures.Sense(mx.array([1.0]), None, mx.array([[-0.5, 0.0]]))
    f_net, f_thrust = mx.array([[0.03, 0.0]]), mx.array([[-0.02, 0.0]])
    out = float(structures.drag_mult(f_net, sn, cfg)[0].item())
    inn = float(structures.drag_mult(f_thrust, sn, cfg)[0].item())
    assert out < inn == 1.0 + cfg.structures.barrier_gain


# --- 5. R1: the bound -----------------------------------------------------------------------------


def test_drag_mult_is_bounded_by_one_plus_barrier_gain_for_any_material() -> None:
    rs = np.random.default_rng(3)
    for bg in (0.5, 1.0, 3.0):
        for ease in (0.0, 0.5, 1.0):
            cfg = apply_overrides(
                cfg_for(3), [f"structures.barrier_gain={bg}", f"structures.exit_ease={ease}"]
            )
            n = 4000
            here = mx.array(rs.random(n).astype(np.float32))
            grad = mx.array(rs.normal(0, 0.3, (n, 3)).astype(np.float32))
            force = mx.array(rs.normal(0, 0.1, (n, 3)).astype(np.float32))
            force = mx.where(mx.array(rs.random(n) < 0.1)[:, None], 0.0, force)  # some zero pushes
            dm = np.asarray(physics.barrier_drag(force, here, grad, cfg))
            assert dm.min() >= 1.0 - 1e-6 and dm.max() <= 1.0 + bg + 1e-5


def test_validation_rejects_a_barrier_above_three() -> None:
    with pytest.raises(ConfigError, match="barrier_gain"):
        apply_overrides(cfg_for(2), ["structures.barrier_gain=3.5"])
    with pytest.raises(ConfigError, match="decay_q12"):
        apply_overrides(cfg_for(2), ["structures.decay_q12=0"])


# --- 6. R3: the direction of the easy exit --------------------------------------------------------


def test_the_easy_exit_discounts_only_motion_out_of_the_material() -> None:
    cfg = cfg_for(2)
    sc = cfg.structures
    n = 4
    here = mx.full((n,), 1.0)
    g = 0.5  # |grad| of m/cap per cell: s = g/(g + g0)
    grad = mx.array([[g, 0.0]] * n)  # the material increases toward +x
    s_conf = g / (g + physics.BARRIER_G0)
    # moving +x is into the material, -x is out of it, +y is across, zero push is neither
    force = mx.array([[0.1, 0.0], [-0.1, 0.0], [0.0, 0.1], [0.0, 0.0]])
    dm = np.asarray(physics.barrier_drag(force, here, grad, cfg))
    full = 1.0 + sc.barrier_gain
    assert np.allclose(dm[0], full) and np.allclose(dm[2], full) and np.allclose(dm[3], full)
    assert np.isclose(dm[1], 1.0 + sc.barrier_gain * (1.0 - sc.exit_ease * s_conf), rtol=1e-6)
    # a plateau (zero gradient) pays the plain barrier whatever the direction
    flat = np.asarray(physics.barrier_drag(force, here, mx.zeros((n, 2)), cfg))
    assert np.allclose(flat, full)
    # exit_ease = 0 is the plain barrier everywhere
    c0 = apply_overrides(cfg, ["structures.exit_ease=0"])
    assert np.allclose(np.asarray(physics.barrier_drag(force, here, grad, c0)), full)


# --- E8: the colouring on lattice sizes -----------------------------------------------------------


def _is_prime(n: int) -> bool:
    return n >= 2 and all(n % q for q in range(2, int(n**0.5) + 1))


@pytest.mark.parametrize("nd", [2, 3])
def test_e8_the_colouring_exists_exactly_for_composite_lattice_sides(nd: int) -> None:
    """Every lattice side from 1 to 130: ``mat_phases`` accepts exactly the composites, and the colouring
    has cells of one colour at least 3 apart (Chebyshev, on the torus in x and y)."""
    for fg in range(1, 131):
        ok = fg >= 4 and not _is_prime(fg)
        if not ok:
            with pytest.raises(ValueError):
                fields.mat_phases(fg, nd)
            continue
        px, nph = fields.mat_phases(fg, nd)
        assert fg % px == 0 and px >= 3 and nph == px * px * (3 if nd == 3 else 1)
        if fg > 24:
            continue  # the pairwise distance check below on the small lattices is the proof by example
        cfg = with_world(cfg_for(nd), fg, nd, fz=5)
        col = np.asarray(fields.mat_colour(cfg))
        shape = col.shape
        for a, b in itertools.combinations(np.ndindex(*shape), 2):
            if col[a] != col[b]:
                continue
            dist = max(
                (abs(i - j) if (nd == 3 and ax == 0) else min(abs(i - j), shape[ax] - abs(i - j)))
                for ax, (i, j) in enumerate(zip(a, b, strict=True))
            )
            assert dist >= 3, (fg, a, b)


def test_e8_validation_refuses_a_prime_lattice_with_the_aperture_on() -> None:
    # FG = 11 is prime; the aperture is the only thing that needs the colouring
    with pytest.raises(ConfigError, match="colouring"):
        load_config(
            preset="tiny", overrides=["structures.enabled=true", "world.size=44", "structures.aperture=0.25"]
        )
    load_config(
        preset="tiny", overrides=["structures.enabled=true", "world.size=44", "structures.aperture=0"]
    )


# --- E9: boundary rows with 17 neighbours ---------------------------------------------------------


def test_e9_the_floor_and_surface_have_seventeen_neighbours_and_the_aperture_uses_the_true_count() -> None:
    cfg = cfg_for(3)
    d = dims(cfg)
    nopen = np.asarray(fields.mat_nopen(mx.zeros(d.field_shape, mx.int32), cfg))
    assert (nopen[0] == 17).all() and (nopen[-1] == 17).all() and (nopen[1:-1] == 26).all()
    assert (structures.nopen_np(np.zeros(d.field_shape, np.int64), cfg) == nopen).all()
    cfg2 = cfg_for(2)
    assert (np.asarray(fields.mat_nopen(mx.zeros(dims(cfg2).field_shape, mx.int32), cfg2)) == 8).all()
    # k = ceil(0.25 * 26) = 7 <= 17, and the largest allowed aperture is 17/26
    assert fields.mat_k(cfg) == 7 and fields.mat_k(cfg2) == 2
    with pytest.raises(ConfigError, match="open neighbours"):
        apply_overrides(cfg, ["structures.aperture=0.7"])  # ceil(18.2) = 19 > 17
    assert fields.mat_k(apply_overrides(cfg, ["structures.aperture=0.65"])) == 17
