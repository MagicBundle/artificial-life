"""Symmetry goldens (DESIGN §48): an instrument-qualification suite.

Each test applies a transformation the engine claims is irrelevant to a state, runs the step, and compares
with the untransformed run (``step . g = g . step``); the table is in
docs/brainstorming-Fable-November.md §1.
``g = e`` is the identity matrix. The verdicts, as measured (CPU device, float32, eager steps):

- storage-row permutation: EXACT (bitwise, every field, every counter) once the three row-indexed random draws
  are keyed by uid; with the engine's own draws it is BROKEN BY DESIGN (the draws follow the row);
- torus translation, box: WITHIN TOLERANCE (float32 rounding, 3e-5) until a quantisation flips;
- mirror, box: WITHIN TOLERANCE, with the brain's handedness carried by its weights;
- hidden-unit sign flip of the Elman RNN: EXACT; permutation: WITHIN TOLERANCE (summation order);
- HEALPix quarter turn about the polar axis: EXACT; half turns about an equatorial axis: WITHIN TOLERANCE;
- tag rotation (kin cue = exp(-|dtag|^2 / s^2)): WITHIN TOLERANCE (a rotation preserves distances).

Every test runs on the CPU device (a stream context), whatever the process default is: the claims are about
the engine's arithmetic, and Metal's float32 reductions are not promised to be reproducible across layouts.
"""

from __future__ import annotations

from collections.abc import Iterator

import mlx.core as mx
import numpy as np
import pytest

import symmetry_helpers as sh
from alife.config import Config, dims, load_config
from alife.state import CounterIdx, State, init_state, to_numpy

PRESETS = ("tiny", "tiny3d", "tiny3s")
BOX = ("tiny", "tiny3d")
# The float deviation allowed in a "within tolerance" row. Measured: 8e-5 for positions near 100 (an ulp
# is 7.6e-6), 8e-6 for everything else; a real defect (a wrong sign, a missed field) is of order 1.
ATOL = 2e-4
NO_BIRTHS = ("repro.min_energy=4000000",)  # mutation is not covariant under a change of weights or tags


@pytest.fixture(scope="module", autouse=True)
def on_cpu() -> Iterator[None]:
    with mx.stream(mx.cpu):
        yield


def cfg_of(preset: str, *extra: str) -> Config:
    return load_config(preset=preset, overrides=["brain.weight_dtype=float32", *extra])


@pytest.fixture(scope="module")
def cfgs() -> dict[str, Config]:
    return {p: cfg_of(p) for p in PRESETS}


@pytest.fixture(scope="module")
def cfgs_nb() -> dict[str, Config]:
    return {p: cfg_of(p, *NO_BIRTHS) for p in PRESETS}


# --- 1. storage-row permutation -------------------------------------------------------------------------


def perm_of(kind: str, n: int) -> np.ndarray:
    return np.arange(n)[::-1].copy() if kind == "reverse" else sh.random_perm(n, 3)


@pytest.mark.parametrize(
    ("preset", "kind", "steps"),
    [("tiny", "random", 40), ("tiny", "reverse", 40), ("tiny3d", "random", 40), ("tiny3s", "random", 30)],
)
def test_row_permutation_is_exact(cfgs, preset: str, kind: str, steps: int) -> None:
    """A permutation of the rows (alive and dead alike), bonds and pending slots re-indexed, gives the same
    population in uid order, bit for bit, in every field, counter and ledger entry: conflict resolution (the
    bond handshake, the birth lanes, the harvest and combat scatters, the neighbour search) does not depend on
    the storage order. Needs the identity-keyed draws (``sh.covariant_rng``); see the next test."""
    cfg = cfgs[preset]
    perm = perm_of(kind, dims(cfg).n)
    tr = sh.lockstep(cfg, steps, forward=lambda s: sh.permute_rows(s, cfg, perm), covariant=True)
    assert tr.first_flip() is None, tr.ints[tr.first_flip() - 1] if tr.first_flip() else None
    assert tr.exact_through(steps, 0.0)
    # the scene is not vacuous: births under lane contention, bonds, combat
    c = tr.counters
    assert tr.births() > 20
    assert c[:, CounterIdx.BIRTHS_BLOCKED].sum() > 0 and c[:, CounterIdx.BONDS_FORMED].sum() > 0
    assert c[:, CounterIdx.ATTACKS].sum() > 0


@pytest.mark.slow
@pytest.mark.parametrize("preset", BOX)
def test_row_permutation_long(cfgs, preset: str) -> None:
    cfg = cfgs[preset]
    perm = sh.random_perm(dims(cfg).n, 11)
    tr = sh.lockstep(cfg, 200, forward=lambda s: sh.permute_rows(s, cfg, perm), covariant=True)
    assert tr.first_flip() is None and tr.exact_through(200, 0.0)


def test_row_permutation_is_broken_by_design_without_covariant_draws(cfgs) -> None:
    """With the engine's own draws the permuted run is not the permuted run: the sensing noise (S5), the bond
    priority (S9) and the birth priority (S16) are drawn per storage row, so a relabelling moves them from
    agent to agent. Attribution, on ``tiny``: the noise alone breaks step 1; with the noise keyed by uid the
    runs agree until the first step with two or more birth requests (the lane order); keying the birth order
    too leaves only the bond priority, which changes no state bit in 60 steps and only the proposer
    attribution (the ``KIN_BOND_*`` counters: in a mutual handshake the priority picks which end is recorded
    as the proposer, the bond is the same); keying all three is exact (the previous test)."""
    cfg = cfgs["tiny"]
    perm = sh.random_perm(dims(cfg).n, 3)
    fwd = lambda s: sh.permute_rows(s, cfg, perm)  # noqa: E731
    raw = sh.lockstep(cfg, 3, forward=fwd)
    assert raw.first_flip() == 1
    noise = sh.lockstep(cfg, 30, forward=fwd, covariant=True, covariant_kw={"bond": False, "birth": False})
    flip = noise.first_flip()
    assert flip is not None
    req = noise.counters[:, CounterIdx.BIRTH_REQUESTS]
    assert flip >= next(t + 1 for t, r in enumerate(req) if r >= 2)
    assert noise.exact_through(flip - 1, 0.0)
    bond_raw = sh.lockstep(cfg, 60, forward=fwd, covariant=True, covariant_kw={"bond": False})
    assert not any(bond_raw.ints) and max(bond_raw.fdev) == 0.0
    assert (
        bond_raw.first_flip() is not None
    )  # a counter differs (KIN_BOND_SUM and KIN_BOND_BASE), no state bit


# --- 2. torus translation (box) ---------------------------------------------------------------------------

# MAX_OCCUPANCY counts agents per neighbour-search hash cell, whose edges (L / G_h) do not line up with a
# translation by whole field cells: the counter is not translation invariant by construction (nothing reads
# it).
OCCUPANCY = (CounterIdx.MAX_OCCUPANCY,)


@pytest.mark.parametrize("preset", BOX)
@pytest.mark.parametrize("shift", [(8, -12), (5, 7)])
def test_translation_within_tolerance(cfgs, preset: str, shift: tuple[int, int]) -> None:
    """Shifting every position, the patch centres and (rolled by whole cells) the resource and chemicals by
    ``shift`` field cells is the same trajectory shifted, to float32 rounding (positions to 3e-5 over 12
    steps); the integer state, the counters and the ledger are identical. No stage reads an absolute
    coordinate."""
    cfg = cfgs[preset]
    g = lambda s: sh.translate(s, cfg, shift)  # noqa: E731
    tr = sh.lockstep(cfg, 12, forward=g, view=g, torus=cfg.world.size, skip_counters=OCCUPANCY)
    assert tr.exact_through(12, ATOL), (tr.ints, tr.fdev)


def test_translation_breaks_by_quantisation_not_by_a_stage(cfgs) -> None:
    """Past 12 steps an integer eventually differs (a harvest, an action cost or a regeneration quantum: a
    float32 value of the shifted world falls on the other side of a ``floor``). Verdict: the step before the
    first flip the worlds still agreed to ``ATOL``, so the flip is rounding noise crossing a quantisation
    boundary, not a stage that reads absolute coordinates."""
    cfg = cfgs["tiny"]
    g = lambda s: sh.translate(s, cfg, (8, -12))  # noqa: E731
    tr = sh.lockstep(cfg, 40, forward=g, view=g, torus=cfg.world.size, skip_counters=OCCUPANCY)
    flip = tr.first_flip()
    assert flip is not None and flip > 12
    assert tr.exact_through(flip - 1, ATOL)


def test_translation_needs_the_fields_moved(cfgs) -> None:
    """Negative control: translating the agents but not the fields is not a symmetry."""
    cfg = cfgs["tiny"]

    def agents_only(s: State) -> State:
        moved = sh.translate(s, cfg, (8, -12))
        return s._replace(pos=moved.pos, patch_pos=moved.patch_pos)

    tr = sh.lockstep(cfg, 4, forward=agents_only, view=lambda s: sh.translate(s, cfg, (8, -12)), torus=128.0)
    assert not tr.exact_through(4, ATOL)


# --- 3. mirror image (box) ----------------------------------------------------------------------------------


def mirror_g(cfg: Config):
    return lambda s: sh.apply_weights(sh.mirror(s, cfg), lambda a, b: sh.mirror_weights(a, b, cfg))


@pytest.mark.parametrize("preset", BOX)
def test_mirror_within_tolerance(cfgs_nb, preset: str) -> None:
    """The box mirrored in x (positions, velocities, headings, patches, field columns) with the brain's
    handedness mirrored (the W1 rows of the lateral inputs and the W2 columns of the turn and the two query
    laterals negated) is the mirrored trajectory: integers, counters and ledger identical, floats to rounding.
    No handedness leaks outside the brain's weights. (Births off: a child's mutation noise is not
    mirrored.)"""
    cfg = cfgs_nb[preset]
    g = mirror_g(cfg)
    tr = sh.lockstep(cfg, 16, forward=g, view=g, torus=cfg.world.size, skip_counters=OCCUPANCY)
    assert tr.exact_through(16, ATOL), (tr.ints, tr.fdev)


def test_mirror_needs_the_weights_mirrored(cfgs_nb) -> None:
    """Negative control: the mirrored world with the original weights is a different world, because the
    network is chiral (it turns left where the original turned left)."""
    cfg = cfgs_nb["tiny"]
    tr = sh.lockstep(cfg, 4, forward=lambda s: sh.mirror(s, cfg), view=mirror_g(cfg), torus=128.0)
    assert not tr.exact_through(4, ATOL)


def test_mirror_breaks_by_quantisation_not_by_a_stage(cfgs_nb) -> None:
    cfg = cfgs_nb["tiny"]
    g = mirror_g(cfg)
    tr = sh.lockstep(cfg, 40, forward=g, view=g, torus=cfg.world.size, skip_counters=OCCUPANCY)
    flip = tr.first_flip()
    assert flip is not None and flip > 16
    assert tr.exact_through(flip - 1, ATOL)


# --- 4. the gauge group of the Elman RNN --------------------------------------------------------------------


def gauge_pair(cfg: Config, seed: int) -> tuple[np.ndarray, np.ndarray]:
    h = dims(cfg).h
    rng = np.random.default_rng(seed)
    return rng.permutation(h), np.where(rng.random(h) < 0.5, -1.0, 1.0)


def test_gauge_forward_pass(cfgs) -> None:
    """One forward pass on random inputs: the relabelled network returns the relabelled hidden state and the
    same output. Sign flips alone are bit-exact (negation is exact and the summation order is unchanged);
    a permutation reorders the sums and agrees to float32 rounding."""
    from alife import brain

    cfg = cfgs["tiny"]
    d = dims(cfg)
    s = init_state(cfg)
    perm, sign = gauge_pair(cfg, 1)
    x = mx.random.uniform(-1.0, 1.0, (d.n, d.lay.D_IN), key=mx.random.key(0))
    h = mx.random.uniform(-1.0, 1.0, (d.n, d.h), key=mx.random.key(1))
    h0, y0 = brain.forward(s.W1, s.W2, x, h, cfg)
    for p, tol in ((np.arange(d.h), 0.0), (perm, 2e-6)):
        gs = sh.gauge_state(s._replace(h=h), cfg, p, sign)
        h1, y1 = brain.forward(gs.W1, gs.W2, x, gs.h, cfg)
        back = sh.ungauge_hidden(to_numpy(h1), p, sign)
        if tol == 0.0:
            assert np.array_equal(back, to_numpy(h0)) and np.array_equal(to_numpy(y1), to_numpy(y0))
        else:
            assert np.abs(back - to_numpy(h0)).max() < tol and np.abs(to_numpy(y1) - to_numpy(y0)).max() < tol


def gauge_view(cfg: Config, perm: np.ndarray, sign: np.ndarray):
    """The comparable form of the baseline: relabel its hidden state; the weights are compared as they are,
    both runs having none born."""
    return lambda s: sh.gauge_state(s, cfg, perm, sign)


@pytest.mark.parametrize(("preset", "steps"), [("tiny", 40), ("tiny3d", 40), ("tiny3s", 20)])
def test_hidden_sign_flip_is_exact(cfgs_nb, preset: str, steps: int) -> None:
    """Flipping the sign of hidden units (their W1 columns and rows, W2 rows and ``h``) leaves the whole
    trajectory bit-identical: tanh is odd, and the sums keep their order."""
    cfg = cfgs_nb[preset]
    ident, sign = np.arange(dims(cfg).h), gauge_pair(cfg, 2)[1]
    g = gauge_view(cfg, ident, sign)
    tr = sh.lockstep(cfg, steps, forward=g, view=g)
    assert tr.first_flip() is None and tr.exact_through(steps, 0.0)


@pytest.mark.parametrize(("preset", "steps"), [("tiny", 40), ("tiny3d", 40), ("tiny3s", 20)])
def test_hidden_permutation_within_tolerance(cfgs_nb, preset: str, steps: int) -> None:
    """A permutation (with signs) of the hidden units reorders the float sums of the two GEMVs: the
    trajectory agrees to rounding (positions to 2e-5 at 40 steps) with integers, counters and ledger
    identical."""
    cfg = cfgs_nb[preset]
    perm, sign = gauge_pair(cfg, 2)
    g = gauge_view(cfg, perm, sign)
    tr = sh.lockstep(cfg, steps, forward=g, view=g)
    assert tr.exact_through(steps, ATOL), (tr.ints, tr.fdev)
    assert max(tr.fdev) > 0.0  # a real reordering, not an accidental identity


def test_hidden_permutation_needs_both_layers_relabelled(cfgs_nb) -> None:
    """Negative control: relabelling the units of W1 but leaving W2 alone is not a symmetry (the hidden state
    starts at zero, so ``h`` itself would be a vacuous control)."""
    cfg = cfgs_nb["tiny"]
    perm, sign = gauge_pair(cfg, 2)
    fwd = lambda s: sh.gauge_state(s, cfg, perm, sign)._replace(W2=s.W2, pend_W2=s.pend_W2)  # noqa: E731
    tr = sh.lockstep(cfg, 4, forward=fwd, view=gauge_view(cfg, perm, sign))
    assert not tr.exact_through(4, ATOL)


# --- 5. HEALPix lattice rotations (planet) ------------------------------------------------------------------


@pytest.mark.parametrize("rotation", [sh.ROT_C4Z, sh.ROT_C2X, sh.ROT_C2Y], ids=["c4z", "c2x", "c2y"])
def test_lattice_symmetry_is_a_permutation_of_the_pixels(cfgs, rotation: np.ndarray) -> None:
    """Feasibility: the signed permutation matrices that fix the HEALPix grid (a quarter turn about the polar
    axis, half turns about the equatorial axes) are exact in float32, and the float32 pixel lookup sends the
    pixel centres to a permutation of the pixels (no centre falls on a boundary, no pixel is lost)."""
    cfg = cfgs["tiny3s"]
    t = sh.pixel_map(cfg, rotation)
    assert np.array_equal(np.sort(t), np.arange(dims(cfg).npix))
    assert not (t == np.arange(len(t))).any()  # no pixel is fixed: the maps are not trivial
    cube = sh.pixel_map(cfg, np.linalg.matrix_power(sh.ROT_C4Z, 4))
    assert np.array_equal(cube, np.arange(dims(cfg).npix))


def test_quarter_turn_about_the_polar_axis_is_exact(cfgs) -> None:
    """The planet turned by 90 degrees about its axis (agents, patches, the pixel axis of the resource and the
    chemicals) runs bit-identically, 25 steps with births, bonds and combat: the lattice's own symmetry holds
    exactly in float32 (the ring pixels, the hash, the kNN, the diffusion tables and the body frames)."""
    cfg = cfgs["tiny3s"]
    g = lambda s: sh.rotate_sphere(s, cfg, sh.ROT_C4Z)  # noqa: E731
    tr = sh.lockstep(cfg, 25, forward=g, view=g)
    assert tr.first_flip() is None and tr.exact_through(25, 0.0)
    assert tr.births() > 20


@pytest.mark.slow
def test_quarter_turn_long(cfgs) -> None:
    cfg = cfgs["tiny3s"]
    g = lambda s: sh.rotate_sphere(s, cfg, sh.ROT_C4Z)  # noqa: E731
    tr = sh.lockstep(cfg, 100, forward=g, view=g)
    assert tr.first_flip() is None and tr.exact_through(100, 0.0)


@pytest.mark.parametrize("rotation", [sh.ROT_C2X, sh.ROT_C2Y], ids=["c2x", "c2y"])
def test_half_turn_about_an_equatorial_axis_within_tolerance(cfgs, rotation: np.ndarray) -> None:
    """A half turn swaps the hemispheres (z -> -z), so float sums over the three components change order:
    equal to rounding (not bitwise), integers, counters and ledger identical, for 15 steps."""
    cfg = cfgs["tiny3s"]
    g = lambda s: sh.rotate_sphere(s, cfg, rotation)  # noqa: E731
    tr = sh.lockstep(cfg, 15, forward=g, view=g)
    assert tr.exact_through(15, ATOL), (tr.ints, tr.fdev)


def test_planet_needs_the_pixel_axis_permuted(cfgs) -> None:
    """Negative control: rotating the agents but not the fields is not a symmetry."""
    cfg = cfgs["tiny3s"]
    from sphere_rotation import rotate_state

    fwd = lambda s: rotate_state(s, sh.ROT_C4Z)  # noqa: E731
    tr = sh.lockstep(cfg, 4, forward=fwd, view=lambda s: sh.rotate_sphere(s, cfg, sh.ROT_C4Z))
    assert not tr.exact_through(4, ATOL)


# --- 6. rotation of the tag vector (kin cue) ----------------------------------------------------------------


@pytest.mark.parametrize(("preset", "steps"), [("tiny", 40), ("tiny3d", 40), ("tiny3s", 20)])
def test_tag_rotation_within_tolerance(cfgs_nb, preset: str, steps: int) -> None:
    """A rotation (or reflection) of the 4-d tag leaves every pairwise distance, hence the kin cue
    ``exp(-|dtag|^2 / tag_scale^2)``, unchanged up to rounding: the trajectory agrees to float32 rounding,
    integers, counters and ledger identical. (Births off: a child's tag is ``clip(tag + noise)``, a hypercube
    clip and per-coordinate noise, which only the 384 signed permutations could respect.)"""
    cfg = cfgs_nb[preset]
    q = sh.random_orthogonal(4, 7)
    g = lambda s: sh.rotate_tags(s, q)  # noqa: E731
    tr = sh.lockstep(cfg, steps, forward=g, view=g)
    assert tr.exact_through(steps, ATOL), (tr.ints, tr.fdev)
    assert max(tr.fdev) > 0.0


def test_tag_scaling_is_not_a_symmetry(cfgs_nb) -> None:
    """Negative control: shrinking the tags changes the pairwise distances and so the kin cue."""
    cfg = cfgs_nb["tiny"]
    g = lambda s: s._replace(tag=s.tag * 0.5)  # noqa: E731
    tr = sh.lockstep(cfg, 6, forward=g, view=lambda s: s)
    assert not tr.exact_through(6, ATOL)


# --- g = e: the harness itself ------------------------------------------------------------------------------


def test_identity_transformations_are_bitwise(cfgs_nb) -> None:
    """The identity member of each family, through the same code path, is bit-identical (the harness adds no
    noise of its own)."""
    cfg = cfgs_nb["tiny"]
    n, h = dims(cfg).n, dims(cfg).h
    ident = np.arange(n)
    cases = {
        "rows": (lambda s: sh.permute_rows(s, cfg, ident), None, True),
        "translate": (lambda s: sh.translate(s, cfg, (0, 0)), lambda s: sh.translate(s, cfg, (0, 0)), False),
        "gauge": (
            lambda s: sh.gauge_state(s, cfg, np.arange(h), np.ones(h)),
            lambda s: sh.gauge_state(s, cfg, np.arange(h), np.ones(h)),
            False,
        ),
        "tag": (lambda s: sh.rotate_tags(s, np.eye(4)), lambda s: sh.rotate_tags(s, np.eye(4)), False),
    }
    for name, (fwd, view, cov) in cases.items():
        tr = sh.lockstep(cfg, 6, forward=fwd, view=view, covariant=cov)
        assert tr.exact_through(6, 0.0), name
