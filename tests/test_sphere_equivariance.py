"""The rotation-equivariance test of the consumer guard (docs/SPHERE.md 2.9 item 3; sphere series S2).

Every consumer that does not depend on the planet axis is run on a sphere scene and on the same scene
rotated by
a fixed random rotation (``sphere_rotation``): positions, velocities, headings and patch centres rotated, the
derived heading recomputed. The results must agree to float32 rounding. A raw ``pos[:, 2]`` (the Cartesian z)
or a ``min_image`` in a consumer breaks the comparison at once, whatever the consumer. S2 covers the
neighbour search, the descriptors and brain inputs, the invariants, the catastrophe and the metrics row; each
later PR that converts a consumer adds its test here.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from alife import bonds, metrics, physics, sensing, spatial
from alife.config import Config, dims, load_config
from alife.fields import FieldObs
from alife.state import StateS, init_state, to_numpy
from alife.step import step
from sphere_fields_helpers import dev  # noqa: F401  (a fixture)
from sphere_rotation import random_rotation, rotate_state


def cfg_of() -> Config:
    return load_config(preset="tiny3s", overrides=["brain.weight_dtype=float32"])


@pytest.fixture(scope="module")
def stepped() -> tuple[Config, StateS]:
    """A tiny3s world after 60 eager steps (bonds, bodies, ages, births have happened)."""
    cfg = cfg_of()
    s = init_state(cfg)
    for _ in range(60):
        s, _ = step(s, cfg)
    mx.eval(s)
    return cfg, s


def close(a: Any, b: Any, path: str = "") -> list[str]:
    """Differences between two metrics rows (floats to 3e-5 relative + 1e-6, everything else exactly)."""
    if isinstance(a, dict):
        return [m for k in sorted(set(a) | set(b)) for m in close(a.get(k), b.get(k), f"{path}.{k}")]
    if isinstance(a, list | tuple):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [m for i, (x, y) in enumerate(zip(a, b, strict=True)) for m in close(x, y, f"{path}[{i}]")]
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None or not math.isclose(a, b, rel_tol=3e-5, abs_tol=1e-6):
            return [f"{path}: {a} != {b}"]
        return []
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


@pytest.mark.usefixtures("dev")
def test_the_metrics_row_is_invariant(stepped) -> None:
    """Heights, bands, the clade and height information, speeds, bodies, energy totals: the row of a rotated
    world equals the row of the world (the fields stay where they are: only the agents rotate)."""
    cfg, s = stepped
    r = random_rotation(11)
    a = metrics.snapshot_metrics(s, cfg, heavy=True)
    b = metrics.snapshot_metrics(rotate_state(s, r), cfg, heavy=True)
    assert (
        a["height_mean"] is not None and 0.0 < a["height_mean"] < cfg.world.height
    )  # a height, not a coordinate
    assert close(a, b) == []
    # the layer occupancy of the carrying capacity depends on the radius only
    wa, wb = metrics.layer_occupancy_weights(s, cfg), metrics.layer_occupancy_weights(rotate_state(s, r), cfg)
    np.testing.assert_allclose(wa, wb, atol=0.0, rtol=0.0)


def test_a_raw_z_would_break_it(stepped) -> None:
    """The negative control: the Cartesian z of the same agents (what ``pos[:, 2]`` returns on a sphere) is
    not
    invariant, so the comparison above is not vacuous."""
    _cfg, s = stepped
    q = rotate_state(s, random_rotation(11))
    alive = to_numpy(s.alive)
    za, zb = to_numpy(s.pos)[alive, 2], to_numpy(q.pos)[alive, 2]
    assert abs(za.mean() - zb.mean()) > 0.5 or abs(np.sort(za) - np.sort(zb)).max() > 1.0
    ha = np.linalg.norm(to_numpy(s.pos)[alive], axis=1)
    hb = np.linalg.norm(to_numpy(q.pos)[alive], axis=1)
    np.testing.assert_allclose(ha, hb, atol=2e-5)


@pytest.mark.usefixtures("dev")
def test_the_neighbour_sets_distances_and_descriptors_are_invariant(stepped) -> None:
    cfg, s = stepped
    r = random_rotation(12)
    q = rotate_state(s, r)
    na = spatial.find_neighbors(s.pos, s.alive, cfg, impl="mlx")
    nb = spatial.find_neighbors(q.pos, q.alive, cfg, impl="mlx")
    ia, va, da = (np.array(a) for a in (na.idx, na.valid, na.dist))
    ib, vb, db = (np.array(a) for a in (nb.idx, nb.valid, nb.dist))
    R = np.float32(cfg.sensing.radius)
    # a pair within float noise of R may be in one list and not the other; everything else must match
    for i in range(len(ia)):
        sa = {int(j): float(x) for j, x, ok in zip(ia[i], da[i], va[i], strict=True) if ok}
        sb = {int(j): float(x) for j, x, ok in zip(ib[i], db[i], vb[i], strict=True) if ok}
        for j in sa.keys() ^ sb.keys():
            assert abs((sa.get(j) or sb[j]) - R) < 1e-3 or len(sa) == dims(cfg).k or len(sb) == dims(cfg).k, (
                i,
                j,
            )
        for j in sa.keys() & sb.keys():
            assert abs(sa[j] - sb[j]) < 3e-5
    assert va.sum() > 100
    if (ia == ib).all() and (va == vb).all():  # same lists: the body-frame descriptors are the same numbers
        da_ = np.array(sensing.neighbor_descriptors(s, na, cfg))
        db_ = np.array(sensing.neighbor_descriptors(q, nb, cfg))
        np.testing.assert_allclose(da_, db_, atol=3e-5)


@pytest.mark.usefixtures("dev")
def test_the_brain_input_is_invariant_without_the_grid_bound_blocks(stepped) -> None:
    """With the field blocks zeroed (they read the planet's cells), the whole input vector of an agent is a
    function of the agents' relative geometry: rotating the world leaves it unchanged."""
    cfg, s = stepped
    q = rotate_state(s, random_rotation(13))
    n = dims(cfg).n
    zero = FieldObs(
        res=mx.zeros((n,)),
        res_grad=mx.zeros((n, 3)),
        chem=mx.zeros((n, 3)),
        chem_grad=mx.zeros((n, 3, 3)),
    )

    def inputs(st: StateS) -> tuple[np.ndarray, Any]:
        nb = spatial.find_neighbors(st.pos, st.alive, cfg, impl="mlx")
        desc = sensing.neighbor_descriptors(st, nb, cfg)
        geom = bonds.bond_geometry(st.pos, st.partner, cfg)
        gr = physics.grounded(st.pos, cfg)
        age = mx.ones((n,), dtype=mx.int32)
        x = sensing.build_inputs(st, age, nb, desc, geom, zero, mx.random.key(2), cfg, grounded=gr)
        return np.array(x), nb

    xa, na = inputs(s)
    xb, nb = inputs(q)
    alive = to_numpy(s.alive)
    same = (np.array(na.idx) == np.array(nb.idx)).all(axis=1) & (
        np.array(na.valid) == np.array(nb.valid)
    ).all(axis=1)
    rows = np.flatnonzero(alive & same)
    assert len(rows) > 0.9 * alive.sum()
    np.testing.assert_allclose(xa[rows], xb[rows], atol=5e-5)
