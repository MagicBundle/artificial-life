"""The rotation-equivariance harness of the consumer guard (docs/SPHERE.md 2.9 item 3; sphere series S2).

A raw ``pos[:, 2]`` (the Cartesian z, not the height), a ``min_image`` or a ``world.size`` in a consumer that
does not depend on the planet axis returns plausible numbers on a sphere and is wrong. Every such consumer is
run on a sphere scene and on the same scene rotated by a fixed rotation ``Rot`` (positions, velocities,
headings and the patch centres rotated; the derived scalar ``heading`` recomputed); the results must agree up
to float32 rounding. A consumer that reads the Cartesian z breaks it at once, whatever the consumer. Each PR
of the series that converts a consumer adds it to ``tests/test_sphere_equivariance.py``.

``rotate_state`` does not move the field arrays (their cells are fixed on the planet), so it is for consumers
that read the agents and the totals of the fields, not the fields' spatial pattern.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from alife import sphere
from alife.state import StateS, to_numpy


def random_rotation(seed: int) -> np.ndarray:
    """A fixed random proper rotation of R³ (float64): the Q factor of a Gaussian matrix, det +1."""
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else q * np.array([1.0, 1.0, -1.0])


def rot(a: mx.array | np.ndarray, rotation: np.ndarray) -> mx.array:
    """``R a`` for each 3-vector of ``a`` (..., 3), computed in float64 and stored as float32."""
    r = np.asarray(rotation, np.float64)
    v = np.asarray(to_numpy(a) if isinstance(a, mx.array) else a, np.float64)
    return mx.array((v @ r.T).astype(np.float32))


def rotate_state(s: StateS, rotation: np.ndarray) -> StateS:
    """The state with every agent's ``pos``, ``vel``, ``hdg`` and the patch centres and drifts rotated by
    ``rotation``; ``heading`` is recomputed as the bearing of the rotated ``hdg`` (it depends on the planet
    axis); the fields stay where they are."""
    pos, hdg = rot(s.pos, rotation), rot(s.hdg, rotation)
    heading = sphere.bearing(hdg, pos)
    return s._replace(
        pos=pos,
        vel=rot(s.vel, rotation),
        hdg=hdg,
        heading=mx.where(s.alive, heading, s.heading),
        patch_pos=rot(s.patch_pos, rotation),
        patch_vel=rot(s.patch_vel, rotation),
    )
