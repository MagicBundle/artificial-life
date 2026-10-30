"""The rotation-equivariance test of the consumer guard for the consumers PR S5 converts (docs/SPHERE.md 2.9
item 3; DESIGN §46.12): the host-side game layer, the levers and the label track give a rotated answer for a
rotated planet. (The observer inputs and the tables are in ``test_sphere_observers_run``, the storm detector
and tracker in ``test_storms_sph``, the developed track in ``test_sphere_labels``.)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from alife import geom
from alife.config import load_config
from alife.game import place, powers
from sphere_rotation import random_rotation


@pytest.fixture(scope="module")
def cfg():
    return load_config(preset="tiny3s", overrides=["brain.weight_dtype=float32"])


def test_a_strike_counts_the_same_agents_when_everything_is_rotated(cfg) -> None:
    rng = np.random.default_rng(1)
    v = rng.normal(size=(600, 3))
    pos = v / np.linalg.norm(v, axis=1, keepdims=True) * rng.uniform(36, 52, (600, 1))
    alive = rng.random(600) < 0.8
    rot = random_rotation(2)
    for lon, lat, radius in [(10.0, 20.0, 25.0), (0.0, 90.0, 18.0), (179.0, -35.0, 30.0)]:
        centre = rot @ geom.from_latlon(lat, lon)
        rlat, rlon = geom.to_latlon(centre)
        a = powers.column_count(pos, alive, [lon, lat], radius, 1024.0, planet=36.0)
        b = powers.column_count(pos @ rot.T, alive, [float(rlon), float(rlat)], radius, 1024.0, planet=36.0)
        assert abs(a - b) <= 1 and a > 0  # a boundary agent may fall either way in float32


def test_the_nearest_patch_is_rotation_equivariant(cfg) -> None:
    rng = np.random.default_rng(3)
    v = rng.normal(size=(7, 3))
    patches = v / np.linalg.norm(v, axis=1, keepdims=True)
    rot = random_rotation(4)
    for _ in range(40):
        lon, lat = rng.uniform(-180, 180), math.degrees(math.asin(rng.uniform(-1, 1)))
        rlat, rlon = geom.to_latlon(rot @ geom.from_latlon(lat, lon))
        assert powers.nearest_patch(patches, lon, lat, 0.0) == powers.nearest_patch(
            patches @ rot.T, float(rlon), float(rlat), 0.0
        )


def test_place_distance_height_and_centroid_are_rotation_equivariant(cfg) -> None:
    rng = np.random.default_rng(5)
    v = rng.normal(size=(30, 3))
    pts = v / np.linalg.norm(v, axis=1, keepdims=True) * rng.uniform(36, 50, (30, 1))
    rot = random_rotation(6)
    q = pts @ rot.T
    np.testing.assert_allclose(place.heights(pts, cfg), place.heights(q, cfg), atol=1e-12)
    np.testing.assert_allclose(
        place.distance(pts[:-1], pts[1:], cfg), place.distance(q[:-1], q[1:], cfg), atol=1e-9
    )
    np.testing.assert_allclose(rot @ np.array(place.centroid(pts, cfg)), place.centroid(q, cfg), atol=1e-9)


def test_a_raw_z_would_break_these_tests(cfg) -> None:
    """The negative control: the planet-frame z of a point (what ``pos[:, 2]`` returns on a sphere) is not a
    height, and a rotated world gives other numbers, so the comparisons above are not vacuous."""
    rng = np.random.default_rng(7)
    v = rng.normal(size=(40, 3))
    pts = v / np.linalg.norm(v, axis=1, keepdims=True) * rng.uniform(36, 50, (40, 1))
    q = pts @ random_rotation(8).T
    assert np.abs(pts[:, 2] - q[:, 2]).max() > 5.0
    assert np.abs(place.heights(pts, cfg) - place.heights(q, cfg)).max() < 1e-12
