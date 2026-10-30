"""The label study on a sphere (DESIGN §46.7; sphere series S5): the Nomad track is the sum of arcs,
developed onto the tangent plane, so that it grows without bound as the torus' unwrapped track does and
``Rg²`` is invariant under a rotation of the planet. The whole study on a tiny3s run is a slow test.
"""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from alife.analysis import labels as L
from alife.config import config_hash
from sphere_rotation import random_rotation


def _walk(path: np.ndarray, radius: float, drift: np.ndarray | None = None):
    """Track one agent along ``path`` (n+1, 3) and return the final TrackState."""
    tr = L.track_start(mx.array([True]), mx.array(path[0:1].astype(np.float32)), radius)
    cuid, uid, alive = mx.array([5]), mx.array([5]), mx.array([True])
    oc = mx.zeros((1, 10), dtype=mx.int32)
    for i in range(1, len(path)):
        d = None if drift is None else mx.array(drift[i : i + 1].astype(np.float32))
        tr = L.track_step(
            cuid, tr, alive, uid, mx.array(path[i : i + 1].astype(np.float32)), oc, 1.0, d, radius
        )
    mx.eval(*tr)
    return tr


def _great_circle(rot: np.ndarray, loops: float, n: int, radius: float, height: float = 0.0) -> np.ndarray:
    th = np.linspace(0.0, 2 * math.pi * loops, n + 1)
    c = np.stack([np.cos(th), np.sin(th), 0 * th], axis=1)
    return (c @ rot.T) * (radius + height)


def test_the_track_is_the_arc_length_of_a_path_that_laps_the_planet() -> None:
    radius, n = 50.0, 1600
    path = _great_circle(random_rotation(4), 2.0, n, radius)  # two laps: 4πR of arc, 2R at most of chord
    u = np.array(_walk(path, radius).u)[0]
    assert np.linalg.norm(u[:2]) == pytest.approx(
        4 * math.pi * radius, rel=2e-3
    )  # the arc sum, not saturated
    assert abs(u[2]) < 1e-3  # no height change
    # the telescoping sum of the displacement vectors saturates at the planet's diameter: the reason for it
    assert np.linalg.norm((path[1:] - path[:-1]).sum(axis=0)) < 1e-6 and 4 * math.pi * radius > 4 * 2 * radius


def test_rg2_grows_with_the_distance_travelled_and_is_a_rotation_invariant() -> None:
    radius = 50.0
    base = _great_circle(np.eye(3), 1.5, 900, radius)
    rg = float(_walk(base, radius).m2[0] / _walk(base, radius).n_ok[0])
    rot = random_rotation(8)
    rg_rot = float(_walk(base @ rot.T, radius).m2[0] / 900)
    assert rg_rot == pytest.approx(rg, rel=1e-3)
    short = float(_walk(_great_circle(np.eye(3), 0.75, 450, radius), radius).m2[0] / 450)
    assert rg > 2.0 * short > 0  # ballistic: Rg² grows as the square of the distance, no saturation


def test_a_small_circle_develops_into_an_arc_of_curvature_cot_colatitude() -> None:
    """The geodesic curvature of the circle at latitude 60 degrees is cot(30°)/R; its developed track is a
    circle arc of that curvature, whose chord has the closed form (an analytic reference)."""
    radius, lat, n = 50.0, math.radians(60.0), 2400
    th = np.linspace(0.0, 2 * math.pi, n + 1)
    path = (
        np.stack([np.cos(lat) * np.cos(th), np.cos(lat) * np.sin(th), np.sin(lat) + 0 * th], axis=1) * radius
    )
    u = np.array(_walk(path, radius).u)[0]
    kappa = 1.0 / math.tan(math.pi / 2 - lat) / radius
    length = 2 * math.pi * math.cos(lat) * radius
    chord = 2.0 / kappa * abs(math.sin(kappa * length / 2.0))
    assert np.linalg.norm(u[:2]) == pytest.approx(chord, rel=3e-3)


def test_the_height_component_is_the_summed_height_change_and_zk_is_the_height() -> None:
    radius = 50.0
    th = np.linspace(0.0, math.pi / 2, 301)  # the height ends at another value than it starts (review L6)
    h = 1.0 + 8.0 * np.sin(th)
    path = np.stack([np.cos(th), np.sin(th), 0 * th], axis=1) * (radius + h)[:, None]
    tr = _walk(path, radius)
    assert float(np.array(tr.u)[0, 2]) == pytest.approx(h[-1] - h[0], abs=2e-3)
    assert float(tr.zk[0]) == pytest.approx(h[-1], abs=2e-3) and float(tr.z0[0]) == pytest.approx(
        h[0], abs=2e-3
    )
    assert float(tr.m2[0]) > 0


def test_a_random_walk_on_the_sphere_is_diffusive_like_the_plane() -> None:
    """Mean squared developed displacement of correlated random headings grows like the step count; the same
    steps on a plane (float64 reference of the development) give the same Rg²."""
    radius, n, step = 60.0, 600, 0.6
    rng = np.random.default_rng(0)
    up = np.array([0.0, 0.0, 1.0])
    pos, frame = up * radius, np.array([1.0, 0.0, 0.0])
    path, plane = [pos.copy()], [np.zeros(2)]
    heading = 0.0
    for _ in range(n):
        heading += rng.normal(0.0, 0.5)
        east, north = frame, np.cross(pos / radius, frame)
        d = math.cos(heading) * east + math.sin(heading) * north
        ang = step / radius
        new = math.cos(ang) * pos + math.sin(ang) * radius * d
        new_up = new / np.linalg.norm(new)
        # transport the frame (minimal rotation) and keep the heading relative to it
        axis = np.cross(pos / radius, new_up)
        s, c = np.linalg.norm(axis), float(np.dot(pos / radius, new_up))
        k = axis / s
        frame = frame * c + np.cross(k, frame) * s + k * np.dot(k, frame) * (1 - c)
        frame -= np.dot(frame, new_up) * new_up
        frame /= np.linalg.norm(frame)
        plane.append(plane[-1] + step * np.array([math.cos(heading), math.sin(heading)]))
        pos = new
        path.append(pos.copy())
    tr = _walk(np.array(path), radius)
    u = np.array(tr.u)[0]
    # the developed track equals the planar walk of the same relative headings, up to the float32 path
    assert np.linalg.norm(u[:2] - plane[-1]) < 0.5
    ref = np.array(plane)
    mean = ref.mean(0)
    rg_ref = float(((ref - mean) ** 2).sum(axis=1).mean())
    assert float(tr.m2[0] / tr.n_ok[0]) == pytest.approx(rg_ref, rel=0.05)


def test_the_drift_is_read_in_the_developed_frame() -> None:
    radius = 50.0
    path = _great_circle(random_rotation(2), 0.5, 300, radius)
    tangent = np.diff(path, axis=0, prepend=path[:1])
    zero = np.zeros_like(path)
    base = np.array(_walk(path, radius, zero).ur)[0]
    tr = _walk(path, radius)
    np.testing.assert_allclose(base, np.array(tr.u)[0], atol=1e-4)
    # a drift equal to the displacement of each step removes the whole horizontal track
    ur = np.array(_walk(path, radius, tangent).ur)[0]
    assert np.linalg.norm(ur[:2]) < 0.05 * np.linalg.norm(np.array(tr.u)[0, :2])


@pytest.mark.slow
def test_the_study_runs_on_a_sphere_run(tmp_path: Path) -> None:
    from alife import cli
    from alife.config import load_config

    out = tmp_path / "run"
    cfg = load_config(preset="tiny3s", overrides=["brain.weight_dtype=float32", "run.compile=false"])
    argv = ["run", "--preset", "tiny3s", "--set", "brain.weight_dtype=float32", "--set", "run.compile=false",
            "--set", "run.checkpoint_every=200", "--set", "run.metrics_every=50",
            "--set", "run.heavy_metrics_every=200",
            "--steps", "600", "--out", str(out), "--quiet"]  # fmt: skip
    assert cli.main(argv) == 0
    rep = L.study(out, start=0, steps=600, warmup=256, cohort_every=64, arch_horizon=48, arch_min_obs=24,
                  intent_horizon=10, n_perm=20, verify=True, log=lambda m: None)  # fmt: skip
    assert rep["intent_version"] == 4 and rep["config_hash"] == config_hash(cfg)
    assert rep["rows_verified"] > 0 and rep["params"]["rules"]["sphere"] is True
    nomad = rep["archetypes"]["nomad"]
    assert nomad["pool"] > 0 and nomad["outcome"] == "rg2"
