"""Voice on the shell (DESIGN §39 on §46.4; sphere series S3): hearing from real positions and headings.

``comm.hear`` is neighbour-based (the intensity falls off with the chord, the bearing is the loudness-weighted
mean of the body-frame offsets the sensing layer computed), so on a sphere its only geometric inputs are the
chord distance and the body frame ``(fwd, lat, up)`` of ``ops.body_axes_s``. The box suites (``test_comm.py``,
parameter ``"s"``) cover the function on synthetic scenes; here the scene is built from positions on the
sphere: the bearing in the listener's frame at places where a planar frame would break (the poles, the
antimeridian), and the rotation of the whole scene, which must change nothing a listener hears.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from alife import comm, sensing, spatial
from alife.config import Config, dims, load_config
from alife.state import init_state, to_numpy
from sphere_rotation import random_rotation, rot


def cfg_of(*extra: str) -> Config:
    return load_config(
        preset="tiny3s", overrides=["brain.weight_dtype=float32", "comm.enabled=true", "comm.width=4", *extra]
    )


def unit(lat_deg: float, lon_deg: float) -> np.ndarray:
    la, lo = math.radians(lat_deg), math.radians(lon_deg)
    return np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])


def frame(up: np.ndarray, hdg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(fwd, lat, up)`` of the body frame: ``lat = up × hdg`` (left of the heading)."""
    return hdg, np.cross(up, hdg), up


def listener_frame(cfg: Config, lat: float, lon: float, h: float, bearing_deg: float):
    """A listener at ``(lat, lon, h)`` heading ``bearing_deg`` clockwise from north: ``(pos, hdg, fwd, lat,
    up)`` as float64."""
    u = unit(lat, lon)
    east = np.cross([0.0, 0.0, 1.0], u)
    east = east / np.linalg.norm(east) if np.linalg.norm(east) > 1e-6 else np.array([0.0, 1.0, 0.0])
    north = np.cross(u, east)
    b = math.radians(bearing_deg)
    hdg = math.cos(b) * north + math.sin(b) * east
    fwd, left, up = frame(u, hdg)
    return u * (cfg.world.radius + h), hdg, fwd, left, up


def scene(cfg: Config, rows: list[tuple[np.ndarray, np.ndarray]], voices: dict[int, list[float]]):
    """A state holding exactly ``rows`` ``(pos, hdg)`` alive (all the others dead, far away) and the voice
    array; returns ``(state, nbr, desc, voice)``."""
    d = dims(cfg)
    s = init_state(cfg)
    pos = np.zeros((d.n, 3), np.float32)
    hdg = np.zeros((d.n, 3), np.float32)
    pos[:] = np.array([0.0, 0.0, -(cfg.world.radius + 6.0)], np.float32)
    hdg[:] = np.array([1.0, 0.0, 0.0], np.float32)
    for i, (p, h) in enumerate(rows):
        pos[i], hdg[i] = p, h
    alive = np.zeros(d.n, bool)
    alive[: len(rows)] = True
    s = s._replace(
        pos=mx.array(pos),
        hdg=mx.array(hdg),
        alive=mx.array(alive),
        vel=mx.zeros_like(s.vel),
        pitch=mx.zeros_like(s.pitch),
    )
    nb = spatial.find_neighbors(s.pos, s.alive, cfg, impl="mlx")
    desc = sensing.neighbor_descriptors(s, nb, cfg)
    v = np.zeros((d.n, d.voice_w), np.float32)
    for r, vals in voices.items():
        v[r, : len(vals)] = vals
    return s, nb, desc, mx.array(v)


PLACES = [
    ("mid", 20.0, 40.0, 30.0),
    ("north pole", 89.99, 10.0, 100.0),
    ("south pole", -89.99, 200.0, 250.0),
    ("antimeridian", 5.0, 179.999, 80.0),
    ("ring seam", 41.8, 45.0, 310.0),
]


@pytest.mark.parametrize(("name", "lat", "lon", "bearing"), PLACES)
def test_the_bearing_of_a_call_is_its_offset_in_the_listeners_frame(name, lat, lon, bearing) -> None:
    """One speaker ahead-left-above of the listener by known body-frame amounts: the bearing columns are the
    offset over the sensing radius in ``(fwd, lat, up)`` (a single source: the weighted mean is its offset),
    the heard intensity ``1 − dist/range``."""
    cfg = cfg_of()
    R, w = cfg.sensing.radius, dims(cfg).voice_w
    p0, h0, fwd, left, up = listener_frame(cfg, lat, lon, 6.0, bearing)
    off = 0.4 * R * fwd + 0.3 * R * left + 0.2 * R * up
    p1 = p0 + off
    # the speaker's own heading is irrelevant to what the listener hears: any tangent
    h1 = np.cross(p1 / np.linalg.norm(p1), fwd)
    h1 = h1 / np.linalg.norm(h1)
    _, nb, desc, voice = scene(cfg, [(p0, h0), (p1, h1)], {1: [1.0, 0.5]})
    h = to_numpy(comm.hear(voice, nb, desc, cfg))
    dist = np.linalg.norm(off)
    a = 1.0 - dist / cfg.sensing.radius
    np.testing.assert_allclose(h[0, :2], [a * 1.0, a * 0.5], rtol=2e-4)
    body = np.array([off @ fwd, off @ left, off @ up]) / R
    np.testing.assert_allclose(h[0, w:], body, atol=2e-4)
    assert not h[1].any()  # the speaker hears nothing of itself, and nobody else is there
    assert not h[2:].any()


def test_the_loudest_source_pulls_the_bearing_and_two_sources_add() -> None:
    cfg = cfg_of()
    R, w = cfg.sensing.radius, dims(cfg).voice_w
    p0, h0, fwd, left, _ = listener_frame(cfg, -35.0, 120.0, 6.0, 200.0)
    a_off, b_off = 0.5 * R * fwd, -0.5 * R * left
    rows = [(p0, h0), (p0 + a_off, h0), (p0 + b_off, h0)]
    _, nb, desc, voice = scene(cfg, rows, {1: [1.0], 2: [0.5]})
    h = to_numpy(comm.hear(voice, nb, desc, cfg))
    wa, wb = (1 - 0.5) * 1.0, (1 - 0.5) * 0.5
    np.testing.assert_allclose(h[0, 0], min(wa + wb, 1.0), rtol=3e-4)
    want = (wa * np.array([0.5, 0.0, 0.0]) + wb * np.array([0.0, -0.5, 0.0])) / (wa + wb)
    np.testing.assert_allclose(h[0, w:], want, atol=3e-4)


def test_what_a_listener_hears_does_not_depend_on_where_the_planet_axis_points() -> None:
    """Rotation equivariance (spec 2.9, item 3) for the voice: rotate every position and heading by a fixed
    rotation and the heard intensities and the body-frame bearings are the same to float32 rounding."""
    cfg = cfg_of()
    d = dims(cfg)
    rs = np.random.default_rng(7)
    rows = []
    for _ in range(40):
        lat, lon, h, b = rs.uniform(-89, 89), rs.uniform(-180, 180), rs.uniform(2, 14), rs.uniform(0, 360)
        p, hd, *_ = listener_frame(cfg, lat, lon, h, b)
        rows.append((p, hd))
    # cluster half of them so that many hear several voices
    for i in range(20, 40):
        rows[i] = (rows[i - 20][0] + rs.normal(0, 2.0, 3), rows[i - 20][1])
    voices = {i: list(rs.random(4)) for i in range(40)}
    rr = random_rotation(11)
    out = []
    for r in (None, rr):
        rows_r = rows if r is None else [(np.array(rot(p, r)), np.array(rot(hd, r))) for p, hd in rows]
        # re-tangent the headings of the moved rows (float32 rounding)
        fixed = []
        for p, hd in rows_r:
            u = p / np.linalg.norm(p)
            hd = hd - (hd @ u) * u
            fixed.append((p, hd / np.linalg.norm(hd)))
        _, nb, desc, voice = scene(cfg, fixed, voices)
        out.append(to_numpy(comm.hear(voice, nb, desc, cfg)))
    heard = [(h[:, : d.voice_w] > 0).any() for h in out]
    assert all(heard)
    np.testing.assert_allclose(out[0][:40, : d.voice_w], out[1][:40, : d.voice_w], atol=2e-4)
    np.testing.assert_allclose(out[0][:40, d.voice_w :], out[1][:40, d.voice_w :], atol=3e-4)
    assert np.abs(out[0][:40, d.voice_w :]).max() > 0.05  # genuinely non-trivial bearings


def test_the_hearing_range_is_a_chord_so_a_call_across_a_ring_seam_is_as_loud_as_anywhere() -> None:
    """Two speakers at the same chord from the listener, one on each side of a HEALPix ring seam and one at a
    pole: the same intensity (the weight is a function of the chord only)."""
    cfg = cfg_of()
    R = cfg.sensing.radius
    vals = []
    for lat, lon in ((10.0, 40.0), (41.81, 44.0), (89.9, 0.0), (5.0, 179.99)):
        p0, h0, fwd, left, _ = listener_frame(cfg, lat, lon, 6.0, 20.0)
        p1 = p0 + 0.6 * R * (0.8 * fwd + 0.6 * left)
        _, nb, desc, voice = scene(cfg, [(p0, h0), (p1, h0)], {1: [1.0]})
        vals.append(float(to_numpy(comm.hear(voice, nb, desc, cfg))[0, 0]))
    np.testing.assert_allclose(vals, vals[0], rtol=3e-4)
