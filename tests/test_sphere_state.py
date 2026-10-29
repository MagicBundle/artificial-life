"""The state of a sphere world: ``StateS``, its schema, the initial state and its draws (``alife.state``;
DESIGN §46.2 and §46.5, sphere series S1).

``StateS`` is ``State3``'s fields in the same order and then ``hdg``; the box states, their digests and their
schema are untouched. The initial state draws from the same keys as the box with sphere shapes (no new stream,
no RNG layout bump); the tests write the RNG table out independently of ``state.sphere_kinematics``.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from alife import config, fields, geom, layout, ops, rng, sphere, state
from alife.config import dims, load_config
from sphere_helpers import sphere_cfg, sphere_enabled  # noqa: F401

pytestmark = pytest.mark.usefixtures("sphere_enabled")  # the sphere is refused by validate until S2

SMALL = ["population.n_max=1100", "population.n_init=64"]
PRESETS = ["tiny3s", "mini3s", "64k3s"]


def _cfg(which: str = "tiny3s", *extra: str):
    return sphere_cfg(which, *SMALL, *extra)


def test_state_s_is_state3_then_hdg() -> None:
    assert state.StateS._fields == (*state.State3._fields, "hdg") and len(state.StateS._fields) == 51
    assert state.EXTRA_SPHERE_FIELDS == ("hdg",)
    assert (*state.AGENT_FIELDS_3D, "hdg") == state.AGENT_FIELDS_S
    assert state.State3._fields[-2:] == ("pitch", "buoy_cap")  # State3 itself is unchanged
    assert state.StateS.__annotations__["hdg"] is mx.array
    assert state.is_sphere_state(state.StateS(*(None,) * 51)) and not state.is_sphere_state(
        state.State3(*(None,) * 50)
    )
    assert state.is_3d_state(state.StateS(*(None,) * 51))  # a sphere state is a 3D state with an extra field
    assert state.StateS.env is state.State.env  # the read-only env view is shared


@pytest.mark.parametrize("which", PRESETS)
def test_expected_layout_is_the_schema_of_the_initial_state(which: str) -> None:
    cfg = _cfg(which)
    d = dims(cfg)
    lay = state.expected_layout(cfg)
    s = state.init_state(cfg)
    assert type(s) is state.StateS is state.state_cls(cfg)
    assert list(lay) == list(s._fields)
    for name, (shape, dt) in lay.items():
        a = getattr(s, name)
        assert tuple(a.shape) == shape and a.dtype == dt, (name, a.shape, shape, a.dtype, dt)
    assert lay["pos"][0] == lay["vel"][0] == lay["hdg"][0] == (d.n, 3)
    assert lay["resource"][0] == (cfg.world.layers, 12 * cfg.world.nside**2)
    assert lay["chem"][0] == (layout.NCH, *lay["resource"][0])
    assert lay["patch_pos"][0] == lay["patch_vel"][0] == (cfg.fields.n_patches, 3)
    assert lay["wx"][0] == (state.N_WX_SPHERE, 12 * d.wn * d.wn)
    assert lay["env_cat_center"][0] == (2,)
    assert state.agent_fields(cfg) == state.AGENT_FIELDS_S


def test_the_box_schema_has_no_sphere_field() -> None:
    cfg = load_config(preset="tiny3d")
    lay = state.expected_layout(cfg)
    assert list(lay) == list(state.State3._fields)
    assert type(state.init_state(load_config(preset="tiny3d", overrides=SMALL))) is state.State3
    assert state.agent_fields(cfg) == state.AGENT_FIELDS_3D and not dims(cfg).sph


def test_dead_rows_rest_at_the_floor_pole_heading_along_x() -> None:
    cfg = _cfg()
    d, s = dims(cfg), state.init_state(cfg)
    dead = slice(64, d.n)
    assert not bool(s.alive[dead].any()) and bool(s.alive[:64].all())
    assert np.array_equal(np.array(s.pos[dead]), np.tile([0.0, 0.0, d.r_lo], (d.n - 64, 1)))
    assert np.array_equal(np.array(s.hdg[dead]), np.tile([1.0, 0.0, 0.0], (d.n - 64, 1)))
    assert (
        not np.array(s.vel).any()
        and not np.array(s.pitch[dead]).any()
        and not np.array(s.heading[dead]).any()
    )
    assert np.array_equal(np.array(s.buoy_cap), np.full(d.n, cfg.mutation.buoy_init, np.float32))
    assert int(s.env_cat_radius) == 0 and np.array(s.env_cat_center).shape == (2,)


def _founders_ref(k_pos, k_head, n: int, cfg):
    """The RNG table of the founders written out in float64: one (n, 3) uniform draw (a, b, c) of the position
    key and one (n, 2) draw (yaw, pitch) of the heading key."""
    d = dims(cfg)
    u = np.array(mx.random.uniform(0.0, 1.0, (n, 3), key=k_pos), np.float64)
    ang = np.array(
        mx.random.uniform(
            mx.array([-math.pi, -layout.PITCH_MAX]), mx.array([math.pi, layout.PITCH_MAX]), (n, 2), key=k_head
        ),
        np.float64,
    )
    ct, lam = 2 * u[:, 0] - 1, 2 * math.pi * u[:, 1]
    st = np.sqrt(1 - ct**2)
    up = np.stack([st * np.cos(lam), st * np.sin(lam), ct], axis=1)
    rho = np.cbrt(d.r_lo**3 + u[:, 2] * (d.r_hi**3 - d.r_lo**3))
    east = np.stack([-up[:, 1], up[:, 0], np.zeros(n)], axis=1)
    east /= np.linalg.norm(east, axis=1)[:, None]
    north = np.cross(up, east)
    yaw = ang[:, 0]
    hdg = np.cos(yaw)[:, None] * north + np.sin(yaw)[:, None] * east
    return up * rho[:, None], hdg, yaw, ang[:, 1]


@pytest.mark.parametrize("which", PRESETS)
def test_founder_draws_follow_the_rng_table(which: str) -> None:
    cfg = _cfg(which)
    ik = rng.init_keys(cfg.run.seed)
    pos, hdg, yaw, pitch = state.sphere_kinematics(
        ik[rng.InitStream.POS], ik[rng.InitStream.HEADING], 5000, cfg
    )
    rp, rh, ry, rpi = _founders_ref(ik[rng.InitStream.POS], ik[rng.InitStream.HEADING], 5000, cfg)
    # float32 against float64 of the same draws (the cube root and the unit vector in float32)
    assert np.abs(np.array(pos) - rp).max() < 1e-4 * dims(cfg).r_hi
    assert np.abs(np.array(hdg) - rh).max() < 2e-4  # a pole-adjacent east/north frame magnifies float32 noise
    assert np.abs(np.array(yaw) - ry).max() < 1e-6 and np.abs(np.array(pitch) - rpi).max() < 1e-6
    # and the state carries exactly these (n_init founders, first rows)
    s = state.init_state(cfg)
    kin = state.sphere_kinematics(ik[rng.InitStream.POS], ik[rng.InitStream.HEADING], 64, cfg)
    assert np.array_equal(np.array(s.pos[:64]), np.array(kin[0])) and np.array_equal(
        np.array(s.hdg[:64]), np.array(kin[1])
    )
    assert np.array_equal(np.array(s.heading[:64]), np.array(kin[2])) and np.array_equal(
        np.array(s.pitch[:64]), np.array(kin[3])
    )


def test_founders_are_uniform_in_the_shell_by_volume_with_a_tangent_unit_heading() -> None:
    cfg = _cfg("tiny3s")
    d = dims(cfg)
    n = 40_000
    ik = rng.init_keys(7)
    pos, hdg, yaw, pitch = (np.array(a, np.float64) for a in state.sphere_kinematics(ik[0], ik[1], n, cfg))
    r = np.linalg.norm(pos, axis=1)
    assert d.r_lo <= r.min() and r.max() <= d.r_hi
    # equal volume: r^3 is uniform on [r_lo^3, r_hi^3] (Kolmogorov distance of the sample)
    x = np.sort((r**3 - d.r_lo**3) / (d.r_hi**3 - d.r_lo**3))
    assert np.abs(x - (np.arange(n) + 0.5) / n).max() < 0.012
    up = pos / r[:, None]
    assert abs(up.mean(0)).max() < 0.012  # equal area on the sphere
    counts = np.bincount(geom.ang2pix(up, 4), minlength=192)
    assert counts.std() / counts.mean() < 0.12
    assert np.abs(np.linalg.norm(hdg, axis=1) - 1).max() < 4e-7  # the invariants of the spec (section 2.1)
    assert np.abs((hdg * up).sum(1)).max() < 4e-7
    assert np.abs(pitch).max() <= layout.PITCH_MAX and abs(pitch.mean()) < 0.01
    assert np.abs(yaw).max() <= math.pi
    # the scalar heading is the bearing of the vector
    b = np.array(sphere.bearing(mx.array(hdg.astype(np.float32)), mx.array(pos.astype(np.float32))))
    diff = np.abs(((b - yaw + math.pi) % (2 * math.pi)) - math.pi)
    assert diff.max() < 2e-4


def test_the_initial_resource_is_the_floor_of_the_cap_times_fertility() -> None:
    cfg = _cfg("tiny3s")
    s = state.init_state(cfg)
    fert = np.array(fields.fertility(s.patch_pos, cfg), np.float32)
    want = np.floor(np.float32(cfg.fields.resource_cap) * fert).astype(np.int32)
    assert np.array_equal(np.array(s.resource), want)
    n_alive = int(s.alive.sum())
    total = int(s.resource.astype(mx.int64).sum()) + int(s.energy.astype(mx.int64).sum())
    assert int(s.total) == total + n_alive * cfg.energy.structure == int(s.total0)
    assert not np.array(s.chem).any()
    assert np.abs(np.linalg.norm(np.array(s.patch_pos), axis=1) - 1).max() < 3e-7


def test_arrays_round_trip_and_the_digest_is_deterministic() -> None:
    cfg = _cfg("tiny3s")
    s = state.init_state(cfg)
    d = state.state_to_arrays(s)
    assert list(d) == list(state.StateS._fields)
    for back in (state.state_from_arrays(d), state.state_from_arrays(d, cfg)):
        assert type(back) is state.StateS and state.state_digest(back) == state.state_digest(s)
    assert state.state_digest(state.init_state(cfg)) == state.state_digest(s)
    other = state.init_state(sphere_cfg("tiny3s", *SMALL, "run.seed=1"))
    assert state.state_digest(other) != state.state_digest(s)
    moved = s._replace(hdg=mx.concatenate([s.hdg[:1] * 0.5, s.hdg[1:]]))
    assert state.state_digest(moved) != state.state_digest(s)  # the digest covers hdg
    short = {k: v for k, v in d.items() if k != "hdg"}
    with pytest.raises(ValueError, match="missing=\\['hdg'\\]"):
        state.state_from_arrays(short, cfg)
    no_wx = {k: v for k, v in d.items() if k != "wx"}
    filled = state.state_from_arrays(no_wx, cfg)  # a file without wx gets the rest array of the sphere
    assert filled.wx.shape == (state.N_WX_SPHERE, 12 * dims(cfg).wn ** 2) and not np.array(filled.wx).any()


def test_a_sphere_state_is_refused_by_the_box_schema_and_vice_versa() -> None:
    cfg = _cfg("tiny3s")
    box_state = state.init_state(load_config(preset="tiny3d", overrides=SMALL))
    d = state.state_to_arrays(box_state)
    with pytest.raises(ValueError, match="do not match StateS fields"):
        state.state_from_arrays(d, cfg)
    sd = state.state_to_arrays(state.init_state(cfg))
    with pytest.raises(ValueError, match="do not match State3 fields"):
        state.state_from_arrays(sd, load_config(preset="tiny3d", overrides=SMALL))


def test_total_energy_and_exact_sum_cover_the_sphere_lattice() -> None:
    cfg = _cfg("64k3s")
    d = dims(cfg)
    assert d.n_cells <= 1 << 20  # exact_sum's length limit, validated
    r = mx.full(d.field_shape, 1_000_000, dtype=mx.int32)
    assert int(ops.exact_sum(r)) == 1_000_000 * d.n_cells


def test_the_sphere_stays_refused_until_s2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "GEOMETRY_SUPPORTED", ("box",))  # what the tree ships: S1 wires nothing
    with pytest.raises(config.ConfigError, match="not enabled yet"):
        _cfg("tiny3s")
