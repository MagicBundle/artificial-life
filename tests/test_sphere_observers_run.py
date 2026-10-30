"""The game observers on a sphere world (DESIGN §46.7; sphere series S5): a tiny3s run with the observers on,
its social records (centroids, colonies, territory on the coarse graph), the identity tables, the sidecar
round trip and the rotation equivariance of the observer's inputs.

Eager on the CPU or Metal (no compiled function: ``run.compile`` is false), 200 steps once per module.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from alife import config, geom
from alife.config import load_config
from alife.game import observe, place, rules, social
from alife.game.observer import Observer, _colonies_from_record, _colonies_from_rows, _kin_from_rows
from alife.game.social import SOCIAL_V1
from alife.state import init_state, to_numpy
from alife.step import step
from sphere_rotation import random_rotation, rotate_state

# thresholds low enough for kin groups to exist in 200 steps of a 400-agent world
SOC = dataclasses.replace(
    SOCIAL_V1, snapshot_every=50, kin_occ=(1, 30), kin_enter=(2, 30), kin_leave=(1, 30), colony_enter=3
)


@pytest.fixture(scope="module")
def world():
    """(cfg, state, observer, records, last acts, state before the last step) after 200 observed steps."""
    saved = config.SPHERE_READY
    config.SPHERE_READY = saved | {"observers"}
    try:
        cfg = load_config(
            preset="tiny3s", overrides=["game.observe=true", "run.compile=false", "brain.weight_dtype=float32"]
        )
    finally:
        config.SPHERE_READY = saved
    s = init_state(cfg)
    obs = Observer.fresh(s, cfg, social=SOC)
    records: list[dict[str, Any]] = []
    prev = s
    acts = None
    for t in range(200):
        prev = s
        s, out = step(s, cfg, observe=True)
        acts = out.acts
        obs.update(s, acts, t + 1)
        mx.eval(s, out, *obs.arrays())
        records += obs.take_records()
    return cfg, s, obs, records, acts, prev


def test_the_observer_uses_the_sphere_rule_set_and_the_coarse_maps(world) -> None:
    cfg, _s, obs, records, _a, _p = world
    assert obs.rules.sphere and obs.rules == dataclasses.replace(rules.RULES_V3, sphere=True)
    assert (obs.social.terr_cells, obs.social.foot_cells) == (social.SPHERE_TERR_NSIDE, social.SPHERE_FOOT_NSIDE)
    assert [r["step"] for r in records] == [50, 100, 150, 200]
    assert all(r["intent_version"] == 4 for r in records)
    assert obs.row_block()["intent_version"] == 4
    assert obs._meta()["intent_version"] == 4
    assert cfg.world.geometry == "sphere"


def test_the_records_state_planet_frame_centroids(world) -> None:
    cfg, s, obs, records, _a, _p = world
    rp, h = cfg.world.radius, cfg.world.height
    last = records[-1]
    kin, col = last["kin_groups"], last["colonies"]
    assert kin["id"] and col["id"], "the thresholds are low enough for both tables to be filled"
    assert {"x", "y", "z", "z_mean"} <= set(kin) and "height" not in kin
    assert col["x"] and "z" in col
    for tab in (kin, col):
        c = np.stack([tab["x"], tab["y"], tab["z"]], axis=1)
        r = np.linalg.norm(c, axis=1)
        assert (r > rp - 1.0).all() and (r < rp + h + 1.0).all()  # a point of the shell (a mean of directions)
    assert all(0.0 <= v <= h for v in kin["z_mean"])
    # the centroid is the mean direction of the members, at their mean radius (place.centroid)
    pos = to_numpy(s.pos).astype(np.float64)
    slot = to_numpy(obs.so.kin)
    for j, sl in enumerate(kin["slot"]):
        m = slot == sl
        if not m.any():
            continue
        want = np.array(place.centroid(pos[m], cfg))
        got = np.array([kin["x"][j], kin["y"][j], kin["z"][j]])
        np.testing.assert_allclose(got, want, atol=0.02)  # the record is at the snapshot, the state is too


def test_the_row_derived_tables_equal_the_record_tables(world) -> None:
    """Without a record the identity tables come from the rows (`_kin_from_rows`); at a snapshot step they must
    agree with the record's (`_kin_from_record`): the sphere's x, y, z, height are the same numbers."""
    cfg, s, obs, records, _a, _p = world
    sl = obs.social_labels()
    kin, col = to_numpy(sl.kin_slot), to_numpy(sl.colony_slot)
    alive = to_numpy(s.alive)
    pos = to_numpy(s.pos)[alive]
    clade = to_numpy(s.clade)[alive].astype(np.int32)
    from_rows = _colonies_from_rows(obs.society, kin[alive], col[alive], pos, cfg)
    from_rec = _colonies_from_record(records[-1]["colonies"], True)
    assert [c["id"] for c in from_rows] == [c["id"] for c in from_rec]
    for a, b in zip(from_rows, from_rec, strict=True):
        assert (a["x"], a["y"], a["z"], a["size"]) == (b["x"], b["y"], b["z"], b["size"])
    krows = _kin_from_rows(obs.society, kin[alive], clade, pos, cfg, True)
    assert krows, "kin groups exist"
    for a in krows:
        assert {"x", "y", "z", "height"} <= set(a) and a["height"] is not None


def test_territory_counts_equal_a_brute_force_bincount(world) -> None:
    """The snapshot's (cell, slot) counts are the histogram of the float32 pixel of each member at the
    territory Nside, and the map's owners are among the tracked kin ids."""
    cfg, s, obs, records, _a, _p = world
    sr = obs.social
    g = sr.slots + 1
    si = social.soc_in(s)
    snap = social.snapshot(obs.so.kin, obs.so, si, None, cfg, sr)
    mx.eval(snap)
    alive = to_numpy(s.alive)
    kin = to_numpy(obs.so.kin)
    cell = to_numpy(snap.cell)
    pix = geom.ang2pix(to_numpy(s.pos).astype(np.float64), sr.terr_cells)
    near = np.array([p in geom.tables(sr.terr_cells).touch[c] or p == c for p, c in zip(pix, cell, strict=True)])
    assert near[alive].all()  # float32 vs float64 differ only by an adjacent pixel
    want = np.bincount((cell * g + kin)[alive & (kin >= 0)], minlength=sr.n_terr(True) * g)
    np.testing.assert_array_equal(to_numpy(snap.terr), want)
    assert to_numpy(snap.terr).sum() == (alive & (kin >= 0)).sum() > 0
    dmap, slots = obs.territory()
    assert dmap.owner.shape == (972,) and dmap.border.shape == (972, 4)
    owners = set(dmap.owner[dmap.owner >= 0].tolist())
    listed = {i for r in records for i in r["kin_groups"]["id"]} | set(slots[slots >= 0].tolist())
    assert owners and owners <= listed  # the window's ids: tracked groups, listed in some record
    last = records[-1]["territory"]
    assert last["owned_cells"] == int((dmap.owner >= 0).sum()) > 0
    assert sum(n for _i, n in zip(*last["cells"].values(), strict=True)) == last["owned_cells"]


def test_the_sidecar_round_trips_on_a_sphere(world, tmp_path) -> None:
    cfg, s, obs, _r, _a, _p = world
    arrays = obs.sidecar()
    assert arrays["host.terr.ema"].shape[1] == 972
    from alife.game.society import Society

    society, why = Society.from_arrays(arrays, cfg, obs.social, step=200)
    assert society is not None, why
    np.testing.assert_array_equal(society.arrays()["host.terr.ema"], arrays["host.terr.ema"])
    assert society.dom.current().border.shape == (972, 4)


def test_the_observer_inputs_are_rotation_invariant(world) -> None:
    """The accumulator inputs of a step: speed, the height, the lift and the radial velocity are the same for
    the rotated world; the Cartesian velocity halves carry a vector whose length is."""
    cfg, s, obs, _r, acts, prev = world
    rs = obs.rules
    rot = random_rotation(21)
    ra = rotate_state(s, rot)
    zero = obs.o.slot_kin
    xa, _xs, _x3 = observe.quantise_inputs(observe.obs_in(s, rs), acts, zero, cfg, rs)
    xb, _ys, _y3 = observe.quantise_inputs(observe.obs_in(ra, rs), acts, zero, cfg, rs)
    xa, xb = np.array(xa), np.array(xb)
    lay = rules.arch_layout(3, rs)
    alive = to_numpy(s.alive)
    for col in (rules.ArchCol.SPEED, lay.wind, lay.zpos, lay.lift):
        assert np.abs(xa[alive, int(col)] - xb[alive, int(col)]).max() <= 1  # the 1/256 quantum of a float round
    vr_a = xa[alive, lay.vrad] - xa[alive, lay.vrad + 1]
    vr_b = xb[alive, lay.vrad] - xb[alive, lay.vrad + 1]
    assert np.abs(vr_a - vr_b).max() <= 2
    cart = lambda x: np.stack([x[:, lay.v_pos + i] - x[:, lay.v_neg + i] for i in range(3)], axis=1)  # noqa: E731
    na, nb = np.linalg.norm(cart(xa)[alive], axis=1), np.linalg.norm(cart(xb)[alive], axis=1)
    assert np.abs(na - nb).max() <= 3
    # the height column is a height (0..256 over the shell), not a planet-frame coordinate
    assert xa[alive, lay.zpos].min() >= 0 and xa[alive, lay.zpos].max() <= 256
    assert np.abs(cart(xa)[alive] - cart(xb)[alive]).max() > 5  # the components themselves do rotate


def test_the_radial_velocity_column_is_the_radial_component(world) -> None:
    cfg, s, obs, _r, acts, _p = world
    rs = obs.rules
    x, _xs, _x3 = observe.quantise_inputs(observe.obs_in(s, rs), acts, obs.o.slot_kin, cfg, rs)
    lay = rules.arch_layout(3, rs)
    pos = to_numpy(s.pos).astype(np.float64)
    vel = to_numpy(s.vel).astype(np.float64)
    alive = to_numpy(s.alive)
    radial = (vel * pos / np.linalg.norm(pos, axis=1, keepdims=True)).sum(axis=1)
    u = cfg.physics.thrust_max / cfg.physics.drag
    want = np.clip(np.round(radial * 256.0 / u), -observe.VQ_MAX, observe.VQ_MAX)
    got = np.array(x)[:, lay.vrad] - np.array(x)[:, lay.vrad + 1]
    assert np.abs(got[alive] - want[alive]).max() <= 1
    assert np.abs(want[alive]).max() > 0, "agents move radially (buoyancy, gravity): the test is not vacuous"
