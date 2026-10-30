"""The director's menu and the player's powers on a sphere (DESIGN §46.7; sphere series S5).

A place is (longitude, latitude) in degrees, a length an arc length at the floor; the engine's state holds the
patches as unit vectors and the catastrophe centre in radians. The levers are checked against independent
numpy geometry (unit vectors, dot products), and the catastrophe against the engine's own kill mask.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from alife import config, geom, lifecycle
from alife.config import load_config
from alife.director import actions, llm, prompt, rules, yoked
from alife.director.actions import ActionError, apply, clamp, env_view, validate_entry
from alife.game import powers
from alife.state import init_state, to_numpy


@pytest.fixture(scope="module")
def cfg():
    saved = config.SPHERE_READY
    config.SPHERE_READY = saved | {"director", "observers"}
    try:
        return load_config(
            preset="tiny3s",
            overrides=["director.enabled=true", "director.rule_every=100", "brain.weight_dtype=float32"],
        )
    finally:
        config.SPHERE_READY = saved


@pytest.fixture(scope="module")
def s0(cfg):
    return init_state(cfg)


def _entry(cfg, s, action, raw, duration=None, step=100):
    c = clamp(action, raw, duration, env_view(s), cfg)
    return {
        "step_applied": step, "source": "rule", "action": action,
        "params": c.params, "prev": c.prev, "duration": c.duration,
    }  # fmt: skip


def _arc(a_lonlat, b_lonlat, radius):
    return float(
        geom.great_circle_distance(
            geom.from_latlon(a_lonlat[1], a_lonlat[0]), geom.from_latlon(b_lonlat[1], b_lonlat[0]), radius
        )
    )


# --- the menu's lengths ------------------------------------------------------------------------------------


def test_the_width_of_a_sphere_is_its_equator(cfg) -> None:
    assert actions.span(cfg) == pytest.approx(2 * math.pi * 36.0)
    lo, hi = actions.radius_bounds(cfg)
    k = actions.span(cfg) / actions.CATASTROPHE_REF_SIZE
    assert lo == pytest.approx(20.0 * k) and hi == pytest.approx(min(200.0 * k, actions.span(cfg) / 2))
    assert hi <= math.pi * cfg.world.radius  # the cap never exceeds the planet (invariants.env range)


# --- a patch move ------------------------------------------------------------------------------------------


def test_a_patch_move_walks_a_great_circle_and_is_stored_as_lon_lat(cfg, s0) -> None:
    view = env_view(s0)
    assert view.patch_pos.shape == (cfg.fields.n_patches, 3)
    e = _entry(cfg, s0, "fertility_shift", {"patch": 1, "dx": 3.0, "dy": -2.0})
    p, prev = e["params"], e["prev"]
    assert set(p) == {"patch", "x", "y"} and -180.0 <= p["x"] < 180.0 and -90.0 <= p["y"] <= 90.0
    arc = _arc([prev["x"], prev["y"]], [p["x"], p["y"]], cfg.world.radius)
    assert arc == pytest.approx(math.hypot(3.0, 2.0), rel=2e-4)
    # the direction: 3 east and 2 north-negative (south) of where it was, to first order
    east, north = geom.local_frame(view.patch_pos[1].astype(np.float64))
    want = geom.from_latlon(p["y"], p["x"]) - view.patch_pos[1]
    assert np.dot(want, east) > 0 and np.dot(want, north) < 0
    validate_entry(e, cfg, 100)
    s1 = apply(s0, e, cfg)
    after = to_numpy(s1.patch_pos)
    np.testing.assert_allclose(after[1], geom.from_latlon(p["y"], p["x"]), atol=2e-6)
    np.testing.assert_array_equal(np.delete(after, 1, axis=0), np.delete(to_numpy(s0.patch_pos), 1, axis=0))
    assert np.abs(np.linalg.norm(after, axis=1) - 1).max() < 1e-6


def test_a_long_move_is_capped_at_a_quarter_of_the_equator(cfg, s0) -> None:
    e = _entry(cfg, s0, "fertility_shift", {"patch": 0, "dx": 500.0, "dy": 0.0})
    arc = _arc([e["prev"]["x"], e["prev"]["y"]], [e["params"]["x"], e["params"]["y"]], cfg.world.radius)
    assert arc == pytest.approx(actions.MOVE_MAX_FRAC * actions.span(cfg), rel=2e-4)
    validate_entry(e, cfg, 100)


def test_the_replay_check_refuses_a_move_that_is_too_long_or_off_the_planet(cfg, s0) -> None:
    e = _entry(cfg, s0, "fertility_shift", {"patch": 0, "dx": 1.0, "dy": 0.0})
    anti = {"x": ((e["prev"]["x"] + 180.0 + 180.0) % 360.0) - 180.0, "y": -e["prev"]["y"]}  # the antipode
    far = {**e, "params": {**e["params"], **anti}}
    with pytest.raises(ActionError, match="quarter of the equator"):
        validate_entry(far, cfg, 100)
    for bad in ({"x": 180.0}, {"x": -181.0}, {"y": 90.5}, {"y": float("nan")}):
        with pytest.raises(ActionError):
            validate_entry({**e, "params": {**e["params"], **bad}}, cfg, 100)
    with pytest.raises(ActionError, match="prev"):
        validate_entry({**e, "prev": {}}, cfg, 100)


# --- the catastrophe ---------------------------------------------------------------------------------------


def test_a_catastrophe_centre_wraps_in_longitude_and_clips_in_latitude(cfg, s0) -> None:
    c = clamp("catastrophe", {"x": 190.0, "y": 95.0, "radius": 15.0, "kill_p": 0.5}, None, env_view(s0), cfg)
    assert c.params["center"] == [-170.0, 90.0]
    assert clamp("catastrophe", {"x": 180.0, "y": 0.0, "radius": 15.0, "kill_p": 0.5}, None, env_view(s0), cfg
                 ).params["center"][0] == -180.0  # fmt: skip
    lo, hi = actions.radius_bounds(cfg)
    big = clamp("catastrophe", {"x": 0.0, "y": 0.0, "radius": 1e6, "kill_p": 0.5}, None, env_view(s0), cfg)
    assert big.params["radius"] == pytest.approx(hi, rel=1e-6)


def test_the_engine_kills_the_cap_the_host_counts(cfg, s0) -> None:
    """`apply` writes the centre in radians; `lifecycle.catastrophe_mask` (the engine) and
    `powers.column_count` (the host) agree on who is inside, at the poles and the date line."""
    lon, lat = 123.0, 61.0
    e = _entry(cfg, s0, "catastrophe", {"x": lon, "y": lat, "radius": 30.0, "kill_p": 0.9})
    validate_entry(e, cfg, 100)
    s1 = apply(s0, e, cfg)
    np.testing.assert_allclose(to_numpy(s1.env_cat_center), np.radians([lon, lat]), atol=1e-6)
    pos, alive = to_numpy(s1.pos), to_numpy(s1.alive)
    inside = powers.column_count(pos, alive, [lon, lat], 30.0, cfg.world.size, planet=cfg.world.radius)
    # the engine's mask with kill_p forced to 1 is the cap itself
    env = s1.env_cat_center, s1.env_cat_radius
    del env
    from alife.state import Env

    probe = Env(*[getattr(s1, f"env_{f}") for f in Env._fields])._replace(cat_kill_p=mx.array(1.0))
    mask = lifecycle.catastrophe_mask(s1.pos, s1.alive, probe, mx.random.key(0), cfg)
    assert int(to_numpy(mask).sum()) == inside
    # the same count by independent geometry
    up = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    c = geom.from_latlon(lat, lon)
    arc = np.arctan2(np.linalg.norm(np.cross(up, c), axis=1), up @ c) * cfg.world.radius
    assert inside == int((alive & (arc <= 30.0 + 1e-3)).sum())


def test_replay_validation_of_a_catastrophe_on_a_sphere(cfg, s0) -> None:
    e = _entry(cfg, s0, "catastrophe", {"x": 10.0, "y": 20.0, "radius": 15.0, "kill_p": 0.5})
    validate_entry(e, cfg, 100)
    for center in ([180.0, 0.0], [0.0, 91.0], [0.0], [0.0, 0.0, 0.0], [0.0, float("inf")]):
        with pytest.raises(ActionError):
            validate_entry({**e, "params": {**e["params"], "center": center}}, cfg, 100)


# --- the rule tier, the random persona, the model ----------------------------------------------------------


def test_the_rule_tier_draws_a_catastrophe_uniform_by_area(cfg, s0) -> None:
    ctx = rules.Context(
        100, 0, "gardener", 5, actions.span(cfg), None, None, env_view(s0), actions.is_sphere(cfg)
    )  # vitals and pacing are not read by this request
    p, persona = rules.RULE_PARAMS, rules.RULE_PARAMS.persona("gardener")
    rng = np.random.default_rng(0)
    draws = [rules._request("catastrophe", ctx, p, persona, rng)[0] for _ in range(4000)]
    lon = np.array([d["x"] for d in draws])
    lat = np.array([d["y"] for d in draws])
    assert lon.min() >= -180 and lon.max() <= 180 and lat.min() >= -90 and lat.max() <= 90
    z = np.sin(np.radians(lat))  # uniform on the sphere: z = sin(lat) is uniform in [-1, 1]
    assert abs(z.mean()) < 0.04 and abs(z.var() - 1 / 3) < 0.02 and abs(lon.mean()) < 6
    # the clamp takes every draw
    lo, hi = actions.radius_bounds(cfg)
    assert all(lo <= clamp("catastrophe", d, None, env_view(s0), cfg).params["radius"] <= hi for d in draws[:50])


def test_the_random_persona_draws_places_in_degrees(cfg, s0) -> None:
    rng = np.random.default_rng(5)
    got = [yoked.sample_params("catastrophe", rng, cfg)[0] for _ in range(500)]
    z = np.sin(np.radians([g["y"] for g in got]))
    assert abs(z.mean()) < 0.1 and max(abs(g["x"]) for g in got) <= 180.0
    mv = [yoked.sample_params("fertility_shift", rng, cfg)[0] for _ in range(200)]
    moves = [m for m in mv if "dx" in m]
    assert moves and max(math.hypot(m["dx"], m["dy"]) for m in moves) <= actions.MOVE_MAX_FRAC * actions.span(cfg)


def test_the_models_schema_and_prompts_state_a_sphere(cfg, s0) -> None:
    specs = llm.param_specs(cfg)["catastrophe"]
    by = {p.name: (p.lo, p.hi) for p in specs}
    assert by["x"] == (-180.0, 180.0) and by["y"] == (-90.0, 90.0)
    assert "planet" in prompt.world_text(cfg) and "longitude" in prompt.world_text(cfg)
    assert "cap of the planet" in prompt.effects(cfg)["catastrophe"]
    schema = llm.intervention_schema(cfg)
    assert schema  # builds


# --- the player's powers -----------------------------------------------------------------------------------


def test_the_nearest_patch_is_by_arc(cfg, s0) -> None:
    pp = to_numpy(s0.patch_pos).astype(np.float64)
    rng = np.random.default_rng(8)
    for _ in range(60):
        lon, lat = rng.uniform(-180, 180), math.degrees(math.asin(rng.uniform(-1, 1)))
        c = geom.from_latlon(lat, lon)
        arcs = np.arctan2(np.linalg.norm(np.cross(pp, c), axis=1), pp @ c)
        assert powers.nearest_patch(pp, lon, lat, 0.0) == int(np.argmin(arcs))
    at = geom.to_latlon(pp[2])
    assert powers.nearest_patch(pp, float(at[1]), float(at[0]), 0.0) == 2


def test_the_patches_block_has_directions_and_degrees(cfg, s0) -> None:
    pp = to_numpy(s0.patch_pos)
    blk = powers.patches_block(pp, np.ones(len(pp), np.float32), cfg.fields.patch_sigma)
    assert len(blk) == len(pp) and {"i", "x", "y", "z", "lon", "lat", "gain", "sigma"} == set(blk[0])
    lat, lon = geom.to_latlon(pp.astype(np.float64))
    assert [b["lon"] for b in blk] == [round(float(v), 2) for v in lon]
    assert abs(math.hypot(blk[0]["x"], blk[0]["y"], blk[0]["z"]) - 1) < 1e-3


def test_a_player_bloom_by_place_and_a_strike_resolve_on_a_sphere(cfg, s0) -> None:
    from types import SimpleNamespace

    pp = to_numpy(s0.patch_pos)
    lat, lon = geom.to_latlon(pp[3].astype(np.float64))
    line = SimpleNamespace(power="bloom", verb="use", id=1, params={"x": float(lon), "y": float(lat), "gain": 1.8})
    req, raw = powers.request_of(line, cfg, pp)
    assert raw["patch"] == 3 and "x" not in raw
    strike = SimpleNamespace(power="strike", verb="use", id=2, params={"x": 10.0, "y": -20.0})
    _req, raw = powers.request_of(strike, cfg, pp)
    lo, hi = actions.radius_bounds(cfg, actions.GAME_SPECS)
    assert lo <= raw["radius"] <= hi
