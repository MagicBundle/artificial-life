"""The ecosystem director in a 3D world (DESIGN §27.10, docs/3D.md PR 4).

The director is host code that edits the ``env_*`` fields and ``patch_pos`` between compiled steps, so the
3D world needs little of it: the validation that refused ``director.enabled`` is gone; the catastrophe is a
vertical column (the step's mask, §27.4) whose radius bounds are scaled by ``L / 1024``; a patch moves in
x and y only (``patch_pos`` stays ``(PP, 2)``); the log header names ``"dims": 3`` (in 3D only, so that no
2D run ever gets a new header); the LLM prompt describes the 3D world; and ``K_est`` weighs each layer's
food by the share of the agents living in it (§27.10).

Two module-scoped runs on ``tiny3d`` carry the end-to-end tests: a scripted live run whose log replays
bitwise (the timed menu bounds are shrunk by a patch around the tests that need it, as in
``test_director_replay.py``), and a real-rules Gardener run, started small enough for the extinction rescue
to fire.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from alife import cli, fields, metrics, runner
from alife.config import Config, apply_overrides, config_from_dict, config_to_dict, dims, load_config
from alife.director import Director, actions, llm, log, pacing, prompt, yoked
from alife.director.actions import MENU, ActionError
from alife.director.rules import RULE_PARAMS, Context, Proposal
from alife.director.vitals import Tracker
from alife.runio import (
    DIRECTOR_ARCHIVE_FILE,
    DIRECTOR_LOG_FILE,
    JOURNAL_FILE,
    RunDir,
    load_checkpoint,
    read_jsonl,
)
from alife.state import CounterIdx, DeathCause, State3, init_state, state_digest, to_numpy
from helpers import make_scene, run_checked, scripted_decide

PERF = {"wall_s", "steps_per_s", "peak_mem_gb"}
EVERY, END, CKPT = 50, 400, 150
BASE = [
    "director.enabled=true",
    f"director.rule_every={EVERY}",
    "run.director=live",
    "run.metrics_every=10",
    f"run.heavy_metrics_every={EVERY}",
    f"run.checkpoint_every={CKPT}",
    "run.keep_checkpoints=20",
    "run.diversity_sample=64",
]


def _np(a: mx.array) -> np.ndarray:
    return to_numpy(a)


@pytest.fixture(scope="module")
def cfg3(tiny3d_cfg: Config) -> Config:
    """``tiny3d`` (f32) with the director enabled, live, and a cadence of 50 steps."""
    return apply_overrides(tiny3d_cfg, BASE)


def prop(action: str, raw: dict, duration: int | None = None) -> Proposal:
    return Proposal(action, raw, duration, "scripted", f"scripted {action}")


# --- the gate is lifted -----------------------------------------------------------------------------------


def test_the_director_validates_in_3d(tiny3d_cfg: Config) -> None:
    """``director.enabled`` was refused with ``world.dims = 3`` until PR 4; every mode now validates."""
    assert tiny3d_cfg.world.dims == 3 and not tiny3d_cfg.director.enabled
    for mode in ("live", "replay"):
        cfg = apply_overrides(tiny3d_cfg, [*BASE, f"run.director={mode}"])
        assert cfg.director.enabled and cfg.run.director == mode and cfg.world.dims == 3
    llm_cfg = apply_overrides(tiny3d_cfg, [*BASE, f"director.llm_every={2 * EVERY}"])
    assert llm_cfg.director.llm_every == 2 * EVERY  # the LLM tier too (its client is never built here)


# --- the catastrophe is a column --------------------------------------------------------------------------


def test_a_catastrophe_through_the_director_path_kills_a_column_at_every_height(micro3d_cfg: Config) -> None:
    """``clamp`` -> ``validate_entry`` -> ``apply`` -> one compiled step: 24 agents stand in an x/y column of
    radius 2.5 at six heights from near the floor to near the surface, 8 more stand 6 away from its axis at
    two heights. Only the column's agents die, at all heights (a sphere of that radius would reach the middle
    heights only), and the step's ledger closes with their corpses."""
    cfg = apply_overrides(micro3d_cfg, ["director.enabled=true"])
    c = cfg.world.size / 2
    heights = [3.0, 7.0, 11.0, 15.0, 19.0, 22.0]
    inside = [{"pos": [c + dx, c + dy, z]} for dx in (-1.5, 1.5) for dy in (-1.5, 1.5) for z in heights]
    ring = [(6.0, 0.0), (-6.0, 0.0), (0.0, 6.0), (0.0, -6.0)]
    outside = [{"pos": [c + dx, c + dy, z]} for dx, dy in ring for z in (4.0, 20.0)]
    s0 = make_scene(cfg, [*inside, *outside])
    n_in = len(inside)
    z0 = _np(s0.pos)[:, 2].copy()

    view = actions.env_view(s0)
    got = actions.clamp("catastrophe", {"x": c, "y": c, "radius": 2.5, "kill_p": 0.9}, None, view, cfg)
    assert got.params["radius"] == 2.5 and got.params["center"] == [c, c] and len(got.params["center"]) == 2
    entry = {
        "step_applied": EVERY,
        "source": "rule",
        "action": "catastrophe",
        "params": got.params,
        "prev": got.prev,
        "duration": None,
    }
    actions.validate_entry(entry, cfg, EVERY)
    s1 = actions.apply(s0, entry, cfg)
    assert _np(s1.env_cat_center).shape == (2,) and float(_np(s1.env_cat_radius)) == 2.5

    s2, hist = run_checked(cfg, 1, s1)  # full_check: the ledger closes, no error bit
    out = hist[0][2]
    dead = np.flatnonzero(_np(s1.alive) & ~_np(s2.alive))
    assert len(dead) == int(_np(out.counters)[CounterIdx.DEATHS_CATASTROPHE]) >= 15
    assert (dead < n_in).all()  # nobody outside the column
    assert (
        z0[dead].min() <= heights[0] and z0[dead].max() >= heights[-1]
    )  # near the floor and near the surface
    dv = _np(out.deaths.valid)  # the first DL deaths by row are logged, the rest counted
    logged = _np(out.deaths.uid)[dv].tolist()
    assert logged == _np(s1.uid)[dead].tolist()[: len(logged)] and len(logged) == min(len(dead), dims(cfg).dl)
    assert (_np(out.deaths.cause)[dv] == DeathCause.CATASTROPHE).all()


def test_a_patch_move_is_x_and_y_only_and_patch_pos_stays_pp2(cfg3: Config) -> None:
    pp = cfg3.fields.n_patches
    s0 = init_state(cfg3)
    assert isinstance(s0, State3) and s0.patch_pos.shape == (pp, 2) and s0.patch_vel.shape == (pp, 2)
    view = actions.env_view(s0)
    assert view.patch_pos.shape == (pp, 2)
    got = actions.clamp("fertility_shift", {"patch": 1, "dx": 5.0, "dy": -3.0}, None, view, cfg3)
    assert set(got.params) == {"patch", "x", "y"} and set(got.prev) == {"patch", "x", "y"}
    entry = {
        "step_applied": EVERY,
        "source": "rule",
        "action": "fertility_shift",
        "params": got.params,
        "prev": got.prev,
        "duration": None,
    }
    actions.validate_entry(entry, cfg3, EVERY)
    s1 = actions.apply(s0, entry, cfg3)
    assert isinstance(s1, State3) and s1.patch_pos.shape == (pp, 2) and s1.patch_pos.dtype == mx.float32
    old, new = _np(s0.patch_pos), _np(s1.patch_pos)
    assert np.array_equal(new[[0, 2, 3]], old[[0, 2, 3]])  # the other patches stay put
    assert new[1].tolist() == pytest.approx([got.params["x"], got.params["y"]])
    assert not np.array_equal(new[1], old[1])
    for extra in ({"z": 3.0}, {"dz": 1.0}):  # a height is not on the menu: a replay rejects it
        bad = {**entry, "params": {**entry["params"], **extra}}
        with pytest.raises(ActionError, match="params must be exactly"):
            actions.validate_entry(bad, cfg3, EVERY)


# --- radius bounds ----------------------------------------------------------------------------------------


def _with_size(cfg: Config, size: float) -> Config:
    """``cfg`` with another world width, unvalidated (the bounds read only ``world.size`` and the dims)."""
    d = config_to_dict(cfg)
    d["world"] = {**d["world"], "size": size}
    return config_from_dict(d)


def test_radius_bounds_are_scaled_by_the_world_width_in_3d_only(tiny_cfg: Config, tiny3d_cfg: Config) -> None:
    assert actions.CATASTROPHE_REF_SIZE == 1024.0 and actions.CATASTROPHE_RADIUS == (20.0, 200.0)
    assert actions.radius_bounds(tiny_cfg) == (20.0, 64.0)  # 2D: the menu's range capped by L/2, as before
    assert actions.radius_bounds(_with_size(tiny_cfg, 1024.0)) == (20.0, 200.0)
    assert actions.radius_bounds(_with_size(tiny_cfg, 40.0)) == (20.0, 20.0)
    assert actions.radius_bounds(tiny3d_cfg) == (2.5, 25.0)  # L = 128: the menu's range times 1/8
    assert actions.radius_bounds(_with_size(tiny3d_cfg, 320.0)) == (6.25, 62.5)  # 64k3d's width
    for size in (48.0, 128.0, 320.0, 1024.0):
        lo, hi = actions.radius_bounds(_with_size(tiny3d_cfg, size))
        assert (lo, hi) == (20.0 * size / 1024.0, 200.0 * size / 1024.0) and hi < size / 2
        # the same share of the floor as the 2D disc of a 1024-wide world at its largest radius
        assert np.pi * hi * hi / size**2 == pytest.approx(np.pi * 200.0**2 / 1024.0**2)


def test_the_scaled_bounds_reach_the_clamp_the_replay_the_schema_and_the_random_sampler(
    cfg3: Config, tiny_cfg: Config
) -> None:
    s0 = init_state(cfg3)
    view = actions.env_view(s0)
    big = actions.clamp(
        "catastrophe", {"x": 10.0, "y": 20.0, "radius": 30.0, "kill_p": 0.5}, None, view, cfg3
    )
    small = actions.clamp(
        "catastrophe", {"x": 10.0, "y": 20.0, "radius": 1.0, "kill_p": 0.5}, None, view, cfg3
    )
    assert (big.params["radius"], small.params["radius"]) == (25.0, 2.5)  # the 2D-sized request is cut
    entry = {"step_applied": EVERY, "source": "rule", "action": "catastrophe", "prev": {}, "duration": None}
    actions.validate_entry({**entry, "params": big.params}, cfg3, EVERY)
    with pytest.raises(ActionError, match="radius"):  # a 2D log's radius is out of range here
        actions.validate_entry({**entry, "params": {**big.params, "radius": 30.0}}, cfg3, EVERY)
    with pytest.raises(ActionError, match="radius"):
        actions.validate_entry({**entry, "params": {**big.params, "radius": 2.0}}, cfg3, EVERY)
    lo, hi = actions.radius_bounds(cfg3)
    cat = {p.name: p for p in llm.param_specs(cfg3)["catastrophe"]}
    assert (cat["radius"].lo, cat["radius"].hi) == (lo, hi) == (2.5, 25.0)
    schema = json.dumps(llm.intervention_schema(cfg3))
    assert '"minimum": 2.5' in schema and '"maximum": 25.0' in schema
    rng = yoked.rng_for(cfg3.run.seed, 100, 0)
    for _ in range(200):
        raw, duration = yoked.sample_params("catastrophe", rng, cfg3)
        assert lo <= raw["radius"] <= hi and 0.0 <= raw["x"] < cfg3.world.size and duration is None
    # 2D is untouched
    assert {p.name: p for p in llm.param_specs(tiny_cfg)["catastrophe"]}["radius"].hi == 64.0


# --- the log header ---------------------------------------------------------------------------------------

HEADER_2D_KEYS = {"kind", "step", "rules_version", "persona", "goal", "rule_every", "budget_per_100k", "seed"}


def test_the_header_names_dims_in_3d_only(tiny_cfg: Config, tiny3d_cfg: Config) -> None:
    d2 = apply_overrides(tiny_cfg, ["director.enabled=true"])
    d3 = apply_overrides(tiny3d_cfg, ["director.enabled=true"])
    h2, h3 = log.make_header(d2, 0, 2), log.make_header(d3, 0, 2)
    assert set(h2) == HEADER_2D_KEYS and "dims" not in h2  # a 2D header keeps its step-2 key set
    assert set(h3) == HEADER_2D_KEYS | {"dims"} and h3["dims"] == 3
    assert {k: v for k, v in h3.items() if k != "dims"} == h2  # nothing else differs
    # the LLM tier's and the yoke's keys come after, in 3D as in 2D
    llm3 = apply_overrides(d3, ["director.llm_every=2000"])
    assert list(log.make_header(llm3, 0, 2))[-3:] == ["dims", "llm_every", "llm_model"]
    # "2D resumes append no header": a restart with the same policy needs none, in either dimension
    for cfg, header in ((d2, h2), (d3, h3)):
        assert not log.needs_header([header], log.make_header(cfg, 900, 2))
        assert log.needs_header([], header)
    # a 2D header (no dims key) and a 3D one are different policies
    assert log.needs_header([h2], log.make_header(d3, 900, 2))


def test_a_log_of_the_other_dimensions_is_refused(cfg3: Config, tiny_cfg: Config) -> None:
    d2 = apply_overrides(tiny_cfg, ["director.enabled=true", f"director.rule_every={EVERY}"])
    h2, h3 = log.make_header(d2, 0, 2), log.make_header(cfg3, 0, 2)
    log.validate_log([h2], d2)
    log.validate_log([h3], cfg3)
    log.validate_log([{k: v for k, v in h2.items() if k != "dims"}], d2)  # absent = 2D
    for header, cfg, written, here in ((h3, d2, "3", "2"), (h2, cfg3, "2", "3")):
        with pytest.raises(log.LogError, match=rf"dims = {written}.*world.dims = {here}"):
            log.validate_log([header], cfg)
    with pytest.raises(log.LogError, match="dims"):  # no log can claim a garbage dimension
        log.validate_log([{**h3, "dims": "3"}], cfg3)


# --- the LLM prompt ---------------------------------------------------------------------------------------

# sha256 of the 2D prompts at the head before PR 4 (tiny, f32, the director on, the goal "grow bodies"; the
# window of ``_window_ctx``): the 2D texts are byte-identical.
PROMPT_2D_SHA = {
    "gardener": "cb813c543a83c5beceba368739a031651a233f860b9a13b67f51e6c29b955e62",
    "trickster": "c5d3c90b8f4c8f6849571d45f19eebb97f795302815465560662393f17df29cc",
    "architect": "7e5f4276434a7f2f32e19129d199393e15c1f446574771bc1b0b7f2f4a662122",
}
USER_2D_SHA = "a5590822fec2ee2382b76f3c3de5753084a9d0c32151ad6ca9cd39ff38a6e412"
CATASTROPHE_2D = "kill a fraction of the agents inside a disc once: founder effects, then recolonisation"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _synthetic_row(step: int) -> dict[str, Any]:
    """A metrics row of a population of 200 (a window of 10 steps), for the prompt and vitals tests."""
    return {
        "step": step,
        "steps": 10,
        "n_alive": 200.0,
        "n_alive_min": 190,
        "births": 5,
        "deaths_starved": 3,
        "deaths_killed": 1,
        "deaths_aged": 0,
        "deaths_catastrophe": 0,
        "emitters_frac": 0.1,
        "sharers_frac": 0.2,
        "mean_bonds_per_agent": 0.5,
        "predation_ratio": 0.1,
        "top_clade_share": 0.4,
        "clades_alive": 7,
        "clade_entropy": 1.5,
        "weight_diversity": 0.3,
        "frac_in_bodies_ge4": 0.05,
        "bodies_hist": {"1": 150},
        "bodies_largest": 6,
        "sigma_q50": 0.05,
        "rate_q50": 0.05,
        "frac_q50": 0.3,
    }


def _window_ctx(cfg: Config, persona: str) -> tuple[Context, Tracker]:
    """A context over ten synthetic metrics rows (a population of 200, K_est 409), for the prompt builders."""
    tr = Tracker(RULE_PARAMS)
    for step in range(10, 101, 10):
        tr.observe(_synthetic_row(step))
    vitals = tr.vitals(tr.close(100), 409.0, 1024)
    env = actions.EnvView(1.0, 0.0, 0, 1.0, 0, np.ones(4, np.float32), np.zeros((4, 2), np.float32))
    ctx = Context(100, 0, persona, 8, cfg.world.size, vitals, pacing.fold([], 100), env)
    return ctx, tr


def _messages(base: Config, persona: str) -> tuple[str, str]:
    cfg = apply_overrides(
        base, ["director.enabled=true", f"director.persona={persona}", "director.goal=grow bodies"]
    )
    ctx, tr = _window_ctx(cfg, persona)
    return (
        prompt.system_message(cfg, RULE_PARAMS),
        prompt.user_message(ctx, RULE_PARAMS, cfg, tr.history, [], [], []),
    )


@pytest.mark.parametrize("persona", ["gardener", "trickster", "architect"])
def test_the_2d_prompts_are_byte_identical(tiny_cfg: Config, persona: str) -> None:
    system, user = _messages(tiny_cfg, persona)
    assert _sha(system) == PROMPT_2D_SHA[persona] and _sha(user) == USER_2D_SHA
    assert "The world is a torus; one step is one time unit." in system
    assert "World 128 x 128, 4 fertility patches." in user and "K_est 409, window low" in user
    assert prompt.EFFECT["catastrophe"] == CATASTROPHE_2D  # the table the 2D tests read is not edited
    assert prompt.effects(tiny_cfg) is prompt.EFFECT and prompt.world_text(tiny_cfg).startswith(
        "The world is a torus"
    )


@pytest.mark.parametrize("persona", ["gardener", "trickster", "architect"])
def test_the_3d_prompts_describe_the_3d_world(tiny3d_cfg: Config, tiny_cfg: Config, persona: str) -> None:
    system, user = _messages(tiny3d_cfg, persona)
    system2d, _ = _messages(tiny_cfg, persona)
    world = (
        "The world is periodic in x and y, with a floor and a surface at height 24; "
        "one step is one time unit."
    )
    assert world in system and "torus" not in system
    assert prompt.effects(tiny3d_cfg) is prompt.EFFECT_3D
    column = prompt.EFFECT_3D["catastrophe"]
    assert "vertical column" in column and "from the floor to the surface" in column
    assert "- catastrophe (cost 3, cooldown " in system and column in system and CATASTROPHE_2D not in system
    assert "radius in [2.5, 25]" in system  # the scaled bounds are in the menu
    # only the catastrophe's wording differs: every other action is described as in 2D
    assert {**prompt.EFFECT, "catastrophe": column} == prompt.EFFECT_3D and column != CATASTROPHE_2D
    for name, text in prompt.EFFECT.items():
        if name != "catastrophe":
            assert f"- {name} (cost" in system and text in system
    assert system2d.count("Menu:") == system.count("Menu:") == 1
    assert "World 128 x 128 x 24, 4 fertility patches. Decide what" in user
    assert f"capacity estimate K_est 409{prompt.K_NOTE_3D}, window low" in user  # the occupancy rule, §27.10


def test_the_3d_prompt_describes_the_occupancy_weighted_k_est(tiny3d_cfg: Config, tiny_cfg: Config) -> None:
    """The qualifier after the capacity estimate says what the number counts: not "floor-layer food only"
    (the rule of PR 4), but each layer's food weighted by where the agents live."""
    _, user = _messages(tiny3d_cfg, "gardener")
    note = prompt.K_NOTE_3D
    assert note == (
        " (food weighted by where the agents live: the floor layer's alone if all are in the floor layer,"
        " every layer's if they are spread evenly)"
    )
    assert (
        "on the floor" not in note
    )  # the layer is z < field_cell, which is not "grounded" (z <= r + floor_eps)
    assert user.count(note) == 1 and "floor-layer food only" not in user
    line = next(ln for ln in user.splitlines() if ln.startswith("viability:"))
    assert "of the capacity estimate K_est 409" + note + ", window low" in line
    _, user2d = _messages(tiny_cfg, "gardener")
    assert "food weighted" not in user2d and "K_est 409, window low" in user2d  # 2D: no qualifier at all


def test_the_director_vitals_use_the_occupancy_weighted_k_est(micro3d_cfg: Config, tmp_path: Path) -> None:
    """``Director._vitals`` reads ``metrics.carrying_capacity`` of the live state: the same patches give a
    floor-only K_est to agents that all rest on the floor and a larger one to agents spread over the column,
    and ``n / K_est`` follows."""
    cfg = apply_overrides(
        micro3d_cfg, ["director.enabled=true", "run.director=live", "director.rule_every=100"]
    )
    zs = {"floor": [1.0] * 8, "column": [2.0, 5.5, 9.5, 13.5, 17.5, 21.0, 3.0, 6.0]}
    got = {}
    for name, z in zs.items():
        s = make_scene(cfg, [dict(pos=(4.0 + 5 * i, 8.0, h)) for i, h in enumerate(z)])
        rd = RunDir.create(tmp_path / name, cfg)
        director = Director(rd, cfg, "live", 0)
        for step in range(10, 101, 10):
            director.observe(_synthetic_row(step))
        v = director._vitals(s, 100)
        assert v.k_est == metrics.carrying_capacity(s, cfg, s.step, s.env) > 0
        assert v.n_over_k == pytest.approx(200.0 / v.k_est)
        got[name] = v
    floor_only = metrics.carrying_capacity(make_scene(cfg, []), cfg)  # nobody alive: the floor layer alone
    assert got["floor"].k_est == floor_only
    # all six layers are counted: two hold more than their even share (weight 1), four hold 1/8 against 1/6
    # (weight 0.75), so the weights sum to 5 and the floor-only value is more than doubled
    assert got["column"].k_est > 2 * floor_only
    assert got["column"].n_over_k < got["floor"].n_over_k / 2


def _spread_scene(cfg: Config, n: int) -> State3:
    """``n`` agents spread evenly over the six layers of ``tiny3d``: a height per layer, a lattice in x, y."""
    heights = (2.0, 6.0, 10.0, 14.0, 18.0, 22.0)
    return make_scene(
        cfg, [dict(pos=(2.0 + 4.0 * (i % 30), 2.0 + 4.0 * (i // 30), heights[i % 6])) for i in range(n)]
    )


def test_a_spread_population_is_flagged_at_three_times_the_floor_only_size(
    tiny3d_cfg: Config, tmp_path: Path
) -> None:
    """A consequence of the occupancy rule for the extinction criterion (§27.10, a known limit): ``n / K_est``
    is judged against the layers the agents live in, so agents spread over the column, such as the founders
    and a rescue's immigrants (placed uniformly in the volume, §27.4), are flagged at about three times the
    population that would be flagged on the floor layer alone. A rescue therefore raises the ``K_est`` it is
    judged by: 200 agents spread over the six layers are at risk (``n / K_est`` under ``risk_k_frac``) where
    200 agents on the floor are not, and the size that clears the flag is about 280 against about 82.
    The 3D calibration of ``risk_k_frac`` is an open item of §27.13; this pins the arithmetic."""
    cfg = apply_overrides(
        tiny3d_cfg, ["director.enabled=true", "run.director=live", "director.rule_every=100"]
    )
    risk = RULE_PARAMS.risk_k_frac
    floor_only = metrics.carrying_capacity(make_scene(cfg, []), cfg)  # nobody alive: the floor layer alone
    spread = {n: metrics.carrying_capacity(_spread_scene(cfg, n), cfg) for n in (40, 200, 250)}
    assert all(k > 3 * floor_only for k in spread.values())  # an even column counts 3.5 floors' food
    got = {}
    for name, s in (
        ("spread", _spread_scene(cfg, 200)),
        ("floor", make_scene(cfg, [dict(pos=(4.0 + 5 * i, 8.0, 1.0)) for i in range(8)])),
    ):
        rd = RunDir.create(tmp_path / name, cfg)
        director = Director(rd, cfg, "live", 0)
        for step in range(10, 101, 10):
            director.observe(_synthetic_row(step))  # 200 alive in every window
        got[name] = director._vitals(s, 100)
    assert got["floor"].k_est == floor_only and not got["floor"].extinction_risk  # 200 / 409 = 0.49
    assert got["spread"].k_est == spread[200] and got["spread"].n_over_k < risk  # 200 / 1 400 = 0.14
    assert got["spread"].extinction_risk
    # the population that clears the flag: 0.2·K_est agents
    assert risk * spread[200] > 3 * risk * floor_only
    assert 250 / spread[250] < risk <= 250 / floor_only  # a recovered 250, still spread out, is still at risk


# --- the scripted live run and its replay -----------------------------------------------------------------

PLAN = {
    50: [
        prop("set_seasons", {"period": 5000, "amp": 0.5}),
        prop("regen_pulse", {"mult": 1.4}, 150),  # reverts at 200, after the checkpoint at 150
        prop("fertility_shift", {"patch": 1, "gain": 1.5}),
    ],
    100: [
        prop("habitat_loss", {"patch": 2}, 200),  # reverts at 300
        prop("immigration_pulse", {"count": 40}),
    ],
    150: [prop("fertility_shift", {"patch": 0, "dx": 12.0, "dy": -9.0})],
    250: [
        prop("catastrophe", {"x": 40.0, "y": 70.0, "radius": 30.0, "kill_p": 0.6}),  # a 2D-sized radius
        prop("drift_change", {"mult": 3.0}),
    ],
    300: [prop("regen_pulse", {"mult": 0.6}, 100)],  # reverts at 400
}


@contextlib.contextmanager
def scripted_policy() -> Iterator[None]:
    """The scripted rules, with the timed menu bounds shrunk so the reverts fall within a 400-step run."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("alife.director.rules.decide", scripted_decide(PLAN))
        for name in ("regen_pulse", "habitat_loss"):
            mp.setitem(MENU, name, dataclasses.replace(MENU[name], duration=(50, 2000)))
        yield


@pytest.fixture
def scripted() -> Iterator[None]:
    """The scripted policy for one test (the tests that validate, replay or resume the scripted log)."""
    with scripted_policy():
        yield


@pytest.fixture(scope="module")
def live(cfg3: Config, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The scripted live run. Its patches end with the run, so the real-rules tests never see them."""
    rd = RunDir.create(tmp_path_factory.mktemp("live3d") / "live", cfg3)
    rd.update_meta(start_step=0)
    with scripted_policy():
        res = runner.run(rd, cfg3, init_state(cfg3), END, quiet=True, director=Director(rd, cfg3, "live", 0))
    assert res.status == "done"
    return rd.path


def rows(path: Path) -> list[dict[str, Any]]:
    return [{k: v for k, v in r.items() if k not in PERF} for r in RunDir.open(path).metrics_rows()]


def digest(path: Path, step: int = END) -> str:
    rd = RunDir.open(path)
    return state_digest(load_checkpoint(rd.checkpoint_at(step), rd.config()))


def test_the_scripted_3d_run_logs_every_action_and_a_revert_across_a_checkpoint(
    live: Path, scripted: None
) -> None:
    recs = read_jsonl(live / DIRECTOR_LOG_FILE)
    headers = [r for r in recs if r.get("kind") == "header"]
    assert len(headers) == 1 and recs[0]["kind"] == "header" and recs[0]["dims"] == 3
    entries = recs[1:]
    assert {e["action"] for e in entries} == set(MENU)
    reverts = [(e["step_applied"], e["action"], e["reverts"]) for e in entries if e["source"] == "revert"]
    assert reverts == [(200, "regen_pulse", 50), (300, "habitat_loss", 100), (400, "regen_pulse", 300)]
    assert [e["step_applied"] for e in entries] == sorted(e["step_applied"] for e in entries)
    assert next(e for e in entries if e["step_applied"] == 200)["params"] == {"mult": 1.0}
    move = next(e for e in entries if e["action"] == "fertility_shift" and "x" in e["params"])
    assert set(move["params"]) == {"patch", "x", "y"} and set(move["prev"]) == {"patch", "x", "y"}
    cat = next(e for e in entries if e["action"] == "catastrophe")
    assert cat["params"]["radius"] == 25.0 and cat["params"]["center"] == [
        40.0,
        70.0,
    ]  # 30 clamped to the bound
    assert sorted({s for s, _ in RunDir.open(live).checkpoints()}) == [0, 150, 300, 400]
    assert (live / JOURNAL_FILE).exists() and (live / DIRECTOR_ARCHIVE_FILE).exists()
    assert cli.main(["inspect", str(live), "--check"]) == 0


def test_the_scripted_3d_run_is_a_valid_3d_run(live: Path) -> None:
    by_step = {r["step"]: r for r in rows(live)}
    assert all(r["ledger_error"] == 0 and r["diag"] & 0xFFFF == 0 for r in by_step.values())
    assert all("HEAT_BUOYANCY" in r["ledger"] for r in by_step.values())
    # the catastrophe applied after the row at 250 kills in the next interval (a column of radius 25), once
    assert by_step[260]["deaths_catastrophe"] > 0 and by_step[260]["env"]["cat_radius"] == 25.0
    assert sum(r["deaths_catastrophe"] for r in by_step.values()) == by_step[260]["deaths_catastrophe"]
    assert sum(r["immigrants"] for r in by_step.values()) == 40  # admitted in the volume (the step's rule)
    rd = RunDir.open(live)
    for step, _ in rd.checkpoints():
        s = load_checkpoint(rd.checkpoint_at(step), rd.config())
        assert isinstance(s, State3) and s.patch_pos.shape == (rd.config().fields.n_patches, 2)
    end = load_checkpoint(rd.checkpoint_at(END), rd.config())
    start = load_checkpoint(rd.checkpoint_at(0), rd.config())
    assert not np.array_equal(
        _np(end.patch_pos), _np(start.patch_pos)
    )  # the patches drifted and one was moved


def test_replay_of_a_3d_log_is_bitwise_identical(
    live: Path, cfg3: Config, tmp_path: Path, scripted: None
) -> None:
    rcfg = apply_overrides(cfg3, ["run.director=replay"])
    rd = RunDir.create(tmp_path / "replay", rcfg)
    rd.update_meta(start_step=0)
    shutil.copyfile(live / DIRECTOR_LOG_FILE, rd.path / DIRECTOR_LOG_FILE)
    res = runner.run(rd, rcfg, init_state(rcfg), END, quiet=True, director=Director(rd, rcfg, "replay", 0))
    assert res.status == "done"
    assert digest(rd.path) == digest(live) and rows(rd.path) == rows(live)
    for step in (150, 300):
        assert digest(rd.path, step) == digest(live, step)
    assert (rd.path / DIRECTOR_LOG_FILE).read_bytes() == (live / DIRECTOR_LOG_FILE).read_bytes()
    assert not (rd.path / JOURNAL_FILE).exists() and not (rd.path / DIRECTOR_ARCHIVE_FILE).exists()
    assert cli.main(["inspect", str(rd.path), "--check"]) == 0


def test_the_3d_director_changes_the_trajectory(live: Path, cfg3: Config, tmp_path: Path) -> None:
    """The comparisons are not vacuous: the same run without the director ends elsewhere."""
    off = apply_overrides(cfg3, ["run.director=off"])
    rd = RunDir.create(tmp_path / "off", off)
    runner.run(rd, off, init_state(off), END, quiet=True)
    assert digest(rd.path) != digest(live)


def test_replay_in_place_and_a_live_resume_from_a_mid_run_checkpoint(
    live: Path, tmp_path: Path, scripted: None
) -> None:
    copy = tmp_path / "copy"
    shutil.copytree(live, copy)
    log_before, journal_before = (copy / DIRECTOR_LOG_FILE).read_bytes(), (copy / JOURNAL_FILE).read_bytes()
    argv = ["resume", str(copy), "--checkpoint", str(CKPT), "--until", str(END), "--quiet"]
    assert cli.main([*argv, "--director", "replay"]) == 0
    assert (copy / DIRECTOR_LOG_FILE).read_bytes() == log_before  # a replay never writes the log
    assert (copy / JOURNAL_FILE).read_bytes() == journal_before
    assert digest(copy) == digest(live) and rows(copy) == rows(live)
    # a live resume re-decides the boundaries after the checkpoint (here from the scripted plan) and appends
    # no header: the policy is the one the 3D header already names
    assert cli.main([*argv, "--director", "live"]) == 0
    assert (copy / DIRECTOR_LOG_FILE).read_bytes() == log_before
    assert (copy / JOURNAL_FILE).read_bytes() == journal_before
    assert [r["kind"] for r in read_jsonl(copy / DIRECTOR_LOG_FILE) if r.get("kind")] == ["header"]
    assert digest(copy) == digest(live) and rows(copy) == rows(live)


# --- the random persona (arm R) in 3D ---------------------------------------------------------------------


def test_a_3d_run_cannot_follow_or_replay_a_2d_log(
    live: Path, tiny_cfg: Config, tmp_path: Path, scripted: None
) -> None:
    """The yoke log and a replay log are pre-validated against the run's dims."""
    d2 = apply_overrides(tiny_cfg, ["director.enabled=true", f"director.rule_every={EVERY}"])
    two_d = tmp_path / "two_d.jsonl"
    two_d.write_text(json.dumps(log.make_header(d2, 0, 2)) + "\n")
    cfg = apply_overrides(
        load_config(preset="tiny3d"),
        [*BASE, "director.persona=random", f"director.yoke_log={two_d}", "brain.weight_dtype=float32"],
    )
    with pytest.raises(log.LogError, match="dims"):
        yoked.load(cfg)
    follow = apply_overrides(cfg, [f"director.yoke_log={live / DIRECTOR_LOG_FILE}"])
    assert yoked.load(follow).steps == {50: 3, 100: 2, 150: 1, 250: 2, 300: 1}  # a 3D log is a fine yoke


def test_the_random_persona_runs_and_replays_in_3d(
    live: Path, cfg3: Config, tmp_path: Path, scripted: None
) -> None:
    """Arm R yoked to the scripted 3D run: its own safety rules, uniformly random moves at the yoked steps
    (a catastrophe's radius within the scaled bounds), and a bitwise replay of its log."""
    steps = 300
    rcfg = apply_overrides(
        cfg3,
        [
            "director.persona=random",
            f"director.yoke_log={live / DIRECTOR_LOG_FILE}",
            "director.budget_per_100k=4",
        ],
    )
    with pytest.MonkeyPatch.context() as mp:
        for name, spec in MENU.items():  # the budget of a tiny run: every move costs 1
            mp.setitem(MENU, name, dataclasses.replace(spec, cost=1))
        rd = RunDir.create(tmp_path / "arm_r", rcfg)
        rd.update_meta(start_step=0)
        res = runner.run(
            rd, rcfg, init_state(rcfg), steps, quiet=True, director=Director(rd, rcfg, "live", 0)
        )
        assert res.status == "done"
        recs = read_jsonl(rd.path / DIRECTOR_LOG_FILE)
        header = recs[0]
        assert header["kind"] == "header" and header["dims"] == 3 and header["persona"] == "random"
        assert header["params_persona"] == "gardener" and header["yoke_sha256"]
        moves = [e for e in recs[1:] if e.get("rule") == "random"]
        assert moves and all(e["yoked_to"] == e["step_applied"] for e in moves)
        for e in moves:
            if e["action"] == "catastrophe":
                assert 2.5 <= e["params"]["radius"] <= 25.0 and len(e["params"]["center"]) == 2
            if e["action"] == "fertility_shift" and "x" in e["params"]:
                assert set(e["params"]) == {"patch", "x", "y"}
        log.validate_log(recs, rcfg)  # every entry is within the 3D menu
        assert cli.main(["inspect", str(rd.path), "--check"]) == 0
        # its replay
        pcfg = apply_overrides(rcfg, ["run.director=replay"])
        rep = RunDir.create(tmp_path / "arm_r_replay", pcfg)
        rep.update_meta(start_step=0)
        shutil.copyfile(rd.path / DIRECTOR_LOG_FILE, rep.path / DIRECTOR_LOG_FILE)
        res = runner.run(
            rep, pcfg, init_state(pcfg), steps, quiet=True, director=Director(rep, pcfg, "replay", 0)
        )
        assert res.status == "done"
    assert digest(rep.path, steps) == digest(rd.path, steps) and rows(rep.path) == rows(rd.path)


# --- real rules: the Gardener in 3D -----------------------------------------------------------------------


FOUNDERS = (
    150  # n / K_est is 0.12 at the first boundary, under 0.2: the rescue fires, and it clears at step 60
)


def _gardener_argv(out: Path, steps: int, *extra: str, founders: int = FOUNDERS) -> list[str]:
    """The Gardener run of the tests. ``founders`` is 150 for the run that is resumed: with 40 (PR 4's value)
    the occupancy rule leaves the extinction flag set, and the rescue blocked by its cooldown, at every
    boundary (``gardener40``, §27.10), and a live resume then journals that blocked note a second time
    (``test_a_blocked_note_is_journalled_once_across_a_resume``: a defect of the director, not of 3D)."""
    return [
        "run",
        "--preset",
        "tiny3d",
        "--set",
        "brain.weight_dtype=float32",
        "--set",
        f"population.n_init={founders}",
        "--set",
        "director.rule_every=20",
        "--set",
        "run.heavy_metrics_every=20",
        "--set",
        "run.diversity_sample=64",
        "--set",
        "run.checkpoint_every=100",
        "--director",
        "live",
        "--steps",
        str(steps),
        "--quiet",
        *extra,
        "--out",
        str(out),
    ]


@pytest.fixture(scope="module")
def gardener(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("gardener3d") / "run"
    assert cli.main(_gardener_argv(out, 200)) == 0
    return out


def test_a_short_gardener_run_in_3d_keeps_the_ledger_exact(gardener: Path) -> None:
    rd = RunDir.open(gardener)
    assert rd.meta()["status"] == "done" and rd.meta()["dims"] == 3
    rs = rd.metrics_rows()
    assert [r["step"] for r in rs] == list(range(10, 201, 10))
    assert all(r["ledger_error"] == 0 and r["diag"] & 0xFFFF == 0 for r in rs)
    assert all(r["ledger"]["HEAT_BUOYANCY"] > 0 for r in rs)
    recs = read_jsonl(gardener / DIRECTOR_LOG_FILE)
    assert recs[0]["kind"] == "header" and recs[0]["dims"] == 3 and recs[0]["persona"] == "gardener"
    rescue = recs[1:]
    assert {(e["step_applied"], e["action"], e["rule"]) for e in rescue} == {
        (20, "regen_pulse", "extinction_rescue"),
        (20, "immigration_pulse", "extinction_rescue"),
    }
    count = next(e for e in rescue if e["action"] == "immigration_pulse")["params"]["count"]
    assert sum(r["immigrants"] for r in rs) == count  # the immigrants arrive in the volume
    # the population ends above its founders and the rescue's immigrants together: the survivors reproduced
    assert rs[-1]["n_alive"] > FOUNDERS + count
    assert rs[0]["env"]["regen_mult"] == 1.0 and rs[2]["env"]["regen_mult"] == 1.5
    assert cli.main(["inspect", str(gardener), "--check"]) == 0


def test_the_vitals_of_a_3d_run_use_the_occupancy_weighted_k_est(gardener: Path, tmp_path: Path) -> None:
    """The ``k_est`` logged at the first boundary is ``metrics.carrying_capacity`` of the state before the
    director acted: the layers' regeneration weighted by where the agents live (§27.10), between the floor
    layer's alone and all the layers'."""
    rcfg = apply_overrides(
        load_config(preset="tiny3d", overrides=["brain.weight_dtype=float32", "population.n_init=150"]),
        [
            "director.enabled=true",
            "director.rule_every=20",
            "run.heavy_metrics_every=20",
            "run.diversity_sample=64",
        ],
    )
    twin = RunDir.create(tmp_path / "twin", rcfg)  # no director: the same trajectory up to the first boundary
    res = runner.run(twin, rcfg, init_state(rcfg), 20, quiet=True)
    s = res.state
    k = metrics.carrying_capacity(s, rcfg, s.step, s.env)
    rate = _np(fields.regen_rate(s.patch_pos, rcfg, s.step, s.env)).astype(np.int64)
    assert rate.shape[0] == 6 and rate.sum() > 2 * rate[0].sum()  # the layers above would more than double it
    c_ref = (
        rcfg.energy.basal_cost
        + metrics.founder_brain_cost(rcfg)
        + 50  # layout.C_REF_MARGIN
        + int(np.floor(np.float32(rcfg.energy.buoyancy_cap_cost) * np.float32(rcfg.mutation.buoy_init)))
    )
    z = _np(s.pos)[_np(s.alive), 2]
    layer = np.clip(np.floor(z / np.float32(rcfg.world.field_cell)).astype(np.int64), 0, 5)
    w = np.minimum(1.0, 6 * np.bincount(layer, minlength=6) / z.size)
    host = float((rate.reshape(6, -1).sum(axis=1) * w).sum()) / c_ref
    assert k == pytest.approx(host, rel=1e-12)
    assert rate[0].sum() / c_ref < k < rate.sum() / c_ref  # floor-only 409 < K_est < all the layers' 1 456
    first = next(e for e in read_jsonl(gardener / DIRECTOR_LOG_FILE) if e.get("kind") != "header")
    assert first["vitals_step"] == 20 and first["vitals"]["k_est"] == round(k, 4)
    assert first["vitals"]["n_over_k"] == round(first["vitals"]["n_alive"] / k, 4)
    assert first["vitals"]["extinction_risk"] is True


def test_a_resumed_3d_live_run_is_the_uninterrupted_run_and_its_log_replays(
    gardener: Path, tmp_path: Path
) -> None:
    """A live resume rebuilds the vital signs from the rows: same log (and no second header), journal and
    archive; the log then replays into the same state and rows."""
    two = tmp_path / "two"
    assert cli.main(_gardener_argv(two, 100)) == 0
    assert cli.main(["resume", str(two), "--until", "200", "--quiet"]) == 0  # inherits the live mode
    for name in (DIRECTOR_LOG_FILE, JOURNAL_FILE, DIRECTOR_ARCHIVE_FILE):
        assert (gardener / name).read_bytes() == (two / name).read_bytes(), name
    assert [r["kind"] for r in read_jsonl(two / DIRECTOR_LOG_FILE) if r.get("kind")] == ["header"]
    assert digest(two, 200) == digest(gardener, 200) and rows(two) == rows(gardener)
    rep = tmp_path / "rep"
    argv = ["run", "--init-from", f"{gardener}:0", "--director", "replay", "--replay-log"]
    argv += [str(gardener / DIRECTOR_LOG_FILE), "--steps", "200", "--set", "run.checkpoint_every=100"]
    assert cli.main([*argv, "--quiet", "--out", str(rep)]) == 0
    assert digest(rep, 200) == digest(gardener, 200) and rows(rep) == rows(gardener)
    assert cli.main(["inspect", str(rep), "--check"]) == 0


# --- the rescue under the occupancy rule: a known limit (§27.10) ------------------------------------------


def _floor_layer_k(s: State3, cfg: Config) -> float:
    """PR 4's ``K_est`` of the state: the live regeneration of the floor layer over ``c_ref``."""
    rate = _np(fields.regen_rate(s.patch_pos, cfg, s.step, s.env)).astype(np.int64)
    c_ref = (
        cfg.energy.basal_cost
        + metrics.founder_brain_cost(cfg)
        + 50  # layout.C_REF_MARGIN
        + int(np.floor(np.float32(cfg.energy.buoyancy_cap_cost) * np.float32(cfg.mutation.buoy_init)))
    )
    return float(rate[0].sum()) / c_ref


@pytest.fixture(scope="module")
def gardener40(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[tuple[int, Any, float]]]:
    """The Gardener run of PR 4 (40 founders, 200 steps), with each boundary's vital signs and the floor-layer
    ``K_est`` of the same state recorded as ``(step, Vitals, floor-only K_est)``."""
    seen: list[tuple[int, Any, float]] = []
    vitals = Director._vitals

    def spy(self: Director, state: State3, t: int) -> Any:
        v = vitals(self, state, t)
        seen.append((t, v, _floor_layer_k(state, self.cfg)))
        return v

    out = tmp_path_factory.mktemp("gardener3d_40") / "run"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Director, "_vitals", spy)
        assert cli.main(_gardener_argv(out, 200, founders=40)) == 0
    return out, seen


def test_a_3d_rescue_raises_its_own_k_est_and_the_flag_stays_set(
    gardener40: tuple[Path, list[tuple[int, Any, float]]],
) -> None:
    """40 founders spread over the column are 0.03 of their ``K_est`` (1 309, 3.2x the floor layer's 409);
    the rescue sends ``0.1·K_est`` immigrants, uniform in the volume, with a 1.5x regeneration pulse, so the
    survivors and the immigrants keep every layer's weight near 1 and ``K_est`` rises to 1 700-2 000. The
    population recovers to 250-290 agents (0.4 to 0.5 of the floor layer's value) and is still at
    ``n / K_est`` 0.14 to 0.15 < ``risk_k_frac``: the extinction flag is set at every boundary, which the
    floor-layer rule of PR 4 cleared at step 60. A limit of the rule to calibrate before a 3D director run
    (§27.10, §27.13); this pins what the rule does so that changing it is a decision."""
    out, seen = gardener40
    risk = RULE_PARAMS.risk_k_frac
    assert [t for t, _, _ in seen] == list(range(20, 201, 20))
    _, v0, floor0 = seen[0]
    assert v0.n_alive == 40 and v0.k_est > 3 * floor0 and v0.n_over_k < 0.05 and v0.extinction_risk
    rescue = [e for e in read_jsonl(out / DIRECTOR_LOG_FILE) if e.get("action") == "immigration_pulse"]
    assert (
        len(rescue) == 1 and abs(rescue[0]["params"]["count"] - RULE_PARAMS.rescue_imm_frac * v0.k_est) <= 1
    )
    for t, v, floor_k in seen[3:]:  # from step 80, when the immigrants and their children are the population
        assert v.n_alive > 5 * 40 and v.k_est > 2.5 * floor_k, (t, v.n_alive, v.k_est)
        assert v.n_over_k < risk <= v.n_alive / floor_k, (t, v.n_over_k, v.n_alive / floor_k)
        assert v.extinction_risk  # the flag the floor layer's K_est would have cleared


@pytest.mark.parametrize("config", ["tiny_cfg", "tiny3d_cfg"])
@pytest.mark.xfail(
    strict=True,
    reason="Director._blocked is per process and not rebuilt on a live resume (DESIGN §27.10): a rule that "
    "stays blocked across the resume is journalled again. Shared with 2D. Remove the marker with the fix.",
)
def test_a_blocked_note_is_journalled_once_across_a_resume(
    config: str, request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    """A condition that persists is noted once, not at every boundary: a resumed director must not note a
    rule again that was already blocked at the boundary before the resume."""
    cfg = apply_overrides(
        request.getfixturevalue(config),
        ["director.enabled=true", "run.director=live", "director.rule_every=100"],
    )
    note = "extinction_rescue (n/K_est 0.15, fall 0.01): blocked [regen_pulse: regen_pulse is cooling down]"
    rd = RunDir.create(tmp_path / "run", cfg)
    Director(rd, cfg, "live", 0)._journal(100, [], [note])  # the process that is interrupted
    Director(rd, cfg, "live", 100)._journal(200, [], [note])  # the resumed process: the rule is still blocked
    lines = read_jsonl(rd.path / JOURNAL_FILE)
    assert [ln["step"] for ln in lines] == [100]
