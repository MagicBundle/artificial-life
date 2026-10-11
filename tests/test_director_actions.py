"""The director's intervention menu: bounds, clamping, strict validation and each action in the real step."""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from alife import fields
from alife.config import Config, ConfigError, apply_overrides, dims
from alife.director import actions
from alife.director.actions import ActionError, EnvView, apply, clamp, env_view, validate_entry
from alife.state import CounterIdx, Diag, LedgerIdx, State, init_state, state_digest, to_numpy
from helpers import run_checked

C = CounterIdx
F32 = np.float32


@pytest.fixture(scope="module")
def dcfg(tiny_cfg: Config) -> Config:
    """``tiny`` (f32) with the director enabled, deciding every 100 steps."""
    return apply_overrides(tiny_cfg, ["director.enabled=true", "director.rule_every=100"])


@pytest.fixture(scope="module")
def s0(dcfg: Config) -> State:
    return init_state(dcfg)


def entry(
    cfg: Config, s: State, action: str, raw: dict, duration: int | None = None, step: int = 100
) -> dict:
    c = clamp(action, raw, duration, env_view(s), cfg)
    return {
        "step_applied": step,
        "source": "rule",
        "action": action,
        "params": c.params,
        "prev": c.prev,
        "duration": c.duration,
    }


def revert_of(e: dict, step: int) -> dict:
    return {
        "step_applied": step,
        "source": "revert",
        "action": e["action"],
        "params": dict(e["prev"]),
        "prev": {},
        "duration": None,
    }


# --- the menu ---


def test_menu_is_the_spec_table() -> None:
    m = actions.MENU
    assert set(m) == {
        "set_seasons",
        "fertility_shift",
        "habitat_loss",
        "regen_pulse",
        "immigration_pulse",
        "catastrophe",
        "drift_change",
    }  # no mutation_floor_nudge
    assert {k: (v.cost, v.cooldown) for k, v in m.items()} == {
        "set_seasons": (1, 50_000),
        "fertility_shift": (1, 20_000),
        "habitat_loss": (2, 50_000),
        "regen_pulse": (1, 30_000),
        "immigration_pulse": (1, 20_000),
        "catastrophe": (3, 100_000),
        "drift_change": (1, 50_000),
    }
    assert m["set_seasons"].bounds == {"period": (2_000, 100_000), "amp": (0.0, 0.9)}  # no runtime mode
    assert m["fertility_shift"].bounds == {"gain": (0.25, 2.0)} and actions.MOVE_MAX_FRAC == 0.25
    assert m["habitat_loss"].duration == (10_000, 200_000) and m["regen_pulse"].duration == (5_000, 50_000)
    assert m["regen_pulse"].bounds == {"mult": (0.4, 1.8)}
    assert m["immigration_pulse"].bounds == {"count": (16, 1024)} and actions.IMM_PENDING_CAP == 1024
    assert m["catastrophe"].bounds == {"kill_p": (0.2, 0.9)} and actions.CATASTROPHE_RADIUS == (20.0, 200.0)
    assert m["drift_change"].bounds == {"mult": (0.5, 10.0)}
    assert set(actions.TIMED) == {"habitat_loss", "regen_pulse"}


# --- live: clamp ---


@pytest.mark.parametrize(
    ("action", "raw", "duration", "want"),
    [
        ("set_seasons", {"period": 10, "amp": -1}, None, {"period": 2_000, "amp": 0.0}),
        ("set_seasons", {"period": 5e6, "amp": 5}, None, {"period": 100_000, "amp": float(F32(0.9))}),
        ("fertility_shift", {"patch": 1, "gain": 9}, None, {"patch": 1, "gain": 2.0}),
        ("fertility_shift", {"patch": 1, "gain": 0.01}, None, {"patch": 1, "gain": 0.25}),
        ("habitat_loss", {"patch": 2}, 1, {"patch": 2, "gain": 0.0}),
        ("regen_pulse", {"mult": 10}, 1, {"mult": float(F32(1.8))}),
        ("regen_pulse", {"mult": 0}, 10**9, {"mult": float(F32(0.4))}),
        ("immigration_pulse", {"count": 1}, None, {"count": 16}),
        ("immigration_pulse", {"count": 5000}, None, {"count": 1024}),
        ("drift_change", {"mult": 0.1}, None, {"mult": 0.5}),
        ("drift_change", {"mult": 99}, None, {"mult": 10.0}),
    ],
)
def test_clamp_brings_a_request_inside_the_menu(
    dcfg: Config, s0: State, action: str, raw: dict, duration: int | None, want: dict
) -> None:
    c = clamp(action, raw, duration, env_view(s0), dcfg)
    assert c.params == want
    validate_entry(
        {
            "step_applied": 100,
            "source": "rule",
            "action": action,
            "params": c.params,
            "prev": c.prev,
            "duration": c.duration,
        },
        dcfg,
        100,
    )


def test_timed_durations_are_clamped_then_rounded_up_to_a_rule_boundary(tiny_cfg: Config, s0: State) -> None:
    cfg = apply_overrides(tiny_cfg, ["director.enabled=true", "director.rule_every=300"])
    view = env_view(init_state(cfg))
    got = {
        (a, d): clamp(a, {"patch": 0, "mult": 1.2}, d, view, cfg).duration
        for a in ("habitat_loss", "regen_pulse")
        for d in (1, 5_000, 12_345, 10**9)
    }
    assert got[("regen_pulse", 1)] == 5_100 and got[("regen_pulse", 5_000)] == 5_100  # 5000 -> 17 * 300
    assert got[("regen_pulse", 12_345)] == 12_600 and got[("regen_pulse", 10**9)] == 50_100
    assert got[("habitat_loss", 1)] == 10_200 and got[("habitat_loss", 10**9)] == 200_100
    assert all(d % 300 == 0 for d in got.values())


def test_regen_pulse_remembers_what_to_restore(dcfg: Config, s0: State) -> None:
    boosted = env_view(s0)._replace(regen_mult=1.4)
    assert clamp("regen_pulse", {"mult": 1.1}, 6_000, boosted, dcfg).prev == {"mult": 1.4}
    # superseding an active pulse: the base value is restored, not the pulse being replaced
    assert clamp("regen_pulse", {"mult": 1.1}, 6_000, boosted, dcfg, restore={"mult": 1.0}).prev == {
        "mult": 1.0
    }
    assert clamp("habitat_loss", {"patch": 3}, 10_000, boosted, dcfg).prev == {"patch": 3, "gain": 1.0}


def test_catastrophe_radius_is_clamped_to_the_world(dcfg: Config, s0: State) -> None:
    L = dcfg.world.size  # 128: the upper bound is min(200, L/2) = 64
    view = env_view(s0)
    lo = clamp("catastrophe", {"x": 1, "y": 2, "radius": 1, "kill_p": 0}, None, view, dcfg).params
    hi = clamp("catastrophe", {"x": 1, "y": 2, "radius": 1e4, "kill_p": 5}, None, view, dcfg).params
    assert (lo["radius"], lo["kill_p"]) == (20.0, float(F32(0.2)))
    assert (hi["radius"], hi["kill_p"]) == (L / 2, float(F32(0.9)))
    big = apply_overrides(dcfg, ["world.size=1024", "population.n_init=256"])
    assert (
        clamp("catastrophe", {"x": 1, "y": 2, "radius": 1e4, "kill_p": 0.5}, None, view, big).params["radius"]
        == 200.0
    )
    wrapped = clamp(
        "catastrophe", {"x": -1.0, "y": L + 3, "radius": 30, "kill_p": 0.5}, None, view, dcfg
    ).params
    assert wrapped["center"] == [L - 1.0, 3.0]


@pytest.mark.parametrize(
    ("action", "raw", "duration"),
    [
        ("nonsense", {}, None),
        ("set_seasons", {"period": 5000}, None),  # missing amp
        ("set_seasons", {"period": float("nan"), "amp": 0.3}, None),
        ("drift_change", {"mult": float("inf")}, None),
        ("drift_change", {"mult": True}, None),
        ("fertility_shift", {"patch": -1, "gain": 1.0}, None),
        ("fertility_shift", {"patch": 4, "gain": 1.0}, None),  # tiny has 4 patches
        ("fertility_shift", {"patch": 0.5, "gain": 1.0}, None),
        ("fertility_shift", {"patch": 0}, None),  # neither a gain nor a move
        ("habitat_loss", {"patch": 0}, None),  # a timed action needs its duration
        ("regen_pulse", {"mult": 1.2}, float("nan")),
    ],
)
def test_clamp_refuses_requests_it_cannot_make_sense_of(
    dcfg: Config, s0: State, action: str, raw: dict, duration: Any
) -> None:
    with pytest.raises(ActionError):
        clamp(action, raw, duration, env_view(s0), dcfg)
    assert issubclass(ActionError, ConfigError)  # the CLI reports it as exit 2


def test_the_immigration_queue_is_capped_in_total(dcfg: Config, s0: State) -> None:
    view = env_view(s0)._replace(imm_request=1000)
    assert clamp("immigration_pulse", {"count": 500}, None, view, dcfg).params == {"count": 24}
    full = view._replace(imm_request=1020)  # less than the minimum pulse would fit
    with pytest.raises(ActionError, match="already pending"):
        clamp("immigration_pulse", {"count": 500}, None, full, dcfg)


def test_a_patch_move_is_at_most_a_quarter_of_the_world_and_wraps_strictly_below_L(
    dcfg: Config, s0: State
) -> None:
    L = dcfg.world.size
    view = env_view(s0)
    pos = np.array(view.patch_pos)
    pos[0] = [L - 0.5, 0.5]
    view = view._replace(patch_pos=pos)
    far = clamp("fertility_shift", {"patch": 0, "dx": 1e3, "dy": 0.0}, None, view, dcfg)
    assert far.params["x"] == pytest.approx((L - 0.5 + L / 4) % L, abs=1e-4) and far.params["y"] == 0.5
    assert far.prev == {"patch": 0, "x": L - 0.5, "y": 0.5}
    diag = clamp("fertility_shift", {"patch": 0, "dx": 3e3, "dy": 4e3}, None, view, dcfg).params
    assert math.hypot(diag["x"] - (L - 0.5 + 0.6 * L / 4 - L), diag["y"] - (0.5 + 0.8 * L / 4)) < 1e-3
    # the float32 seam: remainder(-eps, L) rounds to L, and the guard maps it back to 0
    seam = view._replace(patch_pos=np.zeros_like(pos))
    for dx in (-1e-9, -1e-12, 0.0, 3e-6):
        p = clamp("fertility_shift", {"patch": 1, "dx": dx, "dy": 0.0}, None, seam, dcfg).params
        assert 0.0 <= p["x"] < L and 0.0 <= p["y"] < L
    edge = np.zeros_like(pos)
    edge[2] = [np.nextafter(F32(L), F32(0)), 0.0]
    got = clamp(
        "fertility_shift", {"patch": 2, "dx": 5e-6, "dy": 0.0}, None, view._replace(patch_pos=edge), dcfg
    )
    assert got.params["x"] < L


# --- replay: strict validation ---


def good_entries(cfg: Config, s: State) -> list[dict]:
    pos = to_numpy(s.patch_pos)[0]
    return [
        entry(cfg, s, "set_seasons", {"period": 5000, "amp": 0.4}),
        entry(cfg, s, "fertility_shift", {"patch": 1, "gain": 1.5}),
        entry(cfg, s, "fertility_shift", {"patch": 0, "dx": 10.0, "dy": -5.0}),
        entry(cfg, s, "habitat_loss", {"patch": 2}, 10_000),
        entry(cfg, s, "regen_pulse", {"mult": 1.3}, 5_000),
        entry(cfg, s, "immigration_pulse", {"count": 64}),
        entry(cfg, s, "catastrophe", {"x": pos[0], "y": pos[1], "radius": 30.0, "kill_p": 0.5}),
        entry(cfg, s, "drift_change", {"mult": 3.0}),
    ]


def test_every_clamped_entry_validates(dcfg: Config, s0: State) -> None:
    for e in good_entries(dcfg, s0):
        validate_entry(e, dcfg, 100)
    hab = entry(dcfg, s0, "habitat_loss", {"patch": 2}, 10_000)
    validate_entry(revert_of(hab, 10_100), dcfg, 100)


def mutate(e: dict, **kw: Any) -> dict:
    return {**e, **kw}


def bad_cases(cfg: Config, s: State) -> dict[str, dict]:
    g = {e["action"] + str(i): e for i, e in enumerate(good_entries(cfg, s))}
    seasons, gain, move = g["set_seasons0"], g["fertility_shift1"], g["fertility_shift2"]
    hab, regen, imm, cat, drift = (
        g["habitat_loss3"],
        g["regen_pulse4"],
        g["immigration_pulse5"],
        g["catastrophe6"],
        g["drift_change7"],
    )
    return {
        "unknown action": mutate(seasons, action="mutation_floor_nudge"),
        "unknown source": mutate(seasons, source="llm"),
        "step not a multiple": mutate(seasons, step_applied=150),
        "step zero": mutate(seasons, step_applied=0),
        "step is a float": mutate(seasons, step_applied=200.0),
        "params not an object": mutate(seasons, params=[1, 2]),
        "extra param": mutate(seasons, params={**seasons["params"], "mode": "per_patch"}),
        "missing param": mutate(seasons, params={"period": 5000}),
        "period below": mutate(seasons, params={"period": 1999, "amp": 0.4}),
        "period fractional": mutate(seasons, params={"period": 5000.5, "amp": 0.4}),
        "amp above": mutate(seasons, params={"period": 5000, "amp": 0.95}),
        "amp is nan": mutate(seasons, params={"period": 5000, "amp": float("nan")}),
        "gain below": mutate(gain, params={"patch": 1, "gain": 0.2}),
        "patch out of range": mutate(gain, params={"patch": 4, "gain": 1.0}),
        "patch is a float": mutate(gain, params={"patch": 1.5, "gain": 1.0}),
        "move without prev": mutate(move, prev={}),
        "move beyond L/4": mutate(move, params={**move["params"], "x": (move["prev"]["x"] + 40.0) % 128.0}),
        "move x == L": mutate(move, params={**move["params"], "x": 128.0}),
        "habitat gain nonzero": mutate(hab, params={"patch": 2, "gain": 0.5}),
        "habitat duration short": mutate(hab, duration=9_900),
        "habitat duration long": mutate(hab, duration=200_100 + 100),
        "habitat duration off-grid": mutate(hab, duration=10_050),
        "habitat no duration": mutate(hab, duration=None),
        "regen above": mutate(regen, params={"mult": 1.81}),
        "regen device range": mutate(regen, params={"mult": 4.5}),
        "instant with duration": mutate(drift, duration=1000),
        "count below": mutate(imm, params={"count": 15}),
        "count above": mutate(imm, params={"count": 1025}),
        "count fractional": mutate(imm, params={"count": 20.5}),
        "radius below": mutate(cat, params={**cat["params"], "radius": 19.0}),
        "radius above L/2": mutate(cat, params={**cat["params"], "radius": 65.0}),
        "kill_p below": mutate(cat, params={**cat["params"], "kill_p": 0.1}),
        "kill_p above": mutate(cat, params={**cat["params"], "kill_p": 0.95}),
        "centre not a pair": mutate(cat, params={**cat["params"], "center": [1.0]}),
        "centre outside": mutate(cat, params={**cat["params"], "center": [128.0, 3.0]}),
        "drift above": mutate(drift, params={"mult": 10.5}),
        "revert of an instant action": {**drift, "source": "revert"},
        "revert with a duration": {**revert_of(hab, 10_100), "duration": 100},
        "revert gain device range": {**revert_of(hab, 10_100), "params": {"patch": 2, "gain": 4.5}},
    }


def test_validation_rejects_everything_outside_the_menu(dcfg: Config, s0: State) -> None:
    cases = bad_cases(dcfg, s0)
    assert len(cases) >= 35
    for name, e in cases.items():
        with pytest.raises(ActionError):
            validate_entry(e, dcfg, 100)
            pytest.fail(f"accepted: {name}")
    with pytest.raises(ActionError):
        validate_entry("not a dict", dcfg, 100)


# --- apply: each action through the real compiled step ---


def shapes(s: State) -> list[tuple]:
    return [(a.shape, a.dtype) for a in s]


def test_apply_keeps_every_shape_and_dtype_so_nothing_retraces(dcfg: Config, s0: State) -> None:
    s = s0
    for e in good_entries(dcfg, s0):
        s = apply(s, e, dcfg)
        assert shapes(s) == shapes(s0)


def test_regen_pulse_scales_regeneration_and_the_revert_restores_it(dcfg: Config, s0: State) -> None:
    e = entry(dcfg, s0, "regen_pulse", {"mult": 1.6}, 5_000)
    pulsed = apply(s0, e, dcfg)
    assert float(to_numpy(pulsed.env_regen_mult)) == F32(1.6)
    inj = [
        int(to_numpy(h[0][2].ledger)[LedgerIdx.INJECT_REGEN])
        for h in (run_checked(dcfg, 1, st)[1] for st in (s0, pulsed))
    ]
    assert 0 < inj[0] < inj[1]
    back = apply(pulsed, revert_of(e, 5_100), dcfg)
    assert state_digest(back) == state_digest(s0)  # nothing else changed


def test_habitat_loss_zeroes_a_patch_and_the_revert_restores_its_gain(dcfg: Config, s0: State) -> None:
    shifted = apply(s0, entry(dcfg, s0, "fertility_shift", {"patch": 2, "gain": 1.5}), dcfg)
    e = entry(dcfg, shifted, "habitat_loss", {"patch": 2}, 10_000)
    assert e["prev"] == {"patch": 2, "gain": 1.5}
    lost = apply(shifted, e, dcfg)
    assert to_numpy(lost.env_patch_gain).tolist() == [1.0, 1.0, 0.0, 1.0]
    step = mx.array(0, dtype=mx.int32)
    before = to_numpy(fields.fertility(s0.patch_pos, dcfg, step, shifted.env)).sum()
    after = to_numpy(fields.fertility(s0.patch_pos, dcfg, step, lost.env)).sum()
    assert after < before  # the lost patch no longer contributes (the field is clipped at 1 near its centre)
    restored = apply(lost, revert_of(e, 10_100), dcfg)
    assert to_numpy(restored.env_patch_gain).tolist() == [1.0, 1.0, 1.5, 1.0]
    assert state_digest(restored) == state_digest(shifted)


def test_fertility_shift_gain_and_move_touch_only_their_patch(dcfg: Config, s0: State) -> None:
    g = apply(s0, entry(dcfg, s0, "fertility_shift", {"patch": 1, "gain": 0.5}), dcfg)
    assert to_numpy(g.env_patch_gain).tolist() == [1.0, 0.5, 1.0, 1.0]
    assert np.array_equal(to_numpy(g.patch_pos), to_numpy(s0.patch_pos))
    mv = entry(dcfg, s0, "fertility_shift", {"patch": 3, "dx": 12.0, "dy": -7.0})
    moved = apply(s0, mv, dcfg)
    pos0, pos1 = to_numpy(s0.patch_pos), to_numpy(moved.patch_pos)
    assert pos1[3].tolist() == [F32(mv["params"]["x"]), F32(mv["params"]["y"])] and np.array_equal(
        pos1[:3], pos0[:3]
    )
    assert np.array_equal(to_numpy(moved.env_patch_gain), to_numpy(s0.env_patch_gain))
    assert (pos1 >= 0).all() and (pos1 < dcfg.world.size).all()
    run_checked(dcfg, 3, moved)  # the engine takes the moved patch (and keeps drifting it)


def test_set_seasons_replaces_the_season_parameters(dcfg: Config, s0: State) -> None:
    se = apply(s0, entry(dcfg, s0, "set_seasons", {"period": 4000, "amp": 0.5}), dcfg)
    assert int(to_numpy(se.env_season_period)) == 4000 and float(to_numpy(se.env_season_amp)) == F32(0.5)
    w = to_numpy(fields.season_weights(mx.array(1000, dtype=mx.int32), dcfg, se.env))
    assert np.allclose(
        w, 1 + 0.5 * math.sin(2 * math.pi * 1000 / 4000), atol=1e-6
    )  # global mode: same for all
    off = to_numpy(fields.season_weights(mx.array(1000, dtype=mx.int32), dcfg, s0.env))
    assert np.array_equal(off, np.ones_like(off))
    run_checked(dcfg, 3, se)


def test_immigration_pulse_queues_immigrants_the_step_admits_over_time(dcfg: Config, s0: State) -> None:
    s = apply(s0, entry(dcfg, s0, "immigration_pulse", {"count": 100}), dcfg)
    assert int(to_numpy(s.env_imm_request)) == 100
    s = apply(s, entry(dcfg, s, "immigration_pulse", {"count": 40}), dcfg)
    assert int(to_numpy(s.env_imm_request)) == 140  # added to what is pending
    _, hist = run_checked(dcfg, 8, s)
    assert sum(int(to_numpy(o.counters)[C.IMMIGRANTS]) for _, _, o in hist) == 140
    over = s0._replace(env_imm_request=mx.array(1000, dtype=mx.int32))
    with pytest.raises(ActionError, match="exceed the cap"):
        apply(over, {"action": "immigration_pulse", "params": {"count": 100}}, dcfg)


def test_catastrophe_kills_inside_the_disc_for_one_step(dcfg: Config, s0: State) -> None:
    L = dcfg.world.size
    e = entry(dcfg, s0, "catastrophe", {"x": L / 2, "y": L / 2, "radius": L / 3, "kill_p": 0.9})
    _, ((_, s1, out),) = run_checked(dcfg, 1, apply(s0, e, dcfg))
    assert int(to_numpy(out.counters)[C.DEATHS_CATASTROPHE]) > 20
    assert float(to_numpy(s1.env_cat_kill_p)) == 0.0 and float(to_numpy(s1.env_cat_radius)) == F32(L / 3)


def test_drift_change_scales_patch_motion(tiny_cfg: Config) -> None:
    cfg = apply_overrides(tiny_cfg, ["director.enabled=true", "fields.patch_drift=0.4"])
    s = init_state(cfg)
    fast = apply(s, entry(cfg, s, "drift_change", {"mult": 4.0}), cfg)
    a, _ = run_checked(cfg, 1, s)
    b, _ = run_checked(cfg, 1, fast)
    L, p0 = cfg.world.size, to_numpy(s.patch_pos)

    def shift(x: State) -> np.ndarray:
        d = to_numpy(x.patch_pos) - p0
        return d - L * np.round(d / L)

    assert np.allclose(shift(b), 4 * shift(a), atol=1e-4)


def test_a_scripted_run_of_every_action_closes_the_ledger_and_is_deterministic(
    dcfg: Config, s0: State
) -> None:
    L = dcfg.world.size
    plan = {
        2: ("set_seasons", {"period": 3000, "amp": 0.6}, None),
        3: ("fertility_shift", {"patch": 0, "dx": 9.0, "dy": 9.0}, None),
        4: ("habitat_loss", {"patch": 1}, 10_000),
        6: ("immigration_pulse", {"count": 90}, None),
        8: ("catastrophe", {"x": L / 3, "y": L / 3, "radius": 40.0, "kill_p": 0.8}, None),
        10: ("regen_pulse", {"mult": 0.5}, 5_000),
        12: ("drift_change", {"mult": 5.0}, None),
        14: ("fertility_shift", {"patch": 2, "gain": 1.8}, None),
    }

    def pre(i: int, s: State) -> State:
        if i not in plan:
            return s
        action, raw, duration = plan[i]
        return apply(s, entry(dcfg, s, action, raw, duration), dcfg)

    a, hist = run_checked(dcfg, 24, pre=pre)  # full_check against the edited previous state at every step
    assert sum(int(to_numpy(o.counters)[C.DEATHS_CATASTROPHE]) for _, _, o in hist) > 10
    b, _ = run_checked(dcfg, 24, pre=pre)
    assert state_digest(a) == state_digest(b)


def test_an_unclamped_out_of_range_write_is_caught_by_the_engine_and_by_apply(
    dcfg: Config, s0: State
) -> None:
    raw = s0._replace(env_regen_mult=mx.array(5.0, dtype=mx.float32))  # bypasses clamp and apply
    fn = __import__("alife.step", fromlist=["make_step"]).make_step(dcfg)
    s1, out = fn(raw)
    mx.eval(s1, out)
    assert int(to_numpy(s1.diag)) & Diag.FIELD_RANGE
    bad = {"action": "regen_pulse", "params": {"mult": 5.0}}
    with pytest.raises(ActionError, match="device ranges"):
        apply(s0, bad, dcfg)
    with pytest.raises(ActionError):
        apply(s0, {"action": "drift_change", "params": {"mult": 21.0}}, dcfg)


def test_env_view_reads_the_levers(dcfg: Config, s0: State) -> None:
    v = env_view(s0)
    assert isinstance(v, EnvView) and v.regen_mult == 1.0 and v.imm_request == 0 and v.season_period == 0
    assert v.patch_gain.shape == (dims(dcfg).pp,) and v.patch_pos.shape == (dims(dcfg).pp, 2)
    assert actions.lever_keys("regen_pulse", {"mult": 1}) == ("regen",)
    assert actions.lever_keys("habitat_loss", {"patch": 3, "gain": 0}) == ("patch:3",)
    assert actions.lever_keys("fertility_shift", {"patch": 1, "gain": 1}) == ("patch:1",)
    assert actions.lever_keys("fertility_shift", {"patch": 1, "x": 1, "y": 1}) == ()  # a move holds no lever
    assert actions.lever_keys("catastrophe", {}) == ()
    assert actions.round_up(5000, 300) == 5100 and actions.round_up(5100, 300) == 5100
