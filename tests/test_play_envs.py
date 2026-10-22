"""``ENVS`` (DESIGN §33.1.9, §33.2.7): the environment a player sees, from the state and the director's
log."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alife import play, weather
from alife.runio import RunDir
from play_helpers import FakeClock, decoded, director_log, run_paced, snapshots, world

RUN = play.PlayOptions(rate=float("inf"), paused=False, fresh=True)


def envs(out) -> dict[int, dict]:
    return {r.step: r.f["json"] for r in decoded(out).records if r.tag == "ENVS"}


def test_a_live_director_shows_up_in_effects_levers_and_cooldowns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots(monkeypatch, 10)
    cfg = world("2d", "fields.season_period=100", "fields.season_amplitude=0.5", director=True)
    rd, _, _, out = run_paced(cfg, tmp_path / "w", 50, [], options=RUN)
    entries = {e["action"]: e for e in director_log(rd) if e.get("kind") != "header"}
    pulse = entries["regen_pulse"]
    assert pulse["step_applied"] == 20  # the rule tier acted at its first boundary
    e = envs(out)
    assert list(e) == [0, 10, 20, 30, 40, 50]  # at start and at every metrics boundary
    assert (
        e[10]["effects"] == [] and e[10]["levers"] == {} and e[10]["cooldowns"] == []
    )  # before the director acts
    after = e[20]
    assert after["effects"] == [
        {
            "action": "regen_pulse",
            "params": pulse["params"],
            "start": 20,
            "end": 20 + pulse["duration"],
            "source": "rule",
        }
    ]
    assert after["levers"]["regen_mult"] == pulse["params"]["mult"] and isinstance(
        after["levers"]["imm_request"], int
    )
    ready = {c["action"]: c["ready"] for c in after["cooldowns"]}
    assert (
        set(ready) == {"regen_pulse", "immigration_pulse"} and ready["regen_pulse"] > 20 + pulse["duration"]
    )
    assert (
        e[50]["effects"] == after["effects"] and "imm_request" not in e[50]["levers"]
    )  # the request was consumed
    assert [e[s]["season"]["phase"] for s in (0, 10, 50)] == [0.0, 0.1, 0.5]
    assert e[0]["season"] == {"period": 100, "amp": 0.5, "phase": 0.0}


def test_a_world_without_a_director_or_seasons_has_an_empty_environment(tmp_path: Path) -> None:
    _, _, _, out = run_paced(world("2d"), tmp_path / "w", 20, [], options=RUN)
    assert set(map(json.dumps, envs(out).values())) == {
        json.dumps(
            {"pools": [], "season": None, "levers": {}, "effects": [], "cooldowns": [], "influence": None}
        )
    }


def test_the_warm_pools_are_those_of_the_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshots(monkeypatch, 10)
    cfg = world("storm")
    rd, _, _, out = run_paced(cfg, tmp_path / "w", 20, [], options=RUN, clock=FakeClock())
    p = weather.params_from_config(RunDir.open(rd.path).config())
    for step, obj in envs(out).items():
        want = [
            {"x": round((x0 + vx * step) % p.L, 2), "y": round((y0 + vy * step) % p.L, 2), "amp": amp,
             "sx": sx, "sy": sy, "vx": vx, "vy": vy}
            for x0, y0, sx, sy, amp, vx, vy in p.pools
        ]  # fmt: skip
        assert obj["pools"] == want and want
    assert envs(out)[0]["pools"] != envs(out)[20]["pools"]  # the pools drift
