"""The yoked ``random`` persona, arm R of DIRECTOR.md §9 (DESIGN.md §26.11): validation, the yoke log, the
sampler, a live run yoked to a scripted "arm B", and its replay.

Arm B is a live Gardener run whose discretionary moves are scripted (a monkeypatched ``rules.decide``): two at
step 200, one at 500, plus a safety move (a runaway damper) at 300 whose revert falls at 500. Arm R follows
B's log. The menu is shrunk for this module (timed bounds 100-2000 steps, every cost 1) by a module-scoped
patch, as in ``test_director_replay.py``, so that a 1,200-step run reverts and the budget allows the moves.
The safety rules are silenced too (on a tiny world the real extinction rescue fires at step 100 and spends
budget); the tests that are about them restore them. The patches are undone when the module ends.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import shutil
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alife import cli
from alife.config import PERSONAS, Config, ConfigError, apply_overrides, config_hash, load_config
from alife.director import Director, actions, log, pacing, rules, yoked
from alife.director.actions import MENU
from alife.director.rules import Proposal
from alife.runio import DIRECTOR_LOG_FILE, JOURNAL_FILE, RunDir, load_checkpoint, read_jsonl
from alife.state import init_state, state_digest
from helpers import scripted_decide

REAL_SAFETY = rules.decide_safety

PERF = {"wall_s", "steps_per_s", "peak_mem_gb"}
EVERY, END = 100, 1200
COMMON = [
    "--preset",
    "tiny",
    "--set",
    "brain.weight_dtype=float32",
    "--set",
    f"director.rule_every={EVERY}",
    "--set",
    "run.metrics_every=10",
    "--set",
    "run.heavy_metrics_every=100",
    "--set",
    "run.checkpoint_every=300",
    "--set",
    "run.keep_checkpoints=20",
    "--set",
    "run.diversity_sample=64",
    "--set",
    "director.budget_per_100k=4",
    "--steps",
    str(END),
    "--quiet",
]


def prop(action: str, raw: dict, duration: int | None = None, rule: str = "scripted") -> Proposal:
    return Proposal(action, raw, duration, rule, f"scripted {action}")


PLAN_B = {
    200: [prop("set_seasons", {"period": 5000, "amp": 0.5}), prop("immigration_pulse", {"count": 60})],
    300: [prop("regen_pulse", {"mult": 0.8}, 200, rules.RUNAWAY_DAMPER)],  # a safety move; reverts at 500
    500: [prop("drift_change", {"mult": 3.0})],
}
B_STEPS = {200: 2, 500: 1}


@pytest.fixture(scope="module", autouse=True)
def shrunk_menu() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("alife.director.rules.decide_safety", scripted_decide({}))
        for name, spec in MENU.items():
            changes: dict[str, Any] = {"cost": 1}
            if name in ("regen_pulse", "habitat_loss"):  # reverts within a 1200-step run
                changes["duration"] = (100, 2000)
            mp.setitem(MENU, name, dataclasses.replace(spec, **changes))
        yield


@pytest.fixture(scope="module")
def arm_b(shrunk_menu: None, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Arm B: a Gardener run with scripted moves."""
    out = tmp_path_factory.mktemp("arm_b") / "b"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("alife.director.rules.decide", scripted_decide(PLAN_B))
        assert cli.main(["run", *COMMON, "--director", "live", "--out", str(out)]) == 0
    return out


# Arm C: a Trickster run (budget 6 by default) with six scripted discretionary moves inside one 100k window,
# three at step 200 and three at 500, each of a different action.
PLAN_C = {
    200: [
        prop("set_seasons", {"period": 5000, "amp": 0.5}),
        prop("immigration_pulse", {"count": 60}),
        prop("drift_change", {"mult": 3.0}),
    ],
    500: [
        prop("regen_pulse", {"mult": 1.2}, 200),
        prop("catastrophe", {"x": 8.0, "y": 8.0, "radius": 3.0, "kill_p": 0.3}),
        prop("fertility_shift", {"patch": 0, "dx": 0.5, "dy": 0.5}),
    ],
}
C_STEPS = {200: 3, 500: 3}


@pytest.fixture(scope="module")
def arm_c(shrunk_menu: None, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("arm_c") / "c"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("alife.director.rules.decide", scripted_decide(PLAN_C))
        argv = ["run", *COMMON, "--director", "live", "--persona", "trickster"]
        assert cli.main([*argv, "--set", "director.budget_per_100k=0", "--out", str(out)]) == 0
    return out


def run_r(b: Path, out: Path, *extra: str) -> int:
    argv = [
        "run",
        *COMMON,
        "--director",
        "live",
        "--persona",
        "random",
        "--yoke-log",
        str(b / DIRECTOR_LOG_FILE),
    ]
    return cli.main([*argv, *extra, "--out", str(out)])


@pytest.fixture(scope="module")
def arm_r(arm_b: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("arm_r") / "r"
    assert run_r(arm_b, out) == 0
    return out


def entries(path: Path) -> list[dict[str, Any]]:
    return log.entries_of(read_jsonl(path / DIRECTOR_LOG_FILE))


def rows(path: Path) -> list[dict[str, Any]]:
    return [{k: v for k, v in r.items() if k not in PERF} for r in RunDir.open(path).metrics_rows()]


def digest(path: Path, step: int = END) -> str:
    rd = RunDir.open(path)
    return state_digest(load_checkpoint(rd.checkpoint_at(step), rd.config()))


def discretionary(es: list[dict[str, Any]]) -> Counter[int]:
    return Counter(
        e["step_applied"] for e in es if e["source"] != "revert" and e.get("rule") not in rules.SAFETY_RULES
    )


# --- the yoke log: which moves are yoked -------------------------------------------------------------


def entry(step: int, source: str = "rule", rule: str | None = "scripted", **kw: Any) -> dict[str, Any]:
    e = {"step_applied": step, "source": source, "action": "drift_change", "params": {"mult": 2.0}}
    return {**e, **({"rule": rule} if rule is not None else {}), **kw}


def test_yoked_steps_count_discretionary_moves_per_step() -> None:
    records = [
        {"kind": "header", "step": 0, "persona": "gardener"},
        entry(200, rule="monoculture_breaker"),
        entry(200, rule="stagnation_response"),
        entry(300, rule=rules.EXTINCTION_RESCUE),  # safety rules are never yoked
        entry(300, rule=rules.EXTINCTION_RESCUE),
        entry(400, rule=rules.RUNAWAY_DAMPER),
        entry(500, source="llm", rule=None, llm_call="llm-500"),  # the model's moves have no rule
        entry(500, source="llm", rule=None, llm_call="llm-500"),
        entry(500, source="llm", rule=None, llm_call="llm-500"),
        entry(600, source="revert", rule=None),  # reverts are never yoked
        entry(700, rule="boredom"),
        entry(700, rule="random", yoked_to=700),  # R's own moves count when R is itself the yoke
    ]
    assert yoked.yoked_steps(records) == {200: 2, 500: 3, 700: 2}
    assert yoked.yoked_steps([]) == {} and yoked.yoked_steps([records[0]]) == {}


def test_a_log_whose_entries_name_no_rule_counts_all_but_reverts() -> None:
    """Older logs: nothing says which ``rule`` entry was a safety move, so every one counts (R2)."""
    records = [
        entry(100, rule=None),
        entry(100, rule=None),
        entry(200, source="llm", rule=None),
        entry(300, source="revert", rule=None),
    ]
    assert yoked.yoked_steps(records) == {100: 2, 200: 1}


def test_yoked_steps_skip_what_is_not_an_entry() -> None:
    junk = ["x", 3, None, {"source": "rule"}, {"source": "rule", "step_applied": True}, entry(100)]
    assert yoked.yoked_steps(junk) == {100: 1}


def test_b_log_yields_the_scripted_steps(arm_b: Path) -> None:
    recs = read_jsonl(arm_b / DIRECTOR_LOG_FILE)
    assert yoked.yoked_steps(recs) == B_STEPS
    es = log.entries_of(recs)
    assert [
        (e["step_applied"], e["source"], e.get("rule")) for e in es if e["step_applied"] in (300, 500)
    ] == [
        (300, "rule", "runaway_damper"),
        (500, "revert", None),
        (500, "rule", "scripted"),
    ]


# --- config and the CLI ------------------------------------------------------------------------------------


def test_the_random_persona_and_its_key_in_the_config() -> None:
    tiny = load_config(preset="tiny")
    assert "random" in PERSONAS and tiny.director.yoke_log == ""
    rnd = apply_overrides(tiny, ["director.persona=random", "director.yoke_log=runs/b/director.jsonl"])
    assert rnd.director.yoke_log == "runs/b/director.jsonl"
    assert config_hash(rnd) == config_hash(tiny)  # a policy key: never hashed


@pytest.mark.parametrize(
    ("overrides", "msg"),
    [
        (["director.persona=random"], "yoke_log"),
        (["director.persona=random", "director.yoke_log= "], "yoke_log"),
        (["director.persona=random", "director.yoke_log=b.jsonl", "director.llm_every=100"], "llm_every"),
        (["director.yoke_log=b.jsonl"], "only for the random persona"),
        (["director.persona=trickster", "director.yoke_log=b.jsonl"], "only for the random persona"),
    ],
)
def test_validation_of_the_yoke_keys(overrides: list[str], msg: str) -> None:
    base = apply_overrides(load_config(preset="tiny"), ["director.enabled=true", "director.rule_every=100"])
    with pytest.raises(ConfigError, match=msg):
        apply_overrides(base, overrides)


def test_cli_errors_leave_no_run_directory(
    arm_b: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    good = str(arm_b / DIRECTOR_LOG_FILE)
    argv = ["run", *COMMON, "--director", "live", "--persona", "random"]

    def exit_code(*extra: str, name: str = "x") -> int:
        code = cli.main([*argv, *extra, "--out", str(tmp_path / name)])
        assert not (tmp_path / name).exists(), extra  # nothing was written
        return code

    assert exit_code() == 2  # no --yoke-log
    assert exit_code("--yoke-log", str(tmp_path / "missing.jsonl")) == 2
    assert "no director log" in capsys.readouterr().err
    assert exit_code("--yoke-log", str(tmp_path)) == 2  # a directory
    assert exit_code("--yoke-log", good, "--llm-every", "100") == 2  # no model for R
    assert "llm_every" in capsys.readouterr().err
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"kind": "header"}\nnot json\n{"kind": "header"}\n')
    assert exit_code("--yoke-log", str(bad)) == 2  # broken in the middle
    bad.write_text(json.dumps({**entry(200), "action": "no_such_action"}) + "\n")
    assert exit_code("--yoke-log", str(bad)) == 2  # fails the replay pre-validation
    off = [json.loads(line) for line in (arm_b / DIRECTOR_LOG_FILE).read_text().splitlines()]
    off[1]["step_applied"] = 250
    bad.write_text("".join(json.dumps(r) + "\n" for r in off))
    assert exit_code("--yoke-log", str(bad)) == 2
    assert "multiples of director.rule_every" in capsys.readouterr().err
    # --yoke-log with another persona, and with none
    plain = ["run", *COMMON, "--director", "live", "--yoke-log", good, "--out", str(tmp_path / "y")]
    assert cli.main(plain) == 2 and not (tmp_path / "y").exists()
    assert "only for the random persona" in capsys.readouterr().err


def test_the_yoke_log_is_checked_before_a_fork_or_resume_touches_anything(
    arm_r: Path, tmp_path: Path
) -> None:
    gone = tmp_path / "gone.jsonl"
    copy = tmp_path / "copy"
    shutil.copytree(arm_r, copy)
    before = (copy / DIRECTOR_LOG_FILE).read_bytes()
    assert cli.main(["resume", str(copy), "--until", str(END), "--yoke-log", str(gone), "--quiet"]) == 2
    assert (copy / DIRECTOR_LOG_FILE).read_bytes() == before  # not truncated, not rewritten
    out = tmp_path / "fork"
    fork = ["run", "--init-from", f"{arm_r}:0", "--director", "live", "--yoke-log", str(gone), "--quiet"]
    assert cli.main([*fork, "--steps", "100", "--out", str(out)]) == 2 and not out.exists()


# --- the sampler -----------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg(tiny_cfg: Config) -> Config:
    return apply_overrides(
        tiny_cfg,
        ["director.enabled=true", f"director.rule_every={EVERY}", "run.heavy_metrics_every=100"],
    )


def draws(cfg: Config, action: str, n: int, patches: yoked.Patches | None = None) -> list[tuple]:
    return [
        yoked.sample_params(action, yoked.rng_for(cfg.run.seed, 100 * i, 0), cfg, patches) for i in range(n)
    ]


def test_every_draw_is_inside_the_live_menu_bounds(cfg: Config) -> None:
    size, np_ = cfg.world.size, cfg.fields.n_patches
    rlo, rhi = actions.radius_bounds(cfg)
    for action, spec in MENU.items():
        for raw, duration in draws(cfg, action, 300):
            for key, (lo, hi) in spec.bounds.items():
                if key in raw:
                    assert lo <= raw[key] <= hi, (action, key, raw[key])
            if spec.duration is None:
                assert duration is None
            else:
                assert isinstance(duration, int) and spec.duration[0] <= duration <= spec.duration[1]
            if "patch" in raw:
                assert isinstance(raw["patch"], int) and 0 <= raw["patch"] < np_
            if action == "catastrophe":
                assert 0 <= raw["x"] < size and 0 <= raw["y"] < size and rlo <= raw["radius"] <= rhi
            if "dx" in raw:
                assert math.hypot(raw["dx"], raw["dy"]) <= actions.MOVE_MAX_FRAC * size * (1 + 1e-9)
            if action == "set_seasons":
                assert isinstance(raw["period"], int)
            if action == "immigration_pulse":
                assert isinstance(raw["count"], int)


def test_the_menu_is_read_live_not_at_import(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        MENU, "drift_change", dataclasses.replace(MENU["drift_change"], bounds={"mult": (1.0, 2.0)})
    )
    assert all(1.0 <= raw["mult"] <= 2.0 for raw, _ in draws(cfg, "drift_change", 100))
    monkeypatch.setitem(MENU, "regen_pulse", dataclasses.replace(MENU["regen_pulse"], duration=(700, 900)))
    assert all(700 <= d <= 900 for _, d in draws(cfg, "regen_pulse", 100))


def test_fertility_shift_takes_both_forms_and_a_move_when_no_patch_is_free(cfg: Config) -> None:
    forms = Counter("gain" in raw for raw, _ in draws(cfg, "fertility_shift", 400))
    assert 140 < forms[True] < 260 and 140 < forms[False] < 260  # one half each
    only_moves = yoked.Patches(alive=(0, 1, 2, 3), free=())
    assert all("dx" in raw for raw, _ in draws(cfg, "fertility_shift", 100, only_moves))
    pools = yoked.Patches(alive=(0, 2, 3), free=(2,))
    for raw, _ in draws(cfg, "fertility_shift", 200, pools):
        assert raw["patch"] in ((2,) if "gain" in raw else (0, 2, 3))
    assert {raw["patch"] for raw, _ in draws(cfg, "habitat_loss", 100, pools)} == {2}
    with pytest.raises(ValueError, match="no patch"):
        yoked.sample_params("habitat_loss", yoked.rng_for(0, 0, 0), cfg, yoked.Patches((0,), ()))


def test_wide_ranges_are_log_uniform_and_the_rest_uniform(cfg: Config) -> None:
    """The median of a log-uniform draw is the geometric mean of the bounds, far below the arithmetic one."""
    n = 1500

    def median(action: str, key: str | None = None) -> float:
        d = draws(cfg, action, n)
        return float(np.median([raw[key] if key else dur for raw, dur in d]))

    assert 8_000 < median("set_seasons", "period") < 24_000  # sqrt(2k * 100k) = 14.1k; uniform: 51k
    assert 70 < median("immigration_pulse", "count") < 200  # sqrt(16 * 1024) = 128; uniform: 520
    assert 1.5 < median("drift_change", "mult") < 3.5  # sqrt(0.5 * 10) = 2.2; uniform: 5.25
    lo, hi = MENU["habitat_loss"].duration
    assert 0.6 * math.sqrt(lo * hi) < median("habitat_loss") < 1.6 * math.sqrt(lo * hi)
    assert 0.38 < median("set_seasons", "amp") < 0.52  # uniform on [0, 0.9]
    assert 1.0 < median("regen_pulse", "mult") < 1.2  # uniform on [0.4, 1.8]
    assert 0.5 < median("catastrophe", "kill_p") < 0.6  # uniform on [0.2, 0.9]


def test_draws_are_deterministic_per_seed_step_and_index(cfg: Config) -> None:
    def one(seed: int, t: int, k: int) -> list[tuple]:
        rng = yoked.rng_for(seed, t, k)
        return [yoked.sample_params(a, rng, cfg) for a in MENU]

    assert one(3, 400, 0) == one(3, 400, 0)
    assert len({repr(one(3, 400, 0)), repr(one(4, 400, 0)), repr(one(3, 500, 0)), repr(one(3, 400, 1))}) == 4
    assert max(rules._TAGS.values()) * 2 + 1 < yoked.RANDOM_TAG  # distinct from the Trickster's streams


def test_sampled_params_pass_the_clamp_and_the_replay_validation(cfg: Config) -> None:
    view = actions.env_view(init_state(cfg))
    for action in MENU:
        for raw, duration in draws(cfg, action, 60):
            c = actions.clamp(action, raw, duration, view, cfg)
            e = {"step_applied": EVERY, "source": "rule", "action": action, **c._asdict()}
            actions.validate_entry(json.loads(json.dumps(e)), cfg, EVERY)
            if c.duration is not None:
                assert c.duration % EVERY == 0


def env(**kw: Any) -> actions.EnvView:
    base = {
        "regen_mult": 1.0,
        "season_amp": 0.0,
        "season_period": 0,
        "drift_mult": 1.0,
        "imm_request": 0,
        "patch_gain": np.ones(4, np.float32),
        "patch_pos": np.zeros((4, 2), np.float32),
    }
    return actions.EnvView(**{**base, **kw})


def ctx(t: int = 1000, entries: tuple = (), budget: int = 4, **kw: Any) -> rules.Context:
    return rules.Context(t, 0, "gardener", budget, 128.0, None, pacing.fold(entries, t), env(**kw))  # type: ignore[arg-type]


def fwd(step: int, action: str, params: dict, duration: int | None = None, prev: dict | None = None) -> dict:
    return {
        "step_applied": step,
        "source": "rule",
        "action": action,
        "params": params,
        "prev": prev or {},
        "duration": duration,
    }


def test_every_menu_action_is_reachable_and_eligibility_follows_the_rules(cfg: Config) -> None:
    P = rules.RULE_PARAMS
    options, pools, skipped = yoked.candidates(ctx(), P, cfg)
    assert options == list(MENU) and skipped == [] and pools == yoked.Patches((0, 1, 2, 3), (0, 1, 2, 3))
    hits = Counter()
    for i in range(700):  # the first draw of a boundary picks uniformly
        rng = yoked.rng_for(1, 100 * i, 0)
        hits[options[int(rng.integers(len(options)))]] += 1
    assert set(hits) == set(MENU) and min(hits.values()) > 60  # 100 each in expectation
    # budget: the cost units used in the rolling window leave no room
    spent = tuple(fwd(900, "drift_change", {"mult": 2.0}) for _ in range(4))
    assert yoked.candidates(ctx(entries=spent), P, cfg)[0] == []
    # cooldown (the Gardener's 1.5x of the menu's) from the end of an action's latest effect
    only = yoked.candidates(ctx(entries=(fwd(900, "drift_change", {"mult": 2.0}),), budget=9), P, cfg)
    assert "drift_change" not in only[0] and "drift_change: drift_change is cooling down" in only[2]
    # a timed effect: its action cools down, and a locked patch is not offered to a gain change or a loss
    held = (
        fwd(900, "regen_pulse", {"mult": 0.7}, 500, {"mult": 1.0}),
        fwd(900, "habitat_loss", {"patch": 1, "gain": 0.0}, 500, {"patch": 1, "gain": 1.0}),
    )
    options, pools, skipped = yoked.candidates(ctx(entries=held, budget=9), P, cfg)
    assert {"regen_pulse", "habitat_loss"}.isdisjoint(options) and pools == yoked.Patches(
        (0, 1, 2, 3), (0, 2, 3)
    )
    assert yoked.sample_params("habitat_loss", yoked.rng_for(0, 0, 0), cfg, pools)[0]["patch"] != 1
    # the catastrophe cap and the last-patch rule
    big = (fwd(100, "catastrophe", {"center": [1, 1], "radius": 30.0, "kill_p": 0.5}),)
    assert "catastrophe" not in yoked.candidates(ctx(entries=big, budget=9), P, cfg)[0]
    lone = ctx(patch_gain=np.asarray([0, 0, 1, 0], np.float32))
    options, pools, skipped = yoked.candidates(lone, P, cfg)
    assert "habitat_loss" not in options and any("no patch it may take" in s for s in skipped)
    assert pools.alive == (2,) and "fertility_shift" in options  # a move is always possible
    # a full immigration queue
    assert "immigration_pulse" not in yoked.candidates(ctx(imm_request=1020), P, cfg)[0]
    assert "mutation_floor_nudge" not in MENU  # not implemented, so never drawn


# --- a live run yoked to B ---------------------------------------------------------------------------


def test_r_makes_its_moves_at_b_s_discretionary_steps(arm_b: Path, arm_r: Path) -> None:
    es = entries(arm_r)
    assert discretionary(es) == Counter(B_STEPS) == discretionary(entries(arm_b))
    moves = [e for e in es if e["source"] != "revert"]
    assert [e["step_applied"] for e in moves] == [200, 200, 500]
    for e in moves:
        assert (e["source"], e["rule"], e["persona"], e["yoked_to"]) == (
            "rule",
            "random",
            "random",
            e["step_applied"],
        )
        assert e["reason"].startswith("yoked draw ")
    assert {e["action"] for e in moves} <= set(MENU) and moves[0]["action"] != "mutation_floor_nudge"
    # B's safety entry and its revert are not copied; R's reverts undo R's own timed effects
    assert all(e["rule"] != rules.RUNAWAY_DAMPER for e in moves)
    for rev in (e for e in es if e["source"] == "revert"):
        origin = next(
            e for e in moves if e["step_applied"] == rev["reverts"] and e["action"] == rev["action"]
        )
        assert rev["step_applied"] == origin["step_applied"] + origin["duration"]
    assert RunDir.open(arm_r).config().director.persona == "random"
    assert (
        cli.main(["inspect", str(arm_r), "--check"]) == 0
        and cli.main(["inspect", str(arm_b), "--check"]) == 0
    )


def test_the_header_records_the_yoke_log_and_its_hash(arm_b: Path, arm_r: Path) -> None:
    header = read_jsonl(arm_r / DIRECTOR_LOG_FILE)[0]
    src = arm_b / DIRECTOR_LOG_FILE
    assert header["kind"] == "header" and header["persona"] == "random"
    assert (
        header["yoke_log"] == str(src)
        and header["yoke_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    )
    assert header["budget_per_100k"] == 4 and "yoke_log" not in read_jsonl(src)[0]  # B's header is as before
    journal = read_jsonl(arm_r / JOURNAL_FILE)
    assert all(r["persona"] == "random" for r in journal) and len(journal) >= 3


def test_r_uses_the_gardener_s_budget_and_the_menu_costs(arm_r: Path) -> None:
    rd = RunDir.open(arm_r)
    d = Director(rd, rd.config(), "replay", 0)  # a replay builds the same persona and reads only its own log
    assert d.persona.name == "gardener" and d.name == "random" and d.budget == 4


def test_the_same_run_twice_is_identical(arm_b: Path, arm_r: Path, tmp_path: Path) -> None:
    out = tmp_path / "again"
    assert run_r(arm_b, out) == 0
    assert (out / DIRECTOR_LOG_FILE).read_bytes() == (arm_r / DIRECTOR_LOG_FILE).read_bytes()
    assert digest(out) == digest(arm_r)


def test_a_different_seed_draws_differently_at_the_same_steps(
    arm_b: Path, arm_r: Path, tmp_path: Path
) -> None:
    out = tmp_path / "seed1"
    assert run_r(arm_b, out, "--seed", "1") == 0
    es = entries(out)
    assert discretionary(es) == Counter(B_STEPS)
    key = lambda p: [(e["action"], e["params"]) for e in entries(p) if e["source"] == "rule"]  # noqa: E731
    assert key(out) != key(arm_r)


def test_a_replay_is_bitwise_identical_and_never_reads_the_yoke_log(
    arm_b: Path, arm_r: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    yoke = arm_b / DIRECTOR_LOG_FILE
    hidden = yoke.with_name("hidden.jsonl")
    yoke.rename(hidden)  # any access to it, by any route, would now fail
    try:

        def refuse(*a: Any, **k: Any) -> Any:
            raise AssertionError("a replay read the yoke log")

        monkeypatch.setattr(yoked, "load", refuse)
        real = log.read_log

        def read_log(path: Path) -> Any:
            assert Path(path) != yoke, "a replay read the yoke log"
            return real(path)

        monkeypatch.setattr(log, "read_log", read_log)
        out = tmp_path / "replay"
        argv = ["run", "--init-from", f"{arm_r}:0", "--director", "replay", "--replay-log"]
        argv += [str(arm_r / DIRECTOR_LOG_FILE), "--steps", str(END), "--quiet", "--out", str(out)]
        assert cli.main(argv) == 0
        assert digest(out) == digest(arm_r) and rows(out) == rows(arm_r)
        for step in (300, 600, 900):
            assert digest(out, step) == digest(arm_r, step)
        assert (out / DIRECTOR_LOG_FILE).read_bytes() == (arm_r / DIRECTOR_LOG_FILE).read_bytes()
        assert not (out / JOURNAL_FILE).exists()
        assert cli.main(["inspect", str(out), "--check"]) == 0
        # in place from a checkpoint, too
        copy = tmp_path / "copy"
        shutil.copytree(arm_r, copy)
        code = cli.main(
            [
                "resume",
                str(copy),
                "--director",
                "replay",
                "--checkpoint",
                "600",
                "--until",
                str(END),
                "--quiet",
            ]
        )
        assert code == 0 and digest(copy) == digest(arm_r)
    finally:
        hidden.rename(yoke)


def test_r_differs_from_the_run_without_a_director_and_from_b(arm_b: Path, arm_r: Path) -> None:
    assert digest(arm_r) != digest(arm_b)


def test_a_live_resume_rereads_the_yoke_log_and_regenerates_the_log(arm_r: Path, tmp_path: Path) -> None:
    copy = tmp_path / "copy"
    shutil.copytree(arm_r, copy)
    argv = ["resume", str(copy), "--checkpoint", "300", "--until", str(END), "--director", "live", "--quiet"]
    assert cli.main(argv) == 0
    assert (copy / DIRECTOR_LOG_FILE).read_bytes() == (arm_r / DIRECTOR_LOG_FILE).read_bytes()
    assert digest(copy) == digest(arm_r) and rows(copy) == rows(arm_r)


def test_a_changed_yoke_log_warns_and_the_run_goes_on(
    arm_b: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    yoke = tmp_path / "yoke.jsonl"
    shutil.copyfile(arm_b / DIRECTOR_LOG_FILE, yoke)
    out = tmp_path / "r"
    argv = ["run", *COMMON, "--director", "live", "--persona", "random", "--yoke-log", str(yoke)]
    assert cli.main([*argv, "--steps", "600", "--out", str(out)]) == 0
    assert "has changed" not in capsys.readouterr().err
    with yoke.open("a") as f:  # B went on: one more move at step 700
        f.write(json.dumps({**entry(700), "prev": {"mult": 1.0}, "duration": None}) + "\n")
    assert cli.main(["resume", str(out), "--until", "800", "--quiet"]) == 0
    assert "has changed since this run started following it" in capsys.readouterr().err
    headers = [r for r in read_jsonl(out / DIRECTOR_LOG_FILE) if r.get("kind") == "header"]
    assert len(headers) == 2 and headers[0]["yoke_sha256"] != headers[1]["yoke_sha256"]
    assert headers[1]["yoke_sha256"] == hashlib.sha256(yoke.read_bytes()).hexdigest()
    assert [e["step_applied"] for e in entries(out) if e["source"] == "rule" and e["step_applied"] > 600] == [
        700
    ]
    assert cli.main(["inspect", str(out), "--check"]) == 0


# --- when R cannot make a move, and the safety rules --------------------------------------------------------


def test_nothing_eligible_is_a_journal_note_and_no_crash(arm_b: Path, tmp_path: Path) -> None:
    out = tmp_path / "poor"
    code = run_r(arm_b, out, "--set", "director.budget_per_100k=1")  # room for one move at 200, none at 500
    assert code == 0
    es = [e for e in entries(out) if e["source"] != "revert"]
    assert [e["step_applied"] for e in es] == [200]
    notes = [r["text"] for r in read_jsonl(out / JOURNAL_FILE) if "nothing eligible" in r["text"]]
    assert len(notes) == 2 and notes[0].startswith("yoked step 200: nothing eligible (1 of 2 moves not made)")
    assert notes[1].startswith("yoked step 500: nothing eligible (1 of 1 moves not made)")
    assert "budget 1/1 used" in notes[0]
    assert cli.main(["inspect", str(out), "--check"]) == 0


def test_the_safety_rules_and_reverts_run_for_r_and_no_discretionary_rule_does(
    arm_b: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    damper = prop("regen_pulse", {"mult": 0.8}, 200, rules.RUNAWAY_DAMPER)
    safety_calls: list[int] = []

    def decide_safety(c: Any, p: Any = None) -> Any:
        safety_calls.append(c.t)
        return rules.Decision((damper,) if c.t == 100 else (), ())

    def never(*a: Any, **k: Any) -> Any:
        raise AssertionError("a discretionary rule ran for the random persona")

    monkeypatch.setattr(rules, "decide_safety", decide_safety)
    monkeypatch.setattr(rules, "decide", never)
    monkeypatch.setattr(rules, "decide_discretionary", never)
    out = tmp_path / "safe"
    assert run_r(arm_b, out, "--steps", "600") == 0
    assert safety_calls == [100, 200, 300, 400, 500, 600]
    es = entries(out)
    first = es[0]
    assert (first["step_applied"], first["action"], first["rule"], first["persona"]) == (
        100,
        "regen_pulse",
        "runaway_damper",
        "random",
    )
    assert "yoked_to" not in first  # a safety move is not a yoked one
    rev = next(e for e in es if e["source"] == "revert" and e["reverts"] == 100)
    assert rev["step_applied"] == 300
    moves = [e for e in es if e.get("rule") == "random"]
    assert [e["step_applied"] for e in moves] == [200, 200, 500]
    # R's own regen pulse would have been blocked at 200 while the damper's holds the lever
    assert all(e["action"] != "regen_pulse" for e in moves if e["step_applied"] == 200)


def test_the_real_safety_rules_fire_for_r(
    arm_b: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On the tiny world the extinction rescue triggers at step 100: R takes it, as any persona would, under
    the Gardener's budget; it is not a yoked move and does not count as one."""
    monkeypatch.setattr(rules, "decide_safety", REAL_SAFETY)
    out = tmp_path / "real"
    assert run_r(arm_b, out, "--steps", "200") == 0
    es = entries(out)
    rescue = [e for e in es if e.get("rule") == rules.EXTINCTION_RESCUE]
    assert [(e["step_applied"], e["persona"], e["source"]) for e in rescue] == [(100, "random", "rule")] * 2
    assert {e["action"] for e in rescue} == {"regen_pulse", "immigration_pulse"}
    assert all("yoked_to" not in e for e in rescue)
    assert [(e["step_applied"], e["rule"]) for e in es if e.get("yoked_to")] == [(200, "random")] * 2
    assert cli.main(["inspect", str(out), "--check"]) == 0


# --- R inherits the parameters of the persona that made the yoke log (R3) -----------------------------------


def header_of(path: Path) -> dict[str, Any]:
    return read_jsonl(path / DIRECTOR_LOG_FILE)[0]


@pytest.fixture(scope="module")
def arm_rc(arm_c: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Arm R yoked to the Trickster run: no explicit budget, so the one arm C had."""
    out = tmp_path_factory.mktemp("arm_rc") / "r"
    assert run_r(arm_c, out, "--set", "director.budget_per_100k=0") == 0
    return out


def test_the_scripted_trickster_log(arm_c: Path) -> None:
    recs = read_jsonl(arm_c / DIRECTOR_LOG_FILE)
    assert (recs[0]["persona"], recs[0]["budget_per_100k"]) == ("trickster", 6)
    assert yoked.yoked_steps(recs) == C_STEPS


def test_r_yoked_to_a_gardener_log_inherits_the_gardener(arm_b: Path, arm_r: Path) -> None:
    header = header_of(arm_r)
    assert header["params_persona"] == "gardener" and header["budget_per_100k"] == 4
    assert discretionary(entries(arm_r)) == Counter(B_STEPS)
    rd = RunDir.open(arm_r)
    assert yoked.load(rd.config()).persona == "gardener"  # nothing but the header's persona is read


def test_r_yoked_to_a_trickster_log_uses_the_trickster_s_parameters(arm_c: Path, arm_rc: Path) -> None:
    header = header_of(arm_rc)
    assert (header["persona"], header["params_persona"], header["budget_per_100k"]) == (
        "random",
        "trickster",
        6,
    )
    assert header["yoke_sha256"] == hashlib.sha256((arm_c / DIRECTOR_LOG_FILE).read_bytes()).hexdigest()
    es = entries(arm_rc)
    moves = [e for e in es if e["source"] != "revert" and e.get("rule") == "random"]
    # six moves in one 100k window: more than the Gardener's maximum budget of 4 allows
    assert [e["step_applied"] for e in moves] == [200] * 3 + [500] * 3
    assert len(moves) == 6 > rules.GARDENER.budget_max
    assert all(e["persona"] == "random" and e["yoked_to"] == e["step_applied"] for e in moves)
    assert len({e["action"] for e in moves}) == 6  # a cooldown (half the menu's under the Trickster) holds
    assert cli.main(["inspect", str(arm_rc), "--check"]) == 0


def test_a_trickster_yoke_with_an_explicit_gardener_sized_budget_falls_short(
    arm_c: Path, tmp_path: Path
) -> None:
    out = tmp_path / "four"
    assert run_r(arm_c, out) == 0  # COMMON sets director.budget_per_100k=4
    assert header_of(out)["budget_per_100k"] == 4 and header_of(out)["params_persona"] == "trickster"
    moves = [e for e in entries(out) if e["source"] != "revert"]
    assert [e["step_applied"] for e in moves] == [200] * 3 + [500]  # 4 of the 6


@pytest.mark.parametrize(
    ("log_of", "configured", "expected"),
    [("arm_c", "20", 12), ("arm_c", "5", 5), ("arm_c", "0", 6), ("arm_b", "20", 4), ("arm_b", "3", 3)],
)
def test_an_explicit_budget_is_capped_by_the_inherited_persona_s_maximum(
    request: pytest.FixtureRequest, tmp_path: Path, log_of: str, configured: str, expected: int
) -> None:
    src: Path = request.getfixturevalue(log_of)
    out = tmp_path / "cap"
    assert run_r(src, out, "--set", f"director.budget_per_100k={configured}", "--steps", "100") == 0
    assert header_of(out)["budget_per_100k"] == expected


def test_a_header_without_a_budget_gives_the_persona_s_default(arm_c: Path, tmp_path: Path) -> None:
    yoke = tmp_path / "yoke.jsonl"
    recs = [json.loads(line) for line in (arm_c / DIRECTOR_LOG_FILE).read_text().splitlines()]
    del recs[0]["budget_per_100k"]
    yoke.write_text("".join(json.dumps(r) + "\n" for r in recs))
    out = tmp_path / "r"
    argv = ["run", *COMMON, "--director", "live", "--persona", "random", "--yoke-log", str(yoke)]
    assert cli.main([*argv, "--set", "director.budget_per_100k=0", "--steps", "100", "--out", str(out)]) == 0
    assert (header_of(out)["params_persona"], header_of(out)["budget_per_100k"]) == ("trickster", 6)


def test_inherited_reads_the_headers() -> None:
    h = {"kind": "header", "step": 0}
    assert yoked.inherited([]) == ("gardener", None, ())  # no header: the old behaviour
    assert yoked.inherited([entry(100), h]) == ("gardener", None, ())
    assert yoked.inherited([{**h, "persona": "architect", "budget_per_100k": 5}]) == (
        "architect",
        5,
        ("architect",),
    )
    mixed = [
        {**h, "persona": "trickster", "budget_per_100k": 6},
        entry(100),
        {**h, "step": 300, "persona": "gardener", "budget_per_100k": 2},
    ]
    assert yoked.inherited(mixed) == ("trickster", 6, ("trickster", "gardener"))  # the first header rules
    for bad in (0, -1, 2.5, "6", True):
        with pytest.raises(log.LogError, match="budget_per_100k"):
            yoked.inherited([{**h, "persona": "gardener", "budget_per_100k": bad}])


@pytest.mark.parametrize("persona", ["random", "wizard"])
@pytest.mark.parametrize("where", ["first", "later"])
def test_a_yoke_log_of_a_yoked_or_unknown_persona_exits_2_before_anything_is_written(
    arm_c: Path, tmp_path: Path, capsys: pytest.CaptureFixture, persona: str, where: str
) -> None:
    recs = [json.loads(line) for line in (arm_c / DIRECTOR_LOG_FILE).read_text().splitlines()]
    if where == "first":
        recs[0]["persona"] = persona
    else:  # a resume of the yoked run appended a header with another persona
        recs.insert(3, {**recs[0], "step": 400, "persona": persona})
    yoke = tmp_path / "yoke.jsonl"
    yoke.write_text("".join(json.dumps(r) + "\n" for r in recs))
    out = tmp_path / "x"
    argv = ["run", *COMMON, "--director", "live", "--persona", "random", "--yoke-log", str(yoke)]
    assert cli.main([*argv, "--out", str(out)]) == 2 and not out.exists()
    err = capsys.readouterr().err
    assert ("yoked 'random' run" if persona == "random" else "unknown persona 'wizard'") in err


def test_a_mixed_persona_yoke_log_warns_and_uses_the_first_header(
    arm_c: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    recs = [json.loads(line) for line in (arm_c / DIRECTOR_LOG_FILE).read_text().splitlines()]
    recs.insert(3, {**recs[0], "step": 400, "persona": "gardener", "budget_per_100k": 2})
    yoke = tmp_path / "yoke.jsonl"
    yoke.write_text("".join(json.dumps(r) + "\n" for r in recs))
    out = tmp_path / "r"
    argv = ["run", *COMMON, "--director", "live", "--persona", "random", "--yoke-log", str(yoke)]
    assert cli.main([*argv, "--set", "director.budget_per_100k=0", "--steps", "100", "--out", str(out)]) == 0
    assert "several personas ['trickster', 'gardener']" in capsys.readouterr().err
    assert (header_of(out)["params_persona"], header_of(out)["budget_per_100k"]) == ("trickster", 6)


def test_a_trickster_yoked_r_replays_bitwise_without_the_yoke_log(
    arm_c: Path, arm_rc: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    yoke = arm_c / DIRECTOR_LOG_FILE
    hidden = yoke.with_name("hidden.jsonl")
    yoke.rename(hidden)
    try:
        monkeypatch.setattr(yoked, "load", lambda *a, **k: pytest.fail("a replay read the yoke log"))
        out = tmp_path / "replay"
        argv = ["run", "--init-from", f"{arm_rc}:0", "--director", "replay", "--replay-log"]
        argv += [str(arm_rc / DIRECTOR_LOG_FILE), "--steps", str(END), "--quiet", "--out", str(out)]
        assert cli.main(argv) == 0
        assert digest(out) == digest(arm_rc) and rows(out) == rows(arm_rc)
        assert (out / DIRECTOR_LOG_FILE).read_bytes() == (arm_rc / DIRECTOR_LOG_FILE).read_bytes()
        rd = RunDir.open(arm_rc)
        d = Director(rd, rd.config(), "replay", 0)  # the persona and budget come from R's own header
        assert (d.persona.name, d.name, d.budget) == ("trickster", "random", 6)
    finally:
        hidden.rename(yoke)


def test_the_same_trickster_yoked_run_twice_is_identical(arm_c: Path, arm_rc: Path, tmp_path: Path) -> None:
    out = tmp_path / "again"
    assert run_r(arm_c, out, "--set", "director.budget_per_100k=0") == 0
    assert (out / DIRECTOR_LOG_FILE).read_bytes() == (arm_rc / DIRECTOR_LOG_FILE).read_bytes()
    assert digest(out) == digest(arm_rc)
