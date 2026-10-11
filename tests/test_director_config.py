"""Director build step 2, config: the policy keys, the hash rule, validation and the resume key list."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from alife import cli, runner
from alife.config import (
    DIRECTOR_MODES,
    DIRECTOR_POLICY_KEYS,
    PERSONAS,
    POST_V01_KEYS,
    UNHASHED_KEYS,
    Config,
    ConfigError,
    apply_overrides,
    config_from_dict,
    config_hash,
    config_to_dict,
    load_config,
    validate,
)
from alife.runio import RunDir
from alife.state import init_state

# Hashes computed with the step-1 ``config.py`` (before the policy keys existed): they must not move.
TINY_HASH_V01 = "d25e0eaee7349ae366e3e4e60ebab107cf7a89fcf49be88a12d81099ff2561d5"
TINY_ENABLED = "9e3654a3055412dbf31b5604cadf3d12f298c58ab77324f11a4dea6bd7aeb6d6"
TINY_F32_ENABLED = "659bb10989d7a3f6e0e74410b123e79df6f29a1408031d637fb9f55799a89d04"
TINY_SEASONS_ENABLED = "2354fda8c232234f9f7b634d447e987faf4f62c6d89b62a15bee72a32b020d6d"
K64_HASH = "69b4e21ad6b6016b17c11d6a1d9b3371c03b4b9f590fdb4a04752515c13a01d8"
K64_ENABLED = "f95265a4b9a1bc178097e1d96c90fd199ca2df83cdc60a7ffb777fcdb356ff01"
ENABLED = "director.enabled=true"


def test_existing_hashes_do_not_move() -> None:
    tiny = load_config(preset="tiny")
    assert config_hash(tiny) == TINY_HASH_V01
    assert config_hash(apply_overrides(tiny, [ENABLED])) == TINY_ENABLED  # a step-1 run with the director on
    assert (
        config_hash(load_config(preset="tiny", overrides=["brain.weight_dtype=float32", ENABLED]))
        == TINY_F32_ENABLED
    )
    seasons = ["fields.season_period=40", "fields.season_amplitude=0.3"]
    assert config_hash(load_config(preset="tiny", overrides=[ENABLED, *seasons])) == TINY_SEASONS_ENABLED
    assert config_hash(load_config(preset="64k")) == K64_HASH  # runs/64k_s0
    assert config_hash(load_config(preset="64k", overrides=[ENABLED])) == K64_ENABLED


@pytest.mark.parametrize(
    "override",
    [
        "director.persona=trickster",
        "director.persona=architect",
        "director.goal=bodies of at least 100",
        "director.rule_every=2000",
        "director.budget_per_100k=3",
        "run.director=live",
    ],
)
def test_policy_keys_and_the_mode_never_enter_the_hash(override: str) -> None:
    for base in ([], [ENABLED]):
        cfg = load_config(preset="tiny", overrides=base)
        extra = [ENABLED] if override.startswith("run.") else []  # the mode needs the director on
        assert config_hash(apply_overrides(cfg, [*extra, override])) == config_hash(
            apply_overrides(cfg, extra)
        )


def test_only_enabled_is_hashed() -> None:
    assert set(UNHASHED_KEYS) == {("director", k) for k in DIRECTOR_POLICY_KEYS}
    assert set(UNHASHED_KEYS) <= set(POST_V01_KEYS) and ("director", "enabled") not in UNHASHED_KEYS
    tiny = load_config(preset="tiny")
    assert config_hash(apply_overrides(tiny, [ENABLED])) != config_hash(tiny)
    # a policy-only change never reaches the hash, even from non-default values
    odd = apply_overrides(tiny, ["director.persona=trickster", "director.rule_every=500", "director.goal=x"])
    assert config_hash(odd) == TINY_HASH_V01


def test_old_config_files_load_with_the_new_defaults() -> None:
    cfg = load_config(preset="tiny")
    old = config_to_dict(cfg)
    for sec, key in POST_V01_KEYS:
        del old[sec][key]
    del old["run"]["director"]  # a config.json from before the mode existed
    back = config_from_dict(old)
    validate(back)
    assert back == cfg and config_hash(back) == TINY_HASH_V01
    assert (back.director.persona, back.director.goal, back.director.rule_every) == ("gardener", "", 1000)
    assert back.director.budget_per_100k == 0 and back.run.director == "off"


def test_validation_of_the_director_keys() -> None:
    tiny = apply_overrides(load_config(preset="tiny"), [ENABLED])
    assert PERSONAS == ("gardener", "trickster", "architect") and DIRECTOR_MODES == ("off", "live", "replay")
    for bad, msg in (
        ("director.persona=wizard", "persona"),
        ("director.rule_every=0", "rule_every"),
        ("director.budget_per_100k=-1", "budget_per_100k"),
        ("run.director=maybe", "run.director"),
    ):
        with pytest.raises(ConfigError, match=msg):
            apply_overrides(tiny, [bad])


def test_rule_every_must_be_a_multiple_of_both_metrics_cadences_when_the_director_runs() -> None:
    tiny = apply_overrides(load_config(preset="tiny"), [ENABLED])  # metrics_every 10, heavy 10
    live = apply_overrides(tiny, ["run.director=live"])
    assert live.director.rule_every == 1000
    for ov in ("director.rule_every=1005", "director.rule_every=5"):
        with pytest.raises(ConfigError, match=r"multiple of run\.metrics_every"):
            apply_overrides(live, [ov])
    with pytest.raises(ConfigError, match="multiple"):  # resume overrides of the cadences are revalidated
        apply_overrides(
            live, ["run.metrics_every=30", "run.heavy_metrics_every=30", "director.rule_every=1000"]
        )
    with pytest.raises(ConfigError, match=r"multiple of run\.metrics_every"):
        apply_overrides(live, ["run.heavy_metrics_every=300", "run.checkpoint_every=300"])
    # with the mode off the cadence is free (no hook runs), and so is enabled=true alone
    assert apply_overrides(tiny, ["director.rule_every=1005"]).director.rule_every == 1005
    ok = apply_overrides(
        live, ["run.metrics_every=50", "run.heavy_metrics_every=500", "run.checkpoint_every=500"]
    )
    assert ok.run.metrics_every == 50


def test_a_live_or_replay_mode_needs_the_director_enabled() -> None:
    tiny = load_config(preset="tiny")
    for mode in ("live", "replay"):
        with pytest.raises(ConfigError, match=r"director\.enabled"):
            apply_overrides(tiny, [f"run.director={mode}"])
    assert apply_overrides(tiny, ["run.director=off"]).run.director == "off"  # off is always fine


# --- resume key list ---


@pytest.fixture(scope="module")
def run_dir(tiny_cfg: Config, tmp_path_factory: pytest.TempPathFactory) -> Path:
    cfg = apply_overrides(tiny_cfg, [ENABLED, "run.diversity_sample=64"])
    rd = RunDir.create(tmp_path_factory.mktemp("rk") / "run", cfg)
    rd.update_meta(start_step=0)
    runner.run(rd, cfg, init_state(cfg), 30, quiet=True)
    return rd.path


@pytest.mark.parametrize(
    "override",
    [
        "director.enabled=false",
        "director.nope=1",
        "run.seed=3",
        "population.n_max=2048",
        "fields.season_period=9",
    ],
)
def test_resume_rejects_everything_but_the_policy_keys_and_the_mode(run_dir: Path, override: str) -> None:
    before = {p: p.read_bytes() for p in (run_dir / "config.json", run_dir / "metrics.jsonl")}
    assert cli.main(["resume", str(run_dir), "--set", override, "--checkpoint", "0", "--steps", "0"]) == 2
    assert {p: p.read_bytes() for p in before} == before


def test_resume_accepts_the_policy_keys_and_the_mode(run_dir: Path, tmp_path: Path) -> None:
    copy = tmp_path / "run"
    shutil.copytree(run_dir, copy)
    argv = ["resume", str(copy), "--steps", "0", "--quiet", "--set", "director.persona=trickster"]
    argv += [
        "--set",
        "director.goal=more bodies",
        "--set",
        "director.rule_every=500",
        "--set",
        "director.budget_per_100k=3",
    ]
    assert cli.main(argv) == 0
    d = json.loads((copy / "config.json").read_text())["director"]
    assert (d["persona"], d["goal"], d["rule_every"], d["budget_per_100k"]) == (
        "trickster",
        "more bodies",
        500,
        3,
    )
    assert cli.main(["resume", str(copy), "--steps", "0", "--quiet", "--director", "live"]) == 0
    assert json.loads((copy / "config.json").read_text())["run"]["director"] == "live"
    cfg = RunDir.open(copy).config()
    assert cfg.run.director == "live" and cfg.director.enabled  # resume inherits the mode from now on


def test_resume_of_a_run_without_an_enabled_director_cannot_start_one(
    tiny_cfg: Config, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = apply_overrides(tiny_cfg, ["run.diversity_sample=64"])
    rd = RunDir.create(tmp_path / "plain", cfg)
    runner.run(rd, cfg, init_state(cfg), 20, quiet=True)
    for mode in ("live", "replay"):
        assert cli.main(["resume", str(rd.path), "--director", mode, "--steps", "5"]) == 2
        assert "--init-from" in capsys.readouterr().err  # the message points to a fork
    assert cli.main(["resume", str(rd.path), "--director", "off", "--steps", "0", "--quiet"]) == 0
    assert [s for s, _ in rd.checkpoints()] == [0, 20]
