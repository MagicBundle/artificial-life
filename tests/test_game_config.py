"""§28, §29 game config: ``game.observe`` and ``game.house_rate`` are POST_V01 observer keys, never hashed,
and resume keys. ``house_rate`` is an observer-only key with a working default (1/2048), not "off" (G2)."""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import pytest

from alife import cli, runner
from alife.config import (
    DIRECTOR_POLICY_KEYS,
    OBSERVER_KEYS,
    POST_V01_KEYS,
    UNHASHED_KEYS,
    Config,
    ConfigError,
    GameConfig,
    apply_overrides,
    config_from_dict,
    config_hash,
    config_to_dict,
    load_config,
    validate,
)
from alife.runio import RunDir
from alife.state import init_state
from game_helpers import social_every

# Hashes of the presets on the commit before G1 (905580a): they must not move, with the flag on or off. The
# 2D values are the same literals as tests/test_director_config.py; the 3D ones are pinned here first.
TINY_HASH_V01 = "d25e0eaee7349ae366e3e4e60ebab107cf7a89fcf49be88a12d81099ff2561d5"
TINY_ENABLED = "9e3654a3055412dbf31b5604cadf3d12f298c58ab77324f11a4dea6bd7aeb6d6"
K64_HASH = "69b4e21ad6b6016b17c11d6a1d9b3371c03b4b9f590fdb4a04752515c13a01d8"
K64_ENABLED = "f95265a4b9a1bc178097e1d96c90fd199ca2df83cdc60a7ffb777fcdb356ff01"
TINY3D_HASH = "90e799beeba91e8cb0e2e021ad9595340efa9975eb53f963047c63646e61797a"
TINY3D_ENABLED = "fd966f67d6e7d1685b804108395e0864f27a77ecd52d5ceaeced4855c35286c5"
K64_3D_HASH = "df54434fe947dea8f36e5e23223f9676df1ee784959806c31eafb235012ee242"  # runs/64k3d_s0
K64_3D_ENABLED = "2a4dfd22ed2e28359e6c01526b0460ede280f93ebee382872acca8c412098bba"

OBSERVE = "game.observe=true"
HOUSE_RATE = "game.house_rate=0.25"  # a non-default rate (the default is 1/2048)
ENABLED = "director.enabled=true"


def test_game_keys_are_post_v01_and_observer_keys() -> None:
    game_keys = {("game", f.name) for f in dataclasses.fields(GameConfig)}
    assert game_keys <= set(POST_V01_KEYS)  # a game key missing from the list would move every hash
    assert OBSERVER_KEYS == (
        ("game", "observe"),
        ("game", "house_rate"),
        ("game", "tracers"),
        ("game", "tracer_period"),
        ("arms", "dump_n"),  # the observer keys of the features (ext.ExtSpec.observer_keys, DESIGN §43.5)
    )
    assert game_keys == {
        k for k in OBSERVER_KEYS if k[0] == "game"
    }  # every game key is an observer key: none can move a hash
    assert not set(OBSERVER_KEYS) & set(UNHASHED_KEYS)
    assert tuple(key for _, key in UNHASHED_KEYS) == DIRECTOR_POLICY_KEYS  # unchanged: observe is not in it
    assert "observe" not in DIRECTOR_POLICY_KEYS and "house_rate" not in DIRECTOR_POLICY_KEYS
    assert GameConfig().observe is False and Config().game == GameConfig()
    assert config_to_dict(Config())["game"] == {
        "observe": False,
        "house_rate": 1 / 2048,
        "tracers": 0,
        "tracer_period": 500,
    }
    assert GameConfig().house_rate == 2.0**-11  # exact in binary: the hash cut is 2^53 (§29.4)


def test_house_rate_is_a_working_default_and_validated() -> None:
    """An observer-only key may default to a working value (CLAUDE.md): unlike every hashed post-v0.1 key,
    its default is not "off" (0, which would make the houses the clades plus one per immigrant)."""
    assert GameConfig().house_rate > 0
    base = load_config(preset="tiny")
    for ok in ("0", "0.0", "1", "1.0", "0.25", "0.000488281"):
        assert apply_overrides(base, [f"game.house_rate={ok}"]).game.house_rate == float(ok)
    for bad in ("-0.001", "1.0001", "2", "nan", "inf", "-inf"):
        with pytest.raises(ConfigError, match=r"game\.house_rate must be in \[0, 1\]"):
            apply_overrides(base, [f"game.house_rate={bad}"])
    with pytest.raises(ConfigError, match=r"game\.house_rate"):
        apply_overrides(base, ["game.house_rate=fast"])
    with pytest.raises(ConfigError, match=r"game\.house_rate"):  # a config file with a non-number
        config_from_dict({"game": {"house_rate": "0.5"}})
    assert config_from_dict({"game": {"house_rate": 1}}).game.house_rate == 1.0  # an int is a float


@pytest.mark.parametrize(
    ("preset", "plain", "enabled"),
    [
        ("tiny", TINY_HASH_V01, TINY_ENABLED),
        ("64k", K64_HASH, K64_ENABLED),
        ("tiny3d", TINY3D_HASH, TINY3D_ENABLED),
        ("64k3d", K64_3D_HASH, K64_3D_ENABLED),
    ],
)
def test_observe_never_enters_the_hash(preset: str, plain: str, enabled: str) -> None:
    for extra, want in (([], plain), ([ENABLED], enabled)):
        off = load_config(preset=preset, overrides=extra)
        on = load_config(preset=preset, overrides=[*extra, OBSERVE])
        assert off.game.observe is False and on.game.observe is True
        assert config_hash(off) == want, (preset, extra)
        assert config_hash(on) == want, (preset, extra)
        assert config_hash(apply_overrides(on, ["game.observe=false"])) == want
        # the house rate, at a non-default value, with the observers off and on, at the bounds too
        for rate in (HOUSE_RATE, "game.house_rate=0", "game.house_rate=1"):
            for flag in ([], [OBSERVE]):
                cfg = load_config(preset=preset, overrides=[*extra, *flag, rate])
                assert cfg.game.house_rate == float(rate.split("=")[1]) != GameConfig().house_rate
                assert config_hash(cfg) == want, (preset, extra, flag, rate)


def test_old_config_without_game_section_loads() -> None:
    cfg = load_config(preset="tiny")
    old = config_to_dict(cfg)
    del old["game"]  # a config.json written before the section existed
    back = config_from_dict(old)
    validate(back)
    assert back == cfg and back.game.observe is False and back.game.house_rate == GameConfig().house_rate
    assert config_hash(back) == TINY_HASH_V01
    # a G1-era config.json (a game section without house_rate) loads with the default rate
    g1 = config_to_dict(cfg)
    g1["game"] = {"observe": True}
    assert config_from_dict(g1).game == GameConfig(observe=True)
    assert config_hash(config_from_dict(g1)) == TINY_HASH_V01
    old3 = config_to_dict(load_config(preset="tiny3d"))
    del old3["game"]
    assert config_hash(config_from_dict(old3)) == TINY3D_HASH
    # an empty section is the same as an absent one
    old["game"] = {}
    assert config_from_dict(old) == cfg


def test_unknown_game_key_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"game\.nope"):
        apply_overrides(load_config(preset="tiny"), ["game.nope=1"])
    with pytest.raises(ValueError, match=r"game\.observe"):
        apply_overrides(load_config(preset="tiny"), ["game.observe=maybe"])


# --- resume and fork ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_run(tiny_cfg: Config, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A ``tiny_cfg`` run directory stopped at step 30 (checkpoints at 0 and 30), observers off."""
    rd = RunDir.create(tmp_path_factory.mktemp("game_cfg") / "run", tiny_cfg)
    rd.update_meta(start_step=0)
    runner.run(rd, tiny_cfg, init_state(tiny_cfg), 30, quiet=True)
    return rd.path


def test_resume_accepts_game_observe_key(small_run: Path, tmp_path: Path) -> None:
    copy = tmp_path / "run"
    shutil.copytree(small_run, copy)
    before = RunDir.open(copy).meta()["config_hash"]
    assert cli.main(["resume", str(copy), "--set", OBSERVE, "--steps", "20", "--quiet"]) == 0
    rd = RunDir.open(copy)
    assert rd.config().game.observe is True
    assert json.loads((copy / "config.json").read_text())["game"] == {
        **config_to_dict(Config())["game"],
        "observe": True,
    }
    assert config_hash(rd.config()) == before  # load_checkpoint's hash check passed on the unchanged hash
    assert [r["step"] for r in rd.metrics_rows()] == [10, 20, 30, 40, 50]
    assert [st for st, _ in rd.checkpoints()][-1] == 50


def test_resume_accepts_game_house_rate_key(small_run: Path, tmp_path: Path) -> None:
    """``game.house_rate`` is an observer key: resume may change it (unhashed: the checkpoint still loads)."""
    copy = tmp_path / "run"
    shutil.copytree(small_run, copy)
    before = RunDir.open(copy).meta()["config_hash"]
    assert cli.main(["resume", str(copy), "--set", HOUSE_RATE, "--steps", "20", "--quiet"]) == 0
    rd = RunDir.open(copy)
    assert rd.config().game.house_rate == 0.25 and rd.config().game.observe is False
    assert json.loads((copy / "config.json").read_text())["game"] == {
        "observe": False,
        "house_rate": 0.25,
        "tracers": 0,
        "tracer_period": 500,
    }
    assert config_hash(rd.config()) == before
    assert [r["step"] for r in rd.metrics_rows()] == [10, 20, 30, 40, 50]
    assert all("observers" not in r for r in rd.metrics_rows())  # observers stay off: the key is inert
    assert not (copy / "observers").exists()


def test_resume_with_observe_writes_observer_rows_and_sidecar(
    small_run: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    social_every(monkeypatch, 15)
    copy = tmp_path / "run"
    shutil.copytree(small_run, copy)
    assert not (copy / "observers").exists()  # the source run observed nothing: no observers/, no social log
    capsys.readouterr()
    assert cli.main(["resume", str(copy), "--set", OBSERVE, "--steps", "20"]) == 0
    rd = RunDir.open(copy)
    rows = rd.metrics_rows()
    assert [r["step"] for r in rows if "observers" not in r] == [10, 20, 30]
    assert [r["step"] for r in rows if "observers" in r] == [40, 50]
    assert (copy / "observers" / "step_0000000050.npz").is_file()
    assert "observers:" in capsys.readouterr().err  # the note: no sidecar at the resumed step
    # the social log starts at the resume: only the record of 45, with the resume step as its origin, and its
    # houses are inexact (the observer started after step 0, so the clades stand in for the houses)
    (record,) = rd.social_records()
    assert (record["step"], record["origin"], record["window_start"]) == (45, 30, 30)
    assert record["summary"]["houses"]["exact"] is False
    assert [r["observers"]["social"]["snapshot_step"] for r in rows if "observers" in r] == [30, 45]
    assert all(r["observers"]["social"]["origin"] == 30 for r in rows if "observers" in r)


def test_resume_with_a_changed_house_rate_restarts_the_social_tracking(
    tiny_cfg: Config, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rate that differs from the sidecar's is a meta mismatch: exit 0, the note on stderr, fresh observers
    whose records carry the resume step as their origin; the log keeps the earlier observer's records."""
    social_every(monkeypatch, 15)
    cfg = apply_overrides(tiny_cfg, [OBSERVE, "run.checkpoint_every=30"])
    rd = RunDir.create(tmp_path / "run", cfg)
    rd.update_meta(start_step=0)
    runner.run(rd, cfg, init_state(cfg), 30, quiet=True)
    assert [r["origin"] for r in rd.social_records()] == [0, 0]
    capsys.readouterr()
    assert cli.main(["resume", str(rd.path), "--set", HOUSE_RATE, "--steps", "30"]) == 0
    err = capsys.readouterr().err
    assert "observers: meta.house_rate is 0.00048828125, expected 0.25;" in err
    assert (
        "social tracking restarts at step 30 (houses seeded from clades; new ids, records with origin 30)"
        in err
    )
    rd = RunDir.open(rd.path)
    assert [(r["step"], r["origin"]) for r in rd.social_records()] == [(15, 0), (30, 0), (45, 30), (60, 30)]
    assert rd.config().game.house_rate == 0.25
    # the new rate is now the run's: a further resume (config unchanged) loads the sidecar at 60 silently
    capsys.readouterr()
    assert cli.main(["resume", str(rd.path), "--steps", "15"]) == 0
    assert "observers:" not in capsys.readouterr().err
    assert [(r["step"], r["origin"]) for r in RunDir.open(rd.path).social_records()][-1] == (75, 30)


@pytest.mark.parametrize(
    "override", ["game.nope=1", "game.observe.x=1", "game.house_rate.x=1", "game=1", "game.house=0.5"]
)
def test_resume_rejects_other_game_keys(
    small_run: Path, override: str, capsys: pytest.CaptureFixture[str]
) -> None:
    files = (small_run / "config.json", small_run / "metrics.jsonl")
    before = {p: p.read_bytes() for p in files}
    capsys.readouterr()
    assert cli.main(["resume", str(small_run), "--set", override, "--checkpoint", "0"]) == 2
    assert "game.<key>" in capsys.readouterr().err
    assert {p: p.read_bytes() for p in files} == before  # nothing was truncated or rewritten
    assert [st for st, _ in RunDir.open(small_run).checkpoints()] == [0, 30]


def test_resume_help_names_the_game_key(capsys: pytest.CaptureFixture[str]) -> None:
    """``--set`` of ``resume`` advertises every section it accepts (run, director and the game keys)."""
    capsys.readouterr()
    assert cli.main(["resume", "--help"]) == 0  # main turns argparse's SystemExit(0) into a return code
    text = " ".join(capsys.readouterr().out.split())
    assert "run.KEY=VALUE|director.KEY=VALUE|game.KEY=VALUE" in text


def test_resume_error_lists_the_two_game_keys(small_run: Path, capsys: pytest.CaptureFixture[str]) -> None:
    capsys.readouterr()
    assert cli.main(["resume", str(small_run), "--set", "game.nope=1", "--checkpoint", "0"]) == 2
    err = " ".join(capsys.readouterr().err.split())
    assert "game.<key> for ['house_rate', 'observe', 'tracer_period', 'tracers']" in err


def test_fork_can_set_observe(small_run: Path, tmp_path: Path) -> None:
    out = tmp_path / "fork"
    argv = ["run", "--init-from", f"{small_run}:30", "--out", str(out), "--steps", "0", "--quiet"]
    assert cli.main([*argv, "--set", OBSERVE]) == 0
    fork = RunDir.open(out)
    assert fork.config().game.observe is True
    assert fork.meta()["config_hash"] == RunDir.open(small_run).meta()["config_hash"]
    assert [st for st, _ in fork.checkpoints()] == [30]


def test_fork_with_observe_starts_its_observers_fresh(small_run: Path, tmp_path: Path) -> None:
    import numpy as np

    out = tmp_path / "fork"
    argv = ["run", "--init-from", f"{small_run}:30", "--out", str(out), "--steps", "0", "--quiet"]
    assert cli.main([*argv, "--set", OBSERVE]) == 0
    with np.load(out / "observers" / "step_0000000030.npz", allow_pickle=False) as z:
        assert not z["n_obs"].any()
    assert not (out / "observers" / "social.jsonl").exists()  # a fork's social log starts absent
