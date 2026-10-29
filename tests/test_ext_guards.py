"""Two guards of the feature series (DESIGN §43.7, §43.2): the defaults of the keys that ``config.json`` omits
are frozen, and a config that disagrees with a checkpoint on the feature layout is refused, never guessed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import ext_stub as stub
from alife import config
from alife.cli import main
from alife.config import Config
from alife.runio import CheckpointError, RunDir, load_checkpoint, write_state_file
from alife.state import init_state

# E1. A key left out of config.json (and of the checkpoint metadata) while it equals its default takes the
# CURRENT default when an older run is read back. If a later PR changed such a default, every resumed world
# would silently adopt it with an unchanged hash. So the defaults of omitted keys never change (a behaviour
# change needs a NEW key), and this table is the pin. Adding a key to SERIAL_OMIT_KEYS (a feature's
# lever_keys) adds a row here.
FROZEN_OMITTED_DEFAULTS = {
    ("combat", "protect_floor"): 0.0,
    ("arms", "enabled"): False,
    ("arms", "kappa"): 2.0,
    ("arms", "a0"): 1.0,
    ("arms", "key_floor"): 1.0,
    ("arms", "speed_slow"): 0.0,
    ("arms", "upkeep_atk"): 0.0,
    ("arms", "upkeep_arm"): 0.0,
    ("arms", "init_atk"): 0.0,
    ("arms", "init_arm"): 0.0,
    ("arms", "sigma"): 0.0,
    ("arms", "sigma_phase"): 0.0,
    ("structures", "enabled"): False,
    ("structures", "cap"): 4096,
    ("structures", "build_max"): 16,
    ("structures", "build_cost"): 2,
    ("structures", "decay_q12"): 16,
    ("structures", "shelter_gain"): 0.8,
    ("structures", "barrier_gain"): 3.0,
    ("structures", "shield_gain"): 0.5,
    ("structures", "reach"): 0.5,
    ("structures", "open_frac"): 0.5,
    ("structures", "aperture"): 0.25,
    ("structures", "exit_ease"): 0.75,
    ("structures", "silent_build"): False,
    ("bonds", "share_mode"): 0,
    ("bonds", "share_kappa"): 0.125,
    ("bonds", "share_kin_floor"): 1.0,
    ("roles", "enabled"): False,
    ("roles", "forage_gain"): 0.0,
    ("roles", "forage_defend_pen"): 0.0,
    ("roles", "guard_harvest_pen"): 0.0,
    ("roles", "guard_shield"): 0.0,
    ("roles", "synergy"): 0.0,
    ("roles", "ramp_steps"): 64,
    ("roles", "threshold"): 0.5,
    ("roles", "switch_cost"): 0,
    ("roles", "inherit_role"): False,
    ("comm", "enabled"): False,
    ("comm", "width"): 4,
    ("comm", "range"): 0.0,
    ("comm", "atten_pow"): 1.0,
    ("comm", "kin_exp"): 0.0,
    ("comm", "cost"): 4.0,
    ("comm", "control"): "none",
    (
        "arms",
        "dump_n",
    ): 4096,  # the observer PR (S2/A2): an observer key and the dump cadences of the run section
    ("run", "mat_every"): 0,
    ("run", "arms_every"): 0,
    ("niche", "farm"): False,
    ("niche", "farm_rate"): 8,
    ("niche", "farm_cap"): 4095,
    ("niche", "farm_cost"): 30,
    ("niche", "farm_decay_q16"): 16,
    ("niche", "farm_bonus_frac"): 0.5,
    ("niche", "silent_init"): True,
    ("niche", "control"): "none",
}


def test_the_defaults_of_omitted_keys_are_frozen() -> None:
    assert set(config.SERIAL_OMIT_KEYS) == set(FROZEN_OMITTED_DEFAULTS), (
        "SERIAL_OMIT_KEYS and the frozen table disagree: a new omitted key needs its default pinned here"
    )
    cfg = Config()
    for (sec, key), value in FROZEN_OMITTED_DEFAULTS.items():
        assert getattr(getattr(cfg, sec), key) == value, f"{sec}.{key}: the default of an omitted key changed"
    # every omitted key moves no hash at its default: the run section is outside the hash altogether
    assert {k for k in config.SERIAL_OMIT_KEYS if k[0] != "run"} <= set(config.POST_V01_KEYS)


# E3. Config versus save.


@pytest.fixture(autouse=True)
def _registry(monkeypatch: pytest.MonkeyPatch) -> None:
    stub.install(monkeypatch)


def _cfg(*names: str):
    cfg = config.load_config(preset="tiny", overrides=["brain.weight_dtype=float32"])
    return stub.stub_cfg(cfg, *names) if names else cfg


def test_load_checkpoint_refuses_every_disagreement_with_a_clear_message(tmp_path: Path) -> None:
    for saved, wanted in (
        (("structures",), ()),
        ((), ("structures",)),
        (("structures",), ("comm",)),
        (("bonds", "comm"), ("bonds",)),
    ):
        path = tmp_path / f"{'-'.join(saved) or 'off'}_{'-'.join(wanted) or 'off'}.safetensors"
        write_state_file(path, init_state(_cfg(*saved)), _cfg(*saved))
        with pytest.raises(CheckpointError, match="disagree on the feature layout") as e:
            load_checkpoint(
                path, _cfg(*wanted), check_hash=False
            )  # also where the hash is not checked (a fork)
        assert "layout_features" in str(e.value) and "nothing is guessed" in str(e.value)


def _make_run(tmp_path: Path, names=()):
    """A tiny run directory with its step-0 checkpoint, written by the engine."""
    cfg = _cfg(*names)
    rd = RunDir.create(tmp_path / "run", cfg)
    write_state_file(rd.path / "checkpoints" / "step_0000000000.safetensors", init_state(cfg), cfg)
    (rd.path / "checkpoints" / "LATEST").write_text("step_0000000000.safetensors")
    return rd, cfg


def test_resume_with_a_set_that_changes_the_feature_set_is_refused(tmp_path: Path, capsys) -> None:
    rd, _ = _make_run(tmp_path)
    cfg_json = json.loads((rd.path / "config.json").read_text())
    assert "layout_features" not in rd.meta()
    rc = main(
        ["resume", str(rd.path), "--steps", "2", "--quiet", "--set", f"run.llm_model={stub.PREFIX}structures"]
    )
    err = capsys.readouterr().err
    assert rc != 0 and "disagree on the feature layout" in err
    assert json.loads((rd.path / "config.json").read_text()) == cfg_json  # nothing was written


def test_resume_of_a_feature_run_with_the_feature_dropped_is_refused(tmp_path: Path, capsys) -> None:
    rd, _ = _make_run(tmp_path, ("niche",))
    rc = main(["resume", str(rd.path), "--steps", "2", "--quiet", "--set", "run.llm_model=gemma"])
    assert rc != 0 and "disagree on the feature layout" in capsys.readouterr().err


def test_a_fork_into_another_feature_set_is_refused(tmp_path: Path, capsys) -> None:
    rd, _ = _make_run(tmp_path)
    out = tmp_path / "fork"
    rc = main(
        [
            "run",
            "--init-from",
            str(rd.path),
            "--out",
            str(out),
            "--steps",
            "2",
            "--quiet",
            "--set",
            f"run.llm_model={stub.PREFIX}comm",
        ]
    )
    assert rc != 0 and "disagree on the feature layout" in capsys.readouterr().err
    assert not list(out.glob("checkpoints/*.safetensors"))


def test_every_state_load_goes_through_load_checkpoint() -> None:
    """Resume, fork, inspect, play and the stream all read states with ``load_checkpoint`` (one guard)."""
    src = Path(config.__file__).parent
    offenders = [
        p.name
        for p in src.rglob("*.py")
        if p.name not in ("runio.py", "state.py") and "state_from_arrays(" in p.read_text()
    ]
    assert offenders == [], offenders
