"""Arms races, the observer PR A2 (DESIGN §42.6): the realised-contest counters, ``Acts.ext.arms``, the
``CONTESTED`` flag, the snapshot and interval metrics and the ``arms_<step>.npz`` dump."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from alife import arms, featobs, runner
from alife.config import Config, apply_overrides, load_config
from alife.metrics import snapshot_metrics
from alife.runio import RunDir
from alife.state import CounterIdx, counter_index, init_state, state_digest, to_numpy
from alife.step import ActFlag, make_step

REAL = [
    "arms.enabled=true",
    "arms.kappa=2",
    "arms.key_floor=0.2",
    "arms.speed_slow=0.3",
    "arms.upkeep_atk=20",
    "arms.upkeep_arm=20",
    "arms.init_atk=0.3",
    "arms.init_arm=0.4",
    "arms.sigma=0.05",
    "arms.sigma_phase=0.05",
]
NAMES = ("ARMS_ATK_SUM", "ARMS_VIC_ARM_SUM", "ARMS_YIELD_SUM", "ARMS_MATCH_SUM", "ARMS_VIC_TOX_SUM")


def _cfg(nd: int, extra: list[str] = REAL, *, compile: bool = True) -> Config:
    preset = {2: "tiny", 3: "tiny3d"}[nd]
    cfg = load_config(preset=preset, overrides=["brain.weight_dtype=float32", *extra])
    return cfg if compile else replace(cfg, run=replace(cfg.run, compile=False))


def _start(cfg: Config):
    s = init_state(cfg)
    s = s._replace(birth_step=mx.where(s.alive, -50, s.birth_step))
    mx.eval(s)
    return s


@pytest.mark.parametrize("nd", [2, 3])
def test_the_counters_are_the_sums_of_the_per_row_channel_and_follow_the_numpy_contest(nd: int) -> None:
    cfg = _cfg(nd)
    fn = make_step(cfg, observe=True)
    s = _start(cfg)
    seen = 0
    for _ in range(120):
        prev = s
        s, out = fn(s)
        mx.eval(s, *out)
        c = to_numpy(out.counters)
        contest = to_numpy(out.acts.ext.arms.contest).astype(np.int64)
        flags = to_numpy(out.acts.flags).astype(np.int64)
        att = (flags & ActFlag.ATTACKED.value) != 0
        n_att = int(c[CounterIdx.ATTACKS])
        assert n_att == int(att.sum())
        got = [int(c[counter_index(cfg, n)]) for n in NAMES]
        assert got[:4] == [int(contest[:, k].sum()) for k in range(4)]
        assert got[4] == 0  # no toxin in this stage
        assert not contest[~att].any()  # zero for every row that did not attack
        if n_att:
            seen += 1
            tr = to_numpy(prev.arms).astype(np.float32)
            tgt = to_numpy(out.acts.attack_target)
            i = np.flatnonzero(att)
            m = arms.contest_np(
                tr[i, arms.ATK], tr[tgt[i], arms.ARM], tr[i, arms.KEY], tr[tgt[i], arms.LOCK], cfg
            )
            # the realised product equals the numpy mirror (no shelter, no guard in this world), to the byte
            assert np.abs(contest[i, 2] - np.rint(256.0 * m)).max() <= 1
            assert np.array_equal(contest[i, 0], np.rint(256.0 * tr[i, arms.ATK]))
            assert np.array_equal(contest[i, 1], np.rint(256.0 * tr[tgt[i], arms.ARM]))
            contested = (flags & ActFlag.CONTESTED.value) != 0
            assert np.array_equal(contested[i], (m > 0) & (m < 1))
            assert not contested[~att].any()
    assert seen > 10


def test_the_null_contest_has_unit_factors_and_no_contested_row() -> None:
    cfg = _cfg(2, ["arms.enabled=true"])  # zero traits, key_floor 1: the product is exactly 1
    fn = make_step(cfg, observe=True)
    s = _start(cfg)
    attacks = 0
    for _ in range(80):
        s, out = fn(s)
        mx.eval(s, *out)
        flags = to_numpy(out.acts.flags).astype(np.int64)
        att = (flags & ActFlag.ATTACKED.value) != 0
        contest = to_numpy(out.acts.ext.arms.contest)
        attacks += int(att.sum())
        assert (contest[att][:, 2:] == 256).all() and not (flags & ActFlag.CONTESTED.value).any()
        c = to_numpy(out.counters)
        assert int(c[counter_index(cfg, "ARMS_YIELD_SUM")]) == 256 * int(att.sum())
    assert attacks > 0


@pytest.mark.parametrize("nd", [2, 3])
def test_observing_an_arms_world_changes_no_state_value(nd: int) -> None:
    cfg = _cfg(nd, compile=False)
    plain, obs = _start(cfg), _start(cfg)
    f0, f1 = make_step(cfg), make_step(cfg, observe=True)
    for _ in range(25):
        plain, o0 = f0(plain)
        obs, o1 = f1(obs)
        mx.eval(plain, obs, o0, *o1)
        assert state_digest(plain) == state_digest(obs)
        assert np.array_equal(to_numpy(o0.counters), to_numpy(o1.counters))


def test_an_arms_off_world_has_no_arms_counters_or_channel() -> None:
    cfg = _cfg(2, [])
    _, out = make_step(cfg, observe=True)(_start(cfg))
    mx.eval(*out)
    assert out.acts.ext is None
    assert len(to_numpy(out.counters)) == 27


# --- the numpy statistics -----------------------------------------------------------------------------------


def test_circular_statistics() -> None:
    r, ang = featobs.circular(np.array([0.1, 0.1, 0.1]))
    assert r == pytest.approx(1.0) and ang == pytest.approx(0.1)
    r, _ = featobs.circular(np.array([0.0, 0.5]))  # opposite phases cancel
    assert r == pytest.approx(0.0, abs=1e-12)
    r, ang = featobs.circular(np.array([0.95, 0.05]))  # across the seam: mean direction 0
    assert r > 0.9 and min(ang, 1 - ang) == pytest.approx(0.0, abs=1e-9)
    assert featobs.circular(np.zeros(0)) == (None, None)


def test_pearson_is_none_without_variance() -> None:
    assert featobs.pearson(np.arange(5), 2.0 * np.arange(5) + 1) == pytest.approx(1.0)
    assert featobs.pearson(np.ones(5), np.arange(5)) is None
    assert featobs.pearson(np.arange(1), np.arange(1)) is None


# --- metrics and dumps --------------------------------------------------------------------------------------


def test_the_snapshot_and_interval_rows_carry_the_arms_keys(tmp_path: Path) -> None:
    cfg = apply_overrides(
        _cfg(2), ["run.metrics_every=20", "run.checkpoint_every=40", "run.heavy_metrics_every=40"]
    )
    rd = RunDir.create(tmp_path / "run", cfg)
    assert runner.run(rd, cfg, _start(cfg), 60, quiet=True).status == "done"
    rows = rd.metrics_rows()
    assert len(rows) == 3
    json.dumps(rows, allow_nan=False)
    for key in (
        "arms_atk_mean", "arms_atk_q10", "arms_atk_q50", "arms_atk_q90", "arms_arm_mean", "arms_tox_mean",
        "arms_sig_q90", "arms_key_R", "arms_lock_R", "arms_key_mean", "arms_lock_mean", "arms_arm_bonds_corr",
        "arms_atk_edge_frac", "arms_arm_edge_frac", "arms_att_atk", "arms_vic_arm", "arms_yield",
        "arms_match", "arms_vic_tox",
    ):  # fmt: skip
        assert key in rows[-1], key
    last = rows[-1]
    assert last["arms_atk_mean"] == pytest.approx(0.3, abs=0.2) and 0.0 <= last["arms_key_R"] <= 1.0
    if last["attacks"]:
        assert 0.0 < last["arms_yield"] <= 1.0 and 0.0 <= last["arms_match"] <= 1.0
    else:
        assert last["arms_yield"] is None  # no attack: undefined, not NaN


def test_a_world_without_arms_has_the_old_snapshot_keys() -> None:
    cfg = _cfg(2, [])
    snap = snapshot_metrics(init_state(cfg), cfg, heavy=False)
    assert not [k for k in snap if k.startswith("arms_")]


def test_the_arms_dump_samples_the_alive_rows_by_a_fixed_stride(tmp_path: Path) -> None:
    cfg = apply_overrides(
        _cfg(2), ["run.arms_every=10", "arms.dump_n=7", "run.metrics_every=10", "run.checkpoint_every=30"]
    )
    rd = RunDir.create(tmp_path / "run", cfg)
    assert runner.run(rd, cfg, _start(cfg), 30, quiet=True).status == "done"
    files = rd.dump_files("arms")
    assert [st for st, _ in files] == [10, 20, 30] and files[0][1].name == "arms_0000000010.npz"
    z = rd.read_arms(30)
    assert set(z) == {"step", "uid", "generation", "clade", "birth_step", "energy", "n_bonds", "traits"}
    assert int(z["step"]) == 30 and z["uid"].shape == (7,) and z["traits"].shape == (7, 6)
    assert z["traits"].dtype == np.float32 and z["uid"].dtype == np.int64 and z["n_bonds"].dtype == np.int8
    assert (z["uid"] >= 0).all() and len(set(z["uid"].tolist())) == 7
    assert (z["traits"] >= 0).all() and (z["traits"] <= 1).all()


def test_the_sample_takes_every_row_when_there_are_fewer_than_dump_n() -> None:
    host = {
        "arms.alive": np.array([1, 0, 1, 1, 0], bool),
        "arms.uid": np.arange(5, dtype=np.int64),
        "arms.traits": np.arange(30, dtype=np.float32).reshape(5, 6),
    }
    out = runner.arms_sample(host, 12, 100)
    assert out["uid"].tolist() == [0, 2, 3] and int(out["step"]) == 12
    one = runner.arms_sample(host, 12, 2)  # a fixed stride over the alive rows
    assert one["uid"].tolist() == [0, 2]


def test_the_dump_cadence_needs_arms_and_is_a_resume_key(tmp_path: Path) -> None:
    from alife.cli import RESUME_EXT_KEYS, RESUME_KEYS, _resume_allowed
    from alife.config import ConfigError, config_hash

    with pytest.raises(ConfigError, match="arms_every"):
        apply_overrides(_cfg(2, []), ["run.arms_every=10"])
    assert "arms_every" in RESUME_KEYS and ("arms", "dump_n") in RESUME_EXT_KEYS
    assert _resume_allowed("run.arms_every=50") and _resume_allowed("arms.dump_n=100")
    assert not _resume_allowed("arms.kappa=3")
    a = _cfg(2)
    assert config_hash(a) == config_hash(apply_overrides(a, ["arms.dump_n=17", "run.arms_every=5"]))
