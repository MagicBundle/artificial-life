"""The voice log on the engine (DESIGN §39.14): the replay changes no trajectory (digest equal to a plain
run, with
the counterfactual steps on), is deterministic, and the CLI works end to end on tiny box and sphere worlds."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from alife import cli, layout, runio, runner, spatial
from alife.analysis.emergence import prereg as pr
from alife.analysis.voice import __main__ as vcli
from alife.analysis.voice import replay, spec
from alife.analysis.voice.tables import Tables
from alife.config import apply_overrides, load_config
from alife.runio import RunDir
from alife.state import init_state, state_digest, to_numpy
from alife.step import make_step
from helpers import set_output_bias

RUN = [
    "run.keep_checkpoints=10",
    "run.metrics_every=5",
    "run.checkpoint_every=50",
    "run.heavy_metrics_every=5",
    "brain.weight_dtype=float32",
]
PRESETS = {"box": "tiny3d", "sphere": "tiny3s", "flat": "tiny"}


# founders that attack, bond, harvest and breed, so that every flag of the observer channel is exercised
BIAS = {layout.OUT_HARVEST: 2.0, layout.OUT_BOND: 2.0, layout.OUT_ATTACK: 1.0, layout.OUT_REPRODUCE: 2.0}


def _make_run(kind: str, path: Path, steps: int = 200, *extra: str, bias: dict | None = None) -> Path:
    cfg = load_config(preset=PRESETS[kind], overlays=["comm"], overrides=[*RUN, *extra])
    rd = RunDir.create(path, cfg)
    state = init_state(cfg) if bias is None else set_output_bias(init_state(cfg), cfg, bias)
    assert runner.run(rd, cfg, state, steps, quiet=True).status == "done"
    return path


@pytest.fixture(scope="module")
def box_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _make_run("box", tmp_path_factory.mktemp("voice") / "box", bias=BIAS)


def _record(run: Path, out: Path, **kw: object) -> Tables:
    args = dict(start=50, steps=100, every=10, rows=64, cf_every=2, log=lambda m: None)
    args.update(kw)
    return replay.record(run, out=out, **args)  # type: ignore[arg-type]


def _engine_truth(run: Path, start: int, steps: int) -> dict[int, SimpleNamespace]:
    """What the engine did at every step of the window, by row, read straight from an observed plain run."""
    rd = RunDir.open(run)
    cfg = rd.config()
    spatial.prepare_knn(cfg)
    f = make_step(cfg, observe=True)
    s = runio.load_checkpoint(dict(rd.checkpoints())[start], cfg)
    out = {}
    for t in range(start, start + steps + 1):
        uid, harvest = to_numpy(s.uid), to_numpy(s.last_harvest)
        s, o = f(s)
        mx.eval(s, o.base(), *o.acts.ext.comm)
        c = o.acts.ext.comm
        out[t] = SimpleNamespace(
            uid=uid,
            row={int(u): i for i, u in enumerate(uid)},
            harvest=harvest,
            flags=to_numpy(o.acts.flags).astype(np.int64),
            thrust=to_numpy(o.acts.thrust).astype(np.int64),
            voice=to_numpy(c.voice),
            heard=to_numpy(c.heard),
            audible=to_numpy(c.audible),
        )
    return out


def test_the_tables_hold_what_the_engine_said_heard_and_did(box_run: Path, tmp_path: Path) -> None:
    """The table-to-engine mapping, row by row against an independent observed run: the flags, the thrust,
    the call
    spoken and heard, the audible count and the harvest of the next state (it is the step just taken, not
    the step
    that starts). Swapping voice and heard, zeroing the flags or taking a snapshot one step off fails here."""
    tab = _record(box_run, tmp_path / "a", cf_every=0)
    truth = _engine_truth(box_run, 50, 101)
    h, s = tab.hearers, tab.senders
    flag = {"r_attack": 1, "r_bond": 4, "r_speak": 1024}
    for i in range(len(h["uid"])):
        u, uid = int(h["step"][i]), int(h["uid"][i])
        e, nxt = truth[u], truth[u + 1]
        r = e.row[uid]
        for k, bit in flag.items():
            assert h[k][i] == bool(e.flags[r] & bit), (k, i)
        assert h["c_budding"][i] == bool(e.flags[r] & 8)
        assert h["r_thrust"][i] == abs(e.thrust[r]) and h["r_loud"][i] == int(e.voice[r].sum())
        assert (h["heard"][i] == e.heard[r]).all() and h["audible"][i] == e.audible[r]
        assert h["r_eat"][i] == nxt.harvest[nxt.row[uid]]
    for i in range(len(s["uid"])):
        u, uid = int(s["step"][i]), int(s["uid"][i])
        prev = truth[u - 1]
        assert (s["voice"][i] == prev.voice[prev.row[uid]]).all()  # the call spoken at u - 1
    for k in ("r_speak", "r_loud", "r_eat"):  # the comparison is not vacuous
        assert h[k].any(), k
    assert h["r_attack"].any() or h["r_bond"].any()


def test_a_heard_call_is_at_least_the_loudest_sources_weighted_call(box_run: Path, tmp_path: Path) -> None:
    """The engine adds the audible callers' weighted calls (a saturating sum of non-negative terms), so what a
    hearer heard is at least the weighted call of the source the table names: no violation under ``none``;
    under ``deaf`` everything heard is zero."""
    tab = _record(box_run, tmp_path / "a", cf_every=0)
    h = tab.hearers
    has = h["has_src"] > 0
    a = np.clip(1.0 - h["src_dist"][has], 0.0, 1.0)
    floor = np.floor(a[:, None] * h["src_voice"][has].astype(float)) - 2.0  # the 1/255 quantisation of both
    assert has.sum() > 100 and (h["heard"][has].astype(float) >= floor).all()
    assert (h["audible"][has] > 0).all()
    deaf = _make_run("box", tmp_path / "deaf", 200, "comm.control=deaf")
    td = _record(deaf, tmp_path / "d", cf_every=0)
    assert not td.hearers["heard"].any() and not td.hearers["audible"].any()
    assert td.senders["voice"].any()  # they still speak: the cost and the speaking path stay


def test_the_replay_changes_no_trajectory(box_run: Path, tmp_path: Path) -> None:
    """The observed replay with the counterfactual steps on ends in the state a plain run reaches (digest),
    and
    in the run's own checkpoint; the stored metrics rows on the way are reproduced."""
    tab = _record(box_run, tmp_path / "a")
    rd = RunDir.open(box_run)
    cfg = rd.config()
    spatial.prepare_knn(cfg)
    plain = make_step(cfg)
    s = runio.load_checkpoint(dict(rd.checkpoints())[50], cfg)
    for _ in range(100):
        s, _ = plain(s)
        mx.eval(s)
    assert tab.meta["end_state_digest"] == state_digest(s)
    assert tab.meta["end_digest"]["equal"] and tab.meta["rows_verified"] >= 10
    assert tab.meta["samples"] == 9 and tab.cf["stats"].shape[0] == 5


def test_the_replay_is_deterministic_and_leaves_the_run_alone(box_run: Path, tmp_path: Path) -> None:
    before = sorted(p.name for p in box_run.rglob("*"))
    a = _record(box_run, tmp_path / "a", cf_every=0)
    b = _record(box_run, tmp_path / "b", cf_every=0)
    for side in ("senders", "hearers"):
        for k, v in getattr(a, side).items():
            assert np.array_equal(v, getattr(b, side)[k], equal_nan=True), (side, k)
    assert a.cf == {} and sorted(p.name for p in box_run.rglob("*")) == before


def test_the_tables_hold_what_the_engine_said_and_heard(box_run: Path, tmp_path: Path) -> None:
    tab = _record(box_run, tmp_path / "a", cf_every=0)
    s, h = tab.senders, tab.hearers
    assert s["voice"].shape[1] == 4 and (s["voice"] > 0).any() and (h["audible"] > 0).any()
    assert (h["heard"][h["audible"] == 0] == 0).all()  # nobody in range, nothing heard
    assert h["lineage"].min() >= 0 and h["lineage"].max() < spec.LINEAGE_GROUPS
    assert tab.meta["lineage_groups"]["k"] == spec.LINEAGE_GROUPS
    has = h["has_src"] > 0
    assert has.any() and (h["src_dist"][has] <= 1.0).all() and (h["src_voice"][has].sum(axis=1) > 0).all()


def test_the_counterfactual_leaves_a_deaf_row_alone(box_run: Path, tmp_path: Path) -> None:
    """Rows with no audible neighbour cannot hear anything under any control, so the step under ``deaf`` and
    under ``decoy`` equals the run's own for them; rows that do hear change."""
    tab = _record(box_run, tmp_path / "a", cf_every=1)
    assert list(replay.CF_METRICS) == [
        "d_thrust",
        "flip_attack",
        "flip_bond",
        "flip_speak",
        "d_voice",
        "d_vel",
        "n",
    ]
    st = tab.cf["stats"]  # (samples, modes, groups, metrics)
    assert tab.meta["cf_modes"] == ["deaf", "decoy"]
    silent = st[:, :, 1, :]  # the control group: rows with no audible neighbour (NaN where a sample has none)
    assert (np.nan_to_num(silent[..., :-1]) == 0).all() and (np.nan_to_num(silent[..., -1]) > 0).any()
    assert np.nanmax(st[:, 0, 0, 0]) > 0 and np.nanmax(st[:, 0, 0, 4]) > 0  # deaf changes thrust and calls


@pytest.mark.parametrize("kind", ["sphere", "flat"])
def test_the_log_runs_on_sphere_and_plane(kind: str, tmp_path: Path) -> None:
    run = _make_run(kind, tmp_path / kind, 120)
    tab = _record(run, tmp_path / "o", steps=50, cf_every=2)
    assert tab.meta["end_digest"]["equal"] and len(tab.senders["step"]) > 0
    s = vcli.summary.build(tab, tuple(tab.meta["window"]), n_perm=20, n_perm_block=5, top=2)  # type: ignore[arg-type]
    json.dumps(vcli.jsonable(s), allow_nan=False)


def test_a_run_without_comm_is_refused(tmp_path: Path) -> None:
    cfg = apply_overrides(load_config(preset="tiny"), RUN)
    rd = RunDir.create(tmp_path / "plain", cfg)
    runner.run(rd, cfg, init_state(cfg), 60, quiet=True)
    with pytest.raises(ValueError, match=r"comm\.enabled"):
        replay.record(tmp_path / "plain", start=0, steps=40, out=tmp_path / "o", log=lambda m: None)


def test_the_cli_log_summarize_drift_and_meaning(
    box_run: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    out = tmp_path / "log"
    base = ["--every", "10", "--rows", "64", "--cf-every", "2", "--perms", "20", "--blocks", "4"]
    assert vcli.main(["log", str(box_run), "--from", "50", "--steps", "100", "--out", str(out), *base]) == 0
    for f in ("tables.npz", "summary.json", "summary.md"):
        assert (out / f).is_file(), f
    assert "Voice log" in (out / "summary.md").read_text()
    assert vcli.main(["summarize", str(out), "--perms", "20", "--no-tests"]) == 0
    assert "tests" not in json.loads((out / "summary.json").read_text())
    assert vcli.main(["drift", str(out), str(out), "--out", str(tmp_path / "d.json")]) == 0
    assert json.loads((tmp_path / "d.json").read_text())["population_js"] == 0.0
    # the meaning test on forks of the run: 3 arms x 2 seeds, a small pre-registration
    d = spec.default_prereg(code_git="test")
    d["arms"].update(fork_steps=100, table_window=[0, 100])
    d["fitness"]["blocks"] = 4
    d["sampling"].update(blocks=4, every=10, rows=64)
    d["tests"]["n_perm"] = 20
    d["frozen"] = True
    pr.write_prereg(tmp_path / "prereg.json", d)
    arms = []
    for arm in spec.ARMS:
        for seed in (1, 2):
            fork = tmp_path / f"{arm}_s{seed}"
            args = [
                "run",
                "--init-from",
                f"{box_run}:50",
                "--set",
                f"comm.control={arm}",
                "--seed",
                str(seed),
            ]
            args += ["--steps", "100", "--checkpoint-every", "50", "--out", str(fork), "--quiet"]
            assert cli.main(args) == 0
            lg = tmp_path / "logs" / fork.name
            assert (
                vcli.main(["log", str(fork), "--from", "50", "--steps", "100", "--out", str(lg), *base]) == 0
            )
            arms.append(f"--arm={arm}:{seed}={fork},{lg}")
    mout = tmp_path / "m"
    code = vcli.main(["meaning", *arms, "--prereg", str(tmp_path / "prereg.json"), "--out", str(mout)])
    assert code == 0
    res = json.loads((mout / "meaning.json").read_text())
    assert res["decision"]["verdict"] in spec.VERDICTS and res["confirmatory"] is True
    assert "Voice meaning test" in (mout / "meaning.md").read_text()
    assert vcli.main(["meaning", *arms, "--out", str(tmp_path / "x")]) == 2  # no pre-registration: refused
    assert "pre-registration" in capsys.readouterr().err
    d["tests"]["n_perm"] = 200  # a summary that ran 20 permutations is not the registered test: refused
    pr.write_prereg(tmp_path / "prereg200.json", d)
    code = vcli.main(
        ["meaning", *arms, "--prereg", str(tmp_path / "prereg200.json"), "--out", str(tmp_path / "y")]
    )
    assert code == 2 and "rebuild the summary" in capsys.readouterr().err
