"""§22.5 end-to-end smoke test of the ``alife`` CLI through ``cli.main(argv)`` (no subprocesses). WP5.

One 200-step ``tiny`` run (f32 weights, as ``tiny_cfg``) is shared by the module; tests that change a run
directory work on a copy. Resume, render and fork read back what the run wrote.
"""

from __future__ import annotations

import inspect
import json
import shutil
import struct
import zlib
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from alife import cli
from alife.config import load_config
from alife.runio import RunDir, checkpoint_metadata, load_checkpoint
from alife.state import state_digest, to_numpy

F32 = ["--set", "brain.weight_dtype=float32"]
# CPU MLX matmul is slow (≈ 0.2 s for the default 256-row weight-diversity Gram matrix); 64 rows keep the
# smoke run's 20 heavy-metric intervals cheap. The sample size is an observer key (§17 RunConfig).
SAMPLE = ["--set", "run.diversity_sample=64"]
PERF_KEYS = {"wall_s", "steps_per_s", "peak_mem_gb"}


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("cli") / "run"
    code = cli.main(
        ["run", "--preset", "tiny", *F32, *SAMPLE, "--steps", "200", "--out", str(out), "--quiet"]
    )
    assert code == 0
    return out


def _digest(rd: RunDir, step: int) -> str:
    return state_digest(load_checkpoint(rd.checkpoint_at(step), rd.config()))


def _events(rd: RunDir) -> dict[str, dict[str, np.ndarray]]:
    return {kind: rd.read_events(kind) for kind in ("births", "deaths")}


def _png_size(path: Path) -> tuple[int, int]:
    """(W, H) of a PNG, after checking the signature and the IHDR CRC."""
    raw = path.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    length, kind = struct.unpack(">I4s", raw[8:16])
    assert kind == b"IHDR" and length == 13
    body = raw[16 : 16 + length]
    (crc,) = struct.unpack(">I", raw[16 + length : 20 + length])
    assert crc == zlib.crc32(kind + body)
    w, h = struct.unpack(">II", body[:8])
    return w, h


def test_run_writes_a_complete_run_directory(run_dir: Path) -> None:
    rd = RunDir.open(run_dir)
    cfg = rd.config()
    assert cfg == load_config(
        preset="tiny", overrides=["brain.weight_dtype=float32", "run.diversity_sample=64", "run.steps=200"]
    )
    meta = rd.meta()
    assert meta["status"] == "done" and meta["start_step"] == 0 and meta["parent"] is None
    rows = rd.metrics_rows()
    assert [r["step"] for r in rows] == list(range(10, 201, 10))
    assert all(r["steps"] == 10 and r["ledger_error"] == 0 and r["diag"] & 0xFFFF == 0 for r in rows)
    assert "carrying_capacity_est" in rows[0] and all("carrying_capacity_est" not in r for r in rows[1:])
    assert all("bodies_n" in r for r in rows)  # tiny: heavy metrics every 10 steps
    # keep_checkpoints = 3 plus step 0; LATEST names the last one
    assert [st for st, _ in rd.checkpoints()] == [0, 100, 150, 200]
    assert (rd.checkpoint_dir / "LATEST").read_text().strip() == "step_0000000200.safetensors"
    for kind in ("births", "deaths"):
        assert [(a, b) for a, b, _ in rd.event_files(kind)] == [(0, 50), (50, 100), (100, 150), (150, 200)]
    # the events agree with the counters and with the final state
    ev = _events(rd)
    assert ev["births"]["uid"].size == sum(r["births"] + r["immigrants"] for r in rows)
    deaths = sum(
        r["deaths_starved"]
        + r["deaths_killed"]
        + r["deaths_aged"]
        + r["deaths_catastrophe"]
        - r["deaths_unlogged"]
        for r in rows
    )
    assert ev["deaths"]["uid"].size == deaths
    s = load_checkpoint(rd.checkpoint_at(200), cfg)
    assert int(to_numpy(s.next_uid)) == cfg.population.n_init + ev["births"]["uid"].size
    assert int(to_numpy(s.total)) == rows[-1]["E_total"]
    np.testing.assert_array_equal(
        np.sort(ev["births"]["uid"]), np.arange(256, 256 + ev["births"]["uid"].size)
    )


def test_inspect_check_json(run_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    capsys.readouterr()
    assert cli.main(["inspect", str(run_dir), "--check", "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["ledger_error"] == 0 and info["check"] == {"ok": True, "problems": []}
    assert info["step"] == 200 and info["checkpoints"] == [0, 100, 150, 200]
    assert info["last_metrics"]["step"] == 200 and info["carrying_capacity_est"] > 0
    assert info["state"]["bodies_converged"] is True and info["n_alive"] > 0

    assert cli.main(["inspect", str(run_dir), "--checkpoint", "100"]) == 0
    text = capsys.readouterr().out
    assert "checkpoint step 100" in text and "ledger OK" in text and "carrying capacity" in text
    assert cli.main(["inspect", str(run_dir), "--checkpoint", "7"]) == 4


def test_inspect_agent_lineage(run_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rd = RunDir.open(run_dir)
    births = rd.read_events("births")
    child = births["parent_uid"] >= 256  # a grandchild of a founder, if any; else any child
    uid = int(births["uid"][np.flatnonzero(child)[0]] if child.any() else births["uid"][0])
    capsys.readouterr()
    assert cli.main(["inspect", str(run_dir), "--json", "--agent", str(uid)]) == 0
    agent = json.loads(capsys.readouterr().out)["agent"]
    assert agent["uid"] == uid and agent["ancestors_complete"] is True
    parents = dict(zip(births["uid"].tolist(), births["parent_uid"].tolist(), strict=True))
    chain, p = [], parents[uid]
    while p >= 0:
        chain.append(p)
        p = parents.get(p, -1)
    assert agent["ancestors"] == chain and chain[-1] < 256
    assert agent["birth"]["parent_uid"] == chain[0]
    if agent["alive"]:
        assert agent["fields"]["uid"] == uid and agent["fields"]["parent_uid"] == chain[0]
    else:
        assert agent["death"]["cause"] in ("STARVED", "KILLED", "AGED", "CATASTROPHE")
    assert cli.main(["inspect", str(run_dir), "--agent", "999999"]) == 1


def test_resume_truncates_and_is_bitwise(run_dir: Path, tmp_path: Path) -> None:
    rd0 = RunDir.open(run_dir)
    ref_rows, ref_events, ref_digest = rd0.metrics_rows(), _events(rd0), _digest(rd0, 200)
    copy = tmp_path / "run"
    shutil.copytree(run_dir, copy)

    # Resume an abandoned continuation from step 100: later logs and checkpoints are dropped first.
    assert cli.main(["resume", str(copy), "--checkpoint", "100", "--steps", "30", "--quiet"]) == 0
    rd = RunDir.open(copy)
    assert [r["step"] for r in rd.metrics_rows()] == list(range(10, 131, 10))
    assert [st for st, _ in rd.checkpoints()] == [0, 100, 130]
    for kind in ("births", "deaths"):
        assert [(a, b) for a, b, _ in rd.event_files(kind)] == [(0, 50), (50, 100), (100, 130)]
    assert rd.meta()["status"] == "done"

    # The default target is start_step + run.steps (200); the result equals the uninterrupted run.
    assert cli.main(["resume", str(copy), "--quiet"]) == 0
    assert [st for st, _ in rd.checkpoints()][-1] == 200
    assert _digest(rd, 200) == ref_digest
    rows = rd.metrics_rows()
    assert [r["step"] for r in rows] == [r["step"] for r in ref_rows]
    for a, b in zip(rows, ref_rows, strict=True):
        assert {k: v for k, v in a.items() if k not in PERF_KEYS} == {
            k: v for k, v in b.items() if k not in PERF_KEYS
        }
    ev = _events(rd)
    for kind in ev:
        for col in ev[kind]:
            np.testing.assert_array_equal(ev[kind][col], ref_events[kind][col], err_msg=f"{kind}.{col}")

    # --until and accepted run.* overrides (written back to config.json)
    assert (
        cli.main(["resume", str(copy), "--until", "205", "--set", "run.keep_checkpoints=1", "--quiet"]) == 0
    )
    assert rd.config().run.keep_checkpoints == 1
    assert [st for st, _ in rd.checkpoints()] == [0, 205]
    assert cli.main(["resume", str(copy), "--until", "10"]) == 2  # before the checkpoint
    assert cli.main(["resume", str(copy), "--checkpoint", "150"]) == 4  # pruned


def test_render_png_and_sequence(run_dir: Path, tmp_path: Path) -> None:
    png = tmp_path / "f.png"
    assert cli.main(["render", str(run_dir), "--steps", "0", "--out", str(png)]) == 0
    assert _png_size(png) == (1024, 1024)

    frames = tmp_path / "frames"
    args = ["render", str(run_dir), "--steps", "20", "--every", "5", "--out", f"{frames}/", "--size", "96x64"]
    assert cli.main(args) == 0
    names = sorted(p.name for p in frames.iterdir())
    assert names == [f"frame_{i:06d}.png" for i in range(5)]  # the checkpoint frame plus one per 5 steps
    assert {_png_size(frames / n) for n in names} == {(96, 64)}

    # default output goes to RUN/media; the run itself is untouched
    before = sorted(p.relative_to(run_dir) for p in run_dir.rglob("*") if "media" not in p.parts)
    assert (
        cli.main(["render", str(run_dir), "--checkpoint", "150", "--size", "64x64", "--view", "64,64,2"]) == 0
    )
    assert _png_size(run_dir / "media" / "frame_0000000150.png") == (64, 64)
    assert sorted(p.relative_to(run_dir) for p in run_dir.rglob("*") if "media" not in p.parts) == before
    assert cli.main(["render", str(run_dir), "--steps", "2", "--out", str(tmp_path / "x.png")]) == 2
    assert cli.main(["render", str(run_dir), "--layers", "agents,walls"]) == 2


def test_fork(run_dir: Path, tmp_path: Path) -> None:
    fork = tmp_path / "fork"
    args = ["run", "--init-from", str(run_dir), "--set", "combat.attack_bonded=true", "--out", str(fork)]
    assert cli.main([*args, "--steps", "20", "--quiet"]) == 0
    rd = RunDir.open(fork)
    cfg = rd.config()
    assert cfg.combat.attack_bonded is True and cfg.brain.weight_dtype == "float32"
    meta = rd.meta()
    assert meta["parent"] == f"{run_dir}:200" and meta["start_step"] == 200 and meta["status"] == "done"
    assert [st for st, _ in rd.checkpoints()] == [200, 220]
    src = load_checkpoint(RunDir.open(run_dir).checkpoint_at(200), cfg, check_hash=False)
    start = load_checkpoint(rd.checkpoint_at(200), cfg)
    for name in ("step", "next_uid", "uid", "key", "W1", "pend_slot", "resource"):
        np.testing.assert_array_equal(
            to_numpy(getattr(start, name)), to_numpy(getattr(src, name)), err_msg=name
        )
    assert not to_numpy(start.ledger_cum).any() and int(to_numpy(start.total0)) == int(to_numpy(src.total))
    assert [r["step"] for r in rd.metrics_rows()] == [210, 220]
    assert (
        checkpoint_metadata(rd.checkpoint_at(220))["config_hash"]
        != checkpoint_metadata(RunDir.open(run_dir).checkpoint_at(200))["config_hash"]
    )
    assert cli.main([*args, "--preset", "tiny"]) == 2  # the source state's config is the base


def test_fork_structure_change_rebases_total(run_dir: Path, tmp_path: Path) -> None:
    """§18 Reset: a fork that changes ``energy.structure`` rebases TOTAL by (S_new − S_old)·n_alive."""
    fork = tmp_path / "fork_s"
    args = ["run", "--init-from", str(run_dir), "--set", "energy.structure=5000", "--out", str(fork)]
    assert cli.main([*args, "--steps", "10", "--quiet"]) == 0
    rd = RunDir.open(fork)
    cfg = rd.config()
    src_cfg = RunDir.open(run_dir).config()
    src = load_checkpoint(RunDir.open(run_dir).checkpoint_at(200), src_cfg)
    start = load_checkpoint(rd.checkpoint_at(200), cfg)
    n_alive = int(to_numpy(src.alive).sum())
    delta = n_alive * (cfg.energy.structure - src_cfg.energy.structure)
    assert n_alive > 0 and delta != 0
    assert int(to_numpy(start.total)) == int(to_numpy(src.total)) + delta
    assert int(to_numpy(start.total0)) == int(to_numpy(start.total))
    assert rd.meta()["status"] == "done"


def test_view_and_bench_delegate(run_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``view`` and ``bench`` commands call WP6's public functions with the parsed arguments."""
    from alife import bench, viewer

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(viewer, "run_viewer", lambda cfg, state, **kw: calls.append({"cfg": cfg, **kw}))
    assert (
        cli.main(["view", str(run_dir), "--checkpoint", "150", "--steps-per-frame", "3", "--size", "300x200"])
        == 0
    )
    assert calls[-1]["rundir"].path == run_dir and calls[-1]["size"] == (300, 200)
    assert calls[-1]["steps_per_frame"] == 3
    assert cli.main(["view", "--preset", "tiny", "--set", "run.seed=4"]) == 0
    assert calls[-1]["rundir"] is None and calls[-1]["cfg"].run.seed == 4
    assert cli.main(["view", str(run_dir), "--preset", "tiny"]) == 2

    def no_pygame(cfg: Any, state: Any, **kw: Any) -> None:
        raise ImportError("install the 'viewer' extra")

    monkeypatch.setattr(viewer, "run_viewer", no_pygame)
    assert cli.main(["view", "--preset", "tiny"]) == 1

    seen: dict[str, Any] = {}

    def fake_bench(cfg: Any, **kw: Any) -> dict[str, Any]:
        seen.update(kw, n=cfg.population.n_max, hidden=cfg.brain.hidden)
        return {"ms_per_step": np.float64(1.5), "steps": kw["steps"]}

    monkeypatch.setattr(bench, "run_bench", fake_bench)
    out = tmp_path / "b" / "bench.json"
    argv = ["bench", "--preset", "tiny", "--overlay", "fast", "--steps", "5", "--warmup", "2", "--stages"]
    assert cli.main([*argv, "--json", str(out)]) == 0
    assert json.loads(out.read_text()) == {"ms_per_step": 1.5, "steps": 5}
    assert seen == {
        "steps": 5,
        "warmup": 2,
        "stages": True,
        "determinism": False,
        "brain_variants": False,
        "n": 1024,
        "hidden": 24,
    }
    monkeypatch.setattr(bench, "run_bench", lambda cfg, **kw: {"determinism": {"match": False}})
    assert cli.main(["bench", "--preset", "tiny", "--determinism"]) == 1


def test_bench_smoke(tmp_path: Path) -> None:
    from alife import bench

    if "raise NotImplementedError" in inspect.getsource(bench.run_bench):
        pytest.skip("bench.run_bench (WP6) is not implemented yet")
    out = tmp_path / "bench.json"
    argv = ["bench", "--preset", "tiny", *F32, "--steps", "5", "--warmup", "2", "--json", str(out)]
    assert cli.main(argv) == 0
    assert isinstance(json.loads(out.read_text()), dict)


def test_usage_and_config_errors(tmp_path: Path) -> None:
    assert cli.main([]) == 2
    assert cli.main(["run", "--preset", "nope"]) == 2
    assert cli.main(["run", "--preset", "tiny", "--set", "energy.nope=1", "--out", str(tmp_path / "a")]) == 2
    assert not (tmp_path / "a").exists()
    assert cli.main(["run", "--preset", "tiny", "--metrics-every", "7", "--out", str(tmp_path / "b")]) == 2
    assert cli.main(["resume", str(tmp_path / "missing")]) == 1
    assert cli.main(["--help"]) == 0


# --- the director flags (build step 2) ---

DIR_SET = [
    "--set",
    "director.rule_every=100",
    "--set",
    "run.metrics_every=10",
    "--set",
    "run.heavy_metrics_every=100",
]


def test_director_live_run_writes_its_files_and_replays(tmp_path: Path) -> None:
    out, rep = tmp_path / "live", tmp_path / "rep"
    base = ["--preset", "tiny", *F32, *SAMPLE, *DIR_SET, "--quiet"]
    assert cli.main(["run", *base, "--director", "live", "--steps", "300", "--out", str(out)]) == 0
    rd = RunDir.open(out)
    cfg = rd.config()
    assert (cfg.run.director, cfg.director.enabled) == ("live", True)  # live implies an enabled director
    head = json.loads((out / "director.jsonl").read_text().splitlines()[0])
    assert head["kind"] == "header" and head["rules_version"] == 1 and head["rule_every"] == 100
    assert json.loads((out / "director_archive.json").read_text())["as_of_step"] == 300
    assert "env" in rd.metrics_rows()[0] and cli.main(["inspect", str(out), "--check"]) == 0
    argv = [
        "run",
        "--init-from",
        f"{out}:0",
        "--director",
        "replay",
        "--replay-log",
        str(out / "director.jsonl"),
    ]
    assert cli.main([*argv, "--steps", "300", "--quiet", "--out", str(rep)]) == 0
    assert _digest(RunDir.open(rep), 300) == _digest(rd, 300)
    assert RunDir.open(rep).config().run.director == "replay"
    assert cli.main(["resume", str(rep), "--steps", "0", "--quiet"]) == 0  # inherits replay; applies the log


def test_director_off_and_default_runs_have_no_director_files(tmp_path: Path) -> None:
    for name, extra in (("default", []), ("off", ["--director", "off"])):
        out = tmp_path / name
        assert (
            cli.main(
                [
                    "run",
                    "--preset",
                    "tiny",
                    *F32,
                    *SAMPLE,
                    "--steps",
                    "20",
                    "--quiet",
                    *extra,
                    "--out",
                    str(out),
                ]
            )
            == 0
        )
        assert not any(
            (out / f).exists() for f in ("director.jsonl", "journal.jsonl", "director_archive.json")
        )
        assert "env" not in RunDir.open(out).metrics_rows()[0]  # the env summary is only for enabled runs
        assert RunDir.open(out).config().director.enabled is False
    # director.enabled with the mode off: the environment is inert and nothing is decided
    out = tmp_path / "enabled_off"
    argv = [
        "run",
        "--preset",
        "tiny",
        *F32,
        *SAMPLE,
        "--set",
        "director.enabled=true",
        "--steps",
        "20",
        "--quiet",
    ]
    assert cli.main([*argv, "--out", str(out)]) == 0
    assert not (out / "director.jsonl").exists() and "env" in RunDir.open(out).metrics_rows()[0]


def test_replay_needs_a_state_to_continue(tmp_path: Path) -> None:
    assert (
        cli.main(["run", "--preset", "tiny", "--director", "replay", "--quiet", "--out", str(tmp_path / "x")])
        == 2
    )
    assert (
        cli.main(
            [
                "run",
                "--preset",
                "tiny",
                "--set",
                "run.director=replay",
                "--quiet",
                "--out",
                str(tmp_path / "y"),
            ]
        )
        == 2
    )
    assert not (tmp_path / "x").exists() and not (tmp_path / "y").exists()


def test_fork_resets_the_environment_and_does_not_inherit_the_director(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from alife.state import default_env

    src = tmp_path / "src"
    assert (
        cli.main(
            [
                "run",
                "--preset",
                "tiny",
                *F32,
                *SAMPLE,
                *DIR_SET,
                "--director",
                "live",
                "--steps",
                "100",
                "--quiet",
                "--out",
                str(src),
            ]
        )
        == 0
    )
    # make the source's last checkpoint carry a non-inert environment, as after a director action
    srd, scfg = RunDir.open(src), RunDir.open(src).config()
    s = load_checkpoint(srd.checkpoint_at(100), scfg)
    srd.save_checkpoint(
        s._replace(env_regen_mult=mx.array(1.7, mx.float32), env_imm_request=mx.array(9, mx.int32)), scfg
    )
    capsys.readouterr()
    out = tmp_path / "fork"
    assert cli.main(["run", "--init-from", str(src), "--steps", "10", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert text.count("environment levers were reset") == 1  # one note line
    frd = RunDir.open(out)
    fcfg = frd.config()
    assert fcfg.run.director == "off" and fcfg.director.enabled  # the mode is not inherited, the lever is
    assert not (out / "director.jsonl").exists()  # and the source's director history does not carry over
    start = load_checkpoint(frd.checkpoint_at(100), fcfg)
    for name, want in default_env(fcfg).items():
        np.testing.assert_array_equal(to_numpy(getattr(start, name)), to_numpy(want), err_msg=name)
    # the new config's seasons become the inert environment (the source's were off)
    seasons = tmp_path / "seasons"
    argv = [
        "run",
        "--init-from",
        str(src),
        "--set",
        "fields.season_period=40",
        "--set",
        "fields.season_amplitude=0.5",
    ]
    assert cli.main([*argv, "--steps", "5", "--quiet", "--out", str(seasons)]) == 0
    sstart = load_checkpoint(RunDir.open(seasons).checkpoint_at(100), RunDir.open(seasons).config())
    assert int(to_numpy(sstart.env_season_period)) == 40 and float(
        to_numpy(sstart.env_season_amp)
    ) == np.float32(0.5)
    # a clean source needs no reset and no note; --director live on a fork starts a fresh log
    clean = tmp_path / "clean"
    assert (
        cli.main(
            [
                "run",
                "--init-from",
                f"{src}:0",
                "--director",
                "live",
                *DIR_SET,
                "--steps",
                "100",
                "--out",
                str(clean),
            ]
        )
        == 0
    )
    assert "were reset" not in capsys.readouterr().out
    heads = [json.loads(x) for x in (clean / "director.jsonl").read_text().splitlines() if '"header"' in x]
    assert [h["step"] for h in heads] == [0] and RunDir.open(clean).config().run.director == "live"


def test_view_and_render_help_say_they_ignore_the_director(capsys: pytest.CaptureFixture[str]) -> None:
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction")
    helps = {c.dest: c.help for c in sub._choices_actions}
    assert "without the director" in helps["view"] and "director" in helps["render"]
