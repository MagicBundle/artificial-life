"""§22.4 resume, run-key restrictions, observer purity, version guards and fork semantics (WP5)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from alife import cli, layout, rng, runner
from alife.config import Config, apply_overrides, config_hash
from alife.invariants import full_check
from alife.render import render_frame
from alife.rng import InitStream, init_keys
from alife.runio import CheckpointError, RunDir, load_checkpoint, write_state_file
from alife.state import Diag, State, init_state, state_digest, to_numpy
from alife.step import make_step


def _bare(cfg: Config, steps: int, s: State | None = None) -> State:
    """``steps`` compiled steps with no observer at all."""
    fn = make_step(cfg)
    s = init_state(cfg) if s is None else s
    for _ in range(steps):
        s, out = fn(s)
        mx.eval(s, out)
    return s


@pytest.fixture(scope="module")
def ref100(tiny_cfg: Config) -> tuple[str, str]:
    """Digests of the uninterrupted ``tiny_cfg`` trajectory at steps 50 and 100."""
    s50 = _bare(tiny_cfg, 50)
    d50 = state_digest(s50)
    return d50, state_digest(_bare(tiny_cfg, 50, s50))


def test_resume_is_bitwise(tiny_cfg: Config, ref100: tuple[str, str], tmp_path: Path) -> None:
    """50 steps → save_checkpoint → load_checkpoint → 50 steps == 100 uninterrupted steps, bitwise."""
    rd = RunDir.create(tmp_path / "run", tiny_cfg)
    res = runner.run(rd, tiny_cfg, init_state(tiny_cfg), 50, quiet=True)
    assert (res.status, res.exit_code) == ("done", runner.ExitCode.OK)
    assert state_digest(res.state) == ref100[0]
    del res
    s = load_checkpoint(rd.checkpoint_at(50), tiny_cfg)
    assert state_digest(s) == ref100[0]
    res = runner.run(rd, tiny_cfg, s, 50, quiet=True)
    assert state_digest(res.state) == ref100[1]
    assert state_digest(load_checkpoint(rd.latest_checkpoint(), tiny_cfg)) == ref100[1]
    rows = rd.metrics_rows()
    assert [r["step"] for r in rows] == list(range(10, 101, 10)) and {r["ledger_error"] for r in rows} == {0}


@pytest.fixture(scope="module")
def small_run(tiny_cfg: Config, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A ``tiny_cfg`` run directory stopped at step 30 (checkpoints at 0 and 30)."""
    rd = RunDir.create(tmp_path_factory.mktemp("purity") / "run", tiny_cfg)
    rd.update_meta(start_step=0)
    runner.run(rd, tiny_cfg, init_state(tiny_cfg), 30, quiet=True)
    return rd.path


@pytest.mark.parametrize(
    "override",
    [
        "run.compile=false",
        "run.runtime_checks=off",
        "run.seed=5",
        "energy.basal_cost=7",
        "combat.attack_bonded=true",
        "run.nope=1",
        "director.enabled=true",
    ],
)
def test_resume_rejects_non_observer_keys(small_run: Path, override: str) -> None:
    before = {p: p.read_bytes() for p in (small_run / "config.json", small_run / "metrics.jsonl")}
    assert cli.main(["resume", str(small_run), "--set", override, "--checkpoint", "0"]) == 2
    assert {p: p.read_bytes() for p in before} == before  # nothing was truncated or rewritten
    assert [st for st, _ in RunDir.open(small_run).checkpoints()] == [0, 30]


def test_resume_accepts_cadence_keys(small_run: Path, tmp_path: Path) -> None:
    copy = tmp_path / "run"
    shutil.copytree(small_run, copy)
    argv = [
        "resume",
        str(copy),
        "--steps",
        "0",
        "--set",
        "run.metrics_every=5",
        "--set",
        "run.diversity_sample=8",
    ]
    assert cli.main(argv) == 0
    cfg = RunDir.open(copy).config()
    assert (cfg.run.metrics_every, cfg.run.diversity_sample) == (5, 8)
    assert json.loads((copy / "config.json").read_text())["run"]["metrics_every"] == 5
    # an accepted key whose value fails validation is still a config error
    assert cli.main(["resume", str(copy), "--set", "run.checkpoint_every=7"]) == 2


def test_observer_purity(tiny_cfg_f16: Config, tmp_path: Path) -> None:
    """Metrics and heavy metrics every step, a frame every step and checkpoints every 10 change nothing."""
    steps = 30
    cfg = apply_overrides(
        tiny_cfg_f16,
        [
            "run.metrics_every=1",
            "run.heavy_metrics_every=1",
            "run.checkpoint_every=10",
            "run.diversity_sample=64",  # CPU MLX matmul is slow; the sample size is an observer key
        ],
    )
    assert config_hash(cfg) == config_hash(tiny_cfg_f16)
    frames: list[int] = []

    def observe(s: State, row: dict) -> None:
        img = render_frame(s, cfg, size=(128, 128))
        frames.append(int(to_numpy(img).sum() > 0))

    rd = RunDir.create(tmp_path / "run", cfg)
    res = runner.run(rd, cfg, init_state(cfg), steps, quiet=True, on_interval=observe)
    observed = state_digest(res.state)
    del res
    assert len(frames) == steps and all(frames)
    assert len(rd.metrics_rows()) == steps and all("weight_diversity" in r for r in rd.metrics_rows())
    assert [st for st, _ in rd.checkpoints()] == [0, 10, 20, 30]
    assert observed == state_digest(_bare(tiny_cfg_f16, steps))


def _with_versions(
    src: Path, dst: Path, cfg: Config, monkeypatch: pytest.MonkeyPatch, **versions: int
) -> None:
    """Rewrite the state file ``src`` as ``dst`` while the code's layout versions are ``versions``."""
    s = load_checkpoint(src, cfg)
    with monkeypatch.context() as m:
        for mod, name in ((rng, "RNG_LAYOUT_VERSION"), (layout, "GENOME_LAYOUT_VERSION")):
            if name in versions:
                m.setattr(mod, name, versions[name])
        write_state_file(dst, s, cfg)


@pytest.mark.parametrize("name", ["RNG_LAYOUT_VERSION", "GENOME_LAYOUT_VERSION"])
def test_version_guards(
    name: str, small_run: Path, tiny_cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copy = tmp_path / "run"
    shutil.copytree(small_run, copy)
    rd = RunDir.open(copy)
    current = getattr(rng if name.startswith("RNG") else layout, name)
    ckpt = rd.checkpoint_at(30)
    _with_versions(ckpt, ckpt, tiny_cfg, monkeypatch, **{name: current + 1})
    with pytest.raises(CheckpointError, match=name.lower()):
        load_checkpoint(ckpt, tiny_cfg)
    assert cli.main(["resume", str(copy), "--steps", "1"]) == 4
    assert cli.main(["inspect", str(copy)]) == 4
    assert cli.main(["run", "--init-from", str(copy), "--out", str(tmp_path / "fork")]) == 4
    assert [st for st, _ in rd.checkpoints()] == [0, 30]  # the refused resume touched nothing


def _fork_args(src: str, out: Path, *extra: str) -> list[str]:
    return ["run", "--init-from", src, "--out", str(out), "--steps", "0", "--quiet", *extra]


def test_fork_semantics(small_run: Path, tiny_cfg: Config, tmp_path: Path) -> None:
    src = load_checkpoint(RunDir.open(small_run).checkpoint_at(30), tiny_cfg)
    # A standalone state file (as a viewer snapshot) with a sticky warning bit and a non-zero ledger.
    snap = tmp_path / "snap.safetensors"
    write_state_file(snap, src._replace(diag=mx.array(int(Diag.VCLIP_WARN), dtype=mx.int32)), tiny_cfg)

    def started(out: Path) -> State:
        rd = RunDir.open(out)
        assert [st for st, _ in rd.checkpoints()] == [30] and rd.meta()["start_step"] == 30
        return load_checkpoint(rd.checkpoint_at(30), rd.config())

    assert cli.main(_fork_args(str(snap), tmp_path / "a", "--set", "combat.attack_bonded=true")) == 0
    a = started(tmp_path / "a")
    for name in ("step", "next_uid", "uid", "parent_uid", "birth_step", "key", "pend_slot", "pend_W1", "W1"):
        np.testing.assert_array_equal(to_numpy(getattr(a, name)), to_numpy(getattr(src, name)), err_msg=name)
    assert int(to_numpy(a.diag)) == 0 and not to_numpy(a.ledger_cum).any()
    assert int(to_numpy(a.total0)) == int(to_numpy(a.total)) == int(to_numpy(src.total))
    assert to_numpy(src.ledger_cum).any()
    assert RunDir.open(tmp_path / "a").meta()["parent"] == str(snap)
    assert full_check(a, RunDir.open(tmp_path / "a").config()) == []

    # --seed reseeds the key (and run.seed) only
    assert cli.main(_fork_args(f"{small_run}:30", tmp_path / "b", "--seed", "7")) == 0
    b = started(tmp_path / "b")
    np.testing.assert_array_equal(to_numpy(b.key), to_numpy(init_keys(7)[InitStream.STATE_KEY]))
    assert RunDir.open(tmp_path / "b").config().run.seed == 7
    assert state_digest(
        b._replace(key=src.key, diag=src.diag, ledger_cum=src.ledger_cum, total0=src.total0)
    ) == (state_digest(src))

    # refused: a changed n_init (exit 4), a state violating the new config (exit 4), --preset (exit 2)
    assert cli.main(_fork_args(str(small_run), tmp_path / "c", "--set", "population.n_init=100")) == 4
    tight = ["--set", "mutation.sigma_init=0.005", "--set", "mutation.sigma_max=0.01"]
    assert cli.main(_fork_args(str(small_run), tmp_path / "d", *tight)) == 4
    shape = ["--set", "brain.hidden=16"]
    assert cli.main(_fork_args(str(small_run), tmp_path / "e", *shape)) == 4
    assert cli.main(_fork_args(str(small_run), tmp_path / "f", "--preset", "tiny")) == 2
    assert not any((tmp_path / x).exists() for x in "cdef")
