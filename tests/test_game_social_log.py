"""§29.10 the social log ``observers/social.jsonl`` at run level: one record per snapshot, appended and
fsynced by the runner at the snapshot step, cut by ``truncate_after`` on resume, so that for any stop step
and any checkpoint resumed from the resumed log equals the uninterrupted one byte for byte (and so do the
sidecars and the rows' ``observers`` blocks at the steps both runs have).

The cadence is ``snapshot_every = 15`` with tiny's ``metrics_every = 10``: snapshots fall on row steps (30,
60) and between rows (15, 45), and stops fall on, just after and between snapshots. Logs are compared byte
for byte. The host model of the write order and the cut (``protocol_model.py --runio`` of the G2 spec) is
the exhaustive check of the protocol; these are its tests on real runs."""

from __future__ import annotations

import dataclasses
import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from alife import cli, runio
from alife import runner as runner_mod
from alife.config import Config, apply_overrides
from alife.game.observer import Observer
from alife.game.social import SOCIAL_V1
from alife.runio import RunDir
from alife.state import State, init_state, to_numpy
from alife.step import make_step
from game_helpers import planted_state, social_every

END = 60
SOC15 = dataclasses.replace(SOCIAL_V1, snapshot_every=15)


@pytest.fixture(autouse=True)
def every15(monkeypatch: pytest.MonkeyPatch) -> None:
    social_every(monkeypatch, 15)


def _cfg(cfg: Config, *extra: str) -> Config:
    return apply_overrides(cfg, ["game.observe=true", "run.checkpoint_every=30", *extra])


def _run(cfg: Config, path: Path, steps: int, state: State | None = None) -> RunDir:
    rd = RunDir.create(path, cfg)
    rd.update_meta(start_step=0)
    res = runner_mod.run(rd, cfg, init_state(cfg) if state is None else state, steps, quiet=True)
    assert (res.status, res.exit_code) == ("done", runner_mod.ExitCode.OK)
    return rd


def _resume(path: Path, *args: str) -> None:
    assert cli.main(["resume", str(path), *args, "--quiet"]) == 0


def _same(a: dict[str, np.ndarray] | None, b: dict[str, np.ndarray] | None) -> bool:
    return (
        a is not None
        and b is not None
        and a.keys() == b.keys()
        and all(a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]) for k in a)
    )


def _log(rd: RunDir) -> bytes:
    return rd.social_path.read_bytes()


@pytest.fixture(scope="module")
def references(
    tiny_cfg: Config, tiny3d_cfg: Config, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[dict[str, RunDir]]:
    """The uninterrupted 60-step runs, 2D and 3D (a module-scoped fixture sets the cadence by hand)."""
    out = {}
    with pytest.MonkeyPatch.context() as mp:
        social_every(mp, 15)
        root = tmp_path_factory.mktemp("social_log")
        for name, base in (("2d", tiny_cfg), ("3d", tiny3d_cfg)):
            out[name] = _run(_cfg(base), root / name, END)
    yield out


def test_the_log_of_a_run(references: dict[str, RunDir]) -> None:
    """One record per snapshot after the origin, in step order, ``step`` first, written by the runner."""
    for name, rd in references.items():
        records = rd.social_records()
        assert [r["step"] for r in records] == [15, 30, 45, 60], name
        assert all(next(iter(r)) == "step" and r["origin"] == 0 for r in records)
        assert [r["window_start"] for r in records] == [0, 15, 30, 45]
        assert all("alife_version" not in r and "wall_s" not in r for r in records)  # no host field
        lines = _log(rd).decode("utf-8").splitlines()
        assert len(lines) == 4 and all(
            runio._dumps(json.loads(line)) == line for line in lines
        )  # byte-stable
        assert rd.observer_files() and rd.social_path.parent == rd.observers_dir


def test_social_log_cut_at_every_step(references: dict[str, RunDir], tmp_path: Path) -> None:
    """The log, copied alone into an empty directory and cut by ``truncate_after(s)`` for every s in 0..60,
    holds exactly the reference's lines with step <= s, byte-equal to the original lines (they are
    re-serialised), with and without ``keep_director``."""
    ref = references["2d"]
    lines = _log(ref).splitlines(keepends=True)
    steps = [json.loads(line)["step"] for line in lines]
    assert steps == [15, 30, 45, 60]
    for s in range(END + 1):
        for keep_director in (False, True):
            d = tmp_path / f"cut_{s}_{keep_director}"
            (d / "observers").mkdir(parents=True)
            shutil.copyfile(ref.social_path, d / "observers" / "social.jsonl")
            RunDir(d).truncate_after(s, keep_director=keep_director)
            want = b"".join(line for line, st in zip(lines, steps, strict=True) if st <= s)
            assert (d / "observers" / "social.jsonl").read_bytes() == want, (s, keep_director)
            shutil.rmtree(d)


STOPS = [
    *[pytest.param("2d", s, id=f"2d-stop-{s}") for s in (1, 14, 15, 16, 29, 30, 31, 44, 45, 46, 59)],
    *[pytest.param("3d", s, id=f"3d-stop-{s}") for s in (15, 16, 45)],
]


@pytest.mark.parametrize(("dim", "stop"), STOPS)
def test_social_log_resume_equality(
    dim: str,
    stop: int,
    references: dict[str, RunDir],
    tiny_cfg: Config,
    tiny3d_cfg: Config,
    tmp_path: Path,
) -> None:
    """A run stopped at ``stop`` (on, just after and between snapshots and rows) and resumed as ``alife
    resume`` does ends with the reference's log, sidecar and ``observers`` blocks; for stops past 30, a
    second resume from the checkpoint at 30 (an abandoned continuation) gives them again."""
    ref = references[dim]
    cfg = _cfg(tiny3d_cfg if dim == "3d" else tiny_cfg)
    split = _run(cfg, tmp_path / "split", stop)
    assert [r["step"] for r in split.social_records()] == list(range(15, stop + 1, 15))
    _resume(split.path, "--until", str(END))
    blocks = {r["step"]: r["observers"] for r in ref.metrics_rows()}
    for second in (False, True) if stop > 30 else (False,):
        if second:
            _resume(split.path, "--checkpoint", "30", "--until", str(END))
        split = RunDir.open(split.path)
        assert _log(split) == _log(ref), (dim, stop, second)
        assert _same(split.read_observers(END), ref.read_observers(END))
        resumed = {
            r["step"]: r["observers"] for r in split.metrics_rows() if r["step"] > (30 if second else stop)
        }
        assert resumed and all(json.dumps(blocks[st]) == json.dumps(block) for st, block in resumed.items())


def _fragment(rec: dict, n: int = 40) -> str:
    """The first ``n`` characters of a record's line: a write that a crash cut short."""
    return runio._dumps(rec)[:n]


def test_social_log_after_a_crash(references: dict[str, RunDir], tiny_cfg: Config, tmp_path: Path) -> None:
    """A run to 50 (checkpoints at 0, 30, 50; records 15, 30, 45), then a torn fragment of the next record is
    appended (a crash mid-append). Resumed from the checkpoint at 30: the records of 45 and the torn tail are
    cut, 45 and 60 are written again, and the log is the reference's. Resumed from the latest checkpoint (50),
    whose complete records are all at or before it: the rewrite removes only the tail, and the log is the
    reference's too."""
    ref = references["2d"]
    next_record = ref.social_records()[3]  # the record of step 60
    cfg = _cfg(tiny_cfg)
    base = _run(cfg, tmp_path / "base", 50)
    assert [r["step"] for r in base.social_records()] == [15, 30, 45]
    for source in ("30", "latest"):
        d = tmp_path / f"crash_{source}"
        shutil.copytree(base.path, d)
        rd = RunDir.open(d)
        with open(rd.social_path, "a", encoding="utf-8") as f:
            f.write(_fragment(next_record))
        assert [r["step"] for r in rd.social_records()] == [15, 30, 45]  # the torn tail is ignored on read
        _resume(d, *(["--checkpoint", "30"] if source == "30" else []), "--until", str(END))
        assert _log(RunDir.open(d)) == _log(ref), source
        assert _same(RunDir.open(d).read_observers(END), ref.read_observers(END)), source


def test_social_log_ignores_the_row_cadence(
    references: dict[str, RunDir], tiny_cfg: Config, tmp_path: Path
) -> None:
    """Runs with ``metrics_every`` 5, 10 and 30 (the last with ``heavy_metrics_every`` 30) write the same
    log; a run stopped at 25 and resumed with ``--set run.metrics_every=5`` writes it too, and its rows'
    ``observers`` blocks equal the reference's at the row steps both runs have."""
    ref = references["2d"]
    for me, heavy in ((5, 10), (10, 10), (30, 30)):
        rd = _run(
            _cfg(tiny_cfg, f"run.metrics_every={me}", f"run.heavy_metrics_every={heavy}"),
            tmp_path / f"me{me}",
            END,
        )
        assert _log(rd) == _log(ref), me
        assert [r["step"] for r in rd.metrics_rows()] == list(range(me, END + 1, me))
    split = _run(_cfg(tiny_cfg), tmp_path / "split", 25)
    _resume(split.path, "--set", "run.metrics_every=5", "--until", str(END))
    split = RunDir.open(split.path)
    assert _log(split) == _log(ref)
    assert split.config().run.metrics_every == 5
    blocks = {r["step"]: r["observers"] for r in ref.metrics_rows()}
    resumed = {r["step"]: r["observers"] for r in split.metrics_rows() if r["step"] > 25}
    common = sorted(set(resumed) & set(blocks))
    assert common == [30, 40, 50, 60] and sorted(resumed) == [30, 35, 40, 45, 50, 55, 60]
    assert all(json.dumps(blocks[st]) == json.dumps(resumed[st]) for st in common)


def test_replay_resume_regenerates_the_social_log(tmp_path: Path) -> None:
    """A live-director observing run to 60, then ``alife resume --director replay --checkpoint 30``: the
    director's files are byte-unchanged (a replay never touches the log it replays) and the social log, which
    belongs to the trajectory and is cut like the sidecars, equals the original's."""
    argv = [
        "run", "--preset", "tiny",
        "--set", "brain.weight_dtype=float32",
        "--set", "population.n_init=24",
        "--set", "director.rule_every=20",
        "--set", "run.heavy_metrics_every=20",
        "--set", "run.diversity_sample=64",
        "--set", "run.checkpoint_every=30",
        "--set", "game.observe=true",
        "--director", "live", "--steps", "60", "--quiet",
    ]  # fmt: skip
    out = tmp_path / "live"
    assert cli.main([*argv, "--out", str(out)]) == 0
    rd = RunDir.open(out)
    assert [r["step"] for r in rd.social_records()] == [15, 30, 45, 60]
    log, director = _log(rd), (out / "director.jsonl").read_bytes()
    assert rd.read_jsonl("director.jsonl")  # the director acted
    sidecar = rd.read_observers(60)
    _resume(out, "--director", "replay", "--checkpoint", "30", "--until", "60")
    again = RunDir.open(out)
    assert (out / "director.jsonl").read_bytes() == director
    assert _log(again) == log and _same(again.read_observers(60), sidecar)


def test_settle_paths_agree_between_the_runner_and_the_lazy_reader(tiny_cfg: Config) -> None:
    """Two observers over the same steps: one takes the records after every eval (the runner's path: a
    snapshot is settled at its own step), the other only at the end (it settles at ``update(t + 1)``, the
    labels tool's path, or at its first read). Equal records in the same order, equal ``arrays()`` after every
    eval (the runner's path compared before its ``take_records``), equal sidecars and heads at the end."""
    for steps in (40, 45):  # 45: the lazy reader still has the snapshot of 45 pending at its final read
        cfg = _cfg(tiny_cfg)
        s = init_state(cfg)
        fn = make_step(cfg, observe=True)
        runner_path = Observer.fresh(s, cfg, social=SOC15)
        lazy = Observer.fresh(s, cfg, social=SOC15)
        taken: list[dict] = []
        for t in range(1, steps + 1):
            s, out = fn(s)
            runner_path.update(s, out.acts, t)
            lazy.update(s, out.acts, t)
            mx.eval(s, out.base(), *runner_path.arrays(), *lazy.arrays())
            a, b = runner_path.arrays(), lazy.arrays()
            assert len(a) == len(b), t
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                assert np.array_equal(to_numpy(x), to_numpy(y)), (t, i)
            taken += runner_path.take_records()
        records = lazy.take_records()
        assert [r["step"] for r in records] == list(range(15, steps + 1, 15)) and len(records) >= 2
        assert records == taken and json.dumps(records) == json.dumps(taken)
        assert lazy.take_records() == [] and runner_path.take_records() == []
        assert _same(runner_path.sidecar(), lazy.sidecar())
        assert json.dumps(runner_path.row_block()) == json.dumps(lazy.row_block())


@pytest.mark.parametrize("method", ["row_block", "social_labels", "territory", "take_records"])
def test_a_reader_settles_a_pending_snapshot(tiny_cfg: Config, method: str) -> None:
    """Every reader sees the settled state: an observer that is only ever read with ``method`` (and never
    told to take its records before the end) answers at every step as the observer that takes its records
    at the snapshot step does, and the read does not consume the records."""
    cfg = _cfg(tiny_cfg)
    s = init_state(cfg)
    fn = make_step(cfg, observe=True)
    ref = Observer.fresh(s, cfg, social=SOC15)
    reader = Observer.fresh(s, cfg, social=SOC15)
    taken: list[dict] = []
    for t in range(1, 32):
        s, out = fn(s)
        ref.update(s, out.acts, t)
        reader.update(s, out.acts, t)
        mx.eval(s, out.base(), *ref.arrays(), *reader.arrays())
        taken += ref.take_records()
        if method == "row_block":
            assert json.dumps(reader.row_block()["social"]) == json.dumps(ref.row_block()["social"]), t
        elif method == "social_labels":
            for x, y in zip(reader.social_labels(), ref.social_labels(), strict=True):
                assert np.array_equal(to_numpy(x), to_numpy(y)), t
        elif method == "territory":
            (m1, s1), (m2, s2) = reader.territory(), ref.territory()
            assert (m1 is None) == (m2 is None) and np.array_equal(s1, s2), t
            if m1 is not None:
                assert np.array_equal(m1.owner, m2.owner) and np.array_equal(m1.border, m2.border), t
        else:
            assert reader.take_records() == [r for r in taken if r["step"] == t], t
    if method != "take_records":
        assert reader.take_records() == taken  # reading did not consume them
    assert [r["step"] for r in taken] == [15, 30]


def test_row_head_matches_the_record(tiny_cfg: Config, tmp_path: Path) -> None:
    """Every row head whose ``snapshot_step`` is after its ``origin`` equals ``{social_version, snapshot_step,
    origin, **summary}`` of the record with that origin and step; the first rows (before the first snapshot)
    carry the origin's head and no record. After a fresh fallback at a snapshot step (a resume at 30 with
    another house rate, so the sidecar's meta does not match) the log keeps the old record of 30 (origin 0),
    the resumed rows' heads have origin 30, the head of ``snapshot_step`` 30 with origin 30 joins no record,
    and the record of 45 has origin 30."""
    cfg = _cfg(tiny_cfg)
    rd = _run(cfg, tmp_path / "run", 30)

    def joined(rows: list[dict], records: list[dict]) -> dict[tuple[int, int], int]:
        by_key = {(r["origin"], r["step"]): r for r in records}
        n = {}
        for row in rows:
            head = row["observers"]["social"]
            rec = by_key.get((head["origin"], head["snapshot_step"]))
            if rec is None:
                n[(head["origin"], head["snapshot_step"])] = 0
                continue
            want = {
                "social_version": rec["social_version"],
                "snapshot_step": rec["step"],
                "origin": rec["origin"],
                **rec["summary"],
            }
            assert head == want and json.dumps(head) == json.dumps(want)
            n[(head["origin"], head["snapshot_step"])] = n.get((head["origin"], head["snapshot_step"]), 0) + 1
        return n

    n = joined(rd.metrics_rows(), rd.social_records())
    assert n == {(0, 0): 0, (0, 15): 1, (0, 30): 1}  # rows 10, 20, 30: the initial head joins no record
    assert [r["observers"]["social"]["snapshot_step"] for r in rd.metrics_rows()] == [0, 15, 30]

    _resume(rd.path, "--set", "game.house_rate=0.25", "--until", str(END))
    rd = RunDir.open(rd.path)
    records = rd.social_records()
    assert [(r["origin"], r["step"]) for r in records] == [(0, 15), (0, 30), (30, 45), (30, 60)]
    assert [r["window_start"] for r in records] == [0, 15, 30, 45]
    rows = rd.metrics_rows()
    resumed = [r for r in rows if r["step"] > 30]
    assert [(r["step"], r["observers"]["social"]["origin"]) for r in resumed] == [
        (40, 30),
        (50, 30),
        (60, 30),
    ]
    n = joined(rows, records)
    assert n[(30, 30)] == 0  # the new observer's initial snapshot: the old record of step 30 has origin 0
    assert n[(0, 30)] == 1 and n[(30, 45)] == 1 and n[(30, 60)] == 1
    assert [r["observers"]["social"]["snapshot_step"] for r in resumed] == [30, 45, 60]
    assert all(not r["summary"]["houses"]["exact"] for r in records if r["origin"] == 30)
    assert all(r["summary"]["houses"]["exact"] for r in records if r["origin"] == 0)


@pytest.mark.parametrize("dim", ["2d", "3d"])
def test_resume_equality_with_kin_groups_and_territory(
    dim: str, tiny_cfg: Config, tiny3d_cfg: Config, tmp_path: Path
) -> None:
    """The unseeded tiny worlds track no kin group (their tags are scattered), so the host state of the kin
    tracker, the dominance map and the similarity subsamples would go through a resume unexercised. A world
    whose tags are planted in three clouds tracks three kin groups: a resume at 37 (between snapshots) and at
    30 (on one) ends with the reference's log, byte for byte, and its sidecar member by member."""
    cfg = _cfg(tiny3d_cfg if dim == "3d" else tiny_cfg)
    ref = _run(cfg, tmp_path / "ref", END, planted_state(cfg))
    records = ref.social_records()
    assert [r["step"] for r in records] == [15, 30, 45, 60]
    assert all(r["summary"]["kin_groups"]["tracked"] == 3 for r in records), [
        r["summary"]["kin_groups"] for r in records
    ]
    assert all(
        r["territory"]["owned_cells"] > 0 and sorted(r["kin_groups"]["slot"]) == [0, 1, 2] for r in records
    )
    assert all(r["kin_groups"]["sim_self"][0] is not None for r in records)
    side = ref.read_observers(END)
    assert side is not None and side["host.kin.system"].size > 0 and side["host.terr.ids"].size == 3
    for stop in (30, 37):
        split = _run(cfg, tmp_path / f"split{stop}", stop, planted_state(cfg))
        _resume(split.path, "--until", str(END))
        split = RunDir.open(split.path)
        assert _log(split) == _log(ref), stop
        assert _same(split.read_observers(END), side), stop
        blocks = {r["step"]: r["observers"] for r in ref.metrics_rows()}
        resumed = {r["step"]: r["observers"] for r in split.metrics_rows() if r["step"] > stop}
        assert all(json.dumps(blocks[st]) == json.dumps(b) for st, b in resumed.items())


@pytest.mark.slow
def test_social_log_resume_from_every_step(tiny_cfg: Config, tmp_path: Path) -> None:
    """A reference with ``metrics_every = checkpoint_every = 1`` leaves a checkpoint and a sidecar at every
    step. For every s in 1..59 a copy holding only the checkpoint at s and its sidecar (with the logs, the
    config and the meta), resumed with ``alife resume --checkpoint s --until 60``, ends with the reference's
    log; each copy is removed after use."""
    cfg = _cfg(
        tiny_cfg,
        "run.metrics_every=1",
        "run.checkpoint_every=1",
        "run.heavy_metrics_every=1",
        "run.keep_checkpoints=100",
    )
    ref = _run(cfg, tmp_path / "ref", END)
    assert [st for st, _ in ref.checkpoints()] == list(range(END + 1))
    want = _log(ref)
    assert len(ref.social_records()) == 4
    for s in range(1, END):
        d = tmp_path / f"copy_{s}"
        (d / "checkpoints").mkdir(parents=True)
        (d / "observers").mkdir()
        for name in ("config.json", "meta.json", "metrics.jsonl"):
            shutil.copyfile(ref.path / name, d / name)
        shutil.copyfile(ref.social_path, d / "observers" / "social.jsonl")
        shutil.copyfile(ref.checkpoint_at(s), d / "checkpoints" / ref.checkpoint_at(s).name)
        shutil.copyfile(ref.observer_path(s), d / "observers" / ref.observer_path(s).name)
        _resume(d, "--checkpoint", str(s), "--until", str(END))
        assert _log(RunDir.open(d)) == want, s
        shutil.rmtree(d)
