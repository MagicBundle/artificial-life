"""§18 run directories: checkpoints, metadata, atomic writes, retention, logs and truncation (WP5)."""

from __future__ import annotations

import contextlib
import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from alife import layout, rng, runio
from alife.config import Config, apply_overrides, config_from_dict, config_hash, dims
from alife.runio import CheckpointError, RunDir, checkpoint_metadata, load_checkpoint, write_state_file
from alife.state import State, init_state, state_digest, to_numpy
from helpers import make_scene


def _at_step(s: State, step: int) -> State:
    return s._replace(step=mx.array(step, dtype=mx.int32))


def _scene(cfg: Config, step: int = 0) -> State:
    agents = [{"pos": [2.0 + 3.0 * i, 7.0], "energy": 1000 + i} for i in range(6)]
    return make_scene(cfg, agents, bonds=[(0, 1), (2, 3)], step=mx.array(step, dtype=mx.int32))


def _rewrite_metadata(path: Path, **changes: str) -> None:
    arrays, meta = mx.load(str(path), format="safetensors", return_metadata=True)
    mx.eval(arrays)  # loads are lazy: read everything before truncating the file
    meta.update(changes)
    with open(path, "wb") as f:
        mx.save_safetensors(f, arrays, meta)


# --- state files ----------------------------------------------------------------------------------------


def test_expected_layout_matches_init_state(micro_cfg: Config, tiny_cfg_bf16: Config) -> None:
    for cfg in (micro_cfg, tiny_cfg_bf16):
        spec = runio._expected_layout(cfg)
        assert tuple(spec) == State._fields
        for name, a in zip(State._fields, init_state(cfg), strict=True):
            assert spec[name] == (tuple(a.shape), a.dtype), name


@pytest.mark.parametrize("which", ["tiny_cfg", "tiny_cfg_f16", "tiny_cfg_bf16"])
def test_checkpoint_round_trip(which: str, request: pytest.FixtureRequest, tmp_path: Path) -> None:
    cfg: Config = request.getfixturevalue(which)
    s = init_state(cfg)
    cum = np.arange(len(s.ledger_cum), dtype=np.int64) * (3 << 33)  # values past int32
    s = s._replace(
        step=mx.array(1234, dtype=mx.int32),
        ledger_cum=mx.array(cum),
        total=s.total + (5 << 34),
        diag=mx.array(1 << 17, dtype=mx.int32),
        key=mx.array([0xFFFFFFFF, 7], dtype=mx.uint32),
    )
    path = tmp_path / "x.safetensors"
    write_state_file(path, s, cfg)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.safetensors"]
    back = load_checkpoint(path, cfg)
    assert state_digest(back) == state_digest(s)
    for name, a, b in zip(State._fields, s, back, strict=True):
        assert a.dtype == b.dtype and a.shape == b.shape, name

    meta = checkpoint_metadata(path)
    assert meta["format_version"] == str(runio.FORMAT_VERSION)
    assert meta["rng_layout_version"] == str(rng.RNG_LAYOUT_VERSION)
    assert meta["genome_layout_version"] == str(layout.GENOME_LAYOUT_VERSION)
    assert meta["step"] == "1234"
    assert meta["config_hash"] == config_hash(cfg)
    assert meta["mlx_version"] == mx.__version__
    assert meta["alife_version"]
    assert meta["weight_dtype"] == cfg.brain.weight_dtype
    assert config_from_dict(json.loads(meta["config"])) == cfg


def test_atomic_write_survives_failures(micro_cfg: Config, tmp_path: Path, monkeypatch) -> None:
    rd = RunDir.create(tmp_path / "run", micro_cfg)
    good = rd.save_checkpoint(_scene(micro_cfg, 10), micro_cfg)
    digest = state_digest(load_checkpoint(good, micro_cfg))

    def crash_mid_write(f, arrays, metadata=None):
        f.write(b"partial garbage")
        raise OSError("disk full")

    monkeypatch.setattr(runio.mx, "save_safetensors", crash_mid_write)
    with pytest.raises(OSError, match="disk full"):
        rd.save_checkpoint(_scene(micro_cfg, 20), micro_cfg)
    monkeypatch.undo()

    real_replace = runio.os.replace

    def crash_on_rename(src, dst):
        if str(dst).endswith(".safetensors"):
            raise OSError("power cut")
        real_replace(src, dst)

    monkeypatch.setattr(runio.os, "replace", crash_on_rename)
    with pytest.raises(OSError, match="power cut"):
        rd.save_checkpoint(_scene(micro_cfg, 30), micro_cfg)
    monkeypatch.undo()

    names = sorted(p.name for p in rd.checkpoint_dir.iterdir())
    assert names == ["LATEST", "step_0000000010.safetensors"]  # no tmp files, no partial checkpoints
    assert rd.latest_checkpoint() == good
    assert state_digest(load_checkpoint(rd.latest_checkpoint(), micro_cfg)) == digest


def test_latest_and_retention(micro_cfg: Config, tmp_path: Path) -> None:
    cfg = apply_overrides(micro_cfg, ["run.keep_checkpoints=2"])
    rd = RunDir.create(tmp_path / "run", cfg)
    assert rd.checkpoints() == [] and rd.latest_checkpoint() is None
    s = _scene(cfg)
    for step in (0, 10, 20, 30, 40):
        path = rd.save_checkpoint(_at_step(s, step), cfg)
        assert path.name == f"step_{step:010d}.safetensors"
        assert (rd.checkpoint_dir / "LATEST").read_text().strip() == path.name
        assert rd.latest_checkpoint() == path
    assert [st for st, _ in rd.checkpoints()] == [0, 30, 40]
    assert rd.checkpoint_at(30).name == "step_0000000030.safetensors"
    with pytest.raises(FileNotFoundError):
        rd.checkpoint_at(10)
    # LATEST lost (crash between checkpoint and LATEST): the newest complete checkpoint is used.
    (rd.checkpoint_dir / "LATEST").unlink()
    assert rd.latest_checkpoint() == rd.checkpoint_at(40)
    # LATEST naming a missing file is ignored the same way.
    (rd.checkpoint_dir / "LATEST").write_text("step_0000000999.safetensors\n")
    assert rd.latest_checkpoint() == rd.checkpoint_at(40)


def test_version_hash_and_shape_refusal(micro_cfg: Config, tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "s.safetensors"
    write_state_file(path, _scene(micro_cfg, 5), micro_cfg)
    load_checkpoint(path, micro_cfg)

    for key in ("format_version", "rng_layout_version", "genome_layout_version"):
        bad = tmp_path / f"{key}.safetensors"
        bad.write_bytes(path.read_bytes())
        _rewrite_metadata(bad, **{key: "99"})
        with pytest.raises(CheckpointError, match=key):
            load_checkpoint(bad, micro_cfg)
        with pytest.raises(CheckpointError, match=key):
            load_checkpoint(bad, micro_cfg, check_hash=False)

    # The code's layout moved on incompatibly (a version outside the compatible set): old files are refused.
    monkeypatch.setattr(rng, "RNG_LAYOUT_VERSION", max(rng.RNG_COMPATIBLE_VERSIONS) + 1)
    monkeypatch.setattr(rng, "RNG_COMPATIBLE_VERSIONS", frozenset({max(rng.RNG_COMPATIBLE_VERSIONS) + 1}))
    with pytest.raises(CheckpointError, match="rng_layout_version"):
        load_checkpoint(path, micro_cfg)
    monkeypatch.undo()
    monkeypatch.setattr(layout, "GENOME_LAYOUT_VERSION", layout.GENOME_LAYOUT_VERSION + 1)
    with pytest.raises(CheckpointError, match="genome_layout_version"):
        load_checkpoint(path, micro_cfg)
    monkeypatch.undo()

    # A dynamics change alters the hash: refused unless the caller (a fork) skips the check.
    other = apply_overrides(micro_cfg, ["combat.attack_bonded=true"])
    with pytest.raises(CheckpointError, match="config_hash"):
        load_checkpoint(path, other)
    assert state_digest(load_checkpoint(path, other, check_hash=False)) == state_digest(
        load_checkpoint(path, micro_cfg)
    )
    # run.* is not part of the hash.
    load_checkpoint(path, apply_overrides(micro_cfg, ["run.seed=9", "run.metrics_every=5"]))

    # Shapes and dtypes must match the config even without the hash check.
    for over in (["population.n_max=72"], ["brain.hidden=16"], ["brain.weight_dtype=float16"]):
        with pytest.raises(CheckpointError, match="does not match"):
            load_checkpoint(path, apply_overrides(micro_cfg, over), check_hash=False)


def test_corrupt_and_incomplete_files(micro_cfg: Config, tmp_path: Path) -> None:
    garbage = tmp_path / "g.safetensors"
    garbage.write_bytes(b"\x05\x00")
    with pytest.raises(CheckpointError):
        load_checkpoint(garbage, micro_cfg)
    garbage.write_bytes((50).to_bytes(8, "little") + b"{not json")
    with pytest.raises(CheckpointError, match="corrupt"):
        checkpoint_metadata(garbage)
    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path / "missing.safetensors", micro_cfg)

    # A file missing a State field (or with an extra one) is refused.
    s = _scene(micro_cfg)
    good = tmp_path / "s.safetensors"
    write_state_file(good, s, micro_cfg)
    arrays, meta = mx.load(str(good), format="safetensors", return_metadata=True)
    mx.eval(arrays)
    for mutate in (lambda a: a.pop("rev"), lambda a: a.update(extra=mx.zeros((1,)))):
        d = dict(arrays)
        mutate(d)
        bad = tmp_path / "bad.safetensors"
        with open(bad, "wb") as f:
            mx.save_safetensors(f, d, meta)
        with pytest.raises(CheckpointError, match="State fields"):
            load_checkpoint(bad, micro_cfg)


def test_dump_diag(micro_cfg: Config, tmp_path: Path) -> None:
    rd = RunDir.create(tmp_path / "run", micro_cfg)
    s = _scene(micro_cfg, 17)._replace(diag=mx.array(8, dtype=mx.int32))
    path = rd.dump_diag(s, 17)
    assert path == tmp_path / "run" / "diag" / "step_17.safetensors"
    assert checkpoint_metadata(path)["step"] == "17"
    back = load_checkpoint(path, micro_cfg)
    assert state_digest(back) == state_digest(s) and int(to_numpy(back.diag)) == 8
    assert rd.checkpoints() == []  # diag dumps are not checkpoints


# --- run directory ---------------------------------------------------------------------------------------


def test_create_open_and_meta(micro_cfg: Config, tmp_path: Path) -> None:
    path = tmp_path / "run"
    rd = RunDir.create(path, micro_cfg, parent="other:100")
    assert (path / "checkpoints").is_dir() and (path / "events").is_dir()
    meta = rd.meta()
    assert set(meta) == {
        "format_version", "alife_version", "mlx_version", "python", "platform", "device", "git_rev",
        "created_utc", "seed", "config_hash", "rng_layout_version", "genome_layout_version", "parent",
        "status", "knn_impl",
    }  # fmt: skip
    assert meta["status"] == "running" and meta["parent"] == "other:100"
    assert meta["config_hash"] == config_hash(micro_cfg) and meta["seed"] == micro_cfg.run.seed
    assert meta["rng_layout_version"] == rng.RNG_LAYOUT_VERSION
    text = (path / "config.json").read_text()
    assert text == json.dumps(json.loads(text), sort_keys=True, indent=2) + "\n"  # canonical JSON

    opened = RunDir.open(path)
    assert opened.config() == micro_cfg
    opened.update_meta(status="done", note=3)
    assert RunDir.open(path).meta()["status"] == "done" and opened.meta()["note"] == 3
    with pytest.raises(ValueError, match="status"):
        opened.update_meta(status="bogus")
    changed = apply_overrides(micro_cfg, ["run.steps=77"])
    opened.save_config(changed)
    assert RunDir.open(path).config() == changed
    assert not list(path.glob("*.tmp"))

    with pytest.raises(FileExistsError):
        RunDir.create(path, micro_cfg)
    RunDir.create(path, micro_cfg, overwrite=True)
    assert RunDir.open(path).meta()["status"] == "running" and not (path / "metrics.jsonl").exists()

    precious = tmp_path / "precious"
    precious.mkdir()
    (precious / "notes.txt").write_text("keep me")
    with pytest.raises(FileExistsError, match="not a run directory"):
        RunDir.create(precious, micro_cfg, overwrite=True)
    assert (precious / "notes.txt").exists()
    empty = tmp_path / "empty"
    empty.mkdir()
    RunDir.create(empty, micro_cfg)

    with pytest.raises(FileNotFoundError):
        RunDir.open(precious)
    rd.update_meta(format_version=99)
    with pytest.raises(CheckpointError, match="format_version"):
        RunDir.open(path)


def test_append_metrics(micro_cfg: Config, tmp_path: Path) -> None:
    rd = RunDir.create(tmp_path / "run", micro_cfg)
    rd.append_metrics([{"a": np.int64(3), "step": 10, "x": float("nan"), "v": np.float32(0.5), "l": (1, 2)}])
    rd.append_metrics([])
    rd.append_metrics([{"step": np.int32(20), "nested": {"k": np.arange(3)}}])
    lines = (rd.path / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 2 and all(line.startswith('{"step": ') for line in lines)
    rows = rd.metrics_rows()
    assert rows[0] == {"step": 10, "a": 3, "x": None, "v": 0.5, "l": [1, 2]}
    assert rows[1] == {"step": 20, "nested": {"k": [0, 1, 2]}}
    with pytest.raises(ValueError, match="step"):
        rd.append_metrics([{"a": 1}])


def _births(steps: list[int], lanes: int = 3) -> dict[str, np.ndarray]:
    t = len(steps)
    valid = np.zeros((t, lanes), bool)
    valid[:, 0] = True
    valid[::2, 1] = True
    uid = np.arange(t * lanes, dtype=np.int64).reshape(t, lanes) + (1 << 40)
    return {
        "step": np.repeat(np.array(steps, np.int32)[:, None], lanes, 1),
        "uid": uid,
        "parent_uid": uid - 1,
        "clade": np.ones((t, lanes), np.int32),
        "generation": np.full((t, lanes), 2, np.int32),
        "slot": np.arange(t * lanes, dtype=np.int32).reshape(t, lanes),
        "valid": valid,
    }


def _deaths(steps: list[int]) -> dict[str, np.ndarray]:
    n = len(steps)
    return {
        "step": np.array(steps, np.int32),
        "uid": np.arange(n, dtype=np.int64),
        "age": np.full(n, 30, np.int32),
        "cause": np.arange(n) % 3,
        "generation": np.zeros(n, np.int32),
        "clade": np.zeros(n, np.int32),
    }


def test_write_events(micro_cfg: Config, tmp_path: Path) -> None:
    rd = RunDir.create(tmp_path / "run", micro_cfg)
    rd.write_events(0, 4, _births([0, 1, 2, 3]), _deaths([1, 1, 3]))
    with np.load(rd.events_dir / "births_0000000000_0000000004.npz") as z:
        assert set(z.files) == set(runio.BIRTH_COLUMNS)
        for name, dtype in runio.BIRTH_COLUMNS.items():
            assert z[name].dtype == dtype, name
        np.testing.assert_array_equal(z["step"], [0, 0, 1, 2, 2, 3])
        assert z["uid"][0] == 1 << 40
    with np.load(rd.events_dir / "deaths_0000000000_0000000004.npz") as z:
        assert z["cause"].dtype == np.uint8 and list(z["step"]) == [1, 1, 3]
    assert [(a, b) for a, b, _ in rd.event_files("births")] == [(0, 4)]

    with pytest.raises(ValueError, match="outside"):
        rd.write_events(4, 6, _births([6]), _deaths([]))
    bad = _births([4])
    del bad["slot"]
    with pytest.raises(ValueError, match="missing"):
        rd.write_events(4, 6, bad, _deaths([]))
    assert [(a, b) for a, b, _ in rd.event_files("births")] == [(0, 4)]


def test_read_events(micro_cfg: Config, tmp_path: Path) -> None:
    rd = RunDir.create(tmp_path / "run", micro_cfg)
    empty = rd.read_events("births")
    assert set(empty) == set(runio.BIRTH_COLUMNS) and all(v.size == 0 for v in empty.values())
    assert empty["uid"].dtype == np.int64
    rd.write_events(4, 8, _births([4, 7]), _deaths([5]))
    rd.write_events(0, 4, _births([0, 3]), _deaths([]))
    b = rd.read_events("births")
    assert list(b["step"]) == [0, 0, 3, 4, 4, 7] and b["step"].dtype == np.int32
    d = rd.read_events("deaths")
    assert list(d["step"]) == [5] and d["cause"].dtype == np.uint8
    with pytest.raises(ValueError, match="kind"):
        rd.read_events("lives")


def test_truncate_after(micro_cfg: Config, tmp_path: Path) -> None:
    rd = RunDir.create(tmp_path / "run", micro_cfg)
    s = _scene(micro_cfg)
    for s0, s1 in ((0, 10), (10, 20), (20, 30), (30, 40)):
        rd.append_metrics([{"step": s1, "steps": s1 - s0}])
        rd.write_events(s0, s1, _births(list(range(s0, s1))), _deaths([s0, s1 - 1]))
        if s1 % 20 == 0:
            rd.save_checkpoint(_at_step(s, s1), micro_cfg)
    rd.save_checkpoint(_at_step(s, 0), micro_cfg)
    rd.save_checkpoint(_at_step(s, 40), micro_cfg)
    with open(rd.path / "metrics.jsonl", "a") as f:
        f.write('{"step": 50, "ste')  # a torn append from a crash

    rd.truncate_after(25)
    assert [r["step"] for r in rd.metrics_rows()] == [10, 20]
    assert (rd.path / "metrics.jsonl").read_text().endswith("}\n")
    for kind in ("births", "deaths"):
        assert [(a, b) for a, b, _ in rd.event_files(kind)] == [(0, 10), (10, 20), (20, 25)]
    with np.load(rd.events_dir / "births_0000000020_0000000025.npz") as z:
        assert z["step"].max() == 24 and set(z["step"]) == set(range(20, 25))
    with np.load(rd.events_dir / "deaths_0000000020_0000000025.npz") as z:
        assert list(z["step"]) == [20]
    assert [st for st, _ in rd.checkpoints()] == [0, 20]
    assert rd.latest_checkpoint() == rd.checkpoint_at(20)
    assert (rd.checkpoint_dir / "LATEST").read_text().strip() == "step_0000000020.safetensors"

    rd.truncate_after(20)  # a chunk starting at the checkpoint step belongs to the abandoned future
    assert [(a, b) for a, b, _ in rd.event_files("births")] == [(0, 10), (10, 20)]
    assert [r["step"] for r in rd.metrics_rows()] == [10, 20]
    rd.truncate_after(0)
    assert rd.metrics_rows() == []
    assert [(a, b) for a, b, _ in rd.event_files("births")] == []
    assert [st for st, _ in rd.checkpoints()] == [0]
    assert not list(rd.path.rglob("*.tmp"))


def test_truncate_after_on_fresh_run(micro_cfg: Config, tmp_path: Path) -> None:
    rd = RunDir.create(tmp_path / "run", micro_cfg)
    rd.truncate_after(100)
    assert rd.metrics_rows() == [] and rd.latest_checkpoint() is None
    assert not (rd.path / "metrics.jsonl").exists()
    # the social log is never created by a cut: a run that never observes has no observers/ at all
    assert not rd.observers_dir.exists() and not rd.social_path.exists() and rd.social_records() == []
    rd.truncate_after(0, keep_director=True)
    assert not rd.observers_dir.exists()


class _Stop(BaseException):
    """A power cut: not an ``Exception``, so nothing under test can swallow it."""


@contextlib.contextmanager
def _power_cut(root: Path, k: int) -> Iterator[None]:
    """Raise ``_Stop`` at the ``k``-th (from 0) write or delete under ``root``, before it happens."""
    seen = {"n": 0}
    real = (Path.unlink, runio._atomic_write_text, runio._write_npz)

    def tick(path: Path) -> None:
        if str(path).startswith(str(root)):
            if seen["n"] == k:
                raise _Stop
            seen["n"] += 1

    def unlink(self: Path, *a: object, **kw: object) -> None:
        tick(self)
        real[0](self, *a, **kw)

    def write_text(path: Path, *a: object, **kw: object) -> None:
        tick(path)
        real[1](path, *a, **kw)

    def write_npz(path: Path, *a: object, **kw: object) -> None:
        tick(path)
        real[2](path, *a, **kw)

    with pytest.MonkeyPatch.context() as m:
        m.setattr(Path, "unlink", unlink)
        m.setattr(runio, "_atomic_write_text", write_text)
        m.setattr(runio, "_write_npz", write_npz)
        yield


def test_truncate_after_stopped_anywhere_loses_nothing(micro_cfg: Config, tmp_path: Path) -> None:
    """`alife resume --checkpoint 20` stopped by a power cut at any point of the cut, then a plain `resume`
    (which loads the newest checkpoint and cuts again): a log is never cut while a checkpoint after the step
    is still there to be resumed from (the records between would be lost for good), and finishing the cut
    gives exactly the files of an uninterrupted one. The run has every kind of file the cut touches: rows, the
    director log and archive, the social log, event chunks (one spans the step), checkpoints with sidecars."""
    cfg = apply_overrides(micro_cfg, ["run.keep_checkpoints=3"])
    rd = RunDir.create(tmp_path / "pristine", cfg)
    s = _scene(cfg)
    for s0, s1 in ((0, 30), (30, 60)):
        rd.write_events(s0, s1, _births(list(range(s0, s1))), _deaths([s0, s1 - 1]))
    rd.append_metrics([{"step": t, "steps": 10} for t in range(10, 61, 10)])
    rd.append_jsonl("director.jsonl", [{"step_applied": t, "kind": "x"} for t in (10, 30, 50)])
    rd.write_json("director_archive.json", {"as_of_step": 60})
    rd.append_social([_social(t) for t in range(15, 61, 15)])
    for step in (0, 20, 40, 60):
        rd.save_checkpoint(_at_step(s, step), cfg, observers={"x": np.arange(3, dtype=np.int32)})
    assert [st for st, _ in rd.checkpoints()] == [0, 20, 40, 60]
    base = _tree(rd.path)
    uncut = shutil.copytree(rd.path, tmp_path / "uncut")
    RunDir.open(uncut).truncate_after(20)
    want = _tree(uncut)
    assert want != base and not list(uncut.rglob("*.tmp"))

    def is_log(rel: str) -> bool:  # everything the cut rewrites or deletes but the checkpoints and sidecars
        return not rel.startswith("checkpoints/") and not rel.startswith("observers/step_")

    k = 0
    while True:
        twin = shutil.copytree(rd.path, tmp_path / f"stop{k}")
        with _power_cut(twin, k):
            try:
                RunDir.open(twin).truncate_after(20)
                stopped = False
            except _Stop:
                stopped = True
        here = RunDir.open(twin)
        tree = _tree(twin)
        logs_cut = any(tree.get(rel) != data for rel, data in base.items() if is_log(rel)) or any(
            is_log(rel) for rel in tree.keys() - base.keys()
        )
        later = [st for st, _ in here.checkpoints() if st > 20]
        assert not (logs_cut and later), f"stopped at operation {k}: logs cut with checkpoints {later} left"
        if not later:  # nothing after the step: a plain resume takes the checkpoint at the step
            assert here.latest_checkpoint() == here.checkpoint_at(20), k
        RunDir.open(twin).truncate_after(20)  # the next resume cuts again
        assert _tree(twin) == want, f"stopped at operation {k}: finishing the cut differs from one cut"
        if not stopped:
            break
        k += 1
    assert k >= 10  # the cut has a dozen separate writes and deletes: the loop did stop in the middle of it


def test_save_checkpoint_reads_step_from_state(micro_cfg: Config, tmp_path: Path) -> None:
    rd = RunDir.create(tmp_path / "run", micro_cfg)
    path = rd.save_checkpoint(_scene(micro_cfg, 123), micro_cfg)
    assert path.name == "step_0000000123.safetensors"
    back = load_checkpoint(path, micro_cfg)
    assert int(to_numpy(back.step)) == 123
    assert back.W1.shape == (dims(micro_cfg).n, dims(micro_cfg).w1_rows, dims(micro_cfg).h)


# --- the social log (DESIGN §29.10) -----------------------------------------------------------------------


def _social(step: int, window_start: int | None = None, origin: int = 0) -> dict:
    """A record shaped like a snapshot's: nested tables, floats, None, booleans and a non-ASCII name."""
    return {
        "step": step,
        "social_version": 1,
        "origin": origin,
        "window_start": step - 15 if window_start is None else window_start,
        "summary": {"houses": {"alive": step // 3, "exact": origin == 0}, "colonies": {"converged": True}},
        "houses": {"id": [step, step + 1], "name": ["Thraeloust", "Ærin é"], "size": [4, 2]},
        "relations": {"a": [], "sim": [0.1234, None], "distinct": [True, None]},
        "purity_mean": step / 7,
    }


def _tree(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_append_and_read_social(micro_cfg: Config, tmp_path: Path) -> None:
    rd = RunDir.create(tmp_path / "run", micro_cfg)
    assert runio.SOCIAL_FILE == "social.jsonl" and rd.social_path == rd.path / "observers" / "social.jsonl"
    assert rd.social_records() == []
    rd.append_social([])  # a no-op that creates neither observers/ nor the file
    rd.append_social(iter(()))
    assert not rd.observers_dir.exists() and not rd.social_path.exists()
    # records are written with `step` first, whatever the order of the keys, and numpy values are plain JSON
    odd = {"b": np.int64(3), "step": np.int32(15), "a": [np.float32(0.5), float("nan")]}
    rd.append_social([odd])
    assert rd.social_path.read_text() == '{"step": 15, "b": 3, "a": [0.5, null]}\n'
    rd.append_social([_social(30), _social(45)])
    rd.append_social(_social(s) for s in (60, 75))  # any iterable
    assert [r["step"] for r in rd.social_records()] == [15, 30, 45, 60, 75]
    assert rd.social_records()[1] == _social(30)
    assert all(line.startswith('{"step": ') for line in rd.social_path.read_text().splitlines())
    # the file is not an observer sidecar: the sidecar listing never sees it
    assert rd.observer_files() == []
    # invalid calls raise before anything is written
    before = rd.social_path.read_bytes()
    for bad in (
        [{"a": 1}],  # no step
        [{"step": "90"}],  # not an int
        [{"step": 90.0}],
        [{"step": True}],
        [_social(90), _social(90)],  # not strictly increasing
        [_social(105), _social(100)],
        ["step"],
    ):
        with pytest.raises(ValueError, match="step"):
            rd.append_social(bad)  # type: ignore[arg-type]
        assert rd.social_path.read_bytes() == before
    # a torn last line (a crash mid-append) is ignored on read; a corrupt line elsewhere raises
    with open(rd.social_path, "a") as f:
        f.write('{"step": 90, "social_ver')
    assert [r["step"] for r in rd.social_records()] == [15, 30, 45, 60, 75]
    lines = rd.social_path.read_text().splitlines(keepends=True)
    rd.social_path.write_text("".join([*lines[:2], '{"step": 31, "x\n', *lines[2:]]))
    with pytest.raises(json.JSONDecodeError):
        rd.social_records()


def test_truncate_after_cuts_social(micro_cfg: Config, tmp_path: Path) -> None:
    rd = RunDir.create(tmp_path / "run", micro_cfg)
    rd.append_social([_social(s) for s in (15, 30, 45, 60, 75)])
    original = rd.social_path.read_text().splitlines(keepends=True)
    for keep_director in (False, True):  # the log is cut whatever keep_director is
        twin = tmp_path / f"twin{keep_director}"
        shutil.copytree(rd.path, twin)
        t = RunDir.open(twin)
        t.truncate_after(45, keep_director=keep_director)
        assert t.social_path.read_text() == "".join(original[:3])  # 15, 30, 45 with their original bytes
        t.truncate_after(14, keep_director=keep_director)
        assert t.social_path.read_text() == "" and t.social_records() == []  # all cut: an empty file
        t.append_social([_social(15)])
        assert t.social_path.read_text() == original[0]
    # a cut at a step between two records, and past the last
    twin = tmp_path / "between"
    shutil.copytree(rd.path, twin)
    RunDir.open(twin).truncate_after(59)
    assert RunDir.open(twin).social_path.read_text() == "".join(original[:3])
    RunDir.open(twin).truncate_after(1000)
    assert RunDir.open(twin).social_path.read_text() == "".join(original[:3])

    # a file whose complete records are all <= the step but which ends in a torn tail is still rewritten
    torn = tmp_path / "torn"
    shutil.copytree(rd.path, torn)
    t = RunDir.open(torn)
    t.truncate_after(30)
    with open(t.social_path, "a") as f:
        f.write(_dumps_prefix(_social(45)))
    assert not t.social_path.read_text().endswith("\n")
    t.truncate_after(30)  # no record is removed, the tail is
    assert t.social_path.read_text() == "".join(original[:2])
    t.append_social([_social(45)])  # the next append is not glued to a fragment
    assert [r["step"] for r in t.social_records()] == [15, 30, 45]
    assert t.social_path.read_text() == "".join(original[:3])

    # a complete last line without its newline is kept (and gets one)
    nolf = tmp_path / "nolf"
    shutil.copytree(rd.path, nolf)
    t = RunDir.open(nolf)
    t.social_path.write_text("".join(original[:2]).rstrip("\n"))
    t.truncate_after(30)
    assert t.social_path.read_text() == "".join(original[:2])

    # keep_director=True keeps director.jsonl and cuts the log; keep_director=False cuts both
    for keep_director in (True, False):
        d = tmp_path / f"director{keep_director}"
        shutil.copytree(rd.path, d)
        t = RunDir.open(d)
        t.append_jsonl(
            "director.jsonl", [{"step_applied": 30, "kind": "x"}, {"step_applied": 60, "kind": "y"}]
        )
        t.truncate_after(45, keep_director=keep_director)
        assert [r["step"] for r in t.social_records()] == [15, 30, 45]
        assert len(t.read_jsonl("director.jsonl")) == (2 if keep_director else 1)

    # the log is not an observer sidecar: pruning and the sidecar listing never touch it
    cfg = apply_overrides(micro_cfg, ["run.keep_checkpoints=1"])
    pr = RunDir.create(tmp_path / "prune", cfg)
    pr.append_social([_social(15), _social(30)])
    s = _scene(cfg)
    for step in (0, 10, 20):
        pr.save_checkpoint(_at_step(s, step), cfg, observers={"x": np.zeros(2, np.int32)})
    assert [st for st, _ in pr.observer_files()] == [0, 20] and len(pr.social_records()) == 2
    pr.truncate_after(5)
    assert [st for st, _ in pr.observer_files()] == [0] and len(pr.social_records()) == 0

    # a failing os.replace leaves the old file and no tmp file
    broken = tmp_path / "broken"
    shutil.copytree(rd.path, broken)
    b = RunDir.open(broken)
    real_replace = runio.os.replace

    def crash_on_log(src, dst):
        if str(dst).endswith("social.jsonl"):
            raise OSError("power cut")
        real_replace(src, dst)

    with pytest.MonkeyPatch.context() as m:
        m.setattr(runio.os, "replace", crash_on_log)
        with pytest.raises(OSError, match="power cut"):
            b.truncate_after(30)
    assert b.social_path.read_text() == "".join(original) and not list(b.path.rglob("*.tmp"))


def _dumps_prefix(rec: dict, n: int = 40) -> str:
    return runio._dumps(rec)[:n]


@pytest.mark.parametrize(
    ("bad_line", "error"),
    [
        pytest.param('{"step": 31, "x', json.JSONDecodeError, id="corrupt-middle-line"),
        pytest.param('{"step": "31"}', ValueError, id="step-not-an-int"),
        pytest.param('{"step": 31.0}', ValueError, id="step-a-float"),
        pytest.param('{"step": true}', ValueError, id="step-a-bool"),
        pytest.param('{"x": 1}', ValueError, id="no-step"),
        pytest.param("[31]", ValueError, id="not-an-object"),
    ],
)
def test_truncate_after_social_refuses_a_corrupt_log(
    bad_line: str, error: type[Exception], micro_cfg: Config, tmp_path: Path
) -> None:
    """A log that cannot be parsed whole fails the cut before anything is changed: the checkpoints, a sidecar,
    an event chunk, the metrics rows after the step and the log itself are byte-unchanged."""
    rd = RunDir.create(tmp_path / "run", micro_cfg)
    s = _scene(micro_cfg)
    for s0, s1 in ((0, 30), (30, 60)):
        rd.append_metrics([{"step": s1, "steps": s1 - s0}])
        rd.write_events(s0, s1, _births(list(range(s0, s1))), _deaths([s0, s1 - 1]))
        rd.save_checkpoint(_at_step(s, s1), micro_cfg, observers={"x": np.arange(3, dtype=np.int32)})
    rd.save_checkpoint(_at_step(s, 0), micro_cfg)
    rd.append_social([_social(15), _social(30), _social(45), _social(60)])
    lines = rd.social_path.read_text().splitlines(keepends=True)
    rd.social_path.write_text("".join([*lines[:2], bad_line + "\n", *lines[2:]]))
    before = _tree(rd.path)
    for keep_director in (False, True):
        with pytest.raises(error):
            rd.truncate_after(30, keep_director=keep_director)
        assert _tree(rd.path) == before  # not a byte changed, nothing created
    assert not list(rd.path.rglob("*.tmp"))
    # the error names the line for a record without an int step
    if error is ValueError:
        with pytest.raises(ValueError, match=r"record 3 "):
            rd.truncate_after(30)
    # the same text as the LAST line: a corrupt fragment is a torn append and is dropped; a complete line that
    # is not a record still raises
    rd.social_path.write_text("".join(lines) + bad_line)
    if error is json.JSONDecodeError:
        rd.truncate_after(30)
        assert [r["step"] for r in rd.social_records()] == [15, 30]
    else:
        with pytest.raises(ValueError, match=r"record 5 "):
            rd.truncate_after(30)


def test_resume_refuses_a_corrupt_social_log(
    micro_cfg: Config, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`alife resume` on a directory whose log has a corrupt middle line exits 1 with the traceback on stderr,
    having changed nothing in the directory."""
    from alife import cli, runner

    rd = RunDir.create(tmp_path / "run", micro_cfg)
    rd.save_checkpoint(_at_step(_scene(micro_cfg), 30), micro_cfg)
    rd.append_social([_social(15), _social(30), _social(45)])
    lines = rd.social_path.read_text().splitlines(keepends=True)
    rd.social_path.write_text("".join([lines[0], "not json\n", *lines[1:]]))
    before = _tree(rd.path)
    capsys.readouterr()
    assert cli.main(["resume", str(rd.path), "--steps", "5"]) == int(runner.ExitCode.ERROR) == 1
    assert "JSONDecodeError" in capsys.readouterr().err and _tree(rd.path) == before
