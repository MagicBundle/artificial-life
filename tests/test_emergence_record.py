"""Frame format, readers and the window recorder (DESIGN §44.2): schema, quantisation, certification."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from alife import runner, streamrec
from alife.analysis.emergence import frames as fm
from alife.analysis.emergence import record, traits
from alife.config import Config, apply_overrides
from alife.runio import RunDir, load_checkpoint
from alife.state import init_state, state_digest, to_numpy
from alife.step import make_step

# --- schema, quantisation, registry (no engine) ---


def _toy(rng: np.random.Generator, nd: int = 3, drift: bool = True, fine: bool = True) -> fm.Frames:
    sizes = [5, 7, 4]
    off = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    n = int(off[-1])
    size, height = 160.0, 80.0
    pos = np.column_stack([rng.uniform(0, size, n), rng.uniform(0, size, n), rng.uniform(0, height, n)])[
        :, :nd
    ]
    names = ["tag0", "age"]
    fine_d = None
    if fine:
        fine_d = {
            "fine_step": np.array([0, 5, 10], np.int64),
            "fine_off": np.array([0, 2, 3, 5], np.int64),
            "fine_uid": np.array([1, 2, 1, 1, 2], np.int64),
            "fine_group": np.array([0, 1, 0, 0, 1], np.int32),
            "fine_pos": fm.quantise(pos[:5], size, height),
            "fine_body": np.array([3, 3, 3, 3, 3], np.int32),
        }
    return fm.Frames(
        meta={"size": size, "height": height, "nd": nd, "run": "toy"},
        step=np.array([100, 200, 300], np.int64),
        off=off,
        uid=np.arange(n, dtype=np.int64),
        row=np.arange(n, dtype=np.int32),
        pos=fm.quantise(pos, size, height),
        clade=rng.integers(0, 5, n).astype(np.int32),
        body=np.full(n, -1, np.int32),
        drift=rng.standard_normal((n, nd)).astype(np.float16) if drift else None,
        trait_names=names,
        traits=rng.standard_normal((n, 2)).astype(np.float32),
        fine=fine_d,
    )


def test_frames_round_trip_and_csr(tmp_path: Path) -> None:
    fr = _toy(np.random.default_rng(0))
    p = fm.write_frames(tmp_path / "frames_100_300.npz", fr)
    assert [f.name for f in tmp_path.iterdir()] == ["frames_100_300.npz"]  # atomic: no temp debris
    back = fm.read_frames(p)
    assert back.n_frames == 3 and list(back.step) == [100, 200, 300] and list(back.off) == [0, 5, 12, 16]
    for name in ("uid", "row", "pos", "clade", "body", "drift", "traits"):
        assert np.array_equal(getattr(back, name), getattr(fr, name)), name
    assert back.trait_names == ["tag0", "age"] and back.meta["frames_version"] == fm.FRAMES_VERSION
    assert back.pos.dtype == np.uint16 and back.drift.dtype == np.float16 and back.traits.dtype == np.float32
    f1 = back.frame(1)
    assert len(f1["uid"]) == 7 and "age" in f1 and "drift" in f1
    assert np.array_equal(back.trait("age")[back.span(2)], back.frame(2)["age"])
    assert (
        back.n_fine_frames == 3
        and list(back.fine_frame(0)["uid"]) == [1, 2]
        and len(back.fine_frame(2)["uid"]) == 2
    )
    assert (
        fm.read_frames(
            fm.write_frames(tmp_path / "b.npz", _toy(np.random.default_rng(1), drift=False, fine=False))
        ).drift
        is None
    )


def test_a_wrong_version_or_broken_offsets_is_refused(tmp_path: Path) -> None:
    fr = _toy(np.random.default_rng(0))
    fr.meta["frames_version"] = 1
    p = fm.write_frames(tmp_path / "a.npz", fr)
    z = dict(np.load(p))
    z["meta"] = np.array(json.dumps({**json.loads(str(z["meta"])), "frames_version": 7}))
    np.savez(tmp_path / "b.npz", **z)
    with pytest.raises(fm.FramesError, match="frames_version"):
        fm.read_frames(tmp_path / "b.npz")
    bad = _toy(np.random.default_rng(0))
    bad.off = bad.off.copy()
    bad.off[-1] += 1
    with pytest.raises(fm.FramesError):
        fm.Frames(
            **{k: getattr(bad, k) for k in ("meta", "step", "off", "uid", "row", "pos", "clade", "body")}
        )
    with pytest.raises(fm.FramesError):
        fm.read_frames(tmp_path / "missing.npz")


def test_quantisation_error_bound_and_cell_assignment() -> None:
    rng = np.random.default_rng(1)
    size, height, n = 160.0, 80.0, 200_000
    pos = np.column_stack([rng.uniform(0, size, n), rng.uniform(0, size, n), rng.uniform(0, height, n)])
    q = fm.quantise(pos, size, height)
    err = np.abs(fm.dequantise(q, size, height) - pos)
    assert q.dtype == np.uint16
    assert err[:, :2].max() < size / 65535 and err[:, 2].max() < height / 65535
    assert err[:, :2].max() <= size / 131070 * (1 + 1e-9)  # half a quantum
    cell = 8.0
    cf = np.floor(pos[:, :2] / cell).astype(int)
    cq = np.floor(fm.dequantise(q, size, height)[:, :2] / cell).astype(int)
    cq %= int(size / cell)  # x = L (top code) wraps to cell 0
    differs = (cf != cq).any(axis=1)
    near = (np.abs(pos[:, :2] / cell - np.rint(pos[:, :2] / cell)) * cell < size / 65535).any(axis=1)
    assert not (differs & ~near).any()  # a cell differs only within one quantum of a boundary
    assert differs.mean() < 1e-3
    assert fm.quantise(np.array([[160.0, 0.0, 80.0]]), size, height).tolist() == [[65535, 0, 65535]]
    q2 = fm.quantise(pos[:, :2], size)
    assert q2.shape == (n, 2) and np.abs(fm.dequantise(q2, size) - pos[:, :2]).max() < size / 65535


def test_trait_registry_drops_missing_columns_and_takes_a_feature_line() -> None:
    n = 4
    ctx = {"step": 1000, "tag": np.arange(n * 4, dtype=np.float32).reshape(n, 4), "energy": np.arange(n),
           "birth_step": np.array([0, 100, 200, 300]), "speed": np.ones(n), "n_bonds": np.zeros(n),
           "offspring_frac": np.full(n, 0.5)}  # fmt: skip
    traits.reset()
    try:
        names, v = traits.columns(ctx, n)
        assert names == [
            "tag0",
            "tag1",
            "tag2",
            "tag3",
            "offspring_frac",
            "energy",
            "age",
            "n_bonds",
            "speed",
        ]
        assert "buoy_cap" not in names  # a 2D run has no buoyancy: dropped, not raised
        assert (
            v.shape == (n, 9)
            and v.dtype == np.float32
            and list(v[:, names.index("age")]) == [1000, 900, 800, 700]
        )
        traits.register(traits.Trait("role", lambda c: c.get("role"), feature="bonds"))
        names2, _ = traits.columns({**ctx, "role": np.arange(n)}, n)
        assert names2[-1] == "role"
        assert "role" not in traits.columns(ctx, n)[0]  # the state does not carry it: no column
        late = traits.values_for(ctx, n, ["tag0", "role"])
        assert np.isnan(late[:, 1]).all() and not np.isnan(late[:, 0]).any()
    finally:
        traits.reset()
    assert [t.name for t in traits.registry()][:2] == ["tag0", "tag1"]


def test_frma_reader_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    size, height, ncap = 160.0, 80.0, 40
    hdr = streamrec.legacy_header(
        dims=3, size=size, height=height, n=ncap, bond_slots=4, radius=1.0, field_cell=4.0, fg=40, fz=20,
        nch=3,
        step=0, run="toy", config_hash="h", source="run",
    )  # fmt: skip
    blob = streamrec.header_bytes(hdr)
    truth = []
    for k in range(3):
        alive = np.zeros(ncap, bool)
        alive[: 10 + k] = True
        pos = np.column_stack(
            [rng.uniform(0, size, ncap), rng.uniform(0, size, ncap), rng.uniform(0, height, ncap)]
        )
        uid = np.arange(ncap, dtype=np.int64) + 1000 * k
        bonds = np.array([[0, 1], [1, 2], [5, 6]])
        blob += streamrec.encode_frma(
            step=100 * k, seq=k, alive=alive, pos=pos.astype(np.float32),
            orient=np.zeros((ncap, 2), np.float32),
            rgb=np.zeros(ncap, np.uint32), uid=uid, bonds=bonds, color_index=0,
        )  # fmt: skip
        truth.append((alive, pos, uid))
    p = tmp_path / "w.alstrm"
    p.write_bytes(blob)
    fr = fm.read_frma(p)
    assert fr.n_frames == 3 and list(fr.step) == [0, 100, 200] and list(np.diff(fr.off)) == [10, 11, 12]
    for k, (alive, pos, uid) in enumerate(truth):
        assert np.array_equal(fr.frame(k)["uid"], uid[alive])
        assert np.abs(fr.positions(k) - pos[alive]).max() < size / 65535
    assert list(fr.frame(0)["body"][:8]) == [
        0,
        0,
        0,
        -1,
        -1,
        5,
        5,
        -1,
    ]  # chain 0-1-2, pair 5-6, rest unbonded
    assert (
        (fr.clade == -1).all() and fr.drift is None and fr.trait_names == [] and fr.meta["source"] == "frma"
    )
    assert fm.read_frma(p, every=2).n_frames == 2 and fm.read_frma(p, max_frames=1).n_frames == 1
    with pytest.raises(fm.FramesError, match="truncated"):
        (tmp_path / "t.alstrm").write_bytes(blob[:-12])
        fm.read_frma(tmp_path / "t.alstrm")
    with pytest.raises(fm.FramesError, match="not a play stream"):
        (tmp_path / "x").write_bytes(b"nope" * 8)
        fm.read_frma(tmp_path / "x")


# --- the recorder on a real (tiny3d) run ---


@pytest.fixture(scope="module")
def run_dir(tiny3d_cfg: Config, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 80-step tiny3d run: metrics every 5 steps, saves every 20 kept (0, 20, 40, 60, 80)."""
    cfg = apply_overrides(
        tiny3d_cfg,
        [
            "run.metrics_every=5",
            "run.checkpoint_every=20",
            "run.heavy_metrics_every=5",
            "run.keep_checkpoints=10",
        ],
    )
    path = tmp_path_factory.mktemp("emrec") / "run"
    rd = RunDir.create(path, cfg)
    assert runner.run(rd, cfg, init_state(cfg), 80, quiet=True).status == "done"
    return path


def _files(path: Path) -> dict[str, int]:
    return {str(p.relative_to(path)): p.stat().st_mtime_ns for p in path.rglob("*") if p.is_file()}


@pytest.fixture(scope="module")
def recorded(run_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, int]]:
    out = tmp_path_factory.mktemp("frames")
    before = _files(run_dir)
    path = record.record_window(run_dir, 20, 60, out, tau=10, fine_tau=5, fine_groups=0, log=lambda _: None)
    assert _files(run_dir) == before  # the run directory is read only
    return path, before


def test_recorded_frames_equal_a_plain_run_and_certify(
    run_dir: Path, recorded: tuple[Path, dict[str, int]]
) -> None:
    fr = fm.read_frames(recorded[0])
    assert list(fr.step) == [20, 30, 40, 50, 60] and recorded[0].name == "frames_20_60.npz"
    ver = fr.meta["verification"]
    assert (
        ver["verified"]
        and ver["rows_checked"] >= 7
        and ver["fingerprints"] == [40, 60]
        and ver["end_is_save"]
    )
    assert not fr.has_drift and "drift" not in fr.frame(0)  # no weather in tiny3d: no drift array
    assert fr.nd == 3 and "buoy_cap" in fr.trait_names and "age" in fr.trait_names
    rd = RunDir.open(run_dir)
    cfg = rd.config()
    s = load_checkpoint(rd.checkpoint_at(20), cfg)
    step = make_step(cfg)
    for t in range(20, 60):  # a plain run, no recorder
        if t % 10 == 0:
            alive = to_numpy(s.alive).astype(bool)
            i = (t - 20) // 10
            assert np.array_equal(fr.frame(i)["uid"], to_numpy(s.uid)[alive])
            q = fm.quantise(to_numpy(s.pos)[alive], fr.size, fr.height)
            assert np.array_equal(fr.frame(i)["pos"], q)
            assert np.array_equal(fr.frame(i)["clade"], to_numpy(s.clade)[alive])
        s, _ = step(s)
        mx.eval(s)
    # purity: recording did not change the trajectory (the replayed end state is the run's own save)
    assert (
        fr.meta["verification"]["end_digest"]
        == state_digest(s)
        == state_digest(load_checkpoint(rd.checkpoint_at(60), cfg))
    )
    assert (fr.trait("age")[fr.span(1)] >= 0).all()
    assert np.isnan(fr.trait("speed")[fr.span(0)]).all() and np.isfinite(fr.trait("speed")[fr.span(2)]).any()


def test_recording_is_idempotent(run_dir: Path, recorded: tuple[Path, dict[str, int]]) -> None:
    path = recorded[0]
    m0, logs = path.stat().st_mtime_ns, []
    assert (
        record.record_window(run_dir, 20, 60, path.parent, tau=10, fine_tau=5, fine_groups=0, log=logs.append)
        == path
    )
    assert path.stat().st_mtime_ns == m0 and "not rewritten" in logs[-1]
    record.record_window(
        run_dir, 20, 60, path.parent, tau=20, fine_groups=0, log=lambda _: None
    )  # other parameters: rewritten
    assert fm.read_frames(path).meta["params"]["tau"] == 20


def test_fine_set_follows_the_chosen_group(
    run_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def pick(first: dict, groups: int, per_group: int) -> tuple[np.ndarray, np.ndarray]:
        u = np.sort(first["uid"])[:6]
        return u, np.zeros(6, np.int32)

    monkeypatch.setattr(record, "_pick_fine", pick)
    p = record.record_window(run_dir, 20, 40, tmp_path, tau=10, fine_tau=5, fine_groups=1, log=lambda _: None)
    fr = fm.read_frames(p)
    assert list(fr.fine["fine_step"]) == [20, 25, 30, 35, 40] and fr.n_fine_frames == 5
    f0 = fr.fine_frame(0)
    assert len(f0["uid"]) <= 6 and set(f0["uid"]) <= set(fr.frame(0)["uid"])
    both = fr.fine_frame(2)  # step 30 is also a full frame: same positions
    full = fr.frame(1)
    for u, q in zip(both["uid"], both["pos"], strict=True):
        assert np.array_equal(full["pos"][full["uid"] == u][0], q)


def test_replay_mismatch_refuses_and_writes_nothing(run_dir: Path, tmp_path: Path) -> None:
    """Edge case E21: a tampered metrics row, then a tampered save, each stop the recorder."""
    bad = tmp_path / "run"
    shutil.copytree(run_dir, bad)
    rows = [json.loads(line) for line in (bad / "metrics.jsonl").read_text().splitlines()]
    for r in rows:
        if r["step"] == 35:
            r["births"] += 1
    (bad / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "o"
    with pytest.raises(record.RecordMismatch, match="step 35"):
        record.record_window(bad, 20, 60, out, tau=10, fine_groups=0, log=lambda _: None)
    assert not list(out.glob("*.npz")) if out.exists() else True
    # rows intact, but the save at 60 is another state: the fingerprint disagrees
    bad2 = tmp_path / "run2"
    shutil.copytree(run_dir, bad2)
    src = RunDir.open(bad2).checkpoint_at(20)
    dst = RunDir.open(bad2).checkpoint_at(60)
    shutil.copyfile(src, dst)
    (bad2 / "metrics.jsonl").write_text("")  # only the fingerprint can speak
    with pytest.raises(record.RecordMismatch, match="checkpoint"):
        record.record_window(bad2, 20, 60, out, tau=10, fine_groups=0, log=lambda _: None)
    assert not list(out.glob("*.npz")) if out.exists() else True
    # uncertified recording is allowed explicitly and says so
    p = record.record_window(bad, 20, 40, out, tau=10, fine_groups=0, verify=False, log=lambda _: None)
    assert fm.read_frames(p).meta["verification"]["verified"] is False


def test_start_save_missing_is_refused_clearly(run_dir: Path, tmp_path: Path) -> None:
    """Edge case E22: no save at or before T0 (a run that kept only its newest save)."""
    only_newest = tmp_path / "run"
    shutil.copytree(run_dir, only_newest)
    rd = RunDir.open(only_newest)
    for st, p in rd.checkpoints():
        if st < 80:
            p.unlink()
    with pytest.raises(record.RecordError) as e:
        record.record_window(only_newest, 20, 60, tmp_path / "o", log=lambda _: None)
    msg = str(e.value)
    assert "saves `all`" in msg and "[80]" in msg and "step 20" in msg
    with pytest.raises(record.RecordError, match="steps earlier"):
        record.record_window(
            run_dir, 55, 60, tmp_path / "o", max_lead=10, tau=5, fine_groups=0, log=lambda _: None
        )
    with pytest.raises(record.RecordError, match="0 <= T0 < T1"):
        record.record_window(run_dir, 60, 20, tmp_path / "o", log=lambda _: None)


@pytest.mark.slow
def test_weather_world_records_the_wind_drift(tmp_path: Path) -> None:
    """With weather on the frames carry ``drift`` (the post-step ``weather.agent_drift``, float16)."""
    from weather_helpers import storm3d_config

    cfg = storm3d_config(
        "run.metrics_every=5",
        "run.checkpoint_every=20",
        "run.heavy_metrics_every=5",
        "run.keep_checkpoints=10",
    )
    assert cfg.weather.enabled
    rd = RunDir.create(tmp_path / "run", cfg)
    assert runner.run(rd, cfg, init_state(cfg), 40, quiet=True).status == "done"
    p = record.record_window(
        tmp_path / "run", 20, 40, tmp_path / "o", tau=10, fine_groups=0, log=lambda _: None
    )
    fr = fm.read_frames(p)
    assert fr.has_drift and fr.drift.dtype == np.float16 and fr.drift.shape == (len(fr.uid), 3)
    assert np.isfinite(fr.drift.astype(np.float32)).all() and np.abs(fr.drift.astype(np.float32)).max() > 0
    assert fr.meta["weather"] and fr.meta["verification"]["verified"]
    s = load_checkpoint(rd.checkpoint_at(20), cfg)
    from alife import weather

    d = to_numpy(weather.agent_drift(s.wx, s.pos, s.buoy_cap, s.alive, cfg))[to_numpy(s.alive).astype(bool)]
    assert np.abs(fr.frame(0)["drift"].astype(np.float32) - d).max() <= 1e-3 * max(1.0, np.abs(d).max())
