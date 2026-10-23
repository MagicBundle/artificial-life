"""The storm tracker and the storm layer of a played world (DESIGN §33.3, ``storms.py``): the tracker moved
from ``analysis/w0.py`` and behaves as before, the layer's complete state makes a stopped and resumed play the
same play, and the files it writes (``observers/storms.jsonl``, ``observers/storms_<t>.json``) follow the
checkpoints through ``RunDir.truncate_after`` and ``_prune``."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from alife import runio, storms, streamrec
from alife import weather as W
from alife.analysis import w0
from alife.config import Config, apply_overrides
from alife.runio import RunDir
from alife.state import init_state
from alife.storms import Comp, Detection, StormLayer, Tracker
from alife.weather import START_SET
from play_decode import decode

N = 40
P40 = replace(START_SET, n=N)  # dx 2, L = 80, miss_grace 2, detect_every 25
DC_REF = 0.5


# --- synthetic scenes: detections of drifting blobs on the periodic grid ------------------------------------


def comp(label: int, rect: tuple[int, int, int, int], hemi: int = 0, zeta: float = 1.0) -> Comp:
    j0, i0, h, w = rect
    jj, ii = np.meshgrid(np.arange(j0, j0 + h) % N, np.arange(i0, i0 + w) % N, indexing="ij")
    area = h * w
    cx = (np.angle(np.exp(2j * np.pi * (ii + 0.5) / N).mean()) * N / (2 * np.pi) % N) * P40.dx
    cy = (np.angle(np.exp(2j * np.pi * (jj + 0.5) / N).mean()) * N / (2 * np.pi) % N) * P40.dx
    return Comp(
        label=label, hemi=hemi, x=float(cx), y=float(cy), area=area, r_eq=P40.dx * math.sqrt(area / math.pi),
        r_max=3.0, zeta_max_f0=zeta, wind_max=0.1, h1_min=30.0, dh1=1.0, rain=1.0, n_rain=5, rx=1.0, ry=1.0,
        ext_x=w * P40.dx, ext_y=h * P40.dx, flat=(jj * N + ii).ravel(),
    )  # fmt: skip


def det(t: int, items: list[tuple[tuple[int, int, int, int], int, float]]) -> Detection:
    lab = np.zeros((N, N), np.int32)
    comps = []
    for k, (rect, hemi, zeta) in enumerate(items, start=1):
        c = comp(k, rect, hemi, zeta)
        comps.append(c)
        lab.ravel()[c.flat] = k
    return Detection(t, lab, comps, [], len(comps), 0)


def scene(seed: int, ticks: int = 40) -> list[Detection]:
    """Blobs that drift east across the seam, grow, shrink, vanish for a tick, appear and disappear: births,
    continuations, coasting, splits, mergers and dissipation (the same generator was checked against the
    original W0 tracker on 300 scenes)."""
    rng = np.random.default_rng(seed)
    blobs = [
        [
            int(rng.integers(0, N)),
            int(rng.integers(0, N)),
            int(rng.integers(2, 6)),
            int(rng.integers(2, 8)),
            int(rng.integers(0, 2)),
            float(rng.uniform(0.2, 1.2)),
        ]
        for _ in range(int(rng.integers(2, 6)))
    ]
    out = []
    for k in range(1, ticks + 1):
        items = []
        for b in blobs:
            if rng.random() < 0.08:
                continue
            b[0] = (b[0] + int(rng.integers(-1, 2))) % N
            b[1] = (b[1] + int(rng.integers(0, 3))) % N
            b[2], b[3] = max(2, b[2] + int(rng.integers(-1, 2))), max(2, b[3] + int(rng.integers(-1, 2)))
            b[5] = float(np.clip(b[5] + rng.normal(0, 0.1), 0.1, 1.5))
            items.append(((b[0], b[1], b[2], b[3]), b[4], b[5]))
        if rng.random() < 0.1:
            blobs.append([int(rng.integers(0, N)), int(rng.integers(0, N)), 3, 3, 0, 0.9])
        if rng.random() < 0.1 and len(blobs) > 1:
            blobs.pop(int(rng.integers(0, len(blobs))))
        out.append(det(25 * k, items))
    return out


def jsonable(tick: storms.StormTick) -> str:
    return json.dumps([tick.step, tick.strm, tick.events, tick.line], sort_keys=True)


def run(layer: StormLayer, dets: list[Detection]) -> list[storms.StormTick]:
    return [layer.ingest(d) for d in dets]


# --- the moved tracker --------------------------------------------------------------------------------------


def test_the_study_driver_re_exports_the_tracker() -> None:
    for name in (
        "Frame",
        "read_frame",
        "Comp",
        "Detection",
        "detect",
        "Det",
        "Track",
        "Tracker",
        "min_image_np",
    ):
        assert getattr(w0, name) is getattr(storms, name), name
    assert w0.SPLIT_NEAR_TIE == storms.SPLIT_NEAR_TIE and w0._window is storms._window


def test_predict_is_the_least_squares_line_of_the_last_four_detections() -> None:
    tr = Tracker(P40)
    track = w0.Track(1, 0, [])
    for k, t in enumerate((25, 50, 75, 100, 125)):
        track.dets.append(
            w0.Det(t, 10.0, 10.0, 2.0 + 0.5 * k, 8.0 - 0.25 * k, 3.0, 9, 0, 1.0, 0.1, 30.0, 1.0, 1.0)
        )
    ux, uy = tr.predict(track, 225)  # last at 125 (k = 4): 4 detections 75..125 fit exactly
    assert (ux, uy) == pytest.approx((4.0 + 0.5 / 25 * 100, 7.0 - 0.25 / 25 * 100))
    one = w0.Track(2, 0, [track.dets[0]])
    assert tr.predict(one, 500) == (track.dets[0].ux, track.dets[0].uy)


def test_compacting_changes_no_result() -> None:
    """The layer drops the ended tracks and keeps 40 detections per track; the tracker's results (ids and the
    tracks that matter) are those of a tracker that keeps everything."""
    for seed in range(6):
        full, short = Tracker(P40), Tracker(P40)
        for d in scene(seed, 60):
            assert full.update(d) == short.update(d)
            short.compact(storms.KEEP_DETS)
            assert short.next_id == full.next_id
            a = {t.id: (t.missed, t.parent, t.hemi, len(t.dets) > 0) for t in full.active()}
            b = {t.id: (t.missed, t.parent, t.hemi, len(t.dets) > 0) for t in short.active()}
            assert a == b
            assert all(
                np.array_equal(x.last_flat, y.last_flat) and x.dets[-1] == y.dets[-1]
                for x, y in zip(full.active(), short.active(), strict=True)
            )
    assert all(len(t.dets) <= storms.KEEP_DETS for t in short.tracks)


# --- the layer: events, names, the STRM object --------------------------------------------------------------


def test_a_storm_is_named_at_zeta_half_a_f0_and_keeps_its_name() -> None:
    layer = StormLayer(P40, DC_REF, 25)
    rect = (10, 10, 4, 6)
    ticks = [
        layer.ingest(det(25 * (k + 1), [(rect, 0, z)])) for k, z in enumerate((0.3, 0.49, 0.5, 0.2, 0.1))
    ]
    kinds = [[e["kind"] for e in t.events] for t in ticks]
    assert kinds == [["born"], [], ["named"], [], []]
    assert [s["storms"][0]["named"] for s in (t.strm for t in ticks)] == [False, False, True, True, True]
    name = ticks[2].strm["storms"][0]["name"]
    assert name and ticks[4].strm["storms"][0]["name"] == name
    assert ticks[2].events[0]["name"] == name and ticks[2].events[0]["id"] == 1
    assert ticks[4].strm["storms"][0]["zeta"] == 0.5 and ticks[4].strm["storms"][0]["zeta_now"] == 0.1


def test_names_are_unique_among_live_storms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storms, "raw_house_names", lambda ids, salt: ["Same"])
    layer = StormLayer(P40, DC_REF, 25)
    tick = layer.ingest(det(25, [((2, 2, 4, 4), 0, 0.9), ((20, 20, 4, 4), 1, 0.9)]))
    names = [s["name"] for s in tick.strm["storms"]]
    assert names == ["Same", "Same 2"]


def test_the_generated_name_is_the_house_name_generator_on_the_storm_id() -> None:
    layer = StormLayer(P40, DC_REF, 25)
    tick = layer.ingest(det(25, [((2, 2, 4, 4), 0, 0.9)]))
    from alife.game.society import raw_house_names

    assert tick.strm["storms"][0]["name"] == raw_house_names(np.array([1]), storms.STORM_NAME_SALT)[0]


def test_born_split_merged_dissolved_events_and_the_ended_list() -> None:
    layer = StormLayer(P40, DC_REF, 25)
    a, b = (10, 10, 4, 6), (10, 18, 4, 4)
    first = layer.ingest(det(25, [(a, 0, 0.3), (b, 0, 0.3)]))
    assert [(e["kind"], e["id"], e["other"]) for e in first.events] == [("born", 1, 0), ("born", 2, 0)]
    layer.ingest(det(50, [(a, 0, 0.3), (b, 0, 0.3)]))
    merged = layer.ingest(det(75, [((10, 11, 4, 11), 0, 0.3)]))  # overlaps a by 5 columns and b by 4
    assert [(e["kind"], e["id"], e["other"]) for e in merged.events] == [("merged", 2, 1)]
    assert merged.strm["ended"] == [{"id": 2, "reason": "merged", "into": 1}]
    assert [s["id"] for s in merged.strm["storms"]] == [1]
    # a split: the right piece of a wide storm starts a child
    layer2 = StormLayer(P40, DC_REF, 25)
    big = (10, 10, 4, 12)
    for k in (1, 2, 3):
        layer2.ingest(det(25 * k, [(big, 0, 0.3)]))
    split = layer2.ingest(det(100, [((10, 10, 4, 8), 0, 0.3), ((10, 20, 4, 4), 0, 0.3)]))
    assert [(e["kind"], e["id"], e["other"]) for e in split.events] == [("split", 2, 1)]
    child = next(s for s in split.strm["storms"] if s["id"] == 2)
    assert child["parent"] == 1 and child["age"] == 0
    # dissipation after miss_grace (2) missed ticks: judged at the third; the age runs first to last detection
    gone = [layer2.ingest(det(125 + 25 * k, [])) for k in range(3)]
    assert [len(g.events) for g in gone] == [0, 0, 2]
    assert {(e["kind"], e["id"]) for e in gone[2].events} == {("dissolved", 1), ("dissolved", 2)}
    assert {e["id"]: e["age"] for e in gone[2].events} == {1: 75, 2: 0}
    assert [s["state"] for s in gone[0].strm["storms"]] == ["coasting", "coasting"]
    assert gone[2].strm["storms"] == [] and len(gone[2].strm["ended"]) == 2
    assert {(e["id"], e["reason"]) for e in gone[2].strm["ended"]} == {(1, "dissipated"), (2, "dissipated")}


def test_the_strm_object_wraps_positions_and_keeps_tracks_unwrapped() -> None:
    layer = StormLayer(P40, DC_REF, 25)
    ticks = [layer.ingest(det(25 * k, [((10, 28 + 3 * k, 4, 6), 0, 0.3)])) for k in range(1, 8)]
    last = ticks[-1].strm["storms"][0]
    assert 0 <= last["x"] < P40.L and 0 <= last["y"] < P40.L
    xs = [p[1] for p in last["track"]]
    assert xs == sorted(xs) and xs[-1] > P40.L, "the polyline is continuous across the seam"
    assert [p[0] for p in last["track"]] == [25 * k for k in range(1, 8)]
    assert [p[0] for p in last["pred"]] == [250, 500, 1000]
    assert all(0 <= x < P40.L and 0 <= y < P40.L for _, x, y in last["pred"])
    first = ticks[0].strm["storms"][0]
    assert first["pred"] == [] and first["age"] == 0 and last["age"] == 150 and last["state"] == "active"
    assert len(last["track"]) <= storms.TRACK_POINTS and last["ext"] == [12.0, 8.0]


def test_the_strm_object_is_capped_and_fits_its_record() -> None:
    items = [(((4 * (k % 9)) % N, 4 * (k // 9) % N, 2, 2), k % 2, 0.1 + 0.02 * k) for k in range(40)]
    # forty 2x2 storms on a 9x4 lattice of cells; the 32 with the highest peaks are listed, in id order
    layer = StormLayer(P40, DC_REF, 25)
    tick = layer.ingest(det(25, items))
    listed = [s["id"] for s in tick.strm["storms"]]
    assert len(listed) == 32 and listed == sorted(listed)
    assert set(listed) == set(range(9, 41)), "the lowest-peak storms (ids 1 to 8) are left out"
    rec = streamrec.encode_strm(step=25, seq=0, tick=25, obj=tick.strm)
    assert len(rec) < 8 + 16 + 16 + streamrec.STRM_JSON_CAP + 3


def test_ticks_without_storms_or_events_write_no_line() -> None:
    layer = StormLayer(P40, DC_REF, 25)
    assert layer.ingest(det(25, [])).line is None
    t1 = layer.ingest(det(50, [((10, 10, 4, 4), 0, 0.3)]))
    assert t1.line["step"] == 50 and [s["id"] for s in t1.line["storms"]] == [1]
    assert set(t1.line["storms"][0]) == {"id", "x", "y", "r", "zeta", "wind", "rain"}
    assert [e["kind"] for e in t1.line["events"]] == ["born"]
    t2 = layer.ingest(det(75, [((10, 10, 4, 4), 0, 0.3)]))
    assert t2.line is not None and t2.line["events"] == []


# --- the complete state -------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
def test_a_stopped_and_resumed_layer_is_the_uninterrupted_layer(seed: int) -> None:
    """Play to T, stop (the state through its JSON file), resume, play to T2: the ticks, the STRM objects, the
    events and the log lines equal an uninterrupted play, whichever tick the stop falls on."""
    dets = scene(seed, 50)
    ref = run(StormLayer(P40, DC_REF, 25), dets)
    for stop in (7, 19, 33):
        first = StormLayer(P40, DC_REF, 25)
        head = run(first, dets[:stop])
        state = json.loads(json.dumps(first.state()))
        resumed = StormLayer.restore(P40, DC_REF, state)
        tail = run(resumed, dets[stop:])
        assert [jsonable(t) for t in head + tail] == [jsonable(t) for t in ref], (seed, stop)
        assert json.dumps(resumed.state()) == json.dumps(run_state(dets)), (seed, stop)


def run_state(dets: list[Detection]) -> dict[str, Any]:
    layer = StormLayer(P40, DC_REF, 25)
    run(layer, dets)
    return layer.state()


def test_the_scenes_cover_every_kind_of_event() -> None:
    seen = {
        e["kind"]
        for seed in range(8)
        for t in run(StormLayer(P40, DC_REF, 25), scene(seed, 50))
        for e in t.events
    }
    assert seen == {"born", "named", "split", "merged", "dissolved"}


def test_the_state_carries_what_the_forty_detections_cannot() -> None:
    """A storm older than the stored detections keeps its first detection (age), its peak (so it cannot be
    un-named) and its name; a state written, read and written again is unchanged."""
    layer = StormLayer(P40, DC_REF, 25)
    zetas = [0.9] + [0.2] * 60
    for k, z in enumerate(zetas, start=1):
        tick = layer.ingest(det(25 * k, [((10, 10 + (k % 3), 4, 4), 0, z)]))
    s = tick.strm["storms"][0]
    assert (
        s["age"] == 25 * 60
        and s["zeta"] == 0.9
        and s["named"]
        and len(layer.tracker.tracks[0].dets) == storms.KEEP_DETS
    )
    state = json.loads(json.dumps(layer.state()))
    assert len(state["tracks"][0]["dets"]) == storms.KEEP_DETS and state["tracks"][0]["t_first"] == 25
    again = StormLayer.restore(P40, DC_REF, state)
    assert json.dumps(again.state()) == json.dumps(layer.state())
    nxt = det(25 * 62, [((10, 11, 4, 4), 0, 0.2)])
    assert jsonable(again.ingest(nxt)) == jsonable(layer.ingest(nxt))


def test_a_restored_layer_refuses_a_foreign_state() -> None:
    layer = StormLayer(P40, DC_REF, 25)
    layer.ingest(det(25, [((10, 10, 4, 4), 0, 0.3)]))
    state = layer.state()
    with pytest.raises(ValueError, match="version"):
        StormLayer.restore(P40, DC_REF, {**state, "version": 2})
    with pytest.raises(ValueError, match="parameters"):
        StormLayer.restore(replace(P40, miss_grace=3), DC_REF, state)
    with pytest.raises(ValueError, match="dc_ref"):
        StormLayer(P40, 0.0, 25)


# --- the files ----------------------------------------------------------------------------------------------


def _at_step(s, step: int):
    return s._replace(step=mx.array(step, dtype=s.step.dtype))


@pytest.fixture
def world(micro_cfg: Config, tmp_path: Path):
    cfg = apply_overrides(micro_cfg, ["run.keep_checkpoints=2"])
    return cfg, RunDir.create(tmp_path / "w", cfg), init_state(cfg)


def _play(layer: StormLayer, rd: RunDir, dets: list[Detection]) -> list[storms.StormTick]:
    ticks = run(layer, dets)
    for t in ticks:
        layer.log(rd, t)
    return ticks


def test_every_play_state_file_follows_its_checkpoint(world) -> None:
    """The kin thread table (G3b) is handled like the tracker state: pruned with its checkpoint, cut by
    ``truncate_after``, written and read by the generic ``write_play_state`` / ``read_play_state``."""
    cfg, rd, s = world
    rd.save_checkpoint(_at_step(s, 0), cfg)
    for step in (0, 100, 150):
        for name in runio.PLAY_STATE_NAMES:
            rd.write_play_state(name, step, {"name": name, "step": step})
    rd.save_checkpoint(_at_step(s, 100), cfg)  # 150 has no checkpoint: its files are orphans
    for name in runio.PLAY_STATE_NAMES:
        assert [st for st, _ in rd.play_state_files(name)] == [0, 100], name
        assert rd.read_play_state(name, 100) == {"name": name, "step": 100}
        assert rd.read_play_state(name, 7) is None
    assert rd.storms_state_files() == rd.play_state_files("storms")
    assert rd.read_storms_state(100) == rd.read_play_state("storms", 100)
    assert rd.storms_state_path(100) == rd.observers_dir / "storms_0000000100.json"
    assert rd.play_state_path("kinthreads", 100) == rd.observers_dir / "kinthreads_0000000100.json"
    rd.save_checkpoint(_at_step(s, 200), cfg)
    rd.save_checkpoint(_at_step(s, 300), cfg)  # keep 2: the checkpoints at 0 (kept), 200, 300 remain
    for name in runio.PLAY_STATE_NAMES:
        assert [st for st, _ in rd.play_state_files(name)] == [0], f"{name}: 100 went with its checkpoint"
    for name in runio.PLAY_STATE_NAMES:
        rd.write_play_state(name, 200, {"x": 1})
        rd.write_play_state(name, 300, {"x": 1})
    rd.truncate_after(200)
    for name in runio.PLAY_STATE_NAMES:
        assert [st for st, _ in rd.play_state_files(name)] == [0, 200], name
    rd.play_state_path("kinthreads", 200).write_text("[1, 2]")
    with pytest.raises(ValueError, match="not a JSON object"):
        rd.read_play_state("kinthreads", 200)
    with pytest.raises(ValueError, match="unknown play state"):
        rd.write_play_state("nope", 0, {})
    with pytest.raises(ValueError, match="unknown play state"):
        rd.play_state_files("nope")


def test_the_log_and_the_states_follow_the_checkpoints(world) -> None:
    cfg, rd, s = world
    layer = StormLayer(P40, DC_REF, 25)
    dets = scene(2, 30)
    ticks = _play(layer, rd, dets[:12])  # steps 25 .. 300
    rd.save_checkpoint(_at_step(s, 0), cfg)
    layer.save(rd, 0)
    layer.save(rd, 100)
    rd.save_checkpoint(_at_step(s, 100), cfg)
    assert [st for st, _ in rd.storms_state_files()] == [0, 100]
    steps = [r["step"] for r in rd.storms_lines()]
    assert steps == sorted(steps) and set(steps) <= {t.step for t in ticks}
    # an orphan state (a crash between its write and the checkpoint) is pruned with the next save
    layer.save(rd, 150)
    rd.save_checkpoint(_at_step(s, 200), cfg)
    assert [st for st, _ in rd.storms_state_files()] == [0, 100], "150 had no checkpoint; 200 has no state"
    rd.save_checkpoint(_at_step(s, 300), cfg)  # keep 2: the checkpoints at 0 (kept), 200, 300 remain
    assert [st for st, _ in rd.checkpoints()] == [0, 200, 300]
    assert [st for st, _ in rd.storms_state_files()] == [0], "100 went with its checkpoint"
    # truncate_after cuts the log to the checkpoint's step and removes the later states
    layer.save(rd, 300)
    rd.truncate_after(200)
    assert [st for st, _ in rd.storms_state_files()] == [0]
    assert all(r["step"] <= 200 for r in rd.storms_lines()) and rd.storms_lines()
    n = len(rd.storms_lines())
    rd.truncate_after(200)  # idempotent
    assert len(rd.storms_lines()) == n
    rd.truncate_after(0)
    assert rd.storms_lines() == [] and rd.storms_path.is_file()


def test_truncate_after_leaves_a_directory_without_storms_alone_and_checks_the_log_first(world) -> None:
    cfg, rd, s = world
    rd.save_checkpoint(_at_step(s, 0), cfg)
    rd.truncate_after(0)
    assert not rd.storms_path.exists() and not rd.observers_dir.exists()
    rd.append_storms([{"step": 25, "storms": [], "events": []}])
    rd.append_storms([{"step": 50}, {"step": 75}])
    with rd.storms_path.open("a") as f:
        f.write('{"step": 100')  # a torn last line is dropped
    assert [r["step"] for r in rd.storms_lines()] == [25, 50, 75]
    rd.truncate_after(60)
    assert rd.storms_path.read_text().count("\n") == 2 and rd.storms_path.read_text().endswith("}\n")
    with rd.storms_path.open("a") as f:
        f.write('{"step": 70}\n{"nostep": 1}\n')
    with pytest.raises(ValueError, match="int 'step'"):
        rd.truncate_after(60)
    assert '"nostep"' in rd.storms_path.read_text(), "a log that cannot be parsed is left as it was"
    with pytest.raises(ValueError, match="must increase"):
        rd.append_storms([{"step": 80}, {"step": 80}])
    with pytest.raises(ValueError, match="int 'step'"):
        rd.append_storms([{"x": 1}])
    rd.append_storms([])


def test_a_state_saved_at_a_checkpoint_remembers_the_last_tick(world) -> None:
    """The checkpoint's step is not the layer's step: after a resume the first ``STRM`` still says which tick
    it shows (``detect_step``), and the old files without ``tick_step`` read as before."""
    _, rd, _ = world
    layer = StormLayer(P40, DC_REF, 25)
    run(layer, scene(3, 6))  # the last tick is at 150
    layer.save(rd, 163)  # a checkpoint between two ticks
    back, why = StormLayer.open(rd, P40, DC_REF, 163)
    assert why is None and back.step == 150 and back.state()["tick_step"] == 150
    state = json.loads(rd.storms_state_path(163).read_text())
    assert state["step"] == 163 and state["tick_step"] == 150
    del state["tick_step"]
    old = StormLayer.restore(P40, DC_REF, state)
    assert old.step == 163, "a state without tick_step is read as before"


def test_open_restores_the_state_of_the_checkpoint_or_starts_afresh_and_says_so(world) -> None:
    _, rd, _ = world
    layer = StormLayer(P40, DC_REF, 25)
    run(layer, scene(3, 8))
    layer.save(rd, 200)
    back, why = StormLayer.open(rd, P40, DC_REF, 200)
    assert why is None and not back.restarted and json.dumps(back.state()) == json.dumps(layer.state())
    for step, match in ((300, "no tracker state"), (0, "no tracker state")):
        fresh, why = StormLayer.open(rd, P40, DC_REF, step)
        assert why and match in why and fresh.tracker.next_id == 1 and fresh.step == step
        assert fresh.restarted is (step > 0), "a world that never ran has nothing to renumber"
    other, why = StormLayer.open(rd, P40, DC_REF, 200, tick=50)
    assert why and "another step or tick" in why and other.restarted
    bad, why = StormLayer.open(rd, replace(P40, miss_grace=3), DC_REF, 200)
    assert why and "other weather parameters" in why and bad.restarted
    rd.storms_state_path(200).write_text("{not json")
    junk, why = StormLayer.open(rd, P40, DC_REF, 200)
    assert why and "unusable" in why and junk.restarted
    rd.storms_state_path(200).write_text("[1, 2]")
    assert StormLayer.open(rd, P40, DC_REF, 200)[1] is not None


# --- the maps and the weather -------------------------------------------------------------------------------

PV = replace(START_SET, n=80, gamma=0.0, k_E=0.0, tau_rad=0.0, seed_rate=0.0, p_th=0.0, r_min=1, a_min=12)


def vortex_ticks(n_ticks: int) -> tuple[StormLayer, list[storms.StormTick], W.WeatherState]:
    """The balanced vortex of the W0 spec (0.55 f0 on the north pole row), unforced and dry, stepped for
    ``n_ticks`` detection periods; a storm that is named from its first tick."""
    step = W.make_step(PV, compile=False)
    fo = W.default_forcing(PV)
    ws, key = w0.balanced_vortex(PV), mx.random.key(0)
    layer = StormLayer(PV, DC_REF, 25)
    ticks = []
    for t in range(1, 25 * n_ticks + 1):
        ws, _ = step(ws, fo, key)
        if t % 25 == 0:
            ticks.append(layer.tick_at(ws, t))
    return layer, ticks, ws


@pytest.fixture(scope="module")
def vortex():
    return vortex_ticks(4)


def test_tick_at_reads_the_weather_detects_and_names(vortex) -> None:
    layer, ticks, _ = vortex
    assert [t.step for t in ticks] == [25, 50, 75, 100]
    (born,) = [e for t in ticks for e in t.events if e["kind"] == "born"]
    assert born["id"] == 1 and born["step"] == 25 and born["hemi"] == 0
    assert (born["x"], born["y"]) == pytest.approx((PV.L / 2, PV.L / 4), abs=0.5)
    (named,) = [e for t in ticks for e in t.events if e["kind"] == "named"]
    assert named["id"] == 1 and named["name"]
    s = ticks[-1].strm["storms"][0]
    assert s["named"] and 0.5 <= s["zeta"] < 0.56 and s["age"] == 75 and len(s["track"]) == 4
    assert s["pred"] and s["state"] == "active" and s["r"] > 5
    assert json.dumps(layer.state())  # the state of a real run is plain JSON


def test_tick_at_equals_the_manual_pipeline_and_ingest(vortex) -> None:
    _, _, ws = vortex
    manual = StormLayer(PV, DC_REF, 25)
    # one tick from the final state, by the layer and by hand
    layer = StormLayer(PV, DC_REF, 25)
    a = layer.tick_at(ws, 100)
    b = manual.ingest(w0.detect(w0.read_frame(ws, PV, 100), PV))
    assert (a.step, a.strm, a.events, a.line) == (b.step, b.strm, b.events, b.line)
    assert a.planes is not None and b.planes is None


def test_current_is_a_tick_without_a_detection(vortex) -> None:
    """``current`` (the records of a step that is not a tick: at start and on ``idnt``) gives the planes of
    the weather and the ``STRM`` object of the tracks as they are, and changes nothing in the tracker."""
    layer, ticks, ws = vortex
    before = json.dumps(layer.state(), sort_keys=True)
    cur = layer.current(ws, 100)
    assert json.dumps(layer.state(), sort_keys=True) == before
    assert cur.events == [] and cur.line is None and cur.step == 100
    assert {k: v.tolist() for k, v in cur.planes.items()} == {
        k: v.tolist() for k, v in ticks[-1].planes.items()
    }
    assert cur.strm == ticks[-1].strm  # the same tracks; detect_step is the last tick (100)
    later = layer.current(
        ws, 130
    )  # five detections' worth of steps on: the tracks age, the detection does not
    assert (
        later.strm["detect_step"] == 100
        and later.strm["storms"][0]["age"] == ticks[-1].strm["storms"][0]["age"] + 30
    )
    assert later.strm["storms"][0]["state"] == "coasting"
    assert streamrec.encode_strm(step=130, seq=0, tick=25, obj=later.strm)  # and it encodes


def test_the_map_planes_are_the_weather_quantised(vortex) -> None:
    _, ticks, ws = vortex
    planes = ticks[-1].planes
    assert planes is not None
    n = PV.n
    assert {k: v.shape for k, v in planes.items()} == {
        "P": (n, n), "Dc": (n, n), "zeta": (n, n), "Ts": (n, n), "winds": (4, 40, 40),
    }  # fmt: skip
    fr = w0.read_frame(ws, PV, 100)
    zc = w0.corners_to_centres(fr.zeta.astype(np.float64))
    want = np.rint(127 * np.clip(zc / streamrec.ZETA_REF, -1, 1)).astype(np.int8)
    assert np.array_equal(planes["zeta"], want)
    j, i = np.unravel_index(np.argmax(planes["zeta"]), planes["zeta"].shape)
    assert planes["zeta"].max() >= 60 and (i * PV.dx, j * PV.dx) == pytest.approx(
        (PV.L / 2, PV.L / 4), abs=4.0
    )
    ts = np.array(W.warm_pools(PV, mx.array(100, dtype=mx.int32)))
    assert np.array_equal(planes["Ts"], np.rint(255 * np.clip(ts, 0, 1)).astype(np.uint8))
    w = np.stack([fr.u1, fr.v1, fr.u2, fr.v2]).astype(np.float64)
    coarse = w.reshape(4, 40, 2, 40, 2).mean(axis=(2, 4))
    assert np.array_equal(
        planes["winds"], np.rint(127 * np.clip(coarse / streamrec.U_REF, -1, 1)).astype(np.int8)
    )
    assert planes["winds"].any() and planes["P"].dtype == planes["Dc"].dtype == planes["Ts"].dtype == np.uint8
    # and the record built from them decodes
    data = streamrec.play_header_bytes(
        streamrec.legacy_header(
            dims=2, size=PV.L, height=0.0, n=64, bond_slots=4, radius=1.0, field_cell=4.0, fg=20, fz=1, nch=3,
            step=100, run=None, config_hash="0" * 64, source="preset",
        ),
        streamrec.header_play(
            records=["MAP1"], weather=streamrec.weather_block(enabled=True, dc_ref=DC_REF),
            labels={"intent_version": 2, "arch": [""], "intent": [""]}, energy_cap=100,
        ),
    ) + streamrec.encode_map1(step=100, seq=0, planes=planes, dc_ref=DC_REF)  # fmt: skip
    rec = decode(data).records[0]
    assert rec.tag == "MAP1" and np.array_equal(rec.a["zeta"], planes["zeta"])


def test_the_real_weather_resumes_the_same_play(tmp_path: Path) -> None:
    """The same stop-and-resume equality on the output of the weather model: the layer's state file at tick 2,
    a new layer, ticks 3 and 4 equal the uninterrupted layer's, STRM bytes included."""
    step = W.make_step(PV, compile=False)
    fo = W.default_forcing(PV)
    ws, key = w0.balanced_vortex(PV), mx.random.key(0)
    whole, split = StormLayer(PV, DC_REF, 25), StormLayer(PV, DC_REF, 25)
    got = []
    for t in range(1, 101):
        ws, _ = step(ws, fo, key)
        if t % 25:
            continue
        a = whole.tick_at(ws, t)
        if t == 50:
            state = json.loads(json.dumps(split.state()))
            split = StormLayer.restore(PV, DC_REF, state)
        b = split.tick_at(ws, t)
        sa = streamrec.encode_strm(step=t, seq=0, tick=25, obj=a.strm)
        sb = streamrec.encode_strm(step=t, seq=0, tick=25, obj=b.strm)
        got.append((jsonable(a) == jsonable(b), sa == sb, a.planes["P"].tobytes() == b.planes["P"].tobytes()))
        if t == 25:
            split = StormLayer.restore(PV, DC_REF, json.loads(json.dumps(whole.state())))
    assert all(all(row) for row in got) and len(got) == 4


def test_dc_reference_is_the_high_percentile_and_never_zero() -> None:
    dc = np.zeros((160, 160))
    assert storms.dc_reference(dc) == storms.DC_REF_FLOOR
    dc.ravel()[:200] = np.linspace(1.0, 5.0, 200)
    assert storms.dc_reference(dc) == pytest.approx(float(np.percentile(dc, 99.9)))
    only = np.zeros((160, 160))
    only[3, 3] = 4.0
    assert storms.dc_reference(only) == 4.0, (
        "a single cloud cell: the percentile is zero, the maximum is used"
    )


# --- the chronicle's input ----------------------------------------------------------------------------------


def test_the_layers_events_feed_the_chronicle_and_the_log_rebuilds_them() -> None:
    """The events of ``StormLayer`` are the ``Chronicler``'s input, and the lines of ``storms.jsonl`` give
    back the same chronicle (the backlog of a re-opened world), whatever the scene."""
    from alife.game.chronicle import Chronicler, rebuild_backlog
    from alife.streamrec import check_event

    seen: set[str] = set()
    for seed in range(8):
        layer, chron = StormLayer(P40, DC_REF, 25), Chronicler(0x686F757365)
        lines, live = [], []
        for d in scene(seed, 50):
            tick = layer.ingest(d)
            detected = [d["id"] for d in tick.line["storms"]] if tick.line is not None else []
            live += chron.from_storms(tick.step, tick.events, detected)
            if tick.line is not None:
                lines.append(json.loads(json.dumps(tick.line)))
        for e in live:
            check_event(e)
            seen.add(e["text"].split()[0] if e["kind"] != 8 else "named")
        assert rebuild_backlog([], lines, [], house_salt=0x686F757365, limit=10_000) == live, seed
    assert {"Disturbance", "Storm", "named"} <= seen


def test_the_sea_plane_carries_the_shade_when_the_world_has_one(vortex) -> None:
    """``T_s`` of ``MAP1`` is the map the engine forces the weather with: the warm pools times the shade
    (DESIGN §33.1.7). Without ``t_mult`` it is the bare pool map (the W1 expressions), for ``tick_at`` and
    ``current`` alike, and the shade reaches the plane only (the tracker input is the detection)."""
    _, _, ws = vortex
    shade = 1.0 - 0.05 * mx.broadcast_to(mx.linspace(0.0, 1.0, PV.n)[None, :], (PV.n, PV.n))
    warm = W.warm_pools(PV, mx.array(100, dtype=mx.int32))

    def byte(a: mx.array) -> np.ndarray:
        return np.rint(255 * np.clip(np.array(a), 0, 1)).astype(np.uint8)

    bare, shaded = byte(warm), byte(warm * shade)
    assert not np.array_equal(bare, shaded), "the test shade must move a byte"
    for read in (lambda lay, **kw: lay.tick_at(ws, 100, **kw), lambda lay, **kw: lay.current(ws, 100, **kw)):
        none, withm = read(StormLayer(PV, DC_REF, 25)), read(StormLayer(PV, DC_REF, 25), t_mult=shade)
        assert np.array_equal(none.planes["Ts"], bare)
        assert np.array_equal(withm.planes["Ts"], shaded)
        for k in ("P", "Dc", "zeta", "winds"):
            assert np.array_equal(none.planes[k], withm.planes[k])
        assert none.strm == withm.strm and none.events == withm.events
