"""The play loop sends the feature records (DESIGN §38.9, §39.8, §40.7): ``STRC`` at start and every tick,
``FSIG``
after the ``FOCS`` of a focused organism, the bond flow in ``FRMA``, the ``role`` colour mode and the optional
header blocks; and a world without the features is byte-identical in its header."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from alife import play, streamrec
from alife.config import dims
from alife.state import to_numpy
from play_helpers import decoded, run_paced, snapshots, uids, world

RUN = play.PlayOptions(rate=float("inf"), paused=False, fresh=True, unattended=True)
FEATURES = ("structures.enabled=true", "comm.enabled=true", "bonds.share_mode=1")


def test_a_world_without_the_features_has_the_header_it_always_had(tmp_path: Path, monkeypatch) -> None:
    snapshots(monkeypatch, 25)
    cfg = world("3d")
    _, _, _, out = run_paced(cfg, tmp_path / "run", 25, [])
    play_h = decoded(out).header["play"]
    assert not {"structures", "comm", "niche", "bonds"} & set(play_h)
    assert not set(streamrec.EXT_TAGS) & set(play_h["records"])
    assert decoded(out).header["color_modes"] == list(streamrec.COLOR_MODES)
    assert not [r for r in decoded(out).records if r.tag in streamrec.EXT_TAGS]


def test_the_feature_records_follow_the_state(tmp_path: Path, monkeypatch) -> None:
    snapshots(monkeypatch, 25)
    cfg = world("3d", *FEATURES)
    u = uids(cfg)[0]
    script = [[], [f"focus {u}"], [], [], []]
    _, res, _, out = run_paced(cfg, tmp_path / "run", 50, script)
    dec = decoded(out)
    d = dims(cfg)
    h = dec.header["play"]
    # header: tags in extension order, the blocks, the colour modes with `role`
    assert h["records"][-2:] == ["STRC", "FSIG"] and h["versions"]["STRC"] == 1
    assert h["structures"] == {
        "enabled": True,
        "cap": cfg.structures.cap,
        "cells": [d.fz, d.fg, d.fg],
        "cell": cfg.world.field_cell,
    }
    assert h["comm"]["w"] == cfg.comm.width and h["comm"]["control"] == "none"
    assert h["bonds"] == {"flow": True, "scale": float(cfg.bonds.share_rate)}
    assert dec.header["color_modes"] == [*streamrec.COLOR_MODES, "role"]
    # STRC: at start (0) and at every tick; the last is the final state's material
    strc = [r for r in dec.records if r.tag == "STRC"]
    assert [r.step for r in strc] == [0, 25, 50] and [r.seq for r in strc] == [0, 1, 2]
    mat = to_numpy(res.state.mat)
    idx, lev, _ = streamrec.quantise_cells(mat, cfg.structures.cap)
    assert np.array_equal(strc[-1].a["idx"], idx) and np.array_equal(strc[-1].a["level"], lev)
    assert (strc[-1].f["nx"], strc[-1].f["ny"], strc[-1].f["nz"]) == (d.fg, d.fg, d.fz)
    # FSIG follows every FOCS of the living organism, with the same uid
    tags = [r.tag for r in dec.records]
    focs = [i for i, t in enumerate(tags) if t == "FOCS" and dec.records[i].f["flags"] & 1]
    assert focs
    for i in focs:
        nxt = dec.records[i + 1]
        assert nxt.tag == "FSIG" and nxt.f["uid"] == dec.records[i].f["uid"] and nxt.f["w"] == cfg.comm.width
    # FRMA: the flag is on every frame of a bonds world, one int8 per drawn bond
    frames = [r for r in dec.records if r.tag == "FRMA"]
    assert frames and all(r.flags & streamrec.FLAG_BONDFLOW for r in frames)
    assert all(r.a["flow"].shape == (r.f["n_bonds"],) for r in frames)


def test_the_frame_flow_is_the_scaled_net_outflow_of_the_lower_row(tmp_path: Path, monkeypatch) -> None:
    """A frame written at the final step: each drawn bond's int8 is ``round(127·clip(−net/share_rate))`` of
    the
    lower row's ``bond_flow`` at that slot (positive: lower row to higher row)."""
    snapshots(monkeypatch, 25)
    cfg = world("3d", "bonds.share_mode=1", "bonds.share_rate=100")
    _, res, _, out = run_paced(cfg, tmp_path / "run", 50, [["every 1"]])
    dec = decoded(out)
    last = [r for r in dec.records if r.tag == "FRMA"][-1]
    assert last.step == 50
    s = res.state
    partner, alive = to_numpy(s.partner), to_numpy(s.alive)
    rows = np.arange(partner.shape[0])[:, None]
    drawn = (partner > rows) & alive[:, None] & alive[np.maximum(partner, 0)]
    ii, bb = np.nonzero(drawn)
    net = to_numpy(s.bond_flow)[ii, bb].astype(np.float64)
    want = np.rint(127.0 * np.clip(-net / cfg.bonds.share_rate, -1.0, 1.0)).astype(np.int8)
    assert np.array_equal(last.a["flow"], want)


def test_the_role_colour_mode() -> None:
    import mlx.core as mx

    from alife.config import load_config
    from alife.game.colors import role_colors
    from alife.state import init_state

    cfg = load_config(preset="tiny3d", overrides=["bonds.share_mode=1"])
    s = init_state(cfg)
    role = np.zeros(s.alive.shape[0], np.int8)
    role[:2] = [1, 2]
    calm = s._replace(role=mx.array(role), last_loss=mx.zeros_like(s.last_loss))
    colors = to_numpy(role_colors(calm))
    assert (
        colors[2] == (190 << 16) | (190 << 8) | 190
        and len({int(colors[0]), int(colors[1]), int(colors[2])}) == 3
    )
    hurt = to_numpy(role_colors(calm._replace(last_loss=mx.ones_like(s.last_loss))))
    assert (hurt == (255 << 16)).all()  # agents hurt last step are red in every mode
    plain = init_state(load_config(preset="tiny3d"))  # a state without roles is all grey
    assert (to_numpy(role_colors(plain)) == (190 << 16) | (190 << 8) | 190).all()


def test_a_world_without_bonds_refuses_the_role_colour_mode(tmp_path: Path, monkeypatch) -> None:
    """E5: the engine refuses an unknown control line (the mode is not in this world's header list) and the
    frames stay in the colour mode they were in."""
    snapshots(monkeypatch, 25)
    notes: list[str] = []

    class Noted(play.PlayPacer):
        def __init__(self, *a, **k) -> None:
            super().__init__(*a, **{**k, "note": notes.append})

    cfg = world("3d")
    _, _, _, out = run_paced(cfg, tmp_path / "run", 25, [["color role"]], pacer_class=Noted)
    assert any("unknown colour mode" in m and "role" in m for m in notes)
    assert {r.f["color_index"] for r in decoded(out).records if r.tag == "FRMA"} == {0}


def test_a_bonds_world_accepts_the_role_colour_mode(tmp_path: Path, monkeypatch) -> None:
    snapshots(monkeypatch, 25)
    cfg = world("3d", "bonds.share_mode=1")
    _, _, _, out = run_paced(cfg, tmp_path / "run", 25, [["color role"]])
    dec = decoded(out)
    idx = dec.header["color_modes"].index("role")
    assert idx == 7 and idx in {r.f["color_index"] for r in dec.records if r.tag == "FRMA"}


def test_fsig_hears_is_the_observer_channels_heard_byte(tmp_path: Path, monkeypatch) -> None:
    """With ``game.observe`` the FSIG heard half is ``Acts.ext.comm.heard`` of the step, byte for byte (it was
    quantised twice and saturated to 0 or 255), and ``audible`` is the channel's count."""
    snapshots(monkeypatch, 25)
    cfg = world("3d", "comm.enabled=true", "game.observe=true")
    seen: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    class Capture(play.PlayPacer):
        def after_step(self, state, out, t, social) -> None:
            super().after_step(state, out, t, social)
            if self.obs is not None and self.obs.comm_last is not None:
                heard, aud = (to_numpy(a) for a in self.obs.comm_last)
                seen[t] = (to_numpy(state.uid), heard, aud)

    us = uids(cfg)
    script = [[], ["every 1"], *[[f"focus {u}"] for u in us[:60]], [], []]
    _, _, _, out = run_paced(cfg, tmp_path / "run", 50, script, pacer_class=Capture)
    fsig = [r for r in decoded(out).records if r.tag == "FSIG" and r.f["flags"] & 1]
    assert fsig
    values: set[int] = set()
    for r in fsig:
        uid_row, heard, aud = seen[r.step]
        row = int(np.flatnonzero(uid_row == r.f["uid"])[0])
        assert np.array_equal(r.a["hears"], heard[row]), r.step
        assert r.f["audible"] == int(aud[row])
        values |= {int(v) for v in r.a["hears"]}
    assert values - {0, 255}, f"only saturated values {sorted(values)}: the quantisation is applied twice"
