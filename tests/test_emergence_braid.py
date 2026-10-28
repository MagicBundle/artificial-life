"""Block C of the emergence suite (DESIGN §44.4): braid words and the Burau entropy bound against known
answers, the coordination index, groups that restart inside a window, and winding numbers. CPU, numpy only."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from alife.analysis.emergence import braid, winding
from alife.analysis.emergence import frames as fm
from alife.analysis.emergence.__main__ import main


def stir(gens: list[tuple[int, int]], n: int, frames: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Tracks realising the generator list: the strands in slots ``j, j+1`` swap with a y offset of sign
    ``s``."""
    order = list(range(n))
    xs, ys = [], []

    def snap() -> tuple[np.ndarray, np.ndarray]:
        x = np.zeros(n)
        for slot, s in enumerate(order):
            x[s] = slot
        return x, np.zeros(n)

    x, y = snap()
    xs.append(x)
    ys.append(y)
    for j, sign in gens:
        a, b = order[j], order[j + 1]
        for f in range(1, frames + 1):
            u = f / frames
            x0, y0 = snap()
            x0[a], x0[b] = j + u, j + 1 - u
            off = math.sin(math.pi * u) * 0.3 * sign
            y0[a], y0[b] = off, -off
            xs.append(x0)
            ys.append(y0)
        order[j], order[j + 1] = b, a
    return np.array(xs), np.array(ys)


GENS = [(0, +1), (1, -1)] * 10


@pytest.mark.parametrize("frames", [40, 4])
def test_word_recovery(frames: int) -> None:
    x, y = stir(GENS, 3, frames)
    w, _ = braid.word_from_tracks(x, y)
    assert w == GENS
    # the mirror image (y -> -y) gives the inverse signs
    wm, _ = braid.word_from_tracks(x, -y)
    assert wm == [(j, -s) for j, s in GENS]


def test_known_entropy_and_trivial_words() -> None:
    m = np.eye(3)
    braid.apply_generator(m, 0, +1)
    braid.apply_generator(m, 1, -1)
    # the matrix product B(s2^-1) B(s1); the leading eigenvalue is (3 + sqrt 5) / 2
    assert max(abs(np.linalg.eigvals(m))) == pytest.approx((3 + 5**0.5) / 2, abs=1e-9)
    x, y = stir(GENS, 3)
    w, _ = braid.word_from_tracks(x, y)
    h = braid.log_growth(w, 3) / 10  # ten repeats
    assert h == pytest.approx(math.log((3 + 5**0.5) / 2), rel=0.02)
    assert braid.log_growth([(0, 1), (0, -1)], 3) == pytest.approx(0.0, abs=1e-12)  # sigma sigma^-1
    # sigma_1^2 is reducible: spectral radius 1, polynomial norm growth, so h_T falls like log T / T
    sq = np.eye(3)
    for _ in range(2):
        braid.apply_generator(sq, 0, +1)
    assert max(abs(np.linalg.eigvals(sq))) == pytest.approx(1.0, abs=1e-9)
    assert braid.log_growth([(0, 1)] * 400, 3) / 400 < 0.02


def test_rigid_rotation_entropy_rate_falls_with_time() -> None:
    n = 8
    r = np.linspace(0.5, 1.5, n)
    out = []
    for turns in (1, 10):
        th = np.linspace(0, 2 * np.pi * turns, 400 * turns)
        x, y = np.cos(th)[:, None] * r, np.sin(th)[:, None] * r
        s = braid.braid_stats(x, y, 400 * turns)
        out.append(s["h_step"])
    assert out[1] < 0.3 * out[0]


def test_overflow_safety_and_undersampling() -> None:
    n = 6
    word = [(j % (n - 1), 1 if (j // (n - 1)) % 2 == 0 else -1) for j in range(60_000)]
    word = [(i, s) for i, s in word]
    lg = braid.log_growth(word, n)
    assert math.isfinite(lg) and lg >= 0
    x, y = stir(GENS, 3, frames=1)  # one frame per generator: still a valid word, flagged as coarse
    w, under = braid.word_from_tracks(x, y)
    assert len(w) >= 1 and 0.0 <= under <= 1.0


def brownian(rng: np.random.Generator, n: int, t: int, d: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    p = rng.random((n, 2)) * np.array([n * 0.2, 1.0])
    xs, ys = [], []
    for _ in range(t):
        p = p + rng.normal(0, d, p.shape)
        xs.append(p[:, 0].copy())
        ys.append(p[:, 1].copy())
    return np.array(xs), np.array(ys)


def test_brownian_group_equals_its_surrogate() -> None:
    rng = np.random.default_rng(1)
    x, y = brownian(rng, 20, 1500)
    for kind in ("increment", "position"):
        c = braid.coordination(x, y, 1500, rng, n_surr=20, kind=kind)
        assert abs(c["h_step"] - c["h_surr"]) < 3 * c["h_surr_sd"]


def test_coordinated_flow_has_negative_coordination() -> None:
    rng = np.random.default_rng(2)
    n, t = 10, 200
    # a flock: every strand follows one common random walk, with a small individual jitter
    common = np.cumsum(rng.normal(0, 1.0, (t, 2)), axis=0)
    base = rng.uniform(0, 3, (n, 2))
    x = base[:, 0] + common[:, :1] + rng.normal(0, 0.05, (t, n))
    y = base[:, 1] + common[:, 1:] + rng.normal(0, 0.05, (t, n))
    c = braid.coordination(x, y, t, rng, n_surr=20)
    assert c["coord"] < -0.5 and c["contrast_step"] < 0 and c["n_cross"] < c["cross_surr"]
    # independent strands with the same step statistics are not distinguished from their surrogates
    xi, yi = brownian(rng, n, t, 1.0)
    ci = braid.coordination(xi, yi, t, rng, n_surr=20)
    assert abs(ci["h_step"] - ci["h_surr"]) < 3 * ci["h_surr_sd"]


def test_surrogate_validation() -> None:
    x, y = brownian(np.random.default_rng(3), 5, 30)
    with pytest.raises(ValueError):
        braid.surrogate(x, y, np.random.default_rng(0), "nope")
    sx, _ = braid.surrogate(x, y, np.random.default_rng(0))
    assert sx.shape == x.shape and np.allclose(sx[0], x[0])


# --- groups restart inside the window ---


def swarm_frames(
    rng: np.random.Generator,
    *,
    nf: int = 49,
    n_bodies: int = 30,
    size: int = 10,
    life: int = 20,
    bs: int = 160.0,
) -> fm.Frames:
    """Bodies of ``size`` organisms jittering about fixed centres; every organism dies after ``life`` frames
    and is replaced by a newcomer (a fresh uid), so no group survives a window but a block start can form
    it."""
    L = 160.0
    centres = rng.uniform(20, L - 20, (n_bodies, 2))
    n = n_bodies * size
    uid = np.arange(n, dtype=np.int64)
    birth = rng.integers(0, life, n)
    pos = centres.repeat(size, 0) + rng.normal(0, 0.5, (n, 2))
    body = np.arange(n_bodies, dtype=np.int32).repeat(size)
    ps, us, bd = [], [], []
    nxt = n
    for f in range(nf):
        dead = (f - birth) >= life
        for i in np.where(dead)[0]:
            uid[i] = nxt
            nxt += 1
            birth[i] = f
        pos = pos + rng.normal(0, 0.15, pos.shape)
        ps.append(np.column_stack([pos, np.zeros(n)]).copy())
        us.append(uid.copy())
        bd.append(body.copy())
    allp = np.concatenate(ps)
    return fm.Frames(
        meta={"size": L, "height": 80.0},
        step=np.arange(nf, dtype=np.int64) * 100,
        off=np.arange(nf + 1, dtype=np.int64) * n,
        uid=np.concatenate(us),
        row=np.zeros(len(allp), np.int32),
        pos=fm.quantise(allp, L, 80.0),
        clade=np.zeros(len(allp), np.int32),
        body=np.concatenate(bd),
    )


def test_groups_restart_at_each_block_and_report_skips() -> None:
    fr = swarm_frames(np.random.default_rng(4))
    r = braid.analyse_blocks(fr, n_blocks=4, n_surr=5)
    assert r["sub_window_frames"] == 12
    assert all(p["groups"] > 0 for p in r["per_block"])  # every block forms its own groups
    assert len(r["blocks"]) == 4
    # followed across the whole window almost nobody survives: one group over 48 frames has too few strands
    tr = braid.tracks_of(fr, braid.groups_at(fr, 0)[0][0], 0, 48)
    assert tr == "survivors"


def test_groups_too_large_or_spread_are_skipped_with_reasons() -> None:
    fr = swarm_frames(np.random.default_rng(5), n_bodies=3, size=300)
    g, sk = braid.groups_at(fr, 0)
    assert g == [] and sk["large"] == 3
    g, sk = braid.groups_at(swarm_frames(np.random.default_rng(6), size=5), 0)
    assert g == [] and sk["small"] == 30
    fr = swarm_frames(np.random.default_rng(7), n_bodies=1, size=12, life=10_000)
    pos = fr.positions(0)
    pos[:6, 0] += 70.0  # a body spread over more than L/3
    fr.pos[fr.span(0)] = fm.quantise(pos, 160.0, 80.0)
    assert braid.tracks_of(fr, np.sort(fr.uid[fr.span(0)]), 0, 3) == "extent"


def test_torus_wrap_is_unwrapped_about_the_centroid() -> None:
    fr = swarm_frames(np.random.default_rng(8), n_bodies=1, size=12, life=10_000)
    pos = fr.positions(0)
    pos[:, 0] = np.mod(pos[:, 0] - pos[:, 0].mean() + 0.0, 160.0)  # the body sits across the seam
    for i in range(fr.n_frames):
        s = fr.span(i)
        pos = fr.positions(i)
        pos[:, 0] = np.mod(pos[:, 0] - 80.0 + 0.0, 160.0)
        fr.pos[s] = fm.quantise(pos, 160.0, 80.0)
    tr = braid.tracks_of(fr, np.sort(fr.uid[fr.span(0)]), 0, 5)
    assert not isinstance(tr, str)
    assert (tr[0].max() - tr[0].min()) < 10.0


def test_analyse_blocks_needs_enough_frames() -> None:
    with pytest.raises(ValueError):
        braid.analyse_blocks(swarm_frames(np.random.default_rng(9), nf=5), n_blocks=8)


# --- winding ---


def orbit_tracks(turns: float, size: float = 160.0, n: int = 200) -> list[winding.StormTrack]:
    t = np.arange(n, dtype=np.int64) * 100
    th = np.linspace(0, 2 * np.pi * turns, n)
    cx, cy = 80.0, 80.0
    a = winding.StormTrack(
        1, t, np.mod(cx + 15 * np.cos(th), size), np.mod(cy + 15 * np.sin(th), size), np.full(n, 5.0)
    )
    b = winding.StormTrack(
        2, t, np.mod(cx - 15 * np.cos(th), size), np.mod(cy - 15 * np.sin(th), size), np.full(n, 5.0)
    )
    return [a, b]


def test_pair_winding_counts_turns_exactly() -> None:
    a, b = orbit_tracks(3.0)
    assert winding.pair_winding(a, b, 160.0)["w"] == pytest.approx(3.0, abs=1e-9)
    assert winding.pair_winding(b, a, 160.0)["w"] == pytest.approx(
        3.0, abs=1e-9
    )  # the vector flips, the turn does not
    ra = winding.StormTrack(1, a.t, a.x, 160.0 - a.y, a.r)
    rb = winding.StormTrack(2, b.t, b.x, 160.0 - b.y, b.r)
    assert winding.pair_winding(ra, rb, 160.0)["w"] == pytest.approx(
        -3.0, abs=1e-9
    )  # reflected: opposite sense


def test_zonal_circuit_and_separation_drop() -> None:
    t = np.arange(0, 1001, 10, dtype=np.int64)
    x = np.mod(np.linspace(0, 160.0, len(t)) + 5.0, 160.0)
    tr = winding.StormTrack(1, t, x, np.full(len(t), 40.0), np.full(len(t), 5.0))
    assert winding.zonal_circuits(tr, 160.0) == 1
    back = winding.StormTrack(2, t, np.mod(5.0 - np.linspace(0, 330.0, len(t)), 160.0), tr.y, tr.r)
    assert winding.zonal_circuits(back, 160.0) == -2
    far = winding.StormTrack(3, t, np.mod(tr.x + 80.0, 160.0), np.mod(tr.y + 80.0, 160.0), tr.r)  # 113 apart
    assert winding.pair_winding(tr, far, 160.0)["ticks"] == 0


def test_storm_braid_and_analyse() -> None:
    n = 60
    t = np.arange(n, dtype=np.int64) * 50
    rng = np.random.default_rng(10)
    tracks = []
    for i in range(4):
        tracks.append(
            winding.StormTrack(
                i + 1,
                t,
                60 + 8 * i + np.cumsum(rng.normal(0, 1.0, n)),
                60 + np.cumsum(rng.normal(0, 1.0, n)),
                np.full(n, 4.0),
            )
        )
    sb = winding.storm_braid(tracks, 160.0)
    assert len(sb) == 1 and sb[0]["storms"] == 4
    d = winding.analyse(tracks, 160.0)
    assert d["status"] == "exploratory" and d["storms"] == 4 and d["organisms"] is None
    assert winding.storm_braid(tracks[:2], 160.0) == []


def test_organism_winding_around_a_storm() -> None:
    size, nf = 160.0, 40
    ang0 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    ps, us = [], []
    for f in range(
        nf
    ):  # twelve organisms circling a stationary storm at 0.5 turn per ten frames, plus one far off
        ang = ang0 + 2 * np.pi * 0.05 * f
        p = np.column_stack([80 + 6 * np.cos(ang), 80 + 6 * np.sin(ang)])
        p = np.vstack([p, [[20.0, 20.0]]])
        ps.append(p)
        us.append(np.arange(13, dtype=np.int64))
    allp = np.concatenate(ps)
    fr = fm.Frames(
        meta={"size": size, "height": 0.0},
        step=np.arange(nf, dtype=np.int64) * 100,
        off=np.arange(nf + 1, dtype=np.int64) * 13,
        uid=np.concatenate(us),
        row=np.zeros(len(allp), np.int32),
        pos=fm.quantise(allp, size),
        clade=np.zeros(len(allp), np.int32),
        body=np.full(len(allp), -1, np.int32),
    )
    st = winding.StormTrack(
        1, np.array([0, 3900]), np.array([80.0, 80.0]), np.array([80.0, 80.0]), np.array([6.0, 6.0])
    )
    r = winding.organism_winding(fr, [st])
    assert r["pairs"] == 12 and r["entrained_frac"] == 1.0
    assert r["turns"] == pytest.approx(12 * 0.05 * 39, rel=1e-3)
    assert r["skipped_steps"] == 0


def test_read_storms_jsonl(tmp_path: Path) -> None:
    lines = [
        {"step": 100, "storms": [{"id": 1, "x": 10.0, "y": 20.0, "r": 5.0}], "events": []},
        {
            "step": 200,
            "storms": [{"id": 1, "x": 12.0, "y": 21.0, "r": 5.5}, {"id": 2, "x": 90.0, "y": 90.0, "r": 4.0}],
            "events": [],
        },
    ]
    p = tmp_path / "storms.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n{torn")
    tr = winding.read_storms_jsonl(p)
    assert [t.id for t in tr] == [1, 2] and tr[0].t.tolist() == [100, 200] and tr[1].r.tolist() == [4.0]


def test_cli_braid_and_winding(tmp_path: Path) -> None:
    fr = swarm_frames(np.random.default_rng(11), nf=25, n_bodies=6)
    fr.meta.update({"run": "toy", "window": [0, 2400], "frames_version": 1})
    path = fm.write_frames(tmp_path / "frames_0_2400.npz", fr)
    out = tmp_path / "o"
    assert main(["braid", str(path), "--out", str(out), "--blocks", "4", "--n-surr", "4"]) == 0
    d = json.loads((out / "braid.json").read_text())
    assert d["version"] == braid.BRAID_VERSION and len(d["blocks"]) == 4
    a, b = orbit_tracks(2.0)
    sj = tmp_path / "storms.jsonl"
    sj.write_text(
        "\n".join(
            json.dumps(
                {
                    "step": int(a.t[i]),
                    "storms": [
                        {"id": 1, "x": a.x[i], "y": a.y[i], "r": 5.0},
                        {"id": 2, "x": b.x[i], "y": b.y[i], "r": 5.0},
                    ],
                    "events": [],
                }
            )
            for i in range(len(a.t))
        )
    )
    assert main(["winding", str(sj), "--size", "160", "--out", str(out)]) == 0
    w = json.loads((out / "winding.json").read_text())
    assert w["co_rotating_pairs"] == 1 and w["pairs"][0]["w"] == pytest.approx(2.0, abs=1e-6)
    assert main(["winding", str(tmp_path / "missing.jsonl"), "--size", "160", "--out", str(out)]) == 2


def test_panel_fills_block_c_and_p5(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({"seed": 1, "config_hash": "h", "start_step": 0}))
    (run / "metrics.jsonl").write_text("")
    fr = swarm_frames(np.random.default_rng(12), nf=49, n_bodies=8)
    fr.meta.update({"run": "run", "window": [0, 4800], "frames_version": 1})
    path = fm.write_frames(tmp_path / "frames_0_4800.npz", fr)
    out = tmp_path / "p"
    assert main(["panel", str(run), "--frames", str(path), "--windows", "0:4800", "--out", str(out)]) == 0
    p = json.loads((out / "panel.json").read_text())
    assert p["C"]["status"] == "ok" and p["C"]["braid"]["groups_total"] > 0
    e = p["endpoints"]["P5"]
    assert e["status"] == "ok" and len(e["blocks"]) == 16 and e["confirmatory"] is True
    assert p["C"]["winding"]["status"] == "needs_missing"
