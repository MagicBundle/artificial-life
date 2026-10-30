"""The study driver of the sphere's weather (``alife.analysis.w0_sphere``, S4c): the pool builder, the grid's
irregularities, the exposure statistics against their latitude-matched null, the counts, the storm book
against the storm layer of ``alife play`` on synthetic vortices, and a short study (DESIGN §46.9)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from alife import geom
from alife.analysis import w0_sphere as ws
from alife.storms import NAME_ZETA
from alife.storms_sph import SphereParams, StormLayerS, detect, frame_of_vortices

NW = 16


def test_make_pools_places_equal_pools_at_the_latitude_and_the_angular_drift() -> None:
    pools = ws.make_pools(90.0, 3, 32.0, 16.0)
    assert len(pools) == 6 * 7
    rows = np.array(pools).reshape(6, 7)
    lat = np.degrees(rows[:, 1] / 90.0)
    assert np.allclose(lat, [40, 40, 40, -40, -40, -40])
    lon = np.degrees(rows[:, 0] / 90.0)
    assert np.allclose(lon, [60, 180, 300, 0, 120, 240])  # the southern offset by half a spacing
    assert np.allclose(rows[:, 2:5], [32.0, 16.0, 1.0])
    # stormsph's arc drift on the planet it was made for: -0.009 per unit of weather time at 40 degrees
    assert np.allclose(rows[:, 5], -0.009, atol=1e-6) and np.all(rows[:, 6] == 0.0)
    # the same angular rate on a smaller planet is a smaller arc speed (stormsph22: -0.0048)
    assert np.allclose(np.array(ws.make_pools(48.0, 3, 24.0, 16.0)).reshape(6, 7)[:, 5], -0.0048, atol=1e-6)
    four = np.array(ws.make_pools(90.0, 4, 32.0, 16.0)).reshape(8, 7)
    assert np.allclose(np.degrees(four[:4, 0] / 90.0), [60, 150, 240, 330])
    assert np.allclose(np.degrees(four[4:, 0] / 90.0), [15, 105, 195, 285])
    with pytest.raises(ValueError, match="per_hemisphere"):
        ws.make_pools(90.0, 0, 32.0, 16.0)


def test_the_pools_of_every_overlay_are_the_builders_at_their_calibrated_sizes() -> None:
    for preset, overlay, radius, sx, sy in (
        ("64k3s", "stormsph", 90.0, 40.0, 20.0),
        ("mini3s", "stormsph22", 48.0, 24.0, 16.0),
        ("tiny3s", "stormsph16", 36.0, 18.0, 12.0),
    ):
        cfg = ws.build_config(preset, [overlay])
        assert np.allclose(ws.make_pools(radius, 3, sx, sy), cfg.weather.pools, atol=1e-5), overlay
    # the box's 32 x 16 of S4b and S7a is the same builder with the draft's size
    assert np.allclose(ws.make_pools(90.0, 3, 32.0, 16.0)[2:5], [32.0, 16.0, 1.0])


def test_build_config_takes_pool_keywords_or_a_flat_list_and_validates() -> None:
    a = ws.build_config(
        "tiny3s", ["stormsph16"], ["weather.seed_rate=0.01"], {"per_hemisphere": 3, "sx": 10.0, "sy": 6.0}
    )
    assert a.weather.seed_rate == 0.01 and len(a.weather.pools) == 42
    b = ws.build_config("tiny3s", ["stormsph16"], pools=list(a.weather.pools))
    assert b.weather.pools == a.weather.pools
    with pytest.raises(
        Exception, match=r"latitude|pool"
    ):  # a pool beyond 60 degrees leaves the validated band
        ws.build_config(
            "tiny3s", ["stormsph16"], pools={"per_hemisphere": 3, "sx": 10.0, "sy": 6.0, "lat": 70.0}
        )
    with pytest.raises(ValueError, match=r"weather\.enabled"):
        ws.build_config("tiny3s")


@pytest.mark.parametrize("nside", [4, 16, 22, 42])
def test_the_trivalent_vertices_are_the_eight_points_at_z_two_thirds(nside: int) -> None:
    v = ws.trivalent_vertices(nside)
    assert v.shape == (8, 3)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert np.allclose(np.sort(v[:, 2]), [-2 / 3] * 4 + [2 / 3] * 4, atol=1e-9)
    lon = np.sort(np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0)
    assert np.allclose(lon, [0, 0, 90, 90, 180, 180, 270, 270], atol=1e-6)


@pytest.mark.parametrize("nside", [4, 16])
def test_seam_cells_are_symmetric_and_the_pole_cells_are_the_four_at_each_pole(nside: int) -> None:
    seam = ws.seam_cells(nside)
    c = geom.pixel_centres(nside)
    # the grid is symmetric under a quarter turn about the pole axis and the mirror in the equator
    for f in (lambda x: np.stack([-x[:, 1], x[:, 0], x[:, 2]], 1), lambda x: x * np.array([1, 1, -1.0])):
        img = geom.ang2pix(f(c), nside)
        assert np.array_equal(seam[img], seam)
    # the seams are a thin set that grows like the side, not the area
    assert 0.0 < seam.mean() < 0.8
    pole = ws.pole_cells(nside)
    assert pole.sum() == 8 and np.all(np.abs(c[pole, 2]) > 0.9)
    assert ws.seam_cells(2 * nside).mean() < seam.mean()


def test_exposure_equals_the_null_for_storms_placed_at_random_longitudes_and_exceeds_it_at_a_vertex() -> None:
    ex = ws.Exposure.of(NW, 36.0)
    rng = np.random.default_rng(0)
    lat, lon = np.radians(40.0), rng.uniform(0, 2 * np.pi, 3000)
    cen = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.full(lon.size, np.sin(lat))], 1)
    tid = np.arange(lon.size) // 30
    e = ws.exposure_stats(ex, cen, tid, boot=50)
    for k in ("within_1.5", "within_3"):
        assert abs(e["vertex"][k]["ratio"] - 1.0) < 0.35, e["vertex"][k]
        assert abs(e["seam"][k]["ratio"] - 1.0) < 0.2, e["seam"][k]
    assert e["detections"] == 3000 and e["tracks"] == 100
    # a storm sitting on a trivalent vertex is within 1.5 cell sides of it every time
    v = ws.trivalent_vertices(NW)[0]
    at = ws.exposure_stats(ex, np.tile(v, (50, 1)), np.zeros(50, int), boot=20)
    assert at["vertex"]["within_1.5"]["observed"] == 1.0
    assert at["vertex"]["within_1.5"]["ratio"] > 3.0
    assert ws.exposure_stats(ex, np.zeros((0, 3)), np.zeros(0, int)) == {"detections": 0, "tracks": 0}


def test_the_null_of_a_latitude_follows_the_geometry() -> None:
    """On the latitude of the vertices (asin 2/3) four arcs of 1.5 cell sides each way are near a vertex."""
    ex = ws.Exposure.of(NW, 36.0)
    lat = np.array([math.degrees(math.asin(2 / 3))])
    n = ws._null_shares(ex, lat)
    # the arc of 1.5 cell sides on each side of 4 vertices, of the circle of that latitude
    expect = 4 * 2 * 1.5 * ex.side / (2 * math.pi * 36.0 * math.cos(math.asin(2 / 3)))
    assert n["vertex"][0, 0] == pytest.approx(expect, rel=0.1)
    assert n["vertex"][0, 0] <= n["vertex"][0, 1] <= n["vertex"][0, 2]


def test_count_stats_reports_means_ranges_and_the_joint_share() -> None:
    named = np.array([[2, 3], [3, 3], [1, 4], [0, 2]])
    allc = named + 1
    c = ws.count_stats(named, allc)
    assert (
        c["named"]["north"]["mean"] == 1.5
        and c["named"]["north"]["max"] == 3
        and c["named"]["north"]["min"] == 0
    )
    assert c["named"]["north"]["in_range"] == 0.5 and c["named"]["south"]["in_range"] == 1.0
    assert c["named"]["in_range_both"] == 0.5
    assert c["active"]["in_range_both"] == 0.75
    assert c["named"]["south"]["histogram"] == {2: 1, 3: 2, 4: 1}
    assert ws.count_stats(np.zeros((0, 2), int), np.zeros((0, 2), int)) == {}
    assert ws.count_stats(named, allc, lo=3, hi=3)["named"]["in_range_both"] == 0.25


def test_the_plateau_time_is_where_a_relaxation_stays_within_the_tolerance() -> None:
    t = np.arange(0.0, 20000.0, 500.0)
    tau = 2000.0
    y = 1.0 - np.exp(-t / tau)
    got = ws.plateau_time(t, y, tol=0.05, ref_from=15000.0, smooth=1)
    assert got is not None and abs(got - tau * math.log(20.0)) < 600.0  # 1 - e^(-t/tau) = 0.95 at 3 tau
    assert ws.plateau_time(t, np.ones_like(t), ref_from=10000.0) == 0.0  # flat from the start
    assert ws.plateau_time(t, np.sin(t / 500.0) + 2.0, tol=0.05, ref_from=15000.0) is None  # never settles
    assert ws.plateau_time(t, y, ref_from=1e9) is None  # no late part
    series = [
        {"t": float(a), "ke": float(b), "q_mean": float(b), "p_mean": 1.0} for a, b in zip(t, y, strict=True)
    ]
    n = 800
    counts = np.column_stack([25 * np.arange(1, n + 1), np.minimum(np.arange(n) // 100, 3), np.full(n, 2)])
    sp = ws.spinup_stats(series, counts, 20000)
    assert sp["p_mean"] == 0.0 and 3000.0 < sp["ke"] < 8000.0 and 9000.0 < sp["q_mean"] < 10500.0
    assert sp["settled"] == max(sp["ke"], sp["q_mean"], sp["named_north"], sp["named_south"])
    assert sp["named_south"] == 500.0  # constant: there from the first block (20 ticks of 25 steps)
    assert 5000.0 < sp["named_north"] < 8000.0  # the count steps up to 3 at tick 300: within 1 of 3 at 2


def test_the_turnover_counts_births_and_deaths_of_the_named_storms_in_the_window() -> None:
    st = ws.StudySettings(steps=30000, spinup=10000)

    def tr(named_at: int, t_last: int, ended: int, dets: int = 50) -> dict:
        return {"named_at": named_at, "t_last": t_last, "ended": ended, "dets": dets}

    rows = [
        tr(2000, 30000, 0),  # born in the spin-up, alive at the end: no event
        tr(3000, 15000, 1),  # born in the spin-up, died in the window
        tr(12000, 24000, 1),  # born and died in the window
        tr(20000, 30000, 0),  # born in the window, alive
        tr(-1, 18000, 1),  # never named
        tr(15000, 15100, 1, dets=2),  # a fragment (fewer than the qualifying detections)
        tr(4000, 8000, 1),  # died in the spin-up
    ]
    t = ws._turnover(rows, st)
    assert (t["named_births_in_window"], t["named_deaths_in_window"]) == (2, 2)
    assert t["named_deaths_per_10k"] == pytest.approx(1.0)  # 2 deaths in 20 000 steps
    assert ws._turnover([tr(2000, 30000, 0)], st)["named_deaths_per_10k"] == 0.0  # a frozen world
    flat = [{"t": float(t), "ke": 1e-3, "umax1": 0.08} for t in range(0, 30000, 500)]
    assert ws._window_cv(flat, 10000) == {"ke": 0.0, "umax1": 0.0}
    live = [
        {"t": float(t), "ke": 1e-3 * (1 + 0.2 * math.sin(t / 3000)), "umax1": 0.08}
        for t in range(0, 30000, 500)
    ]
    cv = ws._window_cv(live, 10000)
    assert 0.05 < cv["ke"] < 0.2 and cv["umax1"] == 0.0


def _synthetic_frames(nside: int, radius: float, n: int):
    """Two cyclones drifting east at 40 degrees (one strong, one that never reaches the naming peak) and one
    across the south, for ``n`` ticks of 25 steps."""
    frames = []
    for k in range(n):
        t = 25 * (k + 1)
        out = []
        for lat, lon0, amp, sig in (
            (40.0, 20.0, 0.02, 8.0),
            (-40.0, 200.0, 0.02, 8.0),
            (45.0, 150.0, 0.009, 8.0),
        ):
            lon = math.radians(lon0 + 0.4 * k)
            c = geom.from_latlon(lat, math.degrees(lon))
            out.append({"centre": c, "sigma": sig, "amp": amp * (1.0 + 0.02 * (k % 3))})
        frames.append(frame_of_vortices(nside, t, out, radius))
    return frames


def test_the_storm_book_names_and_counts_as_the_storm_layer_of_play_does() -> None:
    cfg = ws.build_config("tiny3s", ["stormsph16"])
    sp = SphereParams.from_config(cfg)
    book = ws.StormBook(sp)
    layer = StormLayerS(sp, 1.0, 25)
    frames = _synthetic_frames(sp.nside, cfg.world.radius, 24)
    seen = set()
    for fr in frames:
        n_named, n_all, rows = book.update(fr)
        tick = layer.ingest(detect(fr, sp))
        storms = [s for s in tick.strm["storms"] if s["state"] == "active"]
        want_named = (
            sum(1 for s in storms if s["named"] and s["hemi"] == 0),
            sum(1 for s in storms if s["named"] and s["hemi"] == 1),
        )
        want_all = (sum(1 for s in storms if s["hemi"] == 0), sum(1 for s in storms if s["hemi"] == 1))
        assert n_named == want_named and n_all == want_all
        assert len(rows) == sum(n_named)
        seen.add((n_named, n_all))
    assert max(n[0] for n, _ in seen) >= 1 and max(a[0] for _, a in seen) > max(
        n[0] for n, _ in seen
    )  # the weak one
    assert all(p >= NAME_ZETA for tid, p in book.peak.items() if tid in book.named_at)


def test_a_short_study_writes_the_record_the_tables_and_the_arrays(tmp_path) -> None:
    cfg = ws.build_config("tiny3s", ["stormsph16"])
    st = ws.StudySettings(steps=1500, spinup=500, seed=1, tick=50, series_every=5)
    rec = ws.run_study(cfg, st)
    assert rec["planet"]["nw"] == 16 and rec["planet"]["dt"] == pytest.approx(0.829, abs=1e-3)
    assert rec["ticks_in_window"] == 21
    assert set(rec["counts"]) >= {"named", "active", "d5"}
    assert set(rec["lifetime"]) == {"named_hindsight", "all_qualifying", "named_realtime"}
    assert {"named_births_in_window", "named_deaths_per_10k"} <= set(rec["tracks"]) and set(
        rec["climate_cv"]
    ) == {"ke", "umax1"}
    assert rec["_counts"].shape == (30, 5) and rec["_counts"][0, 0] == 50
    assert len(rec["_series"]) == 6 and rec["_series"][0]["ke"] >= 0.0
    counts, series = rec["_counts"].copy(), list(rec["_series"])
    ws.write_outputs(rec, tmp_path / "o")
    assert {p.name for p in (tmp_path / "o").iterdir()} >= {"summary.json", "series.csv", "dets.npz"}
    again = json.loads((tmp_path / "o" / "summary.json").read_text())
    assert again["settings"]["seed"] == 1 and "_series" not in again
    # the same seed repeats exactly: the weather is the engine's chain, no host randomness
    rec2 = ws.run_study(cfg, st)
    assert np.array_equal(counts, rec2["_counts"])
    assert series == rec2["_series"]


def test_the_sweep_runs_a_candidate_in_a_worker_process_and_skips_a_finished_one(tmp_path, capsys) -> None:
    spec = {
        "base": {"preset": "tiny3s", "overlays": ["stormsph16"], "steps": 300, "spinup": 100, "tick": 50},
        "candidates": [
            {
                "name": "c",
                "set": ["weather.seed_rate=0.01"],
                "pools": {"per_hemisphere": 3, "sx": 10, "sy": 6},
            }
        ],
    }
    out = ws.sweep(spec, tmp_path, jobs=2, mem_gb=30.0, per_proc_gb=2.0)
    assert [d.name for d in out] == ["c_s0"] and (out[0] / "summary.json").exists()
    assert json.loads((out[0] / "summary.json").read_text())["weather"]["seed_rate"] == 0.01
    ws.sweep(spec, tmp_path, jobs=2, mem_gb=30.0, per_proc_gb=2.0)
    assert "skip c_s0" in capsys.readouterr().out
