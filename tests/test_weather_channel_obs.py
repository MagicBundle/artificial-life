"""The channel (equatorial band) detection and study driver, W-band-0 (DESIGN §37.4): the hemisphere from
the sign of f, the padded labelling, the detector and tracker on a channel, the latitude statistics from the
table, ``--band``, and a short channel study end to end. The weather step is ``test_weather_channel.py``.

CPU and fast.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from alife import storms, weather_obs
from alife import weather as W
from alife.analysis import w0
from alife.weather import WeatherParams

BAND = W.band_params(96.0)  # Ly = 96, 160 x 48, the recommended candidate (DESIGN §37.4)


# --- detection ----------------------------------------------------------------------------------------------


def test_hemisphere_sign_is_the_sign_of_f() -> None:
    s = weather_obs.hemisphere_sign(BAND)
    assert s.shape == (48, 1) and s.dtype == np.int8
    assert (s[:24] == -1).all() and (s[24:] == 1).all()
    for j in range(48):
        assert s[j, 0] == (1 if BAND.f_of_y((j + 0.5) * BAND.dx) > 0 else -1)
    t = weather_obs.hemisphere_sign(WeatherParams())
    assert t.shape == (160, 1) and (t[:80] == 1).all() and (t[80:] == -1).all()  # the torus: unchanged


def test_padded_labelling_does_not_join_across_a_wall() -> None:
    mask = np.zeros((8, 10), dtype=bool)
    mask[0, 3:6] = True  # the south-most row
    mask[7, 4:7] = True  # the north-most row, overlapping in x
    _, k_torus = weather_obs.label_periodic(mask)
    lab, k = weather_obs.label_channel(mask)
    assert k_torus == 1 and k == 2 and lab.shape == mask.shape
    assert lab[0, 3] != lab[7, 4] and (lab > 0).sum() == mask.sum()
    seam = np.zeros((8, 10), dtype=bool)
    seam[3, 9] = seam[3, 0] = True  # x stays periodic
    assert weather_obs.label_channel(seam)[1] == 1


def _frame(p: WeatherParams, blobs: list[tuple[int, int, int, int, float]]) -> storms.Frame:
    """A frame with rectangular vorticity blobs ``(row0, row1, col0, col1, zeta/f0)`` at the corners and rain
    under them; the rest at rest."""
    zeta = np.zeros((p.ny, p.n), dtype=np.float32)
    rain = np.zeros((p.ny, p.n), dtype=np.float32)
    for r0, r1, c0, c1, z in blobs:
        zeta[r0:r1, c0:c1] = z * p.f0
        rain[r0:r1, c0:c1] = 1e-3
    h1 = np.full((p.ny, p.n), p.H1, dtype=np.float32)
    h2 = np.full((p.ny, p.n), p.H2, dtype=np.float32)
    z = np.zeros((p.ny, p.n), dtype=np.float32)
    return storms.Frame(100, zeta, rain, h1, h2, z, z, z, z, z)


def test_detect_reads_a_channel_with_plain_latitudes() -> None:
    p = BAND
    # a compact north storm 4 rows from the north wall and a south one 4 rows from the south wall
    fr = _frame(p, [(40, 44, 20, 28, 1.0), (4, 8, 60, 68, -1.0)])
    det = storms.detect(fr, p)
    assert det.lab.shape == (p.ny, p.n) and len(det.storms) == 2 and not det.bands
    north = next(c for c in det.storms if c.hemi == 0)
    south = next(c for c in det.storms if c.hemi == 1)
    # the centroids are plain means of y, not circular ones; the corner field is read at the centres
    assert 40 * p.dx < north.y < 45 * p.dx and 4 * p.dx < south.y < 9 * p.dx
    assert 0 < north.x < p.L and north.ry == 1.0
    assert north.r_eq > 0 and north.wind_max == 0.0


def test_a_meridional_extent_is_not_a_band_but_a_zonal_one_is() -> None:
    p = BAND
    tall = _frame(p, [(26, 46, 20, 30, 1.0)])  # 20 rows tall, 10 wide, in the north
    d = storms.detect(tall, p)
    assert len(d.storms) == 1 and not d.bands
    wide = _frame(p, [(30, 34, 0, 100, 1.0)])  # 100 columns = 200 units > L/2: a zonal band
    d = storms.detect(wide, p)
    assert len(d.bands) == 1 and not d.storms


def test_the_tracker_follows_a_storm_along_a_wall() -> None:
    p = BAND
    tr = storms.Tracker(p)
    for k in range(6):
        fr = _frame(p, [(40, 44, 20 + 2 * k, 28 + 2 * k, 1.0)])
        fr = fr._replace(t=25 * (k + 1))
        tr.update(storms.detect(fr, p))
    tracks = tr.finish()
    assert len(tracks) == 1 and len(tracks[0].dets) == 6 and tracks[0].hemi == 0
    ys = [d.y for d in tracks[0].dets]
    assert max(ys) - min(ys) < 1e-9  # no wrap in y: the track does not move meridionally


# --- the study driver ---------------------------------------------------------------------------------------


def test_latitude_stats_use_the_walls_as_poles() -> None:
    p = BAND
    d = w0.pole_distance(np.array([90.0, 6.0]), np.array([0, 1]), p.L, p.pole_ys)
    np.testing.assert_allclose(d, [6.0, 6.0])  # 6 from the north wall, 6 from the south wall
    t = w0.pole_distance(np.array([70.0, 250.0]), np.array([0, 1]), 320.0)
    np.testing.assert_allclose(t, [10.0, 10.0])  # the torus default, unchanged
    lat = w0.latitude_stats([], p, 0, 100, 3)
    assert lat["poles"] == [96.0, 0.0] and lat["equators"] == [48.0] and lat["bin_edges"][-1] == 96.0
    assert w0.latitude_stats([], WeatherParams(), 0, 100, 3)["poles"] == [80.0, 240.0]


def test_load_params_band_flag() -> None:
    assert w0._load_params(None, [], 96.0) == BAND
    assert w0._load_params(None, ["sponge_w=3.0"], 96.0) == replace(BAND, sponge_w=3.0)
    assert w0._load_params(None, ["channel=false"], None).channel is False
    with pytest.raises(ValueError, match="exclusive"):
        w0._load_params("x.json", [], 96.0)


def test_a_channel_study_runs_end_to_end(tmp_path: Path) -> None:
    """``run_study`` + ``assemble_metrics`` + the outputs on a short channel run: the hemisphere from the sign
    of f, the poles from the table, a non-square grid, the wall-strip record, and the figures."""
    p = BAND
    st = w0.Settings(steps=600, t_spin=200, detect_every=25, frames_every=100, movies=False)
    res = w0.run_study(p, st)
    metrics, arrays = w0.assemble_metrics(res)
    ch = metrics["diagnostics"]["channel"]
    assert ch["v_wall_max"] == 0.0 and ch["detections"] == (600 - 200) // 25 + 1
    assert ch["wall_strip_speed"]["max"] >= ch["wall_strip_speed"]["median"] >= 0.0
    assert len(ch["interior_fraction_over_raw"]) == 3
    assert metrics["run"]["ny"] == 48 and metrics["diagnostics"]["agent_wind"] is None
    assert arrays["rows_y"].shape == (48,) and arrays["zonal_mean"].shape == (2, 48)
    assert metrics["health"]["mass_drift_max"] < 1e-5
    assert metrics["diagnostics"]["latitudes"]["poles"] == [96.0, 0.0]
    # the torus record has no channel block, and its record keys are as before
    assert w0.params_record(p, st)  # serialises (the derived block gains Ly and phi_max only for a channel)
    assert "Ly" not in WeatherParams().derived()
    pytest.importorskip("matplotlib")
    out = tmp_path / "study"
    out.mkdir()
    names = w0.write_study_outputs(out, res, metrics, arrays, plots=True, movies=False)
    for want in ("lat_hist.png", "jets.png", "counts.png", "report.md", "metrics.json"):
        assert want in names and (out / want).stat().st_size > 0
    assert any(n.startswith("frames/") for n in names)
