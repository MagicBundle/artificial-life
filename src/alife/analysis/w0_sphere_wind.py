"""The wind-coupling experiment of S4c (DESIGN §46.6, "Time and the Courant number"): does the wind that moves
agents and tracers have to be scaled by the weather time step ``dt``?

One engine step advances the weather by ``dt`` units of weather time (0.79 to 0.83), but the wind force on an
agent, and the step of a tracer, were ``α·u`` per engine step: an air parcel is displaced ``u·dt`` per engine
step in the weather itself (``x_d = x − dt·u/R``, the semi-Lagrangian backtrace of ``weather_sph.step``) and a
tracer ``u``, a factor ``1/dt`` faster against a storm that moves ``dt`` times its own speed. This tool seeds
two identical sets of passive tracers into the weather of a spun-up study and moves them with the engine's own
sampler (``weather_sph.sample_wind``, the cell value, through the vertical map of the shell) under both
couplings, ``α·u`` (scale 1) and ``α·u·dt`` (the clock-consistent one), and reports what differs:

* **retention**: the share of tracers seeded within one storm radius of a named storm's centre that are within
  two radii of it after 250 to 3 000 steps (storms still alive), and the number of revolutions about the
  centre per 1 000 steps of those that stay;
* **dispersal**: the median great-circle distance covered in 1 000 steps by tracers seeded uniformly over the
  shell;
* **exposure**: the share of uniform tracers inside a named storm (within one radius), and the share of
  tracer-ticks at an air speed above the hazard onset ``storm_u0``.

Study code, never imported by the simulation; it writes only the JSON given by ``--out``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from alife import geom, weather_sph
from alife.analysis.w0_sphere import StormBook, build_config
from alife.config import Config, dims
from alife.rng import InitStream, init_keys
from alife.storms_sph import SphereFrame, SphereParams
from alife.weather_spinup import advance_chain, chain_step, spinup_key

LAGS = (250, 500, 1000, 2000, 3000)


def _frame(ws: weather_sph.WeatherStateS, wp: Any, radius: float, t: int) -> SphereFrame:
    zeta = np.array(weather_sph.vorticity(ws, wp, radius), dtype=np.float64)
    spd = np.array(mx.sqrt((ws.u1 * ws.u1).sum(axis=-1)), dtype=np.float64)
    return SphereFrame(t, zeta, np.array(ws.P, dtype=np.float64), np.array(ws.h1, dtype=np.float64), spd)


def _disk_points(
    centre: np.ndarray, radius_arc: float, planet_radius: float, n: int, rng: np.random.Generator
):
    """``n`` unit directions uniform (by area) in the cap of arc radius ``radius_arc`` about ``centre``."""
    east, north = geom.local_frame(centre[None])
    east, north = east[0], north[0]
    r = (radius_arc / planet_radius) * np.sqrt(rng.uniform(size=n))
    phi = rng.uniform(0.0, 2.0 * math.pi, size=n)
    tang = np.cos(phi)[:, None] * east + np.sin(phi)[:, None] * north
    return np.cos(r)[:, None] * centre + np.sin(r)[:, None] * tang


def _azimuth(dirs: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """The azimuth (radians) of directions about ``centre`` in its east-north frame."""
    east, north = geom.local_frame(centre[None])
    return np.arctan2(dirs @ north[0], dirs @ east[0])


def _fold(a: np.ndarray) -> np.ndarray:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def wind_experiment(
    cfg: Config,
    *,
    seed: int = 0,
    spinup: int = 20_000,
    horizon: int = 3000,
    tick: int = 25,
    per_storm: int = 200,
    n_uniform: int = 4000,
    alpha: float = 1.0,
    z_range: tuple[float, float] = (0.3, 0.7),
) -> dict[str, Any]:
    """Run the experiment on a config and return the record (see the module docstring)."""
    wp = weather_sph.params_from_config(cfg)
    radius = cfg.world.radius
    sp = SphereParams.from_config(cfg)
    gm = geom.geometry(cfg)
    height = gm.height_max
    geo = weather_sph.WeatherGeometry.from_config(cfg)
    d = dims(cfg)
    book = StormBook(sp)
    step_fn = chain_step(cfg)
    key = spinup_key(init_keys(seed)[InitStream.STATE_KEY])
    wx = weather_sph.rest_array(cfg, 0)
    t = 0
    while t < spinup:
        wx, key = advance_chain(cfg, wx, t, key, tick, step_fn=step_fn)
        t += tick
        book.update(_frame(weather_sph.unpack(wx), wp, radius, t))
    # the named storms at the start
    storms = []
    for tr in book.tracker.active():
        dd = tr.dets[-1]
        if dd.t == t and tr.id in book.named_at:
            storms.append((tr.id, geom.unit(np.array([dd.x, dd.y, dd.z])), dd.r_eq))  # type: ignore[attr-defined]
    if not storms:
        raise RuntimeError("no named storm at the end of the spin-up: nothing to seed the tracers into")
    rng = np.random.default_rng(seed)
    dirs, owner, r0 = [], [], []
    for sid, c, r_eq in storms:
        pts = _disk_points(c, r_eq, radius, per_storm, rng)
        dirs.append(pts)
        owner += [sid] * per_storm
        r0 += [r_eq] * per_storm
    seeded = np.concatenate(dirs)
    owner_a, r0_a = np.array(owner), np.array(r0)
    n_s = seeded.shape[0]
    uni = geom.unit(rng.normal(size=(n_uniform, 3)))
    z_all = height * rng.uniform(z_range[0], z_range[1], size=n_s + n_uniform)
    start_dirs = np.concatenate([seeded, uni])
    m = start_dirs.shape[0]
    scales = (1.0, wp.dt)
    pos0 = ((radius + z_all)[:, None] * start_dirs).astype(np.float32)
    pos = mx.array(np.concatenate([pos0, pos0]))  # set 0: scale 1, set 1: scale dt
    sc = mx.array(np.concatenate([np.full(m, scales[0]), np.full(m, scales[1])]).astype(np.float32))[:, None]
    r_lo, r_hi = d.r_lo, d.r_hi
    t_start = t
    # per tick state
    cum = np.zeros((2, n_s))
    az_prev = np.zeros((2, n_s))
    for j, (_, c, _) in enumerate(storms):
        sl = slice(j * per_storm, (j + 1) * per_storm)
        az_prev[:, sl] = _azimuth(start_dirs[sl], c)
    res: dict[str, Any] = {lag: {} for lag in LAGS}
    air_hi = np.zeros(2)
    air_n = 0
    in_storm = np.zeros(2)
    t0 = time.time()
    one = step_fn
    tt = mx.array(t, dtype=mx.int32)
    for step_i in range(1, horizon + 1):
        wx, tt, key = one(wx, tt, key)
        t += 1
        ws = weather_sph.unpack(wx)
        aw = weather_sph.sample_wind(ws, pos, wp, geo)
        moved = pos + sc * alpha * aw.u
        rho = mx.sqrt((moved * moved).sum(axis=-1, keepdims=True))
        pos = moved * (mx.clip(rho, r_lo, r_hi) / mx.maximum(rho, 1e-30))
        if step_i % tick:
            continue
        mx.eval(pos, aw.u)
        book.update(_frame(ws, wp, radius, t))
        p = np.array(pos, dtype=np.float64)
        dir_t = geom.unit(p)
        speed = np.linalg.norm(np.array(aw.u, dtype=np.float64), axis=1)
        air_hi += [
            (speed[uni_sl] >= cfg.weather.storm_u0).mean()
            for uni_sl in (slice(n_s, m), slice(m + n_s, 2 * m))
        ]
        air_n += 1
        live = {tr.id: tr for tr in book.tracker.active() if tr.dets[-1].t == t and tr.id in book.named_at}
        # storm centres and radii now
        cen = {sid: geom.unit(np.array([live[sid].dets[-1].x, live[sid].dets[-1].y, live[sid].dets[-1].z]))  # type: ignore[attr-defined]
               for sid in live}  # fmt: skip
        req = {sid: live[sid].dets[-1].r_eq for sid in live}
        for s_i in (0, 1):
            seg = dir_t[s_i * m : s_i * m + n_s]
            for j, (sid, _, _) in enumerate(storms):
                if sid not in cen:
                    continue
                sl = slice(j * per_storm, (j + 1) * per_storm)
                az = _azimuth(seg[sl], cen[sid])
                cum[s_i, sl] += _fold(az - az_prev[s_i, sl])
                az_prev[s_i, sl] = az
            # exposure of the uniform tracers to the named storms now
            if cen:
                cs = np.array(list(cen.values()))
                rs = np.array([req[k] for k in cen])
                du = geom.great_circle_distance(
                    dir_t[s_i * m + n_s : (s_i + 1) * m, None, :], cs[None], radius
                )
                in_storm[s_i] += float((du <= rs[None]).any(axis=1).mean())
        if step_i in LAGS:
            for s_i in (0, 1):
                seg = dir_t[s_i * m : s_i * m + n_s]
                alive = np.array([sid in cen for sid in owner_a])
                dist = np.full(n_s, np.inf)
                for j, (sid, _, _) in enumerate(storms):
                    if sid in cen:
                        sl = slice(j * per_storm, (j + 1) * per_storm)
                        dist[sl] = geom.great_circle_distance(seg[sl], cen[sid], radius)
                keep = alive & (dist <= 2.0 * r0_a)
                res[step_i][f"scale_{('1', 'dt')[s_i]}"] = {
                    "storms_alive": len({o for o, a in zip(owner_a, alive, strict=True) if a}),
                    "retained": float(keep.sum() / max(alive.sum(), 1)),
                    "revolutions_per_1000_median": float(
                        np.median(np.abs(cum[s_i][keep])) / (2 * math.pi) / (step_i / 1000.0)
                    )
                    if keep.any()
                    else None,
                }
                disp = geom.great_circle_distance(dir_t[s_i * m + n_s : (s_i + 1) * m], uni, radius)
                res[step_i][f"scale_{('1', 'dt')[s_i]}"]["uniform_median_displacement"] = float(
                    np.median(disp)
                )
        if step_i % 500 == 0:
            print(f"  tracer step {step_i}/{horizon} {time.time() - t0:.0f}s", flush=True)
    n_ticks = max(horizon // tick, 1)
    share = {
        "uniform_air_speed_over_onset_share": air_hi / air_n,
        "uniform_inside_named_storm_share": in_storm / n_ticks,
    }
    return {
        "planet": {"radius": radius, "nw": sp.nside, "dt": wp.dt},
        "settings": {
            **{"seed": seed, "spinup": spinup, "horizon": horizon, "alpha": alpha},
            **{"per_storm": per_storm, "n_uniform": n_uniform, "t_start": t_start, "storms": len(storms)},
        },
        "storm_radii": [float(r) for _, _, r in storms],
        "by_lag": {str(k): v for k, v in res.items()},
        **{k: {"scale_1": float(v[0]), "scale_dt": float(v[1])} for k, v in share.items()},
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="alife.analysis.w0_sphere_wind", description=__doc__.split("\n")[0])
    ap.add_argument("--preset", required=True)
    ap.add_argument("--overlay", action="append", default=[])
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--pools", help="JSON: the keywords of w0_sphere.make_pools, or a flat list of pools")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spinup", type=int, default=20_000)
    ap.add_argument("--horizon", type=int, default=3000)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cfg = build_config(a.preset, a.overlay, a.set, json.loads(a.pools) if a.pools else None)
    rec = wind_experiment(cfg, seed=a.seed, spinup=a.spinup, horizon=a.horizon, alpha=a.alpha)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rec["by_lag"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
