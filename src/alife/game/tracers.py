"""The tracer rig and the device side of the W2 dumps and coupling block (STORMS §5 control 2, DESIGN §32).
Pure MLX.

A *tracer* is a passive point moved by the wind: every step it advances by ``α_t·u_air(pos)``, with the same
sampler, the same speed limit and the same floor and surface clamps as an organism, and with the coupling
``α_t = α0 + α1·cap`` of a live agent it was last seeded from (``α0`` alone in 2D). It never steers, eats or
dies, so it is the null of "agents against passive drift": what a population with the agents' positions and
couplings would do if its brains were switched off. Every ``game.tracer_period`` steps (about the organisms'
median age at death) all tracers are reseeded at uniformly drawn live agents, so a tracer is an agent that
stops steering, age-matched to an agent's life.

The rig is an observer: it reads the post-step ``wx``, ``pos``, ``buoy_cap`` and ``alive``, writes only its
own ``TracerState`` and draws only from its **own** key (``rng.observer_key(run.seed, TRACER_SALT)``, split
at each reseed and kept in the sidecar); it never reads or advances ``State.key``, so a run with tracers has
the trajectory of a run without. The coupling ``α_t`` uses the config's unscaled ``alpha0``/``alpha1`` and
the total wind ``u_air``, whatever ``weather.wind_scale`` and ``weather.wind_eddy_force`` do to the agents,
so the tracers of the still-air and eddy-force arms move exactly as those of the baseline.

The rest of the module is the device half of two instruments: ``coupling_inputs`` (the per-agent and
per-tracer heights, updraft-boundary index ``g = |∇_h w|``, vertical wind and own vertical velocity that
``weather_obs.coupling_block`` reduces on the host) and ``eco_arrays`` / ``wx_planes`` (the arrays of the run
dumps ``observers/eco_<step>.npz`` and ``observers/wx_<step>.npy``). All are eager, evaluated at a metrics
boundary or a dump step in the step's own ``mx.eval``.

Lint (rules 2 and 5, ``tests/test_lint.py``): no subscript assignment, no host read; dicts are literals.
"""

from __future__ import annotations

from typing import Final, NamedTuple

import mlx.core as mx

from alife import geom, ops, rng, sphere, structures, weather, weather_sph
from alife.config import Config, dims
from alife.state import State, State3
from alife.step import Acts

__all__ = [
    "TRACER_SALT",
    "CouplingIn",
    "TracerState",
    "coupling_inputs",
    "eco_arrays",
    "tracer_init",
    "tracer_key",
    "tracer_reseed",
    "tracer_step",
    "updraft_gradient",
    "wx_planes",
]

_F32 = mx.float32
_I32 = mx.int32

TRACER_SALT: Final = 0x7ACE5  # the "step" of the observer key the rig draws from (any fixed value)


class TracerState(NamedTuple):
    """The rig's state (all of it is in the observer sidecar, DESIGN §32)."""

    pos: mx.array  # (T, nd) float32 tracer positions
    alpha: mx.array  # (T,) float32 coupling to the wind of the agent each was seeded from
    age: mx.array  # (T,) int32 steps since the last reseed
    key: mx.array  # (2,) uint32 the rig's own PRNG key


def tracer_key(seed: int) -> mx.array:
    """The rig's starting key for a run seed: ``rng.observer_key`` at the fixed salt (never ``State.key``)."""
    return rng.observer_key(seed, TRACER_SALT)


def _alpha(cap: mx.array, cfg: Config) -> mx.array:
    """The tracer coupling ``α0 + α1·cap`` (3D) or ``α0`` (2D), from the config's unscaled values."""
    p = weather.params_from_config(cfg)
    if dims(cfg).d3:  # static
        return weather.alpha(cap, p)
    return mx.full(cap.shape, p.alpha0, dtype=_F32)


def tracer_reseed(
    tr: TracerState, pos: mx.array, buoy_cap: mx.array | None, alive: mx.array, cfg: Config
) -> TracerState:
    """Every tracer takes the position and the coupling of a uniformly drawn live row, and its age is 0.

    ``order`` lists the live rows first (the sort key ``(~alive)·N + row`` is unique, so the order does not
    depend on the sort's stability); tracer ``i`` takes ``order[⌊u_i·n_alive⌋]`` with ``u_i`` drawn from the
    rig's own key (split once per reseed). With no live row every tracer takes row 0 (stale, never read). The
    caller passes the post-step ``pos``, ``buoy_cap`` (None in 2D) and ``alive``."""
    n = alive.shape[0]
    t = tr.pos.shape[0]
    k = mx.random.split(tr.key, 2)
    u = mx.random.uniform(shape=(t,), key=k[1])
    n_alive = alive.astype(_I32).sum()
    rank = mx.minimum(mx.floor(u * n_alive.astype(_F32)).astype(_I32), mx.maximum(n_alive - 1, 0))
    order = mx.argsort((~alive).astype(_I32) * n + mx.arange(n, dtype=_I32))
    idx = mx.take(order, rank)
    cap = mx.zeros((n,), _F32) if buoy_cap is None else buoy_cap
    return TracerState(mx.take(pos, idx, axis=0), _alpha(mx.take(cap, idx), cfg), tr.age * 0, k[0])


def tracer_init(pos: mx.array, buoy_cap: mx.array | None, alive: mx.array, cfg: Config) -> TracerState:
    """The rig at a state: ``game.tracers`` tracers seeded at its live rows, from the key of the run seed."""
    t, nd = cfg.game.tracers, pos.shape[-1]
    blank = TracerState(
        mx.zeros((t, nd), _F32), mx.zeros((t,), _F32), mx.zeros((t,), _I32), tracer_key(cfg.run.seed)
    )
    return tracer_reseed(blank, pos, buoy_cap, alive, cfg)


def _wind(
    ws: weather.WeatherState | weather_sph.WeatherStateS,
    pos: mx.array,
    p: weather.WeatherParams,
    cfg: Config,
    valid: mx.array | None = None,
) -> weather.AgentWind:
    """The air velocity at 3D positions: the box's sampler, or the sphere's (planet-frame vectors, DESIGN
    §46.6) through the vertical map of the shell."""
    if dims(cfg).sph:  # static
        return weather_sph.sample_wind(ws, pos, p, weather_sph.WeatherGeometry.from_config(cfg), valid=valid)
    return weather.sample_wind(ws, pos, p, field_cell=cfg.world.field_cell, valid=valid)


def tracer_step(tr: TracerState, wx: mx.array, cfg: Config) -> TracerState:
    """One step of the tracers on the end-of-step weather ``wx``: ``pos ← wrap_xy_clamp_z(pos + v)`` with
    ``v = α_t·u_air(pos)`` (on a sphere times the weather time of a step ``dt``, as the agents' wind force)
    limited to ``physics.v_max`` by its norm, as ``physics.integrate3`` (2D: the
    layer-1 wind and ``ops.wrap``). ``age`` counts up. No random draw. Every output derives from an input
    through an operation (the MLX 0.32 ``mx.compile`` quirk noted at ``step.consume_env``)."""
    p = weather.params_from_config(cfg)
    ph, w = cfg.physics, cfg.world
    ws = weather.unpack(wx)
    d = dims(cfg)
    u = _wind(ws, tr.pos, p, cfg).u if d.d3 else weather.sample_wind_h(ws, tr.pos, p).u  # static
    v = tr.alpha[:, None] * u
    if d.sph:  # static: the wind is in length per unit of weather time, a step is dt of it (DESIGN §46.6)
        v = v * p.dt
    speed = mx.sqrt((v * v).sum(axis=-1))
    v = mx.where((speed > ph.v_max)[:, None], v * (ph.v_max / mx.maximum(speed, ph.v_max))[:, None], v)
    moved = tr.pos + v
    if d.sph:  # static: back into the shell along the radius (the floor and the lid, as a clamp of z)
        rho = mx.sqrt((moved * moved).sum(axis=-1, keepdims=True))
        pos = moved * (mx.clip(rho, d.r_lo, d.r_hi) / mx.maximum(rho, 1e-30))
    else:
        pos = ops.wrap_xy_clamp_z(moved, w.size, d.z_lo, d.z_hi) if d.d3 else ops.wrap(moved, w.size)
    return TracerState(pos, tr.alpha * 1.0, tr.age + 1, tr.key + mx.zeros_like(tr.key))


# --- the coupling block's device half ---


def updraft_gradient(wx: mx.array, cfg: Config) -> mx.array:
    """``g = |∇_h w|`` of the vertical wind at the field-layer centres on the weather cells, ``(FZ, n, n)``
    float32: ``weather.wind_layers`` (the 20-layer diagnostic) and central differences with periodic
    neighbours, ``(w[i+1] - w[i-1])/(2·dx)`` along each axis. 3D only."""
    p = weather.params_from_config(cfg)
    if dims(
        cfg
    ).sph:  # static: (FZ, N) on the weather cells; the gradient is the tangent one of the cell graph
        ws = weather.unpack(wx)
        tab = weather_sph.dev_for(p, cfg.world.radius)
        gm = geom.geometry(cfg)
        mid = geom.layer_radii(gm.radius, gm.height_max, gm.layers)
        mid = mx.array(0.5 * (mid[:-1] + mid[1:]), dtype=_F32)
        pos = (mid[:, None, None] * tab.x[None]).reshape(-1, 3)
        geo = sphere.geometry(cfg)
        w = geo.radial(_wind(ws, pos, p, cfg).u, pos).reshape(mid.shape[0], -1)
        grad = weather_sph._grad(w, tab, True)  # (FZ, N, 3)
        return mx.sqrt((grad * grad).sum(axis=-1))
    fz = round(cfg.world.height / cfg.world.field_cell)
    w = weather.wind_layers(weather.unpack(wx), p, fz, cfg.world.field_cell)[:, 2]
    gx = (mx.roll(w, -1, axis=-1) - mx.roll(w, 1, axis=-1)) / (2.0 * p.dx)
    gy = (mx.roll(w, -1, axis=-2) - mx.roll(w, 1, axis=-2)) / (2.0 * p.dx)
    return mx.sqrt(gx * gx + gy * gy)


def _g_at(g: mx.array, pos: mx.array, cfg: Config) -> mx.array:
    """``g`` of the nearest layer and weather cell of each position, (M,) float32 (a gather)."""
    p = weather.params_from_config(cfg)
    if dims(cfg).sph:  # static: the layer of the position and the weather cell of its direction
        geo = sphere.geometry(cfg)
        k = sphere.layer_index(pos, geo.rp3, geo.dv, geo.layers)
        return mx.take(g.reshape(-1), k * g.shape[-1] + sphere.ang2pix(pos, p.n))
    fz, n = g.shape[0], g.shape[-1]
    k = mx.clip(ops.floor_i32(pos[:, 2] / cfg.world.field_cell), 0, fz - 1)
    i = ops.floor_i32(pos[:, 0] / p.dx) % n
    j = ops.floor_i32(pos[:, 1] / p.dx) % n
    return mx.take(g.reshape(-1), (k * n + j) * n + i)


class CouplingIn(NamedTuple):
    """Per-row inputs of ``weather_obs.coupling_block`` (device arrays; the agents' are over every row, the
    host keeps the live ones)."""

    alive: mx.array  # (N,) bool
    z: mx.array  # (N,) agent heights
    g: mx.array  # (N,) updraft-boundary index at the agent's cell and layer
    w: mx.array  # (N,) the air's vertical velocity at the agent
    vz_own: mx.array  # (N,) the agent's vertical velocity minus the wind drift's
    tz: mx.array  # (T,) tracer heights
    tg: mx.array  # (T,)
    tw: mx.array  # (T,)


def coupling_inputs(s: State | State3, tr: TracerState, cfg: Config) -> CouplingIn:
    """The device arrays the ``coupling`` block of a metrics row reduces, from the post-step state ``s`` and
    the tracers ``tr`` (3D, weather on). ``w`` is the sampled air velocity ``u_air[2]`` at the position;
    ``vz_own = vel_z - drift_z`` with the drift as the engine applies it (``weather.agent_drift``:
    ``wind_scale``
    and ``wind_eddy_force`` included), so it is the agent's own vertical motion."""
    p = weather.params_from_config(cfg)
    ws = weather.unpack(s.wx)
    g = updraft_gradient(s.wx, cfg)
    geo = sphere.geometry(cfg)  # the vertical: z in a box, the radial component on a sphere
    w_a = geo.radial(_wind(ws, s.pos, p, cfg, valid=s.alive).u, s.pos)
    w_t = geo.radial(_wind(ws, tr.pos, p, cfg).u, tr.pos)
    drift = structures.drift_at(s.wx, s.pos, s.buoy_cap, s.alive, getattr(s, "mat", None), cfg)
    return CouplingIn(
        s.alive,
        geo.height(s.pos),
        _g_at(g, s.pos, cfg),
        w_a,
        geo.radial(s.vel - drift, s.pos),
        geo.height(tr.pos),
        _g_at(g, tr.pos, cfg),
        w_t,
    )


# --- the dumps' device half ---------------------------------------------------------------------------------


def eco_arrays(s: State | State3, acts: Acts, tr: TracerState | None, cfg: Config) -> dict[str, mx.array]:
    """The arrays of one ``observers/eco_<step>.npz`` (3D, weather on), over every row (the host keeps the
    live ones): ``alive``, ``uid``, ``clade``, ``cap`` (``buoy_cap``), ``z``, ``vz_own`` (``vel_z`` minus the
    wind drift's z), ``w_air`` (the air's vertical velocity at the agent) and ``emit`` (``Acts.emit`` of this
    step: ``Σ_c floor(256·emit_c)``, the deposit in quanta of 1/256, 12 at the free limit). With tracers also
    ``tracer_z`` and ``tracer_age`` (the tracers' null of the 250-step height persistence)."""
    p = weather.params_from_config(cfg)
    aw = _wind(weather.unpack(s.wx), s.pos, p, cfg, valid=s.alive)
    drift = structures.drift_at(s.wx, s.pos, s.buoy_cap, s.alive, getattr(s, "mat", None), cfg)
    geo = sphere.geometry(cfg)
    base = {
        "alive": s.alive,
        "uid": s.uid,
        "clade": s.clade,
        "cap": s.buoy_cap,
        "z": geo.height(s.pos),
        "vz_own": geo.radial(s.vel - drift, s.pos),
        "w_air": geo.radial(aw.u, s.pos),
        "emit": acts.emit,
    }
    return {**base, "tracer_z": geo.height(tr.pos), "tracer_age": tr.age} if tr is not None else base


def wx_planes(s: State | State3, cfg: Config) -> mx.array:
    """The ten planes of one ``observers/wx_<step>.npy``, ``(10, n, n)`` float32: the nine stored weather
    planes and plane 9 = ``1 - qs`` (the seeding strength ``seed_gain·r`` that the **next** step will use,
    from the end-of-step ``chem`` and weather; zero while ``seed_gain`` is 0). ``chem`` exists only in the
    checkpoints and relaxes at 0.005, so the plane cannot be rebuilt offline."""
    n = weather.params_from_config(cfg).n
    if cfg.weather.seed_gain > 0:  # static
        plane = 1.0 - weather.seeding_qs(s.chem, weather.unpack(s.wx), cfg)
    else:  # a sphere's planes are (N_WX, 12·Nw²): one plane of cells, not of a square
        plane = mx.zeros((12 * n * n,) if dims(cfg).sph else (n, n), _F32)
    return mx.concatenate([s.wx, plane[None]], axis=0)
