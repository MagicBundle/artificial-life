"""The depth-aware oracle of the diel pilot (DESIGN §49.7, PR D1b; docs/DIEL.md §7.0).

A share ``diel.oracle`` percent of the living agents, chosen by a hash of their ``uid`` when the oracle is
assigned (``assign``: a fork that turns it on, or the founders of a fresh run), follow a scripted daily
vertical schedule that their stratum can reach; the rest of the population is the comparison group. The
per-agent int8 field ``diel_orc`` holds ``+s`` (oracle, stratum ``s``), ``-s`` (comparison, stratum ``s``) or
``0`` (unlabelled: dead rows at assignment and immigrants); children inherit it (``on_birth = "inherit"``),
so the comparison is one of lineages. Strata: 1 floor (z band 0 of the four equal bands of
``metrics.z_band``), 2 mid (bands 1 and 2), 3 top (band 3).

In the step (static branch on ``diel.oracle > 0``) the oracle rows' cap is 1 (``cap``, at the start of the
step, so the upkeep is paid as usual) and their decoded ``mo.buoy`` is replaced before the costs (``plan`` /
``apply``): rise at full lift in the lit half (``S > 1/2``) until ``z_s + up``, sink with no lift in the dark
half until ``z_s - down``, and hold with the balancing lift ``gravity / buoyancy_max`` in between. The
buoyancy due is paid by ``energy.pay_costs3`` as for any other row, so the ledger is untouched. The harvest
of both groups is counted per stratum, photic band (height ``>= H - 2·photic_lambda``, docs/DIEL.md §5.1) and
day or night (``count``, 24 counters).

Two schedules (``diel.oracle_schedule``). 1 is D1b's (§49.12): the excursion's target is reached and held
for the rest of the half. 2 is D1c's round trips (§49.14): by day the target is the centre of the lowest field
layer wholly in the photic band (``photic_target``; a stratum that cannot get there and back in ``REACH`` of
the lit half stays home), by night ``down`` below home; each excursion returns home a margin of
``(1 - REACH)/2`` of its half before the half ends, so no oracle row is above its home height at dusk.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, NamedTuple

import mlx.core as mx
import numpy as np

from alife import diel, sphere

if TYPE_CHECKING:
    from alife.config import Config

_F32, _I32, _I8 = mx.float32, mx.int32, mx.int8
N_BANDS = 4  # metrics.N_BANDS (the four equal z bands of the strata)
STRATA = (1, 2, 3)  # floor, mid, top
STRATUM_NAMES = ("floor", "mid", "top")
GROUPS = (("O", 1), ("C", -1))  # oracle (+s) and comparison (-s)
PHOTIC_DEPTH = (
    2.0  # the photic band is the top PHOTIC_DEPTH · photic_lambda of the column (docs/DIEL.md §5.1)
)
REACH = 0.8  # the schedule uses 80 % of the half-day's reachable distance (docs/DIEL.md §7.0)
HASH_MUL = 2654435761  # Knuth's multiplicative hash of the uid, mod 2^32
AGE_WINDOW = (300, 500)  # the ages of the energy rows (inclusive), docs/DIEL.md §7.0


def counter_names() -> tuple[str, ...]:
    """The 24 counters: harvest quanta per group (O oracle, C comparison), stratum, band (PHOTIC or ALL) and
    half of the day (DAY or NIGHT)."""
    return tuple(
        f"DIEL_{g}{s}_{band}_{half}"
        for g, _ in GROUPS
        for s in STRATA
        for band in ("PHOTIC", "ALL")
        for half in ("DAY", "NIGHT")
    )


COUNTERS = counter_names()


# --- host side: the schedule's constants and the assignment ---------------------------------------------


class Schedule(NamedTuple):
    """Per stratum (index 0 unused): the home height ``z_s``, the excursions ``up`` and ``down``, all in
    world units; the balancing lift ``hold``; and the vertical speeds at cap 1 (world units per step), which
    schedule 2 uses to time the way back."""

    home: tuple[float, ...]
    up: tuple[float, ...]
    down: tuple[float, ...]
    hold: float
    v_rise: float
    v_sink: float


def photic_target(cfg: Config) -> float:
    """Schedule 2's day target: the centre of the lowest field layer lying wholly in the photic band
    (``z >= H - PHOTIC_DEPTH·photic_lambda``); layer ``k`` spans ``[k·fc, (k+1)·fc)``, ``fc = field_cell``. A
    band deeper than the column makes it layer 0's centre."""
    fc = cfg.world.field_cell
    z_ph = sphere.geometry(cfg).height_max - PHOTIC_DEPTH * cfg.fields.photic_lambda
    return (max(0, math.ceil(z_ph / fc - 1e-9)) + 0.5) * fc


def schedule(cfg: Config) -> Schedule:
    """The reachable schedule of each stratum (docs/DIEL.md §7.0), with the oracle's cap 1:
    ``v_rise = (buoyancy_max - gravity)/drag`` and ``v_sink = gravity/drag`` (``physics`` integrates
    ``v = F/γ`` along z, ``γ_z = drag``), both clipped at ``v_max``.

    Schedule 1: ``up = min(z_top - z_s, 0.8·v_rise·P·day_frac)`` and
    ``down = min(z_s - 1, 0.8·v_sink·P·(1 - day_frac))`` with ``z_top`` the top band's centre.
    Schedule 2 (round trips, ``trip = 1/v_rise + 1/v_sink`` steps per unit of height): ``up = z_ph - z_s``
    where ``z_ph = photic_target`` is above home and ``up·trip <= 0.8·P·day_frac``, else 0; ``down =
    min(z_s - 1, 0.8·P·(1 - day_frac)/trip)``."""
    h, ph, c = sphere.geometry(cfg).height_max, cfg.physics, cfg.diel
    band = h / N_BANDS
    home = (0.0, 0.5 * band, 2.0 * band, 3.5 * band)  # band 0's centre, bands 1-2's centre, band 3's centre
    z_top = home[3]
    v_rise = min((ph.buoyancy_max - ph.gravity) / ph.drag, ph.v_max)
    v_sink = min(ph.gravity / ph.drag, ph.v_max)
    hold = min(1.0, max(0.0, ph.gravity / ph.buoyancy_max))
    if c.oracle_schedule == 2:  # static: D1c's round trips (§49.14)
        trip = 1.0 / v_rise + 1.0 / v_sink
        z_ph = photic_target(cfg)
        reach_day, reach_night = REACH * c.period * c.day_frac, REACH * c.period * (1.0 - c.day_frac)
        up = tuple(
            z_ph - home[k] if k > 0 and z_ph > home[k] and (z_ph - home[k]) * trip <= reach_day else 0.0
            for k in range(4)
        )
        down = tuple(0.0 if k == 0 else max(0.0, min(home[k] - 1.0, reach_night / trip)) for k in range(4))
        return Schedule(home, up, down, hold, v_rise, v_sink)
    up = tuple(
        0.0 if k == 0 else min(z_top - home[k], REACH * v_rise * c.period * c.day_frac) for k in range(4)
    )
    down = tuple(
        0.0 if k == 0 else max(0.0, min(home[k] - 1.0, REACH * v_sink * c.period * (1.0 - c.day_frac)))
        for k in range(4)
    )
    return Schedule(home, up, down, hold, v_rise, v_sink)


def stratum_of(height: np.ndarray, cfg: Config) -> np.ndarray:
    """(N,) int8 stratum 1..3 of each height: the band ``floor(4·z/H)`` clipped (``metrics.z_band``), band 0
    floor, 1-2 mid, 3 top."""
    band = np.clip(
        np.floor((height.astype(np.float64) * N_BANDS) / sphere.geometry(cfg).height_max), 0, N_BANDS - 1
    )
    return np.where(band == 0, 1, np.where(band == 3, 3, 2)).astype(np.int8)


def chosen(uid: np.ndarray, percent: int) -> np.ndarray:
    """(N,) bool: the uids whose hash ``(uid·2654435761 mod 2^32)/2^32`` is below ``percent/100``."""
    h = (uid.astype(np.uint64) * np.uint64(HASH_MUL)) % np.uint64(1 << 32)
    return h.astype(np.float64) / float(1 << 32) < percent / 100.0


def assign(s: Any, cfg: Config) -> Any:
    """``s`` with ``diel_orc`` assigned from its living rows (host side, once: the pilot's start) and the
    oracle rows' cap set to 1. Dead rows get 0. ``s`` must carry no earlier assignment: the cap of a former
    oracle row is not restorable, so ``cli._prepare_fork`` refuses a fork that changes a non-zero
    ``diel.oracle``."""
    alive = np.asarray(s.alive)
    height = np.asarray(sphere.geometry(cfg).height(s.pos))
    strat = stratum_of(height, cfg)
    pick = chosen(np.asarray(s.uid), cfg.diel.oracle)
    orc = np.where(alive, np.where(pick, strat, -strat), 0).astype(np.int8)
    cap = np.where(alive & pick, np.float32(1.0), np.asarray(s.buoy_cap)).astype(np.float32)
    out = s._replace(diel_orc=mx.array(orc, dtype=_I8), buoy_cap=mx.array(cap, dtype=_F32))
    mx.eval(out)
    return out


# --- in the step -----------------------------------------------------------------------------------------


class Plan(NamedTuple):
    """The oracle's decision of one step (``_Flows.ext.diel``): the start-of-step labels, where the row was
    (photic band), the half of the day, and the scripted lift."""

    orc: mx.array  # (N,) int8: diel_orc at the start of the step
    photic: mx.array  # (N,) bool: start-of-step height in the photic band
    day: mx.array  # () bool: the sun is up (S > 1/2)
    on: mx.array  # (N,) bool: an alive oracle row (its mo.buoy is replaced)
    buoy: mx.array  # (N,) f32: the scripted lift in [0, 1]


def cap(s: Any, cfg: Config) -> mx.array:
    """The oracle rows' cap is 1 (docs/DIEL.md §7.0); the others keep theirs."""
    return mx.where(s.diel_orc > 0, mx.array(1.0, dtype=_F32), s.buoy_cap)


def plan(s: Any, cfg: Config) -> Plan:
    """The scripted lift of every row from its stratum, its start-of-step height and the half of the day.
    Pure MLX: the per-stratum constants are a host table read with ``take``."""
    sc = schedule(cfg)
    z = sphere.geometry(cfg).height(s.pos)
    _, sv = diel.light(s.step, s.diel_ctl[1], cfg)
    day = sv > 0.5
    k = mx.clip(mx.abs(s.diel_orc.astype(_I32)), 0, 3)
    home = mx.take(mx.array(sc.home, dtype=_F32), k)
    target = mx.where(
        day,
        home + mx.take(mx.array(sc.up, dtype=_F32), k),
        home - mx.take(mx.array(sc.down, dtype=_F32), k),
    )
    hold = mx.array(sc.hold, dtype=_F32)
    if cfg.diel.oracle_schedule == 2:  # static: D1c's round trips (§49.14)
        back = returning(s.step, s.diel_ctl[1], day, z, home, sc, cfg)
        buoy = mx.where(
            day,
            mx.where(back, mx.where(z > home, 0.0, hold), mx.where(z < target, 1.0, hold)),
            mx.where(back, mx.where(z < home, 1.0, hold), mx.where(z > target, 0.0, hold)),
        )
    else:
        buoy = mx.where(day, mx.where(z < target, 1.0, hold), mx.where(z > target, 0.0, hold))
    photic = z >= sphere.geometry(cfg).height_max - PHOTIC_DEPTH * cfg.fields.photic_lambda
    return Plan(s.diel_orc, photic, day, s.alive & (s.diel_orc > 0), buoy.astype(_F32))


def time_left(step: mx.array, day_eff: mx.array, day: mx.array, cfg: Config) -> tuple[mx.array, mx.array]:
    """The steps left in the current half of the day and that half's length (f32 scalars). Global mode:
    noon at phase 0, the sun up (``S > 1/2``, i.e. ``cos th > cos(pi·f)``) for phases within ``f·P/2`` of
    noon, ``f = day_eff``. Where the light's half and the phase disagree (the boundary step, float32), the
    time left is 0, so the row heads home."""
    per = cfg.diel.period
    phase = (step % per).astype(_F32)
    half_d = day_eff * per
    t_dusk = 0.5 * half_d - phase
    t_dusk = mx.where(t_dusk < 0, t_dusk + per, t_dusk)
    t_dawn = (per - 0.5 * half_d) - phase
    t_dawn = mx.where(t_dawn < 0, t_dawn + per, t_dawn)
    half = mx.where(day, half_d, per - half_d)
    tau = mx.where(day, t_dusk, t_dawn)
    return mx.where(tau > half, 0.0, tau), half


def returning(
    step: mx.array, day_eff: mx.array, day: mx.array, z: mx.array, home: mx.array, sc: Schedule, cfg: Config
) -> mx.array:
    """(N,) bool, schedule 2: the row must head home now to be there ``(1 - REACH)/2`` of the half before it
    ends: the time left minus that margin is at most the time home (``(z - home)/v_sink`` by day,
    ``(home - z)/v_rise`` by night)."""
    tau, half = time_left(step, day_eff, day, cfg)
    margin = 0.5 * (1.0 - REACH) * half
    t_home = mx.where(day, (z - home) / sc.v_sink, (home - z) / sc.v_rise)
    return tau - margin <= t_home


def apply(mo: Any, p: Plan) -> Any:
    """``mo`` (``actions.Motion3``) with the oracle rows' lift replaced by the schedule's."""
    return mo._replace(buoy=mx.where(p.on, p.buoy, mo.buoy))


def count(s_new: Any, alive0: mx.array, flows: Any, cfg: Config) -> dict[str, mx.array]:
    """The 24 harvest counters (``COUNTERS``) from the step's harvest ``flows.got`` and the plan."""
    p: Plan = flows.ext.diel
    got = mx.where(alive0, flows.got, 0).astype(_I32)
    orc = p.orc.astype(_I32)
    out: dict[str, mx.array] = {}
    zero = mx.array(0, dtype=_I32)
    for g, sign in GROUPS:
        for st in STRATA:
            m = orc == sign * st
            q_all = mx.where(m, got, 0).sum()
            q_ph = mx.where(m & p.photic, got, 0).sum()
            for band, q in (("PHOTIC", q_ph), ("ALL", q_all)):
                out[f"DIEL_{g}{st}_{band}_DAY"] = mx.where(p.day, q, zero)
                out[f"DIEL_{g}{st}_{band}_NIGHT"] = mx.where(p.day, zero, q)
    return out


# --- metrics ------------------------------------------------------------------------------------------------


def snap_dev(s: Any, cfg: Config, heavy: bool) -> dict[str, mx.array]:
    return {"diel.orc": s.diel_orc} if cfg.diel.oracle > 0 else {}


def snap_row(h: Any, cfg: Config) -> dict[str, Any]:
    """Per group and stratum (order O1 O2 O3 C1 C2 C3): ``diel_orc_pop`` (alive rows), ``diel_orc_n`` and
    ``diel_orc_energy`` (the count and the mean energy of the alive rows aged 300 to 500, None when empty):
    the energy rows of the pilot's measure ``G_s`` (docs/DIEL.md §7.0)."""
    if "diel.orc" not in h:
        return {}
    alive = h["alive"].astype(bool)
    orc = h["diel.orc"].astype(np.int64)
    age = int(h["step"]) - 1 - h["birth_step"].astype(np.int64)
    window = alive & (age >= AGE_WINDOW[0]) & (age <= AGE_WINDOW[1])
    e = h["energy"].astype(np.int64)
    pop, n, energy = [], [], []
    for _, sign in GROUPS:
        for st in STRATA:
            m = orc == sign * st
            pop.append(int((alive & m).sum()))
            w = window & m
            n.append(int(w.sum()))
            energy.append(float(e[w].mean()) if w.any() else None)
    return {"diel_orc_pop": pop, "diel_orc_n": n, "diel_orc_energy": energy}


def interval_keys(names: tuple[str, ...], cs: Any, at: int) -> dict[str, int]:
    """The metrics row's interval keys of the counter block: ``diel_harv_<g><s>_<band>_<half>`` in quanta."""
    return {f"diel_harv_{n.removeprefix('DIEL_').lower()}": int(cs[at + i]) for i, n in enumerate(names)}
