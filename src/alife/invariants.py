"""In-graph cheap invariant checks and the exhaustive host check (§22.2, §22.3). Owned by WP4.

``device_check`` runs inside the compiled step and returns the ``Diag`` bits of §22.3. ``full_check`` reads
the whole state on the host (through ``state.to_numpy``) and reports every §22.2 violation as a message.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

from alife import bonds, layout
from alife.config import Config, dims
from alife.state import R_HARD, Diag, LedgerIdx, State, to_numpy, total_energy

# Director env ranges (docs/DIRECTOR.md §3), inclusive (lo, hi); radius and kill_p are set in `_env_ranges`.
ENV_REGEN_MULT: tuple[float, float] = (0.0, 4.0)
ENV_DRIFT_MULT: tuple[float, float] = (0.0, 20.0)
ENV_PATCH_GAIN: tuple[float, float] = (0.0, 4.0)
ENV_SEASON_AMP: tuple[float, float] = (0.0, 1.0)

if TYPE_CHECKING:  # step imports this module; the annotation alone must not create an import cycle
    from alife.step import StepOut

_I32 = mx.int32
_SINK_SLICE = slice(LedgerIdx.HEAT_BASAL, LedgerIdx.HEAT_OVERFLOW + 1)
_VEL_TOL = 1e-6  # §2.2: ‖vel‖ ≤ v_max·(1 + 1e-6) for fp32 rescaling
_SHOW = 5  # offending indices quoted per message


def _ledger_net(ledger: mx.array) -> mx.array:
    """Sources minus sinks as int64 element adds (no int64 reduction on device, §10.8)."""
    net = ledger[LedgerIdx.INJECT_REGEN] + ledger[LedgerIdx.INJECT_IMMIGRATION]
    for i in range(LedgerIdx.HEAT_BASAL, LedgerIdx.HEAT_OVERFLOW + 1):
        net = net - ledger[i]
    return net


def _env_ranges(cfg: Config) -> dict[str, tuple[float, float]]:
    """Inclusive range of each float env field (the integer ones are checked separately)."""
    return {
        "env_regen_mult": ENV_REGEN_MULT,
        "env_season_amp": ENV_SEASON_AMP,
        "env_drift_mult": ENV_DRIFT_MULT,
        "env_patch_gain": ENV_PATCH_GAIN,
        "env_cat_radius": (0.0, cfg.world.size / 2),
        "env_cat_kill_p": (0.0, 1.0),
    }


def env_out_of_range(s: State, cfg: Config) -> mx.array:
    """True if any ``env_*`` value is non-finite or outside its range (reported as ``FIELD_RANGE``)."""
    bad = mx.any(~mx.isfinite(s.env_cat_center)) | (s.env_season_period < 0) | (s.env_imm_request < 0)
    for name, (lo, hi) in _env_ranges(cfg).items():
        a = getattr(s, name)
        bad = bad | mx.any(~mx.isfinite(a) | (a < lo) | (a > hi))
    return bad


def device_check(
    s: State,
    ledger: mx.array,
    prev_total: mx.array,
    max_cell_occ: mx.array,
    max_attackers: mx.array,
    overflow_cells: mx.array,
    v_clips: mx.array,
    cfg: Config,
) -> mx.array:
    """int32 ``Diag`` bits for this step, OR-ed into ``s.diag`` by the step.

    ``s`` is the end-of-step state, ``ledger`` this step's (16,) int64 ledger and ``prev_total`` the
    start-of-step ``TOTAL``. ``TOTAL`` is recomputed from ``s`` (not read from ``s.total``), so a leak
    anywhere in the state sets ``LEDGER``.
    """
    d = dims(cfg)
    alive, dead = s.alive, ~s.alive
    e = s.energy
    total = total_energy(e, alive, s.resource, cfg)
    ledger_bad = ((total - prev_total) != _ledger_net(ledger)) | (
        ledger[LedgerIdx.FLOW_STOLEN] != ledger[LedgerIdx.FLOW_ATTACK_GAIN] + ledger[LedgerIdx.FLOW_SPILL]
    )
    energy_bad = mx.any(alive & ((e < 1) | (e > cfg.energy.e_max))) | mx.any(dead & (e != 0))
    bonded = (s.partner >= 0) | (s.rev >= 0)
    dirty = mx.any(dead & (s.uid != -1)) | mx.any(dead[:, None] & bonded)
    asym = ~bonds.symmetric_ok(s.partner, s.rev, alive)
    nonfinite = ~(
        mx.all(mx.isfinite(s.pos))
        & mx.all(mx.isfinite(s.vel))
        & mx.all(mx.isfinite(s.h))
        & mx.all(mx.isfinite(s.chem))
    )
    field_bad = mx.any((s.resource < 0) | (s.resource > R_HARD))
    if cfg.director.enabled:  # static: the director's env levers are live
        field_bad = field_bad | env_out_of_range(s, cfg)
    headroom = (max_cell_occ >= d.occ_limit) | (max_attackers >= (1 << 15))
    capacity = alive.astype(_I32).sum() > d.n_cap
    pos_bad = mx.any((s.pos < 0) | (s.pos >= cfg.world.size))
    checks = (
        (ledger_bad, Diag.LEDGER),
        (energy_bad, Diag.ENERGY_RANGE),
        (dirty, Diag.DEAD_DIRTY),
        (asym, Diag.BOND_ASYM),
        (nonfinite, Diag.NONFINITE),
        (field_bad, Diag.FIELD_RANGE),
        (headroom, Diag.INT_HEADROOM),
        (capacity, Diag.CAPACITY),
        (pos_bad, Diag.POS_RANGE),
        (overflow_cells > 0, Diag.CELL_OVERFLOW_WARN),
        (v_clips > 0, Diag.VCLIP_WARN),
    )
    bits = mx.array(0, dtype=_I32)
    for cond, bit in checks:
        bits = bits | mx.where(cond, bit.value, 0).astype(_I32)
    return bits


# --- host check ---------------------------------------------------------------------------------------------


def _expected_layout(cfg: Config) -> dict[str, tuple[tuple[int, ...], mx.Dtype]]:
    d = dims(cfg)
    n, m, i32, f32, i64 = d.n, d.m, mx.int32, mx.float32, mx.int64
    return {
        "alive": ((n,), mx.bool_),
        "uid": ((n,), i64),
        "parent_uid": ((n,), i64),
        "clade": ((n,), i32),
        "generation": ((n,), i32),
        "birth_step": ((n,), i32),
        "energy": ((n,), i32),
        "pos": ((n, 2), f32),
        "vel": ((n, 2), f32),
        "heading": ((n,), f32),
        "h": ((n, d.h), f32),
        "msg": ((n, layout.MSG), f32),
        "last_loss": ((n,), i32),
        "last_harvest": ((n,), i32),
        "last_inflow": ((n,), i32),
        "partner": ((n, layout.B), i32),
        "rev": ((n, layout.B), i32),
        "tag": ((n, layout.T), f32),
        "mut_sigma": ((n,), f32),
        "mut_rate": ((n,), f32),
        "offspring_frac": ((n,), f32),
        "brain_cost": ((n,), i32),
        "W1": ((n, d.w1_rows, d.h), d.wdtype),
        "W2": ((n, d.w2_rows, layout.D_OUT), d.wdtype),
        "pend_slot": ((m,), i32),
        "pend_W1": ((m, d.w1_rows, d.h), d.wdtype),
        "pend_W2": ((m, d.w2_rows, layout.D_OUT), d.wdtype),
        "resource": ((d.fg, d.fg), i32),
        "chem": ((layout.NCH, d.fg, d.fg), f32),
        "patch_pos": ((d.pp, 2), f32),
        "patch_vel": ((d.pp, 2), f32),
        "key": ((2,), mx.uint32),
        "step": ((), i32),
        "next_uid": ((), i64),
        "total": ((), i64),
        "total0": ((), i64),
        "ledger_cum": ((len(LedgerIdx),), i64),
        "diag": ((), i32),
        "env_regen_mult": ((), f32),
        "env_season_amp": ((), f32),
        "env_season_period": ((), i32),
        "env_drift_mult": ((), f32),
        "env_patch_gain": ((d.pp,), f32),
        "env_imm_request": ((), i32),
        "env_cat_center": ((2,), f32),
        "env_cat_radius": ((), f32),
        "env_cat_kill_p": ((), f32),
    }


class _Report:
    """Collects violation messages; ``rows`` quotes the first offending indices."""

    def __init__(self) -> None:
        self.problems: list[str] = []

    def rows(self, mask: np.ndarray, what: str) -> None:
        if mask.any():
            idx = np.argwhere(mask)[:_SHOW]
            where = idx[:, 0] if idx.shape[1] == 1 else [tuple(r) for r in idx]
            self.problems.append(f"{what}: {np.count_nonzero(mask)} violation(s), first at {where}")

    def check(self, ok: bool | np.bool_, what: str) -> None:
        if not ok:
            self.problems.append(what)


def _bad_range(a: np.ndarray, lo: float | None, hi: float | None, *, hi_open: bool = False) -> np.ndarray:
    """Elementwise mask of values outside ``[lo, hi]`` (or ``[lo, hi)``), NaN included, reduced to rows."""
    bad = ~np.isfinite(a) if a.dtype.kind == "f" else np.zeros(a.shape, bool)
    with np.errstate(invalid="ignore"):
        if lo is not None:
            bad |= a < lo
        if hi is not None:
            bad |= (a >= hi) if hi_open else (a > hi)
    return bad.reshape(a.shape[0], -1).any(axis=1) if a.ndim > 1 else bad


def _ledger_net_np(ledger: np.ndarray) -> np.int64:
    return ledger[LedgerIdx.INJECT_REGEN] + ledger[LedgerIdx.INJECT_IMMIGRATION] - ledger[_SINK_SLICE].sum()


def _total_np(a: dict[str, np.ndarray], cfg: Config) -> np.int64:
    s = np.int64(cfg.energy.structure)
    alive = a["alive"]
    return (
        a["energy"].astype(np.int64).sum()
        + s * np.count_nonzero(alive)
        + a["resource"].astype(np.int64).sum()
    )


def _check_ledger(r: _Report, a: dict[str, np.ndarray], cfg: Config) -> None:
    total_now, cum = _total_np(a, cfg), a["ledger_cum"]
    r.check(total_now == a["total"], f"ledger: recomputed TOTAL {total_now} != state.total {a['total']}")
    want = a["total0"] + _ledger_net_np(cum)
    r.check(total_now == want, f"ledger: global closure fails, TOTAL {total_now} != total0 + cum = {want}")
    r.rows(cum < 0, "ledger: negative cumulative entry")
    r.check(cum[LedgerIdx.RESERVED] == 0, "ledger: RESERVED entry is not 0")


def _check_bonds(r: _Report, a: dict[str, np.ndarray]) -> None:
    partner, rev, alive = a["partner"], a["rev"], a["alive"]
    n, b = partner.shape
    r.rows((partner < -1) | (partner >= n), "bonds: partner index out of range")
    r.rows((rev < -1) | (rev >= b), "bonds: rev index out of range")
    occ = partner >= 0
    r.rows(occ != (rev >= 0), "bonds: partner < 0 and rev < 0 disagree")
    j, s = np.clip(partner, 0, n - 1), np.clip(rev, 0, b - 1)
    rows = np.arange(n)[:, None]
    good = (
        alive[:, None]
        & alive[j]
        & (partner != rows)
        & (rev >= 0)
        & (partner[j, s] == rows)
        & (rev[j, s] == np.arange(b)[None, :])
    )
    r.rows(occ & ~good, "bonds: asymmetric bond, bond to self or bond with a dead end")
    dup = (partner[:, :, None] == partner[:, None, :]) & occ[:, :, None] & ~np.eye(b, dtype=bool)[None]
    r.rows(dup.any(axis=(1, 2)), "bonds: duplicate partner in a row")


def _check_rows(r: _Report, a: dict[str, np.ndarray], cfg: Config) -> None:
    d = dims(cfg)
    alive, dead = a["alive"], ~a["alive"]
    step, next_uid = a["step"], a["next_uid"]
    uid, e = a["uid"], a["energy"]
    r.check(
        np.count_nonzero(alive) <= d.n_cap, f"capacity: n_alive {np.count_nonzero(alive)} > N_CAP {d.n_cap}"
    )

    # Alive rows.
    r.rows(alive & ((e < 1) | (e > cfg.energy.e_max)), "energy: alive row outside [1, e_max]")
    r.rows(alive & ((uid < 0) | (uid >= next_uid)), "uid: alive uid outside [0, next_uid)")
    alive_uids = uid[alive]
    r.check(np.unique(alive_uids).size == alive_uids.size, "uid: duplicate uid among alive rows")
    r.rows(alive & ((a["parent_uid"] >= uid) | (a["parent_uid"] < -1)), "parent_uid: not in [−1, uid)")
    r.rows(alive & (a["birth_step"] > step - 1), "age: alive row with birth_step > step − 1")

    # Dead rows are canonical (§2.2).
    r.rows(dead & (uid != -1), "dead row: uid != −1")
    r.rows(dead & (e != 0), "dead row: energy != 0")
    for name in ("vel", "h", "msg"):
        r.rows(dead & (a[name] != 0).any(axis=1), f"dead row: {name} != 0")
    for name in ("last_loss", "last_harvest", "last_inflow"):
        r.rows(dead & (a[name] != 0), f"dead row: {name} != 0")
    r.rows(dead[:, None] & ((a["partner"] != -1) | (a["rev"] != -1)), "dead row: bond slot not empty")

    # Ranges, every row (dead rows keep in-range values, §2.2).
    mu, f32 = cfg.mutation, np.float32
    L = cfg.world.size
    r.rows(_bad_range(a["pos"], 0.0, L, hi_open=True), "pos: outside [0, L) or non-finite")
    vel = a["vel"].astype(np.float64)
    speed = np.sqrt((vel * vel).sum(axis=1))
    r.rows(~np.isfinite(speed) | (speed > cfg.physics.v_max * (1 + _VEL_TOL)), "vel: speed > v_max")
    r.rows(
        _bad_range(a["heading"].astype(np.float64), -math.pi, math.pi, hi_open=True),
        "heading: not in [−π, π)",
    )
    r.rows(_bad_range(a["h"], -1.0, 1.0), "h: outside [−1, 1]")
    r.rows(_bad_range(a["msg"], -1.0, 1.0), "msg: outside [−1, 1]")
    r.rows(_bad_range(a["tag"], -1.0, 1.0), "tag: outside [−1, 1]")
    # Genes are stored in f32 and clipped to f32-rounded bounds, so compare in f32.
    r.rows(_bad_range(a["mut_sigma"], f32(mu.sigma_min), f32(mu.sigma_max)), "mut_sigma: outside bounds")
    r.rows(_bad_range(a["mut_rate"], f32(mu.rate_min), f32(mu.rate_max)), "mut_rate: outside bounds")
    r.rows(
        _bad_range(a["offspring_frac"], f32(mu.frac_min), f32(mu.frac_max)), "offspring_frac: outside bounds"
    )
    r.rows(_bad_range(a["clade"], 0, cfg.population.n_init), "clade: outside [0, n_init]")
    r.rows(a["generation"] < 0, "generation: negative")
    for name in ("brain_cost", "last_loss", "last_harvest", "last_inflow"):
        r.rows(a[name] < 0, f"{name}: negative")
    clip = f32(cfg.brain.weight_clip)
    for name in ("W1", "W2", "pend_W1", "pend_W2"):
        r.rows(_bad_range(a[name], -clip, clip), f"{name}: |w| > weight_clip or non-finite")

    # Fields and globals.
    r.rows(_bad_range(a["resource"], 0, R_HARD), "resource: outside [0, R_HARD]")
    r.rows(_bad_range(a["chem"], 0.0, f32(cfg.fields.chem_max)), "chem: outside [0, chem_max] or non-finite")
    r.rows(_bad_range(a["patch_pos"], 0.0, L, hi_open=True), "patch_pos: outside [0, L)")
    r.rows(_bad_range(a["patch_vel"], None, None), "patch_vel: non-finite")
    r.check(step >= 0, f"step: negative ({step})")
    _check_env(r, a, cfg)

    # Pending buffer (§12.6).
    ps = a["pend_slot"]
    r.check(np.unique(ps).size == ps.size, "pending: pend_slot entries are not distinct")
    r.rows((ps < 0) | (ps >= d.n), "pending: pend_slot out of range")
    if step >= 1:
        fresh = alive & (a["birth_step"] == step - 1)
        r.rows(fresh & ~np.isin(np.arange(d.n), ps), "pending: newborn row missing from pend_slot")
    # The next S0 overwrites these rows' weights: each must be dead or last step's newborn (§2.2).
    psc = np.clip(ps, 0, d.n - 1)
    r.rows(alive[psc] & (a["birth_step"][psc] != step - 1), "pending: pend_slot points at an established row")


def _check_env(r: _Report, a: dict[str, np.ndarray], cfg: Config) -> None:
    """Director env levers: finite and in range. Inert values pass, so this is checked for every run."""
    for name, (lo, hi) in _env_ranges(cfg).items():
        r.rows(_bad_range(a[name].reshape(-1, 1), lo, hi), f"{name}: outside [{lo}, {hi}] or non-finite")
    r.rows(_bad_range(a["env_cat_center"].reshape(-1, 1), None, None), "env_cat_center: non-finite")
    r.check(a["env_season_period"] >= 0, f"env_season_period: negative ({a['env_season_period']})")
    r.check(a["env_imm_request"] >= 0, f"env_imm_request: negative ({a['env_imm_request']})")


def _check_step(
    r: _Report,
    a: dict[str, np.ndarray],
    p: dict[str, np.ndarray],
    o: dict[str, np.ndarray] | None,
    cfg: Config,
) -> None:
    """Checks against the previous state (and the step's records, when given)."""
    d = dims(cfg)
    r.check(a["step"] == p["step"] + 1, f"step: {a['step']} does not follow {p['step']}")
    dl = a["ledger_cum"] - p["ledger_cum"]
    dt = a["total"] - p["total"]
    r.check(dt == _ledger_net_np(dl), f"ledger: per-step closure fails, ΔTOTAL {dt} != {_ledger_net_np(dl)}")
    r.check(
        dl[LedgerIdx.FLOW_STOLEN] == dl[LedgerIdx.FLOW_ATTACK_GAIN] + dl[LedgerIdx.FLOW_SPILL],
        "ledger: FLOW_STOLEN != FLOW_ATTACK_GAIN + FLOW_SPILL this step",
    )
    r.rows(dl < 0, "ledger: negative per-step entry")

    alive, uid = a["alive"], a["uid"]
    new_row = alive & (~p["alive"] | (uid != p["uid"]))
    r.rows(new_row & (uid < p["next_uid"]), "resurrection: new alive row with uid < previous next_uid")
    r.check(a["next_uid"] >= p["next_uid"], "uid: next_uid decreased")
    if o is None:
        return
    r.check(np.array_equal(o["ledger"], dl), "ledger: out.ledger != ledger_cum − prev.ledger_cum")
    valid = o["births_valid"]
    nb = np.count_nonzero(valid)
    r.check(
        a["next_uid"] - p["next_uid"] == nb,
        f"uid: next_uid advanced by {a['next_uid'] - p['next_uid']}, not by the {nb} births and immigrants",
    )
    n_alive = np.count_nonzero(alive)
    r.check(nb <= min(d.n_cap - (n_alive - nb), d.m), f"capacity: {nb} births exceed min(free, M)")
    slots, buid = o["births_slot"][valid], o["births_uid"][valid]
    r.check(np.unique(slots).size == slots.size, "births: duplicate slots")
    if ((slots < 0) | (slots >= d.n)).any():
        r.problems.append("births: slot out of range")
        return
    r.rows(~alive[slots], "births: newborn slot is not alive")
    r.rows(uid[slots] != buid, "births: newborn row uid differs from its record")
    r.rows(p["alive"][slots] & (p["uid"][slots] == buid), "births: newborn row held the same uid before")
    r.rows(a["birth_step"][slots] != a["step"] - 1, "births: newborn birth_step != step − 1")
    r.rows(a["pend_slot"][valid] != slots, "pending: pend_slot differs from the birth slots")
    unseen = alive & ~np.isin(uid, p["uid"][p["alive"]])
    r.rows(unseen & ~np.isin(np.arange(d.n), slots), "births: alive row with a new uid is not a newborn")
    dead_uids = o["deaths_uid"][o["deaths_valid"]]
    r.rows(np.isin(uid, dead_uids) & alive, "resurrection: uid from this step's death records is alive")


def full_check(s: State, cfg: Config, prev: State | None = None, out: StepOut | None = None) -> list[str]:
    """Host numpy check of every §22.2 invariant; ``[]`` means OK. Stateless.

    With ``prev`` (the state one step earlier) it also checks the per-step ledger closure and uid issuing;
    with ``out`` (that step's ``StepOut``) the birth records, ``next_uid`` advance and death records too.
    """
    r = _Report()
    want = _expected_layout(cfg)
    for name, arr in zip(State._fields, s, strict=True):
        shape, dtype = want[name]
        if tuple(arr.shape) != shape or arr.dtype != dtype:
            r.problems.append(f"layout: {name} is {arr.dtype}{tuple(arr.shape)}, expected {dtype}{shape}")
    if r.problems:
        return r.problems
    a = {name: to_numpy(arr) for name, arr in zip(State._fields, s, strict=True)}
    _check_ledger(r, a, cfg)
    _check_bonds(r, a)
    _check_rows(r, a, cfg)
    if prev is not None:
        p = {
            name: to_numpy(getattr(prev, name))
            for name in ("alive", "uid", "step", "next_uid", "total", "ledger_cum")
        }
        o = None
        if out is not None:
            o = {
                "ledger": to_numpy(out.ledger),
                "births_valid": to_numpy(out.births.valid),
                "births_slot": to_numpy(out.births.slot),
                "births_uid": to_numpy(out.births.uid),
                "deaths_valid": to_numpy(out.deaths.valid),
                "deaths_uid": to_numpy(out.deaths.uid),
            }
        _check_step(r, a, p, o, cfg)
    return r.problems
