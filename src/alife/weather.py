"""The W0 weather prototype, model B (docs/STORMS.md §3.2): two stacked shallow-water layers with moisture.

A periodic C-grid (160 × 160 cells of 2 units by default), a semi-lagrangian advection step,
forward–backward continuity and pressure with a positivity limiter, a Strang-split Coriolis rotation with
``f(y) = f0·sin(2πy/L)`` (two hemispheres on the torus), evaporation over a warm-pool map, condensation that
moves mass from layer 1 to layer 2 (the latent heating), radiative mass return, damping, and seeded
disturbances. ``sample_wind`` diagnoses the wind the organisms would feel from the layer fields (STORMS §4).
The normative spec is the W0 implementation spec (section numbers in the docstrings below, "spec §"); the
engine does not import this module until W1.

Pure MLX like the simulation modules (§22.6 rule 5): no host reads, no branching on array values (static
branches only on ``WeatherParams`` fields, ``debug`` and the presence of the optional momentum forcing), fixed
shapes, every random draw with ``key=``, no scatter outside ``ops`` (``shade_mult``'s one count), no matmul
(F20), float32 throughout.

MLX 0.32.3's ``mx.compile`` crashes when the same roll of the same array is built twice and a later stage
again builds duplicate rolls of a result that depends on them (spec §2.1). Every stage here takes the shifts
of an array once and reuses them; never apply ``_east``, ``_west``, ``_north`` or ``_south`` twice to one
array.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any, Final, NamedTuple

import mlx.core as mx

from alife import ops

if TYPE_CHECKING:
    from alife.config import Config

_F32 = mx.float32
_I32 = mx.int32

# (x, y, sx, sy, amp, vx, vy): centre, Gaussian widths (sx <= 0: a zonal band), amplitude, drift per step.
# The W0 result (docs/STORMS_W0.md, set r4_3p36_q111_v009): 3 pools per hemisphere, 32 x 16, 36 units from
# each equator, drifting west at 0.009 units per step (D8). The centres keep the 6 decimals of the confirmed
# parameter file, so ``WeatherParams()`` is that set exactly (params hash 7832e060).
DEFAULT_POOLS: tuple[tuple[float, ...], ...] = (
    (53.333333, 36.0, 32.0, 16.0, 1.0, -0.009, 0.0),
    (160.0, 36.0, 32.0, 16.0, 1.0, -0.009, 0.0),
    (266.666667, 36.0, 32.0, 16.0, 1.0, -0.009, 0.0),
    (0.0, 284.0, 32.0, 16.0, 1.0, -0.009, 0.0),
    (106.666667, 284.0, 32.0, 16.0, 1.0, -0.009, 0.0),
    (213.333333, 284.0, 32.0, 16.0, 1.0, -0.009, 0.0),
)

# The pools of the W0 spec's starting set (§3.3): 2 static pools per hemisphere, 48 units from each equator.
START_POOLS: tuple[tuple[float, ...], ...] = (
    (80.0, 48.0, 32.0, 16.0, 1.0, 0.0, 0.0),
    (240.0, 48.0, 32.0, 16.0, 1.0, 0.0, 0.0),
    (0.0, 272.0, 32.0, 16.0, 1.0, 0.0, 0.0),
    (160.0, 272.0, 32.0, 16.0, 1.0, 0.0, 0.0),
)

# Largest stable uniform speed of the dry coupled step (spec §2.5 rule 14, measured at 45 degrees, no
# friction), by (sl_order, kappa_d). Only the tabulated numerics have an entry.
MEAN_FLOW_LIMIT: dict[tuple[int, float], float] = {(1, 0.04): 0.352, (3, 0.10): 0.370}


# ``WeatherParams`` fields added after the W0 study (omitted from ``to_dict`` at their default).
_POST_W0: Final = ("seed_dmin",)


@dataclass(frozen=True)
class WeatherParams:
    """The parameters of the W0 weather model; the defaults are the W0 result (``docs/STORMS_W0.md``).

    The tuning changed four fields of the spec's starting set (§3.3, kept as ``START_SET``): ``gamma`` 24 →
    21.6, ``q_pool`` 1.10 → 1.11, ``pools`` (``START_POOLS`` → ``DEFAULT_POOLS``, with drift) and ``zeta_th``
    0.0075 → 0.00625. Every field is hashable (``pools`` is a tuple of 7-tuples), so a ``WeatherParams`` keys
    ``functools.cache`` and closes over a compiled step. ``validate()`` returns messages (empty = valid); the
    model itself never validates, so a test may build parameters that break a rule on purpose.
    """

    # grid
    n: int = 160
    dx: float = 2.0
    dt: float = 1.0
    height: float = 80.0
    # layers
    H1: float = 32.0
    H2: float = 48.0
    g_top: float = 9.18256e-3
    g_prime: float = 5.16862e-3
    h_min: float = 2.0
    # rotation
    f0: float = 0.025
    # moisture
    Qs: float = 1.0
    gamma: float = 21.6
    tau_c: float = 8.0
    k_E: float = 0.1
    u_gust: float = 0.02
    q_bg: float = 0.70
    q_pool: float = 1.11
    m_min: float = 1.0
    # relaxation and damping
    tau_rad: float = 2000.0
    r1: float = 2e-4
    c_d: float = 2e-3
    r2: float = 1e-4
    c_half: float = 1e-3
    # numerics
    nu: float = 0.0
    kappa_d: float = 0.04
    sl_order: int = 1
    u_safe: float = 0.3
    # warm pools
    pools: tuple[tuple[float, ...], ...] = DEFAULT_POOLS
    # seeds
    seed_rate: float = 0.02
    seed_v: float = 0.02
    seed_sigma: float = 12.0
    seed_jitter: float = 0.5
    seed_q: float = 0.0
    seed_qsigma: float = 6.0
    seed_w2: float = 0.0
    # agent wind
    alpha0: float = 0.25
    alpha1: float = 0.75
    u_max: float = 0.3
    w_edge: float = 0.03
    z_sl: float = 4.0
    # detection (host side, spec §6)
    detect_every: int = 25
    zeta_th: float = 0.00625
    p_th: float = 1e-4
    a_min: int = 25
    r_min: int = 4
    miss_grace: int = 2
    min_dets: int = 3
    # organisms -> weather (W2, DESIGN §32): the floor of the cloud depth that the seeding band uses
    seed_dmin: float = 8.0

    def __post_init__(self) -> None:
        pools = tuple(tuple(float(x) for x in row) for row in self.pools)
        object.__setattr__(self, "pools", pools)

    # --- JSON ---------------------------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Plain Python values (pools as lists of lists), ready for ``json.dumps``. The fields added after W0
        (``_POST_W0``) are left out at their default, so the W0 parameter hashes (``analysis/w0.params_hash``)
        and the parameter files of the W0 runs are unchanged."""
        dflt = WeatherParams.__dataclass_fields__
        return {
            f.name: ([list(row) for row in self.pools] if f.name == "pools" else getattr(self, f.name))
            for f in fields(self)
            if not (f.name in _POST_W0 and getattr(self, f.name) == dflt[f.name].default)
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], base: WeatherParams | None = None) -> WeatherParams:
        """A sparse override over ``base`` (default: the defaults); unknown keys are rejected (like the
        presets)."""
        base = cls() if base is None else base
        names = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - names)
        if unknown:
            raise ValueError(f"unknown weather parameter(s): {', '.join(unknown)}")
        return replace(base, **{k: _coerce(base, k, val) for k, val in d.items()})

    # --- derived values -----------------------------------------------------------------------------------

    @property
    def L(self) -> float:
        return self.n * self.dx

    @property
    def _T(self) -> float:
        return self.g_top * (self.H1 + self.H2) + self.g_prime * self.H1

    @property
    def _D(self) -> float:
        return self.g_top * self.g_prime * self.H1 * self.H2

    def _speeds(self) -> tuple[float, float]:
        disc = math.sqrt(max(self._T**2 - 4.0 * self._D, 0.0))
        return math.sqrt(0.5 * (self._T + disc)), math.sqrt(max(0.5 * (self._T - disc), 0.0))

    @property
    def c_ext(self) -> float:
        """Fast (external) gravity-wave speed."""
        return self._speeds()[0]

    @property
    def c_bc(self) -> float:
        """Slow (baroclinic) gravity-wave speed."""
        return self._speeds()[1]

    @property
    def eps(self) -> float:
        """Moist feedback strength ``γ·Q_s/H1``."""
        return self.gamma * self.Qs / self.H1

    @property
    def courant(self) -> float:
        return self.c_ext * self.dt / self.dx

    @property
    def ld_pole(self) -> float:
        """Dry deformation radius at the poles (``|f| = f0``)."""
        return self.c_bc / self.f0 if self.f0 > 0 else math.inf

    @property
    def c_moist(self) -> float:
        """Slow-mode speed in saturated, converging air (the exact two-layer eigenproblem, spec rule 6b)."""
        t_m = self._T - self.eps * self.H1 * self.g_prime
        d_m = (1.0 - self.eps) * self._D
        return math.sqrt(max(0.5 * (t_m - math.sqrt(max(t_m * t_m - 4.0 * d_m, 0.0))), 0.0))

    @property
    def f_pool(self) -> float:
        """Largest ``|f|`` over the pool centres (``f0`` if any pool drifts in y)."""
        if any(row[6] != 0.0 for row in self.pools):
            return self.f0
        return max(
            (abs(self.f0 * math.sin(2.0 * math.pi * row[1] / self.L)) for row in self.pools), default=0.0
        )

    @property
    def ld_moist_pool(self) -> float:
        return self.c_moist / self.f_pool if self.f_pool > 0 else math.inf

    @property
    def ld_moist_pole(self) -> float:
        return self.c_moist / self.f0 if self.f0 > 0 else math.inf

    def derived(self) -> dict[str, float | None]:
        """The derived values ``params.json`` records (spec §3.3)."""
        fp = self.f_pool
        dh = self.H1 * (1.0 - self.eps) - self.m_min
        r_core = 10.0
        potential = 0.5 * (-fp * r_core + math.sqrt((fp * r_core) ** 2 + 4.0 * self.g_prime * max(dh, 0.0)))
        return {
            "L": self.L,
            "c_ext": self.c_ext,
            "c_bc": self.c_bc,
            "c_moist": self.c_moist,
            "courant": self.courant,
            "eps": self.eps,
            "ld_pole": self.ld_pole,
            "ld_pool": self.c_bc / fp if fp > 0 else math.inf,
            "ld_moist_pool": self.ld_moist_pool,
            "ld_moist_pole": self.ld_moist_pole,
            "ld_moist_pole_cells": self.ld_moist_pole / self.dx,
            "ld_moist_pool_cells": self.ld_moist_pool / self.dx,
            "f_pool": fp,
            "inertial_period_pole": 2.0 * math.pi / self.f0 if self.f0 > 0 else math.inf,
            "inertial_period_pool": 2.0 * math.pi / fp if fp > 0 else math.inf,
            "crossing_steps": self.dx / self.c_bc,
            "potential_intensity": potential,
            "alpha_max": self.alpha0 + self.alpha1,
            "mean_flow_limit": MEAN_FLOW_LIMIT.get((self.sl_order, self.kappa_d)),
        }

    # --- validation (spec §2.5 rules 1-14 and the parameter-only part of 16) ---------------------------

    def validate(self) -> list[str]:
        """Messages for every violated rule of spec §2.5 that needs no ``Config`` (empty = valid)."""
        bad: list[str] = []

        def req(ok: bool, msg: str) -> None:
            if not ok:
                bad.append(msg)

        req(self.n >= 8 and self.dx > 0 and self.dt > 0, "rule 1: need n >= 8, dx > 0, dt > 0")
        req(
            0 < self.h_min < self.H1 and self.h_min < self.H2 and abs(self.H1 + self.H2 - self.height) < 1e-9,
            "rule 2: need 0 < h_min < H1, h_min < H2 and H1 + H2 = height",
        )
        req(
            self.g_top > 0 and self.g_prime > 0 and self.g_prime < self.g_top,
            "rule 3: need g_top > 0, g_prime > 0 and g_prime < g_top",
        )
        if bad:  # the derived speeds below are meaningless (or complex) without these
            return bad
        c_ext, c_bc = self._speeds()
        req(
            c_ext * self.dt / self.dx <= 0.5,
            f"rule 4: fast-mode Courant {c_ext * self.dt / self.dx:.3f} > 0.5",
        )
        req(self.f0 >= 0 and self.f0 * self.dt <= 0.5, "rule 5: need 0 <= f0 and f0*dt <= 0.5")
        if self.f0 > 0:
            req(c_bc / self.f0 >= 3 * self.dx, f"rule 6: L_d {c_bc / self.f0:.3g} < 3 dx (c_bc/f0 >= 3dx)")
            if self.gamma > 0 and self.f_pool > 0:
                req(
                    self.ld_moist_pool >= 3 * self.dx,
                    f"rule 6b: moist deformation radius at the pools {self.ld_moist_pool:.3g} < 3 dx",
                )
        req(
            self.tau_c >= max(self.dt, self.dx / c_bc),
            f"rule 7: tau_c {self.tau_c:g} < max(dt, dx/c_bc = {self.dx / c_bc:.3g})",
        )
        req(
            self.gamma >= 0 and self.eps < 1 and self.H1 * (1 - self.eps) > self.m_min,
            "rule 8: need gamma >= 0, eps = gamma*Qs/H1 < 1 and H1*(1 - eps) > m_min",
        )
        req(0 <= self.q_bg <= self.q_pool <= 1.2, "rule 9: need 0 <= q_bg <= q_pool <= 1.2")
        req(
            self.k_E >= 0 and self.u_gust >= 0 and self.dt * self.k_E * (self.u_safe + self.u_gust) <= 1,
            "rule 10: need k_E, u_gust >= 0 and dt*k_E*(u_safe + u_gust) <= 1",
        )
        req(self.tau_rad == 0 or self.tau_rad >= self.dt, "rule 11: tau_rad must be 0 (off) or >= dt")
        req(
            min(self.r1, self.c_d, self.r2, self.nu, self.kappa_d) >= 0
            and self.nu * self.dt / self.dx**2 + self.kappa_d <= 0.125,
            "rule 11: need r1, c_d, r2, nu, kappa_d >= 0 and nu*dt/dx^2 + kappa_d <= 1/8",
        )
        req(
            0 <= self.seed_rate <= 1
            and self.seed_v >= 0
            and self.dx <= self.seed_sigma <= self.L / 8
            and 0 <= self.seed_jitter < 1
            and self.seed_q >= 0
            and self.seed_qsigma > 0
            and self.seed_w2 >= 0,
            "rule 12: seed parameters out of range (rate in [0,1], dx <= sigma <= L/8, jitter in [0,1))",
        )
        req(self.m_min >= 0, "rule 13: m_min must be >= 0")
        req(
            all(len(row) == 7 and row[4] >= 0 and row[3] > 0 for row in self.pools),
            "rule 13: every pool needs 7 entries, amp >= 0 and sy > 0",
        )
        req(self.sl_order in (1, 3), "rule 14: sl_order must be 1 or 3")
        floor = {1: 0.04, 3: 0.10}.get(self.sl_order, 0.04)
        req(self.kappa_d >= floor, f"rule 14: kappa_d {self.kappa_d:g} < {floor:g} (mean-flow stability)")
        req(0 < self.u_safe <= 0.3, "rule 14: need 0 < u_safe <= 0.3")
        req(
            self.alpha0 >= 0 and self.alpha1 >= 0 and self.alpha0 + self.alpha1 <= 1,
            "rule 16: need alpha0, alpha1 >= 0 and alpha_max = alpha0 + alpha1 <= 1",
        )
        req(
            self.u_max > 0 and self.w_edge >= 0 and self.z_sl > 0,
            "rule 18: need u_max > 0, w_edge >= 0 and z_sl > 0",
        )
        req(
            self.detect_every >= 1
            and self.a_min >= 1
            and self.r_min >= 1
            and self.miss_grace >= 0
            and self.min_dets >= 1
            and self.zeta_th > 0
            and self.p_th >= 0,
            "rule 18: detection parameters out of range",
        )
        return bad


# The starting set of the W0 spec (§3.3), where the tuning began: the reference of the model tests, and the
# base that the tuning rounds' sparse parameter files override (each run's ``params.json`` has its full set).
START_SET = WeatherParams(gamma=24.0, q_pool=1.10, pools=START_POOLS, zeta_th=0.0075)


def _coerce(base: WeatherParams, key: str, val: Any) -> Any:
    """``val`` converted to the type of ``base``'s field ``key`` (pools: a tuple of float 7-tuples)."""
    if key == "pools":
        return tuple(tuple(float(x) for x in row) for row in val)
    cur = getattr(base, key)
    if isinstance(cur, int):
        if val != int(val):
            raise ValueError(f"weather parameter {key} must be an integer, got {val!r}")
        return int(val)
    return type(cur)(val)


def check_world(p: WeatherParams, cfg: Config) -> list[str]:
    """Rules 15 and 16 against a 3D ``Config`` (spec §2.5): world geometry, load limits (STORMS §3.3)."""
    bad: list[str] = []
    w, ph = cfg.world, cfg.physics
    if w.dims != 3:
        bad.append("rule 15: the config must be a 3D world (world.dims = 3)")
    if abs(p.L - w.size) > 1e-9:
        bad.append(f"rule 15: weather L = {p.L:g} must equal world.size = {w.size:g}")
    if abs(p.height - w.height) > 1e-9:
        bad.append(f"rule 15: weather height {p.height:g} must equal world.height = {w.height:g}")
    if not 0 < p.z_sl <= w.field_cell:
        bad.append(f"rule 15: need 0 < z_sl <= world.field_cell = {w.field_cell:g}, got {p.z_sl:g}")
    a_max = p.alpha0 + p.alpha1
    g, bmax = ph.gravity, ph.buoyancy_max
    wind_w = a_max * ph.drag * p.w_edge
    load = ph.thrust_max + max(g + wind_w, bmax - g + wind_w)
    if not load < 2 * ph.radius * ph.k_repel:
        bad.append(
            f"rule 16: thrust_max + max(g + a*drag*w_edge, bmax - g + a*drag*w_edge) = {load:.4g} "
            f"must be < 2*radius*k_repel = {2 * ph.radius * ph.k_repel:.4g}"
        )
    speed = (ph.thrust_max + a_max * ph.drag * p.u_max) / ph.drag if ph.drag > 0 else math.inf
    if not speed <= ph.v_max:
        bad.append(f"rule 16: (thrust_max + a*drag*u_max)/drag = {speed:.4g} must be <= v_max = {ph.v_max:g}")
    return bad


# --- state, forcing, diagnostics --------------------------------------------------------------------------


class WeatherState(NamedTuple):
    """The prognostic fields, all (N, N) float32 (spec §2.2): thicknesses and winds of the two layers on the
    C-grid (u at east faces, v at north faces), the lower-layer humidity ``Q`` (humidity integrated over
    layer 1, in units of ``Q_s`` per unit area), and the rain rate ``P`` and cloud depth ``Dc`` of the last
    step.
    """

    h1: mx.array
    h2: mx.array
    u1: mx.array
    v1: mx.array
    u2: mx.array
    v2: mx.array
    Q: mx.array
    P: mx.array
    Dc: mx.array


class WeatherForcing(NamedTuple):
    """Inputs that are arrays so a lever can change them without a retrace.

    ``T_s`` is the (N, N) warm-pool map in [0, 1]; ``qs`` multiplies ``Q_s`` in the condensation threshold
    (1.0 in W0). The optional momentum forcing ``(fu1, fv1, fu2, fv2)`` is an acceleration on the faces
    (velocity per unit time, added as ``dt·f`` after the pressure kick). Give all four or none: with arrays
    the compiled graph is the same for zeros and for a push, so a caller that wants an impulse passes zeros
    while it is idle.
    """

    T_s: mx.array
    qs: mx.array
    fu1: mx.array | None = None
    fv1: mx.array | None = None
    fu2: mx.array | None = None
    fv2: mx.array | None = None


class WeatherDiag(NamedTuple):
    """Per-step diagnostics, all derived from inputs (MLX 0.32 compile rule); ``E``, ``C``, ``Qseed`` only
    with ``debug``. ``courant_max``, ``speed_max``, ``hyper_max`` and ``n_safety`` are evaluated on the winds
    entering the damping sub-step (``_damp``); the other counters and sums are end-of-step."""

    n_limited: mx.array
    n_clamp: mx.array
    clamp_mass: mx.array
    q_clamp: mx.array
    n_cond_limited: mx.array
    n_relax_limited: mx.array
    n_evap_limited: mx.array
    n_safety: mx.array
    courant_max: mx.array
    speed_max: mx.array
    hyper_max: mx.array
    eps_local_max: mx.array
    seed_on: mx.array
    sum_E: mx.array
    sum_C: mx.array
    sum_Qseed: mx.array
    E: mx.array | None = None
    C: mx.array | None = None
    Qseed: mx.array | None = None


class AgentWind(NamedTuple):
    u: mx.array  # (M, 3): (u_h, v_h, w) after the clips
    n_clip_h: mx.array  # () int32: samples with ||u_h|| > u_max
    n_clip_w: mx.array  # () int32: edge-layer samples with |w| > w_edge
    u_eddy: mx.array | None = None  # (M, 2): u_h minus the zonal mean of the same layer, only with eddy=True


def warm_pools(p: WeatherParams, t: mx.array | None = None) -> mx.array:
    """The warm-pool map ``T_s`` at the cell centres, (N, N) float32 in [0, 1] (spec §3.3).

    ``T_s = clip(Σ amp·exp(−dx²/2σx² − dy²/2σy²), 0, 1)`` with minimum-image distances; ``sx <= 0`` makes a
    zonal band. With every drift ``(vx, vy) = (0, 0)`` the map ignores ``t`` (a static branch); with drift
    ``t`` is an int32 () array (a Python int would be a compile constant) and the centres move linearly with
    wrap.
    """
    n, dx, length = p.n, p.dx, p.L
    xc = (mx.arange(n, dtype=_F32) + 0.5) * dx
    drift = any(row[5] != 0.0 or row[6] != 0.0 for row in p.pools)
    tf = t.astype(_F32) if (drift and t is not None) else None
    total = mx.zeros((n, n), dtype=_F32)
    for x0, y0, sx, sy, amp, vx, vy in p.pools:
        cx, cy = x0, y0
        if tf is not None:
            cx, cy = x0 + vx * tf, y0 + vy * tf
        dyv = ops.min_image(xc - cy, length)[:, None]
        e = -(dyv * dyv) / (2.0 * sy * sy)
        if sx > 0:
            dxv = ops.min_image(xc - cx, length)[None, :]
            e = e - (dxv * dxv) / (2.0 * sx * sx)
        total = total + amp * mx.exp(e)
    return mx.clip(total, 0.0, 1.0)


def default_forcing(p: WeatherParams, t: mx.array | None = None) -> WeatherForcing:
    """``T_s`` from the pools and ``qs`` = 1, no momentum forcing."""
    return WeatherForcing(warm_pools(p, t), mx.ones((p.n, p.n), dtype=_F32))


def init_state(p: WeatherParams, fo: WeatherForcing) -> WeatherState:
    """The rest state: ``h = (H1, H2)``, ``u = v = 0``, ``Q = Q_e(T_s)`` capped at ``Q_s``, ``P = Dc = 0``."""
    assert fo.T_s.dtype == _F32 and fo.qs.dtype == _F32, "weather forcing must be float32"
    shape = (p.n, p.n)
    zero = mx.zeros(shape, dtype=_F32)
    qe = p.Qs * (p.q_bg + (p.q_pool - p.q_bg) * fo.T_s)
    return WeatherState(
        h1=mx.full(shape, p.H1, dtype=_F32),
        h2=mx.full(shape, p.H2, dtype=_F32),
        u1=zero,
        v1=zero,
        u2=zero,
        v2=zero,
        Q=mx.minimum(qe, p.Qs),
        P=zero,
        Dc=zero,
    )


# --- shifts and tables --------------------------------------------------------------------------------------
# Periodic neighbours on the last two axes: east a[j, i+1], west a[j, i-1], north a[j+1, i], south a[j-1, i].
# Each shifted copy of an array is built once and reused (module docstring).


def _east(a: mx.array) -> mx.array:
    return mx.roll(a, -1, axis=-1)


def _west(a: mx.array) -> mx.array:
    return mx.roll(a, 1, axis=-1)


def _north(a: mx.array) -> mx.array:
    return mx.roll(a, -1, axis=-2)


def _south(a: mx.array) -> mx.array:
    return mx.roll(a, 1, axis=-2)


@functools.cache
def _rot_lists(n: int, dx: float, dt: float, f0: float, length: float) -> tuple[list[float], ...]:
    """Pair-rotation angle tables as Python lists (float64 math): cos and sin of ``φ_S[j]``, ``φ_N[j]`` per
    u-row, with ``φ_N[N−1−j] = −φ_S[j]`` and the cosines equal, bitwise, by construction (spec §2.3 S3)."""
    phi_s = [0.25 * dt * f0 * math.sin(2.0 * math.pi * (j + 0.25) * dx / length) for j in range(n)]
    cs = [math.cos(x) for x in phi_s]
    ss = [math.sin(x) for x in phi_s]
    return cs, ss, cs[::-1], [-x for x in ss[::-1]]


def rotation_tables(p: WeatherParams) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """``(cos φ_S, sin φ_S, cos φ_N, sin φ_N)``, each (1, N, 1) float32, built inside the traced function."""
    cs, ss, cn, sn = (
        mx.array(x, dtype=_F32).reshape(1, p.n, 1) for x in _rot_lists(p.n, p.dx, p.dt, p.f0, p.L)
    )
    return cs, ss, cn, sn


# --- S1 advection -------------------------------------------------------------------------------------------


def _cr_weights(a: mx.array) -> list[mx.array]:
    a2, a3 = a * a, a * a * a
    return [
        0.5 * (-a3 + 2.0 * a2 - a),
        0.5 * (3.0 * a3 - 5.0 * a2 + 2.0),
        0.5 * (-3.0 * a3 + 4.0 * a2 + a),
        0.5 * (a3 - a2),
    ]


def semi_lagrangian(a: mx.array, ox: mx.array, oy: mx.array, order: int = 1) -> mx.array:
    """``a`` (K, N, N) sampled at ``(j + oy, i + ox)`` (offsets in cells), periodic, one gather.

    ``order`` 1 is bilinear; 3 is Catmull–Rom over the 4 × 4 points, clamped to the range of the four bilinear
    corners. Offsets, not absolute coordinates, so an x-roll of ``a`` and the offsets commutes bitwise. The
    int32 ``%`` maps −1 to N−1.
    """
    k, n = a.shape[0], a.shape[-1]
    fx, fy = mx.floor(ox), mx.floor(oy)
    ax, ay = ox - fx, oy - fy
    jj = mx.arange(n, dtype=_I32).reshape(1, n, 1)
    ii = mx.arange(n, dtype=_I32).reshape(1, 1, n)
    j0 = (jj + fy.astype(_I32)) % n
    i0 = (ii + fx.astype(_I32)) % n
    koff = (mx.arange(k, dtype=_I32) * (n * n)).reshape(k, 1, 1)
    flat = a.reshape(-1)
    if order == 1:
        r0, r1 = koff + j0 * n, koff + ((j0 + 1) % n) * n
        c0, c1 = i0, (i0 + 1) % n
        g = mx.take(flat, mx.stack([r0 + c0, r0 + c1, r1 + c0, r1 + c1]))
        return (1.0 - ay) * ((1.0 - ax) * g[0] + ax * g[1]) + ay * ((1.0 - ax) * g[2] + ax * g[3])
    rows = [koff + ((j0 + (m - 1)) % n) * n for m in range(4)]
    cols = [(i0 + (m - 1)) % n for m in range(4)]
    g = mx.take(flat, mx.stack([rows[my] + cols[mxi] for my in range(4) for mxi in range(4)]))
    wx, wy = _cr_weights(ax), _cr_weights(ay)
    out = None
    for my in range(4):
        row = wx[0] * g[4 * my] + wx[1] * g[4 * my + 1] + wx[2] * g[4 * my + 2] + wx[3] * g[4 * my + 3]
        out = wy[my] * row if out is None else out + wy[my] * row
    corners = (g[5], g[6], g[9], g[10])
    lo = mx.minimum(mx.minimum(corners[0], corners[1]), mx.minimum(corners[2], corners[3]))
    hi = mx.maximum(mx.maximum(corners[0], corners[1]), mx.maximum(corners[2], corners[3]))
    return mx.minimum(mx.maximum(out, lo), hi)


def _advect(u: mx.array, v: mx.array, p: WeatherParams) -> tuple[mx.array, mx.array]:
    """S1: one Euler backtrace with the winds at n; ``u`` (2, N, N) at u-points, ``v`` at v-points."""
    ev = _east(v)
    sv = _south(v)
    sev = _south(ev)
    wu = _west(u)
    nu = _north(u)
    nwu = _north(wu)
    v_at_u = 0.25 * (v + ev + sv + sev)
    u_at_v = 0.25 * (u + wu + nu + nwu)
    c = -p.dt / p.dx
    ox = mx.concatenate([c * u, c * u_at_v])
    oy = mx.concatenate([c * v_at_u, c * v])
    out = semi_lagrangian(mx.concatenate([u, v]), ox, oy, p.sl_order)
    return out[:2], out[2:]


# --- S2 continuity -----------------------------------------------------------------------------------------


def _continuity(
    h: mx.array, u: mx.array, v: mx.array, big_q: mx.array, p: WeatherParams
) -> tuple[mx.array, ...]:
    """S2: flux-form continuity of both layers and of ``Q`` with the positivity limiter.

    Returns ``(h', Q', n_limited, n_clamp, clamp_mass, q_clamp)``.
    """
    dtdx = p.dt / p.dx
    eh, nh = _east(h), _north(h)
    fx = 0.5 * (h + eh) * u
    fy = 0.5 * (h + nh) * v
    wfx, sfy = _west(fx), _south(fy)
    out = mx.maximum(fx, 0.0) + mx.maximum(-wfx, 0.0) + mx.maximum(fy, 0.0) + mx.maximum(-sfy, 0.0)
    avail = mx.maximum(h - p.h_min, 0.0) * p.dx
    need = p.dt * out
    over = need > avail
    lam = mx.where(over, avail / (p.dt * mx.maximum(out, 1e-30)), 1.0)
    elam, nlam = _east(lam), _north(lam)
    fx = fx * mx.where(fx > 0, lam, elam)
    fy = fy * mx.where(fy > 0, lam, nlam)
    hn = h - dtdx * (fx - _west(fx) + fy - _south(fy))
    q = big_q / h[0]
    eq, nq = _east(q), _north(q)
    gx = fx[0] * mx.where(fx[0] > 0, q, eq)
    gy = fy[0] * mx.where(fy[0] > 0, q, nq)
    qn = big_q - dtdx * (gx - _west(gx) + gy - _south(gy))
    n_limited = over.astype(_I32).sum()
    n_clamp = (hn < p.h_min).astype(_I32).sum() + (qn < 0.0).astype(_I32).sum()
    clamp_mass = mx.maximum(p.h_min - hn, 0.0).sum()
    q_clamp = mx.maximum(-qn, 0.0).sum()
    return mx.maximum(hn, p.h_min), mx.maximum(qn, 0.0), n_limited, n_clamp, clamp_mass, q_clamp


# --- S3 momentum --------------------------------------------------------------------------------------------


def _pair_rot(
    u: mx.array, v: mx.array, dj: int, di: int, c: mx.array, s: mx.array
) -> tuple[mx.array, mx.array]:
    """Rotate every pair ``(u[j,i], v[j+dj, i+di])`` by the per-row angle (cos ``c``, sin ``s``)."""
    vx = mx.roll(v, (-dj, -di), axis=(-2, -1)) if (dj or di) else v
    un = c * u + s * vx
    vxn = -s * u + c * vx
    vn = mx.roll(vxn, (dj, di), axis=(-2, -1)) if (dj or di) else vxn
    return un, vn


def rotate_half(
    u: mx.array, v: mx.array, tab: tuple[mx.array, mx.array, mx.array, mx.array]
) -> tuple[mx.array, mx.array]:
    """Half a step of Coriolis rotation, ``R(½) = ½[B_SE∘B_NW + B_NE∘B_SW]`` (spec §2.3 S3).

    Each ``B_X`` rotates the disjoint (u, nearest-v) pairs of one diagonal exactly, so the rotation never
    increases kinetic energy, for any ``f(y)``. ``tab`` is ``rotation_tables``; ``u``, ``v`` are (..., N, N).
    """
    cs, ss, cn, sn = tab
    au, av = _pair_rot(u, v, 0, 0, cn, sn)
    au, av = _pair_rot(au, av, -1, 1, cs, ss)
    bu, bv = _pair_rot(u, v, -1, 0, cs, ss)
    bu, bv = _pair_rot(bu, bv, 0, 1, cn, sn)
    return 0.5 * (au + bu), 0.5 * (av + bv)


def _pressure_kick(u: mx.array, v: mx.array, h: mx.array, p: WeatherParams) -> tuple[mx.array, mx.array]:
    """The backward half of forward–backward: pressure gradients from the new thicknesses."""
    m2 = p.g_top * (h[0] + h[1])
    m = mx.stack([m2 + p.g_prime * h[0], m2])
    c = p.dt / p.dx
    return u - c * (_east(m) - m), v - c * (_north(m) - m)


# --- S5 / S6 mass exchange between the layers ---------------------------------------------------------------


def exchange_condense(
    h: mx.array, u: mx.array, v: mx.array, mu: mx.array
) -> tuple[mx.array, mx.array, mx.array]:
    """Move the mass ``mu`` (>= 0, centres) from layer 1 to layer 2 with its momentum (spec S5).

    ``u2 += μ^u (u1 − u2)/h2'^u`` with the post-transfer ``h2'``: the face momentum ``h1 u1 + h2 u2`` is
    unchanged, the receiving wind stays between the two old winds and layer 1's wind is untouched.
    """
    h1n, h2n = h[0] - mu, h[1] + mu
    m = mx.stack([mu, h2n])
    on_u, on_v = 0.5 * (m + _east(m)), 0.5 * (m + _north(m))  # [0] mu, [1] h2' on the u- and v-faces
    mu_u, h2_u, mu_v, h2_v = on_u[0], on_u[1], on_v[0], on_v[1]
    u2 = u[1] + mu_u * (u[0] - u[1]) / h2_u
    v2 = v[1] + mu_v * (v[0] - v[1]) / h2_v
    return mx.stack([h1n, h2n]), mx.stack([u[0], u2]), mx.stack([v[0], v2])


def exchange_relax(h: mx.array, u: mx.array, v: mx.array, r: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """Transfer the signed mass ``r`` (centres) from layer 2 to layer 1, each receiving layer taking the
    donor's momentum (spec S6). The returned air carries no vapour (``Q`` is not an argument)."""
    hn = mx.stack([h[0] + r, h[1] - r])
    rr = mx.stack([mx.maximum(r, 0.0), mx.maximum(-r, 0.0)])  # what layer 1 gains, what layer 2 gains
    err, nrr = _east(rr), _north(rr)
    ehn, nhn = _east(hn), _north(hn)
    ru, rv = 0.5 * (rr + err), 0.5 * (rr + nrr)
    hu, hv = 0.5 * (hn + ehn), 0.5 * (hn + nhn)
    un = u + ru * (u[::-1] - u) / hu
    vn = v + rv * (v[::-1] - v) / hv
    return hn, un, vn


# --- S7 damping ---------------------------------------------------------------------------------------------


def _damp(
    u: mx.array, v: mx.array, h: mx.array, p: WeatherParams
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """S7: friction, optional ν-Laplacian, divergence damping, the safety tripwire.

    Returns ``(u, v, n_safety, courant_max, speed_max, hyper_max)``. The speeds and the hyperbolicity ratio
    use the winds entering S7 (post-S6, before friction, damping and the seeds of the same step), whose
    shifts the friction needs anyway; spec S9 lists ``hyper_max`` with the end-of-step diagnostics, and the
    two differ by one step's damping and seed (0.0394 against 0.0374 at the extreme of the 60 000-step start
    set run). Computing it from the final winds would build a second set of centre-average shifts of the
    step's output arrays, which MLX 0.32.3 ``mx.compile`` does not allow (module docstring).
    """
    dt, dx = p.dt, p.dx
    ev = _east(v)
    sv = _south(v)
    sev = _south(ev)
    wu = _west(u)
    nu = _north(u)
    nwu = _north(wu)
    v_at_u = 0.25 * (v + ev + sv + sev)
    u_at_v = 0.25 * (u + wu + nu + nwu)
    su = mx.sqrt(u * u + v_at_u * v_at_u)
    sp = mx.sqrt(v * v + u_at_v * u_at_v)
    cour = mx.maximum(mx.abs(u).max(), mx.abs(v).max()) * (dt / dx)
    speed = mx.maximum(su.max(), sp.max())
    n_safety = (su > p.u_safe).astype(_I32).sum() + (sp > p.u_safe).astype(_I32).sum()
    # long-wave hyperbolicity ratio at the centres, vector form
    du = 0.5 * (u + wu)
    dv = 0.5 * (v + sv)
    d_u, d_v = du[0] - du[1], dv[0] - dv[1]
    hyper = ((d_u * d_u + d_v * d_v) / (p.g_prime * (h[0] + h[1]))).max()
    # friction (implicit): layer 1 linear + quadratic, layer 2 linear
    lin = mx.array([p.r1, p.r2], dtype=_F32).reshape(2, 1, 1)
    quad = mx.array([p.c_d, 0.0], dtype=_F32).reshape(2, 1, 1)
    fu = u / (1.0 + dt * (lin + quad * su))
    fv = v / (1.0 + dt * (lin + quad * sp))
    if p.nu > 0:  # static: the Laplacian of all four components
        k = p.nu * dt / (dx * dx)
        e1, w1, n1, s1 = _east(fu), _west(fu), _north(fu), _south(fu)
        e2, w2, n2, s2 = _east(fv), _west(fv), _north(fv), _south(fv)
        fu = fu + k * (e1 + w1 + n1 + s1 - 4.0 * fu)
        fv = fv + k * (e2 + w2 + n2 + s2 - 4.0 * fv)
    if p.kappa_d > 0:  # static: divergence damping, leaves the vorticity alone
        dd = fu - _west(fu) + fv - _south(fv)
        fu = fu + p.kappa_d * (_east(dd) - dd)
        fv = fv + p.kappa_d * (_north(dd) - dd)
    # tripwire: scale winds above u_safe back to it (a branch-free min(1, u_safe/s))
    fu = fu * (p.u_safe / mx.maximum(su, p.u_safe))
    fv = fv * (p.u_safe / mx.maximum(sp, p.u_safe))
    return fu, fv, n_safety, cour, speed, hyper


# --- S8 disturbances ----------------------------------------------------------------------------------------


def _seed_fields(key: mx.array, p: WeatherParams) -> tuple[mx.array, ...]:
    """``(ψ at corners, on, a, x_s, y_s)`` of this step's disturbance (spec S8); ``ψ = 0`` when not ``on``."""
    k_on, k_x, k_y, k_a = mx.random.split(key, 4)
    length = p.L
    on = (mx.random.uniform(key=k_on) < p.seed_rate).astype(_F32)
    xs = length * mx.random.uniform(key=k_x)
    ys = length * mx.random.uniform(key=k_y)
    a = 1.0 + p.seed_jitter * (2.0 * mx.random.uniform(key=k_a) - 1.0)
    sgn = mx.where(mx.sin(2.0 * math.pi * ys / length) >= 0, 1.0, -1.0)  # cyclonic for the hemisphere
    corner = (mx.arange(p.n, dtype=_F32) + 1.0) * p.dx
    ddx = ops.min_image(corner - xs, length)[None, :]
    ddy = ops.min_image(corner - ys, length)[:, None]
    amp = -sgn * on * a * p.seed_v * p.seed_sigma * math.sqrt(math.e)
    psi = amp * mx.exp(-(ddx * ddx + ddy * ddy) / (2.0 * p.seed_sigma**2))
    return psi, on, a, xs, ys


def psi_winds(psi: mx.array, dx: float) -> tuple[mx.array, mx.array]:
    """The discretely non-divergent winds of a corner streamfunction: ``u = −∂ψ/∂y``, ``v = ∂ψ/∂x``."""
    return -(psi - _south(psi)) / dx, (psi - _west(psi)) / dx


def _seed(
    u: mx.array, v: mx.array, big_q: mx.array, h1: mx.array, key: mx.array, p: WeatherParams
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """S8: add a disturbance with probability ``seed_rate``. Returns ``(u, v, Q, qseed, on)``."""
    psi, on, a, xs, ys = _seed_fields(key, p)
    du, dv = psi_winds(psi, p.dx)
    w2 = p.seed_w2
    gain = mx.array([1.0, w2], dtype=_F32).reshape(2, 1, 1)
    u = u + gain * du
    v = v + gain * dv
    qseed = big_q * 0.0
    if p.seed_q > 0:  # static: a moisture seed, capped by the moist-thickness limiter
        centre = (mx.arange(p.n, dtype=_F32) + 0.5) * p.dx
        cx = ops.min_image(centre - xs, p.L)[None, :]
        cy = ops.min_image(centre - ys, p.L)[:, None]
        qseed = on * a * p.seed_q * p.Qs * mx.exp(-(cx * cx + cy * cy) / (2.0 * p.seed_qsigma**2))
        if p.gamma > 0:
            qseed = mx.minimum(qseed, mx.maximum(h1 - p.gamma * big_q - p.m_min, 0.0) / p.gamma)
        big_q = big_q + qseed
    return u, v, big_q, qseed, on


# --- the step -----------------------------------------------------------------------------------------------


def step(
    ws: WeatherState, fo: WeatherForcing, key: mx.array, p: WeatherParams, *, debug: bool = False
) -> tuple[WeatherState, WeatherDiag]:
    """One weather step (spec §2.3 S1–S9): advection, continuity, rotation and pressure (with the optional
    momentum forcing between the kick and the second half-rotation), evaporation, condensation, radiative
    relaxation, damping, disturbances. ``key`` is a (2,) MLX key used only by S8; the step returns no key.
    """
    dt = p.dt
    zero = ws.h1[0, 0] * 0.0  # a zero derived from the inputs (never return a constant array from compile)
    zero_i = zero.astype(_I32)
    h = mx.stack([ws.h1, ws.h2])
    u = mx.stack([ws.u1, ws.u2])
    v = mx.stack([ws.v1, ws.v2])

    # S1, S2 (both use the winds at n)
    ua, va = _advect(u, v, p)
    h, big_q, n_limited, n_clamp, clamp_mass, q_clamp = _continuity(h, u, v, ws.Q, p)

    # S3: half-rotation, pressure kick, forcing, half-rotation
    tab = rotation_tables(p)
    ua, va = rotate_half(ua, va, tab)
    ua, va = _pressure_kick(ua, va, h, p)
    if fo.fu1 is not None:  # static: presence of the optional forcing (a pytree structure, not a value)
        assert fo.fv1 is not None and fo.fu2 is not None and fo.fv2 is not None
        ua = ua + dt * mx.stack([fo.fu1, fo.fu2])
        va = va + dt * mx.stack([fo.fv1, fo.fv2])
    ua, va = rotate_half(ua, va, tab)

    # S4: evaporation over the warm pools, with the moist-thickness limiter
    u1c = 0.5 * (ua[0] + _west(ua[0]))
    v1c = 0.5 * (va[0] + _south(va[0]))
    qe = p.Qs * (p.q_bg + (p.q_pool - p.q_bg) * fo.T_s)
    evap = p.k_E * (mx.sqrt(u1c * u1c + v1c * v1c) + p.u_gust) * mx.maximum(qe - big_q, 0.0)
    n_evap = zero_i
    if p.gamma > 0:
        cap = mx.maximum(h[0] - p.gamma * big_q - p.m_min, 0.0) / (p.gamma * dt)
        n_evap = (evap > cap).astype(_I32).sum()
        evap = mx.minimum(evap, cap)
    big_q = big_q + dt * evap

    # S5: condensation and mass transfer to layer 2
    cond = mx.maximum(big_q - p.Qs * fo.qs, 0.0) / p.tau_c
    n_cond = zero_i
    if p.gamma > 0:
        bound = (h[0] - p.h_min) / (p.gamma * dt)
        n_cond = (cond > bound).astype(_I32).sum()
        cond = mx.minimum(cond, bound)
    big_q = big_q - dt * cond
    if p.gamma > 0:
        h, ua, va = exchange_condense(h, ua, va, p.gamma * dt * cond)

    # S6: radiative mass return
    n_relax = zero_i
    if p.tau_rad > 0:
        r = dt * (p.H1 - h[0]) / p.tau_rad
        lo, hi = -(h[0] - p.h_min), h[1] - p.h_min
        n_relax = ((r < lo) | (r > hi)).astype(_I32).sum()
        h, ua, va = exchange_relax(h, ua, va, mx.minimum(mx.maximum(r, lo), hi))

    # S7: damping and the safety tripwire
    ua, va, n_safety, courant, speed, hyper = _damp(ua, va, h, p)

    # S8: disturbances
    seed_on, qseed = zero, big_q * 0.0
    if p.seed_rate > 0:  # static
        ua, va, big_q, qseed, seed_on = _seed(ua, va, big_q, h[0], key, p)

    # S9: stored diagnostics
    zi = p.height * h[0] / (h[0] + h[1])
    cloud = (p.height - zi) * cond / (cond + p.c_half)
    new = WeatherState(h[0], h[1], ua[0], va[0], ua[1], va[1], big_q, cond, cloud)
    eps_local = (p.gamma * big_q / h[0]).max() if p.gamma > 0 else zero
    diag = WeatherDiag(
        n_limited=n_limited,
        n_clamp=n_clamp,
        clamp_mass=clamp_mass,
        q_clamp=q_clamp,
        n_cond_limited=n_cond,
        n_relax_limited=n_relax,
        n_evap_limited=n_evap,
        n_safety=n_safety,
        courant_max=courant,
        speed_max=speed,
        hyper_max=hyper,
        eps_local_max=eps_local,
        seed_on=seed_on,
        sum_E=evap.sum() * dt,
        sum_C=cond.sum() * dt,
        sum_Qseed=qseed.sum(),
        E=evap if debug else None,
        C=cond if debug else None,
        Qseed=qseed if debug else None,
    )
    return new, diag


def make_step(p: WeatherParams, *, compile: bool = True, debug: bool = False):
    """``fn(ws, fo, key) -> (ws', diag)`` closed over ``p``, compiled once (like ``step.make_step``)."""

    def fn(ws: WeatherState, fo: WeatherForcing, key: mx.array) -> tuple[WeatherState, WeatherDiag]:
        return step(ws, fo, key, p, debug=debug)

    return mx.compile(fn) if compile else fn


# --- diagnostics of a state ---------------------------------------------------------------------------------


def vorticity(ws: WeatherState, p: WeatherParams) -> mx.array:
    """Layer-1 relative vorticity at the corners ``((i+1)dx, (j+1)dx)``, (N, N)."""
    return (_east(ws.v1) - ws.v1) / p.dx - (_north(ws.u1) - ws.u1) / p.dx


def divergence(ws: WeatherState, p: WeatherParams) -> mx.array:
    """Horizontal divergence of both layers at the centres, (2, N, N)."""
    u = mx.stack([ws.u1, ws.u2])
    v = mx.stack([ws.v1, ws.v2])
    return (u - _west(u) + v - _south(v)) / p.dx


def interface_height(ws: WeatherState, p: WeatherParams) -> mx.array:
    """``z_i = H·h1/(h1 + h2)`` at the centres (the layer interface seen from the column), (N, N)."""
    return p.height * ws.h1 / (ws.h1 + ws.h2)


# --- the wind the organisms feel (spec §4) ------------------------------------------------------------------


def alpha(buoy_cap: mx.array, p: WeatherParams) -> mx.array:
    """The coupling of an organism to the wind, ``α0 + α1·buoy_cap``."""
    return p.alpha0 + p.alpha1 * buoy_cap


class WindFields(NamedTuple):
    """The centre fields of the wind formula, each (N, N): the interface height, the centre-averaged winds of
    both layers, the layer divergences and the interface (Leibniz) term ``J``; and ``zbar`` (4, N), the zonal
    means of the staggered winds that the eddy wind subtracts (a default, so a positional unpack of the old
    eight fields fails loudly)."""

    zi: mx.array
    u1: mx.array
    v1: mx.array
    u2: mx.array
    v2: mx.array
    d1: mx.array
    d2: mx.array
    jump: mx.array
    zbar: mx.array | None = None  # (4, n): x-means of the staggered u1, v1, u2, v2 per row (eddy wind, §32)


def _zbar(ws: WeatherState) -> mx.array:
    """The zonal (x) means of the **staggered** winds ``u1, v1, u2, v2``, (4, n): what ``sample_wind`` reads
    when it interpolates along y, so a zonally uniform wind has exactly zero eddy. No roll."""
    return mx.stack([ws.u1.mean(axis=-1), ws.v1.mean(axis=-1), ws.u2.mean(axis=-1), ws.v2.mean(axis=-1)])


def wind_fields(ws: WeatherState, p: WeatherParams) -> WindFields:
    """Build the centre fields of the wind formula from a weather state.

    ``J = (φ(z_i)·ū1 − ū2)·∇z_i`` is the interface (Leibniz) term, with centred differences of ``z_i``. W1
    builds this once per engine step and hands it to ``sample_wind``/``wind_layers`` (``fields=``); it rolls
    the winds, so never build it twice from one state inside one compiled graph.
    """
    u = mx.stack([ws.u1, ws.u2])
    v = mx.stack([ws.v1, ws.v2])
    wu = _west(u)
    sv = _south(v)
    div = (u - wu + v - sv) / p.dx
    uc, vc = 0.5 * (u + wu), 0.5 * (v + sv)
    zi = interface_height(ws, p)
    zx = (_east(zi) - _west(zi)) / (2.0 * p.dx)
    zy = (_north(zi) - _south(zi)) / (2.0 * p.dx)
    phi = mx.minimum(zi / p.z_sl, 1.0)
    jump = (phi * uc[0] - uc[1]) * zx + (phi * vc[0] - vc[1]) * zy
    return WindFields(zi, uc[0], vc[0], uc[1], vc[1], div[0], div[1], jump, _zbar(ws))


def _wind_formula(
    z: mx.array,
    zi: mx.array,
    winds: tuple[mx.array, mx.array, mx.array, mx.array],
    d1: mx.array,
    d2: mx.array,
    jump: mx.array,
    p: WeatherParams,
) -> tuple[mx.array, mx.array, mx.array]:
    """``(u_h, v_h, w)`` at height ``z`` (spec §4 steps 2–4); every argument broadcasts against the others.

    ``u_h = φ(z)·(u1, v1)`` below the interface and ``(u2, v2)`` above; ``w(z) = −[D1·Φ(min(z, z_i)) +
    1[z > z_i]·(D2·(z − z_i) + J) − D̄·z]`` with ``D̄ = [D1·Φ(z_i) + D2·(H − z_i) + J]/H``, so
    ``w(0) = w(H) = 0`` and ``w`` jumps by ``−J`` at the interface.
    """
    u1, v1, u2, v2 = winds
    zs, big_h = p.z_sl, p.height
    phi = mx.minimum(z / zs, 1.0)
    below = z < zi

    def cap_phi(x: mx.array) -> mx.array:
        return mx.where(x <= zs, x * x / (2.0 * zs), x - 0.5 * zs)

    uh = mx.where(below, phi * u1, u2)
    vh = mx.where(below, phi * v1, v2)
    dbar = (d1 * cap_phi(zi) + d2 * (big_h - zi) + jump) / big_h
    above = mx.where(z > zi, d2 * (z - zi) + jump, 0.0)
    w = -(d1 * cap_phi(mx.minimum(z, zi)) + above - dbar * z)
    return uh, vh, w


def _clip_wind(
    uh: mx.array,
    vh: mx.array,
    w: mx.array,
    edge: mx.array,
    p: WeatherParams,
    valid: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Clip ``||u_h|| <= u_max`` (direction kept) and ``|w| <= w_edge`` where ``edge``; count both. With
    ``valid`` the two counts include only the rows where it is true (the clip itself applies to every row)."""
    sp = mx.sqrt(uh * uh + vh * vh)
    over = sp > p.u_max
    scale = p.u_max / mx.maximum(sp, p.u_max)
    w_over = edge & (mx.abs(w) > p.w_edge)
    if valid is not None:  # static: dead rows hold stale positions, so they are not counted
        over, w_over = over & valid, w_over & valid
    wc = mx.where(edge, mx.clip(w, -p.w_edge, p.w_edge), w)
    return uh * scale, vh * scale, wc, over.astype(_I32).sum(), w_over.astype(_I32).sum()


def _bilinear_rows(table: mx.array, gx: mx.array, gy: mx.array, n: int) -> mx.array:
    """Bilinear periodic sampling of ``table`` (N·N, K), rows ``j·N + i``, at index coordinates (gx, gy)."""
    fx, fy = mx.floor(gx), mx.floor(gy)
    ax, ay = (gx - fx)[:, None], (gy - fy)[:, None]
    i0, j0 = fx.astype(_I32) % n, fy.astype(_I32) % n
    i1, j1 = (i0 + 1) % n, (j0 + 1) % n
    g = mx.take(table, mx.stack([j0 * n + i0, j0 * n + i1, j1 * n + i0, j1 * n + i1]), axis=0)
    return (1.0 - ay) * ((1.0 - ax) * g[0] + ax * g[1]) + ay * ((1.0 - ax) * g[2] + ax * g[3])


def _interp_rows(t: mx.array, g: mx.array, n: int) -> mx.array:
    """Periodic linear interpolation of the rows ``t`` (K, n) at index coordinates ``g`` (M,): (K, M)."""
    f = mx.floor(g)
    a = g - f
    i0 = f.astype(_I32) % n
    i1 = (i0 + 1) % n
    return (1.0 - a) * mx.take(t, i0, axis=1) + a * mx.take(t, i1, axis=1)


def sample_wind(
    ws: WeatherState,
    pos: mx.array,
    p: WeatherParams,
    *,
    field_cell: float = 4.0,
    fields: WindFields | None = None,
    valid: mx.array | None = None,
    eddy: bool = False,
) -> AgentWind:
    """The wind at agent positions ``pos`` (M, 3): pure gathers, any M (spec §4).

    Horizontal and vertical wind from the layer fields, continuous in z (the formula at the agent's own z; the
    20 field layers are used only for the edge clip and for reports): bilinear periodic sampling of the centre
    fields (``h1, h2, D1, D2, J``), of ``u`` on the u-grid and of ``v`` on the v-grid, then ``_wind_formula``.
    ``||u_h||`` is clipped to ``u_max`` and ``|w|`` to ``w_edge`` in the floor and surface field layers
    (``z < field_cell`` or ``z >= height − field_cell``), both counted. Mid-column ``w`` is not clipped.
    ``fields`` are the precomputed ``wind_fields(ws, p)`` (built once per weather step by the caller). With
    ``valid`` ((M,) bool) the two clip counts exclude the rows where it is false (the engine passes
    ``alive``: dead rows rest at stale positions); ``None`` counts every row.

    ``eddy=True`` (W2, DESIGN §32) also returns ``u_eddy`` (M, 2): the clipped ``u_h`` minus ``ū_h``, the
    zonal mean (``WindFields.zbar``) of the same layer at the agent's latitude, linear in y on the rows the
    winds live on (u at ``(j + ½)·dx``, v at ``(j + 1)·dx``), selected by ``z < z_i`` and scaled by ``φ(z)``
    below the interface exactly as ``_wind_formula`` does for the wind itself.
    """
    n = p.n
    wf = wind_fields(ws, p) if fields is None else fields
    cen = mx.stack([ws.h1, ws.h2, wf.d1, wf.d2, wf.jump], axis=-1).reshape(n * n, 5)
    ugrid = mx.stack([ws.u1, ws.u2], axis=-1).reshape(n * n, 2)
    vgrid = mx.stack([ws.v1, ws.v2], axis=-1).reshape(n * n, 2)
    gx, gy = pos[:, 0] / p.dx, pos[:, 1] / p.dx
    z = pos[:, 2]
    c = _bilinear_rows(cen, gx - 0.5, gy - 0.5, n)
    uu = _bilinear_rows(ugrid, gx - 1.0, gy - 0.5, n)
    vv = _bilinear_rows(vgrid, gx - 0.5, gy - 1.0, n)
    zi = p.height * c[:, 0] / (c[:, 0] + c[:, 1])
    uh, vh, w = _wind_formula(z, zi, (uu[:, 0], vv[:, 0], uu[:, 1], vv[:, 1]), c[:, 2], c[:, 3], c[:, 4], p)
    fz = round(p.height / field_cell)
    edge = (z < field_cell) | (z >= (fz - 1) * field_cell)
    uh, vh, w, n_h, n_w = _clip_wind(uh, vh, w, edge, p, valid)
    if not eddy:  # static
        return AgentWind(mx.stack([uh, vh, w], axis=-1), n_h, n_w)
    zb = _zbar(ws) if wf.zbar is None else wf.zbar
    ub = _interp_rows(mx.stack([zb[0], zb[2]]), gy - 0.5, n)
    vb = _interp_rows(mx.stack([zb[1], zb[3]]), gy - 1.0, n)
    phi = mx.minimum(z / p.z_sl, 1.0)
    below = z < zi
    ubh = mx.where(below, phi * ub[0], ub[1])
    vbh = mx.where(below, phi * vb[0], vb[1])
    return AgentWind(mx.stack([uh, vh, w], axis=-1), n_h, n_w, mx.stack([uh - ubh, vh - vbh], axis=-1))


def sample_wind_h(
    ws: WeatherState, pos: mx.array, p: WeatherParams, *, valid: mx.array | None = None, eddy: bool = False
) -> AgentWind:
    """The 2D wind at agent positions ``pos`` (M, 2) (W1, DESIGN §30): the **layer-1** wind ``(u1, v1)``
    bilinear on its own C-grid (u at ``(i + 1, j + ½)·dx``, v at ``(i + ½, j + 1)·dx``), no surface-layer
    factor and no vertical wind, clipped to ``||u_h|| <= u_max``. ``u`` is (M, 2); ``n_clip_w`` is 0.
    ``valid`` as in ``sample_wind``. ``eddy=True`` adds ``u_eddy``: the clipped wind minus the zonal mean of
    layer 1 at the agent's latitude (``sample_wind``'s ``eddy``)."""
    n = p.n
    ugrid = ws.u1.reshape(n * n, 1)
    vgrid = ws.v1.reshape(n * n, 1)
    gx, gy = pos[:, 0] / p.dx, pos[:, 1] / p.dx
    uu = _bilinear_rows(ugrid, gx - 1.0, gy - 0.5, n)[:, 0]
    vv = _bilinear_rows(vgrid, gx - 0.5, gy - 1.0, n)[:, 0]
    sp = mx.sqrt(uu * uu + vv * vv)
    over = sp > p.u_max
    if valid is not None:  # static
        over = over & valid
    scale = p.u_max / mx.maximum(sp, p.u_max)
    n_h = over.astype(_I32).sum()
    u = mx.stack([uu * scale, vv * scale], axis=-1)
    if not eddy:  # static
        return AgentWind(u, n_h, n_h * 0)
    ub = _interp_rows(ws.u1.mean(axis=-1)[None], gy - 0.5, n)[0]
    vb = _interp_rows(ws.v1.mean(axis=-1)[None], gy - 1.0, n)[0]
    return AgentWind(u, n_h, n_h * 0, u - mx.stack([ub, vb], axis=-1))


def wind_layers(
    ws: WeatherState,
    p: WeatherParams,
    fz: int = 20,
    field_cell: float = 4.0,
    fields: WindFields | None = None,
) -> mx.array:
    """The same wind formula at the layer centres ``z_k = (k + ½)·field_cell`` on the weather centres (u and v
    averaged to the centres, no interpolation), (fz, 3, N, N); the clips apply, ``w_edge`` in layers 0 and
    fz−1."""
    wf = wind_fields(ws, p) if fields is None else fields
    kk = mx.arange(fz, dtype=_F32).reshape(fz, 1, 1)
    z = (kk + 0.5) * field_cell
    winds = (wf.u1[None], wf.v1[None], wf.u2[None], wf.v2[None])
    uh, vh, w = _wind_formula(z, wf.zi[None], winds, wf.d1[None], wf.d2[None], wf.jump[None], p)
    edge = (kk == 0) | (kk == fz - 1)
    uh, vh, w, _, _ = _clip_wind(uh, vh, w, edge, p)
    return mx.stack([uh, vh, w], axis=1)


# --- engine coupling (W1, DESIGN §30) -----------------------------------------------------------------------
# The config, the packed state field and the validation. The step wiring (wind, rain, hazard) follows in the
# later build steps; everything here is host-side Python on config values, or builds the rest state.

N_WX = 9  # the arrays of ``WeatherState``, stacked in its field order into ``State.wx``

# The ``WeatherParams`` fields that are not ``[weather]`` keys. W0 tuned none of them after round 0, the
# mean-flow stability rule is tabulated for these numerics only, and every config key is a hashed degree of
# freedom. A later change makes one a key (in ``POST_V01_KEYS``) with this value as its default.
ENGINE_FIXED: Final[dict[str, Any]] = {
    "dt": 1.0,
    "h_min": 2.0,
    "m_min": 1.0,
    "c_half": 1e-3,
    "Qs": 1.0,
    "nu": 0.0,
    "kappa_d": 0.04,
    "sl_order": 1,
    "u_safe": 0.3,
    "seed_q": 0.0,
    "seed_qsigma": 6.0,
    "seed_w2": 0.0,
    "detect_every": 25,
    "zeta_th": 0.00625,
    "p_th": 1e-4,
    "a_min": 25,
    "r_min": 4,
    "miss_grace": 2,
    "min_dets": 3,
    "seed_dmin": 8.0,  # W2: the floor of the cloud depth of the seeding band, 2·field_cell (DESIGN §32)
}

# A column's rain stays below this: the int32 ranges of ``floor_i32``, the increment and ``exact_sum`` hold.
_INT_BOUND: Final = 2.0**30


def pack(ws: WeatherState) -> mx.array:
    """``State.wx``: the nine arrays of ``ws`` stacked in field order, (N_WX, n, n) float32."""
    return mx.stack(list(ws))


def unpack(wx: mx.array) -> WeatherState:
    """The inverse of ``pack`` (nine slices)."""
    return WeatherState(*(wx[i] for i in range(N_WX)))


@functools.cache
def _params(wc: Any, size: float, height: float, dims: int) -> WeatherParams:
    n = wc.cells
    pools = tuple(tuple(wc.pools[i : i + 7]) for i in range(0, len(wc.pools), 7))
    own = {f.name for f in fields(WeatherParams)}
    kw = {f.name: getattr(wc, f.name) for f in fields(wc) if f.name in own and f.name != "pools"}
    return WeatherParams(
        n=n,
        dx=size / n,
        height=height if dims == 3 else wc.H1 + wc.H2,
        pools=pools,
        **ENGINE_FIXED,
        **kw,
    )


def params_from_config(cfg: Config) -> WeatherParams:
    """The ``WeatherParams`` of a config: a name-keyed copy of ``cfg.weather`` plus ``n = cells``, ``dx =
    world.size / cells``, ``height`` (``world.height`` in 3D, ``H1 + H2`` in 2D) and ``ENGINE_FIXED``.
    Cached on the weather section and the three world values it reads. ``validate_config`` guards the inputs
    (``cells`` and ``pools``) before this is built."""
    return _params(cfg.weather, cfg.world.size, cfg.world.height, cfg.world.dims)


@functools.cache
def _rest_off() -> mx.array:
    p = WeatherParams()
    return pack(init_state(p, default_forcing(p, mx.array(0, dtype=_I32))))


def rest_array(cfg: Config, step: mx.array | int = 0) -> mx.array:
    """The rest state of a config as a packed ``wx``: ``h = (H1, H2)``, ``u = v = 0``, ``Q = min(Q_e(T_s(t)),
    Q_s)`` under the pools of step ``step``, ``P = Dc = 0``. ``step`` is a scalar array or a Python int (it
    reaches ``warm_pools`` as an int32 array). A weather-off config gets the step-0 array of the default
    parameters whatever ``step`` is, so a filled array is bitwise the one a native weather-off state carries.
    """
    if not cfg.weather.enabled:
        return _rest_off()
    p = params_from_config(cfg)
    t = (step if isinstance(step, mx.array) else mx.array(step)).astype(_I32)
    return pack(init_state(p, default_forcing(p, t)))


def validate_config(cfg: Config) -> list[str]:
    """Messages for every violated weather rule of an enabled config (empty = valid). Never raises: the guards
    run before any parameter is built, and a failed guard returns at once.

    Order: guards; the ``WeatherParams`` rules (W0 spec §2.5 rules 1-14, 16, 18); the world rules (W-1 the
    cells are a multiple of the field lattice, W-2 the pool centres lie in the domain, the 3D geometry and
    load rules via ``check_world``, the 2D load rules R3 and R4); the coupling keys.
    """
    wc, w, ph = cfg.weather, cfg.world, cfg.physics
    bad: list[str] = []
    # 0. Guards, before ``params_from_config`` computes ``dx = world.size / cells``.
    if not (math.isfinite(w.size) and w.size > 0 and math.isfinite(w.field_cell) and w.field_cell > 0):
        bad.append("weather needs a finite world.size > 0 and world.field_cell > 0")
    n = wc.cells
    if not (isinstance(n, int) and not isinstance(n, bool) and 8 <= n <= 1024):
        bad.append(f"weather.cells must be an integer in [8, 1024], got {n!r}")
    if len(wc.pools) % 7 != 0:
        bad.append(
            "weather.pools must hold 7 numbers per pool (x, y, sx, sy, amp, vx, vy), "
            f"got {len(wc.pools)} values"
        )
    if not all(math.isfinite(x) for x in wc.pools):
        bad.append("weather.pools must be finite numbers")
    if bad:
        return bad
    p = params_from_config(cfg)
    bad += p.validate()
    d3 = w.dims == 3
    # 2. World rules.
    ratio = w.size / w.field_cell
    if ratio == round(ratio) and round(ratio) > 0 and n % round(ratio) != 0:
        bad.append(
            f"rule W-1: weather.cells {n} must be a multiple of the field lattice FG = {round(ratio)} "
            "(the rain is a block mean onto it)"
        )
    for i in range(0, len(wc.pools), 7):
        x, y = wc.pools[i], wc.pools[i + 1]
        if not (0.0 <= x < w.size and 0.0 <= y < w.size):
            bad.append(
                f"rule W-2: pool {i // 7} centre ({x:g}, {y:g}) must lie in [0, world.size = {w.size:g})"
            )
    if d3:
        bad += check_world(p, cfg)
    else:  # R3 and R4: the horizontal load rules, with the surface-layer factor absent (a = alpha0)
        a = wc.alpha0
        if ph.drag > 0:
            speed = (ph.thrust_max + a * ph.drag * wc.u_max) / ph.drag
            if not speed <= ph.v_max:
                bad.append(
                    f"rule 16 (2D): (thrust_max + alpha0*drag*u_max)/drag = {speed:.4g} must be <= "
                    f"v_max = {ph.v_max:g}"
                )
        load = ph.thrust_max + a * ph.drag * wc.u_max
        if not load < 2 * ph.radius * ph.k_repel:
            bad.append(
                f"rule R4 (2D): thrust_max + alpha0*drag*u_max = {load:.4g} must be < 2*radius*k_repel = "
                f"{2 * ph.radius * ph.k_repel:.4g}"
            )
    # 4. Coupling keys.
    if not 0.0 <= wc.storm_p <= 1.0:
        bad.append(f"weather.storm_p must be in [0, 1], got {wc.storm_p}")
    if not wc.storm_u0 >= 0.0:
        bad.append(f"weather.storm_u0 must be >= 0, got {wc.storm_u0}")
    if not wc.storm_span > 0.0:
        bad.append(f"weather.storm_span must be > 0, got {wc.storm_span}")
    if not (wc.storm_break_u == 0.0 or wc.storm_break_u >= wc.storm_u0):
        bad.append(
            f"weather.storm_break_u {wc.storm_break_u} must be 0 (off) or >= weather.storm_u0 "
            f"{wc.storm_u0}: the threshold acts on the hazard speed (the eddy speed under "
            "weather.storm_eddy, else the air speed including the layer-2 jet)"
        )
    if not 0.0 <= wc.rain_floor <= 1.0:
        bad.append(f"weather.rain_floor must be in [0, 1], got {wc.rain_floor}")
    if not d3 and wc.rain_floor != 0.0:
        bad.append(f"weather.rain_floor must be 0 in a 2D world (no layers), got {wc.rain_floor}")
    if cfg.director.enabled and wc.rain_floor != 0.0:
        bad.append(
            "weather.rain_floor must be 0 when director.enabled: the floor K and the seasonal table do not "
            f"carry the rain, got {wc.rain_floor}"
        )
    if not wc.rain_gain >= 0.0:
        bad.append(f"weather.rain_gain must be >= 0, got {wc.rain_gain}")
    elif wc.gamma > 0 and not wc.rain_gain * (wc.H1 + wc.H2 - 2.0 * p.h_min) / wc.gamma <= _INT_BOUND:
        bad.append(
            f"weather.rain_gain {wc.rain_gain:g} too large: rain_gain*(H1 + H2 - 2*h_min)/gamma must be "
            "<= 2^30 (the int32 range of a column's rain)"
        )
    # W2 couplings (DESIGN §32). The comparisons are written so that a NaN fails them.
    if not 0.0 <= wc.seed_gain <= 0.1:
        bad.append(f"weather.seed_gain must be in [0, 0.1], got {wc.seed_gain}")
    if not (wc.seed_c0 > 0.0 and math.isfinite(wc.seed_c0)):
        bad.append(f"weather.seed_c0 must be finite and > 0, got {wc.seed_c0}")
    if not 0.0 <= wc.shade_gain <= 0.05:
        bad.append(f"weather.shade_gain must be in [0, 0.05], got {wc.shade_gain}")
    if not (wc.shade_n0 > 0.0 and math.isfinite(wc.shade_n0)):
        bad.append(f"weather.shade_n0 must be finite and > 0, got {wc.shade_n0}")
    if not 0.0 <= wc.wind_scale <= 1.0:
        bad.append(f"weather.wind_scale must be in [0, 1], got {wc.wind_scale}")
    return bad


# --- the wind the engine applies, and the update (W1 build step 2, DESIGN §30) ---------------------------


class AgentForce(NamedTuple):
    """The wind as the engine step uses it."""

    force: mx.array  # (N, nd) float32: ``alpha·drag·u_air``, the drag force of the moving air
    speed: mx.array  # (N,) float32: the hazard speed: the air speed (clipped wind, jet included), or under
    # ``storm_eddy`` the eddy speed; not the agent's
    breaks: mx.array | None  # (N,) bool: ``speed > storm_break_u``; None when ``storm_break_u == 0``
    clip_bad: mx.array  # () bool: some alive row's wind was clipped (a health signal, W0 saw none)


def agent_wind(
    ws: WeatherState, pos: mx.array, buoy_cap: mx.array | None, alive: mx.array, cfg: Config
) -> AgentForce:
    """The wind force on every row at ``pos`` (start-of-step positions) from the stored weather ``ws``.

    3D (``buoy_cap`` given): ``u_air`` is ``sample_wind`` at ``(x, y, z)`` and the coupling is ``alpha0 +
    alpha1·buoy_cap``. 2D (``buoy_cap is None``): ``u_air`` is the layer-1 wind at ``(x, y)`` and the
    coupling is ``alpha0`` (``alpha1`` is inert without a cap). The force is ``alpha·drag·u_air``:
    ``physics.integrate`` divides it by the same ``drag``, so an unobstructed agent drifts at
    ``alpha·u_air``. ``alive`` keeps dead rows out of the clip counts; their force is zeroed by the physics
    (``survivor``).

    W2 (DESIGN §32), each a static branch on a config constant: ``wind_eddy_force`` drives the agents with
    the eddy wind ``(u_h - ū_h, w)`` instead of ``u_air`` (the weather is untouched); ``wind_scale``
    multiplies the force (0: still air for the agents); ``storm_eddy`` makes ``speed`` the eddy speed
    ``sqrt(|u_h - ū_h|² + w²)``, which the hazard and the bond break read. The defaults take exactly the
    W1 expressions.
    """
    p = params_from_config(cfg)
    wc = cfg.weather
    drag = cfg.physics.drag
    eddy = wc.storm_eddy or wc.wind_eddy_force
    if buoy_cap is not None:  # static: 3D
        wf = wind_fields(ws, p)  # built once per step (module docstring: never twice in one graph)
        aw = sample_wind(ws, pos, p, field_cell=cfg.world.field_cell, fields=wf, valid=alive, eddy=eddy)
        a = alpha(buoy_cap, p)[:, None]
        ue = mx.concatenate([aw.u_eddy, aw.u[:, 2:3]], axis=-1) if eddy else aw.u
    else:
        aw = sample_wind_h(ws, pos, p, valid=alive, eddy=eddy)
        a = p.alpha0
        ue = aw.u_eddy if eddy else aw.u
    drive = ue if wc.wind_eddy_force else aw.u
    force = a * drag * drive
    if wc.wind_scale != 1.0:  # static
        force = wc.wind_scale * force
    hz = ue if wc.storm_eddy else aw.u
    speed = mx.sqrt((hz * hz).sum(axis=-1))
    breaks = speed > cfg.weather.storm_break_u if cfg.weather.storm_break_u > 0 else None
    return AgentForce(force, speed, breaks, (aw.n_clip_h + aw.n_clip_w) > 0)


def agent_drift(
    wx: mx.array, pos: mx.array, buoy_cap: mx.array | None, alive: mx.array, cfg: Config
) -> mx.array:
    """The velocity the wind alone would give a free agent, ``(N, nd)`` float32: ``agent_wind(...).force /
    drag`` (``alpha·u_air``; under ``wind_eddy_force`` the eddy wind, under ``wind_scale`` scaled, as the
    engine applies it). For the observers (DESIGN §32): the labels' velocity is ``vel - drift``. ``wx`` is
    the stored (packed) weather and ``pos`` the positions it is sampled at; the observer calls it on the
    post-step state in its own compiled fold, so the step graph is untouched."""
    return agent_wind(unpack(wx), pos, buoy_cap, alive, cfg).force / cfg.physics.drag


def air_speeds(
    ws: WeatherState, pos: mx.array, buoy_cap: mx.array | None, alive: mx.array, cfg: Config
) -> tuple[mx.array, mx.array]:
    """``(total, eddy)`` air speeds of every row, ``(N,)`` float32 each: ``|u_air|`` (the clipped wind, jet
    and ``w`` included) and ``sqrt(|u_h - ū_h|² + w²)`` (the eddy speed of ``storm_eddy``), whatever the
    config keys. For the metrics rows (eager, once per snapshot); ``agent_wind`` is what the step uses."""
    p = params_from_config(cfg)
    if buoy_cap is not None:  # static: 3D
        aw = sample_wind(ws, pos, p, field_cell=cfg.world.field_cell, valid=alive, eddy=True)
        ue = mx.concatenate([aw.u_eddy, aw.u[:, 2:3]], axis=-1)
    else:
        aw = sample_wind_h(ws, pos, p, valid=alive, eddy=True)
        ue = aw.u_eddy
    return mx.sqrt((aw.u * aw.u).sum(axis=-1)), mx.sqrt((ue * ue).sum(axis=-1))


def advance(
    ws: WeatherState,
    t: mx.array,
    key: mx.array,
    cfg: Config,
    qs: mx.array | None = None,
    t_mult: mx.array | None = None,
) -> tuple[WeatherState, WeatherDiag]:
    """One weather step at engine step ``t`` (an int32 () array): the warm pools of step ``t`` times the shade
    ``t_mult``, the seeding factor ``qs`` on ``Q_s``, no momentum forcing. ``key`` is the ``k_disturb`` key of
    the step's weather stream split.

    ``qs`` and ``t_mult`` are the (n, n) organism forcings of ``organism_forcing`` (DESIGN §32); None (the
    default, and what the step passes with ``seed_gain = 0`` / ``shade_gain = 0``) takes exactly the W1
    expressions: ``qs = 1`` and ``T_s`` unscaled, so without them the result depends on no agent array."""
    p = params_from_config(cfg)
    t_s = warm_pools(p, t)
    if t_mult is not None:  # static: shade (W2)
        t_s = t_s * t_mult
    fo = WeatherForcing(t_s, qs if qs is not None else mx.ones((p.n, p.n), dtype=_F32))
    return step(ws, fo, key, p)


def organism_forcing(
    chem: mx.array, pos: mx.array, alive: mx.array, ws: WeatherState, cfg: Config
) -> tuple[mx.array | None, mx.array | None]:
    """``(qs, t_mult)`` for ``advance`` from the **start-of-step** state: the seeding factor from ``chem`` and
    the stored weather ``ws``, the shade from ``pos`` and ``alive``. Each is None while its gain is 0
    (static), so a world without the couplings builds the W1 graph. No random draw; the engine and the
    staged benchmark call this one function (DESIGN §32)."""
    wc = cfg.weather
    qs = seeding_qs(chem, ws, cfg) if wc.seed_gain > 0 else None
    t_mult = shade_mult(pos, alive, cfg) if wc.shade_gain > 0 else None
    return qs, t_mult


# The W0 health gate (STORMS_W0): hyperbolicity below this, and the local moisture ratio below 1.
_HYPER_GATE: Final = 0.8


def health(wd: WeatherDiag, wind: AgentForce) -> mx.array:
    """() bool: any W0 health-gate quantity fires this step (limiter or clamp counters, hyperbolicity at the
    gate, a local moisture ratio of 1) or a wind sample was clipped. Advisory, ``Diag.WEATHER_WARN``."""
    bad = wind.clip_bad | (wd.hyper_max >= _HYPER_GATE) | (wd.eps_local_max >= 1.0)
    for n in (
        wd.n_limited,
        wd.n_clamp,
        wd.n_cond_limited,
        wd.n_relax_limited,
        wd.n_evap_limited,
        wd.n_safety,
    ):
        bad = bad | (n > 0)
    return bad


# --- rain-fed regrowth (W1 build step 3, DESIGN §30) ------------------------------------------------------


class RainIn(NamedTuple):
    """What the resource field needs of the weather: three (n, n) float32 arrays on the weather grid."""

    P: mx.array  # the rain rate of the last weather step
    zi: mx.array  # the interface height ``H·h1/(h1 + h2)``: the condensation level
    dc: mx.array  # the cloud depth ``D_c``


def rain_in(ws: WeatherState, cfg: Config) -> RainIn:
    """The rain, the condensation level and the cloud depth of ``ws`` (the state after a weather step)."""
    return RainIn(ws.P, interface_height(ws, params_from_config(cfg)), ws.Dc)


def rain_for(ws: WeatherState, cfg: Config) -> RainIn | None:
    """``rain_in`` when rain feeds the resource (``rain_gain > 0``, static), else None: the field update and
    ``K_est`` then take exactly the code path that has no rain term."""
    return rain_in(ws, cfg) if cfg.weather.rain_gain > 0 else None


def _blocks(a: mx.array, fg: int) -> mx.array:
    """(n, n) -> (fg, r, fg, r) with ``r = n // fg`` (validate rule W-1: ``cells % fg == 0``)."""
    r = a.shape[-1] // fg
    return a.reshape(fg, r, fg, r)


def cloud_weights(zi: mx.array, dc: mx.array, cfg: Config) -> mx.array:
    """The share of a column's cloud in each field layer, ``(FZ, FG, FG)`` (3D only): the cloud occupies
    ``[zi, zi + dc]``, layer ``k`` (``[fc·k, fc·(k + 1)]``) has the overlap fraction ``f_k`` and the weight
    ``f_k/Σf``. ``zi`` and ``dc`` are ``(FG, FG)``. One function for the rain food (``rain_regen``) and the
    seeding band (``seeding_qs``): they act on the same layers (DESIGN §32)."""
    w = cfg.world
    fz, fc = round(w.height / w.field_cell), w.field_cell
    k = mx.arange(fz, dtype=_F32).reshape(fz, 1, 1)
    frac = mx.clip((mx.minimum(zi + dc, fc * (k + 1.0)) - mx.maximum(zi, fc * k)) / fc, 0.0, 1.0)
    return frac / mx.maximum(frac.sum(axis=0), 1e-6)


def rain_regen(rain: RainIn, cfg: Config) -> mx.array:
    """The rain as integer regeneration quanta per field cell per step: ``(FG, FG)`` int32 in 2D, ``(FZ, FG,
    FG)`` in 3D (§30; the term added to ``fields.regen_rate``).

    The weather grid is reduced to the field lattice by 2x2 (``cells // FG``) blocks: the rain rate by its
    mean, the condensation level and the cloud depth by their rain-weighted means (a block with one raining
    cell takes that cell's level and depth). ``q = rain_gain·P`` is the column's food per step. 2D:
    ``floor(q)``. 3D: the cloud occupies ``[zi, zi + dc]``; layer ``k`` (``[fc·k, fc·(k + 1)]``) has the
    overlap fraction ``f_k`` and takes ``floor(q·f_k/Σf)``, so a column's food is ``rain_gain·P`` whatever
    the cloud depth (less the floors: about half a quantum per used layer); ``rain_floor·q`` more goes to
    layer 0. Reshapes and reductions only: no scatter, and nothing branches on an array value.

    ``weather.rain_zonal`` (W2, a control) replaces ``P``, ``P·zi`` and ``P·dc`` by their x-means before the
    reduction: the same layer weights, but the food depends on latitude only.
    """
    wc, w = cfg.weather, cfg.world
    fg = round(w.size / w.field_cell)
    rp, pz, pd = rain.P, rain.P * rain.zi, rain.P * rain.dc
    if wc.rain_zonal:  # static (W2): the x-mean of each, so the food depends on latitude only
        rp, pz, pd = (mx.broadcast_to(a.mean(axis=-1, keepdims=True), a.shape) for a in (rp, pz, pd))
    pc = _blocks(rp, fg).mean(axis=(1, 3))
    q = wc.rain_gain * pc
    if w.dims != 3:  # static: one layer
        return ops.floor_i32(q)
    fz = round(w.height / w.field_cell)
    ps = _blocks(rp, fg).sum(axis=(1, 3))
    den = mx.maximum(ps, 1e-30)
    zi = _blocks(pz, fg).sum(axis=(1, 3)) / den
    dc = _blocks(pd, fg).sum(axis=(1, 3)) / den
    share = cloud_weights(zi, dc, cfg)
    out = ops.floor_i32(q * share)
    if wc.rain_floor > 0:  # static: the floor layer's extra share
        floor_q = ops.floor_i32(wc.rain_floor * q)
        out = out + mx.concatenate([floor_q[None], mx.zeros((fz - 1, fg, fg), dtype=_I32)])
    return out


# --- organisms -> weather (W2, DESIGN §32) ---------------------------------------------------------------


# The relative dead band of the seeding excess (``seeding_qs``): above the float32 rounding of a lattice mean.
_SEED_DEADBAND: Final = 1e-4


def _repeat_blocks(a: mx.array, n: int) -> mx.array:
    """(FG, FG) -> (n, n): each value repeated over its ``n // FG`` square block (reshape and broadcast)."""
    fg = a.shape[-1]
    r = n // fg
    return mx.broadcast_to(a[:, None, :, None], (fg, r, fg, r)).reshape(n, n)


def seeding_response(chem: mx.array, ws: WeatherState, cfg: Config) -> mx.array:
    """The saturating seeding response ``r`` on the field lattice, ``(FG, FG)`` float32 in ``[0, 1)``.

    ``chem`` is the field state's chemical channels, ``(NCH, FZ, FG, FG)`` (3D) or ``(NCH, FG, FG)`` (2D);
    ``ws`` the stored weather. The signal is the c2 averaged over the cloud layers, ``b = Σ_k w_k·chem[2, k]``
    with the weights of ``cloud_weights`` (the layers the rain food uses) at the column's interface height
    ``zi`` (plain 2x2 block mean) and cloud depth ``max(block mean of D_c, seed_dmin)`` (plain means and a
    floor: where it does not rain ``D_c`` is 0, so a rain-weighted band would be empty and seeding could
    not start a rain).

    The excess over the lattice mean, ``e = b - mean(b)·(1 + 1e-4)``, goes through ``r = e⁺/(e⁺ + seed_c0)``
    (zero for ``e <= 0``: no anti-seeding). The 1e-4 dead band is far above the float32 rounding of the mean
    over the lattice and of the layer weights (they sum to 1 to 6e-8), so a uniform c2 gives ``r == 0``
    exactly. 2D: the band is the single layer. Reshapes, reductions and arithmetic only; no random draw."""
    wc, w = cfg.weather, cfg.world
    p = params_from_config(cfg)
    fg = round(w.size / w.field_cell)
    if w.dims == 3:  # static
        zi = _blocks(interface_height(ws, p), fg).mean(axis=(1, 3))
        dc = mx.maximum(_blocks(ws.Dc, fg).mean(axis=(1, 3)), p.seed_dmin)
        b = (cloud_weights(zi, dc, cfg) * chem[2]).sum(axis=0)
    else:
        b = chem[2]
    mean = b.mean()
    e = mx.maximum(b - mean * (1.0 + _SEED_DEADBAND), 0.0)
    return e / (e + wc.seed_c0)


def seeding_qs(chem: mx.array, ws: WeatherState, cfg: Config) -> mx.array:
    """The rain-seeding factor on the threshold ``Q_s``, (n, n) float32 in ``[1 - seed_gain, 1]``:
    ``qs = 1 - seed_gain·r`` with ``r`` of ``seeding_response``, repeated over the 2x2 weather cells of each
    field column. The caller evaluates it on the start-of-step state (``organism_forcing``)."""
    return _repeat_blocks(
        1.0 - cfg.weather.seed_gain * seeding_response(chem, ws, cfg), params_from_config(cfg).n
    )


def _box3(a: mx.array) -> mx.array:
    """The periodic 3x3 mean of the last two axes, separable: a 1x3 mean along x, then a 3x1 mean along y of
    that result. ``a`` is rolled by its two x shifts once and the row mean by its two y shifts once (the
    once-per-array rule); the sum is conserved."""
    row = (_west(a) + a + _east(a)) / 3.0
    return (_south(row) + row + _north(row)) / 3.0


def shade_strength(pos: mx.array, alive: mx.array, cfg: Config) -> mx.array:
    """The shade ``s`` on the field lattice, ``(FG, FG)`` float32 in ``[0, shade_gain)``.

    The alive agents are counted per field column (all heights; ``ops.scatter_add_i32``) and averaged over the
    periodic 3x3 neighbourhood, ``d`` agents per column; ``s = shade_gain·d/(d + shade_n0)``. The box mean is
    separable (``_box3``). Dead rows are excluded (their positions are stale); no random draw."""
    wc, w = cfg.weather, cfg.world
    fg = round(w.size / w.field_cell)
    cxy = ops.floor_i32(pos[:, :2] / w.field_cell) % fg
    cnt = ops.scatter_add_i32(fg * fg, cxy[:, 1] * fg + cxy[:, 0], 1, alive).reshape(fg, fg).astype(_F32)
    d = _box3(cnt)
    return wc.shade_gain * d / (d + wc.shade_n0)


def shade_mult(pos: mx.array, alive: mx.array, cfg: Config) -> mx.array:
    """The shade on the warm-pool map ``T_s``, (n, n) float32 in ``[1 - shade_gain, 1]`` (multiply ``T_s``):
    ``1 - s`` of ``shade_strength`` repeated over the 2x2 weather cells of each field column. The caller
    passes the start-of-step ``pos`` and ``alive`` (``organism_forcing``)."""
    return _repeat_blocks(1.0 - shade_strength(pos, alive, cfg), params_from_config(cfg).n)
