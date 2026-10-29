"""Arms races: heritable attack, armour, key and lock, and the contest rule (DESIGN §42, stages 1 and 2).

``State.arms`` is an ``(N, 6)`` float32 array in ``[0, 1]``. Columns (the order of the integration plan,
DESIGN §43.2): ``ATK`` attack strength, ``ARM`` armour, ``TOX`` toxin and ``SIG`` display (both held at 0
until their stages land), ``KEY`` the attacker's key and ``LOCK`` the victim's lock (phases on a circle, in
turns, ``[0, 1)``). They are genes, not brain outputs: they mutate at birth with a fixed sigma
(``child_traits``, drawn from ``Stream.ARMS``) and cost energy and speed.

The contest. For an attack by ``i`` on ``j`` the demand is multiplied by ``P_ij = (1 − r_ij)·K_ij``:

    r = κA / (κA + a0 + ATK + ε)           Tullock repulsion by the victim's armour ``A``; ``r < κ/(κ + a0)``
    K = key_floor + (1 − key_floor)·c²     ``c = (1 + cos 2π(KEY_i − LOCK_j)) / 2``, 1 when aligned

``P`` enters ``energy.protection`` as ``arms_p`` (so the ``combat.protect_floor`` bounds it with the other
protections). With zero traits and ``key_floor = 1`` it is the float 1.0 exactly, so the combat arithmetic is
the legacy one. Every factor is in ``[0, 1]``. ε keeps ``0/0`` out of the bounds (E11).

Costs. ``upkeep`` is an integer metabolic upkeep added to ``brain_cost`` at S8 (so it lands in ``HEAT_BASAL``
and the ledger has no new term); it is rounded by dithering, ``floor(x + u)`` with ``u`` a hash of the row's
``uid`` and the step, so a sub-unit cost is paid on average and not rounded to a free ride (E13). Armour slows
the agent: the physics sees the thrust times ``1 − speed_slow·ARM`` (``physics_act``).

No RNG in the step except the births (``child_traits``); no host sync; no scatter. The numpy mirror
``contest_np`` is the same function in float32 for the analysis and the tests.
"""

from __future__ import annotations

import math
from typing import Any, Final, NamedTuple

import mlx.core as mx
import numpy as np

from alife import ops, rng
from alife.config import Config, dims, immigration_possible

ATK: Final = 0
ARM: Final = 1
TOX: Final = 2
SIG: Final = 3
KEY: Final = 4
LOCK: Final = 5
N_TRAITS: Final = 6
NAMES: Final = ("ATK", "ARM", "TOX", "SIG", "KEY", "LOCK")
PHASES: Final = (KEY, LOCK)
EPS: Final = 1e-6  # keeps the Tullock denominator positive when a0, ATK and the armour are all 0 (E11)
SALT: Final = rng.HASH_SALTS["arms"]  # ops.hash32 salt of the arms feature (DESIGN §43.3)
_TWO_PI: Final = 2.0 * math.pi


def rest_array(cfg: Config) -> mx.array:
    """The ``(N, 6)`` initial traits: founders and the fill of an older checkpoint (``FieldSpec.rest``).

    ``ATK`` and ``ARM`` are ``init_atk`` and ``init_arm`` on every row, ``TOX`` and ``SIG`` 0, and the
    keys and locks are uniform from the initialisation stream ``InitStream.ARMS`` (a function of the seed and
    the row, so a fork from a pre-arms checkpoint starts with a diverse set; no RNG enters the step)."""
    a, n = cfg.arms, dims(cfg).n
    kk, kl = mx.random.split(rng.init_keys(cfg.run.seed)[rng.InitStream.ARMS])
    cols = [
        mx.full((n,), a.init_atk, mx.float32),
        mx.full((n,), a.init_arm, mx.float32),
        mx.zeros((n,), mx.float32),
        mx.zeros((n,), mx.float32),
        mx.random.uniform(0.0, 1.0, (n,), key=kk),
        mx.random.uniform(0.0, 1.0, (n,), key=kl),
    ]
    return mx.stack(cols, axis=1)


# --- inheritance ----------------------------------------------------------------------------------------


def sigma_vector(cfg: Config) -> mx.array:
    """The fixed per-column mutation sigma ``(σ, σ, 0, 0, σ_phase, σ_phase)``; the toxin and display columns
    stay at 0 until their stages land (spec §3.1)."""
    a = cfg.arms
    return mx.array([a.sigma, a.sigma, 0.0, 0.0, a.sigma_phase, a.sigma_phase], dtype=mx.float32)


def child_traits(parent: mx.array, z: mx.array, sigma: mx.array) -> mx.array:
    """``parent + sigma·z`` per column, clipped to ``[0, 1]`` for the magnitudes and wrapped into ``[0, 1)``
    for the phases. The wrap is exact: ``c − floor(c)`` can round to 1.0 for a tiny negative ``c`` in float32,
    which is mapped to 0 (a phase is never 1.0, so an angle never accumulates beyond one turn, E12)."""
    c = parent + sigma * z
    wrapped = c - mx.floor(c)
    wrapped = mx.where(wrapped >= 1.0, 0.0, wrapped)
    is_phase = (mx.arange(N_TRAITS) >= KEY)[None, :]
    return mx.where(is_phase, wrapped, mx.clip(c, 0.0, 1.0))


def make_on_birth(cfg: Config):
    """The ``FieldSpec.on_birth`` hook of the ``arms`` field: newborn rows take their parent's traits mutated
    by ``child_traits`` (one normal draw of ``ks[Stream.ARMS]``); an immigrant takes the founders' values with
    a fresh uniform key and lock. A dying row is left as it is."""
    a = cfg.arms
    sigma = sigma_vector(cfg)
    imm_on = immigration_possible(cfg)  # static

    def hook(old: mx.array, rest: mx.array, b: Any) -> mx.array:
        m = b.parent.shape[0]
        k_mut, k_imm = mx.random.split(b.ks[rng.Stream.ARMS])
        z = mx.random.normal((m, N_TRAITS), key=k_mut)
        child = child_traits(old[b.parent], z, sigma)
        if imm_on:  # static
            u = mx.random.uniform(0.0, 1.0, (m, N_TRAITS), key=k_imm)
            founder = mx.stack(
                [
                    mx.full((m,), a.init_atk, mx.float32),
                    mx.full((m,), a.init_arm, mx.float32),
                    mx.zeros((m,), mx.float32),
                    mx.zeros((m,), mx.float32),
                    u[:, KEY],
                    u[:, LOCK],
                ],
                axis=1,
            )
            child = mx.where(b.imm[:, None], founder, child)
        return ops.place(old, b.newborn, b.born_lane, child)

    return hook


# --- the contest ----------------------------------------------------------------------------------------


def effective_armour(arms: mx.array, partner: mx.array | None, cfg: Config) -> mx.array:
    """The victim's armour used in the repulsion, ``(N,)`` in ``[0, 1]``. The seam of the bonds series
    (``arms.pool``, DESIGN §42.5): until it lands the armour is the agent's own."""
    return mx.clip(arms[:, ARM], 0.0, 1.0)


def contest_factor(arms: mx.array, t0: mx.array, partner: mx.array | None, cfg: Config) -> mx.array | None:
    """``P_ij = (1 − r_ij)·K_ij`` of every attacker ``i`` on its target ``t0[i]``, ``(N,)`` float32 in
    ``[0, 1]``; None when both factors are statically off (``kappa == 0`` and ``key_floor == 1``)."""
    a = cfg.arms
    p: mx.array | None = None
    if a.kappa > 0.0:  # static
        armour = effective_armour(arms, partner, cfg)[t0]
        atk = mx.clip(arms[:, ATK], 0.0, 1.0)
        k_arm = a.kappa * armour
        p = 1.0 - k_arm / (k_arm + (a.a0 + EPS) + atk)
    if a.key_floor < 1.0:  # static
        delta = arms[:, KEY] - arms[:, LOCK][t0]
        c = (1.0 + mx.cos(_TWO_PI * delta)) * 0.5
        k = a.key_floor + (1.0 - a.key_floor) * (c * c)
        p = k if p is None else p * k
    return p


def contest_np(
    atk: np.ndarray, armour: np.ndarray, key: np.ndarray, lock: np.ndarray, cfg: Config
) -> np.ndarray:
    """The numpy mirror of ``contest_factor`` for attackers with ``atk``/``key`` against victims with
    ``armour``/``lock`` (broadcastable arrays), float32 throughout; 1.0 where both factors are off."""
    a = cfg.arms
    f = np.float32
    atk, armour = np.clip(np.asarray(atk, f), 0, 1), np.clip(np.asarray(armour, f), 0, 1)
    p = np.ones(np.broadcast(atk, armour, key, lock).shape, f)
    if a.kappa > 0.0:
        k_arm = f(a.kappa) * armour
        p = p * (f(1) - k_arm / (k_arm + f(a.a0 + EPS) + atk))
    if a.key_floor < 1.0:
        c = (f(1) + np.cos(f(_TWO_PI) * (np.asarray(key, f) - np.asarray(lock, f)))) * f(0.5)
        p = p * (f(a.key_floor) + f(1.0 - a.key_floor) * (c * c))
    return p


# --- observers (DESIGN §42.6) ---------------------------------------------------------------------------

COUNTERS: Final = (
    "ARMS_ATK_SUM",
    "ARMS_VIC_ARM_SUM",
    "ARMS_YIELD_SUM",
    "ARMS_MATCH_SUM",
    "ARMS_VIC_TOX_SUM",
)


class ArmsFlows(NamedTuple):
    """The realised contests of a step (``ExtFlows.arms``): per attacker that passed the S9-partner gate, the
    start-of-step traits and the factors the contest used, ``0`` for every other row. Gathered at S11 from the
    start-of-step ``State.arms``, because a victim's row can be taken by a newborn in the same step."""

    atk: mx.array  # (N,) f32: ATK of the attacker
    arm_v: mx.array  # (N,) f32: the victim's effective armour
    tox_v: mx.array  # (N,) f32: the victim's TOX
    m: mx.array  # (N,) f32: the protection product the demand was multiplied by (``Combat.prot``; 1 if none)
    k: mx.array  # (N,) f32: the key and lock match factor K_ij (1 when the axis is off)


class ArmsActs(NamedTuple):
    """``Acts.ext.arms`` (observer channel): ``round(256·[ATK_i, ARM_target, m_ij, K_ij])`` per attacker."""

    contest: mx.array  # (N, 4) int16


def key_factor(arms: mx.array, t0: mx.array, cfg: Config) -> mx.array | None:
    """``K_ij`` of every attacker on its target ``t0`` with the expression of ``contest_factor`` (so that
    ``(1 − r)·K`` reproduces the engine's product bitwise); None when the key axis is off (static)."""
    a = cfg.arms
    if a.key_floor >= 1.0:  # static
        return None
    delta = arms[:, KEY] - arms[:, LOCK][t0]
    c = (1.0 + mx.cos(_TWO_PI * delta)) * 0.5
    return a.key_floor + (1.0 - a.key_floor) * (c * c)


def flows_of(s: Any, t0: mx.array, cb: Any, partner: mx.array, cfg: Config) -> ArmsFlows:
    """The realised-contest record of a step: ``s.arms`` is the start-of-step array, ``t0`` the clipped attack
    targets, ``cb`` the ``Combat`` (its ``attack`` mask and ``prot``), ``partner`` the post-S9 partner."""
    arms = s.arms
    ok = cb.attack
    one = mx.ones(ok.shape, dtype=mx.float32)
    k = key_factor(arms, t0, cfg)
    m = cb.prot if cb.prot is not None else one
    z = mx.zeros(ok.shape, dtype=mx.float32)
    return ArmsFlows(
        atk=mx.where(ok, mx.clip(arms[:, ATK], 0.0, 1.0), z),
        arm_v=mx.where(ok, effective_armour(arms, partner, cfg)[t0], z),
        tox_v=mx.where(ok, mx.clip(arms[:, TOX], 0.0, 1.0)[t0], z),
        m=mx.where(ok, m, z),
        k=mx.where(ok, one if k is None else k, z),
    )


def acts_of(f: ArmsFlows) -> ArmsActs:
    """The observer sub-tuple: quantised from the values the contest already holds."""
    return ArmsActs(mx.round(256.0 * mx.stack([f.atk, f.arm_v, f.m, f.k], axis=1)).astype(mx.int16))


def contested(f: ArmsFlows) -> mx.array:
    """(N,) bool: the row made a realised attack whose bite the protection product reduced (``0 < m < 1``:
    armour, a mismatched lock or a shelter; ``m`` is 0 for a row that did not attack). The
    ``ActFlag.CONTESTED`` bit."""
    return (f.m > 0.0) & (f.m < 1.0)


def count(s_new: Any, alive0: mx.array, flows: Any, cfg: Config) -> dict[str, mx.array]:
    """The counter block (``COUNTERS``): the Σ round(256·x) over the realised attackers of the step, which
    ``interval_row`` divides by ``ATTACKS`` (the number of realised attackers) and by 256."""
    f = flows.ext.arms

    def q(x: mx.array) -> mx.array:
        return mx.round(256.0 * x).astype(mx.int32).sum()

    return {
        "ARMS_ATK_SUM": q(f.atk),
        "ARMS_VIC_ARM_SUM": q(f.arm_v),
        "ARMS_YIELD_SUM": q(f.m),
        "ARMS_MATCH_SUM": q(f.k),
        "ARMS_VIC_TOX_SUM": q(f.tox_v),
    }


# --- costs ----------------------------------------------------------------------------------------------


def upkeep_float(arms: mx.array, cfg: Config) -> mx.array:
    """The exact (un-rounded) metabolic upkeep per row, ``upkeep_atk·ATK² + upkeep_arm·ARM²``."""
    a = cfg.arms
    return a.upkeep_atk * (arms[:, ATK] * arms[:, ATK]) + a.upkeep_arm * (arms[:, ARM] * arms[:, ARM])


def dither(uid: mx.array, step: mx.array) -> mx.array:
    """A deterministic uniform in ``[0, 1)`` per row from the ``uid`` and the step (integer hash, 24 bits)."""
    h = ops.hash32(uid, step, SALT)
    return mx.right_shift(h, mx.array(8, dtype=mx.uint32)).astype(mx.float32) * (1.0 / (1 << 24))


def upkeep(uid: mx.array, step: mx.array, arms: mx.array, cfg: Config) -> mx.array:
    """The integer upkeep per row, ``floor(x + u)`` with ``u`` the dither: its mean is ``x`` exactly, so a
    cost below one quantum is not free (E13), and it is 0 for ``x == 0`` (``u < 1``). int32, not masked by
    alive (``pay_costs`` masks)."""
    x = upkeep_float(arms, cfg)
    return ops.floor_i32(x + dither(uid, step))


def basal_cost(s: Any, cfg: Config) -> mx.array:
    """The ``brain_cost`` argument of ``pay_costs``: the stored brain cost plus the arms upkeep while it is on
    (static). The stored ``State.brain_cost`` is untouched."""
    a = cfg.arms
    if not a.enabled or (a.upkeep_atk == 0.0 and a.upkeep_arm == 0.0):  # static
        return s.brain_cost
    return s.brain_cost + upkeep(s.uid, s.step, s.arms, cfg)


def combat_kw(s: Any, cfg: Config) -> dict[str, mx.array]:
    """The ``energy.combat`` keyword for the arms contest (empty while arms is off)."""
    return {"arms": s.arms} if cfg.arms.enabled else {}


def physics_act(act: Any, s: Any, cfg: Config) -> Any:
    """``act`` with the thrust slowed by the armour, ``thrust·(1 − speed_slow·ARM)`` (the physics call only;
    the thrust cost was paid on the brain's thrust at S8). The factor is in ``[1 − speed_slow, 1]``, so the
    force only shrinks and the stability inequality of ``validate`` still holds."""
    a = cfg.arms
    if not a.enabled or a.speed_slow == 0.0:  # static
        return act
    return act._replace(thrust=act.thrust * (1.0 - a.speed_slow * mx.clip(s.arms[:, ARM], 0.0, 1.0)))


def founder_upkeep(cfg: Config) -> float:
    """The founders' expected upkeep (``metrics._c_ref``): the dither makes the mean exactly ``x``."""
    a = cfg.arms
    if not a.enabled:
        return 0.0
    return a.upkeep_atk * a.init_atk**2 + a.upkeep_arm * a.init_arm**2


# --- host check -----------------------------------------------------------------------------------------


def check(arrays: Any, cfg: Config) -> list[str]:
    """Host-side check of the numpy state fields: finite, in ``[0, 1]``, phases below 1, and the inert
    columns (``TOX``, ``SIG``) at 0 (their stages have not landed)."""
    x = np.asarray(arrays["arms"])
    problems = []
    if not np.isfinite(x).all() or x.min(initial=0.0) < 0.0 or x.max(initial=0.0) > 1.0:
        problems.append("arms: outside [0, 1] or non-finite")
    elif (x[:, list(PHASES)] >= 1.0).any():
        problems.append("arms: a key or lock phase is not below 1 (unwrapped)")
    if x[:, TOX].any() or x[:, SIG].any():
        problems.append("arms: toxin or display trait non-zero before their stage")
    return problems
