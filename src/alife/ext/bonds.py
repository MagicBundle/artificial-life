"""The declaration of the bonds-that-pay feature (DESIGN §40, §43): a row of ``ext.REGISTRY``.

Footprint (``bonds_ext(cfg)``, true when ``roles.enabled`` or ``bonds.share_mode != 0``): thirteen brain
inputs (``role`` 1, ``bond_role`` B, ``bond_kin`` B, ``bond_flow`` B) and two outputs (``role_a``, ``role_b``)
(2D 82/37, 3D 97/41 with only this feature), the conditional per-agent state fields ``role`` (int8),
``role_dwell`` (int16) and ``bond_flow`` (int16, (N, B)), five counters and the config keys of
``[bonds]`` (``share_mode``, ``share_kappa``, ``share_kin_floor``) and ``[roles]``. The implementation lives
in ``alife.bonds`` (the hooks) and ``alife.energy.share_slots`` (the conductive share); imports of them are
made at call time because ``config`` imports ``ext``.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from alife.ext.base import Block, ExtCtx, ExtSpec, FieldSpec

B = 4  # bond slots per row (layout.B; a leaf module cannot import layout)

_KEYS = (
    ("bonds", "share_mode"),
    ("bonds", "share_kappa"),
    ("bonds", "share_kin_floor"),
    ("roles", "enabled"),
    ("roles", "forage_gain"),
    ("roles", "forage_defend_pen"),
    ("roles", "guard_harvest_pen"),
    ("roles", "guard_shield"),
    ("roles", "synergy"),
    ("roles", "ramp_steps"),
    ("roles", "threshold"),
    ("roles", "switch_cost"),
    ("roles", "inherit_role"),
)
_COUNTERS = ("ROLE_SWITCHES", "N_FORAGERS", "N_GUARDS", "SHIELDED_BY_GUARD", "SHARE_SLOTS_ACTIVE")


def bonds_ext(cfg: Any) -> bool:
    """The feature is on: the sharing mode or the roles need the bond block and the state (static)."""
    return bool(cfg.roles.enabled) or cfg.bonds.share_mode != 0


def _in_blocks(cfg: Any, nd: int) -> tuple[Block, ...]:
    return (
        Block("role", 1, False),
        Block("bond_role", B, False),
        Block("bond_kin", B, False),
        Block("bond_flow", B, False),
    )


def _out_blocks(cfg: Any, nd: int) -> tuple[Block, ...]:
    return (Block("role_a", 1, False), Block("role_b", 1, False))


def _state_fields(cfg: Any, nd: int) -> tuple[FieldSpec, ...]:
    def zeros(shape: Any, dtype: Any) -> Any:
        from alife.config import dims

        return lambda cfg: mx.zeros(shape(dims(cfg)), dtype)

    n = lambda d: (d.n,)  # noqa: E731
    nb = lambda d: (d.n, B)  # noqa: E731
    # conserved=False: the role and its dwell are labels and the bond flow a memory of the last net transfer;
    # none holds energy quanta, so a forced removal by `alife migrate` cannot leak TOTAL.
    role_birth = _inherit_role if cfg.roles.inherit_role else "rest"  # static
    return (
        FieldSpec("role", True, n, mx.int8, zeros(n, mx.int8), role_birth, conserved=False),
        FieldSpec("role_dwell", True, n, mx.int16, zeros(n, mx.int16), "rest", conserved=False),
        FieldSpec("bond_flow", True, nb, mx.int16, zeros(nb, mx.int16), "rest", conserved=False),
    )


def _inherit_role(old: Any, rest: Any, b: Any) -> Any:
    """``roles.inherit_role``: a bud takes its parent's role (pre-birth value), an immigrant starts as a
    generalist; the dwell starts at 0 either way."""
    from alife import ops

    child = mx.where(b.imm, rest[b.slot], old[b.parent])
    return ops.place(old, b.newborn, b.born_lane, child.astype(old.dtype))


def _enable(cfg: Any) -> Any:
    """``alife migrate --enable bonds``: the config with ``roles.enabled`` on, which allocates the bond block
    and the state (``--set bonds.share_mode=2`` moves the sharing)."""
    import dataclasses

    return dataclasses.replace(cfg, roles=dataclasses.replace(cfg.roles, enabled=True))


def _inputs(s: Any, cfg: Any, ctx: ExtCtx) -> Any:
    from alife import bonds

    return bonds.ext_inputs(s, cfg, ctx)


def _decode(s: Any, y: Any, ctx: ExtCtx, cfg: Any) -> Any:
    from alife import roles

    return roles.decode(s, y, ctx, cfg)


def _mask(dec: Any, solvent: Any, cfg: Any) -> Any:
    from alife import roles

    return roles.mask(dec, solvent, cfg)


def _due(s: Any, act: Any, y: Any, cfg: Any) -> Any:
    from alife import roles

    return roles.due(s, act, y, cfg)


def _count(s_new: Any, alive0: Any, flows: Any, cfg: Any) -> Any:
    from alife import bonds

    return bonds.ext_count(s_new, alive0, flows, cfg)


def _check(arrays: Any, cfg: Any) -> list[str]:
    """Host check of the numpy bonds fields: ``role`` in {0..3}, ``role_dwell`` not negative, ``bond_flow`` 0
    on empty slots, and the role fields and flows 0 on a dead row. (Here, not in ``alife.bonds``: the host
    reads of numpy arrays are allowed outside the simulation modules, ``tests/test_lint.py`` rule 5.)"""
    import numpy as np

    role, dwell, flow = (np.asarray(arrays[k]) for k in ("role", "role_dwell", "bond_flow"))
    alive, partner = np.asarray(arrays["alive"]), np.asarray(arrays["partner"])
    bad = []
    if ((role < 0) | (role > 3)).any():
        bad.append(f"role: values outside 0..3 (min {role.min()}, max {role.max()})")
    if (dwell < 0).any():
        bad.append(f"role_dwell: negative (min {dwell.min()})")
    if (flow[partner < 0] != 0).any():
        bad.append("bond_flow: non-zero on an empty slot")
    if (flow[~alive] != 0).any() or role[~alive].any() or dwell[~alive].any():
        bad.append("role, role_dwell or bond_flow: non-zero on a dead row")
    return bad


def _validate(cfg: Any) -> None:
    import dataclasses

    from alife.config import BondConfig, ConfigError, RolesConfig

    bc, rc, bad = cfg.bonds, cfg.roles, []
    d = BondConfig()
    if bc.share_mode not in (0, 1, 2):
        bad.append(f"bonds.share_mode {bc.share_mode} must be 0, 1 or 2")
    elif bc.share_mode == 0:  # inert: the conductive keys keep their defaults, so no hash changes in vain
        bad += [
            f"bonds.{k} is inert while bonds.share_mode is 0: it must keep its default"
            for k in ("share_kappa", "share_kin_floor")
            if getattr(bc, k) != getattr(d, k)
        ]
    else:
        if not 0.0 < bc.share_kappa <= 1.0 / (2 * B):
            bad.append(
                f"bonds.share_kappa {bc.share_kappa} must be in (0, {1.0 / (2 * B)}] (1/(2B): the explicit "
                f"diffusion over up to {B} neighbours cannot overshoot)"
            )
        if not 0.0 <= bc.share_kin_floor <= 1.0:
            bad.append(f"bonds.share_kin_floor {bc.share_kin_floor} must be in [0, 1]")
        if not 1 <= bc.share_rate < 1 << 16:
            bad.append(f"bonds.share_rate {bc.share_rate} must be in [1, 2^16) with bonds.share_mode != 0")
    if not rc.enabled:  # inert: every key must keep its default
        r = RolesConfig()
        bad += [
            f"roles.{f.name} is inert while roles.enabled is false: it must keep its default"
            for f in dataclasses.fields(RolesConfig)
            if f.name != "enabled" and getattr(rc, f.name) != getattr(r, f.name)
        ]
    else:
        hmax = cfg.energy.harvest_max
        if rc.forage_gain < 0 or rc.synergy < 0:
            bad.append(f"roles.forage_gain {rc.forage_gain} and roles.synergy {rc.synergy} must be >= 0")
        elif hmax * (1.0 + rc.forage_gain + rc.synergy) >= 65536:
            bad.append(
                f"energy.harvest_max {hmax} x (1 + forage_gain + synergy) must be < 65 536 "
                "(proportional_take needs a demand below 2^16)"
            )
        elif hmax * max(round(rc.forage_gain * 256), round(rc.synergy * 256), 1) * 256 >= 1 << 31:
            bad.append("roles.forage_gain / roles.synergy overflow the int32 harvest-cap product")
        for k in ("forage_defend_pen", "guard_harvest_pen"):
            if not 0.0 <= getattr(rc, k) < 1.0:
                bad.append(f"roles.{k} {getattr(rc, k)} must be in [0, 1)")
        if not 0.0 <= rc.guard_shield <= 0.9:
            bad.append(
                f"roles.guard_shield {rc.guard_shield} must be in [0, 0.9] (the demand stays positive)"
            )
        if not 1 <= rc.ramp_steps <= 32767:
            bad.append(f"roles.ramp_steps {rc.ramp_steps} must be in [1, 32767] (role_dwell is int16)")
        if not 0.0 < rc.threshold < 1.0:
            bad.append(f"roles.threshold {rc.threshold} must be in (0, 1)")
        if not 0 <= rc.switch_cost < 1 << 15:
            bad.append(f"roles.switch_cost {rc.switch_cost} must be in [0, 2^15)")
    if bad:
        raise ConfigError("; ".join(bad))


SPEC = ExtSpec(
    name="bonds",
    enabled=bonds_ext,
    layout_on=bonds_ext,
    token=lambda cfg: "bonds@1",
    in_blocks=_in_blocks,
    out_blocks=_out_blocks,
    block_names=("role", "bond_role", "bond_kin", "bond_flow", "role_a", "role_b"),
    state_fields=_state_fields,
    counters=lambda cfg: _COUNTERS,
    act_flag="ROLE_SWITCHED",
    metric_prefix="bond_",
    lever_keys=_KEYS,
    overlay="bonds",
    enable=_enable,
    validate=_validate,
    inputs=_inputs,
    decode=_decode,
    mask=_mask,
    due=_due,
    count=_count,
    check=_check,
)
