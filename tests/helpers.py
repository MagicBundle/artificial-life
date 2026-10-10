"""Test helpers: hand-built scenes on top of a fresh ``init_state`` (§22.0)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import mlx.core as mx
import numpy as np

from alife import brain, layout
from alife.config import Config, dims
from alife.state import AGENT_FIELDS, State, init_state, to_numpy, total_energy

# Fused Metal kernels may round float ops differently from eager ones (e.g. FMA contraction), so compiled vs
# eager is bitwise only on CPU. Integer results must still agree exactly on every backend.
ON_GPU = mx.default_device().type == mx.gpu


def make_scene(
    cfg: Config,
    agents: Sequence[Mapping[str, Any]] = (),
    *,
    bonds: Sequence[tuple[int, ...]] = (),
    resource: Any = None,
    chem: Any = None,
    **fields: Any,
) -> State:
    """Build a State holding exactly ``agents`` (every other row dead-canonical).

    - Each agent is a mapping of per-agent State fields (``pos``, ``energy``, ``heading``, ``tag``, ``W1``...)
      plus an optional ``row`` (default: its list index). Unset fields default to a founder-like agent:
      ``uid = list index``, ``energy = e_init``, genes at ``*_init``, zero weights, ``birth_step = −1``.
      ``brain_cost`` defaults to ``l1_cost`` of the row's weights.
    - ``bonds``: ``(i, j)`` pairs (first free slots) or ``(i, slot_i, j, slot_j)``, with row indices.
    - ``resource``: None keeps the initial field, a scalar fills it, an array replaces it. ``chem`` likewise
      (default zeros).
    - ``fields``: any other State field (e.g. ``step``, ``key``, ``next_uid``) set verbatim.
    ``total`` and ``total0`` are recomputed unless given. The returned arrays are evaluated.
    """
    d = dims(cfg)
    s0 = init_state(cfg)
    base = {name: to_numpy(a) for name, a in s0._asdict().items()}
    dtypes = {name: a.dtype for name, a in s0._asdict().items()}
    n, mu = d.n, cfg.mutation

    arr: dict[str, np.ndarray] = {
        "alive": np.zeros(n, bool),
        "uid": np.full(n, -1, np.int64),
        "parent_uid": np.full(n, -1, np.int64),
        "clade": np.zeros(n, np.int32),
        "generation": np.zeros(n, np.int32),
        "birth_step": np.full(n, -1, np.int32),
        "energy": np.zeros(n, np.int32),
        "pos": np.zeros((n, 2), np.float32),
        "vel": np.zeros((n, 2), np.float32),
        "heading": np.zeros(n, np.float32),
        "h": np.zeros((n, d.h), np.float32),
        "msg": np.zeros((n, layout.MSG), np.float32),
        "last_loss": np.zeros(n, np.int32),
        "last_harvest": np.zeros(n, np.int32),
        "last_inflow": np.zeros(n, np.int32),
        "partner": np.full((n, layout.B), -1, np.int32),
        "rev": np.full((n, layout.B), -1, np.int32),
        "tag": np.zeros((n, layout.T), np.float32),
        "mut_sigma": np.full(n, mu.sigma_init, np.float32),
        "mut_rate": np.full(n, mu.rate_init, np.float32),
        "offspring_frac": np.full(n, mu.frac_init, np.float32),
        "brain_cost": np.zeros(n, np.int32),
        "W1": np.zeros((n, d.w1_rows, d.h), np.float32),
        "W2": np.zeros((n, d.w2_rows, layout.D_OUT), np.float32),
    }
    if len(agents) > d.n_cap:
        raise ValueError(f"{len(agents)} agents exceed N_CAP = {d.n_cap}")
    rows_used: list[int] = []
    explicit_cost: set[int] = set()
    for k, spec in enumerate(agents):
        row = int(spec.get("row", k))
        if row in rows_used:
            raise ValueError(f"row {row} used twice")
        rows_used.append(row)
        arr["alive"][row] = True
        arr["uid"][row] = k
        arr["energy"][row] = cfg.energy.e_init
        for key, value in spec.items():
            if key == "row":
                continue
            if key not in AGENT_FIELDS:
                raise KeyError(f"unknown per-agent field {key!r}")
            arr[key][row] = np.asarray(value, dtype=arr[key].dtype)
            if key == "brain_cost":
                explicit_cost.add(row)

    for b in bonds:
        if len(b) == 2:
            i, j = b
            si = int(np.argmax(arr["partner"][i] < 0))
            sj = int(np.argmax(arr["partner"][j] < 0))
            if arr["partner"][i, si] >= 0 or arr["partner"][j, sj] >= 0:
                raise ValueError(f"no free slot for bond {b}")
        else:
            i, si, j, sj = b
        arr["partner"][i, si], arr["rev"][i, si] = j, sj
        arr["partner"][j, sj], arr["rev"][j, sj] = i, si

    w1 = mx.array(arr["W1"]).astype(d.wdtype)
    w2 = mx.array(arr["W2"]).astype(d.wdtype)
    cost = np.asarray(brain.l1_cost(w1, w2, cfg))
    for row in rows_used:
        if row not in explicit_cost:
            arr["brain_cost"][row] = cost[row]

    dead = np.flatnonzero(~arr["alive"])
    out: dict[str, Any] = {name: mx.array(v).astype(dtypes[name]) for name, v in arr.items()}
    out["W1"], out["W2"] = w1, w2
    out["pend_slot"] = mx.array(dead[-d.m :].astype(np.int32))
    out["pend_W1"] = mx.array(base["pend_W1"]).astype(d.wdtype)
    out["pend_W2"] = mx.array(base["pend_W2"]).astype(d.wdtype)
    if resource is None:
        out["resource"] = mx.array(base["resource"])
    else:
        out["resource"] = mx.array(np.broadcast_to(np.asarray(resource, np.int32), (d.fg, d.fg)).copy())
    if chem is None:
        out["chem"] = mx.zeros((layout.NCH, d.fg, d.fg), dtype=mx.float32)
    else:
        out["chem"] = mx.array(np.broadcast_to(np.asarray(chem, np.float32), (layout.NCH, d.fg, d.fg)).copy())
    for name in ("patch_pos", "patch_vel", "key", "step", "ledger_cum", "diag"):
        out[name] = mx.array(base[name]).astype(dtypes[name])
    out["next_uid"] = mx.array(int(arr["uid"].max(initial=-1)) + 1, dtype=mx.int64)
    for name, value in fields.items():
        if name not in State._fields:
            raise KeyError(f"unknown State field {name!r}")
        out[name] = value if isinstance(value, mx.array) else mx.array(np.asarray(value)).astype(dtypes[name])
    total = total_energy(out["energy"], out["alive"], out["resource"], cfg)
    for name in ("total", "total0"):
        if name not in fields:
            out[name] = total
    s = State(**out)
    mx.eval(s)
    return s


def set_output_bias(s: State, cfg: Config, bias: Mapping[int, float]) -> State:
    """Return ``s`` with ``W2[i, H, out] = value`` for each alive row ``i`` and ``out → value`` in ``bias``.

    Row ``H`` of ``W2`` multiplies the constant hidden unit, so this is the output bias of §6. Values are
    clipped to ``±weight_clip`` and ``brain_cost`` of the alive rows is recomputed from the stored weights.
    Used by the whole-step tests to make particular actions fire (§22.5); the engine never seeds a bias.
    """
    d = dims(cfg)
    clip = cfg.brain.weight_clip
    w2 = to_numpy(s.W2).astype(np.float32)
    alive = to_numpy(s.alive)
    for out, value in bias.items():
        w2[alive, d.h, out] = float(np.clip(value, -clip, clip))
    W2 = mx.array(w2).astype(d.wdtype)
    cost = brain.l1_cost(s.W1, W2, cfg)
    s = s._replace(W2=W2, brain_cost=mx.where(s.alive, cost, s.brain_cost))
    mx.eval(s)
    return s
