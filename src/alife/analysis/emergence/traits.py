"""The trait registry of the window recorder (DESIGN §44.2).

A registry is a list of ``Trait(name, getter)``. A getter receives the **alive rows** of one frame as a
mapping of host numpy arrays (``ctx``: every per-agent state field by name, plus ``step``, ``speed`` and
``n_bonds`` computed by the recorder) and returns one value per alive row, or ``None`` when the state does
not carry what it needs. A missing attribute drops the column (a run from before a feature has fewer
columns); it never raises. Adding a feature = one ``register`` line from that feature's PR, the only place a
feature PR touches this suite. Numpy only: the recorder (``record.py``) is the one module that touches MLX.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

Getter = Callable[[Mapping[str, Any]], "np.ndarray | None"]


@dataclass(frozen=True)
class Trait:
    name: str
    getter: Getter
    feature: str = "base"  # which feature adds the column ("base" is always attempted)


def _field(key: str, col: int | None = None) -> Getter:
    """The per-agent state field ``key`` (column ``col`` of a 2D one); None when the state lacks it."""

    def get(ctx: Mapping[str, Any]) -> np.ndarray | None:
        a = ctx.get(key)
        if a is None:
            return None
        a = np.asarray(a)
        if col is not None:
            return a[:, col] if a.ndim == 2 and a.shape[1] > col else None
        return a if a.ndim == 1 else None

    return get


def _age(ctx: Mapping[str, Any]) -> np.ndarray | None:
    b = ctx.get("birth_step")
    return None if b is None else int(ctx["step"]) - np.asarray(b, dtype=np.int64)


BASE: tuple[Trait, ...] = (
    *(Trait(f"tag{i}", _field("tag", i)) for i in range(4)),
    Trait("buoy_cap", _field("buoy_cap")),
    Trait("offspring_frac", _field("offspring_frac")),
    Trait("energy", _field("energy")),
    Trait("age", _age),
    Trait("n_bonds", _field("n_bonds")),
    Trait("speed", _field("speed")),  # displacement over the last tau steps / tau (NaN in the first frame)
)

_REGISTRY: list[Trait] = list(BASE)


def registry() -> list[Trait]:
    """The registered traits in column order (a copy)."""
    return list(_REGISTRY)


def register(trait: Trait) -> None:
    """Append (or replace by name) a trait; a feature PR calls this once per column."""
    for i, t in enumerate(_REGISTRY):
        if t.name == trait.name:
            _REGISTRY[i] = trait
            return
    _REGISTRY.append(trait)


def reset() -> None:
    """Back to the base columns (tests)."""
    _REGISTRY[:] = BASE


def columns(
    ctx: Mapping[str, Any], n: int, traits: list[Trait] | None = None
) -> tuple[list[str], np.ndarray]:
    """``(names, values (n, K) float32)`` of the registered columns the state carries (float32, not the spec's
    float16: age and energy exceed its range).

    The set of columns is decided by the getters on this frame; the recorder calls it on the first frame and
    keeps the names, then ``values_for`` on every frame, so the column set is fixed for a file.
    """
    names, cols = [], []
    for t in traits if traits is not None else _REGISTRY:
        v = t.getter(ctx)
        if v is None:
            continue
        v = np.asarray(v)
        if v.shape != (n,):
            continue  # a getter that returns the wrong shape drops its column rather than corrupt the file
        names.append(t.name)
        cols.append(v.astype(np.float32))
    return names, (np.stack(cols, axis=1) if cols else np.zeros((n, 0), np.float32))


def values_for(ctx: Mapping[str, Any], n: int, names: list[str]) -> np.ndarray:
    """The columns ``names`` on a later frame (NaN where a getter no longer provides the column)."""
    by = {t.name: t for t in _REGISTRY}
    out = np.full((n, len(names)), np.nan, np.float32)
    for j, nm in enumerate(names):
        t = by.get(nm)
        v = t.getter(ctx) if t else None
        if v is not None and np.asarray(v).shape == (n,):
            out[:, j] = np.asarray(v).astype(np.float32)
    return out
