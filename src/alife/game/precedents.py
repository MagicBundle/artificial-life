"""The precedent line of a power's preview (G3b spec §6.4, DESIGN §34.5): what the earlier cases of the
same kind of move did, from a versioned table and not from the world's own log.

A first draft showed "the last time the same action ran", keyed on the action alone. That would have shown a
x0.8 result on a x1.3 preview, it has n = 1, it is absent at first use when the director is off, and it is a
post-hoc pair at the most persuasive place. The table is keyed on ``(action, direction)`` and every row says
how many cases it rests on and where they come from; a key with no row says "no earlier case". The world's own
earlier cases appear only as a ``since`` fact of the power's card.

Host module, no mlx. The sentences are code-written from the row and pass the report lexicon (the lexicon's
lever words are allowed, as in the interventions block): ``tests/test_powers.py``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

__all__ = ["PRECEDENT_VERSION", "TABLE", "Precedent", "direction", "precedent_object"]

PRECEDENT_VERSION: Final = 1


@dataclass(frozen=True)
class Precedent:
    """One row: ``n`` earlier cases of ``factor``, the change of ln ``n_alive`` over the first window with its
    interval, and how long the step took to complete and to reverse after the end."""

    action: str
    direction: str
    factor: float
    n: int
    ln_change: float
    ln_low: float
    ln_high: float
    complete_within: int
    reverses_within: int
    source: str


# RESULTS_3D_RUNS §5.4: the damper's 14 pulses of x0.8 for 10,000 steps with 20,000 steps of data after them
# (ln n_alive -0.136, interval -0.150 to -0.124 over the first 5,000 steps, negative in 14 of 14; the step is
# complete within about 1,000 steps and reverses within about 1,000 steps of the pulse's end).
TABLE: Final[Mapping[tuple[str, str], Precedent]] = {
    ("regen_pulse", "down"): Precedent(
        "regen_pulse", "down", 0.8, 14, -0.136, -0.150, -0.124, 1000, 1000, "RESULTS_3D_RUNS section 5.4"
    ),
}


def direction(action: str, params: Mapping[str, Any]) -> str:
    """The direction bin of a move: ``down`` or ``up`` for a Food pulse by its multiplier, else ``any``."""
    if action == "regen_pulse":
        return "down" if float(params["mult"]) < 1.0 else "up"
    return "any"


def _percent(ln_change: float) -> int:
    return round((1.0 - math.exp(ln_change)) * 100)


def _line(row: Precedent) -> str:
    return (
        f"Earlier x{row.factor:g} pulses ({row.n} in the 3D director runs): the population fell "
        f"{_percent(row.ln_change)}% ({_percent(row.ln_high)}% to {_percent(row.ln_low)}%) within about "
        f"{row.complete_within:,} steps and rose again within about {row.reverses_within:,} steps of the end."
    )


def precedent_object(action: str, params: Mapping[str, Any], label: str) -> dict[str, Any]:
    """The ``precedent`` of a ``PLAN``: always an object for a power, with ``n`` 0 and "No earlier case for
    ..." when the table has no row for the direction of this move. ``step``, ``params`` and ``rows`` keep
    the shape of spec §7.4 (the table has no per-case rows)."""
    bin_ = direction(action, params)
    row = TABLE.get((action, bin_))
    if row is None:
        what = f"x{float(params['mult']):g}" if action == "regen_pulse" else label
        line = f"No earlier case for {what}."
        n, source = 0, None
    else:
        line, n, source = _line(row), row.n, row.source
    return {
        "version": PRECEDENT_VERSION,
        "key": f"{action}:{bin_}",
        "n": n,
        "line": line,
        "source": source,
        "step": None,
        "action": action,
        "params": dict(params),
        "rows": [],
    }
