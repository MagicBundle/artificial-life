"""In-graph cheap invariant checks and the exhaustive host check (§22.2, §22.3). Owned by WP4."""

from __future__ import annotations

import mlx.core as mx

from alife.config import Config
from alife.state import State
from alife.step import StepOut


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
    """int32 ``Diag`` bits for this step, OR-ed into ``s.diag`` by the step."""
    raise NotImplementedError


def full_check(s: State, cfg: Config, prev: State | None = None, out: StepOut | None = None) -> list[str]:
    """Host numpy check of every §22.2 invariant; ``[]`` means OK. Stateless."""
    raise NotImplementedError
