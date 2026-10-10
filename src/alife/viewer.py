"""Optional pygame-ce live viewer (§20.2); pygame is imported lazily. Owned by WP6."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alife.config import Config
from alife.state import State

if TYPE_CHECKING:
    from alife.runio import RunDir


def run_viewer(
    cfg: Config,
    state: State,
    *,
    rundir: RunDir | None = None,
    size: tuple[int, int] = (900, 900),
    steps_per_frame: int = 1,
) -> None:
    """Interactive viewer loop; holds only the current state (donation rule)."""
    raise NotImplementedError
