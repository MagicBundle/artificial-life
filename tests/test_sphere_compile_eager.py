"""The compiled sphere step against the eager one, in both orders (DESIGN §46; sphere series S2, spec 10.1
item 4).

One configuration per child process with an empty kernel cache, and the two orders (eager first, compiled
first)
in different processes: from the same state each step, integer fields, counters and the ledger equal exactly,
floats to 1e-5. The compiled position-to-cell lookups equal the eager ones in a cold process.
"""

from __future__ import annotations

import pytest

import sphere_run
from alife.config import Config, load_config


def tiny3s() -> Config:
    return load_config(preset="tiny3s", overrides=["brain.weight_dtype=float32"])


@pytest.mark.parametrize("order", ["eager-first", "compiled-first"])
def test_the_compiled_step_equals_the_eager_step_in_both_orders(order: str, tmp_path) -> None:
    """One configuration per process and the two orders in different processes (spec 10.1, item 4): from the
    same state each step, integer fields, counters and ledger equal exactly, floats to 1e-5."""
    r = sphere_run.spawn("compare", tiny3s(), tmp_path / "cold", 2, order)
    assert r["mismatch"] == [] and r["steps"] == 2


def test_the_compiled_lookups_equal_the_eager_ones_in_a_cold_process(tmp_path) -> None:
    assert sphere_run.spawn("lookups", tiny3s(), tmp_path / "cold") == {"ok": True}
