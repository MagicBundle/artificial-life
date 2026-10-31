"""The compiled sphere step with event-triggered calls (DESIGN §47, step K1; F21 item 6).

A compiled sphere configuration runs in a child process with an empty kernel cache of its own
(``tests/sphere_run.py``): MLX 0.32.3's compile cache can hand a compiled function another one's kernel when
their constants print alike, and the guard branch is not merged. Children on the weather world of ``tiny3s``
with the hurt and storm channels (the storm threshold is low enough for the calm start's air to open windows):

* ``trajectory``: 60 compiled steps; the 16-term ledger closes, the host check passes every 15 steps, both
  windows open and the voice is spoken and paid (counters non-zero);
* ``compare``: the compiled step equals the eager one from the same states, in both orders (integers exactly,
  floats to 1e-5), with the windows being driven.

Their config also runs ``sphere_check.require_compile_safe`` (built into ``make_step`` for a compiled sphere).
On Metal the trajectory and the comparison run again on the GPU.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import pytest

import sphere_run
from alife.config import Config
from alife.state import LedgerIdx, counter_index
from calls_helpers import calls_cfg

INIT = {"bias": {"OUT_ATTACK": 1.0, "OUT_VOICE": 2.0}}


def cfg_calls() -> Config:
    """tiny3s with its own weather (pools of the sphere tests), hurt + storm; the hazard is off and the storm
    threshold tiny (1e-4): the weather starts calm and gains speed (5 % of the rows are above it at step 20,
    20 % at step 60), so the real air opens windows without killing the population."""
    import dataclasses

    cfg = calls_cfg("sphw", "weather.storm_p=0.0", "weather.storm_u0=0.0001", "calls.hold=6", cost=16.0)
    return dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, compile=True))


@pytest.fixture(scope="module")
def children(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("cold")
    cfg = cfg_calls()
    jobs = {
        "trajectory": ("trajectory", None, (60, 15, "", INIT)),
        "eager-first": ("compare", None, (3, "eager-first", INIT)),
        "compiled-first": ("compare", None, (3, "compiled-first", INIT)),
    }
    if mx.metal.is_available():
        jobs["gpu-trajectory"] = ("trajectory", "gpu", (40, 20, "", INIT))
        jobs["gpu-compare"] = ("compare", "gpu", (2, "eager-first", INIT))

    def go(item):
        name, (kind, device, args) = item
        import json

        argv = [a if not isinstance(a, dict) else json.dumps(a) for a in args]
        try:
            return name, sphere_run.spawn(kind, cfg, Path(root) / name, *argv, device=device)
        except Exception as e:
            return name, e

    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        return dict(ex.map(go, jobs.items()))


def result(children: dict, name: str) -> dict:
    r = children[name]
    if isinstance(r, Exception):
        raise r
    return r


def test_the_compiled_sphere_closes_the_ledger_and_the_windows_work(children) -> None:
    cfg, r = cfg_calls(), result(children, "trajectory")
    assert r["problems"] == [] and r["diag"] == 0 and r["overflow"] == 0 and r["alive"] > 20
    tot = lambda name: int(r["counters_sum"][counter_index(cfg, name)])  # noqa: E731
    assert tot("CALLS_OPEN_HURT") > 0 and tot("CALLS_OPEN_STORM") > 0  # both events opened windows
    assert tot("CALLS_SPOKE_HURT") > 0 and tot("COMM_SPEAKERS") > 0
    assert tot("CALLS_SPOKE_STORM") <= tot("CALLS_OPEN_STORM") and tot("CALLS_SPOKE_HURT") <= tot(
        "CALLS_OPEN_HURT"
    )
    assert r["ledger"][LedgerIdx.HEAT_ACTION] >= tot("COMM_DUE") > 0  # the voice is booked as action heat


@pytest.mark.parametrize("order", ["eager-first", "compiled-first"])
def test_the_compiled_step_equals_the_eager_step_with_calls_on(children, order: str) -> None:
    r = result(children, order)
    assert r["mismatch"] == [] and r["steps"] == 3


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs the Metal GPU")
def test_calls_on_the_gpu_close_the_ledger_and_match_the_eager_step(children) -> None:
    cfg, r = cfg_calls(), result(children, "gpu-trajectory")
    assert r["problems"] == [] and r["diag"] == 0 and r["overflow"] == 0
    assert int(r["counters_sum"][counter_index(cfg, "CALLS_OPEN_HURT")]) > 0
    assert result(children, "gpu-compare")["mismatch"] == []
