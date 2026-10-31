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


VARIANTS = {
    "": (),
    "far-": ("calls.far_range=20", "calls.far_xref=2.0"),
    "reflex-": ("calls.far_range=20", "calls.control=reflex"),
}


def cfg_calls(variant: str = "") -> Config:
    """tiny3s with its own weather (pools of the sphere tests), hurt + storm; the hazard is off and the storm
    threshold tiny (1e-4): the weather starts calm and gains speed (5 % of the rows are above it at step 20,
    20 % at step 60), so the real air opens windows without killing the population."""
    import dataclasses

    cfg = calls_cfg(
        "sphw",
        "weather.storm_p=0.0",
        "weather.storm_u0=0.0001",
        "calls.hold=6",
        *VARIANTS[variant],
        cost=16.0,
    )
    return dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, compile=True))


@pytest.fixture(scope="module")
def children(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("cold")
    jobs = {}
    for v in VARIANTS:  # the K1 world, with the far hearing, and with the reflex oracle
        jobs[f"{v}trajectory"] = (v, "trajectory", None, (60, 15, "", INIT))
        jobs[f"{v}eager-first"] = (v, "compare", None, (3, "eager-first", INIT))
        if not v:  # the far variants run one order (every cold child compiles for minutes on the CPU)
            jobs[f"{v}compiled-first"] = (v, "compare", None, (3, "compiled-first", INIT))
        if mx.metal.is_available():
            jobs[f"{v}gpu-trajectory"] = (v, "trajectory", "gpu", (40, 20, "", INIT))
            jobs[f"{v}gpu-compare"] = (v, "compare", "gpu", (2, "eager-first", INIT))

    def go(item):
        name, (v, kind, device, args) = item
        import json

        argv = [a if not isinstance(a, dict) else json.dumps(a) for a in args]
        try:
            return name, sphere_run.spawn(kind, cfg_calls(v), Path(root) / name, *argv, device=device)
        except Exception as e:
            return name, e

    with ThreadPoolExecutor(max_workers=4) as ex:
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


@pytest.mark.parametrize("variant", ["far-", "reflex-"])
def test_the_compiled_sphere_with_far_hearing_closes_the_ledger_and_hears(children, variant: str) -> None:
    """The far hearing (and the oracle) on the compiled sphere step in a cold child: the grid runs, the ledger
    closes, the host check passes, the far counter is non-zero, and the compiled step equals the eager one."""
    cfg, r = cfg_calls(variant), result(children, f"{variant}trajectory")
    assert r["problems"] == [] and r["diag"] == 0 and r["overflow"] == 0 and r["alive"] > 20
    tot = lambda name: int(r["counters_sum"][counter_index(cfg, name)])  # noqa: E731
    assert tot("CALLS_FAR_HEARD") > 0 and tot("CALLS_SPOKE_STORM") > 0
    assert result(children, f"{variant}eager-first")["mismatch"] == []
    if mx.metal.is_available():
        assert result(children, f"{variant}gpu-trajectory")["problems"] == []
        assert result(children, f"{variant}gpu-compare")["mismatch"] == []
