"""The compiled sphere step with the features on (DESIGN §46.5; sphere series S3).

A compiled sphere configuration runs in a child process with an empty kernel cache of its own (spec 10.1
fallback, ``tests/sphere_run.py``): MLX 0.32.3 can hand a compiled function another one's kernel when their
constants print alike, and the guard that fixes it (``fix/mlx-compile-collision``) is not merged. The children
run side by side (a cold CPU child spends about two minutes compiling), so this file costs one of them in wall
time; it has a file of its own for ``pytest -n --dist loadfile``.

* ``structures``: 120 forced-build steps with every effect (cover, barrier with the easy exit, shield) and the
  aperture rule: the full host check and the ledger, no cell lost its aperture, nobody trapped, walls formed;
* ``all``: the five features and the roles together for 100 steps (the combination the box suites call
  ``coexists``): the same checks and every feature's counter non-zero;
* ``compare``: the compiled step equals the eager one from the same states in both orders, with the features
  forced on (integers exactly, floats to 1e-5);
* on Metal: a short ``all`` trajectory and the compiled-against-eager comparison on the GPU.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import sphere_feature_suite as su
import sphere_run
from alife.config import Config
from alife.state import LedgerIdx, counter_index

INIT = {
    "structures": {"bias": {"OUT_BUILD": 4.0}},
    "all": {"bias": {"OUT_BUILD": 4.0, "OUT_BOND": 4.0, "OUT_CHILD_BOND": 4.0}},
}
STRUCT_EXTRA = (
    "structures.reach=0.5",
    "structures.shield_gain=0.5",
    "structures.barrier_gain=3.0",
    "structures.exit_ease=0.75",
    "structures.aperture=0.4",
    "combat.protect_floor=0.1",
)


def cfg_struct() -> Config:
    return su.make_cfg("structures", *STRUCT_EXTRA, compile=True)


def cfg_all() -> Config:
    return su.make_cfg("all", compile=True)


@pytest.fixture(scope="module")
def children(tmp_path_factory) -> dict:
    """Run the children at once; a failure is re-raised by the test that reads it."""
    root = tmp_path_factory.mktemp("cold")
    jobs = {
        "structures": ("trajectory", cfg_struct(), None, (120, 20, "", INIT["structures"])),
        "all": ("trajectory", cfg_all(), None, (100, 25, "", INIT["all"])),
        "eager-first": ("compare", cfg_all(), None, (2, "eager-first", INIT["all"])),
        "compiled-first": ("compare", cfg_all(), None, (2, "compiled-first", INIT["all"])),
    }
    if mx.metal.is_available():
        jobs["gpu-all"] = ("trajectory", cfg_all(), "gpu", (60, 20, "", INIT["all"]))
        jobs["gpu-compare"] = ("compare", cfg_all(), "gpu", (2, "eager-first", INIT["all"]))

    def go(item):
        name, (kind, cfg, device, args) = item
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


def total(cfg: Config, r: dict, name: str, key: str = "counters_sum") -> int:
    return int(r[key][counter_index(cfg, name)])


def test_structures_compiled_close_the_ledger_and_trap_nobody(children) -> None:
    cfg, r = cfg_struct(), result(children, "structures")
    assert r["problems"] == [] and r["diag"] == 0
    assert r["overflow"] == 0
    assert total(cfg, r, "BUILD_UNITS") > 0
    assert total(cfg, r, "TRAPPED") == 0 and total(cfg, r, "APERTURE_VIOLATIONS") == 0
    assert total(cfg, r, "MAT_CLOSED", "counters_max") > 0  # walls formed
    assert total(cfg, r, "MAT_TOTAL", "counters_last") > 0
    led = r["ledger"]
    assert led[LedgerIdx.HEAT_ACTION] > 0  # the build cost is booked as action heat
    assert led[LedgerIdx.INJECT_REGEN] > 0 and led[LedgerIdx.HEAT_BASAL] > 0


def test_all_features_compiled_close_the_ledger_and_every_feature_acts(children) -> None:
    cfg, r = cfg_all(), result(children, "all")
    assert r["problems"] == [] and r["diag"] == 0 and r["overflow"] == 0
    # (the arms' contest needs attacks on unbonded neighbours, which forced bonders rarely make: the arms
    # counters are checked in the eager runs; here the arms state is covered by the host check above)
    for name in ("BUILD_UNITS", "FARM_YIELD", "COMM_SPEAKERS", "SHARE_SLOTS_ACTIVE"):
        assert total(cfg, r, name) > 0, name
    assert total(cfg, r, "TRAPPED") == 0 and total(cfg, r, "APERTURE_VIOLATIONS") == 0
    assert r["ledger"][LedgerIdx.FLOW_SHARED] > 0 and r["alive"] > 50


@pytest.mark.parametrize("order", ["eager-first", "compiled-first"])
def test_the_compiled_step_equals_the_eager_step_with_the_features_on(children, order: str) -> None:
    r = result(children, order)
    assert r["mismatch"] == [] and r["steps"] == 2


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs the Metal GPU")
def test_the_features_on_the_gpu_close_the_ledger_and_match_the_eager_step(children) -> None:
    cfg, r = cfg_all(), result(children, "gpu-all")
    assert r["problems"] == [] and r["diag"] == 0 and r["overflow"] == 0
    assert total(cfg, r, "BUILD_UNITS") > 0 and total(cfg, r, "SHARE_SLOTS_ACTIVE") > 0
    c = result(children, "gpu-compare")
    assert c["mismatch"] == [] and np.isfinite(r["alive"])
