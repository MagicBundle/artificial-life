"""The trajectory of the compiled sphere step (DESIGN §46; sphere series S2): the exact ledger and every
invariant
over 300 steps, the pinned digest, and the slow gates of the spec (3 000 steps closing the ledger, 20 000
steps
without a hash overflow).

A compiled sphere configuration runs in a child process with an empty kernel cache of its own (spec 10.1
fallback, ``tests/sphere_run.py``): MLX 0.32.3 can hand a compiled function another one's kernel when their
constants print alike, and the guard that fixes it (``fix/mlx-compile-collision``) is not merged. On the CPU a
cold child spends two minutes in the dynamic loader, so the compiled tests have files of their own (``pytest
-n --dist loadfile`` runs them side by side).
"""

from __future__ import annotations

import platform

import mlx.core as mx
import pytest

import sphere_run
from alife.config import Config, load_config
from alife.state import LedgerIdx

# The digest of ``tiny3s`` (float32 weights) after 50 compiled steps, over every field (``state_digest``), per
# platform: the value depends on the MLX version, the machine's float arithmetic and the compiler of the CPU
# kernels (DESIGN §22.0), so it is only checked where it was recorded. An empty entry is not pinned yet: the
# test skips and prints the digest to record. CPU and Metal digests differ from step 1 on (the brain's float
# arithmetic differs by 1e-7 between the devices; the box's tiny3d does the same), so each device has its own.
PIN_STEPS = 50
GOLDEN_SPHERE = {
    "mlx": "0.32.3",
    "digests": {
        "Darwin-arm64-cpu": "ae81d27fb6fbd741b00223be82de4110418bcde605fec953e97c9d07e9b0d0a0",
        "Darwin-arm64-gpu": "0aa7ab53502d5376f2c4a9b2e0c776c2d6a9aac08804fc1ff48a93846b0aa4e3",
        "Linux-x86_64-cpu": "",
    },
}


def tiny3s() -> Config:
    return load_config(preset="tiny3s", overrides=["brain.weight_dtype=float32"])


def platform_key() -> str:
    dev = "gpu" if mx.default_device().type == mx.gpu else "cpu"
    return f"{platform.system()}-{platform.machine()}-{dev}"


@pytest.fixture(scope="module")
def trajectory(tmp_path_factory) -> dict:
    """300 compiled steps of ``tiny3s`` in a cold child: the full host check against the previous state and
    the step's records every 50 steps, the digest at the pin step."""
    return sphere_run.spawn(
        "trajectory", tiny3s(), tmp_path_factory.mktemp("traj") / "cold", 300, 50, str(PIN_STEPS)
    )


def test_the_ledger_closes_and_every_invariant_holds_over_the_trajectory(trajectory) -> None:
    t = trajectory
    assert t["problems"] == [], t["problems"][
        :5
    ]  # full_check incl. the per-step ledger closure, bonds, ranges
    assert t["diag"] == 0  # the compiled checks never fired
    assert t["overflow"] == 0 and t["max_occupancy"] < 32  # no hash cell over capacity
    ledger = t["ledger"]
    sources = ledger[LedgerIdx.INJECT_REGEN] + ledger[LedgerIdx.INJECT_IMMIGRATION]
    sinks = (
        sum(ledger[i] for i in range(LedgerIdx.HEAT_BASAL, LedgerIdx.HEAT_OVERFLOW + 1))
        + ledger[LedgerIdx.HEAT_BUOYANCY]
    )
    assert (
        sources > 0 and sinks > 0 and ledger[LedgerIdx.HEAT_BUOYANCY] > 0
    )  # the 16-term ledger is exercised
    assert t["alive"] > 100


def test_the_digest_of_tiny3s_is_pinned(trajectory) -> None:
    got = trajectory["digests"][str(PIN_STEPS)]
    if mx.__version__ != GOLDEN_SPHERE["mlx"] or platform_key() not in GOLDEN_SPHERE["digests"]:
        pytest.skip(
            f"sphere digest pinned for MLX {GOLDEN_SPHERE['mlx']} on {sorted(GOLDEN_SPHERE['digests'])}"
        )
    want = GOLDEN_SPHERE["digests"][platform_key()]
    if not want:
        pytest.skip(f"sphere digest not pinned yet for {platform_key()}: record {got} in GOLDEN_SPHERE")
    assert got == want


@pytest.mark.slow
def test_three_thousand_steps_close_the_ledger_and_the_hash_never_overflows(tmp_path) -> None:
    """The spec's S2 gate on the tiny world: 3 000 compiled steps, the host check every 250 steps."""
    r = sphere_run.spawn("trajectory", tiny3s(), tmp_path / "cold", 3000, 250, "", timeout=6000)
    assert r["problems"] == [] and r["diag"] == 0 and r["overflow"] == 0 and r["alive"] > 100


@pytest.mark.slow
def test_twenty_thousand_steps_never_overflow_a_hash_cell(tmp_path) -> None:
    """The S2 gate: no hash overflow in 20 000 steps (on a Mac with the kernel this takes a minute)."""
    r = sphere_run.spawn("trajectory", tiny3s(), tmp_path / "cold", 20000, 2000, "", timeout=20000)
    assert r["problems"] == [] and r["diag"] == 0 and r["overflow"] == 0
    assert r["max_occupancy"] < 32
