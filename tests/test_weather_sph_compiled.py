"""The compiled sphere weather: the engine step with the weather on and the weather alone, compiled against
eager in both orders, the 300-step trajectory (ledger, invariants) and the pinned digest. One configuration
per
child process with an empty kernel cache (spec 10.1 fallback; ``tests/sphere_run.py``): the compiled files
have
their own module so that ``pytest -n --dist loadfile`` runs them side by side."""

from __future__ import annotations

import pytest

import sphere_run
from alife.state import LedgerIdx
from sphere_weather_helpers import weather_cfg
from test_sphere_trajectory import platform_key

PIN_STEPS = 50
# The digest of ``tiny3s`` with the weather on (float32 weights, seed_rate 0.2, seed_gain 0.03) after 50
# compiled
# steps, per platform (as ``test_sphere_trajectory``: only checked where recorded).
GOLDEN_WEATHER = {
    "Darwin-arm64-cpu": "149953e3e5205e53b4533f69b3a5c5046370a235df08e3e61dd58c3defd5ed90",
    "Darwin-arm64-gpu": "25a58e78af41171084370a18e9f622c35ebbcfcf22e4d4c8cb94b7e03db629a8",
    "Linux-x86_64-cpu": "",
}


def cfg_w():
    return weather_cfg(
        "tiny3s", "brain.weight_dtype=float32", "weather.seed_rate=0.2", "weather.seed_gain=0.03"
    )


@pytest.fixture(scope="module")
def trajectory(tmp_path_factory) -> dict:
    return sphere_run.spawn(
        "trajectory", cfg_w(), tmp_path_factory.mktemp("wtraj") / "cold", 300, 50, str(PIN_STEPS)
    )


def test_the_ledger_closes_and_every_invariant_holds_over_the_weather_trajectory(trajectory) -> None:
    t = trajectory
    assert t["problems"] == [] and t["overflow"] == 0 and t["alive"] > 50
    assert t["ledger"][LedgerIdx.INJECT_REGEN] > 0


def test_the_digest_of_tiny3s_with_the_weather_is_pinned(trajectory) -> None:
    got = trajectory["digests"][str(PIN_STEPS)]
    want = GOLDEN_WEATHER.get(platform_key())
    if not want:
        pytest.skip(f"weather digest not pinned yet for {platform_key()}: record {got}")
    assert got == want


@pytest.mark.parametrize("order", ["eager-first", "compiled-first"])
def test_the_compiled_engine_step_equals_the_eager_one_with_the_weather(order: str, tmp_path) -> None:
    r = sphere_run.spawn("compare", cfg_w(), tmp_path / "cold", 3, order)
    assert r["mismatch"] == [] and r["steps"] == 3


@pytest.mark.parametrize("order", ["eager-first", "compiled-first"])
def test_the_compiled_weather_alone_equals_the_eager_one_bit_for_bit(order: str, tmp_path) -> None:
    r = sphere_run.spawn("wx_compare", cfg_w(), tmp_path / "cold", 80, order)
    assert r["mismatch"] == [] and r["steps"] == 80 and r["peak_wind"] > 0


def test_the_compiled_lookups_of_the_weather_nside_are_safe(tmp_path) -> None:
    assert sphere_run.spawn("lookups", cfg_w(), tmp_path / "cold") == {"ok": True}
