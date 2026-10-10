"""Determinism of the compiled step (§14.2, §22.4). Owned by WP4.

Same seed → identical state digests; different seeds → different digests; ``runtime_checks`` only touches
``diag``; compiled and eager steps agree; and the pinned ``tiny_cfg`` digest after 50 steps (``golden``).
"""

from __future__ import annotations

import platform

import mlx.core as mx
import numpy as np
import pytest

from alife.config import Config, apply_overrides
from alife.state import State, init_state, state_digest, to_numpy
from alife.step import make_step

STEPS = 50
EAGER_STEPS = 30

# Pinned digest of ``tiny_cfg`` after 50 compiled steps. It is only meaningful for the MLX version and
# platform it was recorded on; anywhere else the test skips, so an MLX upgrade is a deliberate re-pin.
GOLDEN = {
    "mlx": "0.32.3",
    "platform": "Linux-x86_64-cpu",
    "digest": "0f321a0e7343821f08dcc812e443b3dec9b89eda68264a9b44fbe53e92637aa8",
}


def _platform() -> str:
    dev = mx.default_device()
    return f"{platform.system()}-{platform.machine()}-{'gpu' if dev.type == mx.gpu else 'cpu'}"


def _run(cfg: Config, steps: int, *, compile: bool = True) -> tuple[State, np.ndarray, np.ndarray]:
    """``(final state, counters (steps,26), ledgers (steps,16))`` from ``init_state(cfg)``."""
    fn = make_step(cfg, compile=compile)
    s = init_state(cfg)
    counters, ledgers = [], []
    for _ in range(steps):
        s, out = fn(s)
        mx.eval(s, out)
        counters.append(to_numpy(out.counters))
        ledgers.append(to_numpy(out.ledger))
    return s, np.stack(counters), np.stack(ledgers)


@pytest.fixture(scope="module")
def base_run(tiny_cfg: Config) -> tuple[State, np.ndarray, np.ndarray]:
    return _run(tiny_cfg, STEPS)


def test_same_seed_is_identical(tiny_cfg: Config, base_run) -> None:
    s1, c1, l1 = base_run
    s2, c2, l2 = _run(tiny_cfg, STEPS)  # a fresh compile of the same config
    assert state_digest(s1) == state_digest(s2)
    assert np.array_equal(c1, c2) and np.array_equal(l1, l2)
    assert int(to_numpy(s1.step)) == STEPS


def test_different_seeds_differ(tiny_cfg: Config, base_run) -> None:
    s1, c1, _ = base_run
    s2, c2, _ = _run(apply_overrides(tiny_cfg, ["run.seed=1"]), STEPS)
    assert state_digest(s1) != state_digest(s2)
    assert not np.array_equal(c1, c2)


def test_runtime_checks_only_touch_diag(tiny_cfg: Config, base_run) -> None:
    """``runtime_checks = off`` changes nothing but ``diag`` (the checks are side-effect free)."""
    s1, c1, l1 = base_run
    s2, c2, l2 = _run(apply_overrides(tiny_cfg, ["run.runtime_checks=off"]), STEPS)
    assert int(to_numpy(s2.diag)) == 0
    for name in State._fields:
        if name != "diag":
            assert np.array_equal(to_numpy(getattr(s1, name)), to_numpy(getattr(s2, name))), name
    assert np.array_equal(c1, c2) and np.array_equal(l1, l2)


def test_eager_step_matches_compiled(tiny_cfg: Config) -> None:
    """One eager step agrees with the compiled step on every integer field and counter."""
    s = init_state(tiny_cfg)
    a, out_a = make_step(tiny_cfg, compile=False)(s)
    b, out_b = make_step(tiny_cfg, compile=True)(s)
    mx.eval(a, out_a, b, out_b)
    assert np.array_equal(to_numpy(out_a.counters), to_numpy(out_b.counters))
    assert np.array_equal(to_numpy(out_a.ledger), to_numpy(out_b.ledger))
    for name in State._fields:
        x, y = to_numpy(getattr(a, name)), to_numpy(getattr(b, name))
        if x.dtype.kind == "f":
            np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-5, err_msg=name)
        else:
            assert np.array_equal(x, y), name


@pytest.mark.slow
def test_compiled_vs_eager_trajectories(tiny_cfg: Config) -> None:
    """Population and ledger trajectories match over 30 steps (bitwise equality is not promised, §14.2)."""
    _, c1, l1 = _run(tiny_cfg, EAGER_STEPS, compile=True)
    _, c2, l2 = _run(tiny_cfg, EAGER_STEPS, compile=False)
    assert np.array_equal(c1, c2) and np.array_equal(l1, l2)


@pytest.mark.golden
def test_golden(base_run) -> None:
    if mx.__version__ != GOLDEN["mlx"] or _platform() != GOLDEN["platform"]:
        pytest.skip(f"golden digest pinned for MLX {GOLDEN['mlx']} on {GOLDEN['platform']}")
    assert state_digest(base_run[0]) == GOLDEN["digest"]
