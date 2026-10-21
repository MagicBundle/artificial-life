"""Determinism of the compiled 3D step (§14.2, §22.4, §27.8).

Same seed → identical state digests; different seeds → different digests; ``runtime_checks`` only touches
``diag``; compiled and eager steps agree; and the 3D golden digest of ``tiny3d_cfg`` after 50 steps
(``golden``), over ``legacy_state_digest_3d``: the 49 fields ``State3`` had before ``wx`` (DESIGN §30), so
adding the trajectory-neutral weather field did not move the pin and ``GOLDEN_3D`` was not re-pinned.
"""

from __future__ import annotations

import platform

import mlx.core as mx
import numpy as np
import pytest

from alife.config import Config, apply_overrides
from alife.state import (
    LedgerIdx,
    State3,
    init_state,
    legacy_state_digest_3d,
    n_counters,
    state_digest,
    to_numpy,
)
from alife.step import make_step

STEPS = 50

# The 3D digest of ``tiny3d_cfg`` after 50 compiled steps, buoyancy on (so ``HEAT_BUOYANCY`` is in the pin).
# It is only meaningful for the MLX version and platform it was recorded on; anywhere else the test skips.
# The value is over ``legacy_state_digest_3d`` (the 49 pre-weather fields), which for an unchanged trajectory
# is the ``state_digest`` it was recorded with: it was not re-pinned when ``wx`` was added.
# An empty digest is not pinned yet: on the pinned platform the test skips and prints the digest to record.
GOLDEN_3D = {
    "mlx": "0.32.3",
    "platform": "Linux-x86_64-cpu",
    "digest": "ffb171300f5377b0397b24c2783d177365ff4006c30c7c291ac8aeb92148b3a1",
}


def _platform() -> str:
    dev = mx.default_device()
    return f"{platform.system()}-{platform.machine()}-{'gpu' if dev.type == mx.gpu else 'cpu'}"


def _run(cfg: Config, steps: int, *, compile: bool = True) -> tuple[State3, np.ndarray, np.ndarray]:
    """``(final state, counters (steps,30), ledgers (steps,16))`` from ``init_state(cfg)``."""
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
def base_run(tiny3d_cfg: Config) -> tuple[State3, np.ndarray, np.ndarray]:
    return _run(tiny3d_cfg, STEPS)


def test_same_seed_is_identical(tiny3d_cfg: Config, base_run) -> None:
    s1, c1, l1 = base_run
    s2, c2, l2 = _run(tiny3d_cfg, STEPS)  # a fresh compile of the same config
    assert isinstance(s1, State3) and state_digest(s1) == state_digest(s2)
    assert np.array_equal(c1, c2) and np.array_equal(l1, l2)
    assert c1.shape == (STEPS, n_counters(3)) and int(to_numpy(s1.step)) == STEPS
    assert int(to_numpy(s1.ledger_cum)[LedgerIdx.HEAT_BUOYANCY]) > 0  # the golden pin is not vacuous


def test_different_seeds_differ(tiny3d_cfg: Config, base_run) -> None:
    s1, c1, _ = base_run
    s2, c2, _ = _run(apply_overrides(tiny3d_cfg, ["run.seed=1"]), STEPS)
    assert state_digest(s1) != state_digest(s2)
    assert not np.array_equal(c1, c2)


def test_runtime_checks_only_touch_diag(tiny3d_cfg: Config, base_run) -> None:
    s1, c1, l1 = base_run
    s2, c2, l2 = _run(apply_overrides(tiny3d_cfg, ["run.runtime_checks=off"]), STEPS)
    assert int(to_numpy(s2.diag)) == 0
    for name in State3._fields:
        if name != "diag":
            assert np.array_equal(to_numpy(getattr(s1, name)), to_numpy(getattr(s2, name))), name
    assert np.array_equal(c1, c2) and np.array_equal(l1, l2)


def test_eager_step_matches_compiled(tiny3d_cfg: Config) -> None:
    """One eager step agrees with the compiled step on every integer field, counter and ledger entry."""
    s = init_state(tiny3d_cfg)
    a, out_a = make_step(tiny3d_cfg, compile=False)(s)
    b, out_b = make_step(tiny3d_cfg, compile=True)(s)
    mx.eval(a, out_a, b, out_b)
    assert np.array_equal(to_numpy(out_a.counters), to_numpy(out_b.counters))
    assert np.array_equal(to_numpy(out_a.ledger), to_numpy(out_b.ledger))
    for name in State3._fields:
        x, y = to_numpy(getattr(a, name)), to_numpy(getattr(b, name))
        if x.dtype.kind == "f":
            np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-5, err_msg=name)
        else:
            assert np.array_equal(x, y), name


@pytest.mark.golden
def test_golden_3d(base_run) -> None:
    if mx.__version__ != GOLDEN_3D["mlx"] or _platform() != GOLDEN_3D["platform"]:
        pytest.skip(f"3D golden digest pinned for MLX {GOLDEN_3D['mlx']} on {GOLDEN_3D['platform']}")
    s = base_run[0]
    assert int(to_numpy(s.ledger_cum)[LedgerIdx.HEAT_BUOYANCY]) > 0
    digest = legacy_state_digest_3d(s)
    if not GOLDEN_3D["digest"]:
        pytest.skip(f"3D golden digest not pinned yet: record {digest} in GOLDEN_3D (DESIGN §27.8)")
    assert digest == GOLDEN_3D["digest"]
