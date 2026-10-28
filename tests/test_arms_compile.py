"""E4 (DESIGN F21 item 6, I50): every constant the arms layer puts into a compiled function.

MLX names a fused kernel after its op tape and the printed form of its scalar constants back to back, so two
configurations that differ only in a constant can run one kernel when the prints concatenate alike. The arms
constants are ``kappa``, ``a0``, ``key_floor``, ``speed_slow``, ``upkeep_atk``, ``upkeep_arm`` (scalars of
the contest, the upkeep and the thrust factor) and the mutation sigmas (an array, ``sigma_vector``). For each,
configurations differing only in that constant are compiled one after the other in ONE process on the CPU
with a private kernel cache (the case where a collision bites, ``test_mlx_compile_collision``), and every
compiled result must equal its eager one. Values are chosen so that pairs print alike ("12" and "1" next to
"21", "2" and "12" ...).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPT = textwrap.dedent(
    """
    import dataclasses, json, sys
    import mlx.core as mx
    import numpy as np
    mx.set_default_device(mx.cpu)
    from alife import arms
    from alife.config import Config

    rs = np.random.default_rng(3)
    n = 512
    x = mx.array(rs.random((n, 6)).astype(np.float32))
    t0 = mx.array(rs.integers(0, n, n).astype(np.int32))
    uid, step = mx.array(np.arange(n, dtype=np.int64)), mx.array(5, mx.int32)
    z = mx.array(rs.standard_normal((n, 6)).astype(np.float32))

    def make(cfg):
        sig = arms.sigma_vector(cfg)
        def f(a, t, u, s, zz):
            p = arms.contest_factor(a, t, None, cfg)
            p = mx.ones((n,)) if p is None else p
            slow = 1.0 - cfg.arms.speed_slow * mx.clip(a[:, arms.ARM], 0.0, 1.0)
            return p, arms.upkeep(u, s, a, cfg), arms.child_traits(a, zz, sig), slow
        return f

    keep, out = [], []
    for kv in json.loads(sys.argv[1]):
        cfg = Config()
        cfg = dataclasses.replace(cfg, arms=dataclasses.replace(cfg.arms, enabled=True, **kv))
        f = make(cfg)
        cf = mx.compile(f)
        keep.append(cf)
        got, want = cf(x, t0, uid, step, z), f(x, t0, uid, step, z)
        out.append(all(bool(mx.array_equal(g, w).item()) for g, w in zip(got, want)))
    print(json.dumps(out))
    """
)
BASE = {"kappa": 2.0, "a0": 1.0, "key_floor": 0.2, "speed_slow": 0.3, "upkeep_atk": 20.0, "upkeep_arm": 20.0,
        "sigma": 0.03, "sigma_phase": 0.05}  # fmt: skip
VALUES = {
    "kappa": [1.0, 2.0, 12.0, 21.0, 0.5, 0.25],
    "a0": [1.0, 2.0, 12.0, 21.0, 0.5, 0.125],
    "key_floor": [0.0, 0.1, 0.12, 0.2, 0.21, 0.5, 0.9],
    "speed_slow": [0.0, 0.1, 0.12, 0.3, 0.31, 0.9],
    "upkeep_atk": [1.0, 2.0, 12.0, 21.0, 0.5, 20.0],
    "upkeep_arm": [1.0, 2.0, 12.0, 21.0, 0.5, 20.0],
    "sigma": [0.0, 0.01, 0.012, 0.03, 0.5],
    "sigma_phase": [0.0, 0.01, 0.012, 0.05, 0.5],
}


def _run(tmpdir: Path, cases: list[dict]) -> list[bool]:
    tmpdir.mkdir(parents=True, exist_ok=True)  # MLX needs TMPDIR to exist
    env = {**os.environ, "TMPDIR": f"{str(tmpdir).rstrip(os.sep)}{os.sep}"}
    done = subprocess.run(
        [sys.executable, "-c", SCRIPT, json.dumps(cases)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@pytest.mark.parametrize("key", sorted(VALUES))
def test_e4_configs_differing_only_in_one_arms_constant_do_not_share_a_kernel(
    key: str, tmp_path: Path
) -> None:
    cases = [{**BASE, key: v} for v in VALUES[key]]
    assert _run(tmp_path, cases) == [True] * len(cases)


def test_e4_the_grid_of_alike_prints(tmp_path: Path) -> None:
    """kappa and a0 together: (12, 1) next to (1, 21) and (2, 1) next to (21, 2) print alike."""
    cases = [{**BASE, "kappa": k, "a0": a} for k, a in ((12.0, 1.0), (1.0, 21.0), (2.0, 1.0), (21.0, 2.0))]
    # the validated range is kappa/(kappa + a0) <= 0.9, but the compiled function does not care
    assert _run(tmp_path, cases) == [True] * 4
