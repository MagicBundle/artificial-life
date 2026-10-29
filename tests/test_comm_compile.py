"""E4 (DESIGN F21 item 6, I50): every constant the voice channel puts into a compiled function.

MLX names a fused kernel after its op tape and the printed form of its scalar constants back to back, so two
configurations that differ only in a constant can share one kernel when the prints concatenate alike. For each
constant of the comm layer (``range``, ``atten_pow``, ``kin_exp``, ``cost``), configurations differing only
in it are compiled one after the other in ONE process on the CPU with a private kernel cache, and every
compiled result must equal its eager one (and differ between values that must differ). Values are chosen so
that pairs print alike.
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
    import dataclasses, hashlib, json, sys
    import mlx.core as mx
    import numpy as np
    mx.set_default_device(mx.cpu)
    from alife import comm, spatial
    from alife.config import dims, load_config

    base = load_config(preset="tiny", overrides=["comm.enabled=true", "brain.weight_dtype=float32"])
    d = dims(base)
    rs = np.random.default_rng(3)
    n, K, nf = 256, d.k, d.lay.NF
    idx = rs.integers(0, n, (n, K)).astype(np.int32)
    valid = rs.random((n, K)) < 0.8
    dist = rs.uniform(0.2, base.sensing.radius, (n, K)).astype(np.float32)
    desc = np.zeros((n, K, nf), np.float32)
    desc[..., 0] = valid
    desc[..., 1:3] = rs.uniform(-1, 1, (n, K, 2)) * valid[..., None]
    desc[..., 3] = rs.random((n, K)) * valid
    voice = (rs.random((n, d.voice_w)) * (rs.random((n, 1)) < 0.7)).astype(np.float32)
    z = mx.zeros((), mx.int32)
    nbr = spatial.Neighbors(mx.array(idx), mx.array(valid), mx.zeros((n, K, 2)), mx.array(dist), z, z)
    desc, voice = mx.array(desc), mx.array(voice)
    uid = mx.arange(n, dtype=mx.int32)
    step = mx.array(123, dtype=mx.int32)
    sig = mx.array(rs.random((n, d.voice_w)).astype(np.float32) * 0.2)

    def make(cfg):
        def f(v, ds, sg):
            return comm.hear(v, nbr, ds, cfg), comm.due_of(uid, step, sg, cfg).astype(mx.float32)
        return f

    keep, out, sigs = [], [], []
    for kv in json.loads(sys.argv[1]):
        cfg = dataclasses.replace(base, comm=dataclasses.replace(base.comm, **kv))
        dims.cache_clear()
        f = make(cfg)
        cf = mx.compile(f)
        keep.append(cf)
        got, want = cf(voice, desc, sig), f(voice, desc, sig)
        # the due is an integer (exact); the heard values agree to float rounding (a fused pow)
        ok = bool(mx.array_equal(got[1], want[1]).item()) and bool(
            np.allclose(np.asarray(got[0]), np.asarray(want[0]), rtol=1e-5, atol=1e-6)
        )
        out.append(ok)
        sigs.append([hashlib.sha1(np.round(np.asarray(g), 4).tobytes()).hexdigest() for g in got])
    print(json.dumps({"equal": out, "sig": sigs}))
    """
)
BASE = {"enabled": True, "width": 4, "range": 0.0, "atten_pow": 1.0, "kin_exp": 0.0, "cost": 4.0}
# value -> which output it must move: 0 = hear, 1 = the due
VALUES = {
    "range": ([0.0, 3.0, 4.0, 4.5, 5.0, 5.5, 0.3, 3.3], (0,)),
    "atten_pow": ([1.0, 2.0, 20.0, 0.5, 1.5, 3.0, 0.25], (0,)),
    "kin_exp": ([0.0, 1.0, 2.0, 0.5, 3.0, 12.0, 1.2], (0,)),
    "cost": ([4.0, 40.0, 0.4, 14.0, 0.04, 400.0, 4.4, 8.0, 16.0, 64.0], (1,)),
}


def _run(tmpdir: Path, cases: list[dict]) -> dict:
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
def test_e4_configs_differing_only_in_one_comm_constant_do_not_share_a_kernel(
    key: str, tmp_path: Path
) -> None:
    values, moves = VALUES[key]
    cases = [{**BASE, key: v} for v in values]
    res = _run(tmp_path, cases)
    assert res["equal"] == [True] * len(cases)
    for out in moves:
        sigs = [s[out] for s in res["sig"]]
        assert len(set(sigs)) == len(sigs), f"two values of {key} gave one result: a shared kernel"
