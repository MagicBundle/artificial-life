"""E4 (DESIGN F21 item 6, I50): every constant the conductive share puts into a compiled function.

MLX names a fused kernel after its op tape and the printed form of its scalar constants back to back, so two
configurations that differ only in a constant can share one kernel when the prints concatenate alike. For each
constant of the bonds layer, configurations differing only in it are compiled one after the other in ONE
process on the CPU with a private kernel cache, and every compiled result must equal its eager one (and
differ between values that must differ). Values are chosen so that pairs print alike.
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
    from types import SimpleNamespace
    import mlx.core as mx
    import numpy as np
    mx.set_default_device(mx.cpu)
    from alife import bonds, energy as en
    from alife.config import load_config

    base = load_config(preset="tiny", overrides=["bonds.share_mode=1", "roles.enabled=true"])
    rs = np.random.default_rng(3)
    n, B = 256, 4
    partner = np.full((n, B), -1, np.int32)
    rev = np.full((n, B), -1, np.int32)
    for _ in range(300):
        i, j = rs.choice(n, 2, replace=False)
        if j in partner[i] or not (partner[i] < 0).any() or not (partner[j] < 0).any():
            continue
        si, sj = int(np.argmax(partner[i] < 0)), int(np.argmax(partner[j] < 0))
        partner[i, si], rev[i, si], partner[j, sj], rev[j, sj] = j, sj, i, si
    E = mx.array(rs.integers(0, 12_000, n).astype(np.int32))
    share = mx.array(rs.random((n, B)).astype(np.float32))
    kin = mx.array(rs.random((n, B)).astype(np.float32))
    tag = mx.array(rs.normal(0, 0.4, (n, 4)).astype(np.float32))
    solvent = mx.array(rs.random(n) < 0.9)
    partner, rev = mx.array(partner), mx.array(rev)
    bf = mx.array(rs.integers(-2000, 2000, (n, B)).astype(np.int16))

    def make(cfg):
        def f(E_, share_, kin_, tag_, bf_):
            act = SimpleNamespace(share=share_)
            sl = en.share_slots(E_, partner, rev, partner, act, solvent, cfg, kin_)
            k = bonds.slot_kin(tag_, partner, cfg)
            flow_in = mx.clip(bf_.astype(mx.float32) / cfg.bonds.share_rate, -1.0, 1.0)
            return sl.out, sl.inc, sl.got, k, flow_in
        return f

    keep, out, sigs = [], [], []
    for kv in json.loads(sys.argv[1]):
        cfg = dataclasses.replace(base, bonds=dataclasses.replace(base.bonds, **kv))
        f = make(cfg)
        cf = mx.compile(f)
        keep.append(cf)
        got, want = cf(E, share, kin, tag, bf), f(E, share, kin, tag, bf)
        out.append(all(bool(mx.array_equal(g, w).item()) for g, w in zip(got, want)))
        sigs.append([hashlib.sha1(np.asarray(g).tobytes()).hexdigest() for g in got[:3]])
    print(json.dumps({"equal": out, "sig": sigs}))
    """
)
BASE = {
    "share_mode": 1,
    "share_kappa": 0.125,
    "share_kin_floor": 0.5,
    "share_rate": 1000,
    "share_efficiency": 0.9,
}
VALUES = {
    "share_kappa": [0.125, 0.0625, 0.012, 0.02, 0.05, 0.1, 0.09],
    "share_kin_floor": [0.0, 0.1, 0.12, 0.21, 0.5, 0.99],
    "share_rate": [1000, 100, 10, 1010, 101, 12, 21, 4000],
    "share_efficiency": [0.9, 0.09, 0.19, 0.91, 0.5, 1.0],
    "share_mode": [1, 2],
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
def test_e4_configs_differing_only_in_one_bonds_constant_do_not_share_a_kernel(
    key: str, tmp_path: Path
) -> None:
    cases = [{**BASE, key: v} for v in VALUES[key]]
    res = _run(tmp_path, cases)
    assert res["equal"] == [True] * len(cases)
    if key != "share_efficiency":  # the efficiency only moves `got`; the others change `out` or `inc`
        sigs = [tuple(s) for s in res["sig"]]
        assert len(set(sigs)) == len(sigs), "two values of the constant gave one result: a shared kernel"
