"""E4 (DESIGN F21 item 6, §47.7): every constant the calls layer puts into a compiled function.

MLX names a fused kernel after its op tape and the printed form of its scalar constants back to back, and its
CPU backend writes the constants into the kernel source; two configurations that differ only in a constant can
share one kernel when the prints concatenate alike. The scalars of the window update that vary with the config
(``hold + 1``, ``food_q``, ``storm_frac * storm_u0``) are therefore passed as constant arrays of two elements
(``calls.runtime``), which MLX binds as kernel inputs. Here, in ONE process on the CPU with a private kernel
cache, configurations that differ only in one of them are compiled one after the other: each compiled result
must equal its eager one and differ between values that must differ, and no kernel source may contain any of
the values as a printed constant. A positive control, the same function with the scalars baked in, does print
them (so the scan can fail).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

SCRIPT = textwrap.dedent(
    """
    import dataclasses, hashlib, json, re, sys, pathlib
    from types import SimpleNamespace
    import mlx.core as mx
    import numpy as np
    mx.set_default_device(mx.cpu)
    from alife import calls
    from alife.config import load_config

    mode = sys.argv[2]
    base = load_config(preset="tiny", overrides=[
        "weather.enabled=true", "world.size=320", "weather.u_max=0.2", "brain.weight_dtype=float32",
        "comm.enabled=true", "comm.width=4", "calls.enabled=true", "calls.trigger=true"])
    rs = np.random.default_rng(3)
    n = 256
    loss = mx.array(rs.integers(0, 3, n).astype(np.int32))
    harvest = mx.array(rs.integers(0, 6000, n).astype(np.int32))
    speed = mx.array((rs.random(n) * 0.14).astype(np.float32))
    p0 = np.full((n, 4), -1, np.int32); p0[:, 0] = np.where(rs.random(n) < 0.5, 3, -1)
    p1 = p0.copy(); p1[:, 0] = np.where(rs.random(n) < 0.5, p0[:, 0], -1)
    p0, p1 = mx.array(p0), mx.array(p1)
    sv = mx.array(rs.random(n) < 0.9)
    evt = mx.array(rs.integers(0, 70, (n, 4)).astype(np.uint8))

    def make(cfg):
        def f(loss_, harvest_, speed_, p0_, p1_, sv_, evt_):
            s = SimpleNamespace(partner=p0_)
            s1 = SimpleNamespace(last_loss=loss_, last_harvest=harvest_, partner=p1_)
            wind = SimpleNamespace(speed=speed_)
            if mode == "runtime":
                rt = calls.runtime(cfg)
                lv = calls.levels(s, s1, sv_, wind, cfg, rt)
                return calls.window_update(evt_, lv, sv_, rt)
            # the control: the same law with every scalar baked in as a Python number
            c = cfg.calls
            thr = np.float32(c.storm_frac * cfg.weather.storm_u0)
            lv = mx.stack([s1.last_loss > 0, s1.last_harvest >= c.food_q, wind.speed >= float(thr),
                           ((s.partner >= 0) & (s1.partner != s.partner)).any(axis=1)], axis=1) & sv_[:, None]
            left = mx.where(evt_ > 0, evt_ - 1, 0)
            return mx.where(sv_[:, None], mx.where(lv, c.hold + 1, left), 0).astype(mx.uint8)
        return f

    keep, ok, sigs = [], [], []
    for kv in json.loads(sys.argv[1]):
        c = dataclasses.replace(base.calls, **{k: v for k, v in kv.items() if k != "storm_u0"})
        wc = dataclasses.replace(base.weather, storm_u0=kv.get("storm_u0", base.weather.storm_u0))
        cfg = dataclasses.replace(base, calls=c, weather=wc)
        f = make(cfg)
        cf = mx.compile(f)
        keep.append(cf)
        args = (loss, harvest, speed, p0, p1, sv, evt)
        got, want = cf(*args), f(*args)
        ok.append(bool(mx.array_equal(got, want).item()))
        sigs.append(hashlib.sha1(np.asarray(got).tobytes()).hexdigest())
    consts = set()
    for p in pathlib.Path(sys.argv[3]).rglob("*.cpp"):
        for m in re.finditer(r"tmp_\\w+ = (-?[0-9][-+0-9.eE]*);", p.read_text()):
            consts.add(m.group(1))
    print(json.dumps({"equal": ok, "sig": sigs, "consts": sorted(consts)}))
    """
)
# value -> the kernel constant it would print (None: compare as a float)
CASES = {
    "hold": [36, 8, 63, 17, 0, 64, 4],
    "food_q": [4321, 777, 1234, 55, 300, 200],
    "storm_frac": [0.7, 1.3, 0.55, 1.9, 1.0],
}
WIDE = {"hold": [36, 8, 63, 17], "food_q": [4321, 777, 1234, 55], "storm_frac": [0.7, 1.3, 0.55, 1.9]}


def _run(tmpdir: Path, cases: list[dict], mode: str = "runtime") -> dict:
    tmpdir.mkdir(parents=True, exist_ok=True)  # MLX needs TMPDIR to exist
    env = {**os.environ, "TMPDIR": f"{str(tmpdir).rstrip(os.sep)}{os.sep}"}
    done = subprocess.run(
        [sys.executable, "-c", SCRIPT, json.dumps(cases), mode, str(tmpdir)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _same(printed: float, value: float) -> bool:
    """A kernel prints a float32 constant with about ten significant digits (``0.0623699985``), so compare
    after rounding both to float32, with a relative tolerance of one part in 10^6."""
    import numpy as np

    a, b = float(np.float32(printed)), float(np.float32(value))
    return abs(a - b) <= 1e-6 * max(abs(b), 1e-30)


def _detected(consts: list[str], values: list[float]) -> list[bool]:
    """For each value, whether some printed constant of the kernels is that value."""
    printed = [float(c) for c in consts]
    return [any(_same(p, v) for p in printed) for v in values]


@pytest.mark.parametrize("key", sorted(CASES))
def test_e4_configs_differing_only_in_one_calls_constant_do_not_share_a_kernel(
    key: str, tmp_path: Path
) -> None:
    cases = [{key: v} for v in CASES[key]]
    res = _run(tmp_path, cases)
    assert res["equal"] == [True] * len(cases)
    # every distinct value moves the result (food_q and storm_frac move the levels, hold the window length)
    distinct = [s for v, s in zip(CASES[key], res["sig"], strict=True) if v in WIDE[key]]
    assert len(set(distinct)) == len(distinct), f"two values of {key} gave one result: a shared kernel"


def test_no_config_dependent_scalar_is_printed_into_a_kernel(tmp_path: Path) -> None:
    import numpy as np

    cases = [
        {"hold": 36, "food_q": 4321, "storm_frac": 0.7, "storm_u0": 0.0891},
        {"hold": 63, "food_q": 1234, "storm_frac": 1.3, "storm_u0": 0.0623},
        {"hold": 17, "food_q": 777, "storm_frac": 0.55, "storm_u0": 0.0777},
    ]
    classes = {
        "hold + 1": [float(c["hold"] + 1) for c in cases],
        "food_q": [float(c["food_q"]) for c in cases],
        "storm threshold": [float(np.float32(c["storm_frac"] * c["storm_u0"])) for c in cases],
    }
    res = _run(tmp_path / "runtime", cases)
    assert res["equal"] == [True, True, True]
    for name, values in classes.items():
        assert not any(_detected(res["consts"], values)), f"{name} is printed into a kernel: {res['consts']}"
    # the control: with the scalars baked in, the kernels print every value of every class, so the scan
    # (integers and float32 alike) can fail for each of them
    base = _run(tmp_path / "baked", cases, mode="baked")
    assert base["equal"] == [True, True, True]
    for name, values in classes.items():
        assert all(_detected(base["consts"], values)), f"the control does not show {name}: {base['consts']}"
    assert _same(0.0623699985, 0.062369998544454575) and not _same(0.0623, 0.0624)


# --- step K2: the constants of the far hearing --------------------------------------------------------------

FAR_SCRIPT = textwrap.dedent(
    """
    import dataclasses, hashlib, json, re, sys, pathlib
    import mlx.core as mx
    import numpy as np
    mx.set_default_device(mx.cpu)
    from alife import calls
    from alife.config import load_config, dims

    mode = sys.argv[2]
    rs = np.random.default_rng(5)
    n = 512
    xh = mx.array((rs.random(n) * 3.0).astype(np.float32))
    keep, ok, sigs, consts_expected = [], [], [], []
    for kv in json.loads(sys.argv[1]):
        cfg = load_config(preset="tiny3d", overrides=[
            "brain.weight_dtype=float32", "comm.enabled=true", "comm.width=2", "calls.enabled=true",
            f"calls.far_range=18", f"calls.far_xref={kv['xref']}", f"physics.thrust_max={kv['thrust']}",
            f"world.size={kv['size']}"])
        ncol = dims(cfg).fg ** 2
        fcell = mx.array(rs.integers(0, ncol * 7, n).astype(np.int32))

        def f(fcell_, xh_, cfg=cfg, ncol=ncol):
            if mode == "runtime":
                rt = calls.runtime(cfg)
                col = calls.column_of(fcell_, cfg, rt)
                heard = 1.0 - mx.exp(-xh_ / rt.floats[calls.F_XREF])
                push = rt.floats[calls.F_THRUST] * xh_
            else:  # the control: the same law with the scalars baked in as Python numbers
                col = fcell_ % ncol
                heard = 1.0 - mx.exp(-xh_ / cfg.calls.far_xref)
                push = cfg.physics.thrust_max * xh_
            return col, heard, push

        cf = mx.compile(f)
        keep.append(cf)
        got, want = cf(fcell, xh), f(fcell, xh)
        ok.append(all(bool(mx.array_equal(a, b).item()) for a, b in zip(got, want)))
        sigs.append(hashlib.sha1(b"".join(np.asarray(a).tobytes() for a in got)).hexdigest())
    consts = set()
    for p in pathlib.Path(sys.argv[3]).rglob("*.cpp"):
        for m in re.finditer(r"tmp_\\w+ = (-?[0-9][-+0-9.eE]*);", p.read_text()):
            consts.add(m.group(1))
    print(json.dumps({"equal": ok, "sig": sigs, "consts": sorted(consts)}))
    """
)
FAR_CASES = [
    {"xref": 1.7, "thrust": 0.0713, "size": 96},
    {"xref": 3.3, "thrust": 0.1239, "size": 128},
    {"xref": 0.45, "thrust": 0.0345, "size": 160},
]


def _run_far(tmpdir: Path, mode: str) -> dict:
    tmpdir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "TMPDIR": f"{str(tmpdir).rstrip(os.sep)}{os.sep}"}
    done = subprocess.run(
        [sys.executable, "-c", FAR_SCRIPT, json.dumps(FAR_CASES), mode, str(tmpdir)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_no_far_hearing_scalar_is_printed_into_a_kernel(tmp_path: Path) -> None:
    """The column count (a 3D box's ``fcell % Ncol``), the far intensity reference and ``physics.thrust_max``
    of the oracle are runtime arrays: three configurations that differ only in them compile one after the
    other in one process, each equals its eager result, and none of the values is a printed constant of a
    kernel source (the control with the scalars baked in prints them, so the scan can fail)."""
    ncols = [float((c["size"] // 4) ** 2) for c in FAR_CASES]
    classes = {
        "Ncol": ncols,
        "far_xref": [float(np.float32(c["xref"])) for c in FAR_CASES],
        "thrust_max": [float(np.float32(c["thrust"])) for c in FAR_CASES],
    }
    res = _run_far(tmp_path / "runtime", "runtime")
    assert res["equal"] == [True, True, True] and len(set(res["sig"])) == 3
    for name, values in classes.items():
        assert not any(_detected(res["consts"], values)), f"{name} is printed into a kernel: {res['consts']}"
    base = _run_far(tmp_path / "baked", "baked")
    assert base["equal"] == [True, True, True]
    for name in ("Ncol", "thrust_max"):
        assert all(_detected(base["consts"], classes[name])), (
            f"the control does not show {name}: {base['consts']}"
        )
