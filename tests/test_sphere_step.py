"""The sphere step as a whole (DESIGN §46; sphere series S2): the compile-collision self-check of a
compiled sphere
step and the agreement of the CPU and the Metal GPU (the compiled trajectory, its pinned digest and the
compiled-against-eager comparison are in ``test_sphere_trajectory`` and ``test_sphere_compile_eager``).

A compiled sphere configuration runs in a child process with an empty kernel cache of its own (spec 10.1
fallback, ``tests/sphere_run.py``): MLX 0.32.3 can hand a compiled function another one's kernel when their
constants print alike, and the guard that fixes it (``fix/mlx-compile-collision``) is not merged.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from alife import config, sphere_check, step
from alife.config import Config, load_config
from alife.state import init_state, state_digest, state_from_arrays, state_to_arrays, to_numpy


def tiny3s() -> Config:
    return load_config(preset="tiny3s", overrides=["brain.weight_dtype=float32"])


# --- the self-check of a compiled sphere step (spec 10.1, item 3) -----------------------------------------


def test_make_step_checks_a_compiled_sphere_and_nothing_else(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(sphere_check, "require_compile_safe", lambda cfg: calls.append(cfg.world.geometry))
    sph, box = tiny3s(), load_config(preset="tiny3d", overrides=["brain.weight_dtype=float32"])
    step.make_step(sph, compile=False)
    step.make_step(box, compile=True)
    assert calls == []  # eager sphere, compiled box: no check
    step.make_step(sph, compile=True)
    assert calls == ["sphere"]


def test_the_self_check_refuses_a_step_whose_compiled_lookups_are_wrong(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A compile cache that hands back another configuration's kernel makes the compiled lookup differ from
    the
    eager one: the check raises, naming the collision, instead of letting a step place agents in wrong
    cells."""
    cfg = tiny3s()
    monkeypatch.setattr(sphere_check, "_DONE", set())

    def colliding(fn):  # a "compiled" function that returns the wrong integers
        def wrong(*a, **k):
            return tuple(o + 1 for o in fn(*a, **k))

        return wrong

    monkeypatch.setattr(mx, "compile", colliding)
    with pytest.raises(RuntimeError, match="compile collision"):
        sphere_check.require_compile_safe(cfg)
    assert not sphere_check._DONE  # a failed check is not memoised


def test_the_self_check_points_are_in_the_shell_and_avoid_the_pixel_boundaries() -> None:
    cfg = tiny3s()
    pts = sphere_check._points(cfg).astype(np.float64)
    rad = np.linalg.norm(pts, axis=1)
    assert (
        rad.min() >= float(config.dims(cfg).r_lo) - 1e-5 and rad.max() <= float(config.dims(cfg).r_hi) + 1e-5
    )
    assert len(pts) == 384 + 8 + 2
    c1, c2 = (
        sphere_check.constants(cfg),
        sphere_check.constants(load_config(preset="tiny3s", overrides=["world.nside=9", "world.layers=9"])),
    )
    assert c1 != c2


# --- CPU and Metal ----------------------------------------------------------------------------------------


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs the Metal GPU")
def test_cpu_and_metal_agree_on_every_integer_field_over_ten_steps() -> None:
    """From one numpy state, ten eager steps on each device: every integer and boolean field, the counters and
    the ledger are equal; the float fields differ only by rounding (the devices' brain arithmetic differs by
    1e-7, as the box's does: digests are never equal across devices, from the first step)."""
    cfg = tiny3s()
    saved = mx.default_device()
    try:
        mx.set_default_device(mx.cpu)
        s0 = init_state(cfg)
        mx.eval(s0)
        arrays = {k: to_numpy(v) for k, v in state_to_arrays(s0).items()}
        runs = []
        for dev in (mx.cpu, mx.gpu):
            mx.set_default_device(dev)
            s = state_from_arrays({k: mx.array(v) for k, v in arrays.items()}, cfg)
            fn = step.make_step(cfg, compile=False)
            outs = []
            for _ in range(10):
                s, out = fn(s)
                mx.eval(s, out)
                outs.append((to_numpy(out.counters), to_numpy(out.ledger)))
            runs.append((s, outs))
    finally:
        mx.set_default_device(saved)
    (sc, oc), (sg, og) = runs
    for (c1, l1), (c2, l2) in zip(oc, og, strict=True):
        assert np.array_equal(c1, c2) and np.array_equal(l1, l2)
    worst = 0.0
    for name in type(sc)._fields:
        a, b = to_numpy(getattr(sc, name)), to_numpy(getattr(sg, name))
        if a.dtype.kind == "f":
            worst = max(worst, float(np.abs(a.astype(np.float64) - b).max()) if a.size else 0.0)
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=5e-5, err_msg=name)
        else:
            assert np.array_equal(a, b), name
    assert (
        state_digest(sc) != state_digest(sg) or worst == 0.0
    )  # recorded: the digests are not device-independent
