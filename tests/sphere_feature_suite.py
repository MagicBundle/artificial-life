"""The per-feature checks that every feature of the sphere series' S3 runs (DESIGN §46.5): the box suites'
step-level guarantees on a sphere fixture, written once and parametrised by the feature.

The box suites compile their step in the test process; a compiled sphere step must run in a cold child
(spec 10.1 fallback, ``tests/sphere_run.py``), so these checks run the *eager* step of ``tiny3s`` with the
feature forced on (a feature's output bias is set on every founder, as the box suites' ``builders`` do) and
the compiled step is in ``test_sphere_features_compiled``. Each check is one of:

* ``check_run``: the 16-term ledger closes at every step, no error diag bit, ``invariants.full_check`` every
  ten steps (host ranges, the feature's own ``check``), and the feature's counter is non-zero (it acts);
* ``check_observe_pure``: ``step(observe=True)`` changes no state value and no ``StepOut`` field;
* ``check_deterministic``: two runs from one seed give one digest;
* ``check_resume``: a checkpoint written and loaded half-way continues to the uninterrupted digest;
* ``check_off_footprint``: the config without the feature has none of its columns, state fields or counters.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

import helpers
from alife import ext, invariants, layout
from alife.config import Config, dims, load_config
from alife.runio import load_checkpoint, write_state_file
from alife.state import Diag, counter_index, init_state, state_cls, state_digest, to_numpy
from alife.step import make_step


@dataclass(frozen=True)
class Spec:
    overlays: tuple[str, ...] = ()
    overrides: tuple[str, ...] = ()
    bias: tuple[tuple[str, float], ...] = ()  # (output name, bias given to every founder)
    counter: str = ""  # the counter that must be non-zero (the feature acts)
    field: str | None = None  # the State field the feature adds
    feature: str = ""  # the ext registry name


ARMS = (
    "arms.enabled=true",
    "arms.key_floor=0.2",
    "arms.init_atk=0.3",
    "arms.init_arm=0.4",
    "arms.sigma=0.05",
    "arms.sigma_phase=0.05",
    "arms.speed_slow=0.2",
    "arms.upkeep_atk=2",
    "arms.upkeep_arm=2",
    "combat.protect_floor=0.1",
)
SPECS: dict[str, Spec] = {
    "structures": Spec(
        overlays=("build",),
        overrides=("structures.open_frac=0.02",),
        bias=(("OUT_BUILD", 4.0),),
        counter="BUILD_UNITS",
        field="mat",
        feature="structures",
    ),
    "niche": Spec(
        overlays=("niche3d",),
        overrides=("niche.silent_init=false",),  # about half the founders farm: full output starves them
        counter="FARM_YIELD",
        field="soil",
        feature="niche",
    ),
    "comm": Spec(overlays=("comm",), counter="COMM_SPEAKERS", feature="comm"),
    "bonds": Spec(
        overlays=("bonds",),
        bias=(("OUT_BOND", 4.0), ("OUT_CHILD_BOND", 4.0)),
        counter="SHARE_SLOTS_ACTIVE",
        feature="bonds",
    ),
    "arms": Spec(overrides=ARMS, counter="ARMS_YIELD_SUM", field="arms", feature="arms"),
    "all": Spec(
        overlays=("build", "niche3d", "comm", "bonds"),
        overrides=(*ARMS, "structures.open_frac=0.02", "niche.silent_init=false"),
        bias=(("OUT_BUILD", 4.0), ("OUT_BOND", 4.0), ("OUT_CHILD_BOND", 4.0)),
        counter="BUILD_UNITS",
        feature="structures",
    ),
}


def make_cfg(name: str, *extra: str, compile: bool = False) -> Config:
    sp = SPECS[name]
    cfg = load_config(
        preset="tiny3s",
        overlays=sp.overlays,
        overrides=["brain.weight_dtype=float32", *sp.overrides, *extra],
    )
    return cfg if compile else replace(cfg, run=replace(cfg.run, compile=False))


def column(cfg: Config, name: str) -> int:
    """The output column ``name`` (a feature's on the layout instance, a base one in the module)."""
    col = getattr(dims(cfg).lay, name, None)
    return col if col is not None else getattr(layout, name)


def forced_state(name: str, cfg: Config):
    """A fresh state whose founders all have the feature's output bias."""
    s = init_state(cfg)
    bias = {column(cfg, k): v for k, v in SPECS[name].bias}
    return helpers.set_output_bias(s, cfg, bias) if bias else s


def check_run(name: str, steps: int = 60, *extra: str) -> dict[str, int]:
    """Run ``steps`` eager steps; returns the sum of the feature's counter and the last step's alive count."""
    cfg = make_cfg(name, *extra)
    sp = SPECS[name]
    fn = make_step(cfg, compile=False)
    s = forced_state(name, cfg)
    total = 0
    for i in range(steps):
        prev = s
        s, out = fn(s)
        mx.eval(s, out)
        led = to_numpy(out.ledger)
        assert helpers.ledger_net(led, cfg) == int(to_numpy(s.total) - to_numpy(prev.total)), (name, i)
        assert int(to_numpy(s.diag)) & 0xFFFF == 0, (name, i, Diag(int(to_numpy(s.diag))))
        if i % 10 == 9:
            assert invariants.full_check(s, cfg, prev, out) == [], (name, i)
        total += int(to_numpy(out.counters)[counter_index(cfg, sp.counter)])
    assert total > 0, f"{name}: counter {sp.counter} stayed 0 over {steps} steps"
    return {"counter": total, "alive": int(to_numpy(s.alive).sum())}


def check_observe_pure(name: str, steps: int = 12) -> None:
    cfg = make_cfg(name)
    f_off, f_on = make_step(cfg, compile=False), make_step(cfg, compile=False, observe=True)
    a = b = forced_state(name, cfg)
    for t in range(steps):
        a, oa = f_off(a)
        b, ob = f_on(b)
        mx.eval(a, b, oa, ob)
        assert state_digest(a) == state_digest(b), (name, t)
        for x, y in zip(tree_flatten(oa), tree_flatten(ob.base()), strict=True):
            np.testing.assert_array_equal(to_numpy(x[1]), to_numpy(y[1]), err_msg=x[0])


def _final(cfg: Config, s, steps: int):
    fn = make_step(cfg, compile=False)
    for _ in range(steps):
        s, out = fn(s)
        mx.eval(s, out)
    return s


def check_deterministic(name: str, steps: int = 20) -> None:
    cfg = make_cfg(name)
    assert state_digest(_final(cfg, forced_state(name, cfg), steps)) == state_digest(
        _final(cfg, forced_state(name, cfg), steps)
    )


def check_resume(name: str, tmp_path: Path, steps: int = 15) -> None:
    cfg = make_cfg(name)
    s0 = forced_state(name, cfg)
    straight = _final(cfg, s0, 2 * steps)
    half = _final(cfg, s0, steps)
    path = tmp_path / "half.safetensors"
    write_state_file(path, half, cfg)
    resumed = _final(cfg, load_checkpoint(path, cfg), steps)
    assert state_digest(resumed) == state_digest(straight)


def check_off_footprint(name: str) -> None:
    """Without the feature: the dry sphere's layout (84/39), its state class and no extension field."""
    sp = SPECS[name]
    off = load_config(preset="tiny3s", overrides=["brain.weight_dtype=float32"])
    on = make_cfg(name)
    d_off, d_on = dims(off), dims(on)
    assert (d_off.lay.D_IN, d_off.lay.D_OUT) == (84, 39) and ext.active(off) == ()
    assert (d_on.lay.D_IN, d_on.lay.D_OUT) >= (d_off.lay.D_IN, d_off.lay.D_OUT)
    if name != "arms":  # the arms layer is genes and state: it adds no brain column
        assert d_on.lay.D_IN > d_off.lay.D_IN and d_on.lay.D_OUT > d_off.lay.D_OUT
    assert state_cls(off) is state_cls(load_config(preset="tiny3s"))
    s = init_state(off)
    if sp.field is not None:
        assert sp.field not in type(s)._fields
        assert sp.field in type(init_state(on))._fields
    assert sp.feature in {spec.name for spec in ext.active(on)} or name == "all"
