"""§28.7 purity: observing never changes the trajectory (G1).

With ``observe`` on the step returns the observer channel next to the same ``StepOut``, and the ``Observer``
folds the channel into per-row accumulators in the same ``mx.eval`` as the state (the runner's eval pattern).
The state digest and every ``StepOut`` field must equal those of a run with observers off, bit for bit, at
every step: 2D and 3D, compiled and eager, float32 and float16 weights, the director's static branches off and
on. The runner-level tests check the same through ``runner.run`` and the CLI: the run files and the director
log of an observing run equal those of a run without observers, except the ``observers`` additions. This is an
empirical gate (the channel reads only arrays that are already kernel boundaries, or integers, §28.7); it
runs on the CPU in the suite and on Metal before merge.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from alife import cli, runner
from alife.config import Config, apply_overrides, load_config
from alife.game.observe import EVAL_BIT, ObsIn, obs_in
from alife.game.observer import Observer
from alife.game.rules import RULES_V1
from alife.runio import DIRECTOR_LOG_FILE, RunDir, load_checkpoint
from alife.state import CounterIdx as C
from alife.state import State, init_state, state_digest, to_numpy
from alife.step import StepOut, make_step

COMPILED_STEPS = 60
EAGER_STEPS = 20


def _same(a, b) -> bool:
    fa, fb = tree_flatten(a), tree_flatten(b)
    pairs = zip(fa, fb, strict=True)
    return [k for k, _ in fa] == [k for k, _ in fb] and all(
        x.dtype == y.dtype and np.array_equal(to_numpy(x), to_numpy(y)) for (_, x), (_, y) in pairs
    )


def _config(d3: bool, dtype: str, director: bool, compiled: bool = True) -> Config:
    overrides = [
        f"brain.weight_dtype={dtype}",
        *(["director.enabled=true"] if director else []),
        *([] if compiled else ["run.compile=false"]),  # the Observer follows the run's compile mode
    ]
    return load_config(preset="tiny3d" if d3 else "tiny", overrides=overrides)


def _levers(s: State, cfg: Config) -> State:
    """Nonzero director levers: an immigration request and a half-strength catastrophe over the centre.

    Both are set from the config alone, so the observing and the control state get identical edits.
    """
    half = cfg.world.size / 2
    return s._replace(
        env_imm_request=mx.array(16, dtype=mx.int32),
        env_cat_center=mx.array([half, half], dtype=mx.float32),
        env_cat_radius=mx.array(40.0, dtype=mx.float32),
        env_cat_kill_p=mx.array(0.5, dtype=mx.float32),
    )


# 3D, weight dtype, director on, compiled. Eager mode has no fusion, so only the float32 director-off arms are
# run eagerly (the Observer then runs eagerly too). Every arm takes 1 to 4 s on the CPU: none is marked slow.
ARMS = [
    pytest.param(False, "float32", False, True, id="2d-f32-off"),
    pytest.param(True, "float32", False, True, id="3d-f32-off"),
    pytest.param(False, "float32", False, False, id="2d-f32-off-eager"),
    pytest.param(True, "float32", False, False, id="3d-f32-off-eager"),
    pytest.param(False, "float16", False, True, id="2d-f16-off"),
    pytest.param(False, "float32", True, True, id="2d-f32-on"),
    pytest.param(False, "float16", True, True, id="2d-f16-on"),
    pytest.param(True, "float16", True, True, id="3d-f16-on"),
]


@pytest.mark.parametrize(("d3", "dtype", "director", "compiled"), ARMS)
def test_observing_does_not_change_the_trajectory(
    d3: bool, dtype: str, director: bool, compiled: bool
) -> None:
    cfg = _config(d3, dtype, director, compiled)
    steps = COMPILED_STEPS if compiled else EAGER_STEPS
    off = make_step(cfg, compile=compiled)
    on = make_step(cfg, compile=compiled, observe=True)
    s_off, s_on = init_state(cfg), init_state(cfg)
    obs = Observer.fresh(s_on, cfg)
    catastrophes = immigrants = 0
    for t in range(steps):
        if director and t in (0, steps // 2):  # the one-step levers are consumed by the step: set them again
            s_off, s_on = _levers(s_off, cfg), _levers(s_on, cfg)
        s_off, o_off = off(s_off)
        mx.eval(s_off, o_off)
        s_on, o_on = on(s_on)
        obs.update(s_on, o_on.acts, t + 1)
        mx.eval(s_on, o_on.base(), *obs.arrays())  # the runner's pattern: one blocking eval per step
        where = f"step {t}"
        assert type(o_off) is StepOut and type(o_on.base()) is StepOut
        assert state_digest(s_off) == state_digest(s_on), where
        assert _same(o_off, o_on.base()), where
        c = to_numpy(o_off.counters)
        catastrophes += int(c[C.DEATHS_CATASTROPHE])
        immigrants += int(c[C.IMMIGRANTS])
    assert (
        catastrophes > 0 and immigrants > 0
    ) == director  # the director arms run the levers, the others not
    assert int(to_numpy(s_on.step)) == steps
    # the observer actually ran: it accumulated, and (past the intent warm-up) evaluated rows
    assert int(to_numpy(obs.o.n_obs).max()) == steps and int(to_numpy(obs.o.arch).max()) > 0
    assert (steps < RULES_V1.intent_warm) == (not (to_numpy(obs.o.match_intent) & EVAL_BIT).any())


def test_observer_inputs_exclude_the_key() -> None:
    """The observer reads a handful of State fields and never the RNG key or the weights (§28.1)."""
    assert set(ObsIn._fields) == {
        "alive", "uid", "parent_uid", "partner", "vel", "last_harvest", "last_inflow", "last_loss",
    }  # fmt: skip
    assert "key" not in ObsIn._fields
    assert not {"W1", "W2", "pend_W1", "pend_W2", "h", "msg", "tag"} & set(ObsIn._fields)
    s = init_state(load_config(preset="tiny", overrides=["brain.weight_dtype=float32"]))
    oi = obs_in(s)
    assert all(a is getattr(s, f) for f, a in zip(ObsIn._fields, oi, strict=True))


# --- through the runner and the CLI -------------------------------------------------------------------------

PERF_KEYS = frozenset({"wall_s", "steps_per_s", "peak_mem_gb"})  # wall-clock keys of an interval row


def _stored_rows(rd: RunDir) -> list[dict]:
    """The stored metrics rows without the wall-clock keys and without the ``observers`` block."""
    return [
        {k: v for k, v in r.items() if k not in PERF_KEYS and k != "observers"} for r in rd.metrics_rows()
    ]


def _checkpoint_digests(rd: RunDir) -> dict[int, str]:
    cfg = rd.config()
    return {st: state_digest(load_checkpoint(path, cfg)) for st, path in rd.checkpoints()}


def _files(rd: RunDir) -> list[str]:
    return sorted(str(p.relative_to(rd.path)) for p in rd.path.rglob("*") if p.is_file())


@pytest.mark.parametrize("d3", [False, True], ids=["2d", "3d"])
def test_runner_files_identical_except_observers(d3: bool, tmp_path: Path) -> None:
    """60 steps with and without observers: the checkpoints' state digests, the event chunks and the stored
    rows are equal; the only differences are the ``observers`` block of every row of the observing run and its
    ``observers/`` sidecars, which the other run does not have at all."""
    base = load_config(
        preset="tiny3d" if d3 else "tiny",
        overrides=[
            "brain.weight_dtype=float32",
            "run.checkpoint_every=30",
            "run.heavy_metrics_every=30",
            "run.diversity_sample=64",
        ],
    )
    runs: dict[str, RunDir] = {}
    for name in ("off", "on"):
        cfg = apply_overrides(base, [f"game.observe={'true' if name == 'on' else 'false'}"])
        rd = RunDir.create(tmp_path / name, cfg)
        res = runner.run(rd, cfg, init_state(cfg), 60, quiet=True)
        assert (res.status, res.exit_code) == ("done", runner.ExitCode.OK)
        del res
        runs[name] = rd
    off, on = runs["off"], runs["on"]
    assert [st for st, _ in off.checkpoints()] == [st for st, _ in on.checkpoints()] == [0, 30, 60]
    assert _checkpoint_digests(off) == _checkpoint_digests(on)
    for kind in ("births", "deaths"):
        assert [(s0, s1) for s0, s1, _ in off.event_files(kind)] == [
            (s0, s1) for s0, s1, _ in on.event_files(kind)
        ]
        a, b = off.read_events(kind), on.read_events(kind)
        assert a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a), kind
    assert [r["step"] for r in off.metrics_rows()] == list(range(10, 61, 10))
    assert _stored_rows(off) == _stored_rows(on)
    assert all("observers" not in r for r in off.metrics_rows())
    assert all(r["observers"]["intent_version"] == 1 for r in on.metrics_rows())
    assert not off.observers_dir.exists() and off.observer_files() == []
    assert [st for st, _ in on.observer_files()] == [0, 30, 60]
    assert [f for f in _files(off) if f not in _files(on)] == []  # nothing else is missing from the on run
    assert [f for f in _files(on) if f not in _files(off)] == [
        f"observers/step_{st:010d}.npz" for st in (0, 30, 60)
    ]
    # the stored config differs in game.observe only
    c_off, c_on = (json.loads((rd.path / "config.json").read_text()) for rd in (off, on))
    assert c_off.pop("game") == {"observe": False} and c_on.pop("game") == {"observe": True} and c_off == c_on


def test_director_log_unchanged_by_observers(tmp_path: Path) -> None:
    """A live Gardener (rule tier) acts on the run's vital signs; with observers on, its log, the checkpoints
    and the stored rows are those of the run without. Few founders make the director act (a regeneration pulse
    and an immigration pulse at step 20). The log's entries come from coarse vitals and could stay equal after
    a divergence, hence the checkpoint digests as well."""
    argv = [
        "run", "--preset", "tiny",
        "--set", "brain.weight_dtype=float32",
        "--set", "population.n_init=24",
        "--set", "director.rule_every=20",
        "--set", "run.heavy_metrics_every=20",
        "--set", "run.diversity_sample=64",
        "--set", "run.checkpoint_every=30",
        "--director", "live", "--steps", "60", "--quiet",
    ]  # fmt: skip
    runs = {}
    for name, extra in (("off", []), ("on", ["--set", "game.observe=true"])):
        assert cli.main([*argv, *extra, "--out", str(tmp_path / name)]) == 0
        runs[name] = RunDir.open(tmp_path / name)
    off, on = runs["off"], runs["on"]
    log_off, log_on = off.read_jsonl(DIRECTOR_LOG_FILE), on.read_jsonl(DIRECTOR_LOG_FILE)
    # No volatile field exists in the log today (no timestamps): the records are compared whole.
    assert log_off == log_on
    assert {e["action"] for e in log_off if e.get("kind") != "header"} >= {"regen_pulse", "immigration_pulse"}
    assert _checkpoint_digests(off) == _checkpoint_digests(on)
    assert [st for st, _ in on.checkpoints()] == [0, 30, 60]
    assert _stored_rows(off) == _stored_rows(on)
    assert all("observers" in r for r in on.metrics_rows())
    assert max(r["observers"]["alive"] for r in on.metrics_rows()) > 24  # the immigrants were observed
