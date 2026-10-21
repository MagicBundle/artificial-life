"""§28.7 purity: observing never changes the trajectory (G1).

With ``observe`` on the step returns the observer channel next to the same ``StepOut``, and the ``Observer``
folds the channel into per-row accumulators in the same ``mx.eval`` as the state (the runner's eval pattern).
The state digest and every ``StepOut`` field must equal those of a run with observers off, bit for bit, at
every step: 2D and 3D, compiled and eager, float32 and float16 weights, the director's static branches off and
on. The runner-level tests check the same through ``runner.run`` and the CLI: the run files and the director
log of an observing run equal those of a run without observers, except the ``observers`` additions. This is an
empirical gate (the channel reads only arrays that are already kernel boundaries, or integers, §28.7); it
runs on the CPU in the suite and on Metal before merge. The value tests cannot see a regrouping of the fused
float kernels that leaves every value unchanged (it can change Metal's rounding), so a structural test pins
the channel's read set on the traced graph: which trajectory arrays the channel reads, and that S12 is
computed once (§28.2). A second one compares the fused kernels of the compiled step with observe off and on,
because ``mx.compile`` merges identical sub-expressions and so reads more than the eager graph shows.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import re
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from alife import actions, bonds, brain, cli, energy, layout, lifecycle, runner, spatial
from alife import step as step_mod
from alife.config import Config, apply_overrides, config_to_dict, load_config
from alife.game.observe import EVAL_BIT, ObsIn, obs_in
from alife.game.observer import Observer
from alife.game.rules import RULES_V2
from alife.game.social import SOCIAL_V1
from alife.runio import DIRECTOR_LOG_FILE, RunDir, load_checkpoint
from alife.state import CounterIdx as C
from alife.state import State, init_state, state_digest, to_numpy
from alife.step import StepOut, make_step
from game_helpers import social_every

COMPILED_STEPS = 60
EAGER_STEPS = 20
# The step-level arms run the whole observer, social part included, with snapshots every 15 steps so that
# snapshots and their settles happen inside the 60 compiled and 20 eager steps. They do not take the records,
# so each snapshot is settled at the next ``update`` (the labels tool's path); the runner-level arms take them
# at the snapshot step and append them to the social log (the runner's path) with ``social_every``.
SOC15 = dataclasses.replace(SOCIAL_V1, snapshot_every=15)


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
    obs = Observer.fresh(s_on, cfg, social=SOC15)
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
    assert (steps < RULES_V2.intent_warm) == (not (to_numpy(obs.o.match_intent) & EVAL_BIT).any())
    # ... and so did the social layer: its snapshots were taken, settled and recorded inside the run
    records = obs.take_records()
    assert [r["step"] for r in records] == list(range(15, steps + 1, 15))
    assert obs.society.snapshot_step == (steps // 15) * 15
    assert all(r["origin"] == 0 and r["summary"]["houses"]["alive"] > 0 for r in records)
    assert all(
        r["summary"]["houses"]["exact"] and not r["summary"]["houses"]["unknown_immigrant"] for r in records
    )


def test_observer_inputs_exclude_the_key() -> None:
    """The observer reads a handful of State fields and never the RNG key or the weights (§28.1)."""
    assert set(ObsIn._fields) == {
        "alive", "uid", "parent_uid", "partner", "vel", "last_harvest", "last_inflow", "last_loss",
        "energy", "brain_cost",
    }  # fmt: skip
    assert "key" not in ObsIn._fields
    assert not {"W1", "W2", "pend_W1", "pend_W2", "h", "msg", "tag"} & set(ObsIn._fields)
    s = init_state(load_config(preset="tiny", overrides=["brain.weight_dtype=float32"]))
    oi = obs_in(s)
    assert all(a is getattr(s, f) for f, a in zip(ObsIn._fields, oi, strict=True))


# --- the channel's read set, structurally -------------------------------------------------------------------

# The trajectory arrays the observer channel may read (§28.2, §29.2): the decode thrust before S8 (it also
# feeds the cost term), the post-S8 defend and emit (inputs of the S11 gather and of the S17 deposit), the
# solvency mask, the lane kind and parent, the S11 attack, hit and gain, the S9 formed bits and the S12 slot
# flows; and for social tracking the post-S8 attack and bond targets, the S9 proposer bits, the S14 death
# cause (read by the KILLED bit and, with the roster, by the int8 ``death_cause`` field: one tap), the
# neighbour rows, the brain output y (the one new float tap: a kernel output read by two column slices)
# and the birth records' child rows and uids.
CHANNEL_TAPS = frozenset(
    {
        "thrust0", "defend_post", "emit_post", "solvent", "lanes_kind", "lanes_parent",
        "cb_attack", "cb_hit", "cb_gain", "formed", "sl_out", "sl_got",
        "attack_target_post", "bond_target_post", "as_prop", "dt_cause", "nbr_idx", "y",
        "births_slot", "births_uid",
    }
)  # fmt: skip

_DOT_PRIM = re.compile(r'\{ (\d+) \[label ="([^"]+)"')
_DOT_IN = re.compile(r'"([^"]+)" -> (\d+)$')
_DOT_OUT = re.compile(r'(\d+) -> "([^"]+)"$')


def _parse_dot(path: Path) -> tuple[dict[str, str], dict[str, str], dict[str, list[str]]]:
    """The label of every primitive, the producer primitive of every array name, and the input array names of
    every primitive, from an ``mx.export_to_dot`` file. Primitive ids are object addresses: they agree across
    two exports made in the same process, which is what the eager comparison below relies on; the compiled
    comparison uses the labels only."""
    labels: dict[str, str] = {}
    producer: dict[str, str] = {}
    inputs: dict[str, list[str]] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if m := _DOT_PRIM.match(line):
            labels[m.group(1)] = m.group(2)
        elif m := _DOT_IN.match(line):
            inputs.setdefault(m.group(2), []).append(m.group(1))
        elif m := _DOT_OUT.match(line):
            producer[m.group(2)] = m.group(1)
    return labels, producer, inputs


def _channel_taps(
    cfg: Config,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutate: Callable[[pytest.MonkeyPatch], None] | None = None,
) -> tuple[set[str], list[str], int]:
    """Trace ``step(s, cfg, observe=True)`` eagerly (lazy, nothing is evaluated) and find the channel's taps.

    A tap is an array produced by a primitive of the trajectory graph (the new state and ``out.base()``) that
    the channel reads: an input of a primitive only the channel uses, or a channel output itself. Returns the
    names of the allowed taps found, the unlisted taps (the auto-assigned names of the arrays read by a
    channel op, and ``Acts.<field>`` for a channel field that *is* an unlisted trajectory array, returned
    with no op of its own), and the number of ``share_slots`` calls. The allowed arrays are captured by
    wrapping the sub-step functions and named through the keyword arguments of ``export_to_dot``; every other
    channel field is named ``acts_<field>`` so that a field that is a trajectory array is seen. ``mutate``
    installs a deliberate break after the wrappers (the meta-tests).

    This is the *eager* graph. ``mx.compile`` also merges identical sub-expressions (CSE), so a channel
    expression that repeats a trajectory one reads the intermediate of a fused kernel although it lists only
    allowed arrays here: ``test_the_channel_regroups_no_float_kernel_of_the_compiled_step`` covers that.
    """
    cap: dict = {"sl_all": []}
    real = (
        actions.mask, energy.pay_costs, energy.pay_costs3, energy.combat, energy.share_slots,
        bonds.form_bonds, lifecycle.allocate_lanes,
        brain.forward, spatial.find_neighbors, lifecycle.deaths, lifecycle.reproduce,
    )  # fmt: skip
    r_mask, r_pay, r_pay3, r_combat, r_share, r_form, r_lanes, r_fwd, r_nbr, r_deaths, r_repro = real

    def mask(act, keep):
        out = r_mask(act, keep)
        if isinstance(act, actions.Actions):  # the 3D motion mask is a different type
            cap["pre"], cap["post"] = act, out
        return out

    def keep(name: str, fn):
        def wrapped(*a, **k):
            cap[name] = fn(*a, **k)
            return cap[name]

        return wrapped

    def share_slots(*a, **k):
        cap["sl_all"].append(r_share(*a, **k))
        return cap["sl_all"][-1]

    monkeypatch.setattr(actions, "mask", mask)
    monkeypatch.setattr(energy, "pay_costs", keep("c", r_pay))
    monkeypatch.setattr(energy, "pay_costs3", keep("c", r_pay3))
    monkeypatch.setattr(energy, "combat", keep("cb", r_combat))
    monkeypatch.setattr(energy, "share_slots", share_slots)
    monkeypatch.setattr(bonds, "form_bonds", keep("fb", r_form))
    monkeypatch.setattr(lifecycle, "allocate_lanes", keep("lanes", r_lanes))
    monkeypatch.setattr(brain, "forward", keep("fwd", r_fwd))
    monkeypatch.setattr(spatial, "find_neighbors", keep("nbr", r_nbr))
    monkeypatch.setattr(lifecycle, "deaths", keep("dt", r_deaths))
    monkeypatch.setattr(lifecycle, "reproduce", keep("rp", r_repro))
    if mutate is not None:
        mutate(monkeypatch)
    s = init_state(cfg)
    mx.eval(s)
    s1, out = step_mod.step(s, cfg, observe=True)
    monkeypatch.undo()  # also undoes the mutation; the traced graph is complete
    sl = cap["sl_all"][0]
    allowed = {
        "thrust0": cap["pre"].thrust,
        "defend_post": cap["post"].defend,
        "emit_post": cap["post"].emit,
        "solvent": cap["c"].solvent,
        "lanes_kind": cap["lanes"].kind,
        "lanes_parent": cap["lanes"].parent,
        "cb_attack": cap["cb"].attack,
        "cb_hit": cap["cb"].hit,
        "cb_gain": cap["cb"].gain,
        "formed": cap["fb"][2],
        "sl_out": sl.out,
        "sl_got": sl.got,
        "attack_target_post": cap["post"].attack_target,
        "bond_target_post": cap["post"].bond_target,
        "as_prop": cap["fb"][3],
        "dt_cause": cap["dt"].cause,
        "nbr_idx": cap["nbr"].idx,
        "y": cap["fwd"][1],
        "births_slot": cap["rp"].records.slot,
        "births_uid": cap["rp"].records.uid,
    }
    trajectory = [v for _, v in tree_flatten((s1, out.base()))]
    f_traj, f_acts = tmp_path / "trajectory.dot", tmp_path / "acts.dot"
    mx.export_to_dot(str(f_traj), *trajectory)
    # a field that is one of the listed arrays (the same object) is a listed tap; any other field is named so
    # that, if it is a trajectory array, its producer is found (a field with no op is nobody's input)
    seen = {id(a) for a in allowed.values()}
    fields: dict[str, mx.array] = {}
    for name, a in zip(out.acts._fields, out.acts, strict=True):
        if id(a) not in seen:
            seen.add(id(a))
            fields[f"acts_{name}"] = a
    mx.export_to_dot(str(f_acts), *out.acts, **allowed, **fields)
    p_traj, _, _ = _parse_dot(f_traj)
    p_acts, producer, inputs = _parse_dot(f_acts)
    taps: set[str] = set()
    for prim in p_acts.keys() - p_traj.keys():  # primitives only the channel uses
        taps.update(a for a in inputs.get(prim, []) if producer.get(a) in p_traj)
    taps.update(a for a, prim in producer.items() if prim in p_traj and a in allowed)  # outputs read directly
    unlisted = sorted(t for t in taps if t not in allowed)
    unlisted += [f"Acts.{k.removeprefix('acts_')}" for k in fields if producer.get(k) in p_traj]  # no op
    return {t for t in taps if t in allowed}, unlisted, len(cap["sl_all"])


CHANNEL_CONFIGS = [
    pytest.param("tiny", ["brain.weight_dtype=float32"], id="tiny-f32"),
    pytest.param("tiny", ["director.enabled=true"], id="tiny-f16-director"),
    pytest.param("tiny3d", ["brain.weight_dtype=float32"], id="tiny3d-f32"),
]


@pytest.mark.parametrize(("preset", "overrides"), CHANNEL_CONFIGS)
def test_channel_reads_only_the_allowed_arrays(
    preset: str, overrides: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """§28.2, §29.2: S12 is computed once, and the channel reads no float array of the trajectory beyond the
    pre-S8 thrust, the post-S8 defend and emit, the brain output ``y`` (a kernel output, two column slices)
    and the integer and boolean flows. A reader of anything else (for example the post-S8 float thrust or
    the neighbour validity) or a second S12 would add a consumer to a fused float chain, which can change the
    compiled step's rounding on Metal while leaving every value of the CPU tests unchanged."""
    taps, unnamed, n_share = _channel_taps(
        load_config(preset=preset, overrides=overrides), monkeypatch, tmp_path
    )
    assert n_share == 1, f"share_slots was called {n_share} times"
    assert not unnamed, f"the channel reads {len(unnamed)} unlisted trajectory arrays: {unnamed}"
    assert taps == CHANNEL_TAPS  # all of them are read, so the probe is not vacuous
    assert len(CHANNEL_TAPS) == 20  # G1's 12 and the 8 of G2 (§29.2)


def _thrust_after_the_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutant: the channel reads the post-S8 float thrust (same values, a new consumer of a fused chain)."""
    orig = step_mod._acts
    monkeypatch.setattr(
        step_mod, "_acts", lambda f, sl, thrust0, y, alive0, cfg: orig(f, sl, f.act.thrust, y, alive0, cfg)
    )


def _valid_from_nbr_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutant: the channel builds ``nbr`` through ``nbr.valid`` (same values: invalid entries hold the row's
    own index, so ``where(valid, idx, row)`` is ``idx``; one new trajectory array read)."""
    orig = step_mod._acts

    def acts(f, sl, thrust0, y, alive0, cfg):
        rows = mx.arange(f.nbr.idx.shape[0], dtype=mx.int32)[:, None]
        nbr = f.nbr._replace(idx=mx.where(f.nbr.valid, f.nbr.idx, rows))
        return orig(f._replace(nbr=nbr), sl, thrust0, y, alive0, cfg)

    monkeypatch.setattr(step_mod, "_acts", acts)


def _thrust_as_a_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutant: a channel field *is* the post-S8 float thrust (no op of its own: it is no primitive's input, so
    only the field check of ``_channel_taps`` sees it; the ``EMITTED`` flag still taps ``emit_post``)."""
    orig = step_mod._acts
    monkeypatch.setattr(
        step_mod,
        "_acts",
        lambda f, sl, thrust0, y, alive0, cfg: orig(f, sl, thrust0, y, alive0, cfg)._replace(
            emit=f.act.thrust
        ),
    )


def _second_share(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutant: S12 is computed twice (the same values; a second consumer of every float node of the share)."""
    once = energy.share_slots

    def twice(*a, **k):
        once(*a, **k)
        return once(*a, **k)

    monkeypatch.setattr(energy, "share_slots", twice)


MUTANTS = [
    pytest.param(_thrust_after_the_mask, id="post-S8-thrust"),
    pytest.param(_second_share, id="2nd-S12"),
    pytest.param(_valid_from_nbr_valid, id="nbr-valid"),
    pytest.param(_thrust_as_a_field, id="post-S8-thrust-as-a-field"),
]


@pytest.mark.parametrize("mutant", MUTANTS)
def test_the_structural_probe_catches_value_neutral_breaks(
    mutant: Callable[[pytest.MonkeyPatch], None], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every mutant leaves every value, and so every other test of this file, unchanged on the CPU."""
    _, unnamed, n_share = _channel_taps(load_config(preset="tiny"), monkeypatch, tmp_path, mutate=mutant)
    assert unnamed or n_share != 1
    if mutant is _valid_from_nbr_valid:
        assert len(unnamed) == 1 and n_share == 1  # exactly the one new read: nbr.valid
    if mutant is _thrust_as_a_field:
        assert unnamed == ["Acts.emit"] and n_share == 1  # the field itself, which no op reads


@pytest.mark.parametrize("mutant", MUTANTS)
@pytest.mark.parametrize("preset", ["tiny", "tiny3d"])
def test_the_mutants_are_value_neutral(mutant: Callable[[pytest.MonkeyPatch], None], preset: str) -> None:
    """The premise of the meta-test: a mutated channel returns the same arrays as the real one. The field
    mutant cannot: its field holds the trajectory array itself. It leaves the step's own outputs and every
    other field equal."""
    cfg = load_config(preset=preset, overrides=["brain.weight_dtype=float32"])
    s = init_state(cfg)
    mx.eval(s)
    _, real = step_mod.step(s, cfg, observe=True)
    mx.eval(real)
    with pytest.MonkeyPatch.context() as m:
        mutant(m)
        _, mutated = step_mod.step(s, cfg, observe=True)
        mx.eval(mutated)
    if mutant is _thrust_as_a_field:
        assert _same(real.base(), mutated.base())
        same_fields = [
            _same(a, b)
            for name, a, b in zip(real.acts._fields, real.acts, mutated.acts, strict=True)
            if name != "emit"
        ]
        assert all(same_fields) and not _same(real.acts.emit, mutated.acts.emit)
    else:
        assert _same(real, mutated)


# --- the compiled trajectory, structurally -----------------------------------------------------------------

# ``mx.compile`` merges identical sub-expressions (CSE), so the eager read set above is not the compiled one:
# a channel expression that repeats a trajectory sub-expression (the decode's ``tanh(y_turn)``, the solvency
# cancel of the thrust, ...) reads only listed arrays when traced eagerly, but in the compiled step it is the
# same node as the trajectory's, and so it makes an intermediate of a fused float kernel a kernel output (or
# splits the kernel). This compares the fused kernels of the compiled step with observe off and on.
#
# The trajectory graph is the ancestors of the new state and ``out.base()``, exported after one call of the
# compiled step and before anything is evaluated (the export is of the compiled tape, ``Compiled*`` labels
# being fused kernels, not of the JIT cache). A primitive's signature is its label (the operations fused into
# it) and its number of outputs: a kernel whose arithmetic is regrouped, split or given an output has another
# one, and a kernel whose inputs come from a kernel that changed has not. The channel may change only kernels
# made of comparisons, selects, bit operations and casts: they cannot round, so a regrouping of them cannot
# change a result on any device. Today these are exactly three groups, the same in 2D and 3D, f32 and
# f16: the ``HIT`` bit (``Combat.hit`` becomes a kernel output, so the ``ATTACK_HITS`` counter's kernel splits
# before its cast; G1), the ``KILLED`` bit (its comparison is the ``DEATHS_KILLED`` counter's own, so CSE
# merges the two, on purpose, and splits the counter's kernel before its cast) and the post-S8 ``bond_target``
# (the select kernel of the S9 partner gains a second output).
_EXACT_OPS = (
    "Compiled",
    "Broadcast",
    "GreaterEqual",
    "Greater",
    "Equal",
    "Select",
    "BitwiseAnd",
    "AsType",
)
_EXACT_LABEL = re.compile("(?:" + "|".join(_EXACT_OPS) + ")+")


def _kernel_signatures(path: Path) -> collections.Counter[tuple[str, int]]:
    """The multiset of (label, number of outputs) of the primitives of an exported graph. Neither the inputs
    of a kernel nor the output a consumer reads are part of it: a kernel is changed by what is fused into it
    and by what it has to write, not by where its inputs come from."""
    labels, producer, _ = _parse_dot(path)
    n_out = collections.Counter(producer.values())
    return collections.Counter((label, n_out[p]) for p, label in labels.items())


def _compiled_kernels(cfg: Config, observe: bool, path: Path) -> collections.Counter:
    fn = make_step(cfg, compile=True, observe=observe)
    s = init_state(cfg)
    mx.eval(s)
    s1, out = fn(s)  # not evaluated: the arrays still carry the compiled tape
    mx.export_to_dot(str(path), *[v for _, v in tree_flatten((s1, out.base() if observe else out))])
    return _kernel_signatures(path)


def _compiled_difference(
    cfg: Config,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutate: Callable[[pytest.MonkeyPatch], None] | None = None,
) -> tuple[list[str], list[str]]:
    """``(inexact, differing)``: the labels of the kernels that differ between the compiled step with observe
    off and on (``mutate`` applied to the latter), those not made of the exact ops above and all of them."""
    off = _compiled_kernels(cfg, False, tmp_path / "off.dot")
    if mutate is not None:
        mutate(monkeypatch)
    on = _compiled_kernels(cfg, True, tmp_path / "on.dot")
    monkeypatch.undo()
    differing = sorted({sig[0] for sig in (off - on) + (on - off)})
    return [lab for lab in differing if not _EXACT_LABEL.fullmatch(lab)], differing


@pytest.mark.parametrize(("preset", "overrides"), CHANNEL_CONFIGS)
def test_the_channel_regroups_no_float_kernel_of_the_compiled_step(
    preset: str, overrides: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """§28.2, §29.2: with observe on, the fused kernels of the compiled step differ from those with observe
    off only in comparisons, selects, bit operations and casts (see above): no kernel with a
    float operation (a ``Tanh``, ``Multiply``, ``Divide``, ...) is regrouped, split or given an output. It
    is the guard that sees ``mx.compile``'s merging of identical sub-expressions (the eager test cannot)."""
    inexact, differing = _compiled_difference(
        load_config(preset=preset, overrides=overrides), monkeypatch, tmp_path
    )
    assert differing, "the channel changes no kernel at all: the comparison is vacuous"
    assert not inexact, f"the channel changes {len(inexact)} kernel(s) with other operations: {inexact}"


def _cse_mutant(expr: Callable[[Config, mx.array, mx.array, mx.array], mx.array]):
    """Mutant: a channel field depends, value-neutrally, on ``expr``, which repeats a sub-expression of the
    step from allowed arrays only (the field gains ``expr > 1e30``, never true for a bounded value)."""

    def mutate(monkeypatch: pytest.MonkeyPatch) -> None:
        orig = step_mod._acts

        def acts(f, sl, thrust0, y, alive0, cfg):
            a = orig(f, sl, thrust0, y, alive0, cfg)
            junk = (expr(cfg, f.costs.solvent, thrust0, y) > 1e30).astype(mx.int16)
            return a._replace(thrust=a.thrust + junk)

        monkeypatch.setattr(step_mod, "_acts", acts)

    return mutate


CSE_MUTANTS = [
    # actions.mask's cancel of the thrust: a read of the post-S8 float thrust, as G1's mutant, by CSE
    pytest.param(
        _cse_mutant(lambda cfg, solvent, thrust0, y: mx.where(solvent, thrust0, 0.0)), id="cse-thrust"
    ),
    # the decode of the turn, of the contraction and of the defence, from y (a listed tap)
    pytest.param(
        _cse_mutant(lambda cfg, solvent, thrust0, y: cfg.physics.turn_max * mx.tanh(y[:, layout.OUT_TURN])),
        id="cse-turn",
    ),
    pytest.param(
        _cse_mutant(lambda cfg, solvent, thrust0, y: mx.tanh(y[:, layout.OUT_CONTRACT])), id="cse-contract"
    ),
    pytest.param(
        _cse_mutant(lambda cfg, solvent, thrust0, y: mx.sigmoid(y[:, layout.OUT_DEFEND])), id="cse-defend"
    ),
]


@pytest.mark.parametrize("mutant", CSE_MUTANTS)
def test_the_cse_mutants_are_value_neutral(mutant: Callable[[pytest.MonkeyPatch], None]) -> None:
    """The premise of the next test: a mutated channel returns the same arrays as the real one."""
    cfg = load_config(preset="tiny", overrides=["brain.weight_dtype=float32"])
    s = init_state(cfg)
    mx.eval(s)
    _, real = step_mod.step(s, cfg, observe=True)
    mx.eval(real)
    with pytest.MonkeyPatch.context() as m:
        mutant(m)
        _, mutated = step_mod.step(s, cfg, observe=True)
        mx.eval(mutated)
    assert _same(real, mutated)


@pytest.mark.parametrize("mutant", [*CSE_MUTANTS, MUTANTS[0], MUTANTS[3]])
@pytest.mark.parametrize("overrides", [["brain.weight_dtype=float32"], []], ids=["f32", "f16"])
def test_the_compiled_probe_catches_value_neutral_breaks(
    mutant: Callable[[pytest.MonkeyPatch], None],
    overrides: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Every mutant leaves the trajectory unchanged (the field mutant changes that one field) and splits, or
    gives an output to, a fused float kernel of the compiled step. The first four read only listed arrays when
    traced eagerly, which is why the eager test misses them; the last two are caught there too."""
    cfg = load_config(preset="tiny", overrides=overrides)
    inexact, _ = _compiled_difference(cfg, monkeypatch, tmp_path, mutate=mutant)
    assert inexact


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
def test_runner_files_identical_except_observers(
    d3: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """60 steps with and without observers: the checkpoints' state digests, the event chunks and the stored
    rows are equal; the only differences are the ``observers`` block of every row of the observing run and its
    ``observers/`` files (the sidecars and the social log, whose records are at 15, 30, 45 and 60), which the
    other run does not have at all."""
    social_every(monkeypatch, 15)
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
    assert all(r["observers"]["intent_version"] == 2 for r in on.metrics_rows())
    assert not off.observers_dir.exists() and off.observer_files() == []
    assert [st for st, _ in on.observer_files()] == [0, 30, 60]
    assert [f for f in _files(off) if f not in _files(on)] == []  # nothing else is missing from the on run
    assert [f for f in _files(on) if f not in _files(off)] == [
        "observers/social.jsonl",
        *[f"observers/step_{st:010d}.npz" for st in (0, 30, 60)],
    ]
    assert [r["step"] for r in on.social_records()] == [15, 30, 45, 60] and off.social_records() == []
    # the stored config differs in game.observe only
    c_off, c_on = (json.loads((rd.path / "config.json").read_text()) for rd in (off, on))
    default_game = config_to_dict(Config())["game"]  # {"observe": False, "house_rate": 1/2048}
    assert c_off.pop("game") == default_game
    assert c_on.pop("game") == {**default_game, "observe": True}  # the on-run's only game override
    assert c_off == c_on


def test_director_log_unchanged_by_observers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    social_every(monkeypatch, 15)
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
    assert [r["step"] for r in on.social_records()] == [15, 30, 45, 60] and off.social_records() == []
