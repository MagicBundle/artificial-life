"""The step hooks of the extension mechanism against the stub (DESIGN §43.1): ``inputs`` with the sensing
context, ``decode``/``mask`` into ``_Flows.ext``, ``due`` with the brain outputs, ``check``, the per-agent
birth policy and the Dims accessor. Each assertion kills a mutant of the wiring in ``step.py``, ``ext`` or
``invariants``."""

from __future__ import annotations

import dataclasses

import mlx.core as mx
import numpy as np
import pytest

import ext_stub as stub
from alife import brain, ext, invariants, layout, lifecycle, rng
from alife.config import apply_overrides, dims, load_config
from alife.state import LedgerIdx, init_state, to_numpy
from alife.step import make_step

F32 = ["brain.weight_dtype=float32"]


def _base(nd: int):
    return load_config(preset="tiny" if nd == 2 else "tiny3d", overrides=F32)


def _install(monkeypatch: pytest.MonkeyPatch, **kw) -> dict:
    spy: dict = {}
    reg = tuple(stub.spec(n, spy=spy, **kw) for n in stub.ALL)
    monkeypatch.setattr(ext, "REGISTRY", reg)
    return spy


def _pad_to(off, on_cfg, off_cfg):
    """The ``off`` state as a state of ``on_cfg``: base fields copied, the new W1 rows / W2 columns zero (what
    the PR 0b migration will do), so the brain computes the very same base outputs."""
    don, doff = dims(on_cfg), dims(off_cfg)
    s_on = init_state(on_cfg)
    base = doff.lay.D_IN
    extra_in, extra_out = don.lay.D_IN - base, don.lay.D_OUT - doff.lay.D_OUT

    def w1(a):  # (..., rows, H): ext rows between the base inputs and the recurrent rows
        z = mx.zeros((*a.shape[:-2], extra_in, a.shape[-1]), a.dtype)
        return mx.concatenate([a[..., :base, :], z, a[..., base:, :]], axis=-2)

    def w2(a):
        return mx.concatenate([a, mx.zeros((*a.shape[:-1], extra_out), a.dtype)], axis=-1)

    fields = {n: getattr(off, n) for n in type(off)._fields}
    fields.update(W1=w1(off.W1), W2=w2(off.W2), pend_W1=w1(off.pend_W1), pend_W2=w2(off.pend_W2))
    return s_on._replace(**fields)


@pytest.mark.parametrize("nd", [2, 3])
def test_the_stub_due_is_booked_in_heat_action_exactly(nd: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """2D (``pay_costs``) and 3D (``pay_costs3``): HEAT_ACTION with the stub on minus off is the stub due."""
    spy = _install(monkeypatch)
    off_cfg = _base(nd)
    on_cfg = stub.stub_cfg(off_cfg, "comm")
    s_off = init_state(off_cfg)
    s_on = _pad_to(s_off, on_cfg, off_cfg)
    _, out_off = make_step(off_cfg, compile=False)(s_off)
    spy.clear()
    _, out_on = make_step(on_cfg, compile=False)(s_on)
    mx.eval(out_off, out_on)
    (due,) = spy["due"]
    expect = int(to_numpy(due)[np.array(s_off.alive)].sum())
    assert expect > 2 * on_cfg.population.n_init  # the y and act terms contribute
    lo, ln = to_numpy(out_off.ledger), to_numpy(out_on.ledger)
    assert int(ln[LedgerIdx.HEAT_ACTION] - lo[LedgerIdx.HEAT_ACTION]) == expect
    for i in LedgerIdx:  # nothing else moves in the first step: the padded brain computes the same outputs
        if i != LedgerIdx.HEAT_ACTION:
            assert ln[i] == lo[i], i.name
    assert np.array_equal(
        to_numpy(out_off.counters)[: 27 + (3 if nd == 3 else 0)],
        to_numpy(out_on.counters)[: 27 + (3 if nd == 3 else 0)],
    )


@pytest.mark.parametrize("nd", [2, 3])
def test_the_stub_inputs_reach_x(nd: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """The first column of each tail block is the crowd input rebuilt from ``ctx.desc``; the rest is zero."""
    _install(monkeypatch, due=False)
    cfg = stub.stub_cfg(_base(nd), *stub.WITHOUT_ARMS)
    lay = dims(cfg).lay
    seen: dict = {}
    real = brain.forward

    def spy_forward(W1, W2, x, h, c):
        seen["x"] = x
        return real(W1, W2, x, h, c)

    monkeypatch.setattr(brain, "forward", spy_forward)
    s = init_state(cfg)
    for _ in range(3):  # the crowd input is non-trivial once agents have met
        make_step(cfg, compile=False)(s)
    x = np.array(seen["x"])
    crowd = x[:, layout.IN_CROWD]
    assert crowd.max() > 0 and x.shape[1] == lay.D_IN
    for start, width in (
        (lay.IN_MATERIAL, 1), (lay.IN_ROLE, 1), (lay.IN_BOND_ROLE, 4), (lay.IN_BOND_KIN, 4),
        (lay.IN_BOND_FLOW, 4), (lay.IN_VOICE, 4), (lay.IN_VOICE_BEARING, dims(cfg).nd), (lay.IN_FARM, 1),
    ):  # fmt: skip
        assert start is not None
        first = start in (lay.IN_MATERIAL, lay.IN_ROLE, lay.IN_VOICE, lay.IN_FARM)
        if first:
            assert np.array_equal(x[:, start], crowd), start
            assert not x[:, start + 1 : start + width].any()
        else:
            assert not x[:, start : start + width].any()


@pytest.mark.parametrize("starve", [False, True])
def test_decode_mask_and_flows_ext(starve: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """``decode`` sees y, ``mask`` applies solvency, ``_Flows.ext`` reaches the counters."""
    _install(monkeypatch, due=False)
    cfg = stub.stub_cfg(_base(2), "structures")
    s = init_state(cfg)
    if starve:  # nobody can pay the basal cost: every agent is insolvent
        s = s._replace(energy=mx.zeros_like(s.energy))
    seen: dict = {}
    real = brain.forward

    def spy_forward(W1, W2, x, h, c):
        out = real(W1, W2, x, h, c)
        seen["y"] = out[1]
        return out

    monkeypatch.setattr(brain, "forward", spy_forward)
    _, out = make_step(cfg, compile=False)(s)
    mx.eval(out)
    flagged = int(to_numpy(out.counters)[27])  # BUILD_UNITS: the first counter of the structures block
    unmasked = int(np.array(s.alive & (seen["y"][:, 0] > 0)).sum())
    assert unmasked > 0
    assert flagged == (0 if starve else unmasked)


def test_a_failing_stub_check_is_caught_by_full_check(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = stub.stub_cfg(_base(2), "bonds")
    monkeypatch.setattr(ext, "REGISTRY", tuple(stub.spec(n) for n in stub.ALL))  # before dims() caches cfg
    s = init_state(cfg)
    assert invariants.full_check(s, cfg) == []
    monkeypatch.setattr(ext, "REGISTRY", tuple(stub.spec(n, check_fail=True) for n in stub.ALL))
    problems = invariants.full_check(s, cfg)
    assert problems == ["stub bonds: broken on purpose"]
    assert invariants.full_check(init_state(_base(2)), _base(2)) == []  # a feature that is off is not checked


def test_dims_accessor_for_extension_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = dataclasses.replace(stub.spec("comm"), dims=lambda cfg: {"voice_w": 4, "hear_range": 7})
    monkeypatch.setattr(ext, "REGISTRY", (spec,))
    cfg = stub.stub_cfg(apply_overrides(_base(2), ["run.seed=77"]), "comm")  # dims() is cached per config
    d = dims(cfg)
    assert d.ext("comm").hear_range == 7 and d.ext("comm").voice_w == 4 and (d.voice_w, d.msg_w) == (4, 6)
    with pytest.raises(KeyError):
        dims(_base(2)).ext("comm")
    with pytest.raises(KeyError):
        dims(stub.stub_cfg(_base(2), "niche")).ext("comm")


def _reproduce(cfg, s, imm: int):
    ks = rng.step_keys(s.key)[1]
    req = s.alive & (mx.arange(s.alive.shape[0]) < 4)
    lanes = lifecycle.allocate_lanes(
        req,
        s.alive,
        ks[rng.Stream.BIRTH_PRIO],
        cfg,
        s.step,
        imm_request=mx.array(imm, mx.int32) if imm else None,
    )
    rp = lifecycle.reproduce(s, lanes, mx.zeros(s.alive.shape, mx.bool_), ks, cfg)
    mx.eval(rp.state)
    return lanes, rp


@pytest.mark.parametrize("nd", [2, 3])
def test_birth_policy_of_per_agent_extension_fields(nd: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """``on_birth``: "rest" resets (role), "inherit" copies the parent's row for a birth and resets an
    immigrant (arms); non-agent fields (mat) are untouched; old rows are untouched."""
    _install(monkeypatch)
    base = _base(nd)
    cfg = stub.stub_cfg(
        load_config(preset="tiny" if nd == 2 else "tiny3d", overrides=[*F32, "director.enabled=true"]),
        "arms",
        "structures",
        "bonds",
    )
    n = dims(cfg).n
    s = init_state(cfg)
    s = s._replace(
        energy=mx.full(s.energy.shape, 200_000, mx.int32),
        arms=mx.arange(1, n + 1, dtype=mx.float32)[:, None] * mx.ones((1, 6), mx.float32),
        role=mx.full((n,), 3, mx.int8),
        role_dwell=mx.full((n,), 9, mx.int16),
        bond_flow=mx.full((n, 4), 5, mx.int16),
        mat=mx.full(s.mat.shape, 7, mx.int32),
    )
    lanes, rp = _reproduce(cfg, s, imm=2)
    out = rp.state
    kind = np.array(lanes.kind)
    births, imms = kind == lifecycle.LaneKind.BIRTH.value, kind == lifecycle.LaneKind.IMMIGRANT.value
    assert births.sum() >= 2 and imms.sum() == 2
    slot, parent = np.array(lanes.slot), np.array(lanes.parent)
    arms, role = np.array(out.arms), np.array(out.role)
    for lane in np.where(births)[0]:
        assert (arms[slot[lane]] == parent[lane] + 1).all()  # inherited from the parent's pre-birth row
        assert (
            role[slot[lane]] == 0
            and np.array(out.role_dwell)[slot[lane]] == 0
            and not np.array(out.bond_flow)[slot[lane]].any()
        )
    for lane in np.where(imms)[0]:
        assert not arms[slot[lane]].any() and role[slot[lane]] == 0  # an immigrant starts fresh
    new = np.zeros(n, bool)
    new[slot[births | imms]] = True
    assert (role[~new] == 3).all() and (arms[~new][:, 0] == np.arange(1, n + 1)[~new]).all()
    assert (np.array(out.mat) == 7).all()
    assert base is not None


def test_a_hook_birth_policy_gets_the_lanes(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def hook(old, rest, bctx):
        seen["ctx"] = bctx
        return mx.where(bctx.newborn[:, None], 42.0, old)

    _install(monkeypatch)
    arms = stub.spec("arms")
    fs = dataclasses.replace(arms.state_fields(None, 2)[0], on_birth=hook)
    monkeypatch.setattr(ext, "REGISTRY", (dataclasses.replace(arms, state_fields=lambda cfg, nd: (fs,)),))
    cfg = stub.stub_cfg(_base(2), "arms")
    s = init_state(cfg)._replace(energy=mx.full((dims(cfg).n,), 200_000, mx.int32))
    _, rp = _reproduce(cfg, s, imm=0)
    a = np.array(rp.state.arms)
    new = np.array(rp.state.alive) & ~np.array(s.alive)
    assert new.sum() >= 1 and (a[new] == 42.0).all() and (a[~new] == 0).all()
    assert isinstance(seen["ctx"], ext.BirthCtx)
    with pytest.raises(ValueError, match="on_birth"):
        lifecycle._on_birth(dataclasses.replace(fs, on_birth="nonsense"), s.arms, cfg, seen["ctx"], None)
