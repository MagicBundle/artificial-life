"""Conditional extension state fields (DESIGN §43.2): the generated state type, its schema, the checkpoint
fill and refusal rules, the digests, and a compiled step through the generated NamedTuple."""

from __future__ import annotations

import itertools

import mlx.core as mx
import numpy as np
import pytest

import ext_stub as stub
from alife import ext, state
from alife.config import dims, load_config
from alife.state import (
    AGENT_FIELDS,
    AGENT_FIELDS_3D,
    State,
    State3,
    agent_fields,
    expected_layout,
    init_state,
    is_3d_state,
    state_cls,
    state_digest,
    state_from_arrays,
    state_to_arrays,
)
from alife.step import make_step

BASE = {2: State._fields, 3: State3._fields}
SUBSETS = [c for r in range(6) for c in itertools.combinations(stub.ALL, r)]


@pytest.fixture(autouse=True)
def _registry(monkeypatch: pytest.MonkeyPatch) -> None:
    stub.install(monkeypatch)


def _cfg(nd: int, names, **kw):
    preset = "tiny" if nd == 2 else "tiny3d"
    return stub.stub_cfg(load_config(preset=preset, overrides=["brain.weight_dtype=float32"]), *names)


def test_feature_off_is_the_exact_class(monkeypatch: pytest.MonkeyPatch) -> None:
    for nd in (2, 3):
        cfg = _cfg(nd, ())
        assert state_cls(cfg) is (State3 if nd == 3 else State)
        assert agent_fields(cfg) == (AGENT_FIELDS_3D if nd == 3 else AGENT_FIELDS)
        s = init_state(cfg)
        assert type(s) is state_cls(cfg) and list(state_to_arrays(s)) == list(BASE[nd])
        assert set(expected_layout(cfg)) == set(BASE[nd]) and is_3d_state(s) == (nd == 3)
    # comm declares no field: a comm-only config is still the exact class
    assert state_cls(_cfg(2, ("comm",))) is State


@pytest.mark.parametrize("nd", [2, 3])
@pytest.mark.parametrize("names", SUBSETS, ids=lambda s: "+".join(s) or "none")
def test_state_type_for_every_subset(nd: int, names: tuple[str, ...]) -> None:
    cfg = _cfg(nd, names)
    cls = state_cls(cfg)
    fields = [f.name for spec in ext.active(cfg) for f in spec.state_fields(cfg, nd)]
    assert cls._fields == (*BASE[nd], *fields)  # appended after the whole base type, EXTENSIONS order
    assert agent_fields(cfg)[len(AGENT_FIELDS_3D if nd == 3 else AGENT_FIELDS) :] == tuple(
        f.name for spec in ext.active(cfg) for f in spec.state_fields(cfg, nd) if f.per_agent
    )
    want = expected_layout(cfg)
    assert list(want) == list(cls._fields)
    if names:
        assert cls is state_cls(cfg) and ((cls is not State and cls is not State3) or not fields)


@pytest.mark.parametrize("nd", [2, 3])
def test_init_round_trip_and_fill(nd: int) -> None:
    cfg = _cfg(nd, stub.ALL)
    s = init_state(cfg)
    assert isinstance(s, tuple) and type(s) is state_cls(cfg) and is_3d_state(s) == (nd == 3)
    want = expected_layout(cfg)
    for name, a in zip(type(s)._fields, s, strict=True):
        assert (tuple(a.shape), a.dtype) == want[name], name
    d = dims(cfg)
    assert s.arms.shape == (d.n, 6) and s.mat.shape == d.field_shape and s.bond_flow.dtype == mx.int16
    arrays = state_to_arrays(s)
    back = state_from_arrays(arrays, cfg)
    assert type(back) is type(s) and state_digest(back) == state_digest(s)
    assert s.env.regen_mult is s.env_regen_mult  # the read-only env view survives the generated class
    # a file that lacks only extension fields (a pre-feature checkpoint forked onto the feature) is completed
    ext_names = {f.name for f in ext.state_fields(cfg, nd)}
    older = {k: v for k, v in arrays.items() if k not in ext_names}
    filled = state_from_arrays(older, cfg)
    assert state_digest(filled) == state_digest(s)
    assert not np.any(np.array(filled.mat)) and not np.any(np.array(filled.arms))
    # a file with a field the config lacks (a fork turning a feature off) is refused with the extra message
    with pytest.raises(ValueError, match=r"extra=\['arms'"):
        state_from_arrays(arrays, _cfg(nd, ("structures",)))
    # a file missing a BASE field is still refused
    with pytest.raises(ValueError, match="missing"):
        state_from_arrays({k: v for k, v in arrays.items() if k != "energy"}, cfg)


def test_digests_when_a_feature_is_on() -> None:
    off, on = init_state(_cfg(2, ())), init_state(_cfg(2, ("structures",)))
    assert state_digest(off) != state_digest(on)  # a different type and a different W1 shape
    # the legacy digest hashes the frozen v0 names only; on a 2D extension state it still works
    assert len(state.legacy_state_digest(on)) == 64
    assert state.legacy_digest(init_state(_cfg(3, ("niche",)))) == state.legacy_state_digest_3d(
        init_state(_cfg(3, ("niche",)))
    )


@pytest.mark.parametrize("nd", [2, 3])
@pytest.mark.parametrize("compile", [True, False])
def test_a_step_through_the_generated_class(nd: int, compile: bool) -> None:
    """The compile probe: ``mx.compile`` accepts and returns the generated NamedTuple, keeps its type, and the
    extension fields pass through the step unchanged (no feature writes them in PR 0a)."""
    cfg = _cfg(nd, stub.WITHOUT_ARMS)
    fn = make_step(cfg, compile=compile)
    s = init_state(cfg)
    s1, out = fn(s)
    s2, _ = fn(s1)
    mx.eval(s2, out)
    assert type(s1) is type(s) is type(s2) and s2.step.item() == 2
    for name in ("mat", "soil", "role", "role_dwell", "bond_flow"):
        assert np.array_equal(np.array(getattr(s2, name)), np.array(getattr(s, name))), name
    assert int(np.array(s2.diag)) == 0
    assert out.counters.shape == (state.n_counters(nd, False, state.counter_blocks(cfg)),)
