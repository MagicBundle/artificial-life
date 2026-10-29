"""Counter blocks, ActFlag bits and the observer channel of the extension mechanism (DESIGN §43.4, §43.5)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import ext_stub as stub
from alife import ext, metrics, state
from alife.config import apply_overrides, load_config
from alife.state import CounterIdx, counter_blocks, counter_index, n_counters, state_digest
from alife.step import ActFlag, Acts, ExtActs, ExtFlows, make_step

F32 = ["brain.weight_dtype=float32"]


@pytest.fixture(autouse=True)
def _registry(monkeypatch: pytest.MonkeyPatch) -> None:
    stub.install(monkeypatch)


def _cfg(nd: int, names=(), *over: str):
    cfg = load_config(preset="tiny" if nd == 2 else "tiny3d", overrides=[*F32, *over])
    return stub.stub_cfg(cfg, *names) if names else cfg


def test_base_counter_indices_do_not_move() -> None:
    assert (
        n_counters(2) == 27
        and n_counters(3) == 30
        and n_counters(2, True) == 28
        and n_counters(3, True) == 31
    )
    assert state.storm_counter_index(2) == 27 and state.storm_counter_index(3) == 30
    for nd in (2, 3):
        for names in ((), stub.WITHOUT_ARMS):
            cfg = _cfg(nd, names, "combat.protect_floor=0.1")
            assert counter_index(cfg, "N_ALIVE") == 0 and counter_index(cfg, "DEATHS_CATASTROPHE") == 26
    cfg = _cfg(3, (), "combat.protect_floor=0.1")
    assert counter_index(cfg, "GROUNDED") == 29 and counter_blocks(cfg) == (("protect", ("PROT_CAPPED",)),)
    assert counter_index(cfg, "PROT_CAPPED") == 30


def test_blocks_follow_the_documented_order_and_names_are_unique() -> None:
    cfg = _cfg(2, ext.EXTENSIONS, "weather.enabled=true", "world.size=320", "weather.u_max=0.2")
    blocks = counter_blocks(cfg)
    assert [b for b, _ in blocks] == list(ext.EXTENSIONS)  # protect absent at floor 0
    cfg_p = apply_overrides(cfg, ["combat.protect_floor=0.1"])
    names = [n for _, ns in counter_blocks(cfg_p) for n in ns]
    assert (
        next(b for b, _ in counter_blocks(cfg_p)) == "protect"
        and len(names) == len(set(names)) == 1 + 5 + 10 + 5 + 2 + 5
    )
    base = n_counters(2, True)  # 28: 27 + DEATHS_STORM
    assert counter_index(cfg_p, "DEATHS_STORM") == 27 and counter_index(cfg_p, "PROT_CAPPED") == base
    assert (
        counter_index(cfg_p, "ARMS_ATK_SUM") == base + 1 and counter_index(cfg_p, "BUILD_UNITS") == base + 6
    )
    assert counter_index(cfg_p, "SOIL_TOTAL") == n_counters(2, True, counter_blocks(cfg_p)) - 1
    with pytest.raises(KeyError):
        counter_index(_cfg(2), "ARMS_ATK_SUM")  # a feature that is off has no counters
    assert not set(names) & set(CounterIdx.__members__)  # globally unique


@pytest.mark.parametrize("nd", [2, 3])
def test_a_compiled_step_emits_the_blocks(nd: int) -> None:
    cfg = _cfg(nd, stub.WITHOUT_ARMS, "combat.protect_floor=0.1")
    s, fn = state.init_state(cfg), make_step(cfg)
    for _ in range(3):
        s, out = fn(s)
    mx.eval(s, out)
    c = np.array(out.counters)
    assert c.shape == (n_counters(nd, False, counter_blocks(cfg)),) and c.dtype == np.int32
    n_alive = c[CounterIdx.N_ALIVE]
    assert (
        n_alive > 0 and c[counter_index(cfg, "PROT_CAPPED")] == 0
    )  # no protection factor is supplied in PR 0a
    for name in ("structures", "bonds", "comm", "niche"):
        for j, cname in enumerate(stub.COUNTERS[name][1:], start=1):
            assert c[counter_index(cfg, cname)] == n_alive * (j + 1), cname
        first = c[counter_index(cfg, stub.COUNTERS[name][0])]  # the masked decode's flagged rows
        assert 0 < first <= n_alive, name
    # the metrics row takes the blocks as an argument and reports prot_capped_frac
    ledgers = np.zeros((1, 16), np.int64)
    row = metrics.interval_row(10, c[None], ledgers, {}, {}, nd=nd, blocks=counter_blocks(cfg))
    assert row["prot_capped_frac"] is None or row["prot_capped_frac"] == 0
    with pytest.raises(ValueError, match="columns"):
        metrics.interval_row(10, c[None], ledgers, {}, {}, nd=nd)  # the base width is refused, not reshaped


def test_metrics_row_without_blocks_is_unchanged() -> None:
    cfg = _cfg(2)
    s, fn = state.init_state(cfg), make_step(cfg)
    s, out = fn(s)
    row = metrics.interval_row(1, np.array(out.counters)[None], np.array(out.ledger)[None], {}, {})
    assert "prot_capped_frac" not in row and counter_blocks(cfg) == ()


@pytest.mark.parametrize("nd", [2, 3])
def test_observer_channel_flags_are_uint16_and_inert(nd: int) -> None:
    """``Acts.flags`` is uint16 (observer-only); observing changes no state; ``Acts.ext`` is empty."""
    for names in ((), stub.WITHOUT_ARMS):
        cfg = _cfg(nd, names)
        s0 = state.init_state(cfg)
        plain, observed = make_step(cfg), make_step(cfg, observe=True)
        a, out_a = plain(s0)
        b, out_b = observed(s0)
        for _ in range(8):
            a, out_a = plain(a)
            b, out_b = observed(b)
        mx.eval(a, b, out_a.counters, out_b.acts.flags)
        assert state_digest(a) == state_digest(b) and np.array_equal(out_a.counters, out_b.counters)
        acts = out_b.acts
        assert isinstance(acts, Acts) and acts.flags.dtype == mx.uint16
        assert acts.ext is None
        flags = np.array(acts.flags).astype(np.int64)
        assert flags.max() < 128  # no extension bit is set by PR 0a
    assert ExtFlows()._fields == ExtActs()._fields == ext.EXTENSIONS
    assert (ActFlag.KILLED | ActFlag.FARMING).value == 64 + 2048
