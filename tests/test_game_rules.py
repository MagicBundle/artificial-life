"""§28.4 the archetype and intent rules (G1): thresholds at their edges, priority, warm-up, hysteresis, the
step clock, the match bits, and the pinned ``RuleSet`` (``INTENT_VERSION`` 1)."""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Sequence
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from alife import layout
from alife.config import Config, config_from_dict, config_to_dict, dims, load_config
from alife.game import rules
from alife.game.observe import OBS_CAP, ObsState, evaluate, init_obs, obs_in
from alife.game.rules import (
    ARCH_PRIORITY,
    ARCHETYPE_NAMES,
    EVAL_BIT,
    INTENT_NAMES,
    INTENT_VERSION,
    RULES_V1,
    VQ_MAX,
    Archetype,
    Intent,
    RuleSet,
    n_arch_cols,
)
from alife.state import init_state, to_numpy
from alife.step import ActFlag
from game_helpers import Driver, blank_acts, living, make_acts, make_oi

# Pinned at implementation time. Any edit to a threshold, window, period or field of RuleSet changes it, and
# must come with a new INTENT_VERSION and a new pin (the sidecar stores the digest).
RULES_V1_DIGEST = "156ad4963b84ffe9636e9ba4fdd7592632c669bf6e490687d8eb2b38b9aec009"

B = layout.B
GENERALIST_BIT = 1 << (Archetype.GENERALIST.value - 1)
RESTING_BIT = 1 << (Intent.RESTING.value - 1)


def np_(a: mx.array) -> np.ndarray:
    return to_numpy(a)


# --- the pinned rule set ------------------------------------------------------------------------------------


def test_rules_v1_digest_is_pinned() -> None:
    assert INTENT_VERSION == 1
    assert RuleSet() == RULES_V1
    assert RULES_V1.digest() == RULES_V1_DIGEST
    assert len(RULES_V1.digest()) == 64


def test_every_field_enters_the_digest() -> None:
    changed = {
        "arch_k": 5, "intent_k": 4, "every": 4, "hold": 3, "arch_warm": 191, "intent_warm": 23,
        "gatherer": (8, 10), "hunter": (8, 20), "provider": (2, 5), "trader": (2, 3),
        "nomad_speed": (513, 5), "nomad_persist": (10, 25), "settler_speed": (257, 10), "bud_window": 33,
        "hunt": 65, "flee_hurt": 9, "flee_thrust": 129, "defend": 129, "care": (2, 10),
        "bond_age": (25, 120), "signal": 129, "feed": (2, 2), "roam": (769, 10),
    }  # fmt: skip
    assert set(changed) == {f.name for f in dataclasses.fields(RuleSet)}
    digests = {RULES_V1.digest()}
    for name, value in changed.items():
        digests.add(dataclasses.replace(RULES_V1, **{name: value}).digest())
    assert len(digests) == len(changed) + 1


def test_ruleset_rejects_nonsense() -> None:
    for kw in ({"arch_k": 0}, {"intent_k": 31}, {"every": 0}, {"hold": 0}, {"arch_warm": -1},
               {"gatherer": (7, 0)}, {"hunter": (-1, 5)}, {"bond_age": (24, 24)}):  # fmt: skip
        with pytest.raises(ValueError, match="RuleSet"):
            RuleSet(**kw)


def _with_limits(preset: str, value: int | None = None) -> Config:
    cfg = load_config(preset=preset)
    if value is None:
        return cfg
    d = config_to_dict(cfg)
    d["energy"]["harvest_max"] = d["combat"]["attack_max"] = d["bonds"]["share_rate"] = value
    return config_from_dict(d)  # not validated: only the three bounds matter to RuleSet.check


@pytest.mark.parametrize("preset", ["tiny", "64k", "tiny3d", "64k3d"])
def test_ruleset_check_bounds_int32(preset: str) -> None:
    RULES_V1.check(_with_limits(preset))
    top = _with_limits(preset, (1 << 16) - 1)  # the validated extremes ("Contention headroom")
    RULES_V1.check(top)
    RuleSet(arch_k=12).check(top)  # 4·(2^16 − 1)·(2^12 + 1) < 2^31
    with pytest.raises(ValueError, match=r"arch\.(inflow|outflow|share).*k = 13"):
        RuleSet(arch_k=13).check(top)
    # the same k is fine at the default share_rate, until the quantised-velocity columns overflow
    RuleSet(arch_k=15).check(_with_limits(preset))
    with pytest.raises(ValueError, match=r"arch\.(speed|velocity).*k = 16"):
        RuleSet(arch_k=16).check(_with_limits(preset))
    with pytest.raises(ValueError, match=r"intent\.care.*k = 13"):
        RuleSet(intent_k=13).check(top)


INT32_MAX = (1 << 31) - 1


def _income_max(cfg: Config) -> int:
    return cfg.energy.harvest_max + cfg.combat.attack_max + B * cfg.bonds.share_rate


@pytest.mark.parametrize("preset", ["tiny", "64k3d"])
def test_ruleset_check_bounds_the_ratios(preset: str) -> None:
    """The ratio tests ``den·x >= num·y`` run in int32: a sweep RuleSet with large integers is refused (the
    same ratios as v1, scaled by 10^5, wrapped and mislabelled a Gatherer as not one)."""
    cfg = load_config(preset=preset)
    with pytest.raises(ValueError, match=r"ratio gatherer = \(700000, 1000000\)"):
        RuleSet(gatherer=(700000, 1000000), provider=(100000, 500000)).check(cfg)
    RuleSet(gatherer=(70, 100), provider=(10, 50)).check(cfg)  # the same ratios, small
    flow = B * cfg.bonds.share_rate
    largest_mean = {  # the largest compared mean of each int32 ratio test
        "gatherer": _income_max(cfg),
        "hunter": _income_max(cfg),
        "provider": _income_max(cfg),
        "trader": cfg.bonds.share_rate,
        "care": max(flow, cfg.energy.harvest_max),
        "feed": cfg.energy.harvest_max,
    }
    for name, x in largest_mean.items():
        limit = INT32_MAX // x
        dataclasses.replace(RULES_V1, **{name: (limit, 1)}).check(cfg)  # max(num, den)·X <= 2^31 − 1
        dataclasses.replace(RULES_V1, **{name: (1, limit)}).check(cfg)
        for pair in ((limit + 1, 1), (1, limit + 1)):  # numerator and denominator are both bounded
            with pytest.raises(ValueError, match=f"ratio {name} = "):
                dataclasses.replace(RULES_V1, **{name: pair}).check(cfg)
    # Roaming compares den·thrust (thrust <= 256) with the numerator itself
    dataclasses.replace(RULES_V1, roam=(INT32_MAX, 10)).check(cfg)
    dataclasses.replace(RULES_V1, roam=(768, (1 << 23) - 1)).check(cfg)
    with pytest.raises(ValueError, match=r"ratio roam = "):
        dataclasses.replace(RULES_V1, roam=(1 << 31, 10)).check(cfg)
    with pytest.raises(ValueError, match=r"ratio roam = "):
        dataclasses.replace(RULES_V1, roam=(768, 1 << 23)).check(cfg)  # 2^23·256 = 2^31


def test_ratio_at_the_check_limit_does_not_wrap() -> None:
    """The largest ratio ``check`` accepts compares correctly at the largest mean (no int32 wrap)."""
    cfg = load_config(preset="tiny")
    x = _income_max(cfg)
    den = INT32_MAX // x
    rs = dataclasses.replace(RULES_V1, gatherer=(den - 1, den))
    rs.check(cfg)
    m6 = _m6(2, harvest=x)  # all income is harvest: den·x >= (den − 1)·x
    assert bool(np_(rules.archetype_matches(m6, mx.zeros((1, B, 2), mx.int32), cfg, rs))[0, 0])
    with pytest.raises(ValueError, match="ratio gatherer"):
        dataclasses.replace(RULES_V1, gatherer=(den, den + 1)).check(cfg)


def test_ruleset_check_names_the_intent_column() -> None:
    with pytest.raises(ValueError, match=r"intent\.harvest.*k = 20"):
        RuleSet(intent_k=20).check(_with_limits("tiny", (1 << 16) - 1))


# --- archetypes ---------------------------------------------------------------------------------------------


def _m6(nd: int, **cols: Any) -> mx.array:
    """(1, 5+2nd) means: ``harvest``, ``gain``, ``inflow``, ``outflow``, ``speed`` and the velocity halves
    ``vpos`` and ``vneg`` (sequences of nd)."""
    row = np.zeros((1, n_arch_cols(nd)), np.int32)
    for name, col in zip(("harvest", "gain", "inflow", "outflow", "speed"), range(5), strict=True):
        row[0, col] = cols.pop(name, 0)
    row[0, 5 : 5 + nd] = cols.pop("vpos", (0,) * nd)
    row[0, 5 + nd :] = cols.pop("vneg", (0,) * nd)
    assert not cols, cols
    return mx.array(row)


def _arch(cfg: Config, nd: int = 2, slots: Sequence[tuple[int, int]] = (), **cols: Any) -> dict[str, bool]:
    m6s = np.zeros((1, B, 2), np.int32)
    for b, pair in enumerate(slots):
        m6s[0, b] = pair
    match = np_(rules.archetype_matches(_m6(nd, **cols), mx.array(m6s), cfg, RULES_V1))
    assert match.shape == (1, 6) and match.dtype == bool
    return dict(zip(ARCHETYPE_NAMES, match[0].tolist(), strict=False))


def test_archetype_rules_at_their_thresholds_gatherer_hunter_provider(tiny_cfg: Config) -> None:
    # Gatherer: I >= 1 and 10·m_h >= 7·I (I = h + gain + inflow)
    assert _arch(tiny_cfg, harvest=7, inflow=3)["gatherer"]  # 70 >= 70, inclusive
    assert not _arch(tiny_cfg, harvest=6, inflow=4)["gatherer"]  # 60 < 70
    assert _arch(tiny_cfg, harvest=1)["gatherer"]  # I = 1
    assert not _arch(tiny_cfg)["gatherer"]  # zero income
    # Hunter: m_g >= 1 and 20·m_g >= 7·I
    assert _arch(tiny_cfg, harvest=13, gain=7)["hunter"]  # I = 20: 140 >= 140
    assert not _arch(tiny_cfg, harvest=14, gain=6)["hunter"]  # 120 < 140
    assert _arch(tiny_cfg, gain=1)["hunter"]
    assert not _arch(tiny_cfg)["hunter"]  # zero income
    assert not _arch(tiny_cfg, harvest=20)["hunter"]  # no gain
    # Provider: m_out >= 1 and 5·m_out >= I
    assert _arch(tiny_cfg, inflow=10, outflow=2)["provider"]  # 10 >= 10
    assert not _arch(tiny_cfg, inflow=10, outflow=1)["provider"]  # 5 < 10
    assert _arch(tiny_cfg, inflow=5, outflow=1)["provider"]
    assert not _arch(tiny_cfg, inflow=10)["provider"]  # sends nothing
    assert not _arch(tiny_cfg)["provider"]
    # the three can hold together, except Gatherer and Hunter (0.7 + 0.35 > 1)
    both = _arch(tiny_cfg, harvest=14, inflow=1, outflow=3)
    assert both["gatherer"] and both["provider"] and not both["hunter"]


def test_archetype_rule_trader(tiny_cfg: Config) -> None:
    base = {"inflow": 10, "outflow": 2}  # a Provider (5·2 >= 10)
    assert _arch(tiny_cfg, slots=[(2, 4)], **base)["trader"]  # min 2, 2·2 >= 4
    assert not _arch(tiny_cfg, slots=[(2, 5)], **base)["trader"]  # 4 < 5
    assert _arch(tiny_cfg, slots=[(1, 1)], **base)["trader"]  # min 1
    assert not _arch(tiny_cfg, slots=[(0, 5)], **base)["trader"]  # min < 1
    assert not _arch(tiny_cfg, slots=[(0, 0)], **base)["trader"]
    assert _arch(tiny_cfg, slots=[(0, 0), (9, 5), (3, 6)], **base)["trader"]  # any slot qualifies
    # a Trader is first a Provider
    weak = {"inflow": 10, "outflow": 1}
    assert _arch(tiny_cfg, slots=[(2, 4)], **weak) == _arch(tiny_cfg, **weak)
    assert not _arch(tiny_cfg, slots=[(2, 4)], **weak)["trader"]
    assert _arch(tiny_cfg, slots=[(2, 4)], **base)[
        "provider"
    ]  # the match bit keeps Provider where Trader holds


@pytest.mark.parametrize("nd", [2, 3])
def test_archetype_rules_nomad_and_settler(tiny_cfg: Config, nd: int) -> None:
    def vec(**comp: int) -> tuple[int, ...]:
        return tuple(comp.get(c, 0) for c in "xyz"[:nd])

    # speed edge: 5·m_s >= 512
    assert _arch(tiny_cfg, nd, speed=103, vpos=vec(x=103))["nomad"]
    assert not _arch(tiny_cfg, nd, speed=102, vpos=vec(x=102))["nomad"]
    # persistence edge: 25·|m⁺ − m⁻|² >= 9·m_s² (|mean velocity| / speed >= 0.6)
    assert _arch(tiny_cfg, nd, speed=200, vpos=vec(x=120))["nomad"]  # 360000 >= 360000
    assert not _arch(tiny_cfg, nd, speed=200, vpos=vec(x=119))["nomad"]
    assert _arch(tiny_cfg, nd, speed=200, vpos=vec(x=85, y=85))["nomad"]  # |v|² = 14450: 361250 >= 360000
    assert _arch(tiny_cfg, nd, speed=200, vneg=vec(y=120))["nomad"]  # the negative half counts
    assert not _arch(tiny_cfg, nd, speed=200, vpos=vec(x=200), vneg=vec(x=200))["nomad"]  # back and forth
    assert not _arch(tiny_cfg, nd, speed=500)["nomad"]  # fast but no net displacement
    if nd == 3:  # the third component counts
        assert _arch(tiny_cfg, nd, speed=200, vpos=vec(z=120))["nomad"]
        assert not _arch(tiny_cfg, nd, speed=200, vpos=vec(z=119))["nomad"]
        assert _arch(tiny_cfg, nd, speed=200, vneg=vec(z=120))["nomad"]
    # Settler: 10·m_s < 256
    assert _arch(tiny_cfg, nd, speed=25)["settler"]
    assert not _arch(tiny_cfg, nd, speed=26)["settler"]
    assert _arch(tiny_cfg, nd)["settler"]  # at rest, with zero income too
    assert not _arch(tiny_cfg, nd, speed=200, vpos=vec(x=200))["settler"]
    # no archetype but Settler/Nomad depends on movement
    assert not any(_arch(tiny_cfg, nd, speed=60, vpos=vec(x=60)).values())  # a Generalist row


def _reference_labels(row: Sequence[bool]) -> tuple[int, int]:
    match = dict(zip(ARCHETYPE_NAMES[:6], row, strict=True))
    if match["trader"]:
        match["provider"] = False
    hits = [a.value for a in ARCH_PRIORITY if match[a.name.lower()]]
    return (hits[0] if hits else Archetype.GENERALIST.value), (hits[1] if len(hits) > 1 else 0)


def test_archetype_priority_and_secondary() -> None:
    combos = np.array(list(itertools.product([False, True], repeat=6)))
    primary, secondary = (np_(a) for a in rules.archetype_labels(mx.array(combos)))
    assert primary.dtype == np.int32 and secondary.dtype == np.int32
    for row, p, s in zip(combos, primary, secondary, strict=True):
        assert (p, s) == _reference_labels(row.tolist()), row
    names = {a.name.lower(): a.value for a in Archetype}

    def one(*on: str) -> tuple[int, int]:
        row = np.array([[name in on for name in ARCHETYPE_NAMES[:6]]])
        p, s = rules.archetype_labels(mx.array(row))
        return int(np_(p)[0]), int(np_(s)[0])

    # Trader outranks Provider and is never a Provider secondary: Trader + Provider → (Trader, NONE)
    assert one("trader", "provider") == (names["trader"], 0)
    assert one("trader", "provider", "gatherer") == (names["trader"], names["gatherer"])
    assert one("provider", "gatherer") == (names["provider"], names["gatherer"])
    assert one("hunter", "gatherer") == (names["hunter"], names["gatherer"])  # income before movement
    assert one("gatherer", "settler") == (names["gatherer"], names["settler"])
    assert one("provider", "hunter") == (names["provider"], names["hunter"])
    assert one("gatherer", "nomad") == (names["gatherer"], names["nomad"])
    assert one("nomad", "settler") == (names["nomad"], names["settler"])
    assert one("nomad") == (names["nomad"], 0)
    # no rule matched: Generalist, with no secondary
    assert one() == (names["generalist"], 0)
    assert (primary == Archetype.GENERALIST.value).sum() == 1 and secondary[primary == 7].sum() == 0


# --- intents ------------------------------------------------------------------------------------------------


def _intent(
    cfg: Config, *, since_bud: int = OBS_CAP, ages: Sequence[int] = (), kin: Sequence[bool] = (), **cols: int
) -> dict[str, bool]:
    """The nine predicates for a row with the intent means ``cols`` (lower-case ``IntentCol`` names)."""
    m3 = np.zeros((1, rules.N_INTENT_COLS), np.int32)
    for name, value in cols.items():
        m3[0, rules.IntentCol[name.upper()].value] = value
    age = np.zeros((1, B), np.int32)
    age[0, : len(ages)] = ages
    k = np.zeros((1, B), bool)
    k[0, : len(kin)] = kin
    match = np_(
        rules.intent_matches(
            mx.array(m3), mx.array([since_bud], dtype=mx.int32), mx.array(age), mx.array(k), cfg, RULES_V1
        )
    )
    assert match.shape == (1, 9) and match.dtype == bool
    return dict(zip(INTENT_NAMES, match[0].tolist(), strict=False))


def test_intent_predicates_and_order(tiny_cfg: Config) -> None:
    c, hm = tiny_cfg, tiny_cfg.energy.harvest_max
    assert hm == 400
    assert _intent(c, since_bud=31)["budding"] and not _intent(c, since_bud=32)["budding"]
    assert _intent(c, since_bud=0)["budding"]
    assert _intent(c, attack=64)["hunting"] and not _intent(c, attack=63)["hunting"]
    assert _intent(c, hurt=8, thrust=128)["fleeing"]
    assert not _intent(c, hurt=7, thrust=128)["fleeing"] and not _intent(c, hurt=8, thrust=127)["fleeing"]
    assert _intent(c, defend=128)["defending"] and not _intent(c, defend=127)["defending"]
    assert _intent(c, care=40)["caring"] and not _intent(c, care=39)["caring"]  # 10·m >= harvest_max
    assert not _intent(c)["caring"]
    assert _intent(c, emit=128)["signalling"] and not _intent(c, emit=127)["signalling"]
    assert _intent(c, harvest=200)["feeding"] and not _intent(c, harvest=199)["feeding"]  # 2·m >= harvest_max
    assert not _intent(c)["feeding"]
    assert _intent(c, thrust=77)["roaming"] and not _intent(c, thrust=76)["roaming"]  # 10·m >= 768
    # Bonding: a non-kin slot with 24 <= age < 120; kin slots and other ages are ignored
    assert _intent(c, ages=[24])["bonding"] and _intent(c, ages=[119])["bonding"]
    assert not _intent(c, ages=[23])["bonding"] and not _intent(c, ages=[120])["bonding"]
    assert not _intent(c, ages=[OBS_CAP])["bonding"]  # a bond that predates the observation is never "new"
    assert not _intent(c, ages=[50], kin=[True])["bonding"]
    assert _intent(c, ages=[50, 50], kin=[True, False])["bonding"]  # a second, non-kin slot
    assert _intent(c, ages=[5, 5, 5, 60])["bonding"]
    assert not any(_intent(c).values())  # nothing at rest
    # the empty-harvest edge: the thresholds scale with harvest_max, but a zero mean never matches
    zero = config_from_dict(
        {**config_to_dict(c), "energy": {**config_to_dict(c)["energy"], "harvest_max": 0}}
    )
    assert not _intent(zero)["feeding"] and not _intent(zero)["caring"]


def test_first_intent_follows_the_order() -> None:
    first = lambda row: int(np_(rules.first_intent(mx.array([row])))[0])  # noqa: E731
    assert first([False] * 9) == Intent.RESTING.value
    for i in range(9):  # predicate i and every later one hold: i is the first
        assert first([False] * i + [True] * (9 - i)) == i + 1
    assert first([True] * 9) == Intent.BUDDING.value
    assert first([False] * 4 + [True] + [False] * 4) == Intent.CARING.value
    rng = np.random.default_rng(0)
    rows = rng.random((200, 9)) < 0.3
    got = np_(rules.first_intent(mx.array(rows)))
    want = [next((j + 1 for j in range(9) if r[j]), Intent.RESTING.value) for r in rows]
    assert got.tolist() == want and got.dtype == np.int32


def test_hysteresis() -> None:
    hold = RULES_V1.hold

    def run(raws: Sequence[int], label: int = 0, cand: int = 0, n: int = 0) -> list[tuple[int, int, int]]:
        trace = []
        lab, cd, cn = (mx.array([v], dtype=mx.int32) for v in (label, cand, n))
        for raw in raws:
            lab, cd, cn = rules.hysteresis(lab, cd, cn, mx.array([raw], dtype=mx.int32), hold)
            trace.append((int(np_(lab)[0]), int(np_(cd)[0]), int(np_(cn)[0])))
        return trace

    H, F, R, N = Intent.HUNTING.value, Intent.FEEDING.value, Intent.RESTING.value, Intent.NONE.value
    # four consecutive raw labels switch the label (the first label too), and the candidate then equals it
    assert run([R] * 5) == [(N, R, 1), (N, R, 2), (N, R, 3), (R, R, 0), (R, R, 0)]
    # an interruption restarts the count
    assert run([H, H, H, F, H, H, H, H], label=R) == [
        (R, H, 1), (R, H, 2), (R, H, 3), (R, F, 1), (R, H, 1), (R, H, 2), (R, H, 3), (H, H, 0),
    ]  # fmt: skip
    # a return to the current label clears the candidate
    assert run([H, H, R, H], label=R) == [(R, H, 1), (R, H, 2), (R, R, 0), (R, H, 1)]
    # NONE clears at once: label, candidate and count
    assert run([H, H, N], label=R) == [(R, H, 1), (R, H, 2), (N, N, 0)]
    assert run([N], label=F, cand=H, n=3) == [(N, N, 0)]
    # the label survives a single deviant evaluation
    assert run([F, R, R, R], label=R) == [(R, F, 1), (R, R, 0), (R, R, 0), (R, R, 0)]
    # a different hold
    lab, cd, cn = (mx.array([v], dtype=mx.int32) for v in (R, 0, 0))
    lab, cd, cn = rules.hysteresis(lab, cd, cn, mx.array([H], dtype=mx.int32), 1)
    assert int(np_(lab)[0]) == H and int(np_(cn)[0]) == 0


# --- evaluate: warm-up, bits, clock -------------------------------------------------------------------------


def _obs(cfg: Config, n_obs: int, **cols: dict[str, int]) -> ObsState:
    """An observer state with every row at ``n_obs`` observed steps and the given mean columns (as converged
    accumulators ``mean << k``): ``arch=...`` and ``intent=...`` take {lower-case column name: mean}."""
    o = init_obs(obs_in(init_state(cfg)), cfg)
    arch, intent = np_(o.arch).copy(), np_(o.intent).copy()
    for name, mean in cols.get("arch", {}).items():
        arch[:, rules.ArchCol[name.upper()].value] = mean << RULES_V1.arch_k
    for name, mean in cols.get("intent", {}).items():
        intent[:, rules.IntentCol[name.upper()].value] = mean << RULES_V1.intent_k
    return o._replace(
        n_obs=mx.full((dims(cfg).n,), n_obs, dtype=mx.int32), arch=mx.array(arch), intent=mx.array(intent)
    )


def test_warm_gating(micro_cfg: Config) -> None:
    arch, intent = {"harvest": 300}, {"harvest": 300}
    for n_obs, warm in ((191, False), (192, True)):  # archetypes
        e = evaluate(_obs(micro_cfg, n_obs, arch=arch), micro_cfg, RULES_V1)
        la, ma = np_(e.label_arch), np_(e.match_arch)
        if warm:
            assert (la[:, 0] == Archetype.GATHERER.value).all() and (
                la[:, 1] == Archetype.SETTLER.value
            ).all()
            assert (ma == (1 << 0) | (1 << 5)).all()
        else:
            assert not la.any() and not ma.any()
    for n_obs, warm in ((23, False), (24, True)):  # intents
        e = evaluate(_obs(micro_cfg, n_obs, intent=intent), micro_cfg, RULES_V1)
        mi, cand = np_(e.match_intent), np_(e.cand)
        if warm:
            assert (mi == EVAL_BIT | (1 << (Intent.FEEDING.value - 1))).all()
            assert (cand == Intent.FEEDING.value).all() and (np_(e.cand_n) == 1).all()
        else:
            assert not mi.any() and not cand.any() and not np_(e.cand_n).any()
        assert not np_(e.label_intent).any()  # the hysteresis has not passed either way


def test_match_and_eval_bits(micro_cfg: Config) -> None:
    # nothing matched: a Generalist row (mid speed, no income) and a Resting row (no predicate)
    speed = {"speed": 60}
    e = evaluate(_obs(micro_cfg, 400, arch=speed), micro_cfg, RULES_V1)
    assert (np_(e.match_arch) == GENERALIST_BIT).all()
    assert (np_(e.label_arch)[:, 0] == Archetype.GENERALIST.value).all() and not np_(e.label_arch)[:, 1].any()
    assert (np_(e.match_intent) == EVAL_BIT | RESTING_BIT).all()
    # a Provider that is also a Trader keeps both bits, and the label is (Trader, ...)
    o = _obs(micro_cfg, 400, arch={"inflow": 10, "outflow": 2})
    slots = np_(o.arch_slot).copy()
    slots[:, 0] = np.array([2, 4]) << RULES_V1.arch_k
    e = evaluate(o._replace(arch_slot=mx.array(slots)), micro_cfg, RULES_V1)
    want = (1 << 2) | (1 << 3) | (1 << 5)  # provider, trader, settler (at rest)
    assert (np_(e.match_arch) == want).all()
    la = np_(e.label_arch)
    assert (la[:, 0] == Archetype.TRADER.value).all() and (la[:, 1] == Archetype.SETTLER.value).all()
    # a bit for every predicate that holds, in Intent order (bit i − 1), plus EVAL_BIT
    both = {"attack": 64, "defend": 128}
    e = evaluate(_obs(micro_cfg, 400, intent=both), micro_cfg, RULES_V1)
    assert (np_(e.match_intent) == EVAL_BIT | (1 << 1) | (1 << 3)).all()
    assert (np_(e.cand) == Intent.HUNTING.value).all()  # Hunting precedes Defending
    # the bits are int32 and the evaluated flag is the top used bit
    assert e.match_intent.dtype == mx.int32 and EVAL_BIT == 1 << 30
    # a row that is not warm has no bits at all
    cold = np_(evaluate(_obs(micro_cfg, 10, arch=speed, intent=both), micro_cfg, RULES_V1).match_intent)
    assert not cold.any()


def test_pack_bits() -> None:
    m = np.array([[1, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 0], [1] * 6], bool)
    assert np_(rules.pack_bits(mx.array(m))).tolist() == [0b100101, 0, 0b111111]


def _synthetic(cfg: Config, rows: int = 8, **fields: Any):
    return make_oi(living(cfg, rows, **fields)), make_acts(blank_acts(cfg))


def test_evaluation_on_the_step_clock(micro_cfg: Config) -> None:
    """Two observers on the same constant input, started at steps of different phase mod 8, change a label
    only at ``t % 8 == 0`` and agree at every evaluation once past warm-up plus 3·every steps."""
    oi, acts = _synthetic(micro_cfg, last_harvest=300)
    rs = RULES_V1
    a, b = Driver(micro_cfg, oi, start=0), Driver(micro_cfg, oi, start=5)
    settled = 5 + rs.arch_warm + 3 * rs.every
    prev: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    agreed = 0
    for g in range(1, 400):  # the global step; b observes from step 6 on
        obs = {"a": a.step(oi, acts)}
        if g > 5:
            obs["b"] = b.step(oi, acts)
        for name, o in obs.items():
            labels = (np_(o.label_arch), np_(o.label_intent))
            if name in prev and g % rs.every != 0:  # off the clock: nothing changes
                assert all(np.array_equal(x, y) for x, y in zip(labels, prev[name], strict=True)), (name, g)
            prev[name] = labels
        if "b" in obs and g % rs.every == 0 and g >= settled:
            agreed += 1
            for field in ("label_arch", "label_intent", "match_arch", "match_intent"):
                assert np.array_equal(np_(getattr(obs["a"], field)), np_(getattr(obs["b"], field))), (
                    field,
                    g,
                )
    assert agreed > 10
    # the settled label: Gatherer (then Settler) and Feeding on the living rows; dead rows are never labelled
    la = np_(obs["a"].label_arch)
    assert (la[:8, 0] == Archetype.GATHERER.value).all() and (la[:8, 1] == Archetype.SETTLER.value).all()
    assert (np_(obs["a"].label_intent)[:8] == Intent.FEEDING.value).all()
    assert not la[8:].any() and not np_(obs["a"].label_intent)[8:].any()


@pytest.mark.parametrize("phase", range(8))
def test_budding_is_displayable_at_every_phase(micro_cfg: Config, phase: int) -> None:
    """A single bud, at any phase of the 8-step clock, is the displayed intent by the 4th evaluation after it,
    and not before (the label needs 4 consecutive evaluations; the window is every·hold = 32 steps)."""
    rs = RULES_V1
    oi, quiet = _synthetic(micro_cfg, rows=4)
    flags = blank_acts(micro_cfg)
    flags["flags"][:4] = ActFlag.BUDDED.value
    bud = make_acts(flags)
    drv = Driver(micro_cfg, oi)
    t_bud = 104 + phase  # past both warm-ups
    labels: dict[int, int] = {}
    while drv.t < t_bud + 96:
        o = drv.step(oi, bud if drv.t + 1 == t_bud else quiet)
        if drv.t % rs.every == 0:
            labels[drv.t] = int(np_(o.label_intent)[0])
    evals = sorted(t for t in labels if t >= t_bud)  # evaluation steps at or after the bud
    assert labels[evals[2]] == Intent.RESTING.value  # only 3 consecutive evaluations of Budding so far
    assert labels[evals[3]] == Intent.BUDDING.value  # the 4th: since_bud = e4 − t_bud <= 7 + 24 < 32
    assert evals[3] - t_bud < rs.bud_window
    assert labels[evals[-1]] == Intent.RESTING.value  # and the label returns to Resting afterwards
    assert all(labels[t] == Intent.RESTING.value for t in labels if 56 <= t < t_bud)


def test_archetype_names_and_priority_are_consistent() -> None:
    assert ARCHETYPE_NAMES == ("gatherer", "hunter", "provider", "trader", "nomad", "settler", "generalist")
    intents = (
        *("budding", "hunting", "fleeing", "defending", "caring"),
        *("bonding", "signalling", "feeding", "roaming", "resting"),
    )
    assert intents == INTENT_NAMES
    assert [a.value for a in Archetype] == list(range(8)) and [i.value for i in Intent] == list(range(11))
    assert sorted(a.value for a in ARCH_PRIORITY) == [1, 2, 3, 4, 5, 6]
    # the order is part of INTENT_VERSION 1 but not of RuleSet.digest(): it is pinned here as a literal
    assert ARCH_PRIORITY == (
        Archetype.TRADER,
        Archetype.PROVIDER,
        Archetype.HUNTER,
        Archetype.GATHERER,
        Archetype.NOMAD,
        Archetype.SETTLER,
    )
    assert ARCH_PRIORITY.index(Archetype.TRADER) < ARCH_PRIORITY.index(Archetype.PROVIDER)
    assert VQ_MAX == 1 << 15 and n_arch_cols(2) == 9 and n_arch_cols(3) == 11
    assert dims(load_config(preset="tiny3d")).nd == 3
