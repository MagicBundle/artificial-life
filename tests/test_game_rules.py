"""§28.4 the archetype and intent rules: thresholds at their edges (version 2, and version 1 through
``RULES_V1``), priority, warm-up, hysteresis, the Budding forecast and overlay, the step clock, the match
bits, and the pinned ``RuleSet`` (``INTENT_VERSION`` 2; ``RULES_V1`` keeps its G1 pin)."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
from collections.abc import Sequence
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from alife import layout
from alife.config import Config, config_from_dict, config_to_dict, dims, load_config
from alife.game import rules
from alife.game.observe import OBS_CAP, ObsIn, ObsState, evaluate, init_obs, obs_in
from alife.game.rules import (
    ARCH_PRIORITY,
    ARCHETYPE_DISPLAY,
    ARCHETYPE_NAMES,
    EVAL_BIT,
    INTENT_DISPLAY,
    INTENT_NAMES,
    INTENT_VERSION,
    RULES_V1,
    RULES_V2,
    VQ_MAX,
    Archetype,
    BudIn,
    Intent,
    RuleSet,
    n_arch_cols,
)
from alife.state import init_state, to_numpy
from alife.step import ActFlag
from game_helpers import Driver, blank_acts, living, make_acts, make_oi

# Pinned at implementation time. Any edit to a threshold, window, period or field of RuleSet changes it, and
# must come with a new INTENT_VERSION and a new pin (the sidecar stores the digest). Version 1's pin is G1's;
# version 2's was computed offline from the canonical JSON of the calibration study (rules_v2 §2.2).
RULES_V1_DIGEST = "156ad4963b84ffe9636e9ba4fdd7592632c669bf6e490687d8eb2b38b9aec009"
RULES_V2_DIGEST = "dc78f771d515b845501cd609275b362acfbc60c47853e813a6d129d3a5cab4f7"

B = layout.B
GENERALIST_BIT = 1 << (Archetype.GENERALIST.value - 1)
RESTING_BIT = 1 << (Intent.RESTING.value - 1)


def np_(a: mx.array) -> np.ndarray:
    return to_numpy(a)


# --- the pinned rule set ------------------------------------------------------------------------------------


def test_rules_digests_are_pinned() -> None:
    assert INTENT_VERSION == 2
    assert RuleSet() == RULES_V2
    assert RULES_V2.digest() == RULES_V2_DIGEST
    assert RULES_V1.digest() == RULES_V1_DIGEST  # the G1 pin, kept: RULES_V1 still evaluates as version 1
    assert len(RULES_V2.digest()) == 64
    assert RULES_V1 != RULES_V2 and RULES_V1.digest() != RULES_V2.digest()


def test_every_field_enters_the_digest() -> None:
    """Every field moves the digest of the version-2 rules, and of the version-1 rules where it is a field of
    version 1; a post-v1 field moves the version-1 digest only off its inert value."""
    changed = {
        "arch_k": 5, "intent_k": 4, "every": 4, "hold": 3, "arch_warm": 191, "intent_warm": 23,
        "gatherer": (8, 10), "gatherer_level": (4, 5), "hunter": (8, 20), "provider": (2, 5),
        "trader": (2, 3), "nomad_speed": (513, 5), "nomad_persist": (10, 25), "settler_speed": (257, 10),
        "settler_body": 5, "bud_window": 33, "bud_horizon": 11, "bud_cost": 25, "bud_overlay": False,
        "hunt": 65, "flee_hurt": 9, "flee_thrust": 129, "defend": 193, "defend_hurt": 2, "care": (2, 10),
        "bond_age": (25, 120), "bond_rate": 65, "signal": 193, "feed": (4, 4), "roam": (769, 10),
        "intent_order": (1, 2, 3, 4, 5, 6, 7, 8, 9),
    }  # fmt: skip
    assert set(changed) == {f.name for f in dataclasses.fields(RuleSet)}
    digests = {RULES_V2.digest()}
    for name, value in changed.items():
        digests.add(dataclasses.replace(RULES_V2, **{name: value}).digest())
    assert len(digests) == len(changed) + 1
    # off RULES_V1: a version-1 field changes the digest; a post-v1 field only off its inert value (to its
    # RULES_V2 value, which is never inert), and at its inert value it leaves the G1 pin
    post = rules._POST_V1_INERT
    assert set(post) == {
        "settler_body", "gatherer_level", "bud_horizon", "bud_cost", "bud_overlay", "defend_hurt",
        "bond_rate", "intent_order",
    }  # fmt: skip
    v1 = {RULES_V1_DIGEST}
    for name, value in changed.items():
        if name in post:
            assert getattr(RULES_V2, name) != post[name], name
            assert dataclasses.replace(RULES_V1, **{name: post[name]}).digest() == RULES_V1_DIGEST, name
            value = getattr(RULES_V2, name)
        v1.add(dataclasses.replace(RULES_V1, **{name: value}).digest())
    assert len(v1) == len(changed) + 1  # every change differs from each other and from the G1 pin


def test_post_v1_fields_leave_the_digest_while_inert() -> None:
    """A field added after version 1 is left out of the digest at its inert value, so the G1 pin and the
    digest of every G1-era rule set stand; at any other value it changes the digest."""
    assert rules._POST_V1_INERT["settler_body"] == 0
    assert RULES_V1.settler_body == 0 and dataclasses.replace(RULES_V2, **rules._POST_V1_INERT) != RULES_V2
    inert_v2 = dataclasses.replace(RULES_V2, **rules._POST_V1_INERT)
    assert inert_v2.gatherer == (0, 1) and inert_v2.digest() != RULES_V1_DIGEST  # v2's own v1-field values
    sweep = dataclasses.replace(RULES_V1, arch_k=5, arch_warm=96)
    assert dataclasses.replace(sweep, settler_body=0).digest() == sweep.digest()
    on = {dataclasses.replace(RULES_V1, settler_body=n).digest() for n in (1, 3, 4, 9)}
    assert len(on) == 4 and RULES_V1_DIGEST not in on
    assert dataclasses.replace(sweep, settler_body=4).digest() not in {sweep.digest(), *on}
    # the hashed JSON of an inert rule set has none of the post-v1 keys: the G1 digest, recomputed by hand
    fields = {k: v for k, v in dataclasses.asdict(RULES_V1).items() if k not in rules._POST_V1_INERT}
    assert "settler_body" not in fields and "intent_order" not in fields
    text = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == RULES_V1_DIGEST
    # and the version-2 JSON is the one the calibration pinned (every key, none left out)
    text2 = json.dumps(dataclasses.asdict(RULES_V2), sort_keys=True, separators=(",", ":"))
    assert text2 == (
        '{"arch_k":6,"arch_warm":192,"bond_age":[24,120],"bond_rate":64,"bud_cost":24,"bud_horizon":10,'
        '"bud_overlay":true,"bud_window":32,"care":[1,10],"defend":192,"defend_hurt":1,"every":8,'
        '"feed":[3,4],"flee_hurt":8,"flee_thrust":128,"gatherer":[0,1],"gatherer_level":[3,4],"hold":4,'
        '"hunt":64,"hunter":[7,20],"intent_k":3,"intent_order":[1,2,3,5,4,6,8,7,9],"intent_warm":24,'
        '"nomad_persist":[9,25],"nomad_speed":[512,5],"provider":[1,5],"roam":[768,10],"settler_body":4,'
        '"settler_speed":[256,10],"signal":192,"trader":[1,2]}'
    )
    assert hashlib.sha256(text2.encode("utf-8")).hexdigest() == RULES_V2_DIGEST


def test_ruleset_rejects_nonsense() -> None:
    for kw in ({"arch_k": 0}, {"intent_k": 31}, {"every": 0}, {"hold": 0}, {"arch_warm": -1},
               {"gatherer": (7, 0)}, {"hunter": (-1, 5)}, {"bond_age": (24, 24)},
               {"settler_body": -1}, {"settler_body": 1 << 31},
               {"intent_order": (1, 2, 3, 4, 5, 6, 7, 8, 8)}, {"intent_order": [1, 2, 3, 4, 5, 6, 7, 8, 9]},
               {"intent_order": (1, 2, 3, 4, 5, 6, 7, 8)}, {"intent_order": (0, 1, 2, 3, 4, 5, 6, 7, 8)},
               {"bud_cost": -1}, {"bud_horizon": -1}, {"defend_hurt": -1}, {"bond_rate": -1},
               {"gatherer_level": (3, 0)}, {"gatherer_level": (-1, 4)}, {"bud_overlay": 1}):  # fmt: skip
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
    RULES_V2.check(_with_limits(preset))
    top = _with_limits(preset, (1 << 16) - 1)  # the validated extremes ("Contention headroom")
    RULES_V1.check(top)
    RULES_V2.check(top)
    RuleSet(arch_k=12).check(top)  # 4·(2^16 − 1)·(2^12 + 1) < 2^31
    with pytest.raises(ValueError, match=r"arch\.(inflow|outflow|share).*k = 13"):
        RuleSet(arch_k=13).check(top)
    # the same k is fine at the default share_rate, until the quantised-velocity columns overflow
    RuleSet(arch_k=15).check(_with_limits(preset))
    with pytest.raises(ValueError, match=r"arch\.(speed|velocity).*k = 16"):
        RuleSet(arch_k=16).check(_with_limits(preset))
    with pytest.raises(ValueError, match=r"intent\.care.*k = 13"):
        RuleSet(intent_k=13).check(top)


@pytest.mark.parametrize("preset", ["tiny", "64k", "tiny3d", "64k3d"])
def test_ruleset_check_bounds_the_flow_columns(preset: str) -> None:
    """The Budding flow columns bound ``intent_k``: FLOW_NEG is at most ``last_loss`` (<= e_max) + Σ out + the
    four visible costs, 1,004,230 at the defaults, so 11 is the largest ``intent_k`` that fits int32 (12 fails
    naming ``intent.flow_neg``). The columns exist under every rule set, so this holds for v1 sweeps too."""
    cfg = load_config(preset=preset)
    ec = cfg.energy
    assert ec.e_max == 10**6
    bound = (
        ec.e_max
        + B * cfg.bonds.share_rate
        + ec.move_cost
        + ec.defend_cost
        + 3 * ec.emit_cost
        + ec.attack_cost
    )
    assert bound == 1_004_230 and bound * ((1 << 11) + 1) <= INT32_MAX < bound * ((1 << 12) + 1)
    RuleSet(intent_k=11).check(cfg)
    dataclasses.replace(RULES_V1, intent_k=11).check(cfg)
    for rs in (RuleSet(intent_k=12), dataclasses.replace(RULES_V1, intent_k=12)):
        with pytest.raises(ValueError, match=r"intent\.flow_neg.*k = 12"):
            rs.check(cfg)
    # FLOW_POS is at most harvest + attack gain + Σ got, FORMED a flag
    assert cfg.energy.harvest_max + cfg.combat.attack_max + B * cfg.bonds.share_rate == 8400


INT32_MAX = (1 << 31) - 1


def test_ruleset_check_counts_every_term_of_the_flow_columns() -> None:
    """Each visible term of the two flow bounds is in ``check``: with the terms of a column summing to one
    over its int32 limit (``X·(2^k + 1) > 2^31 − 1``, with ``arch_k`` 1 so that no archetype column
    interferes), the column is named, and a term left out of the bound would let the configuration through."""
    rs = dataclasses.replace(RULES_V2, arch_k=1)
    limit = INT32_MAX // ((1 << rs.intent_k) + 1)
    base = config_to_dict(load_config(preset="tiny"))
    zeroed = (
        *(
            ("energy", k)
            for k in ("harvest_max", "e_max", "move_cost", "defend_cost", "emit_cost", "attack_cost")
        ),
        ("combat", "attack_max"),
        ("bonds", "share_rate"),
    )

    def cfg_with(**terms: int) -> Config:
        d = {k: dict(v) if isinstance(v, dict) else v for k, v in base.items()}
        for sec, key in zeroed:
            d[sec][key] = 0
        for name, value in terms.items():
            sec, key = name.split("__")
            d[sec][key] = value
        return config_from_dict(d)  # not validated: only the bounds matter to RuleSet.check

    share = limit // B  # B·share is at most the limit, so the Caring column passes
    rest = limit - B * share
    one_term = {  # a column's term alone, one over the limit
        "e_max": {"energy__e_max": limit + 1},
        "move_cost": {"energy__move_cost": limit + 1},
        "defend_cost": {"energy__defend_cost": limit + 1},
        "emit_cost": {"energy__emit_cost": limit // layout.NCH + 1},  # NCH channels of emit_cost
        "attack_cost": {"energy__attack_cost": limit + 1},
    }
    for kw in one_term.values():
        with pytest.raises(ValueError, match=r"intent\.flow_neg"):
            rs.check(cfg_with(**kw))
    # the Σ out term: B·share_rate plus one more of any other term
    with pytest.raises(ValueError, match=r"intent\.flow_neg"):
        rs.check(cfg_with(bonds__share_rate=share, energy__e_max=rest + 1))
    # FLOW_POS: harvest, attack gain and Σ got, each alone with a unit of another
    for kw in ({"energy__harvest_max": limit, "combat__attack_max": 1},
               {"combat__attack_max": limit, "energy__harvest_max": 1},
               {"bonds__share_rate": share, "energy__harvest_max": rest + 1}):  # fmt: skip
        with pytest.raises(ValueError, match=r"intent\.flow_pos"):
            rs.check(cfg_with(**kw))
    # and exactly at the limits nothing is raised (the ratios are loosened for an income this large)
    ok = dataclasses.replace(rs, hunter=(1, 1), provider=(1, 1), care=(1, 1))
    ok.check(cfg_with(energy__e_max=limit))
    ok.check(cfg_with(bonds__share_rate=share, energy__e_max=rest))
    ok.check(cfg_with(energy__harvest_max=limit))
    ok.check(cfg_with(bonds__share_rate=share, energy__harvest_max=rest))


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
        "gatherer_level": cfg.energy.harvest_max,
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
    # the harvest-level ratio at its largest compared mean, harvest_max, in the same way
    x = cfg.energy.harvest_max
    den = INT32_MAX // x
    rs = dataclasses.replace(RULES_V2, gatherer_level=(den - 1, den))
    rs.check(cfg)
    body = mx.zeros((1,), mx.int32)
    zero_slots = mx.zeros((1, B, 2), mx.int32)
    assert bool(np_(rules.archetype_matches(_m6(2, harvest=x), zero_slots, cfg, rs, body))[0, 0])
    assert not bool(np_(rules.archetype_matches(_m6(2, harvest=x - 1), zero_slots, cfg, rs, body))[0, 0])
    with pytest.raises(ValueError, match="ratio gatherer_level"):
        dataclasses.replace(RULES_V2, gatherer_level=(den, den + 1)).check(cfg)


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


def _arch(
    cfg: Config,
    nd: int = 2,
    slots: Sequence[tuple[int, int]] = (),
    rs: RuleSet = RULES_V1,
    body: int = 0,
    **cols: Any,
) -> dict[str, bool]:
    """The six archetype rules for a row with the given means, under ``rs`` (the version-1 rules unless
    given) and a bond component of ``body`` members."""
    m6s = np.zeros((1, B, 2), np.int32)
    for b, pair in enumerate(slots):
        m6s[0, b] = pair
    match = np_(
        rules.archetype_matches(_m6(nd, **cols), mx.array(m6s), cfg, rs, mx.array([body], dtype=mx.int32))
    )
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


def test_gatherer_v2_is_a_harvest_level(tiny_cfg: Config) -> None:
    """Version 2: m_harvest >= 1 and 4·m_harvest >= 3·harvest_max (300 of 400), whatever the income share."""
    v2 = {"rs": RULES_V2}
    assert tiny_cfg.energy.harvest_max == 400
    assert (
        _arch(tiny_cfg, harvest=300, **v2)["gatherer"] and not _arch(tiny_cfg, harvest=299, **v2)["gatherer"]
    )
    # the share no longer matters: a third of the income is harvest, and it is still a Gatherer
    assert _arch(tiny_cfg, harvest=300, gain=300, **v2)["gatherer"]
    assert _arch(tiny_cfg, harvest=300, inflow=2000, **v2)["gatherer"]
    assert not _arch(tiny_cfg, harvest=299, gain=300, **v2)["gatherer"]
    # no harvest, no Gatherer: any other income
    for income in ({"gain": 400}, {"inflow": 400}, {"gain": 100, "inflow": 100}):
        assert not _arch(tiny_cfg, **income, **v2)["gatherer"]
    assert not _arch(tiny_cfg, **v2)["gatherer"]
    # a world without harvest: any harvest at all is the level (the ratio's right side is 0), none is not
    zero = config_from_dict(
        {**config_to_dict(tiny_cfg), "energy": {**config_to_dict(tiny_cfg)["energy"], "harvest_max": 0}}
    )
    assert _arch(zero, harvest=1, **v2)["gatherer"] and not _arch(zero, harvest=0, **v2)["gatherer"]
    # ... and other income without harvest is not one, even though 0 >= 0 holds for the level there
    assert not _arch(zero, gain=5, **v2)["gatherer"] and not _arch(zero, inflow=5, **v2)["gatherer"]
    # version 1 is the share, unchanged: the v1 assertions hold through RULES_V1
    assert (
        _arch(tiny_cfg, harvest=7, inflow=3)["gatherer"]
        and not _arch(tiny_cfg, harvest=6, inflow=4)["gatherer"]
    )
    assert _arch(tiny_cfg, harvest=1)["gatherer"] and not _arch(tiny_cfg, harvest=300, gain=300)["gatherer"]
    assert _arch(tiny_cfg, harvest=299)["gatherer"]  # a high share at a low level is a v1 Gatherer
    # the two parts compose: share and level both, on a rule set that names both
    both = dataclasses.replace(RULES_V2, gatherer=(7, 10))
    assert _arch(tiny_cfg, harvest=300, rs=both)["gatherer"]
    assert not _arch(tiny_cfg, harvest=300, gain=200, rs=both)["gatherer"]  # level yes, share 0.6 no
    assert not _arch(tiny_cfg, harvest=299, rs=both)["gatherer"]  # share yes, level no


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
    # version 2 adds the colony clause (bond component of 4 or more), at any speed
    assert _arch(tiny_cfg, nd, speed=200, vpos=vec(x=200), rs=RULES_V2, body=4)["settler"]
    assert not _arch(tiny_cfg, nd, speed=200, vpos=vec(x=200), rs=RULES_V2, body=3)["settler"]
    # no archetype but Settler/Nomad depends on movement
    assert not any(_arch(tiny_cfg, nd, speed=60, vpos=vec(x=60)).values())  # a Generalist row


SETTLER4 = dataclasses.replace(RULES_V1, settler_body=4)


def _settler(cfg: Config, rs: RuleSet, body: int | None, nd: int = 2, **cols: Any) -> bool:
    m6 = _m6(nd, **cols)
    arg = None if body is None else mx.array([body], dtype=mx.int32)
    out = rules.archetype_matches(m6, mx.zeros((1, B, 2), mx.int32), cfg, rs, arg)
    return bool(np_(out)[0, ARCHETYPE_NAMES.index("settler")])


@pytest.mark.parametrize("nd", [2, 3])
def test_settler_colony_clause(tiny_cfg: Config, nd: int) -> None:
    """Settler also matches a row in a bond component of at least ``settler_body`` members (body 3 and 4 at
    ``settler_body`` 4), whatever its speed; under ``RULES_V1`` the body is ignored."""
    vec = (200,) + (0,) * (nd - 1)
    fast = {"speed": 200, "vpos": vec}  # moving fast: not a Settler by speed
    assert not _settler(tiny_cfg, SETTLER4, 3, nd, **fast)
    assert _settler(tiny_cfg, SETTLER4, 4, nd, **fast)  # inclusive edge
    assert _settler(tiny_cfg, SETTLER4, 9, nd, **fast)
    assert not _settler(tiny_cfg, SETTLER4, 0, nd, **fast)  # a row reset since the snapshot: body 0
    # slow rows are Settlers by the speed rule at any body
    assert _settler(tiny_cfg, SETTLER4, 0, nd, speed=25) and _settler(tiny_cfg, SETTLER4, 3, nd, speed=25)
    assert not _settler(tiny_cfg, SETTLER4, 3, nd, speed=26)
    # RULES_V1 ignores the body: slow rows with body 9 are Settlers exactly when the speed rule says so
    for speed in (0, 25, 26, 200):
        want = 10 * speed < 256
        assert _settler(tiny_cfg, RULES_V1, 9, nd, speed=speed) is want
        assert _settler(tiny_cfg, RULES_V1, None, nd, speed=speed) is want
        assert _settler(tiny_cfg, RULES_V1, 0, nd, speed=speed) is want
    # a smaller threshold admits smaller components
    assert _settler(tiny_cfg, dataclasses.replace(RULES_V1, settler_body=2), 2, nd, **fast)
    assert not _settler(tiny_cfg, dataclasses.replace(RULES_V1, settler_body=2), 1, nd, **fast)


def test_settler_rules_need_the_body(tiny_cfg: Config) -> None:
    """Rules with ``settler_body`` > 0 raise without a body, never silently label by speed alone; evaluate
    passes the body through, and ignores it under rules that do not use it."""
    with pytest.raises(ValueError, match="settler_body = 4 needs the rows' body sizes"):
        _settler(tiny_cfg, SETTLER4, None, speed=200, vpos=(200, 0))
    o = init_obs(obs_in(init_state(tiny_cfg)), tiny_cfg)
    with pytest.raises(ValueError, match="settler_body"):
        evaluate(o, tiny_cfg, SETTLER4)
    n = dims(tiny_cfg).n
    body = mx.full((n,), 9, dtype=mx.int32)
    got = evaluate(o, tiny_cfg, SETTLER4, body=body)
    assert np.array_equal(
        np_(evaluate(o, tiny_cfg, RULES_V1, body=body).match_arch),
        np_(evaluate(o, tiny_cfg, RULES_V1).match_arch),
    )
    # an all-zero accumulator is at rest: every row matches Settler by speed, and the body adds nothing
    assert np.array_equal(np_(got.match_arch), np_(evaluate(o, tiny_cfg, RULES_V1).match_arch))


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


def _bud_in(
    cfg: Config, *, energy: int | None = None, brain: int = 0, bonds: int = 0, n_obs: int = 1000
) -> BudIn:
    """One row's post-step values for the Budding forecast; the default energy is ``repro.min_energy`` itself,
    which never forecasts a bud (it is already there)."""
    me = cfg.repro.min_energy if energy is None else energy
    return BudIn(*(mx.array([v], dtype=mx.int32) for v in (me, brain, bonds, n_obs)))


def _intent(
    cfg: Config,
    *,
    since_bud: int = OBS_CAP,
    ages: Sequence[int] = (),
    kin: Sequence[bool] = (),
    rs: RuleSet = RULES_V1,
    bud: BudIn | None = None,
    **cols: int,
) -> dict[str, bool]:
    """The nine predicates for a row with the intent means ``cols`` (lower-case ``IntentCol`` names), under
    ``rs`` (the version-1 rules unless given). The forecast of a rule set that has one reads ``bud`` (by
    default a row at ``min_energy``: no Budding)."""
    m3 = np.zeros((1, rules.N_INTENT_COLS), np.int32)
    for name, value in cols.items():
        m3[0, rules.IntentCol[name.upper()].value] = value
    age = np.zeros((1, B), np.int32)
    age[0, : len(ages)] = ages
    k = np.zeros((1, B), bool)
    k[0, : len(kin)] = kin
    if bud is None and rs.bud_horizon > 0:
        bud = _bud_in(cfg)
    match = np_(
        rules.intent_matches(
            mx.array(m3), mx.array([since_bud], dtype=mx.int32), mx.array(age), mx.array(k), cfg, rs, bud=bud
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


def test_intent_predicates_v2_at_their_edges(tiny_cfg: Config) -> None:
    c, hm = tiny_cfg, tiny_cfg.energy.harvest_max
    v2 = {"rs": RULES_V2}
    assert hm == 400
    # Defending: hurt within about 31 steps and the guard at 0.75 (m_defend >= 192)
    assert _intent(c, hurt=1, defend=192, **v2)["defending"]
    assert not _intent(c, hurt=0, defend=255, **v2)["defending"]  # a raised guard that nobody hit
    assert not _intent(c, hurt=1, defend=191, **v2)["defending"]
    assert _intent(c, hurt=200, defend=255, **v2)["defending"]
    # Signalling: m_emit >= 192; Feeding: 4·m_harvest >= 3·harvest_max (300)
    assert _intent(c, emit=192, **v2)["signalling"] and not _intent(c, emit=191, **v2)["signalling"]
    assert _intent(c, harvest=300, **v2)["feeding"] and not _intent(c, harvest=299, **v2)["feeding"]
    assert not _intent(c, **v2)["feeding"]
    # Bonding: a formation rate m_formed >= 64 (a flag mean of 0.25 on the EMA); the slot ages are ignored
    assert _intent(c, formed=64, **v2)["bonding"] and not _intent(c, formed=63, **v2)["bonding"]
    assert not _intent(c, ages=[50], formed=0, **v2)["bonding"]
    assert not _intent(c, ages=[24, 60, 119], formed=0, **v2)["bonding"]
    assert _intent(c, ages=[50], kin=[True], formed=64, **v2)["bonding"]  # kin or not, formed is formed
    # Hunting, Fleeing, Caring and Roaming are the version-1 rules
    assert _intent(c, attack=64, **v2)["hunting"] and not _intent(c, attack=63, **v2)["hunting"]
    assert (
        _intent(c, hurt=8, thrust=128, **v2)["fleeing"]
        and not _intent(c, hurt=7, thrust=128, **v2)["fleeing"]
    )
    assert not _intent(c, hurt=8, thrust=127, **v2)["fleeing"]
    assert _intent(c, care=40, **v2)["caring"] and not _intent(c, care=39, **v2)["caring"]
    assert _intent(c, thrust=77, **v2)["roaming"] and not _intent(c, thrust=76, **v2)["roaming"]
    assert not any(_intent(c, **v2).values())  # nothing at rest
    # Budding is the forecast (budding_forecast edges below): since_bud is ignored
    assert not _intent(c, since_bud=0, **v2)["budding"]
    # under RULES_V1 the version-1 assertions hold unchanged, whatever the new columns say
    v1 = {"rs": RULES_V1}
    assert _intent(c, defend=128, **v1)["defending"] and not _intent(c, defend=127, **v1)["defending"]
    assert _intent(c, emit=128, **v1)["signalling"] and _intent(c, harvest=200, **v1)["feeding"]
    assert _intent(c, ages=[24], **v1)["bonding"] and not _intent(c, formed=255, **v1)["bonding"]
    assert _intent(c, since_bud=31, **v1)["budding"] and not _intent(c, since_bud=32, **v1)["budding"]
    assert not _intent(c, flow_pos=255, **v1)["budding"]


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


def test_first_intent_follows_the_version_2_order() -> None:
    """Caring before Defending and Feeding before Signalling; the columns stay in Intent value order."""
    order = RULES_V2.intent_order
    assert order == (1, 2, 3, 5, 4, 6, 8, 7, 9)
    first = lambda *on: int(  # noqa: E731
        np_(rules.first_intent(mx.array([[i in on for i in range(1, 10)]]), order))[0]
    )
    D, C, S, F = (Intent[n].value for n in ("DEFENDING", "CARING", "SIGNALLING", "FEEDING"))
    assert first(D, C) == C and first(F, S) == F  # v1: Defending and Signalling would win
    assert first(D) == D and first(S) == S and first() == Intent.RESTING.value
    assert first(*range(1, 10)) == Intent.BUDDING.value
    for lower, higher in ((D, C), (S, F)):  # the swapped pairs, v1 against v2
        assert (
            int(np_(rules.first_intent(mx.array([[i in (lower, higher) for i in range(1, 10)]])))[0]) == lower
        )
    rng = np.random.default_rng(1)
    rows = rng.random((200, 9)) < 0.3
    got = np_(rules.first_intent(mx.array(rows), order))
    want = [next((i for i in order if r[i - 1]), Intent.RESTING.value) for r in rows]
    assert got.tolist() == want and got.dtype == np.int32
    # any permutation is a priority
    rev = tuple(reversed(range(1, 10)))
    assert int(np_(rules.first_intent(mx.array([[True] * 9]), rev))[0]) == Intent.ROAMING.value


def test_budding_forecast_at_its_edges(tiny_cfg: Config) -> None:
    """E < me, E + h·(flow − basal − c) >= me within h = 10 steps at c = 24, n_obs >= min_age; int64."""
    cfg = tiny_cfg
    me, ma, ec = cfg.repro.min_energy, cfg.repro.min_age, cfg.energy
    assert (me, ma, ec.basal_cost, ec.bond_upkeep) == (40_000, 20, 100, 5) and RULES_V2.bud_horizon == 10

    def forecast(energy: int, *, pos: int = 500, neg: int = 0, brain: int = 0, bonds: int = 0,
                 n_obs: int = 1000, rs: RuleSet = RULES_V2) -> bool:  # fmt: skip
        m3 = np.zeros((1, rules.N_INTENT_COLS), np.int32)
        m3[0, rules.IntentCol.FLOW_POS.value], m3[0, rules.IntentCol.FLOW_NEG.value] = pos, neg
        b = _bud_in(cfg, energy=energy, brain=brain, bonds=bonds, n_obs=n_obs)
        return bool(np_(rules.budding_forecast(mx.array(m3), b, cfg, rs))[0])

    basal = ec.basal_cost  # no brain cost, no bonds
    edge = me - 10 * (500 - basal - 24)  # E + 10·(500 − basal − 24) = me
    assert edge == 36_240
    assert forecast(edge) and not forecast(edge - 1)
    assert forecast(me - 1) and not forecast(me) and not forecast(me + 5000)  # already there: not about to
    assert not forecast(edge, n_obs=ma - 1) and forecast(edge, n_obs=ma)
    assert not forecast(edge, pos=100, neg=101) and not forecast(me - 1, pos=0, neg=500)  # losing energy
    assert not forecast(me - 1, pos=500, neg=500 + 9000)
    # brain cost and bonds are part of the basal due: each shifts the edge by h times its amount
    assert forecast(edge + 10 * 7, brain=7) and not forecast(edge + 10 * 7 - 1, brain=7)
    assert forecast(edge + 10 * ec.bond_upkeep * 2, bonds=2) and not forecast(edge + 99, bonds=2)
    assert forecast(edge + 100, bonds=2)
    # int64: a brain cost of 2^31 − 1 (nothing in the config bounds it) does not wrap
    assert not forecast(me - 1, brain=(1 << 31) - 1) and not forecast(0, brain=(1 << 31) - 1)
    assert not forecast(me - 1, brain=(1 << 31) - 1, pos=(1 << 31) - 1)  # flow and basal cancel: not enough
    # other horizons and allowances
    assert forecast(me - 1, pos=100 + 24 + 1, rs=dataclasses.replace(RULES_V2, bud_horizon=1))
    assert not forecast(me - 1, pos=100 + 24, rs=dataclasses.replace(RULES_V2, bud_horizon=1))
    assert forecast(me - 1, pos=100 + 1, rs=dataclasses.replace(RULES_V2, bud_horizon=1, bud_cost=0))
    # intent_matches without the post-step values raises under a rule set that has the forecast
    with pytest.raises(ValueError, match="bud_horizon = 10 needs bud"):
        rules.intent_matches(
            mx.zeros((1, 10), mx.int32), mx.zeros((1,), mx.int32), mx.zeros((1, B), mx.int32),
            mx.zeros((1, B), mx.bool_), cfg, RULES_V2,
        )  # fmt: skip
    # ... and not under the version-1 rule, which ignores bud
    rules.intent_matches(
        mx.zeros((1, 10), mx.int32), mx.zeros((1,), mx.int32), mx.zeros((1, B), mx.int32),
        mx.zeros((1, B), mx.bool_), cfg, RULES_V1,
    )  # fmt: skip


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


@pytest.mark.parametrize("rs", [RULES_V1, RULES_V2], ids=["v1", "v2"])
def test_evaluation_on_the_step_clock(micro_cfg: Config, rs: RuleSet) -> None:
    """Two observers on the same constant input, started at steps of different phase mod 8, change a label
    only at ``t % 8 == 0`` and agree at every evaluation once past warm-up plus 3·every steps."""
    # 350: above the Gatherer level (300) with the floor of the long window's mean (A >> 6) still in reach
    oi, acts = _synthetic(micro_cfg, last_harvest=350)
    a, b = Driver(micro_cfg, oi, start=0, rs=rs), Driver(micro_cfg, oi, start=5, rs=rs)
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
            for field in ("label_arch", "label_intent", "held", "match_arch", "match_intent"):
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
    drv = Driver(micro_cfg, oi, rs=rs)
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


@pytest.mark.parametrize("phase", range(8))
def test_budding_forecast_is_displayed_at_once(micro_cfg: Config, phase: int) -> None:
    """Version 2: on a constant income that carries the energy across ``min_energy``, Budding is displayed at
    the first evaluation where the forecast holds, whatever the phase of the 8-step clock, and not before (the
    overlay is outside the hold; the held intent underneath is Feeding)."""
    cfg, rs, rows = micro_cfg, RULES_V2, 4
    me, harvest, basal = cfg.repro.min_energy, 350, cfg.energy.basal_cost
    # at the converged flow (harvest, no costs) the forecast holds from E >= me − 10·(harvest − basal − 24)
    low = me - rs.bud_horizon * (harvest - basal - rs.bud_cost)
    assert 0 < low < me
    slope = 127  # energy gained per step: the forecast window [low, me) is 13+ steps wide, over an evaluation
    quiet = make_acts(blank_acts(cfg))

    def oi_at(t: int) -> ObsIn:
        return make_oi(living(cfg, rows, last_harvest=harvest, energy=slope * (t + phase)))

    drv = Driver(cfg, oi_at(0), rs=rs)
    seen: dict[int, tuple[int, int]] = {}
    while drv.t < 420:
        o = drv.step(oi_at(drv.t + 1), quiet)
        if drv.t % rs.every == 0:
            seen[drv.t] = (int(np_(o.label_intent)[0]), int(np_(o.held)[0]))
    window = [t for t in seen if low <= slope * (t + phase) < me]  # the evaluations where the forecast holds
    assert window and min(window) > 160, window  # past warm-up and the accumulator's convergence
    first = min(window)
    for t, (label, held) in seen.items():
        if t in window:
            assert label == Intent.BUDDING.value and held == Intent.FEEDING.value, t
        elif t >= 60:  # the hold has passed: Feeding is held and displayed, Budding or not
            assert label == held == Intent.FEEDING.value, t
    assert seen[first - 8][0] == Intent.FEEDING.value  # shown at once, not after a hold
    assert seen[max(window) + 8][0] == Intent.FEEDING.value  # and gone at once


def _overlay_inputs(cfg: Config, *, forecast: bool) -> tuple[ObsIn, mx.array]:
    """The post-step ``ObsIn`` and body of an evaluation: every row at an energy where the forecast holds (at
    FLOW_POS mean 500, basal 100 + the rows' brain cost, 0 bonds) or at ``min_energy`` where it cannot."""
    oi = obs_in(init_state(cfg))
    n = dims(cfg).n
    energy = (
        cfg.repro.min_energy - 10 * (500 - cfg.energy.basal_cost - 24) if forecast else cfg.repro.min_energy
    )
    oi = oi._replace(
        energy=mx.full((n,), energy, dtype=mx.int32),
        brain_cost=mx.zeros((n,), dtype=mx.int32),
        partner=mx.full(oi.partner.shape, -1, dtype=mx.int32),
    )
    return oi, mx.zeros((n,), dtype=mx.int32)


def test_budding_overlay_is_displayed_outside_the_hold(micro_cfg: Config) -> None:
    cfg, rs = micro_cfg, RULES_V2
    feeding, hunting = {"harvest": 300, "flow_pos": 500}, {"attack": 64, "flow_pos": 500}

    def at(o: ObsState, forecast: bool, **intent: int) -> ObsState:
        base = _obs(cfg, 400, intent=intent)
        o = base._replace(
            label_intent=o.label_intent,
            held=o.held,
            cand=o.cand,
            cand_n=o.cand_n,
            match_intent=o.match_intent,
        )
        oi, body = _overlay_inputs(cfg, forecast=forecast)
        return evaluate(o, cfg, rs, body=body, oi=oi)

    def row(o: ObsState) -> tuple[int, int, int, int]:
        return tuple(int(np_(getattr(o, f))[0]) for f in ("label_intent", "held", "cand", "cand_n"))

    F, B_, H = Intent.FEEDING.value, Intent.BUDDING.value, Intent.HUNTING.value
    o = _obs(cfg, 400, intent=feeding)
    for i in range(4):  # four evaluations of Feeding: held and displayed at the fourth, never before
        o = at(o, False, **feeding)
        assert row(o)[:2] == ((0, 0) if i < 3 else (F, F))
    assert row(o) == (F, F, F, 0)
    # the forecast holds at evaluation e: Budding is shown at once, the held intent and candidate untouched
    e = at(o, True, **feeding)
    assert row(e) == (B_, F, F, 0)
    assert np_(e.match_intent)[0] & (1 << (Intent.BUDDING.value - 1))  # the Budding bit is carried
    # eight steps later with the forecast off: Feeding is displayed at once
    assert row(at(e, False, **feeding)) == (F, F, F, 0)
    # a Hunting candidate counting through e is not interrupted by the Budding evaluation
    o = o
    seq = []
    for forecast in (False, True, False, False, False):
        o = at(o, forecast, **hunting)
        seq.append(row(o))
    assert seq == [(F, F, H, 1), (B_, F, H, 2), (F, F, H, 3), (H, H, H, 0), (H, H, H, 0)]
    # a row whose only match is the forecast: the held raw is Resting (4 evaluations needed), Budding shown
    blank = _obs(cfg, 400)
    rest = Intent.RESTING.value
    z = at(blank, True, flow_pos=500)
    assert row(z) == (B_, 0, rest, 1)
    # the same row once Resting is held (four evaluations without a match): Budding is shown over Resting,
    # a held intent like any other (spec 3.5), and Resting is back at once when the forecast ends
    r = blank
    for i in range(4):
        r = at(r, False, flow_pos=500)
        assert row(r)[:2] == ((0, 0) if i < 3 else (rest, rest))
    assert row(r) == (rest, rest, rest, 0)
    rb = at(r, True, flow_pos=500)
    assert row(rb) == (B_, rest, rest, 0) and np_(rb.match_intent)[0] & 1
    assert row(at(rb, False, flow_pos=500)) == (rest, rest, rest, 0)
    # not warm: nothing is displayed, even if the forecast holds
    cold = evaluate(_obs(cfg, 10, intent=feeding), cfg, rs, body=_overlay_inputs(cfg, forecast=True)[1],
                    oi=_overlay_inputs(cfg, forecast=True)[0])  # fmt: skip
    assert not np_(cold.label_intent).any() and not np_(cold.match_intent).any() and not np_(cold.held).any()
    # ... also where the row is old enough to bud (n_obs >= min_age = 20) but not yet warm (24): the forecast
    # holds and nothing is displayed
    assert cfg.repro.min_age <= 22 < rs.intent_warm
    oi_f, body_f = _overlay_inputs(cfg, forecast=True)
    cool = evaluate(_obs(cfg, 22, intent=feeding), cfg, rs, body=body_f, oi=oi_f)
    assert not np_(cool.label_intent).any() and not np_(cool.match_intent).any()
    assert np_(evaluate(_obs(cfg, 24, intent=feeding), cfg, rs, body=body_f, oi=oi_f).label_intent).all()
    # the age gate binds where a row is warm before it can bud: with intent_warm 8 and min_age 20 a row of
    # 15 observed steps is evaluated, the forecast holds on its energy and flow, and it is too young to bud;
    # at 19 it is still too young (the gate reads n_obs, not n_obs + 1), at 20 it buds
    young = dataclasses.replace(rs, intent_warm=8)
    assert young.intent_warm <= 15 < cfg.repro.min_age <= 20
    for n_obs, buds in ((15, False), (19, False), (20, True)):
        g = evaluate(_obs(cfg, n_obs, intent=feeding), cfg, young, body=body_f, oi=oi_f)
        bit = np_(g.match_intent) & 1
        assert bool(bit.all()) == buds and bool(bit.any()) == buds
        assert bool((np_(g.label_intent) == B_).all()) == buds
        assert np_(g.match_intent).all()  # evaluated: the other bits (or Resting) are set whatever the age
    # a bond to row 0 is a bond (the count is partner >= 0): it shifts the forecast's edge by h·bond_upkeep
    slot0 = mx.concatenate([mx.zeros_like(oi_f.partner[:, :1]), oi_f.partner[:, 1:]], axis=1)
    bonded = evaluate(_obs(cfg, 400, intent=feeding), cfg, rs, body=body_f, oi=oi_f._replace(partner=slot0))
    assert not np_(bonded.label_intent).any() and not (np_(bonded.match_intent) & (1 << 0)).any()
    unbonded = evaluate(_obs(cfg, 400, intent=feeding), cfg, rs, body=body_f, oi=oi_f)
    assert (np_(unbonded.match_intent) & (1 << 0)).all()
    # the inputs are required, never silently dropped
    with pytest.raises(ValueError, match="needs oi"):
        evaluate(_obs(cfg, 400), cfg, rs, body=_overlay_inputs(cfg, forecast=True)[1])
    with pytest.raises(ValueError, match="settler_body"):
        evaluate(_obs(cfg, 400), cfg, rs, oi=_overlay_inputs(cfg, forecast=True)[0])
    # the version-1 rules need neither, and have no overlay: label and held are the same array of values
    v1 = evaluate(_obs(cfg, 400, intent=feeding), cfg, RULES_V1)
    assert np.array_equal(np_(v1.label_intent), np_(v1.held))


def test_evaluate_holds_the_intent_in_the_rule_sets_order(micro_cfg: Config) -> None:
    """``evaluate`` hands ``rs.intent_order`` to the first match: with Caring and Defending both holding the
    held intent is Caring, with Feeding and Signalling both Feeding; the version-1 order says the opposite."""
    cfg = micro_cfg
    both = {
        Intent.CARING: {"care": 40, "defend": 255, "hurt": 1},
        Intent.FEEDING: {"harvest": 300, "emit": 255},
    }
    v1_wins = {Intent.CARING: Intent.DEFENDING, Intent.FEEDING: Intent.SIGNALLING}
    for want, intent in both.items():
        o = _obs(cfg, 400, intent=intent)
        oi, body = _overlay_inputs(cfg, forecast=False)
        for _ in range(RULES_V2.hold):
            o = evaluate(o, cfg, RULES_V2, body=body, oi=oi)
        assert (np_(o.held) == want.value).all() and (np_(o.label_intent) == want.value).all()
        o = _obs(cfg, 400, intent=intent)
        for _ in range(RULES_V1.hold):
            o = evaluate(o, cfg, RULES_V1)
        assert (np_(o.held) == v1_wins[want].value).all()


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
    # version 2: the intent order is a field (in the digest), the columns grew by three
    assert RULES_V2.intent_order == (1, 2, 3, 5, 4, 6, 8, 7, 9) and RULES_V1.intent_order == tuple(
        range(1, 10)
    )
    assert [i for i in RULES_V2.intent_order] == [
        Intent.BUDDING.value, Intent.HUNTING.value, Intent.FLEEING.value, Intent.CARING.value,
        Intent.DEFENDING.value, Intent.BONDING.value, Intent.FEEDING.value, Intent.SIGNALLING.value,
        Intent.ROAMING.value,
    ]  # fmt: skip
    assert rules.N_INTENT_COLS == 10 and [c.name for c in rules.IntentCol][7:] == [
        "FORMED",
        "FLOW_POS",
        "FLOW_NEG",
    ]


def test_display_names_are_one_table_keyed_by_the_report_keys() -> None:
    """The words a player sees live in two tables; the keys stay the enum names (the reports, the metrics
    stream and the sidecar use them). Coordinator choices C8 and C9: Bonding is shown as Tethering, Generalist
    as Unspecialised."""
    assert tuple(ARCHETYPE_DISPLAY) == ARCHETYPE_NAMES and tuple(INTENT_DISPLAY) == INTENT_NAMES
    renamed = {k: v for k, v in (*ARCHETYPE_DISPLAY.items(), *INTENT_DISPLAY.items()) if v.lower() != k}
    assert renamed == {"generalist": "Unspecialised", "bonding": "Tethering"}
    assert all(v == k.capitalize() for k, v in INTENT_DISPLAY.items() if k != "bonding")
    assert all(v == k.capitalize() for k, v in ARCHETYPE_DISPLAY.items() if k != "generalist")
    with pytest.raises(TypeError):
        INTENT_DISPLAY["bonding"] = "Bonding"  # type: ignore[index]  # read only
