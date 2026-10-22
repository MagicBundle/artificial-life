"""§32 the version-3 rules (the storm world): the column layout, the new fields and their validation, Drifter,
Floater and Sinker at the edges of their thresholds, the match matrix with its placeholder column, and the
priority of the new labels (after every older one)."""

from __future__ import annotations

import dataclasses
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from alife import layout
from alife.config import Config
from alife.game import rules
from alife.game.rules import (
    ARCH_PRIORITY,
    RULES_V1,
    RULES_V2,
    RULES_V3,
    VQ_MAX,
    Archetype,
    RuleSet,
    arch_layout,
    for_2d,
    n_arch_cols,
)
from alife.state import to_numpy

B = layout.B
V3_2D = for_2d(RULES_V3)


def np_(a: mx.array) -> np.ndarray:
    return to_numpy(a)


# Pinned when the pilot window (spec 5.6, C-W2) froze the thresholds (DESIGN §32): the physics-derived
# starting values passed on the pilot and were kept. Any edit to a V3 threshold, window or field changes them
# and must come with a new INTENT_VERSION (the sidecar stores the digest, and the validity rows key on it).
RULES_V3_DIGEST = "3515d96edb93e276363bdefe4e7025ffc0da6d572165f8d1b53d33abb380a17c"
RULES_V3_2D_DIGEST = "268730d9691145a40b018a1206ce934b56df866985db9ac04375d470c5bd3abf"


def test_the_frozen_version_3_rules_are_pinned() -> None:
    assert RULES_V3.digest() == RULES_V3_DIGEST and V3_2D.digest() == RULES_V3_2D_DIGEST
    assert RULES_V2.digest() == "dc78f771d515b845501cd609275b362acfbc60c47853e813a6d129d3a5cab4f7"
    frozen = {
        "wind_rel": True, "vertical": True, "drifter_wind": (38, 1), "drifter_ratio": (1, 2),
        "floater_lift": (77, 1), "floater_z": 26, "floater_vz": 19, "sinker_lift": (26, 1), "sinker_vz": 19,
    }  # fmt: skip
    assert {k: getattr(RULES_V3, k) for k in frozen} == frozen
    # everything else is version 2's
    older = {f.name: getattr(RULES_V3, f.name) for f in dataclasses.fields(RuleSet) if f.name not in frozen}
    assert older == {
        f.name: getattr(RULES_V2, f.name) for f in dataclasses.fields(RuleSet) if f.name not in frozen
    }


# --- the layout ---------------------------------------------------------------------------------------------


def test_the_layout_of_each_rule_set() -> None:
    """Version 2 keeps its 5 + 2·nd columns; wind_rel appends WIND, vertical ZPOS and LIFT."""
    assert (n_arch_cols(2), n_arch_cols(3)) == (9, 11)
    assert (n_arch_cols(2, RULES_V2), n_arch_cols(3, RULES_V1)) == (9, 11)
    assert (n_arch_cols(2, V3_2D), n_arch_cols(3, RULES_V3)) == (10, 14)
    lay = arch_layout(3, RULES_V3)
    assert (lay.nd, lay.v_pos, lay.v_neg, lay.wind, lay.zpos, lay.lift, lay.width) == (
        3,
        5,
        8,
        11,
        12,
        13,
        14,
    )
    lay2 = arch_layout(2, V3_2D)
    assert (lay2.v_pos, lay2.v_neg, lay2.wind, lay2.zpos, lay2.lift, lay2.width) == (5, 7, 9, None, None, 10)
    assert arch_layout(3).wind is None and arch_layout(3).width == 11


def test_the_v3_fields_are_inert_in_the_older_rule_sets() -> None:
    inert = {k: rules._POST_V1_INERT[k] for k in ("wind_rel", "vertical", "drifter_wind", "drifter_ratio")}
    assert all(getattr(RULES_V2, k) == v for k, v in inert.items())
    assert RULES_V3 != RULES_V2 and RULES_V3.digest() != RULES_V2.digest()
    # V3 with its version-3 fields at their inert values is exactly the version-2 rule set, digest included
    off = dataclasses.replace(
        RULES_V3, **{k: rules._POST_V1_INERT[k] for k in rules._POST_V1_INERT if k in V3}
    )
    assert off == RULES_V2 and off.digest() == RULES_V2.digest()


V3 = {
    "wind_rel", "vertical", "drifter_wind", "drifter_ratio", "floater_lift", "floater_z", "floater_vz",
    "sinker_lift", "sinker_vz",
}  # fmt: skip


def test_every_v3_field_enters_the_digest() -> None:
    changed: dict[str, Any] = {
        "drifter_wind": (39, 1), "drifter_ratio": (2, 3), "floater_lift": (78, 1), "floater_z": 27,
        "floater_vz": 20, "sinker_lift": (27, 1), "sinker_vz": 20,
    }  # fmt: skip
    digests = {RULES_V3.digest()}
    for name, value in changed.items():
        digests.add(dataclasses.replace(RULES_V3, **{name: value}).digest())
    digests.add(dataclasses.replace(RULES_V3, vertical=False).digest())
    digests.add(V3_2D.digest())
    assert len(digests) == len(changed) + 3
    assert set(changed) | {"wind_rel", "vertical"} == V3


def test_the_v3_fields_are_validated() -> None:
    with pytest.raises(ValueError, match="vertical needs wind_rel"):
        RuleSet(vertical=True)
    with pytest.raises(ValueError, match="needs wind_rel"):
        RuleSet(drifter_wind=(38, 1), drifter_ratio=(1, 2))
    with pytest.raises(ValueError, match="both set"):
        dataclasses.replace(RULES_V3, drifter_ratio=(0, 1))
    with pytest.raises(ValueError, match="both set"):
        dataclasses.replace(RULES_V3, drifter_wind=(0, 1))
    with pytest.raises(ValueError, match="floater_z"):
        dataclasses.replace(RULES_V3, floater_z=-1)
    with pytest.raises(ValueError, match="floater_lift must be"):
        dataclasses.replace(RULES_V3, floater_lift=(77, 0))
    # a 2D variant has no vertical labels
    assert V3_2D.wind_rel and not V3_2D.vertical and V3_2D.floater_z == 0 and V3_2D.drifter_wind == (38, 1)


def test_check_bounds_the_new_columns_and_rejects_vertical_in_2d(
    tiny_cfg: Config, tiny3d_cfg: Config
) -> None:
    RULES_V3.check(tiny3d_cfg)
    V3_2D.check(tiny_cfg)
    with pytest.raises(ValueError, match=r"vertical.*3D"):
        RULES_V3.check(tiny_cfg)
    # WIND, ZPOS and LIFT are int32 accumulators like the others: a huge window overflows the 15-bit WIND
    with pytest.raises(ValueError, match=r"arch\.(wind|speed|velocity)"):
        dataclasses.replace(RULES_V3, arch_k=17).check(tiny3d_cfg)
    # the new int32 ratio tests are bounded
    for name, largest in (("drifter_wind", VQ_MAX), ("drifter_ratio", VQ_MAX), ("floater_lift", 256)):
        limit = (2**31 - 1) // largest
        with pytest.raises(ValueError, match=f"ratio {name} = "):
            dataclasses.replace(RULES_V3, **{name: (limit + 1, 1)}).check(tiny3d_cfg)


# --- the three rules at their thresholds --------------------------------------------------------------------


def _m6(nd: int, rs: RuleSet, **cols: Any) -> mx.array:
    """(1, width) means of the accumulators of ``rs``: ``speed``, ``vpos``/``vneg`` (sequences of nd),
    ``wind``, ``zpos`` and ``lift``."""
    lay = arch_layout(nd, rs)
    row = np.zeros((1, lay.width), np.int32)
    row[0, rules.ArchCol.SPEED.value] = cols.pop("speed", 0)
    row[0, lay.v_pos : lay.v_pos + nd] = cols.pop("vpos", (0,) * nd)
    row[0, lay.v_neg : lay.v_neg + nd] = cols.pop("vneg", (0,) * nd)
    for name in ("wind", "zpos", "lift"):
        col = getattr(lay, name)
        value = cols.pop(name, 0)
        if col is not None:
            row[0, col] = value
        else:
            assert value == 0, name
    assert not cols, cols
    return mx.array(row)


def _match(cfg: Config, nd: int, rs: RuleSet = RULES_V3, **cols: Any) -> dict[str, bool]:
    m6s = mx.zeros((1, B, 2), mx.int32)
    body = mx.zeros((1,), mx.int32)
    out = np_(rules.archetype_matches(_m6(nd, rs, **cols), m6s, cfg, rs, body))
    assert out.shape == (1, 10) and out.dtype == bool
    return {a.name.lower(): bool(out[0, a.value - 1]) for a in Archetype if a is not Archetype.NONE}


def test_drifter_thresholds(tiny3d_cfg: Config) -> None:
    """m_wind >= 38 (0.015 u/step) and 2·m_wind >= m_speed_rel."""
    cfg = tiny3d_cfg
    assert _match(cfg, 3, speed=76, wind=38)["drifter"]
    assert not _match(cfg, 3, speed=76, wind=37)["drifter"]  # the wind is too small ...
    assert not _match(cfg, 3, speed=74, wind=37)["drifter"]  # ... even when it is most of the motion
    assert not _match(cfg, 3, speed=77, wind=38)["drifter"]  # the wind is under half of the relative speed
    assert _match(cfg, 3, speed=0, wind=38)["drifter"]  # carried, not moving on its own
    assert not _match(cfg, 3)["drifter"]
    # the same in 2D, where the layout has no ZPOS or LIFT
    assert (
        _match(cfg, 2, V3_2D, speed=76, wind=38)["drifter"] and not _match(cfg, 2, V3_2D, wind=37)["drifter"]
    )


def test_floater_thresholds(tiny3d_cfg: Config) -> None:
    """m_lift >= 77 and m_zpos >= 26 and |m_vz_rel| <= 19."""
    cfg = tiny3d_cfg
    base = {"lift": 77, "zpos": 26}
    assert _match(cfg, 3, **base)["floater"]
    assert not _match(cfg, 3, lift=76, zpos=26)["floater"] and not _match(cfg, 3, lift=77, zpos=25)["floater"]
    assert _match(cfg, 3, vpos=(0, 0, 19), **base)["floater"]
    assert not _match(cfg, 3, vpos=(0, 0, 20), **base)["floater"]
    assert _match(cfg, 3, vneg=(0, 0, 19), **base)["floater"]
    assert not _match(cfg, 3, vneg=(0, 0, 20), **base)["floater"]
    # the vertical speed is the z component only (a fast horizontal drift does not stop a Floater)
    assert _match(cfg, 3, vpos=(200, 200, 0), vneg=(0, 0, 0), **base)["floater"]
    # the mean of the halves: an up-and-down of 30 each is no vertical motion on average
    assert _match(cfg, 3, vpos=(0, 0, 30), vneg=(0, 0, 30), **base)["floater"]


def test_sinker_thresholds(tiny3d_cfg: Config) -> None:
    """m_lift < 26 and m_vz_rel <= -19."""
    cfg = tiny3d_cfg
    assert _match(cfg, 3, lift=25, vneg=(0, 0, 19))["sinker"]
    assert not _match(cfg, 3, lift=26, vneg=(0, 0, 19))["sinker"]
    assert not _match(cfg, 3, lift=25, vneg=(0, 0, 18))["sinker"]
    assert not _match(cfg, 3, lift=25, vpos=(0, 0, 19))["sinker"]  # rising
    assert _match(cfg, 3, lift=0, vpos=(0, 0, 10), vneg=(0, 0, 30))["sinker"]  # net -20
    assert not _match(cfg, 3, lift=25)["sinker"]


def test_a_rule_set_without_the_labels_never_matches_them(tiny3d_cfg: Config) -> None:
    for rs in (RULES_V1, RULES_V2):
        m = _match(tiny3d_cfg, 3, rs, speed=100, vneg=(0, 0, 200))
        assert not (m["drifter"] or m["floater"] or m["sinker"] or m["generalist"])
    # wind_rel alone has the Drifter, not the vertical pair
    rs = dataclasses.replace(RULES_V3, vertical=False, floater_lift=(0, 1), floater_z=0, floater_vz=0)
    rs = dataclasses.replace(rs, sinker_lift=(0, 1), sinker_vz=0)
    m = _match(tiny3d_cfg, 3, rs, wind=100, speed=100)
    assert m["drifter"] and not (m["floater"] or m["sinker"])


def test_the_placeholder_column_keeps_the_bits_apart(tiny3d_cfg: Config) -> None:
    """Generalist is bit 6 and always false in the matrix: Drifter, Floater and Sinker are bits 7, 8, 9."""
    cfg = tiny3d_cfg
    for want, kw in (
        (Archetype.DRIFTER, {"wind": 100, "speed": 100}),
        (Archetype.FLOATER, {"lift": 100, "zpos": 100}),
        (Archetype.SINKER, {"lift": 0, "vneg": (0, 0, 100)}),
    ):
        m6s = mx.zeros((1, B, 2), mx.int32)
        match = rules.archetype_matches(_m6(3, RULES_V3, **kw), m6s, cfg, RULES_V3, mx.zeros((1,), mx.int32))
        bits = int(np_(rules.pack_bits(match))[0])
        assert not (bits >> (Archetype.GENERALIST.value - 1)) & 1
        assert (bits >> (want.value - 1)) & 1, want


def test_the_new_labels_come_after_every_older_one(tiny3d_cfg: Config) -> None:
    assert ARCH_PRIORITY[-3:] == (Archetype.DRIFTER, Archetype.FLOATER, Archetype.SINKER)
    cfg = tiny3d_cfg

    def labels(**kw: Any) -> tuple[int, int]:
        m6s = mx.zeros((1, B, 2), mx.int32)
        match = rules.archetype_matches(_m6(3, RULES_V3, **kw), m6s, cfg, RULES_V3, mx.zeros((1,), mx.int32))
        p, s = rules.archetype_labels(match)
        return int(np_(p)[0]), int(np_(s)[0])

    # an at-rest row is a Settler (speed 0 < 25.6); a Drifter, Floater or Sinker only as its secondary label
    assert labels(wind=100) == (Archetype.SETTLER.value, Archetype.DRIFTER.value)
    assert labels(lift=100, zpos=100) == (Archetype.SETTLER.value, Archetype.FLOATER.value)
    assert labels(lift=0, vneg=(0, 0, 100)) == (Archetype.SETTLER.value, Archetype.SINKER.value)
    # moving fast with a persistent heading: a Nomad first
    fast = {"speed": 200, "vpos": (200, 0, 0)}
    assert labels(**fast, wind=120) == (Archetype.NOMAD.value, Archetype.DRIFTER.value)
    # the only match: Drifter is the primary label
    mid = {
        "speed": 60,
        "vpos": (20, 20, 20),
        "vneg": (0, 0, 0),
    }  # not persistent enough for a Nomad, not slow
    assert labels(**mid, wind=40) == (Archetype.DRIFTER.value, 0)
    # no rule matches: Generalist
    assert labels(**mid) == (Archetype.GENERALIST.value, 0)


def test_the_width_must_match_the_layout(tiny3d_cfg: Config) -> None:
    m6s = mx.zeros((1, B, 2), mx.int32)
    with pytest.raises(ValueError, match="columns"):
        rules.archetype_matches(
            _m6(3, RULES_V2), m6s, tiny3d_cfg, RULES_V3, mx.zeros((1,), mx.int32), arch_layout(3, RULES_V3)
        )
