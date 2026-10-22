"""The play stream's colour modes (DESIGN §33.2, ``game/colors.py``): the three render modes are
``render.agent_colors`` unchanged, the four observer modes read the slots and labels through a fixed palette,
hurt agents are red in every mode, and the device slot maps agree with their host twins in ``IDNT``."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from alife import streamrec
from alife.config import Config
from alife.game import colors
from alife.game.colors import (
    ARCH_PALETTE,
    HURT_RGB,
    INTENT_PALETTE,
    NONE_RGB,
    OTHER_RGB,
    SLOT_PALETTE,
    UNKNOWN_RGB,
    ColorInputs,
    house_slot_bytes,
    kin_slot_bytes,
    observer_colors,
    play_colors,
)
from alife.game.observer import _house_bytes, _slot_bytes
from alife.game.rules import ARCHETYPE_DISPLAY, INTENT_DISPLAY
from alife.game.social import HOUSE_UNKNOWN
from alife.render import agent_colors
from alife.state import init_state, to_numpy


def _pack(c: tuple[int, int, int]) -> int:
    return (c[0] << 16) | (c[1] << 8) | c[2]


def test_palettes_are_distinct_and_leave_red_to_the_hurt_rule() -> None:
    assert len(SLOT_PALETTE) == 16 and len(set(SLOT_PALETTE)) == 16
    assert (
        len(ARCH_PALETTE) == len(ARCHETYPE_DISPLAY) + 1
        and len(set(ARCH_PALETTE[1:])) == len(ARCH_PALETTE) - 1
    )
    assert (
        len(INTENT_PALETTE) == len(INTENT_DISPLAY) + 1
        and len(set(INTENT_PALETTE[1:])) == len(INTENT_PALETTE) - 1
    )
    for palette in (SLOT_PALETTE, ARCH_PALETTE, INTENT_PALETTE, (OTHER_RGB, UNKNOWN_RGB, NONE_RGB)):
        assert all(len(c) == 3 and all(0 <= v <= 255 for v in c) for c in palette)
        assert HURT_RGB not in palette
    assert {OTHER_RGB, UNKNOWN_RGB, NONE_RGB}.isdisjoint(SLOT_PALETTE)


def test_the_header_lists_the_modes_this_module_serves() -> None:
    assert colors.PLAY_MODES == streamrec.COLOR_MODES
    assert (
        colors.PLAY_MODES[:3] == ("tag", "energy", "clade") and colors.PLAY_MODES[3:] == colors.OBSERVER_MODES
    )


def _inputs(n: int, **cols: np.ndarray) -> ColorInputs:
    zero = np.zeros(n, np.int32)
    return ColorInputs(
        mx.array(cols.get("house", np.full(n, 255)).astype(np.uint8)),
        mx.array(cols.get("kin", np.full(n, 255)).astype(np.uint8)),
        mx.array(cols.get("arch", zero)),
        mx.array(cols.get("intent", zero)),
    )


def test_slot_modes_colour_by_the_slot_byte() -> None:
    house = np.array([0, 1, 15, 253, 254, 255, 7, 3])
    no_hurt = mx.zeros(len(house), dtype=mx.bool_)
    for mode in ("house", "kin"):
        col = to_numpy(observer_colors(mode, _inputs(len(house), **{mode: house}), no_hurt))
        want = [
            _pack(SLOT_PALETTE[b])
            if b < 16
            else {253: _pack(UNKNOWN_RGB), 254: _pack(OTHER_RGB)}.get(b, _pack(NONE_RGB))
            for b in house.tolist()
        ]
        assert col.tolist() == want, mode
    arch = np.arange(8)
    col = to_numpy(observer_colors("archetype", _inputs(8, arch=arch), mx.zeros(8, dtype=mx.bool_)))
    assert col.tolist() == [_pack(c) for c in ARCH_PALETTE]
    intent = np.arange(11)
    col = to_numpy(observer_colors("intent", _inputs(11, intent=intent), mx.zeros(11, dtype=mx.bool_)))
    assert col.tolist() == [_pack(c) for c in INTENT_PALETTE]


def test_hurt_rows_are_red_in_every_observer_mode() -> None:
    n = 6
    hurt = mx.array([True, False, True, False, False, True])
    inp = _inputs(n, house=np.arange(n), kin=np.arange(n), arch=np.arange(1, n + 1) % 8, intent=np.arange(n))
    for mode in colors.OBSERVER_MODES:
        col = to_numpy(observer_colors(mode, inp, hurt))
        assert (col[[0, 2, 5]] == _pack(HURT_RGB)).all() and (col[[1, 3, 4]] != _pack(HURT_RGB)).all(), mode


def test_out_of_range_labels_are_clamped_not_read_past_the_table() -> None:
    inp = _inputs(2, arch=np.array([99, -3]), intent=np.array([99, -3]))
    off = mx.zeros(2, dtype=mx.bool_)
    assert to_numpy(observer_colors("archetype", inp, off)).tolist() == [
        _pack(ARCH_PALETTE[-1]),
        _pack(ARCH_PALETTE[0]),
    ]
    assert to_numpy(observer_colors("intent", inp, off)).tolist() == [
        _pack(INTENT_PALETTE[-1]),
        _pack(INTENT_PALETTE[0]),
    ]


def test_render_modes_delegate_to_the_renderer(tiny_cfg: Config) -> None:
    s = init_state(tiny_cfg)
    s = s._replace(
        last_loss=mx.where(mx.arange(s.last_loss.shape[0]) % 5 == 0, 1, 0).astype(s.last_loss.dtype)
    )
    for mode in ("tag", "energy", "clade"):
        assert np.array_equal(
            to_numpy(play_colors(s, tiny_cfg, mode)), to_numpy(agent_colors(s, tiny_cfg, mode))
        )


def test_observer_modes_use_the_states_hurt_flag(tiny_cfg: Config) -> None:
    s = init_state(tiny_cfg)
    n = s.last_loss.shape[0]
    hurt = np.arange(n) % 3 == 0
    s = s._replace(last_loss=mx.array(hurt.astype(np.int32)).astype(s.last_loss.dtype))
    col = to_numpy(play_colors(s, tiny_cfg, "kin", _inputs(n, kin=np.arange(n) % 16)))
    assert (col[hurt] == _pack(HURT_RGB)).all()
    assert col[~hurt].tolist() == [_pack(SLOT_PALETTE[i % 16]) for i in np.flatnonzero(~hurt)]


def test_play_colors_refuses_what_it_cannot_serve(tiny_cfg: Config) -> None:
    s = init_state(tiny_cfg)
    with pytest.raises(ValueError, match="needs the observer"):
        play_colors(s, tiny_cfg, "house")
    with pytest.raises(ValueError, match="unknown colour mode"):
        play_colors(s, tiny_cfg, "mauve")
    with pytest.raises(ValueError, match="unknown observer"):
        observer_colors("tag", _inputs(1), mx.zeros(1, dtype=mx.bool_))


def test_device_slot_maps_agree_with_the_idnt_host_twins() -> None:
    rng = np.random.default_rng(5)
    top = np.array([40, 7, 1000, 3, 12], np.int64)  # table order is not id order
    house = np.concatenate(
        [rng.choice(np.concatenate([top, [55, 56, 0]]), 200), [HOUSE_UNKNOWN, -1, 7]]
    ).astype(np.int64)
    dev = to_numpy(house_slot_bytes(mx.array(house), top))
    assert dev.dtype == np.uint8 and np.array_equal(dev, _house_bytes(house, top))
    assert {253, 254, 255} <= set(dev.tolist()) | {255} and dev[house == 7].tolist()[0] == 1 and dev[-1] == 1
    assert (dev[house == -1] == 255).all() and (dev[house == HOUSE_UNKNOWN] == 253).all()
    assert (dev[np.isin(house, [55, 56, 0])] == 254).all()
    empty = to_numpy(house_slot_bytes(mx.array(house), np.zeros(0, np.int64)))
    assert np.array_equal(empty, _house_bytes(house, np.zeros(0, np.int64)))
    assert set(empty.tolist()) == {253, 254, 255}
    kin = np.array([-1, 0, 5, 15, 16, 3], np.int32)
    assert to_numpy(kin_slot_bytes(mx.array(kin), 16)).tolist() == [255, 0, 5, 15, 254, 3]
    assert np.array_equal(to_numpy(kin_slot_bytes(mx.array(kin), 16)), _slot_bytes(kin, 16))
    assert np.array_equal(to_numpy(kin_slot_bytes(mx.array(kin), 4)), _slot_bytes(kin, 4))


def test_a_house_table_of_more_than_sixteen_is_refused() -> None:
    with pytest.raises(ValueError, match="at most 16"):
        house_slot_bytes(mx.zeros(4, dtype=mx.int64), np.arange(17))
