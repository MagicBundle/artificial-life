"""The colour modes of the play stream (DESIGN §33.2): the render modes ``tag energy clade`` and, from the
observer, ``house kin archetype intent``.

``play_colors`` returns the ``(N,) uint32`` ``0x00RRGGBB`` column of ``FRMA.rgb``. The first three modes are
``render.agent_colors`` unchanged (hurt agents are red). The four observer modes colour a row by the slot the
observer holds for it, through a fixed palette, and keep the hurt rule: an agent that lost energy last step is
red in every mode.

**Slots.** A house, kin group or colony row is described by one byte, the value ``IDNT`` carries (§33.1.3):
``0..15`` the slot in the listed top-16 table, ``253`` an unknown immigrant lineage (houses of an observer
attached mid-run), ``254`` listed as other, ``255`` none. ``SLOT_PALETTE`` (16 colours, none red) is
normative: the client draws the cast swatches and the territory overlay with the same table, so a swatch is
the colour of its organisms. Archetype and intent colours are indexed by the ``rules.Archetype`` /
``rules.Intent`` value (0 = unlabelled).

Device code, no host reads: the slot lookups are comparisons and takes, and the whole column is one lazy
graph the caller evaluates with the frame. ``tests/test_play_colors.py`` pins the palette, the lookups and the
agreement of the device and host slot maps.
"""

from __future__ import annotations

from typing import Final, NamedTuple

import mlx.core as mx
import numpy as np

from alife.config import Config
from alife.game.rules import Archetype, Intent
from alife.game.social import HOUSE_UNKNOWN
from alife.render import COLOR_MODES as RENDER_MODES
from alife.render import agent_colors
from alife.state import State

__all__ = [
    "ARCH_PALETTE",
    "HURT_RGB",
    "INTENT_PALETTE",
    "NONE_RGB",
    "OBSERVER_MODES",
    "OTHER_RGB",
    "PLAY_MODES",
    "SLOT_PALETTE",
    "UNKNOWN_RGB",
    "ColorInputs",
    "house_slot_bytes",
    "kin_slot_bytes",
    "observer_colors",
    "play_colors",
]

OBSERVER_MODES: Final = ("house", "kin", "archetype", "intent")
PLAY_MODES: Final = (*RENDER_MODES, *OBSERVER_MODES)  # the play header's ``color_modes``
TABLE: Final = 16  # rows of a house or kin table (``SocialRules.slots``)

HURT_RGB: Final = (255, 0, 0)
OTHER_RGB: Final = (128, 128, 128)  # listed as other (slot 254)
UNKNOWN_RGB: Final = (110, 110, 150)  # an unknown immigrant lineage (slot 253)
NONE_RGB: Final = (56, 56, 56)  # no slot (255) and unlabelled rows

SLOT_PALETTE: Final = (
    (230, 159, 0),
    (86, 180, 233),
    (0, 158, 115),
    (240, 228, 66),
    (0, 114, 178),
    (204, 121, 167),
    (148, 103, 189),
    (140, 86, 75),
    (127, 201, 127),
    (253, 192, 134),
    (56, 108, 176),
    (190, 174, 212),
    (23, 190, 207),
    (188, 189, 34),
    (255, 127, 191),
    (166, 206, 227),
)
# By ``rules.Archetype`` value (index 0 = unlabelled): gatherer, hunter, provider, trader, nomad, settler,
# generalist ("Unspecialised": a neutral light grey, so it does not read as a class of its own).
ARCH_PALETTE: Final = (
    NONE_RGB,
    (0, 158, 115),
    (213, 94, 0),
    (240, 228, 66),
    (86, 180, 233),
    (204, 121, 167),
    (148, 103, 189),
    (200, 200, 200),
)
# By ``rules.Intent`` value (index 0 = none): budding, hunting, fleeing, defending, caring, bonding
# ("Tethering"), signalling, feeding, roaming, resting.
INTENT_PALETTE: Final = (
    NONE_RGB,
    (255, 127, 191),
    (213, 94, 0),
    (240, 228, 66),
    (0, 114, 178),
    (127, 201, 127),
    (148, 103, 189),
    (23, 190, 207),
    (0, 158, 115),
    (230, 159, 0),
    (96, 96, 96),
)
assert len(ARCH_PALETTE) == len(Archetype) and len(INTENT_PALETTE) == len(Intent)
assert len(SLOT_PALETTE) == TABLE
assert HURT_RGB not in (*SLOT_PALETTE, *ARCH_PALETTE, *INTENT_PALETTE)

_SLOT_UNKNOWN: Final = 253
_SLOT_OTHER: Final = 254
_SLOT_NONE: Final = 255


def _rgb(c: tuple[int, int, int]) -> int:
    return (c[0] << 16) | (c[1] << 8) | c[2]


def _table(colors: tuple[tuple[int, int, int], ...]) -> mx.array:
    return mx.array([_rgb(c) for c in colors], dtype=mx.uint32)


# slot byte -> row of the slot table: 0..15, then unknown, other, none
_SLOT_TABLE: Final = (*SLOT_PALETTE, UNKNOWN_RGB, OTHER_RGB, NONE_RGB)


class ColorInputs(NamedTuple):
    """What the observer modes read (device arrays, ``(N,)``): the house and kin slot bytes, the primary
    archetype and the displayed intent."""

    house: mx.array  # uint8 slot byte
    kin: mx.array  # uint8 slot byte
    arch: mx.array  # int32 ``rules.Archetype``
    intent: mx.array  # int32 ``rules.Intent``


def kin_slot_bytes(slot: mx.array, f: int = TABLE) -> mx.array:
    """The slot byte of ``SocState.kin`` / ``colony`` values: ``0..f-1`` as they are, ``f`` (OTHER) to 254, a
    negative value (dead, or no colony) to 255. ``f`` is ``SocialRules.slots`` and at most 16."""
    s = slot.astype(mx.int32)
    return mx.where(s < 0, _SLOT_NONE, mx.where(s >= f, _SLOT_OTHER, s)).astype(mx.uint8)


def house_slot_bytes(house: mx.array, top: np.ndarray) -> mx.array:
    """The slot byte of ``SocState.house`` ids (int64): the index in ``top`` (the listed house ids, in table
    order, at most 16) for a listed house, 253 for ``HOUSE_UNKNOWN``, 254 for any other house, 255 for a dead
    row."""
    ids = np.asarray(top, np.int64).reshape(-1)
    if ids.size > TABLE:
        raise ValueError(f"a house table lists at most {TABLE} houses, got {ids.size}")
    other = mx.where(house >= 0, _SLOT_OTHER, _SLOT_NONE)
    if ids.size == 0:
        slot = other
    else:
        eq = house[:, None] == mx.array(ids)[None, :]
        slot = mx.where(eq.any(axis=1), mx.argmax(eq, axis=1), other)
    return mx.where(house == HOUSE_UNKNOWN, _SLOT_UNKNOWN, slot).astype(mx.uint8)


def _from_table(colors: tuple[tuple[int, int, int], ...], index: mx.array) -> mx.array:
    return mx.take(_table(colors), index.astype(mx.int32))


def observer_colors(mode: str, inputs: ColorInputs, hurt: mx.array) -> mx.array:
    """``(N,) uint32`` colours of an observer mode; rows with ``hurt`` (lost energy last step) are red."""
    if mode == "house" or mode == "kin":
        byte = inputs.house if mode == "house" else inputs.kin
        b = byte.astype(mx.int32)
        index = mx.where(b < TABLE, b, b - _SLOT_UNKNOWN + TABLE)  # 253 -> 16, 254 -> 17, 255 -> 18
        rgb = _from_table(_SLOT_TABLE, index)
    elif mode == "archetype":
        rgb = _from_table(ARCH_PALETTE, mx.clip(inputs.arch, 0, len(ARCH_PALETTE) - 1))
    elif mode == "intent":
        rgb = _from_table(INTENT_PALETTE, mx.clip(inputs.intent, 0, len(INTENT_PALETTE) - 1))
    else:
        raise ValueError(f"unknown observer colour mode {mode!r}; expected one of {OBSERVER_MODES}")
    return mx.where(hurt, mx.array(_rgb(HURT_RGB), dtype=mx.uint32), rgb)


def play_colors(state: State, cfg: Config, mode: str, inputs: ColorInputs | None = None) -> mx.array:
    """``(N,) uint32`` ``0x00RRGGBB`` agent colours in a mode of ``PLAY_MODES``. ``tag``, ``energy`` and
    ``clade`` are ``render.agent_colors``; the observer modes need ``inputs``."""
    if mode in RENDER_MODES:
        return agent_colors(state, cfg, mode)
    if mode not in OBSERVER_MODES:
        raise ValueError(f"unknown colour mode {mode!r}; expected one of {PLAY_MODES}")
    if inputs is None:
        raise ValueError(f"colour mode {mode!r} needs the observer's slots and labels (ColorInputs)")
    return observer_colors(mode, inputs, state.last_loss > 0)
