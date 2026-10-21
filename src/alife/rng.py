"""Versioned RNG key layout (§14.1). ``state.key`` is the only simulation RNG state.

``mx.random.split`` is not prefix-stable (F3), so the stream count is frozen: a new consumer takes a
reserved index, and changing the layout bumps ``RNG_LAYOUT_VERSION``. Taking a reserved index leaves every
existing stream's key unchanged, so the older version stays in ``RNG_COMPATIBLE_VERSIONS`` and its
checkpoints still load and resume. A change that moves an existing key would bump the version without
adding the old one to the set.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final

import mlx.core as mx

NUM_STREAMS: Final = 16
# 2: Stream.CATASTROPHE took reserved index 8; 3: Stream.WEATHER took reserved index 9. The key layout
# is unchanged by both.
RNG_LAYOUT_VERSION: Final = 3
RNG_COMPATIBLE_VERSIONS: Final = frozenset({1, 2, 3})  # layout versions whose checkpoints this code resumes


class Stream(IntEnum):
    """Per-step streams ``ks[Stream.X]``; indices 10–15 are reserved."""

    NOISE = 0
    BOND_PRIO = 1
    BIRTH_PRIO = 2
    PLACEMENT = 3
    MUT_W1 = 4
    MUT_W2 = 5
    MUT_GENES = 6
    IMMIGRANT = 7
    CATASTROPHE = 8
    WEATHER = 9  # W1: split in two statically, the weather disturbance key and the storm hazard key


class InitStream(IntEnum):
    """Initialisation streams from ``init_keys(seed)``; indices 8–15 are reserved."""

    POS = 0
    HEADING = 1
    W1 = 2
    W2 = 3
    TAG = 4
    PATCH_POS = 5
    PATCH_VEL = 6
    STATE_KEY = 7


def step_keys(key: mx.array) -> tuple[mx.array, mx.array]:
    """Advance the state key once: returns ``(key_next (2,), ks (NUM_STREAMS, 2))``."""
    k = mx.random.split(key, 2)
    return k[0], mx.random.split(k[1], NUM_STREAMS)


def init_keys(seed: int) -> mx.array:
    """The ``(NUM_STREAMS, 2)`` initialisation keys for a seed (indexed by ``InitStream``)."""
    return mx.random.split(mx.random.key(seed), NUM_STREAMS)


def observer_key(seed: int, step: int) -> mx.array:
    """Key for observers (metrics sampling, viewer picks); never derived from ``state.key``."""
    return mx.random.key((seed * 1_000_003 + step) % 2**63)
