"""Versioned RNG key layout (§14.1). ``state.key`` is the only simulation RNG state.

``mx.random.split`` is not prefix-stable (F3), so the stream count is frozen: a new consumer takes a
reserved index, and changing the layout bumps ``RNG_LAYOUT_VERSION``. Taking a reserved index leaves every
existing stream's key unchanged, so the older version stays in ``RNG_COMPATIBLE_VERSIONS`` and its
checkpoints still load and resume. A change that moves an existing key would bump the version without
adding the old one to the set.

Feature extensions (DESIGN §43). The reserved per-step indices are assigned by extension index: ``10 + i``
with
arms 10, structures 11, bonds 12, comm 13, niche 14 (15 stays free). A feature draws nothing unless its
spec says
so (only arms does, in ``lifecycle.reproduce`` and at immigrants and init); a ``Stream``/``InitStream`` member
is created by the PR that first uses it, so an unused name never invites a draw. The draw-free features that
need deterministic pseudo-randomness use ``ops.hash32`` with their entry of ``HASH_SALTS`` (well-separated
32-bit constants, not ``10 + i``: salts differing only in low bits give weakly dependent streams, E18).
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final

import mlx.core as mx

NUM_STREAMS: Final = 16
# 2: Stream.CATASTROPHE took reserved index 8; 3: Stream.WEATHER took reserved index 9. The key layout
# is unchanged by both.
RNG_LAYOUT_VERSION: Final = 3
RNG_COMPATIBLE_VERSIONS: Final = frozenset(
    {1, 2, 3, 4}
)  # layout versions whose checkpoints this code resumes
# ``ops.hash32`` salt of each extension: the first 4 bytes (big endian) of
# sha256(b"alife.ext.hash32.salt/" + name), computed once and frozen as literals (changing one changes the
# trajectories of runs with that feature on; ``test_ext_salts.py`` pins the table and checks every pair).
HASH_SALTS: Final = {
    "arms": 0x20D53604,
    "structures": 0x243C7C36,
    "bonds": 0x9E1A3B74,
    "comm": 0x6E00A046,
    "niche": 0x20622137,
}


ARMS_LAYOUT_VERSION: Final = 4  # arms takes ``Stream.ARMS`` = 10 and ``InitStream.ARMS`` = 8 (DESIGN §42.4)


def layout_version(cfg: object) -> int:
    """The RNG layout version a run of ``cfg`` writes into its meta and checkpoints: ``RNG_LAYOUT_VERSION``
    (3)
    while no feature that takes a stream is on, so a feature-off run is byte-identical to main; 4 with
    ``arms.enabled`` (the arms stream, a reserved index, leaves every other stream's key unchanged, so version
    3 stays compatible)."""
    arms = getattr(cfg, "arms", None)
    return ARMS_LAYOUT_VERSION if arms is not None and arms.enabled else RNG_LAYOUT_VERSION


class Stream(IntEnum):
    """Per-step streams ``ks[Stream.X]``; indices 11–15 are reserved."""

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
    ARMS = 10  # arms (§42): the child-trait mutation and the immigrants' key and lock, in lifecycle.reproduce


class InitStream(IntEnum):
    """Initialisation streams from ``init_keys(seed)``; indices 9–15 are reserved."""

    POS = 0
    HEADING = 1
    W1 = 2
    W2 = 3
    TAG = 4
    PATCH_POS = 5
    PATCH_VEL = 6
    STATE_KEY = 7
    ARMS = 8  # arms (§42): the founders' keys and locks


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
