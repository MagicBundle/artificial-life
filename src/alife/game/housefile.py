"""The exact houses of a planet: ``observers/houses_<S>.npz`` (DESIGN §33.3.5).

A house is a pure function of the birth chain (GAMES §6.2): a birth founds one iff ``splitmix64(uid, seed) <
rate * 2^64`` (``social.update``), an immigrant (a birth without a parent) and a founder of the run always do,
every other birth inherits its parent's. An observer attached to a world already running cannot see that
chain, so a planet (a fork of an evolved checkpoint at step ``S``) gets its houses from the offline rebuild of
``analysis/houses.py`` and keeps them in this file, next to the sidecars. The file holds the house and the
house of origin (``house_from``) of the agents alive at ``S``, by uid, and the fork's ``house_rate`` and
``run.seed`` (the rule it was built with).

``Observer.open`` reads it when it has no valid sidecar: at ``T == S`` the seed is the file itself; at ``T >
S`` it is the file *folded* over the planet's own birth records of ``[S, T)`` with the same rule, so a
sidecar lost at a later checkpoint gives back the same houses. The file is kept for the life of the directory
(it does not match the sidecar pattern, so ``RunDir`` neither prunes nor truncates it).

Numpy only (no MLX): ``game/`` does not import ``analysis/``, so the hash is a copy of
``analysis.phylo.splitmix64`` (``tests/test_play_houses.py`` compares them, and ``groups.hash_below``). No
subscript assignment (``tests/test_lint.py``).
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Final, NamedTuple

import numpy as np

from alife.runio import RunDir, _write_npz

__all__ = [
    "FORMAT",
    "HouseFileError",
    "HouseSeed",
    "files",
    "fold",
    "fold_chunks",
    "hash_below",
    "path",
    "read",
    "splitmix64",
    "write",
]

FORMAT: Final = "alife-houses-1"
_FILE_RE: Final = re.compile(r"^houses_(\d{10})\.npz$")
_MASK64: Final = 0xFFFFFFFFFFFFFFFF
_I64: Final = np.dtype(np.int64)


class HouseFileError(Exception):
    """A house file that cannot be used for this observer (absent chunks, another rule, malformed)."""


class HouseSeed(NamedTuple):
    """The houses of a set of agents: three int64 arrays of one length, ascending by uid."""

    uid: np.ndarray
    house: np.ndarray  # the uid of the founder of the agent's house
    house_from: np.ndarray  # the house of the founder's parent, -1 for a founder of the run or an immigrant


def splitmix64(x: np.ndarray, seed: int) -> np.ndarray:
    """``analysis.phylo.splitmix64``: the SplitMix64 finaliser of ``x ^ seed``, uint64 with wrapping."""
    z = np.asarray(x, dtype=np.uint64) ^ np.uint64(seed & _MASK64)
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def hash_below(x: np.ndarray, seed: int, rate: float) -> np.ndarray:
    """``splitmix64(x, seed) < rate * 2^64`` (bool); all True from ``rate >= 1`` and all False at 0, as
    ``groups.hash_below`` and ``phylo.marker_roots`` have it."""
    if not rate >= 0.0:
        raise ValueError(f"hash rate {rate} must be >= 0")
    if rate >= 1.0:
        return np.ones(np.shape(x), dtype=bool)
    if rate == 0.0:
        return np.zeros(np.shape(x), dtype=bool)
    cut = np.uint64(min(int(rate * 2.0**64), 2**64 - 1))
    return splitmix64(x, seed) < cut


def _check(seed: HouseSeed) -> HouseSeed:
    cols = (seed.uid, seed.house, seed.house_from)
    if any(c.dtype != _I64 or c.ndim != 1 for c in cols) or len({c.size for c in cols}) != 1:
        raise HouseFileError("the house arrays are not one-dimensional int64 arrays of one length")
    if seed.uid.size and (np.any(np.diff(seed.uid) <= 0) or seed.uid[0] < 0):
        raise HouseFileError("the uids are not strictly increasing and non-negative")
    return seed


# --- the file -----------------------------------------------------------------------------------------------


def path(rd: RunDir, step: int) -> Path:
    return rd.observers_dir / f"houses_{step:010d}.npz"


def files(rd: RunDir) -> list[tuple[int, Path]]:
    """``(step, path)`` of every house file of the run directory, ascending."""
    d = rd.observers_dir
    if not d.is_dir():
        return []
    return sorted((int(m.group(1)), p) for p in d.iterdir() if (m := _FILE_RE.match(p.name)) and p.is_file())


def write(rd: RunDir, step: int, seed: HouseSeed, *, rate: float, run_seed: int) -> Path:
    """Atomically write ``observers/houses_<step>.npz``: the houses of the agents alive at ``step``, built
    with ``house_rate`` ``rate`` and ``run.seed`` ``run_seed``."""
    _check(seed)
    rd.observers_dir.mkdir(exist_ok=True)
    p = path(rd, step)
    _write_npz(
        p,
        {
            "uid": seed.uid,
            "house": seed.house,
            "house_from": seed.house_from,
            "meta": np.array(
                json.dumps(
                    {"format": FORMAT, "step": int(step), "rate": float(rate), "seed": int(run_seed)},
                    sort_keys=True,
                )
            ),
        },
    )
    return p


def read(rd: RunDir, step: int, *, rate: float, run_seed: int) -> HouseSeed:
    """The seed of ``observers/houses_<step>.npz``. ``HouseFileError`` if the file is unreadable, malformed
    or was built with another ``house_rate`` or ``run.seed`` than the ones given (houses are a function of
    both)."""
    p = path(rd, step)
    try:
        with np.load(p) as z:
            meta = json.loads(str(z["meta"]))
            seed = HouseSeed(z["uid"], z["house"], z["house_from"])
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as e:
        raise HouseFileError(f"{p.name} is unreadable ({e})") from e
    if not isinstance(meta, dict) or meta.get("format") != FORMAT or meta.get("step") != step:
        raise HouseFileError(f"{p.name} is not a version-1 house file of step {step}")
    if meta.get("rate") != float(rate) or meta.get("seed") != int(run_seed):
        raise HouseFileError(
            f"{p.name} was built with house_rate {meta.get('rate')} and seed {meta.get('seed')}, "
            f"not {float(rate)} and {int(run_seed)}"
        )
    return _check(seed)


# --- the fold -----------------------------------------------------------------------------------------------


def fold(
    seed: HouseSeed, uid: np.ndarray, parent_uid: np.ndarray, *, run_seed: int, rate: float
) -> HouseSeed:
    """``seed`` extended by births, in log order: ``uid[i]`` born of ``parent_uid[i]`` (``-1``: an immigrant).

    The rule of ``social.update``: a birth founds a house (its own uid, its parent's house as origin) iff it
    is an immigrant or ``hash_below(uid, run_seed, rate)``; otherwise it takes its parent's house and
    origin. The result holds the seed's agents and every birth (including those that have died since: a
    superset of the agents alive at the end), ascending by uid. Births must have uids above the seed's and
    ascending, and every parent must be a seed agent or an earlier birth (``HouseFileError`` otherwise).
    """
    uid = np.asarray(uid, dtype=_I64)
    parent_uid = np.asarray(parent_uid, dtype=_I64)
    if uid.size != parent_uid.size:
        raise HouseFileError("births: uid and parent_uid differ in length")
    if uid.size == 0:
        return seed
    if np.any(np.diff(uid) <= 0) or (seed.uid.size and uid[0] <= seed.uid[-1]):
        raise HouseFileError("births: uids are not ascending above the seed's")
    n0, nb = seed.uid.size, uid.size
    all_uid = np.concatenate([seed.uid, uid])
    # the table index of each parent (births point into the seed or at earlier births)
    pos = np.searchsorted(all_uid, parent_uid)
    inside = pos < all_uid.size
    found = inside & (all_uid[np.minimum(pos, all_uid.size - 1)] == parent_uid)
    immigrant = parent_uid < 0
    if np.any(~immigrant & ~found):
        raise HouseFileError(f"births: {int((~immigrant & ~found).sum())} parents are in no table")
    par = np.where(immigrant, -1, pos)
    if np.any(par >= n0 + np.arange(nb)):
        raise HouseFileError("births: a parent is not older than its child")
    founded = immigrant | hash_below(uid, run_seed, rate)
    # pointer jumping: a seed agent and a founder are their own root, any other birth points at its parent
    ptr = np.concatenate([np.arange(n0), np.where(founded, n0 + np.arange(nb), par)]).astype(np.int64)
    while True:
        nxt = ptr[ptr]
        if np.array_equal(nxt, ptr):
            break
        ptr = nxt
    # the house and origin of a root: a seed agent's own; a founder's uid, and its parent's resolved house
    root_house = np.concatenate([seed.house, uid])
    parent_house = np.where(immigrant, -1, root_house[ptr[np.maximum(par, 0)]])
    root_from = np.concatenate([seed.house_from, np.where(immigrant, -1, parent_house)])
    return HouseSeed(all_uid, root_house[ptr], root_from[ptr])


def births_between(rd: RunDir, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """``(uid, parent_uid)`` of the birth records of the run directory with ``start <= step < end``, in log
    order. ``HouseFileError`` unless the chunks tile ``[start, end)`` exactly from ``start`` (a chunk that
    straddles ``end`` is trimmed)."""
    uids, parents = [], []
    want = start
    for s0, s1, p in rd.event_files("births"):
        if s1 <= start or s0 >= end:
            continue
        if s0 != want:
            raise HouseFileError(f"the birth chunks do not tile [{start}, {end}): nothing covers {want}")
        try:
            with np.load(p) as z:
                step, u, par = z["step"], z["uid"], z["parent_uid"]
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as e:
            raise HouseFileError(f"{p.name} is unreadable ({e})") from e
        keep = step < end
        uids.append(u[keep].astype(_I64))
        parents.append(par[keep].astype(_I64))
        want = s1
    if want < end:
        raise HouseFileError(f"the birth chunks do not tile [{start}, {end}): nothing covers {want}")
    if not uids:
        return np.zeros(0, _I64), np.zeros(0, _I64)
    return np.concatenate(uids), np.concatenate(parents)


def fold_chunks(rd: RunDir, seed: HouseSeed, s: int, t: int, *, run_seed: int, rate: float) -> HouseSeed:
    """``seed`` (the houses at step ``s``) folded over the run directory's own births of ``[s, t)``."""
    if t < s:
        raise HouseFileError(f"cannot fold the houses of step {s} back to step {t}")
    if t == s:
        return seed
    uid, parent = births_between(rd, s, t)
    return fold(seed, uid, parent, run_seed=run_seed, rate=rate)
