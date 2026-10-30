"""Synthetic worlds with a planted code, as host snapshots (the harness of the voice analysis tests).

``world(rng, n, steps, mode)`` makes ``steps`` consecutive ``Snap`` objects of ``n`` persistent agents on a
2D torus. Each step an agent is *hurt* with probability 0.3 (``loss > 0``) and speaks; a hearer's response is
set by what its loudest audible neighbour said or was. ``mode``:

* ``planted``: the call is a deterministic function of the caller's context (channel 0 loud when hurt, else
  channel 1 soft; 5 % of calls are replaced by noise), and a hearer moves *toward* a source that called
  channel 0 and *away* from one that called channel 1: the code is used.
* ``random``: the call is random and independent of anything; the response is random.
* ``tic``: the call depends on the caller's context but nobody responds to it (information without use).
* ``environment``: the call is random, but the hearer's response depends on the caller's context directly
  (the shared-environment confound: the neighbour's situation reaches the hearer without any call).
"""

from __future__ import annotations

import numpy as np

from alife.analysis.voice import tables
from alife.analysis.voice.tables import Geo, Snap
from alife.geom import Box

SIZE = 60.0
RANGE = 8.0
K = 8
WIDTH = 4


def geo() -> Geo:
    return Geo(geo=Box(SIZE, 4.0, 2, 0.0), d3=False, hear_range=RANGE, atten_pow=1.0)


def _neighbours(pos: np.ndarray) -> np.ndarray:
    d = pos[None, :, :] - pos[:, None, :]
    d -= SIZE * np.round(d / SIZE)
    dist = np.linalg.norm(d, axis=2)
    np.fill_diagonal(dist, np.inf)
    dist = np.where(dist < RANGE, dist, np.inf)
    order = np.argsort(dist, axis=1)[:, :K]
    ok = np.take_along_axis(dist, order, axis=1) < np.inf
    return np.where(ok, order, -1).astype(np.int32)


def _heard(voice: np.ndarray, nbr: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(pos)
    heard = np.zeros((n, WIDTH))
    aud = np.zeros(n, np.uint8)
    for i in range(n):
        for j in nbr[i]:
            if j < 0:
                continue
            d = pos[j] - pos[i]
            d -= SIZE * np.round(d / SIZE)
            a = max(0.0, 1.0 - np.linalg.norm(d) / RANGE)
            heard[i] += a * voice[j]
            aud[i] += a > 0
    return np.minimum(heard, 255).astype(np.uint8), aud


def world(rng: np.random.Generator, n: int = 300, steps: int = 60, mode: str = "planted") -> list[Snap]:
    pos = rng.uniform(0, SIZE, (n, 2))
    uid = np.arange(n, dtype=np.int64)
    clade = rng.integers(0, 5, n).astype(np.int32)
    snaps: list[Snap] = []
    prev_voice = np.zeros((n, WIDTH), np.uint8)
    prev_hurt = np.zeros(n, bool)
    for t in range(steps):
        hurt = rng.random(n) < 0.3
        food = rng.integers(0, 6, n).astype(np.float32)
        energy = rng.uniform(50, 500, n).astype(np.float32)
        voice = np.zeros((n, WIDTH), np.uint8)
        if mode in ("planted", "tic"):
            voice[hurt, 0] = 200
            voice[~hurt, 1] = 90
            flip = rng.random(n) < 0.05
            voice[flip] = rng.integers(0, 256, (int(flip.sum()), WIDTH)).astype(np.uint8)
        else:
            voice = rng.integers(0, 256, (n, WIDTH)).astype(np.uint8)
        nbr = _neighbours(pos)
        heard, aud = _heard(prev_voice, nbr, pos)
        # the response of this step is a move toward / away from the loudest audible neighbour
        flags = np.zeros(n, np.uint16)
        step_vec = rng.normal(0, 0.05, (n, 2))
        j = np.where(nbr >= 0, nbr, 0)
        dd = pos[j] - pos[:, None, :]
        dd -= SIZE * np.round(dd / SIZE)
        wgt = np.clip(1.0 - np.linalg.norm(dd, axis=2) / RANGE, 0.0, 1.0)  # the engine's falloff
        loud = np.where(nbr >= 0, wgt * prev_voice[j].astype(float).sum(axis=2), 0.0)
        src = j[np.arange(n), loud.argmax(axis=1)]
        has = loud.max(axis=1) > 0
        to = pos[src] - pos
        to -= SIZE * np.round(to / SIZE)
        to /= np.maximum(np.linalg.norm(to, axis=1, keepdims=True), 1e-9)
        if mode == "planted":
            sign = np.where(prev_voice[src, 0] > 120, 1.0, -1.0)
            step_vec = np.where(has[:, None], sign[:, None] * to * 0.8, step_vec)
        elif mode == "environment":
            sign = np.where(prev_hurt[src], 1.0, -1.0)
            step_vec = np.where(has[:, None], sign[:, None] * to * 0.8, step_vec)
        snaps.append(
            Snap(
                step=t,
                alive=np.ones(n, bool),
                uid=uid,
                clade=clade,
                lineage=clade.astype(np.int8),
                energy=energy,
                loss=np.where(hurt, 5, 0).astype(np.int32),
                harvest=rng.integers(0, 2, n).astype(np.int32),
                pos=pos.copy(),
                bonds=rng.integers(0, 2, n).astype(np.int8),
                food=food,
                wind=np.zeros(n, np.float32),
                voice=voice,
                heard=heard,
                audible=aud,
                nbr=nbr,
                flags=flags,
                thrust=rng.integers(0, 256, n).astype(np.int16),
            )
        )
        pos = (pos + step_vec) % SIZE
        prev_voice, prev_hurt = voice, hurt
    return snaps


def sample_tables(
    rng: np.random.Generator, mode: str, n: int = 300, steps: int = 80, rows: int = 300
) -> tables.Tables:
    """The tables of a synthetic world: a sample at every step from 1 to steps - 2."""
    snaps = world(rng, n, steps, mode)
    g = geo()
    s_parts, h_parts = [], []
    for u in range(1, steps - 1):
        s, h = tables.build_sample(
            snaps[u - 1], snaps[u], snaps[u + 1], g, rows, np.random.default_rng([7, u])
        )
        s_parts.append(s)
        h_parts.append(h)
    meta = {"run": f"synthetic-{mode}", "window": [0, steps], "width": WIDTH, "control": "none"}
    return tables.Tables(tables.concat(s_parts), tables.concat(h_parts), meta)
