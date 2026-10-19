"""Synthetic observer inputs for the game tests (§28): hand-built ``ObsIn`` and ``Acts`` over a config's
sizes, and a driver that runs the observer's ``update`` and ``evaluate`` on the step clock without a
simulation."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from alife import layout
from alife.config import Config, dims
from alife.game import observer, social
from alife.game.observe import ObsIn, ObsState, evaluate, init_obs, update
from alife.game.rules import RULES_V1, RuleSet
from alife.game.social import SOCIAL_V1, Expo, Rel, SocialRules, SocIn
from alife.game.society import Society, host_snap
from alife.state import State, init_state, to_numpy
from alife.step import ActFlag, Acts


def social_every(monkeypatch: pytest.MonkeyPatch, every: int) -> SocialRules:
    """Make the observers of this test snapshot every ``every`` steps: replaces ``observer.DEFAULT_SOCIAL``
    (the module default that ``Observer.fresh``/``open``, hence the runner, and the bench read at call time;
    DESIGN §29.9) by ``SOCIAL_V1`` with that cadence. A module-scoped fixture, which cannot use
    ``monkeypatch``, passes ``social=`` explicitly instead."""
    rules = dataclasses.replace(SOCIAL_V1, snapshot_every=every)
    monkeypatch.setattr(observer, "DEFAULT_SOCIAL", rules)
    return rules


def blank_oi(cfg: Config) -> dict[str, np.ndarray]:
    """Numpy arrays of an all-dead ``ObsIn`` (``make_oi`` turns them into one)."""
    d = dims(cfg)
    return {
        "alive": np.zeros(d.n, bool),
        "uid": np.full(d.n, -1, np.int64),
        "parent_uid": np.full(d.n, -1, np.int64),
        "partner": np.full((d.n, layout.B), -1, np.int32),
        "vel": np.zeros((d.n, d.nd), np.float32),
        "last_harvest": np.zeros(d.n, np.int32),
        "last_inflow": np.zeros(d.n, np.int32),
        "last_loss": np.zeros(d.n, np.int32),
    }


def make_oi(arrays: Mapping[str, Any]) -> ObsIn:
    return ObsIn(**{f: mx.array(np.asarray(arrays[f])) for f in ObsIn._fields})


def blank_social_acts(n: int, k: int, m: int) -> dict[str, np.ndarray]:
    """The social-tracking additions of ``Acts`` (§29.2) for ``n`` rows, ``k`` neighbours and ``m`` lanes: no
    target, no neighbour, no kin query, no birth lane, no death."""
    return {
        "attack_target": np.full(n, -1, np.int32),
        "bond_target": np.full(n, -1, np.int32),
        "q_kin": np.zeros((n, 2), np.int16),
        "nbr": np.full((n, k), -1, np.int32),
        "born_row": np.full(m, -1, np.int32),
        "born_parent": np.full(m, -1, np.int32),
        "born_uid": np.full(m, -1, np.int64),
        "death_cause": np.full(n, -1, np.int8),
    }


def blank_acts(cfg: Config) -> dict[str, np.ndarray]:
    d = dims(cfg)
    return {
        "attack_gain": np.zeros(d.n, np.int32),
        "share_out": np.zeros((d.n, layout.B), np.int32),
        "share_got": np.zeros((d.n, layout.B), np.int32),
        "flags": np.zeros(d.n, np.uint8),
        "thrust": np.zeros(d.n, np.int16),
        "defend": np.zeros(d.n, np.int16),
        "emit": np.zeros(d.n, np.int16),
        **blank_social_acts(d.n, d.k, d.m),
    }


def make_acts(arrays: Mapping[str, Any]) -> Acts:
    return Acts(**{f: mx.array(np.asarray(arrays[f])) for f in Acts._fields})


def living(cfg: Config, rows: int, **fields: Any) -> dict[str, np.ndarray]:
    """``blank_oi`` with rows ``0..rows-1`` alive (uid = row) and ``fields`` (name → array or scalar,
    broadcast over the alive rows) set."""
    oi = blank_oi(cfg)
    oi["alive"][:rows] = True
    oi["uid"][:rows] = np.arange(rows)
    for name, value in fields.items():
        oi[name][:rows] = value
    return oi


def np_state(o: ObsState) -> dict[str, np.ndarray]:
    return {f: to_numpy(getattr(o, f)) for f in ObsState._fields}


class Driver:
    """One observer on a synthetic constant (or per-step) input, evaluated on the global step clock."""

    def __init__(self, cfg: Config, oi: ObsIn, *, start: int = 0, rs: RuleSet = RULES_V1) -> None:
        self.cfg, self.rs, self.t = cfg, rs, start
        self.o = init_obs(oi, cfg)
        mx.eval(*self.o)

    def step(self, oi: ObsIn, acts: Acts) -> ObsState:
        """Update with one step's inputs; evaluate if the new step count is a multiple of ``rs.every``."""
        self.t += 1
        self.o = update(self.o, oi, acts, self.cfg, self.rs)
        if self.t % self.rs.every == 0:
            self.o = evaluate(self.o, self.cfg, self.rs)
        mx.eval(*self.o)
        return self.o


# --- the social layer (§29): hand-built SocIn, and a driver that feeds a Society as the observer does -------


def blank_si(cfg: Config) -> dict[str, np.ndarray]:
    """Numpy arrays of an all-dead ``SocIn`` (``make_si`` turns them into one)."""
    d = dims(cfg)
    return {
        "alive": np.zeros(d.n, bool),
        "uid": np.full(d.n, -1, np.int64),
        "parent_uid": np.full(d.n, -1, np.int64),
        "clade": np.zeros(d.n, np.int32),
        "partner": np.full((d.n, layout.B), -1, np.int32),
        "tag": np.zeros((d.n, layout.T), np.float32),
        "pos": np.zeros((d.n, d.nd), np.float32),
        "next_uid": np.array(0, np.int64),
    }


def make_si(arrays: Mapping[str, Any]) -> SocIn:
    return SocIn(**{f: mx.array(np.asarray(arrays[f])) for f in SocIn._fields})


def living_si(cfg: Config, rows: int, **fields: Any) -> dict[str, np.ndarray]:
    """``blank_si`` with rows ``0..rows-1`` alive (uid = row, clade = row, ``next_uid`` = rows) and ``fields``
    (name -> array or scalar, broadcast over the alive rows) set."""
    si = blank_si(cfg)
    si["alive"][:rows] = True
    si["uid"][:rows] = np.arange(rows)
    si["clade"][:rows] = np.arange(rows)
    si["next_uid"] = np.array(rows, np.int64)
    for name, value in fields.items():
        si[name][:rows] = value
    return si


def np_soc(so: social.SocState) -> dict[str, np.ndarray]:
    return {f: to_numpy(getattr(so, f)) for f in social.SocState._fields}


class SocDriver:
    """A ``Society`` fed with snapshots of hand-built inputs exactly as the observer will: ``update`` (blank
    acts unless given), ``snapshot``, the host copy, ``Society.settle`` and ``relabel``. ``__init__`` is the
    initial snapshot of a fresh observer; ``advance`` is one later snapshot. A row that appears between two
    snapshots without a birth lane stands for an immigrant (slot OTHER, its own house)."""

    def __init__(
        self,
        cfg: Config,
        si: Mapping[str, Any],
        *,
        sr: SocialRules = SOCIAL_V1,
        step: int = 0,
        society: Society | None = None,
    ) -> None:
        self.cfg, self.sr = cfg, sr
        s = si if isinstance(si, SocIn) else make_si(si)
        self.so = social.init_soc(s, cfg, sr)
        self.society = society or Society(cfg, sr)
        self.step = step
        snap = social.snapshot(self.so.kin, self.so, s, None, cfg, sr)
        self.h = host_snap(snap)
        kin_lut, col_lut = self.society.initialise(self.h, step)
        self.so = social.relabel(self.so, snap, mx.array(kin_lut), mx.array(col_lut))
        mx.eval(*self.so)
        self.kin_lut, self.col_lut = kin_lut, col_lut

    def tick(self, si: Mapping[str, Any], acts: Acts | None = None) -> tuple[SocIn, Acts, mx.array]:
        """One step without a snapshot: ``update`` with ``acts`` (blank by default). Returns the inputs and
        the slots before the step (what ``snapshot`` reads)."""
        s = si if isinstance(si, SocIn) else make_si(si)
        a = acts if acts is not None else make_acts(blank_acts(self.cfg))
        kin0 = self.so.kin
        self.so = social.update(self.so, s, a, self.cfg, self.sr)
        # a row that appears without a birth lane stands for an immigrant: OTHER, and its own house
        new = s.alive & (self.so.kin < 0)
        self.so = self.so._replace(
            kin=mx.where(new, self.sr.slots, self.so.kin), house=mx.where(new, s.uid, self.so.house)
        )
        return s, a, kin0

    def advance(
        self, si: Mapping[str, Any], step: int, acts: Acts | None = None, arch: np.ndarray | None = None
    ) -> dict[str, Any]:
        """One snapshot at ``step``: the update with ``acts``, the snapshot (with ``arch``, the rows' primary
        archetype labels, if given), the settle and the relabel. Returns the snapshot's record."""
        s, a, kin0 = self.tick(si, acts)
        snap = social.snapshot(
            kin0, self.so, s, a, self.cfg, self.sr, None if arch is None else mx.array(arch.astype(np.int32))
        )
        self.h = host_snap(snap)
        before = len(self.society.records)
        self.kin_lut, self.col_lut = self.society.settle(self.h, step)
        self.so = social.relabel(self.so, snap, mx.array(self.kin_lut), mx.array(self.col_lut))
        mx.eval(*self.so)
        self.step = step
        assert len(self.society.records) == before + 1
        return self.society.records[-1]

    def kin_slot_rows(self) -> np.ndarray:
        """The device ``kin`` slots, as numpy."""
        return to_numpy(self.so.kin)


def window_reference(
    prev_kin: np.ndarray, prev_sg: np.ndarray, a: dict[str, np.ndarray], g: int, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One step's increments to (rel, expo, shared) from the channel and the start-of-step slots, written as
    plain loops over the rows (an independent formulation of ``social.update``)."""
    rel = np.zeros((5, g, g), np.int64)
    expo = np.zeros((2, g, g, k), np.int64)
    shared = np.zeros((g, g), np.int64)
    flags = a["flags"].astype(np.int64)
    for i in range(len(prev_kin)):
        if prev_kin[i] < 0:
            assert not flags[i] and not a["share_out"][i].any()
            continue
        nb = a["nbr"][i][a["nbr"][i] >= 0]
        if flags[i] & ActFlag.ATTACKED.value:
            t = a["attack_target"][i]
            rel[Rel.ATTACKS.value, prev_kin[i], prev_kin[t]] += 1
            if flags[t] & ActFlag.KILLED.value:
                rel[Rel.KILLS.value, prev_kin[i], prev_kin[t]] += 1
            for j in nb:
                expo[Expo.ATTACK.value, prev_kin[i], prev_kin[j], len(nb) - 1] += 1
        if flags[i] & ActFlag.HIT.value:
            rel[Rel.HITS.value, prev_kin[i], prev_kin[a["attack_target"][i]]] += 1
        if flags[i] & ActFlag.PROPOSED.value:
            rel[Rel.BONDS.value, prev_kin[i], prev_kin[a["bond_target"][i]]] += 1
            for j in nb:
                expo[Expo.BOND.value, prev_kin[i], prev_kin[j], len(nb) - 1] += 1
        for slot in range(layout.B):
            if prev_sg[i, slot] >= 0:
                rel[Rel.BOND_SLOTS.value, prev_kin[i], prev_sg[i, slot]] += 1
                shared[prev_kin[i], prev_sg[i, slot]] += a["share_out"][i, slot]
    return rel, expo, shared


def life_reference(
    prev_kin: np.ndarray, a: dict[str, np.ndarray], g: int, causes: int
) -> tuple[np.ndarray, np.ndarray]:
    """One step's increments to the window births ``(g,)`` and deaths ``(causes, g)`` from the channel and
    the start-of-step slots, as plain loops (an independent formulation of ``social.update``): a birth lane
    counts for its parent's slot, a row with a death cause for its own."""
    births = np.zeros(g, np.int64)
    deaths = np.zeros((causes, g), np.int64)
    for lane in range(len(a["born_row"])):
        parent = a["born_parent"][lane]
        if a["born_row"][lane] >= 0 and parent >= 0 and prev_kin[parent] >= 0:
            births[prev_kin[parent]] += 1
    for row in range(len(prev_kin)):
        if a["death_cause"][row] >= 0 and prev_kin[row] >= 0:  # a row dead at the start cannot die
            deaths[a["death_cause"][row], prev_kin[row]] += 1
    return births, deaths


def planted_state(cfg: Config) -> State:
    """``init_state`` with the founders' tags planted in three tight clouds, so that kin groups are tracked
    (the scattered tags of an unseeded tiny world track none)."""
    s = init_state(cfg)
    n0, n = cfg.population.n_init, dims(cfg).n
    tag = to_numpy(s.tag).copy()
    centres = np.array([[-0.6, -0.6, -0.6, -0.6], [0.5, 0.5, -0.5, 0.5], [0.0, 0.6, 0.6, -0.6]])
    rng = np.random.default_rng(3)
    tag[:n0] = centres[rng.integers(0, 3, n0)] + rng.normal(0, 0.02, (n0, tag.shape[1]))
    assert tag.shape[0] == n
    return s._replace(tag=mx.array(tag.astype(np.float32)))
