"""The host observer: folds the step's observer channel into accumulators, reports the ``observers`` block of
the metrics row, tracks the social layer and persists the sidecar (docs/GAMES.md §4, §6, §6.6; DESIGN §28.5,
§28.6, §29.9, §29.10).

An ``Observer`` owns one ``ObsState`` (labels, ``game.observe``), one ``SocState`` (houses, kin groups,
colonies, relation windows), a host ``Society`` and the host step counter. ``update(state, acts, t)`` queues
the (lazy) folds, the evaluation of the labels every ``rules.every`` steps and, every
``social.snapshot_every`` steps, the social snapshot; the caller evaluates ``arrays()`` in the same
``mx.eval`` as the state, so the run keeps exactly one blocking eval per step. The observer never stores a
``State`` array (it passes ``obs_in(state)`` and ``soc_in(state)``, a few fields, into its updates and keeps
only the results), never writes the state and never reads its RNG key.

A snapshot is *pending* from its step until it is *settled*: one host transfer, ``Society.settle`` (which
appends the snapshot's record to the records not yet taken) and ``relabel`` (the slots go back to the
device). The runner takes the records right after each step's eval, so it settles at the snapshot step; a
caller that never takes them (the labels tool, the step-level tests) settles at ``update(t + 1)`` or at its
next read (``row_block``, ``sidecar``, ``labels``, ...). Both paths give the same records and, from the
next update on, the same state.

The sidecar ``observers/step_<t>.npz`` holds one array per ``ObsState`` field, ``soc.<field>`` per
``SocState`` field, the society's ``host.*`` members and ``meta`` (a JSON string). ``Observer.open`` loads
it on resume only if every check passes (``_load``) and otherwise starts fresh with a note (never an error).
The snapshot records themselves are in the run's social log (``RunDir.append_social``), not in the sidecar.

Lint (rule 2, ``tests/test_lint.py``): no subscript assignment anywhere in this module, dict keys included.
The block and meta dicts are built as literals and comprehensions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Final, NamedTuple

import mlx.core as mx
import numpy as np

from alife.config import Config, config_hash, dims
from alife.game import observe
from alife.game import social as soc
from alife.game.colors import ColorInputs, house_slot_bytes, kin_slot_bytes
from alife.game.observe import EVAL_BIT, ObsState
from alife.game.rules import ARCHETYPE_NAMES, INTENT_NAMES, INTENT_VERSION, RULES_V2, RuleSet
from alife.game.social import HOUSE_UNKNOWN, SOCIAL_V1, SOCIAL_VERSION, SocialRules, SocState
from alife.game.society import HOST_KEYS, TOP_HOUSES, Society, host_snap, house_names
from alife.runio import CheckpointError, RunDir
from alife.state import State, to_numpy
from alife.step import Acts
from alife.tracking import TRACK_VERSION, DominanceMap, circular_mean

__all__ = [
    "DEFAULT_RULES",
    "DEFAULT_SOCIAL",
    "FLAG_BORN_SINCE",
    "OBSERVER_FORMAT",
    "SIDECAR_FORMAT",
    "IdentityRows",
    "Labels",
    "Observer",
    "SocialLabels",
    "house_bytes",
    "slot_bytes",
]

OBSERVER_FORMAT: Final = 3  # the sidecar's members and their meaning; bump with any change to either
SIDECAR_FORMAT: Final = "alife-observers"  # the ``format`` of the sidecar's meta

# The rules a run's observer uses, looked up at call time (``Observer.fresh`` / ``open`` with ``rules=None``
# or ``social=None``). No config key sets them (DESIGN §28.1, §29.9): the labels tool passes its own
# ``rules``, and tests that need snapshots in a short run replace ``DEFAULT_SOCIAL`` (``monkeypatch``) or
# pass ``social=``.
DEFAULT_RULES: RuleSet = RULES_V2
DEFAULT_SOCIAL: SocialRules = SOCIAL_V1

# Meta fields that must equal the observer's own for a sidecar to load, checked in this order (``format``
# first, then ``observer_format``, so that a sidecar of another layout gets that reason, not a list of
# missing members). ``alife_version`` is not checked.
_META_KEYS: Final = (
    "format",
    "observer_format",
    "intent_version",
    "rules_digest",
    "step",
    "n_max",
    "dims",
    "config_hash",
    "social_version",
    "social_digest",
    "track_version",
    "house_rate",
    "house_seed",
)
_NP_DTYPE: Final = {
    mx.int32: np.dtype(np.int32),
    mx.int64: np.dtype(np.int64),
    mx.bool_: np.dtype(np.bool_),
}  # the dtypes ObsState and SocState use, as they are stored in the sidecar
_SOC_PREFIX: Final = "soc."
# The slot bytes of the play stream's ``IDNT`` rows (DESIGN §33.1.3).
SLOT_UNKNOWN: Final = 253  # a house of an immigrant lineage of an observer attached mid-run
SLOT_OTHER: Final = 254  # listed in no table
SLOT_NONE: Final = 255
FLAG_BORN_SINCE: Final = 1  # IDNT row flag: born after the last snapshot


class Labels(NamedTuple):
    """The per-row labels of the last evaluation (device arrays; zeros mean unlabelled or not evaluated)."""

    arch_primary: mx.array  # (N,) int32 ``rules.Archetype`` (NONE = 0)
    arch_secondary: mx.array  # (N,) int32 ``rules.Archetype``
    arch_match: (
        mx.array
    )  # (N,) int32: bit a−1 set where archetype a's rule matched (Generalist = none matched)
    intent: mx.array  # (N,) int32 displayed ``rules.Intent`` (after the hysteresis)
    intent_match: mx.array  # (N,) int32: bit i−1 = predicate i (Resting = none) | ``EVAL_BIT``


class SocialLabels(NamedTuple):
    """The per-row social labels as of the last settled snapshot (device arrays)."""

    house: mx.array  # (N,) int64 house id (the founder's uid), ``social.HOUSE_UNKNOWN``, -1 dead
    kin_slot: mx.array  # (N,) int32 kin group slot: 0..F-1 named, F = other, -1 dead
    colony_slot: mx.array  # (N,) int32 colony slot: 0..F-1 named, F = other tracked colony, -1 none


class IdentityRows(NamedTuple):
    """The rows and tables of one ``IDNT`` record (DESIGN §33.1.3): the alive rows in ascending row order."""

    row: np.ndarray  # (n,) int64 row index in the state
    uid: np.ndarray  # (n,) int64
    clade: np.ndarray  # (n,) int32
    house: np.ndarray  # (n,) uint8 slot byte: 0..15 listed, 253 unknown lineage, 254 other
    kin: np.ndarray  # (n,) uint8 slot byte: 0..F-1 named, 254 other, 255 none
    colony: np.ndarray  # (n,) uint8 slot byte: 0..F-1 named, 254 other tracked colony, 255 none
    flags: np.ndarray  # (n,) uint8, bit 0 (``FLAG_BORN_SINCE``) = born after the last snapshot
    snapshot_step: int  # the step of the last settled snapshot (the tables' as-of step)
    tables: dict[str, Any]  # the IDNT JSON object


class _Loaded(NamedTuple):
    o: ObsState
    so: SocState
    society: Society


def _shares(counts: np.ndarray, names: tuple[str, ...], denominator: int) -> dict[str, float] | None:
    """``{name: count / denominator}``, or None when nothing was counted."""
    if denominator == 0:
        return None
    return {name: int(c) / denominator for name, c in zip(names, counts, strict=True)}


class Observer:
    """Per-row game observers on one run: accumulators, labels, the social layer, the metrics block, the
    snapshot records and the sidecar."""

    def __init__(
        self,
        cfg: Config,
        o: ObsState,
        so: SocState,
        society: Society,
        step: int,
        rules: RuleSet,
        social: SocialRules,
    ) -> None:
        rules.check(cfg)  # a window that would overflow int32 for this world fails here, loudly
        social.check(cfg)
        self.cfg = cfg
        self.rules = rules
        self.social = social
        self.o = o
        self.so = so
        self.society = society
        self.step = step
        self._snap: soc.Snap | None = None  # the pending snapshot (arrays, lazy until the caller's eval)
        self._snap_step = -1
        self._last_record: dict[str, Any] | None = None  # the newest record taken (identity tables, §33.1.3)

        def fold(
            o_: ObsState, so_: SocState, oi: observe.ObsIn, si: soc.SocIn, a: Acts
        ) -> tuple[ObsState, SocState]:
            return observe.update(o_, oi, a, cfg, rules), soc.update(so_, si, a, cfg, social)

        def fold_eval(
            o_: ObsState, so_: SocState, oi: observe.ObsIn, si: soc.SocIn, a: Acts
        ) -> tuple[ObsState, SocState]:
            o2, so2 = observe.update(o_, oi, a, cfg, rules), soc.update(so_, si, a, cfg, social)
            # the Settler clause reads the body, the Budding forecast the post-step state
            return observe.evaluate(o2, cfg, rules, body=so2.body, oi=oi), so2

        def snap(
            kin0: mx.array, so_: SocState, si: soc.SocIn, a: Acts | None, arch: mx.array | None = None
        ) -> soc.Snap:
            return soc.snapshot(kin0, so_, si, a, cfg, social, arch)

        def relabel(so_: SocState, sn: soc.Snap, kin_lut: mx.array, col_lut: mx.array) -> SocState:
            return soc.relabel(so_, sn, kin_lut, col_lut)

        # The compile mode follows the run's, so labels are reproducible under the same contract as the
        # trajectory (DESIGN §14.2).
        comp = mx.compile if cfg.run.compile else (lambda f: f)
        self._fold, self._fold_eval = comp(fold), comp(fold_eval)
        self._snap_fn, self._relabel = comp(snap), comp(relabel)

    # --- construction -----------------------------------------------------------------------------------

    @classmethod
    def _blank(cls, state: State, cfg: Config, rules: RuleSet, sr: SocialRules) -> Observer:
        """An observer at ``state`` with empty accumulators and no snapshot: what a sidecar is validated
        against (shapes, dtypes, occupants)."""
        o = observe.init_obs(observe.obs_in(state), cfg)
        so = soc.init_soc(soc.soc_in(state), cfg, sr)
        mx.eval(*o, *so)  # no lazy reference to the state's arrays survives (the loop donates them)
        return cls(cfg, o, so, Society(cfg, sr), int(to_numpy(state.step)), rules, sr)

    @classmethod
    def fresh(
        cls,
        state: State,
        cfg: Config,
        *,
        rules: RuleSet | None = None,
        social: SocialRules | None = None,
    ) -> Observer:
        """Observers starting at ``state``: nothing accumulated or labelled, existing bonds of unknown age,
        and the initial social snapshot (ids and slots for what is there; no events, relations or record).
        ``rules`` and ``social`` default to ``DEFAULT_RULES`` and ``DEFAULT_SOCIAL`` as of this call."""
        rules = DEFAULT_RULES if rules is None else rules
        sr = DEFAULT_SOCIAL if social is None else social
        obs = cls._blank(state, cfg, rules, sr)
        si = soc.soc_in(state)
        # no step to read: the temperament is zero; the labels are empty, so is the archetype mix
        snap = obs._snap_fn(obs.so.kin, obs.so, si, None, obs.o.label_arch[:, 0])
        h = host_snap(snap)  # the one host transfer (and the evaluation)
        kin_lut, col_lut = obs.society.initialise(h, obs.step)
        obs.so = obs._relabel(obs.so, snap, mx.array(kin_lut), mx.array(col_lut))
        mx.eval(*obs.so)
        return obs

    @classmethod
    def open(
        cls,
        rd: RunDir,
        state: State,
        cfg: Config,
        *,
        rules: RuleSet | None = None,
        social: SocialRules | None = None,
    ) -> tuple[Observer, str | None]:
        """Observers for a run continuing from ``state``: the sidecar of its step if it is valid for it,
        else fresh. The note (None if the sidecar was loaded) is returned only on a resume, meaning ``rd``
        has a checkpoint at this step; a new run or a fork starts fresh without one. Never an error.
        ``rules`` and ``social`` default to ``DEFAULT_RULES`` and ``DEFAULT_SOCIAL`` as of this call."""
        rules = DEFAULT_RULES if rules is None else rules
        sr = DEFAULT_SOCIAL if social is None else social
        t = int(to_numpy(state.step))
        blank = cls._blank(state, cfg, rules, sr)
        try:
            arrays = rd.read_observers(t)
        except CheckpointError as e:
            loaded, reason = None, f"the sidecar at step {t} is unreadable ({e})"
        else:
            if arrays is None:
                loaded, reason = None, f"no sidecar at step {t}"
            else:
                loaded, reason = blank._load(arrays)
        if loaded is not None:
            return cls(cfg, loaded.o, loaded.so, loaded.society, t, rules, sr), None
        fresh = cls.fresh(state, cfg, rules=rules, social=sr)
        resumed = any(st == t for st, _ in rd.checkpoints())
        note = (
            f"observers: {reason}; labels warm up ({rules.arch_warm} steps for archetypes, "
            f"{rules.intent_warm} for intents); social tracking restarts at step {t} "
            f"(houses seeded from clades; new ids, records with origin {t})"
        )
        return fresh, note if resumed else None

    # --- the sidecar --------------------------------------------------------------------------------------

    def _meta(self) -> dict[str, Any]:
        import alife  # local: the package __init__ imports the step graph

        d = dims(self.cfg)
        return {
            "format": SIDECAR_FORMAT,
            "observer_format": OBSERVER_FORMAT,
            "intent_version": INTENT_VERSION,
            "rules_digest": self.rules.digest(),
            "step": self.step,
            "n_max": d.n,
            "dims": d.nd,
            "config_hash": config_hash(self.cfg),
            "alife_version": alife.__version__,
            "social_version": SOCIAL_VERSION,
            "social_digest": self.social.digest(),
            "track_version": TRACK_VERSION,
            "house_rate": self.cfg.game.house_rate,
            "house_seed": self.cfg.run.seed,
        }

    def sidecar(self) -> dict[str, np.ndarray]:
        """The arrays of ``observers/step_<t>.npz``: every ``ObsState`` field under its name, every
        ``SocState`` field as ``soc.<field>``, the society's ``host.*`` members and ``meta`` (a 0-d unicode
        array holding JSON). One host transfer.

        Settles a pending snapshot first and raises ``RuntimeError`` while a record is untaken: the sidecar
        carries no reports, so a checkpoint must never stand for a snapshot whose record is neither in the
        social log nor anywhere else. The runner takes the records before every checkpoint.
        """
        self._settle()
        if self.society.records:
            raise RuntimeError(
                f"Observer.sidecar: {len(self.society.records)} snapshot record(s) not taken yet "
                "(call take_records() and append them to the run's social log first)"
            )
        mx.eval(*self.o, *self.so)
        return {
            **{f: to_numpy(a) for f, a in zip(ObsState._fields, self.o, strict=True)},
            **{f"{_SOC_PREFIX}{f}": to_numpy(a) for f, a in zip(SocState._fields, self.so, strict=True)},
            **self.society.arrays(),
            "meta": np.array(json.dumps(self._meta(), sort_keys=True)),
        }

    def _load(self, arrays: Mapping[str, np.ndarray]) -> tuple[_Loaded | None, str]:
        """``(loaded, "")`` from a sidecar's arrays if they are valid for this (blank) observer, else
        ``(None, reason)`` with the first failure as the reason. Checked in this order: the meta parses as a
        JSON object; its ``format`` and then ``observer_format``; its other fields equal this observer's
        (``rules_digest`` and ``social_digest`` included, so accumulators scaled for, or hysteresis state left
        by, other rules are never read); the member names; the shapes and dtypes of the ``ObsState`` and
        ``SocState`` members; the observed occupants (``uid`` and ``soc.uid``) equal the state's; and the
        society's host members (``Society.from_arrays``)."""
        if "meta" not in arrays:
            return None, "the sidecar has no meta"
        try:
            meta = json.loads(str(arrays["meta"]))
        except (ValueError, RecursionError):  # not JSON, or nested too deeply to parse
            return None, "the sidecar's meta is not JSON"
        if not isinstance(meta, dict):
            return None, "the sidecar's meta is not a JSON object"
        want = self._meta()
        for key in _META_KEYS:
            if meta.get(key) != want[key]:
                return None, f"meta.{key} is {_short(meta.get(key))}, expected {_short(want[key])}"
        expected = (
            *((f, f) for f in ObsState._fields),
            *((f"{_SOC_PREFIX}{f}", f) for f in SocState._fields),
        )
        names = {name for name, _ in expected} | {*HOST_KEYS, "meta"}
        if set(arrays) != names:
            missing, extra = sorted(names - set(arrays)), sorted(set(arrays) - names)
            return None, f"the sidecar's arrays differ (missing {missing}, unexpected {extra})"
        refs = (
            *((f, getattr(self.o, f)) for f in ObsState._fields),
            *((f"{_SOC_PREFIX}{f}", getattr(self.so, f)) for f in SocState._fields),
        )
        for name, ref in refs:
            a = arrays[name]
            shape, dtype = tuple(ref.shape), _NP_DTYPE[ref.dtype]
            if not isinstance(a, np.ndarray):
                return None, f"array {name!r} is not an array"
            if a.shape != shape or a.dtype != dtype:
                return None, f"array {name!r} is {a.dtype} {a.shape}, expected {dtype} {shape}"
        for name, ref in (("uid", self.o.uid), (f"{_SOC_PREFIX}uid", self.so.uid)):
            if not np.array_equal(arrays[name], to_numpy(ref)):
                return None, "the sidecar's observed uids are not the state's"
        society, reason = Society.from_arrays(arrays, self.cfg, self.social, step=self.step)
        if society is None:
            return None, f"the sidecar's host state is invalid ({reason})"
        o = ObsState(**{f: mx.array(arrays[f]) for f in ObsState._fields})
        so = SocState(**{f: mx.array(arrays[f"{_SOC_PREFIX}{f}"]) for f in SocState._fields})
        mx.eval(*o, *so)
        return _Loaded(o, so, society), ""

    # --- the per-step fold --------------------------------------------------------------------------------

    def update(self, state: State, acts: Acts, t: int) -> None:
        """Queue the fold of the step that produced ``state`` (now at step ``t``): the labels' accumulators
        and the social state; and, when ``t`` is a multiple of ``rules.every``, the evaluation, and when it
        is a multiple of ``social.snapshot_every``, the social snapshot (pending until it is settled). A
        snapshot still pending from the previous step is settled first. Lazy: evaluate ``arrays()`` in the
        step's ``mx.eval``."""
        if t != self.step + 1:
            raise ValueError(f"observer is at step {self.step}; cannot fold step {t}")
        self._settle()
        oi, si = observe.obs_in(state), soc.soc_in(state)
        fold = self._fold_eval if t % self.rules.every == 0 else self._fold
        kin0 = self.so.kin  # the start-of-step slots, which the snapshot's temperament is read with
        self.o, self.so = fold(self.o, self.so, oi, si, acts)
        if t % self.social.snapshot_every == 0:
            # the primary archetype labels as of the last evaluation (reset rows are already unlabelled)
            self._snap = self._snap_fn(kin0, self.so, si, acts, self.o.label_arch[:, 0])
            self._snap_step = t
        self.step = t

    def arrays(self) -> tuple[mx.array, ...]:
        """The observer's arrays (``ObsState``, ``SocState`` and a pending snapshot), for the caller's
        single ``mx.eval``."""
        return (*self.o, *self.so, *(self._snap if self._snap is not None else ()))

    def _settle(self) -> None:
        """Settle the pending snapshot, if any: one host transfer, ``Society.settle`` (which appends the
        snapshot's record to the untaken ones) and ``relabel`` (lazy: evaluated with the next step)."""
        if self._snap is None:
            return
        snap, t = self._snap, self._snap_step
        kin_lut, col_lut = self.society.settle(host_snap(snap), t)
        self.so = self._relabel(self.so, snap, mx.array(kin_lut), mx.array(col_lut))
        self._snap = None

    def take_records(self) -> list[dict[str, Any]]:
        """The records of the snapshots settled since the last call, in snapshot order, and forgets them.
        Settles a pending snapshot first. The runner appends them to the social log right after each step's
        eval (``RunDir.append_social``); a caller that never takes them keeps them in memory."""
        self._settle()
        records = self.society.take_records()
        if records:
            self._last_record = records[-1]
        return records

    def color_inputs(self, houses: np.ndarray) -> ColorInputs:
        """The device arrays the observer colour modes read (``game.colors``): the house slot byte of every
        row against ``houses`` (the listed top-16 house ids, in table order), the kin slot byte, the primary
        archetype and the displayed intent. Lazy: the caller evaluates them with the frame."""
        sl, lb = self.social_labels(), self.labels()
        return ColorInputs(
            house_slot_bytes(sl.house, houses),
            kin_slot_bytes(sl.kin_slot, self.social.slots),
            lb.arch_primary,
            lb.intent,
        )

    def identity_rows(self, state: State) -> IdentityRows:
        """The rows and tables of an ``IDNT`` record at ``state`` (the state the observer was last updated
        with; ``ValueError`` otherwise): per alive row its uid, clade and the house, kin group and colony slot
        bytes, and the JSON tables of the houses, kin groups and colonies (DESIGN §33.1.3).

        The tables are the columns of the last snapshot record taken (``take_records``), copied; before any
        record exists (a fresh observer, or one restored from a sidecar, until its first snapshot) they are
        derived from the rows and the society's names (sizes, plurality clade and centroids of the rows; the
        similarity, archetype mix, births and deaths are None). The slots of the rows are always the
        observer's current ones (``social_labels``): newborns inherit their parent's, and a row born after the
        last snapshot (``uid >= prev_next_uid``) is flagged. One host transfer; no device state changes.
        """
        self._settle()
        if int(to_numpy(state.step)) != self.step:
            raise ValueError(f"identity_rows: the observer is at step {self.step}, the state is not")
        sl = self.social_labels()
        d = dims(self.cfg)
        arrays = (self.so.uid, sl.house, sl.kin_slot, sl.colony_slot, state.clade, state.pos)
        mx.eval(*arrays)
        uid_all, house_all, kin_all, col_all, clade_all, pos_all = (to_numpy(a) for a in arrays)
        rows = np.flatnonzero(uid_all >= 0)
        uid, house, kin, col = uid_all[rows], house_all[rows], kin_all[rows], col_all[rows]
        clade, pos = clade_all[rows].astype(np.int32), pos_all[rows]
        f = self.social.slots
        rec = self._latest_record()
        if rec is None:
            top, houses = _houses_from_rows(house, clade, self.social.name_salt)
            kin_tab = _kin_from_rows(self.society, kin, clade, pos, self.cfg.world.size, d.d3)
            col_tab = _colonies_from_rows(self.society, kin, col, pos, self.cfg.world.size)
        else:
            top, houses = _houses_from_record(rec["houses"])
            kin_tab, col_tab = _kin_from_record(rec["kin_groups"]), _colonies_from_record(rec["colonies"])
        born = uid >= self.society.prev_next_uid
        tables = {
            "houses": houses,
            "exact": self.society.house_origin == 0,
            "origin": self.society.house_origin,
            "kin": kin_tab,
            "colonies": col_tab,
            "unknown_immigrant": int((house == HOUSE_UNKNOWN).sum()),
        }
        return IdentityRows(
            rows.astype(np.int64),
            uid.astype(np.int64),
            clade,
            _house_bytes(house, top),
            _slot_bytes(kin, f),
            _slot_bytes(col, f),
            np.where(born, FLAG_BORN_SINCE, 0).astype(np.uint8),
            self.society.snapshot_step,
            tables,
        )

    def _latest_record(self) -> dict[str, Any] | None:
        """The newest settled snapshot record (untaken, else the last one taken), None before any."""
        if self.society.records:
            return self.society.records[-1]
        return self._last_record

    # --- reporting ----------------------------------------------------------------------------------------

    def labels(self) -> Labels:
        """The per-row labels as of the last evaluation (device arrays)."""
        self._settle()
        o = self.o
        return Labels(o.label_arch[:, 0], o.label_arch[:, 1], o.match_arch, o.label_intent, o.match_intent)

    def host_labels(self) -> dict[str, np.ndarray]:
        """``labels()`` as numpy arrays keyed by field name."""
        return {f: to_numpy(a) for f, a in zip(Labels._fields, self.labels(), strict=True)}

    def social_labels(self) -> SocialLabels:
        """The per-row house, kin group slot and colony slot of the last settled snapshot (device arrays)."""
        self._settle()
        return SocialLabels(self.so.house, self.so.kin, self.so.colony)

    def territory(self) -> tuple[DominanceMap | None, np.ndarray]:
        """``(DominanceMap | None, kin slot -> id)`` of the last settled snapshot (None before the first)."""
        self._settle()
        return self.society.territory()

    def row_block(self) -> dict[str, Any]:
        """The ``observers`` block of a metrics row (DESIGN §28.5, §29.11): counts and shares of the last
        evaluation, then ``social``, the head of the last settled snapshot.

        Comparisons and sums on the device and one host transfer. ``archetype`` and ``archetype_match`` are
        over the rows with an archetype label, ``intent`` over those with a displayed intent, and
        ``intent_match`` (the §4.5 coverage) over the rows evaluated; each is None while its denominator is 0.
        A pending snapshot is settled first (the next ``update`` or read would do the same), so any caller
        may call this at any step.
        """
        self._settle()
        o, rs = self.o, self.rules
        primary = o.label_arch[:, 0]
        n_a, n_i = len(ARCHETYPE_NAMES), len(INTENT_NAMES)
        masks = [
            o.uid >= 0,
            primary != 0,
            (o.match_intent & EVAL_BIT) != 0,
            o.label_intent != 0,
            *[primary == a for a in range(1, n_a + 1)],
            *[(o.match_arch & (1 << (a - 1))) != 0 for a in range(1, n_a + 1)],
            *[o.label_intent == i for i in range(1, n_i + 1)],
            *[(o.match_intent & (1 << (i - 1))) != 0 for i in range(1, n_i + 1)],
        ]
        c = to_numpy(mx.stack([m.astype(mx.int32).sum() for m in masks]))
        alive, arch_n, eval_n, intent_n = (int(x) for x in c[:4])
        by_arch, match_arch = c[4 : 4 + n_a], c[4 + n_a : 4 + 2 * n_a]
        by_intent, match_intent = c[4 + 2 * n_a : 4 + 2 * n_a + n_i], c[4 + 2 * n_a + n_i :]
        return {
            "intent_version": INTENT_VERSION,
            "labels_step": self.step - self.step % rs.every,
            "alive": alive,
            "arch_labelled": arch_n,
            "intent_evaluated": eval_n,
            "intent_labelled": intent_n,
            "archetype": _shares(by_arch, ARCHETYPE_NAMES, arch_n),
            "archetype_match": _shares(match_arch, ARCHETYPE_NAMES, arch_n),
            "intent": _shares(by_intent, INTENT_NAMES, intent_n),
            "intent_match": _shares(match_intent, INTENT_NAMES, eval_n),
            "social": self.society.head(),
        }


# --- the identity tables (IDNT, DESIGN §33.1.3) -------------------------------------------------------------


def _slot_bytes(slot: np.ndarray, f: int) -> np.ndarray:
    """The slot byte of a ``SocState.kin`` / ``colony`` value: ``0..f-1`` as it is, ``f`` (OTHER) to 254, a
    negative value (no colony) to 255. The host twin of ``colors.kin_slot_bytes``."""
    return np.where(slot < 0, SLOT_NONE, np.where(slot >= f, SLOT_OTHER, slot)).astype(np.uint8)


def _house_bytes(house: np.ndarray, top: np.ndarray) -> np.ndarray:
    """The slot byte of house ids against ``top`` (the listed ids in table order): the index for a listed
    house, 253 for an unknown lineage, 254 for any other, 255 for a dead row. The host twin of
    ``colors.house_slot_bytes``."""
    out = np.where(house >= 0, SLOT_OTHER, SLOT_NONE).astype(np.int64)
    if top.size:
        order = np.argsort(top, kind="stable")
        at = np.minimum(np.searchsorted(top[order], house), top.size - 1)
        out = np.where(top[order][at] == house, order[at], out)
    return np.where(house == HOUSE_UNKNOWN, SLOT_UNKNOWN, out).astype(np.uint8)


slot_bytes = _slot_bytes  # public names for the play loop's FOCS record (a row's slots against the tables)
house_bytes = _house_bytes


def _houses_from_record(table: Mapping[str, list[Any]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    ids = table["id"]
    entries = [
        {
            "slot": i,
            "id": ids[i],
            "name": table["name"][i],
            "size": table["size"][i],
            "clade": table["clade"][i],
        }
        for i in range(len(ids))
    ]
    return np.array(ids, np.int64), entries


def _houses_from_rows(
    house: np.ndarray, clade: np.ndarray, salt: int
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """The top houses of the rows by ``(-size, id)``, with the clade of each house's first row: the rule of
    ``Society._houses`` applied to the rows now."""
    sel = house >= 0
    ids, first, counts = np.unique(house[sel], return_index=True, return_counts=True)
    top = np.lexsort((ids, -counts))[:TOP_HOUSES]
    names = house_names(ids[top], salt)
    entries = [
        {"slot": i, "id": int(ids[t]), "name": nm, "size": int(counts[t]), "clade": int(clade[sel][first][t])}
        for i, (t, nm) in enumerate(zip(top.tolist(), names, strict=True))
    ]
    return ids[top].astype(np.int64), entries


def _kin_from_record(table: Mapping[str, list[Any]]) -> list[dict[str, Any]]:
    three = "z_mean" in table
    return [
        {
            "slot": table["slot"][i],
            "id": table["id"][i],
            "name": table["name"][i],
            "named_after": table["named_after"][i],
            "size": table["size"][i],
            "clade": table["clade"][i],
            "clade_share": table["clade_share"][i],
            "sim_self": table["sim_self"][i],
            "x": table["x"][i],
            "y": table["y"][i],
            **({"z": table["z_mean"][i]} if three else {}),
            "arch": table["arch"][i],
            "births": table["births"][i],
            "deaths": table["deaths"][i],
        }
        for i in range(len(table["id"]))
    ]


def _colonies_from_record(table: Mapping[str, list[Any]]) -> list[dict[str, Any]]:
    return [
        {
            "slot": table["slot"][i],
            "id": table["id"][i],
            "size": table["size"][i],
            "kin": table["kin_group"][i],
            "x": table["x"][i],
            "y": table["y"][i],
        }
        for i in range(len(table["id"]))
    ]


def _centre(pos: np.ndarray, size: float) -> tuple[float, float]:
    return round(circular_mean(pos[:, 0], size), 2), round(circular_mean(pos[:, 1], size), 2)


def _kin_from_rows(
    society: Society, kin: np.ndarray, clade: np.ndarray, pos: np.ndarray, size: float, three: bool
) -> list[dict[str, Any]]:
    """The kin table without a record: one entry per slot that holds a group, from the rows of that slot."""
    out = []
    for s in np.flatnonzero(society.kin_slots >= 0).tolist():
        gid = int(society.kin_slots[s])
        name, after = society.kin_names.get(gid, (f"Kin group {gid}", -1))
        m = kin == s
        n = int(m.sum())
        clades, counts = np.unique(clade[m], return_counts=True)
        x, y = _centre(pos[m], size) if n else (None, None)
        out.append(
            {
                "slot": s,
                "id": gid,
                "name": name,
                "named_after": after if after >= 0 else None,
                "size": n,
                "clade": int(clades[np.argmax(counts)]) if n else -1,
                "clade_share": round(int(counts.max()) / n, 4) if n else None,
                "sim_self": None,
                "x": x,
                "y": y,
                **(
                    {"z": round(float(pos[m][:, 2].astype(np.float64).mean()), 2) if n else None}
                    if three
                    else {}
                ),
                "arch": None,
                "births": None,
                "deaths": None,
            }
        )
    return out


def _colonies_from_rows(
    society: Society, kin: np.ndarray, col: np.ndarray, pos: np.ndarray, size: float
) -> list[dict[str, Any]]:
    """The colony table without a record: one entry per slot that holds a colony, with the plurality kin group
    of its rows (ties to the smaller id; -1 if none)."""
    out = []
    for s in np.flatnonzero(society.col_slots >= 0).tolist():
        m = col == s
        n = int(m.sum())
        held = kin[m]
        ids, counts = np.unique(
            society.kin_slots[held[(held >= 0) & (held < society.kin_slots.size)]], return_counts=True
        )
        ids, counts = ids[ids >= 0], counts[ids >= 0]
        x, y = _centre(pos[m], size) if n else (None, None)
        out.append(
            {
                "slot": s,
                "id": int(society.col_slots[s]),
                "size": n,
                "kin": int(ids[np.argmax(counts)]) if ids.size else -1,
                "x": x,
                "y": y,
            }
        )
    return out


def _short(v: Any) -> str:
    """A meta value for a note: long hashes are cut."""
    text = repr(v)
    return text if len(text) <= 24 else text[:21] + "..."
