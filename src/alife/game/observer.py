"""The host observer: folds the step's observer channel into accumulators, reports the ``observers`` block of
the metrics row and persists the sidecar (docs/GAMES.md §4, §6.6; DESIGN §28.5, §28.6).

An ``Observer`` owns one ``ObsState`` (``game.observe``) and the host step counter. ``update(state, acts, t)``
queues the (lazy) update, and the evaluation of the labels every ``rules.every`` steps on the global step
clock; the caller evaluates ``arrays()`` in the same ``mx.eval`` as the state, so the run keeps exactly one
blocking eval per step. The observer never stores a ``State`` array (it passes ``obs_in(state)``, a few
fields, into its update and keeps only the result), never writes the state and never reads its RNG key.

The sidecar ``observers/step_<t>.npz`` holds one array per ``ObsState`` field plus ``meta`` (a JSON string).
``Observer.open`` loads it on resume only if every meta field, the array layout and the observed occupants
match, and otherwise starts fresh with a note (never an error).

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
from alife.game.observe import EVAL_BIT, ObsState
from alife.game.rules import ARCHETYPE_NAMES, INTENT_NAMES, INTENT_VERSION, RULES_V1, RuleSet
from alife.runio import CheckpointError, RunDir
from alife.state import State, to_numpy
from alife.step import Acts

__all__ = ["OBSERVER_FORMAT", "SIDECAR_FORMAT", "Labels", "Observer"]

OBSERVER_FORMAT: Final = 1  # the ObsState fields and their meaning; bump with any change to either
SIDECAR_FORMAT: Final = "alife-observers"  # the ``format`` of the sidecar's meta

# Meta fields that must equal the observer's own for a sidecar to load (``alife_version`` is not checked).
_META_KEYS: Final = (
    "format",
    "observer_format",
    "intent_version",
    "rules_digest",
    "step",
    "n_max",
    "dims",
    "config_hash",
)
_NP_DTYPE: Final = {
    mx.int32: np.dtype(np.int32),
    mx.int64: np.dtype(np.int64),
    mx.bool_: np.dtype(np.bool_),
}  # the dtypes ObsState uses, as they are stored in the sidecar


class Labels(NamedTuple):
    """The per-row labels of the last evaluation (device arrays; zeros mean unlabelled or not evaluated)."""

    arch_primary: mx.array  # (N,) int32 ``rules.Archetype`` (NONE = 0)
    arch_secondary: mx.array  # (N,) int32 ``rules.Archetype``
    arch_match: (
        mx.array
    )  # (N,) int32: bit a−1 set where archetype a's rule matched (Generalist = none matched)
    intent: mx.array  # (N,) int32 displayed ``rules.Intent`` (after the hysteresis)
    intent_match: mx.array  # (N,) int32: bit i−1 = predicate i (Resting = none) | ``EVAL_BIT``


def _shares(counts: np.ndarray, names: tuple[str, ...], denominator: int) -> dict[str, float] | None:
    """``{name: count / denominator}``, or None when nothing was counted."""
    if denominator == 0:
        return None
    return {name: int(c) / denominator for name, c in zip(names, counts, strict=True)}


class Observer:
    """Per-row game observers on one run: accumulators, labels, the metrics block and the sidecar."""

    def __init__(self, cfg: Config, o: ObsState, step: int, rules: RuleSet = RULES_V1) -> None:
        rules.check(cfg)  # a window that would overflow int32 for this world fails here, loudly
        self.cfg = cfg
        self.rules = rules
        self.o = o
        self.step = step

        def fold(o_: ObsState, oi: observe.ObsIn, a: Acts) -> ObsState:
            return observe.update(o_, oi, a, cfg, rules)

        def fold_eval(o_: ObsState, oi: observe.ObsIn, a: Acts) -> ObsState:
            return observe.evaluate(observe.update(o_, oi, a, cfg, rules), cfg, rules)

        # The compile mode follows the run's, so labels are reproducible under the same contract as the
        # trajectory (DESIGN §14.2).
        self._fold = mx.compile(fold) if cfg.run.compile else fold
        self._fold_eval = mx.compile(fold_eval) if cfg.run.compile else fold_eval

    # --- construction -----------------------------------------------------------------------------------

    @classmethod
    def fresh(cls, state: State, cfg: Config, *, rules: RuleSet = RULES_V1) -> Observer:
        """Observers starting at ``state``: nothing accumulated or labelled, existing bonds of unknown age."""
        o = observe.init_obs(observe.obs_in(state), cfg)
        mx.eval(*o)  # no lazy reference to the state's arrays survives (the loop donates them)
        return cls(cfg, o, int(to_numpy(state.step)), rules)

    @classmethod
    def open(
        cls, rd: RunDir, state: State, cfg: Config, *, rules: RuleSet = RULES_V1
    ) -> tuple[Observer, str | None]:
        """Observers for a run continuing from ``state``: the sidecar of its step if it is valid for it,
        else fresh. The note (None if the sidecar was loaded) is returned only on a resume, meaning ``rd``
        has a checkpoint at this step; a new run or a fork starts fresh without one. Never an error."""
        t = int(to_numpy(state.step))
        fresh = cls.fresh(state, cfg, rules=rules)
        try:
            arrays = rd.read_observers(t)
        except CheckpointError as e:
            loaded, reason = None, f"the sidecar at step {t} is unreadable ({e})"
        else:
            if arrays is None:
                loaded, reason = None, f"no sidecar at step {t}"
            else:
                loaded, reason = fresh._load(arrays)
        if loaded is not None:
            return cls(cfg, loaded, t, rules), None
        resumed = any(st == t for st, _ in rd.checkpoints())
        note = (
            f"observers: {reason}; labels warm up ({rules.arch_warm} steps for archetypes, "
            f"{rules.intent_warm} for intents)"
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
        }

    def sidecar(self) -> dict[str, np.ndarray]:
        """The arrays of ``observers/step_<t>.npz``: every ``ObsState`` field under its name, and ``meta``
        (a 0-d unicode array holding JSON). One host transfer."""
        mx.eval(*self.o)
        return {
            **{f: to_numpy(a) for f, a in zip(ObsState._fields, self.o, strict=True)},
            "meta": np.array(json.dumps(self._meta(), sort_keys=True)),
        }

    def _load(self, arrays: Mapping[str, np.ndarray]) -> tuple[ObsState | None, str]:
        """``(state, "")`` from a sidecar's arrays if they are valid for this (fresh) observer, else
        ``(None, reason)``. Valid: the meta fields equal this observer's (``rules_digest`` included, so
        accumulators scaled for, or hysteresis state left by, other rules are never read), the names, shapes
        and dtypes equal ``ObsState``'s, and the observed occupants (``uid``) are the state's."""
        names = {*ObsState._fields, "meta"}
        if set(arrays) != names:
            missing, extra = sorted(names - set(arrays)), sorted(set(arrays) - names)
            return None, f"the sidecar's arrays differ (missing {missing}, unexpected {extra})"
        try:
            meta = json.loads(str(arrays["meta"]))
        except ValueError:
            return None, "the sidecar's meta is not JSON"
        if not isinstance(meta, dict):
            return None, "the sidecar's meta is not a JSON object"
        want = self._meta()
        for key in _META_KEYS:
            if meta.get(key) != want[key]:
                return None, f"meta.{key} is {_short(meta.get(key))}, expected {_short(want[key])}"
        for f, ref in zip(ObsState._fields, self.o, strict=True):
            a = arrays[f]
            shape, dtype = tuple(ref.shape), _NP_DTYPE[ref.dtype]
            if a.shape != shape or a.dtype != dtype:
                return None, f"array {f!r} is {a.dtype} {a.shape}, expected {dtype} {shape}"
        if not np.array_equal(arrays["uid"], to_numpy(self.o.uid)):
            return None, "the sidecar's observed uids are not the state's"
        o = ObsState(**{f: mx.array(arrays[f]) for f in ObsState._fields})
        mx.eval(*o)
        return o, ""

    # --- the per-step fold --------------------------------------------------------------------------------

    def update(self, state: State, acts: Acts, t: int) -> None:
        """Queue the fold of the step that produced ``state`` (now at step ``t``) and, when ``t`` is a
        multiple of ``rules.every``, the evaluation. Lazy: evaluate ``arrays()`` in the step's ``mx.eval``."""
        if t != self.step + 1:
            raise ValueError(f"observer is at step {self.step}; cannot fold step {t}")
        fold = self._fold_eval if t % self.rules.every == 0 else self._fold
        self.o = fold(self.o, observe.obs_in(state), acts)
        self.step = t

    def arrays(self) -> tuple[mx.array, ...]:
        """The observer state's arrays, for the caller's single ``mx.eval``."""
        return tuple(self.o)

    # --- reporting ----------------------------------------------------------------------------------------

    def labels(self) -> Labels:
        """The per-row labels as of the last evaluation (device arrays)."""
        o = self.o
        return Labels(o.label_arch[:, 0], o.label_arch[:, 1], o.match_arch, o.label_intent, o.match_intent)

    def host_labels(self) -> dict[str, np.ndarray]:
        """``labels()`` as numpy arrays keyed by field name."""
        return {f: to_numpy(a) for f, a in zip(Labels._fields, self.labels(), strict=True)}

    def row_block(self) -> dict[str, Any]:
        """The ``observers`` block of a metrics row (DESIGN §28.5): counts and shares of the last evaluation.

        Comparisons and sums on the device and one host transfer. ``archetype`` and ``archetype_match`` are
        over the rows with an archetype label, ``intent`` over those with a displayed intent, and
        ``intent_match`` (the §4.5 coverage) over the rows evaluated; each is None while its denominator is 0.
        """
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
        }


def _short(v: Any) -> str:
    """A meta value for a note: long hashes are cut."""
    text = repr(v)
    return text if len(text) <= 24 else text[:21] + "..."
