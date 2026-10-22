"""Do the game labels predict behaviour? The validity test of docs/GAMES.md §4.5 (DESIGN §28.8).

A labelled row is a claim about what the agent does: a Gatherer harvests near the maximum rate, a Hunting
agent attacks.
This tool checks the claims. It re-simulates a run from a checkpoint with the observers on (``game.observe``
need not be set in the run), and at *cohort* steps ``T`` (multiples of ``rules.every``, so the labels are
fresh) it snapshots the labelled rows. It then follows each cohort row for the next ``H`` steps, while it
stays the same occupant of its row, and asks whether the label at ``T`` ranks the row's later behaviour
above that of the unlabelled rows: the Mann–Whitney AUC against a matched random label.

* **Archetypes** (horizon ``--arch-horizon``, default 1,000 steps): for the rule's match bit, the outcome
  is the harvest rate (Gatherer), the attack rate (Hunter), the share-out rate (Provider), the
  ``min(out, got)`` rate (Trader), the radius of gyration squared of the unwrapped track (Nomad) or its
  negative (Settler). Generalist is reported as prevalence only.

  **The storm world** (DESIGN §32). Where the world has weather, the tracker also integrates the track
  with the wind drift subtracted, ``u_k = u_{k-1} + min_image(Δpos) − drift_k`` (``drift_k`` is
  ``weather.agent_drift`` of the post-step state of step ``k``), and the primary score of Nomad and Settler
  under a ``wind_rel`` rule set is the radius of gyration of that track (``rg2_rel``); the raw ``rg2`` score
  is reported beside it (``auc_raw``, and ``auc_rel`` for the other way round), so that a run under
  ``--rules v2`` on the same window gives the comparison. The three version-3 archetypes have outcomes of
  their own, scored over the first ``--arch-min-obs`` observed steps with permutation strata extended by a
  z band (``Z_BAND_EDGES``): Drifter, the wind share of the horizontal displacement
  ``|Σ drift| / (|Σ drift| + |Σ own|)``; Floater, within the pool aloft at the cohort step
  (``z >= ALOFT·H``, 8 of 80) and not pinned at the lid (``z < z_hi − LID_MARGIN``), the altitude held,
  ``−|z_k − z_T|``; Sinker, over the rows with room to fall, the descent ``z_T − z_k`` normalised by
  ``min(z_T, arch_min_obs·gravity)``. Each of the three also reports its AUC per z band (``by_band``): a
  pooled AUC can come from one band, and the Floater's came from the lid before the pinned rows were left out.
  ``--dump DIR`` writes the
  calibration table: the pooled per-row accumulator means at the cohort step, the 64-step input history and
  every outcome, so that candidate thresholds are scored in seconds with this module's own scoring.
* **Intents** (horizon ``--intent-horizon``, default 10 steps): for the displayed label (``auc_label``) and
  for the predicate bit (``auc_rule``), the number of buds (Budding), attacks (Hunting), ``|thrust|``
  (Fleeing, Roaming and, negated, Resting), defence (Defending), energy sent to kin (Caring), bonds formed
  (Bonding), emission (Signalling) or harvest (Feeding) over the horizon.

**Censoring.** Most rows die before a 1,000-step horizon, and what kills a row depends on what is being
labelled. The archetype score is therefore exposure-normalised: every pool row observed at least
``--arch-min-obs`` steps (default 100) is scored, with count outcomes as rates over its observed steps and
Rg² over the first ``arch-min-obs`` steps, the same exposure for every scored row (Rg² is not additive in
time). Complete cases only (``auc_complete``), and the rows that survived to the horizon (``auc_survived``)
or not (``auc_censored``), are reported beside it, with ``n_short`` (observed under ``arch-min-obs`` steps,
not scored), ``censored_frac`` and the median observed steps. Intents use complete cases at the intent
horizon, where censoring is small; their censored fraction is reported.

**The matched random label** is ``mi.perm_test``: ``x`` is permuted within strata (cohort × age quartile
over the scored rows of the family), which keeps each stratum's prevalence, so a label that only tracks age
or the cohort's phase scores like the random one. ``z`` and ``p`` are against that null, pooled and per
cohort. The verdict is provisional (DESIGN §28.8): ``insufficient`` below 30 positives or negatives or with
fewer than 99 permutations (``p`` cannot reach 0.01), ``pass`` if ``p <= 0.01``, ``auc - null_mean >= 0.05``
and (intents) the label's coverage never exceeds 0.6, else ``fail``.

**Coverage and stability.** At every evaluation step after the warm-up the tool reads ``Observer.row_block()``
and reports the mean and max share of each rule (``intent_match``, ``archetype_match``) and of each displayed
label (``intent``, ``archetype``). A *flip* is a changed displayed intent (or archetype primary) between two
consecutive evaluations of a row that kept its uid and was labelled both times;
``flips_per_1000 = 1000·flips/(pairs·every)``.

The run directory is read only. The report is deterministic for a given run, parameters and device (no wall
time in it), and JSON-safe (``null`` for undefined numbers). ``--no-verify`` skips the check of every stored
``metrics.jsonl`` row on the way (``resim.compare_row``), which also confirms that observing leaves the
trajectory alone on this backend. Usage::

    python -m alife.analysis.labels RUN --from STEP --steps N --out FILE.json [--states DIR]
        [--rules {v2,v3}] [--dump DIR]
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, NamedTuple

import mlx.core as mx
import numpy as np

from alife import layout, ops, spatial, weather
from alife.analysis import mi
from alife.analysis.resim import ResimMismatch, compare_row
from alife.analysis.signals import load_state
from alife.config import Config, config_hash, dims
from alife.game.observe import EVAL_BIT, care_out
from alife.game.observer import Observer, default_rules
from alife.game.rules import (
    ARCHETYPE_NAMES,
    INTENT_NAMES,
    INTENT_VERSION,
    RULES_V2,
    Archetype,
    RuleSet,
    arch_layout,
)
from alife.runio import RunDir
from alife.state import State, to_numpy
from alife.step import ActFlag, Acts, make_step

__all__ = [
    "ARCH_OUTCOMES",
    "INTENT_OUTCOMES",
    "OUTCOMES",
    "auc",
    "score_archetypes",
    "score_intents",
    "study",
    "verdict",
]

# Verdict thresholds (provisional, DESIGN §28.8); recorded in the report's params.
MIN_CLASS: Final = 30  # positives and negatives needed for a verdict
P_MAX: Final = 0.01
MARGIN: Final = 0.05  # auc - null_mean
COVERAGE_MAX: Final = 0.6  # an intent label or rule that covers more of the evaluated rows than this fails
N_AGE_BINS: Final = 4
HISTORY: Final = 64  # steps of per-row input history kept for the dump (DESIGN §32)

# The per-step outcome columns of the cohort trackers (int32 totals).
OUTCOMES: Final = ("bud", "attack", "formed", "thrust", "defend", "emit", "harvest", "care", "out", "got")
_OC: Final = {name: i for i, name in enumerate(OUTCOMES)}
_I32 = mx.int32
_F32 = mx.float32

# Archetype name -> outcome label of the primary score (the rate or Rg² the rule claims).
ARCH_OUTCOMES: Final = {
    "gatherer": "harvest_rate",
    "hunter": "attack_rate",
    "provider": "out_rate",
    "trader": "min_out_got_rate",
    "nomad": "rg2",
    "settler": "neg_rg2",
    "drifter": "wind_share",
    "floater": "z_held",
    "sinker": "descent",
}
# Under a wind-relative rule set the Nomad and Settler outcomes are those of the drift-subtracted track.
ARCH_OUTCOMES_REL: Final = {"nomad": "rg2_rel", "settler": "neg_rg2_rel"}
# The z bands of the version-3 strata, as fractions of the world height (0-4, 4-32, 32-56, 56-80 of H = 80).
Z_BAND_EDGES: Final = (0.05, 0.4, 0.7)
ALOFT: Final = (
    0.1  # "aloft" is z >= 0.1·H (8 of 80): the Floater's pool at T and the steps its outcome counts
)
N_Z_BANDS: Final = len(Z_BAND_EDGES) + 1
# A row within this distance of the lid ``z_hi`` is *pinned*: the surface clamp zeroes its stored ``v_z``,
# which satisfies the Floater rule's vertical-speed bound and its outcome (``z_k = z_T``) whatever the agent
# does, so
# pinned rows are out of the Floater's pool (as grounded rows are out of the Sinker's).
LID_MARGIN: Final = 0.1
# Intent name -> (outcome column, sign); Resting predicts *less* thrust.
INTENT_OUTCOMES: Final = {
    "budding": ("bud", 1),
    "hunting": ("attack", 1),
    "fleeing": ("thrust", 1),
    "defending": ("defend", 1),
    "caring": ("care", 1),
    "bonding": ("formed", 1),
    "signalling": ("emit", 1),
    "feeding": ("harvest", 1),
    "roaming": ("thrust", 1),
    "resting": ("thrust", -1),
}

# --- scoring (host, numpy) ------------------------------------------------------------------------------


def _num(v: float | None) -> float | None:
    """A JSON-safe float: NaN and infinities become None."""
    return None if v is None or not math.isfinite(v) else float(v)


def _auc_from_ranks(x: np.ndarray, r: np.ndarray) -> float:
    """Mann–Whitney AUC of the bool ``x`` from the 0-based average ranks ``r`` of the outcome."""
    n_pos = int(x.sum())
    n_neg = x.size - n_pos
    return float((r @ x.astype(np.float64) - n_pos * (n_pos - 1) / 2.0) / (n_pos * n_neg))


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """``P(y⁺ > y⁻) + ½P(tie)`` for the positives ``x`` (bool), by average ranks; NaN if a class is empty."""
    x = np.asarray(x, dtype=bool)
    if x.all() or not x.any():
        return math.nan
    return _auc_from_ranks(x, mi.rank(np.asarray(y, dtype=np.float64)))


def _score(
    x: np.ndarray, y: np.ndarray, strata: np.ndarray, n_perm: int, rng: np.random.Generator
) -> dict[str, Any]:
    """AUC of ``x`` for ``y`` against the matched random label: ``x`` permuted within ``strata``."""
    x = np.asarray(x, dtype=bool)
    n, n_pos = int(x.size), int(x.sum())
    out: dict[str, Any] = {
        "n": n,
        "n_pos": n_pos,
        "auc": None,
        "null_mean": None,
        "null_sd": None,
        "z": None,
        "p": None,
    }
    if n_pos == 0 or n_pos == n:
        return out
    r = mi.rank(np.asarray(y, dtype=np.float64))  # AUC depends on y through its average ranks only
    res = mi.perm_test(_auc_from_ranks, x, r, n_perm, rng, strata=np.asarray(strata, dtype=np.int64))
    out.update(
        auc=_num(res.mi),
        null_mean=_num(res.null_mean),
        null_sd=_num(res.null_sd),
        z=_num(res.z),
        p=_num(res.p),
    )
    return out


def _rng(seed: int, label: int, variant: int) -> np.random.Generator:
    """The stream of one score: ``[seed, label_index, variant]`` (variant 0 pooled, 1–3 secondary columns,
    ``10 + c`` cohort ``c``, ``20 + b`` z band ``b``). Labels: archetypes 1–7, intent labels ``100 + i``,
    intent rules ``200 + i``."""
    return np.random.default_rng([seed, label, variant])


def verdict(
    score: Mapping[str, Any],
    coverage_max: float | None = None,
    *,
    is_intent: bool = False,
    n_perm: int | None = None,
) -> str:
    """``insufficient`` / ``pass`` / ``fail`` of a pooled score (provisional criterion, DESIGN §28.8).

    ``n_perm`` is the number of permutations of the score's null, when known: ``p`` is at least
    ``1/(1 + n_perm)``, so with fewer than ``1/P_MAX − 1`` of them no label can pass and the verdict is
    ``insufficient``, not ``fail``."""
    n, n_pos = score["n"], score["n_pos"]
    if n_pos < MIN_CLASS or n - n_pos < MIN_CLASS or score["p"] is None or score["null_mean"] is None:
        return "insufficient"
    if n_perm is not None and 1.0 / (1 + n_perm) > P_MAX:
        return "insufficient"
    ok = score["p"] <= P_MAX and score["auc"] - score["null_mean"] >= MARGIN
    if is_intent and coverage_max is not None:
        ok = ok and coverage_max <= COVERAGE_MAX
    return "pass" if ok else "fail"


def _age_strata(cohort: np.ndarray, age: np.ndarray, scored: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(age quartile, cohort·4 + quartile)`` per row; the quartiles are over the scored rows of the family
    (``mi.quantile_bins``: deterministic, tied ages share a bin), 0 elsewhere."""
    quart = np.zeros(cohort.size, dtype=np.int64)
    if scored.any():
        quart[scored] = mi.quantile_bins(age[scored], N_AGE_BINS)
    return quart, cohort.astype(np.int64) * N_AGE_BINS + quart


def _scored_with_cohorts(
    x: np.ndarray,
    y: np.ndarray,
    sel: np.ndarray,
    cohort: np.ndarray,
    quart: np.ndarray,
    strata: np.ndarray,
    cohort_steps: Sequence[int],
    n_perm: int,
    seed: int,
    label: int,
    *,
    per_cohort: bool,
) -> dict[str, Any]:
    """The pooled score over the rows ``sel`` and, if asked, one per cohort (stratified by age quartile)."""
    out = _score(x[sel], y[sel], strata[sel], n_perm, _rng(seed, label, 0))
    if per_cohort:
        cohorts = []
        for c, step in enumerate(cohort_steps):
            m = sel & (cohort == c)
            if not m.any():
                continue
            s = _score(x[m], y[m], quart[m], n_perm, _rng(seed, label, 10 + c))
            cohorts.append({"cohort": int(step), **s})
        out["per_cohort"] = cohorts
    return out


def z_bands(z: np.ndarray, height: float) -> np.ndarray:
    """The version-3 stratum band 0..``N_Z_BANDS`` − 1 of each height (``Z_BAND_EDGES`` of ``height``)."""
    return np.digitize(np.asarray(z, dtype=np.float64), [e * height for e in Z_BAND_EDGES]).astype(np.int64)


def _band_score(
    band: int,
    x: np.ndarray,
    y: np.ndarray,
    sel: np.ndarray,
    strata: np.ndarray,
    height: float,
    n_perm: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """The score of one z band at the cohort step (``Z_BAND_EDGES``), over the rows ``sel`` of that band."""
    edges = [0.0, *(e * height for e in Z_BAND_EDGES), height]
    return {
        "band": band,
        "z_lo": edges[band],
        "z_hi": edges[band + 1],
        **_score(x[sel], y[sel], strata[sel], n_perm, rng),
    }


def score_archetypes(
    tab: Mapping[str, np.ndarray],
    *,
    arch_horizon: int,
    arch_min_obs: int,
    n_perm: int,
    seed: int,
    cohort_steps: Sequence[int] | None = None,
    wind_rel: bool = False,
    height: float = 80.0,
    z_floor: float = 8.0,
    room: float = 1.5,
    z_pin: float | None = None,
    only: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Per-archetype scores from the pooled archetype table, one row per pool row of every cohort.

    Columns: ``cohort`` (index), ``age``, ``primary`` and ``match`` (the ``Archetype`` and the match bits at
    the cohort step), ``n_ok`` (observed steps at the horizon), ``rg2_min`` (Rg² over the first
    ``arch_min_obs`` steps, valid where ``n_ok >= arch_min_obs``), ``rg2_full`` (Rg² at the horizon) and the
    horizon totals ``harvest``, ``attack``, ``out`` and ``got``. A row is *scored* if observed at least
    ``arch_min_obs`` steps; the primary score is on rates over its observed steps (Rg² over the first
    ``arch_min_obs``).

    The storm world (DESIGN §32) adds optional columns: ``rg2_rel_min``/``rg2_rel_full`` (Rg² of the track
    with the wind drift subtracted), ``z0`` (height at the cohort step), ``zk_min`` (height after the first
    ``arch_min_obs`` observed steps) and ``wshare_min`` (the
    wind share of the horizontal displacement over them). With the first, Nomad and Settler get ``auc_raw``
    and ``auc_rel`` beside ``auc`` (the primary, on ``rg2_rel`` where ``wind_rel`` and on ``rg2`` otherwise);
    with the others, Drifter, Floater and Sinker are scored as in the module docstring, ``height`` being the
    world height, ``z_floor`` the altitude of the Floater's pool, ``z_pin`` the height from which a row is
    pinned at the lid and left out of that pool (None: no row is) and ``room`` the Sinker's largest descent.
    Drifter needs no height, so a 2D world scores it too (one z band).
    ``only`` limits the report to the named archetypes (the calibration sweep of one label).
    """
    cohort = np.asarray(tab["cohort"])
    steps = list(cohort_steps) if cohort_steps is not None else list(range(int(cohort.max(initial=-1)) + 1))
    n_ok = np.minimum(np.asarray(tab["n_ok"]), arch_horizon)
    scored = n_ok >= arch_min_obs
    full = n_ok >= arch_horizon
    quart, strata = _age_strata(cohort, np.asarray(tab["age"]), scored)
    per = np.maximum(n_ok, 1).astype(np.float64)
    tot = {k: np.asarray(tab[k], dtype=np.float64) for k in ("harvest", "attack", "out", "got")}
    low = np.minimum(tot["out"], tot["got"])
    rg_min, rg_full = np.asarray(tab["rg2_min"], np.float64), np.asarray(tab["rg2_full"], np.float64)
    has_rel = "rg2_rel_min" in tab
    rel_min = np.asarray(tab["rg2_rel_min"], np.float64) if has_rel else rg_min
    rel_full = np.asarray(tab["rg2_rel_full"], np.float64) if has_rel else rg_full
    use_rel = wind_rel and has_rel
    nom_min, nom_full = (rel_min, rel_full) if use_rel else (rg_min, rg_full)
    # name -> (primary outcome per row, outcome for the complete cases)
    spec: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "gatherer": (tot["harvest"] / per, tot["harvest"]),
        "hunter": (tot["attack"] / per, tot["attack"]),
        "provider": (tot["out"] / per, tot["out"]),
        "trader": (low / per, low),
        "nomad": (nom_min, nom_full),
        "settler": (-nom_min, -nom_full),
    }
    # the version-3 archetypes: (outcome, rows the score is over), strata extended by the z band at T
    extra: dict[str, np.ndarray] = {}
    zb = np.zeros(cohort.size, dtype=np.int64)
    pinned = np.zeros(cohort.size, dtype=bool)
    if "wshare_min" in tab:  # the wind share of the displacement needs no height: scored in 2D too
        spec["drifter"] = (np.asarray(tab["wshare_min"], np.float64),) * 2
        extra["drifter"] = np.ones(cohort.size, dtype=bool)
    if "z0" in tab:
        z0 = np.asarray(tab["z0"], np.float64)
        zb = z_bands(z0, height)
        pinned = z0 >= z_pin if z_pin is not None else pinned
        if "zk_min" in tab:
            held = -np.abs(np.asarray(tab["zk_min"], np.float64) - z0)
            spec["floater"] = (held, held)
            extra["floater"] = (z0 >= z_floor) & ~pinned  # the pool aloft at T, off the lid
            fall = (z0 - np.asarray(tab["zk_min"], np.float64)) / np.maximum(np.minimum(z0, room), 1e-9)
            spec["sinker"] = (fall, fall)
            extra["sinker"] = z0 >= room  # rows with room to fall
    match, primary = np.asarray(tab["match"]), np.asarray(tab["primary"])
    out: dict[str, Any] = {}
    for a in Archetype:
        if a is Archetype.NONE:
            continue
        name = a.name.lower()
        if only is not None and name not in only:  # a calibration sweep scores one label
            continue
        x = ((match >> (a.value - 1)) & 1).astype(bool)
        entry: dict[str, Any] = {
            "pool": int(x.size),
            "prevalence_match": _num(x.mean()) if x.size else None,
            "prevalence_primary": _num((primary == a.value).mean()) if x.size else None,
        }
        if name in spec:
            y_primary, y_complete = spec[name]
            sc, fl = scored, full
            q, st = quart, strata
            if name in extra:  # the z band joins the strata, and the score is over its own rows
                sc, fl = scored & extra[name], full & extra[name]
                q, st = quart * N_Z_BANDS + zb, strata * N_Z_BANDS + zb
            outcome = (
                ARCH_OUTCOMES_REL[name] if use_rel and name in ARCH_OUTCOMES_REL else ARCH_OUTCOMES[name]
            )
            entry.update(outcome=outcome, horizon=arch_horizon, min_obs=arch_min_obs)
            pooled = _scored_with_cohorts(
                x, y_primary, sc, cohort, q, st, steps, n_perm, seed, a.value, per_cohort=True
            )
            entry.update(pooled)
            entry["verdict"] = verdict(pooled, n_perm=n_perm)
            entry["auc_complete"] = _score(x[fl], y_complete[fl], st[fl], n_perm, _rng(seed, a.value, 1))
            entry["auc_survived"] = _score(
                x[sc & fl], y_primary[sc & fl], st[sc & fl], n_perm, _rng(seed, a.value, 2)
            )
            entry["auc_censored"] = _score(
                x[sc & ~fl], y_primary[sc & ~fl], st[sc & ~fl], n_perm, _rng(seed, a.value, 3)
            )
            entry["exposure"] = {side: _exposure(m, n_ok, sc, fl) for side, m in (("pos", x), ("neg", ~x))}
            if name in extra:  # the AUC per z band at T: where the pooled score comes from
                entry["by_band"] = [
                    _band_score(
                        b, x, y_primary, sc & (zb == b), st, height, n_perm, _rng(seed, a.value, 20 + b)
                    )
                    for b in range(N_Z_BANDS)
                ]
            if name == "floater":  # the rows left out of the pool because the lid pins them
                aloft = scored & (z0 >= z_floor)
                entry["pinned"] = {
                    "z_pin": z_pin,
                    "n": int((aloft & pinned).sum()),
                    "n_match": int((aloft & pinned & x).sum()),
                }
            if has_rel and name in ("nomad", "settler"):  # the other way round: the wind-relative comparison
                sign = 1.0 if name == "nomad" else -1.0
                for key, col, seed_variant in (("auc_raw", rg_min, 4), ("auc_rel", rel_min, 5)):
                    entry[key] = _score(
                        x[scored],
                        (sign * col)[scored],
                        strata[scored],
                        n_perm,
                        _rng(seed, a.value, seed_variant),
                    )
        out[name] = entry
    return out


def _exposure(m: np.ndarray, n_ok: np.ndarray, scored: np.ndarray, full: np.ndarray) -> dict[str, Any]:
    """Censoring of the pool rows ``m``: how many were observed too briefly to score (``n_short``), the share
    not observed to the horizon (``censored_frac``, short rows included) and the median observed steps of the
    scored ones."""
    n = int(m.sum())
    sc = m & scored
    return {
        "n_pool": n,
        "n_short": int((m & ~scored).sum()),
        "censored_frac": _num(1.0 - float((m & full).sum()) / n) if n else None,
        "n_ok_median": _num(float(np.median(n_ok[sc]))) if sc.any() else None,
    }


def score_intents(
    tab: Mapping[str, np.ndarray],
    *,
    n_perm: int,
    seed: int,
    intent_horizon: int,
    cohort_steps: Sequence[int] | None = None,
    coverage_max: Mapping[str, Mapping[str, float | None]] | None = None,
) -> dict[str, Any]:
    """Per-intent scores from the pooled intent table, one row per evaluated row of every cohort.

    Columns: ``cohort`` (index), ``age``, ``label`` (displayed ``Intent``, 0 before the hysteresis settles),
    ``match`` (predicate bits | ``EVAL_BIT``), ``complete`` (the row was observed all ``intent_horizon``
    steps) and the horizon totals of the ``OUTCOMES`` the intents use. Complete cases only. ``auc_rule``
    is over the evaluated rows; ``auc_label`` over those with a displayed label (the others have no claim).
    ``coverage_max`` maps ``"intent"`` and ``"intent_match"`` to the max share per name; each verdict
    requires its own (the label's, or the rule's) to stay at most ``COVERAGE_MAX``.
    """
    cohort = np.asarray(tab["cohort"])
    steps = list(cohort_steps) if cohort_steps is not None else list(range(int(cohort.max(initial=-1)) + 1))
    complete = np.asarray(tab["complete"], dtype=bool)
    quart, strata = _age_strata(cohort, np.asarray(tab["age"]), complete)
    label, match = np.asarray(tab["label"]), np.asarray(tab["match"])
    cov = coverage_max or {}
    out: dict[str, Any] = {}
    for i, name in enumerate(INTENT_NAMES, start=1):
        column, sign = INTENT_OUTCOMES[name]
        y = sign * np.asarray(tab[column], dtype=np.float64)
        x_label, x_rule = label == i, ((match >> (i - 1)) & 1).astype(bool)
        sel_label, sel_rule = complete & (label != 0), complete
        a_label = _scored_with_cohorts(
            x_label, y, sel_label, cohort, quart, strata, steps, n_perm, seed, 100 + i, per_cohort=True
        )
        a_rule = _scored_with_cohorts(
            x_rule, y, sel_rule, cohort, quart, strata, steps, n_perm, seed, 200 + i, per_cohort=True
        )
        cov_label = (cov.get("intent") or {}).get(name)
        cov_rule = (cov.get("intent_match") or {}).get(name)
        out[name] = {
            "outcome": ("-" if sign < 0 else "") + column,
            "horizon": intent_horizon,
            "n_pool": int(label.size),
            "n_labelled": int((label != 0).sum()),
            "censored_frac": _num(1.0 - float(complete.mean())) if complete.size else None,
            "prevalence_label": _num(x_label.mean()) if x_label.size else None,
            "prevalence_rule": _num(x_rule.mean()) if x_rule.size else None,
            "auc_label": a_label,
            "auc_rule": a_rule,
            "coverage_max": {"label": cov_label, "rule": cov_rule},
            "verdict": verdict(a_label, cov_label, is_intent=True, n_perm=n_perm),
            "verdict_rule": verdict(a_rule, cov_rule, is_intent=True, n_perm=n_perm),
        }
    return out


# --- cohort trackers (device) ---------------------------------------------------------------------------


class TrackState(NamedTuple):
    """What a cohort accumulates, per row, while the row stays the same occupant (the cohort's uids are a
    constant array kept beside it, so that no output of the compiled update is an input)."""

    ok: mx.array  # (N,) bool: alive with the cohort's uid at every step so far
    n_ok: mx.array  # (N,) int32: steps observed
    tot: mx.array  # (N, len(OUTCOMES)) int32: outcome totals over the observed steps
    pos: mx.array  # (N, nd) float32: the last observed position
    u: mx.array  # (N, nd) float32: the unwrapped track (starts at 0)
    mean: mx.array  # (N, nd) float32: Welford mean of the track
    m2: mx.array  # (N,) float32: Welford sum of squared deviations (Rg² = m2 / n_ok)
    # The storm world (DESIGN §32): the track with the wind drift subtracted (equal to ``u`` without drift),
    # and the height: at the cohort step, now, and the steps spent at or above the Floater's altitude.
    ur: mx.array  # (N, nd) float32: u_k = u_{k-1} + min_image(Δpos) − drift_k
    mean_r: mx.array  # (N, nd) float32
    m2_r: mx.array  # (N,) float32
    z0: mx.array  # (N,) float32: height at the cohort step (0 in 2D)
    zk: mx.array  # (N,) float32: the last observed height


def track_start(member: mx.array, pos: mx.array) -> TrackState:
    """The tracker of a cohort whose members are the rows where ``member``, at positions ``pos``."""
    n, nd = pos.shape
    z = pos[:, 2] if nd == 3 else mx.zeros((n,), dtype=_F32)
    return TrackState(
        ok=member,
        n_ok=mx.zeros((n,), dtype=_I32),
        tot=mx.zeros((n, len(OUTCOMES)), dtype=_I32),
        pos=pos,
        u=mx.zeros((n, nd), dtype=_F32),
        mean=mx.zeros((n, nd), dtype=_F32),
        m2=mx.zeros((n,), dtype=_F32),
        ur=mx.zeros((n, nd), dtype=_F32),
        mean_r=mx.zeros((n, nd), dtype=_F32),
        m2_r=mx.zeros((n,), dtype=_F32),
        z0=z,
        zk=z,
    )


def track_step(
    cuid: mx.array,
    tr: TrackState,
    alive: mx.array,
    uid: mx.array,
    pos: mx.array,
    oc: mx.array,
    size: float,
    drift: mx.array | None = None,
) -> TrackState:
    """One step: a row is observed while ``alive`` with its cohort uid; its outcomes ``oc`` (N, 10) are added,
    and its position extends the unwrapped track (``u_k = u_{k-1} + min_image(pos_k − pos_{k-1})``, every
    component: ``min_image`` is the identity on z, DESIGN §27.1), whose mean and M2 are a Welford update in
    float32.

    ``drift`` (N, nd), the wind drift of the post-step state (``weather.agent_drift``; None: no wind), extends
    the second track, ``ur_k = ur_{k-1} + min_image(pos_k − pos_{k-1}) − drift_k`` (the index convention of
    DESIGN §32: the drift of step ``k`` against the displacement from ``k − 1`` to ``k``); without drift it is
    the first one."""
    ok = tr.ok & alive & (uid == cuid)
    k = ok[:, None]
    n = tr.n_ok + ok.astype(_I32)
    nf = mx.maximum(n, 1).astype(_F32)[:, None]
    step_d = ops.min_image(pos - tr.pos, size)
    u = tr.u + mx.where(k, step_d, 0.0)
    delta = u - tr.mean
    mean = tr.mean + delta / nf
    m2 = tr.m2 + (delta * (u - mean)).sum(axis=1)
    if drift is None:
        ur, mean_r, m2_r = u, mean, m2
    else:
        ur = tr.ur + mx.where(k, step_d - drift, 0.0)
        delta_r = ur - tr.mean_r
        mean_r = tr.mean_r + delta_r / nf
        m2_r = tr.m2_r + (delta_r * (ur - mean_r)).sum(axis=1)
    z = pos[:, 2] if pos.shape[1] == 3 else mx.zeros_like(tr.zk)
    return TrackState(
        ok=ok,
        n_ok=n,
        tot=tr.tot + mx.where(k, oc, 0),
        pos=mx.where(k, pos, tr.pos),
        u=u,
        mean=mx.where(k, mean, tr.mean),
        m2=mx.where(ok, m2, tr.m2),
        ur=ur,
        mean_r=mx.where(k, mean_r, tr.mean_r),
        m2_r=mx.where(ok, m2_r, tr.m2_r),
        z0=tr.z0,
        zk=mx.where(ok, z, tr.zk),
    )


def step_outcomes(acts: Acts, last_harvest: mx.array, kin_prev: mx.array) -> mx.array:
    """(N, len(OUTCOMES)) int32 outcomes of one step, in ``OUTCOMES`` order: bud, attack and bond-formed
    flags (0 or 1), ``|thrust|``, defence and emission (in 1/256 and deposit quanta), the harvest, the energy
    sent to kin (``observe.care_out`` with the ``slot_kin`` read *before* this step's observer update) and
    the energy shared out and got."""
    f = acts.flags.astype(_I32)

    def flag(bit: ActFlag) -> mx.array:
        return ((f & bit.value) != 0).astype(_I32)

    cols = [
        flag(ActFlag.BUDDED),
        flag(ActFlag.ATTACKED),
        flag(ActFlag.FORMED),
        mx.abs(acts.thrust.astype(_I32)),
        acts.defend.astype(_I32),
        acts.emit.astype(_I32),
        last_harvest.astype(_I32),
        care_out(kin_prev, acts.share_out),
        acts.share_out.sum(axis=1),
        acts.share_got.sum(axis=1),
    ]
    return mx.stack(cols, axis=1).astype(_I32)


def _host_track(tr: TrackState) -> dict[str, np.ndarray]:
    """A tracker on the host: ``n_ok``, the ``OUTCOMES`` totals, Rg² and the wind-relative Rg² (float64 from
    the float32 M2), the heights (``z0``, ``zk``) and the horizontal displacements ``u_h`` (raw) and
    ``ur_h`` (drift subtracted), whose difference is the horizontal displacement the wind gave."""
    n = to_numpy(tr.n_ok)
    m2 = to_numpy(tr.m2).astype(np.float64)
    m2_r = to_numpy(tr.m2_r).astype(np.float64)
    tot = to_numpy(tr.tot)
    return {
        "n_ok": n,
        "rg2": m2 / np.maximum(n, 1),
        "rg2_rel": m2_r / np.maximum(n, 1),
        "z0": to_numpy(tr.z0).astype(np.float64),
        "zk": to_numpy(tr.zk).astype(np.float64),
        "u_h": to_numpy(tr.u)[:, :2].astype(np.float64),
        "ur_h": to_numpy(tr.ur)[:, :2].astype(np.float64),
        **{name: tot[:, i].astype(np.int64) for name, i in _OC.items()},
    }


def wind_share(u_h: np.ndarray, ur_h: np.ndarray) -> np.ndarray:
    """The wind share of the horizontal displacement, ``|Σ drift| / (|Σ drift| + |Σ own|)`` with ``Σ own =
    ur_h`` (the displacement less the drift) and ``Σ drift = u_h − ur_h``; 0 where nothing moved."""
    wind = np.linalg.norm(u_h - ur_h, axis=1)
    own = np.linalg.norm(ur_h, axis=1)
    den = wind + own
    return np.where(den > 0, wind / np.where(den > 0, den, 1.0), 0.0)


class _Cohort:
    """One snapshot of the labelled rows at step ``step`` and its tracker."""

    def __init__(self, index: int, step: int, cuid: mx.array, trk: TrackState, snap: dict[str, Any]) -> None:
        self.index, self.step, self.cuid, self.trk, self.snap = index, step, cuid, trk, snap
        self.k = 0
        self.at_min: dict[str, np.ndarray] | None = None
        self.at_arch: dict[str, np.ndarray] | None = None
        self.at_intent: dict[str, np.ndarray] | None = None


def _share_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": _num(sum(values) / len(values)) if values else None,
        "max": _num(max(values)) if values else None,
    }


class _Coverage:
    """Shares of ``Observer.row_block()`` over the evaluation steps after the warm-up."""

    FAMILIES: Final = (
        ("intent_match", INTENT_NAMES),
        ("intent", INTENT_NAMES),
        ("archetype", ARCHETYPE_NAMES),
        ("archetype_match", ARCHETYPE_NAMES),
    )

    def __init__(self) -> None:
        self.n = 0
        self.vals: dict[str, dict[str, list[float]]] = {
            f: {nm: [] for nm in names} for f, names in self.FAMILIES
        }

    def add(self, block: Mapping[str, Any]) -> None:
        self.n += 1
        for fam, _ in self.FAMILIES:
            shares = block[fam]
            if shares is not None:
                for name, v in shares.items():
                    self.vals[fam][name].append(float(v))

    def report(self) -> dict[str, Any]:
        return {
            "evaluations": self.n,
            **{fam: {nm: _share_summary(self.vals[fam][nm]) for nm in names} for fam, names in self.FAMILIES},
        }

    def maxima(self) -> dict[str, dict[str, float | None]]:
        return {
            fam: {nm: _share_summary(self.vals[fam][nm])["max"] for nm in names}
            for fam, names in self.FAMILIES
        }


class _Flips:
    """Label flips between consecutive evaluations of rows that kept their uid and were labelled twice."""

    def __init__(self) -> None:
        self.prev: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.pairs = {"intent": 0, "archetype": 0}
        self.flips = {"intent": 0, "archetype": 0}

    def add(self, uid: np.ndarray, arch: np.ndarray, intent: np.ndarray) -> None:
        if self.prev is not None:
            puid, parch, pintent = self.prev
            same = (uid >= 0) & (uid == puid)
            for fam, now, before in (("archetype", arch, parch), ("intent", intent, pintent)):
                both = same & (now != 0) & (before != 0)
                self.pairs[fam] += int(both.sum())
                self.flips[fam] += int((both & (now != before)).sum())
        self.prev = (uid, arch, intent)

    def per_1000(self, every: int) -> dict[str, float | None]:
        return {
            fam: _num(1000.0 * self.flips[fam] / (self.pairs[fam] * every)) if self.pairs[fam] else None
            for fam in ("intent", "archetype")
        }


# --- the study ------------------------------------------------------------------------------------------


def _concat(parts: list[dict[str, np.ndarray]], keys: Sequence[str]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([p[k] for p in parts]) if parts else np.zeros(0, dtype=np.int64) for k in keys}


def _check_params(
    rules: RuleSet,
    steps: int,
    warmup: int,
    cohort_every: int,
    arch_horizon: int,
    arch_min_obs: int,
    intent_horizon: int,
    n_perm: int,
    seed: int,
) -> None:
    def req(ok: bool, msg: str) -> None:
        if not ok:
            raise ValueError(msg)

    req(steps >= 1 and warmup >= 0, "steps must be >= 1 and warmup >= 0")
    req(
        cohort_every >= rules.every and cohort_every % rules.every == 0,
        f"cohort_every must be a positive multiple of rules.every ({rules.every})",
    )
    req(1 <= arch_min_obs <= arch_horizon, "need 1 <= arch_min_obs <= arch_horizon")
    req(intent_horizon >= 1, "intent_horizon must be >= 1")
    req(n_perm >= 1, "n_perm must be >= 1")
    req(seed >= 0, "seed must be >= 0")


def _bound_check(cfg: Config, horizon: int) -> None:
    """The int32 outcome totals hold ``horizon`` steps of the largest per-step outcome."""
    per_step = max(
        cfg.energy.harvest_max,
        cfg.combat.attack_max,
        layout.B * cfg.bonds.share_rate,
        layout.NCH * 255,
        256,
    )
    if per_step * horizon > (1 << 31) - 1:
        raise ValueError(
            f"an outcome total over {horizon} steps of up to {per_step} per step overflows int32"
        )


def study(
    run: Path | str,
    *,
    start: int,
    steps: int,
    states_dir: Path | None = None,
    warmup: int = 256,
    cohort_every: int = 256,
    arch_horizon: int = 1000,
    arch_min_obs: int = 100,
    intent_horizon: int = 10,
    n_perm: int = 200,
    seed: int = 0,
    verify: bool = True,
    rules: RuleSet | None = None,
    dump: Path | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """The validity study of a run from step ``start`` for ``steps`` steps (see the module docstring).

    Needs a checkpoint of the run (or a cached state in ``states_dir``) at ``start``. The cohorts are the
    multiples of ``rules.every`` from the first at or after ``start + warmup``, every ``cohort_every`` steps,
    as long as a full horizon fits in the study. ``rules`` defaults to ``observer.default_rules(cfg)``.
    With ``dump`` (a directory) the calibration table is written there (``write_dump``). Returns the
    JSON-safe report.
    """
    rd = RunDir.open(Path(run))
    cfg = rd.config()
    rules = default_rules(cfg) if rules is None else rules
    _check_params(
        rules, steps, warmup, cohort_every, arch_horizon, arch_min_obs, intent_horizon, n_perm, seed
    )
    rules.check(cfg)
    horizon = max(arch_horizon, intent_horizon)
    _bound_check(cfg, horizon)
    end = start + steps
    every = rules.every
    t0 = -(-(start + warmup) // every) * every  # the first evaluation step at or after start + warmup
    cohort_steps = list(range(t0, end - horizon + 1, cohort_every))
    if not cohort_steps:
        raise ValueError(
            f"no cohort fits: the first would be at step {t0} and needs {horizon} steps after it, "
            f"but the study ends at step {end}"
        )
    nd = dims(cfg).nd
    size = cfg.world.size

    state = load_state(rd, cfg, states_dir, start)
    obs, note = Observer.open(rd, state, cfg, rules=rules, tracers=False)
    resumed = any(st == start for st, _ in rd.checkpoints())
    source = "sidecar" if resumed and note is None else "fresh"
    rows = {int(r["step"]): r for r in rd.metrics_rows()} if verify else {}
    last_row = max(rows, default=-1)
    log(
        f"labels: {rd.path} from step {start} to {end}, {len(cohort_steps)} cohorts, "
        f"observers {source}, verify={verify}"
    )

    def outcomes_fn(acts: Acts, last_harvest: mx.array, kin_prev: mx.array) -> mx.array:
        return step_outcomes(acts, last_harvest, kin_prev)

    d3 = dims(cfg).d3
    wthr = cfg.weather.enabled
    # the altitude the Floater's outcome counts and the Sinker's largest descent: fixed by the world, not by
    # the rule under study, so a dump scores candidate thresholds against one outcome
    z_floor = ALOFT * cfg.world.height
    room = arch_min_obs * cfg.physics.gravity if d3 else 0.0
    z_pin = (
        dims(cfg).z_hi - LID_MARGIN if d3 else None
    )  # from here a row is pinned at the lid (the Floater's pool)

    def track_fn(
        cuid: mx.array,
        tr: TrackState,
        alive: mx.array,
        uid: mx.array,
        pos: mx.array,
        oc: mx.array,
        drift: mx.array | None,
    ) -> TrackState:
        return track_step(cuid, tr, alive, uid, pos, oc, size, drift)

    def drift_fn(wx: mx.array, pos: mx.array, buoy_cap: mx.array | None, alive: mx.array) -> mx.array:
        return weather.agent_drift(wx, pos, buoy_cap, alive, cfg)

    def feat_fn(
        vel: mx.array, drift: mx.array | None, pos: mx.array, buoy_cap: mx.array | None, buoy: mx.array | None
    ) -> mx.array:
        """The dump's per-step input history row: velocity, drift, height and lift fraction (1/256)."""
        n = vel.shape[0]
        z = pos[:, 2:3] if d3 else mx.zeros((n, 1), dtype=_F32)
        if d3 and buoy is not None:
            lift = ((buoy.astype(_I32) * mx.round(buoy_cap * 256.0).astype(_I32)) >> 8).astype(_F32)[:, None]
        else:
            lift = mx.zeros((n, 1), dtype=_F32)
        return mx.concatenate([vel, vel * 0.0 if drift is None else drift, z, lift], axis=1)

    oc_step = mx.compile(outcomes_fn) if cfg.run.compile else outcomes_fn
    tr_step = mx.compile(track_fn) if cfg.run.compile else track_fn
    drift_step = mx.compile(drift_fn) if cfg.run.compile else drift_fn
    feat_step = mx.compile(feat_fn) if cfg.run.compile else feat_fn
    history: collections.deque[np.ndarray] = collections.deque(maxlen=HISTORY)
    spatial.prepare_knn(cfg)  # the self-check of the fused search, once per process (§4)
    step_fn = make_step(cfg, observe=True)

    copy_points: dict[int, list[str]] = {}
    for k, what in ((intent_horizon, "intent"), (arch_min_obs, "min"), (arch_horizon, "arch")):
        copy_points.setdefault(k, []).append(what)

    coverage, flips = _Coverage(), _Flips()
    active: list[_Cohort] = []
    arch_parts: list[dict[str, np.ndarray]] = []
    intent_parts: list[dict[str, np.ndarray]] = []
    cohort_info: list[dict[str, int]] = []
    counters: list[mx.array] = []
    ledgers: list[mx.array] = []
    checked = 0
    t = start
    t_wall, t_log = time.perf_counter(), t

    while t < end:
        kin_prev = obs.o.slot_kin
        state, out = step_fn(state)
        obs.update(state, out.acts, t + 1)
        trackers: list[mx.array] = []
        buoy_cap = getattr(state, "buoy_cap", None)
        drift = (
            drift_step(state.wx, state.pos, buoy_cap, state.alive) if (wthr and (active or dump)) else None
        )
        feat = feat_step(state.vel, drift, state.pos, buoy_cap, out.acts.buoy) if dump is not None else None
        if active:
            oc = oc_step(out.acts, state.last_harvest, kin_prev)
            for c in active:
                c.trk = tr_step(c.cuid, c.trk, state.alive, state.uid, state.pos, oc, drift)
                trackers.extend(c.trk)
        mx.eval(state, out.base(), *obs.arrays(), *trackers, *([feat] if feat is not None else []))
        if feat is not None:  # the 64-step input history of the dump
            history.append(to_numpy(feat).astype(np.float16))
        if t < last_row:  # only the rows still to be compared need the step's counters
            counters.append(out.counters)
            ledgers.append(out.ledger)
        t += 1

        stored = rows.get(t)
        if stored is not None:
            if stored["steps"] == len(counters):
                bad = compare_row(
                    stored,
                    counters,
                    ledgers,
                    int(to_numpy(state.total)),
                    nd=nd,
                    weather=cfg.weather.enabled,
                )
                if bad:
                    raise ResimMismatch(
                        f"step {t}: observed re-simulation differs from metrics.jsonl in {bad}"
                    )
                checked += 1
            counters, ledgers = [], []

        finished: list[_Cohort] = []
        for c in active:
            c.k += 1
            for what in copy_points.get(c.k, ()):
                h = _host_track(c.trk)
                if what == "intent":
                    c.at_intent = h
                elif what == "min":
                    c.at_min = h
                else:
                    c.at_arch = h
            if c.k == horizon:
                finished.append(c)
        for c in finished:
            active.remove(c)
            _collect(c, arch_parts, intent_parts, intent_horizon, arch_min_obs, wthr, d3)

        if t in cohort_steps:
            active.append(_snapshot(len(cohort_info), t, state, obs, list(history) if dump else None))
            cohort_info.append({"step": t})
        if t % every == 0 and t >= t0:
            coverage.add(obs.row_block())
            lab = obs.host_labels()
            flips.add(to_numpy(obs.o.uid), lab["arch_primary"], lab["intent"])
        if t - t_log >= 1000:
            rate = (t - t_log) / (time.perf_counter() - t_wall)
            log(f"labels: step {t}, {rate:.1f} steps/s, {len(active)} cohorts open, {checked} rows verified")
            t_wall, t_log = time.perf_counter(), t

    assert not active, "every cohort has run its full horizon"
    arch_keys = (
        "cohort", "age", "primary", "match", "n_ok", "rg2_min", "rg2_full", "harvest", "attack", "out", "got",
    )  # fmt: skip
    if wthr:  # the wind-relative track and the wind share (DESIGN §32)
        arch_keys += ("rg2_rel_min", "rg2_rel_full", "wshare_min")
    if d3:
        arch_keys += ("z0", "zk_min")
    if dump is not None:
        arch_keys += ("row", "m6", "hist")
    arch_tab = _concat(arch_parts, arch_keys)
    intent_tab = _concat(
        intent_parts,
        ("cohort", "age", "label", "match", "complete", *sorted({c for c, _ in INTENT_OUTCOMES.values()})),
    )
    for info, n_a, n_i in zip(
        cohort_info,
        np.bincount(arch_tab["cohort"], minlength=len(cohort_info)),
        np.bincount(intent_tab["cohort"], minlength=len(cohort_info)),
        strict=True,
    ):
        info.update(n_archetype_pool=int(n_a), n_intent_pool=int(n_i))
    steps_of = [c["step"] for c in cohort_info]
    arch_scores = score_archetypes(
        arch_tab,
        arch_horizon=arch_horizon,
        arch_min_obs=arch_min_obs,
        n_perm=n_perm,
        seed=seed,
        cohort_steps=steps_of,
        wind_rel=rules.wind_rel,
        height=cfg.world.height,
        z_floor=z_floor,
        room=room,
        z_pin=z_pin,
    )
    intent_scores = score_intents(
        intent_tab,
        n_perm=n_perm,
        seed=seed,
        intent_horizon=intent_horizon,
        cohort_steps=steps_of,
        coverage_max=coverage.maxima(),
    )
    if dump is not None:
        write_dump(dump, arch_tab, intent_tab, cohort_info, rules)
    return {
        "run": str(rd.path),
        "config_hash": config_hash(cfg),
        "start": start,
        "steps": steps,
        "params": {
            "warmup": warmup,
            "cohort_every": cohort_every,
            "arch_horizon": arch_horizon,
            "arch_min_obs": arch_min_obs,
            "intent_horizon": intent_horizon,
            "n_perm": n_perm,
            "seed": seed,
            "verify": verify,
            "age_bins": N_AGE_BINS,
            "verdict": {
                "min_class": MIN_CLASS,
                "p_max": P_MAX,
                "margin": MARGIN,
                "coverage_max": COVERAGE_MAX,
            },
            "rules": dataclasses.asdict(rules),
            "wind_rel_outcomes": bool(wthr and rules.wind_rel),
            "z_floor": z_floor,
            "z_pin": z_pin,
            "sink_room": room,
        },
        "intent_version": INTENT_VERSION,
        "rules_digest": rules.digest(),
        "device": str(mx.default_device()),
        "mlx_version": mx.__version__,
        "observer_start": {"step": start, "source": source, "note": note},
        "rows_verified": checked,
        "cohorts": cohort_info,
        "archetypes": arch_scores,
        "intents": intent_scores,
        "coverage": coverage.report(),
        "flips_per_1000": flips.per_1000(every),
    }


def _snapshot(
    index: int, t: int, state: State, obs: Observer, hist: list[np.ndarray] | None = None
) -> _Cohort:
    """The cohort at step ``t``: the archetype pool (rows with a primary label), the intent pool (rows
    evaluated, ``EVAL_BIT``), the rows' uid, age (``t − 1 − birth_step``) and labels, and a tracker. With
    ``hist`` (the dump) also the pool rows' accumulator means and their last ``HISTORY`` input rows."""
    lab = obs.host_labels()
    uid = to_numpy(obs.o.uid)
    age = t - 1 - to_numpy(state.birth_step).astype(np.int64)
    pool_a = lab["arch_primary"] != 0
    pool_i = (lab["intent_match"] & EVAL_BIT) != 0
    member = pool_a | pool_i
    snap = {
        "rows_a": np.flatnonzero(pool_a),
        "rows_i": np.flatnonzero(pool_i),
        "age": age,
        "primary": lab["arch_primary"],
        "match_arch": lab["arch_match"],
        "label": lab["intent"],
        "match_intent": lab["intent_match"],
    }
    if hist is not None:
        rows = snap["rows_a"]
        snap["m6"] = (to_numpy(obs.o.arch)[rows] >> obs.rules.arch_k).astype(np.int32)
        # (rows, HISTORY, width) float16, the last step last; a short history (the study's first steps) is
        # padded at the front with zeros
        h = np.zeros((rows.size, HISTORY, hist[0].shape[1]), dtype=np.float16)
        if hist:
            h[:, HISTORY - len(hist) :] = np.stack([x[rows] for x in hist], axis=1)
        snap["hist"] = h
    cuid = mx.array(np.where(member, uid, -1).astype(np.int64))
    pos = mx.array(to_numpy(state.pos))
    trk = track_start(mx.array(member), pos)
    mx.eval(cuid, *trk)
    return _Cohort(index, t, cuid, trk, snap)


def _collect(
    c: _Cohort,
    arch_parts: list[dict[str, np.ndarray]],
    intent_parts: list[dict[str, np.ndarray]],
    intent_h: int,
    min_obs: int,
    wthr: bool = False,
    d3: bool = False,
) -> None:
    """Append a finished cohort's pool rows to the pooled tables."""
    s = c.snap
    assert c.at_arch is not None and c.at_min is not None and c.at_intent is not None
    ra, ri = s["rows_a"], s["rows_i"]
    part = {
        "cohort": np.full(ra.size, c.index, dtype=np.int64),
        "age": s["age"][ra],
        "primary": s["primary"][ra],
        "match": s["match_arch"][ra],
        "n_ok": c.at_arch["n_ok"][ra],
        "rg2_min": c.at_min["rg2"][ra],
        "rg2_full": c.at_arch["rg2"][ra],
        **{k: c.at_arch[k][ra] for k in ("harvest", "attack", "out", "got")},
    }
    if wthr:
        part.update(
            rg2_rel_min=c.at_min["rg2_rel"][ra],
            rg2_rel_full=c.at_arch["rg2_rel"][ra],
            wshare_min=wind_share(c.at_min["u_h"][ra], c.at_min["ur_h"][ra]),
        )
    if d3:
        part.update(z0=c.at_min["z0"][ra], zk_min=c.at_min["zk"][ra])
    if "m6" in s:  # the dump
        part.update(row=ra.astype(np.int64), m6=s["m6"], hist=s["hist"])
    arch_parts.append(part)
    columns = sorted({col for col, _ in INTENT_OUTCOMES.values()})
    intent_parts.append(
        {
            "cohort": np.full(ri.size, c.index, dtype=np.int64),
            "age": s["age"][ri],
            "label": s["label"][ri],
            "match": s["match_intent"][ri],
            "complete": c.at_intent["n_ok"][ri] >= intent_h,
            **{k: c.at_intent[k][ri] for k in columns},
        }
    )


def write_dump(
    directory: Path,
    arch_tab: Mapping[str, np.ndarray],
    intent_tab: Mapping[str, np.ndarray],
    cohorts: Sequence[Mapping[str, int]],
    rules: RuleSet,
) -> Path:
    """The calibration table of a study (``labels_dump.npz`` in ``directory``): every column of the pooled
    archetype table under ``arch.<name>`` (with ``arch.m6``, the pool rows' accumulator means at the cohort
    step in the ``arch_layout`` of the rules, and ``arch.hist``, their last ``HISTORY`` input rows
    ``[vel, drift, z, lift]``), the intent table under ``intent.<name>`` and ``meta`` (JSON: the rules, the
    cohort steps and the layout). Candidate thresholds are scored from it with ``score_archetypes``."""
    directory.mkdir(parents=True, exist_ok=True)
    nd = (
        (arch_tab["m6"].shape[1] - 5 - int(rules.wind_rel) - 2 * int(rules.vertical)) // 2
        if "m6" in arch_tab
        else 0
    )
    meta = {
        "rules": dataclasses.asdict(rules),
        "rules_digest": rules.digest(),
        "cohorts": list(cohorts),
        "layout": arch_layout(nd, rules)._asdict(),
        "history": HISTORY,
        "history_columns": "vel[nd], drift[nd], z, lift",
    }
    path = directory / "labels_dump.npz"
    np.savez_compressed(
        path,
        **{f"arch.{k}": v for k, v in arch_tab.items()},
        **{f"intent.{k}": v for k, v in intent_tab.items()},
        meta=np.array(json.dumps(meta, sort_keys=True)),
    )
    return path


# --- command line ---------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m alife.analysis.labels", description=__doc__.split("\n")[0])
    ap.add_argument("run", type=Path, help="run directory (read only)")
    ap.add_argument("--from", dest="start", type=int, required=True, help="step to start from (a checkpoint)")
    ap.add_argument("--steps", type=int, required=True, help="steps to simulate from there")
    ap.add_argument("--out", type=Path, required=True, help="report JSON (outside the run directory)")
    ap.add_argument("--states", type=Path, default=None, help="directory of re-simulated states (resim)")
    ap.add_argument("--warmup", type=int, default=256)
    ap.add_argument(
        "--cohort-every", type=int, default=256, help="a multiple of the rules' evaluation period"
    )
    ap.add_argument("--arch-horizon", type=int, default=1000)
    ap.add_argument("--arch-min-obs", type=int, default=100)
    ap.add_argument("--intent-horizon", type=int, default=10)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-verify", action="store_true", help="skip the check of the stored metrics rows")
    ap.add_argument(
        "--rules",
        choices=("v2", "v3"),
        default=None,
        help="the rule set (default: the observer's default for the world, version 3); v2 runs the version-2 "
        "predicates on the raw velocity, for the wind-relative comparison on the same window",
    )
    ap.add_argument(
        "--dump",
        type=Path,
        default=None,
        help="directory for the calibration table (labels_dump.npz); outside the run directory",
    )
    a = ap.parse_args(argv)
    if a.out.resolve().is_relative_to(a.run.resolve()):
        ap.error("--out must not be inside the run directory")
    if a.dump is not None and a.dump.resolve().is_relative_to(a.run.resolve()):
        ap.error("--dump must not be inside the run directory")
    try:
        report = study(
            a.run,
            start=a.start,
            steps=a.steps,
            states_dir=a.states,
            warmup=a.warmup,
            cohort_every=a.cohort_every,
            arch_horizon=a.arch_horizon,
            arch_min_obs=a.arch_min_obs,
            intent_horizon=a.intent_horizon,
            n_perm=a.n_perm,
            seed=a.seed,
            verify=not a.no_verify,
            rules=RULES_V2 if a.rules == "v2" else None,
            dump=a.dump,
            log=lambda m: print(m, flush=True),
        )
    except ResimMismatch as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
