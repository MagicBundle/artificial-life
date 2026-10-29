"""The arm comparison (DESIGN §44.8): paired block contrasts, the seed rule, Holm per feature, the call.

Panels are indexed by ``(arm, seed)``. Arm names: ``U0``..``U5`` (the build-up in the pre-registered order),
``L_<feature>`` (leave-one-out: U5 minus the feature), ``I_<feature>`` (inert twin in the full stack), ``T0``
and ``T5`` (seed twins of U0 and U5). Per feature ``k`` at build-up position ``i`` and per endpoint:

* ``buildup`` = ``U_i - U_(i-1)``, ``loo`` = ``U5 - L_k``, ``control`` = ``U5 - I_k`` (the control contrast
  separates the feature's function from its cost and layout);
* each contrast is a paired test on the endpoint's block values per seed (``stats.paired_test``), decided over
  the seeds by ``stats.seed_call``: the label comes from the **two-sided** 95% t-intervals of
  ``w2.classify`` (every seed's interval excludes 0 with one sign, the mean beyond the seed-twin noise floor;
  the sign must be the pre-registered one, else ``reversed``); the **one-sided** p-values of the seeds feed
  only
  Stouffer's combination and Holm;
* Holm over the **confirmatory** endpoints within the feature, per series (exploratory endpoints, P2 and P3
  under O8, are reported and take no part; an endpoint that could not be evaluated counts as p = 1, so the
  family does not shrink). Holm is per feature: the five features are not one family, so the family-wise rate
  over all five is not controlled by construction. The null still holds in the simulations (0 of 2000)
  because a call needs three series and three seeds to agree; BH over the features is context;
* the feature is ``supported`` iff some confirmatory endpoint is supported (seed rule and Holm p <= 0.05) in
  all three series, there are at least ``MIN_SEEDS`` seeds with a finite seed-twin noise floor, and its
  scorecard is not worse than its inert twin's. A scorecard that was not evaluated, too few seeds or no
  twins stop the call at ``mixed`` with the reason. ``reversed``: an endpoint significantly lower with the
  feature in all three series. ``mixed`` otherwise when a confirmatory contrast is supported or reversed over
  the seeds; else ``not supported``.

Cumulative ``U5 - U0`` and the interaction ``(U5 - L_k) - (U_k - U_(k-1))`` are reported with intervals, not
tested. Benjamini-Hochberg at 0.10 over the features' smallest Holm-adjusted p is context only.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from alife.analysis.emergence import stats
from alife.analysis.emergence.panel import check_same_version
from alife.analysis.emergence.prereg import PreregError

SERIES: Final = ("buildup", "loo", "control")
MIN_SEEDS: Final = 3  # fewer seeds, or no finite seed-twin noise floor, cannot support a call
CAVEATS: Final = (
    "Three seeds forked from one evolved population test post-fork stochasticity, not independent "
    "evolutionary histories: a claim is 'consistent across these seeds from this source'.",
    "Blocks are autocorrelated even after widening; the intervals are indicative.",
    "Leave-one-out arms are fresh runs from the shared evolved sources, not edits of the full-stack run.",
    "Thresholds and analysis constants are provisional until the calibration step freezes them.",
)


class CompareError(ValueError):
    """The panels cannot be compared (mixed versions, an unknown arm layout)."""


def arm_names(feature: str, order: Sequence[str]) -> dict[str, str]:
    """``{"u": U_k, "prev": U_(k-1), "top": U_n, "loo": L_k, "inert": I_k}`` for ``feature``."""
    i = list(order).index(feature) + 1
    return {
        "u": f"U{i}",
        "prev": f"U{i - 1}",
        "top": f"U{len(order)}",
        "loo": f"L_{feature}",
        "inert": f"I_{feature}",
    }


class _Index:
    def __init__(self, panels: Sequence[Mapping[str, Any]]) -> None:
        self.by: dict[tuple[str, Any], Mapping[str, Any]] = {}
        for p in panels:
            key = (p["header"]["arm"], p["header"]["seed"])
            if key in self.by:
                raise CompareError(f"two panels for arm {key[0]!r} seed {key[1]!r}")
            self.by[key] = p
        self.seeds = sorted({s for _, s in self.by}, key=str)
        self.missing: set[str] = set()

    def blocks(self, arm: str, seed: Any, ep: str) -> np.ndarray | None:
        p = self.by.get((arm, seed))
        if p is None:
            self.missing.add(f"panel {arm} seed {seed}")
            return None
        e = p["endpoints"].get(ep)
        if e is None or e.get("blocks") is None or e.get("status") != "ok":
            self.missing.add(f"{ep} of {arm} seed {seed}")
            return None
        return np.asarray([np.nan if b is None else b for b in e["blocks"]], dtype=np.float64)

    def diff(self, a: str, b: str, seed: Any, ep: str, sign: int = 1) -> np.ndarray | None:
        x, y = self.blocks(a, seed, ep), self.blocks(b, seed, ep)
        if x is None or y is None:
            return None
        if x.shape != y.shape:
            raise CompareError(f"{ep} seed {seed}: {a} has {x.shape[0]} blocks, {b} has {y.shape[0]}")
        return sign * (x - y)


def _floor(ix: _Index, ep: str, top: str) -> float | None:
    """Seed-twin noise floor: U0 vs T0 and U_top vs T_top per seed."""
    diffs = []
    for seed in ix.seeds:
        for u, t in (("U0", "T0"), (top, f"T{top.removeprefix('U')}")):
            d = ix.diff(u, t, seed, ep)
            if d is not None:
                diffs.append(d)
    return stats.noise_floor(diffs)


def _series(ix: _Index, a: str, b: str, ep: str, sign: int, floor: float | None) -> dict[str, Any]:
    per = []
    for seed in ix.seeds:
        x, y = ix.blocks(a, seed, ep), ix.blocks(b, seed, ep)
        if x is None or y is None:
            per.append({"seed": seed, "status": "needs_missing"})
            continue
        per.append({"seed": seed, "status": "ok", **stats.paired_test(x, y, direction=sign)})
    ok = [p for p in per if p["status"] == "ok"]
    if len(ok) < len(per) or len(ok) < 2:
        return {
            "label": "not evaluated",
            "p": None,
            "floor": floor,
            "mean": None,
            "seeds": len(ok),
            "per_seed": per,
        }
    call = stats.seed_call(ok, floor)
    return {**call, "per_seed": per}


def _scorecard_ok(ix: _Index, feature: str, top: str, inert: str) -> tuple[bool | None, str]:
    res = []
    for seed in ix.seeds:
        a, b = ix.by.get((top, seed)), ix.by.get((inert, seed))
        if a is None or b is None:
            return None, f"missing panel for the scorecard comparison (seed {seed})"
        ea, eb = a["endpoints"].get(f"S_{feature}"), b["endpoints"].get(f"S_{feature}")
        va = None if ea is None else ea.get("value")
        vb = None if eb is None else eb.get("value")
        if va is None or vb is None:
            return None, f"scorecard {feature} not evaluated"
        res.append(va >= vb)
    if all(res):
        return True, "primary of the full stack at least the inert twin's in every seed"
    return False, "worse than the inert twin in some seed"


def compare_panels(
    panels: Sequence[Mapping[str, Any]],
    prereg: Mapping[str, Any],
    *,
    exploratory: bool = False,
    prereg_path_hash: str | None = None,
) -> dict[str, Any]:
    """The comparison of ``panels`` under ``prereg`` (see the module docstring).

    ``prereg_path_hash`` is the hash of the frozen file as read (``prereg.load_prereg``'s ``sha256``). Unless
    ``exploratory``, every panel's ``prereg_hash`` must equal it and the file must be ``frozen``; otherwise
    ``PreregError`` (the suite refuses to label ``supported`` on an unregistered analysis). With
    ``exploratory`` the numbers are computed and marked ``exploratory``.
    """
    check_same_version(panels)
    frozen_hash = prereg_path_hash or prereg.get("sha256")
    bad = sorted(
        {str(p["header"]["prereg_hash"]) for p in panels if p["header"]["prereg_hash"] != frozen_hash}
    )
    if not exploratory:
        if not prereg.get("frozen"):
            raise PreregError("the pre-registration is not frozen; use --exploratory to print the numbers")
        if bad:
            raise PreregError(f"panel prereg hashes {bad} differ from the frozen file's {frozen_hash}")
    ix = _Index(panels)
    order = list(prereg["buildup_order"])
    eps = prereg["endpoints"]
    conf = [e for e, m in eps.items() if m.get("confirmatory")]
    alpha = float(prereg["decision"]["alpha"])
    features: dict[str, Any] = {}
    for f in order:
        an = arm_names(f, order)
        table: dict[str, Any] = {}
        for e, meta in eps.items():
            sg, fl = int(meta.get("direction", 1)), _floor(ix, e, an["top"])
            table[e] = {
                "confirmatory": bool(meta.get("confirmatory")),
                "floor": fl,
                "series": {
                    "buildup": _series(ix, an["u"], an["prev"], e, sg, fl),
                    "loo": _series(ix, an["top"], an["loo"], e, sg, fl),
                    "control": _series(ix, an["top"], an["inert"], e, sg, fl),
                },
                "cumulative": _cumulative(ix, an["top"], e, sg),
                "interaction": _interaction(ix, an, e, sg),
            }
        for s in SERIES:
            ps = [table[e]["series"][s]["p"] for e in conf]
            ha = stats.holm([1.0 if p is None else p for p in ps]) if ps else []
            for e, a in zip(conf, ha, strict=True):
                table[e]["series"][s]["p_holm"] = None if not np.isfinite(a) else a
        for e in conf:
            ok = [table[e]["series"][s] for s in SERIES]
            table[e]["supported"] = all(
                c["label"] == "supported" and c.get("p_holm") is not None and c["p_holm"] <= alpha for c in ok
            )
            hp = [c.get("p_holm") for c in ok]
            table[e]["p_holm_max"] = None if any(h is None for h in hp) else max(hp)  # type: ignore[type-var]
        a_ok = [e for e in conf if table[e]["supported"]]
        n_seeds = len(ix.seeds)
        blocked = [e for e in a_ok if n_seeds < MIN_SEEDS or table[e]["floor"] is None]
        a_ok = [e for e in a_ok if e not in blocked]
        rev = [e for e in conf if all(table[e]["series"][s]["label"] == "reversed" for s in SERIES)]
        sc_ok, sc_note = _scorecard_ok(ix, f, an["top"], an["inert"])
        evidence = any(
            table[e]["series"][s]["label"] in ("supported", "reversed") for e in conf for s in SERIES
        )
        if a_ok and sc_ok:
            call, why = "supported", f"supported on {', '.join(a_ok)} in all three series; {sc_note}"
        elif a_ok:
            call, why = (
                "mixed",
                f"endpoints {', '.join(a_ok)} supported in all series, but scorecard: {sc_note}",
            )
        elif blocked:
            call = "mixed"
            why = (
                f"{', '.join(blocked)} would be supported, but a call needs at least {MIN_SEEDS} seeds "
                f"(have {n_seeds}) and a finite seed-twin noise floor (T0/T5 panels)"
            )
        elif rev:
            call, why = (
                "reversed",
                f"{', '.join(rev)} significantly lower with the feature in all three series",
            )
        elif evidence:
            call, why = (
                "mixed",
                "a contrast is supported over the seeds, but not in all three series on one endpoint",
            )
        else:
            call, why = "not supported", "no confirmatory contrast shows evidence"
        pmin = [table[e]["p_holm_max"] for e in conf if table[e]["p_holm_max"] is not None]
        features[f] = {
            "arms": an,
            "call": call,
            "reason": why,
            "exploratory": bool(exploratory),
            "p_min_holm": min(pmin) if pmin else None,
            "endpoints": table,
        }
    names = [f for f in order if features[f]["p_min_holm"] is not None]
    q = stats.bh([features[f]["p_min_holm"] for f in names])
    q_by = dict(zip(names, q, strict=True))
    for f in order:
        features[f]["bh_q"] = q_by.get(f)
    return {
        "version": 1,
        "prereg": {
            "sha256": frozen_hash,
            "frozen": bool(prereg.get("frozen")),
            "exploratory": bool(exploratory),
            "panel_hashes_differ": bad,
        },
        "order": order,
        "seeds": ix.seeds,
        "confirmatory": conf,
        "alpha": alpha,
        "features": features,
        "bh_q_level": float(prereg["decision"]["bh_q"]),
        "missing": sorted(ix.missing),
        "not_claimed": list(CAVEATS),
    }


def _cumulative(ix: _Index, top: str, ep: str, sg: int) -> dict[str, Any]:
    """``U_top - U0`` per seed: mean and t-interval (reported, not tested)."""
    out = []
    for seed in ix.seeds:
        d = ix.diff(top, "U0", seed, ep, sg)
        out.append(None if d is None else stats.t_interval(d))
    return {"per_seed": out}


def _interaction(ix: _Index, an: Mapping[str, str], ep: str, sg: int) -> dict[str, Any]:
    out = []
    for seed in ix.seeds:
        a = ix.diff(an["top"], an["loo"], seed, ep, sg)
        b = ix.diff(an["u"], an["prev"], seed, ep, sg)
        out.append(None if a is None or b is None else stats.t_interval(a - b))
    return {"per_seed": out}


def _cell(c: Mapping[str, Any]) -> str:
    if c["label"] == "not evaluated":
        return "not evaluated"
    p = "-" if c["p"] is None else f"{c['p']:.3g}"
    h = c.get("p_holm")
    return f"{c['label']} (mean {c['mean']:.3g}, p {p}" + (f", Holm {h:.3g})" if h is not None else ")")


def to_markdown(cmp: Mapping[str, Any]) -> str:
    pr = cmp["prereg"]
    L = ["# Arm comparison", ""]
    tag = " EXPLORATORY (not a registered analysis)" if pr["exploratory"] else ""
    L += [f"- prereg `{pr['sha256']}` frozen={pr['frozen']}{tag}"]
    L += [f"- seeds {cmp['seeds']}, confirmatory endpoints {cmp['confirmatory']}, alpha {cmp['alpha']}", ""]
    L += ["## Calls", "", "| feature | call | BH q | reason |", "|---|---|---|---|"]
    for f in cmp["order"]:
        x = cmp["features"][f]
        q = "-" if x["bh_q"] is None else f"{x['bh_q']:.3g}"
        L.append(f"| {f} | {x['call']}{' (exploratory)' if x['exploratory'] else ''} | {q} | {x['reason']} |")
    for f in cmp["order"]:
        L += ["", f"## {f}", "", "| endpoint | floor | buildup | loo | control |", "|---|---|---|---|---|"]
        for e, t in cmp["features"][f]["endpoints"].items():
            fl = "-" if t["floor"] is None else f"{t['floor']:.3g}"
            tag = "" if t["confirmatory"] else " (exploratory)"
            L.append(f"| {e}{tag} | {fl} | " + " | ".join(_cell(t["series"][s]) for s in SERIES) + " |")
    if cmp["missing"]:
        L += ["", "## Missing", ""] + [f"- {m}" for m in cmp["missing"][:50]]
    L += ["", "## Not claimed", ""] + [f"- {s}" for s in cmp["not_claimed"]]
    return "\n".join(L) + "\n"


def write_compare(out_dir: Path, cmp: Mapping[str, Any]) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, text in (
        ("compare.json", json.dumps(cmp, indent=2, default=_json) + "\n"),
        ("compare.md", to_markdown(cmp)),
    ):
        p = out_dir / name
        tmp = out_dir / f".{name}.tmp{os.getpid()}"
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, p)
        paths.append(p)
    return paths[0], paths[1]


def _json(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))
