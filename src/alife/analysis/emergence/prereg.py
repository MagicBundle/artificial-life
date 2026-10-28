"""The pre-registration file and its hash (DESIGN §44.8.5).

``prereg.json`` fixes what the arm comparison will test before any feature arm is read: the contrasts and
their direction, the endpoint definitions and which of them are confirmatory, the build-up order, the decision
constants and the (calibrated) analysis constants. Its sha256 (over the canonical JSON of everything except
the ``sha256`` key) is stored in every later panel; ``compare`` refuses to label ``supported`` when a panel's
hash differs from the frozen file's, or when the file is not frozen (``--exploratory`` prints the same numbers
marked exploratory).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Final

PREREG_VERSION: Final = 1
HASH_KEY: Final = "sha256"

# O2 (owner): the build-up order. Feature ids as in scorecards.FEATURES (farming = niche).
BUILDUP_ORDER: Final = ("arms", "structures", "bonds", "comm", "niche")

ENDPOINTS: Final[dict[str, dict[str, Any]]] = {
    "P1": {"block": "A", "name": "A^N_new", "direction": 1, "confirmatory": True,
           "definition": "Channon normalised new activity (Bedau-Packard activity classes), per block"},
    "P2": {"block": "A", "name": "MODES novelty rate", "direction": 1, "confirmatory": False,
           "definition": "cumulative new phenotype classes per block (adapted MODES; exploratory, O8)"},
    "P3": {"block": "A", "name": "MODES complexity", "direction": 1, "confirmatory": False,
           "definition": "p90 channel count of the phenotype skeletons (adapted MODES; exploratory, O8)"},
    "P4": {"block": "B", "name": "own-motion EP", "direction": 1, "confirmatory": True,
           "definition": "entropy production of own motion, nats per organism per 1000 steps (lower bound)"},
    "P5": {"block": "C", "name": "braid coordination contrast", "direction": 1, "confirmatory": True,
           "definition": "h_real - h_surr over bodies >= 8, per crossing (coordination index, not a score)"},
}  # fmt: skip

SCORECARD_PRIMARY: Final = {
    "arms": "escalation_index",
    "bonds": "verdict_level",
    "comm": "transfer_bits",
    "niche": "max_layer_persist",
    "structures": "builder_fixation_minus_zero_column_control",
}


class PreregError(ValueError):
    """The pre-registration is unreadable, inconsistent with its stored hash, or does not match a panel."""


def default_prereg(order: tuple[str, ...] = BUILDUP_ORDER, code_git: str | None = None) -> dict[str, Any]:
    """A fresh, **unfrozen** pre-registration with the §44 defaults. The analysis constants below are the
    provisional ones the calibration step (§8.5) freezes."""
    return {
        "version": PREREG_VERSION,
        "frozen": False,
        "code_git": code_git,
        "panel_version": 1,
        "buildup_order": list(order),
        "endpoints": {k: dict(v) for k, v in ENDPOINTS.items()},
        "scorecard_primary": dict(SCORECARD_PRIMARY),
        "contrasts": {
            "buildup": "U_k - U_(k-1)",
            "loo": "U5 - L_k",
            "control": "U5 - I_k",
            "reported_not_tested": ["U5 - U0", "interaction (U5 - L_k) - (U_k - U_(k-1))"],
        },
        "decision": {
            "alpha": 0.05,
            "bh_q": 0.10,
            "holm_within": "feature, per series, over the confirmatory endpoints",
            "supported": "a confirmatory endpoint supported in buildup, loo and control (same sign, Holm), "
            "and the scorecard not worse than the inert twin",
            "seed_rule": "w2.classify over seeds, Stouffer p across seeds",
            "blocks": {
                "window": 16,
                "run": 8,
            },  # 16 x 625 for a 10 000-step window (8 gives ~0.56 power at d = 1.5)
            "min_seeds": 3,
            "widen_at_abs_r1": 0.3,
        },
        "analysis": {  # provisional until the calibration run (E7)
            "ep_tau": 100,
            "ep_cell": 8,
            "stationarity_threshold": None,
            "modes_codebook": None,
            "braid_window": None,
            "fine_tau": 5,
            "noise_floors": None,
        },
    }


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def prereg_hash(prereg: dict[str, Any]) -> str:
    """sha256 over the canonical JSON of the pre-registration without its ``sha256`` key."""
    body = {k: v for k, v in prereg.items() if k != HASH_KEY}
    return hashlib.sha256(canonical(body)).hexdigest()


def write_prereg(path: Path, prereg: dict[str, Any]) -> str:
    """Write ``prereg`` with its hash (atomic); returns the hash."""
    h = prereg_hash(prereg)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps({**prereg, HASH_KEY: h}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return h


def load_prereg(path: Path) -> dict[str, Any]:
    """Read a prereg file and check its stored hash against its content (``PreregError`` on a difference)."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise PreregError(f"{path}: cannot read the pre-registration ({e})") from e
    if d.get("version") != PREREG_VERSION:
        raise PreregError(f"{path}: prereg version {d.get('version')}, this code reads {PREREG_VERSION}")
    if d.get(HASH_KEY) != prereg_hash(d):
        raise PreregError(f"{path}: the stored sha256 does not match the content (the file was edited)")
    return d
