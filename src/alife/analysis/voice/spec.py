"""Constants and the pre-registration of the voice analysis (DESIGN §39.14).

Everything a verdict depends on is here, so it can be frozen (``meaning prereg --freeze``) and hashed before
the arms are read: the context and response variables, the thresholds, the family of tests, the fitness
readouts and the blocks. ``default_prereg`` is the file's content; ``emergence.prereg`` supplies the canonical
hash and the atomic write.
"""

from __future__ import annotations

from typing import Any, Final

from alife.analysis.voice import VOICE_ANALYSIS_VERSION

# --- sampling -------------------------------------------------------------------------------------------
SAMPLE_EVERY: Final = 50  # a sample is taken every SAMPLE_EVERY steps (three consecutive observed steps)
SAMPLE_ROWS: Final = 2048  # rows drawn per sample, for senders and for hearers
WINDOW: Final = 10_000  # the default recorded window, in steps
N_BLOCKS: Final = 8  # equal blocks of the window (the suite's window blocking)
SEED: Final = 0  # of the row draws and the permutations

# --- the variables --------------------------------------------------------------------------------------
# Sender (and hearer) context, from the start-of-step state: name -> (levels, definition).
CONTEXTS: Final[dict[str, tuple[int, str]]] = {
    "hurt": (2, "energy was taken by an attack in the previous step (last_loss > 0)"),
    "food": (3, "resource at the agent's field cell, terciles (fewer levels if zero-inflated)"),
    "energy": (3, "energy, terciles of the pooled sample"),
    "crowd": (3, "number of sensed neighbours, terciles of the pooled sample"),
    "bonded": (2, "has at least one bond partner"),
    "height": (3, "height above the floor (3D worlds), terciles; constant in 2D"),
    "wind": (3, "wind drift speed (worlds with weather), terciles; constant without weather"),
    "budding": (2, "is a parent of a birth lane in this step (about to reproduce)"),
}
# Receiver response, in the step in which the call is heard.
RESPONSES: Final[dict[str, tuple[int, str]]] = {
    "move": (3, "|thrust|, terciles of the pooled sample"),
    "attack": (2, "attacked (ActFlag.ATTACKED)"),
    "bond": (2, "formed a bond (ActFlag.FORMED)"),
    "eat": (2, "harvested food in the step (last_harvest > 0 at the next state)"),
    "speak": (2, "spoke (ActFlag.SPOKE)"),
    "loud": (3, "total call intensity, terciles of the pooled sample (silent is the lowest level)"),
    "approach": (3, "toward (+) or away from (-) the loudest audible source; 0 also without a source"),
}
DIST_BINS: Final = 4  # distance quartiles of a source (as a share of the hearing range)

# --- tests ----------------------------------------------------------------------------------------------
N_PERM: Final = 200  # permutations of a pooled test (the smallest p is 1/201)
N_PERM_BLOCK: Final = 20  # permutations of a block (for the null mean only)
MIN_STRATUM: Final = 30  # a stratum with fewer rows takes no part in a conditional measure
TOP_LINEAGES: Final = 8  # lineages reported one by one (the rest are pooled as "other")
LINEAGE_STRATA: Final = 16  # lineages kept as strata of the within-lineage code test

# Thresholds (provisional until the calibration of the suite, DESIGN §44.8; recorded in every report).
THRESH: Final[dict[str, dict[str, float]]] = {
    "code": {"z": 4.0, "bits": 0.02, "p": 0.01},  # I(call; caller's context)
    "transfer": {"z": 4.0, "bits": 0.05, "p": 0.01},  # I(source context; receiver response | ...): the spec's
    "reception": {"z": 4.0, "bits": 0.01, "p": 0.01},  # I(call heard; response | own context)
}
ALPHA: Final = 0.05
COST_TOLERANCE: Final = 0.25  # relative difference of the voice cost per agent-step between arms

# --- the arms -------------------------------------------------------------------------------------------
ARMS: Final = ("none", "deaf", "decoy")
SEEDS: Final = (1, 2, 3)
FORK_STEPS: Final = 30_000
TABLE_WINDOW: Final = (20_000, 30_000)  # steps after the fork that the tables record
FITNESS_BLOCKS: Final = 12  # equal blocks of the fork's steps (0, FORK_STEPS]
FITNESS: Final[dict[str, str]] = {
    "population": "mean number alive (n_alive)",
    "births": "births per agent-step (births / (steps * n_alive))",
    "energy": "mean energy per agent (energy_mean)",
}
VERDICTS: Final = ("language", "signalling without meaning", "no evidence")


def default_prereg(code_git: str | None = None) -> dict[str, Any]:
    """The pre-registration of the meaning test, unfrozen. ``meaning prereg`` writes it (``--freeze`` marks
    it frozen); every report stores its hash."""
    return {
        "version": 1,
        "kind": "voice_meaning",
        "analysis_version": VOICE_ANALYSIS_VERSION,
        "frozen": False,
        "code_git": code_git,
        "sampling": {
            "every": SAMPLE_EVERY,
            "rows": SAMPLE_ROWS,
            "window": WINDOW,
            "blocks": N_BLOCKS,
            "seed": SEED,
        },
        "contexts": {k: v[0] for k, v in CONTEXTS.items()},
        "responses": {k: v[0] for k, v in RESPONSES.items()},
        "dist_bins": DIST_BINS,
        "tests": {
            "n_perm": N_PERM,
            "n_perm_block": N_PERM_BLOCK,
            "min_stratum": MIN_STRATUM,
            "family": "max over the (context, response) pairs of the permutation z; one permutation is "
            "applied to all pairs, so the p-value is for the family",
        },
        "thresholds": {k: dict(v) for k, v in THRESH.items()},
        "arms": {
            "names": list(ARMS),
            "seeds": list(SEEDS),
            "fork_steps": FORK_STEPS,
            "table_window": list(TABLE_WINDOW),
            "controls": "comm.control=none|deaf|decoy from one checkpoint; cost is paid in every arm",
        },
        "fitness": {
            "readouts": dict(FITNESS),
            "blocks": FITNESS_BLOCKS,
            "direction": "none minus control, positive = hearing helps",
            "holm": "over the readouts, within each contrast (none-deaf, none-decoy); the clause needs one "
            "readout that passes in both contrasts",
            "seed_rule": "w2.classify over seeds on the paired block intervals, Stouffer p across seeds",
        },
        "cost": {"tolerance": COST_TOLERANCE, "readout": "comm_due per agent-step"},
        "decision": {
            "alpha": ALPHA,
            "language": "code AND transfer (none) AND transfer(none) > transfer(deaf) and > transfer(decoy) "
            "(paired block intervals, Holm over the two contrasts) AND reception AND fitness clause AND "
            "cost matched",
            "signalling without meaning": "the calls carry information (code, transfer or reception "
            "passes) but the language clauses do not all hold; the detail says which clause failed",
            "no evidence": "no information clause passes",
        },
    }
