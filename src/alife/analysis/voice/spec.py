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
N_BLOCKS: Final = 16  # equal blocks of the window (the suite's window blocking, §44.8)
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
    "eat": (3, "energy harvested in the step (last_harvest of the next state), terciles"),
    "speak": (2, "spoke (ActFlag.SPOKE)"),
    "loud": (3, "total call intensity, terciles of the pooled sample (silent is the lowest level)"),
    "approach": (3, "toward (+) or away from (-) the loudest audible source; 0 also without a source"),
}
DIST_BINS: Final = 4  # distance quartiles of a source (as a share of the hearing range)

# --- tests ----------------------------------------------------------------------------------------------
N_PERM: Final = 200  # permutations of a pooled test (the smallest p is 1/201)
N_PERM_BLOCK: Final = 20  # permutations of a block (for the null mean only)
MIN_STRATUM: Final = 30  # a stratum with fewer rows takes no part in a conditional measure
TOP_LINEAGES: Final = 8  # tag clusters reported one by one
LINEAGE_GROUPS: Final = (
    8  # tag clusters (k-means of the kin tags at the window start) that birth carries along
)
LINEAGE_STRATA: Final = 16  # tag clusters kept as strata of the within-cluster code test

# Decision constants of the rule version 3 (DESIGN §39.14; chosen after the v2 results were read, see the
# amendments in the pre-registration). Bits are the excess over the shuffle null, odd blocks, per seed.
MIN_EFFECT: Final = (
    0.005  # a call-specific transfer (none minus decoy) at least this is a meaning, not a trace
)
EQUIVALENCE: Final = (
    0.005  # the one-sided 95% upper bound below this: no detectable meaning (TOST, alpha 0.05)
)
USE_THRUST: Final = (
    0.01  # counterfactual (none to decoy): mean |thrust| change of audible rows, share of full thrust
)
SPEAKING_MIN: Final = 0.05  # share of sampled rows that speak, below which there is no signalling to speak of
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
VERDICTS: Final = (
    "language",
    "signalling, no detectable meaning",
    "signalling with trace meaning",
    "no evidence",
)
RULES_VERSION: Final = 3


AMENDMENTS: Final[list[dict[str, Any]]] = [
    {
        "id": "A1",
        "date": "2026-10-30",
        "from": "ec55e292012e83bd37e291382c1fcf01d4005c78115d700dcad8f56e3dce5227 (v1, code_git 9726159)",
        "to": "3804eebebfc93dde14117f87ba03f8985ce33ac27385c87d14dad01f04ff4549 (v2, code_git 85f0a7e)",
        "what": "the `eat` response coded in three levels (last_harvest > 0 was true for every row) and the "
        "lineage rule (k-means of the tags, inherited by birth) written into the file",
        "reason": "both were fixed on the source world's probe, before any arm table or result existed",
        "results_read": False,
        "recorded": "retroactively, in v3: the v2 file does not contain this amendment",
    },
    {
        "id": "A2",
        "date": "2026-10-31",
        "from": "3804eebebfc93dde14117f87ba03f8985ce33ac27385c87d14dad01f04ff4549 (v2)",
        "to": "this file (rules_version 3)",
        "what": "the decision rule: the absolute transfer gate (0.05 bits in the none arm) is replaced by a "
        "contrast-based minimum effect (none minus decoy, 0.005 bits) and an equivalence bound (one-sided "
        "95% upper bound below 0.005 bits); the contrast pair is selected on the even blocks by the contrast "
        "itself; `code` and `reception` leave the decision (reception is stratified on the audible count "
        "and the distance bin; the counterfactual is the use clause); a seed-twin noise floor enters the "
        "seed rule; the labels are renamed; the sampling parameters actually used are registered and checked",
        "reason": "independent review of 1be5a33: the absolute gate could not return `language` for any "
        "realistic effect (H1), the pair was chosen for the largest whole transfer, which is the shared "
        "environment (H2), `code` is sender-side and `reception` is confounded by geometry (M1), the "
        "seeds are near-replicates (M2)",
        "results_read": True,
        "results_read_detail": "the v2 verdict (signalling without meaning, both worlds), its numbers and "
        "the reviewer's post hoc 95% bound on the call-specific transfer (about 0.0034 bits) were known when "
        "the constants of this file were chosen; the minimum effect and the equivalence bound (0.005 bits, "
        "about 7% of the sender-side code of 0.07 bits) are therefore not blind. v3 is a revised rule, "
        "decided after seeing the v2 results.",
        "v2_verdict_kept": "meaning_64k3s_v2, meaning_allfeat_v2, meaning_64k3s_early_v2",
    },
]


def default_prereg(code_git: str | None = None) -> dict[str, Any]:
    """The pre-registration of the meaning test (rules version 3), unfrozen. ``meaning prereg`` writes it
    (``--freeze`` marks it frozen); every report stores its hash."""
    return {
        "version": 1,
        "kind": "voice_meaning",
        "rules_version": RULES_VERSION,
        "analysis_version": VOICE_ANALYSIS_VERSION,
        "frozen": False,
        "code_git": code_git,
        "amendments": [dict(a) for a in AMENDMENTS],
        "sampling": {
            "every": SAMPLE_EVERY,
            "rows": SAMPLE_ROWS,
            "window": WINDOW,
            "blocks": N_BLOCKS,
            "seed": SEED,
        },
        "lineage": {
            "groups": LINEAGE_GROUPS,
            "rule": "k-means of the kin tags of the rows alive at the window start; a birth inherits its "
            "parent's cluster (immigrants are unassigned); a descriptive stratum, not a lineage",
        },
        "contexts": {k: v[0] for k, v in CONTEXTS.items()},
        "responses": {k: v[0] for k, v in RESPONSES.items()},
        "dist_bins": DIST_BINS,
        "tests": {
            "n_perm": N_PERM,
            "n_perm_block": N_PERM_BLOCK,
            "min_stratum": MIN_STRATUM,
            "family": "max over the (context, response) pairs of the permutation z; the pairs share the "
            "random keys of each permutation, so the p-value is for the family. Reported, not a gate: the "
            "rows are not exchangeable (repeated animals), every pooled z is far above any threshold",
            "reception_strata": "own crowd, hurt, bonded, audible count, distance bin of the source, quarter "
            "of the window",
            "pair_selection": "the transfer pair with the largest none-minus-decoy contrast of the raw "
            "Miller-Madow information, averaged over the seeds and the even blocks; estimated on the odd "
            "blocks as mi minus the null mean (20 permutations)",
            "cut_points": "every arm of a seed is coded with the none arm's cut points",
        },
        "contrast": {
            "primary": "none minus decoy (the content cut; the presence and the loudness are kept)",
            "secondary": "none minus deaf (contaminated by the knock-out of the evolved inputs)",
            "min_effect_bits": MIN_EFFECT,
            "equivalence_bits": EQUIVALENCE,
            "alpha_one_sided": ALPHA,
            "unit": "the seed (one evolved checkpoint per world): the mean of each seed's odd blocks, the "
            "one-sided t bound over the seeds with df = seeds - 1",
            "noise_floor": "median |mean| of the same-arm seed-twin differences of the odd-block values "
            "(stats.noise_floor), applied to the seed rule",
        },
        "use": {
            "counterfactual": "none to decoy, one step from the same state: the mean |thrust| change of the "
            "audible rows, every seed, at least use_thrust; the rows with no audible neighbour must change "
            "by exactly 0",
            "use_thrust": USE_THRUST,
        },
        "speaking_min": SPEAKING_MIN,
        "arms": {
            "names": list(ARMS),
            "seeds": list(SEEDS),
            "fork_steps": FORK_STEPS,
            "table_window": list(TABLE_WINDOW),
            "controls": "comm.control=none|deaf|decoy from one checkpoint; cost is paid in every arm",
            "caveat": "the deaf arm knocks out inputs the evolved brains are tuned to: a fitness gap against "
            "deaf is the cost of removing the heard inputs plus any value of information; only the decoy "
            "contrast separates the content",
        },
        "fitness": {
            "readouts": dict(FITNESS),
            "blocks": FITNESS_BLOCKS,
            "direction": "none minus control, positive = hearing helps",
            "holm": "over the readouts, within each contrast (none-deaf, none-decoy); the clause needs one "
            "readout that passes in both contrasts",
            "seed_rule": "w2.classify over seeds on the paired block intervals with the seed-twin noise "
            "floor, Stouffer p across seeds",
        },
        "cost": {"tolerance": COST_TOLERANCE, "readout": "comm_due per agent-step"},
        "decision": {
            "alpha": ALPHA,
            "language": "speaking AND use AND the primary contrast positive (seed rule with the floor and a "
            "one-sided lower bound above 0) with mean >= min_effect AND the secondary contrast positive AND "
            "the fitness clause AND cost matched",
            "signalling with trace meaning": "speaking, and the primary contrast positive but language not "
            "established (mean below min_effect, or another clause failed; the detail says which)",
            "signalling, no detectable meaning": "speaking, the primary contrast not positive and its "
            "one-sided 95% upper bound below equivalence_bits; the label carries the bound",
            "no evidence": "nobody speaks, or the primary contrast is neither positive nor bounded below "
            "equivalence_bits (not enough to say)",
        },
    }
