"""The blacklist of the world report's second line (docs/DESIGN.md §31, R6): words and phrases that no report
sentence may contain, whatever the template.

The report's sentences are all written by code from a closed set of templates (``game.templates``); this
module
is the independent check that none of them says more than the facts do. It lists, as data:

- ``V1`` causal and connective words, ``V2`` presupposition (success, failure, necessity), ``V3`` inference
  and hedges, ``V4`` motive, intent and relation words, ``V5`` persons, ``V6`` forecast and modal words,
  ``V7`` taxonomy (species, tribe, evolution), ``V8`` evaluative and superlative words, ``T2`` spelled-out
  numbers and fractions, ``G`` magnitude and dominance words: *never*, in any report string, with display
  names masked.
- ``L1`` lever words (regrowth, patch, immigrant, catastrophe, ...): only in the interventions block and the
  ticker (and the engine's own death cause "catastrophe" in D-2).
- ``L2`` label names (Hunter, Resting, ...): only in the label sentences (LB-a, LB-i), through the label slot.
  They include the storm world's display words (Grazer, Drifter, Floater, Sinker;
  ``game.rules.ARCHETYPE_STORM_DISPLAY``, DESIGN §32.7). The L1 stem of the patch-drift lever is ``drift``
  without "drifter(s)", which is a label name.
- ``L3`` storm words (storm, disturbance, hemisphere, ...): only in the storm sentences of the report (the
  ``ST-*`` templates, the storm headline and their ticker lines; DESIGN §34.4), so a sentence about a lineage
  or the population never talks about the weather.

A term is a word (``"because"``), a phrase (``"as a result"``), a stem ending in ``-`` (``"caus-"``: the stem
and any letters after it, so ``"cause"``, ``"causes"`` and ``"causing"`` all match; an inflection test in
``tests/test_game_report_templates.py`` pins that stems match their inflections and no word they should
not), or ``"re:<regex>"``. Matching is on lower-cased text, at word boundaries.

Every list is data: ``LEXICON_DIGEST`` pins them (a record stores it) and the corpus regression
(``tests/data/report_corpus.json``) pins what they reject.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Final

__all__ = [
    "LABEL_LISTS",
    "LEVER_LISTS",
    "LEXICON",
    "LEXICON_DIGEST",
    "LEXICON_VERSION",
    "NEVER_LISTS",
    "STORM_LISTS",
    "compile_terms",
    "scan",
]

# 2: the storm world's display words joined L2 and the L1 stem "drift" stopped matching "drifter" (W2a,
# DESIGN §32.13).
# 3: the storm words joined as L3 (allowed in the storm sentences only), and V8's "last" no longer matches
# "did not last" (the sentence on the kin group changes too short to list; G3b, DESIGN §34.4).
LEXICON_VERSION: Final = 3

LEXICON: Final[Mapping[str, tuple[str, ...]]] = {
    "T2": (
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
        "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
        "hundred-", "thousand-", "million-", "half", "halved", "halves", "double-", "twice",
        "triple-", "re:[a-z]*fold", "dozen-", "percent-", "per cent", "single", "sole",
    ),
    "L1": (
        "regen-", "regrowth", "pulse-", "immigra-", "founder-", "catastroph-", "re:drift(?!ers?\\b)[a-z]*",
        "season-", "fertil-", "patch-", "habitat-", "oasis", "oases", "barren", "interven-", "director-",
        "command-", "damper-", "lever-", "cooldown-", "budget-", "boom-", "famine-", "drought-",
        "bloom-", "introduc-",
    ),
    "L2": (
        "gatherer-", "grazer-", "hunter-", "hunting", "provider-", "trader-", "nomad-", "settler-",
        "drifter-", "floater-", "sinker-", "generalist-", "re:unspeciali[sz]ed", "budding", "fleeing",
        "defending", "feeding", "caring", "bonding", "tethering", "re:signall?ing", "roaming", "resting",
        "re:hunt(?:s|ed)?", "re:fle(?:e|es|d)", "re:defend(?:s|ed)?", "re:roam(?:s|ed)?",
        "re:tether(?:s|ed)?",
    ),
    "V1": (
        "because", "caus-", "due to", "owing to", "thanks to", "as a result", "result-", "therefore",
        "thus", "hence", "consequen-", "so that", "lead-", "led", "driv-", "drove", "trigger-",
        "spark-", "prompt-", "induc-", "forc-", "effect-", "affect-", "impact-", "influenc-",
        "contribut-", "responsib-", "fuel-", "boost-", "dampen-", "damp-", "offset-",
        "re:stabili[sz][a-z]*", "re:destabili[sz][a-z]*", "mitigat-", "prevent-", "help-", "hinder-",
        "enabl-", "allow-", "permit-", "suppress-", "promot-", "encourag-", "foster-", "stimulat-",
        "spur-", "push-", "pressur-", "outpac-", "through", "via", "in response", "respond-",
        "react-", "counter-", "combat-", "thriv-", "benefit-", "suffer-",
    ),
    "V2": (
        "despite", "in spite of", "although", "though", "yet", "nevertheless", "nonetheless",
        "regardless", "fail-", "succe-", "manag-", "unable", "finally", "requir-", "need-",
        "necessar-", "necessit-", "in order to", "re:too\\s+[a-z'-]+\\s+(?:for|to)",
        're:to\\s+(?:prevent|rescue|mitigate|stabili[sz]e|boost|offset|break|increase|reduce|raise|lower|restore|counter|spark)',
    ),
    "V3": (
        "suggest-", "indicat-", "re:impl(?:y|ies|ied|ying)", "sign of", "signal-", "threaten-",
        "evidenc-", "reflect-", "re:mean(?:s|ing)?", 're:show(?:s|ed|ing)?\\s+that', "likely",
        "unlikely", "probabl-", "perhaps", "possibl-", "potential-", "apparent-", "seem-", "appear-",
    ),
    "V4": (
        "want-", "wish-", "seek-", "sought", "desir-", "re:tr(?:y|ies|ied|ying)", "attempt-", "aim-",
        "intend-", "intent-", "plan-", "strateg-", "decid-", "choos-", "chose-", "hope-",
        "re:hop(?:ed|ing)", "fear-", "re:hat(?:e|es|ed|red|ing)", "re:lov(?:e|es|ed|ing)", "reveng-",
        "ambiti-", "greed-", "struggl-", "determin-", "eager-", "desperat-", "aggressi-", "cooperat-",
        "friend-", "enem-", "re:all(?:y|ies|ied|iance[a-z]*)", "war-", "peace-", "hostil-",
        "re:trad(?:e|es|ed|ing)", "betray-", "punish-", "protect-", "rescu-", "disrupt-", "break-",
        "shatter-", "inject-", "control-",
    ),
    "V5": (
        "i", "me", "my", "we", "us", "our", "you", "your",
    ),
    "V6": (
        "will", "would", "could", "might", "may", "shall", "should", "must", "going to", "expect-",
        "forecast-", "predict-", "project-", "soon", "about to", "poised", "set to", "imminent-",
        "danger-", "risk-",
    ),
    "V7": (
        "re:species", "re:races?", "re:tribe[a-z]*", "re:nations?", "re:civili[sz]ation[a-z]*",
        "re:empires?", "re:kingdoms?", "dynast-", "re:clans?", "re:famil(?:y|ies)",
        "re:super-?organism[a-z]*", "genom-", "genetic-", "evolv-", "evolution-", "adapt-", "select-",
        "fitness", "mutat-",
    ),
    "V8": (
        "massive", "huge", "enormous", "vast", "tiny", "extrem-", "record-", "unprecedented",
        "historic-", "significant-", "great-", "best", "worst", "critical-", "dangerous-", "concern-",
        "alarm-", "worr-", "healthy", "good", "bad", "total-", "complete-", "absolute-", "first",
        "re:(?<!not\\s)last", "never", "ever", "always", "stagna-", "monocultur-", "homogen-",
    ),
    "G": (
        "dominan-", "dominat-", "majority", "prevail-", "sharp-", "steep-", "dramatic-",
        "precipitous-", "plummet-", "collaps-", "crash-", "freefall", "surg-", "soar-", "explo-",
        "rapid-", "strong-", "extinct-",
    ),
    "L3": (
        "storm-", "disturbanc-", "hemispher-", "cyclon-", "hurricane-", "typhoon-", "tornado-", "vortex",
        "vortices", "vorticity", "zeta", "eddy", "eddies", "monsoon-", "gale-", "squall-", "tempest-",
        "wind-", "weather", "rain-", "cloud-",
    ),
}  # fmt: skip

# Lists that no report string may match; the lever words, the label names and the storm words are allowed in
# places (R6).
NEVER_LISTS: Final = ("T2", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "G")
LEVER_LISTS: Final = ("L1",)
LABEL_LISTS: Final = ("L2",)
STORM_LISTS: Final = ("L3",)

LEXICON_DIGEST: Final = hashlib.sha256(
    json.dumps({"version": LEXICON_VERSION, "lists": LEXICON}, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def _term(t: str) -> str:
    if t.startswith("re:"):
        return t[3:]
    core = re.escape(t[:-1]) + "[a-z]*" if t.endswith("-") else re.escape(t)
    return core.replace(r"\ ", r"\s+")


def compile_terms(*terms: str) -> re.Pattern[str]:
    """The pattern of ``terms`` over lower-cased text: whole words, stems and phrases (see the module doc)."""
    return re.compile(r"(?<![a-z])(?:" + "|".join(_term(t) for t in terms) + r")(?![a-z])")


_COMPILED: Final = {name: compile_terms(*terms) for name, terms in LEXICON.items()}


def scan(text: str, lists: tuple[str, ...] | None = None) -> list[tuple[str, str]]:
    """``[(list name, matched text)]`` for every match of ``lists`` (default: all) in ``text``, in list order
    then text order. The caller masks display names first; ``text`` is lower-cased here."""
    low = text.lower()
    return [
        (name, m.group(0))
        for name in (LEXICON if lists is None else lists)
        for m in _COMPILED[name].finditer(low)
    ]
