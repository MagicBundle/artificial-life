"""One interface for the five per-feature scorecards (DESIGN §44.6).

The sibling feature PRs (arms, bonds, comm, niche/farming, structures) each define their own measurement and
pre-registered readings; this suite only collects them. A feature lands an **adapter** here (about 30 lines:
call its own entry point, map the output to ``ScorecardResult``) and replaces its stub in ``SCORECARDS``.
Until then the stub reports ``needs_missing`` and the panel prints "not evaluated", never a zero.

Conventions: ``endpoint["primary"]`` is the confirmatory scalar, direction +1 = more emergent (the primaries
are frozen in the pre-registration: arms escalation index, bonds verdict level, comm transfer bits, niche
max-layer persist, structures builder fixation minus the zero-column control's). Two-sided features use
``|.|`` only where the sibling defines it.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from alife.analysis.emergence.frames import Frames
    from alife.runio import RunDir

FEATURES = ("arms", "bonds", "comm", "niche", "structures")
SCORECARD_VERSION = 1  # of this interface; each result carries its sibling's own ``version``

# feature -> the module a sibling PR adds (checked with find_spec, never imported by the stub)
SIBLING_MODULES: dict[str, str] = {
    "arms": "alife.analysis.arms",
    "bonds": "alife.analysis.bonds",
    "comm": "alife.analysis.comm",
    "niche": "alife.analysis.niche",
    "structures": "alife.analysis.emergence.structures",
}


@dataclass(frozen=True)
class Criterion:
    name: str
    value: float | None
    threshold: float | None
    passed: bool | None
    note: str = ""


@dataclass(frozen=True)
class ScorecardResult:
    feature: str  # "arms" | "bonds" | "comm" | "niche" | "structures"
    version: str  # the sibling module's scorecard version; provisional thresholds say so
    level: int | None  # ordinal verdict when the sibling defines one (bonds 0-3), else None
    label: str  # e.g. "escalation", "language", "stasis"
    endpoint: dict[str, float | None]  # named scalars, +1 = more emergent; the confirmatory one is "primary"
    criteria: list[Criterion] = field(default_factory=list)
    provisional: bool = True
    needs_missing: list[str] = field(default_factory=list)  # absent data -> "not evaluated", never zero

    @property
    def primary(self) -> float | None:
        return self.endpoint.get("primary")

    @property
    def evaluated(self) -> bool:
        return not self.needs_missing and self.primary is not None

    def to_json(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "version": self.version,
            "level": self.level,
            "label": self.label,
            "endpoint": dict(self.endpoint),
            "criteria": [vars(c).copy() for c in self.criteria],
            "provisional": self.provisional,
            "needs_missing": list(self.needs_missing),
            "evaluated": self.evaluated,
        }


class Scorecard(Protocol):
    name: str

    def available(self, run: RunDir) -> list[str]:
        """What is missing for this run ([] when everything needed exists)."""
        ...

    def compute(
        self,
        run: RunDir,
        window: tuple[int, int],
        frames: Frames | None,
        controls: Mapping[str, RunDir],
        out: Path,
    ) -> ScorecardResult: ...


class StubScorecard:
    """The placeholder of a feature whose analysis module has not landed: always ``needs_missing``."""

    def __init__(self, name: str, module: str | None = None) -> None:
        self.name = name
        self.module = module or SIBLING_MODULES.get(name, f"alife.analysis.{name}")

    def available(self, run: RunDir) -> list[str]:
        try:
            found = importlib.util.find_spec(self.module) is not None
        except (ImportError, ValueError):  # a parent package that is missing
            found = False
        return [] if found else [f"module {self.module} (the {self.name} analysis has not landed)"]

    def compute(
        self,
        run: RunDir,
        window: tuple[int, int],
        frames: Frames | None,
        controls: Mapping[str, RunDir],
        out: Path,
    ) -> ScorecardResult:
        return not_evaluated(self.name, self.available(run) or [f"adapter for {self.name} (stub)"])


def not_evaluated(feature: str, needs: list[str], version: str = "none") -> ScorecardResult:
    return ScorecardResult(
        feature=feature,
        version=version,
        level=None,
        label="not evaluated",
        endpoint={"primary": None},
        provisional=True,
        needs_missing=list(needs),
    )


def _stub(name: str) -> Callable[[], Scorecard]:
    return lambda: StubScorecard(name)


# Lazy factories: a sibling PR replaces its entry with ``lambda: <its adapter class>()`` (imported inside the
# lambda), so an absent or broken sibling module is a needs_missing entry, never an ImportError at import.
SCORECARDS: dict[str, Callable[[], Scorecard]] = {f: _stub(f) for f in FEATURES}


def _comm() -> Scorecard:
    from alife.analysis.comm import CommScorecard  # the voice log's adapter (DESIGN 39.14)

    return CommScorecard()


SCORECARDS["comm"] = _comm


def evaluate(
    feature: str,
    run: RunDir,
    window: tuple[int, int],
    frames: Frames | None,
    controls: Mapping[str, RunDir],
    out: Path,
) -> ScorecardResult:
    """One scorecard through the registry, degrading to ``needs_missing`` on an absent module, an unknown
    feature, or an ``available`` list that is not empty. A sibling's own exception propagates (a bug is not
    missing data)."""
    factory = SCORECARDS.get(feature)
    if factory is None:
        return not_evaluated(feature, [f"unknown feature {feature!r}"])
    try:
        card = factory()
    except ImportError as e:
        return not_evaluated(feature, [f"import failed: {e}"])
    need = card.available(run)
    if need:
        return not_evaluated(feature, need)
    return card.compute(run, window, frames, controls, out)


def not_worse(active: ScorecardResult, inert: ScorecardResult) -> bool | None:
    """The feature's scorecard is not worse than its inert twin's (§44.8 clause b): ``primary`` of the active
    arm at least the inert arm's. None when either side was not evaluated."""
    if not (active.evaluated and inert.evaluated):
        return None
    assert active.primary is not None and inert.primary is not None
    return bool(active.primary >= inert.primary)
