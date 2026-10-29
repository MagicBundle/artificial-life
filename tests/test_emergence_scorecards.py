"""The scorecard interface (DESIGN §44.6): the registry degrades to ``needs_missing``, a fake maps through."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from alife.analysis.emergence import panel, scorecards
from alife.analysis.emergence.scorecards import Criterion, ScorecardResult, StubScorecard


def test_every_feature_has_a_stub_that_reports_needs_missing(tmp_path: Path) -> None:
    for f in scorecards.FEATURES:
        r = scorecards.evaluate(f, SimpleNamespace(), (0, 100), None, {}, tmp_path)
        assert not r.evaluated and r.needs_missing and r.primary is None and r.label == "not evaluated"
    assert scorecards.evaluate("telepathy", SimpleNamespace(), (0, 1), None, {}, tmp_path).needs_missing


def test_absent_sibling_module_is_a_missing_entry_not_an_import_error() -> None:
    card = StubScorecard("arms", "alife.analysis.no_such_module_xyz")
    assert "no_such_module_xyz" in card.available(SimpleNamespace())[0]
    card = StubScorecard("nested", "alife.analysis.no_pkg.no_mod")
    assert card.available(SimpleNamespace())
    present = StubScorecard("stats", "alife.analysis.emergence.stats")
    assert present.available(SimpleNamespace()) == []


class _Fake:
    name = "fake"

    def available(self, run: object) -> list[str]:
        return []

    def compute(self, run, window, frames, controls, out) -> ScorecardResult:
        return ScorecardResult(
            feature="comm",
            version="fake-1",
            level=2,
            label="language",
            endpoint={"primary": 0.12, "z": 5.0},
            criteria=[
                Criterion("transfer z", 5.0, 4.0, True),
                Criterion("bits", 0.12, 0.05, True, "provisional"),
            ],
            provisional=True,
        )


def test_a_fake_scorecard_maps_through_the_registry_and_the_panel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(scorecards.SCORECARDS, "comm", lambda: _Fake())
    r = scorecards.evaluate("comm", SimpleNamespace(), (0, 10), None, {}, tmp_path)
    assert r.evaluated and r.primary == 0.12 and r.level == 2 and len(r.criteria) == 2
    p = panel.new_panel(
        run="r", arm="U5", seed=0, config_hash="h", fork_step=0, windows=[(0, 10)], scorecards=[r]
    )
    card = p["D"]["cards"]["comm"]
    assert card["evaluated"] and card["criteria"][1]["note"] == "provisional"
    ep = p["endpoints"]["S_comm"]
    assert ep["status"] == "ok" and ep["value"] == 0.12
    assert p["endpoints"]["S_arms"]["status"] == "needs_missing"  # not requested: never zero
    md = panel.to_markdown(p)
    assert "comm: language" in md and "arms: not evaluated" in md


def test_primary_direction_and_not_worse() -> None:
    def res(v: float | None) -> ScorecardResult:
        return ScorecardResult(
            "bonds", "t", 1, "x", {"primary": v}, needs_missing=[] if v is not None else ["dump"]
        )

    assert scorecards.not_worse(res(2), res(1)) is True
    assert scorecards.not_worse(res(1), res(1)) is True
    assert scorecards.not_worse(res(0), res(1)) is False
    assert scorecards.not_worse(res(None), res(1)) is None
    assert scorecards.not_worse(res(1), res(None)) is None
