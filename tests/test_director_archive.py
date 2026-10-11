"""The director's behaviour archive: novelty, membership, the coverage grid and the deterministic rebuild."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alife.config import Config, apply_overrides
from alife.director import Director
from alife.director.archive import Archive
from alife.director.rules import RULE_PARAMS
from alife.director.vitals import GRID_AXES, Tracker
from alife.runio import RunDir
from test_director_vitals import row


def small(rho: float = 0.0, exclude: int = 2, k: int = 2) -> Archive:
    return Archive(k=k, exclude=exclude, rho=rho, bins=10, axes=(0, 1, 2))


def at(x: float, y: float = 0.0, z: float = 0.0, w: float = 0.0) -> tuple[float, ...]:
    return (x, y, z, w)


def test_grid_cells_bin_each_axis_and_keep_one_in_the_last_bin() -> None:
    a = small()
    assert a.cell(at(0.0, 0.05, 0.1)) == (0, 0, 1)
    assert a.cell(at(0.099, 0.5, 0.95)) == (0, 5, 9)
    assert a.cell(at(1.0, 1.0, 1.0)) == (9, 9, 9)  # b = 1 is the top bin, not an 11th
    assert Archive(5, 3, 0.02, 10, GRID_AXES).axes == (0, 3, 6)  # emitters, bodies >= 4, predation


def test_novelty_is_the_mean_distance_to_the_k_nearest_without_the_recent_entries() -> None:
    a = small(rho=0.0, exclude=2, k=2)
    for i, x in enumerate((0.0, 0.2, 0.4, 0.6)):
        a.observe(i * 10, at(x))
    assert len(a.entries) == 4
    # Distances are Euclidean / sqrt(4): the pool excludes the 2 newest entries (x = 0.4, 0.6).
    assert a.novelty(at(0.9)) == pytest.approx(((0.9 - 0.2) + (0.9 - 0.0)) / 2 / 2)
    assert Archive(2, 0, 0.0, 10, (0, 1, 2)).novelty(at(0.9)) is None  # an empty archive scores nothing
    full = small(exclude=0)
    for i, x in enumerate((0.0, 0.2, 0.4, 0.6)):
        full.observe(i, at(x))
    assert full.novelty(at(0.9)) == pytest.approx(((0.9 - 0.6) + (0.9 - 0.4)) / 2 / 2)  # nothing excluded


def test_a_descriptor_joins_the_archive_only_above_rho_or_while_there_is_nothing_to_compare() -> None:
    a = small(rho=0.1, exclude=1, k=1)
    assert a.observe(0, at(0.0)) == (None, True, True)  # nothing to compare with
    assert a.observe(1, at(0.0)).added  # still nothing: the only entry is excluded
    assert a.novelty(at(0.0)) == 0.0
    near = a.observe(2, at(0.1))  # distance 0.1 / 2 = 0.05 <= rho
    assert near.novelty == pytest.approx(0.05) and not near.added
    far = a.observe(3, at(0.5))  # 0.25 > rho
    assert far.novelty == pytest.approx(0.25) and far.added
    assert [s for s, _ in a.entries] == [0, 1, 3]


def test_coverage_counts_cells_ever_visited_added_or_not() -> None:
    a = small(rho=10.0, exclude=0)  # nothing but the first descriptor is ever added
    first = a.observe(100, at(0.05))
    again = a.observe(200, at(0.06))  # same cell
    other = a.observe(300, at(0.55))
    assert (first.new_cell, again.new_cell, other.new_cell) == (True, False, True)
    assert a.coverage == 2 and len(a.entries) == 1
    assert a.cells == {(0, 0, 0): 100, (5, 0, 0): 300}
    assert a.last_new_cell == 300
    assert a.new_cells_since(100) == 1 and a.new_cells_since(299) == 1 and a.new_cells_since(300) == 0
    assert small().last_new_cell is None


def test_to_json_is_plain_json_with_the_as_of_step() -> None:
    a = small(rho=0.0, exclude=1)
    a.observe(10, at(0.3, 0.4))
    a.observe(20, at(0.9, 0.1))
    doc = json.loads(json.dumps(a.to_json(25)))
    assert doc["as_of_step"] == 25 and doc["coverage"] == 2 and doc["axes"] == [0, 1, 2]
    assert doc["cells"] == [[3, 4, 0, 10], [9, 1, 0, 20]]
    assert doc["entries"] == [[10, [0.3, 0.4, 0.0, 0.0]], [20, [0.9, 0.1, 0.0, 0.0]]]


def drifting_rows(n: int, seed: int = 3) -> list[dict[str, Any]]:
    """Rows every 10 steps whose descriptors wander, with a regime change half-way."""
    g = np.random.default_rng(seed)
    out = []
    for i in range(1, n + 1):
        half = 0.3 if i <= n // 2 else 0.7
        out.append(
            row(
                i * 10,
                emitters_frac=float(np.clip(half + g.normal(0, 0.05), 0, 1)),
                sharers_frac=float(np.clip(0.1 + g.normal(0, 0.02), 0, 1)),
                predation_ratio=float(np.clip(0.05 + g.normal(0, 0.02), 0, 1)),
                frac_in_bodies_ge4=float(np.clip(half / 2 + g.normal(0, 0.03), 0, 1)),
                bodies_hist={"1": 70},
                bodies_largest=5,
            )
        )
    return out


def test_rebuild_from_rows_equals_the_live_archive(tiny_cfg: Config, tmp_path: Path) -> None:
    """A7: the tracker rebuilt from ``metrics.jsonl`` is the one the live run built."""
    every = 50
    rows = drifting_rows(120)
    live = Tracker(RULE_PARAMS)
    for r in rows:  # live: vitals are computed at each boundary too (they must not change the fold)
        live.observe(r)
        if r["step"] % every == 0:
            live.vitals(live.close(r["step"]), 500.0, 1000)
    assert live.archive.coverage > 3 and len(live.archive.entries) > 3

    cfg = apply_overrides(
        tiny_cfg, ["director.enabled=true", "run.director=live", f"director.rule_every={every}"]
    )
    rd = RunDir.create(tmp_path / "run", cfg)
    stop = 900  # a checkpoint mid-run: rows past it are not part of the rebuild
    rd.append_metrics([r for r in rows if r["step"] <= stop + 30])  # rows beyond the start step are ignored
    director = Director(rd, cfg, "live", start_step=stop)
    rebuilt = director.tracker
    ref = Tracker(RULE_PARAMS)
    for r in rows:
        if r["step"] <= stop:
            ref.observe(r)
            if r["step"] % every == 0:
                ref.close(r["step"])
    assert rebuilt.archive.to_json(stop) == ref.archive.to_json(stop)
    assert [(h.step, h.b, h.obs) for h in rebuilt.history] == [(h.step, h.b, h.obs) for h in ref.history]
    assert (rebuilt.origin, rebuilt.mono_since, len(rebuilt.rows)) == (
        ref.origin,
        ref.mono_since,
        len(ref.rows),
    )
    # Continuing the live tracker from the same point gives the same archive as the full live pass.
    full = Tracker(RULE_PARAMS)
    for r in rows:
        full.observe(r)
        if r["step"] % every == 0:
            full.close(r["step"])
    for r in rows:
        if r["step"] > stop:
            rebuilt.observe(r)
            if r["step"] % every == 0:
                rebuilt.close(r["step"])
    assert rebuilt.archive.to_json(1200) == full.archive.to_json(1200) == live.archive.to_json(1200)


def test_rebuild_keeps_the_rows_after_the_last_boundary_for_the_next_window(
    tiny_cfg: Config, tmp_path: Path
) -> None:
    cfg = apply_overrides(tiny_cfg, ["director.enabled=true", "run.director=live", "director.rule_every=100"])
    rd = RunDir.create(tmp_path / "run", cfg)
    rd.append_metrics([row(s) for s in range(10, 131, 10)])
    d = Director(rd, cfg, "live", start_step=130)
    assert [r["step"] for r in d.tracker.rows] == [110, 120, 130] and len(d.tracker.history) == 1


def test_stagnation_signals_track_new_cells_and_the_descriptor_shift() -> None:
    tr = Tracker(RULE_PARAMS)
    for i in range(1, 6):
        tr.observe(row(i * 100_000, steps=100_000, emitters_frac=0.3))
        v = tr.vitals(tr.close(i * 100_000), 500.0, 1000)
    assert v.coverage == 1 and v.new_cells_200k == 0 and v.last_new_cell == 100_000
    assert v.b_shift == pytest.approx(0.0) and v.novelty == pytest.approx(0.0)
    tr.observe(row(600_000, steps=100_000, emitters_frac=0.9))  # a new regime: a new cell, a shift, novelty
    v = tr.vitals(tr.close(600_000), 500.0, 1000)
    assert v.coverage == 2 and v.new_cells_200k == 1 and v.last_new_cell == 600_000
    assert v.b_shift > 0.1 and v.novelty > 0.1
