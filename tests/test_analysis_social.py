"""The social study (``alife.analysis.social``, DESIGN §29.12, docs/GAMES.md §6): the churn and the quantiles
on synthetic input, and the study end to end on tiny and tiny3d: the documented report, determinism, the run
directory left byte-unchanged, the stored metrics rows and social records checked (and a diverging one
detected), and the cases in which the records are not comparable."""

from __future__ import annotations

import dataclasses
import json
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alife import runner
from alife.analysis import social
from alife.analysis.resim import ResimMismatch
from alife.config import Config, apply_overrides
from alife.game.observer import Observer
from alife.game.social import SOCIAL_V1, SOCIAL_VERSION
from alife.game.society import SnapInfo
from alife.runio import RunDir, _dumps
from alife.state import init_state
from alife.tracking import TRACK_VERSION, Diagnostics, Event, EventKind
from game_helpers import planted_state, social_every
from helpers import set_output_bias
from test_game_social import BIAS

RUN = ["run.metrics_every=5", "run.checkpoint_every=10", "run.heavy_metrics_every=5"]
STUDY: dict[str, Any] = {"start": 50, "steps": 200, "every": 20}
TOP_KEYS = {
    "run",
    "config_hash",
    "start",
    "steps",
    "params",
    "social_version",
    "social_digest",
    "track_version",
    "device",
    "mlx_version",
    "observer_start",
    "rows_verified",
    "records_verified",
    "records_comparable",
    "records_note",
    "snapshots",
    "churn",
    "timing",
}
SNAP_KEYS = {"step", "n_alive", "thresholds", "kin_groups", "colonies", "houses"}
THRESHOLD_KEYS = {"n_ref", "rescaled", "occ_min", "kin_enter", "kin_leave"}
FAMILY_KEYS = {
    "n_prev",
    "n_cur",
    "n_admitted",
    "n_matched",
    "jaccard_matched",
    "jaccard_unmatched_best",
    "events",
}
KIN_KEYS = {*FAMILY_KEYS, "occupied_bins", "unaffiliated_frac", "tracked", "sizes", "sim_self"}
COLONY_KEYS = {*FAMILY_KEYS, "tracked", "members", "largest", "converged", "match"}
QUANTILE_KEYS = {"n", "p10", "p50", "p90"}
EVENT_KEYS = {"born", "split", "merged", "dissolved"}
CHURN_KEYS = {"born", "ended_within", "share"}
HOUSE_KEYS = {
    "alive",
    "unknown_immigrant",
    "members",
    "seeded_members",
    "founded_members",
    "founded_houses",
    "founded_share",
}


def _make_run(cfg: Config, path: Path, *extra: str, steps: int = 60, planted: bool = False) -> Path:
    cfg = apply_overrides(cfg, [*RUN, *extra])
    rd = RunDir.create(path, cfg)
    state = planted_state(cfg) if planted else init_state(cfg)
    assert runner.run(rd, cfg, state, steps, quiet=True).status == "done"
    return path


@pytest.fixture(scope="module")
def run_dir(tiny_cfg: Config, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 60-step tiny run without observers: metrics every 5 steps, checkpoints every 10."""
    return _make_run(tiny_cfg, tmp_path_factory.mktemp("social") / "run")


@pytest.fixture(scope="module")
def observing_run(tiny_cfg: Config, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 60-step observing run with snapshots every 15 steps (records at 15, 30, 45, 60) and checkpoints at
    0, 30 and 60. A module-scoped fixture cannot use ``monkeypatch``: it patches the default by context."""
    path = tmp_path_factory.mktemp("social_observing") / "run"
    with pytest.MonkeyPatch.context() as mp:
        social_every(mp, 15)
        return _make_run(tiny_cfg, path, "game.observe=true", "run.checkpoint_every=30")


@pytest.fixture(scope="module")
def planted_run(tiny_cfg: Config, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An observing 60-step run whose founders' tags are planted in three tight clouds: it tracks three kin
    groups (the scattered tags of an unseeded tiny world track none)."""
    path = tmp_path_factory.mktemp("social_planted") / "run"
    with pytest.MonkeyPatch.context() as mp:
        social_every(mp, 15)
        return _make_run(tiny_cfg, path, "game.observe=true", "run.checkpoint_every=30", planted=True)


@pytest.fixture(scope="module")
def run_dir_3d(tiny3d_cfg: Config, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _make_run(tiny3d_cfg, tmp_path_factory.mktemp("social3d") / "run")


def _files(path: Path) -> dict[str, tuple[int, int]]:
    return {
        str(p.relative_to(path)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(path.rglob("*"))
        if p.is_file()
    }


def _bytes(path: Path) -> dict[str, bytes]:
    return {str(p.relative_to(path)): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()}


def _deterministic(rep: Mapping[str, Any]) -> str:
    """The report without its timing, serialised (the rest must be identical across runs)."""
    return json.dumps({k: v for k, v in rep.items() if k != "timing"}, sort_keys=False)


def _edit_record(run: Path, step: int) -> None:
    """Change one value of the stored record of ``step`` (a valid line: same origin, digest and keys)."""
    path = run / "observers" / "social.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for r in records:
        if r["step"] == step:
            r["summary"]["houses"]["alive"] += 1
    path.write_text("".join(_dumps(r) + "\n" for r in records))


def ev(kind: EventKind, system: int, snapshot: int = 0) -> Event:
    return Event(step=snapshot * 10, kind=kind, system=system, size=5, other=-1, overlap=0)


# --- the churn and the quantiles on synthetic input -----------------------------------------------------


def test_churn_counts_ids_that_end_within_three_snapshots_of_their_birth() -> None:
    B, S, M, D = EventKind.BORN, EventKind.SPLIT, EventKind.MERGED, EventKind.DISSOLVED
    events = [
        [ev(B, 1), ev(S, 2)],  # snapshot 0: ids 1 and 2 appear
        [ev(B, 3)],  # 1: id 3 appears
        [ev(D, 1, 2)],  # 2: id 1 ends, 2 snapshots after its birth
        [ev(M, 3, 3)],  # 3: id 3 ends, 2 snapshots after
        [],  # 4
        [ev(B, 4)],  # 5
        [ev(D, 2, 6)],  # 6: id 2 ends 6 snapshots after its birth: not churn
        [ev(D, 4, 7)],  # 7: id 4 ends 2 snapshots after
        [ev(B, 5)],  # 8: id 5 never ends
    ]
    out = social.churn(events)
    assert out == {"born": 5, "ended_within": 3, "share": 0.6}
    assert set(out) == CHURN_KEYS


def test_churn_boundary_is_inclusive_and_an_id_ended_before_the_study_is_ignored() -> None:
    B, D = EventKind.BORN, EventKind.DISSOLVED
    # ended exactly 3 snapshots after its birth counts; 4 does not
    assert social.churn([[ev(B, 1)], [], [], [ev(D, 1)]])["ended_within"] == 1
    assert social.churn([[ev(B, 1)], [], [], [], [ev(D, 1)]])["ended_within"] == 0
    assert social.churn([[ev(B, 1)], [], [], [], [ev(D, 1)]], within=4)["ended_within"] == 1
    # an id born before the study and ended in it is not a birth of the study
    out = social.churn([[ev(D, 7)], [ev(B, 8)]])
    assert out == {"born": 1, "ended_within": 0, "share": 0.0}


def test_churn_of_nothing_has_no_share() -> None:
    assert social.churn([]) == {"born": 0, "ended_within": 0, "share": None}
    assert social.churn([[], [], []]) == {"born": 0, "ended_within": 0, "share": None}
    assert social.churn([[ev(EventKind.DISSOLVED, 1)]])["share"] is None


def test_quantiles() -> None:
    assert social._q([]) == {"n": 0, "p10": None, "p50": None, "p90": None}
    q = social._q([5, 1, 2, 4, 3])
    assert q == {"n": 5, "p10": 1.4, "p50": 3.0, "p90": 4.6}
    assert social._q([0.25]) == {"n": 1, "p10": 0.25, "p50": 0.25, "p90": 0.25}


def _diag(n_prev: int, n_cur: int, n_adm: int, matched: list[float], unmatched: list[float]) -> Diagnostics:
    import numpy as np

    return Diagnostics(
        n_prev, n_cur, n_adm, len(matched), np.array(matched, np.float64), np.array(unmatched, np.float64)
    )


def test_the_snapshot_entry_is_built_from_the_societys_diagnostics() -> None:
    """The report builder on synthetic diagnostics: quantiles of the Jaccard values, the event counts (all
    four kinds), the sizes by rank, the unaffiliated fraction, the chaining quantiles over the groups that
    have a ``sim_self`` (None below two members is left out), and the summary of colonies and houses."""
    B, S, M, D = EventKind.BORN, EventKind.SPLIT, EventKind.MERGED, EventKind.DISSOLVED
    info = SnapInfo(
        step=45,
        n_alive=200,
        thresholds={"n_ref": 190, "rescaled": True, "occ_min": 2, "kin_enter": 8, "kin_leave": 4},
        kin_diag=_diag(3, 4, 2, [1.0, 0.5, 0.25], [0.1]),
        col_diag=_diag(0, 0, 0, [], []),
        kin_events=[ev(B, 4, 3), ev(S, 5, 3), ev(M, 1, 3), ev(D, 2, 3), ev(B, 6, 3)],
        col_events=[],
        occupied_bins=17,
        unaffiliated=50,
        kin_sizes={1: 30, 2: 90, 3: 60},
        col_sizes={},
        sim_self={1: 0.5, 2: None, 3: 0.25},
        house_ids=np.array([5, 40, 100, 130]),
        house_sizes=np.array([10, 20, 30, 40]),
    )
    record = {
        "summary": {
            "houses": {"alive": 12, "exact": True, "unknown_immigrant": 3},
            "kin_groups": {"tracked": 3, "other": 20},
            "colonies": {"tracked": 0, "members": 0, "largest": 0, "converged": True, "match": "uid"},
        }
    }
    e = social._snapshot_entry(info, record, 100)
    assert set(e) == SNAP_KEYS and e["step"] == 45 and e["n_alive"] == 200
    assert e["thresholds"] == info.thresholds
    # houses: ids below the start's next_uid (100) were seeded, the others founded since; 3 rows are unknown
    assert e["houses"] == {
        "alive": 12,
        "unknown_immigrant": 3,
        "members": 100,
        "seeded_members": 30,
        "founded_members": 70,
        "founded_houses": 2,
        "founded_share": 0.7,
    }
    assert social._snapshot_entry(info, record, 0)["houses"]["founded_share"] == 1.0
    assert social._snapshot_entry(info, record, 10_000)["houses"]["founded_share"] == 0.0
    kin = e["kin_groups"]
    assert kin["events"] == {"born": 2, "split": 1, "merged": 1, "dissolved": 1}
    assert kin["jaccard_matched"] == {"n": 3, "p10": 0.3, "p50": 0.5, "p90": 0.9}
    assert kin["jaccard_unmatched_best"] == {"n": 1, "p10": 0.1, "p50": 0.1, "p90": 0.1}
    assert (kin["n_prev"], kin["n_cur"], kin["n_admitted"], kin["n_matched"]) == (3, 4, 2, 3)
    assert kin["occupied_bins"] == 17 and kin["unaffiliated_frac"] == 0.25
    assert kin["tracked"] == 3 and kin["sizes"] == [90, 60, 30]
    assert kin["sim_self"] == {"n": 2, "p10": 0.275, "p50": 0.375, "p90": 0.475}
    col = e["colonies"]
    assert col["events"] == {"born": 0, "split": 0, "merged": 0, "dissolved": 0}
    assert col["jaccard_matched"]["n"] == 0 and col["jaccard_matched"]["p50"] is None
    assert (col["tracked"], col["members"], col["largest"], col["converged"], col["match"]) == (
        0, 0, 0, True, "uid",
    )  # fmt: skip
    # an empty world: no fraction to take
    empty = social._snapshot_entry(
        info._replace(
            n_alive=0,
            unaffiliated=0,
            kin_sizes={},
            sim_self={},
            house_ids=np.zeros(0, np.int64),
            house_sizes=np.zeros(0, np.int64),
        ),
        record,
        100,
    )
    assert empty["kin_groups"]["unaffiliated_frac"] is None and empty["kin_groups"]["sim_self"]["n"] == 0
    assert empty["houses"]["members"] == 0 and empty["houses"]["founded_share"] is None


# --- the study ----------------------------------------------------------------------------------------------


def _documented(rep: Mapping[str, Any], *, every: int) -> None:
    assert set(rep) == TOP_KEYS
    assert rep["social_version"] == SOCIAL_VERSION and rep["track_version"] == TRACK_VERSION
    p = rep["params"]
    assert p["every"] == every and p["churn_within"] == social.CHURN_WITHIN
    assert p["match"] == ("uid" if every <= SOCIAL_V1.uid_spacing else "footprint")
    assert p["social"]["snapshot_every"] == every and p["social"]["slots"] == SOCIAL_V1.slots
    assert set(rep["observer_start"]) == {"step", "source", "note", "next_uid"}
    for i, s in enumerate(rep["snapshots"]):
        assert set(s) == SNAP_KEYS, i
        assert set(s["thresholds"]) == THRESHOLD_KEYS
        assert s["step"] % every == 0 and s["step"] > rep["start"]
        kin, col = s["kin_groups"], s["colonies"]
        assert set(kin) == KIN_KEYS and set(col) == COLONY_KEYS and set(s["houses"]) == HOUSE_KEYS
        for fam in (kin, col):
            for q in ("jaccard_matched", "jaccard_unmatched_best"):
                assert set(fam[q]) == QUANTILE_KEYS
            assert set(fam["events"]) == EVENT_KEYS
            assert fam["n_matched"] <= min(fam["n_prev"], fam["n_cur"])
            assert fam["jaccard_matched"]["n"] == fam["n_matched"]
        assert set(kin["sim_self"]) == QUANTILE_KEYS and kin["sizes"] == sorted(kin["sizes"], reverse=True)
        assert kin["tracked"] == len(kin["sizes"]) and col["match"] == p["match"]
    assert set(rep["churn"]) == {"within", "kin_group", "colony"}
    assert set(rep["churn"]["kin_group"]) == CHURN_KEYS and set(rep["churn"]["colony"]) == CHURN_KEYS
    assert set(rep["timing"]) == {"steps", "snapshots"} and rep["timing"]["steps"] == rep["steps"]
    assert [t["step"] for t in rep["timing"]["snapshots"]] == [s["step"] for s in rep["snapshots"]]
    json.dumps(rep, allow_nan=False)  # JSON-safe: no NaN or infinity


def test_study_on_tiny(run_dir: Path) -> None:
    before = _files(run_dir)
    logs: list[str] = []
    rep = social.study(run_dir, **STUDY, log=logs.append)
    _documented(rep, every=20)
    assert _files(run_dir) == before  # read only: names, sizes and mtimes
    assert rep["rows_verified"] > 0 and rep["start"] == 50 and rep["steps"] == 200
    assert rep["observer_start"]["step"] == 50 and rep["observer_start"]["source"] == "fresh"
    assert [s["step"] for s in rep["snapshots"]] == list(range(60, 250, 20))
    assert rep["social_digest"] == dataclasses.replace(SOCIAL_V1, snapshot_every=20).digest()
    assert any("social:" in m for m in logs)
    # every colony event of the snapshots is a birth or an end of the churn count
    born = sum(s["colonies"]["events"]["born"] + s["colonies"]["events"]["split"] for s in rep["snapshots"])
    assert rep["churn"]["colony"]["born"] == born
    kin_born = sum(
        s["kin_groups"]["events"]["born"] + s["kin_groups"]["events"]["split"] for s in rep["snapshots"]
    )
    assert rep["churn"]["kin_group"]["born"] == kin_born
    # a run without observers has no social log: the records are not comparable
    assert rep["records_verified"] == 0 and rep["records_comparable"] is False
    assert rep["records_note"] == "the run has no social log"


def test_the_study_splits_house_members_by_when_the_house_was_founded(
    tiny_cfg: Config, tmp_path: Path
) -> None:
    """With ``game.house_rate = 1`` every birth founds a house: the members of houses founded since the
    study's start (ids from the start state's ``next_uid`` on) are none at the start and grow, the seeded
    members shrink with their founders, and the two always make the members of the houses of known lineage
    (all alive rows but the unknown immigrants). With rate 0 nothing is founded (no immigrant arrives)."""
    shares = {}
    for rate in ("1.0", "0"):
        cfg = apply_overrides(tiny_cfg, [*RUN, f"game.house_rate={rate}"])
        rd = RunDir.create(tmp_path / f"run{rate}", cfg)
        state = set_output_bias(init_state(cfg), cfg, BIAS)  # births from step 21 on
        assert runner.run(rd, cfg, state, 60, quiet=True).status == "done"
        rep = social.study(rd.path, **STUDY, log=lambda _: None)
        first_uid = rep["observer_start"]["next_uid"]
        assert first_uid > cfg.population.n_init  # births before the start: their houses count as seeded
        for s in rep["snapshots"]:
            h = s["houses"]
            assert h["members"] == h["seeded_members"] + h["founded_members"]
            assert h["members"] == s["n_alive"] - h["unknown_immigrant"]
            assert h["founded_houses"] <= h["alive"] and h["founded_members"] >= h["founded_houses"]
        shares[rate] = [s["houses"]["founded_share"] for s in rep["snapshots"]]
        if rate == "1.0":
            founded = [s["houses"]["founded_houses"] for s in rep["snapshots"]]
            assert founded[-1] > founded[0] >= 0 and 0.0 < shares[rate][0] < shares[rate][-1] < 1.0
    assert set(shares["0"]) == {0.0}


def test_the_report_is_deterministic_apart_from_the_timing(run_dir: Path) -> None:
    a = social.study(run_dir, **STUDY, log=lambda _: None)
    b = social.study(run_dir, **STUDY, log=lambda _: None)
    assert _deterministic(a) == _deterministic(b)
    c = social.study(run_dir, **{**STUDY, "every": 25}, log=lambda _: None)
    assert _deterministic(a) != _deterministic(c)
    assert c["social_digest"] != a["social_digest"] and c["params"]["every"] == 25


def test_the_study_reports_the_trackers_of_the_run(run_dir: Path) -> None:
    """The thresholds and counts of a snapshot are the society's own: a tiny world holds colonies (bonds)
    and the first snapshot after the start matches the observer's initial ids."""
    rep = social.study(run_dir, **STUDY, log=lambda _: None)
    first = rep["snapshots"][0]
    th = first["thresholds"]
    # n_ref follows the population only past a 12.5% move: the thresholds are those of n_ref
    assert first["n_alive"] > 0 and abs(th["n_ref"] - first["n_alive"]) <= th["n_ref"] >> SOCIAL_V1.n_band
    assert th["occ_min"] == max(SOCIAL_V1.kin_occ[0], th["n_ref"] >> SOCIAL_V1.kin_occ[1])
    assert th["kin_enter"] == max(SOCIAL_V1.kin_enter[0], th["n_ref"] >> SOCIAL_V1.kin_enter[1])
    assert th["kin_leave"] == max(SOCIAL_V1.kin_leave[0], th["n_ref"] >> SOCIAL_V1.kin_leave[1])
    assert th["rescaled"] is False
    assert first["colonies"]["converged"] is True
    assert sum(s["colonies"]["tracked"] for s in rep["snapshots"]) > 0
    assert 0.0 <= first["kin_groups"]["unaffiliated_frac"] <= 1.0
    assert first["houses"]["alive"] > 0


def test_study_on_tiny3d(run_dir_3d: Path) -> None:
    rep = social.study(run_dir_3d, start=50, steps=100, every=20, log=lambda _: None)
    _documented(rep, every=20)
    assert rep["rows_verified"] > 0 and [s["step"] for s in rep["snapshots"]] == [60, 80, 100, 120, 140]


def test_footprint_matching_beyond_the_uid_spacing(run_dir: Path) -> None:
    """A cadence over ``uid_spacing`` (300) matches by footprint: the report says so and still runs."""
    rep = social.study(run_dir, start=50, steps=310, every=301, log=lambda _: None)
    _documented(rep, every=301)
    assert rep["params"]["match"] == "footprint" and [s["step"] for s in rep["snapshots"]] == [301]
    assert rep["snapshots"][0]["colonies"]["match"] == "footprint"


def test_study_without_a_snapshot_has_empty_tables(run_dir: Path) -> None:
    rep = social.study(run_dir, start=50, steps=10, every=250, log=lambda _: None)
    _documented(rep, every=250)
    assert rep["snapshots"] == [] and rep["timing"]["snapshots"] == []
    assert rep["churn"]["kin_group"] == {"born": 0, "ended_within": 0, "share": None}


def test_time_components_adds_a_timing_per_snapshot(run_dir: Path) -> None:
    plain = social.study(run_dir, **{**STUDY, "steps": 60}, log=lambda _: None)
    timed = social.study(run_dir, **{**STUDY, "steps": 60}, time_components=True, log=lambda _: None)
    assert _deterministic({**plain, "params": None}) == _deterministic({**timed, "params": None})
    assert plain["params"]["time_components"] is False and timed["params"]["time_components"] is True
    for t in plain["timing"]["snapshots"]:
        assert set(t) == {"step", "eval_ms", "before_median_ms", "settle_ms"}
    for t in timed["timing"]["snapshots"]:
        assert set(t) == {"step", "eval_ms", "before_median_ms", "settle_ms", "components_ms"}
        assert t["components_ms"] > 0 and t["settle_ms"] > 0 and t["eval_ms"] > 0
    # eight steps precede the first snapshot (the study starts 10 steps before it)
    assert all(t["before_median_ms"] > 0 for t in timed["timing"]["snapshots"])


def test_the_study_runs_the_observers_of_the_world_with_game_observe_on(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run need not have set ``game.observe``: the study's config has it on, and the social rules are
    passed to the observer explicitly."""
    seen: list[tuple[bool, str]] = []
    real = Observer.open.__func__  # type: ignore[attr-defined]

    def spy(cls, rd, state, cfg, *, rules=None, social=None, tracers=True):
        assert tracers is False  # the analysis tools never run the tracer rig
        seen.append((cfg.game.observe, social.digest()))
        return real(cls, rd, state, cfg, rules=rules, social=social, tracers=tracers)

    monkeypatch.setattr(Observer, "open", classmethod(spy))
    assert RunDir.open(run_dir).config().game.observe is False
    rep = social.study(run_dir, start=50, steps=10, every=10, log=lambda _: None)
    assert seen == [(True, rep["social_digest"])]


def test_the_snapshot_step_is_timed_against_the_eight_steps_before_it(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The five steps before the snapshot at 60 take 0.2 s more, the last four do not: the median of the
    eight steps 52..59 (four slow, four fast) lies between the two; a median over fewer steps would be fast,
    over more would be slow. The snapshot step itself is not part of its own reference."""
    calls = 0
    real = social.make_step

    def make(cfg, **kw):
        fn = real(cfg, **kw)

        def slow(s):
            nonlocal calls
            calls += 1
            if calls <= 5:  # the steps 51..55
                time.sleep(0.2)
            return fn(s)

        return slow

    monkeypatch.setattr(social, "make_step", make)
    rep = social.study(run_dir, start=50, steps=10, every=10, verify=False, log=lambda _: None)
    (t,) = rep["timing"]["snapshots"]
    assert t["step"] == 60 and 60 < t["before_median_ms"] < 170, t
    assert t["eval_ms"] < 100  # the snapshot step is fast, and not slow because of its neighbours


def test_the_study_prepares_the_fused_search_before_it_builds_the_step(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """As ``analysis.labels`` does (DESIGN §4): the self-check of the fused kNN kernel runs once per process,
    before ``make_step``, so a 3D study on Metal never uses the kernel unchecked."""
    order: list[str] = []
    real = social.make_step
    monkeypatch.setattr(social.spatial, "prepare_knn", lambda cfg: order.append("prepare_knn"))

    def make(cfg, **kw):
        order.append("make_step")
        return real(cfg, **kw)

    monkeypatch.setattr(social, "make_step", make)
    social.study(run_dir, start=50, steps=10, every=10, verify=False, log=lambda _: None)
    assert order == ["prepare_knn", "make_step"]


def test_study_detects_a_diverging_row(run_dir: Path, tmp_path: Path) -> None:
    bad = tmp_path / "run"
    shutil.copytree(run_dir, bad)
    rows = [json.loads(line) for line in (bad / "metrics.jsonl").read_text().splitlines()]
    for r in rows:
        if r["step"] == 55:
            r["births"] += 1
    (bad / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ResimMismatch, match="step 55"):
        social.study(bad, **STUDY, log=lambda _: None)
    out = tmp_path / "report.json"
    argv = [str(bad), "--from", "50", "--steps", "200", "--every", "20", "--out", str(out)]
    assert social.main(argv) == 1 and not out.exists()
    # --no-verify skips the stored rows
    assert social.main([*argv, "--no-verify"]) == 0 and out.exists()


def test_study_from_a_cached_state(run_dir: Path, tmp_path: Path) -> None:
    from alife.analysis import resim

    cache = tmp_path / "cache"
    resim.resimulate(run_dir, [55], cache, log=lambda _: None)
    rep = social.study(run_dir, start=55, steps=100, every=20, states_dir=cache, log=lambda _: None)
    assert rep["observer_start"]["source"] == "fresh" and rep["rows_verified"] > 0
    assert [s["step"] for s in rep["snapshots"]] == [60, 80, 100, 120, 140]


def test_study_rejects_bad_parameters(run_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="steps"):
        social.study(run_dir, **{**STUDY, "steps": 0})
    with pytest.raises(ValueError, match="every"):
        social.study(run_dir, **{**STUDY, "every": 0})
    with pytest.raises(FileNotFoundError):
        social.study(run_dir, **{**STUDY, "start": 51})  # no checkpoint there
    out = tmp_path / "r.json"
    base = [str(run_dir), "--out", str(out), "--steps", "20"]
    assert social.main([*base, "--from", "50", "--every", "0"]) == 2
    assert social.main([*base, "--from", "51"]) == 2
    assert social.main([*base[:-1], "0", "--from", "50"]) == 2  # --steps 0
    assert not out.exists()


# --- the stored social records ---------------------------------------------------------------------------


def test_the_study_checks_its_records_against_the_stored_log(observing_run: Path) -> None:
    before = _bytes(observing_run)
    rep = social.study(observing_run, start=30, steps=30, every=15, log=lambda _: None)
    _documented(rep, every=15)
    assert _bytes(observing_run) == before  # read only, the log included
    assert rep["observer_start"] == {
        "step": 30,
        "source": "sidecar",
        "note": None,
        "next_uid": rep["observer_start"]["next_uid"],
    }
    assert rep["observer_start"]["next_uid"] > 0
    assert [s["step"] for s in rep["snapshots"]] == [45, 60]
    assert rep["records_verified"] == 2 and rep["records_comparable"] is True and rep["records_note"] is None
    assert rep["rows_verified"] > 0
    # the report's snapshots are the stored records': thresholds and the summary counts agree
    stored = {r["step"]: r for r in RunDir.open(observing_run).social_records()}
    for s in rep["snapshots"]:
        rec = stored[s["step"]]
        assert (
            s["thresholds"] == rec["thresholds"] and s["houses"]["alive"] == rec["summary"]["houses"]["alive"]
        )
        assert s["colonies"]["members"] == rec["summary"]["colonies"]["members"]
        assert s["colonies"]["tracked"] == rec["summary"]["colonies"]["tracked"]
        assert s["kin_groups"]["tracked"] == rec["summary"]["kin_groups"]["tracked"]
    assert rep["social_digest"] == stored[45]["social_digest"]


def test_a_stored_record_that_differs_is_a_mismatch(observing_run: Path, tmp_path: Path) -> None:
    bad = tmp_path / "run"
    shutil.copytree(observing_run, bad)
    _edit_record(bad, 45)
    with pytest.raises(ResimMismatch, match=r"step 45.*social.*summary"):
        social.study(bad, start=30, steps=30, every=15, log=lambda _: None)
    out = tmp_path / "report.json"
    argv = [str(bad), "--from", "30", "--steps", "30", "--every", "15", "--out", str(out)]
    assert social.main(argv) == 1 and not out.exists()
    # the stored rows are fine, and --no-verify reads no record
    assert social.main([*argv, "--no-verify"]) == 0
    rep = json.loads(out.read_text())
    assert rep["records_verified"] == 0 and rep["records_comparable"] is False
    assert rep["records_note"] == "verification is off (--no-verify)" and rep["rows_verified"] == 0


def test_records_are_not_comparable_under_another_cadence(observing_run: Path) -> None:
    rep = social.study(observing_run, start=30, steps=30, every=20, log=lambda _: None)
    assert rep["observer_start"]["source"] == "fresh"  # the sidecar's social digest is another one
    assert "social_digest" in rep["observer_start"]["note"]
    assert rep["records_verified"] == 0 and rep["records_comparable"] is False
    assert "differ" in rep["records_note"]  # step 60 falls on a stored record of another digest
    assert rep["rows_verified"] > 0 and [s["step"] for s in rep["snapshots"]] == [40, 60]


def test_records_are_not_comparable_after_a_fresh_start(observing_run: Path, tmp_path: Path) -> None:
    """Without the sidecar the study's observer starts fresh at ``--from``: ids and the origin differ from
    the run's, so its records are not compared (and are not an error)."""
    fresh = tmp_path / "run"
    shutil.copytree(observing_run, fresh)
    RunDir.open(fresh).observer_path(30).unlink()
    rep = social.study(fresh, start=30, steps=30, every=15, log=lambda _: None)
    assert rep["observer_start"]["source"] == "fresh"
    assert rep["records_verified"] == 0 and rep["records_comparable"] is False
    assert (
        rep["records_note"]
        == "the study's observer started fresh (origin 30), the stored records have origin 0"
    )
    assert rep["rows_verified"] > 0


def test_a_study_from_the_start_of_the_run_checks_every_record(observing_run: Path) -> None:
    """From step 0 the observer is the run's own: all four records of the log are re-simulated and equal."""
    rep = social.study(observing_run, start=0, steps=60, every=15, log=lambda _: None)
    assert rep["observer_start"]["source"] == "sidecar" and rep["records_verified"] == 4
    assert [s["step"] for s in rep["snapshots"]] == [15, 30, 45, 60] and rep["rows_verified"] > 0


def test_records_of_another_digest_are_not_compared_even_from_the_same_origin(observing_run: Path) -> None:
    """From step 0 with another cadence the study starts fresh at the run's own origin (0), so only the social
    digest tells the study's record of step 60 from the stored one: it must not be compared (nor raise)."""
    rep = social.study(observing_run, start=0, steps=60, every=20, log=lambda _: None)
    assert rep["observer_start"]["source"] == "fresh"
    assert [s["step"] for s in rep["snapshots"]] == [20, 40, 60]
    assert rep["records_verified"] == 0 and rep["records_comparable"] is False
    assert (
        rep["records_note"]
        == "the study's social rules (cadence or thresholds) differ from the stored records' digest"
    )


def test_a_study_past_the_end_of_the_log_compares_what_it_can(observing_run: Path) -> None:
    """From 30 for 60 steps: the records of 45 and 60 are compared, those of 75 and 90 have no stored twin."""
    rep = social.study(observing_run, start=30, steps=60, every=15, log=lambda _: None)
    assert [s["step"] for s in rep["snapshots"]] == [45, 60, 75, 90]
    assert rep["records_verified"] == 2 and rep["records_comparable"] is True


def test_a_study_of_a_world_with_kin_groups(planted_run: Path) -> None:
    """Three tracked kin groups: their counts, sizes, Jaccard quantiles and ``sim_self`` are reported, and the
    study's records (kin groups, territory and relations included) equal the stored ones."""
    rep = social.study(planted_run, start=30, steps=30, every=15, log=lambda _: None)
    _documented(rep, every=15)
    assert rep["records_verified"] == 2 and rep["observer_start"]["source"] == "sidecar"
    stored = {r["step"]: r for r in RunDir.open(planted_run).social_records()}
    assert all(stored[t]["summary"]["kin_groups"]["tracked"] == 3 for t in (45, 60))
    for s in rep["snapshots"]:
        kin = s["kin_groups"]
        assert kin["tracked"] == 3 and len(kin["sizes"]) == 3 and sum(kin["sizes"]) <= s["n_alive"]
        assert kin["n_prev"] == kin["n_matched"] == kin["jaccard_matched"]["n"] == 3
        assert kin["events"] == {"born": 0, "split": 0, "merged": 0, "dissolved": 0}
        assert kin["jaccard_matched"]["p10"] > 0.5 and kin["unaffiliated_frac"] < 0.5
        sim = kin["sim_self"]  # the chaining report: one value per slotted group
        assert sim["n"] == 3 and 0.0 < sim["p10"] <= sim["p50"] <= sim["p90"] <= 1.0
    assert rep["churn"]["kin_group"] == {"born": 0, "ended_within": 0, "share": None}


def test_cli(run_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out" / "report.json"
    argv = [
        str(run_dir),
        "--from",
        "50",
        "--steps",
        "60",
        "--every",
        "20",
        "--time-components",
        "--out",
        str(out),
    ]
    assert social.main(argv) == 0
    rep = json.loads(out.read_text())
    assert set(rep) == TOP_KEYS and rep["params"]["every"] == 20 and rep["params"]["time_components"] is True
    assert "wrote" in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:  # the report must not land in the run directory
        social.main([*argv[:-2], "--out", str(run_dir / "report.json")])
    assert e.value.code == 2 and not (run_dir / "report.json").exists()


def test_the_default_cadence_is_the_runs_social_rules(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``--every`` the study uses ``observer.DEFAULT_SOCIAL`` as of the call."""
    rules = social_every(monkeypatch, 25)
    rep = social.study(run_dir, start=50, steps=60, log=lambda _: None)
    assert rep["params"]["every"] == 25 and rep["social_digest"] == rules.digest()
    assert [s["step"] for s in rep["snapshots"]] == [75, 100]
