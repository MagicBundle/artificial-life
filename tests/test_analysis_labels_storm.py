"""§32 the labels study in the storm world: the wind-relative track (``rg2_rel``), the outcomes of Drifter,
Floater and Sinker, their scoring (strata extended by a z band), the calibration dump and the ``--rules``
option."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from alife import runner
from alife.analysis import labels
from alife.config import Config, apply_overrides
from alife.game.rules import RULES_V2, RULES_V3, Archetype, arch_layout
from alife.runio import RunDir
from alife.state import init_state, to_numpy
from weather_helpers import storm3d_config

BIT = {a: 1 << (a.value - 1) for a in Archetype if a is not Archetype.NONE}


def _np_min_image(d: np.ndarray, size: float) -> np.ndarray:
    return d - size * np.round(d / size)


# --- the tracker with the drift subtracted ----------------------------------------------------------------


def test_tracker_integrates_the_track_with_the_drift_subtracted() -> None:
    """``ur_k = ur_{k-1} + min_image(Δpos_k) − drift_k``, its Rg² against a numpy double; the raw track, the
    heights are unchanged by the drift."""
    n, nd, size, steps = 6, 3, 20.0, 12
    rng = np.random.default_rng(21)
    member = np.array([True, True, True, False, True, True])
    cuid = np.where(member, np.arange(n) + 100, -1).astype(np.int64)
    alive = np.ones((steps, n), bool)
    alive[7:, 1] = False  # row 1 dies
    uid = np.broadcast_to(np.arange(n, dtype=np.int64) + 100, (steps, n)).copy()
    start = rng.uniform(1, size - 1, (n, nd)).astype(np.float32)
    start[:, 2] = [10.0, 3.0, 12.0, 5.0, 2.0, 9.0]
    drift = rng.normal(0, 0.05, (steps, n, nd)).astype(np.float32)
    own = rng.normal(0, 0.1, (steps, n, nd))
    pos = np.zeros((steps, n, nd))
    cur = start.astype(np.float64)
    for k in range(steps):
        cur = cur + own[k] + drift[k]
        cur[:, 2] = np.clip(cur[:, 2], 0.0, 15.0)
        pos[k] = np.mod(cur, size) if True else cur
        pos[k, :, 2] = cur[:, 2]
    pos = pos.astype(np.float32)
    oc = np.zeros((steps, n, len(labels.OUTCOMES)), np.int32)

    trk = labels.track_start(mx.array(member), mx.array(start))
    for k in range(steps):
        trk = labels.track_step(
            mx.array(cuid), trk, mx.array(alive[k]), mx.array(uid[k]), mx.array(pos[k]), mx.array(oc[k]),
            size, mx.array(drift[k]),
        )  # fmt: skip
    mx.eval(*trk)
    host = labels._host_track(trk)

    ok = member.copy()
    last = start.astype(np.float64)
    ur, u = np.zeros((n, nd)), np.zeros((n, nd))
    pts_r: list[list[np.ndarray]] = [[] for _ in range(n)]
    zlast = start[:, 2].astype(np.float64)
    for k in range(steps):
        ok = ok & alive[k]
        step = _np_min_image(pos[k].astype(np.float64) - last, size)
        step[:, 2] = pos[k, :, 2] - last[:, 2]  # min_image is the identity on z
        u = u + np.where(ok[:, None], step, 0.0)
        ur = ur + np.where(ok[:, None], step - drift[k], 0.0)
        last = np.where(ok[:, None], pos[k], last)
        zlast = np.where(ok, pos[k, :, 2], zlast)
        for i in np.flatnonzero(ok):
            pts_r[i].append(ur[i].copy())
    for i in np.flatnonzero(host["n_ok"] > 0):
        pr = np.array(pts_r[i])
        ref = ((pr - pr.mean(axis=0)) ** 2).sum(axis=1).mean()
        assert host["rg2_rel"][i] == pytest.approx(ref, rel=3e-4, abs=1e-6), i
    assert np.allclose(host["ur_h"], ur[:, :2], atol=2e-4) and np.allclose(host["u_h"], u[:, :2], atol=2e-4)
    assert np.allclose(host["zk"], zlast, atol=1e-5)
    assert np.allclose(host["z0"], start[:, 2]) and host["n_ok"].tolist() == [12, 7, 12, 0, 12, 12]
    # the raw Rg² is not the wind-relative one
    assert not np.allclose(host["rg2"][[0, 2]], host["rg2_rel"][[0, 2]], rtol=1e-3)


def test_without_drift_the_two_tracks_are_one() -> None:
    n, nd, size = 8, 2, 10.0
    rng = np.random.default_rng(4)
    member = np.ones(n, bool)
    cuid = np.arange(n).astype(np.int64)
    pos0 = rng.uniform(0, size, (n, nd)).astype(np.float32)
    trk = labels.track_start(mx.array(member), mx.array(pos0))
    for _ in range(9):
        pos0 = np.mod(pos0 + rng.normal(0, 0.5, (n, nd)).astype(np.float32), size).astype(np.float32)
        trk = labels.track_step(
            mx.array(cuid), trk, mx.ones(n, dtype=mx.bool_), mx.array(cuid), mx.array(pos0),
            mx.zeros((n, len(labels.OUTCOMES)), dtype=mx.int32), size,
        )  # fmt: skip
    mx.eval(*trk)
    host = labels._host_track(trk)
    assert np.array_equal(host["rg2"], host["rg2_rel"]) and np.array_equal(host["u_h"], host["ur_h"])
    assert not host["z0"].any() and not host["zk"].any()  # 2D: the height columns are zeros


def test_the_wind_share_of_the_displacement() -> None:
    u_h = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    own = np.array([[3.0, 4.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])  # Σ own = ur_h
    share = labels.wind_share(u_h, own)
    assert share[0] == 0.0  # all own: the drift is zero
    assert share[1] == 0.0  # nothing moved
    assert share[2] == 1.0  # all wind
    assert share[3] == pytest.approx(0.5)  # |drift| = 1, |own| = 1
    # the share is a ratio of norms: a wind against the own motion counts as wind
    assert labels.wind_share(np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]))[0] == pytest.approx(0.5)


def test_z_bands_follow_the_world_height() -> None:
    z = np.array([0.0, 3.99, 4.0, 31.9, 32.0, 55.9, 56.0, 80.0])
    assert labels.z_bands(z, 80.0).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    frac = np.array([0.0, 0.04, 0.05, 0.39, 0.4, 0.69, 0.7, 1.0])
    assert labels.z_bands(frac * 24.0, 24.0).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


# --- scoring ------------------------------------------------------------------------------------------------


def _table(groups: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """A pooled archetype table: each group is ``count`` identical rows (match, ``n_ok``, outcomes)."""
    keys = (
        "cohort", "age", "primary", "match", "n_ok", "rg2_min", "rg2_full", "harvest", "attack", "out", "got",
        "rg2_rel_min", "rg2_rel_full", "z0", "zk_min", "wshare_min",
    )  # fmt: skip
    cols: dict[str, list[float]] = {k: [] for k in keys}
    for g in groups:
        for _ in range(g["count"]):
            row = dict.fromkeys(keys, 0) | {"n_ok": 100, **g}
            row["age"] = len(cols["age"])
            for k in keys:
                cols[k].append(row[k])
    return {
        k: np.array(v, dtype=np.float64 if k.startswith(("rg2", "z", "wshare")) else np.int64)
        for k, v in cols.items()
    }


def test_nomad_and_settler_are_scored_on_the_wind_relative_track_under_a_wind_relative_rule_set() -> None:
    nomad = BIT[Archetype.NOMAD]
    # the raw Rg² says the Nomads are the small ones, the wind-relative one says they are the large ones
    tab = _table(
        [
            {"count": 20, "match": nomad, "rg2_min": 1, "rg2_rel_min": 100},
            {"count": 20, "match": 0, "rg2_min": 50, "rg2_rel_min": 10},
        ]
    )  # fmt: skip
    kw: dict[str, Any] = {"arch_horizon": 100, "arch_min_obs": 20, "n_perm": 5, "seed": 0}
    rel = labels.score_archetypes(tab, wind_rel=True, **kw)["nomad"]
    raw = labels.score_archetypes(tab, wind_rel=False, **kw)["nomad"]
    assert rel["outcome"] == "rg2_rel" and rel["auc"] == 1.0 and raw["outcome"] == "rg2" and raw["auc"] == 0.0
    # both ways are reported beside the primary
    assert (rel["auc_raw"]["auc"], rel["auc_rel"]["auc"]) == (0.0, 1.0)
    assert (raw["auc_raw"]["auc"], raw["auc_rel"]["auc"]) == (0.0, 1.0)
    settler = labels.score_archetypes(tab, wind_rel=True, **kw)["settler"]
    assert settler["outcome"] == "neg_rg2_rel"
    # a table without the wind-relative columns (a world without weather) is the version-2 scoring
    plain = {k: v for k, v in tab.items() if not k.startswith(("rg2_rel", "z0", "zk", "wshare"))}
    out = labels.score_archetypes(plain, wind_rel=True, **kw)
    assert out["nomad"]["outcome"] == "rg2" and "auc_raw" not in out["nomad"]
    assert set(out["drifter"]) == {"pool", "prevalence_match", "prevalence_primary"}


def test_drifter_floater_and_sinker_outcomes() -> None:
    drifter, floater, sinker = BIT[Archetype.DRIFTER], BIT[Archetype.FLOATER], BIT[Archetype.SINKER]
    kw: dict[str, Any] = {"arch_horizon": 100, "arch_min_obs": 20, "n_perm": 5, "seed": 0, "wind_rel": True}
    kw |= {"height": 80.0, "z_floor": 8.0, "room": 1.5}
    # Drifter: a high wind share of the displacement
    tab = _table(
        [
            {"count": 15, "match": drifter, "wshare_min": 0.9, "z0": 60.0},
            {"count": 15, "match": 0, "wshare_min": 0.2, "z0": 60.0},
        ]
    )
    d = labels.score_archetypes(tab, **kw)["drifter"]
    assert d["outcome"] == "wind_share" and d["auc"] == 1.0 and (d["n"], d["n_pos"]) == (30, 15)
    # Floater: scored among the rows aloft (z0 >= 8): the altitude held, -|z_k - z_T|, in either direction
    tab = _table(
        [
            {"count": 15, "match": floater, "z0": 20.0, "zk_min": 20.2},
            {"count": 15, "match": 0, "z0": 20.0, "zk_min": 18.0},
            {"count": 10, "match": 0, "z0": 2.0, "zk_min": 0.0},  # grounded at T: not in the pool
        ]
    )
    f = labels.score_archetypes(tab, **kw)["floater"]
    assert f["outcome"] == "z_held" and f["auc"] == 1.0 and (f["n"], f["n_pos"]) == (30, 15)
    assert f["per_cohort"][0]["n"] == 30 and f["pool"] == 40
    # Sinker: the descent over the available room min(z0, 1.5), over the rows with room (z0 >= 1.5)
    tab = _table(
        [
            {"count": 15, "match": sinker, "z0": 30.0, "zk_min": 28.5},  # fell 1.5 of min(30, 1.5): 1.0
            {"count": 15, "match": 0, "z0": 30.0, "zk_min": 30.0},
            {"count": 10, "match": 0, "z0": 1.0, "zk_min": 0.0},  # no room: not in the pool
        ]
    )
    s = labels.score_archetypes(tab, **kw)["sinker"]
    assert s["outcome"] == "descent" and s["auc"] == 1.0 and (s["n"], s["n_pos"]) == (30, 15)
    # near the floor the outcome is a share of the room, not of the fall: 1.0 of 1.0 equals 1.5 of 1.5
    tab = _table(
        [
            {"count": 15, "match": sinker, "z0": 1.6, "zk_min": 0.1},  # room min(1.6, 1.5) = 1.5, fell 1.5
            {"count": 15, "match": 0, "z0": 30.0, "zk_min": 29.0},  # fell 1.0 of 1.5
        ]
    )
    assert labels.score_archetypes(tab, **kw)["sinker"]["auc"] == 1.0


def test_the_floater_is_not_credited_for_height_alone_or_for_a_direction() -> None:
    """Within the pool aloft, rows of one z band that match and rows that do not change altitude alike:
    no AUC above chance, whatever the match does (the grounded third is out of the pool); and a Floater that
    drifts up is as held as one that drifts down."""
    floater = BIT[Archetype.FLOATER]
    kw: dict[str, Any] = {"arch_horizon": 100, "arch_min_obs": 20, "n_perm": 5, "seed": 0, "wind_rel": True}
    kw |= {"height": 80.0, "z_floor": 8.0}
    tab = _table(
        [
            {"count": 20, "match": floater, "z0": 20.0, "zk_min": 19.0},
            {"count": 20, "match": 0, "z0": 20.0, "zk_min": 19.0},
            {"count": 30, "match": 0, "z0": 1.0, "zk_min": 0.0},  # the grounded third
        ]
    )
    out = labels.score_archetypes(tab, **kw)
    assert out["floater"]["auc"] == 0.5 and out["floater"]["n"] == 40
    tab = _table(
        [
            {"count": 10, "match": floater, "z0": 20.0, "zk_min": 20.5},
            {"count": 10, "match": floater, "z0": 20.0, "zk_min": 19.5},
            {"count": 20, "match": 0, "z0": 20.0, "zk_min": 17.0},
        ]
    )
    assert labels.score_archetypes(tab, **kw)["floater"]["auc"] == 1.0


# --- the study ----------------------------------------------------------------------------------------------

STUDY: dict[str, Any] = {
    "start": 50, "steps": 300, "warmup": 200, "cohort_every": 40, "arch_horizon": 60, "arch_min_obs": 20,
    "intent_horizon": 10, "n_perm": 5,
}  # fmt: skip


@pytest.fixture(scope="module")
def storm_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    cfg: Config = apply_overrides(
        storm3d_config("population.n_init=128"),
        ["run.metrics_every=5", "run.checkpoint_every=10", "run.heavy_metrics_every=5", "game.observe=true"],
    )
    rd = RunDir.create(tmp_path_factory.mktemp("storm_labels") / "run", cfg)
    assert runner.run(rd, cfg, init_state(cfg), 60, quiet=True).status == "done"
    return rd.path


@pytest.mark.slow
def test_study_on_a_storm_world_scores_the_new_labels_and_writes_the_dump(
    storm_run: Path, tmp_path: Path
) -> None:
    dump = tmp_path / "dump"
    rep = labels.study(storm_run, **STUDY, dump=dump, log=lambda _: None)
    assert rep["rows_verified"] > 0 and rep["params"]["wind_rel_outcomes"] is True
    assert rep["intent_version"] == 3 and rep["rules_digest"] == RULES_V3.digest()
    for name in ("drifter", "floater", "sinker"):
        a = rep["archetypes"][name]
        assert a["outcome"] == labels.ARCH_OUTCOMES[name] and a["verdict"] in ("pass", "fail", "insufficient")
    assert rep["archetypes"]["nomad"]["outcome"] == "rg2_rel"
    assert "auc_raw" in rep["archetypes"]["nomad"] and "auc_rel" in rep["archetypes"]["settler"]
    json.dumps(rep, allow_nan=False)
    # the dump: the pooled table, the accumulator means at the cohort step, and the 64-step input history
    z = np.load(dump / "labels_dump.npz", allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    lay = arch_layout(3, RULES_V3)
    assert meta["rules_digest"] == RULES_V3.digest() and meta["layout"]["width"] == lay.width == 14
    rows = z["arch.cohort"].size
    assert rows > 0 and z["arch.m6"].shape == (rows, 14) and z["arch.m6"].dtype == np.int32
    assert z["arch.hist"].shape == (rows, 64, 8) and z["arch.hist"].dtype == np.float16
    assert z["arch.row"].shape == (rows,) and "intent.label" in z.files
    assert np.array_equal(z["arch.primary"] != 0, np.ones(rows, bool))
    # the means in the dump are the labels' inputs: the primary label is recomputable from them
    from alife.game import rules as R

    cfg = RunDir.open(storm_run).config()
    match = np.asarray(
        R.archetype_matches(
            mx.array(z["arch.m6"]),
            mx.zeros((rows, R.layout.B, 2), mx.int32),
            cfg,
            RULES_V3,
            mx.zeros((rows,), mx.int32),
        )
    )
    bits = to_numpy(R.pack_bits(mx.array(match)))
    # every rule that reads the means alone (not the bond slots or the colony size) is recomputable from it
    keep = sum(1 << (a.value - 1) for a in (Archetype.GATHERER, Archetype.HUNTER, Archetype.PROVIDER))
    keep |= sum(
        1 << (a.value - 1) for a in (Archetype.NOMAD, Archetype.DRIFTER, Archetype.FLOATER, Archetype.SINKER)
    )
    assert np.array_equal(bits & keep, z["arch.match"] & keep)
    # the V2 rules on the same window give the comparison
    v2 = labels.study(storm_run, **STUDY, rules=RULES_V2, log=lambda _: None)
    assert v2["rules_digest"] == RULES_V2.digest() and v2["archetypes"]["nomad"]["outcome"] == "rg2"
    assert "auc_rel" in v2["archetypes"]["nomad"]


@pytest.mark.slow
def test_the_command_line_takes_the_rules_and_the_dump(storm_run: Path, tmp_path: Path) -> None:
    argv = [str(storm_run), "--from", "50", "--steps", "300", "--out", str(tmp_path / "r.json")]
    argv += ["--warmup", "200", "--cohort-every", "40", "--arch-horizon", "60", "--arch-min-obs", "20"]
    argv += ["--n-perm", "5", "--rules", "v2", "--dump", str(tmp_path / "d")]
    assert labels.main(argv) == 0
    rep = json.loads((tmp_path / "r.json").read_text())
    assert rep["rules_digest"] == RULES_V2.digest() and (tmp_path / "d" / "labels_dump.npz").is_file()
    with pytest.raises(SystemExit):
        labels.main([*argv[:-2], "--dump", str(storm_run / "dump")])  # inside the run directory
