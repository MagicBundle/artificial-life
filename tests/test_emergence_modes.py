"""Block A of the emergence suite (DESIGN §44.5): MODES on phenotype skeletons against planted histories, the
codebook, and the activity adapter over analysis/report.py. CPU, numpy only."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from alife.analysis import phylo, report
from alife.analysis import w2 as w2mod
from alife.analysis.emergence import activity_adapter as aa
from alife.analysis.emergence import frames as fm
from alife.analysis.emergence import modes
from alife.analysis.emergence.__main__ import main

FEATS = ["buoy_cap", "offspring_frac", "f0", "f1", "f2", "f3"]
NAMES = ["tag0", "tag1", "tag2", "tag3", *FEATS, "energy", "n_bonds", "speed"]


def codebook() -> modes.Codebook:
    """Six columns with cut points at 1 and 2: a trait value ``d + 0.5`` is bin ``d``."""
    return modes.Codebook(
        columns=list(FEATS),
        tag_mean=[0.0] * 4,
        tag_pc=[1.0, 0.0, 0.0, 0.0],
        edges=[[1.0, 2.0]] * 6,
        channel_thresholds={"move": 0.5},
        energy_edges=[1.0, 2.0],
        height_bands=10,
    )


def digits(c: np.ndarray) -> np.ndarray:
    return np.stack([(c // 3**k) % 3 for k in range(6)], axis=1).astype(np.float32) + 0.5


def history(
    rng: np.random.Generator,
    classes_per_frame: list[np.ndarray],
    *,
    n: int = 2000,
    step: int = 100,
    bonded: bool = False,
) -> fm.Frames:
    """Frames where organisms of frame ``f`` take classes from ``classes_per_frame[f]`` (array of class ids,
    one per organism)."""
    nf = len(classes_per_frame)
    cols = []
    uid = []
    pos = []
    for f, c in enumerate(classes_per_frame):
        d = digits(c)
        t = np.zeros((len(c), len(NAMES)), np.float32)
        for j, nm in enumerate(FEATS):
            t[:, NAMES.index(nm)] = d[:, j]
        t[:, NAMES.index("energy")] = 1.5
        t[:, NAMES.index("speed")] = rng.uniform(0, 1, len(c))
        cols.append(t)
        uid.append(np.arange(len(c)) + f * 10_000_000)
        pos.append(
            np.column_stack(
                [rng.uniform(0, 160, len(c)), rng.uniform(0, 160, len(c)), rng.uniform(0, 80, len(c))]
            )
        )
    off = np.concatenate([[0], np.cumsum([len(c) for c in classes_per_frame])]).astype(np.int64)
    allp = np.concatenate(pos)
    body = (np.arange(len(allp)) // 4).astype(np.int32) if bonded else np.full(len(allp), -1, np.int32)
    return fm.Frames(
        meta={"size": 160.0, "height": 80.0},
        step=np.arange(nf, dtype=np.int64) * step,
        off=off,
        uid=np.concatenate(uid).astype(np.int64),
        row=np.zeros(len(allp), np.int32),
        pos=fm.quantise(allp, 160.0, 80.0),
        clade=np.zeros(len(allp), np.int32),
        body=body,
        trait_names=NAMES,
        traits=np.concatenate(cols),
    )


def test_planted_novelty_is_recovered_and_churn_is_not_novelty() -> None:
    rng = np.random.default_rng(1)
    cb = codebook()
    nf, n, period = 81, 2000, 10
    active = [0, 1, 2, 3, 4]
    frames_c = []
    for f in range(nf):
        if f > 0 and f % period == 0:
            active.append(100 + f)  # a new class persists from now on
        c = rng.choice(active, n)
        c[:3] = rng.integers(0, 729, 3)  # three one-off organisms per frame: churn below the 0.1% threshold
        frames_c.append(c)
    fr = history(rng, frames_c, n=n)
    seen = np.zeros(cb.n_codes, bool)
    seen[[0, 1, 2, 3, 4]] = True
    r = modes.analyse(fr, cb, seen=seen, n_blocks=8)
    per = r["novelty"]["per_block"]
    # eight introductions at frames 10..80; the last needs a second frame to be persistent, so seven count
    assert sum(per) == 7 and max(per) == 1
    # churn alone is no novelty
    only = [rng.choice([0, 1, 2, 3, 4], n) for _ in range(nf)]
    for c in only:
        c[:3] = rng.integers(0, 729, 3)
    r0 = modes.analyse(history(rng, only, n=n), cb, seen=seen)
    assert sum(r0["novelty"]["per_block"]) == 0
    # the shuffled-frame baseline moves the new classes to random blocks
    assert r["novelty"]["shuffled_frames_mean"] != per


def test_hill_numbers_on_known_abundances() -> None:
    m = 7
    counts = np.zeros(729, np.int64)
    counts[:m] = 100
    assert modes.hill(counts, counts > 0) == pytest.approx((m, m, m))
    counts[0] = 700  # unequal: D0 > D1 > D2
    d0, d1, d2 = modes.hill(counts, counts > 0)
    assert d0 > d1 > d2 > 1
    rng = np.random.default_rng(2)
    fr = history(rng, [rng.integers(0, m, 3500) for _ in range(6)], n=3500)
    e = modes.analyse(fr, codebook(), n_blocks=2)["ecology"]
    assert e["hill0"][0] == pytest.approx(m) and e["hill1"][0] == pytest.approx(m, rel=0.02)


def test_change_per_generation_is_recovered_and_unstructured_equals_its_baseline() -> None:
    rng = np.random.default_rng(3)
    cb = codebook()
    n, q = 20_000, 0.2
    parent_c = rng.integers(0, 729, n)
    child_c = np.where(rng.random(n) < q, rng.integers(0, 729, n), parent_c)  # a flip changes the class
    flipped = child_c != parent_c
    fr = history(rng, [parent_c, child_c], n=n)
    births = {"uid": fr.uid[n:], "parent_uid": fr.uid[:n]}
    codes = modes.skeletons(fr, cb)
    r = modes.change_edge(fr, codes, births, 1)
    assert r["share"] == pytest.approx(flipped.mean(), abs=1e-9)
    assert r["share"] == pytest.approx(q, rel=0.1)
    assert r["coverage"] == 1.0
    base = modes.change_edge(fr, codes, births, 1, np.random.default_rng(0))
    assert base["share"] > 0.95  # permuting children destroys the association
    # unstructured history: the child ignores its parent, observed equals the baseline
    unstructured = history(rng, [parent_c, rng.integers(0, 729, n)], n=n)
    cu = modes.skeletons(unstructured, cb)
    obs = modes.change_edge(unstructured, cu, births, 1)["share"]
    bas = modes.change_edge(unstructured, cu, births, 1, np.random.default_rng(1))["share"]
    assert obs == pytest.approx(bas, abs=0.02)


def test_change_pop_is_zero_for_a_constant_mix_and_positive_for_a_shift() -> None:
    rng = np.random.default_rng(4)
    cb = codebook()
    same = history(rng, [rng.integers(0, 4, 4000) for _ in range(14)], n=4000)
    c0 = modes.analyse(same, cb, n_blocks=1)["change"]["pop"][0]
    assert c0 < 0.05
    shift = history(rng, [rng.integers(0, 4, 4000) + (4 if f >= 7 else 0) for f in range(14)], n=4000)
    assert modes.analyse(shift, cb, n_blocks=1)["change"]["pop"][0] > 0.3


def test_codebook_is_shared_and_round_trips(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    ref = history(rng, [rng.integers(0, 729, 1500) for _ in range(4)], n=1500)
    # the base tags give the first principal component: give them structure
    ref.traits[:, :4] = rng.normal(size=(len(ref.uid), 4)) * [3, 1, 0.5, 0.2]
    cb = modes.build_codebook([ref])
    assert (
        cb.columns[0] == "tag_pc1"
        and cb.columns[1:3] == ["buoy_cap", "offspring_frac"]
        and len(cb.columns) == 6
    )
    assert abs(sum(p * p for p in cb.tag_pc) - 1.0) < 1e-9 and abs(cb.tag_pc[0]) > 0.9
    path = modes.write_codebook(tmp_path / "codebook.json", cb)
    cb2 = modes.read_codebook(path)
    assert cb2.to_json() == cb.to_json()
    other = history(rng, [rng.integers(0, 729, 1500) for _ in range(4)], n=1500)
    other.traits[:, :4] = rng.normal(size=(len(other.uid), 4))
    assert np.array_equal(modes.skeletons(other, cb), modes.skeletons(other, cb2))
    bad = cb.to_json() | {"version": 99}
    with pytest.raises(ValueError):
        modes.Codebook.from_json(bad)
    assert modes.skeletons(other, cb).max() < cb.n_codes
    with pytest.raises(ValueError):
        modes.build_codebook(
            [
                history(rng, [rng.integers(0, 9, 100)], n=100).__class__(
                    meta=ref.meta,
                    step=ref.step[:1],
                    off=ref.off[:2],
                    uid=ref.uid[:1500],
                    row=ref.row[:1500],
                    pos=ref.pos[:1500],
                    clade=ref.clade[:1500],
                    body=ref.body[:1500],
                    trait_names=["energy"],
                    traits=ref.traits[:1500, :1],
                )
            ]
        )


def test_channels_complexity_and_lineage() -> None:
    rng = np.random.default_rng(6)
    cb = codebook()
    fr = history(rng, [rng.integers(0, 9, 1000) for _ in range(3)], n=1000)
    j = NAMES.index("n_bonds")
    fr.traits[:, j] = (np.arange(len(fr.uid)) % 2).astype(np.float32)  # half bonded
    ch, used = modes.channel_counts(fr, cb)
    assert used == ["bond", "move"] and ch.max() == 2 and ch.min() == 0
    modes.register_channel("probe", "f0", "pos")
    try:
        ch2, used2 = modes.channel_counts(fr, cb)
        assert used2[-1] == "probe" and ch2.max() == 3
    finally:
        modes.CHANNELS.pop("probe")
    with pytest.raises(ValueError):
        modes.register_channel("x", "f0", "bad")
    assert modes.lineage_max({1: 1, 2: 3, 3: 2}, {3: 2, 2: 1}, [3]) == 3
    assert modes.lineage_max({}, {}, [1]) is None


def test_ecology_pairs_and_niches() -> None:
    rng = np.random.default_rng(7)
    c = rng.integers(0, 3, 4000)
    fr = history(rng, [c] * 11, n=4000, bonded=True)
    e = modes.analyse(fr, codebook(), n_blocks=1)["ecology"]
    assert e["interaction_pairs"][0] >= 3  # all class pairs and self pairs co-occur in bodies often enough
    assert e["niches"][0] >= 5  # height bands x one energy tercile


# --- the activity adapter over report.py ---


def population(
    n0: int, steps: int, b: int, rng: np.random.Generator, select: float, gap: int = 5
) -> phylo.Tree:
    n = n0 + (steps - gap) * b
    parent = np.full(n, -1, np.int32)
    birth = np.full(n, -1, np.int32)
    death = np.full(n, steps, np.int32)
    z = np.zeros(n)
    alive = list(range(n0))
    nxt = n0
    for t in range(gap, steps):
        for i in sorted(rng.choice(len(alive), b, replace=False), reverse=True):
            death[alive[i]] = t
            alive[i] = alive[-1]
            alive.pop()
        pool = np.array([a for a in alive if birth[a] <= t - gap], dtype=np.int64)
        w = np.exp(select * (z[pool] - z[pool].max()))
        for p in rng.choice(pool, b, p=w / w.sum()):
            parent[nxt], birth[nxt], z[nxt] = p, t, z[p] + rng.normal(0, 0.1)
            alive.append(nxt)
            nxt += 1
    return phylo.Tree(parent, birth, death, steps)


def small_settings() -> report.Settings:
    return report.Settings(
        grid_every=100, marker_rates=(0.02,), cohort_horizons=(500,), shadows=1, windows=()
    )


@pytest.fixture(scope="module")
def activity_flats() -> dict[str, tuple[dict[str, np.ndarray], report.Settings]]:
    out = {}
    for name, sel, seed in (("neutral", 0.0, 1), ("selected", 8.0, 2)):
        st = small_settings()
        tree = population(300, 3000, 2, np.random.default_rng(seed), sel)
        flat, _ = report.compute(tree, st, log=lambda m: None)
        out[name] = (flat, st)
    return out


def test_adapter_reads_p1_from_report_arrays(activity_flats) -> None:
    blk = w2mod.Blocking(0, 600, 300, 8)
    res = {}
    for name, (flat, st) in activity_flats.items():
        summary = report.summarise(flat, st)
        ep = aa.endpoint(flat, st, summary, 0, space="cohort_h500", blocking=blk)
        assert len(ep["blocks"]) == 8 and ep["space"] == "cohort_h500" and ep["valid"]
        assert ep["channon_class"] in report.CLASSES
        res[name] = ep
    # selection raises new normalised activity above the neutral shadow; a neutral tree has none above it
    sel = [x for x in res["selected"]["excess_blocks"] if x is not None]
    neu = [x for x in res["neutral"]["excess_blocks"] if x is not None]
    assert np.mean(sel) > np.mean(neu)
    # the series is the one report.py's classification reads
    flat, st = activity_flats["selected"]
    s = aa.p1_series(flat, st, "cohort_h500")
    assert s["theta"] is not None and s["real"].shape == s["grid"].shape


def test_adapter_picks_a_space_and_runs_end_to_end(activity_flats, monkeypatch, tmp_path: Path) -> None:
    flat = dict(activity_flats["selected"][0])
    flat["real/marker_0.01/D"] = flat["real/marker_0.02/D"]
    assert aa.pick_space(flat, 100_000) == "marker_0.01" and aa.pick_space(flat, 1_000_000) == "marker_0.01"
    flat["real/cohort_h200k/D"] = flat["real/marker_0.01/D"]
    assert aa.pick_space(flat, 1_000_000) == "cohort_h200k" and aa.pick_space(flat, 10_000) == "marker_0.01"
    with pytest.raises(KeyError):
        aa.pick_space({}, 1)
    tree = population(300, 1500, 2, np.random.default_rng(3), 8.0)
    cov = {"t0": 0, "t_end": 1500, "min_parent_child_gap": 5}
    monkeypatch.setattr(phylo, "load_tree", lambda run, start=None: (tree, cov))
    r = aa.analyse_arm(
        tmp_path,
        st=small_settings(),
        space="marker_0.02",
        out=tmp_path / "o",
        blocking=w2mod.Blocking(0, 300, 150, 8),
    )
    assert (tmp_path / "o" / "activity.npz").is_file() and r["timing_s"]["total"] > 0
    assert len(r["endpoint"]["blocks"]) == 8
    r2 = aa.analyse_arm(
        tmp_path,
        st=small_settings(),
        space="marker_0.02",
        out=tmp_path / "o",
        blocking=w2mod.Blocking(0, 300, 150, 8),
    )
    assert r2["endpoint"]["blocks"] == r["endpoint"]["blocks"]  # reuse of the cached arrays
    assert all(math.isfinite(v) for v in r["timing_s"].values())


# --- CLI and panel ---


def test_cli_modes_and_panel(tmp_path: Path) -> None:
    rng = np.random.default_rng(8)
    fr_a = history(rng, [rng.integers(0, 9, 1500) for _ in range(33)], n=1500)
    fr_a.traits[:, :4] = rng.normal(size=(len(fr_a.uid), 4))
    fr_a.meta.update({"run": "toy", "window": [0, 3200], "frames_version": 1})
    ref = fm.write_frames(tmp_path / "ref.npz", fr_a)
    fr_b = history(rng, [rng.integers(0, 9, 1500) for _ in range(33)], n=1500)
    fr_b.traits[:, :4] = rng.normal(size=(len(fr_b.uid), 4))
    fr_b.meta.update({"run": "toy", "window": [4000, 7200], "frames_version": 1})
    win = fm.write_frames(tmp_path / "win.npz", fr_b)
    out = tmp_path / "m"
    assert main(["modes", str(win), "--reference", str(ref), "--out", str(out)]) == 0
    d = json.loads((out / "modes.json").read_text())
    assert d["windows"][0]["status"] == "exploratory" and (out / "codebook.json").is_file()
    assert (
        main(["modes", str(win), "--codebook", str(out / "codebook.json"), "--out", str(tmp_path / "m2")])
        == 0
    )
    assert main(["modes", str(win), "--out", str(tmp_path / "m3")]) == 2
    run = tmp_path / "run"
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({"seed": 1, "config_hash": "h", "start_step": 0}))
    (run / "metrics.jsonl").write_text("")
    pout = tmp_path / "p"
    assert (
        main(
            [
                "panel",
                str(run),
                "--frames",
                str(win),
                "--windows",
                "4000:7200",
                "--reference",
                str(ref),
                "--out",
                str(pout),
            ]
        )
        == 0
    )
    p = json.loads((pout / "panel.json").read_text())
    assert p["A"]["status"] == "ok" and p["A"]["modes"]["windows"]
    for k in ("P2", "P3"):
        assert p["endpoints"][k]["status"] == "ok" and len(p["endpoints"][k]["blocks"]) == 16
        assert p["endpoints"][k]["confirmatory"] is False
    assert p["endpoints"]["P1"]["status"] == "needs_missing"  # no activity requested
    assert p["A"]["needs_missing"] == ["activity (run events)"]


def test_cli_activity_error_on_a_missing_run(tmp_path: Path) -> None:
    assert main(["activity", str(tmp_path / "nope"), "--out", str(tmp_path / "o")]) == 2


# --- review fixes ---


def test_extended_codebook_sees_a_feature_the_reference_lacks() -> None:
    rng = np.random.default_rng(40)
    cb = codebook()
    narrow = modes.replace(cb, columns=FEATS[:4], edges=cb.edges[:4])
    c = rng.integers(0, 81, 3000)  # classes over the first four columns
    ref = history(rng, [c] * 6, n=3000)
    arm_cls = c + 81 * rng.choice([0, 2], 3000) * 3  # f2 digit 0 or 2, f3 digit 0: two feature levels
    arm = history(rng, [arm_cls] * 6, n=3000)
    seen = modes.persistent_classes(ref, narrow)
    plain = modes.analyse(arm, narrow, seen=seen, n_blocks=2)
    cbx = modes.extend_codebook(narrow, arm)
    assert cbx.columns == [*FEATS[:4], "f2", "f3"] and cbx.meta["extended"]
    ext = modes.analyse(arm, cbx, seen=modes.pad_seen(seen, cbx), n_blocks=2)
    assert plain["classes_present"] < ext["classes_present"]
    assert sum(ext["novelty"]["per_block"]) > sum(plain["novelty"]["per_block"])
    assert modes.extend_codebook(cbx, arm) is cbx  # nothing left to add
    assert (
        len(modes.pad_seen(seen, cbx)) == cbx.n_codes
        and modes.pad_seen(seen, cbx)[: len(seen)].tolist() == seen.tolist()
    )


def test_activity_cache_is_keyed_by_a_fingerprint(monkeypatch, tmp_path: Path) -> None:
    tree = population(300, 1500, 2, np.random.default_rng(41), 8.0)
    cov = {"t0": 0, "t_end": 1500, "min_parent_child_gap": 5}
    monkeypatch.setattr(phylo, "load_tree", lambda run, start=None: (tree, cov))
    blk = w2mod.Blocking(0, 300, 150, 8)
    one = report.Settings(grid_every=100, marker_rates=(0.02,), cohort_horizons=(500,), shadows=1, windows=())
    two = report.Settings(grid_every=100, marker_rates=(0.02,), cohort_horizons=(500,), shadows=2, windows=())
    out = tmp_path / "o"
    aa.analyse_arm(tmp_path, st=one, space="marker_0.02", out=out, blocking=blk)
    assert "neutral2/grid" not in np.load(out / "activity.npz").files
    aa.analyse_arm(tmp_path, st=two, space="marker_0.02", out=out, blocking=blk)  # other shadows: recomputed
    assert "neutral2/grid" in np.load(out / "activity.npz").files
    cov["t_end"] = 1400  # other end of the events: recomputed again
    r = aa.analyse_arm(tmp_path, st=two, space="marker_0.02", out=out, blocking=blk)
    assert "real" in r["timing_s"]


def test_compare_refuses_p1_in_different_spaces() -> None:
    from alife.analysis.emergence import panel_blocks

    def panel_with(space: str) -> dict:
        return {"endpoints": {"P1": {"status": "ok", "space": space}}}

    panel_blocks.check_p1_spaces([panel_with("cohort_h200k"), panel_with("cohort_h200k")])
    with pytest.raises(panel_blocks.SpaceMismatch):
        panel_blocks.check_p1_spaces([panel_with("cohort_h200k"), panel_with("marker_0.01")])
    panel_blocks.check_p1_spaces(
        [panel_with("cohort_h200k"), {"endpoints": {"P1": {"status": "needs_missing"}}}]
    )
