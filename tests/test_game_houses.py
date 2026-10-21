"""§29.4 houses at run level: the live house of every agent equals the offline ``marker_roots`` definition
on a run's phylogeny, the sidecars and the social log of a run are deterministic, and a fork's houses are
seeded from clades (exact only for a fork from step 0)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alife import cli, runner
from alife.analysis import phylo
from alife.config import Config, apply_overrides
from alife.game.social import HOUSE_UNKNOWN
from alife.runio import RunDir, load_checkpoint
from alife.state import init_state, to_numpy
from game_helpers import social_every

IMMIGRATION = ["population.immigration_every=5", "population.immigration_count=4"]


def _observing(cfg: Config, *extra: str) -> Config:
    return apply_overrides(cfg, ["game.observe=true", "run.checkpoint_every=100", *extra])


def _run(cfg: Config, path: Path, steps: int) -> RunDir:
    rd = RunDir.create(path, cfg)
    rd.update_meta(start_step=0)
    res = runner.run(rd, cfg, init_state(cfg), steps, quiet=True)
    assert (res.status, res.exit_code) == ("done", runner.ExitCode.OK)
    return rd


@pytest.mark.parametrize("rate", [0.0, 0.125, 1.0], ids=["rate-0", "rate-1-8", "rate-1"])
def test_houses_equal_offline_marker_roots(
    rate: float, tiny_cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tiny run from step 0 with immigration, 300 steps: in the sidecars at 100, 200 and 300, ``soc.house``
    of every alive row is the uid of the nearest root of its lineage under ``phylo.marker_roots`` of the
    run's phylogeny at the same rate and seed (a house never changes in a life, so the end-of-run tree serves
    every step); every head and record says the houses are exact."""
    social_every(monkeypatch, 50)
    cfg = _observing(tiny_cfg, *IMMIGRATION, f"game.house_rate={rate}", "run.seed=7")
    rd = _run(cfg, tmp_path / "run", 300)
    tree, report = phylo.load_tree(rd.path)
    assert report["chunks_tile"] and report["checkpoint_check"]["ok"]
    roots = phylo.marker_roots(tree, rate, cfg.run.seed)
    near = phylo.nearest_root(
        tree.parent, roots
    )  # uid of the nearest root, per uid (the tree index is the uid)
    n_init = cfg.population.n_init
    seen = set()
    for step in (100, 200, 300):
        side = rd.read_observers(step)
        assert side is not None
        uid, house = side["soc.uid"], side["soc.house"]
        alive = uid >= 0
        assert alive.sum() > 20 and (house[~alive] == -1).all()
        assert np.array_equal(house[alive], near[uid[alive]].astype(np.int64)), (rate, step)
        assert np.array_equal(uid, side["uid"])
        assert (house[alive] != HOUSE_UNKNOWN).all()  # from step 0 every lineage has a house
        seen |= set(house[alive].tolist())
    # which kinds of house the run produced: founders and immigrants always; hashed births at 0 < rate
    parent_of = {h: int(tree.parent[h]) for h in seen}
    hashed = [h for h in seen if h >= n_init and parent_of[h] >= 0]
    immigrant = [h for h in seen if h >= n_init and parent_of[h] < 0]
    if rate == 0.0:
        assert not hashed and immigrant  # no birth founds a house: founders and immigrants only
    else:
        assert hashed and immigrant  # at 1/8 and at 1: some house is neither a founder nor an immigrant
    records = rd.social_records()
    assert [r["step"] for r in records] == [50, 100, 150, 200, 250, 300]
    assert all(
        r["summary"]["houses"]["exact"] and not r["summary"]["houses"]["unknown_immigrant"] for r in records
    )
    assert all(r["observers"]["social"]["houses"]["exact"] is True for r in rd.metrics_rows())
    assert all(r["origin"] == 0 and r["window_start"] == r["step"] - 50 for r in records)


def test_houses_are_deterministic(tiny_cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two identical runs give identical sidecars (``soc.*`` and ``host.*`` included) and identical social
    log bytes."""
    social_every(monkeypatch, 15)
    cfg = _observing(tiny_cfg, *IMMIGRATION, "game.house_rate=0.125", "run.checkpoint_every=30")
    a, b = _run(cfg, tmp_path / "a", 90), _run(cfg, tmp_path / "b", 90)
    assert [st for st, _ in a.observer_files()] == [st for st, _ in b.observer_files()] == [0, 30, 60, 90]
    for step in (0, 30, 60, 90):
        x, y = a.read_observers(step), b.read_observers(step)
        assert x is not None and y is not None and x.keys() == y.keys()
        assert any(k.startswith("soc.") for k in x) and any(k.startswith("host.") for k in x)
        for k in x:
            if k == "meta":  # the version of the code is the same; compare it whole too
                assert str(x[k]) == str(y[k])
            else:
                assert x[k].dtype == y[k].dtype and np.array_equal(x[k], y[k]), (step, k)
    assert a.social_path.read_bytes() == b.social_path.read_bytes() and len(a.social_records()) == 6


def test_houses_of_a_fork_are_seeded_from_clades(
    tiny_cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fork taken after immigrants arrived starts fresh observers: on its alive rows the house is the clade
    for founder lineages and ``HOUSE_UNKNOWN`` for immigrant lineages, and its heads and records say the
    houses are inexact. Its social log starts absent (the source's is neither copied nor read), and its first
    record has the fork step as its origin and window start. A fork from step 0 holds founders only: exact."""
    social_every(monkeypatch, 15)
    cfg = _observing(tiny_cfg, *IMMIGRATION, "run.checkpoint_every=30")
    src = _run(cfg, tmp_path / "src", 60)
    log = src.social_path.read_bytes()
    n_init = cfg.population.n_init

    out = tmp_path / "fork"
    argv = ["run", "--init-from", f"{src.path}:30", "--out", str(out), "--quiet"]
    assert cli.main([*argv, "--steps", "0"]) == 0
    fork = RunDir.open(out)
    assert not fork.social_path.exists() and fork.social_records() == []  # absent until a snapshot
    side = fork.read_observers(30)
    assert side is not None
    state = load_checkpoint(fork.checkpoint_at(30), fork.config())
    alive = to_numpy(state.alive)
    clade = to_numpy(state.clade)[alive]
    house = side["soc.house"][alive]
    assert (to_numpy(state.uid)[alive] == side["soc.uid"][alive]).all()
    immigrant = clade == n_init  # immigrants and all their descendants carry clade n_init
    assert immigrant.any() and (~immigrant).any()
    assert (house[immigrant] == HOUSE_UNKNOWN).all()
    assert (house[~immigrant] == clade[~immigrant]).all()  # the founders' house is their clade
    assert src.social_path.read_bytes() == log  # the source's log is untouched

    out2 = tmp_path / "fork_run"
    assert (
        cli.main(["run", "--init-from", f"{src.path}:30", "--out", str(out2), "--quiet", "--steps", "30"])
        == 0
    )
    fork = RunDir.open(out2)
    records = fork.social_records()
    assert [r["step"] for r in records] == [45, 60]  # only the fork's own records
    assert (records[0]["origin"], records[0]["window_start"]) == (30, 30)
    assert all(not r["summary"]["houses"]["exact"] for r in records)
    assert records[0]["summary"]["houses"]["unknown_immigrant"] > 0
    assert all(r["observers"]["social"]["houses"]["exact"] is False for r in fork.metrics_rows())
    assert src.social_path.read_bytes() == log and len(src.social_records()) == 4

    # a fork from step 0 starts at origin 0 with founders only: its houses are exact
    zero = tmp_path / "fork0"
    assert (
        cli.main(["run", "--init-from", f"{src.path}:0", "--out", str(zero), "--quiet", "--steps", "30"]) == 0
    )
    f0 = RunDir.open(zero)
    assert [(r["step"], r["origin"]) for r in f0.social_records()] == [(15, 0), (30, 0)]
    assert all(r["summary"]["houses"]["exact"] for r in f0.social_records())
    assert all(r["observers"]["social"]["houses"]["exact"] is True for r in f0.metrics_rows())
    assert src.social_path.read_bytes() == log
