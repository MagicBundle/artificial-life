"""Houses at forks (DESIGN §33.3.5): the exact rebuild of a planet's houses from its source's birth chain
(``analysis/houses.py``), the house file (``game/housefile.py``), ``Observer.fresh(houses=)``, the file read
by ``Observer.open`` (itself, or folded over the planet's own births), and ``alife play --houses``.

The source is a 600-step run of the 2D ``tiny`` world from step 0 with immigration and a house rate of 1/4, so
that a sidecar of the *live* observer holds the houses every rebuild must reproduce: founders, immigrants
(each their own house) and hash-founded births (with their house of origin)."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from alife import cli, groups, play, runner
from alife.analysis import houses, phylo
from alife.config import Config, apply_overrides
from alife.game import housefile
from alife.game.housefile import HouseFileError, HouseSeed
from alife.game.observer import Observer
from alife.runio import RunDir, load_checkpoint
from alife.state import init_state, to_numpy
from game_helpers import social_every
from play_decode import decode

RATE, SEED, SNAP = 0.25, 7, 50
CADENCE = {"metrics_every": 10, "heavy_metrics_every": 10, "checkpoint_every": 10, "keep_checkpoints": 3}
EARLY, FORK, END = 100, 300, 600  # checkpoints of the source; the planets are made at FORK


def _cfg(tiny_cfg: Config, *extra: str) -> Config:
    return apply_overrides(
        tiny_cfg,
        [
            "game.observe=true",
            "run.checkpoint_every=100",
            "run.keep_checkpoints=8",
            "population.immigration_every=5",
            "population.immigration_count=4",
            f"game.house_rate={RATE}",
            f"run.seed={SEED}",
            *extra,
        ],
    )


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory, tiny_cfg: Config) -> Iterator[Path]:
    """The source run: observed from step 0 to 600, checkpoints at 0, 100, ..., 600 and their sidecars."""
    cfg = _cfg(tiny_cfg)
    with pytest.MonkeyPatch.context() as mp:
        social_every(mp, SNAP)
        rd = RunDir.create(tmp_path_factory.mktemp("housesrc") / "src", cfg)
        rd.update_meta(start_step=0)
        res = runner.run(rd, cfg, init_state(cfg), END, quiet=True)
    assert (res.status, res.exit_code) == ("done", runner.ExitCode.OK)
    assert [s for s, _ in rd.checkpoints()] == list(range(0, END + 1, 100))
    yield rd.path


@pytest.fixture
def cfg(tiny_cfg: Config) -> Config:
    return _cfg(tiny_cfg)


@pytest.fixture
def snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    social_every(monkeypatch, SNAP)


def live(rd: RunDir, step: int) -> HouseSeed:
    """The houses the live observer held at ``step``: ``soc.house`` and ``soc.house_from`` of the alive rows,
    by ascending uid."""
    side = rd.read_observers(step)
    assert side is not None
    uid = side["soc.uid"]
    alive = uid >= 0
    order = np.argsort(uid[alive])
    return HouseSeed(
        uid[alive][order].astype(np.int64),
        side["soc.house"][alive][order].astype(np.int64),
        side["soc.house_from"][alive][order].astype(np.int64),
    )


def assert_same(a: HouseSeed, b: HouseSeed, what: str = "") -> None:
    for name in HouseSeed._fields:
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype == y.dtype == np.int64 and np.array_equal(x, y), (what, name)


def alive_part(h: HouseSeed, alive_uid: np.ndarray) -> HouseSeed:
    """``h`` restricted to the uids in ``alive_uid`` (a fold also holds the births that died)."""
    keep = np.isin(h.uid, alive_uid)
    return HouseSeed(h.uid[keep], h.house[keep], h.house_from[keep])


def rebuild(source: Path, step: int, cfg: Config) -> HouseSeed:
    return houses.rebuild(source, step, rate=cfg.game.house_rate, seed=cfg.run.seed)


# --- the rebuild ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("step", [EARLY, FORK, END])
def test_the_rebuild_equals_the_live_houses_of_a_run_from_step_0(
    source: Path, cfg: Config, step: int
) -> None:
    """Founders, immigrants and hash-founded births: house *and* house of origin of every alive agent equal
    the live observer's. Steps below the end have later events in the logs."""
    got, want = rebuild(source, step, cfg), live(RunDir.open(source), step)
    assert_same(got, want, step)
    n_init = cfg.population.n_init
    imm = (got.house >= n_init) & (
        got.house_from == -1
    )  # an immigrant lineage: founded by a parentless birth
    hashed = got.house_from >= 0
    assert imm.sum() > 5 and hashed.sum() > 5 and (got.house[~imm & ~hashed] < n_init).all()


def test_the_rebuild_follows_the_planets_rate_and_seed(source: Path, cfg: Config) -> None:
    """One rule applies to the past and to the future: another rate or seed is another partition."""
    base = rebuild(source, FORK, cfg)
    other_rate = houses.rebuild(source, FORK, rate=0.5, seed=SEED)
    other_seed = houses.rebuild(source, FORK, rate=RATE, seed=SEED + 1)
    none = houses.rebuild(source, FORK, rate=0.0, seed=SEED)
    for other in (other_rate, other_seed):
        assert np.array_equal(other.uid, base.uid) and not np.array_equal(other.house, base.house)
    assert (none.house_from[none.house >= cfg.population.n_init] == -1).all()  # only founders and immigrants
    assert np.unique(other_rate.house).size > np.unique(base.house).size


def test_the_summary_counts_the_houses(source: Path, cfg: Config) -> None:
    h = rebuild(source, FORK, cfg)
    s = houses.summary(h, top=3)
    assert s["alive"] == h.uid.size and s["houses"] == np.unique(h.house).size
    assert [t["size"] for t in s["top"]] == sorted((t["size"] for t in s["top"]), reverse=True)
    assert sum(np.count_nonzero(h.house == t["id"]) for t in s["top"]) == sum(t["size"] for t in s["top"])


def _random_tree(n: int, n_init: int, imm_every: int, depth: int, seed: int) -> phylo.Tree:
    """A phylogeny with indices = uids: ``n_init`` founders, an immigrant every ``imm_every``-th birth, every
    other birth a child of one of the ``depth`` agents before it (deep chains)."""
    rng = np.random.default_rng(seed)
    parent = np.full(n, -1, np.int32)
    for i in range(n_init, n):
        if (i - n_init) % imm_every:
            parent[i] = rng.integers(max(0, i - depth), i)
    birth = np.where(parent < 0, np.where(np.arange(n) < n_init, -1, 1), 1).astype(np.int32)
    return phylo.Tree(parent, birth, np.full(n, 5, np.int32), 5)


@pytest.mark.parametrize("rate", [0.0, 0.05, 0.3, 1.0])
def test_houses_of_a_deep_phylogeny_follow_the_rule_birth_by_birth(rate: float) -> None:
    """The house and the house of origin of every agent of a tree 2,000 births deep, against the rule of
    ``social.update`` applied one birth at a time (which is how the live observer got them)."""
    tree = _random_tree(2500, 50, imm_every=12, depth=6, seed=2)
    alive = np.arange(0, tree.n, 3, dtype=np.int64)
    got = houses.houses_of(tree, alive, rate=rate, seed=SEED)
    empty = HouseSeed(*(np.zeros(0, np.int64),) * 3)
    want = _naive_fold(empty, np.arange(tree.n, dtype=np.int64), tree.parent.astype(np.int64), SEED, rate)
    assert got.uid.tolist() == alive.tolist()
    assert [(int(h), int(f)) for h, f in zip(got.house, got.house_from, strict=True)] == [
        want[int(u)] for u in alive
    ]
    if 0.0 < rate < 1.0:  # the origin is a real house of an older agent, and often not a root's own uid
        assert (got.house_from > -1).sum() > 100 and (got.house_from != tree.parent[got.house]).any()


def test_houses_of_a_tree_folded_forward_are_the_houses_of_the_longer_tree() -> None:
    """The planet's two ways to the same houses: rebuilt at the later step, or rebuilt at the fork and folded
    over the births since (what ``Observer.open`` does for a lost sidecar)."""
    rng = np.random.default_rng(5)
    cut, n = 1800, 3000
    head = _random_tree(cut, 40, imm_every=9, depth=8, seed=5)
    alive_cut = np.arange(
        0, cut, 2, dtype=np.int64
    )  # who lives at the cut; the births since descend from them
    later = np.arange(cut, n, dtype=np.int64)
    pool = np.concatenate([alive_cut, later])
    parent = np.array(
        [
            -1
            if k % 9 == 0
            else pool[rng.integers(0, alive_cut.size)]
            if rng.random() < 0.2
            else pool[rng.integers(max(0, alive_cut.size + k - 8), alive_cut.size + k)]
            for k in range(later.size)
        ],
        np.int64,
    )
    full_parent = np.concatenate([head.parent, parent]).astype(np.int32)
    full = phylo.Tree(full_parent, np.ones(n, np.int32), np.full(n, 5, np.int32), 5)
    seed = houses.houses_of(head, alive_cut, rate=0.2, seed=SEED)
    folded = housefile.fold(seed, later, parent, run_seed=SEED, rate=0.2)
    assert_same(folded, houses.houses_of(full, folded.uid, rate=0.2, seed=SEED))
    assert folded.uid.size == alive_cut.size + later.size and (folded.house_from > -1).sum() > 100


# --- what makes a source unusable -----------------------------------------------------------------------


def _copy(source: Path, tmp_path: Path, *skip: str) -> RunDir:
    dst = tmp_path / "copy"
    shutil.copytree(source, dst, ignore=shutil.ignore_patterns(*skip) if skip else None)
    return RunDir.open(dst)


def test_a_complete_source_has_no_problem(source: Path) -> None:
    rd = RunDir.open(source)
    assert [houses.source_problem(rd, s) for s in (EARLY, FORK, END)] == [None, None, None]


def test_a_source_without_events_is_unusable(source: Path, tmp_path: Path, cfg: Config) -> None:
    rd = _copy(source, tmp_path, "events")  # an archive keeps the checkpoints only
    why = houses.source_problem(rd, FORK)
    assert why is not None and "no births event chunks" in why and "only checkpoints" in why
    with pytest.raises(houses.HouseError, match="no births event chunks"):
        rebuild(rd.path, FORK, cfg)


def test_a_gap_in_the_chunks_names_the_missing_range(source: Path, tmp_path: Path, cfg: Config) -> None:
    rd = _copy(source, tmp_path)
    next(p for s0, _, p in rd.event_files("births") if s0 == EARLY).unlink()
    why = houses.source_problem(rd, END)
    assert why is not None and "births event chunks" in why and "[100, 600)" in why
    assert houses.source_problem(rd, EARLY) is None  # [0, 100) is still covered
    with pytest.raises(houses.HouseError, match=r"\[100, 600\)"):
        rebuild(rd.path, END, cfg)


def test_missing_death_chunks_make_a_source_unusable_too(source: Path, tmp_path: Path) -> None:
    rd = _copy(source, tmp_path)
    for _, _, p in rd.event_files("deaths"):
        p.unlink()
    why = houses.source_problem(rd, FORK)
    assert why is not None and "deaths event chunks" in why


def test_chunks_that_do_not_start_at_step_0_are_unusable(source: Path, tmp_path: Path) -> None:
    rd = _copy(source, tmp_path)
    for kind in ("births", "deaths"):
        next(p for s0, _, p in rd.event_files(kind) if s0 == 0).unlink()
    why = houses.source_problem(rd, FORK)
    assert why is not None and "births event chunks" in why and "do not start at step 0" in why
    assert "do not start at step 0" in why


def test_a_source_that_is_itself_a_fork_cannot_be_rebuilt(source: Path, tmp_path: Path, cfg: Config) -> None:
    rd = _copy(source, tmp_path)
    rd.update_meta(start_step=EARLY)
    why = houses.source_problem(rd, FORK)
    assert why is not None and "itself a fork" in why and "start_step 100" in why
    with pytest.raises(houses.HouseError, match="itself a fork"):
        rebuild(rd.path, FORK, cfg)


def test_chunks_that_do_not_match_the_checkpoint_are_refused(
    source: Path, tmp_path: Path, cfg: Config
) -> None:
    """A birth log that stops short of the checkpoint's uids (a chunk replaced by an empty one)."""
    rd = _copy(source, tmp_path)
    # drop the last births of [200, 300): the checkpoint then holds uids the log does not know
    path = next(p for s0, _, p in rd.event_files("births") if s0 == 200)
    z = dict(np.load(path))
    keep = np.arange(z["uid"].size) < z["uid"].size // 2
    np.savez_compressed(path, **{k: v[keep] for k, v in z.items()})
    with pytest.raises(houses.HouseError):
        rebuild(rd.path, FORK, cfg)


# --- the house file -------------------------------------------------------------------------------------


def test_a_house_file_round_trips_and_is_not_a_sidecar(source: Path, tmp_path: Path, cfg: Config) -> None:
    rd = _copy(source, tmp_path)
    seed = rebuild(rd.path, FORK, cfg)
    p = housefile.write(rd, FORK, seed, rate=RATE, run_seed=SEED)
    assert p == rd.observers_dir / f"houses_{FORK:010d}.npz" and p.exists()
    assert housefile.files(rd) == [(FORK, p)]
    assert_same(housefile.read(rd, FORK, rate=RATE, run_seed=SEED), seed)
    assert FORK in {s for s, _ in rd.observer_files()}  # the real sidecar of that step, not the house file
    assert all(f.name.startswith("step_") for _, f in rd.observer_files())
    assert not list(rd.observers_dir.glob("*.tmp")) and not list(rd.observers_dir.glob(".*"))


def test_the_run_directory_keeps_the_house_file(source: Path, tmp_path: Path, cfg: Config) -> None:
    """Pruning and truncating cut sidecars, never the house file."""
    rd = _copy(source, tmp_path)
    housefile.write(rd, FORK, rebuild(rd.path, FORK, cfg), rate=RATE, run_seed=SEED)
    rd.truncate_after(EARLY)  # the checkpoints and sidecars after 100 go
    assert [s for s, _ in rd.checkpoints()] == [0, EARLY] and [s for s, _ in housefile.files(rd)] == [FORK]
    rd._prune(keep=1)
    assert [s for s, _ in housefile.files(rd)] == [FORK]


@pytest.mark.parametrize(
    "kwargs, match",
    [({"rate": 0.5}, "house_rate"), ({"run_seed": SEED + 1}, "seed")],
    ids=["other-rate", "other-seed"],
)
def test_a_file_built_with_another_rule_is_refused(
    source: Path, tmp_path: Path, cfg: Config, kwargs: dict[str, float], match: str
) -> None:
    rd = _copy(source, tmp_path)
    housefile.write(rd, FORK, rebuild(rd.path, FORK, cfg), rate=RATE, run_seed=SEED)
    with pytest.raises(HouseFileError, match=match):
        housefile.read(rd, FORK, **{"rate": RATE, "run_seed": SEED, **kwargs})


def test_a_malformed_or_torn_house_file_is_refused(source: Path, tmp_path: Path, cfg: Config) -> None:
    rd = _copy(source, tmp_path)
    seed = rebuild(rd.path, FORK, cfg)
    p = housefile.write(rd, FORK, seed, rate=RATE, run_seed=SEED)
    with pytest.raises(HouseFileError, match="step 100"):
        shutil.copyfile(p, housefile.path(rd, EARLY))
        housefile.read(rd, EARLY, rate=RATE, run_seed=SEED)  # a file of another step
    bad = HouseSeed(seed.uid[::-1].copy(), seed.house, seed.house_from)
    with pytest.raises(HouseFileError, match="increasing"):
        housefile.write(rd, FORK + 1, bad, rate=RATE, run_seed=SEED)
    with pytest.raises(HouseFileError, match="int64"):
        housefile.write(
            rd,
            FORK + 2,
            HouseSeed(seed.uid.astype(np.int32), seed.house, seed.house_from),
            rate=RATE,
            run_seed=SEED,
        )
    p.write_bytes(p.read_bytes()[:40])  # torn
    with pytest.raises(HouseFileError, match="unreadable"):
        housefile.read(rd, FORK, rate=RATE, run_seed=SEED)
    with pytest.raises(HouseFileError, match="unreadable"):
        housefile.read(rd, FORK + 99, rate=RATE, run_seed=SEED)  # absent


# --- the hash and the fold ------------------------------------------------------------------------------


def test_the_numpy_hash_is_the_one_of_the_phylogeny_and_of_the_step() -> None:
    import mlx.core as mx

    rng = np.random.default_rng(3)
    x = np.concatenate([rng.integers(0, 2**40, 200), [0, 1, 2**62, 2**63 - 1]]).astype(np.int64)
    for seed in (0, 7, 2**40 + 5, -3):
        assert np.array_equal(housefile.splitmix64(x, seed), phylo.splitmix64(x, seed))
        assert np.array_equal(
            housefile.splitmix64(x, seed), np.array(groups.splitmix64(mx.array(x), seed), copy=False)
        )
    idx = np.arange(500, dtype=np.int64)  # marker_roots hashes the tree index, which is the uid
    tree = phylo.Tree(
        np.zeros(idx.size, np.int32), np.ones(idx.size, np.int32), np.full(idx.size, 2, np.int32), 3
    )
    for rate in (0.0, 1 / 2048, 0.125, 0.5, 0.75, 0.999, 1.0):
        mine = housefile.hash_below(x, 7, rate)
        assert np.array_equal(mine, np.array(groups.hash_below(mx.array(x), 7, rate), copy=False)), rate
        assert np.array_equal(housefile.hash_below(idx, 7, rate), phylo.marker_roots(tree, rate, 7)), rate
    with pytest.raises(ValueError):
        housefile.hash_below(x, 7, -0.1)


def _naive_fold(
    seed: HouseSeed, uid: np.ndarray, parent: np.ndarray, run_seed: int, rate: float
) -> dict[int, tuple[int, int]]:
    """The rule of ``social.update`` birth by birth, in a dictionary."""
    table = {int(u): (int(h), int(f)) for u, h, f in zip(seed.uid, seed.house, seed.house_from, strict=True)}
    founds = housefile.hash_below(uid, run_seed, rate)
    for u, p, fnd in zip(uid.tolist(), parent.tolist(), founds.tolist(), strict=True):
        if p < 0:
            table[u] = (u, -1)
        elif fnd:
            table[u] = (u, table[p][0])
        else:
            table[u] = table[p]
    return table


@pytest.mark.parametrize("rate", [0.0, 0.1, 0.6, 1.0])
def test_the_fold_equals_the_rule_birth_by_birth(rate: float) -> None:
    rng = np.random.default_rng(11)
    n0, nb = 40, 3000
    seed = HouseSeed(
        np.arange(100, 100 + n0, dtype=np.int64),
        rng.integers(0, 6, n0).astype(np.int64) + 90,
        rng.integers(-1, 6, n0).astype(np.int64) + 90,
    )
    uid = np.arange(1000, 1000 + nb, dtype=np.int64)
    # a parent is a seed agent or an earlier birth, an immigrant in one of 15; deep chains are common
    pool = np.concatenate([seed.uid, uid])
    parent = np.array(
        [-1 if rng.random() < 1 / 15 else pool[rng.integers(max(0, n0 + i - 30), n0 + i)] for i in range(nb)]
    )
    got = housefile.fold(seed, uid, parent, run_seed=5, rate=rate)
    want = _naive_fold(seed, uid, parent, 5, rate)
    assert got.uid.tolist() == sorted(want)
    assert [(int(h), int(f)) for h, f in zip(got.house, got.house_from, strict=True)] == [
        want[int(u)] for u in got.uid
    ]
    half = nb // 2  # folding in two stages is folding at once
    twice = housefile.fold(
        housefile.fold(seed, uid[:half], parent[:half], run_seed=5, rate=rate),
        uid[half:],
        parent[half:],
        run_seed=5,
        rate=rate,
    )
    assert_same(twice, got)
    assert housefile.fold(seed, uid[:0], parent[:0], run_seed=5, rate=rate) is seed


def test_the_fold_refuses_births_that_do_not_fit() -> None:
    seed = HouseSeed(np.array([5, 6], np.int64), np.array([5, 5], np.int64), np.array([-1, -1], np.int64))
    ok = np.array([10, 11], np.int64)
    kw = {"run_seed": 0, "rate": 0.5}
    with pytest.raises(HouseFileError, match="ascending"):
        housefile.fold(seed, np.array([11, 10]), np.array([5, 5]), **kw)
    with pytest.raises(HouseFileError, match="ascending"):
        housefile.fold(seed, np.array([6]), np.array([5]), **kw)  # not above the seed's uids
    with pytest.raises(HouseFileError, match="parents"):
        housefile.fold(seed, ok, np.array([5, 99]), **kw)  # a parent in no table
    with pytest.raises(HouseFileError, match="older"):
        housefile.fold(seed, ok, np.array([11, 5]), **kw)  # a parent born after its child
    with pytest.raises(HouseFileError, match="length"):
        housefile.fold(seed, ok, np.array([5]), **kw)


def test_the_fold_of_the_births_of_the_log_equals_the_live_houses(source: Path, cfg: Config) -> None:
    """The file of step 100 folded over the births of [100, 600) is the live observer's houses at step 600."""
    rd = RunDir.open(source)
    seed = rebuild(source, EARLY, cfg)
    folded = housefile.fold_chunks(rd, seed, EARLY, END, run_seed=SEED, rate=RATE)
    want = live(rd, END)
    assert_same(alive_part(folded, want.uid), want)
    assert folded.uid.size > want.uid.size  # the births that died since stay in the table
    mid = housefile.fold_chunks(rd, seed, EARLY, FORK, run_seed=SEED, rate=RATE)
    assert_same(alive_part(mid, live(rd, FORK).uid), live(rd, FORK))
    assert housefile.fold_chunks(rd, seed, EARLY, EARLY, run_seed=SEED, rate=RATE) is seed


def test_births_between_needs_chunks_that_tile_the_window(source: Path, tmp_path: Path) -> None:
    rd = RunDir.open(source)
    uid, parent = housefile.births_between(rd, EARLY, 350)  # a chunk straddling 350 is trimmed
    full_uid, _ = housefile.births_between(rd, EARLY, END)
    steps = np.concatenate([np.load(p)["step"] for s0, _, p in rd.event_files("births") if s0 >= EARLY])
    assert uid.size == int((steps < 350).sum()) and full_uid[: uid.size].tolist() == uid.tolist()
    assert uid.size == parent.size and housefile.births_between(rd, END, END)[0].size == 0
    with pytest.raises(HouseFileError, match="tile"):
        housefile.births_between(rd, EARLY, END + 100)  # nothing covers [600, 700)
    cp = _copy(source, tmp_path)
    next(p for s0, _, p in cp.event_files("births") if s0 == FORK).unlink()
    with pytest.raises(HouseFileError, match="tile"):
        housefile.births_between(cp, EARLY, END)
    with pytest.raises(HouseFileError, match="tile"):
        housefile.births_between(
            cp, EARLY + 50, END
        )  # a window that starts inside a chunk is not tiled from its start


# --- the observer ---------------------------------------------------------------------------------------


def _fork(source: Path, tmp_path: Path, cfg: Config, *, file: bool = True) -> tuple[RunDir, Config]:
    """A fork directory of the source at ``FORK`` with its house file, as ``alife play --init-from`` makes it
    (without play.json)."""
    rd = RunDir.create(tmp_path / "fork", cfg, parent=f"{source}:{FORK}")
    rd.update_meta(start_step=FORK)
    if file:
        housefile.write(rd, FORK, rebuild(source, FORK, cfg), rate=RATE, run_seed=SEED)
    return rd, cfg


def test_fresh_observers_with_the_file_have_the_exact_houses(
    source: Path, tmp_path: Path, cfg: Config, snapshots: None
) -> None:
    rd, _ = _fork(source, tmp_path, cfg)
    state = load_checkpoint(RunDir.open(source).checkpoint_at(FORK), cfg)
    obs, note = Observer.open(rd, state, cfg)
    assert note is None  # (a start, not a resume; the file is as it should be)
    assert obs.society.house_origin == 0 and obs.row_block()["social"]["houses"]["exact"] is True
    assert obs.row_block()["social"]["origin"] == 0 and obs.society.snapshot_step == FORK
    sl = obs.social_labels()
    house = to_numpy(sl.house)
    uid = to_numpy(obs.so.uid)
    order = np.argsort(uid[uid >= 0])
    want = live(RunDir.open(source), FORK)
    assert np.array_equal(house[uid >= 0][order], want.house)
    assert np.array_equal(to_numpy(obs.so.house_from)[uid >= 0][order], want.house_from)
    ident = obs.identity_rows(state)
    assert ident.tables["exact"] is True and ident.tables["origin"] == 0
    assert ident.tables["unknown_immigrant"] == 0
    assert 8 <= len(ident.tables["houses"]) <= 16  # the real factions, not the 4 clades of the fallback


def test_the_clade_houses_are_the_fallback_and_say_so(
    source: Path, tmp_path: Path, cfg: Config, snapshots: None
) -> None:
    rd, _ = _fork(source, tmp_path, cfg, file=False)
    state = load_checkpoint(RunDir.open(source).checkpoint_at(FORK), cfg)
    obs, note = Observer.open(rd, state, cfg)
    assert note is None and obs.society.house_origin == FORK
    assert obs.row_block()["social"]["houses"]["exact"] is False
    assert obs.identity_rows(state).tables["unknown_immigrant"] > 0  # immigrant lineages: no house at all


def test_fresh_with_a_partial_seed_is_not_exact(source: Path, cfg: Config, snapshots: None) -> None:
    state = load_checkpoint(RunDir.open(source).checkpoint_at(FORK), cfg)
    full = rebuild(source, FORK, cfg)
    half = HouseSeed(full.uid[::2].copy(), full.house[::2].copy(), full.house_from[::2].copy())
    obs = Observer.fresh(state, cfg, houses=half)
    assert obs.society.house_origin == FORK and obs.row_block()["social"]["houses"]["exact"] is False
    uid, house = to_numpy(obs.so.uid), to_numpy(obs.so.house)
    listed = np.isin(uid, half.uid)
    ref = dict(zip(full.uid.tolist(), full.house.tolist(), strict=True))
    assert all(house[i] == ref[int(uid[i])] for i in np.flatnonzero(listed))  # the listed rows are seeded
    empty = HouseSeed(*(np.zeros(0, np.int64),) * 3)
    assert Observer.fresh(state, cfg, houses=empty).society.house_origin == FORK
    unsorted = HouseSeed(full.uid[::-1].copy(), full.house, full.house_from)
    with pytest.raises(ValueError, match="ascending"):
        Observer.fresh(state, cfg, houses=unsorted)


def test_a_fork_keeps_the_exact_houses_of_the_source_to_the_end(
    source: Path, tmp_path: Path, cfg: Config, snapshots: None
) -> None:
    """The fork continues the source's trajectory from step 300: its sidecar at 600 holds the houses the
    source's live observer held, house of origin included, and every record says exact."""
    rd, _ = _fork(source, tmp_path, cfg)
    state = load_checkpoint(RunDir.open(source).checkpoint_at(FORK), cfg)
    res = runner.run(rd, cfg, state, END - FORK, quiet=True)
    assert res.status == "done"
    assert_same(live(rd, END), live(RunDir.open(source), END))
    records = rd.social_records()
    assert [r["step"] for r in records] == list(range(FORK + SNAP, END + 1, SNAP))
    assert all(r["origin"] == 0 and r["summary"]["houses"]["exact"] for r in records)
    assert (
        records[0]["window_start"] == FORK
    )  # the window starts at the fork, the houses are exact all the same
    assert all(
        r["observers"]["social"]["houses"]["exact"] is True for r in rd.metrics_rows() if "observers" in r
    )
    assert [s for s, _ in housefile.files(rd)] == [FORK]


def test_a_sidecar_lost_at_a_later_checkpoint_gives_the_same_houses(
    source: Path, tmp_path: Path, cfg: Config, snapshots: None
) -> None:
    rd, _ = _fork(source, tmp_path, cfg)
    runner.run(rd, cfg, load_checkpoint(RunDir.open(source).checkpoint_at(FORK), cfg), END - FORK, quiet=True)
    want = live(rd, END)
    rd.observer_path(END).unlink()
    state = load_checkpoint(rd.checkpoint_at(END), cfg)
    obs, note = Observer.open(rd, state, cfg)
    assert note is not None and "no sidecar at step 600" in note and "house file" in note
    assert obs.society.house_origin == 0 and obs.row_block()["social"]["houses"]["exact"] is True
    uid = to_numpy(obs.so.uid)
    order = np.argsort(uid[uid >= 0])
    assert np.array_equal(to_numpy(obs.so.house)[uid >= 0][order], want.house)
    assert np.array_equal(to_numpy(obs.so.house_from)[uid >= 0][order], want.house_from)
    assert np.array_equal(uid[uid >= 0][order], want.uid)


def test_a_valid_sidecar_wins_over_the_file(
    source: Path, tmp_path: Path, cfg: Config, snapshots: None
) -> None:
    rd, _ = _fork(source, tmp_path, cfg)
    runner.run(rd, cfg, load_checkpoint(RunDir.open(source).checkpoint_at(FORK), cfg), END - FORK, quiet=True)
    obs, note = Observer.open(rd, load_checkpoint(rd.checkpoint_at(END), cfg), cfg)
    assert note is None and obs.society.house_origin == 0  # (loaded: origin 0 is in the sidecar)


def test_a_lost_sidecar_without_the_birth_chunks_falls_back_with_a_note(
    source: Path, tmp_path: Path, cfg: Config, snapshots: None
) -> None:
    rd, _ = _fork(source, tmp_path, cfg)
    runner.run(rd, cfg, load_checkpoint(RunDir.open(source).checkpoint_at(FORK), cfg), END - FORK, quiet=True)
    rd.observer_path(END).unlink()
    for _, _, p in rd.event_files("births"):
        p.unlink()
    obs, note = Observer.open(rd, load_checkpoint(rd.checkpoint_at(END), cfg), cfg)
    assert note is not None and "house file of step 300 cannot be used at step 600" in note and "tile" in note
    assert obs.society.house_origin == END and obs.row_block()["social"]["houses"]["exact"] is False


def test_a_file_of_another_rule_is_not_used_by_the_observer(
    source: Path, tmp_path: Path, cfg: Config, snapshots: None
) -> None:
    rd, _ = _fork(source, tmp_path, cfg, file=False)
    housefile.write(rd, FORK, rebuild(source, FORK, cfg), rate=0.5, run_seed=SEED)
    state = load_checkpoint(RunDir.open(source).checkpoint_at(FORK), cfg)
    obs, note = Observer.open(rd, state, cfg)
    assert note is not None and "house_rate" in note  # even on a start that is not a resume
    assert obs.society.house_origin == FORK


def test_a_file_of_a_later_step_is_ignored(
    source: Path, tmp_path: Path, cfg: Config, snapshots: None
) -> None:
    """The newest file at or before the observer's step is the one used: a later one is not even read."""
    rd, _ = _fork(source, tmp_path, cfg, file=False)
    rd.observers_dir.mkdir(exist_ok=True)
    housefile.path(rd, END).write_bytes(b"not a house file")
    state = load_checkpoint(RunDir.open(source).checkpoint_at(FORK), cfg)
    obs, note = Observer.open(rd, state, cfg)
    assert note is None and obs.society.house_origin == FORK
    housefile.write(rd, FORK, rebuild(source, FORK, cfg), rate=RATE, run_seed=SEED)
    obs, note = Observer.open(rd, state, cfg)  # and with the file of the step, the later one is still ignored
    assert note is None and obs.society.house_origin == 0
    with pytest.raises(HouseFileError, match="back to step"):
        housefile.fold_chunks(rd, rebuild(source, END, cfg), END, FORK, run_seed=SEED, rate=RATE)


# --- alife play --houses --------------------------------------------------------------------------------


@pytest.fixture
def short_cadence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(play, "PLAY_CADENCE", CADENCE)
    social_every(monkeypatch, 10)


def play_args(src: Path | str, out: Path, *extra: str, step: int = FORK) -> list[str]:
    return [
        "play", "--init-from", f"{src}:{step}", "--out", str(out), "--no-control", "--rate", "max",
        "--until", str(step + 10), *extra,
    ]  # fmt: skip


def first_idnt(data: bytes) -> dict[str, object]:
    return next(r for r in decode(data).records if r.tag == "IDNT").f["json"]


def test_play_makes_a_planet_with_the_exact_houses(
    source: Path, tmp_path: Path, cfg: Config, short_cadence: None, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    out = tmp_path / "planet"
    assert cli.main(play_args(source, out)) == 0
    cap = capfdbinary.readouterr()
    err = cap.err.decode(errors="replace")
    d = decode(cap.out)
    assert d.header["play"]["planet"] == {"source": f"{source}:{FORK}", "houses": "exact"}
    assert f"houses: rebuilding the exact houses of {source} at step {FORK}" in err
    ident = first_idnt(cap.out)
    assert ident["exact"] is True and ident["origin"] == 0 and ident["unknown_immigrant"] == 0
    rd = RunDir.open(out)
    pj = rd.read_play_json()
    assert pj is not None and pj["planet"]["houses"] == "exact" and rd.meta()["start_step"] == FORK
    assert_same(housefile.read(rd, FORK, rate=RATE, run_seed=SEED), rebuild(source, FORK, cfg))
    assert [s for s, _ in housefile.files(rd)] == [FORK]


def test_houses_clade_skips_the_rebuild(
    source: Path, tmp_path: Path, short_cadence: None, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    out = tmp_path / "planet"
    assert cli.main(play_args(source, out, "--houses", "clade")) == 0
    cap = capfdbinary.readouterr()
    assert decode(cap.out).header["play"]["planet"]["houses"] == "clade"
    assert b"rebuilding" not in cap.err
    ident = first_idnt(cap.out)
    assert ident["exact"] is False and ident["origin"] == FORK and ident["unknown_immigrant"] > 0
    rd = RunDir.open(out)
    assert housefile.files(rd) == [] and rd.read_play_json()["planet"]["houses"] == "clade"  # type: ignore[index]


def test_an_archive_only_source_falls_back_under_auto_and_fails_under_exact(
    source: Path, tmp_path: Path, short_cadence: None, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    archive = tmp_path / "archive"
    shutil.copytree(source, archive, ignore=shutil.ignore_patterns("events"))
    # exact: exit 4 before anything is loaded or written, the missing chunks named
    out = tmp_path / "x"
    assert cli.main(play_args(archive, out, "--houses", "exact")) == 4
    cap = capfdbinary.readouterr()
    assert b"no births event chunks" in cap.err and cap.out == b"" and not out.exists()
    # auto: one note, clade houses, the planet is made
    out = tmp_path / "a"
    assert cli.main(play_args(archive, out)) == 0
    cap = capfdbinary.readouterr()
    err = cap.err.decode(errors="replace")
    assert err.count("houses: ") == 1 and "no births event chunks" in err and "clades (approximate)" in err
    assert decode(cap.out).header["play"]["planet"]["houses"] == "clade"
    assert first_idnt(cap.out)["exact"] is False
    assert housefile.files(RunDir.open(out)) == []
    assert RunDir.open(out).read_play_json()["planet"]["houses"] == "clade"  # type: ignore[index]


def test_a_gap_in_the_chunks_is_an_error_under_exact_and_a_note_under_auto(
    source: Path, tmp_path: Path, short_cadence: None, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    rd = _copy(source, tmp_path)
    next(p for s0, _, p in rd.event_files("births") if s0 == EARLY).unlink()
    assert cli.main(play_args(rd.path, tmp_path / "x", "--houses", "exact")) == 4
    assert b"[100, 300)" in capfdbinary.readouterr().err and not (tmp_path / "x").exists()
    assert cli.main(play_args(rd.path, tmp_path / "a", "--houses", "auto")) == 0
    assert b"[100, 300)" in capfdbinary.readouterr().err
    assert RunDir.open(tmp_path / "a").read_play_json()["planet"]["houses"] == "clade"  # type: ignore[index]


def test_a_fork_of_a_planet_is_not_rebuilt(
    source: Path, tmp_path: Path, short_cadence: None, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    """A planet has start_step > 0: its logs do not hold the births before it (and its own house file is not
    a source of houses for a new planet)."""
    first = tmp_path / "planet"
    assert cli.main(play_args(source, first)) == 0
    capfdbinary.readouterr()
    assert cli.main(play_args(first, tmp_path / "x", "--houses", "exact", step=FORK + 10)) == 4
    assert b"itself a fork" in capfdbinary.readouterr().err and not (tmp_path / "x").exists()
    out = tmp_path / "second"
    assert cli.main(play_args(first, out, step=FORK + 10)) == 0
    cap = capfdbinary.readouterr()
    assert b"itself a fork" in cap.err
    assert decode(cap.out).header["play"]["planet"]["houses"] == "clade"
    assert first_idnt(cap.out)["exact"] is False


def test_a_bare_state_file_has_no_events(
    source: Path, tmp_path: Path, short_cadence: None, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    ck = RunDir.open(source).checkpoint_at(FORK)
    args = ["play", "--init-from", str(ck), "--out", str(tmp_path / "x"), "--no-control", "--rate", "max"]
    assert cli.main([*args, "--until", str(FORK + 10), "--houses", "exact"]) == 4
    assert b"state file" in capfdbinary.readouterr().err and not (tmp_path / "x").exists()
    assert cli.main([*args, "--until", str(FORK + 10)]) == 0
    assert decode(capfdbinary.readouterr().out).header["play"]["planet"]["houses"] == "clade"


def test_the_rebuild_uses_the_planets_seed(
    source: Path, tmp_path: Path, short_cadence: None, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    """``--set run.seed=N`` re-partitions the past and the future with one rule: the file holds the houses
    of the new seed, and the observer accepts it."""
    out = tmp_path / "planet"
    assert cli.main(play_args(source, out, "--set", f"run.seed={SEED + 1}")) == 0
    cap = capfdbinary.readouterr()
    rd = RunDir.open(out)
    seed = housefile.read(rd, FORK, rate=RATE, run_seed=SEED + 1)
    assert not np.array_equal(seed.house, houses.rebuild(source, FORK, rate=RATE, seed=SEED).house)
    assert first_idnt(cap.out)["exact"] is True


def test_houses_applies_to_a_planet_only(
    source: Path, tmp_path: Path, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    out = tmp_path / "planet"
    assert cli.main(play_args(source, out, "--houses", "clade")) == 0
    capfdbinary.readouterr()
    assert cli.main(["play", str(out), "--houses", "exact", "--no-control"]) == 2
    assert b"--houses" in capfdbinary.readouterr().err
    assert (
        cli.main(
            ["play", "--preset", "tiny", "--houses", "auto", "--no-control", "--out", str(tmp_path / "p")]
        )
        == 2
    )
    assert b"--houses applies to a planet" in capfdbinary.readouterr().err and not (tmp_path / "p").exists()
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["play", "--init-from", "x", "--houses", "some"])


def test_a_played_planet_resumes_with_its_exact_houses(
    source: Path, tmp_path: Path, short_cadence: None, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    out = tmp_path / "planet"
    assert cli.main(play_args(source, out)) == 0
    capfdbinary.readouterr()
    assert cli.main(["play", str(out), "--no-control", "--rate", "max", "--until", str(FORK + 20)]) == 0
    cap = capfdbinary.readouterr()
    assert decode(cap.out).header["play"]["planet"] == {"source": f"{source}:{FORK}", "houses": "exact"}
    assert first_idnt(cap.out)["exact"] is True  # the sidecar of the checkpoint carries origin 0


# --- the command line of the rebuild --------------------------------------------------------------------


def test_the_tool_prints_a_summary_and_writes_the_file(
    source: Path, tmp_path: Path, cfg: Config, capsys: pytest.CaptureFixture[str]
) -> None:
    target = RunDir.create(tmp_path / "target", cfg)
    assert houses.main([str(source), "--step", str(FORK), "--out", str(target.path)]) == 0
    out = capsys.readouterr()
    assert f"step {FORK}, house_rate {RATE}, seed {SEED}" in out.out and "houses; largest:" in out.out
    assert f"houses_{FORK:010d}.npz" in out.err
    assert_same(housefile.read(target, FORK, rate=RATE, run_seed=SEED), rebuild(source, FORK, cfg))
    assert houses.main([str(source)]) == 0  # the latest checkpoint by default
    assert "step 600" in capsys.readouterr().out
    assert houses.main([str(source), "--step", str(FORK), "--rate", "0", "--seed", "1"]) == 0
    assert "house_rate 0.0, seed 1" in capsys.readouterr().out


def test_the_tool_exits_4_for_a_source_without_events_and_never_writes_into_the_source(
    source: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rd = _copy(source, tmp_path, "events")
    assert houses.main([str(rd.path), "--step", str(FORK)]) == 4
    assert "no births event chunks" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        houses.main([str(source), "--step", str(FORK), "--out", str(source)])
    assert housefile.files(RunDir.open(source)) == [] and capsys.readouterr().out == ""
