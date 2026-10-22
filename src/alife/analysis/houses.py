"""The exact houses of a planet: rebuilt from a source run's birth chain (DESIGN §33.3.5, GAMES §6.2).

A house is a pure function of the birth chain: a birth founds one iff ``splitmix64(uid, seed) < rate * 2^64``,
an immigrant (a birth without a parent) and a founder of the run always do, and every other birth inherits its
parent's (``social.update``; offline: ``phylo.marker_roots``). An observer attached to a world already running
cannot see that chain, so a fork of an evolved checkpoint (a planet) would start with the clades as houses:
4 factions where the rule gives 79 (``64k3d_s0`` at step 500,000, ``docs/GAMES.md`` §6.2). This tool rebuilds
the real ones from the source's event chunks:

1. ``phylo.load_tree(SRC, end=S)``: the phylogeny of the steps before the fork step ``S``. Tree indices equal
   uids for a source that starts at step 0 (``load_tree`` raises unless the birth uids are contiguous from
   ``n_init``), so ``marker_roots`` applies as it is. A source that is itself a fork cannot be rebuilt.
2. For every agent alive in the checkpoint at ``S``: ``house`` = the uid of its nearest root and
   ``house_from`` = the house of that root's parent (``-1`` for a root without parent).

``seed`` and ``rate`` are the *planet's* ``run.seed`` and ``game.house_rate``: one rule applies to the past
and to the future. The result is a ``game.housefile.HouseSeed``; ``python -m alife.analysis.houses`` prints
it as a summary and, with ``--out RUN``, writes ``RUN/observers/houses_<S>.npz`` (3.7 s and 2.6 GB for the
22.5 M agents of ``64k3d_s0`` at 500,000 steps, CPU).

A source whose event chunks are missing or do not cover ``[0, S)`` contiguously (most of the archive on the
external drive keeps checkpoints only) is *unusable*: ``source_problem`` says why from a directory listing,
before any load.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from alife.analysis import phylo
from alife.game.housefile import HouseSeed
from alife.runio import CheckpointError, RunDir

__all__ = ["HouseError", "houses_of", "main", "rebuild", "source_problem", "summary"]


class HouseError(CheckpointError):
    """The exact houses of a checkpoint cannot be rebuilt from its run (exit code 4 of the CLI)."""


def _first_gap(spans: list[tuple[int, int]], step: int) -> int | None:
    """The first step of ``[0, step)`` that no chunk covers, or None; chunks must start where the last ended
    (a chunk that reaches past ``step`` is fine)."""
    want = 0
    for s0, s1 in sorted(spans):
        if want >= step:
            break
        if s0 != want:
            return want
        want = s1
    return want if want < step else None


def source_problem(rd: RunDir, step: int) -> str | None:
    """Why the houses of the checkpoint at ``step`` of ``rd`` cannot be rebuilt, or None if they can. From a
    listing of the run directory only (no chunk is read)."""
    origin = int(rd.meta().get("start_step") or 0)
    if origin != 0:
        return (
            f"{rd.path} is itself a fork (start_step {origin}): the births before step {origin} are not in "
            "its logs"
        )
    for kind in ("births", "deaths"):
        spans = [(s0, s1) for s0, s1, _ in rd.event_files(kind)]
        if not spans:
            return f"{rd.path} has no {kind} event chunks (events/ is absent): only checkpoints were kept"
        gap = _first_gap(spans, step)
        if gap is not None:
            return (
                f"the {kind} event chunks of {rd.path} do not cover steps [{gap}, {step}) "
                f"(they tile [0, {gap}))"
                if gap
                else f"the {kind} event chunks of {rd.path} do not start at step 0"
            )
    return None


def rebuild(src: Path | str, step: int, *, rate: float, seed: int) -> HouseSeed:
    """The houses of the agents alive in ``src``'s checkpoint at ``step``, under ``house_rate`` ``rate`` and
    ``run.seed`` ``seed``. ``HouseError`` if the events are missing, do not cover ``[0, step)`` or are
    inconsistent with the checkpoint."""
    rd = RunDir.open(Path(src))
    why = source_problem(rd, step)
    if why is not None:
        raise HouseError(why)
    try:
        ck = rd.checkpoint_at(step)
        tree, report = phylo.load_tree(rd.path, end=step, check_checkpoint=False)
        alive = phylo._alive_uids(ck)
    except (ValueError, OSError, KeyError, EOFError, zipfile.BadZipFile) as e:
        raise HouseError(f"{rd.path}: the event logs cannot be read as a birth chain ({e})") from e
    if not (report["chunks_tile"] and report["end_covered"]):
        raise HouseError(f"{rd.path}: the event chunks do not tile [0, {step}) ({report})")
    if alive.size and (alive[0] < 0 or alive[-1] >= tree.n):
        raise HouseError(
            f"{rd.path}: the checkpoint at {step} holds uids the birth log does not know "
            f"(largest {int(alive[-1])}, the log ends at {tree.n - 1})"
        )
    return houses_of(tree, alive, rate=rate, seed=seed)


def houses_of(tree: phylo.Tree, alive: np.ndarray, *, rate: float, seed: int) -> HouseSeed:
    """The houses of the agents ``alive`` (ascending uids, which are tree indices) in a phylogeny whose
    indices are uids: ``house`` is the uid of the nearest root of ``phylo.marker_roots(tree, rate, seed)``
    and ``house_from`` the house of that root's parent (``-1`` for a root without one)."""
    roots = phylo.marker_roots(tree, rate, seed)
    near = phylo.nearest_root(tree.parent, roots).astype(np.int64)
    house = near[alive]
    par = tree.parent[house].astype(np.int64)  # the founder's parent, -1 for a root without one
    house_from = np.where(par >= 0, near[np.maximum(par, 0)], -1)
    return HouseSeed(alive.astype(np.int64), house, house_from)


def summary(h: HouseSeed, *, top: int = 5) -> dict[str, Any]:
    """Counts of a seed: agents, houses and the shares of the largest."""
    ids, counts = np.unique(h.house, return_counts=True)
    order = np.lexsort((ids, -counts))
    return {
        "alive": int(h.uid.size),
        "houses": int(ids.size),
        "top": [
            {"id": int(ids[i]), "size": int(counts[i]), "share": round(float(counts[i] / h.uid.size), 4)}
            for i in order[:top]
        ],
        "founded_here": int(np.count_nonzero(h.house_from >= 0)),
    }


# --- command line -------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m alife.analysis.houses", description=__doc__.split("\n")[0])
    ap.add_argument("run", type=Path, help="the source run directory (read only)")
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: the latest)")
    ap.add_argument("--rate", type=float, default=None, help="game.house_rate (default: the run's)")
    ap.add_argument("--seed", type=int, default=None, help="run.seed (default: the run's)")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="RUN",
        help="write RUN/observers/houses_<step>.npz (the planet's directory; never the source's)",
    )
    a = ap.parse_args(argv)
    try:
        rd = RunDir.open(a.run)
        cfg = rd.config()
        step = a.step if a.step is not None else max(st for st, _ in rd.checkpoints())
        rate = cfg.game.house_rate if a.rate is None else a.rate
        seed = cfg.run.seed if a.seed is None else a.seed
        if a.out is not None and a.out.resolve() == rd.path.resolve():
            ap.error("--out must be a different directory than the source")
        h = rebuild(rd.path, step, rate=rate, seed=seed)
        if a.out is not None:
            from alife.game import housefile

            p = housefile.write(RunDir.open(a.out), step, h, rate=rate, run_seed=seed)
            print(f"wrote {p}", file=sys.stderr)
    except (CheckpointError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 4 if isinstance(e, CheckpointError) else 2
    print(f"step {step}, house_rate {rate}, seed {seed}")
    s = summary(h)
    print(f"{s['alive']} alive in {s['houses']} houses; largest:")
    for t in s["top"]:
        print(f"  {t['id']}: {t['size']} ({100 * t['share']:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
