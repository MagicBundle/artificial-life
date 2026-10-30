"""The three information tests on one run's tables (numpy only; DESIGN §39.14).

* **code**: ``I(S; C_k | sample step)`` for each sender context ``C_k`` and the call type ``S``: do calls say
  anything about the caller's situation? A second family conditions on the lineage as well (``code_lineage``),
  so the part of the code that is a lineage's private habit shows as the gap.
* **reception**: ``I(H; A_r | own context)`` for the call type heard ``H`` and each response ``A_r``: do
  hearers
  act differently on different calls, given their own situation and the time block?
* **transfer** (the language score of the spec): ``I(C_src,k ; A_r | C_i,k, crowd, distance bin, block)``:
  does
  the situation of the loudest audible caller reach the hearer's behaviour beyond what the hearer's own
  situation says? Its null shuffles the source's context among hearers in the same stratum. Part of it is
  carried by the shared environment (neighbours are in similar places), which the shuffle null cannot remove;
  the ``deaf`` and ``decoy`` arms keep that part and cut the calls, so the **arm contrast** of ``meaning`` is
  what isolates the calls.

``Prepared`` codes the tables once; ``run_families`` is the pooled test (max-T family p-values) and
``pair_blocks`` the per-block excess of one named pair.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

from alife.analysis.voice import measures as ms
from alife.analysis.voice import spec, tables
from alife.analysis.voice.tables import Edges, Tables

FAMILIES = ("code", "code_lineage", "reception", "transfer")


class Prepared(NamedTuple):
    """The coded tables of one run."""

    width: int
    edges: Edges
    sym_thr: float  # intensity split of the call type (senders)
    heard_thr: float  # the same for the superposed heard vector
    n_sym: int
    s_sym: np.ndarray  # (ns,) call type of each sender row
    s_ctx: dict[str, np.ndarray]
    s_step: np.ndarray
    s_lineage: np.ndarray
    s_block: np.ndarray
    h_heard: np.ndarray  # (nh,) call type heard
    h_ctx: dict[str, np.ndarray]
    h_resp: dict[str, np.ndarray]
    h_src_ctx: dict[str, np.ndarray]
    h_src_sym: np.ndarray
    h_has_src: np.ndarray  # bool
    h_dist_bin: np.ndarray
    h_block: np.ndarray
    h_step: np.ndarray
    h_lineage: np.ndarray
    h_src_lineage: np.ndarray
    dist_edges: np.ndarray
    n_blocks: int


def prepare(
    tab: Tables,
    window: tuple[int, int],
    n_blocks: int = spec.N_BLOCKS,
    *,
    edges: Edges | None = None,
    sym_thr: float | None = None,
    heard_thr: float | None = None,
    dist_edges: np.ndarray | None = None,
) -> Prepared:
    """Code ``tab``. ``edges`` / ``sym_thr`` / ``heard_thr`` / ``dist_edges`` fixed from another run (the arm
    contrast codes every arm of a seed with the ``none`` arm's cut points) default to this run's own."""
    s, h = tab.senders, tab.hearers
    w = tab.width
    edges = tables.fit_edges(s, h) if edges is None else edges
    sym_thr = ms.symbol_threshold(s["voice"]) if sym_thr is None else sym_thr
    heard_thr = ms.symbol_threshold(h["heard"]) if heard_thr is None else heard_thr
    if dist_edges is None:
        dist_edges = ms.quantile_edges(h["src_dist"][h["has_src"] > 0], spec.DIST_BINS)
    start, end = window
    return Prepared(
        width=w,
        edges=edges,
        sym_thr=sym_thr,
        heard_thr=heard_thr,
        n_sym=ms.alphabet(w),
        s_sym=ms.symbols(s["voice"], sym_thr),
        s_ctx=tables.code_context(s, edges),
        s_step=s["step"],
        s_lineage=s["lineage"],
        s_block=tables.block_of(s["step"], start, end - start, n_blocks),
        h_heard=np.where(h["audible"] > 0, ms.symbols(h["heard"], heard_thr), 0).astype(np.int8),
        h_ctx=tables.code_context(h, edges),
        h_resp=tables.code_response(h, edges),
        h_src_ctx=tables.code_context(h, edges, prefix="src_c_"),
        h_src_sym=ms.symbols(h["src_voice"], sym_thr),
        h_has_src=h["has_src"] > 0,
        h_dist_bin=ms.code_by_edges(np.nan_to_num(h["src_dist"], nan=0.0), dist_edges),
        h_block=tables.block_of(h["step"], start, end - start, n_blocks),
        h_step=h["step"],
        h_lineage=h["lineage"],
        h_src_lineage=h["src_lineage"],
        dist_edges=np.asarray(dist_edges),
        n_blocks=n_blocks,
    )


def _lineage_bucket(lineage: np.ndarray, top: int) -> np.ndarray:
    """The ``top`` most frequent lineages keep their id, the rest share one stratum."""
    u, c = np.unique(lineage, return_counts=True)
    keep = u[np.argsort(-c, kind="stable")[:top]]
    return np.where(np.isin(lineage, keep), lineage, -99)


def usable(ctx: dict[str, np.ndarray]) -> list[str]:
    return [k for k, v in ctx.items() if tables.n_levels(v) > 1]


def build_family(p: Prepared, family: str, mask: np.ndarray | None = None) -> list[ms.Pair]:
    """The pairs of one family (see the module docstring) on the usable rows. ``mask`` (bool over the family's
    own table, senders for the code families and hearers otherwise) keeps a subset of the rows."""
    if family in ("code", "code_lineage"):
        sel = np.ones(len(p.s_sym), bool) if mask is None else mask
        strata = ms.compact(p.s_step[sel])[0]
        if family == "code_lineage":
            strata = ms.combine(strata, ms.compact(_lineage_bucket(p.s_lineage[sel], spec.LINEAGE_STRATA))[0])
        sym = p.s_sym[sel]
        ctx = {k: v[sel] for k, v in p.s_ctx.items()}
        return [ms.Pair(f"S~{k}", sym, p.n_sym, ctx[k], tables.n_levels(ctx[k]), strata) for k in usable(ctx)]
    sel = np.ones(len(p.h_heard), bool) if mask is None else mask
    if family == "reception":
        own = ms.combine(
            p.h_ctx["crowd"][sel],
            p.h_ctx["hurt"][sel],
            p.h_ctx["food"][sel],
            p.h_ctx["bonded"][sel],
            np.maximum(p.h_block[sel], 0),
        )
        heard = p.h_heard[sel]
        return [
            ms.Pair(f"H~{r}", heard, p.n_sym, v[sel], tables.n_levels(v[sel]), own)
            for r, v in p.h_resp.items()
            if tables.n_levels(v[sel]) > 1
        ]
    if family == "transfer":
        m = p.h_has_src & sel
        src = {k: v[m] for k, v in p.h_src_ctx.items()}
        out = []
        for k in usable(src):
            strata = ms.combine(
                p.h_ctx[k][m], p.h_ctx["crowd"][m], p.h_dist_bin[m], np.maximum(p.h_block[m], 0)
            )
            for r, v in p.h_resp.items():
                if tables.n_levels(v[m]) > 1:
                    out.append(
                        ms.Pair(
                            f"{k}>{r}", src[k], tables.n_levels(src[k]), v[m], tables.n_levels(v[m]), strata
                        )
                    )
        return out
    raise ValueError(f"unknown family {family!r}")


def row_block(p: Prepared, family: str) -> np.ndarray:
    """The block of each row of the family's pairs."""
    if family.startswith("code"):
        return p.s_block
    if family == "reception":
        return p.h_block
    return p.h_block[p.h_has_src]


def run_family(
    p: Prepared, family: str, n_perm: int, rng: np.random.Generator, min_size: int = spec.MIN_STRATUM
) -> ms.FamilyResult | None:
    """The pooled test of one family, or None when it has no usable pair or fewer than 100 rows."""
    pairs = build_family(p, family)
    if not pairs or len(pairs[0].x) < 100:
        return None
    return ms.family_test(pairs, n_perm, rng, min_size)


def pair_blocks(
    p: Prepared,
    family: str,
    name: str,
    n_perm: int,
    rng: np.random.Generator,
    parity: int | None = None,
    min_size: int = spec.MIN_STRATUM,
) -> np.ndarray:
    """The excess bits of the pair ``name`` per block (NaN where a block cannot be estimated). ``parity`` 0 or
    1 computes only the even or odd blocks. A pair absent from this run's family gives all-NaN."""
    nb = p.n_blocks
    pairs = {q.name: q for q in build_family(p, family)}
    if name not in pairs:
        return np.full(nb, np.nan)
    use = None if parity is None else [b for b in range(nb) if b % 2 == parity]
    return ms.block_excess(pairs[name], row_block(p, family), nb, n_perm, rng, min_size, use)


def run_all(
    tab: Tables,
    window: tuple[int, int],
    *,
    n_perm: int = spec.N_PERM,
    n_perm_block: int = spec.N_PERM_BLOCK,
    seed: int = spec.SEED,
    families: tuple[str, ...] = FAMILIES,
    prepared: Prepared | None = None,
    log: Callable[[str], None] = lambda m: None,
) -> dict[str, Any]:
    """Pooled test and block series of every family of one run, JSON-safe."""
    p = prepared or prepare(tab, window)
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {
        "n_senders": len(p.s_sym),
        "n_hearers": len(p.h_heard),
        "n_hearers_with_source": int(p.h_has_src.sum()),
        "sym_thr": p.sym_thr,
        "heard_thr": p.heard_thr,
        "edges": p.edges,
        "dist_edges": p.dist_edges.tolist(),
        "families": {},
    }
    for fam in families:
        res = run_family(p, fam, n_perm, rng)
        if res is None:
            out["families"][fam] = None
            continue
        log(f"voice: {fam}: best {res.names[res.best]} z={res.z[res.best]:.1f}")
        best = res.names[res.best]
        out["families"][fam] = {
            "best": res.best_summary(),
            "pairs": res.table(),
            "blocks": [
                None if not np.isfinite(v) else float(v)
                for v in pair_blocks(p, fam, best, n_perm_block, rng, None)
            ],
        }
    return out
