"""MODES adapted to phenotype skeletons (DESIGN §44.5). Numpy only; **exploratory, outside the Holm family**
(O8).

MODES (Dolson, Vostinar, Wiser, Ofria 2019, Artificial Life 25:50-73) proposes four hallmarks of open-ended
evolution: change, novelty, complexity and ecological potential. The formulas here were **not** checked
against the paper (abstract only) and are our adaptation to continuous genomes; they are labelled adapted
wherever they are printed.

* **Skeleton.** A tuple of three-bin quantile indices of up to six trait columns: the first principal
component of the four tag traits, ``buoy_cap``, ``offspring_frac`` and the feature columns the file carries.
The bin edges come from a **codebook** built once on a reference window and shared by every arm (at most 729
classes). A class is *present* at a frame when it holds at least 0.1% of the alive organisms and is
*persistent* when present at two consecutive frames (suppresses mutational noise).
* **Change**: (i) the Jensen-Shannon distance between the class abundances at ``t`` and ``t - 1000`` steps;
  (ii) the share of parent-offspring pairs, both observed, whose skeletons differ (needs the births log).
* **Novelty**: classes persistent in the window that were never persistent in the reference or an earlier
  window, per block; the baseline is the same count with the frame order shuffled (a within-frame permutation
  of class labels leaves the abundances, and so this count, unchanged, so it cannot serve here).
* **Complexity**: the *channel count* of an organism (behavioural channels above their reference threshold):
  mean, 90th percentile and the maximum along the observed ancestry of the last frame. A proxy; the channels
  the frames carry are ``bond`` and ``move`` until features register theirs (``register_channel``), so the
  percentile is coarse in a pre-feature world. The compressed-genome proxy of the specification needs genomes
  the frames do not hold and is not computed.
* **Ecology**: Hill numbers (q = 0, 1, 2) of the present classes; occupied niches (height band x energy
tercile, a proxy: game labels and resource regimes are not in the frames) with at least 0.5% of the mass;
class pairs co-occurring in a body (a proxy for bond pairs, the partner lists are not in the frames) with at
least 20 occurrences per frame.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

import numpy as np

from alife.analysis.emergence.epr import WINDOW_BLOCKS, js_divergence
from alife.analysis.emergence.frames import Frames, dequantise

MODES_VERSION: Final = 1
N_BINS: Final = 3
MAX_COLS: Final = 6
PRESENT_FRAC: Final = 0.001
NICHE_FRAC: Final = 0.005
PAIR_MIN: Final = 20
CHANGE_LAG: Final = 1000  # steps
TAG_COLS: Final = ("tag0", "tag1", "tag2", "tag3")
CORE_COLS: Final = ("buoy_cap", "offspring_frac")
NOT_FEATURE: Final = {*TAG_COLS, *CORE_COLS, "energy", "age", "n_bonds", "speed"}
SEED: Final = 0x30DE5

# channel name -> (trait column, rule): "pos" counts values > 0, "hi" values above the reference median
CHANNELS: dict[str, tuple[str, str]] = {"bond": ("n_bonds", "pos"), "move": ("speed", "hi")}


def register_channel(name: str, column: str, rule: str = "pos") -> None:
    """A feature PR adds one line: a behavioural channel read from one registered trait column."""
    if rule not in ("pos", "hi"):
        raise ValueError(f"unknown channel rule {rule!r}")
    CHANNELS[name] = (column, rule)


CAVEATS: Final = (
    "MODES formulas are our adaptation (the paper was not available beyond its abstract); exploratory (O8).",
    "Complexity is a channel-count proxy, not Kolmogorov complexity.",
    "Niche and interaction pairs are proxies (height x energy; co-membership of a body).",
)


@dataclass
class Codebook:
    """Fixed skeleton definition shared by all arms (``codebook.json``)."""

    columns: list[str]  # "tag_pc1" is the synthetic first principal component of the tags
    tag_mean: list[float]
    tag_pc: list[float]
    edges: list[list[float]]  # per column: the two cut points
    channel_thresholds: dict[str, float]  # "hi" channels: the reference median of the column
    energy_edges: list[float]  # tercile cuts of energy (niche proxy)
    height_bands: int
    version: int = MODES_VERSION
    n_ref_rows: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_codes(self) -> int:
        return N_BINS ** len(self.columns)

    def to_json(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Codebook:
        if d.get("version") != MODES_VERSION:
            raise ValueError(f"codebook version {d.get('version')}, this code reads {MODES_VERSION}")
        return cls(**d)


def _columns(fr: Frames, names: Sequence[str]) -> np.ndarray:
    return np.stack([fr.trait(n).astype(np.float64) for n in names], axis=1)


def build_codebook(refs: Sequence[Frames], *, max_rows: int = 200_000, max_cols: int = MAX_COLS) -> Codebook:
    """The codebook from reference frames (every ``k``-th row is used up to ``max_rows``). Needs the tag
    traits; feature columns are those beyond the base registry that the first reference carries."""
    fr0 = refs[0]
    names = list(fr0.trait_names)
    if not all(t in names for t in TAG_COLS):
        raise ValueError("the reference frames need tag0..tag3")
    cols = ["tag_pc1"] + [c for c in CORE_COLS if c in names]
    cols += [c for c in names if c not in NOT_FEATURE][: max(max_cols - len(cols), 0)]
    cols = cols[:max_cols]

    def sample(fr: Frames) -> dict[str, np.ndarray]:
        n = len(fr.uid)
        step = max(1, n // max(max_rows // len(refs), 1))
        idx = np.arange(0, n, step)
        return {nm: fr.trait(nm)[idx].astype(np.float64) for nm in names}

    pooled: dict[str, list[np.ndarray]] = {}
    for fr in refs:
        for k, v in sample(fr).items():
            pooled.setdefault(k, []).append(v)
    data = {k: np.concatenate(v) for k, v in pooled.items()}
    tags = np.stack([data[t] for t in TAG_COLS], axis=1)
    ok = np.isfinite(tags).all(1)
    mean = tags[ok].mean(0)
    cov = np.cov((tags[ok] - mean).T)
    w, v = np.linalg.eigh(cov)
    pc = v[:, int(np.argmax(w))]
    if pc[np.argmax(np.abs(pc))] < 0:
        pc = -pc  # a fixed sign convention
    edges = []
    for c in cols:
        x = (tags[ok] - mean) @ pc if c == "tag_pc1" else data[c]
        x = x[np.isfinite(x)]
        edges.append([float(q) for q in np.quantile(x, [1 / 3, 2 / 3])] if x.size else [0.0, 0.0])
    thr = {}
    for ch, (col, rule) in CHANNELS.items():
        if rule == "hi" and col in data:
            x = data[col][np.isfinite(data[col])]
            thr[ch] = float(np.median(x)) if x.size else 0.0
    en = data.get("energy", np.zeros(1))
    en = en[np.isfinite(en)]
    en_edges = [float(q) for q in np.quantile(en, [1 / 3, 2 / 3])] if en.size else [0.0, 0.0]
    height = fr0.height if fr0.nd == 3 else 0.0
    return Codebook(
        columns=cols,
        tag_mean=[float(m) for m in mean],
        tag_pc=[float(p) for p in pc],
        edges=edges,
        channel_thresholds=thr,
        energy_edges=en_edges,
        height_bands=max(1, round(height / 8.0)) if height else 1,
        n_ref_rows=len(tags),
        meta={
            "reference": [r.meta.get("run") for r in refs],
            "windows": [r.meta.get("window") for r in refs],
        },
    )


def extend_codebook(cb: Codebook, fr: Frames, max_cols: int = MAX_COLS) -> Codebook:
    """The codebook plus the feature columns ``fr`` carries that the reference did not (up to ``max_cols``).

    A reference without the feature (the baseline) cannot hold its cut points, so the new columns take tercile
    cuts from the arm under test and are listed in ``meta["extended"]``; without them P2 and P3 would be blind
    to the feature. Existing columns keep their codes (new columns are the high digits), so an earlier
    ``seen`` set stays valid after ``pad_seen``."""
    extra = [c for c in fr.trait_names if c not in NOT_FEATURE and c not in cb.columns]
    add = extra[: max(max_cols - len(cb.columns), 0)]
    if not add:
        return cb
    cuts = []
    for c in add:
        x = fr.trait(c).astype(np.float64)
        x = x[np.isfinite(x)]
        cuts.append([float(q) for q in np.quantile(x, [1 / 3, 2 / 3])] if x.size else [0.0, 0.0])
    return replace(
        cb,
        columns=[*cb.columns, *add],
        edges=[*cb.edges, *cuts],
        meta={**cb.meta, "extended": {c: "cut points of this arm" for c in add}},
    )


def pad_seen(seen: np.ndarray, cb: Codebook) -> np.ndarray:
    """``seen`` (bool over the classes of an earlier, narrower codebook) padded to ``cb.n_codes``."""
    if len(seen) >= cb.n_codes:
        return seen
    return np.concatenate([seen, np.zeros(cb.n_codes - len(seen), bool)])


def write_codebook(path: Path, cb: Codebook) -> Path:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    try:
        tmp.write_text(json.dumps(cb.to_json(), indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def read_codebook(path: Path) -> Codebook:
    return Codebook.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


# --- skeletons, classes ---


def skeletons(fr: Frames, cb: Codebook) -> np.ndarray:
    """The class code (int32) of every row: ``sum_k bin_k * 3^k`` over the codebook columns; a column the file
    lacks, or a NaN, contributes bin 0."""
    code = np.zeros(len(fr.uid), np.int64)
    have = set(fr.trait_names)
    for k, (c, e) in enumerate(zip(cb.columns, cb.edges, strict=True)):
        if c == "tag_pc1":
            if not all(t in have for t in TAG_COLS):
                continue
            x = (_columns(fr, TAG_COLS) - np.asarray(cb.tag_mean)) @ np.asarray(cb.tag_pc)
        elif c in have:
            x = fr.trait(c).astype(np.float64)
        else:
            continue
        b = np.searchsorted(np.asarray(e), x, side="right")
        b[~np.isfinite(x)] = 0
        code += b.astype(np.int64) * N_BINS**k
    return code.astype(np.int32)


def persistent_classes(fr: Frames, cb: Codebook) -> np.ndarray:
    """Bool ``(n_codes,)``: the classes persistent at some frame of ``fr`` (the ``seen`` set after a
    window)."""
    return persistent(class_counts(fr, skeletons(fr, cb), cb.n_codes)).any(0)


def frame_index(fr: Frames) -> np.ndarray:
    return np.repeat(np.arange(fr.n_frames), np.diff(fr.off))


def class_counts(fr: Frames, codes: np.ndarray, n_codes: int) -> np.ndarray:
    """``(F, n_codes)`` abundances per frame."""
    f = frame_index(fr)
    return np.bincount(f * n_codes + codes, minlength=fr.n_frames * n_codes).reshape(fr.n_frames, n_codes)


def persistent(counts: np.ndarray) -> np.ndarray:
    """``(F, n_codes)`` bool: present (>= 0.1% of the alive) at the frame and at a neighbouring frame."""
    tot = counts.sum(1, keepdims=True)
    present = counts >= np.maximum(PRESENT_FRAC * tot, 1)
    nb = np.zeros_like(present)
    nb[1:] |= present[:-1]
    nb[:-1] |= present[1:]
    return present & nb


def channel_counts(fr: Frames, cb: Codebook) -> tuple[np.ndarray, list[str]]:
    """Channels used per row and the channel names the file supports."""
    have = set(fr.trait_names)
    n = np.zeros(len(fr.uid), np.int16)
    used = []
    for ch, (col, rule) in CHANNELS.items():
        if col not in have:
            continue
        x = fr.trait(col).astype(np.float64)
        with np.errstate(invalid="ignore"):
            on = x > 0 if rule == "pos" else x > cb.channel_thresholds.get(ch, np.inf)
        n += on.astype(np.int16)
        used.append(ch)
    return n, used


# --- statistics ---


def block_of_frames(n_frames: int, n_blocks: int) -> np.ndarray:
    return (np.arange(n_frames) * n_blocks) // max(n_frames, 1)


def _block_mean(series: np.ndarray, blk: np.ndarray, n_blocks: int) -> list[float | None]:
    out: list[float | None] = []
    for b in range(n_blocks):
        v = series[blk == b]
        v = v[np.isfinite(v)]
        out.append(float(v.mean()) if v.size else None)
    return out


def change_pop(fr: Frames, counts: np.ndarray) -> np.ndarray:
    """sqrt of the JS divergence (nats) between abundances at ``t`` and ``t - 1000`` steps; nan without a
    frame that far back."""
    out = np.full(fr.n_frames, np.nan)
    steps = fr.step
    for t in range(fr.n_frames):
        j = int(np.searchsorted(steps, steps[t] - CHANGE_LAG))
        if j < t and steps[j] == steps[t] - CHANGE_LAG:
            out[t] = math.sqrt(max(js_divergence(counts[t], counts[j]), 0.0))
    return out


def load_births(run: Path, t0: int, t1: int) -> dict[str, np.ndarray]:
    """``uid`` and ``parent_uid`` of the births with ``t0 < step <= t1`` from the run's event chunks."""
    uid, par = [], []
    for p in sorted(Path(run, "events").glob("births_*.npz")):
        s0, s1 = (int(x) for x in p.stem.split("_")[1:3])
        if s1 <= t0 or s0 > t1:
            continue
        z = np.load(p)
        m = (z["step"] > t0) & (z["step"] <= t1)
        uid.append(z["uid"][m])
        par.append(z["parent_uid"][m])
    cat = (lambda v: np.concatenate(v)) if uid else (lambda v: np.zeros(0, np.int64))
    return {"uid": cat(uid), "parent_uid": cat(par)}


def first_observation(fr: Frames, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(uids, first code, first frame)`` of every organism seen."""
    uids, idx = np.unique(fr.uid, return_index=True)
    return uids, codes[idx], frame_index(fr)[idx]


def change_edge(
    fr: Frames,
    codes: np.ndarray,
    births: dict[str, np.ndarray],
    n_blocks: int,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Share of parent-offspring pairs (both observed in the frames) whose skeletons differ, per block of the
    child's first frame; ``rng`` permutes the children's codes among children (the baseline: same code
    marginals, no heritable association). ``coverage`` is the observed pairs over the births of the window."""
    uids, c0, f0 = first_observation(fr, codes)
    cu, pu = births["uid"], births["parent_uid"]
    ic = np.searchsorted(uids, cu)
    ic = np.clip(ic, 0, max(len(uids) - 1, 0))
    ip = np.clip(np.searchsorted(uids, pu), 0, max(len(uids) - 1, 0))
    ok = (len(uids) > 0) & (uids[ic] == cu) & (uids[ip] == pu)
    cc, pc, fc = c0[ic[ok]], c0[ip[ok]], f0[ic[ok]]
    if rng is not None:
        cc = rng.permutation(cc)
    diff = (cc != pc).astype(np.float64)
    blk = block_of_frames(fr.n_frames, n_blocks)[fc] if len(fc) else np.zeros(0, np.int64)
    blocks = [float(diff[blk == b].mean()) if (blk == b).any() else None for b in range(n_blocks)]
    return {
        "blocks": blocks,
        "share": float(diff.mean()) if diff.size else None,
        "pairs": int(ok.sum()),
        "coverage": float(ok.sum() / len(cu)) if len(cu) else None,
    }


def novelty(
    pers: np.ndarray, seen: np.ndarray, n_blocks: int, order: np.ndarray | None = None
) -> tuple[list[int], np.ndarray]:
    """New classes per block: persistent classes whose first persistent frame is in the block and which are
    not in ``seen`` (bool ``(n_codes,)``, the reference and earlier windows). ``order`` reorders the frames
    (the shuffled baseline). Returns the per-block counts and the window's persistent classes."""
    p = pers if order is None else pers[order]
    anyp = p.any(0)
    first = np.where(anyp, p.argmax(0), -1)
    blk = block_of_frames(len(p), n_blocks)
    new = anyp & ~seen
    counts = [int((new & (first >= 0) & (blk[np.maximum(first, 0)] == b)).sum()) for b in range(n_blocks)]
    return counts, anyp


def hill(counts: np.ndarray, present: np.ndarray) -> tuple[float, float, float]:
    """Hill numbers q = 0, 1, 2 of the abundances of the present classes."""
    c = counts[present].astype(np.float64)
    if c.sum() <= 0:
        return (0.0, 0.0, 0.0)
    p = c / c.sum()
    return (float(len(p)), float(math.exp(-(p * np.log(p)).sum())), float(1.0 / (p * p).sum()))


def niches(fr: Frames, cb: Codebook) -> np.ndarray:
    """Occupied niches per frame: height band x energy tercile with at least 0.5% of that frame's mass."""
    f = frame_index(fr)
    z = dequantise(fr.pos, fr.size, fr.height)[:, 2] if fr.nd == 3 else None
    band = (
        np.clip((z / max(fr.height, 1e-9) * cb.height_bands).astype(np.int64), 0, cb.height_bands - 1)
        if z is not None
        else np.zeros(len(fr.uid), np.int64)
    )
    en = np.searchsorted(np.asarray(cb.energy_edges), fr.trait("energy")) if "energy" in fr.trait_names else 0
    niche = band * N_BINS + en
    k = cb.height_bands * N_BINS
    cnt = np.bincount(f * k + niche, minlength=fr.n_frames * k).reshape(fr.n_frames, k)
    tot = cnt.sum(1, keepdims=True)
    return (cnt >= NICHE_FRAC * np.maximum(tot, 1)).sum(1).astype(np.float64)


def pair_counts(fr: Frames, codes: np.ndarray, n_codes: int, stride: int = 10) -> np.ndarray:
    """Distinct class pairs co-occurring in a body with at least ``PAIR_MIN`` occurrences, at every
    ``stride``-th frame (nan for the others): the interaction-diversity proxy."""
    out = np.full(fr.n_frames, np.nan)
    for i in range(0, fr.n_frames, stride):
        s = fr.span(i)
        body, code = fr.body[s], codes[s]
        m = body >= 0
        if not m.any():
            out[i] = 0.0
            continue
        key = body[m].astype(np.int64) * n_codes + code[m]
        uk, cnt = np.unique(key, return_counts=True)
        b = uk // n_codes
        c = uk % n_codes
        bounds = np.flatnonzero(np.diff(b, prepend=-1))
        bounds = np.append(bounds, len(b))
        tot: dict[int, int] = {}
        for lo, hi in itertools.pairwise(bounds):
            cs, ns = c[lo:hi].tolist(), cnt[lo:hi].tolist()
            for x, (cx, nx) in enumerate(zip(cs, ns, strict=True)):
                if nx > 1:
                    tot[cx * n_codes + cx] = tot.get(cx * n_codes + cx, 0) + nx * (nx - 1) // 2
                for cy, ny in zip(cs[x + 1 :], ns[x + 1 :], strict=True):
                    tot[cx * n_codes + cy] = tot.get(cx * n_codes + cy, 0) + nx * ny
        out[i] = float(sum(v >= PAIR_MIN for v in tot.values()))
    return out


def lineage_max(
    c_last: dict[int, int], parent_of: dict[int, int], final: Sequence[int], depth: int = 10_000
) -> int | None:
    """The maximum channel count along the observed ancestry of the final organisms (ancestors not observed
    end the chain)."""
    best = None
    for u in final:
        x = u
        for _ in range(depth):
            if x not in c_last:
                break
            best = c_last[x] if best is None else max(best, c_last[x])
            x = parent_of.get(x, -1)
            if x < 0:
                break
    return best


def analyse(
    fr: Frames,
    cb: Codebook,
    *,
    seen: np.ndarray | None = None,
    births: dict[str, np.ndarray] | None = None,
    n_blocks: int = WINDOW_BLOCKS,
    seed: int = SEED,
    n_shuffle: int = 20,
) -> dict[str, Any]:
    """MODES of one frames file against a codebook: per-block series (the units of the comparison) plus window
    summaries. ``seen`` are the classes persistent in the reference or earlier windows (default none)."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, fr.n_frames]))
    nc = cb.n_codes
    codes = skeletons(fr, cb)
    counts = class_counts(fr, codes, nc)
    pers = persistent(counts)
    blk = block_of_frames(fr.n_frames, n_blocks)
    seen = np.zeros(nc, bool) if seen is None else seen
    nov, win_pers = novelty(pers, seen, n_blocks)
    shuffled = []
    for _ in range(n_shuffle):
        sc, _ = novelty(pers, seen, n_blocks, rng.permutation(fr.n_frames))
        shuffled.append(sc)
    cp = change_pop(fr, counts)
    h = np.array([hill(counts[i], pers[i]) for i in range(fr.n_frames)])
    chan, used = channel_counts(fr, cb)
    f = frame_index(fr)
    rows = [chan[f == i] for i in range(fr.n_frames)]
    p90 = np.array([np.percentile(r, 90) if r.size else np.nan for r in rows])
    mean_c = np.array([r.mean() if r.size else np.nan for r in rows])
    edge = None
    edge_null = None
    lineage = None
    if births is not None:
        edge = change_edge(fr, codes, births, n_blocks)
        edge_null = change_edge(fr, codes, births, n_blocks, rng)
        last = slice(fr.span(fr.n_frames - 1).start, fr.span(fr.n_frames - 1).stop)
        c_last = dict(zip(fr.uid[last].tolist(), chan[last].tolist(), strict=True))
        # the count of every observed organism at its last observation (ancestors have died by the last frame)
        rev_u = fr.uid[::-1]
        ru, ri = np.unique(rev_u, return_index=True)
        c_all = dict(zip(ru.tolist(), chan[::-1][ri].tolist(), strict=True))
        parent_of = dict(zip(births["uid"].tolist(), births["parent_uid"].tolist(), strict=True))
        lineage = lineage_max(c_all, parent_of, list(c_last))
    nich = niches(fr, cb)
    pairs = pair_counts(fr, codes, nc)
    return {
        "version": MODES_VERSION,
        "status": "exploratory",
        "adapted": True,
        "n_blocks": n_blocks,
        "codebook": {"columns": list(cb.columns), "extended": cb.meta.get("extended")},
        "classes_present": int(win_pers.sum()),
        "novelty": {
            "per_block": nov,
            "cumulative": [int(x) for x in np.cumsum(nov)],
            "shuffled_frames_mean": [float(np.mean([s[b] for s in shuffled])) for b in range(n_blocks)],
        },
        "change": {
            "pop": _block_mean(cp, blk, n_blocks),
            "edge": edge,
            "edge_permuted": edge_null,
        },
        "complexity": {
            "channels_used": used,
            "mean": _block_mean(mean_c, blk, n_blocks),
            "p90": _block_mean(p90, blk, n_blocks),
            "max_along_lineage": lineage,
        },
        "ecology": {
            "hill0": _block_mean(h[:, 0], blk, n_blocks),
            "hill1": _block_mean(h[:, 1], blk, n_blocks),
            "hill2": _block_mean(h[:, 2], blk, n_blocks),
            "niches": _block_mean(nich, blk, n_blocks),
            "interaction_pairs": _block_mean(pairs, blk, n_blocks),
        },
        "persistent_classes": [int(c) for c in np.flatnonzero(win_pers)],
        "caveats": list(CAVEATS),
    }
