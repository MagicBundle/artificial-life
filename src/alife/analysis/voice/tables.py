"""From observed steps to the tables of the voice analysis (numpy only; the replay lives in ``replay.py``).

A **snapshot** (``Snap``) is the host copy of one observed step: the start-of-step state's columns the
analysis
reads and the step's observer channel (what each row said, what it heard, who was sensed, what it did). A
**sample** at step ``u`` uses three consecutive snapshots ``u-1``, ``u`` and ``u+1``:

* **senders**, rows alive at ``u-1``: the call spoken at ``u-1`` (it is what the neighbours hear at ``u``)
  with
  the caller's context at the start of ``u-1``, when it chose the call;
* **hearers**, rows alive at ``u`` and ``u+1`` (same occupant): the call heard at ``u`` (the inputs that
  drove the decision), the hearer's own context at ``u``, its response during ``u`` (and the harvest and
  motion that the next state shows), and its **source**, the sensed neighbour whose call weighs most at the
  hearer (hearing weight ``a(d) * total intensity``, ``a`` the engine's falloff), with that caller's call,
  distance and context at ``u-1``. The source choice ignores the kin weighting (``comm.kin_exp``, off by
  default); the heard vector is the engine's own either way.

Rows are drawn at random (``SAMPLE_ROWS`` per sample, seeded by the step) so a log stays small. The tables
keep
raw values (counts, energies, intensities); the coding into a few levels is done at analysis time from pooled
quantiles (``code_context`` and friends), so the cut points are stored and the same for senders and hearers.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from alife.analysis.voice import measures as ms
from alife.analysis.voice.spec import CONTEXTS, RESPONSES

TABLES_VERSION = 1

# ActFlag bits (alife.step.ActFlag; tests pin the equality, so this module needs no MLX).
FLAG_ATTACKED, FLAG_FORMED, FLAG_BUDDED, FLAG_SPOKE = 1, 4, 8, 1024

CTX_RAW = ("loss", "food", "energy", "crowd", "bonds", "height", "wind", "budding")
APPROACH_COS = 0.2  # |cos| of the displacement and the direction to the source below which it is "neither"


class Snap(NamedTuple):
    """One observed step on the host. ``voice`` / ``heard`` are the observer channel of the step that starts
    at
    ``step`` (what was said in it, what was heard as its inputs); the other columns are the start state's."""

    step: int
    alive: np.ndarray  # (N,) bool
    uid: np.ndarray  # (N,) int64
    clade: np.ndarray  # (N,) int32
    lineage: np.ndarray  # (N,) int8 kin group of descent (see ``kin_groups``), -1 unassigned
    energy: np.ndarray  # (N,) float32
    loss: np.ndarray  # (N,) int32 last_loss
    harvest: np.ndarray  # (N,) int32 last_harvest (what the previous step harvested)
    pos: np.ndarray  # (N, d)
    bonds: np.ndarray  # (N,) int8 number of bond partners
    food: np.ndarray  # (N,) float32 resource at the row's field cell
    wind: np.ndarray  # (N,) float32 wind drift speed (0 without weather)
    voice: np.ndarray  # (N, W) uint8
    heard: np.ndarray  # (N, W) uint8
    audible: np.ndarray  # (N,) uint8
    nbr: np.ndarray  # (N, K) int32, -1 invalid
    flags: np.ndarray  # (N,) uint16
    thrust: np.ndarray  # (N,) int16


class Geo(NamedTuple):
    """What the sampler needs of the world: the host geometry and the hearing law of the engine."""

    geo: Any  # alife.geom Box | Sphere
    d3: bool
    hear_range: float
    atten_pow: float


def _ctx_raw(sn: Snap, rows: np.ndarray, g: Geo) -> dict[str, np.ndarray]:
    nb = sn.nbr[rows]
    height = g.geo.height(sn.pos[rows]) if g.d3 else np.zeros(len(rows), np.float32)
    return {
        "loss": sn.loss[rows],
        "food": sn.food[rows],
        "energy": sn.energy[rows],
        "crowd": (nb >= 0).sum(axis=1).astype(np.int16),
        "bonds": sn.bonds[rows],
        "height": np.asarray(height, dtype=np.float32),
        "wind": sn.wind[rows],
        "budding": ((sn.flags[rows] & FLAG_BUDDED) != 0).astype(np.int8),
    }


def _pick_source(prev: Snap, cur: Snap, rows: np.ndarray, g: Geo) -> tuple[np.ndarray, ...]:
    """For the hearers ``rows``: ``(has, j, dist/range)`` of the sensed neighbour with the largest hearing
    weight ``a(d) * total intensity spoken at u-1`` (the occupant of the row must be the same at u-1 and
    u)."""
    nb = cur.nbr[rows]  # (m, K)
    ok = nb >= 0
    j = np.where(ok, nb, 0)
    same = cur.alive[j] & prev.alive[j] & (cur.uid[j] == prev.uid[j])
    d = g.geo.distance(cur.pos[rows][:, None, :], cur.pos[j])  # (m, K)
    a = np.clip(1.0 - d / g.hear_range, 0.0, 1.0) ** g.atten_pow
    loud = prev.voice[j].astype(np.float64).sum(axis=2)  # (m, K)
    w = np.where(ok & same, a * loud, 0.0)
    k = w.argmax(axis=1)
    m = np.arange(len(rows))
    has = w[m, k] > 0
    return has, j[m, k], np.where(has, d[m, k] / g.hear_range, np.nan)


def build_sample(
    prev: Snap, cur: Snap, nxt: Snap, g: Geo, rows: int, rng: np.random.Generator
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """The sender and hearer rows of the sample at step ``cur.step`` (see the module docstring)."""
    # senders: alive at u-1
    cand = np.flatnonzero(prev.alive)
    srows = np.sort(rng.choice(cand, size=min(rows, cand.size), replace=False)) if cand.size else cand
    senders: dict[str, np.ndarray] = {
        "step": np.full(srows.size, cur.step, np.int32),
        "uid": prev.uid[srows],
        "clade": prev.clade[srows],
        "lineage": prev.lineage[srows],
        "voice": prev.voice[srows],
        **{f"c_{k}": v for k, v in _ctx_raw(prev, srows, g).items()},
    }
    # hearers: alive at u and u+1, same occupant
    ok = cur.alive & nxt.alive & (cur.uid == nxt.uid)
    cand = np.flatnonzero(ok)
    hrows = np.sort(rng.choice(cand, size=min(rows, cand.size), replace=False)) if cand.size else cand
    m = hrows.size
    has, j, dist = _pick_source(prev, cur, hrows, g)
    disp = g.geo.displacement(cur.pos[hrows], nxt.pos[hrows])  # (m, d) moved during u
    to_src = g.geo.displacement(cur.pos[hrows], cur.pos[j])
    nd, ns = np.linalg.norm(disp, axis=1), np.linalg.norm(to_src, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = (disp * to_src).sum(axis=1) / (nd * ns)
    approach = np.where(has & (nd > 0) & (ns > 0), np.sign(cos) * (np.abs(cos) > APPROACH_COS), 0).astype(
        np.int8
    )
    fl = cur.flags[hrows]
    hearers: dict[str, np.ndarray] = {
        "step": np.full(m, cur.step, np.int32),
        "uid": cur.uid[hrows],
        "clade": cur.clade[hrows],
        "lineage": cur.lineage[hrows],
        "heard": cur.heard[hrows],
        "audible": cur.audible[hrows],
        **{f"c_{k}": v for k, v in _ctx_raw(cur, hrows, g).items()},
        "r_thrust": np.abs(cur.thrust[hrows].astype(np.int32)).astype(np.int16),
        "r_attack": ((fl & FLAG_ATTACKED) != 0).astype(np.int8),
        "r_bond": ((fl & FLAG_FORMED) != 0).astype(np.int8),
        "r_eat": nxt.harvest[hrows].astype(np.int32),
        "r_speak": ((fl & FLAG_SPOKE) != 0).astype(np.int8),
        "r_loud": cur.voice[hrows].astype(np.int32).sum(axis=1).astype(np.int16),
        "r_approach": approach,
        "has_src": has.astype(np.int8),
        "src_uid": np.where(has, cur.uid[j], -1),
        "src_clade": np.where(has, cur.clade[j], -1),
        "src_lineage": np.where(has, cur.lineage[j], -1).astype(np.int8),
        "src_voice": np.where(has[:, None], prev.voice[j], 0).astype(np.uint8),
        "src_dist": np.where(has, dist, np.nan).astype(np.float32),
    }
    # the source caller's context at u-1, when it chose the call (zeros where there is no source)
    sc = _ctx_raw(prev, j, g)
    for k, v in sc.items():
        hearers[f"src_c_{k}"] = np.where(has, v, 0).astype(v.dtype)
    return senders, hearers


class Tables:
    """Sender and hearer rows plus the run's meta, with npz round trip."""

    def __init__(
        self,
        senders: dict[str, np.ndarray],
        hearers: dict[str, np.ndarray],
        meta: dict[str, Any],
        cf: dict[str, np.ndarray] | None = None,
    ):
        self.senders, self.hearers, self.meta, self.cf = senders, hearers, meta, cf or {}

    @property
    def width(self) -> int:
        return int(self.senders["voice"].shape[1])

    def save(self, path: Path) -> None:
        arrays: dict[str, np.ndarray] = {}
        arrays.update({f"s_{k}": v for k, v in self.senders.items()})
        arrays.update({f"h_{k}": v for k, v in self.hearers.items()})
        arrays.update({f"cf_{k}": v for k, v in self.cf.items()})
        arrays["meta_json"] = np.frombuffer(
            json.dumps({**self.meta, "tables_version": TABLES_VERSION}, sort_keys=True).encode("utf-8"),
            dtype=np.uint8,
        )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, **arrays)
        tmp.replace(path)

    @staticmethod
    def load_meta(path: Path) -> dict[str, Any]:
        """Only the meta of a ``tables.npz`` (cheap: the arrays are not read)."""
        with np.load(Path(path), allow_pickle=False) as z:
            return json.loads(bytes(z["meta_json"]).decode("utf-8"))

    @classmethod
    def load(cls, path: Path) -> Tables:
        with np.load(Path(path), allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta_json"]).decode("utf-8"))
            if meta.get("tables_version") != TABLES_VERSION:
                raise ValueError(
                    f"{path}: tables version {meta.get('tables_version')}, expected {TABLES_VERSION}"
                )
            pick = lambda p: {k[len(p) :]: z[k] for k in z.files if k.startswith(p)}  # noqa: E731
            return cls(pick("s_"), pick("h_"), meta, pick("cf_"))


def concat(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        return {}
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


# --- coding -----------------------------------------------------------------------------------------------

Edges = dict[str, list[float]]


def fit_edges(senders: Mapping[str, np.ndarray], hearers: Mapping[str, np.ndarray]) -> Edges:
    """The cut points of the continuous contexts and responses: terciles of the pooled sender sample
    (contexts)
    and of the hearer sample (``move``, ``loud``, ``eat``)."""
    edges: Edges = {}
    for k in ("food", "energy", "crowd", "height", "wind"):
        edges[k] = ms.quantile_edges(senders[f"c_{k}"], CONTEXTS[k][0]).tolist()
    edges["move"] = ms.quantile_edges(hearers["r_thrust"], RESPONSES["move"][0]).tolist()
    edges["loud"] = ms.quantile_edges(hearers["r_loud"], RESPONSES["loud"][0]).tolist()
    edges["eat"] = ms.quantile_edges(hearers["r_eat"], RESPONSES["eat"][0]).tolist()
    return edges


def code_context(raw: Mapping[str, np.ndarray], edges: Edges, prefix: str = "c_") -> dict[str, np.ndarray]:
    """The context codes (int8) of a table's raw columns; a context without spread has one level."""
    out = {
        "hurt": (raw[f"{prefix}loss"] > 0).astype(np.int8),
        "bonded": (raw[f"{prefix}bonds"] > 0).astype(np.int8),
        "budding": (raw[f"{prefix}budding"] > 0).astype(np.int8),
    }
    for k in ("food", "energy", "crowd", "height", "wind"):
        out[k] = ms.code_by_edges(raw[f"{prefix}{k}"], np.asarray(edges[k]))
    return out


def code_response(h: Mapping[str, np.ndarray], edges: Edges) -> dict[str, np.ndarray]:
    return {
        "move": ms.code_by_edges(h["r_thrust"], np.asarray(edges["move"])),
        "attack": h["r_attack"].astype(np.int8),
        "bond": h["r_bond"].astype(np.int8),
        "eat": ms.code_by_edges(h["r_eat"], np.asarray(edges["eat"])),
        "speak": h["r_speak"].astype(np.int8),
        "loud": ms.code_by_edges(h["r_loud"], np.asarray(edges["loud"])),
        "approach": (h["r_approach"].astype(np.int8) + 1).astype(np.int8),
    }


def n_levels(codes: np.ndarray) -> int:
    return int(codes.max(initial=0)) + 1


def block_of(step: np.ndarray, start: int, window: int, n_blocks: int) -> np.ndarray:
    """Block index of each sample step in the window ``(start, start + window]`` (-1 outside)."""
    b = (np.asarray(step, dtype=np.int64) - start - 1) // (window // n_blocks)
    return np.where((np.asarray(step) > start) & (b < n_blocks), b, -1).astype(np.int32)


def kin_groups(tag: np.ndarray, k: int, seed: int = 0, iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's k-means of the kin tags ``(n, T)``: ``(labels (n,) int8, centroids (k, T))``. Deterministic
    for a
    seed; an empty cluster keeps its centre. The tags are the engine's heritable kin marker, so a group is a
    cluster of kin; the genealogy then carries the label to the descendants (``replay``)."""
    tag = np.asarray(tag, dtype=np.float64)
    k = max(1, min(k, len(tag)))
    rng = np.random.default_rng(seed)
    c = tag[rng.choice(len(tag), k, replace=False)]
    lab = np.zeros(len(tag), np.int64)
    for _ in range(iters):
        lab = ((tag[:, None, :] - c[None]) ** 2).sum(axis=2).argmin(axis=1)
        new = np.stack([tag[lab == i].mean(axis=0) if (lab == i).any() else c[i] for i in range(k)])
        if np.allclose(new, c):
            break
        c = new
    order = np.argsort(-np.bincount(lab, minlength=k), kind="stable")  # group 0 is the largest
    rank = np.empty(k, np.int64)
    rank[order] = np.arange(k)
    return rank[lab].astype(np.int8), c[order]


def assign_groups(tag: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """The nearest centroid (int8) of each tag."""
    d = ((np.asarray(tag, np.float64)[:, None, :] - centroids[None]) ** 2).sum(axis=2)
    return d.argmin(axis=1).astype(np.int8)
