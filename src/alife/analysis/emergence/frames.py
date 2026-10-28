"""The frame file of the window recorder and its readers (DESIGN §44.2). Numpy only.

``frames_<T0>_<T1>.npz`` (``FRAMES_VERSION`` 1) holds the alive organisms at a dense cadence in CSR layout:
frame ``i`` is the row range ``off[i]:off[i+1]`` of every per-organism array. Positions are quantised to
``uint16`` (``quantise``): x, y over ``[0, L)`` and z over ``[0, H]``, so the error is at most half a
quantum, ``L/131070``, and always below ``L/65535`` (0.0024 units in the mini world). The optional **fine
set** follows the members of the ``fine_groups`` largest bodies at ``T0`` every ``fine_tau`` steps (positions
and body label only). Two readers produce a ``Frames``: ``read_frames`` (the recorder's file) and
``read_frma`` (a play stream's ``FRMA`` records, without clade, traits or drift: an alternative source, not
the primary path).
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

FRAMES_VERSION: Final = 1
QMAX: Final = 65535

_STREAM_MAGIC: Final = b"ALSTRM\x00\x01"  # streamrec.MAGIC, repeated so this module stays MLX-free
_HEAD: Final = struct.Struct("<QIHH")
_FRMA_PREFIX: Final = struct.Struct("<IIIII")  # n, n_bonds, color_index, span, N


class FramesError(ValueError):
    """A frame file or stream that cannot be read (wrong version, truncated, inconsistent offsets)."""


def extent(nd: int, size: float, height: float) -> np.ndarray:
    """Per-axis extent used by the quantiser: ``(L, L)`` in 2D, ``(L, L, H)`` in 3D."""
    return np.asarray([size, size, height][:nd] if nd == 3 else [size, size], dtype=np.float64)


def quantise(pos: np.ndarray, size: float, height: float = 0.0) -> np.ndarray:
    """Positions ``(n, nd)`` to ``uint16`` codes ``round(x / extent * 65535)``, clipped to the code range."""
    p = np.asarray(pos, dtype=np.float64)
    q = np.rint(p / extent(p.shape[1], size, height) * QMAX)
    return np.clip(q, 0, QMAX).astype(np.uint16)


def dequantise(q: np.ndarray, size: float, height: float = 0.0) -> np.ndarray:
    """The positions of the codes, float64 ``(n, nd)`` (x, y can be exactly ``L``: take them modulo ``L``)."""
    q = np.asarray(q)
    return q.astype(np.float64) * extent(q.shape[1], size, height) / QMAX


@dataclass
class Frames:
    """The frames of one window (see the module docstring). ``fine`` is None when no fine set was recorded."""

    meta: dict[str, Any]
    step: np.ndarray  # (F,) int64
    off: np.ndarray  # (F+1,) int64
    uid: np.ndarray  # (n,) int64
    row: np.ndarray  # (n,) int32
    pos: np.ndarray  # (n, nd) uint16
    clade: np.ndarray  # (n,) int32
    body: np.ndarray  # (n,) int32, -1 unbonded
    drift: np.ndarray | None = None  # (n, nd) float16, wind drift; absent without weather
    trait_names: list[str] = field(default_factory=list)
    traits: np.ndarray | None = None  # (n, K) float32
    fine: dict[str, np.ndarray] | None = (
        None  # fine_step, fine_off, fine_uid, fine_group, fine_pos, fine_body
    )

    def __post_init__(self) -> None:
        n = int(self.off[-1]) if len(self.off) else 0
        if len(self.off) != len(self.step) + 1 or self.off[0] != 0 or (np.diff(self.off) < 0).any():
            raise FramesError("the CSR offsets do not match the frame steps")
        for name in ("uid", "row", "clade", "body"):
            if len(getattr(self, name)) != n:
                raise FramesError(f"{name} has {len(getattr(self, name))} rows, the offsets say {n}")
        if len(self.pos) != n or (self.drift is not None and len(self.drift) != n):
            raise FramesError("pos or drift row count differs from the offsets")
        if self.traits is None:
            self.traits = np.zeros((n, 0), np.float32)
        if self.traits.shape != (n, len(self.trait_names)):
            raise FramesError("traits shape does not match trait_names and the offsets")

    # geometry
    @property
    def n_frames(self) -> int:
        return len(self.step)

    @property
    def nd(self) -> int:
        return int(self.pos.shape[1])

    @property
    def size(self) -> float:
        return float(self.meta["size"])

    @property
    def height(self) -> float:
        return float(self.meta.get("height", 0.0))

    @property
    def has_drift(self) -> bool:
        return self.drift is not None

    def span(self, i: int) -> slice:
        return slice(int(self.off[i]), int(self.off[i + 1]))

    def positions(self, i: int) -> np.ndarray:
        """Float64 positions of frame ``i`` (dequantised)."""
        return dequantise(self.pos[self.span(i)], self.size, self.height)

    def trait(self, name: str) -> np.ndarray:
        """One trait column over all rows of all frames (KeyError if the file lacks it)."""
        assert self.traits is not None
        return self.traits[:, self.trait_names.index(name)]

    def frame(self, i: int) -> dict[str, np.ndarray]:
        """The arrays of frame ``i`` by name (views)."""
        s = self.span(i)
        out = {
            "uid": self.uid[s],
            "row": self.row[s],
            "pos": self.pos[s],
            "clade": self.clade[s],
            "body": self.body[s],
        }
        if self.drift is not None:
            out["drift"] = self.drift[s]
        assert self.traits is not None
        for j, nm in enumerate(self.trait_names):
            out[nm] = self.traits[s, j]
        return out

    # the fine set
    @property
    def n_fine_frames(self) -> int:
        return 0 if self.fine is None else len(self.fine["fine_step"])

    def fine_frame(self, i: int) -> dict[str, np.ndarray]:
        if self.fine is None:
            raise FramesError("no fine set in this file")
        o = self.fine["fine_off"]
        s = slice(int(o[i]), int(o[i + 1]))
        return {
            k.removeprefix("fine_"): v[s] for k, v in self.fine.items() if k not in ("fine_step", "fine_off")
        }


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_frames(path: Path, fr: Frames) -> Path:
    """Write ``fr`` atomically (tmp file then rename); returns ``path``."""
    meta = {**fr.meta, "frames_version": FRAMES_VERSION, "trait_names": list(fr.trait_names)}
    arrays: dict[str, np.ndarray] = {
        "step": fr.step.astype(np.int64),
        "off": fr.off.astype(np.int64),
        "uid": fr.uid.astype(np.int64),
        "row": fr.row.astype(np.int32),
        "pos": fr.pos.astype(np.uint16),
        "clade": fr.clade.astype(np.int32),
        "body": fr.body.astype(np.int32),
        "traits": np.asarray(fr.traits, np.float32),
        "meta": np.array(json.dumps(meta, sort_keys=True)),
    }
    if fr.drift is not None:
        arrays["drift"] = fr.drift.astype(np.float16)
    if fr.fine is not None:
        arrays.update({k: np.asarray(v) for k, v in fr.fine.items()})
    _atomic_savez(Path(path), arrays)
    return Path(path)


def read_frames(path: Path) -> Frames:
    """Read a recorder file. Raises ``FramesError`` on a different ``frames_version`` or missing arrays."""
    try:
        z = np.load(Path(path), allow_pickle=False)
    except (OSError, ValueError) as e:
        raise FramesError(f"{path}: not a frame file ({e})") from e
    with z:
        names = set(z.files)
        if "meta" not in names:
            raise FramesError(f"{path}: no meta entry")
        meta = json.loads(str(z["meta"]))
        if meta.get("frames_version") != FRAMES_VERSION:
            raise FramesError(
                f"{path}: frames_version {meta.get('frames_version')}, this code reads {FRAMES_VERSION}"
            )
        need = {"step", "off", "uid", "row", "pos", "clade", "body", "traits"}
        if need - names:
            raise FramesError(f"{path}: missing arrays {sorted(need - names)}")
        fine_keys = [k for k in ("fine_step", "fine_off", "fine_uid", "fine_group", "fine_pos", "fine_body")]
        fine = {k: z[k] for k in fine_keys} if all(k in names for k in fine_keys) else None
        return Frames(
            meta=meta,
            step=z["step"],
            off=z["off"],
            uid=z["uid"],
            row=z["row"],
            pos=z["pos"],
            clade=z["clade"],
            body=z["body"],
            drift=z["drift"] if "drift" in names else None,
            trait_names=list(meta.get("trait_names", [])),
            traits=z["traits"],
            fine=fine,
        )


# --- the play stream's FRMA records ---


def _components(n: int, pairs: np.ndarray) -> np.ndarray:
    """Min-index component label of ``n`` nodes under the undirected ``pairs`` (-1 for isolated nodes)."""
    lab = np.arange(n, dtype=np.int64)
    if len(pairs):
        a, b = pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64)
        while True:
            m = np.minimum(lab[a], lab[b])
            new = lab.copy()
            np.minimum.at(new, a, m)
            np.minimum.at(new, b, m)
            new = new[new]  # pointer jumping
            if (new == lab).all():
                break
            lab = new
    bonded = np.zeros(n, bool)
    if len(pairs):
        bonded[pairs.ravel()] = True
    return np.where(bonded, lab, -1).astype(np.int32)


def read_frma(path: Path, *, every: int = 1, max_frames: int | None = None) -> Frames:
    """The ``FRMA`` records of a play stream file as ``Frames`` (every ``every``-th record, at most
    ``max_frames``).

    ``clade`` is -1, there are no traits and no drift (the stream carries colour, not genome); ``body`` is
    the bond component of the listed rows; ``row`` is the listed index. Positions are re-quantised (float32
    in the stream, so the error is the quantiser's). Raises ``FramesError`` on a bad header or a truncated
    record.
    """
    buf = Path(path).read_bytes()
    if buf[:8] != _STREAM_MAGIC:
        raise FramesError(f"{path}: not a play stream (magic {buf[:8]!r})")
    (hl,) = struct.unpack_from("<I", buf, 8)
    header = json.loads(buf[12 : 12 + hl].decode("ascii"))
    nd, size, height = int(header["dims"]), float(header["L"]), float(header.get("height", 0.0))
    p, k = 12 + hl, 0
    steps, ns, uids, rows, poss, bodies = [], [], [], [], [], []
    while p + 8 <= len(buf):
        tag, ln = buf[p : p + 4], struct.unpack_from("<I", buf, p + 4)[0]
        if p + 8 + ln > len(buf):
            raise FramesError(f"{path}: truncated {tag!r} record at byte {p}")
        if tag == b"FRMA":
            if k % every == 0:
                q = p + 8
                step, _seq, _ver, _flags = _HEAD.unpack_from(buf, q)
                n, nb, _ci, _span, ncap = _FRMA_PREFIX.unpack_from(buf, q + _HEAD.size)
                o = q + _HEAD.size + _FRMA_PREFIX.size + 4 * (-(-ncap // 32))
                pos = np.frombuffer(buf, "<f4", 3 * n, o).reshape(n, 3)
                o += 12 * n + 4 * n + 4 * n  # pos, orient (2 x f2), rgb
                uid = np.frombuffer(buf, "<i8", n, o)
                o += 8 * n
                bonds = np.frombuffer(buf, "<u4", 2 * nb, o).reshape(nb, 2)
                steps.append(step)
                ns.append(n)
                uids.append(uid)
                rows.append(np.arange(n, dtype=np.int32))
                poss.append(quantise(pos[:, :nd].astype(np.float64), size, height))
                bodies.append(_components(n, bonds))
            k += 1
            if max_frames is not None and len(steps) >= max_frames:
                break
        p += 8 + ln
    off = np.concatenate([[0], np.cumsum(ns)]).astype(np.int64)
    cat = np.concatenate
    n_all = int(off[-1])
    meta = {"source": "frma", "size": size, "height": height, "nd": nd, "run": header.get("run"), "tau": None}
    return Frames(
        meta=meta,
        step=np.asarray(steps, np.int64),
        off=off,
        uid=cat(uids).astype(np.int64) if uids else np.zeros(0, np.int64),
        row=cat(rows) if rows else np.zeros(0, np.int32),
        pos=cat(poss) if poss else np.zeros((0, nd), np.uint16),
        clade=np.full(n_all, -1, np.int32),
        body=cat(bodies) if bodies else np.zeros(0, np.int32),
    )
