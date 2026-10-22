"""A reference decoder of the play-loop stream records, written from DESIGN §33.1, not from the writer.

It must not import the encoder module: every constant and offset is spelled out here on purpose, so the tests
check the encoders against the text of the specification, and the Swift decoder (``PlayRecords.swift``) is
checked against the same fixtures. Like ``stream_decode`` it is strict: it asserts the 4-byte alignment and
zero padding, every length, every count against ``N``, the slot bytes against their tables, and that the
records account for every byte of the input. Records of an unknown tag, or of a version it does not know, are
skipped by length and listed in ``Decoded.skipped``.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from stream_decode import decode_header

KNOWN = ("FRMA", "IDNT", "INTN", "TERR", "EVNT", "STRM", "MAP1", "ENVS", "CLCK", "FOCS")
VERSION = 1
RESERVED_SLOTS = (253, 254, 255)
MAX_TABLE = 16
MAX_EVENTS = 64
MAX_STORMS = 32
EVENT_KEYS = {"id", "step", "kind", "fam", "a", "b", "x", "y", "z", "imp", "text"}
EVENT_FAMILIES = ("kin", "house", "storm", "focus", "director", "system")
MAP_COARSE = 40


@dataclass
class Rec:
    tag: str
    offset: int  # of the tag in the input
    length: int  # the u32 length field
    step: int
    seq: int
    version: int
    flags: int
    f: dict[str, Any] = field(default_factory=dict)  # scalar fields and JSON
    a: dict[str, np.ndarray] = field(default_factory=dict)  # arrays
    sections: dict[str, bytes] = field(default_factory=dict)  # raw bytes of each section, padding included


@dataclass
class Decoded:
    header: dict[str, Any]
    records: list[Rec]
    skipped: list[tuple[str, int, str]]  # (tag, offset, "tag" | "version")


def _take(p: bytes, off: int, size: int, what: str) -> tuple[bytes, int]:
    assert off + size <= len(p), f"{what} runs past the record"
    return p[off : off + size], off + size


def _json(p: bytes, off: int, json_len: int, what: str) -> tuple[Any, bytes, int]:
    """``(object, section bytes with padding, next offset)``; the padding must be zero."""
    padded = (json_len + 3) // 4 * 4
    raw, nxt = _take(p, off, padded, what)
    assert raw[json_len:] == b"\x00" * (padded - json_len), f"{what} padding is not zero"
    return json.loads(raw[:json_len].decode("utf-8")), raw, nxt


def decode(data: bytes) -> Decoded:
    header, pos = decode_header(data)
    n = header["N"]
    records: list[Rec] = []
    skipped: list[tuple[str, int, str]] = []
    while pos < len(data):
        assert pos + 8 <= len(data), "truncated record prefix"
        tag = data[pos : pos + 4].decode("latin-1")
        (length,) = struct.unpack_from("<I", data, pos + 4)
        assert length % 4 == 0, f"{tag} length {length} is not a multiple of 4"
        assert pos + 8 + length <= len(data), f"truncated {tag} record"
        p = data[pos + 8 : pos + 8 + length]
        if tag not in KNOWN:
            skipped.append((tag, pos, "tag"))
        else:
            assert length >= 16, f"{tag} is shorter than the common head"
            step, seq, version, flags = struct.unpack_from("<QIHH", p, 0)
            if version != VERSION:
                skipped.append((tag, pos, "version"))
            else:
                rec = Rec(tag, pos, length, step, seq, version, flags)
                _DECODERS[tag](rec, p, n)
                records.append(rec)
        pos += 8 + length
    return Decoded(header, records, skipped)


def _frma(r: Rec, p: bytes, n_cap: int) -> None:
    n, n_bonds, color_index, span, cap = struct.unpack_from("<IIIII", p, 16)
    assert cap == n_cap, f"FRMA N {cap} differs from the header's {n_cap}"
    words = (n_cap + 31) // 32
    off = 36
    r.sections["bitmap"], off = _take(p, off, 4 * words, "bitmap")
    bits = np.unpackbits(np.frombuffer(r.sections["bitmap"], np.uint8), bitorder="little")
    assert not bits[n_cap:].any(), "bitmap bits at or past N are set"
    alive = bits[:n_cap].astype(bool)
    assert int(alive.sum()) == n, "FRMA n is not the popcount of the bitmap"
    for name, size in (
        ("pos", 12 * n),
        ("orient", 4 * n),
        ("rgb", 4 * n),
        ("uid", 8 * n),
        ("bonds", 8 * n_bonds),
    ):
        r.sections[name], off = _take(p, off, size, name)
    assert off == len(p), f"{len(p) - off} unexplained bytes in an FRMA"
    assert len(p) == 36 + 4 * words + 28 * n + 8 * n_bonds
    bonds = np.frombuffer(r.sections["bonds"], "<u4").reshape(n_bonds, 2).copy()
    assert (bonds[:, 0] < bonds[:, 1]).all() and (bonds < n).all(), "a bond is not i < j < n"
    assert r.flags & ~1 == 0, "FRMA flags bits other than 0 are set"
    r.f = {"n": n, "n_bonds": n_bonds, "color_index": color_index, "span": span, "N": cap}
    r.a = {
        "alive": alive,
        "rows": np.flatnonzero(alive),
        "pos": np.frombuffer(r.sections["pos"], "<f4").reshape(n, 3).copy(),
        "orient": np.frombuffer(r.sections["orient"], "<f2").reshape(n, 2).copy(),
        "rgb": np.frombuffer(r.sections["rgb"], "<u4").copy(),
        "uid": np.frombuffer(r.sections["uid"], "<i8").copy(),
        "bonds": bonds,
    }


def _slots_ok(values: np.ndarray, size: int, what: str) -> None:
    ok = (values < size) | np.isin(values, RESERVED_SLOTS)
    assert ok.all(), f"{what} slot {int(values[~ok][0])} is outside its table of {size}"


def _entry_slots_ok(values: np.ndarray, entries: list[dict[str, Any]], what: str) -> None:
    """A row's slot byte is the ``slot`` field of an entry of its table or 253 to 255; the entries' slots are
    distinct and below ``MAX_TABLE`` (a sticky slot leaves gaps, so the position is not the slot)."""
    slots = [e["slot"] for e in entries]
    assert all(isinstance(s, int) and 0 <= s < MAX_TABLE for s in slots) and len(set(slots)) == len(slots)
    ok = np.isin(values, slots) | np.isin(values, RESERVED_SLOTS)
    assert ok.all(), f"{what} slot {int(values[~ok][0])} is not the slot of a listed entry"


def _idnt(r: Rec, p: bytes, n_cap: int) -> None:
    n_rows, snapshot_step, json_len, reserved = struct.unpack_from("<IIII", p, 16)
    assert reserved == 0
    obj, r.sections["json"], off = _json(p, 32, json_len, "IDNT json")
    r.sections["rows"], off = _take(p, off, 16 * n_rows, "IDNT rows")
    assert off == len(p) and n_rows <= n_cap
    raw = np.frombuffer(r.sections["rows"], np.uint8).reshape(n_rows, 16)
    for key in ("houses", "kin", "colonies"):
        assert len(obj[key]) <= MAX_TABLE
    house, kin, colony = raw[:, 12], raw[:, 13], raw[:, 14]
    _entry_slots_ok(house, obj["houses"], "house")
    _entry_slots_ok(kin, obj["kin"], "kin")
    _entry_slots_ok(colony, obj["colonies"], "colony")
    r.f = {"n_rows": n_rows, "snapshot_step": snapshot_step, "json": obj}
    r.a = {
        "uid": np.frombuffer(raw[:, :8].tobytes(), "<i8").copy(),
        "clade": np.frombuffer(raw[:, 8:12].tobytes(), "<i4").copy(),
        "house": house.copy(),
        "kin": kin.copy(),
        "colony": colony.copy(),
        "flags": raw[:, 15].copy(),
    }


def _intn(r: Rec, p: bytes, n_cap: int) -> None:
    n_rows, labels_step, intent_version, reserved = struct.unpack_from("<IIII", p, 16)
    assert reserved == 0
    r.sections["rows"], off = _take(p, 32, 12 * n_rows, "INTN rows")
    assert off == len(p) and n_rows <= n_cap
    raw = np.frombuffer(r.sections["rows"], np.uint8).reshape(n_rows, 12)
    r.f = {"n_rows": n_rows, "labels_step": labels_step, "intent_version": intent_version}
    r.a = {
        "uid": np.frombuffer(raw[:, :8].tobytes(), "<i8").copy(),
        "arch_primary": raw[:, 8].copy(),
        "arch_secondary": raw[:, 9].copy(),
        "intent": raw[:, 10].copy(),
        "flags": raw[:, 11].copy(),
    }


def _terr(r: Rec, p: bytes, n_cap: int) -> None:
    nx, ny, snapshot_step, json_len, reserved = struct.unpack_from("<HHIII", p, 16)
    assert reserved == 0
    obj, r.sections["json"], off = _json(p, 32, json_len, "TERR json")
    cells = nx * ny
    padded = (2 * cells + 3) // 4 * 4
    planes, off = _take(p, off, padded, "TERR planes")
    assert off == len(p) and planes[2 * cells :] == b"\x00" * (padded - 2 * cells)
    r.sections["owner"], r.sections["purity"] = planes[:cells], planes[cells : 2 * cells]
    owner = np.frombuffer(r.sections["owner"], np.uint8).reshape(ny, nx).copy()
    assert len(obj["ids"]) <= MAX_TABLE
    _slots_ok(owner.reshape(-1), len(obj["ids"]), "territory owner")
    r.f = {"nx": nx, "ny": ny, "snapshot_step": snapshot_step, "json": obj}
    r.a = {"owner": owner, "purity": np.frombuffer(r.sections["purity"], np.uint8).reshape(ny, nx).copy()}


def _strm(r: Rec, p: bytes, n_cap: int) -> None:
    json_len, tick, n_storms, reserved = struct.unpack_from("<IIII", p, 16)
    assert reserved == 0
    obj, r.sections["json"], off = _json(p, 32, json_len, "STRM json")
    assert off == len(p) and n_storms == len(obj["storms"]) <= MAX_STORMS
    r.f = {"tick": tick, "n_storms": n_storms, "json": obj}


def _evnt(r: Rec, p: bytes, n_cap: int) -> None:
    count, json_len = struct.unpack_from("<II", p, 16)
    obj, r.sections["json"], off = _json(p, 24, json_len, "EVNT json")
    assert off == len(p) and isinstance(obj, list) and count == len(obj) <= MAX_EVENTS
    for e in obj:
        assert set(e) == EVENT_KEYS and e["fam"] in EVENT_FAMILIES and e["imp"] in (0, 1, 2)
        assert len(e["text"]) <= 120
    assert r.flags & ~1 == 0
    r.f = {"count": count, "backlog": bool(r.flags & 1), "json": obj}


def _map1(r: Rec, p: bytes, n_cap: int) -> None:
    n, nw, f0, p_ref, dc_ref, u_ref, zeta_ref = struct.unpack_from("<HHfffff", p, 16)
    assert p[40:48] == b"\x00" * 8
    assert r.flags & 3 == 3 and r.flags & ~3 == 0, "version 1 records carry T_s and winds"
    assert nw == MAP_COARSE and n % nw == 0
    off = 48
    for name, size in (("P", n * n), ("Dc", n * n), ("zeta", n * n), ("Ts", n * n), ("winds", 4 * nw * nw)):
        r.sections[name], off = _take(p, off, size, name)
    tail = (off + 3) // 4 * 4
    assert len(p) == tail and p[off:] == b"\x00" * (tail - off), "unexplained bytes in a MAP1"
    r.f = {"n": n, "nw": nw, "f0": f0, "p_ref": p_ref, "dc_ref": dc_ref, "u_ref": u_ref, "zeta_ref": zeta_ref}
    r.a = {
        "P": np.frombuffer(r.sections["P"], np.uint8).reshape(n, n).copy(),
        "Dc": np.frombuffer(r.sections["Dc"], np.uint8).reshape(n, n).copy(),
        "zeta": np.frombuffer(r.sections["zeta"], np.int8).reshape(n, n).copy(),
        "Ts": np.frombuffer(r.sections["Ts"], np.uint8).reshape(n, n).copy(),
        "winds": np.frombuffer(r.sections["winds"], np.int8).reshape(4, nw, nw).copy(),
    }


def _envs(r: Rec, p: bytes, n_cap: int) -> None:
    (json_len,) = struct.unpack_from("<I", p, 16)
    obj, r.sections["json"], off = _json(p, 20, json_len, "ENVS json")
    assert off == len(p)
    r.f = {"json": obj}


def _clck(r: Rec, p: bytes, n_cap: int) -> None:
    assert len(p) == 16 + 44
    (
        rate,
        achieved,
        step_ms,
        frame_ms,
        late,
        stall,
        reason,
        saving,
        flags,
        fe,
        chapter,
        in_chapter,
        chapter_len,
    ) = struct.unpack_from("<ffffIIBBHIIII", p, 16)
    assert reason <= 5 and saving <= 1
    r.sections["body"] = p[16:]
    r.f = {
        "rate": rate,
        "achieved": achieved,
        "step_ms": step_ms,
        "frame_ms": frame_ms,
        "late_steps": late,
        "save_stall_steps": stall,
        "reason": reason,
        "saving": saving,
        "flags": flags,
        "fe": fe,
        "chapter": chapter,
        "in_chapter": in_chapter,
        "chapter_len": chapter_len,
    }


def _focs(r: Rec, p: bytes, n_cap: int) -> None:
    assert len(p) == 16 + 68
    v = struct.unpack_from("<qIIffffffiIIqBBBBBBBB", p, 16)
    r.sections["body"] = p[16:]
    names = [
        *(
            "uid",
            "row",
            "flags",
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "energy",
            "age",
            "generation",
            "parent_uid",
        ),
        *("n_bonds", "house", "kin", "colony", "arch_primary", "arch_secondary", "intent", "label_flags"),
    ]
    r.f = dict(zip(names, v, strict=True))
    for k in ("house", "kin", "colony"):
        assert r.f[k] < MAX_TABLE or r.f[k] in RESERVED_SLOTS


_DECODERS = {
    "FRMA": _frma,
    "IDNT": _idnt,
    "INTN": _intn,
    "TERR": _terr,
    "STRM": _strm,
    "EVNT": _evnt,
    "MAP1": _map1,
    "ENVS": _envs,
    "CLCK": _clck,
    "FOCS": _focs,
}
