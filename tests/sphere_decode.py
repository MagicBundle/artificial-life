"""A reference decoder of the protocol 2 (sphere world) play stream, written from the format text
"stream protocol v2, the sphere world" (DESIGN §46.7, stream part), not from the writer.

It does not import ``alife``: every constant and offset of the v2 differences is spelled out here. The records
that protocol 2 leaves unchanged (INTN, CLCK, FSIG, RPRT, PLAN, and the common part of IDNT, FOCS, STRM,
EVNT, ALRT) are decoded by the protocol 1 reference decoder ``play_decode``; the header, ``FRMA``, ``MAPH``
(replaces ``MAP1``), ``STRC``/``FSOL``, ``TERR`` and ``ENVS`` are decoded here, and the places carried by the
other records get the v2 checks. Like ``play_decode`` it is strict: 4-byte alignment, zero padding, every
length, every count against ``N``, slot bytes against their tables, and that the records account for every
byte of the input. A protocol 1 stream is refused with ``ProtocolError``; the box decoders are untouched.

Orientation (section 3): ``heading_from_orientation`` and ``bearing_of`` implement ``(yaw, pitch)`` against
the local frame ``(east, north, up)`` of a planet-frame position.
"""

from __future__ import annotations

import itertools
import json
import math
import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import play_decode as _pd

MAGIC = b"ALSTRM\x00\x01"
PROTOCOL = 2
VERSION = 1
POLE_EPS = 1e-6  # |z x up| below this: a pole, east = (1, 0, 0)
SHELL_TOL = 1e-3  # |p| tolerance against the shell, "up to float32 rounding"
EDGE_RTOL = 1e-6  # shell_edges against planet_radius and planet_radius + height
VOLUME_RTOL = 1e-5  # equal-volume layers
MAX_NPIX_U16 = 65535
MAPH_FLAGS = 3

KNOWN = (*(t for t in _pd.KNOWN if t != "MAP1"), "MAPH")
_REUSED = ("INTN", "CLCK", "FSIG", "RPRT", "PLAN")  # unchanged records, decoded by play_decode
_POSITIONAL_LEGACY = {"lon", "lat", "longitude", "latitude", "cx", "cy"}


class ProtocolError(ValueError):
    """The stream is not protocol 2 (a protocol 1 stream is refused)."""


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


# ---------------------------------------------------------------------------------------------------------
# Orientation (section 3)


def _unit_basis(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(up, east, north)`` for planet-frame positions ``pos`` of shape (..., 3).

    ``up = p/|p|``; ``east = normalise(z x up)``, and ``(1, 0, 0)`` when ``|z x up| < 1e-6`` (a pole);
    ``north = up x east``."""
    pos = np.asarray(pos, np.float64)
    norm = np.linalg.norm(pos, axis=-1, keepdims=True)
    assert (norm > 0).all(), "a position at the planet's centre has no up"
    up = pos / norm
    # z x up = (-uy, ux, 0)
    ze = np.stack([-up[..., 1], up[..., 0], np.zeros_like(up[..., 0])], axis=-1)
    zn = np.linalg.norm(ze, axis=-1, keepdims=True)
    pole = zn < POLE_EPS
    east = np.where(pole, np.array([1.0, 0.0, 0.0]), ze / np.where(pole, 1.0, zn))
    north = np.cross(up, east)
    return up, east, north


def heading_from_orientation(pos: np.ndarray, yaw: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """The heading (unit vector) of agents at ``pos`` (n, 3) with bearing ``yaw`` and elevation ``pitch``:
    ``t = cos(yaw) north + sin(yaw) east`` then ``forward = cos(pitch) t + sin(pitch) up``."""
    up, east, north = _unit_basis(pos)
    yaw = np.asarray(yaw, np.float64)[..., None]
    pitch = np.asarray(pitch, np.float64)[..., None]
    t = np.cos(yaw) * north + np.sin(yaw) * east
    return np.cos(pitch) * t + np.sin(pitch) * up


def bearing_of(pos: np.ndarray, heading: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The inverse of ``heading_from_orientation``: ``(yaw, pitch)`` of ``heading`` (n, 3; any positive scale)
    at ``pos`` (n, 3). ``yaw = atan2(t . east, t . north)`` of the tangent part ``t`` of the heading, mapped
    to [-pi, pi); ``pitch = atan2(h . up, |t|)`` in [-pi/2, pi/2]. A vertical heading (``|t|`` below 1e-9 of
    ``|h|``) has no bearing and gets yaw 0."""
    up, east, north = _unit_basis(pos)
    h = np.asarray(heading, np.float64)
    radial = np.sum(h * up, axis=-1)
    t = h - radial[..., None] * up
    tn = np.linalg.norm(t, axis=-1)
    yaw = np.arctan2(np.sum(t * east, axis=-1), np.sum(t * north, axis=-1))
    yaw = np.where(yaw >= math.pi, yaw - 2 * math.pi, yaw)  # [-pi, pi)
    vertical = tn < 1e-9 * np.linalg.norm(h, axis=-1)
    yaw = np.where(vertical, 0.0, yaw)
    pitch = np.arctan2(radial, tn)
    return yaw, pitch


# ---------------------------------------------------------------------------------------------------------
# Header


def decode_header(data: bytes) -> tuple[dict[str, Any], int]:
    """``(header object, offset of the first record)``; refuses anything but protocol 2."""
    assert data[:8] == MAGIC, data[:8]
    assert len(data) >= 12, "truncated header prefix"
    (json_len,) = struct.unpack_from("<I", data, 8)
    assert json_len % 4 == 0 and len(data) >= 12 + json_len, "header length"
    text = data[12 : 12 + json_len]
    assert len(text) - len(text.rstrip(b" ")) < 4, "the JSON is padded with fewer than 4 spaces"

    def _no_constant(c: str) -> Any:
        raise AssertionError(f"header JSON holds {c}")

    h = json.loads(text.decode("ascii"), parse_constant=_no_constant)
    assert isinstance(h, dict), "the header is a JSON object"
    protocol = h.get("protocol")
    if protocol != PROTOCOL:
        raise ProtocolError(
            f"protocol {protocol!r} stream: this decoder reads protocol {PROTOCOL} (sphere) only"
            + (" (use play_decode for a box stream)" if protocol == 1 else "")
        )
    _check_header(h)
    return h, 12 + json_len


def _int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


REQUIRED_KEYS = [
    *("dims", "FG", "FZ", "N", "B", "height", "radius", "geometry", "planet_radius", "shell_edges"),
    *("nside_field", "nside_weather", "nside_territory", "ring_order", "f0", "weather_depth"),
]


def _check_header(h: dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in h]
    assert not missing, f"header keys missing: {missing}"
    assert "L" not in h, "a sphere header has no image length L"
    assert h["dims"] == 3 and _int(h["dims"]), "dims must be 3"
    assert _int(h["FG"]) and h["FG"] == 0, "FG must be 0 (no lattice side)"
    assert _int(h["FZ"]) and h["FZ"] >= 1, "FZ is the number of radial layers"
    assert _int(h["N"]) and h["N"] > 0 and _int(h["B"]) and h["B"] >= 0, "N and B are ints"
    assert _num(h["height"]) and h["height"] > 0, "height is the shell thickness"
    assert _num(h["radius"]) and h["radius"] > 0, "radius is the body radius of an agent"
    assert h["geometry"] == "sphere", 'geometry must be "sphere"'
    rp = h["planet_radius"]
    assert _num(rp) and rp > 0, "planet_radius"
    edges = h["shell_edges"]
    assert isinstance(edges, list) and len(edges) == h["FZ"] + 1, "len(shell_edges) must be FZ + 1"
    assert all(_num(e) for e in edges), "shell_edges are numbers"
    assert all(b > a for a, b in itertools.pairwise(edges)), "shell_edges are not ascending"
    assert math.isclose(edges[0], rp, rel_tol=EDGE_RTOL), "first shell edge is not planet_radius"
    assert math.isclose(edges[-1], rp + h["height"], rel_tol=EDGE_RTOL), "last shell edge is not Rp + H"
    vols = np.diff(np.asarray(edges, np.float64) ** 3)
    assert np.allclose(vols, vols.mean(), rtol=VOLUME_RTOL), "layers are not of equal volume"
    assert _int(h["nside_field"]) and h["nside_field"] >= 2, "nside_field is an int >= 2"
    assert _int(h["nside_weather"]) and h["nside_weather"] >= 0, "nside_weather is an int >= 0"
    assert _int(h["nside_territory"]) and h["nside_territory"] >= 2, "nside_territory is an int >= 2"
    assert h["ring_order"] == "ring", 'ring_order must be "ring"'
    assert _num(h["f0"]), "f0 is a number"
    assert _num(h["weather_depth"]) and h["weather_depth"] >= 0
    assert h["nside_weather"] > 0 or h["weather_depth"] == 0, "weather_depth is 0 without weather"
    _check_play(h)


def _check_play(h: dict[str, Any]) -> None:
    play = h.get("play")
    if play is None:  # an `alife stream` header
        return
    npix = 12 * h["nside_field"] ** 2
    records = play.get("records")
    if records is not None:
        assert "MAP1" not in records, "a v2 header never lists MAP1"
    wx = play.get("weather")
    if wx is not None:
        assert isinstance(wx.get("enabled"), bool) and wx["enabled"] == (h["nside_weather"] > 0), (
            "play.weather.enabled vs nside_weather"
        )  # a weather-off world still lists the block, enabled false, nside 0
        assert wx["nside"] == h["nside_weather"] and wx["coarse"] == h["nside_field"], "play.weather nsides"
        assert {"enabled", "nside", "coarse", "f0", "name_zeta", "refs"} <= set(wx)
    for name in ("structures", "niche"):
        block = play.get(name)
        if block is not None and "cells" in block:
            assert block["cells"] == [h["FZ"], 1, npix], f"play.{name}.cells is [FZ, 1, Npix]"
    labels = play.get("labels")
    if labels is not None:
        assert labels["intent_version"] == 4, "play.labels.intent_version is 4 on a sphere"
    versions = play.get("versions") or {}
    if records is not None and "MAPH" in records:
        assert versions.get("MAPH") == 1, "play.versions has MAPH: 1 when MAPH is listed"


# ---------------------------------------------------------------------------------------------------------
# Stream


def decode(data: bytes) -> Decoded:
    header, pos = decode_header(data)
    records: list[Rec] = []
    skipped: list[tuple[str, int, str]] = []
    while pos < len(data):
        assert pos + 8 <= len(data), "truncated record prefix"
        tag = data[pos : pos + 4].decode("latin-1")
        (length,) = struct.unpack_from("<I", data, pos + 4)
        assert length % 4 == 0, f"{tag} length {length} is not a multiple of 4"
        assert pos + 8 + length <= len(data), f"truncated {tag} record"
        p = data[pos + 8 : pos + 8 + length]
        assert tag != "MAP1", "a protocol 2 stream never carries MAP1"
        if tag not in KNOWN:
            skipped.append((tag, pos, "tag"))
        else:
            assert length >= 16, f"{tag} is shorter than the common head"
            step, seq, version, flags = struct.unpack_from("<QIHH", p, 0)
            if version != VERSION:
                skipped.append((tag, pos, "version"))
            else:
                rec = Rec(tag, pos, length, step, seq, version, flags)
                _DECODERS[tag](rec, p, header)
                records.append(rec)
        pos += 8 + length
    _pd._check_labels(header, records)
    _pd._check_alerts(header, records)
    return Decoded(header, records, skipped)


def _shell_bounds(h: dict[str, Any]) -> tuple[float, float]:
    return h["planet_radius"], h["planet_radius"] + h["height"]


def _frma(r: Rec, p: bytes, h: dict[str, Any]) -> None:
    n_cap = h["N"]
    assert len(p) >= 36, "FRMA is shorter than its fixed part"
    n, n_bonds, color_index, span, cap = struct.unpack_from("<IIIII", p, 16)
    assert cap == n_cap, f"FRMA N {cap} differs from the header's {n_cap}"
    words = (n_cap + 31) // 32
    off = 36
    r.sections["bitmap"], off = _pd._take(p, off, 4 * words, "bitmap")
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
        r.sections[name], off = _pd._take(p, off, size, name)
    flow = None
    if r.flags & 2:  # bonds feature: one int8 per drawn bond follows the pairs, zero padded to 4 bytes
        raw, off = _pd._take(p, off, 4 * ((n_bonds + 3) // 4), "bond flow")
        assert raw[n_bonds:] == b"\x00" * (len(raw) - n_bonds), "bond flow padding is not zero"
        flow = np.frombuffer(raw[:n_bonds], np.int8).copy()
        r.sections["flow"] = raw
    assert off == len(p), f"{len(p) - off} unexplained bytes in an FRMA"
    assert len(p) == 36 + 4 * words + 28 * n + 8 * n_bonds + (
        4 * ((n_bonds + 3) // 4) if flow is not None else 0
    )
    bonds = np.frombuffer(r.sections["bonds"], "<u4").reshape(n_bonds, 2).copy()
    assert (bonds[:, 0] < bonds[:, 1]).all() and (bonds < n).all(), "a bond is not i < j < n"
    assert r.flags & ~3 == 0, "FRMA flags bits other than 0 and 1 are set"
    pos = np.frombuffer(r.sections["pos"], "<f4").reshape(n, 3).copy()
    orient = np.frombuffer(r.sections["orient"], "<f2").reshape(n, 2).copy()
    p64, o64 = pos.astype(np.float64), orient.astype(np.float64)
    assert np.isfinite(p64).all() and np.isfinite(o64).all(), "non-finite position or orientation"
    rp, rtop = _shell_bounds(h)
    radius = np.linalg.norm(p64, axis=-1)
    bad = (radius < rp - SHELL_TOL) | (radius > rtop + SHELL_TOL)
    assert not bad.any(), (
        f"FRMA position {int(np.flatnonzero(bad)[0])} has |p| = {radius[bad][0]} outside the shell"
    )
    assert ((o64[:, 0] >= -math.pi) & (o64[:, 0] < math.pi)).all(), "yaw outside [-pi, pi)"
    assert (np.abs(o64[:, 1]) <= math.pi / 2 + 2**-9).all(), "pitch outside [-pi/2, pi/2]"
    r.f = {"n": n, "n_bonds": n_bonds, "color_index": color_index, "span": span, "N": cap}
    r.a = {
        "alive": alive,
        "rows": np.flatnonzero(alive),
        "pos": pos,
        "orient": orient,
        "height": radius - rp,
        "heading": heading_from_orientation(p64, o64[:, 0], o64[:, 1]),
        "rgb": np.frombuffer(r.sections["rgb"], "<u4").copy(),
        "uid": np.frombuffer(r.sections["uid"], "<i8").copy(),
        "bonds": bonds,
        **({"flow": flow} if flow is not None else {}),
    }


def _cells(r: Rec, p: bytes, h: dict[str, Any]) -> None:
    """STRC and FSOL: nx ny nz flags count, ascending u32 indices, u8 levels, padding; on a sphere
    ``(nx, ny, nz) = (Npix, 1, FZ)`` and the flat index is ``layer * Npix + pixel``. ``a["levels"]`` is the
    dense (FZ, Npix) uint8 array."""
    assert len(p) >= 28, "cell record shorter than its fixed part"
    nx, ny, nz, flags, count = struct.unpack_from("<HHHHI", p, 16)
    npix = 12 * h["nside_field"] ** 2
    assert npix <= MAX_NPIX_U16, "Npix does not fit a u16: such a world does not send cell records"
    assert (nx, ny, nz) == (npix, 1, h["FZ"]), (
        f"cell record dims {(nx, ny, nz)} vs header {(npix, 1, h['FZ'])}"
    )
    assert flags == r.flags and flags & ~1 == 0, "cell record body flags differ from the head flags"
    assert count <= 65536 and (flags & 1 == 0 or count == 65536)
    off = 28
    raw_i, off = _pd._take(p, off, 4 * count, "indices")
    raw_l, off = _pd._take(p, off, count, "levels")
    pad = (-count) % 4
    pd_, off = _pd._take(p, off, pad, "padding")
    assert pd_ == b"\x00" * pad and off == len(p), "cell record padding or length"
    idx = np.frombuffer(raw_i, "<u4").copy()
    lev = np.frombuffer(raw_l, np.uint8).copy()
    assert (np.diff(idx.astype(np.int64)) > 0).all(), "cell indices are not strictly ascending"
    assert (idx < nx * ny * nz).all(), "a cell index is at or past FZ * Npix"
    assert (lev >= 1).all(), "a listed cell has level 0"
    levels = np.zeros(nz * npix, np.uint8)
    levels[idx] = lev
    r.f = {"nx": nx, "ny": ny, "nz": nz, "flags": flags, "count": count}
    r.a = {"idx": idx, "level": lev, "levels": levels.reshape(nz, npix)}


def _terr(r: Rec, p: bytes, h: dict[str, Any]) -> None:
    assert len(p) >= 32
    nx, ny, snapshot_step, json_len, reserved = struct.unpack_from("<HHIII", p, 16)
    assert reserved == 0
    assert (nx, ny) == (12 * h["nside_territory"] ** 2, 1), f"TERR dims {(nx, ny)} vs the header's"
    obj, r.sections["json"], off = _pd._json(p, 32, json_len, "TERR json")
    cells = nx * ny
    padded = (2 * cells + 3) // 4 * 4
    planes, off = _pd._take(p, off, padded, "TERR planes")
    assert off == len(p) and planes[2 * cells :] == b"\x00" * (padded - 2 * cells)
    r.sections["owner"], r.sections["purity"] = planes[:cells], planes[cells : 2 * cells]
    owner = np.frombuffer(r.sections["owner"], np.uint8).copy()
    assert len(obj["ids"]) <= _pd.MAX_TABLE
    _pd._slots_ok(owner, len(obj["ids"]), "territory owner")
    r.f = {"nx": nx, "ny": ny, "snapshot_step": snapshot_step, "json": obj}
    r.a = {"owner": owner, "purity": np.frombuffer(r.sections["purity"], np.uint8).copy()}


def _maph(r: Rec, p: bytes, h: dict[str, Any]) -> None:
    """MAPH (section 4): the weather planes on the field's HEALPix grid, dequantised to float64."""
    assert len(p) >= 48
    nside, k, f0, p_ref, dc_ref, u_ref, zeta_ref = struct.unpack_from("<HHfffff", p, 16)
    assert r.flags == MAPH_FLAGS, f"MAPH head flags {r.flags} != 3 (T_s and winds both present)"
    assert p[40:48] == b"\x00" * 8, "MAPH reserved bytes are not zero"
    assert nside == h["nside_field"], "MAPH nside differs from nside_field"
    assert h["nside_weather"] > 0 and h["nside_weather"] == nside * 2**k, "nside_weather != nside * 2**k"
    assert all(math.isfinite(v) and v >= 0 for v in (p_ref, dc_ref, u_ref, zeta_ref)), "MAPH reference scales"
    assert math.isfinite(f0)
    npix = 12 * nside * nside
    off = 48
    for name, size in (("P", npix), ("Dc", npix), ("zeta", npix), ("Ts", npix), ("winds", 4 * npix)):
        r.sections[name], off = _pd._take(p, off, size, name)
    tail = (off + 3) // 4 * 4
    assert len(p) == 16 + (32 + 8 * npix + 3) // 4 * 4, "MAPH length"
    assert len(p) == tail and p[off:] == b"\x00" * (tail - off), "unexplained bytes in a MAPH"
    qP = np.frombuffer(r.sections["P"], np.uint8).copy()
    qDc = np.frombuffer(r.sections["Dc"], np.uint8).copy()
    qz = np.frombuffer(r.sections["zeta"], np.int8).copy()
    qTs = np.frombuffer(r.sections["Ts"], np.uint8).copy()
    qw = np.frombuffer(r.sections["winds"], np.int8).reshape(4, npix).copy()
    r.f = {
        "nside": nside,
        "k": k,
        "npix": npix,
        "f0": f0,
        "p_ref": p_ref,
        "dc_ref": dc_ref,
        "u_ref": u_ref,
        "zeta_ref": zeta_ref,
    }
    r.a = {
        "q_P": qP,
        "q_Dc": qDc,
        "q_zeta": qz,
        "q_Ts": qTs,
        "q_winds": qw,
        "P": (qP / 255.0) ** 2 * float(p_ref),
        "Dc": qDc / 255.0 * float(dc_ref),
        "zeta": qz / 127.0 * float(zeta_ref),
        "Ts": qTs / 255.0,
        "winds": qw / 127.0 * float(u_ref),  # rows: u1, v1, u2, v2 (eastward, northward)
    }


def _envs(r: Rec, p: bytes, h: dict[str, Any]) -> None:
    (json_len,) = struct.unpack_from("<I", p, 16)
    obj, r.sections["json"], off = _pd._json(p, 20, json_len, "ENVS json")
    assert off == len(p) and json_len <= 8 * 1024
    inf = obj.get("influence")
    if inf is not None:
        assert set(inf) == {"have", "cap", "window", "free_at"} and inf["have"] <= inf["cap"]
        assert all(isinstance(v, int) for v in inf["free_at"])
    for patch in obj.get("patches", []):
        assert set(patch) == {"i", "x", "y", "z", "lon", "lat", "gain", "sigma"}, "ENVS patch keys"
        assert all(_num(patch[k]) for k in ("x", "y", "z", "lon", "lat", "gain", "sigma"))
        norm = math.sqrt(patch["x"] ** 2 + patch["y"] ** 2 + patch["z"] ** 2)
        assert abs(norm - 1) < 1e-3, "patch x, y, z is not a unit vector"
        assert -180 <= patch["lon"] <= 180 and -90 <= patch["lat"] <= 90, "patch lon, lat in degrees"
    for name, power in (obj.get("powers") or {}).items():
        assert set(power) == {"ready", "busy"}, name
    assert all(e["source"] in ("rule", "llm", "player") for e in obj.get("effects", []))
    if h["nside_weather"] == 0:
        assert obj.get("pools") == [], "ENVS.pools is an empty list on a sphere without weather"
    r.f = {"json": obj}


def _place(obj: dict[str, Any], h: dict[str, Any], what: str) -> None:
    """x, y, z are numbers or null; a Cartesian point cannot lie beyond the shell's outer radius."""
    vals = [obj[k] for k in ("x", "y", "z")]
    assert all(v is None or _num(v) for v in vals), f"{what}: x, y, z are numbers or null"
    if all(v is not None for v in vals):
        assert math.sqrt(sum(v * v for v in vals)) <= _shell_bounds(h)[1] + 0.05, f"{what}: beyond the shell"


def _evnt(r: Rec, p: bytes, h: dict[str, Any]) -> None:
    _pd._evnt(r, p, h["N"])
    for e in r.f["json"]:
        _place(e, h, f"EVNT@{r.offset}")


def _alrt(r: Rec, p: bytes, h: dict[str, Any]) -> None:
    _pd._alrt(r, p, h["N"])
    for a in r.f["json"]:
        _place(a, h, f"ALRT@{r.offset}")


def _strm(r: Rec, p: bytes, h: dict[str, Any]) -> None:
    _pd._strm(r, p, h["N"])
    rp = h["planet_radius"]
    for s in r.f["json"]["storms"]:
        assert all(_num(s.get(k)) for k in ("x", "y", "z", "r")) and s["r"] > 0, "storm x, y, z, r"
        assert not (set(s) & _POSITIONAL_LEGACY), "a storm has no positional keys beyond x, y, z, r"
        radius = math.sqrt(s["x"] ** 2 + s["y"] ** 2 + s["z"] ** 2)
        assert abs(radius - rp) <= 1e-2 + 1e-3 * rp, "a storm centre lies at the floor radius"


def _idnt(r: Rec, p: bytes, h: dict[str, Any]) -> None:
    _pd._idnt(r, p, h["N"])
    obj = r.f["json"]
    rp, rtop = _shell_bounds(h)
    for table in ("kin", "colonies"):
        for e in obj[table]:
            assert all(_num(e.get(k)) for k in ("x", "y", "z")), f"IDNT {table}: x, y, z centroid"
            assert not (set(e) & _POSITIONAL_LEGACY), f"IDNT {table}: nothing in it is a longitude"
            c = math.sqrt(e["x"] ** 2 + e["y"] ** 2 + e["z"] ** 2)
            assert c <= rtop + 0.05, f"IDNT {table}: centroid beyond the shell"
            if table == "kin":
                assert _num(e.get("height")), "IDNT kin: height (mean height)"
                assert -0.01 <= e["height"] <= rtop - rp + 0.01, "IDNT kin: mean height outside the shell"


def _focs(r: Rec, p: bytes, h: dict[str, Any]) -> None:
    _pd._focs(r, p, h["N"])
    pos = np.array([r.f["x"], r.f["y"], r.f["z"]], np.float64)
    vel = np.array([r.f["vx"], r.f["vy"], r.f["vz"]], np.float64)
    rp, rtop = _shell_bounds(h)
    radius = float(np.linalg.norm(pos))
    assert rp - SHELL_TOL <= radius <= rtop + SHELL_TOL, f"FOCS |p| = {radius} outside the shell"
    up = pos / radius
    r.f["v_radial"] = float(vel @ up)
    r.f["height"] = radius - rp


def _reused(tag: str):
    def run(r: Rec, p: bytes, h: dict[str, Any]) -> None:
        _pd._DECODERS[tag](r, p, h["N"])

    return run


_DECODERS = {
    "FRMA": _frma,
    "IDNT": _idnt,
    "TERR": _terr,
    "STRM": _strm,
    "EVNT": _evnt,
    "MAPH": _maph,
    "ENVS": _envs,
    "FOCS": _focs,
    "ALRT": _alrt,
    "STRC": _cells,
    "FSOL": _cells,
    **{t: _reused(t) for t in _REUSED},
}
assert set(_DECODERS) == set(KNOWN)
