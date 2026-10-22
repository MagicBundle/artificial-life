"""The play-loop stream records, ALSTRM v1 additions (DESIGN §33.1, docs/GAMES.md, G3a).

Host module, numpy only (no ``mlx`` import): every encoder is a pure function of arrays and dicts, so the
fixtures of ``tests/play_fixtures.py`` are device independent. The records are written by ``alife play`` after
the header of ``alife stream`` (§27.11) and decoded by ``AlifeCore/PlayRecords.swift``;
``tests/play_decode.py`` is an independent decoder written from §33.1, not from this file.

Conventions (§33.1): little endian; a record is a 4-byte tag, a ``u32`` payload length (a multiple of 4) and
the payload, which starts with the common head ``u64 step, u32 seq, u16 version, u16 flags`` (16 bytes);
every section starts at a multiple of 4 bytes; JSON is UTF-8 from ``json_bytes`` with ``json_len`` the
unpadded length followed by 0 to 3 zero bytes; unknown tags are skipped by length, unknown versions are
skipped and counted.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

import numpy as np

MAGIC: Final = b"ALSTRM\x00\x01"
PROTOCOL: Final = 1
VERSION: Final = 1  # the layout version of every tag below
TAGS: Final = ("FRMA", "IDNT", "INTN", "TERR", "EVNT", "STRM", "MAP1", "ENVS", "CLCK", "FOCS")
VERSIONS: Final = dict.fromkeys(TAGS, VERSION)
HEAD: Final = struct.Struct("<QIHH")  # step, seq, version, flags
HEAD_SIZE: Final = 16
FRMA_PREFIX: Final = struct.Struct("<IIIII")  # n, n_bonds, color_index, span, N
CLCK_BODY: Final = struct.Struct("<ffffIIBBHIIII")  # 44 bytes
FOCS_BODY: Final = struct.Struct("<qIIffffffiIIqBBBBBBBB")  # 68 bytes
MAP1_PREFIX: Final = struct.Struct("<HHfffff8x")  # n, nw, f0, p_ref, dc_ref, u_ref, zeta_ref, 8 reserved
assert HEAD.size == HEAD_SIZE and CLCK_BODY.size == 44 and FOCS_BODY.size == 68 and MAP1_PREFIX.size == 32

COLOR_MODES: Final = ("tag", "energy", "clade", "house", "kin", "archetype", "intent")  # play's header list
FLAG_DIAG: Final = 1 << 0  # FRMA: the state has an invariant (error) diag bit set
FLAG_BACKLOG: Final = 1 << 0  # EVNT: the events rebuilt at start
FLAG_TS: Final = 1 << 0  # MAP1: the T_s plane is present
FLAG_WINDS: Final = 1 << 1  # MAP1: the wind blocks are present
FLAG_BORN_SINCE: Final = 1 << 0  # IDNT row flag: born after the last snapshot

SLOT_UNKNOWN: Final = 253  # house of an immigrant lineage of an observer attached mid-run
SLOT_OTHER: Final = 254  # listed in no top-16 table
SLOT_NONE: Final = 255
RESERVED_SLOTS: Final = (SLOT_UNKNOWN, SLOT_OTHER, SLOT_NONE)
MAX_TABLE: Final = 16  # rows of a cast table (houses, kin groups, colonies)

MAX_EVENTS: Final = 64
MAX_STORMS: Final = 32
MAX_TEXT: Final = 120
EVENT_FAMILIES: Final = ("kin", "house", "storm", "focus", "director", "system")
EVENT_KINDS: Final = {
    1: "kin_born",
    2: "kin_split",
    3: "kin_merged",
    4: "kin_dissolved",
    5: "house_founded",
    6: "house_extinct",
    7: "storm_born",
    8: "storm_named",
    9: "storm_merged",
    10: "storm_split",
    11: "storm_dissolved",
    12: "director_entry",
    13: "focus_death",
    14: "focus_handoff",
    15: "checkpoint",
    16: "renumbered",
    17: "extinct",
}
EVENT_KEYS: Final = frozenset(["id", "step", "kind", "fam", "a", "b", "x", "y", "z", "imp", "text"])
CLCK_REASONS: Final = {0: "running", 1: "user pause", 2: "end of steps", 3: "chapter", 4: "alert", 5: "error"}

# The weather maps (MAP1): reference scales of the companded and linear quantisation (DESIGN §33.1).
F0: Final = 0.025
P_REF: Final = 0.004
U_REF: Final = 0.1
ZETA_REF: Final = F0
NAME_ZETA: Final = 0.5  # a storm is named when its peak lower-layer vorticity reaches 0.5 f0 (C11)
MAP_CELLS: Final = 160
MAP_COARSE: Final = 40

# Caps (bytes) a record may not exceed; the tests pin them.
IDNT_JSON_CAP: Final = 16 * 1024
EVNT_JSON_CAP: Final = 32 * 1024
STRM_JSON_CAP: Final = 24 * 1024
ENVS_JSON_CAP: Final = 8 * 1024
TERR_JSON_CAP: Final = 1024


class RecordError(ValueError):
    """An encoder was given something the format cannot carry (a writer bug, never a data condition)."""


# --- primitives -----------------------------------------------------------------------------------------


def pad4(b: bytes) -> bytes:
    return b + b"\x00" * (-len(b) % 4)


def json_bytes(obj: Any) -> bytes:
    """The UTF-8 JSON of ``obj`` as the records carry it; NaN and infinities are a writer error."""
    try:
        return json.dumps(obj, allow_nan=False, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except ValueError as e:
        raise RecordError(f"not JSON: {e}") from e


def _le(a: Any, dtype: str) -> bytes:
    return np.ascontiguousarray(a, dtype=dtype).tobytes()


def record(tag: str, step: int, seq: int, flags: int, body: bytes, version: int = VERSION) -> bytes:
    """``tag``, the payload length and the payload: the 16-byte common head and ``body`` (a multiple of 4)."""
    raw = tag.encode("ascii")
    if len(raw) != 4:
        raise RecordError(f"tag {tag!r} is not 4 bytes")
    if len(body) % 4:
        raise RecordError(f"{tag} body of {len(body)} bytes is not a multiple of 4")
    if not (0 <= step < 1 << 64 and 0 <= seq < 1 << 32 and 0 <= flags < 1 << 16):
        raise RecordError(f"{tag} head out of range: step {step}, seq {seq}, flags {flags}")
    payload = HEAD.pack(step, seq, version, flags) + body
    return raw + struct.pack("<I", len(payload)) + payload


class Sequencer:
    """The per-tag ``seq`` counters of one stream, from 0."""

    def __init__(self) -> None:
        self._next = dict.fromkeys(TAGS, 0)

    def take(self, tag: str) -> int:
        n = self._next[tag]
        self._next = {**self._next, tag: n + 1}
        return n


def header_bytes(info: Mapping[str, Any]) -> bytes:
    """``MAGIC``, the u32 length of the JSON and the JSON, space-padded to a multiple of 4 (as ``stream``)."""
    body = json.dumps(info, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    body += b" " * (-len(body) % 4)
    return MAGIC + struct.pack("<I", len(body)) + body


# --- the header -----------------------------------------------------------------------------------------


def legacy_header(
    *,
    dims: int,
    size: float,
    height: float,
    n: int,
    bond_slots: int,
    radius: float,
    field_cell: float,
    fg: int,
    fz: int,
    nch: int,
    step: int,
    run: str | None,
    config_hash: str,
    source: str,
) -> dict[str, Any]:
    """The keys of an ``alife stream`` header with the values a play stream pins: ``mode`` simulate,
    ``every`` 1, ``fields`` off, ``color`` tag and the play colour modes (so ``StreamHeader.validate``
    accepts it)."""
    if source not in ("run", "preset"):
        raise RecordError(f"a play header's source is run or preset, not {source!r}")
    return {
        "protocol": PROTOCOL,
        "dims": dims,
        "L": float(size),
        "height": float(height),
        "N": n,
        "B": bond_slots,
        "radius": float(radius),
        "field_cell": float(field_cell),
        "FG": fg,
        "FZ": fz,
        "NCH": nch,
        "color_modes": list(COLOR_MODES),
        "color": "tag",
        "fields": False,
        "field_z": 0,
        "step": step,
        "run": run,
        "config_hash": config_hash,
        "mode": "simulate",
        "every": 1,
        "source": source,
    }


def weather_block(*, enabled: bool, dc_ref: float | None) -> dict[str, Any]:
    """The ``play.weather`` header object: the geometry and reference scales of ``MAP1`` and ``STRM``."""
    if enabled and not (dc_ref is not None and math.isfinite(dc_ref) and dc_ref > 0):
        raise RecordError("a weather world needs a positive dc_ref")
    return {
        "enabled": bool(enabled),
        "cells": MAP_CELLS,
        "dx": 2.0,
        "f0": F0,
        "coarse": MAP_COARSE,
        "name_zeta": NAME_ZETA,
        "refs": {"p": P_REF, "dc": dc_ref if enabled else 0.0, "u": U_REF, "zeta": ZETA_REF},
    }


def header_play(
    *,
    records: Sequence[str],
    weather: Mapping[str, Any],
    labels: Mapping[str, Any],
    energy_cap: int,
    planet: Mapping[str, Any] | None = None,
    base_rate: float = 14.4,
    rates: Sequence[float] = (0.25, 0.5, 1, 2),
    tick: int = 25,
    snapshot_every: int = 250,
    frame_hz: int = 30,
    chapter: int = 6000,
    cast: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """The ``play`` header key: the tags actually sent (a 2D or dry world has no ``MAP1``/``STRM``), their
    versions and the constants the client needs."""
    bad = [t for t in records if t not in TAGS]
    if bad or len(set(records)) != len(records):
        raise RecordError(f"records must be distinct known tags, got {list(records)}")
    return {
        "records": list(records),
        "versions": {t: VERSIONS[t] for t in records},
        "base_rate": base_rate,
        "rates": list(rates),
        "tick": tick,
        "snapshot_every": snapshot_every,
        "frame_hz": frame_hz,
        "chapter": chapter,
        "weather": dict(weather),
        "labels": dict(labels),
        "cast": dict(cast)
        if cast is not None
        else {"kin": MAX_TABLE, "house": MAX_TABLE, "colony": MAX_TABLE},
        "energy_cap": energy_cap,
        "events": {str(k): v for k, v in EVENT_KINDS.items()},
        "planet": dict(planet) if planet is not None else {"source": None, "houses": "none"},
    }


def play_header_bytes(legacy: Mapping[str, Any], play: Mapping[str, Any]) -> bytes:
    """The stream preamble and header object: the legacy keys and ``play`` (old readers ignore it)."""
    if "play" in legacy:
        raise RecordError("the legacy header already has a play key")
    return header_bytes({**legacy, "play": dict(play)})


# --- FRMA -----------------------------------------------------------------------------------------------


def _popcount_rank(alive: np.ndarray) -> np.ndarray:
    """Index in the listed rows of each row (meaningful for alive rows)."""
    return np.cumsum(alive.astype(np.int64)) - 1


def alive_bitmap(alive: np.ndarray) -> bytes:
    """``4·ceil(N/32)`` bytes: ``u32`` words (little endian), bit ``r`` of the stream = row ``r`` alive."""
    return pad4(np.packbits(alive.astype(np.uint8), bitorder="little").tobytes())


def encode_frma(
    *,
    step: int,
    seq: int,
    alive: np.ndarray,
    pos: np.ndarray,
    orient: np.ndarray,
    rgb: np.ndarray,
    uid: np.ndarray,
    bonds: np.ndarray,
    color_index: int,
    span: int = 0,
    diag_error: bool = False,
) -> bytes:
    """One ``FRMA``: the alive rows of an ``N``-row state.

    ``alive`` (N,) bool; ``pos`` (N,2|3) float (z = 0 when 2 columns); ``orient`` (N,2) (yaw, pitch); ``rgb``
    (N,) ``0x00RRGGBB``; ``uid`` (N,) int64; ``bonds`` (k,2) **row** pairs ``i < j``, both alive, in the order
    the stream lists them (ascending ``i``, then slot order); they are written as listed-row indices.
    """
    alive = np.asarray(alive).astype(bool)
    n_cap = alive.shape[0]
    n = int(alive.sum())
    pos = np.asarray(pos, np.float32)
    if pos.shape[1] == 2:
        pos = np.concatenate([pos, np.zeros((n_cap, 1), np.float32)], axis=1)
    bonds = np.asarray(bonds, np.int64).reshape(-1, 2)
    if bonds.size:
        if not (bonds[:, 0] < bonds[:, 1]).all() or bonds.min() < 0 or bonds.max() >= n_cap:
            raise RecordError("a bond is not a pair of rows i < j inside N")
        if not (alive[bonds[:, 0]] & alive[bonds[:, 1]]).all():
            raise RecordError("a bond ends in a dead row")
    listed = _popcount_rank(alive)[bonds] if bonds.size else np.zeros((0, 2), np.int64)
    body = (
        FRMA_PREFIX.pack(n, len(bonds), color_index, span, n_cap)
        + alive_bitmap(alive)
        + _le(pos[alive], "<f4")
        + _le(np.asarray(orient, np.float32)[alive], "<f2")
        + _le(np.asarray(rgb, np.uint32)[alive], "<u4")
        + _le(np.asarray(uid, np.int64)[alive], "<i8")
        + _le(listed, "<u4")
    )
    return record("FRMA", step, seq, FLAG_DIAG if diag_error else 0, body)


def frma_length(n_cap: int, n: int, n_bonds: int) -> int:
    """The payload length of an ``FRMA`` (§33.1)."""
    return HEAD_SIZE + FRMA_PREFIX.size + 4 * math.ceil(n_cap / 32) + 28 * n + 8 * n_bonds


# --- tables: IDNT, INTN, TERR ---------------------------------------------------------------------------


def _check_slot(values: np.ndarray, valid: Iterable[int], what: str) -> None:
    """Every slot byte of ``values`` is the slot of a table entry (``valid``) or reserved (253 to 255)."""
    v = np.asarray(values)
    ok = np.isin(v, list(valid)) | np.isin(v, RESERVED_SLOTS)
    if not ok.all():
        raise RecordError(f"{what} slot {int(v[~ok][0])} is not the slot of a listed entry")


def _entry_slots(entries: Sequence[Mapping[str, Any]], key: str) -> list[int]:
    """The ``slot`` fields of a table's entries: distinct integers below ``MAX_TABLE``. A slot is sticky in
    the observer (a group keeps it while it lives), so a table can have gaps and its entry's position is not
    its slot."""
    slots = [e.get("slot") for e in entries]
    if not all(isinstance(s, int) and not isinstance(s, bool) and 0 <= s < MAX_TABLE for s in slots):
        raise RecordError(f"IDNT {key} entries need an integer slot below {MAX_TABLE}")
    if len(set(slots)) != len(slots):
        raise RecordError(f"IDNT {key} entries repeat a slot")
    return slots  # type: ignore[return-value]


def encode_idnt(
    *,
    step: int,
    seq: int,
    snapshot_step: int,
    uid: np.ndarray,
    clade: np.ndarray,
    house: np.ndarray,
    kin: np.ndarray,
    colony: np.ndarray,
    flags: np.ndarray,
    tables: Mapping[str, Any],
) -> bytes:
    """One ``IDNT``: ``tables`` (the JSON: ``houses``, ``exact``, ``origin``, ``kin``, ``colonies``,
    ``unknown_immigrant``) and one 16-byte row per alive row, ascending row:
    ``i64 uid | i32 clade | u8 house | u8 kin | u8 colony | u8 flags``. A row's slot byte is the ``slot``
    field of an entry of its table (the entries' positions need not equal their slots) or 253 to 255."""
    for key in ("houses", "kin", "colonies"):
        if len(tables.get(key, ())) > MAX_TABLE:
            raise RecordError(f"IDNT {key} lists more than {MAX_TABLE} rows")
    n = len(uid)
    for column, key, what in (
        (house, "houses", "house"),
        (kin, "kin", "kin"),
        (colony, "colonies", "colony"),
    ):
        _check_slot(np.asarray(column), _entry_slots(tables.get(key, ()), key), what)
    js = json_bytes(tables)
    if len(js) > IDNT_JSON_CAP:
        raise RecordError(f"IDNT json of {len(js)} bytes exceeds {IDNT_JSON_CAP}")
    rows = np.concatenate(
        [
            np.frombuffer(_le(uid, "<i8"), np.uint8).reshape(n, 8),
            np.frombuffer(_le(clade, "<i4"), np.uint8).reshape(n, 4),
            np.stack([np.asarray(a, np.uint8) for a in (house, kin, colony, flags)], axis=1).reshape(n, 4),
        ],
        axis=1,
    )
    body = struct.pack("<IIII", n, snapshot_step, len(js), 0) + pad4(js) + rows.tobytes()
    return record("IDNT", step, seq, 0, body)


def encode_intn(
    *,
    step: int,
    seq: int,
    labels_step: int,
    intent_version: int,
    uid: np.ndarray,
    arch_primary: np.ndarray,
    arch_secondary: np.ndarray,
    intent: np.ndarray,
    flags: np.ndarray,
) -> bytes:
    """One ``INTN``: ``n_rows × 12 B``, each ``i64 uid | u8 arch_primary | u8 arch_secondary | u8 intent |
    u8 flags`` (bit 0 evaluated, bit 1 Budding overlay); the bytes are ``rules.Archetype`` / ``rules.Intent``
    values."""
    n = len(uid)
    rows = np.concatenate(
        [
            np.frombuffer(_le(uid, "<i8"), np.uint8).reshape(n, 8),
            np.stack(
                [np.asarray(a, np.uint8) for a in (arch_primary, arch_secondary, intent, flags)], axis=1
            ).reshape(n, 4),
        ],
        axis=1,
    )
    body = struct.pack("<IIII", n, labels_step, intent_version, 0) + rows.tobytes()
    return record("INTN", step, seq, 0, body)


def encode_terr(
    *,
    step: int,
    seq: int,
    snapshot_step: int,
    ids: Sequence[int],
    owner: np.ndarray,
    purity: np.ndarray,
) -> bytes:
    """One ``TERR``: ``owner`` and ``purity`` are ``(ny, nx)`` uint8 (row ``y``, column ``x``); ``owner`` is a
    slot of ``ids`` (the kin id of each slot, -1 for a slot that holds none) or 255."""
    owner = np.asarray(owner, np.uint8)
    purity = np.asarray(purity, np.uint8)
    if owner.shape != purity.shape or owner.ndim != 2:
        raise RecordError("owner and purity must be (ny, nx) of one shape")
    ny, nx = owner.shape
    if len(ids) > MAX_TABLE:
        raise RecordError(f"TERR lists more than {MAX_TABLE} kin ids")
    _check_slot(owner.reshape(-1), [i for i, k in enumerate(ids) if k >= 0], "territory owner")
    js = json_bytes({"ids": list(ids)})
    if len(js) > TERR_JSON_CAP:
        raise RecordError("TERR json is too long")
    body = (
        struct.pack("<HHIII", nx, ny, snapshot_step, len(js), 0)
        + pad4(js)
        + pad4(owner.tobytes() + purity.tobytes())
    )
    return record("TERR", step, seq, 0, body)


# --- STRM, EVNT, ENVS -----------------------------------------------------------------------------------


def encode_strm(*, step: int, seq: int, tick: int, obj: Mapping[str, Any]) -> bytes:
    """One ``STRM``: ``obj`` is ``{"detect_step", "storms": [...], "ended": [...]}``, at most 32 storms."""
    storms = obj.get("storms", ())
    if len(storms) > MAX_STORMS:
        raise RecordError(f"STRM lists {len(storms)} storms, the cap is {MAX_STORMS}")
    js = json_bytes(obj)
    if len(js) > STRM_JSON_CAP:
        raise RecordError(f"STRM json of {len(js)} bytes exceeds {STRM_JSON_CAP}")
    return record("STRM", step, seq, 0, struct.pack("<IIII", len(js), tick, len(storms), 0) + pad4(js))


def check_event(e: Mapping[str, Any]) -> None:
    """The shape of one event: the 11 keys, a known family, ``imp`` 0 to 2, text up to 120 characters."""
    if set(e) != EVENT_KEYS:
        raise RecordError(f"event keys {sorted(e)} are not {sorted(EVENT_KEYS)}")
    if not (isinstance(e["id"], int) and e["id"] >= 0 and isinstance(e["step"], int) and e["step"] >= 0):
        raise RecordError("event id and step must be non-negative integers")
    if not isinstance(e["kind"], int) or e["fam"] not in EVENT_FAMILIES or e["imp"] not in (0, 1, 2):
        raise RecordError(
            f"event kind, family or importance out of range: {e['kind']!r} {e['fam']!r} {e['imp']!r}"
        )
    if not (isinstance(e["a"], int) and isinstance(e["b"], int)):
        raise RecordError("event entity ids must be integers")
    if not isinstance(e["text"], str) or len(e["text"]) > MAX_TEXT:
        raise RecordError(f"event text must be a string of at most {MAX_TEXT} characters")


def encode_evnt(*, step: int, seq: int, events: Sequence[Mapping[str, Any]], backlog: bool = False) -> bytes:
    """One ``EVNT``: at most 64 events as a JSON array."""
    if len(events) > MAX_EVENTS:
        raise RecordError(f"EVNT carries {len(events)} events, the cap is {MAX_EVENTS}")
    for e in events:
        check_event(e)
    js = json_bytes(list(events))
    if len(js) > EVNT_JSON_CAP:
        raise RecordError(f"EVNT json of {len(js)} bytes exceeds {EVNT_JSON_CAP}")
    body = struct.pack("<II", len(events), len(js)) + pad4(js)
    return record("EVNT", step, seq, FLAG_BACKLOG if backlog else 0, body)


def encode_envs(*, step: int, seq: int, obj: Mapping[str, Any]) -> bytes:
    """One ``ENVS``: ``{"pools", "season", "levers", "effects", "cooldowns", "influence"}``."""
    js = json_bytes(obj)
    if len(js) > ENVS_JSON_CAP:
        raise RecordError(f"ENVS json of {len(js)} bytes exceeds {ENVS_JSON_CAP}")
    return record("ENVS", step, seq, 0, struct.pack("<I", len(js)) + pad4(js))


# --- MAP1 -----------------------------------------------------------------------------------------------


def quantise_map1(
    *,
    p: np.ndarray,
    dc: np.ndarray,
    zeta: np.ndarray,
    ts: np.ndarray,
    winds: np.ndarray,
    dc_ref: float,
    p_ref: float = P_REF,
    u_ref: float = U_REF,
    zeta_ref: float = ZETA_REF,
) -> dict[str, np.ndarray]:
    """The planes of a ``MAP1`` from the weather fields (``(n, n)`` for the first four, ``(4, n, n)`` for
    the winds ``u1, v1, u2, v2``), computed in float64:

    - ``P`` u8 ``round(255·sqrt(clip(P/p_ref, 0, 1)))`` (decode ``(b/255)²·p_ref``);
    - ``Dc`` u8 ``round(255·clip(Dc/dc_ref, 0, 1))``; ``zeta`` i8 ``round(127·clip(zeta/zeta_ref, -1, 1))``;
    - ``Ts`` u8 ``round(255·clip(Ts, 0, 1))``;
    - ``winds`` i8 ``(4, nw, nw)``: block means (``n/nw`` per side) ``round(127·clip(w/u_ref, -1, 1))``.
    """
    p, dc, zeta, ts = (np.asarray(a, np.float64) for a in (p, dc, zeta, ts))
    w = np.asarray(winds, np.float64)
    n = p.shape[0]
    nw = MAP_COARSE
    if not (p.shape == dc.shape == zeta.shape == ts.shape == (n, n) and w.shape == (4, n, n) and n % nw == 0):
        raise RecordError(f"MAP1 planes must be (n, n) and winds (4, n, n) with n a multiple of {nw}")
    b = n // nw
    coarse = w.reshape(4, nw, b, nw, b).mean(axis=(2, 4))
    return {
        "P": np.rint(255.0 * np.sqrt(np.clip(p / p_ref, 0.0, 1.0))).astype(np.uint8),
        "Dc": np.rint(255.0 * np.clip(dc / dc_ref, 0.0, 1.0)).astype(np.uint8),
        "zeta": np.rint(127.0 * np.clip(zeta / zeta_ref, -1.0, 1.0)).astype(np.int8),
        "Ts": np.rint(255.0 * np.clip(ts, 0.0, 1.0)).astype(np.uint8),
        "winds": np.rint(127.0 * np.clip(coarse / u_ref, -1.0, 1.0)).astype(np.int8),
    }


def encode_map1(
    *,
    step: int,
    seq: int,
    planes: Mapping[str, np.ndarray],
    dc_ref: float,
    p_ref: float = P_REF,
    u_ref: float = U_REF,
    zeta_ref: float = ZETA_REF,
    f0: float = F0,
) -> bytes:
    """One ``MAP1`` from quantised ``planes`` (``quantise_map1``): ``P``, ``Dc``, ``zeta``, ``Ts`` of
    ``(n, n)`` and ``winds`` of ``(4, nw, nw)``. Every version-1 record carries ``T_s`` and the winds (flags
    bits 0 and 1), so a newer record never lacks what an older one had."""
    n = planes["P"].shape[0]
    nw = planes["winds"].shape[1]
    for k in ("P", "Dc", "zeta", "Ts"):
        if planes[k].shape != (n, n):
            raise RecordError(f"MAP1 plane {k} has shape {planes[k].shape}, not ({n}, {n})")
    if planes["winds"].shape != (4, nw, nw):
        raise RecordError("MAP1 winds must be (4, nw, nw)")
    body = (
        MAP1_PREFIX.pack(n, nw, f0, p_ref, dc_ref, u_ref, zeta_ref)
        + _le(planes["P"], "u1")
        + _le(planes["Dc"], "u1")
        + _le(planes["zeta"], "i1")
        + _le(planes["Ts"], "u1")
        + _le(planes["winds"], "i1")
    )
    return record("MAP1", step, seq, FLAG_TS | FLAG_WINDS, pad4(body))


def dequantise_p(b: np.ndarray, p_ref: float = P_REF) -> np.ndarray:
    """Rain rate from the companded byte: ``(b/255)²·p_ref``."""
    return (np.asarray(b, np.float64) / 255.0) ** 2 * p_ref


# --- CLCK, FOCS -----------------------------------------------------------------------------------------


def encode_clck(
    *,
    step: int,
    seq: int,
    rate: float,
    achieved: float,
    step_ms: float,
    frame_ms: float,
    late_steps: int,
    save_stall_steps: int,
    reason: int,
    saving: bool,
    fe: int,
    chapter: int,
    in_chapter: int,
    chapter_len: int,
    flags: int = 0,
) -> bytes:
    """One ``CLCK`` (44 bytes after the head): the clock and the scheduler's counters (§33.1)."""
    if reason not in CLCK_REASONS:
        raise RecordError(f"CLCK reason {reason} is not one of {sorted(CLCK_REASONS)}")
    body = CLCK_BODY.pack(
        rate,
        achieved,
        step_ms,
        frame_ms,
        late_steps,
        save_stall_steps,
        reason,
        int(saving),
        flags,
        fe,
        chapter,
        in_chapter,
        chapter_len,
    )
    return record("CLCK", step, seq, 0, body)


FOCS_ALIVE: Final = 1 << 0
FOCS_HANDOFF: Final = 1 << 1
FOCS_FOLLOWED: Final = 1 << 2


def encode_focs(
    *,
    step: int,
    seq: int,
    uid: int,
    row: int,
    flags: int,
    pos: Sequence[float],
    vel: Sequence[float],
    energy: int,
    age: int,
    generation: int,
    parent_uid: int,
    n_bonds: int,
    house: int,
    kin: int,
    colony: int,
    arch_primary: int,
    arch_secondary: int,
    intent: int,
    label_flags: int,
) -> bytes:
    """One ``FOCS`` (68 bytes after the head): the focused organism's row and the observer's slots."""
    for name, s in (("house", house), ("kin", kin), ("colony", colony)):
        if not (0 <= s < MAX_TABLE or s in RESERVED_SLOTS):
            raise RecordError(f"FOCS {name} slot {s} is not a slot or 253 to 255")
    body = FOCS_BODY.pack(
        uid,
        row,
        flags,
        *pos,
        *vel,
        energy,
        age,
        generation,
        parent_uid,
        n_bonds,
        house,
        kin,
        colony,
        arch_primary,
        arch_secondary,
        intent,
        label_flags,
    )
    return record("FOCS", step, seq, 0, body)


def encode_unknown(tag: str, step: int, seq: int, body: bytes, *, version: int = VERSION) -> bytes:
    """A record of any tag (for the fixtures' unknown tag and the version tests)."""
    return record(tag, step, seq, 0, body, version)


def flatten(parts: Iterable[bytes]) -> bytes:
    return b"".join(parts)
