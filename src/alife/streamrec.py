"""The play-loop stream records, ALSTRM v1 additions (DESIGN §33.1, §34.6, docs/GAMES.md, G3a and G3b).

Host module, numpy only (no ``mlx`` import): every encoder is a pure function of arrays and dicts, so the
fixtures of ``tests/play_fixtures.py`` are device independent. The records are written by ``alife play`` after
the header of ``alife stream`` (§27.11) and decoded by ``AlifeCore/PlayRecords.swift``;
``tests/play_decode.py`` is an independent decoder written from §33.1, not from this file.

Conventions (§33.1): little endian; a record is a 4-byte tag, a ``u32`` payload length (a multiple of 4) and
the payload, which starts with the common head ``u64 step, u32 seq, u16 version, u16 flags`` (16 bytes);
every section starts at a multiple of 4 bytes; JSON is UTF-8 from ``json_bytes`` with ``json_len`` the
unpadded length followed by 0 to 3 zero bytes; unknown tags are skipped by length, unknown versions are
skipped and counted.

The sphere world (DESIGN §46.7) writes protocol 2: the header names the planet
(``legacy_header(sphere=...)``), the weather map is ``MAPH`` (``encode_maph``) over the ring-ordered HEALPix
cells of the coarse weather grid, ``TERR`` is indexed by the territory pixels, and ``STRC``/``FSOL`` by
``layer·Npix + pixel``. Every other
record is as in a box world; a box stream is byte for byte what it was.
"""

from __future__ import annotations

import itertools
import json
import math
import struct
import threading
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

import numpy as np

MAGIC: Final = b"ALSTRM\x00\x01"
PROTOCOL: Final = 1
PROTOCOL_SPHERE: Final = (
    2  # the header of a sphere world (DESIGN §46.7); old clients refuse it by this number
)
VERSION: Final = 1  # the layout version of every tag below
TAGS: Final = (
    "FRMA", "IDNT", "INTN", "TERR", "EVNT", "STRM", "MAP1", "ENVS", "CLCK", "FOCS", "ALRT", "RPRT", "PLAN",
)  # fmt: skip
VERSIONS: Final = dict.fromkeys(TAGS, VERSION)
# The tags of the feature extensions (DESIGN §43.5), in ``ext.EXTENSIONS`` order: sent only while their
# feature
# is on. ``tests/test_ext_registry.py`` pins them to ``ext.REGISTRY[...].stream_tags`` (this module imports no
# ``mlx``, so it cannot read the registry).
EXT_TAGS: Final = ("STRC", "FSIG", "FSOL")  # structures, comm, farming
EXT_VERSIONS: Final = dict.fromkeys(EXT_TAGS, VERSION)
SPHERE_TAGS: Final = ("MAPH",)  # the sphere's weather map (DESIGN §46.7): sent only in a protocol 2 stream
SPHERE_VERSIONS: Final = dict.fromkeys(SPHERE_TAGS, VERSION)
ALL_TAGS: Final = (*TAGS, *EXT_TAGS, *SPHERE_TAGS)
HEAD: Final = struct.Struct("<QIHH")  # step, seq, version, flags
HEAD_SIZE: Final = 16
FRMA_PREFIX: Final = struct.Struct("<IIIII")  # n, n_bonds, color_index, span, N
CLCK_BODY: Final = struct.Struct("<ffffIIBBHIIII")  # 44 bytes
FOCS_BODY: Final = struct.Struct("<qIIffffffiIIqBBBBBBBB")  # 68 bytes
MAP1_PREFIX: Final = struct.Struct("<HHfffff8x")  # n, nw, f0, p_ref, dc_ref, u_ref, zeta_ref, 8 reserved
MAPH_PREFIX: Final = struct.Struct("<HHfffff8x")  # nside, k, f0, p_ref, dc_ref, u_ref, zeta_ref, 8 reserved
CELLS_PREFIX: Final = struct.Struct("<HHHHI")  # STRC, FSOL: nx, ny, nz, flags, count
FSIG_PREFIX: Final = struct.Struct("<qBBH")  # FSIG: uid, W, flags, audible
assert HEAD.size == HEAD_SIZE and CLCK_BODY.size == 44 and FOCS_BODY.size == 68 and MAP1_PREFIX.size == 32
assert MAPH_PREFIX.size == 32
assert CELLS_PREFIX.size == 12 and FSIG_PREFIX.size == 12

COLOR_MODES: Final = ("tag", "energy", "clade", "house", "kin", "archetype", "intent")  # play's header list
EXT_COLOR_MODES: Final = ("role",)  # appended to the header list while the bonds feature is on (DESIGN §40.7)
FLAG_BONDFLOW: Final = 1 << 1  # FRMA: an int8 flow per drawn bond follows the bond pairs (bonds feature)
FLAG_TRUNC: Final = (
    1 << 0
)  # STRC, FSOL: more cells were non-empty than ``CELLS_MAX``; the highest levels are kept
FLAG_HEARD: Final = 1 << 0  # FSIG body flags: the "hears" half is valid (the observer channel ran)
CELLS_MAX: Final = 65536  # cells in one STRC or FSOL
FSIG_MAX_W: Final = 8
FLAG_DIAG: Final = 1 << 0  # FRMA: the state has an invariant (error) diag bit set
FLAG_BACKLOG: Final = 1 << 0  # EVNT, ALRT, RPRT: the records rebuilt at start
FLAG_FINAL: Final = 1 << 1  # RPRT: no later report of this chapter and mode will replace it
FLAG_REPLY: Final = 1 << 0  # PLAN: the reply to a ``power preview``
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
    18: "kin_formed",
    19: "kin_ended",
    20: "power_applied",
    21: "power_ended",
}
EVENT_KEYS: Final = frozenset(["id", "step", "kind", "fam", "a", "b", "x", "y", "z", "imp", "text"])
CLCK_REASONS: Final = {0: "running", 1: "user pause", 2: "end of steps", 3: "chapter", 4: "alert", 5: "error"}
CLCK_FLAG_CHAPTER_PAUSE: Final = 1 << 0  # CLCK.flags: the pause at a chapter end is armed
CLCK_FLAG_ALERT_PAUSE: Final = 1 << 1  # CLCK.flags: at least one alert class is armed to pause
CLCK_FLAG_SAVES_NEWEST: Final = 1 << 2  # CLCK.flags: the world keeps only the newest save (``saves newest``
# or ``single``)
CLCK_FLAG_SAVES_SINGLE: Final = 1 << 3  # CLCK.flags: with bit 2, the origin goes too (``saves single``)
CLCK_FLAGS: Final = (
    CLCK_FLAG_CHAPTER_PAUSE | CLCK_FLAG_ALERT_PAUSE | CLCK_FLAG_SAVES_NEWEST | CLCK_FLAG_SAVES_SINGLE
)
SAVE_MODES: Final = ("all", "newest", "single")  # the save modes of ``saves`` lines and ``play.saves.mode``

# The weather maps (MAP1): reference scales of the companded and linear quantisation (DESIGN §33.1).
F0: Final = 0.025
P_REF: Final = 0.004
U_REF: Final = 0.1
ZETA_REF: Final = F0
NAME_ZETA: Final = 0.5  # a storm is named when its peak lower-layer vorticity reaches 0.5 f0 (C11)
MAP_CELLS: Final = 160  # the 320 world's weather grid; the default of ``weather_block``
MAP_DX: Final = 2.0
MAP_COARSE: Final = 40

# Caps (bytes) a record may not exceed; the tests pin them.
IDNT_JSON_CAP: Final = 16 * 1024
EVNT_JSON_CAP: Final = 32 * 1024
STRM_JSON_CAP: Final = 24 * 1024
ENVS_JSON_CAP: Final = 8 * 1024
TERR_JSON_CAP: Final = 1024
ALRT_JSON_CAP: Final = 24 * 1024
RPRT_JSON_CAP: Final = 64 * 1024
PLAN_JSON_CAP: Final = 8 * 1024

# G3b records (DESIGN §34.6).
MAX_ALERTS: Final = 16  # alerts in one ALRT
ALERT_ID_BASE: Final = (
    600  # an alert id is first_step * 1000 + 600 + index (the chronicle's families use 0..500)
)
ALERT_TITLE_MAX: Final = 48
ALERT_TEXT_MAX: Final = 160
ALERT_IMPORTANCE: Final = (0, 1, 2, 3)  # 0 log only, 1 stack, 2 stack and toast, 3 toast and armed pause
ALERT_STATES: Final = ("raised", "updated", "cleared")
ALERT_SURFACES: Final = ("alert", "log")
ALERT_REF_KINDS: Final = ("storm", "kin", "house", "world", "power", "organism")
ALERT_KEYS: Final = frozenset(
    [
        "id",
        "version",
        "step",
        "class",
        "key",
        "state",
        "surface",
        "imp",
        "title",
        "text",
        "x",
        "y",
        "z",
        "ref",
        "pause",
        "expires",
        "watched",
        "calibrated",
    ]
)
ALERT_CLASS_KEYS: Final = frozenset(["imp", "pause", "label", "calibrated", "surface"])
ALERT_CAP: Final = {"n": 4, "window": 1500, "exempt": ["imp3", "pause", "chapter", "engine"]}
REPORT_MODES: Final = {0: "template", 1: "select", 2: "away"}
REPORT_KEYS: Final = frozenset(
    [
        "chapter",
        "n_abs",
        "a",
        "b",
        "mode",
        "final",
        "withheld",
        "headline",
        "lead",
        "body",
        "interventions",
        "ticker",
        "since",
        "entities",
        "versions",
        "facts_digest",
        "latency_s",
        "fallback",
    ]
)
PLAN_KINDS: Final = ("power", "intent")
PLAN_STATUSES: Final = ("preview", "queued", "applied", "refused", "since", "cancelled")
PLAN_KEYS: Final = frozenset(
    [
        "id",
        "kind",
        "status",
        "power",
        "action",
        "raw",
        "params",
        "clamped",
        "duration",
        "step_applied",
        "ends",
        "cost",
        "influence",
        "ready",
        "holds",
        "text",
        "caution",
        "precedent",
        "since",
        "refused",
    ]
)
PLAN_TEXT_MAX: Final = 160
PLAN_CAUTION_MAX: Final = 100
EFFECT_SOURCES: Final = ("rule", "llm", "player")


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


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


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
    """The per-tag ``seq`` counters of one stream, from 0. Thread-safe: a worker (the report job) takes the
    numbers of its own tag while the loop takes the others', and without the lock two threads taking different
    tags could lose an increment and repeat a ``seq``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next = dict.fromkeys(ALL_TAGS, 0)

    def take(self, tag: str) -> int:
        with self._lock:
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
    color_modes: Sequence[str] = COLOR_MODES,
    sphere: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The keys of an ``alife stream`` header with the values a play stream pins: ``mode`` simulate,
    ``every`` 1, ``fields`` off, ``color`` tag and the play colour modes (so ``StreamHeader.validate``
    accepts it). ``color_modes`` is ``COLOR_MODES`` plus ``EXT_COLOR_MODES`` of the features that are on.

    ``sphere`` (``sphere_block``) makes it a protocol 2 header of a planet: ``protocol`` is 2, ``L`` is
    absent, ``FG`` is 0 (a sphere has no lattice side; ``FZ`` is the layer count) and the planet's keys follow
    ``FZ``. ``radius`` stays the body radius of the agents."""
    if source not in ("run", "preset"):
        raise RecordError(f"a play header's source is run or preset, not {source!r}")
    head = {
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
        "color_modes": list(color_modes),
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
    return head if sphere is None else sphere_header(head, sphere)


def sphere_block(
    *,
    planet_radius: float,
    shell_edges: Sequence[float],
    nside_field: int,
    nside_weather: int,
    nside_territory: int,
    weather_depth: float,
) -> dict[str, Any]:
    """The planet keys of a protocol 2 header (DESIGN §46.7): ``geometry`` "sphere", ``planet_radius`` (the
    planet's ``Rp``; the header's ``radius`` is the agents' body radius), ``shell_edges`` (the ``FZ + 1``
    layer boundary radii, ascending, from ``Rp`` to ``Rp + H``), the three Nsides (``nside_weather`` is 0
    without weather), ``ring_order`` "ring" (every cell index is the RING pixel), ``f0`` and
    ``weather_depth`` (``H1 + H2``, 0 without weather)."""
    edges = [float(e) for e in shell_edges]
    if not (planet_radius > 0 and len(edges) >= 3 and all(a < b for a, b in itertools.pairwise(edges))):
        raise RecordError("a sphere needs a positive planet_radius and ascending shell_edges (FZ + 1 >= 3)")
    if not (_is_int(nside_field) and nside_field >= 2 and _is_int(nside_territory) and nside_territory >= 2):
        raise RecordError("nside_field and nside_territory are integers >= 2")
    if not (_is_int(nside_weather) and nside_weather >= 0 and weather_depth >= 0):
        raise RecordError("nside_weather is an integer >= 0 and weather_depth >= 0")
    return {
        "geometry": "sphere",
        "planet_radius": float(planet_radius),
        "shell_edges": edges,
        "nside_field": int(nside_field),
        "nside_weather": int(nside_weather),
        "nside_territory": int(nside_territory),
        "ring_order": "ring",
        "f0": F0,
        "weather_depth": float(weather_depth),
    }


def sphere_header(head: Mapping[str, Any], sphere: Mapping[str, Any]) -> dict[str, Any]:
    """``head`` (a ``legacy_header`` dict) as a protocol 2 header: ``protocol`` 2, no ``L``, ``FG`` 0, and the
    ``sphere_block`` keys after ``FZ`` (JSON key order is not significant; this one is documented)."""
    keys = [k for k in head if k != "L"]
    cut = keys.index("FZ") + 1  # the planet's keys follow FZ
    pinned = {"protocol": PROTOCOL_SPHERE, "FG": 0}
    return {
        **{k: pinned.get(k, head[k]) for k in keys[:cut]},
        **sphere,
        **{k: pinned.get(k, head[k]) for k in keys[cut:]},
    }


def weather_block_sphere(*, enabled: bool, dc_ref: float | None, nside: int, coarse: int) -> dict[str, Any]:
    """The ``play.weather`` header object of a sphere (DESIGN §46.7): the weather Nside ``nside`` (0 when the
    weather is off), the Nside ``coarse`` of the ``MAPH`` planes (the field Nside; ``nside / coarse`` is a
    power of two) and the reference scales of ``MAPH`` and ``STRM``."""
    if enabled and not (dc_ref is not None and math.isfinite(dc_ref) and dc_ref > 0):
        raise RecordError("a weather world needs a positive dc_ref")
    return {
        "enabled": bool(enabled),
        "nside": int(nside) if enabled else 0,
        "coarse": int(coarse),
        "f0": F0,
        "name_zeta": NAME_ZETA,
        "refs": {"p": P_REF, "dc": dc_ref if enabled else 0.0, "u": U_REF, "zeta": ZETA_REF},
    }


def weather_block(
    *, enabled: bool, dc_ref: float | None, cells: int = MAP_CELLS, dx: float = MAP_DX
) -> dict[str, Any]:
    """The ``play.weather`` header object: the geometry and reference scales of ``MAP1`` and ``STRM``.

    ``cells`` and ``dx`` are the weather layer's (``WeatherParams.n`` and ``.dx``: 160 and 2.0 for the 320
    world, 80 and 2.0 for the mini world, DESIGN §37.4); ``coarse`` and the reference scales are fixed."""
    if enabled and not (dc_ref is not None and math.isfinite(dc_ref) and dc_ref > 0):
        raise RecordError("a weather world needs a positive dc_ref")
    return {
        "enabled": bool(enabled),
        "cells": int(cells),
        "dx": float(dx),
        "f0": F0,
        "coarse": MAP_COARSE,
        "name_zeta": NAME_ZETA,
        "refs": {"p": P_REF, "dc": dc_ref if enabled else 0.0, "u": U_REF, "zeta": ZETA_REF},
    }


def labels_block(
    *, intent_version: int, arch: Sequence[str], intent: Sequence[str], gate: Mapping[str, Any]
) -> dict[str, Any]:
    """The ``play.labels`` header object (DESIGN §33.1.1, §33.7): the rule set's version, the display names
    indexed by the byte of ``INTN``/``IDNT``/``FOCS`` (index 0 is the empty name of "none"; a weather world
    says Grazer for Gatherer, §32.7), and the ``gate``: whether N1's held-out verdict table has a row for
    this world and rule set (``key_found``) and the bytes of the labels that pass it (``archetypes``,
    ``intents``, ascending). A label is valid only if its byte is listed and the environment is at its
    defaults (no lever in ``ENVS.levers``, the client's check); everything else is provisional."""
    if not arch or not intent or arch[0] != "" or intent[0] != "":
        raise RecordError("label name tables start with the empty name of 'none'")
    need = {"rules_digest", "world_key", "key_found", "archetypes", "intents"}
    if set(gate) != need:
        raise RecordError(f"a label gate has exactly the keys {sorted(need)}, got {sorted(gate)}")
    for key, names in (("archetypes", arch), ("intents", intent)):
        vals = gate[key]
        if list(vals) != sorted(set(vals)) or any(
            not isinstance(v, int) or isinstance(v, bool) or not 1 <= v < len(names) for v in vals
        ):
            raise RecordError(f"gate.{key} must be ascending bytes of the name table, got {list(vals)}")
    if not gate["key_found"] and (gate["archetypes"] or gate["intents"]):
        raise RecordError("a gate without a verdict row lists no valid label")
    return {
        "intent_version": int(intent_version),
        "arch": list(arch),
        "intent": list(intent),
        "gate": {
            "rules_digest": str(gate["rules_digest"]),
            "world_key": str(gate["world_key"]),
            "key_found": bool(gate["key_found"]),
            "archetypes": [int(v) for v in gate["archetypes"]],
            "intents": [int(v) for v in gate["intents"]],
        },
    }


def alerts_block(
    classes: Mapping[str, Mapping[str, Any]], cap: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The ``play.alerts`` header object (DESIGN §34.6): the registry of alert classes (each ``imp`` 0 to 3,
    ``pause`` the class's default, ``label`` its display name, ``calibrated`` whether a measurement backs
    its threshold, ``surface`` ``alert`` or ``log``) and the global rate cap. ``ALRT`` records name only
    classes of this registry."""
    out: dict[str, Any] = {}
    for name, c in classes.items():
        if set(c) != ALERT_CLASS_KEYS or not isinstance(name, str) or not name:
            raise RecordError(f"alert class {name!r} has keys {sorted(c)}, not {sorted(ALERT_CLASS_KEYS)}")
        if c["imp"] not in ALERT_IMPORTANCE or c["surface"] not in ALERT_SURFACES:
            raise RecordError(f"alert class {name!r}: importance or surface out of range")
        if not (
            isinstance(c["pause"], bool) and isinstance(c["calibrated"], bool) and isinstance(c["label"], str)
        ):
            raise RecordError(f"alert class {name!r}: pause and calibrated are booleans, label a string")
        out = {
            **out,
            name: {
                "imp": c["imp"], "pause": c["pause"], "label": c["label"], "calibrated": c["calibrated"],
                "surface": c["surface"],
            },
        }  # fmt: skip
    if not out:
        raise RecordError("the alert registry is empty")
    return {"version": VERSION, "classes": out, "cap": dict(cap) if cap is not None else dict(ALERT_CAP)}


def autopause_block(*, chapter: bool, alert: Mapping[str, bool]) -> dict[str, Any]:
    """The ``play.autopause`` header object: whether the world pauses at a chapter end and, per alert class,
    whether a raised alert of the class pauses it (the defaults; the client sends its own choice back with
    ``autopause`` lines)."""
    if not isinstance(chapter, bool) or not all(isinstance(v, bool) for v in alert.values()):
        raise RecordError("autopause flags are booleans")
    return {"chapter": chapter, "alert": dict(alert)}


def saves_block(*, mode: str, keep: int, adopted: bool = False) -> dict[str, Any]:
    """The ``play.saves`` header object (DESIGN §34.10, §34.11): the save mode the world starts in (``all``
    keeps the newest ``keep`` saves; ``newest`` only the newest, with the origin; ``single`` only the newest,
    the origin included). The client reads the mode back from ``CLCK.flags`` bits 2 and 3 afterwards.
    ``adopted`` (written only when true): the world is a research run played in place, which refuses a
    ``saves newest`` or ``saves single`` line (its checkpoints are the research's)."""
    if mode not in SAVE_MODES:
        raise RecordError(f"the save mode is one of {SAVE_MODES}, got {mode!r}")
    if not _is_int(keep) or keep < 1:
        raise RecordError("saves.keep is a positive integer")
    if not isinstance(adopted, bool):
        raise RecordError("saves.adopted is a boolean")
    return {"mode": mode, "keep": keep, **({"adopted": True} if adopted else {})}


def powers_block(powers: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The ``play.powers`` header object: the powers the player can use, by name (``action`` the director
    action behind it, ``label``, ``cost`` in influence units, ``cooldown`` in steps, and whatever bounds the
    client draws its sliders from)."""
    for name, p in powers.items():
        if not isinstance(name, str) or not name or not {"action", "label", "cost", "cooldown"} <= set(p):
            raise RecordError(f"power {name!r} needs action, label, cost and cooldown")
        if not (_is_int(p["cost"]) and _is_int(p["cooldown"]) and p["cost"] >= 0 and p["cooldown"] >= 0):
            raise RecordError(f"power {name!r}: cost and cooldown are non-negative integers")
    return {name: dict(p) for name, p in powers.items()}


def cells_block(*, cap: int, cells: Sequence[int], cell: float, **extra: Any) -> dict[str, Any]:
    """The ``play.structures`` and ``play.niche`` header objects (DESIGN §38.9, §41.14): the grid the sparse
    cell records (``STRC``, ``FSOL``) index, ``cells`` = ``[FZ, FG, FG]`` (``[1, FG, FG]`` in 2D), ``cell``
    the
    world length of one cell and ``cap`` the material or soil cap the levels are scaled by."""
    if not (_is_int(cap) and cap >= 1):
        raise RecordError("cells.cap is a positive integer")
    c = [int(v) for v in cells]
    if len(c) != 3 or any(v < 1 or v > 0xFFFF for v in c):
        raise RecordError(f"cells is [FZ, FG, FG] with each in [1, 65535], got {list(cells)}")
    if not (_is_num(cell) and cell > 0):
        raise RecordError("cells.cell is a positive length")
    return {"enabled": True, **extra, "cap": int(cap), "cells": c, "cell": float(cell)}


def comm_block(*, w: int, hear_range: float, control: str) -> dict[str, Any]:
    """The ``play.comm`` header object (DESIGN §39.8): the number of voice channels, the hearing range and the
    engine control (``none``, ``deaf`` or ``decoy``)."""
    if not (_is_int(w) and 1 <= w <= FSIG_MAX_W):
        raise RecordError(f"comm.w is in [1, {FSIG_MAX_W}]")
    if control not in ("none", "deaf", "decoy"):
        raise RecordError(f"comm.control is none, deaf or decoy, got {control!r}")
    return {"enabled": True, "w": int(w), "range": float(hear_range), "control": control}


def bonds_block(*, scale: float) -> dict[str, Any]:
    """The ``play.bonds`` header object (DESIGN §40.7): written exactly when ``FRMA`` carries the bond flow
    (``FLAG_BONDFLOW``); ``scale`` is ``bonds.share_rate`` (an int8 flow of 127 is that many quanta a
    step)."""
    if not (_is_num(scale) and scale > 0):
        raise RecordError("bonds.scale is a positive number")
    return {"flow": True, "scale": float(scale)}


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
    alerts: Mapping[str, Any] | None = None,
    autopause: Mapping[str, Any] | None = None,
    powers: Mapping[str, Any] | None = None,
    game_specs: str | None = None,
    saves: Mapping[str, Any] | None = None,
    structures: Mapping[str, Any] | None = None,
    comm: Mapping[str, Any] | None = None,
    niche: Mapping[str, Any] | None = None,
    bonds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``play`` header key: the tags actually sent (a 2D or dry world has no ``MAP1``/``STRM``), their
    versions and the constants the client needs. G3b's keys (``alerts_block``, ``autopause_block``,
    ``powers_block`` and the name of the game table) are written only when given, like the other optional
    keys of the headers, so a world that does not use them has the header it always had. ``saves``
    (``saves_block``, §34.10) is optional in the same way."""
    bad = [t for t in records if t not in ALL_TAGS]
    if bad or len(set(records)) != len(records):
        raise RecordError(f"records must be distinct known tags, got {list(records)}")
    extra: dict[str, Any] = {}
    if alerts is not None:
        extra = {**extra, "alerts": dict(alerts)}
    if autopause is not None:
        extra = {**extra, "autopause": dict(autopause)}
    if powers is not None:
        extra = {**extra, "powers": dict(powers)}
    if game_specs is not None:
        if not isinstance(game_specs, str) or not game_specs:
            raise RecordError("game_specs names the game table")
        extra = {**extra, "game_specs": game_specs}
    if saves is not None:
        extra = {**extra, "saves": dict(saves)}
    for key, block in (("structures", structures), ("comm", comm), ("niche", niche), ("bonds", bonds)):
        if block is not None:  # the feature extensions (DESIGN §43.5): written only while their feature is on
            extra = {**extra, key: dict(block)}
    return {
        "records": list(records),
        "versions": {t: {**VERSIONS, **EXT_VERSIONS, **SPHERE_VERSIONS}[t] for t in records},
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
        **extra,
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
    flow: np.ndarray | None = None,
) -> bytes:
    """One ``FRMA``: the alive rows of an ``N``-row state. ``flow`` ((k,) int8, one per bond, DESIGN §40.7)
    sets
    ``FLAG_BONDFLOW`` and follows the bond pairs, padded to 4 bytes.

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
    flags = FLAG_DIAG if diag_error else 0
    if flow is not None:
        fl = np.asarray(flow)
        if fl.dtype != np.int8 or fl.shape != (len(bonds),):
            raise RecordError(
                f"FRMA flow is (k,) int8 with k = {len(bonds)} bonds, got {fl.dtype} {fl.shape}"
            )
        body = body + pad4(_le(fl, "i1"))
        flags |= FLAG_BONDFLOW
    return record("FRMA", step, seq, flags, body)


def frma_length(n_cap: int, n: int, n_bonds: int, *, flow: bool = False) -> int:
    """The payload length of an ``FRMA`` (§33.1); ``flow`` adds the bond-flow section (§40.7)."""
    base = HEAD_SIZE + FRMA_PREFIX.size + 4 * math.ceil(n_cap / 32) + 28 * n + 8 * n_bonds
    return base + (4 * math.ceil(n_bonds / 4) if flow else 0)


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


def check_envs(obj: Mapping[str, Any]) -> None:
    """The G3b keys of an ``ENVS`` object, when present: ``influence`` (None, or ``have``, ``cap``,
    ``window`` and the ``free_at`` steps at which spent units come back), ``patches`` (the food patches the
    powers target: ``i x y gain sigma``), ``powers`` (per power name ``ready``, the step it is ready or
    None, and ``busy``, the reason it is not usable or None) and the source of every effect (``rule``,
    ``llm`` or ``player``)."""
    influence = obj.get("influence")
    if influence is not None:
        ok = (
            isinstance(influence, Mapping)
            and set(influence) == {"have", "cap", "window", "free_at"}
            and all(_is_int(influence[k]) and influence[k] >= 0 for k in ("have", "cap", "window"))
            and isinstance(influence["free_at"], list)
            and all(_is_int(v) and v >= 0 for v in influence["free_at"])
            and influence["have"] <= influence["cap"]
        )
        if not ok:
            raise RecordError(f"ENVS influence {influence!r} is not {{have, cap, window, free_at}}")
    for i, p in enumerate(obj.get("patches", ())):
        if not (isinstance(p, Mapping) and set(p) == {"i", "x", "y", "gain", "sigma"} and _is_int(p["i"])):
            raise RecordError(f"ENVS patch {i} is not {{i, x, y, gain, sigma}}")
        if not all(_is_num(p[k]) for k in ("x", "y", "gain", "sigma")):
            raise RecordError(f"ENVS patch {i} has a value that is not a finite number")
    for name, p in (obj.get("powers") or {}).items():
        ok = isinstance(p, Mapping) and set(p) == {"ready", "busy"}
        ok = (
            ok
            and (p["ready"] is None or _is_int(p["ready"]))
            and (p["busy"] is None or isinstance(p["busy"], str))
        )
        if not (ok and isinstance(name, str)):
            raise RecordError(f"ENVS power {name!r} is not {{ready, busy}}")
    for e in obj.get("effects", ()):
        if e.get("source") not in EFFECT_SOURCES:
            raise RecordError(f"ENVS effect source {e.get('source')!r} is not one of {EFFECT_SOURCES}")


def encode_envs(*, step: int, seq: int, obj: Mapping[str, Any]) -> bytes:
    """One ``ENVS``: ``{"pools", "season", "levers", "effects", "cooldowns", "influence"}``, and, from G3b,
    ``patches`` and ``powers`` (``check_envs``)."""
    check_envs(obj)
    js = json_bytes(obj)
    if len(js) > ENVS_JSON_CAP:
        raise RecordError(f"ENVS json of {len(js)} bytes exceeds {ENVS_JSON_CAP}")
    return record("ENVS", step, seq, 0, struct.pack("<I", len(js)) + pad4(js))


# --- ALRT, RPRT, PLAN (G3b) -----------------------------------------------------------------------------


def _text(v: Any, limit: int, what: str) -> None:
    if not isinstance(v, str) or len(v) > limit:
        raise RecordError(f"{what} must be a string of at most {limit} characters")


def _position(a: Mapping[str, Any]) -> None:
    for axis in ("x", "y", "z"):
        if a[axis] is not None and not _is_num(a[axis]):
            raise RecordError(f"alert {axis} must be a finite number or null")


def check_alert(a: Mapping[str, Any], registry: Iterable[str] | None = None) -> None:
    """The shape of one alert (§34.6): the 18 keys, an id of the alert range (``first_step * 1000 + 600 +
    index``), a class of the header registry (``registry``, when given), a state and a surface of the closed
    lists, importance 0 to 3, ``title`` up to 48 characters and ``text`` up to 160, a ``ref`` of a known kind,
    and a ``version`` that is 1 when raised and 2 or more when updated."""
    if set(a) != ALERT_KEYS:
        raise RecordError(f"alert keys {sorted(a)} are not {sorted(ALERT_KEYS)}")
    if not (_is_int(a["id"]) and a["id"] >= 0 and a["id"] % 1000 >= ALERT_ID_BASE):
        raise RecordError(f"alert id {a['id']!r} is not first_step * 1000 + {ALERT_ID_BASE} + index")
    if not (_is_int(a["version"]) and a["version"] >= 1 and _is_int(a["step"]) and a["step"] >= 0):
        raise RecordError("alert version and step must be integers, version at least 1")
    if not isinstance(a["class"], str) or (registry is not None and a["class"] not in set(registry)):
        raise RecordError(f"alert class {a['class']!r} is not in the header's registry")
    if not isinstance(a["key"], str):
        raise RecordError("alert key must be a string")
    if (
        a["state"] not in ALERT_STATES
        or a["surface"] not in ALERT_SURFACES
        or a["imp"] not in ALERT_IMPORTANCE
    ):
        raise RecordError(
            f"alert state, surface or importance out of range: {a['state']!r} {a['surface']!r} {a['imp']!r}"
        )
    if (a["state"] == "raised" and a["version"] != 1) or (a["state"] == "updated" and a["version"] < 2):
        raise RecordError(f"alert state {a['state']!r} does not fit version {a['version']}")
    _text(a["title"], ALERT_TITLE_MAX, "alert title")
    _text(a["text"], ALERT_TEXT_MAX, "alert text")
    _position(a)
    ref = a["ref"]
    if ref is not None and not (
        isinstance(ref, Mapping)
        and set(ref) == {"kind", "id"}
        and ref["kind"] in ALERT_REF_KINDS
        and _is_int(ref["id"])
    ):
        raise RecordError(f"alert ref {ref!r} is not {{kind, id}} of a known kind")
    if not (
        isinstance(a["pause"], bool) and isinstance(a["watched"], bool) and isinstance(a["calibrated"], bool)
    ):
        raise RecordError("alert pause, watched and calibrated are booleans")
    if a["expires"] is not None and not _is_int(a["expires"]):
        raise RecordError("alert expires must be a step or null")


def encode_alrt(
    *,
    step: int,
    seq: int,
    alerts: Sequence[Mapping[str, Any]],
    backlog: bool = False,
    registry: Iterable[str] | None = None,
) -> bytes:
    """One ``ALRT``: at most 16 alerts as a JSON array (``u32 count``, ``u32 json_len``, JSON from offset 24,
    zero padding). The client keeps the newest ``version`` of an ``id``. ``registry`` is the class names of
    the header's ``alerts_block``."""
    if len(alerts) > MAX_ALERTS:
        raise RecordError(f"ALRT carries {len(alerts)} alerts, the cap is {MAX_ALERTS}")
    classes = None if registry is None else set(registry)
    for a in alerts:
        check_alert(a, classes)
    if len({a["id"] for a in alerts}) != len(alerts):
        raise RecordError("an ALRT names an alert id twice")
    js = json_bytes(list(alerts))
    if len(js) > ALRT_JSON_CAP:
        raise RecordError(f"ALRT json of {len(js)} bytes exceeds {ALRT_JSON_CAP}")
    body = struct.pack("<II", len(alerts), len(js)) + pad4(js)
    return record("ALRT", step, seq, FLAG_BACKLOG if backlog else 0, body)


def _sentence(v: Any, what: str, keys: frozenset[str]) -> None:
    if not (
        isinstance(v, Mapping)
        and set(v) == keys
        and isinstance(v["text"], str)
        and isinstance(v["refs"], list)
    ):
        raise RecordError(f"report {what} is not {sorted(keys)} with a text and a list of refs")
    if not all(isinstance(r, str) for r in v["refs"]):
        raise RecordError(f"report {what} refs are entity keys (strings)")


def check_report(r: Mapping[str, Any]) -> None:
    """The shape of the JSON of an ``RPRT`` (§34.6): the 18 keys, ``mode`` 0 (template), 1 (select) or 2
    (away), sentences with a ``text`` and their ``refs`` (keys of ``entities``), the cut steps ``a <= b``,
    the versions of the four parts of the report, and a ``withheld`` list that is empty or leaves the report
    empty."""
    if set(r) != REPORT_KEYS:
        raise RecordError(f"report keys {sorted(r)} are not {sorted(REPORT_KEYS)}")
    if not all(_is_int(r[k]) and r[k] >= 0 for k in ("chapter", "n_abs", "a", "b")) or r["a"] > r["b"]:
        raise RecordError("report chapter, n_abs, a and b must be non-negative integers with a <= b")
    if r["mode"] not in REPORT_MODES or not isinstance(r["final"], bool):
        raise RecordError(f"report mode {r['mode']!r} or final {r['final']!r} out of range")
    if not (isinstance(r["withheld"], list) and all(isinstance(w, str) for w in r["withheld"])):
        raise RecordError("report withheld is a list of reasons")
    for part in ("headline", "lead"):
        if r[part] is not None:
            _sentence(r[part], part, frozenset(["id", "text", "refs"]))
    for part, keys in (
        ("body", frozenset(["id", "text", "origin", "refs"])),
        ("interventions", frozenset(["id", "text", "refs"])),
    ):
        if not isinstance(r[part], list):
            raise RecordError(f"report {part} is a list")
        for v in r[part]:
            _sentence(v, part, keys)
    ids = [v["id"] for v in (r["headline"], r["lead"]) if v is not None]
    ids += [v["id"] for part in ("body", "interventions") for v in r[part]]
    if len(set(ids)) != len(ids):
        raise RecordError("report sentence ids are unique (a client lists the sentences by id)")
    if r["withheld"] and (
        r["headline"] is not None or r["lead"] is not None or r["body"] or r["interventions"]
    ):
        raise RecordError("a withheld report is empty: no guess is sent")
    if not (
        isinstance(r["ticker"], list)
        and all(isinstance(t, Mapping) and set(t) == {"step", "text"} for t in r["ticker"])
    ):
        raise RecordError("report ticker is a list of {step, text}")
    since = r["since"]
    if not (
        isinstance(since, Mapping)
        and set(since) == {"influence", "alerts", "ending"}
        and isinstance(since["alerts"], Mapping)
        and isinstance(since["ending"], list)
    ):
        raise RecordError("report since is {influence, alerts, ending}")
    if not isinstance(r["entities"], Mapping) or not isinstance(r["versions"], Mapping):
        raise RecordError("report entities and versions are objects")
    if set(r["versions"]) != {"report", "template", "checker", "lexicon"} or not all(
        _is_int(v) for v in r["versions"].values()
    ):
        raise RecordError("report versions are {report, template, checker, lexicon} integers")
    if not isinstance(r["facts_digest"], str) or not (r["latency_s"] is None or _is_num(r["latency_s"])):
        raise RecordError("report facts_digest is a string and latency_s a number or null")
    if r["fallback"] is not None and not isinstance(r["fallback"], str):
        raise RecordError("report fallback is a reason or null")


def encode_rprt(*, step: int, seq: int, obj: Mapping[str, Any], backlog: bool = False) -> bytes:
    """One ``RPRT``: ``u32 chapter``, ``u32 mode``, ``u32 json_len``, ``u32 reserved`` and the JSON from
    offset 32 (up to 64 KiB). ``chapter``, ``mode`` and the *final* flag (bit 1) are those of ``obj``, so the
    head and the JSON cannot disagree; the client keeps the newest record per ``(chapter, mode)``."""
    check_report(obj)
    js = json_bytes(obj)
    if len(js) > RPRT_JSON_CAP:
        raise RecordError(f"RPRT json of {len(js)} bytes exceeds {RPRT_JSON_CAP}")
    body = struct.pack("<IIII", obj["chapter"], obj["mode"], len(js), 0) + pad4(js)
    flags = (FLAG_BACKLOG if backlog else 0) | (FLAG_FINAL if obj["final"] else 0)
    return record("RPRT", step, seq, flags, body)


def check_plan(p: Mapping[str, Any]) -> None:
    """The shape of the JSON of a ``PLAN`` (§34.6): the 20 keys, a ``kind`` and ``status`` of the closed
    lists, a ``text`` up to 160 characters and cautions up to 100, ``refused`` a reason exactly when the
    status is ``refused``, and ``step_applied`` a step when the status is ``applied``."""
    if set(p) != PLAN_KEYS:
        raise RecordError(f"plan keys {sorted(p)} are not {sorted(PLAN_KEYS)}")
    if not (_is_int(p["id"]) and p["id"] >= 0):
        raise RecordError("plan id is the client's request number, a non-negative integer")
    if p["kind"] not in PLAN_KINDS or p["status"] not in PLAN_STATUSES:
        raise RecordError(f"plan kind {p['kind']!r} or status {p['status']!r} out of range")
    for k in ("power", "action", "holds"):
        if p[k] is not None and not isinstance(p[k], str):
            raise RecordError(f"plan {k} is a string or null")
    for k in ("duration", "step_applied", "ends", "ready"):
        if p[k] is not None and not _is_int(p[k]):
            raise RecordError(f"plan {k} is a step count or null")
    if not (_is_int(p["cost"]) and p["cost"] >= 0):
        raise RecordError("plan cost is a non-negative integer")
    if not all(isinstance(p[k], Mapping) for k in ("raw", "params")):
        raise RecordError("plan raw and params are objects")
    if not (isinstance(p["clamped"], list) and all(isinstance(c, str) for c in p["clamped"])):
        raise RecordError("plan clamped is a list of parameter names")
    _text(p["text"], PLAN_TEXT_MAX, "plan text")
    if not isinstance(p["caution"], list):
        raise RecordError("plan caution is a list")
    for c in p["caution"]:
        _text(c, PLAN_CAUTION_MAX, "plan caution")
    inf = p["influence"]
    if inf is not None and not (
        isinstance(inf, Mapping) and set(inf) == {"have", "cap", "window", "free_at"}
    ):
        raise RecordError("plan influence is {have, cap, window, free_at} or null")
    for k in ("precedent", "since"):
        if p[k] is not None and not isinstance(p[k], Mapping):
            raise RecordError(f"plan {k} is an object or null")
    if (p["status"] == "refused") != isinstance(p["refused"], str):
        raise RecordError("plan refused is a reason exactly when the status is refused")
    if p["status"] == "applied" and p["step_applied"] is None:
        raise RecordError("an applied plan names its step")


def encode_plan(*, step: int, seq: int, obj: Mapping[str, Any], reply: bool = False) -> bytes:
    """One ``PLAN``: ``u32 json_len`` and the JSON from offset 20 (up to 8 KiB). ``reply`` (flag bit 0) marks
    the answer to a ``power preview``; a ``preview`` status is always one."""
    check_plan(obj)
    if obj["status"] == "preview" and not reply:
        raise RecordError("a preview is the reply to a preview request")
    js = json_bytes(obj)
    if len(js) > PLAN_JSON_CAP:
        raise RecordError(f"PLAN json of {len(js)} bytes exceeds {PLAN_JSON_CAP}")
    return record("PLAN", step, seq, FLAG_REPLY if reply else 0, struct.pack("<I", len(js)) + pad4(js))


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


def encode_maph(
    *,
    step: int,
    seq: int,
    planes: Mapping[str, np.ndarray],
    nside: int,
    k: int,
    dc_ref: float,
    p_ref: float = P_REF,
    u_ref: float = U_REF,
    zeta_ref: float = ZETA_REF,
    f0: float = F0,
) -> bytes:
    """One ``MAPH`` from quantised ``planes`` (``quantise_maph``) of ``12·nside²`` ring-ordered cells: ``P``,
    ``Dc``, ``zeta``, ``Ts`` of ``(npix,)`` and ``winds`` of ``(4, npix)``. ``k`` is the log2 of the ratio of
    the weather Nside to ``nside`` (informational: the weather resolution the planes were pooled from). Like
    ``MAP1`` every record carries ``T_s`` and the winds (flags bits 0 and 1)."""
    npix = 12 * nside * nside
    for key in ("P", "Dc", "zeta", "Ts"):
        if planes[key].shape != (npix,):
            raise RecordError(f"MAPH plane {key} has shape {planes[key].shape}, not ({npix},)")
    if planes["winds"].shape != (4, npix):
        raise RecordError(f"MAPH winds must be (4, {npix})")
    if not (_is_int(nside) and 2 <= nside < 0x10000 and _is_int(k) and 0 <= k < 0x10000):
        raise RecordError("MAPH nside and k are 16-bit unsigned integers, nside >= 2")
    body = (
        MAPH_PREFIX.pack(nside, k, f0, p_ref, dc_ref, u_ref, zeta_ref)
        + _le(planes["P"], "u1")
        + _le(planes["Dc"], "u1")
        + _le(planes["zeta"], "i1")
        + _le(planes["Ts"], "u1")
        + _le(planes["winds"], "i1")
    )
    return record("MAPH", step, seq, FLAG_TS | FLAG_WINDS, pad4(body), version=SPHERE_VERSIONS["MAPH"])


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
    """One ``CLCK`` (44 bytes after the head): the clock and the scheduler's counters (§33.1). ``flags`` bit 0
    is "the pause at a chapter end is armed", bit 1 "at least one alert class is armed to pause" (G3b), bit 2
    "the world keeps only the newest save" (§34.10)."""
    if reason not in CLCK_REASONS:
        raise RecordError(f"CLCK reason {reason} is not one of {sorted(CLCK_REASONS)}")
    if flags & ~CLCK_FLAGS:
        raise RecordError(f"CLCK flags {flags:#x} set bits other than the pauses and the save mode")
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


# --- STRC, FSOL: sparse cell levels (DESIGN §38.9, §41.14) -------------------------------------------------


def quantise_cells(grid: np.ndarray, cap: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """``(idx u32, level u8, truncated)`` of a material or soil grid: the flat z-major indices of the
    non-empty
    cells, ascending, and ``level = 1 + (254·m) // cap`` (a non-empty cell is at least 1, a full one 255).
    More than ``CELLS_MAX`` non-empty cells keep the highest levels (ties: the lowest index first) and set
    ``truncated``."""
    if not (_is_int(cap) and cap >= 1):
        raise RecordError("cap is a positive integer")
    # the engine keeps 0 <= m <= cap; a value outside (an edit of the grid, a changed cap on resume) is
    # clamped
    # on the wire rather than ending a play session from the stream hook
    m = np.clip(np.asarray(grid).reshape(-1).astype(np.int64), 0, cap)
    idx = np.flatnonzero(m > 0)
    level = 1 + (254 * m[idx]) // cap
    truncated = idx.size > CELLS_MAX
    if truncated:
        keep = np.sort(np.lexsort((idx, -level))[:CELLS_MAX])  # highest level, then lowest index
        idx, level = idx[keep], level[keep]
    return idx.astype("<u4"), level.astype("u1"), bool(truncated)


def encode_cells(tag: str, *, step: int, seq: int, shape: Sequence[int], grid: np.ndarray, cap: int) -> bytes:
    """One ``STRC`` or ``FSOL`` of ``grid`` (``(FG, FG)`` or ``(FZ, FG, FG)`` integer units, 0..``cap``)."""
    if tag not in ("STRC", "FSOL"):
        raise RecordError(f"{tag!r} is not a sparse cell record")
    nz, ny, nx = (int(v) for v in shape)
    g = np.asarray(grid)
    if g.size != nz * ny * nx:
        raise RecordError(f"{tag} grid has {g.size} cells, the header says {nz}x{ny}x{nx}")
    idx, level, trunc = quantise_cells(g, cap)
    flags = FLAG_TRUNC if trunc else 0
    body = CELLS_PREFIX.pack(nx, ny, nz, flags, idx.size) + idx.tobytes() + level.tobytes()
    return record(tag, step, seq, flags, pad4(body))


def encode_strc(*, step: int, seq: int, shape: Sequence[int], mat: np.ndarray, cap: int) -> bytes:
    """One ``STRC``: the built material of the world (``structures.cap`` is ``cap``)."""
    return encode_cells("STRC", step=step, seq=seq, shape=shape, grid=mat, cap=cap)


def encode_fsol(*, step: int, seq: int, shape: Sequence[int], soil: np.ndarray, cap: int) -> bytes:
    """One ``FSOL``: the farmed soil of the world (``niche.farm_cap`` is ``cap``)."""
    return encode_cells("FSOL", step=step, seq=seq, shape=shape, grid=soil, cap=cap)


# --- FSIG: the focused organism's voice (DESIGN §39.8) ------------------------------------------------------


def encode_fsig(
    *,
    step: int,
    seq: int,
    uid: int,
    says: np.ndarray,
    hears: np.ndarray | None,
    audible: int = 0,
) -> bytes:
    """One ``FSIG``: ``says`` and ``hears`` are ``(W,)`` uint8 (what the organism said after the step and what
    it heard before it); ``hears`` None clears ``FLAG_HEARD`` and writes zeros."""
    say = np.asarray(says)
    w = say.shape[0]
    if say.ndim != 1 or not 1 <= w <= FSIG_MAX_W:
        raise RecordError(f"FSIG says is a vector of 1 to {FSIG_MAX_W} channels, got shape {say.shape}")
    if say.dtype != np.uint8:
        raise RecordError("FSIG says is uint8")
    if hears is not None and (hears.shape != say.shape or hears.dtype != np.uint8):
        raise RecordError("FSIG hears is a uint8 vector like says")
    if not (_is_int(audible) and 0 <= audible <= 0xFFFF):
        raise RecordError("FSIG audible is a u16")
    flags = FLAG_HEARD if hears is not None else 0
    heard = hears if hears is not None else np.zeros(w, np.uint8)
    body = (
        FSIG_PREFIX.pack(uid, w, flags, audible if hears is not None else 0) + say.tobytes() + heard.tobytes()
    )
    return record("FSIG", step, seq, 0, pad4(body))


def encode_unknown(tag: str, step: int, seq: int, body: bytes, *, version: int = VERSION) -> bytes:
    """A record of any tag (for the fixtures' unknown tag and the version tests)."""
    return record(tag, step, seq, 0, body, version)


def flatten(parts: Iterable[bytes]) -> bytes:
    return b"".join(parts)
