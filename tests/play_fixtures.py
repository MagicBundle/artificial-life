"""The play-record fixtures for the Swift decoder: `tests/data/play_tiny{3d,2d,3d_n70}.{bin,json}`.

Each fixture is a complete play stream (the header with its `play` key and one of every record the world
sends) built from hand-set arrays by `alife.streamrec`: no engine, no device, no floating-point dynamics, so
the bytes are the same on every machine. Every number is exactly representable (dyadic floats), every JSON
number is written by the same `json.dumps`. The scenes:

- `3d` (N = 64, L = 48): alive rows with gaps (0-4, 10, 21), a bond across the periodic seam, a row with two
  bonds listed out of partner order, a hurt (red) agent, uids above 2^32, a second `FRMA` four steps later
  with a reused row (new uid) and a dead row between alive ones, an `IDNT` between snapshots with slots 253
  to 255 and a newborn flag, `TERR`, `ENVS`, a full `MAP1` (`T_s` and winds), a `STRM` with a named storm and
  a disturbance, a backlog `EVNT` and a live one, `CLCK`, `FOCS` with a hand-off, and a record of an unknown
  tag (`ZZZZ`).
- `2d` (N = 64, no weather): the same without `MAP1` and `STRM`; z = 0.
- `3d_n70` (N = 70): three bitmap words, the last partial (rows 64 to 69), which a multiple of 64 hides; a
  bond between rows 31 and 32 and between 63 and 64, and a second `FRMA` with every row alive.
- `3d_g3b` (N = 64, G3b, DESIGN section 34.6): the header keys `alerts`, `autopause`, `powers` and
  `game_specs`, and §34.10's `saves` with `CLCK.flags` bit 2 in the last clock; a chapter-end scene at step
  12000 with `ENVS` (a player effect, `influence`, `patches`, `powers`), `IDNT` kin entries with a `thread`,
  `EVNT` of kinds 18 to 21, a backlog `ALRT` and `RPRT`, a live `ALRT`
  (importance 0 to 3; raised, updated and cleared; a ref of each kind) and two `RPRT` (a template report and a
  select report still pending), `PLAN` records of every status (a preview reply, queued, applied,
  since, refused, cancelled, and an intent refusal), `CLCK` with reasons 3 and 4 and the 1-based chapter
  clock, and an `ALRT` of an unknown version (skipped). The three older fixtures are unchanged by it, so the
  Swift decoder of G3a still reads them.

The JSON sidecar holds the expected header object and, per record, the head fields, the decoded fields and a
SHA-256 of every section, produced by `play_decode` (an independent decoder written from DESIGN §33.1).

Regenerate after a deliberate change of the layout:

    uv run python tests/play_fixtures.py --write

`tests/test_stream_records.py::test_fixtures_match_the_code` rebuilds all three and compares them byte for
byte on every device.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from alife import streamrec as sr
from play_decode import Rec, decode

DATA = Path(__file__).resolve().parent / "data"
KINDS = ("3d", "2d", "3d_n70", "3d_g3b")
UID0 = 5_000_000_000  # above 2^32: the uid columns are 64-bit
CONFIG_HASH = "0" * 64


def fixture_name(kind: str) -> str:
    return f"play_tiny{kind}"


def paths(kind: str) -> tuple[Path, Path]:
    return DATA / f"{fixture_name(kind)}.bin", DATA / f"{fixture_name(kind)}.json"


# --- scenes ---------------------------------------------------------------------------------------------

# row, x, y, z, yaw, pitch, rgb
_ROWS_A = (
    (0, 1.0, 24.0, 6.0, 0.5, 0.5, 0x4080C0),
    (1, 47.0, 24.0, 7.0, -1.25, -0.25, 0xC04020),  # across the seam from row 0
    (2, 10.0, 10.0, 2.0, 3.0, 0.0, 0xFF0000),  # hurt: red in every mode
    (3, 12.0, 11.5, 3.5, -3.0, 1.0, 0x20A060),
    (4, 30.0, 40.0, 12.0, 1.0, -1.25, 0xFFDC28),
    (10, 31.0, 41.0, 12.5, 2.0, 0.75, 0x1E3CC8),
    (21, 40.0, 5.0, 9.0, 0.0, 0.125, 0x3CC878),
)
_ROWS_B = (  # four steps later: row 4 reused by a newborn, row 10 dead, a new row 5, row 0 and 1 moved
    (0, 1.25, 24.0, 6.0, 0.5, 0.5, 0x4080C0),
    (1, 47.25, 24.0, 7.0, -1.25, -0.25, 0xC04020),
    (2, 10.0, 10.25, 2.0, 3.0, 0.0, 0xFF0000),
    (3, 12.0, 11.75, 3.5, -3.0, 1.0, 0x20A060),
    (4, 20.5, 8.5, 1.0, 0.25, 0.0, 0xFFFFFF),
    (5, 20.75, 9.0, 1.5, -0.25, 0.0, 0x808080),
    (21, 40.0, 5.0, 9.0, 0.0, 0.125, 0x3CC878),
)


def _arrays(n_cap: int, rows: tuple[tuple[Any, ...], ...], uid_of: dict[int, int], dims: int):
    alive = np.zeros(n_cap, bool)
    pos = np.zeros((n_cap, 3), np.float32)
    orient = np.zeros((n_cap, 2), np.float32)
    rgb = np.zeros(n_cap, np.uint32)
    uid = np.full(n_cap, -1, np.int64)
    for row, x, y, z, yaw, pitch, color in rows:
        alive[row] = True
        pos[row] = (x, y, z if dims == 3 else 0.0)
        orient[row] = (yaw, pitch if dims == 3 else 0.0)
        rgb[row] = color
        uid[row] = uid_of[row]
    return alive, pos, orient, rgb, uid


def _tables() -> dict[str, Any]:
    return {
        "houses": [
            {"slot": 0, "id": 5000000000, "name": "Aldric", "size": 4, "clade": 3},
            {"slot": 1, "id": 5000000030, "name": "Brennar", "size": 2, "clade": 0},
        ],
        "exact": True,
        "origin": 0,
        "kin": [
            {
                "slot": 0,
                "id": 17,
                "name": "Aldric kin",
                "named_after": 0,
                "size": 3,
                "clade": 3,
                "clade_share": 1.0,
                "sim_self": 0.875,
                "x": 6.0,
                "y": 24.0,
                "z": 6.5,
                "arch": [0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0],
                "births": 2,
                "deaths": [1, 0, 0, 0, 0],
            },
            {
                "slot": 1,
                "id": 21,
                "name": "Brennar kin",
                "named_after": None,
                "size": 2,
                "clade": 0,
                "clade_share": 0.5,
                "sim_self": 0.75,
                "x": 11.0,
                "y": 10.75,
                "z": 2.75,
                "arch": None,
                "births": 0,
                "deaths": None,
            },
            {
                "slot": 2,
                "id": 22,
                "name": "Cestria",
                "named_after": 1,
                "size": 2,
                "clade": 9,
                "clade_share": 0.75,
                "sim_self": 0.5,
                "x": 30.5,
                "y": 40.5,
                "z": 12.25,
                "arch": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                "births": 1,
                "deaths": [0, 0, 0, 0, 0],
            },
        ],
        "colonies": [{"slot": 0, "id": 3, "size": 2, "kin": 17, "x": 11.0, "y": 10.75}],
        "unknown_immigrant": 0,
    }


def _storm_obj(step: int) -> dict[str, Any]:
    return {
        "detect_step": step,
        "storms": [
            {
                "id": 7,
                "name": "Merrow",
                "named": True,
                "hemi": 0,
                "x": 4.5,
                "y": 20.25,
                "r": 6.5,
                "ext": [7.0, 6.0],
                "zeta": 0.75,
                "zeta_now": 0.625,
                "wind": 0.0625,
                "rain": 0.001953125,
                "age": 1250,
                "parent": 0,
                "state": "active",
                "track": [[75, 3.0, 19.0], [100, 4.5, 20.25]],
                "pred": [[250, 8.0, 21.0], [500, 12.0, 22.0], [1000, 20.0, 25.0]],
            },
            {
                "id": 9,
                "name": None,
                "named": False,
                "hemi": 1,
                "x": 40.0,
                "y": 70.5,
                "r": 4.0,
                "ext": [4.0, 4.5],
                "zeta": 0.25,
                "zeta_now": 0.25,
                "wind": 0.03125,
                "rain": 0.0009765625,
                "age": 100,
                "parent": 7,
                "state": "coasting",
                "track": [[100, 40.0, 70.5]],
                "pred": [],
            },
        ],
        "ended": [{"id": 5, "reason": "merged", "into": 7}],
    }


def _events() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def ev(i, step, kind, fam, a, b, x, y, z, imp, text):
        return {
            "id": i,
            "step": step,
            "kind": kind,
            "fam": fam,
            "a": a,
            "b": b,
            "x": x,
            "y": y,
            "z": z,
            "imp": imp,
            "text": text,
        }

    backlog = [
        ev(96000, 96, 1, "kin", 17, 0, 6.0, 24.0, 6.5, 1, "A kin group of 3 appeared."),
        ev(99000, 99, 8, "storm", 7, 0, 4.5, 20.25, None, 2, "Storm Merrow reached named strength."),
    ]
    live = [
        ev(100000, 100, 14, "focus", 5000000007, 5000000014, 1.0, 24.0, 6.0, 1, "Follow passed to a child."),
    ]
    return backlog, live


def _pools() -> dict[str, Any]:
    return {
        "pools": [{"x": 24.0, "y": 12.0, "amp": 0.5, "sx": 6.0, "sy": 4.0, "vx": 0.0, "vy": 0.0}],
        "season": {"period": 4000, "amp": 0.25, "phase": 0.0},
        "levers": {"env_season_amp": 0.25},
        "effects": [
            {"action": "bloom", "params": {"x": 24.0, "y": 12.0}, "start": 90, "end": None, "source": "rule"}
        ],
        "cooldowns": [{"action": "bloom", "ready": 400}],
        "influence": None,
    }


def _terr_planes() -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:32, 0:32]
    owner = np.where((x < 22) & (y < 22), (x // 11 + y // 11) % 3, 255).astype(np.uint8)
    purity = ((x * 8 + y * 3) % 256).astype(np.uint8)
    return owner, purity


def _map_planes() -> dict[str, np.ndarray]:
    y, x = np.mgrid[0:160, 0:160]
    return {
        "P": ((x * 3 + y * 5) % 256).astype(np.uint8),
        "Dc": ((x + 2 * y) % 251).astype(np.uint8),
        "zeta": (((x * 7 - y * 3) % 255) - 127).astype(np.int8),
        "Ts": ((x * 255) // 159).astype(np.uint8),
        "winds": ((np.arange(4 * 40 * 40).reshape(4, 40, 40) * 11) % 255 - 127).astype(np.int8),
    }


ARCH = (
    "Gatherer",
    "Hunter",
    "Provider",
    "Trader",
    "Nomad",
    "Settler",
    "Unspecialised",
    "Drifter",
    "Floater",
    "Sinker",
)  # by rules.Archetype value - 1


def _labels(kind: str) -> dict[str, Any]:
    """The `play.labels` object of the rule set of version 3 (DESIGN §33.7): ten archetypes (Drifter,
    Floater and Sinker are bytes 8 to 10), a storm world's names (Grazer for Gatherer) in the weather
    fixtures, and the N1 gate. `3d` and `2d` are real situations: no verdict row for a version-3 world, so no
    label is valid. `3d_n70` carries a made-up gate with a row and some passing labels, to exercise the
    decoder (the digest and the key are not any world's)."""
    storm = kind == "3d"
    arch = ["", *("Grazer" if storm and a == "Gatherer" else a for a in ARCH)]
    gate = {
        "rules_digest": "d" * 64,
        "world_key": "0" * 64,
        "key_found": kind == "3d_n70",
        "archetypes": [2, 3] if kind == "3d_n70" else [],
        "intents": [10] if kind == "3d_n70" else [],
    }
    return sr.labels_block(
        intent_version=3,
        arch=arch,
        intent=[
            "",
            "Budding",
            "Hunting",
            "Fleeing",
            "Defending",
            "Caring",
            "Tethering",
            "Signalling",
            "Feeding",
            "Roaming",
            "Resting",
        ],
        gate=gate,
    )


G3A_TAGS = tuple(
    t for t in sr.TAGS if t not in ("ALRT", "RPRT", "PLAN")
)  # the ten records of the first fixtures

ALERT_CLASSES = {
    "chapter": {"imp": 3, "pause": True, "label": "Chapter complete", "calibrated": True, "surface": "alert"},
    "pop_fall": {
        "imp": 3,
        "pause": True,
        "label": "Population fall",
        "calibrated": False,
        "surface": "alert",
    },
    "storm_named": {"imp": 2, "pause": False, "label": "Storm named", "calibrated": True, "surface": "alert"},
    "lineage_formed": {
        "imp": 2,
        "pause": False,
        "label": "Lineage formed",
        "calibrated": False,
        "surface": "log",
    },
    "pop_surge": {
        "imp": 1,
        "pause": False,
        "label": "Population surge",
        "calibrated": False,
        "surface": "alert",
    },
    "storm_strong": {
        "imp": 1,
        "pause": False,
        "label": "Storm strengthening",
        "calibrated": True,
        "surface": "log",
    },
}
POWERS = {
    "food": {
        "action": "regen_pulse",
        "label": "Food",
        "cost": 1,
        "cooldown": 3000,
        "scope": "global",
        "duration": [3000, 12000],
        "params": {"mult": [[0.6, 0.9], [1.15, 1.5]]},
    },
    "strike": {
        "action": "catastrophe",
        "label": "Strike",
        "cost": 1,
        "cooldown": 3000,
        "scope": "column",
        "duration": None,
        "params": {"radius": [6.25, 25.0], "kill_p": [0.3, 0.6]},
    },
}


def _header(kind: str) -> bytes:
    three = kind != "2d"
    n = 70 if kind == "3d_n70" else 64
    weather = sr.weather_block(enabled=kind == "3d", dc_ref=0.5 if kind == "3d" else None)
    records = [t for t in G3A_TAGS if kind == "3d" or t not in ("MAP1", "STRM")]
    if kind == "3d_n70":
        records = ["FRMA", "IDNT", "INTN", "CLCK"]
    g3b: dict[str, Any] = {}
    if kind == "3d_g3b":
        records = ["FRMA", "IDNT", "EVNT", "ENVS", "CLCK", "ALRT", "RPRT", "PLAN"]
        g3b = {
            "alerts": sr.alerts_block(ALERT_CLASSES),
            "autopause": sr.autopause_block(
                chapter=True, alert={name: bool(c["pause"]) for name, c in ALERT_CLASSES.items()}
            ),
            "powers": sr.powers_block(POWERS),
            "game_specs": "game-v0",
            "saves": sr.saves_block(mode="newest", keep=3),  # §34.10: the player switched recording off
        }
    legacy = sr.legacy_header(
        dims=3 if three else 2,
        size=48.0,
        height=24.0 if three else 0.0,
        n=n,
        bond_slots=4,
        radius=1.0,
        field_cell=4.0,
        fg=12,
        fz=6 if three else 1,
        nch=3,
        step={"3d_n70": 10, "3d_g3b": 12000}.get(kind, 100),
        run=None,
        config_hash=CONFIG_HASH,
        source="preset",
    )
    play = sr.header_play(
        records=records,
        weather=weather,
        labels=_labels(kind),
        energy_cap=16384,
        planet={"source": "runs/example:500000", "houses": "exact"} if kind in ("3d", "3d_g3b") else None,
        **g3b,
    )
    return sr.play_header_bytes(legacy, play)


def _build_main(kind: str) -> bytes:
    """The `3d` and `2d` streams."""
    three = kind == "3d"
    dims = 3 if three else 2
    uid_a = {r[0]: UID0 + 7 * r[0] for r in _ROWS_A}
    uid_b = {**uid_a, 4: UID0 + 100, 5: UID0 + 101}
    seq = sr.Sequencer()
    parts = [_header(kind)]

    def frma(step: int, rows, uids, bonds, color_index: int, span: int, diag: bool = False) -> bytes:
        alive, pos, orient, rgb, uid = _arrays(64, rows, uids, dims)
        return sr.encode_frma(
            step=step,
            seq=seq.take("FRMA"),
            alive=alive,
            pos=pos[:, :dims],
            orient=orient,
            rgb=rgb,
            uid=uid,
            bonds=np.array(bonds, np.int64),
            color_index=color_index,
            span=span,
            diag_error=diag,
        )

    parts.append(frma(100, _ROWS_A, uid_a, [(0, 1), (2, 10), (2, 3)], 0, 0))
    rows_a = [r[0] for r in _ROWS_A]
    uids = np.array([uid_a[r] for r in rows_a], np.int64)
    parts.append(
        sr.encode_idnt(
            step=100,
            seq=seq.take("IDNT"),
            snapshot_step=96,
            uid=uids,
            clade=np.array([3, 3, 0, 5, 9, 1, 2], np.int32),
            house=np.array([0, 0, 1, 1, 254, 253, 255], np.uint8),
            kin=np.array([0, 0, 1, 1, 2, 2, 255], np.uint8),
            colony=np.array([255, 255, 0, 0, 255, 255, 254], np.uint8),
            flags=np.array([0, 0, 0, 0, 0, sr.FLAG_BORN_SINCE, 0], np.uint8),
            tables=_tables(),
        )
    )
    parts.append(
        sr.encode_intn(
            step=100,
            seq=seq.take("INTN"),
            labels_step=96,
            intent_version=3,
            uid=uids,
            arch_primary=np.array([1, 1, 2, 7, 8, 0, 3], np.uint8),
            arch_secondary=np.array([0, 2, 0, 0, 1, 0, 0], np.uint8),
            intent=np.array([9, 10, 2, 8, 6, 0, 1], np.uint8),
            flags=np.array([1, 1, 1, 1, 1, 0, 3], np.uint8),
        )
    )
    owner, purity = _terr_planes()
    parts.append(
        sr.encode_terr(
            step=100, seq=seq.take("TERR"), snapshot_step=96, ids=[17, 21, 22], owner=owner, purity=purity
        )
    )
    parts.append(sr.encode_envs(step=100, seq=seq.take("ENVS"), obj=_pools()))
    if three:
        parts.append(sr.encode_map1(step=100, seq=seq.take("MAP1"), planes=_map_planes(), dc_ref=0.5))
        parts.append(sr.encode_strm(step=100, seq=seq.take("STRM"), tick=25, obj=_storm_obj(100)))
    backlog, live = _events()
    parts.append(sr.encode_evnt(step=100, seq=seq.take("EVNT"), events=backlog, backlog=True))
    parts.append(
        sr.encode_clck(
            step=100,
            seq=seq.take("CLCK"),
            rate=1.0,
            achieved=0.984375,
            step_ms=15.5,
            frame_ms=3.25,
            late_steps=0,
            save_stall_steps=0,
            reason=0,
            saving=False,
            fe=2,
            chapter=0,
            in_chapter=100,
            chapter_len=6000,
        )
    )
    parts.append(sr.encode_unknown("ZZZZ", 100, 0, b"later version"[:12]))
    parts.append(frma(104, _ROWS_B, uid_b, [(0, 1), (2, 3), (4, 5)], 3, 4, diag=True))
    parts.append(sr.encode_evnt(step=104, seq=seq.take("EVNT"), events=live))
    parts.append(
        sr.encode_focs(
            step=104,
            seq=seq.take("FOCS"),
            uid=UID0 + 100,
            row=4,
            flags=sr.FOCS_ALIVE | sr.FOCS_HANDOFF | sr.FOCS_FOLLOWED,
            pos=(20.5, 8.5, 1.0 if three else 0.0),
            vel=(0.25, -0.125, 0.0),
            energy=4096,
            age=4,
            generation=12,
            parent_uid=UID0 + 28,
            n_bonds=1,
            house=0,
            kin=2,
            colony=255,
            arch_primary=1,
            arch_secondary=0,
            intent=9,
            label_flags=1,
        )
    )
    parts.append(
        sr.encode_clck(
            step=104,
            seq=seq.take("CLCK"),
            rate=-1.0,
            achieved=55.5,
            step_ms=17.75,
            frame_ms=3.5,
            late_steps=3,
            save_stall_steps=21,
            reason=1,
            saving=True,
            fe=2,
            chapter=0,
            in_chapter=104,
            chapter_len=6000,
            flags=0,
        )
    )
    return sr.flatten(parts)


def _g3b_events() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def ev(step, kind, fam, a, b, x, y, imp, text):
        return {
            "id": step * 1000 + {"kin": 0, "director": 300}[fam] + kind % 7,
            "step": step,
            "kind": kind,
            "fam": fam,
            "a": a,
            "b": b,
            "x": x,
            "y": y,
            "z": None,
            "imp": imp,
            "text": text,
        }

    backlog = [
        ev(11500, 18, "kin", 17, 0, 6.0, 24.0, 1, "Aldric has held together since step 11000: 360 members."),
        ev(
            11750,
            19,
            "kin",
            21,
            17,
            11.0,
            10.75,
            2,
            "Brennar merged into Aldric at step 11500; it lasted 750 steps.",
        ),
    ]
    live = [
        ev(12000, 20, "director", 0, 0, 24.0, 12.0, 1, "The player set regen to 1.3 for 6000 steps."),
        ev(12001, 21, "director", 0, 0, None, None, 0, "The player's regen setting ended."),
    ]
    return backlog, live


def _alert(index: int, first: int, **kw: Any) -> dict[str, Any]:
    a = {
        "id": first * 1000 + 600 + index,
        "version": 1,
        "step": first,
        "class": "pop_surge",
        "key": "world",
        "state": "raised",
        "surface": "alert",
        "imp": 1,
        "title": "Population surge",
        "text": "The population rose from 40,000 to 44,500 in 250 steps.",
        "x": None,
        "y": None,
        "z": None,
        "ref": None,
        "pause": False,
        "expires": first + 3000,
        "watched": False,
        "calibrated": False,
    }
    return {**a, **kw}


def _report(chapter: int, mode: int, final: bool, **kw: Any) -> dict[str, Any]:
    entities = {
        "K17": {"kind": "kin", "id": 17, "name": "Aldric", "x": 6.0, "y": 24.0},
        "S7": {"kind": "storm", "id": 7, "name": "Merrow", "x": 4.5, "y": 20.25},
    }
    r = {
        "chapter": chapter,
        "n_abs": chapter,
        "a": 6000 * (chapter - 1),
        "b": 6000 * chapter,
        "mode": mode,
        "final": final,
        "withheld": [],
        "headline": {"id": "HS", "text": "Storm Merrow reached named strength", "refs": ["S7"]},
        "lead": {"id": "L", "text": "The population went from 40,000 to 44,500 (+11.2%).", "refs": []},
        "body": [
            {
                "id": "TO",
                "text": "270,000 were born and 268,000 died: a turnover of 6.0 times the population.",
                "origin": "sheet",
                "refs": [],
            },
            {
                "id": "K1",
                "text": "Aldric has held together since step 11000: 360 members.",
                "origin": "sheet",
                "refs": ["K17"],
            },
        ],
        "interventions": [
            {
                "id": "I1",
                "text": "At step 12000 the player set food regrowth to 1.3 times base for 6000 steps.",
                "refs": [],
            }
        ],
        "ticker": [{"step": 11750, "text": "Storm Merrow reached named strength."}],
        "since": {
            "influence": {"have": 4, "cap": 6},
            "alerts": {"pop_surge": 1},
            "ending": [{"action": "regen_pulse", "end": 18000}],
        },
        "entities": entities,
        "versions": {"report": 2, "template": 2, "checker": 1, "lexicon": 3},
        "facts_digest": "f" * 64,
        "latency_s": None,
        "fallback": None,
    }
    return {**r, **kw}


def _plan(status: str, **kw: Any) -> dict[str, Any]:
    p = {
        "id": 3,
        "kind": "power",
        "status": status,
        "power": "food",
        "action": "regen_pulse",
        "raw": {"mult": 1.3, "steps": 6000},
        "params": {"mult": 1.3},
        "clamped": [],
        "duration": 6000,
        "step_applied": None,
        "ends": None,
        "cost": 1,
        "influence": {"have": 5, "cap": 6, "window": 18000, "free_at": [30000]},
        "ready": None,
        "holds": "regen",
        "text": "Food x1.3 for 6000 steps (6.9 min at 1x), cost 1, influence 5 of 6 after.",
        "caution": ["holds the regen lever until step 18000"],
        "precedent": {"step": None, "action": "regen_pulse", "params": {}, "rows": []},
        "since": None,
        "refused": None,
    }
    return {**p, **kw}


def _build_g3b() -> bytes:
    """The G3b scene: a chapter end at step 12000 of a 3D world, then the first step of the next chapter."""
    seq = sr.Sequencer()
    uid_a = {r[0]: UID0 + 7 * r[0] for r in _ROWS_A}
    parts = [_header("3d_g3b")]
    alive, pos, orient, rgb, uid = _arrays(64, _ROWS_A, uid_a, 3)
    parts.append(
        sr.encode_frma(
            step=12000,
            seq=seq.take("FRMA"),
            alive=alive,
            pos=pos,
            orient=orient,
            rgb=rgb,
            uid=uid,
            bonds=np.array([(0, 1), (2, 3)], np.int64),
            color_index=0,
        )
    )
    tables = _tables()
    tables = {**tables, "kin": [{**k, "thread": 4 + i} for i, k in enumerate(tables["kin"])]}
    rows = [r[0] for r in _ROWS_A]
    parts.append(
        sr.encode_idnt(
            step=12000,
            seq=seq.take("IDNT"),
            snapshot_step=11750,
            uid=np.array([uid_a[r] for r in rows], np.int64),
            clade=np.array([3, 3, 0, 5, 9, 1, 2], np.int32),
            house=np.array([0, 0, 1, 1, 254, 253, 255], np.uint8),
            kin=np.array([0, 0, 1, 1, 2, 2, 255], np.uint8),
            colony=np.array([255, 255, 0, 0, 255, 255, 254], np.uint8),
            flags=np.zeros(7, np.uint8),
            tables=tables,
        )
    )
    envs = {
        **_pools(),
        "effects": [
            {
                "action": "regen_pulse",
                "params": {"mult": 1.3},
                "start": 12000,
                "end": 18000,
                "source": "player",
            },
            {
                "action": "bloom",
                "params": {"x": 24.0, "y": 12.0},
                "start": 9000,
                "end": None,
                "source": "rule",
            },
        ],
        "influence": {"have": 4, "cap": 6, "window": 18000, "free_at": [13000, 15500]},
        "patches": [
            {"i": 0, "x": 24.0, "y": 12.0, "gain": 1.5, "sigma": 6.0},
            {"i": 1, "x": 8.5, "y": 40.0, "gain": 1.25, "sigma": 4.5},
        ],
        "powers": {
            "food": {"ready": None, "busy": None},
            "strike": {"ready": 15000, "busy": "cooldown until step 15000"},
        },
    }
    parts.append(sr.encode_envs(step=12000, seq=seq.take("ENVS"), obj=envs))
    backlog, live = _g3b_events()
    parts.append(sr.encode_evnt(step=12000, seq=seq.take("EVNT"), events=backlog, backlog=True))
    names = set(ALERT_CLASSES)
    parts.append(
        sr.encode_alrt(
            step=12000,
            seq=seq.take("ALRT"),
            backlog=True,
            registry=names,
            alerts=[
                _alert(
                    0,
                    11250,
                    **{
                        "class": "pop_fall",
                        "imp": 3,
                        "state": "raised",
                        "title": "Population fell",
                        "text": "The population fell from 42,873 to 37,310 in 250 steps.",
                        "pause": True,
                        "expires": None,
                    },
                )
            ],
        )
    )
    parts.append(sr.encode_rprt(step=12000, seq=seq.take("RPRT"), backlog=True, obj=_report(1, 0, True)))
    parts.append(
        sr.encode_clck(
            step=12000,
            seq=seq.take("CLCK"),
            rate=1.0,
            achieved=0.984375,
            step_ms=15.5,
            frame_ms=3.25,
            late_steps=0,
            save_stall_steps=13,
            reason=3,
            saving=True,
            fe=2,
            chapter=2,
            in_chapter=6000,
            chapter_len=6000,
            flags=sr.CLCK_FLAG_CHAPTER_PAUSE | sr.CLCK_FLAG_ALERT_PAUSE,
        )
    )
    # the live records of the chapter end
    parts.append(
        sr.encode_alrt(
            step=12000,
            seq=seq.take("ALRT"),
            registry=names,
            alerts=[
                _alert(
                    0,
                    12000,
                    **{
                        "class": "chapter",
                        "key": "2",
                        "imp": 3,
                        "title": "Chapter 2 complete",
                        "text": "Chapter 2 ended at step 12000.",
                        "pause": True,
                        "expires": None,
                        "ref": {"kind": "world", "id": 0},
                    },
                ),
                _alert(
                    1,
                    12000,
                    **{
                        "class": "storm_named",
                        "key": "S7",
                        "imp": 2,
                        "title": "Storm Merrow",
                        "text": "Storm Merrow reached named strength.",
                        "x": 4.5,
                        "y": 20.25,
                        "z": 3.5,
                        "ref": {"kind": "storm", "id": 7},
                    },
                ),
                _alert(
                    2,
                    12000,
                    **{
                        "class": "lineage_formed",
                        "key": "K17",
                        "surface": "log",
                        "imp": 0,
                        "title": "Aldric formed",
                        "text": "Aldric has held together since step 11000.",
                        "x": 6.0,
                        "y": 24.0,
                        "z": 6.5,
                        "ref": {"kind": "kin", "id": 17},
                        "watched": True,
                    },
                ),
                _alert(
                    3,
                    11500,
                    step=12000,
                    **{
                        "class": "pop_surge",
                        "version": 2,
                        "state": "updated",
                        "imp": 1,
                        "text": "The population rose from 40,000 to 46,000 in 500 steps.",
                        "ref": {"kind": "power", "id": 3},
                        "calibrated": True,
                    },
                ),
                _alert(
                    0,
                    11250,
                    step=12000,
                    **{
                        "class": "pop_fall",
                        "version": 3,
                        "state": "cleared",
                        "imp": 3,
                        "pause": True,
                        "title": "Population fell",
                        "text": "The population is back above 0.9 of its maximum.",
                        "ref": {"kind": "organism", "id": UID0 + 7},
                        "expires": None,
                    },
                ),
                _alert(
                    5,
                    11750,
                    step=12000,
                    **{
                        "class": "storm_strong",
                        "key": "S7",
                        "surface": "log",
                        "imp": 0,
                        "title": "Storm strengthening",
                        "text": "Storm Merrow reached 0.7 times the naming threshold.",
                        "ref": {"kind": "house", "id": 5000000000},
                    },
                ),
            ],
        )
    )
    parts.append(sr.encode_rprt(step=12000, seq=seq.take("RPRT"), obj=_report(2, 0, True)))
    parts.append(
        sr.encode_rprt(
            step=12000,
            seq=seq.take("RPRT"),
            obj=_report(2, 1, False, fallback="ordering", latency_s=None),
        )
    )
    parts.append(
        sr.encode_rprt(
            step=12000,
            seq=seq.take("RPRT"),
            obj=_report(
                2,
                2,
                True,
                withheld=["the cut 11000 is not the step of a metrics row"],
                headline=None,
                lead=None,
                body=[],
                interventions=[],
                ticker=[],
                entities={},
            ),
        )
    )
    parts.append(sr.encode_plan(step=12000, seq=seq.take("PLAN"), reply=True, obj=_plan("preview")))
    parts.append(sr.encode_plan(step=12000, seq=seq.take("PLAN"), obj=_plan("queued", ready=None)))
    parts.append(
        sr.encode_plan(
            step=12000,
            seq=seq.take("PLAN"),
            obj=_plan("applied", step_applied=12000, ends=18000, ready=21000),
        )
    )
    parts.append(
        sr.encode_plan(
            step=12000,
            seq=seq.take("PLAN"),
            obj=_plan(
                "since",
                step_applied=12000,
                ends=18000,
                since={
                    "from": 12000,
                    "rows": [
                        {
                            "step": 12250,
                            "alive": 44100,
                            "starved": 5600,
                            "killed": 5200,
                            "born": 11000,
                            "storm": 3,
                            "catastrophe": 0,
                        }
                    ],
                },
            ),
        )
    )
    parts.append(
        sr.encode_plan(
            step=12000,
            seq=seq.take("PLAN"),
            reply=True,
            obj=_plan(
                "refused",
                id=4,
                power="strike",
                action="catastrophe",
                params={},
                clamped=[],
                duration=None,
                holds=None,
                cost=1,
                refused="Strike is cooling down until step 15000.",
                caution=[],
                precedent=None,
            ),
        )
    )
    parts.append(
        sr.encode_plan(
            step=12000,
            seq=seq.take("PLAN"),
            obj=_plan(
                "cancelled", step_applied=12000, ends=12001, text="Food ended early at the player's request."
            ),
        )
    )
    parts.append(
        sr.encode_plan(
            step=12000,
            seq=seq.take("PLAN"),
            reply=True,
            obj=_plan(
                "refused",
                id=9,
                kind="intent",
                power=None,
                action=None,
                raw={},
                params={},
                clamped=[],
                duration=None,
                cost=0,
                influence=None,
                holds=None,
                text="intent is not available in this build",
                caution=[],
                precedent=None,
                refused="intent is not available in this build",
            ),
        )
    )
    parts.append(sr.encode_evnt(step=12001, seq=seq.take("EVNT"), events=live))
    parts.append(sr.encode_unknown("ALRT", 12001, seq.take("ALRT"), b"a later version"[:12], version=2))
    parts.append(
        sr.encode_clck(
            step=12001,
            seq=seq.take("CLCK"),
            rate=0.0,
            achieved=0.0,
            step_ms=15.5,
            frame_ms=3.25,
            late_steps=0,
            save_stall_steps=13,
            reason=4,
            saving=False,
            fe=2,
            chapter=3,
            in_chapter=1,
            chapter_len=6000,
            flags=sr.CLCK_FLAG_CHAPTER_PAUSE | sr.CLCK_FLAG_SAVES_NEWEST,
        )
    )
    return sr.flatten(parts)


def _build_n70() -> bytes:
    """N = 70: three bitmap words, the last with bits 64 to 69."""
    n = 70
    seq = sr.Sequencer()
    parts = [_header("3d_n70")]

    def frame(step: int, rows: list[int], bonds, span: int) -> bytes:
        alive = np.isin(np.arange(n), rows)
        r = np.arange(n, dtype=np.float32)
        return sr.encode_frma(
            step=step,
            seq=seq.take("FRMA"),
            alive=alive,
            pos=np.stack([r, 2 * r, 3 * r + 0.5], axis=1),
            orient=np.stack([r * 0.0625, -r * 0.03125], axis=1),
            rgb=(0x010203 + np.arange(n)).astype(np.uint32),
            uid=UID0 + np.arange(n, dtype=np.int64) * 3,
            bonds=np.array(bonds, np.int64),
            color_index=0,
            span=span,
        )

    rows = [0, 31, 32, 63, 64, 69]
    parts.append(frame(10, rows, [(31, 32), (63, 64), (0, 69)], 0))
    uid = UID0 + np.array(rows, np.int64) * 3
    parts.append(
        sr.encode_idnt(
            step=10,
            seq=seq.take("IDNT"),
            snapshot_step=10,
            uid=uid,
            clade=np.array([1, 2, 3, 4, 5, 6], np.int32),
            house=np.full(6, 255, np.uint8),
            kin=np.full(6, 255, np.uint8),
            colony=np.full(6, 255, np.uint8),
            flags=np.zeros(6, np.uint8),
            tables={
                "houses": [],
                "exact": False,
                "origin": 10,
                "kin": [],
                "colonies": [],
                "unknown_immigrant": 0,
            },
        )
    )
    parts.append(
        sr.encode_intn(
            step=10,
            seq=seq.take("INTN"),
            labels_step=0,
            intent_version=3,
            uid=uid,
            arch_primary=np.zeros(6, np.uint8),
            arch_secondary=np.zeros(6, np.uint8),
            intent=np.zeros(6, np.uint8),
            flags=np.zeros(6, np.uint8),
        )
    )
    parts.append(frame(11, list(range(n)), [], 1))
    parts.append(
        sr.encode_clck(
            step=11,
            seq=seq.take("CLCK"),
            rate=0.0,
            achieved=0.0,
            step_ms=0.0,
            frame_ms=0.0,
            late_steps=0,
            save_stall_steps=0,
            reason=1,
            saving=False,
            fe=1,
            chapter=0,
            in_chapter=11,
            chapter_len=6000,
        )
    )
    return sr.flatten(parts)


def build_bytes(kind: str) -> bytes:
    if kind == "3d_g3b":
        return _build_g3b()
    return _build_n70() if kind == "3d_n70" else _build_main(kind)


# --- sidecar --------------------------------------------------------------------------------------------


def _record_entry(index: int, r: Rec) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "index": index,
        "tag": r.tag,
        "offset": r.offset,
        "length": r.length,
        "step": r.step,
        "seq": r.seq,
        "version": r.version,
        "flags": r.flags,
    }
    a = r.a
    f = dict(r.f)
    if r.tag == "FRMA":
        f["rows"] = [int(v) for v in a["rows"]]
        f["bonds"] = a["bonds"].tolist()
        f["agents"] = [
            {
                "index": i,
                "row": int(a["rows"][i]),
                "pos": [float(v) for v in a["pos"][i]],
                "yaw": float(a["orient"][i, 0]),
                "pitch": float(a["orient"][i, 1]),
                "rgb": int(a["rgb"][i]),
                "uid": int(a["uid"][i]),
            }
            for i in sorted({0, len(a["rows"]) // 2, len(a["rows"]) - 1})
            if len(a["rows"])
        ]
    elif r.tag == "IDNT":
        f["rows"] = [
            {
                "uid": int(a["uid"][i]),
                "clade": int(a["clade"][i]),
                "house": int(a["house"][i]),
                "kin": int(a["kin"][i]),
                "colony": int(a["colony"][i]),
                "flags": int(a["flags"][i]),
            }
            for i in range(len(a["uid"]))
        ]
    elif r.tag == "INTN":
        f["rows"] = [
            {
                "uid": int(a["uid"][i]),
                "arch_primary": int(a["arch_primary"][i]),
                "arch_secondary": int(a["arch_secondary"][i]),
                "intent": int(a["intent"][i]),
                "flags": int(a["flags"][i]),
            }
            for i in range(len(a["uid"]))
        ]
    elif r.tag == "TERR":
        f["owner_sum"] = int(a["owner"].astype(np.int64).sum())
        f["purity_sum"] = int(a["purity"].astype(np.int64).sum())
        f["owner_row3"] = a["owner"][3].tolist()
    elif r.tag == "MAP1":
        f["samples"] = {
            "P": [int(v) for v in a["P"][[0, 1, 17, 159], [0, 5, 100, 159]]],
            "Dc": [int(v) for v in a["Dc"][[0, 1, 17, 159], [0, 5, 100, 159]]],
            "zeta": [int(v) for v in a["zeta"][[0, 1, 17, 159], [0, 5, 100, 159]]],
            "Ts": [int(v) for v in a["Ts"][[0, 1, 17, 159], [0, 5, 100, 159]]],
            "winds": [int(a["winds"][k, 3, 7]) for k in range(4)],
        }
    entry["fields"] = f
    entry["sha256"] = {k: hashlib.sha256(v).hexdigest() for k, v in r.sections.items()}
    return entry


def sidecar(kind: str, data: bytes) -> dict[str, Any]:
    d = decode(data)
    return {
        "about": "Expected decoding of the stream in the .bin file next to this one (DESIGN section 33.1). "
        "Regenerate with `uv run python tests/play_fixtures.py --write`.",
        "kind": kind,
        "bytes": len(data),
        "header": d.header,
        "skipped": [{"tag": t, "offset": o, "why": w} for t, o, w in d.skipped],
        "records": [_record_entry(i, r) for i, r in enumerate(d.records)],
    }


def sidecar_text(obj: dict[str, Any]) -> str:
    return json.dumps(obj, indent=2, allow_nan=False) + "\n"


def main(argv: list[str]) -> int:
    write = "--write" in argv
    for kind in KINDS:
        data = build_bytes(kind)
        bin_path, json_path = paths(kind)
        print(f"{bin_path.name}: {len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()[:16]}")
        if write:
            DATA.mkdir(exist_ok=True)
            bin_path.write_bytes(data)
            json_path.write_text(sidecar_text(sidecar(kind, data)), encoding="utf-8")
    if not write:
        print("(dry run: pass --write to update tests/data)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
