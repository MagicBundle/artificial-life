"""The play-loop stream records (``alife.streamrec``, DESIGN §33.1) against the independent decoder.

No engine and no device: the encoders are pure functions of numpy arrays, and ``tests/play_decode.py`` is
written from the text of §33.1. The committed fixtures (``tests/data/play_tiny*.bin``) are decoded and checked
against their sidecars, and rebuilt byte for byte, on every device.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

import play_fixtures as pf
from alife import streamrec as sr
from play_decode import decode
from stream_decode import HEADER_KEYS, MAGIC, decode_header

SRC = Path(sr.__file__)


# --- helpers --------------------------------------------------------------------------------------------


def _header(n: int, records: tuple[str, ...] = sr.TAGS) -> bytes:
    legacy = sr.legacy_header(
        dims=3,
        size=48.0,
        height=24.0,
        n=n,
        bond_slots=4,
        radius=1.0,
        field_cell=4.0,
        fg=12,
        fz=6,
        nch=3,
        step=0,
        run=None,
        config_hash="h",
        source="preset",
    )
    play = sr.header_play(
        records=records,
        weather=sr.weather_block(enabled=True, dc_ref=0.5),
        labels={
            "intent_version": 3,
            "arch": ["", *(f"A{i}" for i in range(10))],
            "intent": ["", *(f"I{i}" for i in range(10))],
        },
        energy_cap=100,
    )
    return sr.play_header_bytes(legacy, play)


def _frma(n_cap: int, alive: np.ndarray, bonds=(), step: int = 5, seq: int = 0, **kw) -> bytes:
    r = np.arange(n_cap, dtype=np.float32)
    return sr.encode_frma(
        step=step,
        seq=seq,
        alive=alive,
        pos=np.stack([r, 2 * r, 3 * r], axis=1),
        orient=np.stack([r * 0.0625, -r * 0.125], axis=1),
        rgb=(0x10000 + np.arange(n_cap)).astype(np.uint32),
        uid=(1 << 40) + np.arange(n_cap, dtype=np.int64),
        bonds=np.array(bonds, np.int64).reshape(-1, 2),
        color_index=2,
        **kw,
    )


def _one(n_cap: int, record: bytes):
    d = decode(_header(n_cap) + record)
    assert len(d.records) == 1 and not d.skipped
    return d.records[0]


# --- fixtures -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", pf.KINDS)
def test_fixtures_match_the_code(kind: str) -> None:
    """The committed files are what the encoders write now (no device is involved)."""
    data = pf.build_bytes(kind)
    bin_path, json_path = pf.paths(kind)
    assert bin_path.read_bytes() == data, f"{bin_path.name} is stale: run tests/play_fixtures.py --write"
    assert json_path.read_text(encoding="utf-8") == pf.sidecar_text(pf.sidecar(kind, data))


@pytest.mark.parametrize("kind", pf.KINDS)
def test_fixture_decodes_to_its_sidecar(kind: str) -> None:
    bin_path, json_path = pf.paths(kind)
    data = bin_path.read_bytes()
    side = json.loads(json_path.read_text(encoding="utf-8"))
    d = decode(data)
    assert side["bytes"] == len(data) and side["kind"] == kind
    assert d.header == side["header"]
    assert [(t, w) for t, _, w in d.skipped] == [(s["tag"], s["why"]) for s in side["skipped"]]
    assert len(d.records) == len(side["records"])
    for rec, exp in zip(d.records, side["records"], strict=True):
        assert (rec.tag, rec.offset, rec.length, rec.step, rec.seq, rec.version, rec.flags) == (
            exp["tag"],
            exp["offset"],
            exp["length"],
            exp["step"],
            exp["seq"],
            exp["version"],
            exp["flags"],
        )
        for name, raw in rec.sections.items():
            assert hashlib.sha256(raw).hexdigest() == exp["sha256"][name], f"{rec.tag} {name}"
        assert set(rec.sections) == set(exp["sha256"])


def test_fixture_contents_cover_the_cases_the_spec_names() -> None:
    d = decode(pf.paths("3d")[0].read_bytes())
    by_tag = {}
    for r in d.records:
        by_tag.setdefault(r.tag, []).append(r)
    assert sorted(by_tag) == sorted(pf.G3A_TAGS), "the three first fixtures have the ten records of G3a"
    assert [r.step for r in by_tag["FRMA"]] == [100, 104] and [r.seq for r in by_tag["FRMA"]] == [0, 1]
    a, b = by_tag["FRMA"]
    assert a.a["rows"].tolist() == [0, 1, 2, 3, 4, 10, 21]  # gaps between alive rows
    assert b.a["rows"].tolist() == [0, 1, 2, 3, 4, 5, 21]  # row 10 is gone, row 4 is reused with a new uid
    assert a.a["uid"][4] != b.a["uid"][4] and a.a["uid"].max() > 2**32
    assert b.flags & sr.FLAG_DIAG and not a.flags & sr.FLAG_DIAG
    assert a.a["bonds"].tolist() == [
        [0, 1],
        [2, 5],
        [2, 3],
    ]  # row 2's bonds out of partner order; (0,1) across the seam
    assert abs(a.a["pos"][0, 0] - a.a["pos"][1, 0]) > 40  # that bond spans the seam of L = 48
    assert [x[0] for x in d.skipped] == ["ZZZZ"]
    backlog, live = by_tag["EVNT"]
    assert backlog.f["backlog"] and not live.f["backlog"]
    idnt = by_tag["IDNT"][0]
    assert (
        set(idnt.a["house"].tolist()) >= {253, 254, 255} and idnt.a["flags"].tolist()[5] == sr.FLAG_BORN_SINCE
    )
    assert idnt.f["snapshot_step"] < idnt.step  # an idnt between snapshots
    assert by_tag["FOCS"][0].f["flags"] & sr.FOCS_HANDOFF
    n70 = decode(pf.paths("3d_n70")[0].read_bytes())
    frames = [r for r in n70.records if r.tag == "FRMA"]
    assert n70.header["N"] == 70 and len(frames[0].sections["bitmap"]) == 12  # three words
    assert frames[0].a["rows"].tolist() == [0, 31, 32, 63, 64, 69] and frames[1].f["n"] == 70
    two = decode(pf.paths("2d")[0].read_bytes())
    assert {r.tag for r in two.records}.isdisjoint({"MAP1", "STRM"})
    assert two.header["play"]["records"] == [t for t in pf.G3A_TAGS if t not in ("MAP1", "STRM")]
    assert all(
        (r.a["pos"][:, 2] == 0).all() and (r.a["orient"][:, 1] == 0).all()
        for r in two.records
        if r.tag == "FRMA"
    )


# --- the header -----------------------------------------------------------------------------------------


def test_play_header_keeps_the_legacy_keys_and_adds_play() -> None:
    header, end = decode_header(_header(64))
    assert set(HEADER_KEYS) <= set(header) and set(header) - set(HEADER_KEYS) == {"play"}
    assert header["mode"] == "simulate" and header["source"] == "preset" and header["every"] == 1
    assert header["fields"] is False and header["color"] == "tag" and header["field_z"] == 0
    assert header["color_modes"][:3] == ["tag", "energy", "clade"]  # the indices equal `stream`'s
    assert header["color_modes"] == ["tag", "energy", "clade", "house", "kin", "archetype", "intent"]
    assert (
        header["FG"] * header["field_cell"] == header["L"]
        and header["FZ"] * header["field_cell"] == header["height"]
    )
    assert header["play"]["versions"] == dict.fromkeys(sr.TAGS, 1) and end % 4 == 0
    # the same preamble as `alife stream` writes (stream.header_bytes), byte for byte
    from alife import stream

    assert sr.header_bytes({"a": 1, "b": [1.5, None]}) == stream.header_bytes({"a": 1, "b": [1.5, None]})
    assert sr.MAGIC == stream.MAGIC == MAGIC
    with pytest.raises(sr.RecordError):
        sr.header_play(records=["FRMA", "NOPE"], weather={}, labels={}, energy_cap=1)
    with pytest.raises(sr.RecordError):
        sr.weather_block(enabled=True, dc_ref=None)
    with pytest.raises(sr.RecordError, match="source"):
        sr.legacy_header(
            dims=2,
            size=1.0,
            height=0.0,
            n=1,
            bond_slots=1,
            radius=1.0,
            field_cell=1.0,
            fg=1,
            fz=1,
            nch=1,
            step=0,
            run=None,
            config_hash="h",
            source="file",
        )


# --- FRMA -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n_cap", [1, 31, 32, 33, 63, 64, 65, 70, 96, 1000])
def test_frma_round_trip_and_length(n_cap: int) -> None:
    rng = np.random.default_rng(n_cap)
    for density in (0.0, 0.3, 1.0):
        alive = rng.random(n_cap) < density if density < 1 else np.ones(n_cap, bool)
        rows = np.flatnonzero(alive)
        pairs = [(int(a), int(b)) for a, b in zip(rows[:-1:2], rows[1::2], strict=False)]
        blob = _frma(n_cap, alive, bonds=pairs, span=3)
        rec = _one(n_cap, blob)
        n = len(rows)
        assert (
            rec.length
            == sr.frma_length(n_cap, n, len(pairs))
            == 36 + 4 * math.ceil(n_cap / 32) + 28 * n + 8 * len(pairs)
        )
        assert len(blob) == 8 + rec.length and rec.length % 4 == 0
        assert rec.f == {"n": n, "n_bonds": len(pairs), "color_index": 2, "span": 3, "N": n_cap}
        np.testing.assert_array_equal(rec.a["rows"], rows)
        np.testing.assert_array_equal(rec.a["uid"], (1 << 40) + rows)
        np.testing.assert_array_equal(rec.a["rgb"], 0x10000 + rows)
        np.testing.assert_array_equal(
            rec.a["pos"], np.stack([rows, 2 * rows, 3 * rows], axis=1).astype(np.float32)
        )
        np.testing.assert_array_equal(rec.a["orient"][:, 0], (rows * 0.0625).astype(np.float16))
        # a bond lists indices into the listed rows: rows[index] is the row pair that was given
        got = [(int(rows[i]), int(rows[j])) for i, j in rec.a["bonds"]]
        assert got == pairs


def test_frma_bitmap_is_lsb_first_words_and_popcount_is_n() -> None:
    alive = np.zeros(70, bool)
    alive[[0, 1, 31, 32, 64, 69]] = True
    rec = _one(70, _frma(70, alive))
    words = struct.unpack("<3I", rec.sections["bitmap"])
    assert words == (1 | 1 << 1 | 1 << 31, 1, 1 | 1 << 5)
    assert sum(w.bit_count() for w in words) == rec.f["n"] == 6


def test_frma_bonds_across_the_seam_keep_pair_order_and_slot_order() -> None:
    alive = np.zeros(16, bool)
    alive[[0, 2, 3, 10, 15]] = True
    # rows 0 and 15 are across the seam; row 2's bonds are listed (2, 10) then (2, 3), not sorted by partner
    rec = _one(16, _frma(16, alive, bonds=[(0, 15), (2, 10), (2, 3)]))
    assert rec.a["bonds"].tolist() == [[0, 4], [1, 3], [1, 2]]


def test_frma_2d_has_z_zero_and_ignores_pitch_columns() -> None:
    alive = np.array([True, False, True])
    blob = sr.encode_frma(
        step=1,
        seq=0,
        alive=alive,
        pos=np.array([[1.0, 2.0], [0, 0], [3.0, 4.0]]),
        orient=np.array([[0.5, 0.0], [0, 0], [0.25, 0.0]]),
        rgb=np.array([1, 2, 3]),
        uid=np.array([7, -1, 9]),
        bonds=np.zeros((0, 2)),
        color_index=0,
    )
    rec = _one(3, blob)
    assert rec.a["pos"].tolist() == [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]] and rec.a["uid"].tolist() == [7, 9]


def test_frma_refuses_what_it_cannot_carry() -> None:
    alive = np.array([True, True, False])
    with pytest.raises(sr.RecordError, match="dead row"):
        _frma(3, alive, bonds=[(0, 2)])
    with pytest.raises(sr.RecordError, match="i < j"):
        _frma(3, alive, bonds=[(1, 0)])
    with pytest.raises(sr.RecordError, match=r"inside N|i < j"):
        _frma(3, alive, bonds=[(0, 9)])


def test_record_head_fields_and_ranges() -> None:
    rec = _one(8, _frma(8, np.ones(8, bool), step=2**40 + 5, seq=9))
    assert (rec.step, rec.seq, rec.version, rec.flags) == (2**40 + 5, 9, 1, 0)
    with pytest.raises(sr.RecordError):
        sr.record("FRMA", -1, 0, 0, b"")
    with pytest.raises(sr.RecordError):
        sr.record("FRMA", 0, 0, 1 << 16, b"")
    with pytest.raises(sr.RecordError, match="multiple of 4"):
        sr.record("FRMA", 0, 0, 0, b"abc")
    seq = sr.Sequencer()
    assert [seq.take("FRMA"), seq.take("FRMA"), seq.take("CLCK"), seq.take("FRMA")] == [0, 1, 0, 2]


# --- IDNT, INTN, TERR -----------------------------------------------------------------------------------


def _tables(houses: int = 2, kin: int = 3, colonies: int = 1) -> dict:
    return {
        "houses": [{"slot": i} for i in range(houses)],
        "exact": True,
        "origin": 0,
        "kin": [{"slot": i} for i in range(kin)],
        "colonies": [{"slot": i} for i in range(colonies)],
        "unknown_immigrant": 0,
    }


def test_idnt_rows_are_16_bytes_and_slots_are_checked() -> None:
    n = 5
    uid = np.array([2**40, 1, 2, 3, 2**62], np.int64)
    kw = dict(
        step=7,
        seq=0,
        snapshot_step=4,
        uid=uid,
        clade=np.array([-1, 0, 5, 2**31 - 1, 7], np.int32),
        house=np.array([0, 1, 253, 254, 255], np.uint8),
        kin=np.array([0, 1, 2, 255, 254], np.uint8),
        colony=np.array([0, 255, 255, 254, 253], np.uint8),
        flags=np.array([0, 1, 0, 1, 0], np.uint8),
    )
    rec = _one(8, sr.encode_idnt(tables=_tables(), **kw))
    assert len(rec.sections["rows"]) == 16 * n and rec.f["snapshot_step"] == 4
    for key in ("uid", "clade", "house", "kin", "colony", "flags"):
        np.testing.assert_array_equal(rec.a[key], np.asarray(kw[key]))
    assert set(rec.f["json"]) == {"houses", "exact", "origin", "kin", "colonies", "unknown_immigrant"}
    for key in ("house", "kin", "colony"):
        # a slot one past a table of 3 is refused; the last slot and the reserved bytes 253 to 255 are fine
        bad = {
            **kw,
            "house": np.zeros(5, np.uint8),
            "kin": np.zeros(5, np.uint8),
            "colony": np.zeros(5, np.uint8),
        }
        with pytest.raises(sr.RecordError, match="not the slot of a listed entry"):
            sr.encode_idnt(**{**bad, key: np.array([3, 0, 0, 0, 0], np.uint8)}, tables=_tables(3, 3, 3))
        sr.encode_idnt(**{**bad, key: np.array([2, 0, 253, 254, 255], np.uint8)}, tables=_tables(3, 3, 3))
    with pytest.raises(sr.RecordError, match="more than 16"):
        sr.encode_idnt(**kw, tables=_tables(houses=17))


def test_idnt_slots_may_be_sparse_because_a_slot_is_sticky() -> None:
    """A kin group keeps its slot while it lives, so a table can have gaps: a row names the ``slot`` field of
    an entry, not an entry's position."""
    tables = _tables(houses=1, kin=0, colonies=0)
    tables["kin"] = [{"slot": 2}, {"slot": 7}, {"slot": 15}]
    tables["colonies"] = [{"slot": 4}]
    kw = dict(
        step=7,
        seq=0,
        snapshot_step=4,
        uid=np.arange(4, dtype=np.int64),
        clade=np.zeros(4, np.int32),
        house=np.array([0, 254, 253, 255], np.uint8),
        colony=np.array([4, 254, 255, 4], np.uint8),
        flags=np.zeros(4, np.uint8),
    )
    rec = _one(8, sr.encode_idnt(kin=np.array([2, 7, 15, 254], np.uint8), tables=tables, **kw))
    np.testing.assert_array_equal(rec.a["kin"], [2, 7, 15, 254])
    for gap in (0, 1, 3, 8, 14):  # below the table size, but no entry owns it
        with pytest.raises(sr.RecordError, match=f"kin slot {gap} is not the slot"):
            sr.encode_idnt(kin=np.array([gap, 2, 2, 2], np.uint8), tables=tables, **kw)
    for bad, match in (
        ([{"slot": 1}, {"slot": 1}], "repeat a slot"),
        ([{"slot": 16}], "below 16"),
        ([{}], "below 16"),
    ):
        with pytest.raises(sr.RecordError, match=match):
            sr.encode_idnt(kin=np.array([254] * 4, np.uint8), tables={**tables, "kin": bad}, **kw)


def test_idnt_json_is_padded_unpadded_length_and_utf8() -> None:
    for name in ("a", "ab", "abc", "abcd", "é", "日本"):
        tables = {**_tables(), "houses": [{"slot": 0, "name": name}]}
        rec = _one(
            8,
            sr.encode_idnt(
                step=1,
                seq=0,
                snapshot_step=1,
                uid=np.array([1], np.int64),
                clade=np.array([0], np.int32),
                house=np.array([0], np.uint8),
                kin=np.array([255], np.uint8),
                colony=np.array([255], np.uint8),
                flags=np.array([0], np.uint8),
                tables=tables,
            ),
        )
        assert rec.f["json"]["houses"][0]["name"] == name
        raw = rec.sections["json"]
        js = sr.json_bytes(tables)
        assert len(raw) % 4 == 0 and raw[: len(js)] == js and raw[len(js) :] == b"\x00" * (len(raw) - len(js))
    # an empty population: zero rows
    rec = _one(
        8,
        sr.encode_idnt(
            step=1,
            seq=0,
            snapshot_step=1,
            uid=np.zeros(0, np.int64),
            clade=np.zeros(0, np.int32),
            house=np.zeros(0, np.uint8),
            kin=np.zeros(0, np.uint8),
            colony=np.zeros(0, np.uint8),
            flags=np.zeros(0, np.uint8),
            tables=_tables(0, 0, 0),
        ),
    )
    assert rec.f["n_rows"] == 0


def test_json_rejects_nan_and_infinity() -> None:
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(sr.RecordError):
            sr.json_bytes({"x": bad})
        with pytest.raises(sr.RecordError):
            sr.encode_envs(step=0, seq=0, obj={"pools": [], "v": bad})


def test_intn_rows_are_12_bytes() -> None:
    uid = np.array([5, 2**41], np.int64)
    rec = _one(
        8,
        sr.encode_intn(
            step=9,
            seq=0,
            labels_step=8,
            intent_version=3,
            uid=uid,
            arch_primary=np.array([1, 7], np.uint8),
            arch_secondary=np.array([0, 3], np.uint8),
            intent=np.array([9, 1], np.uint8),
            flags=np.array([1, 3], np.uint8),
        ),
    )
    assert len(rec.sections["rows"]) == 24 and rec.f == {"n_rows": 2, "labels_step": 8, "intent_version": 3}
    np.testing.assert_array_equal(rec.a["uid"], uid)
    assert rec.a["intent"].tolist() == [9, 1] and rec.a["flags"].tolist() == [1, 3]


def test_terr_planes_and_padding() -> None:
    owner, purity = pf._terr_planes()
    rec = _one(
        8, sr.encode_terr(step=3, seq=0, snapshot_step=2, ids=[17, 21, 22], owner=owner, purity=purity)
    )
    np.testing.assert_array_equal(rec.a["owner"], owner)
    np.testing.assert_array_equal(rec.a["purity"], purity)
    assert rec.f["json"] == {"ids": [17, 21, 22]} and (rec.f["nx"], rec.f["ny"]) == (32, 32)
    assert rec.length == 16 + 16 + ((len(sr.json_bytes({"ids": [17, 21, 22]})) + 3) // 4 * 4) + 2 * 1024
    # an odd plane size pads the tail to a multiple of 4
    rec = _one(
        8,
        sr.encode_terr(
            step=3,
            seq=1,
            snapshot_step=2,
            ids=[1],
            owner=np.zeros((3, 5), np.uint8),
            purity=np.ones((3, 5), np.uint8),
        ),
    )
    assert rec.length % 4 == 0 and rec.a["purity"].shape == (3, 5)
    # a slot that holds no kin group (-1 in ids) owns nothing
    with pytest.raises(sr.RecordError, match="not the slot of a listed entry"):
        sr.encode_terr(
            step=3,
            seq=0,
            snapshot_step=2,
            ids=[-1, 5],
            owner=np.zeros((2, 2), np.uint8),
            purity=np.zeros((2, 2), np.uint8),
        )
    with pytest.raises(sr.RecordError, match="not the slot of a listed entry"):
        sr.encode_terr(
            step=3,
            seq=0,
            snapshot_step=2,
            ids=[1],
            owner=np.full((2, 2), 1, np.uint8),
            purity=np.zeros((2, 2), np.uint8),
        )


# --- STRM, EVNT, ENVS -----------------------------------------------------------------------------------


def test_strm_envs_and_caps() -> None:
    rec = _one(8, sr.encode_strm(step=100, seq=0, tick=25, obj=pf._storm_obj(100)))
    assert rec.f["n_storms"] == 2 and rec.f["tick"] == 25
    assert set(rec.f["json"]) == {"detect_step", "storms", "ended"}
    keys = {
        "id",
        "name",
        "named",
        "hemi",
        "x",
        "y",
        "r",
        "ext",
        "zeta",
        "zeta_now",
        "wind",
        "rain",
        "age",
        "parent",
        "state",
        "track",
        "pred",
    }
    assert all(set(s) == keys for s in rec.f["json"]["storms"])
    storm = pf._storm_obj(0)["storms"][0]
    many = {"detect_step": 0, "storms": [storm] * 32, "ended": []}
    blob = sr.encode_strm(step=0, seq=0, tick=25, obj=many)
    assert len(blob) <= 24 * 1024 + 64  # budget: STRM <= 24 KB for 32 storms
    with pytest.raises(sr.RecordError, match="cap"):
        sr.encode_strm(step=0, seq=0, tick=25, obj={**many, "storms": [storm] * 33})
    rec = _one(8, sr.encode_envs(step=1, seq=0, obj=pf._pools()))
    assert set(rec.f["json"]) == {"pools", "season", "levers", "effects", "cooldowns", "influence"}


def _event(i: int = 1, **kw) -> dict:
    e = {
        "id": i,
        "step": 1,
        "kind": 1,
        "fam": "kin",
        "a": 2,
        "b": 0,
        "x": 1.0,
        "y": 2.0,
        "z": None,
        "imp": 1,
        "text": "A kin group appeared.",
    }
    return {**e, **kw}


def test_evnt_shape_and_caps() -> None:
    events = [_event(i, text="é" * 120) for i in range(64)]
    blob = sr.encode_evnt(step=5, seq=0, events=events)
    assert len(blob) <= 32 * 1024 + 64  # budget: EVNT <= 32 KB for 64 events
    rec = _one(8, blob)
    assert rec.f["count"] == 64 and rec.f["json"][3]["text"] == "é" * 120 and not rec.f["backlog"]
    assert _one(8, sr.encode_evnt(step=5, seq=0, events=[], backlog=True)).f == {
        "count": 0,
        "backlog": True,
        "json": [],
    }
    with pytest.raises(sr.RecordError, match="cap"):
        sr.encode_evnt(step=5, seq=0, events=[*events, _event(99)])
    for bad in (
        _event(text="x" * 121),
        _event(fam="zzz"),
        _event(imp=3),
        _event(kind="kin_born"),
        _event(id=-1),
        {k: v for k, v in _event().items() if k != "z"},
        {**_event(), "extra": 1},
    ):
        with pytest.raises(sr.RecordError):
            sr.encode_evnt(step=5, seq=0, events=[bad])
    assert set(sr.EVENT_KINDS) == set(range(1, 22))


# --- MAP1 -----------------------------------------------------------------------------------------------


def test_map1_layout_and_sizes() -> None:
    planes = pf._map_planes()
    blob = sr.encode_map1(step=100, seq=0, planes=planes, dc_ref=0.5)
    rec = _one(8, blob)
    assert rec.flags == 3 and (rec.f["n"], rec.f["nw"]) == (160, 40)
    assert rec.length == 16 + 32 + 4 * 160 * 160 + 4 * 40 * 40 and 100_000 < len(blob) < 110_000  # 109 KB
    assert (rec.f["f0"], rec.f["p_ref"], rec.f["dc_ref"], rec.f["u_ref"], rec.f["zeta_ref"]) == (
        np.float32(0.025),
        np.float32(0.004),
        0.5,
        np.float32(0.1),
        np.float32(0.025),
    )
    # planes are row-major [cy][cx]; the first byte of the planes is at payload offset 48
    for k in ("P", "Dc", "zeta", "Ts", "winds"):
        np.testing.assert_array_equal(rec.a[k], planes[k])
    assert rec.a["P"][1, 0] == planes["P"][1, 0] != planes["P"][0, 1]  # [cy][cx], not [cx][cy]
    with pytest.raises(sr.RecordError):
        sr.encode_map1(step=0, seq=0, planes={**planes, "Ts": planes["Ts"][:-1]}, dc_ref=0.5)


def test_map1_quantisation_round_trips_within_the_stated_error() -> None:
    rng = np.random.default_rng(3)
    n, p_ref, dc_ref, u_ref, z_ref = 160, 0.004, 0.5, 0.1, 0.025
    p = rng.random((n, n)) * 0.006  # beyond p_ref clips
    dc = rng.random((n, n)) * 0.7
    zeta = (rng.random((n, n)) - 0.5) * 0.08
    ts = rng.random((n, n)) * 1.2 - 0.1
    winds = (rng.random((4, n, n)) - 0.5) * 0.3
    q = sr.quantise_map1(p=p, dc=dc, zeta=zeta, ts=ts, winds=winds, dc_ref=dc_ref)
    assert q["P"].dtype == np.uint8 and q["zeta"].dtype == np.int8 and q["winds"].shape == (4, 40, 40)
    # P: companded, decode (b/255)^2·p_ref; error in b is <= 0.5, so in P at most (2b+0.5)/255^2·p_ref
    dec = sr.dequantise_p(q["P"])
    clipped = np.minimum(p, p_ref)
    assert (np.abs(dec - clipped) <= (2 * 255 * 0.5 + 0.25) / 255**2 * p_ref + 1e-12).all()
    assert (np.abs(q["Dc"] / 255 * dc_ref - np.minimum(dc, dc_ref)) <= 0.5 / 255 * dc_ref + 1e-12).all()
    assert (np.abs(q["zeta"] / 127 * z_ref - np.clip(zeta, -z_ref, z_ref)) <= 0.5 / 127 * z_ref + 1e-12).all()
    assert (np.abs(q["Ts"] / 255 - np.clip(ts, 0, 1)) <= 0.5 / 255 + 1e-12).all()
    coarse = winds.reshape(4, 40, 4, 40, 4).mean(axis=(2, 4))
    assert (
        np.abs(q["winds"] / 127 * u_ref - np.clip(coarse, -u_ref, u_ref)) <= 0.5 / 127 * u_ref + 1e-12
    ).all()
    # the extremes and the signs
    assert (
        sr.quantise_map1(
            p=np.full((160, 160), 1.0),
            dc=np.zeros((160, 160)),
            zeta=np.full((160, 160), -1.0),
            ts=np.ones((160, 160)),
            winds=np.zeros((4, 160, 160)),
            dc_ref=1.0,
        )["P"].max()
        == 255
    )
    assert (q["zeta"][zeta < 0] <= 0).all() and (q["zeta"][zeta > 0] >= 0).all()
    with pytest.raises(sr.RecordError):
        sr.quantise_map1(
            p=p[:100, :100],
            dc=dc[:100, :100],
            zeta=zeta[:100, :100],
            ts=ts[:100, :100],
            winds=winds[:, :100, :100],
            dc_ref=0.5,
        )


# --- CLCK, FOCS -----------------------------------------------------------------------------------------


def test_clck_and_focs_bodies() -> None:
    rec = decode(pf.paths("3d")[0].read_bytes()).records
    clck = [r for r in rec if r.tag == "CLCK"]
    assert [len(r.sections["body"]) for r in clck] == [44, 44] and clck[0].length == 60
    assert clck[1].f["rate"] == -1.0 and clck[1].f["saving"] == 1 and clck[1].f["late_steps"] == 3
    assert (
        clck[1].f["save_stall_steps"] == 21 and clck[1].f["reason"] == 1 and clck[1].f["chapter_len"] == 6000
    )
    focs = next(r for r in rec if r.tag == "FOCS")
    assert len(focs.sections["body"]) == 68 and focs.length == 84
    assert (
        focs.f["uid"] == pf.UID0 + 100 and focs.f["parent_uid"] == pf.UID0 + 28 and focs.f["energy"] == 4096
    )
    assert (focs.f["x"], focs.f["y"], focs.f["z"], focs.f["vy"]) == (20.5, 8.5, 1.0, -0.125)
    assert (focs.f["house"], focs.f["kin"], focs.f["colony"]) == (0, 2, 255)
    assert focs.f["flags"] == sr.FOCS_ALIVE | sr.FOCS_HANDOFF | sr.FOCS_FOLLOWED
    with pytest.raises(sr.RecordError, match="reason"):
        sr.encode_clck(
            step=0,
            seq=0,
            rate=1,
            achieved=1,
            step_ms=1,
            frame_ms=1,
            late_steps=0,
            save_stall_steps=0,
            reason=9,
            saving=False,
            fe=1,
            chapter=0,
            in_chapter=0,
            chapter_len=1,
        )
    with pytest.raises(sr.RecordError, match="slot"):
        sr.encode_focs(
            step=0,
            seq=0,
            uid=1,
            row=0,
            flags=1,
            pos=(0, 0, 0),
            vel=(0, 0, 0),
            energy=1,
            age=0,
            generation=0,
            parent_uid=-1,
            n_bonds=0,
            house=16,
            kin=0,
            colony=0,
            arch_primary=0,
            arch_secondary=0,
            intent=0,
            label_flags=0,
        )


# --- evolution rules ------------------------------------------------------------------------------------


def test_unknown_tags_and_versions_are_skipped_by_length() -> None:
    """Every tag is skipped when its version is not 1 (FRMA included), and an unknown tag by its length."""
    alive = np.ones(8, bool)
    frma = bytearray(_frma(8, alive))
    frma[8 + 12 : 8 + 14] = struct.pack("<H", 2)  # the version field of the common head
    strm = bytearray(
        sr.encode_strm(step=1, seq=0, tick=25, obj={"detect_step": 1, "storms": [], "ended": []})
    )
    strm[8 + 12 : 8 + 14] = struct.pack("<H", 9)
    good = sr.encode_envs(step=1, seq=0, obj={"pools": []})
    zzzz = sr.record("ZZZZ", 5, 0, 0, b"\x01" * 16)
    d = decode(_header(8) + bytes(frma) + zzzz + bytes(strm) + good)
    assert [r.tag for r in d.records] == ["ENVS"]
    assert [(t, why) for t, _, why in d.skipped] == [
        ("FRMA", "version"),
        ("ZZZZ", "tag"),
        ("STRM", "version"),
    ]
    # new flag bits and new JSON keys are compatible: a reader that does not know them ignores them
    extra = sr.encode_envs(step=1, seq=1, obj={"pools": [], "future": {"k": 1}})
    assert decode(_header(8) + extra).records[0].f["json"]["future"] == {"k": 1}


def test_a_stream_with_no_play_records_still_decodes() -> None:
    d = decode(_header(8))
    assert d.records == [] and d.skipped == []


# --- budgets and structure ------------------------------------------------------------------------------


def test_record_budgets_at_64k() -> None:
    n_cap, n = 65536, 42000
    assert sr.frma_length(n_cap, n, 40000) < 1_600_000  # about 1.5 MB at 42k alive
    assert sr.frma_length(n_cap, n, 0) < 21 * n_cap  # smaller than FRM1's 21 bytes per slot
    rows = np.zeros((n, 16), np.uint8)
    assert rows.nbytes + 16 * 2 + 32 + sr.IDNT_JSON_CAP <= 16 * n_cap + 16 * 1024 + 64  # IDNT <= 16·N + 16 KB


def test_streamrec_is_numpy_only_and_free_of_subscript_assignment() -> None:
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not ({"mlx", "alife"} & imported), imported
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign | ast.AugAssign | ast.AnnAssign):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            assert not any(isinstance(s, ast.Subscript) for t in targets for s in ast.walk(t)), n.lineno


def test_decoder_does_not_import_the_encoder() -> None:
    tree = ast.parse((Path(__file__).parent / "play_decode.py").read_text(encoding="utf-8"))
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not any("alife" in m or "streamrec" in m for m in names), names


def test_the_decoder_checks_the_label_bytes_against_the_header() -> None:
    """``INTN`` carries the header's ``intent_version`` and its (and ``FOCS``'s) archetype and intent bytes
    name an entry of the header's tables: the decoder is strict about them."""

    def intn(version: int, arch: int, intent: int) -> bytes:
        return sr.encode_intn(
            step=9,
            seq=0,
            labels_step=8,
            intent_version=version,
            uid=np.array([5], np.int64),
            arch_primary=np.array([arch], np.uint8),
            arch_secondary=np.array([0], np.uint8),
            intent=np.array([intent], np.uint8),
            flags=np.array([1], np.uint8),
        )

    assert _one(8, intn(3, 10, 10)).f["intent_version"] == 3  # the last entry of each table
    for bad, why in (
        (intn(2, 1, 1), "intent_version"),
        (intn(3, 11, 1), "archetype"),
        (intn(3, 1, 11), "intent"),
    ):
        with pytest.raises(AssertionError, match=why):
            _one(8, bad)


# --- G3b: ALRT, RPRT, PLAN, and the additions to ENVS, CLCK, EVNT (DESIGN §34.6) ---------------------------

CLASSES = pf.ALERT_CLASSES


def _alert(index: int = 0, first: int = 12000, **kw) -> dict:
    return pf._alert(index, first, **kw)


def _plan(status: str = "queued", **kw) -> dict:
    return pf._plan(status, **kw)


def _g3b_header(n: int = 64) -> bytes:
    legacy = sr.legacy_header(
        dims=3,
        size=48.0,
        height=24.0,
        n=n,
        bond_slots=4,
        radius=1.0,
        field_cell=4.0,
        fg=12,
        fz=6,
        nch=3,
        step=0,
        run=None,
        config_hash="h",
        source="preset",
    )
    play = sr.header_play(
        records=["ALRT", "RPRT", "PLAN"],
        weather=sr.weather_block(enabled=False, dc_ref=None),
        labels={"intent_version": 3, "arch": ["", "A"], "intent": ["", "I"]},
        energy_cap=100,
        alerts=sr.alerts_block(CLASSES),
    )
    return sr.play_header_bytes(legacy, play)


def _decode_one(record: bytes):
    d = decode(_g3b_header() + record)
    assert len(d.records) == 1 and not d.skipped
    return d.records[0]


def test_the_g3b_fixture_has_every_new_record_and_every_case_the_spec_names() -> None:
    d = decode(pf.paths("3d_g3b")[0].read_bytes())
    by_tag: dict[str, list] = {}
    for r in d.records:
        by_tag.setdefault(r.tag, []).append(r)
    assert sorted(by_tag) == ["ALRT", "CLCK", "ENVS", "EVNT", "FRMA", "IDNT", "PLAN", "RPRT"]
    assert [(t, w) for t, _, w in d.skipped] == [("ALRT", "version")], (
        "an ALRT of version 2 is skipped by length"
    )
    play = d.header["play"]
    assert play["records"] == ["FRMA", "IDNT", "EVNT", "ENVS", "CLCK", "ALRT", "RPRT", "PLAN"]
    assert play["versions"] == dict.fromkeys(play["records"], 1) and play["game_specs"] == "game-v0"
    assert set(play["alerts"]["classes"]) == set(CLASSES) and play["alerts"]["cap"]["n"] == 4
    assert play["autopause"] == {
        "chapter": True,
        "alert": {
            "chapter": True,
            "pop_fall": True,
            "storm_named": False,
            "lineage_formed": False,
            "pop_surge": False,
            "storm_strong": False,
        },
    }
    assert set(play["powers"]) == {"food", "strike"} and play["events"]["20"] == "power_applied"
    backlog_alrt, live_alrt = by_tag["ALRT"]
    assert backlog_alrt.f["backlog"] and not live_alrt.f["backlog"] and backlog_alrt.flags == 1
    alerts = live_alrt.f["json"]
    assert {a["imp"] for a in alerts} == {0, 1, 2, 3} and {a["state"] for a in alerts} == set(sr.ALERT_STATES)
    assert {a["ref"]["kind"] for a in alerts if a["ref"]} == set(sr.ALERT_REF_KINDS)
    assert {a["surface"] for a in alerts} == {"alert", "log"} and any(a["watched"] for a in alerts)
    assert all(a["id"] % 1000 >= 600 and a["id"] // 1000 <= a["step"] for a in alerts), (
        "ids: first step, 600 + index"
    )
    assert any(a["id"] // 1000 != a["step"] for a in alerts), (
        "an alert keeps its id when it is updated at a later step"
    )
    modes = [(r.f["chapter"], r.f["mode"], r.f["backlog"], r.f["final"]) for r in by_tag["RPRT"]]
    assert modes == [(1, 0, True, True), (2, 0, False, True), (2, 1, False, False), (2, 2, False, True)]
    assert by_tag["RPRT"][-1].f["json"]["withheld"] and by_tag["RPRT"][-1].f["json"]["headline"] is None
    plans = by_tag["PLAN"]
    assert [p.f["json"]["status"] for p in plans] == [
        "preview",
        "queued",
        "applied",
        "since",
        "refused",
        "cancelled",
        "refused",
    ]
    assert [p.f["reply"] for p in plans] == [True, False, False, False, True, False, True]
    assert (
        plans[-1].f["json"]["kind"] == "intent" and plans[0].f["json"]["id"] == plans[1].f["json"]["id"] == 3
    )
    envs = by_tag["ENVS"][0].f["json"]
    assert {e["source"] for e in envs["effects"]} == {"player", "rule"}
    assert envs["influence"] == {"have": 4, "cap": 6, "window": 18000, "free_at": [13000, 15500]}
    assert len(envs["patches"]) == 2 and envs["powers"]["strike"]["busy"].startswith("cooldown")
    evnt_kinds = {e["kind"] for r in by_tag["EVNT"] for e in r.f["json"]}
    assert evnt_kinds == {18, 19, 20, 21}
    clcks = by_tag["CLCK"]
    assert [(c.f["reason"], c.f["flags"], c.f["chapter"], c.f["in_chapter"]) for c in clcks] == [
        (3, 3, 2, 6000),
        (4, 1, 3, 1),
    ]
    idnt = by_tag["IDNT"][0].f["json"]
    assert [k["thread"] for k in idnt["kin"]] == [4, 5, 6]
    assert max(len(r.sections["json"]) for r in by_tag["RPRT"]) < sr.RPRT_JSON_CAP


def test_g3b_header_keys_are_written_only_when_given() -> None:
    plain = sr.header_play(records=["FRMA"], weather={}, labels={}, energy_cap=1)
    assert not {"alerts", "autopause", "powers", "game_specs"} & set(plain), "G3a's header is what it was"
    rich = sr.header_play(
        records=["FRMA", "ALRT", "RPRT", "PLAN"],
        weather={},
        labels={},
        energy_cap=1,
        alerts=sr.alerts_block(CLASSES),
        autopause=sr.autopause_block(chapter=False, alert={"pop_fall": True}),
        powers=sr.powers_block(pf.POWERS),
        game_specs="game-v0",
    )
    assert {"alerts", "autopause", "powers", "game_specs"} <= set(rich) and rich["versions"]["PLAN"] == 1
    assert list(rich["alerts"]["classes"]["chapter"]) == ["imp", "pause", "label", "calibrated", "surface"]
    assert rich["alerts"]["cap"] == {"n": 4, "window": 1500, "exempt": ["imp3", "pause", "chapter", "engine"]}
    for bad in (
        {},
        {"x": {"imp": 4, **{k: v for k, v in CLASSES["chapter"].items() if k != "imp"}}},
        {"x": {**CLASSES["chapter"], "surface": "toast"}},
        {"x": {**CLASSES["chapter"], "pause": 1}},
        {"x": {"imp": 1}},
        {"": CLASSES["chapter"]},
    ):
        with pytest.raises(sr.RecordError):
            sr.alerts_block(bad)
    with pytest.raises(sr.RecordError):
        sr.autopause_block(chapter=1, alert={})  # type: ignore[arg-type]
    with pytest.raises(sr.RecordError):
        sr.powers_block({"food": {"action": "regen_pulse"}})
    with pytest.raises(sr.RecordError):
        sr.powers_block({"food": {"action": "a", "label": "Food", "cost": -1, "cooldown": 0}})
    with pytest.raises(sr.RecordError, match="game table"):
        sr.header_play(records=["FRMA"], weather={}, labels={}, energy_cap=1, game_specs="")
    assert dict.fromkeys(sr.TAGS, 1) == sr.VERSIONS and sr.TAGS[-3:] == ("ALRT", "RPRT", "PLAN")


def test_alrt_layout_count_padding_and_flags() -> None:
    alerts = [_alert(0, title="Population fell\u2014caf\u00e9"), _alert(1, id=12000_601 + 0)]
    alerts[1] = _alert(1)
    data = sr.encode_alrt(step=12000, seq=7, alerts=alerts, registry=CLASSES, backlog=True)
    assert (
        data[:4] == b"ALRT" and len(data) % 4 == 0 and struct.unpack_from("<I", data, 4)[0] == len(data) - 8
    )
    count, json_len = struct.unpack_from("<II", data, 8 + 16)
    assert count == 2 and 8 + 24 + json_len <= len(data) < 8 + 24 + json_len + 4
    assert data[8 + 24 + json_len :] == b"\x00" * (len(data) - 8 - 24 - json_len)
    r = _decode_one(data)
    assert (r.step, r.seq, r.version, r.flags, r.f["count"]) == (12000, 7, 1, 1, 2) and r.f["json"] == alerts
    assert r.f["json"][0]["title"].endswith("caf\u00e9"), "UTF-8, not escaped"
    assert not _decode_one(sr.encode_alrt(step=1, seq=0, alerts=[], registry=CLASSES)).f["json"]


def test_alrt_refuses_what_the_registry_and_the_text_say_it_cannot_carry() -> None:
    ok = _alert()
    cases = {
        "sixteen at most": [_alert(i) for i in range(17)],
        "a class outside the registry": [{**ok, "class": "made_up"}],
        "a state outside the list": [{**ok, "state": "removed"}],
        "a surface outside the list": [{**ok, "surface": "toast"}],
        "importance 4": [{**ok, "imp": 4}],
        "raised is version 1": [{**ok, "version": 2}],
        "updated is version 2 or more": [{**ok, "state": "updated", "version": 1}],
        "an id below the alert range": [{**ok, "id": 12000_500}],
        "a title of 49 characters": [{**ok, "title": "t" * 49}],
        "a text of 161 characters": [{**ok, "text": "t" * 161}],
        "a position that is not finite": [{**ok, "x": float("inf")}],
        "a ref of an unknown kind": [{**ok, "ref": {"kind": "planet", "id": 1}}],
        "a ref with another key": [{**ok, "ref": {"kind": "kin", "id": 1, "name": "x"}}],
        "pause as a number": [{**ok, "pause": 1}],
        "a missing key": [{k: v for k, v in ok.items() if k != "watched"}],
        "an extra key": [{**ok, "extra": 1}],
        "the same id twice": [ok, {**ok, "key": "other"}],
    }
    for what, alerts in cases.items():
        with pytest.raises(sr.RecordError):
            sr.encode_alrt(step=1, seq=0, alerts=alerts, registry=CLASSES)
        assert what
    # the limits themselves are fine
    edge = [_alert(i, title="t" * 48, text="t" * 160) for i in range(16)]
    assert len(_decode_one(sr.encode_alrt(step=1, seq=0, alerts=edge, registry=CLASSES)).f["json"]) == 16
    # a registry is optional for the encoder, and the decoder, which has the header, checks it
    odd = sr.encode_alrt(step=1, seq=0, alerts=[{**ok, "class": "made_up"}])
    with pytest.raises(AssertionError, match="registry"):
        decode(_g3b_header() + odd)


def test_alrt_json_cap() -> None:
    big = [_alert(i, title="t" * 48, text="\u20ac" * 160) for i in range(16)]  # 3-byte characters
    assert len(sr.json_bytes(big)) < sr.ALRT_JSON_CAP, "sixteen of the longest alerts fit"
    long_keys = [_alert(i, key="k" * 2000) for i in range(16)]  # the key is the one field with no limit
    with pytest.raises(sr.RecordError, match="exceeds"):
        sr.encode_alrt(step=1, seq=0, alerts=long_keys, registry=CLASSES)


def test_rprt_layout_and_the_head_is_the_json() -> None:
    obj = pf._report(3, 1, True)
    data = sr.encode_rprt(step=18000, seq=2, obj=obj, backlog=True)
    assert data[:4] == b"RPRT" and len(data) % 4 == 0
    chapter, mode, json_len, reserved = struct.unpack_from("<IIII", data, 8 + 16)
    assert (chapter, mode, reserved) == (3, 1, 0) and 8 + 32 + json_len <= len(data) < 8 + 32 + json_len + 4
    r = _decode_one(data)
    assert (r.flags, r.f["chapter"], r.f["mode"], r.f["backlog"], r.f["final"]) == (3, 3, 1, True, True)
    assert r.f["json"] == obj
    live = _decode_one(sr.encode_rprt(step=1, seq=0, obj=pf._report(3, 1, False)))
    assert (live.flags, live.f["final"], live.f["backlog"]) == (0, False, False)
    # the independent decoder holds the writer to the head: a flag that is not the JSON's is refused
    raw = bytearray(data)
    raw[8 + 14 : 8 + 16] = struct.pack("<H", 1)  # final cleared in the flags, set in the JSON
    with pytest.raises(AssertionError, match="final"):
        decode(_g3b_header() + bytes(raw))


def test_rprt_refuses_a_malformed_report() -> None:
    ok = pf._report(2, 0, True)
    bad = {
        "mode 3": {**ok, "mode": 3},
        "a after b": {**ok, "a": 7000, "b": 6000},
        "a missing key": {k: v for k, v in ok.items() if k != "ticker"},
        "body that is not a list": {**ok, "body": "x"},
        "a sentence without refs": {**ok, "lead": {"id": "L", "text": "t"}},
        "refs that are not keys": {**ok, "headline": {"id": "H", "text": "t", "refs": [7]}},
        "versions missing a part": {**ok, "versions": {"report": 2}},
        "a withheld report that says something": {**ok, "withheld": ["no row"]},
        "latency as a string": {**ok, "latency_s": "fast"},
        "final as a number": {**ok, "final": 1},
        "a negative chapter": {**ok, "chapter": -1},
    }
    for what, obj in bad.items():
        with pytest.raises(sr.RecordError):
            sr.encode_rprt(step=1, seq=0, obj=obj)
        assert what
    with pytest.raises(sr.RecordError, match="exceeds"):
        sr.encode_rprt(
            step=1,
            seq=0,
            obj={
                **ok,
                "body": [
                    {"id": f"B{i}", "text": "t" * 1000, "origin": "sheet", "refs": []} for i in range(70)
                ],
            },
        )
    big = {
        **ok,
        "body": [{"id": f"B{i}", "text": "t" * 1000, "origin": "sheet", "refs": []} for i in range(50)],
    }
    assert 8 * 1024 < len(sr.json_bytes(big)) < sr.RPRT_JSON_CAP
    assert len(_decode_one(sr.encode_rprt(step=1, seq=0, obj=big)).f["json"]["body"]) == 50, "up to 64 KiB"
    empty = {
        **ok,
        "withheld": ["the cut 11000 is not the step of a metrics row"],
        "headline": None,
        "lead": None,
        "body": [],
        "interventions": [],
        "ticker": [],
        "entities": {},
    }
    assert _decode_one(sr.encode_rprt(step=1, seq=0, obj=empty)).f["json"]["withheld"]
    # a ref that is not an entity of the report is the decoder's to refuse
    dangling = sr.encode_rprt(step=1, seq=0, obj={**ok, "entities": {}})
    with pytest.raises(AssertionError, match="entities"):
        decode(_g3b_header() + dangling)


def test_plan_layout_statuses_and_the_reply_flag() -> None:
    data = sr.encode_plan(step=12000, seq=1, obj=_plan("preview"), reply=True)
    assert data[:4] == b"PLAN" and len(data) % 4 == 0
    (json_len,) = struct.unpack_from("<I", data, 8 + 16)
    assert 8 + 20 + json_len <= len(data) < 8 + 20 + json_len + 4
    r = _decode_one(data)
    assert (r.flags, r.f["reply"], r.f["json"]["status"]) == (1, True, "preview")
    for status in sr.PLAN_STATUSES:
        extra = (
            {"refused": "no"}
            if status == "refused"
            else {"step_applied": 12000}
            if status == "applied"
            else {}
        )
        reply = status in ("preview", "refused")
        r = _decode_one(sr.encode_plan(step=1, seq=0, obj=_plan(status, **extra), reply=reply))
        assert r.f["json"]["status"] == status and r.f["reply"] is reply
    with pytest.raises(sr.RecordError, match="reply"):
        sr.encode_plan(step=1, seq=0, obj=_plan("preview"))


def test_plan_refuses_a_malformed_plan() -> None:
    bad = {
        "an unknown status": {**_plan(), "status": "done"},
        "an unknown kind": _plan("queued", kind="spell"),
        "a missing key": {k: v for k, v in _plan().items() if k != "caution"},
        "a text of 161 characters": _plan(text="t" * 161),
        "a caution of 101 characters": _plan(caution=["c" * 101]),
        "a refusal without a reason": _plan("refused", refused=None),
        "a reason without a refusal": _plan("queued", refused="no"),
        "an applied plan without its step": _plan("applied"),
        "a negative cost": _plan(cost=-1),
        "an id that is not a number": _plan(id="3"),
        "clamped names that are not strings": _plan(clamped=[1]),
        "an influence of another shape": _plan(influence={"have": 1}),
    }
    for what, obj in bad.items():
        with pytest.raises(sr.RecordError):
            sr.encode_plan(step=1, seq=0, obj=obj, reply=True)
        assert what
    with pytest.raises(sr.RecordError, match="exceeds"):
        sr.encode_plan(step=1, seq=0, obj=_plan(raw={"k": "v" * 9000}))
    edge = _plan(text="t" * 160, caution=["c" * 100] * 3)
    assert _decode_one(sr.encode_plan(step=1, seq=0, obj=edge)).f["json"]["caution"] == ["c" * 100] * 3


def test_envs_additions_are_checked_when_present_and_old_objects_still_pass() -> None:
    base = {"pools": [], "season": None, "levers": {}, "effects": [], "cooldowns": [], "influence": None}
    assert _one_envs(base)
    full = {
        **base,
        "influence": {"have": 4, "cap": 6, "window": 18000, "free_at": [13000]},
        "patches": [{"i": 0, "x": 1.0, "y": 2.0, "gain": 1.5, "sigma": 6.0}],
        "powers": {"food": {"ready": None, "busy": None}, "strike": {"ready": 15000, "busy": "cooling down"}},
        "effects": [{"action": "bloom", "params": {}, "start": 0, "end": None, "source": "player"}],
    }
    assert _one_envs(full)["patches"][0]["gain"] == 1.5
    bad = {
        "influence missing a key": {**base, "influence": {"have": 4, "cap": 6, "window": 18000}},
        "more influence than the cap": {
            **base,
            "influence": {"have": 7, "cap": 6, "window": 18000, "free_at": []},
        },
        "free_at as a number": {**base, "influence": {"have": 1, "cap": 6, "window": 18000, "free_at": 5}},
        "a patch without sigma": {**base, "patches": [{"i": 0, "x": 1.0, "y": 2.0, "gain": 1.5}]},
        "a patch at infinity": {
            **base,
            "patches": [{"i": 0, "x": float("inf"), "y": 2.0, "gain": 1.5, "sigma": 1.0}],
        },
        "a power without busy": {**base, "powers": {"food": {"ready": None}}},
        "a busy that is a number": {**base, "powers": {"food": {"ready": None, "busy": 3}}},
        "an effect of another source": {**base, "effects": [{"action": "a", "source": "random"}]},
    }
    for what, obj in bad.items():
        with pytest.raises(sr.RecordError):
            sr.encode_envs(step=1, seq=0, obj=obj)
        assert what


def _one_envs(obj: dict) -> dict:
    d = decode(_g3b_header() + sr.encode_envs(step=1, seq=0, obj=obj))
    return d.records[0].f["json"]


def test_clck_flags_are_the_two_pause_bits() -> None:
    assert (sr.CLCK_FLAG_CHAPTER_PAUSE, sr.CLCK_FLAG_ALERT_PAUSE, sr.CLCK_FLAGS) == (1, 2, 3)
    kw = dict(
        step=12000,
        seq=0,
        rate=1.0,
        achieved=1.0,
        step_ms=1.0,
        frame_ms=1.0,
        late_steps=0,
        save_stall_steps=0,
        reason=3,
        saving=True,
        fe=1,
        chapter=2,
        in_chapter=6000,
        chapter_len=6000,
    )
    for flags in (0, 1, 2, 3):
        r = _decode_one(sr.encode_clck(**kw, flags=flags))
        assert r.f["flags"] == flags and r.f["reason"] == 3 and r.f["chapter"] == 2
    with pytest.raises(sr.RecordError, match="flags"):
        sr.encode_clck(**kw, flags=4)
    raw = bytearray(sr.encode_clck(**kw, flags=3))
    raw[8 + 16 + 26 : 8 + 16 + 28] = struct.pack("<H", 8)  # the flags field of the body
    with pytest.raises(AssertionError, match="flags"):
        decode(_g3b_header() + bytes(raw))


def test_power_event_kinds_are_in_the_registry_and_encode() -> None:
    assert sr.EVENT_KINDS[20] == "power_applied" and sr.EVENT_KINDS[21] == "power_ended"
    assert sr.EVENT_KINDS[18] == "kin_formed" and sr.EVENT_KINDS[19] == "kin_ended"
    e = {
        "id": 12000_300,
        "step": 12000,
        "kind": 20,
        "fam": "director",
        "a": 0,
        "b": 0,
        "x": None,
        "y": None,
        "z": None,
        "imp": 1,
        "text": "The player set regen to 1.3 for 6000 steps.",
    }
    assert _decode_one(sr.encode_evnt(step=12000, seq=0, events=[e])).f["json"] == [e]
    assert pf.paths("3d_g3b")[0].stat().st_size < 20_000, (
        "the new records are small: a chapter's worth is some KB"
    )


def test_the_decoder_of_the_new_records_is_written_without_the_encoder() -> None:
    text = (Path(__file__).parent / "play_decode.py").read_text()
    assert "streamrec" not in text.replace("``streamrec``", "") or "import streamrec" not in text
    for needle in ("ALRT", "RPRT", "PLAN", "def _alrt", "def _rprt", "def _plan"):
        assert needle in text
