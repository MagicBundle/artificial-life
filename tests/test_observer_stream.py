"""The observer-PR stream records (``STRC``, ``FSIG``, ``FSOL``, the ``FRMA`` bond flow and the ``play``
header
blocks of the feature extensions) against the independent decoder of ``tests/play_decode.py``, which is
written
from ``stream_contract.md``, not from the encoder. No engine and no device."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from alife import streamrec as sr
from play_decode import decode
from test_stream_records import _frma, _header


def _one(record: bytes, n: int = 8):
    d = decode(_header(n, ("FRMA", "FOCS", *sr.EXT_TAGS)) + record)
    assert len(d.records) == 1 and not d.skipped
    return d.records[0]


# --- STRC and FSOL ------------------------------------------------------------------------------------------


def test_levels_are_one_plus_254m_over_cap_and_a_non_empty_cell_is_at_least_one() -> None:
    cap = 4096
    m = np.zeros((2, 4, 4), np.int64)
    m.reshape(-1)[[0, 5, 6, 20, 31]] = [1, 16, cap // 2, cap - 1, cap]
    idx, lev, trunc = sr.quantise_cells(m, cap)
    assert idx.tolist() == [0, 5, 6, 20, 31] and not trunc
    assert lev.tolist() == [1 + (254 * v) // cap for v in (1, 16, cap // 2, cap - 1, cap)]
    assert lev.min() == 1 and lev.max() == 255  # a full cell is 255
    assert idx.dtype == np.dtype("<u4") and lev.dtype == np.uint8


def test_an_empty_grid_is_a_valid_record_with_no_cells() -> None:
    rec = sr.encode_strc(step=7, seq=0, shape=(1, 4, 4), mat=np.zeros((4, 4), np.int32), cap=4096)
    r = _one(rec)
    assert r.tag == "STRC" and r.f["count"] == 0 and r.step == 7 and r.flags == 0
    assert len(rec) == 8 + 16 + 12  # tag, length, head, prefix: no padding for zero cells


@pytest.mark.parametrize("tag", ["STRC", "FSOL"])
def test_a_cell_record_round_trips_through_the_independent_decoder(tag: str) -> None:
    rng = np.random.default_rng(3)
    shape = (3, 5, 7)
    grid = np.where(rng.random(shape) < 0.3, rng.integers(1, 4097, shape), 0)
    rec = sr.encode_cells(tag, step=123, seq=4, shape=shape, grid=grid, cap=4096)
    r = _one(rec)
    assert r.tag == tag and (r.step, r.seq, r.version) == (123, 4, 1)
    assert (r.f["nx"], r.f["ny"], r.f["nz"]) == (7, 5, 3)
    flat = grid.reshape(-1)
    want = np.flatnonzero(flat)
    assert np.array_equal(r.a["idx"], want)  # z-major flat index, ascending
    assert np.array_equal(r.a["level"], 1 + (254 * flat[want]) // 4096)
    assert len(rec) % 4 == 0 and len(rec) == 8 + 16 + 12 + ((5 * want.size + 3) // 4) * 4


def test_the_index_is_z_major_with_x_fastest() -> None:
    grid = np.zeros((2, 3, 4), np.int32)
    grid[1, 2, 3] = 4096  # z = 1, y = 2, x = 3
    r = _one(sr.encode_strc(step=0, seq=0, shape=(2, 3, 4), mat=grid, cap=4096))
    assert r.a["idx"].tolist() == [(1 * 3 + 2) * 4 + 3] and r.a["level"].tolist() == [255]


def test_more_than_the_cap_keeps_the_highest_levels_and_sets_the_flag() -> None:
    n = sr.CELLS_MAX + 100
    side = 300
    grid = np.zeros((1, side, side), np.int64)
    flat = grid.reshape(-1)
    flat[:n] = 1 + (np.arange(n) % 4096)  # levels spread over 1..255
    idx, lev, trunc = sr.quantise_cells(grid, 4096)
    assert trunc and idx.size == sr.CELLS_MAX and (np.diff(idx.astype(np.int64)) > 0).all()
    kept_min = lev.min()
    dropped = np.setdiff1d(np.arange(n), idx)
    dropped_lev = 1 + (254 * flat[dropped]) // 4096
    assert dropped_lev.max() <= kept_min  # nothing dropped is higher than anything kept
    r = _one(sr.encode_strc(step=1, seq=0, shape=(1, side, side), mat=grid, cap=4096))
    assert r.flags == sr.FLAG_TRUNC and r.f["flags"] == sr.FLAG_TRUNC and r.f["count"] == sr.CELLS_MAX


def test_cells_outside_the_cap_and_a_wrong_shape_are_writer_errors() -> None:
    idx, lev, _ = sr.quantise_cells(
        np.array([[5000, -1, 7]]), 4096
    )  # clamped to [0, cap], never a session-ending error
    assert idx.tolist() == [0, 2] and lev.tolist() == [255, 1 + 254 * 7 // 4096]
    with pytest.raises(sr.RecordError):
        sr.encode_strc(step=0, seq=0, shape=(1, 3, 3), mat=np.zeros((4, 4)), cap=4096)
    with pytest.raises(sr.RecordError):
        sr.encode_cells("MAP1", step=0, seq=0, shape=(1, 1, 1), grid=np.zeros(1), cap=4096)


# --- FSIG ---------------------------------------------------------------------------------------------------


def test_fsig_round_trip_with_and_without_the_heard_half() -> None:
    says = np.array([0, 17, 255, 3], np.uint8)
    hears = np.array([9, 0, 128, 255], np.uint8)
    r = _one(sr.encode_fsig(step=9, seq=1, uid=-5 + (1 << 40), says=says, hears=hears, audible=7))
    assert r.tag == "FSIG" and r.f == {"uid": (1 << 40) - 5, "w": 4, "flags": 1, "audible": 7}
    assert np.array_equal(r.a["says"], says) and np.array_equal(r.a["hears"], hears)
    r2 = _one(sr.encode_fsig(step=9, seq=2, uid=3, says=says, hears=None, audible=7))
    assert r2.f["flags"] == 0 and r2.f["audible"] == 0 and not r2.a["hears"].any()  # no heard half: zeros


@pytest.mark.parametrize("w", [1, 2, 3, 4, 5, 8])
def test_fsig_is_padded_to_four_bytes_for_every_width(w: int) -> None:
    rec = sr.encode_fsig(step=0, seq=0, uid=1, says=np.arange(w, dtype=np.uint8), hears=np.zeros(w, np.uint8))
    assert len(rec) % 4 == 0 and len(rec) == 8 + 16 + 12 + ((2 * w + 3) // 4) * 4
    assert _one(rec).f["w"] == w


def test_fsig_writer_errors() -> None:
    with pytest.raises(sr.RecordError):
        sr.encode_fsig(step=0, seq=0, uid=1, says=np.zeros(9, np.uint8), hears=None)
    with pytest.raises(sr.RecordError):
        sr.encode_fsig(step=0, seq=0, uid=1, says=np.zeros(4, np.float32), hears=None)
    with pytest.raises(sr.RecordError):
        sr.encode_fsig(step=0, seq=0, uid=1, says=np.zeros(4, np.uint8), hears=np.zeros(3, np.uint8))


# --- FRMA bond flow -----------------------------------------------------------------------------------------


def test_frma_without_flow_is_unchanged_and_with_flow_sets_flag_bit_1() -> None:
    alive = np.array([1, 1, 1, 0, 1, 1, 1, 1], bool)
    bonds = [(0, 1), (1, 2), (4, 5)]
    plain = _frma(8, alive, bonds)
    assert plain == _frma(8, alive, bonds, flow=None) and _one(plain).flags == 0
    assert "flow" not in _one(plain).a
    flow = np.array([-127, 0, 64], np.int8)
    with_flow = _frma(8, alive, bonds, flow=flow)
    r = _one(with_flow)
    assert r.flags == sr.FLAG_BONDFLOW and np.array_equal(r.a["flow"], flow)
    assert len(with_flow) - len(plain) == 4  # three int8 padded to four bytes
    assert r.a["bonds"].shape == (3, 2)
    assert sr.frma_length(8, 7, 3, flow=True) == sr.frma_length(8, 7, 3) + 4


def test_frma_flow_needs_one_int8_per_bond() -> None:
    alive = np.ones(8, bool)
    with pytest.raises(sr.RecordError):
        _frma(8, alive, [(0, 1)], flow=np.zeros(2, np.int8))
    with pytest.raises(sr.RecordError):
        _frma(8, alive, [(0, 1)], flow=np.zeros(1, np.int16))


# --- the header ---------------------------------------------------------------------------------------------


def test_the_header_blocks_are_optional_and_validate() -> None:
    kw = dict(
        records=["FRMA"],
        weather=sr.weather_block(enabled=False, dc_ref=None),
        labels={"intent_version": 3, "arch": [""], "intent": [""]},
        energy_cap=1,
    )
    base = sr.header_play(**kw)
    for key in ("structures", "comm", "niche", "bonds"):
        assert key not in base  # a world without the features has the header it always had
    on = sr.header_play(
        **{**kw, "records": ["FRMA", "STRC", "FSIG"]},
        structures=sr.cells_block(cap=4096, cells=[6, 12, 12], cell=4.0),
        comm=sr.comm_block(w=4, hear_range=6.0, control="none"),
        bonds=sr.bonds_block(scale=1000),
    )
    assert on["structures"] == {"enabled": True, "cap": 4096, "cells": [6, 12, 12], "cell": 4.0}
    assert on["comm"] == {"enabled": True, "w": 4, "range": 6.0, "control": "none"}
    assert on["bonds"] == {"flow": True, "scale": 1000.0}
    assert on["versions"] == {"FRMA": 1, "STRC": 1, "FSIG": 1}
    with pytest.raises(sr.RecordError):
        sr.header_play(**{**kw, "records": ["FRMA", "XXXX"]})
    for bad in (dict(cap=0, cells=[1, 2, 2], cell=1.0), dict(cap=1, cells=[1, 2], cell=1.0)):
        with pytest.raises(sr.RecordError):
            sr.cells_block(**bad)
    with pytest.raises(sr.RecordError):
        sr.comm_block(w=9, hear_range=1.0, control="none")
    with pytest.raises(sr.RecordError):
        sr.comm_block(w=4, hear_range=1.0, control="mute")


def test_an_older_reader_skips_the_new_tags_by_length() -> None:
    """The forward rule of §33.1: a decoder that does not know a tag skips it by its length."""
    import play_decode

    rec = sr.encode_strc(step=1, seq=0, shape=(1, 3, 3), mat=np.eye(3, dtype=np.int32) * 100, cap=4096)
    stream = _header(8, ("FRMA", "STRC")) + rec + _frma(8, np.ones(8, bool))
    known = play_decode.KNOWN
    try:
        play_decode.KNOWN = tuple(t for t in known if t != "STRC")
        d = play_decode.decode(stream)
    finally:
        play_decode.KNOWN = known
    assert [t for t, _, _ in d.skipped] == ["STRC"] and [r.tag for r in d.records] == ["FRMA"]
    (length,) = struct.unpack_from("<I", rec, 4)
    assert len(rec) == 8 + length


def test_ext_tags_are_distinct_and_versioned() -> None:
    assert sr.EXT_TAGS == ("STRC", "FSIG", "FSOL") and not set(sr.EXT_TAGS) & set(sr.TAGS)
    assert dict.fromkeys(sr.EXT_TAGS, 1) == sr.EXT_VERSIONS
    seq = sr.Sequencer()
    assert [seq.take("STRC"), seq.take("STRC"), seq.take("FSIG")] == [0, 1, 0]
