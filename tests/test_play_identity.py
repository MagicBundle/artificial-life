"""``Observer.identity_rows`` and ``Observer.color_inputs`` (DESIGN §33.1.3, §33.2): the rows and tables of an
``IDNT`` record from a real observed run, on a 2D and a 3D world.

The tables are the columns of the last snapshot record, copied; before a record exists they are derived from
the rows. Both paths give the same keys, encode through ``streamrec`` and decode with the independent decoder.
The slot bytes of the rows are the observer's current ones; a row born after the last snapshot is flagged."""

from __future__ import annotations

import dataclasses
import json
from typing import Any, NamedTuple

import mlx.core as mx
import numpy as np
import pytest

from alife import layout, streamrec
from alife.config import Config, apply_overrides, dims, load_config
from alife.game.observer import FLAG_BORN_SINCE, IdentityRows, Observer
from alife.game.social import HOUSE_UNKNOWN, SOCIAL_V1
from alife.state import State, init_state, to_numpy
from alife.step import make_step
from game_helpers import planted_state
from play_decode import decode

SOC15 = dataclasses.replace(SOCIAL_V1, snapshot_every=15)
STEPS = (0, 20, 30, 33)  # fresh; between the first and second snapshot; a snapshot; just after


class Moment(NamedTuple):
    step: int
    rows: IdentityRows
    record: dict[str, Any] | None  # the newest record taken at this moment
    uid: np.ndarray  # the state's uid column
    clade: np.ndarray
    next_uid: int
    house: np.ndarray  # SocialLabels at this moment (host)
    kin: np.ndarray
    colony: np.ndarray
    inputs: Any  # ColorInputs, host copies of house and kin bytes


@pytest.fixture(scope="module", params=["tiny", "tiny3d"])
def world(request: pytest.FixtureRequest) -> tuple[Config, dict[int, Moment], Observer, State]:
    cfg = apply_overrides(
        load_config(preset=request.param, overrides=["brain.weight_dtype=float32"]),
        ["game.observe=true"],
    )
    s = planted_state(cfg)
    fn = make_step(cfg, observe=True)
    ob = Observer.fresh(s, cfg, social=SOC15)
    moments: dict[int, Moment] = {}
    last: dict[str, Any] | None = None

    def capture(t: int, state: State) -> None:
        sl = ob.social_labels()
        inp = ob.color_inputs(np.array([h["id"] for h in ob.identity_rows(state).tables["houses"]], np.int64))
        mx.eval(*inp)
        moments[t] = Moment(
            t,
            ob.identity_rows(state),
            last,
            to_numpy(state.uid),
            to_numpy(state.clade),
            int(to_numpy(state.next_uid)),
            to_numpy(sl.house),
            to_numpy(sl.kin_slot),
            to_numpy(sl.colony_slot),
            (to_numpy(inp.house), to_numpy(inp.kin)),
        )

    capture(0, s)
    for t in range(1, max(STEPS) + 1):
        s, out = fn(s)
        ob.update(s, out.acts, t)
        mx.eval(s, out.base(), *ob.arrays())
        recs = ob.take_records()
        last = recs[-1] if recs else last
        if t in STEPS:
            capture(t, s)
    return cfg, moments, ob, s


def _header(cfg: Config, n: int) -> bytes:
    d = dims(cfg)
    legacy = streamrec.legacy_header(
        dims=d.nd,
        size=float(cfg.world.size),
        height=float(cfg.world.height) if d.d3 else 0.0,
        n=n,
        bond_slots=layout.B,
        radius=float(cfg.physics.radius),
        field_cell=float(cfg.world.field_cell),
        fg=d.fg,
        fz=d.fz,
        nch=layout.NCH,
        step=0,
        run=None,
        config_hash="0" * 64,
        source="preset",
    )
    play = streamrec.header_play(
        records=["IDNT"],
        weather=streamrec.weather_block(enabled=False, dc_ref=None),
        labels={"intent_version": 2, "arch": [""], "intent": [""]},
        energy_cap=100,
    )
    return streamrec.play_header_bytes(legacy, play)


def _encode(cfg: Config, m: Moment) -> bytes:
    r = m.rows
    return streamrec.encode_idnt(
        step=m.step,
        seq=0,
        snapshot_step=r.snapshot_step,
        uid=r.uid,
        clade=r.clade,
        house=r.house,
        kin=r.kin,
        colony=r.colony,
        flags=r.flags,
        tables=r.tables,
    )


KEYS = ("houses", "exact", "origin", "kin", "colonies", "unknown_immigrant")


def test_rows_are_the_alive_rows_in_row_order(world) -> None:
    _, moments, _, _ = world
    for t, m in moments.items():
        r = m.rows
        alive = np.flatnonzero(m.uid >= 0)
        assert np.array_equal(r.row, alive) and np.array_equal(r.uid, m.uid[alive]), t
        assert np.array_equal(r.clade, m.clade[alive]) and r.clade.dtype == np.int32, t
        assert all(
            a.dtype == np.uint8 and a.shape == alive.shape for a in (r.house, r.kin, r.colony, r.flags)
        )
        assert set(r.tables) == set(KEYS), t
        assert alive.size > 20


def test_the_slots_of_the_rows_are_the_observers(world) -> None:
    _, moments, _, _ = world
    f = SOC15.slots
    for t, m in moments.items():
        r, rows = m.rows, m.rows.row
        kin, col = m.kin[rows], m.colony[rows]
        assert np.array_equal(r.kin, np.where(kin < 0, 255, np.where(kin >= f, 254, kin))), t
        assert np.array_equal(r.colony, np.where(col < 0, 255, np.where(col >= f, 254, col))), t
        by_slot = {h["slot"]: h["id"] for h in r.tables["houses"]}
        for byte, hid in zip(r.house.tolist(), m.house[rows].tolist(), strict=True):
            if hid == HOUSE_UNKNOWN:
                assert byte == 253
            elif byte < 16:
                assert by_slot[byte] == hid
            else:
                assert byte == 254 and hid not in by_slot.values()
        # the colour inputs are the same bytes, computed on the device
        assert np.array_equal(m.inputs[0][rows], r.house) and np.array_equal(m.inputs[1][rows], r.kin), t


def test_the_tables_are_the_records_columns_copied(world) -> None:
    cfg, moments, _, _ = world
    for t in (20, 30, 33):
        m = moments[t]
        rec, tab = m.record, m.rows.tables
        assert rec is not None and m.rows.snapshot_step == rec["step"] == (15 if t == 20 else 30)
        h = rec["houses"]
        assert [(e["slot"], e["id"], e["name"], e["size"], e["clade"]) for e in tab["houses"]] == [
            (i, *row) for i, row in enumerate(zip(h["id"], h["name"], h["size"], h["clade"], strict=True))
        ]
        k = rec["kin_groups"]
        assert len(tab["kin"]) == len(k["id"]) >= 2, "the planted world tracks kin groups"
        for i, e in enumerate(tab["kin"]):
            assert e["slot"] == k["slot"][i] and e["id"] == k["id"][i] and e["name"] == k["name"][i]
            assert e["named_after"] == k["named_after"][i] and e["size"] == k["size"][i]
            assert e["clade"] == k["clade"][i] and e["clade_share"] == k["clade_share"][i]
            assert (e["x"], e["y"]) == (k["x"][i], k["y"][i]) and e["sim_self"] == k["sim_self"][i]
            assert (
                e["arch"] == k["arch"][i] and e["births"] == k["births"][i] and e["deaths"] == k["deaths"][i]
            )
            assert ("z" in e) == dims(cfg).d3 and (not dims(cfg).d3 or e["z"] == k["z_mean"][i])
        c = rec["colonies"]
        assert [(e["slot"], e["id"], e["size"], e["kin"], e["x"], e["y"]) for e in tab["colonies"]] == list(
            zip(c["slot"], c["id"], c["size"], c["kin_group"], c["x"], c["y"], strict=True)
        )
        assert tab["exact"] is (rec["origin"] == 0) and tab["origin"] == rec["origin"]
        assert tab["unknown_immigrant"] == int((m.house[m.rows.row] == HOUSE_UNKNOWN).sum())


def test_before_any_record_the_tables_come_from_the_rows(world) -> None:
    cfg, moments, _, _ = world
    m = moments[0]
    assert m.record is None
    tab, r = m.rows.tables, m.rows
    assert 2 <= len(tab["kin"]) <= 16 and len(tab["houses"]) == min(16, len(np.unique(m.house[r.row])))
    for e in tab["kin"]:
        assert e["size"] == int((r.kin == e["slot"]).sum()) > 0
        assert e["sim_self"] is None and e["arch"] is None and e["births"] is None and e["deaths"] is None
        assert 0 < e["clade_share"] <= 1 and ("z" in e) == dims(cfg).d3
        assert e["named_after"] is None or isinstance(e["named_after"], int)
    # the same key sets as the record path
    later = moments[30].rows.tables
    assert {frozenset(e) for e in tab["kin"]} == {frozenset(e) for e in later["kin"]}
    assert {frozenset(e) for e in tab["colonies"]} == {frozenset(e) for e in later["colonies"]} or not tab[
        "colonies"
    ]
    assert {frozenset(e) for e in tab["houses"]} == {frozenset(e) for e in later["houses"]}
    assert tab["exact"] is True and tab["origin"] == 0  # the observer started at step 0: exact houses
    for e in tab["colonies"]:
        assert e["size"] == int((r.colony == e["slot"]).sum()) and e["kin"] >= -1


def test_born_since_the_last_snapshot_is_flagged(world) -> None:
    _, moments, _, _ = world
    at30, at33 = moments[30], moments[33]
    assert not at30.rows.flags.any(), "no row is newer than the snapshot at the snapshot step"
    want = at33.uid[at33.rows.row] >= at30.next_uid
    assert want.any() and np.array_equal(at33.rows.flags == FLAG_BORN_SINCE, want)
    assert np.array_equal(moments[0].rows.flags, np.zeros_like(moments[0].rows.flags))


def test_every_moment_encodes_and_decodes(world) -> None:
    cfg, moments, _, _ = world
    n = int(to_numpy(init_state(cfg).alive).shape[0])
    for t, m in moments.items():
        data = _header(cfg, n) + _encode(cfg, m)
        rec = decode(data).records[0]
        assert rec.tag == "IDNT" and rec.step == t and rec.f["snapshot_step"] == m.rows.snapshot_step
        assert json.dumps(rec.f["json"], sort_keys=True) == json.dumps(m.rows.tables, sort_keys=True), t
        np.testing.assert_array_equal(rec.a["uid"], m.rows.uid)
        np.testing.assert_array_equal(rec.a["kin"], m.rows.kin)
        np.testing.assert_array_equal(rec.a["house"], m.rows.house)


def test_reading_the_identity_changes_nothing_and_checks_the_step(world) -> None:
    _, _, ob, s = world
    before = {k: v.copy() for k, v in ob.sidecar().items() if k != "meta"}
    first = ob.identity_rows(s)
    again = ob.identity_rows(s)
    after = {k: v for k, v in ob.sidecar().items() if k != "meta"}
    assert before.keys() == after.keys() and all(np.array_equal(before[k], after[k]) for k in before)
    assert json.dumps(first.tables) == json.dumps(again.tables)
    stale = s._replace(step=s.step + 1)
    with pytest.raises(ValueError, match="identity_rows"):
        ob.identity_rows(stale)
