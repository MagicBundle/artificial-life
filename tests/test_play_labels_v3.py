"""The play stream carries the version-3 labels (DESIGN §33.7): ten archetypes (Drifter, Floater and Sinker
are bytes 8 to 10), the storm names of a weather world, and the held-out verdict gate N1 applies (§31,
§32.13).

The gate is ``report.label_gate``: the one predicate (``label_passes``) that N1's label facts and per-fact
check also read. A version-3 world has no row in ``LABEL_VERDICTS`` until W2b's held-out windows add one, so
its header lists no valid label; a row added to the table is read by the header and by N1 alike."""

from __future__ import annotations

import dataclasses
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alife import play, streamrec
from alife.config import apply_overrides
from alife.game import observer, report
from alife.game.colors import ARCH_PALETTE, NONE_RGB
from alife.game.rules import (
    ARCHETYPE_DISPLAY,
    ARCHETYPE_NAMES,
    ARCHETYPE_STORM_DISPLAY,
    INTENT_DISPLAY,
    INTENT_NAMES,
    INTENT_VERSION,
    RULES_V3,
    Archetype,
    Intent,
    for_2d,
)
from alife.game.social import SOCIAL_V1
from play_decode import decode
from play_helpers import run_paced, snapshots, world

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _header(tmp_path: Path, kind: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    snapshots(monkeypatch, 10)
    out = io.BytesIO()
    cfg = world(kind)
    run_paced(cfg, tmp_path / kind, 3, [[]], out=out)
    return decode(out.getvalue()).header["play"]["labels"]


def test_the_names_are_indexed_by_the_byte_and_are_the_display_tables() -> None:
    arch, intent = report.label_names(INTENT_VERSION, storm=False)
    assert INTENT_VERSION == 3 and len(arch) == len(Archetype) == 11 and len(intent) == len(Intent) == 11
    assert arch[0] == intent[0] == ""
    for a in Archetype:
        if a is not Archetype.NONE:
            assert arch[a.value] == ARCHETYPE_DISPLAY[a.name.lower()]
    for i in Intent:
        if i is not Intent.NONE:
            assert intent[i.value] == INTENT_DISPLAY[i.name.lower()]
    assert arch[8:] == ["Drifter", "Floater", "Sinker"]


def test_a_storm_world_says_grazer_and_nothing_else_changes() -> None:
    plain, _ = report.label_names(INTENT_VERSION, storm=False)
    storm, intent = report.label_names(INTENT_VERSION, storm=True)
    assert storm[1] == "Grazer" == ARCHETYPE_STORM_DISPLAY["gatherer"] and plain[1] == "Gatherer"
    assert [s for s, p in zip(storm, plain, strict=True) if s != p] == ["Grazer"]
    assert intent == report.label_names(INTENT_VERSION, storm=False)[1]
    # N1 prints the same word for the same label, in the same two cases
    assert report._label_display("gatherer", 3, True) == "Grazer"
    assert report._label_display("gatherer", 3, False) == "Gatherer"


@pytest.mark.parametrize("kind,first", [("2d", "Gatherer"), ("storm", "Grazer")])
def test_the_header_of_a_played_world_has_the_version_3_names_and_a_closed_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, first: str
) -> None:
    labels = _header(tmp_path, kind, monkeypatch)
    cfg = world(kind)
    assert labels["intent_version"] == 3
    assert (
        labels["arch"][0] == ""
        and labels["arch"][1] == first
        and labels["arch"][8:]
        == [
            "Drifter",
            "Floater",
            "Sinker",
        ]
    )
    assert len(labels["arch"]) == 11 and len(labels["intent"]) == 11
    gate = labels["gate"]
    rules = RULES_V3 if kind == "storm" else for_2d(RULES_V3)  # a 2D world has no vertical labels
    assert gate == {
        "rules_digest": rules.digest(),
        "world_key": report.world_key(cfg),
        "key_found": False,
        "archetypes": [],
        "intents": [],
    }, "no version-3 world has a held-out verdict yet: no label is valid"
    assert (rules.digest() == RULES_V3.digest()) == (kind == "storm")


def test_the_gate_reads_the_digest_of_the_observers_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = dataclasses.replace(RULES_V3, arch_warm=RULES_V3.arch_warm + 1)
    monkeypatch.setattr(observer, "DEFAULT_RULES", other)
    labels = _header(tmp_path, "storm", monkeypatch)
    assert labels["gate"]["rules_digest"] == other.digest() != RULES_V3.digest()


# --- the gate against a table with rows ------------------------------------------------------------------


def _table(cfg: Any, rows: dict[str, tuple[str, int, float, float, int]], version: int = 3) -> dict[Any, Any]:
    digest = (RULES_V3 if cfg.world.dims == 3 else for_2d(RULES_V3)).digest()
    return {(version, digest, cfg.world.dims, report.world_key(cfg)): rows}


PASS = ("pass", 5000, 0.30, 1 / 201, 200)


def test_a_row_that_passes_lists_its_bytes_and_a_failing_one_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = world("storm")
    rows = {
        "hunter": PASS,  # archetype 2
        "drifter": PASS,  # archetype 8
        "floater": ("insufficient", 20, 0.4, 1 / 201, 200),
        "sinker": ("fail", 9000, -0.1, 1.0, 200),
        "resting": PASS,  # intent 10
        "budding": ("pass", 99, 0.3, 1 / 201, 200),  # too few positives
        "roaming": ("pass", 5000, 0.09, 1 / 201, 200),  # margin below the report's bar
        "fleeing": ("pass", 5000, 0.30, 0.05, 200),  # not the smallest permutation p
    }
    monkeypatch.setattr(report, "LABEL_VERDICTS", _table(cfg, rows))
    gate = report.label_gate(cfg, 3, RULES_V3.digest())
    assert gate["key_found"] is True and gate["archetypes"] == [2, 8] and gate["intents"] == [10]
    # the same predicate decides every row, for the header and for N1's label facts
    assert {k for k, r in rows.items() if report.label_passes(r, report.REPORT_V1)} == {
        "hunter",
        "drifter",
        "resting",
    }
    # the block encodes and the client's list is the writer's
    arch, intent = report.label_names(3, True)
    block = streamrec.labels_block(intent_version=3, arch=arch, intent=intent, gate=gate)
    assert block["gate"]["archetypes"] == [2, 8] and block["arch"][8] == "Drifter"


def test_another_world_or_rule_set_or_dimension_has_no_row(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = world("storm")
    monkeypatch.setattr(report, "LABEL_VERDICTS", _table(cfg, {"hunter": PASS}))
    assert report.label_gate(cfg, 3, RULES_V3.digest())["key_found"]
    assert not report.label_gate(cfg, 2, RULES_V3.digest())["key_found"], "another rule set version"
    assert not report.label_gate(cfg, 3, "0" * 64)["key_found"], "another rule digest"
    other = apply_overrides(cfg, ["weather.rain_gain=0.0123"])
    assert report.world_key(other) != report.world_key(cfg)
    assert not report.label_gate(other, 3, RULES_V3.digest())["key_found"], "another world"
    # the director and game sections are not part of the key (N1's world key)
    same = apply_overrides(cfg, ["game.tracers=16"])
    assert report.world_key(same) == report.world_key(cfg)


def test_the_shipped_table_has_no_version_3_row_and_no_version_2_one() -> None:
    assert {k[0] for k in report.LABEL_VERDICTS} == {1}, (
        "V2 and V3 narrate no label until their windows exist"
    )
    for kind in ("2d", "storm"):
        assert not report.label_gate(world(kind), INTENT_VERSION, RULES_V3.digest())["key_found"]


# --- the encoder -------------------------------------------------------------------------------------------


def _gate(**kw: Any) -> dict[str, Any]:
    return {
        "rules_digest": "d",
        "world_key": "k",
        "key_found": True,
        "archetypes": [1],
        "intents": [1],
        **kw,
    }


def test_labels_block_refuses_what_a_client_cannot_read() -> None:
    arch, intent = ["", "A", "B"], ["", "I"]
    ok = streamrec.labels_block(intent_version=3, arch=arch, intent=intent, gate=_gate())
    assert json.loads(json.dumps(ok)) == ok and set(ok) == {"intent_version", "arch", "intent", "gate"}
    for bad in (
        _gate(archetypes=[3]),  # past the table
        _gate(archetypes=[0]),  # none is never valid
        _gate(archetypes=[2, 1]),  # not ascending
        _gate(intents=[1, 1]),
        _gate(intents=[True]),
        _gate(key_found=False),  # a list without a row
    ):
        with pytest.raises(streamrec.RecordError):
            streamrec.labels_block(intent_version=3, arch=arch, intent=intent, gate=bad)
    with pytest.raises(streamrec.RecordError):
        streamrec.labels_block(intent_version=3, arch=["x"], intent=intent, gate=_gate())
    with pytest.raises(streamrec.RecordError):
        streamrec.labels_block(intent_version=3, arch=arch, intent=intent, gate={"key_found": False})


def test_the_bytes_of_the_storm_archetypes_travel_in_intn_and_have_colours() -> None:
    uid = np.arange(3, dtype=np.int64)
    rec = streamrec.encode_intn(
        step=5,
        seq=0,
        labels_step=4,
        intent_version=3,
        uid=uid,
        arch_primary=np.array([8, 9, 10], np.uint8),
        arch_secondary=np.array([0, 8, 0], np.uint8),
        intent=np.array([10, 9, 1], np.uint8),
        flags=np.array([1, 1, 3], np.uint8),
    )
    got = decode(_stream(rec)).records[0]
    assert got.a["arch_primary"].tolist() == [8, 9, 10] and got.f["intent_version"] == 3
    assert len(ARCH_PALETTE) == 11 and len(set(ARCH_PALETTE[1:])) == 10 and ARCH_PALETTE[8:] != [NONE_RGB] * 3


def _stream(*records: bytes) -> bytes:
    cfg = world("2d")
    from alife import layout
    from alife.config import dims

    d = dims(cfg)
    legacy = streamrec.legacy_header(
        dims=d.nd, size=cfg.world.size, height=0.0, n=d.n, bond_slots=layout.B, radius=cfg.physics.radius,
        field_cell=cfg.world.field_cell, fg=d.fg, fz=d.fz, nch=layout.NCH, step=0, run=None,
        config_hash="0" * 64, source="preset",
    )  # fmt: skip
    play_key = streamrec.header_play(
        records=["INTN"],
        weather=streamrec.weather_block(enabled=False, dc_ref=None),
        labels=streamrec.labels_block(
            intent_version=3,
            arch=["", *ARCHETYPE_NAMES],
            intent=["", *INTENT_NAMES],
            gate=_gate(archetypes=[], intents=[], key_found=False),
        ),
        energy_cap=100,
    )
    return streamrec.play_header_bytes(legacy, play_key) + b"".join(records)


def test_the_widest_idnt_table_with_ten_shares_a_group_fits_its_cap() -> None:
    """The kin groups' archetype mix is one share per archetype (10 in version 3, 7 before): sixteen groups of
    full-precision shares, ids and names at their widest still fit the 16 KiB of an ``IDNT`` table."""
    rng = np.random.default_rng(7)
    kin = [
        {
            "slot": s,
            "id": 2**62 + s,
            "name": "Wwwwwwwww kin",
            "named_after": s,
            "size": 40_000,
            "clade": 2**31 - 1,
            "clade_share": float(rng.random()),
            "sim_self": float(rng.random()),
            "x": float(rng.random() * 1000),
            "y": float(rng.random() * 1000),
            "z": float(rng.random() * 1000),
            "arch": [float(v) for v in rng.random(len(ARCHETYPE_NAMES))],
            "births": 10**9,
            "deaths": [10**9] * 5,
        }
        for s in range(16)
    ]
    houses = [
        {"slot": s, "id": 2**62 + s, "name": "Wwwwwwwww", "size": 40_000, "clade": 2**31 - 1}
        for s in range(16)
    ]
    colonies = [
        {"slot": s, "id": 2**62 + s, "size": 40_000, "kin": 2**62, "x": 999.123456789, "y": 999.123456789}
        for s in range(16)
    ]
    tables = {
        "houses": houses,
        "exact": True,
        "origin": 0,
        "kin": kin,
        "colonies": colonies,
        "unknown_immigrant": 0,
    }
    assert len(kin[0]["arch"]) == 10
    size = len(streamrec.json_bytes(tables))
    assert size <= streamrec.IDNT_JSON_CAP * 0.75, size
    n = 3
    streamrec.encode_idnt(
        step=1, seq=0, snapshot_step=0, uid=np.arange(n, dtype=np.int64), clade=np.zeros(n, np.int32),
        house=np.zeros(n, np.uint8), kin=np.zeros(n, np.uint8), colony=np.zeros(n, np.uint8),
        flags=np.zeros(n, np.uint8), tables=tables,
    )  # fmt: skip


def test_default_social_is_unchanged_by_the_labels() -> None:
    assert observer.DEFAULT_SOCIAL is SOCIAL_V1 and play.PLAY_TICK > 0
