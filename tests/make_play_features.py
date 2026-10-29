"""The feature-layer play stream of `tests/data/play_features.bin`, written with the real encoders.

A 3D world (L 48, height 24, field cell 4: FG 12, FZ 6, N 8) with all five feature layers on: the header
keys `play.structures`, `play.comm`, `play.niche` and `play.bonds`, the `role` colour mode, one `FRMA` with
the bond flow, and one each of `STRC`, `FSOL` and `FSIG`. It is the Swift decoder's check against the writer.
Needs the observer series' `streamrec` (`encode_strc` and the rest); `tests/test_gui_sync.py` compares the
committed file with `build()` once the writer is on the branch.
Regenerate: `python tests/make_play_features.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from alife import streamrec as sr

OUT = Path(__file__).resolve().parent / "data" / "play_features.bin"


def build() -> bytes:
    fg, fz, n = 12, 6, 8
    modes = (*sr.COLOR_MODES, *sr.EXT_COLOR_MODES)
    legacy = sr.legacy_header(
        dims=3,
        size=48.0,
        height=24.0,
        n=n,
        bond_slots=4,
        radius=1.0,
        field_cell=4.0,
        fg=fg,
        fz=fz,
        nch=3,
        step=100,
        run=None,
        config_hash="0" * 64,
        source="preset",
        color_modes=modes,
    )
    labels = sr.labels_block(
        intent_version=3,
        arch=["", "Gatherer"],
        intent=["", "Budding"],
        gate={
            "rules_digest": "d" * 64,
            "world_key": "0" * 64,
            "key_found": False,
            "archetypes": [],
            "intents": [],
        },
    )
    play = sr.header_play(
        records=["FRMA", "FOCS", "STRC", "FSIG", "FSOL"],
        weather=sr.weather_block(enabled=False, dc_ref=None),
        labels=labels,
        energy_cap=16384,
        structures=sr.cells_block(cap=4096, cells=[fz, fg, fg], cell=4.0),
        comm=sr.comm_block(w=3, hear_range=8.0, control="none"),
        niche=sr.cells_block(cap=4095, cells=[fz, fg, fg], cell=4.0, farm=True),
        bonds=sr.bonds_block(scale=1000.0),
    )
    alive = np.zeros(n, bool)
    alive[[1, 2, 5]] = True
    pos = np.zeros((n, 3), np.float32)
    pos[[1, 2, 5]] = [[6, 6, 6], [10, 6, 6], [30, 30, 12]]
    orient = np.zeros((n, 2), np.float32)
    rgb = np.zeros(n, np.uint32)
    rgb[[1, 2, 5]] = [0xF0E442, 0x0072B2, 0xBEBEBE]
    uid = np.zeros(n, np.int64)
    uid[[1, 2, 5]] = [10, 20, 30]
    frma = sr.encode_frma(
        step=100,
        seq=0,
        alive=alive,
        pos=pos,
        orient=orient,
        rgb=rgb,
        uid=uid,
        bonds=np.array([[1, 2], [2, 5]]),
        color_index=modes.index("role"),
        flow=np.array([-100, 33], np.int8),
    )
    mat = np.zeros((fz, fg, fg), np.int32)
    mat[0, 1, 1] = 4096
    mat[2, 5, 7] = 1000
    soil = np.zeros((fz, fg, fg), np.int32)
    soil[0, 3, 3] = 4095
    parts = [
        sr.play_header_bytes(legacy, play),
        frma,
        sr.encode_strc(step=100, seq=0, shape=(fz, fg, fg), mat=mat, cap=4096),
        sr.encode_fsol(step=100, seq=0, shape=(fz, fg, fg), soil=soil, cap=4095),
        sr.encode_fsig(
            step=100,
            seq=0,
            uid=10,
            says=np.array([255, 0, 128], np.uint8),
            hears=np.array([10, 20, 30], np.uint8),
            audible=2,
        ),
    ]
    return b"".join(parts)


if __name__ == "__main__":
    OUT.write_bytes(build())
    print("wrote", OUT, OUT.stat().st_size, "bytes", file=sys.stderr)
    sys.exit(0)
