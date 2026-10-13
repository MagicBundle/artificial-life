"""A reference decoder of the ``alife stream`` protocol, written from DESIGN §27.11, not from the writer.

Every constant and offset is spelled out here on purpose: the tests then check the writer against the text of
the specification, and the same layout is what PR 5's Swift decoder implements. ``decode`` is strict: it
asserts the magic, the 4-byte alignment and zero padding, the head's reserved bytes, every section length
and that the records account for every byte of the input.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

MAGIC = b"ALSTRM\x00\x01"
HEADER_KEYS = (
    "protocol",
    "dims",
    "L",
    "height",
    "N",
    "B",
    "radius",
    "field_cell",
    "FG",
    "FZ",
    "NCH",
    "color_modes",
    "color",
    "fields",
    "field_z",
    "step",
    "run",
    "config_hash",
    "mode",
    "every",
    "source",
)


@dataclass
class Frame:
    offset: int  # of the "FRM1" tag in the input
    length: int  # the u32 length field: bytes after the 8-byte record prefix
    step: int
    n_alive: int
    flags: int
    n_bonds: int
    color_index: int
    reserved: bytes
    alive: np.ndarray  # (N,) bool
    pos: np.ndarray  # (N, 3) float32
    orient: np.ndarray  # (N, 2) float16: yaw, pitch
    rgb: np.ndarray  # (N,) uint32, 0x00RRGGBB
    bonds: np.ndarray  # (n_bonds, 2) uint32
    field: np.ndarray | None  # (FG*FG,) uint32 when flags bit 0 is set
    sections: dict[str, bytes]  # the raw bytes of each section, padding included

    @property
    def field_layer(self) -> int | None:
        return self.flags >> 16 if self.flags & 1 else None


def _section(payload: bytes, off: int, size: int, name: str) -> tuple[bytes, int]:
    assert off + size <= len(payload), f"{name} section runs past the frame"
    return payload[off : off + size], off + size


def decode_header(data: bytes) -> tuple[dict[str, Any], int]:
    """``(header object, offset of the first record)``."""
    assert data[:8] == MAGIC, data[:8]
    (json_len,) = struct.unpack_from("<I", data, 8)
    assert json_len % 4 == 0 and len(data) >= 12 + json_len
    text = data[12 : 12 + json_len]
    assert len(text) - len(text.rstrip(b" ")) < 4, "the JSON is padded with fewer than 4 spaces"
    return json.loads(text.decode("ascii")), 12 + json_len


def decode(data: bytes) -> tuple[dict[str, Any], list[Frame]]:
    """The header object and the frames of a whole stream; unknown record tags are skipped."""
    header, pos = decode_header(data)
    n, fg = header["N"], header["FG"]
    frames: list[Frame] = []
    while pos < len(data):
        assert pos + 8 <= len(data), "truncated record prefix"
        tag = data[pos : pos + 4]
        (length,) = struct.unpack_from("<I", data, pos + 4)
        assert pos + 8 + length <= len(data), "truncated record"
        payload = data[pos + 8 : pos + 8 + length]
        if tag == b"FRM1":
            frames.append(_frame(payload, n, fg, pos, length))
        pos += 8 + length
    return header, frames


def _frame(payload: bytes, n: int, fg: int, offset: int, length: int) -> Frame:
    assert len(payload) >= 32
    step, n_alive, flags, n_bonds, color_index = struct.unpack_from("<QIIII", payload, 0)
    reserved = payload[24:32]
    assert reserved == b"\x00" * 8
    off = 32
    sections: dict[str, bytes] = {}
    sections["alive"], off = _section(payload, off, (n + 3) // 4 * 4, "alive")
    assert sections["alive"][n:] == b"\x00" * (len(sections["alive"]) - n)
    sections["pos"], off = _section(payload, off, 12 * n, "pos")
    sections["orient"], off = _section(payload, off, 4 * n, "orient")
    sections["rgb"], off = _section(payload, off, 4 * n, "rgb")
    sections["bonds"], off = _section(payload, off, 8 * n_bonds, "bonds")
    field = None
    if flags & 1:
        sections["field"], off = _section(payload, off, 4 * fg * fg, "field")
        field = np.frombuffer(sections["field"], "<u4").copy()
    assert off == len(payload), f"{len(payload) - off} unexplained bytes in a frame"
    alive = np.frombuffer(sections["alive"], np.uint8, n)
    assert set(np.unique(alive)) <= {0, 1}
    return Frame(
        offset=offset,
        length=length,
        step=step,
        n_alive=n_alive,
        flags=flags,
        n_bonds=n_bonds,
        color_index=color_index,
        reserved=reserved,
        alive=alive.astype(bool),
        pos=np.frombuffer(sections["pos"], "<f4").reshape(n, 3).copy(),
        orient=np.frombuffer(sections["orient"], "<f2").reshape(n, 2).copy(),
        rgb=np.frombuffer(sections["rgb"], "<u4").copy(),
        bonds=np.frombuffer(sections["bonds"], "<u4").reshape(n_bonds, 2).copy(),
        field=field,
        sections=sections,
    )
