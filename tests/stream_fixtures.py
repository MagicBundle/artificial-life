"""The `alife stream` fixtures for PR 5's Swift decoder: `tests/data/stream_tiny_{2d,3d}.bin` and sidecars.

Each fixture is a complete stream (header and three frames) of a hand-built scene on a small `tiny` / `tiny3d`
config (N = 64, L = 48), written by the real `Streamer`. The frames cover a bond across the periodic seam, a
hurt (red) agent, dead rows between alive ones, the three colour modes, a field slice (in 3D at layers 0
and 5) and the invariant-violation flag. The scenes are built, not stepped, so the bytes do not depend on the
platform's floating-point dynamics. The JSON sidecar holds the expected header object and, per frame, the head
fields, a SHA-256 of every section, the alive rows, the bonds and the first agents, decoded by `stream_decode`
(an independent decoder written from DESIGN §27.11).

Regenerate after a deliberate protocol or palette change, on the CPU device:

    uv run python tests/stream_fixtures.py --write

`tests/test_stream.py::test_fixtures_match_the_code` rebuilds both and compares them byte for byte on the CPU
device (the colour arithmetic is float32, and a GPU may round a ramp differently); the committed files are
decoded and checked against their sidecars on any device.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx

# As a script: the CPU device before any array exists (the evaluation arms own the GPU).
if __name__ == "__main__":
    mx.set_default_device(mx.cpu)

import numpy as np

from alife.config import Config, load_config
from alife.stream import Streamer, View
from stream_decode import decode

DATA = Path(__file__).resolve().parent / "data"
KINDS = ("2d", "3d")
OVERRIDES = [
    "population.n_max=64",
    "population.max_births_per_step=8",
    "population.n_init=8",
    "world.size=48",
    "brain.weight_dtype=float32",
]  # tests/helpers.MICRO3D, on either preset
# row, (x, y), z, yaw, pitch, tag, energy, clade, last_loss, how far the scene variants move it (dx, dy)
_AGENTS = (
    (0, (1.0, 24.0), 6.0, 0.5, 0.5, (1.0, -1.0, 0.5, 0.0), 4000, 3, 0, (1, 0)),
    (1, (47.0, 24.0), 7.0, -1.25, -0.25, (-1.0, 1.0, 0.0, 0.0), 1000, 3, 0, (1, 0)),
    (2, (10.0, 10.0), 2.0, 3.0, 0.0, (0.25, 0.25, 0.25, 0.0), 500, 0, 1, (0, 1)),
    (3, (12.0, 11.5), 3.5, -3.0, 1.0, (-0.5, 0.5, 1.0, 0.0), 8000, 5, 0, (0, 1)),
    (4, (30.0, 40.0), 12.0, 1.0, -1.25, (0.0, 0.0, 0.0, 0.0), 2500, 9, 0, (0, 0)),
    (10, (31.0, 41.0), 12.5, 2.0, 0.75, (1.0, 1.0, 1.0, 0.0), 100, 1, 0, (0, 1)),
    (21, (40.0, 5.0), 9.0, 0.0, 0.125, (-1.0, -1.0, -1.0, 0.0), 6000, 2, 0, (0, 0)),
)


def fixture_config(kind: str) -> tuple[Config, dict[str, Any]]:
    """`(cfg, {"preset": ..., "overrides": [...]})` of fixture `kind`."""
    spec = {"preset": "tiny3d" if kind == "3d" else "tiny", "overrides": OVERRIDES}
    return load_config(preset=spec["preset"], overrides=spec["overrides"]), spec


def scene_agents(kind: str, variant: int) -> list[dict[str, Any]]:
    """The agents of scene `variant` (0..2) as `make_scene` specs; variant 2 has lost the agent in row 4."""
    out = []
    for row, (x, y), z, yaw, pitch, tag, energy, clade, loss, (sx, sy) in _AGENTS:
        if variant == 2 and row == 4:
            continue
        d = 0.25 * variant
        spec: dict[str, Any] = {
            "row": row,
            "pos": (x + sx * d, y + sy * d, z) if kind == "3d" else (x + sx * d, y + sy * d),
            "heading": yaw,
            "tag": tag,
            "energy": energy,
            "clade": clade,
            "last_loss": loss,
        }
        if kind == "3d":
            spec["pitch"] = pitch
        out.append(spec)
    return out


def fixture_scene(kind: str, cfg: Config, variant: int):
    from helpers import make_scene  # not at import: helpers reads the default device (ON_GPU)

    shape = (6, 12, 12) if kind == "3d" else (12, 12)
    resource = (np.arange(math.prod(shape)).reshape(shape) * 13 % 9000).astype(np.int32)
    chem = (np.arange(3 * math.prod(shape)).reshape((3, *shape)) % 7 * 0.5).astype(np.float32)
    return make_scene(
        cfg,
        scene_agents(kind, variant),
        bonds=[(0, 1), (2, 3), *([(4, 10)] if variant == 1 else [])],
        resource=resource,
        chem=chem,
        step=mx.array(100 + 4 * variant, dtype=mx.int32),
        diag=mx.array(1 if variant == 2 else 0, dtype=mx.int32),
    )


def _energy_margin_ok(cfg: Config, kind: str) -> bool:
    """The `energy` colour rounds a float that depends on `log1p`; require every fixture energy to be far
    from a rounding boundary, so a 1 ulp difference between platforms cannot change a byte. (The tag and
    field colours of the fixture are exact in float32 or plain IEEE arithmetic.)"""
    ramp = ((0.0, (30, 60, 200)), (0.5, (60, 200, 120)), (1.0, (255, 220, 40)))
    for spec in scene_agents(kind, 0):
        t = min(max(math.log1p(spec["energy"]) / math.log1p(cfg.energy.e_max), 0.0), 1.0)
        rgb = list(ramp[0][1])
        (prev_t, prev_c), *rest = ramp
        for ti, ci in rest:
            frac = min(max((t - prev_t) / (ti - prev_t), 0.0), 1.0)
            rgb = [v + frac * (b - a) for v, a, b in zip(rgb, prev_c, ci, strict=True)]
            prev_t, prev_c = ti, ci
        if any(abs(v % 1.0 - 0.5) < 1e-3 for v in rgb):
            return False
    return True


def build(kind: str) -> tuple[bytes, dict[str, Any]]:
    """`(stream bytes, sidecar object)` of fixture `kind`."""
    cfg, spec = fixture_config(kind)
    assert _energy_margin_ok(cfg, kind), "an energy colour sits on a rounding boundary: change the energies"
    buf = io.BytesIO()
    sm = Streamer(cfg, buf, mode="simulate", source="preset", run=None, view=View(), every=4)
    s0, s1, s2 = (fixture_scene(kind, cfg, v) for v in range(3))
    sm.begin(s0)
    sm.send(s0)
    sm.view.color, sm.view.fields, sm.view.field_z = "energy", True, 0
    sm.send(s1)
    sm.view.color, sm.view.field_z = "clade", 5 if kind == "3d" else 0
    sm.send(s2)
    data = buf.getvalue()
    return data, sidecar(kind, spec, data)


def sidecar(kind: str, spec: dict[str, Any], data: bytes) -> dict[str, Any]:
    """The expected decoding of `data` (by the independent decoder)."""
    header, frames = decode(data)
    out_frames = []
    for i, fr in enumerate(frames):
        agents = [
            {
                "row": int(row),
                "pos": [float(v) for v in fr.pos[row]],
                "yaw": float(fr.orient[row, 0]),
                "pitch": float(fr.orient[row, 1]),
                "rgb": f"#{int(fr.rgb[row]):06x}",
            }
            for row in np.flatnonzero(fr.alive)[:4]
        ]
        out_frames.append(
            {
                "index": i,
                "offset": fr.offset,
                "length": fr.length,
                "step": fr.step,
                "n_alive": fr.n_alive,
                "flags": fr.flags,
                "n_bonds": fr.n_bonds,
                "color_index": fr.color_index,
                "color": header["color_modes"][fr.color_index],
                "field_layer": fr.field_layer,
                "diag_error": bool(fr.flags & 2),
                "alive_rows": [int(r) for r in np.flatnonzero(fr.alive)],
                "bonds": fr.bonds.tolist(),
                "agents": agents,
                "field_first": [f"#{int(v):06x}" for v in fr.field[:6]] if fr.field is not None else None,
                "sha256": {k: hashlib.sha256(v).hexdigest() for k, v in fr.sections.items()},
            }
        )
    return {
        "about": "Expected decoding of the stream in the .bin file next to this one (DESIGN section 27.11). "
        "Regenerate with `uv run python tests/stream_fixtures.py --write`.",
        "kind": kind,
        "config": spec,
        "bytes": len(data),
        "header": header,
        "frames": out_frames,
    }


def sidecar_text(obj: dict[str, Any]) -> str:
    return json.dumps(obj, indent=2) + "\n"


def paths(kind: str) -> tuple[Path, Path]:
    return DATA / f"stream_tiny_{kind}.bin", DATA / f"stream_tiny_{kind}.json"


def main(argv: list[str]) -> int:
    write = "--write" in argv
    for kind in KINDS:
        data, side = build(kind)
        bin_path, json_path = paths(kind)
        print(f"{bin_path.name}: {len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()[:16]}")
        if write:
            DATA.mkdir(exist_ok=True)
            bin_path.write_bytes(data)
            json_path.write_text(sidecar_text(side), encoding="utf-8")
    if not write:
        print("(dry run: pass --write to update tests/data)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
