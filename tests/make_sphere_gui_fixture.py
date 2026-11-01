"""The sphere world's fixtures for the Swift GUI tests (`gui/Tests/AlifeCoreTests/Sphere*Tests.swift`).

Two files, both produced with the engine's own code and read by the Swift tests:

* ``tests/data/play_sphere.bin``: a protocol 2 play stream written with the real encoders (`streamrec`): a
  planet of radius 20 and thickness 8 (Nside 4, three equal-volume layers, territory Nside 3, weather Nside 8
  pooled to 4), one ``FRMA`` with five agents (one at the north pole), ``MAPH``, ``TERR``, ``STRC``, ``FSOL``,
  ``ENVS`` with pools and food patches, ``STRM`` with a storm and its track, ``EVNT``, ``CLCK``;
* ``tests/data/sphere_gui.json``: what the Swift code must reproduce, computed by ``alife.geom`` and by the
  independent decoder ``tests/sphere_decode.py`` (the HEALPix lookup of random directions at four Nsides, the
  pixel centres and corners, the layer of a radius, the headings of the agents, the dequantised ``MAPH``
  planes).

Regenerate with ``python tests/make_sphere_gui_fixture.py``; ``tests/test_gui_sync.py`` compares the committed
files with ``build()`` (the Swift tests are only worth what these files are).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from alife import geom
from alife import streamrec as sr

HERE = Path(__file__).resolve().parent
OUT_BIN = HERE / "data" / "play_sphere.bin"
OUT_JSON = HERE / "data" / "sphere_gui.json"

RP, H, FZ = 20.0, 8.0, 3
NSIDE, NSIDE_W, NSIDE_T = 4, 8, 3
N, B = 16, 4
NPIX = geom.npix(NSIDE)
ALIVE_ROWS = (1, 2, 4, 7, 9)


def _unit(lat: float, lon: float) -> np.ndarray:
    return geom.from_latlon(lat, lon)


def agents() -> dict[str, np.ndarray]:
    """Five agents in the planet frame: (lat, lon, height, yaw, pitch), the angles exact in float16."""
    spec = [  # row, lat, lon, height, yaw, pitch
        (1, 20.0, 30.0, 2.0, 0.5, 0.125),
        (2, 90.0, 0.0, 1.0, -1.0, 0.0),
        (4, -45.0, -120.0, 7.5, 3.0, -0.25),
        (7, 0.0, 179.0, 0.5, -3.0, 0.5),
        (9, -90.0, 0.0, 3.0, 0.25, -0.5),
    ]
    pos = np.zeros((N, 3), np.float32)
    orient = np.zeros((N, 2), np.float32)
    alive = np.zeros(N, bool)
    uid = np.zeros(N, np.int64)
    rgb = np.zeros(N, np.uint32)
    for k, (row, lat, lon, h, yaw, pitch) in enumerate(spec):
        pos[row] = _unit(lat, lon) * (RP + h)
        orient[row] = (yaw, pitch)
        alive[row] = True
        uid[row] = 100 + k
        rgb[row] = 0x336699 + 0x101010 * k
    return {"pos": pos, "orient": orient, "alive": alive, "uid": uid, "rgb": rgb}


def maph_planes() -> dict[str, np.ndarray]:
    """Quantised planes of a known pattern (so that the expected values are formulas, not a simulation)."""
    i = np.arange(NPIX)
    return {
        "P": ((i * 7) % 256).astype(np.uint8),
        "Dc": ((i * 5 + 3) % 256).astype(np.uint8),
        "zeta": (((i * 3) % 255) - 127).astype(np.int8),
        "Ts": ((i * 11) % 256).astype(np.uint8),
        "winds": np.stack([(((i * (a + 2)) % 255) - 127).astype(np.int8) for a in range(4)]),
    }


def sphere_head() -> dict[str, Any]:
    edges = [float(e) for e in geom.layer_radii(RP, H, FZ)]
    legacy = sr.legacy_header(
        dims=3,
        size=0.0,
        height=H,
        n=N,
        bond_slots=B,
        radius=1.0,
        field_cell=4.0,
        fg=0,
        fz=FZ,
        nch=3,
        step=1000,
        run=None,
        config_hash="0" * 64,
        source="preset",
        sphere=sr.sphere_block(
            planet_radius=RP,
            shell_edges=edges,
            nside_field=NSIDE,
            nside_weather=NSIDE_W,
            nside_territory=NSIDE_T,
            weather_depth=40.0,
        ),
    )
    labels = sr.labels_block(
        intent_version=4,
        arch=["", "Grazer"],
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
        records=["FRMA", "IDNT", "TERR", "EVNT", "STRM", "ENVS", "CLCK", "STRC", "FSOL", "MAPH"],
        weather=sr.weather_block_sphere(enabled=True, dc_ref=20.0, nside=NSIDE_W, coarse=NSIDE),
        labels=labels,
        energy_cap=16384,
        structures=sr.cells_block(cap=4096, cells=[FZ, 1, NPIX], cell=4.0),
        niche=sr.cells_block(cap=4095, cells=[FZ, 1, NPIX], cell=4.0, farm=True),
    )
    return {"legacy": legacy, "play": play, "bytes": sr.play_header_bytes(legacy, play)}


def build() -> bytes:
    a = agents()
    head = sphere_head()
    step = 1000
    frma = sr.encode_frma(
        step=step,
        seq=0,
        alive=a["alive"],
        pos=a["pos"],
        orient=a["orient"],
        rgb=a["rgb"],
        uid=a["uid"],
        bonds=np.array([[1, 2], [4, 7]]),
        color_index=0,
        span=50,
    )
    maph = sr.encode_maph(
        step=step, seq=0, planes=maph_planes(), nside=NSIDE, k=int(math.log2(NSIDE_W // NSIDE)), dc_ref=20.0
    )
    nt = geom.npix(NSIDE_T)
    owner = np.full(nt, 255, np.uint8)
    owner[::3], owner[1::3] = 0, 1
    purity = (np.arange(nt) % 256).astype(np.uint8)
    terr = sr.encode_terr(
        step=step,
        seq=0,
        snapshot_step=750,
        ids=[11, 12],
        owner=owner.reshape(1, nt),
        purity=purity.reshape(1, nt),
    )
    mat = np.zeros((FZ, 1, NPIX), np.int32)
    mat[0, 0, 5], mat[1, 0, 100], mat[2, 0, NPIX - 1] = 4096, 1000, 2048
    soil = np.zeros((FZ, 1, NPIX), np.int32)
    soil[0, 0, 40] = 4095
    centre = _unit(10.0, 100.0) * RP
    storm = {
        "id": 3,
        "name": "Merrow",
        "named": True,
        "hemi": 0,
        "x": round(float(centre[0]), 2),
        "y": round(float(centre[1]), 2),
        "z": round(float(centre[2]), 2),
        "r": 6.5,
        "ext": [6.0, 7.0],
        "zeta": 0.8,
        "zeta_now": 0.6,
        "wind": 0.05,
        "rain": 0.2,
        "age": 400,
        "parent": -1,
        "state": "active",
        "track": [[900, *storm_xyz(_unit(9.0, 98.0) * RP)], [1000, *storm_xyz(centre)]],
        "pred": [[250, *storm_xyz(_unit(11.0, 103.0) * RP)]],
    }
    strm = sr.encode_strm(
        step=step, seq=0, tick=40, obj={"detect_step": step, "storms": [storm], "ended": []}
    )
    pools = []
    for lon, lat in ((60.0, 40.0), (-120.0, -40.0)):
        v = _unit(lat, lon) * RP
        pools.append(
            {
                "x": round(float(v[0]), 2),
                "y": round(float(v[1]), 2),
                "z": round(float(v[2]), 2),
                "lon": lon,
                "lat": lat,
                "amp": 1.0,
                "sx": 10.0,
                "sy": 6.0,
                "vx": -0.009,
                "vy": 0.0,
            }
        )
    patches = []
    for i, (lat, lon) in enumerate(((5.0, -30.0), (-60.0, 150.0))):
        u = _unit(lat, lon)
        patches.append(
            {
                "i": i,
                "x": round(float(u[0]), 4),
                "y": round(float(u[1]), 4),
                "z": round(float(u[2]), 4),
                "lon": lon,
                "lat": lat,
                "gain": 1.0,
                "sigma": 12.0,
            }
        )
    envs = sr.encode_envs(
        step=step,
        seq=0,
        obj={
            "pools": pools,
            "season": None,
            "levers": {},
            "effects": [],
            "cooldowns": [],
            "influence": None,
            "patches": patches,
        },
    )
    ev_pos = _unit(-33.0, 77.0) * (RP + 2.0)
    evnt = sr.encode_evnt(
        step=step,
        seq=0,
        events=[
            {
                "id": 1000100,
                "step": step,
                "kind": 7,
                "fam": "storm",
                "a": 3,
                "b": 0,
                "x": round(float(ev_pos[0]), 2),
                "y": round(float(ev_pos[1]), 2),
                "z": round(float(ev_pos[2]), 2),
                "imp": 1,
                "text": "A storm is born at 33.0S 77.0E.",
            }
        ],
    )
    clck = sr.encode_clck(
        step=step,
        seq=0,
        rate=1.0,
        achieved=1.0,
        step_ms=4.0,
        frame_ms=33.0,
        late_steps=0,
        save_stall_steps=0,
        reason=0,
        saving=False,
        fe=0,
        chapter=0,
        in_chapter=1000,
        chapter_len=6000,
    )
    return b"".join(
        [
            head["bytes"],
            frma,
            maph,
            terr,
            sr.encode_strc(step=step, seq=0, shape=(FZ, 1, NPIX), mat=mat, cap=4096),
            sr.encode_fsol(step=step, seq=0, shape=(FZ, 1, NPIX), soil=soil, cap=4095),
            envs,
            strm,
            evnt,
            clck,
        ]
    )


def storm_xyz(centre: np.ndarray) -> list[float]:
    return [round(float(v), 2) for v in centre]


def expected() -> dict[str, Any]:
    """What the Swift code must produce, from `geom` and the independent decoder (not from the writer)."""
    import sphere_decode  # the independent reference decoder (tests/)

    rng = np.random.default_rng(20261030)
    healpix: dict[str, Any] = {}
    for nside in (2, 4, 8, 21):
        v = rng.normal(size=(300, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        poles = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], float)
        v = np.concatenate([v, poles])
        npix = geom.npix(nside)
        sample = np.unique(np.concatenate([np.arange(0, npix, max(1, npix // 120)), [0, npix - 1]]))
        xyf = geom.pix2xyf(nside, sample)
        corners = []
        for ix, iy, face in zip(xyf[0][:40], xyf[1][:40], xyf[2][:40], strict=True):
            fx, fy = ix / nside, iy / nside
            corners.append(
                [
                    geom.xyf2vec(fx + dx / nside, fy + dy / nside, np.array(face)).tolist()
                    for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 1))
                ]
            )
        healpix[str(nside)] = {
            "points": v.tolist(),
            "pixels": geom.ang2pix(v, nside).tolist(),
            "sample": sample.tolist(),
            "centres": geom.pix2vec(nside, sample).tolist(),
            "xyf": [xyf[0].tolist(), xyf[1].tolist(), xyf[2].tolist()],
            "corner_pixels": sample[:40].tolist(),
            "corners": corners,
        }
    a = agents()
    rows = [r for r in range(N) if a["alive"][r]]
    pos = a["pos"][rows].astype(np.float64)
    orient = a["orient"][rows].astype(np.float16).astype(np.float64)
    heading = sphere_decode.heading_from_orientation(pos, orient[:, 0], orient[:, 1])
    lat, lon = geom.to_latlon(pos)
    edges = geom.layer_radii(RP, H, FZ)
    radii = np.linspace(RP - 0.5, RP + H + 0.5, 23)
    planes = maph_planes()
    p_ref, u_ref, zeta_ref, dc_ref = sr.P_REF, sr.U_REF, sr.ZETA_REF, 20.0
    return {
        "healpix": healpix,
        "planet": {"radius": RP, "height": H, "layers": FZ, "edges": edges.tolist(), "nside": NSIDE},
        "layer_radii": radii.tolist(),
        "layer_of": geom.layer_of(radii, RP, H, FZ).tolist(),
        "agents": {
            "rows": rows,
            "pos": pos.tolist(),
            "yaw": orient[:, 0].tolist(),
            "pitch": orient[:, 1].tolist(),
            "heading": heading.tolist(),
            "lat": lat.tolist(),
            "lon": lon.tolist(),
            "text": [geom.place_text(p, RP) for p in pos],
        },
        "maph": {
            "p": ((planes["P"] / 255.0) ** 2 * p_ref).tolist(),
            "dc": (planes["Dc"] / 255.0 * dc_ref).tolist(),
            "zeta": (planes["zeta"] / 127.0 * zeta_ref).tolist(),
            "ts": (planes["Ts"] / 255.0).tolist(),
            "winds": (planes["winds"] / 127.0 * u_ref).tolist(),
        },
        "place_texts": [
            [lat_, lon_, geom.place_text(geom.from_latlon(lat_, lon_))]
            for lat_, lon_ in ((0.0, 0.0), (12.34, 45.67), (-89.9, -179.5), (45.0, 180.0 - 1e-9))
        ],
    }


def main() -> int:
    sys.path.insert(0, str(HERE))
    OUT_BIN.write_bytes(build())
    OUT_JSON.write_text(json.dumps(expected(), separators=(",", ":")), encoding="utf-8")
    print(
        "wrote",
        OUT_BIN,
        OUT_BIN.stat().st_size,
        "bytes and",
        OUT_JSON,
        OUT_JSON.stat().st_size,
        "bytes",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
