"""Stream protocol v2 (the sphere world): the encoders against the independent decoder ``sphere_decode``
(written from ``stream_v2_spec.md``, not from ``streamrec``), a real ``alife play`` of a tiny3s world, the
legacy ``alife stream`` header, and the box streams' unchanged bytes (DESIGN §46.7; sphere series S5).
"""

from __future__ import annotations

import dataclasses
import io
import json
import struct
from pathlib import Path

import numpy as np
import pytest

import sphere_decode as sd
from alife import geom, play, streamrec
from alife.config import load_config
from alife.game import observer
from alife.game.social import SOCIAL_V1
from alife.state import to_numpy
from play_helpers import CADENCE, run_paced

END = 60
SOC = dataclasses.replace(SOCIAL_V1, snapshot_every=20, kin_occ=(1, 30), kin_enter=(2, 30), kin_leave=(1, 30))


@pytest.fixture(scope="module")
def played(tmp_path_factory: pytest.TempPathFactory):
    """A tiny3s world played 60 steps with the observers and the structures: (cfg, run dir, result, bytes)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(observer, "DEFAULT_SOCIAL", SOC)
        cfg = load_config(
            preset="tiny3s",
            overrides=[
                *CADENCE,
                "game.observe=true",
                "run.compile=false",
                "structures.enabled=true",
                "structures.build_max=64",
            ],
        )
        opts = play.PlayOptions(rate=float("inf"), paused=False, fresh=True, unattended=True, tick=10)
        rd, res, _pacer, out = run_paced(cfg, tmp_path_factory.mktemp("sp") / "w", END, [], options=opts)
    return cfg, rd, res, out.getvalue()


@pytest.fixture(scope="module")
def decoded(played) -> sd.Decoded:
    return sd.decode(played[3])


def by_tag(dec: sd.Decoded, tag: str) -> list[sd.Rec]:
    return [r for r in dec.records if r.tag == tag]


# --- the header -------------------------------------------------------------------------------------------


def test_the_header_is_protocol_2_and_names_the_planet(played, decoded) -> None:
    cfg, _rd, _res, _data = played
    h = decoded.header
    assert h["protocol"] == 2 and "L" not in h and h["FG"] == 0 and h["FZ"] == cfg.world.layers
    assert h["geometry"] == "sphere" and h["dims"] == 3 and h["height"] == cfg.world.height
    assert h["planet_radius"] == cfg.world.radius and h["radius"] == cfg.physics.radius
    np.testing.assert_allclose(
        h["shell_edges"], geom.layer_radii(cfg.world.radius, cfg.world.height, cfg.world.layers)
    )
    assert (h["nside_field"], h["nside_weather"], h["nside_territory"]) == (cfg.world.nside, 0, 9)
    assert h["ring_order"] == "ring" and h["f0"] == 0.025 and h["weather_depth"] == 0.0
    play = h["play"]
    assert "MAPH" not in play["records"] and "MAP1" not in play["records"]
    assert play["structures"]["cells"] == [cfg.world.layers, 1, 12 * cfg.world.nside**2]
    assert play["labels"]["intent_version"] == 4
    assert play["weather"]["enabled"] is False and play["weather"]["nside"] == 0


def test_the_sphere_decoder_refuses_a_protocol_1_header(played) -> None:
    """The magic keeps its last byte (1) and the header says 2: a reader that ties the two (the app's
    ``StreamHeader.validate``: ``protocol == magic version``) refuses a sphere stream until it is updated."""
    assert (
        played[3][:8] == b"ALSTRM\x00\x01"
        and json.loads(played[3][12:].split(b"\x00")[0][: struct.unpack("<I", played[3][8:12])[0]])[
            "protocol"
        ]
        == 2
    )
    box = load_config(preset="tiny3d", overrides=["brain.weight_dtype=float32", "game.observe=false"])
    head = streamrec.header_bytes(
        streamrec.legacy_header(
            dims=3, size=box.world.size, height=box.world.height, n=64, bond_slots=4, radius=1.0,
            field_cell=4.0, fg=32, fz=6, nch=3, step=0, run=None, config_hash="x", source="preset",
        )
    )  # fmt: skip
    with pytest.raises(sd.ProtocolError):
        sd.decode_header(head)


# --- the records ------------------------------------------------------------------------------------------


def test_every_frame_is_in_the_shell_and_its_orientation_rebuilds_the_heading(played, decoded) -> None:
    cfg, _rd, res, _data = played
    frames = by_tag(decoded, "FRMA")
    assert len(frames) >= 2
    last = frames[-1]
    assert last.step == END
    s = res.state
    alive = to_numpy(s.alive)
    np.testing.assert_array_equal(last.a["alive"], alive)
    pos = to_numpy(s.pos)[alive]
    np.testing.assert_array_equal(last.a["pos"], pos)  # float32 positions, bit for bit
    rp, h = cfg.world.radius, cfg.world.height
    assert last.a["height"].min() >= -1e-3 and last.a["height"].max() <= h + 1e-3
    # the stored (yaw, pitch) are f16 of the engine's bearing and pitch; the rebuilt tangent heading is hdg
    fwd = last.a["heading"]
    up = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    pitch = to_numpy(s.pitch)[alive].astype(np.float64)
    tangent = fwd - np.sin(pitch)[:, None] * up
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    hdg = to_numpy(s.hdg)[alive].astype(np.float64)
    hdg /= np.linalg.norm(hdg, axis=1, keepdims=True)
    level = np.abs(pitch) < 1.2  # near the vertical the tangent part is ill-conditioned
    assert level.sum() > 50
    assert np.abs((tangent * hdg).sum(axis=1)[level]).min() > 0.99  # f16 yaw: ~1e-3 rad
    assert np.abs(fwd @ np.ones(3)).max() > 0  # decoded, not empty
    assert rp > 0


def test_the_sparse_cell_record_is_the_material_by_layer_and_pixel(played, decoded) -> None:
    cfg, _rd, res, _data = played
    recs = by_tag(decoded, "STRC")
    assert recs and recs[-1].step == END
    mat = to_numpy(res.state.mat)
    cap = cfg.structures.cap
    want = np.zeros(mat.shape, np.uint8)
    nz = mat > 0
    want[nz] = 1 + (254 * np.minimum(mat[nz], cap)) // cap
    levels = recs[-1].a["levels"]
    assert levels.shape == mat.shape == (cfg.world.layers, 12 * cfg.world.nside**2)
    np.testing.assert_array_equal(levels, want)
    assert recs[-1].f["ny"] == 1 and recs[-1].f["nz"] == cfg.world.layers and nz.any()


def test_the_territory_record_is_indexed_by_the_ring_pixels_of_the_territory_nside(played, decoded) -> None:
    terr = by_tag(decoded, "TERR")
    assert terr
    t = terr[-1]
    assert t.f["ny"] == 1 and t.f["nx"] == 12 * 9**2 == len(t.a["owner"]) == len(t.a["purity"])
    ids = t.f["json"]["ids"]
    assert len(ids) <= 16
    owned = t.a["owner"] != 255
    assert (t.a["owner"][owned] < len(ids)).all()


def test_the_tables_and_events_state_planet_frame_places(played, decoded) -> None:
    cfg, *_ = played
    rp, h = cfg.world.radius, cfg.world.height
    idnt = by_tag(decoded, "IDNT")
    assert idnt
    for rec in idnt:
        for col in rec.f["json"]["colonies"]:
            r = np.linalg.norm([col["x"], col["y"], col["z"]])
            assert rp - 1 < r < rp + h + 1
    ev = [e for r in by_tag(decoded, "EVNT") for e in r.f["json"]]
    for e in ev:
        if e["x"] is not None and e["z"] is not None:
            assert np.linalg.norm([e["x"], e["y"], e["z"]]) < rp + h + 0.05


def test_the_run_is_pure_a_played_world_equals_a_plain_run(played, tmp_path: Path) -> None:
    """Playing a sphere (the stream, the pacer, the observers) leaves the trajectory alone."""
    from play_helpers import checkpoint_digests, run_plain

    cfg, rd, _res, _data = played
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(observer, "DEFAULT_SOCIAL", SOC)
        plain, _ = run_plain(cfg, tmp_path / "plain", END)
    assert checkpoint_digests(plain) == checkpoint_digests(rd)


# --- MAPH, from synthetic fields --------------------------------------------------------------------------


def _head(stream_body: bytes, nside_field: int = 2, nside_weather: int = 4) -> bytes:
    h = {
        "protocol": 2, "dims": 3, "height": 16.0, "N": 4, "B": 4, "radius": 1.0, "field_cell": 4.0, "FG": 0,
        "FZ": 2, "geometry": "sphere", "planet_radius": 36.0,
        "shell_edges": geom.layer_radii(36.0, 16.0, 2).tolist(),
        "nside_field": nside_field, "nside_weather": nside_weather, "nside_territory": 9,
        "ring_order": "ring", "f0": 0.025, "weather_depth": 80.0, "play": {"records": ["MAPH"]},
    }  # fmt: skip
    return streamrec.header_bytes(h) + stream_body


def test_maph_round_trips_through_the_independent_decoder() -> None:
    nw, nc = 8, 4
    rng = np.random.default_rng(2)
    npw = geom.npix(nw)
    centres = geom.pixel_centres(nw)
    p = rng.uniform(0, 0.004, npw)
    dc = rng.uniform(0, 1.0, npw)
    zeta = rng.uniform(-0.03, 0.03, npw)
    ts = rng.uniform(0, 1, npw)
    wind = rng.uniform(-0.1, 0.1, (2, npw, 3))
    wind -= (wind * centres).sum(-1, keepdims=True) * centres  # tangent winds
    q = streamrec.quantise_maph(
        p=p, dc=dc, zeta=zeta, ts=ts, winds=wind, nside_weather=nw, nside=nc, dc_ref=1.0
    )
    rec = streamrec.encode_maph(step=7, seq=3, planes=q, nside=nc, k=1, dc_ref=1.0)
    head = json.dumps  # noqa: F841
    h = {
        "protocol": 2, "dims": 3, "height": 16.0, "N": 4, "B": 4, "radius": 1.0, "field_cell": 4.0, "FG": 0,
        "FZ": 2, "geometry": "sphere", "planet_radius": 36.0,
        "shell_edges": geom.layer_radii(36.0, 16.0, 2).tolist(),
        "nside_field": nc, "nside_weather": nw, "nside_territory": 9, "ring_order": "ring", "f0": 0.025,
        "weather_depth": 80.0,
        "play": {"records": ["MAPH"], "versions": {"MAPH": 1},
                 "weather": {"enabled": True, "nside": nw, "coarse": nc, "f0": 0.025, "name_zeta": 0.5,
                             "refs": {}}},
    }  # fmt: skip
    dec = sd.decode(streamrec.header_bytes(h) + rec)
    [m] = dec.records
    assert (m.step, m.seq, m.flags) == (7, 3, 3) and m.f["nside"] == nc and m.f["k"] == 1
    # the pooled fields written out independently: the mean of the children, the wind as a vector, then the
    # local east and north at the coarse centre
    kids = geom.children_table(nc, nw // nc)
    cc = geom.pixel_centres(nc)
    east, north = geom.local_frame(cc)
    np.testing.assert_allclose(m.a["P"], np.minimum(p[kids].mean(1), 0.004), atol=0.004 * 2 / 255)
    np.testing.assert_allclose(m.a["Dc"], dc[kids].mean(1), atol=1 / 255)
    np.testing.assert_allclose(m.a["Ts"], ts[kids].mean(1), atol=1 / 255 + 1e-9)
    np.testing.assert_allclose(m.a["zeta"], np.clip(zeta[kids].mean(1), -0.025, 0.025), atol=0.025 / 127)
    cw = wind[:, kids, :].mean(2)
    want = np.stack([(cw * east).sum(-1), (cw * north).sum(-1)], 1).reshape(4, -1)
    np.testing.assert_allclose(m.a["winds"], np.clip(want, -0.1, 0.1), atol=0.1 / 127)
    # a uniform eastward wind decodes as purely eastward at every cell (the frame is right, not just the sum)
    ew = np.broadcast_to(geom.local_frame(centres)[0] * 0.05, (2, npw, 3))
    q2 = streamrec.quantise_maph(
        p=p, dc=dc, zeta=zeta, ts=ts, winds=np.array(ew), nside_weather=nw, nside=nc, dc_ref=1.0
    )
    dec2 = sd.decode(
        streamrec.header_bytes(h) + streamrec.encode_maph(step=1, seq=0, planes=q2, nside=nc, k=1, dc_ref=1.0)
    )
    w = dec2.records[0].a["winds"]
    assert np.abs(w[0] - 0.05).max() < 0.004 and np.abs(w[1]).max() < 0.004


def test_maph_refuses_what_it_cannot_encode() -> None:
    npix = geom.npix(2)
    q = {"P": np.zeros(npix, np.uint8), "Dc": np.zeros(npix, np.uint8), "zeta": np.zeros(npix, np.int8),
         "Ts": np.zeros(npix, np.uint8), "winds": np.zeros((4, npix), np.int8)}  # fmt: skip
    streamrec.encode_maph(step=0, seq=0, planes=q, nside=2, k=0, dc_ref=1.0)
    with pytest.raises(streamrec.RecordError):
        streamrec.encode_maph(step=0, seq=0, planes={**q, "winds": q["winds"][:3]}, nside=2, k=0, dc_ref=1.0)
    with pytest.raises(streamrec.RecordError, match="power of two"):
        streamrec.quantise_maph(
            p=np.zeros(geom.npix(6)), dc=np.zeros(geom.npix(6)), zeta=np.zeros(geom.npix(6)),
            ts=np.zeros(geom.npix(6)), winds=np.zeros((2, geom.npix(6), 3)), nside_weather=6, nside=4,
            dc_ref=1.0,
        )  # fmt: skip


def test_the_sphere_block_validates() -> None:
    ok = dict(planet_radius=36.0, shell_edges=[36.0, 40.0, 52.0], nside_field=8, nside_weather=0,
              nside_territory=9, weather_depth=0.0)  # fmt: skip
    assert streamrec.sphere_block(**ok)["ring_order"] == "ring"
    for bad in ({"shell_edges": [36.0, 36.0, 52.0]}, {"planet_radius": 0.0}, {"nside_field": 1},
                {"shell_edges": [36.0, 52.0]}):  # fmt: skip
        with pytest.raises(streamrec.RecordError):
            streamrec.sphere_block(**{**ok, **bad})


# --- the legacy stream ------------------------------------------------------------------------------------


def test_alife_stream_writes_the_same_header_and_a_frame(played) -> None:
    from alife.state import init_state
    from alife.stream import Streamer, View

    cfg = played[0]
    s = init_state(cfg)
    buf = io.BytesIO()
    st = Streamer(cfg, buf, view=View(fields=True, field_z=1))
    st.begin(s)
    st.send(s)
    data = buf.getvalue()
    n = struct.unpack("<I", data[8:12])[0]
    h = json.loads(data[12 : 12 + n])
    assert h["protocol"] == 2 and "L" not in h and h["FG"] == 0 and h["planet_radius"] == cfg.world.radius
    play_h = sd.decode(streamrec.header_bytes({**h, "play": {"records": []}})).header  # same header checks
    assert play_h["nside_field"] == cfg.world.nside
    frame = data[12 + n :]
    assert frame[:4] == b"FRM1"
    npix = 12 * cfg.world.nside**2
    n_alive = struct.unpack("<QIIIIQ", frame[8:40])[1]
    assert n_alive == int(to_numpy(s.alive).sum())
    n_rows = s.alive.shape[0]  # head, alive bytes, pos 12, orient 4, rgb 4, no bonds, the field layer
    assert len(frame) == 8 + 32 + ((n_rows + 3) // 4) * 4 + 12 * n_rows + 4 * n_rows + 4 * n_rows + 4 * npix


def test_a_box_header_is_unchanged_by_the_sphere_keys() -> None:
    kw = dict(dims=3, size=320.0, height=80.0, n=64, bond_slots=4, radius=1.0, field_cell=4.0, fg=80, fz=20,
              nch=3, step=0, run=None, config_hash="h", source="preset")  # fmt: skip
    box = streamrec.legacy_header(**kw)
    assert box["protocol"] == 1 and box["L"] == 320.0 and "geometry" not in box and box["FG"] == 80
    assert list(box)[:3] == ["protocol", "dims", "L"]
    assert streamrec.ALL_TAGS[: len(streamrec.TAGS)] == streamrec.TAGS and "MAPH" not in streamrec.TAGS


def test_alife_play_as_a_process_writes_a_decodable_protocol_2_stream(tmp_path: Path) -> None:
    """The real thing: a subprocess on the CPU (compiled step, one configuration per process), real pipes."""
    from play_helpers import finish, spawn

    out = tmp_path / "w"
    args = [
        "play",
        "--preset",
        "tiny3s",
        "--set",
        "brain.weight_dtype=float32",
        "--out",
        str(out),
        "--until",
        "30",
    ]
    code, data, err = finish(spawn(*args), "rate max\nrun\n")
    assert code == 0, err
    dec = sd.decode(data)
    assert dec.header["protocol"] == 2 and dec.header["geometry"] == "sphere" and "L" not in dec.header
    tags = [r.tag for r in dec.records]
    assert tags[0] == "FRMA" and "IDNT" in tags and "ENVS" in tags and dec.skipped == []
    last = by_tag(dec, "FRMA")[-1]
    assert last.step == 30 and last.a["height"].min() > -1e-3
