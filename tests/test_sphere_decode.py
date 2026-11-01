"""Tests of the protocol 2 reference decoder ``sphere_decode`` ALONE: streams built by hand with
``struct.pack`` from the format text (no encoder involved), the decoded values, the refusals, and the
orientation maths."""

from __future__ import annotations

import json
import math
import struct

import numpy as np
import pytest

import sphere_decode as sd

RP, H, FZ = 10.0, 2.0, 2
NS_F, NS_T, NS_W = 2, 2, 4  # field, territory, weather Nside
NPIX_F = 12 * NS_F**2  # 48
NPIX_T = 12 * NS_T**2
N_CAP = 8


# ---- builders -------------------------------------------------------------------------------------------


def _edges() -> list[float]:
    v0, v1 = RP**3, (RP + H) ** 3
    e = [(v0 + (v1 - v0) * i / FZ) ** (1 / 3) for i in range(FZ + 1)]
    e[0], e[-1] = RP, RP + H
    return e


def header_obj(**over) -> dict:
    h = {
        "protocol": 2,
        "dims": 3,
        "height": H,
        "N": N_CAP,
        "B": 4,
        "radius": 0.5,
        "FG": 0,
        "FZ": FZ,
        "geometry": "sphere",
        "planet_radius": RP,
        "shell_edges": _edges(),
        "nside_field": NS_F,
        "nside_weather": NS_W,
        "nside_territory": NS_T,
        "ring_order": "ring",
        "f0": 0.025,
        "weather_depth": 3.0,
        "play": {
            "records": ["FRMA", "MAPH", "STRC", "TERR", "EVNT", "ENVS", "STRM", "FOCS", "CLCK"],
            "weather": {
                "enabled": True,
                "nside": NS_W,
                "coarse": NS_F,
                "f0": 0.025,
                "name_zeta": "zeta",
                "refs": {},
            },
            "structures": {"cells": [FZ, 1, NPIX_F]},
            "niche": {"cells": [FZ, 1, NPIX_F]},
            "versions": {"MAPH": 1},
        },
    }
    for k, v in over.items():
        if v is None:
            h.pop(k, None)
        else:
            h[k] = v
    return h


def header_bytes(obj: dict) -> bytes:
    text = json.dumps(obj, separators=(",", ":")).encode("ascii")
    text += b" " * (-len(text) % 4)
    return sd.MAGIC + struct.pack("<I", len(text)) + text


def rec(tag: str, body: bytes, step=100, seq=1, version=1, flags=0) -> bytes:
    payload = struct.pack("<QIHH", step, seq, version, flags) + body
    assert len(payload) % 4 == 0
    return tag.encode() + struct.pack("<I", len(payload)) + payload


def zpad(b: bytes) -> bytes:
    return b + b"\x00" * (-len(b) % 4)


def frma(pos, yaw_pitch, bonds=((0, 1),), flow=None, cap=N_CAP, flags=None) -> bytes:
    n = len(pos)
    bitmap = sum(1 << i for i in range(n)) | 0  # rows 0..n-1 alive
    body = struct.pack("<IIIII", n, len(bonds), 0, 7, cap) + struct.pack("<I", bitmap)
    body += np.asarray(pos, "<f4").tobytes() + np.asarray(yaw_pitch, "<f2").tobytes()
    body += np.arange(n, dtype="<u4").tobytes()  # rgb
    body += np.arange(1, n + 1, dtype="<i8").tobytes()  # uid
    body += np.asarray(bonds, "<u4").reshape(-1, 2).tobytes()
    if flow is not None:
        body += zpad(np.asarray(flow, np.int8).tobytes())
    return rec("FRMA", body, flags=(2 if flow is not None else 0) if flags is None else flags)


def maph(planes=None, flags=3, nside=NS_F, k=1, extra=b"", reserved=b"\x00" * 8) -> bytes:
    npix = 12 * nside * nside
    if planes is None:
        i = np.arange(npix)
        planes = (
            ((i * 5) % 256).astype(np.uint8),  # P
            ((i * 3 + 1) % 256).astype(np.uint8),  # Dc
            (((i * 7) % 255) - 127).astype(np.int8),  # zeta
            ((i * 11) % 256).astype(np.uint8),  # Ts
            (((np.arange(4 * npix) * 13) % 255) - 127).astype(np.int8),  # winds u1 v1 u2 v2
        )
    body = struct.pack("<HHfffff", nside, k, 0.025, 2.0, 4.0, 8.0, 0.5) + reserved
    body += b"".join(a.tobytes() for a in planes) + extra
    return rec("MAPH", body, flags=flags), planes


def cells(tag, idx, lev, dims=(NPIX_F, 1, FZ), pad=None, flags=0) -> bytes:
    count = len(idx)
    body = struct.pack("<HHHHI", *dims, flags, count)
    body += np.asarray(idx, "<u4").tobytes() + np.asarray(lev, np.uint8).tobytes()
    body += b"\x00" * ((-count) % 4) if pad is None else pad
    return rec(tag, body, flags=flags)


def terr(owner, purity, ids=(7, 9), nx=NPIX_T, pad=b"") -> bytes:
    js = json.dumps({"ids": list(ids)}).encode()
    jpad = b"\x00" * (-len(js) % 4)
    body = struct.pack("<HHIII", nx, 1, 5000, len(js), 0) + js + jpad
    body += bytes(owner) + bytes(purity)
    body += pad if pad else b"\x00" * (-(2 * nx) % 4)
    return rec("TERR", body)


def evnt(events) -> bytes:
    js = json.dumps(events).encode()
    return rec("EVNT", struct.pack("<II", len(events), len(js)) + zpad(js))


def envs(obj) -> bytes:
    js = json.dumps(obj).encode()
    return rec("ENVS", struct.pack("<I", len(js)) + zpad(js))


def strm(storms) -> bytes:
    js = json.dumps({"storms": storms}).encode()
    return rec("STRM", struct.pack("<IIII", len(js), 12, len(storms), 0) + zpad(js))


def clck() -> bytes:
    return rec("CLCK", struct.pack("<ffffIIBBHIIII", 10.0, 9.5, 1.0, 2.0, 3, 4, 2, 1, 5, 6, 7, 8, 9))


def focs(x, y, z, vx, vy, vz) -> bytes:
    body = struct.pack(
        "<qIIffffffiIIqBBBBBBBB", 3, 1, 0, x, y, z, vx, vy, vz, 50, 10, 2, -1, 0, 255, 255, 255, 0, 0, 0, 0
    )
    return rec("FOCS", body)


POS = [(10.5, 0.0, 0.0), (0.0, 11.0, 0.0), (0.0, 0.0, 11.9)]
YP = [(0.0, 0.0), (math.pi / 2, 0.0), (0.0, math.pi / 4)]


def good_stream() -> bytes:
    maph_rec, _ = maph()
    idx = [0, 5, 47, 48, 95]
    ev = [
        {
            "id": 1,
            "step": 100,
            "kind": "k",
            "fam": "kin",
            "a": 1,
            "b": None,
            "x": 1.0,
            "y": 2.0,
            "z": 3.0,
            "imp": 1,
            "text": "t",
        }
    ]
    return (
        header_bytes(header_obj())
        + frma(POS, YP, flow=[3])
        + maph_rec
        + cells("STRC", idx, [1, 2, 3, 4, 255])
        + cells("FSOL", [47], [9])
        + terr([0, 1, 255] + [0] * (NPIX_T - 3), list(range(NPIX_T)))
        + evnt(ev)
        + envs(
            {
                "patches": [
                    {
                        "i": 0,
                        "x": 0.0,
                        "y": 0.6,
                        "z": 0.8,
                        "lon": 90.0,
                        "lat": 53.13,
                        "gain": 1.0,
                        "sigma": 2.0,
                    }
                ]
            }
        )
        + strm([{"id": 1, "x": 0.0, "y": 10.0, "z": 0.0, "r": 3.0}])
        + clck()
        + focs(0.0, 11.0, 0.0, 0.0, 2.0, 5.0)
        + rec("ZZZZ", b"\x00" * 16)
    )


# ---- positive decoding ---------------------------------------------------------------------------------


def test_good_stream_decodes():
    d = sd.decode(good_stream())
    assert d.header["protocol"] == 2
    assert [r.tag for r in d.records] == [
        "FRMA",
        "MAPH",
        "STRC",
        "FSOL",
        "TERR",
        "EVNT",
        "ENVS",
        "STRM",
        "CLCK",
        "FOCS",
    ]
    assert d.skipped == [("ZZZZ", d.skipped[0][1], "tag")]


def test_frma_values_and_position_in_shell():
    r = sd.decode(good_stream()).records[0]
    assert r.f["n"] == 3 and r.f["n_bonds"] == 1
    assert np.allclose(r.a["pos"], POS)
    assert np.allclose(r.a["height"], [0.5, 1.0, 1.9])
    assert r.a["bonds"].tolist() == [[0, 1]]
    assert r.a["flow"].tolist() == [3]
    yaw = np.float64(np.float16(math.pi / 2))
    pit = np.float64(np.float16(math.pi / 4))
    # agent 0 at +x: up = x, east = z x x = y, north = x x y = z; yaw 0 heads north
    assert np.allclose(r.a["heading"][0], [0, 0, 1], atol=1e-12)
    # agent 1 at +y: east = z x y = -x; yaw ~ pi/2 heads east
    assert np.allclose(r.a["heading"][1], [-math.sin(yaw), 0, math.cos(yaw)], atol=1e-12)
    # agent 2 at the north pole: east = x, north = y; pitch pi/4 above the tangent plane, bearing north
    assert np.allclose(r.a["heading"][2], [0, math.cos(pit), math.sin(pit)], atol=1e-12)
    assert np.allclose(np.linalg.norm(r.a["heading"], axis=1), 1.0)


def test_maph_dequantisation():
    d = sd.decode(good_stream())
    r = d.records[1]
    _, planes = maph()
    p, dc, z, ts, w = planes
    assert (r.f["nside"], r.f["k"], r.f["npix"]) == (2, 1, 48)
    assert r.f["f0"] == pytest.approx(0.025, rel=1e-6)
    assert r.a["P"].shape == (48,) and r.a["winds"].shape == (4, 48)
    assert np.allclose(r.a["P"], (p / 255.0) ** 2 * 2.0)
    assert np.allclose(r.a["Dc"], dc / 255.0 * 4.0)
    assert np.allclose(r.a["zeta"], z / 127.0 * 0.5)
    assert np.allclose(r.a["Ts"], ts / 255.0)
    assert np.allclose(r.a["winds"], w.reshape(4, 48) / 127.0 * 8.0)
    # plane-major: row 1 is v1, i.e. the second block of 48 bytes
    assert r.a["q_winds"][1].tolist() == w[48:96].tolist()
    assert len(d.records[1].sections["winds"]) == 192


def test_strc_dense_levels():
    d = sd.decode(good_stream())
    strc, fsol = d.records[2], d.records[3]
    lv = strc.a["levels"]
    assert lv.shape == (FZ, NPIX_F) and lv.dtype == np.uint8
    assert lv[0, 0] == 1 and lv[0, 5] == 2 and lv[0, 47] == 3 and lv[1, 0] == 4 and lv[1, 47] == 255
    assert int((lv > 0).sum()) == 5
    assert fsol.a["levels"][0, 47] == 9 and int((fsol.a["levels"] > 0).sum()) == 1


def test_terr_arrays():
    r = sd.decode(good_stream()).records[4]
    assert r.a["owner"].shape == (12 * NS_T**2,) == r.a["purity"].shape
    assert r.a["owner"][:3].tolist() == [0, 1, 255]
    assert r.a["purity"].tolist() == list(range(NPIX_T))
    assert r.f["json"] == {"ids": [7, 9]} and r.f["snapshot_step"] == 5000


def test_other_records():
    d = sd.decode(good_stream())
    ev, en, st, ck, fo = d.records[5], d.records[6], d.records[7], d.records[8], d.records[9]
    assert ev.f["json"][0]["z"] == 3.0
    assert en.f["json"]["patches"][0]["lat"] == pytest.approx(53.13)
    assert st.f["json"]["storms"][0]["r"] == 3.0
    assert ck.f["fe"] == 6 and ck.f["reason"] == 2
    assert fo.f["v_radial"] == pytest.approx(2.0) and fo.f["height"] == pytest.approx(1.0)


def test_frma_without_flow_and_empty():
    s = header_bytes(header_obj()) + frma(POS, YP, bonds=())
    assert sd.decode(s).records[0].a["bonds"].shape == (0, 2)
    s = header_bytes(header_obj()) + frma([], np.zeros((0, 2)), bonds=())
    assert sd.decode(s).records[0].a["heading"].shape == (0, 3)


def test_header_without_weather_and_unknown_version_skipped():
    h = header_obj(nside_weather=0, weather_depth=0.0)
    h["play"].pop("weather")
    s = header_bytes(h) + rec("TERR", b"\x00" * 40, version=2)
    d = sd.decode(s)
    assert d.skipped == [("TERR", d.skipped[0][1], "version")] and d.records == []


# ---- refusals ------------------------------------------------------------------------------------------


def bad(stream: bytes, match: str, exc=AssertionError):
    with pytest.raises(exc, match=match):
        sd.decode(stream)


def test_protocol_1_refused():
    h = header_obj(protocol=1)
    with pytest.raises(sd.ProtocolError, match="protocol 1 stream"):
        sd.decode(header_bytes(h))
    with pytest.raises(sd.ProtocolError, match="protocol 3"):
        sd.decode(header_bytes(header_obj(protocol=3)))


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"FG": 8}, "FG must be 0"),
        ({"L": 16.0}, "no image length"),
        ({"shell_edges": [10.0, 11.0]}, "FZ \\+ 1"),
        ({"shell_edges": [10.0, 12.0, 11.0]}, "not ascending"),
        ({"shell_edges": [10.5, 11.1, 12.0]}, "first shell edge"),
        ({"shell_edges": [10.0, 11.0, 12.5]}, "last shell edge"),
        ({"shell_edges": [10.0, 11.5, 12.0]}, "equal volume"),
        ({"dims": 2}, "dims"),
        ({"geometry": "box"}, "sphere|geometry"),
        ({"ring_order": "nested"}, "ring_order|ring"),
        ({"nside_field": 1}, "nside_field|>= 2|assert"),
        ({"nside_territory": 1}, "nside_territory|assert"),
        ({"weather_depth": 3.0, "nside_weather": 0}, "weather_depth is 0 without weather"),
        ({"planet_radius": None}, "header keys missing"),
    ],
)
def test_header_refusals(over, match):
    bad(header_bytes(header_obj(**over)), match)


def test_header_play_refusals():
    h = header_obj()
    h["play"]["records"].append("MAP1")
    bad(header_bytes(h), "MAP1")
    h = header_obj()
    h["play"]["structures"]["cells"] = [FZ, 1, 47]
    bad(header_bytes(h), "structures.cells")
    h = header_obj()
    h["play"]["labels"] = {"intent_version": 3, "arch": [], "intent": []}
    bad(header_bytes(h), "intent_version is 4")
    h = header_obj()
    h["play"]["weather"]["coarse"] = 8
    bad(header_bytes(h), "play.weather")


def test_position_outside_shell():
    hdr = header_bytes(header_obj())
    for p in ((9.0, 0, 0), (0, 12.5, 0), (0, 0, 0.5)):
        bad(hdr + frma([p, (10.5, 0, 0), (0, 11, 0)], YP), "outside the shell")
    # inside the 1e-3 float32 allowance is fine
    sd.decode(hdr + frma([(10.0 - 5e-4, 0, 0), (0, 12.0 + 5e-4, 0), (0, 0, 11)], YP))


def test_frma_other_refusals():
    hdr = header_bytes(header_obj())
    bad(hdr + frma(POS, YP, bonds=((1, 0),)), "i < j < n")
    bad(hdr + frma(POS, YP, bonds=((0, 3),)), "i < j < n")
    bad(hdr + frma(POS, YP, cap=16), "differs from the header")
    bad(hdr + frma(POS, [(4.0, 0)] * 3), "yaw outside")
    bad(hdr + frma(POS, [(0, 1.7)] * 3), "pitch outside")


def test_strc_refusals():
    hdr = header_bytes(header_obj())
    bad(hdr + cells("STRC", [0, 96], [1, 1]), "at or past FZ")
    bad(hdr + cells("STRC", [5, 5], [1, 1]), "strictly ascending")
    bad(hdr + cells("STRC", [6, 5], [1, 1]), "strictly ascending")
    bad(hdr + cells("STRC", [0, 1], [1, 0]), "level 0")
    bad(hdr + cells("STRC", [0, 1], [1, 1], pad=b"\x01\x00"), "padding")
    bad(hdr + cells("STRC", [0], [1], dims=(NPIX_F, 2, FZ)), "dims")
    bad(hdr + cells("STRC", [0], [1], dims=(64, 1, FZ)), "dims")
    bad(hdr + cells("FSOL", [0], [1], dims=(NPIX_F, 1, FZ + 1)), "dims")
    bad(hdr + cells("STRC", [0], [1], flags=4), "flags")


def test_terr_refusals():
    hdr = header_bytes(header_obj())
    own, pur = [0] * NPIX_T, [0] * NPIX_T
    bad(hdr + terr(own, pur, nx=NPIX_T + 4)[:0] + terr([0] * 64, [0] * 64, nx=64), "TERR dims")
    bad(hdr + terr([2] + [0] * (NPIX_T - 1), pur), "slot")
    sd.decode(hdr + terr([253, 254, 255] + [0] * (NPIX_T - 3), pur))
    # 2 * 48 = 96 is a multiple of 4, so build a case with a padded plane via ids-free nx? use json padding
    js = b'{"ids":[7]}'  # 11 bytes: one pad byte
    body = struct.pack("<HHIII", NPIX_T, 1, 1, len(js), 0) + js + b"\x01" + bytes(own) + bytes(pur)
    bad(hdr + rec("TERR", body), "padding")


def test_maph_refusals():
    hdr = header_bytes(header_obj())
    good, _ = maph()
    sd.decode(hdr + good)
    bad(hdr + maph(flags=1)[0], "flags")
    bad(hdr + maph(flags=2)[0], "flags")
    bad(hdr + maph(flags=7)[0], "flags")
    bad(hdr + maph(extra=b"\x00" * 4)[0], "MAPH length|unexplained")
    bad(hdr + maph(reserved=b"\x00" * 7 + b"\x01")[0], "reserved")
    bad(hdr + maph(k=2)[0], "nside_weather")
    bad(hdr + maph(nside=4, k=0)[0], "nside differs")
    bad(header_bytes(header_obj(nside_weather=0, weather_depth=0.0)) + good, "nside_weather")
    # a truncated plane block (drop the last wind byte, keep alignment by dropping four)
    short = good[:-4]
    short = short[:4] + struct.pack("<I", len(short) - 8) + short[8:]
    bad(hdr + short, "runs past|MAPH length")


def test_envs_strm_evnt_refusals():
    hdr = header_bytes(header_obj())
    patch = {"i": 0, "x": 0.0, "y": 0.6, "z": 0.8, "lon": 90.0, "lat": 53.13, "gain": 1.0, "sigma": 2.0}
    old = {k: v for k, v in patch.items() if k not in ("z", "lon", "lat")}
    bad(hdr + envs({"patches": [old]}), "patch keys")
    bad(hdr + envs({"patches": [{**patch, "y": 0.7}]}), "unit vector")
    bad(hdr + envs({"patches": [{**patch, "lat": 120.0}]}), "lon, lat")
    nw = header_obj(nside_weather=0, weather_depth=0.0)
    nw["play"].pop("weather")
    bad(header_bytes(nw) + envs({"pools": [{"x": 0.0}]}), "pools")  # without weather: the key stays, empty
    sd.decode(header_bytes(nw) + envs({"pools": []}))
    sd.decode(hdr + envs({"pools": [{"x": 0.0}]}))  # weather on: pools may be present
    bad(hdr + strm([{"id": 1, "lon": 3.0, "x": 0.0, "y": 10.0, "z": 0.0, "r": 3.0}]), "positional")
    bad(hdr + strm([{"id": 1, "x": 0.0, "y": 14.0, "z": 0.0, "r": 3.0}]), "floor radius")
    bad(hdr + strm([{"id": 1, "x": 0.0, "y": 10.0, "r": 3.0}]), "x, y, z, r")
    ev = {
        "id": 1,
        "step": 1,
        "kind": "k",
        "fam": "kin",
        "a": None,
        "b": None,
        "x": 1.0,
        "y": 2.0,
        "z": "q",
        "imp": 0,
        "text": "",
    }
    bad(hdr + evnt([ev]), "numbers or null")
    bad(hdr + evnt([{**ev, "z": 99.0}]), "beyond the shell")


def test_focs_outside_shell():
    hdr = header_bytes(header_obj())
    bad(hdr + focs(0.0, 20.0, 0.0, 0, 0, 0), "outside the shell")


def test_alignment_trailing_and_truncation():
    s = good_stream()
    bad(s + b"\x00", "truncated record prefix")
    bad(s + b"ABCD", "truncated record prefix")
    bad(s[:-4], "truncated|runs past")
    hdr = header_bytes(header_obj())
    bad(hdr + b"ZZZZ" + struct.pack("<I", 6) + b"\x00" * 6, "multiple of 4")
    bad(hdr + b"FRMA" + struct.pack("<I", 8) + b"\x00" * 8, "shorter than the common head")
    bad(hdr + b"MAP1" + struct.pack("<I", 16) + b"\x00" * 16, "never carries MAP1")


def test_header_text_padding():
    hdr = header_bytes(header_obj())
    n = struct.unpack_from("<I", hdr, 8)[0]
    longer = hdr[:8] + struct.pack("<I", n + 4) + hdr[12:] + b"    "
    bad(longer, "padded with fewer than 4 spaces")
    bad(b"ALSTRM\x00\x02" + hdr[8:], "ALSTRM")


# ---- orientation ---------------------------------------------------------------------------------------


def _wrap(a):
    return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi


def _random_positions(rng, n):
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return u * rng.uniform(RP, RP + H, size=(n, 1))


def test_basis_known_points():
    pos = np.array([[11, 0, 0], [0, 11, 0], [0, 0, 11], [0, 0, -11], [1e-7 * 11, 0, 11]], float)
    zero = np.zeros(5)
    north = sd.heading_from_orientation(pos, zero, zero)
    east = sd.heading_from_orientation(pos, zero + math.pi / 2, zero)
    up = sd.heading_from_orientation(pos, zero, zero + math.pi / 2)
    assert np.allclose(north, [[0, 0, 1], [0, 0, 1], [0, 1, 0], [0, -1, 0], [0, 1, 0]], atol=1e-6)
    assert np.allclose(east, [[0, 1, 0], [-1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]], atol=1e-6)
    assert np.allclose(up, pos / np.linalg.norm(pos, axis=1, keepdims=True), atol=1e-12)


def test_pole_rule_threshold():
    # |z x up| = 5e-7 < 1e-6: east is (1, 0, 0) even though the point is displaced towards -y
    p = np.array([[0.0, -5e-7, 1.0]])
    h = sd.heading_from_orientation(p, np.array([math.pi / 2]), np.array([0.0]))
    assert np.allclose(h, [[1, 0, 0]], atol=1e-6)
    # |z x up| = 2e-6 > 1e-6: east is the normalised z x up = (-uy, ux, 0) = (+1, 0, 0) here
    p = np.array([[0.0, -2e-6, 1.0]])
    assert np.allclose(
        sd.heading_from_orientation(p, np.array([math.pi / 2]), np.array([0.0])), [[1, 0, 0]], atol=1e-5
    )
    p = np.array([[2e-6, 0.0, 1.0]])  # z x up = (0, 2e-6, 0): east is +y here, not +x
    assert np.allclose(
        sd.heading_from_orientation(p, np.array([math.pi / 2]), np.array([0.0])), [[0, 1, 0]], atol=1e-5
    )
    p = np.array([[5e-7, 0.0, 1.0]])  # below the threshold: east is forced to +x
    assert np.allclose(
        sd.heading_from_orientation(p, np.array([math.pi / 2]), np.array([0.0])), [[1, 0, 0]], atol=1e-5
    )


def test_roundtrip_random_headings():
    rng = np.random.default_rng(1)
    n = 500
    pos = _random_positions(rng, n)
    yaw = rng.uniform(-math.pi, math.pi, n)
    pitch = rng.uniform(-1.5, 1.5, n)
    fwd = sd.heading_from_orientation(pos, yaw, pitch)
    assert np.allclose(np.linalg.norm(fwd, axis=1), 1.0)
    y2, p2 = sd.bearing_of(pos, fwd)
    assert np.allclose(_wrap(y2 - yaw), 0, atol=1e-9)
    assert np.allclose(p2, pitch, atol=1e-9)
    assert ((y2 >= -math.pi) & (y2 < math.pi)).all()
    # pitch is the elevation above the tangent plane: sin(pitch) = forward . up
    up = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    assert np.allclose(np.sum(fwd * up, axis=1), np.sin(pitch))
    # bearing_of is scale invariant
    y3, p3 = sd.bearing_of(pos, 7.5 * fwd)
    assert np.allclose(_wrap(y3 - yaw), 0, atol=1e-9) and np.allclose(p3, pitch, atol=1e-9)


def test_roundtrip_at_the_poles():
    rng = np.random.default_rng(2)
    n = 200
    pos = np.zeros((n, 3))
    pos[:, 2] = np.where(np.arange(n) % 2 == 0, 1.0, -1.0) * rng.uniform(RP, RP + H, n)
    yaw = rng.uniform(-math.pi, math.pi, n)
    pitch = rng.uniform(-1.5, 1.5, n)
    y2, p2 = sd.bearing_of(pos, sd.heading_from_orientation(pos, yaw, pitch))
    assert np.allclose(_wrap(y2 - yaw), 0, atol=1e-9) and np.allclose(p2, pitch, atol=1e-9)


def test_roundtrip_special_values():
    pos = np.array([[11.0, 0, 0]] * 6)
    yaw = np.array([0.0, -math.pi, math.pi / 2, -math.pi / 2, 3.1415, 0.0])
    pitch = np.array([0.5, 0.0, -0.5, 1.0, 0.2, 0.0])
    y2, p2 = sd.bearing_of(pos, sd.heading_from_orientation(pos, yaw, pitch))
    assert np.allclose(_wrap(y2 - yaw), 0, atol=1e-9) and np.allclose(p2, pitch, atol=1e-9)
    assert y2[1] == -math.pi  # [-pi, pi): the cut maps +pi to -pi
    # straight up: the bearing is undefined (reported as 0), the pitch is pi/2
    y, p = sd.bearing_of(
        pos[:1], sd.heading_from_orientation(pos[:1], np.array([1.0]), np.array([math.pi / 2]))
    )
    assert y[0] == 0.0 and p[0] == pytest.approx(math.pi / 2)
    y, p = sd.bearing_of(pos[:1], -pos[:1])
    assert y[0] == 0.0 and p[0] == pytest.approx(-math.pi / 2)


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def test_rotation_about_z_equivariance():
    # z x (R up) = R (z x up) for a rotation R about z, so east, north and up all rotate with R: away from
    # the poles neither yaw nor pitch changes when the planet (positions and headings) is rotated about z.
    rng = np.random.default_rng(3)
    n = 300
    pos = _random_positions(rng, n)
    h = rng.normal(size=(n, 3))
    y0, p0 = sd.bearing_of(pos, h)
    for a in (0.3, -2.0, math.pi, 5.9):
        r = _rot_z(a)
        y1, p1 = sd.bearing_of(pos @ r.T, h @ r.T)
        assert np.allclose(_wrap(y1 - y0), 0, atol=1e-9)
        assert np.allclose(p1, p0, atol=1e-9)


def test_rotation_about_z_at_the_poles_shifts_yaw():
    # At a pole east is the fixed vector (1, 0, 0), which does not rotate with the planet, while the position
    # stays on the axis: the heading rotates by alpha against a fixed frame.  At the north pole (north = +y,
    # east = +x) the bearing is measured clockwise from +y, so a counter-clockwise rotation by alpha gives
    # yaw -> yaw - alpha; at the south pole (north = -y) the sense flips: yaw -> yaw + alpha.
    # Pitch is unchanged.
    rng = np.random.default_rng(4)
    n = 100
    for sign, shift in ((+1.0, -1.0), (-1.0, +1.0)):
        pos = np.zeros((n, 3))
        pos[:, 2] = sign * 11.0
        h = rng.normal(size=(n, 3))
        y0, p0 = sd.bearing_of(pos, h)
        for a in (0.4, -1.3, 3.0):
            r = _rot_z(a)
            y1, p1 = sd.bearing_of(pos @ r.T, h @ r.T)
            assert np.allclose(pos @ r.T, pos)
            assert np.allclose(_wrap(y1 - (y0 + shift * a)), 0, atol=1e-9)
            assert np.allclose(p1, p0, atol=1e-9)


def test_other_axis_rotation_is_not_equivariant():
    # the frame is tied to the planet's z axis: a rotation about x changes the bearing
    rng = np.random.default_rng(5)
    pos = _random_positions(rng, 50)
    h = rng.normal(size=(50, 3))
    c, s = math.cos(0.7), math.sin(0.7)
    rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    y0, _ = sd.bearing_of(pos, h)
    y1, _ = sd.bearing_of(pos @ rx.T, h @ rx.T)
    assert not np.allclose(_wrap(y1 - y0), 0, atol=1e-3)
    # but the pitch (elevation above the local tangent plane) is a rotation invariant of the pair (p, h)
    _, p0 = sd.bearing_of(pos, h)
    _, p1 = sd.bearing_of(pos @ rx.T, h @ rx.T)
    assert np.allclose(p1, p0, atol=1e-9)
