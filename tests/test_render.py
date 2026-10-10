"""WP6 rendering tests (§20.1, §25 WP6): rasteriser geometry and compositing, PNG writer, VideoWriter."""

from __future__ import annotations

import struct
import warnings
import zlib
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from alife import render
from alife.config import Config, dims
from alife.render import DEFAULT_LAYERS, VideoWriter, Viewport, render_frame, write_png
from alife.state import init_state, state_digest, to_numpy
from helpers import make_scene

SIZE = (96, 96)  # micro_cfg (L = 48): 2 px per unit, disc radius ρ = 2, 8 px per field cell
AGENTS = frozenset({"agents"})
BONDS = frozenset({"bonds"})
GREY = (128, 128, 128)
WHITE = (255, 255, 255)


def _frame(s, cfg: Config, size=SIZE, **kw) -> np.ndarray:
    img = render_frame(s, cfg, size, **kw)
    assert img.dtype == mx.uint8 and img.shape == (size[1], size[0], 3)
    return to_numpy(img)


def _disc(cx: int, cy: int, rho: int) -> set[tuple[int, int]]:
    """(row, col) pixels of the §20.1 disc stencil centred on pixel (cx, cy)."""
    return {
        (cy + dy, cx + dx)
        for dy in range(-rho, rho + 1)
        for dx in range(-rho, rho + 1)
        if dx * dx + dy * dy <= rho * rho
    }


def _lit(img: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in zip(*np.nonzero(img.any(axis=2)), strict=True)}


def _px(x: float, centre: float, scale: float, extent: int, L: float) -> int:
    """Reference world → pixel column/row for a sample that needs no wrapping."""
    d = x - centre
    d -= L * round(d / L)
    return int(np.floor(extent / 2 + d * scale))


# --- render_frame -----------------------------------------------------------------------------------------


def test_shape_dtype_and_purity(tiny_cfg: Config) -> None:
    s = init_state(tiny_cfg)
    before = state_digest(s)
    for size in ((64, 48), (17, 33)):
        img = render_frame(s, tiny_cfg, size)
        assert img.shape == (size[1], size[0], 3) and img.dtype == mx.uint8
    for color in render.COLOR_MODES:
        assert render_frame(s, tiny_cfg, (32, 32), color=color).shape == (32, 32, 3)
    assert render_frame(s, tiny_cfg, (32, 32), layers=frozenset()).shape == (32, 32, 3)
    assert state_digest(s) == before


def test_rejects_bad_arguments(micro_cfg: Config) -> None:
    s = make_scene(micro_cfg, [{"pos": (5.0, 5.0)}])
    with pytest.raises(ValueError, match="layers"):
        render_frame(s, micro_cfg, SIZE, layers=frozenset({"agents", "sky"}))
    with pytest.raises(ValueError, match="color"):
        render_frame(s, micro_cfg, SIZE, color="rainbow")
    with pytest.raises(ValueError, match="size"):
        render_frame(s, micro_cfg, (0, 10))
    with pytest.raises(ValueError, match="zoom"):
        render_frame(s, micro_cfg, SIZE, Viewport(1.0, 1.0, 0.0))


def test_single_agent_lights_expected_pixels(micro_cfg: Config) -> None:
    L = micro_cfg.world.size
    pos, tag = (12.25, 30.75), (0.5, -0.5, 1.0, 0.0)
    s = make_scene(micro_cfg, [{"pos": pos, "heading": 0.0, "tag": tag, "energy": 20000}])
    img = _frame(s, micro_cfg, layers=AGENTS)
    cx, cy = _px(pos[0], L / 2, 2.0, 96, L), _px(pos[1], L / 2, 2.0, 96, L)
    assert (cx, cy) == (24, 61)
    disc = _disc(cx, cy, 2)
    assert _lit(img) == disc
    nose = (cy, _px(pos[0] + 0.7, L / 2, 2.0, 96, L))  # heading 0: one unit of 0.7·r along +x
    assert nose == (61, 25) and nose in disc
    expect = tuple(int(np.floor(64 + 191 * (t + 1) / 2 + 0.5)) for t in tag[:3])
    for r, c in disc - {nose}:
        assert tuple(img[r, c]) == expect
    assert tuple(img[nose]) == WHITE

    # Hurt last step → red; the nose follows the heading (π/2: +y is down the image).
    s = make_scene(micro_cfg, [{"pos": pos, "heading": np.pi / 2, "tag": tag, "last_loss": 7}])
    img = _frame(s, micro_cfg, layers=AGENTS)
    assert _lit(img) == disc
    nose = (_px(pos[1] + 0.7, L / 2, 2.0, 96, L), cx)
    assert tuple(img[nose]) == WHITE
    assert all(tuple(img[p]) == (255, 0, 0) for p in disc - {nose})


def test_dead_rows_are_not_drawn(micro_cfg: Config) -> None:
    s = make_scene(micro_cfg, [{"pos": (10.0, 10.0)}])
    s = s._replace(alive=mx.zeros_like(s.alive))
    assert not _frame(s, micro_cfg, layers=DEFAULT_LAYERS - {"resource", "chem"}).any()


def test_priority_energy_and_order_independence(micro_cfg: Config) -> None:
    """Higher priority (energy) wins on overlap whatever the row order; the nose (255) beats every disc."""
    rich = {"pos": (20.0, 20.0), "heading": np.pi, "tag": (1.0, 1.0, 1.0, 0.0), "energy": 40000}
    poor = {"pos": (21.0, 20.0), "heading": np.pi, "tag": (-1.0, -1.0, -1.0, 0.0), "energy": 1000}
    a = _frame(make_scene(micro_cfg, [rich, poor]), micro_cfg, layers=AGENTS)
    b = _frame(make_scene(micro_cfg, [dict(poor, row=3), dict(rich, row=9)]), micro_cfg, layers=AGENTS)
    np.testing.assert_array_equal(a, b)
    rich_disc = _disc(48 - 8, 40, 2)  # x = 20 → col 40, y = 20 → row 40
    poor_disc = _disc(48 - 6, 40, 2)
    poor_nose = (40, _px(21.0 - 0.7, 24.0, 2.0, 96, 48.0))
    for p in rich_disc - {poor_nose, (40, 38)}:
        assert tuple(a[p]) == (255, 255, 255)  # tag +1 → 255 (and the rich nose at col 38 is white too)
    for p in poor_disc - rich_disc:
        assert tuple(a[p]) == (64, 64, 64)
    assert tuple(a[poor_nose]) == WHITE


def test_permuted_rows_give_identical_frames(micro_cfg: Config) -> None:
    rng = np.random.default_rng(3)
    n = 40
    agents = [
        {
            "pos": rng.uniform(0, 48, 2),
            "heading": rng.uniform(-np.pi, np.pi),
            "tag": rng.uniform(-1, 1, 4),
            "energy": int(rng.integers(1, 60000)),
            "clade": int(rng.integers(0, 8)),
            "last_loss": int(rng.integers(0, 2)),
        }
        for _ in range(n)
    ]
    agents[1]["pos"] = agents[0]["pos"] + 0.3  # guaranteed overlaps, including an exact priority tie
    agents[1]["energy"] = agents[0]["energy"]
    resource = rng.integers(0, 9000, (12, 12))
    s1 = make_scene(micro_cfg, agents, resource=resource, chem=rng.uniform(0, 5, (3, 12, 12)))
    rows = rng.permutation(dims(micro_cfg).n_cap)[:n]
    s2 = make_scene(
        micro_cfg,
        [dict(a, row=int(r)) for a, r in zip(agents, rows, strict=True)],
        resource=resource,
        chem=to_numpy(s1.chem),
    )
    for color in render.COLOR_MODES:
        f1 = _frame(s1, micro_cfg, color=color)
        np.testing.assert_array_equal(f1, _frame(s1, micro_cfg, color=color))  # repeatable
        np.testing.assert_array_equal(f1, _frame(s2, micro_cfg, color=color))  # row-order independent


def test_bond_across_seam_drawn_on_both_sides(micro_cfg: Config) -> None:
    s = make_scene(micro_cfg, [{"pos": (0.5, 24.0)}, {"pos": (47.5, 24.0)}], bonds=[(0, 1)])
    img = _frame(s, micro_cfg, layers=BONDS)
    lit = _lit(img)
    assert lit and {r for r, _ in lit} == {48}
    cols = {c for _, c in lit}
    assert cols & {0, 1} and cols & {94, 95}  # both edges of the seam
    assert all(c <= 1 or c >= 94 for c in cols)  # the minimum-image segment, not the long way round
    assert all(tuple(img[p]) == GREY for p in lit)
    # An ordinary bond lies on its segment and sits under the agent discs.
    s = make_scene(micro_cfg, [{"pos": (10.0, 10.0)}, {"pos": (13.0, 10.0)}], bonds=[(0, 1)])
    img = _frame(s, micro_cfg, layers=BONDS)
    assert {c for _, c in _lit(img)} <= set(range(20, 27)) and {r for r, _ in _lit(img)} == {20}
    full = _frame(s, micro_cfg, layers=AGENTS | BONDS)
    assert any(tuple(full[p]) == GREY for p in _lit(img))
    assert all(tuple(full[p]) != GREY for p in _disc(20, 20, 2) | _disc(26, 20, 2))
    # A dead or absent partner draws nothing.
    s = make_scene(micro_cfg, [{"pos": (10.0, 10.0)}])
    assert not _frame(s, micro_cfg, layers=BONDS).any()


def test_disc_wraps_across_frame_edges(micro_cfg: Config) -> None:
    s = make_scene(micro_cfg, [{"pos": (0.1, 47.9)}])
    lit = _lit(_frame(s, micro_cfg, layers=AGENTS))
    assert lit == {(r % 96, c % 96) for r, c in _disc(0, 95, 2)}


def test_viewport_culling_pan_and_zoom(micro_cfg: Config) -> None:
    vp = Viewport(40.0, 40.0, 4.0)  # 8 px per unit: x, y ∈ [34, 46) visible, ρ = 8
    far = make_scene(micro_cfg, [{"pos": (10.0, 10.0)}])
    assert not _frame(far, micro_cfg, viewport=vp, layers=AGENTS).any()
    centre = make_scene(micro_cfg, [{"pos": (40.0, 40.0)}])
    lit = _lit(_frame(centre, micro_cfg, viewport=vp, layers=AGENTS))
    assert lit == _disc(48, 48, 8)
    off = make_scene(micro_cfg, [{"pos": (47.5, 40.0)}])  # disc spans cols 100..116: just off the frame
    assert not _frame(off, micro_cfg, viewport=vp, layers=AGENTS).any()
    edge = make_scene(micro_cfg, [{"pos": (46.5, 40.0)}])  # centre col 100: only cols 92..95 visible
    lit = _lit(_frame(edge, micro_cfg, viewport=vp, layers=AGENTS))
    assert lit == {(r, c) for r, c in _disc(100, 48, 8) if c < 96}

    # Panning across the seam: the view centred on x = 0 shows x = 47 left of centre and x = 1 right of it.
    s = make_scene(micro_cfg, [{"pos": (47.0, 24.0)}, {"pos": (1.0, 24.0), "tag": (1.0, 1.0, 1.0, 0.0)}])
    img = _frame(s, micro_cfg, viewport=Viewport(0.0, 24.0), layers=AGENTS)
    assert _lit(img) == _disc(46, 48, 2) | _disc(50, 48, 2)
    assert tuple(img[48, 52]) == (255, 255, 255) and tuple(img[48, 44]) != (255, 255, 255)
    # The viewport centre is itself taken modulo L.
    np.testing.assert_array_equal(img, _frame(s, micro_cfg, viewport=Viewport(-48.0, 72.0), layers=AGENTS))

    # Zooming out tiles the torus: zoom 0.5 shows each agent 2×2 times (1 px per unit, ρ = 1).
    one = make_scene(micro_cfg, [{"pos": (10.5, 30.5)}])
    lit = _lit(_frame(one, micro_cfg, viewport=Viewport(24.0, 24.0, 0.5), layers=AGENTS))
    cx, cy = _px(10.5, 24.0, 1.0, 96, 48.0), _px(30.5, 24.0, 1.0, 96, 48.0)
    want = set()
    for kx in (0, 48):
        for ky in (-48, 0, 48):
            want |= {(r, c) for r, c in _disc(cx + kx, cy + ky, 1) if 0 <= r < 96 and 0 <= c < 96}
    assert lit == want and len(lit) == 4 * 5


def test_non_square_frame_tiles_long_axis(micro_cfg: Config) -> None:
    s = make_scene(micro_cfg, [{"pos": (24.0, 24.0)}])
    lit = _lit(_frame(s, micro_cfg, size=(192, 96), layers=AGENTS))  # scale 2: the world spans 96 px
    want = _disc(96, 48, 2) | _disc(0, 48, 2) | _disc(192, 48, 2)
    assert lit == {(r, c) for r, c in want if 0 <= c < 192}


def test_background_layers(micro_cfg: Config) -> None:
    res = (np.arange(144).reshape(12, 12) * 60).astype(np.int32)  # 0 … 8580, past resource_cap = 8000
    s = make_scene(micro_cfg, [], resource=res)
    img = _frame(s, micro_cfg, layers=frozenset({"resource"}))
    cells = res[np.arange(96)[:, None] // 8, np.arange(96)[None, :] // 8]  # 8 px per field cell
    f32 = np.float32  # the reference follows the renderer's float32 arithmetic (x.5 cases round alike)
    g = np.floor(f32(30) + f32(150) * np.minimum(cells.astype(f32) / f32(8000), f32(1)) + f32(0.5))
    np.testing.assert_array_equal(img[..., 1], g.astype(np.uint8))
    assert not img[..., 0].any() and not img[..., 2].any()

    chem = np.zeros((3, 12, 12), np.float32)
    chem[0], chem[1], chem[2] = 1.0, 0.0, 3.0
    s = make_scene(micro_cfg, [], resource=4000, chem=chem)
    img = _frame(s, micro_cfg, layers=frozenset({"resource", "chem"}))
    t0, t2 = 90 * 1 / 2, 90 * 3 / 4
    want = (np.floor(t0 + 0.6 * t2 + 0.5), 105, np.floor(t2 + 0.5))  # c0 red + c2 violet; G = 30 + 75
    assert (img.reshape(-1, 3) == np.array(want, np.uint8)).all()
    assert not _frame(s, micro_cfg, layers=frozenset()).any()


def test_color_modes(micro_cfg: Config) -> None:
    s = make_scene(micro_cfg, [{"pos": (24.3, 24.3), "clade": 5, "energy": micro_cfg.energy.e_max}])
    centre = (48, 48)
    h = (5 * 2654435761) % 2**32 & 0xFFFFFF
    img = _frame(s, micro_cfg, layers=AGENTS, color="clade")
    assert tuple(img[centre]) == (h >> 16, (h >> 8) & 255, h & 255)
    img = _frame(s, micro_cfg, layers=AGENTS, color="energy")
    assert tuple(img[centre]) == (255, 220, 40)
    lo = make_scene(micro_cfg, [{"pos": (24.3, 24.3), "energy": 100}])
    dim = _frame(lo, micro_cfg, layers=AGENTS, color="energy")[centre].astype(int)
    assert dim.sum() < img[centre].astype(int).sum()


def test_large_stencil_chunks_match(micro_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """Splitting the disc scatter into row chunks (bounded memory at deep zoom) leaves the frame unchanged."""
    rng = np.random.default_rng(5)
    s = make_scene(
        micro_cfg, [{"pos": rng.uniform(20, 28, 2), "tag": rng.uniform(-1, 1, 4)} for _ in range(12)]
    )
    vp = Viewport(24.0, 24.0, 3.0)
    whole = _frame(s, micro_cfg, viewport=vp)
    monkeypatch.setattr(render, "_SAMPLE_BUDGET", 300)  # ρ = 6: 113-pixel stencil → 2 rows per chunk
    np.testing.assert_array_equal(whole, _frame(s, micro_cfg, viewport=vp))


def _culling_scene(cfg: Config):
    rng = np.random.default_rng(9)
    pos = rng.uniform(0, 48, (40, 2))
    pos[:6] = [(0.2, 24.0), (47.8, 24.0), (24.0, 0.1), (24.0, 47.9), (0.3, 47.7), (47.6, 0.4)]  # seams
    agents = [{"pos": p, "tag": rng.uniform(-1, 1, 4), "heading": rng.uniform(-3, 3)} for p in pos]
    return make_scene(cfg, agents, bonds=[(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)])


@pytest.mark.parametrize(
    ("size", "vp"),
    [
        (SIZE, Viewport(0.0, 0.0, 1.0)),
        (SIZE, Viewport(24.0, 24.0, 0.5)),
        (SIZE, Viewport(47.5, 3.0, 4.0)),
        (SIZE, Viewport(10.0, 40.0, 13.0)),
        ((150, 70), Viewport(47.0, 1.0, 2.7)),
        ((70, 150), Viewport(1.0, 46.0, 9.0)),
    ],
)
def test_culling_is_exact(micro_cfg: Config, monkeypatch: pytest.MonkeyPatch, size, vp: Viewport) -> None:
    """Culling rows whose stencil misses the frame changes no pixel (seams, tiling, deep zoom, non-square)."""
    s = _culling_scene(micro_cfg)
    culled = _frame(s, micro_cfg, size=size, viewport=vp)
    monkeypatch.setattr(render, "_maybe_visible", lambda coord, *a: mx.ones(coord.shape, dtype=mx.bool_))
    np.testing.assert_array_equal(culled, _frame(s, micro_cfg, size=size, viewport=vp))


def test_deep_zoom_splats_only_visible_rows(micro_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """At deep zoom the disc scatter covers the few rows near the view, not all N (cost was ∝ zoom² × N)."""
    s = _culling_scene(micro_cfg)
    rows: list[int] = []
    once = render._splat_once
    monkeypatch.setattr(render, "_splat_once", lambda c, x, *a: rows.append(x.shape[0]) or once(c, x, *a))
    p = to_numpy(s.pos)[10]
    img = _frame(s, micro_cfg, viewport=Viewport(float(p[0]), float(p[1]), 40.0), layers=AGENTS)
    assert img.any() and 0 < sum(rows) <= 4, rows  # agent 10's disc and nose (plus any close neighbour)


# --- PNG ------------------------------------------------------------------------------------------------


def _read_png(path: Path) -> np.ndarray:
    """Minimal PNG decoder for the files write_png produces; verifies every chunk CRC."""
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, chunks = 8, []
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind, body = data[pos + 4 : pos + 8], data[pos + 8 : pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length : pos + 12 + length])
        assert crc == zlib.crc32(kind + body) & 0xFFFFFFFF, kind
        chunks.append((kind, body))
        pos += 12 + length
    assert chunks[0][0] == b"IHDR" and chunks[-1] == (b"IEND", b"")
    w, h, depth, ctype, comp, filt, interlace = struct.unpack(">IIBBBBB", chunks[0][1])
    assert (depth, comp, filt, interlace) == (8, 0, 0, 0)
    ch = {0: 1, 2: 3, 6: 4}[ctype]
    raw = zlib.decompress(b"".join(b for k, b in chunks if k == b"IDAT"))
    rows = np.frombuffer(raw, np.uint8).reshape(h, 1 + w * ch)
    assert (rows[:, 0] == 0).all()
    img = rows[:, 1:].reshape(h, w, ch)
    return img[:, :, 0] if ch == 1 else img


@pytest.mark.parametrize("shape", [(5, 7, 3), (1, 1, 3), (6, 3, 4), (4, 9)])
def test_write_png_roundtrip(tmp_path: Path, shape: tuple[int, ...]) -> None:
    img = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    path = tmp_path / "x.png"
    write_png(path, img)
    np.testing.assert_array_equal(_read_png(path), img)
    assert [p.name for p in tmp_path.iterdir()] == ["x.png"]  # no temp file left behind


def test_write_png_frame_and_validation(tmp_path: Path, micro_cfg: Config) -> None:
    s = make_scene(micro_cfg, [{"pos": (5.0, 5.0)}])
    frame = render_frame(s, micro_cfg, (40, 30))
    write_png(tmp_path / "f.png", frame)  # an mx.array is accepted (read through to_numpy)
    np.testing.assert_array_equal(_read_png(tmp_path / "f.png"), to_numpy(frame))
    write_png(tmp_path / "t.png", to_numpy(frame)[:, ::2])  # non-contiguous view
    np.testing.assert_array_equal(_read_png(tmp_path / "t.png"), to_numpy(frame)[:, ::2])
    with pytest.raises(ValueError, match="uint8"):
        write_png(tmp_path / "bad.png", np.zeros((2, 2, 3), np.float32))
    with pytest.raises(ValueError, match="image must be"):
        write_png(tmp_path / "bad.png", np.zeros((2, 2, 2), np.uint8))


# --- VideoWriter ----------------------------------------------------------------------------------------


def _frames(n: int, w: int, h: int) -> list[np.ndarray]:
    rng = np.random.default_rng(1)
    return [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def test_video_writer_falls_back_to_png_sequence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render, "_find_ffmpeg", lambda: None)
    frames = _frames(3, 8, 6)
    with pytest.warns(UserWarning, match="ffmpeg not found"):
        vw = VideoWriter(tmp_path / "out" / "v.mp4", (8, 6), fps=10)
    assert vw.fallback and vw.out_dir == tmp_path / "out" / "v"
    for f in frames:
        vw.write(f)
    vw.close()
    vw.close()  # idempotent
    names = sorted(p.name for p in (tmp_path / "out" / "v").iterdir())
    assert names == ["frame_000000.png", "frame_000001.png", "frame_000002.png"]
    for name, f in zip(names, frames, strict=True):
        np.testing.assert_array_equal(_read_png(tmp_path / "out" / "v" / name), f)
    assert not (tmp_path / "out" / "v.mp4").exists()
    with pytest.raises(ValueError, match="closed"):
        vw.write(frames[0])


def test_video_writer_unstartable_ffmpeg_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render, "_find_ffmpeg", lambda: str(tmp_path / "no-such-ffmpeg"))
    with pytest.warns(UserWarning, match="could not start ffmpeg"):
        vw = VideoWriter(tmp_path / "v.mp4", (4, 4))
    assert vw.fallback
    vw.write(_frames(1, 4, 4)[0])
    vw.close()
    assert (tmp_path / "v" / "frame_000000.png").exists()


def test_video_writer_directory_and_validation(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with VideoWriter(tmp_path / "frames", (5, 4)) as vw:
            assert vw.fallback
            vw.write(_frames(1, 5, 4)[0])
            with pytest.raises(ValueError, match="frame must be"):
                vw.write(np.zeros((5, 4, 3), np.uint8))  # transposed
    assert vw.frames == 1 and (tmp_path / "frames" / "frame_000000.png").exists()
    with pytest.raises(ValueError, match="suffix"):
        VideoWriter(tmp_path / "v.gif", (4, 4))
    with pytest.raises(ValueError, match="size"):
        VideoWriter(tmp_path / "frames2", (0, 4))


def test_video_writer_reports_ffmpeg_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = tmp_path / "ffmpeg"
    fake.write_text("#!/bin/sh\necho boom >&2\nexit 3\n")
    fake.chmod(0o755)
    monkeypatch.setattr(render, "_find_ffmpeg", lambda: str(fake))
    vw = VideoWriter(tmp_path / "v.mp4", (64, 64))
    assert not vw.fallback
    with pytest.raises(RuntimeError, match="boom"):
        for f in _frames(64, 64, 64):  # 768 KiB: more than a pipe buffer, so a write or close must fail
            vw.write(f)
        vw.close()
    vw.close()  # already closed: no second error


@pytest.mark.skipif(render._find_ffmpeg() is None, reason="no ffmpeg binary available")
def test_video_writer_ffmpeg(tmp_path: Path) -> None:
    path = tmp_path / "v.mp4"
    with VideoWriter(path, (33, 17), fps=5) as vw:  # odd size: padded to even for yuv420p
        assert not vw.fallback
        for f in _frames(4, 33, 17):
            vw.write(f)
    assert vw.frames == 4 and path.stat().st_size > 0
    assert path.read_bytes()[4:8] == b"ftyp"
