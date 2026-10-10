"""WP6 viewer tests (§20.2): the headless ``ViewerSession`` always; the pygame loop only with pygame-ce."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from alife import viewer
from alife.config import Config
from alife.render import COLOR_MODES, DEFAULT_LAYERS
from alife.runio import RunDir, checkpoint_metadata, load_checkpoint
from alife.state import init_state, state_digest, to_numpy
from alife.step import make_step
from alife.viewer import ViewerSession
from helpers import make_scene

SIZE = (96, 96)  # micro_cfg (L = 48): 2 px per unit at zoom 1


@pytest.fixture(autouse=True)
def _no_frame_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exact per-frame step counts below: a cold compile must not cut a frame short."""
    monkeypatch.setattr(viewer, "FRAME_BUDGET_S", math.inf)


def test_frame_budget_bounds_steps_per_frame(micro_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """Past ``FRAME_BUDGET_S`` a frame stops stepping (after ≥ 1 step), so the loop gets back to its events;
    how steps split into frames never changes the trajectory."""
    full = ViewerSession(micro_cfg, init_state(micro_cfg), size=SIZE, steps_per_frame=8)
    assert full.tick() and int(to_numpy(full.state.step)) == 8
    monkeypatch.setattr(viewer, "FRAME_BUDGET_S", 0.0)
    cut = ViewerSession(micro_cfg, init_state(micro_cfg), size=SIZE, steps_per_frame=8)
    for k in range(1, 9):
        assert cut.tick() and int(to_numpy(cut.state.step)) == k
    assert state_digest(cut.state) == state_digest(full.state)
    assert cut.advance(3) == 3  # explicit advances (``.``) ignore the budget


def _tree(path: Path) -> set[str]:
    return {str(p.relative_to(path)) for p in path.rglob("*")}


def test_module_import_does_not_import_pygame() -> None:
    code = "import sys, alife.viewer; assert 'pygame' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_missing_pygame_gives_install_hint(micro_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pygame", None)  # makes `import pygame` raise ImportError
    with pytest.raises(ImportError, match="viewer"):
        viewer.run_viewer(micro_cfg, init_state(micro_cfg))


def test_viewing_never_changes_the_trajectory(micro_cfg: Config, tmp_path: Path) -> None:
    fn = make_step(micro_cfg)
    s = init_state(micro_cfg)
    for _ in range(6):
        s, out = fn(s)
        mx.eval(s, out)
    bare = state_digest(s)

    vs = ViewerSession(micro_cfg, init_state(micro_cfg), size=SIZE, steps_per_frame=2)
    vs.rundir = RunDir.create(tmp_path / "run", micro_cfg)
    vs.render()
    assert vs.tick()  # 2 steps
    vs.select_at(10, 10)
    vs.zoom_at(30, 40, 2.0)
    vs.handle_key("c")
    vs.handle_key("s")
    vs.render()
    vs.handle_key(" ")  # pause
    assert not vs.tick()
    vs.handle_key(".")  # single step while paused
    vs.handle_key("p")
    vs.handle_key(" ")  # resume
    vs.tick()  # 2 steps
    vs.handle_key(".")  # pauses again and steps once
    frame, info = vs.render()
    assert info.step == 6 and vs.paused
    assert state_digest(vs.state) == bare
    assert frame.shape == (SIZE[1], SIZE[0], 3) and frame.dtype == np.uint8


def test_hud_values_and_interval(micro_cfg: Config) -> None:
    vs = ViewerSession(micro_cfg, init_state(micro_cfg), size=SIZE)
    _, info = vs.render()
    assert (info.step, info.n_alive, info.bonds, info.interval_steps) == (
        0,
        micro_cfg.population.n_init,
        0,
        0,
    )
    assert info.selected is None and info.diag == 0
    every = micro_cfg.run.metrics_every
    vs.advance(every - 1)
    _, info = vs.render()
    assert info.interval_steps == every - 1  # partial interval until the first one completes
    vs.advance(1)
    _, info = vs.render()
    assert info.interval_steps == every  # latched
    vs.advance(1)
    _, info = vs.render()
    assert info.interval_steps == every and info.step == every + 1
    text = " ".join(line for line, _ in vs.hud_lines(info))
    assert f"step {every + 1}" in text and "steps/s" in text and "births/deaths" in text


def test_hud_flags_errors_and_extinction(micro_cfg: Config) -> None:
    s = make_scene(micro_cfg, [], diag=mx.array(1 | (1 << 16), dtype=mx.int32))
    vs = ViewerSession(micro_cfg, s, size=SIZE)
    _, info = vs.render()
    lines = vs.hud_lines(info)
    assert "EXTINCT" in lines[0][0]
    errors = [text for text, err in lines if err]
    assert errors == ["INVARIANT VIOLATION: diag LEDGER"]
    assert vs.select_at(5, 5) is None


def test_select_nearest_with_minimum_image(micro_cfg: Config) -> None:
    agents = [
        {"pos": (1.0, 24.0), "energy": 1234, "generation": 3, "clade": 5, "birth_step": -1},
        {"pos": (24.0, 24.0)},
        {"pos": (40.0, 24.0)},
    ]
    s = make_scene(micro_cfg, agents, bonds=[(0, 1)], step=mx.array(7, dtype=mx.int32))
    vs = ViewerSession(micro_cfg, s, size=SIZE)
    # Pixel column 95 is world x ≈ 47.75: across the seam agent 0 (x = 1) is nearest, not agent 2.
    assert vs.select_at(95, 48) == 0
    _, info = vs.render()
    a = info.selected
    assert a is not None
    assert (a.uid, a.energy, a.age, a.generation, a.clade, a.bonds) == (0, 1234, 7, 3, 5, 1)
    assert (a.x, a.y) == (1.0, 24.0)
    assert "uid 0  energy 1234  age 7  gen 3  clade 5  bonds 1" in [t for t, _ in vs.hud_lines(info)]
    assert vs.select_at(*vs.world_to_screen(39.0, 25.0)) == 2


def test_selected_agent_death_is_reported(micro_cfg: Config) -> None:
    s = make_scene(micro_cfg, [{"pos": (10.0, 10.0)}, {"pos": (30.0, 30.0)}])
    vs = ViewerSession(micro_cfg, s, size=SIZE)
    assert vs.select_at(*vs.world_to_screen(10.0, 10.0)) == 0
    vs._state = s._replace(alive=mx.array(np.arange(micro_cfg.population.n_max) == 1))
    _, info = vs.render()
    assert info.selected is None
    assert "uid 0 has died" in [t for t, _ in vs.hud_lines(info)]


def test_pan_and_zoom_geometry(micro_cfg: Config) -> None:
    vs = ViewerSession(micro_cfg, init_state(micro_cfg), size=SIZE)
    L = micro_cfg.world.size
    assert vs.scale == 2.0
    assert vs.screen_to_world(48, 48) == (24.25, 24.25)
    for px, py in ((0, 0), (17, 80), (95, 95)):
        assert vs.world_to_screen(*vs.screen_to_world(px, py)) == (px + 0.5, py + 0.5)

    vs.pan(10, -4)  # drag right/up: the world follows the cursor
    assert (vs.viewport.cx, vs.viewport.cy) == (19.0, 26.0)
    vs.pan(-100, 0)  # wraps around the torus
    assert vs.viewport.cx == pytest.approx((19.0 + 50.0) % L)

    before = vs.screen_to_world(20, 70)
    vs.zoom_at(20, 70, 3.0)
    assert vs.viewport.zoom == 3.0
    assert vs.screen_to_world(20, 70) == pytest.approx(before, abs=1e-9)
    vs.zoom_at(0, 0, 1e9)
    assert vs.viewport.zoom == viewer.ZOOM_MAX
    vs.zoom_at(0, 0, 1e-12)
    assert vs.viewport.zoom == viewer.ZOOM_MIN
    with pytest.raises(ValueError):
        vs.resize((0, 10))


def test_keys(micro_cfg: Config) -> None:
    vs = ViewerSession(micro_cfg, init_state(micro_cfg), size=SIZE)
    assert vs.layers == DEFAULT_LAYERS
    vs.handle_key("1")
    assert vs.layers == DEFAULT_LAYERS - {"agents"}
    for k in "234":
        vs.handle_key(k)
    assert vs.layers == frozenset()
    frame, _ = vs.render()
    assert not frame.any()  # nothing drawn, black base
    vs.handle_key("1")
    assert vs.layers == {"agents"}

    assert [vs.color] + [(vs.handle_key("c"), vs.color)[1] for _ in COLOR_MODES] == [*COLOR_MODES, "tag"]
    for _ in range(12):
        vs.handle_key("+")
    assert vs.steps_per_frame == viewer.SPF_MAX
    for _ in range(12):
        vs.handle_key("-")
    assert vs.steps_per_frame == 1
    vs.handle_key("=")
    assert vs.steps_per_frame == 2
    vs.dirty = False
    vs.handle_key("x")  # unknown keys are ignored
    assert not vs.dirty


def test_snapshot_into_run_directory(micro_cfg: Config, tmp_path: Path) -> None:
    rd = RunDir.create(tmp_path / "run", micro_cfg)
    rd.save_checkpoint(init_state(micro_cfg), micro_cfg)
    before = _tree(rd.path)
    vs = ViewerSession(micro_cfg, init_state(micro_cfg), rundir=rd, size=SIZE)
    vs.advance(3)
    path = vs.snapshot()
    assert path == rd.path / "media" / "snap_3.safetensors"
    assert vs.save_png() == rd.path / "media" / "view_3.png"
    assert _tree(rd.path) - before == {"media", "media/snap_3.safetensors", "media/view_3.png"}
    assert (rd.path / "checkpoints" / "LATEST").read_text().strip() == "step_0000000000.safetensors"
    assert checkpoint_metadata(path)["step"] == "3"
    assert state_digest(load_checkpoint(path, micro_cfg)) == state_digest(vs.state)
    png = (rd.path / "media" / "view_3.png").read_bytes()
    assert png.startswith(b"\x89PNG\r\n\x1a\n")


def test_snapshot_without_run_directory(
    micro_cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    vs = ViewerSession(micro_cfg, init_state(micro_cfg), size=SIZE)
    vs.handle_key("s")
    assert (tmp_path / "media" / "snap_0.safetensors").is_file()
    assert "snap_0.safetensors" in vs.message


# --- the pygame loop (headless SDL) -----------------------------------------------------------------------


def test_run_viewer_headless(micro_cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pg = pytest.importorskip("pygame")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    rd = RunDir.create(tmp_path / "run", micro_cfg)
    seen: list[ViewerSession] = []
    real_session = viewer.ViewerSession

    def spy(*args, **kwargs):
        seen.append(real_session(*args, **kwargs))
        return seen[-1]

    monkeypatch.setattr(viewer, "ViewerSession", spy)

    def key(k: int, ch: str = "") -> object:
        return pg.event.Event(pg.KEYDOWN, key=k, unicode=ch, mod=0, scancode=0)

    def btn(kind: int, pos: tuple[int, int]) -> object:
        return pg.event.Event(kind, pos=pos, button=1)

    script = [
        [],
        [key(pg.K_c, "c"), key(pg.K_1, "1"), pg.event.Event(pg.MOUSEWHEEL, x=0, y=1, flipped=False)],
        [btn(pg.MOUSEBUTTONDOWN, (40, 40)), btn(pg.MOUSEBUTTONUP, (41, 40))],  # click: select
        [
            btn(pg.MOUSEBUTTONDOWN, (10, 10)),
            pg.event.Event(pg.MOUSEMOTION, pos=(30, 10), rel=(20, 0), buttons=(1, 0, 0)),
            btn(pg.MOUSEBUTTONUP, (30, 10)),
        ],  # drag: pan
        [key(pg.K_SPACE, " ")],
        [],  # paused and idle
        [key(pg.K_PERIOD, "."), key(pg.K_KP_PLUS), key(pg.K_s, "s"), key(pg.K_p, "p")],
        [pg.event.Event(pg.VIDEORESIZE, size=(80, 64), w=80, h=64)],
        [key(pg.K_ESCAPE)],
    ]
    frames = iter(script)
    monkeypatch.setattr(pg.event, "get", lambda *a, **k: next(frames))

    viewer.run_viewer(micro_cfg, init_state(micro_cfg), rundir=rd, size=SIZE, steps_per_frame=2)

    (vs,) = seen
    assert next(frames, None) is None  # the whole script ran
    step = int(to_numpy(vs.state.step))
    assert step == 2 * 4 + 1  # four running frames of 2 steps, then one single step while paused
    assert vs.paused and vs.color == "energy" and "agents" not in vs.layers
    assert vs.viewport.zoom == viewer.ZOOM_STEP and vs.viewport.cx != micro_cfg.world.size / 2
    assert vs.selected_uid is not None and vs.steps_per_frame == 4
    assert vs.size == (80, 64) and vs.frame is not None and vs.frame.shape == (64, 80, 3)
    assert (rd.path / "media" / f"snap_{step}.safetensors").is_file()
    assert (rd.path / "media" / f"view_{step}.png").is_file()
    assert not pg.display.get_init()
