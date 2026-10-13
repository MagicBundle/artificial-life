"""`alife stream` (DESIGN §27.11): the frame protocol, the control channel, the sources, `--follow`, purity.

The protocol is checked against `stream_decode`, a reader written from the specification text. Most tests
run the streamer in process into a `BytesIO` on small hand-built scenes (`stream_fixtures.scene_agents`:
zero-weight agents, so the stepped trajectories are cheap and exact), with one compiled step shared by the
module. The process-level behaviour (a closed reader, nothing but protocol bytes on stdout) runs through
`cli.main` with fd 1 captured and, where a real pipe is the point, through a subprocess that forces the CPU
device first (the Mac GPU belongs to the evaluation arms).
"""

from __future__ import annotations

import io
import json
import os
import struct
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

import stream_decode as dec
import stream_fixtures as fx
from alife import cli, render, stream
from alife.config import Config, config_hash, dims, load_config
from alife.runio import RunDir, write_state_file
from alife.state import State, State3, init_state, state_digest, to_numpy
from alife.step import make_step
from helpers import MICRO3D, ON_GPU, make_scene

KINDS = ("2d", "3d")


# --- fixtures and helpers -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfgs(micro_cfg: Config, micro3d_cfg: Config) -> dict[str, Config]:
    return {"2d": micro_cfg, "3d": micro3d_cfg}


@pytest.fixture(scope="module")
def steps(cfgs: dict[str, Config]) -> dict[str, Callable[[Any], Any]]:
    """One compiled step per dimension, shared by every test of the module."""
    return {k: make_step(c) for k, c in cfgs.items()}


def scene(cfg: Config, kind: str, step: int = 0) -> State | State3:
    """Seven zero-weight agents in rows 0-4, 10 and 21 with bonds (0,1) across the seam, (2,3) and (4,10)."""
    return make_scene(
        cfg,
        fx.scene_agents(kind, 0),
        bonds=[(0, 1), (2, 3), (4, 10)],
        step=mx.array(step, dtype=mx.int32),
    )


def advance(step_fn: Callable[[Any], Any], s: State | State3, n: int) -> State | State3:
    for _ in range(n):
        s, out = step_fn(s)
        mx.eval(s, out)
    return s


def run_sim(
    cfg: Config,
    state: State | State3,
    step_fn: Callable[[Any], Any],
    *,
    steps: int | None = None,
    every: int = 1,
    control: Any = None,
    view: stream.View | None = None,
    **kw: Any,
) -> tuple[bytes, stream.StreamStats]:
    buf = io.BytesIO()
    stats = stream.serve(
        cfg,
        mode="simulate",
        source="preset",
        run=None,
        state=[state],
        view=view,
        every=every,
        max_steps=steps,
        control=control,
        out=buf,
        step_fn=step_fn,
        **kw,
    )
    return buf.getvalue(), stats


class ScriptControl:
    """Poll `k` returns `script[k]` (a list of lines or a callable returning one); a spent script is EOF."""

    def __init__(self, script: Sequence[Any]) -> None:
        self.script = list(script)
        self.timeouts: list[float | None] = []
        self.closed = False

    def poll(self, timeout: float | None = 0) -> list[str]:
        self.timeouts.append(timeout)
        k = len(self.timeouts) - 1
        item = self.script[k] if k < len(self.script) else ["quit"]
        return list(item() if callable(item) else item)

    def close(self) -> None:
        self.closed = True


def snapshot(s: State | State3) -> dict[str, np.ndarray]:
    """What a frame shows, as numpy copies."""
    names = ["alive", "pos", "heading", "partner", "tag", "energy", "clade", "last_loss", "step", "diag"]
    if isinstance(s, State3):
        names.append("pitch")
    return {k: to_numpy(getattr(s, k)) for k in names}


def ref_bonds(h: dict[str, np.ndarray]) -> list[list[int]]:
    """Each bond once, from its lower row, both ends alive; ascending row then slot (written out by loops)."""
    out = []
    for i in range(len(h["alive"])):
        for b in range(h["partner"].shape[1]):
            j = int(h["partner"][i, b])
            if j > i and h["alive"][i] and h["alive"][j]:
                out.append([i, j])
    return out


def ref_rgb(h: dict[str, np.ndarray], cfg: Config, color: str) -> np.ndarray:
    """Agent colours from the palette of DESIGN §20.1, in numpy (energy to within one level per channel)."""
    f32 = np.float32
    if color == "tag":
        t = np.clip(h["tag"][:, :3].astype(f32), f32(-1), f32(1))
        c = np.floor(f32(64) + f32(191) * (t + f32(1)) / f32(2) + f32(0.5)).astype(np.uint32)
        packed = (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]
    elif color == "clade":
        packed = (h["clade"].astype(np.uint32) * np.uint32(2654435761)) & np.uint32(0xFFFFFF)
    else:
        raise AssertionError(color)
    return np.where(h["last_loss"] > 0, np.uint32(0xFF0000), packed).astype(np.uint32)


def check_frame(fr: dec.Frame, h: dict[str, np.ndarray], cfg: Config, color: str = "tag") -> None:
    """The frame shows exactly the state `h`: head, alive mask, positions, yaw, pitch, bonds and colours."""
    assert fr.step == int(h["step"]) and fr.n_alive == int(h["alive"].sum())
    np.testing.assert_array_equal(fr.alive, h["alive"])
    nd = h["pos"].shape[1]
    np.testing.assert_array_equal(fr.pos[:, :nd], h["pos"])
    np.testing.assert_array_equal(fr.orient[:, 0], h["heading"].astype(np.float16))
    if nd == 3:
        np.testing.assert_array_equal(fr.pos[:, 2], h["pos"][:, 2])
        np.testing.assert_array_equal(fr.orient[:, 1], h["pitch"].astype(np.float16))
    else:  # a 2D stream carries z = 0 and pitch = 0 on every row
        assert not fr.pos[:, 2].any() and not fr.orient[:, 1].any()
    assert fr.bonds.tolist() == ref_bonds(h) and fr.n_bonds == len(ref_bonds(h))
    np.testing.assert_array_equal(fr.rgb, ref_rgb(h, cfg, color))
    assert bool(fr.flags & 2) == bool(int(h["diag"]) & 0xFFFF)


def micro_args(kind: str) -> list[str]:
    preset = "tiny3d" if kind == "3d" else "tiny"
    return ["--preset", preset, *(x for o in MICRO3D for x in ("--set", o))]


def stdout_frames(capfdbinary: pytest.CaptureFixture[bytes]) -> tuple[dict[str, Any], list[dec.Frame], bytes]:
    """The captured fd 1, strictly decoded (every byte accounted for)."""
    out = capfdbinary.readouterr().out
    header, frames = dec.decode(out)
    return header, frames, out


# --- the protocol ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_header(kind: str, cfgs: dict[str, Config], steps: dict[str, Any]) -> None:
    cfg = cfgs[kind]
    data, stats = run_sim(cfg, scene(cfg, kind, 12), steps[kind], steps=0, every=5)
    assert data[:8] == b"ALSTRM\x00\x01" == stream.MAGIC
    (json_len,) = struct.unpack_from("<I", data, 8)
    assert json_len % 4 == 0 and (12 + json_len) % 4 == 0
    header, frames = dec.decode(data)
    assert list(header) == list(dec.HEADER_KEYS)
    d = dims(cfg)
    assert header == {
        "protocol": 1,
        "dims": int(kind[0]),
        "L": 48.0,
        "height": 24.0 if kind == "3d" else 0.0,
        "N": 64,
        "B": 4,
        "radius": cfg.physics.radius,
        "field_cell": cfg.world.field_cell,
        "FG": d.fg,
        "FZ": d.fz,
        "NCH": 3,
        "color_modes": ["tag", "energy", "clade"],
        "color": "tag",
        "fields": False,
        "field_z": 0,
        "step": 12,
        "run": None,
        "config_hash": config_hash(cfg),
        "mode": "simulate",
        "every": 5,
        "source": "preset",
    }
    assert header["color_modes"] == list(render.COLOR_MODES)
    assert (header["FG"], header["FZ"]) == ((12, 6) if kind == "3d" else (12, 1))
    assert len(frames) == 1 and stats.ended == "steps" and stats.steps == 0
    assert data[12 + json_len : 12 + json_len + 4] == b"FRM1"


@pytest.mark.parametrize(("n_steps", "every"), [(0, 1), (6, 1), (6, 2), (7, 3), (5, 10)])
def test_frame_count_and_steps(
    n_steps: int, every: int, cfgs: dict[str, Config], steps: dict[str, Any]
) -> None:
    """A bound of S steps with `--every K` writes S // K + 1 frames, stamped with the absolute steps."""
    cfg = cfgs["2d"]
    data, stats = run_sim(cfg, scene(cfg, "2d", 40), steps["2d"], steps=n_steps, every=every)
    header, frames = dec.decode(data)
    assert len(frames) == n_steps // every + 1 == stats.frames
    assert [f.step for f in frames] == [40 + k * every for k in range(len(frames))]
    assert header["step"] == 40 and header["every"] == every
    assert stats.steps == n_steps and stats.ended == "steps"


@pytest.mark.parametrize("kind", KINDS)
def test_frames_equal_the_stepped_state(kind: str, cfgs: dict[str, Config], steps: dict[str, Any]) -> None:
    """Alive mask, positions, yaw, pitch, bonds and colours of every frame are those of the stepped state;
    a 2D stream has z = 0 and pitch = 0 on every row, a 3D one the state's z and pitch."""
    cfg, step_fn = cfgs[kind], steps[kind]
    cur = scene(cfg, kind)
    refs = [snapshot(cur)]
    for _ in range(6):
        cur = advance(step_fn, cur, 1)
        refs.append(snapshot(cur))
    # The scene is not static: the agent in row 10 dies in the first step and takes its bond with it.
    assert [int(r["alive"].sum()) for r in refs] == [7, *[6] * 6]
    assert [len(ref_bonds(r)) for r in refs] == [3, *[2] * 6]
    data, _ = run_sim(cfg, scene(cfg, kind), step_fn, steps=6)
    header, frames = dec.decode(data)
    assert len(frames) == 7
    for fr, h in zip(frames, refs, strict=True):
        check_frame(fr, h, cfg)
    if kind == "3d":
        assert frames[0].pos[frames[0].alive, 2].min() > 0 and frames[0].orient[:, 1].any()
        assert (frames[0].pos[frames[0].alive, 2] != frames[3].pos[frames[0].alive, 2]).any()
    else:
        assert not np.hstack([f.pos[:, 2] for f in frames]).any() and header["dims"] == 2


@pytest.mark.parametrize("kind", KINDS)
def test_alive_section_is_padded_to_four_bytes(kind: str) -> None:
    """With N not a multiple of 4 the alive mask is followed by zero bytes up to the next 4-byte boundary, and
    every later section stays aligned (the decoder checks the padding, the section sizes and the total)."""
    preset = "tiny3d" if kind == "3d" else "tiny"
    for n_max in (66, 67, 69):
        cfg = load_config(preset=preset, overrides=[*MICRO3D, f"population.n_max={n_max}"])
        data, _ = run_sim(cfg, init_state(cfg), lambda s: pytest.fail("no step"), steps=0)
        header, (fr,) = dec.decode(data)
        assert header["N"] == n_max and len(fr.sections["alive"]) == (n_max + 3) // 4 * 4 > n_max
        assert fr.length == 32 + len(fr.sections["alive"]) + 20 * n_max + 8 * fr.n_bonds
        assert fr.alive.sum() == fr.n_alive == cfg.population.n_init


def test_a_bond_to_a_dead_row_is_not_streamed(cfgs: dict[str, Config], steps: dict[str, Any]) -> None:
    """A bond needs both ends alive (a consistent state has that; the frame does not rely on it)."""
    cfg = cfgs["2d"]
    s = scene(cfg, "2d")
    alive = to_numpy(s.alive)
    alive[1] = False  # row 1 dies but keeps its partner slots: (0,1) must not be drawn, (2,3) still is
    s = s._replace(alive=mx.array(alive))
    (fr,) = dec.decode(run_sim(cfg, s, steps["2d"], steps=0)[0])[1]
    assert fr.bonds.tolist() == [[2, 3], [4, 10]] and fr.n_bonds == 2


@pytest.mark.parametrize("kind", KINDS)
def test_colour_modes_and_fields(kind: str, cfgs: dict[str, Config], steps: dict[str, Any]) -> None:
    """`--color` and `--fields/--field-z` set the first frame; the palette is the renderer's."""
    cfg = cfgs[kind]
    s = scene(cfg, kind)
    h = snapshot(s)
    for color in ("tag", "clade"):
        data, _ = run_sim(cfg, scene(cfg, kind), steps[kind], steps=0, view=stream.View(color=color))
        header, (fr,) = dec.decode(data)
        assert header["color"] == color and fr.color_index == render.COLOR_MODES.index(color)
        np.testing.assert_array_equal(fr.rgb, ref_rgb(h, cfg, color))
    data, _ = run_sim(cfg, scene(cfg, kind), steps[kind], steps=0, view=stream.View(color="energy"))
    (fr,) = dec.decode(data)[1]
    e = h["energy"]
    lo, hi = int(np.argmin(np.where(h["alive"], e, 1 << 30))), int(np.argmax(np.where(h["alive"], e, -1)))
    assert fr.color_index == 1 and fr.rgb[lo] != fr.rgb[hi]
    assert fr.field is None and fr.field_layer is None and not fr.flags & 1


def test_field_slices_equal_the_renderer_background(cfgs: dict[str, Config], steps: dict[str, Any]) -> None:
    """2D: the field section is the colour of every cell of the lattice, `render_frame`'s background; 3D: the
    colours of one z layer, selected by `field_z` and named by the frame's flags."""
    cfg = cfgs["2d"]
    res = (np.arange(144).reshape(12, 12) * 60).astype(np.int32)
    chem = (np.arange(3 * 144).reshape(3, 12, 12) % 7 * 0.5).astype(np.float32)
    s = make_scene(cfg, [], resource=res, chem=chem)
    data, _ = run_sim(cfg, s, steps["2d"], steps=0, view=stream.View(fields=True))
    (fr,) = dec.decode(data)[1]
    assert fr.flags & 1 and fr.field_layer == 0 and fr.field is not None and fr.field.shape == (144,)
    # One pixel per field cell: the frame is the (12, 12) background of resource and chem.
    img = to_numpy(render.render_frame(s, cfg, (12, 12), layers=frozenset({"resource", "chem"})))
    want = (img[..., 0].astype(np.uint32) << 16) | (img[..., 1].astype(np.uint32) << 8) | img[..., 2]
    np.testing.assert_array_equal(fr.field.reshape(12, 12), want)

    cfg3 = cfgs["3d"]
    res3 = (np.arange(6 * 144).reshape(6, 12, 12) * 13 % 9000).astype(np.int32)
    chem3 = (np.arange(3 * 6 * 144).reshape(3, 6, 12, 12) % 7 * 0.5).astype(np.float32)
    s3 = make_scene(cfg3, [], resource=res3, chem=chem3)
    layers = {}
    for z in (0, 2, 5):
        view = stream.View(fields=True, field_z=z)
        data, _ = run_sim(cfg3, s3, steps["3d"], steps=0, view=view)
        (fr3,) = dec.decode(data)[1]
        assert fr3.field_layer == z and fr3.flags >> 16 == z
        layers[z] = fr3.field
    # Reference per layer, float32 as the renderer: g = 30 + 150·min(r/cap, 1); chem tints.
    f32, cap = np.float32, np.float32(cfg3.fields.resource_cap)
    tint = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.6, 0.0, 1.0))
    for z, got in layers.items():
        g = f32(30) + f32(150) * np.minimum(res3[z].reshape(-1).astype(f32) / cap, f32(1))
        rgb = g[:, None] * np.array([0, 1, 0], f32)
        c = np.maximum(chem3[:, z].reshape(3, -1), f32(0))
        for ch in range(3):
            rgb = rgb + (f32(90) * c[ch] / (c[ch] + f32(1)))[:, None] * np.array(tint[ch], f32)
        q = np.clip(np.floor(rgb + f32(0.5)), 0, 255).astype(np.uint32)
        np.testing.assert_array_equal(got, (q[:, 0] << 16) | (q[:, 1] << 8) | q[:, 2])
    assert len({tuple(v) for v in layers.values()}) == 3  # the layers differ


def test_colour_helpers_validate(cfgs: dict[str, Config]) -> None:
    s2, s3 = scene(cfgs["2d"], "2d"), scene(cfgs["3d"], "3d")
    for s, cfg, bad in ((s2, cfgs["2d"], 1), (s3, cfgs["3d"], 6), (s3, cfgs["3d"], -1)):
        with pytest.raises(ValueError, match="layer"):
            render.field_colors(s, cfg, bad)
    with pytest.raises(ValueError, match="color mode"):
        render.agent_colors(s2, cfgs["2d"], "kin")
    for color in render.COLOR_MODES:  # both dims, every mode: one packed colour per row, no priority byte
        for s, cfg in ((s2, cfgs["2d"]), (s3, cfgs["3d"])):
            c = to_numpy(render.agent_colors(s, cfg, color))
            assert c.dtype == np.uint32 and c.shape == (64,) and not (c >> 24).any()
    assert render.COLOR_MODES == ("tag", "energy", "clade")  # no "kin" mode (docs/3D.md §12 Q16)


def test_energy_colours_follow_the_ramp(cfgs: dict[str, Config]) -> None:
    cfg = cfgs["2d"]
    emax = cfg.energy.e_max
    s = make_scene(cfg, [{"energy": 0}, {"energy": emax}, {"energy": 1000, "last_loss": 3}])
    c = to_numpy(render.agent_colors(s, cfg, "energy"))
    assert [int(v) for v in c[:3]] == [(30 << 16) | (60 << 8) | 200, (255 << 16) | (220 << 8) | 40, 0xFF0000]


def test_unknown_record_tags_are_skippable(cfgs: dict[str, Config], steps: dict[str, Any]) -> None:
    """The reference decoder skips a record it does not know by its length (the protocol's extension rule)."""
    cfg = cfgs["2d"]
    data, _ = run_sim(cfg, scene(cfg, "2d"), steps["2d"], steps=1)
    header, frames = dec.decode(data)
    _, first = dec.decode_header(data)
    patched = data[:first] + b"MSG1" + struct.pack("<I", 8) + b"abcdefgh" + data[first:]
    assert [f.step for f in dec.decode(patched)[1]] == [f.step for f in frames] and header


# --- purity and the director -----------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_streaming_does_not_change_the_trajectory(
    kind: str, cfgs: dict[str, Config], steps: dict[str, Any]
) -> None:
    """The state a stream ends in is the bare stepped state, with or without frames, colours, fields and
    pauses between the steps (`view`'s purity test for the stream)."""
    cfg, step_fn = cfgs[kind], steps[kind]
    bare = state_digest(advance(step_fn, scene(cfg, kind), 9))
    _, quiet = run_sim(cfg, scene(cfg, kind), step_fn, steps=9, every=1000)
    assert quiet.frames == 1 and state_digest(quiet.state) == bare
    _, busy = run_sim(cfg, scene(cfg, kind), step_fn, steps=9, every=2)
    assert state_digest(busy.state) == bare
    script = [
        [],
        ["color energy"],
        ["fields on", "pause"],
        ["step 3"],
        [],
        [],
        ["run"],
        ["fields z 0"],
        ["every 1"],
    ]
    data, ctl = run_sim(cfg, scene(cfg, kind), step_fn, steps=9, control=ScriptControl([*script, *[[]] * 4]))
    assert ctl.steps == 9 and ctl.ended == "steps" and state_digest(ctl.state) == bare
    assert len(dec.decode(data)[1]) > 3


def test_stream_help_says_it_ignores_the_director() -> None:
    """`stream` re-simulates with the checkpoint's environment levers and applies no director log (like
    `view`), so its help says a stream of a director run diverges, and `--follow` is the faithful view."""
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction")
    helps = {c.dest: " ".join(c.help.split()) for c in sub._choices_actions}
    text = " ".join(sub.choices["stream"].format_help().split())
    for flat in (helps["stream"], text):
        assert "without the director" in flat and "--follow" in flat
        assert "faithful view of a director run" in flat
        assert "never writes into the run directory" in flat and "stdin EOF = quit" in flat


# --- the control channel --------------------------------------------------------------------------------


def test_control_script(cfgs: dict[str, Config], steps: dict[str, Any]) -> None:
    """Every command, in a fixed schedule: frames at the expected steps, colours and field layers."""
    cfg = cfgs["3d"]
    script = [
        [],  # step 1
        [],  # step 2: frame (every 2)
        ["color energy"],  # a redraw of step 2, then step 3
        ["every 1"],  # step 4: frame
        ["pause"],  # idle
        ["step 3"],  # step 5
        [],  # step 6
        [],  # step 7: the one frame of the batch
        ["fields on", "fields z 2"],  # idle: one redraw
        ["run"],  # step 8: frame
        ["bogus", "", "# a comment"],  # step 9: frame; the comment and the blank line are silent
        ["quit"],
    ]
    ctl = ScriptControl(script)
    data, stats = run_sim(cfg, scene(cfg, "3d"), steps["3d"], every=2, control=ctl)
    header, frames = dec.decode(data)
    got = [(f.step, f.color_index, f.field_layer) for f in frames]
    assert got == [
        (0, 0, None),
        (2, 0, None),
        (2, 1, None),
        (4, 1, None),
        (7, 1, None),
        (7, 1, 2),
        (8, 1, 2),
        (9, 1, 2),
    ]
    assert stats.ended == "quit" and stats.steps == 9 and ctl.closed
    # A running stream only peeks at the channel; a paused one with nothing to do blocks on it (no spinning).
    assert ctl.timeouts[:5] == [0] * 5 and ctl.timeouts[5] is None
    assert [k for k, t in enumerate(ctl.timeouts) if t is None] == [5, 8, 9]
    assert header["every"] == 2 and header["fields"] is False  # the header is the starting view


def test_control_eof_ends_the_stream(cfgs: dict[str, Config], steps: dict[str, Any]) -> None:
    cfg = cfgs["2d"]
    data, stats = run_sim(cfg, scene(cfg, "2d"), steps["2d"], control=ScriptControl([[], [], []]))
    assert stats.ended == "quit" and stats.steps == 3 and len(dec.decode(data)[1]) == 4


def test_bound_wins_over_a_paused_stream(cfgs: dict[str, Config], steps: dict[str, Any]) -> None:
    """`--steps` ends the stream when reached even while a `step` batch is pending; no frame for the batch."""
    cfg = cfgs["2d"]
    ctl = ScriptControl([["step 10"], [], []])
    data, stats = run_sim(cfg, scene(cfg, "2d"), steps["2d"], steps=2, control=ctl)
    assert stats.ended == "steps" and stats.steps == 2
    assert [f.step for f in dec.decode(data)[1]] == [0]


def test_pause_cancels_a_pending_step_batch(cfgs: dict[str, Config], steps: dict[str, Any]) -> None:
    cfg = cfgs["2d"]
    script = [["step 5"], ["pause"], ["step 1"], [], ["run"], []]
    data, stats = run_sim(cfg, scene(cfg, "2d"), steps["2d"], steps=4, control=ScriptControl(script))
    # `step 5` runs one step, `pause` drops the other four, `step 1` runs one more and writes its frame,
    # `run` resumes with every = 1 (the bound ends it at step 4).
    assert [f.step for f in dec.decode(data)[1]] == [0, 2, 3, 4] and stats.steps == 4


def test_a_colour_and_a_layer_in_one_batch_write_one_frame(
    cfgs: dict[str, Config], steps: dict[str, Any]
) -> None:
    """`fields z K` with fields off shows nothing new, but must not drop the redraw `color` asked for."""
    cfg = cfgs["3d"]
    script = [["color clade", "fields z 3"], ["fields z 4"], ["fields on"], []]
    data, _ = run_sim(cfg, scene(cfg, "3d"), steps["3d"], steps=0, control=ScriptControl(script))
    got = [(f.step, f.color_index, f.field_layer) for f in dec.decode(data)[1]]
    assert got == [(0, 0, None), (0, 2, None)]  # the script is cut by `steps=0` after its first poll
    data, _ = run_sim(cfg, scene(cfg, "3d"), steps["3d"], steps=3, control=ScriptControl(script))
    got = [(f.step, f.color_index, f.field_layer) for f in dec.decode(data)[1]]
    # `fields z 4` with fields still off writes nothing; `fields on` then redraws with layer 4 (set by the
    # line before), and the grid frames follow.
    assert got == [(0, 0, None), (0, 2, None), (1, 2, None), (2, 2, None), (2, 2, 4), (3, 2, 4)]


def test_invalid_control_lines_are_ignored(
    cfgs: dict[str, Config], steps: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    bad = [
        "bogus",
        "step x",
        "step 0",
        "step 1 2",
        "every 0",
        "every",
        "color neon",
        "color",
        "fields",
        "fields maybe",
        "fields z",
        "fields z 99",
        "fields z -1",
        "pause now",
        "quit please",
        "Pause",
    ]
    cfg = cfgs["2d"]
    data, stats = run_sim(cfg, scene(cfg, "2d"), steps["2d"], steps=2, control=ScriptControl([bad, [], []]))
    err = capsys.readouterr().err
    assert err.count("ignored control line") == len(bad)
    assert stats.steps == 2 and [f.step for f in dec.decode(data)[1]] == [0, 1, 2]  # nothing took effect
    assert "fields z 99" in err and "outside [0, 1)" in err  # 2D has one layer


def test_parse_command_grammar() -> None:
    P = stream.parse_command
    assert P("") is None and P("   ") is None and P("# note") is None
    assert P("pause") == stream.Command("pause") and P("  run  ") == stream.Command("run")
    assert P("step") == stream.Command("step", 1) and P("step 12") == stream.Command("step", 12)
    assert P("every 5") == stream.Command("every", 5)
    assert P("color clade") == stream.Command("color", "clade")
    assert P("fields on") == stream.Command("fields", sub="on") and P("fields off").sub == "off"
    assert P("fields z 3", fz=6) == stream.Command("fields", 3, sub="z")
    assert P("quit") == stream.Command("quit")
    for line in ("fields z 6", "fields z ٣", "step ²", "step +1", "color Tag", "fields on off", "run 1"):
        with pytest.raises(stream.ControlError):
            P(line, fz=6)


def test_fd_control_reads_lines_and_eof() -> None:
    r, w = os.pipe()
    ctl = stream.FdControl(r)
    try:
        assert ctl.poll(0) == []
        os.write(w, b"pause\nstep 2")
        assert ctl.poll(5) == ["pause"]  # the partial line waits for its newline
        os.write(w, b"\ncolor energy\n\nfields on\n")
        got: list[str] = []
        while len(got) < 4:
            got += ctl.poll(5)
        assert got == ["step 2", "color energy", "", "fields on"]
        os.write(w, b"x" * (stream.MAX_LINE + 10) + b"\nafter\n")  # an overlong line is dropped
        got = []
        while "after" not in got:
            got += ctl.poll(5)
        assert got == ["after"]
        os.write(w, b"last")  # EOF flushes an unterminated line, then reads as quit
        os.close(w)
        w = -1
        got = []
        while "quit" not in got:
            got += ctl.poll(5)
        assert got == ["last", "quit"]
    finally:
        if w >= 0:
            os.close(w)
        os.close(r)


def test_no_control_never_reports_lines() -> None:
    ctl = stream.NoControl()
    t0 = time.perf_counter()
    assert ctl.poll(0) == [] and ctl.poll(0.05) == [] and time.perf_counter() - t0 >= 0.04
    ctl.close()


# --- the pipe and stdout --------------------------------------------------------------------------------


class _BrokenSink(io.BytesIO):
    """A reader that goes away after `limit` writes."""

    def __init__(self, limit: int, exc: type[OSError] = BrokenPipeError) -> None:
        super().__init__()
        self.limit, self.exc, self.writes = limit, exc, 0

    def write(self, b: Any) -> int:
        self.writes += 1
        if self.writes > self.limit:
            raise self.exc(32, "Broken pipe")
        return super().write(b)


@pytest.mark.parametrize("exc", [BrokenPipeError, ConnectionResetError])
@pytest.mark.parametrize("limit", [0, 1, 2, 4])
def test_closed_reader_is_a_quiet_stop(
    limit: int,
    exc: type[OSError],
    cfgs: dict[str, Config],
    steps: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = cfgs["2d"]
    ctl = ScriptControl([[]] * 50)
    sink = _BrokenSink(limit, exc)
    stats = stream.serve(
        cfg, mode="simulate", source="preset", run=None, state=[scene(cfg, "2d")], out=sink, control=ctl,
        step_fn=steps["2d"],
    )  # fmt: skip
    assert stats.ended == "pipe" and ctl.closed
    assert capsys.readouterr().err == ""  # no traceback, no message


def test_protocol_output_redirects_stdout_to_stderr(capfdbinary: pytest.CaptureFixture[bytes]) -> None:
    with stream.protocol_output() as out:
        print("a stray print")
        os.write(1, b"a raw write to fd 1\n")  # a C library's printf, a child process
        sys.stdout.write("via sys.stdout\n")
        sys.stdout.flush()
        out.write(b"PROTOCOL")
    print("after")
    sys.stdout.flush()
    os.write(1, b"restored\n")
    got = capfdbinary.readouterr()
    assert got.out == b"PROTOCOL" + b"after\n" + b"restored\n"
    for text in (b"a stray print", b"a raw write to fd 1", b"via sys.stdout"):
        assert text in got.err and text not in got.out


def test_stdout_holds_only_protocol_bytes(
    cfgs: dict[str, Config], steps: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capfdbinary: Any
) -> None:
    """Even a step that prints (and writes to fd 1 itself) cannot corrupt the stream: the CLI's stdout decodes
    strictly, and the noise is on stderr."""
    real = steps["3d"]

    def noisy(state: Any) -> Any:
        print("noise from a step")
        os.write(1, b"raw noise from a step\n")
        return real(state)

    monkeypatch.setattr(stream, "make_step", lambda cfg: noisy)
    assert cli.main(["stream", *micro_args("3d"), "--steps", "3", "--every", "1", "--no-control"]) == 0
    got = capfdbinary.readouterr()
    header, frames = dec.decode(got.out)  # strict: it fails on any stray byte
    assert [f.step for f in frames] == [0, 1, 2, 3] and header["dims"] == 3
    assert got.err.count(b"noise from a step") == 6 and b"raw noise from a step" in got.err


# --- sources and arguments (through cli.main) -------------------------------------------------------------


@pytest.fixture
def shared_step(monkeypatch: pytest.MonkeyPatch, steps: dict[str, Any]) -> None:
    """`cli.main` builds its own step; use the module's compiled ones (the configs are equal)."""
    monkeypatch.setattr(stream, "make_step", lambda cfg: steps["3d" if dims(cfg).d3 else "2d"])


def _tree(path: Path) -> dict[str, tuple[int, int, bytes]]:
    out = {}
    for p in sorted(path.rglob("*")):
        rel = str(p.relative_to(path))
        out[rel] = (p.stat().st_mtime_ns, p.stat().st_size, p.read_bytes() if p.is_file() else b"")
    return out


def make_run(tmp_path: Path, kind: str, cfg: Config, checkpoint_steps: Sequence[int]) -> RunDir:
    """A run directory with a (built, not stepped) checkpoint at each of `checkpoint_steps`."""
    rd = RunDir.create(tmp_path / "run", cfg)
    for k in checkpoint_steps:
        rd.save_checkpoint(scene(cfg, kind, k), cfg)
    return rd


def freeze(path: Path) -> list[Path]:
    """Make a directory tree read-only (a stream must not need to write there)."""
    dirs = [path, *(p for p in path.rglob("*") if p.is_dir())]
    for d in dirs:
        d.chmod(0o555)
    return dirs


def thaw(dirs: list[Path]) -> None:
    for d in dirs:
        d.chmod(0o755)


@pytest.mark.parametrize("kind", KINDS)
def test_run_source_writes_nothing(
    kind: str, cfgs: dict[str, Config], tmp_path: Path, shared_step: None, capfdbinary: Any
) -> None:
    cfg = cfgs[kind]
    rd = make_run(tmp_path, kind, cfg, [0, 3])
    before = _tree(rd.path)
    frozen = freeze(rd.path)
    try:
        code = cli.main(["stream", str(rd.path), "--steps", "4", "--every", "2", "--no-control"])
    finally:
        thaw(frozen)
    assert code == 0 and _tree(rd.path) == before and not rd.media_dir.exists()
    header, frames, _ = stdout_frames(capfdbinary)
    assert header["source"] == "run" and header["run"] == str(rd.path) and header["step"] == 3  # LATEST
    assert [f.step for f in frames] == [3, 5, 7]
    check_frame(frames[0], snapshot(scene(cfg, kind, 3)), cfg)


def test_run_source_checkpoint_file_and_preset(
    cfgs: dict[str, Config], tmp_path: Path, shared_step: None, capfdbinary: Any
) -> None:
    cfg = cfgs["3d"]
    rd = make_run(tmp_path, "3d", cfg, [0, 3])
    assert cli.main(["stream", str(rd.path), "--checkpoint", "0", "--steps", "0", "--no-control"]) == 0
    header, frames, _ = stdout_frames(capfdbinary)
    assert header["step"] == 0 and header["source"] == "run" and len(frames) == 1
    # A standalone state file (a viewer snapshot): the config comes from the file.
    snap = tmp_path / "snap.safetensors"
    write_state_file(snap, scene(cfg, "3d", 9), cfg)
    assert cli.main(["stream", str(snap), "--steps", "2", "--color", "clade", "--no-control"]) == 0
    header, frames, _ = stdout_frames(capfdbinary)
    assert (header["source"], header["run"], header["step"], header["color"]) == (
        "file",
        str(snap),
        9,
        "clade",
    )
    assert [f.step for f in frames] == [9, 10, 11]
    assert all(f.color_index == 2 for f in frames)
    # A preset: founders from init_state, no files at all.
    assert (
        cli.main(["stream", *micro_args("3d"), "--steps", "2", "--fields", "--field-z", "4", "--no-control"])
        == 0
    )
    header, frames, _ = stdout_frames(capfdbinary)
    assert (header["source"], header["run"], header["step"], header["fields"], header["field_z"]) == (
        "preset",
        None,
        0,
        True,
        4,
    )
    assert header["config_hash"] == config_hash(cfg) and [f.step for f in frames] == [0, 1, 2]
    assert all(f.field_layer == 4 for f in frames)
    check_frame(frames[0], snapshot(init_state(cfg)), cfg)


def test_2d_preset_stream(cfgs: dict[str, Config], shared_step: None, capfdbinary: Any) -> None:
    cfg = cfgs["2d"]
    assert cli.main(["stream", *micro_args("2d"), "--steps", "4", "--every", "2", "--no-control"]) == 0
    header, frames, _ = stdout_frames(capfdbinary)
    assert header["dims"] == 2 and header["FZ"] == 1 and header["height"] == 0.0
    assert [f.step for f in frames] == [0, 2, 4]
    check_frame(frames[0], snapshot(init_state(cfg)), cfg)
    assert not any(f.pos[:, 2].any() or f.orient[:, 1].any() for f in frames)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        ([], "needs a source"),
        (["--set", "run.seed=1"], "needs a source"),
        (["--preset", "tiny", "somewhere"], "not both"),
        (["--preset", "tiny", "--checkpoint", "0"], "--checkpoint needs RUN"),
        (["--follow", "--preset", "tiny"], "--follow needs a run directory"),
        (["--follow", "FILE"], "--follow needs a run directory"),
        (["--follow", "RUN", "--steps", "3"], "--steps does not apply"),
        (["--follow", "RUN", "--every", "2"], "--every does not apply"),
        (["--follow", "RUN", "--checkpoint", "0"], "--checkpoint does not apply"),
        (["--preset", "tiny", "--field-z", "1"], "outside [0, 1)"),
        (["--preset", "tiny3d", "--field-z", "6"], "outside [0, 6)"),
    ],
)
def test_bad_arguments_exit_2(
    argv: list[str], message: str, tmp_path: Path, capsys: pytest.CaptureFixture[str], cfgs: dict[str, Config]
) -> None:
    rd = make_run(tmp_path, "2d", cfgs["2d"], [0])
    snap = tmp_path / "snap.safetensors"
    write_state_file(snap, scene(cfgs["2d"], "2d"), cfgs["2d"])
    argv = [str(rd.path) if a == "RUN" else str(snap) if a == "FILE" else a for a in argv]
    assert cli.main(["stream", *argv, "--no-control"]) == 2
    assert message in capsys.readouterr().err


def test_state_file_refuses_a_checkpoint_flag(tmp_path: Path, cfgs: dict[str, Config], capsys: Any) -> None:
    snap = tmp_path / "snap.safetensors"
    write_state_file(snap, scene(cfgs["2d"], "2d"), cfgs["2d"])
    assert cli.main(["stream", str(snap), "--checkpoint", "0", "--no-control"]) == 2
    assert "applies to a run directory" in capsys.readouterr().err
    assert cli.main(["stream", str(snap), "--every", "0"]) == 2  # argparse: an integer >= 1


def test_run_without_checkpoints_is_a_checkpoint_error(
    tmp_path: Path, cfgs: dict[str, Config], capsys: pytest.CaptureFixture[str]
) -> None:
    rd = make_run(tmp_path, "2d", cfgs["2d"], [])
    assert cli.main(["stream", str(rd.path), "--no-control"]) == 4
    assert cli.main(["stream", str(rd.path), "--follow", "--no-control"]) == 4
    assert "no checkpoint" in capsys.readouterr().err


def test_cli_control_over_a_pipe(
    cfgs: dict[str, Config], shared_step: None, monkeypatch: pytest.MonkeyPatch, capfdbinary: Any
) -> None:
    """Without `--no-control` the CLI reads `CONTROL_FD`; EOF there ends the stream."""
    r, w = os.pipe()
    os.write(w, b"color energy\nfields on\n")
    os.close(w)  # the lines are queued, then EOF: quit
    monkeypatch.setattr(stream, "CONTROL_FD", r)
    try:
        assert cli.main(["stream", *micro_args("2d")]) == 0
    finally:
        os.close(r)
    header, frames, _ = stdout_frames(capfdbinary)
    assert header["color"] == "tag" and len(frames) >= 1  # the first frame precedes the commands


# --- follow ---------------------------------------------------------------------------------------------


def follow(rd: RunDir, cfg: Config, script: Sequence[Any], **kw: Any) -> tuple[bytes, stream.StreamStats]:
    buf = io.BytesIO()
    stats = stream.serve(
        cfg,
        mode="follow",
        source="run",
        run=str(rd.path),
        rd=rd,
        out=buf,
        control=ScriptControl(script),
        poll_s=0.01,
        **kw,
    )
    return buf.getvalue(), stats


def test_follow_streams_each_new_checkpoint(cfgs: dict[str, Config], tmp_path: Path) -> None:
    """The latest checkpoint first, then a frame for each newer one; `pause` holds, `step` fetches the newest
    once, `color` and `fields` redraw the current checkpoint, `every` is ignored; nothing is stepped."""
    cfg = cfgs["3d"]
    rd = make_run(tmp_path, "3d", cfg, [0, 3])

    def save(k: int) -> Callable[[], list[str]]:
        def go() -> list[str]:
            rd.save_checkpoint(scene(cfg, "3d", k), cfg)
            return []

        return go

    script = [
        [],  # nothing new: no frame
        save(6),  # a new checkpoint: a frame
        ["pause"],
        save(9),  # held back while paused
        [],
        ["every 5", "color clade"],  # `every` ignored; a redraw of step 6
        ["step"],  # the newest checkpoint, once
        ["fields on", "fields z 1"],  # one redraw
        ["run"],
        [],
    ]
    data, stats = follow(rd, cfg, script)
    header, frames = dec.decode(data)
    assert (header["mode"], header["every"], header["step"], header["source"]) == ("follow", 0, 3, "run")
    assert [(f.step, f.color_index, f.field_layer) for f in frames] == [
        (3, 0, None),
        (6, 0, None),
        (6, 2, None),
        (9, 2, None),
        (9, 2, 1),
    ]
    assert stats.ended == "quit" and stats.steps == 0
    for fr in frames[:2]:
        check_frame(fr, snapshot(scene(cfg, "3d", fr.step)), cfg)


def test_follow_writes_nothing_and_never_steps(
    cfgs: dict[str, Config], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = cfgs["2d"]
    rd = make_run(tmp_path, "2d", cfg, [0, 3])
    monkeypatch.setattr(stream, "make_step", lambda c: pytest.fail("follow must not build a step"))
    before = _tree(rd.path)
    frozen = freeze(rd.path)
    try:
        data, stats = follow(rd, cfg, [[], [], ["color energy"], []])
    finally:
        thaw(frozen)
    assert _tree(rd.path) == before
    assert [f.step for f in dec.decode(data)[1]] == [3, 3] and stats.ended == "quit"


def test_follow_survives_a_pruned_checkpoint(
    cfgs: dict[str, Config], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint that vanishes between listing and reading is skipped; the next poll gets the new one."""
    cfg = cfgs["2d"]
    rd = make_run(tmp_path, "2d", cfg, [0, 2])
    real = stream.load_checkpoint
    calls: list[Path] = []

    def flaky(path: Path, c: Config) -> Any:
        calls.append(path)
        if len(calls) == 2:
            raise FileNotFoundError(path)
        return real(path, c)

    def new_checkpoint() -> list[str]:
        rd.save_checkpoint(scene(cfg, "2d", 5), cfg)
        return []

    monkeypatch.setattr(stream, "load_checkpoint", flaky)
    data, _ = follow(rd, cfg, [new_checkpoint, [], []])
    assert [f.step for f in dec.decode(data)[1]] == [2, 5]


# --- subprocesses: real pipes ------------------------------------------------------------------------------

CPU_MAIN = (
    "import sys, mlx.core as mx; mx.set_default_device(mx.cpu); "
    "from alife.cli import main; raise SystemExit(main(sys.argv[1:]))"
)


def _spawn(*args: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-c", CPU_MAIN, "stream", *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _read_exact(f: Any, n: int) -> bytes:
    data = f.read(n)
    assert len(data) == n, "the stream ended early"
    return data


def read_record(f: Any) -> tuple[bytes, bytes]:
    tag = _read_exact(f, 4)
    (n,) = struct.unpack("<I", _read_exact(f, 4))
    return tag, _read_exact(f, n)


def read_header(f: Any) -> bytes:
    magic = _read_exact(f, 12)
    (n,) = struct.unpack("<I", magic[8:])
    return magic + _read_exact(f, n)


def test_reader_closing_early_exits_zero_without_a_traceback() -> None:
    proc = _spawn(*micro_args("2d"), "--every", "1")
    try:
        assert proc.stdout is not None
        read_header(proc.stdout)
        assert read_record(proc.stdout)[0] == b"FRM1"
        proc.stdout.close()  # the viewer window closed
        code = proc.wait(timeout=120)
        err = proc.stderr.read().decode() if proc.stderr else ""
    finally:
        proc.kill()
        for f in (proc.stdin, proc.stderr):
            if f:
                f.close()
    assert code == 0, err
    assert "Traceback" not in err and "BrokenPipe" not in err and "Exception ignored" not in err


def test_stdin_commands_and_eof_over_real_pipes() -> None:
    """Commands written to stdin take effect between steps; closing stdin ends the stream with exit 0, and the
    whole of stdout is protocol."""
    proc = _spawn(*micro_args("2d"), "--every", "1000000")
    out = io.BytesIO()
    try:
        assert proc.stdin is not None and proc.stdout is not None
        out.write(read_header(proc.stdout))
        tag, payload = read_record(proc.stdout)  # the first frame, before any step
        out.write(tag + struct.pack("<I", len(payload)) + payload)
        proc.stdin.write(b"pause\ncolor clade\n")
        proc.stdin.flush()
        tag, payload = read_record(proc.stdout)  # the redraw of the paused state
        out.write(tag + struct.pack("<I", len(payload)) + payload)
        proc.stdin.write(b"step 2\n")
        proc.stdin.flush()
        tag, payload = read_record(proc.stdout)
        out.write(tag + struct.pack("<I", len(payload)) + payload)
        proc.stdin.close()  # EOF = quit
        rest, err = proc.stdout.read(), proc.stderr.read() if proc.stderr else b""
        out.write(rest)
        code = proc.wait(timeout=120)
    finally:
        proc.kill()
    assert code == 0, err
    header, frames = dec.decode(out.getvalue())  # strict: nothing but records
    assert [f.color_index for f in frames] == [0, 2, 2] and rest == b""
    assert frames[2].step == frames[1].step + 2 and frames[0].step == 0 == header["step"]


def test_follow_over_real_pipes(cfgs: dict[str, Config], tmp_path: Path) -> None:
    cfg = cfgs["2d"]
    rd = make_run(tmp_path, "2d", cfg, [0, 3])
    proc = _spawn(str(rd.path), "--follow", "--poll", "0.05")
    try:
        assert proc.stdin is not None and proc.stdout is not None
        header = dec.decode_header(read_header(proc.stdout))[0]
        tag, payload = read_record(proc.stdout)
        assert header["mode"] == "follow" and tag == b"FRM1" and struct.unpack_from("<Q", payload)[0] == 3
        rd.save_checkpoint(scene(cfg, "2d", 8), cfg)  # a runner writing a new checkpoint
        tag, payload = read_record(proc.stdout)
        assert struct.unpack_from("<Q", payload)[0] == 8
        proc.stdin.close()
        assert proc.stdout.read() == b"" and proc.wait(timeout=60) == 0
    finally:
        proc.kill()


# --- the committed fixtures ------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_committed_fixtures_decode_as_their_sidecars_say(kind: str) -> None:
    """On any device: the files PR 5 reads are valid streams and agree with their JSON sidecars."""
    bin_path, json_path = fx.paths(kind)
    data = bin_path.read_bytes()
    side = json.loads(json_path.read_text(encoding="utf-8"))
    assert side == fx.sidecar(kind, side["config"], data)  # the sidecar is the decoding of the bytes
    header, frames = dec.decode(data)
    assert side["bytes"] == len(data) and side["header"] == header
    assert header["dims"] == int(kind[0]) and header["N"] == 64 and len(frames) == 3
    assert [f["step"] for f in side["frames"]] == [100, 104, 108]
    assert [f["field_layer"] for f in side["frames"]] == [None, 0, 5 if kind == "3d" else 0]
    assert [f["color"] for f in side["frames"]] == ["tag", "energy", "clade"]
    assert side["frames"][2]["diag_error"] and not side["frames"][0]["diag_error"]
    assert side["frames"][0]["bonds"] == [[0, 1], [2, 3]] and side["frames"][1]["n_bonds"] == 3
    z = np.hstack([f.pos[f.alive, 2] for f in frames])
    assert (z > 0).all() if kind == "3d" else not z.any()


@pytest.mark.skipif(ON_GPU, reason="the fixture bytes are pinned on the CPU device (float32 colour ramps)")
@pytest.mark.parametrize("kind", KINDS)
def test_fixtures_match_the_code(kind: str) -> None:
    """The committed fixtures are what the current writer produces. After a deliberate change of the protocol
    or the palette, regenerate them: `uv run python tests/stream_fixtures.py --write` (CPU device)."""
    data, side = fx.build(kind)
    bin_path, json_path = fx.paths(kind)
    assert bin_path.read_bytes() == data, "stream bytes changed: regenerate the fixtures"
    assert json_path.read_text(encoding="utf-8") == fx.sidecar_text(side), "sidecar changed: regenerate"
    assert fx.build(kind)[0] == data  # deterministic


def test_fixture_configs_are_the_test_micro_configs(cfgs: dict[str, Config]) -> None:
    """The fixtures use the `micro` sizes of the test suite (`conftest.micro_cfg`, `micro3d_cfg`)."""
    for kind in KINDS:
        cfg, spec = fx.fixture_config(kind)
        assert cfg == cfgs[kind] and spec["overrides"] == list(MICRO3D)
