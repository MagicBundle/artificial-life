"""§27.1 dims in run directories and tools: metadata (absent = 2D), cross-dimension refusals, the one State
schema and the 2D-only tools."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from alife import cli, config, runio, state, viewer
from alife.analysis import signals
from alife.config import PRESETS, Config, ConfigError, config_from_dict, config_to_dict, is_3d, load_config
from alife.render import render_frame
from alife.runio import CheckpointError, RunDir, checkpoint_metadata, load_checkpoint, write_state_file
from alife.state import State, init_state

# A valid 3D variant of tiny or micro once 3D is enabled (height 24, chem_diffusion ≤ 1/6, a benthic food
# band), as CLI flags.
THREE_D = [
    *("--set", "world.dims=3"),
    *("--set", "world.height=24"),
    *("--set", "fields.chem_diffusion=0.16,0.1,0.16"),
    *("--set", "fields.benthic_lambda=8"),
]


def _as_3d(cfg: Config, height: float = 24.0) -> Config:
    """``cfg`` as a (not validated) 3D world: no 3D state exists before docs/3D.md PR 2."""
    d = config_to_dict(cfg)
    return config_from_dict({**d, "world": {**d["world"], "dims": 3, "height": height}})


def _rewrite_metadata(path: Path, **changes: str) -> None:
    arrays, meta = mx.load(str(path), format="safetensors", return_metadata=True)
    mx.eval(arrays)  # loads are lazy: read everything before truncating the file
    meta.update(changes)
    with open(path, "wb") as f:
        mx.save_safetensors(f, arrays, meta)


def _snapshot(root: Path) -> dict[Path, bytes]:
    return {p: p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture
def source_run(micro_cfg: Config, tmp_path: Path) -> Path:
    """A 2D run directory holding only its step-0 checkpoint."""
    rd = RunDir.create(tmp_path / "src", micro_cfg)
    rd.update_meta(start_step=0)
    rd.save_checkpoint(init_state(micro_cfg), micro_cfg)
    return rd.path


# --- metadata ---------------------------------------------------------------------------------------------


def test_2d_meta_has_no_dims(micro_cfg: Config, tmp_path: Path) -> None:
    rd = RunDir.create(tmp_path / "run", micro_cfg)
    meta = rd.meta()
    assert "dims" not in meta and meta["genome_layout_version"] == 1
    path = tmp_path / "s.safetensors"
    write_state_file(path, init_state(micro_cfg), micro_cfg)
    ck = checkpoint_metadata(path)
    assert "dims" not in ck and ck["genome_layout_version"] == "1"


def test_3d_meta_records_dims(micro_cfg: Config, tmp_path: Path) -> None:
    cfg3 = _as_3d(micro_cfg)
    meta = RunDir.create(tmp_path / "run", cfg3).meta()
    assert (meta["dims"], meta["genome_layout_version"]) == (3, 2)
    path = tmp_path / "s.safetensors"
    write_state_file(path, init_state(micro_cfg), cfg3)  # the metadata only; a 2D state stands in
    ck = checkpoint_metadata(path)
    assert (ck["dims"], ck["genome_layout_version"]) == ("3", "2")


def test_cross_dims_checkpoint_refused(micro_cfg: Config, tmp_path: Path) -> None:
    path = tmp_path / "s.safetensors"
    write_state_file(path, init_state(micro_cfg), micro_cfg)
    load_checkpoint(path, micro_cfg)
    cfg3 = _as_3d(micro_cfg)
    for check_hash in (True, False):
        with pytest.raises(CheckpointError, match=r"dims='2' \(expected '3'\): cross-dimension"):
            load_checkpoint(path, cfg3, check_hash=check_hash)
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(path.read_bytes())
    _rewrite_metadata(bad, dims="3")
    for check_hash in (True, False):
        with pytest.raises(CheckpointError, match=r"dims='3' \(expected '2'\): cross-dimension"):
            load_checkpoint(bad, micro_cfg, check_hash=check_hash)


# --- CLI refusals -----------------------------------------------------------------------------------------


def test_fork_set_dims3_exit2_touches_nothing(source_run: Path, tmp_path: Path) -> None:
    before = _snapshot(source_run)
    out = tmp_path / "fork"
    argv = ["run", "--init-from", str(source_run), "--out", str(out), "--quiet", *THREE_D]
    assert cli.main(argv) == 2  # validate: world.dims = 3 is not enabled yet
    assert not out.exists() and _snapshot(source_run) == before


def test_fork_across_dims_refused(
    source_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(config, "DIMS_SUPPORTED", (2, 3))
    before = _snapshot(source_run)
    out = tmp_path / "fork"
    argv = ["run", "--init-from", str(source_run), "--out", str(out), "--quiet", *THREE_D]
    assert cli.main(argv) == 4
    assert "world.dims must equal the source's 2" in capsys.readouterr().err
    assert not out.exists() and _snapshot(source_run) == before


def test_fork_across_heights_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(config, "DIMS_SUPPORTED", (2, 3))
    cfg2 = load_config(preset="tiny")
    cfg3 = load_config(preset="tiny", overrides=THREE_D[1::2])
    src = tmp_path / "src.safetensors"
    write_state_file(src, init_state(cfg2), cfg3)  # 3D metadata; refused before any array is read
    before = src.read_bytes()
    out = tmp_path / "fork"
    argv = ["run", "--init-from", str(src), "--out", str(out), "--quiet", "--set", "world.height=32"]
    assert cli.main(argv) == 4
    assert "world.height must equal the source's 24.0" in capsys.readouterr().err
    assert not out.exists() and src.read_bytes() == before


def test_resume_set_dims3_exit2(source_run: Path) -> None:
    before = _snapshot(source_run)
    for item in ("world.dims=3", "world.height=24"):
        assert cli.main(["resume", str(source_run), "--set", item]) == 2
    assert _snapshot(source_run) == before


def test_render_and_view_refuse_3d_before_loading(
    micro_cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(config, "DIMS_SUPPORTED", (2, 3))
    cfg3 = load_config(preset="tiny", overrides=THREE_D[1::2])
    rd = RunDir.create(tmp_path / "run3", cfg3)
    (rd.checkpoint_dir / "step_0000000000.safetensors").write_bytes(b"")  # never read: refused first
    calls: list[object] = []
    monkeypatch.setattr(viewer, "run_viewer", lambda *a, **k: calls.append(a))
    before = _snapshot(rd.path)
    assert cli.main(["render", str(rd.path)]) == 2
    assert cli.main(["view", str(rd.path)]) == 2
    assert cli.main(["view", "--preset", "tiny", *THREE_D]) == 2
    assert "supports 2D worlds only" in capsys.readouterr().err
    assert calls == [] and _snapshot(rd.path) == before and not rd.media_dir.exists()


def test_2d_only_tools_refuse_3d(micro_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    s, cfg3 = init_state(micro_cfg), _as_3d(micro_cfg)
    with pytest.raises(ConfigError, match="the MLX renderer supports 2D worlds only"):
        render_frame(s, cfg3, (8, 8))
    with pytest.raises(ConfigError, match="the live viewer"):
        viewer.ViewerSession(cfg3, s)
    imports: list[None] = []
    monkeypatch.setattr(viewer, "_import_pygame", lambda: imports.append(None))
    with pytest.raises(ConfigError, match="the live viewer"):
        viewer.run_viewer(cfg3, s)
    assert imports == []  # refused before pygame is imported
    with pytest.raises(ConfigError, match="the signal analysis"):
        signals.decode(s, cfg3)


def test_signal_analysis_refuses_3d_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "DIMS_SUPPORTED", (2, 3))
    rd = RunDir.create(tmp_path / "run3", load_config(preset="tiny", overrides=THREE_D[1::2]))
    built: list[object] = []
    monkeypatch.setattr(signals, "make_step", lambda *a, **k: built.append(a))
    before = _snapshot(rd.path)
    with pytest.raises(ConfigError, match="the signal analysis supports 2D worlds only"):
        signals.analyse(rd.path, [0], states_dir=None, workers=1, log=lambda *_: None)
    assert built == [] and _snapshot(rd.path) == before


def test_require_2d_message(micro_cfg: Config) -> None:
    assert config.require_2d(micro_cfg, "tool") is None and not is_3d(micro_cfg)
    cfg3 = _as_3d(micro_cfg)
    with pytest.raises(ConfigError) as ei:
        config.require_2d(cfg3, "tool")
    msg = str(ei.value)
    assert msg.startswith("tool supports 2D worlds only (world.dims = 3)") and "docs/3D.md" in msg


# --- one schema -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("overlays", [(), ("fast",)])
def test_one_schema(preset: str, overlays: tuple[str, ...]) -> None:
    cfg = load_config(preset=preset, overlays=overlays)
    assert runio._expected_layout is state.expected_layout
    spec = state.expected_layout(cfg)
    assert runio._expected_layout(cfg) == spec
    assert tuple(spec) == State._fields


def test_schema_follows_dims(micro_cfg: Config) -> None:
    cfg3 = _as_3d(micro_cfg)
    d2, d3 = state.expected_layout(micro_cfg), state.expected_layout(cfg3)
    n, fg, f32 = 64, 12, mx.float32
    assert d2["pos"] == d2["vel"] == ((n, 2), f32) and d3["pos"] == d3["vel"] == ((n, 3), f32)
    assert d2["W1"][0] == (n, 101, 32) and d3["W1"][0] == (n, 116, 32)
    assert d2["W2"][0] == (n, 33, 35) and d3["pend_W2"][0] == (8, 33, 39)
    assert d2["resource"] == ((fg, fg), mx.int32) and d3["resource"] == ((6, fg, fg), mx.int32)
    assert d3["chem"] == ((3, 6, fg, fg), f32)
    unchanged = {k for k in d2 if d2[k] == d3[k]}
    assert unchanged == set(d2) - {"pos", "vel", "W1", "W2", "pend_W1", "pend_W2", "resource", "chem"}


# --- ledger names -----------------------------------------------------------------------------------------


def test_heat_buoyancy_is_an_alias_of_slot_15() -> None:
    L = state.LedgerIdx
    assert L.HEAT_BUOYANCY is L.RESERVED and L.HEAT_BUOYANCY == 15 and L.HEAT_BUOYANCY.name == "RESERVED"
    assert len(L) == state.N_LEDGER == 16 and [i.value for i in L] == list(range(16))
    assert state.sinks(2) == state.SINKS and state.sinks(3) == (*state.SINKS, L.HEAT_BUOYANCY)
    assert state.ledger_names(2) == tuple(i.name for i in L)  # the 2D metrics keys, RESERVED included
    names3 = state.ledger_names(3)
    assert names3[15] == "HEAT_BUOYANCY" and names3[:15] == state.ledger_names(2)[:15]
