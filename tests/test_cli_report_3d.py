"""§27.9 and §27.11 what the CLI reports of a 3D world: ``inspect`` (config block, height line, grounded
share, agent details), the run announcement, the metrics rows of a real run and ``bench``'s environment.
Every addition is 3D only; the 2D summary and bench environment keep their exact key sets.

One 30-step ``tiny3d`` run (f32 weights, heavy metrics at every row) is shared by the module.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest

from alife import bench, cli, layout, metrics
from alife.config import Config, dims
from alife.runio import RunDir, load_checkpoint, write_state_file
from alife.state import to_numpy
from helpers import make_scene

F32 = ["--set", "brain.weight_dtype=float32"]
SAMPLE = ["--set", "run.diversity_sample=64"]  # cheap heavy metrics on CPU, as test_cli_smoke_3d.py


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("cli3d_report") / "run"
    code = cli.main(
        ["run", "--preset", "tiny3d", *F32, *SAMPLE, "--steps", "30", "--out", str(out), "--quiet"]
    )
    assert code == 0
    return out


def _inspect(run_dir: Path, capsys: pytest.CaptureFixture[str], *args: str) -> dict:
    capsys.readouterr()
    assert cli.main(["inspect", str(run_dir), "--json", *args]) == 0
    return json.loads(capsys.readouterr().out)


# --- inspect ------------------------------------------------------------------------------------------------


def test_inspect_json_reports_dims_and_height(run_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    info = _inspect(run_dir, capsys)
    c, s = info["config"], info["state"]
    assert c["dims"] == 3 and c["world_height"] == 24.0 and c["field_grid_z"] == 6
    k = list(c).index("field_grid")
    assert list(c)[k + 1 : k + 4] == ["dims", "world_height", "field_grid_z"]
    rd = RunDir.open(run_dir)
    cfg = rd.config()
    state = load_checkpoint(rd.checkpoint_at(30), cfg)
    alive = to_numpy(state.alive)
    z = to_numpy(state.pos)[alive, 2].astype(np.float64)
    d = dims(cfg)
    assert info["n_alive"] == int(alive.sum()) > 0
    assert s["height_mean"] == pytest.approx(z.mean(), rel=1e-12)
    assert s["height_q25"] <= s["height_q50"] <= s["height_q75"]
    assert d.z_lo <= s["height_q25"] and s["height_q75"] <= d.z_hi
    assert sum(s["band_frac"]) == pytest.approx(1.0) and len(s["band_frac"]) == 4
    assert info["grounded_frac"] == pytest.approx(float((z <= d.z_ground).mean()), rel=1e-12)
    assert "body_z_extent_mean" in s and "body_z_extent_max" in s  # inspect takes the heavy snapshot
    assert info["carrying_capacity_est"] == metrics.carrying_capacity(state, cfg)


def test_inspect_text_has_a_height_line(run_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    capsys.readouterr()
    assert cli.main(["inspect", str(run_dir)]) == 0
    out = capsys.readouterr().out
    (line,) = [ln for ln in out.splitlines() if ln.startswith("height (world 24)")]
    assert "quartiles" in line and "grounded" in line and "surface" in line and "bands [" in line
    assert "dims=3" in out and "world_height=24" in out and "field_grid_z=6" in out


def test_inspect_agent_reports_the_3d_fields(run_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rd = RunDir.open(run_dir)
    cfg = rd.config()
    state = load_checkpoint(rd.checkpoint_at(30), cfg)
    uid = int(to_numpy(state.uid)[np.flatnonzero(to_numpy(state.alive))[0]])
    agent = _inspect(run_dir, capsys, "--agent", str(uid))["agent"]
    f = agent["fields"]
    assert agent["alive"] and len(f["pos"]) == 3 and len(f["vel"]) == 3
    assert abs(f["pitch"]) <= layout.PITCH_MAX and 0.0 <= f["buoy_cap"] <= 1.0
    assert agent["height"] == f["pos"][2]
    assert agent["grounded"] is (f["pos"][2] <= dims(cfg).z_ground)


def test_inspect_an_extinct_3d_state(
    micro3d_cfg: Config, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With nobody alive every height statistic is None; the text line says so and `inspect` still exits 0."""
    path = tmp_path / "extinct.safetensors"
    write_state_file(path, make_scene(micro3d_cfg, []), micro3d_cfg)
    capsys.readouterr()
    assert cli.main(["inspect", str(path)]) == 0
    out = capsys.readouterr().out
    (line,) = [ln for ln in out.splitlines() if ln.startswith("height (world")]
    assert line.endswith("nobody alive") and "quartiles" not in line
    assert cli.main(["inspect", str(path), "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    s = info["state"]
    assert info["n_alive"] == 0 and info["grounded_frac"] is None
    assert s["height_mean"] is None and s["band_frac"] is None and s["clade_height_mi"] is None


# --- the rows of a real run ---------------------------------------------------------------------------------


def test_rows_of_a_3d_run_carry_the_3d_keys(run_dir: Path) -> None:
    rows = RunDir.open(run_dir).metrics_rows()
    assert [r["step"] for r in rows] == [10, 20, 30]
    for r in rows:
        assert 0.0 <= r["grounded_frac"] <= 1.0 and r["z_floor_clamps"] >= 0 and r["z_surface_clamps"] >= 0
        assert r["buoy_heat"] == r["ledger"]["HEAT_BUOYANCY"] > 0
        assert 1.0 <= r["height_mean"] <= 23.0 and sum(r["band_frac"]) == pytest.approx(1.0)
        assert r["speed_mean"] >= 0.0 and 0.0 <= r["pitch_abs_mean"] <= 1.4 and r["clade_height_mi"] >= 0.0
        assert 0.0 <= r["buoy_cap_q10"] <= r["buoy_cap_q50"] <= r["buoy_cap_q90"] <= 1.0
        assert "body_z_extent_mean" in r and "grounded_frac" in r and r["ledger_error"] == 0


def test_the_first_row_estimates_the_capacity_by_occupancy(run_dir: Path) -> None:
    """The row is written at step 10, with the patches drifted from their step-0 positions and the founders
    still spread over the column: the occupancy-weighted estimate (§27.10), between the floor layer's alone
    (about 409: tiny's 426 less the larger founder brain and the cap upkeep) and every layer's (1 456)."""
    rd = RunDir.open(run_dir)
    cfg = rd.config()
    first = rd.metrics_rows()[0]["carrying_capacity_est"]
    k0 = metrics.carrying_capacity(load_checkpoint(rd.checkpoint_at(0), cfg), cfg)
    assert first == pytest.approx(k0, rel=5e-2)
    assert 1000 < first < 1500  # the six layers hold about 3.6 floors' food, and the founders use them all


# --- announce, summary and bench: 3D only -------------------------------------------------------------------


def test_summary_and_announce_name_the_height_only_in_3d(
    tiny_cfg: Config, tiny3d_cfg: Config, capsys: pytest.CaptureFixture[str]
) -> None:
    s2, s3 = cli._summary(tiny_cfg), cli._summary(tiny3d_cfg)
    assert not {"dims", "world_height", "field_grid_z"} & set(s2)
    k = list(s2).index("field_grid") + 1
    assert list(s3) == [*list(s2)[:k], "dims", "world_height", "field_grid_z", *list(s2)[k:]]
    assert (s3["dims"], s3["world_height"], s3["field_grid_z"]) == (3, 24.0, 6)
    rd = types.SimpleNamespace(path=Path("runs/x"))
    capsys.readouterr()
    cli._announce(rd, tiny_cfg, 0, 10, False)  # type: ignore[arg-type]
    cli._announce(rd, tiny3d_cfg, 0, 10, False)  # type: ignore[arg-type]
    two, three = capsys.readouterr().out.splitlines()
    assert "L=128 weights" in two and "height" not in two
    assert "L=128 height=24 weights" in three


def test_bench_environment_and_report_name_the_height_only_in_3d(
    tiny_cfg: Config, tiny3d_cfg: Config
) -> None:
    e2, e3 = bench._environment(tiny_cfg), bench._environment(tiny3d_cfg)
    assert not {"dims", "world_height", "field_grid_z"} & set(e2["config"])
    assert (e3["config"]["dims"], e3["config"]["world_height"], e3["config"]["field_grid_z"]) == (3, 24.0, 6)
    base = {
        "compile_s": 1.0,
        "ms_per_step": 2.0,
        "ms_per_step_median": 2.0,
        "ms_per_step_p90": 3.0,
        "steps_per_s": 500.0,
        "peak_mem_gb": 0.1,
        "n_alive_start": 10,
        "n_alive_end": 11,
        "births": 2,
        "deaths": 1,
        "diag_ok": True,
        "diag_flags": [],
        "invariants_ok": True,
        "donation": {"verdict": "ok", "excess_gb": 0.0, "excess_over_weights": 0.0},
    }
    t2 = bench.format_report({**e2, **base}).splitlines()[1]
    t3 = bench.format_report({**e3, **base}).splitlines()[1]
    assert "height" not in t2 and "L=128 K=" in t2
    assert "L=128 height=24 K=" in t3
    assert t3.count("H=") == 1 and f"H={dims(tiny3d_cfg).h}" in t3  # H stays the hidden size
