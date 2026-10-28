"""The panel (DESIGN §44.7), the CLI shell, and the lint rules of the package (§44.9)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from alife.analysis.emergence import __main__ as cli
from alife.analysis.emergence import panel, prereg

PKG = Path(panel.__file__).parent


def _run_dir(tmp_path: Path, rows: list[dict] | None = None) -> Path:
    run = tmp_path / "mini_x"
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({"seed": 4, "config_hash": "abc", "start_step": 500000}))
    rows = (
        rows
        if rows is not None
        else [{"step": 100 * i, "n_alive": 1000 + i, "ledger_error": 0} for i in range(1, 6)]
    )
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run


def _listing(path: Path) -> dict[str, int]:
    return {str(p.relative_to(path)): p.stat().st_mtime_ns for p in path.rglob("*")}


def test_panel_schema_is_stable() -> None:
    p = panel.new_panel(run="r", arm="U0", seed=1, config_hash="h", fork_step=500000, windows=[(0, 10)])
    assert tuple(p) == ("header", "A", "B", "C", "D", "endpoints", "not_claimed")
    assert tuple(p["header"]) == (
        "panel_version", "run", "arm", "seed", "config_hash", "fork_step", "windows", "versions",
        "prereg_hash",
        "gates", "flags",
    )  # fmt: skip
    assert p["header"]["panel_version"] == panel.PANEL_VERSION == 1
    assert list(p["endpoints"]) == [
        "P1",
        "P2",
        "P3",
        "P4",
        "P5",
        "S_arms",
        "S_bonds",
        "S_comm",
        "S_niche",
        "S_structures",
    ]
    assert tuple(p["header"]["gates"]) == ("population", "ledger", "resim")
    assert set(p["endpoints"]["P1"]) == {
        "status",
        "value",
        "blocks",
        "direction",
        "confirmatory",
        "twin_spread",
        "needs_missing",
    }
    for k in "ABC":
        assert p[k]["status"] == "needs_missing"  # never a zero
    assert set(p["D"]["cards"]) == {"arms", "bonds", "comm", "niche", "structures"}
    assert any("composite" in s for s in p["not_claimed"]) and any(
        "lower bound" in s for s in p["not_claimed"]
    )
    assert p["endpoints"]["P2"]["confirmatory"] is False and p["endpoints"]["P4"]["confirmatory"] is True
    panel.validate(p)
    q = json.loads(json.dumps(p))
    q["header"]["panel_version"] = 2
    with pytest.raises(panel.PanelError):
        panel.validate(q)


def test_health_gates() -> None:
    rows = [{"step": s, "n_alive": 1000, "ledger_error": 0} for s in range(100, 600, 100)]
    g = panel.health_gates(rows, (0, 500), population_ref=1100, resim_ok=True)
    assert g["population"]["ok"] and g["ledger"]["ok"] and g["resim"]["ok"]
    bad = panel.health_gates(rows, (0, 500), population_ref=300)  # 3.3x the reference
    assert bad["population"]["ok"] is False and bad["resim"]["ok"] is None
    rows[2]["ledger_error"] = 5
    assert panel.health_gates(rows, (0, 500))["ledger"]["ok"] is False
    p = panel.new_panel(run="r", arm="a", seed=0, config_hash="h", fork_step=0, windows=[], gates=bad)
    assert p["header"]["flags"]["blocked"] is True


def test_build_panel_from_a_synthetic_run_leaves_the_run_alone(tmp_path: Path) -> None:
    run = _run_dir(tmp_path)
    before = _listing(run)
    p = panel.build_panel(run, [(0, 500)], arm="U0", prereg_hash="f" * 64, population_ref=1000)
    out = tmp_path / "out"
    pj, pm = panel.write_panel(out, p)
    assert _listing(run) == before
    assert panel.read_panel(pj)["header"] == p["header"]
    assert (
        p["header"]["seed"] == 4
        and p["header"]["fork_step"] == 500000
        and p["header"]["prereg_hash"] == "f" * 64
    )
    md = pm.read_text()
    assert "not evaluated" in md and "## Not claimed" in md and "composite" in md
    assert not [f for f in out.iterdir() if f.name.startswith(".")]  # atomic writes leave no temp file


def test_cli_prereg_synth_compare_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pre = tmp_path / "prereg.json"
    assert cli.main(["prereg", "--out", str(pre), "--freeze"]) == 0
    h = prereg.load_prereg(pre)["sha256"]
    d = tmp_path / "synth"
    args = ["synth", "--out", str(d), "--seed", "3", "--prereg-hash", h]
    args += [x for e in ("P1", "P4", "P5") for x in ("--effect", f"bonds:{e}=3")]
    assert cli.main(args) == 0
    files = sorted(d.glob("panel_*.json"))
    assert len(files) == 54
    out = tmp_path / "cmp"
    assert cli.main(["compare", *map(str, files), "--prereg", str(pre), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "bonds: supported" in printed
    cmp = json.loads((out / "compare.json").read_text())
    assert cmp["features"]["bonds"]["call"] == "supported" and cmp["prereg"]["frozen"]
    md = (out / "compare.md").read_text()
    assert (
        "Three seeds forked from one evolved population" in md
        and "fresh runs from the shared evolved sources" in md
    )
    # a different prereg file: the panels' hash no longer matches, the comparison refuses
    other = tmp_path / "other.json"
    p2 = prereg.default_prereg()
    p2["frozen"] = True
    p2["analysis"]["ep_cell"] = 16
    prereg.write_prereg(other, p2)
    assert cli.main(["compare", *map(str, files), "--prereg", str(other), "--out", str(out)]) == 2
    assert (
        cli.main(["compare", *map(str, files), "--prereg", str(other), "--out", str(out), "--exploratory"])
        == 0
    )
    assert "EXPLORATORY" in (out / "compare.md").read_text()


def test_cli_panel_scorecards_and_stubs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = _run_dir(tmp_path)
    out = tmp_path / "o"
    assert cli.main(["panel", str(run), "--windows", "0:500", "--arm", "U0", "--out", str(out)]) == 0
    assert (out / "panel.json").is_file()
    assert cli.main(["modes", "x"]) == 2 and "E5" in capsys.readouterr().err
    assert cli.main(["modes"]) == 2
    with pytest.raises(SystemExit):
        cli.main(["record", str(run), "--window", "5:1"])  # not A < B


# --- lint (§44.9 test 8) ---


def _sources() -> dict[str, str]:
    return {p.name: p.read_text() for p in PKG.glob("*.py")}


def test_mlx_only_in_record_and_no_scipy_or_network() -> None:
    for name, text in _sources().items():
        code = "\n".join(re.sub(r'""".*?"""', "", text, flags=re.S).splitlines())
        mlx_import = re.search(r"^\s*(import|from)\s+mlx", code, flags=re.M)
        assert bool(mlx_import) == (name == "record.py"), name
        assert not re.search(
            r"^\s*(import|from)\s+(scipy|sklearn|requests|urllib|http|socket|aiohttp)", code, flags=re.M
        ), name
        assert "lmstudio" not in text.lower() and "ALIFE_LLM" not in text


def test_no_write_into_a_run_directory(tmp_path: Path) -> None:
    """The package writes only where ``--out`` says: source scan plus a before/after listing of a tmp run."""
    for name, text in _sources().items():
        for m in re.finditer(r"(write_text|write_bytes|open\([^)]*[\"'][wax])", text):
            line = text[: m.start()].count("\n") + 1
            assert name in ("panel.py", "compare.py", "prereg.py", "frames.py", "__main__.py", "epr_io.py"), (
                name,
                line,
            )
    run = _run_dir(tmp_path)
    before = _listing(run)
    out = tmp_path / "o"
    cli.main(["panel", str(run), "--windows", "0:500", "--out", str(out)])
    assert _listing(run) == before
