"""The diel oracle pilot analysis (docs/DIEL.md 7.0) on synthetic metrics.jsonl files."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from alife.analysis import diel_pilot as dp

PERIOD = 100
REPO = Path(__file__).resolve().parents[1]


def make_run(path, effects=(0.10, 0.10, 0.10), noise=0.01, seed=0, days=45, start=0, nulls=False, harv=None):
    """A run with 2 rows per day; O energy = C energy * exp(effect_s) * noise."""
    rng = np.random.default_rng(seed)
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"diel": {"period": PERIOD}}))
    if start:
        (path / "meta.json").write_text(json.dumps({"start_step": start}))
    lines = []
    for d in range(days):
        for k in range(2):
            c = 100.0 * np.exp(rng.normal(0, noise, 3))
            o = c * np.exp(np.array(effects) + rng.normal(0, noise, 3))
            en = [float(x) for x in (*o, *c)]
            n = [30, 20, 10, 150, 100, 50]
            if nulls and k == 1:
                en[0], n[1] = None, 0
            row = {
                "step": start + d * PERIOD + k * 50,
                "diel_orc_pop": [30, 20, 10, 150, 100, 50],
                "diel_orc_n": n,
                "diel_orc_energy": en,
            }
            for g in "oc":
                for s in (1, 2, 3):
                    for kind in ("photic", "all"):
                        for ph in ("day", "night"):
                            v = (harv or {}).get(f"{g}{s}_{kind}_{ph}", 0)
                            row[f"diel_harv_{g}{s}_{kind}_{ph}"] = v
            lines.append(json.dumps(row))
    (path / "metrics.jsonl").write_text("\n".join(lines) + "\n")
    return path


def runs(tmp_path, **kw):
    return [make_run(tmp_path / f"r{i}", seed=i, **kw) for i in range(3)]


def test_planted_effect_is_go(tmp_path):
    r = dp.analyse(runs(tmp_path), boot=200)
    assert r.decision == "GO" and not r.straddling
    assert abs(r.g["pooled"] - 0.10) < 0.01
    assert r.lo["pooled"] >= 0.05
    assert all(r.holm_pass.values())


def test_no_effect_not_go_and_negative_is_stop(tmp_path):
    r0 = dp.analyse(runs(tmp_path / "a", effects=(0, 0, 0), noise=0.05), boot=200)
    assert r0.decision != "GO"
    r1 = dp.analyse(runs(tmp_path / "b", effects=(-0.05,) * 3, noise=0.005), boot=200)
    assert r1.decision == "STOP" and not r1.straddling


def test_one_stratum_effect_is_small(tmp_path):
    r = dp.analyse(runs(tmp_path, effects=(0.0, 0.0, 0.30), noise=0.01), boot=200)
    assert r.decision == "SMALL" and not r.straddling
    assert r.holm_pass == {"1": False, "2": False, "3": True}


def test_straddling_is_small_flagged(tmp_path):
    r = dp.analyse(runs(tmp_path, effects=(0.03,) * 3, noise=0.005), boot=200)
    assert r.decision == "SMALL" and r.straddling


def test_nulls_skipped_and_window(tmp_path):
    clean = dp.analyse([make_run(tmp_path / "a", seed=1)], boot=50)
    dirty = dp.analyse([make_run(tmp_path / "b", seed=1, nulls=True)], boot=50)
    assert np.isfinite(dirty.g["1"]) and abs(dirty.g["1"] - clean.g["1"]) < 0.02
    assert abs(dirty.g["2"] - clean.g["2"]) < 0.02
    assert dirty.n_days["b"] == 30  # days 10..39 only


def test_fork_start_step(tmp_path):
    r = dp.load_run(make_run(tmp_path / "f", start=5000), 10, 40)
    assert r.days[0] == 10 and r.days[-1] == 39


def test_bootstrap_reproducible(tmp_path):
    rs = [dp.load_run(p) for p in runs(tmp_path)]
    a, b, c = dp.bootstrap(rs, 50, 7), dp.bootstrap(rs, 50, 7), dp.bootstrap(rs, 50, 8)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_holm():
    adj, ok = dp.holm([0.04, 0.001, 0.02])
    assert np.allclose(adj, [0.04, 0.003, 0.04]) and ok == [True, True, True]
    assert dp.holm([0.03, 0.03, 0.9])[1] == [False, False, False]


def test_photic_share(tmp_path):
    h = {"o1_photic_day": 3, "o1_all_day": 4, "o1_photic_night": 1, "o1_all_night": 4}
    r = dp.load_run(make_run(tmp_path / "h", harv=h))
    s = dp.photic_share([r])
    assert s["o1_day"] == 0.75 and s["o1_night"] == 0.25
    assert s["c1_day"] is None and s["o2_day"] is None


def test_rule_hash_stable():
    assert dp.rule_hash() == dp.rule_hash()
    assert dp.rule_hash() == "41a16f09002d86983882fb02d2c934b3079db48d3dee570b0274e214563be0f6"
    assert dp.rule_hash(boot=500) != dp.rule_hash()


def test_cli(tmp_path, capsys):
    ps = runs(tmp_path)
    out = tmp_path / "o.json"
    assert dp.main([*map(str, ps), "--boot", "50", "--json", str(out)]) == 0
    assert "decision: GO" in capsys.readouterr().out
    assert json.loads(out.read_text())["decision"] == "GO"


def test_driver_dry_run(tmp_path):
    root = tmp_path / "pilot_root"
    p = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "diel_pilot.py"), "--out", str(root), "--dry-run"],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = p.stdout.strip().splitlines()
    assert len(lines) == 7  # 3 seeds x (warm-up + pilot) + analysis
    assert sum("--steps 40000" in x for x in lines) == 3
    assert (
        sum("--init-from" in x and "diel.oracle=20" in x and "diel.oracle_schedule=2" in x for x in lines)
        == 3
    )
    assert "alife.analysis.diel_pilot2" in lines[-1]
    v1 = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "diel_pilot.py"),
            "--out",
            str(root),
            "--dry-run",
            "--schedule",
            "1",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert not any("oracle_schedule" in x for x in v1) and v1[-1].split()[4] == "alife.analysis.diel_pilot"
    warm = tmp_path / "old"
    w = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "diel_pilot.py"),
            "--out",
            str(root),
            "--dry-run",
            "--warm-from",
            str(warm),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert len(w) == 4 and sum(f"--init-from {warm}/warm_" in x for x in w) == 3
    own = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "diel_pilot.py"),
            "--out",
            str(root),
            "--dry-run",
            "--cap",
            "own",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert sum("diel.oracle_cap=own" in x and "diel.oracle_schedule=2" in x for x in own) == 3
    assert not root.exists()
    root.mkdir()
    (root / "warm_21").mkdir()
    q = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "diel_pilot.py"), "--out", str(root)],
        capture_output=True,
        text=True,
    )
    assert q.returncode == 2 and "refusing" in q.stderr
