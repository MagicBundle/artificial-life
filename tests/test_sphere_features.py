"""The five features on the shell, each and all together (DESIGN §46.5; sphere series S3).

Every feature of ``tests/sphere_feature_suite.SPECS`` runs on ``tiny3s`` with its action forced on: the
16-term ledger closes at every step (no 17th term), no error diag bit, ``invariants.full_check`` (the host
ranges, the aperture, the bond graph) every ten steps, and the feature acts (its counter is non-zero). The
combination of all of them keeps the observer channel pure, is deterministic and resumes from a checkpoint to
the uninterrupted digest; a config without the features has none of their footprint. The box suites' own
step-level tests run on the sphere through the parameter ``"s"`` of their helpers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import sphere_feature_suite as su
from alife import config


def test_the_five_features_and_the_roles_are_ready_on_a_sphere() -> None:
    assert {"structures", "niche", "comm", "bonds", "roles", "arms", "observers", "director"} == config.SPHERE_READY
    assert set(config.SPHERE_FEATURE_PR) - config.SPHERE_READY == {"weather"}


@pytest.mark.parametrize("name", sorted(su.SPECS))
def test_each_feature_runs_on_the_shell_with_a_closing_ledger(name: str) -> None:
    out = su.check_run(name, 50)
    assert out["counter"] > 0 and out["alive"] > 0


@pytest.mark.parametrize("name", sorted(su.SPECS))
def test_an_off_config_has_none_of_the_footprint(name: str) -> None:
    su.check_off_footprint(name)


def test_all_features_together_are_pure_deterministic_and_resumable(tmp_path: Path) -> None:
    su.check_observe_pure("all", 10)
    su.check_deterministic("all", 15)
    su.check_resume("all", tmp_path, 12)


def test_a_run_with_every_feature_writes_metrics_dumps_and_resumes(tmp_path: Path) -> None:
    """``alife run`` / ``resume`` on a sphere with all the features (eager: the compiled step is in
    ``test_sphere_features_compiled``): every metrics row closes the ledger and carries the feature keys, the
    grid dumps have the shell's shape ``(FZ, Npix)``, and the resume continues the run."""
    import numpy as np

    from alife import cli
    from alife.runio import RunDir

    out = tmp_path / "run"
    sets = [
        *("--set", "brain.weight_dtype=float32"),
        *("--set", "run.compile=false"),
        *("--set", "arms.enabled=true"),
        *("--set", "run.mat_every=20"),
        *("--set", "run.niche_every=20"),
        *("--set", "run.arms_every=20"),
        *("--set", "niche.silent_init=false"),
        *("--set", "structures.open_frac=0.02"),
    ]
    args = [
        *("--preset", "tiny3s"),
        *(x for o in ("build", "niche3d", "bonds", "comm") for x in ("--overlay", o)),
    ]
    assert cli.main(["run", *args, *sets, *("--steps", "40"), *("--out", str(out)), "--quiet"]) == 0
    rd = RunDir.open(out)
    rows = rd.metrics_rows()
    assert rows and all(r["ledger_error"] == 0 and r["diag"] & 0xFFFF == 0 for r in rows)
    keys = ("build_units", "farm_cells", "comm_speaker_frac", "bond_role_frac_gen", "arms_atk_mean")
    assert all(any(k in r for r in rows) for k in keys), [k for k in keys if not any(k in r for r in rows)]
    assert sum(r["build_units"] for r in rows) > 0 and all(r["build_trapped"] == 0 for r in rows)
    assert np.load(rd.dump_path("mat", 20)).shape == (8, 768) and np.load(
        rd.dump_path("niche", 20)
    ).shape == (8, 768)
    assert cli.main(["resume", str(out), "--until", "60", "--quiet"]) == 0
    assert RunDir.open(out).meta()["status"] == "done" and RunDir.open(out).metrics_rows()[-1]["step"] >= 60


# The digest of ``tiny3s`` (float32 weights) after 25 EAGER steps on the CPU, recorded at the S2 head
# (522c0ae, before any S3 change) and unchanged by S3: the features off leave the dry sphere bit-identical.
# Pinned for the platform and MLX version it was recorded on, like ``test_sphere_trajectory.GOLDEN_SPHERE``
# (whose compiled pin proves the same for the compiled step).
DRY_EAGER_25 = {"mlx": "0.32.3", "Darwin-arm64-cpu": "55ecf2bc9f94a53918bb03e21d2780ea"}


def test_the_dry_sphere_is_bit_identical_to_the_one_before_the_features() -> None:
    import platform

    import mlx.core as mx

    from alife.config import load_config
    from alife.state import init_state, state_digest
    from alife.step import make_step

    key = f"{platform.system()}-{platform.machine()}-{'gpu' if mx.default_device().type == mx.gpu else 'cpu'}"
    if mx.__version__ != DRY_EAGER_25["mlx"] or key not in DRY_EAGER_25:
        pytest.skip(f"pinned for MLX {DRY_EAGER_25['mlx']} on Darwin-arm64-cpu")
    cfg = load_config(preset="tiny3s", overrides=["brain.weight_dtype=float32", "run.compile=false"])
    s, fn = init_state(cfg), make_step(cfg, compile=False)
    for _ in range(25):
        s, out = fn(s)
        mx.eval(s, out)
    assert state_digest(s).startswith(DRY_EAGER_25[key])
