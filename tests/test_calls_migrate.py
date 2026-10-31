"""`alife migrate --enable calls` onto an evolved comm world (DESIGN §47.8, §43.8): the real registry.

The first form (`--enable calls`) changes nothing a world can observe: no brain block, no state field, no
counter, the same layout strings, bit-identical steps. The second (`--set calls.trigger=true --set
calls.events=...`) adds the zero `evt` window and gates the voice. The source must already carry comm with the
width of the events (a width change of an enabled feature is never a migration)."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from alife import cli, ext
from alife.config import dims, load_config
from alife.runio import RunDir, load_checkpoint
from alife.state import LedgerIdx, counter_index, init_state, state_digest, to_numpy
from alife.step import make_step

F32 = "brain.weight_dtype=float32"
WIDTH2 = ["comm.enabled=true", "comm.width=2", "comm.cost=8"]


def _source(tmp_path: Path, preset: str, steps: int = 40):
    cfg = load_config(preset=preset, overrides=[F32, *WIDTH2, "repro.min_age=100000"])
    fn, s = make_step(cfg), init_state(cfg)
    for _ in range(steps):
        s, _ = fn(s)
    mx.eval(s)
    rd = RunDir.create(tmp_path / "src", cfg)
    rd.save_checkpoint(s, cfg)
    rd.wait_saves()
    return cfg, s, rd.path


def _migrate(src: Path, out: Path, *args: str, capsys) -> int:
    rc = cli.main(["migrate", str(src), *args, "--out", str(out), "--quiet"])
    if rc:
        print(capsys.readouterr().err)
    return rc


@pytest.mark.parametrize("preset", ["tiny", "tiny3d"])
def test_enable_alone_is_invisible_and_the_trigger_adds_the_zero_window(
    preset: str, tmp_path: Path, capsys
) -> None:
    cfg, s, src = _source(tmp_path, preset)
    # (1) --enable calls: nothing a world can observe changes
    out1 = tmp_path / "enabled"
    assert _migrate(src, out1, "--enable", "calls", capsys=capsys) == 0
    run1 = RunDir.open(out1)
    cfg1, meta1 = run1.config(), run1.meta()
    assert cfg1.calls.enabled and not cfg1.calls.trigger and meta1["migrated_from"]["added"] == ["calls"]
    assert meta1["layout_features"] == "comm2@1" == ext.layout_features(cfg)  # no brain footprint
    assert meta1["genome_layout_version"] == meta1["migrated_from"]["source_genome_layout_version"]
    s1 = load_checkpoint(run1.checkpoints()[-1][1], cfg1)
    assert type(s1)._fields == type(s)._fields and state_digest(s1) == state_digest(s)
    a, oa = make_step(cfg)(s)
    b, ob = make_step(cfg1)(s1)
    mx.eval(a, b, oa, ob)
    assert state_digest(a) == state_digest(b)  # bit-identical, floats included
    assert np.array_equal(to_numpy(oa.counters), to_numpy(ob.counters))
    assert np.array_equal(to_numpy(oa.ledger), to_numpy(ob.ledger))

    # (2) the trigger: the zero window, the gated voice
    out2 = tmp_path / "trigger"
    args = [
        "--enable",
        "calls",
        "--set",
        "calls.trigger=true",
        "--set",
        "calls.events=3",
        "--set",
        "calls.hold=2",
    ]
    assert _migrate(src, out2, *args, capsys=capsys) == 0
    run2 = RunDir.open(out2)
    cfg2, meta2 = run2.config(), run2.meta()
    assert cfg2.calls.trigger and cfg2.calls.events == 3 and meta2["layout_features"] == "comm2@1"
    s2 = load_checkpoint(run2.checkpoints()[-1][1], cfg2)
    d = dims(cfg2)
    assert s2.evt.shape == (d.n, 2) and s2.evt.dtype == mx.uint8 and not to_numpy(s2.evt).any()
    assert state_digest(s2) != state_digest(s) and to_numpy(s2.W1).shape == to_numpy(s.W1).shape
    assert np.array_equal(to_numpy(s2.W1), to_numpy(s.W1)) and np.array_equal(
        to_numpy(s2.msg), to_numpy(s.msg)
    )
    # the first step: every window is closed, so the evolved population is silent (without the trigger it
    # speaks)
    fn2 = make_step(cfg2)
    s2, o2 = fn2(s2)
    mx.eval(s2, o2)
    assert int(to_numpy(o2.counters)[counter_index(cfg2, "COMM_SPEAKERS")]) == 0
    assert int(to_numpy(oa.counters)[counter_index(cfg, "COMM_SPEAKERS")]) > 0
    opened = 0
    for _ in range(40):
        s2, o2 = fn2(s2)
        mx.eval(s2, o2)
        led = to_numpy(o2.ledger)
        assert led[LedgerIdx.FLOW_STOLEN] == led[LedgerIdx.FLOW_ATTACK_GAIN] + led[LedgerIdx.FLOW_SPILL]
        opened += int(to_numpy(o2.counters)[counter_index(cfg2, "CALLS_OPEN_HURT")])
    assert int(to_numpy(s2.diag)) & 0xFFFF == 0
    # it resumes through the CLI
    assert cli.main(["resume", str(out2), "--steps", "3", "--quiet"]) == 0


def test_the_trigger_cannot_be_added_to_a_world_that_already_has_calls(tmp_path: Path, capsys) -> None:
    """Turning the trigger on later changes the shape of a state field of a kept feature, which a migration
    refuses: either enable and trigger in one call, or fork the run (`alife run --init-from`)."""
    _, _, src = _source(tmp_path, "tiny")
    mid = tmp_path / "mid"
    assert _migrate(src, mid, "--enable", "calls", capsys=capsys) == 0
    capsys.readouterr()
    rc = cli.main(["migrate", str(mid), "--set", "calls.trigger=true", "--set", "calls.events=3",
                   "--out", str(tmp_path / "late"), "--quiet"])  # fmt: skip
    assert rc != 0 and "evt" in capsys.readouterr().err
    # a fork fills the window from its rest value
    fork = tmp_path / "fork"
    assert cli.main(["run", "--init-from", str(mid), "--steps", "6", "--out", str(fork), "--quiet",
                     "--set", "calls.trigger=true", "--set", "calls.events=3"]) == 0  # fmt: skip
    rd = RunDir.open(fork)
    sf = load_checkpoint(rd.latest_checkpoint(), rd.config())
    assert sf.evt.shape[1] == 2 and int(to_numpy(sf.step)) > 0


def test_a_source_without_comm_or_with_another_width_is_refused(tmp_path: Path, capsys) -> None:
    cfg = load_config(preset="tiny", overrides=[F32])
    s = init_state(cfg)
    rd = RunDir.create(tmp_path / "plain", cfg)
    rd.save_checkpoint(s, cfg)
    rd.wait_saves()
    rc = cli.main(["migrate", str(rd.path), "--enable", "calls", "--out", str(tmp_path / "o1"), "--quiet"])
    assert rc != 0 and "calls.enabled needs comm.enabled" in capsys.readouterr().err
    # comm of width 4 cannot become the two channels of events 5/3 (a width change is never a migration)
    cfg4 = load_config(preset="tiny", overrides=[F32, "comm.enabled=true"])
    rd4 = RunDir.create(tmp_path / "w4", cfg4)
    rd4.save_checkpoint(init_state(cfg4), cfg4)
    rd4.wait_saves()
    rc = cli.main(["migrate", str(rd4.path), "--enable", "calls", "--set", "calls.trigger=true",
                   "--set", "calls.events=3", "--out", str(tmp_path / "o2"), "--quiet"])  # fmt: skip
    assert rc != 0 and "comm.width == the number of events" in capsys.readouterr().err
    # a migration changes feature keys only
    rc = cli.main(["migrate", str(rd4.path), "--enable", "calls", "--set", "energy.e_init=9",
                   "--out", str(tmp_path / "o3"), "--quiet"])  # fmt: skip
    assert rc != 0


def test_the_enable_hook_switches_only_enabled() -> None:
    cfg = load_config(preset="tiny", overrides=[F32, *WIDTH2])
    spec = next(sp for sp in ext.REGISTRY if sp.name == "calls")
    on = spec.enable(cfg)
    assert on.calls.enabled and not on.calls.trigger and on.calls.events == 15 and on.calls.hold == 4
    assert on.comm == cfg.comm
