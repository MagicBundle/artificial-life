"""W1 build step 2 (DESIGN §30, spec §9.1, §9.8, §9.10, §9.12): the weather update inside the engine step.

The inert-on identity (weather on with every coupling at zero is the weather-off engine), compiled against
eager for the fused ``agent_wind + advance`` graph, the weather being agent-independent, resume equality,
``Diag.WEATHER_WARN`` and the device check.

CPU and fast (the whole file runs in about 20 s: the compiled forms compile the step in 1 to 2 s on the CPU).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from alife import cli, invariants, runner
from alife import weather as W
from alife.config import Config, WeatherConfig, apply_overrides, dims
from alife.rng import InitStream, Stream, init_keys, step_keys
from alife.runio import RunDir, load_checkpoint
from alife.state import (
    ERROR_MASK,
    Diag,
    State3,
    init_state,
    legacy_digest,
    state_digest,
    to_numpy,
)
from alife.step import make_step
from helpers import ledger_net
from weather_helpers import fixed_wx, storm2d_config, storm3d_config

STEPS = 50


@pytest.fixture(scope="module", params=["3d", "2d"])
def cfg(request: pytest.FixtureRequest) -> Config:
    return storm3d_config() if request.param == "3d" else storm2d_config()


def _inert(c: Config) -> Config:
    """Weather on, every coupling off: no wind (alpha 0), no rain, no hazard, no bond break."""
    wc = replace(c.weather, alpha0=0.0, alpha1=0.0, rain_gain=0.0, storm_p=0.0, storm_break_u=0.0)
    return replace(c, weather=wc)


def _off(c: Config) -> Config:
    return replace(c, weather=WeatherConfig())


def _run(c: Config, steps: int, *, compile: bool, s0=None):
    """``(legacy digests per step, counters, ledgers, final state)``."""
    s = init_state(c) if s0 is None else s0
    fn = make_step(c, compile=compile)
    digs, cnts, leds = [], [], []
    for _ in range(steps):
        s, out = fn(s)
        mx.eval(s, out)
        digs.append(legacy_digest(s))
        cnts.append(to_numpy(out.counters))
        leds.append(to_numpy(out.ledger))
    return digs, np.stack(cnts), np.stack(leds), s


def _same_counters(off: np.ndarray, on: np.ndarray) -> bool:
    """The weather-on counters are the weather-off ones plus the last entry, ``DEATHS_STORM`` (0 here)."""
    return on.shape[1] == off.shape[1] + 1 and np.array_equal(off, on[:, :-1]) and not on[:, -1].any()


# --- the inert-on identity (spec §9.1) ----------------------------------------------------------------------


def test_inert_weather_on_is_the_weather_off_engine_eager(cfg: Config) -> None:
    """Legacy fields, counters and ledgers are bitwise equal over 20 eager steps; ``wx`` differs."""
    a = _run(_off(cfg), 20, compile=False)
    b = _run(_inert(cfg), 20, compile=False)
    assert a[0] == b[0]
    assert _same_counters(a[1], b[1]) and np.array_equal(a[2], b[2])
    assert int(to_numpy(b[3].diag)) == 0  # the health gate stays quiet on a calm start
    assert not np.array_equal(to_numpy(a[3].wx), to_numpy(b[3].wx))  # the weather did advance


def test_inert_weather_on_is_the_weather_off_engine_compiled(cfg: Config) -> None:
    a = _run(_off(cfg), STEPS, compile=True)
    b = _run(_inert(cfg), STEPS, compile=True)
    assert a[0] == b[0]
    assert _same_counters(a[1], b[1]) and np.array_equal(a[2], b[2])
    assert not np.array_equal(to_numpy(a[3].wx), to_numpy(b[3].wx))


# --- the weather is agent-independent (spec §3.5, §9.12) ----------------------------------------------------


def _standalone_wx(c: Config, steps: int) -> mx.array:
    """The engine's key chain driven without an engine: ``state.key -> step_keys -> ks[WEATHER] -> split``."""
    key = init_keys(c.run.seed)[InitStream.STATE_KEY]
    ws = W.unpack(W.rest_array(c))
    for t in range(steps):
        key, ks = step_keys(key)
        kw = mx.random.split(ks[Stream.WEATHER], 2)
        ws, _ = W.advance(ws, mx.array(t, dtype=mx.int32), kw[0], c)
    return W.pack(ws)


def test_weather_is_a_function_of_config_and_seed_only() -> None:
    """30 engine steps with agents born and dying, and with another ``n_init``: ``wx`` is bitwise the
    standalone model's (the hazard-on form is added with the hazard). Seed 2 draws a disturbance at step 28,
    so the key chain is exercised."""
    c = storm3d_config("run.seed=2")
    ref = to_numpy(_standalone_wx(c, 30))
    calm = to_numpy(_standalone_wx(storm3d_config("run.seed=2", "weather.seed_rate=0.0"), 30))
    assert not np.array_equal(ref, calm)  # a disturbance did enter: the test is sensitive to the key chain
    for n_init in (64, 12):
        c2 = storm3d_config("run.seed=2", n_init=n_init)
        s = init_state(c2)
        fn = make_step(c2, compile=False)
        for _ in range(30):
            s, out = fn(s)
            mx.eval(s, out)
        assert np.array_equal(to_numpy(s.wx), ref), n_init
        assert int(to_numpy(s.alive).sum()) > 0
    assert not np.array_equal(ref, to_numpy(W.rest_array(c)))


def test_the_seed_changes_the_weather_and_the_alive_set_does_not() -> None:
    c0, c2 = storm3d_config(), storm3d_config("run.seed=2")
    assert not np.array_equal(to_numpy(_standalone_wx(c0, 30)), to_numpy(_standalone_wx(c2, 30)))
    s = init_state(c2)
    killed = s._replace(alive=mx.zeros_like(s.alive))  # nobody alive: the weather steps all the same
    fn = make_step(c2, compile=False)
    for _ in range(30):
        killed, _ = fn(killed)
        mx.eval(killed)
    assert np.array_equal(to_numpy(killed.wx), to_numpy(_standalone_wx(c2, 30)))


# --- compiled equals eager (spec §9.8) ---------------------------------------------------------------------


def _same_up_to_float_noise(a, b, cls) -> None:
    """The ``test_eager_step_matches_compiled`` form: integers exact, floats ``allclose``; ``wx`` is bitwise
    (it is agent-independent and the fused probe was bitwise equal on the CPU)."""
    for name in cls._fields:
        x, y = to_numpy(getattr(a, name)), to_numpy(getattr(b, name))
        if name == "wx":
            assert np.array_equal(x, y), "wx must be bitwise equal compiled and eager"
        elif x.dtype.kind == "f":
            np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-5, err_msg=name)
        else:
            assert np.array_equal(x, y), name


def test_the_compiled_step_matches_the_eager_step(cfg: Config) -> None:
    """50 steps; each step is run by the compiled and the eager function **from the same input state** (the
    compiled trajectory). Free-running the two diverges after a few dozen steps with the weather off as well
    (a 1.5e-8 float difference flips a discrete choice; 2D ``tiny`` at step 26), so equality of whole
    trajectories is not a property of the engine (DESIGN §14.2); equality of one step is."""
    c = apply_overrides(cfg, ["weather.storm_p=0.0"])
    s = init_state(c)._replace(wx=fixed_wx(c, u1=0.12, v1=-0.08, u2=0.05, v2=0.03))  # winds that act
    fe, fc = make_step(c, compile=False), make_step(c, compile=True)
    for t in range(STEPS):
        se, oe = fe(s)
        sc, oc = fc(s)
        mx.eval(se, oe, sc, oc)
        assert np.array_equal(to_numpy(oe.counters), to_numpy(oc.counters)), t
        assert np.array_equal(to_numpy(oe.ledger), to_numpy(oc.ledger)), t
        for x, y in zip(oe.deaths, oc.deaths, strict=True):
            x, y = to_numpy(x), to_numpy(y)
            if x.dtype.kind == "f":
                np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-5)
            else:
                assert np.array_equal(x, y), t
        _same_up_to_float_noise(se, sc, type(se))
        s = sc
    assert dims(c).nd == (3 if isinstance(s, State3) else 2)
    assert int(to_numpy(s.alive).sum()) > 0 and not int(to_numpy(s.diag)) & ERROR_MASK


def test_the_fused_wind_and_update_graph_compiles_and_equals_eager(cfg: Config) -> None:
    """The graph the step traces: one ``unpack``, ``wind_fields`` once, ``sample_wind``, ``weather.step`` and
    ``pack`` in one ``mx.compile``, run twice (the MLX duplicate-roll bug is the reason this is a test)."""
    d = dims(cfg)
    s = init_state(cfg)

    def fused(wx, pos, cap, alive, t, key):
        ws = W.unpack(wx)
        wind = W.agent_wind(ws, pos, cap if d.d3 else None, alive, cfg)
        ws2, wd = W.advance(ws, t, key, cfg)
        return W.pack(ws2), wind.force, wind.speed, W.health(wd, wind)

    comp = mx.compile(fused)
    wx = fixed_wx(cfg, u1=0.15, v1=0.05, u2=-0.1)
    cap = s.buoy_cap if d.d3 else mx.zeros((d.n,))
    key = mx.random.key(11)
    we, wc = wx, wx
    for t in range(3):
        tt = mx.array(t, dtype=mx.int32)
        re = fused(we, s.pos, cap, s.alive, tt, key)
        rc = comp(wc, s.pos, cap, s.alive, tt, key)
        mx.eval(re, rc)
        for x, y in zip(re, rc, strict=True):
            assert np.array_equal(to_numpy(x), to_numpy(y))
        we, wc = re[0], rc[0]


# --- resume equality with the weather on (spec §9.10) -------------------------------------------------------


_TIMING = {"wall_s", "steps_per_s", "peak_mem_gb"}  # the metrics keys that measure the host, not the run


def _stable(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k not in _TIMING} for r in rows]


def _resume_cfg(c: Config) -> Config:
    return apply_overrides(c, ["run.compile=false", "run.metrics_every=10", "run.checkpoint_every=20"])


def test_a_weather_run_resumes_bitwise(
    cfg: Config, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    c = _resume_cfg(cfg)
    rd = RunDir.create(tmp_path / "ref", c)
    res = runner.run(rd, c, init_state(c), 40, quiet=True)
    assert (res.status, res.exit_code) == ("done", runner.ExitCode.OK)
    full_digest = state_digest(res.state)
    full_wx = to_numpy(res.state.wx).copy()
    rows = _stable(rd.metrics_rows())
    assert [r["step"] for r in rows] == [10, 20, 30, 40] and {r["ledger_error"] for r in rows} == {0}
    del res

    # the checkpoint file carries wx of the right shape; the config.json round-trips the weather table
    assert RunDir.open(rd.path).config() == c
    ck = load_checkpoint(rd.checkpoint_at(20), c)
    assert tuple(ck.wx.shape) == (W.N_WX, c.weather.cells, c.weather.cells)
    assert json.loads((rd.path / "config.json").read_text())["weather"]["enabled"] is True

    # an interrupted run: 20 steps, then resume from the checkpoint to 40
    rd2 = RunDir.create(tmp_path / "split", c)
    r1 = runner.run(rd2, c, init_state(c), 20, quiet=True)
    del r1
    s = load_checkpoint(rd2.checkpoint_at(20), c)
    r2 = runner.run(rd2, c, s, 20, quiet=True)
    assert state_digest(r2.state) == full_digest and np.array_equal(to_numpy(r2.state.wx), full_wx)
    assert _stable(rd2.metrics_rows()) == rows
    del r2

    # the same through the CLI: truncate the first run back to its step-20 checkpoint and run 20 more steps
    assert cli.main(["resume", str(rd.path), "--checkpoint", "20", "--steps", "20"]) == 0
    capsys.readouterr()
    assert state_digest(load_checkpoint(rd.checkpoint_at(40), c)) == full_digest
    assert _stable(RunDir.open(rd.path).metrics_rows()) == rows
    assert cli.main(["inspect", str(rd.path), "--check", "--json"]) == 0


# --- Diag.WEATHER_WARN and the device check (spec §7.3) -----------------------------------------------------


def _ok_diag(c: Config) -> W.WeatherDiag:
    """The diagnostics of a calm rest-state step."""
    ws = W.unpack(W.rest_array(c))
    return W.advance(ws, mx.array(0, dtype=mx.int32), mx.random.key(0), c)[1]


def _force(clip_bad: bool = False) -> W.AgentForce:
    return W.AgentForce(mx.zeros((4, 3)), mx.zeros((4,)), None, mx.array(clip_bad))


@pytest.mark.parametrize(
    "field",
    ["n_limited", "n_clamp", "n_cond_limited", "n_relax_limited", "n_evap_limited", "n_safety"],
)
def test_health_fires_on_each_w0_gate_quantity(field: str) -> None:
    c = storm3d_config()
    wd = _ok_diag(c)
    assert not bool(W.health(wd, _force()))
    assert bool(W.health(wd._replace(**{field: wd.n_limited * 0 + 1}), _force()))


def test_health_fires_on_the_gates_and_on_a_clipped_wind() -> None:
    wd = _ok_diag(storm3d_config())
    one = wd.hyper_max * 0.0
    assert not bool(W.health(wd._replace(hyper_max=one + 0.79, eps_local_max=one + 0.99), _force()))
    assert bool(W.health(wd._replace(hyper_max=one + 0.8), _force()))
    assert bool(W.health(wd._replace(eps_local_max=one + 1.0), _force()))
    assert bool(W.health(wd, _force(clip_bad=True)))


def test_weather_warn_is_an_advisory_bit_outside_the_error_mask() -> None:
    assert Diag.WEATHER_WARN == 1 << 18 and not Diag.WEATHER_WARN & ERROR_MASK
    assert runner.diag_names(int(Diag.WEATHER_WARN | Diag.VCLIP_WARN)) == ["VCLIP_WARN", "WEATHER_WARN"]


def test_device_check_folds_the_weather_into_nonfinite_and_the_warn_bit(cfg: Config) -> None:
    s0 = init_state(cfg)
    s1, out = make_step(cfg, compile=False)(s0)
    mx.eval(s1, out)

    def check(s, weather_bad=None) -> int:
        zero = mx.array(0, dtype=mx.int32)
        bits = invariants.device_check(
            s, out.ledger, s0.total, zero, zero, zero, zero, cfg, weather_bad=weather_bad
        )
        return int(to_numpy(bits))

    assert check(s1) == 0
    assert check(s1, mx.array(False)) == 0
    assert check(s1, mx.array(True)) == Diag.WEATHER_WARN
    bad = np.array(to_numpy(s1.wx))
    bad[3, 5, 7] = np.nan  # a NaN in u2
    assert check(s1._replace(wx=mx.array(bad))) == Diag.NONFINITE
    off = _off(cfg)  # weather off: the stored array is not read (it is the carried rest state)
    s_off = init_state(off)._replace(wx=mx.array(bad))
    s_off1, out_off = make_step(off, compile=False)(s_off)
    zero = mx.array(0, dtype=mx.int32)
    assert (
        int(
            to_numpy(
                invariants.device_check(s_off1, out_off.ledger, s_off.total, zero, zero, zero, zero, off)
            )
        )
        == 0
    )


def test_a_clipped_wind_sets_the_warn_bit_in_the_step_and_it_is_sticky() -> None:
    c = storm3d_config()
    s = init_state(c)._replace(wx=fixed_wx(c, u1=0.5, v1=0.3))  # |u_h| = 0.58 > u_max away from the floor
    fn = make_step(c, compile=False)
    s, out = fn(s)
    mx.eval(s, out)
    assert int(to_numpy(s.diag)) & Diag.WEATHER_WARN and not int(to_numpy(s.diag)) & ERROR_MASK
    s2 = s._replace(wx=fixed_wx(c))
    for _ in range(3):
        s2, out = fn(s2)
    assert int(to_numpy(s2.diag)) & Diag.WEATHER_WARN  # sticky like every diag bit


def test_the_ledger_is_unaffected_by_the_wind(cfg: Config) -> None:
    """The wind moves agents, not energy: the ledger closes every step under a strong wind."""
    s = init_state(cfg)._replace(wx=fixed_wx(cfg, u1=0.15, v1=0.1, u2=-0.1))
    fn = make_step(cfg, compile=False)
    for _ in range(10):
        prev = s
        s, out = fn(s)
        mx.eval(s, out)
        led = to_numpy(out.ledger).astype(np.int64)
        assert invariants.full_check(s, cfg, prev, out) == []
        assert int(to_numpy(s.total)) - int(to_numpy(prev.total)) == ledger_net(led, cfg)
    assert int(to_numpy(s.diag)) & ERROR_MASK == 0
