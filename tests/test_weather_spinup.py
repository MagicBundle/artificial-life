"""The weather-only spin-up (coordinator choice C17; DESIGN §30): ``weather_spinup`` and ``alife run
--weather-spinup N``.

The chain is the engine's own (``weather.advance`` with ``step_keys``), so a chain started from the engine's
step and key reproduces the engine's ``wx`` bitwise; the spin-up proper runs on a separate key chain and ends
in the pools of the step the agents start at. CPU; the storms themselves need about 15,000 steps, so these
tests check the mechanics on short chains.
"""

from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import pytest

from alife import cli, invariants, weather_spinup
from alife import weather as W
from alife.config import Config
from alife.rng import InitStream, Stream, init_keys, step_keys
from alife.runio import RunDir, load_checkpoint
from alife.state import ERROR_MASK, init_state, to_numpy
from alife.step import make_step
from weather_helpers import storm2d_config, storm3d_config


@pytest.fixture(scope="module", params=["3d", "2d"])
def cfg(request: pytest.FixtureRequest) -> Config:
    return storm3d_config() if request.param == "3d" else storm2d_config()


def _state_key(cfg: Config) -> mx.array:
    return init_keys(cfg.run.seed)[InitStream.STATE_KEY]


def test_the_chain_is_the_engines_weather(cfg: Config) -> None:
    """Started from the engine's step 0 and state key, ``advance_chain`` gives the engine's ``wx`` at step 30
    bitwise (compiled and eager chains; seed 2 draws a disturbance at step 28, so the keys matter)."""
    c = storm3d_config("run.seed=2") if cfg.world.dims == 3 else storm2d_config("run.seed=2")
    s = init_state(c)
    fn = make_step(c, compile=False)
    for _ in range(30):
        s, out = fn(s)
        mx.eval(s, out)
    calm = weather_spinup.advance_chain(
        storm3d_config("run.seed=2", "weather.seed_rate=0.0")
        if cfg.world.dims == 3
        else storm2d_config("run.seed=2", "weather.seed_rate=0.0"),
        W.rest_array(c),
        0,
        _state_key(c),
        30,
    )[0]
    assert not np.array_equal(to_numpy(calm), to_numpy(s.wx))  # a disturbance did enter
    for compile in (True, False):
        wx, key = weather_spinup.advance_chain(c, W.rest_array(c), 0, _state_key(c), 30, compile=compile)
        assert np.array_equal(to_numpy(wx), to_numpy(s.wx)), compile
        assert np.array_equal(to_numpy(key), to_numpy(s.key)), compile  # the chain consumed the same keys


def test_the_chain_splits_cleanly(cfg: Config) -> None:
    """Chunks are an evaluation detail: 20 steps in one call are 7 + 13 in two, and the progress callback sees
    every chunk."""
    calls: list[tuple[int, int]] = []
    key = _state_key(cfg)
    a, ka = weather_spinup.advance_chain(
        cfg, W.rest_array(cfg), 0, key, 20, progress=lambda d, t: calls.append((d, t))
    )
    b, kb = weather_spinup.advance_chain(cfg, W.rest_array(cfg), 0, key, 7)
    b, kb = weather_spinup.advance_chain(cfg, b, 7, kb, 13)
    assert np.array_equal(to_numpy(a), to_numpy(b)) and np.array_equal(to_numpy(ka), to_numpy(kb))
    assert calls == [(16, 20), (20, 20)] and weather_spinup.CHUNK == 16


def test_spinup_ends_in_the_pools_of_the_start_step(cfg: Config) -> None:
    """``n = 0`` is the rest state the state would carry; for ``n > 0`` the weather has evolved, is
    deterministic, depends on the state key, and keeps the conserved column mass."""
    key = _state_key(cfg)
    rest = to_numpy(W.rest_array(cfg, 500))
    assert np.array_equal(to_numpy(weather_spinup.spinup(cfg, 0, step=500, key=key)), rest)
    wx = to_numpy(weather_spinup.spinup(cfg, 60, step=500, key=key))
    assert wx.shape == rest.shape and np.isfinite(wx).all() and not np.array_equal(wx, rest)
    assert np.array_equal(wx, to_numpy(weather_spinup.spinup(cfg, 60, step=500, key=key)))
    other = to_numpy(weather_spinup.spinup(cfg, 60, step=500, key=init_keys(7)[InitStream.STATE_KEY]))
    assert not np.array_equal(wx, other)
    p = W.params_from_config(cfg)
    assert abs((wx[0] + wx[1]).sum() / (p.n * p.n * (p.H1 + p.H2)) - 1.0) < 1e-5
    assert np.abs(wx[2]).max() > 0  # the winds are no longer the rest state's zeros


def test_the_start_step_sets_the_pool_phase_not_the_key(cfg: Config) -> None:
    """Two spin-ups of the same length ending at different steps start under different pools, so they
    differ; a spin-up from before step 0 (negative times) is the same code."""
    key = _state_key(cfg)
    a = to_numpy(weather_spinup.spinup(cfg, 40, step=40, key=key))  # steps 0 .. 39
    b = to_numpy(weather_spinup.spinup(cfg, 40, step=5000, key=key))
    assert not np.array_equal(a, b)
    assert np.array_equal(
        a,
        to_numpy(
            weather_spinup.advance_chain(cfg, W.rest_array(cfg, 0), 0, weather_spinup.spinup_key(key), 40)[0]
        ),
    )
    c = to_numpy(weather_spinup.spinup(cfg, 40, step=0, key=key))  # steps -40 .. -1
    assert np.isfinite(c).all() and not np.array_equal(c, to_numpy(W.rest_array(cfg, 0)))


def _chain_keys(key: mx.array, steps: int) -> set[bytes]:
    """Every key the engine's chain makes from ``key`` over ``steps`` steps: the chain key itself, and at each
    step the next chain key, the 16 stream keys of ``step_keys`` and the two halves of the weather split."""
    out = {to_numpy(key).tobytes()}
    for _ in range(steps):
        key, ks = step_keys(key)
        out |= {to_numpy(k).tobytes() for k in (key, *ks, *mx.random.split(ks[Stream.WEATHER], 2))}
    return out


def test_the_spinup_key_chain_is_not_the_engines(cfg: Config) -> None:
    """The engine takes ``split(key, 2)`` and then ``split(child, 16)`` from the state key (``step_keys``);
    the spin-up chain starts at ``split(key, 3)[2]``. Over 50 steps and 3 seeds the chains share no key."""
    for seed in (0, 1, 7):
        key = init_keys(seed)[InitStream.STATE_KEY]
        spin = weather_spinup.spinup_key(key)
        assert not np.array_equal(to_numpy(spin), to_numpy(key))
        mine, theirs = _chain_keys(spin, 50), _chain_keys(key, 50)
        assert len(mine) == len(theirs) == 1 + 50 * 19  # the chain key and 19 distinct keys per step
        assert not mine & theirs, seed


def test_a_state_with_a_spun_up_weather_steps_cleanly() -> None:
    """The engine continues from the spun-up array at the step it was built for: the full check passes, the
    ledger closes and the winds that the agents feel are the weather's."""
    c = storm3d_config()
    s = init_state(c)
    wx = weather_spinup.spinup(c, 80, step=int(to_numpy(s.step)), key=s.key)
    s = s._replace(wx=wx)
    assert invariants.full_check(s, c) == []
    fn = make_step(c)
    for _ in range(3):
        s, out = fn(s)
        mx.eval(s, out)
    assert invariants.full_check(s, c) == [] and not int(to_numpy(s.diag)) & ERROR_MASK


def test_errors() -> None:
    c = storm2d_config()
    with pytest.raises(ValueError, match="n_steps"):
        weather_spinup.advance_chain(c, W.rest_array(c), 0, _state_key(c), -1)
    off = storm2d_config("weather.enabled=false", "weather.u_max=0.3")
    with pytest.raises(ValueError, match=r"weather\.enabled"):
        weather_spinup.spinup(off, 3, step=0, key=_state_key(off))


# --- the command line ---------------------------------------------------------------------------------------

STORM2D = ("world.size=320", "weather.enabled=true", "weather.u_max=0.2", "brain.weight_dtype=float32")


def _args(out, *extra: str, steps: int = 2) -> list[str]:
    sets = [x for kv in STORM2D for x in ("--set", kv)]
    cadence = ["--metrics-every", "1", "--checkpoint-every", "2"]
    return [
        "run",
        "--preset",
        "tiny",
        *sets,
        "--steps",
        str(steps),
        *cadence,
        "--quiet",
        "--out",
        str(out),
        *extra,
    ]


def _config_of(out) -> Config:
    return RunDir.open(out).config()


def test_run_with_a_weather_spinup_starts_the_agents_on_the_spun_up_weather(tmp_path) -> None:
    out = tmp_path / "run"
    assert cli.main(_args(out, "--weather-spinup", "30")) == 0
    rd = RunDir.open(out)
    assert rd.meta()["weather_spinup"] == 30
    c = _config_of(out)
    start = load_checkpoint(rd.path / "checkpoints" / "step_0000000000.safetensors", c)
    s0 = init_state(c)
    want = weather_spinup.spinup(c, 30, step=0, key=s0.key)
    assert np.array_equal(to_numpy(start.wx), to_numpy(want))
    assert not np.array_equal(to_numpy(start.wx), to_numpy(W.rest_array(c)))
    # without the flag: the calm start, and no meta entry
    plain = tmp_path / "plain"
    assert cli.main(_args(plain)) == 0
    assert "weather_spinup" not in RunDir.open(plain).meta()
    calm = load_checkpoint(plain / "checkpoints" / "step_0000000000.safetensors", _config_of(plain))
    assert np.array_equal(to_numpy(calm.wx), to_numpy(W.rest_array(c)))


def test_run_spinup_zero_is_the_calm_start(tmp_path) -> None:
    out = tmp_path / "run"
    assert cli.main(_args(out, "--weather-spinup", "0")) == 0
    c = _config_of(out)
    start = load_checkpoint(out / "checkpoints" / "step_0000000000.safetensors", c)
    assert np.array_equal(to_numpy(start.wx), to_numpy(W.rest_array(c)))


def test_the_resumed_run_keeps_the_spun_up_weather(tmp_path) -> None:
    """The spin-up is in the checkpoint: a resume continues from it as an uninterrupted run would."""
    a, b = tmp_path / "a", tmp_path / "b"
    assert cli.main(_args(a, "--weather-spinup", "25", steps=4)) == 0
    assert cli.main(_args(b, "--weather-spinup", "25", steps=2)) == 0
    assert cli.main(["resume", str(b), "--steps", "2", "--quiet"]) == 0
    ca, cb = _config_of(a), _config_of(b)
    sa = load_checkpoint(a / "checkpoints" / "step_0000000004.safetensors", ca)
    sb = load_checkpoint(b / "checkpoints" / "step_0000000004.safetensors", cb)
    assert np.array_equal(to_numpy(sa.wx), to_numpy(sb.wx))
    assert np.array_equal(to_numpy(sa.pos), to_numpy(sb.pos))


def test_spinup_needs_the_weather_on_and_writes_nothing_otherwise(tmp_path, capsys) -> None:
    out = tmp_path / "off"
    code = cli.main(
        ["run", "--preset", "tiny", "--steps", "2", "--quiet", "--out", str(out), "--weather-spinup", "5"]
    )
    assert code == 2 and not out.exists()
    assert "--weather-spinup needs weather.enabled" in capsys.readouterr().err
    assert cli.main(["run", "--preset", "tiny", "--weather-spinup", "-1"]) == 2  # argparse: not negative


def test_a_fork_spins_up_under_the_pools_of_its_own_step(tmp_path, capsys) -> None:
    """Fork a weather-off run with the weather on: the new run's start checkpoint holds the spin-up that ends
    in the pools of the fork step. A source that already has the weather on is refused."""
    src = tmp_path / "src"
    args = ["run", "--preset", "tiny", "--set", "world.size=320", "--set", "brain.weight_dtype=float32"]
    args += ["--steps", "6", "--metrics-every", "1", "--checkpoint-every", "3", "--quiet", "--out", str(src)]
    assert cli.main(args) == 0
    out = tmp_path / "fork"
    fork = ["run", "--init-from", f"{src}:3", "--set", "weather.enabled=true", "--set", "weather.u_max=0.2"]
    fork += ["--steps", "2", "--metrics-every", "1", "--checkpoint-every", "2", "--quiet", "--out", str(out)]
    fork += ["--weather-spinup", "20"]
    assert cli.main(fork) == 0
    c = _config_of(out)
    rd = RunDir.open(out)
    assert rd.meta()["weather_spinup"] == 20 and rd.meta()["start_step"] == 3
    start = load_checkpoint(out / "checkpoints" / "step_0000000003.safetensors", c)
    want = weather_spinup.spinup(c, 20, step=3, key=start.key)
    assert np.array_equal(to_numpy(start.wx), to_numpy(want))
    # the pool phase is the fork step's: the spin-up is not the one that ends at step 0
    assert not np.array_equal(
        to_numpy(start.wx), to_numpy(weather_spinup.spinup(c, 20, step=0, key=start.key))
    )
    capsys.readouterr()
    again = tmp_path / "again"
    code = cli.main(["run", "--init-from", f"{out}:3", "--steps", "2", "--quiet", "--out", str(again),
                     "--weather-spinup", "5"])  # fmt: skip
    assert code == 2 and not again.exists()
    assert "already has an evolving weather state" in capsys.readouterr().err
    meta = json.loads((out / "meta.json").read_text())
    assert meta["weather_spinup"] == 20


# --- the cache of spun-up states (S4c) ---


@pytest.fixture
def cache_in(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ALIFE_CACHE_DIR", str(tmp_path))
    return tmp_path / "weather_spinup"


def _files(d) -> list:
    return sorted(d.glob("spinup_*.npz")) if d.exists() else []


def test_a_cached_spinup_is_read_back_bitwise_and_skips_the_chain(
    cfg: Config, cache_in, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = _state_key(cfg)
    calls: list[tuple[int, int]] = []
    first = to_numpy(
        weather_spinup.spinup(
            cfg, 40, step=100, key=key, cache=True, progress=lambda d, t: calls.append((d, t))
        )
    )
    assert len(_files(cache_in)) == 1 and calls[-1] == (40, 40)
    plain = to_numpy(weather_spinup.spinup(cfg, 40, step=100, key=key))
    assert np.array_equal(first, plain)  # what is stored is what the chain makes

    def boom(*a, **k):
        raise AssertionError("the chain ran on a cache hit")

    monkeypatch.setattr(weather_spinup, "advance_chain", boom)
    calls.clear()
    again = to_numpy(
        weather_spinup.spinup(
            cfg, 40, step=100, key=key, cache=True, progress=lambda d, t: calls.append((d, t))
        )
    )
    assert np.array_equal(again, first) and again.dtype == np.float32 and calls == [(40, 40)]


def test_every_input_of_the_chain_is_in_the_key(cfg: Config, cache_in) -> None:
    import dataclasses

    key = _state_key(cfg)
    base = dict(step=100, key=key, cache=True)
    weather_spinup.spinup(cfg, 20, **base)
    variants = [
        (cfg, 20, {**base, "step": 101}),  # the pools' phase
        (cfg, 21, base),  # the length
        (cfg, 20, {**base, "key": init_keys(9)[InitStream.STATE_KEY]}),  # the state key
        (cfg, 20, {**base, "compile": False}),  # compiled and eager chains
        (dataclasses.replace(cfg, weather=dataclasses.replace(cfg.weather, seed_rate=0.031)), 20, base),
    ]
    for i, (c, n, kw) in enumerate(variants, start=2):
        weather_spinup.spinup(c, n, **kw)
        assert len(_files(cache_in)) == i, i
    # the run section is not an input: a different seed or step count of the run reuses the entry
    other = dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, steps=7, seed=3))
    weather_spinup.spinup(other, 20, **base)
    assert len(_files(cache_in)) == len(variants) + 1


def test_a_damaged_or_foreign_entry_is_a_miss_and_is_replaced(cfg: Config, cache_in) -> None:
    key = _state_key(cfg)
    good = to_numpy(weather_spinup.spinup(cfg, 30, step=50, key=key, cache=True))
    (path,) = _files(cache_in)
    path.write_bytes(b"not an npz")
    assert np.array_equal(to_numpy(weather_spinup.spinup(cfg, 30, step=50, key=key, cache=True)), good)
    with np.load(path) as z:  # rewritten by the miss
        assert z["wx"].shape == good.shape
    text = weather_spinup.cache_key(cfg, 30, step=50, key=key, compile=True)
    np.savez(path, wx=np.zeros_like(good), key=np.array("another key"))  # a key that does not match
    assert weather_spinup.cache_load(text, mx.array(good)) is None
    np.savez(path, wx=np.full(good.shape, np.nan, np.float32), key=np.array(text))  # not finite
    assert weather_spinup.cache_load(text, mx.array(good)) is None
    np.savez(path, wx=good[:, :3], key=np.array(text))  # wrong shape
    assert weather_spinup.cache_load(text, mx.array(good)) is None
    assert np.array_equal(to_numpy(weather_spinup.spinup(cfg, 30, step=50, key=key, cache=True)), good)


def test_the_cache_is_off_by_default_ignores_a_failed_write_and_zero_steps(
    cfg: Config, cache_in, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    key = _state_key(cfg)
    weather_spinup.spinup(cfg, 20, step=10, key=key)
    weather_spinup.spinup(cfg, 0, step=10, key=key, cache=True)
    assert _files(cache_in) == []  # no flag: nothing written; zero steps: nothing to keep
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setenv("ALIFE_CACHE_DIR", str(blocker / "sub"))  # not a directory: the write fails quietly
    wx = to_numpy(weather_spinup.spinup(cfg, 20, step=10, key=key, cache=True))
    assert np.array_equal(wx, to_numpy(weather_spinup.spinup(cfg, 20, step=10, key=key)))
    monkeypatch.setenv("ALIFE_SPINUP_CACHE", "0")
    assert not weather_spinup.cache_enabled()
    monkeypatch.delenv("ALIFE_SPINUP_CACHE")
    assert weather_spinup.cache_enabled()


def test_an_edit_of_the_weather_sources_changes_the_key(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    key = _state_key(cfg)
    a = weather_spinup.cache_key(cfg, 10, step=0, key=key, compile=True)
    monkeypatch.setattr(weather_spinup, "_code_digest", lambda: "edited")
    assert weather_spinup.cache_key(cfg, 10, step=0, key=key, compile=True) != a


def test_the_command_line_caches_a_spinup_unless_switched_off(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``alife run --weather-spinup N`` writes the spin-up to the cache and a second launch of the same world
    starts from the entry, with the same start checkpoint; ``ALIFE_SPINUP_CACHE=0`` writes nothing."""
    monkeypatch.setenv("ALIFE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("ALIFE_SPINUP_CACHE")  # the suite switches it off; this test is of the default
    a, b = tmp_path / "a", tmp_path / "b"
    assert cli.main(_args(a, "--weather-spinup", "30")) == 0
    assert len(_files(tmp_path / "cache" / "weather_spinup")) == 1
    monkeypatch.setattr(weather_spinup, "advance_chain", lambda *x, **k: pytest.fail("the chain ran"))
    assert cli.main(_args(b, "--weather-spinup", "30")) == 0
    sa = load_checkpoint(a / "checkpoints" / "step_0000000000.safetensors", _config_of(a))
    sb = load_checkpoint(b / "checkpoints" / "step_0000000000.safetensors", _config_of(b))
    assert np.array_equal(to_numpy(sa.wx), to_numpy(sb.wx))
    monkeypatch.undo()
    monkeypatch.setenv("ALIFE_CACHE_DIR", str(tmp_path / "off"))
    monkeypatch.setenv("ALIFE_SPINUP_CACHE", "0")
    assert cli.main(_args(tmp_path / "c", "--weather-spinup", "30")) == 0
    assert _files(tmp_path / "off" / "weather_spinup") == []


def test_a_prebuilt_step_makes_the_same_chain_in_many_short_calls(cfg: Config) -> None:
    """``chain_step`` is the step ``advance_chain`` builds per call: built once and passed in, five calls of
    seven steps are the one call of thirty-five, bitwise (the study drivers tick every 25 steps)."""
    key = _state_key(cfg)
    one, ko = weather_spinup.advance_chain(cfg, W.rest_array(cfg), 0, key, 35)
    fn = weather_spinup.chain_step(cfg)
    wx, k = W.rest_array(cfg), key
    for i in range(5):
        wx, k = weather_spinup.advance_chain(cfg, wx, 7 * i, k, 7, step_fn=fn)
    assert np.array_equal(to_numpy(wx), to_numpy(one)) and np.array_equal(to_numpy(k), to_numpy(ko))
    with pytest.raises(ValueError, match=r"weather\.enabled"):
        weather_spinup.chain_step(storm3d_config("weather.enabled=false"))
