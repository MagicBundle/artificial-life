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
        (
            dataclasses.replace(cfg, world=dataclasses.replace(cfg.world, height=cfg.world.height + 4.0)),
            20,
            base,
        ),
    ]
    for i, (c, n, kw) in enumerate(variants, start=2):
        weather_spinup.spinup(c, n, **kw)
        assert len(_files(cache_in)) == i, i
    # the run section is not an input: a different seed or step count of the run reuses the entry
    other = dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, steps=7, seed=3))
    weather_spinup.spinup(other, 20, **base)
    assert len(_files(cache_in)) == len(variants) + 1


def test_the_build_and_the_device_and_the_format_are_in_the_key(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The MLX version, the device and the cache version each change the key text (a spin-up of another build,
    another device or another entry format is another entry)."""
    key = _state_key(cfg)
    a = weather_spinup.cache_key(cfg, 10, step=0, key=key, compile=True)
    assert weather_spinup.cache_key(cfg, 10, step=0, key=key, compile=True) == a
    monkeypatch.setattr(mx, "__version__", "0.0.0-other")
    assert weather_spinup.cache_key(cfg, 10, step=0, key=key, compile=True) != a
    monkeypatch.undo()
    monkeypatch.setattr(mx, "default_device", lambda: "another device")
    assert weather_spinup.cache_key(cfg, 10, step=0, key=key, compile=True) != a
    monkeypatch.undo()
    monkeypatch.setattr(weather_spinup, "CACHE_VERSION", weather_spinup.CACHE_VERSION + 1)
    assert weather_spinup.cache_key(cfg, 10, step=0, key=key, compile=True) != a


def test_each_digested_source_changes_the_digest_and_no_other_file_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path

    base = weather_spinup._code_digest()
    real = Path.read_bytes
    pkg = Path(weather_spinup.__file__).resolve().parent
    for name in weather_spinup.DIGESTED_SOURCES:
        assert (pkg / name).is_file(), name

        def edited(self: Path, name: str = name) -> bytes:
            return real(self) + (b"#" if self.name == name else b"")

        monkeypatch.setattr(Path, "read_bytes", edited)
        assert weather_spinup._code_digest() != base, name
        monkeypatch.undo()
    assert len(set(weather_spinup.DIGESTED_SOURCES)) == len(weather_spinup.DIGESTED_SOURCES)
    assert weather_spinup._code_digest() == base


def _executed_sources(c: Config) -> set[str]:
    """The files under ``alife/`` whose functions run in a short eager spin-up of ``c``."""
    import sys

    seen: set[str] = set()

    def prof(frame, event, arg) -> None:
        if event == "call":
            f = frame.f_code.co_filename.replace("\\", "/")
            if "/src/alife/" in f:
                seen.add(f.split("/src/alife/")[1])

    sys.setprofile(prof)
    try:
        mx.eval(weather_spinup.spinup(c, 3, step=10, key=_state_key(c), compile=False))
    finally:
        sys.setprofile(None)
    return seen


def test_every_source_a_spinup_executes_is_in_the_digest(cfg: Config) -> None:
    """The digest must cover the code the chain runs: trace a spin-up of the box (this fixture, 2D and 3D) and
    of the sphere and fail on an executed file that is not in ``DIGESTED_SOURCES``."""
    from alife.config import load_config

    sphere = load_config(preset="tiny3s", overlays=["stormsph16"])
    seeds = load_config(preset="tiny3s", overlays=["stormsph16"], overrides=["weather.seed_gain=0.03"])
    for c in (cfg, sphere, seeds):
        missing = _executed_sources(c) - set(weather_spinup.DIGESTED_SOURCES)
        assert not missing, f"executed by a spin-up but not in the digest: {sorted(missing)}"


def test_the_spinup_reads_only_the_keyed_sections_and_no_other_section_matters(cfg: Config) -> None:
    """The key holds the sections of ``KEYED_SECTIONS`` only: a spy on the configuration shows that the chain
    reads no other section (box 2D and 3D, sphere), and a configuration with a number of every other section
    changed has the same key and the same spin-up, bit for bit."""
    import dataclasses

    from alife.config import load_config

    class Spy:
        def __init__(self, c: Config, seen: set[str]) -> None:
            self._c, self._seen = c, seen

        def __getattr__(self, name: str):
            self._seen.add(name)
            return getattr(self._c, name)

    sphere = load_config(preset="tiny3s", overlays=["stormsph16"])
    for c in (cfg, sphere):
        seen: set[str] = set()
        mx.eval(weather_spinup.spinup(Spy(c, seen), 3, step=10, key=_state_key(c), compile=False))
        assert seen <= set(weather_spinup.KEYED_SECTIONS), sorted(seen)
        changes = {}
        for f in dataclasses.fields(Config):
            if f.name in weather_spinup.KEYED_SECTIONS:
                continue
            sec = getattr(c, f.name)
            num = next(
                (g.name for g in dataclasses.fields(sec) if type(getattr(sec, g.name)) in (int, float)), None
            )
            if num is not None:
                changes[f.name] = dataclasses.replace(sec, **{num: getattr(sec, num) + 1})
        assert len(changes) >= 8, sorted(changes)  # director, population, brain, run, ... all changed
        other = dataclasses.replace(c, **changes)
        key = _state_key(c)
        k0 = weather_spinup.cache_key(c, 10, step=0, key=key, compile=False)
        assert weather_spinup.cache_key(other, 10, step=0, key=key, compile=False) == k0
        a = to_numpy(weather_spinup.spinup(c, 5, step=10, key=key, compile=False))
        b = to_numpy(weather_spinup.spinup(other, 5, step=10, key=key, compile=False))
        assert np.array_equal(a, b)


def test_a_damaged_or_foreign_entry_is_a_miss_and_is_replaced(cfg: Config, cache_in) -> None:
    key = _state_key(cfg)
    good = to_numpy(weather_spinup.spinup(cfg, 30, step=50, key=key, cache=True))
    (path,) = _files(cache_in)
    data = path.read_bytes()
    damaged = {
        "garbage": b"not an npz",
        "zero bytes": b"",
        "truncated zip": data[: len(data) // 2],
        "truncated by ten bytes": data[:-10],
        "zip magic only": b"PK\x03\x04",
    }
    text = weather_spinup.cache_key(cfg, 30, step=50, key=key, compile=True)
    for name, blob in damaged.items():
        path.write_bytes(blob)
        assert weather_spinup.cache_load(text, mx.array(good)) is None, name  # a miss, not an exception
        assert not path.exists(), name  # and the bad entry is removed
        path.write_bytes(blob)
        again = to_numpy(weather_spinup.spinup(cfg, 30, step=50, key=key, cache=True))
        assert np.array_equal(again, good), name
        with np.load(path) as z:  # rewritten by the miss
            assert z["wx"].shape == good.shape, name
    np.savez(path, wx=np.zeros_like(good), key=np.array("another key"))  # a key that does not match
    assert weather_spinup.cache_load(text, mx.array(good)) is None
    np.savez(path, wx=np.full(good.shape, np.nan, np.float32), key=np.array(text))  # not finite
    assert weather_spinup.cache_load(text, mx.array(good)) is None
    np.savez(path, wx=good[:, :3], key=np.array(text))  # wrong shape
    assert weather_spinup.cache_load(text, mx.array(good)) is None
    assert np.array_equal(to_numpy(weather_spinup.spinup(cfg, 30, step=50, key=key, cache=True)), good)


def test_the_cache_folder_is_capped_by_use_and_its_temporaries_are_removed(
    cfg: Config, cache_in, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import time

    monkeypatch.setattr(weather_spinup, "CACHE_MAX_ENTRIES", 3)
    key = _state_key(cfg)
    stale = cache_in / "spinup_dead.99999.tmp.npz"
    cache_in.mkdir(parents=True)
    stale.write_bytes(b"x")
    old = time.time() - 2 * weather_spinup.CACHE_TMP_MAX_AGE
    os.utime(stale, (old, old))
    fresh = cache_in / "spinup_live.99998.tmp.npz"  # a write in progress elsewhere: left alone
    fresh.write_bytes(b"x")
    for n in range(5):  # five entries, oldest first
        weather_spinup.spinup(cfg, 10 + n, step=100, key=key, cache=True)
        for f in _files(cache_in):
            if f.stat().st_mtime > time.time() - 30:
                os.utime(f, (time.time() - 40 + n, time.time() - 40 + n))

    def entries() -> list:
        return [f for f in _files(cache_in) if not f.name.endswith(".tmp.npz")]

    kept = entries()
    assert len(kept) <= 3 and not stale.exists() and fresh.exists()
    # a use refreshes an entry: load the oldest kept one, store a new one, and it survives
    oldest = min(kept, key=lambda f: f.stat().st_mtime)
    text = next(
        weather_spinup.cache_key(cfg, 10 + n, step=100, key=key, compile=True)
        for n in range(5)
        if weather_spinup._entry(weather_spinup.cache_key(cfg, 10 + n, step=100, key=key, compile=True))
        == oldest
    )
    assert weather_spinup.cache_load(text, W.rest_array(cfg, 100)) is not None
    weather_spinup.spinup(cfg, 30, step=100, key=key, cache=True)
    assert oldest.exists() and len(entries()) <= 3
    # a failed write leaves no temporary
    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    weather_spinup.spinup(cfg, 31, step=100, key=key, cache=True)
    assert [f for f in cache_in.iterdir() if f.name.endswith(".tmp.npz") and f != fresh] == []


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
