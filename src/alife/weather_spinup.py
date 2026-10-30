"""Weather-only spin-up: advance the storm weather alone, before any agent feels it (coordinator choice C17).

A run that turns the weather on starts from the calm rest state, and storms need about 15,000 steps to mature
(DESIGN §30; STORMS_W0 §0). Because the weather is agent-independent in W1, its trajectory is a function of
the config, the key chain and the step alone, so those steps can be run without agents: ``alife run
--weather-spinup N`` (a fresh run or a ``--init-from`` fork) advances the weather ``N`` steps on its own and
starts the agents on the result.

Host-side driver (not a simulation module): it evaluates the weather every few steps. The update it applies
is ``weather.advance``, the function the engine step calls, so a chain started from the engine's own step and
key reproduces the engine's ``wx`` bitwise (``tests/test_weather_spinup.py``).

Time. The weather at engine step ``t`` is ``advance(wx, t, key_t)`` with the warm pools of step ``t``. The
spin-up for a state at step ``S`` runs the steps ``S - N .. S - 1`` (``S - N`` may be negative: the pools
simply sit where they were ``N`` steps earlier), starting from the rest state of the pools at step ``S - N``.
It therefore ends consistent with the pools of step ``S``, the first step the agents feel, and a fork's
pool-phase transient (DESIGN §30) does not arise. The keys come from a chain derived from the state key by a
third split that no engine stream uses, so the disturbances of the spin-up repeat none of the draws the engine
will make from the state's own key. The spun-up weather is thus a function of (config, state key, N, S).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
import numpy as np

from alife import weather
from alife.config import Config, config_to_dict
from alife.rng import Stream, step_keys

CHUNK = 16  # weather steps per evaluation: enough to hide dispatch latency, small enough to bound the graph
# The spin-up, in engine steps, that the launcher of a planet asks for (``alife play --init-from ...
# --weather-spinup N``), per sphere preset: the length at which the kinetic energy, the humidity and the storm
# count of that planet's weather have settled, times 1.3 (DESIGN §46.6, ``alife.analysis.w0_sphere``).
# ``PlanetSpec.sphereSpinups`` in ``gui/Sources/AlifeCore/PlayLaunch.swift`` is the launcher's copy;
# ``tests/test_gui_sync.py`` keeps the two equal.
SPHERE_SPINUP: dict[str, int] = {"64k3s": 15_000, "tiny3s": 10_000, "mini3s": 8_000}
CACHE_VERSION = 1  # the format of a cache entry and of its key (bump it when either changes)
CACHE_OFF_ENV = "ALIFE_SPINUP_CACHE"  # set to "0" to switch the cache of the command line off


def spinup_key(key: mx.array) -> mx.array:
    """The start of the spin-up key chain, from the state key. The engine takes ``split(key, 2)`` and then
    ``split(child, 16)`` from ``key`` (``rng.step_keys``); a split into three is a different draw, so the
    chain is separate (tests: no key in common over 50 steps)."""
    return mx.random.split(key, 3)[2]


StepFn = Callable[[mx.array, mx.array, mx.array], tuple[mx.array, mx.array, mx.array]]


def chain_step(cfg: Config, *, compile: bool = True) -> StepFn:
    """One step of the chain, ``(wx, t, key) -> (wx, t + 1, key_next)``: at each step
    ``key, ks = step_keys(key)`` and the disturbance key is ``split(ks[WEATHER], 2)[0]``, the engine's.
    Compiled once (``advance_chain`` builds it per call; a caller that advances in many short pieces builds
    it once and passes it)."""
    if not cfg.weather.enabled:
        raise ValueError("weather_spinup needs weather.enabled = true")

    def one(wx: mx.array, t: mx.array, key: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        key_next, ks = step_keys(key)
        ws, _ = weather.advance(weather.unpack(wx), t, mx.random.split(ks[Stream.WEATHER], 2)[0], cfg)
        return weather.pack(ws), t + 1, key_next

    return mx.compile(one) if compile else one


def advance_chain(
    cfg: Config,
    wx: mx.array,
    t0: int,
    key: mx.array,
    n_steps: int,
    *,
    compile: bool = True,
    progress: Callable[[int, int], None] | None = None,
    step_fn: StepFn | None = None,
) -> tuple[mx.array, mx.array]:
    """Advance the packed weather ``wx`` through the steps ``t0 .. t0 + n_steps - 1`` with the engine's key
    chain (``chain_step``). Returns ``(wx, key)`` after the last step, evaluated. With the state's key and
    step this reproduces the engine's ``wx`` bitwise. ``progress(done, total)`` is called after each
    evaluated chunk. ``step_fn`` is a ``chain_step`` built by the caller (``compile`` is then unused)."""
    if n_steps < 0:
        raise ValueError(f"n_steps must be >= 0, got {n_steps}")
    fn = chain_step(cfg, compile=compile) if step_fn is None else step_fn
    t = mx.array(t0, dtype=mx.int32)
    done = 0
    while done < n_steps:
        for _ in range(min(CHUNK, n_steps - done)):
            wx, t, key = fn(wx, t, key)
            done += 1
        mx.eval(wx, t, key)
        if progress is not None:
            progress(done, n_steps)
    mx.eval(wx, key)
    return wx, key


# --- the cache of spun-up states (S4c) ---------------------------------------------------------------------
#
# A spin-up is a pure function of its inputs: the configuration (the weather and the world it runs on), the
# state key, the step the agents start at (the pools' phase), the length and the compiled or eager chain, on
# one device and one build of the weather code. The result of a long one (20 000 steps of a 64k3s planet are a
# minute on a quiet GPU, many on a busy one) is kept in a file keyed by all of those, so that launching the
# same world again (the same source checkpoint, overlay and length) costs a file read. The entry carries its
# key in clear and a load checks it, the shape, the dtype and finiteness; any failure is a miss, and a failed
# write is ignored (the cache is an optimisation, never a dependency).


def cache_enabled() -> bool:
    """The command line uses the cache unless ``ALIFE_SPINUP_CACHE=0``."""
    return os.environ.get(CACHE_OFF_ENV, "1") != "0"


def cache_dir() -> Path:
    """``$ALIFE_CACHE_DIR/weather_spinup`` or ``~/.cache/alife/weather_spinup`` (the convention of
    ``geom._disk_cache_dir``)."""
    base = os.environ.get("ALIFE_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "alife")
    return Path(base) / "weather_spinup"


def _code_digest() -> str:
    """The digest of the sources a spin-up depends on, so that an edit of the weather code never reads an
    entry the old code wrote."""
    here = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for name in ("weather.py", "weather_sph.py", "weather_tab.py", "weather_spinup.py", "geom.py", "rng.py"):
        h.update(name.encode())
        h.update((here / name).read_bytes())
    return h.hexdigest()


def cache_key(cfg: Config, n_steps: int, *, step: int, key: mx.array, compile: bool) -> str:
    """The key of a spin-up, as the canonical text hashed into the file name: the whole configuration but
    ``run`` (which no step of the chain reads), ``n_steps``, ``step``, the bytes of the state ``key``, the
    compile flag, the MLX version and device, the digest of the weather sources and the cache version."""
    conf = config_to_dict(cfg)
    conf.pop("run", None)
    blob = {
        "cache": CACHE_VERSION,
        "config": conf,
        "n": int(n_steps),
        "step": int(step),
        "key": [int(v) for v in np.array(key).reshape(-1)],
        "compile": bool(compile),
        "mlx": mx.__version__,
        "device": str(mx.default_device()),
        "code": _code_digest(),
    }
    return json.dumps(blob, sort_keys=True, separators=(",", ":"))


def _entry(text: str) -> Path:
    return cache_dir() / f"spinup_{hashlib.sha256(text.encode()).hexdigest()[:40]}.npz"


def cache_load(text: str, like: mx.array) -> mx.array | None:
    """The cached spin-up for the key ``text`` (shaped and typed like ``like``), or None on any mismatch."""
    try:
        with np.load(_entry(text), allow_pickle=False) as z:
            if str(z["key"]) != text:
                return None
            wx = z["wx"]
    except (OSError, ValueError, KeyError):
        return None
    if wx.shape != tuple(like.shape) or wx.dtype != np.float32 or not np.isfinite(wx).all():
        return None
    return mx.array(wx)


def cache_store(text: str, wx: mx.array) -> None:
    """Write the entry atomically (a temporary file, then a rename); a failure is silent."""
    path = _entry(text)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
        np.savez(tmp, wx=np.array(wx, dtype=np.float32), key=np.array(text))
        os.replace(tmp, path)
    except OSError:
        pass


def spinup(
    cfg: Config,
    n_steps: int,
    *,
    step: int,
    key: mx.array,
    compile: bool = True,
    progress: Callable[[int, int], None] | None = None,
    cache: bool = False,
) -> mx.array:
    """The packed weather after ``n_steps`` weather-only steps, for agents that start at engine step
    ``step`` with state key ``key``: the rest state of the pools of step ``step - n_steps``, advanced
    through the steps ``step - n_steps .. step - 1`` on the chain ``spinup_key(key)``. ``n_steps = 0`` is
    the rest state of the pools of ``step``, which is what the state already carries. With ``cache`` the
    result is read from, and written to, the cache of spun-up states (``cache_key``); ``progress`` then sees
    one call ``(n_steps, n_steps)`` on a hit."""
    start = step - n_steps
    wx = weather.rest_array(cfg, start)
    text = None
    if cache and n_steps > 0:
        text = cache_key(cfg, n_steps, step=step, key=key, compile=compile)
        hit = cache_load(text, wx)
        if hit is not None:
            if progress is not None:
                progress(n_steps, n_steps)
            return hit
    wx, _ = advance_chain(cfg, wx, start, spinup_key(key), n_steps, compile=compile, progress=progress)
    if text is not None:
        cache_store(text, wx)
    return wx
