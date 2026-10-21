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

from collections.abc import Callable

import mlx.core as mx

from alife import weather
from alife.config import Config
from alife.rng import Stream, step_keys

CHUNK = 16  # weather steps per evaluation: enough to hide dispatch latency, small enough to bound the graph


def spinup_key(key: mx.array) -> mx.array:
    """The start of the spin-up key chain, from the state key. The engine takes ``split(key, 2)`` and then
    ``split(child, 16)`` from ``key`` (``rng.step_keys``); a split into three is a different draw, so the
    chain is separate (tests: no key in common over 50 steps)."""
    return mx.random.split(key, 3)[2]


def advance_chain(
    cfg: Config,
    wx: mx.array,
    t0: int,
    key: mx.array,
    n_steps: int,
    *,
    compile: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[mx.array, mx.array]:
    """Advance the packed weather ``wx`` through the steps ``t0 .. t0 + n_steps - 1`` with the engine's key
    chain: at each step ``key, ks = step_keys(key)`` and the disturbance key is ``split(ks[WEATHER], 2)[0]``.
    Returns ``(wx, key)`` after the last step, evaluated. With the state's key and step this reproduces the
    engine's ``wx`` bitwise. ``progress(done, total)`` is called after each evaluated chunk."""
    if n_steps < 0:
        raise ValueError(f"n_steps must be >= 0, got {n_steps}")
    if not cfg.weather.enabled:
        raise ValueError("weather_spinup needs weather.enabled = true")

    def one(wx: mx.array, t: mx.array, key: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        key_next, ks = step_keys(key)
        ws, _ = weather.advance(weather.unpack(wx), t, mx.random.split(ks[Stream.WEATHER], 2)[0], cfg)
        return weather.pack(ws), t + 1, key_next

    fn = mx.compile(one) if compile else one
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


def spinup(
    cfg: Config,
    n_steps: int,
    *,
    step: int,
    key: mx.array,
    compile: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> mx.array:
    """The packed weather after ``n_steps`` weather-only steps, for agents that start at engine step
    ``step`` with state key ``key``: the rest state of the pools of step ``step - n_steps``, advanced
    through the steps ``step - n_steps .. step - 1`` on the chain ``spinup_key(key)``. ``n_steps = 0`` is
    the rest state of the pools of ``step``, which is what the state already carries."""
    start = step - n_steps
    wx = weather.rest_array(cfg, start)
    wx, _ = advance_chain(cfg, wx, start, spinup_key(key), n_steps, compile=compile, progress=progress)
    return wx
