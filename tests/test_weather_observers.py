"""W1 and the observer layer (G2): the death-cause rows of the social state under the weather (§30.7, §29).

``DeathCause.STORM`` = 4 exists with the weather on; ``social.DEATH_CAUSES`` is the literal pre-storm tuple
and ``social.death_causes(cfg)`` adds ``storm`` when ``cfg.weather.enabled``. These tests pin that the
weather-off state, sidecar and social records keep their pre-W1 shapes, and that a weather-on observed run
counts every storm death in its own row without an out-of-range write. CPU.
"""

from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from alife import layout
from alife.config import Config, WeatherConfig, dims
from alife.game.observe import ObsState
from alife.game.observer import Observer
from alife.game.social import DEATH_CAUSES, SOCIAL_V1, death_causes, init_soc, soc_in, update
from alife.state import CounterIdx as C
from alife.state import DeathCause, init_state, storm_counter_index, to_numpy
from alife.step import make_step
from game_helpers import planted_state
from helpers import set_output_bias
from weather_helpers import fixed_wx, storm2d_config, storm3d_config

STORM_ALL = ("weather.storm_p=1", "weather.storm_u0=0", "weather.storm_span=1e-06")
SOME_STORM = ("weather.storm_p=0.02", "weather.storm_u0=0", "weather.storm_span=1e-06")
BIAS = {  # reproduction, attacks and harvesting fire: births, kills, starvation
    layout.OUT_HARVEST: 2.0,
    layout.OUT_BOND: 2.0,
    layout.OUT_REPRODUCE: 2.0,
    layout.OUT_CHILD_BOND: 2.0,
    layout.OUT_ATTACK: 1.0,
}
IMMIGRATION = ("population.immigration_every=5", "population.immigration_count=8")
WIND = {"u1": 0.1, "u2": 0.1}
CAUSE_COUNTERS = (C.DEATHS_STARVED, C.DEATHS_KILLED, C.DEATHS_AGED, C.DEATHS_CATASTROPHE)


def _np(a: mx.array) -> np.ndarray:
    return to_numpy(a)


def _off(cfg: Config) -> Config:
    return replace(cfg, weather=WeatherConfig())


@pytest.fixture(scope="module", params=["3d", "2d"])
def kind(request: pytest.FixtureRequest) -> str:
    return request.param


def _cfg(kind: str, *overrides: str) -> Config:
    return storm3d_config(*overrides) if kind == "3d" else storm2d_config(*overrides)


# --- the tuples -------------------------------------------------------------------------------------------


def test_death_causes_are_the_pre_storm_tuple_plus_storm_under_weather() -> None:
    assert DEATH_CAUSES == ("starved", "killed", "aged", "catastrophe")
    on = storm3d_config()
    assert death_causes(_off(on)) == DEATH_CAUSES
    assert death_causes(on) == (*DEATH_CAUSES, "storm")
    # the rows are DeathCause values: row i is the cause with value i, STORM the first one after the literal
    assert [DeathCause(i).name.lower() for i in range(len(death_causes(on)))] == list(death_causes(on))
    assert len(DeathCause) == len(death_causes(on)) and len(DEATH_CAUSES) == DeathCause.STORM


def test_the_social_state_has_a_storm_row_only_with_weather_on(kind: str) -> None:
    on = _cfg(kind)
    g = SOCIAL_V1.slots + 1
    for cfg, rows in ((on, 5), (_off(on), 4)):
        so = init_soc(soc_in(init_state(cfg)), cfg, SOCIAL_V1)
        mx.eval(*so)
        assert so.deaths.shape == (rows, g) and so.deaths.dtype == mx.int64, cfg.weather.enabled
        assert so.births.shape == (g,)
    SOCIAL_V1.check(on)  # the bin counts fit with the extra row


# --- the channel ------------------------------------------------------------------------------------------


def test_acts_death_cause_shows_storm_deaths(kind: str) -> None:
    """With a prescribed wind and ``storm_p`` = 1 every agent alive at the start dies as ``STORM``: the
    channel says 4 on exactly those rows and -1 elsewhere, and the count is ``DEATHS_STORM``."""
    cfg = _cfg(kind, *STORM_ALL)
    nd = dims(cfg).nd
    s0 = init_state(cfg)._replace(wx=fixed_wx(cfg, **WIND))
    n0 = int(_np(s0.alive.sum()))
    s1, out = make_step(cfg, observe=True)(s0)
    mx.eval(s1, out)
    dc = _np(out.acts.death_cause)
    assert dc.dtype == np.int8
    assert np.array_equal(dc == DeathCause.STORM.value, _np(s0.alive))
    assert set(np.unique(dc)) == {-1, DeathCause.STORM.value}
    c = _np(out.counters)
    assert c[storm_counter_index(nd)] == n0 and (dc == 4).sum() == n0
    assert all(c[i] == 0 for i in CAUSE_COUNTERS)


def test_a_weather_on_observed_run_counts_each_cause_in_its_own_row(kind: str) -> None:
    """60 real steps with a prescribed wind, a 15% storm hazard and ageing: per step the window's rows are the
    cumulative counters (the four old causes and the storm), and their total is the total number of deaths, so
    no storm death was dropped or written into another cause's bins (the scatter has ``5·G`` bins)."""
    cfg = _cfg(kind, *SOME_STORM, *IMMIGRATION, "population.max_age=40")
    nd = dims(cfg).nd
    sr = SOCIAL_V1
    g = sr.slots + 1
    wx = fixed_wx(cfg, **WIND)
    s = set_output_bias(init_state(cfg), cfg, BIAS)
    energy = _np(s.energy).copy()
    energy[np.flatnonzero(_np(s.alive))[:8]] = 0  # eight rows starve at once
    s = s._replace(energy=mx.array(energy))
    so = init_soc(soc_in(s), cfg, sr)
    first = _np(s.alive)
    so = so._replace(kin=mx.array(np.where(first, np.arange(len(first)) % 4 % g, -1).astype(np.int32)))
    fn = make_step(cfg, observe=True)
    want = np.zeros(5, np.int64)
    for t in range(60):
        s, out = fn(s._replace(wx=wx))
        so = update(so, soc_in(s), out.acts, cfg, sr)
        mx.eval(s, out, *so)
        c = _np(out.counters)
        want += np.array([*(c[i] for i in CAUSE_COUNTERS), c[storm_counter_index(nd)]])
        assert np.array_equal(_np(so.deaths).sum(axis=1), want), t
    assert want[4] > 0 and want[2] > 0 and want[0] > 0, want  # storm, age and starvation deaths all occurred
    assert _np(so.deaths).shape == (5, g) and want.sum() > 20


# --- the weather-off identity -----------------------------------------------------------------------------


def test_weather_off_sidecar_and_social_records_keep_their_shapes(kind: str) -> None:
    """The members of a sidecar are the same with the weather on or off; only ``soc.deaths`` is one row taller
    with it on, and a record's per-group ``deaths`` list is 4 long with it off, 5 with it on. A sidecar of the
    other width is refused (``Observer.open`` then starts fresh)."""
    sr = replace(SOCIAL_V1, snapshot_every=10)
    on = _cfg(kind, *SOME_STORM)
    off = _off(on)
    side, records = {}, {}
    for name, cfg in (("on", on), ("off", off)):
        s = set_output_bias(planted_state(cfg), cfg, BIAS)  # planted tags: kin groups get tracked
        obs = Observer.fresh(s, cfg, social=sr)
        fn = make_step(cfg, observe=True)
        wx = fixed_wx(cfg, **WIND) if cfg.weather.enabled else None
        recs = []
        for t in range(1, 41):
            s, out = fn(s if wx is None else s._replace(wx=wx))
            obs.update(s, out.acts, t)
            mx.eval(s, out.base(), *obs.arrays())
            recs += obs.take_records()
        side[name], records[name] = obs.sidecar(), recs
    assert set(side["on"]) == set(side["off"])
    for k in side["off"]:
        if k == "soc.deaths":
            assert side["off"][k].shape[0] == 4 and side["on"][k].shape[0] == 5
        elif k.startswith("soc.") or k in ObsState._fields:  # the host members depend on the run
            assert side["on"][k].shape == side["off"][k].shape, k
    assert [r["step"] for r in records["off"]] == [10, 20, 30, 40] == [r["step"] for r in records["on"]]
    for name, width in (("off", 4), ("on", 5)):
        lists = [d for r in records[name] for d in r["kin_groups"]["deaths"] if d is not None]
        assert lists and all(len(d) == width for d in lists), name
    # the other width does not load
    s_on = init_state(on)
    blank = Observer._blank(s_on, on, Observer.fresh(s_on, on, social=sr).rules, sr)
    arrays = {**side["off"], "soc.uid": _np(blank.so.uid), "uid": _np(blank.o.uid)}
    loaded, reason = blank._load(arrays)
    assert loaded is None and reason, reason
