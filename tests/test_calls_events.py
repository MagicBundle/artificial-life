"""Event-triggered calls, step K1 (DESIGN §47): the levels of the four events on planted states, the exact
window sequences for ``hold`` in {0, 1, 4, 64} (the uint8 wrap), the gate on the voice, a newborn on a reused
row (K3) and the edge cases K1 to K6, K11, K13 and K14. Hand-built scenes on the CPU, eager steps."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import helpers
from alife import calls, comm, invariants, layout
from alife.config import dims
from alife.ext.calls import event_names
from alife.state import CounterIdx, LedgerIdx, counter_index, init_state, to_numpy
from alife.step import make_step
from calls_helpers import WEATHER, WORLDS, calls_cfg, patch_wind, storm_threshold
from weather_helpers import fixed_wx, storm2d_config

F32 = np.float32


def ch(cfg, name: str) -> int:
    """The voice channel (and the column of ``evt``) of an event."""
    return event_names(cfg.calls.events).index(name)


def step1(cfg, s):
    s1, out = make_step(cfg, compile=False)(s)
    mx.eval(s1, out)
    return s1, out


# --- the levels, term by term, on synthetic stage arrays -----------------------------------------------


def _stage(n: int, **kw):
    """(s, s1, sv, wind) with the arrays the levels read; rows default to nothing happening."""
    z = np.zeros(n, np.int32)
    p0 = np.full((n, 4), -1, np.int32)
    s = SimpleNamespace(partner=mx.array(kw.get("p0", p0)))
    s1 = SimpleNamespace(
        last_loss=mx.array(kw.get("loss", z)),
        last_harvest=mx.array(kw.get("harvest", z)),
        partner=mx.array(kw.get("p1", kw.get("p0", p0))),
    )
    sv = mx.array(np.asarray(kw.get("sv", np.ones(n, bool)), bool))
    wind = SimpleNamespace(speed=mx.array(np.asarray(kw.get("speed", np.zeros(n)), np.float32)))
    return s, s1, sv, wind


def test_each_level_reads_its_own_term_and_the_survivors_only() -> None:
    cfg = calls_cfg("box2w", "calls.food_q=50", "calls.storm_frac=0.5", "weather.storm_u0=0.08", events=15)
    assert event_names(15) == ("hurt", "food", "storm", "lost") and cfg.comm.width == 4
    thr = storm_threshold(cfg)
    assert thr == F32(0.04)
    below = np.nextafter(thr, F32(0.0))
    p0 = np.full((8, 4), -1, np.int32)
    p0[6, 0] = p0[7, 0] = 3
    p1 = p0.copy()
    p1[6, 0] = p1[7, 0] = -1  # the bond of rows 6 and 7 is gone
    s, s1, sv, wind = _stage(
        8,
        loss=[0, 1, 0, 0, 0, 0, 0, 9],
        harvest=[0, 0, 50, 49, 0, 0, 0, 999],
        speed=[0, 0, 0, 0, thr, below, 0, 1.0],
        p0=p0,
        p1=p1,
        sv=[1, 1, 1, 1, 1, 1, 1, 0],
    )
    lv = to_numpy(calls.levels(s, s1, sv, wind, cfg, calls.runtime(cfg)))
    assert lv.shape == (8, 4) and lv.dtype == bool
    want = np.zeros((8, 4), bool)
    want[1, ch(cfg, "hurt")] = True  # energy taken
    want[2, ch(cfg, "food")] = True  # harvest at food_q (49 is not enough)
    want[4, ch(cfg, "storm")] = True  # exactly the threshold counts, one ulp below does not
    want[6, ch(cfg, "lost")] = True  # a bond of the start of the step is gone
    assert np.array_equal(lv, want)  # row 7 (not a survivor) has every level false


def test_lost_is_a_bond_that_is_gone_or_was_refilled_by_another_partner() -> None:
    cfg = calls_cfg("box2", events=8 | 1)  # hurt and lost
    L = ch(cfg, "lost")
    p0 = np.full((6, 4), -1, np.int32)
    p0[0, 0] = 5  # unchanged
    p0[1, 0] = 5  # broken
    p0[2, 1] = 5  # a bond in the second slot, broken
    p0[3, 0] = 5  # refilled by another partner in the same step (edge case K5)
    p0[4, 0] = 5  # broken, but the row does not survive
    p1 = p0.copy()
    p1[1, 0] = -1
    p1[2, 1] = -1
    p1[3, 0] = 4
    p1[4, 0] = -1
    p1[5, 2] = 0  # a bond that did not exist at the start (formed this step): not lost
    s, s1, sv, wind = _stage(6, p0=p0, p1=p1, sv=[1, 1, 1, 1, 0, 1])
    lv = to_numpy(calls.levels(s, s1, sv, wind, cfg, calls.runtime(cfg)))[:, L]
    assert lv.tolist() == [False, True, True, True, False, False]


# --- the window: exact sequences, the uint8 wrap -------------------------------------------------------


def _pulse(cfg, levels: list[bool], alive: list[bool] | None = None) -> list[int]:
    """``evt`` of one row after each step, driven through ``calls.window_update`` (uint8, as the state)."""
    rt = calls.runtime(cfg)
    evt = mx.zeros((1, 1), mx.uint8)
    out = []
    for i, lv in enumerate(levels):
        sv = mx.array([True if alive is None else alive[i]])
        evt = calls.window_update(evt, mx.array([[lv]]), sv, rt)
        assert evt.dtype == mx.uint8 and evt.shape == (1, 1)
        out.append(int(evt[0, 0].item()))
    return out


@pytest.mark.parametrize("hold", [0, 1, 4, 64])
def test_a_one_step_level_opens_hold_plus_one_steps_and_closed_stays_closed(hold: int) -> None:
    cfg = calls_cfg("box2", f"calls.hold={hold}")
    n = hold + 6
    got = _pulse(cfg, [t == 1 for t in range(n)])
    # after step 1 the window holds hold + 1 steps, then counts down by one a step, to 0, and stays 0: a
    # uint8 would wrap 0 - 1 to 255 (the v0 update maximum(evt - 1, 0) re-opened every closed window)
    want = [0, hold + 1, *range(hold, -1, -1)]
    want += [0] * (n - len(want))
    assert got == want
    assert max(got) <= hold + 1 and 255 not in got
    # the gate opens at t + 1 iff evt > 0: hold + 1 open steps, starting right after the level step
    assert sum(1 for v in got[:-1] if v > 0) == hold + 1
    assert [t + 1 for t, v in enumerate(got) if v > 0] == list(range(2, 2 + hold + 1))


@pytest.mark.parametrize("hold", [0, 1, 4, 64])
def test_window_sequences_match_a_python_model(hold: int) -> None:
    cfg = calls_cfg("box2", f"calls.hold={hold}")
    rs = np.random.default_rng(hold)
    n = 3 * hold + 40
    levels = (rs.random(n) < 0.08).tolist()
    levels[0] = False
    alive = (rs.random(n) > 0.02).tolist()
    w, want = 0, []
    for lv, al in zip(
        levels, alive, strict=True
    ):  # the model: survivors only; level re-arms; else count down
        w = 0 if not al else (hold + 1 if lv else (w - 1 if w > 0 else 0))
        want.append(w)
    assert _pulse(cfg, levels, alive) == want
    # a level that stays true keeps the window at hold + 1
    assert _pulse(cfg, [True] * 5) == [hold + 1] * 5


def test_a_closed_window_is_never_reopened_k14() -> None:
    """evt = 0 with no level stays 0 over many steps and many rows, in the dtype of the state."""
    cfg = calls_cfg("box2", "calls.hold=64")
    rt = calls.runtime(cfg)
    evt = mx.zeros((50, 2), mx.uint8)
    for _ in range(300):
        evt = calls.window_update(evt, mx.zeros((50, 2), mx.bool_), mx.ones((50,), mx.bool_), rt)
    assert evt.dtype == mx.uint8 and not to_numpy(evt).any()
    # and the update never exceeds hold + 1, whatever it starts from
    big = mx.full((4, 2), 65, mx.uint8)
    nxt = to_numpy(calls.window_update(big, mx.ones((4, 2), mx.bool_), mx.ones((4,), mx.bool_), rt))
    assert nxt.max() == 65 and nxt.dtype == np.uint8


def test_dead_rows_have_closed_windows_k6() -> None:
    cfg = calls_cfg("box2", "calls.hold=4")
    evt = mx.full((3, 1), 5, mx.uint8)
    out = calls.window_update(
        evt, mx.ones((3, 1), mx.bool_), mx.array([True, False, True]), calls.runtime(cfg)
    )
    assert to_numpy(out)[:, 0].tolist() == [5, 0, 5]


# --- the whole step: the sources of the levels ---------------------------------------------------------


def _at(nd: int, x: float, y: float, z: float = 6.0) -> list[float]:
    return [x, y, z][:nd]


def _w2(cfg, **bias: float):
    d = dims(cfg)
    w2 = np.zeros((d.w2_rows, d.lay.D_OUT), np.float32)
    for name, v in bias.items():
        w2[d.h, getattr(layout, name)] = v
    return w2


@pytest.mark.parametrize("nd", [2, 3])
def test_hurt_opens_the_victims_window_and_not_the_attackers(nd: int) -> None:
    cfg = calls_cfg("box2" if nd == 2 else "box3", "calls.hold=4", micro=True)
    h = ch(cfg, "hurt")
    victim = dict(pos=_at(nd, 24, 24), energy=60_000, birth_step=-50)
    attacker = dict(pos=_at(nd, 25.5, 24), energy=60_000, birth_step=-50, W2=_w2(cfg, OUT_ATTACK=4.0))
    s = helpers.make_scene(cfg, [victim, attacker], resource=0)
    s1, out = step1(cfg, s)
    assert to_numpy(out.counters)[CounterIdx.ATTACK_HITS] >= 1
    evt = to_numpy(s1.evt)
    assert to_numpy(s1.last_loss)[0] > 0 and evt[0, h] == 5 and evt[1].tolist() == [0, 0]
    assert evt[0, 1 - h] == 0
    s2, _ = step1(cfg, s1)  # the window counts down while nothing hits (the attack goes on: it re-arms)
    assert to_numpy(s2.evt)[0, h] in (4, 5)
    assert invariants.full_check(s1, cfg, s, out) == []


def test_food_boundary_in_the_step_is_the_harvest_of_the_step() -> None:
    """Two harvest levels in one scene: an agent that harvests ``got`` opens the window iff ``got >=
    food_q``; ``food_q = got`` opens it and ``got + 1`` does not (the level reads ``last_harvest``)."""
    base = calls_cfg("box2", "energy.harvest_max=2000", micro=True)
    agents = [dict(pos=_at(2, 8, 8), birth_step=-50, W2=_w2(base, OUT_HARVEST=3.0))]
    s0 = helpers.make_scene(base, agents, resource=900)
    got = int(to_numpy(step1(base, s0)[0].last_harvest)[0])
    assert 1 < got < 2000
    for q, want in ((got, 5), (got + 1, 0), (got - 1, 5), (1, 5), (2000, 0)):
        cfg = replace(base, calls=replace(base.calls, food_q=q))
        s1, _ = step1(cfg, s0)
        assert to_numpy(s1.evt)[0, ch(cfg, "food")] == want, q


def _pair(cfg, nd: int, dist: float = 1.5, **second):
    a = dict(pos=_at(nd, 20, 20), birth_step=-50, energy=60_000)
    b = dict(pos=_at(nd, 20 + dist, 20), birth_step=-50, energy=60_000, **second)
    return helpers.make_scene(cfg, [a, b], bonds=[(0, 1)], resource=0)


@pytest.mark.parametrize("nd", [2, 3])
def test_lost_fires_on_a_voluntary_release_k13(nd: int) -> None:
    cfg = calls_cfg("box2" if nd == 2 else "box3", "calls.hold=2", events=1 + 8, micro=True)
    L = ch(cfg, "lost")
    s = _pair(cfg, nd)
    w2 = np.array(s.W2)
    w2[0, dims(cfg).h, layout.OUT_RELEASE : layout.OUT_RELEASE + 4] = 4.0  # row 0 lets go
    s = s._replace(W2=mx.array(w2))
    s1, _ = step1(cfg, s)
    assert (to_numpy(s1.partner) == -1).all()  # the bond is gone at both ends
    evt = to_numpy(s1.evt)
    assert evt[0, L] == 3 and evt[1, L] == 3  # both ends lost it
    assert not evt[:, 1 - L].any()
    # a pair that keeps its bond has no level
    s_keep = _pair(cfg, nd)
    s2, _ = step1(cfg, s_keep)
    assert (to_numpy(s2.partner)[:2, 0] == [1, 0]).all() and not to_numpy(s2.evt).any()


@pytest.mark.parametrize("nd", [2, 3])
def test_lost_fires_on_an_overstretched_bond(nd: int) -> None:
    cfg = calls_cfg("box2" if nd == 2 else "box3", events=1 + 8, micro=True)
    s = _pair(cfg, nd, dist=6.0)  # break_length 4
    s1, _ = step1(cfg, s)
    assert (to_numpy(s1.partner) == -1).all()
    assert (to_numpy(s1.evt)[:2, ch(cfg, "lost")] == 5).all()


@pytest.mark.parametrize("nd", [2, 3])
def test_lost_fires_on_the_survivor_when_its_partner_dies(nd: int) -> None:
    cfg = calls_cfg("box2" if nd == 2 else "box3", events=1 + 8, micro=True)
    s = helpers.make_scene(
        cfg,
        [
            dict(pos=_at(nd, 20, 20), birth_step=-50, energy=60_000),
            dict(pos=_at(nd, 21.5, 20), birth_step=-50, energy=0),  # starves at S14
        ],
        bonds=[(0, 1)],
        resource=0,
    )
    s1, out = step1(cfg, s)
    alive, evt = to_numpy(s1.alive), to_numpy(s1.evt)
    assert alive[0] and not alive[1] and to_numpy(s1.partner)[0, 0] == -1
    assert evt[0, ch(cfg, "lost")] == 5 and not evt[1].any()  # the dead row holds nothing (K6)
    assert invariants.full_check(s1, cfg, s, out) == []


def test_lost_fires_on_a_storm_break_of_the_bond(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = calls_cfg("box2w", "weather.storm_break_u=0.07", "weather.storm_p=0.0", events=4 + 8, micro=True)
    patch_wind(monkeypatch, lambda n: np.full(n, 0.08))
    s = _pair(cfg, 2)
    s1, _ = step1(cfg, s)
    assert (to_numpy(s1.partner) == -1).all()  # the air speed is above storm_break_u
    evt = to_numpy(s1.evt)[:2]
    assert (evt[:, ch(cfg, "lost")] == 5).all() and (evt[:, ch(cfg, "storm")] == 5).all()


# --- storm: the sheltered air speed against the felt threshold -----------------------------------------


@pytest.mark.parametrize("world", sorted(WEATHER))
def test_storm_threshold_is_storm_frac_times_storm_u0_on_every_world(
    world: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = calls_cfg(world, "calls.storm_frac=0.5", "weather.storm_u0=0.08", "weather.storm_p=0.0", micro=True)
    thr = storm_threshold(cfg)
    speeds = np.array([0.0, np.nextafter(thr, F32(0)), thr, np.nextafter(thr, F32(1)), 0.1, 0.0, 0.0, 0.0])
    patch_wind(monkeypatch, lambda n: np.pad(speeds, (0, n - len(speeds))))
    nd = 2 if world.startswith("box2") else 3
    pos = [[10 + 12 * i, 10 + 9 * (i % 4), 6.0][:nd] for i in range(8)]
    if world.startswith("sph"):  # points on the shell, far apart
        r = dims(cfg).r_lo + 1.0
        pos = [
            (r * np.array(v) / np.linalg.norm(v)).tolist()
            for v in np.random.default_rng(3).normal(size=(8, 3))
        ]
    agents = [dict(pos=p, birth_step=-50, energy=60_000) for p in pos]
    s = helpers.make_scene(cfg, agents, resource=0)
    s1, _ = step1(cfg, s)
    evt = to_numpy(s1.evt)[:8, ch(cfg, "storm")]
    assert (evt > 0).tolist() == [False, False, True, True, True, False, False, False]
    assert (evt[evt > 0] == 5).all()


def test_storm_reads_the_sheltered_speed_r13() -> None:
    """A real uniform wind of 0.05: open at ``storm_frac * storm_u0`` = 0.04, closed at 0.06; and with a full
    cell of material (shelter 0.8, so 0.01) closed at 0.04: the event reads the speed after ``apply_shelter``,
    as the hazard does (R13)."""
    over = ["structures.enabled=true", "weather.storm_p=0.0", "weather.storm_u0=0.04"]
    base = storm2d_config(
        *over,
        "comm.enabled=true",
        "comm.width=2",
        "calls.enabled=true",
        "calls.trigger=true",
        "calls.events=5",
    )
    d = dims(base)
    spots = [(20.0, 30.0), (80.0, 55.0)]
    scene = helpers.make_scene(
        base, [dict(pos=p, birth_step=-50) for p in spots], resource=0, wx=fixed_wx(base, u1=0.05)
    )
    full = mx.full(d.field_shape, base.structures.cap, dtype=mx.int32)

    def storm_windows(cfg, mat):
        s1, _ = step1(cfg, scene._replace(mat=mat))
        return to_numpy(s1.evt)[:2, ch(cfg, "storm")].tolist()

    empty = mx.zeros(d.field_shape, dtype=mx.int32)
    assert storm_windows(base, empty) == [5, 5]  # 0.05 >= 0.04
    assert storm_windows(base, full) == [0, 0]  # 0.01 < 0.04
    hi = replace(base, calls=replace(base.calls, storm_frac=1.5))  # threshold 0.06
    assert storm_windows(hi, empty) == [0, 0]


# --- the gate on the voice -----------------------------------------------------------------------------

QUIET = (
    "energy.harvest_cost=0",
    "energy.defend_cost=0",
    "energy.grip_cost=0",
    "energy.msg_cost=0",
    "energy.emit_cost=0",
    "energy.move_cost=0",
    "energy.turn_cost=0",
    "energy.contract_cost=0",
    "energy.attack_cost=0",
)


@pytest.mark.parametrize("nd", [2, 3])
def test_a_closed_channel_is_silent_and_free_and_an_open_one_speaks_k1(nd: int) -> None:
    cfg = calls_cfg("box2" if nd == 2 else "box3", *QUIET, "comm.cost=8.0", micro=True)
    d, lay = dims(cfg), dims(cfg).lay
    w2 = np.zeros((d.w2_rows, lay.D_OUT), np.float32)
    w2[d.h, lay.OUT_VOICE : lay.OUT_VOICE + 2] = [0.5, 0.2]  # a positive call on both channels
    pos = [_at(nd, 4.0 + 5 * i, 4.0 + 15 * (i % 2)) for i in range(6)]

    n = d.n
    win = np.zeros((n, 2), np.uint8)
    win[0] = [3, 0]  # row 0: only channel 0 open
    win[1] = [0, 2]  # row 1: only channel 1 open
    win[2] = [1, 1]  # row 2: both
    # rows 3 to 5: both closed
    s = helpers.make_scene(cfg, [dict(W2=w2, energy=5000, pos=p) for p in pos], evt=mx.array(win))
    s1, out = step1(cfg, s)
    voice = to_numpy(s1.msg)[:6, layout.MSG :]
    t0, t1 = np.float32(np.tanh(0.5)), np.float32(np.tanh(0.2))
    assert (
        np.allclose(voice[0], [t0, 0]) and np.allclose(voice[1], [0, t1]) and np.allclose(voice[2], [t0, t1])
    )
    assert not voice[3:].any()  # closed windows with a positive output: exactly silent
    # the cost is the existing dithered comm.cost on the open channels only (an exact integer)
    uid, step = s.uid[:6], s.step
    sig = mx.array(np.where(win[:6] > 0, [t0, t1], 0).astype(np.float32))
    due = to_numpy(comm.due_of(uid, step, sig, cfg))
    c, led = to_numpy(out.counters), to_numpy(out.ledger)
    assert c[counter_index(cfg, "COMM_DUE")] == due.sum() and due[3:].sum() == 0 and due[:3].min() > 0
    assert c[counter_index(cfg, "COMM_DUE")] == led[LedgerIdx.HEAT_ACTION]
    assert c[counter_index(cfg, "COMM_SPEAKERS")] == 3
    assert c[counter_index(cfg, "CALLS_OPEN_HURT")] == 2 and c[counter_index(cfg, "CALLS_SPOKE_HURT")] == 2
    assert c[counter_index(cfg, "CALLS_OPEN_FOOD")] == 2 and c[counter_index(cfg, "CALLS_SPOKE_FOOD")] == 2
    # the same scene with the trigger off: every row speaks on both channels and pays for both
    off = replace(cfg, calls=replace(cfg.calls, enabled=False, trigger=False, events=15, hold=4))
    s_off = helpers.make_scene(off, [dict(W2=w2, energy=5000, pos=p) for p in pos])
    s2, out2 = step1(off, s_off)
    assert (to_numpy(s2.msg)[:6, layout.MSG :] > 0).all()
    assert to_numpy(out2.counters)[counter_index(off, "COMM_SPEAKERS")] == 6


def test_the_event_to_hearing_latency_is_two_steps() -> None:
    """Hurt at step 0: the window is open for step 1, whose call is stored in ``msg`` for step 2 to hear."""
    cfg = calls_cfg("box2", "calls.hold=4", micro=True)
    h = ch(cfg, "hurt")
    d = dims(cfg)
    w2 = np.zeros((d.w2_rows, d.lay.D_OUT), np.float32)
    w2[d.h, d.lay.OUT_VOICE : d.lay.OUT_VOICE + 2] = 1.0
    victim = dict(pos=_at(2, 24, 24), energy=60_000, birth_step=-50, W2=w2)
    attacker = dict(pos=_at(2, 25.5, 24), energy=60_000, birth_step=-50,
                    W2=_w2(cfg, OUT_ATTACK=4.0))  # fmt: skip
    s = helpers.make_scene(cfg, [victim, attacker], resource=0)
    s1, _ = step1(cfg, s)  # step 0: the victim is hit and, its window being closed, says nothing
    assert to_numpy(s1.evt)[0, h] == 5 and not to_numpy(s1.msg)[0, layout.MSG :].any()
    s2, _ = step1(cfg, s1)  # step 1: the window gates the voice open
    assert to_numpy(s2.msg)[0, layout.MSG + h] > 0.7 and to_numpy(s2.msg)[0, layout.MSG + 1 - h] == 0.0


@pytest.mark.parametrize("nd", [2, 3])
def test_an_insolvent_speaker_is_silent_k4(nd: int) -> None:
    cfg = calls_cfg("box2" if nd == 2 else "box3", "comm.cost=64.0", events=1 + 8, micro=True)
    d, lay = dims(cfg), dims(cfg).lay
    w2 = np.zeros((d.w2_rows, lay.D_OUT), np.float32)
    w2[d.h, lay.OUT_VOICE : lay.OUT_VOICE + 2] = 3.0
    win = np.zeros((d.n, 2), np.uint8)
    win[:2] = 3
    s = helpers.make_scene(
        cfg,
        [dict(W2=w2, energy=150, pos=_at(nd, 5, 5)), dict(W2=w2, energy=9000, pos=_at(nd, 30, 30))],
        evt=mx.array(win),
    )
    s1, out = step1(cfg, s)
    msg = to_numpy(s1.msg)
    assert not msg[0, layout.MSG :].any() and msg[1, layout.MSG :].all()
    evt = to_numpy(s1.evt)
    assert not to_numpy(s1.alive)[0] and not evt[0].any()  # it starved: a dead row holds no window
    assert to_numpy(s1.alive)[1] and evt[1].tolist() == [2, 2]  # the solvent speaker's window counts down
    assert helpers.ledger_net(to_numpy(out.ledger), cfg) == int(to_numpy(s1.total) - to_numpy(s.total))


# --- K3: a newborn starts with closed windows, also on a row reused in the same step -------------------


def _reuse_scene(monkeypatch: pytest.MonkeyPatch, *extra: str, bud: bool = True):
    """Row 0 (energy 0) starves at S14 while bonded to row 1 and standing in a storm; row 2 buds into the
    first free row, which is row 0 (the E15 scene of the bonds suite). Every row has the same wind, above the
    threshold, and the previous occupant of row 0 had a bond: the arrays of its row say `storm` and `lost`."""
    cfg = calls_cfg("box2w", "weather.storm_p=0.0", *extra, events=1 + 4 + 8, micro=True)
    patch_wind(monkeypatch, lambda n: np.full(n, 0.08))
    agents = [
        dict(pos=(10.0, 10.0), energy=0, birth_step=-100),
        dict(pos=(11.5, 10.0), energy=300_000, birth_step=-100),
        dict(pos=(30.0, 30.0), energy=400_000, birth_step=-100),
    ]
    d = dims(cfg)
    win = np.zeros((d.n, 3), np.uint8)
    win[0] = [5, 5, 5]  # the dying occupant has every window open
    s = helpers.make_scene(cfg, agents, bonds=[(0, 1)], resource=0, evt=mx.array(win))
    if bud:
        s = helpers.set_output_bias(s, cfg, {layout.OUT_REPRODUCE: 4.0, layout.OUT_CHILD_BOND: 4.0})
        w2 = np.array(s.W2)
        w2[:2, d.h, layout.OUT_REPRODUCE] = -4.0  # only row 2 buds
        s = s._replace(W2=mx.array(w2))
    return cfg, s


def test_a_newborn_on_a_reused_row_has_closed_windows_k3(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, s = _reuse_scene(monkeypatch)
    hurt, storm, lost = (ch(cfg, n) for n in ("hurt", "storm", "lost"))
    uid0 = int(to_numpy(s.uid)[0])
    s1, out = step1(cfg, s)
    alive, uid = to_numpy(s1.alive), to_numpy(s1.uid)
    assert alive[:3].all() and uid[0] != uid0  # row 0 died and was reused in the same step
    assert np.asarray(out.births.slot)[np.asarray(out.births.valid)].tolist() == [0]
    partner, evt = to_numpy(s1.partner), to_numpy(s1.evt)
    assert partner[0].tolist() == [2, -1, -1, -1]  # the newborn's only bond is its parent (a child bond)
    # the newborn: closed. Its predecessor's wind (above the threshold) and bond (to a living partner, gone
    # with its death) are not its levels, nor is the child bond it was born with
    assert evt[0].tolist() == [0, 0, 0]
    # the survivors: in the storm (all rows are); row 1 lost its bond to the dead row 0; row 2 did not (the
    # child bond is made after the stage the levels are read at)
    assert evt[1].tolist()[storm] == 5 and evt[1].tolist()[lost] == 5 and evt[1, hurt] == 0
    assert evt[2].tolist()[storm] == 5 and evt[2].tolist()[lost] == 0
    assert invariants.full_check(s1, cfg, s, out) == []


def test_an_immigrant_on_a_reused_row_has_closed_windows_k3(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, s = _reuse_scene(
        monkeypatch, "population.immigration_every=1", "population.immigration_count=1", bud=False
    )
    uid0 = int(to_numpy(s.uid)[0])
    s1, _ = step1(cfg, s)
    uid, alive = to_numpy(s1.uid), to_numpy(s1.alive)
    assert alive[0] and uid[0] != uid0 and to_numpy(s1.parent_uid)[0] == -1  # an immigrant took row 0
    assert not to_numpy(s1.evt)[0].any()
    assert (to_numpy(s1.evt)[1, ch(cfg, "storm")], to_numpy(s1.evt)[2, ch(cfg, "storm")]) == (5, 5)


# --- every world: the windows follow their sources over a random run -----------------------------------


@pytest.mark.parametrize("world", WORLDS)
def test_random_runs_keep_the_window_law_and_the_invariants(
    world: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hurt, food and (with the weather) storm windows of every survivor equal the law applied to the levels
    read back from the state (``last_loss``, ``last_harvest``, the patched air speed); newborn and dead rows
    are closed; ``full_check`` holds; and the windows were exercised (both open and closed, no wrap)."""
    wx = world in WEATHER
    events = 7 if wx else 3
    cfg = calls_cfg(world, "calls.food_q=20", "calls.hold=3", "energy.harvest_max=800", events=events)
    d = dims(cfg)
    if wx:
        speed = lambda n: np.where(np.arange(n) % 3 == 0, 0.07, 0.0)  # noqa: E731
        patch_wind(monkeypatch, speed)
    names = event_names(events)
    fn = make_step(cfg, compile=False)
    s = helpers.set_output_bias(init_state(cfg), cfg, {layout.OUT_ATTACK: 1.0, layout.OUT_HARVEST: 2.0})
    w2 = np.array(s.W2)
    w2[1::2, d.h, layout.OUT_HARVEST] = -4.0  # half of the rows never harvest: the food level is not constant
    s = s._replace(W2=mx.array(w2))
    seen = {n: [0, 0] for n in names}
    steps = 25 if world.startswith("sph") else 60
    for t in range(steps):
        prev = s
        s, out = fn(s)
        mx.eval(s, out)
        pa, na = to_numpy(prev.alive), to_numpy(s.alive)
        same = pa & na & (to_numpy(prev.uid) == to_numpy(s.uid))
        evt, pevt = to_numpy(s.evt), to_numpy(prev.evt)
        assert evt.dtype == np.uint8 and evt.max() <= cfg.calls.hold + 1
        assert not evt[~na].any() and not evt[na & ~same].any()  # dead rows and newborns: closed
        level = {
            "hurt": to_numpy(s.last_loss) > 0,
            "food": to_numpy(s.last_harvest) >= cfg.calls.food_q,
        }
        if wx:
            level["storm"] = speed(d.n) >= storm_threshold(cfg)
        for c, n in enumerate(names):
            want = np.where(level[n], cfg.calls.hold + 1, np.maximum(pevt[:, c].astype(int) - 1, 0))
            assert np.array_equal(evt[same, c], want[same]), (t, n)
            seen[n][0] += int((evt[same, c] > 0).sum())
            seen[n][1] += int((evt[same, c] == 0).sum())
        if t % 8 == 0:
            assert invariants.full_check(s, cfg, prev, out) == [], t
    for n in names:
        assert seen[n][0] > 0 and seen[n][1] > 0, (n, seen)  # the event happened, and not always


def test_a_lone_agent_and_an_empty_world_run() -> None:
    """K10: n_alive 1 (and 0 after it starves) is a valid world for the windows."""
    cfg = calls_cfg("box2", "population.n_init=1", "calls.hold=2")
    s = init_state(cfg)
    fn = make_step(cfg, compile=False)
    for _ in range(5):
        s, out = fn(s)
        mx.eval(s, out)
    assert invariants.full_check(s, cfg) == []
    empty = helpers.make_scene(cfg, [])
    s1, out = step1(cfg, empty)
    assert not to_numpy(s1.alive).any() and not to_numpy(s1.evt).any()
    assert int(to_numpy(out.counters)[CounterIdx.N_ALIVE]) == 0


def test_a_trigger_only_run_has_no_brain_footprint_k11() -> None:
    """The brain of a calls world is the brain of the same comm world: no input, no output, the same layout
    version and weights shapes, the same ``layout_features``; only ``evt`` and the counters are new."""
    from alife import ext

    on = calls_cfg("box2")
    off = replace(on, calls=type(on.calls)())
    assert ext.layout_features(on) == ext.layout_features(off) == "comm2@1"
    assert ext.mask(on) == ext.mask(off) and dims(on).lay == dims(off).lay
    a, b = init_state(on), init_state(off)
    assert a.W1.shape == b.W1.shape and a.W2.shape == b.W2.shape and a.msg.shape == b.msg.shape
    assert [f for f in type(a)._fields if f not in type(b)._fields] == ["evt"]
    assert a.evt.shape == (dims(on).n, 2) and a.evt.dtype == mx.uint8 and not to_numpy(a.evt).any()
    assert [n for n in ext.layout_blocks(on, 2) if n[0] == "calls"] == []
