"""Far hearing in the whole step (DESIGN §47.7, calls step K2): the ledger closes with it on in the six
worlds at three call costs, the CALLS_FAR_HEARD counter against an independent recount, determinism, resume
and observer purity, the compiled step against the eager one, the bench mirror, the controls (``nofar``,
``reflex``, ``reflex_near``: the oracle's direction, magnitude, its place in the wind term, its own digest),
the migration of the far blocks and the bench report. Eager unless a test says otherwise."""

from __future__ import annotations

import math
import platform
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

import calls_grid_helpers as G
import helpers
from alife import bench, calls, cli, invariants, layout, physics, weather
from alife.config import dims, load_config
from alife.ext.calls import event_names
from alife.runio import RunDir, load_checkpoint, write_state_file
from alife.state import (
    counter_blocks,
    counter_index,
    init_state,
    state_digest,
    to_numpy,
)
from alife.step import make_step
from calls_helpers import WEATHER, WORLDS, calls_cfg, storm_threshold
from test_calls_step import STEPS, forced, run, wind_pattern

FAR = {"box2": "18", "box2w": "18", "box3": "18", "box3w": "18", "sph": "20", "sphw": "20"}


def far_cfg(world: str, *extra: str, **kw):
    return calls_cfg(world, f"calls.far_range={FAR[world]}", "calls.far_xref=2.0", *extra, **kw)


# --- ledger, counters, determinism --------------------------------------------------------------------------


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("cost", [0.0, 32.0, 512.0])
def test_the_ledger_closes_with_far_hearing_on_and_the_far_counter_is_sound(
    world: str, cost: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No new ledger term (far hearing costs nothing and moves no energy): TOTAL is conserved every step in
    the six worlds, the host check passes, and ``CALLS_FAR_HEARD`` is bounded by the alive rows and
    positive."""
    wind_pattern(monkeypatch, world)
    cfg = far_cfg(world, events=7 if world in WEATHER else 3, cost=cost)
    idx = counter_index(cfg, "CALLS_FAR_HEARD")
    heard = 0
    for t, (prev, s, out) in enumerate(run(cfg, STEPS[world], compile=False)):
        led, c = to_numpy(out.ledger), to_numpy(out.counters)
        assert led.shape == (16,) and helpers.ledger_net(led, cfg) == int(
            to_numpy(s.total) - to_numpy(prev.total)
        )
        assert 0 <= c[idx] <= int(to_numpy(prev.alive).sum())
        assert int(to_numpy(s.diag)) & 0xFFFF == 0
        heard += int(c[idx])
        if t % 10 == 0:
            assert invariants.full_check(s, cfg, prev, out) == [], t
    assert heard > 0


@pytest.mark.parametrize("world", ["box2", "box3w", "sph"])
def test_the_far_counter_equals_an_independent_recount(world: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """``CALLS_FAR_HEARD`` of a step = the alive rows whose float64 reference (brute force table, loops)
    hears something from the stored voice of the start of the step."""
    wind_pattern(monkeypatch, world)
    cfg = far_cfg(world, events=7 if world in WEATHER else 3)
    rows, _ = G.ref_table(cfg)
    idx = counter_index(cfg, "CALLS_FAR_HEARD")
    for t, (prev, _s, out) in enumerate(run(cfg, 6, compile=False)):
        if t == 0:
            continue  # no voice is stored before the first step
        ns = SimpleNamespace(
            alive=prev.alive, pos=prev.pos, heading=prev.heading, hdg=getattr(prev, "hdg", None)
        )
        hr, _ = G.ref_hear(cfg, ns, to_numpy(prev.msg)[:, layout.MSG :], rows)
        assert int(to_numpy(out.counters)[idx]) == int(((hr > 0).any(axis=1) & to_numpy(prev.alive)).sum()), t


@pytest.mark.parametrize("world", ["box2", "box3w", "sph"])
def test_determinism_resume_and_observer_purity_with_far_hearing(
    world: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wind_pattern(monkeypatch, world)
    cfg = far_cfg(world, events=7 if world in WEATHER else 3)
    last = lambda g: list(g)[-1][1]  # noqa: E731
    a = last(run(cfg, 30))
    assert state_digest(a) == state_digest(last(run(cfg, 30)))
    half = last(run(cfg, 15))
    path = tmp_path / "half.safetensors"
    write_state_file(path, half, cfg)
    back = load_checkpoint(path, cfg)
    assert state_digest(last(run(cfg, 15, back))) == state_digest(
        a
    )  # K12: the far input is a function of the state
    plain, obs = list(run(cfg, 12)), list(run(cfg, 12, observe=True))
    assert state_digest(plain[-1][1]) == state_digest(obs[-1][1])
    for (_, _, o1), (_, _, o2) in zip(plain, obs, strict=True):
        assert np.array_equal(to_numpy(o1.counters), to_numpy(o2.counters))


@pytest.mark.parametrize("world", ["box2", "box2w", "box3"])
def test_compiled_and_eager_steps_agree_with_far_hearing(world: str) -> None:
    cfg = far_cfg(world, events=7 if world in WEATHER else 3)
    d = dims(cfg)
    s = forced(cfg, init_state(cfg))
    rs = np.random.default_rng(4)
    s = s._replace(
        evt=(mx.array((rs.random((d.n, d.voice_w)) < 0.5).astype(np.uint8) * 3) * s.alive[:, None]).astype(
            mx.uint8
        )
    )
    s = s._replace(
        msg=mx.concatenate(
            [s.msg[:, : layout.MSG], mx.array(rs.random((d.n, d.voice_w)).astype(np.float32)) * 0.9], axis=1
        )
    )
    a, oa = make_step(cfg, compile=False)(s)
    b, ob = make_step(cfg, compile=True)(s)
    mx.eval(a, oa, b, ob)
    assert np.array_equal(to_numpy(oa.counters), to_numpy(ob.counters))
    assert np.array_equal(to_numpy(oa.ledger), to_numpy(ob.ledger))
    assert to_numpy(oa.counters)[counter_index(cfg, "CALLS_FAR_HEARD")] > 0
    for name in type(s)._fields:
        x, y = to_numpy(getattr(a, name)), to_numpy(getattr(b, name))
        if x.dtype.kind == "f":
            np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-5, err_msg=name)
        else:
            assert np.array_equal(x, y), name


@pytest.mark.parametrize("world", ["box2", "box3w", "sph", "sphw"])
@pytest.mark.parametrize("control", ["none", "nofar", "reflex", "reflex_near"])
def test_the_staged_pipeline_equals_the_step_with_far_hearing(
    world: str, control: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if control.startswith("reflex") and world not in WEATHER:
        pytest.skip("the oracle needs the storm event")
    wind_pattern(monkeypatch, world)
    cfg = far_cfg(world, f"calls.control={control}", events=7 if world in WEATHER else 3)
    cfg = replace(cfg, run=replace(cfg.run, compile=False))
    s0 = forced(cfg, init_state(cfg))
    fused, staged = make_step(cfg), bench._StagedStep(cfg)
    s, holder = s0, [s0]
    for i in range(4):
        s, out = fused(s)
        out_staged = staged(holder)
        mx.eval(s, out, holder[0], out_staged)
        assert state_digest(holder[0]) == state_digest(s), i
        for a, b in zip(tree_flatten(out), tree_flatten(out_staged), strict=True):
            assert a[0] == b[0]
            np.testing.assert_array_equal(to_numpy(a[1]), to_numpy(b[1]), err_msg=a[0])


def test_the_interval_row_has_the_far_heard_share(monkeypatch: pytest.MonkeyPatch) -> None:
    """``calls_far_heard_frac`` = CALLS_FAR_HEARD per alive agent-step, in [0, 1], and the counter block ends
    with CALLS_FAR_HEARD."""
    wind_pattern(monkeypatch, "box2w")
    cfg = far_cfg("box2w", events=13, cost=8.0)
    trace = list(run(cfg, 25, compile=False))
    blocks = counter_blocks(cfg)
    assert blocks[-1][0] == "calls" and blocks[-1][1][-1] == "CALLS_FAR_HEARD"
    idx = counter_index(cfg, "CALLS_FAR_HEARD")
    alive_steps = sum(int(to_numpy(p.alive).sum()) for p, *_ in trace)
    far = sum(int(to_numpy(o.counters)[idx]) for *_, o in trace)
    assert 0 < far <= alive_steps
    off = replace(cfg, calls=replace(cfg.calls, far_range=0.0, far_xref=1.0))
    assert (
        "CALLS_FAR_HEARD" not in dict(counter_blocks(off))["calls"]
    )  # K1's counters are unchanged without it


# --- the controls -------------------------------------------------------------------------------------------


def test_nofar_equals_far_hearing_with_the_far_weights_zeroed() -> None:
    """``nofar`` feeds zeros to the far block, so its trajectory is that of the far world whose far W1 rows
    are zero (same layout, brain, cost), and its far counter reads 0 while the far world's does not."""
    cfg = far_cfg("box2", events=3)
    nofar = replace(cfg, calls=replace(cfg.calls, control="nofar"))
    d = dims(cfg)
    lay = d.lay
    s0 = forced(cfg, init_state(cfg))
    w1 = np.array(s0.W1)
    w1[:, lay.IN_VOICE_FAR : lay.IN_FAR_BEARING + 2 * d.voice_w, :] = 0.0
    zeroed = s0._replace(W1=mx.array(w1))
    far_state, nofar_state = zeroed, s0  # nofar starts with the nonzero far rows: they must not matter
    f_far, f_nofar = make_step(cfg, compile=False), make_step(nofar, compile=False)
    heard_far = heard_nofar = 0
    for _ in range(15):
        far_state, oa = f_far(far_state)
        nofar_state, ob = f_nofar(nofar_state)
        mx.eval(far_state, nofar_state, oa, ob)
        heard_far += int(to_numpy(oa.counters)[counter_index(cfg, "CALLS_FAR_HEARD")])
        heard_nofar += int(to_numpy(ob.counters)[counter_index(nofar, "CALLS_FAR_HEARD")])
    assert heard_nofar == 0 and heard_far > 0
    # the two states differ in the far rows only (the unzeroed start of nofar); everything else is the same
    for name in type(far_state)._fields:
        if name in ("W1", "pend_W1"):
            continue
        assert np.array_equal(to_numpy(getattr(far_state, name)), to_numpy(getattr(nofar_state, name))), name
    keep = np.ones(d.w1_rows, bool)
    keep[lay.IN_VOICE_FAR : lay.IN_FAR_BEARING + 2 * d.voice_w] = False
    assert np.array_equal(to_numpy(far_state.W1)[:, keep], to_numpy(nofar_state.W1)[:, keep])


# the oracle -------------------------------------------------------------------------------------------------


def _plant(cfg, rows: list[dict]) -> SimpleNamespace:
    """A namespace of ``dims(cfg).n`` rows with the given agents: ``pos`` (xy or xyz / planet vector),
    ``heading`` (box) or ``hdg`` (sphere), ``voice`` (W,), the rest dead."""
    d = dims(cfg)
    n = d.n
    alive = np.zeros(n, bool)
    pos = np.zeros((n, d.nd), np.float32)
    hd = np.zeros(n, np.float32)
    hdg = np.tile(np.array([1.0, 0.0, 0.0], np.float32), (n, 1))
    voice = np.zeros((n, d.voice_w), np.float32)
    if d.sph:
        pos[:] = (0.0, 0.0, cfg.world.radius + 3.0)
    elif d.d3:
        pos[:, 2] = d.z_lo
    for i, r in enumerate(rows):
        alive[i] = True
        pos[i] = r["pos"]
        if "heading" in r:
            hd[i] = r["heading"]
        if "hdg" in r:
            hdg[i] = r["hdg"]
        if "voice" in r:
            voice[i] = r["voice"]
    return SimpleNamespace(
        alive=mx.array(alive), pos=mx.array(pos), heading=mx.array(hd), hdg=mx.array(hdg), voice=voice
    )


def _force(cfg, ns) -> np.ndarray:
    rt = calls.runtime(cfg)
    h, b = calls.hear_far(ns, mx.array(ns.voice), G.columns(cfg, ns), rt, cfg)
    f = calls.oracle_force(ns, cfg, None, (h, b))
    mx.eval(f)
    return np.array(f)


@pytest.mark.parametrize("heading", [0.0, 1.3, -2.0])
def test_the_reflex_force_points_away_from_the_storm_callers_with_magnitude_thrust_max(
    heading: float,
) -> None:
    cfg = far_cfg("box2w", "calls.control=reflex", events=5)
    ph = cfg.physics.thrust_max
    storm = [0.0, 0.9]  # channel 1 is the storm
    rows = [{"pos": (64.0, 64.0), "heading": heading}]
    rows += [{"pos": (76.0 + 0.3 * k, 64.0 + 0.2 * k), "voice": storm} for k in range(5)]  # to the east
    rows += [{"pos": (64.0, 110.0), "heading": 0.0}]  # out of earshot
    rows += [{"pos": (52.0, 64.0), "voice": [0.9, 0.0]}]  # a `hurt` call 12 west: not the storm channel
    ns = _plant(cfg, rows)
    f = _force(cfg, ns)
    assert np.hypot(*f[0]) == pytest.approx(ph, rel=1e-5)
    assert f[0, 0] < 0 and abs(f[0, 1]) < ph * math.sin(math.radians(25.0))  # away from the east, +- a column
    assert not f[1:].any()  # nobody else hears a storm: the callers are within each other's near range
    assert np.isfinite(f).all()
    # without the storm callers the hurt call alone moves nobody
    quiet = _plant(cfg, [rows[0], rows[-1]])
    assert not _force(cfg, quiet).any()


def test_the_reflex_acts_only_on_a_storm_heard_above_the_threshold() -> None:
    quiet = far_cfg("box2w", "calls.control=reflex", "calls.far_xref=50.0", events=5)
    loud = far_cfg("box2w", "calls.control=reflex", "calls.far_xref=1.0", events=5)
    rows = [{"pos": (64.0, 64.0), "heading": 0.0}, {"pos": (76.0, 64.0), "voice": [0.0, 0.9]}]
    ns = _plant(loud, rows)
    assert np.hypot(*_force(loud, ns)[0]) == pytest.approx(loud.physics.thrust_max, rel=1e-5)
    h, _ = G.hear(quiet, ns, ns.voice)
    assert 0.0 < h[0, 1] < calls.ORACLE_HEARD_MIN  # audible, and below the threshold
    assert not _force(quiet, ns)[0].any()


def test_the_reflex_force_on_a_sphere_is_tangent_and_away() -> None:
    cfg = far_cfg("sphw", "calls.control=reflex", events=5)
    rs = np.random.default_rng(1)
    rad = cfg.world.radius + 3.0
    for trial in range(5):
        u = rs.normal(size=3)
        u /= np.linalg.norm(u)
        t = rs.normal(size=3)
        t -= (t @ u) * u
        hdg = t / np.linalg.norm(t)
        lat = np.cross(u, hdg)
        away = math.cos(1.1) * hdg + math.sin(1.1) * lat
        uc = u * math.cos(12.0 / cfg.world.radius) + away * math.sin(12.0 / cfg.world.radius)
        rows = [{"pos": u * rad, "hdg": hdg}] + [{"pos": uc * rad, "voice": [0.0, 0.9]}] * 3
        f = _force(cfg, _plant(cfg, rows))[0].astype(np.float64)
        assert np.linalg.norm(f) == pytest.approx(cfg.physics.thrust_max, rel=1e-4)
        assert abs(f @ u) < 1e-5 * cfg.physics.thrust_max  # horizontal: tangent to the sphere
        assert (f / np.linalg.norm(f)) @ away < -math.cos(math.radians(25.0)), trial  # opposite the storm


def _spy_wind(monkeypatch: pytest.MonkeyPatch, store: list) -> None:
    orig = physics.forces

    def spy(*a, **k):
        store.append(None if k.get("wind") is None else np.array(k["wind"]))
        return orig(*a, **k)

    monkeypatch.setattr(physics, "forces", spy)


def test_the_oracle_force_is_added_to_the_wind_term(monkeypatch: pytest.MonkeyPatch) -> None:
    """The physics receives ``wind + oracle`` through its existing ``wind=`` argument: with a constant wind
    force ``F0`` on every row, the extra force of a reflex step is the oracle's (norm at most ``thrust_max``,
    equal to the independent recomputation from the float64 reference), and a ``none`` step receives exactly
    ``F0``."""
    cfg0 = far_cfg("box2w", events=5)
    thr = storm_threshold(cfg0)
    f0 = np.array([0.003, -0.002], np.float32)

    def fake(ws, pos, buoy_cap, alive, cfg):
        n = pos.shape[0]
        return weather.AgentForce(
            mx.broadcast_to(mx.array(f0), (n, 2)), mx.full((n,), float(thr) + 0.01), None, mx.array(False)
        )

    monkeypatch.setattr(weather, "agent_wind", fake)
    seen: dict[str, list] = {}
    states = {}
    for control in ("none", "reflex"):
        cfg = far_cfg("box2w", f"calls.control={control}", events=5)
        d = dims(cfg)
        s = forced(cfg, init_state(cfg))
        s = s._replace(evt=(mx.ones((d.n, 2)) * 3 * s.alive[:, None]).astype(mx.uint8))
        fn = make_step(cfg, compile=False)
        store: list = []
        _spy_wind(monkeypatch, store)
        prev = s
        for _ in range(3):
            prev, (s, _o) = s, fn(s)
            mx.eval(s)
        seen[control], states[control] = store, prev
        monkeypatch.undo()
        monkeypatch.setattr(weather, "agent_wind", fake)
    for w in seen["none"]:
        assert np.array_equal(w, np.broadcast_to(f0, w.shape))
    prev = states["reflex"]
    cfg = far_cfg("box2w", "calls.control=reflex", events=5)
    rows, _ = G.ref_table(cfg)
    ns = SimpleNamespace(alive=prev.alive, pos=prev.pos, heading=prev.heading)
    hr, br = G.ref_hear(cfg, ns, to_numpy(prev.msg)[:, layout.MSG :], rows)
    extra = seen["reflex"][-1] - f0
    alive = to_numpy(prev.alive)
    ph = cfg.physics.thrust_max
    expect = np.zeros_like(extra)
    for i in np.flatnonzero(alive):
        if hr[i, 1] >= calls.ORACLE_HEARD_MIN and abs(hr[i, 1] - calls.ORACLE_HEARD_MIN) > 1e-4:
            fwd, lat = G.frame(cfg, ns, i)
            v = br[i, 2] * fwd + br[i, 3] * lat
            expect[i] = -ph * v / max(np.linalg.norm(v), 1e-3)
    ok = alive & (np.abs(hr[:, 1] - calls.ORACLE_HEARD_MIN) > 1e-4)
    np.testing.assert_allclose(extra[ok], expect[ok], atol=2e-5)
    assert (np.linalg.norm(extra[ok], axis=1) > 0.99 * ph).sum() > 20  # many rows are pushed at full strength
    assert np.linalg.norm(extra, axis=1).max() <= ph * 1.0001


def test_a_reflex_world_calls_every_open_window_at_half(monkeypatch: pytest.MonkeyPatch) -> None:
    """The comm outputs of the event channels are ignored (their bias here would say 2.0, a full call):
    an open window stores exactly 0.5, a closed one 0, and the cost is paid (the voice due is booked)."""
    wind_pattern(monkeypatch, "box2w")
    cfg = far_cfg("box2w", "calls.control=reflex", events=5)
    d = dims(cfg)
    s = forced(cfg, init_state(cfg))
    rs = np.random.default_rng(2)
    openw = (rs.random((d.n, 2)) < 0.5) & to_numpy(s.alive)[:, None]
    s = s._replace(evt=mx.array(openw.astype(np.uint8) * 2))
    s1, out = make_step(cfg, compile=False)(s)
    mx.eval(s1, out)
    voice = to_numpy(s1.msg)[:, layout.MSG :]
    solvent = to_numpy(s1.alive)  # a row that died this step stores nothing
    want = np.where(openw, 0.5, 0.0)
    assert np.array_equal(voice[solvent], want[solvent])
    assert int(to_numpy(out.counters)[counter_index(cfg, "COMM_DUE")]) > 0


def test_reflex_near_hears_the_near_block_only() -> None:
    """The near oracle on a real neighbour search: a storm caller 2 units away pushes the hearer straight away
    from it with ``thrust_max``, one 16 units away (beyond the sensing radius) does not."""
    from alife import sensing, spatial

    cfg = far_cfg("box2w", "calls.control=reflex_near", events=5)
    d = dims(cfg)

    def near_force(caller_x: float) -> np.ndarray:
        s0 = init_state(cfg)
        n = d.n
        alive = np.zeros(n, bool)
        alive[:2] = True
        pos = np.zeros((n, 2), np.float32)
        pos[0], pos[1] = (64.0, 64.0), (caller_x, 64.0)
        msg = np.zeros((n, d.msg_w), np.float32)
        msg[1, layout.MSG + 1] = 0.9
        s = s0._replace(
            alive=mx.array(alive), pos=mx.array(pos), heading=mx.zeros((n,), mx.float32), msg=mx.array(msg)
        )
        nbr = spatial.find_neighbors(s.pos, s.alive, cfg)
        ctx = SimpleNamespace(nbr=nbr, desc=sensing.neighbor_descriptors(s, nbr, cfg))
        f = calls.oracle_force(s, cfg, ctx, None)
        mx.eval(f)
        return np.array(f)

    ph = cfg.physics.thrust_max
    f = near_force(66.0)
    assert np.hypot(*f[0]) == pytest.approx(ph, rel=1e-4) and f[0, 0] < 0 and abs(f[0, 1]) < 1e-4
    assert not f[1].any()  # the caller hears no storm of its own
    assert not near_force(80.0)[0].any()


# own digest -------------------------------------------------------------------------------------------------

# The state digest of `forced(box2w world, events=5, cost 32)` after 12 EAGER steps on the CPU for each
# control, recorded at the K2 head: each control is trajectory-changing and has a digest of its own. Pinned
# for the platform and MLX version it was recorded on, like the sphere goldens.
GOLDEN_CONTROLS = {
    "mlx": "0.32.3",
    "Darwin-arm64-cpu": {
        "none": "f4437853a6297c780c54fcc462cbd273",
        "nofar": "0fe52d42d996f814f27a349a62a48453",
        "reflex": "f0c6c9f45886ae0262472033770f8a94",
        "reflex_near": "70a1e53aa43780dc71119958a1dd16d8",
    },
}


def _control_digest(control: str, monkeypatch: pytest.MonkeyPatch) -> str:
    wind_pattern(monkeypatch, "box2w")
    cfg = far_cfg("box2w", f"calls.control={control}", events=5)
    last = list(run(cfg, 12, compile=False))[-1][1]
    return state_digest(last)


def test_every_control_is_trajectory_changing_with_a_digest_of_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digests = {c: _control_digest(c, monkeypatch) for c in ("none", "nofar", "reflex", "reflex_near")}
    assert len(set(digests.values())) == 4, digests
    again = _control_digest("reflex", monkeypatch)
    assert again == digests["reflex"]  # reproducible
    key = f"{platform.system()}-{platform.machine()}-{'gpu' if mx.default_device().type == mx.gpu else 'cpu'}"
    pins = GOLDEN_CONTROLS.get(key)
    if mx.__version__ != GOLDEN_CONTROLS["mlx"] or not pins:
        pytest.skip(f"pinned for MLX {GOLDEN_CONTROLS['mlx']} on Darwin-arm64-cpu")
    for control, digest in digests.items():
        assert digest.startswith(pins[control]), (control, digest)


# --- migration ----------------------------------------------------------------------------------------------


F32 = "brain.weight_dtype=float32"


def _comm_source(tmp_path: Path, preset: str):
    cfg = load_config(
        preset=preset,
        overrides=[F32, "comm.enabled=true", "comm.width=2", "comm.cost=8", "repro.min_age=100000"],
    )
    fn, s = make_step(cfg), init_state(cfg)
    for _ in range(40):
        s, _ = fn(s)
    mx.eval(s)
    rd = RunDir.create(tmp_path / "src", cfg)
    rd.save_checkpoint(s, cfg)
    rd.wait_saves()
    return cfg, s, rd.path


@pytest.mark.parametrize("preset", ["tiny", "tiny3d"])
def test_far_blocks_migrate_with_zero_rows_and_one_step_equals_the_source(
    preset: str, tmp_path: Path, capsys
) -> None:
    cfg, s, src = _comm_source(tmp_path, preset)
    out = tmp_path / "far"
    args = ["--enable", "calls", "--set", "calls.far_range=18", "--set", "calls.far_xref=2"]
    assert cli.main(["migrate", str(src), *args, "--out", str(out), "--quiet"]) == 0
    err = capsys.readouterr().err
    assert "without calls.trigger" in err  # the first form cannot gain the trigger later
    run1 = RunDir.open(out)
    cfg1, meta = run1.config(), run1.meta()
    assert meta["layout_features"] == "comm2@1,calls2@1" and cfg1.calls.far_range == 18.0
    d0, d1 = dims(cfg), dims(cfg1)
    assert d1.lay.D_IN == d0.lay.D_IN + 6 and d1.lay.D_OUT == d0.lay.D_OUT
    s1 = load_checkpoint(run1.checkpoints()[-1][1], cfg1)
    w0, w1 = to_numpy(s.W1), to_numpy(s1.W1)
    lo, hi = d1.lay.IN_VOICE_FAR, d1.lay.IN_FAR_BEARING + 4
    assert w1.shape[1] == w0.shape[1] + 6 and not w1[:, lo:hi].any()  # six zero rows
    assert np.array_equal(w1[:, :lo], w0[:, :lo]) and np.array_equal(
        w1[:, hi:], w0[:, lo:]
    )  # recurrent rows shifted
    (a, oa), (b, ob) = make_step(cfg, compile=False)(s), make_step(cfg1, compile=False)(s1)
    mx.eval(a, b, oa, ob)
    assert int(to_numpy(a.next_uid)) == int(to_numpy(s.next_uid))  # a birth-free window
    for name in type(a)._fields:
        if name in {"W1", "W2", "pend_W1", "pend_W2"}:
            continue
        x, y = to_numpy(getattr(a, name)), to_numpy(getattr(b, name))
        if x.dtype.kind == "f":
            np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-6, err_msg=name)
        else:
            assert np.array_equal(x, y), name
    assert np.array_equal(to_numpy(oa.ledger), to_numpy(ob.ledger))
    assert int(to_numpy(ob.counters)[counter_index(cfg1, "CALLS_FAR_HEARD")]) > 0
    assert cli.main(["resume", str(out), "--steps", "3", "--quiet"]) == 0


def test_far_hearing_cannot_be_added_to_a_world_that_already_has_calls(tmp_path: Path, capsys) -> None:
    """DESIGN §47.5 (review L2 of K1): the token of a kept feature changes (none to ``calls2@1``), which a
    migration refuses (nor can a fork: a checkpoint never crosses layout_features); enable and far hearing
    in one call from the source."""
    _cfg, _s, src = _comm_source(tmp_path, "tiny")
    mid = tmp_path / "mid"
    assert cli.main(["migrate", str(src), "--enable", "calls", "--out", str(mid), "--quiet"]) == 0
    err = capsys.readouterr().err
    assert "without calls.trigger and without calls.far_range" in err and "second migration" in err
    rc = cli.main(
        ["migrate", str(mid), "--set", "calls.far_range=18", "--out", str(tmp_path / "late"), "--quiet"]
    )
    assert rc != 0 and "calls" in capsys.readouterr().err
    # a fork cannot do it either (it adds a brain block: a checkpoint never crosses layout_features); the way
    # is to migrate the SOURCE with --enable calls and the far hearing in one call (the test above)
    rc = cli.main(["run", "--init-from", str(mid), "--steps", "4", "--out", str(tmp_path / "fork"), "--quiet",
                   "--set", "calls.far_range=18"])  # fmt: skip
    assert rc != 0 and "layout_features" in capsys.readouterr().err


# --- the bench ----------------------------------------------------------------------------------------------


def test_bench_reports_the_far_gate_and_the_parts() -> None:
    cfg = far_cfg("box2w", events=5)
    cfg = replace(cfg, run=replace(cfg.run, compile=False))
    rep = bench.run_bench(cfg, steps=4, warmup=1, stages=True, weather_pairs=1, calls_arm=True)
    assert "calls" in rep and "calls_far" in rep and rep["calls_far"]["calls_config"]["far_range"] == 18.0
    parts = rep["stages"]["calls_parts"]
    assert {"eval_overhead", "gate", "window", "grid", "deposit", "apply"} <= set(parts)
    assert all(v >= 0.0 or k == "gate" for k, v in parts.items())
    text = bench.format_report(rep)
    assert "calls far hearing" in text
    assert event_names(5) == ("hurt", "storm")
