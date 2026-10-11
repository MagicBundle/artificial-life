"""Director build step 1 (docs/DIRECTOR.md §3): the live environment state ``State.env``.

Covers inert-default bit-identity, each knob's effect, the ledger with catastrophes and immigration requests,
checkpoints (including pre-director files), the config hash rule, the RNG stream and the env invariants.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from alife import cli, fields, invariants, layout, lifecycle, rng, runner
from alife.config import (
    POST_V01_KEYS,
    Config,
    apply_overrides,
    config_from_dict,
    config_hash,
    config_to_dict,
    dims,
    immigration_possible,
    load_config,
    validate,
)
from alife.runio import FORMAT_VERSION, CheckpointError, RunDir, load_checkpoint, write_state_file
from alife.state import (
    ENV_FIELDS,
    V0_FIELDS,
    CounterIdx,
    DeathCause,
    Diag,
    Env,
    LedgerIdx,
    State,
    default_env,
    init_state,
    legacy_state_digest,
    state_digest,
    state_from_arrays,
    state_to_arrays,
    to_numpy,
)
from alife.step import make_step

C = CounterIdx
TINY_HASH_V01 = "d25e0eaee7349ae366e3e4e60ebab107cf7a89fcf49be88a12d81099ff2561d5"
DIRECTOR = "director.enabled=true"


def _np(a: mx.array) -> np.ndarray:
    return to_numpy(a)


@pytest.fixture(scope="module")
def dcfg(tiny_cfg: Config) -> Config:
    """``tiny`` (f32) with the director enabled and inert env values."""
    return apply_overrides(tiny_cfg, [DIRECTOR])


def _set_env(s: State, **kw: object) -> State:
    """``s`` with ``env_<name>`` replaced by ``kw[name]`` (converted to the field's dtype)."""
    upd = {f"env_{k}": mx.array(np.asarray(v), dtype=getattr(s, f"env_{k}").dtype) for k, v in kw.items()}
    return s._replace(**upd)


def _run(cfg: Config, steps: int, s: State | None = None, *, pre=None, check: bool = True):
    """``steps`` compiled steps; ``pre(i, s)`` may edit the state before step ``i``.

    Returns the final state and the per-step ``(prev, state, out)`` triples. With ``check`` every step passes
    ``full_check`` (ledger closure, uid issuing, records) and has no error diag bit.
    """
    fn = make_step(cfg)
    s = init_state(cfg) if s is None else s
    hist = []
    for i in range(steps):
        if pre is not None:
            s = pre(i, s)
        prev = s
        s, out = fn(s)
        mx.eval(s, out)
        if check:
            problems = invariants.full_check(s, cfg, prev, out)
            assert problems == [], f"step {i}: {problems}"
            assert int(_np(s.diag)) & 0xFFFF == 0, f"step {i}: diag {Diag(int(_np(s.diag)))!r}"
        hist.append((prev, s, out))
    return s, hist


# --- state layout and digests ---


def test_default_env_and_view(tiny_cfg: Config) -> None:
    seasonal = apply_overrides(tiny_cfg, ["fields.season_period=40", "fields.season_amplitude=0.4"])
    s = init_state(seasonal)
    e = default_env(seasonal)
    assert tuple(e) == ENV_FIELDS and all(np.array_equal(_np(getattr(s, k)), _np(v)) for k, v in e.items())
    env = s.env
    assert isinstance(env, Env) and env._fields == tuple(k[4:] for k in ENV_FIELDS)
    assert float(_np(env.regen_mult)) == 1.0 and float(_np(env.drift_mult)) == 1.0
    assert float(_np(env.season_amp)) == np.float32(0.4) and int(_np(env.season_period)) == 40
    assert _np(env.patch_gain).tolist() == [1.0] * dims(seasonal).pp
    assert int(_np(env.imm_request)) == 0 and float(_np(env.cat_radius)) == 0.0
    assert float(_np(env.cat_kill_p)) == 0.0 and _np(env.cat_center).tolist() == [0.0, 0.0]
    assert env.patch_gain is s.env_patch_gain  # a view of the flat fields


def test_legacy_digest_covers_exactly_the_v0_fields(tiny_cfg: Config) -> None:
    s = init_state(tiny_cfg)
    h = hashlib.sha256()
    for name in V0_FIELDS:  # the pre-env encoding, spelled out
        a = getattr(s, name)
        h.update(f"{name}|{a.dtype}|{tuple(a.shape)}|".encode())
        h.update(np.ascontiguousarray(_np(a)).tobytes())
    assert legacy_state_digest(s) == h.hexdigest()
    assert legacy_state_digest(s) != state_digest(s)
    moved = _set_env(s, regen_mult=2.0)  # env participates in the full digest only
    assert legacy_state_digest(moved) == legacy_state_digest(s) and state_digest(moved) != state_digest(s)


# --- inert bit-identity ---

_SEASONAL_IMM = [
    "population.immigration_every=20",
    "population.immigration_count=16",
    "population.immigration_floor=40",
    "fields.season_period=50",
    "fields.season_amplitude=0.6",
    "fields.season_mode=per_patch",
    "population.max_age=120",
]


@pytest.mark.parametrize("overrides", [[], _SEASONAL_IMM], ids=["plain", "seasons+immigration"])
def test_enabled_but_inert_matches_director_off(tiny_cfg: Config, overrides: list[str]) -> None:
    off = apply_overrides(tiny_cfg, overrides)
    on = apply_overrides(off, [DIRECTOR])
    s_off, h_off = _run(off, 60, check=False)
    s_on, h_on = _run(on, 60, check=False)
    assert legacy_state_digest(s_on) == legacy_state_digest(s_off)
    for (_, _, o1), (_, _, o2) in zip(h_off, h_on, strict=True):
        c1, c2 = _np(o1.counters), _np(o2.counters)
        assert np.array_equal(c1, c2[: len(c1)]) and c2[C.DEATHS_CATASTROPHE] == 0
        assert np.array_equal(_np(o1.ledger), _np(o2.ledger))
    assert all(np.array_equal(_np(getattr(s_on, k)), _np(v)) for k, v in default_env(on).items())
    assert sum(int(_np(o.counters)[C.N_ALIVE]) for _, _, o in h_on) > 0


def test_director_off_passes_env_through_unchanged(tiny_cfg: Config) -> None:
    marked = _set_env(init_state(tiny_cfg), regen_mult=3.0, imm_request=7, cat_kill_p=0.5, drift_mult=9.0)
    s, _ = _run(tiny_cfg, 5, marked, check=False)
    for k in ENV_FIELDS:
        assert np.array_equal(_np(getattr(s, k)), _np(getattr(marked, k))), k


# --- knobs ---


def test_regen_mult_scales_regeneration(dcfg: Config) -> None:
    s0 = init_state(dcfg)
    pos, step = s0.patch_pos, mx.array(0, dtype=mx.int32)
    base = _np(fields.regen_rate(pos, dcfg, step, s0.env)).astype(np.int64)
    for mult in (0.0, 0.5, 2.0, 3.7):
        env = _set_env(s0, regen_mult=mult).env
        got = _np(fields.regen_rate(pos, dcfg, step, env))
        want = np.floor(
            (np.float32(dcfg.fields.regen_max) * _np(fields.fertility(pos, dcfg, step, env)))
            * np.float32(mult)
        )
        assert np.array_equal(got, want.astype(np.int32)), mult
    assert np.array_equal(base, _np(fields.regen_rate(pos, dcfg)))  # inert == the config-only rate
    # Whole step: INJECT_REGEN follows the multiplier (the field starts below its cap everywhere).
    inj = {}
    for mult in (0.0, 1.0, 2.0):
        _, (h0,) = _run(dcfg, 1, _set_env(s0, regen_mult=mult))
        inj[mult] = int(_np(h0[2].ledger)[LedgerIdx.INJECT_REGEN])
    assert inj[0.0] == 0 and 0 < inj[1.0] < inj[2.0]


def test_patch_gain_disables_and_weights_patches(tiny_cfg: Config) -> None:
    one = apply_overrides(tiny_cfg, [DIRECTOR, "fields.n_patches=1"])
    s = init_state(one)
    step = mx.array(3, dtype=mx.int32)
    pos = s.patch_pos
    cx, cy = (int(v // one.world.field_cell) for v in _np(pos)[0])
    base = float(one.fields.fertility_base)
    on = _np(fields.fertility(pos, one, step, s.env))
    off = _np(fields.fertility(pos, one, step, _set_env(s, patch_gain=[0.0]).env))
    assert on[cy, cx] > base + 0.5  # a patch centre is fertile
    assert np.all(off == np.float32(base))  # gain 0: the patch contributes nothing anywhere
    # Several patches: gain weights each patch's Gaussian (float64 reference).
    many = apply_overrides(tiny_cfg, [DIRECTOR])
    sm = init_state(many)
    gain = np.array([0.0, 1.0, 0.5, 2.0], np.float32)
    got = _np(fields.fertility(sm.patch_pos, many, step, _set_env(sm, patch_gain=gain).env))
    want = np.clip(many.fields.fertility_base + _gauss_sum(_np(sm.patch_pos), many, gain), 0.0, 1.0)
    assert np.allclose(got, want, atol=2e-6)


def _gauss_sum(patch_pos: np.ndarray, cfg: Config, weights: np.ndarray) -> np.ndarray:
    """float64 reference: Σ_p w_p · torus Gaussian_p on the cell grid, indexed [y, x]."""
    fg, L, fc = dims(cfg).fg, cfg.world.size, cfg.world.field_cell
    c = (np.arange(fg) + 0.5) * fc
    yy, xx = np.meshgrid(c, c, indexing="ij")
    total = np.zeros((fg, fg))
    for (px, py), w in zip(patch_pos.astype(np.float64), weights, strict=True):
        dx, dy = xx - px, yy - py
        dx -= L * np.round(dx / L)
        dy -= L * np.round(dy / L)
        total += w * np.exp(-(dx * dx + dy * dy) / (2 * cfg.fields.patch_sigma**2))
    return total


def test_drift_mult_scales_patch_motion(tiny_cfg: Config) -> None:
    cfg = apply_overrides(tiny_cfg, [DIRECTOR, "fields.patch_drift=0.4"])
    s = init_state(cfg)
    pos0, vel = _np(s.patch_pos), _np(s.patch_vel)
    L = cfg.world.size
    for mult in (0.0, 1.0, 5.0):
        s1, _ = _run(cfg, 1, _set_env(s, drift_mult=mult))
        moved = np.mod(pos0 + vel * np.float32(mult), L).astype(np.float32)
        assert np.allclose(_np(s1.patch_pos), moved, atol=1e-4), mult
    frozen, _ = _run(cfg, 12, _set_env(s, drift_mult=0.0))
    assert np.array_equal(_np(frozen.patch_pos), pos0)
    fast, _ = _run(cfg, 1, _set_env(s, drift_mult=5.0))
    slow, _ = _run(cfg, 1, s)

    def shift(st: State) -> np.ndarray:  # torus displacement of the patches
        d = _np(st.patch_pos) - pos0
        return d - L * np.round(d / L)

    assert np.allclose(shift(fast), 5 * shift(slow), atol=1e-4) and np.abs(shift(slow)).max() > 0.1


def test_season_override_follows_env(tiny_cfg: Config) -> None:
    # Seasons off in the config; the env switches them on.
    cfg = apply_overrides(tiny_cfg, [DIRECTOR, "fields.season_mode=per_patch"])
    s = init_state(cfg)
    pp = dims(cfg).pp
    assert float(_np(s.env_season_amp)) == 0.0 and int(_np(s.env_season_period)) == 0
    off = _np(fields.season_weights(mx.array(7, dtype=mx.int32), cfg, s.env))
    assert np.array_equal(off, np.ones(pp, np.float32))  # period 0: every weight is exactly 1
    env = _set_env(s, season_amp=0.5, season_period=20).env
    for t in (0, 5, 13, 20, 47):
        got = _np(fields.season_weights(mx.array(t, dtype=mx.int32), cfg, env))
        phase = 2 * math.pi * (t % 20) / 20 + 2 * math.pi * np.arange(pp) / pp
        assert np.allclose(got, 1 + 0.5 * np.sin(phase), atol=1e-5), t
    # Period 0 stays "off" even with a non-zero amplitude, and the modulo never sees a zero.
    still_off = _set_env(s, season_amp=0.9, season_period=0).env
    assert np.array_equal(_np(fields.season_weights(mx.array(5, dtype=mx.int32), cfg, still_off)), off)
    # Fertility (hence regeneration) follows the override, and changing it back restores the baseline.
    t = mx.array(5, dtype=mx.int32)
    f_off = _np(fields.fertility(s.patch_pos, cfg, t, s.env))
    f_on = _np(fields.fertility(s.patch_pos, cfg, t, env))
    assert not np.array_equal(f_off, f_on)
    assert np.array_equal(f_off, _np(fields.fertility(s.patch_pos, cfg)))
    # With the config's own seasons and inert env the weights equal the config-only ones bit for bit.
    seasonal = apply_overrides(cfg, ["fields.season_period=30", "fields.season_amplitude=0.7"])
    ss = init_state(seasonal)
    for t_ in (0, 9, 29, 30, 101):
        step_ = mx.array(t_, dtype=mx.int32)
        assert np.array_equal(
            _np(fields.season_weights(step_, seasonal, ss.env)), _np(fields.season_weights(step_, seasonal))
        )


# --- immigration request ---


def test_imm_request_is_admitted_over_steps_and_carried(dcfg: Config) -> None:
    d, S = dims(dcfg), dcfg.energy.structure
    request = 3 * d.m + 5
    s0 = _set_env(init_state(dcfg), imm_request=request)
    _, hist = _run(dcfg, 6, s0)
    left, admitted = request, 0
    for prev, st, out in hist:
        c, led = _np(out.counters), _np(out.ledger)
        assert int(_np(prev.env_imm_request)) == left
        want = min(left, d.m - int(c[C.BIRTHS]))  # births take lanes first; capacity is far away
        assert c[C.IMMIGRANTS] == want
        assert led[LedgerIdx.INJECT_IMMIGRATION] == want * (dcfg.energy.e_init + S)
        left -= want
        admitted += want
        assert int(_np(st.env_imm_request)) == left
    assert admitted == request - left and left == 0 and admitted == request
    assert hist[0][2].counters[C.IMMIGRANTS].item() > 0  # the pulse starts at once ...
    assert sum(int(_np(o.counters)[C.IMMIGRANTS]) > 0 for _, _, o in hist) >= 4  # ... and spans several steps


def test_imm_request_is_capped_by_capacity_and_the_rest_is_kept(tiny_cfg: Config) -> None:
    cfg = apply_overrides(tiny_cfg, [DIRECTOR, "population.n_init=990", "population.n_max=1024"])
    cfg = apply_overrides(cfg, ["population.max_births_per_step=32"])
    d = dims(cfg)
    s0 = _set_env(init_state(cfg), imm_request=500)
    s, hist = _run(cfg, 4, s0)
    assert int(_np(s.alive).sum()) <= d.n_cap
    adm = sum(int(_np(o.counters)[C.IMMIGRANTS]) for _, _, o in hist)
    assert int(_np(s.env_imm_request)) == 500 - adm > 0  # capacity ran out; the remainder waits
    tot = np.sum([_np(o.counters) for _, _, o in hist], axis=0)
    deaths = tot[C.DEATHS_STARVED] + tot[C.DEATHS_KILLED] + tot[C.DEATHS_AGED]
    assert 990 + tot[C.BIRTHS] + adm - deaths == int(_np(s.alive).sum()) <= d.n_cap


def test_imm_request_competes_with_the_other_demand_by_max(micro_cfg: Config) -> None:
    cfg = apply_overrides(
        micro_cfg, [DIRECTOR, "population.immigration_every=1", "population.immigration_count=3"]
    )
    d = dims(cfg)
    alive = mx.array(np.arange(d.n) < 10)
    key = mx.random.key(0)
    step = mx.array(0, dtype=mx.int32)
    no_req = mx.zeros((d.n,), dtype=mx.bool_)
    for req, want in ((0, 3), (2, 3), (5, 5), (100, d.m)):
        lanes = lifecycle.allocate_lanes(
            no_req, alive, key, cfg, step, imm_request=mx.array(req, dtype=mx.int32)
        )
        assert int(_np(lanes.n_imm)) == want, req
    lanes = lifecycle.allocate_lanes(no_req, alive, key, cfg, step)  # no request: the periodic rule alone
    assert int(_np(lanes.n_imm)) == 3
    # Director on, no config rule: the request is the only demand; director off: the argument is not needed.
    only = apply_overrides(micro_cfg, [DIRECTOR])
    assert immigration_possible(only) and not immigration_possible(micro_cfg)
    lanes = lifecycle.allocate_lanes(no_req, alive, key, only, step, imm_request=mx.array(4, dtype=mx.int32))
    assert int(_np(lanes.n_imm)) == 4
    lanes = lifecycle.allocate_lanes(no_req, alive, key, only, step)
    assert int(_np(lanes.n_imm)) == 0


# --- catastrophe ---


def _inside(pos: np.ndarray, centre: np.ndarray, radius: float, L: float) -> np.ndarray:
    d = pos - centre
    d -= L * np.round(d / L)
    return (d * d).sum(axis=1) <= radius * radius


@pytest.mark.parametrize("kill_p", [1.0, 0.5])
def test_catastrophe_kills_inside_the_radius_only(dcfg: Config, kill_p: float) -> None:
    L = dcfg.world.size
    centre = np.array([5.0, 6.0], np.float32)  # near the seam: the torus minimum image matters
    radius = 40.0
    s0 = _set_env(init_state(dcfg), cat_center=centre, cat_radius=radius, cat_kill_p=kill_p)
    pos0, alive0, uid0 = _np(s0.pos), _np(s0.alive), _np(s0.uid)
    inside = alive0 & _inside(pos0, centre, radius, L)
    assert 20 < inside.sum() < alive0.sum()
    s1, ((_, _, out),) = _run(dcfg, 1, s0)
    c = _np(out.counters)
    dead = alive0 & ~_np(s1.alive)
    killed = c[C.DEATHS_CATASTROPHE]
    assert killed == dead.sum() and not (dead & ~inside).any()  # nobody outside the disc dies
    if kill_p == 1.0:
        assert killed == inside.sum()
    else:
        sd = math.sqrt(inside.sum() * kill_p * (1 - kill_p))
        assert abs(killed - kill_p * inside.sum()) < 5 * sd and 0 < killed < inside.sum()
    dv = _np(out.deaths.valid)
    assert (_np(out.deaths.cause)[dv] == DeathCause.CATASTROPHE).all()
    logged = _np(out.deaths.uid)[dv].tolist()  # the first DL deaths by row index are logged
    assert logged == uid0[dead].tolist()[: len(logged)] and len(logged) == min(killed, dims(dcfg).dl)
    assert c[C.DEATHS_UNLOGGED] == max(killed - dims(dcfg).dl, 0)
    assert c[C.DEATHS_STARVED] + c[C.DEATHS_KILLED] + c[C.DEATHS_AGED] == 0
    # The corpses go through FLOW_CORPSE (E + S each); the one-step event resets kill_p.
    assert _np(out.ledger)[LedgerIdx.FLOW_CORPSE] >= killed * dcfg.energy.structure
    assert float(_np(s1.env_cat_kill_p)) == 0.0
    assert float(_np(s1.env_cat_radius)) == radius and np.array_equal(_np(s1.env_cat_center), centre)
    # Next step: no further catastrophe deaths, and the draw does not leave a trace in the counters.
    _, ((_, _, out2),) = _run(dcfg, 1, s1)
    assert _np(out2.counters)[C.DEATHS_CATASTROPHE] == 0


def test_catastrophe_is_inert_without_radius_or_probability(dcfg: Config) -> None:
    base = init_state(dcfg)
    for kw in (
        dict(cat_center=[60.0, 60.0], cat_radius=0.0, cat_kill_p=1.0),
        dict(cat_center=[60.0, 60.0], cat_radius=50.0, cat_kill_p=0.0),
    ):
        _, ((_, _, out),) = _run(dcfg, 1, _set_env(base, **kw))
        assert _np(out.counters)[C.DEATHS_CATASTROPHE] == 0


def test_catastrophe_mask_uses_start_of_step_positions_and_the_torus(dcfg: Config) -> None:
    s = init_state(dcfg)
    L = dcfg.world.size
    pos = np.zeros((dims(dcfg).n, 2), np.float32)
    pos[:4] = [[1.0, 1.0], [L - 1.0, L - 1.0], [L / 2, L / 2], [10.0, 1.0]]
    env = _set_env(s, cat_center=[0.0, 0.0], cat_radius=3.0, cat_kill_p=1.0).env
    alive = mx.array(np.arange(dims(dcfg).n) < 5)  # row 4 sits at (0, 0) too
    mask = _np(lifecycle.catastrophe_mask(mx.array(pos), alive, env, mx.random.key(1), dcfg))
    assert mask[:5].tolist() == [True, True, False, False, True] and not mask[5:].any()


# --- ledger with the new paths ---


def test_ledger_closes_with_catastrophes_and_immigration_requests(dcfg: Config) -> None:
    L = dcfg.world.size
    plan = {
        4: dict(cat_center=[L / 2, L / 2], cat_radius=L / 3, cat_kill_p=0.6),
        6: dict(imm_request=70),
        12: dict(cat_center=[10.0, 100.0], cat_radius=L / 2, cat_kill_p=1.0, imm_request=5),
        13: dict(regen_mult=0.0, drift_mult=7.0),
        20: dict(regen_mult=3.0, patch_gain=[0.0, 2.0, 1.0, 0.5], season_amp=0.8, season_period=9),
        30: dict(cat_center=[0.0, 0.0], cat_radius=L / 2, cat_kill_p=0.9, imm_request=40),
    }

    def pre(i: int, s: State) -> State:
        return _set_env(s, **plan[i]) if i in plan else s

    s, hist = _run(dcfg, 45, pre=pre)  # full_check per step includes per-step and global ledger closure
    cat = sum(int(_np(o.counters)[C.DEATHS_CATASTROPHE]) for _, _, o in hist)
    imm = sum(int(_np(o.counters)[C.IMMIGRANTS]) for _, _, o in hist)
    assert cat > 50 and imm >= 75
    cum = _np(s.ledger_cum)
    assert cum[LedgerIdx.INJECT_IMMIGRATION] == imm * (dcfg.energy.e_init + dcfg.energy.structure)
    assert int(_np(s.total)) == int(_np(s.total0)) + int(
        cum[LedgerIdx.INJECT_REGEN]
        + cum[LedgerIdx.INJECT_IMMIGRATION]
        - cum[LedgerIdx.HEAT_BASAL : LedgerIdx.HEAT_OVERFLOW + 1].sum()
    )
    s2, _ = _run(dcfg, 45, pre=pre, check=False)
    assert state_digest(s2) == state_digest(s)  # deterministic given the same lever script


def test_compiled_and_eager_agree_on_env_writeback(dcfg: Config) -> None:
    s0 = _set_env(init_state(dcfg), imm_request=50, cat_center=[64.0, 64.0], cat_radius=30.0, cat_kill_p=0.7)
    outs = []
    for compile_ in (True, False):
        fn = make_step(dcfg, compile=compile_)
        s, out = fn(s0)
        mx.eval(s, out)
        outs.append((int(_np(s.env_imm_request)), float(_np(s.env_cat_kill_p)), _np(out.counters)))
    assert outs[0][:2] == outs[1][:2] == (50 - int(outs[0][2][C.IMMIGRANTS]), 0.0)
    assert np.array_equal(outs[0][2], outs[1][2])


# --- checkpoints ---


def _levers() -> dict[str, object]:
    return dict(
        regen_mult=1.5,
        season_amp=0.25,
        season_period=33,
        drift_mult=2.5,
        patch_gain=[1.0, 0.0, 2.0, 0.75],
        imm_request=12,
        cat_center=[3.5, 99.25],
        cat_radius=17.0,
        cat_kill_p=0.125,
    )


def test_checkpoint_round_trip_with_non_default_env(dcfg: Config, tmp_path: Path) -> None:
    s, _ = _run(dcfg, 5, _set_env(init_state(dcfg), **_levers()), check=False)
    path = tmp_path / "s.safetensors"
    write_state_file(path, s, dcfg)
    back = load_checkpoint(path, dcfg)
    assert state_digest(back) == state_digest(s)
    assert float(_np(back.env_drift_mult)) == 2.5 and _np(back.env_patch_gain).tolist() == [1, 0, 2, 0.75]
    assert list(state_to_arrays(back)) == list(State._fields)
    # Resuming from the loaded state continues the uninterrupted trajectory.
    a, _ = _run(dcfg, 5, s, check=False)
    b, _ = _run(dcfg, 5, back, check=False)
    assert state_digest(a) == state_digest(b)


def _write_v1_file(path: Path, s: State, cfg: Config) -> None:
    """A pre-director checkpoint: the 38 v0 arrays only, RNG layout version 1."""
    arrays = {k: v for k, v in state_to_arrays(s).items() if k not in ENV_FIELDS}
    meta = {
        "format_version": str(FORMAT_VERSION),
        "rng_layout_version": "1",
        "genome_layout_version": str(layout.GENOME_LAYOUT_VERSION),
        "step": str(int(_np(s.step))),
        "config_hash": config_hash(cfg),
        "config": json.dumps(config_to_dict(cfg), sort_keys=True),
    }
    mx.save_safetensors(str(path), arrays, meta)


def test_old_checkpoint_loads_with_inert_env_and_resumes(tiny_cfg: Config, tmp_path: Path) -> None:
    s, _ = _run(tiny_cfg, 8, check=False)
    path = tmp_path / "v1.safetensors"
    _write_v1_file(path, s, tiny_cfg)
    back = load_checkpoint(path, tiny_cfg)  # config hash is checked, and matches (no director section)
    assert legacy_state_digest(back) == legacy_state_digest(s)
    assert state_digest(back) == state_digest(s)  # s has inert env too
    assert all(np.array_equal(_np(getattr(back, k)), _np(v)) for k, v in default_env(tiny_cfg).items())
    a, _ = _run(tiny_cfg, 10, s, check=False)
    b, _ = _run(tiny_cfg, 10, back, check=False)
    assert state_digest(a) == state_digest(b)
    # A fork (check_hash=False) of an old file under a director-enabled config gets inert env of that config.
    seasonal = apply_overrides(tiny_cfg, [DIRECTOR, "fields.season_period=30", "fields.season_amplitude=0.5"])
    with pytest.raises(CheckpointError, match="config_hash"):
        load_checkpoint(path, seasonal)
    fork = load_checkpoint(path, seasonal, check_hash=False)
    assert legacy_state_digest(fork) == legacy_state_digest(s)
    assert float(_np(fork.env_season_amp)) == np.float32(0.5) and int(_np(fork.env_season_period)) == 30
    _run(seasonal, 5, fork)  # and it runs, passing every check


def test_missing_non_env_key_and_foreign_versions_are_refused(tiny_cfg: Config, tmp_path: Path) -> None:
    s = init_state(tiny_cfg)
    arrays = state_to_arrays(s)
    no_diag = {k: v for k, v in arrays.items() if k not in ENV_FIELDS and k != "diag"}
    with pytest.raises(ValueError, match="diag"):
        state_from_arrays(no_diag, tiny_cfg)
    with pytest.raises(ValueError, match="env_cat_radius"):  # without cfg nothing is filled
        state_from_arrays({k: v for k, v in arrays.items() if k != "env_cat_radius"})
    partial = {k: v for k, v in arrays.items() if k != "env_cat_radius"}
    assert float(_np(state_from_arrays(partial, tiny_cfg).env_cat_radius)) == 0.0
    with pytest.raises(ValueError, match="bogus"):
        state_from_arrays({**arrays, "bogus": mx.array(0)}, tiny_cfg)
    # A file whose arrays miss a non-env key is a CheckpointError at load.
    meta = {
        "format_version": str(FORMAT_VERSION),
        "rng_layout_version": "1",
        "genome_layout_version": str(layout.GENOME_LAYOUT_VERSION),
        "config_hash": config_hash(tiny_cfg),
    }
    bad = tmp_path / "bad.safetensors"
    mx.save_safetensors(str(bad), no_diag, meta)
    with pytest.raises(CheckpointError, match="diag"):
        load_checkpoint(bad, tiny_cfg)
    # RNG layout versions: 1 and 2 load, anything else is refused.
    for version, ok in (("1", True), ("2", True), ("3", False), ("0", False)):
        p = tmp_path / f"v{version}.safetensors"
        mx.save_safetensors(str(p), arrays, {**meta, "rng_layout_version": version})
        if ok:
            load_checkpoint(p, tiny_cfg)
        else:
            with pytest.raises(CheckpointError, match="rng_layout_version"):
                load_checkpoint(p, tiny_cfg)


def test_old_run_directory_resumes(tiny_cfg: Config, tmp_path: Path) -> None:
    """A run written before the director (no ``director`` in config.json, v1 checkpoints) still resumes."""
    run = tmp_path / "old"
    rd = RunDir.create(run, tiny_cfg)
    res = runner.run(rd, tiny_cfg, init_state(tiny_cfg), 20, quiet=True)
    s20 = res.state
    for _step, path in rd.checkpoints():
        _write_v1_file(path, load_checkpoint(path, tiny_cfg), tiny_cfg)
    conf = json.loads((run / "config.json").read_text())
    del conf["director"]
    (run / "config.json").write_text(json.dumps(conf, sort_keys=True, indent=2) + "\n")
    meta = json.loads((run / "meta.json").read_text())
    meta["rng_layout_version"] = 1
    (run / "meta.json").write_text(json.dumps(meta))
    assert cli.main(["inspect", str(run), "--check"]) == 0
    assert cli.main(["resume", str(run), "--steps", "10", "--quiet"]) == 0
    resumed = load_checkpoint(RunDir.open(run).latest_checkpoint(), tiny_cfg)
    assert legacy_state_digest(resumed) == legacy_state_digest(_run(tiny_cfg, 10, s20, check=False)[0])


# --- invariants ---

_BAD_ENV = {
    "regen_mult_high": dict(regen_mult=4.5),
    "regen_mult_negative": dict(regen_mult=-0.5),
    "regen_mult_nan": dict(regen_mult=float("nan")),
    "drift_mult_high": dict(drift_mult=21.0),
    "patch_gain_high": dict(patch_gain=[1.0, 5.0, 1.0, 1.0]),
    "patch_gain_negative": dict(patch_gain=[1.0, 1.0, -1.0, 1.0]),
    "season_amp_high": dict(season_amp=1.5),
    "season_period_negative": dict(season_period=-3),
    "cat_radius_high": dict(cat_radius=65.0),
    "cat_radius_negative": dict(cat_radius=-1.0),
    "cat_kill_p_high": dict(cat_kill_p=1.5),
    "cat_center_nan": dict(cat_center=[float("nan"), 0.0]),
    "imm_request_negative": dict(imm_request=-1),
}


@pytest.mark.parametrize("bad", list(_BAD_ENV.values()), ids=list(_BAD_ENV))
def test_out_of_range_env_is_flagged(tiny_cfg: Config, dcfg: Config, bad: dict[str, object]) -> None:
    key = next(iter(bad))
    s = _set_env(init_state(dcfg), **bad)
    problems = invariants.full_check(s, dcfg)
    assert any(f"env_{key}" in p for p in problems), problems
    # Inside the compiled step (director enabled) the sticky FIELD_RANGE bit is raised ...
    s1, _ = _run(dcfg, 1, s, check=False)
    assert int(_np(s1.diag)) & Diag.FIELD_RANGE
    # ... and with the director off the levers are inert, so only the host check looks at them.
    s2, _ = _run(tiny_cfg, 1, _set_env(init_state(tiny_cfg), **bad), check=False)
    assert not int(_np(s2.diag)) & Diag.FIELD_RANGE
    assert any(
        f"env_{key}" in p for p in invariants.full_check(_set_env(init_state(tiny_cfg), **bad), tiny_cfg)
    )


def test_inert_and_boundary_env_values_pass(dcfg: Config) -> None:
    assert invariants.full_check(init_state(dcfg), dcfg) == []
    edge = dict(
        regen_mult=4.0,
        drift_mult=20.0,
        patch_gain=[0.0, 4.0, 0.0, 4.0],
        season_amp=1.0,
        season_period=0,
        cat_radius=dcfg.world.size / 2,
        cat_kill_p=1.0,
        imm_request=0,
    )
    s, _ = _run(dcfg, 2, _set_env(init_state(dcfg), **edge))
    assert not int(_np(s.diag)) & Diag.FIELD_RANGE


# --- config ---


def test_config_hash_rules() -> None:
    cfg = load_config(preset="tiny")
    assert config_hash(cfg) == TINY_HASH_V01  # the empty director section is not hashed
    assert ("director", "enabled") in POST_V01_KEYS
    on = apply_overrides(cfg, [DIRECTOR])
    assert config_hash(on) != config_hash(cfg)
    assert config_to_dict(cfg)["director"] == {
        "enabled": False
    }  # saved in config.json, dropped from the hash
    old = config_to_dict(cfg)
    del old["director"]  # a config.json written before the section existed
    back = config_from_dict(old)
    validate(back)
    assert back == cfg and config_hash(back) == TINY_HASH_V01 and not back.director.enabled
    assert immigration_possible(on) and not immigration_possible(cfg)
    assert immigration_possible(
        apply_overrides(cfg, ["population.immigration_floor=10", "population.immigrants_per_step=2"])
    )


# --- RNG ---


def test_catastrophe_stream_leaves_existing_streams_unchanged() -> None:
    assert rng.Stream.CATASTROPHE == 8 and rng.RNG_LAYOUT_VERSION == 2
    assert {1, 2} == set(rng.RNG_COMPATIBLE_VERSIONS)
    key_next, ks = rng.step_keys(mx.random.key(123))
    # Keys of streams 0-7 (and the carried key) as produced by layout version 1: a change here moves a stream.
    v1 = [
        [2443448303, 1255329790],
        [696899775, 813758272],
        [3525600158, 2667720966],
        [1805548085, 290087427],
        [65993009, 3377793283],
        [3653767322, 1893185005],
        [4140191767, 864777987],
        [1675230999, 3874586779],
    ]
    assert _np(ks)[:8].tolist() == v1 and _np(key_next).tolist() == [1896456402, 17229315]
    assert _np(ks)[rng.Stream.CATASTROPHE].tolist() == [3063393795, 3258762642]
