"""§27.8 the 3D invariants: ``device_check`` (``HEAT_BUOYANCY`` as a sink, the z and pitch bounds under
``POS_RANGE``, non-finite ``pitch``/``buoy_cap`` under ``NONFINITE``) and ``full_check`` (the closures with
slot 15, the 2D-only ``RESERVED`` rule, and ``_check_rows_3d`` with a corruption case for each of its rules
and its own meta-test)."""

from __future__ import annotations

import ast
import inspect
import sys
from collections.abc import Callable
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from alife import invariants, layout, physics
from alife.config import Config, dims
from alife.state import Diag, LedgerIdx, State, State3, sinks, to_numpy, total_energy
from helpers import as_3d, ledger_net, make_scene

f32 = np.float32
PM32 = f32(layout.PITCH_MAX)
Z_LO, Z_HI = 1.0, 23.0  # cfg3: radius 1, height 24 (asserted by the scene)


@pytest.fixture(scope="module")
def cfg3(micro_cfg: Config) -> Config:
    """``micro`` in 3D (N 64, L 48, height 24: z in [1, 23]) with an active buoyancy cap and its costs."""
    return as_3d(
        micro_cfg,
        physics={"buoyancy_max": 0.05, "gravity": 0.03},
        mutation={"buoy_init": 0.25, "buoy_sigma": 0.02},
        energy={"buoyancy_cost": 120, "buoyancy_cap_cost": 30},
    )


@pytest.fixture(scope="module")
def scene3(cfg3: Config) -> State3:
    """A healthy 3D state: agents through the volume, some on the floor and at the surface (inclusive bounds,
    leaving them or at rest), pitches up to ±PITCH_MAX, caps at both ends, and bonds."""
    d, L = dims(cfg3), cfg3.world.size
    assert (d.z_lo, d.z_hi) == (Z_LO, Z_HI)
    rs = np.random.default_rng(31)
    agents: list[dict[str, Any]] = [
        dict(
            pos=(rs.uniform(0, L), rs.uniform(0, L), rs.uniform(d.z_lo, d.z_hi)),
            vel=(0.1, -0.2, rs.uniform(-0.2, 0.2)),
            heading=rs.uniform(-3, 3),
            pitch=rs.uniform(-1.2, 1.2),
            buoy_cap=rs.uniform(0, 1),
            energy=int(rs.integers(20_000, 120_000)),
            birth_step=-30,
        )
        for _ in range(20)
    ]
    edge = [  # (z, v_z, pitch, buoy_cap)
        (d.z_lo, 0.0, PM32, 0.0),
        (d.z_lo, 0.05, -PM32, 1.0),
        (d.z_hi, 0.0, 0.0, 1.0),
        (d.z_hi, -0.05, 0.3, 0.0),
    ]
    for k, (z, vz, p, cap) in enumerate(edge):
        agents[k].update(pos=(agents[k]["pos"][0], agents[k]["pos"][1], z), vel=(0.1, 0.0, vz), pitch=p)
        agents[k]["buoy_cap"] = cap
    return make_scene(cfg3, agents, bonds=[(0, 1), (1, 2), (5, 6)], step=5)


def _np(a: mx.array) -> np.ndarray:
    return to_numpy(a)


def _device(s: State | State3, cfg: Config, ledger: mx.array, prev_total: mx.array) -> int:
    zero = mx.array(0, dtype=mx.int32)
    return int(_np(invariants.device_check(s, ledger, prev_total, zero, zero, zero, zero, cfg)))


def _zero_ledger() -> mx.array:
    return mx.zeros((len(LedgerIdx),), dtype=mx.int64)


def _set(s: State3, name: str, fn: Callable[[np.ndarray], Any]) -> State3:
    a = _np(getattr(s, name)).copy()
    fn(a)
    return s._replace(**{name: mx.array(a).astype(getattr(s, name).dtype)})


def _alive_row(s: State3, k: int = 0) -> int:
    return int(np.flatnonzero(_np(s.alive))[k])


def _dead_row(s: State3) -> int:
    return int(np.flatnonzero(~_np(s.alive))[0])


def _at(
    name: str, row: Callable[[State3], int], value: Any, col: int | None = None
) -> Callable[[State3], State3]:
    def f(s: State3) -> State3:
        r = row(s)
        return _set(s, name, lambda a: a.__setitem__(r if col is None else (r, col), value))

    return f


def _moving(z: float, vz: float) -> Callable[[State3], State3]:
    """An alive row (the 10th) moved to height ``z`` with stored velocity (0, 0, vz)."""

    def f(s: State3) -> State3:
        r = _alive_row(s, 10)
        s = _set(s, "pos", lambda a: a.__setitem__((r, 2), z))
        return _set(s, "vel", lambda a: a.__setitem__(r, (0.0, 0.0, vz)))

    return f


def _z(off: float, top: bool = False) -> Callable[[State3], State3]:
    """``pos.z`` of an alive row (``top``) or a dead row set ``off`` beyond the surface or below the floor."""

    def f(s: State3) -> State3:
        r = _alive_row(s, 11) if top else _dead_row(s)
        return _set(s, "pos", lambda a: a.__setitem__((r, 2), Z_HI + off if top else Z_LO - off))

    return f


# (id, corruption, full_check fragment): one case or more for each rule of _check_rows_3d.
ROWS_3D: list[tuple[str, Callable[[State3], State3], str]] = [
    ("dead_row_below_floor", _z(0.25), "pos: z outside"),
    ("alive_row_above_surface", _z(1e-3, top=True), "pos: z outside"),
    ("nan_z", _at("pos", _alive_row, np.nan, col=2), "pos: z outside"),
    ("pitch_over_max", _at("pitch", _alive_row, np.nextafter(PM32, f32(np.inf))), "pitch: outside"),
    ("pitch_under_min", _at("pitch", _dead_row, -np.nextafter(PM32, f32(np.inf))), "pitch: outside"),
    ("nan_pitch", _at("pitch", _alive_row, np.nan), "pitch: outside"),
    ("cap_over_one", _at("buoy_cap", _alive_row, np.nextafter(f32(1), f32(2))), "buoy_cap: outside"),
    ("cap_negative", _at("buoy_cap", _dead_row, -0.01), "buoy_cap: outside"),
    ("nan_cap", _at("buoy_cap", _alive_row, np.nan), "buoy_cap: outside"),
    ("falling_through_floor", _moving(Z_LO, -0.1), "on the floor moving down"),
    ("rising_through_surface", _moving(Z_HI, 0.1), "at the surface moving up"),
]

# The corruptions with a Diag bit (§22.3 coverage in 3D, §27.8).
DEVICE_3D: list[tuple[str, Diag]] = [
    ("dead_row_below_floor", Diag.POS_RANGE),
    ("alive_row_above_surface", Diag.POS_RANGE),
    ("nan_z", Diag.NONFINITE),
    ("pitch_over_max", Diag.POS_RANGE),
    ("pitch_under_min", Diag.POS_RANGE),
    ("nan_pitch", Diag.NONFINITE),
    ("nan_cap", Diag.NONFINITE),
]


# --- healthy states -----------------------------------------------------------------------------------------


def test_scene_is_clean_with_inclusive_bounds(cfg3: Config, scene3: State3) -> None:
    a = {k: _np(v) for k, v in scene3._asdict().items()}
    d = dims(cfg3)
    assert {d.z_lo, d.z_hi} <= set(a["pos"][a["alive"], 2].tolist())  # the bounds are reached and allowed
    assert {PM32, -PM32} <= set(a["pitch"][a["alive"]].tolist()) and {0.0, 1.0} <= set(a["buoy_cap"].tolist())
    assert invariants.full_check(scene3, cfg3) == []
    assert _device(scene3, cfg3, _zero_ledger(), scene3.total) == 0


def test_integrated_states_pass_the_clamp_rules(cfg3: Config, scene3: State3) -> None:
    """``physics.integrate3`` leaves no stored velocity into the floor or the surface at it, however hard an
    agent is pushed into either (the clamp rule of §27.3 makes the full_check rule exact)."""
    rs = np.random.default_rng(7)
    s, d = scene3, dims(cfg3)
    n, alive = d.n, s.alive
    floors = surfaces = 0
    for _ in range(20):
        force = mx.array(rs.normal(0, 0.4, (n, 3)).astype(f32))
        z = rs.choice([d.z_lo, d.z_hi, 12.0], n) + rs.uniform(-0.3, 0.3, n) * (rs.random(n) < 0.5)
        pos = _np(s.pos).copy()
        pos[:, 2] = np.where(_np(alive), np.clip(z, d.z_lo, d.z_hi), pos[:, 2])
        on_floor = physics.grounded(mx.array(pos), cfg3)
        it = physics.integrate3(
            mx.array(pos), s.heading, s.pitch, force, mx.zeros((n,)), mx.zeros((n,)),
            mx.array(rs.uniform(-0.3, 0.3, n).astype(f32)), on_floor, alive, cfg3,
        )  # fmt: skip
        s = s._replace(pos=it.pos, vel=it.vel, heading=it.heading, pitch=it.pitch)
        assert invariants.full_check(s, cfg3) == []
        assert _device(s, cfg3, _zero_ledger(), s.total) == 0
        floors, surfaces = floors + int(_np(it.z_floor)), surfaces + int(_np(it.z_surface))
    assert floors > 20 and surfaces > 20  # both clamps were exercised


# --- the ledger: HEAT_BUOYANCY is a sink in 3D, RESERVED stays 0 in 2D --------------------------------------


def _paid_buoyancy(s: State3, cfg: Config, q: int) -> tuple[State3, mx.array]:
    """``s`` after the first alive agent pays ``q`` quanta of buoyancy, with the step's ledger (slot 15)."""
    r = _alive_row(s)
    s1 = _set(s, "energy", lambda a: a.__setitem__(r, a[r] - q))
    led = np.zeros(len(LedgerIdx), np.int64)
    led[LedgerIdx.HEAT_BUOYANCY] = q
    s1 = s1._replace(
        total=total_energy(s1.energy, s1.alive, s1.resource, cfg),
        ledger_cum=s.ledger_cum + mx.array(led),
        step=s.step + 1,
    )
    mx.eval(s1)
    return s1, mx.array(led)


def test_heat_buoyancy_closes_the_ledger_in_3d(cfg3: Config, scene3: State3) -> None:
    s1, led = _paid_buoyancy(scene3, cfg3, 37)
    assert ledger_net(led, cfg3) == int(_np(s1.total) - _np(scene3.total)) == -37
    assert invariants.full_check(s1, cfg3, scene3) == []  # per-step and global closure, no RESERVED rule
    assert _device(s1, cfg3, led, scene3.total) == 0
    unbooked = _np(led).copy()
    unbooked[LedgerIdx.HEAT_BUOYANCY] = 0  # the payment without its sink
    assert _device(s1, cfg3, mx.array(unbooked), scene3.total) == Diag.LEDGER
    bad = s1._replace(ledger_cum=scene3.ledger_cum)
    problems = invariants.full_check(bad, cfg3, scene3)
    assert any("global closure" in p for p in problems) and any("per-step closure" in p for p in problems)


def test_slot_15_is_not_a_sink_in_2d(micro_cfg: Config, cfg3: Config, scene3: State3) -> None:
    """Slot 15 is a sink in 3D only: in 2D it is ``RESERVED``, outside the closure and checked to be 0."""
    led = _zero_ledger()
    led15 = mx.array(np.eye(len(LedgerIdx), dtype=np.int64)[LedgerIdx.HEAT_BUOYANCY] * 5)
    assert invariants._ledger_net_np(_np(led15), 3) == -5 and invariants._ledger_net_np(_np(led15), 2) == 0
    assert (
        int(_np(invariants._ledger_net(led15, 3))) == -5 and int(_np(invariants._ledger_net(led15, 2))) == 0
    )
    s2 = make_scene(micro_cfg, [dict(pos=(3.0, 4.0))])
    assert _device(s2, micro_cfg, led15, s2.total) == 0 and _device(s2, micro_cfg, led, s2.total) == 0
    assert _device(scene3, cfg3, led15, scene3.total) == Diag.LEDGER  # 3D: 5 quanta booked but not spent
    s2r = s2._replace(ledger_cum=led15, total0=s2.total)
    assert any("RESERVED" in p for p in invariants.full_check(s2r, micro_cfg))


def test_sinks_and_ledger_net_helpers(micro_cfg: Config, cfg3: Config) -> None:
    assert sinks(2) == tuple(range(LedgerIdx.HEAT_BASAL, LedgerIdx.HEAT_OVERFLOW + 1))
    assert sinks(3) == (*sinks(2), LedgerIdx.HEAT_BUOYANCY)
    led = np.arange(1, 17, dtype=np.int64)
    want2 = led[0] + led[1] - led[2:7].sum()
    assert ledger_net(led, micro_cfg) == want2 and ledger_net(mx.array(led), cfg3) == want2 - led[15]
    for nd, want in ((2, want2), (3, want2 - led[15])):
        assert (
            invariants._ledger_net_np(led, nd) == want == int(_np(invariants._ledger_net(mx.array(led), nd)))
        )


# --- device_check: the 3D bits ------------------------------------------------------------------------------


@pytest.mark.parametrize(("case", "bit"), DEVICE_3D, ids=[c for c, _ in DEVICE_3D])
def test_both_checks_report_3d(cfg3: Config, scene3: State3, case: str, bit: Diag) -> None:
    corrupt, msg = next((c, m) for k, c, m in ROWS_3D if k == case)
    bad = corrupt(scene3)
    bits = _device(bad, cfg3, _zero_ledger(), scene3.total)
    assert bits & bit, f"{Diag(bits)!r} lacks {bit!r}"
    assert any(msg in p for p in invariants.full_check(bad, cfg3))


def test_device_check_3d_bounds_are_float32_and_inclusive(cfg3: Config, scene3: State3) -> None:
    d, r = dims(cfg3), _alive_row(scene3, 12)
    zero, total = _zero_ledger(), scene3.total
    for z, ok in ((d.z_lo, True), (d.z_hi, True), (np.nextafter(f32(d.z_lo), f32(0)), False),
                  (np.nextafter(f32(d.z_hi), f32(np.inf)), False)):  # fmt: skip
        s = _set(scene3, "pos", lambda a, z=z: a.__setitem__((r, 2), z))
        assert _device(s, cfg3, zero, total) == (0 if ok else Diag.POS_RANGE), z
    for p, ok in ((PM32, True), (-PM32, True), (np.nextafter(PM32, f32(4)), False)):
        s = _set(scene3, "pitch", lambda a, p=p: a.__setitem__(r, p))
        assert _device(s, cfg3, zero, total) == (0 if ok else Diag.POS_RANGE), p


def test_cap_range_is_full_check_only(cfg3: Config, scene3: State3) -> None:
    """A finite ``buoy_cap`` outside [0, 1] has no Diag bit (like the other genes, §22.3)."""
    for case in ("cap_over_one", "cap_negative"):
        corrupt = next(c for k, c, _ in ROWS_3D if k == case)
        assert _device(corrupt(scene3), cfg3, _zero_ledger(), scene3.total) == 0


# --- full_check: every _check_rows_3d rule has a corruption case --------------------------------------------


@pytest.mark.parametrize(("corrupt", "msg"), [c[1:] for c in ROWS_3D], ids=[c[0] for c in ROWS_3D])
def test_full_check_reports_3d(cfg3: Config, scene3: State3, corrupt, msg: str) -> None:
    problems = invariants.full_check(corrupt(scene3), cfg3)
    assert any(msg in p for p in problems), problems


_CHECK_FNS_3D = ("_check_rows_3d",)


def _rule_sites_3d() -> set[int]:
    """Source lines of every ``r.check`` / ``r.rows`` / ``r.problems.append`` in ``_check_rows_3d``."""
    tree = ast.parse(inspect.getsource(invariants))
    sites = set()
    for fn in tree.body:
        if not (isinstance(fn, ast.FunctionDef) and fn.name in _CHECK_FNS_3D):
            continue
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            target, attr = ast.unparse(node.func.value), node.func.attr
            if (attr in ("check", "rows") and target == "r") or (attr == "append" and target == "r.problems"):
                sites.add(node.lineno)
    return sites


class _SitedProblems3(list):
    """A problem list that remembers the ``_check_rows_3d`` line of each message (None for other rules)."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[int | None] = []

    def append(self, item: str) -> None:
        f = sys._getframe(1)
        while f is not None and f.f_code.co_name not in _CHECK_FNS_3D:
            f = f.f_back
        self.lines.append(None if f is None else f.f_lineno)
        super().append(item)


def test_every_3d_rule_has_a_corruption_case(
    cfg3: Config, scene3: State3, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``_check_rows_3d`` rule counts as covered when a ``ROWS_3D`` fragment matches a problem that this
    very rule emitted (and no other rule's), so a new 3D rule cannot ship without a corruption case."""
    reports: list[_SitedProblems3] = []

    class Recording(invariants._Report):
        def __init__(self) -> None:
            self.problems = _SitedProblems3()
            reports.append(self.problems)

    monkeypatch.setattr(invariants, "_Report", Recording)
    covered: set[int | None] = set()
    ambiguous: list[str] = []
    for case, corrupt, msg in ROWS_3D:
        reports.clear()
        invariants.full_check(corrupt(scene3), cfg3)
        (probs,) = reports
        lines = {line for p, line in zip(probs, probs.lines, strict=True) if msg in p}
        if len(lines) != 1 or None in lines:
            ambiguous.append(f"{case}: {msg!r} matched rules at lines {lines}")
        covered |= lines
    assert not ambiguous, ambiguous
    sites = _rule_sites_3d()
    assert len(sites) >= 5 and covered <= sites  # the site scan itself is not vacuous
    missing = sorted(sites - covered)
    src = inspect.getsource(invariants).splitlines()
    assert not missing, "_check_rows_3d rules without a corruption case:\n" + "\n".join(
        f"  invariants.py:{n}: {src[n - 1].strip()}" for n in missing
    )


def test_check_rows_3d_runs_only_in_3d(micro_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(invariants, "_check_rows_3d", lambda *a: calls.append(1))
    assert invariants.full_check(make_scene(micro_cfg, [dict(pos=(3.0, 4.0))]), micro_cfg) == []
    assert calls == []
