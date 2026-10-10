"""§9 physics: attenuated thrust, springs, repulsion, grip locomotion (F11), integration and stability."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from alife import physics
from alife.actions import Actions
from alife.config import Config, ConfigError, apply_overrides, dims, stability_number
from alife.layout import MSG, NCH, B
from alife.spatial import Neighbors


def _actions(n: int, **kw: np.ndarray | float) -> Actions:
    """All-zero (inactive) actions with the given fields overridden."""
    z = np.zeros(n, np.float32)
    base = {
        "thrust": z, "turn": z, "msg": np.zeros((n, MSG), np.float32), "emit": np.zeros((n, NCH), np.float32),
        "harvest": z, "grip": z, "contract": z, "bond_want": np.zeros(n, bool),
        "release": np.zeros((n, B), bool), "share": np.zeros((n, B), np.float32), "attack": np.zeros(n, bool),
        "defend": z, "reproduce": np.zeros(n, bool), "child_bond": np.zeros(n, bool),
        "bond_k": np.full(n, -1, np.int32), "bond_target": np.full(n, -1, np.int32),
        "attack_k": np.full(n, -1, np.int32), "attack_target": np.full(n, -1, np.int32),
    }  # fmt: skip
    for k, v in kw.items():
        base[k] = np.broadcast_to(np.asarray(v, base[k].dtype), base[k].shape).copy()
    return Actions(**{k: mx.array(v) for k, v in base.items()})


def _min_image(d: np.ndarray, L: float) -> np.ndarray:
    return (d - np.float32(L) * np.round(d / np.float32(L))).astype(np.float32)


def _neighbors(pos: np.ndarray, alive: np.ndarray, cfg: Config) -> Neighbors:
    """Brute-force K nearest alive agents within R in float32 (the §4 contract without cell truncation)."""
    n, k, R = pos.shape[0], dims(cfg).k, cfg.sensing.radius
    off = _min_image(pos[None, :, :] - pos[:, None, :], cfg.world.size)  # (N,N,2): me → j
    d2 = (off * off).sum(-1)
    ok = alive[:, None] & alive[None, :] & ~np.eye(n, dtype=bool) & (d2 < np.float32(R * R))
    d2 = np.where(ok, d2, np.inf)
    order = np.argsort(d2, axis=1, kind="stable")[:, :k]
    pd2 = np.take_along_axis(d2, order, 1)
    valid = np.isfinite(pd2)
    rows = np.arange(n)[:, None]
    idx = np.where(valid, order, rows).astype(np.int32)
    offk = np.where(valid[..., None], off[rows, order], 0).astype(np.float32)
    dist = np.where(valid, np.sqrt(np.where(valid, pd2, 0)), R).astype(np.float32)
    return Neighbors(
        mx.array(idx),
        mx.array(valid),
        mx.array(offk),
        mx.array(dist),
        mx.array(0, mx.int32),
        mx.array(0, mx.int32),
    )


def _scene(cfg: Config, xy: list[tuple[float, float]], bonds_: list[tuple[int, int]] = ()):
    n = dims(cfg).n
    pos = np.zeros((n, 2), np.float32)
    pos[: len(xy)] = xy
    alive = np.zeros(n, bool)
    alive[: len(xy)] = True
    partner = np.full((n, B), -1, np.int32)
    rev = np.full((n, B), -1, np.int32)
    for i, j in bonds_:
        si, sj = int(np.argmax(partner[i] < 0)), int(np.argmax(partner[j] < 0))
        partner[i, si], rev[i, si], partner[j, sj], rev[j, sj] = j, sj, i, si
    return pos, alive, partner


def _step(cfg: Config, pos, heading, alive, partner, act: Actions):
    nbr = _neighbors(pos, alive, cfg)
    sv, p_ = mx.array(alive), mx.array(partner)
    f = physics.forces(mx.array(pos), mx.array(heading), act, nbr, sv, p_, cfg)
    out = physics.integrate(mx.array(pos), mx.array(heading), f, act.grip, act.turn, sv, cfg)
    return tuple(np.array(a) for a in out), np.array(f)


def _dist(a: np.ndarray, b: np.ndarray, L: float) -> float:
    return float(np.linalg.norm(_min_image((b - a).astype(np.float32), L)))


def test_thrust_attenuation_by_contacts(micro_cfg: Config) -> None:
    """Thrust factor 1, 0.75, 0.5, 0.25, 0, 0 for 0..5 contacts; dead or distant neighbours do not count."""
    ph = micro_cfg.physics
    ring = [(1.5, 0.0), (-1.5, 0.0), (0.0, 1.5), (0.0, -1.5), (1.1, 1.1)]
    for nc in range(6):
        xy = [(20.0, 20.0)] + [(20.0 + dx, 20.0 + dy) for dx, dy in ring[:nc]]
        xy += [(20.0, 23.5), (16.0, 20.0)]  # within R but not in contact
        pos, alive, partner = _scene(micro_cfg, xy)
        n = pos.shape[0]
        heading = np.zeros(n, np.float32)
        heading[0] = 0.7
        nbr = _neighbors(pos, alive, micro_cfg)
        fs = []
        for t in (1.0, 0.0):
            act = _actions(n, thrust=np.where(np.arange(n) == 0, t, 0.0))
            f = physics.forces(
                mx.array(pos), mx.array(heading), act, nbr, mx.array(alive), mx.array(partner), micro_cfg
            )
            fs.append(np.array(f)[0])
        got = fs[0] - fs[1]
        att = max(0.0, 1.0 - nc / ph.thrust_contact_limit)
        expect = ph.thrust_max * att * np.array([math.cos(0.7), math.sin(0.7)])
        assert np.allclose(got, expect, atol=1e-7), (nc, got, expect)
    # A dead neighbour in contact does not count (and does not repel).
    pos, alive, partner = _scene(micro_cfg, [(20.0, 20.0), (21.5, 20.0)])
    n = pos.shape[0]
    nbr = _neighbors(pos, alive, micro_cfg)
    survivor = alive.copy()
    survivor[1] = False
    f = physics.forces(
        mx.array(pos), mx.zeros(n), _actions(n, thrust=1.0 * (np.arange(n) == 0)), nbr, mx.array(survivor),
        mx.array(partner), micro_cfg,
    )  # fmt: skip
    assert np.allclose(np.array(f)[0], [ph.thrust_max, 0.0], atol=1e-8) and np.all(np.array(f)[1:] == 0)


def test_overlapping_pair_separates(micro_cfg: Config) -> None:
    pos, alive, partner = _scene(micro_cfg, [(20.0, 20.0), (21.0, 20.3)])
    n = pos.shape[0]
    heading = np.zeros(n, np.float32)
    act = _actions(n)
    two_r = 2 * micro_cfg.physics.radius
    d0 = _dist(pos[0], pos[1], micro_cfg.world.size)
    com0 = pos[:2].mean(0)
    for _ in range(150):
        (pos, _vel, heading, _nclip), f = _step(micro_cfg, pos, heading, alive, partner, act)
        assert np.array_equal(f[0], -f[1])  # repulsion is antisymmetric bitwise
        d = _dist(pos[0], pos[1], micro_cfg.world.size)
        assert d >= d0 - 1e-6
        d0 = d
    assert d >= two_r * (1 - 1e-5)
    assert np.allclose(pos[:2].mean(0), com0, atol=1e-5)


@pytest.mark.parametrize("c", [0.0, 1.0, -0.4])
def test_bonded_pair_relaxes_to_rest_length(micro_cfg: Config, c: float) -> None:
    bc = micro_cfg.bonds
    pos, alive, partner = _scene(micro_cfg, [(20.0, 20.0), (23.6, 21.0)], [(0, 1)])
    n = pos.shape[0]
    heading = np.zeros(n, np.float32)
    act = _actions(n, contract=np.where(alive, c, 0.0))
    for _ in range(400):
        (pos, _, heading, _), f = _step(micro_cfg, pos, heading, alive, partner, act)
        assert np.array_equal(f[0], -f[1])
    rest = bc.rest_length * (1 + bc.contract_gain * c)
    two_r = 2 * micro_cfg.physics.radius
    # Below 2r the spring settles against repulsion: k_s(d − ℓ) = k_r(2r − d).
    k_s, k_r = bc.k_spring, micro_cfg.physics.k_repel
    expect = rest if rest >= two_r else (k_s * rest + k_r * two_r) / (k_s + k_r)
    assert abs(_dist(pos[0], pos[1], micro_cfg.world.size) - expect) < 1e-4


def _crawl(cfg: Config, grip_on: bool, steps: int = 500, period: int = 100) -> float:
    """A two-agent body along x with a square-wave muscle; returns the centre-of-mass displacement."""
    L = cfg.world.size
    pos, alive, partner = _scene(cfg, [(20.0, 24.0), (22.0, 24.0)], [(0, 1)])
    n = pos.shape[0]
    heading = np.zeros(n, np.float32)
    start = pos[:2].astype(np.float64).copy()
    disp = np.zeros((2, 2))
    nbr = None
    for t in range(steps):
        extend = (t % period) < period // 2
        contract = np.where(alive, 1.0 if extend else -1.0, 0.0)
        grip = np.zeros(n)
        if grip_on:  # rear (0) anchors while extending, front (1) anchors while contracting
            grip[0 if extend else 1] = 1.0
        act = _actions(n, contract=contract, grip=grip)
        nbr = _neighbors(pos, alive, cfg)
        sv, p_ = mx.array(alive), mx.array(partner)
        f = physics.forces(mx.array(pos), mx.array(heading), act, nbr, sv, p_, cfg)
        new_pos, vel, h2, _ = physics.integrate(
            mx.array(pos), mx.array(heading), f, act.grip, act.turn, sv, cfg
        )
        disp += np.array(vel)[:2]
        pos, heading = np.array(new_pos), np.array(h2)
    end = start + disp
    assert np.allclose(_min_image((pos[:2] - start).astype(np.float32), L), disp, atol=1e-3)
    return float(np.linalg.norm(end.mean(0) - start.mean(0)))


def test_contraction_without_grip_does_not_move(micro_cfg: Config) -> None:
    """F11: springs and repulsion are internal, so with uniform drag the centre of mass stays put."""
    assert _crawl(micro_cfg, grip_on=False) < 1e-3


def test_phase_locked_grip_crawls(micro_cfg: Config) -> None:
    """F11: grip phase-locked to contraction (inchworm) moves the body more than 1 u in 500 steps."""
    assert _crawl(micro_cfg, grip_on=True) > 1.0


def test_integrate_clip_wrap_heading_and_dead_rows(micro_cfg: Config) -> None:
    ph = micro_cfg.physics
    L = micro_cfg.world.size
    n = 8
    pos = np.array(
        [[47.9, 10.0], [0.05, 0.05], [10.0, 10.0], [10.0, 10.0], [5.0, 5.0], [1e-8, 47.99], [3, 3], [4, 4]],
        np.float32,
    )
    heading = np.array([3.1, -3.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], np.float32)
    turn = np.array([0.3, -0.3, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0], np.float32)
    force = np.zeros((n, 2), np.float32)
    force[0] = (0.3, 0.0)  # crosses x = L
    force[1] = (-0.2, -0.2)  # crosses both zero seams
    force[2] = (3.0, 4.0)  # |v| = 5 > v_max: clipped
    force[3] = (2.5, 0.0)  # with grip 1, γ = 5: |v| = 0.5 = v_max, not clipped
    force[4] = (9.0, 9.0)  # dead: not integrated, not counted
    force[5] = (-1e-8, 1e-2)
    grip = np.zeros(n, np.float32)
    grip[3] = 1.0
    survivor = np.ones(n, bool)
    survivor[4] = False
    p2, v2, h2, nclip = physics.integrate(
        mx.array(pos), mx.array(heading), mx.array(force), mx.array(grip), mx.array(turn), mx.array(survivor),
        micro_cfg,
    )  # fmt: skip
    p2, v2, h2 = np.array(p2), np.array(v2), np.array(h2)
    assert nclip.dtype == mx.int32 and int(nclip) == 1
    assert np.all((p2 >= 0) & (p2 < L)) and np.all(np.isfinite(p2))
    assert np.allclose(p2[0], [0.2, 10.0], atol=1e-5) and np.allclose(p2[1], [L - 0.15, L - 0.15], atol=1e-5)
    assert np.allclose(v2[2], [0.3, 0.4], atol=1e-6)
    assert np.linalg.norm(v2, axis=1).max() <= ph.v_max * (1 + 1e-6)
    assert np.allclose(v2[3], [0.5, 0.0]) and np.allclose(p2[3], [10.5, 10.0])
    assert np.all(v2[4] == 0) and np.array_equal(p2[4], pos[4]) and h2[4] == heading[4]
    assert np.all((h2 >= -np.pi) & (h2 < np.pi))
    assert np.isclose(h2[0], 3.4 - 2 * np.pi, atol=1e-5) and np.isclose(h2[1], -3.4 + 2 * np.pi, atol=1e-5)
    assert np.allclose(v2[6:], 0) and np.array_equal(p2[6:], pos[6:])


def _hex_cluster(cfg: Config, rings: int, spacing: float, rs: np.random.Generator):
    pts = []
    for q in range(-rings, rings + 1):
        for r in range(-rings, rings + 1):
            if abs(q + r) <= rings:
                pts.append((spacing * (q + r / 2), spacing * r * math.sqrt(3) / 2))
    xy = np.array(pts) + cfg.world.size / 2 + rs.normal(0, 0.05, (len(pts), 2))
    bonds_: list[tuple[int, int]] = []
    deg = np.zeros(len(pts), int)
    d = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    edges = [(a, b) for a in range(len(pts)) for b in range(a + 1, len(pts))]
    for i, j in sorted(edges, key=lambda e: d[e]):
        if d[i, j] < spacing * 1.2 and deg[i] < B and deg[j] < B:
            bonds_.append((i, j))
            deg[i] += 1
            deg[j] += 1
    return [tuple(p) for p in xy], bonds_


def _potential(cfg: Config, pos: np.ndarray, alive: np.ndarray, partner: np.ndarray) -> float:
    """Elastic energy of springs (relaxed rest length) and contacts, in float64."""
    L, k_s, k_r = cfg.world.size, cfg.bonds.k_spring, cfg.physics.k_repel
    two_r, l0 = 2 * cfg.physics.radius, cfg.bonds.rest_length
    idx = np.flatnonzero(alive)
    p = pos[idx].astype(np.float64)
    off = p[None] - p[:, None]
    off -= L * np.round(off / L)
    d = np.linalg.norm(off, axis=-1)
    iu = np.triu_indices(len(idx), 1)
    dd = d[iu]
    e = 0.5 * k_r * np.sum(np.where(dd < two_r, (two_r - dd) ** 2, 0.0))
    for a, row in enumerate(idx):
        for j in partner[row]:
            if j > row:
                b = int(np.flatnonzero(idx == j)[0])
                e += 0.5 * k_s * (d[a, b] - l0) ** 2
    return float(e)


def test_hex_cluster_is_stable_and_dissipative(micro_cfg: Config) -> None:
    """At default stiffness a compressed bonded hexagonal cluster stays finite and never gains energy."""
    rs = np.random.default_rng(3)
    xy, bonds_ = _hex_cluster(micro_cfg, rings=2, spacing=1.7, rs=rs)  # 19 agents, compressed
    pos, alive, partner = _scene(micro_cfg, xy, bonds_)
    n = pos.shape[0]
    heading = np.zeros(n, np.float32)
    act = _actions(n)
    e_prev = _potential(micro_cfg, pos, alive, partner)
    e0 = e_prev
    for _ in range(120):
        (pos, _vel, heading, nclip), _ = _step(micro_cfg, pos, heading, alive, partner, act)
        assert np.all(np.isfinite(pos)) and int(nclip) == 0
        e = _potential(micro_cfg, pos, alive, partner)
        assert e <= e_prev + 1e-6
        e_prev = e
    assert e_prev < 0.5 * e0


def test_stability_number_rule_rejects_stiff_configs(micro_cfg: Config) -> None:
    assert stability_number(micro_cfg) == pytest.approx(2 * (B * 0.05 + 8 * 0.08) / 1.0)
    with pytest.raises(ConfigError, match="stability_number"):
        apply_overrides(micro_cfg, ["physics.k_repel=0.1"])


def test_forces_and_integrate_compile(micro_cfg: Config) -> None:
    rs = np.random.default_rng(5)
    xy, bonds_ = _hex_cluster(micro_cfg, rings=1, spacing=1.9, rs=rs)
    pos, alive, partner = _scene(micro_cfg, xy, bonds_)
    n = pos.shape[0]
    nbr = _neighbors(pos, alive, micro_cfg)
    act = _actions(n, thrust=rs.uniform(-1, 1, n), contract=rs.uniform(-1, 1, n), grip=rs.uniform(0, 1, n),
                   turn=rs.uniform(-0.3, 0.3, n))  # fmt: skip
    heading = mx.array(rs.uniform(-3, 3, n).astype(np.float32))

    def f(pos, heading, act, nbr, sv, partner):
        F = physics.forces(pos, heading, act, nbr, sv, partner, micro_cfg)
        return physics.integrate(pos, heading, F, act.grip, act.turn, sv, micro_cfg)

    args = (mx.array(pos), heading, act, nbr, mx.array(alive), mx.array(partner))
    for a, b in zip(f(*args), mx.compile(f)(*args), strict=True):
        assert np.allclose(np.array(a), np.array(b), atol=1e-6)
