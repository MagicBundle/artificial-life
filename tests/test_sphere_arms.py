"""Arms on the shell (DESIGN §42 on §46.3; sphere series S3): the contest between two neighbours.

The contest and the traits are neighbour- and gene-based and have no geometry of their own; the box suites
run on the sphere through ``tests/test_arms_step.py`` (parameter ``"s"``: null equivalence, the ledger with
real traits, determinism, the staged pipeline, the armour's slowing). Here an attack between two agents at
known points of the sphere (the middle, the pole, the antimeridian) is reduced by the victim's armour and by
a key that does not fit its lock, as the Tullock form says, and the energy the attacker gains is booked in
the ledger.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

import helpers
from alife import arms, invariants, layout
from alife.config import Config, dims, load_config
from alife.state import LedgerIdx, init_state, to_numpy, total_energy
from alife.step import make_step

ARMS = ["arms.enabled=true", "arms.kappa=2", "arms.a0=1", "arms.key_floor=0.2"]


def cfg_of(*extra: str) -> Config:
    cfg = load_config(preset="tiny3s", overrides=["brain.weight_dtype=float32", *ARMS, *extra])
    return cfg


def duel(cfg: Config, lat: float, lon: float, *, atk: float, armour: float, key: float, lock: float):
    """Row 0 attacks row 1, a chord 1.5 away along the local east; nobody else is alive. Row 1 is passive."""
    d = dims(cfg)
    s = init_state(cfg)
    la, lo = math.radians(lat), math.radians(lon)
    u = np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])
    east = np.cross([0.0, 0.0, 1.0], u)
    east = east / np.linalg.norm(east)
    p0 = u * (cfg.world.radius + 6.0)
    p1 = p0 + 1.5 * east
    pos = np.zeros((d.n, 3), np.float32)
    pos[:] = np.array([0.0, 0.0, -(cfg.world.radius + 6.0)], np.float32)
    pos[0], pos[1] = p0, p1
    hdg = np.zeros((d.n, 3), np.float32)
    hdg[:] = np.array([1.0, 0.0, 0.0], np.float32)
    hdg[0] = hdg[1] = east
    alive = np.zeros(d.n, bool)
    alive[:2] = True
    al = mx.array(alive)
    energy = mx.where(al, 200_000, 0).astype(mx.int32)
    tr = np.array(to_numpy(s.arms))
    tr[0, arms.ATK], tr[1, arms.ARM] = atk, armour
    tr[0, arms.KEY], tr[1, arms.LOCK] = key, lock
    s = s._replace(
        pos=mx.array(pos),
        hdg=mx.array(hdg),
        alive=al,
        energy=energy,
        uid=mx.where(al, s.uid, -1),
        arms=mx.array(tr),
        W1=mx.zeros_like(s.W1),
        W2=mx.zeros_like(s.W2),
        birth_step=mx.full(s.birth_step.shape, -100, mx.int32),
    )
    tot = total_energy(s.energy, s.alive, s.resource, cfg)
    s = s._replace(total=tot, total0=tot)
    s = helpers.set_output_bias(s, cfg, {layout.OUT_ATTACK: 4.0})
    w2 = to_numpy(s.W2).copy()
    w2[1, d.h, layout.OUT_ATTACK] = -4.0  # the victim does not fight back
    s = s._replace(W2=mx.array(w2))
    mx.eval(s)
    return s


def gain(cfg: Config, s) -> int:
    s1, out = make_step(cfg, compile=False)(s)
    mx.eval(s1, out)
    assert invariants.full_check(s1, cfg, s, out) == []
    assert helpers.ledger_net(to_numpy(out.ledger), cfg) == int(to_numpy(s1.total) - to_numpy(s.total))
    return int(to_numpy(out.ledger)[LedgerIdx.FLOW_ATTACK_GAIN])


@pytest.mark.parametrize(
    ("name", "lat", "lon"), [("mid", 20.0, 40.0), ("pole", 89.99, 5.0), ("seam", 3.0, 179.99)]
)
def test_armour_and_a_wrong_key_reduce_the_energy_an_attack_takes(name, lat, lon) -> None:
    cfg = cfg_of()
    bare = gain(cfg, duel(cfg, lat, lon, atk=0.0, armour=0.0, key=0.3, lock=0.3))
    armoured = gain(cfg, duel(cfg, lat, lon, atk=0.0, armour=1.0, key=0.3, lock=0.3))
    wrong_key = gain(cfg, duel(cfg, lat, lon, atk=0.0, armour=0.0, key=0.0, lock=0.5))
    assert bare > 0
    # r = κA/(κA + a0 + ATK) = 2/3 for A = 1, ATK = 0: the demand is scaled by 1/3 (the integer arithmetic of
    # the damage allows a few quanta of slack); the key at a half-turn from the lock scales by key_floor 0.2
    assert armoured == pytest.approx(bare / 3.0, rel=0.05, abs=60)
    assert wrong_key == pytest.approx(bare * 0.2, rel=0.05, abs=60)
    assert 0 < armoured < bare and 0 < wrong_key < bare


def test_the_attackers_strength_weakens_the_armour_on_the_sphere() -> None:
    cfg = cfg_of()
    weak = gain(cfg, duel(cfg, 10.0, 80.0, atk=0.0, armour=1.0, key=0.3, lock=0.3))
    strong = gain(cfg, duel(cfg, 10.0, 80.0, atk=1.0, armour=1.0, key=0.3, lock=0.3))
    assert strong > weak  # r = 2/(2 + 1 + ATK): the stronger attacker is repelled less
