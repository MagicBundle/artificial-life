"""§27.2 the 3D neighbour descriptor: the seven 2D features with the offset in the yaw-and-pitch body frame,
then ``NBR_DZ`` (the up component / R)."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from alife import layout, sensing
from alife.config import Config, dims
from alife.spatial import Neighbors
from alife.state import State, State3
from helpers import as_3d, make_scene

PM = layout.PITCH_MAX


@pytest.fixture(scope="module")
def cfg3(micro_cfg: Config) -> Config:
    return as_3d(micro_cfg)


def _neighbors(s: State | State3, cfg: Config) -> Neighbors:
    """Brute-force kNN (alive, within R, nearest first, ties by row); minimum image on x and y only."""
    d = dims(cfg)
    pos, alive = np.array(s.pos, np.float64), np.array(s.alive)
    n, k, R, L = d.n, d.k, cfg.sensing.radius, cfg.world.size
    idx = np.tile(np.arange(n, dtype=np.int32)[:, None], (1, k))
    valid = np.zeros((n, k), bool)
    off = np.zeros((n, k, d.nd), np.float32)
    dist = np.full((n, k), R, np.float32)
    live = np.flatnonzero(alive)
    for i in live:
        o = pos[live] - pos[i]
        o[:, :2] -= L * np.round(o[:, :2] / L)
        dd = np.sqrt((o * o).sum(1))
        cand = sorted((dd[c], int(live[c]), c) for c in range(len(live)) if live[c] != i and dd[c] < R)
        for slot, (dj, j, c) in enumerate(cand[:k]):
            idx[i, slot], valid[i, slot], off[i, slot], dist[i, slot] = j, True, o[c], dj
    zero = mx.array(0, mx.int32)
    return Neighbors(mx.array(idx), mx.array(valid), mx.array(off), mx.array(dist), zero, zero)


def _agents(rng: np.random.Generator, n: int, z: np.ndarray | None) -> list[dict]:
    """``n`` agents in a 14×14 patch (so most see neighbours) with random heading, tag, energy, msg; 3D when
    ``z`` is given (then also a random pitch)."""
    out = []
    for i in range(n):
        a = {
            "pos": [*rng.uniform(17.0, 31.0, 2), *([] if z is None else [z[i]])],
            "heading": rng.uniform(-math.pi, math.pi),
            "tag": rng.normal(0, 0.3, layout.T),
            "energy": int(rng.integers(100, 5000)),
            "msg": rng.uniform(-1, 1, layout.MSG),
        }
        out.append(a)
    return out


def test_descriptor_3d_is_8_wide_and_level_scenes_match_2d(micro_cfg: Config, cfg3: Config) -> None:
    """All agents at one height with pitch 0: the first seven features equal the 2D descriptor of the same
    scene bit for bit (the 3D rotation reduces to the 2D one), and NBR_DZ is 0."""
    rng = np.random.default_rng(0)
    ag2 = _agents(rng, 30, None)
    bonds = [(0, 1), (2, 3), (4, 5)]
    for i, j in bonds:  # partners side by side, so each lists the other
        ag2[j]["pos"] = [ag2[i]["pos"][0] + 0.8, ag2[i]["pos"][1] + 0.3]
    ag3 = [{**a, "pos": [*a["pos"], 12.0], "pitch": 0.0} for a in ag2]
    s2, s3 = make_scene(micro_cfg, ag2, bonds=bonds), make_scene(cfg3, ag3, bonds=bonds)
    assert isinstance(s3, State3)
    nbr2, nbr3 = _neighbors(s2, micro_cfg), _neighbors(s3, cfg3)
    assert np.array_equal(np.array(nbr2.idx), np.array(nbr3.idx)) and np.array(nbr2.valid).sum() > 100
    d2 = np.array(sensing.neighbor_descriptors(s2, nbr2, micro_cfg))
    d3 = np.array(sensing.neighbor_descriptors(s3, nbr3, cfg3))
    lay = dims(cfg3).lay
    assert d3.shape == (dims(cfg3).n, dims(cfg3).k, lay.NF) and lay.NF == 8 and d3.dtype == np.float32
    assert np.array_equal(d3[..., : layout.NF], d2)
    assert np.all(d3[..., lay.NBR_DZ] == 0.0)
    assert d2[..., layout.NBR_BONDED].sum() == 2 * len(bonds)  # the bonded feature is exercised


def test_descriptor_3d_body_frame(cfg3: Config) -> None:
    """Random yaw and pitch in a 3D scene: (dx, dy, dz) are the offset's (fwd, lat, up) components / R, the
    other features are the 2D ones, invalid lanes are zero and every feature is in [−1, 1]."""
    rng = np.random.default_rng(1)
    ag = _agents(rng, 40, rng.uniform(6.0, 18.0, 40))
    for a in ag:
        a["pitch"] = rng.uniform(-PM, PM)
    s = make_scene(cfg3, ag, bonds=[(0, 1), (6, 7)])
    nbr = _neighbors(s, cfg3)
    desc = np.array(sensing.neighbor_descriptors(s, nbr, cfg3))
    lay, R = dims(cfg3).lay, cfg3.sensing.radius
    valid = np.array(nbr.valid)
    assert valid.sum() > 100
    th, ph = np.array(s.heading, np.float64), np.array(s.pitch, np.float64)
    fwd = np.stack([np.cos(ph) * np.cos(th), np.cos(ph) * np.sin(th), np.sin(ph)], -1)[:, None, :]
    lat = np.stack([-np.sin(th), np.cos(th), np.zeros_like(th)], -1)[:, None, :]
    up = np.stack([-np.sin(ph) * np.cos(th), -np.sin(ph) * np.sin(th), np.cos(ph)], -1)[:, None, :]
    off = np.array(nbr.off, np.float64)
    want = np.stack([(off * fwd).sum(-1), (off * lat).sum(-1), (off * up).sum(-1)], -1) / R
    got = desc[..., [layout.NBR_DX, layout.NBR_DY, lay.NBR_DZ]]
    np.testing.assert_allclose(got[valid], want[valid], atol=1e-6)
    assert np.all(desc[..., layout.NBR_PRESENT] == valid)
    assert np.all(desc[~valid] == 0.0)
    assert np.all(np.abs(desc) <= 1.0)
    # The other features do not depend on the frame: the same scene seen level gives the same values.
    level = s._replace(pitch=mx.zeros_like(s.pitch))
    flat = np.array(sensing.neighbor_descriptors(level, nbr, cfg3))
    others = [layout.NBR_PRESENT, layout.NBR_KIN, layout.NBR_RELE, layout.NBR_MSG0, layout.NBR_BONDED]
    assert np.array_equal(flat[..., others], desc[..., others])


def test_descriptor_3d_named_case(cfg3: Config) -> None:
    """Nose 45° up, facing +x: a neighbour 3 straight above is ahead-and-up, one 3 ahead is ahead-and-down."""
    s = make_scene(
        cfg3,
        [
            {"pos": [10.0, 10.0, 10.0], "heading": 0.0, "pitch": math.pi / 4},
            {"pos": [10.0, 10.0, 13.0]},
            {"pos": [13.5, 10.0, 10.0]},
        ],
    )
    desc = np.array(sensing.neighbor_descriptors(s, _neighbors(s, cfg3), cfg3))
    lay, c = dims(cfg3).lay, math.cos(math.pi / 4)
    cols = [layout.NBR_DX, layout.NBR_DY, lay.NBR_DZ]
    np.testing.assert_allclose(desc[0, 0, cols], [3 * c / 6, 0.0, 3 * c / 6], atol=1e-6)  # j = 1, distance 3
    np.testing.assert_allclose(desc[0, 1, cols], [3.5 * c / 6, 0.0, -3.5 * c / 6], atol=1e-6)  # j = 2


def test_descriptor_3d_compiles(cfg3: Config) -> None:
    rng = np.random.default_rng(2)
    ag = _agents(rng, 30, rng.uniform(6.0, 18.0, 30))
    for a in ag:
        a["pitch"] = rng.uniform(-PM, PM)
    s = make_scene(cfg3, ag)
    nbr = _neighbors(s, cfg3)
    eager = np.array(sensing.neighbor_descriptors(s, nbr, cfg3))
    comp = np.array(mx.compile(lambda s_, n_: sensing.neighbor_descriptors(s_, n_, cfg3))(s, nbr))
    np.testing.assert_allclose(comp, eager, atol=1e-6)
