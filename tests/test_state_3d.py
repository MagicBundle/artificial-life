"""§27.4 the 3D state: ``State3`` (State's 47 fields, then pitch and buoy_cap), its schema, founders uniform
in the volume, dead rows on the floor, (de)serialisation, digests, checkpoints and ``make_scene``."""

from __future__ import annotations

import math
import random
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from alife import cli, fields, layout, ops
from alife.config import Config, dims
from alife.runio import CheckpointError, checkpoint_metadata, load_checkpoint, write_state_file
from alife.state import (
    AGENT_FIELDS,
    AGENT_FIELDS_3D,
    ENV_FIELDS,
    EXTRA_3D_FIELDS,
    Env,
    State,
    State3,
    agent_fields,
    expected_layout,
    init_state,
    legacy_state_digest,
    state_cls,
    state_digest,
    state_from_arrays,
    state_to_arrays,
    to_numpy,
)
from helpers import as_3d, make_scene

PITCH_MAX_F32 = np.float32(layout.PITCH_MAX)


@pytest.fixture(scope="module")
def cfg3(micro_cfg: Config) -> Config:
    """``micro`` as a 3D world: N 64, L 48, height 24 (FZ 6, FG 12), with an active buoyancy cap."""
    return as_3d(micro_cfg, physics={"buoyancy_max": 0.05}, mutation={"buoy_init": 0.25, "buoy_sigma": 0.02})


@pytest.fixture(scope="module")
def tiny3(tiny_cfg: Config) -> Config:
    return as_3d(tiny_cfg)


def _host(s: State | State3) -> dict[str, np.ndarray]:
    return {k: to_numpy(v) for k, v in s._asdict().items()}


# --- the type ---------------------------------------------------------------------------------------------


def test_state3_fields(micro_cfg: Config, cfg3: Config) -> None:
    assert len(State._fields) == 47  # State itself is unchanged
    assert State3._fields[:47] == State._fields and State3._fields[47:] == EXTRA_3D_FIELDS
    assert EXTRA_3D_FIELDS == ("pitch", "buoy_cap") and AGENT_FIELDS_3D == AGENT_FIELDS + EXTRA_3D_FIELDS
    assert state_cls(micro_cfg) is State and agent_fields(micro_cfg) == AGENT_FIELDS
    assert state_cls(cfg3) is State3 and agent_fields(cfg3) == AGENT_FIELDS_3D
    s = init_state(cfg3)
    assert isinstance(s.env, Env) and tuple(f"env_{n}" for n in s.env._fields) == ENV_FIELDS
    assert s.env.cat_center.shape == (2,)  # the catastrophe is a vertical column: its centre stays 2D


def test_schema_3d(cfg3: Config) -> None:
    spec = expected_layout(cfg3)
    assert tuple(spec) == State3._fields
    d, f32 = dims(cfg3), mx.float32
    assert spec["pos"] == spec["vel"] == ((64, 3), f32)
    assert spec["pitch"] == spec["buoy_cap"] == ((64,), f32)
    assert spec["resource"] == ((6, 12, 12), mx.int32) and spec["chem"] == ((3, 6, 12, 12), f32)
    assert spec["W1"][0] == (64, d.w1_rows, d.h) == (64, 116, 32) and spec["W2"][0] == (64, 33, 39)
    s = init_state(cfg3)
    assert type(s) is State3
    for name, a in zip(State3._fields, s, strict=True):
        assert (tuple(a.shape), a.dtype) == spec[name], name


def test_state3_compiles(cfg3: Config) -> None:
    """F12 holds for State3: ``mx.compile`` takes and returns it."""
    s = init_state(cfg3)
    out = mx.compile(lambda t: t._replace(step=t.step + 1, pitch=t.pitch * 0.5))(s)
    assert type(out) is State3 and int(to_numpy(out.step)) == 1


# --- founders and dead rows -------------------------------------------------------------------------------


def test_founders_in_the_volume(tiny3: Config) -> None:
    d, cfg = dims(tiny3), tiny3
    n0, L = cfg.population.n_init, cfg.world.size
    a = _host(init_state(cfg))
    fo, de = np.arange(d.n) < n0, np.arange(d.n) >= n0
    pos = a["pos"]
    assert np.all((pos[:, :2] >= 0) & (pos[:, :2] < L))
    assert np.all((pos[:, 2] >= d.z_lo) & (pos[:, 2] <= d.z_hi))
    span = d.z_hi - d.z_lo
    z = pos[fo, 2]
    assert z.min() < d.z_lo + 0.1 * span and z.max() > d.z_hi - 0.1 * span  # uniform in z, not on a plane
    assert np.all((a["heading"] >= -np.pi) & (a["heading"] < np.pi))
    pitch = a["pitch"][fo]
    assert np.all(np.abs(a["pitch"]) <= PITCH_MAX_F32)
    assert pitch.min() < -0.8 * layout.PITCH_MAX and pitch.max() > 0.8 * layout.PITCH_MAX
    # Dead rows: on the floor at (0, 0, z_lo), at rest, facing 0 (every range invariant holds on them).
    assert np.array_equal(pos[de], np.tile(np.float32([0.0, 0.0, d.z_lo]), (de.sum(), 1)))
    assert np.all(a["vel"] == 0) and a["vel"].shape == (d.n, 3)
    assert np.all(a["heading"][de] == 0) and np.all(a["pitch"][de] == 0)
    assert np.all(a["buoy_cap"] == np.float32(cfg.mutation.buoy_init))
    ref_total = (
        a["energy"].astype(np.int64).sum() + n0 * cfg.energy.structure + a["resource"].astype(np.int64).sum()
    )
    assert a["total"] == a["total0"] == ref_total


def test_buoy_cap_starts_at_buoy_init(cfg3: Config) -> None:
    a = _host(init_state(cfg3))
    assert np.all(a["buoy_cap"] == np.float32(0.25))


def test_founders_deterministic_and_draw_from_the_2d_keys(tiny_cfg: Config, tiny3: Config) -> None:
    s3 = init_state(tiny3)
    assert state_digest(init_state(tiny3)) == state_digest(s3)
    reseeded = as_3d(tiny_cfg, run={"seed": 1})
    assert state_digest(init_state(reseeded)) != state_digest(s3)
    # The same InitStream keys: draws whose shape does not depend on dims are identical to the 2D world's.
    a2, a3 = _host(init_state(tiny_cfg)), _host(s3)
    for name in ("tag", "patch_pos", "patch_vel", "key", "uid", "clade", "energy", "mut_sigma"):
        assert np.array_equal(a2[name], a3[name]), name
    # The floor layer's initial resource is the 2D field, bit for bit (V[0] = 1).
    assert np.array_equal(a3["resource"][0], a2["resource"])


def test_initial_resource_follows_the_vertical_profile(tiny3: Config) -> None:
    s = init_state(tiny3)
    fert = fields.fertility(s.patch_pos, tiny3)
    want = ops.floor_i32(tiny3.fields.resource_cap * fert)
    assert np.array_equal(to_numpy(s.resource), to_numpy(want))
    per_layer = to_numpy(s.resource).astype(np.int64).sum(axis=(1, 2))
    assert np.all(np.diff(per_layer) < 0)  # the benthic band thins with height
    assert np.all(to_numpy(s.chem) == 0) and s.chem.shape == (layout.NCH, *dims(tiny3).field_shape)


# --- (de)serialisation, digests, checkpoints --------------------------------------------------------------


def test_arrays_round_trip_3d(cfg3: Config) -> None:
    s = init_state(cfg3)
    d = state_to_arrays(s)
    assert list(d) == list(State3._fields)
    items = list(d.items())
    random.Random(0).shuffle(items)
    for back in (state_from_arrays(dict(items)), state_from_arrays(dict(items), cfg3)):
        assert type(back) is State3 and state_digest(back) == state_digest(s)
        assert all(x is y for x, y in zip(s, back, strict=True))


def test_cross_dims_arrays_refused(micro_cfg: Config, cfg3: Config) -> None:
    d3 = state_to_arrays(init_state(cfg3))
    d2 = state_to_arrays(init_state(micro_cfg))
    with pytest.raises(ValueError, match=r"State fields: missing=\[\], extra=\['buoy_cap', 'pitch'\]"):
        state_from_arrays(d3, micro_cfg)
    with pytest.raises(ValueError, match=r"State3 fields: missing=\['buoy_cap', 'pitch'\], extra=\[\]"):
        state_from_arrays(d2, cfg3)
    no_env = {k: v for k, v in d3.items() if k not in ENV_FIELDS}
    with pytest.raises(ValueError, match="env_cat_radius"):  # the pre-director fill is for 2D files only
        state_from_arrays(no_env, cfg3)


def test_digests_3d(cfg3: Config) -> None:
    s = init_state(cfg3)
    base = state_digest(s)
    assert len(base) == 64
    assert state_digest(s._replace(pitch=s.pitch + 0.125)) != base
    assert state_digest(s._replace(buoy_cap=s.buoy_cap * 0.5)) != base
    with pytest.raises(TypeError, match="2D states only"):
        legacy_state_digest(s)


def test_checkpoint_round_trip_3d(micro_cfg: Config, cfg3: Config, tmp_path: Path) -> None:
    s = init_state(cfg3)
    path = tmp_path / "s3.safetensors"
    write_state_file(path, s, cfg3)
    assert checkpoint_metadata(path)["dims"] == "3"
    back = load_checkpoint(path, cfg3)
    assert type(back) is State3 and state_digest(back) == state_digest(s)
    with pytest.raises(CheckpointError, match="cross-dimension"):
        load_checkpoint(path, micro_cfg, check_hash=False)


# --- make_scene and inspect ---------------------------------------------------------------------------------


def test_make_scene_3d(cfg3: Config) -> None:
    d = dims(cfg3)
    s = make_scene(
        cfg3,
        agents=[
            {"pos": (1.0, 2.0, 3.0), "pitch": 0.5, "buoy_cap": 0.75},
            {"row": 10, "pos": (40.0, 40.0, d.z_hi), "heading": -1.0},
        ],
        bonds=[(0, 10)],
        resource=7,
    )
    assert type(s) is State3
    a = _host(s)
    assert np.flatnonzero(a["alive"]).tolist() == [0, 10]
    assert a["pos"][0].tolist() == [1.0, 2.0, 3.0] and a["pos"][10, 2] == np.float32(d.z_hi)
    dead = ~a["alive"]
    assert np.all(a["pos"][dead] == np.float32([0.0, 0.0, d.z_lo]))
    assert a["pitch"][0] == np.float32(0.5) and a["pitch"][10] == 0 and np.all(a["pitch"][dead] == 0)
    assert a["buoy_cap"][0] == np.float32(0.75) and a["buoy_cap"][10] == np.float32(0.25)
    assert a["resource"].shape == d.field_shape and np.all(a["resource"] == 7)
    assert a["chem"].shape == (layout.NCH, *d.field_shape)
    assert a["total"] == a["energy"].sum() + 2 * cfg3.energy.structure + 7 * d.n_cells
    for name, arr in zip(State3._fields, s, strict=True):
        assert arr.dtype == getattr(init_state(cfg3), name).dtype, name
    with pytest.raises(ValueError):
        make_scene(cfg3, agents=[{"pos": (1.0, 2.0)}])  # a 2D position in a 3D world


def test_agent_info_lists_the_3d_fields(cfg3: Config) -> None:
    s = make_scene(cfg3, agents=[{"pos": (1.0, 2.0, 3.0), "pitch": -0.25}])
    info = cli._agent_info(0, s, cfg3, None)
    assert info is not None and set(EXTRA_3D_FIELDS) <= set(info["fields"])
    assert info["fields"]["pos"] == [1.0, 2.0, 3.0] and math.isclose(info["fields"]["pitch"], -0.25)
