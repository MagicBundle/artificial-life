"""Bonds that pay, the observer PR B3 (DESIGN §40.6): the snapshot statistics on constructed bond graphs (the
numpy references), the interval keys and the ledger identity of the flow matrix."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from alife import featobs, runner
from alife.config import apply_overrides, load_config
from alife.layout import B
from alife.metrics import snapshot_metrics
from alife.runio import RunDir
from alife.state import init_state, to_numpy

# --- the numpy statistics on constructed graphs -------------------------------------------------------------


def _path(roles: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """A path graph over len(roles) rows: ``partner`` (n, B) with slots 0 (next) and 1 (previous)."""
    n = len(roles)
    partner = np.full((n, B), -1, np.int32)
    for i in range(n - 1):
        partner[i, 0] = i + 1
        partner[i + 1, 1] = i
    return partner, np.array(roles)


def test_assortativity_of_an_alternating_path_is_minus_one_and_of_blocks_is_positive() -> None:
    partner, role = _path([1, 2, 1, 2, 1, 2])
    alive = np.ones(6, bool)
    ii, bb = featobs.edges(partner, alive)
    jj = partner[ii, bb]
    assert featobs.assortativity(role[ii], role[jj]) == pytest.approx(-1.0)
    partner, role = _path([1, 1, 1, 2, 2, 2])
    ii, bb = featobs.edges(partner, alive)
    assert featobs.assortativity(role[ii], role[partner[ii, bb]]) > 0.3
    assert featobs.assortativity(np.zeros(0, int), np.zeros(0, int)) is None
    assert featobs.assortativity(np.array([1, 1]), np.array([1, 1])) is None  # one category: undefined


def test_edges_lists_each_bond_once_from_the_lower_row_and_skips_dead_ends() -> None:
    partner, _ = _path([0, 0, 0, 0])
    alive = np.array([1, 1, 0, 1], bool)
    ii, bb = featobs.edges(partner, alive)
    assert (ii.tolist(), partner[ii, bb].tolist()) == ([0], [1])


def test_gini() -> None:
    assert featobs.gini(np.ones(10)) == pytest.approx(0.0, abs=1e-12)
    assert featobs.gini(np.array([0, 0, 0, 12.0])) == pytest.approx(0.75)  # (n - 1) / n for one holder
    assert featobs.gini(np.zeros(4)) is None and featobs.gini(np.zeros(0)) is None


def test_the_flow_matrix_sums_to_the_received_flow_and_orients_sender_to_receiver() -> None:
    partner, role = _path([1, 2, 0])  # forager - guard - generalist
    flow = np.zeros((3, B), np.int64)
    flow[1, 1] = 40  # the guard (row 1) received 40 from row 0 through slot 1 (partner 0): forager -> guard
    flow[0, 0] = -40  # and row 0 sent 40 (net negative at the sender)
    flow[2, 1] = 7  # row 2 received 7 from row 1 (guard -> generalist)
    flow[1, 0] = -7
    alive = np.ones(3, bool)
    m = featobs.flow_matrix(partner, role, flow, alive)
    assert m[1, 2] == 40 and m[2, 0] == 7 and m.sum() == 47  # [sender role, receiver role]
    assert not featobs.flow_matrix(
        partner, role, flow, np.array([1, 0, 1], bool)
    ).any()  # dead ends carry none


def test_body_role_entropy_and_its_null() -> None:
    # two bodies of 4: one mixed (1,2,1,2), one pure (1,1,1,1)
    role = np.array([1, 2, 1, 2, 1, 1, 1, 1, 0])
    labels = np.array([0, 0, 0, 0, 4, 4, 4, 4, 8])
    alive = np.ones(9, bool)
    ent, rows, body = featobs.role_entropy(role, labels, alive)
    assert ent == pytest.approx(0.5)  # (1 bit * 4 + 0 * 4) / 8; the singleton is not a body
    assert rows.tolist() == list(range(8)) and body.tolist() == [0] * 4 + [1] * 4
    z = featobs.role_diff_z(role[rows], body, ent, seed=1)
    # the observed mix is as even as it can be inside body 0 and perfectly pure in body 1: entropy 0.5 is
    # below a random split of {1,2,1,2,1,1,1,1} (mean about 0.74), so roles are clustered: z > 0
    assert z is not None and z > 0
    assert featobs.role_diff_z(role[rows], body, ent, seed=1) == z  # fixed seed: reproducible
    assert featobs.role_diff_z(np.ones(8, int), body, 0.0, seed=1) is None  # no spread in the null
    only_small = featobs.role_entropy(role, np.arange(9), alive)
    assert only_small[0] is None  # no body of size >= 4


def test_pearson_and_circular_helpers_are_shared_with_arms() -> None:
    assert featobs.pearson(np.array([1.0, 2, 3]), np.array([3.0, 2, 1])) == pytest.approx(-1.0)


# --- a real world -------------------------------------------------------------------------------------------


def test_the_snapshot_keys_of_a_bonds_world_are_valid_json(tmp_path: Path) -> None:
    cfg = apply_overrides(
        load_config(
            preset="tiny3d",
            overrides=["brain.weight_dtype=float32", "roles.enabled=true", "bonds.share_mode=1"],
        ),
        ["run.metrics_every=20", "run.checkpoint_every=40", "run.heavy_metrics_every=20"],
    )
    rd = RunDir.create(tmp_path / "run", cfg)
    assert runner.run(rd, cfg, init_state(cfg), 60, quiet=True).status == "done"
    last = rd.metrics_rows()[-1]
    json.dumps(last, allow_nan=False)
    for key in (
        "bond_role_frac_gen", "bond_role_frac_forager", "bond_role_frac_guard", "bond_role_assort_r",
        "bond_fg_edge_frac", "bond_fg_edge_expect", "bond_flow_by_role", "bond_net_giver_frac",
        "bond_flow_gini",
        "bond_cheat_idx", "bond_nonkin_flow_share", "bond_shield_cover", "bond_body_role_entropy",
        "bond_role_diff_z", "bond_shielded_frac", "bond_flux",
    ):  # fmt: skip
        assert key in last, key
    shares = [last[f"bond_role_frac_{r}"] for r in ("gen", "forager", "guard")]
    assert sum(shares) == pytest.approx(1.0)
    assert np.asarray(last["bond_flow_by_role"]).shape == (3, 3)


def test_the_flow_matrix_of_a_real_state_is_bounded_by_the_ledger_flow() -> None:
    cfg = load_config(preset="tiny3d", overrides=["brain.weight_dtype=float32", "bonds.share_mode=1"])
    s = init_state(cfg)
    n = s.alive.shape[0]
    # hand-set one live bond with a net flow, as the step would leave it
    partner = np.full((n, B), -1, np.int32)
    partner[0, 0], partner[1, 0] = 1, 0
    rev = np.full((n, B), -1, np.int32)
    rev[0, 0], rev[1, 0] = 0, 0
    flow = np.zeros((n, B), np.int16)
    flow[1, 0], flow[0, 0] = 25, -25
    s = s._replace(partner=mx.array(partner), rev=mx.array(rev), bond_flow=mx.array(flow))
    snap = snapshot_metrics(s, cfg, heavy=False)
    assert np.asarray(snap["bond_flow_by_role"]).sum() == 25
    assert snap["bond_net_giver_frac"] == pytest.approx(0.5)  # one of the two bonded rows is a net giver
    assert to_numpy(s.bond_flow).sum() == 0  # the net flow of a closed bond is conserved
