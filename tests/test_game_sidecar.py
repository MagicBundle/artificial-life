"""§28.5/§28.6, §29.9/§29.10 the host observer, its metrics block and the sidecar (G1, G2): round trip through
``RunDir``, the fallback to fresh observers on any mismatch, resumed labels, social state and social log equal
to uninterrupted ones, pruning and truncation, and the ``observers`` block of the metrics rows.

The module-scoped fixtures cannot use ``monkeypatch``, so they pass ``social=SOC15`` (snapshots every 15
steps) explicitly; function-scoped tests set the module default with ``social_every``."""

from __future__ import annotations

import dataclasses
import io
import json
import shutil
import signal
import zipfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from alife import cli, runner
from alife.config import Config, apply_overrides, dims
from alife.game import social as soc
from alife.game.observe import EVAL_BIT, OBS_CAP, ObsState, evaluate, init_obs, obs_in, update
from alife.game.observer import DEFAULT_RULES, OBSERVER_FORMAT, SIDECAR_FORMAT, Labels, Observer
from alife.game.rules import (
    ARCHETYPE_NAMES,
    INTENT_NAMES,
    INTENT_VERSION,
    RULES_V1,
    RULES_V2,
    Archetype,
    RuleSet,
)
from alife.game.social import SOCIAL_V1, SOCIAL_VERSION, SocState
from alife.game.society import HOST_KEYS, Society, host_snap
from alife.runio import CheckpointError, RunDir, _write_npz, load_checkpoint
from alife.state import State, init_state, to_numpy
from alife.step import make_step
from alife.tracking import TRACK_VERSION
from game_helpers import social_every

OBS = ["game.observe=true", "run.checkpoint_every=30"]
BLOCK_KEYS = [
    "intent_version",
    "labels_step",
    "alive",
    "arch_labelled",
    "intent_evaluated",
    "intent_labelled",
    "archetype",
    "archetype_match",
    "intent",
    "intent_match",
    "social",
]
SOC15 = dataclasses.replace(SOCIAL_V1, snapshot_every=15)  # snapshots at 15, 30, 45, 60 inside the fixtures
SETTLER4 = dataclasses.replace(RULES_V1, settler_body=4)
SETTLER16 = dataclasses.replace(SETTLER4, arch_warm=16)  # archetypes are labelled within a short test
V2_WARM16 = dataclasses.replace(RULES_V2, arch_warm=16)  # the same for the version-2 rules (settler_body 4)
AT = 61  # the step of the module's observer: 24-step warm-up + 4 evaluations to the first intent label
WARM_NOTE = "labels warm up (192 steps for archetypes, 24 for intents)"
RESTART_NOTE = "social tracking restarts at step"
HEAD_KEYS = ["social_version", "snapshot_step", "origin", "houses", "kin_groups", "colonies"]


def _observing(cfg: Config, *extra: str) -> Config:
    return apply_overrides(cfg, [*OBS, *extra])


def _run(cfg: Config, path: Path, steps: int) -> RunDir:
    rd = RunDir.create(path, cfg)
    res = runner.run(rd, cfg, init_state(cfg), steps, quiet=True)
    assert (res.status, res.exit_code) == ("done", runner.ExitCode.OK)
    return rd


def _continue(rd: RunDir, cfg: Config, steps: int, *, quiet: bool = True) -> None:
    """Resume ``rd`` from its latest checkpoint for ``steps`` more steps, as ``alife resume`` does."""
    path = rd.latest_checkpoint()
    assert path is not None
    state = load_checkpoint(path, cfg)
    rd.truncate_after(int(to_numpy(state.step)))
    res = runner.run(rd, cfg, state, steps, quiet=quiet)
    assert (res.status, res.exit_code) == ("done", runner.ExitCode.OK)


def _same(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> bool:
    return a.keys() == b.keys() and all(
        a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and np.array_equal(a[k], b[k]) for k in a
    )


def _meta(arrays: dict[str, np.ndarray]) -> dict:
    return json.loads(str(arrays["meta"]))


def _manual(
    cfg: Config, steps: int, records: list[dict] | None = None, rules: RuleSet | None = None
) -> tuple[State, Observer]:
    """``steps`` observed steps, driven as the runner drives them: one eval per step, then the records of the
    snapshot settled at that step are taken (and appended to ``records`` if given), so that a sidecar taken
    at any step never meets an untaken record."""
    s = init_state(cfg)
    fn = make_step(cfg, observe=True)
    obs = Observer.fresh(s, cfg, rules=rules, social=SOC15)
    for t in range(steps):
        s, out = fn(s)
        obs.update(s, out.acts, t + 1)
        mx.eval(s, out.base(), *obs.arrays())
        taken = obs.take_records()
        if records is not None:
            records.extend(taken)
    return s, obs


# --- the observer -----------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def observed(tiny_cfg: Config) -> tuple[Config, State, Observer]:
    """An observer after ``AT`` steps of ``tiny``: intents are labelled (warm-up 24 steps, then 4 evaluations
    to switch), archetypes not yet (192-step warm-up)."""
    cfg = _observing(tiny_cfg)
    s, obs = _manual(cfg, AT)
    return cfg, s, obs


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "eager"])
@pytest.mark.parametrize(
    "rules",
    [
        RULES_V1,
        dataclasses.replace(RULES_V1, every=4, hold=2),
        SETTLER16,
        RULES_V2,
        dataclasses.replace(RULES_V2, every=4, hold=2),
        V2_WARM16,
    ],
    ids=["v1", "every4-hold2", "settler4-warm16", "v2", "v2-every4-hold2", "v2-warm16"],
)
def test_observer_is_the_reference_fold(tiny_cfg: Config, compiled: bool, rules: RuleSet) -> None:
    """On real steps the Observer (compiled, as the run is, or eager) equals the reference fold, field by
    field: G1's ``update`` after every step and ``evaluate`` at the steps that are multiples of
    ``rules.every``; and the social part: ``social.update`` after every step, and at the snapshot steps
    (every 15 here, so snapshots and relabels fall inside the 80 steps) ``snapshot``, the host settle and
    ``relabel``. The Settler clause (``settler_body`` 4) reads the body of the last snapshot: the reference
    ``evaluate`` takes the body from the observer's ``so.body`` after the same step, before the step's own
    snapshot is settled (the labels of step t are evaluated from the previous snapshot). The version-2 rules
    also read the step's post-step ``ObsIn`` (the Budding forecast); the reference passes it likewise."""
    cfg = _observing(tiny_cfg, f"run.compile={'true' if compiled else 'false'}")
    s = init_state(cfg)
    fn = make_step(cfg, observe=True)
    obs = Observer.fresh(s, cfg, rules=rules, social=SOC15)
    ref = init_obs(obs_in(s), cfg)
    ref_so = soc.init_soc(soc.soc_in(s), cfg, SOC15)
    ref_society = Society(cfg, SOC15)
    snap0 = soc.snapshot(ref_so.kin, ref_so, soc.soc_in(s), None, cfg, SOC15)  # the initial snapshot
    kin_lut, col_lut = ref_society.initialise(host_snap(snap0), 0)
    ref_so = soc.relabel(ref_so, snap0, mx.array(kin_lut), mx.array(col_lut))
    evaluated = snapshots = 0
    for t in range(1, 81):
        s, out = fn(s)
        obs.update(s, out.acts, t)
        body = obs.so.body  # the body the step's evaluation used: before this step's snapshot is settled
        kin0 = ref_so.kin
        ref = update(ref, obs_in(s), out.acts, cfg, rules)
        ref_so = soc.update(ref_so, soc.soc_in(s), out.acts, cfg, SOC15)
        ref_body = ref_so.body
        if t % rules.every == 0:
            ref = evaluate(ref, cfg, rules, body=ref_body if rules.settler_body else None, oi=obs_in(s))
            evaluated += 1
        snap = soc.snapshot(kin0, ref_so, soc.soc_in(s), out.acts, cfg, SOC15) if t % 15 == 0 else None
        mx.eval(s, out.base(), *obs.arrays(), *ref, *ref_so, *(snap if snap is not None else ()))
        if snap is not None:
            kin_lut, col_lut = ref_society.settle(host_snap(snap), t)
            ref_so = soc.relabel(ref_so, snap, mx.array(kin_lut), mx.array(col_lut))
            snapshots += 1
        assert np.array_equal(to_numpy(body), to_numpy(ref_body)), t
        records = obs.take_records()  # the runner's path: settle at the snapshot step
        assert [r["step"] for r in records] == ([t] if t % 15 == 0 else []), t
        # the records are the reference society's: the observer hands the snapshot the start-of-step slots
        assert records == ref_society.take_records(), t
        for f in ObsState._fields:
            assert np.array_equal(to_numpy(getattr(obs.o, f)), to_numpy(getattr(ref, f))), (t, f)
        for f in SocState._fields:
            assert np.array_equal(to_numpy(getattr(obs.so, f)), to_numpy(getattr(ref_so, f))), (t, f)
    assert evaluated == 80 // rules.every and int(to_numpy(obs.o.label_intent).max()) > 0
    assert snapshots == 5 and obs.society.snapshot_step == 75 and ref_society.snapshot_step == 75
    if (
        rules.settler_body and rules.arch_warm < 80
    ):  # the clause was in play: rows in a body of >= 4 matched Settler that the speed rule
        # alone (the same accumulators, no clause) would not have
        bit = 1 << (Archetype.SETTLER.value - 1)
        without = evaluate(obs.o, cfg, dataclasses.replace(rules, settler_body=0), oi=obs_in(s)).match_arch
        with_clause = to_numpy(obs.o.match_arch)
        assert int(((with_clause & bit) != 0).sum()) > int(((to_numpy(without) & bit) != 0).sum())
        assert int((to_numpy(obs.so.body) >= rules.settler_body).sum()) > 0
    assert obs.row_block()["labels_step"] == 80 - 80 % rules.every
    assert obs.row_block()["social"] == ref_society.head()


def test_the_settler_clause_reads_the_body_of_the_new_occupant(micro_cfg: Config) -> None:
    """The evaluation of step t reads the body as of the step's own update: a row reused in the step it
    died (its occupant changed) has body 0 there, though the dead occupant's body was 9. With the speed rule
    off and no warm-up the new occupant is evaluated from zero accumulators, so the clause alone decides its
    Settler bit."""
    cfg = apply_overrides(micro_cfg, ["population.immigration_every=1", "population.immigration_count=1"])
    rules = dataclasses.replace(RULES_V1, settler_body=4, settler_speed=(0, 1), arch_warm=0)
    s = init_state(cfg)
    fn = make_step(cfg, observe=True)
    obs = Observer.fresh(s, cfg, rules=rules, social=SOC15)
    for t in range(1, 8):
        s, out = fn(s)
        obs.update(s, out.acts, t)
        mx.eval(s, out.base(), *obs.arrays())
        obs.take_records()
    assert bool(to_numpy(s.alive)[0])
    old_uid = int(to_numpy(s.uid)[0])
    alive = to_numpy(obs.so.uid) >= 0
    obs.so = obs.so._replace(
        body=mx.array(np.where(alive, 9, 0).astype(np.int32))
    )  # every row in a body of 9
    energy = to_numpy(s.energy).copy()
    energy[0] = 0  # row 0 starves at S14; an immigrant lands on it at S16 of the same step
    s = s._replace(energy=mx.array(energy))
    s, out = fn(s)
    obs.update(s, out.acts, 8)  # step 8 is an evaluation step
    mx.eval(s, out.base(), *obs.arrays())
    assert int(to_numpy(s.uid)[0]) != old_uid and bool(to_numpy(s.alive)[0])
    bit = 1 << (Archetype.SETTLER.value - 1)
    match = to_numpy(obs.o.match_arch)
    assert match[0] & bit == 0  # the new occupant: body 0
    assert ((match[1:8] & bit) != 0).all()  # the other founders carry their body of 9: Settlers by the clause
    assert to_numpy(obs.so.body)[0] == 0 and (to_numpy(obs.so.body)[1:8] == 9).all()


def test_update_requires_consecutive_steps(observed: tuple[Config, State, Observer]) -> None:
    cfg, s, _ = observed
    obs = Observer.fresh(s, cfg)
    assert obs.step == AT
    _, out = make_step(cfg, observe=True)(s)
    for t in (AT, AT + 2, 0):
        with pytest.raises(ValueError, match="cannot fold step"):
            obs.update(s, out.acts, t)
    assert obs.step == AT


def test_a_window_that_overflows_int32_is_refused(tiny_cfg: Config) -> None:
    s = init_state(tiny_cfg)
    with pytest.raises(ValueError, match="overflows int32"):
        Observer.fresh(s, tiny_cfg, rules=RuleSet(arch_k=20))
    assert Observer.fresh(s, tiny_cfg, rules=RuleSet(arch_k=12)).rules.arch_k == 12


def test_fresh_starts_unlabelled_with_bonds_of_unknown_age(tiny_cfg: Config) -> None:
    s = init_state(tiny_cfg)
    obs = Observer.fresh(s, tiny_cfg)
    o = {f: to_numpy(a) for f, a in zip(ObsState._fields, obs.o, strict=True)}
    assert obs.step == 0 and not o["n_obs"].any() and not o["arch"].any() and not o["intent"].any()
    assert (o["since_bud"] == OBS_CAP).all() and not o["label_intent"].any()
    alive = to_numpy(s.alive)
    assert (o["uid"] == np.where(alive, to_numpy(s.uid), -1)).all()
    block = obs.row_block()
    assert block["alive"] == int(alive.sum()) and block["labels_step"] == 0
    assert [block[k] for k in ("arch_labelled", "intent_evaluated", "intent_labelled")] == [0, 0, 0]
    assert [block[k] for k in ("archetype", "archetype_match", "intent", "intent_match")] == [None] * 4
    head = block["social"]  # the initial snapshot: a head without a record (its snapshot is the origin)
    assert list(head) == HEAD_KEYS and head["social_version"] == SOCIAL_VERSION
    assert head["snapshot_step"] == head["origin"] == 0
    assert head["houses"] == {"alive": int(alive.sum()), "exact": True, "unknown_immigrant": 0}
    assert obs.take_records() == [] and obs.society.records == []


def test_labels_and_the_block_agree(observed: tuple[Config, State, Observer]) -> None:
    cfg, s, obs = observed
    lab = obs.labels()
    assert isinstance(lab, Labels) and Labels._fields == (
        "arch_primary",
        "arch_secondary",
        "arch_match",
        "intent",
        "intent_match",
    )
    n = dims(cfg).n
    assert all(a.shape == (n,) and a.dtype == mx.int32 for a in lab)
    host = obs.host_labels()
    assert list(host) == list(Labels._fields) and all(isinstance(v, np.ndarray) for v in host.values())
    o = obs.o
    assert np.array_equal(host["intent"], to_numpy(o.label_intent))
    assert np.array_equal(host["arch_primary"], to_numpy(o.label_arch)[:, 0])
    assert np.array_equal(host["arch_secondary"], to_numpy(o.label_arch)[:, 1])
    # the block is the counts of those labels
    block = obs.row_block()
    intent, bits = host["intent"], host["intent_match"]
    evaluated = (bits & EVAL_BIT) != 0
    assert block["alive"] == int((to_numpy(o.uid) >= 0).sum()) == int(to_numpy(s.alive).sum())
    assert block["intent_labelled"] == int((intent > 0).sum()) > 0
    assert block["intent_evaluated"] == int(evaluated.sum()) >= block["intent_labelled"]
    assert block["arch_labelled"] == 0 and block["archetype"] is None and block["archetype_match"] is None
    assert block["labels_step"] == AT - AT % 8  # the last evaluation
    for i, name in enumerate(INTENT_NAMES, start=1):
        assert block["intent"][name] == pytest.approx((intent == i).sum() / (intent > 0).sum(), abs=1e-12)
        want = ((bits & (1 << (i - 1))) != 0).sum() / evaluated.sum()
        assert block["intent_match"][name] == pytest.approx(want, abs=1e-12)
    assert abs(sum(block["intent"].values()) - 1.0) < 1e-12


def test_archetype_block_once_archetypes_are_warm(tiny_cfg: Config) -> None:
    cfg = _observing(tiny_cfg)
    s, obs = _manual(cfg, 200)  # 192-step warm-up: rows observed that long are labelled at step 192 and 200
    host = obs.host_labels()
    primary, bits = host["arch_primary"], host["arch_match"]
    block = obs.row_block()
    assert block["arch_labelled"] == int((primary > 0).sum()) > 0
    assert list(block["archetype"]) == list(ARCHETYPE_NAMES) == list(block["archetype_match"])
    assert abs(sum(block["archetype"].values()) - 1.0) < 1e-12
    for a, name in enumerate(ARCHETYPE_NAMES, start=1):
        assert block["archetype"][name] == pytest.approx(
            (primary == a).sum() / (primary > 0).sum(), abs=1e-12
        )
        want = ((bits & (1 << (a - 1))) != 0).sum() / (primary > 0).sum()
        assert block["archetype_match"][name] == pytest.approx(want, abs=1e-12)
    assert (bits[primary == 0] == 0).all()  # unlabelled rows have no bits
    generalist = 1 << (Archetype.GENERALIST.value - 1)  # the bit stands for "no rule matched"
    assert ((bits & generalist) != 0).sum() == (primary == Archetype.GENERALIST.value).sum()
    assert int(to_numpy(s.step)) == 200


# --- the sidecar ------------------------------------------------------------------------------------------


def test_sidecar_round_trip(observed: tuple[Config, State, Observer], tmp_path: Path) -> None:
    cfg, s, obs = observed
    side = obs.sidecar()
    assert set(side) == {
        *ObsState._fields,
        *(f"soc.{f}" for f in SocState._fields),
        *HOST_KEYS,
        "meta",
    }
    assert "host.pending" not in side  # no report is stored in the sidecar: every record is in the social log
    assert side["meta"].dtype.kind == "U" and side["meta"].ndim == 0
    assert side["host.summary"].dtype.kind == "U" and side["host.summary"].ndim == 0
    meta = _meta(side)
    d = dims(cfg)
    assert meta | {"alife_version": None, "config_hash": None} == {
        "format": SIDECAR_FORMAT,
        "observer_format": OBSERVER_FORMAT,
        "intent_version": INTENT_VERSION,
        "rules_digest": DEFAULT_RULES.digest(),
        "step": AT,
        "n_max": d.n,
        "dims": 2,
        "config_hash": None,
        "alife_version": None,
        "social_version": SOCIAL_VERSION,
        "social_digest": SOC15.digest(),
        "track_version": TRACK_VERSION,
        "house_rate": cfg.game.house_rate,
        "house_seed": cfg.run.seed,
    }
    assert OBSERVER_FORMAT == 3 and meta["house_rate"] == 1 / 2048
    assert (
        DEFAULT_RULES is RULES_V2
        and meta["rules_digest"] == RULES_V2.digest()
        and meta["intent_version"] == 2
    )
    rd = RunDir.create(tmp_path / "run", cfg)
    path = rd.save_checkpoint(s, cfg, observers=side)
    assert rd.observer_files() == [(AT, rd.observer_path(AT))] and rd.observer_path(AT).is_file()
    assert rd.observer_path(AT) == rd.path / "observers" / f"step_{AT:010d}.npz"
    back = rd.read_observers(AT)
    assert back is not None and _same(side, back)
    with np.load(rd.observer_path(AT), allow_pickle=False) as z:  # no pickle is needed to read it
        assert sorted(z.files) == sorted(side)
    assert rd.read_observers(AT - 1) is None and not [
        p for p in rd.observers_dir.iterdir() if p.suffix == ".tmp"
    ]
    opened, note = Observer.open(rd, load_checkpoint(path, cfg), cfg, social=SOC15)
    assert note is None and opened.step == AT
    assert _same(opened.sidecar(), side)
    for f in ObsState._fields:
        assert np.array_equal(to_numpy(getattr(opened.o, f)), to_numpy(getattr(obs.o, f))), f
    for f in SocState._fields:
        assert np.array_equal(to_numpy(getattr(opened.so, f)), to_numpy(getattr(obs.so, f))), f
    # the block, head included, is equal as a dict and in key order at every level
    assert opened.row_block() == obs.row_block()
    assert json.dumps(opened.row_block()) == json.dumps(obs.row_block())
    assert list(opened.row_block()["social"]) == HEAD_KEYS
    # the restored society continues as the original would (the same slots, the same ids)
    assert opened.society.kin_slots.tolist() == obs.society.kin_slots.tolist()
    assert opened.society.col_slots.tolist() == obs.society.col_slots.tolist()
    assert opened.society.take_records() == [] and opened.society.snapshot_step == 60


@pytest.fixture(scope="module")
def saved(observed: tuple[Config, State, Observer], tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A run directory whose checkpoint at step AT has a valid sidecar."""
    cfg, s, obs = observed
    rd = RunDir.create(tmp_path_factory.mktemp("sidecar") / "run", cfg)
    rd.save_checkpoint(s, cfg, observers=obs.sidecar())
    return rd.path


def _with_meta(**fields) -> object:
    def edit(a: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {**a, "meta": np.array(json.dumps({**_meta(a), **fields}))}

    return edit


def _g1_sidecar(a: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """What a format-1 (G1) sidecar holds: the ``ObsState`` members and a meta without the social fields."""
    meta = {k: v for k, v in _meta(a).items() if not k.startswith(("social_", "track_", "house_"))}
    keep = {k: v for k, v in a.items() if k in ObsState._fields}
    return {**keep, "meta": np.array(json.dumps({**meta, "observer_format": 1}, sort_keys=True))}


def _g2_sidecar(a: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """What a format-2 (G2, version-1 rules) sidecar holds: no ``held``, seven intent columns and
    ``observer_format`` 2 (the members of the social layer are the same)."""
    keep = {k: v for k, v in a.items() if k != "held"}
    keep["intent"] = a["intent"][:, :7].copy()
    meta = {**_meta(a), "observer_format": 2, "intent_version": 1, "rules_digest": RULES_V1.digest()}
    return {**keep, "meta": np.array(json.dumps(meta, sort_keys=True))}


def _host(**edits: np.ndarray) -> object:
    def edit(a: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {**a, **edits}

    return edit


def _with_memberships(**edits: np.ndarray) -> object:
    """Host members with a short membership list for the kin family (ids 1 and 2 below next_id 3)."""
    base = {
        "host.kin.next_id": np.array(3, np.int64),
        "host.kin.system": np.array([1, 2], np.int64),
        "host.kin.key": np.array([5, 9], np.int64),
    }
    return _host(**{**base, **edits})


I64 = np.int64
MISMATCHES = [
    *[
        pytest.param(_with_meta(**{key: value}), f"meta.{key}", id=f"meta-{key}")
        for key, value in (
            ("format", "other"),
            ("observer_format", OBSERVER_FORMAT + 1),
            ("intent_version", INTENT_VERSION + 1),
            ("rules_digest", "0" * 64),
            ("step", AT - 1),
            ("n_max", 7),
            ("dims", 3),
            ("config_hash", "f" * 64),
            ("social_version", SOCIAL_VERSION + 1),
            ("social_digest", "0" * 64),
            ("track_version", TRACK_VERSION + 1),
            ("house_rate", 0.25),
            ("house_seed", 12345),
        )
    ],
    pytest.param(lambda a: {**a, "meta": np.array("not json {")}, "not JSON", id="meta-garbage"),
    pytest.param(lambda a: {**a, "meta": np.array("[1, 2]")}, "not a JSON object", id="meta-list"),
    # nested past the parser's recursion limit: a reason, not a RecursionError out of `alife resume`
    pytest.param(
        lambda a: {**a, "meta": np.array("[" * 100_000 + "]" * 100_000)}, "not JSON", id="meta-too-deep"
    ),
    pytest.param(lambda a: {k: v for k, v in a.items() if k != "meta"}, "has no meta", id="meta-missing"),
    pytest.param(lambda a: {k: v for k, v in a.items() if k != "arch"}, "missing ['arch']", id="missing"),
    pytest.param(
        lambda a: {k: v for k, v in a.items() if k != "soc.kin"}, "missing ['soc.kin']", id="missing-soc"
    ),
    pytest.param(
        lambda a: {k: v for k, v in a.items() if k != "host.n_ref"},
        "missing ['host.n_ref']",
        id="missing-host",
    ),
    pytest.param(lambda a: {**a, "extra": np.zeros(3, np.int32)}, "unexpected ['extra']", id="extra"),
    pytest.param(lambda a: {**a, "arch": a["arch"].astype(np.int64)}, "array 'arch'", id="dtype"),
    pytest.param(lambda a: {**a, "slot_kin": a["slot_kin"].astype(np.int32)}, "array 'slot_kin'", id="bool"),
    pytest.param(lambda a: {**a, "intent": a["intent"][:, :3]}, "array 'intent'", id="shape"),
    pytest.param(lambda a: {**a, "uid": np.flip(a["uid"]).copy()}, "uids", id="uid"),
    pytest.param(lambda a: {**a, "soc.uid": np.flip(a["soc.uid"]).copy()}, "uids", id="soc-uid"),
    pytest.param(
        lambda a: {**a, "soc.kin": a["soc.kin"].astype(np.int64)}, "array 'soc.kin'", id="soc-dtype"
    ),
    pytest.param(lambda a: {**a, "soc.rel": a["soc.rel"][:, :-1]}, "array 'soc.rel'", id="soc-shape"),
    pytest.param(
        lambda a: {**a, "soc.house": a["soc.house"].astype(np.int32)}, "array 'soc.house'", id="soc-house"
    ),
    # a format-1 sidecar gets the format reason, not a list of the missing soc.* / host.* members
    pytest.param(_g1_sidecar, "meta.observer_format is 1, expected 3;", id="format-1"),
    # ... and so does the previous format (G2, 7 intent columns, no `held`), before its other differences
    pytest.param(_g2_sidecar, "meta.observer_format is 2, expected 3;", id="format-2"),
    # one case per check of Society.from_arrays (the host members; the cases of test_game_society apply to it)
    pytest.param(_host(**{"host.n_ref": np.array(5, np.int32)}), "host.n_ref is int32", id="host-dtype"),
    pytest.param(_host(**{"host.prev_step": np.zeros(1, I64)}), "host.prev_step is int64", id="host-rank"),
    pytest.param(_host(**{"host.summary": np.array(5)}), "host.summary is", id="host-summary-dtype"),
    pytest.param(
        _host(**{"host.summary": np.array("{no")}), "host.summary is not JSON", id="host-summary-json"
    ),
    pytest.param(
        _host(**{"host.summary": np.array("[" * 100_000 + "]" * 100_000)}),
        "host.summary is not JSON",
        id="host-summary-too-deep",
    ),
    pytest.param(
        _host(**{"host.kin.names": np.array('[[100000000000000000000, "X", null]]')}),
        "host.kin.names holds an item",
        id="host-names-over-int64",
    ),
    pytest.param(
        _host(**{"host.kin.names": np.array("[" * 100_000 + "]" * 100_000)}),
        "host.kin.names is not JSON",
        id="host-names-too-deep",
    ),
    pytest.param(
        _host(**{"host.summary": np.array("[1]")}), "host.summary is not an object", id="host-summary-list"
    ),
    pytest.param(
        _host(**{"host.summary": np.array('{"houses": {}}')}),
        "host.summary is not an object",
        id="host-summary-keys",
    ),
    pytest.param(_host(**{"host.kin.next_id": np.array(0, I64)}), "next_id", id="host-next-id"),
    pytest.param(
        _with_memberships(**{"host.kin.key": np.array([5], I64)}), "differ in length", id="host-lengths"
    ),
    pytest.param(
        _with_memberships(**{"host.kin.system": np.array([1, 3], I64)}), "outside", id="host-system-range"
    ),
    pytest.param(_with_memberships(**{"host.kin.key": np.array([9, 5], I64)}), "sorted", id="host-unsorted"),
    pytest.param(
        _with_memberships(**{"host.kin.slots": np.zeros(3, I64)}),
        "host.kin.slots has shape",
        id="host-slots-shape",
    ),
    pytest.param(
        _with_memberships(**{"host.kin.slots": np.array([9] + [-1] * 15, I64)}),
        "host.kin.slots holds an id that is not",
        id="host-slot-unknown",
    ),
    pytest.param(
        _with_memberships(**{"host.kin.slots": np.array([1, 1] + [-1] * 14, I64)}),
        "twice",
        id="host-slot-twice",
    ),
    pytest.param(
        _host(**{"host.terr.ema": np.zeros((3, 1024), I64)}), "host.terr.ema has shape", id="host-terr-shape"
    ),
    pytest.param(
        _host(**{"host.terr.ids": np.array([3, 3], I64), "host.terr.ema": np.zeros((3, 1024), I64)}),
        "host.terr.ids holds an id twice",
        id="host-terr-ids",
    ),
    pytest.param(_host(**{"host.prev_step": np.array(AT + 1, I64)}), "host.prev_step", id="host-prev-late"),
    pytest.param(
        _host(**{"host.window_start": np.array(AT + 5, I64)}), "host.window_start", id="host-window-late"
    ),
    pytest.param(
        _host(**{"host.house_origin": np.array(AT + 1, I64)}), "host.house_origin", id="host-origin-late"
    ),
    pytest.param(_host(**{"host.n_ref": np.array(-2, I64)}), "host.n_ref is -2", id="host-n-ref"),
]


def _copy_with(saved: Path, dst: Path, edit) -> RunDir:
    """A copy of ``saved`` whose sidecar at step AT is ``edit(arrays)``."""
    shutil.copytree(saved, dst)
    rd = RunDir.open(dst)
    arrays = rd.read_observers(AT)
    assert arrays is not None
    _write_npz(rd.observer_path(AT), edit(arrays))
    return rd


@pytest.mark.parametrize(("edit", "reason"), MISMATCHES)
def test_sidecar_mismatch_falls_back_to_fresh(
    saved: Path, tmp_path: Path, tiny_cfg: Config, edit, reason: str
) -> None:
    cfg = _observing(tiny_cfg)
    rd = _copy_with(saved, tmp_path / "run", edit)
    state = load_checkpoint(rd.checkpoint_at(AT), cfg)
    obs, note = Observer.open(rd, state, cfg, social=SOC15)
    assert note is not None and note.startswith("observers: ") and reason in note and WARM_NOTE in note
    assert f"; {RESTART_NOTE} {AT} (houses seeded from clades; new ids, records with origin {AT})" in note
    assert obs.step == AT and not to_numpy(obs.o.n_obs).any()  # fresh: nothing accumulated or labelled
    assert not to_numpy(obs.o.label_intent).any() and not to_numpy(obs.o.label_arch).any()
    # fresh social tracking: a new origin, the houses seeded from clades (exact only at origin 0)
    assert obs.society.house_origin == AT and obs.society.snapshot_step == AT
    assert obs.row_block()["social"]["origin"] == AT and obs.row_block()["social"]["snapshot_step"] == AT
    assert obs.take_records() == []


def test_a_format_1_sidecar_note_is_exactly_the_format(saved: Path, tmp_path: Path, tiny_cfg: Config) -> None:
    """A G1 sidecar is refused for its observer_format before any other check, whatever its members."""
    cfg = _observing(tiny_cfg)
    rd = _copy_with(saved, tmp_path / "run", _g1_sidecar)
    state = load_checkpoint(rd.checkpoint_at(AT), cfg)
    _, note = Observer.open(rd, state, cfg, social=SOC15)
    assert note is not None and note.startswith(
        "observers: meta.observer_format is 1, expected 3; labels warm up"
    )
    assert "missing" not in note and "soc." not in note and "host." not in note


def test_the_default_social_rules_do_not_read_a_sidecar_of_another_cadence(
    saved: Path, tmp_path: Path, tiny_cfg: Config
) -> None:
    """`saved` was written with snapshots every 15 steps; the module default (250) is another social digest,
    hence a fresh start with the note: houses, ids and records are comparable only under the same rules."""
    cfg = _observing(tiny_cfg)
    rd = _copy_with(saved, tmp_path / "run", lambda a: a)
    state = load_checkpoint(rd.checkpoint_at(AT), cfg)
    obs, note = Observer.open(rd, state, cfg)
    assert note is not None and "meta.social_digest" in note and RESTART_NOTE in note
    assert obs.social.snapshot_every == 250
    obs, note = Observer.open(rd, state, cfg, social=SOC15)
    assert note is None and obs.social.snapshot_every == 15


def test_a_corrupt_or_absent_sidecar_falls_back_to_fresh(
    saved: Path, tmp_path: Path, tiny_cfg: Config
) -> None:
    cfg = _observing(tiny_cfg)
    rd = _copy_with(saved, tmp_path / "run", lambda a: a)
    state = load_checkpoint(rd.checkpoint_at(AT), cfg)
    path = rd.observer_path(AT)
    path.write_bytes(b"this is not a zip file")
    with pytest.raises(CheckpointError, match="unreadable observer sidecar"):
        rd.read_observers(AT)
    obs, note = Observer.open(rd, state, cfg, social=SOC15)
    assert note is not None and "unreadable" in note and WARM_NOTE in note and RESTART_NOTE in note
    assert obs.step == AT and not to_numpy(obs.o.n_obs).any()
    path.write_bytes(path.read_bytes()[:0])  # a truncated file is unreadable too
    assert "unreadable" in (Observer.open(rd, state, cfg, social=SOC15)[1] or "")
    path.unlink()
    obs, note = Observer.open(rd, state, cfg, social=SOC15)
    assert note is not None and f"no sidecar at step {AT}" in note and WARM_NOTE in note


def _bare_npy(arrays: dict[str, np.ndarray]) -> bytes:
    """A valid .npy file (np.load returns an ndarray, not an archive)."""
    buf = io.BytesIO()
    np.save(buf, arrays["arch"])
    return buf.getvalue()


def _zip_with_a_non_npy_member(arrays: dict[str, np.ndarray]) -> bytes:
    """A zip with every member name of a sidecar, but ``arch`` stored without the .npy suffix (np.load's
    NpzFile then returns its raw bytes)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for k, v in arrays.items():
            if k == "arch":
                z.writestr("arch", b"not an npy member")
            else:
                member = io.BytesIO()
                np.save(member, v)
                z.writestr(k + ".npy", member.getvalue())
    return buf.getvalue()


WRONG_FILES = [
    pytest.param(_bare_npy, "not an npz archive", id="bare-npy"),
    pytest.param(_zip_with_a_non_npy_member, "not arrays", id="non-npy-member"),
]


@pytest.mark.parametrize(("make", "reason"), WRONG_FILES)
def test_a_wrong_kind_of_file_is_an_unreadable_sidecar(
    make, reason: str, saved: Path, tmp_path: Path, tiny_cfg: Config
) -> None:
    """np.load accepts these (no OSError, no BadZipFile), but they are not an npz of arrays: the same
    fallback to fresh observers with a note, never an exception (§28.6)."""
    cfg = _observing(tiny_cfg)
    rd = _copy_with(saved, tmp_path / "run", lambda a: a)
    arrays = rd.read_observers(AT)
    assert arrays is not None
    rd.observer_path(AT).write_bytes(make(arrays))
    with pytest.raises(CheckpointError, match=f"unreadable observer sidecar.*{reason}"):
        rd.read_observers(AT)
    state = load_checkpoint(rd.checkpoint_at(AT), cfg)
    obs, note = Observer.open(rd, state, cfg, social=SOC15)
    assert note is not None and "unreadable" in note and WARM_NOTE in note
    assert obs.step == AT and not to_numpy(obs.o.n_obs).any()


@pytest.mark.parametrize(("make", "reason"), WRONG_FILES)
def test_a_resume_over_a_wrong_kind_of_file_exits_zero(
    make, reason: str, saved: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rd = _copy_with(saved, tmp_path / "run", lambda a: a)
    arrays = rd.read_observers(AT)
    assert arrays is not None
    rd.observer_path(AT).write_bytes(make(arrays))
    capsys.readouterr()
    assert cli.main(["resume", str(rd.path), "--steps", "10"]) == 0
    assert f"observers: the sidecar at step {AT} is unreadable" in capsys.readouterr().err
    assert RunDir.open(rd.path).meta()["status"] == "done"


def test_load_rejects_a_member_that_is_not_an_array(saved: Path, tiny_cfg: Config) -> None:
    """Defence in depth: ``_load`` (a public method of the Observer) does not raise on non-array members."""
    cfg = _observing(tiny_cfg)
    rd = RunDir.open(saved)
    state = load_checkpoint(rd.checkpoint_at(AT), cfg)
    arrays = rd.read_observers(AT)
    assert arrays is not None
    obs = Observer.fresh(state, cfg, social=SOC15)
    loaded, reason = obs._load({**arrays, "arch": b"not an array"})  # type: ignore[dict-item]
    assert loaded is None and "array 'arch' is not an array" in reason
    loaded, reason = obs._load({**arrays, "soc.kin": b"not an array"})  # type: ignore[dict-item]
    assert loaded is None and "array 'soc.kin' is not an array" in reason
    assert obs._load(arrays)[0] is not None  # the sidecar itself is valid for it


def test_another_ruleset_never_reads_these_accumulators(saved: Path, tiny_cfg: Config) -> None:
    cfg = _observing(tiny_cfg)
    rd = RunDir.open(saved)
    state = load_checkpoint(rd.checkpoint_at(AT), cfg)
    sweep = dataclasses.replace(RULES_V1, arch_k=5, arch_warm=96)
    obs, note = Observer.open(rd, state, cfg, rules=sweep, social=SOC15)
    assert note is not None and "meta.rules_digest" in note and "96 steps for archetypes" in note
    assert obs.rules is sweep and not to_numpy(obs.o.n_obs).any()
    slower = dataclasses.replace(RULES_V1, every=4)  # even the evaluation period is part of the identity
    assert "meta.rules_digest" in (Observer.open(rd, state, cfg, rules=slower, social=SOC15)[1] or "")
    assert "meta.rules_digest" in (Observer.open(rd, state, cfg, rules=SETTLER4, social=SOC15)[1] or "")
    assert (
        Observer.open(rd, state, cfg, rules=RULES_V1, social=SOC15)[1] is not None
    )  # not the version-1 rules
    assert (
        Observer.open(rd, state, cfg, rules=RULES_V2, social=SOC15)[1] is None
    )  # the rules it was written under (the default)
    # the social rules are part of the identity too: any threshold or the cadence
    other = dataclasses.replace(SOC15, kin_enter=(9, 9))
    obs, note = Observer.open(rd, state, cfg, social=other)
    assert note is not None and "meta.social_digest" in note and obs.social is other


def test_a_sidecar_of_version_1_rules_is_refused_by_version_2_and_loads_under_version_1(
    tiny_cfg: Config, tmp_path: Path
) -> None:
    """The rules' digest is the identity of the accumulators (the Budding flow columns and the hold state
    mean something only under the rules that wrote them): a sidecar written under ``RULES_V1`` is refused
    under ``RULES_V2`` with a note naming ``rules_digest``, and loads under ``RULES_V1``, and the reverse."""
    cfg = _observing(tiny_cfg)
    for written, other in ((RULES_V1, RULES_V2), (RULES_V2, RULES_V1)):
        s, obs = _manual(cfg, 40, rules=written)
        rd = RunDir.create(tmp_path / f"run_{written.digest()[:6]}", cfg)
        path = rd.save_checkpoint(s, cfg, observers=obs.sidecar())
        state = load_checkpoint(path, cfg)
        refused, note = Observer.open(rd, state, cfg, rules=other, social=SOC15)
        assert note is not None and "meta.rules_digest" in note and not to_numpy(refused.o.n_obs).any()
        kept, note = Observer.open(rd, state, cfg, rules=written, social=SOC15)
        assert note is None and np.array_equal(to_numpy(kept.o.n_obs), to_numpy(obs.o.n_obs))
        assert to_numpy(kept.o.n_obs).any()
        assert _meta(obs.sidecar())["rules_digest"] == written.digest()


def test_the_note_is_only_for_a_resume(saved: Path, tmp_path: Path, tiny_cfg: Config) -> None:
    """A new run or a fork has no checkpoint at its starting step in its own directory: a fresh start there is
    the expected case, not worth a note."""
    cfg = _observing(tiny_cfg)
    state = load_checkpoint(RunDir.open(saved).checkpoint_at(AT), cfg)
    elsewhere = RunDir.create(tmp_path / "fork", cfg)
    obs, note = Observer.open(elsewhere, state, cfg, social=SOC15)
    assert note is None and not to_numpy(obs.o.n_obs).any() and obs.step == AT
    assert obs.society.house_origin == AT  # a fork's houses start at its own origin (exact only at step 0)
    rd = _copy_with(saved, tmp_path / "resume", lambda a: a)
    rd.observer_path(AT).unlink()  # a checkpoint without a sidecar: a resume, so the note
    assert "no sidecar" in (Observer.open(rd, state, cfg, social=SOC15)[1] or "")


def test_a_resume_with_a_corrupt_sidecar_exits_zero(
    saved: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rd = _copy_with(saved, tmp_path / "run", lambda a: a)
    rd.observer_path(AT).write_bytes(b"garbage")
    capsys.readouterr()
    assert cli.main(["resume", str(rd.path), "--steps", "20"]) == 0
    assert f"observers: the sidecar at step {AT} is unreadable" in capsys.readouterr().err
    rows = [r for r in RunDir.open(rd.path).metrics_rows() if r["step"] > AT]
    assert rows and all("observers" in r for r in rows)
    assert all(
        r["observers"]["intent_labelled"] == 0 for r in rows if r["step"] < AT + 24
    )  # warming up again


# --- resume -----------------------------------------------------------------------------------------------


def _log(rd: RunDir) -> bytes:
    return rd.social_path.read_bytes()


@pytest.mark.parametrize(
    ("d3", "total", "stop"),
    [
        pytest.param(False, 60, 15, id="2d-15+45-on-a-snapshot"),
        pytest.param(False, 60, 16, id="2d-16+44-after-a-snapshot"),
        pytest.param(False, 60, 30, id="2d-30+30"),
        pytest.param(False, 60, 37, id="2d-37+23"),
        pytest.param(True, 60, 16, id="3d-16+44-after-a-snapshot"),
        pytest.param(True, 60, 30, id="3d-30+30"),
        pytest.param(True, 60, 37, id="3d-37+23"),
        pytest.param(False, 240, 137, id="2d-137+103-archetypes"),
    ],
)
def test_resume_labels_equal_uninterrupted(
    d3: bool,
    total: int,
    stop: int,
    tiny_cfg: Config,
    tiny3d_cfg: Config,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run stopped at ``stop`` (not a multiple of the evaluation period 8; 15 is a snapshot step and 16 the
    step after one, neither a row step) and resumed has, bit for bit, the observer state (labels and social
    state, the host society included), the ``observers`` blocks (the ``social`` head in the same key order at
    every level) and the social log of the uninterrupted run, and says nothing on stderr."""
    social_every(monkeypatch, 15)
    cfg = _observing(tiny3d_cfg if d3 else tiny_cfg)
    ref = _run(cfg, tmp_path / "ref", total)
    split = _run(cfg, tmp_path / "split", stop)
    assert stop in [st for st, _ in split.observer_files()]  # the stop is checkpointed, with its sidecar
    assert [r["step"] for r in split.social_records()] == list(range(15, stop + 1, 15))
    capsys.readouterr()
    _continue(split, cfg, total - stop, quiet=False)
    assert "observers:" not in capsys.readouterr().err  # the sidecar at `stop` was valid
    a, b = ref.read_observers(total), split.read_observers(total)
    assert a is not None and b is not None and _same(a, b)
    if total > 200:  # past the archetype warm-up: the comparison covers labelled rows
        assert (a["label_arch"][:, 0] > 0).any() and (a["label_intent"] > 0).any()
    blocks = {r["step"]: r["observers"] for r in ref.metrics_rows()}
    resumed = {r["step"]: r["observers"] for r in split.metrics_rows() if r["step"] > stop}
    assert resumed and all(blocks[st] == block for st, block in resumed.items())
    assert all(json.dumps(blocks[st]) == json.dumps(block) for st, block in resumed.items())
    assert blocks[total]["intent_labelled"] > 0
    assert _log(split) == _log(ref) and len(ref.social_records()) == total // 15


@pytest.mark.parametrize("d3", [False, True], ids=["2d", "3d"])
@pytest.mark.parametrize("where", ["row-35", "step-16"])
def test_resume_after_a_sigint_equals_uninterrupted(
    d3: bool,
    where: str,
    tiny_cfg: Config,
    tiny3d_cfg: Config,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Ctrl-C delivered while the interval row of step 35 is written (35 is neither a multiple of 8 nor a
    scheduled checkpoint; ``metrics_every`` 5), or raised by the observer's update of step 16 (the step after
    the snapshot of 15, between rows; ``metrics_every`` 10), stops the run cleanly with a checkpoint and its
    sidecar at that step; the resumed run ends with the observer state, the ``observers`` blocks and the
    social log of the uninterrupted one."""
    social_every(monkeypatch, 15)
    extra = ["run.metrics_every=5", "run.heavy_metrics_every=5"] if where == "row-35" else []
    cfg = _observing(tiny3d_cfg if d3 else tiny_cfg, *extra)
    ref = _run(cfg, tmp_path / "ref", 60)
    stop = 35 if where == "row-35" else 16
    fired: list[int] = []

    def interrupt(state: State, row: dict) -> None:
        if row["step"] == 35:
            signal.raise_signal(signal.SIGINT)

    real_update = Observer.update

    def update_then_interrupt(self: Observer, state: State, acts, t: int) -> None:
        real_update(self, state, acts, t)
        if t == 16 and not fired:  # the latch is read after the update, in the same iteration
            fired.append(t)
            signal.raise_signal(signal.SIGINT)

    if where == "step-16":
        monkeypatch.setattr(Observer, "update", update_then_interrupt)
    split = RunDir.create(tmp_path / "split", cfg)
    before = signal.getsignal(signal.SIGINT)
    res = runner.run(
        split, cfg, init_state(cfg), 60, quiet=True, on_interval=interrupt if where == "row-35" else None
    )
    assert (res.status, res.exit_code) == ("interrupted", runner.ExitCode.INTERRUPTED)
    assert int(to_numpy(res.state.step)) == stop and signal.getsignal(signal.SIGINT) is before
    del res
    want = [0, 30, 35] if where == "row-35" else [0, 16]
    assert [st for st, _ in split.observer_files()] == want  # the stop has its sidecar too
    assert [r["step"] for r in split.social_records()] == list(range(15, stop + 1, 15))
    capsys.readouterr()
    _continue(split, cfg, 60 - stop, quiet=False)
    assert "observers:" not in capsys.readouterr().err  # the sidecar at the stop was valid
    a, b = ref.read_observers(60), split.read_observers(60)
    assert a is not None and b is not None and _same(a, b)
    blocks = {r["step"]: r["observers"] for r in ref.metrics_rows()}
    resumed = {r["step"]: r["observers"] for r in split.metrics_rows() if r["step"] > stop}
    assert sorted(resumed) == ([40, 45, 50, 55, 60] if where == "row-35" else [20, 30, 40, 50, 60])
    assert all(json.dumps(blocks[st]) == json.dumps(block) for st, block in resumed.items())
    assert _log(split) == _log(ref)


# --- pruning and truncation -------------------------------------------------------------------------------


def test_prune_and_truncate_cut_sidecars(
    tiny_cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    social_every(monkeypatch, 15)
    cfg = _observing(tiny_cfg, "run.keep_checkpoints=1")
    rd = _run(cfg, tmp_path / "keep1", 120)
    assert [st for st, _ in rd.checkpoints()] == [0, 120]  # step 0 and the newest
    assert [st for st, _ in rd.observer_files()] == [0, 120]  # a sidecar lives as long as its checkpoint
    # pruning never removes the social log: it holds every record, though most checkpoints are gone
    assert [r["step"] for r in rd.social_records()] == list(range(15, 121, 15))

    cfg = _observing(tiny_cfg, "run.keep_checkpoints=3")
    rd = _run(cfg, tmp_path / "keep3", 120)
    assert [st for st, _ in rd.checkpoints()] == [0, 60, 90, 120]
    assert [st for st, _ in rd.observer_files()] == [0, 60, 90, 120]
    assert [r["step"] for r in rd.social_records()] == list(range(15, 121, 15))
    twin = tmp_path / "twin"
    shutil.copytree(rd.path, twin)
    original = _log(rd)
    rd.truncate_after(60)
    assert [st for st, _ in rd.checkpoints()] == [0, 60] and [st for st, _ in rd.observer_files()] == [0, 60]
    assert rd.read_observers(60) is not None and rd.read_observers(90) is None
    assert [r["step"] for r in rd.social_records()] == [15, 30, 45, 60]  # the log is cut with the sidecars
    assert _log(rd) == b"".join(original.splitlines(keepends=True)[:4])
    RunDir.open(twin).truncate_after(90, keep_director=True)  # a replay keeps the director files, not these
    assert [st for st, _ in RunDir.open(twin).observer_files()] == [0, 60, 90]
    assert [r["step"] for r in RunDir.open(twin).social_records()] == list(range(15, 91, 15))

    # an orphan (a crash between the sidecar and its checkpoint) is removed by the next prune
    shutil.copyfile(rd.observer_path(60), rd.observer_path(45))
    assert [st for st, _ in rd.observer_files()] == [0, 45, 60]
    rd.save_checkpoint(load_checkpoint(rd.checkpoint_at(60), cfg), cfg)
    assert [st for st, _ in rd.observer_files()] == [
        0
    ]  # 60 was rewritten without observers: its sidecar is gone
    assert [r["step"] for r in rd.social_records()] == [15, 30, 45, 60]  # and the log is not a sidecar

    off = _run(apply_overrides(tiny_cfg, ["run.checkpoint_every=30"]), tmp_path / "off", 60)
    assert not (off.path / "observers").exists() and off.observer_files() == [] and off.social_records() == []


def test_checkpoint_without_observers_removes_a_stale_sidecar(
    observed: tuple[Config, State, Observer], tmp_path: Path
) -> None:
    cfg, s, obs = observed
    rd = RunDir.create(tmp_path / "run", cfg)
    rd.save_checkpoint(s, cfg)  # without observers: no observers/ directory is made
    assert not rd.observers_dir.exists() and rd.observer_files() == []
    rd.save_checkpoint(s, cfg, observers=obs.sidecar())
    assert rd.read_observers(AT) is not None
    rd.save_checkpoint(s, cfg)  # the same step, saved without observers: the sidecar of the older save goes
    assert rd.read_observers(AT) is None and rd.observer_files() == []
    assert [st for st, _ in rd.checkpoints()] == [AT]
    rd.save_checkpoint(s, cfg, observers={})  # an empty mapping is no sidecar
    assert rd.observer_files() == []


# --- the metrics block ------------------------------------------------------------------------------------


def test_metrics_block_only_when_observing(
    tiny_cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    social_every(monkeypatch, 15)
    cfg = _observing(tiny_cfg)
    on = _run(cfg, tmp_path / "on", 60)
    off = _run(apply_overrides(cfg, ["game.observe=false"]), tmp_path / "off", 60)
    assert all("observers" not in r for r in off.metrics_rows())
    rows = on.metrics_rows()
    assert [r["step"] for r in rows] == [10, 20, 30, 40, 50, 60]
    for row in rows:
        assert list(row)[-1] == "observers"  # appended last (after carrying_capacity_est on the first row)
        block = row["observers"]
        assert list(block) == BLOCK_KEYS and block["intent_version"] == INTENT_VERSION
        assert block["labels_step"] == row["step"] - row["step"] % 8
        assert 0 < block["alive"] <= dims(cfg).n
        for key, names in (
            ("archetype", ARCHETYPE_NAMES),
            ("archetype_match", ARCHETYPE_NAMES),
            ("intent", INTENT_NAMES),
            ("intent_match", INTENT_NAMES),
        ):
            shares = block[key]
            assert shares is None or list(shares) == list(names)
            assert shares is None or all(0.0 <= v <= 1.0 for v in shares.values())
        for key in ("archetype", "intent"):  # exclusive labels: the shares sum to 1, or there are none
            assert block[key] is None or abs(sum(block[key].values()) - 1.0) < 1e-12
    assert "carrying_capacity_est" in rows[0] and list(rows[0])[-2] == "carrying_capacity_est"
    first, last = rows[0]["observers"], rows[-1]["observers"]
    assert last["alive"] == int(to_numpy(load_checkpoint(on.checkpoint_at(60), cfg).alive).sum())
    assert first["intent_labelled"] == 0 and first["intent"] is None and first["intent_match"] is None
    assert first["arch_labelled"] == 0 and first["archetype"] is None  # nothing is warm at step 10
    assert last["intent_labelled"] > 0 and last["intent_evaluated"] >= last["intent_labelled"]
    assert last["arch_labelled"] == 0 and last["archetype"] is None  # the archetype warm-up is 192 steps
    # JSON-clean: the stored row is exactly what the observer reported
    assert json.loads(json.dumps(last)) == last
    # the social head: the last settled snapshot (rows at 10, 20, ..., 60; snapshots at 15, 30, 45, 60; the
    # initial snapshot, at the origin 0, before the first), no reports, and the record it names
    assert [r["observers"]["social"]["snapshot_step"] for r in rows] == [0, 15, 30, 30, 45, 60]
    records = {(r["origin"], r["step"]): r for r in on.social_records()}
    assert sorted(step for _, step in records) == [15, 30, 45, 60]
    for row in rows:
        head = row["observers"]["social"]
        assert list(head) == HEAD_KEYS and head["social_version"] == SOCIAL_VERSION and head["origin"] == 0
        assert "reports" not in head and "reports" not in row["observers"]
        assert head["houses"]["exact"] is True and head["houses"]["unknown_immigrant"] == 0
        if head["snapshot_step"] > head["origin"]:
            rec = records[(head["origin"], head["snapshot_step"])]
            assert head == {
                "social_version": rec["social_version"],
                "snapshot_step": rec["step"],
                "origin": rec["origin"],
                **rec["summary"],
            }
            assert json.dumps(head) == json.dumps(
                {
                    "social_version": rec["social_version"],
                    "snapshot_step": rec["step"],
                    "origin": rec["origin"],
                    **rec["summary"],
                }
            )
        else:
            assert (head["origin"], head["snapshot_step"]) not in records  # the initial snapshot: no record
    assert len(json.dumps(rows[-1]["observers"]["social"])) < 512  # the head's budget (0.5 KB)
    assert off.social_records() == [] and not off.observers_dir.exists()


def test_sidecar_refuses_untaken_records(tiny_cfg: Config, tmp_path: Path) -> None:
    """A checkpoint must never stand for a snapshot whose record is in neither the social log nor the
    sidecar: at a snapshot step ``sidecar()`` raises until ``take_records()`` has run, then succeeds, and the
    record is not lost by the refusal."""
    cfg = _observing(tiny_cfg)
    s = init_state(cfg)
    fn = make_step(cfg, observe=True)
    obs = Observer.fresh(s, cfg, social=SOC15)
    assert obs.sidecar()["meta"].ndim == 0  # nothing pending, nothing untaken: fine at any step
    for t in range(1, 16):
        s, out = fn(s)
        obs.update(s, out.acts, t)
        mx.eval(s, out.base(), *obs.arrays())
    with pytest.raises(RuntimeError, match=r"1 snapshot record\(s\) not taken"):
        obs.sidecar()  # the snapshot of step 15 is settled by the call, its record is untaken
    with pytest.raises(RuntimeError, match="not taken"):
        obs.sidecar()  # and still is
    records = obs.take_records()
    assert [r["step"] for r in records] == [15] and records[0]["origin"] == 0
    side = obs.sidecar()
    assert _meta(side)["step"] == 15 and obs.take_records() == []
    # a later snapshot behaves the same: the refusal never discards, the next take returns exactly it
    for t in range(16, 31):
        s, out = fn(s)
        obs.update(s, out.acts, t)
        mx.eval(s, out.base(), *obs.arrays())
    with pytest.raises(RuntimeError):
        obs.sidecar()
    assert [r["step"] for r in obs.take_records()] == [30]
    rd = RunDir.create(tmp_path / "run", cfg)
    rd.save_checkpoint(s, cfg, observers=obs.sidecar())
    assert rd.read_observers(30) is not None
