"""§28.5/§28.6 the host observer, its metrics block and the sidecar (G1): round trip through ``RunDir``, the
fallback to fresh observers on any mismatch, resumed labels equal to uninterrupted ones, pruning and
truncation, and the ``observers`` block of the metrics rows."""

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
from alife.game.observe import EVAL_BIT, OBS_CAP, ObsState, evaluate, init_obs, obs_in, update
from alife.game.observer import OBSERVER_FORMAT, SIDECAR_FORMAT, Labels, Observer
from alife.game.rules import ARCHETYPE_NAMES, INTENT_NAMES, INTENT_VERSION, RULES_V1, Archetype, RuleSet
from alife.runio import CheckpointError, RunDir, _write_npz, load_checkpoint
from alife.state import State, init_state, to_numpy
from alife.step import make_step

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
]
AT = 61  # the step of the module's observer: 24-step warm-up + 4 evaluations to the first intent label
WARM_NOTE = "labels warm up (192 steps for archetypes, 24 for intents)"


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


def _manual(cfg: Config, steps: int) -> tuple[State, Observer]:
    """``steps`` observed steps, driven as the runner drives them (one eval per step)."""
    s = init_state(cfg)
    fn = make_step(cfg, observe=True)
    obs = Observer.fresh(s, cfg)
    for t in range(steps):
        s, out = fn(s)
        obs.update(s, out.acts, t + 1)
        mx.eval(s, out.base(), *obs.arrays())
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
    "rules", [RULES_V1, dataclasses.replace(RULES_V1, every=4, hold=2)], ids=["v1", "every4-hold2"]
)
def test_observer_is_the_reference_fold(tiny_cfg: Config, compiled: bool, rules: RuleSet) -> None:
    """On real steps the Observer (compiled, as the run is, or eager) equals the reference fold: ``update``
    after every step and ``evaluate`` at the steps that are multiples of ``rules.every``, field by field."""
    cfg = _observing(tiny_cfg, f"run.compile={'true' if compiled else 'false'}")
    s = init_state(cfg)
    fn = make_step(cfg, observe=True)
    obs = Observer.fresh(s, cfg, rules=rules)
    ref = init_obs(obs_in(s), cfg)
    evaluated = 0
    for t in range(1, 81):
        s, out = fn(s)
        obs.update(s, out.acts, t)
        ref = update(ref, obs_in(s), out.acts, cfg, rules)
        if t % rules.every == 0:
            ref = evaluate(ref, cfg, rules)
            evaluated += 1
        mx.eval(s, out.base(), *obs.arrays(), *ref)
        for f in ObsState._fields:
            assert np.array_equal(to_numpy(getattr(obs.o, f)), to_numpy(getattr(ref, f))), (t, f)
    assert evaluated == 80 // rules.every and int(to_numpy(obs.o.label_intent).max()) > 0
    assert obs.row_block()["labels_step"] == 80 - 80 % rules.every


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
    assert set(side) == {*ObsState._fields, "meta"}
    assert side["meta"].dtype.kind == "U" and side["meta"].ndim == 0
    meta = _meta(side)
    d = dims(cfg)
    assert meta | {"alife_version": None, "config_hash": None} == {
        "format": SIDECAR_FORMAT,
        "observer_format": OBSERVER_FORMAT,
        "intent_version": INTENT_VERSION,
        "rules_digest": RULES_V1.digest(),
        "step": AT,
        "n_max": d.n,
        "dims": 2,
        "config_hash": None,
        "alife_version": None,
    }
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
    opened, note = Observer.open(rd, load_checkpoint(path, cfg), cfg)
    assert note is None and opened.step == AT
    assert _same(opened.sidecar(), side)
    for f in ObsState._fields:
        assert np.array_equal(to_numpy(getattr(opened.o, f)), to_numpy(getattr(obs.o, f))), f
    assert opened.row_block() == obs.row_block()


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
        )
    ],
    pytest.param(lambda a: {**a, "meta": np.array("not json {")}, "not JSON", id="meta-garbage"),
    pytest.param(lambda a: {**a, "meta": np.array("[1, 2]")}, "not a JSON object", id="meta-list"),
    pytest.param(lambda a: {k: v for k, v in a.items() if k != "arch"}, "missing ['arch']", id="missing"),
    pytest.param(lambda a: {**a, "extra": np.zeros(3, np.int32)}, "unexpected ['extra']", id="extra"),
    pytest.param(lambda a: {**a, "arch": a["arch"].astype(np.int64)}, "array 'arch'", id="dtype"),
    pytest.param(lambda a: {**a, "slot_kin": a["slot_kin"].astype(np.int32)}, "array 'slot_kin'", id="bool"),
    pytest.param(lambda a: {**a, "intent": a["intent"][:, :3]}, "array 'intent'", id="shape"),
    pytest.param(lambda a: {**a, "uid": np.flip(a["uid"]).copy()}, "uids", id="uid"),
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
    obs, note = Observer.open(rd, state, cfg)
    assert note is not None and note.startswith("observers: ") and reason in note and WARM_NOTE in note
    assert obs.step == AT and not to_numpy(obs.o.n_obs).any()  # fresh: nothing accumulated or labelled
    assert not to_numpy(obs.o.label_intent).any() and not to_numpy(obs.o.label_arch).any()


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
    obs, note = Observer.open(rd, state, cfg)
    assert note is not None and "unreadable" in note and WARM_NOTE in note
    assert obs.step == AT and not to_numpy(obs.o.n_obs).any()
    path.write_bytes(path.read_bytes()[:0])  # a truncated file is unreadable too
    assert "unreadable" in (Observer.open(rd, state, cfg)[1] or "")
    path.unlink()
    obs, note = Observer.open(rd, state, cfg)
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
    obs, note = Observer.open(rd, state, cfg)
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
    obs = Observer.fresh(state, cfg)
    loaded, reason = obs._load({**arrays, "arch": b"not an array"})  # type: ignore[dict-item]
    assert loaded is None and "array 'arch' is not an array" in reason


def test_another_ruleset_never_reads_these_accumulators(saved: Path, tiny_cfg: Config) -> None:
    cfg = _observing(tiny_cfg)
    rd = RunDir.open(saved)
    state = load_checkpoint(rd.checkpoint_at(AT), cfg)
    sweep = dataclasses.replace(RULES_V1, arch_k=5, arch_warm=96)
    obs, note = Observer.open(rd, state, cfg, rules=sweep)
    assert note is not None and "meta.rules_digest" in note and "96 steps for archetypes" in note
    assert obs.rules is sweep and not to_numpy(obs.o.n_obs).any()
    slower = dataclasses.replace(RULES_V1, every=4)  # even the evaluation period is part of the identity
    assert "meta.rules_digest" in (Observer.open(rd, state, cfg, rules=slower)[1] or "")
    assert Observer.open(rd, state, cfg, rules=RULES_V1)[1] is None  # the rules it was written under


def test_the_note_is_only_for_a_resume(saved: Path, tmp_path: Path, tiny_cfg: Config) -> None:
    """A new run or a fork has no checkpoint at its starting step in its own directory: a fresh start there is
    the expected case, not worth a note."""
    cfg = _observing(tiny_cfg)
    state = load_checkpoint(RunDir.open(saved).checkpoint_at(AT), cfg)
    elsewhere = RunDir.create(tmp_path / "fork", cfg)
    obs, note = Observer.open(elsewhere, state, cfg)
    assert note is None and not to_numpy(obs.o.n_obs).any() and obs.step == AT
    rd = _copy_with(saved, tmp_path / "resume", lambda a: a)
    rd.observer_path(AT).unlink()  # a checkpoint without a sidecar: a resume, so the note
    assert "no sidecar" in (Observer.open(rd, state, cfg)[1] or "")


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


@pytest.mark.parametrize(
    ("d3", "total", "stop"),
    [
        pytest.param(False, 60, 30, id="2d-30+30"),
        pytest.param(False, 60, 37, id="2d-37+23"),
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
) -> None:
    """A run stopped at ``stop`` (not a multiple of the evaluation period 8) and resumed has, bit for bit, the
    observer state and the ``observers`` blocks of the uninterrupted run, and says nothing on stderr."""
    cfg = _observing(tiny3d_cfg if d3 else tiny_cfg)
    ref = _run(cfg, tmp_path / "ref", total)
    split = _run(cfg, tmp_path / "split", stop)
    assert stop in [st for st, _ in split.observer_files()]  # the stop is checkpointed, with its sidecar
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
    assert blocks[total]["intent_labelled"] > 0


@pytest.mark.parametrize("d3", [False, True], ids=["2d", "3d"])
def test_resume_after_a_sigint_equals_uninterrupted(
    d3: bool,
    tiny_cfg: Config,
    tiny3d_cfg: Config,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A Ctrl-C delivered while the interval row of step 35 is written (35 is neither a multiple of 8 nor a
    scheduled checkpoint) stops the run cleanly with a checkpoint and its sidecar at 35; the resumed run ends
    with the observer state and the ``observers`` blocks of the uninterrupted one."""
    cfg = _observing(tiny3d_cfg if d3 else tiny_cfg, "run.metrics_every=5", "run.heavy_metrics_every=5")
    ref = _run(cfg, tmp_path / "ref", 60)

    def interrupt(state: State, row: dict) -> None:
        if row["step"] == 35:
            signal.raise_signal(signal.SIGINT)

    split = RunDir.create(tmp_path / "split", cfg)
    before = signal.getsignal(signal.SIGINT)
    res = runner.run(split, cfg, init_state(cfg), 60, quiet=True, on_interval=interrupt)
    assert (res.status, res.exit_code) == ("interrupted", runner.ExitCode.INTERRUPTED)
    assert int(to_numpy(res.state.step)) == 35 and signal.getsignal(signal.SIGINT) is before
    del res
    assert [st for st, _ in split.observer_files()] == [0, 30, 35]  # the stop has its sidecar too
    capsys.readouterr()
    _continue(split, cfg, 25, quiet=False)
    assert "observers:" not in capsys.readouterr().err  # the sidecar at 35 was valid
    a, b = ref.read_observers(60), split.read_observers(60)
    assert a is not None and b is not None and _same(a, b)
    blocks = {r["step"]: r["observers"] for r in ref.metrics_rows()}
    resumed = {r["step"]: r["observers"] for r in split.metrics_rows() if r["step"] > 35}
    assert sorted(resumed) == [40, 45, 50, 55, 60] and all(
        blocks[st] == block for st, block in resumed.items()
    )


# --- pruning and truncation -------------------------------------------------------------------------------


def test_prune_and_truncate_cut_sidecars(tiny_cfg: Config, tmp_path: Path) -> None:
    cfg = _observing(tiny_cfg, "run.keep_checkpoints=1")
    rd = _run(cfg, tmp_path / "keep1", 120)
    assert [st for st, _ in rd.checkpoints()] == [0, 120]  # step 0 and the newest
    assert [st for st, _ in rd.observer_files()] == [0, 120]  # a sidecar lives as long as its checkpoint

    cfg = _observing(tiny_cfg, "run.keep_checkpoints=3")
    rd = _run(cfg, tmp_path / "keep3", 120)
    assert [st for st, _ in rd.checkpoints()] == [0, 60, 90, 120]
    assert [st for st, _ in rd.observer_files()] == [0, 60, 90, 120]
    twin = tmp_path / "twin"
    shutil.copytree(rd.path, twin)
    rd.truncate_after(60)
    assert [st for st, _ in rd.checkpoints()] == [0, 60] and [st for st, _ in rd.observer_files()] == [0, 60]
    assert rd.read_observers(60) is not None and rd.read_observers(90) is None
    RunDir.open(twin).truncate_after(90, keep_director=True)  # a replay keeps the director files, not these
    assert [st for st, _ in RunDir.open(twin).observer_files()] == [0, 60, 90]

    # an orphan (a crash between the sidecar and its checkpoint) is removed by the next prune
    shutil.copyfile(rd.observer_path(60), rd.observer_path(45))
    assert [st for st, _ in rd.observer_files()] == [0, 45, 60]
    rd.save_checkpoint(load_checkpoint(rd.checkpoint_at(60), cfg), cfg)
    assert [st for st, _ in rd.observer_files()] == [
        0
    ]  # 60 was rewritten without observers: its sidecar is gone

    off = _run(apply_overrides(tiny_cfg, ["run.checkpoint_every=30"]), tmp_path / "off", 60)
    assert not (off.path / "observers").exists() and off.observer_files() == []


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


def test_metrics_block_only_when_observing(tiny_cfg: Config, tmp_path: Path) -> None:
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
