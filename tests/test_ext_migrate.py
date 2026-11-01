"""The layout migration (DESIGN §43.8, PR 0b) against the stub features: ``state.migrate_layout`` and
``alife migrate``. E2 (enable order), the zero-row equivalence gate, the refusals (removal, width change, base
layout), new-field policy, compressed and plain sources, provenance, and E3's test side (a migrated world
resumes; a mismatched config is refused)."""

from __future__ import annotations

import dataclasses
import itertools
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import ext_stub as stub
from alife import brain, cli, ext, layout, migrate, runio
from alife.config import dims, load_config
from alife.runio import CheckpointError, RunDir, checkpoint_metadata, load_checkpoint, write_state_file
from alife.state import (
    LedgerIdx,
    MigrationError,
    init_state,
    layout_change,
    migrate_layout,
    state_cls,
    state_digest,
    state_from_arrays,
    state_to_arrays,
    to_numpy,
)
from alife.step import make_step

ALL = stub.ALL
PAIRS = list(itertools.combinations(ALL, 2))
F32 = ["brain.weight_dtype=float32"]


@pytest.fixture(autouse=True)
def _registry(monkeypatch: pytest.MonkeyPatch):
    stub.install(monkeypatch)
    dims.cache_clear()  # dims() is cached on the config: a config built under another registry would be stale
    yield
    dims.cache_clear()


def _cfg(nd: int, names=(), dtype: str = "float32"):
    cfg = load_config(preset="tiny" if nd == 2 else "tiny3d", overrides=[f"brain.weight_dtype={dtype}"])
    return stub.stub_cfg(cfg, *names) if names else cfg


def _enable(cfg, *names):
    by = {sp.name: sp for sp in ext.REGISTRY}
    for n in names:
        cfg = by[n].enable(cfg)
    return cfg


def _arrays(cfg):
    return state_to_arrays(init_state(cfg))


def _migrate(arrays, cfg_old, cfg_new, **kw):
    return migrate_layout(arrays, cfg_old, cfg_new, **kw)


def _same(a, b) -> None:
    assert list(a) == list(b)
    for k in a:
        assert a[k].shape == b[k].shape and a[k].dtype == b[k].dtype, k
        np.testing.assert_array_equal(np.array(a[k]), np.array(b[k]), err_msg=k)


def _canon(names) -> tuple[str, ...]:
    return tuple(n for n in ALL if n in names)


# --- E2: the order of enabling does not matter ------------------------------------------------------------


@pytest.mark.parametrize("nd", [2, 3])
@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: "+".join(p))
def test_e2_pairs_enable_order_is_irrelevant(nd: int, pair: tuple[str, str]) -> None:
    """A then B == B then A == both at once: config, layout, offsets, version string and every array."""
    a, b = pair
    c0 = _cfg(nd)
    s0 = _arrays(c0)
    ca, cb = _enable(c0, a), _enable(c0, b)
    cab, cba = _enable(ca, b), _enable(cb, a)
    both = _enable(c0, a, b)
    assert cab == cba == both  # the same config (the canonical order), so the same hash
    via_a = _migrate(_migrate(s0, c0, ca), ca, cab)
    via_b = _migrate(_migrate(s0, c0, cb), cb, cba)
    direct = _migrate(s0, c0, both)
    _same(via_a, direct)
    _same(via_b, direct)
    lay = dims(both).lay
    assert (dims(cab).lay, dims(cba).lay) == (lay, lay)
    assert ext.layout_features(cab) == ext.layout_features(cba) == ext.layout_features(both)
    assert ext.mask(cab) == ext.mask(cba) == ext.mask(both)


@pytest.mark.parametrize(
    "have", [(), ("comm",), ("structures", "niche")], ids=lambda h: "+".join(h) or "none"
)
def test_e2_all_permutations_of_the_rest_agree(have: tuple[str, ...]) -> None:
    """Every order of enabling the other features (a migration per step) equals the one-shot migration, from
    a source that already carries some features (their weights shift when an earlier tail joins)."""
    c0 = _cfg(2, have)
    s0 = _arrays(c0)
    rest = [n for n in ALL if n not in have][:4]  # 24 orders of four features
    target = _enable(c0, *rest)
    direct = _migrate(s0, c0, target)
    for order in itertools.permutations(rest):
        cfg, arrays = c0, s0
        for n in order:
            nxt = _enable(cfg, n)
            arrays = _migrate(arrays, cfg, nxt)
            cfg = nxt
        assert cfg == target
        _same(arrays, direct)


# --- the structure of the migrated weights ----------------------------------------------------------------


@pytest.mark.parametrize("nd", [2, 3])
def test_new_rows_and_columns_are_zero_and_the_old_ones_move_with_their_blocks(nd: int) -> None:
    have = ("bonds",)
    c_old, c_new = _cfg(nd, have), _cfg(nd, ALL)
    s_old = _arrays(c_old)
    s_new = _migrate(s_old, c_old, c_new)
    lo, ln = dims(c_old).lay, dims(c_new).lay
    w1o, w1n = np.array(s_old["W1"]), np.array(s_new["W1"])
    w2o, w2n = np.array(s_old["W2"]), np.array(s_new["W2"])
    base_in = lo.D_IN - sum(w for *_, w, _ in lo.EXT_IN)
    assert (w1n.shape[1], w2n.shape[2]) == (ln.D_IN + dims(c_new).h, ln.D_OUT)
    np.testing.assert_array_equal(w1n[:, :base_in], w1o[:, :base_in])
    np.testing.assert_array_equal(w1n[:, ln.D_IN :], w1o[:, lo.D_IN :])  # the recurrent rows shifted
    old_blocks = {(o, n): (st, w) for o, n, st, w, _ in lo.EXT_IN}
    for owner, name, start, width, _ in ln.EXT_IN:
        blk = w1n[:, start : start + width]
        if (owner, name) in old_blocks:  # bonds' weights are kept, at their new offsets
            st0, w0 = old_blocks[(owner, name)]
            np.testing.assert_array_equal(blk, w1o[:, st0 : st0 + w0])
        else:
            assert not blk.any(), (owner, name)
    old_out = {(o, n): (st, w) for o, n, st, w, _ in lo.EXT_OUT}
    for owner, name, start, width, _ in ln.EXT_OUT:
        blk = w2n[:, :, start : start + width]
        if (owner, name) in old_out:
            st0, w0 = old_out[(owner, name)]
            np.testing.assert_array_equal(blk, w2o[:, :, st0 : st0 + w0])
        else:
            assert not blk.any(), (owner, name)
    # the pending weights get the same treatment
    assert np.array(s_new["pend_W1"]).shape[1:] == w1n.shape[1:]
    assert np.array(s_new["pend_W2"]).shape[1:] == w2n.shape[1:]
    # brain_cost is an L1 sum: unchanged
    np.testing.assert_array_equal(np.array(s_new["brain_cost"]), np.array(s_old["brain_cost"]))


def test_msg_gets_zero_columns_for_the_voice(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = tuple(stub.spec(n, voice_w=4 if n == "comm" else 0) for n in ALL)
    monkeypatch.setattr(ext, "REGISTRY", reg)
    dims.cache_clear()
    c0, c1 = _cfg(2), _cfg(2, ("comm",))
    assert dims(c1).msg_w == 6 and dims(c0).msg_w == 2
    s0 = _arrays(c0)
    s0["msg"] = mx.ones_like(s0["msg"]) * 0.5
    s1 = _migrate(s0, c0, c1)
    msg = np.array(s1["msg"])
    assert msg.shape[1] == 6 and (msg[:, :2] == 0.5).all() and not msg[:, 2:].any()
    state_from_arrays(s1, c1)  # a state of the new config


def test_new_state_fields_take_their_rest_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """The initial value of a new field is the feature's own ``rest`` (here 7 for the soil)."""
    base = stub.spec("niche")
    field = dataclasses.replace(
        base.state_fields(None, 2)[0], rest=lambda cfg: mx.full(dims(cfg).field_shape, 7, mx.int32)
    )
    reg = tuple(
        dataclasses.replace(sp, state_fields=lambda cfg, nd, f=field: (f,)) if sp.name == "niche" else sp
        for sp in ext.REGISTRY
    )
    monkeypatch.setattr(ext, "REGISTRY", reg)
    dims.cache_clear()
    c0 = _cfg(2)
    c1 = _enable(c0, "niche", "bonds")
    out = _migrate(_arrays(c0), c0, c1)
    assert (np.array(out["soil"]) == 7).all()
    assert not np.array(out["role"]).any() and np.array(out["bond_flow"]).shape == (dims(c1).n, 4)
    assert list(out) == list(state_cls(c1)._fields)  # canonical order of the new type


# --- refusals ---------------------------------------------------------------------------------------------


def test_removal_is_refused_unless_forced() -> None:
    c_on, c_off = _cfg(2, ("bonds", "comm")), _cfg(2, ("comm",))
    s = _arrays(c_on)
    with pytest.raises(MigrationError, match=r"forced|fresh runs"):
        _migrate(s, c_on, c_off)
    out = _migrate(s, c_on, c_off, allow_removal=True)
    state_from_arrays(out, c_off)  # a state of the config without bonds: no role fields, narrower weights
    assert set(out) == set(state_cls(c_off)._fields)
    # the evolved weights of the kept block are untouched; brain_cost follows the weights that are left
    cost = brain.l1_cost(out["W1"], out["W2"], c_off)
    np.testing.assert_array_equal(np.array(out["brain_cost"]), np.array(cost))


def test_adding_and_removing_in_one_migration_is_a_removal() -> None:
    c_a, c_b = _cfg(2, ("bonds",)), _cfg(2, ("comm",))
    with pytest.raises(MigrationError, match="forced"):
        _migrate(_arrays(c_a), c_a, c_b)


def test_a_width_change_is_never_a_migration(monkeypatch: pytest.MonkeyPatch) -> None:
    """The width of a block follows a config key (here the seed stands for ``comm.voice_w``): the same feature
    on in both configs with another width is refused, even when removal is allowed."""
    wide = tuple(
        dataclasses.replace(
            sp,
            token=lambda cfg: f"bonds{cfg.run.seed}@1",
            in_blocks=lambda cfg, nd: (stub.Block("role", 1 + cfg.run.seed),),
        )
        if sp.name == "bonds"
        else sp
        for sp in ext.REGISTRY
    )
    monkeypatch.setattr(ext, "REGISTRY", wide)
    dims.cache_clear()
    c1 = load_config(preset="tiny", overrides=[*F32, "run.seed=1"])
    c1 = stub.stub_cfg(c1, "bonds")
    c2 = dataclasses.replace(c1, run=dataclasses.replace(c1.run, seed=2))
    with pytest.raises(MigrationError, match="never a migration"):
        layout_change(c1, c2, allow_removal=True)
    with pytest.raises(MigrationError):
        _migrate(_arrays(c1), c1, c2, allow_removal=True)


def test_a_different_base_layout_is_refused() -> None:
    c0 = _cfg(2)
    other = load_config(preset="tiny", overrides=[*F32, "brain.hidden=24"])
    with pytest.raises(MigrationError, match="base layout differs"):
        layout_change(c0, other)
    c3 = _cfg(3)
    with pytest.raises(MigrationError, match="base layout differs"):
        layout_change(c0, c3)


# --- the equivalence gate: zero rows reproduce the old behaviour ------------------------------------------


@pytest.mark.parametrize("nd", [2, 3])
@pytest.mark.parametrize(("dtype", "atol"), [("float32", 1e-6), ("float16", 5e-3)])
def test_the_brain_of_a_migrated_world_is_the_old_brain_plus_silent_outputs(
    nd: int, dtype: str, atol: float
) -> None:
    """With zero rows/columns: the same hidden state and legacy outputs, and the new outputs are exactly 0,
    whatever the new inputs are."""
    c_old, c_new = _cfg(nd, ("bonds",), dtype), _cfg(nd, ALL, dtype)
    arr = _arrays(c_old)
    assert arr["W1"].dtype == getattr(mx, dtype)
    new = _migrate(arr, c_old, c_new)
    lo, ln = dims(c_old).lay, dims(c_new).lay
    n = dims(c_old).n
    x_old = mx.random.normal((n, lo.D_IN), key=mx.random.key(1))
    x_new = mx.random.normal((n, ln.D_IN), key=mx.random.key(2))  # the new inputs are noise
    base_in = lo.D_IN - sum(w for *_, w, _ in lo.EXT_IN)
    old_blocks = {(o, nm): (st, w) for o, nm, st, w, _ in lo.EXT_IN}
    for owner, name, start, width, _ in ln.EXT_IN:  # the old blocks carry the old inputs
        if (owner, name) in old_blocks:
            st0, _ = old_blocks[(owner, name)]
            x_new = mx.concatenate(
                [x_new[:, :start], x_old[:, st0 : st0 + width], x_new[:, start + width :]], 1
            )
    x_new = mx.concatenate([x_old[:, :base_in], x_new[:, base_in:]], axis=1)
    h = arr["h"]
    h_old, y_old = brain.forward(arr["W1"], arr["W2"], x_old, h, c_old)
    h_new, y_new = brain.forward(new["W1"], new["W2"], x_new, h, c_new)
    np.testing.assert_allclose(np.array(h_new), np.array(h_old), atol=atol, rtol=0)
    base_out = lo.D_OUT - sum(w for *_, w, _ in lo.EXT_OUT)
    np.testing.assert_allclose(
        np.array(y_new[:, :base_out]), np.array(y_old[:, :base_out]), atol=atol, rtol=atol
    )
    old_out = {(o, nm) for o, nm, *_ in lo.EXT_OUT}
    for owner, name, start, width, _ in ln.EXT_OUT:
        if (owner, name) not in old_out:
            assert not np.array(y_new[:, start : start + width]).any()


@pytest.mark.parametrize("nd", [2, 3])
@pytest.mark.parametrize("names", [("structures",), ("comm", "niche"), stub.WITHOUT_ARMS])
def test_one_step_of_the_migrated_world_equals_the_source(
    nd: int, names, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate of PR 0b: every new feature's effects off (no stub due), one step of the migrated population
    reproduces the source's step on the shared state; the extension counters are the only addition."""
    stub.install(monkeypatch, due=False)
    c_old = _cfg(nd)
    c_new = _enable(c_old, *names)
    s_old = init_state(c_old)
    arrays = _migrate(state_to_arrays(s_old), c_old, c_new)
    s_new = state_from_arrays(arrays, c_new)
    # a birth-free window: reproduction is unrequested for one step (mutation draws of the new shapes would
    # differ at the first birth, documented in §43.8)
    out_old = make_step(c_old, compile=False)(s_old)
    out_new = make_step(c_new, compile=False)(s_new)
    (t_old, so), (t_new, sn) = out_old, out_new
    mx.eval(t_old, t_new)
    # the window is birth-free (a birth draws its mutation noise with the new shapes, §43.8): asserted
    assert int(to_numpy(t_old.next_uid)) == int(to_numpy(s_old.next_uid))
    assert int(to_numpy(t_new.next_uid)) == int(to_numpy(s_old.next_uid))
    weights = {"W1", "W2", "pend_W1", "pend_W2"}
    for name in type(t_old)._fields:
        if name in weights:
            continue
        a, b = np.array(getattr(t_old, name)), np.array(getattr(t_new, name))
        if name == "msg":
            b = b[:, : a.shape[1]]
        if a.dtype.kind == "f":  # the GEMV sums extra zero terms: float32 reduction order, a few ulp
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6, err_msg=name)
        else:
            np.testing.assert_array_equal(a, b, err_msg=name)
    np.testing.assert_array_equal(to_numpy(so.ledger), to_numpy(sn.ledger))


# --- the run-directory tool -------------------------------------------------------------------------------


def _source_run(tmp_path: Path, names=(), nd: int = 2, *, compressed: bool = False) -> tuple[Path, Path]:
    """A run directory with one checkpoint (step 0 of a stub-config world); returns (run, checkpoint)."""
    cfg = _cfg(nd, names)
    rd = RunDir.create(tmp_path / f"src_{nd}_{'-'.join(names) or 'none'}", cfg)
    s = init_state(cfg)
    if compressed:
        rd.compress = True
    path = rd.save_checkpoint(s, cfg)
    rd.wait_saves()
    return rd.path, path


def test_cli_migrates_a_run_and_the_source_is_untouched(tmp_path: Path, capsys) -> None:
    run, ck = _source_run(tmp_path)
    before = (ck.read_bytes(), (run / "meta.json").read_bytes(), (run / "config.json").read_bytes())
    out = tmp_path / "out"
    code = cli.main(["migrate", str(run), "--enable", "bonds", "--enable", "comm", "--out", str(out)])
    assert code == 0, capsys.readouterr().err
    assert before == (ck.read_bytes(), (run / "meta.json").read_bytes(), (run / "config.json").read_bytes())
    meta = RunDir.open(out).meta()
    cfg = RunDir.open(out).config()
    assert meta["layout_features"] == "bonds@1,comm4@1" == ext.layout_features(cfg)
    assert meta["genome_layout_version"] == 1 + 4 * ext.mask(cfg) and meta["start_step"] == 0
    prov = meta["migrated_from"]
    assert (
        prov["added"] == ["bonds", "comm"] and prov["removed"] == [] and prov["source_layout_features"] == ""
    )
    assert prov["source_genome_layout_version"] == 1 and meta["parent"].startswith(str(run))
    (new_ck,) = [p for _, p in RunDir.open(out).checkpoints()]
    assert checkpoint_metadata(new_ck)["layout_features"] == "bonds@1,comm4@1"
    load_checkpoint(new_ck, cfg)  # loads with the config of the run
    assert not runio.is_compressed(new_ck)


def test_cli_chain_equals_one_call(tmp_path: Path) -> None:
    """E2 through the tool: two migrations in either order and one call give the same state and layout."""
    run, _ = _source_run(tmp_path)
    one, ab, ba = tmp_path / "one", tmp_path / "ab", tmp_path / "ba"
    assert (
        cli.main(
            ["migrate", str(run), "--enable", "structures", "--enable", "niche", "--out", str(one), "--quiet"]
        )
        == 0
    )
    assert (
        cli.main(["migrate", str(run), "--enable", "structures", "--out", str(tmp_path / "a"), "--quiet"])
        == 0
    )
    assert cli.main(["migrate", str(tmp_path / "a"), "--enable", "niche", "--out", str(ab), "--quiet"]) == 0
    assert cli.main(["migrate", str(run), "--enable", "niche", "--out", str(tmp_path / "b"), "--quiet"]) == 0
    assert (
        cli.main(["migrate", str(tmp_path / "b"), "--enable", "structures", "--out", str(ba), "--quiet"]) == 0
    )
    digests = []
    for d in (one, ab, ba):
        rd = RunDir.open(d)
        cfg = rd.config()
        digests.append(
            (
                state_digest(load_checkpoint(rd.latest_checkpoint(), cfg)),
                rd.meta()["layout_features"],
                rd.meta()["genome_layout_version"],
            )
        )
    assert digests[0] == digests[1] == digests[2]


def test_a_compressed_source_gives_a_compressed_output(tmp_path: Path) -> None:
    run, ck = _source_run(tmp_path, compressed=True)
    assert runio.is_compressed(ck)
    out = tmp_path / "out"
    assert cli.main(["migrate", f"{run}:0", "--enable", "comm", "--out", str(out), "--quiet"]) == 0
    rd = RunDir.open(out)
    new_ck = rd.latest_checkpoint()
    assert runio.is_compressed(new_ck)
    assert runio.state_file_is_complete(new_ck, 0)
    s = load_checkpoint(new_ck, rd.config())
    plain = init_state(_cfg(2))
    # the migrated population is the source's on the shared fields
    assert np.array_equal(np.array(s.pos), np.array(plain.pos)) and np.array_equal(
        np.array(s.energy), np.array(plain.energy)
    )


def test_a_bare_state_file_and_a_plain_source(tmp_path: Path) -> None:
    cfg = _cfg(3)
    path = tmp_path / "s.safetensors"
    write_state_file(path, init_state(cfg), cfg)
    out = tmp_path / "out"
    assert cli.main(["migrate", str(path), "--enable", "bonds", "--out", str(out), "--quiet"]) == 0
    rd = RunDir.open(out)
    assert rd.meta()["dims"] == 3 and not runio.is_compressed(rd.latest_checkpoint())


def test_cli_refuses_removal_and_force_allows_it(tmp_path: Path, capsys) -> None:
    run, _ = _source_run(tmp_path, ("bonds", "comm"))
    out = tmp_path / "out"
    code = cli.main(["migrate", str(run), "--set", "run.llm_model=stub:comm", "--out", str(out)])
    assert code == 2 and "migration refused" in capsys.readouterr().err and not out.exists()
    code = cli.main(
        [
            "migrate",
            str(run),
            "--set",
            "run.llm_model=stub:comm",
            "--force-removal",
            "--out",
            str(out),
            "--quiet",
        ]
    )
    assert code == 0
    meta = RunDir.open(out).meta()
    assert meta["layout_features"] == "comm4@1" and meta["migrated_from"]["forced_removal"] is True
    assert meta["migrated_from"]["removed"] == ["bonds"]


def test_cli_only_changes_feature_keys(tmp_path: Path, capsys) -> None:
    run, _ = _source_run(tmp_path)
    out = tmp_path / "out"
    code = cli.main(
        ["migrate", str(run), "--enable", "bonds", "--set", "energy.structure=1", "--out", str(out)]
    )
    assert code == 2 and "feature keys only" in capsys.readouterr().err and not out.exists()
    assert cli.main(["migrate", str(run), "--out", str(out)]) == 2  # nothing to migrate
    assert cli.main(["migrate", str(run), "--enable", "nosuch", "--out", str(out)]) == 2
    assert not out.exists()


def test_cli_refuses_an_existing_output(tmp_path: Path) -> None:
    run, _ = _source_run(tmp_path)
    out = tmp_path / "out"
    assert cli.main(["migrate", str(run), "--enable", "bonds", "--out", str(out), "--quiet"]) == 0
    assert cli.main(["migrate", str(run), "--enable", "bonds", "--out", str(out), "--quiet"]) == 1
    assert (
        cli.main(["migrate", str(run), "--enable", "bonds", "--out", str(out), "--overwrite", "--quiet"]) == 0
    )


def test_the_help_says_leave_one_out_arms_are_fresh_runs(capsys) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["migrate", "--help"])
    out = capsys.readouterr().out
    assert "FRESH RUNS" in out and "E23" in out and "--force-removal" in out


# --- E3, the test side: a migrated world resumes, a mismatched config is refused --------------------------


def test_e3_a_migrated_world_resumes_and_a_mismatched_config_is_refused(tmp_path: Path, capsys) -> None:
    run, _ = _source_run(tmp_path)
    out = tmp_path / "out"
    assert (
        cli.main(
            ["migrate", str(run), "--enable", "bonds", "--enable", "niche", "--out", str(out), "--quiet"]
        )
        == 0
    )
    rd = RunDir.open(out)
    cfg = rd.config()
    # resume: the loader accepts the migrated checkpoint under the run's own config ...
    s = load_checkpoint(rd.latest_checkpoint(), cfg)
    assert int(to_numpy(s.step)) == 0
    make_step(cfg, compile=False)(s)  # and the world steps
    # ... and the CLI resumes it (zero further steps: the run's step budget is the source's remaining ones)
    assert cli.main(["resume", str(out), "--steps", "2", "--quiet"]) == 0
    assert int(checkpoint_metadata(RunDir.open(out).latest_checkpoint())["step"]) == 2
    # a mismatched config (one more feature, one fewer, or another one) is refused by the loader
    ck = RunDir.open(out).checkpoints()[0][1]
    for names in (ALL, ("bonds",), ("comm", "niche")):
        with pytest.raises(CheckpointError, match="layout_features") as e:
            load_checkpoint(ck, _cfg(2, names), check_hash=False)
        assert "alife migrate" in str(e.value)
    # the unmigrated source cannot be loaded into the migrated config either (nothing is guessed)
    with pytest.raises(CheckpointError, match="layout_features"):
        load_checkpoint(_source_run(tmp_path / "s2")[1], cfg, check_hash=False)


# --- review fixes: the output may not endanger the source, locks, conserved fields, atomicity -------------


def _migrate_cli(*args: str) -> int:
    return cli.main(["migrate", *args, "--quiet"])


def _snap(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_h1_overwrite_never_replaces_the_source(tmp_path: Path, capsys) -> None:
    run, ck = _source_run(tmp_path / "A")
    before = _snap(run)
    # out == source run
    assert _migrate_cli(str(run), "--enable", "bonds", "--out", str(run), "--overwrite") == 2
    assert "migration refused" in capsys.readouterr().err and _snap(run) == before
    # out is an ancestor of the source (a directory with a meta.json)
    par = tmp_path / "C"
    par.mkdir()
    (par / "meta.json").write_text("{}")
    run2, _ = _source_run(par)
    before2 = _snap(run2)
    assert _migrate_cli(str(run2), "--enable", "bonds", "--out", str(par), "--overwrite") == 2
    assert run2.exists() and _snap(run2) == before2
    # out inside the source
    inner = run / "checkpoints" / "x"
    assert _migrate_cli(str(run), "--enable", "bonds", "--out", str(inner)) == 2
    assert not inner.exists() and _snap(run) == before
    # a bare state file inside a run, out = its run directory
    assert _migrate_cli(str(ck), "--enable", "bonds", "--out", str(run), "--overwrite") == 2
    assert ck.exists() and _snap(run) == before
    # a symbolic link out
    link = tmp_path / "link"
    link.symlink_to(run)
    assert _migrate_cli(str(run), "--enable", "bonds", "--out", str(link), "--overwrite") == 2
    assert run.exists() and _snap(run) == before


def test_m1_a_run_being_played_is_not_replaced(tmp_path: Path, capsys) -> None:
    import fcntl

    run, _ = _source_run(tmp_path / "s")
    victim, _ = _source_run(tmp_path / "v", ("niche",))
    before = _snap(victim)
    fd = open(victim / "play.lock", "w")  # noqa: SIM115
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert _migrate_cli(str(run), "--enable", "bonds", "--out", str(victim), "--overwrite") == 2
        assert "play.lock" in capsys.readouterr().err
        # the pre-existing hole of RunDir.create(overwrite=True) (alife run --overwrite)
        with pytest.raises(FileExistsError, match=r"play\.lock"):
            RunDir.create(victim, _cfg(2), overwrite=True)
        assert {k: v for k, v in _snap(victim).items() if k != "play.lock"} == {
            k: v for k, v in before.items() if k != "play.lock"
        }
    finally:
        fd.close()


def test_a_failed_migration_leaves_no_output_and_overwrite_is_swapped_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _ = _source_run(tmp_path / "s")
    out = tmp_path / "out"
    monkeypatch.setattr(
        RunDir, "save_checkpoint", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    with pytest.raises(RuntimeError):
        migrate.migrate_checkpoint(runio_latest(run), out, enable=["bonds"])
    assert not out.exists() and not [p for p in tmp_path.iterdir() if p.name.startswith(".out")]
    monkeypatch.undo()
    stub.install(monkeypatch)
    assert _migrate_cli(str(run), "--enable", "bonds", "--out", str(out)) == 0
    assert _migrate_cli(str(run), "--enable", "comm", "--out", str(out), "--overwrite") == 0
    assert RunDir.open(out).meta()["layout_features"] == "comm4@1"
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".out")]


def runio_latest(run: Path) -> Path:
    return RunDir.open(run).latest_checkpoint()


def test_noop_and_pointless_force_are_refused(tmp_path: Path, capsys) -> None:
    run, _ = _source_run(tmp_path / "s", ("bonds",))
    out = tmp_path / "out"
    assert _migrate_cli(str(run), "--enable", "bonds", "--out", str(out)) == 2
    assert "nothing to migrate" in capsys.readouterr().err and not out.exists()
    run0, _ = _source_run(tmp_path / "s0")
    assert _migrate_cli(str(run0), "--enable", "comm", "--force-removal", "--out", str(out)) == 2
    assert "no feature is removed" in capsys.readouterr().err and not out.exists()


def test_provenance_stores_absolute_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, ck = _source_run(tmp_path / "s")
    monkeypatch.chdir(tmp_path)
    assert _migrate_cli(str(ck.relative_to(tmp_path)), "--enable", "bonds", "--out", "out") == 0
    meta = RunDir.open(tmp_path / "out").meta()
    assert Path(meta["migrated_from"]["source"]).is_absolute() and Path(meta["parent"]).is_absolute()


def test_m3_a_removed_field_with_conserved_quanta_must_be_at_rest(monkeypatch: pytest.MonkeyPatch) -> None:
    """A field that holds energy quanta can be dropped by a forced removal only while it is at rest."""
    reg = tuple(
        dataclasses.replace(
            sp,
            state_fields=lambda cfg, nd: tuple(
                dataclasses.replace(f, conserved=True) for f in stub.spec("niche").state_fields(cfg, nd)
            ),
        )
        if sp.name == "niche"
        else sp
        for sp in ext.REGISTRY
    )
    monkeypatch.setattr(ext, "REGISTRY", reg)
    dims.cache_clear()
    c_on, c_off = _cfg(2, ("niche",)), _cfg(2)
    arrays = _arrays(c_on)
    out = _migrate(arrays, c_on, c_off, allow_removal=True)  # at rest: dropped
    assert "soil" not in out
    arrays = {**arrays, "soil": arrays["soil"] + 3}
    with pytest.raises(MigrationError, match="conserved quanta"):
        _migrate(arrays, c_on, c_off, allow_removal=True)
    # a field that is not marked is dropped under --force-removal whatever it holds (the flag is the consent)
    plain_on = _cfg(2, ("structures",))
    arr = {**_arrays(plain_on)}
    arr["mat"] = arr["mat"] + 5
    assert "mat" not in _migrate(arr, plain_on, _cfg(2), allow_removal=True)


# --- built structures on an evolved world (DESIGN §38, §43.8) ---------------------------------------------


@pytest.mark.parametrize("preset", ["tiny", "tiny3d"])
def test_migrating_an_evolved_world_onto_structures(preset: str, tmp_path: Path, capsys, monkeypatch) -> None:
    """The real registry (not the stub): an evolved world gets zero ``material`` rows and a zero ``build``
    column, ``mat`` starts at 0, nobody builds until mutation finds it, and the energy ledger and the material
    identity close after the migration."""
    monkeypatch.undo()  # drop the autouse stub registry
    dims.cache_clear()
    from alife.state import counter_index

    cfg = load_config(preset=preset, overrides=F32)
    fn, s = make_step(cfg), init_state(cfg)
    for _ in range(40):  # births, deaths and attacks have happened
        s, _ = fn(s)
    mx.eval(s)
    rd = RunDir.create(tmp_path / "src", cfg)
    rd.save_checkpoint(s, cfg)
    rd.wait_saves()
    out = tmp_path / "out"
    code = cli.main(["migrate", str(rd.path), "--enable", "structures", "--out", str(out), "--quiet"])
    assert code == 0, capsys.readouterr().err
    run = RunDir.open(out)
    cfg2, meta = run.config(), run.meta()
    d = dims(cfg2)
    assert cfg2.structures.enabled and meta["layout_features"] == "structures@1"
    assert meta["migrated_from"]["added"] == ["structures"]
    s2 = load_checkpoint(run.checkpoints()[-1][1], cfg2)
    alive = np.array(s2.alive)
    assert not np.array(s2.mat).any()  # the material starts empty
    assert not np.array(s2.W1)[alive][:, d.lay.IN_MATERIAL].any()  # zero new row
    assert not np.array(s2.W2)[alive][:, :, d.lay.OUT_BUILD].any()  # zero new column
    fn2, first, prev_total = make_step(cfg2), None, 0
    names = ("BUILD_UNITS", "MAT_WASTED", "MAT_DECAYED", "MAT_TOTAL")
    for _ in range(40):
        s2, o2 = fn2(s2)
        mx.eval(s2, o2)
        c = to_numpy(o2.counters)
        b, w, dcy, tot = (int(c[counter_index(cfg2, n)]) for n in names)
        assert tot == prev_total + b - w - dcy
        prev_total, first = tot, b if first is None else first
        led = to_numpy(o2.ledger)
        assert led[LedgerIdx.FLOW_STOLEN] == led[LedgerIdx.FLOW_ATTACK_GAIN] + led[LedgerIdx.FLOW_SPILL]
    assert int(to_numpy(s2.diag)) & 0xFFFF == 0
    assert first == 0  # zero build weights: nobody builds at the first step (mutation at births may wake it)
    from alife.ext import REGISTRY

    spec = next(sp for sp in REGISTRY if sp.name == "structures")
    assert all(not f.conserved for f in spec.state_fields(cfg2, d.nd))  # the material holds no energy


# --- bonds that pay on an evolved world (DESIGN §40, §43.8) -----------------------------------------------


@pytest.mark.parametrize("preset", ["tiny", "tiny3d"])
def test_migrating_an_evolved_world_onto_bonds(preset: str, tmp_path: Path, capsys, monkeypatch) -> None:
    """The real registry: ``--enable bonds`` (``roles.enabled``) plus ``--set bonds.share_mode=2`` gives the
    evolved population 13 zero input rows and 2 zero output columns, empty role state, and a conductive world
    whose ledger closes; the new fields hold no energy (``conserved`` False)."""
    monkeypatch.undo()  # drop the autouse stub registry
    dims.cache_clear()
    from alife.state import counter_index

    cfg = load_config(preset=preset, overrides=F32)
    fn, s = make_step(cfg), init_state(cfg)
    for _ in range(40):
        s, _ = fn(s)
    mx.eval(s)
    rd = RunDir.create(tmp_path / "src", cfg)
    rd.save_checkpoint(s, cfg)
    rd.wait_saves()
    out = tmp_path / "out"
    code = cli.main(
        [
            "migrate",
            str(rd.path),
            "--enable",
            "bonds",
            "--set",
            "bonds.share_mode=2",
            "--out",
            str(out),
            "--quiet",
        ]
    )
    assert code == 0, capsys.readouterr().err
    run = RunDir.open(out)
    cfg2, meta = run.config(), run.meta()
    d = dims(cfg2)
    assert cfg2.roles.enabled and cfg2.bonds.share_mode == 2 and meta["layout_features"] == "bonds@1"
    assert meta["migrated_from"]["added"] == ["bonds"]
    s2 = load_checkpoint(run.checkpoints()[-1][1], cfg2)
    alive = np.array(s2.alive)
    assert (
        not np.array(s2.role).any() and not np.array(s2.role_dwell).any() and not np.array(s2.bond_flow).any()
    )
    w1, w2 = np.array(s2.W1)[alive], np.array(s2.W2)[alive]
    for name in ("IN_ROLE", "IN_BOND_ROLE", "IN_BOND_KIN", "IN_BOND_FLOW"):
        lo = getattr(d.lay, name)
        assert not w1[:, lo : lo + (1 if name == "IN_ROLE" else 4)].any(), name  # zero new rows
    assert not w2[:, :, d.lay.OUT_ROLE_A : d.lay.OUT_ROLE_B + 1].any()  # zero new columns
    fn2, active = make_step(cfg2), 0
    for _ in range(40):
        s2, o2 = fn2(s2)
        mx.eval(s2, o2)
        led = to_numpy(o2.ledger)
        assert led[LedgerIdx.FLOW_STOLEN] == led[LedgerIdx.FLOW_ATTACK_GAIN] + led[LedgerIdx.FLOW_SPILL]
        active += int(to_numpy(o2.counters)[counter_index(cfg2, "SHARE_SLOTS_ACTIVE")])
    assert int(to_numpy(s2.diag)) & 0xFFFF == 0
    assert active > 0  # the obligate share runs on the migrated world at once
    from alife.ext import REGISTRY

    spec = next(sp for sp in REGISTRY if sp.name == "bonds")
    assert all(not f.conserved for f in spec.state_fields(cfg2, d.nd))
    # the enable hook alone allocates the layout and leaves the sharing at the legacy gift
    assert spec.enable(cfg).roles.enabled and spec.enable(cfg).bonds.share_mode == 0


# --- communication on an evolved world: the silent start (DESIGN §39.6, §43.8) ----------------------------


@pytest.mark.parametrize("preset", ["tiny", "tiny3d"])
def test_migrating_an_evolved_world_onto_comm_is_a_silent_start(
    preset: str, tmp_path: Path, capsys, monkeypatch
) -> None:
    """The real registry: ``--enable comm`` gives the evolved population W + nd zero input rows (before the
    recurrent rows), W zero output columns and W zero ``msg`` columns; nobody speaks at the first step
    (``max(tanh(0), 0) = 0``), the ledger closes, and a one-step equivalence holds: the migrated brain makes
    the same legacy actions (the same positions, energies and ledger) as the unmigrated one."""
    monkeypatch.undo()  # drop the autouse stub registry
    dims.cache_clear()
    from alife.state import counter_index, migrate_state

    cfg = load_config(preset=preset, overrides=F32)
    fn, s = make_step(cfg), init_state(cfg)
    for _ in range(40):
        s, _ = fn(s)
    mx.eval(s)
    rd = RunDir.create(tmp_path / "src", cfg)
    rd.save_checkpoint(s, cfg)
    rd.wait_saves()
    out = tmp_path / "out"
    assert cli.main(["migrate", str(rd.path), "--enable", "comm", "--out", str(out), "--quiet"]) == 0, (
        capsys.readouterr().err
    )
    run = RunDir.open(out)
    cfg2, meta = run.config(), run.meta()
    d = dims(cfg2)
    assert (
        cfg2.comm.enabled
        and meta["layout_features"] == "comm4@1"
        and meta["migrated_from"]["added"] == ["comm"]
    )
    s2 = load_checkpoint(run.checkpoints()[-1][1], cfg2)
    alive = np.array(s2.alive)
    assert s2.msg.shape == (d.n, layout.MSG + 4) and not np.array(s2.msg)[:, layout.MSG :].any()
    w1, w2 = np.array(s2.W1)[alive], np.array(s2.W2)[alive]
    lo = d.lay.IN_VOICE
    assert not w1[:, lo : lo + 4 + d.nd].any() and not w2[:, :, d.lay.OUT_VOICE : d.lay.OUT_VOICE + 4].any()
    # the legacy weights moved only by the inserted rows: the recurrent rows start after the new tail
    old = np.array(s.W1)[alive]
    assert np.array_equal(w1[:, : d.lay.D_IN - 4 - d.nd], old[:, : d.lay.D_IN - 4 - d.nd])
    assert np.array_equal(w1[:, d.lay.D_IN :], old[:, d.lay.D_IN - 4 - d.nd :])
    fn2 = make_step(cfg2)
    s2, o2 = fn2(s2)
    mx.eval(s2, o2)
    assert int(to_numpy(o2.counters)[counter_index(cfg2, "COMM_SPEAKERS")]) == 0  # silent at the first step
    for _ in range(40):
        s2, o2 = fn2(s2)
        mx.eval(s2, o2)
        led = to_numpy(o2.ledger)
        assert led[LedgerIdx.FLOW_STOLEN] == led[LedgerIdx.FLOW_ATTACK_GAIN] + led[LedgerIdx.FLOW_SPILL]
    assert int(to_numpy(s2.diag)) & 0xFFFF == 0
    # the enable hook alone switches comm on and leaves every other key where it was
    spec = next(sp for sp in ext.REGISTRY if sp.name == "comm")
    assert (
        spec.enable(cfg).comm.enabled
        and spec.enable(cfg).comm.cost == 4.0
        and spec.enable(cfg).comm.control == "none"
    )
    assert all(not f.conserved for f in spec.state_fields(cfg2, d.nd))

    # one-step equivalence, with reproduction unrequested (mutation noise is drawn with the new shape)
    quiet = load_config(preset=preset, overrides=[*F32, "repro.min_age=100000"])
    q_new = spec.enable(quiet)
    sq = init_state(quiet)
    f0 = make_step(quiet)
    for _ in range(10):
        sq, _ = f0(sq)
    mx.eval(sq)
    sm = migrate_state(sq, quiet, q_new)
    a, oa = make_step(quiet)(sq)
    b, ob = make_step(q_new)(sm)
    mx.eval(a, b, oa, ob)
    for name in (
        "energy",
        "alive",
        "resource",
        "partner",
    ):  # exact; the floats to rounding (Metal GEMV order)
        assert np.array_equal(to_numpy(getattr(a, name)), to_numpy(getattr(b, name))), name
    for name in ("pos", "vel", "heading"):
        np.testing.assert_allclose(
            to_numpy(getattr(a, name)), to_numpy(getattr(b, name)), atol=1e-5, err_msg=name
        )
    assert np.array_equal(to_numpy(oa.ledger), to_numpy(ob.ledger))
    assert np.allclose(to_numpy(a.h), to_numpy(b.h), atol=1e-6)
    assert not to_numpy(b.msg)[:, layout.MSG :].any()
    np.testing.assert_allclose(to_numpy(a.msg), to_numpy(b.msg)[:, : layout.MSG], atol=1e-5)


# --- farming on an evolved world: the silent start (DESIGN §41, §43.8) --------------------------------------


@pytest.mark.parametrize("preset", ["tiny", "tiny3d"])
def test_migrating_an_evolved_world_onto_farming_is_a_silent_start(
    preset: str, tmp_path: Path, capsys, monkeypatch
) -> None:
    """The real registry: ``--enable niche`` gives the evolved population one zero ``farm`` input row and one
    zero ``farm_put`` output column, ``soil`` starts at 0, nobody farms at the first step (``max(tanh(0), 0) =
    0``), the ledger and the soil identity close, and a one-step equivalence holds: the migrated brain makes
    the same legacy actions (the same positions, energies, resource and ledger) as the unmigrated one."""
    monkeypatch.undo()  # drop the autouse stub registry
    dims.cache_clear()
    from alife.state import counter_index, migrate_state

    cfg = load_config(preset=preset, overrides=F32)
    fn, s = make_step(cfg), init_state(cfg)
    for _ in range(40):
        s, _ = fn(s)
    mx.eval(s)
    rd = RunDir.create(tmp_path / "src", cfg)
    rd.save_checkpoint(s, cfg)
    rd.wait_saves()
    out = tmp_path / "out"
    assert cli.main(["migrate", str(rd.path), "--enable", "niche", "--out", str(out), "--quiet"]) == 0, (
        capsys.readouterr().err
    )
    run = RunDir.open(out)
    cfg2, meta = run.config(), run.meta()
    d = dims(cfg2)
    assert (
        cfg2.niche.farm
        and meta["layout_features"] == "niche.farm@1"
        and meta["migrated_from"]["added"] == ["niche"]
    )
    s2 = load_checkpoint(run.checkpoints()[-1][1], cfg2)
    alive = np.array(s2.alive)
    assert not np.array(s2.soil).any()  # the soil starts empty
    w1, w2 = np.array(s2.W1)[alive], np.array(s2.W2)[alive]
    assert not w1[:, d.lay.IN_FARM].any() and not w2[:, :, d.lay.OUT_FARM_PUT].any()  # zero row, zero column
    old = np.array(s.W1)[
        alive
    ]  # the legacy rows moved only by the inserted one (recurrent rows after the tail)
    assert np.array_equal(w1[:, : d.lay.D_IN - 1], old[:, : d.lay.D_IN - 1])
    assert np.array_equal(w1[:, d.lay.D_IN :], old[:, d.lay.D_IN - 1 :])
    fn2 = make_step(cfg2)
    s2, o2 = fn2(s2)
    mx.eval(s2, o2)
    assert int(to_numpy(o2.counters)[counter_index(cfg2, "FARMERS")]) == 0  # silent at the first step
    tot = int(np.array(s2.soil).sum())
    for _ in range(40):
        s2, o2 = fn2(s2)
        mx.eval(s2, o2)
        c, led = to_numpy(o2.counters), to_numpy(o2.ledger)
        tot += int(c[counter_index(cfg2, "SOIL_PUT")]) - int(c[counter_index(cfg2, "SOIL_DECAYED")])
        assert tot == int(np.array(s2.soil).sum())
        assert led[LedgerIdx.FLOW_STOLEN] == led[LedgerIdx.FLOW_ATTACK_GAIN] + led[LedgerIdx.FLOW_SPILL]
    assert int(to_numpy(s2.diag)) & 0xFFFF == 0
    # the enable hook alone switches farming on and leaves every other key where it was
    spec = next(sp for sp in ext.REGISTRY if sp.name == "niche")
    assert spec.enable(cfg).niche.farm and spec.enable(cfg).niche.farm_cost == 30
    assert spec.enable(cfg).niche.silent_init and spec.enable(cfg).niche.control == "none"
    assert all(not f.conserved for f in spec.state_fields(cfg2, d.nd))

    # one-step equivalence, with reproduction unrequested (mutation noise is drawn with the new shape)
    quiet = load_config(preset=preset, overrides=[*F32, "repro.min_age=100000"])
    q_new = spec.enable(quiet)
    sq = init_state(quiet)
    f0 = make_step(quiet)
    for _ in range(10):
        sq, _ = f0(sq)
    mx.eval(sq)
    sm = migrate_state(sq, quiet, q_new)
    a, oa = make_step(quiet)(sq)
    b, ob = make_step(q_new)(sm)
    mx.eval(a, b, oa, ob)
    for name in ("energy", "alive", "resource", "partner"):
        assert np.array_equal(to_numpy(getattr(a, name)), to_numpy(getattr(b, name))), name
    for name in ("pos", "vel", "heading"):
        np.testing.assert_allclose(
            to_numpy(getattr(a, name)), to_numpy(getattr(b, name)), atol=1e-5, err_msg=name
        )
    assert np.array_equal(to_numpy(oa.ledger), to_numpy(ob.ledger))
    assert np.allclose(to_numpy(a.h), to_numpy(b.h), atol=1e-6)
    assert not to_numpy(b.soil).any()
