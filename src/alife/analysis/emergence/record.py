"""The window recorder (DESIGN §44.2): dense frames of a run window, re-simulated and certified.

The checkpoints of a run (every 10 000 steps) are too sparse for any trajectory measure. ``record_window``
replays a window from the newest save at or before its start with the engine's own step (as
``analysis.resim`` does: one blocking eval per step), reads the post-step state every ``tau`` steps (and the
members of the largest bodies every ``fine_tau``), and writes ``frames_<T0>_<T1>.npz`` outside the run.

* **Certified.** The replay is compared with the recorded run before a file is written (edge case E21):
  every ``metrics.jsonl`` row on the way is recomputed and compared (``resim.compare_row``), and the state
  digest is compared with every kept checkpoint the replay passes, the window end included when it is a save.
  A difference raises ``RecordMismatch`` and no file is written. The same device, MLX version and config as
  the run are assumed (the §14.2 determinism contract).
* **Needs the start save** (E22). A run that kept only its newest save (``saves single``/``newest``, or
  ``run.keep_checkpoints`` too small) cannot be measured: ``record_window`` refuses with the saves it found.
  Measurement runs use saves ``all`` with ``keep_checkpoints`` covering every window start.
* **Observer only.** The state that is stepped is never written; the run directory is read only.
* **Idempotent.** A finished file with the same parameters is not rewritten.

This is the only module of the package that touches MLX.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from alife import metrics, runio, spatial, structures
from alife.analysis import resim
from alife.analysis.emergence import traits as traits_mod
from alife.analysis.emergence.frames import (
    FRAMES_VERSION,
    Frames,
    FramesError,
    quantise,
    read_frames,
    write_frames,
)
from alife.config import dims
from alife.runio import RunDir
from alife.state import state_digest, to_numpy
from alife.step import make_step

MAX_LEAD = 50_000  # steps between the start save and T0 beyond which the replay is refused


class RecordError(RuntimeError):
    """The window cannot be recorded (no start save, bad arguments, nothing to verify against)."""


class RecordMismatch(RecordError):
    """The replay differs from the recorded run (a row of ``metrics.jsonl`` or a checkpoint fingerprint)."""


def frames_path(out_dir: Path, t0: int, t1: int) -> Path:
    return Path(out_dir) / f"frames_{t0}_{t1}.npz"


def start_checkpoint(rd: RunDir, t0: int, max_lead: int = MAX_LEAD) -> tuple[int, Path]:
    """The newest save at or before ``t0`` (E22); ``RecordError`` naming the saves found if there is none."""
    kept = rd.checkpoints()
    before = [(st, p) for st, p in kept if st <= t0]
    if not before:
        raise RecordError(
            f"{rd.path}: measurement needs a save at or before step {t0}, but the saves kept are "
            f"{[st for st, _ in kept]} (a run saved with `saves single` or `newest`, or with "
            f"run.keep_checkpoints too small, drops older saves). Measurement runs must use saves `all` "
            f"with keep_checkpoints covering every window start."
        )
    st, p = before[-1]
    if t0 - st > max_lead:
        raise RecordError(
            f"{rd.path}: the newest save at or before step {t0} is step {st}, {t0 - st} steps earlier "
            f"(limit {max_lead}); keep a closer save (saves `all`, a larger run.keep_checkpoints)"
        )
    return st, p


class _Reader:
    """Host reads of the post-step state; holds the previous full frame for the speed column."""

    def __init__(self, cfg: Any, names: list[str] | None = None) -> None:
        self.cfg = cfg
        self.d = dims(cfg)
        self.names = names
        self.prev: tuple[int, np.ndarray, np.ndarray] | None = None  # step, sorted uid, pos
        self.size = float(cfg.world.size)
        self.height = float(cfg.world.height) if self.d.nd == 3 else 0.0

    def _bonds(self, s: Any) -> tuple[np.ndarray, np.ndarray]:
        """``(n_bonds (N,), body label (N,))`` host arrays of all rows."""
        partner = to_numpy(s.partner)
        alive = to_numpy(s.alive).astype(bool)
        valid = (partner >= 0) & alive[:, None] & alive[np.maximum(partner, 0)]
        nb = valid.sum(axis=1).astype(np.int32)
        lab, _ = metrics.components(s.partner, s.alive, self.cfg.run.components_iters)
        lab = to_numpy(lab).astype(np.int32)
        return nb, np.where(nb > 0, lab, -1).astype(np.int32)

    def full(self, s: Any, t: int) -> dict[str, Any]:
        alive = to_numpy(s.alive).astype(bool)
        idx = np.flatnonzero(alive)
        n = len(idx)
        pos = to_numpy(s.pos)[idx].astype(np.float64)
        uid = to_numpy(s.uid)[idx].astype(np.int64)
        nb_all, body_all = self._bonds(s)
        fields = {"tag": to_numpy(s.tag), "offspring_frac": to_numpy(s.offspring_frac)}
        fields.update(energy=to_numpy(s.energy), birth_step=to_numpy(s.birth_step))
        buoy = getattr(s, "buoy_cap", None)
        if buoy is not None:
            fields["buoy_cap"] = to_numpy(buoy)
        ctx: dict[str, Any] = {k: v[idx] for k, v in fields.items()}
        ctx.update(step=t, n_bonds=nb_all[idx], speed=self._speed(t, uid, pos))
        drift = None
        if self.cfg.weather.enabled:
            dr = structures.drift_at(s.wx, s.pos, buoy, s.alive, getattr(s, "mat", None), self.cfg)
            drift = to_numpy(dr)[idx].astype(np.float16)
        if self.names is None:
            self.names, tr = traits_mod.columns(ctx, n)
        else:
            tr = traits_mod.values_for(ctx, n, self.names)
        order = np.argsort(uid, kind="stable")
        self.prev = (t, uid[order], pos[order])
        return {
            "uid": uid,
            "row": idx.astype(np.int32),
            "pos": quantise(pos, self.size, self.height),
            "clade": to_numpy(s.clade)[idx].astype(np.int32),
            "body": body_all[idx],
            "drift": drift,
            "traits": tr,
        }

    def _speed(self, t: int, uid: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Displacement since the previous full frame, per step (torus-aware in x, y; NaN if new)."""
        out = np.full(len(uid), np.nan, np.float32)
        if self.prev is None or t <= self.prev[0]:
            return out
        pt, puid, ppos = self.prev
        j = np.clip(np.searchsorted(puid, uid), 0, max(len(puid) - 1, 0))
        ok = (puid[j] == uid) if len(puid) else np.zeros(len(uid), bool)
        d = pos[ok] - ppos[j[ok]]
        d[:, :2] -= self.size * np.rint(d[:, :2] / self.size)
        out[ok] = (np.sqrt((d * d).sum(axis=1)) / (t - pt)).astype(np.float32)
        return out

    def bodies(self, s: Any) -> tuple[np.ndarray, np.ndarray]:
        """``(uid, body label)`` of the alive rows now (labels are valid within this frame only)."""
        alive = to_numpy(s.alive).astype(bool)
        idx = np.flatnonzero(alive)
        _, body_all = self._bonds(s)
        return to_numpy(s.uid)[idx].astype(np.int64), body_all[idx]

    def fine(self, s: Any, members: np.ndarray, group: np.ndarray) -> tuple[np.ndarray, ...]:
        """``(uid, group, pos q, body)`` of the fine members alive now (``members`` sorted)."""
        alive = to_numpy(s.alive).astype(bool)
        idx = np.flatnonzero(alive)
        uid = to_numpy(s.uid)[idx].astype(np.int64)
        j = np.clip(np.searchsorted(members, uid), 0, len(members) - 1)
        hit = members[j] == uid
        rows = idx[hit]
        _, body_all = self._bonds(s)
        pos = to_numpy(s.pos)[rows].astype(np.float64)
        return uid[hit], group[j[hit]], quantise(pos, self.size, self.height), body_all[rows]


def _pick_fine(
    uid: np.ndarray, body: np.ndarray, groups: int, per_group: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(sorted uids, group index)`` of the members of the ``groups`` largest bodies of one frame (body"
    "labels
    are per frame; only the uid sets are carried forward)."""
    labs, counts = np.unique(body[body >= 0], return_counts=True)
    order = np.lexsort((labs, -counts))[:groups]
    uids, grp = [], []
    for g, k in enumerate(order):
        m = np.sort(uid[body == labs[k]])[:per_group]
        uids.append(m)
        grp.append(np.full(len(m), g, np.int32))
    if not uids:
        return np.zeros(0, np.int64), np.zeros(0, np.int32)
    u, g = np.concatenate(uids), np.concatenate(grp)
    o = np.argsort(u, kind="stable")
    return u[o], g[o]


def _params(
    rd: RunDir, t0: int, t1: int, tau: int, fine_tau: int, fine_groups: int, fine_max: int, restart: int
) -> dict[str, Any]:
    meta = rd.meta()
    sha = hashlib.sha256()
    for r in rd.metrics_rows():  # the run's own rows in the window: a different run or edit changes the key
        if t0 <= int(r["step"]) <= t1:
            sha.update(json.dumps(r, sort_keys=True).encode())
    return {
        "config_hash": meta.get("config_hash"),
        "run": rd.path.name,
        "seed": meta.get("seed"),
        "created_utc": meta.get("created_utc"),
        "rows_sha": sha.hexdigest(),
        "fine_restart": restart,
        "t0": t0,
        "t1": t1,
        "tau": tau,
        "fine_tau": fine_tau,
        "fine_groups": fine_groups,
        "fine_max": fine_max,
    }


def record_window(
    run: Path,
    t0: int,
    t1: int,
    out_dir: Path,
    *,
    tau: int = 100,
    fine_tau: int = 5,
    fine_groups: int = 8,
    fine_max: int = 250,
    fine_restart: int | None = None,
    verify: bool = True,
    max_lead: int = MAX_LEAD,
    log: Callable[[str], None] = print,
) -> Path:
    """Record steps ``t0..t1`` of ``run`` into ``out_dir/frames_<t0>_<t1>.npz`` and return the path.

    Frames are at ``t0, t0+tau, ...`` (up to ``t1``). The fine set picks the ``fine_groups`` largest bodies
    (at
    most ``fine_max`` members each, as uid sets) and **restarts every ``fine_restart`` steps** (default: a
    block, ``(t1 - t0) / 8`` rounded to a multiple of ``fine_tau``), because organisms turn over with a
    half-life of about 1 000 steps; it is read every ``fine_tau`` steps, per epoch (``fine_epoch``), and
    skipped
    when ``fine_groups`` is 0. ``T1`` must be a save of the run (the end is certified by the state digest;
    otherwise only aggregate metrics rows would certify it) and not beyond the run's last step.
    ``verify=False`` skips the certification (recorded as unverified) but not the range check.
    """
    if not 0 <= t0 < t1:
        raise RecordError(f"the window must satisfy 0 <= T0 < T1, got {t0}:{t1}")
    if tau < 1 or (fine_groups and (fine_tau < 1 or tau % fine_tau)):
        raise RecordError("tau must be positive and a multiple of fine_tau")
    rd = RunDir.open(Path(run))
    cfg = rd.config()
    out = frames_path(out_dir, t0, t1)
    restart = fine_restart or max(fine_tau, ((t1 - t0) // 8 // fine_tau) * fine_tau)
    if fine_groups and restart % fine_tau:
        raise RecordError("fine_restart must be a multiple of fine_tau")
    t, ckpt = start_checkpoint(rd, t0, max_lead)  # E22 first: its message names the saves found
    saves = dict(rd.checkpoints())
    last = max([*saves, *(int(r["step"]) for r in rd.metrics_rows())], default=0)
    if t1 > last:
        raise RecordError(f"{rd.path}: the window ends at step {t1}, beyond the run's last step {last}")
    if verify and t1 not in saves:
        raise RecordError(
            f"{rd.path}: T1 = {t1} is not a save (saves at {sorted(saves)}); the end of a certified "
            f"window must "
            f"be a save so that its state digest can be compared. Choose a save step or pass verify=False"
        )
    ran = rd.meta().get("device")
    if verify and ran and ran != str(mx.default_device()):
        raise RecordError(
            f"{rd.path}: the run was made on {ran} but this process uses {mx.default_device()}; the replay "
            f"would not reproduce it (§14.2). Use the run's device or pass verify=False"
        )
    want = _params(rd, t0, t1, tau, fine_tau, fine_groups, fine_max, restart)
    if out.is_file():
        try:
            old = read_frames(out).meta
            if old.get("params") == want and (old.get("verification", {}).get("verified") or not verify):
                log(f"record: {out} exists, not rewritten")
                return out
        except FramesError:
            pass
    state = runio.load_checkpoint(ckpt, cfg)
    rows = {int(r["step"]): r for r in rd.metrics_rows()} if verify else {}
    kept = {st: p for st, p in rd.checkpoints() if t < st <= t1} if verify else {}
    log(f"record: {rd.path} {t0}:{t1}, replay from save {t}, tau {tau}, verify={verify}")

    spatial.prepare_knn(cfg)
    step_fn = make_step(cfg)
    reader = _Reader(cfg)
    full_steps = set(range(t0, t1 + 1, tau))
    fine_steps = set(range(t0, t1 + 1, fine_tau)) if fine_groups else set()
    frames: list[dict[str, Any]] = []
    fine_rows: list[tuple[int, tuple[np.ndarray, ...]]] = []
    members = group = np.zeros(0, np.int64)
    epoch_of: list[int] = []
    cur = -1
    counters: list[mx.array] = []
    ledgers: list[mx.array] = []
    checked_rows, fingerprints = 0, []
    wall = time.perf_counter()

    def take(s: Any, step: int) -> None:
        nonlocal members, group, cur
        if step in full_steps:
            frames.append(reader.full(s, step) | {"step": step})
        if fine_groups and step in fine_steps:
            ep = min((step - t0) // restart, -(-(t1 - t0) // restart) - 1)
            if ep != cur:  # a new epoch: re-pick the groups from the bodies of this very frame
                cur = ep
                members, group = _pick_fine(*reader.bodies(s), fine_groups, fine_max)
            if len(members):
                fine_rows.append((step, reader.fine(s, members, group)))
                epoch_of.append(ep)

    if t >= t0:
        take(state, t)
    while t < t1:
        state, o = step_fn(state)
        mx.eval(state, o.counters, o.ledger)
        counters.append(o.counters)
        ledgers.append(o.ledger)
        t += 1
        stored = rows.get(t)
        if stored is not None:
            if stored["steps"] == len(counters):
                bad = resim.compare_row(
                    stored,
                    counters,
                    ledgers,
                    int(to_numpy(state.total)),
                    nd=dims(cfg).nd,
                    weather=cfg.weather.enabled,
                )
                if bad:
                    raise RecordMismatch(
                        f"step {t}: the replay differs from metrics.jsonl in {bad}; no file written"
                    )
                checked_rows += 1
            counters, ledgers = [], []
        if t in kept:
            ref = runio.load_checkpoint(kept[t], cfg)
            if state_digest(ref) != state_digest(state):
                raise RecordMismatch(
                    f"step {t}: the replayed state differs from the run's checkpoint; no file written"
                )
            fingerprints.append(t)
        if t >= t0:
            take(state, t)
    if verify and (t1 not in fingerprints):  # unreachable while T1 is a save; kept as the invariant
        raise RecordError(f"{rd.path}: no end fingerprint at {t1}")
    end_digest = state_digest(state)

    meta = _meta(
        rd,
        cfg,
        reader,
        want,
        {
            "verified": bool(verify),
            "rows_checked": checked_rows,
            "fingerprints": fingerprints,
            "end_digest": end_digest,
            "end_is_save": t1 in fingerprints,
        },
        ckpt,
        wall,
    )
    fr = _assemble(frames, fine_rows, reader, meta, epoch_of)
    write_frames(out, fr)
    log(f"record: wrote {out} ({fr.n_frames} frames, {len(fr.uid)} rows, {time.perf_counter() - wall:.0f} s)")
    return out


def _meta(
    rd: RunDir,
    cfg: Any,
    reader: _Reader,
    params: dict[str, Any],
    ver: dict[str, Any],
    ckpt: Path,
    wall: float,
) -> dict[str, Any]:
    return {
        "frames_version": FRAMES_VERSION,
        "run": rd.path.name,
        "params": params,
        "window": [params["t0"], params["t1"]],
        "nd": reader.d.nd,
        "size": reader.size,
        "height": reader.height,
        "n_cap": reader.d.n,
        "start_save": ckpt.name,
        "weather": bool(cfg.weather.enabled),
        "alife_version": rd.meta().get("alife_version"),
        "mlx_version": mx.__version__,
        "device": str(mx.default_device()),
        "source": "replay",
        "verification": ver,
        "wall_s": round(time.perf_counter() - wall, 1),
    }


def _assemble(
    frames: list[dict[str, Any]],
    fine_rows: list[Any],
    reader: _Reader,
    meta: dict[str, Any],
    epochs: list[int] | None = None,
) -> Frames:
    nd = meta["nd"]
    cat = np.concatenate
    sizes = [len(f["uid"]) for f in frames]
    fine = None
    if fine_rows:
        fs = [r[1] for r in fine_rows]
        fine = {
            "fine_step": np.asarray([r[0] for r in fine_rows], np.int64),
            "fine_off": np.concatenate([[0], np.cumsum([len(x[0]) for x in fs])]).astype(np.int64),
            "fine_uid": cat([x[0] for x in fs]).astype(np.int64),
            "fine_epoch": np.asarray(epochs or [0] * len(fine_rows), np.int32),
            "fine_group": cat([x[1] for x in fs]).astype(np.int32),
            "fine_pos": cat([x[2] for x in fs]).astype(np.uint16),
            "fine_body": cat([x[3] for x in fs]).astype(np.int32),
        }
    names = reader.names or []
    return Frames(
        meta=meta,
        step=np.asarray([f["step"] for f in frames], np.int64),
        off=np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64),
        uid=cat([f["uid"] for f in frames]) if frames else np.zeros(0, np.int64),
        row=cat([f["row"] for f in frames]) if frames else np.zeros(0, np.int32),
        pos=cat([f["pos"] for f in frames]) if frames else np.zeros((0, nd), np.uint16),
        clade=cat([f["clade"] for f in frames]) if frames else np.zeros(0, np.int32),
        body=cat([f["body"] for f in frames]) if frames else np.zeros(0, np.int32),
        drift=cat([f["drift"] for f in frames]) if frames and frames[0]["drift"] is not None else None,
        trait_names=names,
        traits=cat([f["traits"] for f in frames]) if frames else None,
        fine=fine,
    )
