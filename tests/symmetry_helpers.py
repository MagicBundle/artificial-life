"""Symmetry goldens: the transformations and the comparison (DESIGN §48; test code only, no engine change).

A symmetry golden applies a transformation ``g`` that the engine claims is irrelevant to a state, runs the
step, transforms back (or canonicalises) and compares with the untransformed run. ``g = e`` is the identity
matrix; these are the other members of the table of ``docs/brainstorming-Fable-November.md`` §1.

- ``permute_rows``: a permutation of the storage rows of every per-agent array, ``partner`` and ``pend_slot``
  re-indexed. ``canonical`` reads a state in uid order, so that two states that differ only by a
  relabelling of rows compare equal.
- ``covariant_rng``: the engine's per-row random draws (the sensing noise, the bond priority, the birth
  priority) are indexed by *storage row*, so a permutation of rows moves them from agent to agent. Inside this
  context manager those three draws are replaced by hashes of ``(key, uid)``, which move with the agent. It
  is a test double for the one thing the permutation row cannot hold fixed otherwise; ``permute_rows`` without
  it is the "broken by design" verdict.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
from collections.abc import Iterator
from typing import Any

import mlx.core as mx
import numpy as np

from alife import bonds, layout, lifecycle, ops, sensing, sphere
from alife.config import Config, dims
from alife.state import CounterIdx, agent_fields, init_state, to_numpy
from alife.step import step
from sphere_rotation import rotate_state

_U32 = mx.uint32
_SALT = 0x5EED0001


def permute_rows(s: Any, cfg: Config, perm: np.ndarray) -> Any:
    """``s`` with new row ``i`` holding old row ``perm[i]`` in every per-agent array; ``partner`` and
    ``pend_slot`` (row indices) are re-indexed through the inverse permutation."""
    n = len(perm)
    inv = np.empty(n, np.int64)
    inv[perm] = np.arange(n)
    idx = mx.array(perm.astype(np.int32))
    inv_m = mx.array(inv.astype(np.int32))
    out = {name: getattr(s, name)[idx] for name in agent_fields(cfg)}
    out["partner"] = mx.where(out["partner"] >= 0, inv_m[mx.maximum(out["partner"], 0)], -1).astype(
        s.partner.dtype
    )
    out["pend_slot"] = inv_m[s.pend_slot].astype(s.pend_slot.dtype)
    ns = s._replace(**out)
    mx.eval(ns)
    return ns


def random_perm(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n)


def canonical(s: Any, cfg: Config) -> dict[str, np.ndarray]:
    """The state in uid order: alive rows only, ``partner`` as the partner's uid, the pending child weights
    flushed into the rows, every global field as it is. Dead rows are checked for canonical contents by
    ``dead_rows_clean``, not compared."""
    W1, W2 = lifecycle.flush_pending(s.W1, s.W2, s.pend_slot, s.pend_W1, s.pend_W2)
    alive = to_numpy(s.alive)
    uid = to_numpy(s.uid)
    rows = np.flatnonzero(alive)
    rows = rows[np.argsort(uid[rows], kind="stable")]
    partner = to_numpy(s.partner)
    uid_of_partner = np.where(partner >= 0, uid[np.maximum(partner, 0)], -1)
    names = agent_fields(cfg)
    out: dict[str, np.ndarray] = {}
    for name in names:
        if name == "partner":
            out[name] = uid_of_partner[rows]
        elif name == "W1":
            out[name] = to_numpy(W1)[rows]
        elif name == "W2":
            out[name] = to_numpy(W2)[rows]
        else:
            out[name] = to_numpy(getattr(s, name))[rows]
    for name in type(s)._fields:
        if name not in names and name not in ("pend_slot", "pend_W1", "pend_W2"):
            out[name] = to_numpy(getattr(s, name))
    return out


def differences(
    a: dict[str, np.ndarray],
    b: dict[str, np.ndarray],
    *,
    atol: float = 0.0,
    torus: float | None = None,
    skip: tuple[str, ...] = (),
) -> list[str]:
    """The entries of two canonical states that differ, as ``name: max|d|`` strings (empty = equal).

    ``atol = 0`` is bitwise equality (NaN equals NaN). With ``atol > 0`` integer entries are still compared
    exactly and float entries to ``atol`` absolute; angles (``heading``) are compared circularly, and with
    ``torus = L`` the x and y of ``pos`` and ``patch_pos`` are compared modulo ``L``."""
    bad: list[str] = []
    for name in a:
        if name in skip:
            continue
        x, y = a[name], b[name]
        if x.shape != y.shape:
            bad.append(f"{name}: shape {x.shape} != {y.shape}")
            continue
        if x.dtype.kind != "f":
            if not np.array_equal(x, y):
                bad.append(f"{name}: differs (max|d| = {np.abs(x.astype(np.int64) - y).max():.3g})")
            continue
        if atol == 0.0:
            if not np.array_equal(x, y, equal_nan=True):
                bad.append(f"{name}: differs (max|d| = {np.nanmax(np.abs(x.astype(np.float64) - y)):.3g})")
            continue
        d = x.astype(np.float64) - y.astype(np.float64)
        if name == "heading":
            d -= 2.0 * np.pi * np.round(d / (2.0 * np.pi))
        elif torus is not None and name in ("pos", "patch_pos"):
            d[..., :2] -= torus * np.round(d[..., :2] / torus)
        if not np.all(np.isnan(x) == np.isnan(y)) or np.nanmax(np.abs(d), initial=0.0) > atol:
            bad.append(f"{name}: max|d| = {np.nanmax(np.abs(d), initial=0.0):.3g}")
    return bad


def dead_rows_clean(s: Any, cfg: Config) -> bool:
    """Dead rows hold no live energy, bonds or ages: the dead-canonical contract (``Diag.DEAD_DIRTY``)."""
    dead = ~to_numpy(s.alive)
    return bool(
        (to_numpy(s.energy)[dead] == 0).all()
        and (to_numpy(s.partner)[dead] == -1).all()
        and (to_numpy(s.uid)[dead] == -1).all()
    )


def float_deviation(
    a: dict[str, np.ndarray], b: dict[str, np.ndarray], *, torus: float | None = None
) -> float:
    """The largest absolute difference over the float entries of two canonical states (angles circular, the
    torus coordinates modulo ``torus``); ``inf`` when the float entries differ in NaN pattern."""
    worst = 0.0
    for name, x in a.items():
        if x.dtype.kind != "f":
            continue
        y = b[name]
        d = x.astype(np.float64) - y.astype(np.float64)
        if name == "heading":
            d -= 2.0 * np.pi * np.round(d / (2.0 * np.pi))
        elif torus is not None and name in ("pos", "patch_pos"):
            d[..., :2] -= torus * np.round(d[..., :2] / torus)
        if not np.array_equal(np.isnan(x), np.isnan(y)):
            return math.inf
        worst = max(worst, float(np.nanmax(np.abs(d), initial=0.0)))
    return worst


@dataclasses.dataclass
class Trace:
    """One lockstep run of a world ``a`` and its transform ``b``: per step, the integer entries that differ
    (``ints``), the largest float deviation (``fdev``), and whether the population and the step's counters and
    ledger agree (``flows``)."""

    ints: list[list[str]] = dataclasses.field(default_factory=list)
    fdev: list[float] = dataclasses.field(default_factory=list)
    flows: list[bool] = dataclasses.field(default_factory=list)
    counters: np.ndarray | None = None  # (steps, n_counters) of the baseline run

    def exact_through(self, k: int, atol: float) -> bool:
        """Integer state, counters and ledger identical, and floats within ``atol``, at steps ``1..k``."""
        return all(
            not i and f and d <= atol
            for i, f, d in zip(self.ints[:k], self.flows[:k], self.fdev[:k], strict=True)
        )

    def first_flip(self) -> int | None:
        """The first step (1-based) where an integer entry, the population, a counter or the ledger differ."""
        return next(
            (t + 1 for t, (i, f) in enumerate(zip(self.ints, self.flows, strict=True)) if i or not f), None
        )

    def births(self) -> int:
        assert self.counters is not None
        return int(self.counters[:, CounterIdx.BIRTHS].sum())


def lockstep(
    cfg: Config,
    steps: int,
    *,
    forward: Any,
    view: Any = None,
    torus: float | None = None,
    skip_counters: tuple[CounterIdx, ...] = (),
    covariant: bool = False,
    covariant_kw: dict[str, bool] | None = None,
) -> Trace:
    """Run ``init_state(cfg)`` and ``forward(init_state(cfg))`` for ``steps`` eager steps and compare after
    each: the baseline seen through ``view`` (default: as it is) against the transformed world, canonically
    (uid order), the per-step counters (except ``skip_counters``) and the ledger. For an equivariance
    ``step . g = g . step`` use ``forward = view = g``; for a relabelling that ``canonical`` already undoes
    (a row permutation) ``view`` is the identity. ``covariant`` switches the identity-keyed draws on."""
    ctx = covariant_rng(**(covariant_kw or {})) if covariant else contextlib.nullcontext()
    tr = Trace()
    keep = np.array([i for i in range(len(CounterIdx)) if i not in skip_counters])
    s0 = init_state(cfg)
    a, b = s0, forward(s0)
    ctr = []
    with ctx:
        for _ in range(steps):
            if covariant:
                set_ident(a)
            a, oa = step(a, cfg)
            if covariant:
                set_ident(b)
            b, ob = step(b, cfg)
            mx.eval(a, b, oa, ob)
            ca, cb = canonical(a if view is None else view(a), cfg), canonical(b, cfg)
            same_n = len(ca["uid"]) == len(cb["uid"])
            if not same_n:
                tr.ints.append(["population differs"])
                tr.fdev.append(math.inf)
                tr.flows.append(False)
                ctr.append(to_numpy(oa.counters))
                break
            tr.ints.append(differences({k: v for k, v in ca.items() if v.dtype.kind != "f"}, cb))
            tr.fdev.append(float_deviation(ca, cb, torus=torus))
            xa, xb = to_numpy(oa.counters), to_numpy(ob.counters)
            ok = np.array_equal(xa[keep], xb[keep]) and np.array_equal(
                to_numpy(oa.ledger), to_numpy(ob.ledger)
            )
            tr.flows.append(ok)
            ctr.append(xa)
    tr.counters = np.array(ctr)
    return tr


# --- the row-covariant draws -----------------------------------------------------------------------------


class _Ident:
    """The per-step identity of every row (set by the driver before each step)."""

    value: mx.array | None = None


IDENT = _Ident()


def set_ident(s: Any) -> None:
    """Identity of a row for the covariant draws: the uid of an alive row, a distinct negative number for a
    dead one (a dead row's draw is never used)."""
    n = s.alive.shape[0]
    IDENT.value = mx.where(s.alive, s.uid, -1 - mx.arange(n, dtype=mx.int64))


def _u24(key: mx.array, ident: mx.array) -> mx.array:
    """Per-identity uniform hash in [0, 2^24) from the key's two words and the identity."""
    h = ops.hash32(ident, key[0], _SALT)
    h = ops.hash32(h, key[1], _SALT ^ 0x1234567)
    return mx.right_shift(h, mx.array(8, dtype=_U32)).astype(mx.int32)


def _cov_uniform(orig: Any) -> Any:
    def uniform(low: Any = 0.0, high: Any = 1.0, shape: Any = None, dtype: Any = mx.float32, key: Any = None):
        if key is not None and IDENT.value is not None and tuple(shape or ()) == tuple(IDENT.value.shape):
            u = _u24(key, IDENT.value).astype(mx.float32) / float(1 << 24)
            return (low + (high - low) * u).astype(dtype)
        return orig(low, high, shape, dtype, key=key)

    return uniform


def _cov_prio(n: int, key: mx.array) -> mx.array:
    """Bond priorities: the rank of each row's identity hash (unique, in ``[0, n)``)."""
    u = _u24(key, IDENT.value)
    return mx.argsort(mx.argsort(u))


def _cov_order(n: int, key: mx.array) -> mx.array:
    """Birth order: the rows in increasing order of their identity hash."""
    return mx.argsort(_u24(key, IDENT.value))


class _RandomShim:
    def __init__(self, base: Any, **over: Any) -> None:
        self._base, self._over = base, over

    def __getattr__(self, name: str) -> Any:
        return self._over.get(name, getattr(self._base, name))


class _MxShim:
    """``mx`` with a ``random`` namespace whose named functions are replaced."""

    def __init__(self, **over: Any) -> None:
        self.random = _RandomShim(mx.random, **over)

    def __getattr__(self, name: str) -> Any:
        return getattr(mx, name)


@contextlib.contextmanager
def covariant_rng(*, noise: bool = True, bond: bool = True, birth: bool = True) -> Iterator[None]:
    """Replace the three per-row draws of the step by identity-keyed ones (see the module docstring). The
    step must run eagerly and ``set_ident`` must be called with the start-of-step state before each step."""
    saved = (sensing.mx, bonds.mx, lifecycle.mx)
    if noise:
        sensing.mx = _MxShim(uniform=_cov_uniform(mx.random.uniform))  # type: ignore[assignment]
    if bond:
        bonds.mx = _MxShim(permutation=_cov_prio)  # type: ignore[assignment]
    if birth:
        lifecycle.mx = _MxShim(permutation=_cov_order)  # type: ignore[assignment]
    try:
        yield
    finally:
        sensing.mx, bonds.mx, lifecycle.mx = saved


# --- torus translation and mirror image of the box world ---------------------------------------------------


def translate(s: Any, cfg: Config, shift: tuple[int, int]) -> Any:
    """The box world translated by ``shift = (kx, ky)`` field cells: every position and patch centre moved by
    ``k·field_cell`` (wrapped), the resource and chemical arrays rolled by ``k`` cells in x and y. In 3D the
    height is not moved (the floor and the surface are not on the torus)."""
    fc, L = cfg.world.field_cell, cfg.world.size
    kx, ky = shift
    off = np.zeros(s.pos.shape[-1], np.float32)
    off[0], off[1] = kx * fc, ky * fc
    ofs = mx.array(off)
    pos = ops.wrap(s.pos + ofs[None, :], L) if s.pos.shape[-1] == 2 else _wrap_xy(s.pos + ofs[None, :], L)
    pos = mx.where(s.alive[:, None], pos, s.pos)
    pp = ops.wrap(s.patch_pos + mx.array(off[:2])[None, :], L)
    res = mx.roll(mx.roll(s.resource, ky, axis=-2), kx, axis=-1)
    chem = mx.roll(mx.roll(s.chem, ky, axis=-2), kx, axis=-1)
    ns = s._replace(pos=pos, patch_pos=pp, resource=res, chem=chem)
    mx.eval(ns)
    return ns


def _wrap_xy(p: mx.array, L: float) -> mx.array:
    return mx.concatenate([ops.wrap(p[..., :2], L), p[..., 2:]], axis=-1)


def mirror(s: Any, cfg: Config) -> Any:
    """The box world mirrored in the plane ``x = L/2`` of the torus (``x -> L - x``, wrapped): positions,
    velocities, headings (``h -> pi - h``), the patch centres and drifts, and the field arrays (cell columns
    reversed). The weights are *not* touched: pair this with ``mirror_weights`` (the brain reads handedness
    through its lateral inputs and its turn and query outputs)."""
    L = cfg.world.size
    flip = mx.array([-1.0] + [1.0] * (s.pos.shape[-1] - 1), dtype=mx.float32)
    x = ops.wrap(L - s.pos[:, 0], L)
    pos = mx.concatenate([x[:, None], s.pos[:, 1:]], axis=-1)
    ns = s._replace(
        pos=mx.where(s.alive[:, None], pos, s.pos),
        vel=s.vel * flip,
        heading=mx.where(s.alive, ops.wrap_angle(math.pi - s.heading), s.heading),
        patch_pos=mx.concatenate([ops.wrap(L - s.patch_pos[:, :1], L), s.patch_pos[:, 1:]], axis=-1),
        patch_vel=s.patch_vel * mx.array([-1.0, 1.0]),
        resource=mx.flip(s.resource, axis=-1),
        chem=mx.flip(s.chem, axis=-1),
    )
    mx.eval(ns)
    return ns


def lateral_inputs(cfg: Config) -> list[int]:
    """The brain inputs that change sign under a mirror: every lateral component (body frame fwd, lat, up)."""
    dims_in = dims(cfg).lay.D_IN
    idx = [layout.IN_VEL + 1, layout.IN_RES_GRAD + 1]
    idx += [layout.IN_CHEM_GRAD + 2 * c + 1 for c in range(layout.NCH)]
    idx += [layout.IN_NBR + layout.NF * n + layout.NBR_DY for n in range(layout.NS)]
    idx += [layout.IN_BOND + layout.BOND_W * b + layout.BOND_DY for b in range(layout.B)]
    assert dims_in > max(idx)
    return idx


def lateral_outputs(cfg: Config) -> list[int]:
    """The brain outputs that change sign under a mirror: the turn and the lateral weight of both queries."""
    return [layout.OUT_TURN, layout.OUT_Q_BOND + layout.NBR_DY - 1, layout.OUT_Q_ATTACK + layout.NBR_DY - 1]


def mirror_weights(W1: mx.array, W2: mx.array, cfg: Config) -> tuple[mx.array, mx.array]:
    """The weights of the mirrored brain: the W1 rows of the lateral inputs and the W2 columns of the lateral
    outputs negated, so that ``y`` of the mirrored inputs is the mirror of ``y`` (negation is exact in
    floating point)."""
    d = dims(cfg)
    s1 = np.ones(d.w1_rows, np.float32)
    s1[lateral_inputs(cfg)] = -1.0
    s2 = np.ones(d.lay.D_OUT, np.float32)
    s2[lateral_outputs(cfg)] = -1.0
    return W1 * mx.array(s1)[None, :, None].astype(W1.dtype), W2 * mx.array(s2)[None, None, :].astype(
        W2.dtype
    )


def apply_weights(s: Any, fn: Any) -> Any:
    """``s`` with ``fn(W1, W2)`` applied to the live and the pending weights."""
    W1, W2 = fn(s.W1, s.W2)
    P1, P2 = fn(s.pend_W1, s.pend_W2)
    ns = s._replace(W1=W1, W2=W2, pend_W1=P1, pend_W2=P2)
    mx.eval(ns)
    return ns


# --- the gauge group of the Elman RNN -------------------------------------------------------------------


def gauge_weights(W1: mx.array, W2: mx.array, perm: np.ndarray, sign: np.ndarray, cfg: Config):
    """The hidden-unit relabelling ``h'_j = sign_j h_perm[j]`` of the RNN ``h' = tanh([x, h] W1)``,
    ``y = [h', 1] W2``: ``W1'[i, c] = sign_c W1[i, perm_c]`` for the input rows, ``W1'[D_IN + j, c] =
    sign_j sign_c W1[D_IN + perm_j, perm_c]`` for the recurrent rows, ``W2'[j] = sign_j W2[perm_j]`` (the bias
    row stays). ``tanh`` is odd, so the new network computes the same ``y`` and the relabelled ``h``."""
    d = dims(cfg)
    din, h = d.lay.D_IN, d.h
    rows = np.concatenate([np.arange(din), din + perm])
    sr = np.concatenate([np.ones(din), sign]).astype(np.float32)
    w1 = W1[:, mx.array(rows)][:, :, mx.array(perm)]
    w1 = w1 * mx.array(sr)[None, :, None] * mx.array(sign.astype(np.float32))[None, None, :]
    w2 = W2[:, mx.array(np.concatenate([perm, [h]]))]
    w2 = w2 * mx.array(np.concatenate([sign, [1.0]]).astype(np.float32))[None, :, None]
    return w1.astype(W1.dtype), w2.astype(W2.dtype)


def gauge_state(s: Any, cfg: Config, perm: np.ndarray, sign: np.ndarray) -> Any:
    """``s`` in the relabelled gauge: weights as ``gauge_weights``, hidden state ``sign * h[:, perm]``."""
    ns = apply_weights(s, lambda a, b: gauge_weights(a, b, perm, sign, cfg))
    ns = ns._replace(h=s.h[:, mx.array(perm)] * mx.array(sign.astype(np.float32)))
    mx.eval(ns)
    return ns


def ungauge_hidden(h: np.ndarray, perm: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """The hidden state of the original gauge from the relabelled one: ``h[:, perm_j] = sign_j h'_j``."""
    out = np.empty_like(h)
    out[:, perm] = h * sign.astype(h.dtype)
    return out


# --- the kin cue: rotation of the tag vector ----------------------------------------------------------------


def random_orthogonal(n: int, seed: int) -> np.ndarray:
    """A random orthogonal ``n x n`` matrix (float64; the Q of a Gaussian matrix, either handedness)."""
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n)))
    return q * np.sign(np.diag(r))


def rotate_tags(s: Any, q: np.ndarray) -> Any:
    """``s`` with every tag replaced by ``Q tag`` (float64 product, stored as float32)."""
    tag = to_numpy(s.tag).astype(np.float64) @ q.T
    ns = s._replace(tag=mx.array(tag.astype(np.float32)))
    mx.eval(ns)
    return ns


# --- the HEALPix lattice group on the sphere world ----------------------------------------------------------

# Exact signed permutation matrices that map the HEALPix pixel centres onto themselves: a quarter turn about
# the polar axis, and half turns about the x and y axes (which swap the hemispheres).
ROT_C4Z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float64)
ROT_C2X = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float64)
ROT_C2Y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], np.float64)


def pixel_map(cfg: Config, rotation: np.ndarray) -> np.ndarray:
    """``T[j]``: the pixel that the centre of pixel ``j`` lands in after ``rotation`` (float32 lookup of
    ``sphere.ang2pix`` on the rotated centres). A lattice symmetry gives a permutation of ``range(Npix)``."""
    d = dims(cfg)
    centres = np.asarray(sphere.pix2vec(mx.arange(d.npix, dtype=mx.int32), d.nside), np.float64)
    moved = mx.array((centres @ rotation.T).astype(np.float32))
    return to_numpy(sphere.ang2pix(moved, d.nside)).astype(np.int64)


def rotate_sphere(s: Any, cfg: Config, rotation: np.ndarray) -> Any:
    """The planet rotated by a lattice symmetry: agents, patches (``sphere_rotation.rotate_state``) and the
    field arrays, whose pixel axis is permuted by ``pixel_map``."""
    t = pixel_map(cfg, rotation)
    inv = np.empty_like(t)
    inv[t] = np.arange(len(t))
    ns = rotate_state(s, rotation)
    ns = ns._replace(resource=s.resource[..., mx.array(inv)], chem=s.chem[..., mx.array(inv)])
    mx.eval(ns)
    return ns
