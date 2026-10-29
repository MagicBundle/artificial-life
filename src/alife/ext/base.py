"""Declaration types of the feature extension mechanism (DESIGN §43).

A leaf module: ``dataclasses`` and ``typing`` only, no ``config``, ``state``, ``layout`` or ``mlx`` import, so
that ``config.dims`` -> ``layout`` -> ``ext`` has no cycle. A declaration duck-types ``cfg`` (it reads
``cfg.<section>.<key>``) and ``dims`` (the derived sizes); the implementation of a feature lives in its own
``alife/<feature>.py``, never here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, NamedTuple

# The canonical order of the features. It orders state fields, counter blocks, ActFlag bits, mask bits, config
# sections and the ``layout_features`` string. ``niche`` is farming (marks would join the same feature).
EXTENSIONS: Final[tuple[str, ...]] = ("arms", "structures", "bonds", "comm", "niche")
# The order of the brain tails: the same list with the arms perception tail last (arms stages 1 to 5 have no
# brain footprint).
TAIL_ORDER: Final[tuple[str, ...]] = ("structures", "bonds", "comm", "niche", "arms")
MAX_EXTENSIONS: Final = 31  # mask bits: genome_layout_version = base + 4·mask stays in int32
ACT_FLAG_BASE: Final = (
    128  # Acts.flags bit of extension i is ACT_FLAG_BASE << i (bits 1..64 are the base flags)
)


@dataclass(frozen=True)
class Block:
    """One named run of brain inputs or outputs; ``lay.IN_<NAME>`` / ``lay.OUT_<NAME>`` derive from
    ``name``."""

    name: str
    width: int
    silent: bool = False  # zero-initialise the W1 rows / W2 columns of founders and immigrants


class ExtCtx(NamedTuple):
    """What the step has computed when the extension hooks run (S2 to S6): the neighbour search, the sensed
    descriptors, the bond geometry and the field cell of each row. Passed to ``inputs`` and ``decode``."""

    nbr: Any  # spatial.Neighbors
    desc: Any  # (N, K, NF) neighbour descriptors
    geom: Any  # bonds.BondGeom
    fcell: Any  # (N,) int32 start-of-step field cell


class BirthCtx(NamedTuple):
    """The lanes of ``lifecycle.reproduce`` for an ``on_birth`` hook: ``parent``/``slot`` per lane (M,), the
    ``birth``/``imm`` lane masks (M,), ``newborn`` (N,) and ``born_lane`` (N,), ``ks`` the step keys."""

    parent: Any
    slot: Any
    birth: Any
    imm: Any
    newborn: Any
    born_lane: Any
    ks: Any


@dataclass(frozen=True)
class FieldSpec:
    """One extension field of the state (conditional: it exists only while its feature is on).

    ``on_birth`` (per-agent fields only) is what ``lifecycle.reproduce`` writes into the row of a newborn, for
    birth lanes and immigrant lanes alike: ``"rest"`` (the field's ``rest`` value: role, dwell and flow
    memories start fresh), ``"inherit"`` (a birth copies its parent's row, an immigrant takes ``rest``: a
    heritable trait that mutates elsewhere), or a hook ``(old, rest, BirthCtx) -> new`` returning the whole
    (N, ...) array (arms draws its mutated traits here). A dying row is left as it is: ``alive`` guards every
    reader and a birth overwrites the whole row.
    """

    name: str
    per_agent: bool  # leading dim N: joins ``agent_fields(cfg)`` (reproduce, inspect)
    shape: Callable[[Any], tuple[int, ...]]  # (dims) -> shape
    dtype: Any  # an mx.Dtype (typed Any: this module imports no mlx)
    rest: Callable[[Any], Any]  # (cfg) -> array: the initial value and the fill of an older checkpoint
    on_birth: str | Callable[[Any, Any, BirthCtx], Any] = "rest"
    # True for a field that holds conserved energy quanta: ``alife migrate --force-removal`` can drop it only
    # while it is at its ``rest`` value (anything else would leak TOTAL), never silently otherwise.
    conserved: bool = False


def _never(cfg: Any) -> bool:
    return False


def _no_blocks(cfg: Any, nd: int) -> tuple[Block, ...]:
    return ()


def _no_names(cfg: Any, nd: int) -> tuple[str, ...]:
    return ()


def _no_fields(cfg: Any, nd: int) -> tuple[FieldSpec, ...]:
    return ()


def _no_dims(cfg: Any) -> Mapping[str, int]:
    return {}


def _no_strs(cfg: Any) -> tuple[str, ...]:
    return ()


def _ok(cfg: Any) -> None:
    return None


@dataclass(frozen=True)
class ExtSpec:
    """The declaration of one feature. Every hook defaults to "nothing", so a declaration states only what the
    feature has. ``block_version`` bumps with any change to the feature's blocks (like
    ``GENOME_LAYOUT_VERSION``)."""

    name: str
    block_version: int = 1
    enabled: Callable[[Any], bool] = _never
    layout_on: Callable[[Any], bool] = _never  # has a brain footprint (arms: the perception tail only)
    token: Callable[[Any], str] = field(default=lambda cfg: "")  # "comm4@1": widths and block version
    in_blocks: Callable[[Any, int], tuple[Block, ...]] = _no_blocks
    out_blocks: Callable[[Any, int], tuple[Block, ...]] = _no_blocks
    block_names: tuple[
        str, ...
    ] = ()  # every block name the feature can declare (Layout.IN_*/OUT_* resolve here)
    nbr_features: Callable[[Any, int], tuple[str, ...]] = _no_names  # descriptor columns (arms: ("sig",))
    dims: Callable[[Any], Mapping[str, int]] = _no_dims  # extra Dims fields: voice_w, hear_range ...
    state_fields: Callable[[Any, int], tuple[FieldSpec, ...]] = _no_fields
    counters: Callable[[Any], tuple[str, ...]] = _no_strs  # the names of the feature's counter block
    act_flag: str | None = None  # the ActFlag member name (its value is fixed by the extension index)
    dump_kinds: tuple[str, ...] = ()
    color_modes: tuple[str, ...] = ()
    stream_tags: tuple[str, ...] = ()
    metric_prefix: str = ""
    lever_keys: tuple[tuple[str, str], ...] = ()  # every (section, key) the feature adds to Config
    # the lever keys that only a host-side observer reads: omitted from ``config_hash`` at every value
    # (``config.OBSERVER_KEYS``) and allowed on a resume
    observer_keys: tuple[tuple[str, str], ...] = ()
    cli_only_keys: tuple[tuple[str, str], ...] = ()  # lever keys the GUI does not carry (``*.control``)
    overlay: str | None = None  # world overlay name (presets/<overlay>.toml)
    # (cfg) -> cfg with the feature switched on (its ``enabled`` key; the start-of-run lever keys stay at the
    # config's value, ``--set`` moves them): what ``alife migrate --enable`` applies (DESIGN §43.8). None =
    # the feature has no switch of its own and is enabled with ``--set``.
    enable: Callable[[Any], Any] | None = None
    validate: Callable[[Any], None] = _ok
    # Step hooks (static: called only for an enabled feature, inside the traced step):
    inputs: Callable[[Any, Any, ExtCtx], Sequence[Any]] | None = None  # (s, cfg, ctx) -> (N, w) per in-block
    # (s, y, ctx, cfg) -> the feature's decoded brain outputs (a pytree of arrays), kept in ``_Flows.ext``
    decode: Callable[[Any, Any, ExtCtx, Any], Any] | None = None
    due: Callable[[Any, Any, Any, Any], Any] | None = None  # (s, act, y, cfg) -> (N,) int32 due, or None
    mask: Callable[[Any, Any, Any], Any] | None = (
        None  # (decoded, solvent, cfg) -> decoded masked by solvency
    )
    count: Callable[[Any, Any, Any, Any], Mapping[str, Any]] | None = None  # (s_new, alive0, flows, cfg)
    # Snapshot metrics (host side, once per metrics row; DESIGN §43.5): ``snap_dev(s, cfg, heavy)`` names the
    # device
    # arrays that join the one transfer of ``metrics.snapshot_metrics``; ``snap_row(host, cfg)`` turns the
    # host arrays (``host`` = the whole transfer, those of every feature included) into the keys of the row,
    # which must start with the feature's ``metric_prefix``.
    snap_dev: Callable[[Any, Any, bool], Mapping[str, Any]] | None = None
    snap_row: Callable[[Mapping[str, Any], Any], Mapping[str, Any]] | None = None
    check: Callable[[Mapping[str, Any], Any], Sequence[str]] | None = None  # host check: (numpy fields, cfg)

    @property
    def index(self) -> int:
        return EXTENSIONS.index(self.name)

    @property
    def tail_rank(self) -> int:
        return TAIL_ORDER.index(self.name)

    @property
    def flag_value(self) -> int:
        return ACT_FLAG_BASE << self.index
