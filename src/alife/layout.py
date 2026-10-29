"""Structural constants and genome I/O layout (§2.1, §5, §7, §27.1).

The module constants are the 2D layout. A 3D world (``world.dims = 3``) appends a tail to the input and output
vectors and keeps every index below unchanged; ``for_dims(nd)`` is the per-dims view and the only source of
the totals ``D_IN``/``D_OUT`` outside this module. Changing any value here changes the genome layout: bump
``GENOME_LAYOUT_VERSION`` (2D) or ``GENOME_LAYOUT_VERSION_3D`` with it.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from dataclasses import dataclass
from typing import Final

from alife import ext

GENOME_LAYOUT_VERSION: Final = 1
GENOME_LAYOUT_VERSION_3D: Final = 2  # one version sequence: a future 2D bump takes 3 (§27.1)

# --- structural symbols (§2.1) ---------------------------------------------------------------------------
B: Final = 4  # bond slots per agent
NS: Final = 3  # nearest neighbours given explicit input blocks
NF: Final = 7  # features per sensed neighbour
F: Final = 6  # targeting descriptor dims (NF minus `present`)
MSG: Final = 2  # broadcast message channels
NCH: Final = 3  # chemical field channels
T: Final = 4  # genome tag dims
D_IN: Final = 69  # brain inputs
D_OUT: Final = 35  # brain outputs

# --- neighbour descriptor features desc[..., f] (§5) ------------------------------------------------------
NBR_PRESENT: Final = 0
NBR_DX: Final = 1
NBR_DY: Final = 2
NBR_KIN: Final = 3
NBR_RELE: Final = 4
NBR_MSG0: Final = 5
NBR_BONDED: Final = 6

# --- per-bond-slot input block features (§5, inputs 49 + 5b + f) --------------------------------------------
BOND_PRESENT: Final = 0
BOND_DX: Final = 1
BOND_DY: Final = 2
BOND_MSG0: Final = 3
BOND_RELE: Final = 4
BOND_W: Final = 5  # width of one bond-slot block

# --- input indices x[:, IN_*] (§5) -----------------------------------------------------------------------
IN_BIAS: Final = 0
IN_ENERGY: Final = 1
IN_AGE: Final = 2
IN_VEL: Final = 3  # 2 wide: (fwd, lat)
IN_NBONDS: Final = 5
IN_HURT: Final = 6
IN_FED: Final = 7
IN_GIFT: Final = 8
IN_NOISE: Final = 9
IN_CLOCK: Final = 10  # 2 wide: (sin, cos)
IN_RES: Final = 12
IN_RES_GRAD: Final = 13  # 2 wide: (fwd, lat)
IN_CHEM: Final = 15  # NCH wide
IN_CHEM_GRAD: Final = 18  # 2·NCH wide: (c0 fwd, c0 lat, c1 fwd, c1 lat, c2 fwd, c2 lat)
IN_CROWD: Final = 24
IN_KIN_MEAN: Final = 25
IN_MSG_MEAN: Final = 26  # MSG wide
IN_NBR: Final = 28  # NS·NF wide: 28 + 7n + f
IN_BOND: Final = 49  # B·BOND_W wide: 49 + 5b + f

# --- output indices y[:, OUT_*] (§7) ---------------------------------------------------------------------
OUT_THRUST: Final = 0
OUT_TURN: Final = 1
OUT_MSG: Final = 2  # MSG wide
OUT_EMIT: Final = 4  # NCH wide
OUT_HARVEST: Final = 7
OUT_GRIP: Final = 8
OUT_CONTRACT: Final = 9
OUT_BOND: Final = 10
OUT_RELEASE: Final = 11  # B wide
OUT_SHARE: Final = 15  # B wide
OUT_ATTACK: Final = 19
OUT_DEFEND: Final = 20
OUT_REPRODUCE: Final = 21
OUT_CHILD_BOND: Final = 22
OUT_Q_BOND: Final = 23  # F wide
OUT_Q_ATTACK: Final = 29  # F wide

# --- fixed input normalisation constants (§5) and numeric guards (§9, §10.10) ------------------------------
LR_SCALE: Final = 2.0
AGE_SCALE: Final = 1000.0
RES_SAT: Final = 2.0
RES_GRAD_GAIN: Final = 4.0
EPS_LEN: Final = 1e-6
C_REF_MARGIN: Final = 50

# --- 3D orientation (§27.2) ------------------------------------------------------------------------------
# Pitch φ is clipped to ±PITCH_MAX, never wrapped. MLX clips in float32: range checks compare against
# float32(PITCH_MAX). A constant, not a config key, so it never enters a config hash.
PITCH_MAX: Final = math.radians(80.0)

# --- block tables (name, start, width); they tile [0, D_IN) and [0, D_OUT) -------------------------------
INPUT_BLOCKS: Final[tuple[tuple[str, int, int], ...]] = (
    ("bias", IN_BIAS, 1),
    ("energy", IN_ENERGY, 1),
    ("age", IN_AGE, 1),
    ("vel", IN_VEL, 2),
    ("nbonds", IN_NBONDS, 1),
    ("hurt", IN_HURT, 1),
    ("fed", IN_FED, 1),
    ("gift", IN_GIFT, 1),
    ("noise", IN_NOISE, 1),
    ("clock", IN_CLOCK, 2),
    ("res", IN_RES, 1),
    ("res_grad", IN_RES_GRAD, 2),
    ("chem", IN_CHEM, NCH),
    ("chem_grad", IN_CHEM_GRAD, 2 * NCH),
    ("crowd", IN_CROWD, 1),
    ("kin_mean", IN_KIN_MEAN, 1),
    ("msg_mean", IN_MSG_MEAN, MSG),
    ("nbr", IN_NBR, NS * NF),
    ("bond", IN_BOND, B * BOND_W),
)

OUTPUT_BLOCKS: Final[tuple[tuple[str, int, int], ...]] = (
    ("thrust", OUT_THRUST, 1),
    ("turn", OUT_TURN, 1),
    ("msg", OUT_MSG, MSG),
    ("emit", OUT_EMIT, NCH),
    ("harvest", OUT_HARVEST, 1),
    ("grip", OUT_GRIP, 1),
    ("contract", OUT_CONTRACT, 1),
    ("bond", OUT_BOND, 1),
    ("release", OUT_RELEASE, B),
    ("share", OUT_SHARE, B),
    ("attack", OUT_ATTACK, 1),
    ("defend", OUT_DEFEND, 1),
    ("reproduce", OUT_REPRODUCE, 1),
    ("child_bond", OUT_CHILD_BOND, 1),
    ("q_bond", OUT_Q_BOND, F),
    ("q_attack", OUT_Q_ATTACK, F),
)

NBR_FEATURE_NAMES: Final[tuple[str, ...]] = ("present", "dx", "dy", "kin", "rele", "msg0", "bonded")
BOND_FEATURE_NAMES: Final[tuple[str, ...]] = ("present", "dx", "dy", "msg0", "rele")

INPUT_NAMES: Final[tuple[str, ...]] = (
    "bias",
    "energy",
    "age",
    "vel_fwd",
    "vel_lat",
    "nbonds",
    "hurt",
    "fed",
    "gift",
    "noise",
    "clock_sin",
    "clock_cos",
    "res",
    "res_grad_fwd",
    "res_grad_lat",
    *(f"chem{c}" for c in range(NCH)),
    *(f"chem{c}_grad_{ax}" for c in range(NCH) for ax in ("fwd", "lat")),
    "crowd",
    "kin_mean",
    *(f"msg_mean{m}" for m in range(MSG)),
    *(f"nbr{n}_{f}" for n in range(NS) for f in NBR_FEATURE_NAMES),
    *(f"bond{b}_{f}" for b in range(B) for f in BOND_FEATURE_NAMES),
)

OUTPUT_NAMES: Final[tuple[str, ...]] = (
    "thrust",
    "turn",
    *(f"msg{m}" for m in range(MSG)),
    *(f"emit{c}" for c in range(NCH)),
    "harvest",
    "grip",
    "contract",
    "bond",
    *(f"release{b}" for b in range(B)),
    *(f"share{b}" for b in range(B)),
    "attack",
    "defend",
    "reproduce",
    "child_bond",
    *(f"q_bond{i}" for i in range(F)),
    *(f"q_attack{i}" for i in range(F)),
)


# --- the layout per dims (§27.1) ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    """The genome I/O layout of an ``nd``-dimensional world (``for_dims``).

    3D appends a tail to the 2D vectors ("append-only"): every module constant above keeps its value and
    meaning in both dims, so only the totals, the descriptor width ``NF``, the query width ``F`` and the
    version depend on dims. Indices that exist only in 3D are None in 2D (never −1, which would read the last
    column).

    The feature extensions (DESIGN §43) append blocks after the base total, in ``ext.TAIL_ORDER``, cumulative
    over the enabled features: ``EXT_IN``/``EXT_OUT`` (empty without any) list ``(owner, name, start,
    width, silent)`` and
    ``lay.IN_<NAME>``/``lay.OUT_<NAME>`` (``__getattr__``) give the offset of a registered block, None while
    its feature is off. Without extension blocks this is today's layout object, field for field.
    """

    nd: int
    D_IN: int
    D_OUT: int
    NF: int  # features per sensed neighbour: 7, or 8 with NBR_DZ last
    F: int  # targeting descriptor width NF − 1 (the bond and attack queries)
    INPUT_BLOCKS: tuple[tuple[str, int, int], ...]
    OUTPUT_BLOCKS: tuple[tuple[str, int, int], ...]
    INPUT_NAMES: tuple[str, ...]
    OUTPUT_NAMES: tuple[str, ...]
    NBR_FEATURE_NAMES: tuple[str, ...]
    # 3D only (None in 2D).
    NBR_DZ: int | None = None  # desc[..., NBR_DZ]: body-frame up offset / R
    IN_VEL_UP: int | None = None
    IN_RES_GRAD_UP: int | None = None
    IN_CHEM_GRAD_UP: int | None = None  # NCH wide
    IN_NBR_DZ: int | None = None  # NS wide
    IN_BOND_DZ: int | None = None  # B wide
    IN_HEIGHT: int | None = None
    IN_PITCH: int | None = None
    IN_GROUNDED: int | None = None
    OUT_PITCH: int | None = None
    OUT_BUOYANCY: int | None = None
    OUT_Q_BOND_DZ: int | None = None  # the query's 7th component, after the 2D block OUT_Q_BOND
    OUT_Q_ATTACK_DZ: int | None = None
    # Feature extension blocks (empty without an enabled feature with a brain footprint).
    EXT_IN: tuple[tuple[str, str, int, int, bool], ...] = ()
    EXT_OUT: tuple[tuple[str, str, int, int, bool], ...] = ()

    def __getattr__(self, name: str) -> int | None:
        """``IN_<BLOCK>``/``OUT_<BLOCK>`` of a registered extension block: its offset, or None when the
        feature is off; any other unknown name is an ``AttributeError``."""
        if name.startswith(("IN_", "OUT_")):
            key = name.split("_", 1)[1]
            if key in ext.registered_block_names():
                table = self.EXT_IN if name.startswith("IN_") else self.EXT_OUT
                return next((start for _, bname, start, _, _ in table if bname.upper() == key), None)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def silent_in_rows(self) -> tuple[tuple[int, int], ...]:
        """``(start, width)`` of the input blocks whose W1 rows start at zero for founders and immigrants."""
        return tuple((start, width) for _, _, start, width, silent in self.EXT_IN if silent)

    def silent_out_cols(self) -> tuple[tuple[int, int], ...]:
        """``(start, width)`` of the output blocks whose W2 columns start at zero."""
        return tuple((start, width) for _, _, start, width, silent in self.EXT_OUT if silent)


def _layout_2d() -> Layout:
    return Layout(
        nd=2,
        D_IN=D_IN,
        D_OUT=D_OUT,
        NF=NF,
        F=F,
        INPUT_BLOCKS=INPUT_BLOCKS,
        OUTPUT_BLOCKS=OUTPUT_BLOCKS,
        INPUT_NAMES=INPUT_NAMES,
        OUTPUT_NAMES=OUTPUT_NAMES,
        NBR_FEATURE_NAMES=NBR_FEATURE_NAMES,
    )


def _layout_3d() -> Layout:
    """The 2D vectors unchanged, then the 3D-only tail (docs/3D.md §4, §27.1). Frozen: any change to these
    tables bumps ``GENOME_LAYOUT_VERSION_3D``."""
    in_tail = (
        ("vel_up", 69, 1),  # body-frame up component of vel / v_max
        ("res_grad_up", 70, 1),
        ("chem_grad_up", 71, NCH),
        ("nbr_dz", 74, NS),  # up offset / R of the NS nearest neighbours
        ("bond_dz", 77, B),  # up offset / break_length per bond slot
        ("height", 81, 1),  # z / world.height
        ("pitch", 82, 1),  # sin(pitch)
        ("grounded", 83, 1),  # 1 on the floor, else 0
    )
    out_tail = (
        ("pitch", 35, 1),  # pitch change turn_max·tanh(y)
        ("buoyancy", 36, 1),  # buoyancy fraction max(tanh(y), 0)
        ("q_bond_dz", 37, 1),  # 7th bond-query component, after OUT_Q_BOND's six
        ("q_attack_dz", 38, 1),  # 7th attack-query component
    )
    at = {name: start for name, start, _ in in_tail}
    out_at = {name: start for name, start, _ in out_tail}
    return Layout(
        nd=3,
        D_IN=84,
        D_OUT=39,
        NF=NF + 1,
        F=F + 1,
        INPUT_BLOCKS=INPUT_BLOCKS + in_tail,
        OUTPUT_BLOCKS=OUTPUT_BLOCKS + out_tail,
        INPUT_NAMES=(
            *INPUT_NAMES,
            "vel_up",
            "res_grad_up",
            *(f"chem{c}_grad_up" for c in range(NCH)),
            *(f"nbr{n}_dz" for n in range(NS)),
            *(f"bond{b}_dz" for b in range(B)),
            "height",
            "pitch",
            "grounded",
        ),
        OUTPUT_NAMES=(*OUTPUT_NAMES, "pitch", "buoyancy", "q_bond_dz", "q_attack_dz"),
        NBR_FEATURE_NAMES=(*NBR_FEATURE_NAMES, "dz"),
        NBR_DZ=NF,
        IN_VEL_UP=at["vel_up"],
        IN_RES_GRAD_UP=at["res_grad_up"],
        IN_CHEM_GRAD_UP=at["chem_grad_up"],
        IN_NBR_DZ=at["nbr_dz"],
        IN_BOND_DZ=at["bond_dz"],
        IN_HEIGHT=at["height"],
        IN_PITCH=at["pitch"],
        IN_GROUNDED=at["grounded"],
        OUT_PITCH=out_at["pitch"],
        OUT_BUOYANCY=out_at["buoyancy"],
        OUT_Q_BOND_DZ=out_at["q_bond_dz"],
        OUT_Q_ATTACK_DZ=out_at["q_attack_dz"],
    )


def _with_ext(lay: Layout, blocks: ext.ExtBlocks) -> Layout:
    """``lay`` with the extension blocks appended after its base totals, in the order given
    (``TAIL_ORDER``)."""
    d_in, d_out = lay.D_IN, lay.D_OUT
    ext_in: list[tuple[str, str, int, int, bool]] = []
    ext_out: list[tuple[str, str, int, int, bool]] = []
    in_blocks, out_blocks = list(lay.INPUT_BLOCKS), list(lay.OUTPUT_BLOCKS)
    in_names, out_names = list(lay.INPUT_NAMES), list(lay.OUTPUT_NAMES)
    nbr_names = list(lay.NBR_FEATURE_NAMES)
    for owner, ins, outs, nbr in blocks:
        for bname, width, silent in ins:
            ext_in.append((owner, bname, d_in, width, silent))
            in_blocks.append((bname, d_in, width))
            in_names += [bname] if width == 1 else [f"{bname}{i}" for i in range(width)]
            d_in += width
        for bname, width, silent in outs:
            ext_out.append((owner, bname, d_out, width, silent))
            out_blocks.append((bname, d_out, width))
            out_names += [bname] if width == 1 else [f"{bname}{i}" for i in range(width)]
            d_out += width
        nbr_names += nbr
    extra = len(nbr_names) - len(lay.NBR_FEATURE_NAMES)  # descriptor columns (arms perception) widen NF and F
    return dataclasses.replace(
        lay,
        D_IN=d_in,
        D_OUT=d_out,
        NF=lay.NF + extra,
        F=lay.F + extra,
        INPUT_BLOCKS=tuple(in_blocks),
        OUTPUT_BLOCKS=tuple(out_blocks),
        INPUT_NAMES=tuple(in_names),
        OUTPUT_NAMES=tuple(out_names),
        NBR_FEATURE_NAMES=tuple(nbr_names),
        EXT_IN=tuple(ext_in),
        EXT_OUT=tuple(ext_out),
    )


@functools.cache
def _for_dims(nd: int, blocks: ext.ExtBlocks) -> Layout:
    if nd == 2:
        lay = _layout_2d()
    elif nd == 3:
        lay = _layout_3d()
    else:
        raise ValueError(f"no genome layout for dims = {nd} (2 or 3)")
    return _with_ext(lay, blocks) if blocks else lay


def for_dims(nd: int, blocks: ext.ExtBlocks = ()) -> Layout:
    """The layout of an ``nd``-dimensional world (2 or 3) with the extension ``blocks`` of the enabled
    features
    (``ext.layout_blocks(cfg, nd)``, cumulative in ``TAIL_ORDER``); ``config.dims(cfg).lay`` holds the
    config's
    one. Cached on both arguments; without blocks it is today's layout."""
    return _for_dims(nd, tuple(blocks))


def genome_layout_version(nd: int, mask: int = 0) -> int:
    """The genome layout version of an ``nd``-dimensional world, read at call time (tests monkeypatch it):
    ``base(nd) + 4·mask``, where bit ``i`` of ``mask`` is set when extension ``i`` has a brain footprint
    (``ext.mask(cfg)``). It decodes uniquely (``base = v % 4``, ``mask = v // 4``); the ``layout_features``
    meta string says which widths and block versions (DESIGN §43.2)."""
    if nd == 2:
        base = GENOME_LAYOUT_VERSION
    elif nd == 3:
        base = GENOME_LAYOUT_VERSION_3D
    else:
        raise ValueError(f"no genome layout for dims = {nd} (2 or 3)")
    return base + 4 * mask
