"""Structural constants and genome I/O layout (§2.1, §5, §7).

Changing any value here changes the genome layout: bump ``GENOME_LAYOUT_VERSION`` with it.
"""

from __future__ import annotations

from typing import Final

GENOME_LAYOUT_VERSION: Final = 1

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

# --- fixed input normalisation constants (§5) and numeric guards (§9, §10.10) ---------------------------------
LR_SCALE: Final = 2.0
AGE_SCALE: Final = 1000.0
RES_SAT: Final = 2.0
RES_GRAD_GAIN: Final = 4.0
EPS_LEN: Final = 1e-6
C_REF_MARGIN: Final = 50

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
