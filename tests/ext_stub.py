"""Test-only extension declarations (DESIGN §43): the machinery of PR 0a, exercised before a feature lands.

Five ``ExtSpec`` rows named after the five features, with the REAL block widths of the integration plan
(``docs``/DESIGN §43.2: structures 1/1, bonds 13/2, comm W + nd/W with W = 4, farming 1/1, arms perception 7/1
plus a descriptor column), conditional state fields, counter blocks, a comm-style action due and zero-valued
brain inputs. A stub is switched on by the config, with no new key: ``cfg.run.llm_model`` (a string the engine
never reads outside the LLM tier) is ``"stub:<name>,<name>"``; ``stub_cfg(cfg, "comm", "bonds")`` builds one.
``install(monkeypatch)`` replaces ``ext.REGISTRY`` for a test.
"""

from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import pytest

from alife import ext
from alife.config import Config, dims
from alife.ext.base import Block, ExtSpec, FieldSpec

PREFIX = "stub:"
ALL = ext.EXTENSIONS
WITHOUT_ARMS = ("structures", "bonds", "comm", "niche")  # arms' descriptor column is not wired into sensing


def stub_names(cfg: Config) -> tuple[str, ...]:
    m = cfg.run.llm_model
    return tuple(m.removeprefix(PREFIX).split(",")) if m.startswith(PREFIX) else ()


def stub_cfg(cfg: Config, *names: str) -> Config:
    return replace(cfg, run=replace(cfg.run, llm_model=PREFIX + ",".join(names)))


def _on(name: str):
    return lambda cfg: name in stub_names(cfg)


def _blocks(spec: list[tuple[str, int, bool]]):
    return lambda cfg, nd: tuple(Block(n, w, s) for n, w, s in spec)


def _field(name, per_agent, shape, dtype) -> FieldSpec:
    def rest(cfg: Config) -> mx.array:
        return mx.zeros(shape(dims(cfg)), dtype)

    return FieldSpec(name, per_agent, shape, dtype, rest)


def _fields(name: str):
    n = lambda d: (d.n,)  # noqa: E731
    cells = lambda d: d.field_shape  # noqa: E731
    return {
        "arms": (_field("arms", True, lambda d: (d.n, 6), mx.float32),),
        "structures": (_field("mat", False, cells, mx.int32),),
        "bonds": (
            _field("role", True, n, mx.int8),
            _field("role_dwell", True, n, mx.int16),
            _field("bond_flow", True, lambda d: (d.n, 4), mx.int16),
        ),
        "comm": (),
        "niche": (_field("soil", False, cells, mx.int32),),
    }[name]


COUNTERS = {
    "arms": ("ARMS_ATK_SUM", "ARMS_VIC_ARM_SUM", "ARMS_YIELD_SUM", "ARMS_MATCH_SUM", "ARMS_VIC_TOX_SUM"),
    "structures": (
        "BUILD_UNITS",
        "MAT_WASTED",
        "MAT_DECAYED",
        "MAT_TOTAL",
        "MAT_CELLS",
        "SHIELDED_ATTACKS",
        "TRAPPED",
        "APERTURE_VIOLATIONS",
        "SLOWED",
        "MAT_CLOSED",
    ),
    "bonds": ("ROLE_SWITCHES", "N_FORAGERS", "N_GUARDS", "SHIELDED_BY_GUARD", "SHARE_SLOTS_ACTIVE"),
    "comm": ("COMM_SPEAKERS", "COMM_DUE"),
    "niche": ("FARMERS", "FARM_YIELD", "SOIL_PUT", "SOIL_DECAYED", "SOIL_TOTAL"),
}
FLAGS = {
    "arms": "CONTESTED",
    "structures": "BUILDING",
    "bonds": "ROLE_SWITCHED",
    "comm": "SPOKE",
    "niche": "FARMING",
}
TOKENS = {
    "arms": "arms.perc@1",
    "structures": "structures@1",
    "bonds": "bonds@1",
    "comm": "comm4@1",
    "niche": "niche.farm@1",
}
PREFIXES = {"arms": "arms_", "structures": "build_", "bonds": "bond_", "comm": "comm_", "niche": "farm_"}
# (inputs, outputs, silent) with the plan's widths; comm's bearing block is nd wide
IN = {
    "structures": [("material", 1, False)],
    "bonds": [("role", 1, False), ("bond_role", 4, False), ("bond_kin", 4, False), ("bond_flow", 4, False)],
    "comm": [("voice", 4, False)],  # + ("voice_bearing", nd) below
    "niche": [("farm", 1, True)],
    "arms": [("self_atk", 1, True), ("self_arm", 1, True), ("self_tox", 1, True), ("self_sig", 1, True),
             ("nbr_sig", 3, True)],
}  # fmt: skip
OUT = {
    "structures": [("build", 1, False)],
    "bonds": [("role_a", 1, False), ("role_b", 1, False)],
    "comm": [("voice", 4, False)],
    "niche": [("farm_put", 1, True)],
    "arms": [("q_attack_sig", 1, True)],
}
NAMES = {k: tuple(n for n, _, _ in IN[k] + OUT[k]) + (("voice_bearing",) if k == "comm" else ()) for k in ALL}


def _in(name: str):
    def f(cfg: Config, nd: int) -> tuple[Block, ...]:
        blocks = [Block(n, w, s) for n, w, s in IN[name]]
        if name == "comm":
            blocks.append(Block("voice_bearing", nd, False))
        return tuple(blocks)

    return f


def _inputs(name: str):
    def f(s, cfg: Config):
        nd = dims(cfg).nd
        n = s.alive.shape[0]
        return tuple(mx.zeros((n, b.width), mx.float32) for b in _in(name)(cfg, nd))

    return f


def _count(name: str):
    def f(s_new, alive0, flows, cfg):
        n_alive = s_new.alive.astype(mx.int32).sum()
        return {c: n_alive * (j + 1) for j, c in enumerate(COUNTERS[name])}

    return f


def _due(s, act, cfg):  # comm-style: 2 quanta per living agent, folded into HEAT_ACTION by pay_costs
    return 2 * s.alive.astype(mx.int32)


def spec(name: str, *, due: bool | None = None, voice_w: int = 0) -> ExtSpec:
    return ExtSpec(
        name=name,
        enabled=_on(name),
        layout_on=_on(name),
        token=lambda cfg, name=name: TOKENS[name],
        block_names=NAMES[name],
        in_blocks=_in(name),
        out_blocks=_blocks(OUT[name]),
        nbr_features=(lambda cfg, nd: ("sig",)) if name == "arms" else (lambda cfg, nd: ()),
        dims=(lambda cfg: {"voice_w": voice_w}) if voice_w else (lambda cfg: {}),
        state_fields=lambda cfg, nd, name=name: _fields(name),
        counters=lambda cfg, name=name: COUNTERS[name],
        act_flag=FLAGS[name],
        metric_prefix=PREFIXES[name],
        inputs=_inputs(name),
        due=_due if (name == "comm" if due is None else due) else None,
        count=_count(name),
    )


def registry(**kw) -> tuple[ExtSpec, ...]:
    return tuple(spec(n, **kw) for n in ALL)


def install(monkeypatch: pytest.MonkeyPatch, **kw) -> tuple[ExtSpec, ...]:
    reg = registry(**kw)
    monkeypatch.setattr(ext, "REGISTRY", reg)
    return reg
