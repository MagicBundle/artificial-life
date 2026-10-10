"""Shared fixtures (§22.0). Configs are frozen, so session scope is safe."""

from __future__ import annotations

import pytest

from alife.config import Config, load_config


@pytest.fixture(scope="session")
def tiny_cfg() -> Config:
    """The ``tiny`` preset with float32 weights: the default for whole-step tests (F19)."""
    return load_config(preset="tiny", overrides=["brain.weight_dtype=float32"])


@pytest.fixture(scope="session")
def tiny_cfg_f16() -> Config:
    """The unmodified ``tiny`` preset (fp16 weights, the production dtype)."""
    return load_config(preset="tiny")


@pytest.fixture(scope="session")
def tiny_cfg_bf16() -> Config:
    """``tiny`` with bfloat16 weights, for digest / full_check / metrics smoke tests (F17)."""
    return load_config(preset="tiny", overrides=["brain.weight_dtype=bfloat16"])


@pytest.fixture(scope="session")
def micro_cfg() -> Config:
    """Hand-built unit-scene config: N = 64, M = 8, n_init = 8, L = 48 (G_h = 8, FG = 12), f32 weights."""
    return load_config(
        preset="tiny",
        overrides=[
            "population.n_max=64",
            "population.max_births_per_step=8",
            "population.n_init=8",
            "world.size=48",
            "brain.weight_dtype=float32",
        ],
    )
