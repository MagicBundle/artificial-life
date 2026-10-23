"""Shared fixtures (§22.0). Configs are frozen, so session scope is safe."""

from __future__ import annotations

import platform

import mlx.core as mx
import pytest

import worker_cache
from alife.config import Config, load_config

_JIT: list[worker_cache.Session] = []


def pytest_configure(config: pytest.Config) -> None:
    """MLX CPU kernels compile into a private cache per test process (``worker_cache``; DESIGN §22.0), so that
    the workers of ``pytest -n`` and concurrent pytest runs never share a half-built library."""
    if worker_cache.cap_applies():
        worker_cache.limit_mlx_buffer_cache()
    if (session := worker_cache.isolate_for(config)) is not None:
        _JIT.append(session)


def pytest_unconfigure(config: pytest.Config) -> None:
    while _JIT:
        _JIT.pop().close()


def pytest_report_header(config: pytest.Config) -> str:
    """The MLX version and default device in the header, so that a saved log (the Metal purity gate of DESIGN
    §28.8) says where it ran."""
    system = f"{platform.system()} {platform.machine()}"
    return f"mlx {mx.__version__}, default device {mx.default_device()}, {system}"


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


@pytest.fixture(scope="session")
def tiny3d_cfg() -> Config:
    """The ``tiny3d`` preset with float32 weights: the default for whole-step 3D tests (§27.8)."""
    return load_config(preset="tiny3d", overrides=["brain.weight_dtype=float32"])


@pytest.fixture(scope="session")
def micro3d_cfg() -> Config:
    """``micro_cfg``'s sizes on the ``tiny3d`` preset (``helpers.MICRO3D``): N = 64, L = 48, height 24."""
    from helpers import MICRO3D  # not at import: helpers reads the default device (ON_GPU)

    return load_config(preset="tiny3d", overrides=list(MICRO3D))


def pytest_addoption(parser: pytest.Parser) -> None:
    """Options of the ``llm``-marked test of the world report (tests/test_analysis_world_report.py): the id of
    a model that is already loaded and the chat-completions URL. Without ``--model`` (or both
    ``ALIFE_LLM_URL`` and ``ALIFE_LLM_MODEL``) that test is skipped; ``ALIFE_LLM_MODEL`` alone does not."""
    parser.addoption("--model", default=None, help="model id for the llm-marked world report test")
    parser.addoption(
        "--llm-url", default=None, help="chat-completions URL for the llm-marked world report test"
    )
