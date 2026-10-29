"""Configs of the sphere series' tests: the three preset geometries of the spec on top of ``tiny3d`` (the
sphere presets arrive in S2; the sphere is refused until then, so tests widen ``GEOMETRY_SUPPORTED``)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from alife import config
from alife.config import Config, load_config

SPHERE_SETS = {
    "tiny3s": ["world.radius=36", "world.height=16", "world.nside=8", "world.layers=8"],
    "mini3s": ["world.radius=48", "world.height=24", "world.nside=11", "world.layers=12"],
    "64k3s": ["world.radius=90", "world.height=40", "world.nside=21", "world.layers=18"],
}
BASE = ["world.geometry=sphere", "world.size=1024", "sensing.cell_capacity=32", "physics.floor_eps=0.05"]


@pytest.fixture
def sphere_enabled(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """What S2 does for good: the sphere rules proper apply, not the refusal."""
    monkeypatch.setattr(config, "GEOMETRY_SUPPORTED", ("box", "sphere"))
    yield


def sphere_cfg(which: str = "tiny3s", *overrides: str) -> Config:
    """A validated sphere config (call under ``sphere_enabled``)."""
    return load_config(preset="tiny3d", overrides=[*BASE, *SPHERE_SETS[which], *overrides])
