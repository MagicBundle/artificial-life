"""What the sphere GUI (PR S6) needs of the engine's control and stream text, found while building it.

* A place given to a power is a longitude and a latitude in degrees (DESIGN §46.7), which are signed: the
  control line ``power bloom id=1 x=-30.5 y=-60`` must parse (the first grammar took plain decimals, so the
  southern hemisphere and the west could not be named).
* The food patches of an ``ENVS`` of a sphere (``i x y z lon lat gain sigma``) pass the writer's own check.
* The Swift decoder's fixtures are what the writers make (``tests/make_sphere_gui_fixture.py``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from alife import play
from alife import streamrec as sr

ROOT = Path(__file__).resolve().parents[1]


def test_a_place_of_a_power_may_be_signed() -> None:
    line = play._parse_power(["bloom", "id=3", "x=-30.5", "y=-60", "gain=1.5"])
    assert line.power == "bloom" and line.id == 3
    assert dict(line.params)["x"] == -30.5 and dict(line.params)["y"] == -60.0


@pytest.mark.parametrize(
    "bad", ["x=--3", "y=+4", "x=-", "x=1e3", "x=- 3", "gain=-1", "radius=-5", "kill=-0.1", "mult=-2"]
)
def test_only_the_place_is_signed(bad: str) -> None:
    with pytest.raises(play.ControlError):
        play._parse_power(["strike", "id=1", bad])


def test_the_patches_of_a_sphere_pass_the_writers_check() -> None:
    sphere = {"i": 0, "x": 0.5, "y": 0.5, "z": 0.7071, "lon": 45.0, "lat": 45.0, "gain": 1.0, "sigma": 12.0}
    sr.check_envs({"patches": [sphere]})
    sr.check_envs(
        {"patches": [{"i": 0, "x": 1.0, "y": 2.0, "gain": 1.0, "sigma": 3.0}]}
    )  # a box's, unchanged
    for bad in (
        {k: v for k, v in sphere.items() if k != "lat"},  # z without lon, lat
        {**sphere, "extra": 1.0},
        {"i": 0, "x": 1.0, "y": 2.0, "gain": 1.0, "sigma": 3.0, "lon": 1.0},  # lon without z
    ):
        with pytest.raises(sr.RecordError):
            sr.check_envs({"patches": [bad]})


def test_the_sphere_gui_fixtures_are_what_the_writers_make() -> None:
    spec = importlib.util.spec_from_file_location(
        "make_sphere_gui_fixture", ROOT / "tests" / "make_sphere_gui_fixture.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.path.insert(0, str(ROOT / "tests"))
    spec.loader.exec_module(module)
    assert module.build() == (ROOT / "tests" / "data" / "play_sphere.bin").read_bytes()
    import json

    committed = json.loads((ROOT / "tests" / "data" / "sphere_gui.json").read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(module.expected()))
