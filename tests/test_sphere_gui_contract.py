"""What the sphere GUI (PR S6) needs of the engine's control and stream text, found while building it.

* A place given to a power is a longitude and a latitude in degrees (DESIGN §46.7), which are signed: the
  control line ``power bloom id=1 x=-30.5 y=-60`` must parse (the first grammar took plain decimals, so the
  southern hemisphere and the west could not be named).
* The food patches of an ``ENVS`` of a sphere (``i x y z lon lat gain sigma``) pass the writer's own check.
* The Swift decoder's fixtures are what the writers make (``tests/make_sphere_gui_fixture.py``).
"""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

import sphere_decode as sd
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


def _close(got: object, want: object, where: str, *, atol: float, rtol: float) -> None:
    """Structural equality: integers, booleans, strings and the shapes of containers exactly; floats (and
    float arrays) within ``atol + rtol·|want|``. The fixtures are made with ``np.sin``, ``np.cos`` and
    ``np.cbrt``, whose last bit is the platform's libm's, so a float is compared by value, not bytes."""
    if isinstance(got, np.ndarray) or isinstance(want, np.ndarray):
        g, w = np.asarray(got), np.asarray(want)
        assert g.shape == w.shape and g.dtype == w.dtype, f"{where}: {g.dtype}{g.shape} != {w.dtype}{w.shape}"
        if g.dtype.kind == "f":
            assert np.allclose(g, w, rtol=rtol, atol=atol, equal_nan=False), f"{where}: floats differ"
        else:
            assert np.array_equal(g, w), f"{where}: arrays differ"
    elif isinstance(want, dict):
        assert isinstance(got, dict) and list(got) == list(want), f"{where}: keys differ"
        for k in want:
            _close(got[k], want[k], f"{where}.{k}", atol=atol, rtol=rtol)
    elif isinstance(want, list | tuple):
        assert isinstance(got, list | tuple) and len(got) == len(want), f"{where}: lengths differ"
        for k, (g, w) in enumerate(zip(got, want, strict=True)):
            _close(g, w, f"{where}[{k}]", atol=atol, rtol=rtol)
    elif isinstance(want, float) or isinstance(got, float):
        assert not isinstance(want, bool | str)
        assert abs(float(got) - float(want)) <= atol + rtol * abs(want), f"{where}: {got!r} != {want!r}"
    else:
        assert got == want and type(got) is type(want), f"{where}: {got!r} != {want!r}"


def _stream_close(got: bytes, want: bytes) -> None:
    """Two protocol 2 streams are the same up to the last bit of a float: the independent decoder reads both
    (so both are well-formed), and the headers, the record sequence and every field and array agree, exactly
    for integers and bytes of levels and ownership, within 1e-6 (relative) for float32 arrays and, for the
    JSON of a record, within the 0.01 of the two decimals its places are rounded to (a value within an ulp of
    ``.xx5`` may round the other way on another libm)."""
    a, b = sd.decode(got), sd.decode(want)
    _close(a.header, b.header, "header", atol=1e-9, rtol=1e-9)
    assert a.skipped == b.skipped == []
    assert [(r.tag, r.length, r.step, r.seq, r.version, r.flags) for r in a.records] == [
        (r.tag, r.length, r.step, r.seq, r.version, r.flags) for r in b.records
    ]
    for ra, rb in zip(a.records, b.records, strict=True):
        _close(ra.f, rb.f, ra.tag, atol=0.0101, rtol=1e-6)
        _close(ra.a, rb.a, ra.tag, atol=1e-5, rtol=1e-6)
        assert ra.sections.keys() == rb.sections.keys()
        for name in ra.sections:
            if name != "pos":  # the float32 positions are the arrays above; every other section is integers
                assert ra.sections[name] == rb.sections[name], f"{ra.tag}.{name} bytes differ"


def test_the_sphere_gui_fixtures_are_what_the_writers_make() -> None:
    """The committed stream and expectations equal what ``make_sphere_gui_fixture`` makes. The stream's JSON
    carries floats printed in full (the ``cbrt`` of the layer edges in the header) and its positions are
    float32 of ``sin``/``cos`` products, so its bytes are the maker's platform's: byte-equal on Darwin (where
    they are regenerated), equal in structure and by value elsewhere (the CI's Linux libm failed the byte
    comparison alone)."""
    spec = importlib.util.spec_from_file_location(
        "make_sphere_gui_fixture", ROOT / "tests" / "make_sphere_gui_fixture.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)

    sys.path.insert(0, str(ROOT / "tests"))
    spec.loader.exec_module(module)
    committed_bin = (ROOT / "tests" / "data" / "play_sphere.bin").read_bytes()
    built = module.build()
    _stream_close(built, committed_bin)
    if platform.system() == "Darwin":
        assert built == committed_bin
    committed = json.loads((ROOT / "tests" / "data" / "sphere_gui.json").read_text(encoding="utf-8"))
    made = json.loads(json.dumps(module.expected()))
    _close(made, committed, "sphere_gui.json", atol=1e-9, rtol=1e-9)
    if platform.system() == "Darwin":
        assert committed == made
