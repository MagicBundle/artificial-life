"""The GUI's levers, preset picker and director enums stay in step with the engine (gui/Sources/AlifeCore).

Plain text checks on the Swift sources: they need no Swift toolchain and no GPU. The GUI passes
`--set section.key=value` for a key, and pre-fills a fork from the same keys of `config.json`. A key the
engine does not have would be rejected at Start (exit 2) or never matched, so each name is checked against
`Config` here. The Swift tests (`cd gui && swift test`) pin the arguments themselves.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from alife import config

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "gui" / "Sources" / "AlifeCore"

# A `--set` key the GUI passes, as a helper call (`decimal("mutation.rate_min", ...)`) or an interpolated
# argument (`"fields.season_mode=\(mode)"`); and a key it reads from a run's config.json
# (`decimalText("fields", "x")`).
SET_KEY = re.compile(r'(?:decimal|integer)\("([a-z_]+\.[a-z_0-9]+)"|"([a-z_]+\.[a-z_0-9]+)=')
READ_KEY = re.compile(
    r'(?:decimalText|integerText|value|string|integer|bool|number|int|double)\("([a-z_]+)", "([a-z_0-9]+)"'
)

# The 3D keys that are command-line only: the grounding tolerance is a numerical detail of the lattice
# (World3DLevers documents the choice).
CLI_ONLY_3D = {("physics", "floor_eps")}


def sources() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(CORE.glob("*.swift"))}


def engine_keys() -> set[tuple[str, str]]:
    cfg = config.Config()
    keys = set()
    for section in dataclasses.fields(cfg):
        for key in dataclasses.fields(getattr(cfg, section.name)):
            keys.add((section.name, key.name))
    return keys


def swift_keys(text: str) -> set[tuple[str, str]]:
    keys = set()
    for m in SET_KEY.finditer(text):
        section, key = (m.group(1) or m.group(2)).split(".")
        keys.add((section, key))
    keys.update((m.group(1), m.group(2)) for m in READ_KEY.finditer(text))
    return keys


def swift_enum_values(text: str, name: str) -> set[str]:
    """The raw values of `enum NAME: String` cases: `case a, b` and `case c = "x"`."""
    m = re.search(rf"enum {name}\b[^{{]*\{{(.*?)^\}}", text, re.S | re.M)
    assert m, f"enum {name} not found"
    values = set()
    for line in re.finditer(r"^\s*case\s+(.+)$", m.group(1), re.M):
        for item in line.group(1).split(","):
            parsed = re.fullmatch(r'\s*(\w+)\s*(?:=\s*"([^"]+)")?\s*', item)
            assert parsed, f"cannot read the case {item!r} of {name}"
            values.add(parsed.group(2) or parsed.group(1))
    return values


def test_the_scan_finds_the_levers() -> None:
    """A regex that stops matching must fail here, not pass by checking nothing."""
    keys = set().union(*(swift_keys(t) for t in sources().values()))
    for expected in [
        ("mutation", "rate_min"),
        ("population", "immigration_floor"),
        ("fields", "season_mode"),
        ("director", "rule_every"),
        ("run", "llm_url"),
        ("physics", "gravity"),
        ("energy", "buoyancy_cost"),
        ("world", "height"),
    ]:
        assert expected in keys, expected
    assert len(keys) >= 30


def test_every_swift_key_is_a_config_field() -> None:
    known = engine_keys()
    for name, text in sources().items():
        unknown = sorted(swift_keys(text) - known)
        assert not unknown, f"{name} names keys the engine's Config does not have: {unknown}"


def test_the_3d_levers_cover_the_3d_keys() -> None:
    """A new 3D key must be made a lever (or documented command-line only) in World3DLevers.swift."""
    text = sources()["World3DLevers.swift"]
    found = swift_keys(text)
    assert found == set(config.THREE_D_KEYS) - CLI_ONLY_3D, (
        "World3DLevers.swift and config.THREE_D_KEYS disagree: "
        f"only in Swift {sorted(found - set(config.THREE_D_KEYS))}, "
        f"only in the engine {sorted(set(config.THREE_D_KEYS) - CLI_ONLY_3D - found)}"
    )
    # The pre-fill of a fork reads the same keys from config.json (Levers.swift).
    levers = sources()["Levers.swift"]
    read = {(m.group(1), m.group(2)) for m in READ_KEY.finditer(levers)}
    assert set(config.THREE_D_KEYS) - CLI_ONLY_3D <= read


def test_the_preset_picker_lists_the_engine_presets() -> None:
    values = swift_enum_values(sources()["Levers.swift"], "Preset")
    assert values == set(config.PRESETS)
    for name in values:
        assert (ROOT / "src" / "alife" / "presets" / f"{name}.toml").is_file(), name


def test_the_director_enums_match_the_engine() -> None:
    text = sources()["DirectorSettings.swift"]
    assert swift_enum_values(text, "Persona") == set(config.PERSONAS)
    assert swift_enum_values(text, "DirectorMode") == set(config.DIRECTOR_MODES)
