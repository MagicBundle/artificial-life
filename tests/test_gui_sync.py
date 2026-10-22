"""The GUI's levers, preset picker and director enums stay in step with the engine (gui/Sources/AlifeCore).

Plain text checks on the Swift sources: they need no Swift toolchain and no GPU. The GUI passes
`--set section.key=value` for a key, and pre-fills a fork from the same keys of `config.json`. A key the
engine does not have would be rejected at Start (exit 2) or never matched, so each name is checked against
`Config` here. The Swift tests (`cd gui && swift test`) pin the arguments themselves.
"""

from __future__ import annotations

import dataclasses
import json
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


# The config.json files the Swift argument tests (gui/Tests/AlifeCoreTests/ArgvCases.swift) fork and resume
# from, and how the engine writes them.
GUI_CONFIGS = {
    "dev": {"preset": "dev"},
    "tiny": {"preset": "tiny"},
    "dev_live": {
        "preset": "dev",
        "overrides": [
            "run.director=live",
            "director.enabled=true",
            "director.persona=trickster",
            "director.rule_every=1000",
            "fields.season_period=5000",
            "fields.season_amplitude=0.5",
            "fields.season_mode=per_patch",
            "population.immigration_floor=100",
            "mutation.rate_min=0.01",
        ],
    },
}


def test_the_gui_argument_fixtures_are_what_the_engine_writes() -> None:
    """tests/data/gui_config_*.json stand for real run directories; a new config key must be in them too.

    If this fails, regenerate the files with
    `config.save_config(config.load_config(**GUI_CONFIGS[name]), path)`. The Swift argument table does not
    change with them: a key at its default is not passed.
    """
    for name, kwargs in GUI_CONFIGS.items():
        path = ROOT / "tests" / "data" / f"gui_config_{name}.json"
        written = config.config_to_dict(config.load_config(**kwargs))
        assert json.loads(path.read_text(encoding="utf-8")) == json.loads(json.dumps(written)), path.name


# --- the `play` header key (DESIGN §33.1) ---------------------------------------------------------------


def swift_struct_keys(text: str, name: str) -> set[str]:
    """The JSON keys `struct NAME` decodes: its stored properties (depth 1 of its body, nested types
    excluded), renamed by the `CodingKeys` enum of the same body (`case a = "b"`; `case a, b` keep names)."""
    m = re.search(rf"struct {name}\b[^{{]*\{{", text)
    assert m, f"struct {name} not found"
    depth, i = 1, m.end()
    while depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    lines, depth_at = text[m.end() : i].split("\n"), 1
    props: list[str] = []
    renames: dict[str, str] = {}
    in_keys = False
    for line in lines:
        stripped = line.strip()
        if depth_at == 1 and (v := re.match(r"(?:package |private |public )?var (\w+)\s*:", stripped)):
            props.append(v.group(1))
        if re.match(r"(?:private )?enum CodingKeys", stripped):
            in_keys = True
        elif in_keys and stripped.startswith("case "):
            for part in stripped[5:].split(","):
                part = part.strip()
                if "=" in part:
                    k, raw = (x.strip() for x in part.split("="))
                    renames[k] = raw.strip('"')
        depth_at += line.count("{") - line.count("}")
        if in_keys and stripped == "}":
            in_keys = False
    return {renames.get(p, p) for p in props}


def test_the_play_header_keys_match_playheader() -> None:
    """The keys `streamrec.header_play` writes are the keys `PlayHeader` (and its nested types) reads: a key
    the writer adds without the client, or the client expects without the writer, is a protocol drift."""
    from alife import streamrec as sr

    text = (CORE / "PlayRecords.swift").read_text(encoding="utf-8")
    play = sr.header_play(
        records=["FRMA"],
        weather=sr.weather_block(enabled=True, dc_ref=0.5),
        labels={"intent_version": 2, "arch": [""], "intent": [""]},
        energy_cap=1,
    )
    assert swift_struct_keys(text, "PlayHeader") == set(play)
    assert swift_struct_keys(text, "Weather") == set(play["weather"])
    assert swift_struct_keys(text, "Refs") == set(play["weather"]["refs"])
    assert swift_struct_keys(text, "Labels") == set(play["labels"])
    assert swift_struct_keys(text, "Cast") == set(play["cast"])
    assert swift_struct_keys(text, "Planet") == set(play["planet"])
    # the legacy header keys the Swift header reads are those `legacy_header` writes, plus `play`
    legacy = sr.legacy_header(
        dims=3, size=1.0, height=1.0, n=1, bond_slots=1, radius=1.0, field_cell=1.0, fg=1, fz=1, nch=1,
        step=0, run=None, config_hash="h", source="preset",
    )  # fmt: skip
    stream_header = (CORE / "StreamProtocol.swift").read_text(encoding="utf-8")
    assert swift_struct_keys(stream_header, "StreamHeader") == {*legacy, "play"}
