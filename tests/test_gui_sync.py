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
        stored = depth_at == 1 and "{" not in stripped  # a computed property is no JSON key
        if stored and (v := re.match(r"(?:package |private |public )?var (\w+)\s*:", stripped)):
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
        labels=sr.labels_block(
            intent_version=3,
            arch=["", "A"],
            intent=["", "I"],
            gate={
                "rules_digest": "d",
                "world_key": "k",
                "key_found": True,
                "archetypes": [1],
                "intents": [1],
            },
        ),
        energy_cap=1,
    )
    g3b = sr.header_play(
        records=["FRMA"],
        weather=sr.weather_block(enabled=False, dc_ref=None),
        labels=play["labels"],
        energy_cap=1,
        alerts=sr.alerts_block(
            {"chapter": {"imp": 3, "pause": True, "label": "c", "calibrated": True, "surface": "alert"}}
        ),
        autopause=sr.autopause_block(chapter=True, alert={"chapter": True}),
        powers=sr.powers_block({"food": {"action": "a", "label": "l", "cost": 1, "cooldown": 1}}),
        game_specs="game-v0",
        saves=sr.saves_block(mode="all", keep=3, adopted=True),
    )
    assert swift_struct_keys(text, "PlayHeader") == set(g3b) and set(g3b) - set(play) == {
        "alerts", "autopause", "powers", "game_specs", "saves",
    }  # fmt: skip
    assert swift_struct_keys(text, "Weather") == set(play["weather"])
    assert swift_struct_keys(text, "Refs") == set(play["weather"]["refs"])
    assert swift_struct_keys(text, "Labels") == set(play["labels"])
    assert swift_struct_keys(text, "Gate") == set(play["labels"]["gate"])
    assert swift_struct_keys(text, "Cast") == set(play["cast"])
    assert swift_struct_keys(text, "Planet") == set(play["planet"])
    assert swift_struct_keys(text, "Alerts") == set(g3b["alerts"])
    assert swift_struct_keys(text, "Class") == set(next(iter(g3b["alerts"]["classes"].values())))
    assert swift_struct_keys(text, "Cap") == set(g3b["alerts"]["cap"])
    assert swift_struct_keys(text, "Autopause") == set(g3b["autopause"])
    assert swift_struct_keys(text, "Saves") == set(g3b["saves"])
    # the legacy header keys the Swift header reads are those `legacy_header` writes, plus `play`
    legacy = sr.legacy_header(
        dims=3, size=1.0, height=1.0, n=1, bond_slots=1, radius=1.0, field_cell=1.0, fg=1, fz=1, nch=1,
        step=0, run=None, config_hash="h", source="preset",
    )  # fmt: skip
    stream_header = (CORE / "StreamProtocol.swift").read_text(encoding="utf-8")
    assert swift_struct_keys(stream_header, "StreamHeader") == {*legacy, "play"}


def swift_body(text: str, header: str) -> str:
    """The text of the declaration that starts with `header` (`struct X` or `enum X`), braces included."""
    found = re.search(rf"{re.escape(header)}\b", text)
    assert found, f"{header} not found"
    start = found.start()
    i = text.index("{", start)
    depth, j = 0, i
    while True:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
        if depth == 0:
            return text[start:j]


def swift_nested_cases(text: str, name: str) -> set[str]:
    """The raw values of a nested `enum NAME: String` (one line or several): `case a, b`, `case c = "x"`."""
    body = swift_body(text, f"enum {name}")
    body = body[body.index("{") + 1 :]
    values = set()
    for m in re.finditer(r"\bcase\s+([^\n}]+)", body):
        for item in m.group(1).split(","):
            parsed = re.fullmatch(r'\s*(\w+)\s*(?:=\s*"([^"]+)")?\s*', item)
            assert parsed, f"cannot read the case {item!r} of {name}"
            values.add(parsed.group(2) or parsed.group(1))
    return values


def fixture_json(tag: str) -> list[object]:
    """The JSON object of every `tag` record of the G3b fixture, as the independent decoder read it."""
    sidecar = json.loads((ROOT / "tests" / "data" / "play_tiny3d_g3b.json").read_text(encoding="utf-8"))
    return [r["fields"]["json"] for r in sidecar["records"] if r["tag"] == tag]


def keys_of(objects: list[dict]) -> set[str]:
    return {k for o in objects for k in o}


def test_the_g3b_records_have_the_keys_the_swift_types_decode() -> None:
    """ALRT, RPRT and PLAN: the keys `streamrec` checks (and the fixture, read by the independent Python
    decoder) are the keys of `PlayAlert`, `ReportBody` and `PlanBody` and their nested types."""
    from alife import streamrec as sr

    text = (CORE / "PlayRecords.swift").read_text(encoding="utf-8")
    assert swift_struct_keys(text, "PlayAlert") == set(sr.ALERT_KEYS)
    assert (
        swift_struct_keys(text, "Ref")
        == {"kind", "id"}
        == keys_of([a["ref"] for b in fixture_json("ALRT") for a in b if a["ref"]])
    )  # type: ignore[index]
    body = swift_body(text, "struct ReportBody")
    assert swift_struct_keys(body, "ReportBody") == set(sr.REPORT_KEYS)
    reports = [r for r in fixture_json("RPRT") if not r["withheld"]]  # type: ignore[index]
    assert swift_struct_keys(body, "Sentence") == keys_of([r["headline"] for r in reports])
    assert swift_struct_keys(body, "Line") == keys_of([b for r in reports for b in r["body"]])  # type: ignore[index]
    assert swift_struct_keys(body, "Tick") == keys_of([t for r in reports for t in r["ticker"]])  # type: ignore[index]
    assert swift_struct_keys(body, "Entity") == keys_of([e for r in reports for e in r["entities"].values()])  # type: ignore[index]
    assert swift_struct_keys(body, "Since") == keys_of([r["since"] for r in reports])
    assert swift_struct_keys(body, "Ending") == keys_of([e for r in reports for e in r["since"]["ending"]])  # type: ignore[index]
    assert swift_struct_keys(body, "Versions") == keys_of([r["versions"] for r in reports])
    plan = swift_body(text, "struct PlanBody")
    assert swift_struct_keys(plan, "PlanBody") == set(sr.PLAN_KEYS)
    plans = fixture_json("PLAN")
    since = [p["since"] for p in plans if p["since"]]  # type: ignore[index]
    # `disc` is a strike's organisms in its column: only a Strike's card carries it (the fixture's is a
    # Food's), and `powers.Card` writes it; the Swift type reads it as an optional.
    assert swift_struct_keys(plan, "Since") - {"disc"} == keys_of(since)
    assert "disc" in swift_struct_keys(plan, "Since")
    assert '"disc"' in (ROOT / "src" / "alife" / "game" / "powers.py").read_text(encoding="utf-8")
    assert swift_struct_keys(plan, "Disc") == {"before", "after"}
    assert swift_struct_keys(plan, "Row") == keys_of([r for s in since for r in s["rows"]])  # type: ignore[index]


def test_the_g3b_environment_and_identity_keys_match() -> None:
    """The keys G3b adds to `ENVS` (influence, patches, powers) and `IDNT.kin` (thread), and the decoded flag
    bits and caps, are the engine's."""
    from alife import streamrec as sr

    text = (CORE / "PlayRecords.swift").read_text(encoding="utf-8")
    envs = fixture_json("ENVS")[0]
    env = swift_body(text, "struct PlayEnvironment")
    assert swift_struct_keys(env, "PlayEnvironment") == set(envs)  # type: ignore[arg-type]
    assert swift_struct_keys(env, "Influence") == set(envs["influence"])  # type: ignore[index]
    assert swift_struct_keys(env, "Patch") == keys_of(envs["patches"])  # type: ignore[index]
    assert swift_struct_keys(env, "PowerState") == keys_of(list(envs["powers"].values()))  # type: ignore[index]
    kin = [k for r in fixture_json("IDNT") for k in r["kin"]]  # type: ignore[index]
    assert swift_struct_keys(swift_body(text, "struct IdentityTables"), "Kin") == keys_of(kin)
    sidecar = json.loads((ROOT / "tests" / "data" / "play_tiny3d_g3b.json").read_text(encoding="utf-8"))
    power = sidecar["header"]["play"]["powers"]["food"]
    assert swift_struct_keys(swift_body(text, "struct PlayHeader"), "Power") == set(power)
    # the caps and flag bits the decoder enforces are the writer's
    assert f"maxAlerts = {sr.MAX_ALERTS}" in text
    for name, cap in (
        ("maxAlertJSON", sr.ALRT_JSON_CAP),
        ("maxReportJSON", sr.RPRT_JSON_CAP),
        ("maxPlanJSON", sr.PLAN_JSON_CAP),
    ):
        assert f"{name} = {cap >> 10} << 10" in text, name
    assert f"maxTitle = {sr.ALERT_TITLE_MAX}" in text and f"maxText = {sr.ALERT_TEXT_MAX}" in text
    assert f"maxText = {sr.PLAN_TEXT_MAX}" in text and f"maxCaution = {sr.PLAN_CAUTION_MAX}" in text
    assert f"idBase: UInt64 = {sr.ALERT_ID_BASE}" in text
    assert (sr.CLCK_FLAG_CHAPTER_PAUSE, sr.CLCK_FLAG_ALERT_PAUSE, sr.CLCK_FLAG_SAVES_NEWEST) == (1, 2, 4)
    assert "var chapterPauseArmed: Bool { flags & 1 != 0 }" in text
    assert "var alertPauseArmed: Bool { flags & 2 != 0 }" in text
    assert "var savesNewestOnly: Bool { flags & 4 != 0 }" in text
    for name, values in (
        ("State", sr.ALERT_STATES),
        ("Surface", sr.ALERT_SURFACES),
        ("Status", sr.PLAN_STATUSES),
        ("Kind", sr.PLAN_KINDS),
    ):
        owner = (
            swift_body(text, "struct PlayAlert")
            if name in ("State", "Surface")
            else swift_body(text, "struct PlanBody")
        )
        assert swift_nested_cases(owner, name) == set(values), name
    assert swift_nested_cases(swift_body(text, "struct Ref"), "Kind") == set(sr.ALERT_REF_KINDS)


def test_the_powers_grid_names_what_the_header_and_the_grammar_have() -> None:
    """The Powers tab reads the header's bounds (``play.powers``) and sends ``power`` lines: the powers it
    lists,
    the keys of each line and the header's parameter names are the engine's (DESIGN §34.7)."""
    from alife import play
    from alife.config import Config
    from alife.game import powers

    text = (CORE / "PowersModel.swift").read_text(encoding="utf-8")
    order = re.search(r"static let order: \[PowerCommand.Name\] = \[([^\]]*)\]", text)
    assert order is not None
    assert tuple(n.strip().lstrip(".") for n in order.group(1).split(",")) == play.POWER_NAMES
    # the keys of each power's line, in the order they are sent (`lineKeys`), are keys the engine takes for it
    keys = re.search(r"var lineKeys: \[PowerCommand.Key\] \{(.*?)\n    \}", text, re.S)
    assert keys is not None
    swift_keys = {
        m.group(1): [k.strip().lstrip(".") for k in m.group(2).split(",")]
        for m in re.finditer(r"case \.(\w+): \[([^\]]*)\]", keys.group(1))
    }
    for name, spec in powers.POWER_SPECS.items():
        assert set(swift_keys[name]) <= set(spec.keys), name
        optional = {"x", "y"} if name in ("bloom", "blight") else set()  # (the window names the patch itself)
        assert set(swift_keys[name]) >= set(spec.keys) - optional, name
    # the header's parameter names: each is a line key, but `kill_p`, which the line spells `kill`
    for name, entry in powers.header_block(Config()).items():
        for param in entry["params"]:
            line_key = "kill" if param == "kill_p" else param
            assert line_key in play.POWER_KEYS, (name, param)
    assert 'name == "kill_p" ? .kill' in text


def test_the_world_keys_are_the_documented_ones() -> None:
    """``WorldKeys`` is the World window's key table; DESIGN §34.7 lists every key it binds."""
    text = (CORE / "WorldKeys.swift").read_text(encoding="utf-8")
    bound: dict[str, str] = {}
    for labels, action in re.findall(r"case ((?:\"[^\"]+\"(?:, )?)+): return \.(\w+)", text):
        for key in re.findall(r'"([^"]+)"', labels):
            bound[key] = action
    assert bound == {
        " ": "togglePause",
        ".": "step",
        "c": "cycleColor",
        "1": "toggleAgents",
        "2": "toggleBonds",
        "3": "cycleTerritory",
        "f": "followSelection",
        "[": "slower",
        "]": "faster",
        "r": "home",
        "h": "home",
        "n": "northUp",
        "i": "toggleMinimap",
        "t": "viewPreset",
        "o": "viewPreset",
        "l": "viewPreset",
        "s": "viewPreset",
        "\\u{1B}": "escape",
        "g": "continueRun",
        "p": "showPowers",
        "a": "flyToAlert",
        "k": "acknowledgeAlert",
        "\\r": "confirmPower",
        "\\u{3}": "confirmPower",
    }
    design = (ROOT / "docs" / "DESIGN.md").read_text(encoding="utf-8")
    section = design[design.index("### 34.7") :]
    for action in ("escape", "continueRun", "showPowers", "flyToAlert", "acknowledgeAlert", "confirmPower"):
        assert action in section, action
    # Esc never resumes and never acknowledges (G3b review round 2, finding 21): it has its own action
    assert bound["\\u{1B}"] == "escape"


def test_the_power_grammar_the_window_sends_is_the_engines() -> None:
    from alife import play

    text = (CORE / "PlayCommand.swift").read_text(encoding="utf-8")
    assert swift_nested_cases(text, "Name") == set(play.POWER_NAMES)
    assert swift_nested_cases(text, "Key") == set(play.POWER_KEYS)
    assert "isWhole: Bool { self == .steps || self == .patch }" in text
    assert {"steps", "patch"} == play.POWER_INT_KEYS


# --- the `alife play` command line and control grammar (DESIGN §33.2.6, §33.2.8) -------------------------


def play_subparser() -> object:
    import argparse

    from alife import cli

    parser = cli.build_parser()
    action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return action.choices["play"]


def test_the_play_launch_flags_are_flags_of_alife_play() -> None:
    """Every `--flag` literal of PlayLaunch.swift is an option of `alife play`; the enums behind the flags
    that take a choice are the engine's."""
    text = (CORE / "PlayLaunch.swift").read_text(encoding="utf-8")
    flags = set(re.findall(r'"(--[a-z][a-z-]*)"', text))
    assert flags, "no flag found in PlayLaunch.swift"
    parser = play_subparser()
    options = {s for a in parser._actions for s in a.option_strings}  # type: ignore[attr-defined]
    assert flags <= options, sorted(flags - options)
    # The enums behind --houses and --director are the engine's choices.
    choices = {s: a.choices for a in parser._actions for s in a.option_strings if a.choices}  # type: ignore[attr-defined]
    assert swift_enum_values(text, "PlayHouses") == set(choices["--houses"])
    assert swift_enum_values(text, "PlayDirector") == set(choices["--director"])
    assert "storm" in choices["--overlay"]
    assert re.search(r'"--overlay", "storm"', text)


def test_the_play_launch_set_keys_are_config_fields() -> None:
    text = (CORE / "PlayLaunch.swift").read_text(encoding="utf-8")
    keys = {tuple(m.split(".")) for m in re.findall(r'"([a-z_]+\.[a-z_0-9]+)=', text)}
    assert keys == {("director", "enabled")}
    assert keys <= engine_keys()


def test_the_play_control_lines_the_window_sends_parse() -> None:
    """tests/data/play_command_lines.txt is what PlayCommandTests (Swift) checks PlayCommand against; the
    engine's own parser accepts every line of it, and every command name of the grammar is there."""
    from alife import play

    text = (ROOT / "tests" / "data" / "play_command_lines.txt").read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    assert lines and len(set(lines)) == len(lines)
    names = set()
    for line in lines:
        command = play.parse_play_command(line)
        assert command is not None, line
        names.add(command.name)
    assert names == set(play.COMMANDS)


def test_the_save_modes_the_window_sends_are_the_engines() -> None:
    """`SaveMode` (PlayCommand.swift) is the grammar of `saves` and of the header's `play.saves.mode`; the
    strip's words are the engine's help text (DESIGN section 34.10)."""
    from alife import play
    from alife import streamrec as sr

    text = (CORE / "PlayCommand.swift").read_text(encoding="utf-8")
    assert swift_enum_values(text, "SaveMode") == set(play.SAVE_MODES) == set(sr.SAVE_MODES)
    for mode in play.SAVE_MODES:
        assert play.parse_play_command(f"saves {mode}") == play.PlayCommand("saves", mode)
        assert sr.saves_block(mode=mode, keep=3)["mode"] == mode
    views = (ROOT / "gui" / "Sources" / "AlifeControl" / "StorageViews.swift").read_text(encoding="utf-8")
    assert (
        "Recording off keeps only the newest save; your world stays resumable; logs are still written."
        in views
    )


def test_the_play_world_file_keys_match_playworldfile() -> None:
    """The keys the launcher decodes from play.json are the ones `cli` gives `RunDir.write_play_json`."""
    from alife import play

    text = (CORE / "PlayWorlds.swift").read_text(encoding="utf-8")
    assert swift_struct_keys(text, "PlayWorldFile") == {
        "play_version", "base_rate", "cadence", "tick", "weather", "planet",
    }  # fmt: skip
    assert swift_struct_keys(text, "Cadence") == set(play.PLAY_CADENCE)
    assert swift_struct_keys(text, "Planet") == {"source", "step", "weather_spinup", "houses", "made_at_step"}
    assert swift_struct_keys(text, "Weather") == {"dc_ref"}


# --- the palettes (DESIGN §33.2.2) and the overlay constants the shaders share --------------------------


def swift_palette(text: str, name: str) -> list[int]:
    """The `0xRRGGBB` literals of `static let NAME: [UInt32]` (`none`: the unlabelled colour, as -1)."""
    m = re.search(rf"static let {name}: \[UInt32\] = \[(.*?)\]", text, re.S)
    assert m, f"palette {name} not found"
    return [
        int(v, 16) if v.startswith("0x") else -1 for v in re.findall(r"0x[0-9A-Fa-f]{6}|\bnone\b", m.group(1))
    ]


def test_the_world_palettes_are_the_engines() -> None:
    """The client draws swatches and territory with the table the server paints `FRMA.rgb` with."""
    from alife.game import colors

    def rgb(c: tuple[int, int, int]) -> int:
        return (c[0] << 16) | (c[1] << 8) | c[2]

    text = (CORE / "WorldPalette.swift").read_text(encoding="utf-8")
    assert swift_palette(text, "slots") == [rgb(c) for c in colors.SLOT_PALETTE]
    none = rgb(colors.NONE_RGB)

    def unlabelled(values: list[int]) -> list[int]:  # index 0 is `none` in the Swift
        return [none if v == -1 else v for v in values]

    assert unlabelled(swift_palette(text, "archetypes")) == [rgb(c) for c in colors.ARCH_PALETTE]
    assert unlabelled(swift_palette(text, "intents")) == [rgb(c) for c in colors.INTENT_PALETTE]
    for name, value in (
        ("unknown", colors.UNKNOWN_RGB),
        ("other", colors.OTHER_RGB),
        ("none", colors.NONE_RGB),
    ):
        m = re.search(rf"static let {name}: UInt32 = (0x[0-9A-Fa-f]{{6}})", text)
        assert m and int(m.group(1), 16) == rgb(value), name


def test_the_world_levels_match_the_spec_constants() -> None:
    """The thresholds of DESIGN §33.4.7 (0.12 L, 3 px) and §36.5 (the planet level from 0.45 of a world:
    the view never shows more than 0.49 L) are `WorldLevels.swift`'s."""
    levels = (CORE / "WorldLevels.swift").read_text(encoding="utf-8")
    assert re.search(r"planetHalfExtent: Float = 0\.45\b", levels)
    assert re.search(r"organismDistance: Float = 0\.12\b", levels)
    assert re.search(r"flatDiscPixels: Float = 3\b", levels)


def test_the_event_kind_numbers_are_the_engines() -> None:
    """The client acts on a few `EVNT` kinds by number (a hand-off moves the selection, a click on a row
    selects its entity): `EventKind.registry` is the registry of `streamrec`, and the names are the
    chronicle's."""
    from alife import streamrec
    from alife.game import chronicle

    text = (CORE / "WorldSelection.swift").read_text(encoding="utf-8")
    m = re.search(r"static let registry: \[Int: String\] = \[(.*?)\n    \]", text, re.S)
    assert m, "EventKind.registry not found"
    swift = {int(k): v for k, v in re.findall(r'(\d+): "([a-z_]+)"', m.group(1))}
    assert swift == dict(streamrec.EVENT_KINDS)
    assert {v: k for k, v in swift.items()} == dict(chronicle.KIND)
    # the named constants are those numbers
    for name, key in (
        ("kinBorn", "kin_born"),
        ("focusDeath", "focus_death"),
        ("focusHandoff", "focus_handoff"),
        ("stormNamed", "storm_named"),
        ("stormDissolved", "storm_dissolved"),
        ("extinct", "extinct"),
    ):
        found = re.search(rf"\b{name} = (\d+)", text)
        assert found and int(found.group(1)) == chronicle.KIND[key], name


def test_the_cast_causes_and_the_hemispheres_are_the_engines() -> None:
    """The cast lists a kin group's deaths by cause in the engine's order, and a storm's hemisphere by the
    chronicle's words."""
    from alife import state
    from alife.game import chronicle

    text = (CORE / "CastPolicy.swift").read_text(encoding="utf-8")
    m = re.search(r"deathCauses = \[(.*?)\]", text, re.S)
    assert m
    assert re.findall(r'"([a-z]+)"', m.group(1)) == [c.name.lower() for c in state.DeathCause]
    inspector = (CORE / "Inspector.swift").read_text(encoding="utf-8")
    for word in chronicle.HEMISPHERES:
        assert f'"{word}"' in text, f"{word} in CastPolicy.swift"
        assert f'"{word}"' in inspector, f"{word} in Inspector.swift"


def test_the_chronicle_minute_is_sixty_base_rate_steps() -> None:
    """A minute of game time is 60·B steps: 864 at the base rate the engine sends in the header."""
    from alife import play

    text = (CORE / "WorldChronicle.swift").read_text(encoding="utf-8")
    assert re.search(r"Int\(\(60 \* baseRate\)\.rounded\(\)\)", text)
    assert round(60 * play.BASE_RATE) == 864
