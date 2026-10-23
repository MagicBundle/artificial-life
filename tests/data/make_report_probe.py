"""Write ``tests/data/report_probe/*.json``: the twelve sheets of the narrator probe (docs/DESIGN.md §31), in the
fact-sheet format of ``game.report`` (version 2: the kin group events are the thread events of the chronicle,
``th`` below). The numbers are those of the probe's synthetic chapters (2D and 3D, quiet and eventful, a collapse
after a catastrophe, an unobserved run); ``report_crafted.json`` holds the crafted false or misleading sentences
of the two reviews on these sheets.

    PYTHONPATH=tests python tests/data/make_report_probe.py [OUT_DIR]

No simulation. The sheets are deterministic; a test checks that the committed files are what this writes."""

# ruff: noqa: E501  (expected report texts and fixture tables are long lines)

from __future__ import annotations

import json
import sys
from pathlib import Path

from report_helpers import deaths, group, houses, interv, mk_sheet, pop, thread, turnover, untold

NAMES = {12: "Braelix", 31: "Shaemund", 7: "Torvane", 19: "Kraiost", 5: "Flauth", 2: "Dreanor", 14: "Glouth",
         9: "Pheask", 3: "Stiorn", 27: "Chuarck", 4: "Blaxen", 40: "Vorith", 44: "Gloarn", 45: "Phouth",
         8: "Gloarn", 21: "Brauth", 15: "Clevask", 17: "Draeck", 52: "Shoum", 51: "Prialth"}  # fmt: skip


def th(kind, tid, size, *, into=None, parent=None, first, told, end=None, initial=False, peak=None):
    """A thread event: ``formed`` (from ``parent``, a kin id) or ``ended`` (into ``into``, or dissolved)."""
    other = NAMES[into] if into is not None else NAMES[parent] if parent is not None else None
    return thread(kind, tid, NAMES[tid], size, other=other, first=first, told=told, end=end, initial=initial,
                  peak=peak)  # fmt: skip


def births(b, imm=0):
    return {"kind": "births", "births": b, "immigrants": imm}


def top(fs, ts, alive_from, alive_to):
    fc, tc = round(fs * alive_from), round(ts * alive_to)
    return {"kind": "top_lineage", "from_share": fc / alive_from, "to_share": tc / alive_to, "from_count": fc,
            "to_count": tc, "from_of": alive_from, "to_of": alive_to}  # fmt: skip


def lin(a, b):
    return {"kind": "lineages", "from": a, "to": b}


def lab(key, family, fc, fo, tc, to):
    return {"kind": "label", "label": key.capitalize(), "key": key, "family": family, "from_share": fc / fo,
            "to_share": tc / to, "from_count": fc, "to_count": tc, "from_of": fo, "to_of": to}  # fmt: skip


def rel(a, b, attacks, kills):
    return {"kind": "relation", "a": a, "b": b, "attacks": attacks, "kills": kills}


def col(tracked, largest):
    return {"kind": "colonies", "tracked": tracked, "largest_size": largest}


def sheets() -> dict[str, dict]:
    out = {}

    def add(name, a, b, kinds, ids, dims=2, hid=3, hname="Zaeld"):
        if "deaths" in kinds and "births" in kinds:  # the sheet builder's turnover fact
            p0 = kinds["population"][0]
            kinds = {**kinds, "turnover": [turnover(kinds["births"][0]["births"], kinds["deaths"][0]["total"],
                                                    p0["from"], p0["to"])]}  # fmt: skip
        ents = [("kin_group", i, NAMES[i]) for i in ids] + (
            [("house", hid, hname)] if "houses" in kinds else []
        )
        out.update({name: mk_sheet(a=a, b=b, kinds=kinds, ents=ents, dims=dims)})

    add("s01_2d_quiet", 1_506_000, 1_512_000, {
        "population": [pop(33_862, 34_410, 33_102)], "deaths": [deaths(15_940, 2_100, 180)], "births": [births(18_768)],
        "lineages": [lin(9, 9)], "top_lineage": [top(0.412, 0.418, 33_862, 34_410)], "untold": [untold(3)],
        "label": [lab("settler", "archetype", 20_690, 33_862, 21_260, 34_410)],
        "group": [group(12, 6_120, 34_410, 1, 2_010, 1_870, 240), group(7, 4_300, 34_410, 2, 1_480, 1_420, 190)],
        "houses": [houses(14, 14, 0, 0, size=9_800)], "colonies": [col(11, 40)]}, [12, 7])  # fmt: skip
    add("s02_2d_eventful", 1_512_000, 1_518_000, {
        "population": [pop(34_410, 21_140, 20_914)], "deaths": [deaths(36_050, 5_060, 190)], "births": [births(28_030)],
        "lineages": [lin(9, 7)], "top_lineage": [top(0.418, 0.471, 34_410, 21_140)],
        "label": [lab("hunter", "archetype", 2_925, 34_410, 2_558, 21_140), lab("resting", "intent", 6_194, 34_410, 5_496, 21_140),
                  lab("nomad", "archetype", 860, 34_410, 402, 21_140)],
        "group": [group(12, 5_020, 21_140, 1, 3_100, 4_900, 610), group(31, 1_830, 21_140, 2, 700, 80, 10)],
        "thread": [th("formed", 31, 1_210, parent=12, first=1_513_000, told=1_513_750), th("ended", 7, 980, first=1_508_000, told=1_517_000, end=1_516_000),
                   th("formed", 19, 640, first=1_514_500, told=1_515_250), th("ended", 5, 520, into=12, first=1_509_000, told=1_517_500, end=1_517_000)],
        "houses": [houses(14, 12, 0, 2, size=6_700)], "relation": [rel(12, 31, 412, 61)], "colonies": [col(7, 40)]},
        [12, 31, 7, 19, 5])  # fmt: skip
    add("s03_2d_lineage_collapse", 2_106_000, 2_112_000, {
        "population": [pop(28_900, 26_400, 25_870)], "deaths": [deaths(14_200, 3_900, 260)], "births": [births(15_860)],
        "lineages": [lin(14, 3)], "top_lineage": [top(0.44, 0.86, 28_900, 26_400)],
        "label": [lab("settler", "archetype", 15_895, 28_900, 18_744, 26_400), lab("hunter", "archetype", 3_179, 28_900, 1_584, 26_400)],
        "group": [group(2, 16_104, 26_400, 1, 7_900, 6_300, 1_500), group(14, 1_180, 26_400, 2, 400, 2_300, 600)],
        "thread": [th("ended", 14, 2_020, into=2, first=2_100_000, told=2_109_000, end=2_108_250), th("ended", 9, 1_140, first=2_101_000, told=2_110_250, end=2_109_500)],
        "houses": [houses(11, 4, 0, 7, largest=1, size=21_800)], "colonies": [col(5, 40)]}, [2, 14, 9], hid=1, hname="Kreoth")  # fmt: skip
    add("s04_2d_rule_intervention", 1_518_000, 1_524_000, {
        "population": [pop(45_020, 37_800, 37_610)], "deaths": [deaths(29_400, 3_100, 350)], "births": [births(25_630)],
        "lineages": [lin(8, 8)], "top_lineage": [top(0.39, 0.41, 45_020, 37_800)],
        "intervention": [interv(0, "regen_pulse", 1_519_000, {"mult": 0.800000011920929}, duration=10_000)],
        "label": [lab("resting", "intent", 7_653, 45_020, 9_072, 37_800)],
        "group": [group(12, 7_940, 37_800, 1, 3_900, 4_400, 520)],
        "thread": [th("formed", 27, 900, parent=3, first=1_522_000, told=1_522_750)], "houses": [houses(13, 13, 0, 0, size=15_200)]},
        [12, 3, 27])  # fmt: skip
    add("s05_2d_player_command", 1_524_000, 1_530_000, {
        "population": [pop(12_800, 14_950, 12_210)], "deaths": [deaths(7_450, 820, 90)], "births": [births(9_010, 500)],
        "lineages": [lin(5, 6)], "top_lineage": [top(0.52, 0.47, 12_800, 14_950)],
        "intervention": [interv(0, "immigration_pulse", 1_525_000, {"count": 500}, source="llm", rule=None)],
        "group": [group(4, 3_310, 14_950, 1, 2_400, 1_900, 260)],
        "thread": [th("formed", 40, 410, first=1_526_000, told=1_526_750)], "houses": [houses(6, 7, 1, 0, largest=2, size=6_900)]},
        [4, 40], hid=2, hname="Mivask")  # fmt: skip
    add("s06_2d_houses_merges", 1_530_000, 1_536_000, {
        "population": [pop(40_200, 37_700, 37_100)], "deaths": [deaths(26_900, 4_400, 300)], "births": [births(29_100)],
        "lineages": [lin(8, 7)], "top_lineage": [top(0.43, 0.45, 40_200, 37_700)],
        "label": [lab("caring", "intent", 482, 40_200, 716, 37_700), lab("signalling", "intent", 16_080, 40_200, 13_949, 37_700)],
        "group": [group(12, 6_800, 37_700, 1, 3_000, 2_700, 400), group(44, 2_100, 37_700, 2, 1_900, 300, 40)],
        "thread": [th("ended", 31, 1_500, into=12, first=1_530_000, told=1_532_000, end=1_531_250), th("formed", 45, 1_020, parent=12, first=1_533_500, told=1_534_250),
                   th("formed", 44, 700, first=1_532_000, told=1_532_750), th("ended", 19, 610, first=1_530_250, told=1_535_750, end=1_535_000)],
        "houses": [houses(12, 9, 0, 3, size=14_100)], "relation": [rel(12, 44, 210, 31), rel(12, 19, 96, 9)], "colonies": [col(14, 40)]},
        [12, 31, 44, 45, 19])  # fmt: skip
    add("s07_3d_quiet", 450_000, 456_000, {
        "population": [pop(41_310, 41_900, 40_950)], "deaths": [deaths(19_200, 6_100, 410)], "births": [births(26_300)],
        "lineages": [lin(5, 5)], "top_lineage": [top(0.331, 0.334, 41_310, 41_900)],
        "label": [lab("hunter", "archetype", 8_675, 41_310, 9_008, 41_900), lab("resting", "intent", 18_176, 41_310, 18_855, 41_900)],
        "group": [group(2, 7_960, 41_900, 1, 3_400, 3_200, 1_100)], "houses": [houses(9, 9, 0, 0, largest=6, size=13_363)]},
        [2], dims=3, hid=6, hname="Phouth")  # fmt: skip
    add("s08_3d_eventful", 456_000, 462_000, {
        "population": [pop(41_900, 30_250, 30_010)], "deaths": [deaths(16_800, 12_400, 400)], "births": [births(17_950)],
        "lineages": [lin(5, 5)], "top_lineage": [top(0.334, 0.36, 41_900, 30_250)],
        "label": [lab("hunter", "archetype", 8_799, 41_900, 8_773, 30_250), lab("hunting", "intent", 6_285, 41_900, 6_655, 30_250),
                  lab("fleeing", "intent", 419, 41_900, 484, 30_250), lab("resting", "intent", 18_855, 41_900, 11_495, 30_250)],
        "group": [group(2, 5_140, 30_250, 1, 2_300, 4_900, 2_100), group(8, 2_480, 30_250, 2, 900, 1_400, 800)],
        "thread": [th("formed", 21, 2_300, parent=2, first=457_000, told=457_750), th("ended", 8, 1_100, into=2, first=456_250, told=460_000, end=459_250),
                   th("formed", 15, 520, first=460_500, told=461_250)],
        "houses": [houses(9, 8, 0, 1, largest=6, size=9_900)], "relation": [rel(2, 8, 905, 210)]},
        [2, 8, 21, 15], dims=3, hid=6, hname="Phouth")  # fmt: skip
    add("s09_3d_collapse_catastrophe", 462_000, 468_000, {
        "population": [pop(30_250, 9_800, 8_900)], "deaths": [deaths(7_100, 3_300, 150, 9_400)], "births": [births(9_500)],
        "lineages": [lin(5, 2)], "top_lineage": [top(0.38, 0.74, 30_250, 9_800)],
        "intervention": [interv(0, "catastrophe", 463_000, {"center": [120.0, 88.0], "radius": 40.0, "kill_p": 0.6000000238418579},
                                source="llm", rule=None),
                         interv(1, "regen_pulse", 465_000, {"mult": 1.5}, rule="extinction_rescue", duration=5_000)],
        "group": [group(15, 2_900, 9_800, 1, 1_400, 700, 200)],
        "thread": [th("ended", 2, 3_900, first=458_000, told=463_750, end=462_750), th("ended", 8, 1_700, first=459_000, told=465_000, end=464_000)],
        "houses": [houses(8, 4, 0, 4, largest=6, size=6_200)]}, [15, 2, 8], dims=3, hid=6, hname="Phouth")  # fmt: skip
    add("s10_3d_storm_free", 300_000, 306_000, {
        "population": [pop(38_000, 39_200, 37_640)], "deaths": [deaths(17_100, 5_200, 380)], "births": [births(23_880)],
        "lineages": [lin(6, 6)], "top_lineage": [top(0.29, 0.31, 38_000, 39_200)],
        "group": [group(3, 5_600, 39_200, 1, 2_500, 2_100, 700)],
        "thread": [th("formed", 17, 640, parent=3, first=303_000, told=303_750)], "houses": [houses(10, 10, 0, 0, largest=4, size=11_800)]},
        [3, 17], dims=3, hid=4, hname="Gloarn")  # fmt: skip
    add("s11_2d_two_interventions", 1_536_000, 1_542_000, {
        "population": [pop(37_700, 48_100, 37_520)], "deaths": [deaths(21_300, 3_700, 280)], "births": [births(35_680)],
        "lineages": [lin(7, 7)], "top_lineage": [top(0.45, 0.44, 37_700, 48_100)],
        "intervention": [interv(0, "regen_pulse", 1_527_000, {"mult": 0.800000011920929}, duration=10_000, end=1_537_000,
                                end_by="elapsed", restored={"mult": 1.0}),
                         interv(1, "set_seasons", 1_539_000, {"period": 20_000, "amp": 0.4000000059604645}, rule="stagnation_response")],
        "label": [lab("nomad", "archetype", 1_017, 37_700, 1_828, 48_100), lab("roaming", "intent", 8_294, 37_700, 9_620, 48_100)],
        "group": [group(12, 8_300, 48_100, 1, 4_100, 3_300, 500)],
        "thread": [th("formed", 52, 1_400, parent=12, first=1_538_000, told=1_538_750), th("formed", 51, 450, first=1_540_500, told=1_541_250)],
        "houses": [houses(9, 9, 0, 0, size=19_000)], "relation": [rel(12, 52, 133, 17)]}, [12, 52, 51])  # fmt: skip
    add("s12_2d_unobserved_llm", 792_000, 798_000, {
        "population": [pop(21_400, 18_900, 18_640)], "deaths": [deaths(14_100, 1_900, 260)], "births": [births(13_660, 200)],
        "intervention": [interv(0, "regen_pulse", 793_000, {"mult": 1.5}, source="llm", rule=None, duration=5_000),
                         interv(1, "immigration_pulse", 796_000, {"count": 200}, source="llm", rule=None)]}, [])  # fmt: skip
    return out


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "report_probe"
    out.mkdir(parents=True, exist_ok=True)
    for name, sheet in sheets().items():
        (out / f"{name}.json").write_text(
            json.dumps(sheet, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"wrote {len(sheets())} sheets to {out}")
