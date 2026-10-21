"""The templates, the lexicon and the checker of the world report (docs/DESIGN.md §31): the catalogue is valid and
pinned, every template renders and passes the checker on scripted and random sheets, each rule R1 to R7 passes
and fails where it should, and the crafted sentences of the reviews and the corpus of the director's journals are
rejected. CPU only; no socket."""

# ruff: noqa: E501  (expected report texts and fixture tables are long lines)

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pytest

from alife.game import lexicon, templates
from alife.game import report as R
from alife.game.templates import TEMPLATES, Slot, Template
from report_helpers import A0, deaths, full_sheet, interv, mk_sheet, pop, random_sheet

DATA = Path(__file__).parent / "data"

# --- the catalogue -----------------------------------------------------------------------------------------------


def test_catalogue_is_valid_and_complete() -> None:
    templates.validate(TEMPLATES)
    roles = {t.role for t in TEMPLATES.values()}
    assert roles == {"lead", "body", "headline", "block", "ticker"}
    # every id of the specification's catalogue (§4.2), with the variants this implementation names
    expect = {
        "P-up", "P-down", "P-flat", "P-near", "P-zero", "P-low", "D-1", "D-2", "B-1", "B-2", "LN", "TL",
        "G-r1", "G-r2", "G-r3", "G-s", "G-c", "E-born", "E-born-o", "E-split", "E-merged", "E-dissolved",
        "E-dissolved-o", "CE-born", "CE-split", "CE-merged", "CE-dissolved", "H-1", "H-1n", "H-1e", "H-1ne",
        "H-1x", "H-1xn", "H-1xe", "H-1xne", "H-2", "LB-a", "LB-i", "R", "C", "C-1", "END", "HP-up", "HP-down",
        "HP-flat", "HP-near", "HP-dip", "HE-born", "HE-split", "HE-merged", "HE-dissolved", "HH", "HL", "I-regen", "I-loss",
        "I-seasons", "I-drift", "I-fert", "I-move", "I-imm", "I-cat", "S-regen", "S-gain", "S-seasons",
        "S-drift", "I-more", "X-regen", "X-regen-1", "X-gain", "X-gain-1", "X-replaced", "X-open", "W", "T-pop",
    }  # fmt: skip
    assert expect <= set(TEMPLATES)
    assert {t for t in TEMPLATES if t.startswith("T:")} == {
        "T:" + t for t in TEMPLATES if t.startswith(("I-", "E-")) and t != "I-more"
    }
    assert set(templates.SECTION_OF) <= set(TEMPLATES)
    assert set(templates.SECTION_OF.values()) == set(templates.SECTION_ORDER)
    assert templates.SECTION_ORDER[0] == "D" and templates.SECTION_ORDER[-1] == "END"


def test_validate_rejects_bad_templates() -> None:
    ok = Template("X", "body", "There {be} {n} {d}.", {"n": Slot("deaths", "total", "int")},
                  {"be": ("n", "be"), "d": ("n", "death")})  # fmt: skip

    def bad(**kw) -> None:
        t = ok._replace(**kw)
        with pytest.raises(ValueError):
            templates.validate({"X": t})

    templates.validate({"X": ok})
    bad(text="There {be} 5 {n} {d}.")  # a literal digit
    bad(text="There {be} {n}k {d}.")  # a suffix glued to a number
    bad(text="There {be} {n}{d}.")  # two slots touching
    bad(text="There {be} x{n} {d}.")  # a prefix glued
    bad(text="There {be} {n} {d} {zz}.")  # an undeclared token
    bad(text="There {be} {d}.")  # a declared slot not used
    bad(role="essay")
    bad(agree={"be": ("n", "be"), "d": ("m", "death")})  # a marker on an undeclared count
    bad(slots={"n": Slot("deaths", "total", "pct")})  # a marker on a non-integer slot
    bad(agree={"be": ("n", "be"), "d": ("n", "nonesuch")})
    # a colon after a slot and a parenthesis before one are allowed (the ticker and the coordinates)
    templates.validate({"X": ok._replace(text="Step {n}: ({be}, {d})")})


def test_every_template_text_has_no_digit_and_slot_boundaries() -> None:
    for t in TEMPLATES.values():
        bare = templates._TOKEN.sub("", t.text)
        assert not any(c.isdigit() for c in bare), t.id
        for m in templates._TOKEN.finditer(t.text):
            if m.start():
                assert t.text[m.start() - 1] in " (", (t.id, m.group(0))
            if m.end() < len(t.text):
                assert t.text[m.end()] in " ,.;:)", (t.id, m.group(0))
        assert t.role == "headline" or t.text.endswith("."), t.id
        assert t.role != "headline" or not t.text.endswith("."), t.id


def _values_for(t: Template, **over) -> dict:
    out = {}
    for name, sl in t.slots.items():
        out.update(
            {
                name: {"int": 5, "id": 7, "pct": 0.5, "change": 0.1, "lever": 0.5}.get(
                    sl.fmt, "the Braelix kin"
                )
            }
        )
    out.update(over)
    return out


COUNT_SLOTS = [
    (t.id, count) for t in TEMPLATES.values() for count in sorted({c for c, _ in t.agree.values()})
]


@pytest.mark.parametrize(("tid", "count"), COUNT_SLOTS)
def test_agreement_at_zero_one_two(tid: str, count: str) -> None:
    """Every count slot at 0, 1 and 2: the singular forms exactly at 1, never '1 {plural}'."""
    t = TEMPLATES[tid]
    for n in (0, 1, 2):
        text = templates.render(tid, _values_for(t, **{count: n}))
        for marker, (c, key) in t.agree.items():
            one, many = templates.AGREE_WORDS[key]
            want = one if (n if c == count else 5) == 1 else many
            assert re.search(rf"\b{want}\b", text), (tid, n, marker, text)
        if n == 1:
            plurals = "|".join(m for _, m in templates.AGREE_WORDS.values())
            # the count is followed by its own noun, which is singular
            assert not re.search(rf"(?<![\d,.])1 ({plurals})\b", text), (tid, text)
        if n == 2:
            assert not re.search(
                r"(?<![\d,.])2 (was|is|death|birth|member|agent|house|colony|step|setting)\b", text
            ), text


def test_agreement_phrases() -> None:
    def r(tid, **kw):
        return templates.render(tid, _values_for(TEMPLATES[tid], **kw))

    assert r("D-1", total=1, starved=1, killed=0, aged=0).startswith("There was 1 death: 1 by starvation,")
    assert r("D-1", total=2).startswith("There were 2 deaths:")
    assert r("B-2", births=1, immigrants=1) == "There was 1 birth and 1 arrival from outside."
    assert r("B-2", births=0, immigrants=3) == "There were 0 births and 3 arrivals from outside."
    assert r("C-1", tracked=1, largest_size=1) == "1 colony was tracked at step 5; it had 1 member."
    assert (
        r("C", tracked=2, largest_size=40) == "2 colonies were tracked at step 5; the largest had 40 members."
    )
    assert r("H-1", alive_to=1).startswith("1 house had living members at step")
    assert r("H-1ne", first_seen=1, extinct=1).endswith("; 1 was new and 1 died out.")
    assert r("H-1ne", first_seen=2, extinct=3).endswith("; 2 were new and 3 died out.")
    assert r("R", kills=1, attacks=12).endswith(
        "12 times; 1 of the attacks was on a victim that died in the same step."
    )
    assert r("R", kills=0, attacks=12).endswith(
        "12 times; 0 of the attacks were on a victim that died in the same step."
    )
    assert r("G-c", born=1).startswith("In the chapter, 1 agent was born into")
    assert r("HH", extinct=1) == "1 house died out"
    assert r("I-more", n=1) == "1 further setting is in place and is not listed."
    assert r("I-more", n=4) == "4 further settings are in place and are not listed."
    assert r("I-imm", count=1).endswith("queued 1 immigrant.")
    assert r("END", alive=0) == "The run stopped at step 5 with 0 agents alive."


def test_formats() -> None:
    assert templates.fmt_int(1506000) == "1,506,000"
    assert templates.fmt_pct(0.0866) == "8.7%"
    assert templates.fmt_change(0.276) == "+27.6%" and templates.fmt_change(-0.3863) == "-38.6%"
    assert templates.fmt_change(-0.0001) == "-0.0%"
    assert templates.fmt_lever(0.6000000238418579) == "0.6"
    assert (
        templates.fmt_lever(1.0) == "1"
        and templates.fmt_lever(0.25) == "0.25"
        and templates.fmt_lever(-0.0) == "0"
    )
    assert templates.fmt_lever(300.5) == "300.5"
    # ids are not counts: no thousands separator in a fallback name (B6)
    sheet = mk_sheet(ents=[("kin_group", 1830, None)])
    assert [e["display"] for e in sheet["entities"]] == ["kin group 1830"]


def test_source_words_and_rule_names() -> None:
    from alife.director import rules as dr
    from alife.director import yoked

    assert frozenset(dr._TAGS) | {yoked.RULE} == templates.RULE_NAMES
    assert templates.source_words("llm", None) == "the director's model"
    assert (
        templates.source_words("llm", "boredom") == "the director's model"
    )  # the model's entries carry no rule
    assert templates.source_words("rule", "runaway_damper") == "the director's rule runaway_damper"
    assert templates.source_words("rule", None) == "the director"  # an entry with no rule (N20)
    assert (
        templates.source_words("rule", "not_a_rule") == "the director"
    )  # an unknown identifier is never printed
    for name in templates.RULE_NAMES:
        assert re.fullmatch(r"[a-z]+(_[a-z]+)*", name)


def test_vocabulary_is_the_template_words() -> None:
    words = set()
    for t in TEMPLATES.values():
        words |= templates.template_words(t.text)
    assert words <= templates.TEMPLATE_VOCAB
    assert 100 < len(templates.TEMPLATE_VOCAB) < 160  # "about 130"
    for banned in ("after", "before", "since", "until", "following", "amid", "so", "then", "as", "because",
                   "despite", "recovered", "gathered", "fed", "guarded", "settled", "rested", "effect",
                   "probability", "mostly", "majority", "dominant", "raised", "cut"):  # fmt: skip
        assert banned not in templates.TEMPLATE_VOCAB


def test_section_and_salience_orders_are_pinned() -> None:
    assert templates.SECTION_ORDER == (
        "D", "P-low", "E", "G", "B", "LN", "TL", "H", "LB", "R", "CE", "C", "END",
    )  # fmt: skip
    assert templates.SALIENCE_ORDER == (
        "D", "P-low", "E", "G-top", "B", "LN", "TL", "H", "LB", "R", "G-rest", "CE", "C", "END",
    )  # fmt: skip


# --- the lexicon ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lst", "hits", "misses"),
    [
        ("V1", ["it caused this", "causing a fall", "because of it", "due to food", "as a result", "led to", "leaders",
                "it prevents", "helped", "the effect", "pushing", "responds"], ["cautious", "helmet", "line", "lineage"]),
        ("V2", ["it failed", "success", "succeeded", "needed", "required", "in order to", "to prevent a fall"],
         ["fair", "request"]),
        ("V3", ["suggests", "implied", "it means", "perhaps", "seems", "appeared"], ["seed", "meaningless"]),
        ("V4", ["wanted", "tried", "tries", "to protect", "rescued", "control", "planned", "attempts"],
         ["plain", "wasp"]),
        ("V5", ["I", "my", "you"], ["in", "is"]),
        ("V6", ["will fall", "should", "forecasts", "going to", "set to rise", "expected"], ["expert", "the setting"]),
        ("V7", ["species", "adapted", "evolving", "mutations", "a clan"], ["adopt", "clap"]),
        ("V8", ["massive", "the worst", "significantly", "last", "first", "total", "totals"], ["lasting", "firstly"]),
        ("T2", ["one", "twenty", "half", "twice", "threefold", "per cent", "thousands"], ["done", "none", "stone"]),
        ("L1", ["regrowth", "regenerating", "patches", "an immigrant", "catastrophic", "the director's"], ["pat", "pace"]),
        ("L2", ["Hunter", "hunting", "hunted", "settlers", "Unspecialised", "Tethering", "bonding"], ["hunger", "bond"]),
        ("G", ["dominant", "collapse", "extinction", "majority", "sharply"], ["domain", "colleague"]),
    ],
)  # fmt: skip
def test_stems_match_their_inflections(lst: str, hits: list[str], misses: list[str]) -> None:
    """A stem matches its inflections ('caus-': cause, causes, causing) and no word that only starts like it."""
    for text in hits:
        assert lexicon.scan(text, (lst,)), (lst, text)
    for text in misses:
        assert not lexicon.scan(text, (lst,)), (lst, text, lexicon.scan(text, (lst,)))


def test_lexicon_lists_are_data_and_pinned() -> None:
    assert set(lexicon.LEXICON) == {*lexicon.NEVER_LISTS, *lexicon.LEVER_LISTS, *lexicon.LABEL_LISTS}
    assert all(isinstance(t, str) and t for terms in lexicon.LEXICON.values() for t in terms)
    assert len(lexicon.LEXICON_DIGEST) == 64


def _bare(text: str) -> str:
    return templates._TOKEN.sub("X", text)


def test_r6_lists_are_absent_from_every_template() -> None:
    """The templates themselves never match the 'never' lists; lever words only in the block and the ticker
    (D-2's 'catastrophe' aside); label names in no template (they come through the label slot)."""
    for t in TEMPLATES.values():
        text = _bare(t.text)
        assert not lexicon.scan(text, lexicon.NEVER_LISTS), (t.id, lexicon.scan(text, lexicon.NEVER_LISTS))
        assert not lexicon.scan(text, lexicon.LABEL_LISTS), t.id
        lever = [w for _, w in lexicon.scan(text, lexicon.LEVER_LISTS)]
        if t.role not in templates.BLOCK_ROLES:
            assert lever == (["catastrophe"] if t.id == "D-2" else []), (t.id, lever)
    for words in (*templates.SOURCE_WORDS.values(), *templates.RANK_WORDS):
        assert not lexicon.scan(words.replace("{rule}", "x"), lexicon.NEVER_LISTS)
    # the agreement words
    for one, many in templates.AGREE_WORDS.values():
        assert not lexicon.scan(f"{one} {many}", lexicon.NEVER_LISTS + lexicon.LABEL_LISTS)


def test_r6check_every_rendered_string_of_a_full_sheet() -> None:
    """The lexicon over a sheet that triggers every template and the ticker: zero hits outside the
    exemptions (revision 2's LB-i contained the V4 stem 'intent-')."""
    sheet = full_sheet()
    body, heads, block, excluded = R.candidates(sheet)
    assert excluded == []
    seen = {c["template"] for c in body}
    assert seen >= {"D-2", "P-low", "E-split", "E-dissolved", "E-born", "E-merged", "CE-born", "G-r1", "G-r2",
                    "G-s", "G-c", "B-2", "LN", "TL", "H-1ne", "H-2", "LB-a", "R", "C", "END"}  # fmt: skip
    texts = [(c["template"], c["text"], False) for c in body]
    texts += [(h["template"], h["text"], False) for h in heads]
    texts += [(b["template"], b["text"], True) for b in block]
    texts += [("lead", R.lead_of(sheet)["text"], False)]
    texts += [("ticker", t["text"], True) for t in R.ticker(sheet)]
    assert len(texts) > 30
    for tid, text, is_block in texts:
        masked = R._mask(text, [(e["display"], e["name"] or str(e["id"])) for e in sheet["entities"]])
        if is_block:
            masked = R._mask_words(masked, sorted(templates.RULE_NAMES))
        assert not lexicon.scan(masked, lexicon.NEVER_LISTS), (tid, text)
        assert not lexicon.scan(masked, lexicon.LABEL_LISTS) or tid in ("LB-a", "LB-i"), (tid, text)
        lever = [w for _, w in lexicon.scan(masked, lexicon.LEVER_LISTS)]
        assert is_block or lever in ([], ["catastrophe"]), (tid, text, lever)


# --- the checker --------------------------------------------------------------------------------------------------


def _cands(sheet):
    body, heads, block, excluded = R.candidates(sheet)
    assert excluded == []
    return {c["template"]: c for c in body}, heads, block


def test_full_sheet_passes_every_rule() -> None:
    sheet = full_sheet()
    by, _heads, block = _cands(sheet)
    assert {"D-2", "P-low", "E-split", "E-dissolved", "E-born", "E-merged", "G-r1", "G-r2", "G-s", "G-c", "B-2", "LN",
            "TL", "H-1ne", "H-2", "LB-a", "R", "CE-born", "C", "END"} <= set(by)  # fmt: skip
    for c in by.values():
        assert R.check_string(c["text"], c, sheet) == [], c["text"]
    for b in block:
        assert R.check_string(b["text"], b, sheet, block=True) == [], b["text"]
    assert {b["template"] for b in block} >= {
        "S-gain", "I-regen", "X-regen-1", "I-loss", "X-gain", "I-seasons", "X-replaced", "I-drift", "X-open", "I-fert",
        "I-move", "I-imm", "I-cat",
    }  # fmt: skip


def _check(sheet, tid, facts, block=False):
    ctx = R._Ctx(sheet)
    c = R._inst(ctx, tid, facts)
    return R.check_string(c["text"], c, sheet, block, ctx=ctx), c


def test_r1_template_and_slots() -> None:
    sheet = full_sheet()
    by, _, _ = _cands(sheet)
    c = by["D-2"]
    assert R.check_string(c["text"], c, sheet) == []
    assert "R1:no template" in R.check_string(c["text"], None, sheet)
    assert any(
        r.startswith("R1:unknown template")
        for r in R.check_string(c["text"], {**c, "template": "D-9"}, sheet)
    )
    # other slots than the template's
    assert any(
        r.startswith("R1:slots differ")
        for r in R.check_string(c["text"], {**c, "slots": {"total": c["slots"]["total"]}}, sheet)
    )
    # a slot that reads another field, another kind of fact, a fact that does not exist
    wrong = {**c, "slots": {**c["slots"], "aged": [c["slots"]["aged"][0], "killed"]}}
    assert any("R1:slot aged" in r for r in R.check_string(c["text"], wrong, sheet))
    wrong = {**c, "slots": {**c["slots"], "aged": ["F1", "aged"]}}
    assert any("R1:slot aged" in r for r in R.check_string(c["text"], wrong, sheet))
    wrong = {**c, "slots": {**c["slots"], "aged": ["F999", "aged"]}}
    assert any("R1:slot aged" in r for r in R.check_string(c["text"], wrong, sheet))
    # the text is not what the template gives
    assert "R1:the text is not the template filled from the sheet" in R.check_string(
        c["text"].replace("5,060", "5,061"), c, sheet
    )
    # cites that are not the facts of the slots
    assert any("cites" in r for r in R.check_string(c["text"], {**c, "cites": ["F1"]}, sheet))
    # a block template used as a body string and the reverse
    b = next(x for x in R.candidates(sheet)[2] if x["template"] == "I-regen")
    assert "R1:block and body strings differ" in R.check_string(b["text"], b, sheet, block=False)
    assert "R1:block and body strings differ" in R.check_string(c["text"], c, sheet, block=True)


def test_r3_closed_vocabulary_and_name_masking() -> None:
    sheet = full_sheet()
    by, _, _ = _cands(sheet)
    c = by["G-r1"]
    assert R.check_string(c["text"], c, sheet) == []
    why = R.check_string(c["text"].replace("largest", "dominant"), c, sheet)
    assert "R3:dominant" in why
    # a word that is not in the vocabulary, glued to a number or to a name token, is found
    why = R.check_string("The Braelix kin recovered quickly.", None, sheet)
    assert "R3:recovered" in why and "R3:quickly" in why and "R3:braelix" not in why
    # only the name tokens are masked: 'the', 'kin' and 'House' are vocabulary, an invented name is not
    assert "R3:velmar" in R.check_string("The Velmar kin formed.", None, sheet)
    assert "R3:house" not in R.check_string("House Zaeld formed.", None, sheet)


def test_r4_number_tokens() -> None:
    sheet = full_sheet()
    by, _, _ = _cands(sheet)
    c = by["B-2"]
    assert "R4:the number tokens are not the rendered slots" in R.check_string(
        c["text"].replace("7 arrivals", "70 arrivals"), c, sheet
    )
    # numbers glued (the slot boundary) cannot be produced by a template; a text with them fails R1 and R4
    why = R.check_string("There were 5,0201,830 births.", None, sheet)
    assert "R1:no template" in why
    why = R.check_string("The population fell to 21,140k.", None, sheet)
    assert "R3:k" in why


def _gate(sheet, tid, **facts):
    """The R5 problems of template ``tid`` for the given facts (copies of the sheet's facts may be altered:
    the gates are the checker's own function of the facts a string cites)."""
    return R._gate_problems(R._Ctx(sheet), tid, facts)


def test_r5_gates_are_rederived_from_the_facts() -> None:
    sheet = full_sheet()
    ctx = R._Ctx(sheet)
    up = mk_sheet(kinds={"population": [pop(100, 130, 100)]})
    down = mk_sheet(kinds={"population": [pop(100, 60, 50)]})
    near = mk_sheet(kinds={"population": [pop(41_310, 41_312)]})
    zero = mk_sheet(kinds={"population": [pop(0, 5)]})
    flat = mk_sheet(kinds={"population": [pop(100, 100)]})
    assert _gate(up, "P-up", population=R._Ctx(up).one("population")) == []
    assert _gate(down, "P-up", population=R._Ctx(down).one("population"))  # 'rose' on a fall
    assert _gate(up, "P-down", population=R._Ctx(up).one("population"))
    assert _gate(near, "P-up", population=R._Ctx(near).one("population"))  # C7: never 'rose' for +0.0%
    assert _gate(near, "P-near", population=R._Ctx(near).one("population")) == []
    assert _gate(flat, "P-near", population=R._Ctx(flat).one("population"))
    assert _gate(zero, "P-zero", population=R._Ctx(zero).one("population")) == []
    assert _gate(up, "P-zero", population=R._Ctx(up).one("population"))
    assert _gate(flat, "P-flat", population=R._Ctx(flat).one("population")) == []
    assert _gate(up, "P-flat", population=R._Ctx(up).one("population"))
    assert _gate(up, "HP-down", population=R._Ctx(up).one("population"))
    # 'dipped' only for a deep dip (a minimum under 80% of both ends' lower one)
    crash = mk_sheet(kinds={"population": [pop(1000, 1000, 300)]})
    assert _gate(crash, "HP-dip", population=R._Ctx(crash).one("population")) == []
    assert _gate(up, "HP-dip", population=R._Ctx(up).one("population"))  # no minimum
    shallow = mk_sheet(kinds={"population": [pop(1000, 1000, 900)]})
    assert _gate(shallow, "HP-dip", population=R._Ctx(shallow).one("population"))
    # a colony that went on under a new id has neither colony sentence
    ev0 = {"kind": "event", "family": "colony", "event": "born", "system": 22, "other": 21, "size": 78,
           "overlap": 76, "s0": 2500, "s1": 2750, "reid": True}  # fmt: skip
    assert R._event_template(ev0) is None and R._event_template({**ev0, "event": "dissolved"}) is None
    assert R._event_template({**ev0, "reid": False}) == "CE-born"
    assert R._event_template({**ev0, "event": "split"}) == "CE-split"
    # 'lowest' only if the minimum is below both ends
    assert _gate(up, "P-low", population=R._Ctx(up).one("population"))
    # rank words need the carried rank
    g = {f["entity"]: f for f in ctx.kinds["group"]}
    assert _gate(sheet, "G-r1", group=g[1]) == []
    assert _gate(sheet, "G-r1", group=g[2])
    assert _gate(sheet, "G-r2", group=g[1])
    assert _gate(sheet, "G-r1", group=g[5])  # a group with no carried rank is never 'the largest'
    # the event verb equals the event's kind; overlap only on a born event with an origin
    ev = {(e["event"], e["system"]): e for e in ctx.kinds["event"]}
    assert _gate(sheet, "E-split", event=ev[("split", 2)]) == []
    assert _gate(sheet, "E-merged", event=ev[("split", 2)])
    assert _gate(sheet, "E-born", event=ev[("dissolved", 3)])
    assert _gate(sheet, "E-dissolved-o", event=ev[("dissolved", 3)])  # no 'other' to name
    assert _gate(sheet, "E-dissolved", event=ev[("born", 4)])
    assert _gate(sheet, "CE-born", event=ev[("born", 4)])  # a kin-group event is not a colony's
    # no headline for a re-identified group (HE); H-2 only for a strictly largest house; died out; new
    reid = {**ev[("born", 4)], "reid": True}
    assert _gate(sheet, "HE-born", event=reid)
    h = ctx.one("houses")
    assert _gate(sheet, "H-2", houses={**h, "largest_strict": False})
    assert _gate(sheet, "H-1ne", houses=h) == []
    assert _gate(sheet, "H-1ne", houses={**h, "first_seen": 0})  # 'new' with none new
    assert _gate(sheet, "H-1e", houses={**h, "extinct": 0})  # 'died out' with none
    assert _gate(sheet, "H-1", houses={**h, "exact": False})  # an inexact count without its clause
    assert _gate(sheet, "H-1x", houses=h)
    assert _gate(sheet, "HH", houses={**h, "exact": False})
    # deaths: D-1 has no catastrophe clause when there were catastrophe deaths
    d = ctx.one("deaths")
    assert _gate(sheet, "D-1", deaths=d)
    assert _gate(sheet, "D-2", deaths={**d, "catastrophe": 0, "total": d["total"] - d["catastrophe"]})
    # arrivals; colonies; the end of the run
    b = ctx.one("births")
    assert _gate(sheet, "B-1", births=b)
    assert _gate(sheet, "B-2", births={**b, "immigrants": 0})
    c = ctx.one("colonies")
    assert _gate(sheet, "C-1", colonies=c) and _gate(sheet, "C", colonies={**c, "tracked": 1})
    assert _gate(sheet, "END", run_end={**ctx.one("run_end"), "alive": 3})
    # a relation below the attack gate, or killing more than it attacked
    r = ctx.one("relation")
    assert _gate(sheet, "R", relation={**r, "attacks": 9})
    assert _gate(sheet, "R", relation={**r, "kills": r["attacks"] + 1})
    # labels: only a citable label of the sheet's key, with enough labelled agents
    lb = ctx.one("label")
    assert _gate(sheet, "LB-a", label=lb) == []
    assert _gate(sheet, "LB-i", label=lb)  # the family differs
    assert _gate(sheet, "LB-a", label={**lb, "key": "gatherer"})  # failed validity
    assert _gate(sheet, "LB-a", label={**lb, "key": "fleeing"})  # n_pos 79
    assert _gate(sheet, "LB-a", label={**lb, "key": "bonding"})  # margin 0.058
    assert _gate(sheet, "LB-a", label={**lb, "from_of": 99})


def test_r5_gates_of_the_block() -> None:
    sheet = full_sheet()
    ctx = R._Ctx(sheet)
    iv = {f["action"] + str(f["entry"]): f for f in ctx.kinds["intervention"]}
    regen, loss = iv["regen_pulse0"], iv["habitat_loss1"]
    assert _gate(sheet, "X-regen-1", intervention=regen) == []
    assert _gate(sheet, "X-regen", intervention=regen)  # restored 1.0 is 'its base rate'
    assert _gate(sheet, "X-gain", intervention=loss) == []
    assert _gate(sheet, "X-replaced", intervention=regen)  # it elapsed
    assert _gate(sheet, "X-open", intervention=regen)
    assert _gate(sheet, "I-loss", intervention=regen)  # the action is another
    assert _gate(sheet, "X-regen-1", intervention=iv["drift_change3"])
    assert _gate(sheet, "X-open", intervention=iv["immigration_pulse6"])  # a point has no end
    move = iv["fertility_shift5"]  # a patch move is a point too: no "still in place", no end clause at all
    assert _gate(sheet, "I-move", intervention=move) == []
    for x in ("X-open", "X-replaced", "X-gain", "X-gain-1"):
        assert _gate(
            sheet,
            x,
            intervention={
                **move,
                "end": A0 + 2000,
                "end_by": "replaced",
                "restored": {"gain": 1.0, "patch": 6},
            },
        )
    assert R._end_template({**move, "end": A0 + 2000, "end_by": "replaced", "open_to": None}) is None
    dropped = mk_sheet(kinds={"population": [pop(100, 100)]}, excluded={"interventions_dropped": 3})
    assert _gate(dropped, "I-more") == []
    assert _gate(sheet, "I-more")  # nothing was dropped
    st = ctx.kinds["standing"][0]
    assert _gate(sheet, "S-gain", standing=st) == []
    assert (
        "S-pos" not in TEMPLATES
    )  # a patch position is a point event: no standing line, no persistence claim
    assert _gate(sheet, "S-regen", standing=st)
    # W only if the sentences of windows are on, the window fact exists and lies inside the span and the chapter
    w = {
        "id": "F90",
        "kind": "window",
        "intervention": regen["id"],
        "entry": 0,
        "w0": A0 + 500,
        "w1": A0 + 1500,
        "from": 5,
        "to": 6,
    }
    assert any("window" in x for x in R._gate_problems(ctx, "W", {"window": w}))  # the rules have it off
    on = R.ReportRules(window_sentences=True)
    s2 = mk_sheet(rules=on, kinds={"intervention": [regen]})
    c2 = R._Ctx(s2, on)
    reg2 = c2.one("intervention")
    ok = {
        "id": "F90",
        "kind": "window",
        "intervention": reg2["id"],
        "w0": A0 + 500,
        "w1": A0 + 1500,
        "from": 5,
        "to": 6,
    }
    assert R._gate_problems(c2, "W", {"window": ok}) == []
    assert R._gate_problems(c2, "W", {"window": {**ok, "w1": A0 + 1501}})  # past the span's end
    assert R._gate_problems(c2, "W", {"window": {**ok, "w0": A0 + 400}})  # before it started


def test_r5_gated_words_need_their_template() -> None:
    sheet = full_sheet()
    by, _, _ = _cands(sheet)
    c = by["B-1" if "B-1" in by else "B-2"]
    # a gated word in a template that does not own it
    t = by["E-split"]
    text = t["text"].replace("split", "merged")
    why = R.check_string(text, t, sheet)
    assert any("merged" in r for r in why)
    assert any("R5" in r or "R1" in r for r in why)
    assert c  # the candidates exist


def test_r6_and_r7() -> None:
    sheet = full_sheet()
    by, _, _ = _cands(sheet)
    c = by["B-2"]
    # lever words outside the block are R6 hits (here injected into the text, so R1 fails too)
    assert any(r.startswith("R6:L1") for r in R.check_string("There was a catastrophe.", None, sheet))
    assert not any(
        r.startswith("R6:L1") for r in R.check_string("There was a catastrophe.", None, sheet, block=True)
    )
    assert any(
        r.startswith("R6:V1") for r in R.check_string("The population fell because of this.", None, sheet)
    )
    assert any(r.startswith("R6:L2") for r in R.check_string("The Hunter label matched.", None, sheet))
    assert any(r.startswith("R6:T2") for r in R.check_string("The population halved.", None, sheet))
    long = "There were " + "5 " * 130 + "births."
    assert any(r.startswith("R7") for r in R.check_string(long, None, sheet))
    h = "x" * 101
    assert any(r.startswith("R7") for r in R.check_string(h, {"template": "HP-up", "slots": {}}, sheet))
    assert c


def test_check_string_is_pure() -> None:
    sheet = full_sheet()
    before = json.dumps(sheet, sort_keys=True)
    by, _, _ = _cands(sheet)
    c = by["TL"]
    a = R.check_string(c["text"], c, sheet)
    b = R.check_string(c["text"], c, sheet)
    assert a == b == []
    assert json.dumps(sheet, sort_keys=True) == before
    assert R.check_string("The The.", None, sheet) == R.check_string("The The.", None, sheet)


# --- random sheets -----------------------------------------------------------------------------------------------


def test_2000_random_sheets_have_no_exclusion() -> None:
    """Every template that a random sheet reaches renders and passes the checker; none is excluded (a candidate
    excluded in a real build is a bug); the body is in the order of sections, with the lead first."""
    seen = set()
    for seed in range(2000):
        sheet = random_sheet(random.Random(seed))
        body, heads, block, excluded = R.candidates(sheet)
        assert excluded == [], (seed, excluded)
        lines = R._ticker_lines(sheet)
        seen |= (
            {c["template"] for c in body} | {h["template"] for h in heads} | {b["template"] for b in block}
        )
        seen |= {R.lead_of(sheet)["template"]} | {x["template"] for x in lines}
        for c in body:
            assert R.check_string(c["text"], c, sheet) == [], (seed, c["text"])
        for x in lines:
            assert R.check_string(x["text"], x, sheet, block=True) == [], (seed, x["text"])
        rep = R.template_report(sheet)
        assert rep["lead"] and rep["headline"]
        by_id = {c["id"]: c for c in body}
        secs = [templates.SECTION_ORDER.index(by_id[x["id"]]["section"]) for x in rep["body"]]
        assert secs == sorted(secs), seed
        assert len({x["id"] for x in rep["body"]}) == len(rep["body"])
        assert all(x["text"].startswith("Step ") for x in lines)
    missing = {t for t in TEMPLATES if t not in seen}
    # the window sentence is off; the rest of the catalogue is reached by the random sheets
    assert missing == {"W"}, missing


def test_templates_pin_digests() -> None:
    """Every table the stored records depend on is pinned: a change is a deliberate change of a version."""
    assert templates.TEMPLATE_DIGEST == "6d9d8a7b48238599ae81aa61d0a58aefd027309993f46f4790433c3f198a65dc"
    assert lexicon.LEXICON_DIGEST == "bdc398239891aa65cbbfb72106a8b6d80413de866a92ab3260dd2aa1cdee516b"
    assert R.REPORT_V1.digest() == "5382d0c0491ef8579aabcfc24a66b9ddff8329c972a005f8f1d6478302ed116d"
    assert R.label_table_digest() == "9d8b3f339aad261a5e1458511c1c4e1d7434cd838c50e06b3aba0c3b2e395abf"
    assert R.known_social_digest() == "4551a6aa56c3d6c8fdc1def7f8fd586c7da1a93d6476625ac4896bb892a22fd4"
    assert (R.REPORT_VERSION, R.CHECKER_VERSION, templates.TEMPLATE_VERSION, lexicon.LEXICON_VERSION) == (
        1,
        1,
        1,
        1,
    )
    # the label table's rules digest is the version-1 rule set's, and the social digest the default social rules'
    from alife.game.rules import RULES_V1
    from alife.game.social import SOCIAL_V1

    assert RULES_V1.digest() == "156ad4963b84ffe9636e9ba4fdd7592632c669bf6e490687d8eb2b38b9aec009"
    assert {k[1] for k in R.LABEL_VERDICTS} == {RULES_V1.digest()}
    assert set(R.KNOWN_SOCIAL) == {SOCIAL_V1.digest()} and R.KNOWN_SOCIAL[SOCIAL_V1.digest()] == (250, 16)


def test_deaths_without_a_catastrophe_use_d1() -> None:
    sheet = mk_sheet(kinds={"population": [pop(100, 90)], "deaths": [deaths(5, 1, 0, 0)]})
    body, _, _, ex = R.candidates(sheet)
    assert ex == [] and body[0]["template"] == "D-1" and body[0]["mandatory"]
    assert body[0]["text"] == "There were 6 deaths: 5 by starvation, 1 by attack and 0 by old age."
    sheet = mk_sheet(kinds={"population": [pop(100, 90)], "deaths": [deaths(0, 1, 0, 0)]})
    assert R.candidates(sheet)[0][0]["text"].startswith("There was 1 death: 0 by starvation, 1 by attack")
    # the rule identifier is masked in the block and refused in a body
    iv = interv(0, "regen_pulse", A0 + 500, {"mult": 0.5}, duration=5000)
    s = mk_sheet(kinds={"population": [pop(100, 90)], "intervention": [iv]})
    block = R.candidates(s)[2]
    assert (
        "runaway_damper" in block[0]["text"]
        and R.check_string(block[0]["text"], block[0], s, block=True) == []
    )


# --- the probe sheets, the crafted items and the corpus -----------------------------------------------------------


def probe_sheets() -> dict[str, dict]:
    return {p.name[:3]: json.loads(p.read_text()) for p in sorted((DATA / "report_probe").glob("*.json"))}


def test_the_committed_probe_sheets_are_what_the_maker_writes(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, str(DATA))
    try:
        import make_report_probe
    finally:
        sys.path.remove(str(DATA))
    for name, sheet in make_report_probe.sheets().items():
        committed = json.loads((DATA / "report_probe" / f"{name}.json").read_text())
        assert committed == json.loads(json.dumps(sheet, sort_keys=True)), name
    assert len(list((DATA / "report_probe").glob("*.json"))) == 12


def test_the_twelve_probe_sheets_have_no_exclusion() -> None:
    sheets = probe_sheets()
    assert len(sheets) == 12
    for name, sheet in sheets.items():
        body, heads, _block, excluded = R.candidates(sheet)
        assert excluded == [] and heads and body, name
        rep = R.template_report(sheet)
        assert rep["lead"] and rep["headline"]["text"] and len(rep["body"]) >= 1
        assert [t for t in R.ticker(sheet)], name
        assert R.facts_digest(sheet)  # JSON with no NaN
    # the probe's eventful sheet renders the frames of tracking.Event
    s02 = [c["text"] for c in R.candidates(sheets["s02"])[0]]
    assert (
        "Between step 1,513,000 and step 1,513,250, the Shaemund kin split from the Braelix kin with 1,210 members."
        in s02
    )
    assert ("Between step 1,517,000 and step 1,517,250, the Flauth kin merged into the Braelix kin; it had 520 members at "
            "step 1,517,000.") in s02  # fmt: skip


def test_the_crafted_items_are_rejected_and_none_is_a_candidate() -> None:
    """The 55 crafted false or misleading sentences of the two reviews (A1 to H7) and the two reply fixtures B1, B2, each
    on the probe sheet it was written for: none equals a string the report can produce, and each is rejected by
    `check_string` with R1 (it is no template) and the other rules its words break (pinned)."""
    sheets = probe_sheets()
    items = json.loads((DATA / "report_crafted.json").read_text(encoding="utf-8"))
    assert len(items) == 57 and sum(1 for i in items if i["id"][0] in "ACDEFGH") == 55
    assert {i["id"] for i in items if i["id"].startswith("B")} == {
        "B1",
        "B2",
    }  # the headline of B1 and the old template line
    produced: dict[str, set[str]] = {}
    for key, sheet in sheets.items():
        body, heads, block, _ = R.candidates(sheet)
        produced.update({key: {c["text"] for c in body} | {h["text"] for h in heads} | {b["text"] for b in block}
                          | {R.lead_of(sheet)["text"]} | {t["text"] for t in R.ticker(sheet)}})  # fmt: skip
    pinned_only_r1 = set()
    for it in items:
        sheet = sheets[it["sheet"]]
        text = it["text"]
        assert text not in produced[it["sheet"]], it["id"]
        assert text.rstrip(".") not in {p.rstrip(".") for p in produced[it["sheet"]]}, it["id"]
        why = R.check_string(text, None, sheet)
        cats = sorted({w.split(":")[0] + (":" + w.split(":")[1] if w.startswith("R6") else "") for w in why})
        assert "R1" in cats and cats == it["expect"], (it["id"], cats)
        if cats == ["R1"]:
            pinned_only_r1.add(it["id"])
        # claiming a template does not help: the text is not what any template gives
        for tid in ("P-up", "G-r1", "E-split", "D-1"):
            c = next((x for x in R.candidates(sheet)[0] if x["template"] == tid), None)
            if c is not None:
                assert R.check_string(text, c, sheet), (it["id"], tid)
    # the items whose words are all in the vocabulary are held by R1 alone (only a template renders a string)
    assert pinned_only_r1 == {"A9", "C1", "C11", "D13", "F1", "G1", "G2", "G3", "H1"}


def test_the_corpus_of_the_directors_journals_is_rejected() -> None:
    """The lexicon rejects every credit, presupposition, temporal and other causal sentence of the director's 131
    journals and each of the 61 reasons; the closed vocabulary rejects all of them (per-list counts pinned)."""
    corpus = json.loads((DATA / "report_corpus.json").read_text(encoding="utf-8"))
    assert corpus["journals"] == 131 and len(corpus["sentences"]) == 275 and len(corpus["reasons"]) == 61
    sheet = mk_sheet(kinds={"population": [pop(100, 100)]})
    by_class: dict[str, list[str]] = {}
    for s in corpus["sentences"]:
        by_class.setdefault(s["class"], []).append(s["text"])
    assert {k: len(v) for k, v in by_class.items()} == {
        "credit": 24,
        "presume": 24,
        "temporal": 10,
        "causal_other": 5,
        "other": 212,
    }
    for cls in ("credit", "presume", "temporal", "causal_other"):
        for text in by_class[cls]:
            assert lexicon.scan(text, lexicon.NEVER_LISTS), (cls, text)
    for text in corpus["reasons"]:
        assert lexicon.scan(text, lexicon.NEVER_LISTS), text
    everything = [s["text"] for s in corpus["sentences"]] + corpus["reasons"]
    for text in everything:
        why = R.check_string(text, None, sheet)
        assert "R1:no template" in why and any(w.startswith("R3:") for w in why), text
    counts = {name: 0 for name in lexicon.NEVER_LISTS}
    for text in everything:
        for name in {n for n, _ in lexicon.scan(text, lexicon.NEVER_LISTS)}:
            counts[name] += 1
    assert counts == {
        "T2": 39,
        "V1": 174,
        "V2": 122,
        "V3": 37,
        "V4": 84,
        "V5": 91,
        "V6": 88,
        "V7": 52,
        "V8": 120,
        "G": 116,
    }
    assert sum(1 for s in by_class["other"] if lexicon.scan(s, lexicon.NEVER_LISTS)) == 199
