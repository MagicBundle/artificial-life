"""The rule tier: safety rules, personas, and the pacing folds (budget, cooldown, active effects, drama)."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from alife.config import Config, apply_overrides
from alife.director import actions, pacing, rules
from alife.director.actions import EnvView, validate_entry
from alife.director.rules import ARCHITECT, GARDENER, RULE_PARAMS, TRICKSTER, Context, decide
from alife.director.vitals import Vitals, aggregate
from test_director_vitals import row

PP = 6
P = RULE_PARAMS
K = 1000


def vitals(t: int, **kw: Any) -> Vitals:
    base: dict[str, Any] = dict(
        step=t,
        origin=max(0, t - 10_000),
        window=aggregate([row(t, steps=K)]),
        n_alive=900.0,
        n_alive_min=880.0,
        k_est=1000.0,
        n_over_k=0.9,
        fall=0.01,
        extinction_risk=False,
        pop_frac_cap=0.4,
        largest_body_frac=0.01,
        top_clade_share=0.3,
        mono_since=None,
        novelty=0.05,
        novelty_trace=(),
        coverage=12,
        new_cells_200k=2,
        last_new_cell=t - K,
        b_shift=0.05,
    )
    return Vitals(**{**base, **kw})


def env(**kw: Any) -> EnvView:
    pos = np.tile(np.array([[64.0, 64.0]], np.float32), (PP, 1))
    base: dict[str, Any] = dict(
        regen_mult=1.0,
        season_amp=0.0,
        season_period=0,
        drift_mult=1.0,
        imm_request=0,
        patch_gain=np.ones(PP, np.float32),
        patch_pos=pos,
    )
    return EnvView(**{**base, **kw})


def fe(step: int, action: str, params: dict | None = None, duration: int | None = None, **kw: Any) -> dict:
    """A forward log entry, with the fields the folds read."""
    default = {
        "regen_pulse": {"mult": 1.2},
        "habitat_loss": {"patch": 0, "gain": 0.0},
        "immigration_pulse": {"count": 32},
        "set_seasons": {"period": 5000, "amp": 0.3},
        "fertility_shift": {"patch": 1, "gain": 1.2},
        "catastrophe": {"center": [1.0, 1.0], "radius": 30.0, "kill_p": 0.5},
        "drift_change": {"mult": 2.0},
    }[action]
    prev = (
        {"mult": 1.0}
        if action == "regen_pulse"
        else {"patch": (params or default).get("patch", 0), "gain": 1.0}
    )
    return {
        "step_applied": step,
        "source": "rule",
        "action": action,
        "params": params or default,
        "prev": prev,
        "duration": duration,
        **kw,
    }


def rv(e: dict, step: int) -> dict:
    return {**e, "step_applied": step, "source": "revert", "params": dict(e["prev"]), "duration": None}


def ctx(
    t: int = 300_000, persona: str = "gardener", entries: tuple = (), seed: int = 7, **kw: Any
) -> Context:
    v = kw.pop("v", None) or vitals(t, **kw.pop("vit", {}))
    return Context(
        t=t,
        seed=seed,
        persona=persona,
        budget=rules.effective_budget(0, P.persona(persona)),
        world_size=128.0,
        vitals=v,
        pacing=pacing.fold(entries, t),
        env=kw.pop("env", None) or env(),
    )


def with_gardener(**kw: Any) -> rules.RuleParams:
    """RuleParams whose Gardener is edited (the rules look the persona up by name)."""
    return dataclasses.replace(P, personas=(dataclasses.replace(GARDENER, **kw), TRICKSTER, ARCHITECT))


def over(e: dict) -> tuple[dict, dict]:
    """A timed entry and the revert logged at its end."""
    return e, rv(e, e["step_applied"] + e["duration"])


# --- pacing folds ---


def test_fold_reads_only_entries_up_to_t() -> None:
    entries = [fe(1000, "regen_pulse", duration=5000), fe(7000, "immigration_pulse"), fe(9000, "catastrophe")]
    early = pacing.fold(entries, 6000)
    assert early.last_step == 1000 and early.last_action == "regen_pulse"
    assert pacing.budget_used(early, 6000) == 1
    full = pacing.fold(entries, 9000)
    assert full.last_action == "catastrophe" and pacing.budget_used(full, 9000) == 1 + 1 + 3
    assert pacing.fold(entries[:2], 7000) == pacing.fold(entries, 7000)  # later entries change nothing


def test_budget_is_a_rolling_window_over_step_applied_with_free_reverts() -> None:
    reg = fe(0, "regen_pulse", duration=5000)
    entries = [reg, fe(50_000, "habitat_loss", duration=10_000), rv(reg, 5000)]
    p = pacing.fold(entries, 100_000)
    assert pacing.budget_used(p, 99_999) == 1 + 2 and pacing.budget_used(p, 100_000) == 2  # step 0 just left
    assert pacing.budget_used(p, 150_000) == 0 and pacing.budget_used(p, 149_999) == 2  # 50k + 100k window
    assert pacing.budget_used(pacing.fold([rv(reg, 5000)], 5000), 5000) == 0  # a revert costs nothing


def test_cooldown_counts_from_the_end_of_the_effect() -> None:
    p = pacing.fold([fe(1000, "regen_pulse", duration=10_000)], 100_000)
    assert p.ends["regen_pulse"] == 11_000
    assert not pacing.cooldown_ready(p, "regen_pulse", 11_000 + 29_999, 30_000)
    assert pacing.cooldown_ready(p, "regen_pulse", 11_000 + 30_000, 30_000)
    assert pacing.cooldown_ready(p, "catastrophe", 0, 100_000)  # never taken
    inst = pacing.fold([fe(1000, "immigration_pulse")], 100_000)
    assert pacing.cooldown_ready(
        inst, "immigration_pulse", 1000 + 20_000, 20_000
    )  # an instant action ends at once


def test_active_effects_one_per_lever_with_supersede_and_revert() -> None:
    a = fe(1000, "regen_pulse", {"mult": 0.8}, 10_000)
    b = {
        **fe(5000, "regen_pulse", {"mult": 1.5}, 20_000),
        "prev": {"mult": 1.0},
    }  # supersedes a, base restored
    h = fe(2000, "habitat_loss", {"patch": 3, "gain": 0.0}, 10_000)
    p = pacing.fold([a, h, b], 6000)
    assert set(p.active) == {"regen", "patch:3"}
    assert p.active["regen"].end == 25_000 and p.active["regen"].restore == {"mult": 1.0}
    assert [e.key for e in pacing.due_reverts(p, 12_000)] == ["patch:3"]  # a's end passed but b replaced it
    assert [e.key for e in pacing.due_reverts(p, 25_000)] == ["patch:3", "regen"]
    done = pacing.fold([a, h, b, rv(h, 12_000)], 12_000)
    assert set(done.active) == {"regen"}


def test_drama_decays_with_its_half_life() -> None:
    p = pacing.fold([fe(0, "catastrophe")], 0)
    assert pacing.drama(p, 0, 50_000) == 3.0
    assert pacing.drama(p, 50_000, 50_000) == pytest.approx(1.5)
    assert pacing.drama(p, 100_000, 50_000) == pytest.approx(0.75)
    assert pacing.steps_since_last(p, 1234) == 1234 and pacing.steps_since_last(pacing.fold([], 5), 5) is None


# --- budget and personas ---


def test_effective_budget_defaults_to_the_persona_and_is_bounded_by_it() -> None:
    assert [rules.effective_budget(0, P.persona(n)) for n in ("gardener", "trickster", "architect")] == [
        2,
        6,
        4,
    ]
    assert (
        rules.effective_budget(3, GARDENER) == 3
        and rules.effective_budget(99, GARDENER) == GARDENER.budget_max
    )
    assert rules.effective_budget(99, TRICKSTER) == TRICKSTER.budget_max


# --- safety rules ---


def test_a_healthy_world_gets_no_decision() -> None:
    for persona in ("gardener", "trickster", "architect"):
        d = decide(ctx(persona=persona))
        assert d.proposals == () and d.notes == ()


def test_extinction_rescue_boosts_regeneration_and_brings_immigrants() -> None:
    d = decide(ctx(vit=dict(extinction_risk=True, n_over_k=0.1, fall=0.5)))
    assert [(p.action, p.rule) for p in d.proposals] == [
        ("regen_pulse", "extinction_rescue"),
        ("immigration_pulse", "extinction_rescue"),
    ]
    reg, imm = d.proposals
    assert reg.raw == {"mult": 1.5} and reg.duration == 20_000
    assert imm.raw == {"count": 100} and "override" in reg.reason  # 0.1 * K_est


def test_the_rescue_reason_names_the_ratio_the_flag_read_and_its_size_reads_k_est() -> None:
    """2D: the reason is as it has always been ("n/K_est"). 3D (vitals carry the floor layer's K, §27.10): the
    flag was set by ``n / K_floor``, so the reason names that ratio; the rescue's size is still ``0.1·K_est``
    of the occupancy-weighted estimate, which every other consumer keeps."""
    d2 = decide(ctx(vit=dict(extinction_risk=True, n_over_k=0.1, fall=0.05)))
    assert d2.proposals[0].reason == "n/K_est 0.1, fall 0.05 (override)"
    assert d2.proposals[1].raw == {"count": 100}
    d3 = decide(
        ctx(vit=dict(extinction_risk=True, n_over_k=0.5, k_floor=300.0, n_over_k_floor=0.15, fall=0.05))
    )
    assert [p.action for p in d3.proposals] == ["regen_pulse", "immigration_pulse"]
    assert all(p.reason == "n/K_floor 0.15, fall 0.05 (override)" for p in d3.proposals)
    assert d3.proposals[1].raw == {"count": 100}  # 0.1 of k_est (1 000), not of the floor layer's 300


def test_a_blocked_3d_rescue_is_noted_with_the_floor_layer_ratio() -> None:
    t = 600_000
    spent = (*over(fe(t - 20_000, "habitat_loss", duration=10_000)), fe(t - 15_000, "catastrophe"))
    rescue = fe(t - 50_000, "regen_pulse", duration=20_000, rule="extinction_rescue")
    risk = dict(extinction_risk=True, n_over_k=0.5, k_floor=300.0, n_over_k_floor=0.15, fall=0.01)
    again = decide(ctx(t, entries=(*spent, *over(rescue)), vit=risk))  # the override is used up
    assert again.proposals == () and "blocked" in again.notes[0]
    assert again.notes[0].startswith("extinction_rescue (n/K_floor 0.15, fall 0.01): blocked [")


def test_extinction_rescue_overrides_budget_and_cooldown_at_most_once_per_200k() -> None:
    t = 600_000
    spent = (
        *over(fe(t - 20_000, "habitat_loss", duration=10_000)),
        fe(t - 15_000, "catastrophe"),
    )  # 5/2 used
    risk = dict(extinction_risk=True, n_over_k=0.1)
    first = decide(ctx(t, entries=spent, vit=risk))
    assert [p.action for p in first.proposals] == ["regen_pulse", "immigration_pulse"]  # budget waived
    rescue = fe(t - 50_000, "regen_pulse", duration=20_000, rule="extinction_rescue")
    again = decide(ctx(t, entries=(*spent, *over(rescue)), vit=risk))
    assert again.proposals == () and "blocked" in again.notes[0]  # the override is used up: budget applies
    old = fe(t - 200_000, "regen_pulse", duration=20_000, rule="extinction_rescue")
    assert len(decide(ctx(t, entries=(*over(old), *spent), vit=risk)).proposals) == 2  # it may override again
    # Within 200k but with budget and cooldowns free, an ordinary rescue still goes ahead.
    ordinary = fe(t - 150_000, "regen_pulse", duration=20_000, rule="extinction_rescue")
    assert len(decide(ctx(t, entries=over(ordinary), vit=risk)).proposals) == 2


def test_a_rescue_supersedes_an_active_pulse_but_an_ordinary_response_does_not() -> None:
    t = 600_000
    damper = fe(t - 5_000, "regen_pulse", {"mult": 0.8}, 10_000, rule="runaway_damper")
    risk = dict(extinction_risk=True, n_over_k=0.1)
    d = decide(ctx(t, entries=(damper,), vit=risk))
    assert [p.action for p in d.proposals] == [
        "regen_pulse",
        "immigration_pulse",
    ]  # the override takes the lever
    blocked = decide(ctx(t, entries=(damper,), vit=dict(pop_frac_cap=0.99)))
    assert blocked.proposals == () and "runaway_damper" in blocked.notes[0]


def test_runaway_damper_is_a_mild_regen_pulse_below_one() -> None:
    for vit in (dict(pop_frac_cap=0.96), dict(largest_body_frac=0.06)):
        d = decide(ctx(vit=vit))
        (p,) = d.proposals
        assert (p.action, p.rule) == ("regen_pulse", "runaway_damper")
        assert p.raw["mult"] < 1.0 and p.duration == P.damper_duration
    assert decide(ctx(vit=dict(pop_frac_cap=0.95, largest_body_frac=0.05))).proposals == ()  # strictly above


def test_monoculture_breaker_needs_the_share_sustained_for_the_persona_window() -> None:
    t = 400_000
    streak = dict(top_clade_share=0.95, mono_since=t - 90_000)
    assert decide(ctx(t, vit=streak)).proposals == ()  # 90k < 100k
    d = decide(ctx(t, vit={**streak, "mono_since": t - 100_000}))
    (p,) = d.proposals
    assert p.rule == "monoculture_breaker" and p.action == "immigration_pulse"  # the Gardener's top weight
    # the Trickster's windows are scaled by 0.5, so 60k is enough for it
    assert decide(ctx(t, "trickster", vit={**streak, "mono_since": t - 60_000})).proposals != ()


def test_monoculture_breaker_rearms_after_acting() -> None:
    t = 400_000
    vit = dict(top_clade_share=0.95, mono_since=t - 300_000)
    fired = (fe(t - 60_000, "immigration_pulse", rule="monoculture_breaker"),)
    assert decide(ctx(t, entries=fired, vit=vit)).proposals == ()
    old = (fe(t - 110_000, "immigration_pulse", rule="monoculture_breaker"),)
    assert decide(ctx(t, entries=old, vit=vit)).proposals != ()


def trace(t: int, novelty: float, n: int = 40, span: int = 200_000) -> tuple:
    return tuple((t - span + i * span // n, novelty) for i in range(1, n + 1))


def test_stagnation_needs_low_novelty_and_no_new_cell_over_the_whole_window() -> None:
    t = 600_000
    calm = dict(novelty_trace=trace(t, 0.01), last_new_cell=t - 210_000, origin=0)
    d = decide(ctx(t, vit=calm))
    (p,) = d.proposals
    assert p.rule == "stagnation_response" and p.action == "immigration_pulse"
    assert decide(ctx(t, vit={**calm, "last_new_cell": t - 150_000})).proposals == ()  # a new cell
    assert decide(ctx(t, vit={**calm, "novelty_trace": trace(t, 0.05)})).proposals == ()  # novelty is not low
    assert (
        decide(ctx(t, vit={**calm, "origin": t - 100_000})).proposals == ()
    )  # not observed for a window yet
    assert decide(ctx(t, vit={**calm, "novelty_trace": ()})).proposals == ()
    fired = (fe(t - 100_000, "immigration_pulse", rule="stagnation_response"),)
    assert decide(ctx(t, entries=fired, vit=calm)).proposals == ()  # re-arms one window after acting


def test_the_gardener_takes_the_heaviest_eligible_response_and_never_a_blocked_one() -> None:
    t = 600_000
    calm = dict(novelty_trace=trace(t, 0.01), last_new_cell=None, origin=0)
    recent_imm = (
        fe(t - 60_000, "immigration_pulse"),
    )  # cooling down for 1.5 * 20k only until t - 30k: ready
    assert decide(ctx(t, entries=recent_imm, vit=calm)).proposals[0].action == "immigration_pulse"
    cooling = (fe(t - 10_000, "immigration_pulse"),)  # within the cooldown, but also within min_gap
    assert decide(ctx(t, entries=cooling, vit=calm)).proposals == ()
    assert "only 10000 steps" in decide(ctx(t, entries=cooling, vit=calm)).notes[0]
    far = (fe(t - 55_000, "immigration_pulse"),)  # past min_gap 50k, cooldown 30k over: ready again
    assert decide(ctx(t, entries=far, vit=calm)).proposals[0].action == "immigration_pulse"
    # With immigration queue full the next-heaviest response (seasons or a shift, weight 2) is taken.
    full = env(imm_request=1020)
    assert decide(ctx(t, vit=calm, env=full)).proposals[0].action in ("set_seasons", "fertility_shift")


def test_gardener_and_architect_are_deterministic_and_respect_their_budget() -> None:
    t = 600_000
    calm = dict(novelty_trace=trace(t, 0.01), last_new_cell=None, origin=0)
    for persona in ("gardener", "architect"):
        a, b = decide(ctx(t, persona, vit=calm)), decide(ctx(t, persona, vit=calm, seed=7))
        assert a == b and len(a.proposals) == 1
    # budget: the Gardener has 2 units; two recent units leave no room for a 1-unit response
    spent = (fe(t - 60_000, "catastrophe"),)  # 3 units within 100k, past min_gap
    d = decide(ctx(t, entries=spent, vit=calm))
    assert d.proposals == () and "budget" in d.notes[0]


def test_the_trickster_samples_with_a_seed_and_never_repeats_its_last_action() -> None:
    calm = lambda t: dict(novelty_trace=trace(t, 0.01), last_new_cell=None, origin=0)  # noqa: E731
    seen = set()
    for i in range(60):
        t = 400_000 + 1000 * i
        d = decide(ctx(t, "trickster", seed=i, vit=calm(t)))
        assert len(d.proposals) == 1
        seen.add(d.proposals[0].action)
        last = "set_seasons"
        e = (fe(t - 60_000, last),)
        d2 = decide(ctx(t, "trickster", entries=e, seed=i, vit=calm(t)))
        assert d2.proposals and d2.proposals[0].action != last
    assert len(seen) >= 4  # varied shocks
    same = [decide(ctx(500_000, "trickster", seed=5, vit=calm(500_000))) for _ in range(3)]
    assert same[0] == same[1] == same[2]  # stateless seeding: same inputs, same decision
    other = {
        decide(ctx(500_000, "trickster", seed=s, vit=calm(500_000))).proposals[0].action for s in range(30)
    }
    assert len(other) > 1


def test_at_most_one_catastrophe_per_100k_whatever_the_persona() -> None:
    # A Gardener variant that halves cooldowns (50k) and wants nothing but catastrophes.
    only_cat = with_gardener(
        stagnation=(("catastrophe", 1.0),),
        min_gap=0,
        drama_max=99.0,
        cooldown_scale=0.5,
        budget=12,
        budget_max=12,
    )
    t = 700_000
    calm = dict(novelty_trace=trace(t, 0.01), last_new_cell=None, origin=0)

    def go(entries: tuple) -> Any:
        return decide(ctx(t, entries=entries, vit=calm)._replace(budget=12), only_cat)

    assert go(()).proposals[0].action == "catastrophe"
    assert go((fe(t - 60_000, "catastrophe"),)).proposals == ()  # past the scaled cooldown, inside the cap
    assert go((fe(t - 100_000, "catastrophe"),)).proposals[0].action == "catastrophe"


def test_the_trickster_gets_bored_the_gardener_does_not() -> None:
    t = 200_000
    quiet = dict(origin=0)
    assert decide(ctx(t, "trickster", vit=quiet)).proposals != ()  # 200k quiet > idle_gap 150k
    assert decide(ctx(120_000, "trickster", vit=quiet)).proposals == ()
    assert decide(ctx(t, "gardener", vit=quiet)).proposals == ()
    assert decide(ctx(t, "architect", vit=quiet)).proposals == ()
    just_acted = (fe(t - 40_000, "drift_change"),)
    assert decide(ctx(t, "trickster", entries=just_acted, vit=quiet)).proposals == ()


def test_discretionary_responses_wait_for_min_gap_and_a_low_drama_level() -> None:
    t = 600_000
    calm = dict(novelty_trace=trace(t, 0.01), last_new_cell=None, origin=0)
    hot = tuple(fe(t - 51_000 + i, "drift_change") for i in range(10))  # far too much recent drama
    assert decide(ctx(t, entries=hot, vit=calm)).proposals == ()
    safe = decide(ctx(t, entries=hot, vit=dict(pop_frac_cap=0.99)))
    assert (
        safe.proposals == () and "regen_pulse is cooling" not in safe.notes[0]
    )  # the damper is not discretionary


@pytest.mark.parametrize("persona", ["gardener", "trickster", "architect"])
def test_every_response_clamps_and_validates_as_a_menu_entry(tiny_cfg: Config, persona: str) -> None:
    """Whatever the rules propose is something ``clamp`` accepts and ``validate_entry`` passes."""
    cfg = apply_overrides(
        tiny_cfg, ["director.enabled=true", "director.rule_every=1000", "fields.n_patches=6"]
    )
    view = env()
    free = dataclasses.replace(
        P,
        personas=tuple(
            dataclasses.replace(
                p, min_gap=0, drama_max=1e9, budget_max=99, budget=99, cooldown_scale=0.0, idle_gap=1
            )
            for p in P.personas
        ),
    )
    actions_seen = set()
    for seed in range(40):
        t = 400_000 + 1000 * seed
        calm = dict(
            novelty_trace=trace(t, 0.01), last_new_cell=None, origin=0, mono_since=0, top_clade_share=0.95
        )
        c = ctx(t, persona, seed=seed, vit=calm, env=view)._replace(budget=99)
        for prop in decide(c, free).proposals:
            cl = actions.clamp(prop.action, prop.raw, prop.duration, view, cfg)
            e = {"step_applied": t, "source": "rule", "action": prop.action, **cl._asdict()}
            validate_entry(e, cfg, 1000)
            actions_seen.add(prop.action)
    assert actions_seen  # and the deterministic personas keep to their one favourite
    if persona != "trickster":
        assert len(actions_seen) <= 2


def test_rule_params_and_version_are_recorded() -> None:
    assert rules.RULES_VERSION == 1 and isinstance(P, rules.RuleParams)
    assert {p.name for p in P.personas} == {"gardener", "trickster", "architect"}
    assert P.persona("architect") is ARCHITECT
    with pytest.raises(StopIteration):
        P.persona("nobody")
