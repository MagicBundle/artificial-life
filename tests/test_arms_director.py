"""E24: the arms keys are not levers of the Director (DESIGN §26, §42.5).

The Director acts through a fixed menu on the environment (``State.env_*``: the field gain, patches, the kill
mask, immigration) and never writes the config. The arms keys are start-of-run config choices (static, in the
compiled step) and must stay out of its reach: no menu parameter or env field carries an arms key's name, no
director module refers to the arms section, and the policy keys of a resume are disjoint from them.
"""

from __future__ import annotations

from pathlib import Path

from alife import config, director
from alife.director import actions
from alife.state import Env


def _arms_keys() -> set[str]:
    return {k for _, k in config.FEATURE_KEYS["arms"]}


def test_no_menu_parameter_is_an_arms_key() -> None:
    params = {
        p for table in (actions.MENU, actions.GAME_SPECS) for spec in table.values() for p in spec.bounds
    }
    assert params and not (params & (_arms_keys() - {"enabled"}))
    assert "enabled" not in params


def test_no_env_field_or_policy_key_is_an_arms_key() -> None:
    assert not (set(Env._fields) & _arms_keys())
    assert not (set(config.DIRECTOR_POLICY_KEYS) & _arms_keys())
    assert all(sec != "arms" for sec, _ in config.UNHASHED_KEYS)  # arms keys are all hashed


def test_the_director_modules_never_touch_the_arms_section() -> None:
    root = Path(director.__file__).parent
    # "arms" also names the evaluation arms (A, B, C, R) in the director's own prose: look for the section
    needles = ("cfg.arms", ".arms.", "ArmsConfig", '"arms"', "from alife import arms", "alife.arms")
    hits = [p.name for p in root.rglob("*.py") if any(n in p.read_text() for n in needles)]
    assert hits == [], hits
