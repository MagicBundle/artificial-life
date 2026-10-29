"""E24: the comm keys are not levers of the Director (DESIGN §26, §39).

The Director acts through a fixed menu on the environment (``State.env_*``) and never writes the config. The
comm keys (``[comm]``) are start-of-run config choices, static in the compiled step, and must stay out of its
reach: no menu parameter or env field carries one of their names, no director module refers to the comm
section, and the policy keys of a resume are disjoint from them.
"""

from __future__ import annotations

from pathlib import Path

from alife import config, director
from alife.director import actions
from alife.state import Env


def _keys() -> set[str]:
    return {k for _, k in config.FEATURE_KEYS["comm"]}


def test_no_menu_parameter_is_a_comm_key() -> None:
    params = {
        p for table in (actions.MENU, actions.GAME_SPECS) for spec in table.values() for p in spec.bounds
    }
    assert params and not (params & (_keys() - {"range", "cost"}))
    # ``range`` and ``cost`` are common words: the menu has none named so either
    assert not (params & {"range", "cost"})


def test_no_env_field_or_policy_key_is_a_comm_key() -> None:
    assert not (set(Env._fields) & _keys())
    assert not (set(config.DIRECTOR_POLICY_KEYS) & _keys())
    assert all(sec != "comm" for sec, _ in config.UNHASHED_KEYS)  # all hashed


def test_the_director_modules_never_touch_the_comm_section() -> None:
    root = Path(director.__file__).parent
    needles = ("cfg.comm", ".comm.", "CommConfig", "comm.control", "comm.enabled")
    hits = [p.name for p in root.rglob("*.py") if any(n in p.read_text() for n in needles)]
    assert hits == [], hits
