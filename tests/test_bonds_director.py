"""E24: the bonds keys are not levers of the Director (DESIGN §26, §40).

The Director acts through a fixed menu on the environment (``State.env_*``) and never writes the config. The
bonds keys (``bonds.share_*`` and ``[roles]``) are start-of-run config choices, static in the compiled step,
and must stay out of its reach: no menu parameter or env field carries one of their names, no director module
refers to the roles section or the new bond keys, and the policy keys of a resume are disjoint from them.
"""

from __future__ import annotations

from pathlib import Path

from alife import config, director
from alife.director import actions
from alife.state import Env


def _keys() -> set[str]:
    return {k for _, k in config.FEATURE_KEYS["bonds"]}


def test_no_menu_parameter_is_a_bonds_key() -> None:
    params = {
        p for table in (actions.MENU, actions.GAME_SPECS) for spec in table.values() for p in spec.bounds
    }
    assert params and not (params & _keys())


def test_no_env_field_or_policy_key_is_a_bonds_key() -> None:
    assert not (set(Env._fields) & _keys())
    assert not (set(config.DIRECTOR_POLICY_KEYS) & _keys())
    assert all(sec != "roles" and k not in _keys() for sec, k in config.UNHASHED_KEYS)  # all hashed


def test_the_director_modules_never_touch_the_new_bond_keys() -> None:
    root = Path(director.__file__).parent
    needles = ("cfg.roles", ".roles.", "RolesConfig", "share_mode", "share_kappa", "share_kin_floor")
    hits = [p.name for p in root.rglob("*.py") if any(n in p.read_text() for n in needles)]
    assert hits == [], hits
