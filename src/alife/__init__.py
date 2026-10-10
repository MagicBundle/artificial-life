"""alife: a neural artificial-life engine on Apple MLX (see docs/DESIGN.md)."""

from alife.config import Config, load_config
from alife.state import State, init_state
from alife.step import make_step

__version__ = "0.1.0"

__all__ = ["Config", "State", "__version__", "init_state", "load_config", "make_step"]
