"""Write ``tests/data/report_run_synth/`` (the synthetic observed run of the world report's tests): the files
``tests/report_synth.write_run`` writes with its defaults. Run from the repository root:

    PYTHONPATH=tests python tests/data/make_report_run_synth.py [OUT_DIR]

No simulation and no MLX computation (``alife.config`` is imported for ``save_config``); the files are
deterministic, and a test pins that the committed ones equal a fresh write."""

from __future__ import annotations

import sys
from pathlib import Path

import report_synth

if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "report_run_synth"
    report_synth.write_run(out)
    print(f"wrote {out}")
