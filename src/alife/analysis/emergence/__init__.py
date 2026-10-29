"""The emergence measurement suite (DESIGN §44): host-side, observer-only, numpy.

Run it with ``python -m alife.analysis.emergence <command>`` (``record``, ``panel``, ``compare``, ``prereg``,
``scorecards``, ``synth``; ``epr``, ``braid``, ``winding``, ``modes`` and ``activity`` land with their
blocks). Only ``record.py`` touches MLX; nothing here changes the simulation, a run directory or a checkpoint.
"""
