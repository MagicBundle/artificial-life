"""Offline analysis of finished runs (host-only; never imported by the simulation or the step).

- ``mi``: mutual-information estimators with shuffle nulls.
- ``resim``: re-simulate a run in memory to steps it no longer has checkpoints for.
- ``signals``: information carried by emitted chemicals and messages (docs/SIGNALS.md).
- ``labels``: the validity study of the game labels, AUC against matched random labels (docs/GAMES.md §4.5).
- ``phylo``: the phylogeny rebuilt from the event logs, lineage labels, the neutral and permuted shadows.
- ``houses``: the exact houses of a planet (a fork of an evolved checkpoint), from its source's births.
- ``activity``: Bedau–Packard evolutionary-activity statistics on lineage components (docs/ACTIVITY.md).
- ``report``: the ``alife activity`` report: classification, threshold sweeps, ``summary.json``.
- ``plots``: its figures (matplotlib, the optional ``analysis`` extra).
"""
