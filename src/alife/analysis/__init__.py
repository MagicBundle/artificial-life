"""Offline analysis of finished runs (host-only; never imported by the simulation or the step).

- ``mi``: mutual-information estimators with shuffle nulls.
- ``resim``: re-simulate a run in memory to steps it no longer has checkpoints for.
- ``signals``: information carried by emitted chemicals and messages (docs/SIGNALS.md).
- ``phylo``: the phylogeny rebuilt from the event logs, lineage labels, the neutral and permuted shadows.
- ``activity``: Bedau–Packard evolutionary-activity statistics on lineage components (docs/ACTIVITY.md).
- ``report``: the ``alife activity`` report: classification, threshold sweeps, ``summary.json``.
- ``plots``: its figures (matplotlib, the optional ``analysis`` extra).
"""
