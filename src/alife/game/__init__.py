"""The games vertical (docs/GAMES.md, DESIGN §28): build step G1, the observer layer.

``observe`` is the pure-MLX observer state and its per-step update, ``rules`` the archetype and intent rules
(versioned by ``INTENT_VERSION``) over the accumulators. Both read only the step's observer channel
(``step.Acts``) and a few ``State`` fields, never write the ``State`` and never read its RNG key: observing
leaves the trajectory bitwise unchanged. This package is imported explicitly (``alife.game.observe``,
``alife.game.rules``); nothing is imported here.
"""
