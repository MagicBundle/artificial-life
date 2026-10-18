"""The games vertical (docs/GAMES.md, DESIGN §28): build step G1, the observer layer.

``observe`` is the pure-MLX observer state and its per-step update, ``rules`` the archetype and intent rules
(versioned by ``INTENT_VERSION``) over the accumulators. Both read only the step's observer channel
(``step.Acts``) and a few ``State`` fields, never write the ``State`` and never read its RNG key: observing
leaves the trajectory bitwise unchanged. ``observer`` is the host ``Observer``: it folds the channel into the
accumulators in the step's single ``mx.eval``, reports the ``observers`` block of the metrics rows and writes
and reads the sidecar; it reads the ``State`` only through ``observe.obs_in`` (and its step counter when it
starts). This package is imported explicitly (``alife.game.observe``, ``alife.game.rules``,
``alife.game.observer``); nothing is imported here.
"""
