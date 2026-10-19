"""The games vertical (docs/GAMES.md, DESIGN §28, §29): build steps G1 (the observer layer) and G2 (social
tracking).

``observe`` is the pure-MLX observer state and its per-step update, ``rules`` the archetype and intent rules
(versioned by ``INTENT_VERSION``) over the accumulators. Both read only the step's observer channel
(``step.Acts``) and a few ``State`` fields, never write the ``State`` and never read its RNG key: observing
leaves the trajectory bitwise unchanged. ``social`` is the pure-MLX social state (houses, kin group and
colony slots, the relation windows; ``SocialRules`` versioned by ``SOCIAL_VERSION``) and ``society`` its host
half (trackers, territory, relations, the snapshot records). ``observer`` is the host ``Observer``: it folds
the channel into the accumulators and the social state in the step's single ``mx.eval``, settles the
snapshots, reports the ``observers`` block of the metrics rows, hands the snapshot records to the runner (the
run's social log) and writes and reads the sidecar; it reads the ``State`` only through ``observe.obs_in``
and ``social.soc_in`` (and its step counter when it starts). This package is imported explicitly
(``alife.game.observe``, ``alife.game.rules``, ``alife.game.social``, ``alife.game.society``,
``alife.game.observer``); nothing is imported here.
"""
