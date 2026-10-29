"""The consumer audit of DESIGN §46.12 as data (sphere series, PR S0): every place of ``src/alife`` that still
reads the box geometry directly, with the PR that converts it.

A sphere world's state is Cartesian, so `pos[:, 2]` is not the height, `vel[:, 2]` is not the radial speed,
`ops.min_image` is the identity, and `world.size` / `dims(cfg).fg` describe a lattice the sphere does not
have.
These raw uses keep type-checking and return plausible numbers. ``tests/test_lint.py`` counts them per
file and
pattern and compares them with this table:

* an unlisted raw use, or a count larger than listed, fails (new code uses the accessors
  ``alife.geom.geometry(cfg)`` / ``alife.sphere.geometry(cfg)`` or adds a line here, visible to a reviewer);
* a count that is smaller than listed also fails: lower it (the ratchet only tightens);
* an entry whose reason is ``converted`` names the PR (``until``, the number of the sphere series' PR
S<k>) that
  must have deleted it: the test fails once ``CURRENT_PR`` reaches ``until`` with the entry still present.
  Each PR of the series bumps ``CURRENT_PR`` and deletes the entries it converts.

Reasons: ``pair`` (a pair expression on chord offsets, where ``min_image`` is the identity on a sphere),
``box-only tool`` (a 2D or box-specific tool the sphere refuses), ``definition`` (the config and the accessors
themselves), ``box branch`` (the box half of a function the sphere replaces by a separate one), ``converted``
(the consumer is converted in PR S<until>).
"""

from __future__ import annotations

from dataclasses import dataclass

CURRENT_PR = 0  # the sphere series' PR this tree is: S0. Bump it in each PR (S1 sets 1, ...)
REASONS = ("pair", "box-only tool", "definition", "box branch", "converted")


@dataclass(frozen=True)
class Allow:
    file: str  # relative to src/alife
    pattern: str  # one of test_lint.GEOMETRY_PATTERNS
    count: int
    reason: str
    until: int | None = None  # for "converted": the PR S<until> that must delete the entry


ALLOWLIST: tuple[Allow, ...] = (
    # analysis/emergence/epr.py: the E-series measures: declared box-only, `validate` refuses the
    # emergence recorder on a sphere (S2)
    Allow("analysis/emergence/epr.py", "lattice", 8, "box-only tool"),
    # analysis/emergence/lattice.py: ditto
    Allow("analysis/emergence/lattice.py", "lattice", 3, "box-only tool"),
    # analysis/emergence/record.py: ditto
    Allow("analysis/emergence/record.py", "world.size", 1, "box-only tool"),
    Allow("analysis/emergence/record.py", "world.height", 1, "box-only tool"),
    # analysis/labels.py: G1 labels; the Nomad track needs the sum of great-circle arcs; INTENT_VERSION 4
    # (S5, thresholds in S7)
    Allow("analysis/labels.py", "z_index", 3, "converted", 5),
    Allow("analysis/labels.py", "wrap", 1, "converted", 5),
    Allow("analysis/labels.py", "world.size", 1, "converted", 5),
    Allow("analysis/labels.py", "world.height", 2, "converted", 5),
    # analysis/signals.py: 2D only: `require_2d` already refuses dims 3, which covers the sphere
    Allow("analysis/signals.py", "world.size", 1, "box-only tool"),
    # analysis/w0.py: the W0 storm study driver (S4)
    Allow("analysis/w0.py", "z_index", 4, "converted", 4),
    Allow("analysis/w0.py", "wrap", 1, "converted", 4),
    Allow("analysis/w0.py", "world.size", 3, "converted", 4),
    Allow("analysis/w0.py", "world.height", 4, "converted", 4),
    Allow("analysis/w0.py", "world.field_cell", 3, "converted", 4),
    Allow("analysis/w0.py", "lattice", 15, "converted", 4),
    # analysis/w2.py: the W2 study driver (S4)
    Allow("analysis/w2.py", "world.height", 3, "converted", 4),
    # bench.py: step benchmark scenes: a sphere scene builder in S2
    Allow("bench.py", "world.size", 2, "converted", 2),
    Allow("bench.py", "world.height", 1, "converted", 2),
    Allow("bench.py", "lattice", 2, "converted", 2),
    # bonds.py: pair expressions on chord offsets: min_image is the identity there (DESIGN §46.3)
    Allow("bonds.py", "wrap", 1, "pair"),
    Allow("bonds.py", "world.size", 2, "pair"),
    # cli.py: inspect, replay, runs (S5)
    Allow("cli.py", "world.size", 3, "converted", 5),
    Allow("cli.py", "world.height", 1, "converted", 5),
    Allow("cli.py", "lattice", 2, "converted", 5),
    # config.py: the definitions of the keys, Dims, validate
    Allow("config.py", "world.size", 17, "definition"),
    Allow("config.py", "world.height", 7, "definition"),
    Allow("config.py", "world.field_cell", 15, "definition"),
    # director/__init__.py: levers and prompts in x, y (S5)
    Allow("director/__init__.py", "world.size", 1, "converted", 5),
    # director/actions.py: ditto
    Allow("director/actions.py", "wrap", 1, "converted", 5),
    Allow("director/actions.py", "world.size", 9, "converted", 5),
    # director/llm.py: ditto
    Allow("director/llm.py", "world.size", 1, "converted", 5),
    # director/prompt.py: ditto
    Allow("director/prompt.py", "world.height", 2, "converted", 5),
    # director/yoked.py: ditto
    Allow("director/yoked.py", "world.size", 1, "converted", 5),
    # ext/niche.py: lattice shape and neighbourhood of the farming layer (S3)
    Allow("ext/niche.py", "world.size", 1, "converted", 3),
    Allow("ext/niche.py", "world.height", 1, "converted", 3),
    Allow("ext/niche.py", "world.field_cell", 2, "converted", 3),
    Allow("ext/niche.py", "lattice", 2, "converted", 3),
    # ext/structures.py: lattice shape, neighbourhood and colouring of the structures layer (S3)
    Allow("ext/structures.py", "world.size", 1, "converted", 3),
    Allow("ext/structures.py", "world.height", 1, "converted", 3),
    Allow("ext/structures.py", "world.field_cell", 2, "converted", 3),
    Allow("ext/structures.py", "lattice", 2, "converted", 3),
    # fields.py: resource, chemicals, cell index, patches (S1)
    Allow("fields.py", "z_index", 1, "converted", 1),
    Allow("fields.py", "xy_slice", 1, "converted", 1),
    Allow("fields.py", "wrap", 4, "converted", 1),
    Allow("fields.py", "world.size", 3, "converted", 1),
    Allow("fields.py", "world.field_cell", 4, "converted", 1),
    Allow("fields.py", "lattice", 21, "converted", 1),
    # game/observe.py: the G1 observer's quantised z (S5)
    Allow("game/observe.py", "z_index", 1, "converted", 5),
    Allow("game/observe.py", "world.height", 1, "converted", 5),
    # game/observer.py: host observer (S5)
    Allow("game/observer.py", "world.size", 2, "converted", 5),
    Allow("game/observer.py", "world.height", 1, "converted", 5),
    Allow("game/observer.py", "world.field_cell", 1, "converted", 5),
    # game/powers.py: the player's powers (S5)
    Allow("game/powers.py", "xy_slice", 2, "converted", 5),
    Allow("game/powers.py", "world.size", 3, "converted", 5),
    # game/social.py: social state (S5)
    Allow("game/social.py", "world.size", 2, "converted", 5),
    # game/society.py: houses and territory (S5)
    Allow("game/society.py", "z_index", 1, "converted", 5),
    Allow("game/society.py", "world.size", 3, "converted", 5),
    Allow("game/society.py", "world.height", 1, "converted", 5),
    # game/tracers.py: tracer rig: vz_own, w_air, tracer_z (S5)
    Allow("game/tracers.py", "z_index", 7, "converted", 5),
    Allow("game/tracers.py", "wrap", 2, "converted", 5),
    Allow("game/tracers.py", "world.size", 2, "converted", 5),
    Allow("game/tracers.py", "world.height", 1, "converted", 5),
    Allow("game/tracers.py", "world.field_cell", 6, "converted", 5),
    # geom.py: the accessors' own definitions (host)
    Allow("geom.py", "z_index", 1, "definition"),
    Allow("geom.py", "xy_slice", 1, "definition"),
    Allow("geom.py", "world.size", 1, "definition"),
    Allow("geom.py", "world.height", 2, "definition"),
    Allow("geom.py", "world.field_cell", 1, "definition"),
    Allow("geom.py", "lattice", 4, "definition"),
    # invariants.py: norm, tangent and radius invariants (S2)
    Allow("invariants.py", "z_index", 1, "converted", 2),
    Allow("invariants.py", "world.size", 3, "converted", 2),
    # knn_metal.py: the box search; the sphere search is a separate function (S2)
    Allow("knn_metal.py", "z_index", 1, "box branch"),
    Allow("knn_metal.py", "xy_slice", 1, "box branch"),
    Allow("knn_metal.py", "world.size", 1, "box branch"),
    Allow("knn_metal.py", "world.height", 2, "box branch"),
    # lifecycle.py: placement, founders, catastrophe cap (S2)
    Allow("lifecycle.py", "xy_slice", 1, "converted", 2),
    Allow("lifecycle.py", "wrap", 4, "converted", 2),
    Allow("lifecycle.py", "world.size", 3, "converted", 2),
    Allow("lifecycle.py", "lattice", 3, "converted", 2),
    # metrics.py: observers (S5)
    Allow("metrics.py", "z_index", 1, "converted", 5),
    Allow("metrics.py", "world.height", 1, "converted", 5),
    Allow("metrics.py", "lattice", 6, "converted", 5),
    # niche.py: farming (S3)
    Allow("niche.py", "lattice", 4, "converted", 3),
    # physics.py: forces and integration (S2)
    Allow("physics.py", "z_index", 1, "converted", 2),
    Allow("physics.py", "wrap", 2, "converted", 2),
    Allow("physics.py", "world.size", 2, "converted", 2),
    # play.py: the play loop (S5)
    Allow("play.py", "world.size", 3, "converted", 5),
    Allow("play.py", "world.height", 1, "converted", 5),
    Allow("play.py", "world.field_cell", 3, "converted", 5),
    Allow("play.py", "lattice", 8, "converted", 5),
    # render.py: 2D renderer: box-only tool
    Allow("render.py", "wrap", 4, "box-only tool"),
    Allow("render.py", "world.size", 3, "box-only tool"),
    Allow("render.py", "world.field_cell", 1, "box-only tool"),
    Allow("render.py", "lattice", 3, "box-only tool"),
    # sensing.py: descriptors and gradients (S2)
    Allow("sensing.py", "z_index", 1, "converted", 2),
    Allow("sensing.py", "world.height", 1, "converted", 2),
    # spatial.py: the box search; the sphere search is a separate function (S2)
    Allow("spatial.py", "z_index", 4, "box branch"),
    Allow("spatial.py", "xy_slice", 2, "box branch"),
    Allow("spatial.py", "wrap", 3, "box branch"),
    Allow("spatial.py", "world.size", 3, "box branch"),
    Allow("spatial.py", "lattice", 12, "box branch"),
    # sphere.py: the accessors' own definitions (device)
    Allow("sphere.py", "z_index", 2, "definition"),
    Allow("sphere.py", "wrap", 1, "definition"),
    Allow("sphere.py", "world.size", 1, "definition"),
    Allow("sphere.py", "world.height", 3, "definition"),
    Allow("sphere.py", "lattice", 5, "definition"),
    # state.py: init, cell layout (S1)
    Allow("state.py", "wrap", 2, "converted", 1),
    Allow("state.py", "world.size", 2, "converted", 1),
    Allow("state.py", "lattice", 3, "converted", 1),
    # step.py: step wiring (S2)
    Allow("step.py", "lattice", 6, "converted", 2),
    # stream.py: stream v2 (S5)
    Allow("stream.py", "world.size", 1, "converted", 5),
    Allow("stream.py", "world.height", 1, "converted", 5),
    Allow("stream.py", "world.field_cell", 1, "converted", 5),
    Allow("stream.py", "lattice", 4, "converted", 5),
    # structures.py: structures (S3)
    Allow("structures.py", "lattice", 2, "converted", 3),
    # viewer.py: 2D viewer: box-only tool
    Allow("viewer.py", "wrap", 1, "box-only tool"),
    Allow("viewer.py", "world.size", 7, "box-only tool"),
    # weather.py: wind sampling, vertical profile, remap (S4)
    Allow("weather.py", "z_index", 1, "converted", 4),
    Allow("weather.py", "xy_slice", 1, "converted", 4),
    Allow("weather.py", "wrap", 6, "converted", 4),
    Allow("weather.py", "world.size", 12, "converted", 4),
    Allow("weather.py", "world.height", 5, "converted", 4),
    Allow("weather.py", "world.field_cell", 14, "converted", 4),
)
