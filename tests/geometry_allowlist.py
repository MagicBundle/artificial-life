"""The consumer audit of DESIGN §46.12 as data (sphere series, PR S0): every place of ``src/alife`` that still
reads the box geometry directly, with the PR that converts it.

A sphere world's state is Cartesian, so `pos[:, 2]` is not the height, `vel[:, 2]` is not the radial speed,
`ops.min_image` is the identity, and `world.size` / `dims(cfg).fg` describe a lattice the sphere does not
have.
These raw uses keep type-checking and return plausible numbers. ``tests/test_lint.py`` counts them per
file and
pattern (``tests/geometry_scan.py``) and compares them with this table:

* an unlisted raw use, or a count larger than listed, fails (new code uses the accessors
  ``alife.geom.geometry(cfg)`` / ``alife.sphere.geometry(cfg)`` or adds a line here, visible to a reviewer);
* a count that is smaller than listed also fails: lower it (the ratchet only tightens);
* an entry whose reason is ``converted`` names the PR (``until``, the number of the sphere series' PR
S<k>) that
  must have deleted it: the test fails once ``CURRENT_PR`` reaches ``until`` with the entry still present.
  Each PR of the series bumps ``CURRENT_PR`` and deletes the entries it converts;
* ``AUDIT`` is the audit table of the series spec (section 2.9) file by file, with the reason and
converting PR
  it assigns; ``test_the_allowlist_agrees_with_the_spec_audit`` requires every audited file to be listed
  with that
  reason and PR, or to be named in ``NO_RAW_USE_TODAY`` on purpose, and every other listed file to be named in
  ``NOT_IN_SPEC_AUDIT`` (the discrepancies between the lint and the spec, both ways).

Reasons: ``pair`` (a pair expression on chord offsets, where ``min_image`` is the identity on a sphere),
``box-only tool`` (a 2D or box-specific tool the sphere refuses), ``definition`` (unused in the table:
the files that define the geometry are not counted at all, see
``geometry_scan.DEFINITION_FILES``), ``box branch`` (the box half of a function the sphere replaces by a
separate one),
``converted`` (the consumer is converted in PR S<until>).
"""

from __future__ import annotations

from dataclasses import dataclass

CURRENT_PR = 2  # the sphere series' PR this tree is: S2. Bump it in each PR (S3 sets 3, ...)
REASONS = ("pair", "box-only tool", "box branch", "converted")


@dataclass(frozen=True)
class Allow:
    file: str  # relative to src/alife
    pattern: str  # one of geometry_scan.GEOMETRY_PATTERNS
    count: int
    reason: str
    until: int | None = None  # for "converted": the PR S<until> that must delete the entry


_C, _B, _P, _BB = "converted", "box-only tool", "pair", "box branch"
# The audit table of the spec (2.9): file -> (reason, converting PR). `analysis/w0.py`: the spec says S5
# for the
# storm tracker and S4 for the study driver; the file is the study driver (the tracker is `storms.py`).
AUDIT: dict[str, tuple[str, int | None]] = {
    "spatial.py": (_BB, None),
    "knn_metal.py": (_BB, None),
    "physics.py": (_BB, None),
    "sensing.py": (_BB, None),
    "lifecycle.py": (_BB, None),
    "invariants.py": (_BB, None),
    "fields.py": (_BB, None),  # S1 converted it: the sphere half is fields_sph.py; the box half stays
    "state.py": (_BB, None),  # S1 converted it: the sphere branches use Dims; the box half stays
    "energy.py": (_P, None),
    "bonds.py": (_P, None),
    "weather.py": (_C, 4),
    "weather_obs.py": (_C, 4),
    "weather_spinup.py": (_C, 4),
    "bench.py": (_BB, None),
    "ext/structures.py": (_C, 3),
    "ext/niche.py": (_C, 3),
    "structures.py": (_C, 3),
    "niche.py": (_C, 3),
    "metrics.py": (_C, 5),
    "featobs.py": (_C, 5),
    "groups.py": (_C, 5),
    "tracking.py": (_C, 5),
    "streamrec.py": (_C, 5),
    "stream.py": (_C, 5),
    "play.py": (_C, 5),
    "cli.py": (_C, 5),
    "runio.py": (_C, 5),
    "game/observe.py": (_C, 5),
    "game/observer.py": (_C, 5),
    "game/tracers.py": (_C, 5),
    "game/society.py": (_C, 5),
    "game/powers.py": (_C, 5),
    "game/social.py": (_C, 5),
    "game/alerts.py": (_C, 5),
    "game/report.py": (_C, 5),
    "game/kinthreads.py": (_C, 5),
    "game/housefile.py": (_C, 5),
    "analysis/labels.py": (_C, 5),
    "storms.py": (_C, 5),
    "analysis/w0.py": (_C, 4),
    "analysis/w0_plots.py": (_C, 4),
    "analysis/w2.py": (_C, 4),
    "director/actions.py": (_C, 5),
    "director/yoked.py": (_C, 5),
    "director/llm.py": (_C, 5),
    "director/prompt.py": (_C, 5),
    "director/__init__.py": (_C, 5),
    "render.py": (_B, None),
    "viewer.py": (_B, None),
    "analysis/signals.py": (_B, None),
    "analysis/emergence/epr.py": (_B, None),
    "analysis/emergence/record.py": (_B, None),
    "analysis/emergence/lattice.py": (_B, None),
    "migrate.py": (_B, None),
}
# Audited files with no raw use of the seven patterns today: named on purpose (a use appearing there fails).
NO_RAW_USE_TODAY = frozenset(
    {
        "energy.py",  # pair code reads offsets from bonds/spatial; no raw z or wrap of its own
        "featobs.py",  # the spec counts 12 uses of the observers' own arrays; none of the seven patterns
        "weather_obs.py",
        "weather_spinup.py",
        "groups.py",
        "streamrec.py",
        "game/kinthreads.py",
        "game/housefile.py",
        "migrate.py",
    }
)
# Listed files that the spec's audit table does not name (the lint finds them; the spec's grep counts differ).
NOT_IN_SPEC_AUDIT = frozenset(
    {
        "step.py",
        "analysis/activity.py",
        "analysis/emergence/braid.py",
        "analysis/emergence/frames.py",
        "analysis/emergence/modes.py",
        "analysis/emergence/winding.py",
    }
)

ALLOWLIST: tuple[Allow, ...] = (
    # analysis/activity.py: evolutionary-activity analysis of finished runs: positions and size (S5; not
    # in the spec table)
    Allow("analysis/activity.py", "z_index", 1, "converted", 5),
    # analysis/emergence/braid.py: the E-series measures: declared box-only, `validate` refuses the
    # emergence recorder on a sphere (S2)
    Allow("analysis/emergence/braid.py", "xy_slice", 1, "box-only tool"),
    Allow("analysis/emergence/braid.py", "wrap", 2, "box-only tool"),
    # analysis/emergence/epr.py: ditto
    Allow("analysis/emergence/epr.py", "wrap", 1, "box-only tool"),
    Allow("analysis/emergence/epr.py", "lattice", 8, "box-only tool"),
    # analysis/emergence/frames.py: ditto
    Allow("analysis/emergence/frames.py", "z_index", 1, "box-only tool"),
    # analysis/emergence/lattice.py: ditto (holds its own torus min_image)
    Allow("analysis/emergence/lattice.py", "z_index", 1, "box-only tool"),
    Allow("analysis/emergence/lattice.py", "xy_slice", 9, "box-only tool"),
    Allow("analysis/emergence/lattice.py", "wrap", 3, "box-only tool"),
    Allow("analysis/emergence/lattice.py", "lattice", 3, "box-only tool"),
    # analysis/emergence/modes.py: ditto
    Allow("analysis/emergence/modes.py", "z_index", 2, "box-only tool"),
    # analysis/emergence/record.py: ditto
    Allow("analysis/emergence/record.py", "xy_slice", 2, "box-only tool"),
    Allow("analysis/emergence/record.py", "wrap", 1, "box-only tool"),
    Allow("analysis/emergence/record.py", "world.size", 1, "box-only tool"),
    Allow("analysis/emergence/record.py", "world.height", 1, "box-only tool"),
    # analysis/emergence/winding.py: ditto (unwrapped torus tracks)
    Allow("analysis/emergence/winding.py", "z_index", 3, "box-only tool"),
    Allow("analysis/emergence/winding.py", "xy_slice", 1, "box-only tool"),
    Allow("analysis/emergence/winding.py", "wrap", 6, "box-only tool"),
    # analysis/labels.py: G1 labels; the Nomad track needs the sum of great-circle arcs; INTENT_VERSION 4
    # (S5, thresholds in S7)
    Allow("analysis/labels.py", "z_index", 2, "converted", 5),
    Allow("analysis/labels.py", "xy_slice", 2, "converted", 5),
    Allow("analysis/labels.py", "wrap", 1, "converted", 5),
    Allow("analysis/labels.py", "world.size", 1, "converted", 5),
    Allow("analysis/labels.py", "world.height", 2, "converted", 5),
    # analysis/signals.py: 2D only: `require_2d` already refuses dims 3, which covers the sphere
    Allow("analysis/signals.py", "wrap", 1, "box-only tool"),
    Allow("analysis/signals.py", "world.size", 1, "box-only tool"),
    # analysis/w0.py: the W0 storm study driver (S4)
    Allow("analysis/w0.py", "z_index", 29, "converted", 4),
    Allow("analysis/w0.py", "xy_slice", 1, "converted", 4),
    Allow("analysis/w0.py", "wrap", 35, "converted", 4),
    Allow("analysis/w0.py", "world.size", 3, "converted", 4),
    Allow("analysis/w0.py", "world.height", 4, "converted", 4),
    Allow("analysis/w0.py", "world.field_cell", 3, "converted", 4),
    Allow("analysis/w0.py", "lattice", 15, "converted", 4),
    # analysis/w0_plots.py: W0 plots (S4)
    Allow("analysis/w0_plots.py", "z_index", 1, "converted", 4),
    # analysis/w2.py: the W2 study driver (S4)
    Allow("analysis/w2.py", "world.height", 3, "converted", 4),
    # bench.py: the environment block, the play-bench centre and list indexing; the staged step has its
    # sphere branches (S2: `--stages` works on a sphere)
    Allow("bench.py", "z_index", 4, "box branch"),
    Allow("bench.py", "world.size", 2, "box branch"),
    Allow("bench.py", "world.height", 1, "box branch"),
    Allow("bench.py", "lattice", 2, "box branch"),
    # bonds.py: pair expressions on chord offsets: min_image is the identity there (DESIGN §46.3)
    Allow("bonds.py", "wrap", 1, "pair"),
    Allow("bonds.py", "world.size", 2, "pair"),
    # cli.py: inspect, replay, runs (S5); the fork guard of the geometry keys is S2's (DESIGN §46.13)
    Allow("cli.py", "z_index", 2, "converted", 5),
    Allow("cli.py", "wrap", 2, "converted", 5),
    Allow("cli.py", "world.size", 3, "converted", 5),
    Allow("cli.py", "world.height", 2, "converted", 5),
    Allow("cli.py", "lattice", 3, "converted", 5),
    # director/__init__.py: levers and prompts in x, y (S5)
    Allow("director/__init__.py", "world.size", 1, "converted", 5),
    # director/actions.py: ditto
    Allow("director/actions.py", "wrap", 4, "converted", 5),
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
    # fields.py: S1 converted it: every function the sphere replaces dispatches on `dims(cfg).sph` to
    # `fields_sph` and keeps its box half (the counts are those box halves). 3 of the 21 lattice reads are
    # in `mat_colour` / `update_mat` (the built material, converted in S3 with `structures`)
    Allow("fields.py", "z_index", 3, "box branch"),
    Allow("fields.py", "xy_slice", 1, "box branch"),
    Allow("fields.py", "wrap", 4, "box branch"),
    Allow("fields.py", "world.size", 3, "box branch"),
    Allow("fields.py", "world.field_cell", 4, "box branch"),
    Allow("fields.py", "lattice", 21, "box branch"),
    # game/alerts.py: alerts (S5)
    Allow("game/alerts.py", "z_index", 1, "converted", 5),
    # game/observe.py: the G1 observer's quantised z (S5)
    Allow("game/observe.py", "z_index", 1, "converted", 5),
    Allow("game/observe.py", "world.height", 1, "converted", 5),
    # game/observer.py: host observer (S5)
    Allow("game/observer.py", "z_index", 3, "converted", 5),
    Allow("game/observer.py", "wrap", 2, "converted", 5),
    Allow("game/observer.py", "world.size", 2, "converted", 5),
    Allow("game/observer.py", "world.height", 1, "converted", 5),
    Allow("game/observer.py", "world.field_cell", 1, "converted", 5),
    # game/powers.py: the player's powers (S5)
    Allow("game/powers.py", "xy_slice", 2, "converted", 5),
    Allow("game/powers.py", "wrap", 2, "converted", 5),
    Allow("game/powers.py", "world.size", 3, "converted", 5),
    # game/report.py: the world report (S5)
    Allow("game/report.py", "z_index", 2, "converted", 5),
    Allow("game/report.py", "xy_slice", 1, "converted", 5),
    # game/social.py: social state (S5)
    Allow("game/social.py", "world.size", 2, "converted", 5),
    # game/society.py: houses and territory (S5)
    Allow("game/society.py", "z_index", 2, "converted", 5),
    Allow("game/society.py", "wrap", 3, "converted", 5),
    Allow("game/society.py", "world.size", 3, "converted", 5),
    Allow("game/society.py", "world.height", 1, "converted", 5),
    # game/tracers.py: tracer rig: vz_own, w_air, tracer_z (S5)
    Allow("game/tracers.py", "z_index", 14, "converted", 5),
    Allow("game/tracers.py", "wrap", 2, "converted", 5),
    Allow("game/tracers.py", "world.size", 2, "converted", 5),
    Allow("game/tracers.py", "world.height", 1, "converted", 5),
    Allow("game/tracers.py", "world.field_cell", 6, "converted", 5),
    # invariants.py: the box ranges; the sphere half is `_check_rows_sph` and device_check's own branch (S2)
    Allow("invariants.py", "z_index", 3, "box branch"),
    Allow("invariants.py", "world.size", 3, "box branch"),
    # knn_metal.py: the box search; the sphere search is a separate function (S2)
    Allow("knn_metal.py", "z_index", 1, "box branch"),
    Allow("knn_metal.py", "xy_slice", 1, "box branch"),
    Allow("knn_metal.py", "world.size", 1, "box branch"),
    Allow("knn_metal.py", "world.height", 2, "box branch"),
    # lifecycle.py: the box placement and catastrophe column; the sphere has its own functions (S2)
    Allow("lifecycle.py", "z_index", 1, "box branch"),
    Allow("lifecycle.py", "xy_slice", 1, "box branch"),
    Allow("lifecycle.py", "wrap", 4, "box branch"),
    Allow("lifecycle.py", "world.size", 3, "box branch"),
    Allow("lifecycle.py", "lattice", 3, "box branch"),
    # metrics.py: observers (S5)
    Allow("metrics.py", "world.height", 1, "converted", 5),
    Allow("metrics.py", "lattice", 6, "converted", 5),
    # niche.py: farming (S3)
    Allow("niche.py", "lattice", 4, "converted", 3),
    # physics.py: the box forces and integrators; the sphere has `forces_s` / `integrate_s` (S2)
    Allow("physics.py", "z_index", 7, "box branch"),
    Allow("physics.py", "xy_slice", 3, "box branch"),
    Allow("physics.py", "wrap", 2, "box branch"),
    Allow("physics.py", "world.size", 2, "box branch"),
    # play.py: the play loop (S5)
    Allow("play.py", "z_index", 4, "converted", 5),
    Allow("play.py", "xy_slice", 2, "converted", 5),
    Allow("play.py", "wrap", 4, "converted", 5),
    Allow("play.py", "world.size", 3, "converted", 5),
    Allow("play.py", "world.height", 1, "converted", 5),
    Allow("play.py", "world.field_cell", 3, "converted", 5),
    Allow("play.py", "lattice", 8, "converted", 5),
    # render.py: 2D renderer: box-only tool
    Allow("render.py", "z_index", 2, "box-only tool"),
    Allow("render.py", "wrap", 5, "box-only tool"),
    Allow("render.py", "world.size", 3, "box-only tool"),
    Allow("render.py", "world.field_cell", 1, "box-only tool"),
    Allow("render.py", "lattice", 3, "box-only tool"),
    # runio.py: run metadata and listing (S5); migrate/refusal checks are S2's
    Allow("runio.py", "z_index", 2, "converted", 5),
    # sensing.py: the box frame (to_body3); the sphere branches use `ops.to_body_s` (S2)
    Allow("sensing.py", "z_index", 6, "box branch"),
    Allow("sensing.py", "xy_slice", 6, "box branch"),
    Allow("sensing.py", "world.height", 1, "box branch"),
    # spatial.py: the box search; the sphere search is a separate function (S2)
    Allow("spatial.py", "z_index", 5, "box branch"),
    Allow("spatial.py", "xy_slice", 2, "box branch"),
    Allow("spatial.py", "wrap", 3, "box branch"),
    Allow("spatial.py", "world.size", 3, "box branch"),
    Allow("spatial.py", "lattice", 12, "box branch"),
    # state.py: S1 converted it: the sphere branches of `_initial_state` and `expected_layout` use the
    # geometry-aware `Dims` (`field_shape` is (FZ, Npix) on a sphere); these are the box halves and
    # shared reads
    Allow("state.py", "wrap", 2, "box branch"),
    Allow("state.py", "world.size", 2, "box branch"),
    Allow("state.py", "lattice", 3, "box branch"),
    # step.py: step wiring; the box half of the column occupancy (S2; not in the spec table)
    Allow("step.py", "lattice", 6, "box branch"),
    # storms.py: the storm tracker and its host torus helper min_image_np (S5: tracker on vortices
    # crossing the poles)
    Allow("storms.py", "wrap", 13, "converted", 5),
    # stream.py: stream v2 (S5)
    Allow("stream.py", "world.size", 1, "converted", 5),
    Allow("stream.py", "world.height", 1, "converted", 5),
    Allow("stream.py", "world.field_cell", 1, "converted", 5),
    Allow("stream.py", "lattice", 4, "converted", 5),
    # structures.py: structures (S3)
    Allow("structures.py", "lattice", 2, "converted", 3),
    # tracking.py: generic tracking and circular_mean (S5: the Nomad track, kin groups on the sphere)
    Allow("tracking.py", "z_index", 1, "converted", 5),
    Allow("tracking.py", "wrap", 1, "converted", 5),
    # viewer.py: 2D viewer: box-only tool
    Allow("viewer.py", "wrap", 8, "box-only tool"),
    Allow("viewer.py", "world.size", 7, "box-only tool"),
    # weather.py: wind sampling, vertical profile, remap (S4)
    Allow("weather.py", "z_index", 11, "converted", 4),
    Allow("weather.py", "xy_slice", 2, "converted", 4),
    Allow("weather.py", "wrap", 6, "converted", 4),
    Allow("weather.py", "world.size", 12, "converted", 4),
    Allow("weather.py", "world.height", 5, "converted", 4),
    Allow("weather.py", "world.field_cell", 14, "converted", 4),
)
