# Diel vertical migration: a day-night light cycle on the photic food band (design specification)

Status: **Revision 2: decisions recorded (§9), approved by the owner on 2026-11-04.** Revision 1 (`spec_rev1.md`) answered the independent review (`SP/diel_review/review.md`, §11); Revision 0 is `spec_rev0.md`; §12 lists the changes of Revision 2. Docs only, nothing built or run. Date 2026-11-04. Source: `docs/brainstorming-Fable-November.md`
§5 item 4 (line 203) and §9 item 1 (line 353). Paths are relative to the repo root (main checkout, read-only). Evidence is
cited as `file:line`; a statement with no citation is a design proposal, and "UNVERIFIED" marks what I could not check.

## 0. Summary

* A light factor `L(t, place)` in [0, 1] multiplies only the **photic term** of the vertical food profile. Food stays an
  integer quantum source booked as `INJECT_REGEN`; **no ledger term is added** and the 16-term ledger closes by
  construction (§2.4).
* Two modes: `global` (one clock for the whole world; the only mode in the box) and `terminator` (sphere only: a subsolar
  longitude that moves, so a planet has a day side and a night side, with permanent twilight at the poles because the
  planet has no axial tilt).
* A matched control is built into the same code path (`diel.control = "mean"`): the photic term is replaced by its exact
  day-mean, so the control has the same mean food in every cell and no cycle.
* Organisms are unlikely to evolve a daily rhythm from food cues alone (§4; food cues are themselves world-synchronous, so
  arm N is not "clock-free"): a lifespan of order 650 steps is far shorter than a day of 4,000 steps, and the only clock
  input is per-agent age. Recommendation: one new silent brain input, a **depth-independent sky light**, through the extension
  registry (D_IN 84 -> 85).
* Observers are host-side (a `diel_*` block in the metrics row plus `alife.analysis.diel`); the migration statistic is the
  first-harmonic amplitude of the population's median height against local solar time, with the off-period frequencies of
  the same run as its noise floor.
* One player power, **Long Night** (`lengthen_night`): cost 1, timed, ramped, no energy path.
* Pre-registered claim C1 is "a daily vertical **rhythm**" in the box (`mini3d`, global mode) with lineages as the unit,
  paired against the mean-matched control; the word "**evolved migration**" is earned only by a discriminating endpoint
  (E5, a mute fork). Before any long run, an **oracle pilot** (scripted migrators) bounds the benefit and defines the
  go/no-go (§7.0), depth-aware and stratified by band. The box world is `mini3d` with **height 40** and `P = 4,000`
  (recorded amendment A1). A **sphere repeat is pre-committed** whatever the box outcome (§7.5): three arms on `mini3s`,
  weather on and off.
* Build: ten PRs; the engine core (D1) is 600 to 900 source lines; total GPU about 27 h (core) to 38 h (everything) (§8).

## 1. How light and food work today (evidence)

### 1.1 Food

* Food is the int32 grid `State.resource`; regeneration is `inc = clip(min(regen_rate, resource_cap - R), 0, inf)` and
  `INJECT_REGEN = sum(inc)` (`src/alife/fields.py:344-368`, docstring 351-361). The same clip with the farm bonus is at
  `fields.py:370-395`.
* The rate is `regen_rate = floor(regen_max * fertility(patch_pos, cfg, step, env))`, scaled by `env.regen_mult` before the
  floor, plus the weather's integer rain after it (`fields.py:122-142`). The director's Food power therefore scales
  everything inside the product, including any diel factor, and does not scale rain (`fields.py:129-135`).
* `fertility` is `clip(fertility_base + sum_p w_p * gaussian_p, 0, 1)` (`fields.py:62-102`); in 3D it returns
  `fert[None] * vertical_profile(cfg)` (`fields.py:100-101`); on the sphere it dispatches to `fields_sph.fertility`
  (`fields.py:79-84`, `fields_sph.py:198-205`), `cells[None, :] * profile(cfg)`.
* The seasonal weights `w_p = 1 + A sin(2 pi t/P + phi_p)` act on the **patches** (the horizontal pattern), with
  `t mod P` taken in integers first (`fields.py:42-59`, 52-55). `season_mode` is `global` or `per_patch`
  (`config.py:192-194`, `SEASON_MODES` at `config.py:1177`, validation `config.py:1677-1689`).
* The vertical profile, box: `V[k] = min(1, exp(-k fc/benthic_lambda) + photic_amp exp(-(FZ-1-k) fc/photic_lambda))`, the
  photic term only when `photic_amp > 0`, `V[0] = 1` exactly, built on the host from config constants as a (FZ,1,1) float32
  array (`fields.py:105-119`; DESIGN §27.5 line 5095). Sphere: same law at the layers' lower interface heights, (FZ,1)
  (`fieldtab.py:164-177`, `fields_sph.py:112-119`). Keys: `photic_amp`, `photic_lambda` (`config.py:196-197`; validated in 3D
  only, `config.py:1700-1706`; both are in the 3D key list, `config.py:980-981`). Presets with the band: `64k3d`, `mini3d`,
  `tiny3d`, `64k3s`, `mini3s`, `tiny3s` all use `photic_amp = 0.5`, `photic_lambda = 8.0`
  (`src/alife/presets/*.toml`, e.g. `64k3d.toml:85-86`, `mini3s.toml:58-59`).
* The initial resource is `floor(resource_cap * fertility(patch_pos, cfg))` with `step=None`, i.e. the time average
  (`state.py:567`; docstring `fields.py:66-69`). The carrying capacity estimate also uses the time average
  (`metrics.py:743-756`).
* The step calls `fields.update_resource(s2.resource, s2.patch_pos, cfg, s2.step, env, rain)` at `step.py:433` (the farm
  variant at 428); `s2.step` is still `t` there because the `_replace(... step=t+1)` follows at 435. `regen_inc` is booked
  as `LedgerIdx.INJECT_REGEN` (`step.py:455-456, 615-622`; `state.py:25`).

* **Every caller that needs the light factor or the step** (Revision 1; the "two places" of Revision 0 was wrong):
  `fields.fertility`/`regen_rate`/`vertical_profile` take no state argument, and are called from `step.py:428, 433`
  (with `niche.py:148` for the farm), `state.py:567`, `bench.py:1134` (a copy of the step), `metrics.py:285`
  (`regen` probe), `metrics.py:726-729` (`_layer_regen`, there is no `regen_table`), `metrics.py:740` (`carrying_capacity`),
  `metrics.py:814` (the seasonal probe) and `director/seasons.py`; many tests call them (`tests/test_sphere_fields.py`
  28 references, `test_director_actions.py`, `test_fields_3d.py`, `test_game_report.py`, `test_weather_coupling.py`).
  Design: one new optional keyword `diel: DielIn | None = None` (the effective day fraction as an array) on `fertility`,
  `regen_rate`, `update_resource(_farm)` and `niche.update`. `None` means "the config's `day_frac`", so every existing call
  and every analysis probe is unchanged; only `step.py` and the `bench.py` mirror pass the live `diel_ctl`.

* In the 500,000-step `64k3d` run, layers 15-18 hold 0.175 and the surface layer 0.132 of the regeneration, layer 0 holds
  0.263 (`docs/RESULTS_3D.md:333-334`); occupancy follows food with log correlation 0.994 to 0.996 (`RESULTS_3D.md` §4.2).
  The population is bimodal: about 53% in the floor band and 35% in the top band at the end (`RESULTS_3D.md` §4.1 table).
  So a static vertical split already exists; a day-night cycle makes the photic share (about 0.31 of regeneration, about
  0.15 of it moving on average at day fraction 0.5) time-varying.
* The floor is the richest layer per cell (`V[0] = 1` against the surface's `photic_amp` = 0.5 at most). The surface is
  attractive only by crowding relief (an ideal free distribution), not by a higher rate. The migration incentive is
  therefore: the photic band is an extra day-time resource; a static floor-dweller forgoes it, a static surface-dweller
  loses it at night, a migrator gets both at the price of buoyancy.

### 1.3 Buoyancy and motion

* Rising costs energy, sinking is free. Buoyancy due `floor(buoyancy_cost (buoy_cap b)^2) + floor(buoyancy_cap_cost
  buoy_cap)`, the cap upkeep paid every step (`energy.py:139-141`; DESIGN §27.7). Force: `buoyancy_max * buoy_cap * b - gravity`
  along z (`physics.py:138`; sphere `physics.py:246`). The motion is overdamped, `v = F/gamma` with `drag = 1`, `gamma` in
  [1, 5] (`config.py:86-90`, DESIGN §9).
* Vertical speeds from the preset constants (`gravity 0.015`, `buoyancy_max 0.04`, `thrust_max 0.1`): buoyancy-only sink
  0.015 units/step, buoyancy-only rise at cap 1 about 0.025 (at the preset's `buoy_init 0.5`, 0.005), thrust up to 0.1.
  Time to leave the photic band (about 2.5 e-folds = 20 units) by sinking alone: about 1,300 steps. (My arithmetic from the
  constants, not a measurement.)
* Founders already sink and stratify: grounded share peaks at 0.265 at step 3,100 (`RESULTS_3D.md` §4.1).

### 1.4 What organisms sense

* The brain is an Elman RNN (`src/alife/brain.py:1`), so a hidden state exists and can integrate; inputs are the 69 base
  inputs plus the 3D tail to D_IN = 84 (`layout.py:271-331`).
* Light is **not** sensed. The nearest inputs: `IN_RES` (local resource), `IN_RES_GRAD` and `IN_RES_GRAD_UP`,
  `IN_HEIGHT` = `z/height` (`sensing.py:241`), `IN_FED`, `IN_ENERGY`, and `IN_CLOCK` (`layout.py:60`).
* `IN_CLOCK` is **per-agent age**: `phase = (age mod sensing.clock_period) * 2pi/clock_period`, default period 32
  (`sensing.py:159`, 218-219; `config.py:70`). It carries no world time. Newborns restart it at 0.
* Extension inputs enter through `ext_in` after the base tail (`sensing.py:246-247`).

### 1.5 Where a light cycle acts

1. `fields.vertical_profile` / `fields_sph.profile` (the photic term), consumed by `fields.fertility` (`fields.py:100-101`)
   and `fields_sph.fertility` (`fields_sph.py:205`); nothing else reads the profile, but its callers are many (§1.1).
2. `step.py`: the ramp of `diel_ctl` (`ExtSpec` has no per-step state hook: its hooks are inputs, decode, due, mask, count
   and snapshots, `ext/base.py:140-163`), the oracle override (§7.0) and the `bench.py` mirror.
3. A sensing hook (§4) and a render uniform (§5.4). Harvest, diffusion, the cap, the ledger and the farm bonus all read
   `regen_rate` or `inc` downstream and do not change.

## 2. The cycle

### 2.1 Definition

Let `P = diel.period` (steps), `t` the step index, `th = 2 pi (t mod P)/P` (integer modulo first, as `fields.py:52-55`).
Define the **sine of the sun's elevation** `h`:

* `global`: `h = cos(th)`; the same for every cell, a scalar (no per-cell table in the cycling arm). Box: the only mode. Noon at `t = 0 mod P`.
* `terminator` (sphere): the subsolar point is on the equator at longitude `-th` (the planet turns eastward, the sun
  crosses the sky westward). `s(t) = (cos th, -sin th, 0)` in planet axes (z is the polar axis: `fields_sph.sphere_point`
  gives `cos(theta) = z`, `lambda = atan2(y, x)`; `fields_sph.py:125-131`, `geom.py:943-944`). For a cell with unit
  vector `u_c`, `h = u_c . s(t)` = `cos(lat) cos(lon + th)`, computed as three elementwise products and a sum (no matmul,
  DESIGN F20; same pattern as `fields_sph.fertility_cells`, `fields_sph.py:176-196`).

Light (day_frac range, Revision 1): the peak must reach full light, `(1 - h0)/tw >= 0.5`, i.e. `day_frac >=
acos(1 - tw/2)/pi` (0.161 at `tw = 0.25`); validation and the power's lower bound use it (at `day_frac = 0.05` the noon
`L` would be only 0.57). `x = clip((h - h0)/tw + 0.5, 0, 1)`, `S = x^2 (3 - 2x)`, `L = n + (1 - n) S`, with
`h0 = cos(pi * day_frac_eff)`, `tw = diel.twilight`, `n = diel.night_level`. For an equatorial cell the sun is up for the
fraction `day_frac` of the cycle. At the poles `h = 0` always, so `L = S(0.5 - h0/tw)` is constant: **permanent twilight
at the poles** (no tilt); the day-night amplitude shrinks as `cos(lat)`. Analyses restrict to |lat| < 60 degrees (§5.2).
Axial tilt (a declination moving with the season) is a natural later link to seasons and is **out of scope**.

### 2.2 Shape and the photic term

Replace the photic term, not the profile: `V(k, c, t) = min(1, B[k] + L(t, c) * Pk[k])`, with `B[k] = exp(-z_k/benthic_lambda)`
and `Pk[k] = photic_amp * exp(-(z_{FZ-1} - z_k)/photic_lambda)` the host-built float32 constants of today's profile split in
two (same arithmetic, so `L = 1` reproduces the static profile bit for bit; tested). Shapes: box (FZ,1,1) in `global`;
sphere (FZ,1) in `global` and (FZ, Npix) in `terminator`, which is the shape `cells[None,:] * V` already broadcasts to
(`fields_sph.py:205`). The clip at 1 and `V[0] = 1` are kept (the photic term at layer 0 is of order 1e-5, so `V[0]` is
exactly 1 for every `L`). At night with `n = 0` the top layers have `V = B[k]` of order 1e-5, which `floor` turns into 0
regeneration: the photic layers are food-free at night (UNVERIFIED at preset values; checked in D1's reference test).

### 2.3 Composition with seasons, rain, the director

* Seasons modulate the patch weights inside `fert`; the diel factor modulates `V`. The regeneration is the product
  `regen_max * clip(base + sum w_p g_p) * V(k, c, t)` before the floor, so the two are **separable and commute**. A day-mean
  of the product equals `regen_max * fert * V_bar` when `fert` is constant over a day; with seasons on, the error is of
  order `A * 2 pi P / P_season`. Rule: `season_period == 0` or `season_period >= 8 * diel.period` (error). The first claim
  runs with seasons off.
* Rain (`weather.rain_regen`) is added after the floor and is not scaled (`fields.py:140-141`); it would feed photic layers
  at night. Claim runs have weather off; with weather on the contrast is diluted and reported as such.
* `env.regen_mult` scales the product, diel included (`fields.py:136-139`); the director's Food power therefore needs no
  change. The rule tier's fall criterion has a seasonal exemption (`director/seasons.py`, DESIGN §26.5) and **no diel
  exemption**; a rule-tier director in a diel world is allowed but unvalidated, claim runs use `director` off, and the
  player mode decides nothing (DESIGN §34.5 "Modes"). D1 therefore makes `alife run --director live` refuse a diel world
  (player, replay and off are allowed) until a diel exemption (the analogue of `metrics.seasonal_capacity_table`,
  `metrics.py:791-818`) exists; the exemption is listed as deferred.

### 2.4 Constraints and the ledger

* Pure MLX, fixed shapes, no host sync, no Python branching on array values. Branches are on config constants only:
  `diel.period > 0` (static), `diel.mode`, `diel.control`. The one run-time value, the effective day fraction
  (§6), is an array that enters through `h0 = cos(pi * day_frac_eff)` and a `clip`; no branch.
* No scatter or setitem (`test_lint`). The only new device arrays are one (FZ,Npix) or (Npix,) elementwise expression per
  step; at `64k3s` that is 95,256 cells, below the noise of the existing 95k-cell diffusion (0.9 ms, DESIGN §46.5 "Cost").
* **Ledger.** Food is created only in `INJECT_REGEN = sum(inc)` (`fields.py:354-355`, `step.py:622`, sources
  `state.py:45`). `L` changes the *size* of `regen_rate`, not the accounting path: whatever `inc` is, it is booked, and the
  identity `TOTAL = TOTAL0 + cum(INJECT_REGEN + INJECT_IMMIGRATION) - cum(sinks)` holds exactly (`invariants.py:43, 221`;
  `metrics.py:377`). The photic source is "declared" by the config (`photic_amp`, `diel.*`); there is no stored light energy,
  no 17th slot. The farm and rain paths are untouched. **Required test:** the ledger closes on every step of two full days
  in `tiny3d` and `tiny3s` with the cycle on, and `sum(INJECT_REGEN)` over a day with no agents and a non-binding cap equals
  a host float64 reference of `sum floor(regen_max * fert * V)` exactly.
* No RNG stream is added (`RNG_LAYOUT_VERSION` unchanged); the cycle is a pure function of `(cfg, t, cell)`.
* Floating point: `sin`/`cos`/`exp` are not bit-identical between CPU and Metal; seasons and the Gaussians already have
  per-platform golden digests, and the diel golden follows the same rule (CPU pin and Metal pin).

### 2.5 The matched control

`diel.control = "mean"` replaces `L` by nothing: `V` is the **constant host table**
`V_bar[k, c] = mean over 720 equally spaced phases of min(1, B[k] + L(phase, c) Pk[k])`, built in float64 and cast, cached
like `fields_sph.profile`. The regeneration it feeds has, cell by cell, the day-mean of the cycling arm's regeneration
(before the floor and before seasonal covariance, §2.3). It runs through the same fields code, the same sensing formula with the sensed light **`L_bar(c)`, the day mean of `L`**
at the agent's cell (a host constant; in the box a single number), never the live `L` (Revision 1, M2), and the same layout,
so it is an **inert twin** in the sense of DESIGN §44.1.
`V_bar` also serves `step=None` (initial resource, `K_est`; §1.1) when the cycle is on. The nominal mean-food ratio is reported by a new
regeneration-share tool (`alife.analysis.diel regen`, built in D3; `metrics._layer_regen`, `metrics.py:726-729`, is its base)
and must be 1 +/- 0.002. The **realized** ratio differs: `resource_cap = 8000` (`config.py:181`) clips the photic peaks
(about 850 q/step at `regen_max 1700`, `inc = min(regen, cap - R)`, `fields.py:366`) and the realized `INJECT_REGEN` of D is
below K's even at equal nominal means; §7.3 P2 tightens and uses the ratio as a covariate.

## 3. Configuration

New dataclass `DielConfig`, section `[diel]`, one registry row `diel` appended to `ext.EXTENSIONS`
(`ext/base.py:22`) and `ext.REGISTRY` (`ext/__init__.py:62-69`); its `lever_keys` feed `POST_V01_KEYS` and
`SERIAL_OMIT_KEYS` automatically (`config.py:1037-1046`, DESIGN §43.7), so existing hashes and files are unchanged and
`config.json` omits the section at defaults.

| key | default | meaning |
|---|---|---|
| `diel.period` | 0 | steps per day; 0 = off (the feature switch: `ExtSpec.enabled`) |
| `diel.mode` | "global" | "global" or "terminator" (sphere only) |
| `diel.day_frac` | 0.5 | fraction of the cycle the sun is up (equator), [acos(1 - tw/2)/pi, 0.95) |
| `diel.twilight` | 0.25 | width of the dawn/dusk ramp in units of `h`, (0, 1] |
| `diel.night_level` | 0.0 | photic multiplier at night, [0, 1) (moonlight) |
| `diel.control` | "none" | "none" or "mean" (§2.5) |
| `diel.sense` | false | adds the light brain input (§4); the only key with a layout footprint |
| `diel.extinction` | 0.0 | 0 = sky light, independent of depth (the pre-registered choice, §4.2); > 0 = e-folding depth of an attenuated light |
| `diel.sense_control` | "none" | "none" or "mute": the sensed light replaced by `L_bar` (the mute fork of E5; allowed on a resume) |
| `diel.oracle` | 0 | pilot only: percent of the living agents at the pilot's start (by `uid` hash) that get the oracle flag, fixed then and inherited by children (§7.0); their buoyancy is scripted to the ideal schedule; needs `control = "none"` |
| `diel.silent_init` | true | the input's W1 row starts at zero for founders and immigrants (as `niche.silent_init`) |

Validation (in `ext.validate_all`, `ext/__init__.py` order; patterns of `ext/niche.py` `_validate`):

* While `period == 0`, every other key must equal its default (so no config changes its hash without effect; the niche
  rule, `ext/niche.py:116`).
* `period` in [400, 2^20]; `period % run.metrics_every == 0` and `period / metrics_every >= 12` (the analysis needs at
  least 12 rows per day); `world.dims == 3`; `fields.photic_amp > 0` (else there is nothing to modulate); `mode ==
  "terminator"` needs `world.geometry == "sphere"`; `day_frac` in the range above; `0 < twilight <= 1`; `0 <= night_level < 1`;
  `control == "mean"`, `sense`, `sense_control`, `oracle` need `period > 0`; `sense_control == "mute"` needs `sense`; `oracle` in [0, 100] and needs `control == "none"`; `extinction >= 0`. **`sense = true` is refused by `validate` until D2 lands** (D1 adds the key, not the block).
* Seasons: `season_period == 0` or `season_period >= 8 * period` (§2.3).
* A terminator mode refuses a world whose `world.size` rule or `nside` the sphere already refuses; nothing new there.

**Identity (hard requirement).** With `diel.period = 0`: no traced expression changes (all diel code is behind the static
`cfg.diel.period > 0`), `layout_on` is false so `D_IN`, `genome_layout_version` and the `layout_features` meta are
unchanged (`ext/__init__.py:77-83`), the conditional state field `diel_ctl` does not exist (`ext/base.py:51-59`), no counter
is added, `config.json`, checkpoint metadata and every preset hash are byte-identical. Gate: `tests/identity_matrix.py
--base <commit>` with 0 differences (DESIGN §43.9; the new optional keywords of §1.1 default to None), plus a new matrix case "tiny3d and tiny3s, diel on, 50 steps" that is
*new* (pinned per platform, not compared to the base). Also `tests/test_ext_guards.py` rows for the new keys' frozen
defaults (DESIGN §43.7 "defaults of omitted keys never change").

## 4. What organisms can sense

### 4.1 Can a rhythm be inferred from food?

* The only world-time-like inputs are food-based (`IN_RES`, gradients) and local. An agent at the floor at night sees only
  benthic food; nothing reaches it from the surface, so it has no cue for dawn. An agent at the surface sees the food
  appear and vanish, which is a cue only if it is already there.
* Lifespan: the calls study measured "lifespan about 650 steps" in the `storm160` mini3d world (`docs/RESULTS_CALLS.md:415`;
  UNVERIFIED for the dry boxes and the sphere; `metrics.py:407` reports `age_mean` per row). With `P = 4,000` an
  individual lives about 0.16 of a day. Neither an individual nor, at that ratio, a hidden-state oscillator started at birth
  can hold the world's phase; the only inherited information is the genome. A rhythm must therefore be **reactive to a
  world-synchronous cue**, as in nature (light intensity), not a free-running clock.
* Without a light cue the experiment is close to "can DVM evolve from food gradients alone", a harder and less informative
  claim. Arm N is **not clock-free**: `IN_RES` and `IN_RES_GRAD_UP` are themselves world-synchronous cues (food appears at dawn
  in the photic cells), but only for agents already in them.

### 4.2 Options

* **A, nothing.** No layout change. Cheapest, weakest. Kept as arm N (exploratory) in the claim.
* **B, a light input (recommended).** One brain input `light`, width 1, on a log scale:
  `I = L(t, c_agent) * g(depth)`, `light = clip(1 + ln(I + e)/ln(1/e), 0, 1)`, `e = 1e-3` (a code constant),
  `depth = H - z` (box `z = pos[:,2]`; sphere `sphere.geometry(cfg).height(pos)`, as `sensing.py:239`).
  **The cue must reach the floor (review H1).** With the Revision 0 default (`g = exp(-depth/8)`), an agent on the floor of
  `mini3d` (depth 80) senses `L * 4.5e-5`, i.e. `light` = 0.007 by day and 0 by night; at depth 55 it is 0.1, at 70 0.007; the
  usable range was about 55 units, not 70, and the 53% of the population on the floor (`RESULTS_3D.md` §4.1) get no cue
  and cannot wake the silent W1 row. Two fixes were weighed: (i) a long extinction (`g = exp(-depth/35)`: floor sees 0.10 by
  day, `light` = 0.67; night 0) and (ii) **sky light, `g = 1`**. **Chosen: sky light (`extinction = 0`), pre-registered.**
  Reason: with an attenuated light the same input value means "deep by day" and "shallow at dusk", so time and depth are
  confounded and the brain must de-confound them from `IN_HEIGHT`; sky light is a pure clock-like cue (depth is already
  `IN_HEIGHT`), it reaches every agent and it makes the control exact (`L_bar` is a constant). Cost: less realistic as optics
  (a floor dweller sees the sky). A **sensitivity arm** with `extinction = 35` is pre-registered as exploratory (§7.2).
  Dawn versus dusk is read from the hidden state (the Elman memory).
  `c_agent` is the pixel of `ctx.fcell` on the sphere (`fcell % Npix`, `ext.base.ExtCtx`, `ext/base.py:36-45`); in the box there is no cell dependence. With `sense_control = "mute"` or `control = "mean"`, `L` is replaced by `L_bar(c_agent)`. Time used is
  the step `t` of the sensing call, the same `t` that regenerates the food at the end of that step (§1.1).
* **C, retarget `IN_CLOCK`** (listed for completeness, not a serious candidate): no layout change, but it overwrites the
  per-agent age clock (UNVERIFIED whether evolved brains use it), breaks forks of existing sources, and gives a clock rather
  than a sense.

### 4.3 Layout, token and migration (option B)

* New `ExtSpec` `diel`: `enabled = period > 0`, `layout_on = period > 0 and sense`, one `Block("light", 1, silent=cfg.diel.silent_init)`, token
  `"diel1@1"`, `block_version = 1`; extension index 6 (mask bit 6, so the version is base + 4*64), `TAIL_ORDER` after `niche` (last before `arms`)
  (`EXTENSIONS` is `ext/base.py:18`, `TAIL_ORDER` line 22). The genome layout version becomes `base + 4*mask` (`layout.py:389-400`; `MAX_EXTENSIONS = 31`,
  `ext/base.py:25`); 3D `D_IN` 84 -> 85, `D_OUT` unchanged. No change to the 2D layout (the feature is 3D only).
* Inputs hook: `inputs(s, cfg, ctx) -> (N,1)` f32, returns the formula above. `silent` zero-initialises the W1 row of
  founders and immigrants (`brain.py:34`), so the cost term (L1 of weights, `brain_cost_per_l1`) and the behaviour of a
  founder are unchanged until mutation wakes the row.
* `alife migrate --enable diel` (`ExtSpec.enable`, DESIGN §43.8) adds the block to an evolved source (for example
  `runs/mini3d_s0`, 500,000 steps, `docs/HANDOVER.md:726`): the new W1 row is zero, so the migrated population behaves as
  before until it adapts (`state.py:851-855` describes the same property).
* **Hand-computed goldens of the log formula** `f(I) = clip(1 + ln(I + e)/ln(1/e), 0, 1)`, `e = 1e-3`, `ln(1/e) = 6.9078`
  (D2's test must reproduce them to 1e-3 and then compare the device against a host float64 reference on a grid):
  `f(0) = 0`; `f(1) = 1` (the raw value 1.0001 is clipped); `f(4.5e-5) = 0.0065` (the floor at depth 80 under the Revision 0
  attenuation, 0.007); `f(8.3e-4) = 0.0875`. **The coordinator's list gave `f(8.3e-4) ~ 0.13`; I get 0.0875** (`ln(1.83e-3) =
  -6.3035`, `1 - 6.3035/6.9078`). A value of 0.13 needs `I = 1.45e-3`. I record 0.0875 and ask the coordinator to check the
  source of the 0.13 before D2 pins any number.
* Layout-affecting keys (`sense`) are start-of-run; a fork changes them only through the migration (DESIGN §43.7).
* The inert twin (`control = "mean"`, `sense = true`) has the same layout and cost, so a contrast measures the *temporal*
  information only.

**Recommendation: B with sky light.** It is a cue every agent can read, it needs one input, and the registry machinery and
silent start exist and were used by farming (`ext/niche.py`). Cost: a layout bump for diel worlds only, a migration path, and
a less realistic optics than an attenuated light (the sensitivity arm covers it). The GUI lever file is in D1 (§8).

## 5. Observers, metrics, chronicle and GUI

### 5.1 Metrics row (host side, once per row)

`ExtSpec.snap_dev`/`snap_row` (`ext/base.py:162`; pattern `ext/calls.py:140, 316`) add a `diel_` block to the metrics
row (every `metrics_every = 100` steps, `config.py:304`): `diel_light` (spatial mean of `L`), `diel_day_eff` (effective day
fraction), and three 8-vectors over **local solar time bins** (bin `b` = `floor(8 * frac(H/(2 pi) + 0.5))`, `H` the hour
angle from noon; global mode: `H = th`, which needs **no positions**): `diel_bin_n` (agents, |lat| < 60 degrees on the
sphere), `diel_bin_top` (the share of them in the photic band, `z >= H_world - 2*photic_lambda`, a code constant) and
`diel_bin_z` (their mean height as a fraction of the shell). Only `terminator` needs x, y for the longitude (one more
(N,2) host array per row; UNVERIFIED that the transfer allows it cheaply). The oracle pilot adds `diel_orc_*` (§7.0). Together with the existing `height_q25/50/75`, `band_frac`, `surface_frac` and `age_mean`
(`metrics.py:218-233, 407`) this is sufficient for the whole analysis; no new observer channel (§28) and no per-agent dump
are required for the primary endpoint.

### 5.2 The migration statistic

Per window of `W` whole days (default 20), per run:

* **Primary variable (Revision 1): the photic-band share** `u_b` = count-weighted mean over the window's rows of
  `diel_bin_top`, bin `b` of 8. Rationale: 53% of the population sits on the floor (`RESULTS_3D.md` §4.1) and the realistic
  migrants are top-band dwellers moving a few units (§7.0), so the population median barely moves; a share responds to them.
  Secondary variable: the mean height `diel_bin_z`, as a fraction of the shell. One definition each; the offline fit
  (`alife.analysis.diel amplitude`) uses the same bins plus a day-block bootstrap, not the per-row median.
* First harmonic: `a = (2/8) sum_b u_b cos(H_b)`, `b' = (2/8) sum_b u_b sin(H_b)`, with the bin-centre hour angle `H_b = 2 pi (b + 1/2)/8 - pi` (bin 4 holds local noon, §5.1), raw amplitude
  `A_raw = sqrt(a^2 + b'^2)` and **phase** `phi = atan2(b', a)` (0 = highest at local noon; pi = highest at midnight).
* **Bias.** `A_raw` is a magnitude, so `E[A_raw] > 0` under the null and a paired contrast of two noisy magnitudes is
  biased. Use the **bias-corrected amplitude** `A = sqrt(max(A_raw^2 - mean_i A_raw(P'_i)^2, 0))`, with `P'_i` the off-
  frequency periods below, and report `A_raw` and the floor separately as well.
* **Noise floor from the run itself:** `A_raw` at periods `P' = 0.73 P, 0.87 P, 1.15 P, 1.37 P, 1.61 P` (frequencies that
  carry no light); the detection rule is `A_raw > max(A_min, 3 * max_i A_raw(P'_i))`. This protects against slow drift and
  clade turnover masquerading as a daily signal.
* Terminator mode: pooling by local solar time removes the common-phase assumption; additionally report a **chase**
  statistic, the amplitude of the first harmonic of the longitude of the population's density centre in the sun's frame
  (horizontal following). The terminator's speed at the equator is `2 pi R_p/P` = 0.075 units/step at `mini3s`
  (`R_p = 48`, `P = 4,000`) against a swim speed of at most `thrust_max/drag` = 0.1: chasing the sun is feasible near the
  equator at `mini3s` and not at `64k3s` with `P = 3,000` (0.19). This is why the first claim is in the box (§7).
* Individual level (exploratory): for agents present in at least 5 consecutive `eco` dumps (`runner.py:25`;
  `eco_every`, `config.py:315-320`, which today needs `game.observe` and weather on: relaxing that is an optional D3 item),
  the fraction whose own first harmonic exceeds the 95th percentile of the phase-shuffled null.

### 5.3 Chronicle events and alerts (grounded in validated events)

The chronicle states what a record says, never why (`game/chronicle.py:1-14`); texts are closed templates checked by the
lexicon (`TEMPLATES` at `chronicle.py:63-104`, `tests/test_play_chronicle.py`).

* New events, from a host `Chronicler.from_diel(row)` fed by the metrics rows (kin threads are the model,
  `chronicle.py:24-33`): `diel_begun` when the detection rule of §5.2 has held for 3 consecutive window updates at window
  `W = 6` days (24,000 steps, 28 min at 1x, so the event is rare by construction); `diel_lost` when it has failed for 3. Hysteresis is part of the rule. Templates (provisional,
  **phase-neutral**, Revision 1): "Since step {s0}, the share of organisms in the upper layers has varied once a day, by
  {amp:.0%} of the population, highest around hour {hh} of 24 after local noon ({days} days)." and "The daily variation of
  that share has not held since step {s}." The hour is read from the measured `phi`, so a reversed migration is told as
  such; the detector itself is phase-agnostic. No cause is stated.
* Event kinds 22 `diel_begun`, 23 `diel_lost` are an addition to `EVENT_KINDS` (`streamrec.py:92-114`, the header lists the
  table at 641); the stream rule is compatible additions (DESIGN §33.1). **Id space:** `id = step*1000 + family base + index`
  with bases 0..500 used and alerts from 600 (`chronicle.py:56`, `streamrec.py:149`); no family is free, so diel events go
  in the `system` family (base 500, 100 ids, shared with checkpoint/renumbered/extinct).
* One alert class `diel` (surface `log`, importance 1, no pause) in `game/alerts.py:144-162`; `constants_digest` is
  re-pinned.
* "Validated": a detector that reaches the player must have a **validity study** like G1's (DESIGN §28.8): planted worlds
  with a known amplitude and phase, a no-cycle null, a cycle-without-migration case, and the control arm's real data for the
  false-positive rate per 100 days. Thresholds (`A_min`, factor 3, `W`, hysteresis 3) are provisional until D3's study
  freezes them, with a version constant (`DIEL_RULES_VERSION`), as `INTENT_VERSION` (`game/rules.py:INTENT_VERSION`).
* The live report (`game/report.py`, `REPORT_VERSION = 2`, line 119) gains one clause for the chapter ("this chapter
  covered {n} days; the share of organisms in the upper layers varied by {amp} of the population") behind the same lexicon; a version bump.
* The world's existing vertical archetypes Floater and Sinker (`game/rules.py:197-198, 324-335`) are *not* used for the
  daily event: their windows are EMA-averaged and would smooth the day away.

### 5.4 GUI

* **Visible day and night: two renderers (Revision 1).** (1) The *Sphere view*: `u.sun` (xyz direction, w ambient) with
  a soft terminator `smoothstep(-0.3, 0.7, dot(n, sun))` (`ShadersSphere.swift:53, 224-227`, used at 304, 333, 530) is set
  **relative to the camera** (`SphereRenderer.swift:28-29, 342-344`). In a diel world set `u.sun.xyz` to the planet-frame
  `s(t)` of §2.1 and lower `w`. Organisms are lit by the same term (`ShadersSphere.swift:530`), so give them a lighting floor
  (about 0.35) or they vanish at night. (2) The *Globe* (the default overview): its key light is **fixed in view space** and
  `light.w` is the limb haze weight, not an ambient (`GlobeUniforms.swift:21-29`); it has no night side and "lower w" does not
  apply. D6 adds a world-frame sun transformed into view space and a night weight in `shade`, and the organism glow
  (`ShadersGlobe.swift:226`, added after the lighting) is what shows life on the dark side. **Both views are unchanged when
  `ENVS.diel` is absent.** The client computes `s(t)` from the clock; no per-frame data.
* **Stream.** `ENVS` (`play.py:2254-2305`, DESIGN §33.1.9) gains a `diel` object `{period, mode, day_frac_eff, twilight,
  night_level, control}` (a compatible addition; absent when off). `CLCK` already carries the step.
* **Watch a day.** The rate is `BASE_RATE = 14.4` steps/s times `rate`, accepted anywhere in [0.05, 16]
  (`play.py:103-114, 362`), so no server change. Arithmetic (corrected): `rate = P/(14.4 * T)` for a day of `T` seconds;
  at `P = 4,000`, `T = 30` needs rate 9.26 (133 steps/s) and `T = 40` rate 6.9 (99 steps/s); 16x gives 17 s (230 steps/s).
  The button targets `T = 40 s`. Measured throughput: `mini3d` 150 steps/s (`HANDOVER.md:726`), so it holds there;
  `64k3d` 75 steps/s (`DESIGN.md:6478`), so a day takes about 53 s; **`64k3s` with weather on is unmeasured** and the
  Day button is promised on the mini worlds only. The strip shows "day = N s at the current speed" from the achieved rate.
  Camera: Home with a sun-following heading option.
* `FeatureLevers`: `gui/Sources/AlifeCore/DielLevers.swift` naming exactly the feature's non-CLI keys (`oracle` is CLI-only
  via `cli_only_keys`); built in **D1**, because the config keys land there and `tests/test_gui_sync.py` fails until it exists
  (CLAUDE.md: keep the lever list in sync; DESIGN §43.7, §45.1).

## 6. The player power: Long Night

* Name `night`, action `lengthen_night`, the player's label "Night". A row in `actions.GAME_SPECS` only
  (`director/actions.py:95-107`), so `MENU` stays byte-identical (DESIGN §34.5 "The game table"); `GAME_POWERS["night"] =
  "lengthen_night"` (`actions.py:108-113`). `Spec(cost 1, cooldown 4_000, {"day": (0.2, 0.4)}, duration (4_000, 12_000),
  quantum 250)`: the duration range is one to three days at `P = 4,000` and matches Food's 3,000 to 12,000. The target day
  fraction must be at least `max(0.2, acos(1 - tw/2)/pi)` (the range floor of the Spec, raised to the full-peak bound of §2.1
  when the twilight is wide; else refused as Food's dead zone, `excluded`); default 0.3.
* **State.** One conditional extension field `diel_ctl` (2,) f32 = `(target, effective)` day fraction, rest value
  `(day_frac, day_frac)`, `per_agent False`, `conserved False` (`ext/base.py:51-59`); it exists only while `diel` is on, so
  every other state and digest is unchanged. The step moves the effective value toward the target by at most
  `1/(2P)` per step (`clip`, no branch), so the night lengthens over about half a day and the light never jumps. `apply` writes
  the target (`actions.apply`, `actions.py:711-735`, pattern `regen_pulse` -> `env_regen_mult`); the revert (timed) restores
  the previous target; `lever_keys` returns `("night",)` (`actions.py:160-167`) for lever exclusivity with a second Night.
  The effective day fraction replaces `day_frac` in `h0` (§2.1). The step edit is in `step.py` (and the `bench.py` mirror),
  not a registry row (§1.5). Notes: the revert ramp (0.3 to 0.5 takes 1,600 steps) runs outside the cooldown clock, which
  counts from the end of the effect (harmless), and the f32 ramp may leave `diel_ctl` one ulp off its rest value after a
  Night, so a state digest differs from "never used" (fine; noted).
* **Ledger.** None. Like Food, Bloom and Blight it writes a value the step already reads (DESIGN §34.5.1 "The ledger"):
  food created remains `INJECT_REGEN = sum(inc)`. Test: the ledger closes at every row of a session with Night and the other
  four powers; the mean-food ratio of a Night against no Night is reported (it is below 1 by design: Night is a *transfer*
  from day to night plus a net loss of photic food, which the precedent text states).
* **Rules it falls under.** Influence `PLAYER_BUDGET = 6` per 18,000 steps (`actions.py:93-94`); cooldown from the end of the
  action's latest effect; the refusal order "cooldown, then units, then lever" and the lexicon (DESIGN §34.5.1). Rule P: the
  record is applied after the checkpoint at its step (DESIGN §34.5). Replay applies the logged record; the effective-value
  ramp is a function of the state so a replay reproduces it.
* **Surfaces.** `chronicle._POWER_LABEL` gets `"lengthen_night": "Night"` (`chronicle.py:108-113`), a template
  `power_night` ("Night lengthened to {day:.0%} of the day from step {step:,} for {duration:,} steps."), EVNT kinds 20/21
  reused; a `PLAN` caution ("the photic layers stay dark for N more steps"); the precedent table has no row: "No earlier
  case". `ENVS.powers` lists it.
* **Why this power.** It is the cheapest lever on the one new thing in the world, it is visible (the night side widens on the
  globe) and it has a measurable ecological consequence (the amplitude and phase statistic must follow). It cannot be abused
  into extinction: the photic share is about 0.31 of regeneration (§1.1) and the effect lasts at most three days.

## 7. Pre-registered first claim (C1), oracle gate and bounds

### 7.0 The oracle pilot (go/no-go before any long run; review H3; decision 5)

* **Why.** M1's lesson (`RESULTS_CALLS.md` §16: "the oracle's ceiling on exposure is not a ceiling on net benefit"). A food
  share bounds the gain only before the costs. Rising is slow and costs energy (buoyancy rise at cap 1 about 0.025 units/step,
  at the founders' cap 0.5 only 0.005; `energy.py:139-141`), so a single textbook schedule is not a fair ceiling: a floor
  agent cannot do what a top agent does.
* **The oracle is depth-aware (decision 5a).** Each scripted agent follows the daily schedule **its own band can reach**.
  Strata (fixed, §below): floor (band 0 of the four equal z bands, `metrics.z_band`, `metrics.py:182`), mid (bands 1-2), top
  (band 3). The agent has a home height `z_s` (the stratum's centre at its assignment) and, from its own cap, a reachable
  excursion `up = min(z_top_band_centre - z_s, 0.8 * v_rise(cap) * P * day_frac)` and `down = min(z_s - 1, 0.8 * v_sink *
  P * (1 - day_frac))`, `v_sink = gravity/drag`. In the lit half it rises (`mo.buoy` = 1) until `z_s + up`, then holds with the
  balancing lift `b = gravity/(buoyancy_max * buoy_cap)` (clipped to [0,1]); in the dark half it sinks (`b = 0`) until `z_s -
  down`, then holds. A step.py override of the decoded `mo.buoy` before the costs (static branch on `oracle > 0`), so the
  buoyancy due is paid as usual and the ledger is untouched. The oracle's cap is raised to 1 for its group (it pays the cap
  upkeep, `buoyancy_cap_cost`), otherwise the founders' 0.5 cap makes the ceiling a floor; a no-cap-change variant is
  exploratory. A thrust oracle (pitch up, `move_cost`) is exploratory.
* **Assignment and strata (decision 5b).** At the pilot's start (after a 40,000-step cycle-on warm-up, the founder boom
  over) 20% of the living agents (by `uid` hash) get the oracle flag and each records the band it is in; the flag and the
  stratum are **fixed then and inherited** by children (a conditional per-agent int8 field `diel_orc`, 0 none, 1-3 the
  stratum, `on_birth = "inherit"`, `ext/base.py:51-59`; immigrants get 0). The rest of the population is the comparison group
  of the same stratum (the same inheritance of the band tag with flag 0 recorded as -stratum), so the comparison is within
  one world, same cells, same cycle. Lifespan is about 650 steps, so a start cohort is gone in about 1,000 steps: the
  inheritance is what makes a 40-day pilot a comparison of lineages, and its caveat is that the groups differ in genes as
  well as in behaviour (a random split of a mixed population; the seeds and the bootstrap carry that noise).
* **Measure.** Per stratum, `G_s = ln(mean energy of oracle agents / mean energy of their comparison group)` at ages 300 to
  500, pooled over days 10 to 40 of the pilot, 3 seeds, day-block bootstrap; pooled `G` over the strata weighted by
  stratum size; the energy rows are the `age_mean` machinery (`metrics.py:407`) per `diel_orc` value. **Also reported** per
  stratum, by day and by night (`L > 0.5`): the photic harvest share (harvest q in the photic layers over total harvest q),
  from 12 counters (3 strata x {photic, all} x {day, night}, ExtSpec `count`, DESIGN §43.4), to tell "could not reach"
  (the oracle's photic share did not rise) from "reached, did not pay" (it rose and `G_s` did not).
* **Decision rule (frozen before the pilot, hashed; thresholds are the owner's/reviewer's to confirm):**
  Holm adjustment across the three strata.
  **GO** if the pooled lower 90% bound of `G` >= 0.05.
  **SMALL** if the pooled bound is below that but some stratum's Holm-adjusted lower 90% bound >= 0.05 (label **`daily
  rhythm, small, niche`**: run stage 1 only; the label ceiling follows).
  **STOP** if the pooled upper 90% bound < 0.02 **and** every stratum's upper bound < 0.02. STOP means "the ideal schedule
  of each band pays less than 2% in this economy": the box long runs are not spent and the claim is refuted for this
  economy, not for the idea. Any other outcome (bounds straddling) is read as SMALL's stage-1 only, never as GO.
* A GO followed by a negative C1 refutes "evolves under this sense and period in this world", not the possibility.

### 7.1 Claim, world and arms

**Claim (Revision 1).** In a 3D water column with a light cycle on the photic band, a **daily vertical rhythm** of the
photic-band share appears in the evolved population and is absent under the mean-matched control with the same layout, cost
and mean food. The words "**evolved migration**" are used only if E5 (the mute fork) also holds, because mortality at the
surface and passive sinking can produce a daily oscillation with no behaviour at all (lifespan about 0.16 of a day, so each
individual lives inside one phase; review H2).

* World: `mini3d` + overlay `diel` with **`world.height = 40`** (`world.size` 160 satisfies height <= size/2, DESIGN §27.1)
  (`period 4,000`, `day_frac 0.5`, `twilight 0.25`, `night_level 0`, `mode global`, `sense true`, `extinction 0` sky light),
  seasons off, weather off, director off; 300,000 steps = 75 days; speed at height 40 is UNVERIFIED (assumed at least the
  150 steps/s of `mini3d`, `HANDOVER.md:726`). **P5 is arithmetic, not a measurement, and fails at the preset height.** With
  `gravity 0.015`, `buoyancy_max 0.04`, `photic_lambda 8`, `T_cross = (H - 1 - photic_lambda)/v_rise` with `v_rise = 0.04 -
  0.015 = 0.025` units/step at cap 1: at `H = 80`, 71/0.025 = **2,840 steps > P*day_frac = 2,000**; at `H = 40`,
  31/0.025 = **1,240 <= 2,000**. **Amendment A1 (recorded and hashed before the pilot): height 40.** Food is recalibrated in
  the pilot (a half-height column holds less food; the K arm is the reference for P4). The check is on the full crossing, not
  on the 20-unit photic band of Revision 0; at the founders' cap 0.5 the rise is 0.005 and the crossing needs the cap
  to evolve, which the oracle's cap-1 group bounds.
* Arms per lineage: **D** (cycle, sense), **K** (`control = "mean"`, sense; inert twin: senses `L_bar`), **N** (cycle, no
  sense; exploratory, not clock-free, §4.1). Lineage = fresh run from `run.seed` 21..26, same seed in D and K (paired), 300,000
  steps, metrics every 100. Sensitivity arm S (exploratory): D with `extinction = 35`.
* Stage rule as M1: stage 1 = 3 lineages; the sd of the paired difference is computed **without** the mean or sign; if
  `h = t(0.95, n-1) * sd/sqrt(n)` exceeds 0.03 (units: share of the population), add lineages up to 6 then 9; the minimum
  effect is `max(0.03, 2h)`. The paired sd is an assumption until stage 1 (no data).

### 7.2 Endpoints (frozen before the runs, hashed as in DESIGN §44.8)

* **E1 (primary)**: bias-corrected `A` (§5.2) of the photic-band share over the last 20 days (steps 220,000 to 300,000), per
  lineage and arm; the paired contrast `A_D - A_K`, with the log ratio of realized `INJECT_REGEN` (D/K) as a covariate.
* **E2**: phase concentration of the D lineages' `phi`: resultant length >= 0.8, all within +/- pi/2 of the mean.
* **E4**: the detection rule of §5.2 true in the last 20 days in D and false in K.
* **E5 (discriminates "evolved", promoted from exploratory; review H2)**: **the mute fork.** At step 300,000 each D
  lineage is forked twice for 20 days (80,000 steps, about 9 min each): `U` unchanged and `M` with `sense_control = "mute"`
  (the sensed light replaced by `L_bar`, the same input K gets; allowed on a resume, cf. the niche controls `none/ablate/mute`,
  `ext/niche.py` `CONTROLS`). Same food cycle, same genomes, same population; only the cue is cut. Passive mechanisms
  (surface mortality, sinking, diffusion) survive the cut; a brain-driven rhythm does not. Define `A_U` and `A_M` over days 2
  to 20 after the fork (day 1 is the transient). E5 holds if the paired `A_U - A_M` has a one-sided 95% lower bound
  > minimum effect **and** `A_M <= 0.5 A_U` in a majority of lineages. Why this over the individual-level or the transplant
  test: it needs no new dump (the individual test needs `eco_every` to drop its `game.observe`/weather gate,
  `config.py:1825-1832`, and lifespans of 650 steps give about 6 samples per life), costs 6 short forks instead of 12
  runs for the transplant, and it isolates the cue from the ecology.
* Exploratory: E3 (trend first vs last 20 days; dropped from the confirmatory set because the first window holds the founder
  boom, `RESULTS_3D.md` §4.1, grounded share peaking at step 3,100); arms N and S against K; the W1 row norm of the light
  input and the fraction of agents with a non-zero row; `buoy_cap_q50`; day versus night photic harvest share; the
  individual-level test; the transplant fork; the sphere repeat (`mini3s`, `terminator`, `P = 4,000`) with the chase
  statistic; the oracle's thrust variant.

### 7.3 Preconditions (failure = "inconclusive", not "negative"); remedies pre-registered

* **P1** photic share of regeneration `s_p` in [0.15, 0.50] from the new regeneration-share tool (§2.5); about 0.31 in
  `64k3d` (`RESULTS_3D.md:333-334`). Bound on a benefit before costs: `s_p * (1 - mean L)`; the oracle (§7.0) supplies the net.
* **P2 (tightened, M1)** nominal D/K regeneration 1 +/- 0.002, and realized `INJECT_REGEN` D/K over the last 20 days 1 +/-
  0.01 (was 0.03); if the cap clips more, the **remedy** is to raise `resource_cap` in both arms to the smallest value that
  puts the nominal peak clipping at or below 0.5% of photic regeneration (host computation; it changes the world, so the
  pilot is rerun once). The realized ratio also enters E1 as a covariate.
* **P3 (extended, M1)** a night is felt: the standing stock of the photic layers at **dusk and at dawn** is recorded; pass if
  dawn stock <= 50% of dusk stock. The surface is not food-free at night: `resource_diffusion_q8 = 4` (`config.py:187`)
  moves stock from mid-water (0.489 of stock, 0.079 of regeneration, `RESULTS_3D.md` §4.2) over a 2,000-step night.
* **The fallback ladder (decision 3; replaces Revision 1's R1/R2/R3; each step is a recorded amendment, hashed with the
  pre-registration before the step is run).** Step 0: `P = 4,000`, `H = 40` (A1). If P1, P2 or P3 fail there: **one**
  fallback, step 1: `P = 8,000`, `H = 80` (A2; `T_cross` 2,840 <= 4,000), the pilot is re-run in full (oracle included). If
  P1-P3 fail again: **STOP** (no claim runs). No other world change is allowed; the cap raise of P2 is a calibration of the
  same step, hashed as part of it. Constants (`gravity 0.015`, `buoyancy_max 0.04`, photic 0.5/8, `regen_max 1700`,
  `resource_cap 8000`, `resource_diffusion_q8 4`) were re-read on main by the coordinator on 2026-11-04.
* **The late-bloom extension (pre-registered; decision 3).** If E1 is null over the last 20 days (steps 220,000 to 300,000) but the offline detector of §5.2 (the D3 tool, `DIEL_RULES_VERSION`, independent of D5b) fires in the final
  10% of the run (steps >= 270,000), the lineage's D and K runs are resumed for 100,000 more steps and E1 is re-evaluated once
  over the last 20 days of the extension. Any result so obtained is labelled **`late`** and reported as **optional stopping**
  (both evaluations shown, no p-value adjustment claimed); it can never upgrade a label above `daily rhythm (late, optional
  stopping)`. Extra cost at most 12 runs x 100,000 steps (about 2.2 h).
* **P4** population in D within 0.8 to 1.25 of K; lifespan from `age_mean` stated with the ratio lifespan/`P`;
  `ledger_error` 0 on every row; no `diag` error bit.
* **P5** the full crossing (above).

### 7.4 Labels (the decision table is closed; review L7)

Evaluate in order; the first row that matches is the label.

| E1 lower bound vs min effect | E2 | E4 | E5 | label |
|---|---|---|---|---|
| contrast significantly negative | any | any | any | `reversed` |
| a precondition failed | any | any | any | `inconclusive` |
| > min effect | yes | yes | yes | `daily migration evolved` |
| > min effect | yes | yes | no | `daily rhythm (not shown to be behavioural)` |
| > min effect | no or not yet | any | any | `daily rhythm, phase not concentrated` |
| in (0, min effect] | any | any | any | `daily rhythm, small` |
| <= 0 | any | any | any | `no daily rhythm found` (not a refutation, M1's wording) |

I have no prior on the outcome; M1 was negative with a bounded pressure (`RESULTS_CALLS.md:461`).

### 7.5 The sphere repeat (pre-committed) and the claim-product gap (review M7; decisions 2 and 6)

The claim is in the **box with a global clock, weather off**. The product is the **sphere with a terminator, weather on**.
A result in the box says nothing certain about the planet. Therefore the **sphere repeat is pre-committed and runs whatever
the box outcome is** (a box STOP at the oracle gate does not cancel it; it has its own oracle pilot, same rule, on `mini3s`):

* World `mini3s` (Rp 48, H 24: `T_cross` = (24-1-8)/0.025 = 600 steps <= 2,000), `P = 4,000`, `sense true`, sky light.
* **Three arms** per lineage and weather setting: **K** (`control = "mean"`), **G** (`mode = "global"`: the clock without
  geography), **T** (`mode = "terminator"`). Contrasts T-K (the planet), G-K (the clock), T-G (the geography). Weather off
  and weather on (`stormsph22` overlay, `mini3s.toml` header), so 6 cells per lineage; 3 lineages (stage 1 only, descriptive
  plus the pre-registered ratio below), 300,000 steps. Endpoints are E1 and E5 of §7.2 on local-solar-time bins, restricted to
  |lat| < 60 degrees, plus the chase statistic (§5.2): the equatorial terminator speed at `mini3s` is 0.075 units/step against
  a swim speed of at most 0.1, so a horizontal chase is possible and is reported, not excluded.
* **Rain dilution (decision 6): accepted, not engineered.** Rain is added after the floor and is not scaled
  (`fields.py:140-141`), so with weather on the photic layers get food at night. Pre-registered ratio per arm and lineage:
  **`R_w = A(weather on) / A(weather off)`** (bias-corrected amplitudes, same seed). **Trigger: if the median `R_w` over
  lineages is below about 0.5 in T or G, a later PR adds `diel.rain_follow`** (rain term in the photic layers scaled by
  `L/L_bar`); it is not built now. The rain share of regeneration in `stormsph22` is UNVERIFIED and is read in the pilot.

## 8. Build plan

Small PRs; sizes are source lines / test lines, ranges. Every PR: ruff check, ruff format, `pytest -m "not slow"`, the
identity matrix 0 differences with the feature off, DESIGN section + Review-log row in the same change (IDs from the
coordinator), new keys in the registry, goldens unmoved, CLAUDE.md and `docs/HANDOVER.md` updated (CLAUDE.md conventions).

| PR | content | size | tests |
|---|---|---|---|
| D0 | this spec as DESIGN §49 (the last section is 48, `DESIGN.md:15073`; unless the coordinator has reserved 49) + Review-log row | docs | none |
| D1 | `DielConfig` + validation (`sense` refused), registry row (no footprint), host builders (`fieldtab`, `fields_sph`), the `diel=` keyword threaded through `fertility`, `regen_rate`, `update_resource(_farm)`, `niche.update`, `state.py:567`, `metrics.py:285/726-729/740/814`, **the `bench.py:1134` mirror**, `step.py` ramp of `diel_ctl`, `V_bar`, conditional `diel_ctl`, CLI refusal of `--director live`, `diel` overlay, `DielLevers.swift` (lever file) | src 600-900 / tests 600-900 | identity matrix off; `L = 1` equals the static profile bit for bit; reference regeneration over a day; ledger closes 2 days on `tiny3d`/`tiny3s`; `V_bar`; validation table; geometry (subsolar longitude, poles constant); **checkpoint, resume and `alife migrate` round trip of `diel_ctl`** (a 2D/3D file without it is completed to rest); `bench` mirror digest equals the step; **Metal purity gate** (as G1's: the in-graph code is pure MLX, no host sync) ; sync test; per-platform golden |
| D1b | the depth-aware oracle (`step.py` override, conditional per-agent `diel_orc` with inheritance, 12 harvest counters, per-stratum energy rows, pilot script) | src 250-400 / tests 250-400 | ledger closes with 20% oracle; stratum fixed and inherited, immigrants 0; schedule per band against a host reference; checkpoint/resume of `diel_orc`; the override is the only change |
| D2 | the light input: ExtSpec blocks, token, sensing hook, `sense` allowed, `sense_control`, `migrate --enable diel` | src 150-250 / tests 250-400 | layout version base+4*64; a migrated population's digest unchanged (silent row); formula against a host reference **and the four hand-computed goldens of §4.3 (f(0)=0, f(1)=1, f(4.5e-5)=0.0065, f(8.3e-4)=0.0875)**; `K` and mute give `L_bar`; resume with `sense_control` changed |
| D3 | `diel_*` metrics block, `alife.analysis.diel` (`regen`, `profile`, `amplitude`, `prereg`, `compare`), validity study with planted worlds | src 400-700 / tests 400-600 | planted amplitude/phase; null false-positive rate; bias of `A` under the null; off-frequency floor; prereg hash |
| D4 | **oracle pilot first** (go/no-go, §7.0), then P1-P5 with the ladder, stage rule, box runs, mute forks, late-bloom resumes if triggered, `docs/RESULTS_DIEL.md` | scripts 200-350; GPU in the table below | preconditions and the ladder as assertions; amendments hashed |
| D4s | the sphere repeat (pre-committed): `mini3s` oracle pilot, 3 arms x weather on/off x 3 lineages, `R_w`, `docs/RESULTS_DIEL.md` addendum | scripts 100-200; GPU below | as D4 |
| D5a | stream: `ENVS.diel` (compatible addition), `CLCK` unchanged | src 60-120 / tests 80-150 | absent when off; round trip |
| D6 | GUI: Globe night side (world-frame sun in view space), Sphere view sun and organism floor, clock-strip day readout, "Day" rate (needs D5a, not D1) | Swift 300-500 / tests 150-250 | pure tests of `s(t)` and the rate; GPU luminance day/night per view; existing views unchanged without `ENVS.diel` |
| D5b | game layer: event kinds 22-23, `from_diel`, alert class, report clause, power `lengthen_night`, lexicon pass (after D3's validity study) | src 300-500 / tests 350-550 | the power session test, replay equality, Rule P at a chapter end, ledger with all five powers, templates under the lexicon, constants digest re-pinned |

Order: D1 -> D1b -> D4 oracle (go/no-go) -> D2 -> D3 -> D4 claim -> D4s (after the box, whatever its outcome); D5a -> D6 may run in parallel after D1 (the picture does
not depend on the claim); D5b (Long Night, decision 4) after D3's validity study.

**GPU budget** (all UNVERIFIED speeds: 150 steps/s assumed for the box at height 40, 100 steps/s for `mini3s`, which is unmeasured; Metal, one process at a time):

| item | runs | steps | hours |
|---|---|---|---|
| box oracle pilot (3 seeds, 40k warm-up + 160k) | 3 | 200k | 1.1 |
| box P1-P3 pilot (D and K, 40k, 3 seeds) | 6 | 40k | 0.5 |
| ladder step 1 re-run, if triggered | up to 9 | | up to 2 |
| box claim D and K, 6 lineages | 12 | 300k | 6.7 |
| exploratory arms N and S, 6 lineages | 12 | 300k | 6.7 (optional) |
| mute forks U and M, 6 lineages | 12 | 80k | 1.8 |
| late-bloom resumes, worst case | 12 | 100k | 2.2 |
| sphere oracle pilot (`mini3s`, 3 seeds) | 3 | 200k | 1.7 |
| sphere repeat, 3 arms x 2 weather x 3 lineages | 18 | 300k | 15 |
| **total** | | | **about 27 (core) to 38 (everything)** |

**What only the owner can verify.** (1) Whether the day-night looks right in both views and the time-lapse reads as a day at
the "Day" speed. (2) Metal numbers: golden re-pin on the M5, steps/s of the diel worlds, whether the Day speed is sustained.
(3) The wording of the chronicle lines and the power's caution. (4) The merge decisions: games PRs need the owner's go; the
coordinator assigns Review-log IDs, GPU time and `runs/` space. (5) Whether a negative or STOP result is acceptable for the
games vertical. (6) The thresholds of §7.0.

## 9. Owner decisions, 2026-11-04

The owner delegated the Revision 1 section on decisions to a reviewer model; the coordinator verified its two findings on
main (constants above; P5 at `H = 80` is 71/0.025 = 2,840 > 2,000 steps, deterministic). **The decisions below are recorded
as made; the spec was approved by the owner on 2026-11-04.**

1. **Sense: B, the silent sky-light input.** D2's test carries the four hand-computed goldens of the log formula (§4.3; note
   the discrepancy on the fourth value). Attenuated light (`extinction` 35) stays exploratory arm S.
2. **The claim is in the box; the sphere repeat is pre-committed whatever the box outcome**, with three arms (mean, global,
   terminator) on `mini3s`, weather on and off (§7.5).
3. **Day length 4,000 with height 40 fixed up front** (amendment A1; P5 is arithmetic). Fallback ladder in the
   pre-registration: if P1-P3 fail at `H = 40`, one fallback to `P = 8,000`, `H = 80`, pilot re-run, then STOP; each step hashed
   (§7.3). Late-bloom extension pre-registered, labelled "late", reported as optional stopping (§7.3).
4. **Long Night, ramped, after D3's validity study** (D5b).
5. **Oracle gate**, depth-aware, stratified by the band fixed at the pilot's start, Holm across the three strata; GO pooled
   lower bound >= 0.05; SMALL (pooled below, some stratum's Holm-adjusted bound >= 0.05, label "daily rhythm, small, niche");
   STOP (pooled and every stratum's upper bound < 0.02); photic harvest share per stratum by day and night (§7.0).
6. **Rain: accept the dilution**; pre-registered `R_w = A(weather on)/A(weather off)` with a trigger below about 0.5 for a
   later `rain_follow` PR (§7.5).

**Still open for the owner/coordinator:** (a) the four-number check of decision 1; (b) whether a box STOP should also stop
the sphere repeat's long runs (the spec says no, with its own oracle first); (c) the thresholds of §7.0 are the delegated
reviewer's, the owner has not seen them; (d) the Review-log IDs and DESIGN §49.

## 10. Not verified / risks

* No simulation or test was run (the machine is timing the suite). Every number above not quoted from a document is my
  arithmetic from constants and is marked.
* Stock buffering (P3): if the resource stock outlasts a night in the photic layers, the day-night difference in available
  food is much smaller than the regeneration difference. The `64k3d` standing stock at the surface layer is 0.049 of the
  total against 0.132 of regeneration (`RESULTS_3D.md:334`), which suggests fast turnover, not a measurement of the night.
* Lifespan against day length (§4.1) is taken from a different world (`storm160`).
* I did not read `game/report.py`, the Swift client beyond the sun uniform, `director/seasons.py` or all of `ext/calls.py`;
  the report clause, the lexicon and the stream changes in §5.3-§5.4 are designed from DESIGN §33-34 only.
* That `snap_dev` can cheaply carry x, y positions, that a feature with a state field but no brain footprint is accepted by
  the registry (the arms precedent suggests yes: `ext/base.py:20-22`), and that `DESIGN` section number 49 is free (confirmed in D0).
* Floating point differences between CPU and Metal in `sin`/`cos`/`exp` may move a `floor` boundary; the precedent is the
  per-platform golden.
* Revision 1: the oracle thresholds, the `G` metric (energy at ages 300 to 500) and the P5 figures are my arithmetic and
  need the pilot; the Globe/Sphere renderer facts (M5) were checked at `GlobeUniforms.swift:21-29` and taken from the review
  for the shader lines; whether `bench.py:1134` needs more than the keyword is not checked; `sense_control` as a resume-
  allowed key follows the niche `control` precedent (UNVERIFIED how `resume` filters keys).

## 11. Review findings and what was done (Revision 1)

| finding | disposition |
|---|---|
| H1 cue invisible on the floor | Accepted. Formula arithmetic confirmed (depth 80: 4.5e-5). Chose **sky light** (`extinction = 0`), reason in §4.2; attenuated light (35) kept as exploratory arm S; arm N no longer called clock-free. |
| H2 "evolved" not shown | Accepted. Claim reworded to "a daily vertical rhythm"; **E5 mute fork** promoted to the label rule (§7.2, §7.4), `sense_control` key added; reason for choosing it over the individual and transplant tests given. |
| H3 no oracle | Accepted. §7.0: oracle pilot with `diel.oracle`, metric `G`, GO/SMALL/STOP rule, what a STOP and a later negative each refute; P5 now checks the full crossing with pre-registered remedies (height 40, else P 8,000). |
| M1 realized food, stock | Accepted. P2 tightened to 1 +/- 0.01 with a cap remedy and the ratio as a covariate; P3 records dusk and dawn stock; remedy order R1-R3 pre-registered. |
| M2 control's sensed light | Accepted: `L_bar` (§2.5, §4.3). |
| M3 template asserts a phase | Accepted: phase-neutral template with the measured hour (§5.3); rarity stated. |
| M4 engine surface | Accepted. Callers listed (§1.1), `diel=None` keyword, `step.py` ramp and `bench.py` mirror (§1.5), `carrying_capacity`, D1 at 600-900 lines, `--director live` refusal in place of an exemption (exemption deferred). |
| M5 two renderers | Accepted (checked `GlobeUniforms.swift:21-29`): both views handled, organism floor, unchanged without `ENVS.diel` (§5.4). |
| M6 Day arithmetic | Accepted: 9.26x/133 steps/s for 30 s, 6.9x/99 for 40 s; button targets 40 s; `64k3s` marked unmeasured. |
| M7 claim vs product world | Accepted: §7.5 and owner decisions 2 and 6. |
| L1 `regen_table` absent | Accepted; replaced by a D3 tool based on `_layer_regen`. |
| L2 DESIGN number | Accepted: §49 (last section is 48). |
| L3 line numbers | Partly accepted: `EXTENSIONS` is `ext/base.py:18`, `TAIL_ORDER` 22 (fixed). **Disputed:** `clock_period` is at `config.py:70` (verified by grep: line 65 is not it); Revision 0 was right. |
| L4 `day_frac` range | Accepted: lower bound `acos(1 - tw/2)/pi`, also on the power. |
| L5 positions and mean/median | Accepted: x,y only in terminator; share and mean defined once. |
| L6 bias of `A` | Accepted: bias-corrected amplitude and the floor reported separately; paired sd stated as an assumption. |
| L7 label gaps | Accepted: closed table §7.4. |
| L8 E3 window | Accepted: E3 made exploratory. |
| L9 | No action (review says fine). |
| L10 ramp and ulp | Noted in §6. |
| Build plan items | Done: lever file in D1, bench mirror, checkpoint/resume/migrate tests, `sense` refused until D2, Metal purity gate, CLAUDE.md/HANDOVER update, D5/D6 reordered (D5a before D6), director note. |
| Review Q2 (table vs in-graph L) | Adopted: in-graph scalar `L` in the cycling arm; the table only for the control. |

## 12. Change list, Revision 2 (against `spec_rev1.md`)

| where | change |
|---|---|
| status, §0 | "approved by the owner on 2026-11-04"; summary: height 40, pre-committed sphere repeat, ten PRs, GPU 22 to 36 h (Revision 2 as written; recomputed to 27 to 38 h from the §8 table after D0) |
| §4.3 | the four hand-computed goldens of the log formula for D2's test; **f(8.3e-4) is 0.0875 by my computation, not 0.13: flagged, not adopted** |
| §7.0 | oracle made depth-aware (per-band reachable schedule, cap-1 group); strata fixed at the pilot's start and inherited (`diel_orc` field); pooled and stratified `G`, Holm over three strata; GO/SMALL/STOP as decided; photic harvest share per stratum by day and night (12 counters); Revision 1's variant clause removed (the ladder covers the worlds) |
| §7.1 | height 40 as amendment A1; P5 shown as arithmetic (71/0.025 = 2,840 at H=80; 31/0.025 = 1,240 at H=40) |
| §7.3 | R1/R2/R3 replaced by the ladder (H=40,P=4,000 -> P=8,000,H=80 -> STOP), amendments hashed; late-bloom extension |
| §7.5 | rewritten: pre-committed sphere repeat, three arms, weather on and off, `R_w` and its 0.5 trigger |
| §8 | D1b enlarged; D2 test goldens; D4 ladder and late bloom; new D4s; GPU budget table |
| §9 | replaced by the recorded decisions and four open items |
