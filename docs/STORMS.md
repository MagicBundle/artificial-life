# Storm World: an alien-planet storm game on alife (design proposal)

Status: **proposal, nothing built.** This is the owner's concept of 2026-10-18: a 3D game in which players track and
influence the meteorology of an alien planet, built on alife's evolutionary engine and not looking like a petri dish.
Owner decisions so far:
- game-plausible physics is enough;
- storm systems and an evolving aerial biosphere are combined and coupled;
- G3 onwards target the 3D storm world (§9.1 (a), 2026-10-19), gated by W0;
- W0 runs model B on the torus with two hemispheres, a separate warm-pool map and a 160 × 160 weather grid, against
  the exit criteria of §8 as written (§9.2, §9.3, §9.7, §9.9, 2026-10-19).

It builds on `docs/GAMES.md`, the gaming vertical. G1's observers are built in PR #27, where DESIGN §28 lands;
its Metal purity gate passed on 2026-10-19. G2–G5 are designed, not built. `docs/DESIGN.md` stays normative, and
the open decisions are in §9. `ObsIn`, `INTENT_VERSION` and `OBSERVER_FORMAT` refer to G1 as built in PR #27, so this
note merges after it.

## 1. Summary

- **The concept.** A planet whose weather and airborne life co-evolve:
  - storms form from warm seas, moisture and rotation;
  - organisms ride updrafts, graze in sunlit air and hunt in storm walls;
  - lineages adapt to the storm climate and push back on it by seeding rain and shading the seas;
  - the player names, tracks, forecasts and steers storms through climate levers.
- **Novelty.** None of the storm, planet or alife games checked couples evolving agents and weather in both
  directions (a limited search, §10).
- **Precedent.** Sagan & Salpeter (1976) speculated about sinkers, floaters and hunters in Jupiter's clouds, with
  habitats at the boundaries of up- and downdrafts.
- **Physics.** Recommended: a 2.5D model, two-layer moist rotating shallow water.
  - Condensation moves mass between the layers. That is the latent heating that drives convergence and updrafts.
  - It runs on a weather grid decoupled from the field lattice.
  - It gives a diagnosed 3D wind for the organisms.
  - It needs no pressure solve, at an estimated under 1 ms per engine step against 69 ms for a 64k3d step.
  - Spontaneous storm formation in this configuration is unproven, so an offline prototype (W0) gates any engine work.
- **The coupling adds no energy term.** The 3D ledger is full: 16 of 16 slots.
  - Wind is a drag force inside the force sum, as gravity is a body force with no ledger term.
  - Rain feeds regrowth through `INJECT_REGEN`.
  - Storm deaths take the corpse path.
  - Weather fields are not energy.
- **Decided:** G3 onwards target the 3D storm world, which reverses GAMES §10.7's 64k 2D choice for G3–G5 (§9.1).
- **Decided for W0** (§9): model B, the two-hemisphere torus, the warm-pool map and the 160 × 160 grid.
- **Main open decision** (§9.8): the target step rate (3D runs at 14.4 steps/s).

## 2. Starting point

| needed | in the engine today | evidence |
|---|---|---|
| ground and sky | periodic x and y (L 320); floor and surface clamps (height 80); gravity 0.015 | DESIGN §27.2–27.3 |
| lift | output `OUT_BUOYANCY`, heritable `buoy_cap`, lift `0.04·cap·b` (a cap of 0.375 offsets gravity); cost in `HEAT_BUOYANCY` | DESIGN §27.4, §27.7 |
| food aloft | vertical profile V: 1 at the floor, 0.5 at the surface, 0.01–0.05 in mid-column (layers 6–14) | DESIGN §27.5 |
| wind, moisture, advection | **none** | a grep of `src/alife` |
| a load limit on vertical forcing | `validate`: `thrust_max + max(g, bmax − g) < 2r·k_repel` (64k3d: 0.125 against 0.16) | `config.py` |
| speed | 64k3d 14.4 steps/s (C = 48); neighbour search 79% of the step; S17 fields stage 0.42 ms; fused kNN kernel unbuilt | `docs/bench`; DESIGN §27.13 |
| life in still air | layer shares track food shares (log r 0.994–0.996); 88% of agents in the floor and top bands; a surface clade floats on lift (mean cap 0.44) while the median cap falls (0.19 in seed 0, 0.285 in seed 1; selection and hitchhiking not separated); kill share 0.50; emitters 0.86–0.88; lifetimes 700–950 steps | `docs/RESULTS_3D.md`, `docs/RESULTS_3D_RUNS.md`; speed 0.025–0.034 per step (`speed_mean`, 50k windows after step 50k, `runs/64k3d_s{0,1,2}/metrics.jsonl`) |

## 3. The weather layer

### 3.1 Model options

| model | gives | cost per step (est.) | assessment |
|---|---|---|---|
| A. kinematic (analytic vortices, curl noise) | moving winds, no dynamics | 0.1–0.3 ms | the fallback if B fails W0; a scripted feel |
| **B. two-layer moist rotating shallow water (recommended)** | storms from moisture convergence and rotation; jets; a diagnosed 3D wind | under 1 ms | 2.5D. Condensation as inter-layer mass flux is the established two-layer moist-convective shallow-water form (Lambaerts, Lapeyre, Zeitlin & Bouchut 2011; Rostami & Zeitlin 2018). Genesis here is unproven |
| C. 3D moist Boussinesq (stable fluids, Stam 1999; vorticity confinement, Fedkiw et al. 2001) | true convective towers | 3–6 ms (4–9% of a 64k3d step) | affordable, but storm organisation on this grid is untested and it needs more tuning |
| D. cloud automata (Dobashi 2000; coupled map lattices, Yanagita & Kaneko) | cloud shapes and phases | small | a rendering aid only |

The costs are estimates scaled from the measured 0.42 ms of S17. None is a measurement.

### 3.2 Model B

**Grid.** A weather grid decoupled from `field_cell`, for example 160 × 160 cells of 2 units, periodic like the world.
On the 80-cell field lattice (4-unit cells) the target deformation radius of 8–16 units (step 4 below) is only 2–4 cells
wide, too coarse to resolve a vortex; 2-unit cells give 4–8.

**State per column** (float32, not energy):
- layer thicknesses h₁, h₂;
- layer winds (u₁, v₁), (u₂, v₂);
- the lower-layer bulk humidity Q, humidity integrated over layer 1 (Lambaerts et al. 2011);
- a sea-surface temperature map T_s, a separate warm-pool map (§9.7);
- the rain rate P and cloud depth D_c of the last step, for the coupling (§4).

**Each engine step** (Δt = 1; no sub-cadence, since a k-step skip would need a second compiled graph and the
estimated cost is about 1% of the step):
1. **Advection.** Semi-Lagrangian bilinear backtrace of the winds. Gathers only.
2. **Continuity.** `h_k ← h_k − Δt·∇·(h_k u_k)`, and `Q ← Q − Δt·∇·(Q u₁)`, on the same C-grid face fluxes built from
   `mx.roll`. Convergence therefore concentrates moisture.
   - A positivity limiter scales each cell's outgoing fluxes by `min(1, (h_k − h_min)/(Δt·Σ outflow))`, and each face
     uses the factor of its upwind cell; Q uses h₁'s factor.
   - Mass and moisture are then conserved to float rounding, with h_k ≥ h_min. Any residual clamp is counted as a
     diagnostic.
3. **Pressure.** Gradients with two reduced gravities (g_top, g′). Validation requires `c_f·Δt/Δx ≤ 0.5` for the
   fast mode, with `c_f² = ½[T + √(T² − 4D)]`, `T = g_top(H₁+H₂) + g′H₁` and `D = g_top·g′·H₁H₂`.
4. **Rotation.** `f(y) = f₀ sin(2πy/L)`, applied as a rotation by fΔt with `f₀Δt ≤ 0.5`. The cross components are
   averaged to u- and v-points on the C-grid, which does not increase the norm.
   - A linear β-plane cannot be periodic, so this gives two hemispheres of opposite spin.
   - The deformation radius targets 0.1–0.2 of the 80-unit equator-to-pole span.
   - Beta drift may gather storms near y = L/4 and 3L/4, where |f| peaks; W0 checks for this. In shallow water,
     moist-convective storms built a polar cyclone, with the outcome set by eddy size relative to planetary radius
     (O'Neill, Emanuel & Flierl 2015).
5. **Evaporation.** `E = k_E·(|u₁| + u_gust)·max(Q_e(T_s) − Q, 0)`, with `Q_e(T_s) ≤ Q_s`. Warm seas bring Q close to
   the threshold and convergence carries it over. Wind feeds evaporation (WISHE-like), and the gust term moistens still
   air.
6. **Condensation.** `C = max(Q − Q_s, 0)/τ`, with Q_s constant and lowered locally by seeding (§4), and τ of a few
   steps to avoid grid-scale blow-up.
   - It removes Q, falls as rain P, and moves mass γ·C from layer 1 to layer 2, limited by the positivity bound.
   - The moved mass carries its momentum: `u₂ ← u₂ + Δt·γC·(u₁ − u₂)/h₂`, the upper-layer drag of the two-layer form.
   - That mass transfer is the latent heating: it thins the lower layer, lowers its pressure and drives convergence.
   - There is no separate temperature variable. In this form one would act only through the saturation level, with the
     wrong sign.
7. **Damping.** Friction on layer 1, a weak Rayleigh drag on u₂, and relaxation of h toward rest cap the intensity.

The weather update runs in S17 before `update_resource` and stores P and D_c for that step.

**The wind the organisms feel** is diagnosed on the 20 field layers:
- **Horizontal.** u₁ below the interface height h₁ and u₂ above, sampled bilinearly. A surface-layer factor takes it
  to 0 at the floor.
- **Vertical.** `w(z) = −∫₀^z (∇·u_h − D̄) dz′`, where D̄ is the column-mean divergence. Then w = 0 at the floor and
  the lid with no local taper. D̄ is the free-surface residual that a fixed lid cannot absorb, spread evenly over the
  column. The load limit is met by clipping |w| to `weather.w_edge` in the floor and surface field cells, counted per
  step.

**Clouds** are diagnosed, not prognostic, for rendering and shading:
- in each column, condensate fills the layers from the condensation level up to a top set by C and w;
- an anvil spreads with the upper-layer divergence;
- rain falls beneath.

### 3.3 Engine rules

- **Off by default.**
  - A hashed `weather.enabled`, false, in `POST_V01_KEYS`. It is dims-agnostic: in 2D it uses only the horizontal
    wind.
  - With it off every trajectory is unchanged. The weather fields enter both `State` and `State3`, so `state_digest`
    and the checkpoint key sets change for every run. The weather-off identity checks therefore use frozen field
    lists:
    - the 2D golden and digest matrix use `legacy_state_digest` (`V0_FIELDS`) and do not move;
    - in 3D, whichever of G4 and W1 lands first freezes `State3`'s field list as it then stands (49 names today) in a
      `legacy_state_digest_3d`. `GOLDEN_3D` is then not re-pinned and still proves the weather-off and lever-off paths,
      and GAMES §5.4's compatibility note is updated to match.
- **Old checkpoints.** `state_from_arrays` fills the weather fields of older 2D and 3D checkpoints with the rest state
  (h = H₁, H₂; u = 0; Q = Q₀; P = 0; D_c = 0), so a fork can turn weather on. Test this on a `runs/64k3d_s0` checkpoint.
- **RNG.** Seed disturbances and the hazard draw take a reserved stream, `Stream.WEATHER` = 9, drawn every step with
  weather on and split into (k_disturb, k_hazard). `RNG_LAYOUT_VERSION` becomes 3, with {1, 2, 3} compatible, as
  `CATASTROPHE` did (DESIGN §14.1).
- **Purity.** Pure MLX: gathers and rolls only, scatters only in `ops.py`, fixed iteration counts, no float32 GPU
  matmul (F20). Float32 fields are deterministic per backend, like the chemicals (DESIGN §13.3).
- **Load limits** (new validation rules):
  - with `α_max = α₀ + α₁` (`buoy_cap ≤ 1`, §4), validate requires `α₀ ≥ 0` and `α_max ≤ 1`;
  - the sampled agent wind is clipped to the hashed `weather.u_max` (‖u_h‖) and `weather.w_edge` (|w| in the floor and
    surface cells);
  - the non-penetration rule becomes `thrust_max + max(g + α_max·drag·w_edge, bmax − g + α_max·drag·w_edge) <
    2r·k_repel`. In 64k3d at α_max = 1 that means w_edge < 0.035;
  - the speed rule becomes `(thrust_max + α_max·drag·u_max)/drag ≤ v_max`;
  - W1 and W2 report `max_occupancy` and cell overflow under storm forcing, since converging wind piles agents. Keep
    C = 48, with 64 as the fallback.
- **Moisture budget.** Moisture is conserved to float rounding by the flux form. It is not energy.

## 4. Coupling the weather and the organisms

| direction | mechanism | hook | ledger |
|---|---|---|---|
| weather → organisms | **wind**: a drag force `F_w = α·drag·u_air(pos)` inside `forces3`, not contact-attenuated. Then `v = F/γ`: grip shelters grounded agents, and the `v_max` clip and the §2.2 invariants hold | `physics.forces3` | none, like gravity |
| weather → organisms | **heritable wind coupling**: `α = α₀ + α₁·buoy_cap`. Riding updrafts then depends on the lift trait and can be selected; with α independent of the cap, free lift would make a low cap cheaper | as above | none |
| weather → organisms | **calibration**: against the measured speeds (0.024–0.034 per step). Background wind about 0.01 or less outside storms; horizontal storm cores up to 0.1–0.2 (≤ `v_max` − `thrust_max`); vertical within the load limits | W0, W1 | — |
| weather → organisms | **rain-fed food at the cloud**: `regen_rate(patch_pos, cfg, step, env, rain=P, cloud=D_c)` adds κP to the field layers that span the interface h₁ (the condensation height), remapped 2:1 from the weather grid, plus an optional floor term. `K_est` and the floor `K` call it with the stored fields, so they rebuild exactly the regeneration injected and the safety rules stay calibrated (tested) | `fields.regen_rate`; `metrics.carrying_capacity` | `INJECT_REGEN` |
| weather → organisms | **cloud shade**: the same call dims the photic term by D_c | the fertility profile | `INJECT_REGEN` |
| weather → organisms | **storm hazard**: a kill probability from wind speed above a threshold, drawn from `Stream.WEATHER`. Its own mask goes to `lifecycle.deaths`, independent of the director. Storm deaths are excess deaths only, and a row in both masks dies as `CATASTROPHE`. `DeathCause.STORM` = 4. Counters are `CounterIdx` (27), then `Counter3D` (3, in 3D), then `DEATHS_STORM` with weather on: `n_counters(nd, weather) = 27 + 3·[nd = 3] + [weather]`. `interval_row`, the viewer and bench take the flag, so weather-off rows and widths are unchanged. Bond breaking is explicit (OR with wind above a threshold): storm-scale shear is far too weak to stretch a bond | `lifecycle`; `state.n_counters` | the corpse path |
| organisms → weather | **rain seeding**: c2 averaged over the cloud layers (the field layers spanning h₁, which also get the rain-fed food), taken as its excess over that band's lattice mean and upsampled to the weather grid; through a saturating response it lowers Q_s by at most a fixed factor | the condensation step | none. Emission costs `floor(20·Σemit)` in `HEAT_ACTION`, which is 0 below Σemit = 0.05 while the deposit is up to 12 quanta per channel, so a little seeding is free. W2 calibrates κ and the response so that sub-threshold emission alone does not trigger rain-out |
| organisms → weather | **shade on the seas**: dense populations lower T_s or the evaporation rate below them, so there is less moisture, fewer storms and less rain-fed food. That is a negative loop. Full Daisyworld regulation also needs a growth response that peaks in temperature (Watson & Lovelock 1983), which is later work | the evaporation step | none |

- **Seeding.** c2 already carries a seed-dependent background: a mean of 4.8–6.6 in seed 0 and 0.2–1.4 in seed 1 over
  300k–500k, and about 6 in seed 2 at 300k, against a clip of 16. The excess is used for that reason. κ is calibrated on a
  still-air control of each seed, and W2 reports c2 emission per clade, not the emitter share (0.74–0.92 in every
  run). c2's range is about 6 cells, so the effect is regional.
- **Who benefits from seeding.** The rain-fed food appears in the cloud layers, so seeding pays only for agents that
  live there. The benefit is also shared, which invites free-riders.
- **Senses.** Wind is felt through each agent's own velocity (`IN_VEL`, `IN_VEL_UP`) and nuclei through the chemical
  inputs. Moisture is not sensed without a dedicated input, which bumps the genome layout of each world that gets it:
  `GENOME_LAYOUT_VERSION` in 2D (which also shifts the 3D tail), or `GENOME_LAYOUT_VERSION_3D` for a 3D-only input. A
  dedicated nuclei channel (NCH = 4) changes the shared `layout.NCH` and bumps both.

## 5. The aerial biosphere

- **What weather adds.** Mid-water is empty where food is scarce, and the photic band already supports a surface
  clade floating on lift. Weather adds moving, patchy food aloft (rain and shade) and vertical transport.
- **Roles to look for** (the first three are Sagan & Salpeter's; the rest are this design's mapping):

  | role | how it lives | traits |
  |---|---|---|
  | sinker | small and fast-breeding; grazes the sunlit band before falling out | low cap, frequent budding |
  | floater | holds altitude in updrafts and grazes aloft | high cap and α, buoyancy tied to the updraft it feels |
  | hunter | patrols storm walls, where the hazard scatters prey | thrust, attack, wind tolerance |
  | scavenger | lives on the floor on falling corpses (4% of regeneration as "marine snow") | benthic |
  | storm-rider | lives in cloud layers and seeds the rain that feeds it | emission on c2, high cap |

- **The test that it works** (W2), at least two seeds per arm, against two controls:
  1. still air with the same mean food profile;
  2. passive tracers that an observer moves at `α_t·u_air(pos)`, where α_t = α₀ + α₁·cap of a randomly sampled live
     agent (resampled at each seeding), with the same floor and surface clamps and the same clipped wind;
  3. an arm with α₁ = 0, where the wind coupling does not depend on the cap.

  Adapted agents should sit at updraft boundaries more often than tracers and hold altitude relative to w. The cap is
  reported, but it is not the criterion.
- **Limits.** Swarms will be density clusters, not bonded bodies: bodies are adhesion sheets today. Behaviour must be
  read relative to the wind.

## 6. The game layer

### 6.1 From `docs/GAMES.md`

**Carries over:**
- the observer contract and labels (G1, built, draft);
- the intent pipeline (GAMES §5.3);
- the environment-only director with logged, replayable actions (DESIGN §26);
- `GAME_SPECS` pacing, era changes by fork, and the stream extensions (GAMES §5.5, §7.4, §3). These are designed, not
  built. The wall-clock figures double at 14.4 steps/s: a rule boundary every 17 s instead of 8.3.

**Re-skinned:**

| GAMES unit | storm-game unit | change |
|---|---|---|
| archetypes | grazer, hunter, drifter, floater, sinker | `ObsIn` gains height and buoyancy use; speed and persistence become wind-relative (an `INTENT_VERSION` and `OBSERVER_FORMAT` bump, in W2) |
| kin groups, houses | lineages | names only; avoid "species" (asexual budding, neutral tags) |
| colonies | swarms | density clusters, matched by footprint |
| tracked systems | storms | G2's event model (born, merged, split, dissolved). Matching is on cell footprints at a cadence where a storm moves less than its radius (from W0's drift), or by a TITAN-style assignment on centroid distance and size. The labeller runs on the host on the 2D maps |
| intent vector (GAMES §7.1) | a parallel storm schema (warmth, wetness, wind, seasonality, storminess, biomass, predation, disaster; targets `storm` and `lineage`) | same wrapper and compiler pattern |

### 6.2 Player verbs

| verb | lever | new work |
|---|---|---|
| name, track and forecast storms | observers | storm tracker; forecast records and scoring |
| warm or cool a sea, move a warm current | — | `env_warm_pool` (centre, gain, drift), timed and reverted |
| seasons | `set_seasons` (global phase) | opposite hemispheric seasons need a phase from sign(f(y)) at each warm pool (`per_patch` staggers by patch index, not by hemisphere) |
| insolation | `regen_pulse` as a food-only proxy | `env_insolation` |
| cloud seeding | — | `env_seed` (a regional rain multiplier) |
| steer or spin up a storm | — | `env_wind_impulse` (centre, vector, radius, duration) |
| lightning strike | `catastrophe` (a column in 3D) | none; a direct power |
| introduce organisms | `immigration_pulse` | archive immigrants (unbuilt) |

The new levers go into a `STORM_MENU`, following the `GAME_MENU` design of GAMES §5.4 (G4, unbuilt), so that the
research `MENU` stays byte-identical.

### 6.3 Goals and loops

- **Forecasting as a skill.**
  - Score track error at fixed lead times against the tracker's realised centroids. Draw the cone from the player's
    own past errors (NHC sizes its cone to hold two thirds of five years' errors).
  - Score event forecasts with the Brier skill score against climatology.
  - The reference to beat is the extrapolated track (TITAN-style). Only storms the player has not touched since the
    forecast are scored, which `commands.jsonl` shows. Replay is bitwise, so scoring can run after the fact.
- **Stewardship.**
  - Keep the biosphere viable through storm seasons.
  - Storm mortality will trip the extinction-risk flag. Rules version 2 reads the forced fall from the analytic season
    map and cannot cover emergent storms, so a new rules version would use the fall net of logged storm deaths, and
    capacity from the regeneration actually injected.
- **Discovery.** A storm bestiary over intensity, size, lifetime, track and spin, on the director archive's novelty
  grid. New lineages and adaptations come from the observers.
- **Two clocks.**
  - Storms are paced by the weather parameters and can run at any step rate.
  - The biosphere's clock is fixed in steps. At 14.4 steps/s a lifetime is about 66 s and there are about 6
    generations per 10 minutes, against 13 at 30 steps/s. Lineages therefore adapt across sessions.

### 6.4 Presentation

- **Framing.**
  - The camera sits inside the atmosphere. x and y wrap, so the client tiles the domain out to a fogged horizon and
    never shows the box.
  - The floor is drawn as the planet surface, with seas and warm currents from T_s, and the lid as haze.
  - The strategic view is a planet map built from the 2D weather maps, with both hemispheres and the equators
    (y = 0 and L/2) marked.
- **Stream records** (compatible within protocol v1, DESIGN §27.11):
  - `VOL1`: the diagnosed cloud and rain volume, u8 per cell. That is 128 KB per scalar per frame on 80 × 80 × 20, or
    about 11.5 MB/s for three scalars at 30 fps (est.). Alternatively, 2D maps of condensate, cloud-top height and rain
    rate, from which the client builds the volume.
  - `STRM`: storm id, name, centroid, extent, intensity, track and cone.
  - `EVNT`, `IDNT` and `INTN` as in GAMES §3.
- **Client.** A volumetric cloud renderer (Horizon Zero Dawn's clouds budget about 2 ms of GPU), rain and wind
  tracers as client particles, organisms as swarms of lights, and lightning from events. The current Swift viewer
  draws spheres in a box, so this is new client work.
- **Faithful view.** The director runs in the viewed loop, as in the `alife play` process (GAMES §3.1).

### 6.5 Playability and the GUI

Owner direction (2026-10-19): a Civilization-like cadence in a high-autonomy world. The player observes regularly,
gives few inputs and watches the consequences unfold. The narrative observer and world reports are core features. The rest of this section is the coordinator's choice under the
owner's delegation of 2026-10-19, to revisit (cadence, the GUI architecture, direct powers in G3).

| Civilization | storm world | at 14.4 steps/s |
|---|---|---|
| turn | chapter of 6,000 steps: report, influence refresh, save point; optional auto-pause | 6.9 min |
| end-turn summary | World Report; a template ticker line per 250-step snapshot | chapter end; 17 s |
| advisors | code predicates over vitals, tracks and events, worded by the narrator; never causal | — |
| notifications | alerts raised by the world, each with a place to fly to; auto-pause per class | event-driven |
| eras | seasons; era changes by fork at a chapter boundary (GAMES §7.4) | 2.3–21 min |

- **Input.** One influence-costed decision per chapter, at most three alert-driven moves, forecasts optional.
- **Time controls.** Pause, slow (0.25–0.5×, interpolated), 1×.
  - At 14.4 steps/s, 1× is the maximum, so fast-forward needs §9.8 headroom.
  - Time-lapse is playback of recorded, decimated frames (GAMES §8.2), not simulation.
- **World Report** (N1 offline after G2; live in G3; storm facts in W3).
  - Code builds at most 40 facts: validated labels, social events, vitals, logged entries, storm tracks.
  - The model writes placeholders, never numbers or names. A versioned checker drops failing sentences, with a
    template as the fallback.
  - No model text is fed back. Reports go in `reports.jsonl`, and replay never calls the model.
  - Until storm-world labels are re-validated, only exact counts, events and storms are narrated.
- **Consequences.**
  - Exact counts by cause, kin group and region, from device bincounts. Death records are capped at 2,048 per step.
  - Markers at `step_applied`, worded "since", never "because"; a storm × kin impact matrix (W3).
  - After G4, a counterfactual twin: a fork without the entry, exact for this world, opt-in, run while paused or
    between sessions.
- **Direct powers** (G3: the built regional actions; W4: the storm verbs).
  - A player power grid lands them within 2 s, not at rule boundaries.
  - Each has a short-cooldown `GAME_SPECS` row and stays outside the intent pipeline.
- **Cast.** All live storms, the top 5–8 lineages (kin groups named after their dominant house) and one focused
  organism, handed off at death to its eldest descendant or nearest kin.
- **GUI.** One window with two areas; the research window stays as the Lab.
  - The world view is the Metal view. It zooms from the planet map to a storm, a swarm and an organism. Storms,
    lineages, swarms, organisms and warm pools are clickable, with follow modes and cones.
  - The control center docks left, right or bottom. It holds the time controls, the intent prompt with preview and
    confirm, the verbs, the chronicle, reports, alerts, forecasts and scores.
- **Data path.** One `alife play` process. Presentation-only lines never change the trajectory, and picking is a client
  hit test.

  | channel | content |
  |---|---|
  | stdout, ALSTRM v1 | `FRM1` and alive-only frames, `IDNT`, `INTN`, `TERR`, `EVNT`, `STRM`, `VOL1`; proposed `MAP1` (weather maps), `ENVS` (effects, cooldowns, warm pools), `PLAN`, `ALRT`, `RPRT`, `FOCS` |
  | stdin, ≤ 4,096 B per line | §27.11 commands, plus `intent`, `confirm`, `cancel`, `power`, `forecast`, `focus`, `rate`, `alerts` |
  | files | `commands.jsonl`, `director.jsonl`, `reports.jsonl`, `observers/social.jsonl`: history and replay |
- **Measured before W4.**
  - W0 steerability: track deflection per impulse, and the storm response to a warm-pool change.
  - The combined 3D play-loop bench: step, observers, weather, stream, client rendering, and an LLM call in flight.

## 7. Risks

| risk | mitigation |
|---|---|
| Storms do not form. Showman's forced shallow water gives vortices only at deformation radii below 0.03 planetary radii, which this grid cannot resolve (the torus acts as a sphere of radius L/2π ≈ 51 units); it predicts jets otherwise, so B relies on the moist-convective feedback | W0 with exit criteria (§8), and model A as the fallback |
| Converging wind piles agents, overflows hash cells or breaks non-penetration | the load-limit rules (§3.3); tapered w; occupancy reported in W1 and W2; C = 64 as the fallback |
| Organisms only drift | a weak background wind; α tied to the cap; analysis relative to the wind; the tracer control |
| 3D speed (14.4 steps/s) | storms can be paced at any rate; for the biosphere's clock, re-derive `GAME_SPECS` at 14.4 steps/s, or use a 16k3d world (not benchmarked) or the fused kNN kernel (§9.8). C = 24 is ruled out: still-air occupancy peaks at 37–39 |
| Feedback runaway | the weather's own damping and a cap on the seeding effect, tested in W0 against a prescribed nuclei field; the negative shade loop. Director rules act only on the population |
| Engine regressions | everything off by default; a frozen legacy 3D digest; inert fill for old checkpoints; the 2D digest matrix |

## 8. Plan

Each step is one PR, merged before the next.

1. **W0, offline prototype.** No engine change: a standalone MLX script with passive tracers.
   - **Status (2026-10-19):** every criterion but the cost (1.116 ms) passes; see [`docs/STORMS_W0.md`](STORMS_W0.md).
   - **Exit criteria** (accepted by the owner as written, 2026-10-19):
     - 2–6 storms per hemisphere at any time;
     - median storm lifetime of at least 3,000 steps;
     - extrapolation track error at a 1,000-step lead of 0.5–2 storm radii, so forecasting is a skill but not hopeless;
     - peak tracer density below C per hash cell, with tracers on exactly the agents' wind (clip included), reported
       for the floor, mid-column and surface bands separately;
     - at most 1 ms per step on the M5 Pro;
     - storm latitudes reported (clustering at L/4 and 3L/4).
   - **Deliverables:**
     - parameters, storm drift per 250 steps, movies and the cost;
     - the wave speeds c_ext and c_bc, and the Froude number u/c_bc and Δh₁/H₁ in storm cores. The 0.1–0.2 core winds
       of §4 need c_bc of at least 0.2–0.4;
     - the maximum (u₁ − u₂)²/(g′(h₁+h₂)), which must stay below 1, the long-wave hyperbolicity bound;
     - wall-clock at 14.4 and 30 steps/s; the storm-to-organism lifetime ratio; steerability (track deflection per wind
       impulse; genesis or displacement lag after a warm-pool change).
   - **Fallback:** A, if B fails within a fixed tuning budget.
2. **W1, engine weather layer.**
   - `weather.enabled`, the fields, `Stream.WEATHER`, wind as a drag force, rain into regeneration and `K_est`, and the
     hazard.
   - The frozen legacy 3D digest, inert fill, the load rules, and tests.
   - A DESIGN section and a Review-log row.
3. **W2, couplings, observers and runs.** Seeding, shade, the wind-relative observer fields, and two seeds against
   still-air and tracer controls. Results document.
4. **W3, storm tracking and forecasting** on G2's event model.
5. **W4, storm levers and intent** on G4's pipeline (`STORM_MENU`).
6. **W5, client.** `VOL1`, `STRM` and the volumetric view.

**Interleaving with GAMES G2–G5** (§9.1 decided for the storm world): G2, W0 with N1, W1, W2, G3 (the play loop on the
storm world, with `STRM` and a 2D weather map as the first client), W3, G4 with W4 (one golden-safe change), W5, then G5.
W0 has no engine code, so it can run alongside G2. If W0 fails and model A also fails its budget, §9.1 comes back to
the owner.

## 9. Decisions (owner)

Decided 2026-10-19: items 1, 2, 3, 7 and 9. Items 4, 5, 6 and 8 are open; they are needed by W1 (4, 5), W2 (6)
and G3 (8).

1. **Which world G3 onwards target.** *Decided 2026-10-19:* **(a)**.
   - **(a) The 3D storm world** (gated by W0). It reopens:
     - GAMES §10.7 (64k 2D) and §10.4 (the case for `attack_mult` rests on 2D's thin predation; 3D's kill share is
       0.50);
     - G1's deferred 3D signal tools;
     - G3's face-on HUD;
     - the rate question of item 8.
   - **(b) Keep 64k 2D for G3–G5**, with storms after G5.
   - **(c) Weather on 2D first**, since W1 is dims-agnostic: storms and steering at 59 steps/s, with no floaters, as a
     staging step.
2. **Weather model.** *Decided 2026-10-19:* B. The options were B, C (3–6 ms, real towers, more tuning) or A (scripted).
3. **Geometry.** *Decided 2026-10-19:* a torus with sinusoidal rotation and two hemispheres. The alternative was one
   f-plane where every storm spins the same way.
4. **Weather energy.** Through regrowth and the corpse path only (*recommended*), or storm damage as a 17th ledger slot
   (state, checkpoints and goldens change).
5. **Senses.** Wind through velocity and nuclei through c2 (*recommended*), or dedicated wind or moisture inputs (a
   genome-layout bump).
6. **First couplings.** Rain seeding and shade (*recommended*), plus temperature-peaked growth later.
7. **Heat sources.** *Decided 2026-10-19:* a separate warm-pool map T_s with its own levers, so storms do not simply
   form over the food patches, and food follows rain. The alternative was the fertility patches (no new lever, but food
   and storms cannot be set apart).
8. **Target step rate for the storm world.** 14.4 steps/s with `GAME_SPECS` re-derived, the fused kNN kernel first,
   or a smaller 3D world. C = 24 is ruled out by occupancy. Restated as wall-clock targets (§6.5): forecast feedback
   within 15–70 s, storm arcs of 3–10 min, lever response within 1–2 min, a power lands within 2 s, fast-forward of at
   least 2× where offered. The fused kNN kernel is tried first, a smaller world only if it misses; decided before W2
   (coordinator's choice under the owner's delegation of 2026-10-19, to revisit).
9. **Weather grid.** *Decided 2026-10-19:* 160 × 160 at 2 units. The alternative was the 80-cell field lattice.

## 10. Sources

- **Aerial life.** Sagan & Salpeter 1976, *ApJS* 32:737: https://ntrs.nasa.gov/citations/19770038871.
- **Moist shallow water:**
  - two-layer form: Lambaerts, Lapeyre, Zeitlin & Bouchut 2011, *Phys. Fluids* 23:046603, https://hal.science/hal-00724559;
  - one-layer form and bulk humidity: Lambaerts, Lapeyre & Zeitlin 2011, *JAS* 68:1234,
    https://www.lmd.ens.fr/glapeyre/papers/jlglvz2011.pdf;
  - Rostami & Zeitlin 2018, *QJRMS* 144:1450: https://doi.org/10.1002/qj.3292;
  - O'Neill, Emanuel & Flierl 2015: https://dspace.mit.edu/handle/1721.1/100773;
  - Showman 2007: https://experts.arizona.edu/en/publications/numerical-simulations-of-forced-shallow-water-turbulence-effects-/.
- **Fluids:**
  - Stam 1999: https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/ns.pdf;
  - Fedkiw, Stam & Jensen 2001: https://graphics.stanford.edu/papers/smoke/smoke.pdf;
  - Dobashi 2000: https://graphics.stanford.edu/courses/cs348b-competition/cs348b-05/clouds/index.html;
  - Yanagita & Kaneko: https://arxiv.org/abs/chao-dyn/9609011.
- **Feedback.** Daisyworld, Watson & Lovelock 1983:
  https://www.jameslovelock.org/biological-homeostasis-of-the-global-environment-the-parable-of-daisyworld/.
- **Tracking and forecast scoring:**
  - TITAN: https://impacts.ucar.edu/en/publications/titan-thunderstorm-identification-tracking-analysis-and-nowcastin/;
  - NHC cone: https://www.nhc.noaa.gov/aboutcone.shtml;
  - Brier score: https://en.wikipedia.org/wiki/Brier_score.
- **Games:**
  - *Storm Chasers*: https://store.steampowered.com/app/862800/;
  - *SimEarth*: https://en.wikipedia.org/wiki/SimEarth;
  - Horizon Zero Dawn clouds: https://www.guerrilla-games.com/read/the-real-time-volumetric-cloudscapes-of-horizon-zero-dawn.
