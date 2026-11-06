# Playability brainstorm: planets side by side, a sun, meteors (2026-11-06)

Owner's prompt (2026-11-06, while away): "Brainstorm on additional configurations or new designs that emphasize
playability. Some ideas in my mind: planets side by side, a visible sun around which one or several planets orbit with
the ability to switch from one planet to another (this implies smaller planets given compute limits on a local
machine), meteor impacts as a catastrophic event."

This is a brainstorm, not a spec: nothing here is decided or built. Section 7 lists the decisions that are the owner's.
It builds on `docs/brainstorming-Fable-November.md` §5 (topographies; long winters from an eccentric orbit) and §9
(day and night first), and on what exists: the diel light cycle (`DESIGN.md` §49), the player powers (§34), the sphere
weather (§46.6) and the bench of 2026-11-06.

**Summary.**
- A Mac M5 Pro plays **one 128k3s planet, two 64k3s, or four to six mini3s at the 1× rate** (14.4 steps/s); 256k3s and
  576k3s are below 1× on their own (§1).
- The cheapest large gain is **making the sun visible**: the engine already has a moving sun (diel `terminator` mode),
  but the app lights the globe from over the camera's shoulder. Showing the real terminator, a sun disc and the night
  side is mostly app work (§2).
- **Meteors** fit the existing machinery: a graded Strike plus a crater field, an impact winter and a storm kick, with
  a warning the player can act on (§3).
- **A small solar system** (three to five mini3s-sized planets in lockstep, a stylised orrery, fly-in to switch) is
  affordable in GPU time, but it is the largest item: the app plays one world at a time today, and off-screen planets
  have no offline advance (§4). Meteors that carry organisms between planets (panspermia) are what make it a system
  rather than tabs.
- Recommended order: the visible sun, then meteors, then orbit and seasons, then twin planets, then the system (§6).

---

## 1. The budget: what one Mac can play

Measured on the M5 Pro with Metal (DESIGN §46.10 for the small planets; the bench of 2026-11-06 for the large ones,
storms on, quiet GPU):

| Planet | Organisms (cap) | ms/step | Steps/s | Peak memory | Save |
|---|---|---|---|---|---|
| tiny3s | 1,024 | 4.9 (features + weather) | 206 | < 0.1 GB | small |
| mini3s | 20,480 | 6.8 (at 16k rows) | 148 | 0.3 GB | small |
| 64k3s | 65,536 | 21.6 (all features + weather) | 46 | 1.3 GB | ~0.7 GB |
| 128k3s | 131,072 | 42 (55k to 83k alive) | 24 | 6–12 GB | 1.3 GB |
| 256k3s | 262,144 | 90–98 (110k to 162k alive) | 10–11 | 12–22 GB | 2.6 GB |
| 576k3s | 589,824 | 233 (219k to 119k alive, founders) | 4.3 | 26 GB | 5.8 GiB |

Reading it:
- **GPU time is the budget, not memory**, below 128k3s. A step has a fixed cost of about 2.7 ms plus about 0.25 ms per
  thousand organisms (mini3s fit, §46.10), so a tiny planet is launch-bound.
- At the 1× rate a world needs `14.4 × ms/step` ms of GPU per second: 64k3s about 310 ms, mini3s about 100 ms. So one
  second of GPU holds **about three 64k3s planets or eight to ten mini3s** in theory. Several processes contend for the
  GPU, so the real figure is lower; it has not been measured (§5, check 1).
- 256k3s and 576k3s cannot be played at 1×: they are watch-and-wait worlds. The extra-large ladder answers the owner's
  question of 2026-11-05: **128k3s is the largest planet that plays at 1× on this Mac**.

So "smaller planets" for a system means mini3s-sized (about 20k organisms each), or a new preset between mini3s and
64k3s (for example 32k, `R` about 64), which would need its own storm calibration (§5, risk 2).

---

## 2. A visible sun

**What exists.** The diel cycle (§49, `src/alife/diel.py`) has a `terminator` mode on the sphere: the subsolar point
sits on the equator and moves once around the planet per `diel.period` steps; the light `L = max(0, cos(lat)·cos(lon +
θ))`-like term multiplies the photic food term only (no ledger term; food stays `INJECT_REGEN`). Long Night is a
power (§49.8). The research pilots found no diel migration (§49.14), but the light cycle itself is built, cheap and
visual. **The app does not show it**: the sphere renderer lights the globe from a light over the camera's left
shoulder (`SphereRenderer.swift:343`), and no Swift file reads the diel phase.

**Design.**
- **S1, show the engine's sun.** The stream carries the diel phase and period (a header key plus the step, which the
  records already have; a stream protocol change, kept in step by `tests/test_gui_sync.py`). The renderer computes the
  subsolar point from the step and lights the floor and the lid from it: a day side, a night side, a twilight band,
  a sun disc in the sky shader, and the camera light kept only as a dim fill so the night side stays readable. A
  "Show sunlight" switch keeps today's look available.
- **S2, the night side has life.** Organisms glow faintly at night (emissive term on the agent shader, presentation
  only), so the night side reads as populated, not empty. Calls (§39, §47) could flash, as the November brainstorm's
  "bioluminescence as the visible form of calls".
- **S3, a tidally locked planet.** `diel.period` at infinity in terminator mode: a permanent day side, a permanent
  night side and a habitable ring along the terminator (November §5, topography 1). One engine key (`diel.locked`, or
  a period of 0 meaning locked; off by default), and the weather pools placed along the ring. The ring is a
  one-dimensional habitat with excursions: migration and niche partitioning by light.

**Cost.** S1 and S2: app work plus one stream key, no trajectory change (presentation only). S3: one inert-when-off
engine key, its ExtSpec handling under §43, and a calibration of where the pools go.

---

## 3. Meteors

**What exists.** Strike (`catastrophe`, `lifecycle.py:85-108`) kills agents in a radial column with probability
`kill_p`, radius 20 to 80 in 1024-world units, capped on a sphere by C30's area scaling (at most 11.5 % of the
columns); it leaves food and fields untouched. Blight (`habitat_loss`) and Food (`regen_pulse`) change food. Every
power is a logged `director.jsonl` entry applied after the checkpoint at its step (rule P), so a played world replays
bitwise. There is no terrain: the floor is a fixed sphere.

**The event, as the player sees it.**
1. **Warning.** "A bolide is approaching: impact in about 3,000 steps near 23° N, 140° E" (an alert of the importance-3
   class, which pauses, §34). The approach is drawn as a moving point in the sky, brighter each chapter.
2. **Decision.** The player can spend influence on a response: *Deflect* (moves the impact point, costly), *Scatter*
   (a Food power elsewhere, to draw organisms away), or nothing.
3. **Impact.** A flash, a shock ring across the floor, then a dark crater.
4. **Aftermath.** A dust veil lowers the light planet-wide for some thousands of steps (an impact winter); the crater
   greens later and becomes fertile ground; the chronicle tells it ("the crater at 23° N greened; house Vell settled
   it first").

**The model, in tiers** (each tier is usable alone):
- **M1, no new state.** A graded Strike: kill probability falling with distance from the centre instead of a flat
  column. A Blight-like food removal inside the crater radius. A planet-wide light dip (the diel light multiplier, or a
  `regen_pulse` below 1) for the winter. Optionally a weather kick: heat and vorticity injected at the impact point, so
  an impact can spawn a storm. Energy through the existing paths (corpses for the killed, the existing food-removal
  term); this needs checking against the ledger before the build.
- **M2, a crater field.** A per-cell level, like the farming soil level (§41), that first lowers regrowth inside the
  crater and then, as it decays, raises it above normal (fertile ejecta): disturbance and succession, the classic
  emergent pattern of the November §5 "fire and succession" item. The floor shades by this field.
- **M3, ejecta to other planets** (needs §4): an impact copies a sample of the struck organisms' genomes into a
  pending import for another planet of the system (panspermia).

**Who throws them.** Three options, not exclusive: random world events from a seeded schedule inside the step (a pure
function of the seed and the step, so replay needs no log; `meteor.rate`, off by default); a player power *Meteor*
(a Strike with an aftermath); a director action. Random events give the world its own story; the power gives the
player a large, legible lever.

**A research claim to pre-register.** The intermediate disturbance hypothesis: diversity (houses, archetypes) peaks at
an intermediate impact rate. The rate is one key, so the experiment is a sweep.

**Cost.** M1: one engine action (inert when off), one power in `GAME_SPECS`, an alert class, app effects. M2: one
state field under §43 with its own exact identity, like the soil level. M3: after §4.

---

## 4. Several planets and a sun they orbit

### 4.1 What "side by side" can mean

- **A, twin planets in comparison.** Two worlds from the same seed and save: one the player plays, one left alone.
  Shown side by side, the difference is the player's doing; determinism makes the untouched twin exact. This is the
  evaluation's yoked control (§9 of DIRECTOR) as a game mode. Cost: 2× the planet.
- **B, a double planet.** Two bodies close together in one sky, each its own world. Mostly a rendering idea unless
  organisms can cross (M3).
- **C, a small solar system.** A sun and three to five planets on orbits, each a world with its own climate, the player
  flying between them.

### 4.2 The system (C)

**The planets differ by their place.** The orbit sets each planet's light and seasons, all as pure functions of the step
(no host sync):
- distance: the light amplitude (an inner hot planet with high light and high hazard; an outer cold one);
- axial tilt: the subsolar latitude moves as `δ(t) = ε·sin(2πt/Y)`, which gives seasons by latitude (the diel
  terminator with a moving latitude; today's diel has no tilt);
- eccentricity: a light amplitude that swells and fades over the year, which gives the November brainstorm's "long
  winters";
- one tidally locked planet (S3), one stormy giant, small calm ones (storms are fragile on small planets; §5, risk 2).

A year of, say, 30 days of 1,200 steps (36,000 steps) lasts about 42 minutes at 1×; a chapter (6,000 steps) is five
days. A year must span several generations for seasons to select anything; that needs measuring first.

**Time.** All planets step in lockstep on one system clock, so exchanges between them happen at fixed steps and every
planet replays from its own log. The alternative, freezing the planets that are not on screen, is cheaper but makes the
system a set of tabs: nothing happens elsewhere while the player watches one planet.

**Switching.** An orrery view (stylised, not to scale: the sun, orbit lines, the planets as small globes with their day
sides lit, the system clock), and a flight into a planet (the existing great-circle Home flight, extended from the
system view). The chronicle and alerts are per planet with one system stack: "Ilen: a storm named; Taro: population fell
22 %".

**Exchange between planets.** Panspermia by meteors (M3): an impact on one planet sends a sample of genomes that
arrives later on another as immigrants with those genomes. The engine has immigration (random founders) but no genome
import, so this is one new path: an import pulse whose genomes are logged in the receiving planet's `director.jsonl`,
so it replays. A player power *Seed* (send organisms from a living planet to a dead one) uses the same path. Founder
effects and divergence across planets become visible as palettes drifting apart.

### 4.3 Architectures

| | One process per planet (A1) | One process, a planet axis (A2) |
|---|---|---|
| Engine change | none for the planets; a parent process (or the app) runs N `alife play` children in lockstep | the state gains a leading planet axis; per-planet scalars become array inputs (CLAUDE.md, F21 item 6) |
| Planet sizes | any mix | one shape for all (same preset), padded |
| GPU efficiency | N fixed costs, contention | one fixed cost for all; best for small planets |
| Risk | GPU scheduling between processes is unmeasured | a deep engine change: every per-world array, the spatial hash, the weather, saves, streams |
| Fit with today's rules | one compiled configuration per process holds | holds if the differing scalars are arrays |

Recommendation: **A1 first**. It needs no engine change, keeps every existing guard, and the bench of §5 check 1 says
whether it is fast enough. A2 only if the measured contention is bad and the system wants many small planets.

**App work** is the largest part either way: the app plays one world at a time (APP R5, `play.lock`), refuses to watch
a second world beside a played one (APP.md:644), and its stream carries one planet with no world id (STREAM_V2). A
system needs a multi-world host, one stream per planet with a planet id, and the orrery scene.

---

## 5. Checks and risks before any build

Checks (each a few hours, no code):
1. **Contention.** Two and four mini3s `alife bench` processes at once, then two 64k3s: steps/s each against alone.
   Decides A1 against A2 and the number of planets.
2. **Generation time against the year.** Median lifespan and generation interval on mini3s and 64k3s from existing
   runs: sets the day and year lengths.
3. **Light against food.** How much the photic term moves the food on a sphere at full and half light (the diel pilot
   data has this): sets how far apart the inner and outer planets can be before one starves.

Risks:
1. **The diel pilots were negative** (§49.14): light cycles did not evolve migration. The sun is proposed here for play
   and legibility, not as a research claim.
2. **Small-planet storms are fragile.** mini3s and tiny3s sit on a knife edge (SPHERE.md S4c: the planets are smaller
   than the weather's deformation radius). Weather on every planet of a system means a calibration each; small planets
   may simply be calm.
3. **Meteors and the ledger.** Food removal and the crater's later fertility must go through existing ledger terms;
   M2 needs its own exact identity, as the soil level has.
4. **Replay.** Every trajectory change (impacts by the player, imports between planets) must be a logged entry or a
   pure function of the seed and the step; presentation (the sun disc, the orrery, the glow) must leave the digest
   unchanged (`tests/test_play_purity.py`).

---

## 6. A possible order

| Step | Content | Kind | Rough size |
|---|---|---|---|
| 1 | Visible sun (S1, S2): the engine's terminator on the globe, a sun disc, night glow | app + one stream key | 1–2 PRs |
| 2 | Meteors M1: the event, warning, Deflect, impact winter, storm kick; power and alert | engine action + app | 2–3 PRs |
| 3 | Orbit and seasons for one planet: tilt, eccentricity, tidal lock (S3) | engine keys + app | 1–2 PRs |
| 4 | Meteors M2: crater field and succession; the disturbance sweep | engine state + experiment | 1–2 PRs |
| 5 | Twin planets (A): two worlds side by side, one played | app multi-world host | 2–3 PRs |
| 6 | The system (C): lockstep parent, orrery, switching, panspermia (M3, Seed) | app + engine import path | 4–6 PRs |

Steps 1 to 3 each give a visible change on one planet with no multi-world work; step 5 builds the multi-world host that
step 6 needs, in a smaller form.

---

## 7. Decisions for the owner

1. **First:** the visible sun (step 1), meteors (step 2), or both as one "sky" series? *Recommended: both.*
2. **Meteors:** random world events, a player power, or both? With a warning the player can act on? *Recommended: both,
   with a warning.*
3. **Off-screen planets:** keep stepping in lockstep, or freeze when not shown? *Recommended: lockstep, for up to four
   mini3s-sized planets, if check 1 confirms.*
4. **Planet size in a system:** mini3s as it is, or a new preset of about 32k (needs a storm calibration)?
5. **Panspermia** (organisms crossing between planets): in or out?
6. **The orrery:** stylised (recommended) or to scale?
7. **Checks 1 to 3 of §5:** run them now (measurement only, a few GPU hours), or wait?
