# Brainstorming with Fable, November 2026

A record of the brainstorming session of 2026-11-03 between the owner and Claude (Fable 5.1), after the one-app program
was merged (main `6b37340`, `alife` the only app). Context: two Perplexity "deep research" reports the owner shared, one
assessing the project's novelty against prior work (Division Blocks, ALIEN, JaxLife, Polyworld) and one proposing
commercial iterations of "a reproducible artificial-life instrument with exact energy accounting and controlled
experiments". Nothing here is decided; it is a set of directions to revisit. No code was written.

---

## 1. The Hopf fibration idea (owner's prompt: an agent's text proposing fiber-invariance testing)

**Verdict:** the agent's text is right about the method and wrong about the star of it. Strip away the Hopf fibration
and what remains is *metamorphic testing with a group action* (Chen et al., 1998; "equivariance tests" in ML): pick
transformations that should leave the observable outcome unchanged, apply them, check invariants. Most of its five-layer
architecture already exists in alife:

| The agent's layer | alife today |
|---|---|
| Execution capture | checkpoints, config hash, git rev, RNG layout version, bit-exact replay |
| Behavioural equivalence | metrics, the §39 labels, standardized readouts |
| Structured intervention | forks with `--set`, the controls `nofar`, `deaf`, `decoy` |
| Invariant checking | the exact ledger, `full_check`, the diag bits |
| Regression preservation | golden digests, the identity matrix |

**Calls M1 was already a fiber intervention:** the `decoy` control keeps the energetic cost and message count fixed and
changes only the information content. The formalism describes what M1 did.

**Where Hopf structure genuinely exists in the project:**
1. The sphere camera is on S³ (`SphereCamera.swift` composes poses with `simd_quatf`). Unit quaternions double-cover
   SO(3); "quaternion → the point of the planet you look at" is the Hopf map, the fiber is the camera's roll. The bundle's
   nontriviality means a continuous "north up" cannot exist over the whole planet (no global section); the poles are
   where it must fail, which the owner's play notes and `keepNorthUp` already met. Lesson: store the pose as a
   quaternion, never a heading angle; treat north-up as a local preference with a declared seam.
2. Seasons are a literal S¹ (`season_amplitude · sin(2πt/P + φ_p)`). Forking lineages at several season phases is a
   cheap, real fiber test: does the storm-death contrast depend on when you intervene?
3. The weather is chiral (`hemisphere_sign`, vorticity): parity is a symmetry of the box world and not of the planet.

The agent's "normalized complex pair z ∈ S³ of agent phases" has no counterpart in the engine.

**The actionable program: a symmetry table and symmetry goldens** (also the "instrument qualification suite" the
second report asked for). Transform the initial state, run, transform back, compare digests; the identity matrix is the
g = e member.

| Transformation | Should be exact? | A violation means |
|---|---|---|
| Permuting agent storage rows | yes (scatters are order-independent) | an order-dependent conflict resolution |
| Torus translation (box) | yes up to rounding | a stage using absolute coordinates |
| HEALPix discrete rotations (planet) | yes for the lattice's own group | the float32 lookup or kNN breaking its geometry |
| Mirror image (box only) | yes | handedness leaking in |
| Sign flip / permutation of a hidden unit with its weights | yes, exactly (the Elman RNN's gauge group) | a brain depending on a representation it shouldn't |
| Global rotation of `tag` preserving pairwise similarity | yes | the kin cue reading more than similarity |
| Seed, season phase | no: statistical only | these define the ensemble, not the dynamics |

Caveats: Metal's reduced precision breaks some rows at the bit level even when the design is right, so each row needs a
written verdict ("exact", "within tolerance", "broken by design here"); the RNN gauge group is discrete, so mutation
almost never moves along it.

**Further applications:** canonicalize genomes before counting novelty (MODES, activity statistics), so representational
redundancy never reads as novelty; season-phase forks as a cheap robustness map; metamorphic testing as the
model-agnostic sellable version for external systems; "declare the seam" as a UI principle (the preset guide's captions
already do it); chirality as an experiment (do organisms on the chiral planet become chiral?).

**Not to do:** a general fiber-bundle framework. The value is the table and M1's decoy design generalized to each feature.

---

## 2. Commercial applications of the current approach

Marked by how much is already built; the recurring risk is bespoke integration.

**The instrument pointed at other systems**
- *Coordination audits for multi-agent AI deployments.* Replay a run, replace inter-agent messages with matched dummy
  traffic (the decoy control), hold the budget fixed, see whether outcomes change; add an exact ledger for tokens and
  tool calls so no agent spends a budget twice. Gap: LLM calls are nondeterministic; replay needs captured responses.
  The one with real pull now.
- *Yoked-random evaluation of AI decision-makers.* The director evaluation found the LLM barely beat a random controller
  on the same budget and cadence. "Does it beat a yoked random arm?" transfers to any system claiming it improves an
  outcome by intervening. Reuse: `director.yoke_log` and the evaluation protocol.
- *Chronicle-driven QA reports.* The narration layer, grounded in validated events, writes the bug report developers
  actually read. Gap: grounding labels in the customer's events was the hard part of G3.
- *A qualification suite for other agent-based models* (Mesa, NetLogo): inject a known accounting error, show the
  model's checks catch it. The smallest honest product.

**The app and engine as products**
- *A planet on your desk:* the live planet with chronicle as an ambient consumer app; modest revenue, but it pays for the
  App Store path (D2) and needs a starter world (D1 option C).
- *Reproducible worlds as shareable content:* seed + config hash reproduce the same planet anywhere; legends with
  verification. Gap: cross-hardware bit-identity is unproven.
- *Teaching scientific reasoning through play:* "correlation versus communication" built on M1 verbatim.

**Iterations of the approach**
- Pre-registration as a file format and tool (freeze, hash, amend, apply a stage rule).
- Blind sequential sample-size rules as a library (the stage rule used only the spread, never the effect).
- A counterfactual-replay API: "fork at any save point, change one thing, measure, compare" as one call.

**First test:** two conversations before any code: a team running LLM agents in production (decoy-message audit), and
a game studio with a systemic economy (a chronicle-style bug report mocked from their events).

---

## 3. The approach applied to computer science, engineering and economic modeling

Separate the approach into four transferable parts: (1) ecological selection without a score; (2) an exact conservation
ledger; (3) replay, forks, knock-outs, neutral shadows, pre-registration; (4) an environment-only director with a yoked
random control. Parts 2 to 4 transfer anywhere; part 1 only where the real system is also unoptimized (economies and
ecosystems more than engineered artefacts).

**Economic modeling, the natural home.** The engine is already an economy:

| alife | economy |
|---|---|
| exact energy ledger | stock-flow consistency (Godley and Lavoie) |
| organism, birth, death | firm or household entry and exit |
| no fitness function | evolutionary economics (Nelson and Winter): routines persist by survival |
| built structures that outlive the builder, with decay | capital goods and depreciation |
| farming the soil | infrastructure and public goods |
| arms races | competitive R&D, advertising, the Red Queen |
| voice, calls, decoy control | cheap talk, market signalling, and the test of whether it carries information |
| storms | exogenous shocks |
| director with a budget, ledger-checked | a regulator or central bank with constrained tools |
| yoked random director | placebo policy |

Products: a stock-flow-consistent evolutionary ABM with pre-registered policy experiments (central banks, regulators);
incentive-design testing for platform and token economies (exploits found before launch); market signalling under the
decoy test (modeling, not advice).

**Computer science:** ecological fuzzing (inputs as organisms paying execution time, niches as program regions; the
testable claim is diversity maintenance versus AFL's collapse; precedents Tierra, Avida); co-evolutionary red-teaming
that resists mode collapse (the arms-race layer against a classifier); cooperation protocols against evolved cheaters
(bonds that pay); a token and tool-call ledger for agent orchestration (part 2 alone).

**Engineering:** energy-aware controller synthesis (the smallest controller surviving a joule budget, auditable through
the ledger); resilience of networked infrastructure under shocks with a yoked random operator; emergent construction
for swarm robotics (built structures persisting beyond the builder).

**The enabling iteration:** a domain-agnostic substrate: an "organism" is any row with a conserved resource, a "world" any
space or graph; ledger, replay, controls, pre-registration and director stay. Caveat: the approach has so far produced
careful negatives more often than emergent brilliance; its value is that it tells the truth about emergence.

**Pick:** the economic ABM, because every piece of the engine already has a meaning there.

---

## 4. The economic ABM, applied to policymaking

**The model:** rows are households and firms; energy is money, the integer ledger closing exactly, money created only
by a declared source and destroyed only by a declared sink (stock-flow consistent by construction); food is demand and
inputs; structures are capital with decay as depreciation; soil is infrastructure; arms races are R&D; bonds are supply
chains and employment, roles are occupations; voice is prices and advertising; storms are shocks; seasons the exogenous
cycle; chapters are quarters and the chronicle writes the report. Nothing optimizes: routines are evolved controllers
that persist because they survived (Nelson and Winter, taken literally).

**The policymaker is the director:** a budget, instruments, cooldowns, actions checked against the ledger. Every study
has the calls shape: a pre-registered claim, forks from a common save point, a yoked random policy arm, independent
evolutionary runs as the unit, bands with an honesty threshold on the noise. Two controls transfer with force: the yoked
random policymaker (a policy that cannot beat random spending of the same budget has no business being targeted) and the
oracle ceiling (the perfect-information policymaker shows the best a policy could do on its target and what it costs
everywhere else; "hits the target, wrecks the economy" becomes measured). The neutral shadow separates inequality that
strategy produced from inequality that luck produced.

| Challenge | Already modeled by | Pre-registered question |
|---|---|---|
| AI and labour displacement | a cost-lowering capital good; roles; bonds as employment | Under which transfer rules does demand recover, and does any beat a random transfer of the same budget? |
| Concentration of compute and market power | niche construction as lock-in | Do interoperability or access mandates reduce concentration more than random intervention, at what cost to the investment that built the niche? |
| Industrial policy and subsidies | the director's budget with a ledger-forced source | Does targeted subsidy beat random subsidy on the same budget? |
| Tariffs and fragmentation | regions as planets or hemispheres; trade as far hearing with a distance cost; storms | Where on the efficiency–resilience curve does reshoring land, measured by survival through shocks? |
| Energy transition and the grid | energy is literal; seasons give intermittency; capacity is a structure with decay | Does a price alone build enough capacity before the shocks, versus mandates, versus random spending? |
| Inequality and dynasties | houses and lineages with inheritance | How much concentration is luck? The neutral shadow is the null; a wealth tax is tested against it. |
| Public debt and money | the ledger's source and sink, interest as a cost | Which financing of the same spending sustains the population through a shock? |
| Platform and attention economies | signalling under the decoy test | Does advertising carry information or only cost? |
| Programmable money | the integer ledger is a programmable currency with exact settlement | Policy rules on money itself, tested with no accounting ambiguity. |

**Claims:** conditional and mechanistic, never forecasts. Calibration through stylized facts as validation gates (heavy-
tailed firm sizes, size-independent growth, wealth distribution shape, cycle lengths), as G1 to G5 gated the pilot.
Expect mostly careful negatives; that rarity is the value, since most policy argument has no control arm.

**A first study:** "Targeted versus random industrial subsidy." Six evolved economies forked at a common save point; arms
none, targeted by rule, random with the same budget and cadence, oracle. Frozen readouts: survival through the next
shock season, output, concentration, fate of the subsidized firms. "Supported" only if targeted beats random with a
one-sided bound above zero after Holm, noise within limit; otherwise "no effect of interest", published as such.

---

## 5. The game studio angle: new topographies and dynamics

(Section written with the owner's prompt of the same date; see the reply in the session for the full text, reproduced
here.)

**Thesis: design edges, not areas.** Complexity in the current worlds appears at boundaries (floor and surface, the
storm bands, the shore of a food patch). Each topography below adds a new kind of edge, and each has a visual signature
and a pre-registrable claim.

Topographies:
1. *Tidally locked planet.* A permanent day side (hot, barren), a frozen night side, and a habitable twilight ring along
   the terminator; wind blows day to night, storms ride the ring. A one-dimensional habitat with two-dimensional
   excursions: a gradient drives migration and niche partitioning. Visual: a glowing ring of life around a dark world.
2. *Archipelago with rafts.* A height field on the sphere with a sea level; islands separated by water organisms cannot
   cross unless a built structure floats. Allopatric divergence visible as palettes drifting apart; a raft landing is a
   founder event for the chronicle. Sea level is a player power.
3. *Tidal shoreline.* A moon raises and lowers the sea on a cycle; the intertidal band floods and drains. Amphibious
   strategies and a resource pulse. Visual: a breathing coastline.
4. *Diel vertical migration.* A day–night light cycle on the existing photic food band and buoyancy: the whole population
   rises and sinks daily, as it does in Earth's oceans. Cheapest of all, and the most striking time-lapse.
5. *Long winters.* An eccentric orbit or a binary sun: seasons lasting generations, with a dormant state (spores, seed
   banks) that costs nothing to hold. The planet goes dark, then blooms. New dynamic: storage and dormancy strategies.
6. *Sponge world.* A porous planet: chambers joined by tunnels rather than open space. Refuges, chokepoints, sieges;
   structures that block tunnels become fortifications. Visual: a cutaway with glowing chambers.
7. *A planet that organisms build.* Structures accumulate into topography (reefs), which changes the flow of water and
   wind: niche construction at planetary scale, with the weather coupling back. Visual: the planet's shape changing
   over epochs.
8. *Ring world or cylinder.* Gravity outward, horizon curving up, the far side visible overhead, a shadow band sweeping
   round for day and night. Migratory waves chase the light.
9. *Möbius or projective wrap.* A lap flips handedness, so mirror-image lineages meet. Chirality becomes an ecological
   variable, which ties to the chirality experiment of section 1. Hard to visualize honestly; a research toy.

Dynamics that add emergence on any topography: currents and upwelling (advection of food and organisms; gyres and
hotspots); fire and succession (a disturbance consuming structures and soil, leaving fertile ground; the classic
emergent pattern); parasites and contagion along bonds (an immune arms race; waves of colour); decomposition and a
detritivore niche; bioluminescence as the visible form of calls.

Visualisation: epoch time-lapses; lineage rivers (a Sankey of the houses over time); heat maps of niche construction;
the view from inside a sponge world.

Priorities by cost against payoff: diel migration (uses buoyancy and the photic band, needs a light cycle), the
terminator ring (uses the sphere and weather), long winters with dormancy (one new state and a ledger term), the
archipelago with rafts (height field plus floating structures), fire and succession, then the sponge and the
reef-building planet, which need new geometry. Engine constraints: fixed shapes, pure MLX, a ledger term for every new
energy path, and the HEALPix lattice for anything on the sphere. Each needs a pre-registered claim of what emerges that
did not before.

---

## 6. A quantum substrate (owner: "could quantum physics be a new substrate for an iteration?")

Owner's steer first: diel vertical migration is the agreed easy next step; ideas 7 to 9 (the reef-building planet, the
ring world, the Möbius wrap) are of particular interest for later iterations.

**Three meanings, one real.** Running the engine on quantum hardware: no (a few hundred to a few thousand noisy qubits,
nothing maps). Quantum rules as a new physics for the world on classical hardware: yes (precedents: quantum cellular
automata, quantum walks, "quantum game of life" toys, Zurek's quantum Darwinism; none with evolving organisms).
Quantum principles as metaphors: keep the exact rules, drop the mystique.

**Feasible design: one-body quantum organisms.** Each organism is its own wavefunction over the lattice, no entanglement
between organisms by default (1,024 organisms × 3,072 HEALPix cells ≈ 3 M complex numbers). Movement is a quantum walk
whose coin the brain sets; food is a potential; eating is a measurement that collapses position, with the integer energy
transfer at that cell (the ledger stays exact there); death is absorption into a sink, a ledger term; reproduction
cannot copy (no-cloning), so mutation is a law of the substrate.

**New ecology from the rules:** ballistic spread (linear in time versus square root); the sensing tradeoff (sensing is
partial measurement and decoheres the walker: strong sensing is classical, slow and sighted; weak sensing is quantum,
fast and blind; evolution tunes the point); tunnelling as a third option for barriers (sponge walls, archipelago water);
interference between an organism's own routes; fermionic versus bosonic species (tiling versus clumping; a Bose–Einstein
condensation as a visible phase transition); bonds as entanglement for pairs only, with a Bell test as the limit of the
decoy control ("is the coordination real?"); reversibility, so rewind is physics rather than a save point.

**Experiments:** quantum versus classical walkers under the same energy budget on patchy food, with sensing costing
coherence (spreading is known; finding is not); where on the sensing–coherence line evolution settles, by patchiness.

**Costs:** norm conservation does not close exactly in floating point, worse under Metal's reduced precision (F20): either
the project's first ledger with a stated tolerance, or reversible integer automata (Margolus, Toffoli) that keep exact
reversibility but lose complex amplitudes; this choice decides the iteration. No entanglement beyond pairs. A
probability cloud is harder to read than a dot.

**Meets ideas 7 to 9:** HEALPix exists for spherical-harmonic transforms, so a wavefunction on the planet decomposes into
the modes of a hydrogen atom's orbitals, drawn on the globe; a ring world quantizes momentum into standing waves; a
Möbius wrap gives twisted boundary conditions and an Aharonov–Bohm phase per lap, so chirality enters as a phase rather
than a mirror; the reef-building planet becomes a potential that organisms sculpt.

**First step, if pursued:** one quantum walker on the existing planet, no brain, no energy: the walk, a food potential,
a glowing cloud with collapse flashes. A week decides whether the picture carries the idea.

---

## 7. Quantum rules for systems optimisation (owner confirmed "quantum rules on classical hardware" is the intent)

**Anchor:** quantum-inspired optimisation on classical hardware is a real category (simulated quantum annealing,
Toshiba's Simulated Bifurcation Machine, Fujitsu's Digital Annealer, for Ising/QUBO). A classical simulation of quantum
dynamics has no asymptotic speedup (Grover's square root does not survive simulation); what survives is a different
search dynamics: tunnelling through tall thin barriers, ballistic exploration, interference between routes. These are
heuristics whose worth must be measured, which is what the project's discipline is for.

**The twist: an ecology of quantum walkers on the objective.** The objective is the landscape (depth = food); walkers'
coins are evolved. Mapping: measurement with its energy cost = one function evaluation, counted exactly by the ledger;
sensing strength that costs coherence = an annealing schedule tuned by selection; tunnelling = barrier crossing;
persistent decaying structures = pheromone memory (ant colony) and tabu lists; entangled bonded pairs = learned variable
linkage (estimation-of-distribution algorithms); fermionic exclusion = niching; long winters and dormancy = restarts and
archives for moving optima; arms races = adversarial and robust optimisation; no-cloning = mutation as a law. Cost: an
objective reintroduces fitness; the framing is "the environment is the objective, survival is the selection".

**Where it could win:** dynamic landscapes (diversity lets a population follow a moving optimum); robust and adversarial
optimisation (co-evolved worst cases); many objectives (a spread along the front by exclusion); constrained budgets (an
exact ledger of evaluations, memory and time); learning the optimiser by survival over a landscape class. Where it
won't: static QUBO benchmarks against strong incumbents, including exact solvers.

**Systems domains:** cluster scheduling under changing inputs; VLSI placement and floorplanning (where RL placement
claims lacked honest baselines); compiler autotuning; query plans; routing under failures; grid dispatch with
intermittent supply; and the engine's own dial calibrations, done by hand so far.

**Caveats:** no speedup, only dynamics; a simulated walk costs about what the classical search costs, so the gain is in
where it looks; the incumbents are good.

**First experiment:** a dynamic benchmark (moving peaks, dynamic knapsack); arms: quantum-walker ecology, classical-walker
ecology with the same rules, well-tuned random search with the same exact evaluation budget, the known optimum as the
oracle ceiling; pre-registered readout (tracking error over time) and bands. The first bar is beating random search
under an exact ledger.

---

## 8. Open-source landscape for the quantum substrate (four Haiku searches, 2026-11-03)

Full tables, 44 verified entries, are in `docs/brainstorming-research/` (`quantum_alife.md`, `quantum_optimisers.md`,
`quantum_evolutionary.md`, `substrate_tooling.md`). The sixteen that matter were re-checked through the GitHub API
(licence, last push) on 2026-11-03.

**The gap is real.** No open project puts quantum-walk organisms in an artificial life. The nearest, Alvarez-Rodriguez
et al.'s "Quantum Artificial Life" on IBM hardware (arXiv 1711.09442, 2017), has no code. No quantum Lenia exists. No
open quantum walk on a sphere or on a non-orientable surface was found. No project combines a walker ecology with a
quantum-walk substrate for optimisation; QWMO (MIT, June 2026, a preprint's code) is the nearest and is a population
optimiser with probabilistic operators, not a walk.

**Reuse candidates (permissive licences):**
- QuantumWalk.jl (MIT, Julia; discrete/continuous/Szegedy walks, spatial search): the reference for walk kernels and
  for checking an MLX walk's correctness. Hiperwalk (LGPL-3.0, Python, active 2026) is the HPC alternative; copyleft.
- QuantumGameOfLife.jl (MIT, active 2026): unitary QCA with measurement rules, the closest "quantum game of life".
  lhillber/qca is the most complete unitary-evolution code but has no licence and is stale (2022).
- simulated-bifurcation-algorithm (MIT, 171 stars), dwave-samplers (Apache-2.0: SA, SQA, tabu), OpenJij (Apache-2.0,
  SQA): the quantum-inspired optimiser baselines any claim must beat.
- gmpb (MIT, 2026): a generalized moving-peaks generator; pateda (MIT, 2026): a classical estimation-of-distribution
  twin; DEAP (LGPL-3.0, 6,400 stars) for GA/ES baselines and its classic Moving Peaks. EDOLAB (GPL-3.0, MATLAB) is the
  reference protocol for dynamic-optimisation benchmarking, not reusable code.
- COCO (numbbo) and IOHexperimenter: the methodological references for budget-honest reporting (runtime in function
  evaluations); licences unconfirmed (NOASSERTION), neither covers Ising/QUBO, and neither has an exact ledger.
- healpy (GPL-2.0, CPU) as the spherical-harmonic oracle; s2fft (MIT, JAX/CUDA) as the differentiable reference. No
  Metal or MLX spherical-harmonic transform exists; one would be written and checked against healpy.
- Reversible CA: no maintained library (margolus-cpp is archived); integer block rules written directly in MLX are
  simpler than adapting anything found.

**Substrate facts confirmed:** MLX has fft/ifft/fft2/fftn on Metal, in float32/complex64 only; no float64 on Metal;
complex matmul on Metal is slower than the elementwise path (MLX issues #3972, #1905, #2076). So the walk would run in
complex64 on the GPU with a CPU reference check, which is the project's existing F20 rule applied to complex numbers.

**What this changes in the plan:** nothing in the ordering (diel migration first), and it strengthens the quantum idea's
novelty claim: the first experiment of section 7 (quantum walkers versus a classical twin versus tuned random search
under an exact ledger, on gmpb's moving peaks, with SB/SQA/SA as further baselines) would be the first of its kind
that we could find. Licences to settle before reuse: IOHexperimenter, COCO, lhillber/qca.

---

## 9. Priorities for enjoyment on the desktop (from the reusable code)

The annealers and benchmark harnesses serve research and optimisation, not the desktop. In order of feeling per week:
1. Day and night with the diel migration (no external code): a planet turning through night, the population rising at
   dusk and sinking at dawn; a "watch a day" camera speed.
2. A food landscape that breathes: gmpb's moving-peaks maths reimplemented in MLX as a deterministic food driver;
   hotspots drift, split and merge; the chronicle gets real events.
3. The planet's harmonics as an observer: healpy host-side; low spherical-harmonic modes drawn as overlays on the globe
   and charted over time as a heartbeat in Report. No engine change.
4. One quantum walker as a proof of beauty: QuantumWalk.jl as reference, an MLX walk on the HEALPix sphere drawn as a
   glowing cloud with collapse flashes; a week decides the quantum iteration.
Then the owner's chosen worlds: the reef-building planet, the ring world, the Möbius wrap, each its own design pass.
Every new dynamic should produce chronicle events and one player power.

---

## 10. All-in on emergence: evolving space-time with a fraction of quantum walkers (owner's prompt)

**Premises:** space evolves only if something selects it: topology must be a built structure (made at a cost, decaying,
persisting beyond builders, inherited). Quantum walkers are not random: unitary dynamics, randomness only at
measurement; their interference depends on the holes and cycles of the space, so a quantum fraction injects randomness
shaped by the geometry and probes it. The engine is closer to a graph world than it looks: organisms use neighbours and
distances, not coordinates; bonds are persistent links; edges can be rows (fixed maximum, born and dying,
dead-canonical), so fixed shapes, pure MLX, the ledger and replay survive.

**Computably:** space is a dynamic graph (nodes hold food and soil; edges carry adjacency, distance, capacity), with the
current worlds as initial graphs. Organisms build and erode edges (niche construction of topology; a space ledger term).
Dimension is emergent and local (spectral dimension, ball growth): the new map. Time can vary by region (a per-node
clock rate as a mask): time dilation with a geography. Quantum walkers, a fixed fraction, walk the graph with a
time-dependent generator, decohere when sensing, collapse when eating; their spread reports the geometry. Precedents
without life: Wolfram's hypergraph rewriting, causal dynamical triangulations, graph cellular automata.

**Game loop:** a living graph drawn by a force-directed 3D embedding with dimension as colour, the embedding's
unfaithfulness declared on screen; powers: fold, tear, slow time here, seed a bloom; chronicle events about space.

**Stages:** (1) graph substrate reproducing today's sphere and torus from their graphs (identity matrix); (2) edge
dynamics with the space ledger term; (3) dimension and time observers, host-side first; (4) the quantum fraction, after
the one-walker proof of beauty; (5) powers over space and the chronicle vocabulary.

**First claim:** do organisms shape space in a way that persists beyond the builders and raises descendants' survival,
against a world where the same edge changes happen at random (the yoked random control applied to geometry)?

**Risks:** legibility first (the viewer track must run ahead); float32 graph distances and amplitudes on Metal need CPU
checks; scope: a new engine generation, to begin only after the quantum walker's picture and diel migration.

---

## 11. Synergies with "Quadrillion-parameter transformers and how to train them" (Q Labs Research, Nov 2026, https://qlabs.sh/research/parameters)

**The post:** a scaling law for transformers on fixed data: larger models for more epochs with weight decay scaled with
size and epochs keep improving with compute; snapshot ensembling along a run, then deep ensembling across seeds, is a
scaling axis once compute is large relative to data; verified at 16× the fitted range; companion paper arXiv:2610.37842;
no code. A methods source, not a scaling truth; its brain comparison is loose.

**Real synergy: the ecology is in the data-constrained regime.** A fixed world is a fixed dataset; seasons and generations
are epochs; the project has observed the headline problem from the other side (activity bounded in every arm; the 64k
run stalls after about a million steps). Twins of their levers: weight decay ↔ brain cost (`brain_cost_per_l1`) and
mutation pressure; epochs ↔ run length; snapshot ensembling ↔ checkpoints along a lineage; deep ensembling ↔
independent lineages (M1's unit); new data ↔ environmental turnover (storms, seasons, a moving food landscape).
Hypothesis: on a fixed world, longer runs and larger populations stall unless a regularizer or environmental turnover
grows with them. Lineages-versus-length is a budget question the stage rule answered by hand; it could be fitted. The
fitting discipline (one law across axes, hold out the largest scale, require the extrapolation to land) transfers whole;
the preset ladder and the brain-size overlay are the grid, priced by the preset guide's speeds. Mode connectivity is the
neutral-network picture from evolutionary theory (the neutral shadow; the hidden-unit gauge symmetry).

**Not a synergy:** no backprop or gradients here; tiny recurrent brains; their numbers do not transfer.

**Experiment, "scaling laws for emergence":** grid over world size (three presets), brain size (two), run length (four),
lineages (one to six); frozen readouts (persistent-novelty rate, time to the first persistent body, storm survival); fit
a law with a stall term in place of their overfitting penalty, hold out 256k, test the extrapolation; then find which
lever (brain cost, mutation, turnover) bends a flat law. Either outcome is a result; cost is real GPU time, partly
covered by existing runs.

---

## 12. Synergies with "Subquadratic Attention with a Hierarchical Tree-Structured Indexer" (Zyphra, arXiv:2610.42112)

**The paper:** ZHA builds a tree of recursively compressed key summaries over a growing KV cache (compressed keys are
tangent-plane approximations under learned reference queries), routes each query top-down with top-p selection, and
attends to the leaves reached; cost grows sublinearly because the relevant set grows as a power law. Converted from
pretrained models by a short KL distillation; a four-parameter geometry-to-accuracy model (RMSE 1.44 pp, leave-one-
length-out); a roofline model with a worker-wave term; no kernels released.

**Synergies, ranked:**
1. HEALPix nested ordering is already a quadtree per base pixel, and the calls far table (47.7) is a one-level chunk
   summary with measured edge losses (6 to 11 % error in dense scenes, 10 to 36 % audibility flips, 12 to 16 % of
   pairs heard by neither block on the floor sphere). Recursive summaries with top-down routing would give
   planetary-scale far sensing at sublinear cost with fewer edge losses; fixed shapes force top-K with a learned mask
   rather than top-p.
2. A hierarchy detector, host-side (§44): a sublinear power law in the number of organisms that materially influence
   each organism's next action, as population grows, is a signature of multi-level organisation (collective
   individuality). No engine change.
3. A memory tree for the chronicle: chapters summarized into eras and ages, with top-down descent for reports; the
   "legends mode" made tractable without machine learning.
4. Summaries organisms evolve: heritable per-organism reference queries weighting the far table's columns, so sensing
   strategy evolves; speculative, subject to the 5 % step-cost gate.
5. Shared discipline: geometry-to-accuracy models validated by held-out extrapolation, and roofline modelling, match
   the project's gates and `bench --stages`.

**Not a synergy:** no transformer in the engine; the director's local LLM cannot be converted (about 4B distillation
tokens, unreleased kernels). A tree over nested HEALPix cells for sparse wavefunctions (adaptive refinement) is a note
for the quantum substrate, not a plan.

**First step:** the hierarchy-detector observer; then a design note on nested-HEALPix far sensing with the calls
far-table measurements as the baseline to beat.

---

## 13. Image and video generation as an artistic endeavour

**Aesthetic = the project's four constraints:** conservation, replay, no fitness function, honesty. Every pixel is
simulation; every image is an edition with a seed; nothing is chosen for looking good; the artist is the director,
logged and replayable.

**Images:** conserved-ink painting (organisms as brushes, soil as pigment, structures as marks, decay as fading, death
returning pigment; a finite ink budget, like a sand mandala); editions by seed (sixty-four seeds, each print carrying
seed, config hash and git revision, regenerable bit for bit); dials as a series (a print series along one dial is a
visual scaling law); the planet in harmonics (low spherical-harmonic modes of the population drawn on the globe);
diptychs of the same world (mirror or row-permuted twins, bit-identical, the symmetry goldens as art); light that
collapses (quantum walkers as long exposures, collapse as shutter, interference fringes on the HEALPix sphere).

**Video:** the nature documentary of a world that never existed (time-lapses, chronicle as narration, chapters as acts,
narration that cannot lie because every event is validated); coarse to fine (one continuous zoom from ages to a single
afternoon along the summary hierarchy of section 12); lawful palindromes (the reversible substrate played backward; a
Möbius world whose second lap is the mirror of its first); space that folds (the evolving graph with dimension as
colour, the embedding's seam declared in frame); sound (the voice channel sonified, storms as rumble, the ledger as a
drone of constant loudness).

**Installations:** a planet on a wall (the app as a gallery piece, visitors holding the director's powers under a
budget, the archive as the piece); authorship under control (visitor-steered versus yoked random director, side by
side; can anyone tell?).

**Already there:** `alife render` (PNG and video from checkpoints, layers, colour modes); the preset snapshot tool drives
the 3D and sphere renderers offscreen; replay gives editions; the chronicle gives narration. A batch renderer with
provenance metadata in each file is a small step.

**First piece:** the seed grid, sixty-four `tiny3s` worlds at 30,000 steps from one camera, printed with their seeds.

---

## 14. The graph substrate as the backbone (assessment of a local model's reading of this file)

**Accepted:** the spatial graph (§10), the economic graph (§3, §4: bonds as supply chains and employment) and the quantum
walk's graph (§6, §7) are one abstraction with different edge semantics, so the graph substrate is the single build that
de-risks all three, and its stage-1 acceptance test (behavioural identity with today's sphere and torus) is a test the
project already knows how to run.

**Refinements:** (1) the engine's field layer is already a graph (exact int64 diffusion over HEALPix neighbours is a graph
Laplacian; hash cells are adjacency; far hearing reads columns); the agent layer is not (continuous positions, Cartesian
forces, a distance-based neighbour search). Stage 1 therefore builds a node graph carrying fields, adjacency and
distances with agents continuous inside nodes; the identity test applies to the field layer; the economic ABM is the
first consumer that needs the graph alone. The test is cheap; the build is the scale of the sphere series. (2) The
force-directed embedding will lie: smooth it and never re-solve per frame; pair every picture with topological readouts
(spectral dimension, cycle counts, ball growth), dimension-as-colour first; anchor the layout to the initial geometry;
draw stress or confidence as a per-region seam. (3) Time dilation is a binary per-node update mask, never a fractional
step (the integer ledger forbids it); the quantum fraction enters first as observers that probe the geometry, then as
actors that inject geometry-shaped randomness, each gated on the one-walker proof of beauty.

**Revised order:** diel migration; then the graph substrate's stage 1 as the backbone; then the economic ABM, the
quantum fraction and evolving space, which otherwise each pay for their own graph.
