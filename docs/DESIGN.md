# alife v0: design document (final)

A neural artificial-life engine in Python on Apple MLX. Agents ("dots") have mutable per-agent recurrent neural
controllers. They sense, signal, move, bond, split, exert forces, harvest, share energy, fight, reproduce with
mutation, and die when their energy runs out. There is no backpropagation and no fitness function. Selection
is ecological only.

- **Target machine:** one MacBook Pro (M5 Pro, 48 GB unified memory, Metal GPU).
- **CI:** Linux with `mlx[cpu]`.
- **Default code path:** pure MLX ops. There is no `mx.fast.metal_kernel` anywhere in v0.

This document is the single source of truth for implementers. Module boundaries, function signatures, numeric
defaults, tie-break rules, sub-step order and RNG-key usage are normative. A behaviour this document does not
pin down is a documentation bug: raise it rather than guess.

Provenance: this design merges three competing proposals.

| Source | What it contributes |
|---|---|
| "throughput" (base) | Execution model, neighbour search, perf model, module skeleton |
| "correctness" | Invariant machinery, bond back-pointers, ledger, RNG discipline, tests |
| "emergence" | Affordances, ecology, handshake, capacity reserve, metrics |

Every fatal flaw the judges raised is fixed. The fixes are listed in the Decision log (§24). A later review round
is resolved in the Review log at the end of this document.

---

## 0. Verified facts this design relies on

All of these were run on CPU MLX 0.32.3 in the design container. Scripts live in the design scratchpad
(`design_proto/final/f*.py`; earlier ones in `design_proto/`, `design_proto/correctness/`,
`design_proto/emergence/` and `judge_proto/`). Nothing was measured on a Metal GPU. §15 is therefore a model,
and `alife bench` exists to replace it with measurements.

| # | Fact | Consequence |
|---|---|---|
| F1 | A row scatter into a large array that is also read elsewhere in the same evaluated graph forces a **full copy** of that array. `mx.async_eval` pipelining also defeats donation. Measured with the merged layout (32k × 101 × 32 + 32k × 33 × 35, fp32, 562 MB, compiled step with brain read and parent gather): writing the pending rows at the **start of the next step** (S0) gave +48 MB peak, i.e. in place. Keeping a Python reference to the previous state gave +1186 MB peak and 2.4× the time (`f4_donate.py`). | Deferred birth flush at S0 (§3, §12.6). Sync eval once per step. Observers must not retain old states. |
| F2 | `a[idx] = v` (setitem) mutates the caller's Python array object in eager mode. `out = mx.array(a); out[idx] = v` does not mutate the caller and still writes in place under `mx.compile` (+24 MB peak on a 424 MB tensor, `f8_setitem.py`). | Every setitem goes through `ops.set_rows`, which uses the copy-wrapper form. |
| F3 | `mx.random.split(k, 4) != split(k, 8)[:4]`: split is not prefix-stable. | Exactly `NUM_STREAMS = 16` per-step streams, versioned (§14). |
| F4 | `randint` and `uniform` give duplicate values at scale, and `mx.random.bits` does not exist in 0.32.3. `mx.random.permutation(N)` returns unique `uint32`, taking 7.7 ms at 64k on CPU. | Unique priorities always come from a permutation. |
| F5 | `argsort`, `argmax`, `searchsorted` and `permutation` return `uint32`. `cumsum(bool)` returns `int32`. `argmax` returns the first index on ties. `>>` on int32 is arithmetic. | Cast with `.astype(mx.int32)` after each of these ops. Tie rules below rely on first-index `argmax`. |
| F6 | `int32.sum()` silently wraps. A 3-limb (11-bit) int32 decomposition gives exact int64 totals for any non-negative int32 array of ≤ 2²⁰ elements (`f6_ops.py`). | `ops.exact_sum`. The simulation path has no int64 reduction and no int64 scatter. |
| F7 | `.at[idx].add` and `.at[idx].maximum` on int32 are correct with duplicate indices. A scatter-max with base −1 and invalid lanes writing −1 to index 0 is a correct "placement" (`f1_api.py`). uint32 scatter-max works. | All contested writes are integer, hence order-independent and deterministic on GPU. |
| F8 | `mx.remainder(-1e-8, 1024.0) == 1024.0` in fp32. | `ops.wrap` adds a `where(x >= L, x - L, x)` guard. |
| F9 | Propose/accept/mutual bond handshake with permutation priority and stored back-pointers (`rev`): over 300 randomized rounds of breaks, formations, deaths and revivals, the symmetry invariant held after every phase, no agent ever acted as both proposer and acceptor, and 14 k bonds formed (`f2_bonds.py`). | §8. |
| F10 | `proportional_take` (f32 ratio, scaled 2¹⁵, −1 guard). Over 800 random trials with demands < 2¹⁶ and supplies < 2²⁴: grants never exceed demand, grants per pool never exceed supply, uncontested pools are granted exactly, and leftover in contested pools is ≤ contenders + 2·D/2¹⁵ + 1 quanta (`f3_take.py`). | One contention primitive for harvest and combat (§10). |
| F11 | Overdamped two-agent body with a contracting spring: the centre of mass moves **0.000** in 2000 steps. With phase-locked grip (per-agent drag 1 to 5), it moves 4.8 units (`f5_grip.py`). | Contraction alone cannot propel a body. The `grip` output is added (§9). |
| F12 | `mx.compile` accepts and returns nested `NamedTuple` pytrees (State, StepOut with nested records) and int64 scalars. Compiled output equals eager output (`f7_nt.py`). | `State` is a `NamedTuple`. |
| F13 | Cell-sort neighbour search with candidates gathered from the cell-sorted position copy equals brute-force kNN whenever no cell overflows. Overflow is detected and counted (throughput V6, correctness V8). | §4. |
| F14 | Bond-graph connected components by min-label propagation plus pointer jumping match union-find (throughput V10, correctness V12). | §19. |
| F15 | safetensors round-trips bool, int32, int64, uint32 keys, fp16, bf16, 0-d arrays and string metadata. It does **not** preserve key order: `mx.load` returns keys in arbitrary order (`revise_proto/r2_fert.py`). Resume (k steps, save, load, k steps) is bitwise identical to 2k uninterrupted steps (throughput V11, correctness V10). | §18. Loading matches keys by name as a set; State is rebuilt in `State._fields` order (§2.2). |
| F16 | `argpartition(kth=K-1)` followed by a sort of the K survivors is 2.3× faster than a full argsort on CPU. On Metal, ArgPartition currently lowers to a full sort, but rows of 144 still sort within one threadgroup. | §4. Budget in §15 assumes a sort. |
| F17 | `np.asarray` of a bfloat16 `mx.array` raises (`'bfloat16' is not a valid PEP 3118 buffer format string`). `a.astype(mx.float32)` is an exact widening and then converts. | Every host read goes through `state.to_numpy` (§16.2). |
| F18 | Soft repulsion cannot bound compression when a crowd thrusts toward a common point: with plain thrust, 300 agents converging under the §4 search (C = 16, K = 8) reach 183–300 agents in one hash cell and near-zero spacing; with no C cap it is still 35–115 per cell (`critic_proto/c9*.py`). Scaling each agent's thrust by `max(0, 1 − n_contacts/4)` gives median spacing 1.77–1.88, ≤ 15 agents per hash cell, ≤ 8 per field cell and zero overflow, for 300 or 1 000 agents, with or without grip (`critic_proto/c9c_fix.py`, `revise_proto/r1_crowd_scale.py`). | Contact-attenuated thrust (§9); C sizing (§4). |
| F19 | CPU cost of the heaviest tiny-preset sub-steps (N = 1 024, compiled, 4-core runner): neighbours 5–6 ms; brain 3.2 ms in f32 but 11 ms in fp16 (CPU fp16 matmul is ≈ 3× slower); mutation randoms for M = 64 lanes ≈ 12.5 ms; S0 flush ≈ 2 ms (`critic_proto/c7_tiny_cost.py`). | CI defaults to f32 weights and `tiny` uses M = 32 (§22.0). |
| F20 | On an M5 Pro GPU (MLX 0.32.3), float32 `matmul` is reduced precision: max relative error ≈ 8e-4 against the CPU stream on a 256×4096×256 product; `exp` is accurate (≈ 3e-7 relative). Measured on the target Mac, not in the design container. | Anything that must be accurate avoids GPU matmul: `fertility` sums separable Gaussians elementwise; `weight_diversity` computes its Gram on the CPU stream. Brains keep GPU matmul (fp16 weights; precision is irrelevant to selection). |

---

## 0.1 Fidelity: the thread's verbs mapped to v0 mechanisms

| Verb in the thread | v0 mechanism | Section |
|---|---|---|
| SENSE | Self state, 3 nearest neighbours, pooled neighbourhood, 4 bond slots, resource field and 3 chemical fields, all in the body frame | §5 |
| SIGNAL | (a) Two broadcast `msg` channels that neighbours and bond partners read next step. (b) Deposits into 3 diffusing chemical fields with different length scales | §5, §7, §13 |
| MOVE | Thrust along own heading, plus turn | §7, §9 |
| BOND | Consent handshake (propose, accept, or mutual), ≤ 4 bonds per agent, brain-chosen partner | §8 |
| SPLIT | (a) **Division:** reproduction splits energy by a heritable fraction, and the child may stay bonded. (b) **Fission:** per-slot bond release splits a body | §8, §12 |
| EXERT FORCES | Thrust transmitted through springs; brain-controlled `contract` (muscle) changes rest lengths; `grip` (substrate anchoring) makes contraction produce locomotion; soft repulsion | §9 |
| HARVEST | Effort-controlled draw from a regenerating resource field; contested cells are shared proportionally in exact integers | §10 |
| SHARE ENERGY | Per-bond-slot transfers with 90% efficiency | §8, §10 |
| FIGHT | Query-targeted theft (70% efficiency, 30% spilled to the field); a costly `defend` output | §11 |
| REPRODUCE WITH MUTATION | Division with self-adaptive heritable mutation scale and rate, a heritable offspring fraction and drifting kin tags | §12 |
| DIE | Energy reaches 0, or senescence. Corpse (structure plus residual energy) is recycled into the field | §10 |
| "Neural cellular automaton" | Neural agents in continuous space, plus grid fields with local regeneration, diffusion and decay | §13 |
| "Start random, no backprop, no fitness" | Random founder brains, no gradients, no objective, no reseeding. Immigration is optional and off by default | §6, §12 |
| "Distributed" | Single device in v0. The step is a pure function on fixed shapes with local rules and global int64 uids, so it can be split across devices later | §23 |

---

## 1. World model and rationale

**Hybrid world.** Agents are continuous 2-D particles on a periodic torus. Resource and chemical fields live on a
regular lattice, which is the cellular-automaton substrate.

- **Why particles for agents.**
  - Bonded soft bodies, springs, contraction, thrust and collisions need sub-cell geometry, as in ALIEN.
  - A one-agent-per-cell lattice (Coralai) cannot express bond lengths or forces, and it makes moving and
    splitting discrete and anisotropic.
  - Particle state is O(N) struct-of-arrays.
- **Why a lattice for fields.**
  - Regeneration, diffusion and decay are cheap, deterministic stencils (`mx.roll`).
  - Exact integer resource bookkeeping per cell is natural.
  - "Many agents on one patch" competition is explicit.
- **Two grids, both periodic.**
  - The *hash grid* for neighbour search has `G_h = floor(L / R)` cells per side, with cell side
    `cell_h = L / G_h ≥ R`, where `R = sensing.radius`.
  - The *field grid* has `FG = L / field_cell` cells per side. This must be an integer, which is validated.
  - Every agent maps to the single field cell that contains its centre.
- **Boundary conditions.** Periodic in both axes. There are no walls and no edge artefacts. Displacements use
  the minimum-image convention `d − L·round(d/L)`. `round` is half-to-even and odd-symmetric, so
  `min_image(−d) == −min_image(d)` bitwise and bond forces are exactly antisymmetric.
- **Units.**
  - Length: agent radius `r = physics.radius = 1.0`. Diameter, bond rest length and offspring distance are all
    2.0.
  - Time: 1 step, `dt = 1`. Every rate in this document is per step.
  - Energy: integer **quanta (q)**, stored as int32.
  - Angles: radians. Headings lie in `[−π, π)`.
- **Presets.** Density at full capacity is ≈ 0.06 agents/u², about 19% area coverage. Patchy fertility
  concentrates agents locally.

| preset | `n_max` (N) | `M` (births/step) | `n_init` | `L` | `G_h` (cell) | `FG` (cell 4) |
|---|---|---|---|---|---|---|
| `tiny` (tests/CI) | 1 024 | 32 | 256 | 128 | 21 (6.10) | 32 |
| `dev` | 16 384 | 256 | 4 096 | 512 | 85 (6.02) | 128 |
| `64k` (defaults) | 65 536 | 1 024 | 16 384 | 1 024 | 170 (6.02) | 256 |
| `256k` | 262 144 | 4 096 | 65 536 | 2 048 | 341 (6.01) | 512 |

The `fast` overlay sets `sensing.k_neighbors = 6`, `sensing.cell_capacity = 12` and `brain.hidden = 24`. It
combines with any preset.

---

## 2. State schema

### 2.1 Symbols

Structural constants are fixed in `layout.py` for v0. Changing any of them changes the genome layout and requires a
code change plus a bump of `GENOME_LAYOUT_VERSION`. The values below are the 2D layout; a 3D world appends a tail to
the input and output vectors, and `layout.for_dims(nd)` (held as `Dims.lay`) gives the per-dims totals (§27.1).

| symbol | value | meaning |
|---|---|---|
| `B` | 4 | bond slots per agent |
| `NS` | 3 | nearest neighbours given explicit input blocks |
| `NF` | 7 | features per sensed neighbour (2D; 3D: 8, `NBR_DZ` last, §27.2) |
| `F` | 6 | targeting descriptor dims (= NF minus the `present` flag; 2D; 3D: 7, §27.2) |
| `MSG` | 2 | broadcast message channels |
| `NCH` | 3 | chemical field channels |
| `T` | 4 | genome tag dims |
| `D_IN` | 69 | brain inputs (2D; 3D: 84, §27.1) |
| `D_OUT` | 35 | brain outputs (2D; 3D: 39, §27.1) |

The remaining symbols are config-derived (`config.dims(cfg)`):

| symbol | value | meaning |
|---|---|---|
| `N` | `n_max` | row capacity |
| `M` | `max_births_per_step` | birth lanes |
| `N_CAP` | `N − M` | population cap |
| `DL` | `2M` | death-record lanes |
| `K` | `k_neighbors` | neighbour list width |
| `C` | `cell_capacity` | hash-cell scan capacity |
| `Q` | `9C` | candidates per agent (2D; `3^nd·C`, 27C in 3D, §27.1) |
| `H` | `hidden` | hidden units |
| `W1_ROWS` | `D_IN + H` | rows of `W1` (`D_IN` of the world's dims) |
| `W2_ROWS` | `H + 1` | rows of `W2` |
| `FG` | `L / field_cell` | field grid side |
| `PP` | `n_patches` | fertility patches |
| `nd`, `lay` | `world.dims`, `layout.for_dims(nd)` | 2 or 3, and the genome I/O layout (§27.1); `d3 = (nd == 3)` |
| `FZ`, `n_cells`, `field_shape` | `height / field_cell`; `FZ·FG²`; `(FG, FG)` or `(FZ, FG, FG)` | field layers (1 in 2D), cells and lattice shape (§27.1) |
| `G_z`, `cell_hash_z` | `floor(height / R)`; `height / G_z` | hash layers and their height (1 and ∞ in 2D, §27.1) |
| `wdtype` | `brain.weight_dtype` | weight storage dtype (`float16` by default on the Mac; the default CI fixture uses `float32` for speed and fp16 has targeted CI tests, §22.0) |

**Capacity reserve.** The population never exceeds `N_CAP = N − M`. At reproduction time at least `M` rows are
therefore dead, and every birth lane, padding included, targets a **distinct** dead row. No sink row, dummy row
or duplicate-index set-scatter exists anywhere in the design.

### 2.2 `State` (a `typing.NamedTuple`; field order is the canonical order)

The canonical order is `State._fields`. It is the order used by `state_to_arrays`, `state_digest` and every
host iteration. Checkpoint files are keyed by name, and their key order is not significant (F15). The shapes are
those of a 2D world; `state.expected_layout(cfg)` is the one schema for both dims (§27.1). A 3D world differs in
the rows marked "3D" and, through `W1_ROWS` and `D_OUT` (§2.1), in the weight widths, and its state is the parallel
type `State3`: these 47 fields, then `pitch` and `buoy_cap` (§27.4).

**Per-agent rows (leading dim N).** "Canonical when dead" means the value is forced at death and checked by the
invariants. Other dead-row values are stale and must never be read without an `alive` mask.

| field | shape | dtype | meaning | invariants |
|---|---|---|---|---|
| `alive` | (N,) | bool | row holds a living agent | `alive.sum() ≤ N_CAP` |
| `uid` | (N,) | int64 | globally unique id, assigned in birth order, never reused | alive ⇒ `0 ≤ uid < next_uid`, unique among alive. Canonical when dead: −1 |
| `parent_uid` | (N,) | int64 | parent's uid; −1 for founders and immigrants | alive ⇒ `parent_uid < uid` |
| `clade` | (N,) | int32 | founder index `0..n_init−1`, inherited unchanged; immigrants use `n_init` | `0 ≤ clade ≤ n_init` |
| `generation` | (N,) | int32 | founders and immigrants 0; child = parent + 1 | ≥ 0 |
| `birth_step` | (N,) | int32 | index of the step during which the agent was born; founders −1 | alive ⇒ `birth_step ≤ step − 1` |
| `energy` | (N,) | int32 | stored energy (q) | alive ⇒ `1 ≤ E ≤ e_max`. Canonical when dead: 0 |
| `pos` | (N,2); 3D (N,3) | float32 | position (x, y); 3D (x, y, z) | `0 ≤ pos < L`, finite; 3D `z ∈ [z_lo, z_hi]` (§27.4) |
| `vel` | (N,2); 3D (N,3) | float32 | displacement during the last step | `‖vel‖ ≤ v_max·(1+1e-6)`. Canonical when dead: 0 |
| `heading` | (N,) | float32 | body orientation | `[−π, π)` |
| `h` | (N,H) | float32 | recurrent hidden state | `[−1, 1]`. Canonical when dead: 0 |
| `msg` | (N,MSG) | float32 | broadcast emitted last step | `[−1, 1]`. Canonical when dead: 0 |
| `last_loss` | (N,) | int32 | energy taken from me by attackers last step | ≥ 0. Canonical when dead: 0 |
| `last_harvest` | (N,) | int32 | energy harvested last step | ≥ 0. Canonical when dead: 0 |
| `last_inflow` | (N,) | int32 | energy received along bonds last step, after loss | ≥ 0. Canonical when dead: 0 |
| `partner` | (N,B) | int32 | partner row per bond slot, −1 = empty | bond symmetry (§8.1). Canonical when dead: −1 |
| `rev` | (N,B) | int32 | slot index of the back-pointer in the partner's row, −1 = empty | bond symmetry. Canonical when dead: −1 |
| `tag` | (N,T) | float32 | neutral heritable marker (kin cue, render colour) | `[−1, 1]` |
| `mut_sigma` | (N,) | float32 | heritable mutation scale σ | `[sigma_min, sigma_max]` |
| `mut_rate` | (N,) | float32 | heritable per-weight mutation probability p | `[rate_min, rate_max]` |
| `offspring_frac` | (N,) | float32 | heritable fraction of disposable energy given to a child | `[frac_min, frac_max]` |
| `brain_cost` | (N,) | int32 | per-step brain upkeep (q), fixed at birth from L1 of the stored weights | ≥ 0 |
| `W1` | (N, W1_ROWS, H) | wdtype | `[x, h] → hidden` weights; bias via `x[0] = 1` | `abs(w) ≤ weight_clip`. A row born in the last step is stale until S0 (see `pend_*`) |
| `W2` | (N, W2_ROWS, D_OUT) | wdtype | `[h', 1] → outputs` | same |

**Pending birth buffer (deferred heavy writes, §12.6).**

| field | shape | dtype | meaning |
|---|---|---|---|
| `pend_slot` | (M,) | int32 | destination rows of last step's birth lanes. They are distinct. Padding lanes point at distinct dead rows. |
| `pend_W1` | (M, W1_ROWS, H) | wdtype | child `W1` rows for those lanes |
| `pend_W2` | (M, W2_ROWS, D_OUT) | wdtype | child `W2` rows for those lanes |

**Fields.**

| field | shape | dtype | meaning | invariants |
|---|---|---|---|---|
| `resource` | (FG,FG); 3D (FZ,FG,FG) | int32 | resource quanta per cell, indexed `[y, x]` (part of the energy total) | `0 ≤ R ≤ R_HARD = 2²⁴ − 1` |
| `chem` | (NCH,FG,FG); 3D (NCH,FZ,FG,FG) | float32 | chemical concentrations (not energy) | `[0, chem_max]`, finite |
| `patch_pos` | (PP,2) | float32 | fertility patch centres | `[0, L)` |
| `patch_vel` | (PP,2) | float32 | constant patch drift velocities, drawn at init | — |

**Globals.**

| field | shape | dtype | meaning |
|---|---|---|---|
| `key` | (2,) | uint32 | PRNG key for the next step; advanced exactly once per step |
| `step` | () | int32 | number of completed steps |
| `next_uid` | () | int64 | next uid to assign |
| `total` | () | int64 | `TOTAL` (§10.1) at the end of the last completed step |
| `total0` | () | int64 | `TOTAL` at initialisation |
| `ledger_cum` | (16,) | int64 | cumulative ledger since step 0 (indices `LedgerIdx`, §10.8) |
| `diag` | () | int32 | **sticky** invariant bitmask, OR-accumulated every step (§22.3) |

**Director environment (live levers, `docs/DIRECTOR.md` §3).** Nine flat fields appended after `diag` (the 38
fields above are the frozen v0 layout, `state.V0_FIELDS`). They are simulation state: checkpointed, changeable
between steps without a recompile, and read by the compiled step only when `cfg.director.enabled` (static,
§3, §17). `State.env` is a read-only `Env` NamedTuple view (`regen_mult`, `season_amp`, ...) and
`state.ENV_FIELDS` lists the flat names. Checkpoint keys are the flat names.

| field | shape | dtype | meaning | inert default | range (checked) |
|---|---|---|---|---|---|
| `env_regen_mult` | () | float32 | multiplier on per-cell regeneration (§13.1) | 1.0 | [0, 4] |
| `env_season_amp` | () | float32 | replaces `fields.season_amplitude` | `fields.season_amplitude` | [0, 1] |
| `env_season_period` | () | int32 | replaces `fields.season_period`; 0 = seasons off | `fields.season_period` | ≥ 0 |
| `env_drift_mult` | () | float32 | multiplier on patch drift velocity | 1.0 | [0, 20] |
| `env_patch_gain` | (PP,) | float32 | per-patch fertility gain; 0 disables a patch | ones | [0, 4] |
| `env_imm_request` | () | int32 | immigrants still to be admitted (§12.7); the un-admitted remainder carries over | 0 | ≥ 0 |
| `env_cat_center` | (2,) | float32 | catastrophe centre (torus coordinates) | zeros | finite |
| `env_cat_radius` | () | float32 | catastrophe radius; 0 = none | 0 | [0, L/2] |
| `env_cat_kill_p` | () | float32 | per-agent kill probability inside the radius; reset to 0 after the step it applied in (§12.9) | 0 | [0, 1] |

`state.default_env(cfg)` returns the inert arrays for a config. `env_season_amp` is allowed up to 1 (inclusive)
because `validate` allows `season_amplitude = 1`; the weight `1 − A` is then 0 at the trough, never negative.

**Derived quantities** are never stored, to remove a class of inconsistency.
- During step `t` (the value of `state.step` at the start of the step), `age = t − 1 − birth_step` (int32).
  Founders therefore have age 0 during step 0. A child born during step `t` has age 0 during step `t + 1`, which
  is its first brain evaluation.
- `n_bonds = (partner ≥ 0).sum(1)`.

**Initial values** (`init_state`).
- Founders occupy rows `[0, n_init)` with `uid = arange(n_init)`, `parent_uid = −1`, `clade = arange(n_init)`,
  `generation = 0`, `birth_step = −1`, `energy = e_init`, random `pos`, `heading`, `tag`, `W1`, `W2` (§14.1), genes
  at their `*_init` values, and `brain_cost = l1_cost` of their rows. `next_uid = n_init`, `step = 0`, `diag = 0`,
  `ledger_cum = 0`, and the `env_*` fields at `default_env(cfg)`.
- Dead rows at init hold the canonical dead values, plus `pos = 0` (3D: `(0, 0, z_lo)`, §27.4), `heading = 0`,
  `tag = 0`, genes at their `*_init` values, `clade = 0`, `generation = 0`, `parent_uid = −1`, `birth_step = −1`,
  `brain_cost = 0` and zero weights. So every unqualified range invariant above also holds for dead rows at init, and
  it keeps holding: a dead row's genes, tag, pos and heading are either these values or the in-range values of the
  agent that last lived there.

**Memory.**
- Weights: `W1` is 101·32·2 = 6 464 B and `W2` is 33·35·2 = 2 310 B per agent (fp16).
- Other per-agent state: ≈ 290 B.
- Total ≈ **9.1 KB/agent**: 0.60 GB at 64k and 2.4 GB at 256k.
- Pending buffer: `M · 8.8 KB` (9 MB at 64k).
- Fields: < 5 MB.

### 2.3 Python types

```python
class State(NamedTuple):           # every field is an mx.array, in exactly the order of §2.2
    alive: mx.array; uid: mx.array; parent_uid: mx.array; clade: mx.array; generation: mx.array
    birth_step: mx.array; energy: mx.array; pos: mx.array; vel: mx.array; heading: mx.array
    h: mx.array; msg: mx.array; last_loss: mx.array; last_harvest: mx.array; last_inflow: mx.array
    partner: mx.array; rev: mx.array; tag: mx.array; mut_sigma: mx.array; mut_rate: mx.array
    offspring_frac: mx.array; brain_cost: mx.array; W1: mx.array; W2: mx.array
    pend_slot: mx.array; pend_W1: mx.array; pend_W2: mx.array
    resource: mx.array; chem: mx.array; patch_pos: mx.array; patch_vel: mx.array
    key: mx.array; step: mx.array; next_uid: mx.array; total: mx.array; total0: mx.array
    ledger_cum: mx.array; diag: mx.array
    env_regen_mult: mx.array; env_season_amp: mx.array; env_season_period: mx.array; env_drift_mult: mx.array
    env_patch_gain: mx.array; env_imm_request: mx.array; env_cat_center: mx.array; env_cat_radius: mx.array
    env_cat_kill_p: mx.array
    @property
    def env(self) -> Env: ...      # grouped read-only view of the env_* fields (Env.regen_mult, ...)
```

`NamedTuple` is a pytree that `mx.compile` accepts and returns (F12). Updates use `s._replace(...)`. There is no
class hierarchy anywhere. `State` and `Config` are the only central types; a 3D world uses `State3`, a parallel
NamedTuple with `State`'s fields first (§27.4).

---

## 3. Step pipeline

`step(s: State, cfg: Config) -> tuple[State, StepOut]` is a pure function. `make_step(cfg)` compiles it once per
config. The config is closed over as Python constants, so shapes are static and `key` and `step` are array inputs,
which means there is no retracing.

There are no host syncs, no `.item()`, no Python branching on array values and no data-dependent shapes inside the
step. The only Python branches are on config constants. Examples: `immigration_floor > 0` (or `immigration_every > 0`),
`director.enabled` and `runtime_checks == "cheap"`.

**Director branches.** Every read or write of `State.env`, the `CATASTROPHE` draw, the widened immigration gate
and the seasonal override sit behind a static `if cfg.director.enabled:`. With the director off the traced graph
is the one it was before the env fields existed, and the `env_*` arrays pass through unchanged. With it on and
inert env values, the trajectory is bit-identical to director-off (tested, `test_director_env.py`).

Ordered sub-steps:
- **"Start-of-step"** means the input `s`.
- Each sub-step reads only what it lists.
- **Key** refers to `ks[Stream.X]` (§14).

| # | sub-step (function) | reads | produces | key |
|---|---|---|---|---|
| S0 | **flush pending births**: `lifecycle.flush_pending` | `W1, W2, pend_*` | `W1, W2` with last step's child rows written in place. Everything after S0 reads only these. | — |
| S1 | **keys**: `rng.step_keys` | `key` | `key_next`, `ks (16,2)` | — |
| S2 | **neighbours**: `spatial.find_neighbors` | `pos, alive` | `nbr: Neighbors` | — |
| S3 | **bond geometry**: `bonds.bond_geometry` | `pos, partner` | `geom: BondGeom` (start-of-step offsets and lengths) | — |
| S4 | **field sampling**: `fields.cell_index`, `fields.sample_fields` | `pos, heading, resource, chem` | `fcell (N,)`, `fobs: FieldObs` | — |
| S5 | **sense**: `sensing.neighbor_descriptors`, `sensing.build_inputs` | `s`, `nbr`, `geom`, `fobs`, `age` | `desc (N,K,NF)`, `x (N,D_IN)` | `NOISE` |
| S6 | **think**: `brain.forward` | `W1, W2, x, h` | `h_new (N,H)`, `y (N,D_OUT)` | — |
| S7 | **decode and target**: `actions.decode` | `y, desc, nbr, partner, alive` | `act: Actions` (masked by `alive`) | — |
| S8 | **costs and solvency**: `energy.pay_costs`, then `actions.mask(act, solvent)` | `energy, act, n_bonds, brain_cost, alive` | `E`, `solvent`, `paid_basal`, `paid_action`, and `act` with all actions of insolvent agents cancelled | — |
| S9 | **bonds**: `bonds.break_bonds`, then `bonds.form_bonds` | `partner, rev, act.release, geom.length, act.bond_want, act.bond_target, alive & solvent` | `partner, rev`, broken and formed counts | `BOND_PRIO` |
| S10 | **harvest**: `energy.harvest` | `E, resource, fcell, act.harvest, solvent` | `E, resource, got` | — |
| S11 | **combat**: `energy.combat` | `E, resource, fcell, act`, post-S9 `partner` | `E, resource, lost, gain, spill, residue` (I13); attacks on a target that became a bond partner at S9 are dropped unless `attack_bonded` | — |
| S12 | **share**: `energy.share` | `E`, post-S9 `partner, rev`, start-of-step `s.partner`, `act.share`, `solvent` | `E, out, inflow, heat`; a slot sends only if its partner is unchanged since the start of the step | — |
| S13 | **cap**: `energy.cap_energy` | `E` | `E ≤ e_max`, `overflow` | — |
| S14 | **deaths**: `lifecycle.deaths` | `alive, solvent, E, lost, age, uid, generation, clade, partner, rev, resource, fcell`; director: `cat_mask` from `lifecycle.catastrophe_mask(pos, alive, env, ks[CATASTROPHE])` (start-of-step `pos`) | `survivor` mask (`alive` after deaths); corpses into `resource`; dead rows canonicalised; bonds of the dead dropped on both sides; death records | — |
| S15 | **physics**: `physics.forces`, `physics.integrate` | start-of-step `pos, heading`; `nbr`; post-S14 `partner`; `act.thrust, act.contract, act.grip, act.turn`; `survivor` | `pos, vel, heading` for survivors; `v_clips`. Right after S15, survivors' `h ← h_new`, `msg ← act.msg`, `last_loss/harvest/inflow` are written and everything else gets 0 (§16.3). | — |
| S16 | **reproduction and immigration**: `lifecycle.allocate_lanes`, `lifecycle.reproduce` | the in-flight state after S15, `act.reproduce`, `act.child_bond`, `solvent`; director: `env_imm_request` | newborn rows (small arrays), parents debited, child bonds, `pend_*` (child weights), birth records | `BIRTH_PRIO`, `PLACEMENT`, `MUT_W1`, `MUT_W2`, `MUT_GENES`, `IMMIGRANT` |
| S17 | **fields**: `fields.deposit_chem`, `fields.update_chem`, `fields.update_resource`, `fields.drift_patches` | `chem, act.emit, fcell, solvent, resource, patch_pos, patch_vel`; director: `env_regen_mult, env_season_*, env_patch_gain, env_drift_mult` | `chem, resource, patch_pos`, `regen_injected` | — |
| S18 | **bookkeeping**: `step.finish` (private) | everything, including `desc`, `x`, `act` and `as_prop` for the kin counters | `step + 1`; `key = key_next`; `total`; `ledger_cum`; `counters`; `diag` OR-ed with `invariants.device_check(...)`. Director: after S17 `step.consume_env` writes `env_imm_request ← max(request − n_imm, 0)` and `env_cat_kill_p ← 0` | — |

**Why this order.**
- **Decisions use the start-of-step snapshot.** Sensing, targets, bond geometry and forces all read start-of-step
  positions, so a step is a pure function of the previous state. Forces use the post-S14 bond topology: bonds
  formed this step act immediately, and bonds to the dead do not.
- **Costs before income.** An agent that cannot pay this step's costs is insolvent. It pays everything it has,
  all its actions are cancelled, and it dies at S14 even if a bond partner's share at S12 gives it energy (that
  inflow becomes part of its corpse). This removes the "free attack while dying" artefact.
- **Decisions refer to start-of-step bonds.** The brain chose `share[i,b]` and its attack target from the
  start-of-step bonds. S9 can empty a slot and refill it with a new partner in the same step, so S12 sends only along
  slots whose partner is unchanged, and S11 drops an attack whose target became a partner at S9 (unless
  `attack_bonded`).
- **Break before form.** Slots freed by a break are reusable in the same step.
- **Harvest before combat, combat before share.** Victims are robbed of what they hold after harvesting, and
  transfers along bonds see the post-combat balance.
- **Cap before deaths.** Energy leaves S13 in `[0, e_max]`.
- **Deaths before reproduction.** Freed rows are reusable, and bonds pointing at dead rows are dropped *before* a
  newborn can occupy the row. A stale pointer would otherwise silently bind to the newborn.
- **Physics before reproduction.** Children are placed next to the parent's new position and do not move in their
  birth step.
- **Fields last.** Agents see a fully updated field next step.
- **Heavy weights written at S0 of the next step.** A newborn's brain first runs in the next step's S6, so this is
  semantically invisible. It keeps `W1` and `W2` donatable (F1).

`StepOut` (returned and never fed back into the simulation):

```python
class BirthRecords(NamedTuple):   # (M,) each
    uid: mx.array; parent_uid: mx.array; clade: mx.array; generation: mx.array; slot: mx.array
    valid: mx.array               # bool: lane is a birth or an immigrant this step
class DeathRecords(NamedTuple):   # (DL,) each, DL = 2M
    uid: mx.array; age: mx.array; cause: mx.array; generation: mx.array; clade: mx.array; valid: mx.array
class StepOut(NamedTuple):
    counters: mx.array            # (N_COUNTERS,) int32, CounterIdx (§19.1)
    ledger: mx.array              # (16,) int64, this step's ledger (LedgerIdx, §10.8)
    births: BirthRecords
    deaths: DeathRecords
```

---

## 4. Spatial indexing: fixed-K neighbour search in pure MLX (2D; 3D: §27.2)

**Parameters.**
- `R = sensing.radius = 6.0`, which is 3 agent diameters.
- `G_h = floor(L/R)` and `cell_h = L/G_h ≥ R`, so the 3×3 block of cells covers the sensing disc.
- `C = sensing.cell_capacity = 16`, giving `Q = 9C = 144` candidates (2D; the 3×3×3 search of §27.2 has 27C).
- `K = sensing.k_neighbors = 8`.
- `validate` requires `G_h ≥ 3` (otherwise the 3×3 stencil would visit a cell twice), `NS ≤ K ≤ Q`, and
  `R ≥ max(bonds.form_range, combat.range, 2·physics.radius)`, so every interaction partner is inside the sensing
  radius.

```python
def find_neighbors(pos, alive, cfg) -> Neighbors:
    G, C, K, L, R = d.g_hash, d.c, d.k, cfg.world.size, cfg.sensing.radius
    cxy   = mx.floor(pos / d.cell_hash).astype(i32) % G                   # (N,2); % G maps pos≈L to cell 0
    cid   = mx.where(alive, cxy[:, 1] * G + cxy[:, 0], G * G)             # dead → sentinel after all real cells
    order = mx.argsort(cid).astype(i32)                                   # (N,)
    start = mx.searchsorted(cid[order], mx.arange(G * G + 1, dtype=i32), side="left").astype(i32)
    count = start[1:] - start[:-1]                                        # (G²,) occupancy of real cells
    spos  = pos[order]                                                    # cell-sorted copy: runs of C are contiguous
    inv   = ops.scatter_add_i32(N, order, mx.arange(N, dtype=i32))        # inv[i] = sorted rank of i (unique idx)
    ncx   = (cxy[:, None, 0] + DX9) % G;  ncy = (cxy[:, None, 1] + DY9) % G   # (N,9), DX9/DY9 = 3×3 offsets, row-major
    ncid  = ncy * G + ncx
    s, e  = start[ncid], start[ncid + 1]                                  # (N,9)
    slot  = (s[..., None] + mx.arange(C, dtype=i32)).reshape(N, Q)       # (N,Q) sorted positions
    ok    = (slot < mx.repeat(e, C, axis=1)) & (slot != inv[:, None]) & alive[:, None]
    sl    = mx.minimum(slot, N - 1)
    off   = ops.min_image(spos[sl] - pos[:, None, :], L)                  # (N,Q,2)  neighbour − self
    d2    = (off * off).sum(-1)
    d2    = mx.where(ok & (d2 < R * R), d2, mx.inf)
    part  = mx.argpartition(d2, kth=K - 1, axis=1)[:, :K].astype(i32)     # K smallest, unordered
    pd2   = mx.take_along_axis(d2, part, 1)
    o     = mx.argsort(pd2, axis=1).astype(i32)                           # nearest first
    part, pd2 = mx.take_along_axis(part, o, 1), mx.take_along_axis(pd2, o, 1)
    valid = pd2 < mx.inf
    self_ = mx.arange(N, dtype=i32)[:, None]
    idx   = mx.where(valid, order[mx.take_along_axis(sl, part, 1)], self_)  # ids resolved only for K winners
    offk  = mx.where(valid[..., None], mx.take_along_axis(off, part[..., None], 1), 0.0)
    dist  = mx.where(valid, mx.sqrt(pd2), R)
    return Neighbors(idx, valid, offk, dist,
                     overflow_cells=(count > C).sum().astype(i32), max_occupancy=count.max().astype(i32))
```

`Neighbors(NamedTuple)` has these fields:
- `idx (N,K) int32`. Invalid entries hold **the agent's own index**, so gathers never go out of bounds.
- `valid (N,K) bool`. Every consumer must AND with `valid`.
- `off (N,K,2) f32`, the minimum-image vector from me to the neighbour; 0 where invalid (3D: (N,K,3), §27.2).
- `dist (N,K) f32`; `R` where invalid.
- `overflow_cells () int32` and `max_occupancy () int32`.

Rows are sorted by distance, ascending. Among exactly equal distances the order is defined by the backend's
sort, which is deterministic and involves no atomics.

**Complexity.**
- Sorting: one N-element argsort, plus a `(G²+1)` searchsorted.
- Gathers and distances: O(N·Q).
- Selection: an O(N·Q) partition, which is a per-row sort of 144 on Metal (F16).
- Transient memory at 64k: `slot` 38 MB, `off` 75 MB, `d2` 38 MB, `part` 2 MB. That is ≈ 0.15–0.25 GB.

**Sizing C from physics** (this corrects the miscounts in all three proposals, and the review's crowd finding).
- Hexagonal packing at spacing `s` has density `2/(√3·s²)`. A hash cell has area `cell_h² ≈ 36.3 u²`. C = 16
  overflows when the local spacing drops below ≈ 1.62 u.
- At natural spacing `s = 2r = 2.0` that is **10.5 agents per cell**.
- A fully contracted body has rest length `L0·(1 − contract_gain) = 1.5`. It settles where spring tension balances
  repulsion: `k_s(d − 1.5) = k_r(2 − d)`, so `d = (1.5·0.05 + 2·0.08)/0.13 = 1.81`. That gives **12.8 per cell**.
- **Thrust is the real threat, not bodies.** One agent thrusting into a free neighbour already compresses the pair
  to `d = 1.375` (0.96 if the neighbour grips), i.e. ≈ 22 per cell. A crowd that thrusts toward a common point (a
  resource peak) is worse: thrust acts as a body force, so the pressure at the centre grows with the crowd's radius,
  and no finite soft repulsion can hold it. Once a cell holds more than C agents, the agents past the first C
  cannot see each other, stop repelling, and pile in further. Measured with plain thrust: 183–300 agents in one
  hash cell (F18).
- **Fix: contact-attenuated thrust (§9).** An agent's thrust is scaled by `max(0, 1 − n_contacts/thrust_contact_limit)`,
  with `thrust_contact_limit = 4`, so only agents on the surface of a crowd push. The pressure is then bounded by
  a surface term that does not grow with crowd size. Measured under maximal converging thrust: median spacing
  1.77–1.88, ≤ 15 agents per hash cell, ≤ 8 per field cell, zero overflow, for 300 and 1 000 agents with or without
  grip (F18).
- So `C = 16` covers relaxed bodies, contracted bodies and maximal converging crowds, the last with a thin margin
  (15 of 16). Transient overflow stays possible and is a non-fatal warning (failure mode 1).
- The `fast` overlay (`C = 12`) accepts overflow inside dense crowds in exchange for −25% neighbour traffic.

**Failure modes and handling.**
1. **Cell overflow** (`count > C`). Only the first C agents of a cell, in sorted order, are visible. This is
   deterministic but arbitrary within the cell. It is counted every step (`CELL_OVERFLOW`, `MAX_OCCUPANCY`), and
   the diag warning bit `CELL_OVERFLOW_WARN` is set. The truncated list is still a valid subset, which is tested.
   Overflow is self-reinforcing (invisible agents do not repel), which is why thrust is contact-attenuated (§9);
   `test_crowd.py` asserts that a converging crowd at full thrust never overflows `C = 16`.
2. **K saturation.** In dense clusters only the K nearest are sensed and repelled, so the neighbour relation is not
   symmetric. Repulsion can then be slightly asymmetric. This only matters when one agent has ≥ 8 neighbours closer
   than 2r. It is accepted and documented. Bond forces do not depend on neighbour lists and are exactly
   antisymmetric.
3. **Degenerate grids.** `G_h < 3` is rejected by `validate`.
4. **Positions at the seam.** The `ops.wrap` guard (F8) keeps `pos < L`, and `% G` is a second guard.
5. **Empty world.** All lists are invalid and every consumer is masked. This is tested.

---

## 5. Sensing: exact input vector layout (`D_IN = 69`, float32; 2D, 3D: §27.1)

Notation:
- **Body frame:** `fwd = (cos θ, sin θ)` and `lat = (−sin θ, cos θ)`, so `to_body(v) = (v·fwd, v·lat)` (2D; the
  3D frame adds pitch and an `up` axis, §27.2).
- **Log ratio:** `lr(a, b) = clip(log((a + q0)/(b + q0)) / LR_SCALE, −1, 1)` with `q0 = max(energy.basal_cost, 1)`
  (the `max` keeps `lr` and `IN_ENERGY` defined when `basal_cost = 0`) and `LR_SCALE = 2`. Likewise a zero
  `attack_max`, `harvest_max` or `share_rate` is read as 1 in `IN_HURT`, `IN_FED` and `IN_GIFT`.
- **Kin similarity:** `sim(i, j) = exp(−‖tag_i − tag_j‖² / tag_scale²)` with `tag_scale = 0.5`. It is 1 for
  clones. For unrelated founders it is ≈ e⁻¹⁰.
- **Missing entities** (absent neighbour, empty bond slot) give an all-zero block, and an explicit `present` feature
  marks presence.

The index constants live in `layout.py` (`IN_*`). Tests import them by name.

| idx | name | definition |
|---|---|---|
| 0 | `IN_BIAS` | 1.0 |
| 1 | `IN_ENERGY` | `log1p(E/q0) / log1p(e_max/q0)` ∈ [0,1] |
| 2 | `IN_AGE` | `tanh(age / AGE_SCALE)`, `AGE_SCALE = 1000` |
| 3, 4 | `IN_VEL` (fwd, lat) | `clip(to_body(vel) / v_max, −1, 1)` |
| 5 | `IN_NBONDS` | `n_bonds / B` |
| 6 | `IN_HURT` | `min(last_loss / attack_max, 1)` |
| 7 | `IN_FED` | `min(last_harvest / harvest_max, 1)` |
| 8 | `IN_GIFT` | `min(last_inflow / share_rate, 1)` |
| 9 | `IN_NOISE` | `U[−1, 1)`, drawn from stream `NOISE`. Breaks symmetry between clones |
| 10, 11 | `IN_CLOCK` (sin, cos) | `sin, cos(2π·age / clock_period)`, `clock_period = 32` |
| 12 | `IN_RES` | `min(R_cell / resource_cap, RES_SAT) / RES_SAT`, `RES_SAT = 2` |
| 13, 14 | `IN_RES_GRAD` (fwd, lat) | `g = ((R[y,x+1] − R[y,x−1]), (R[y+1,x] − R[y−1,x])) / (2·resource_cap)`, then `clip(RES_GRAD_GAIN·to_body(g), −1, 1)`, `RES_GRAD_GAIN = 4` |
| 15–17 | `IN_CHEM` (c = 0..2) | `c / (c + chem_half_sat)`, with `chem_half_sat = 1.0` |
| 18–23 | `IN_CHEM_GRAD` (c0 fwd, c0 lat, c1 fwd, c1 lat, c2 fwd, c2 lat) | `tanh(to_body(central difference / 2) / chem_half_sat)` |
| 24 | `IN_CROWD` | `Σ valid / K` |
| 25 | `IN_KIN_MEAN` | mean of `sim(i, j)` over valid neighbours (0 if none) |
| 26, 27 | `IN_MSG_MEAN` | mean of `msg_j[0]` and `msg_j[1]` over valid neighbours (0 if none) |
| 28 + 7n + f | `IN_NBR` (n = 0, 1, 2 nearest) | descriptor feature f (table below), copied from `desc[:, n, f]` |
| 49 + 5b + f | `IN_BOND` (b = 0..3) | f: 0 `present`; 1, 2 `clip(to_body(geom.off) / break_length, −1, 1)`; 3 partner `msg[0]`; 4 `lr(E_partner, E_i)` |

**Neighbour descriptor** `desc (N,K,NF=7)` (`sensing.neighbor_descriptors`; 3D: `NF = 8` with `NBR_DZ` last, §27.2).
It is computed for all K neighbours, used for sensing (first NS) and targeting (all K), and zeroed where invalid.

| f | name | definition |
|---|---|---|
| 0 | `NBR_PRESENT` | 1.0 |
| 1 | `NBR_DX` | `off·fwd / R` |
| 2 | `NBR_DY` | `off·lat / R` |
| 3 | `NBR_KIN` | `sim(i, j)` |
| 4 | `NBR_RELE` | `lr(E_j, E_i)` |
| 5 | `NBR_MSG0` | `msg_j[0]` |
| 6 | `NBR_BONDED` | 1.0 if `j ∈ partner[i, :]` ((N,K,B) compare) |

The targeting descriptor is `φ = desc[..., 1:]` (F = 6 dims; 7 in 3D, §27.2).

Field samples use the agent's start-of-step field cell `fcell = cy·FG + cx`, where `(cx, cy) = floor(pos / field_cell) mod FG`.
`fields.sample_fields` computes rows 12–23 completely (gather, gradient, body-frame rotation and the normalisation
above) and returns them as `FieldObs` (§16.2); `sensing.build_inputs` copies them unchanged, so the normalisation
lives in exactly one module.
Gradients use the four periodic neighbour cells. Bond-slot blocks keep their slot index for the bond's whole
lifetime, so a recurrent brain can attach roles to particular slots.

**Contract.** `x` is finite and `|x| ≤ 1` element-wise (`tanh`, `clip` and `min` guards). `test_sensing.py` asserts
this, and checks every index on hand-built scenes. The bond-offset clip matters: sensing reads start-of-step lengths,
and a bond that survived the previous S9 at length ≤ `break_length` can be stretched to `break_length + 2·v_max = 5`
by that step's physics.

The named constants `LR_SCALE`, `AGE_SCALE`, `RES_SAT` and `RES_GRAD_GAIN` live in `layout.py`, because they shape
the genome's inputs.

**Traffic.**
- Gathers of shape (N,K): `tag` (4 floats), `energy`, `msg`. Gathers of shape (N,B): `msg`, `energy`.
- Field: 5 gathers × 4 fields.
- Output: `x` is (N,69) f32.
- Total ≈ 60 MB per step at 64k.

---

## 6. Brain / controller

**Architecture.** A per-agent Elman RNN with one hidden layer. Biases are folded in through the constant input and a
constant hidden unit.

```text
z  = concat(x (D_IN), h (H))                 (N, W1_ROWS = 101)
h' = tanh(z @ W1)                            W1: (N, 101, 32)
y  = concat(h', 1) @ W2                      W2: (N, 33, 35)     raw logits, decoded in §7
```

- **Parameters:** `P = 101·32 + 33·35 = 4 387` per agent, which is 8.8 KB in fp16.
- **Storage:** two dense tensors, not one flat genome. Reshaping a slice of a flat genome would force a copy every
  step.

**Batched evaluation (`brain.forward`).**
- `impl = "matmul"` (default) computes `(z.astype(wdtype)[:, None, :] @ W1)[:, 0]`.
- `impl = "einsum"` computes `mx.einsum("nr,nrh->nh", z.astype(wdtype), W1)` (the same operand dtypes as
  `matmul`, so the A/B compares kernels, not an implicit f32 upcast of `W1`).
- `tanh` and all decoding run in float32. Outputs are cast with `.astype(mx.float32)`.
- Two heavy kernels read every weight exactly once per step, so the brain is memory-bound (§15).
- `alife bench --brain-variants` A/B-tests both `impl` values and dtypes on Metal. The winner is set in the preset;
  the code does not change.
- Dead rows are evaluated too, because shapes are fixed, and masked afterwards. Compacting the live rows would cost
  a gather larger than the masked work.

**Recurrent memory.** `h` persists across steps. It is set to 0 for non-survivors right after S15 (§16.3), and to 0
for newborns in S16.

**Init (`brain.init_weights`, founders and immigrants).**
- `W1 ~ N(0, init_scale² / W1_ROWS)`. The recurrent rows `D_IN..W1_ROWS−1` are additionally multiplied by
  `recurrent_scale = 0.5`.
- `W2 ~ N(0, init_scale² / W2_ROWS)`.
- Values are clipped to `±weight_clip = 4` and cast to `wdtype`.
- **No output bias is seeded.** Founders are fully random.
- Founder rows `[0, n_init)` are initialised and all other rows are zero.
- Keys: `InitStream.W1` and `InitStream.W2`.

**Weight cost (the per-weight metabolic cost).**
- `brain_cost = floor(brain_cost_per_l1 · (Σ|W1_row| + Σ|W2_row|))`, computed in f32 from the **stored** (cast)
  weights.
- It is computed **once at birth**, only on the M lane rows. There is never a per-step pass over all weights.
- With `brain_cost_per_l1 = 0.1`, a founder (L1 ≈ 370) pays ≈ 37 q/step. That is 37% of basal metabolism, so
  sparse, small-magnitude brains are selected for.

**Mutation (`brain.mutate_rows`, M lanes, all in float32, then cast to `wdtype`):**

```text
W_c = clip(W_p + 1[u < p_c] · σ_c · ε, −weight_clip, weight_clip)      u ~ U[0,1), ε ~ N(0,1), per weight
```

- Keys: `ks[MUT_W1]` and `ks[MUT_W2]`. Each is split into `(k_mask, k_eps)` with `mx.random.split(k, 2)`.
- `σ_c` and `p_c` are the child's already-mutated genes (§12.4).
- **fp16 caveat.** Near |w| ≈ 2–4 the fp16 spacing is 0.002–0.004, so perturbations smaller than half a spacing
  round away. This is an effective σ floor for large weights. It is accepted and documented.
  `weight_dtype = "float32"` removes it at twice the brain bandwidth.

---

## 7. Actions: output layout and decoding (`D_OUT = 35`; 2D, 3D: §27.1)

`actions.decode(y, desc, nbr, partner, alive, cfg) -> Actions` is pure and elementwise apart from target
selection. Binary decisions are deterministic thresholds. Exploration comes from mutation and from the noise
input, never from action sampling.

| idx | `OUT_*` | decode | meaning / used in |
|---|---|---|---|
| 0 | `THRUST` | `a = tanh(y)` ∈ (−1,1) | force `thrust_max·a·fwd` (S15) |
| 1 | `TURN` | `turn_max·tanh(y)`, `turn_max = 0.3` rad | heading change (S15) |
| 2, 3 | `MSG` | `tanh(y)` | broadcast, stored in `msg`, read by others next step |
| 4–6 | `EMIT` | `max(tanh(y), 0)` ∈ [0,1) | chemical deposit per channel (S17) |
| 7 | `HARVEST` | `sigmoid(y)` | harvest effort (S10) |
| 8 | `GRIP` | `sigmoid(y)` | substrate grip, which raises own drag (S15) |
| 9 | `CONTRACT` | `tanh(y)` ∈ (−1,1) | bond rest-length modulation, i.e. muscle (S15) |
| 10 | `BOND` | `y > 0` | willing to propose or accept a bond (S9) |
| 11–14 | `RELEASE` | `y > bonds.release_threshold (1.0)` | release bond slot b (S9) |
| 15–18 | `SHARE` | `max(tanh(y), 0)` | per-slot share intensity (S12); sent only if the slot's partner is unchanged since the start of the step |
| 19 | `ATTACK` | `y > combat.threshold (0.5)` and a target exists | attack (S11); dropped there if the target became a bond partner at S9 and `attack_bonded` is false |
| 20 | `DEFEND` | `sigmoid(y)` | damage reduction, costly (S11) |
| 21 | `REPRODUCE` | `y > 0` | request division (S16) |
| 22 | `CHILD_BOND` | `y > 0` | the child stays bonded to the parent (S16) |
| 23–28 | `Q_BOND` | raw ℝ⁶ | bond-target query |
| 29–34 | `Q_ATTACK` | raw ℝ⁶ | attack-target query |

**Evolvable targeting (attention over neighbours).** The descriptor `φ_k ∈ ℝ⁶` is defined in §5 (ℝ⁷ in 3D,
where each query gains its `*_DZ` column, §27.2). Target selection works as follows.

```python
has_free  = (partner < 0).any(1)
can_bond  = alive & (y[:, OUT_BOND] > 0) & has_free                         # (N,) for every row
bonded    = desc[..., NBR_BONDED] > 0.5
elig_b    = nbr.valid & (nbr.dist < form_range) & ~bonded & can_bond[nbr.idx]
elig_a    = nbr.valid & (nbr.dist < combat.range) & (attack_bonded | ~bonded)
score_b   = mx.where(elig_b, (phi * q_bond[:, None, :]).sum(-1), -mx.inf)
bond_k    = mx.where(elig_b.any(1) & bond_want, mx.argmax(score_b, 1).astype(i32), -1)
bond_target = mx.where(bond_k >= 0, take(nbr.idx, bond_k), -1)                 # same pattern for attack
```

- Score ties go to the lowest k, i.e. the nearest neighbour (first-index `argmax`, F5).
- If nothing is eligible, the action is cancelled (`attack = False`, `bond_target = −1`).
- Kin preference, energy preference and signal-gated partnering are therefore continuous heritable traits. For
  example, a negative weight on `sim` in `q_attack` means "eat strangers".

```python
class Actions(NamedTuple):      # all (N,...) and already masked by `alive`
    thrust: mx.array; turn: mx.array; msg: mx.array; emit: mx.array; harvest: mx.array
    grip: mx.array; contract: mx.array; bond_want: mx.array; release: mx.array   # release (N,B) bool
    share: mx.array             # (N,B) f32
    attack: mx.array; defend: mx.array; reproduce: mx.array; child_bond: mx.array
    bond_k: mx.array; bond_target: mx.array; attack_k: mx.array; attack_target: mx.array   # int32, −1 = none
def decode(y, desc, nbr, partner, alive, cfg) -> Actions
def mask(act: Actions, keep: mx.array) -> Actions   # bools → False, floats → 0, k/targets → −1 where ~keep
```

The 3D-only outputs (`OUT_PITCH`, `OUT_BUOYANCY`) decode into the parallel `Motion3`, and `mask` returns the type it
is given (§27.2).

---

## 8. Bonds

### 8.1 Representation and symmetry invariant

- `partner (N,B) int32` and `rev (N,B) int32`, with −1 meaning empty.
- For every occupied `(i, b)` with `j = partner[i,b]` and `r = rev[i,b]`, the invariant is:
  `alive[i] ∧ alive[j] ∧ j ≠ i ∧ 0 ≤ r < B ∧ partner[j,r] = i ∧ rev[j,r] = b`.
- There are no duplicate partners in a row, and `partner < 0 ⇔ rev < 0`.
- The back-pointer makes every bond operation a **gather** from flat index `j·B + r`. No scatter is involved, so
  there are no conflicts.
- Rest length is not stored. It is a symmetric function of both ends' `contract` (§9).

### 8.2 Breaking (S9a): OR-symmetric, per slot

```python
flag  = valid & (act.release | (geom.length > break_length))         # (N,B), start-of-step lengths
other = flag.reshape(-1)[(max(partner,0)*B + max(rev,0)).reshape(-1)].reshape(N,B)
brk   = valid & (flag | other)                                        # identical decision on both ends
partner, rev = where(brk, -1, partner), where(brk, -1, rev)
```

Either end can break a bond unilaterally (fission). Both rows clear it in the same step whatever the
floating-point details, because the decision is the OR of two booleans.

### 8.3 Formation (S9b): propose / accept / mutual handshake with consent

Each agent gains **at most one** bond per step (F9). The steps are:

```python
ff, has_free = ops.first_free(partner)                    # first free slot (argmax of partner<0), post-break
can   = can_bond & has_free                              # can_bond = alive & solvent & act.bond_want (argument)
tgt   = mx.where(can & (target >= 0), target, -1)        # target = act.bond_target (argument)
tgt   = mx.where((tgt >= 0) & can[max(tgt,0)], tgt, -1)   # a target must itself be able to bond now
prop  = tgt >= 0
prio  = mx.random.permutation(N, key=ks[BOND_PRIO]).astype(i32)      # unique priorities in [0,N)
best  = ops.scatter_max_i32(N, tgt, prio, prop)            # best proposer priority per target; −1 if none
won   = prop & (best[max(tgt,0)] == prio)                  # I am the winning proposer at my target
winner = ops.scatter_max_i32(N, tgt, arange(N), won)       # ≤ 1 winner per target, so this is a placement
can_accept = can & ~won                                    # a winning proposer never accepts
as_prop = won & (can_accept[max(tgt,0)] | ((tgt[max(tgt,0)] == arange(N)) & won[max(tgt,0)]))   # accepted, or mutual
as_acc  = can_accept & (winner >= 0)
new_p   = mx.where(as_prop, tgt, mx.where(as_acc, winner, -1))          # never both (proved, tested)
put     = (new_p >= 0)[:, None] & (arange(B)[None, :] == ff[:, None])
partner = mx.where(put, new_p[:, None], partner)
rev     = mx.where(put, ff[max(new_p,0)][:, None], rev)
```

**Why this is conflict-free.**
- An agent pairs as a proposer only if it is the unique winner at its target. It pairs as an acceptor only with its
  unique winner, and only if it did not itself win elsewhere.
- So each agent ends up in at most one new pair.
- Each agent writes only its own row, into its own first free slot. It learns the partner's slot by gathering
  `ff[new_p]`.
- The pair's view is symmetric: if `i` proposed to `j` and `j` accepted, then `as_prop[i]` and `as_acc[j]` with
  `winner[j] = i`.

**Consent.** Both ends must be `bond_want` (willing), solvent, have a free slot, and (for the proposer) have chosen
the target by query.

**Duplicates and self-bonds.** These are impossible: targets exclude current partners (`NBR_BONDED`) and self
(self is never in `nbr`).

**Priorities.** These come from a permutation, so they are unique with no index bias (F4).

### 8.4 Child bonds (S16)

**Parent side.** Let `child_of[p]` be the child's row for parents (full-N map, −1 otherwise), and recompute
`ff, has_free` after S9 and S14. Then:

```python
cb = (child_of >= 0) & act.child_bond & has_free
partner = where(cb[:,None] & (arange(B)[None,:] == ff[:,None]), child_of[:,None], partner)
rev     = where(..., 0, rev)
```

**Child side.** For newborn rows whose parent has `cb` set: `partner[c,0] = parent` and `rev[c,0] = ff[parent]`.

Each parent has at most one child per step and child rows are fresh, so this is conflict-free. The bond starts
unstressed, because `offspring_distance = rest_length = 2.0`. Budding with `child_bond` is the main route to clonal
multicellular bodies. Adult handshakes let bodies merge and form chimeras.

### 8.5 Death cleanup (S14)

```python
keep = (partner >= 0) & survivor[:, None] & survivor[max(partner, 0)]
partner, rev = where(keep, partner, -1), where(keep, rev, -1)
```

Both sides drop the bond by gather. This runs before any birth reuses a row.

### 8.6 Max bonds, splitting bodies, what flows along bonds

**Max bonds.** `B = 4` gives chains, branched trees and 2-D sheets.

**Splitting bodies.**
- **Fission:** any per-slot release or overstretch (`> break_length = 4.0`) cuts the body. The fragments become new
  connected components (§19).
- **Division:** reproduction (§12), optionally bonded.

**What flows along bonds.**
- **Force:** springs with brain-controlled rest length (§9).
- **Energy:** per-slot `share` with 90% efficiency (§10.6), only along slots whose partner the brain saw at the
  start of the step.
- **Information:** per-slot partner `msg[0]`, relative position and relative energy (§5 inputs 49–68).
- **Upkeep:** `bond_upkeep = 5 q` per bond end per step.

### 8.7 Device symmetry check

`bonds.symmetric_ok(partner, rev, alive) -> mx.array (bool scalar)` checks everything in §8.1 with
O(N·B) gathers plus an (N,B,B) duplicate compare. It runs every step inside `device_check` (§22.3).

---

## 9. Physics integration (S15; 2D, 3D: §27.3)

**Model.** Overdamped (Stokes) dynamics: velocity is proportional to force, with no inertia. This suits cell-scale
agents, cannot oscillate, and has a simple stability criterion.

Forces are evaluated at start-of-step positions using the post-S14 bond topology.

```text
F_i = thrust_max · a_i · att_i · fwd_i                                            thrust_max = 0.1
    + Σ_b valid  k_s · (d_ib − ℓ_ib) · û_ib                                         k_s = bonds.k_spring = 0.05
    + Σ_k valid ∧ survivor[j] ∧ d_ik < 2r :  −k_r · (2r − d_ik) · û_ik             k_r = physics.k_repel = 0.08
ℓ_ib = L0 · (1 + contract_gain · (c_i + c_j)/2)       L0 = 2.0, contract_gain = 0.25  → ℓ ∈ [1.5, 2.5]
γ_i  = drag · (1 + grip_gain · grip_i)                drag = 1.0, grip_gain = 4.0      → γ ∈ [1, 5]
v_i  = F_i / γ_i ;   v_i ← v_i · min(1, v_max/‖v_i‖)    v_max = 0.5  (each clip counted in V_CLIPS)
x_i  ← wrap(x_i + v_i) ;   θ_i ← wrap_angle(θ_i + turn_i) ;   vel_i ← v_i          (survivors only)
att_i = max(0, 1 − n_contact_i / thrust_contact_limit)                             thrust_contact_limit = 4
n_contact_i = #{k : valid ∧ survivor[j] ∧ d_ik < 2r}                               (the repulsion mask)
```

`û_ib` is the unit vector `min_image(x_j − x_i)/max(d, EPS_LEN)` with `EPS_LEN = 1e-6` (`layout.py`). Non-survivors
get `vel = 0` and keep their position.

**Contact-attenuated thrust** (F18). Thrust is scaled down by the number of contacts the agent already has, so in
a crowd only the surface pushes. Without it, a crowd thrusting toward one point is compressed by a body force that
grows with its radius, collapses below the hash-cell capacity, and then loses its own repulsion (§4). A free agent
and an agent with one contact keep 100% and 75% of their thrust, so terminal speed and single-agent behaviour are
unchanged. Inside a body, contacts only exist below the relaxed rest length, so a contracted body moves mainly by
contraction and grip. The factor is bounded forcing, not stiffness, so the stability argument below is unaffected.
Newborns are placed in S16 and are not integrated in their birth step.

**Locomotion: the fix for a fatal flaw shared by all three proposals.**
- Spring forces are internal and exactly antisymmetric. With uniform drag, `Σ_i γ v_i = Σ_i F_thrust,i`, so
  contraction alone moves the centre of mass by exactly 0 (F11).
- `grip` makes drag per-agent and time-varying. Then `Σ_i γ_i v_i = Σ thrust`, but the *geometric* centre moves when
  grip is phase-locked to contraction, i.e. inchworm crawling.
- In the prototype, a two-cell body moved 4.8 u in 2000 steps with zero thrust. With grip off it moved 0.000.
- So "exert forces" (contract and grip) is genuinely a locomotion channel, distinct from thrust.
- Grip costs energy (§10.3).

**Stability (rigorous for the linearised system).**
- Explicit Euler on `ẋ = F(x)/γ` is stable if `λ_max(stiffness)/γ_min < 2`.
- Each spring contributes a Hessian block with eigenvalues ≤ `k_s`. Its transverse geometric term
  `k_s(1 − ℓ/d) < k_s`.
- Each contact contributes eigenvalues ≤ `k_r`. Its transverse term is negative.
- A node has at most B springs and at most K contacts (repulsion is computed only over its K neighbours).
- By Gershgorin, `σ = 2·(B·k_s + K·k_r)/drag = 2·(0.2 + 0.64) = 1.68 < 2`. `validate` requires `σ ≤ 1.8`. Grip only
  increases γ, which only helps.
- Realistic hexagonal contact geometry gives `λ_max = 6(k_s + k_r) = 0.78` (emergence V8), a 2.6× margin.
- The `v_max` clip is a safety net, not part of the stability argument.

**Other validated relations.**
- `thrust_max < 2r·k_r` (0.1 < 0.16): repulsion can always stop full thrust before agents fully overlap.
- `v_max ≤ r`: no tunnelling.
- `L0·(1 + contract_gain) < break_length`.
- `offspring_distance < break_length`.

**Terminal speed** under full thrust with no grip is 0.1 u/step, so crossing the 64k world takes about 10⁴ steps.

All physics is gathers plus elementwise work. There is no float scatter, so it is deterministic.

---

## 10. Energy model and exact ledger

### 10.1 Representation

- All conserved energy is **int32 quanta**: `energy (N,)` and `resource (FG,FG)` (2D; 3D (FZ,FG,FG), §27.5).
- Every agent also holds a fixed **structural energy** `S = energy.structure = 10 000 q`. It is paid by the parent
  at birth and returned to the field as part of the corpse, which makes corpses food.
- The conserved total is
  `TOTAL = exact_sum(energy) + S · n_alive + exact_sum(resource)` (int64; `state.total_energy`).
- Chemicals are not energy.

**Rules.**
1. Every float→int conversion is `floor` followed by a clamp to what is actually available.
2. Every debit has an equal credit: another agent, the field, or a named heat sink.
3. Every contested scatter is an int32 scatter-add (`ops.scatter_add_i32`). Integer addition is associative, so the
   ledger closes **bit-exactly** on any backend and in any atomic order.
4. Fractions of integers use Q8 fixed point: `mul_q8(x, q) = (x·q) >> 8` with `q = round(f·256)` computed on the host.

### 10.2 Sources

- **Regeneration (S17)** (§13.2). Each cell gains
  `inc = clip(min(regen_rate_c, resource_cap − R_c), 0, ∞)`, giving `INJECT_REGEN = Σ inc`. Regeneration stops at
  `resource_cap`. Corpses and spills may push `R` above the cap; regeneration is then 0.
- **Immigration (S16, optional).** Each immigrant brings `e_init + S`, giving `INJECT_IMMIGRATION`.

### 10.3 Costs (S8, `energy.pay_costs`), per alive agent per step

```text
basal_due  = basal_cost(100) + brain_cost + bond_upkeep(5)·n_bonds                  → HEAT_BASAL
action_due = floor(move_cost(100)·a²) + floor(turn_cost(10)·|turn|/turn_max) + floor(grip_cost(20)·grip)
           + floor(contract_cost(5)·|contract|·n_bonds) + floor(msg_cost(5)·Σ|msg|) + floor(emit_cost(20)·Σ emit)
           + floor(harvest_cost(10)·harvest) + floor(defend_cost(20)·defend) + attack_cost(50)·[attack]   → HEAT_ACTION
solvent     = alive & (E ≥ basal_due + action_due)
paid_basal  = min(basal_due, E);   paid_action = min(action_due, E − paid_basal);   E ← E − paid_basal − paid_action
```

- Each `floor` term is computed in f32 and cast to int32.
- `n_bonds` counts start-of-step bonds.
- **Insolvent** agents (`alive & ~solvent`) end S8 at `E = 0`. `actions.mask(act, solvent)` cancels every action
  they chose this step: no harvest, attack, share, bond, reproduce or emit, and zero thrust, turn, contract and
  grip. They **always** die at S14 with cause `STARVED`: the death rule includes `~solvent` (§10.7), so a bond
  partner's share at S12 cannot rescue them; any inflow becomes part of their corpse. Hence
  `DEATHS_STARVED ≥ INSOLVENT` every step.

### 10.4 Harvest (S10, `energy.harvest`)

```text
demand = clip(min(floor(harvest·harvest_max(400)), e_max − E), 0, harvest_max)     for solvent agents, else 0
take, taken = ops.proportional_take(demand, fcell, solvent, resource.flat)
E += take;  resource.flat −= taken;  got = take                                     → FLOW_HARVEST
```

**`ops.proportional_take(demand, target, active, supply)`** is the single contention primitive (F10):

```python
ONE = 1 << 15
d = where(active, demand, 0);  t = where(active, target, 0)            # demand ∈ [0, 2^16), supply ∈ [0, 2^24)
D = scatter_add_i32(P, t, d)                                            # total demand per pool
ratio = floor(f32(supply) * ONE / f32(max(D, 1))).astype(i32) - 1      # −1 absorbs f32 rounding
ratio = where(D <= supply, ONE, clip(ratio, 0, ONE))
take  = (d * ratio[t]) >> 15                                            # d·ratio < 2^31
taken = scatter_add_i32(P, t, take)
```

Guarantees, proved and property-tested:
- `0 ≤ take ≤ demand`.
- `taken ≤ supply` per pool.
- Uncontested pools (`D ≤ supply`) are served exactly.
- Unallocated remainder in a contested pool is ≤ contenders + 2D/2¹⁵ + 1 quanta, and it stays in the pool.

**Why the proof holds.** `supply` < 2²⁴ is exact in f32. The scaled ratio is ≤ 2¹⁵, so its f32 error is < 2⁻⁷ and
the floor overshoots by at most 1, which the −1 absorbs. The algorithm is order-independent.

**Precondition:** per-pool `D < 2³¹`, i.e. fewer than 2¹⁵ contenders per pool. It is monitored by the
`INT_HEADROOM` diag bit (§10.9).

### 10.5 Combat (S11)

See §11. Theft moves energy agent → agent (efficiency 0.7). The other 30% spills into the field at the victim's
cell. All flows are internal.

### 10.6 Share along bonds (S12, `energy.share`): gather-only, per slot

```text
same[i,b] = (partner[i,b] == partner0[i,b])            partner0 = start-of-step s.partner; partner, rev = post-S9
out[i,b] = where(same[i,b] & partner[i,b] ≥ 0 & solvent_i, min(floor(share[i,b]·share_rate(1000)), E_i // B), 0)
in[i,b]  = where(partner[i,b] ≥ 0, out.flat[partner[i,b]·B + rev[i,b]], 0)
g[i,b]   = mul_q8(in[i,b], q8(share_efficiency = 0.9) = 230);   heat = in − g        → HEAT_SHARE
E_i     ← E_i − Σ_b out[i,b] + Σ_b g[i,b];   inflow_i = Σ_b g[i,b]                   (FLOW_SHARED = Σ out)
```

`Σ_b out ≤ E_i` by the `// B` cap, so E stays ≥ 0. Because bonds are symmetric, every `out` is received exactly once.

`same` stops energy flowing to a partner the brain never sensed: S9 can break a slot and refill it with a new
partner in the same step (`first_free` reuses the lowest free slot), and a new bond can fill an empty slot whose
`share` output was meant for nothing. A bond cannot be broken and re-formed with the same partner in one step,
because targets exclude start-of-step partners (§8.3), so comparing `partner` alone is exact. The receiving side
needs no gate: it gathers the sender's already-gated `out`. A newly formed bond carries energy from the next step on.

### 10.7 Cap, death, recycling, reproduction

- **Cap (S13).** `overflow = max(E − e_max, 0)` and `E −= overflow` → `HEAT_OVERFLOW`. This guarantees `E ≤ e_max`
  before deaths, so harvest demand is never negative. That fixes the emergence proposal's fatal flaw.
- **Death (S14).** `dead = alive & (~solvent | (E == 0) | (max_age > 0 & age ≥ max_age))`.
  - `corpse = E + S` is scatter-added into `resource` at the dead agent's `fcell` → `FLOW_CORPSE`. For an
    insolvent agent, `E` is whatever bond inflow it received at S12, so the ledger is unchanged.
  - The field is then hard-capped: `R_over = max(R − R_HARD, 0)` and `R −= R_over` → `HEAT_OVERFLOW`.
  - Cause, first match wins: `STARVED` if `~solvent`; `KILLED` if `E == 0` and `lost > 0` this step; `STARVED` if
    `E == 0`; else `AGED`.
  - **Catastrophe (director only, §12.9).** With `cat_mask`, `dead |= alive & cat_mask & ~dead`, and those rows get
    the new cause `CATASTROPHE` (3). Catastrophe deaths are *excess* deaths only: an agent that any other rule
    already kills keeps that cause, so the precedence above is unchanged and `INSOLVENT ≤ DEATHS_STARVED` still
    holds. The corpse (`E + S`) takes the ordinary corpse path above, so there is no new ledger term.
  - **Rescue.** A solvent agent drained to 0 by combat at S11 survives if bond inflow at S12 lifts it above 0. This
    is intended: it is a cooperative defence that bodies can evolve. Insolvent agents cannot be rescued.
- **Reproduction (S16).** For each parent:
  - `Dp = E − birth_cost − S`. Eligibility guarantees `Dp ≥ min_energy − birth_cost − S = 28 000`.
  - `frac_q8 = floor(offspring_frac·256 + 0.5)`, which lies in [26, 230] at the default bounds and in
    [`frac_q8(frac_min)`, `frac_q8(frac_max)`] ⊆ [1, 255] for every valid config (§17).
  - `endow = (Dp · frac_q8) >> 8`.
  - The parent keeps `Dp − endow ≥ 1`, and the child gets `endow ≥ 1`.
  - Ledger: `birth_cost` → `HEAT_BIRTH`; `S` → `FLOW_STRUCTURE` (agent E → child structure); `endow` →
    `FLOW_ENDOWMENT`.
  - Mutation itself is free. Brain cost is paid continuously.

### 10.8 The ledger

`LedgerIdx` (`state.py`) has 16 entries, int64. The per-step vector is `StepOut.ledger`, and the cumulative vector
is `state.ledger_cum`.

| idx | name | kind | idx | name | kind |
|---|---|---|---|---|---|
| 0 | `INJECT_REGEN` | source | 8 | `FLOW_STOLEN` (gross taken from victims) | internal |
| 1 | `INJECT_IMMIGRATION` | source | 9 | `FLOW_ATTACK_GAIN` | internal |
| 2 | `HEAT_BASAL` | sink | 10 | `FLOW_SPILL` | internal |
| 3 | `HEAT_ACTION` | sink | 11 | `FLOW_SHARED` (gross out) | internal |
| 4 | `HEAT_SHARE` | sink | 12 | `FLOW_CORPSE` (E + S) | internal |
| 5 | `HEAT_BIRTH` | sink | 13 | `FLOW_ENDOWMENT` | internal |
| 6 | `HEAT_OVERFLOW` | sink | 14 | `FLOW_STRUCTURE` | internal |
| 7 | `FLOW_HARVEST` | internal | 15 | `RESERVED` (always 0) | — |

No director action adds a ledger term. A catastrophe death is a corpse (`FLOW_CORPSE`), an immigration request is
admitted through the ordinary immigrant lanes (`INJECT_IMMIGRATION`), and a regeneration multiplier or patch gain
only changes the integer regeneration amount (`INJECT_REGEN`).

**Per-step closure.** This is checked on device every step (`Diag.LEDGER`) and in tests on the host:

```text
TOTAL(after) − state.total == L[0] + L[1] − (L[2] + L[3] + L[4] + L[5] + L[6])
L[FLOW_STOLEN] == L[FLOW_ATTACK_GAIN] + L[FLOW_SPILL]
```

**Global closure.** This is checked on the host at every metrics interval, and gives `ledger_error`, which must be 0:

```text
TOTAL(now, recomputed in numpy int64) == total0 + cum[0] + cum[1] − Σ_{k=2..6} cum[k] == state.total
```

Any lost or created quantum anywhere in history is detected exactly. The per-step check also attributes it to a
step.

**How it is computed on device.**
- Per-agent flows are stacked into an `(N, F_A)` int32 matrix, and per-cell flows into `(FG², F_F)`.
- Count × constant terms are formed directly in int64 from int32 counts: `HEAT_BIRTH = int64(n_birth)·birth_cost`,
  `FLOW_STRUCTURE = int64(n_birth)·S`, `INJECT_IMMIGRATION = int64(n_imm)·(e_init + S)`, and the `S·n_alive` term
  of `TOTAL` is `int64(n_alive)·S` (at 256k, `n_alive·S` exceeds 2³¹).
- Each matrix is reduced with `ops.exact_sum(·, axis=0)`.
- The reduction uses 3-limb (11-bit) int32 sums combined in int64 (F6). It is exact for non-negative values when the
  axis length is ≤ 2²⁰, so there is no int64 reduction and no int64 scatter on Metal.

### 10.9 Integer headroom

These bounds are validated by `config.validate` or monitored by diag bits.

| quantity | bound | how it is guaranteed |
|---|---|---|
| `E` | `≤ e_max < 2²⁰` | cap step (S13); validate |
| `R` | `≤ R_HARD = 2²⁴ − 1` (exact in f32) | hard cap after corpse and spill deposits |
| per-agent per-step flows | `< 2²¹` (corpse ≤ e_max + S) | validate `S < 2²⁰` |
| `exact_sum` inputs | `0 ≤ x < 2³¹`, length ≤ 2²⁰ | validate `n_max ≤ 2²⁰`, `FG ≤ 1024` |
| `proportional_take` | demand < 2¹⁶; contenders per pool < 2¹⁵ | validate `harvest_max, attack_max, share_rate < 2¹⁶`; `INT_HEADROOM` if max field-cell occupancy ≥ `occ_limit` or max attackers per target ≥ 2¹⁵ |
| corpse and spill scatter into R | `R_HARD + occ·(2·e_max + S) < 2³¹` (per agent in the cell: corpse ≤ e_max + S, spill plus residue ≤ lost ≤ e_max) | `Dims.occ_limit = min(2¹⁵ − 1, (2³¹ − 1 − R_HARD) // (2·e_max + S))` = 1 060 at defaults; same `INT_HEADROOM` bit |
| Q8 products | `Dp·230 < 2²⁸`, `grant·179 < 2²⁴`, `in·230 < 2²⁴` | the bounds above |
| int64 (`uid`, totals, ledger) | — | elementwise ops only; count × constant ledger terms are cast to int64 before the multiply (§10.8) |

`INT_HEADROOM` stays an **error** bit, because past `occ_limit` an int32 scatter can genuinely wrap. It is a
precondition monitor, not an observation of emergent behaviour: with contact-attenuated thrust (§9) the measured
maximum under a converging crowd at full thrust is 8 agents per field cell (F18), two orders of magnitude below the
limit. `occ_limit` is derived from the config rather than a magic number, so it is the real int32 bound.

### 10.10 Carrying capacity

`metrics.carrying_capacity(s, cfg)` (time-averaged: seasons use `w = 1`, §13.1) estimates `K_est = Σ_c regen_rate_c / c_ref`, where
`c_ref = basal_cost + E[founder brain_cost] + C_REF_MARGIN`. `regen_rate` uses the state's patch positions, and
`E[founder brain_cost]` is the analytic expectation under the §6 init distribution
(`metrics.founder_brain_cost`, 37.6 q at the defaults). With the 64k defaults:
- Mean fertility is ≈ 0.33–0.36 over seeds (overlapping patches clip at 1), so regen is ≈ 7.0–7.5 M q/step
  (`revise_proto/r2_fert.py`, 8 seeds).
- `c_ref ≈ 100 + 37 + 50` (`C_REF_MARGIN = 50`, `layout.py`).
- So `K_est ≈ 37–40 k` agents, about 60% of `N_CAP`. `alife inspect` prints the value computed for the actual seed.

Local depletion and predation push the real figure lower. `alife inspect` prints the estimate. The defaults are a
starting point, tuned with the slow ecology test (§22.5).

---

## 11. Combat (S11, `energy.combat`)

**Who can attack whom.**
- The attacker must be alive and solvent, with `y[ATTACK] > 0.5`.
- It must have a query-selected target (§7) among its K neighbours within `combat.range = 3.0` (contact plus 1 u).
- Bond partners are excluded unless `combat.attack_bonded = true`. Targeting (§7) excludes start-of-step partners.
  `energy.combat` also drops any attack whose target is a partner after S9 (a bond formed this step), with an
  (N,B) compare `(partner == attack_target[:, None]).any(1)`. The attempt's `attack_cost` was already paid at S8.
- Attacking self is impossible, because self is never a neighbour.
- The attempt costs `attack_cost = 50 q`, charged in S8.

**Demand.** `want_i = floor(attack_max(4000) · (1 − defend_efficiency(0.75) · defend[target_i]))`, computed in f32
and cast to int32. `defend` is the target's own decoded output, already zeroed if the target is insolvent.

**Simultaneous attacks.** All attacks resolve against one snapshot, the post-harvest `E`.

```text
grant, taken = ops.proportional_take(want, attack_target, attack, E)     # pool = victims' energy (P = N)
D       = scatter_add_i32(N, attack_target, want, attack)                # total demand per victim
residue = where(D ≥ E, E − taken, 0)                                     # fully covered victims are drained
lost    = taken + residue
gain  = mul_q8(grant, q8(efficiency = 0.7) = 179)                        → FLOW_ATTACK_GAIN
spill = grant − gain → scatter_add_i32 into resource at fcell[attack_target]   → FLOW_SPILL
residue → scatter_add_i32 into resource at the victim's own fcell              → FLOW_SPILL
E ← E − lost + gain                                    (FLOW_STOLEN = Σ lost = Σ gain + Σ spill + Σ residue)
```

- Mutual attacks (i↔j) and chains resolve consistently. Each victim loses at most its snapshot energy, so `E ≥ 0`.
- **Draining.** A victim whose attackers together demand at least all it holds (`D ≥ E`) loses everything and
  ends S11 at `E = 0`. `proportional_take` alone would leave it the few quanta its −1 guard keeps in a contested
  pool (§10.4), so it would survive S14 and die insolvent one step later as `STARVED`, which made `KILLED`
  practically unreachable. The `residue` (≤ contenders + 2D/2¹⁵ + 1 quanta) spills into the victim's cell.
- Gains may exceed `e_max`. S13 caps them, with the excess going to heat.
- **Theft efficiency.** 70% reaches the attacker and 30% spills into the field at the victim's cell. Predation is
  therefore lossy (it pays only against energy-rich prey) and messy (it feeds scavengers).
- **Feedback.** `lost` becomes `last_loss` (the `IN_HURT` input). A victim with `E == 0` at S14 dies with cause
  `KILLED`; bond inflow at S12 can rescue a drained victim (§10.7).
- **Counters** (assembled in `step.finish`, §16.3, over the attackers that survive the S11 gate):
  - `ATTACKS` is the number of attackers; `ATTACK_HITS` counts `grant > 0`.
  - `KIN_ATTACK_SUM = Σ round(256·desc[i, attack_k_i, NBR_KIN])` and
    `KIN_ATTACK_BASE = Σ round(256·x[i, IN_KIN_MEAN])`. These feed `kin_attack_bias` (§19).
- **Kin recognition** is entirely learned, through the `sim` component of `q_attack · φ`.

---

## 12. Reproduction and mutation

### 12.1 Trigger

`req = survivor & solvent & act.reproduce & (E ≥ min_energy (40 000)) & (age ≥ min_age (20))`. This is evaluated at
S16, after deaths and physics. Reproduction is **division**: the parent's energy is split by its heritable
`offspring_frac`, and the child optionally stays bonded.

### 12.2 Lane allocation under capacity pressure (`lifecycle.allocate_lanes`)

```python
perm   = mx.random.permutation(N, key=ks[BIRTH_PRIO]).astype(i32)   # unique random order (F4)
ppos, pvalid = ops.rank_select(req[perm], M)                       # first M requesters in that order
slot, _ = ops.rank_select(~survivor, M)                            # the M lowest-index dead rows: always exist
n_surv, n_req = survivor.sum(), req.sum()                          # int32 counts
room    = N_CAP - n_surv                                           # ≥ 0, since n_surv ≤ alive_prev ≤ N_CAP
n_birth = minimum(minimum(n_req, M), room)
n_imm   = clip(min(floor − n_surv − n_birth, imm_per_step, M − n_birth, room − n_birth), 0, M)  # 0 if disabled (static); §12.7
kind    = where(k < n_birth, BIRTH, where(k < n_birth + n_imm, IMMIGRANT, PAD))   # k = arange(M); LaneKind enum
parent  = mx.where(kind == BIRTH, perm[ppos], 0)                   # (M,); non-birth lanes use row 0 (§12.6)
```

`ops.rank_select(mask, n)` computes `c = cumsum(mask)` and `idx = searchsorted(c, arange(1, n+1), "left")`, then
returns `minimum(idx, len−1)` and `valid = arange(n) < c[-1]`.

**Properties** (property-tested):
- Birth lanes are a prefix.
- Parents are unique, alive and eligible.
- Slots are unique and dead.
- `n_alive` after births is ≤ `N_CAP`.
- A full population gives 0 births.
- Winners are uniformly random among requesters (χ² test across seeds).
- Children fill the lowest-index free rows, which keeps the occupied set compact.

**Blocked requests.** Requesters that are not served keep their energy and pay nothing (`BIRTHS_BLOCKED`).

### 12.3 Writing newborn rows: gather-based placement (no non-int32 scatter)

```python
born_lane = ops.scatter_max_i32(N, slot, arange(M), kind > 0)        # row → lane, −1 if not newborn
child_of  = ops.scatter_max_i32(N, parent, slot, kind == BIRTH)      # parent row → child row
newborn   = born_lane >= 0
arr = ops.place(arr, newborn, born_lane, lane_vals)                  # where(newborn, lane_vals[max(lane,0)], arr)
```

Every per-agent small array is written with `place`. This includes the int64 `uid`, so no int64 scatter is ever
needed. Parents are debited with `where(child_of >= 0, E − birth_cost − S − endow, E)`.

**Child row values.** Here `p` is the parent row. `pos_p` and `θ_p` are post-S15 values. The `PLACEMENT`,
`MUT_*` and `IMMIGRANT` keys are split as listed in §14. The table is the 2D one; a 3D world places children in a
cone behind the parent and immigrants in the volume, and also writes `pitch` and `buoy_cap` (§27.4).

| field | birth lane | immigrant lane |
|---|---|---|
| `alive` | True | True |
| `uid` | `next_uid + k` (k = lane index) | same |
| `parent_uid` | `uid[p]` | −1 |
| `clade` | `clade[p]` | `n_init` |
| `generation` | `generation[p] + 1` | 0 |
| `birth_step` | `t` (this step's index) | `t` |
| `energy` | `endow[p]` | `e_init` |
| `pos` | `wrap(pos_p + offspring_distance·(cos φ, sin φ))`, with `φ = θ_p + π + U(−0.5, 0.5)` | `U[0, L)²` |
| `heading` | `wrap_angle(φ)` (facing away from the parent) | `U[−π, π)` |
| `vel`, `h`, `msg`, `last_loss`, `last_harvest`, `last_inflow` | 0 | 0 |
| `partner`, `rev` | −1, then the child bond in slot 0 (§8.4) | −1 |
| `tag` | `clip(tag_p + tag_sigma·n, −1, 1)` | `U[−1, 1)ᵀ` |
| `mut_sigma` | §12.4 | `sigma_init` |
| `mut_rate` | §12.4 | `rate_init` |
| `offspring_frac` | §12.4 | `frac_init` |
| `brain_cost` | `l1_cost` of the child's stored rows | `l1_cost` of the fresh rows |
| `W1`, `W2` | `mutate_rows(W1[p], W2[p], …)`, written at the next S0 | `init_weights` rows, written at the next S0 |

After the step, `next_uid += n_birth + n_imm`.

### 12.4 Heritable genes and weight mutation

The strategy genes mutate first, and the mutated values are then used for the weights. Keys:
`ks[MUT_GENES]` is split into 4 (σ, p, tag, frac; 2D, 3D: 5 with the buoyancy cap, §27.4), and `ks[MUT_W1]` /
`ks[MUT_W2]` are used as in §6.

```text
σ_c   = clip(σ_p · exp(tau_sigma(0.15) · n₁), sigma_min(0.002), sigma_max(0.3))            σ_init = 0.03
p_c   = clip(sigmoid(logit(p_p) + tau_rate(0.15) · n₂), rate_min(0.001), rate_max(0.5))     p_init = 0.05
tag_c = clip(tag_p + tag_sigma(0.02) · n₃, −1, 1)                                           (fixed rate: neutral clock)
f_c   = clip(sigmoid(logit(f_p) + tau_frac(0.1) · n₄), frac_min(0.1), frac_max(0.9))         f_init = 0.5
W_c   = clip(W_p + 1[u < p_c] · σ_c · ε, ±4)                                                 (brain.mutate_rows)
```

**Why tags drift at a fixed rate.** Tag distance then tracks genealogical distance:
`E‖Δtag‖² ≈ T·g·tag_sigma² = 0.0016·g` for g generations apart, so `sim ≈ exp(−0.0064·g)`, which is ≈ 0.5 at
g ≈ 108. That makes `sim` an honest kin cue that selection can learn to use or ignore. Heritable tag rates, which
would allow mimicry, are deferred.

### 12.5 Child bonds

See §8.4. A child bond requires `act.child_bond` of the parent and a free parent slot after S9 and S14.

### 12.6 Deferred heavy writes

The child weight rows are **not** written into `W1`/`W2` at S16. Instead:
- `pend_slot ← slot`, `pend_W1 ← lane W1 rows`, `pend_W2 ← lane W2 rows`, for all M lanes.
- S0 of the next step writes them with `ops.set_rows` (copy-wrapper setitem, F2).

**Why** (measured, F1): MLX updates a scatter target in place only if nothing else in the evaluated graph consumes
the old buffer. With the flush at S0, the old `W1`/`W2` are consumed only by the flush. Everything downstream (the
brain matmul, the parent gathers) reads the post-flush array.

**Padding lanes.** These write mutated copies of row 0 into **distinct dead rows**. This is harmless: dead-row
weights are stale by definition, and every lane index is unique, so the write is deterministic.

**Semantics** are unchanged, because a newborn's brain first runs in the next step's S6.

**Consequences.**
1. Checkpoints contain `pend_*`, so resume is exact.
2. Between steps (for `step ≥ 1`), rows with `birth_step == step − 1` have stale `W1`/`W2`. Observers that read
   weights (genome metrics, `inspect --agent`) exclude them or read their rows from `pend_*`.
3. The pending invariant: `pend_slot` entries are distinct, and for `step ≥ 1` every alive row with
   `birth_step == step − 1` appears in `pend_slot`. At step 0 the founders have `birth_step = −1 = step − 1`, but
   `init_state` writes their weights directly, so the second clause does not apply.

**Initial state.** `pend_slot = arange(N − M, N)`, which are dead rows because `n_init ≤ N_CAP`, and `pend_W* = 0`.

### 12.7 Immigration (optional, off by default)

- Immigration is enabled iff `population.immigration_floor > 0` **or** `population.immigration_every > 0`
  (`config.immigration_on`). The lane and founder-drawing gates in `lifecycle` use `config.immigration_possible(cfg)
  = immigration_on(cfg.population) or cfg.director.enabled`, because the director can request immigrants at any
  time. This is a static Python branch, so with the default config the code path does not exist in the compiled
  graph. The runner's extinction stop (§18) uses `immigration_on` plus a host check of `env_imm_request`.
- **Floor rule.** When `n_surv + n_birth < floor`, the demand is `min(floor − n_surv − n_birth, immigrants_per_step (64))`.
- **Periodic rule** (`immigration_every = P > 0`, `immigration_count = c`). On every step with index `t` such
  that `(t + 1) % P == 0` (`t = state.step` during the step) the demand is `c`; on other steps it is 0. The test
  is array arithmetic on `state.step`, so it stays inside the compiled step with no host sync.
- The two rules combine by `max`: `n_imm = clip(min(max(floor demand, periodic demand), M − n_birth,
  room − n_birth), 0, M)` with `room = max(N_CAP − n_surv, 0)`. Births take lanes first, so a due step with
  all `M` lanes used by births admits no immigrants.
- **Director request** (`env_imm_request = r`; `allocate_lanes(..., imm_request=r)`). With the director on, the
  demand is `max(floor demand, periodic demand, r)` before the same caps (`M − n_birth`, `room − n_birth`). The step
  then writes back `r' = max(r − n_imm, 0)` (`step.consume_env`), so whatever the lanes or the capacity could not
  admit stays requested and arrives over the following steps: a large pulse is spread over `⌈r / M⌉` steps or more.
  If the other rules already admit at least `r` immigrants this step, the request is fully consumed by them. The
  request is capped by capacity like any immigrant, and a request that capacity keeps refusing remains pending.
- It prevents extinction from ending long experiments, and it can also inject fresh genetic variation on a
  schedule. Both rules use the same lanes, the `IMMIGRANT` RNG stream, and are ledger-accounted
  (`INJECT_IMMIGRATION`, `IMMIGRANTS` counter).
- Immigrants share clade `n_init`. The extinction stop (§18) is off whenever either rule is enabled.

### 12.8 Lineage

- **State:** `uid` (int64, assigned in birth order, never reused), `parent_uid`, `clade`, `generation` and
  `birth_step`.
- **Events:** each step's `BirthRecords` (M lanes) and `DeathRecords` (DL = 2M lanes) are flushed to
  `events/*.npz` at metrics intervals (§18). Deaths beyond DL in one step are counted in `DEATHS_UNLOGGED`.
- Together these allow the full phylogeny to be reconstructed offline.

### 12.9 Catastrophe (director, off unless `director.enabled`)

A catastrophe is a one-step event: set `env_cat_center`, `env_cat_radius` and `env_cat_kill_p` between steps.
In the next step `lifecycle.catastrophe_mask(pos, alive, env, ks[Stream.CATASTROPHE], cfg)` marks an alive agent
iff its start-of-step position lies within `env_cat_radius` of `env_cat_center` by the torus minimum image
(`‖min_image(pos − center)‖² ≤ radius²`, and `radius > 0`) **and** its uniform draw `u_i < env_cat_kill_p`
(`u ~ U[0,1)^N` from the `CATASTROPHE` stream; the draw is made every step the director is enabled, so stream use
never depends on the levers). `deaths(..., cat_mask=)` kills the marked agents that would otherwise survive (§10.7)
with cause `CATASTROPHE`, counted in `DEATHS_CATASTROPHE`. After the step `env_cat_kill_p` is reset to 0, so it
cannot repeat; `env_cat_center` and `env_cat_radius` stay as set (they are inert at `kill_p = 0`, and the last
geometry stays inspectable). In 3D the disc is a vertical column: only x and y count (§27.4).

---

## 13. Fields: the cellular-automaton substrate

All fields are periodic, indexed `[y, x]`, with flat index `cy·FG + cx` (2D; a 3D lattice is `[z, y, x]`, bounded
in z, with the z-major flat index `(cz·FG + cy)·FG + cx`, §27.5).
- `cell_index(pos)` returns `(floor(pos.y / field_cell) % FG)·FG + (floor(pos.x / field_cell) % FG)`.

### 13.1 Fertility patches (non-stationary environment)

```text
gx[p, x] = exp(−min_image(xc − px_p)² / (2σ_p²)),   gy[p, y] likewise    (xc = (x + 0.5)·field_cell; σ_p = patch_sigma)
fert     = clip(fertility_base + Σ_p gy_p ⊗ gx_p, 0, 1)                (FG,FG); separable torus Gaussians, elementwise (no matmul, F20)
regen_rate = floor(regen_max · fert).astype(int32)                       regen_max = 320 q/cell/step
patch_pos ← wrap(patch_pos + patch_vel)                                  |patch_vel| = patch_drift = 0.005 u/step
```

In 3D, `fert` and `regen_rate` are (FZ, FG, FG): this `fert` times a vertical profile by layer, whose floor entry is
exactly 1 (§27.5).

**Seasons (optional, off by default).** With `fields.season_period = P > 0` and `fields.season_amplitude = A > 0`
each patch's Gaussian is weighted by `w_p(t) = 1 + A·sin(2π·t/P + φ_p)`, so
`fert = clip(fertility_base + Σ_p w_p · gy_p ⊗ gx_p, 0, 1)` with `t` = the step index during the step
(`fertility(patch_pos, cfg, step)`, `regen_rate(…, step)`, `update_resource(…, step)`; `step` defaults to None).
`φ_p = 0` for `season_mode = "global"` (all patches wax and wane together) and `2π·p/PP` for `"per_patch"`
(evenly staggered phases). The phase is a pure function of `t`, so there is no new state, and `t mod P` is
taken in integers so it stays exact on long runs. `A ∈ [0, 1]` keeps `w_p ≥ 0`; the mean of `w_p` over a period is 1.
- Seasons are a static branch: with `P = 0` or `A = 0` no multiplication is traced and `fertility` is
  bit-identical to the unweighted sum (the golden digest is unchanged).
- **Director env (static `director.enabled`; the functions take an optional `env`).** The functions take
  `env: Env | None = None`; the step passes `s.env` only when the director is enabled.
  - *Seasons.* `season_weights(step, cfg, env)` uses `env.season_amp` and `env.season_period` in place of the
    config values (`season_mode` stays a config constant). `env.season_period = 0` means off: every weight is 1,
    and the integer modulo is taken by `max(period, 1)` so it is never by zero. With the director on, seasonal
    weighting is always traced (a `1 + 0·sin(…)` weight is exactly 1).
  - *Patch gain.* The Gaussian of patch `p` is weighted by `env.patch_gain[p] · w_p(t)`; gain 0 removes the patch
    from the fertility sum (and a gain above 1 makes a richer patch). Gain applies when a `step` is given, so the
    initial field and the carrying-capacity estimate (`step = None`) are unaffected.
  - *Regeneration.* `regen_rate = floor_i32((regen_max · fertility) · env.regen_mult)`. The product
    `regen_max · fertility` is evaluated first, so `regen_mult = 1.0` is exact.
  - *Drift.* `patch_pos ← wrap(patch_pos + patch_vel · env.drift_mult)`; the velocity itself has no RNG.
  - With inert env values (`mult = 1`, `gain = 1`, config season values) each of these is the identity, so a
    director-enabled run is bit-identical to a director-off run.
- Regeneration stays integer and ledger-accounted (`INJECT_REGEN = Σ inc`), so the ledger closes exactly.
- `step = None` means the time average `w = 1`. It is used for the initial resource field (`init_state`) and for
  the carrying-capacity estimate (§10.10), which is therefore the season-averaged one.

- At init, `patch_pos ~ U[0, L)²` (`InitStream.PATCH_POS`) and `patch_vel = patch_drift·(cos ψ, sin ψ)` with
  `ψ ~ U[0, 2π)` (`InitStream.PATCH_VEL`).
- A patch crosses the 64k world in ≈ 2·10⁵ steps. The environment therefore never reaches a final equilibrium,
  but changes slowly relative to generation time.
- `patch_drift = 0` gives a static environment.
- Cost per step: `PP·FG·2` exponentials and a `(FG,PP)@(PP,FG)` matmul, which is negligible.

### 13.2 Resource (int32 quanta)

- **Init:** `R = floor(resource_cap(8000) · fert₀)`.
- **Per step (S17):** regenerate (§10.2), then diffuse with the exact integer flux stencil (`ops.int_diffuse`):
  ```text
  q  = (R · resource_diffusion_q8(4)) >> 8                       per direction (≈ 1.6%)
  R ← R − 4q + roll(q, +1, 0) + roll(q, −1, 0) + roll(q, +1, 1) + roll(q, −1, 1)
  ```
- **Properties:**
  - Total resource is conserved exactly.
  - `R` stays ≥ 0, because `4q ≤ R` (validate requires `resource_diffusion_q8 ≤ 64`).
  - `R` stays ≤ `R_HARD`: an output cell is at most `R_i − 4q_i + 4·q(R_HARD) ≤ R_HARD + (4·d_q8 − 4)/256`,
    which is `< R_HARD + 1` because `R_HARD ≡ −1 (mod 256)`. (Floor rounding means diffusion is not a strict
    average: a cell may exceed the local maximum by a few quanta, but never `R_HARD`.)
  - The product `R·64 < 2³⁰`.
- Setting `resource_diffusion_q8 = 0` disables diffusion.
- Corpses (S14) and combat spill (S11) are int scatter-adds into `R`.

### 13.3 Chemical / signal channels (NCH = 3, float32)

**Deposit.** Each solvent agent emits `floor(emit_c · 256)` integer quanta per channel. These are accumulated with
`scatter_add_i32` at flat index `c·FG² + fcell`, which is deterministic. Then
`chem += dep_q · chem_deposit_scale (1/256)`, so a full-strength emitter adds 1.0 per step.

**Update.** A 5-point explicit stencil via `mx.roll`, clamped to `[0, chem_max = 16]`:

```text
chem_c ← (1 − λ_c) · (chem_c + D_c · (roll₄(chem_c) − 4·chem_c))
```

| channel | D | λ (decay) | length scale √(D/λ) | intended use |
|---|---|---|---|---|
| c0 | 0.20 | 0.10 | 1.4 cells (5.7 u) | contact and body-local |
| c1 | 0.10 | 0.03 | 1.8 cells (7.3 u) | neighbourhood |
| c2 | 0.24 | 0.005 | 6.9 cells (28 u) | long-range quorum and trails |

- `validate` requires `0 ≤ D_c ≤ 0.25` (the explicit-stencil stability limit) and `0 ≤ λ_c < 1`.
- Chemicals are float, but they are not energy. Their deposit is integer and the stencil is elementwise, so they
  are deterministic.
- Emission costs energy (`emit_cost`), so signalling is honest-cost.

### 13.4 Agents read and write

| field | agents write | agents read |
|---|---|---|
| resource | harvest (−), corpses (+), spill (+) | value and gradient at their cell |
| chem | emission | value and gradient per channel |

---

## 14. RNG and determinism

### 14.1 Key threading (`rng.py`)

- `state.key` (uint32[2]) is the only simulation RNG state.
- Every step: `k = split(state.key, 2)`, `key_next = k[0]`, `ks = split(k[1], NUM_STREAMS)`.
- `NUM_STREAMS = 16` is frozen, because split is not prefix-stable (F3). `RNG_LAYOUT_VERSION = 2` is stored in every
  checkpoint.
- New consumers take a reserved index. Every layout change bumps the version. A change that moves no existing
  stream key (taking a reserved index, as `CATASTROPHE` did for version 2) keeps the older versions in
  `rng.RNG_COMPATIBLE_VERSIONS = {1, 2}`, so their checkpoints load and resume. Loading a checkpoint whose version is
  not in that set is refused with exit code 4. A change that did move a key must bump the version without adding
  the old one to the set.

| `Stream` | idx | consumer | draws |
|---|---|---|---|
| `NOISE` | 0 | `IN_NOISE` | `uniform(−1, 1, (N,))` |
| `BOND_PRIO` | 1 | handshake priorities | `permutation(N)` |
| `BIRTH_PRIO` | 2 | parent order | `permutation(N)` |
| `PLACEMENT` | 3 | child angle jitter | `uniform(−0.5, 0.5, (M,))` |
| `MUT_W1` | 4 | `W1` mutation | `split(·, 2)` → `uniform (M,W1_ROWS,H)`, `normal (M,W1_ROWS,H)` |
| `MUT_W2` | 5 | `W2` mutation | same, with shape `(M,W2_ROWS,D_OUT)` |
| `MUT_GENES` | 6 | σ, p, tag, frac | `split(·, 4)` → `normal (M,)`, `normal (M,)`, `normal (M,T)`, `normal (M,)` |
| `IMMIGRANT` | 7 | immigrant pos, heading, tag, W1, W2 | `split(·, 5)`; drawn only if immigration is possible (`immigration_possible`) |
| `CATASTROPHE` | 8 | director catastrophe kill draw | `uniform(0, 1, (N,))`; drawn every step only if `director.enabled` |
| — | 9–15 | reserved | — |

**Init** (`rng.init_keys(seed)`): `split(key(seed), 16)`, with these `InitStream` slots:
- 0 `POS` (founder positions, `U[0, L)²`)
- 1 `HEADING`
- 2 `W1`
- 3 `W2`
- 4 `TAG`
- 5 `PATCH_POS`
- 6 `PATCH_VEL`
- 7 `STATE_KEY` (becomes `state.key`)
- 8–15 reserved.

A 3D world draws from these same keys and streams with 3D-only shapes and split counts (`InitStream.POS` and
`HEADING`, `PLACEMENT`, `MUT_GENES` and the `IMMIGRANT` split's position and heading keys): no new stream and no
`RNG_LAYOUT_VERSION` bump, and every 2D draw is unchanged (§27.4).

**Observers** use `rng.observer_key(seed, step) = mx.random.key((seed * 1_000_003 + step) % 2**63)`. Examples are
genome-diversity sampling and viewer picks. Observers never touch `state.key`.

### 14.2 Determinism contract

Bitwise-identical trajectories are guaranteed when all of these match:
- seed and resolved config;
- code version, MLX version and backend (CPU or Metal);
- `run.compile` flag, and `run.runtime_checks` (it decides whether warning bits are OR-ed into `diag`, which is a
  State field and part of the digest).

`run.seed`, `run.compile` and `run.runtime_checks` are therefore fixed for the lifetime of a run directory: resume
rejects overrides of them (§18). With a director, the decision log (`director.jsonl`, §26) is one more input: the same
log replays bitwise (§26.8). The guarantee covers checkpoint/resume, and holds **whatever** the metrics, render, checkpoint or viewer cadence
(observer purity).

**How each mechanism is made deterministic.**
1. **No float scatter-add anywhere in the simulation.** All contested writes are int32 add or max, which are
   associative, so GPU atomic order cannot matter. Rendering uses uint32 scatter-max.
2. **Setitem writes use unique indices only.** Weight rows go through the capacity reserve (§2.1).
3. **Unique priorities come from permutations.** No random floats are used as sort keys.
4. **Sorts, partitions, reductions and matmuls** have no atomics in MLX, so tie orders are deterministic per backend.
5. **No global RNG.** `mx.random.seed` is never called, and every `mx.random.*` call passes `key=`. Both rules are
   lint-tested.

**Not promised:**
- CPU ≡ Metal;
- compiled ≡ eager (fusion may change float rounding);
- equality across MLX versions.

Every invariant, including exact ledger closure, holds on all backends regardless.

**Acceptance on the Mac.** `alife bench --determinism` runs 2 × 200 steps of the 64k preset on the GPU and compares
SHA-256 state digests.

**Digests.** `state_digest(s)` hashes every `State` field, `env_*` included. `legacy_state_digest(s)` hashes exactly
the 38 frozen v0 fields (`state.V0_FIELDS`) in v0 order with the same per-field encoding, so on a director-off
state it equals what `state_digest` returned before the `env_*` fields existed. Bit-identity checks against
pre-director trajectories (the golden pin, `alife bench` baselines) use `legacy_state_digest`; everything else uses
the stricter `state_digest`. With the director enabled and inert env values, `legacy_state_digest` equals the
director-off one at every step (§3).

**Golden test.** `tests/test_determinism.py::test_golden` pins the `tiny_cfg` (tiny preset, f32 weights)
`legacy_state_digest` after 50 steps (the pin was not changed by the env fields). It runs only
when `mlx.__version__` equals the version recorded next to the golden value, so an MLX upgrade becomes a deliberate
change.

---

## 15. Performance budget (M5 Pro), compile, syncs and memory

### 15.1 Assumptions (to be replaced by `alife bench` output on day one)

- **Memory bandwidth.** The M4 Pro has 273 GB/s peak, and the M5 Pro is assumed to be ≥ that. Effective rates are
  assumed to be **120–220 GB/s**: the low end for gather-heavy kernels, the high end for streaming GEMV and
  elementwise kernels. Batched tiny-GEMV efficiency on Metal is unmeasured, and bench A/B-tests it.
- **Dispatch.** 5–8 µs per kernel. After `mx.compile` fusion of elementwise chains, there are ≈ 250–320 kernels
  per step: ≈ 150 gather, scatter, reduce, sort, matmul and RNG nodes plus ≈ 100–170 fused elementwise clusters.
  This is extrapolated from the throughput prototype's 1 363-node graph, of which ≈ 700 nodes are free views.
- **Host build** of the compiled call: 0.5 ms measured on CPU. **Sync bubble:** ≈ 0.1–0.3 ms.
- **Sorts:**
  - one N-element argsort for the hash grid;
  - two N-element permutations (bond priority and birth priority), each a sort;
  - ArgPartition, which lowers to a full sort of 144-element rows on Metal (F16).

### 15.2 Memory traffic per step, 64k preset

Settings: N = 65 536, K = 8, C = 16, Q = 144, B = 4, M = 1 024, H = 32, fp16 weights, FG = 256.

| stage | dominant traffic | MB/step |
|---|---|---|
| S0 flush | `M × 8.8 KB` read plus in-place write | 18 |
| S2 neighbours | key/argsort/searchsorted ≈ 10; `slot` (N,Q) i32 W+R 75; `spos[slot]` gather 75 R(L2) + 75 W; `d2` 75 R + 38 W; partition 38 R + 38 W; K-sort and ids ≈ 10 | ≈ 430 |
| S3–S5 geometry, fields, sensing | (N,K) gathers of tag/E/msg, (N,K,B) bonded compare, `desc` (N,K,7) write, field gathers, `x` (N,69) f32 | ≈ 70 |
| S6 brain | `W1` 424 + `W2` 151 (each read once) + `z`, `h`, `y` ≈ 25 | ≈ 600 |
| S7 decode and targets | `desc` reads, `can_bond[nbr]` gathers, scores | ≈ 40 |
| S8–S14 costs, bonds, energy, deaths | (N,B) gathers, int scatters on N and FG², a permutation, death records | ≈ 60 |
| S15 physics | partner positions, `nbr.off`, elementwise | ≈ 15 |
| S16 reproduction | permutation; rank_selects; parent rows 9; RNG uniform+normal over (M,P) f32 36 W + 36 R; mutated rows 9; `place` gathers of ≈ 22 small arrays ≈ 25 | ≈ 130 |
| S17 fields | fertility (separable), regen, int diffusion, chem deposit and stencil on 3×256² | ≈ 15 |
| S18 bookkeeping, device_check, exact sums | (N,10) and (FG²,2) limb sums, bond symmetry gather, finiteness reductions | ≈ 30 |
| **total** | | **≈ 1.4 GB** |

### 15.3 Time model

| | 64k (`64k` preset) | 256k (`256k` preset: L 2048, M 4096, FG 512) |
|---|---|---|
| bandwidth (1.4 GB or 5.6 GB at 120–220 GB/s) | 6.4–11.7 ms | 25–47 ms |
| sorts: argsort + 2 permutations + per-row partition sort | 2–4 ms | 6–12 ms |
| dispatch (≈ 300 kernels) | 1.5–2.4 ms | 1.5–2.4 ms |
| host build + sync bubble (one `mx.eval` per step) | 0.6–1.2 ms | 0.6–1.2 ms |
| **per step** | **11–19 ms → 53–90 steps/s** | **33–63 ms → 16–30 steps/s** |

**Verdict.**
- **≥ 30 steps/s at 64k is met with a 1.8–3× margin**, even at the pessimistic end of every assumption.
- **256k runs at roughly 16–30 steps/s with defaults.** Levers, in order:
  1. The `fast` overlay (`C = 12`, `K = 6`, `H = 24`): −25% neighbour traffic and −29% brain traffic, which gives
     ≈ 20–35 steps/s at 256k.
  2. `brain.impl` and `weight_dtype` chosen by bench.
  3. Deferred: a fallback-guarded fused Metal kNN kernel (≈ −1.6 GB/step at 256k), periodic re-sorting of agent rows
     by cell for locality, and dead-row compaction for the brain.

### 15.4 `mx.compile` and sync discipline

- **One compiled function, `step`, per config.** Config scalars, shapes and dtypes are closed over as Python
  constants, and `key`/`step` are array inputs, so there is no retracing. Compile time is reported by bench and is
  expected to be seconds.
- **Zero host syncs inside a step.** There is no `.item()`, no `np.asarray`, no Python branching on array values,
  and no variable-length op.
- **Exactly one blocking `mx.eval(state, out)` per step** in the runner. `mx.async_eval` is **not offered**,
  because it defeats weight donation (F1). That would cost +1 full weight copy per step and ≈ 5 queued copies of
  peak memory.
- **Donation rule.** No Python reference to a previous `State` may survive a step call. `runner`, `viewer`, `bench`
  and metrics hold only the current state and drop references before calling `step` again. Tests that keep `prev`
  states are allowed, because they do not measure performance.
- **Host reads** happen only every `metrics_every` steps:
  - stacked counters and ledgers;
  - event records;
  - `diag`;
  - `energy`/`resource` for the host ledger check;
  - snapshot metrics.

  Unified memory makes `np.asarray` of small arrays nearly free.
- **Donation health** (`alife bench`): after warm-up, bench measures `mx.get_peak_memory()` − the resident
  baseline over 10 steps (`excess`), then over one calibration step that deliberately keeps a reference to its
  input state (`excess_copy`, the known-copy case of F1). It warns unless `excess_copy − excess` exceeds
  0.25 × (bytes of `W1` + `W2`), which means the weights are being copied. The step's own transients cancel in
  the difference; an absolute bound on `excess` cannot work, because those transients are ≈ 0.3× the f32 and
  ≈ 0.64× the fp16 weight bytes (measured on CPU for tiny and dev), while a retained old state can even lower
  `excess` (it is part of the baseline). Measured: healthy steps save 0.64–1.05× the weight bytes; a retained
  state saves 0.
  - **Fallback** if Metal behaves differently from CPU: run S0 as its own compiled call, evaluated before the main
    step. That costs one extra sync per step.

### 15.5 Memory footprint

| | 64k | 256k |
|---|---|---|
| state (weights + small arrays + fields) | 0.60 GB | 2.4 GB |
| pending buffer | 9 MB | 36 MB |
| transients (neighbours ≈ 0.25 GB / 1.0 GB; mutation ≈ 60 MB / 0.24 GB) | ≈ 0.35 GB | ≈ 1.3 GB |
| **total** | **< 1.1 GB** | **≈ 3.8 GB** |
| checkpoint size | 0.6 GB | 2.4 GB |

One million agents would need ≈ 15 GB. That fits in 48 GB, at an estimated 4–8 steps/s.

---

## 16. Package layout and public signatures

```text
artificial-life/
  pyproject.toml                 uv; mlx (darwin) | mlx[cpu] (linux); extras: viewer, video, analysis; dev group
  README.md
  .github/workflows/ci.yml
  docs/DESIGN.md                 this document
  src/alife/
    __init__.py                  __version__; re-exports Config, State, load_config, init_state, make_step
    presets/{tiny,dev,64k,256k,fast}.toml   package data
    config.py                    Config dataclasses, Dims, presets, load/save/override/validate/hash
    layout.py                    structural constants, IN_*/OUT_*/NBR_*/BOND_* indices, names, GENOME_LAYOUT_VERSION; Layout per dims
    state.py                     State, LedgerIdx, CounterIdx, Diag, DeathCause, init_state, totals, (de)serialisation, to_numpy
    rng.py                       Stream, InitStream, NUM_STREAMS, RNG_LAYOUT_VERSION, RNG_COMPATIBLE_VERSIONS, step_keys, init_keys, observer_key
    ops.py                       the only home of scatters/setitem; contention, limb sums, placement, torus math
    spatial.py                   Neighbors, find_neighbors
    fields.py                    patches, fertility, sampling, chem deposit/update, resource update
    sensing.py                   neighbor_descriptors, build_inputs
    brain.py                     init_weights, forward, mutate_rows, l1_cost
    actions.py                   Actions, decode, mask
    bonds.py                     BondGeom, geometry, break/form, child bonds, death drop, springs, symmetry check
    physics.py                   forces, integrate
    energy.py                    pay_costs, harvest, combat, share, cap_energy
    lifecycle.py                 flush_pending, deaths, allocate_lanes, reproduce, records
    step.py                      StepOut, step, make_step
    invariants.py                device_check (in-graph), full_check (host, exhaustive)
    metrics.py                   components, snapshot_metrics, interval_row, carrying_capacity
    runio.py                     RunDir, checkpoints, metrics/events writers, truncation, diag dumps
    runner.py                    run loop, cadences, SIGINT handling, exit codes
    render.py                    Viewport, render_frame, write_png, VideoWriter
    viewer.py                    optional pygame-ce live viewer (lazy import)
    bench.py                     throughput, stages, donation health, determinism, brain variants
    cli.py                       argparse entry point `alife`
    analysis/                    offline, host-only numpy analysis of finished runs (never imported by the step;
                                 not under the §22.6 lint scope); phylo, activity, report, plots, mi, resim, signals: §19.3
  tests/                         §22
```

### 16.1 Conventions (normative for every module)

- **Purity.** All simulation functions are pure: arrays in, arrays out. They never call `mx.eval` and never read
  array values on the host. The only Python control flow is on config constants.
- **Argument and return types.** Every array argument is an `mx.array` and `cfg` is a `Config`. Dims come from
  `config.dims(cfg)`, which is cached. Every function returns new arrays or NamedTuples and never mutates its
  inputs.
- **dtypes.** Indices and counts are `int32`. Energies and resources are `int32`. Geometry and activations are
  `float32`. `uid` and the ledger are `int64`. Weights are `wdtype`. `.astype(mx.int32)` follows every
  `argsort`, `argmax`, `searchsorted` and `permutation`.
- **Scatters and setitem.** `.at[...]` and subscript assignment appear only in `ops.py`, plus uint32 compositing in
  `render.py`. `ops` helpers assert integer dtypes. This is lint-tested (§22.6).
- **Host reads.** Outside tests, every `mx.array → numpy` conversion (digests, `full_check`, metrics, `inspect`,
  `dump_diag` readers) goes through `state.to_numpy`, which widens bf16 to f32 on device first (F17).
- **Invalid indices.** `−1` means "none" for bonds and targets. Gathers use `mx.maximum(idx, 0)` and mask the
  result. Neighbour padding is the agent's own index with `valid = False`.

### 16.2 Signatures

```python
# config.py
class ConfigError(ValueError): ...
@dataclass(frozen=True) class WorldConfig; PopulationConfig; SensingConfig; BrainConfig; PhysicsConfig
@dataclass(frozen=True) class BondConfig; EnergyConfig; CombatConfig; ReproConfig; MutationConfig
@dataclass(frozen=True) class FieldConfig; DirectorConfig; RunConfig    # §17
@dataclass(frozen=True) class Config: world; population; sensing; brain; physics; bonds; energy; combat;
                                      repro; mutation; fields; director; run
@dataclass(frozen=True) class Dims:
    n: int; m: int; n_cap: int; dl: int; k: int; c: int; q: int; g_hash: int; cell_hash: float; fg: int
    h: int; w1_rows: int; w2_rows: int; params: int; pp: int; wdtype: mx.Dtype
    occ_limit: int                                                        # §10.9: min(2^15−1, (2^31−1−R_HARD)//(2·e_max+S))
    nd: int; d3: bool; lay: layout.Layout; fz: int; n_cells: int; field_shape: tuple[int, ...]
    g_hash_z: int; cell_hash_z: float                                     # world dimensions (§27.1); q = 3^nd·C
    z_lo: float; z_hi: float; z_ground: float                             # float32 z bounds (§27.2); 0.0 in 2D
def dims(cfg: Config) -> Dims                                             # functools.cache
POST_V01_KEYS; UNHASHED_KEYS; THREE_D_KEYS                                # §17; the 3D keys (§27.1)
DIMS_SUPPORTED: tuple[int, ...] = (2,)                                    # world.dims values validate accepts (§27.1)
def is_3d(cfg: Config) -> bool                                            # world.dims == 3 (static)
def require_2d(cfg: Config, what: str) -> None                            # ConfigError for a 3D config in a 2D-only tool
PRESETS: tuple[str, ...] = ("tiny", "dev", "64k", "256k")                 # plus overlay "fast"
def load_config(path: str | Path | None = None, *, preset: str | None = None,
                overlays: Sequence[str] = (), overrides: Sequence[str] = ()) -> Config   # validates
def apply_overrides(cfg: Config, overrides: Sequence[str]) -> Config      # "section.key=value", typed parse
def config_to_dict(cfg: Config) -> dict[str, dict[str, Any]]
def config_from_dict(d: Mapping[str, Any]) -> Config                      # strict: unknown keys → ConfigError
def save_config(cfg: Config, path: Path) -> None                          # canonical JSON (sorted keys, indent 2)
def validate(cfg: Config) -> None                                         # raises ConfigError listing ALL problems
def config_hash(cfg: Config) -> str                                       # sha256 hex of canonical JSON minus "run";
                                                                          # POST_V01_KEYS at defaults, and sections they empty, omitted
def immigration_on(pop: PopulationConfig) -> bool                         # floor or periodic rule
def immigration_possible(cfg: Config) -> bool                             # immigration_on or director.enabled (static gate)
def stability_number(cfg: Config) -> float                                # 2(B k_s + K k_r)/drag

# layout.py  (module constants: the 2D layout; Layout per dims, §27.1)
B = 4; NS = 3; NF = 7; F = 6; MSG = 2; NCH = 3; T = 4; D_IN = 69; D_OUT = 35; GENOME_LAYOUT_VERSION = 1
GENOME_LAYOUT_VERSION_3D = 2
@dataclass(frozen=True) class Layout: nd; D_IN; D_OUT; NF; F; INPUT_BLOCKS; OUTPUT_BLOCKS; INPUT_NAMES; OUTPUT_NAMES
                                      NBR_FEATURE_NAMES; NBR_DZ, IN_VEL_UP, ..., OUT_Q_ATTACK_DZ (3D only; None in 2D)
def for_dims(nd: int) -> Layout                                    # functools.cache; 2 or 3, else ValueError
def genome_layout_version(nd: int) -> int                          # reads the module globals at call time
IN_BIAS = 0; IN_ENERGY = 1; ...; IN_NBR = 28; IN_BOND = 49        # every index in §5
NBR_PRESENT = 0; NBR_DX = 1; NBR_DY = 2; NBR_KIN = 3; NBR_RELE = 4; NBR_MSG0 = 5; NBR_BONDED = 6
BOND_PRESENT = 0; BOND_DX = 1; BOND_DY = 2; BOND_MSG0 = 3; BOND_RELE = 4; BOND_W = 5
OUT_THRUST = 0; ...; OUT_Q_BOND = 23; OUT_Q_ATTACK = 29            # every index in §7
LR_SCALE = 2.0; AGE_SCALE = 1000.0; RES_SAT = 2.0; RES_GRAD_GAIN = 4.0   # §5 input normalisation
EPS_LEN = 1e-6                                                     # §9 unit-vector guard
C_REF_MARGIN = 50                                                  # §10.10 carrying-capacity estimate
PITCH_MAX = math.radians(80.0)                                     # 3D pitch clip (§27.2)
INPUT_NAMES: tuple[str, ...]   # len D_IN;  OUTPUT_NAMES: tuple[str, ...]  # len D_OUT

# state.py
class LedgerIdx(IntEnum): ...      # §10.8 (16); HEAT_BUOYANCY = 15 is an alias of RESERVED (§27.1)
def sinks(nd: int) -> tuple[LedgerIdx, ...]                                # SINKS, plus slot 15 in 3D
def ledger_names(nd: int) -> tuple[str, ...]                               # slot 15: RESERVED (2D) | HEAT_BUOYANCY (3D)
def expected_layout(cfg: Config) -> dict[str, tuple[tuple[int, ...], mx.Dtype]]   # the one State schema (§27.1)
class CounterIdx(IntEnum): ...     # §19.1 (N_COUNTERS = 27)
class Diag(IntFlag): ...           # §22.3
class DeathCause(IntEnum): STARVED = 0; KILLED = 1; AGED = 2; CATASTROPHE = 3
R_HARD: Final = (1 << 24) - 1
class State(NamedTuple): ...       # §2.3 (47 fields: V0_FIELDS (38) + ENV_FIELDS (9)); State.env -> Env view
class Env(NamedTuple): regen_mult; season_amp; season_period; drift_mult; patch_gain; imm_request; cat_center; cat_radius; cat_kill_p
ENV_FIELDS: Final = ("env_regen_mult", ...); V0_FIELDS: Final = State._fields[:38]
class State3(NamedTuple): ...      # §27.4: State's 47 fields, then EXTRA_3D_FIELDS = ("pitch", "buoy_cap"); .env as State
AGENT_FIELDS; AGENT_FIELDS_3D = AGENT_FIELDS + EXTRA_3D_FIELDS               # per-agent fields (leading dim N)
def state_cls(cfg: Config) -> type                                         # State3 if world.dims == 3, else State
def agent_fields(cfg: Config) -> tuple[str, ...]                           # AGENT_FIELDS_3D | AGENT_FIELDS
def default_env(cfg: Config) -> dict[str, mx.array]                        # inert env_* arrays for cfg
def init_state(cfg: Config) -> State | State3                              # uses cfg.run.seed; evaluated arrays
def volume_kinematics(k_pos: mx.array, k_head: mx.array, n: int, cfg: Config) -> tuple[mx.array, mx.array, mx.array]
                    # 3D (pos (n,3), heading, pitch) uniform in the volume: founders and immigrants (§27.4)
def total_energy(energy: mx.array, alive: mx.array, resource: mx.array, cfg: Config) -> mx.array   # int64 (); int64(n_alive)·S
def age(step: mx.array, birth_step: mx.array) -> mx.array                  # step − 1 − birth_step
def state_to_arrays(s: State | State3) -> dict[str, mx.array]
def state_from_arrays(d: Mapping[str, mx.array], cfg: Config | None = None) -> State | State3
                    # type state_cls(cfg) (without cfg: State3 iff the keys hold EXTRA_3D_FIELDS); key SET must equal its
                    # _fields, built in _fields order; exception: with a 2D cfg, a set missing only env_* keys (a
                    # pre-director checkpoint) is filled from default_env(cfg)
def state_digest(s: State | State3) -> str                                 # sha256 over all fields in type(s)._fields order (host)
def legacy_state_digest(s: State) -> str                                   # same encoding over V0_FIELDS only (§14.2); TypeError for State3
def to_numpy(a: mx.array) -> np.ndarray                                    # the only host read path: bf16 → f32 on device (exact, F17)

# rng.py
class Stream(IntEnum): NOISE = 0; BOND_PRIO = 1; BIRTH_PRIO = 2; PLACEMENT = 3; MUT_W1 = 4; MUT_W2 = 5
                       MUT_GENES = 6; IMMIGRANT = 7; CATASTROPHE = 8
class InitStream(IntEnum): POS = 0; HEADING = 1; W1 = 2; W2 = 3; TAG = 4; PATCH_POS = 5; PATCH_VEL = 6; STATE_KEY = 7
NUM_STREAMS: Final = 16; RNG_LAYOUT_VERSION: Final = 2; RNG_COMPATIBLE_VERSIONS: Final = frozenset({1, 2})
def step_keys(key: mx.array) -> tuple[mx.array, mx.array]                 # (key_next (2,), ks (16,2))
def init_keys(seed: int) -> mx.array                                       # (16,2)
def observer_key(seed: int, step: int) -> mx.array

# ops.py
def scatter_add_i32(size: int, idx: mx.array, val: mx.array, mask: mx.array | None = None) -> mx.array
def scatter_max_i32(size: int, idx: mx.array, val: mx.array, mask: mx.array) -> mx.array  # base −1, val ≥ 0
def set_rows(a: mx.array, idx: mx.array, rows: mx.array) -> mx.array      # out = mx.array(a); out[idx] = rows (unique idx)
def proportional_take(demand: mx.array, target: mx.array, active: mx.array, supply: mx.array
                      ) -> tuple[mx.array, mx.array]                       # (take (n,), taken (P,))
def exact_sum(x: mx.array, axis: int | None = None) -> mx.array           # int32 ≥ 0 → int64, 3 limbs
def rank_select(mask: mx.array, n: int) -> tuple[mx.array, mx.array]      # (idx (n,), valid (n,))
def place(arr: mx.array, newborn: mx.array, lane: mx.array, lane_vals: mx.array) -> mx.array
def first_free(partner: mx.array) -> tuple[mx.array, mx.array]            # (slot (N,) i32, has_free (N,) bool); full row → slot 0
def min_image(d: mx.array, L: float) -> mx.array
def wrap(x: mx.array, L: float) -> mx.array                                 # [0, L) with the F8 guard
def wrap_xy_clamp_z(p: mx.array, L: float, z_lo: float, z_hi: float) -> mx.array   # (...,3): wrap x, y; clip z (§27.2)
def wrap_angle(a: mx.array) -> mx.array                                     # [−π, π) in f32 and f64 terms (−f32(π) < −math.pi is lifted one ulp)
def to_body(v: mx.array, cos_h: mx.array, sin_h: mx.array) -> mx.array     # (...,2) → (fwd, lat)
def body_axes(cos_h: mx.array, sin_h: mx.array, cos_p: mx.array, sin_p: mx.array
              ) -> tuple[mx.array, mx.array, mx.array]                    # 3D: (fwd, lat, up), each (...,3) (§27.2)
def to_body3(v: mx.array, cos_h: mx.array, sin_h: mx.array, cos_p: mx.array, sin_p: mx.array
             ) -> mx.array                                                 # (...,3) → (fwd, lat, up) (§27.2)
def int_diffuse(r: mx.array, d_q8: int) -> mx.array
def floor_i32(x: mx.array) -> mx.array
def q8(f: float) -> int                                                      # host: round(f·256)
def mul_q8(x: mx.array, q: int) -> mx.array                                 # (x·q) >> 8

# spatial.py
class Neighbors(NamedTuple): idx; valid; off; dist; overflow_cells; max_occupancy    # §4; off (N,K,nd) (§27.2)
def find_neighbors(pos: mx.array, alive: mx.array, cfg: Config) -> Neighbors

# fields.py
class FieldObs(NamedTuple): res: mx.array; res_grad: mx.array; chem: mx.array; chem_grad: mx.array
                            # all f32 and already the final §5 inputs (sensing copies them into rows 12–23):
                            # res (N,) = IN_RES; res_grad (N,2) = IN_RES_GRAD; chem (N,NCH) = IN_CHEM;
                            # chem_grad (N,NCH,2) = IN_CHEM_GRAD as (fwd, lat) per channel
def init_patches(key_pos: mx.array, key_vel: mx.array, cfg: Config) -> tuple[mx.array, mx.array]
def season_weights(step: mx.array, cfg: Config, env: Env | None = None) -> mx.array   # (PP,) f32 (§13.1)
def fertility(patch_pos: mx.array, cfg: Config, step: mx.array | None = None,
              env: Env | None = None) -> mx.array                         # (FG,FG) f32 in [0,1]; 3D (FZ,FG,FG) (§27.5)
def vertical_profile(cfg: Config) -> mx.array                              # 3D: (FZ,1,1) f32, V[0] = 1 (§27.5)
def regen_rate(patch_pos: mx.array, cfg: Config, step: mx.array | None = None,
               env: Env | None = None) -> mx.array                        # (FG,FG) int32; 3D (FZ,FG,FG)
def cell_index(pos: mx.array, cfg: Config) -> mx.array                    # (N,) int32; 3D z-major (§27.5)
def sample_fields(resource: mx.array, chem: mx.array, fcell: mx.array, heading: mx.array, cfg: Config) -> FieldObs
def deposit_chem(emit: mx.array, fcell: mx.array, active: mx.array, cfg: Config) -> mx.array   # (NCH*FG*FG,) int32
def update_chem(chem: mx.array, dep_q: mx.array, cfg: Config) -> mx.array
def update_resource(resource: mx.array, patch_pos: mx.array, cfg: Config, step: mx.array | None = None,
                    env: Env | None = None) -> tuple[mx.array, mx.array]
                    # (resource', inc (FG,FG) int32) — regen then int diffusion
def drift_patches(patch_pos: mx.array, patch_vel: mx.array, cfg: Config, env: Env | None = None) -> mx.array

# sensing.py
def neighbor_descriptors(s: State, nbr: Neighbors, cfg: Config) -> mx.array          # (N,K,NF) f32; 3D NF = 8 (§27.2)
def build_inputs(s: State, age: mx.array, nbr: Neighbors, desc: mx.array, geom: BondGeom,
                 fobs: FieldObs, key: mx.array, cfg: Config) -> mx.array             # (N,D_IN) f32

# brain.py
def init_weights(key_w1: mx.array, key_w2: mx.array, n_rows: int, cfg: Config) -> tuple[mx.array, mx.array]
def forward(W1: mx.array, W2: mx.array, x: mx.array, h: mx.array, cfg: Config) -> tuple[mx.array, mx.array]
                    # (h_new (N,H) f32, y (N,D_OUT) f32)
def mutate_rows(W1p: mx.array, W2p: mx.array, sigma: mx.array, rate: mx.array,
                key_w1: mx.array, key_w2: mx.array, cfg: Config) -> tuple[mx.array, mx.array]   # wdtype rows
def l1_cost(W1r: mx.array, W2r: mx.array, cfg: Config) -> mx.array      # (rows,) int32

# actions.py
class Actions(NamedTuple): ...      # §7
class Motion3(NamedTuple): pitch: mx.array; buoy: mx.array                  # 3D-only outputs (§27.2)
def decode(y: mx.array, desc: mx.array, nbr: Neighbors, partner: mx.array, alive: mx.array, cfg: Config) -> Actions
def decode_motion3(y: mx.array, alive: mx.array, cfg: Config) -> Motion3    # dims = 3 only
def mask[A: (Actions, Motion3)](act: A, keep: mx.array) -> A              # returns the type it is given

# bonds.py
class BondGeom(NamedTuple): off: mx.array; length: mx.array; valid: mx.array    # (N,B,nd), (N,B), (N,B)
def bond_geometry(pos: mx.array, partner: mx.array, cfg: Config) -> BondGeom
def n_bonds(partner: mx.array) -> mx.array                                         # (N,) int32
def break_bonds(partner: mx.array, rev: mx.array, flag: mx.array) -> tuple[mx.array, mx.array, mx.array]
                    # (partner, rev, broken (N,B) bool)
def form_bonds(partner: mx.array, rev: mx.array, can_bond: mx.array, target: mx.array, key: mx.array
               ) -> tuple[mx.array, mx.array, mx.array, mx.array]
                    # (partner, rev, formed (N,) bool, as_proposer (N,) bool)
def drop_dead(partner: mx.array, rev: mx.array, survivor: mx.array) -> tuple[mx.array, mx.array]
def add_child_bonds(partner: mx.array, rev: mx.array, child_of: mx.array, want: mx.array,
                    born_parent: mx.array) -> tuple[mx.array, mx.array, mx.array]    # (partner, rev, made (N,) bool)
    # child_of: parent row → child row; born_parent: newborn row → parent row (−1 otherwise, immigrants too);
    # want: the parent's act.child_bond; made: parents that bonded (Σ made = CHILD_BONDS). Child side uses slot 0.
def spring_forces(pos: mx.array, partner: mx.array, contract: mx.array, cfg: Config) -> mx.array   # (N,nd)
def symmetric_ok(partner: mx.array, rev: mx.array, alive: mx.array) -> mx.array      # () bool

# physics.py
def forces(pos: mx.array, heading: mx.array, act: Actions, nbr: Neighbors, survivor: mx.array,
           partner: mx.array, cfg: Config) -> mx.array                              # (N,2)
def integrate(pos: mx.array, heading: mx.array, force: mx.array, grip: mx.array, turn: mx.array,
              survivor: mx.array, cfg: Config) -> tuple[mx.array, mx.array, mx.array, mx.array]
                    # (pos, vel, heading, n_clipped () int32); asserts (N,2) positions
def grounded(pos: mx.array, cfg: Config) -> mx.array                        # 3D: (N,) bool, z ≤ z_ground (§27.3)
def forces3(pos: mx.array, heading: mx.array, pitch: mx.array, act: Actions, mo: Motion3, buoy_cap: mx.array,
            nbr: Neighbors, survivor: mx.array, partner: mx.array, cfg: Config) -> mx.array    # 3D: (N,3) (§27.3)
class Integrated3(NamedTuple): pos; vel; heading; pitch; v_clips; z_floor; z_surface      # counts () int32
def integrate3(pos: mx.array, heading: mx.array, pitch: mx.array, force: mx.array, grip: mx.array, turn: mx.array,
               dpitch: mx.array, on_floor: mx.array, survivor: mx.array, cfg: Config) -> Integrated3   # 3D (§27.3)

# energy.py
class Costs(NamedTuple): energy; solvent; paid_basal; paid_action                  # (N,) each
def pay_costs(energy, act: Actions, n_bonds, brain_cost, alive, cfg: Config) -> Costs
def harvest(energy, resource, fcell, act: Actions, solvent, cfg: Config
            ) -> tuple[mx.array, mx.array, mx.array]                               # (E, resource, got)
class Combat(NamedTuple): energy; resource; lost; gain; spill; residue; attack; hit; max_attackers
                    # (N,) int32 ×5 except resource (FG,FG); gain, spill per attacker; lost, residue per victim
                    # (residue: what a fully drained victim loses beyond the grants, spilled into its own cell, §11);
                    # attack, hit (N,) bool (attackers after the S9-partner gate; grant > 0); max_attackers () int32
def combat(energy, resource, fcell, act: Actions, partner, cfg: Config) -> Combat     # partner: post-S9
class Share(NamedTuple): energy; out; inflow; heat          # (N,) each; out and heat are per-agent sums over slots
def share(energy, partner, rev, partner0, act: Actions, solvent, cfg: Config) -> Share
                    # partner, rev: post-S9; partner0: start-of-step s.partner (the `same` gate, §10.6)
def cap_energy(energy, cfg: Config) -> tuple[mx.array, mx.array]                   # (E, overflow (N,))

# lifecycle.py
class LaneKind(IntEnum): PAD = 0; BIRTH = 1; IMMIGRANT = 2
class DeathRecords(NamedTuple): uid; age; cause; generation; clade; valid           # (DL,)
class BirthRecords(NamedTuple): uid; parent_uid; clade; generation; slot; valid     # (M,)
class Deaths(NamedTuple): survivor; dead; energy; uid; partner; rev; resource; corpse; r_overflow; cause; records
    # (N,) except resource, r_overflow (FG,FG); corpse = E + S on dead rows else 0; cause = DeathCause on dead
    # rows, −1 elsewhere. Unused record lanes (deaths and births) hold uid/parent_uid/slot −1 and 0 elsewhere.
def flush_pending(W1, W2, pend_slot, pend_W1, pend_W2) -> tuple[mx.array, mx.array]   # via ops.set_rows
def deaths(alive, solvent, energy, lost, age, uid, generation, clade, partner, rev, resource, fcell, cfg: Config,
           *, cat_mask: mx.array | None = None) -> Deaths
    # dead = alive & (~solvent | E == 0 | aged) | excess catastrophe deaths (cat_mask); cause precedence in §10.7
def catastrophe_mask(pos: mx.array, alive: mx.array, env: Env, key: mx.array, cfg: Config) -> mx.array   # (N,) bool, §12.9
                    # 3D: a vertical column, x and y only (§27.4)
class Lanes(NamedTuple): parent; slot; kind; n_req; n_birth; n_imm                 # (M,),(M,),(M,),(),(),()
def allocate_lanes(req: mx.array, survivor: mx.array, key: mx.array, cfg: Config, step: mx.array | None = None,
                   *, imm_request: mx.array | None = None) -> Lanes     # imm_request: §12.7 (director)
class Repro(NamedTuple): state: State; records: BirthRecords; endow: mx.array; n_child_bonds: mx.array
def reproduce(s: State, lanes: Lanes, child_bond: mx.array, ks: mx.array, cfg: Config) -> Repro
    # s: in-flight state (alive = survivor, post-physics pos/heading, post-S14 bonds/energy).
    # Writes small arrays via ops.place, debits parents, adds child bonds, sets pend_*, advances next_uid.
    # 3D (State3): cone placement with reflection, immigrants in the volume, pitch and buoy_cap rows (§27.4).

# step.py
class StepOut(NamedTuple): counters; ledger; births: BirthRecords; deaths: DeathRecords     # §3
def step(s: State, cfg: Config) -> tuple[State, StepOut]
def consume_env(s: State, env: Env, lanes: Lanes) -> State                 # env_imm_request carry-over, env_cat_kill_p reset (§12.7, §12.9)
def make_step(cfg: Config, *, compile: bool | None = None) -> Callable[[State], tuple[State, StepOut]]
    # compile=None → cfg.run.compile

# invariants.py
def device_check(s: State, ledger: mx.array, prev_total: mx.array, max_cell_occ: mx.array,
                 max_attackers: mx.array, overflow_cells: mx.array, v_clips: mx.array, cfg: Config) -> mx.array
                    # int32 Diag bits for this step (ORed into s.diag by step)
def env_out_of_range(s: State, cfg: Config) -> mx.array                   # device bool: any env_* value non-finite / outside §2.2 ranges
def full_check(s: State, cfg: Config, prev: State | None = None, out: StepOut | None = None) -> list[str]
                    # host numpy; [] == OK. Per-step ledger = s.ledger_cum − prev.ledger_cum; births and immigrants
                    # from out.births.valid. Stateless: a cumulative dead-uid set, if wanted, lives in the test harness.

# metrics.py
def components(partner: mx.array, alive: mx.array, iters: int) -> tuple[mx.array, mx.array]   # labels, converged
def snapshot_metrics(s: State, cfg: Config, *, heavy: bool) -> dict[str, Any]     # host floats/ints/lists
def interval_row(step: int, counters: np.ndarray, ledgers: np.ndarray, snap: Mapping[str, Any],
                 perf: Mapping[str, float]) -> dict[str, Any]                     # counters (T,27), ledgers (T,16)
def carrying_capacity(s: State, cfg: Config, step: mx.array | None = None, env: Env | None = None) -> float
                    # step + env: the live regeneration (the director's K_est, §26.5)

# runio.py
FORMAT_VERSION: Final = 1
class RunDir:
    path: Path
    @classmethod
    def create(cls, path: Path, cfg: Config, *, overwrite: bool = False, parent: str | None = None) -> "RunDir"
    @classmethod
    def open(cls, path: Path) -> "RunDir"
    def config(self) -> Config
    def meta(self) -> dict[str, Any]
    def update_meta(self, **fields: Any) -> None
    def save_checkpoint(self, s: State, cfg: Config) -> Path        # write_state_file + LATEST + prune
    def checkpoints(self) -> list[tuple[int, Path]]
    def latest_checkpoint(self) -> Path | None
    def append_metrics(self, rows: Iterable[Mapping[str, Any]]) -> None
    def append_jsonl(self, name: str, records: Iterable[Mapping[str, Any]]) -> None   # append + fsync (director files)
    def read_jsonl(self, name: str) -> list[dict[str, Any]]                          # a torn last line is ignored
    def write_json(self, name: str, obj: Any) -> None                                # atomic (director_archive.json)
    def write_events(self, s0: int, s1: int, births: Mapping[str, np.ndarray], deaths: Mapping[str, np.ndarray]) -> None
    def truncate_after(self, step: int, *, keep_director: bool = False) -> None   # also cuts the director files (§26.3)
    def dump_diag(self, s: State, step: int) -> Path
def write_state_file(path: Path, s: State, cfg: Config) -> None   # atomic (*.tmp + os.replace); checkpoint metadata
def checkpoint_metadata(path: Path) -> dict[str, str]
def load_checkpoint(path: Path, cfg: Config, *, check_hash: bool = True) -> State
                    # CheckpointError on dims/version/shape mismatch, or config_hash mismatch when check_hash (fork passes False)

# runner.py
class ExitCode(IntEnum): OK = 0; ERROR = 1; CONFIG = 2; INVARIANT = 3; CHECKPOINT = 4; INTERRUPTED = 130
class RunResult(NamedTuple): state: State; status: str; exit_code: ExitCode     # status: done|extinct|violation|interrupted
def run(rd: RunDir, cfg: Config, state: State, steps: int, *, quiet: bool = False,
        on_interval: Callable[[State, dict[str, Any]], None] | None = None,
        director: Director | None = None) -> RunResult        # the director hook: §26.1

# render.py
@dataclass(frozen=True) class Viewport: cx: float; cy: float; zoom: float = 1.0
DEFAULT_LAYERS: frozenset[str] = frozenset({"resource", "chem", "bonds", "agents"})
def render_frame(s: State, cfg: Config, size: tuple[int, int] = (1024, 1024), viewport: Viewport | None = None,
                 layers: frozenset[str] = DEFAULT_LAYERS, color: str = "tag") -> mx.array   # (H,W,3) uint8
def write_png(path: Path, img: np.ndarray) -> None
class VideoWriter:
    def __init__(self, path: Path, size: tuple[int, int], fps: int = 30) -> None
    def write(self, frame: np.ndarray) -> None
    def close(self) -> None

# viewer.py
def run_viewer(cfg: Config, state: State, *, rundir: RunDir | None = None, size: tuple[int, int] = (900, 900),
               steps_per_frame: int = 1) -> None        # ImportError with an install hint without pygame-ce
class ViewerSession: ...           # the viewer minus the window (state, viewport, keys, selection, HUD, snapshots)

# bench.py
def run_bench(cfg: Config, *, steps: int = 200, warmup: int = 20, stages: bool = False,
              determinism: bool = False, brain_variants: bool = False) -> dict[str, Any]   # JSON-compatible
def format_report(report: dict[str, Any]) -> str                            # terminal summary of a report

# cli.py
def main(argv: Sequence[str] | None = None) -> int
```

### 16.3 `step` wiring (normative pseudocode)

```python
def step(s, cfg):
    d = dims(cfg); t = s.step
    W1, W2 = lifecycle.flush_pending(s.W1, s.W2, s.pend_slot, s.pend_W1, s.pend_W2)        # S0
    s = s._replace(W1=W1, W2=W2)
    key_next, ks = rng.step_keys(s.key)                                                    # S1
    a = state.age(t, s.birth_step)
    nbr = spatial.find_neighbors(s.pos, s.alive, cfg)                                      # S2
    geom = bonds.bond_geometry(s.pos, s.partner, cfg)                                      # S3
    fcell = fields.cell_index(s.pos, cfg)                                                  # S4
    fobs = fields.sample_fields(s.resource, s.chem, fcell, s.heading, cfg)
    desc = sensing.neighbor_descriptors(s, nbr, cfg)                                       # S5
    x = sensing.build_inputs(s, a, nbr, desc, geom, fobs, ks[Stream.NOISE], cfg)
    h_new, y = brain.forward(s.W1, s.W2, x, s.h, cfg)                                      # S6
    act = actions.decode(y, desc, nbr, s.partner, s.alive, cfg)                            # S7
    nb0 = bonds.n_bonds(s.partner)
    c = energy.pay_costs(s.energy, act, nb0, s.brain_cost, s.alive, cfg)                   # S8
    act = actions.mask(act, c.solvent)
    flag = act.release | (geom.length > cfg.bonds.break_length)                            # S9
    partner, rev, broken = bonds.break_bonds(s.partner, s.rev, flag & geom.valid)
    partner, rev, formed, as_prop = bonds.form_bonds(partner, rev, s.alive & c.solvent & act.bond_want,
                                                     act.bond_target, ks[Stream.BOND_PRIO])
    E, resource, got = energy.harvest(c.energy, s.resource, fcell, act, c.solvent, cfg)  # S10
    cb = energy.combat(E, resource, fcell, act, partner, cfg)                              # S11
    sh = energy.share(cb.energy, partner, rev, s.partner, act, c.solvent, cfg)             # S12
    E, e_over = energy.cap_energy(sh.energy, cfg)                                          # S13
    dt = lifecycle.deaths(s.alive, c.solvent, E, cb.lost, a, s.uid, s.generation, s.clade, partner, rev,
                          cb.resource, fcell, cfg)                                         # S14
    F = physics.forces(s.pos, s.heading, act, nbr, dt.survivor, dt.partner, cfg)           # S15
    pos, vel, heading, v_clips = physics.integrate(s.pos, s.heading, F, act.grip, act.turn, dt.survivor, cfg)
    s1 = s._replace(alive=dt.survivor, energy=dt.energy, uid=dt.uid, partner=dt.partner, rev=dt.rev,
                    resource=dt.resource, pos=pos, vel=vel, heading=heading,
                    h=mx.where(dt.survivor[:, None], h_new, 0.0),
                    msg=mx.where(dt.survivor[:, None], act.msg, 0.0),
                    last_loss=mx.where(dt.survivor, cb.lost, 0), last_harvest=mx.where(dt.survivor, got, 0),
                    last_inflow=mx.where(dt.survivor, sh.inflow, 0))
    req = dt.survivor & act.reproduce & (dt.energy >= cfg.repro.min_energy) & (a >= cfg.repro.min_age)
    lanes = lifecycle.allocate_lanes(req, dt.survivor, ks[Stream.BIRTH_PRIO], cfg)        # S16
    rp = lifecycle.reproduce(s1, lanes, act.child_bond, ks, cfg)
    s2 = rp.state
    dep = fields.deposit_chem(act.emit, fcell, c.solvent, cfg)                            # S17
    chem = fields.update_chem(s2.chem, dep, cfg)
    resource, regen_inc = fields.update_resource(s2.resource, s2.patch_pos, cfg)
    patch_pos = fields.drift_patches(s2.patch_pos, s2.patch_vel, cfg)
    return finish(s2._replace(chem=chem, resource=resource, patch_pos=patch_pos, key=key_next, step=t + 1),
                  s, cfg, c=c, got=got, cb=cb, sh=sh, e_over=e_over, dt=dt, lanes=lanes, rp=rp,
                  regen_inc=regen_inc, dep=dep, nbr=nbr, v_clips=v_clips, fcell=fcell,
                  desc=desc, x=x, act=act, as_prop=as_prop, formed=formed, broken=broken)   # S18
```

`act.msg` is masked by `alive` and by solvency, so insolvent and dead agents broadcast 0.

**With `director.enabled`** (static) the wiring above gains four things and nothing else changes:
`env = s.env`; S14 passes `cat_mask=lifecycle.catastrophe_mask(s.pos, s.alive, env, ks[Stream.CATASTROPHE], cfg)`
to `deaths`; S16 passes `imm_request=env.imm_request` to `allocate_lanes`; S17 passes `env` to `update_resource` and
`drift_patches`, and afterwards `step.consume_env(s2, env, lanes)` writes the carried-over request and resets
`env_cat_kill_p`. Both outputs are computed from inputs (`max(request − n_imm, 0)`, `kill_p · 0`): MLX 0.32
`mx.compile` fails (`_Map_base::at`) on a constant output next to arrays that use it. `bench._StagedStep` mirrors
these arguments.

`finish` is a private helper in `step.py`. It also computes
`max_cell_occ = ops.scatter_add_i32(FG², fcell, 1, s.alive).max()`, the start-of-step field-cell occupancy that
`device_check` needs for the headroom bit. It:
1. builds the per-agent flow matrix `[paid_basal, paid_action, got, cb.lost, cb.gain, cb.spill + cb.residue, sh.out,
   sh.heat, e_over, dt.corpse]` (all (N,) int32; `sh.out` and `sh.heat` are already summed over slots; the spill
   column is `FLOW_SPILL`), the per-cell matrix
   `[regen_inc, dt.r_overflow]` and the lane `endow` vector;
2. reduces them with `ops.exact_sum`;
3. forms the count × constant terms in int64 (§10.8): `HEAT_BIRTH = int64(n_birth)·birth_cost`,
   `FLOW_STRUCTURE = int64(n_birth)·S`, `INJECT_IMMIGRATION = int64(n_imm)·(e_init + S)`;
4. assembles `ledger (16,)`, adds it to `ledger_cum`, and computes the new `total` (via `state.total_energy`, whose
   `S·n_alive` term is int64);
5. assembles `counters (27,)`. The kin counters are computed here and nowhere else:
   - `ATTACKS = Σ cb.attack`, `ATTACK_HITS = Σ cb.hit`;
   - `KIN_ATTACK_SUM = Σ_{cb.attack} round(256·desc[i, act.attack_k, NBR_KIN])`,
     `KIN_ATTACK_BASE = Σ_{cb.attack} round(256·x[i, IN_KIN_MEAN])`;
   - `BOND_PROPOSERS = Σ as_prop`; `KIN_BOND_SUM = Σ_{as_prop} round(256·desc[i, act.bond_k, NBR_KIN])`,
     `KIN_BOND_BASE = Σ_{as_prop} round(256·x[i, IN_KIN_MEAN])`. A mutual pair has two proposers and counts twice.
     Every `as_prop` row has `bond_k ≥ 0`, so the gather is valid;
6. ORs `invariants.device_check(...)` into `diag` (skipped when `runtime_checks == "off"`). With the director
   enabled it also ORs `FIELD_RANGE` if the *start-of-step* env values were out of range
   (`invariants.env_out_of_range(s)`), because `env_imm_request` and `env_cat_kill_p` are consumed by the step and
   the end-of-step state would hide a bad value.

---

## 17. Config

The config is a set of frozen dataclasses, one per section. **Defaults are the `64k` preset.** Every tunable numeric
constant of the simulation step lives here; fixed normalisation constants that shape the genome's inputs, and
numeric guards, are named constants in `layout.py` (§16.2). Derived limits (`occ_limit`) are computed in `Dims`.
Rendering colours are local to `render.py`.

```python
@dataclass(frozen=True)
class WorldConfig:
    size: float = 1024.0              # torus side L
    field_cell: float = 4.0           # FG = size / field_cell (must be an integer ≤ 1024)
    dims: int = 2                     # 2 | 3, static (§27.1); added after v0.1
    height: float = 0.0               # 3D world height (floor z = 0, surface z = height); 0 in 2D

@dataclass(frozen=True)
class PopulationConfig:
    n_max: int = 65536                # N rows (≤ 2^20)
    n_init: int = 16384               # founders, rows [0, n_init)
    max_births_per_step: int = 1024   # M lanes; N_CAP = n_max − M
    max_age: int = 10000              # senescence death; 0 disables
    immigration_floor: int = 0        # 0 disables immigration (static)
    immigrants_per_step: int = 64
    immigration_every: int = 0        # 0 disables periodic immigration (static); added after v0.1
    immigration_count: int = 0        # founders requested on steps t with (t + 1) % every == 0

@dataclass(frozen=True)
class SensingConfig:
    radius: float = 6.0               # R; hash cell ≥ R
    k_neighbors: int = 8              # K
    cell_capacity: int = 16           # C
    tag_scale: float = 0.5
    clock_period: int = 32
    chem_half_sat: float = 1.0

@dataclass(frozen=True)
class BrainConfig:
    hidden: int = 32                  # H
    weight_dtype: str = "float16"     # float16 | bfloat16 | float32
    impl: str = "matmul"              # matmul | einsum
    init_scale: float = 1.0
    recurrent_scale: float = 0.5
    weight_clip: float = 4.0

@dataclass(frozen=True)
class PhysicsConfig:
    radius: float = 1.0               # agent radius r
    thrust_max: float = 0.1
    turn_max: float = 0.3
    drag: float = 1.0
    grip_gain: float = 4.0
    v_max: float = 0.5
    k_repel: float = 0.08
    thrust_contact_limit: float = 4.0 # thrust × max(0, 1 − contacts/limit) (§9)
    gravity: float = 0.0              # 3D only (§27.3): downward body force; added after v0.1
    floor_eps: float = 0.0            # 3D only: grounded iff z ≤ radius + floor_eps
    buoyancy_max: float = 0.0         # 3D only: upward force at buoy_cap = 1 and full OUT_BUOYANCY; 0 = none

@dataclass(frozen=True)
class BondConfig:
    rest_length: float = 2.0
    k_spring: float = 0.05
    contract_gain: float = 0.25
    form_range: float = 3.0
    break_length: float = 4.0
    release_threshold: float = 1.0
    share_rate: int = 1000            # q per slot per step at share = 1
    share_efficiency: float = 0.9

@dataclass(frozen=True)
class EnergyConfig:
    e_init: int = 50000
    e_max: int = 1000000              # < 2^20
    structure: int = 10000            # S
    basal_cost: int = 100
    brain_cost_per_l1: float = 0.1
    bond_upkeep: int = 5
    move_cost: int = 100
    turn_cost: int = 10
    grip_cost: int = 20
    contract_cost: int = 5
    msg_cost: int = 5
    emit_cost: int = 20
    harvest_cost: int = 10
    defend_cost: int = 20
    attack_cost: int = 50
    harvest_max: int = 400

@dataclass(frozen=True)
class CombatConfig:
    range: float = 3.0
    threshold: float = 0.5
    attack_max: int = 4000
    efficiency: float = 0.7
    defend_efficiency: float = 0.75
    attack_bonded: bool = False

@dataclass(frozen=True)
class ReproConfig:
    min_energy: int = 40000
    min_age: int = 20
    birth_cost: int = 2000
    offspring_distance: float = 2.0

@dataclass(frozen=True)
class MutationConfig:
    sigma_init: float = 0.03; sigma_min: float = 0.002; sigma_max: float = 0.3; tau_sigma: float = 0.15
    rate_init: float = 0.05;  rate_min: float = 0.001;  rate_max: float = 0.5;  tau_rate: float = 0.15
    frac_init: float = 0.5;   frac_min: float = 0.1;    frac_max: float = 0.9;  tau_frac: float = 0.1
    tag_sigma: float = 0.02
    buoy_init: float = 0.0;   buoy_sigma: float = 0.0                       # 3D only (§27.4): the buoyancy cap

@dataclass(frozen=True)
class FieldConfig:
    resource_cap: int = 8000
    regen_max: int = 320
    fertility_base: float = 0.05
    n_patches: int = 24
    patch_sigma: float = 48.0
    patch_drift: float = 0.005
    resource_diffusion_q8: int = 4
    chem_diffusion: tuple[float, float, float] = (0.20, 0.10, 0.24)
    chem_decay: tuple[float, float, float] = (0.10, 0.03, 0.005)
    chem_deposit_scale: float = 1.0 / 256.0
    chem_max: float = 16.0
    benthic_lambda: float = 0.0       # 3D only (§27.5): e-folding height of the floor food band; added after v0.1
    photic_amp: float = 0.0           # 3D only: surface food band amplitude in [0, 1]; 0 = off
    photic_lambda: float = 0.0        # 3D only: its e-folding depth below the surface

@dataclass(frozen=True)
class DirectorConfig:                  # added after v0.1; docs/DIRECTOR.md
    enabled: bool = False              # static: the compiled step reads State.env (§2.2, §3); off = today's graph
    persona: str = "gardener"          # gardener | trickster | architect | random (§26.11); the keys below are policy (§26.2), unhashed
    goal: str = ""                     # the Architect's goal text (stored, not interpreted before the LLM tier)
    rule_every: int = 1000             # steps between rule-tier decisions; multiple of the metrics cadences
    budget_per_100k: int = 0           # cost units per rolling 100k steps; 0 = the persona's default
    llm_every: int = 0                 # LLM tier (§26.10): steps between calls; 0 = off, else a multiple of rule_every
    llm_temperature: float = 0.4       # sampling temperature of the request
    llm_timeout_s: float = 60.0        # seconds the run waits for a reply before the rule tier takes over
    llm_max_tokens: int = 400          # reply length limit of the request
    yoke_log: str = ""                 # random persona only (§26.11): the director.jsonl whose discretionary moves it matches

@dataclass(frozen=True)
class RunConfig:                       # excluded from config_hash; on resume only observer/cadence keys may change (§18)
    seed: int = 0
    steps: int = 100_000
    metrics_every: int = 100
    checkpoint_every: int = 10_000     # multiple of metrics_every
    keep_checkpoints: int = 3          # plus the step-0 checkpoint, which is always kept
    compile: bool = True
    runtime_checks: str = "cheap"      # cheap | off
    heavy_metrics_every: int = 1000    # components + weight diversity; multiple of metrics_every
    components_iters: int = 32
    diversity_sample: int = 256
    director: str = "off"              # off | live | replay (§26.2); needs director.enabled; resume may change it
    llm_url: str = "http://localhost:1234/v1/chat/completions"   # the LLM tier's endpoint (§26.10)
    llm_model: str = "gemma-4-26b-a4b-it-mlx@8bit"               # the model named in each request; resume may change both

@dataclass(frozen=True)
class Config:
    world: WorldConfig = WorldConfig(); population: PopulationConfig = PopulationConfig()
    sensing: SensingConfig = SensingConfig(); brain: BrainConfig = BrainConfig()
    physics: PhysicsConfig = PhysicsConfig(); bonds: BondConfig = BondConfig()
    energy: EnergyConfig = EnergyConfig(); combat: CombatConfig = CombatConfig()
    repro: ReproConfig = ReproConfig(); mutation: MutationConfig = MutationConfig()
    fields: FieldConfig = FieldConfig(); director: DirectorConfig = DirectorConfig(); run: RunConfig = RunConfig()
```

**Presets** are TOML files in `alife/presets/` holding sparse overrides of the defaults.

| preset | overrides |
|---|---|
| `tiny` | `population: n_max 1024, n_init 256, max_births_per_step 32`; `world.size 128`; `fields: n_patches 4, patch_sigma 12`; `run: steps 500, metrics_every 10, checkpoint_every 50, heavy_metrics_every 10` |
| `dev` | `population: n_max 16384, n_init 4096, max_births_per_step 256`; `world.size 512`; `fields: n_patches 12, patch_sigma 32` |
| `64k` | (empty: the defaults) |
| `256k` | `population: n_max 262144, n_init 65536, max_births_per_step 4096`; `world.size 2048`; `fields.n_patches 96` |
| `fast` (overlay) | `sensing: k_neighbors 6, cell_capacity 12`; `brain.hidden 24` |

**Loading** (`load_config`). Precedence, lowest first:
1. dataclass defaults;
2. `preset` (a name, default `64k`);
3. `overlays` (names, in order);
4. the user `path` (TOML via `tomllib`, or JSON, chosen by extension);
5. `overrides` (`--set section.key=value`, repeatable).

Rules:
- Each override value is parsed with the target field's type: `int`, `float`, `bool` (`true`/`false`), `str`, or a
  tuple of floats (`0.2,0.1,0.24`).
- Unknown sections or keys are errors.
- The result is validated.
- Only the standard library is used; there is no `tomli-w`.

**Saving.** Each run writes the fully resolved config to `config.json` as canonical JSON (sorted keys, tuples as
lists). It reloads exactly: `config_from_dict(config_to_dict(c)) == c`. `config_hash` covers every section
except `run`. **Keys added after v0.1 are omitted from the hash at their defaults** (`config.POST_V01_KEYS`:
`population.immigration_every`, `population.immigration_count`, `fields.season_period`,
`fields.season_amplitude`, `fields.season_mode`, `director.enabled`, `world.dims`, `world.height` (§27.1),
`physics.gravity`, `physics.floor_eps`, `physics.buoyancy_max`, `mutation.buoy_init`, `mutation.buoy_sigma`,
`fields.benthic_lambda`, `fields.photic_amp`, `fields.photic_lambda` (§27.1, §27.3–§27.5); the
director's policy keys `persona`, `goal`, `rule_every`, `budget_per_100k`, (step 3) `llm_every`, `llm_temperature`, `llm_timeout_s`, `llm_max_tokens` and (§26.11) `yoke_log` are also in the list but are **omitted at every value**, `config.UNHASHED_KEYS`,
because only `enabled` changes the compiled step). A section that this leaves empty
(`director`) is omitted from the hashed dict too, so the hash does not change when a section is added. A
`config.json` that predates them therefore loads with the
defaults (sparse dicts fill missing keys, and a missing `director` section is the default one), hashes exactly as
before, and resumes without a hash mismatch; the
`tiny` preset's hash `d25e0eae…61d5` is pinned in `tests/test_features.py`. Setting any of them off default
changes the hash, as for any dynamics key.

**`validate`** collects every violation and raises one `ConfigError`:
- **World geometry:** `size / field_cell` is an integer with `3 ≤ FG ≤ 1024`; `G_h = floor(size/radius) ≥ 3`.
- **World dimensions** (§27.1): `world.dims ∈ {2, 3}` and in `config.DIMS_SUPPORTED` (`(2,)`: dims = 3 is not
  enabled yet); in 2D every 3D key (`config.THREE_D_KEYS`) keeps its default (`height` and the gravity, buoyancy and
  vertical-profile keys 0); the 3D-only rules are listed in §27.1.
- **Interaction ranges:** `radius ≥ max(form_range, combat.range, 2·physics.radius)`; `NS ≤ K ≤ 9·C` (2D;
  `27·C` in 3D, §27.1).
- **Population:** `64 ≤ n_max ≤ 2²⁰`; `1 ≤ M ≤ n_max // 4`; `1 ≤ n_init ≤ N_CAP`; `immigration_floor ≤ N_CAP`;
  `immigrants_per_step ≥ 1` when `immigration_floor > 0` (otherwise an extinct world could never refill, yet
  the §18 extinction stop is off); `immigration_every ≥ 0`; `1 ≤ immigration_count` when
  `immigration_every > 0`; `immigration_count ≤ M`.
- **Director:** `director.enabled` is a plain bool. With it on, `fields.n_patches` also fixes the length of
  `env_patch_gain`, and immigration is possible (`immigration_possible`) even when `immigration_floor = 0`.
  `persona ∈ {gardener, trickster, architect, random}`; the `random` persona needs a non-empty `yoke_log` and
  `llm_every = 0`, and `yoke_log` is empty for every other persona (§26.11); `rule_every ≥ 1`; `budget_per_100k ≥ 0`; `run.director ∈ {off, live,
  replay}`; when `run.director` is not `off`: `director.enabled` is true and `rule_every` is a multiple of both
  `run.metrics_every` and `run.heavy_metrics_every` (§26.2). LLM tier (§26.10): `llm_every` is 0 or a positive
  multiple of `rule_every`; `0 ≤ llm_temperature ≤ 2`; `llm_timeout_s` positive and finite; `llm_max_tokens ≥ 1`;
  `run.llm_model` non-empty; with `llm_every > 0`, `run.llm_url` is an http(s) URL with a host.
- **Fields (seasons):** `season_period ≥ 0`; `0 ≤ season_amplitude ≤ 1`; `season_period ≥ 1` when
  `season_amplitude > 0`; `season_mode ∈ {global, per_patch}`.
- **Physics stability:** `stability_number ≤ 1.8`; `thrust_max < 2·radius·k_repel`; `v_max ≤ physics.radius`;
  `thrust_contact_limit > 0`.
- **Bond geometry:** `rest_length·(1 + contract_gain) < break_length`; `form_range < break_length`;
  `offspring_distance < break_length`.
- **Energy bounds:** `e_max < 2²⁰`; `1 ≤ e_init ≤ e_max`; `0 ≤ structure < 2²⁰`;
  `min_energy > birth_cost + structure`. With the engine's quantised fraction
  `frac_q8(f) = floor(float32(f)·256 + 0.5)` (§10.7): `frac_q8(frac_min) ≥ 1`,
  `((min_energy − birth_cost − structure)·frac_q8(frac_min)) >> 8 ≥ 1` and `frac_q8(frac_max) ≤ 255`, so the
  child gets `endow ≥ 1` and the parent keeps `Dp − endow ≥ 1`.
- **Contention headroom:** `harvest_max`, `attack_max` and `share_rate` are each `< 2¹⁶`.
- **Resource field:** `0 < resource_cap < 2²⁴ − 1`; `0 ≤ regen_max ≤ resource_cap`;
  `0 ≤ resource_diffusion_q8 ≤ 64` (2D; `≤ 42` in 3D, §27.1).
- **Chemicals:** every `chem_diffusion` is in `[0, 0.25]` (2D; `≤ 1/6` in 3D, §27.1); every `chem_decay` is in
  `[0, 1)`; both tuples have length `NCH`.
- **Rates and gene bounds:** efficiencies are in `(0, 1]`; `0 < frac_min ≤ frac_init ≤ frac_max < 1`;
  `0 < sigma_min ≤ sigma_init ≤ sigma_max`; `0 < rate_min ≤ rate_init ≤ rate_max < 1`.
- **Enumerations:** `weight_dtype ∈ {float16, bfloat16, float32}`; `impl ∈ {matmul, einsum}`;
  `runtime_checks ∈ {cheap, off}`.
- **Cadences:** `checkpoint_every % metrics_every == 0`; `heavy_metrics_every % metrics_every == 0`.
- **Sanity** (so the rules above are well defined): `size`, `field_cell`, `sensing.radius`, `tag_scale`,
  `chem_half_sat`, `physics.radius`, `drag`, `turn_max`, `v_max`, `weight_clip`, `patch_sigma`, `chem_max` are
  `> 0`; `hidden`, `cell_capacity`, `clock_period`, `n_patches`, `metrics_every`, `keep_checkpoints`,
  `components_iters` are `≥ 1`; `diversity_sample ≥ 2`; costs, `max_age`, immigration counts, `birth_cost`,
  `min_age` and `steps` are `≥ 0`; `0 ≤ seed < 2⁶³`.

---

## 18. Run directory format

```text
runs/<name>/                                   default name: <preset>-<YYYYmmdd-HHMMSS>-s<seed>
  config.json                                  resolved config (canonical JSON)
  meta.json                                    see below
  metrics.jsonl                                one JSON object per metrics interval (§19.2), key "step" first
  checkpoints/
    step_0000000000.safetensors                every State field by name (state_to_arrays), env_* included
    step_0000010000.safetensors
    LATEST                                     text: file name of the newest complete checkpoint
  events/
    births_<s0:010d>_<s1:010d>.npz             step i32, uid i64, parent_uid i64, clade i32, generation i32, slot i32
    deaths_<s0:010d>_<s1:010d>.npz             step i32, uid i64, age i32, cause u8, generation i32, clade i32
  diag/step_<step>.safetensors                 written automatically when an error diag bit is found (exit 3)
  director.jsonl                               director runs only: the decision log (header + entries), §26.3
  journal.jsonl                                director runs only: one-liners and LLM call records, never read by the simulation
  director_archive.json                        director runs only: the behaviour archive at the last checkpoint
  media/                                       render and video outputs; viewer snapshots snap_<step>.safetensors
```

- **`meta.json` fields:**
  - versions: `format_version`, `alife_version`, `mlx_version`, `python`;
  - environment: `platform`, `device` (`mx.default_device()`), `git_rev` (or null), `created_utc`;
  - run identity: `seed`, `config_hash`, `rng_layout_version`, `genome_layout_version` (of the world's dims),
    `dims` (3D only; absent = 2D, §27.1), `parent`, `start_step` (the step the directory started from: 0, or the
    fork step);
  - `status`, one of `running | done | extinct | violation | interrupted`.
- **Checkpoint metadata** (safetensors string metadata, written by `runio.write_state_file` for checkpoints and
  viewer snapshots alike): `format_version`, `rng_layout_version`, `genome_layout_version`, `step`, `config_hash`,
  `alife_version`, `mlx_version`, `weight_dtype`, and `config` (the canonical config JSON), so a state file is
  self-describing; a 3D state also records `dims` (absent = 2D, §27.1).
- **Versions and pre-director files.** `load_checkpoint` accepts any `rng_layout_version` in
  `rng.RNG_COMPATIBLE_VERSIONS` ({1, 2}); new files write 2 (`meta.json` too). A file written before the `env_*`
  fields (38 arrays) loads with inert env values from `default_env(cfg)`: `state_from_arrays(d, cfg)` fills the
  missing keys only when every missing key is an `env_*` field, and any other mismatch still raises. A pre-director
  run directory therefore resumes under a director-off config (same hash), and forks (`--init-from`, hash check off)
  under a director-enabled config start with inert env. `env_*` shapes and dtypes are checked like every field.
- **Atomic writes.** Each checkpoint is written to `*.tmp` and moved into place with `os.replace`. Only then is
  `LATEST` rewritten, also atomically. A crash never corrupts the newest complete checkpoint.
- **Retention.** The newest `keep_checkpoints` are kept, plus `step_0000000000`.
- **Cadence.** Scheduled checkpoints happen only at metrics boundaries. Event records are read to the host at every
  metrics boundary and written as one chunk per checkpoint interval, `[previous checkpoint step, step)`. The runner
  always flushes metrics and events *before* writing a checkpoint, so logs are consistent with every checkpoint.
- **Unscheduled stops** (end of run, extinction, SIGINT). The runner finishes the current step, writes a partial
  interval row whose `steps` field is the actual number of steps in it, flushes events, and checkpoints at that
  step. On resume the next boundary is the next multiple of `metrics_every`, so the first resumed interval may also
  be short. Every interval row therefore carries `steps`, and consumers weight by it. Extinction (no agent alive,
  immigration off, and no director `env_imm_request` pending) is read from each step's already evaluated `N_ALIVE`
  counter, so it costs no extra sync; the pending-request check is a host read of an evaluated scalar in the runner.
- **Violations.** An error `diag` bit, or a non-zero host `ledger_error`, found at a metrics boundary: the row and the
  pending events are flushed, the state is dumped to `diag/`, `status = violation`, exit 3. An invalid state is never
  checkpointed, so resume restarts from the last good checkpoint.
- **Resume** (`alife resume RUN`):
  1. Read `config.json`.
  2. Load `LATEST`, or `--checkpoint STEP`. Without `--steps`/`--until` the run continues to
     `meta.start_step + run.steps`.
  3. Refuse with exit 4 if `dims` (§27.1), `format_version`, `rng_layout_version`, `genome_layout_version` or
     `config_hash` differ.
     Resume accepts only `--set run.*` overrides of observer and cadence keys (`steps`, `metrics_every`,
     `checkpoint_every`, `keep_checkpoints`, `heavy_metrics_every`, `components_iters`, `diversity_sample`), the
     director mode `run.director` (`--director MODE` is a shortcut, translated before this check) and the director's
     policy keys `director.persona|goal|rule_every|budget_per_100k`. Overriding `run.seed`, `run.compile` or
     `run.runtime_checks`, or any dynamics key (`director.enabled` included), is a config error (exit 2): those
     changes go through a fork. `--director live|replay` on a run whose `director.enabled` is false is a config error
     that points to `run --init-from`. The accepted overrides are written back to `config.json`, and the cadences are
     revalidated (§17).
  4. `truncate_after(step)`: drop metrics rows with `step > ckpt_step`, delete or trim event chunks that extend
     past it, and delete checkpoints after it (they belong to the abandoned continuation). An event chunk
     `*_<s0>_<s1>.npz` holds the records with `s0 ≤ step < s1`, where a record's `step` is the index of the step
     during which it happened, so a checkpoint at `state.step = s1` contains exactly the chunks ending at `s1`; a
     trimmed chunk is renamed `*_<s0>_<ckpt_step>.npz`. The director files are cut the same way, unless the mode is
     `replay`, which never touches `director.jsonl` (§26.3). A `replay` log is pre-validated before anything is
     truncated.
  5. Continue. The checkpoint holds the full state (key, `pend_*`, ledger), so the resumed trajectory is
     **bitwise identical** to an uninterrupted one. This is tested.
- **Fork** (`alife run --init-from RUN[:STEP]|FILE.safetensors --out NEW [--set …]`) starts a new run directory
  from another run's population (or a viewer snapshot) with a changed dynamics config. Examples: predation off
  (`combat.threshold` raised), different regeneration, `combat.attack_bonded = true`.
  - **Config.** It starts from the source state file's embedded `config`, then applies `--overlay`, `--config` and
    `--set`. `--preset` is rejected with `--init-from`.
  - **Shapes.** All state shapes must match the new config, `world.dims` and `world.height` must equal the
    source's (§27.1), and `population.n_init` must equal the source's (clade bins are `n_init + 1`); otherwise
    exit 4.
  - **Kept:** `step`, `next_uid`, every uid, `birth_step` (so derived ages stay correct), lineage fields, `pend_*`,
    fields and patches.
  - **Reset:** `diag = 0`; `ledger_cum = 0`; `total0 = total = loaded total + (S_new − S_old)·n_alive`. Structure
    energy is not in any `E`, so the conserved total moves with `energy.structure`; a loaded total that was
    already inconsistent stays inconsistent and fails the admission check.
  - **Key.** `state.key` is kept, unless `--seed` is given, in which case `key = init_keys(seed)[STATE_KEY]`.
  - **Environment.** All `env_*` fields are reset to `default_env(new_cfg)` (inert, with the new config's seasons),
    and a note line is printed if that changed anything. The source's director history does not carry over: a fork
    starts with `run.director = off` (whatever the source ran) unless `--director` is given, and a new log starts
    empty. The one exception is a replay fork (`--director replay --replay-log LOG`), which copies `LOG` verbatim
    (§26.8). Resume never resets the environment.
  - **Admission check.** `invariants.full_check(s, new_cfg)` runs on the loaded state before the first step. A
    state that violates the new config (e.g. `E > e_max` after lowering `e_max`, genes outside new bounds) is
    refused with exit 4.
  - `meta.parent` records the origin.

---

## 19. Metrics for emergence

### 19.1 Per-step counters (`CounterIdx`, `StepOut.counters`, int32 (27,))

| idx | name | idx | name |
|---|---|---|---|
| 0 | `N_ALIVE` (end of step) | 13 | `BONDS_BROKEN` (pairs, release, strain and death) |
| 1 | `BIRTHS` | 14 | `CHILD_BONDS` |
| 2 | `BIRTH_REQUESTS` | 15 | `BONDS_TOTAL` (pairs, end of step) |
| 3 | `BIRTHS_BLOCKED` | 16 | `SHARERS` (agents with Σ out > 0) |
| 4 | `IMMIGRANTS` | 17 | `EMITTERS` (agents with any emit quanta > 0) |
| 5 | `DEATHS_STARVED` | 18 | `CELL_OVERFLOW` (hash cells with count > C) |
| 6 | `DEATHS_KILLED` | 19 | `MAX_OCCUPANCY` (hash cell) |
| 7 | `DEATHS_AGED` | 20 | `V_CLIPS` |
| 8 | `DEATHS_UNLOGGED` | 21 | `KIN_ATTACK_SUM` (Σ round(256·sim) over attackers) |
| 9 | `INSOLVENT` (≤ `DEATHS_STARVED`) | 22 | `KIN_ATTACK_BASE` (Σ round(256·kin_mean) over attackers) |
| 10 | `ATTACKS` (after the S9-partner gate) | 23 | `KIN_BOND_SUM` (Σ round(256·sim(i, bond target)) over `as_prop` rows) |
| 11 | `ATTACK_HITS` | 24 | `KIN_BOND_BASE` (Σ round(256·kin_mean) over `as_prop` rows) |
| 12 | `BONDS_FORMED` (pairs) | 25 | `BOND_PROPOSERS` (`as_prop` rows; a mutual pair counts 2) |
| | | 26 | `DEATHS_CATASTROPHE` (director catastrophe deaths, §12.9; 0 when the director is off) |

`DEATHS_STARVED + DEATHS_KILLED + DEATHS_AGED + DEATHS_CATASTROPHE` equals the number of deaths in the step.

All four kin counters and `BOND_PROPOSERS` are assembled by `step.finish` (§16.3).

`BONDS_BROKEN` is the decrease in occupied slots across S9a plus across S14, halved. The runner keeps the per-step
`StepOut`s of an interval as small evaluated arrays and stacks them on the host at the flush. That gives counters
`(T,27)` and ledgers `(T,16)` as int64 sums, so no step's events are lost.

### 19.2 Interval row (`metrics.jsonl`)

Every row has these groups:
- **Header:** `step`, `steps` (steps in this interval; short after an unscheduled stop or a resume, §18), `wall_s`,
  `steps_per_s`, `peak_mem_gb`.
- **Population (interval sums of event counters; the levels `n_alive` and `bonds_total` are interval means, and
  `mean_bonds_per_agent`, `sharers_frac` and `emitters_frac` are weighted by `N_ALIVE`):** `n_alive`, `n_alive_min`
  (the lowest per-step population of the interval), `births`,
  `birth_requests`, `births_blocked`, `immigrants`, `deaths_starved`, `deaths_killed`, `deaths_aged`,
  `deaths_catastrophe` (always present, 0 with the director off), `deaths_unlogged`, `insolvent`.
- **Interactions:**
  - `attacks`, `attack_hits`;
  - `bonds_formed`, `bonds_broken`, `child_bonds`, `bonds_total`, `mean_bonds_per_agent`;
  - `sharers_frac`, `emitters_frac`;
  - `predation_ratio` = Σ`FLOW_STOLEN` / Σ`FLOW_HARVEST`.
- **Kin:**
  - `kin_attack_bias = (KIN_ATTACK_SUM − KIN_ATTACK_BASE) / (256·ATTACKS)`. Negative means kin are spared.
  - `kin_bond_bias = (KIN_BOND_SUM − KIN_BOND_BASE) / (256·BOND_PROPOSERS)`. Positive means kin are preferred
    as partners. Both biases are null when their denominator is 0.
- **Energy:**
  - `E_agents`, `E_structure`, `E_field`, `E_total`, all host int64;
  - `ledger` (interval sums by `LedgerIdx` name);
  - **`ledger_error`** (must be 0, §10.8);
  - `energy_mean`, `energy_median`.
- **Life history:** `age_mean`, `generation_max`, `generation_mean`.
- **Lineage** (int scatter of alive into `clade` bins, size `n_init + 1`): `clades_alive`, `clade_entropy`,
  `clade_simpson` (= 1 − Σp²), `top_clade_share`.
- **Genome:**
  - `tag_var` (trace of the tag covariance over alive agents);
  - `sigma_q10/q50/q90`, `rate_q10/q50/q90`, `frac_q10/q50/q90`;
  - `brain_cost_mean`.
- **Fields:** `resource_mean`, `chem_mean` (list of 3).
- **Environment** (`env`, only when `director.enabled`): the live levers at the snapshot (`regen_mult`, `season_amp`,
  `season_period`, `drift_mult`, `imm_request`, `cat_radius`, `cat_kill_p`, and the range and zero-count of
  `patch_gain`). The row is written before the director acts at that step, so `env` shows what the interval ran with.
- **Health:**
  - `diag`;
  - `cell_overflow` (interval sum), `max_occupancy` (interval max), `v_clips`;
  - `carrying_capacity_est` (first row only).

**Heavy metrics** are added every `heavy_metrics_every` steps:
- **Bodies.** Connected components over bonds on the GPU:
  - Labels start as `lab = arange(N)`.
  - `components_iters` rounds of: `m = min_b where(valid, lab[partner], N)`; **root hooking**, which lowers
    `lab[lab[i]]` to `m[i]` (a scatter-min, done as `ops.scatter_max_i32` of `N − 1 − m`); `lab = minimum(lab, m)`;
    then `lab = lab[lab]` (pointer jumping). `valid` also requires both ends alive. Without hooking, a chain laid
    over random rows needs O(length) rounds; with it a shuffled 2 000-chain converges in about 10.
  - `converged` means one extra round changes nothing. At convergence `lab` is the smallest row of the component.
  - Sizes come from `scatter_add_i32(N, lab, 1, alive)`; `bodies_hist` counts components (alive singletons in
    bin 1).

  Reported values:
  - `bodies_n` (size ≥ 2), `bodies_largest`, `bodies_mean_size` (agent-weighted, size ≥ 2);
  - `bodies_hist` over the bins 1, 2, 3–4, 5–8, 9–16, 17–32, 33–64, 65+;
  - `frac_in_bodies_ge4`;
  - `bodies_converged`.

  If `bodies_converged` is false, the body counts are upper bounds.
- **`weight_diversity`.** The sample is up to `diversity_sample` alive agents with `birth_step < step − 1` (so the
  rows are flushed; at `step = 0` every founder is flushed, as in §12.6), chosen as the first eligible rows of
  `permutation(N, key=observer_key(seed, step))`. The metric is the mean pairwise L2 distance between
  their flattened `[W1, W2]`, cast to f32 on device (so bf16 works, F17), centred on the sample mean, with
  exact elementwise squared norms and the cross-term Gram matmul on the CPU stream (F20).

`snapshot_metrics` performs **one** host transfer of a dict of small arrays. It never touches `state.key`, and the
observer-purity test enforces this.

### 19.3 Offline analysis (deferred tooling, complete data)

The data for these analyses is already recorded: births and deaths with uids, plus checkpoints. The scripts are
deferred, except the first and the last.
- Bedau–Packard evolutionary activity: `alife activity RUN` (`alife.analysis`, method and results in
  `docs/ACTIVITY.md`). It rebuilds the phylogeny from the event logs, builds a neutral shadow (Bedau) and a
  permuted heritability null, computes the activity statistics on two lineage component definitions (cohort
  subtrees and paraphyletic marker components), normalises them (shadow difference; Channon's component
  normalisation against a reset shadow) and classifies the activity as none, bounded or unbounded per window,
  with every threshold swept. Plots need the optional extra `analysis` (matplotlib). It reads forks: the founders
  are the agents alive at the run's first step (`meta.json` `start_step`, read from the fork's checkpoint at that
  step), or at `--start STEP` for a run from step 0; founders may be parents from step 0 (their ages are not
  recorded), and step indices count from that start while the stored grids are run steps.
- The director evaluation (`docs/DIRECTOR.md` §9): `python -m alife.analysis.evaluation` (results in
  `docs/EVALUATION.md`). It recomputes the director's vital signs (`director.vitals.Tracker`) from every arm's
  `metrics.jsonl`, the control included, and checks them against the vitals the live directors logged; then
  coverage, novelty, persistent novelty, turnover, viability, clades, lineages, interventions, the LLM record and
  the activity classes per arm and seed.
- MODES-style change, novelty, complexity and ecology.
- Phylogeny shape.
- Division-of-labour indices.
- Signal–behaviour mutual information: `python -m alife.analysis.signals` (method and results in
  `docs/SIGNALS.md`). It recomputes each agent's `emit` deposit quanta and `msg` by S0–S8 of the step as a pure
  function of a state, and estimates their MI with the agent's resource, harvest, energy, crowding and kin counts
  and its next-step thrust, turn and attack (KSG on tie-broken ranks, `alife.analysis.mi`), against shuffle
  nulls, also restricted to emitters and conditioned on patch, clade and crowding. States a run no longer has
  are re-simulated in memory by `alife.analysis.resim`, which checks every `metrics.jsonl` row on the way
  exactly and writes them outside the run directory.

---

## 20. Rendering and viewing on macOS

### 20.1 Headless rasteriser (`render.py`, pure MLX, runs in CI)

`render_frame(s, cfg, size=(W,H), viewport, layers, color) -> (H,W,3) uint8`. The steps are:

1. **Viewport.** `scale = zoom · min(W,H) / L` pixels per unit, centred on `(cx, cy)`. The default is the whole
   world (`cx = cy = L/2`, `zoom = 1`). World coordinates are taken modulo L, so panning across the seam wraps.
2. **Background** (`resource`, `chem` layers). Each pixel maps to a field cell (one gather).
   - Base colour: `(0, 30 + 150·min(R/resource_cap, 1), 0)`.
   - Chemicals add tints: c0 red, c1 blue, c2 violet, each `90·c/(c+1)`.
3. **Bonds** (`bonds` layer). For each `(i, b)` with `partner > i`, so each bond is drawn once, 8 points are sampled
   along the minimum-image segment and splatted with priority 1, grey.
4. **Agents** (`agents` layer).
   - Each agent stamps a disc of pixel radius `ρ = max(1, round(r·scale))`. The stencil offsets are a static Python
     constant for the given ρ.
   - Priority is `128 + floor(127·min(E/min_energy, 1))`.
   - Colour by `color`:
     - `tag`: `64 + 191·(tag[0:3] + 1)/2`;
     - `energy`: a ramp;
     - `clade`: `(clade·2654435761) mod 2²⁴`.
   - Agents with `last_loss > 0` are drawn red.
   - A one-pixel heading nose at `pos + 0.7·r·fwd` uses priority 255.
5. **Compositing.** Each sample packs to `(prio << 24) | (R << 16) | (G << 8) | B` as uint32. They are combined
   with `zeros(H·W + 1, uint32).at[pix].maximum(packed)`, where off-viewport samples go to the extra sink pixel.
   Integer max is order-independent, so frames are deterministic and golden-image tests are possible. Pixels with
   priority 0 show the background.
6. `np.asarray` happens once per frame.

The emergence prototype measured 16 ms for 64k agents into 512² on CPU.

Pinned details (impl, WP6):
- **Screen orientation and periodicity.** Pixel column grows with world `x` and pixel row with world `y`, like
  `resource[y, x]`. An agent's centre pixel is `floor(W/2 + min_image(x − cx)·scale)`. Stencil offsets are added
  in pixel space, and each sample is wrapped into `[−P/2, P/2)` around the frame centre, with `P = L·scale`, then
  drawn at every periodic image that falls inside the frame. So a disc or bond on the seam shows on both sides,
  `zoom < 1` tiles the torus, and the long axis of a non-square frame tiles too. Background pixel centres map
  back to the wrapped field cell.
- **Unspecified colours.** Bond points are `t = (k+1)/9`, `k = 0..7`, which excludes the endpoints. The violet
  tint is `(0.6, 0, 1)·t2`. Without the `resource` layer the base is black. The `energy` ramp is piecewise linear
  over `log1p(E)/log1p(e_max)`, with stops blue `(30, 60, 200)`, green `(60, 200, 120)` and yellow
  `(255, 220, 40)`. The nose is white. Float colours are rounded half-up.
- **Viewport culling.** Before each splat, rows that are invalid (dead, or a bond not drawn) or whose stencil
  box misses the frame in every periodic image are dropped with one host read, and only the rest are gathered
  and splatted. The test is conservative (one pixel of slack per side), so frames are unchanged; the cost
  follows the visible rows instead of `N × πρ²`, which grows with zoom² (dev at zoom 64: 36 s → 0.04 s per
  900² frame on CPU).
- **Memory bound.** A splat whose sample count still exceeds 2²³ (many visible rows at deep zoom, or far
  zoom-out) is split into row chunks, with an `mx.eval` between chunks. Integer max makes the frame identical.
  `render_frame` is an eager observer: it is not compiled.
- **`VideoWriter` paths.** A path with no suffix is a PNG-sequence directory, with no warning (`alife render
  --out dir/`). A video suffix (`.mp4`, `.mov`, `.mkv` or `.m4v`) falls back to `<path minus suffix>/` with a
  warning. Any other suffix is a `ValueError`. Frames are named `frame_000000.png`, …. The encoder pads odd sizes
  to even ones for `yuv420p`, and ffmpeg failures raise `RuntimeError` with ffmpeg's stderr.

`write_png(path, img)` uses the standard library only: `zlib`, `struct` and the CRC. There is no Pillow dependency.

`VideoWriter` pipes raw RGB24 frames to `ffmpeg -f rawvideo … -c:v libx264 -pix_fmt yuv420p -crf 18`.
- The binary comes from `imageio_ffmpeg.get_ffmpeg_exe()` (extra `video`, which bundles a static ffmpeg, so
  Homebrew is not needed) or from `ffmpeg` on `PATH`.
- Without either, it falls back to a PNG sequence in a directory, with a warning.

### 20.2 Live viewer (`viewer.py`, optional extra `viewer = ["pygame-ce>=2.5"]`)

**Why pygame-ce.** The frame is already an RGB array composited on the GPU by `render_frame`. The viewer only needs
a window, a fast blit (`pygame.surfarray.blit_array`), keyboard and mouse events, and text. pygame-ce provides
exactly that:
- arm64 macOS wheels;
- SDL2 with Metal-backed presentation and no OpenGL;
- a minimal main-thread loop, which macOS requires.

**Rejected alternatives:**
- vispy, moderngl, pyglet: they depend on OpenGL, which is deprecated on macOS, with no MLX↔GL zero-copy benefit.
- Qt, pyqtgraph: a heavy dependency and licensing surface.
- matplotlib: too slow for 30 fps at 1 MP.
- DearPyGui: a viable second choice if richer widgets are ever needed.

**Loop.** Each frame runs up to `steps_per_frame` compiled steps (sync eval), then one `render_frame`, then one
blit, so there is one host read per frame. A frame stops stepping early (after at least one step) once
`FRAME_BUDGET_S = 0.1` s of wall time is spent, so a large `steps_per_frame` on a big preset cannot stall event
handling for seconds (macOS would mark the window unresponsive); `.` and explicit advances ignore the budget.
The viewer holds only the current state, which respects the donation rule. Viewing never changes the trajectory.

**Controls:**
- Space: pause. `.`: single step. `+`/`−`: steps per frame.
- Drag: pan. Wheel: zoom.
- `1`–`4`: toggle the agents, bonds, resource and chem layers. `c`: cycle the colour mode.
- Click: select the nearest agent (MLX `argmin` over minimum-image distances). The HUD then shows uid, energy, age,
  generation, clade and bonds.
- `s`: write a standalone snapshot `media/snap_<step>.safetensors` (in the attached run directory, else the current
  directory) with `runio.write_state_file`. It never touches `checkpoints/`, `LATEST`, `metrics.jsonl` or `events/`,
  so run-directory consistency (§18) is preserved; `alife run --init-from FILE` can fork from it. `p`: save a PNG.
- The viewer never appends to `metrics.jsonl` or `events/`.
- The HUD line shows step, population, births/deaths per interval, bonds, steps/s.

Pinned details (impl, WP6):
- `+`/`−` double/halve steps per frame within `[1, 1024]` (`=` and the keypad keys also work). `.` pauses and
  steps once. Esc or `q` quits. The wheel zooms by 1.25 per notch about the cursor, within `[0.25, 256]`. A press
  that moves more than 4 px is a drag, otherwise a click.
- The HUD interval is `run.metrics_every` steps: births/deaths are summed on the device and shown for the last
  completed interval (the running one until the first completes). A selected agent that dies is reported as
  dead; error diag bits are shown in red, and the viewer keeps running.
- Without a run directory, `s` writes `./media/snap_<step>.safetensors`. `p` writes `media/view_<step>.png`
  next to the snapshots.
- The loop's logic lives in `ViewerSession`, which has no pygame dependency and is tested headless; the
  pygame loop is tested with `SDL_VIDEODRIVER=dummy` and skipped without pygame-ce.

---

## 21. CLI (`alife`, argparse, entry point `alife.cli:main`)

```text
alife run      [--preset tiny|dev|64k|256k] [--overlay fast ...] [--config PATH] [--set section.key=value ...]
               [--seed S] [--steps N] [--out DIR] [--overwrite] [--init-from RUN[:STEP]|FILE.safetensors]
               [--metrics-every N] [--checkpoint-every N] [--checks cheap|off] [--no-compile] [--quiet]
               [--director off|live|replay] [--replay-log LOG]
               [--llm | --llm-every N] [--llm-url URL] [--llm-model NAME] [--persona NAME] [--goal TEXT]
               [--yoke-log PATH]
alife resume   RUN [--steps N | --until STEP] [--checkpoint STEP|latest] [--set run.key=value ...]
               [--set director.key=value ...] [--director off|live|replay] [--quiet]
               [--llm | --llm-every N] [--llm-url URL] [--llm-model NAME] [--persona NAME] [--goal TEXT]
               [--yoke-log PATH]
               (only observer/cadence run.* keys, run.director and the director's policy keys, §18; dynamics
               changes go through run --init-from)
alife render   RUN [--checkpoint STEP|latest] [--steps N] [--every K] [--out PATH(.png|.mp4|dir)]
               [--size WxH] [--fps 30] [--layers agents,bonds,resource,chem] [--color tag|energy|clade]
               [--view CX,CY,ZOOM]
alife inspect  RUN [--checkpoint STEP|latest] [--json] [--check] [--agent UID]
alife view     [RUN | --preset P [--set ...]] [--checkpoint STEP|latest] [--steps-per-frame N] [--size WxH]
alife bench    [--preset P] [--overlay fast] [--set ...] [--steps 200] [--warmup 20] [--stages]
               [--determinism] [--brain-variants] [--json PATH]
alife llm-check [--llm-url URL] [--llm-model NAME] [--timeout SECONDS] [--preset P]
alife activity RUN [--out DIR] [--grid N] [--marker-rates R,...] [--horizons H,...] [--shadows N] [--seed S]
               [--windows A:B,...] [--start STEP] [--reuse] [--no-plots] [--quiet]
```

**`activity`** never modifies the run's files: it writes `activity.npz`, `summary.json`, `classification.txt` and the
figures to `--out` (default `RUN/activity`), §19.3 and `docs/ACTIVITY.md`. `--start` (default: the run's first step,
the fork step for a fork) analyses from that step, with the agents alive then as founders.

**`run`** creates the run directory and the initial state (or loads `--init-from`), then calls `runner.run`. Every
metrics interval it prints one status line: step, population, births/deaths, bonds, largest body (heavy steps),
steps/s, `ledger OK`.

**LLM and persona flags** (`run`, including a fork, and `resume`; §26.10) are shortcuts for `--set` items, like
`--director`, and take precedence over `--set`: `--llm` is `director.llm_every=25000`, `--llm-every N` is
`director.llm_every=N` (`--llm` and `--llm-every` exclude each other), `--llm-url` / `--llm-model` are `run.llm_url` /
`run.llm_model`, `--persona` / `--goal` / `--yoke-log` are `director.persona` / `director.goal` / `director.yoke_log`
(§26.11). `--llm` or a positive `--llm-every`
needs the effective mode to be `live` (`--director live`, or a resume of a live run): in any other mode the command
exits 2 before anything is written, since a replay never calls the model.

**`llm-check`** (director build step 4, Review-log row I31) tests the LLM tier's endpoint without a run. It builds the
director's own `llm.Client` (§26.10) from `--llm-url` / `--llm-model` (default `run.llm_url` / `run.llm_model`), the
request timeout `--timeout` (default 60 s, any positive finite number) and the other `director.llm_*` defaults, and sends
ONE chat request with the real `response_format` schema (generated from `--preset P`'s config, default the engine's
default preset) and a two-line prompt that asks for a single `none` action and the journal `ok`. It prints one line to
stdout and exits:
- `OK model=NAME latency=S.SSs` and exit 0 when the reply is a valid reply per `llm.parse_reply`;
- `FAIL <kind>: <detail>` and exit 1 (`ExitCode.ERROR`) otherwise, where `<kind>` is the client's (`connection`,
  `timeout`, `http`, `protocol`, `oversize`, `empty`) or `invalid` (not JSON, or a schema violation);
- a malformed URL (not http/https), an empty model or a non-positive timeout is a usage or config error (exit 2) with
  nothing on stdout.
It never reads or writes a run directory and never starts a simulation. A server that substitutes another model for
an unknown name (LM Studio does) still answers OK: the check shows reachability and schema validity, not the model name.

**`render`** re-simulates deterministically from the checkpoint in memory. It never alters the run, apart from
writing into `RUN/media/` when `--out` is omitted.
- `--steps 0` produces a single PNG.
- Otherwise it produces a video (or PNG sequence) with one frame every K steps.

**`inspect`** prints:
- a config summary and the carrying-capacity estimate;
- the checkpoint list;
- population, energy and ledger status;
- a body and clade summary;
- the last metrics row.

Options:
- `--check` runs `invariants.full_check` and exits 3 on any violation.
- `--agent UID` prints that agent's row, its bonds and its ancestor chain from `events/`.

**`bench`** prints and writes JSON:
- compile time, ms/step, steps/s, peak memory;
- the **donation-health** verdict.

Options:
- `--stages` compiles and evaluates each sub-step group separately to attribute time. The total is slightly
  inflated by the extra syncs. The groups follow §15.2: S0, S1–S2, S3–S5, S6, S7, S8–S14, S15, S16, S17, S18.
  Their wiring mirrors `step.step` exactly (a test checks bitwise equality in eager mode), and S0 still updates
  the weights in place.
- `--determinism` runs 2 × `--steps` steps (default 200) and compares SHA-256 digests.
- `--brain-variants` times `brain.forward` over all N rows for `impl ∈ {matmul, einsum}` ×
  `weight_dtype ∈ {float16, bfloat16, float32}`, reporting ms and weight GB/s. A variant the backend rejects
  is reported with its error rather than aborting the bench.
- The throughput run also reports the final `diag` flags and an `invariants.full_check` of the final state,
  so day-one backend gaps (e.g. int64 on Metal, §23.3) show up in the bench output.

**Exit codes** (`runner.ExitCode`):

| code | meaning |
|---|---|
| 0 | OK, including a clean extinction with `status = extinct` |
| 1 | unexpected error; also `llm-check` when the check fails (`FAIL ...`) |
| 2 | config error |
| 3 | invariant violation: diag bits are printed by name and the state is dumped to `diag/` |
| 4 | checkpoint incompatible |
| 130 | interrupted by SIGINT after a clean flush and checkpoint |

**Director flags** (§26). `--director live|replay` is a shortcut for `--set run.director=MODE` and also sets
`director.enabled=true` on `run`; `--director off` sets only the mode. `--replay-log` goes with `--director replay`
and `--init-from` (a replay needs a state to continue); other combinations, an unreadable or invalid log, and
`--director live|replay` on `resume` of a run without `director.enabled`, are config errors (exit 2) reported before
anything is written. `view` and `render` re-simulate without the director (§26.8). The persona and goal are set with
`--set director.persona=...` / `director.goal=...` in this step.

Pinned details (impl, WP5):
- `--seed`, `--steps`, `--metrics-every`, `--checkpoint-every`, `--checks` and `--no-compile` are `run.*` overrides
  applied after `--set`. `run.steps` counts steps from the directory's start (`meta.start_step`), so a fork runs
  `run.steps` steps after the fork step. `resume` also takes `--quiet`.
- `render`, `inspect` and `view` accept a standalone state file (a viewer snapshot) in place of `RUN`; its embedded
  config is used, and default outputs go to `./media/`. `render` names them `frame_<step:010d>.png` (`--steps 0`)
  and `render_<s0:010d>_<s1:010d>.mp4`; a `--steps > 0` sequence holds the start frame plus one per `K` steps.
- `inspect --json` prints one object: `step`, `config` (summary and `config_hash`), `carrying_capacity_est`,
  `ledger_error`, `diag` and `diag_flags`, `state` (heavy `snapshot_metrics`), `checkpoints`, `last_metrics`, and
  `check` / `agent` when asked. An unknown `--agent` uid exits 1. The ancestor chain follows `events/` births up
  to a founder or immigrant (`ancestors_complete`).
- Usage errors exit 2 (argparse); a missing run directory or an existing `--out` exits 1; `bench --determinism`
  exits 1 when the two digests differ. The CLI hands its state to `runner.run`/`run_viewer` without keeping a
  reference (the donation rule, §15.4).

---

## 22. Testing strategy (Linux CI, `mlx[cpu]`, pytest)

### 22.0 Setup

- **Fixtures** (`tests/conftest.py`):
  - `tiny_cfg` is the `tiny` preset with `brain.weight_dtype = "float32"`. It is the default for whole-step tests,
    because CPU fp16 matmul is ≈ 3× slower than f32 (F19).
  - `tiny_cfg_f16` is the unmodified `tiny` preset (fp16, the production dtype). It is used in targeted tests:
    `test_brain`, the mutation tests, one whole-step invariant stress variant, and the observer-purity run.
  - `tiny_cfg_bf16` is the same with `bfloat16`, used only by a digest / `full_check` / metrics smoke test (F17).
  - `micro_cfg` is hand-built for unit scenes: `n_max 64`, `M 8`, `n_init 8`, `world.size 48`, which gives
    `G_h = 8` and `FG = 12`.
  - Scenes are built with `make_scene(cfg, agents=[...])`, a test helper that writes explicit positions,
    energies, bonds and weights into a fresh `init_state`.
- **References** are plain numpy or Python loops, and are compared exactly.
- **Markers:**
  - `slow`: nightly only.
  - `golden`: runs only if the MLX version matches.
- **Budget:** the non-slow suite finishes in < 3 min on a CI runner. The tiny preset (f32, M = 32) is estimated at
  ≈ 20–30 ms/step on a 4-core CPU runner, assembled from measured sub-steps (F19: neighbours 5–6 ms, brain 3 ms,
  mutation ≈ 6 ms at M = 32, flush ≈ 2 ms, plus ≈ 250 small kernels). **Measured (WP4):** ≈ 30 ms/step for the
  compiled `tiny_cfg` step on a shared 4-core x86 container (MLX 0.32.3, CPU), and ≈ 0.3 s/step for `dev`. The
  WP4 fast suite (≈ 1 100 tiny steps with a host `full_check` after 200 + 7 × 60 of them) takes ≈ 60 s, so the
  step counts below stand.
- **`golden`** also pins the platform (`system-machine-device`): CPU kernels may round differently across
  architectures, so the digest is only compared where it was recorded.
- **CI** (`.github/workflows/ci.yml`, ubuntu-latest): `astral-sh/setup-uv`,
  `uv sync --locked --group dev --extra viewer --extra video` (so the pygame loop and ffmpeg tests run rather than
  skip), `uv run ruff check`, `uv run ruff format --check`, `uv run pytest -m "not slow"`. A nightly job runs
  `-m slow`. Both jobs set `TMPDIR=$RUNNER_TEMP` and cache `$RUNNER_TEMP/mlx`: MLX's CPU backend JIT-compiles
  every fused kernel with `g++` into `$TMPDIR/mlx`, and a cold cache measured ≈ 280–310 s for the fast suite
  versus ≈ 100 s warm, so the < 3 min budget assumes a warm (cached) JIT directory.
- **Python.** `requires-python = ">=3.12,<3.15"` and a committed `.python-version` (3.12, the minimum supported version, so CI tests the lower bound): mlx 0.32.x has
  wheels only for cp312–cp314, so an unbounded range lets uv pick an interpreter that cannot install it.

### 22.1 Unit tests per module (each owned by the work package that owns the module, §25)

| file | asserts |
|---|---|
| `test_config.py` | Defaults and every preset/overlay validate. Each `validate` rule rejects a crafted bad config, and all problems are reported at once. TOML and JSON load. Precedence: defaults < preset < overlay < file < `--set`. Typed override parsing (int, float, bool, tuple). Unknown key → `ConfigError`. `config_from_dict(config_to_dict(c)) == c`. Hash ignores `run.*` and changes with any dynamics field. `stability_number` value. |
| `test_layout.py` | `D_IN == 69`, `D_OUT == 35`. Input and output blocks are disjoint and tile `[0, D)`. Name tuples have matching lengths. |
| `test_state.py` | `init_state` shapes and dtypes match §2.2 for every preset. Founders are alive in rows `[0, n_init)` and everything else is dead-canonical. `pend_slot` lies in the dead tail. Founder `uid == arange(n_init)`, `next_uid == n_init`, and dead rows hold the §2.2 initial values. `total == total0 == total_energy(...)`. `total_energy` is exact when `n_alive·S > 2³¹` (a 262 144-row all-alive mask). `state_to_arrays`/`state_from_arrays` round-trip, including from a dict with shuffled key order; a missing or extra key is rejected. `state_digest` is stable and independent of dict order. `to_numpy` handles bf16. The 38 v0 field names and their order, and `ENV_FIELDS` appended after them, are frozen by a literal list. |
| `test_rng.py` | `step_keys` is deterministic, and its streams are pairwise distinct. `init_keys(s)` differs across seeds. Layout constants are frozen: a snapshot of `Stream`, `NUM_STREAMS`, `RNG_LAYOUT_VERSION` and `RNG_COMPATIBLE_VERSIONS` is compared, so changing them fails until the version is bumped. |
| `test_ops.py` | `proportional_take`: 500 random instances against the guarantees of §10.4, including demand = 0, supply = 0, a single pool and 2¹⁶ − 1 demands. `exact_sum` matches numpy int64 at max range and length 2²⁰. `rank_select` matches `np.flatnonzero`. `scatter_*_i32` reject float dtypes and handle duplicates. `place` and `set_rows` do not mutate inputs (F2). `first_free` handles full rows. `min_image` is antisymmetric bitwise. `wrap` never returns L (F8). `wrap_angle` range. `int_diffuse` conserves exactly and stays ≥ 0 on random and extreme fields. `mul_q8`/`q8`. |
| `test_spatial.py` | Equals brute-force kNN on random torus scenes when overflow-free: 20% dead, agents within R of edges and corners, empty world, a single agent, K > available. Dead and self are never listed. Invalid entries have `idx == self` and `dist == R`. Rows are sorted by distance. A crowded cell sets `overflow_cells > 0` and still yields a valid subset. Deterministic. |
| `test_fields.py` | Fertility matches a numpy torus-Gaussian reference, and the separable form is exact. `regen_rate` lies in range. Regen respects the cap and `inc` sums to the injected amount. Resource diffusion conserves exactly. Chem decays by `(1−λ)` per step without deposits, and diffusion conserves mass when λ = 0. Deposit sums are exact. A planted ramp gives the right gradient sign and body-frame rotation. Patches drift and wrap. |
| `test_sensing.py` | Hand-built scenes produce the exact expected value at every named index: rotated neighbour offset → `IN_NBR+NBR_DX/DY`; bonded partner → `IN_BOND` block; kin of a clone = 1; resource ramp → `IN_RES_GRAD`. Absent neighbours and empty slots give zeros with `present = 0`. A bond of length 4.5 (stretched past `break_length` by the previous step's physics) gives clipped offsets. Every input is finite and `abs(x) ≤ 1`. |
| `test_brain.py` | `forward` matches a numpy reference (fp32 exact to 1e-5; fp16 to 2e-2), and `matmul` and `einsum` agree. Init statistics: per-tensor std within 5%, the recurrent block scaled, and the clip holds. `mutate_rows`: σ = 0 gives a child equal to its parent; the changed fraction ≈ p (binomial bound); changes are clipped; the fp16 cast happens after the f32 math. `l1_cost` matches numpy over the stored dtype. |
| `test_actions.py` | Every decode range and threshold at extreme logits. The target is the argmax over eligible neighbours only, and ties go to the lowest k. With nothing eligible, the target is −1 and `attack = False`. Bonded partners are excluded from attack unless `attack_bonded`. `mask` zeros everything. |
| `test_bonds.py` | Handshake: mutual case; acceptance case; a winning proposer never accepts; ≤ 1 new bond per agent; consent (an unwilling target never bonds); no self or duplicate bonds; full rows block. `as_proposer` is correct. A 300-round randomized fuzz with breaks, forms, deaths and revivals preserves the §8.1 invariant after every phase (port of F9). Breaking: a flag on one side clears both sides. `drop_dead`. `add_child_bonds` is symmetric with `rev` correct. `spring_forces` gives `F_i + F_j == 0` **bitwise** for an isolated pair and 0 at rest length. `symmetric_ok` fires on single-entry corruption of `partner` or `rev`. |
| `test_physics.py` | An overlapping pair separates to ≥ 2r. A bonded pair relaxes to the rest length. Contraction without grip: centre-of-mass displacement is < 1e-3 u over 500 steps. It is analytically 0, and the tolerance covers fp32 rounding of positions. With phase-locked grip it is > 1 u (F11). The speed clip is counted. `wrap` keeps `0 ≤ pos < L`. A hexagonal cluster at default stiffness stays finite and does not gain energy. A config with `stability_number > 1.8` is rejected. Thrust attenuation: factor 1, 0.75, …, 0 for 0..4 contacts. |
| `test_crowd.py` (WP4: it integrates `spatial` and `physics`) | **Converging crowd:** 300 agents at full thrust toward one point for 600 steps, using the real `find_neighbors` (C = 16, K = 8), `physics.forces` and `physics.integrate`, with and without grip: `max_occupancy ≤ C` and `overflow_cells == 0` at every step, and median nearest-neighbour spacing ≥ 1.6 at the end (F18). With `thrust_contact_limit` set very large (attenuation effectively off) the same scene overflows, which proves the test is not vacuous. |
| `test_energy.py` | Each function conserves its own sub-ledger exactly. Costs: `paid ≤ E`, insolvency drives E to 0 and cancels actions. Harvest: proportional, exact, demand never negative even at `E = e_max`. Combat: a mutual attack, three attackers on a poor victim, defend reduces demand, spill lands in the victim's cell, `E ≥ 0`. Share: a chain A→B→C is exact with 10% heat, and Σ out ≤ E. Share gating: a slot broken and refilled with a new partner in the same step sends nothing, a newly formed bond in a previously empty slot sends nothing, and an unchanged bond sends. Combat gating: an attack on a target that became a partner at S9 is dropped (no grant, no `attack` flag), and kept when `attack_bonded`. Cap overflow → heat. |
| `test_lifecycle.py` | `deaths`: causes and their precedence; an insolvent agent that received bond inflow (E > 0) still dies as `STARVED` with corpse = inflow + S; a solvent victim drained to 0 and rescued by inflow survives; corpse = E + S into the correct cell; the hard cap; dead rows canonicalised; bonds dropped on both sides; records compacted; unlogged counter. `allocate_lanes` (≥ 300 property trials): unique dead slots; unique eligible parents; the birth prefix; capacity pressure; zero requests; full population → 0 births; padding lanes target distinct dead rows and use parent row 0 regardless of request counts; χ² fairness across seeds. `reproduce`: endowment arithmetic, the parent keeps ≥ 1, uids are `next_uid + k`, lineage fields, placement distance = 2.0, gene bounds. **Child-row completeness:** fill a dead row's every per-agent field with sentinel garbage, force a birth into it, run one more step (the S0 flush), and assert that no sentinel survives in any field, `W1`/`W2` rows included. Immigration: lanes, clade `n_init`, ledger. Catastrophe: `deaths(cat_mask=)` kills only agents that would otherwise survive, every other cause keeps precedence, a masked dead row stays unchanged, an all-False mask changes nothing. |
| `test_director_env.py` | `State.env` view and `default_env`; `legacy_state_digest` is exactly the v0 encoding. Director enabled with inert env equals director off (`legacy_state_digest`, counters, ledgers; 60 compiled steps; plain and seasons+immigration); director off passes `env_*` through. Each knob: `regen_mult` (0 / 0.5 / 2 / 3.7 against a numpy floor, whole-step `INJECT_REGEN`), `patch_gain` (0 removes a lone patch; weights match a float64 reference), `drift_mult` (0 freezes; ×5 displacement), the season override (period 0 → weights exactly 1; sine reference; config seasons + inert env bit-identical to director off). `imm_request`: admitted over several steps with the remainder carried, `INJECT_IMMIGRATION` matching, capped by capacity with the rest left pending, `max` with the periodic rule. Catastrophe: deaths only inside the radius (torus minimum image), `kill_p` 1 and 0.5, cause `CATASTROPHE`, counter, records, `kill_p` reset, radius 0 / `kill_p` 0 inert. Ledger closes under scripted catastrophes, requests and lever changes (`full_check` every step, deterministic digest), compiled vs eager write-back, checkpoint round trip with non-default env, pre-director (v1, 38 arrays) checkpoint loads inert and resumes (also as a run directory), a missing non-env key and an RNG version outside {1, 2} are refused, out-of-range env flagged by `full_check` and (enabled) `FIELD_RANGE`, config hash rules, stream 8 and the frozen keys of streams 0–7. |
| `test_director_*.py` (vitals, archive, actions, rules, log, replay, config) | Build step 2, see §26.9. |

### 22.2 Whole-step invariant tests (`test_step_invariants.py`)

`tiny_cfg` runs compiled for 200 steps. The `slow` (nightly) suite runs it for 300 steps and repeats the whole
file, stress variants included, under `tiny_cfg_f16`, so the production dtype gets full invariant coverage nightly. After **every** step,
`invariants.full_check(s, cfg, prev, out)` must return `[]`, and `s.diag & ERROR_MASK == 0`. The harness also keeps
the cumulative set of uids seen in death records and asserts none of them is alive (`full_check` itself is
stateless). `full_check` verifies:
1. **Ledger.** The per-step closure (§10.8) holds exactly, recomputed in numpy int64 from
   `s.ledger_cum − prev.ledger_cum` and `s.total − prev.total`, and so does the global closure. `ledger_error == 0`.
2. **Bond symmetry** (§8.1). No duplicates, no bonds on dead rows, `n_bonds ≤ B`.
3. **No resurrection.**
   - Every row alive at `t+1` but not at `t`, or whose uid changed, has `uid ≥ next_uid(t)` (from `prev`).
   - `next_uid` advances by exactly `births + immigrants` (`out.births.valid.sum()`).
   - Alive uids are unique.
   - Together with uid monotonicity these imply that a uid from a death record never reappears; the harness
     checks that directly as well.
4. **Capacity.**
   - `n_alive ≤ N_CAP`.
   - Births ≤ `min(free, M)`.
   - Newborn rows (`out.births.slot` where valid) were not alive with the same uid in `prev`, and every alive row
     whose uid is not in `prev` is one of them.
5. **Dead rows are canonical** (§2.2), and **alive rows** satisfy `1 ≤ E ≤ e_max`.
6. **Ranges.**
   - `0 ≤ pos < L`, finite.
   - `‖vel‖ ≤ v_max·(1 + 1e-6)` (the §2.2 tolerance for fp32 rescaling).
   - `heading ∈ [−π, π)`.
   - `h ∈ [−1, 1]`.
   - Genes within bounds; `tag ∈ [−1, 1]`.
   - `0 ≤ R ≤ R_HARD`; `chem ∈ [0, chem_max]`.
7. **Pending buffer.** `pend_slot` entries are distinct, and (for `step ≥ 1`, §12.6) every alive row with
   `birth_step == step − 1` is in `pend_slot`.
8. **Age.** `birth_step ≤ step − 1` for alive rows.

**Stress variants** (each 60 steps in the fast suite and 150 steps under `slow`, on `tiny_cfg`):
- forced high attack logits;
- every agent reproducing (capacity pressure);
- a dense cluster start (the cell-overflow path);
- immigration on;
- `max_age = 30` (age deaths);
- `fast` overlay;
- `tiny_cfg_f16` (the production dtype).

**Mutation testing of the checks.** Every invariant is deliberately corrupted in a copy of a healthy state, and
the test asserts that the checks report it. This proves they are not vacuous.
- **Both** `device_check` and `full_check` must report the corruptions that have a Diag bit (§22.3): one asymmetric
  `rev`, a dead row with energy, a dead row with a uid or a bond, a leaked quantum in `resource`, a NaN position,
  `pos ≥ L`, a negative resource, and `n_alive > N_CAP`.
- **Only** `full_check` must report the rest, which no cheap device reduction can see: a duplicate alive uid, a gene,
  `h`, `heading` or `tag` out of range, `chem > chem_max`, a broken pending-buffer invariant, and a dead row with
  non-canonical `vel`, `h`, `msg` or `last_*`.
- A corrupted state is fed to `device_check` with `prev_total` = the healthy state's `total` and a zero ledger, so
  `LEDGER` fires exactly when the corruption changes `TOTAL`. The test asserts that the expected bit is set (other
  bits may also fire).

### 22.3 Runtime checks (`invariants.device_check`) and `Diag`

These run inside the compiled step when `runtime_checks == "cheap"` (the default). They cost O(N·B) gathers plus
≈ 15 reductions, about 2–3% of a step. The result is ORed into the sticky `state.diag`. The runner reads it at each
metrics interval. On any error bit it dumps the state to `diag/`, prints the decoded flags, and exits 3.

| bit | name | condition |
|---|---|---|
| 1 | `LEDGER` | per-step closure fails, or `STOLEN ≠ GAIN + SPILL` |
| 2 | `ENERGY_RANGE` | alive with `E < 1` or `E > e_max`, or dead with `E ≠ 0` |
| 4 | `DEAD_DIRTY` | dead row with `uid ≠ −1`, or any bond |
| 8 | `BOND_ASYM` | `bonds.symmetric_ok` is false |
| 16 | `NONFINITE` | non-finite `pos`, `vel`, `h` or `chem` |
| 32 | `FIELD_RANGE` | `R < 0` or `R > R_HARD`; with `director.enabled`, also any `env_*` value non-finite or outside its §2.2 range (end-of-step state in `device_check`, start-of-step state in `step.finish`) |
| 64 | `INT_HEADROOM` | `max_cell_occ ≥ Dims.occ_limit` (1 060 at defaults, §10.9) or `max_attackers ≥ 2¹⁵`: a real int32 precondition, unreachable with contact-attenuated thrust (F18) |
| 128 | `CAPACITY` | `n_alive > N_CAP` |
| 256 | `POS_RANGE` | `pos < 0` or `pos ≥ L` |
| 1<<16 | `CELL_OVERFLOW_WARN` | warning: `overflow_cells > 0` |
| 1<<17 | `VCLIP_WARN` | warning: `v_clips > 0` |

`ERROR_MASK = 0xFFFF`. Warning bits never stop a run, and they are reported in metrics.

**Coverage.** The Diag bits cover exactly these §2.2/§8.1/§10 invariants: ledger closure and the theft identity
(`LEDGER`); alive energy range and dead energy 0 (`ENERGY_RANGE`); dead `uid = −1` and dead rows without bonds
(`DEAD_DIRTY`); bond symmetry, no duplicates, `partner < 0 ⇔ rev < 0` (`BOND_ASYM`); finiteness of `pos`, `vel`, `h`
and `chem` (`NONFINITE`); resource range (`FIELD_RANGE`); int32 preconditions (`INT_HEADROOM`); the population cap
(`CAPACITY`); and position range (`POS_RANGE`). Everything else in §22.2 (uid uniqueness and monotonicity, gene,
`h`, `heading`, `tag` and `chem` ranges, the pending-buffer invariant, the remaining dead-row canonical fields, age)
is checked only by the host `full_check`.

### 22.4 Determinism tests

**`test_determinism.py`** (WP4):
- The same seed run twice (50 compiled steps each) gives identical `state_digest`. Different seeds give different
  digests.
- **`legacy_state_digest`** (§14.2) is the golden's digest. Director-enabled-inert vs director-off bit-identity is in
  `test_director_env.py`.
- **Compiled vs eager** (marker `slow`). Population and ledger trajectories match for 30 steps. Bitwise equality is
  not required, but it was observed on CPU.
- **`test_golden`** (marker `golden`): the `tiny_cfg` digest after 50 steps equals the pinned value.

**`test_resume_purity.py`** (WP5):
- **Resume.** 50 steps → `RunDir.save_checkpoint` → `load_checkpoint` → 50 steps equals 100 uninterrupted steps
  bitwise. This is also exercised through the CLI. Resume rejects `--set run.compile=…`, `run.runtime_checks`,
  `run.seed` and dynamics keys (exit 2).
- **Observer purity** (`tiny_cfg_f16`). A run with `metrics_every = 1`, `heavy_metrics_every = 1`, a `render_frame`
  every step and a checkpoint every 10 steps has the same digest as a bare run.
- **Version guards.** Loading a checkpoint whose `RNG_LAYOUT_VERSION` is outside `RNG_COMPATIBLE_VERSIONS` or whose
  `GENOME_LAYOUT_VERSION` was altered fails with `CheckpointError`.
- **Fork semantics.** A fork keeps `step`, `next_uid` and uids, resets `diag` and `ledger_cum`, reseeds `key` only
  with `--seed`, refuses a changed `n_init` and a state that fails `full_check` under the new config (exit 4).

### 22.5 End-to-end and ecology

**`test_cli_smoke.py`**, through `cli.main(argv)` without subprocesses:
1. `run --preset tiny --steps 200 --out tmp` produces `metrics.jsonl` rows, checkpoints, `LATEST` and event files.
2. `inspect --check --json` exits 0 and reports `ledger_error == 0`.
3. `resume --steps 50`, then truncation is verified.
4. `render --steps 0 --out f.png` writes a valid PNG.
5. `render --steps 20 --every 5 --out frames/` writes a PNG sequence (ffmpeg is not needed).
6. `run --init-from tmp --set combat.attack_bonded=true --out fork` starts a fork.
7. `bench --preset tiny --steps 5 --warmup 2` writes JSON.

**`test_affordances.py`** runs `tiny` for 400 steps with the pinned seed. Every affordance must be exercised: the
interval totals of births, `DEATHS_STARVED + DEATHS_KILLED`, attacks with hits, bonds formed, bonds broken, child
bonds, sharers, emitters and harvest are all > 0, and the population is > 0 at the end.
- If a parameter change makes an affordance unreachable, this test fails.
- The test builds founders with a fixed bias row from a test-only helper, so every action fires.
- This affects the test only, not the engine defaults (founders are fully random).

**`test_ecology.py`** (marker `slow`) runs the `dev` preset (float32 weights, as `tiny_cfg`, F19) with default random
founders for 3 000 steps. The
population must be neither extinct nor pinned at `N_CAP` from step 0. Births > 0, bonds > 0, and at least one
attack lands. This test is the tuning harness for the defaults in §17.

### 22.6 Static and lint tests (`test_lint.py`, AST-based)

1. `.at[` appears only in `ops.py` and `render.py`. In `ops.py`, every helper asserts an integer dtype.
2. Subscript assignment to an array (`x[...] = ...`) appears only in `ops.set_rows`.
3. Every `mx.random.<fn>(` call passes `key=`, and `mx.random.seed` is never called.
4. `mx.async_eval` is never used in `src/alife`.
5. There is no `.item()`, `np.asarray` or `float(`/`int(` on arrays inside the simulation modules (`spatial`,
   `fields`, `sensing`, `brain`, `actions`, `bonds`, `physics`, `energy`, `lifecycle`, `step`, `invariants`).
   This is a heuristic AST check plus an allow-list.

### 22.7 Performance bench (not gating)

`alife bench` (§21) is the instrument for §15. On the Mac, its JSON is committed to `docs/bench/<machine>.json`. A
v0 acceptance checklist item is: `bench --preset 64k --determinism` passes, donation health is OK, and the bench
reports ≥ 30 steps/s. CI runs only the tiny smoke invocation in `test_cli_smoke.py`.

---

## 23. Scope split, risks and mitigations

### 23.1 In v0 (the scaffold must run end to end)

- **World and physics:** a torus with continuous particles; overdamped physics with thrust, turn, springs,
  `contract`, `grip` and soft repulsion; fixed-K hash-grid neighbours.
- **Agents:** Elman RNN brains with per-agent fp16 weights, evaluated with batched matmul.
- **Senses:** self, fields, 3 nearest neighbours, pooled neighbourhood and 4 bond slots.
- **Signals:** 2 broadcast channels and 3 chemical fields with distinct length scales.
- **Bonds:** a consent handshake with brain-chosen partners, per-slot release and share, child bonds, and
  OR-symmetric breaking with stored back-pointers.
- **Ecology:**
  - harvest with exact proportional contention;
  - combat with brain-chosen targets, defend and field spill;
  - division with heritable offspring fraction, self-adaptive heritable σ and p, and kin tags;
  - death with structural corpses;
  - drifting fertility patches.
- **Bookkeeping:** the exact integer ledger, the device diag bitmask and the host full check; deterministic RNG with
  versioned streams; deferred birth flush; optional immigration.
- **Runs:** checkpoints, resume and forks; metrics including bodies, lineages, genome diversity and kin bias; event
  logs.
- **Visuals:** the MLX rasteriser, PNG and video output, and the pygame-ce viewer.
- **Tooling:** the CLI (`run`, `resume`, `render`, `inspect`, `view`, `bench`) and the full CPU test suite with CI.

### 23.2 Explicitly deferred

1. **Custom Metal kernels.** A fused neighbour search with top-K, a tiled GEMV and a splat renderer, all behind a
   capability check with automatic fallback to the pure-MLX path and an equality test against it.
2. **Locality and compaction.** Periodic re-sorting of agent rows by cell, with a `partner`/`rev` remap through the
   inverse permutation, and compacting live rows for the brain. A row-span neighbour scan that shares capacity across
   adjacent cells.
3. **Richer genomes.** Variable-topology or indirect genomes (NEAT, hypernetworks), knockout and node-reset
   mutations, sexual recombination, and horizontal transfer along bonds. Heritable tag rate (mimicry). Variable
   body size and morphology.
4. **Richer environment.** Evolvable or neural rules for the field CA (a true NCA field). Seasons, catastrophes and
   scheduled interventions. Obstacles. Bilinear field sampling. Anisotropic drag and swimming.
5. **Richer analysis.** Division-of-labour index, signal mutual information, trophic EMA, offline phylogeny and
   MODES tooling, and a sweep command. (Sequential sweeps can already be scripted around `alife run`.)
6. **Viewing.** A web viewer, and 3-D.
7. **Multi-device and distributed** operation. Nothing in v0 paints this into a corner:
   - All agent rules are local: K neighbours, own bonds and back-pointers, the own field cell.
   - Global couplings are isolated in four places: the neighbour sort, `allocate_lanes` / uid issuing, the
     handshake priorities, and metrics.
   - Integer ledgers sum exactly across devices.
   - Fields are a lattice, ready for halo exchange.
   - uids can later be `local_counter·n_dev + dev_id`.
   - The step is a pure function of state.

### 23.3 Key risks and mitigations

| risk | impact | mitigation |
|---|---|---|
| Weight donation behaves differently on Metal than on CPU (F1) | +1 full weight copy per step (+3–5 ms at 64k, +12–20 ms at 256k); 2× peak memory | S0 deferred flush plus sync eval; `bench` donation-health check. Fallback: S0 as a separate compiled call with one extra sync. |
| Batched tiny GEMV is inefficient on Metal | the brain is 2–3× slower than the bandwidth bound | `bench --brain-variants` (matmul vs einsum, three dtypes); `fast` overlay (H = 24); a Metal GEMV kernel later |
| ArgPartition is a full sort on Metal; neighbour traffic is large | the neighbour stage dominates | Budgeted as a sort in §15; `fast` overlay (C = 12, K = 6); fused kNN kernel deferred |
| int64 support gaps on Metal | compile or runtime errors | int64 is used only for elementwise ops (uid, totals, ledger) and gathers. No int64 reduction (3-limb `exact_sum`), no int64 scatter (`place`), no int64 sort. Detected on day one by `bench`. |
| Ecology collapses or saturates | nothing to observe | Carrying-capacity estimate; `test_affordances` and the slow ecology test; optional immigration floor; forks for parameter rescue; all rates in config |
| Few adult bonds | weak multicellularity | The propose/accept handshake, which forms bonds far more readily than mutual-only. Child bonds as the guaranteed route. `BONDS_FORMED` metrics. |
| Cell overflow in dense bodies or converging crowds | missed contacts, which reinforce themselves (invisible agents do not repel) | Contact-attenuated thrust bounds crowd pressure (§9, F18); C sized from contracted bodies and measured crowds (§4); `test_crowd.py`; counter plus warning bit; `cell_capacity` knob |
| fp16 mutation quantisation | small mutations of large weights are lost | Documented; `weight_dtype = float32` option; weights clipped to ±4 |
| CPU ≠ GPU numerics | CI cannot reproduce Mac trajectories | Determinism is promised per backend; tests use invariants and statistics; `bench --determinism` on the Mac |
| MLX API drift | breakage on upgrade | Pin `mlx>=0.32.3,<0.33`; the golden test flags behaviour changes; lint tests guard the risky APIs |
| Observer code accidentally retains old states | a silent 2× slowdown | Donation rule (§15.4); bench health check; viewer and runner code review checklist |

---

## 24. Decision log

| # | decision | alternatives considered | why |
|---|---|---|---|
| D1 | Hybrid world: continuous particles on a torus plus lattice fields | pure lattice NCA (Coralai); pure particles with no fields | Bonds, forces and collisions need geometry. Fields give the CA substrate and exact per-cell bookkeeping. All three proposals agreed. |
| D2 | **One compiled step; child weights buffered in `pend_*` and flushed at S0 of the next step; one sync `mx.eval` per step** (from throughput) | Same-graph scatter (correctness: measured full copy per step, fatal); two compiled phases with an eval boundary (emergence: in place but a second function and a mid-step sync); `async_eval` pipelining | Measured in place in one graph (F1). Fewest moving parts. `async_eval` defeats donation, so it is not offered. |
| D3 | **Capacity reserve `N_CAP = N − M`** (from emergence) | permanent SINK row (throughput: broke the "dead uid = −1" invariant, fatal); zero dummy row (correctness: extra row in every weight tensor) | Every lane targets a distinct dead row, so there is no duplicate-index write anywhere and no special rows. It costs 1.6% of capacity. |
| D4 | **Newborn small arrays written by gather (`ops.place`) through int32 scatter-max maps** (from correctness) | M-lane set-scatters into every array | Only int32 scatters exist. int64 `uid` needs no int64 scatter, which is a Metal risk. The cost is ≈ 25 MB/step at 64k. |
| D5 | **Stored `rev` back-pointers; OR-symmetric per-slot breaking** (from correctness) | recompute `rev` each step (emergence: silent wrong gather if symmetry ever breaks); no `rev` with release-all (throughput) | Every bond flow is a gather, symmetry is cheap to check on device, and per-slot release and share are possible. |
| D6 | **Propose/accept/mutual handshake with permutation priorities** (emergence + correctness) | mutual-only nearest (correctness); mutual-only query argmax (throughput: rare adult bonds); random-bit priorities (index bias) | Consent is kept and adult bonds form far more readily. Unique priorities with no index bias. Fuzz-verified (F9). |
| D7 | **Attention-style targeting** `argmax_k q·φ_k` for bonds and attacks (throughput/emergence) | nearest eligible (correctness) | Partner choice, kin bias and prey choice become heritable continuous traits. |
| D8 | **Overdamped physics plus a `grip` output (per-agent drag)** (new) | inertial semi-implicit Euler (throughput, correctness); overdamped with 2 substeps (emergence) | All three proposals wrongly claimed contraction could propel bodies (fatal). Grip makes inchworm locomotion real (F11). One substep keeps the kernel count down, and stability is proven by Gershgorin (§9). |
| D9 | **`proportional_take`** (f32 ratio, 2¹⁵ scale, −1 guard) for harvest and combat (from emergence, with demand caps) | integer product `(d·min(A,D))//D` (throughput: headroom rule evaluated to 2³⁴ on its own defaults, fatal); equal-split `fair_share` (correctness: wastes the pool when requests differ) | Overflow-free for any realistic contention, proportional, exact, and property-tested (F10). |
| D10 | **Energy cap step S13 with excess → heat; demand clipped ≥ 0** (from correctness) | no cap (emergence: E above E_max made harvest demand negative and overflowed int32, fatal) | Fixes the flaw. Keeps `E ≤ e_max < 2²⁰` as a hard bound for every product. |
| D11 | **Structural energy S paid at birth and returned as the corpse; combat spill into the field** (from emergence) | recycle a fraction of metabolism (throughput); a death threshold `e_death` (correctness) | Corpses become food for starved and killed agents too. It creates scavenging niches, and the ledger stays a simple identity. |
| D12 | **Solvency gating**: insolvent agents pay everything, all their actions are cancelled, and they die at S14 regardless of bond inflow (`~solvent` is in the death rule; inflow joins the corpse) | let unpaid actions execute (correctness); let bond partners rescue insolvent agents | Removes the "free attack while dying" artefact. One rule, and easy to test. Rescue stays possible for solvent combat victims (§10.7). |
| D13 | **16-entry named ledger; per-step closure checked on device; global closure checked on host; 3-limb `exact_sum`** (correctness + throughput) | host-only aggregate (throughput); int64 device reductions with a fallback (correctness) | Leaks are attributed to a step immediately. No int64 reductions on Metal, and a single code path. |
| D14 | **Chemical fields as float32 with integer-quantised deposits** (from throughput) | int32 fixed-point signals (correctness: its headroom bound was wrong, and 2²⁸·32 wraps, fatal) | Chemicals are not energy. Deposits are deterministic integers and stencils are elementwise, so the field is still deterministic. |
| D15 | **Frozen 16-stream RNG with `RNG_LAYOUT_VERSION`; permutations for priorities** (from correctness) | a free-form key-splitting tree | Split is not prefix-stable (F3), and randint/uniform duplicate at scale (F4). |
| D16 | **fp16 weights by default on the Mac; f32 mutation math; CI's default fixture uses f32, with targeted fp16 tests** | fp16 in every CI test (CPU fp16 matmul is ≈ 3× slower, F19, which breaks the CI budget); fp16 never in CI (throughput) | Halves the dominant traffic on the GPU. CI still exercises the production dtype in the brain, mutation, a whole-step stress variant and the observer-purity run. |
| D17 | **Neighbour hash cell = R = 6 (3 diameters), C = 16, K = 8, gathering from the cell-sorted copy** (from throughput, re-sized) | R = 8 with C = 16 (throughput: undersized, since hexagonal packing gives 18.5 per cell); C = 12 (correctness) and C = 8 (emergence), both undersized for contracted bodies | C is sized from contracted-body packing (12.8 per cell) and from measured converging crowds under contact-attenuated thrust (≤ 15 per cell, F18). Q = 144 is the price; the `fast` overlay trades it down. |
| D18 | **Structural dims (B, NS, MSG, NCH, T) fixed in `layout.py`**, H/K/C from config | a config-derived `Layout` object (correctness) | Index constants are importable and test-pinned. Changing the genome layout is a deliberate, versioned code change. Amended for 3D (§27.1): the module constants stay the 2D layout, and `layout.for_dims(nd)` returns a frozen per-dims `Layout` (the 2D vectors plus an appended 3D tail) keyed only by the static `world.dims`, with its own `GENOME_LAYOUT_VERSION_3D`; no other config key shapes it. |
| D19 | **Heritable σ, p and offspring_frac; neutral fixed-rate tags** (from emergence) | heritable σ only (throughput, correctness); brain-output child share (correctness) | Evolvability and life history evolve. Tags stay an honest kin cue. |
| D20 | **Drifting fertility patches** (from emergence), default 0.005 u/step | static patches | A non-stationary environment for open-endedness, at negligible cost (separable Gaussians). |
| D21 | **`State` as a `NamedTuple`** | a frozen dataclass plus dict conversion | A native `mx.compile` pytree (F12), `_replace`, and no conversion layer. |
| D22 | **pygame-ce viewer (optional extra); stdlib PNG; ffmpeg via the `imageio-ffmpeg` extra** | vispy/moderngl (OpenGL is deprecated on macOS), Qt, matplotlib, Pillow | The GPU already composites the frame, so the viewer only blits. No mandatory heavy dependencies. |
| D23 | **TOML or JSON in, canonical JSON out (stdlib only)** | TOML out via `tomli-w` | No extra dependency, and an exact round trip. |
| D24 | **Immigration optional, static-gated, sharing the birth lanes** | always on; reseeding on extinction | Default runs are "start random, no intervention". The anti-extinction tool costs nothing when off. |
| D25 | **Deferred from v0:** sweeps, DoL/MI metrics, knockout and node-reset mutations, trophic EMA, Metal kernels | including them (emergence) | Keeps v0 lean. The data for the deferred analyses is already logged. |
| D26 | **Contact-attenuated thrust**: `thrust × max(0, 1 − n_contacts/4)` (review) | rely on C sizing alone (fails: crowds collapse, F18); a pressure term from exact per-cell counts (only partly effective in the critic's test); stiffer repulsion (capped by the stability bound `σ ≤ 1.8`) | Converts thrust from a body force into a surface force, so crowd pressure is bounded regardless of crowd size. Uses contacts the physics step already computes. Measured effective. |
| D27 | **Resume changes only observer/cadence `run.*` keys; all dynamics changes go through a fork** (review) | `resume --allow-config-change` with `config.<step>.json` and a `replayable` flag | One mechanism instead of two. A run directory always holds one config, and its trajectory is always replayable. |
| D28 | **Decisions are applied only to the bonds they were made for**: share needs an unchanged partner; attacks on a fresh partner are dropped (review) | apply decisions to whatever occupies the slot after S9 | A brain never sends energy to, or attacks, a partner it did not sense. Two cheap (N,B) compares. |

**Fatal flaws from the judges, and their fixes.**
- **Throughput:**
  - The headroom rule evaluated to 2³⁴ (D9, §10.9).
  - The SINK row gave `uid = 0` (D3, D4).
  - The age definition contradicted itself (age is now derived from `birth_step`, §2.2).
  - Bodies could not swim by contraction (D8).
  - C was undersized (D17).
- **Correctness:**
  - The same-graph weight scatter and `async_eval` forced weight copies (D2).
  - The signal field overflowed int32 (D14).
  - Contraction was claimed to propel bodies (D8).
  - C = 12 was undersized (D17).
- **Emergence:**
  - E_max was not enforced, so harvest demand could go negative and overflow (D10).
  - Contraction was claimed to propel bodies (D8).
  - C = 8 was undersized (D17).

---

## 25. Implementation work packages

There are seven packages. **WP0 lands first.** WP1, WP2 and WP3 then proceed in parallel against the WP0 skeleton.
WP4 integrates them. WP5 and WP6 can start on WP0 and finish on WP4. Each package owns its files exclusively.
Shared types and signatures are frozen by §16. Changing one requires editing this document first.

**Milestones:**
- **M1:** WP0 merged.
- **M2:** the WP1, WP2 and WP3 unit tests are green.
- **M3:** WP4 passes the whole-step invariants, determinism and `test_affordances`.
- **M4:** WP5 and WP6 pass the CLI smoke test.
- **M5 (on the Mac):** `bench --preset 64k --determinism` passes, donation health is OK, and the bench reports
  ≥ 30 steps/s.

### WP0: Foundations and skeleton (blocks everything)

**Files:**
- `pyproject.toml`, `README.md`, `.github/workflows/ci.yml`
- `src/alife/__init__.py`, `config.py`, `presets/*.toml`, `layout.py`, `state.py`, `rng.py`, `ops.py`
- `tests/conftest.py`, `tests/helpers.py` (`make_scene`)

**Skeleton.** WP0 also creates **every other module file** with the exact NamedTuples, enums, constants and
function signatures of §16. Each function body is `raise NotImplementedError`. That way every package imports
stable names from day one.

**Hand-over.** WP0 implements nine small functions inside files owned later by other packages: five that
`init_state` needs, and four that WP4's unit work needs, so that WP4 does not wait on WP2 or WP3:
- `brain.init_weights`, `brain.l1_cost` and `brain.mutate_rows` (WP2 then owns the rest of `brain.py`);
- `fields.init_patches`, `fields.fertility` and `fields.regen_rate` (WP1 then owns the rest of `fields.py`);
- `bonds.drop_dead`, `bonds.add_child_bonds` and `bonds.symmetric_ok` (WP3 then owns the rest of `bonds.py`).

Each is under ≈ 15 lines. WP0 writes their unit tests into the owning module's test file (`test_brain`,
`test_fields`, `test_bonds`), which the owning package then takes over. The owning package may refactor them but not
change their signatures or semantics.

**Implements:**
- all of §2 and §17 (config with dims, presets, validate, hash, load and save);
- `layout.py` with every index from §5 and §7;
- `state.py` (`init_state`, `total_energy`, `age`, digest, (de)serialisation, enums `LedgerIdx`, `CounterIdx`,
  `Diag`, `DeathCause`);
- `rng.py`;
- every `ops.py` helper.

**Tests:** `test_config`, `test_layout`, `test_state`, `test_rng`, `test_ops`, `test_lint` (§22.6).

`pyproject.toml` essentials:

```toml
[project]
name = "alife"
requires-python = ">=3.12,<3.15"
dependencies = [
  "mlx>=0.32.3,<0.33; sys_platform == 'darwin'",
  "mlx[cpu]>=0.32.3,<0.33; sys_platform == 'linux'",
  "numpy>=2.0",
]
[project.optional-dependencies]
viewer = ["pygame-ce>=2.5"]
video  = ["imageio-ffmpeg>=0.5"]
[dependency-groups]
dev = ["pytest>=8", "ruff>=0.6"]
[project.scripts]
alife = "alife.cli:main"
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.pytest.ini_options]
markers = ["slow: long-running (nightly)", "golden: MLX-version-pinned trajectory digest"]
[tool.ruff]
line-length = 110
```

### WP1: Space and fields

**Files:** `spatial.py`, `fields.py` (everything except the WP0 hand-over functions).

**Depends on:** WP0 (`ops`, `config`, `State`).

**Implements:** §4 (`find_neighbors`) and §13 (`cell_index`, `sample_fields`, `deposit_chem`, `update_chem`,
`update_resource`, `drift_patches`).

**Tests:** `test_spatial`, `test_fields`.

### WP2: Mind (sense, think, decide)

**Files:** `sensing.py`, `brain.py` (`forward`; plus ownership of the WP0 hand-over functions), `actions.py`.

**Depends on:** WP0. It uses the `Neighbors`, `FieldObs` and `BondGeom` types from the skeleton. Tests build
these NamedTuples directly, so the package does not wait for WP1 or WP3.

**Implements:** §5, §6 and §7, including target selection and `mask`.

**Tests:** `test_sensing`, `test_brain`, `test_actions`.

### WP3: Interactions (bonds, physics, energy)

**Files:** `bonds.py`, `physics.py`, `energy.py`.

**Depends on:** WP0. It uses the `Actions` and `Neighbors` types from the skeleton, which tests construct directly.

**Implements:** §8 (`bond_geometry`, `n_bonds`, `break_bonds`, `form_bonds`, `spring_forces`; `drop_dead`,
`add_child_bonds` and `symmetric_ok` come from the WP0 hand-over), §9 (`forces` including contact-attenuated thrust,
`integrate`), and §10.3–§10.6 plus §11 (`pay_costs`, `harvest`, `combat` including the S9-partner gate, `share`
including the `same` gate, `cap_energy`).

**Tests:** `test_bonds` (including the F9 fuzz port), `test_physics` (including the grip locomotion test),
`test_energy`.

### WP4: Lifecycle, step and invariants (integration)

**Files:** `lifecycle.py`, `step.py`, `invariants.py`.

**Depends on:**
- WP0 for the unit work: `deaths`, `allocate_lanes`, `reproduce`, `flush_pending`, `device_check`, `full_check`.
  Their cross-package callees (`bonds.drop_dead`, `bonds.add_child_bonds`, `bonds.symmetric_ok`,
  `brain.mutate_rows`) are WP0 hand-over functions, so this work starts at M1.
- WP1, WP2 and WP3 for `step` and `test_crowd`.

**Implements:** §3, §12, §16.3 (including `finish`), §22.2 and §22.3.

**Tests:** `test_lifecycle`, `test_crowd`, `test_step_invariants` (including the stress variants and the mutation
testing of the checks), `test_determinism` (same seed, different seed, compiled vs eager, golden),
`test_affordances`, `test_ecology` (slow).

WP4 also owns **tuning the §17 defaults** until `test_affordances` and the slow ecology test pass. Any default it
changes must be reflected in this document.

### WP5: Runs, metrics and CLI

**Files:** `metrics.py`, `runio.py`, `runner.py`, `cli.py`.

**Depends on:**
- WP0 for `runio`, and for `metrics` on hand-built states.
- WP4 for `runner` and `cli`. Until then, these can be developed against a stub step that only advances `step` and
  `key`.

**Implements:** §18, §19 and §21, except the `view` and `bench` command bodies, which call WP6.

**Tests:**
- `test_metrics`: components vs union-find, including a 2 000-chain and a non-converged flag; clade statistics;
  kin-bias arithmetic; `interval_row` aggregation; carrying capacity.
- `test_runio`: checkpoint round-trip of every dtype and metadata; atomic write with a simulated failure;
  `LATEST`; retention; `truncate_after`; version and hash refusal; `dump_diag`.
- `test_cli_smoke`.
- `test_resume_purity` (resume, run-key restrictions, observer purity, version guards, fork semantics; §22.4).

### WP6: Rendering, viewer and bench

**Files:** `render.py`, `viewer.py`, `bench.py`.

**Depends on:** WP0 for `render` (it needs only `State` from `init_state`), WP4 for `viewer` and `bench`.

**Implements:** §20, plus the bench of §15.4 and §21 (stages, donation health, determinism, brain variants).

**Tests:**
- `test_render`: shape and dtype; a single agent lights the expected pixels; a bond across the seam is drawn on both
  sides; viewport culling; frames are deterministic; a valid PNG header and CRC (decoded with `zlib`);
  `VideoWriter` falls back to a PNG sequence when ffmpeg is absent.
- `test_bench`: a tiny smoke run of every part (JSON-compatible report), the staged wiring equal to `step`
  bitwise in eager mode, and the donation check flagging a step whose caller retains the old state; plus the
  bench invocation in `test_cli_smoke`.
- `test_viewer`: `ViewerSession` headless (trajectory unchanged by viewing, keys, pan/zoom, selection across
  the seam, HUD, snapshots that leave the run directory's checkpoints and logs alone). The viewer import is
  guarded, and the pygame loop test (`SDL_VIDEODRIVER=dummy`) is skipped when pygame-ce is absent.

### Cross-package rules

1. Only `ops.py` (and `render.py` for uint32 compositing) may scatter or setitem. A package that needs a new
   primitive asks WP0's owner to add it to `ops.py`, with a test.
2. A NamedTuple field or signature in §16 changes only after this document is edited in the same PR.
3. Each package's tests must pass with `pytest -m "not slow"` in under 60 s on CI, except WP4's, which have
   120 s (its fast suite is ≈ 1 100 compiled tiny steps: 200 invariant steps, 7 × 60 stress steps, 400
   affordance steps, ≈ 150 determinism steps, plus `test_crowd` and ≈ 10 compiles). The whole non-slow suite stays
   under 3 min. If the measured tiny step time (§22.0) breaks this, steps move to `slow` before budgets grow.
4. No package adds a runtime dependency beyond `mlx` and `numpy`. Optional extras are listed in WP0.

---

## 26. Director: vital signs, rule tier, decision log, replay (build step 2) and the LLM tier (build step 3)

`docs/DIRECTOR.md` is the design rationale; this section is normative for what build steps 2 and 3 implement. §26.1 to
§26.9 describe the director core of step 2, which is the whole behaviour when `director.llm_every = 0`; §26.10 is the
LLM tier of step 3 and says exactly what it changes. There is no GUI panel (step 4) and no evaluation harness
(step 5). The compiled step is unchanged from build step 1 (§2.2, §12.9, §13.1): the director is host code in the
`alife.director` package that runs **between** compiled steps.

### 26.1 Modules and the runner hook

| module | content |
|---|---|
| `director/__init__.py` | `Director` (the controller the runner calls), `make_director(rd, cfg, start_step)`, `describe(entry)` |
| `director/vitals.py` | `Window` / `aggregate` (rule-window aggregates of metrics rows), `descriptor`, `Tracker` (window buffer, boundary history, archive, streaks), `Vitals` |
| `director/archive.py` | `Archive`: novelty search and the coverage grid |
| `director/actions.py` | `MENU`, `EnvView`, `clamp` (live), `validate_entry` (replay), `apply`, `lever_keys` |
| `director/pacing.py` | `fold(entries, t)`: budget, cooldowns, active effects, drama, as pure folds of the log |
| `director/rules.py` | `RuleParams` / `PersonaParams`, `RULES_VERSION`, `decide(ctx, params)` (and, for step 3, `decide_safety`, `decide_discretionary`, `eligibility`, `cooldown_steps`), `rng_for` |
| `director/log.py` | `director.jsonl` header, `validate_log`, `read_log` |
| `director/llm.py` | step 3: the reply schema generated from `MENU`, `parse_reply`, the stdlib `Client` (§26.10) |
| `director/prompt.py` | step 3: the system and user messages of a call (§26.10) |
| `director/yoked.py` | the `random` persona (§26.11): the yoke log (`yoked_steps`, `inherited`, `load`), the eligible actions, `sample_params`, the per-draw generator |

**Hook** (`runner.run(..., director=None)`). `director=None` is today's behaviour exactly. With a director, at every
metrics row, after the row is appended and after the violation check (so never for a violated state):
1. `director.observe(row)`;
2. if `director.due(t)` (`t % director.rule_every == 0`) **and the run is not extinct at this step**:
   `state = director.on_boundary(state, row, t)`. In live mode this computes the decision from the row and the
   vital signs, appends the entries to `director.jsonl` (fsync), and returns the state with them applied; in replay
   mode it applies the log's entries for `t` and decides nothing;
3. then the checkpoint write (if due), followed by `director.on_checkpoint(t)` (live: write `director_archive.json`).

The checkpoint at `t` therefore already contains the applied environment, and a resume never re-applies anything.
A SIGINT or the end of the run at a rule boundary runs the hook like any other. `Director.status_lines()` gives the
console lines (`director @ <t>: <description>`) printed after the status line unless `--quiet`.

### 26.2 Modes, config and the hash

`run.director` is `off` (default), `live` or `replay`. It is a `run.*` key: not hashed, stored in `config.json`,
inherited by resume, and changeable on resume (it is in `cli.RESUME_KEYS`). `live` and `replay` need
`director.enabled` (`validate`). `off` builds no `Director` at all: with `director.enabled = true` the environment
levers keep whatever values they hold, nothing is decided or reverted.

`DirectorConfig` gains four **policy** keys, read only by the host-side director:

| key | default | meaning |
|---|---|---|
| `director.persona` | `gardener` | `gardener`, `trickster`, `architect` or `random` (§26.6, §26.11) |
| `director.goal` | `""` | the Architect's goal text; stored in the log header, and given to the model as data when the LLM tier is on (§26.10) |
| `director.rule_every` | 1000 | steps between rule-tier decisions |
| `director.budget_per_100k` | 0 | cost units per rolling 100k steps; 0 = the persona's default; bounded by the persona's maximum |

`config.UNHASHED_KEYS` lists them: they are omitted from `config_hash` **at every value** (they are also in
`POST_V01_KEYS`, so old `config.json` files without them load). Only `director.enabled` stays hashed. The hashes of
every earlier run are unchanged (`TINY_HASH_V01`; the step-1 `enabled = true` hashes; the `64k` preset's hash,
pinned in `tests/test_director_config.py`). Resume accepts `--set director.<persona|goal|rule_every|budget_per_100k>`
besides the observer/cadence `run.*` keys (§18). Step 3 adds four more policy keys (`llm_every`, `llm_temperature`,
`llm_timeout_s`, `llm_max_tokens`, §26.10), unhashed and resumable in the same way, and the run-level keys
`run.llm_url` and `run.llm_model` (resume keys; not hashed, since no `run.*` key is).

`validate`: `persona` is one of the four (the `random` persona's extra rules are in §26.11); `rule_every ≥ 1`; `budget_per_100k ≥ 0`; `run.director ∈ {off, live,
replay}`; and when the mode is not `off`, `director.enabled` is true and `rule_every` is a multiple of both
`run.metrics_every` and `run.heavy_metrics_every` (so the heavy vital signs exist at every boundary). Resume
overrides of the cadences are revalidated like any other.

### 26.3 Files

```text
runs/<name>/
  director.jsonl           the decision log (the source of truth for replay)
  journal.jsonl            rule-tier one-liners, for people; never read by the simulation
  director_archive.json    the behaviour archive at the last checkpoint ("as_of_step"); analysis only
```

**`director.jsonl`.** One JSON object per line. The first line is a header
`{"kind": "header", "step", "rules_version", "persona", "goal", "rule_every", "budget_per_100k", "seed"}` (`step` is the
step the director started at; `budget_per_100k` is the effective budget). A new header is appended whenever a live
director starts with a policy that differs from the latest header (a resume with another persona, cadence, budget or
`RULES_VERSION`). Every other line is an *entry*:

| field | meaning |
|---|---|
| `step_applied` | the rule boundary the entry was applied at (a positive multiple of `rule_every`) |
| `source` | `rule`, `llm` (step 3: an action the model chose, §26.10) or `revert` (the end of a timed effect) |
| `persona`, `rule`, `reason` | who, which rule (`extinction_rescue`, `runaway_damper`, `monoculture_breaker`, `stagnation_response`, `boredom`; absent on reverts and on `llm` entries) and a one-line why (for `llm`, the model's `expected_effect`, cleaned and at most 240 characters) |
| `llm_call` | `llm` entries only: the id of the call, `llm-<boundary step>`, which names its record in `journal.jsonl` |
| `action`, `params` | a menu action and its **explicit, clamped, absolute** parameters (§26.4) |
| `prev` | the values the action returns to when a timed effect ends (for other actions, the values it replaced) |
| `duration` | timed actions only: steps until the revert, already rounded up to a multiple of `rule_every` |
| `reverts` | reverts only: the `step_applied` of the entry being undone |
| `vitals_step`, `vitals` | the metrics row step the decision used (the boundary) and a compact vitals dict (rounded to 4 places) |

Entries are canonicalised through JSON before they are applied, so what a replay reads back is exactly what the live
run applied. Entries are in non-decreasing `step_applied`; a boundary's reverts come first.

**`journal.jsonl`.** `{"step", "persona", "text"}`: one line per entry, plus one line when a rule newly triggers but is
blocked (a condition that persists is noted once, not at every boundary). With the LLM tier on, each call adds one
`{"kind": "llm_call", ...}` record (§26.10).

**`director_archive.json`.** `Archive.to_json(as_of_step)`: the parameters, `coverage`, every visited grid cell with the
step it was first visited, and every archive entry (step and descriptor). It is rewritten atomically at each
checkpoint of a live run and **never read back**: the archive is rebuilt from `metrics.jsonl` (§26.5).

**Resume truncation.** `RunDir.truncate_after(step, keep_director=False)` also cuts `director.jsonl` and
`journal.jsonl` to the records with `step_applied` (or `step`, for headers and journal lines) `≤ step` and deletes a
`director_archive.json` whose `as_of_step > step`. The files are rewritten whole, so a torn last line disappears. A
replay passes `keep_director=True`: **replay never truncates or writes any director file**. `RunDir` has generic
`append_jsonl` (append + fsync), `read_jsonl` (a torn last line is ignored) and `write_json` (atomic).

### 26.4 The menu: bounds, clamping, validation, application

| action | params (logged) | bounds | cost | cooldown |
|---|---|---|---|---|
| `set_seasons` | `period`, `amp` | period 2k–100k, amp 0–0.9 | 1 | 50k |
| `fertility_shift` | `patch`, and `gain` **or** `x`, `y` | gain 0.25–2.0; a move of at most L/4 (minimum image) from `prev` | 1 | 20k |
| `habitat_loss` | `patch`, `gain` = 0; `duration` | duration 10k–200k | 2 | 50k |
| `regen_pulse` | `mult`; `duration` | mult 0.4–1.8, duration 5k–50k | 1 | 30k |
| `immigration_pulse` | `count` | 16–1024; the pending `env_imm_request` plus the count may not exceed 1024 | 1 | 20k |
| `catastrophe` | `center` [x, y], `radius`, `kill_p` | radius in [20, min(200, L/2)], kill_p 0.2–0.9 | 3 | 100k |
| `drift_change` | `mult` | 0.5–10, permanent until changed | 1 | 50k |

- **Seasons** change `env_season_period` / `env_season_amp` only; the mode stays `fields.season_mode` and is not a
  runtime lever. The mutation-floor nudge is not implemented.
- **Patch moves** are host edits of `patch_pos`, wrapped with `ops.wrap` (so the f32 result is strictly below L); the
  log holds the absolute new coordinates and the old ones in `prev`. The drift continues from the new position.
- **Immigration** is always random founders; it adds to `env_imm_request`, which the step admits over several steps.
- **Costs and cooldowns** are the §5 table of `docs/DIRECTOR.md`. A cooldown counts from the **end** of the effect
  (for an instant action, from `step_applied`).
- **Timed effects** (`habitat_loss`, `regen_pulse`) end with a `revert` entry at `step_applied + duration`, with the
  explicit restore values (`habitat_loss`: the patch's prior gain; `regen_pulse`: the prior multiplier). A new
  effect on a lever that has one active is refused by the rules, except a rescue's `regen_pulse`, which supersedes
  the active pulse and restores the original base value. A **lever** is `regen` or `patch:<p>` (a gain change also
  takes the patch's lever; a move does not).
- **Live: `clamp`** brings a rule's raw request inside the menu bounds *and* the step-1 device ranges
  (`invariants.ENV_*`), canonicalises floats through float32 (so the logged value is the applied value) and rounds a
  duration up to a multiple of `rule_every`. It raises `ActionError` for an unknown action or an unusable request.
- **Replay: `validate_entry`** rejects instead of clamping: unknown action or source, a `step_applied` that is not a
  positive multiple of `rule_every`, a parameter set that is not exactly the action's, any value outside the bounds
  (menu and device ranges), a move of more than L/4 or without `prev`, a duration that is out of bounds or off the
  cadence, a revert of an action without a timed effect. `log.validate_log` adds: records in non-decreasing step, and
  every revert has an active effect on its lever. A bad log is a config error (exit 2) before the first step.
- **`apply(state, entry, cfg)`** edits only `env_*` and (moves) `patch_pos` with `_replace`, converting to each field's
  dtype and shape (no retrace), and fails with `ActionError` if the result would leave `invariants.env_out_of_range`.
  The live director validates every entry it builds before applying it.

### 26.5 Vital signs, the descriptor and the archive

Computed at each rule boundary from the `metrics.jsonl` rows since the previous boundary (`aggregate`): means are
weighted by `row["steps"]` (short rows exist after an interrupted run) and skip None; snapshot values (clade shares,
gene medians, bodies, weight diversity) take the latest non-None value; the wall-clock keys (`wall_s`,
`steps_per_s`, `peak_mem_gb`) are never read. `kill share = Σ deaths_killed / Σ all deaths` (catastrophe deaths
included), `singleton fraction = bodies_hist["1"] / n_alive`. `n_alive_min` is the lowest per-step population of a row
(rows without it fall back to `n_alive`).

- **`K_est`** is recomputed at every live boundary by `metrics.carrying_capacity(state, cfg, state.step, state.env)`:
  the regeneration of the *current* environment (seasonal at the step, scaled by the levers). It is used only at the
  boundary, never stored.
- **Viability:** `n_alive / K_est`; `fall = 1 − (window minimum / highest window mean over the last risk_fall_window)`;
  the **extinction-risk flag** is `n_alive/K_est < risk_k_frac` or `fall > risk_fall`.
- **Descriptor `b`** (ten components, each clipped to [0, 1], a missing one counts as 0, None for an empty world):
  `emitters_frac`, `sharers_frac`, `mean_bonds_per_agent / bonds_per_agent_max`, `frac_in_bodies_ge4`, singleton
  fraction, kill share, `predation_ratio / predation_max`, and the medians of σ, rate and offspring fraction
  log-scaled onto their fixed `RuleParams` ranges.
- **Archive** (`Archive`): the **novelty** of `b` is the mean RMS distance (Euclidean / √10) to its `novelty_k` nearest
  archive members, **excluding the most recent `novelty_exclude` entries**; `b` joins the archive when its novelty
  exceeds `archive_rho` (or when there is nothing to compare to). **Coverage** is the number of cells of a
  `grid_bins`³ grid over `emitters_frac × frac_in_bodies_ge4 × predation_ratio` that any observed descriptor has
  visited; each cell remembers the step it was first visited.
- **Stagnation signals:** `b_shift` (RMS distance between `b` now and `b` about `shift_window` steps ago) and the
  number of new cells in the last `stagnation_window` steps.
- **Rebuild.** The `Tracker` is a deterministic fold of the rows. On start or resume, `Director` replays the on-disk
  rows with `step ≤ start step` through the same `observe` / `close` calls live mode uses, so the rebuilt archive,
  history and streaks equal the live ones (tested); the rows after the last boundary stay buffered for the next
  window. The archive file is never consulted.

### 26.6 The rule tier

`decide(ctx, params)` is a pure function of the vital signs, the pacing folds, the live environment values and the
seed. Rules in priority order; the first that yields an action wins, and one that triggers but is blocked leaves a
journal note:

1. **Extinction rescue:** the extinction-risk flag. A `regen_pulse` (`rescue_regen`) plus an `immigration_pulse` of
   `rescue_imm_frac × K_est` founders. It **overrides cooldown and budget at most once per `rescue_override_gap`**
   (it may also supersede an active regen pulse); inside that gap it needs ordinary eligibility.
2. **Runaway damper:** population `> damper_pop_frac × N_CAP` (`dims(cfg).n_cap`, the hard row capacity, not `K_est`)
   or the largest body `> damper_body_frac` of the population: a `regen_pulse` of `damper_mult` (< 1).
3. **Monoculture breaker:** `top_clade_share > mono_share` for `mono_window × window_scale` steps (the streak runs
   from the start of the first window above the threshold; it re-arms after the rule acts): the persona picks an
   immigration pulse, seasons or habitat loss.
4. **Stagnation response:** over `stagnation_window × window_scale` steps, which the run has covered: no new grid cell
   and mean novelty below `novelty_low` (re-arms one window after acting): the persona picks from the menu.
5. **Boredom** (Trickster only): `idle_gap` steps without any intervention since the last one (or since the first
   window the run directory saw): as the stagnation response.

**Eligibility** of an action: its cooldown (scaled by the persona; **a catastrophe never less than `catastrophe_gap`
= 100k**, whatever the persona, the override included), the rolling budget (`cost` against `budget` over the last
100k steps of `step_applied`), a free lever (one active effect per lever), room in the immigration queue, and for
the discretionary rules (3–5) `min_gap` steps since the last intervention and a decayed cost sum ("drama", half-life
`drama_half_life`) below `drama_max`. Safety rules 1–2 skip the last two checks.

**Personas without an LLM.**
- **Gardener** (default): the deterministic argmax of its weights over the eligible responses; parameters at the
  fixed fraction `param_u` of each range; long windows, long cooldowns, small budget, so it rarely acts.
- **Trickster:** a seeded weighted draw over the eligible responses that **never repeats its last action**, random
  parameters inside the ranges, short windows and cooldowns, a larger budget, and the boredom rule.
- **Architect:** the Gardener's machinery with its own weights and budget; safety rules only. The goal text is
  recorded in the header and not interpreted.

**Randomness** (a patch index, a catastrophe centre, the Trickster's sampling) is stateless host numpy,
`np.random.Generator(PCG64(SeedSequence([run.seed, step, tag])))`, no `mx.random`. The values are logged, so replay
draws nothing. The objective J of `docs/DIRECTOR.md` §6 is not optimised: there is no forward model.

### 26.7 `RuleParams` (rules version 1)

Every threshold, window, weight, range and budget is a field of the frozen `RuleParams` (`RULES_VERSION = 1`,
recorded in the log header; change the version with any value).

| group | values |
|---|---|
| Descriptor | `bonds_per_agent_max` 4.0; `predation_max` 0.25; `sigma_range` (1e-3, 0.5); `rate_range` (5e-4, 0.5); `frac_range` (0.05, 0.95) (log-scaled) |
| Archive | `novelty_k` 5; `novelty_exclude` 3; `archive_rho` 0.025; `grid_bins` 10; `history_keep` 400k steps |
| Viability / rescue | `risk_k_frac` 0.2; `risk_fall` 0.3 within `risk_fall_window` 50k; `rescue_override_gap` 200k; `rescue_regen` (1.5, 20k); `rescue_imm_frac` 0.1 |
| Damper | `damper_pop_frac` 0.95; `damper_body_frac` 0.05; `damper_mult` 0.8; `damper_duration` 10k |
| Monoculture / stagnation | `mono_share` 0.9; `mono_window` 100k; `novelty_low` 0.02; `stagnation_window` 200k; `shift_window` 100k |
| Pacing | `drama_half_life` 50k; `catastrophe_gap` 100k |
| Response ranges | seasons period (10k, 40k), amp (0.3, 0.7); shift gain (0.5, 1.6), move (0.4, 1.0) of L/4; habitat duration (20k, 100k); regen boom (1.2, 1.6) for (10k, 30k); immigrants (1%, 4%) of the population; catastrophe radius (0.1, 0.25) of L, kill_p (0.3, 0.7); drift (2, 6) |

| persona | budget (max) | window_scale | cooldown_scale | min_gap | drama_max | idle_gap | selection |
|---|---|---|---|---|---|---|---|
| gardener | 2 (4) | 1.0 | 1.5 | 50k | 3 | none | argmax, `param_u` 0.3 |
| trickster | 6 (12) | 0.5 | 0.5 | 20k | 8 | 150k | sampled, no repeat |
| architect | 4 (8) | 1.0 | 1.0 | 30k | 5 | none | argmax, `param_u` 0.5 |

Response weights (menu order): monoculture, Gardener / Architect: seasons 2, habitat loss 1, immigration 3;
Trickster: seasons 2, habitat loss 2, immigration 1. Stagnation, Gardener: seasons 2, shift 2, habitat loss 0.5,
regen 1, immigration 3, drift 1; Architect: seasons 2, shift 2, habitat loss 1, regen 1, immigration 2, drift 1;
Trickster: seasons 2, shift 2, habitat loss 2, regen 2, immigration 1.5, catastrophe 1, drift 2. The Gardener and
the Architect never take a catastrophe.

**Calibration.** `archive_rho` and `novelty_low` were set against the `runs/64k_s0` rows (read only), replayed through
the tracker with a 1000-step rule window: the per-window novelty is dominated by sampling noise (median ≈ 0.02), so
`archive_rho` sits just above it (212 archive entries from 1504 boundaries, coverage 32). For a Gardener the
stagnation response is armed from step 1.199M (200k after the cell first visited at 999k) until a new cell appears at
1.279M, and not before 1.1M, which matches the stall seen by eye. The monoculture breaker never arms on that run (top clade share ≤ 0.84), the damper and the rescue never
trigger. These are first values for the evaluation of §9 of `docs/DIRECTOR.md`, not tuned results.

### 26.8 Determinism, replay and what is not guaranteed

- **Replay is bitwise.** A `director.jsonl` replayed with the same seed, config, backend and starting state gives a
  bitwise-identical trajectory (same `state_digest`, same metrics rows apart from the perf keys), in a new
  directory or in place from any checkpoint, and for any `metrics_every` that still visits every log step (the log
  steps are multiples of `rule_every`, which is a multiple of the cadences). Tested for a scripted policy that takes
  every action and reverts across a checkpoint, and for a real-rules run.
- **A resumed live run may decide differently** from an uninterrupted one when the resume interrupted a window (the
  first rows are short, so the aggregates differ). The log is the source of truth. When the checkpoint falls on a rule
  boundary with unbroken rows, the resumed log and archive are byte-identical to the uninterrupted ones (tested).
- **Director off, or enabled with inert levers, is bit-identical to today** (§14.2, `test_director_env.py`).
- **Replay modes.** `resume RUN --director replay` applies the run's own log at the recorded steps from the checkpoint on
  (entries at or before the checkpoint are already in the state). `run --init-from SRC[:STEP] --director replay
  --replay-log LOG` copies the log verbatim into the new run directory and replays it; entries at or before the
  fork's start step are skipped. The environment of a fork is always reset (§18), so a fork replay reproduces the source
  only when it forks from a step before the first log entry (normally `SRC:0`).
- **The viewer and `render` re-simulate without the director.** They keep the checkpoint's environment levers but
  apply no log entry, so after a director action they diverge from the run they were opened on. Replay in the viewer
  is not implemented.
- A live decision depends on `K_est`, which uses a float32 GPU computation; it is deterministic on a given backend
  but not across backends. Replay does not recompute it.

### 26.9 Tests

| file | covers |
|---|---|
| `test_director_vitals.py` | each signal from synthetic rows; None and absent heavy keys; short-row weighting; wall-clock keys ignored; descriptor ranges; the tracker's streaks, risk flag and shift |
| `test_director_archive.py` | cells; novelty with the recent-entries exclusion; ρ membership; coverage and first-visit steps; the JSON; deterministic rebuild from rows equals the live archive (also through `Director`) |
| `test_director_actions.py` | the menu table; bounds and clamps; refusals; duration rounding; the move cap and the f32 wrap guard; strict validation (39 bad entries); each action through the real compiled step with `full_check(prev=edited)`; an unclamped out-of-range write raises `FIELD_RANGE` |
| `test_director_rules.py` | each safety rule; the rescue override once per 200k; one catastrophe per 100k; budget, cooldown-from-end, one-active-effect and drama folds; persona determinism; the Trickster never repeats; every proposal clamps and validates |
| `test_director_log.py` | append and fsync; torn lines; resume truncation; replay never truncates; headers; log validation; the CLI's exit 2 before anything is written |
| `test_director_replay.py` | the bitwise replay tests of §26.8 (fresh directory, in place at three checkpoints, another metrics cadence, a late fork, live resume, real rules) and a slow one with the real timed bounds |
| `test_director_config.py` | unchanged hashes, the unhashed policy keys, validation, the resume key list and its refusals |
| `test_director_random.py` | the `random` persona (§26.11): which of a log's entries are yoked; validation and the exit-2 file checks; the sampler (bounds, reachability, log-uniform shape, determinism, clamp and replay validation); eligibility; a live run yoked to a scripted arm B, its header, its replay (bitwise, never reading the yoke log), live resume, a changed yoke log, nothing eligible, the safety rules; the parameters R inherits (a scripted Trickster yoke log with six moves in one 100k window, budget from the header and caps, a missing budget, a `random` or unknown persona exit 2, mixed headers, bitwise replay) |
| `test_director_llm.py` | step 3 (§26.10): the schema and the reply validator; a mock OpenAI-compatible server for valid, invalid, empty, slow, refused, 500 and redirected replies, clamps, duplicates, `none`, budget and cooldown blocks, the request body and the prompt; roles; no network with the tier off or in a replay; a live run with the mock model and its bitwise replay; an optional real-server test (`@pytest.mark.llm`, needs `ALIFE_LLM_URL`) |
| edits | `test_director_actions.py` and `test_director_log.py` (the "unknown source" example is no longer `llm`), `test_director_config.py` (the LLM keys, the flags; `PERSONAS` gains `random`), `test_runner.py` (hook order; no action on a violated or extinct state; the checkpoint holds the applied env), `test_metrics.py`, `test_cli_smoke.py` (`--director` flows, fork reset), `test_lint.py` (the director package is scanned) |

### 26.10 The LLM tier (build step 3)

A local OpenAI-compatible server (LM Studio) picks the discretionary interventions. Decisions L1 to L11 of the step-3
spec; `docs/DIRECTOR.md` §7 has the rationale. **With `director.llm_every = 0` (the default) nothing in this section
applies: no client exists, nothing touches the network, and the behaviour and the log are exactly those of §26.1 to
§26.9.**

**Config and flags.** `director.llm_every` (0 = off, else a multiple of `rule_every`; `--llm` sets 25000, `--llm-every N`
sets N), `director.llm_temperature` (0.4), `director.llm_timeout_s` (60), `director.llm_max_tokens` (400); and
`run.llm_url` (`http://localhost:1234/v1/chat/completions`) and `run.llm_model` (`gemma-4-26b-a4b-it-mlx@8bit`). The
`director.*` keys are policy keys (unhashed, `POST_V01_KEYS`, `UNHASHED_KEYS`); every hash of §26.2 is unchanged. The
tier exists in live mode only: `Director` builds the `Client` only for `mode == "live"` with `llm_every > 0`, and a
replay never does. `--llm` / `--llm-every N > 0` with any other effective mode is a `ConfigError` (§21). The log header
records `llm_every` and `llm_model` while the tier is on (and nothing about it while it is off, so a step-2 log keeps
its header).

**Roles (L2).** At every rule boundary the order is: (1) the reverts that are due; (2) the safety rules, extinction
rescue and runaway damper, `rules.decide_safety`; (3) at a boundary with `t % llm_every == 0`, the model (below).
Between LLM boundaries the rule tier takes **no** discretionary move. The discretionary rules (monoculture breaker,
stagnation response, the Trickster's boredom, `rules.decide_discretionary`) run only as the fallback of a failed call.
`rules.decide` is the safety rules followed by the discretionary ones and is what the director calls with the tier off
(`RULES_VERSION` is unchanged). The prompt shows the reverts and safety actions already taken at the boundary.

**Call (L1).** Synchronous, inside the existing hook (after the row and the violation check, before the checkpoint).
No thread and no pending request, so landing steps never depend on wall-clock latency. A failure of any kind, and a
reply of which nothing could be applied, hands the boundary to the rule tier's discretionary rules:

| outcome | meaning | fallback |
|---|---|---|
| `ok` | at least one action applied | no |
| `none` | the reply was a valid `none`: a decision to do nothing | no |
| `failed` | timeout, connection error, HTTP error (a redirect included), a body that is not a chat completion, an empty `content`, an oversized response | yes |
| `invalid` | the content is not a valid reply (below) | yes |
| `rejected` | a valid reply, none of whose actions could be applied | yes |

**Request (L6).** One POST of `{model, messages: [system, user], temperature, max_tokens, seed, stream: false,
response_format}` to `run.llm_url`, with `response_format = {"type": "json_schema", "json_schema": {"name":
"intervention", "strict": true, "schema": S}}`. `seed` is a deterministic int31 of `(run.seed, step)`; a server may
ignore it. `S` is **generated from `actions.MENU`** and the config, never copied:
`{"actions": [1 to 2 items], "journal": string}`, each item an `anyOf` of one variant per menu action plus `none`; a
variant is `{"action": enum[name], "params": {exact required keys, additionalProperties: false, every number with
minimum and maximum}, "expected_effect": string}`. `journal` has `maxLength` 600 and `expected_effect` 240 (without
`maxLength` models loop inside a string until `max_tokens`; without numeric bounds they emit runaway integers).
Parameters (the names a rule's `Proposal.raw` takes): `set_seasons` `period` (integer), `amp`; `fertility_shift` `patch`
(0 to `n_patches − 1`), `gain`; `habitat_loss` `patch`, `duration`; `regen_pulse` `mult`, `duration`;
`immigration_pulse` `count`; `catastrophe` absolute `x`, `y` in `[0, L]`, `radius` (the clamp's range), `kill_p`;
`drift_change` `mult`. **`fertility_shift` is offered in its gain form only**: the patch *move* stays a rule-tier action,
so each menu action is one schema variant. Stdlib `urllib` only.

**Client containment (L11).** The URL is user config. The client (`llm.Client`) refuses a URL that is not `http(s)` with
a host, ignores proxy environment variables, follows **no** redirect (a 3xx is an `http` failure), reads at most 1 MiB,
enforces the timeout as a total deadline (each socket operation, and the body in chunks), and never raises: every
failure is an `Exchange` with an `error` string of the form `<kind>: <detail>`, the detail drawn from our own code and
never from the server.

**Validation (L8).** `llm.parse_reply` treats every byte as untrusted: one JSON object with exactly the keys `actions`
and `journal`, no duplicate keys, no `NaN` / `Infinity`, 1 to 2 actions, each with exactly `action`, `params` and
`expected_effect`, an `action` in the menu or `none`, `params` with exactly the action's keys and finite numbers
(integers where the menu has them; `true` is not a number), strings within their `maxLength` (a violation of any of
these is `invalid`). Numeric *ranges* are not checked there: the step-2 clamp brings a number inside its bounds, the
journal records each change (`clamps`, rounding to a rule boundary included), and what the clamp cannot make sense of (a
patch that does not exist) is a rejected proposal. A repeated action name is dropped (the first wins). `none` next to
another action is ignored and the other action kept; `none` alone is a decision. Each surviving action becomes a
`Proposal` and takes the rule tier's path: `rules.eligibility` (budget, the action's cooldown with the catastrophe
cap, a full immigration queue, one active effect per lever; the persona's minimum gap and drama, which only shape the
rule tier's discretion, do not apply), then `Director._take` (clamp, validate, apply), with the pacing and the
environment refolded after each accepted action. Text from the reply (`expected_effect`, `journal`) is stripped of
control and format characters, collapsed and cut to its limit; nothing is evaluated, no text is ever fed back into a
prompt (earlier interventions are shown by action name and numbers), and the reply is used for nothing beyond menu
selection.

**Prompt (L7).** `prompt.system_message`: the persona paragraph (the Architect's also carries `director.goal`, verbatim,
between `<goal>` tags and introduced as data), the rules of the reply, and the menu with an effect description per action,
bounds, cost and cooldown length (models shown bounds alone choose `immigration_pulse` for everything).
`prompt.user_message`: availability (cooldowns remaining, active timed effects, budget left, the actions that are
affordable and not cooling down), the latest vital signs, the trends of seven of them over the last 8 rule windows, the
safety actions and reverts of this boundary, the last five interventions with their measured effect (the change in
population, top clade share, clades, emitters, sharers, kill share and novelty from the intervention's boundary to 50k
steps later, or to now with "(so far)"). About 2.4k tokens for the `tiny` preset.

**Logging (L9).** An applied action is an entry with `source: "llm"`, the persona, the clamped `params`, `reason` = the
model's `expected_effect`, no `rule` and `llm_call: "llm-<step>"`. `journal.jsonl` gets one record per call:
`{"kind": "llm_call", "step", "persona", "llm_call", "model", "url", "latency_s", "messages" (the raw prompt),
"response" (the raw text, or null), "finish_reason", "usage", "outcome", "error", "validation" (what the parser dropped
or ignored), "accepted", "rejected" (each with the reason), "clamps" ({action, param, requested, applied}),
"fallback" (null, or {"tier": "rule", "actions": [...]}), "journal" (the model's text)}`. The console prints one line per
call (`director @ <t>: llm 3.5s: 2 action(s) applied`) before the entries' lines.

**Replay (L10).** `llm` entries replay like rule entries: `actions.SOURCES = ("rule", "llm", "revert")`, so the
pre-validation accepts them, and replay never calls the model. A replay of a live run made with the (mock) model is
bitwise identical in digest, rows and log (tested, also with `urllib` and sockets patched to fail). A resumed live run
re-asks the model for the boundaries after the checkpoint (the journal and log are truncated to it first), so its
decisions need not equal the uninterrupted run's; the log is the source of truth.

**Real server.** `tests/test_director_llm.py` has one test marked `llm` that runs only when `ALIFE_LLM_URL` is set (and
optionally `ALIFE_LLM_MODEL`); the default runs never need LM Studio.

### 26.11 The `random` persona: arm R, yoked to another run (evaluation)

Arm R of `docs/DIRECTOR.md` §9 is a control that perturbs as often and as late as arm B, but chooses nothing. The owner
decided it is **yoked**: it reads B's `director.jsonl` and intervenes at exactly the steps where B made a discretionary
move, with the same number of moves, picking uniformly random eligible actions with random parameters. That isolates the
quality of B's choices from the amount and timing of perturbation. Decisions R1 to R7 of the spec; everything below
applies only to `director.persona = random`, and with any other persona nothing changes (the hashes, the logs and the
trajectories are as before).

**Config and CLI (R1).** `random` joins `config.PERSONAS`. `director.yoke_log` (str, default `""`) is a policy key like
the others: in `POST_V01_KEYS` and `UNHASHED_KEYS` (so no hash changes and a resume may change it). `--yoke-log PATH` on
`run` (a fork included) and `resume` is its shortcut. `validate`: `random` needs a non-empty `yoke_log` and
`llm_every = 0`; a non-empty `yoke_log` with any other persona is an error. At the start of a **live** `random` director
(`run`, fork and `resume`, checked in the CLI before any directory is created or touched, and again by `Director`; exit
2) the yoke log must exist, pass the replay pre-validation of §26.3 against this run's config, have every yoked step
a multiple of this run's `rule_every`, and have headers that name a persona R can inherit from (R3: not `random`, not an
unknown name). The log header records `yoke_log` (as given), `yoke_sha256` (of the file's bytes), `params_persona` (the
persona R inherited, R3) and, as always, the effective `budget_per_100k`, only for this persona. Paths are used as given, relative to the working directory.

**Which moves are yoked (R2).** From the yoke log, every entry whose `source` is `rule` or `llm` and whose `rule` is not a
safety rule (`rules.SAFETY_RULES` = `extinction_rescue`, `runaway_damper`). Reverts never are. They are counted per
`step_applied`. A log whose entries carry no `rule` field (older logs) has nothing to tell a safety move by: its `llm`
entries and its `rule` entries all count, reverts excepted.

**Inherited parameters (R3).** R lives by the `PersonaParams` of the persona that made the yoke log, not by a fixed
persona: arm R yoked to B (Gardener) is paced as B, yoked to C (Trickster, budget 6 per 100k) as C. `yoked.inherited`
reads the persona and the effective `budget_per_100k` from the yoke log's **first header** (the run's own start). A later
header (a resume that changed the persona or budget) is not followed: R keeps the first one's parameters for the whole
run and a live start prints a stderr warning when the headers name several personas. (Following the persona in force at
each yoked step would also need R's safety rules, budget window and cooldowns to switch mid-run; not built.) Inherited
are the budget, the budget maximum, `cooldown_scale` and, through `rules.RuleParams.persona`, the thresholds the safety
rules read. Not inherited, because they belong to discretionary choice, which R does not have: the Trickster's boredom
rule, the persona's minimum gap and drama gating, its action weights and sampling, and the model. A header with no
`persona` (or no header) gives the Gardener, the pre-R3 behaviour. A header persona of `random` (the yoke log is itself a
yoked run: yoke R to the run it followed) or a name with no parameters is `LogError`, exit 2, checked in any header of
the log. The budget: an explicit `director.budget_per_100k` is capped by the inherited persona's maximum; `0` (the
default) takes the header's `budget_per_100k` (what the yoked run actually had), or the inherited persona's default if the
header recorded none; a malformed one (not a positive integer) is `LogError`. In replay (which reads no yoke log) the
persona and budget are the ones R's own first header recorded (`params_persona`, `budget_per_100k`; the Gardener for a log
written before R3). Consequence: R yoked to a Gardener log is the same as before for any explicit budget; with
`budget_per_100k = 0` it now has the budget B's header recorded (2 for a default B, as before).

**Behaviour at a boundary (R3, continued).** The entries and the journal still say `random`. Reverts and the safety rules run as for any persona
(`rules.decide_safety`); there is no boredom rule, no other discretionary rule and no model. At a yoked step with count
`k`, `k` times: take the menu actions that `rules.eligibility` allows now (budget, cooldown, one effect per lever, the
catastrophe cap, a full immigration queue; the persona's minimum gap and drama do not apply, as for the model) and, for
`habitat_loss`, a free alive patch while at least two are alive; pick one uniformly; draw its parameters; take it
through the normal path (clamp, validate, apply, log); refold the pacing. If nothing is eligible, or the clamp refuses
every candidate, the journal gets `yoked step <t>: nothing eligible (<n> of <k> moves not made) [why, per action]` and
the remaining moves of the step are not made. Entries: `source: "rule"`, `rule: "random"`, `persona: "random"`,
`yoked_to: <step>`, `reason: "yoked draw i of k"`. `mutation_floor_nudge` is not on the menu, so it is never drawn. With
a budget of 2 (a default Gardener) a catastrophe (cost 3) is never eligible; with the Trickster's 6 it is.

**Sampling (R4).** `yoked.rng_for(seed, t, draw)` is `Generator(PCG64(SeedSequence([run.seed, t, 100, draw])))` (the
Trickster's streams are 0 to 9): stateless, one per draw. It picks the action, then `yoked.sample_params(action, rng, cfg,
patches)` returns `(raw, duration)` as a rule's `Proposal` does, within the live `actions.MENU` bounds, `radius_bounds(cfg)`
and the world size. Log-uniform: season period, immigrant count, every duration (`habitat_loss`, `regen_pulse`), drift
multiplier. Uniform: amp, gain, the regen multiplier, `kill_p`, the catastrophe radius, the centre (over the torus) and the
patch index. `fertility_shift` is a gain change or a move with probability 1/2 each; a move has a uniform direction and a
uniform length in `[0, L/4]`; with no free patch it is a move. Integers are rounded after sampling; `actions.clamp` does
the rest (durations round up to `rule_every`) and its output is what is logged.

**Replay and resume (R5).** R's own log replays like any other (§26.8): `Director` loads the yoke log in live mode only, so
a replay never opens it (tested with the file gone). A live `resume` re-reads it; if its sha256 differs from the latest
header's, a warning goes to stderr and the run continues, and a new header records the file now in use.

**GUI (R6).** The Director panel's persona menu gains Random; with it selected the panel shows a "Yoked to run" picker
(runs that have a `director.jsonl`; the resumed run itself is excluded) passed as `--yoke-log=runs/<run>/director.jsonl`,
and hides the LLM controls (forced off: `--llm-every 0` when the baseline had it on). Leaving Random for another persona
passes `--yoke-log=` to clear the key. `DirectorSettings.validate` mirrors R1 (a run chosen, and it has a log).

**Running the arms.** Run B first. Then run R from the same fork point, with the same seed, `rule_every` and metrics
cadences: `alife run --init-from SRC:STEP --director live --persona random --yoke-log runs/B/director.jsonl ...`; for the
Trickster arm C, `--yoke-log runs/C/director.jsonl` (leave `director.budget_per_100k` at 0 so that R has C's budget, 6;
an explicit value is capped by the Trickster's 12). R's moves
are yoked to B's steps, not to B's state, so the two trajectories diverge after the first intervention; the counts match as
long as R has budget, an unlocked lever and a free cooldown at each yoked step (the journal says when it did not).

---

## 27. 3D world (`world.dims = 3`)

`docs/3D.md` is the design rationale and the six-PR plan (its §10); this section is normative for what is built and
grows with each PR. PR 1 (§27.1) adds the dims constant and its plumbing only: `validate` still rejects
`world.dims = 3`, and every 2D trajectory, config hash, metrics row, checkpoint array and metadata key set is
unchanged (`config.json` and the embedded config only gain the two new keys at their defaults; the existing tests
and the golden digest are untouched). PR 2 (§27.2–§27.5) adds the 3D geometry, physics, state and static lattice as
functions the step does not call yet: `world.dims = 3` is still rejected, and 2D is again unchanged (config files
gain the PR 2 keys at their defaults).

### 27.0 Overview and decision record

The 3D world is periodic in x and y on `[0, L)` as in 2D and bounded in z by a floor (z = 0) and a surface
(z = `world.height`); agents stay in `[r, height − r]`. Gravity, a buoyancy actuator and a vertical food profile
come in PR 2 and sinking corpses in PR 3, which also wires them into the step. `height` (or `Lz`) names the world
height in code and docs: `H` is the hidden size.

**Owner decisions** (docs/3D.md §0 and §11): 3D with gravity and a floor; a `world.dims` config constant that keeps
2D bit-identical; a Swift Metal viewer in AlifeControl; after the 2D director evaluation. On 2026-10-13:

| # | question | decision |
|---|---|---|
| Q1 | vertical food | a benthic band plus an optional photic band; `fields.photic_amp` defaults to 0 (off) |
| Q2 | surface | a clamp at z = height − r that zeroes the inward velocity, without grounding (no floor friction) |
| Q3 | buoyancy | brain output `OUT_BUOYANCY` with a heritable cap `buoy_cap ∈ [0, 1]` (effective cap `buoy_cap·physics.buoyancy_max`), additive Gaussian mutation with a clip (`mutation.buoy_sigma`); upward force `b = cap·max(tanh y, 0)`; cost `buoyancy_cost·(b/bmax)² + buoyancy_cap_cost·buoy_cap` per step, both booked in `HEAT_BUOYANCY` (ledger slot 15) |
| Q4 | immigrants | uniform in the volume |

**Accepted technical choices** (one line each in docs/3D.md §12): the append-only genome layout (§27.1); a parallel
`State3` (State's 47 fields, then `pitch`, `buoy_cap`), so `State` and every 2D digest stay as they are; ledger slot
15 keeps the canonical name `RESERVED` with `HEAT_BUOYANCY` as an alias named per dims in metrics; the existing RNG
streams with 3D-only draw shapes (`RNG_LAYOUT_VERSION` stays 2); offspring in a cone behind the parent with vertical
reflection; `Counter3D` (27–29) appended to `StepOut.counters` in 3D only; mirrored field z-gradients; the vertical
profile by layer index (floor layer = 2D fertility) with no grounded gate on harvest; a vertical-column catastrophe;
one step pipeline with static `if d3:` branches; grip only on the floor, never slowing sinking; `IN_HEIGHT` (z /
height) rather than "depth"; a `tiny3d` preset (PR 3); a 3D golden digest recorded in CI.

### 27.1 Scope and the dims constant (PR 1)

**Keys.** `world.dims` (int, 2 or 3, default 2) and `world.height` (float, default 0). Both are static: they change
the compiled step and the state shapes. Both are in `config.POST_V01_KEYS` (omitted from `config_hash` at their
defaults, so every existing run keeps its hash) and in `config.THREE_D_KEYS`, never in `UNHASHED_KEYS`.
`config.is_3d(cfg)` is the host predicate; simulation code branches on `dims(cfg).d3`.

**Gate and validation.** `world.dims` must be 2 or 3 and in `config.DIMS_SUPPORTED`, which is `(2,)` until PR 3
(read at call time, so tests of 3D code can widen it); dims = 3 reports "world.dims = 3 is not enabled yet". In a
2D world every key of `THREE_D_KEYS` except `world.dims` must keep its default. The 3D rules apply only when
dims = 3 (gated or not), each with its own message, and no 2D message changed. They round only finite ratios, so a
non-finite `size` or `height` is a listed problem, never an exception:
- `height` is finite and `> 2·physics.radius`;
- `height / field_cell` is an integer `FZ ≥ 2`;
- `FZ·FG² ≤ 2²⁰` (the limit of `ops.exact_sum`, which sums the resource for `TOTAL`);
- `G_z = floor(height / R) ≥ 1`;
- `2·height ≤ size`. Then every `|dz| < L/2`, so `ops.min_image` is exactly the identity on z
  (`round(dz/L) = ±0`), and bond and neighbour offsets need no z-specific code;
- `NS ≤ K ≤ 27·C` (the 3×3×3 stencil); the 2D rule `K ≤ 9·C` applies in 2D only;
- every `chem_diffusion ≤ 1/6` (explicit 7-point stencil);
- `resource_diffusion_q8 ≤ 42`: with six neighbours, `6·((R·d) >> 8) ≤ R` keeps every cell ≥ 0.

PR 2 adds the rules of its keys, each with its own message (their meaning is in §27.3–§27.5):
- `height − 2·r ≥ 2·repro.offspring_distance`, so a child reflected at the floor or the surface lands in range
  at the full offset (§27.4);
- `0 ≤ gravity < thrust_max·sin(PITCH_MAX)`: an agent can always climb at full pitch;
- `floor_eps ≥ 0` and `r + floor_eps < field_cell`: a grounded agent is in the floor layer;
- `buoyancy_max` is finite and `≥ 0`, and `thrust_max + max(gravity, buoyancy_max − gravity) < 2·r·k_repel`, the
  3D form of the §9 rule `thrust_max < 2·r·k_repel`: full thrust plus the net vertical load cannot push a resting
  agent through the one below it (or, buoyant, above it);
- `buoy_init ∈ [0, 1]` and `buoy_sigma ≥ 0` (finite), both 0 when `buoyancy_max = 0`;
- `benthic_lambda > 0`; `photic_amp ∈ [0, 1]`; `photic_lambda > 0` when `photic_amp > 0`;
- `director.enabled` is false: the director's levers, vital signs and log stay 2D until PR 4.

**`Dims`** gains `nd`, `d3`, `lay` (`layout.for_dims(nd)`), `fz`, `n_cells` (`FZ·FG²`), `field_shape` (`(FG, FG)`
in 2D, `(FZ, FG, FG)` in 3D), `g_hash_z` and `cell_hash_z` (1 and ∞ in 2D), and derives `Q = 3^nd·C`,
`W1_ROWS = lay.D_IN + H` and `params` from `lay.D_OUT`. Every 2D value equals the old one.

**Layout.** The module constants of `layout.py` are the 2D layout, unchanged. `layout.Layout` is a frozen
per-dims view and `layout.for_dims(nd)` returns it (cached; any other `nd` is a `ValueError`). A 3D world uses
the **append-only** layout: the 2D input and output vectors unchanged, then a 3D-only tail, so every `IN_*`,
`OUT_*`, `NBR_*` and `BOND_*` index keeps its meaning in both dims.

| inputs | index | content |
|---|---|---|
| 2D inputs | 0–68 | as §5 |
| `IN_VEL_UP` | 69 | body-frame up component of `vel / v_max` |
| `IN_RES_GRAD_UP` | 70 | resource gradient, up |
| `IN_CHEM_GRAD_UP` | 71–73 | chemical gradients, up (NCH wide) |
| `IN_NBR_DZ` | 74–76 | up offset / R of the NS nearest neighbours |
| `IN_BOND_DZ` | 77–80 | up offset / `break_length` per bond slot |
| `IN_HEIGHT` | 81 | z / height |
| `IN_PITCH` | 82 | sin φ |
| `IN_GROUNDED` | 83 | 1 on the floor, else 0 |

| outputs | index | content |
|---|---|---|
| 2D outputs | 0–34 | as §7 |
| `OUT_PITCH` | 35 | pitch change `turn_max·tanh(y)` |
| `OUT_BUOYANCY` | 36 | buoyancy fraction `max(tanh(y), 0)` |
| `OUT_Q_BOND_DZ` | 37 | 7th component of the bond query |
| `OUT_Q_ATTACK_DZ` | 38 | 7th component of the attack query |

So `D_IN = 84` and `D_OUT = 39`. The neighbour descriptor is `NF = 8` wide with `NBR_DZ = 7` last, `F = 7`, and a
query is `[y[OUT_Q_BOND : +6], y[OUT_Q_BOND_DZ]]`, in the order of `φ = desc[..., 1:]`. The 2D blocks keep their
widths (`IN_NBR` is NS·7, `IN_BOND` is B·5), and the module constants `NF = 7` and `F = 6` remain valid as those
block widths in both dims. Indices that exist only in 3D are None in a 2D `Layout`, never −1. With H = 32 a 3D
genome has (84 + 32)·32 + 33·39 = 4 999 parameters (2D: 4 387). Outside `layout.py`, `D_IN`, `D_OUT` and the
genome version are read only through `dims(cfg).lay` and `layout.genome_layout_version(nd)`; a test scans `src`.

**Genome layout version.** `layout.genome_layout_version(nd)` reads the module globals at call time:
`GENOME_LAYOUT_VERSION = 1` for 2D and `GENOME_LAYOUT_VERSION_3D = 2` for 3D (one sequence: a future 2D bump takes
3). The 3D tables are frozen now; no 3D checkpoint can exist before PR 3, and after it any change bumps the version.

**Metadata.** `meta.json` and checkpoint metadata record the genome version of the world's dims and, for a 3D
world only, `dims` (an int in `meta.json`, a string in checkpoint metadata). An absent key means 2D, so every 2D
file keeps its exact key set. `FORMAT_VERSION` and `RNG_LAYOUT_VERSION` are unchanged.

**Refusals.**
- `load_checkpoint` refuses a file whose `dims` (absent = 2) differs from `world.dims` ("cross-dimension
  resume/fork is not supported", exit 4), with or without the hash check, before reading any array.
- A fork refuses a `world.dims` or `world.height` different from the source's (exit 4, nothing written); resume
  refuses both as dynamics keys (exit 2). Until 3D is enabled a fork to dims = 3 already fails validation (exit 2).
- The 2D-only tools call `config.require_2d(cfg, what)` (`ConfigError`, exit 2) before any state is loaded or
  built: `alife render` and `alife view`, `render.render_frame`, `viewer.ViewerSession` and `viewer.run_viewer`
  (before pygame is imported), and `analysis.signals` (`decode`, `analyse`).

**One schema.** `state.expected_layout(cfg)` is the shape and dtype of every State field: `pos`/`vel` `(N, nd)`,
the weight widths from `lay`, `resource` `field_shape`, `chem` `(NCH, *field_shape)`. `load_checkpoint` and
`full_check` compare against it; `runio._expected_layout` is an alias kept for an existing test.

**Ledger names.** `LedgerIdx.HEAT_BUOYANCY = 15` is an `IntEnum` alias of `RESERVED`: it is not iterated, not
counted by `len`, and its `.name` is `"RESERVED"`. `state.sinks(nd)` is `SINKS` plus slot 15 in 3D and
`state.ledger_names(nd)` names slot 15 `HEAT_BUOYANCY` in 3D. PR 1 only defines them; PR 3 books and checks the term.

**Tests.** `tests/test_layout_dims.py` (2D view equals the module constants, 3D tiling, the 2D prefix, the version,
the `src` scan), `tests/test_dims_config.py` (defaults, pinned hashes, the gate, each 3D rule, the `Dims` fields) and
`tests/test_dims_runio.py` (metadata, cross-dims refusals, fork and resume, the 2D-only tools, one schema). The PR 2
keys: `tests/test_dims_config_pr2.py` (defaults, hashes, 2D rejection, each rule above, the float32 z bounds,
`z_ground ≥ z_lo`).

### 27.2 Geometry, orientation and neighbour search (PR 2)

**Coordinates and bounds.** `pos` is (x, y, z). x and y are periodic on `[0, L)` as in 2D; z is bounded by the
floor (z = 0) and the surface (z = `height`), and an agent's centre stays in `[z_lo, z_hi]`. `Dims.z_lo` is the
smallest float32 `≥ r` and `Dims.z_hi` the largest float32 `≤ height − r`; `Dims.z_ground`, the grounded threshold
(§27.3), is the largest float32 `≤ r + floor_eps`, raised to `z_lo` when no float32 lies in `[r, r + floor_eps]`
(as for `r = 0.7` with `floor_eps = 0`). All three are 0.0 in 2D. They are computed on the host in `config.dims`, so
a clip with them (MLX casts Python bounds to float32) and a float64 range check agree.
`ops.wrap_xy_clamp_z(p, L, z_lo, z_hi)` maps (…, 3) positions into the world: x and y as `ops.wrap` (with the F8
guard), z clipped and never wrapped (−0.3 would become `L − 0.3`). `ops.wrap` and `ops.min_image` stay elementwise
and unchanged. Because `2·height ≤ L` (§27.1), every in-range `|dz| < L/2`, so `min_image(dz) = dz − L·(±0) = dz`
exactly: offsets of 3-vectors (neighbours, bonds) need no z-specific minimum image.

**Orientation.** The yaw θ is `heading`, wrapped to `[−π, π)` as in 2D. The pitch φ is `State3.pitch`, clipped to
`[−PITCH_MAX, PITCH_MAX]` with `layout.PITCH_MAX = 80°` and never wrapped (checked against `float32(PITCH_MAX)`).
Gravity defines up, so there is no roll. The body frame is

```text
fwd = (cos φ·cos θ, cos φ·sin θ, sin φ)
lat = (−sin θ, cos θ, 0)
up  = fwd × lat = (−sin φ·cos θ, −sin φ·sin θ, cos φ)
```

right-handed and orthonormal, with a horizontal lateral axis; `(θ, φ) = (0, 0)` gives the world axes and a positive
pitch raises the nose. `ops.body_axes(cos θ, sin θ, cos φ, sin φ)` returns the three axes in world coordinates (the
thrust direction, §27.3). `ops.to_body3(v, cos θ, sin θ, cos φ, sin φ)` rotates (…, 3) world vectors into
(fwd, lat, up): `h = vx·cos θ + vy·sin θ` and `lat = vy·cos θ − vx·sin θ` exactly as `to_body`, then
`fwd = h·cos φ + vz·sin φ` and `up = vz·cos φ − h·sin φ`. That is the projection on `body_axes`, so the sensed
forward and the thrust direction are one frame; at φ = 0 with `vz = 0` it reduces to `to_body`.

**Neighbour search.** `spatial.find_neighbors` dispatches to `_find_neighbors_3d` when dims = 3 (the 2D body is
unchanged). Its contract is §4's, with `off` (N,K,3):
- The hash grid has `G_h × G_h × G_z` cells: `G_z = floor(height / R)` layers of height
  `cell_hash_z = height / G_z ≥ R`, so the 3×3×3 stencil covers the sensing sphere and `Q = 27·C`.
- The cell of a position is `cx`, `cy` as in 2D (with the `% G` seam guard) and
  `cz = clip(floor(z / cell_hash_z), 0, G_z − 1)`, never wrapped: a z at the surface that divides to `G_z` in
  float32 (z = height, and for some heights the largest float32 below it) lands in the top layer. The cell id is
  z-major, `(cz·G + cy)·G + cx`; dead rows take the sentinel `G²·G_z`.
- The stencil is dz outermost, then the 2D 3×3 (`_DX27`, `_DY27`, `_DZ27`). x and y neighbour cells wrap. A layer
  outside `[0, G_z)` keeps a clamped index but an empty run (`e = s`), so no layer is visited twice: `G_z` may be 1
  or 2, and only x and y need `G ≥ 3`.
- Distances are per axis, `dx, dy = min_image(·, L)` and a plain `dz`, with `d2 = dx² + dy² + dz²`; no (N,Q,3)
  array is formed. Selection is §4's, and only the K winners' offsets are gathered into `off` (N,K,3). The z offset
  is the plain float32 difference `z_j − z_i`.
- `overflow_cells` and `max_occupancy` count the `G²·G_z` cells: a column of agents over one x/y cell overflows only
  where a single 3D cell holds more than C. Equal to brute-force kNN whenever no cell does (F13). The candidate
  arrays are 3× the 2D ones at the same C; C and K for gravity piles are sized by measurement (docs/3D.md §12 Q15).

**Neighbour descriptor.** In 3D `sensing.neighbor_descriptors` reads `State3.pitch` and returns `desc (N,K,8)`:
the seven §5 features, with (`NBR_DX`, `NBR_DY`) the (fwd, lat) components of `off / R` from `to_body3`, then
`NBR_DZ = 7`, the up component of `off / R`. Each is clipped to `[−1, 1]` (every component of an offset inside the
sphere is at most R) and zeroed where invalid. The targeting descriptor `φ = desc[..., 1:]` is seven wide:
`[dx, dy, kin, rele, msg0, bonded, dz]`. `IN_NBR` stays `desc[:, :NS, :7]`; the NS nearest `NBR_DZ` values are the
tail block `IN_NBR_DZ` (wired in PR 3).

**Action decode.** In 3D `actions.decode` builds each query as its 2D block of six followed by its `*_DZ` column,
`[y[OUT_Q_BOND : +6], y[OUT_Q_BOND_DZ]]` and likewise for attack, in the order of `φ`. In 2D the `*_DZ` indices are
None and the query is the 2D block, with the same trace as before. The 3D-only outputs are decoded into a parallel
type, so `Actions` keeps its fields:

```python
class Motion3(NamedTuple):           # (N,) f32 each, masked like Actions
    pitch: mx.array                  # turn_max·tanh(y[OUT_PITCH]), the pitch change (clipped in §27.3)
    buoy: mx.array                   # max(tanh(y[OUT_BUOYANCY]), 0) ∈ [0, 1): the buoyancy fraction
def decode_motion3(y, alive, cfg) -> Motion3      # dims = 3 only (asserted); masked by alive
def mask[A: (Actions, Motion3)](act: A, keep) -> A    # returns the type it is given
```

`buoy` follows EMIT and SHARE: zero buoyancy is exactly reachable, and so free (§27.0 Q3). The step masks `Motion3`
by `alive` and then by `solvent`, as it does `Actions` (PR 3).

**2D-body guards.** Static asserts on the last axis: `ops.to_body` takes (…, 2), `ops.to_body3` and
`ops.wrap_xy_clamp_z` take (…, 3), and the 2D body of `find_neighbors`, `physics.integrate` (§27.3) and the 2D
cell index (§27.5) take (N, 2) positions. Two more guards are not on positions: 2D `sample_fields` takes a 2D
lattice (§27.5) and `decode_motion3` requires dims = 3. They are free under `mx.compile` and leave the 2D trace
unchanged, so a missed 3D branch fails at trace time instead of silently dropping z.

**Tests.** `tests/test_ops_3d.py` (`wrap_xy_clamp_z`; the frame: orthonormal, right-handed, named angles, the
round trip, broadcasting, φ = 0 against `to_body`; `min_image` as the identity below L/2; the guards).
`tests/test_spatial_3d.py` (exact brute-force kNN on random volume scenes; the floor, the surface and the x/y seams
with the eight corners; thin slabs, `G_z = 1, 2`; z not periodic; R strict along z; the top layer; overflow over
`G²·G_z` cells and a column that does not overflow; the empty world; dispatch and the 2D guard; compiled equals
eager). `tests/test_sensing_3d.py` (a level scene against the 2D descriptor bit for bit; the body frame against a
float64 reference; a named case; compile). `tests/test_actions_3d.py` (`Motion3` ranges, columns, masking and the
3D-only assert; the DZ query order and independence; the 2D decode as the 3D one's prefix; compile).

### 27.3 Physics: gravity, buoyancy, the floor and the surface (PR 2)

**Forces** (`physics.forces3`, (N,3)). They are §9's, evaluated at the start-of-step positions with the post-S14
bonds, the same contact mask and the same attenuation `att`, plus two body forces on z:

```text
F_i = thrust_max · a_i · att_i · fwd_i                       fwd_i = body_axes(θ_i, φ_i)[0] (§27.2)
    + Σ_b springs (§9) + Σ_k contacts (§9)                   on the (N,B,3) and (N,K,3) offsets
    + (buoyancy_max · buoy_cap_i · b_i − gravity) · ẑ          b_i = Motion3.buoy ∈ [0, 1); not attenuated
```

- Gravity and buoyancy are body forces: `att` does not scale them. A crowded agent (F18) loses its thrust but still
  sinks, and buoyancy can lift it. An isolated agent sinks at `gravity / drag` per step (overdamped) and climbs at
  full pitch at `(thrust_max·sin φ − gravity) / drag`, which `gravity < thrust_max·sin(PITCH_MAX)` (§27.1) keeps
  positive. Gravity has no ledger term (an overdamped body force does no bookkeeping work); the buoyancy costs and
  their `HEAT_BUOYANCY` term come in PR 3.
- Springs and contacts are §9's expressions on 3-vectors. `min_image` is the identity on z (§27.2), so `bonds` has
  no 3D code (`BondGeom.off` is (N,B,nd)) and the bitwise pair antisymmetry of §9 holds in 3D. The body forces act
  on z only, so x and y stay antisymmetric under gravity.
- Non-survivors get 0.

**Grounding.** `physics.grounded(pos, cfg) = z ≤ Dims.z_ground` (§27.2), from the start-of-step position, inclusive:
for a float32 z, exactly `z ≤ r + floor_eps` or `z = z_lo`, so an agent resting on the floor is always grounded, and
with `floor_eps = 0` exactly the agents at `z = z_lo` are. It is the one predicate for the floor friction below,
`IN_GROUNDED` and the `GROUNDED` counter (PR 3). The surface never grounds (§27.0 Q2). `r + floor_eps < field_cell`
(§27.1) puts every grounded agent in the floor layer.

**Integration** (`physics.integrate3 → Integrated3`, survivors only):

```text
γ_xy = drag · (1 + grip_gain · grip · [grounded])        γ_z = drag
v    = (F_x / γ_xy, F_y / γ_xy, F_z / γ_z) ;   v ← v · min(1, v_max/‖v‖)    (3D norm; each clip counted)
p    = pos + v ;   pos' = (wrap(p_x), wrap(p_y), clip(p_z, z_lo, z_hi))   (ops.wrap_xy_clamp_z)
floor clamp:   p_z ≤ z_lo and v_z < 0 ;   surface clamp:   p_z ≥ z_hi and v_z > 0
vel' = (v_x, v_y, 0 on a clamped row, else v_z)
θ'   = wrap_angle(θ + turn) ;   φ' = clip(φ + Δφ, −PITCH_MAX, PITCH_MAX)    Δφ = Motion3.pitch
```

- Grip acts only on the floor and only on x and y: it never slows sinking (§27.0, docs/3D.md §12 Q14). It still
  costs `grip_cost` everywhere (§10.3).
- The `v_max` clip is on the 3D speed and comes before the clamp, so a resting agent under gravity spends part of
  `v_max` on a `v_z` that the clamp then cancels (`√(0.5² − 0.03²) ≈ 0.499` left for x and y at `gravity = 0.03`).
- A clamp moves the agent to the boundary (an agent 0.2 above the floor falling at 0.5 lands at `z_lo`; it does not
  hang 0.2 above it) and zeroes only the stored `v_z` of the clamped rows; x and y move and wrap as usual. `vel` is
  read only by sensing (`IN_VEL`, `IN_VEL_UP`, PR 3), so an agent resting on the floor senses no vertical motion.
  Under gravity every resting floor agent clamps every step, so the floor-clamp count is about the grounded
  population.
- `Integrated3(pos, vel, heading, pitch, v_clips, z_floor, z_surface)`: the three counts are () int32 (`V_CLIPS`
  and the `Counter3D` clamp counts of PR 3).
- Non-survivors keep `pos`, `heading` and `pitch` and get `vel = 0`. The pitch is clipped, never wrapped, at the
  float32 `PITCH_MAX`.
- Stability: gravity and buoyancy are constant forces and grip only raises γ, so §9's Gershgorin bound is unchanged.
  The load rule `thrust_max + max(gravity, buoyancy_max − gravity) < 2r·k_repel` (§27.1) is the 3D form of §9's
  `thrust_max < 2r·k_repel`.

**A level floor is the 2D physics.** With every agent on the floor at pitch 0 and `gravity = buoyancy_max = 0`,
`forces3` and `integrate3` give x and y equal to `forces` and `integrate`, bit for bit on CPU (every agent is
grounded, so grip is §9's), z stays at `z_lo` and nothing is clamped. F11 (contraction alone does not move the centre
of mass; phase-locked grip crawls) carries over to the floor. `physics.integrate` asserts (N,2) positions, since it
would wrap z (§27.2 guards).

**Crowding (measured, not yet a mechanism).** `tests/test_crowd_3d.py` (slow) runs 300 agents through
`find_neighbors`, `forces3` and `integrate3` in the `tiny` geometry made 3D (L 128, height 24, R 6, K 8), from a
start with at most 16 agents per hash cell. With C = 64 nothing overflows and the physics alone reaches a maximum
cell occupancy of 28 for a floor pile under `gravity = 0.03` and 37 for a ball converging by thrust without gravity,
with up to 10 true contacts per agent (K = 8 lists at most 8). With the default C = 16 both overflow on most steps,
and the agents beyond the first C of an overflowing cell, which no one sees, collapse onto each other (minimum
centre distance 0). A dense 3D packing holds about 40 agents per 6³ hash cell (FCC) and 12 contacts per agent,
against about 11 per 2D cell and 6 contacts in 2D, so contact-attenuated thrust (F18), sufficient in 2D, does not
keep a 3D crowd within C = 16. C and K for `64k3d` are sized in PR 6 (docs/3D.md §12 Q15).

**Tests.** `tests/test_physics_3d.py` (sinking and rising at the net force over drag; the floor and surface clamps
with their counts, the seam wrap of a clamped row and dead rows; landing at `z_lo` and resting there under gravity;
`grounded` inclusive at `z_ground` and never at the surface; a radius float32 cannot represent (r = 0.7) resting at
`z_lo` and grounded; grip in x and y only, only when grounded; the 3D speed clip before the clamp; the pitch clip;
the body forces not attenuated and the thrust along the 3D forward axis; repulsion and spring antisymmetry across the
seams and in z, with and without gravity; a vertical contact and a vertical bond against the 2D forces rotated onto
z; the level floor against `forces`/`integrate` bit for bit; the 2D guard; compile) and `tests/test_crowd_3d.py`
(slow, informational).

### 27.4 The 3D state, the buoyancy-cap trait and placement (PR 2)

**`State3`.** A 3D world's state is `state.State3`, a NamedTuple parallel to `State`: `State`'s 47 fields in the
same order, then `state.EXTRA_3D_FIELDS = ("pitch", "buoy_cap")`. `State` itself, every 2D digest and the 2D
checkpoint key set are unchanged. `State3.env` is the same `Env` view; `env_cat_center` stays (2,), because the
catastrophe is a vertical column. `state_cls(cfg)` and `agent_fields(cfg)` (`AGENT_FIELDS_3D = AGENT_FIELDS +
EXTRA_3D_FIELDS`) pick per dims; `mx.compile` takes and returns `State3` as it does `State` (F12).

| field | shape | dtype | meaning | invariants |
|---|---|---|---|---|
| `pos` | (N,3) | float32 | position (x, y, z) | x, y in `[0, L)`; `z ∈ [z_lo, z_hi]` |
| `vel` | (N,3) | float32 | displacement during the last step | as in 2D |
| `resource`, `chem` | (FZ,FG,FG), (NCH,FZ,FG,FG) | int32, float32 | the lattice, `[z, y, x]` (§27.5) | as in 2D |
| `pitch` | (N,) | float32 | pitch φ; the yaw is `heading` | `[−PITCH_MAX, PITCH_MAX]` (float32 bound) |
| `buoy_cap` | (N,) | float32 | heritable buoyancy cap, normalised: the effective cap is `buoy_cap·physics.buoyancy_max` | `[0, 1]` |

**Bounds.** `z_lo`, `z_hi` and `PITCH_MAX` are the float32 bounds of §27.2.

**Schema and serialisation.** `expected_layout(cfg)` appends `pitch` and `buoy_cap` in 3D. `state_to_arrays` and
`state_digest` follow `type(s)._fields`. `state_from_arrays(d, cfg)` builds `state_cls(cfg)` (without `cfg`:
`State3` exactly when the keys include `EXTRA_3D_FIELDS`) and requires its exact key set, so a state never crosses
dimensions (after the `dims` metadata check of §27.1); the pre-director `env_*` fill applies to 2D only.
`legacy_state_digest` is 2D only (`TypeError`: a 3D state has no v0 layout). `load_checkpoint`, `full_check` and
`alife inspect` read the fields of the state's own type.

**Initial state** (`init_state`, 3D branch; the 2D lines are unchanged).
- Founders are uniform in the volume (`state.volume_kinematics`, which immigrants share): `InitStream.POS` draws
  (n0, 3) uniform over `[0, L)² × [z_lo, z_hi]`, then `ops.wrap_xy_clamp_z` (x and y wrapped as `wrap`, z clipped to
  the bounds and never wrapped). `InitStream.HEADING` draws (n0, 2): the yaw uniform in `[−π, π)` (`wrap_angle`)
  and the pitch uniform in `[−PITCH_MAX, PITCH_MAX]`.
- Every other draw (tag, the weights at their 3D widths, the patches, the state key) is the 2D call on the 2D key,
  so the tags, patches and state key equal the 2D world's for the same seed.
- Dead rows hold `pos = (0, 0, z_lo)` (on the floor, so the z range holds on every row), `vel = 0`, `heading = 0`
  and `pitch = 0`. `buoy_cap = mutation.buoy_init` on every row, without a draw.
- `resource = floor(resource_cap·fertility)` is (FZ, FG, FG) (§27.5) and `chem` is zero; `TOTAL` sums every layer.

**The buoyancy-cap trait.** `buoy_cap ∈ [0, 1]` is stored normalised: the effective cap is
`buoy_cap·physics.buoyancy_max` (§27.3), so a fork that changes `buoyancy_max` rescales every agent's cap without
invalidating any state. Founders and immigrants start at `mutation.buoy_init`, without a draw. A child inherits its
parent's cap with additive Gaussian noise and a clip, like `tag` (§12.4):

```text
cap_c = clip(cap_p + buoy_sigma · n₅, 0, 1)          n₅ ~ N(0, 1), the fifth draw of MUT_GENES
```

so both ends are reachable and can be left again (logit-space noise, as for `offspring_frac`, would make 0
absorbing). In 3D `MUT_GENES` splits into five whenever dims = 3, even with `buoyancy_max = 0`, so switching the
actuator on or off never shifts the other genes' draws; at `buoy_sigma = 0` the cap is inherited unchanged.
`buoy_init` and `buoy_sigma` are 0 when `buoyancy_max = 0` (§27.1). The cap's energy costs (`energy.buoyancy_cost`,
`energy.buoyancy_cap_cost`, booked in `HEAT_BUOYANCY`) come in PR 3.

**Placement** (`lifecycle.reproduce`, 3D branch; the 2D lines are unchanged).
- A child is placed at `D = offspring_distance` in a cone behind its parent, "behind" being the direction
  `(θ_p + π, −φ_p)`. With `u = U(−0.5, 0.5)²` from `PLACEMENT`, the yaw is `ψ = θ_p + π + u₀` and the pitch
  `χ = clip(u₁ − φ_p, ±PITCH_MAX)`. If `z_p + D·sin χ` leaves `[z_lo, z_hi]`, `χ ← −χ` (a reflection at the floor or
  the surface). Then `pos = wrap_xy_clamp_z(pos_p + D·(cos χ·cos ψ, cos χ·sin ψ, sin χ))`, `heading = wrap_angle(ψ)`
  and `pitch = χ`: the child faces away from its parent (`lifecycle._offspring_kinematics_3d`).
- The offset keeps its length D (by default the child bond's rest length, §8.4): `height − 2r ≥ 2D` (§27.1) puts
  the reflected point in range, and the final clamp only absorbs rounding. (A uniform direction with a z clamp could
  shorten the offset to 0, where the spring and the repulsion both vanish.) With a level parent and no pitch jitter
  the x and y of a child are §12.3's 2D rule (bit for bit on CPU) and its z is the parent's.
- Immigrants are uniform in the volume: `state.volume_kinematics` on the `IMMIGRANT` split's position and heading
  keys, as for the founders, with `buoy_cap = buoy_init`.
- Newborn rows also get `pitch` and `buoy_cap` (`ops.place`); `vel` (N,3) is cleared as in 2D. No invariant would
  catch a missed write (the previous occupant's value is in range), so a test fills a dead row with sentinels in every
  field of `AGENT_FIELDS_3D` and checks that a birth and an immigrant overwrite each.
- `lifecycle.deaths` stays 2D until PR 3, which sinks corpses to the floor cell.

**Catastrophe.** In 3D the catastrophe is a vertical column: `catastrophe_mask` measures the x/y minimum-image
distance from the (2,) `env_cat_center`, so every agent above the disc is in it, at any height. The director stays
rejected in 3D until PR 4, which also scales the radius bounds (docs/3D.md §12 Q10).

**RNG.** The 3D draws use the existing keys with 3D-only shapes and split counts (§14.1): no new stream, no
`RNG_LAYOUT_VERSION` bump, and every 2D draw is unchanged.

| key | 2D draw | 3D draw |
|---|---|---|
| `InitStream.POS` | `uniform(0, L, (n0, 2))` | `uniform((0, 0, z_lo), (L, L, z_hi), (n0, 3))` |
| `InitStream.HEADING` | `uniform(−π, π, (n0,))` | `uniform((−π, −PITCH_MAX), (π, PITCH_MAX), (n0, 2))`: yaw, pitch |
| founders' `buoy_cap` | — | `mutation.buoy_init`, no draw |
| `Stream.PLACEMENT` | `uniform(−0.5, 0.5, (M,))` | `uniform(−0.5, 0.5, (M, 2))`: yaw and pitch jitter |
| `Stream.MUT_GENES` | `split(·, 4)`: σ, p, tag, frac | `split(·, 5)`: σ, p, tag, frac, then `normal (M,)` for `buoy_cap`, whatever `buoyancy_max` |
| `Stream.IMMIGRANT` | `split(·, 5)`: pos `(M, 2)`, heading `(M,)`, tag, W1, W2 | the same split; pos `(M, 3)` and heading `(M, 2)` drawn as the founders'; `buoy_cap = buoy_init`, no draw |

**Tests.** `tests/test_state_3d.py` (the type and schema, compile, founders, their draws against the RNG table
above, dead rows, the shared 2D draws, the initial resource, `full_check`, round trips, cross-dims refusals, digests,
checkpoints, `make_scene`, `inspect`).
`tests/test_lifecycle_3d.py` (a level parent without pitch jitter against the 2D rule; the offset length, the
reflection at the floor and the surface, the facing and the cone; the child rows against the `PLACEMENT` and
five-way `MUT_GENES` draws, unstressed child bonds and untouched other rows; the cap gene against a numpy reference,
both clip ends and the same other genes with the actuator off; immigrants in the volume at `buoy_init`, drawn as the
RNG table says; sentinel completeness over `AGENT_FIELDS_3D` for a birth and an immigrant; the catastrophe column;
compile). `tests/helpers.volume_draws` writes out the table's founder and immigrant draws independently of
`state.volume_kinematics`; `tests/helpers.make_scene` builds a `State3` for a 3D config, and `tests/helpers.as_3d(cfg)`
is the tests' 3D variant of a 2D config (height 24, `chem_diffusion ≤ 1/6`, `benthic_lambda = 8`), built without
`validate` while dims = 3 is gated.

### 27.5 The static 3D lattice and the vertical food profile (PR 2)

**Shapes and index.** `resource` is (FZ, FG, FG) int32 and `chem` (NCH, FZ, FG, FG) float32, indexed `[z, y, x]`
with `FZ = height / field_cell`; layer `k` covers `z ∈ [k·field_cell, (k + 1)·field_cell)`. x and y are periodic
and z is bounded. `fields.cell_index` returns the z-major flat index `(cz·FG + cy)·FG + cx`, with x and y as in 2D
(the `% FG` seam guard) and `cz = clip(floor(z / field_cell), 0, FZ − 1)`, never wrapped. Layer 0's index equals the
2D index, and the floor cell below a position is `fcell % FG²`. The index fits int32 (`FZ·FG² ≤ 2²⁰`, §27.1).

**Vertical profile.** `fields.vertical_profile(cfg)`, (FZ, 1, 1) float32, by layer index:

```text
V[k] = min(1, exp(−k·fc / benthic_lambda) + photic_amp · exp(−(FZ − 1 − k)·fc / photic_lambda))     fc = field_cell
```

A benthic band at the floor plus an optional photic band under the surface; its term exists only when
`photic_amp > 0` (a static branch; `photic_amp = 0`, the default, is exactly the benthic profile). `V[0] = 1`
exactly, so the floor layer is the 2D fertility bit for bit. The profile is by layer index, not by z at cell centres.
It is built on the host from config constants (one float32 array; no matmul, F20).

**Fertility and regeneration.** `fertility` computes the (FG, FG) field of §13.1 unchanged (seasons and the
director's env included) and in 3D returns `fert[None]·V`, one float32 product per layer, never a (PP, FZ, FG, FG)
array.
`regen_rate` and the initial resource follow its shape. Harvest is not gated on grounding: a grounded agent is in
layer 0 (`r + floor_eps < field_cell`, §27.1). The 3D resource and chemical dynamics (no-flux diffusion in z,
deposits, gradients, sinking corpses) come in PR 3; until then `update_resource`, `update_chem`, `deposit_chem` and
`sample_fields` are 2D only.

**2D-body guards.** The 2D cell index asserts (N, 2) positions and 2D `sample_fields` a 2D lattice. The asserts are
static (free under `mx.compile`, the 2D trace unchanged), so a missed 3D branch fails loudly instead of dropping z
(the 2D index) or folding the layer into y (`_nbr_cells`).

**Tests.** `tests/test_fields_3d.py`: the z-major index against a reference, layer 0 against the 2D index, the z
clamp, the highest grounded z in layer 0, the guards, both profiles, the floor layer bit for bit (plain, seasonal,
director env) and the 3D regeneration.

---

## Review log

Each critic issue was verified against the text and, where it made a numeric claim, re-run in
`revise_proto/` (or the critic's `critic_proto/` scripts were re-run unchanged). One line per issue.

| # | issue | resolution |
|---|---|---|
| R1 | Converging crowds collapse past C; runaway overflow; `INT_HEADROOM` aborts emergent swarms (major) | **Fixed.** Reproduced (183–300 per hash cell). Added contact-attenuated thrust (§9, D26, `physics.thrust_contact_limit = 4`), verified size-independent for 300/1 000 agents with/without grip (F18); rewrote C sizing (§4); `INT_HEADROOM` now uses the real derived int32 bound `Dims.occ_limit` (1 060 at defaults, corrected per-agent bound) and stays an error because past it a scatter can wrap (§10.9); added `test_crowd.py`. |
| R2 | Insolvent agents can be rescued by share; text says they die (minor) | **Fixed** together with R7: insolvent agents always die (option b); solvent combat victims drained to 0 may be rescued by inflow, documented as intended (option a for that case, §10.7, §11), with tests. |
| R3 | Share/attack decisions applied to bonds changed at S9 (minor) | **Fixed.** `same = partner == s.partner` gate in share (§10.6); combat drops attacks on post-S9 partners unless `attack_bonded` (§11); signatures, wiring, tests, D28. |
| R4 | Tiny CPU step cost / fp16 in CI breaks budgets (minor) | **Fixed.** Re-ran `c7_tiny_cost.py` (confirmed). `tiny_cfg` is now f32, fp16 has targeted fast tests plus the full invariant suite nightly, `tiny` uses M = 32, stress variants 60 steps (150 under `slow`), invariant run 200 steps; CPU estimate restated from measurements (§22.0, F19, D16). |
| R5 | safetensors does not preserve key order (minor) | **Fixed.** Confirmed; `state_from_arrays` checks the key set and builds in `State._fields` order; digests iterate `_fields` (§2.2, §16.2, F15). |
| R6 | Carrying-capacity numbers too high (minor) | **Fixed.** Confirmed over 8 seeds: fertility 0.33–0.36, regen 7.0–7.5 M q/step, K_est ≈ 37–40 k (§10.10). |
| R7 | Insolvent agents do not always die (major; duplicate of R2) | **Fixed.** `dead = alive & (~solvent \| E == 0 \| aged)`, cause `STARVED` first; `solvent` added to `lifecycle.deaths` signature and wiring; inflow joins the corpse, ledger unchanged; test added. |
| R8 | `device_check` cannot detect every corrupted invariant (major) | **Fixed.** Diag coverage listed (§22.3); "both checks" assertion restricted to Diag-covered corruptions, the rest are `full_check`-only; corrupted states are fed with the healthy `prev_total` and a zero ledger (§22.2). |
| R9 | WP4 unit work secretly depends on WP2/WP3 (major) | **Fixed** (option a): `drop_dead`, `add_child_bonds`, `symmetric_ok` and `mutate_rows` join the WP0 hand-over (§25). |
| R10 | Bond offset inputs can exceed 1 (minor) | **Fixed.** Clipped to [−1, 1] (§5); test with a 4.5-length bond. |
| R11 | bf16 breaks every numpy path (minor) | **Fixed.** Confirmed; `state.to_numpy` widens bf16 → f32 on device and is the only host read path (§16.1–16.2, F17); `tiny_cfg_bf16` smoke test. |
| R12 | Resume can change `run.compile` / `runtime_checks` and silently break bitwise identity (minor) | **Fixed.** `run.seed`, `run.compile`, `run.runtime_checks` are fixed per run directory; resume accepts only observer/cadence `run.*` keys (§14.2, §18); tested. |
| R13 | Viewer `s` writes a checkpoint that breaks run-dir consistency (minor) | **Fixed.** `s` writes a standalone `media/snap_<step>.safetensors` via `runio.write_state_file`; the viewer never touches checkpoints, LATEST, metrics or events; forks accept snapshot files (§20.2, §18, §21). |
| R14 | Fork semantics underspecified; "mutation frozen" impossible (minor) | **Fixed.** Kept/reset fields, key reseeding with `--seed`, `n_init` must match, admission `full_check` under the new config, config source = the state file's embedded config; "mutation frozen" removed (§18). |
| R15 | `full_check` lacks the inputs its checks need (minor) | **Fixed.** `full_check(s, cfg, prev=None, out=None)`; per-step ledger from `ledger_cum` difference; births from `out`; dead-uid set kept by the harness (§16.2, §22.2). |
| R16 | Kin counters have no owner; `KIN_BOND_BASE` undefined (minor) | **Fixed.** `finish` owns them with explicit `desc`, `x`, `act`, `as_prop`; `Combat` returns `attack`/`hit` masks; added counter 25 `BOND_PROPOSERS` as the `kin_bond_bias` denominator (N_COUNTERS = 26; §11, §16.3, §19). |
| R17 | `Share` field shapes ambiguous (minor) | **Fixed.** `out` and `heat` are (N,) per-agent sums over slots (§16.2, §16.3). |
| R18 | Initial uids, `next_uid` and dead-row init values unspecified (minor) | **Fixed.** Founder `uid = arange(n_init)`, `next_uid = n_init`, explicit dead-row init values (§2.2); test_state asserts them. |
| R19 | Count × constant ledger terms and `S·n_alive` need int64 (minor) | **Fixed.** Confirmed int32 wrap at 262 144·10 000; int64 terms specified in §10.8 and `finish`; test with `n_alive·S > 2³¹`. |
| R20 | Padding-lane parent: code vs prose (minor) | **Fixed.** `parent = where(kind == BIRTH, perm[ppos], 0)` (§12.2); test pins row 0. |
| R21 | Two `vel` tolerances (minor) | **Fixed.** `full_check` uses `v_max·(1 + 1e-6)` (§22.2). |
| R22 | `test_determinism.py` owned by two packages (minor) | **Fixed.** Split into `test_determinism.py` (WP4) and `test_resume_purity.py` (WP5) (§22.4, §25). |
| R23 | WP4 tests exceed the 60 s per-package limit (minor) | **Fixed.** Fewer fast-suite steps (see R4), eager comparison moved to `slow`, WP4 budget 120 s, total still < 3 min (§25 rule 3). |
| R24 | `resume --allow-config-change` duplicates fork (minor) | **Fixed.** Removed, with `config.<step>.json`, `meta.replayable` and the load flag; `load_checkpoint` keeps only `check_hash` for forks (§18, §21, D27). |
| R25 | "Every numeric constant lives in config or layout.py" is false (minor) | **Fixed.** Named `LR_SCALE`, `AGE_SCALE`, `RES_SAT`, `RES_GRAD_GAIN`, `EPS_LEN`, `C_REF_MARGIN` in `layout.py`; `OCC_LIMIT` became derived `Dims.occ_limit`; the claim is narrowed (render colours stay local) (§5, §9, §16.2, §17). |
| R26 | SIGINT/extinction checkpoints contradict "metrics boundaries only" (minor) | **Fixed.** Unscheduled stops write a partial interval row with its `steps` count, flush, and checkpoint; resumed first interval may be short; rows carry `steps` (§18, §19.2). |
| R27 | Symbol M overloaded in D17 and fatal-flaw list (minor) | **Fixed.** Replaced with C. |

Totals: 27 issues, 27 fixed (R2 and R7 are the same defect), 0 rejected.

**Implementation notes** (deviations found while implementing; one line each).

| # | note |
|---|---|
| I1 | impl: (WP0) The pending-buffer invariant misfired at step 0 (founders have `birth_step = −1 = step − 1` but are not in `pend_slot`); it now applies for `step ≥ 1` (§12.6, §22.2). |
| I2 | impl: (WP0) `validate` also checks basic sanity (positive sizes, counts ≥ 1, seed range) so that every listed rule and `stability_number` are well defined (§17). |
| I3 | impl: (WP0) `wrap_angle` lifts `−float32(π)` (which is `< −math.pi`) by one ulp so the range holds in float32 and float64 checks; `first_free` returns slot 0 for full rows; `add_child_bonds` argument semantics written out (§16.2). |
| I4 | impl: (WP0) `proportional_take` clamps the f32 ratio to `ONE + 1` before the int32 cast; uncontested pools ignore that value, so results are unchanged but the cast can no longer overflow (§10.4). |
| I5 | impl: (WP2) `FieldObs` contents pinned: `sample_fields` returns the final §5 values of rows 12–23 and `build_inputs` copies them, so WP1 and WP2 cannot both (or neither) normalise (§5, §16.2). |
| I6 | impl: (WP2) `q0 = max(basal_cost, 1)` and zero `attack_max`/`harvest_max`/`share_rate` read as 1 in the §5 inputs, so `x` stays finite for every valid config; identical at the defaults (§5). |
| I7 | impl: (WP2) `impl = "einsum"` casts `z` to `wdtype` like `matmul` (§6); `IN_CLOCK` takes its phase from `age mod clock_period` (exact for large ages, same value); sensing also ANDs `nbr.valid` with both ends' `alive`, and targeting with the neighbour's (no-ops under the §4 contract). |
| I8 | impl: (WP1) §13.2's "diffusion is an average" was imprecise under floor rounding; `R ≤ R_HARD` still holds exactly because `R_HARD ≡ −1 (mod 256)` (proof in §13.2, property-tested for `d_q8 ∈ [1, 64]`). No code change. |
| I9 | impl: (WP3) `form_bonds` also drops proposals to self or to a current partner (a no-op under the §7 targeting contract; without it a self-target would pass the mutual test and self-bond); `bond_geometry` gives empty slots `off = 0`, `length = 0`; `physics.forces` returns 0 for non-survivors (§8.3, §9). |
| I10 | impl: (WP4) `Deaths`/record field semantics written out (§16.2); `allocate_lanes` clamps `room` at 0 (identical while the cap holds); `device_check` recomputes `TOTAL` from the state rather than reading `s.total`; `full_check` additionally checks shapes/dtypes, that `pend_slot` rows are dead or last step's newborns (the next S0 overwrites them), non-negative ledgers and, given `prev`/`out`, `step + 1`, `out.ledger` and `pend_slot` vs the birth slots (§22.2). |
| I11 | impl: (WP5) Bodies add root hooking to label propagation (a shuffled 2 000-chain needs O(length) rounds without it, ≈ 10 with it); `weight_diversity` keeps founders eligible at step 0 and pins the sample to the first eligible rows of an observer permutation; level counters are interval means; event chunks are half-open `[s0, s1)` and `truncate_after` also deletes later checkpoints; `E[founder brain_cost]` is analytic (§10.10, §18, §19.2). |
| I12 | impl: (WP6) §20.1 left screen orientation, seam and zoom-out wrapping, bond sample spacing, the violet and energy-ramp colours, the nose colour, large-stencil memory and `VideoWriter` path handling unspecified; they are pinned in §20.1 "Pinned details". |
| I13 | impl: (WP4) Combat drains a victim whose attackers together demand at least all it holds: `proportional_take`'s −1 guard otherwise left it ≥ 1 quantum, so it survived S14 and died insolvent a step later as `STARVED`, making `KILLED` practically unreachable. The remainder is `Combat.residue` (per victim), spilled into the victim's cell and booked as `FLOW_SPILL` (§10.9, §11, §16.2, §16.3). `golden` also pins the platform; measured tiny step time recorded (§22.0, §22.4). |
| I14 | impl: (WP6) The donation-health rule of §15.4 ("peak − baseline > 0.5 × weight bytes") misfires: step transients alone are ≈ 0.64× the fp16 weight bytes on CPU (tiny and dev), and a retained old state lowers the excess because it sits in the baseline. Bench now calibrates with one step that keeps its input alive and warns unless that step peaks > 0.25 × weight bytes higher. Viewer controls, HUD interval and snapshot paths pinned (§20.2); bench stage groups and variant timing pinned (§21); `ViewerSession` and `bench.format_report` added (§16.2). MLX 0.32 `mx.compile` fails (`_Map_base::at`) when a constant array is an output beside arrays that use it (`Lanes.n_imm` with immigration off); the staged bench returns the lane counts stacked. |
| I15 | impl: (WP5) Runner and CLI details pinned: `meta.start_step` and the default resume target, event chunks per checkpoint interval, the violation path (rows and events flushed, `diag/` dump, no checkpoint), extinction read from the evaluated per-step `N_ALIVE`, shortcut-flag precedence, state files accepted by `render`/`inspect`/`view`, `inspect --json` keys (§18, §21). |
| I16 | impl: (gate) Release check: the §3 S11 row now lists `residue` (I13); `bench` reports `n_alive_start` at the start of the timed window and adds `immigrants`, so start + births + immigrants − deaths = end; `format_report` takes `report` as in §16.2. No behaviour change to the step. |
| I17 | rev: §10.7/§17 — `validate` checked `floor(Dp_min·frac_min) ≥ 1` on the unquantised fraction, so `frac_min < 0.5/256` (frac_q8 = 0: child alive with E = 0) or `frac_max ≥ 255.5/256` (frac_q8 = 256: parent left with E = 0) passed and later tripped `ENERGY_RANGE`. Now `frac_q8(frac_min) ≥ 1`, `(Dp_min·frac_q8(frac_min)) >> 8 ≥ 1`, `frac_q8(frac_max) ≤ 255` (f32, as the engine computes it) and `e_init ≥ 1`. Defaults unchanged. |
| I18 | rev: §18 Fork — a fork that changed `energy.structure` was always refused ("recomputed TOTAL != state.total"): the conserved total moves with S because structure energy is in no `E`. Reset is now `total0 = total = loaded total + (S_new − S_old)·n_alive`; a loaded total that was already wrong is still refused. |
| I19 | rev: §17 — `immigration_floor > 0` with `immigrants_per_step = 0` admitted no immigrant yet disabled the extinction stop, so a dead world ran to `run.steps` as `done`. `validate` now requires `immigrants_per_step ≥ 1` when immigration is on. |
| I20 | rev: §22.2 — eight relational `full_check` rules (resurrection by an old uid, `next_uid` decreasing, births > min(free, M), duplicate/out-of-range/dead/reused birth slots, newborn `birth_step`, `pend_slot` vs births) and ten host-only rules (uid ≥ `next_uid`, partner/rev index range, partner/rev disagreement, duplicate partner, negative cumulative/per-step ledger, `RESERVED`, negative `brain_cost`, non-finite `patch_vel`, negative `step`, `pend_slot` range, per-step closure) had no corruption case. Cases added, plus a meta-test that every `r.check`/`r.rows` site in `full_check` is hit by a case whose fragment matches only that rule. |
| I21 | rev: §10.9/§22.3 — no test checked what the step passes to `device_check`. An eager-step test now compares the recorded `max_cell_occ`, `max_attackers`, `overflow_cells` and `v_clips` with independent counts and asserts `VCLIP_WARN`; a 1 100-agent single-cell scene asserts `INT_HEADROOM` from a real step. |
| I22 | rev: §22.0 CI — the test job installs the `viewer` and `video` extras (the pygame loop test always skipped before), and both jobs cache MLX's CPU JIT directory (`TMPDIR=$RUNNER_TEMP`, `$RUNNER_TEMP/mlx`): a cold cache measured ≈ 280–310 s for the fast suite against the 3 min budget. |
| I23 | rev: §20.1 — `render_frame` splatted every row's full disc stencil, so cost grew with zoom² × N (dev, 900², zoom 64: 36 s per frame). `_splat` now culls invalid rows and rows whose stencil box misses every periodic image (one host read, conservative, frames unchanged) and `_disc_stencil` is vectorised: 0.04 s at zoom 64, 0.09 s at zoom 256. |
| I24 | rev: §20.2 — the viewer ran up to 1 024 steps per frame without pumping events (tens of seconds at 64k–256k). A frame now stops stepping after `FRAME_BUDGET_S = 0.1` s (at least one step); the trajectory is unchanged. |
| I25 | rev: packaging — `requires-python` had no upper bound and mlx 0.32.x has no cp315 wheels or sdist, so uv could pick 3.15 and `uv sync` failed. Now `>=3.12,<3.15` (uv.lock relocked: only cp315 wheel entries dropped) and `.python-version` pins 3.12 (the lower bound, so CI covers it); README states the range. |
| I26 | metal: first run on the M5 Pro: 459/464 fast tests passed. Four float-precision failures, two causes. (1) GPU float32 matmul is reduced precision (F20): `fertility` now sums `gy_p ⊗ gx_p` elementwise, and `weight_diversity` centres the sample, uses exact elementwise norms and runs its Gram on the CPU stream (new close-kin accuracy test; it fails against the old code on CPU too). (2) Fused Metal kernels round differently from eager ops: `test_lifecycle_compiles_and_matches_eager` compares floats to 1e-6 on GPU and stays bitwise on CPU; integers stay exact everywhere. |
| I27 | feat: two optional levers, both off by default and invisible to old runs. (1) Periodic immigration (§12.7): `population.immigration_every` / `immigration_count` add up to `count` random founders on steps with `(t+1) % every == 0`, combined with the floor rule by `max` and capped by free lanes and capacity; same lanes, `IMMIGRANT` stream, clade, ledger term and counters; static branch plus array arithmetic on `state.step`. (2) Seasons (§13.1): `fields.season_period` / `season_amplitude` / `season_mode` weight each patch by `1 + A·sin(2πt/P + φ_p)` (φ_p = 0 or 2πp/PP); `fertility` / `regen_rate` / `update_resource` take an optional `step`, regeneration stays integer and `INJECT_REGEN`-accounted, the carrying-capacity estimate uses the time average. Compatibility: keys added after v0.1 are omitted from `config_hash` at their defaults, so existing run directories load with defaults and resume; the tiny hash and the golden digest are unchanged (`tests/test_features.py`). |
| I28 | feat: director build step 1 (`docs/DIRECTOR.md` §3): the engine's live environment state, off by default and invisible to old runs. (1) Nine flat `State` fields `env_*` appended after `diag` (regen multiplier, season amplitude and period, drift multiplier, per-patch gain, immigration request, catastrophe centre, radius and kill probability; `State.env` is a read-only `Env` view, `default_env(cfg)` the inert values), read by the compiled step only behind the static `director.enabled` (`DirectorConfig`, `POST_V01_KEYS`; `config_hash` also drops a section the POST_V01 deletions empty). Enabled with inert values is bit-identical to off. (2) New `DeathCause.CATASTROPHE` (3) and counter `DEATHS_CATASTROPHE` (index 26, `N_COUNTERS` 27; `metrics.jsonl` rows always carry `deaths_catastrophe`), on a new RNG stream (`Stream.CATASTROPHE = 8`, `RNG_LAYOUT_VERSION` 2 with `RNG_COMPATIBLE_VERSIONS = {1, 2}`: loading accepts the set, streams 0–7 are unchanged). Catastrophe deaths are excess only (every other cause keeps precedence, so `INSOLVENT ≤ DEATHS_STARVED` holds), the corpse takes the ordinary path, no new ledger term. (3) `env_imm_request` is a third immigration demand combined by `max`; the step carries over `max(request − n_imm, 0)`, so a large pulse arrives over several steps; `env_cat_kill_p` resets to 0 after the step it applied in (`step.consume_env`, outputs derived from inputs for the MLX 0.32 compile quirk). `immigration_possible` widens the two static immigration gates. (4) `state_digest` still hashes all fields (now stricter); `legacy_state_digest` hashes the 38 v0 fields, and the golden pin, unchanged, uses it. Pre-director checkpoints (38 arrays, RNG layout 1) load with inert env (`state_from_arrays(d, cfg)`); env ranges are checked by `full_check` always and by `device_check` (`FIELD_RANGE`) when enabled, including the start-of-step values of the consumed fields. Runner: the extinction stop also waits for a pending `env_imm_request`. Tests: `tests/test_director_env.py` and the edits listed in §22. |
| I29 | feat: director build step 2 (`docs/DIRECTOR.md` §10): the host-side director core, no LLM, no change to the compiled step. (1) Package `alife.director` (vitals, archive, actions, pacing, rules, log, controller): vital signs from `metrics.jsonl` rows, a behaviour archive with novelty and a coverage grid, the §5 menu with budgets, cooldowns and timed reverts, the safety rules and the three personas, `director.jsonl` (header, entries with explicit clamped params, revert entries) and `journal.jsonl`, replay. (2) Runner: optional `director` hook after the metrics row and the violation check, before the checkpoint; never on a violated or extinct state; the checkpoint holds the applied env. (3) Config: `run.director` (off/live/replay, a resume key) and the policy keys `director.persona|goal|rule_every|budget_per_100k`, which are unhashed at every value (`UNHASHED_KEYS`); `validate` ties `rule_every` to the metrics cadences; all earlier hashes, `TINY_HASH_V01` included, are unchanged. (4) CLI: `--director`, `--replay-log`; resume accepts the policy keys; a fork resets the env to `default_env` and starts with the director off; a bad replay log is exit 2 before anything is written. (5) Metrics: additive `n_alive_min` and, with the director enabled, an `env` summary; `carrying_capacity` takes optional `step`/`env`. (6) `RunDir.append_jsonl/read_jsonl/write_json`, `truncate_after(keep_director=)`. Deviation from the proposal: seasons have no runtime mode (only period and amplitude), `mutation_floor_nudge` and archive immigrants are not implemented, the damper's capacity is `N_CAP`. Replay of a log is bitwise (§26.8); viewer and `render` re-simulate without the director. Tests: `tests/test_director_*.py` and the edits in §26.9. |
| I30 | feat: director build step 3 (`docs/DIRECTOR.md` §7): the LLM tier, off by default, no change to the compiled step or to any earlier hash. (1) `director.llm_every` (0 = off, else a multiple of `rule_every`), `llm_temperature`, `llm_timeout_s`, `llm_max_tokens` (policy keys: `POST_V01_KEYS` and `UNHASHED_KEYS`) and `run.llm_url`, `run.llm_model` (resume keys); CLI `--llm`, `--llm-every`, `--llm-url`, `--llm-model`, `--persona`, `--goal` on `run` (fork included) and `resume`, which exit 2 unless the effective mode is `live`. (2) `director/llm.py` (the JSON schema generated from `actions.MENU` with `minimum`/`maximum`/`maxLength`, the untrusted-reply validator, a stdlib client that follows no redirect, ignores proxies, caps size and wall time and never raises) and `director/prompt.py` (persona, goal as delimited data, menu with effect descriptions, vital signs and trends, this boundary's safety actions, the last five interventions with their measured effect, budget). (3) Roles: the rule tier keeps reverts and the safety rules at every boundary (`rules.decide_safety`); at `t % llm_every == 0` the model makes the discretionary moves and the rule tier's discretionary rules (`decide_discretionary`) are only the fallback of a failed, invalid or fully refused call; the call is synchronous inside the existing hook. With the tier off the director calls `rules.decide` as before: same decisions, same log, `RULES_VERSION` unchanged. (4) `source: "llm"` entries (with `llm_call`), accepted by `validate_entry` and `validate_log`; one `llm_call` record per call in `journal.jsonl` (raw prompt and response, outcome, clamps, rejections, fallback); the header records `llm_every`/`llm_model` only while the tier is on. Replay never builds a client. Deviations from the proposal: synchronous instead of async (reply applied at the same boundary), `fertility_shift` offered to the model in its gain form only, no `--no-llm` (use `--llm-every 0`). Tests: `tests/test_director_llm.py`, the `llm` marker, and the edits in §26.9 (two existing tests used `llm` as their example of an unknown source). |
| I31 | feat: director build step 4 (`docs/DIRECTOR.md` §14): the GUI Director panel, plus one CLI command. `alife llm-check` (§21) sends one tiny structured-output request under the director's own `llm.Client` and `response_format` and prints `OK model=... latency=...s` (exit 0) or `FAIL <kind>: <detail>` (exit 1, `ExitCode.ERROR`; exit 2 for a malformed URL, empty model or non-positive `--timeout`). No engine, config or hash change; no new config key; Python changes are `cli.py` and `tests/test_cli_llm_check.py` (mock-server cases for OK, HTTP error, empty content, invalid JSON, schema violation, timeout, closed port). |
| I32 | feat: the `random` persona for evaluation arm R (`docs/DIRECTOR.md` §9, §15), yoked to another run's discretionary moves. (1) `director.persona = random` (`PERSONAS`) and the policy key `director.yoke_log` (`POST_V01_KEYS`, `UNHASHED_KEYS`: no hash changes); `--yoke-log PATH` on `run`, fork and `resume`; `validate` ties them together (a yoke log and `llm_every = 0` for `random`, no yoke log for any other persona). (2) `director/yoked.py`: `yoked_steps` (discretionary entries of a log per step: `source` rule or llm, not a safety rule, never a revert; `rules.SAFETY_RULES` are named constants shared with the rules), `load` (exists, replay pre-validation, steps on R's boundaries, sha256), `candidates` (the eligible menu actions via `rules.eligibility`), `sample_params` (uniform and log-uniform over the live menu bounds) and a stateless per-draw generator. (3) `Director`: R uses the Gardener's `PersonaParams` and keeps reverts and the safety rules; at a yoked step it makes the same number of random eligible moves, or journals "nothing eligible"; entries `rule: random`, `persona: random`, `yoked_to`; the header records `yoke_log` and `yoke_sha256`; a live resume warns on stderr when the file changed. Replay never reads the yoke log. (4) The CLI checks the yoke log (exit 2) before any directory is created or touched; the GUI offers Random with a "Yoked to run" picker. No engine, step or hash change; every other persona is byte-identical. Interpretations: the sampler returns `(raw, duration)` like a rule's proposal; the patch of a gain change or a loss is drawn among the free ones (a draw cannot be refused for a locked patch); a new header is appended when a resume finds another yoke file; the one existing test that pinned `PERSONAS` was updated. Tests: `tests/test_director_random.py`. |
| I33 | feat: offline evolutionary-activity analysis (§19.3, `docs/ACTIVITY.md`), no change to the engine, the step, any config key or hash, or the golden digest. (1) Package `alife.analysis` (host-only numpy, never imported by the step, outside the §22.6 lint scope): `phylo` (the phylogeny rebuilt from the event logs by streaming the chunks, with a coverage report against `metrics.jsonl` and the final checkpoint; pointer-jumping lineage labels; Bedau's neutral shadow and a permuted heritability null), `activity` (one pass over grid windows: cohort-subtree and paraphyletic marker components; Bedau–Packard `D`, `A_cum`, `Ā`, `A_new` histograms; Channon's component-normalised activity against a reset shadow, computed in expectation; threshold crossings), `report` (classification none/bounded/unbounded per window with every threshold swept, `summary.json`) and `plots`. (2) CLI `alife activity RUN` (§21), writing to `RUN/activity` by default and never modifying the run's files. (3) Optional extra `analysis = ["matplotlib>=3.9"]` (§16); without it the command writes tables only. Deviations from the 1998 paper, documented in the note: the primary `A_new` window is `[a', 10^0.2 a']`, because the literal `a' ± 0.05 (a_max − a')` degenerates on a heavy-tailed run; the asymptotic definitions are applied as trends over finite windows. Limitation: runs that start at step 0 only (forks need founders from a checkpoint). Tests: `tests/test_analysis_activity.py`. |
| I34 | feat: offline signal-information analysis (§19.3, `docs/SIGNALS.md`), no change to the engine, the step, any config key or hash, or the golden digest. (1) `alife.analysis.mi` (numpy only): KSG estimator 1 on ranks with random tie-breaking (exactly MI-preserving; the integer, zero-inflated signals and targets made the Gao et al. 2017 mixed estimator biased by about +0.07 bits), a Miller–Madow binned cross-check, shuffle nulls with z and permutation p, conditional MI over strata. (2) `alife.analysis.resim`: in-memory re-simulation to steps without a checkpoint, every `metrics.jsonl` row compared exactly, states written with `runio.write_state_file` to a cache outside the run (`analysis_cache/`, git-ignored). `runs/64k_s0` changed dynamics at its 120k resume (I26's `fields.fertility` change), so 0–120k is re-simulated with the run's original commit `f09b32a`; all 10,000 rows to 1M match. (3) `alife.analysis.signals`: `decode` mirrors S0–S8 of `step.step` (a test checks it against the stored `msg` and the `EMITTERS` counter), per-agent tables, the estimate grid and markdown/trend tables; entry point `python -m alife.analysis.signals`, no `alife` subcommand and no new dependency. Tests: `tests/test_analysis_mi.py`, `tests/test_analysis_resim.py`, `tests/test_analysis_signals.py`. |
| I35 | feat: the `random` persona inherits the pacing parameters of the persona that made its yoke log (§26.11, `docs/DIRECTOR.md` §9, §15), so that arm R can be yoked to the Trickster arm C (budget 6 per 100k) as well as to B. Before, `yoked.PARAMS_PERSONA = "gardener"` fixed R's budget, cooldown scale and thresholds, capping it at the Gardener's maximum of 4. (1) `yoked.inherited` reads the persona and the effective `budget_per_100k` of the yoke log's first header; `yoked.load` returns them (`Yoke.persona`, `.budget`, `.personas`) and refuses (`LogError`, exit 2, before anything is written) a header persona of `random` or an unknown name; `PARAMS_PERSONA` became `FALLBACK_PERSONA` (the Gardener, for a log with no header or no persona). (2) Budget: an explicit `director.budget_per_100k` is capped by the inherited persona's maximum, `0` takes the header's budget, else the persona's default (`yoked.effective_budget`). (3) R's header gains `params_persona`; a replay takes persona and budget from its own first header. Only budgets, cooldowns and safety-rule thresholds are inherited; the boredom rule, minimum gap, drama gating and the model stay off. R yoked to a Gardener log makes the same entries and reaches the same state digest as before (checked against the previous code); its header has the new field, and with `budget_per_100k = 0` it takes B's recorded budget rather than the Gardener default (the same for a default B). Interpretations: a yoke log that was resumed with another persona uses the first header's persona (stderr warning), not the persona in force at each step, since R's budget window and cooldowns are one policy per run; the persona check covers every header. No engine, step, config key or hash change. Tests: `tests/test_director_random.py`. |
| I36 | feat: 3D world PR 1, dims plumbing (`docs/3D.md` §10 item 1; new §27.0 and §27.1), no behaviour change: `validate` still rejects `world.dims = 3` (`config.DIMS_SUPPORTED = (2,)`), and every 2D trajectory, config hash, metrics row and checkpoint and `meta.json` key set is unchanged (golden digest not re-pinned, no `FORMAT_VERSION` or `RNG_LAYOUT_VERSION` bump; a CPU digest matrix of 2D trajectories and runs matched bit for bit). (1) `world.dims` (2) and `world.height` (0) in `POST_V01_KEYS` and the new `THREE_D_KEYS`; `is_3d`, `require_2d`; `Dims` gains `nd`, `d3`, `lay`, `fz`, `n_cells`, `field_shape`, `g_hash_z`, `cell_hash_z`, and `q`, `w1_rows`, `params` go through `lay`; the world-dimensions validation (2D: the 3D keys keep their defaults; 3D: height, FZ, FZ·FG² ≤ 2²⁰, G_z, `2·height ≤ size`, `K ≤ 27·C`, `chem_diffusion ≤ 1/6`, `resource_diffusion_q8 ≤ 42`), every 2D message verbatim; only finite ratios are rounded (a non-finite `size` or `height` is a `ConfigError`, not an `OverflowError`). (2) `layout.Layout`, `for_dims`, `genome_layout_version` and `GENOME_LAYOUT_VERSION_3D = 2` with the frozen append-only 3D tables (D_IN 84 with inputs 69–83 ending in `IN_HEIGHT`, `IN_PITCH`, `IN_GROUNDED`; D_OUT 39 with `OUT_PITCH`, `OUT_BUOYANCY`, `OUT_Q_BOND_DZ`, `OUT_Q_ATTACK_DZ`); the D_IN/D_OUT reads in brain, sensing, metrics, bench, state and the schema go through `dims(cfg).lay` (equal Python ints in 2D); D18 amended. (3) `state.expected_layout` is the one State schema (`runio._expected_layout` stays as an alias for `tests/test_runio.py`; the private `invariants._expected_layout` is removed); `LedgerIdx.HEAT_BUOYANCY` alias of slot 15, `sinks(nd)`, `ledger_names(nd)` (definitions only). (4) `dims` metadata only when dims ≠ 2 (absent = 2D); `load_checkpoint` refuses a cross-dimension file; a fork refuses a `world.dims`/`world.height` change (exit 4); `render`, `view`, `render_frame`, the viewer and `analysis.signals` refuse a 3D config before loading (`require_2d`, exit 2). (5) `tests/helpers.make_scene` takes the pos, vel, W2, resource and chem shapes from the config's state (2D unchanged). Interpretations: `actions.decode` keeps `layout.F` (the 2D query block width in both dims; PR 2 appends the z component); `cli._agent_info` and the `make_scene` State class wait for `State3` (PR 2). Tests: `tests/test_layout_dims.py`, `tests/test_dims_config.py`, `tests/test_dims_runio.py`. |
| I37 | feat: 3D world PR 2, the geometry core (`docs/3D.md` §10 item 2; new §27.2–§27.5), still gated: `validate` rejects `world.dims = 3` (`DIMS_SUPPORTED = (2,)`) and the step does not call the 3D code yet (PR 3); every 2D trajectory, config hash, metrics row and checkpoint key set is unchanged (golden digest not re-pinned, no `FORMAT_VERSION` or `RNG_LAYOUT_VERSION` bump; the CPU digest matrix of 2D trajectories matched bit for bit). (1) Config: `physics.gravity`, `floor_eps`, `buoyancy_max`, `mutation.buoy_init`, `buoy_sigma`, `fields.benthic_lambda`, `photic_amp`, `photic_lambda` (all 0, in `POST_V01_KEYS` and `THREE_D_KEYS`, rejected off default in 2D, no 2D message changed); the 3D rules of §27.1 (room to reflect a child, `gravity < thrust_max·sin(PITCH_MAX)`, `r + floor_eps < field_cell`, the 3D load rule, the buoyancy keys 0 without the actuator, the profile keys, the director rejected until PR 4); `Dims.z_lo`, `z_hi`, `z_ground` as float32-exact host values (`z_ground` the largest float32 `≤ r + floor_eps` and at least `z_lo`, so a floor agent is always grounded); `layout.PITCH_MAX = 80°`. (2) State: `State3` (State's 47 fields, then `pitch`, `buoy_cap`), `EXTRA_3D_FIELDS`, `AGENT_FIELDS_3D`, `state_cls`, `agent_fields`; type-generic schema, (de)serialisation and digests (`legacy_state_digest` 2D only); founders uniform in the volume (`volume_kinematics`), dead rows at (0, 0, z_lo), `buoy_cap = buoy_init`; `load_checkpoint`, `full_check`, `inspect` and `tests/helpers.make_scene` take `State3`. (3) Geometry: `ops.wrap_xy_clamp_z`, `body_axes`, `to_body3`; `spatial._find_neighbors_3d` (G²·G_z cells, z clamped and never wrapped, a 3×3×3 stencil whose out-of-range layers are empty, Q = 27C); the 8-wide descriptor ending in `NBR_DZ`; 3D queries in `decode`, and `Motion3`/`decode_motion3`. (4) Physics: `grounded`, `forces3` (thrust along the 3D forward axis; gravity and buoyancy as body forces that contacts do not attenuate), `integrate3`/`Integrated3` (grip on x and y only and only when grounded, the 3D `v_max` clip before the clamp, floor and surface clamps that move the agent to the boundary and zero only the stored `v_z`, counted; the pitch clip); `bonds` works on (N,B,3) unchanged (docstrings only). (5) Lifecycle: the buoyancy-cap gene (`split(MUT_GENES, 5)` whenever dims = 3, additive Gaussian noise with a clip), children in a cone behind the parent with vertical reflection, immigrants uniform in the volume at `buoy_init`, the `pitch`/`buoy_cap` row writes, the vertical-column catastrophe. (6) Lattice: `resource` (FZ, FG, FG) and `chem` (NCH, FZ, FG, FG), the z-major `cell_index`, `vertical_profile` (V[0] = 1; the photic term only when `photic_amp > 0`), `fertility` and `regen_rate` over layers (the floor layer is the 2D fertility bit for bit). (7) Static guards in the 2D bodies that would otherwise take 3D input silently: last-axis asserts in `to_body`, `cell_index`, `find_neighbors` and `integrate`, and a lattice-rank assert in `sample_fields`, each with a refusal test. Interpretations: no `ops.min_image_xy` and no `bonds` z flag, since `2·height ≤ L` (§27.1) makes `min_image` the identity on z; the buoyancy costs, the field dynamics, sinking corpses and the step wiring are PR 3, and the scaled catastrophe radius bounds PR 4; the PR 1 3D fixtures of `tests/test_dims_config.py` and `tests/test_dims_runio.py` gained `fields.benthic_lambda` (now required in 3D), the `THREE_D_KEYS` pin checks the PR 1 prefix, and one BAD_3D case uses values that keep it the only problem. Measured: C = 16 and K = 8 do not hold 3D crowds (§27.3: peak cell occupancy 28 to 37 at C = 64 and up to 10 contacts per agent; overflow and collapse at C = 16), to be sized in PR 6. Tests: `tests/test_dims_config_pr2.py`, `tests/test_state_3d.py`, `tests/test_fields_3d.py`, `tests/test_ops_3d.py`, `tests/test_spatial_3d.py`, `tests/test_sensing_3d.py`, `tests/test_actions_3d.py`, `tests/test_physics_3d.py`, `tests/test_lifecycle_3d.py`, `tests/test_crowd_3d.py` (slow). |
| I38 | feat: `alife activity` reads forks, and the director evaluation (`docs/DIRECTOR.md` §9, `docs/EVALUATION.md`, §19.3, §21), no change to the engine, the step, any config key or hash, or the golden digest. (1) `phylo.load_tree(run, start=)`: a fork's founders are the agents alive in its checkpoint at `meta.json` `start_step` (source uids kept; parents and deaths mapped by uid), and `start` restricts any tree to the agents alive at that step and their descendants (`phylo.restrict`); step indices count from the start (report `t0`, `origin`), the stored grids are run steps (`report.compute(step0=)`), and the windows and the cohort-pool settling start at `t0`. Founders may be parents from step 0 (`phylo.eligible_from`, used by the exposures and the neutral shadow) and `min_gap(founder_ages=False)` ignores their children, since their ages are not recorded; on a run from step 0 both change nothing (no birth before the founders' minimum age): on `runs/64k_s0` all 820 arrays of `activity.npz` and the summary are identical. Catastrophe deaths count in the death totals. CLI `--start STEP` (§21). This lifts I33's limitation. (2) `alife.analysis.evaluation` (entry point `python -m alife.analysis.evaluation`) and `evaluation_plots`: per arm and seed, the director's vital signs refolded with `Tracker` from `metrics.jsonl` (equal to the logged vitals of every live entry), coverage and cells outside the control, novelty and excursion from the control, persistent novelty against the control and against the pre-intervention level, turnover, viability against a fixed per-seed reference population (base run, 70k–120k), clades, lineages from the event logs, intervention counts, entropy and cost, the LLM's validity, clamp rate and latency, the activity classes, a seeded journal sample and a Markdown table. Tests: `tests/test_analysis_activity.py`, `tests/test_analysis_evaluation.py`. |

**Open questions for the user.**
1. Contact-attenuated thrust makes a contracted body's interior agents (those with contacts) push less, so dense
   bodies move mainly by contraction and grip. Acceptable, or should bonded contacts be excluded from the count
   (at the cost of large bodies again being able to compress crowds)?
2. Should a bond partner be able to rescue a combat victim drained to 0 at S12 (current design: yes), or should
   `E == 0` after combat be final?
