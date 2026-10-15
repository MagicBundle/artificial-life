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
| `tiny3d` (3D tests/CI, §27.8) | 1 024 | 32 | 256 | 128 | 21 (6.10) × 4 layers | 32 × 6 layers |
| `64k3d` (3D at 64k scale, §27.13, provisional) | 65 536 | 1 024 | 16 384 | 320 (height 80) | 53 (6.04) × 13 layers (6.15) | 80 × 20 layers |

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
| S8 | **costs and solvency**: `energy.pay_costs`, then `actions.mask(act, solvent)` (3D: `energy.pay_costs3`, §27.7) | `energy, act, n_bonds, brain_cost, alive` | `E`, `solvent`, `paid_basal`, `paid_action`, and `act` with all actions of insolvent agents cancelled | — |
| S9 | **bonds**: `bonds.break_bonds`, then `bonds.form_bonds` | `partner, rev, act.release, geom.length, act.bond_want, act.bond_target, alive & solvent` | `partner, rev`, broken and formed counts | `BOND_PRIO` |
| S10 | **harvest**: `energy.harvest` | `E, resource, fcell, act.harvest, solvent` | `E, resource, got` | — |
| S11 | **combat**: `energy.combat` | `E, resource, fcell, act`, post-S9 `partner` | `E, resource, lost, gain, spill, residue` (I13); attacks on a target that became a bond partner at S9 are dropped unless `attack_bonded` | — |
| S12 | **share**: `energy.share` | `E`, post-S9 `partner, rev`, start-of-step `s.partner`, `act.share`, `solvent` | `E, out, inflow, heat`; a slot sends only if its partner is unchanged since the start of the step | — |
| S13 | **cap**: `energy.cap_energy` | `E` | `E ≤ e_max`, `overflow` | — |
| S14 | **deaths**: `lifecycle.deaths` | `alive, solvent, E, lost, age, uid, generation, clade, partner, rev, resource, fcell`; director: `cat_mask` from `lifecycle.catastrophe_mask(pos, alive, env, ks[CATASTROPHE])` (start-of-step `pos`) | `survivor` mask (`alive` after deaths); corpses into `resource`; dead rows canonicalised; bonds of the dead dropped on both sides; death records | — |
| S15 | **physics**: `physics.forces`, `physics.integrate` (3D: `forces3`, `integrate3`, §27.3) | start-of-step `pos, heading`; `nbr`; post-S14 `partner`; `act.thrust, act.contract, act.grip, act.turn`; `survivor` | `pos, vel, heading` for survivors; `v_clips`. Right after S15, survivors' `h ← h_new`, `msg ← act.msg`, `last_loss/harvest/inflow` are written and everything else gets 0 (§16.3). | — |
| S16 | **reproduction and immigration**: `lifecycle.allocate_lanes`, `lifecycle.reproduce` | the in-flight state after S15, `act.reproduce`, `act.child_bond`, `solvent`; director: `env_imm_request` | newborn rows (small arrays), parents debited, child bonds, `pend_*` (child weights), birth records | `BIRTH_PRIO`, `PLACEMENT`, `MUT_W1`, `MUT_W2`, `MUT_GENES`, `IMMIGRANT` |
| S17 | **fields**: `fields.deposit_chem`, `fields.update_chem`, `fields.update_resource`, `fields.drift_patches` | `chem, act.emit, fcell, solvent, resource, patch_pos, patch_vel`; director: `env_regen_mult, env_season_*, env_patch_gain, env_drift_mult` | `chem, resource, patch_pos`, `regen_injected` | — |
| S18 | **bookkeeping**: `step.finish` (private) | everything, including `desc`, `x`, `act` and `as_prop` for the kin counters | `step + 1`; `key = key_next`; `total`; `ledger_cum`; `counters`; `diag` OR-ed with `invariants.device_check(...)`. Director: after S17 `step.consume_env` writes `env_imm_request ← max(request − n_imm, 0)` and `env_cat_kill_p ← 0` | — |

A 3D world runs the same sub-steps in the same order, with static 3D branches in S4, S5, S7, S8, S15 and S18
(§27.8).

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
    counters: mx.array            # (n_counters(nd),) int32: CounterIdx (§19.1), then Counter3D in 3D (§27.8)
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
Gradients use the four periodic neighbour cells (2D; 3D adds a mirrored z difference and the up components go to
the input tail, §27.6). Bond-slot blocks keep their slot index for the bond's whole lifetime, so a recurrent brain
can attach roles to particular slots.

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
- In 3D, `energy.pay_costs3` adds the pitch change to the turn term and a buoyancy due, paid last and booked in
  `HEAT_BUOYANCY` (§27.7).

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
  - `corpse = E + S` is scatter-added into `resource` at the dead agent's `fcell` → `FLOW_CORPSE` (2D; in 3D at
    the floor cell below it, §27.6). For an insolvent agent, `E` is whatever bond inflow it received at S12, so the
    ledger is unchanged.
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
| 7 | `FLOW_HARVEST` | internal | 15 | `RESERVED` (always 0 in 2D); in 3D `HEAT_BUOYANCY` (§27.7) | — (3D: sink) |

No director action adds a ledger term. A catastrophe death is a corpse (`FLOW_CORPSE`), an immigration request is
admitted through the ordinary immigrant lanes (`INJECT_IMMIGRATION`), and a regeneration multiplier or patch gain
only changes the integer regeneration amount (`INJECT_REGEN`).

**Per-step closure.** This is checked on device every step (`Diag.LEDGER`) and in tests on the host (2D; in 3D
`L[15]` is a sink too, §27.7):

```text
TOTAL(after) − state.total == L[0] + L[1] − (L[2] + L[3] + L[4] + L[5] + L[6])
L[FLOW_STOLEN] == L[FLOW_ATTACK_GAIN] + L[FLOW_SPILL]
```

**Global closure.** This is checked on the host at every metrics interval, and gives `ledger_error`, which must be 0
(2D; in 3D the sinks are `state.sinks(3)`, which adds slot 15, §27.7):

```text
TOTAL(now, recomputed in numpy int64) == total0 + cum[0] + cum[1] − Σ_{k=2..6} cum[k] == state.total
```

Any lost or created quantum anywhere in history is detected exactly. The per-step check also attributes it to a
step.

**How it is computed on device.**
- Per-agent flows are stacked into an `(N, F_A)` int32 matrix, and per-cell flows into `(FG², F_F)` (3D: `paid_buoy`
  is one more agent column and the cells are the `FZ·FG²` of the volume, §27.7).
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
| `proportional_take` | demand < 2¹⁶; contenders per pool < 2¹⁵ | validate `harvest_max, attack_max, share_rate < 2¹⁶`; `INT_HEADROOM` if max field-cell occupancy (3D: the x/y column occupancy, §27.7) ≥ `occ_limit` or max attackers per target ≥ 2¹⁵ |
| corpse and spill scatter into R | `R_HARD + occ·(2·e_max + S) < 2³¹` (per agent in the cell: corpse ≤ e_max + S, spill plus residue ≤ lost ≤ e_max; in 3D `occ` is per x/y column, because corpses sink to the floor cell, §27.7) | `Dims.occ_limit = min(2¹⁵ − 1, (2³¹ − 1 − R_HARD) // (2·e_max + S))` = 1 060 at defaults; same `INT_HEADROOM` bit |
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

In 3D, `c_ref` also holds a founder's buoyancy-cap upkeep and the regeneration is weighted by where the agents live:
each layer counts with the weight `min(1, FZ·p_k)`, `p_k` being its share of the alive agents, so a layer at or above its
even share `1/FZ` counts fully and a sparser one in proportion to its occupancy (§27.7, §27.10).

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
- **Per step (S17):** regenerate (§10.2), then diffuse with the exact integer flux stencil (`ops.int_diffuse`; 3D:
  `ops.int_diffuse_3d`, no flux through the floor and the surface, §27.6):
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
`scatter_add_i32` at flat index `c·FG² + fcell` (3D: `c·FZ·FG² + fcell`, §27.6), which is deterministic. Then
`chem += dep_q · chem_deposit_scale (1/256)`, so a full-strength emitter adds 1.0 per step.

**Update.** A 5-point explicit stencil via `mx.roll`, clamped to `[0, chem_max = 16]` (2D; 3D: a 7-point stencil
with mirrored z boundaries, §27.6):

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
change. The 3D golden (`state_digest` of `tiny3d_cfg`) is §27.8's.

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

| | 64k | 256k | 64k3d (estimate, §27.13) |
|---|---|---|---|
| state (weights + small arrays + fields) | 0.60 GB | 2.4 GB | 0.69 GB |
| pending buffer | 9 MB | 36 MB | ≈ 10 MB |
| transients (neighbours ≈ 0.25 GB / 1.0 GB / **2.0 to 2.7 GB**; mutation ≈ 60 MB / 0.24 GB / 60 MB) | ≈ 0.35 GB | ≈ 1.3 GB | ≈ 2.1 to 2.8 GB |
| **total** | **< 1.1 GB** | **≈ 3.8 GB** | **≈ 2.8 to 3.5 GB** |
| checkpoint size | 0.6 GB | 2.4 GB | 0.69 GB |

The 64k3d column is arithmetic from the shapes, not a measurement: its neighbour search holds `Q = 27·C = 1 296`
candidates per agent at the provisional `C` = 48, 9× the 2D 144, so one `(N, Q)` array is 340 MB (§27.13). At `C` = 24 the
figures were 648 candidates, 170 MB and a total of 1.8 to 2.2 GB; the cloud CPU measured a peak of 1.99 GB for `64k3d` at
`C` = 24 (cloud-7), which scales to about 3.2 GB at 48.

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
    presets/{tiny,dev,64k,256k,tiny3d,64k3d,fast}.toml   package data
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
    stream.py                    `alife stream`: the binary frame protocol, control lines, --follow (§27.11)
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
DIMS_SUPPORTED: tuple[int, ...] = (2, 3)                                  # world.dims values validate accepts (§27.1)
def is_3d(cfg: Config) -> bool                                            # world.dims == 3 (static)
def require_2d(cfg: Config, what: str) -> None                            # ConfigError for a 3D config in a 2D-only tool
PRESETS: tuple[str, ...] = ("tiny", "dev", "64k", "256k", "tiny3d", "64k3d")   # plus overlay "fast"; 3D: §27.8, §27.13
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
class Counter3D(IntEnum): Z_FLOOR_CLAMPS = 27; Z_SURFACE_CLAMPS = 28; GROUNDED = 29    # 3D only (§27.8)
def n_counters(nd: int) -> int                                             # len(StepOut.counters): 27 | 30 (3D)
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
def int_diffuse_3d(r: mx.array, d_q8: int) -> mx.array                     # (FZ,FG,FG): no flux in z, d_q8 ≤ 42 (§27.6)
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
                            # chem_grad (N,NCH,2) = IN_CHEM_GRAD as (fwd, lat) per channel;
                            # 3D: res_grad (N,3), chem_grad (N,NCH,3) as (fwd, lat, up) (§27.6)
def init_patches(key_pos: mx.array, key_vel: mx.array, cfg: Config) -> tuple[mx.array, mx.array]
def season_weights(step: mx.array, cfg: Config, env: Env | None = None) -> mx.array   # (PP,) f32 (§13.1)
def fertility(patch_pos: mx.array, cfg: Config, step: mx.array | None = None,
              env: Env | None = None) -> mx.array                         # (FG,FG) f32 in [0,1]; 3D (FZ,FG,FG) (§27.5)
def vertical_profile(cfg: Config) -> mx.array                              # 3D: (FZ,1,1) f32, V[0] = 1 (§27.5)
def regen_rate(patch_pos: mx.array, cfg: Config, step: mx.array | None = None,
               env: Env | None = None) -> mx.array                        # (FG,FG) int32; 3D (FZ,FG,FG)
def cell_index(pos: mx.array, cfg: Config) -> mx.array                    # (N,) int32; 3D z-major (§27.5)
def sample_fields(resource: mx.array, chem: mx.array, fcell: mx.array, heading: mx.array, cfg: Config,
                  *, pitch: mx.array | None = None) -> FieldObs              # pitch required in 3D (§27.6)
def deposit_chem(emit: mx.array, fcell: mx.array, active: mx.array, cfg: Config) -> mx.array   # (NCH*n_cells,) int32
def update_chem(chem: mx.array, dep_q: mx.array, cfg: Config) -> mx.array     # 3D: 7-point, mirrored z (§27.6)
def update_resource(resource: mx.array, patch_pos: mx.array, cfg: Config, step: mx.array | None = None,
                    env: Env | None = None) -> tuple[mx.array, mx.array]
                    # (resource', inc (FG,FG) int32) — regen then int diffusion; 3D (FZ,FG,FG), int_diffuse_3d
def drift_patches(patch_pos: mx.array, patch_vel: mx.array, cfg: Config, env: Env | None = None) -> mx.array

# sensing.py
def neighbor_descriptors(s: State, nbr: Neighbors, cfg: Config) -> mx.array          # (N,K,NF) f32; 3D NF = 8 (§27.2)
def build_inputs(s: State, age: mx.array, nbr: Neighbors, desc: mx.array, geom: BondGeom,
                 fobs: FieldObs, key: mx.array, cfg: Config,
                 *, grounded: mx.array | None = None) -> mx.array                   # (N,D_IN) f32; grounded: 3D (§27.6)

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
class Costs3(NamedTuple): energy; solvent; paid_basal; paid_action; paid_buoy      # (N,) each; 3D (§27.7)
def pay_costs3(energy, act: Actions, mo: Motion3, n_bonds, brain_cost, buoy_cap, alive, cfg: Config) -> Costs3
                    # 3D: the pitch in the turn term, the buoyancy due paid last (HEAT_BUOYANCY, §27.7)
def harvest(energy, resource, fcell, act: Actions, solvent, cfg: Config
            ) -> tuple[mx.array, mx.array, mx.array]                               # (E, resource, got)
class Combat(NamedTuple): energy; resource; lost; gain; spill; residue; attack; hit; max_attackers
                    # (N,) int32 ×5 except resource (FG,FG) (3D: (FZ,FG,FG)); gain, spill per attacker; lost, residue per victim
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
    # (N,) except resource, r_overflow (FG,FG) (3D: (FZ,FG,FG), corpses in the floor cell, §27.6);
    # corpse = E + S on dead rows else 0; cause = DeathCause on dead
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
def step(s: State | State3, cfg: Config) -> tuple[State | State3, StepOut]   # State3 in 3D (§27.8)
def consume_env(s: State, env: Env, lanes: Lanes) -> State                 # env_imm_request carry-over, env_cat_kill_p reset (§12.7, §12.9)
def make_step(cfg: Config, *, compile: bool | None = None) -> Callable[[State], tuple[State, StepOut]]
    # compile=None → cfg.run.compile; in 3D the callable takes and returns State3 (§27.8)

# invariants.py
def device_check(s: State, ledger: mx.array, prev_total: mx.array, max_cell_occ: mx.array,
                 max_attackers: mx.array, overflow_cells: mx.array, v_clips: mx.array, cfg: Config) -> mx.array
                    # int32 Diag bits for this step (ORed into s.diag by step); 3D terms in §27.8
def env_out_of_range(s: State, cfg: Config) -> mx.array                   # device bool: any env_* value non-finite / outside §2.2 ranges
def full_check(s: State, cfg: Config, prev: State | None = None, out: StepOut | None = None) -> list[str]
                    # host numpy; [] == OK. Per-step ledger = s.ledger_cum − prev.ledger_cum; births and immigrants
                    # from out.births.valid. Stateless: a cumulative dead-uid set, if wanted, lives in the test harness.
                    # 3D: the closures over state.sinks(3), no RESERVED rule, and _check_rows_3d (§27.8)

# metrics.py
def components(partner: mx.array, alive: mx.array, iters: int) -> tuple[mx.array, mx.array]   # labels, converged
def snapshot_metrics(s: State, cfg: Config, *, heavy: bool) -> dict[str, Any]     # host floats/ints/lists
def interval_row(step: int, counters: np.ndarray, ledgers: np.ndarray, snap: Mapping[str, Any],
                 perf: Mapping[str, float], *, nd: int = 2) -> dict[str, Any]
                    # counters (T, n_counters(nd)), ledgers (T,16); the ledger keyed by ledger_names(nd) (§27.8)
def carrying_capacity(s: State, cfg: Config, step: mx.array | None = None, env: Env | None = None) -> float
                    # step + env: the live regeneration (the director's K_est, §26.5)
                    # 3D: each layer's regeneration weighted by the share of the alive agents in it (§27.10)

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
COLOR_MODES: tuple[str, ...] = ("tag", "energy", "clade")
def agent_colors(s: State | State3, cfg: Config, color: str) -> mx.array   # (N,) uint32 0x00RRGGBB, 2D and 3D (§20.1, §27.11)
def field_colors(s: State | State3, cfg: Config, layer: int = 0) -> mx.array   # (FG²,) uint32 of one lattice layer

# viewer.py
def run_viewer(cfg: Config, state: State, *, rundir: RunDir | None = None, size: tuple[int, int] = (900, 900),
               steps_per_frame: int = 1) -> None        # ImportError with an install hint without pygame-ce
class ViewerSession: ...           # the viewer minus the window (state, viewport, keys, selection, HUD, snapshots)

# stream.py  (§27.11)
MAGIC: Final = b"ALSTRM\x00\x01"; PROTOCOL: Final = 1
@dataclass class View: color: str = "tag"; fields: bool = False; field_z: int = 0
class Streamer:                                   # header + frames of one stream to a binary file object
    def __init__(self, cfg: Config, out: BinaryIO, *, mode: str = "simulate", source: str = "preset",
                 run: str | None = None, view: View | None = None, every: int = 1) -> None
    def begin(self, state: State | State3) -> None;  def send(self, state: State | State3) -> None
def serve(cfg: Config, *, mode: str, source: str, run: str | None, state: list[State | State3] | None = None,
          rd: RunDir | None = None, view: View | None = None, every: int = 1, max_steps: int | None = None,
          poll_s: float = 1.0, control: Control | None = None, out: BinaryIO | None = None,
          step_fn: StepFn | None = None) -> StreamStats     # a closed reader ends it with ended == "pipe", no error
def protocol_output() -> ContextManager[BinaryIO]  # private dup of fd 1; sys.stdout and fd 1 -> stderr meanwhile
class FdControl: ...                              # stdin control lines on a daemon thread (os.read); EOF = "quit"

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

**In 3D** (static `d3`, the table of §27.8) S4 also computes the grounding and passes the pitch, S7 adds
`decode_motion3`, S8 calls `pay_costs3`, S15 calls `forces3` and `integrate3` and writes the pitch back, and
`finish` adds the `HEAT_BUOYANCY` column, `Counter3D` and the column occupancy; the 2D lines above are unchanged.

`finish` is a private helper in `step.py`. It also computes
`max_cell_occ = ops.scatter_add_i32(FG², fcell, 1, s.alive).max()`, the start-of-step field-cell occupancy that
`device_check` needs for the headroom bit (in 3D the x/y column occupancy, `fcell mod FG²`, §27.7). It:
1. builds the per-agent flow matrix `[paid_basal, paid_action, got, cb.lost, cb.gain, cb.spill + cb.residue, sh.out,
   sh.heat, e_over, dt.corpse]` (all (N,) int32; `sh.out` and `sh.heat` are already summed over slots; the spill
   column is `FLOW_SPILL`), the per-cell matrix
   `[regen_inc, dt.r_overflow]` and the lane `endow` vector;
2. reduces them with `ops.exact_sum`;
3. forms the count × constant terms in int64 (§10.8): `HEAT_BIRTH = int64(n_birth)·birth_cost`,
   `FLOW_STRUCTURE = int64(n_birth)·S`, `INJECT_IMMIGRATION = int64(n_imm)·(e_init + S)`;
4. assembles `ledger (16,)`, adds it to `ledger_cum`, and computes the new `total` (via `state.total_energy`, whose
   `S·n_alive` term is int64);
5. assembles `counters (27,)` (3D: (30,), `Counter3D` appended, §27.8). The kin counters are computed here and
   nowhere else:
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
    buoyancy_cost: int = 0;  buoyancy_cap_cost: int = 0                     # 3D only (§27.7): buoyancy use, cap upkeep

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
| `tiny3d` | `tiny`'s, plus `world: dims 3, height 24`; `physics: gravity 0.03, floor_eps 0.1, buoyancy_max 0.04`; `energy: buoyancy_cost 16, buoyancy_cap_cost 8`; `mutation: buoy_init 0.5, buoy_sigma 0.02`; `fields: chem_diffusion [0.16, 0.10, 0.16], benthic_lambda 8, photic_amp 0.5, photic_lambda 8` (provisional, §27.8) |
| `64k3d` | `64k`'s, plus `world: size 320, dims 3, height 80`; `physics: gravity 0.015, floor_eps 0.1, buoyancy_max 0.04`; `sensing.cell_capacity 48` (provisional, was 24 until cloud-7); `energy: brain_cost_per_l1 0.0905, buoyancy_cost 16, buoyancy_cap_cost 8`; `mutation: buoy_init 0.5, buoy_sigma 0.02`; `fields: n_patches 8, patch_sigma 24, regen_max 1700, chem_diffusion [0.16, 0.10, 0.16], benthic_lambda 8, photic_amp 0.5, photic_lambda 8` (provisional, §27.13) |
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
`fields.benthic_lambda`, `fields.photic_amp`, `fields.photic_lambda`, `energy.buoyancy_cost`,
`energy.buoyancy_cap_cost` (§27.1, §27.3–§27.5, §27.7); the
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
- **World dimensions** (§27.1): `world.dims ∈ {2, 3}` and in `config.DIMS_SUPPORTED` (`(2, 3)` since PR 3,
  §27.8); in 2D every 3D key (`config.THREE_D_KEYS`) keeps its default (`height` and the gravity, buoyancy and
  vertical-profile keys 0); the 3D-only rules are listed in §27.1 (the buoyancy costs' in §27.7).
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

### 19.1 Per-step counters (`CounterIdx`, `StepOut.counters`, int32 (27,); 2D, 3D (30,): §27.8)

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
`(T,27)` and ledgers `(T,16)` as int64 sums, so no step's events are lost. In 3D the counters are `(T,30)`: the 27
above, then `state.Counter3D` (§27.8).

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
  - `ledger` (interval sums by `LedgerIdx` name; `state.ledger_names(nd)`, so slot 15 is `HEAT_BUOYANCY` in 3D,
    §27.8);
  - **`ledger_error`** (must be 0, §10.8);
  - `energy_mean`, `energy_median`.
- **Life history:** `age_mean`, `generation_max`, `generation_mean`.
- **Lineage** (int scatter of alive into `clade` bins, size `n_init + 1`): `clades_alive`, `clade_entropy`,
  `clade_simpson` (= 1 − Σp²), `top_clade_share`.
- **Genome:**
  - `tag_var` (trace of the tag covariance over alive agents);
  - `sigma_q10/q50/q90`, `rate_q10/q50/q90`, `frac_q10/q50/q90`;
  - `brain_cost_mean`.
- **Fields:** `resource_mean`, `chem_mean` (list of 3). In 3D both are over every cell of the volume (§27.8).
- **3D** (only when `world.dims = 3`, §27.9): the interval keys `z_floor_clamps`, `z_surface_clamps`, `grounded_frac`
  and `buoy_heat` after `v_clips`, and the snapshot keys `height_*`, `surface_frac`, `speed_mean`, `pitch_abs_mean`,
  `buoy_cap_q*`, `band_frac` and `clade_height_mi` after `diag` (heavy: `body_z_extent_mean` and `_max`). A 2D row
  has none of them.
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
  with every threshold swept. Plots need the optional extra `analysis` (matplotlib).
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
- **Colour helpers (PR 4).** The per-agent and per-cell colour code of `render_frame` is split out, with the 2D path
  and its palette unchanged (the frames of every colour mode and layer set are bit for bit the same as before the
  split): `_agent_rgb(s, cfg, color)` is the (N,3) float colour of a state (2D or 3D) in a mode of `COLOR_MODES`,
  and `_cell_rgb(resource, chem, cfg, layers)` the `0x00RRGGBB` colour of `n` field cells from flat `(n,)` resource and
  `(NCH, n)` chemical arrays; `_background` calls it on the whole 2D lattice. Two public wrappers serve the 3D
  stream (§27.11) and need no 2D world: `agent_colors(s, cfg, color)` is `(N,)` uint32 `0x00RRGGBB`, and
  `field_colors(s, cfg, layer)` the `(FG²,)` colours of one lattice layer (the 2D lattice, or z layer `layer` of a
  3D one). `render_frame` itself still refuses a 3D config.
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
- It draws a 2D plane: `run_viewer` and `ViewerSession` refuse a 3D config (`require_2d`, exit 2). A 3D run is viewed
  through `alife stream` and the AlifeControl 3D viewer (§27.11, §27.12).

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
alife run      [--preset tiny|dev|64k|256k|tiny3d|64k3d] [--overlay fast ...] [--config PATH] [--set section.key=value ...]
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
alife stream   [RUN|FILE | --preset P [--set ...]] [--checkpoint STEP|latest] [--every K] [--steps N]
               [--follow [--poll SECONDS]] [--color tag|energy|clade] [--fields] [--field-z K] [--no-control]
alife bench    [--preset P] [--overlay fast] [--set ...] [--steps 200] [--warmup 20] [--stages]
               [--determinism] [--brain-variants] [--json PATH]
alife llm-check [--llm-url URL] [--llm-model NAME] [--timeout SECONDS] [--preset P]
alife activity RUN [--out DIR] [--grid N] [--marker-rates R,...] [--horizons H,...] [--shadows N] [--seed S]
               [--windows A:B,...] [--reuse] [--no-plots] [--quiet]
```

**`activity`** never modifies the run's files: it writes `activity.npz`, `summary.json`, `classification.txt` and the
figures to `--out` (default `RUN/activity`), §19.3 and `docs/ACTIVITY.md`.

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

**`stream`** (3D world PR 4, §27.11) writes the world as a binary frame stream on stdout, for the AlifeControl 3D
viewer (2D worlds too, with z = 0). It re-simulates like `view`, without the director, or with `--follow` streams the
run's newest checkpoints (the faithful view of a director run); it reads control lines (`pause`, `run`, `step [N]`,
`every K`, `color MODE`, `fields on|off|z K`, `quit`) from stdin, whose EOF ends it, and never writes into the run
directory. `--field-z K` implies `--fields`, and `--preset` has no default here. The protocol, the control grammar and
the exit codes (a closed reader is exit 0) are normative in §27.11; the viewer that reads it is §27.12.

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
| 0 | OK, including a clean extinction with `status = extinct`; for `stream` also a closed reader, `quit` and stdin EOF |
| 1 | unexpected error; also `llm-check` when the check fails (`FAIL ...`) |
| 2 | config error |
| 3 | invariant violation: diag bits are printed by name and the state is dumped to `diag/` |
| 4 | checkpoint incompatible |
| 130 | interrupted by SIGINT after a clean flush and checkpoint |

**Director flags** (§26). `--director live|replay` is a shortcut for `--set run.director=MODE` and also sets
`director.enabled=true` on `run`; `--director off` sets only the mode. `--replay-log` goes with `--director replay`
and `--init-from` (a replay needs a state to continue); other combinations, an unreadable or invalid log, and
`--director live|replay` on `resume` of a run without `director.enabled`, are config errors (exit 2) reported before
anything is written. `view`, `render` and `stream` re-simulate without the director (§26.8). The persona and goal are set with
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
  to a founder or immigrant (`ancestors_complete`). The 3D additions are in §27.11.
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
  - 3D (§27.8): `tiny3d_cfg` is the `tiny3d` preset with f32 weights, and `micro3d_cfg` the `micro_cfg` sizes on
    `tiny3d` (`tests/helpers.MICRO3D`: height 24, `FZ = 6`, `G_z = 4`).
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

In 3D the closures include `HEAT_BUOYANCY`, `RESERVED == 0` is not checked, and `_check_rows_3d` adds the z, pitch,
`buoy_cap` and boundary-velocity rules (§27.8); `tests/test_step_3d.py` runs `micro3d_cfg` through the same
checks after every step.

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
| 64 | `INT_HEADROOM` | `max_cell_occ ≥ Dims.occ_limit` (1 060 at defaults, §10.9; in 3D per x/y column, §27.7) or `max_attackers ≥ 2¹⁵`: a real int32 precondition, unreachable with contact-attenuated thrust (F18) |
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
is checked only by the host `full_check`. The 3D terms of both checks (no new bit) are in §27.8.

### 22.4 Determinism tests

**`test_determinism.py`** (WP4):
- The same seed run twice (50 compiled steps each) gives identical `state_digest`. Different seeds give different
  digests.
- **`legacy_state_digest`** (§14.2) is the golden's digest. Director-enabled-inert vs director-off bit-identity is in
  `test_director_env.py`.
- **Compiled vs eager** (marker `slow`). Population and ledger trajectories match for 30 steps. Bitwise equality is
  not required, but it was observed on CPU.
- **`test_golden`** (marker `golden`): the `tiny_cfg` digest after 50 steps equals the pinned value.

**`test_determinism_3d.py`**: the same checks on `tiny3d_cfg` (eager against compiled in the fast suite), and
`test_golden_3d` (§27.8).

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

**`test_cli_smoke_3d.py`** does the same for a 100-step `tiny3d` run (§27.8): clean rows, `inspect --check`,
bitwise resume, the 2D-only tools refusing it, and a fork that may not change `world.dims` or `world.height`.

**`test_affordances.py`** runs `tiny` for 400 steps with the pinned seed. Every affordance must be exercised: the
interval totals of births, `DEATHS_STARVED + DEATHS_KILLED`, attacks with hits, bonds formed, bonds broken, child
bonds, sharers, emitters and harvest are all > 0, and the population is > 0 at the end.
- If a parameter change makes an affordance unreachable, this test fails.
- The test builds founders with a fixed bias row from a test-only helper, so every action fires.
- This affects the test only, not the engine defaults (founders are fully random).

**`test_affordances_3d.py`** exercises the 3D affordances in one-step `micro3d_cfg` scenes (§27.8), and runs `tiny3d`
for 400 compiled steps with the pinned seed and the 2D bias row (§27.13): every affordance above, plus floor and
surface clamps, `HEAT_BUOYANCY`, heights and pitch in range, a mutating buoyancy cap, deposits in the volume, and
movement in x and y apart from sinking (a passive agent loses `gravity/drag` of height per step and does not move
sideways).

**`test_ecology.py`** (marker `slow`) runs the `dev` preset (float32 weights, as `tiny_cfg`, F19) with default random
founders for 3 000 steps. The
population must be neither extinct nor pinned at `N_CAP` from step 0. Births > 0, bonds > 0, and at least one
attack lands. This test is the tuning harness for the defaults in §17.

**`test_ecology_3d.py`** (marker `slow`) is its 3D counterpart on `DEV3D`, the `64k3d` preset shrunk (§27.13): 2 500
steps with float32 weights, the 2D assertions that carry over, floor contacts, `HEAT_BUOYANCY`, some agents resting and
some swimming in the final state and hash-cell overflow in at most 1% of the steps (at the provisional `C` = 48). It is
collected only under `-m slow` (the nightly job and the cloud worker).

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
step the director started at; `budget_per_100k` is the effective budget; a 3D run adds `"dims": 3` after `seed`, and a
2D header never has the key, §27.10). A new header is appended whenever a live
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
| `vitals_step`, `vitals` | the metrics row step the decision used (the boundary) and a compact vitals dict (rounded to 4 places): `n_alive`, `k_est`, `n_over_k`, `fall`, `extinction_risk`, `top_clade_share`, `novelty`, `coverage`, `new_cells_200k`, `b_shift`; a 3D run adds `k_floor` and `n_over_k_floor` after `n_over_k` (§27.10), a 2D entry never has them |

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
| `catastrophe` | `center` [x, y], `radius`, `kill_p` | radius in [20, min(200, L/2)] (3D: both ends times L/1024, §27.10), kill_p 0.2–0.9 | 3 | 100k |
| `drift_change` | `mult` | 0.5–10, permanent until changed | 1 | 50k |

- **Seasons** change `env_season_period` / `env_season_amp` only; the mode stays `fields.season_mode` and is not a
  runtime lever. The mutation-floor nudge is not implemented.
- **Patch moves** are host edits of `patch_pos`, wrapped with `ops.wrap` (so the f32 result is strictly below L); the
  log holds the absolute new coordinates and the old ones in `prev`. The drift continues from the new position. In 3D
  `patch_pos` is (PP, 2) as well and a move is x and y only (§27.10).
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
  cadence, a revert of an action without a timed effect. `log.validate_log` adds: records in non-decreasing step, every
  revert has an active effect on its lever, and every header's `dims` (absent = 2) equals `world.dims` (§27.10). A bad log is a config error (exit 2) before the first step.
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
  the regeneration of the *current* environment (seasonal at the step, scaled by the levers; in 3D weighted by the
  layers the agents live in, §27.10). It is used only at the boundary, never stored. In 3D the boundary also computes
  `metrics.floor_carrying_capacity(state, cfg, state.step, state.env)`, the floor layer's regeneration alone over
  `c_ref` (`k_floor`; §27.10), which the extinction-risk flag reads; 2D has one lattice and one `K_est`, and its vitals
  carry no `k_floor`.
- **Viability:** `n_alive / K_est`; `fall = 1 − (window minimum / highest window mean over the last risk_fall_window)`;
  the **extinction-risk flag** is `n_alive/K_est < risk_k_frac` or `fall > risk_fall` (3D: `n_alive/K_floor`, with
  `K_floor = k_floor`, in place of `n_alive/K_est`, §27.10).
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
- **The viewer, `render` and `stream` re-simulate without the director.** They keep the checkpoint's environment
  levers but apply no log entry, so after a director action they diverge from the run they were opened on. Replay in
  the viewer is not implemented; `alife stream --follow` shows a director run faithfully, because it streams the
  checkpoints the runner wrote with the applied environment (§27.11).
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
| `test_director_3d.py` | the director in 3D (§27.10): validation accepts it; the column catastrophe through the director path; patch moves keep `patch_pos` (PP, 2); the scaled radius bounds; the header's `dims` and the refusal of a log of the other dimensions; the 3D prompt and the byte-identical 2D one; a scripted `tiny3d` run, its replay (fresh, in place, live resume) and the `random` persona on it; a real-rules Gardener run with the occupancy-weighted `K_est` (PR 6: the weights, the prompt qualifier, `Director._vitals`) |
| edits | `test_director_actions.py` and `test_director_log.py` (the "unknown source" example is no longer `llm`), `test_director_config.py` (the LLM keys, the flags; `PERSONAS` gains `random`), `test_runner.py` (hook order; no action on a violated or extinct state; the checkpoint holds the applied env), `test_metrics.py`, `test_cli_smoke.py` (`--director` flows, fork reset), `test_lint.py` (the director package is scanned); PR 4 of the 3D world: `test_dims_config_pr2.py` (the 3D case of the director rejection is gone, §27.10) |

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
steps later, or to now with "(so far)"). About 2.4k tokens for the `tiny` preset. In a 3D world the world sentence, the
catastrophe's effect text, the world size line and the capacity estimate's qualifier differ (§27.10); the 2D messages
are unchanged byte for byte.

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
gain the PR 2 keys at their defaults). PR 3 (§27.6–§27.8) adds the field dynamics, the buoyancy costs and their
ledger term, and the step wiring, and enables `world.dims = 3` with the `tiny3d` preset; 2D is again unchanged
(config files gain the two cost keys at their defaults). PR 4 (§27.9–§27.11) adds the 3D metrics keys and the
floor-layer carrying capacity, lifts the director's 3D refusal, extends the CLI reporting and adds `alife stream`,
the frame protocol of the 3D viewer; the compiled step, the state and every 2D output are unchanged (a new module,
`stream.py`, and a split of the renderer's colour code, whose 2D frames are bit for bit the same). PR 5 (§27.12) is the
Swift side, in `gui/`, the docs and one test of the engine's suite (`tests/test_gui_sync.py`, with the config files it
checks): the 3D viewer window of AlifeControl, the decoder of §27.11, the 3D levers; the engine is untouched. PR 6
(§27.13) is the ecology: the carrying capacity of §27.10 becomes occupancy-weighted (a host-side change of one metric;
the compiled step, the state and every 2D output are unchanged) while the director's extinction-risk test reads the floor
layer's capacity alone (§27.10), the provisional `64k3d` preset (hash capacity `C` = 48), and the 3D
affordance and ecology tests; all of that needs no GPU and is built, and the measurements that do are listed in §27.13.

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

**Gate and validation.** `world.dims` must be 2 or 3 and in `config.DIMS_SUPPORTED`, which is `(2, 3)` since PR 3
enabled 3D (`(2,)` before; read at call time, so a test can narrow it); a dims outside it reports "world.dims = 3
is not enabled yet". In a 2D world every key of `THREE_D_KEYS` except `world.dims` must keep its default. The 3D
rules apply whenever dims = 3 (also under a narrowed gate), each with its own message, and no 2D message changed.
They round only finite ratios, so a non-finite `size` or `height` is a listed problem, never an exception:
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
- `benthic_lambda > 0`; `photic_amp ∈ [0, 1]`; `photic_lambda > 0` when `photic_amp > 0`.

(PR 2 and PR 3 also rejected `director.enabled` in 3D, until the director's levers, vital signs and log had a 3D
form; PR 4 lifted the rule, §27.10.)

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
  refuses both as dynamics keys (exit 2). (Before PR 3 enabled 3D, a fork to dims = 3 failed validation first,
  exit 2.) A 3D fork also refuses a changed `physics.radius` (exit 4, nothing written): the radius sets `z_lo` and
  `z_hi` (§27.2), and the source's rows, dead ones included, sit within its own bounds.
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
tail block `IN_NBR_DZ` (§27.6).

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
  their `HEAT_BUOYANCY` term are §27.7's.
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
  read only by sensing (`IN_VEL`, `IN_VEL_UP`, §27.6), so an agent resting on the floor senses no vertical motion.
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
keep a 3D crowd within C = 16. C and K for `64k3d` are sized in PR 6 (docs/3D.md §12 Q15; C = 48, provisional, §27.13).

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
`energy.buoyancy_cap_cost`, booked in `HEAT_BUOYANCY`) are §27.7's.

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
- `lifecycle.deaths` sinks corpses to the floor cell (§27.6).

**Catastrophe.** In 3D the catastrophe is a vertical column: `catastrophe_mask` measures the x/y minimum-image
distance from the (2,) `env_cat_center`, so every agent above the disc is in it, at any height. The director's menu
scales the radius bounds in 3D (docs/3D.md §12 Q10, §27.10).

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
`validate`, so a unit test can switch single 3D mechanisms on or off.

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
deposits, gradients, sinking corpses) are §27.6.

**2D-body guards.** The 2D cell index asserts (N, 2) positions and 2D `sample_fields` a 2D lattice. The asserts are
static (free under `mx.compile`, the 2D trace unchanged), so a missed 3D branch fails loudly instead of dropping z
(the 2D index) or folding the layer into y (`_nbr_cells`).

**Tests.** `tests/test_fields_3d.py`: the z-major index against a reference, layer 0 against the 2D index, the z
clamp, the highest grounded z in layer 0, the guards, both profiles, the floor layer bit for bit (plain, seasonal,
director env) and the 3D regeneration.

### 27.6 Field dynamics, sinking corpses and the 3D inputs (PR 3)

**Resource diffusion** (`ops.int_diffuse_3d`, called by `update_resource` in 3D after the regeneration of §27.5,
whose `inc` has the lattice's (FZ, FG, FG) shape; `INJECT_REGEN = Σ inc` as in 2D):

```text
q = (R · resource_diffusion_q8) >> 8
R ← R − (4 + n_z(k))·q + roll(q, ±1, x) + roll(q, ±1, y) + q[k − 1] + q[k + 1]      n_z(k) = [k > 0] + [k < FZ − 1]
```

The layers below the floor and above the surface contribute 0 and z is never rolled, so nothing crosses either
boundary and the total is conserved exactly (the stencil is a sum of pairwise fluxes). A cell sends at most `6q`:
`resource_diffusion_q8 ≤ 42` (§27.1) keeps `6q ≤ R`, hence `R ≥ 0`. The §13.2 cap argument carries over with six
neighbours while `6·d_q8 < 256`: an output cell is at most `R_HARD + 6·(d_q8 − 1)/256 < R_HARD + 1`. Both bounds
fail at 43: an interior cell of 256 with empty neighbours ends at −2, and an empty interior cell with six
saturated neighbours receives `6·q(R_HARD) > R_HARD`. With `FZ = 1` the stencil is `int_diffuse`. The resource has no decay, so
an ungrazed column fills towards `resource_cap` in every layer: the vertical profile shapes the regeneration rate
and the initial stock, and a standing vertical contrast exists only under grazing (a tuning question for `64k3d`).

**Chemicals.** `deposit_chem` accumulates at `c·n_cells + fcell` (`n_cells = FZ·FG²`; in 2D the same `c·FG² + fcell`),
so a deposit in the top layers stays in its own channel. `update_chem` in 3D (`_update_chem_3d`) is §13.3's update
with a 7-point stencil and mirrored ghost layers:

```text
lap  = roll(c, ±1, x) + roll(c, ±1, y) + c[k + 1] + c[k − 1] − 6·c          c[−1] = c[0],  c[FZ] = c[FZ − 1]
chem ← clip((1 − λ_c)·(c + D_c·lap), 0, chem_max)                         after c = chem + dep_q·chem_deposit_scale
```

A boundary layer therefore has the 5-point stencil of its four x/y neighbours and its one z neighbour: no flux
through the floor or the surface, mass conserved (to float rounding) at `λ = 0`, and stable for `D_c ≤ 1/6` (§27.1).
The 2D expression and its float addition order are unchanged.

**Sinking corpses** (S14, `lifecycle.deaths`). A corpse `E + S` is scatter-added into the floor cell below the dead
agent, `fcell mod FG²` = `(0, cy, cx)`, instead of its own cell; the term is still `FLOW_CORPSE`. The `R_HARD` cap
then applies over the whole volume (`r_overflow` (FZ, FG, FG), `HEAT_OVERFLOW`). A floor cell collects the corpses of
its whole column, so §10.9's occupancy bound applies per column (§27.7). Only corpses sink (docs/3D.md §12 Q19):
harvest takes from the agent's own 3D cell, and combat spill and residue go to the victim's own 3D cell
(`energy.harvest` and `energy.combat` are shape-agnostic).

**Field samples** (`sample_fields(…, pitch=)`, `_sample_fields_3d`; `pitch` is required in 3D). `res` and `chem` are
§5's at the agent's own 3D cell. The gradients are central differences over the six neighbour cells
(`_nbr_cells_3d`: x and y periodic within the layer, z clamped to `[0, FZ − 1]`), with the divisor kept at 2:

```text
g = (R[x+1] − R[x−1], R[y+1] − R[y−1], R[z+1] − R[z−1]) / (2·resource_cap)       z ± 1 clamped (mirror)
res_grad  = clip(RES_GRAD_GAIN · to_body3(g), −1, 1)                             (N,3) as (fwd, lat, up)
chem_grad = tanh(to_body3(central difference / 2) / chem_half_sat)               (N,NCH,3)
```

In the floor and surface layers the clamped index mirrors the cell itself, so the z component is half the one-sided
slope, consistent with the no-flux boundary (docs/3D.md §12 Q8). `to_body3` is the frame of §27.2, so the sensed
gradient and the thrust direction agree.

**Inputs** (`build_inputs(…, grounded=)`, `D_IN = 84`). Every vector is taken in the yaw-and-pitch frame
(`ops.to_body3`): its (fwd, lat) components fill the §5 blocks at their 2D indices, and its up component goes to the
tail. `grounded` is `physics.grounded(s.pos, cfg)` (§27.3), passed in by the step; it is required in 3D.

| idx | name | definition |
|---|---|---|
| 3, 4 | `IN_VEL` | (fwd, lat) of `clip(to_body3(vel) / v_max, −1, 1)` |
| 13, 14 | `IN_RES_GRAD` | (fwd, lat) of `res_grad` |
| 18–23 | `IN_CHEM_GRAD` | (fwd, lat) of `chem_grad` per channel |
| 28 + 7n + f | `IN_NBR` | `desc[:, n, :7]`: `NBR_DX`, `NBR_DY` are (fwd, lat) of `off / R` (§27.2) |
| 49 + 5b + f | `IN_BOND` | f = 1, 2: (fwd, lat) of `clip(to_body3(geom.off) / break_length, −1, 1)` |
| 69 | `IN_VEL_UP` | up of the velocity above |
| 70 | `IN_RES_GRAD_UP` | up of `res_grad` |
| 71–73 | `IN_CHEM_GRAD_UP` | up of `chem_grad`, per channel |
| 74–76 | `IN_NBR_DZ` | `desc[:, n, NBR_DZ]` of the NS nearest (0 for an absent neighbour) |
| 77–80 | `IN_BOND_DZ` | up of the clipped bond offset per slot (0 for an empty slot) |
| 81 | `IN_HEIGHT` | `z / height` |
| 82 | `IN_PITCH` | `sin φ` |
| 83 | `IN_GROUNDED` | 1 if `grounded`, else 0 |

Every other input is §5's. The contract `|x| ≤ 1` holds: `z / height ∈ [z_lo, z_hi] / height ⊂ (0, 1)`,
`|sin φ| ≤ sin 80° ≈ 0.985`, and every offset, velocity and gradient component is clipped or a `tanh`. Dead rows get
stale values, as in 2D (`IN_GROUNDED` is 1 on a dead row resting at `z_lo`); consumers mask by `alive`. A level scene
(one height, pitch 0, no vertical velocity) gives the 2D inputs bit for bit on CPU, because `to_body3` reduces to
`to_body`. The 2D input expressions are unchanged; the 3D branches are static. The step calls these in 3D from §27.8.

**Tests.** `tests/test_ops_3d.py` (`int_diffuse_3d`: exact conservation at FZ 2, 3 and 5 on random, saturated and
near-`R_HARD` fields; a numpy mirror reference; no z wrap; the range for `d_q8` 1, 4, 17 and 42; the spike values
`256 − 24` with 4 to each of six neighbours in the interior and `256 − 20` with 4 to each of five on the floor; the
`d_q8 = 43` counterexample; FZ = 1 equals `int_diffuse`; compile). `tests/test_fields_3d.py` (`update_chem` against a
float64 mirror reference, a uniform field decaying exactly in every layer, mass conservation without decay, no z
wrap; `update_resource` regeneration and exact conservation; `sample_fields` against a float64 reference including
the boundary layers and the x/y seams, a vertical ramp (positive up, halved at the floor and the surface, rotated by
the pitch), the pitch required; `deposit_chem` exact over the volume). `tests/test_lifecycle_3d.py` (a corpse from
every layer lands in its column's floor cell and the total is unchanged; the `R_HARD` cap at the floor cell, with
compile). `tests/test_sensing_3d.py` (a float64 reference of all 84 inputs on a random volume scene; a level scene
against the 2D inputs bit for bit; a pitch rotation case; `IN_GROUNDED` inclusive at `z_ground` and false above it and
at the surface, with `IN_HEIGHT`; `grounded` required; dead and padded neighbours and empty bond slots zero the DZ
inputs; `|x| ≤ 1` at `±PITCH_MAX`; compiled against eager).

### 27.7 Energy: the pitch turn cost, the buoyancy costs and `HEAT_BUOYANCY` (PR 3)

**Costs** (S8 in 3D, `energy.pay_costs3 → Costs3`; the 2D `pay_costs` and `Costs` are unchanged). §10.3's dues with
the pitch change in the turn term, plus a third due for buoyancy:

```text
basal_due  = basal_cost + brain_cost + bond_upkeep·n_bonds                                  → HEAT_BASAL
action_due = §10.3's, with the turn term floor(turn_cost·(|turn| + |Δφ|)/turn_max)          → HEAT_ACTION
buoy_due   = floor(buoyancy_cost·(buoy_cap·b)²) + floor(buoyancy_cap_cost·buoy_cap)          → HEAT_BUOYANCY
solvent    = alive & (E ≥ basal_due + action_due + buoy_due)
paid_basal = min(basal_due, E);   paid_action = min(action_due, E − paid_basal)
paid_buoy  = min(buoy_due, E − paid_basal − paid_action);   E ← E − paid_basal − paid_action − paid_buoy
```

- `Δφ = Motion3.pitch` and `b = Motion3.buoy` (§27.2), masked by `alive`. The lift is `buoyancy_max·buoy_cap·b`
  (§27.3), so `buoy_cap·b` is the fraction of `buoyancy_max` in use: the use term is quadratic like
  `move_cost·a²`, and no lift costs nothing. The upkeep is linear in the cap and paid every step by every alive
  agent, also at `b = 0`: holding a cap is a running cost, using it another.
- Each `floor` is float32 cast to int32 (§10.3). Both buoyancy terms are traced only when `buoyancy_max > 0` (a
  static branch); without the actuator the due is 0 (written `E·0`), and `validate` requires both costs to be 0.
  `paid_buoy = min(due, E − paid_basal − paid_action)` derives from `E` whatever the due, so it is never a
  constant output under the compile rule of I14.
- The payment order is basal, action, buoyancy, with solvency on the total. A solvent agent pays exactly §10.3's
  basal and action amounts; the order only decides how an insolvent agent's energy is split. Insolvent agents end
  S8 at `E = 0`, `actions.mask(·, solvent)` cancels their `Actions` and their `Motion3`, and they die `STARVED` at
  S14 (§10.3). Dead rows pay 0.
- With `Δφ = 0` and `buoy_cap = 0` the result is `pay_costs` bit for bit on CPU: `|turn| + 0` is exact and the
  buoyancy due is 0.

**Keys.** `energy.buoyancy_cost` and `energy.buoyancy_cap_cost` (int, quanta per step, default 0) are in
`POST_V01_KEYS` and `THREE_D_KEYS`, so a 2D world must keep them at 0 and no 2D hash moves. Each is `≥ 0` (§17's
per-cost rule lists them) and, in 3D, `< 2¹⁶` (each term is at most its coefficient, like the demands of §10.9);
both must be 0 when `buoyancy_max = 0`.

**`HEAT_BUOYANCY`.** In 3D, ledger slot 15 is the sink `HEAT_BUOYANCY = Σ paid_buoy` (§27.8). `RESERVED` stays the
canonical `LedgerIdx` name and stays 0 in 2D, where it is still checked; `state.sinks(3)` is §10.8's five sinks plus
slot 15, and `state.ledger_names(3)` names the slot in metrics rows. The 3D closures:

```text
TOTAL(after) − state.total == L[0] + L[1] − (L[2] + L[3] + L[4] + L[5] + L[6] + L[15])
TOTAL(now) == total0 + cum[0] + cum[1] − Σ_{k ∈ sinks(3)} cum[k] == state.total
S8:  E_in − E_out == paid_basal + paid_action + paid_buoy
```

`device_check` (`LEDGER`), `full_check` and `ledger_error` evaluate them over `sinks(nd)` (§27.8). Gravity has no
term (§27.3), and every other flow keeps its §10 term: harvest takes from the agent's own 3D cell, combat spill and
residue go to the victim's own 3D cell (docs/3D.md §12 Q19), and corpses sink to the floor cell under `FLOW_CORPSE`
(§27.6).

**Column headroom.** §10.9's bound `R_HARD + occ·(2·e_max + S) < 2³¹` holds per destination cell. In 3D a floor cell
receives the corpses of every layer of its column (§27.6) as well as the spills and residues of its own layer, so
`occ` counts the agents of an x/y column: the step gives `device_check` the largest column occupancy,
`scatter_add_i32(FG², fcell mod FG², 1, alive).max()` (§27.8), checked against the unchanged `Dims.occ_limit`
(`INT_HEADROOM`). A column holds at least as many agents as any of its cells, so the bound also covers the cells
above the floor. A dense packing (FCC, density 0.74) puts about `0.74·field_cell²·height / (4π r³/3)` agents in a
column: 68 at field cell 4, height 24 and r = 1, and 226 at height 80, against `occ_limit = 1 060`.

**Carrying capacity.** In 3D `c_ref` (§10.10) also holds a founder's cap upkeep `floor(buoyancy_cap_cost·buoy_init)`
(float32, as `pay_costs3` computes it). The regeneration is weighted by the layers the agents live in (§27.10): PR 3
summed every layer, which gave `tiny3d` 1 456 against 426 for `tiny`; PR 4 took the floor layer alone, 409, where the
numerators are equal (the floor layer regenerates as `tiny`'s lattice, §27.5) and a 3D founder costs more (a wider
brain, the cap upkeep), and PR 6 weighs the layers by occupancy, which is that 409 when every agent is on the floor.
The two populations run about level (peaks 441 and 426 in the first 400 steps on CPU).

**Tests.** `tests/test_energy_3d.py` (`pay_costs3` against a float32 reference on mixed solvent and insolvent
populations, with the per-function closure and dead rows paying 0; the exact buoyancy terms and the payment order;
the upkeep without lift; the pitch in the turn term; level and capless equal to `pay_costs` bit for bit, with and
without the actuator; compile, including the `paid_buoy` that is 0 by construction; the keys: defaults, the pinned
hash, rejection in 2D and each 3D rule).

### 27.8 Step wiring, counters, invariants, metrics and enablement (PR 3)

**Step wiring** (`step.step`). One pipeline (§3, §16.3) with static `if d3:` branches (`d3 = dims(cfg).d3`); every
2D line is unchanged under the `else`. The 3D branches:

| stage | 3D |
|---|---|
| S2, S3 | `find_neighbors` dispatches to its 3D body (§27.2); `bond_geometry` takes (N,3) unchanged |
| S4 | `gr = physics.grounded(s.pos)` from the start-of-step positions (not masked by `alive`); `sample_fields(…, pitch=s.pitch)` (§27.6) |
| S5 | the 8-wide descriptor; `build_inputs(…, grounded=gr)` |
| S7 | also `mo = actions.decode_motion3(y, alive)` (§27.2) |
| S8 | `energy.pay_costs3(E, act, mo, n_bonds, brain_cost, buoy_cap, alive)` (§27.7); `mo` and `act` masked by `solvent` |
| S14 | corpses sink to the floor cell (§27.6) |
| S15 | `forces3(pos, heading, pitch, act, mo, buoy_cap, …)`, `integrate3(…, mo.pitch, gr, survivor)`; the pitch is written back with `pos`, `vel` and `heading` (§27.3) |
| S16, S17 | the 3D `reproduce` (§27.4); `deposit_chem`, `update_chem` and `update_resource` on the volume (§27.6) |
| S18 | `paid_buoy` booked in slot 15, `Counter3D` appended, `INT_HEADROOM` on the column occupancy (below and §27.7) |

The one grounding predicate feeds `IN_GROUNDED`, the floor friction of `integrate3` and the `GROUNDED` counter, so the
three always agree. Every output of the compiled step stays derived from its inputs (the MLX 0.32 compile rule of
I14): `paid_buoy` is computed from `E` even when it is 0 without the actuator (§27.7), and the `Integrated3` counts
are reductions.
`bench._StagedStep` mirrors every branch (its `decode` stage also returns `Motion3`, `interact` masks it, `physics`
returns `Integrated3`), and `tests/test_bench.py` checks it bit for bit against `step`, with and without the actuator.

**Counters** (`state.Counter3D`, `state.n_counters(nd)`). `CounterIdx` keeps its 27 members (§19.1). A 3D
`StepOut.counters` is (30,): the 27, then

| idx | name | counts |
|---|---|---|
| 27 | `Z_FLOOR_CLAMPS` | survivors stopped by the floor this step (pre-clamp `z ≤ z_lo` with `v_z < 0`; `Integrated3.z_floor`, §27.3) |
| 28 | `Z_SURFACE_CLAMPS` | survivors stopped by the surface (pre-clamp `z ≥ z_hi` with `v_z > 0`) |
| 29 | `GROUNDED` | agents grounded at the start of the step that survived it (`physics.grounded(s.pos) & s.alive & ~dead`; `IN_GROUNDED` is not masked). A row that dies in the step (starvation, age, combat, a catastrophe) is not counted, so `Σ GROUNDED ≤ Σ N_ALIVE` and the interval's `grounded_frac` (§27.9) is a share in [0, 1]; counted over the start-of-step alive set (as PR 3 first defined it), a mass death inflated the numerator against the end-of-step `N_ALIVE` (review of PR 4) |

`n_counters(nd)` is 27 in 2D and 30 in 3D. `metrics.interval_row` and the viewer's accumulator size by it; the
runner stacks the step's counters as they come. Under gravity every surviving agent resting on the floor clamps
every step, so `Z_FLOOR_CLAMPS` is about `GROUNDED` and is not an anomaly signal; the surface count is the separate one.

**Ledger.** In 3D the step stacks `paid_buoy` as one more per-agent column of §10.8's exact sum and books it in slot
15 (`HEAT_BUOYANCY`, §27.7); in 2D the slot stays the constant 0. `StepOut.ledger` stays (16,).

**Invariants** (§22.2 and §22.3 in 3D). There is no new `Diag` bit. The 2D expressions are unchanged and the 3D
terms are separate static additions.
- `device_check`: `LEDGER` closes over `sinks(nd)`; `NONFINITE` also covers `pitch` and `buoy_cap`; `POS_RANGE`
  also flags z outside `[z_lo, z_hi]` and `|pitch| > PITCH_MAX`, both against the float32 bounds the engine clips
  with, on every row (a dead row holds `(0, 0, z_lo)` from init or the last in-range position and pitch of its
  agent, §2.2, §27.3); `INT_HEADROOM` receives the column occupancy (§27.7). The 2D rule `0 ≤ pos < L` also
  runs on z, where the z bounds imply it (`2·height ≤ L`).
- `full_check`: the closures over `sinks(nd)`; the rule `RESERVED == 0` in 2D only; and `_check_rows_3d`, on every
  row: z in `[z_lo, z_hi]`, `|pitch| ≤ float32(PITCH_MAX)` and `buoy_cap ∈ [0, 1]` (each finite), and no alive agent
  at `z_lo` with `v_z < 0` or at `z_hi` with `v_z > 0`. The last two rules are exact: the clamp of §27.3 zeroes the
  stored `v_z` of every row it moves to a boundary, and a move it does not clamp ends strictly inside. A finite
  `buoy_cap` out of range is reported by `full_check` only, like the other genes.
- `_check_rows_3d` has its own meta-test (`_CHECK_FNS_3D`): each of its rules must be hit by a 3D corruption case
  whose message fragment matches that rule only. `full_check` calls it in 3D only, so the 2D meta-test of I20 and
  its sites are unchanged.

**Metrics** (§19.2 in 3D).
- `interval_row(…, *, nd = 2)` takes counters `(T, n_counters(nd))` and refuses a 2-D block of another width
  (`ValueError`) up front, where a reshape by 27 would fail later with a misleading step-count mismatch (a 3D
  `(9, 30)` block reshapes to ten rows against nine ledgers).
  The ledger is keyed by `ledger_names(nd)`, so a 3D row has `HEAT_BUOYANCY` where a 2D row keeps `RESERVED`. The
  runner and `analysis.resim` pass the run's dims. The `Counter3D` columns reach the row as the interval keys of
  §27.9.
- `snapshot_metrics`: `ledger_error` over `sinks(nd)`; `chem_mean` per channel over all `FZ·FG²` cells;
  `resource_mean` per cell of the volume (`E_field / n_cells`), so the upper layers dilute it compared with a 2D
  floor.
- `carrying_capacity`: §27.7, and the occupancy-weighted rule of §27.10.

**Enablement.** `config.DIMS_SUPPORTED = (2, 3)`: a config with `world.dims = 3` validates under §27.1's rules and
runs through `alife run`, `resume`, fork, `inspect` and `bench`. Still refused in 3D: the 2D-only tools of §27.1
(`render`, `view`, `analysis.signals`; exit 2). `director.enabled` was refused until PR 4 (§27.10). 2D is unchanged:
every 2D trajectory, config hash, metrics row and checkpoint key set (the CPU digest matrix of 2D trajectories and
runs matches bit for bit; the 2D golden is not re-pinned).

**`tiny3d`** (`presets/tiny3d.toml`; the 3D tests, CI and a first look) is `tiny` plus every key of
`THREE_D_KEYS` and a 3D-stable `chem_diffusion`. The values are provisional; 3D tuning is docs/3D.md PR 6.

| key | value | why |
|---|---|---|
| `world.dims`, `world.height` | 3, 24 | `FZ = 6`, `G_z = 4`; 24 rather than 32 keeps contacts nearer `tiny`'s rate (`2·height ≤ L = 128`) |
| `physics.gravity` | 0.03 | sinking 0.03 per step; climbing at full pitch `0.1·sin 80° − 0.03 ≈ 0.068` |
| `physics.floor_eps` | 0.1 | grounded within 0.1 of `z_lo` |
| `physics.buoyancy_max` | 0.04 | a full cap lifts at `0.04 − 0.03 = 0.01` per step; a founder's (0.5) only slows sinking |
| `energy.buoyancy_cost` | 16 | hovering costs the same either way: buoyancy at `buoy_cap·b = 0.75` is `floor(16·0.75²) = 9` q/step, thrust at full pitch (`a ≈ 0.305`) `floor(100·0.305²) = 9` |
| `energy.buoyancy_cap_cost` | 8 | 4 q/step at a founder's cap 0.5 (≈ 3% of its basal and brain cost, ≈ 142 q), 8 at a full cap |
| `mutation.buoy_init`, `buoy_sigma` | 0.5, 0.02 | founders halfway; the cap drifts like `tag` |
| `fields.chem_diffusion` | 0.16, 0.10, 0.16 | `≤ 1/6` (§27.1) |
| `fields.benthic_lambda`, `photic_amp`, `photic_lambda` | 8, 0.5, 8 | a benthic band and a weaker photic band (§27.5) |

It keeps `tiny`'s `C = 16` and `K = 8`, which a floor pile under gravity overflows (§27.3, a warning bit); they are
sized for `64k3d` in PR 6. `tiny3d` with the `fast` overlay validates too (`K = 6 ≤ 27·12`).

**3D golden digest.** `tests/test_determinism_3d.py::test_golden_3d` (marker `golden`) pins `state_digest` (every
field of `State3`; `legacy_state_digest` is 2D only) of `tiny3d_cfg` after 50 compiled steps. The actuator and both
costs are on, so `ledger_cum[HEAT_BUOYANCY] > 0` (asserted) and the pin covers slot 15. Like §14.2's golden it is
keyed by the MLX version and the platform, and it is recorded on the CI platform, `Linux-x86_64-cpu` with MLX
0.32.3, never on the Mac (CPU kernels round differently across architectures). Procedure: while
`GOLDEN_3D["digest"]` is empty the test skips everywhere; on the pinned platform the skip message carries the computed
digest, which is then committed into `GOLDEN_3D`. A change of the 3D trajectory re-pins it the same way, deliberately. The
value `ffb17130…` was recorded this way at PR 3's head `3bbcab0` and reproduced in a second process.

**Tests.** `tests/test_invariants_3d.py` (a healthy volume scene with the bounds reached; integrated states pushed
into the floor and the surface pass the clamp rules; the closures with `HEAT_BUOYANCY`, and the same arrays failing
without it; slot 15 not a sink in 2D, where `RESERVED` is reported; both checks on each 3D corruption with a Diag bit;
the float32 bounds inclusive; `buoy_cap` range `full_check` only; a corruption case for every `_check_rows_3d` rule
and the meta-test; `_check_rows_3d` not run in 2D). `tests/test_metrics_3d.py` (`Counter3D` after `CounterIdx`; a
`(9, 30)` interval in 3D, refused as 2D; the 2D row keeps `RESERVED`; the snapshot over the volume with the closure
over slot 15; the carrying capacity with the upkeep; resim's comparison with the 3D names). `tests/test_step_3d.py`
(60 compiled `micro3d_cfg` steps, half the founders starting on the floor, `full_check` after each, the 2D counter
rules on the first 27, `Counter3D` against the state transition and the per-step and global closures; slot 15 equal to
the step's `Σ paid_buoy` and to the float32 due; more `HEAT_BUOYANCY` under an `OUT_BUOYANCY` bias; slot 15 off by one
caught by `device_check` and by both host closures; slot 15 zero in a 2D run; floor and surface clamps and grounding
in a scene; `GROUNDED` counting only the survivors of a mass death, with a `grounded_frac` of 1; an x/y column of
`occ_limit` agents, no 3D cell near the limit, trips `INT_HEADROOM`).
`tests/test_affordances_3d.py` (one eager step per `micro3d_cfg` scene: three attackers drain a victim, with the
closure; with regeneration and diffusion off the corpse lands in the floor cell and the spill and residue in the
victim's own cell; a mid-water and a grounded agent each harvest their own cell, never the floor cell below a
mid-water agent; a vertical pair bonds and its spring acts along z; `OUT_PITCH` steers and pays the turn cost; a
full cap rises at the net force and pays the float32 due; a zero cap does not lift, a cap without lift pays only the
upkeep; the surface clamps without grounding, and the brain sees `IN_GROUNDED` and `IN_HEIGHT`).
`tests/test_determinism_3d.py` (same seed, other seed, `runtime_checks`, eager against compiled, the golden).
`tests/test_cli_smoke_3d.py` (a 100-step `tiny3d` run through `cli.main`: clean rows with `HEAT_BUOYANCY`,
`inspect --check`, a truncating and bitwise resume, `render` and `view` refused with nothing written, a fork that
changes `world.dims`, `world.height` or `physics.radius` refused with exit 4, `analysis.resim` verifying the stored
rows and reproducing the step-100 state). New functions: `test_bench.py` (the staged pipeline equals the step in
3D, with and without the actuator; every bench part on `tiny3d`), `test_runner.py` (SIGINT and a bitwise resume in
3D; immigrants in the volume at `buoy_init`), `test_config.py` (`test_preset_table_tiny3d`), `test_cli_smoke.py` (a
2D row keeps `RESERVED`).

### 27.9 Metrics in 3D (PR 4)

A 3D row (§19.2) has the 2D keys unchanged, in their order, plus the keys below; a 2D row has none of them, so every
2D `metrics.jsonl` row is byte-identical. All are host-side observers (`metrics.py`): one host transfer as before,
no change to the compiled step, the state or the RNG. "Alive" means over the alive agents; the keys are `None` when
nobody is alive. Heights are the agent centre `z` in world units, `0 < z_lo ≤ z ≤ z_hi < height`; "height" is the
world height `world.height` (never `H`, the hidden size).

**Snapshot keys** (`metrics._snapshot_3d`, after `diag`; the order below):

| key | definition |
|---|---|
| `height_mean`, `height_q25`, `height_q50`, `height_q75` | mean and quartiles of `z` (`np.quantile`, linear, float64) |
| `surface_frac` | share of alive agents with `z ≥ z_hi − floor_eps`: within the grounding margin of the surface bound. The surface never grounds (Q2), so this is the mirror of the grounding band; with `floor_eps = 0` it is the share resting on the clamp |
| `speed_mean` | mean of `‖vel‖` (3-D) of the last step. The clamp zeroes `v_z` on a row it moves to a boundary, so a floor-resting agent contributes its horizontal speed |
| `pitch_abs_mean` | mean of `|pitch|` in radians, in `[0, PITCH_MAX]` |
| `buoy_cap_q10`, `_q50`, `_q90` | quantiles of the heritable cap `buoy_cap ∈ [0, 1]` (as `sigma_q*`) |
| `band_frac` | list of 4 shares of alive agents in the equal z bands `[b·height/4, (b+1)·height/4)`, `b = 0..3`; the top band includes the surface and a height on an edge belongs to the band above (`metrics.z_band`: `floor(4·z / height)`, with `4·z` exact in binary). Sums to 1 |
| `clade_height_mi` | plug-in mutual information in nats between an agent's clade (binned as `clade_counts`, immigrants in bin `n_init`) and its z band, from the host joint histogram (`np.bincount`). It is 0 exactly for a single clade and for independent labellings (the ratio `n_ab·n / (n_a·n_b)` is formed from exact int64 products); it is biased upwards by about `(clades − 1)(bands − 1) / (2n)` nats for small samples, so it is a descriptive measure of vertical niche separation, not a test. **When the clades are close to singletons it saturates at the entropy of the band and carries no information about vertical niche separation**: every run starts with `n_init` clades of one agent, and on `tiny3d` (seed 0, 6 000 steps, measured in review) the value stays at 0.90 to 1.00 of `H(band)` (1.381 of 1.381 at step 10, 0.34 of 0.37 at step 6 000) while the bias estimate is 0.6 to 1.5, larger than the value at every step. Read it only where the clades are few and large (a long run, or a coarsened clade labelling); a measure that is not dominated by singletons (large clades only, a bias correction or a shuffle baseline) is open for PR 6 |

The heavy snapshot (every `heavy_metrics_every`) adds, after the `bodies_*` keys and before `weight_diversity`,
`body_z_extent_mean` and `body_z_extent_max`: over the bodies (components of at least two alive agents, as
`bodies_n`), the extent `max z − min z` of the member centres, their unweighted mean over bodies and their maximum.
`z` is bounded and never wrapped, so no unwrapping is needed (x and y, which are periodic, have no extent key).
Both are `None` with no body, and are computed from the same bond-component labels as `bodies_*` (a
non-converged labelling is reported by `bodies_converged`, as there).

**Interval keys** (`interval_row(…, nd = 3)`, after `v_clips`): `z_floor_clamps` and `z_surface_clamps` (interval sums
of `Counter3D`), `grounded_frac` (`Σ GROUNDED / Σ N_ALIVE`: the share of agent-steps spent grounded, weighted like
the other per-agent fractions of the row, in [0, 1] because `GROUNDED` counts only the agents that survive the step
(§27.8) while `N_ALIVE` is the end-of-step count; `None` without alive steps) and `buoy_heat` (the interval sum of ledger slot
15, equal to `ledger["HEAT_BUOYANCY"]`, repeated as a plain key). Under gravity every grounded agent clamps each
step, so `z_floor_clamps ≈ Σ GROUNDED` (§27.8). No snapshot key is named `grounded_frac`: `interval_row` raises on a
key shared with the snapshot, and the snapshot has no grounded key at all (`alife inspect` computes the share of the
checkpoint state, §27.11). `analysis.resim` recomputes the interval keys with `nd` and compares them with the stored
row, as it does the shared ones; a row stored before PR 4 lacks them and is compared on the keys it has.

**`carrying_capacity_est`** (first row, `inspect`, the director): the occupancy-weighted rule of §27.10.

**Tests.** `tests/test_metrics_3d.py`:
- the snapshot keys against an independent host reference (quantiles, bands, surface margin, speed, pitch, the cap's
  quantiles, `clade_height_mi` by `Counter`/`math.log`, `body_z_extent_*` over three known bodies);
- band edges; the mutual-information limits (determined, independent, a single clade exactly 0, a hand value);
- the exact key sets and order in 2D (the 2D dict is unchanged, light and heavy) and 3D; the empty population (all
  `None`); a snapshot never colliding with the interval keys;
- the interval keys against host sums, `grounded_frac` weighting and `None`, the 2D row without them, resim flagging
  each of them;
- the carrying capacity (§27.10).

### 27.10 The director in 3D (PR 4)

**Carrying capacity (`K_est`)** (PR 4: the floor layer only; PR 6: occupancy-weighted, owner decision 2026-10-14). In 3D
`metrics.carrying_capacity` weighs the regeneration of each layer by where the alive agents live:

```text
K_est = Σ_k regen_k · min(1, FZ·p_k) / c_ref
```

- `regen_k` is the integer regeneration of layer `k`, the sum over its (FG, FG) slice of `fields.regen_rate(…)`; with the
  live levers (`step`, `env`) it is the live rate, as in 2D.
- `p_k` is the share of the alive agents whose field cell is in layer `k`: `fields.cell_index(pos) // FG²`, the layer
  `clip(floor(z / field_cell), 0, FZ − 1)` that harvest reads and deposits in (§27.5), computed by the engine's own
  float32 expression so a height on a layer edge falls where harvest puts it.
- `c_ref` is §10.10's, with the founder's cap upkeep (§27.7); `FZ` is the number of layers.
- `w_k = min(1, FZ·p_k)` is 1 for a layer holding at least its even share `1/FZ` of the agents and proportional to its
  occupancy below that. So `K_est` equals **the floor layer's alone** when every agent is in it (the benthic limit, the
  PR 4 rule, the 2D lattice's), **every layer's** when the column is used evenly (`p_k = 1/FZ`), and the floor layer's
  alone when **nobody is alive** (there is no occupancy to weigh, so `K_est` is defined for an extinct world). The sum of
  the weights is at most `FZ` and at least 1 with anyone alive.
- **For a handful of survivors the weights saturate.** With `n ≤ FZ` alive, an occupied layer holds at least
  `1/n ≥ 1/FZ` of them, so `w_k = 1` for every layer anyone lives in and 0 elsewhere. Twenty survivors one to a layer give
  the plain sum, 67 700 on `64k3d` against 17 800 for the floor layer alone; six survivors one to a layer on `tiny3d`
  give 1 456 against 409, so the extinction rescue (`round(0.1·K_est)` immigrants) sends 146 where the floor-layer rule
  sent 41; and a single survivor in the top layer gives that layer's regeneration alone, less than the floor-only value.
  `K_est` is therefore not continuous at `n = 0`: if the last survivors are spread out it is at its largest just before
  a total collapse and falls to the floor-only value when the last one dies. The nobody-alive rule keeps an *empty*
  world's capacity from being undefined, it does not keep a collapsing one's from rising.
- Founders uniform in `[z_lo, z_hi]` fill the two edge layers less than the others (`p_0 = p_{FZ−1} = (field_cell −
  r)/(height − 2r)`, 3/78 on `64k3d`, so `w = 0.77` there), which is why uniform founders give 61 500 on `64k3d` and
  not the 67 700 of the plain sum over its layers.

Every consumer reads it except the director's extinction-risk test (below): the
`carrying_capacity_est` of the first metrics row (the state at the first row), `alife inspect`, and the director's
`n / K_est` (the crowding ratio of the log and the prompt), rescue size and prompt capacity line (whose qualifier now
reads "food weighted by where the agents live: the floor layer's alone if all are in the floor layer, every layer's if
they are spread evenly", `prompt.K_NOTE_3D`; the floor layer is `z < field_cell`, which is not the grounded state, `z ≤ r +
floor_eps`: a population hovering at `z` 2 to 3.9 is entirely in it and not grounded). 2D is unchanged (a static
branch; the value is the lattice's integer regeneration over `c_ref` as before). The reasons for replacing the
floor-only rule:
- It undercounts a population that has colonised the column. On `tiny3d` (cloud-6, 20 000 steps) seed 1 spreads
  through the water after about 10 000 steps (28% grounded, a surface band of 23%, a body spanning 19 of 24 units) and
  the floor-only `K_est` (about 450) is then half the realised population.
- The summed layers (PR 3: 1 456 on `tiny3d`, 3.6 floor lattices' worth) count food that no agent can reach when the
  population rests on the floor; the occupancy weights count a layer's food only as far as agents are there to use
  it, and they follow the population as it moves.
- Food above the floor still costs lift to reach (§27.7); the rule does not price that, it only counts the layers in use.

What it is and is not. It depends on the state's agents as well as its patches (so it is a state function, not a
property of the seed), it is an upper estimate of what the occupied layers supply and not a prediction of the realised
population (on `tiny3d` at g = 0.03 the population is at 0.56 of the floor-only value and 0.39 of the
occupancy-weighted one at step 2 000, a ratio that holds from step 2 000 to about 8 000 to 10 000 and is not a fixed
point: the long runs grow again, seed 0 to 1.08 of the floor-only value at 20 000 steps and seed 1 to the cap; at
g = 0.015 it is 0.2 to 0.35 for thousands of steps, §27.13), and a retuning of the vertical profile moves it whenever
the column is in use. On `DEV3D` (`64k3d` shrunk, §27.13) the cloud run (cloud-7, 2 500 steps, C = 24) read 14 190 on
the initial state and 12 759 on the final one against a realised 3 100 agents, **about 4× the population**, and the
initial `K_est` of `64k3d` itself is 65 167, above its `n_cap` of 64 512. So the occupancy-weighted value is a ceiling on
what the occupied layers could feed, not a level the population approaches. The threshold that reads it (`rescue_imm_frac`
0.1) was set against a 2D `K_est` and is unchanged; its 3D calibration is §27.13's open item. `risk_k_frac` (0.2) no
longer reads it in 3D: the extinction-risk test divides by the floor layer's capacity, below.

**The extinction-risk test reads the floor layer's `K` (owner decision 2026-10-15).** In 3D the director's flag is
`n_alive / K_floor < risk_k_frac` **or** `fall > risk_fall`, where

```text
K_floor = regen_0 / c_ref        (metrics.floor_carrying_capacity)
```

is the floor layer's integer regeneration over the same `c_ref` (PR 4's rule), live under the director's `step` and
`env` like `K_est`, whoever is alive and wherever they are. It is carried in the vitals as `k_floor` with its ratio
`n_over_k_floor`; `Vitals.n_over_k_risk` is the ratio the flag compares (`n_over_k_floor` in 3D, `n_over_k` in 2D).
`Director._vitals` computes `k_floor` for a 3D config only (a static branch). **Nothing else changes:** `k_est` and
`n_over_k` stay the occupancy-weighted values, so the capacity estimate and the crowding ratio of the viability line,
the log's `k_est` and `n_over_k`, the rescue's size (`round(rescue_imm_frac · k_est)`), the first metrics row and
`alife inspect` read the occupancy rule, as does every other consumer; the `fall` criterion reads no `K`.
- **Why.** The occupancy-weighted value is a function of where the agents are, which made it a poor divisor for a
  threshold on extinction: (1) a rescue raises its own `K_est` (immigrants placed uniformly in the volume put every layer
  near its even share), so the flag stayed set for a population that had recovered, and the rescue stayed wanted and
  blocked by its cooldown at every boundary (below); (2) spread founders and immigrants were flagged at about three times
  the floor layer's size (`n/K_est` 0.17 to 0.19 at the start of a healthy `tiny3d` run, 0.21 to 0.30 for thousands of
  steps afterwards at `64k3d`'s gravity); (3) cloud-7 measured the occupancy-weighted value at about 4× the realised
  population on `DEV3D`, so 0.2 of it is not a measure of risk. The floor layer's food is the stable base of a
  population: it moves with the patches and the live levers (as the 2D `K_est` does) and not with the agents' depth or
  with a rescue.
- **Margins.** The ratio the flag reads is the floor-only series measured in PR 4 and review: 0.54 to 0.71 on `tiny3d`
  at g = 0.03 (0.55 to 0.70 at 0.015, 0.63 at step 10 and 1.01 at 110 after the founders' first reproduction), 0.70 to
  0.73 on `DEV3D` at steps 1 300 to 2 500 (3 000 to 3 100 agents on a floor-layer `K` of 4 260 to 4 350; the cloud-7 run
  ends at 3 134 agents, in the same range), and at `64k3d`'s start 16 384 / 17 800 = 0.92. The margin to 0.2 is a factor of 2.7
  to 3.5 in ordinary runs, where it was 4 to 25% under the occupancy rule. None of it is measured at `64k3d`.
- **The rescue.** The 3D Gardener of the tests (`tests/test_director_3d.py`, 40 founders) is at `n/K_floor` 0.098 at the
  first boundary: the rescue fires, with 131 immigrants (0.1 of the occupancy-weighted `K_est` of 1 309, where the
  floor-layer rule of PR 4 sent 41), the population reaches 250 to 290, and the flag is clear from step 60 (the `fall`
  criterion sets it once, at step 40, which the rescue's cooldown blocks: one journal note) and stays clear, the
  population at 0.40 to 0.48 of the floor layer's `K` from step 80. The rescue still raises `K_est` (3.2×; its size and
  the viability line read it), but that no longer feeds the flag. **The size is still the occupancy-weighted one**, which
  is a ceiling of the kind the previous bullet describes. It matters where `0.1·K` is under the menu's cap of 1 024
  immigrants (`IMM_PENDING_CAP`, `actions.MENU`): on `tiny3d` (131 against 41), not on `64k3d`, where `0.1·K_floor` is
  already 1 780 and an ordinary rescue asks for the cap whichever `K` sizes it. The rescue size is part
  of the open calibration of §27.13 (to measure 5).
- **Prompt.** In 3D the viability line gives both: `population 200 = 0.489 of the capacity estimate K_est 409
  (food weighted by where the agents live: …), 0.8 of the floor-layer capacity K_floor 250 (the extinction-risk flag
  compares the population with this one, not with K_est), window low …` (`prompt.K_FLOOR_NOTE_3D`; the numbers are the
  tests'). The clause appears when the vitals carry `k_floor`, so the 2D messages are byte-identical (pinned by
  sha256).
- **Log.** A 3D entry's `vitals` has `k_floor` and `n_over_k_floor` after `n_over_k`; a 2D entry has the keys it has
  always had, in the same order, so 2D `director.jsonl` files are byte-identical. The reason of a 3D rescue reads
  `n/K_floor 0.098, fall 0.0` where 2D reads `n/K_est …`.
- **`RULES_VERSION` stays 1.** The rule tier's behaviour changes in 3D only, no 2D log, replay, prompt or `inspect`
  line changes by a byte (the 2D director tests and the sha256 pins pass unchanged), and `RULES_VERSION` is the version
  of the rule parameters (`RuleParams`: thresholds, windows, weights, ranges, budgets), none of which changed. A bump
  would rewrite `rules_version` in the header of every 2D log and make a live resume of a running 2D evaluation arm
  append a second header (`log.needs_header` compares the policy), for no change of its behaviour. The same holds for
  PR 6's own `K_est` rule, which did not bump it either. The consequence is the one already recorded under "Resuming a
  live run" below: the header does not say which `K` the flag read, so a 3D live run begun under the occupancy rule
  (none of value exists: 3D has only run in tests) should be replayed or restarted, not live-resumed here.

`tests/test_metrics_3d.py`: the weights and `K_est` on built scenes against a float64 host reference written
independently of `fields` (all benthic, evenly spread, mixed, the top layer alone, nobody alive, dead rows with
positions in upper layers not counted, heights on and around the layer edges, the live environment), the benthic limit
equal to the floor-only value and the even column to the plain sum, a 2D state unaffected, and the band of the lazy
initial state of `64k3d` (§27.13). The PR 4 floor-layer tests now describe a world nobody lives in (their scenes have
no alive agent), and `tiny3d` against `tiny` is compared with every founder on the floor.

**Scope.** The director is host code (§26.1): it reads the metrics rows and edits the `env_*` fields and `patch_pos`
between compiled steps. The compiled step (apart from what §27.4 and §27.6 already built: the column mask and the
volume immigrants), the state, the RNG, `RULES_VERSION`, the ten-component descriptor of §26.5 and every rule
parameter of §26.7 are unchanged in 3D, and the director reads none of the 3D row keys of §27.9. `validate` no longer
rejects `director.enabled` with `world.dims = 3`: the live and replay modes, the LLM tier and the `random` persona
(§26.11) run in 3D, under the same menu, budget and cooldowns. The 3D differences are these.

| part | 3D |
|---|---|
| catastrophe | a **vertical column**: `catastrophe_mask` measures the x/y minimum-image distance from `env_cat_center`, which stays (2,), so `params.center` is `[x, y]` as in 2D and every agent over the disc dies with probability `kill_p`, at any height (§27.4) |
| radius bounds | `actions.radius_bounds(cfg)` = `[20·L/1024, min(200·L/1024, L/2)]` in 3D; 2D keeps `[20, min(200, L/2)]` (`tiny3d` `[2.5, 25]`, a width-320 world `[6.25, 62.5]`). The 2D range was written for a 1024-wide world, where the largest disc covers π·(200/1024)² = 12% of the floor; at width 320 it would cover 78% (docs/3D.md §12 Q10). The scaling keeps the largest column at 12% of the floor at any width. `CATASTROPHE_RADIUS` and `CATASTROPHE_REF_SIZE = 1024` are the constants. The bounds apply to the live clamp, to replay validation (a 2D-sized radius in a 3D log is `ActionError`, exit 2), to the LLM schema and prompt, and to the `random` persona's sampler. The rules' own request (`cat_radius_frac`, 0.1 to 0.25 of L) is clamped to the upper bound |
| patch moves | x and y only. `patch_pos` and `patch_vel` stay `(PP, 2)`; a move's `params` are `{patch, x, y}` and `prev` `{patch, x, y}`, wrapped with `ops.wrap` and capped at L/4 in the x/y minimum image, as in 2D; a `z` (or `dz`) is not on the menu, and `validate_entry` rejects it. A patch is a column: its gain multiplies the fertility of every layer (§27.5), so `habitat_loss` and `fertility_shift` act on the whole column, and `regen_pulse` and `set_seasons` act on every layer |
| immigration | `immigration_pulse` adds founders to `env_imm_request` as in 2D; the step admits them uniformly in the volume at `buoy_init` (§27.4) |
| vital signs | the ten-component descriptor, the window aggregates and the archive are functions of the rows and are unchanged; `K_est` is the occupancy-weighted value above, so `n/K_est` is the population against what the layers its agents live in supply. The vitals also carry `k_floor` and `n_over_k_floor` (3D only), and **the extinction-risk flag compares `n_over_k_floor`, not `n_over_k`, with `risk_k_frac`** (owner decision 2026-10-15, above) |
| log header | `"dims": 3` after `seed`; a 2D header has no such key and keeps its key set (§26.3). `needs_header` compares headers without the start step, so a resumed 2D or 3D run with an unchanged policy appends no header |
| log validation | `log.validate_log` (replay, `--replay-log`, the yoke log of the `random` persona) raises `LogError` (exit 2) for a header whose `dims` (absent = 2) differs from `world.dims`: a 2D log is never replayed or followed in 3D, nor the reverse |
| LLM prompt | `prompt.effects(cfg)` selects `EFFECT` (2D, byte-identical to build step 3) or `EFFECT_3D`, which differs in the catastrophe line: a vertical column, a disc in x and y from the floor to the surface; `prompt.world_text(cfg)` says "periodic in x and y, with a floor and a surface at height" followed by the number `world.height` where the 2D text says "a torus"; the user message gives the world as `L x L x height` (three numbers), qualifies the capacity estimate with `prompt.K_NOTE_3D` (" (food weighted by where the agents live: the floor layer's alone if all are in the floor layer, every layer's if they are spread evenly)", §27.10; PR 4 had "(floor-layer food only)") and adds, on the viability line, the ratio and number of the floor layer's capacity with `prompt.K_FLOOR_NOTE_3D` (", 0.8 of the floor-layer capacity K_floor 250 (the extinction-risk flag compares the population with this one, not with K_est)", from the vitals' `k_floor`); the catastrophe radius bounds are the scaled ones. Nothing else changes, and the 2D system and user messages are byte-identical (pinned by sha256) |

**Calibration (provisional; measured with PR 4's floor-layer `K_est` of 409, which is again the `K` the flag reads).**
The occupancy-weighted ratio `n/K_est`, which is the `n_over_k` of the log and the prompt (and was the first criterion's
ratio under PR 6's rule), is nearer 0.2 than the floor layer's: on the same `tiny3d` run (seed 0, 2 000 steps, director
off, sampled at steps 10, 50, 100, 110, 200, 300, 500, 1 000, 1 500 and 2 000) it is 0.19 at step 10 (256 founders
spread over the column, `K_est` 1 362), 0.33 at steps 50 to 110, 0.27 to 0.28 at 200, 300 and 500 and 0.33 to 0.39 from
step 1 000 on (`K_est` falling to 608 as the agents settle on the floor, 77% grounded at step 2 000), against the 0.54 to
0.71 of the floor layer's ratio, which since 2026-10-15 is the one the first criterion compares with the threshold. The
second criterion (`fall`) reads no `K` and is as measured. The extinction-risk flag has two criteria (§26.7):
`n/K < risk_k_frac` (0.2, with `K` the floor layer's `K_floor` in 3D) **or** `fall > risk_fall` (0.3), where `fall` is
the drop of the window minimum below the highest window mean. Measured on
`tiny3d` with the director off (seed 0, 2 000 steps, `K_est` 409), the first criterion is far from firing: `n/K_est`
is 0.63 at step 10, 1.01 at 110 after the founders' first reproduction and 0.54 to 0.71 from step 210 on, and the same
run has 77% of the agents grounded at the end (the ecology is PR 6's). The second criterion depends on the rule cadence,
because the founders' boom and bust is a fall of that size within a few hundred steps. Measured in review (CPU,
Gardener, seed 0, 6 000 steps, `K_est` 408.6, vitals recomputed with `director.vitals.Tracker` over the stored rows):

| `rule_every` | first extinction-risk flag |
|---|---|
| 50 | step 200 (`n/K_est` 0.728, `fall` 0.325) |
| 100 | step 100 (`n/K_est` 1.029, `fall` 0.348) |
| 200 | step 200 (`fall` 0.309) |
| 500, 1 000 | none (at 1 000: `n/K_est` 0.55 to 0.68, `fall` 0.04 to 0.16) |

So the flag is off at baseline at the default cadence of 1 000 steps, and a rescue there needs a real collapse; at
`rule_every ≤ 200` the early boom and bust trips the fall criterion and the rescue fires at the first boundary. The
2D `tiny` preset behaves the same way (it flags at cadences 50, 100, 200 and 500), so this is a property of the rule
set on a small, boom-and-bust preset and not of the third dimension, but a test or a run that uses a short cadence
on `tiny3d` (the tests use 50) must expect a rescue. These values belong to the provisional `tiny3d`; the `64k3d`
preset has to be measured the same way, at its own cadence, before the first director run.

**The occupancy-weighted ratio at `64k3d`'s gravity, 0.015 (review; CPU, director off; what the flag read under PR 6's
rule).** The numbers above are `tiny3d`'s own gravity, 0.03. At 0.015 the population sits lower against the
occupancy-weighted `K_est` for thousands of steps:
- `tiny3d`, seeds 0 and 3, sampled every 10 steps: `n/K_est` has a minimum of 0.207 and 0.246 over steps 50 to 500
  (medians 0.224 and 0.259), 0.215 and 0.238 over 500 to 1 000 (medians 0.221 and 0.251) and medians of 0.26 to 0.30 from
  step 1 000 to 6 000. Seeds 1 and 2 (sampled every 250 steps) and g = 0.01 (seed 0) read 0.20 to 0.26 at steps 500 to
  1 500. The floor-only ratio is 0.55 to 0.70 throughout.
- Over steps 10 to 40 every seed is under 0.2 (0.17 to 0.19: the 256 founders are spread over the column), so under
  the occupancy rule a `rule_every` of 40 or less flagged a healthy start (the floor layer's ratio is 0.63 at step 10).
- `DEV3D` (`64k3d` shrunk, §27.13): `n/K_est` falls from 0.30 at step 300 to 0.236 to 0.249 at steps 1 000 to 2 500
  (3 000 to 3 100 agents, `K_est` 12 400 to 13 200, a floor-layer share of 0.48 against the 0.70 of the sizing).
- The "0.27 to 0.39" and the 0.39 above are the g = 0.03 values (seed 1 at 0.03 reads 0.35 to 0.49 at steps 1 000 to
  6 000).

No run crossed 0.2 after step 50, so the margin of the occupancy ratio was thin (4 to 25%), and a director run on
`64k3d` would have had to calibrate `risk_k_frac` against it first. The flag now reads the floor layer's ratio, 0.55 to
0.70 on `tiny3d` at 0.015 and 0.70 to 0.73 on `DEV3D` (a margin of 2.7 to 3.5×, above). `risk_k_frac` at `64k3d` is still
to be confirmed in the tuning run (§27.13, to measure 5), but it is no longer a gate with a thin margin.

**A rescue raises its own `K_est` (review; the Gardener of `tests/test_director_3d.py` with PR 4's 40 founders), which
kept the flag set under PR 6's rule and no longer does.** The
rescue admits its immigrants uniformly in the volume (§27.4), which puts every layer near its even share, and its 1.5×
regeneration pulse scales every layer that counts. 40 founders spread over the column give `K_est` 1 309 (3.2× the floor
layer's 409) and `n/K_est` 0.03: a rescue of 131 immigrants. The population then reaches 250 to 290 (twice what the
floor-layer rule produced in the same run, 133 to 140, with its 41 immigrants), with `K_est` at 1 700 to 2 000 and
`n/K_est` at 0.14 to 0.15 at every boundary from step 60 to 200: **the extinction flag stays set for a population that
has recovered**, where the floor-layer rule cleared it from step 60 to step 140 (`n/K_est` 0.21 to 0.23 on its `K_est` of 615). The same arithmetic
at `64k3d`: 5 000 benthic agents and 1 024 uniform immigrants give `p_k` of about 0.0085 in each upper layer, weights of
about 0.17, `K_est` = 1.5 × (17 800 + 0.17 × 49 900) = 39 400 and `n/K_est` = 0.15, against 6 024 / 26 700 = 0.23 for the
floor-layer rule, which clears. And before the rescue, 40 spread founders already give a `K_est` 3.2× the floor-only
value, so the 0.2 threshold on the occupancy ratio fell at 262 agents instead of 82.

*Resolution (owner decision 2026-10-15).* PR 6 left this as it was (the rule was the owner's of 2026-10-14, and the
director's rules and thresholds belong to the evaluation), pinned by two tests and with three options for the
calibration gate: weigh `p_k` over the agents older than a few hundred steps, so that a rescue's first generation does
not count; a 3D threshold for the risk criterion; or the floor-layer value for the risk criterion and the occupancy-weighted
one for the rescue's size. The owner took the third (above): the flag reads `n/K_floor`, the rescue's size and every other
consumer keep `K_est`. The two tests were replaced by their converses
(`test_a_spread_population_is_judged_against_the_floor_layer_not_the_occupancy_weighted_k` on built scenes: the same flag
for a spread and a benthic population of the same size, set below 0.2·409 = 82 agents for both;
`test_a_3d_rescue_raises_its_own_k_est_but_no_longer_keeps_the_flag_set` on the 40-founder run: `K_est` 3.2× the floor
layer's, `n/K_est` 0.14 to 0.15, `n/K_floor` 0.40 to 0.48, flag clear from step 60 to the end).

**Resuming a live run (review).**
- *The `K` rules are not in the log header* (`rules_version` is still 1 and no config hash changed), so a 3D live run
  begun before PR 6 and live-resumed on this code switches from the floor-only to the occupancy-weighted `K_est` for the
  vitals' `k_est` and `n_over_k` (on `tiny3d` `n/K_est` at step 10 goes from 0.63 to 0.19) and for the rescue's size
  (0.1·K_est) at the resume with no trace in `director.jsonl`. Its extinction flag does not change: it reads the floor
  layer's `K`, which is PR 4's rule (the decision of 2026-10-15; under PR 6's occupancy rule the flag itself would have
  switched). A replay is not affected (it applies the logged entries and never recomputes the vitals), and no 3D director
  run of any value exists (3D has only run in tests), but such a run should be **replayed or restarted, not
  live-resumed**. If a marked switch is wanted, the 3D header can carry a `k_est_rule` key (3D only, so that 2D headers
  keep their key set) and `needs_header` appends a header at the first resume.
- *The blocked-rule note is per process.* `Director._blocked`, the rules whose blocked note was journalled at the
  previous boundary (a persisting condition is noted once), is not rebuilt from the run directory: a rule that stays
  blocked across a live resume (an extinction rescue under its cooldown, a monoculture breaker) is noted again at the
  resumed process's first boundary, so the `journal.jsonl` of a resumed run is not byte-identical to the uninterrupted
  run's. `director.jsonl`, the archive, the rows and the state are identical. The journal holds only the *new* notes, so
  the set cannot be read back from it; the fix needs the previous boundary's state or a persisted set (for example in
  `director_archive.json`, written at each checkpoint). The code is shared with 2D. It is **not fixed in PR 6** (the
  evaluation runs on the director; it is a separate PR), `test_a_blocked_note_is_journalled_once_across_a_resume` pins it
  as a strict `xfail` (2D and 3D) to be un-marked with the fix, and the end-to-end resume test below starts 40 founders,
  for which the flag has cleared at step 60 and no rule is blocked at the resume point (step 100; under PR 6's
  occupancy rule 40 founders left the flag set and the rescue blocked at every boundary, so that test started 150, for
  which the floor-layer rule fires no rescue at all). The resume equality claimed below is therefore that of the log, the
  archive, the rows and the digest, and of the journal only while no rule stays blocked across the resume.

**Not changed.** `RULES_VERSION` is 1 and the vitals descriptor has ten components in 3D as in 2D, so the logs of a 2D
and a 3D run share a rules version; a 3D rule set (a vertical-niche signal such as `clade_height_mi`, a persona that
reads `band_frac`) would be a new `RULES_VERSION` and is not part of PR 4. The extinction-risk test's `K` in 3D (above)
changes the rule tier's behaviour in 3D only and leaves 2D byte-identical, which is why `RULES_VERSION` stays 1 there too.
`alife stream` and `view` re-simulate
without the director (§27.11); `--follow` is the faithful view of a director run.

**Tests.** `tests/test_director_3d.py`:
- the validation accepts every mode in 3D;
- one scene through `clamp`, `validate_entry`, `apply` and a compiled step: 24 agents in a radius-2.5 column at six
  heights die (about 90%, at the lowest and the highest height), 8 outside it do not, the deaths are logged with cause
  `CATASTROPHE`, and `full_check` closes the ledger;
- a patch move: `(PP, 2)` kept, the other patches untouched, a `z` or `dz` rejected;
- the radius bounds (2D unchanged; `tiny3d`, widths 48 to 1024, the 12% share), their effect on the clamp, on replay
  validation, on the schema and on the sampler;
- the header (`dims` in 3D only, the 2D key set pinned, no header for an unchanged policy) and the refusal of a log of
  the other dimensions (replay, yoke);
- the prompts (sha256 of the 2D system messages of the three personas and of the user message, the 3D wording, the
  per-action effect texts);
- a scripted live run of 400 steps on `tiny3d` that takes all seven actions (a revert across a checkpoint, a 2D-sized
  radius clamped to 25, a patch move, an immigration pulse admitted in the volume) with its replay bitwise (fresh
  directory, in place from a checkpoint, a live resume with no second header), the run being a valid 3D run with the
  ledger exact;
- the `random` persona yoked to that log, and its replay;
- a real-rules Gardener run (40 founders, PR 4's value again: `n/K_floor` is 0.098 at the first boundary, so the
  extinction rescue fires, and the flag clears at step 60; PR 6 started 150 because the occupancy ratio left 40 flagged
  for the whole run, and under the floor-layer rule 150 founders fire no rescue) with the ledger exact, a final
  population above the founders and the rescue's immigrants together, its logged `k_est` equal to the
  occupancy-weighted `carrying_capacity` of the state before the director acted (against a host reference of the
  weights, and between the floor layer's value and all the layers'), its logged `k_floor` and `n_over_k_floor` against
  `floor_carrying_capacity`, the rescue's reason naming `n/K_floor` and the logged vitals' key order, a resume equal to the
  uninterrupted run (log, journal, archive, digest; the journal as qualified above), and its replay;
- `Director._vitals` on built scenes: the same patches give a floor-only `K_est` to agents resting on the floor and
  more than twice that to agents spread over the six layers, and `n/K_est` follows (PR 6), while `k_floor` is the same
  for both and equals an independent computation of the floor layer's `K`;
- the extinction flag against the two `K`s on built scenes: 200 agents spread over the six layers (`n/K_est` 0.14,
  `n/K_floor` 0.49) are not flagged, and the flag is the same for a spread and a benthic population of 60, 80, 85, 200
  and 250 agents, set at 60 and 80 and clear above 0.2·409 = 82
  (`test_a_spread_population_is_judged_against_the_floor_layer_not_the_occupancy_weighted_k`);
- the same Gardener run with each boundary's vitals and the floor-layer `K` of the same state: the rescue sized by
  `K_est` 3.2× the floor layer's (131 immigrants, not 41), the flag clear from step 60 where `n/K_est` (0.14 to 0.15)
  stays under the threshold (`test_a_3d_rescue_raises_its_own_k_est_but_no_longer_keeps_the_flag_set`);
- the prompt: both ratios on the viability line in 3D, each with its number and the clause saying which one the flag
  reads, and no such clause in 2D (`test_the_3d_prompt_says_which_capacity_the_extinction_flag_reads`; the 2D sha256 pins
  are unchanged);
- a strict `xfail`, 2D and 3D: a blocked note is journalled once across a resume
  (`test_a_blocked_note_is_journalled_once_across_a_resume`, the defect above).

Also: `tests/test_director_vitals.py` (the flag reads `n/K_floor` when the vitals carry `k_floor` and `n/K_est` without it,
at the threshold's edges and with the capacities either way round; the 2D vitals carry no `k_floor` and log exactly the
ten keys of build step 2 in their order; the 3D log adds `k_floor` and `n_over_k_floor` after `n_over_k`; `fall` is
independent of both), `tests/test_director_rules.py` (the rescue's reason names `n/K_floor` in 3D and `n/K_est` in 2D, the
count stays `0.1·k_est`, a blocked rescue is noted with the floor ratio) and `tests/test_metrics_3d.py`
(`floor_carrying_capacity`: the floor layer's integer regeneration over `c_ref` for any population, against an
independent float64 reference, equal to `carrying_capacity` in the benthic limit and for nobody alive, live under
`step` and `env`, and equal to the one `K_est` in 2D).

**Tests edited.** `test_dims_config_pr2.py::BAD_PR2` lost its `director.enabled` case: the rule it pinned is gone. PR 6's edits for the occupancy rule are in its Review-log row (I42); the converses of its two limit-pinning tests and the Gardener's return to 40 founders are in the row's extension.

### 27.11 The CLI in 3D: reporting, refusals and `alife stream` (PR 4)

**Reporting.** Every addition is 3D only; the 2D output, JSON key sets and `bench` environment are unchanged.
- **Config summary** (`cli._summary`, used by `inspect` and the run announcement): `dims`, `world_height` and
  `field_grid_z` (the layer count `FZ`) after `field_grid`. The announcement reads `L=128 height=24`.
- **`inspect`**: `config` has the three keys above; `state` has the §27.9 snapshot keys (heavy); the top level has
  `grounded_frac`, the share of the alive agents grounded in the checkpoint state (`physics.grounded`; `None` with
  nobody alive; the rows' `grounded_frac` is an interval mean). The text output adds one line, `height (world 24): mean …,
  quartiles …, grounded …, surface …, bands […]; speed …, |pitch| …, buoyancy cap median …`. `--agent UID` lists the
  (x, y, z) `pos`, `pitch` and `buoy_cap` among the fields and adds `height` (z) and `grounded` (`z ≤ Dims.z_ground`).
- **`bench`**: the environment `config` has `dims`, `world_height` and `field_grid_z`, and the text report prints
  `height=…` after `L=…` (`H=` stays the hidden size).

`tests/test_cli_report_3d.py`: `inspect --json` on a real 30-step `tiny3d` run reports `dims`, `world_height`,
`field_grid_z`, the height keys against the checkpoint state and the grounded share; the text line; `--agent`; the rows
of the run (interval and snapshot keys, the first row's occupancy-weighted `carrying_capacity_est`); the summary, the
announcement and `bench`'s environment and report in 3D and without the keys in 2D.

**Refusals.** `render`, `view` (and `render.render_frame`, `viewer.ViewerSession`) and `analysis.signals` keep their
`require_2d` refusal (`ConfigError`, exit 2, before the state is loaded): the pygame viewer and the MLX renderer draw a
2D plane, and the signal decoder mirrors S0–S8 of the 2D step; porting it to the 3D inputs and outputs is out of scope
for PR 4. The message names `alife stream`. A 3D run is viewed through the stream below, by the AlifeControl 3D viewer
(§27.12).

**`alife stream`** writes the world as a binary frame stream on stdout, for an external viewer (docs/3D.md §8). It
serves 2D and 3D worlds alike (a 2D stream carries z = 0 and pitch = 0, so the viewer can show it), is an observer
that never writes into the run directory, and shares the renderer's palette (§20.1) so a frame has the colours of a
`render` image.

```text
alife stream  [RUN|FILE | --preset P [--set section.key=value ...]] [--checkpoint STEP|latest]
              [--every K] [--steps N] [--follow [--poll SECONDS]]
              [--color tag|energy|clade] [--fields] [--field-z K] [--no-control]
```

`--field-z K` implies `--fields`. The `--preset` of `stream` has no default (without a source the command exits 2); the
other commands' `--preset` defaults to the engine's default preset.

- **Sources** (exactly one): `RUN`, a run directory (the `--checkpoint` step, default the latest); `FILE`, a standalone
  state file (a viewer snapshot; its embedded config); `--preset P` with `--set` (founders from `init_state`; a
  preset is required, there is no default). `RUN`/`FILE` together with `--preset`/`--set`, `--checkpoint` with a
  preset or a state file, or neither source, is a config error (exit 2). A run with no checkpoint is exit 4.
- **Simulating stream** (default). It re-simulates from the state like `view`: one compiled step and one blocking
  `mx.eval` per step, and the loop holds the only reference to the state (the donation rule, §15.4). It writes the
  header, a frame of the starting state, and then a frame after every `K`-th step (`--every K`, default 1), counted
  from the starting state: a bound of `--steps N` steps (default: unbounded) writes `N // K + 1` frames, the frame of
  step `k·K` being stamped with `start + k·K`. It ends after `N` steps, on `quit` or stdin EOF, or when the reader
  closes. **It re-simulates without the director**, as `view` and `render` do (§26.8): it keeps the checkpoint's
  environment levers but applies none of `director.jsonl`, so after an intervention it diverges from a director run.
  Nothing it does changes the trajectory (a test compares its final state digest with bare stepping, with and without
  pauses, colour changes and field slices).
- **`--follow`** (`RUN` only; `--steps`, `--every` and `--checkpoint` are refused) steps nothing. It streams the run's
  latest checkpoint and, polling the directory every `--poll` seconds (default 1), a frame of the newest checkpoint
  whenever it is newer than the last one streamed: checkpoints written between two polls are skipped (the stream
  samples a fast run at the poll rate), and a file identified by name, size and mtime is streamed once (checkpoints are
  written atomically). It is the **faithful view of a director run**: the checkpoint holds the environment the
  director applied. `pause` holds the stream on its current checkpoint, `step` fetches the newest one once, `color`
  and `fields` redraw the current checkpoint, `every` is ignored (with a note on stderr), `run` resumes. A checkpoint
  or `LATEST` that disappears between the listing and the read (the runner's pruning, or a resume's truncation) is
  skipped and the next poll carries on: the loader reports a missing file as `CheckpointError`, which the stream
  treats as a vanished file when the path no longer exists. A checkpoint that exists but cannot be loaded (corrupt, or
  of another config) is an error, exit 4 with a message on stderr, even after frames have been written. It only
  reads: the run directory may be read-only, and a test freezes it.
- **Processes and descriptors.** The protocol goes to a private duplicate of fd 1 (`stream.protocol_output`); while the
  stream runs, `sys.stdout` and fd 1 itself are pointed at stderr, so a stray `print`, a C library's `printf` and a
  child process cannot corrupt the framing. What the original `sys.stdout`, `sys.__stdout__` and the C library's
  stdout buffer still hold when the stream ends is flushed **before** fd 1 is restored (a pipe's C stdio buffer is
  only written at `exit`, which would put it after the last frame), so it lands on stderr. Diagnostics go to stderr.
  Control lines are read from stdin by a thread using `os.read` on the raw descriptor (a daemon thread blocked in a
  buffered reader can abort interpreter shutdown). `--no-control` leaves stdin alone (no control, and its EOF is not
  a quit). A closed reader (`BrokenPipeError`, `ConnectionResetError`) ends the stream with exit 0, no message and no
  traceback. A stream that writes nothing cannot learn of it from a failed write (a paused stream, a followed run
  that has stopped, a running one whose `--every` skips the frame), so it asks the descriptor (`poll` for `POLLHUP` or
  `POLLERR` on the write side, a no-op for a file or a device): once per `--poll` seconds when following, once a
  second (`stream.IDLE_WAIT_S`) when paused, and after every step that writes no frame. A non-blocking stdout that
  is full (a raw write returning `None`, EAGAIN) is a slow reader, not a closed one: the writer waits until the
  descriptor drains and loses no byte.
- **Exit codes**: 0 after `--steps`, `quit`, stdin EOF or a closed reader; 2 for a config error (nothing written to
  stdout); 4 for a missing or incompatible checkpoint (likewise, except for an unreadable checkpoint of a followed
  run, above); 1 unexpected; 130 SIGINT (`runner.ExitCode`). A decoder treats EOF inside a record as truncation (the
  process died).
- **Help.** The `stream` help says that it re-simulates without the director and that `--follow` is the faithful
  view of a director run (a test pins it, as for `view`).

**Protocol (version 1; normative, the Swift decoder of §27.12 is written from this text).** All integers are little endian;
floats are IEEE 754 (`f32` binary32, `f16` binary16). A stream is a header and then records until EOF.

| offset | size | field |
|---|---|---|
| 0 | 8 | magic `41 4C 53 54 52 4D 00 01`: `ALSTRM`, a zero byte, and the **major protocol version** (1) |
| 8 | 4 | `u32 json_len`: the byte length of the JSON text, a multiple of 4 |
| 12 | `json_len` | the header object: UTF-8 (ASCII) JSON, followed by 0 to 3 spaces of padding (inside `json_len`) |
| `12 + json_len` | | records |

*Header object* (every key is always present, in this order; a decoder ignores keys it does not know):

| key | type | meaning |
|---|---|---|
| `protocol` | int | 1, as the last magic byte |
| `dims` | int | 2 or 3 |
| `L` | float | `world.size`: x and y are periodic on `[0, L)` |
| `height` | float | `world.height` (z is bounded on `[0, height]`; an agent's centre lies in `[radius, height − radius]`); 0.0 when `dims` is 2 |
| `N` | int | rows in every per-agent section of a frame: the state's agent capacity, `population.n_max` |
| `B` | int | bond slots per agent (4) |
| `radius` | float | `physics.radius`, the agent radius |
| `field_cell` | float | the field cell size; `FG = L / field_cell` cells per side |
| `FG`, `FZ` | int | field cells per side, and field layers (`FZ = height / field_cell`; 1 when `dims` is 2) |
| `NCH` | int | chemical channels (3) |
| `color_modes` | list of str | `["tag", "energy", "clade"]`: the index of a mode is the frame's `color_index` |
| `color`, `fields`, `field_z` | str, bool, int | the view of the first frame: colour mode, field section on or off, field layer |
| `step` | int | the step of the first frame |
| `run` | str or null | the `RUN` or `FILE` argument as given; null for a preset |
| `config_hash` | str | `config_hash(cfg)` of the streamed config (§17) |
| `mode` | str | `"simulate"` or `"follow"` |
| `every` | int | steps per frame (`--every`); 0 in follow mode |
| `source` | str | `"run"`, `"file"` or `"preset"` |

*Records.* Each record is `tag` (4 bytes), `u32 len` and `len` payload bytes; the next record follows at once. Every
record has a multiple of 4 bytes, so each starts 4-byte aligned. A decoder skips a record whose tag it does not know
by its `len`. The only tag of version 1 is `FRM1`, a frame, with this payload (offsets from the start of the payload):

| offset | size | field |
|---|---|---|
| 0 | 8 | `u64 step`: the step of the streamed state |
| 8 | 4 | `u32 n_alive`: the number of rows with `alive = 1` |
| 12 | 4 | `u32 flags` (below) |
| 16 | 4 | `u32 n_bonds` |
| 20 | 4 | `u32 color_index`: the colour mode of this frame's `rgb`, an index into `color_modes` |
| 24 | 8 | reserved, zero |
| 32 | `4·ceil(N/4)` | **alive**: `N × u8`, 1 for an alive row, then zero padding to a multiple of 4 |
| | `12·N` | **pos**: `N × (f32 x, f32 y, f32 z)`. Always 3 floats; z is 0 in a 2D stream |
| | `4·N` | **orient**: `N × (f16 yaw, f16 pitch)` in radians. Yaw is `heading`, in `[−π, π)`; pitch is `State3.pitch`, in `[−80°, 80°]`, 0 in a 2D stream. The forward axis is `(cos φ·cos θ, cos φ·sin θ, sin φ)` for yaw θ and pitch φ (§27.2) |
| | `4·N` | **rgb**: `N × u32` `0x00RRGGBB` (R in bits 16–23; the top byte is 0) |
| | `8·n_bonds` | **bonds**: `n_bonds × (u32 i, u32 j)` with `i < j` and both rows alive, each bond once, in ascending `i` and then slot order |
| | `4·FG²` | **field** (only when flags bit 0 is set): `FG² × u32` `0x00RRGGBB`, row-major `[cy][cx]`: the cell `(cx, cy)` covers `[cx·field_cell, (cx+1)·field_cell)` in x and y; in 3D it is the lattice layer named by `flags >> 16` |

`flags`: bit 0 = a field section follows; bit 1 = the state's `diag` has an error bit set (an invariant violation, as the
pygame HUD shows); bits 2 to 15 are reserved zero; bits 16 to 31 are the field layer (0 without a field section). The
payload length is exactly `32 + 4·ceil(N/4) + 20·N + 8·n_bonds` plus `4·FG²` with a field. Section padding is zero
bytes. Row `r` of the alive, pos, orient and rgb sections is agent slot `r` of the state; the values of a row with
`alive = 0` are unspecified and a decoder ignores them (the writer copies the state's dead-row fill and colours it
like an alive row). Colours are those of `render.agent_colors` (agents hurt in the last step, `last_loss > 0`, red
`0xFF0000` in every mode; the `tag`, `energy` and `clade` palettes of §20.1) and `render.field_colors` (the §20.1
background of resource and chemicals for one layer); there is no `kin` mode.

*Versioning.* The last magic byte and `protocol` are the major version: a change to the header layout or to the layout of
an existing record bumps it, and a decoder refuses a major version it does not know. Compatible additions keep the
version: new header keys, new `flags` bits (a decoder ignores bits it does not know), a nonzero use of the reserved
bytes by a later version (a version-1 writer writes zeros), and new record tags (skipped by length).

*Control grammar.* One command per line on stdin (LF; surrounding whitespace is ignored); a blank line or a line whose
first token starts with `#` is ignored; tokens are lower case and separated by whitespace; `N` and `K` are decimal
integers.

```text
command = "pause" | "run" | "step" [N] | "every" K | "color" MODE | "fields" ( "on" | "off" ) | "fields" "z" K | "quit"
```

Lines are applied one at a time, between steps, in the order received, however the pipe chunks them: a line takes
effect, and writes the frame it asks for, before the next line is looked at. `pause` stops stepping and frames and
cancels the pending `step` batches. `run` resumes (and cancels them too). `step [N]` (N ≥ 1, default 1) pauses, runs N
steps and writes exactly one frame after the last, whatever `every` is; a `step` received while another batch is
running or queued is queued behind it and has a frame of its own, so `step 3` followed by `step 2` writes a frame
after step 3 and another after step 5. `every K` (K ≥ 1) sets the frame period. `color MODE` (`tag`, `energy` or
`clade`) sets the colour of later frames and writes a frame of the current state now. `fields on|off` includes or omits
the field section and writes a frame now. `fields z K` (0 ≤ K < FZ) chooses the layer and writes a frame now if fields
are on. `quit` ends the stream, as does EOF of stdin; the lines after `quit` in the same read are dropped, and the
frames that earlier lines asked for are already written (so `printf 'color clade\n' | alife stream ...` ends with the
clade frame). A line that is not a valid command, or longer than 4096 bytes (its LF not counted: a line of exactly 4096
bytes is accepted), is ignored with one line on stderr per dropped line; nothing else changes. A paused stream with
nothing to do waits on stdin without using CPU (it wakes once a second to look at the output descriptor). Commands
change only what is written and when: the trajectory is the same.

*Fixtures for the decoder.* `tests/data/stream_tiny_2d.bin` and `stream_tiny_3d.bin` are complete streams of a built
(not stepped) scene on `tiny` / `tiny3d` with `micro` sizes (N = 64, L = 48, height 24): three frames (steps 100, 104,
108) with a bond across the periodic seam, a hurt (red) agent, dead rows between alive ones, the three colour modes,
a field slice (3D: layers 0 and 5) and the invariant flag. The first frame has three bonds, row 2 holding two of them
in slot order (partner 10, then 3, so the stream lists (2,10) before (2,3)); the second frame has a fourth bond. The
JSON next to each (`.json`) holds the expected header and, per frame, the head fields, the alive rows, the bonds, the
first agents (position, yaw, pitch, colour), the first field cells and the SHA-256 of every section, produced by an
independent decoder (`tests/stream_decode.py`, written from this section). `tests/stream_fixtures.py` regenerates them
on the CPU device (`uv run python tests/stream_fixtures.py --write`); the test rebuilds both and compares the bytes
(skipped on a GPU default device, where the float32 colour arithmetic may round differently; the committed files are
decoded and checked against their sidecars on every device).

**Tests.** `tests/test_stream.py` (every protocol claim is checked with the independent decoder):
- the header (bytes, alignment, key order and values in 2D and 3D) and frame count `steps // every + 1` with absolute
  step stamps; unknown record tags skipped;
- the frames of a stepped scene equal the stepped state: alive mask, positions, yaw, pitch, bonds (a death along the
  way removes its bonds) and colours; a 2D stream has z = 0 and pitch = 0 on every row; the bond order (row 2 of the
  scene holds partners 10 and 3 in that slot order, which row-major, slot-major and partner-sorted orders all list
  differently); the flag word (bit 1 for an error bit of `diag` and not for the routine warning bits, bits 2 to 15 zero,
  bit 0 and the layer bits only with a field section, whatever layer the view has selected);
- colour modes, the field slice against the renderer's background (2D, one pixel per cell) and against a float32
  reference per layer (3D), the colour helpers' argument checks;
- purity: the final state digest equals bare stepping, with every frame cadence and a control script;
- the control grammar (a parser table; every command in a fixed schedule with the expected frames; each redraw command
  writing its own frame, the same frames whether the lines arrive in one read or one per read; a redraw written before
  `quit` or EOF in the same read, through a real pipe too; queued `step` batches with a frame each; invalid lines; EOF
  as quit; `--steps` winning over a pending `step`; a paused stream waiting on the channel with a bounded timeout; the
  stdin reader with partial lines, a line of exactly 4096 bytes accepted, overlong lines in one write or in pieces
  dropped with one note on stderr each and the line after them intact, and EOF);
- a closed reader (`BrokenPipeError` and `ConnectionResetError`, at each write) is a quiet stop; a full non-blocking
  pipe makes the writer wait and loses no byte; `reader_closed` on a pipe, a file and a device; a paused stream, a
  running one whose `--every` writes no frame, and a followed one with nothing to write notice a closed reader; `protocol_output` sends prints and raw fd-1 writes to
  stderr and restores both; a step that prints cannot corrupt the stream; a subprocess whose step leaves output in the
  C library's stdout buffer and in `sys.__stdout__` still ends with a stdout that decodes strictly;
- the sources through `cli.main` (RUN at LATEST and at `--checkpoint`, FILE, preset; `--field-z` alone turning the field
  section on; the argument errors, exit 2 and 4; the `--follow`, `--poll` and source arguments reaching `serve`; the
  help saying `--preset` has no default); a run directory and a followed one that are read-only, unchanged;
- `--follow`: the latest first, new checkpoints, pause, step, redraws, `every` ignored, a file that vanishes between
  the stat and the read (deleted for real, so the real loader raises its `CheckpointError`), `LATEST` vanishing, a
  checkpoint that exists but cannot be loaded reported, nothing stepped;
- four subprocesses on the CPU device with real pipes: a reader that closes early exits 0 with an empty stderr, stdin
  commands and EOF with a stdout that decodes strictly, `--follow` seeing a checkpoint written by another process, and
  `--follow --no-control` exiting once the reader goes;
- the committed fixtures: decoded against their sidecars on every device and rebuilt byte for byte on the CPU device;
  `stream_fixtures.build` asserts that no colour computed from an inexact float32 expression (the energy ramp, the
  field greens and tints) lies within 1e-3 of a rounding boundary, so the byte comparison does not depend on how a
  platform rounds the last bit.

**Tests edited.** `tests/test_lint.py` scans `stream` for subscript assignment (`SETITEM_MODULES`).

### 27.12 The 3D viewer in AlifeControl (PR 5)

PR 5 is the Swift side of the 3D world: the stream decoder and the camera (pure logic, with tests), the levers of the
3D keys, and a second window of AlifeControl that draws an `alife stream` with Metal. It changes no engine code, no
config key, no hash and no digest; the 2D GUI is unchanged (a table of its command lines, captured from the sources of
the previous commit before the refactor, pins them byte for byte). The protocol is §27.11; the rationale is
docs/3D.md §8.

**Package** (`gui/Package.swift`, Swift 6 language mode, macOS 14, no resources, no unsafe flags):

| target | content |
|---|---|
| `AlifeCore` (library; `package` access; no AppKit, no Metal) | `Levers` and `World3DLevers`, `DirectorSettings`, `PresetFile` (the scalar reader of `src/alife/presets/NAME.toml`, which `Cadence` stands on), `MetricsTailer`, `FrameDecoder` (the protocol), `StreamCommand` (the control lines), `Camera` |
| `AlifeControl` (executable) | the app and its run controller, and the viewer: `StreamReader`, `StreamController`, `Shaders`, `Renderer` (with `DrawGate`), `ViewerMTKView`, `ViewerModel`, `Viewer3DView`, `ViewerSupport` (the keys, the speed ladder, the pointer-to-camera mapping, the stream's arguments, the snapshot path), `WindowCloseObserver` (with `WindowGroups`) |
| `AlifeCoreTests`, `AlifeControlTests` | Swift Testing; the second imports the executable with `@testable` |

The build and the checks are `swift build -Xswiftc -warnings-as-errors` and `swift test` in `gui/`. Nothing in the
package uses `@unchecked Sendable`, `nonisolated(unsafe)` or `assumeIsolated`.

**Decoder** (`FrameDecoder`). It is written from §27.11 alone and is checked against the same two fixtures and JSON
sidecars as the Python decoder. It is incremental and owns its buffer, so the events are the same for any chunking of
the input; it refuses a bad magic (as soon as one byte differs), a major version other than 1, a header that cannot
describe a stream (a missing key, `run` included, which is a string or null; a value out of its range; or one that
breaks a relation of §27.11: `FG·field_cell = L`, `FZ·field_cell = height` and `height > 0` in 3D, `height = 0` and `FZ
= 1` in 2D, `color` among `color_modes`, `field_z` below `FZ`, `mode` and `source` among the values of version 1; the
viewer copies the initial view into its state, so it must be one it can show), a frame whose length disagrees with its
content (before the payload arrives), a frame that contradicts the header (`n_alive`, an alive byte above 1, a colour
index, a bond that is not `i < j < N`, a field layer outside `FZ`) and an end of input inside a record; a record of an
unknown tag is skipped by its length. An error is sticky and follows the events that precede it: `feed` returns the
events before an error and throws it from the next call, and `failure` shows it at once, for a caller that will not get
another chunk soon. Frames keep the orientation as the binary16 bit patterns that were streamed (the renderer uploads a
`half2` per row), and `Half.float` converts (`Float16` exists only on arm64). The decoder bounds `N` (2²²), `FG` and
`FZ` (1 024), `B` (64) and `n_bonds` (`N·B`) so that a corrupt header cannot ask for absurd memory.

**The stream process** (`StreamController`, main-actor isolated, one at a time). The viewer starts
`.venv/bin/alife stream runs/NAME ...` (no shell) with three pipes:
- **stdout** carries the protocol and never goes through the line splitter of the run log (which cuts on 0x0A and
  0x0D, bytes that occur inside every frame; a test shows that the fixtures contain them). A dedicated `Thread`
  (`StreamReader`) calls `read(2)` in a loop, so that the pipe is drained as fast as the engine writes whatever the main
  actor does (a full 64 KiB pipe would stall the simulation), and feeds a `FrameDecoder`. `FileHandle.read(upToCount:)`
  is not used: on a pipe it can wait for the whole count, which for a paused stream is forever (a stand-in process test
  found it). The newest header and frame are left in a mailbox (`OSAllocatedUnfairLock<StreamDelivery>`; a frame that
  is replaced before it is taken counts as dropped) and the main actor is woken through an `AsyncStream<Void>` with
  `bufferingNewest(1)`, so wake-ups coalesce. The thread wakes it only after a chunk that completed a header or a frame
  or ended the stream (a pipe delivers a frame of a few megabytes in chunks of at most 64 KiB, and a wake-up per chunk
  would be a redraw of the old frame per chunk), and a decoder error is published by the chunk that meets it, not by
  the next (a paused stream, or a followed run that has stopped, sends no next one). The thread finishes the mailbox
  with `complete`, or with the decoder's error, at end of file, and keeps draining (and discarding) after an error.
- **stderr** goes line by line to the run log as `[3d] ...` (red), with the command line (`$ ...`) before it; the last
  non-blank line is what a failure reports.
- **stdin** carries the control lines of §27.11 (`StreamCommand`: pause, run, step [N], every K, color MODE, fields
  on|off, fields z K, quit); the write is the throwing `FileHandle.write(contentsOf:)`. The descriptor has
  `F_SETNOSIGPIPE`, and the app also ignores SIGPIPE (`AppDelegate`), so a write to a process that has gone is an error,
  not a signal.

*Order of events.* The session finishes only when the process has exited **and** the reader and the stderr drain have
reached end of file, and the exit status is looked at after the last frame has been applied, so the last frame and the
status cannot race. A decoder error other than a truncation stops the process (SIGTERM; nothing more can be read); a
truncation is not reported by itself, because the exit status and stderr say why the process died. The phase is `idle`
(nothing open, or closed on purpose), `starting` (until the header), `live`, `ended` (exit 0, an input that ended
between two records) or `failed` (the message holds the decoder's error, or the exit status and the last stderr line).

*Stopping* is exactly once per process: a `quit` line, then stdin is closed (its EOF is a quit too), then one SIGTERM
if the process is still there after `terminationGrace` (2 s). It happens when the viewer window closes (a
`WindowCloseObserver` on the window's `willClose`, and `onDisappear`), when another stream replaces it (the old
session's end is then ignored), and, without the grace, when the app terminates (`RunController.shutdown`), for every
process that has not finished, the ones that a newer stream replaced included (their SIGTERM after the grace would die
with the app). A stream has nothing to save, which is why this is not the single SIGINT that stops a run.

*Closing the last main window* closes the viewer window with it, so the app quits as it did when it had one window
(`applicationShouldTerminateAfterLastWindowClosed`). The main scene is a `WindowGroup`, so there can be several main
windows (File > New Window): `WindowGroups` counts them, and the close of one that is not the last leaves the viewer
alone. The close is noticed through `willClose`, and the observer view is held strongly across the hop to the main
actor, so that SwiftUI taking the closing window's content down first cannot make the close go unnoticed.

**Sources.** The viewer window shows one run at a time, as a **follow** (`--follow`: the run's newest checkpoints, the
faithful view of a director run; a run writes its step-0 checkpoint as it starts (`runner.run`, before the first
step) and then one every `run.checkpoint_every` steps, so the first frame comes at once, and the stream exits 4, shown
in the window, only when it is opened in the moments before that checkpoint exists or on a run directory that has none) or
a **re-simulation** (`alife stream runs/NAME [--every K]`: the latest checkpoint
stepped again in a separate process, which shares the GPU with a running simulation and **ignores the director**,
§27.11; the window says so). "Open 3D viewer" in the main window follows the run that is in progress and re-simulates
any other; the source switch in the viewer window restarts the stream with the other one. It works for 2D runs too
(z = 0, the face-on camera). The view survives a restart of the same run: the arguments carry `--color`, `--field-z`
and, for a re-simulation, `--every`. Another run starts at field layer 0 (with the slice still on if it was): a run
with fewer layers, a 2D run having one, would refuse a layer past its last (`alife stream` exits 2), and the window
could not change it. The pygame viewer's button is disabled for a 3D run (`alife view` refuses it, exit 2), with a
note pointing at the 3D viewer; the run's dimensions come from its `config.json` (`world.dims`, 2 when absent).

*Controls and what they send.* Space pause or run; `.` step (a followed run fetches its newest checkpoint once instead,
and keeps being followed: the viewer does not show it as paused); `+` and `-` double or halve `every` within [1, 1024]
(a re-simulation only; this is the speed); `c` the next colour mode of the header; `f` the field section on or off and
`[` `]` its layer (clamped to `FZ − 1`). The viewer keeps what it asked for and takes the initial view from the header;
the colour and the field section of a frame show what is drawn.

**Rendering** (`Renderer`, `Shaders`; raw Metal through `MTKView`, no RealityKit or SceneKit).
- *Shaders* are one MSL string compiled with `makeLibrary(source:options:)` (language version 3.0) when the first
  stream is shown (an idle window, such as one that macOS reopens, creates no Metal device): the installed app is the
  release binary alone (no resource bundle) and the Metal Toolchain is a separate download. A compile error reaches the
  window as text. A test checks the entry points against the source and the
  balance of its brackets; only a device can compile it (see Tests).
- *Agents* are instanced impostors: four vertices per row of the frame (the capacity `N`), a dead row clipped away in
  the vertex shader, a billboard that faces the eye and is as large as the silhouette of the sphere about the agent that
  holds its body and its nose (radius `1.25 r`, seen from at least `1.5 r`), and a fragment shader that intersects the
  pixel's ray with the body (radius `r`) and with a smaller, paler sphere at `0.75 r` along the forward axis `(cos φ·cos
  θ, cos φ·sin θ, sin φ)` (the nose), shades the nearer by a headlight and writes the true depth. The miss distance is
  taken from the perpendicular component of the offset, so it keeps its precision far from the eye.
- *Bonds* are a line list of four vertices per bond, two half segments from the agents to the midpoint of the
  **minimum-image** offset (`rint` in x and y, not in z), so a bond across the periodic seam is two stubs towards the
  seam and not a line across the world; each half has the colour of its agent, paler. Lines are one pixel wide.
- *The frame* is the twelve edges of the box and, in 3D, a translucent floor (z = 0) and surface (z = `height`).
- *The field slice* is the frame's `FG²` colours as a nearest-filtered texture on the plane at the middle of its layer
  (`(k + ½)·field_cell`), 60 % opaque in 3D. In 2D it is the opaque background, drawn first so that it neither hides nor
  z-fights with the agents and bonds, which lie in its plane. Translucent planes are drawn last, farthest from the eye
  first, with depth test and no depth write.
- *Buffers*: the arrays of a frame (alive, pos, orient, rgb, bonds) are copied as they are into one of three shared
  `MTLBuffer` sets (and the field into one of three textures), a semaphore lets three draws be in flight, and a set
  remembers which frame it holds, so a redraw from another angle copies nothing once all three have the frame. A draw
  into the window never waits for the GPU on the main thread, and takes its set **before** its drawable: asking
  `CAMetalLayer` for a drawable blocks, for up to a second, when all three are in use, so a second semaphore lets two
  draws into the window be in flight (one fewer than the layer has drawables; the one on screen is not free), and a draw
  that finds no set or no place (a simulation is using the GPU) or no drawable is retried after 100 ms (a drawable that
  stays missing is retried 20 times, and then waits for the next cue). `DrawGate` holds the ring and the two
  semaphores, and advances the ring only when a draw has been committed: one that fails before that, in the upload or
  the encoder, has used no set and gives its permissions back, so the ring and the semaphore cannot drift apart until a
  draw writes a set that the GPU is still reading. A snapshot (no drawable) waits up to 10 s for a set.
- *On demand*: the view is paused with `enableSetNeedsDisplay`, so there is no display link: the GPU is idle until a
  frame arrives, the camera moves, a display option changes or the window is resized. SwiftUI updates the view for every
  change of the stream's or the model's state (the speed readout, the status line), and the view is redrawn only when
  its display options, header or frame changed.
- *Snapshot*: `s` (or the button) renders the current view offscreen at the drawable's size and writes
  `runs/NAME/media/view3d_<step>.png`, next to the 2D viewer's `view_<step>.png`, with ImageIO. The viewer writes nothing
  else into a run directory, and `alife stream` writes nothing at all.

**Camera and input** (`Camera` in `AlifeCore`, `ViewerMTKView`, `ViewerModel`). One pose (target, distance, yaw, pitch),
edited two ways: *orbit* turns it about the target, *fly* about the eye. A 3D world starts at the overview (outside the
south face, above the floor, x to the right), a 2D world face on from the −z side with x to the right and y down, as
the 2D PNGs (the plane at z = 0). The home pose fits the window and is fitted again when the window is resized, until
the user moves the camera. The clip planes follow the pose: the far plane is at the farthest point of the world from the
eye and the near plane at a quarter of the distance to the nearest (never nearer than 0.1), wherever the eye is, so
zooming out or flying away never clips the world, and the depth buffer's resolution is spent on the world (a depth step
at the home pose is a thousandth of a world unit in a 1 024-wide world, where the fixed planes 0.1 and `8 L` gave 0.75,
most of an agent's radius). Left drag orbits or looks (right, middle and Option drag pan); the wheel and the trackpad
zoom (in fly mode they move ahead); a pinch zooms; the arrow keys orbit and dolly, or fly (Shift: five times as far; Page
Up and Page Down for up and down); `o` toggles orbit and fly, `r` resets the view. `1` `2` `3` toggle the agents, the
bonds and the frame, `4` the field slice (`f`), `s` or `p` the snapshot.

**Levers** (`World3DLevers`). The 3D keys `world.height`, `physics.gravity`, `physics.buoyancy_max`,
`energy.buoyancy_cost`, `energy.buoyancy_cap_cost`, `fields.benthic_lambda`, `fields.photic_amp` and
`fields.photic_lambda` appear in a "3D world levers" box for a 3D preset or run. A new run diffs them against the
**preset** (`tiny3d` sets values the engine defaults, all 0, lack: against the defaults every key would be passed and
"set back to 0" could not be said), a fork against its source's `config.json`, as the 2D levers do; the 2D levers keep
the engine defaults, so a 2D command line does not change. `world.dims` is never an input (the preset or the source
fixes it: a 3D world needs a preset's sizes, diffusion and food profile together, and the engine rejects `dev` as 3D); a
fork also keeps its source's `world.height`; a resume has no levers. `physics.floor_eps`, `mutation.buoy_init` and
`mutation.buoy_sigma` stay CLI-only (the first is a numerical detail of the lattice; the other two are the start and the
noise of the cap, and the engine requires both to be 0 when `physics.buoyancy_max` is 0, so switching the actuator off
passes them as 0, and a nonzero cost with no actuator is a validation error that mirrors the engine's). The other
direction is refused: from a baseline whose actuator is off, `buoy_init` and `buoy_sigma` are 0, and switching
`physics.buoyancy_max` on there would be accepted by the engine and do nothing (every cap stays 0, so nothing lifts and
no cost is paid), so it is a validation error that says to set them on the command line; a baseline with an actuator, or
with a cap that can evolve, is unaffected. Making the two keys levers instead is open (HANDOVER). The status box shows
`height_mean`, `grounded_frac` and `surface_frac` of a 3D run. `tests/test_gui_sync.py` checks the Swift keys and preset
names against `Config`.

**Tests** (`swift test`; none needs a GPU or the engine, and `tests/data/stream_tiny_{2d,3d}.bin` are read from the
repository):
- `AlifeCoreTests`: the 2D command lines (69 configurations: 46 hand-written ones and 23 that choose a preset through
  its file or fork and resume from `config.json` files written by the engine, `tests/data/gui_config_*.json`, which
  `tests/test_gui_sync.py` checks against `load_config`), the 3D lever rules, `PresetFile` against the reader it
  replaced (equal on every file `tomllib` accepts; a key set twice is the exception, which the engine refuses), the
  decoder (the fixtures against their sidecars down to the SHA-256 of every section; chunking 1, 7, 4096 and whole;
  every refusal; truncation at every byte; unknown records), the control lines, the camera, `MetricsTailer`.
- `AlifeControlTests`: the reader and mailbox on a real pipe (fixtures cut in pieces of 7 bytes, a truncated stream,
  junk that must still be drained, coalescing and the drop count); the controller against a stand-in script that plays
  `alife stream` with real processes and pipes (the last frame before the end, the initial view, exit status and stderr,
  a signal, a corrupt stream stopped, a launch failure, commands in order with exactly one `quit`, the SIGTERM after the
  grace, a new stream replacing an old one whose end is then ignored and whose process the quit still ends, the source
  switch, the field layer kept across a restart and reset for another run, no speed and no pause for a followed run, a
  write to a closed pipe); the reader's wake-ups (none for a partial frame) and a decoder error published on an open
  pipe; the draw gate (the ring after a failed draw, the limit on draws into the window) and the window groups; the
  keys, the speed ladder, the pointer-to-camera mapping (including that a drag right moves the near side of the scene
  right), the planes and their order, the uniform layout (160 and 32 bytes) and the shader source (entry points,
  brackets), the PNG writer; and the GPU tests below.

*The GPU tests* (`GPURenderTests`) are opt in: `ALIFE_GPU_TESTS=1 swift test --filter GPURenderTests` (they are skipped
otherwise, and then nothing creates a Metal device). They compile the shaders, check that each layer draws alone and
that nothing draws with every layer off, that every alive agent of both fixtures appears at the pixel the camera
predicts, that every pixel around the agents and on a sparse grid elsewhere has the colour of a CPU port of the sphere
shader (`ReferenceSphere`; the colour channels, the headlight, the yaw and pitch, the nose on the side the agent faces
and only there; the port itself is checked on the CPU), that a bond across the seam is two stubs and not a line across
the world, that the 2D field has the orientation `[cy][cx]` and the 3D slice blends 60 % at its layer (layer 0 and layer
5 of 6), and that a snapshot is a PNG; images go to `$ALIFE_GPU_TESTS_OUT` or a directory under the temporary directory.

**Status and limits.** The decoder, the controller and the camera are tested; the shaders and the drawing are not run in
the PR's checks (the GPU was running evaluation arms): the first run of the GPU tests and a click-through (HANDOVER) are
the checks to make. Known limits: one-pixel bond lines, no multisampling (sphere edges alias), no picking of an agent,
no state snapshot (`s` of the 2D viewer writes one; here it is a PNG), and a re-simulation shares the GPU with a run.

### 27.13 Presets and tuning (PR 6)

PR 6 is the ecology of the 3D world: a preset at 64k scale, the tests that must pass before a long run, and the first
runs. This section is what needs no GPU: the occupancy-weighted `K_est` (§27.10), the `64k3d` preset, and the
tests (the 3D affordance run, the preset table, the `K_est` band of the initial state, the slow ecology test). The
values below are **provisional**: they come from the cloud runs summarised next and from arithmetic, and none of them
has been measured at 64k scale on the Mac (the list of what is, last).

**Cloud tuning evidence** (CPU runs of the cloud worker, 2026-10-13, reported in `results/cloud-3.md` and
`results/cloud-6.md` on the branch `claude/cloud-results`; `tiny3d` and `DEV3D` below are not the final presets):
- **Gravity sets the vertical ecology** (`tiny3d`, 10 000 steps, seed 0, a sweep of `physics.gravity`). At 0.01 about 40%
  of the agents are grounded, about 20% are at the surface and the clade-height information (`clade_height_mi`, caveat
  in §27.9) is about 0.70: clades occupy their own depths. At 0.03 (the `tiny3d` value) 76 to 83% are benthic and the
  information is 0.30 to 0.41. The recommendation was 0.01 to 0.015 (0.02 was not tested); the owner chose 0.015
  (2026-10-14), the top of that range.
- **The buoyancy costs barely matter** (a grid of the use cost 4 to 16 and the upkeep 2 to 8). The upkeep sets the energy
  share of the cap (2.5% of basal against 0.3% across the grid) and so the selection gradient on it, and the median cap
  stays at 0.50 in every cell. The `tiny3d` costs (16 and 8) are kept.
- **A long `tiny3d` run** (20 000 steps): the ledger exact and no hash overflow at C = 16. Seed 0 stays benthic (82%
  grounded, population 486). Seed 1 colonises the water column after about 10 000 steps (28% grounded, a surface band of
  23%, a body spanning 19 of 24 units) and reaches `n_cap` (992) at about 16 000. The floor-only `K_est`
  (about 450) then underestimates the population by half: the reason for the occupancy rule (§27.10). The cap spreads
  on seed 0 and drifts down on seed 1 (median 0.46).
- **Hash capacity and neighbours** (`DEV3D`: 16 384 rows, L 160, height 40, 2 000 steps; and the crowd tests of
  `tests/test_crowd_3d.py`). Under gravity a floor pile overflows C = 16 (peak occupancy 38, on 716 steps) and collapses
  onto a point; C = 64 holds, with a peak occupancy of 28 for the floor pile and 36 for a ball converging by thrust
  without gravity, and true contacts reach 10 against the K = 8 neighbours listed. In the ordinary `DEV3D` run no cell
  overflowed at C = 16 (peak occupancy 13); C = 48 is bit-identical to it and costs 2.1× the time and 1.7× the memory.
  **That `DEV3D` ran at the default `regen_max` of 320** (its floor-only `K_est` was 792; the preset's 1 700 feeds
  about 5× as many agents per floor cell, `C` against the food below), as did `tiny3d` (peak occupancy 13). **The
  slow `DEV3D` test of this PR, run on the cloud worker at the preset's `regen_max` and `C = 24` (cloud-7, 2 500 steps,
  Linux CPU), failed its overflow gate and nothing else:** one floor hash cell was over `C` continuously from step 2 151
  (at most one cell at any step, 299 steps, 11.96% against the gate of 1%), the window peaks of the occupancy rising
  12, 9, 10, 11, 14, 16, 18, 18, 28, 31 over ten windows of 250 steps (one floor pile, still growing at step 2 500; the
  grounded share rose from 0.04 to 0.30 and 47% of the agents were in the bottom 4 units). That is the reason for
  `C = 48` below. **K = 12 cannot pass `validate`**: the stability number `2(B·k_spring + K·k_repel)/drag ≤ 1.8` needs
  `k_repel ≤ 0.058`, and non-penetration (`thrust_max + max(g, buoyancy_max − g) < 2 r k_repel`, §27.3) needs more than
  0.065 at `tiny3d`'s gravity (0.0625 at 0.015). Making room takes a larger `drag` (about 1.3) or a smaller
  `thrust_max`; at equal physics K = 12 and K = 8 barely differ.
- **Seeding** (the cloud `DEV3D`: 4 096 founders, 5× its floor-only `K_est` of 792, 2.5× its `K_est` at 70% of the
  agents in the floor layer and 1.4× its `K_est` at step 0, the last two scaled from this preset's 16-seed means by
  320/1 700): the population booms to 5 900 at step 100, falls to 4 100 at step 200 and to about 1 000 at step 2 000
  (still declining). It is a case of over-seeding by 2.5× or more; seeding at 1× was not run. On `tiny3d` the
  first generation multiplies the founders by 1.4 to 1.7 within about 100 steps (256 to 441 by step 50, a peak of 442
  at step 64; 415 at step 100, 292 at step 200), whatever the food, because the founders start with a surplus
  (`e_init` 50 000 against `repro.min_energy` 40 000).

**The `64k3d` preset** (`presets/64k3d.toml`): `64k`'s defaults, a 3D world and the changes below. Valid with the `fast`
overlay too (`K = 6 ≤ 27·12`).

| key | value | why |
|---|---|---|
| `world.size`, `dims`, `height` | 320, 3, 80 | `FG = 80`, `FZ = 20` (128 000 field cells, 2× the 2D 64k lattice), hash grid 53 × 53 × 13 (cells 6.04 × 6.04 × 6.15), `2·height ≤ L`. The volume, 8.19 M, gives the 2D neighbour density (about 3.6 neighbours within the sensing radius) at 33 000 agents spread uniformly |
| `physics.gravity` | 0.015 | the owner's choice between the sweeps above. Sinking is 0.015 per step, climbing at full pitch 0.0835 (`0.1·sin 80° − g`). Hovering costs about 2 q/step either way: thrust at `a = g/(0.1·sin 80°) = 0.152` is `floor(100·a²) = 2`, buoyancy at 0.375 of its maximum is `floor(16·0.375²) = 2` (9 q/step on `tiny3d`), so the cap range is worth using; a founder's cap (0.5) lifts 0.02, more than gravity |
| `physics.floor_eps`, `buoyancy_max` | 0.1, 0.04 | as `tiny3d`. `thrust_max + max(g, bmax − g) = 0.125 < 2·r·k_repel = 0.16` |
| `sensing.k_neighbors`, `cell_capacity` | 8, **48 (provisional)** | K = 8 for the validation conflict above. **C = 48 is provisional, pending `alife bench --preset 64k3d --stages` on the Mac; lowering `regen_max` is the alternative if it proves too costly.** It was 24 (between the 16 that never overflowed in an ordinary run at `regen_max` 320 and the 28 that a floor pile reached in a crowd) until the cloud run of the slow `DEV3D` test overflowed it (a floor pile of 31, still growing, over `C` on 11.96% of the steps against a gate of 1%). 48 holds that pile with a margin of 1.5× and the recommendation of the crowd evidence was 48 to 64 (a pile of 28 at C = 64, a converging ball of 36). `Q = 27·C = 1 296` candidates per agent, 9× the 144 of 2D (`tiny3d`: 432; 648 at C = 24). Cost (below): on the cloud CPU about 2.1× the step time of C = 16 and 1.6× that of C = 24 (`DEV3D`), and 340 MB per `(N, Q)` array. The test of the new value has not been run (the slow `DEV3D` test is for the cloud worker); the pile may still be growing at 2 500 steps, so a longer run is the check. To be confirmed on the Mac by `bench --stages` and by `cell_overflow` and `max_occupancy` at the benthic steady state of a real run |
| `energy.brain_cost_per_l1` | 0.0905 | the 3D brain is wider (4 999 weights against 4 387) and its expected L1 norm is 10.5% larger (each weight is smaller: `init_scale/√rows`), so `0.1/1.105` keeps a founder's `brain_cost` at the 2D value: `founder_brain_cost` is 37.63 q/step against 37.64. A 3D founder otherwise pays 4 q/step more for its brain, on top of the cap upkeep |
| `energy.buoyancy_cost`, `buoyancy_cap_cost` | 16, 8 | the cloud grid found no effect of either; 4 q/step of upkeep at a founder's cap, about 3% of its basal and brain cost (137.6) |
| `mutation.buoy_init`, `buoy_sigma` | 0.5, 0.02 | as `tiny3d` |
| `fields.n_patches`, `patch_sigma` | 8, 24 | the patches cover 28% of the floor (`8·2πσ²/L²`), 2D 64k's 33% (24 patches of σ 48 in 1 024); the mean floor fertility is 0.31, 2D's 0.33 to 0.36 |
| `fields.chem_diffusion` | 0.16, 0.10, 0.16 | `≤ 1/6` in 3D (§27.6), as `tiny3d` |
| `fields.benthic_lambda`, `photic_amp`, `photic_lambda` | 8, 0.5, 8 | as `tiny3d`: `V = 1, 0.61, 0.37, 0.22, …` down from the floor and 0.5, 0.30, 0.18, … down from the surface (every layer is `min(1, benthic + photic)`); all layers together hold 3.8 floor layers' regeneration |
| `fields.regen_max` | **1 700** | `K_est` of about 33 000, below |
| `population.n_max`, `max_births_per_step`, `n_init` | 65 536, 1 024, 16 384 | below; `n_init` is 0.49 of the 70% `K_est`, not "near K" (owner sign-off needed) |
| `run.*` | the `64k` defaults | metrics every 100 steps, heavy every 1 000, checkpoint every 10 000; the director's `rule_every` of 1 000 validates as in 2D |

**`regen_max` and `K_est`.** The target is a `K_est` of about 33 000, the agent count at which the volume has the 2D
neighbour density. `K_est` is a function of the agents' layers (§27.10), so the sizing states its assumption. With the
layer sums `R_k = Σ regen_rate(…)[k]` (at `regen_max` 1 700, 16 seeds: `R_0` 3.2 to 3.6 M, `R_1/R_0 = 0.606`, `Σ_k R_k =
3.8·R_0`) and `c_ref` 191.63 (basal 100, founder brain 37.6, margin 50, cap upkeep 4):

| agents in the floor layer, the rest spread evenly over the other 19 | `K_est` (mean of 16 seeds; range) |
|---|---|
| all (the floor layer only) | 17 800 (16 800 to 18 700) |
| 90% | 23 100 |
| 80% | 28 300 |
| **70% (the assumption)** | **33 600 (31 600 to 35 100)** |
| 60% | 38 800 |
| 50% | 44 100 |
| founders uniform in the volume (the value at step 0) | 61 500 (58 200 to 64 900) |
| every layer at its even share (the plain sum) | 67 700 |

The *steady state* is taken as **70% of the agents in the floor layer**, which is an **upper-end** assumption, not a
central one. The cloud sweeps (`tiny3d`, 10 000 steps, seed 0) give the share of the agents in the lowest quarter of the
column (`band_frac[0]`, `z < 6`, wider than the floor layer `z < 4`) as 0.59 to 0.65 at g = 0.01 and 0.88 to 0.92 at
0.03. 0.015 is a quarter of the way from the first to the second, and a linear interpolation (an assumption in itself)
gives 0.66 to 0.72 for `z < 6`, so the narrower floor layer is at or under about 0.7 there. CPU runs of `tiny3d` at
g = 0.015 (review; seeds 0, 1, 2, 12 000 steps) read, for seed 0, a floor-layer share of 0.44 at step 500, 0.53 at
1 000, 0.58 at 1 500, 0.63 at 3 000, 0.66 at 6 000 and 0.69 at 12 000 (seeds 1 and 2: 0.53 to 0.62 at 1 500, 0.56 and
0.61 at 3 000, then 0.34 and 0.46 at 12 000, as they spread into the column again); the `DEV3D` of this PR (below)
reads 0.36 at step 1 000, 0.46 at 2 000 and 0.48 at 2 500, still rising. So 70% is where a `tiny3d` run that sinks arrives after thousands of steps and `DEV3D` has not reached it
in 2 500; 60% (`K_est` 38 800, `n_init/K_est` 0.42) or 50% (44 100, 0.37) describe a run of a few thousand steps better.
The sizing is kept at the owner's "mostly benthic" (70%: 1 700 gives 33 600), and the first tuning run reads the realised
share before `regen_max` moves; the sensitivity is 5 300 per 10 points of floor-layer share. The first estimate of about
1 000, made when every layer was to be counted (39 800 for an even column), gives 19 800 at 70%. At step 0 the founders
are uniform in the volume, so `K_est` starts at 61 500 and falls towards the benthic value as they sink; `n_init/K_est`
is 0.27 at step 0 and 0.49 at 70% (the occupancy ratio; the extinction-risk flag reads the floor layer's, §27.10, where
`n_init/K_floor` is 0.92 at step 0). **The 70% is a steady state of a taller
column than the cloud swept**: the sweeps ran at height 24, where a passive agent sinks from the surface in `height/g` =
800 steps (g = 0.03) or 2 400 (g = 0.01); at `64k3d`'s 80 and 0.015 that is 5 300 steps (2 700 in `DEV3D`), against a
lifetime of about 1 700 steps on `tiny3d` (150 births per 1 000 steps for 250 agents), and children are born where their
parents are. Founders that swim down faster than they sink get there sooner (`DEV3D` had 67 737 floor clamps in its first
300 steps, and 21% of its agents in the floor layer at step 300), but the floor-layer share of a `64k3d` run may approach
70% slowly or settle below it; `K_est` then stays above 33 600 (it is 38 800 at 60%), which makes the realised population
fraction of `K_est` lower, not the capacity estimate wrong.

**Population sizing.** `n_max` 65 536 and `max_births_per_step` 1 024 are 64k's. `n_init` = 16 384 (64k's too) is 0.49 of
the 70% `K_est` and 0.92 of the floor-only one; the owner asked for a value "near the expected K", and this is a hedge
against over-seeding, **to be signed off** (below). The first generation multiplies the founders by 1.4 to 1.7 within
about 100 steps (observed above: `tiny3d` 256 to 442 at step 64; the `DEV3D` of this PR 4 096 to 6 827 at step 57, 1.67×),
so it peaks at 23 000 to 28 000, which is 0.68 to 0.83 of 33 600, under `K_est` at step 0 (61 500) and under what it
will be as the founders sink. The population then falls back to where the food holds it. Measured, none of it at
`64k3d`:

| run (director off) | population against `K_est` |
|---|---|
| `tiny3d`, g = 0.03, seed 0, step 2 000 | 0.56 of the floor-only, 0.39 of the occupancy-weighted (237 agents against 424 and 608) |
| `tiny3d`, g = 0.015, seeds 0, 1, 2 (review, CPU), steps 500 to 6 000 | medians of 0.57 to 0.72 of the floor-only, 0.21 to 0.34 of the occupancy-weighted |
| the same, steps 6 000 to 12 000 | seed 0 0.30 to 0.35 of the occupancy-weighted; seeds 1 and 2 grow again, to medians of 0.52 and 0.40 (up to 0.73), 1.44 and 0.95 of the floor-only |
| `tiny3d`, g = 0.03, seed 0, 20 000 steps (cloud-6) | 242 agents at 2 000, 291 at 10 000, 486 at 20 000 (1.08 of the floor-only 450); seed 1 reaches `n_cap` (992) at about 16 000 |
| `DEV3D` (this PR, g = 0.015; review, CPU, 2 500 steps), steps 1 300 to 2 500 | 3 000 to 3 100 agents: 0.70 to 0.73 of the floor-only `K_est` (4 260 to 4 350), 0.24 to 0.25 of the occupancy-weighted (13 000 to 12 400), 0.73 to 0.76 of `n_init` |
| `DEV3D` (cloud-7, g = 0.015, C = 24, 2 500 steps, Linux CPU) | 4 096 founders, a peak of 6 821 within the first 250 steps, 3 013 to 3 134 agents from step 1 500: 0.24 of the occupancy-weighted `K_est` of the final state (12 759; 14 190 for the initial one), grounded share 0.30 (the floor-layer share was not recorded) |
| 2D 64k | about 0.8 |

So the occupancy-weighted `K_est` overestimates the realised population by about 4× on `DEV3D` (cloud-7), and the
initial state of `64k3d` reads 65 167 in `alife inspect` (one seed, just above the 16-seed range of 58 200 to 64 900
above) against an `n_cap` of 64 512: a number that no population reaches, one of the reasons the extinction-risk test
reads the floor layer's `K` instead (§27.10).

The settled fraction is not one number: it depends on gravity, the seed and the time (the long `tiny3d` runs are
still growing after 10 000 steps), and **a short tuning run's value is a lower bound**. If `DEV3D`'s fraction carries over
to `64k3d` (it is that preset's geometry at a quarter of the floor, with a floor-layer share of 0.48 and rising), the
population settles near 0.7 × 17 800 = 12 500 (0.37 of the 70% `K_est`, 0.76 of `n_init`) at a few thousand steps, and
more later. That is below the 33 000 the sizing aimed at, which the 2D neighbour density asks for; the measure for
`regen_max` is therefore the population against the *floor-only* `K_est` (about 0.7) and the realised floor-layer share
at more than 10 000 steps, not one reading at 2 000. The earlier version of this text said that seeding at `K_est` would
peak at 47 000 to 57 000 and crash as the cloud `DEV3D` did: that is an extrapolation, since that run was seeded at 2.5×
(of the 70% value) to 5× (of the floor-only one), and `DEV3D` here, seeded at 0.47 of the 70% value, boomed by 1.67× and
fell by 55% (6 827 to 3 090) all the same. The Mac tuning run decides `n_init`.

**`C` against the food.** The cloud evidence for `C` (no overflow at C = 16, peak occupancy 13, on `tiny3d` and the cloud
`DEV3D`) was taken at the default `regen_max` of 320. At 1 700 the same floor feeds about 5× as many agents, and `C` caps
how many a hash cell can hold. The number of agents that the floor-layer regeneration inside one bottom hash cell can
feed (`Σ regen / c_ref` over the cell's area, each field cell supersampled 4 × 4; seeds 0 to 3; review script
`scratchpad/rev/v5.py`):

| preset | mean | 99th percentile | max | `C` |
|---|---|---|---|---|
| 2D `64k` | 1.37 | 3.84 | 5.22 | 16 |
| `tiny3d` | 0.99 | 3.72 | 4.29 | 16 |
| cloud `DEV3D`, `regen_max` 320 | 1.25 | 4.37 | 5.11 | 16 |
| **`64k3d`** | 6.41 | 19.96 | 23.29 | 48 (24 until cloud-7) |
| **`DEV3D` of this PR**, `regen_max` 1 700 | 6.67 | 23.20 | 27.17 | 48 (24 until cloud-7) |

So the room between `C` and what the food can feed in the hottest cell is more than 3× in 2D and in the cloud runs, was
about 1× at the preset with `C` = 24 (the layer-1 food shares the hash cell and adds to it) and is 1.8× to 2.1× at
`C` = 48 (against the observed pile of 31, 1.5×). It is a bound on what the food *could*
feed, not a prediction: consuming the full 1 700 q/step of a fertility-1 field cell at `harvest_max` 400 takes at least
4.25 agents in one 4 × 4 cell, a close-packed crowd, and the population settles near 0.7 of the floor-only `K_est`. But it
also means that the benthic steady state of `64k3d` is dense: 70% of 33 000 agents in the floor layer is 0.23 per unit²,
about 26 agents within the sensing radius of a floor agent against 3.6 in 2D, and a floor pile that overflows collapses
onto a point (above). The cloud run of the slow test confirmed it at `C` = 24 (a floor pile of 31). `C = 48` is
**provisional**: the Mac's `bench --stages` decides whether it is affordable, and `cell_overflow` and `max_occupancy` at
the benthic steady state of a real run (not a 300-step probe) whether it is enough; a lower `regen_max` with `K_est`
recalibrated is the alternative if `C` = 48 costs too much.

**Cost and memory (arithmetic from the shapes; not measured).** A row holds 10 279 bytes (`W1` 116 × 32 and `W2` 33 × 39
in float16 are 10.0 KB; 9.0 KB in 2D): 0.674 GB for 65 536 rows and 0.686 GB for the whole state with the pending lanes
and the fields (0.5 MB resource, 1.5 MB chemicals). The neighbour search gathers and ranks 1 296 candidates per agent
at `C` = 48 (648 at 24, 432 on `tiny3d`, 144 in 2D), the fields are 2× the 2D 64k lattice, the brain 1.14×: docs/3D.md §9
estimated 30 to 40 steps/s against 61 in 2D for `C = 16`, and that figure is not to be expected at `C` = 48. **The
neighbour search's transients are larger than the state**: `N·Q` = 84.9 M, so one `(N, Q)` int32 or float32 array is
340 MB (170 MB at `C` = 24, 38 MB in 2D, §4), and `_find_neighbors_3d` builds the slot, the clipped slot, the three
coordinate gathers and offsets, `d2` (twice, with the mask) and the full `argpartition` index array: six to eight such
arrays live at once, **2.0 to 2.7 GB at the peak** (1.0 to 1.4 GB at `C` = 24), 9× the 2D search and 3 to 4× the 0.69 GB
of the state, less where `mx.compile` fuses them. The total of **about 2.8 to 3.5 GB** (§15.5; 1.8 to 2.2 GB at `C` = 24)
is an estimate that `bench --stages` has to check.

The only measurements are the cloud CPU's. Cloud-6 (Part C, `DEV3D`, 16 384 rows): 1.56 steps/s at `C` = 16 and 0.75 at
`C` = 48, so `C` = 48 costs 2.1× the step time and 1.7× the memory of `C` = 16, and about 1.6× the time of `C` = 24 (1.2
steps/s, interpolated). Cloud-7 ran `64k3d` itself at `C` = 24 on the same kind of CPU: a median of about 4.3 s/step and
a peak of 1.99 GB; the same scaling puts `C` = 48 at about 7 s/step and, with the state and the fixed arrays at about
0.76 GB and the transients proportional to `Q`, about 3.2 GB (inside the 2.8 to 3.5 GB above). A CPU is not the Mac's
GPU: these numbers say how much `C` costs, not how fast `64k3d` runs. **`C` = 48 is provisional until `bench --stages`
has been run on the Mac; if it is too costly, the alternative is a lower `regen_max`** (fewer agents per hash cell, so a
smaller `C` suffices; `K_est` and `n_init` recalibrated).

**To measure on the Mac, once its GPU is free** (none of it is done):
1. `alife bench --preset tiny3d --stages`, then `--preset 64k3d --stages`: steps/s, the time of each stage (the
   neighbour search is the one to watch), peak memory; the estimate above is the thing to check.
2. `C`: `cell_overflow` and `max_occupancy` at the benthic steady state of a real `64k3d` run (thousands of steps, with
   the floor-layer share where it settles, not a 300-step probe) and in a crowded one, against 48 (`C` against the food,
   above; the pile was still growing at 2 500 steps of `DEV3D` at `C` = 24, so a longer run is the check): a lower
   `regen_max` with `K_est` recalibrated, or a larger `C` (the crowd evidence reached 28 for a floor pile and 36 for a
   converging ball; C = 64 held both), if the floor overflows; a lower `C` only if the bench finds 48 too costly and the
   peak stays far under it. Re-run the slow `DEV3D` test at `C` = 48 (the cloud worker; about 50 to 55 minutes there
   against 32 at `C` = 24).
3. A `64k3d` tuning run (seed 0, a few thousand steps, then more than 10 000 where the budget allows: the long `tiny3d`
   runs grow again after about 10 000 steps): the first-generation peak against 23 000 to 28 000, the settled population
   against both the occupancy-weighted and the floor-only `K_est` (`DEV3D` predicts about 12 500, 0.7 of the floor-only),
   the floor-layer share (the 70% assumption; `DEV3D` read 0.48 at 2 500 steps), the grounded share against the sweeps,
   the `n/K_floor` series (the extinction-risk criterion, §27.10; 0.92 at the start, about 0.7 expected) next to the
   occupancy-weighted `n/K_est`, `fall` at the cadence of the run, the ledger, `cell_overflow`. Then `regen_max`, `n_init` (**the owner signs off the 0.49 of the 70% `K_est`**) and `gravity` are
   adjusted and the preset's comments updated.
4. `K = 12` (§27.13 above) needs a physics change (`drag` or `thrust_max`) and a run to decide whether it is worth it.
5. **The director in 3D, before any 3D director run** (§27.10): confirm `risk_k_frac` against the `n/K_floor` series of
   `64k3d` (the extinction-risk test reads the floor layer's `K` since 2026-10-15; `tiny3d` and `DEV3D` sit at 0.55 to 0.73,
   2.7 to 3.5× above 0.2, and the start at 0.92, so this is a confirmation and no longer a gate with a thin margin),
   decide the rescue's size, which still reads the occupancy-weighted `K_est` (a ceiling that reads 4× the population on
   `DEV3D`; it is the menu's 1 024-immigrant cap at `64k3d` whichever `K` sizes it), and `Director._blocked` on a live
   resume.
6. The first long run, analysed with `alife activity` (the signal tools are 2D-only, `require_2d`), is the results PR.

**Tests.**
- `tests/test_config.py::test_preset_table_3d`: the derived sizes of `64k3d` (N, M, L, height, FG, FZ, the hash grid
  G × G × G_z, `C` = 48 and `Q = 27·C` = 1 296, `D_IN`, `D_OUT`, the weights), exactly the keys that differ from `64k`, the values of §27.13's
  table, the founder brain cost equal to the 2D one within 0.2%, the validation inequality, and the `fast` overlay.
- `tests/test_metrics_3d.py`: the occupancy rule (§27.10), its behaviour for `n ≤ FZ` survivors (every occupied layer
  weighs 1, the plain sum for one per layer, a lone top-layer survivor under the floor-only value, the jump at `n = 0`) and
  `test_k_est_band_of_the_64k3d_initial_state`: on the lazy
  initial state of `64k3d` (`state._initial_state`, evaluating only the alive mask, the positions and the patches; the
  (65 536, 116, 32) weights are never materialised, as in `test_state.test_schema_every_preset`) for three seeds,
  `K_est` of the uniform founders is 55 000 to 68 000, of 70% of them on the floor 30 000 to 36 500 (and `n_init` is
  0.4 to 0.6 of it), of all of them on the floor 16 000 to 19 500.
- `tests/test_affordances_3d.py`: 400 compiled `tiny3d` steps (seed 0, the 2D bias row, CPU, about 7 s): every 2D
  affordance, floor and surface clamps and grounded agents, `HEAT_BUOYANCY > 0` and equal to ledger slot 15, `full_check`
  clean, every alive height in `[z_lo, z_hi]` at every step and both bounds reached, pitch within ±80° and *changing*
  for surviving agents (the founders' random pitch makes "non-zero" prove nothing; a run with the pitch frozen has
  none), deposits in all three chemical channels of the volume and in the broadcast messages (two steps in, the layers
  three or more above the floor already hold every channel: a deposit on the floor layer alone cannot have diffused
  there), children's caps different from their parents', movement in x and y (also of grounded agents, whom the floor
  holds) apart from sinking, and six passive agents that lose `gravity/drag` of height per step for 40 steps without
  moving sideways.
- `tests/test_ecology_3d.py` (marker `slow`, not run by the fast suite): `DEV3D` = `64k3d` with `population: n_max 16 384,
  max_births_per_step 256, n_init 4 096`, `world: size 160, height 40`, `fields: n_patches 4, patch_sigma 17`, `sensing:
  cell_capacity 48` (set in the test and checked equal to the preset's; **provisional**, §27.13) and float32
  weights (a quarter of the floor, the same patch coverage, `K_est` 8 700 at 70% benthic and 15 600 for uniform founders,
  `n_init` 0.47 of the first), 2 500 steps: never extinct, not pinned at `N_CAP`, births, bonds and landed attacks, no
  error flag and `full_check` clean, floor clamps > 0 (also in the last 500 steps), `HEAT_BUOYANCY > 0`, some agents
  resting on the floor and some swimming **in the final state** (a run-wide `GROUNDED / N_ALIVE` lies strictly between
  0 and 1 whatever happens, because the founders start uniform in z), hash cells over capacity in **at most 1% of the
  steps** (a floor pile that overflows does so for hundreds of steps in a row; a count of overflowing cells per agent and
  step would admit one cell overflowing on half the steps), and the initial `K_est` in 12 000 to 19 000. **The cloud run
  of this test at `C` = 24 (cloud-7) failed the overflow gate and passed every other assertion** (§27.13); the gate is
  unchanged and `C` is now 48, a value that has **not been run**: it is the cloud worker's or the nightly job's to check
  (a floor pile of 31 that was still growing at step 2 500 fits under 48, a longer pile may not). The 2 500-step run has
  not been done locally in the form of the test (the GPU is busy and it belongs to the cloud worker or the nightly
  job). **Its duration**: it took 32 minutes on the x86 cloud worker at `C` = 24 (1 944 s, run beside the fast suite),
  and about 1.6× that, 50 to 55 minutes, is to be expected at `C` = 48 (cloud-6 Part C ran this geometry at 1.56
  steps/s median with C = 16 and 0.75 with C = 48, so about 1.2 steps/s at C = 24); with the 30 minutes of the rest of
  the slow suite on that worker (cloud-7) it is about 80 to 85 minutes, under the nightly job's 120-minute timeout but
  with less room than before. A review run of the same configuration on the Mac CPU for 2 500 steps (14
  minutes; the assertions were not applied to it) read the population of the table above and a floor-layer share of 0.48 at
  the end. A 300-step CPU probe of the same configuration (79 s; the test's
  assertions were not applied to it) read: 4 096 founders, a first-generation peak of 6 827 at step 57 (1.67×),
  6 323 at step 100, 4 213 at step 300; the floor-layer share 0.08 at step 25 and 0.21 at step 300, the grounded share
  0.01 and 0.10, `K_est` 14 500 at step 25 and 14 000 at step 300 (the column is still in use), no hash overflow, 67 737
  floor clamps, 6.1 M quanta in `HEAT_BUOYANCY`. The `K_est` of the initial state over 16 seeds is 13 800 to 16 500
  (mean 15 600) for uniform founders, 7 700 to 9 100 (mean 8 700) at 70% in the floor layer, 4 000 to 4 700 on the floor
  only; the test's band (12 000 to 19 000) holds for all 16.

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
| I38 | feat: 3D world PR 3, field dynamics, energy and enablement (`docs/3D.md` §10 item 3; new §27.6–§27.8): `world.dims = 3` is enabled (`DIMS_SUPPORTED = (2, 3)`) and runs end to end with the new, provisional `tiny3d` preset; every 2D trajectory, config hash, metrics row (slot 15 still `RESERVED`) and checkpoint key set is unchanged (2D golden not re-pinned, no `FORMAT_VERSION`, `RNG_LAYOUT_VERSION` or genome-version bump; the CPU digest matrix of 2D trajectories and runs matched bit for bit after each change). (1) Fields: `ops.int_diffuse_3d` (exact no-flux integer diffusion, z never rolled; `int_diffuse` unchanged), the 7-point chemical stencil with mirrored ghost layers, `deposit_chem` strided by `n_cells`, `sample_fields(…, pitch=)` with z gradients (clamped index, divisor 2), corpses sinking to the floor cell `fcell mod FG²` under `FLOW_CORPSE` (combat spill and residue stay in the victim's own 3D cell), and the input tail 69–83 (`build_inputs(…, grounded=)`). (2) Energy: `pay_costs3`/`Costs3` (turn term on `|turn| + |Δφ|`; `buoyancy_cost·(buoy_cap·b)² + buoyancy_cap_cost·buoy_cap` per step, the upkeep paid even without lift; solvency on the total; paid basal, action, buoyancy), the keys `energy.buoyancy_cost` and `buoyancy_cap_cost` (int, 0; `POST_V01_KEYS`, `THREE_D_KEYS`; in 3D `< 2¹⁶` and 0 without the actuator), booked in `HEAT_BUOYANCY`, ledger slot 15 (`RESERVED` stays the canonical name and is still checked 0 in 2D). (3) Invariants and metrics: the closures over `sinks(nd)` in `device_check`, `full_check` and `ledger_error`; z, pitch and non-finite `pitch`/`buoy_cap` under the existing `POS_RANGE` and `NONFINITE` bits; `_check_rows_3d` with its own meta-test; `Counter3D` (27–29) and `n_counters(nd)`; `interval_row(…, nd=)` refuses a counter block of the wrong width instead of reshaping it; per-dims ledger names; `chem_mean` over the volume; the founder's cap upkeep in `c_ref`. (4) Step: static `d3` branches at S4, S5, S7, S8, S15 and S18 with every 2D line unchanged, `StepOut.counters` (30,) with `Counter3D` appended, `paid_buoy` as an 11th exact-sum column, `INT_HEADROOM` on the x/y column occupancy (a floor cell receives the corpses of its whole column), every output derived from its inputs (I14); `bench._StagedStep` mirrors each branch. (5) Enablement: `tiny3d` (values and reasons in §27.8; hovering costs 9 q/step by buoyancy and by thrust), fixtures `tiny3d_cfg` and `micro3d_cfg`, a 3D fork refusing a changed `physics.radius` as it refuses `world.height` (exit 4; the radius sets `z_lo` and `z_hi`, §27.1), and the 3D golden test (`GOLDEN_3D`: `state_digest` of `tiny3d_cfg` after 50 steps with `HEAT_BUOYANCY > 0`; its digest is empty until recorded on Linux-x86_64-cpu with MLX 0.32.3, where the skip message prints it). Interpretations: the diffusion bound is `resource_diffusion_q8 ≤ 42` as PR 1 validates, not 43 (an empty cell between six saturated ones would exceed `R_HARD`, §27.6); `Motion3` is masked by solvency for symmetry with `Actions`, though an insolvent agent dies at S14 and its `Motion3` has no effect; `metrics.jsonl` rows do not carry the `Counter3D` columns yet (PR 4); the 3D `K_est` sums every layer's regeneration and is not comparable to a 2D one (1 456 on `tiny3d` against 426 on `tiny`, §27.7), its meaning to be settled before the director runs in 3D; the GUI's preset list and 3D levers wait for PR 5. Existing tests that pinned the pre-enablement gate were edited: `test_dims_config.py` (`DIMS_SUPPORTED == (2, 3)`; the rejection test narrows it to `(2,)`; `test_defaults_are_2d` leaves out `tiny3d`), `test_dims_runio.py` (the exit-2 fork test narrows it to `(2,)`; the schema test pins `state_cls(cfg)` to `State` or `State3` by dims and expects its fields) and `test_dims_config_pr2.py` (the PR 2 `THREE_D_KEYS` pin checks `[2:10]`). Tests: `tests/test_energy_3d.py`, `tests/test_invariants_3d.py`, `tests/test_metrics_3d.py`, `tests/test_step_3d.py`, `tests/test_affordances_3d.py`, `tests/test_determinism_3d.py`, `tests/test_cli_smoke_3d.py`, and new functions in `test_ops_3d`, `test_fields_3d`, `test_lifecycle_3d`, `test_sensing_3d`, `test_bench`, `test_runner`, `test_config` and `test_cli_smoke`. |
| I39 | feat: 3D world PR 4, metrics, the director and the CLI in 3D, and `alife stream` (`docs/3D.md` §10 item 4; new §27.9–§27.11): no change to the 2D compiled step, the state, the RNG, any config key's meaning, the golden digests or any 2D output, and in 3D the compiled step changes only in the `GROUNDED` counter (review fixes below) (the CPU digest matrix of 2D trajectories and runs, 2D director runs included, matched bit for bit after each of the three stages; 2D metrics rows, director logs, LLM prompts, `inspect` output and the `render` palette are byte-identical, the prompts pinned by sha256). (1) Metrics (§27.9): 3D-only snapshot keys after `diag` (`height_mean`, `height_q25/50/75`, `surface_frac`, `speed_mean`, `pitch_abs_mean`, `buoy_cap_q10/50/90`, `band_frac` over four equal z bands, `clade_height_mi`, and with the heavy snapshot `body_z_extent_mean/max`) and interval keys after `v_clips` (`z_floor_clamps`, `z_surface_clamps`, `grounded_frac` = Σ `GROUNDED` / Σ `N_ALIVE`, `buoy_heat`); no snapshot key is named `grounded_frac` (`interval_row` raises on a clash). (2) `carrying_capacity` counts the **floor layer's** regeneration only in 3D (owner decision: the benthic analogue of the 2D lattice; `c_ref` keeps the founder's cap upkeep; `tiny3d` 409 against `tiny` 426, 1 456 when every layer was summed); the first row, `inspect` and the director's `n/K_est` read it; it ignores the photic band and is revisited with the tuning in PR 6 (§27.10). (3) Director (§27.10): `validate` no longer refuses `director.enabled` in 3D; `actions.radius_bounds` scales the catastrophe radius by L/1024 in 3D only (`tiny3d` [2.5, 25]); patches move in x and y only (`patch_pos` stays (PP, 2)); the log header gains `"dims": 3` in 3D only and `validate_log` refuses a log of the other dimensions; `prompt.EFFECT_3D`, `effects(cfg)` and `world_text(cfg)` give the 3D wording, with a "(floor-layer food only)" qualifier on `K_est`; `RULES_VERSION`, the ten-component descriptor and every rule parameter are unchanged; `n/K_est` on `tiny3d` with the director off is 0.54 to 0.71 after step 210, so the `n/K_est` criterion of the extinction flag is off at baseline; the `fall > 0.3` criterion is off at the default cadence of 1 000 steps and fires at cadences of 200 or less (the founders' boom and bust, as on 2D `tiny`; provisional, to be measured on `64k3d`). (4) CLI (§27.11): `dims`, `world_height`, `field_grid_z` in the summary, `height=` in the announcement, a height line, a top-level `grounded_frac` and (for `--agent`) `height` and `grounded` in `inspect`, the height in `bench`'s environment and report; `render`, `view` and `analysis.signals` keep their `require_2d` refusal (porting the signal decoder is out of scope). (5) `alife stream` (§27.11, `stream.py`): a framed binary protocol on a private duplicate of fd 1 (`ALSTRM\0\1` magic, a JSON header, `FRM1` records of a 32-byte head and the sections alive, pos, orient, rgb, bonds and an optional field slice, all little endian and 4-byte aligned), colours computed with the renderer's palette, a simulating mode that re-simulates like `view` (one `mx.eval` per step, donation rule) and `--follow`, which reads the run's newest checkpoints and is the faithful view of a director run; stdin control lines (`pause`, `run`, `step [N]`, `every K`, `color MODE`, `fields on`, `fields off`, `fields z K`, `quit`; EOF = quit); a closed reader is exit 0 with no traceback; the run directory is never written. `render._background` is split into `_cell_rgb` (a flat per-cell colour on which the 2D code calls it unchanged) with the public `agent_colors` and `field_colors`; the fixtures `tests/data/stream_tiny_{2d,3d}.bin` and their JSON sidecars serve PR 5's Swift decoder tests. Interpretations: heights are the agent centre z; `surface_frac` counts `z ≥ z_hi − floor_eps`; `clade_height_mi` is a plug-in estimate in nats; a body's extent is max z − min z of its members; a 2D stream carries three floats per position (z = 0) and pitch 0; the field layer and an invariant-violation flag travel in the frame's `flags` (the 8 reserved head bytes stay zero); the command has no default preset and `--follow` refuses `--steps`, `--every` and `--checkpoint`; `step` in follow mode fetches the newest checkpoint once; the stdin reader uses `os.read` (a daemon thread blocked in a buffered reader can abort interpreter shutdown); the fixtures are built scenes, not stepped runs, so their bytes do not depend on the platform's floating-point dynamics (the byte comparison runs on the CPU device only); `validate_log`'s refusal of a log of the other dimensions and the "(floor-layer food only)" qualifier are additions beyond the plan. **Review fixes** (a second commit on the PR): (a) `Counter3D.GROUNDED` counts the agents that were grounded at the start of the step and survive it (`grounded & alive0 & ~dead`, §27.8): counted over the start-of-step alive set it was set against the end-of-step `N_ALIVE`, so a mass death (a catastrophe, an extinction interval) pushed `grounded_frac` above 1 (2.86 in a probe; 1 514 grounded agent-steps against 1 406 floor clamps in an ordinary `tiny3d` row at step 1 010); a change to a 3D output counter only, so the 2D step, the state, the RNG and the 3D golden digest are untouched. (b) `alife stream`: control lines are applied one at a time, so each `color` and `fields` line writes its own frame whatever the pipe read-chunking, a redraw is written before a `quit` or EOF that arrives with it, and `step` batches queue with a frame each; an overlong control line is noted on stderr; the C library's stdout buffer and `sys.__stdout__` are flushed before fd 1 is restored (a pipe's block-buffered C stdio had been writing into the protocol stream after the last frame, at `exit`); a full non-blocking stdout waits instead of reading as a closed reader; a closed reader is noticed when nothing is being written (`poll` for hang-up, each `--poll` seconds when following and each second when paused); `--follow` skips a checkpoint that vanishes between the stat and the read (the real loader raises `CheckpointError` for a missing file, not `FileNotFoundError`) and a vanished `LATEST`, and describes itself as streaming the newest checkpoint at each poll, not every one; `stream.py` has no subscript assignment (`test_lint` scans it); `stream` has its own `--preset` help; the fixtures were regenerated with a row of two bonds in slot order and with field colours and chemical levels away from rounding boundaries (asserted in `build`). (c) Documentation: §27.10's calibration states both flag criteria and the cadence, §27.9 says `clade_height_mi` saturates at the band entropy while clades are near singletons (the estimator is left for PR 6), the stream bullets of §27.11, the `stream.py` entry in §16, `height` for `H` in §27.10 and the wrapping of `docs/3D.md`. Existing tests edited: `test_carrying_capacity_3d_includes_the_cap_upkeep` takes its regeneration from layer 0 (the rule changed on purpose) and `test_dims_config_pr2.py::BAD_PR2` lost its `director.enabled` case (the rule it pinned is gone); the review fixes changed the helper `test_step_3d._counter3d_ref` (`GROUNDED` over survivors) and the constant `test_lint.SETITEM_MODULES` (adds `stream`); no other existing test function changed. New tests of the review fixes are in `tests/test_stream.py`, `tests/test_step_3d.py` (a mass death) and `tests/test_cli_report_3d.py` (an extinct state). Tests: `tests/test_metrics_3d.py` (extended), `tests/test_cli_report_3d.py`, `tests/test_director_3d.py`, `tests/test_stream.py` (with `tests/stream_decode.py`, an independent decoder, and `tests/stream_fixtures.py`). |
| I40 | feat: 3D world PR 5, the Swift 3D viewer in AlifeControl (`docs/3D.md` §10 item 5; new §27.12; provisional row, to be revised after the first run of the GPU tests and the click-through), in three commits (the third is the fixes of a review, below) and in `gui/`, the docs and one test of the engine's suite (`tests/test_gui_sync.py`, with three `config.json` files in `tests/data`) only: no engine code, config key, hash, golden digest or 2D output changes, and the 2D GUI's command lines are pinned byte for byte by a table of 69 configurations captured from the sources of the previous commit before the refactor (46 of them in the first two commits, 23 added in the review and captured from the same sources in the same way). (1) Package: `AlifeCore` (library, `package` access, no AppKit or Metal: `Levers` and `World3DLevers`, `DirectorSettings`, `PresetFile`, `MetricsTailer`, `FrameDecoder`, `StreamCommand`, `Camera`), `AlifeControl` (executable) and the test targets `AlifeCoreTests` and `AlifeControlTests` (the second imports the executable with `@testable`, so the renderer and the controller are testable without another library target); Swift 6 language mode, macOS 14, no resources, no unsafe flags, and no `@unchecked Sendable`, `nonisolated(unsafe)` or `assumeIsolated`. (2) Decoder (§27.12): `FrameDecoder` written from §27.11 alone and checked against the two committed fixtures and their sidecars (every field and the SHA-256 of every section), for any chunking, with every refusal, truncation at every byte and unknown records skipped; sticky errors that follow the events before them; binary16 kept as bits and converted by hand (`Float16` is arm64-only). (3) Process (`StreamController`, `StreamReader`): the protocol bytes never go through the run log's line splitter; a thread reads the pipe with `read(2)` into the decoder and leaves the newest frame in an `OSAllocatedUnfairLock` mailbox, waking the main actor through an `AsyncStream<Void>` with `bufferingNewest(1)`; stderr goes to the log as `[3d] ...`; control lines go to stdin through the throwing `write(contentsOf:)` (`F_SETNOSIGPIPE` on the descriptor, and SIGPIPE ignored at launch); the session finishes only after exit and end of file on both pipes, so the last frame and the status cannot race; stopping is once per process (`quit`, EOF, one SIGTERM after 2 s), on window close, replacement and (without the grace) app termination; a corrupt stream stops its process; closing the main window closes the viewer window so the app still quits. (4) Renderer (`Renderer`, `Shaders`): raw Metal through `MTKView`, the shaders one MSL string compiled when the first stream is shown (the installed app has no resource bundle and the Metal Toolchain is not installed); ray-traced sphere impostors with a pale nose, bond half segments to the minimum-image midpoint, box edges, translucent floor, surface and field-slice planes (drawn farthest first; the 2D field is the opaque background), three shared buffer sets with a semaphore, on-demand drawing (no display link), a draw that never waits for the GPU on the main thread (retry after 100 ms), snapshot to `runs/NAME/media/view3d_<step>.png`. (5) Input and window: orbit and fly camera (`Camera`), the keys of §20.2 where there is an equivalent, follow (the run in progress; the faithful view of a director run) or re-simulate (says that it ignores the director), `Open 3D viewer` for 2D and 3D runs, the pygame viewer's button disabled for a 3D run, a "3D world levers" box (new run against the preset, fork against its source, `world.dims` never an input, a fork keeps its height, `floor_eps`, `buoy_init` and `buoy_sigma` CLI-only), the 3D status row. (6) Tests (counts after the review fixes: `AlifeCoreTests` 91, `AlifeControlTests` 60 that run and 8 opt-in GPU tests; the first two commits had 80, 42 and 7): `AlifeCoreTests` 80 (79 of the first commit and the home pose), `AlifeControlTests` 42 that run (the reader on a real pipe; the controller against a stand-in script with real processes: last frame before the end, exit status and stderr, signal, corrupt stream, launch failure, command order and a single `quit`, SIGTERM after the grace, replacement, source switch, a write to a closed pipe; keys, speed, pointer mapping, planes, uniform layout, shader structure, PNG) and 7 opt-in GPU tests (`ALIFE_GPU_TESTS=1`; skipped otherwise; shaders compile, layers independent, every alive agent where the camera says, the seam bond as two stubs, field orientation and 3D slice blending, snapshot; the review added an eighth, below); `tests/test_gui_sync.py` ties the Swift keys, presets, personas and director modes to the engine. Interpretations: `read(2)` rather than `FileHandle.read(upToCount:)` (which can wait for the whole count on a pipe; a stand-in process test found it); a draw skipped for lack of a buffer set is retried, not waited for; `s` is a PNG here and `p` too (the 2D viewer's `s` writes a state file); bonds are one-pixel lines and there is no multisampling; the view keeps its colour mode, field layer and speed across a restart through the arguments; a 3D world's levers are not shown for a resume (it has none). Not run: the shaders and the drawing on a device (the GPU was busy with the evaluation arms, and the Metal compiler needs the Metal Toolchain or a device), the app itself, and the installed bundle; the opt-in GPU tests and the HANDOVER click-through are the checks to make first. Existing tests edited: none of the repository's from before the PR (`CameraTests` gained one test; `StreamFrame` gained an explicit memberwise `package init`, and `Camera.home(for:aspect:)`, for the viewer and its tests). **Review fixes** (the third commit; each was a reviewer's finding, verified against the code, and none changes the engine): (a) the field layer of the previous stream was passed as `--field-z` to a different run, and `alife stream` exits 2 for a layer past the run's last (a 2D run has one layer), which left the window with no way to recover: the layer is now carried only across a restart of the same run, and another run starts at layer 0. (b) `step` marked a followed run as paused, but the engine's `step` in follow mode only refreshes. (c) The reader woke the main actor after every pipe chunk and `apply` published a property on every wake-up, so each partial chunk of a frame redrew the old frame: it wakes only for a header, a frame or the end, and the view is redrawn only when its content changed. (d) `draw(in:)` asked for the drawable before the semaphore, so the main thread could block in `nextDrawable()` for up to a second and, on a timeout, lose the redraw: the ticket (a buffer set, and a place among the two draws into the window that leave a drawable for the screen) now comes first and a missing drawable is retried; and the ring index moved on before the encoding, so three failure paths kept it while giving the semaphore back, and a later draw could write a set that the GPU was reading: `DrawGate` moves the ring on only after the commit. (e) The close observer held the view weakly across the hop to the main actor, and closing any main window closed the viewer: the view is held, and `WindowGroups` counts the main windows (the main scene is a `WindowGroup`). (f) App termination ended only the current stream's process: it ends every one that has not finished. (g) A decoder error that arrived in the same read as an earlier event waited for the next bytes, which a paused or finished stream never sends: `FrameDecoder.failure` shows it and the reader publishes it at once. (h) The header accepted a missing `run` key and values that break §27.11 (`FG·field_cell = L`, `FZ`, the 2D shape, `color`, `field_z`, `mode`, `source`). (i) The clip planes were fixed when the pose was built (near 0.1, far `8 max(L, height)`): zooming out about four times clipped the world, and in a 1 024-wide world a depth step at the home pose was 0.75 world units; they follow the pose now. (j) The billboard did not cover the nose when the eye was within about two radii of an agent: it is the silhouette of the sphere that holds body and nose. (k) Switching `physics.buoyancy_max` on from a baseline without an actuator was accepted and did nothing (the caps start at `buoy_init` and drift by `buoy_sigma`, both 0 and both CLI-only): it is refused with a message; whether those two keys should be levers is the owner's decision. (l) `PresetFile` equals the cadence reader it replaced on every file `tomllib` accepts (a table header without its `]` names no table, as before; a key set twice is the last value, where the old reader kept the last integer of at least 1, and the engine refuses such a file). (m) Documentation: a followed run has its step-0 checkpoint from the start (the text said the first came after `checkpoint_every` steps, so item 9 of the click-through could not be reproduced), the counts, the file lists, the package row and the place and wording of the HANDOVER checklist. New tests: one or more for each of the above; an eighth GPU test that compares every pixel around the agents with a CPU port of the sphere shader (the colour channels, the headlight, the yaw and the pitch, the nose), and the 3D slice test at both field layers of the fixture; 23 more 2D command lines, from `config.json` files written by the engine and presets chosen through their files. Not run, as before: the GPU tests (the new one included; a mistake in the port would show as a failing test, to be checked against the shader first), the app, and the installed bundle. |
| I42 | feat: 3D world PR 6, ecology (`docs/3D.md` §10 item 6; new §27.13; provisional row and number: the evaluation PR holds I41 and the coordinator renumbers at merge; the GPU parts and the long run are not done): no change to the compiled step, the state, the RNG, any config key or its default, any config hash, golden digest or 2D output (the CPU digest comparison of the 2D config hashes, the initial 2D `K_est` of `tiny` and `dev` and the `tiny` state digest after 30 compiled steps against origin/main 748ad5b is identical; the only differences are the new preset and its two hashes). (1) **`K_est` in 3D is occupancy-weighted** (owner decision 2026-10-14; §27.10): `Σ_k regen_k·min(1, FZ·p_k)/c_ref`, `p_k` the share of the alive agents in field layer `k` (`fields.cell_index // FG²`, the layer harvest uses), the floor layer alone when nobody is alive, so it equals PR 4's floor-only value in the benthic limit and the plain sum over the layers for an even column; `metrics.layer_occupancy_weights` is new, the 2D branch is the same integer sum over `c_ref`; the first row, `inspect`, the director's `n/K_est`, rescue size and prompt read it; the prompt qualifier is now `prompt.K_NOTE_3D` (the 2D messages stay byte-identical, pinned by sha256; after review it says "in the floor layer", not "on the floor": the layer is `z < field_cell`, which is not the grounded state). Behaviour recorded after review: with `n ≤ FZ` alive every occupied layer has weight 1, so `K_est` is not continuous at `n = 0` (largest just before a total collapse of a spread population; §27.10). (2) **`64k3d`** (`presets/64k3d.toml`, `config.PRESETS`, the Swift `Preset` enum and the `PresetFileTests` list; owner decision: gravity 0.015): 320 × 320 × 80, FG 80, FZ 20, hash grid 53 × 53 × 13, `K = 8`, `C = 24` **provisional** (the evidence for it was taken at `regen_max` 320; at 1 700 the food can feed about 20 agents in the hottest floor hash cell, a margin of about 1×, so `cell_overflow` and `max_occupancy` at the benthic steady state and `bench --stages` on the Mac decide), `brain_cost_per_l1` 0.0905 (founder brain 37.63 q against 2D's 37.64), buoyancy costs 16 and 8, `regen_max` 1 700 for a `K_est` of 33 600 at 70% of the agents in the floor layer (an upper-end assumption: `DEV3D` read 48% at 2 500 steps; 61 500 at step 0, 17 800 floor only; mean of 16 seeds), `n_init` 16 384 (0.49 of that `K_est`, a hedge and not the "near K" the owner asked for, to be signed off: the first generation multiplies the founders by 1.4 to 1.7, the cloud crash was at 2.5× to 5× the `K_est`, and that seeding at 1× would crash is an extrapolation), the 64k run cadences; the derivations, the cloud evidence (gravity sweep, buoyancy costs, `C` and `K`, the `K = 12` conflict with `validate`) and what remains to be measured are §27.13. (3) **Tests** (new): `test_config.py::test_preset_table_3d`; in `test_metrics_3d.py` the occupancy weights and `K_est` on built scenes against a float64 host reference (all benthic, even column, mixed, top layer, nobody alive, dead rows, layer edges, the live environment, a 2D state, a few survivors: saturated weights and the jump at `n = 0`) and `test_k_est_band_of_the_64k3d_initial_state` (the lazy `_initial_state`, three seeds, the weights never materialised); in `test_director_3d.py` the prompt qualifier, `Director._vitals` on built scenes, a spread population flagged at about three times the floor-only size, the 40-founder Gardener run (a rescue raises its own `K_est` and the flag stays set), and a strict `xfail` for the blocked-note journal across a resume (2D and 3D); in `test_affordances_3d.py` 400 compiled `tiny3d` steps (every affordance, floor and surface clamps, `HEAT_BUOYANCY`, heights and pitch in range, the mutating cap, deposits in the volume, swimming apart from sinking) and a passive-agent sinking test; `test_ecology_3d.py` (marker `slow`, `DEV3D`, 2 500 steps; collected only under `-m slow`, **not run**: its helpers were exercised by a 300-step CPU probe, 6 827 agents at the step-57 peak and 4 213 at step 300, no overflow, floor clamps and `HEAT_BUOYANCY` positive, and its new assertions by a 60-step run; the full run belongs to the cloud worker or the nightly job and takes 30 to 40 minutes there, 11 on the Mac CPU). (4) **Existing tests edited, all for the rule changes above**: `test_dims_config.py::test_defaults_are_2d` leaves out `64k3d` as it leaves out `tiny3d`; `test_metrics_3d.py::test_carrying_capacity_of_tiny3d_is_comparable_to_tiny` puts every founder on the floor (the benthic limit), and the docstrings and one comment of the four PR 4 floor-layer tests say that their scenes have nobody alive (their assertions are unchanged and pass: with nobody alive the floor layer alone counts); `test_cli_report_3d.py::test_the_first_row_estimates_the_capacity_on_the_floor_layer` is now `…_by_occupancy` (the first row against the step-0 checkpoint to 5%, and between 1 000 and 1 500 instead of 300 to 500); `test_director_3d.py` asserts the new qualifier text, `…_use_the_floor_layer_k_est` is now `…_use_the_occupancy_weighted_k_est` (the logged `k_est` against a host reference of the weights, and between the floor layer's value and all the layers'), the Gardener fixture starts 150 founders instead of 40 (`_gardener_argv` takes `founders=`; see the findings below) and its assertion `n_alive > 40` is now `n_alive > 150 + the rescue's immigrants` (at 150 founders the old bound passed after a 70% decline), and the row dict of `_window_ctx` became the helper `_synthetic_row` (no assertion changed). **Two findings in the director, not fixed** (§27.10). (a) With 40 founders the occupancy rule leaves `n/K_est` at 0.14 to 0.15 for the whole 200-step run: a rescue raises its own `K_est` (immigrants uniform in the volume, a 1.5× pulse), so the extinction flag stays set for a population that has recovered (250 to 290 agents, twice the floor-layer rule's), and the rescue stays wanted and blocked by its cooldown at every boundary; pinned by `test_a_3d_rescue_raises_its_own_k_est_and_the_flag_stays_set` and a scene test; the 3D calibration of `risk_k_frac` (and the rescue's feedback) is an explicit gate before a 3D director run (§27.13). (b) `Director._blocked`, which journals a blocked rule only when it was not blocked at the previous boundary, is kept per process and not rebuilt on a live resume, so the resumed process journals the note once more, and `test_a_resumed_3d_live_run_is_the_uninterrupted_run_and_its_log_replays` (journal bytes equal) failed at 40 founders. `director.jsonl`, the archive, the rows and the state are identical across the resume. The code is shared with 2D; the director is left as it is (the evaluation runs on it), the fixture starts 150 founders, for which the flag clears at step 60 and no rule is blocked at the resume, and `test_a_blocked_note_is_journalled_once_across_a_resume` is a strict `xfail` (2D and 3D) to be un-marked by the fix (it needs a persisted set: the journal holds only the new notes). Also recorded: the `K_est` rule is not in the log header, so a 3D live run begun before PR 6 and live-resumed here switches rule without a trace and should be replayed or restarted. Interpretations: the nobody-alive case counts the floor layer (so an extinct run reports PR 4's value); uniform founders weigh the two edge layers 0.77 (`p = (field_cell − r)/(height − 2r)` against 1/FZ), so an even-in-z start is 91% of the plain sum; the steady state of the sizing is taken as 70% of the agents in the floor layer (an interpolation between the cloud sweeps at g = 0.01 and 0.03 that sits at the upper end); on `tiny3d` at its own gravity 0.03 the n/K_est of the extinction-risk criterion is 0.19 at step 10 and 0.27 to 0.39 later under the new rule (0.54 to 0.71 under the old), and at `64k3d`'s 0.015 it is 0.17 to 0.19 at steps 10 to 40 and 0.21 to 0.30 for thousands of steps (`DEV3D` 0.24 to 0.25), so its margin to 0.2 is smaller (§27.10). Swift: `Preset.k64_3d` ("64k3d") and a `64k3d` test each in `World3DLeverTests` (the preset's values, and an unchanged run passing the preset name alone) and `PresetFileTests` (the scalars, the engine-default cadence); `swift build -Xswiftc -warnings-as-errors` and `swift test` pass (the GPU suites stay skipped). **Review fixes** (a second commit; the findings were verified against CPU probes and `results/cloud-3.md`, `cloud-6.md`): (a) tests: the slow ecology test checks resting and swimming agents in the *final state* (the run-wide grounded ratio lay in (0, 1) from the founders' uniform start whatever happened) and floor clamps in the last 500 steps, and bounds hash-cell overflow by the fraction of steps with any overflow (at most 1%) instead of overflowing cells per agent-step (which admitted a cell overflowing on half the steps); the 400-step run counts surviving agents whose pitch changes (the founders' random pitch made "non-zero" pass with the pitch dynamics frozen; checked by a mutation, and counted by `uid` so a child taking a freed row is not a change) and `test_signals_reach_the_water` checks that the layers three or more above the floor hold every chemical after two steps (a mutation that deposits on the floor layer alone passed before); `n_alive > 150 + immigrants`; the comment of the column scene (two layers at weight 1, four at 0.75); new tests for a spread population, the 40-founder rescue, a few survivors and the blocked-note `xfail`. (b) Docs: §27.10 records the saturation for `n ≤ FZ`, the rescue's feedback, the 3D calibration of `risk_k_frac` at g = 0.015 (thin margin; a gate before a 3D director run), the K_est-rule and blocked-note consequences of a live resume, and the wording "in the floor layer" (§10.10 gives the weight `min(1, FZ·p_k)`); §27.13 quotes the gravity recommendation as 0.01 to 0.015 (0.02 untested), 28 for the floor pile and 36 for the ball, says the evidence for `C` was taken at `regen_max` 320 and adds `C` against the food (about 20 agents in the hottest floor hash cell at 1 700), the 70% as an upper-end assumption with the `DEV3D` and `tiny3d` shares, the settled fractions (they grow again after 10 000 steps, so a short run's value is a lower bound), `n_init` as an extrapolation to be signed off, the `(N, Q)` transients (1.0 to 1.4 GB; §15.5 has a `64k3d` column) and the runtime of the slow test (30 to 40 minutes on the cloud worker); the preset's comments agree with it. Not changed: the director code (`_blocked`, the rescue's feedback, the header), `regen_max`, `n_init`, `C`. Not done (no GPU): `bench --stages`, confirming `C` at the benthic steady state, a `64k3d` tuning run, the cloud/CI run of `test_ecology_3d.py`. **Extension (third commit; the owner's decision of 2026-10-15 and the cloud-7 run of this PR; where it contradicts the text above, it supersedes it).** (5) **The director's extinction-risk test reads the floor layer's `K` in 3D** (owner decision 2026-10-15; §27.10): `metrics.floor_carrying_capacity` is new (the floor layer's live regeneration over `c_ref`, PR 4's rule; in 2D it returns `carrying_capacity`); `Vitals` gains `k_floor`, `n_over_k_floor` and `n_over_k_risk` (3D only; `k_floor` and `n_over_k_floor` default to `None`, so 2D vitals carry nothing new), `Tracker.vitals` takes an optional `k_floor`, and `Director._vitals` passes it for a 3D config only; the flag is `n/K_floor < risk_k_frac` or `fall > risk_fall`. `k_est`, `n_over_k` (the crowding ratio of the log and the prompt), the rescue's size (`0.1·k_est`), the first row and `inspect` keep the occupancy-weighted value. The 3D prompt's viability line gains the floor ratio and number (`prompt.K_FLOOR_NOTE_3D`); a 3D log entry's `vitals` gains `k_floor` and `n_over_k_floor` after `n_over_k`; a 3D rescue's reason reads `n/K_floor …`. **2D is byte-identical**: the vitals' log keys and order (pinned by a test), the reason, the prompts (the sha256 pins pass unchanged) and every 2D director test. **`RULES_VERSION` stays 1**: the rule parameters are unchanged, 2D logs and replays do not change by a byte, and a bump would rewrite the header of every 2D log and make a live resume of a running 2D evaluation arm append a second header (§27.10). It resolves finding (a) above: the rescue no longer keeps its own flag set (40 founders: `n/K_floor` 0.098, a rescue of 131 immigrants, the flag clear from step 60 at 0.40 to 0.48), narrows the 3D calibration gate to a confirmation of `risk_k_frac` against `n/K_floor` (the margin is 2.7 to 3.5×, not 4 to 25%; the interpretations above that call it smaller are superseded) and to the rescue's size (still `0.1·K_est`, 131 against 41 on `tiny3d`, the menu's cap of 1 024 at `64k3d`), and the live-resume caveat now concerns the vitals' `K` and the rescue's size, not the flag. Cloud-7 measured the occupancy-weighted `K_est` at about 4× the realised `DEV3D` population (14 190 initial and 12 759 final against about 3 100 agents) and `64k3d`'s initial state at 65 167, above `n_cap`: the rule is kept for every consumer but the flag. **Existing tests edited:** the Gardener fixture returns to PR 4's 40 founders (`FOUNDERS`; at 150 the floor-layer rule fires no rescue, `n/K_floor` 0.37; the flag clears at step 60 and no rule is blocked at the resume, so the journal still compares equal; `_gardener_argv` loses its `founders=` parameter, the `gardener40` fixture is merged into `gardener_run`, and `n_alive > FOUNDERS + immigrants` replaces `n_alive > 150 + immigrants`); `test_a_spread_population_is_flagged_at_three_times_the_floor_only_size` is now `…_is_judged_against_the_floor_layer_not_the_occupancy_weighted_k` and `test_a_3d_rescue_raises_its_own_k_est_and_the_flag_stays_set` is now `…_but_no_longer_keeps_the_flag_set` (each pins the converse); `_window_ctx` gives a 3D context a `k_floor` of 250 and the two 3D prompt tests assert the floor clause; `test_the_director_vitals_use_the_occupancy_weighted_k_est` and `test_the_vitals_of_a_3d_run_use_the_occupancy_weighted_k_est` also check `k_floor`, `n_over_k_floor`, the reason and the key order; the two strict `xfail`s of `Director._blocked` are kept. **New tests:** `test_metrics_3d.py` (`test_floor_k_is_the_floor_layer_alone_wherever_the_agents_live` on six scenes against an independent float64 reference, `test_floor_k_follows_the_live_environment`, `test_floor_k_in_2d_is_the_one_k`), `test_director_vitals.py` (`test_2d_vitals_have_one_k_and_the_log_keys_of_build_step_2`, `test_3d_extinction_risk_reads_the_floor_layer_k_and_every_other_reader_keeps_k_est`, `test_3d_fall_criterion_is_independent_of_the_capacities`, `test_3d_compact_adds_the_floor_k_after_n_over_k_and_is_json_ready`), `test_director_rules.py` (`test_the_rescue_reason_names_the_ratio_the_flag_read_and_its_size_reads_k_est`, `test_a_blocked_3d_rescue_is_noted_with_the_floor_layer_ratio`), `test_director_3d.py` (`test_the_3d_prompt_says_which_capacity_the_extinction_flag_reads`); three mutations (the flag reading `n_over_k`, `Director._vitals` passing no `k_floor`, the rescue sized by `k_floor`) are caught by three, four and two tests. (6) **`C` = 48 (provisional) for `64k3d`**: the cloud-7 run of `test_ecology_3d.py` on Linux CPU at `C` = 24 failed its overflow gate and nothing else (one floor hash cell over `C` continuously from step 2 151, a pile of 31 still growing, 11.96% of the steps against 1%). `64k3d.toml` and the `DEV3D` configuration of the test (`CELL_CAPACITY`, set and checked equal to the preset's; the gate is unchanged) now have `cell_capacity` 48, `test_preset_table_3d` asserts `C` = 48 and `Q` = 27·48 = 1 296, and §27.13, §15.5 and `docs/3D.md` carry the derived numbers: `(N, Q)` arrays of 340 MB (170 at `C` = 24), transients of 2.0 to 2.7 GB, a total of 2.8 to 3.5 GB (3.2 GB scaled from cloud-7's 1.99 GB peak for `64k3d` at `C` = 24), and the cost from the cloud CPU (`DEV3D`: 2.1× the step time and 1.7× the memory of `C` = 16, about 1.6× the time of `C` = 24; `64k3d` about 7 s/step on that CPU at `C` = 48 against 4.3 at 24). **`C` = 48 is provisional pending `bench --stages` on the Mac; a lower `regen_max` is the alternative if it proves too costly.** **Checks (CPU only, the GPU was busy):** `ruff check` and `ruff format --check` clean; the fast suite on the CPU device: 1 464 passed, 3 skipped, 2 strict xfails (`Director._blocked`) and the known Mac-CPU failure `test_lifecycle.py::test_lifecycle_compiles_and_matches_eager`; the 2D digest matrix against origin/main 748ad5b (config hashes, state digests of 30 compiled steps, the 2D director runs including their `director.jsonl`, journal and archive bytes) bit-identical, the only differences being the new `64k3d` hashes (the preset's hash changes with `C`). Not run: the slow test at `C` = 48 (the cloud worker's, about 50 to 55 minutes against 32), `bench --stages`, the Swift tests (`gui/` is unchanged and the GUI reads no `cell_capacity`). |

**Open questions for the user.**
1. Contact-attenuated thrust makes a contracted body's interior agents (those with contacts) push less, so dense
   bodies move mainly by contraction and grip. Acceptable, or should bonded contacts be excluded from the count
   (at the cost of large bodies again being able to compress crowds)?
2. Should a bond partner be able to rescue a combat victim drained to 0 at S12 (current design: yes), or should
   `E == 0` after combat be final?
