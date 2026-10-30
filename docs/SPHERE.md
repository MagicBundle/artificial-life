# The sphere world (`world.geometry = "sphere"`): specification

> **Status in the repository (PR S0, 2026-10-29).** This is the series specification, revision 1, kept here so that the design lives with the code. `docs/DESIGN.md` §46 is
> normative for what is built; where the two differ, §46 and the Review-log row I77 win. S0 built section 1 (the grid), the geometry part of 2.2 and 2.3, 2.9, 7.1, 7.2 and the S0 row of
> section 10; the deviations are: coverage and the rigorous angle are geodesic angles (the review used chords, 0.1 to 0.2 % lower), `validate` caps the derived `Nh` at 64, the exact coverage
> angles for Nside 2..72 ship as `src/alife/data/healpix_coverage_reach2.json` (no 3 s wait per process), `lsq_weights` exists for the 4 edge and the 8 touching neighbours,
> the greedy colouring gives 15, 16, 15 colours at Nside 8, 11, 21, D10 changed (no astropy-healpix: `tests/hp_oracle.py`), and the geometry lint scans all arrays indexed `[..., 2]`,
> not only names ending in `pos`/`vel`. The probe scripts named below are not in the repository; the numbers that S0 builds are reproduced by its tests.
>
> **S1 (2026-10-30).** Built section 3.1 (resource, chemicals), the fields part of 2.1 (`StateS`), 2.7 (patches, founders) and the S1 row of section 10; see DESIGN §46.5. Deviations: the gradient uses the 4 edge neighbours
> (`W (Npix, 4, 3)`); the chemical sub-steps are 4, 4, 3 (the 64k3s row sum 16.4 gives 3, not 4), at most 16 by a `validate` rule; the chemical update costs about +0.7 ms at 64k3s on Metal (not +0.3); `wx` is a zero placeholder
> `(11, 12·(2·Nf)²)` until S4; the founders and `ops.body_axes_s`/`to_body_s` come in S1; the horizontal diffusivity at the floor is 1.2 to 1.4 times the box's `d_q8/256·field_cell²`; the S1 spherical-harmonic gate allows 4 %
> at Nside 8, l = 6 (the measured −3.4 %; the spec's own text says 3.3 %); `fieldtab.py` and `fields_sph.py` are geometry-definition files of the lint.

> **S2 (2026-10-30).** Built section 2 (agents), 2.9 (the readiness gates and the rotation harness), 7.2 (`GEOMETRY_SUPPORTED`, `SPHERE_READY`) and the S2 row of section 10; see DESIGN §46.3, §46.4, §46.10. Deviations: the child sits at a chord exactly `D`
> from the parent (the arc is solved, the spec's arc `D·cosχ` is not a chord); the compile-collision guard was not merged, so the 10.1 fallback is in force (child processes with their own `TMPDIR`, both orders, `sphere_check` in `make_step`); CPU and
> Metal digests are never equal (as for the box) but every integer field, counter and ledger entry is; the box's "velocity into the bound" invariant has no sound sphere form and is dropped; `run`/`resume` work on a sphere
> with two metrics lines converted early, `stream`/`activity`/`migrate` refuse it and `play` is refused through the observers gate; the `fast` overlay is invalid on a sphere. **The stop rule of 2.4 is exceeded by the literal per-alive-agent
> metric at the volume-matched `mini3s` (search 2.09x, step 2.0 to 2.2x of `mini3d`) and not at equal count, higher density or 64k (1.1 to 1.5x); see DESIGN §46.10. Reported, not optimised.**

Status: proposal for owner review, 2026-10-29; **Revision 1** (same day) applies the independent review `review.md` (changelog at the end). Planned identifiers: DESIGN §46, Review-log row I77 (I76 is taken by a GUI fix).
Owner request (2026-10-29): "a true sphere and not a wrapped sheet". This document is a specification only: no engine code
was written or changed. Read-only inputs: main at 7ab28cf (DESIGN §27, §30, §32, §37 to §43, §45, F20; `src/alife/*`; the
GUI globe). The probe scripts in this folder are numpy-only and ran on the CPU (a few minutes in total, the longest 65 s); the
numbers below marked "(probe)" come from them. Numbers marked "(estimate)" are not measured.

Probe index (all in `sphere/`, run with `.venv/bin/python -I <file>`): `grid_probe.py` (three tessellations, cell
statistics), `healpix_probe.py` (HEALPix at the Nsides the presets use), `reach_probe.py` (hash stencil of reach 1 and 2),
`hier_probe.py` (Nside N to 2N refinement), `transport_probe.py` (float32 motion and heading transport on the sphere),
`sl_probe.py` (semi-Lagrangian transport on HEALPix cells), `orient_probe.py` (where the grid's trivalent vertices sit),
`sizing.py` (the preset arithmetic).

---

## 0. Decisions at a glance

| # | question | decision |
|---|---|---|
| 1 | grid | **HEALPix, RING ordering, any integer Nside** (12·Nside² equal-area cells), one family of resolutions: field/hash Nside `Nf`, weather `Nw = r·Nf` with `r` a power of two, territory `Nt`. No tilt of the grid against the planet axis (R1 dropped it). Cubed sphere is the runner-up (1.3), icosahedral hex is kept only as the weather fallback, lat-lon is rejected |
| 2 | agent position | **Cartesian `pos` (N,3) in the planet frame** (every 3D pair expression survives), height `z = |pos| − Rp`, up `û = pos/|pos|`; heading is a **tangent unit vector `hdg` (N,3)** plus the existing pitch; the scalar `heading` stays as a derived, output-only azimuth. Because `pos[:,2]` and `vel[:,2]` then type-check but are wrong, **a geometry accessor layer and a lint rule guard every consumer** (2.9) |
| 3 | motion | exponential-map step (series, no trig), minimal-rotation transport of `hdg`, mandatory per-step renormalisation (probe: invariants hold to 1e-7 over 3·10^5 steps; without it `|hdg|` drifts by 0.11) |
| 4 | gravity, floor, lid | radial gravity (constant magnitude), floor = sphere of radius `Rp + r`, **hard lid** at `Rp + H − r`, optional soft band (default off) |
| 5 | pair geometry | Euclidean chord in R³ for neighbours, contacts, springs, combat and hearing; `world.size` is the image length of `min_image` and must exceed `4(Rp+H)`, so `min_image` is the identity **on pair offsets** (bonds/energy/featobs pair code unchanged; every non-pair use is audited, 2.9) |
| 6 | radial structure | `FZ` **equal-volume** layers, so every field cell has the same volume: regeneration per cell, harvest, `R_HARD` and the equilibrium (uniform quanta = uniform density) keep their box meaning. **The diffusion arithmetic does not carry over verbatim**: int64 fluxes with geometric (physical-isotropic) weights and a proven cap bound (3.1) |
| 7 | neighbour search | table-driven stencil over (cell, radial layer), **reach 2 only** (25 angular cells, 5 radial layers; 1.27× the box in candidate volume at 64k3s, against 2.6× for reach 1); `Nh` and `C` are derived from exact tables, not constants; new Metal kernel variant |
| 8 | weather | collocated Cartesian-vector finite volume with **gradient-corrected faces, a Rhie-Chow compact pressure term and a biharmonic filter** (all three are required for stability, measured in 4.3), exact Coriolis rotation about `û`, `f = 2Ω sin(lat)`; non-circular S4a gates (analytic TC2, divergence consistency, noise stability, imprint); fallback trigger and cost in 4.5. The weather may trail the dry sphere |
| 9 | atmosphere vs shell | the shell (H 16 to 40) is **mapped onto** the model atmosphere (`H1+H2 = 80`): wind profile, interface, cloud and rain/seeding bands are evaluated at `z·(H1+H2)/H` (4.2); the confirmed W0 set is kept |
| 10 | 2D | the sphere is **3D shell only** (`world.dims = 3` required); a "surface" world is a thin shell preset |
| 11 | compatibility | `world.geometry = "box"` default; every new key in `POST_V01_KEYS`, omitted at defaults; the identity matrix stays bit-identical; **no torus-to-sphere checkpoint migration** (7.4) |
| 12 | presets | `tiny3s` (Rp 36, H 16), `mini3s` (Rp 48, H 24), `64k3s` (Rp 90, H 40); overlay `stormsph` |
| 13 | build | 8 PRs S0 to S7 (section 10); S0 can start now; the MLX compile-collision guard is a stated prerequisite of the first compiled sphere step (10.1); CPU-only until S2's kernel and S4's weather speed, then a short Metal gate each |

---

## 1. The grid

### 1.1 What is fake today

The box is a torus in x and y (`ops.wrap`, `ops.min_image`, `% G` hash seams, `roll` stencils in `fields.py` and
`weather.py`), bounded in z. The weather's Coriolis is `f(y) = f0·sin(2πy/L)` on that torus: two "equators" (y = 0 and
L/2), two "poles" (y = L/4, 3L/4) that are lines, and a flat metric. The GUI's globe (`GlobeMap.swift`, `Morph.swift`) is a
Web-Mercator wrapping of the torus with polar rows at 85.05 degrees: it draws a sphere but the world under it is a sheet
(the y seam is a pole row closed by a cap, longitude scale is 2πR·cosφ/L). None of that survives; a sphere world needs a
metric, a gravity direction, a Coriolis axis and cell tables that are those of a sphere.

### 1.2 The criteria

C1 fixed-shape gather stencils (neighbour tables) in MLX; C2 equal area, because the engine's per-cell integers (regeneration
`floor(regen_max·fert)`, `R_HARD`, pairwise-flux diffusion `q = (R·d)>>8`, the ledger's `FLOW_HARVEST`) assume equal cells and
conservation holds for any cell shape but fairness (uniform density at equilibrium) does not; C3 pole/edge singularities;
C4 position-to-cell in closed form on the device (no per-agent table search); C5 neighbour-search cost, because the
neighbour search is 75 to 80 % of a 3D step (DESIGN §27.13, `presets/64k3d.toml`); C6 diffusion/advection accuracy; C7 weather
(Coriolis, CFL, port effort); C8 rendering/overlay; C9 licence.

### 1.3 Measured comparison (probe, `grid_probe.py`, `healpix_probe.py`, `reach_probe.py`)

About 2 400 cells per tessellation (`grid_probe.py` samples points and directions; for HEALPix the **exact** coverage, the distance between the boundary of a cell and the boundary of its stencil union, was recomputed in revision 1 with the reviewer's independent implementation and is 3 to 8 % lower than the sampled values, see the table). "Coverage" is the largest radius r such that a ball of radius r around any point of a
cell lies inside the cell's stencil (self plus all cells that share an edge or a vertex), in units of the mean cell side
`sqrt(A)`; the hash cell must have coverage at least the sensing radius R. "Stencil cost" is `n_stencil·A/R_cov²` in units of
R² with `A` chosen so that coverage equals R: a perfect square 3×3 grid gives 9 (the box).

| | HEALPix (Nside 14, 2 352) | equiangular cubed sphere (N 20, 2 400) | icosahedral hex/pent Voronoi, raw (ν 15, 2 252) | lat-lon |
|---|---|---|---|---|
| area max/min | **1 exactly** (sampled 1.12 is counting noise) | **1.36** analytic (cell/mean 0.865 to 1.176) | 2.0 sampled; pentagons 0.62 of mean, hexes 0.68 to 1.22 (an optimised SCVT approaches 1.1, more work) | ∝ cos(lat): 1 000+ at the pole rows |
| edge neighbours / touching | 4 / 8 (7 for 24 cells) | 4 / 8 (7 for 24 cells) | 6 / 6 (5 for 12 cells) | 4 / 8 but pole rows have hundreds of neighbours |
| singular points | 8 trivalent vertices (z = ±2/3, φ = k·90°) | 8 trivalent cube corners | 12 pentagons | 2 poles, unusable CFL |
| position to cell | closed form: z, φ, two branches (polar cap, equatorial), integer arithmetic; round trip over all pixels verified (probe) | closed form: argmax axis, two `atan` | none: walking from the previous cell plus a coarse lookup | closed form |
| hash coverage, reach 1 | **exact 0.685 (Nside 21) to 0.711 (Nside 4)** √A, falling with Nside (sampled: 0.70 to 0.75) | 0.80 √A sampled (cost 14.1; not recomputed, expected 3 to 8 % lower) | 0.45 √A on the raw grid (cost 34.6; ≈ 18 if optimised) | n/a |
| hash coverage, reach 2 | **exact 1.385 (Nside 21), 1.406 (11), 1.421 (8), 1.466 (4)** √A (sampled: 1.45); at the presets 34.3 R³ with 5 radial layers vs the box's 27 R³ (1.27×) | not measured (expected ≈ 1.5) | n/a | n/a |
| edge-neighbour centre distance / √A | min 0.81, mean 1.02, max 1.40 (Nside 24) | similar, not measured | uniform 1 (hex) | → 0 at the poles |
| LSQ gradient conditioning (8 touching nbrs) | max 2.2, mean 1.3 (Nside 24) | not measured | ≈ 1 | → ∞ |
| weather port | collocated vector scheme (4.3) | per-face C-grid possible, but 120° corner angles and covariant metric terms (FV3 practice) | TRiSK C-grid is the proper scheme (MPAS), heavy | the classical pole problem |
| licence of references | astropy-healpix BSD-3, cds-healpix-rust Apache-2.0; **healpy GPL-2.0: not linked, not even used as oracle** | CubedSphere.jl MIT (reference only) | MPAS licence unverified, stripy LGPL | n/a |

Findings that drive the choice:

1. Equal area decides the fields. Equal quanta per cell is the equilibrium of the engine's pairwise integer diffusion; on the
   cubed sphere that is a 17 % density bias in a cube-shaped pattern, and fixing it needs per-pair area weights inside an
   int32 flux (a product of two quanta: overflow) or float fluxes (not exact). HEALPix removes the problem for the equilibrium
   and the regeneration; the diffusion arithmetic itself still needs its own int64 form (3.1).
2. The hash stencil is the cost driver and the box's 3×3×3 is hard to beat. With the exact coverage and the discrete Nsides of
   the presets, a reach-1 stencil costs `9·A·3T/R³` = 70.6 R³ at 64k3s (`Nh = 10`, A = 84.8, 6 layers of 6.67), 2.6 times the
   box's 27 R³; **reach 2** (self, touching and their touching cells, at most 25, with five radial layers of R/2) costs
   `25·19.23·5·3.077/R³` = 34.2 R³ (mini3s 34.6, tiny3s 39.3), 1.27 to 1.45 times the box. That is possible only because the
   stencil is a table; the box's stencil is arithmetic. (Revision 1 corrected the first draft's 1.8× and 1.1×, which used
   a continuous Nside and the sampled coverage.) Reach 2 is therefore the only supported stencil (one code path). Using the cubed
   sphere for the hash alone would save about 24 % more at reach 1 and perhaps 10 to 15 % at reach 2: an optional later
   optimisation (`hash_cell` is one function, 2.2).
3. The raw icosahedral hex grid is a poor hash (34.6) and has no closed-form lookup; its strength (6 equidistant neighbours,
   TRiSK) matters only for the weather, where it is the fallback.
4. Lat-lon is rejected on C3/C5/C7 alone: the cell side shrinks as cos(lat) (at 85° it is 11 times smaller than at the
   equator), the explicit-step limit follows the smallest cell, and the stencils at the pole rows are not fixed-shape.

### 1.4 Recommendation: HEALPix, RING ordering, arbitrary Nside

Definition (Gorski et al., ApJ 622, 759, 2005): 12 base rhombi refined N×N; `Npix = 12·N²`, every pixel the same area
`4π·r²/Npix`. We implement it ourselves from the published formulas (`ang2pix`, `pix2vec`, the neighbour algorithm), in float64
on the host for the tables and in float32 MLX for the device lookup; astropy-healpix (BSD-3) is a **dev-only** oracle that the
tests use when it is installed (skip otherwise); healpy (GPL-2.0) is not used anywhere. Owner confirmation of that dev
dependency is decision D10.

Facts verified by the probes: (a) for every Nside tried (4, 5, 6, 10, 12, 24) the own `ang2pix`/`pix2vec` round trip is exact
over all pixels; (b) 24 pixels (3 around each of the 8 trivalent vertices) have 7 touching neighbours, all others 8, every
pixel has exactly 4 edge neighbours; (c) the grid of Nside `2^m·N` is an exact 2^m × 2^m refinement of the grid of Nside `N` for
**arbitrary** N (N = 5, 8, 11, 12, 21, 23 tested: every coarse pixel has exactly 4^m fine pixels, no pixel straddles), which
is what the rain block mean needs without restricting Nside to powers of two (the NESTED index is not used; the parent-child
table is built from pixel centres).

**No tessellation tilt (revision 1, review M3).** The first draft tilted the grid by 60° to move the 8 trivalent vertices (z = ±2/3,
φ = k·90°, at planet latitude ±41.8° for an aligned grid) off the storm pools. The weather's real grid problem is the polar-cap
facet seams (4.1), and a tilt does not take them out of the storm belt: the share of the belt |lat| in [25°, 55°] that falls on
seam cells (cells with ix or iy in {0, N−1} of a polar facet) is 5.1 / 6.8 / 8.4 / 7.2 / 5.8 / 5.3 % for tilts 0 / 30 / 45 / 60 / 75 / 90°
(reviewer's `t7_tilt.py`, reproduced); with a tilt of 60° the grid poles sit at planet latitude ±30°, inside the belt. So `Q` is the identity,
the lookup is `ang2pix(û)`, no tilt is stored or sent, and the seams are handled in the scheme (gradient-corrected faces) and judged by an
imprint gate (4.4). Pools drift across all longitudes anyway (a circuit takes 18 000 to 36 000 steps), so no pool placement avoids
the vertices; the vertices are 8 of 21 168 cells.

### 1.5 Resolutions, hierarchy and the radial structure

Three angular resolutions, all HEALPix, none a free parameter except `world.nside`:

| name | Nside | role | rule |
|---|---|---|---|
| field `Nf` | `world.nside` | resource, chem, material, soil, hash (reach 2) | cell side ≈ `world.field_cell` (4); `world.layers = FZ` |
| hash `Nh` | derived | neighbour search | the largest Nside whose **exact** reach-2 coverage angle `cov2(Nh)·√(4π/Npix)` is at least the rigorous angle `2·asin(R/(2·r_lo))`, `r_lo = Rp + r` (computed in S0 by the boundary-to-boundary algorithm, about 3 s at Nside 21, cached; no constant). **In all three presets `Nh = Nf`**, with margins 12.0 % (tiny3s), 6.8 % (mini3s), 2.3 % (64k3s) over the rigorous angle, so one `ang2pix` serves both |
| weather `Nw` | `weather.cells` | shallow-water layers | `Nw = r·Nf` with `r` a power of two (exact for any integer ratio, review: the parent of a fine ring pixel is `(ix//r, iy//r, face)`, so the parent-child table needs no pixel centres); the rain remap is a mean of `r²` weather cells per field cell |
| territory `Nt` | game | houses, territory film, TERR record | 12·Nt² ≈ 1 024 (Nt = 9: 972 cells, the box's 32 × 32 `TERR` grid) |

**Equal-volume layers.** With `Rp` the planet radius, `H` the shell thickness, `r_top = Rp + H`:

```text
r_k³  = Rp³ + k·(r_top³ − Rp³)/FZ          k = 0..FZ, layer k is r_k ≤ r < r_{k+1}; z_k = r_k − Rp
layer(p) = clip(floor((|p|³ − Rp³)/ΔV), 0, FZ−1),   ΔV = (r_top³ − Rp³)/FZ        |p|³ = r2·sqrt(r2)
```

Every (angular cell, layer) has the volume `V = (4π/3)(r_top³ − Rp³)/(Npix·FZ)`, the same for all cells
(sizing table 7.3: 63 to 65 against the box's 64 = 4³). The layers get thinner upward (64k3s: 3.24 at the floor, 1.63 at the
lid), which is the price; the floor layer must still contain a grounded agent (rule 7.2). The index uses `r²` and one `sqrt`
(float32 error 1e-5 in r against ΔV ≈ 80 000: harmless). The vertical food profile `V[k]` of `fields.vertical_profile` is
evaluated at the layer's lower interface height `z_k` (`exp(−z_k/λ)`, as the box's `k·field_cell`), a host table, so `V[0] = 1` exactly as in the box (revision 1, L1).

**Hash layers** are equal-thickness: `G_z = floor(2H/R)`, `T_h = H/G_z ≥ R/2`, layer `clip(floor(z/T_h), 0, G_z−1)`
(thickness, not volume, is what covers a sensing ball). Hash angular cells are bigger at the lid by `(r_top/Rp)²` (2.1 to
2.25 in the presets), so a hash cell near the lid holds that many more agents at equal density: coverage is judged at
the floor (conservative), occupancy at the lid (section 8).

---

## 2. Agents

### 2.1 State and representation

A sphere world's state is `state.StateS`, a NamedTuple parallel to `State3`: `State3`'s fields in the same order, then
`EXTRA_SPHERE_FIELDS = ("hdg",)`. The box states and every digest are untouched (the same device as `State3` in §27.4).

| field | shape | meaning | invariant (tolerances in `Dims`) |
|---|---|---|---|
| `pos` | (N,3) f32 | **Cartesian position in the planet frame** | `r_lo − ε ≤ |pos| ≤ r_hi + ε` with `r_lo = Rp + z_lo`, `r_hi = Rp + z_hi` (the box's float32 bounds), ε = 4 ulp(r_top) ≈ 6e-5 |
| `vel` | (N,3) f32 | displacement during the last step (Cartesian) | as in 3D |
| `hdg` | (N,3) f32 | **heading**: unit vector tangent to the sphere at `pos` | `| |hdg| − 1 | ≤ 4e-7`, `|hdg·û| ≤ 4e-7` |
| `heading` | (N,) f32 | **derived, output only**: bearing of `hdg` from local north, `atan2(hdg·east, hdg·north)` with `+π` mapped to `−π` (the range is `[−π, π)`, review L2), `east = normalise(ẑ × û)` (x̂ when `|ẑ × û| < 1e-6`), `north = û × east`. Never read by the dynamics; written at S15; `full_check` verifies it to 1e-4 | |
| `pitch`, `buoy_cap` | (N,) | as `State3` | |
| `patch_pos` | (PP,3) | patch centres, unit vectors | norm 1 ± 4e-7 |
| `patch_vel` | (PP,3) | patch drift as an **angular-velocity vector** ω (rad/step, ω ⟂ the initial patch position): the patch rotates rigidly about ω (series, renormalised), so it stays on its small circle for ever (a stored Cartesian tangent vector stops being tangent as the patch moves: the reviewer's `t10_patch.py` parks every patch at its drift pole after ~9 000 steps) | ‖ω‖ constant |
| `env_cat_center` | (2,) | catastrophe cap centre `(lon, lat)` in radians | lat ∈ [−π/2, π/2] |
| `resource` | (FZ, Npix) i32 | `Npix = 12·Nf²`, flat cell id `k·Npix + pix` | as the box |
| `chem` | (NCH, FZ, Npix) f32 | | |
| `wx` | (N_WX, 12·Nw²) f32 | `N_WX = 11`: h1, h2, the two layers' velocity as tangent 3-vectors (6 planes), Q, P, Dc | |
| `mat`, `soil` | (FZ, Npix) | conditional extension fields (§43) | |

Dead rows hold `pos = (0, 0, r_lo)`, `hdg = (1, 0, 0)`, `vel = 0`, `pitch = 0`.

*Rejected: unit vector + scalar height as the state.* It makes the radial clamp and `grounded` exact (a scalar `z`), but every
reader of `s.pos` (tracking, metrics, dumps, stream, featobs, the game) would receive a unit vector. The Cartesian state costs
a tolerance of ε = 6e-5 on the radial invariants, which is 100 times below the smallest `floor_eps` allowed (rule 7.2:
`floor_eps ≥ 0.01`), and keeps `pos` a world position everywhere. *Rejected: a scalar yaw in a local east/north frame:* the
frame is singular at the two poles (the hairy-ball theorem), the angle then jumps for an agent passing near a pole; a vector
has no singularity.

### 2.2 Position to cell (device functions, `alife/sphere.py`)

```text
r2 = pos·pos ;  r = sqrt(r2) ;  û = pos / r ;  z = r − Rp                  (elementwise; no matmul)
pix  = ang2pix(û, Nside)         z_c = û_z ;  s2 = 1 − z_c² computed as (x²+y²)/r2 (stable at the poles, no 1−|z|)
                                 φ = atan2(y, x) mod 2π ; tt = φ/(π/2) ; equatorial branch |z_c| ≤ 2/3 and polar-cap branch
                                 both evaluated, mx.where selects ; integer ring arithmetic in int32 ; ~60 elementwise ops
field cell  fcell = layer(pos)·Npix_f + pix_f                       (z-major, as the box's flat index)
floor cell  fcell mod Npix_f                                        (corpses sink here, §27.6)
hash cell   cid = clip(floor(z/T_h), 0, G_z−1)·Npix_h + pix_h       (dead rows: the sentinel Npix_h·G_z)
```

`ang2pix` near a cell boundary can place a point one cell off because of float32 rounding; both uses are insensitive (the hash
stencil covers neighbours with margin, and a field cell lookup is a bin). The polar and equatorial branches meet continuously.
`hash_cell` and `field_cell` are separate functions so the hash can later move to another tessellation.

### 2.3 Motion

Per step, per survivor, after the force `F` (N,3) and `γ`:

```text
v   = F_t/γ_t + (F·û)û/γ_r  with  F_t = F − (F·û)û ;  γ_t = drag·(1 + grip_gain·grip·[grounded]) , γ_r = drag    (grip tangent only)
v   ← v·min(1, v_max/|v|)                    3D norm clip, each clip counted (V_CLIPS), as integrate3
vr  = v·û ;  vt = v − vr·û ;  w = vt/(Rp + z) ;  σ² = w·w ;  c = 1 − σ²/2 + σ⁴/24 ;  s = 1 − σ²/6 + σ⁴/120     (|σ| ≤ 0.5/45 = 0.011)
û'  = normalise(c·û + s·w)                   exact exponential map to rounding (series error σ⁶/720 ≈ 1e-15)
r'  = |pos| + vr ;  at_floor = r' ≤ r_lo & vr < 0 ;  at_lid = r' ≥ r_hi & vr > 0 ;  r' = clip(r', r_lo, r_hi)
pos' = r'·û'
k   = s·(û × w) ;  κ = c ;  hdg' = hdg + k×hdg + k×(k×hdg)/(1 + κ)       minimal rotation û→û' = exact parallel transport along the great circle
                                             (k and κ come from the construction, û×û' = s·(û×w) and û·û' = c: no cancellation of two nearly equal unit vectors, review L3)
hdg'' = cos(turn)·hdg' + sin(turn)·(û' × hdg')                          yaw turn about the new up
hdg'' ← normalise(hdg'' − (hdg''·û')û')                                 renormalisation: mandatory
vel' = v with vr set to 0 on at_floor / at_lid rows
```

`vel` is the commanded step `v` (tangent plus radial parts at the old point), not the chord between the old and the new position: that chord has an extra radial
component `−|vt|²/(2(Rp+z))`, up to 0.003 per step at the largest step on the smallest planet. The box has no such term; `IN_VEL` and the observers read `vel`, so
the convention is the box's (review L6).

No branch depends on array values; the series replaces `sin`/`cos`/division by `|vt|`, so `vt = 0` is exact. `1 + κ ≥ 2 − 1e-4`:
no antipodal singularity. Dead rows keep `pos`, `hdg`, `pitch`.

Probe (`transport_probe.py`, 400 agents, 3·10^5 steps, steps up to 0.5, 10 % of the agents started within 0.06 rad of a pole,
radius 45 (the smallest planet), float32 against a float64 run on the same inputs):

| | `| |û| − 1 |` max | `|hdg·û|` max | `| |hdg| − 1 |` max | position vs float64, max | heading vs float64, max |
|---|---|---|---|---|---|
| float32 with renormalisation | 9.7e-8 | 7.0e-8 | 1.0e-7 | 0.068 units | 1.7e-3 rad |
| float32 without | 7.0e-4 | 3.4e-5 | **0.114** | 1.7 units (median 0.18) | 0.48 rad |

The position error is a random walk of rounding (the body radius is 1); renormalisation is not optional. **Holonomy test**
(walk the octant triangle pole → equator → equator → pole in 0.05-unit steps): the heading ends turned by π/2 to 1.8e-15
(float64) and 7e-7 (float32), the spherical excess. Both go into `tests/test_sphere_motion.py`.

**Frame.** `fwd = cosφ·hdg + sinφ·û`, `lat = û × hdg`, `up = cosφ·û − sinφ·hdg`: the box's frame of §27.2 with (`hdg`, `û`) in
place of (cos θ, sin θ; ẑ). `ops.body_axes_s` and `ops.to_body_s(v, hdg, û, cosφ, sinφ)` return/project on it (three dot
products); at φ = 0 on a locally level scene it reproduces `to_body3` (flat-limit test below).

**Forces.** `physics.forces_s` is `forces3` with `fwd` from the frame and gravity/buoyancy along `û`:
`F = thrust·att·fwd + springs + contacts + (buoyancy_max·buoy_cap·b − gravity)·û (+ wind)`. Springs and contacts are the box
expressions on the (N,K,3) and (N,B,3) chord offsets (section 2.5).

**Gravity** is constant in magnitude and radial. (At `r_top/Rp = 1.5` an inverse-square law would lower it to 0.44 of its floor
value in the lid; the box has none; D5.) **Grounded:** `z ≤ z_ground`, as `physics.grounded`, with
`z = |pos| − Rp` (float32 `r − Rp`, error 2e-5): validation requires `floor_eps ≥ 0.01`.

**The atmosphere top (the owner dislikes organisms drifting to outer space).** A hard lid at `r_hi = Rp + H − r` is the box's
surface clamp (Q2 of §27.0) made radial: the outward velocity is zeroed (`Z_SURFACE_CLAMPS` counts them). Optional soft band
`physics.ceiling_band` (units, default 0 = off): over the top `ceiling_band` of the shell the *outward* component of thrust and
buoyancy is multiplied by `1 − smoothstep` of the depth into the band, so the pile-up against the lid seen in 64k3d
("thin single-clade sheets pressed against the surface or the floor", RESULTS_3D) softens. Both are a static branch. The lid
is the visible atmosphere in the GUI (section 5).

### 2.4 Neighbour search

`spatial.find_neighbors` dispatches to `_find_neighbors_sph` on a sphere. Same structure as `_find_neighbors_3d`: cell id,
`argsort`, `searchsorted` run starts, a `(N, Q)` candidate gather. The difference is the stencil:

```text
S_ang ≤ 25 angular cells (reach 2): table nbr[Npix_h, 25] int32, entry = the cell itself first, then the touching cells and
                                    their touching cells (24 at most; fewer next to the 8 trivalent vertices), padded with −1
S_rad = 5 radial layers of thickness T_h ≥ R/2   a layer outside [0, G_z) has an empty run, as in §27.2
Q = 125·C
ncid = (clip(kh + dk, 0, G_z−1))·Npix_h + nbr[pix_h, a]  ;  run [s, e) = [start[ncid], start[ncid+1]) ; empty if nbr = −1 or the layer is out of range
d    = pos_j − pos_i  (plain difference: min_image(·, size) is the exact identity, 2.5) ; d2 = d·d ; strict d2 < R² ; K smallest
```

Selection, ties, `overflow_cells`, `max_occupancy`, `hidden_agents` are §4's, over `Npix_h·G_z` cells. Equal to brute-force kNN
whenever no cell holds more than C (F13), at every pole, trivalent vertex and layer edge (the test enumerates them).
**Reach 2 only (revision 1, review M7 and D7):** the first draft left the stencil to a benchmark; with the exact coverage the discrete
search volume is 34.3 R³ at 64k3s (1.27× the box) against 70.6 R³ (2.6×) for reach 1, so there is one code path and no `sensing.hash_reach` key.
If the S2 benchmark shows the search costing more than 2× mini3d per alive agent, S2 stops and escalates to the owner (it does not fall back to
reach 1 silently); the contingencies are the cubed-sphere hash and a shallower shell.

**`C` is derived, not guessed:** the largest hash cell is the lid cell, volume `V_lid = (Ω_cell·r_top²)·T_h`; the repo's crowd tests reach 0.167
agents/unit³ (36 in a 216 cell). `validate` requires `C ≥ ceil(0.167·V_lid) + 4`: tiny3s 142 units³ → 28, mini3s 135 → 27, 64k3s 124 → 25, so
the presets use `C = 32` (provisional; S2/S7 measure the peak occupancy as the box's `C = 24 → 48` was).

**Metal kernel.** `knn_metal.scan_select` gets a sphere mode: the arithmetic stencil (`py_mod`, `_DX27`) is replaced by the
`nbr` table read (`nbr[pix·S + t]`, `−1` skips), the minimum image by a plain difference, nothing else changes (one thread per
sorted rank, `fp contract(off)`, stable insertion by `(d², q)`, `precise::sqrt`, K ≤ 64). It must return the pure-MLX search's
arrays bit for bit (`knn_metal.self_check` gets a sphere scene; `VERIFIED_MLX` unchanged). The pure-MLX path is the reference
and the CPU path; on a GPU it is the fallback only (at 64k3s `Q = 125·32 = 4 000`: `(N, Q)` arrays three times the size of 64k3d's (Q = 1 296), 1 GB
each, 3 to 6 GB in all; documented, and why the kernel is a prerequisite of any 64k3s run).

### 2.5 Pair geometry: chord distance, and why `world.size` becomes an image length

Bodies are balls in R³ and every pair interaction (sensing radius, contact `dist < 2r`, repulsion `k_repel(2r − d)`, springs with
rest length, combat range, hearing range) is a function of the Euclidean offset between centres, which on the sphere is the
chord. The chord is shorter than the arc by `δ²/24` of its length: at the sensing radius 6 on the smallest planet (Rp 36) that
is 1.2e-3 relative, and 1.7e-4 at the bond rest length 3 on Rp 48. It is the physically right quantity, not an
approximation to be corrected. Consequently **no pair expression changes**: `ops.min_image(d, L)` is `d − L·rint(d/L)`; with
`world.size = L > 4(Rp + H)` every offset has `|d| ≤ 2(Rp+H) < L/2`, `rint(d/L) = ±0` exactly, and the result is bitwise `d`.
`bonds._partner_offsets`, `featobs`, `energy.combat`, `spatial` and the Metal kernel keep their code; the sphere sets
`world.size` (default 1024, so a preset need not state it) and `config.validate` enforces `size > 4(Rp+H)`. A test checks the identity over all neighbour and partner offsets of a random shell scene **by value** (the only bit difference the reviewer found is `−0.0 → +0.0`, review L8; compare with `==` or add 0.0). The identity holds for **pair offsets only**: any other use of `min_image`/`wrap` on a sphere is a bug, and 2.9 is the mechanism that finds it. `world.size` is *not* a length of the sphere elsewhere: `Dims` is
computed per geometry, `validate` skips the box-derived rules (`G_hash ≥ 3`, `FG = size/field_cell`, `FZ = height/field_cell`, `2·height ≤ size`) for a sphere,
and the geometry lint (2.9) bans `cfg.world.size` outside pair code. The default 1 024 passes `size > 4(Rp+H)` up to `r_top = 255`; a larger planet must set `size`
(review L5: no geometry-dependent default).

### 2.6 Sensing, the five feature channels, bonds

| item | sphere form |
|---|---|
| descriptors (`sensing.neighbor_descriptors`) | the 8-wide descriptor of §27.2 with (fwd, lat, up) from `to_body_s`; clipping, validity, kin, message columns unchanged |
| `IN_VEL`, `IN_VEL_UP` | `to_body_s(vel)` at the start-of-step frame |
| `IN_RES_GRAD`, `IN_CHEM_GRAD` (+ `_UP`) | tangent least-squares gradient (below) projected on (fwd, lat), radial difference over the true layer spacing on `up`; scale kept: `g = ∇R·field_cell/resource_cap`, so a box and a sphere world feed the brain the same magnitudes |
| `IN_HEIGHT` | `z/H`; `IN_PITCH`, `IN_GROUNDED` as in 3D. **`D_IN = 84`, `D_OUT = 39`, genome layout version 2: unchanged**, so a sphere world has the 3D genome layout (and a brain is not tied to a geometry) |
| field gradient | `g_t = Σ_k W[pix,k]·(R_k − R_c)` over the 4 edge neighbours of the agent's cell in its layer (table `W` (Npix,4,3), tangent vectors, float64 least squares on the host, condition ≤ 2.2 on the 8-neighbour version, probe); radial central difference over `z_{k+1}−z_{k−1}` mirrored at floor and lid as in §27.6 |
| bonds, partners, springs | unchanged code; rest length and break length are chords |
| combat, comm hearing (`comm.hear`), arms, roles | neighbour-based, unchanged; `bearing` is in the body frame already |
| catastrophe | a cap: all agents with `û·c > cos(radius/Rp)`, `c(lon, lat)` from `env_cat_center`; the box's column becomes the cap's radial column (all heights), the same semantics |
| structures `nopen`, R2 aperture | neighbourhood = the 8 touching cells × 3 layers (26; 23 at the 24 seven-neighbour cells; 17 on floor/lid layer); the colouring that keeps closing cells ≥ 3 apart is a precomputed greedy colouring of the distance-2 touching graph, `colour[pix]` (greedy gives **15, 16, 18 colours** at Nside 8, 11, 21 with two orderings, conflict degree 24, lower bound 9: the first draft's "≤ 13" was wrong; S3 minimises and `n_phase` takes the count as data: 45 to 54 phases against the box's 27 to 48, so closing is slower in proportion) times `layer mod 3`, replacing `cx mod px + px·(cy mod px)`; `k = ceil(aperture·n_nbr)` per cell from the table |
| farming (soil, §41) | a cell array like `mat`; stencils through the same tables |

### 2.7 Birth, founders, immigrants

* **Child placement** (`lifecycle._offspring_kinematics_sph`): same draws as §27.4 (`Stream.PLACEMENT` `(M,2)`). With `t̂` the parent's
  `−hdg` rotated about `û` by the yaw jitter `u₀`, tangent arc `D·cosχ`, `χ = clip(u₁ − φ_p, ±PITCH_MAX)` (reflected at floor and
  lid as in 3D): `û_c` by the exponential map of `D·cosχ·t̂`, `z_c = z_p + D·sinχ`, `hdg_c = transport(t̂)` (the child faces away
  from its parent). Offset length D to rounding.
* **Founders and immigrants** uniform in the shell *by volume*, with the **same draw shape** as the box, `(n, 3)` uniforms
  `(a, b, c)`: `cosθ = 2a − 1`, `λ = 2πb`, `r³ = r_lo³ + c·(r_hi³ − r_lo³)` (exact, no rejection, equal-area by Archimedes), and
  the heading by the existing `(n, 2)` draw: yaw uniform in the tangent plane at the point, pitch uniform. No new RNG stream, no
  `RNG_LAYOUT_VERSION` bump.
* Fertility patches: `(PP, 2)` uniform draws become a point by the same `(cosθ, λ)` map; the drift direction angle ψ is taken in
  the local east/north frame once at init; the patch's drift is stored as the angular-velocity vector `ω = (û_p × t̂_ψ)·patch_drift/Rp` and applied as a rigid
  rotation about ω each step (the director's `drift_mult` scales ω). Warm pools drift in longitude only (`vy = 0` is a validation rule, 4.1).

### 2.8 Exactness

No ledger term is added and none changes. Every energy path is integer and cell-wise: harvest from the agent's own
(layer, pix) cell, combat spill and residue to the victim's cell, corpses to the floor cell of the column, rain into columns
(`INJECT_REGEN`), `R_HARD` overflow to `HEAT_OVERFLOW`, buoyancy in slot 15. The 16-term ledger closes by the same argument as
§27.7; the closure test of §22 runs unchanged on a sphere fixture. Geometry enters only through float32 positions and the
integer cell index, i.e. through which cell an agent is in, never through an energy amount. Column occupancy for `INT_HEADROOM`
is the largest angular-cell count (`scatter_add_i32(Npix_f, fcell mod Npix_f, 1, alive).max()`); at 64k3s the densest column
holds about 136 (FCC) against `occ_limit` 1 060.

### 2.9 The consumer guard: geometry accessors, a lint rule, a rotation test, readiness gates (review H4)

The Cartesian state makes `pos[:, 2]` (the Cartesian z, not the height), `vel[:, 2]` (not the radial speed), `pos[:, :2]`, `ops.wrap*` and `ops.min_image`
(the identity on a sphere) keep type-checking and return plausible numbers in every consumer outside the pair code. A grep of `src/alife` finds
the raw z/vel-z pattern in about 25 files, `min_image`/`wrap` in 20, `world.size` in 28, `world.height` in 24 and the lattice shape (`field_cell`, `fg`, `fz`, `g_hash`)
in 40. Four mechanisms, none depending on remembering:

1. **Geometry accessors** (S0). `alife.geom.geometry(cfg)` (host, numpy) and `alife.sphere.geometry(cfg)` (device, MLX) return a static object
   (`Box` or `Sphere`, a trace-time constant) with `height(pos)`, `up(pos)`, `radial(vec, pos)`, `tangent(vec, pos)`, `displacement(a, b)` (the box's `min_image`; a plain
   difference on a sphere), `distance(a, b)` (chord) and `arc(a, b)`, `cell_of(pos)`, `centroid(points, weights)` (host), `bearing(...)`. The `Box` methods are the current expressions
   written once (`pos[:, 2]` and so on), so converting a box consumer to an accessor is bit-identical (the identity matrix proves it) and costs nothing.
2. **Lint** (S0, in `tests/test_lint.py` beside the scatter rules). A scan of `src/alife` flags, outside `tests/geometry_allowlist.py`, the patterns
   `pos|vel` (and names ending in `pos`/`vel`) indexed `[..., 2]`, `[:, :2]` or `[..., :2]`; `ops.wrap`, `ops.wrap_xy_clamp_z`, `ops.min_image`; `world.size`, `world.height`,
   `world.field_cell`, `dims(...).fg/.fz/.g_hash/.cell_hash`. Each allowlist entry names the file, the pattern, a count (a **ratchet**: a larger count fails), the reason
   (`pair`, `box-only tool`, `definition`, `converted in S<k>`) and, for `converted in S<k>`, the PR that must delete it: the test fails as soon as the current PR number reaches `k`
   with the entry still present. New code cannot add a raw use without a reviewer-visible allowlist line.
3. **Rotation-equivariance test** (S2, extended by every PR that converts a consumer). Every consumer that does not depend on the planet axis (metrics, observers, labels, tracers,
   tracking, society, report, storms tracker, analysis) is run on a sphere scene and on the same scene rotated by a fixed random rotation `Rot` (positions, headings, velocities,
   cell tables permuted accordingly); results must agree up to float32 rounding. A raw `pos[:, 2]` or `min_image` breaks it at once, whatever the consumer.
4. **Readiness gates** (S2). `config.SPHERE_READY` (a frozenset read at call time, like `DIMS_SUPPORTED`) lists what a sphere run may enable; `validate` refuses the rest with the PR
   that will enable it: `run.observe`/`observer.*` and `game.*` (S5), `director.enabled` (S5), `weather.enabled` (S4), `structures`, `niche`, `comm`, `bonds`, `arms` (S3), the E-series
   emergence tools (declared box-only). Nothing runs through an unconverted consumer.

**Audit: every consumer and the PR that converts it** (counts from `grep` of main 7ab28cf; "z/vz" = raw `[:, 2]` on pos/vel, "wrap" = `min_image`/`wrap*`):

| consumer | what it reads | PR |
|---|---|---|
| `spatial.py`, `knn_metal.py` | box search (z, wrap, size) | S2: the sphere search is a separate function; the box lines are allowlisted (`box branch`) |
| `physics.py`, `sensing.py`, `lifecycle.py` (z 2, wrap 7, size 3), `fields.py` (z 3, wrap 4), `state.py` (wrap 4), `invariants.py` (z 2, size 3), `energy.py`, `bonds.py` (pair wrap 3) | dynamics, placement, catastrophe (`lifecycle.py:95` uses `min_image` on the column distance), invariants, cell index | S1 (`fields`, `state`) and S2 (the rest); `bonds`/`energy` pair code allowlisted `pair` |
| `weather.py` (z 3, wrap 6, size 6, height 13), `weather_obs.py`, `weather_spinup.py` | wind sampling, vertical profile, remap, params | S4 |
| `bench.py` (size 2, height 1, scene builder) | step benchmark scenes | S2 (a sphere scene builder; the S2 gate needs it) |
| `ext/structures.py`, `ext/niche.py`, `structures.py`, `niche.py` | lattice shape, neighbourhood, colouring | S3 |
| `metrics.py` (z 1), `featobs.py` (12), `groups.py`, `tracking.py` (43 `circular_mean`/size uses), `streamrec.py`, `stream.py`, `play.py` (z 1, size 3), `cli.py` (`inspect`, `replay`, `runs`), `runio.py` | observers, stream, runs listing, metadata | S5 |
| `game/observe.py` (the G1 observer's quantised z), `game/observer.py` (z 1, size 2), `game/tracers.py` (z 11, wrap 3: vz_own, w_air, tracer_z), `game/society.py` (z 2, size 3), `game/powers.py` (z 2, size 3), `game/social.py` (size 2), `game/alerts.py`, `game/report.py`, `game/kinthreads.py`, `game/housefile.py` | labels, territory, houses, powers, alerts | S5 |
| `analysis/labels.py` (z 2, wrap 6, `INTENT_VERSION` thresholds calibrated on a torus; the Nomad "unwrapped track" `u_k = u_{k−1} + min_image(Δpos)` telescopes to `pos_k − pos_0` and saturates at the planet size: needs the sum of great-circle arcs) | G1 labels | S5 (thresholds re-derived in S7: `INTENT_VERSION` 4, provisional) |
| `storms.py` (wrap 9, size), `analysis/w0.py` (z 8, wrap 18, size 4, height 9), `analysis/w0_plots.py`, `analysis/w2.py` | storm detection and the W0 study driver | S5 (tracker) and S4 (study driver) |
| `director/actions.py` (wrap 2, size 8), `director/yoked.py`, `director/llm.py`, `director/prompt.py` (height 2), `director/__init__.py` | levers and prompts in x,y | S5 |
| `render.py` (wrap 5), `viewer.py` (wrap 1), `analysis/signals.py` | 2D-only tools; already refuse dims 3 via `require_2d` (covers the sphere) | none (`box-only tool`, allowlisted) |
| `analysis/emergence/*` (z 5, wrap 17, height 15) | the E-series measures | declared box-only; a sphere variant is out of scope, `validate` refuses the emergence recorder on a sphere |
| `migrate.py` | checkpoint migration | S2: refuses a sphere (no migration) |

---

## 3. Fields, material, soil

### 3.1 Resource and chemicals (rewritten in revision 1, review H5; weights and bounds recomputed with `review_probes/t3_cond.py`, reproduced)

* **Regeneration**: `floor(regen_max·fertility[layer, pix])`, `fertility = clip(base + Σ_p w_p·exp(−d²_{p,pix}/2σ²), 0, 1)·V[k]`,
  where `d²` is the chord² between the patch centre and the cell centre in length units, `Rp²·(2 − 2û_c·û_p)`, elementwise `(PP, Npix)` (three multiplies, no
  matmul, F20); the pools of 4.1 use the same chord convention (review L7). Seasons, per-patch gain, director env: unchanged.
* **Resource diffusion.** The first draft wrote `(R·d_q8·w) >> 16` in int32. `R` reaches `R_HARD = 2²⁴ − 1` (one corpse deposits up to `e_max = 10⁶` into a floor
  cell, DESIGN §10.9) and `d_q8·w` is 1 600 to 9 100, so the product wraps for `R` above 0.24 M (a fresh corpse on the floor of tiny3s: radial weight 846,
  `3.4·10⁹` at `R = 10⁶`). The box's `R·d_q8 ≤ 2²⁴·42 = 7·10⁸` fits by design (§27.6). Revision 1:

  ```text
  a_ij = d_q8·w_ij                  integer, w_ij the symmetric pair weight in 1/256 units (a host table: w[pix,k] tangential, wz[k] per radial interface)
  q_ij = (int64(R_i)·a_ij) >> 16    int64 product (≤ 2²⁴·2¹⁷ = 2⁴¹; MLX int64 is available on the GPU, groups.splitmix64 already uses uint64), cast to int32
  R_i ← R_i − Σ_j q_ij + Σ_j q_ji      every q is a pairwise antisymmetric transfer, so the total is conserved exactly
  ```

  With `w = 256` everywhere this is the box's `(R·d_q8) >> 8` exactly (a test). Proven bounds: `R_i' ≥ 0` because `Σ_j q_ij ≤ R_i·Σ_j α_ij ≤ R_i` where
  `α_ij = a_ij/2¹⁶` and `Σ_j α_ij ≤ 0.9` is a validated rule (so `d_q8 ≤ floor(0.9·65536/rowsum_max)`: **11, 11, 14** for tiny3s, mini3s, 64k3s with row sums of 5 233,
  5 329 and 4 189 in 1/256 units against the box's 1 536; the default `d_q8 = 4` gives `Σα ≤ 0.33`). `R_i' ≤ R_HARD` when every `R ≤ R_HARD`: inflow from a neighbour is at
  most `floor(R_HARD·α_ij)`, hence `R_i' ≤ R_HARD − (1 − Σα)(R_HARD − R_i) + 6` and the bound holds for `R_HARD − R_i ≥ 6/(1 − Σα) = 60`; below that the tables are checked by exhaustive
  enumeration of `R_i ∈ [R_HARD − 60, R_HARD]` against the all-`R_HARD` neighbourhood (the worst case, floors are monotone) for **every distinct row of weights** by a host function
  `ops.check_diffusion_tables` called from `validate`; the reviewer-style counterexample (a row with `Σα > 1`) is the negative test. My enumeration over the realistic rows and 2 000
  random rows with `Σα ≤ 0.95` found an excess of 0 everywhere. Corner tests in S1: a floor cell at `R_HARD` with empty neighbours; one at `10⁶` in tiny3s's
  floor (the case that wrapped); `R_HARD` everywhere (uniform, nothing moves).
* **What the weights mean (an honest change of the box claim).** The default weights are the *geometric conductances* of a finite-volume cell (tangential
  `edge length·thickness/centre distance`, radial `face area/layer spacing`, each divided by the cell volume), normalised so the floor-layer **mean tangential** weight is 256:
  `resource_diffusion_q8` is the box's physical diffusivity at the floor, `D = d_q8/256·field_cell²` per step, isotropic in physical units. The cells are thinner than wide,
  so the **vertical exchange rate per step is larger than the horizontal one**: radial weights 510 to 2 269 (floor to lid) against tangential 149 to 437 (floor) and
  123 to 136 (mean at the lid); at the floor about 2×, at the lid up to 9×. The box has cubes (1:1). Consequence, stated: vertical contrast in the resource
  (the benthic/photic profile the 3D presets were tuned around) is erased 2 to 9 times faster per step than in the box. If S7 finds the profile washed out, the owner's
  choices are (i) a vertical weight multiplier `fields.vdiff` (default 1; the weights are a table, so this is free), or (ii) normalising by the maximum row sum, which keeps
  `d_q8 ≤ 42` meaning but slows the horizontal diffusion 2.7 to 3.5 times. The first draft's "every integer rule carries over unchanged" is withdrawn.
* **Chemicals.** `lap_i = Σ_j C_ij (c_j − c_i)` with `C_ij = w_ij/256` (the same tables, mirrored ghost layers) is a float update, conserving mass to rounding at `λ = 0`. The explicit
  stability limit `chem_diffusion·max_i Σ_j C_ij ≤ 1` fails with the box's default `[0.16, 0.10, 0.16]` (row sums 16.4 to 20.8 against 6, so `0.16·20.8 = 3.3`): the lid layers are
  1.4 to 1.6 thick. The update is therefore **sub-stepped statically**, `n_sub = ceil(D_c·max_i Σ_j C_ij)` sub-steps of `D_c/n_sub` (4 for the default; a host constant of
  the config), the decay `λ_c` applied once. Cost about +0.3 ms at 64k3s (estimate).
* **Deposit/harvest/corpse/column occupancy**: `ops.scatter_add_i32` on the flat (layer, pix) index; no new scatter outside `ops.py`.
* **Carrying capacity** (`c_ref`, metrics' `K_est`): sums over cells are unchanged because cells are equal (the equal-volume property); the layer weights use `V[k]`.
* **R9 gate restated (review M4).** The reviewer's spectrum of the geometric 4-neighbour Laplacian (`t3b_lap.py`, reproduced at Nside 8 and 12): multiplet *means*
  are right (l = 1..4: −0.6 to +0.4 %, l = 6 within 3.3 %), individual modes spread **−17 % to +23 %** within a multiplet (l = 2: −11/+23 %, l = 3: −17/+20 %), because the edge-neighbour
  `L/D` ratio ranges 0.63 to 1.69. The S1 gate is therefore: multiplet mean within 3 % of `l(l+1)` for l = 1..6 and spread within ±25 % (the grid anisotropy is a real, documented effect: diffusion varies by about ±20 % with
  direction relative to the grid), identical for the sub-stepped chemical update.

### 3.2 Material, soil, voice, bonds, arms

`mat` and `soil` are `(FZ, Npix)` int32; deposit, decay and the effects (`structures.sense`, `mat_table`, shelter, barrier, shield)
are cell-wise with the tables above. The `STRC`/`FSOL` records use `nx = Npix, ny = 1, nz = FZ`, which keeps the flat index
`k·Npix + pix` of the box's `CELLS_PREFIX`. Comm, bonds that pay, arms and roles have no lattice and are neighbour-based:
they are unchanged and each gets its sphere fixture in S3.

---

## 4. Weather (rewritten in revision 1: review H1, H2, H3, M8; every number in 4.3 and 4.4 is from `rev1/wproto.py` and `rev1/rh4.py`, float64 numpy, CPU)

### 4.1 What carries over, and the scheme

The model (`weather.py`, STORMS_W0): two stacked shallow-water layers with moisture, `H1 = 32, H2 = 48`, semi-Lagrangian momentum advection, flux-form
continuity with a positivity limiter, forward-backward pressure, Coriolis, evaporation over warm pools, condensation between the layers, radiative return, damping,
seeded disturbances, `sample_wind`, rain remap, the W2 couplings. State per cell: `h1, h2, Q, P, Dc` and the two layers' velocities as tangent 3-vectors (`N_WX = 11`).

**Host tables** (float64, built once, cast): per cell the 4 edge neighbours `nb4`; per edge the arc length `ℓ`, the edge midpoint `x_f`, the **true edge normal** `n` (the tangent
at `x_f`, perpendicular to the edge, pointing out of the cell: **not** the centre-to-centre direction, which makes the divergence error grow like 1/h, see 4.3), the unit
tangent `e` at `x_f` of the geodesic joining the two centres, the centre distance `d`, and `δ = x_f − x_mid` with `x_mid` the normalised midpoint of the two centres (the edge midpoint is
up to 0.42 √A off it at the polar-cap facet seams, 0.03 inside the cap, 0.008 on the equatorial facets); per cell the 8 touching neighbours with least-squares weights `W8` (tangent 3-vectors,
`1/d²`-weighted) and the fan-triangle inverses `T[c,k]` (7 or 8 triangles, 2×2).

**One step** (single layer shown; the second layer and the moisture use the same face values; unit-sphere quantities times `R`):

```text
G m   = Σ_k W8[k]·(m_k − m_c)/R                    pressure-potential gradient, m = g_top(h1+h2) + g'·h1 in layer 1, g_top(h1+h2) in layer 2 (weather._pressure_kick)
J u   = Σ_k (u_k − u_c) ⊗ W8[k]/R                  velocity gradient (3×3), likewise ∇h
S1  u* = proj_c( fan( u, normalise(x_c − dt·u/R) ) )          semi-Lagrangian on Cartesian components, re-projected on the tangent plane at x_c (error ~1e-6 per step)
S2  face velocity  u_f = ½(u_c+u_q) + ½(J_c+J_q)·δ·R          gradient-corrected (review H1)
    face depth     h_f = ½(h_c+h_q) + ½(∇h_c+∇h_q)·δ·R
    face normal    u_n = u_f·n  −  dt·(e·n)·[ (m_q − m_c)/d  −  ½(G_c+G_q)·e ]      Rhie-Chow: the compact pressure difference replaces the wide gradient (review H2)
    h  ← h − dt/A · Σ_k ℓ_k h_f u_n                   flux form: antisymmetric per edge, mass conserved to rounding; the positivity limiter of weather._continuity is applied to
                                                      the outflow of each cell exactly as in the box; Q is advected with the same u_n and the upwind q = Q/h
S3  u ← R(−f dt/2) u* ;  u ← u − dt·G m(h_new) ;  u ← R(−f dt/2) u ;  re-project        R(θ): rotation about û by θ, f = f0·(ẑ_planet·û): exact, norm preserving, no pair-rotation tables
S4  h ← h + a4·B h ;  u ← u + a4·B u ;  re-project           B = −L², L the 4-neighbour Laplacian with symmetric weights ℓ/d (normalised to mean 1); a4 = 1e-3 (ENGINE_FIXED)
    condensation, exchange, evaporation, radiative return, damping r1, c_d, r2: pointwise, unchanged
```

`divergence`, `vorticity` for the diagnostics, the interface height and the tracker are `∇·u = Σ_k W8·(u_k − u_c)` and `ζ = û·Σ_k W8 × (u_k − u_c)` (surface divergence and normal curl of a
tangent field are exactly the tangential differences); the seed blobs give `u = û × ∇ψ` in closed form; pools are `(x, y, sx, sy, amp, vx, vy)` read as arc lengths east and
north of the origin on the equator, Gaussian in the tangent-plane components (east, north at the pool centre) of the chord vector to the cell centre (the same chord convention as the
patches, review L7), **`vy = 0` required** (a pool with `vy ≠ 0` leaves the validated band and `dlon = vx/(Rp·cos lat)` diverges at the pole, review M2), centres within |lat| ≤ 60°, drift
in longitude at the pool's own latitude; the rain remap and the seeding expand are means and repeats over the `r²` weather cells of a field cell (the parent-child table).
The wind sampled at an agent is the gathered tangent vector re-projected onto the agent's tangent plane (review L4: the cell vector is tangent at the cell centre, up to 2 % of |u| off).

**The fan interpolation** replaces bilinear. In the cell `c` of the departure point, the point's barycentric coordinates in the triangles (c, nb_k, nb_{k+1}) around the centre are
computed for all 7 or 8 triangles and the one with the largest minimum coordinate wins (fixed shape). "Cell value + own LSQ gradient" blows up (forward Euler in time) and
constant-per-cell stalls (probe `sl_probe.py`). Williamson test 1 after one revolution, peak retained: fan 0.153 / 0.265 at Nside 24 / 48, the box's bilinear on a plane of the same cell size and
Courant number 0.066 / 0.132: the sphere's advection is **2 to 2.3 times less diffusive** than the box's, so storm statistics will shift; the damping keys are the S4c calibration levers.

### 4.2 The vertical map: shell onto atmosphere (review H3; D4 changed)

The first draft relaxed `H1 + H2 = world.height` to `≥` and left the vertical couplings alone. They break: `sample_wind` takes the interface at `zi = height·h1/(h1+h2)` (≈ 32 at rest for
`H1 + H2 = 80`), `cloud_weights` overlaps the cloud `[zi, zi + dc]` with the layers, and `rain_regen`, `seeding_response`/`seeding_qs` use those weights. For `tiny3s` and `mini3s` (H = 16, 24) **no layer
overlaps [32, 32 + dc]**: `frac.sum = 0`, zero rain food and no seeding, a storm world without food. At `64k3s` (H = 40) all the rain falls in the top 5 of 18 layers.

**Decision: map the shell onto the atmosphere.** With `s = (H1 + H2)/H` (5 for tiny3s, 3.33 for mini3s, 2 for 64k3s with `H1 + H2 = 80`):

| quantity | weather unit → shell unit |
|---|---|
| height of an agent in the wind formula (shear layer `φ(z) = min(z_w/z_sl, 1)`, the interface test `z < zi`) | `z_w = s·z`; `zi_shell = zi/s = H·h1/(h1+h2)` (9.6 for mini3s, 16 for 64k3s, 6.4 for tiny3s) |
| vertical wind | `w_shell = w_w/s` |
| cloud `[zi, zi+dc]` | `[zi/s, (zi+dc)/s]`, overlapped with the **actual layer intervals** `[z_k, z_{k+1}]` (non-uniform), `frac_k = overlap/thickness_k`, as `cloud_weights` but with the layer table instead of `field_cell·k` |
| `sample_wind` edge clip (`z < field_cell`, `z ≥ (fz−1)·field_cell`) | `z < z_1` and `z ≥ z_{FZ−1}` |
| rule R2 (vertical non-penetration) | `w_edge/s` replaces `w_edge` (easier); R3 unchanged |
| rule `H1 + H2 = world.height` | removed for a sphere: any `H1 + H2 > 0`; `s` is derived; `WeatherParams` gains no field (the map lives in a `WeatherGeometry` helper), so the box's params hash is untouched |

Effect: the cloud covers the same *relative* depth in the shell as in the box, so every preset gets rain food and seeding, and the confirmed W0 set (`g_top`, `g_prime`, the deformation radius 13 to 19, `H1`, `H2`) is kept.
Organisms above `zi_shell` (about 40 % of the shell height) feel layer 2 as in the box. Every use of `world.height` in `weather.py` (13 sites) goes through `WeatherGeometry` in S4; until then they
are allowlisted (2.9). **The S4 audit tests**: for every preset the column food equals `rain_gain·P` less the floors (`rain_regen` documents ≈ ½ quantum per used layer), the seeding band is non-empty, and
a prescribed wind profile gives the same relative-height profile as the box. **Comparability** (review, D3): each shell is thinner than the box analogue (24, 80, 80), so the 3D ecology (buoyancy layering,
benthic/photic profile) is not comparable to RESULTS_3D; results documents say so.

### 4.3 Measured behaviour of the scheme (prototype `rev1/wproto.py`; R = 90, `c_ext = 0.857`, Courant `C = c_ext·dt/(0.81√A)`)

*Static divergence of solid-body rotation (exactly divergence-free), max / rms over all cells:*

| Nside | plain faces (arithmetic mean, true normals) | gradient-corrected faces | convergence |
|---|---|---|---|
| 6 | 0.31 / 0.091 | 0.028 / 0.0056 | |
| 12 | 0.33 / 0.073 | 0.019 / 0.0028 | |
| 24 | 0.34 / 0.055 | 0.0107 / 0.0012 | max first order, rms second |

The plain-face error does not converge in the max norm (confined to the polar-cap facet seams, where the edge midpoint is far from the chord midpoint: the reviewer's cause, confirmed); with
centre-to-centre normals it grows like 1/h (review: 4.1 / 9.4 / 20 at Nside 6 / 12 / 24). Mass is conserved either way, so a mass gate cannot see it: a persistent ridge along the 8 seam meridians
forms in ~10 steps for ζ ~ 0.01. **Caveat found in revision 1:** for the gradient field of z (`div = −2z`) the max error is a constant 0.2 (10 %) at the 8 pole-vertex cells (the four cells meeting at each pole;
cell-centre value versus cell mean), the rms converges (0.033 / 0.015 / 0.0073). It is recorded and gated as bounded, not hidden.

*Stability* (linear gravity-wave noise, 1 500 steps, `f = 0`, no advection; "UNSTABLE@n" = grows by 10³ at step n):

| scheme | Nside 6, 12, 24 |
|---|---|
| plain faces, wide gradient (first draft) | unstable at every Courant 0.3 to 1.1, in 29 to 174 steps (the reviewer's min Re K < 0, confirmed) |
| gradient-corrected + Rhie-Chow, no filter | stable for hundreds of steps but a **real eigenvalue 1.004 to 1.007 per step at every Courant** (Nside 4, 6; dt-independent; the mode is a ±1 pattern on 8 cells at |z| = 0.5), unstable at ~460 to 840 steps |
| + biharmonic filter `a4 = 2·10⁻⁴` | ρ = 1.00000 at Nside 4, 6; Nside 12 stable; **Nside 24 unstable at ~800 steps** |
| + `a4 = 1·10⁻³` (**adopted**) | stable for 1 500 steps at C = 0.3, 0.5, 0.55, 0.6 at Nside 12 and 24 and for 1 200 steps at Nside 42 (also `a4 = 3·10⁻³`); **unstable at C = 0.65**, so the measured limit is **≈ 0.62** |

So (a) the first draft's "the Rhie-Chow term is a measure against checkerboards" understated it: it is what makes the scheme stable at all; (b) the reviewer's linear estimate of the limit (0.83, from
`λ_max` of the pure compact operator) is not what the full operator does: measured 0.62 with the wide-gradient parts; the first draft's "stable to 1.2" was wrong; (c) the used Courant numbers (0.46, 0.47, 0.48) leave a
margin of 1.3, not 2.5, and the rule is `C_used ≤ 0.8·C_measured(Nw)` with `C_measured` from the S4a noise test at each preset's `Nw`; (d) the filter is part of the scheme (`a4 = 1·10⁻³`, an `ENGINE_FIXED` value
until S4c): its damping per step is `a4·(k·dx)⁴`, 6.4 % at the 2Δ mode and 4·10⁻⁵ (e-folding 25 000 steps) at the storm scale `k·dx ≈ 0.45`, below the radiative damping (`r1 = 2·10⁻⁴`).

*Williamson test 2* (steady geostrophic flow, `R = 90`, `f0 = 0.025`, `u0 = 0.1`, `h0 = 80`, the Coriolis axis tilted with the flow as in the definition; **my first run tilted the flow but not f and
reported nonsense for α = 45°: the test needs `f = f0·(x·a)`**), 2 model days (1 005 steps), normalised `l2(h) = ‖h − h(0)‖/‖h(0)‖` (Williamson's measure):

| Nside | α = 0° | α = 45° | `l2(u)` (0° / 45°) | `dE/E` | mass | seam-cell error share (area share) |
|---|---|---|---|---|---|---|
| 12 | 2.5·10⁻³ | 1.9·10⁻³ | 7.1·10⁻² / 5.8·10⁻² | −1.2·10⁻³ | 10⁻¹⁵ | 0.38 / 0.16 (0.10) |
| 24 | 1.1·10⁻³ | 9.8·10⁻⁴ | 3.5·10⁻² / 3.0·10⁻² | −6·10⁻⁴ | 10⁻¹⁵ | 0.12 / 0.08 (0.05) |
| 48 | 6.0·10⁻⁴ | 6.4·10⁻⁴ | 1.6·10⁻² / 2.5·10⁻² | −3·10⁻⁴ | 2·10⁻¹⁵ | 0.04 / 0.03 (0.03) |

Plain faces blow up in 154 to 349 steps; gradient-corrected + Rhie-Chow without the filter drifts to `l2(h)` 0.02 to 0.11 in 2 days. *Rossby-Haurwitz wave 4* (TC6 initial state scaled to the model, `ω = K = 0.1076 Ω`,
Froude number 0.14), phase shift of the m = 4 component of h against the non-divergent theory `ν = [R(3+R)ω − 2Ω]/[(1+R)(2+R)]` (0.21 rad per model day):

| Nside | ratio measured/theory after 1 day | after 2 days | `dE/E` (2 days) |
|---|---|---|---|
| 12 | 0.795 | 0.675 | −1.2·10⁻² |
| 24 | 0.897 | 0.836 | −8·10⁻³ |
| 48 | 0.941 | 0.900 | −5·10⁻³ |

The waves propagate **slow by 6 % (Nside 48) to 10 % (24) per day, converging at about first order**: the semi-Lagrangian fan interpolation and the filter are the cause; the consequence for the
storms is that beta drift and Rossby propagation are 6 to 10 % too slow at the presets' `Nw` (S4c may compensate or accept).

### 4.4 The S4a gates (analytic or published references, fixed before the MLX port; the prototype's numbers above are the first evidence, not the threshold)

| # | gate | reference | threshold |
|---|---|---|---|
| G1 | divergence of solid-body rotation, and of `∇z`, on Nside 12 / 24 / 48 and on the three presets' `Nw`, **all cells including the seams and the 8 pole-vertex cells** | exact (analytic) | solid rotation: max ≤ 3 % of `|ω|` and converging, rms second order; `∇z`: rms converging at ≥ first order, max ≤ 12 % and attributed (pole-vertex cells) |
| G2 | linear noise stability, 1 500 steps, all three `Nw`, at `1.2·C_used` (and a measured `C_limit` reported) | stability theory | no growth; `C_limit ≥ 1.25·C_used`; ρ of the dense one-step map at Nside ≤ 6 ≤ 1 + 10⁻⁶ |
| G3 | Williamson TC2, α = 0° and 45°, 2 model days (and 5 at the end of S4b), at the preset `Nw` | analytic steady state | `l2(h) ≤ 5·10⁻³` at Nside 24, ≥ 1.5× smaller per doubling; `|dmass|/m ≤ 10⁻¹²`; `|dE/E| ≤ 5·10⁻³` per 2 days |
| G4 | Rossby-Haurwitz 4 (above) | non-divergent theory (valid to `Fr² ≈ 2 %`) | phase ratio within 12 % of 1 after 1 day at Nside 24, within 8 % at 48, error decreasing with refinement |
| G5 | imprint: TC2 error share on seam cells ≤ 3× their area share at the preset `Nw`; `l2(h)` over α ∈ {0°, 45°, 90°} within a factor 2; belt exposure of the seam cells reported (5 to 8 % for every tilt, review M3) | symmetry | stated |
| G6 | mass conservation of the two-layer + moisture model under the limiter, `clamp_mass` reported | exact | relative 10⁻¹² |
| G7 | flow over an isolated mountain (TC5 geometry: a cone, `b` added to the potential): no blow-up in 15 days, mass and energy drift bounded, qualitative lee wave | **no analytic or independent reference is available offline**; only the conservation checks are gates, the field comparison is run against the same scheme at twice the resolution (self-convergence, labelled as such) | conservation only |

G1 to G4 and G6 are independent of the scheme's output; G5 is a symmetry check; G7's field comparison is self-referential and says so. The thresholds of G3 and G4 are the levels
second-order finite-volume models reach at about 2.5° (cell side `√A/R` = 2.4° at Nside 24); the prototype passes them with margins of 5× (G3) and 1.2× (G4 at 24).

### 4.5 When the fallback triggers, and what it costs (review M8)

**Trigger.** After the allotted S4a budget (two iterations of the face reconstruction and the stabiliser, each re-running G1 to G6 in numpy; about two working days), the fallback is taken if any of:
G1 or G3 fails at the preset `Nw`; the filter strength needed for G2 exceeds `a4 = 3·10⁻³` (storm-scale damping 1.2·10⁻⁴ per step, comparable to the radiative damping); or, in S4c, the storm criteria of STORMS_W0
(2 to 6 storms per hemisphere, lifetime > 3 000) cannot be met with the damping levers within their ranges.

**Cost.** The first draft said "two more PRs". The fallback is an icosahedral SCVT mesh generator (the raw mesh has area ratio 2.0), the TRiSK operators (edge normals, kite areas, tangential reconstruction weights,
energy-conserving Coriolis), a new MLX weather module with no `roll`, a conservative remap table to the HEALPix field cells and back, the engine coupling again, and S4b/S4c redone: **S4 again, 4 to 5 PRs**.
Variant B above is the cheap intermediate that is tried first and is what the budget measures. The field/hash/agent half of the sphere (S0 to S3) does not depend on this choice, so `Nw = r·Nf` is **not** a constraint on D8: a TRiSK mesh
would carry its own resolution and the remap table.

**S4a scope (review M8).** The prototype is the linear/nonlinear shallow-water core (single and two layers) plus the pools and the pointwise physics needed for the imprint and storm-count tests, not a numpy port of all of
`weather.py` (1 690 lines): about 600 lines on top of the existing `rev1/wproto.py` (300 lines, exact tables from the independent HEALPix of `review_probes/hp.py`).

---

## 5. Observers, stream, dumps, metrics, game layer, director

The per-consumer audit with the converting PR (raw z/vz, `min_image`, `world.size`, `world.height`, lattice shape) is the table of 2.9; this section is the design of the host side. Everything host-side that touches positions gets one small library, `alife/geom.py` (pure numpy, device independent, fully
tested in S0): `ang2pix/pix2vec/neighbours4/neighbours8` for any Nside, cell areas, `great_circle_distance`,
`mean_direction`, `to_latlon`, `local_frame(û)` (east/north), `circular_mean` replacement, flood fill on the cell graph,
`layer_edges(Rp, H, FZ)`. The device twin is `alife/sphere.py`.

| module | geometry dependence today | sphere treatment | PR |
|---|---|---|---|
| `featobs.py`, `weather_obs.py` | lattice reads by `fcell`, 12 + 13 `min_image`/size uses | cell ids are flat (layer, pix): lattice reads unchanged; torus distances to geodesic/chord helpers | S5 |
| `metrics.py` | `z` bands from `pos[:,2]`, `hash_occupancy` over `(G_z, G, G)`, FG² layers, carrying capacity | `z = |pos| − Rp` helper; occupancy over `(G_z, Npix_h)`; per-layer sums over the flat index; `area` metrics use cell area | S5 |
| `groups.plane_cell`, `tracking.grid_clusters` | coarse plane cells; clusters on a lattice with 4 face neighbours | `groups.sphere_cell(pos, Nt)` (device, HEALPix) and `grid_clusters` over the coarse cell graph with `neighbours4` | S5 |
| `game/society.py` (20 sites), `housefile`, `kinthreads`, `tracers`, `observer`, `social` | territory cells, `foot_cells`, centroids with `circular_mean`, torus distances | coarse cells `Nt`; centroids as `mean_direction` (unit vector, re-expressed as lat/lon for text); distances geodesic | S5 |
| `storms.py`, `weather_obs` | vorticity maps, connected components on the weather grid, track matching by distance, radius from area | components over the weather cell graph; centroid by mean direction; radius `sqrt(Σ A_c/π)`; speed in arc units; the vorticity unit stays the reference `f0 = 0.025` (the sphere uses `f0 = 0.025` too) | S5 |
| `game/alerts`, `powers`, `report`, `chronicle` | positions in event texts; `powers` places patches/blight/catastrophe by x,y | positions as (lat, lon, height) in text; powers pick by unit vector or (lon, lat) and a cap radius | S5 |
| `stream.py`, `streamrec.py` | header `L`, `FG`; FRMA positions and yaw; `MAP1` | below | S5 |
| `analysis/*` | 2D-only tools already refuse 3D; xy-based 3D analyses (body extents, height bands) use the torus | sphere variants where the result is cited (bodies, clades by height); height bands from `z` | S5/S7 |
| director (`director.py` rules, `State.env`) | `env_cat_center` x,y; patch levers; menu scales radius bounds in 3D | `(lon, lat)` centre drawn uniform by area; patch_gain levers unchanged; radius bounds scale with `Rp` | S5 |

**Stream protocol v2.** `MAGIC` keeps its major byte; `PROTOCOL` becomes 2 for sphere worlds only (a box stream still writes 1,
byte for byte). The header gains `geometry: "sphere"`, `radius` (planet), `height`, `shell_edges` (FZ+1 layer radii), `nside_field`, `nside_weather`,
`nside_territory`, `ring_order: "ring"`, `f0`, `weather_depth` (`H1+H2`); `L` is omitted and `FG` is 0. FRMA positions stay float32 x, y, z
(planet frame); the orientation pair stays (yaw, pitch) with yaw defined as the bearing from local north of 2.1 (east = normalise(ẑ × û)), so the record layout is
unchanged and the client rebuilds `hdg = cosψ·north + sinψ·east` (the convention is fixed in the protocol text and checked by the independent decoder). Record changes: `MAP1` becomes
`MAPH` (the same planes over `12·Nc²` ring-ordered coarse weather cells, `Nc = Nw/2^k`, with `nside` in the prefix instead of `n, nw`), `TERR` carries
coarse HEALPix ids (`Nt = 9`, 972 cells ≤ the 1 024 JSON cap), `STRC`/`FSOL` use `nx = Npix`, `EVNT` x,y,z are planet-frame Cartesian. Old clients refuse
`protocol 2` by the existing header check. `tests/stream_decode.py` and `tests/play_decode.py` (the independent decoders) get the sphere
variants first, so the encoder is checked against text, not against itself.

**Dumps and checkpoints.** `mat`, `soil`, arms dumps are `(FZ, Npix)`; a `geometry.json` sidecar in the run directory states `Rp, H, Nside, FZ` so offline tools need not
import the engine. `meta.json` and checkpoint metadata gain `geometry` (absent = box, as `dims` was in §27.1); `radius`/`nside`/`layers` are in the config hash.

---

## 6. The GUI

The sphere world gets its own renderer path; the box world's paths (Mercator globe, Morph, torus seam copies) are left exactly as they
are and keep serving box worlds. Nothing in the sphere path morphs: a planet is a planet at every zoom.

| area | change |
|---|---|
| `StreamProtocol` / `PlayRecords` | decode header v2, `MAPH`, sphere `TERR`/`STRC`/`FSOL`; a `WorldGeometry` enum (`box(L, H)`, `sphere(Rp, H, edges, nside…)`) carried by `WorldStore` |
| globe renderer (new `SphereWorldView`, MSL in `ShadersSphere.swift`) | planet = an icosphere mesh at `Rp`; surface colour from a per-cell buffer indexed in the fragment shader by `healpix_ring(n̂)` (about 25 lines of MSL, the same closed form as the engine's `ang2pix`), so overlays are seamless at any zoom with no texture and no cube map; the atmosphere is a translucent shell at `Rp + H` (the lid, visible); sun and terminator optional; organisms are instanced (position, `fwd`/`up` from the transmitted yaw/pitch); bonds are chord lines |
| camera (`Camera`, `WorldPose`, `WorldVantage`, `FlightPath`, `WorldFollow`) | orbit about the planet centre, a "surface hover" pose (target `û_t`, height, bearing, tilt) for close-ups: the camera simply descends, the horizon curves, no tangent-plane morph; the flat close-up chart, when wanted, is the **azimuthal equidistant projection about the target** (geodesic distance and bearing from `û_t`, computed in the vertex shader) for 60° around it; flights follow great circles (slerp) |
| global map | an equal-area Mollweide chart of the HEALPix cells, recentrable in longitude, for the overview and the minimap (`WorldInsetGlobe`): replaces the Mercator globe for sphere worlds |
| picking (`WorldPicker`, `WorldProjector`) | ray against the three spheres (floor, lid, an organism's bounding ball) and `healpix_ring` on the hit; organisms by nearest-to-ray among visible instances |
| overlays (`MapPalette`, `FeatureLayers`) | material/soil/resource/food per cell on the floor sphere or on a chosen layer shell (`field_z` slider as today); storms as vorticity-coloured cells plus tracked glyphs and trails; weather wind as streamline lines on the sphere |
| HUD, inspector, labels (`WorldHUD`, `Inspector`, `WorldReadout`, `WorldLabels`) | positions as latitude, longitude, height; distances in arc units; compass; `LabelValidity` unchanged |
| levers/launcher (`World3DLevers`, `Levers`, `PresetFile`, `PlayLaunch`, `WorldLauncherModel`) | `world.geometry`, `radius`, `nside`, `layers`, `ceiling_band`; `Preset` enum gains `tiny3s, mini3s, 64k3s` (`test_gui_sync` requires each name to equal `config.PRESETS` and have a TOML); `PlayLaunch.arguments` chooses `stormsph` by the source's `world.geometry`, never by an alias |
| retired for sphere worlds only | `Torus`, `SeamCopies`, `Morph`, `GlobeMap` (Mercator) |

Swift tests: pure-math ports of `healpix_ring` (the shader's function, tested against the Python tables), the projection and slerp,
picking, header/record decoders, the lever sync; `cd gui && swift build` as the gate.

---

## 7. Configuration, compatibility, presets

### 7.1 Keys (all `POST_V01_KEYS`, omitted from `config_hash`/`config.json` at the default)

New `config.SPHERE_KEYS` (the sphere analogue of `THREE_D_KEYS`; in a box world every key but `world.geometry` must keep its default, so no
box config changes hash without changing behaviour):

| key | default | meaning |
|---|---|---|
| `world.geometry` | `"box"` | `"box"` or `"sphere"`; static (changes the compiled step and the state) |
| `world.radius` | 0.0 | planet radius `Rp` (sphere) |
| `world.nside` | 0 | field/hash Nside `Nf` |
| `world.layers` | 0 | `FZ` equal-volume layers |
| `world.height` | (existing) | shell thickness `H` |
| `physics.ceiling_band` | 0.0 | soft lid band; 0 = off |
| `weather.cells` | (existing) | on a sphere: `Nw`, not a grid side |

There is no `hash_reach` key (reach 2 only, 2.4) and no tilt key (1.4). `tests/test_gui_sync.py` gets a `SPHERE_KEYS` lever-coverage test in S0 with a CLI-only set, like `CLI_ONLY_3D` (review M6).

### 7.2 Validation (messages follow the existing style; the sphere rules apply whenever `geometry = "sphere"`)

geometry in `GEOMETRY_SUPPORTED` (read at call time, `("box",)` until S2 enables it); `world.dims = 3`; the features a sphere run may enable are the set `SPHERE_READY` (2.9, rule 4);
`radius` finite, `> 0`, `≥ 20`; `height > 2r` and `height − 2r ≥ 2·repro.offspring_distance`; `height ≤ radius` (thin shell: bounds the area ratio by 4 and the hash occupancy); `size > 4(radius + height)`
(2.5); the box-derived rules (`G_hash ≥ 3`, `FG`, `FZ = height/field_cell`, `2·height ≤ size`) are skipped; `nside ≥ 2` (HEALPix at Nside 1 has 6 touching neighbours, review), `layers ≥ 2`; the floor layer holds a grounded agent:
`r + floor_eps < z_1`; `floor_eps ≥ 0.01`; cell volume within `[0.4, 2.5]·field_cell³`; `FZ·12·Nf² ≤ 2²⁰` (the `exact_sum` limit of §27.1); the derived `Nh` from the exact reach-2 coverage table
(1.5) with `Nh ≥ 2`; `C ≥ ceil(0.167·V_lid) + 4` and `K ≤ Q = 125·C`; the diffusion rules of 3.1 from the tables (`Σα ≤ 0.9`, hence `d_q8 ≤ floor(0.9·65536/rowsum_max)`, `check_diffusion_tables`,
the chemical sub-step count); weather: `Nw = r·Nf` with `r` a power of two, `C_used ≤ 0.8·C_measured(Nw)` from the table the S4a test writes, pools within |lat| ≤ 60° with `vy = 0`, `H1 + H2 > 0`
(the vertical map of 4.2 replaces `H1 + H2 = height`); the 3D rules that survive unchanged (gravity vs thrust, buoyancy load rule R2, `0 ≤ gravity < thrust_max·sin(PITCH_MAX)`).
In a box world `world.radius`, `nside`, `layers`, `ceiling_band` are rejected ("inert while geometry is box"). `WeatherParams` gains no field, so the box's params hash is unchanged (review M9).

### 7.3 Presets (all provisional; calibration is S7) — `sizing.py`

| | `tiny3s` | `mini3s` | `64k3s` |
|---|---|---|---|
| analogue | `tiny3d` (128², H 24) | `mini3d` (160², H 80) | `64k3d` (320², H 80) |
| `Rp`, `H`, `r_top` | 36, 16, 52 | 48, 24, 72 | 90, 40, 130 |
| floor area (box) | 16 286 (16 384) | 28 953 (25 600) | 101 788 (102 400) |
| shell volume (box) | 0.39 M (0.39 M) | 1.10 M (2.05 M) | 6.15 M (8.19 M) |
| top/floor area | 2.09 | 2.25 | 2.09 |
| field `Nf`, `FZ`, cells, cell volume | 8, 8, 6 144, 64.1 | 11, 12, 17 424, 63.1 | 21, 18, 95 256, 64.6 |
| floor / lid layer thickness | 2.8 / 1.5 | 3.0 / 1.4 | 3.2 / 1.6 |
| hash (reach 2): `Nh = Nf`, `G_z`, `C` (provisional, ≥ `ceil(0.167·V_lid)+4` = 28 / 27 / 25); exact coverage margin over the rigorous angle | 8, 5, 32; 12.0 % | 11, 8, 32; 6.8 % | 21, 13, 32; 2.3 % |
| `d_q8` bound (`Σα ≤ 0.9`), default 4 | 11 | 11 | 14 |
| chemical sub-steps for `D_c = 0.16` | 4 | 4 | 4 |
| vertical map `s = (H1+H2)/H` | 5 | 3.33 | 2 |
| weather `Nw` (cells, √A, Courant) | 16 (3 072, 2.30, 0.46) | 22 (5 808, 2.23, 0.47) | 42 (21 168, 2.19, 0.48) |
| `n_max`, `n_init` | 1 024, 256 | 8 192, 2 048 | 49 152, 12 288 |
| alive expected | ~2 000 to 6 700 | ~5 600 to 11 800 | ~31 000 to 41 600 |
| `n_patches`, `patch_sigma` | 4, 12 | 3, 18 | 8, 24 |

"Alive expected" is the range between the volume density of mini3d/64k3d and the floor-area density (0.409/unit²); `regen_max`
is recalibrated in S7 from a short run, as `64k3d` was. The gravity, buoyancy, energy costs and mutation keys are the 3D presets'.
`stormsph` (a world overlay, never an alias, validated against the world): `f0 = 0.025`, pools at ±40° latitude (f at the pool 0.0162,
as 64k3d), 24×16 pools for mini3s and 32×16 for 64k3s (two overlay files if the shapes differ, as `storm160`), `rain_gain`, `storm_*`
the box's provisional values; `β = f0·cos(lat)/Rp` at the pools is 2.1e-4 (64k3s) and 4.0e-4 (mini3s) against 3.7e-4 in 64k3d.

### 7.4 Compatibility and the missing migration

* `geometry = "box"`: every code path, trajectory, `StepOut`, metrics row, config hash, checkpoint key set and golden digest is unchanged;
  the new keys are absent from `config.json`; `State`/`State3` untouched; `Dims` gains fields with box-neutral values; `step` gains static `if d.sph:` branches
  beside the existing `if d3:`. Gates: `tests/identity_matrix.py --base <main sha>` (OK: bit-identical) in every PR from S0, the 3D tiny/64k golden digests,
  `test_gui_sync`, `test_lint`. The sphere cases cannot be in that matrix (the base has no sphere): their digests are pinned as literals in a separate test.
* Checkpoint metadata `geometry`; `load_checkpoint` refuses a file whose geometry differs from the config ("cross-geometry resume/fork is not supported", exit 4); fork
  refuses a changed `geometry`, `radius`, `nside`, `layers` (the state shapes); resume refuses them as dynamics keys. `FORMAT_VERSION`, `RNG_LAYOUT_VERSION` unchanged.
* **No torus-to-sphere checkpoint migration** (owner instruction). Consequences: (1) no evolved population carries over: the 500 000-step `mini3d_s0` (31 min idle) and
  the 9.5-hour `64k3d_s0`, and every saved planet, stay box worlds; a sphere starts from founders and must be evolved (estimate, 500 000 steps: `mini3s` 45 to 60 minutes,
  `64k3s` 10 to 15 hours); (2) no result of RESULTS_3D/RESULTS_3D_RUNS transfers numerically (different volume, curvature, density at the lid): comparisons are qualitative or via a
  flat-limit run (the last bullet of this section); (3) the launcher must show a world's geometry and refuse cross-geometry forks with a message; (4) the genome layout is shared (84 → 39), so a later
  optional tool could re-seed a sphere with the genomes of a box run (a reseed, not a migration: positions, fields and weather start fresh); not planned here.
* **What 2D means.** The sphere is the 3D shell only. A 2D sphere (no gravity, `D_IN = 69`) would be a third state class, a second set of tables and a second GUI
  path for a world nobody asked for; a thin shell (`H` a few body radii, `layers 2`) is the surface world. `world.dims = 2` with `geometry = "sphere"` is rejected.
* **Flat-limit test** (review M9 corrected it): a sphere of radius 2 000 violates `FZ·12·Nf² ≤ 2²⁰` and the cell-volume bound at cell side 4 (3.1 M cells), so the test builds the config without `validate`, with the fields off (`FZ = 2`, a coarse
  `field_cell`), agents confined to 40 units, and compares with the box 3D step over 20 steps with a tolerance of 1 % **of the step length** (float32 ulp at 2 000 is 1.2e-4 against steps of 0.03 to 0.5, so 1e-3 absolute was too tight when the displacement is ~0.1). It catches every sign and frame
  error without needing bitwise equality; a float64 numpy reference of the sphere step is the second oracle.
* **Format note** (review M9): a sphere checkpoint read by pre-sphere code sees `geometry` absent (= box) and fails on shapes; `FORMAT_VERSION` stays 1 but the refusal text says "written by a newer engine (geometry)".
* **Identity matrix and `test_gui_sync`**: `Dims`, `WeatherParams` and every config addition are hash-neutral (omitted at defaults). `test_the_preset_picker_lists_the_engine_presets` would fail from S2 on (S2 adds `tiny3s` and its TOML; the Swift `Preset` enum gets it only in S6): the test gets an explicit `SPHERE_PRESETS_PENDING_GUI` exemption in S2, removed in S6 (review M6).

---

## 8. Performance, memory

Basis: DESIGN §37.3, steady state, weather on, M5 Pro, fused Metal kernel: `mini3d` 4.5 ms/step (10.5 k alive; dry 3.4), `64k3d` 14.9 ms (dry 13.3). The neighbour
search is about 75 to 80 % of a dry 3D step (§27.13). The sphere's candidate volume per agent relative to the box with the exact coverage and the discrete Nsides of the presets
(1.3, review M7): **reach 2: 34.2 R³ at 64k3s (1.27×; 1.45× at tiny3s)** against the box's 27 R³ (reach 1 would be 70.6 R³, 2.6×, and is not supported); times about 1.3 for the lid-side cells (the share of the population in the top layers of
64k3d was 35 %, and a lid hash cell holds 2.1 to 2.25× the volume of a floor cell), times the alive ratio. The first draft's 1.1× and 1.8× used a continuous Nside and the sampled coverage.

| stage | `64k3d` | `64k3s` (estimate) | note |
|---|---|---|---|
| neighbour search (Metal) | ~10 ms at 42 k alive | 1.27·1.3·0.76 ≈ 1.25× → 12.5 ms, plus the run overhead of 125 runs per thread against 27 (unknown) | S2 gate: more than 2× mini3d per alive agent stops S2 and goes to the owner (2.4) |
| rest of the dry step (brain, S8 costs, lifecycle, fields) | ~3.3 ms | ~2.5 ms (fewer alive) | |
| sphere geometry per agent (`ang2pix`, frame, exp map, transport, ~250 elementwise ops on (N,3)) | – | +0.3 ms | |
| field diffusion by gather tables instead of rolls (95 k cells × 6), int64 flux | ~0.1 ms | +0.3 ms | |
| chemical sub-steps (4 × the 7-point update on 3 channels) | included | +0.3 ms | 3.1 |
| weather (box: ~1.6 ms at 25.6 k cells) | 1.6 ms | ~3.5 ms (21 k cells; gathers instead of rolls, the 8-triangle fan, the face corrections, the filter) | S4b gate: ≤ 4 ms |
| **total** | 14.9 ms | **~19.5 ms (≈ 51 steps/s)** | ±30 % |
| `mini3s` (vs 4.5 ms) | | ~5.5 ms | 5.6 to 11.8 k alive |

Memory: the state grows by `hdg` (N,3) f32: +0.8 MB at N = 49 152; `wx` 11 planes of 21 168 cells = 0.93 MB (the box 9 × 25 600 = 0.92 MB). Tables (host-built, float64 then cast, kept as
constants or built inside the traced function as `weather._rot_lists` is): hash stencil (5 292 × 25) i32 0.5 MB; field `nbr4`, the conductance weights; weather tables (21 168 × 8 × 6 fan and weights, edge tables) f32 about 6 MB. Checkpoint size changes by under 1 %.
The pure-MLX search path at 64k3s has `Q = 125·32 = 4 000`: `(N, Q)` arrays three times 64k3d's (Q = 1 296), a fallback, not a target.

---

## 9. Risk list

| # | risk | why | mitigation |
|---|---|---|---|
| R1 | weather scheme quality (A-grid noise, grid imprint at the polar-cap seams and the 8 trivalent vertices, 6 to 10 % slow Rossby propagation, different numerical diffusion) | highest uncertainty, the model was tuned on a periodic C-grid; the first design was inconsistent at the seams and unstable | measured scheme (4.3: gradient-corrected faces, Rhie-Chow, biharmonic filter, limit C ≈ 0.62); gates G1 to G7 (4.4) before any MLX port; fallback trigger and cost in 4.5 (S4 again); ship the dry sphere first |
| R2 | neighbour-search cost (75 to 80 % of the step) | reach-2 stencil 1.27× the box at 64k3s (1.45× tiny3s), lid cells 2.1 to 2.25× more agents, 125 runs per thread | reach 2 only; S2 stop rule at 2× per alive agent; optional cubed-sphere hash; thin shells; `C` derived from `V_lid` and measured on a Mac run |
| R3 | float32 near the poles and face edges | unit vectors and a closed-form lookup | Cartesian state has no pole; `ang2pix` computes `1−z²` from `(x²+y²)/r²`; boundary flips are harmless by construction; tests at both poles, the 8 trivalent vertices, the 4 equatorial vertices and every layer interface in float32 against float64 |
| R4 | heading drift and frame errors | transport on a curved manifold | probe: invariants 1e-7 over 3·10^5 steps; mandatory renormalisation; holonomy test; flat-limit test; float64 reference model |
| R5 | determinism | gather order, `argsort` ties | same sort and stable insertion as the box (F16, F21); compiled = eager tests; CPU digest pinned; Metal vs MLX path bitwise (`self_check`) |
| R6 | Metal kernel variant | new table reads inside the fused kernel | change confined to the stencil and the distance; bitwise test vs the pure-MLX path on adversarial scenes (ties, poles, empty runs); `VERIFIED_MLX` rules unchanged |
| R7 | MLX compile bugs: the constant-collision of F21 item 6 (`ang2pix` is exactly its pattern: `floor(x·c).astype(int32) % g` with several distinct integer and float constants per Nside, Rp, FZ, G_z, Npix) and I14 (outputs derived from inputs, duplicate rolls/gathers) | many gathers of the same tables; sphere tests parametrise Nside in one process; the CPU kernel cache is machine-wide | **prerequisite in 10.1**: the guard of branch `fix/mlx-compile-collision` (I53) merged before the first compiled sphere step, or the fallback there; tables built once per trace; `compile = eager` for every `ang2pix`/layer check in a permuted case order; stage-wise `tests/compile_probe.py` in S2/S4b |
| R8 | scatter rule and F20 | new cell counts, LSQ weights | only `ops.scatter_add_i32`; all weights are host tables and sums of products, no matmul; `test_lint` extended to the sphere modules |
| R9 | grid anisotropy of the 4-neighbour diffusion | cell shape varies 0.81 to 1.40; edge `L/D` 0.63 to 1.69 | conductance weights (3.1); S1 gate restated: multiplet means within 3 %, spread within ±25 % (individual modes spread −17 to +23 %: an ecological effect to document, not hide) |
| R10 | ecology differs (thin shell, lid sheets, jet exposure) | different volume and curvature | calibration runs S7; results doc states what does not transfer; the soft lid band |
| R11 | owner GPU use | probes must be tiny | all gates up to S2's kernel are CPU; the Metal gates are short (a few minutes) and announced to the coordinator |
| R12 | silent wrongness in the ~30 consumers of `pos[:,2]`, `vel[:,2]`, `min_image`, `world.size/height` (review H4) | the Cartesian state makes them type-check | accessors + geometry lint with ratchet + rotation-equivariance test + readiness gates + the audit table (2.9) |
| R13 | int32 overflow in the diffusion flux, and the vertical/horizontal mixing ratio (review H5) | `R_HARD·d_q8·w` exceeds 2³¹ | int64 flux, proven bounds, `check_diffusion_tables`, corner tests (3.1) |
| R14 | the rain/seeding vertical placement is empty for a shell thinner than the atmosphere (review H3) | `cloud_weights` overlaps `[32, 32+dc]` with the layers | the vertical map of 4.2 and its S4 audit tests |

---

## 10. Build plan

Each PR is independently testable, keeps the box bit-identical (identity matrix + goldens + lint + `ruff` + `pytest -m "not slow"` + `swift build` for S6), and
updates DESIGN §46 and a Review-log row in the same change (I77 is the series row; each PR appends a line). **S0 can start as soon as this revision is accepted**: it needs
nothing from S1 to S7 and no decision still open (the tilt, the stencil reach and the coverage constants are settled).

### 10.1 Prerequisite: the MLX compile-collision guard (review M5)

MLX 0.32.3's compile cache can hand a compiled function another one's fused kernel when their scalar constants print alike (F21 item 6; `floor(x·c).astype(int32) % g` with `(c, g)` = (12.0, 1) and (1.0, 21)),
and the CPU kernel cache is machine-wide (`$TMPDIR/mlx/0.32.3/cpu/`), so "one config per process" is not enough on the CPU. `ang2pix`, the layer index and the hash cell are exactly that pattern with several
distinct constants per Nside, Rp, FZ, G_z and Npix, and the sphere tests parametrise them. **Prerequisite of S2** (the first PR that compiles the sphere step) and of every compiled sphere test from S0 on that
parametrises a constant: the unmerged branch `fix/mlx-compile-collision` (I53, renumbered from I50, `tests/jit_cache.py`: a private CPU kernel cache per pytest session; `tests/compile_probe.py`;
`tests/test_mlx_compile_collision.py`; DESIGN F21 item 6, §15.4, §22.0) is merged first (needs the coordinator's go). **Fallback if it is not merged when S2 starts:** (1) S0/S1 tests are eager, and a compiled
test uses constants passed as runtime `mx.array` inputs (nothing printable to collide); (2) every compiled sphere test runs one configuration per subprocess with its own `TMPDIR`; (3) `make_step` on a sphere runs a cheap
compiled-vs-eager self-check of `ang2pix` and the layer index for its own constants and refuses to run on a mismatch; (4) `compile = eager` cases run in a permuted order.

| PR | scope | what its tests prove | gates | device |
|---|---|---|---|---|
| **S0** geometry and guards | `alife/geom.py` (host) and `alife/sphere.py` (device): HEALPix `ang2pix/pix2vec`, **exact** neighbour tables from shared pixel corners (touching, edge, ring-2, fan triangles, LSQ weights, parent-child `(ix//r, iy//r, face)`, colouring), the **exact coverage algorithm** (boundary to boundary, cached) and `Nh` derivation, layer edges, `C` rule; the geometry accessors; the geometry lint with `tests/geometry_allowlist.py` (the audit of 2.9 as data); config keys, `SPHERE_KEYS` (+ lever-coverage test with a CLI-only set), validation with `GEOMETRY_SUPPORTED = ("box",)`; `Dims` fields | round trip and equal area over all pixels (Nside 2..64); 24 seven-neighbour pixels, 4 edge neighbours, symmetric tables, 8 trivalent vertices (against the reviewer-independent method); exact coverage reproduces 1.385 / 1.406 / 1.421 (reach 2, Nside 21 / 11 / 8) and 0.685 to 0.711 (reach 1); an agent at the extreme point of the worst cell, at the floor, at `R − 10⁻³`, is found (F13 at the margin); `ang2pix` float32 vs float64 (mismatches adjacent only) including poles, the 8 vertices, ring boundaries; hierarchy for integer ratios; astropy-healpix when installed; equal-volume layer identities; `min_image` identity by value; transport, holonomy and drift tests (the probes as tests); lint: ratchet and allowlist consistency; config defaults/hash unchanged, every rule has a message | identity matrix OK; `pytest`, `ruff`; no engine change | CPU |
| **S1** fields | `StateS` skeleton and `expected_layout`; sphere `init_state` (fields only); fertility on the sphere, patch rotation by ω, `int_diffuse_sph` (**int64 flux**, geometric weights), `check_diffusion_tables`, chemical sub-stepping, `sample_fields_sph`, deposit; vertical profile at interface heights; the conductance tables | exact conservation on random, saturated and near-`R_HARD` fields; **corner tests**: a floor cell at `R_HARD` with empty neighbours, one at 10⁶ in tiny3s's floor (the case that wrapped int32), uniform `R_HARD`; the cap bound against the exhaustive enumeration; `w = 256` reproduces the box stencil bit for bit; uniform quanta = uniform density; a numpy mirror; chemicals vs a float64 reference and mass conserved; spherical-harmonic gate restated (means within 3 %, spread ±25 %); gradient of a smooth field vs analytic in the body frame; draws vs the RNG table; patches stay on their circles for 10⁵ steps; compile = eager | box identity | CPU (integers exact: Metal agrees by construction; one Metal smoke) |
| **S2** agents | `physics.forces_s/integrate_s`, `ops.body_axes_s/to_body_s`, `spatial._find_neighbors_sph` (+ the Metal variant), sensing, lifecycle placement/founders/immigrants, catastrophe cap, `invariants.py` (norm, tangent, radius with tolerances), `step` wiring, counters, `tiny3s`; **`bench.py` sphere scene builder**; `cli.py` (`inspect`, `replay`, `runs`), `runio` metadata, `migrate.py` refusal; `SPHERE_READY` and the readiness gates; the rotation-equivariance harness; the `SPHERE_PRESETS_PENDING_GUI` exemption; sphere enabled in `GEOMETRY_SUPPORTED` | brute-force kNN equality at poles, vertices, layer edges, overflow; kernel = MLX path bitwise; pair antisymmetry; forces/integration vs float64; flat-limit test (7.4); level-surface equals `forces3` on a local patch; clamps and counts; ledger closes 3 000 steps; compile = eager (permuted order); digest of `tiny3s` pinned; sentinel completeness of newborn rows; rotation equivariance of metrics/invariants | **prerequisite 10.1**; identity matrix OK; bench `mini3s` ≤ 1.6× mini3d per alive agent (stop rule at 2×, 2.4); no hash overflow in 20 000 steps | CPU, then Metal: kernel test and bench (short) |
| **S3** features | structures (colouring minimised, `nopen`, effects), niche/soil, comm, bonds that pay, arms on the sphere; their counters and ledger | the existing per-feature suites run on a sphere fixture; anti-trapping invariant holds on the sphere graph; ledger closure per feature; off = unchanged | identity matrix OK | CPU |
| **S4** weather | **S4a** numpy float64 prototype (the `rev1/wproto.py` core plus the pools and pointwise physics) and gates G1 to G7 (4.4); **S4b** MLX port (`weather_sph`), `WeatherGeometry` (the vertical map), engine coupling (wind force, rain remap, shelter, seeding, hazard), `weather_spinup`, `stormsph`; **S4c** calibration and the W0-style study driver on the sphere | S4a: G1 to G7; S4b: MLX vs prototype to float32 rounding, mass conservation, calm start and spin-up reproducible, couplings on prescribed fields, **the rain/seeding audit of 4.2 for every preset**; S4c: storm statistics inside the STORMS_W0 criteria (2 to 6 storms per hemisphere, lifetime > 3 000), seam-cell exposure of storms | S4a pass, or the fallback of 4.5; weather ≤ 4 ms at 21 k cells | S4a CPU; S4b/S4c Metal (minutes) |
| **S5** observers | the audit of 2.9 for every S5 row: stream v2 and decoders, `MAPH`/`TERR`/`STRC`/`FSOL`, dumps, `geometry.json`, metrics, groups/tracking/society, `game.observe`, labels (`INTENT_VERSION` 4, thresholds provisional), tracers, storms tracker, game layer, alerts, director | decoders written from the spec text; round trips; the rotation-equivariance test over every converted consumer; society/territory on the coarse graph vs a brute-force reference; storms tracker on synthetic vortices crossing the poles and vertices; Nomad track as arc sum; director replay on a sphere log; box streams byte-identical | identity matrix, stream tests, `ruff` | CPU |
| **S6** GUI | section 6 | Swift unit tests (shader function mirror, projections, picking, decoders, levers); `test_gui_sync` (exemption removed); `swift build`; a recorded session on the Mac | owner's look | Mac (GPU, interactive) |
| **S7** presets, calibration, docs | the three presets, `stormsph`, `regen_max`/`C`/`n_init`, a 3 000-step slow test, the first long runs (`mini3s` 500 000 steps; 64k3s later), `docs/SPHERE.md`, `RESULTS_SPHERE.md`, DESIGN §46 final, HANDOVER, CLAUDE.md | slow tests: ledger exact, no overflow, population in band; run results with the caveats of 7.4 | owner sign-off | Metal runs |

Dependencies: S0 → S1 → S2 → S3; S4a can start after S0 in parallel (it needs only the exact tables); S4b needs S1 and the S4a pass; S5 needs S2 (S4 for the weather parts); S6 needs S5's
header and records and can start on synthetic streams after S2. The dry sphere (S0 to S3, S5, S6) is a complete world without storms; S4 can trail it. Rough size
(lines of code + tests, estimate): S0 1 400 + 1 200, S1 800 + 900, S2 1 900 + 1 700 (+ 300 Metal), S3 500 + 800, S4 2 000 + 1 500 (+ 600 prototype on top of the 300 that exist), S5 2 600 + 1 800,
S6 3 500 Swift + tests, S7 docs and scripts.

---

## 11. Owner decisions (the coordinator may take the recommendations under delegation and record them as revisitable choices)

| # | decision | recommendation | consequence if otherwise |
|---|---|---|---|
| D1 | grid | **HEALPix** for the fields, the hash and the agents (equal area, closed-form lookup, tables); cubed sphere only as a later hash optimisation. **The weather mesh is undecided until S4a** (variant B on HEALPix is tried first; TRiSK only if 4.5 triggers), and `Nw = r·Nf` is not a constraint on D8 | cubed everywhere: 17 % density bias pattern or per-pair area weights; ~24 % cheaper hash at reach 1 |
| D2 | 2D sphere | **3D shell only**; a surface world is a thin shell | a 2D sphere is a new state class and GUI path |
| D3 | shell thickness and presets | **`tiny3s` 36/16, `mini3s` 48/24, `64k3s` 90/40**; thin shell (`H ≤ Rp`); populations 6 to 42 k. Each shell is thinner than the box analogue (24, 80, 80), so the 3D ecology (buoyancy layering, benthic/photic profile) is **not comparable** to RESULTS_3D | a thicker shell (H 80 on Rp 90: top area 3.6×, volume 17.5 M) costs 2 to 3× in search and population |
| D4 | atmosphere vs biosphere in the weather (**changed**) | **keep `H1 + H2 = 80` and map the shell onto it** (4.2): wind profile, interface, cloud and rain/seeding bands at `z·(H1+H2)/H`; the rule `H1 + H2 = height` is dropped for a sphere. Relaxing the rule alone (the first draft) leaves tiny3s/mini3s without rain food | rescale H1, H2 to the shell: deformation radius halves, retuning S4; rain defined directly on the shell is the alternative (also needs writing) |
| D5 | lid and gravity | **hard radial lid** plus the optional soft band (default off); **constant gravity** | inverse-square gravity changes the vertical ecology across the shell (0.44 at the lid) |
| D6 | migration | **none** (as instructed); an optional genome re-seed tool is a later, separate decision | |
| D7 | hash reach (**decided now**) | **reach 2 only**, no `hash_reach` key; `Nh` and `C` derived from exact tables; S2 stop rule at 2× mini3d per alive agent (2.4) | reach 1 costs 2.6× the box and needs `C` of 120+ |
| D8 | weather risk | **gate S4a (G1 to G7, non-circular), fallback TRiSK with the trigger and the cost of 4.5; let the weather trail the dry sphere** | holding the whole sphere for the weather delays the globe |
| D9 | GUI | **separate sphere renderer, no Mercator/torus morph for sphere worlds**; Mollweide and azimuthal charts for flat views; box paths kept | adapting the morph would keep a fake in the new world |
| D10 | dependencies and licences | **own HEALPix implementation; astropy-healpix (BSD-3) dev-only test oracle (a convenience: the reviewer's independent 100-line implementation with exact combinatorial neighbours agrees); no healpy (GPL-2.0), not even as oracle** | adding astropy to the dev group needs a yes |
| D11 | names | `tiny3s`, `mini3s`, `64k3s`, overlay `stormsph`, config `world.geometry` | |
| D12 | tessellation frame (**dropped**) | no tilt: `Q` is the identity; the seams are fixed in the scheme and judged by the imprint gate G5 (1.4) | |
| D13 | vertical mixing in the resource | **geometric (physical-isotropic) weights**, stated honestly (vertical exchange 2× to 9× the horizontal); `fields.vdiff` is held in reserve if S7 finds the benthic/photic profile washed out (3.1) | normalising by the maximum row sum keeps `d_q8 ≤ 42` meaning but slows horizontal diffusion 2.7 to 3.5× |
| D14 | compile-collision guard | **merge `fix/mlx-compile-collision` (I53) before S2**; fallback in 10.1 | the fallback costs subprocess-per-config tests |

---

## 12. DESIGN §46 outline and the Review-log row

§46 The sphere world: 46.0 overview and decision record (section 0), 46.1 geometry and tables (1), 46.2 state and motion (2.1 to 2.3), 46.3 neighbour search and
the kernel (2.4, 2.5), 46.4 sensing, birth, exactness (2.6 to 2.8), 46.5 fields and the five features (3), 46.6 weather (4), 46.7 observers, stream v2, dumps
(5), 46.8 GUI (6), 46.9 configuration, validation, presets, compatibility (7), 46.10 measurements and sizing (8, probes), 46.11 tests, 46.12 the consumer guard (2.9). §27 gains a cross-reference line;
§17 the new keys; §20/§36 a note that the Mercator globe is box-only; §22 the sphere fixtures; §16 the new modules (`geom.py`, `sphere.py`, `weather_sph.py`).

Review-log row I77 (to be written with S0, extended by each PR): "feat: the sphere world (§46): `world.geometry = "sphere"`, a HEALPix shell with a radial
gravity and a hard lid, Cartesian state with a tangent heading, equal-volume layers; box worlds bit-identical (identity matrix); no torus-to-sphere migration".
CLAUDE.md gets one sentence in its first paragraph (docs/SPHERE.md and §46).

### Open questions that only a run can settle

1. The real cost of the reach-2 stencil in the Metal kernel (125 runs per thread against 27): only the S2 benchmark can tell; the estimate brackets it and the stop rule protects the plan.
2. Whether the collocated weather reproduces the STORMS_W0 storm statistics after retuning the damping (the sphere advects 2 to 2.3× less diffusively, Rossby waves run 6 to 10 % slow): S4c.
3. The population per unit volume in a thin curved shell, and the resource profile under 2 to 9× faster vertical mixing: `regen_max`, `C` and D13 are provisional.
4. Whether the 10 % divergence error at the 8 pole-vertex cells (4.3) matters for the dynamics: G1 reports it; a polar vortex would show it.
5. The slow Rossby propagation: compensate in S4c or accept.


---

## Revision 1 (2026-10-29): changelog against `review.md`

Method: every number in dispute was re-measured, not accepted from either side. New probes in `rev1/`: `wproto.py` (float64 weather-core prototype on the reviewer's independent HEALPix `hp.py`: static divergence, noise stability,
one-step eigenvalues, Williamson 2), `rh4.py` (Rossby-Haurwitz 4), plus the reviewer's `t2_cov.py`, `t3_cond.py`, `t3b_lap.py`, `t6_colour.py`, `t7_tilt.py`, `t9_dmin.py`, `t10_patch.py` re-run. Logs: `rev1/stab42.log`, `rev1/tc2_48.log`.

### Findings

| ID | verdict | change |
|---|---|---|
| **H1** divergence inconsistent at the cap seams | **accepted, reproduced** (plain max 0.31 / 0.33 / 0.34 at Nside 6 / 12 / 24, not converging) | 4.1 defines the true edge normal, the edge tables, gradient-corrected face values for `u` and `h`; 4.3 gives the table (corrected max 0.028 / 0.019 / 0.0107, rms second order); gate G1 over **every** cell. New finding recorded: a constant 10 % error at the 8 pole-vertex cells for `∇z` (4.3), gated as bounded |
| **H2** unstable without Rhie-Chow, wrong Courant limit, circular gates | **accepted in substance; two numbers disputed and corrected both ways.** Plain scheme unstable at every Courant: reproduced (29 to 174 steps). *The reviewer's remedy was not enough*: Rhie-Chow with gradient-corrected faces still has a real eigenvalue 1.004 to 1.007 per step at every Courant (a ±1 mode on 8 cells at \|z\| = 0.5), unstable in 500 to 840 steps. *A weak biharmonic filter (`a4 = 1e-3`) is part of the scheme*: stable to C = 0.6 at Nside 12, 24, 42; unstable at 0.65. The reviewer's limit 0.83 is the pure compact operator's; **measured full-operator limit ≈ 0.62**, margin of the presets 1.3 | 4.1 (the step, with the exact Rhie-Chow form including the `(e·n)` factor and `kappa = 1`), 4.3 (stability table), 4.4 (gates G1 to G7 with analytic/published references: TC2, Rossby-Haurwitz phase speed, divergence, noise stability, imprint; G7 labelled self-referential), 4.5 (the fallback trigger) |
| **H3** rain/seeding vertical placement for every H | **accepted, confirmed by reading `cloud_weights`, `rain_regen`, `seeding_response`, `sample_wind`**; the first draft's D4 relaxed a rule and left tiny3s/mini3s without rain food | 4.2: the vertical map `s = (H1+H2)/H`; table of every mapped quantity; `WeatherGeometry` helper (no `WeatherParams` field, hash-neutral); S4 audit tests for every preset; D4 changed |
| **H4** silent wrongness in the `[:,2]`/`min_image` consumers | **accepted** (grep counts reproduced: z/vz in about 25 files, `min_image`/`wrap` in 20, `world.size` 28, `world.height` 24, lattice shape 40) | new 2.9: accessors, geometry lint with a ratchet allowlist, rotation-equivariance test, `SPHERE_READY` readiness gates, per-consumer audit table with the converting PR (includes `invariants.py`, `game/observe.py`, `analysis/labels.py` + the Nomad track + `INTENT_VERSION`, `render.py`, `viewer.py`, `bench.py`, `weather_spinup.py`, `analysis/w0.py`, `cli.py`, `runio.py`, `migrate.py`, the emergence tools) |
| **H5** int32 overflow in the diffusion | **accepted, reproduced** (row sums 5 233 / 5 329 / 4 189 against 1 536; radial weights 510 to 2 269) | 3.1 rewritten: int64 flux, `Σα ≤ 0.9` rule (`d_q8 ≤ 11 / 11 / 14`, the reviewer's 12 / 12 / 15 used `Σα ≤ 1`), the cap bound with its proof structure and the exhaustive `check_diffusion_tables`, corner tests, the honest statement that vertical exchange is 2 to 9× the horizontal (D13), the chemical sub-stepping (also needed: the box's `D_c ≤ 1/6` rule fails by 3.3×) |
| **M1** coverage constants optimistic | **accepted, reproduced** (exact 1.385 at Nside 21 vs the probe's 1.45; reach 1 0.685 vs 0.70) | constants deleted; `Nh` derived from the exact table with the rigorous angle (1.5, 7.2); S0 computes and tests it; margins 12.0 / 6.8 / 2.3 % reproduced |
| **M2** patch and pool drift | **accepted, reproduced** (the patch parks at 90°) | patches rotate rigidly about a stored ω (2.1, 2.7); pools: `vy = 0` required, centres within 60° (4.1) |
| **M3** tilt rests on the wrong feature | **accepted, reproduced** (belt on seam cells 5.1 to 8.4 % for every tilt; grid pole at ±30° for tilt 60) | tilt and D12 dropped, `Q` removed from the lookup, the tables, the stream header (1.4, 11) |
| **M4** R9 unsatisfiable, colouring claim false | **accepted, reproduced** (spread −17 to +23 %; colours 15 / 16 / 18) | gate restated (3.1); colouring text and `n_phase` as data (2.6) |
| **M5** compile-collision hazard | **accepted** (coordinator's instruction as well) | 10.1: prerequisite of S2 (and of compiled sphere tests that parametrise a constant), with a four-point fallback; R7; D14 |
| **M6** `test_gui_sync` blocks S2 | **accepted** | `SPHERE_PRESETS_PENDING_GUI` exemption in S2, removed in S6; `SPHERE_KEYS` lever test in S0 (7.1, 7.4, 10) |
| **M7** search-cost arithmetic and `C` | **accepted, arithmetic reproduced** (reach 2: 34.2 R³ = 1.27× at 64k3s; reach 1: 70.6 R³ = 2.6×) | 1.3 finding 2, 2.4, 8 corrected; **D7 decided: reach 2 only**, `hash_reach` key removed (the probe agrees: no reason to keep reach 1); `C` derived from `V_lid` (presets 32); the S2 stop rule |
| **M8** TRiSK fallback not costed; S4a prototype scope | **accepted** | 4.5: trigger, cost (S4 again, 4 to 5 PRs), variant B tried first, prototype scope (600 lines on top of 300, not a port of `weather.py`) |
| **M9** plan gaps | **accepted** | S2 now lists the `bench.py` scene builder, `invariants`, `cli`, `runio`, `migrate` refusal, `weather_spinup` (S4); `WeatherParams` hash neutrality (7.2); the flat-limit test corrected (7.4: no `validate`, fields off, tolerance 1 % of the step); the format note (7.4); the G1 observer in the audit (2.9) |
| **L1** vertical profile | accepted | `exp(−z_k/λ)` at the lower interface, `V[0] = 1` (1.5) |
| **L2** heading range | accepted | `+π → −π` (2.1) |
| **L3** transport axis | accepted | `k = s·(û × w)`, `κ = c` (2.3) |
| **L4** wind at the agent | accepted | re-projected on the agent's tangent plane (4.1) |
| **L5** `world.size` default | accepted in part | no geometry-dependent default (rejected: it would hide the rule); the default 1 024 works up to `r_top = 255`, larger planets set `size`; `validate` skips the box-derived rules (2.5, 7.2) |
| **L6** `vel` is the commanded step | accepted | documented (2.3) |
| **L7** chord vs arc | accepted | chord everywhere (patches 3.1, pools 4.1) |
| **L8** signed zero | accepted | by-value comparison in the test (2.5) |

### Coordinator directives

* **D7 = reach 2 now**: the probe agrees (above). **D12 dropped** (M3). **M5 stated as a prerequisite** of S2 with a fallback (10.1).
* **S0 readiness**: yes. S0 (10, row S0) depends on nothing still open: the grid, the exact tables, the exact coverage, the layers, the accessors and the lint, the config keys and validation are all fixed by this revision. What S0 must **not** freeze: the diffusion tables and bounds are S1's, the weather tables (edge normals, `δ`, fan inverses) are S4's, though the exact HEALPix neighbour/corner machinery they need is S0's.

### Rejected or deferred

* The reviewer's suggestion to add a θ-parameterised Rhie-Chow share: replaced by the full term plus the filter, because the θ analysis was linear and the nonlinear scheme still needed the filter (H2 row).
* A geometry-dependent default for `world.size` (L5), as above.
* The icosahedral/TRiSK fallback is not costed lower than the reviewer's reading (M8): accepted as S4 again.
* A fully **a priori** G3/G4: the thresholds are literature/physics levels but were set after the prototype existed; the table says so and shows the margins (5× and 1.2×). A reviewer who wants G4 stricter will see the 1.2× margin at Nside 24.
* Not re-run: the reviewer's `t5_mlx_ang2pix.py` (float32 lookup, CPU and GPU agree) and `t11_minimage.py` (GPU), both accepted as stated; no GPU was used in this revision.

### New measurements added in revision 1 (not asked for)

Rossby-Haurwitz phase speed 6 to 10 % slow at the presets' `Nw`; the TC2 α = 45° pitfall (Coriolis axis must tilt with the flow); the 1.004 eigenvalue and its cure; the chemical stability failure (3.3×) of the box's `D_c`; the pole-vertex divergence error.
