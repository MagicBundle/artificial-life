# Research directions: the computational study of emergence

Status as of 2026-10-11. This note assesses add-on research projects for `alife` against prior work.

**Inputs.**
- A local LLM (Qwen 3.8) proposed eight candidate projects, P1–P8.
- Four literature scans covered prior work: digital-evolution platforms, emergence and open-endedness metrics, cooperation and signalling studies, and AI agents driving ALife research.

**Evidence caveat.** The scans could not fetch full texts. Every source below comes from search snippets, abstracts or secondary coverage, and figures marked (unverified) are second-hand. Read the primary source before citing anything. Also run an arXiv full-text search before making any public novelty claim.

## 1. Position against prior work

GPU neural-agent ecologies already exist:
- **JaxLife** reports rudimentary communication, agriculture and tool use, with complexity growing with compute.
- **EcoEvoJax** runs evolvable recurrent nets on a grid, with thousands of agents.
- **Coralai** evolves NEAT networks inside a neural cellular automaton.
- **ALIEN** runs CUDA particle organisms.

A claim of emergent communication or agriculture is therefore not new. It would have to be framed against JaxLife.

No platform found combines the following:

| feature | nearest prior work |
|---|---|
| exact integer energy ledger with a closure test | none. Avida's "merit" and EcoEvoJax's energy are accounted but not conserved. Flow-Lenia conserves mass by design. ALIEN's conservation is only qualitative. |
| bit-exact deterministic replay | Coralai saves the RNG state in checkpoints, but bit-exact reproducibility is not claimed |
| an environment-only director acting during a run | ASAL++ (an LLM picks Lenia targets, scored by a vision-language judge) and the ASAL outer loop (a parameter search before the run) |
| director decisions logged and replayable | none found |
| LLM director compared against random or parameter-sweep baselines on the same budget | none found. ASAL and ASAL++ report no such controls. |
| per-agent recurrent controllers in continuous space with bonds, bodies and a ledger | each piece exists separately (Polyworld, ALIEN, Coralai). No single prior work combines them. |

**The novelty is the combination, not any single feature.** The most defensible contribution is the environment-only director:
- its actions are checked against the ledger;
- its decisions are replayable;
- it is compared with matched non-LLM baselines.

That is P6 below, and most of it is already built (`docs/DIRECTOR.md`, steps 1–3).

The ledger also matters as a safeguard against reward hacking, a documented failure of self-modifying agents. A Darwin Gödel Machine lineage removed its own hallucination-detection markers to reach a perfect score. A ledger that must close exactly rules out a whole class of such hacks by construction.

## 2. The candidate projects

| | project | prior work | open gap | fit with this engine |
|---|---|---|---|---|
| P1 | Energy economics of cooperation: per-body and per-clade energy flows from the ledger | Goldsby et al. 2014 ("dirty work", Avida); Babajanyan et al. 2024 (biofilm ABM). Both measure frequencies, not energy flows. | real: no platform has a ledger that closes | the ledger is global (16 terms), so per-body attribution needs per-agent flow counters. That is engine work with a new ledger test. |
| P2 | Multicellularity and the cheater problem | well covered: Goldsby 2012/2014; Moreno & Ofria 2022; Babajanyan 2024; "Games of multicellularity" (arXiv 1605.07690) | small: a cheater load defined by the ledger, with relatedness measured from the genealogy rather than assumed | **blocked:** in `64k_s0` sharing fell to 1.6% and bonds last about 12 steps, so no bodies exist to be cheated. A world that sustains bodies is needed first. |
| P3 | Emergent signalling and honesty | Floreano et al. 2007 and Mitri et al. 2009/2010 (evolved robots; honesty tracks relatedness); Macmillan-Scott & Musolesi 2025 (costly signalling does not guarantee honesty); Kane & Zollman 2015; JaxLife | low as a headline result | cheap as a diagnostic. Mitri 2010 found that signal frequency predicts reliability poorly, so "emitters 81% → 26%" says nothing about honesty. |
| P4 | Open-endedness and novelty | Bedau–Packard activity statistics (Bedau et al. 1998); MODES (Dolson et al., Artificial Life 25(1), partly Avida-specific); Channon on normalisation | medium: applying these to a continuous neural ecology with director arms | **feasible now:** `events/` logs births with `parent_uid`, so the phylogeny can be rebuilt offline (DESIGN §12.8) |
| P5 | Phase transitions and criticality | Chorowski et al. 2024; Sakiyama 2023; for pitfalls, Touboul & Destexhe 2010 and Clauset et al. 2009 | medium | expensive (parameter sweeps × seeds), with a hard methodology. Defer. |
| P6 | Frontier LLM as an offline director writing `director.jsonl` | ASAL++; OMNI-EPIC; Eureka | **clearest gap** (§1) | steps 1–3 are built. The evaluation remains, and it needs baselines (§4). |
| P7 | Frontier LLM as a rule-evolver that rewrites engine code | Darwin Gödel Machine; AlphaEvolve; FunSearch; OMNI-EPIC | low novelty, high cost (one DGM run is reported at about $22k, unverified) | letting an LLM write engine code breaks the ledger and determinism guarantees that make P6 credible. Not recommended. |
| P8 | Causal emergence and coarse-graining | Hoel et al. 2013 (effective information); Rosas et al. 2020 (ΦID); IIT/PyPhi | open but contested: coarse-grainings are not unique, PID axioms conflict, and Φ is infeasible beyond about 12 nodes | viable only on 10–50 coarse-grained macro variables, reusing P4's data. A secondary analysis. |

## 3. Measurement rules

The project already has two metric artefacts:
- `weight_diversity` before step 120k is unreliable;
- the kill share undercounts delayed combat deaths.

The literature adds these rules:

- **Continuous metrics, with sensitivity reported.** Discontinuous thresholds manufacture apparent transitions (Schaeffer et al., NeurIPS 2023). Report sensitivity to every threshold, every normalisation and every component definition.
- **Bias-corrected information measures.** Plug-in mutual information and transfer entropy are biased upward on finite samples. Use KSG/kNN or Miller–Madow estimators against shuffle or surrogate nulls. Use conditional transfer entropy where common drivers exist, such as a shared rich patch.
- **No all-pairs measures at full population.** All pairs over 10⁴ agents is about 5·10⁷ pairs. Subsample, or coarse-grain to macro variables (niche densities, trait means).
- **No criticality from a single fit.** Use Clauset maximum-likelihood fits, KS and likelihood-ratio tests on untruncated event logs, plus threshold, bin-width and subsampling sweeps. Prefer crackling-noise scaling relations to a single exponent.
- **Measure relatedness from the genealogy; do not assume it.**
- **Compute offline from event logs** so a new metric needs no re-run. Any new in-step logging goes in a preallocated device buffer behind a config key that defaults to off (`config.POST_V01_KEYS`).
- **Report across seeds.** A single trajectory is an anecdote.

## 4. Recommended order

1. **P4, offline.** Compute Bedau–Packard activity statistics on `64k_s0` from `events/`, normalised against a neutral shadow. This classifies the stall after about 1M steps as no, bounded or unbounded activity. Clade is too coarse a component (4 remain), so use lineage subtrees, quantised tag vectors, or both. The result becomes the outcome metric for every later experiment. Done: `docs/ACTIVITY.md` (`alife activity`): bounded activity in all three windows; lineage components only, since tags are not recorded per birth; forks are not yet readable by the tool.
2. **P3 diagnostic, offline.** Estimate mutual information between each signal channel (`emit`, `msg`) and resource, energy, kin and next action at several checkpoints, against a shuffle null. Done: `docs/SIGNALS.md`: the remaining signals report local crowding and the coming stop (up to 0.34 bits), a little resource, almost no energy or kin and no attack intent; information rose while emitters fell. The receiver side, which honesty needs, is not done.
3. **P6, the DIRECTOR.md §9 evaluation, with two additions.** Report the step-1 activity statistics alongside archive coverage and persistent novelty. Then add:
   - **A matched random-director arm.** It uses the same menu, budgets and cooldowns, with actions drawn uniformly. Without it, "the director helps" cannot be separated from "any perturbation helps", which is the gap noted in ASAL and ASAL++.
   - **At least a second seed per arm**, since a single-seed comparison is not evidence.

   The owner accepted both on 2026-10-11, and they are now in DIRECTOR.md §9 and §10. Each arm and seed costs
   about 6 hours on the M5 Pro.
4. **P1, then P2.** Add per-agent flow counters with a ledger-closure test. Pursue P2 only if some director arm sustains bodies.
5. **P5 and P8** as later analyses on the same data.
6. **P7:** not pursued.

## 5. Sources

All sources were found via search; full texts were not read.

**Platforms**
- Tierra: https://tomray.me/pubs/tierra/node8.html
- Avida: Lenski, Ofria, Pennock & Adami, Nature 2003, https://lenski.mmg.msu.edu/papers/nature2003/index.html
- Polyworld: Yaeger & Sporns, ALife XII 2010, https://www.shinyverse.org/larryy/YaegerEtAl2010_EvolutionarySelectionOfNetworkStructureAndFunction_ALife12.pdf
- Lenia: Chan 2023, https://arxiv.org/abs/2304.05639
- Flow-Lenia: https://arxiv.org/abs/2212.07906 ; https://arxiv.org/pdf/2506.08569
- Leniabreeder: https://arxiv.org/pdf/2406.04235
- ALIEN: https://github.com/chrxh/alien
- Coralai: https://arxiv.org/abs/2406.09654
- EcoEvoJax: https://arxiv.org/pdf/2302.09334
- JaxLife: https://arxiv.org/pdf/2409.00853
- Neural MMO: https://arxiv.org/abs/1903.00784
- Neuroevolution Arena (2026): https://arxiv.org/pdf/2608.10323

**Metrics**
- Hoel, Albantakis & Tononi, PNAS 2013: https://pmc.ncbi.nlm.nih.gov/articles/PMC3856819
- Rosas et al., PLOS Comput Biol 2020: https://arxiv.org/pdf/2004.08220
- JIDT: https://github.com/jlizier/jidt
- IIT limits: https://arxiv.org/pdf/1712.09644 ; https://arxiv.org/pdf/1902.04321
- MODES: https://peerj.com/preprints/27249v2
- Channon: https://gpbib.pmacs.upenn.edu/gp-html/Channon_2001_PAT.html ; https://www.tim-taylor.com/papers/channon2024editorial.pdf
- Assembly-theory critique: https://arxiv.org/abs/2403.06629
- Shalizi & Crutchfield: https://arxiv.org/abs/cond-mat/9907176
- Clauset, Shalizi & Newman: https://arxiv.org/abs/0706.1062
- Schaeffer et al.: https://arxiv.org/abs/2304.15004

**Cooperation, signalling and criticality**
- Goldsby 2014: https://biologue.plos.org/?p=15822
- Moreno & Ofria 2022: https://arxiv.org/pdf/2104.10081
- Babajanyan et al. 2024: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11722410/
- "Games of multicellularity": https://arxiv.org/pdf/1605.07690
- Floreano et al. 2007: doi 10.1016/j.cub.2007.01.058
- Mitri, Floreano & Keller 2010: https://infoscience.epfl.ch/record/150720
- Macmillan-Scott & Musolesi 2025: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12440204/
- Kane & Zollman 2015: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4562630/
- Chorowski et al. 2024: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11303522/
- Touboul & Destexhe 2010: https://arxiv.org/abs/0910.0805

**AI agents in ALife research**
- ASAL: https://arxiv.org/pdf/2412.17799
- ASAL++: https://arxiv.org/html/2509.22447v1
- AI Scientist critique: https://arxiv.org/pdf/2502.14297
- OMNI-EPIC: https://arxiv.org/pdf/2405.15568
- Eureka: https://arxiv.org/abs/2310.12931
- FunSearch: https://github.com/google-deepmind/funsearch
- Darwin Gödel Machine: https://arxiv.org/pdf/2505.22954
- LLM rule-setters in ecological simulations: https://arxiv.org/abs/2505.05863 ; https://arxiv.org/html/2510.21965v1
