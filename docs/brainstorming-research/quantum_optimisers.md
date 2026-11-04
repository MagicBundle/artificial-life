# Quantum-inspired optimisers on classical hardware: verified projects

Verified via GitHub pages and the GitHub API (`pushed_at`) on 2026-11-03. "Last activity" is the year of the last push.
Relevance is judged against an ecology of quantum walkers on an objective landscape with an exact evaluation ledger.

| Project | URL | Language | License | Last activity | What it does | Relevance to walker ecology + exact ledger | Verdict |
|---|---|---|---|---|---|---|---|
| dwave-samplers | https://github.com/dwavesystems/dwave-samplers | Python (C++ core) | Apache-2.0 | 2026 | Classical BQM samplers: SA, simulated quantum annealing (SQA), tabu, steepest descent, random, planar exact, tree decomposition | Good classical baseline set, including an SQA sampler; no walker ecology; ledger/read-count accounting not verified | reuse (baseline) |
| OpenJij | https://github.com/OpenJij/OpenJij | C++ core, Python API | Apache-2.0 | 2026 | Ising/QUBO framework: SQA, continuous-time SQA (CSQA), SA, HUBO support | Clean reference for SQA samplers with a QUBO API; no evaluation ledger | reuse (baseline) or reference only |
| simulated-bifurcation-algorithm (bqth29) | https://github.com/bqth29/simulated-bifurcation-algorithm | Python (CPU/GPU via PyTorch) | MIT | 2026 | Simulated bifurcation for QUBO/Ising/TSP/portfolio; not the Toshiba reference code | Only verified open SB implementation; no walkers; useful as a comparison solver | reuse (baseline) |
| dwave-neal | https://github.com/dwavesystems/dwave-neal | C++ with Python (dimod) wrapper | Apache-2.0 | 2022 (last push); repo updated 2026 | Simulated annealing sampler for Ising graphs | Simple SA baseline; the repo's last push is 2022, and a search snippet (PyPI) says it is deprecated in favour of dwave-samplers (README on GitHub does not say so) | reference only |
| IOHexperimenter | https://github.com/IOHprofiler/IOHexperimenter | C++ with Python interface | Not identified (LICENSE.md linked; GitHub reports "Other") | 2026 | Experiment framework for iterative optimisation heuristics: problem suites (BBOB, PBO), logging to IOHanalyzer | Closest verified tool to an exact evaluation ledger: per-run logging of evaluations and targets; the README does not confirm per-evaluation counting in so many words, so check the logger source | reference only (check licence before reuse) |
| COCO (numbbo/coco) | https://github.com/numbbo/coco | C with Python | Not identified (LICENSE linked; GitHub reports "Other") | 2026 | Benchmarking platform for black-box continuous optimisers; runtime measured in f-evaluations | Strongest methodological reference for budget-honest reporting (ERT over f-evaluations, budget-free protocol); continuous only, no Ising/QUBO; README notes code has moved to other numbbo repos | reference only |
| pathintegral-qmc (hadsed) | https://github.com/hadsed/pathintegral-qmc | Python (Cython) | None stated (GitHub licence field null) | 2017 | Path-integral QMC simulation of quantum annealing on arbitrary Ising Hamiltonians (Martonák, Santoro, Tosatti 2002 method) | Reference implementation of Trotterised SQA; stale and unlicensed, so reading only | reference only |
| piqmc (therooler) | https://github.com/therooler/piqmc | Python with Cython sampler | "Do No Harm" (non-OSI, per README) | 2021 | Classical SA and PIMC simulated quantum annealing for spin models (2D EA, SK, Wishart planted); code for arXiv 2101.10154 | Second SQA reference with experiment scripts; the licence restricts reuse | reference only |
| ising-optimisation (harveynw) | https://github.com/harveynw/ising-optimisation | Python (inferred) | Not stated | Not determined (page shows no date) | Toy library: Trotterised quantum annealing (`sim_quantum_anneal`) and classical annealing | Small educational SQA code; no budget accounting | reference only |
| QWMO (OmerSamuk) | https://github.com/OmerSamuk/QWMO | Python | MIT | Not determined (page shows no date) | Quantum-inspired population optimiser for multimodal landscapes with three probabilistic operators balancing exploration and exploitation | Closest verified match to a population of walkers on a landscape; accompanies a June 2026 Zenodo preprint; not peer-reviewed as far as checked | reference only (check method against preprint) |
| QuantumWalk.jl | https://github.com/iitis/QuantumWalk.jl | Julia | MIT | 2020 (last push); updated 2026 | Toolkit for discrete and continuous-time quantum walks, Szegedy walks, spatial search, optimal measurement time | Quantum-walk substrate for a walker prototype; not an optimiser over a landscape | reference only |
| Hiperwalk | https://github.com/hiperwalk/hiperwalk | Jupyter Notebook (Python package) | LGPL-3.0 | 2026 | HPC simulator for quantum walks on graphs (optional CPU/GPU) | Possible engine for simulating walks on graph landscapes; copyleft licence; not an optimiser | reference only |
| qwopt | https://github.com/Chibikuri/qwopt (the fetch resolved to https://github.com/cocori-rs/qwopt; check owner) | Jupyter Notebook | Apache-2.0 | 2023 | Rule-based optimiser for Szegedy quantum walks | Only verified quantum-walk optimiser found; stale and small | unrelated (stale) |

## Papers without verified code (flagged, not in the table)

- VeloxQ: a fast and efficient QUBO solver, arXiv 2501.19221. Benchmarks against D-Wave, Kipu Quantum, brute force and branch-and-bound. No repository was verified.
- Comprehensive benchmark of an Ising machine on Max-Cut, arXiv 2507.22117. Uses MQLib instances (2025). No repository was verified.
- Kanao and Goto (2022), simulated bifurcation, Communications Physics. The algorithm is open via the bqth29 package, but no official reference code was verified.
- Scientific Reports 2022, benchmark of D-Wave HSS, Toshiba SBM and Fujitsu Digital Annealer against neal (arXiv 2104.14096). Commercial solvers are closed. The study's data and code were not verified.
- Microsoft Quantum Inspired Optimization (QIO) solver. A 2023 Azure blog says it went on GitHub under MIT with KPMG. No repository URL was verified, so it is not in the table. Substochastic Monte Carlo and population annealing are described in a search snippet, but not confirmed from primary sources.

## Notes for the ecology-of-walkers design

- Few of these projects offer evaluation-budget accounting. COCO (budget-free, f-evaluation runtime) and IOHexperimenter (per-run logging) are the most useful references for an exact ledger. Neither covers Ising/QUBO directly.
- No verified project combines walker ecologies with a quantum-walk substrate. QWMO is the nearest to walker populations on a landscape.
- Unresolved: the licences for IOHexperimenter and COCO (LICENSE files linked, but the GitHub API reports "Other"/NOASSERTION), the status of dwave-neal relative to dwave-samplers, and the owner of the qwopt repository.
- Claims about recent maintenance from third-party indexers were not used. Years here come from the GitHub API.
