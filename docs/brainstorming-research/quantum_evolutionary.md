# Quantum-inspired and dynamic-optimisation open-source survey (verified 2026-11-03)

Target: an ecology of evolved quantum walkers that tracks moving optima under an exact evaluation budget, compared against random search and a classical twin.

Verification: each repository below was fetched (HTML page and/or GitHub API `pushed_at`, `license`, `language`). "Last activity" is the year of the last push as reported by the API. Star counts are from the API.

## Verified entries (10)

| Project | URL | Language | License | Last activity | What it does | Relevance to target | Verdict |
|---|---|---|---|---|---|---|---|
| DEAP | https://github.com/DEAP/deap | Python | LGPL-3.0 | 2026 | Distributed evolutionary algorithms framework; includes a classic Moving Peaks generator in `deap.benchmarks.movingpeaks` (6456 stars) | Exact-budget GA/ES baselines and a classic moving-optima generator; LGPL needs care if code is copied | reuse (baselines and MPB generator) |
| EvoTorch | https://github.com/nnaisense/evotorch | Python (PyTorch) | Apache-2.0 | 2026 | GPU-capable evolutionary computation library on PyTorch (1151 stars) | Strong GA/ES/random-search baselines, but PyTorch dependency does not fit the MLX pipeline | reference only |
| EDOLAB | https://github.com/EvoMindLab/EDOLAB | MATLAB (Octave port) | GPL-3.0 | 2026 | Platform for evolutionary dynamic optimisation: about 25 to 27 dynamic optimisers and four benchmark generators (GMPB, MPB, FPs, GDBG) (65 stars) | Best reference for the dynamic-benchmark protocol and generator definitions; the GPL and MATLAB stack rule out direct reuse | reference only (protocol and definitions) |
| gmpb (cmudrc) | https://github.com/cmudrc/gmpb | Python | MIT | 2026 | Python implementation of the Generalized Moving Peaks Benchmark (fully non-separable, 15-problem suite from arXiv:2107.11019) (1 star) | Candidate generator of moving optima with a realistic landscape; MIT allows reuse | reuse (candidate generator) |
| pateda | https://github.com/rsantana-isg/pateda | Python | MIT | 2026 | Python implementation of the MATEDA suite of estimation-of-distribution algorithms, including UMDA and BOA (2 stars) | Classical probabilistic-model baseline (UMDA, BOA) for the classical twin | reuse (classical baseline) |
| qpso (ngroup) | https://github.com/ngroup/qpso | Python | MIT | 2019 (last push) | Python implementation of quantum-behaved PSO (QDPSO, delta potential well) (22 stars). Stale: PyPI 0.0.1 from 2018 | Closest code to the QPSO dynamics; too small and stale to build on, useful as a reference for the update rule | reference only |
| IspML/QuantumGA-Knapsack | https://github.com/IspML/QuantumGA-Knapsack | Python | none listed | 2019 (last push) | Fork of mjBM/Quantum-Evolutionary-Algorithm-Knapsack-Python; classic QEA with Q-bit observe and repair, credits Han and Kim (0 stars, 3 commits) | Han and Kim QEA on a static knapsack; a reference for the Q-bit update, but no licence and no dynamic support | reference only |
| Quantum-Inspired-Evolutionary-Algorithm (lccalluchi; found as cheems-dev) | https://github.com/lccalluchi/Quantum-Inspired-Evolutionary-Algorithm | C++ | none listed | not shown (screenshots dated 2021) | QIEA for the knapsack problem, an algorithm-design course project (36 commits) | Little beyond the static knapsack; the search URL redirected to a different owner | unrelated (reference at most) |
| Anpy (davall16; found as Daniels1610) | https://github.com/davall16/Anpy | Python | MIT | 2023 | OOP library for ant colony optimisation applied to the 0/1 knapsack (0 stars, 18 commits) | Classical ACO baseline on a knapsack; not dynamic, and the search URL redirected to a different owner | reference only |
| swarmpy_tsp | https://github.com/g0bel1n/swarmpy_tsp | Python | MIT badge, no LICENSE file (unconfirmed) | not shown | Pipeline-style ACO library for the travelling salesman problem (2 commits) | Not relevant beyond a generic ACO baseline | unrelated |

## Papers or resources without verified code

- **Han and Kim, quantum-inspired evolutionary algorithm (2000 CEC, 2002 IEEE TEC).** No official code release was found. The QEA definition is the basis for the entries above. Flag: paper without code.
- **Generalised Dynamic Benchmark Generator, Li and Yang.** Covers binary, real and combinatorial spaces. No repository was located or verified. Flag: paper without verified code.
- **Dynamic MKP benchmark, Skackauskas and Kalganova (Systems and Soft Computing 2022, DOI 10.1016/j.sasc.2022.200041).** A deterministic method generates 1405 dynamic multidimensional knapsack datasets (State000 to State100). The datasets are on IEEE DataPort (DOI 10.21227/6bfm-bj82). The paper cites a generator and datasets on GitHub, but the URLs were not located. Flag: paper with data; generator code not verified. Relevant to a dynamic knapsack setting.
- **barcaroli/QGA (R, CRAN) and Lahoz-Beltra QuantumGeneticAlgorithms (Python, ResearchCodesHub).** The R package is described as a QGA first proposed by Han and Kim (2000). The repositories were not fetched. Unverified.
- **StarPSO (QuantumPSO engine, Xi, Sun and Xu variant).** A Python library described in a 2026 paper. The repository URL was not verified. Unverified.
- **CILib (Java, SourceForge) and rlevo-evolution (Rust, docs.rs).** Coevolution frameworks described in documentation only; repositories not fetched. Unverified.
- **L2L (igitugraz) and rust-es-optimizer (FlixCoder, archived).** Evolution-strategies optimisers from the Salimans et al. paper. Not fetched beyond the search snippet. Unverified.

## Gaps

- No mature, verified open-source competitive or adversarial co-evolution library was found.
- No verified "learning to optimise by evolution" codebase beyond the evolution-strategies optimisers above.
- pyEDA, pymoo and EDAspy were not verified.
- The GitHub HTML pages did not show commit dates; the year column comes from the GitHub API `pushed_at` field.

## Notes

- Three URLs redirected to a different owner (EDOLAB-platform to EvoMindLab, Daniels1610/Anpy to davall16/Anpy, cheems-dev to lccalluchi). The table uses the URL that resolved.
- The qpso PyPI release (0.0.1, 2018) was taken from search results and not fetched.
