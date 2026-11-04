# Quantum walk / QCA / quantum ALife projects (verified 2026-11-03)

All entries below were checked by fetching the repository or paper page. "not shown" means the fetched page did not display that field; no dates were inferred. Dates are the main gap: only two entries show an archive date.

| Project | URL | Language | License | Last activity | What it does | Relation to concept | Verdict |
|---|---|---|---|---|---|---|---|
| Hiperwalk | https://github.com/hiperwalk/hiperwalk | Python (Jupyter listed; C/hiperblas core) | LGPL-3.0 | not shown (1,409 commits; 12 stars) | HPC simulator of quantum walks on graphs (README); project site hiperwalk.org | Closest engine for walker kernels on lattices/graphs; the page does not confirm coined vs continuous-time, and no lattice-sphere or measurement support is documented | Reuse candidate for the walk kernel, after checking coined and CTQW support |
| QuantumWalks (lucainnocenti) | https://github.com/lucainnocenti/QuantumWalks | Mathematica | not shown | not shown (20 commits) | 1D discrete-time walks with 2D coins (Hadamard, random unitary) | Lattice DTQW reference; no organisms or measurement | Reference only (needs Mathematica and QM package) |
| Quantum-walks (jishnurajendran) | https://github.com/jishnurajendran/quantum-walks | Python | BSD-3-Clause | not shown (18 commits) | DTQW with custom shift and coin operators; Parrondo-game scripts | Clean small DTQW code; game-theory framing only | Reference only (small project) |
| quantumwalk (mhurliman) | https://github.com/mhurliman/quantumwalk | Python (Cirq) | not shown | not shown (5 commits) | Line walk with Ry/Hadamard coin; 4D hypercube quantum-walk search | Tutorial-level lattice walk; no organisms | Reference only |
| margolus-cpp (magnetrwn) | https://github.com/magnetrwn/margolus-cpp | C++ (CMake layout; not stated) | MIT | archived, read-only 2026-04-04 | Classical reversible CA on Margolus 2x2 blocks (billiard-ball, gas rulesets), forward/backward stepping | Reversibility and block partitioning, but classical bits, no amplitudes | Reference only (archived) |
| quantum-cellular-automaton (BenjaminDecker) | https://github.com/BenjaminDecker/quantum-cellular-automaton | Python | MIT | archived 2024-03-14 | Classical simulation of the Quantum Game of Life via TDVP/MPS; blinker and single initial states | Most direct "quantum game of life" code; tensor-network, not per-organism wavefunctions | Reference only (superseded by the Julia rewrite) |
| QuantumGameOfLife.jl (BenjaminDecker) | https://github.com/BenjaminDecker/QuantumGameOfLife.jl | Julia | MIT | not shown (74 commits) | Unitary QCA from a Wolfram elementary-CA rule code (MA thesis method); TDVP; measurement plots | Unitary QCA with measurement, which matches the eating-as-measurement idea; no organisms, no spheres | Reference only (rule-based, not organisms) |
| lhillber/qca | https://github.com/lhillber/qca | Python (inferred from file names) | not shown | not shown (114 commits) | Quantum CA on elementary CA rules, analog and digital; state prep, time evolution, entanglement measures | The most complete unitary QCA code of the set (time evolution, entanglement) | Reuse candidate for evolution code, after license check |
| Val6789/qca | https://github.com/Val6789/qca | JavaScript (inferred) | not stated | not shown (523 commits) | Browser/Electron simulator of quantum-dot cellular automata (Q-dots), with tutorials | "QCA" here means quantum-dot cellular automata (electrostatic cells), not unitary QCA | Unrelated (name clash) |
| Quantum Artificial Life in an IBM Quantum Computer (Alvarez-Rodriguez et al.) | https://arxiv.org/abs/1711.09442 (Sci. Rep. 2018, doi 10.1038/s41598-018-33125-3) | n/a | n/a | 2017-2018 | Quantum biomimetic protocol: self-replication, mutation, interaction, death, and entanglement across generations on IBM ibmqx4 | The closest existing quantum-ALife model (paper). **Paper without code**: the arXiv page links no repository | Reference only (paper without code) |
| Lenia (Chakazul) | https://github.com/Chakazul/Lenia | Python, Matlab, JavaScript (multi-language) | MIT | not shown (174 commits) | Classical continuous 2D CA with life-like species | Template for a continuous organism-based ALife; no quantum version found | Unrelated to quantum; reference for the classical template only |

Notes:

- **No quantum Lenia found.** Searches returned only the classical Lenia repos and variants (Flow Lenia, Particle Lenia).
- **No quantum-walk ALife code found on GitHub.** The only quantum ALife code-like work is the Alvarez-Rodriguez paper above, which has no public repository.
- **Reversible classical CA exists** (margolus-cpp), but there is no open quantum Margolus-partitioned unitary automaton simulator in the results.

Not verified (search snippet only, page fetch failed or not attempted):

- **SQWalk** (PyPI, https://pypi.org/project/sqwalk): stochastic quantum walk simulator on QuTiP for continuous- and discrete-time walks on arbitrary adjacency matrices, labelled "in development." The PyPI fetch failed.
- **QWalk** (LNCC, https://qubit.lncc.br/qwalk/): older C simulator for 1D/2D lattice walks; Windows binaries. Not fetched.
- **pqca** (PyPI, https://pypi.org/project/pqca): partitioned QCA on Qiskit Aer, MIT per search snippet. The PyPI fetch failed and the GitHub source was not fetched.
- **QuantRS2 quantum CA module** (docs.rs source only): Margolus, Moore, and von Neumann neighborhoods; the repository location was not confirmed.

Shortlist for follow-up: Hiperwalk (walk kernel), lhillber/qca (unitary evolution), QuantumGameOfLife.jl (measurement and QCA rules), and the Alvarez-Rodriguez model (organism rules). Before reuse, confirm licenses (lhillber/qca) and last-commit dates (most entries), and check pqca and SQWalk directly.
