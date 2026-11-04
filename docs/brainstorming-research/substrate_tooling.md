# Open-source tooling for a quantum-rules substrate: verified survey (2026-11-03)

"Last activity" is the year shown on the fetched page. "not verified" means the page did not show a date; the footer "2026" is GitHub's copyright, not activity.

## Table

| Project | URL | Language | License | Last activity | What it offers for this substrate | Known limits | Verdict |
|---|---|---|---|---|---|---|---|
| Apple MLX | https://github.com/ml-explore/mlx | C++ / Python | MIT | 2026 (issue activity Aug 2026) | Array framework on Metal; `fft`, `ifft`, `fft2`, `fftn`, `rfft*`, `stft` listed in the docs. Good fit for the GPU step. | Public FFT on Metal is float32/complex64 only (issue #3972, Aug 2026). No float64 on Metal (issue #1905; maintainers suggest CPU for high-precision steps). Complex divide with complex denominator was missing in old issue #361 (current status not verified). Complex matmul on Metal (issue #2076) was ~30% slower than the ops path. The FFT docs page does not state dtype rules. | use (fp32 complex on GPU; keep reference checks on CPU) |
| mlx-nufft | https://pypi.org/project/mlx-nufft/ | Python / Metal | Apache-2.0 | 2026 (v0.3.3, Oct 2026 per PyPI) | Non-uniform FFT types 1-3 in dims 1-3 on Apple GPUs, via Metal/MLX. | Apple silicon only. Computes fp32-grade (eps clamped at 1e-6). Its "fp64-grade" intro claim conflicts with that. Type 3 error ~5e-5. modeord=1 not implemented. | reference (only if non-uniform sampling is needed) |
| healpy | https://github.com/healpy/healpy | Python (wraps C++ HEALPix, cfitsio) | GPL-2.0 | not verified (no release shown) | Reference CPU spherical-harmonic transform on HEALPix: `map2alm`, pixel weights, ring ordering, polarisation. Multi-threaded. | CPU only. Needs ring ordering; accuracy depends on pixel-weight files from separate healpy-data repo. GPL-2.0 licence constrains redistribution. | reference (CPU oracle for SH checks) |
| jax-healpy | https://github.com/CMBSciPol/jax-healpy (docs: https://jax-healpy.readthedocs.io/) | Python / JAX | GPL-3.0 | 2026 (docs at version 0.6.dev1; release date not verified) | JAX port of HEALPix functions, drop-in for many healpy calls, autodiff, GPU via JAX/XLA. Spherical harmonics via s2fft. | Needs s2fft extra. Recommends 64-bit (default 32-bit, which only matches healpy at small nside). No Metal backend found. Docs' beta status not confirmed on fetched page. | reference (CUDA/CPU path; not a Metal path) |
| s2fft | https://github.com/astro-informatics/s2fft | Python (C/C++/CUDA extension optional) | MIT | 2023 copyright; latest release not verified | Differentiable spherical-harmonic and Wigner transforms in JAX or PyTorch; CUDA extension speeds HEALPix transforms. | GPU and TPU via JAX/CUDA; no Metal path found. Release/activity date not confirmed. | reference (candidate SH engine on CUDA; not on Apple GPU) |
| margolus-cpp | https://github.com/magnetrwn/margolus-cpp | C++ | MIT | 2026 (archived 4 Apr 2026, read-only) | Reversible cellular automaton simulator for Margolus-neighbourhood (2x2 block) rules; forward and backward stepping; save/load; CLI and C++ library. Matches the exact-reversibility option. | Archived, so no upstream fixes. Only 42 commits. Library section may be outdated (author's note in search result). | reference (read the rule design; do not depend on it) |
| Golly | https://golly.sourceforge.io/ | C++ (app) | not stated on fetched page | 2025 (v5.0, Oct 2025) | Cellular automaton explorer, up to 256 states. Hiroshima University datasets for reversible rules (Morita) exist for Golly. | No native reversible-rule support; workarounds need extra history states. Licence not confirmed. Not a library. | reference (visual check of small reversible rules) |
| critters (dmishin) | https://github.com/dmishin/critters | C++ core, Python ctypes | not stated | not verified (6 commits shown) | Block-automaton "Critters" rule with Margolus neighbourhood, 2 states. | README: installation "Not supported yet"; requires Python 2; Windows needs MinGW. | avoid |
| QuTiP | https://github.com/QuTiP/qutip | Python | BSD-3-Clause | not verified (no release shown) | Closed and open quantum dynamics; use as a reference for small Hilbert spaces and unitary evolution checks. | No lattice or spin-chain utilities found on the page. Small systems only; not built for lattice-scale work. | reference (small-system correctness oracle) |
| Qiskit Aer | https://github.com/Qiskit/qiskit-aer | C++ / Python | Apache-2.0 | not verified (no release shown) | Statevector, density-matrix and unitary simulators; a reference oracle for gate-level checks. | GPU support is CUDA only, x86_64 Linux only. No Metal. No lattice or walk templates. | reference (statevector oracle, CPU) |
| PennyLane | https://github.com/PennyLaneAI/pennylane | Python | Apache-2.0 | not verified (no release shown) | Quantum and differentiable circuit library; a reference for gradients through circuits. | Default-qubit backend name not confirmed on the fetched page. No lattice or graph-walk templates found. Lower priority than QuTiP or Aer. | reference (optional; low priority) |

## Verified gaps (by topic)

1. **MLX complex and precision.** The FFT docs page lists the functions but not dtypes or devices. Dtype facts come from GitHub issues: fp32/complex64 only for FFT on Metal, and no float64 on Metal. Treat every Metal complex result as fp32 and check it against a CPU reference. This matches the repo's F20 concern about GPU float32 accuracy.
2. **Spherical harmonics on HEALPix.** Verified: healpy (CPU, GPL-2.0) and jax-healpy plus s2fft (JAX, GPU via CUDA). No Metal or MLX port was found. An MLX SH transform would have to be written from scratch, using healpy as the oracle.
3. **Reversible CA libraries.** Only small, personal projects were found. margolus-cpp is the most complete but is archived. No maintained general-purpose Margolus library was found. Integer-only block rules in MLX are likely simpler than adapting these.
4. **Quantum simulators as references.** QuTiP and Qiskit Aer are verified as small-system references. Neither has lattice or quantum-walk support. Both are CPU-first or CUDA-only.
5. **Quantum walks on spheres and non-orientable graphs.** No open implementation for a sphere was found. Möbius support was found only as a lead (below). Build the graph in NetworkX or similar, then run a discrete walk.

## Unverified leads (not in the table)

- **qwsim** (Rodrigues, Paunković, Mateus): C simulator that claims Möbius-strip support for a 2D single-walker lattice. Listed at http://qwsim.weebly.com/. The site was not fetched, so the licence and availability are unknown.
- **hiperwalk** (PyPI, https://pypi.org/project/hiperwalk/2.0a6): Python quantum-walk package for general graphs (NetworkX, NumPy, SciPy). The fetch did not return a licence or repository, and geometry support is unknown.
- **Golly reversible-rule datasets** (Hiroshima University repository, https://ir.lib.hiroshima-u.ac.jp/00055227): reversible rule files for Golly, from search results only.

## Summary

- **Use:** MLX (fp32 complex on GPU, with CPU checks).
- **Reference:** healpy (CPU SH oracle), jax-healpy and s2fft (CUDA/JAX SH path), margolus-cpp (reversible-CA design), Golly (visual check), QuTiP and Qiskit Aer (small-system oracles), PennyLane (optional), mlx-nufft (only if non-uniform sampling is needed).
- **Avoid:** critters (install not supported, Python 2).
- **No open implementation found:** quantum walks on a sphere. Möbius support is an unverified lead (qwsim).
