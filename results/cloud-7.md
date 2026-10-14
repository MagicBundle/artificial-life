cloud-7 (PARTIAL: steps 2 and 3 done): fast suite green, both goldens pass, 2 expected xfails. **test_ecology_3d FAILS on the overflow gate: hash cells over capacity in 299 of 2500 steps, peak occupancy 31 at C = 24.** Every other assertion passed. The rest of the slow suite (23/23) and the digest compare (bit-identical) are done. The diagnostic rerun reproduces the failure exactly: **one floor cell** overflows continuously from step 2151, and occupancy rises steadily to 31. K_est (14,190 initial, 12,759 final) is about 4× the realised population (about 3,100). The 64k3d probe is still running.

# cloud-7: 3D PR 6 (`3d/pr6-64k3d`) on Linux CPU

| item | value |
|---|---|
| commit tested | `401399fbf099064925653c007232321664805593` (origin/3d/pr6-64k3d, detached, clean) |
| baseline | cloud-1, main `afe52f5`; slow suite also compared against cloud-5 |
| `uname -m` | x86_64, 4 cores |
| Python / mlx | 3.12.3 / 0.32.3; CPU device |
| env | `uv sync --locked --group dev` (exit 0) |

## Step 2: fast suite

**Command:** `uv run pytest -m "not slow" -q -rs -rx`, run under `nice -n 5` in parallel with step 3.

**Result:** **1451 passed, 2 skipped, 2 xfailed**, 24 deselected, 0 failed, in 24 m 28 s.

- **xfails.** Exactly the two expected strict xfails:
  - `tests/test_director_3d.py::test_a_blocked_note_is_journalled_once_across_a_resume[tiny_cfg]`
  - `tests/test_director_3d.py::test_a_blocked_note_is_journalled_once_across_a_resume[tiny3d_cfg]`

  The reason given for both: "Director._blocked is per process and not rebuilt on a live resume (DESIGN §27.10)
  … Remove the marker with the fix."
- **Skips.** The 2 skips were not itemised, because `-rx` replaced `-rs` in the summary. The count matches the
  usual two in cloud-5 (no `ALIFE_LLM_URL`, no pygame).
- **Goldens.** Rerun verbosely with
  `uv run pytest tests/test_determinism.py::test_golden tests/test_determinism_3d.py::test_golden_3d -v -rs`:

  ```
  tests/test_determinism.py::test_golden PASSED
  tests/test_determinism_3d.py::test_golden_3d PASSED
  ```


## Step 3 (key): `tests/test_ecology_3d.py`, first full run

**Command:** `uv run pytest tests/test_ecology_3d.py -m slow -v -s -rA --durations=0`

**Result:** **FAILED** after 1944 s (32 m 24 s), run in parallel with the fast suite.

**The failing assertion is the last one in `check_dev3d`:**

```
E       AssertionError: hash cells over capacity in 299 of 2500 steps (peak occupancy 31, 31 in the last 500) at C = 24
E       assert np.float64(0.1196) <= 0.01
```

**Every assertion before it passed, in source order:**
- Not pinned at `n_cap`, and alive at the end.
- Births, bonds and attack hits all non-zero.
- No diag error bits, and **`invariants.full_check` empty**: the ledger closes and z and pitch are in range.
- Floor clamps occurred, both overall and in the last 500 steps.
- `HEAT_BUOYANCY` > 0.
- At the end, some but not all agents rest on the floor.

The overflow gate is the only failure. The run dict shows `k0` (K_est of the initial state) = **14189.6**.

**No diagnostics printed.** The test prints nothing on success or failure, so `-s` yielded no population
trajectory, floor shares or K_est line. To get them I am rerunning the test's own `run_dev3d(dev3d_cfg(), 2500)`
in a script that saves the counters and prints the trajectory. That is the same code path, so it is
deterministic. Results below.

<details><summary>Full pytest output</summary>

```
collecting ... collected 1 item

tests/test_ecology_3d.py::test_dev3d_ecology_is_alive_and_unsaturated FAILED

=================================== FAILURES ===================================
_________________ test_dev3d_ecology_is_alive_and_unsaturated __________________

    @pytest.mark.slow
    def test_dev3d_ecology_is_alive_and_unsaturated() -> None:
        cfg = dev3d_cfg()
>       check_dev3d(cfg, run_dev3d(cfg, STEPS))

tests/test_ecology_3d.py:123: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

cfg = Config(world=WorldConfig(size=160.0, field_cell=4.0, dims=3, height=40.0), population=PopulationConfig(n_max=16384, n_...ple=256, director='off', llm_url='http://localhost:1234/v1/chat/completions', llm_model='gemma-4-26b-a4b-it-mlx@8bit'))
run = {'state': State3(alive=array([True, True, True, ..., False, False, False], dtype=bool), uid=array([9296, 8006, 11081, ...       [ 650562,       0,  446633, ...,   55654,   40000,   11550]],
      shape=(2500, 16)), 'k0': 14189.632874613966}

    def check_dev3d(cfg: Config, run: dict[str, Any]) -> None:
        """The ecology assertions on a ``run_dev3d`` result."""
        d = dims(cfg)
        s, cnt, led = run["state"], run["counters"], run["ledgers"]
        n = cnt[:, C.N_ALIVE]
        totals = cnt.sum(axis=0)
        assert d.fz == 10 and d.q == 27 * 24 and cfg.sensing.cell_capacity == 24  # DEV3D keeps 64k3d's C and K
        # --- the 2D checks of test_ecology.py ---
        assert (n == d.n_cap).mean() < 0.5, "population pinned at N_CAP"
        assert 0 < n[-1] < d.n_cap
        assert totals[C.BIRTHS] > 0
        assert totals[C.BONDS_FORMED] + totals[C.CHILD_BONDS] > 0 and cnt[-1, C.BONDS_TOTAL] > 0
        assert totals[C.ATTACK_HITS] > 0
        assert int(to_numpy(s.diag)) & ERROR_MASK == 0
        assert invariants.full_check(s, cfg) == []  # the ledger closes, z and pitch are in range, every row valid
        # --- the 3D terms ---
        assert totals[Counter3D.Z_FLOOR_CLAMPS] > 0, "nobody ever reached the floor"
        assert led[:, LedgerIdx.HEAT_BUOYANCY].sum() > 0, "the buoyancy actuator was never paid for"
        assert cnt[-LATE:, Counter3D.Z_FLOOR_CLAMPS].sum() > 0, "nobody reached the floor in the last steps"
        # the final state, not the run-wide ratio: the founders start uniform in z, so GROUNDED / N_ALIVE over the
        # whole run lies strictly between 0 and 1 whatever the population does afterwards
        alive = to_numpy(s.alive)
        resting = to_numpy(physics.grounded(s.pos, cfg))[alive]
        assert 0 < resting.sum() < alive.sum(), f"{resting.sum()} of {alive.sum()} rest on the floor at the end"
        over = cnt[:, C.CELL_OVERFLOW] > 0
>       assert over.mean() <= OVERFLOW_STEPS_MAX, (
            f"hash cells over capacity in {over.sum()} of {over.size} steps (peak occupancy "
            f"{cnt[:, C.MAX_OCCUPANCY].max()}, {cnt[-LATE:, C.MAX_OCCUPANCY].max()} in the last {LATE}) at C = 24"
        )
E       AssertionError: hash cells over capacity in 299 of 2500 steps (peak occupancy 31, 31 in the last 500) at C = 24
E       assert np.float64(0.1196) <= 0.01
E        +  where np.float64(0.1196) = <built-in method mean of numpy.ndarray object at 0x7fc247769e90>()
E        +    where <built-in method mean of numpy.ndarray object at 0x7fc247769e90> = array([False, False, False, ...,  True,  True,  True], shape=(2500,)).mean

tests/test_ecology_3d.py:112: AssertionError
============================== slowest durations ===============================
1944.22s call     tests/test_ecology_3d.py::test_dev3d_ecology_is_alive_and_unsaturated

(2 durations < 0.005s hidden.  Use -vv to show these durations.)
=========================== short test summary info ============================
FAILED tests/test_ecology_3d.py::test_dev3d_ecology_is_alive_and_unsaturated
======================== 1 failed in 1944.46s (0:32:24) ========================

```
</details>


## Step 4: the rest of the slow suite

**Command:** `uv run pytest -m slow -q -rs --durations=25 --deselect tests/test_ecology_3d.py`, under `nice -n 5`
and in parallel with the diagnostic rerun.

**Result:** **23 passed**, 0 failed, in 30 m 25 s.

**Comparison with cloud-5:**
- The collected slow test IDs, excluding `test_ecology_3d`, are identical to cloud-5's 23.
- All pass, as they did there. No regression.
- Main's long tests took 913 s and 730 s (cloud-5: 1019 s and 836 s).

## Step 5: digest compare against the cloud-1 main baseline

**Command:** `uv run python /tmp/tools/digest_compare.py --baseline <cloud-1 main digests> --out results/cloud-7-pr6-digests.json`

**Result:** **exit 0: "OK: bit-identical to the 2D baseline (CPU device)"**, in 2 m 42 s.
- The 314 INFO lines are additions at defaults, including the new `64k3d` preset hash.
- No DIFF lines.


## Step 3b: diagnostic rerun of the DEV3D ecology run

**Script.** It imports `tests/test_ecology_3d.py` and calls `run_dev3d(dev3d_cfg(), 2500)`, then
`check_dev3d`. It saves all per-step counters and ledgers to `results/cloud-7-eco3d-counters.npz`, using the
`CounterIdx`, `Counter3D` and `LedgerIdx` columns, and the final-state summary to
`results/cloud-7-eco3d-final.json`. Wall time 37 min.

**Reproduction.** `check_dev3d` fails with the identical message:
`hash cells over capacity in 299 of 2500 steps (peak occupancy 31, 31 in the last 500) at C = 24`.

**Trajectory in 250-step windows.**
- "grounded share" is the mean of `GROUNDED / N_ALIVE` over the window.
- "max cell_overflow" is the number of hash cells over C at the worst step.

| steps | n_alive (end) | n min–max | grounded share (mean) | births | deaths starved/killed/aged | steps with overflow | max cell_overflow | peak occupancy | floor clamps | HEAT_BUOYANCY |
|---|---|---|---|---|---|---|---|---|---|---|
| 1–250 | 4451 | 4094–6821 | 0.043 | 3773 | 2842/576/0 | 0 | 0 | 12 | 50231 | 5322549 |
| 251–500 | 3664 | 3664–4450 | 0.129 | 585 | 1169/203/0 | 0 | 0 | 9 | 115832 | 3788306 |
| 501–750 | 3383 | 3374–3663 | 0.181 | 614 | 715/180/0 | 0 | 0 | 10 | 143577 | 3314448 |
| 751–1000 | 3192 | 3192–3386 | 0.223 | 621 | 668/144/0 | 0 | 0 | 11 | 165877 | 3086174 |
| 1001–1250 | 3116 | 3109–3187 | 0.248 | 659 | 568/167/0 | 0 | 0 | 14 | 176465 | 2944294 |
| 1251–1500 | 3043 | 3036–3116 | 0.270 | 668 | 584/157/0 | 0 | 0 | 16 | 186918 | 2866373 |
| 1501–1750 | 3034 | 3024–3061 | 0.277 | 676 | 509/176/0 | 0 | 0 | 18 | 189690 | 2831301 |
| 1751–2000 | 3013 | 2985–3049 | 0.294 | 715 | 581/155/0 | 0 | 0 | 18 | 200480 | 2800681 |
| 2001–2250 | 3085 | 2996–3090 | 0.300 | 757 | 494/191/0 | 49 | 1 | 28 | 204557 | 2805277 |
| 2251–2500 | 3134 | 3075–3142 | 0.304 | 762 | 539/174/0 | 250 | 1 | 31 | 210353 | 2863877 |


**When and where it overflows:**
- The first overflow is at **step 2151**. After that it is continuous: one run of 294 steps to the end, plus one
  earlier run of 5 steps. 299 steps in total (11.96%).
- `cell_overflow` is **never more than 1**, so a **single** hash cell is over capacity.
- The window peaks rise steadily: 12 → 9 → 10 → 11 → 14 → 16 → 18 → 18 → 28 → 31. This is one floor pile that
  keeps growing. It is not transient crowding.
- Over the 2500 steps, the per-step peak occupancy is ≤ 24 on 2340 steps and 25–31 on the other 160.

**Population:**
- 4096 founders boom to 6821 within the first 250 steps, then decline to about 3000–3150 by about step 1500
  and hold there.
- The final population is 3134, against `n_cap` 16,384.

**Floor and depth:**
- The grounded share rises steadily from 0.04 (founders uniform in z) to 0.30 at the end; 0.313 at the final
  state.
- Final z quantiles (min, q25, median, q75, q90, max): 1.0, 1.0, 4.96, 21.2, 36.2, 39.0.
- Histogram in 4-unit bins from 0 to 40: `[1474, 311, 199, 176, 154, 116, 116, 129, 137, 322]`.
- The distribution is **bimodal**: 47% in the bottom 4 units, a thin water column, and a surface layer of 10%
  in the top bin.

**K_est (`metrics.carrying_capacity`):**
- 14,190 on the initial state and 12,759 on the final state, against a realised steady population of about
  3,100. **The occupancy-weighted 3D K_est on this branch overestimates by about 4×.** On main it
  underestimated by about 2× (cloud-6).
- If the director's n/K uses this estimate, it would read about 0.24 at equilibrium.

**Ledger.** The ledger closes: `full_check` is empty, as in the test. `HEAT_BUOYANCY` is about 2.8M per 250
steps at steady state.

**Reading.**
- The gate fails because of one persistent floor pile that grows past C = 24 late in the run. It is still
  growing at step 2500.
- A longer run would likely make it worse, not better. Raising C to 32 would pass this run (peak 31) with no
  headroom.
- The question for the coordinator is whether to treat a single persistent pile as acceptable or as a design
  problem. If it is a problem, options are a softer floor-food gradient or crowding costs. The gate itself
  measures what it should.
