# Director evaluation (`DIRECTOR.md` §9)

Status: **draft, runs in progress** (2026-10-13). Seed 0 arms A, B and C and seed 1 arm A are done; the results
sections are filled when the queue finishes (about 2026-10-14 21:00).

## Answer

Pending.

## 1. Setup

Every arm forks the same 120,000-step state of its seed and runs to step 1,500,000 (1.38M steps), so within a seed
the director is the only difference. The arms ran one at a time on the M5 Pro (Metal), from the queue in
`runs/eval_queue/` (its README holds the controls).

| arm | director | seed 0 run | seed 1 run |
|---|---|---|---|
| A | off | `runs/64k_s0`, steps 120k–1.5M (not re-run, see below) | `runs/eval_s1_A` |
| B | rule tier, Gardener | `runs/eval_s0_B` | `runs/eval_s1_B` |
| C | rule tier + LLM every 25k, Trickster | `runs/eval_s0_C` | `runs/eval_s1_C` |
| R | `random`, yoked to B | `runs/eval_s0_R` | `runs/eval_s1_R` |
| RC | `random`, yoked to C, with C's pacing and budget (PR #14) | `runs/eval_s0_RC` | `runs/eval_s1_RC` |
| D | not run (§1.4) | | |

### 1.1 Fork sources

- **Seed 0:** `runs/eval_bases/s0_step_0000120000.safetensors`, byte-identical to
  `analysis_cache/64k_s0/step_0000120000.safetensors` (sha256 `8dd42c57c868ba45…`), the re-simulated state of
  `runs/64k_s0` at 120,000 verified against all 1,200 metrics rows (`docs/SIGNALS.md` §3).
- **Seed 1:** `runs/eval_s1_base`, a fresh `alife run --preset 64k --seed 1 --steps 120000`, forked at its checkpoint
  `runs/eval_s1_base:120000`.
- **Arm A of seed 0 is `runs/64k_s0` itself.** A fork with the director off would repeat it: arm B's metrics rows
  equal `64k_s0`'s on every one of the 10,790 rows before B's first action (1,199,000), wall-clock fields aside, with
  one exception, `diag`: `64k_s0` carries the sticky warning bit `CELL_OVERFLOW_WARN` (65536) from before 120k,
  which the fork cleared. Every arm is analysed up to step 1,500,000; `64k_s0` itself ran on to 1,504,009.

### 1.2 Commands and code

Common flags: `--steps 1380000 --set run.checkpoint_every=50000 --set run.keep_checkpoints=40`. `S0` is the seed-0
fork file above, `S1` is `runs/eval_s1_base:120000`.

| arm | command (after `alife run --init-from S0|S1 ... --out runs/eval_s<seed>_<arm>`) | code |
|---|---|---|
| A (seed 1) | (no director flags) | a154d45 |
| B | `--director live --persona gardener` | a154d45 |
| C | `--director live --persona trickster --llm` (model `gemma-4-26b-a4b-it-mlx@8bit`, LM Studio, loaded with `lms load ... -c 8192 --gpu max`, checked with `alife llm-check` before each C arm) | a154d45 |
| R | `--director live --persona random --yoke-log runs/eval_s<seed>_B/director.jsonl` | 0071863 (PR #11) |
| RC | `--director live --persona random --yoke-log runs/eval_s<seed>_C/director.jsonl` | b849ba9 (PR #14; `src/` and `tests/` identical to its merge afe52f5) |

- The three checkouts have the same engine: director off is digest-identical across them, and PR #14 changes only
  how R picks its pacing parameters (a Gardener-yoked R is byte-identical under it).
- `rule_every` is the default 1,000 in every arm, and `budget_per_100k` the default (the persona's: Gardener 2,
  Trickster 6). R takes the Gardener's pacing (B's); RC takes the Trickster's and budget 6 from C's log header.
- Each arm ended with `alife inspect RUN --check` (the full invariant and ledger check), which passed.

Wall time (simulated steps only) per arm: pending table.

### 1.3 Smoke runs

The queue was started on 2026-10-12, before this evaluation was specified with smoke runs, so A, B and C ran
without a 2,000-step smoke run; each of them ran with the ledger checked every step and passed `inspect --check`
at the end. For R, a 2,000-step smoke run from the seed-0 fork with the R code and flags was made before the R arms
started (2026-10-13): the ledger closed (`full_check: OK`), `director.jsonl` was written with the yoke path and
sha256 in its header, the yoke parser read one yoked move at step 1,199,000 from `runs/eval_s0_B/director.jsonl`
(B's single discretionary entry), and its 20 metrics rows equal B's.

### 1.4 Interruption of s0_R

At the owner's request, s0_R was stopped with one SIGINT at step 792,173 (2026-10-14 01:02; a `STOP` file kept the
next arm from starting) and resumed from that checkpoint at 22:41 the same day (`alife resume`, via `r_arms.sh`).
Its only yoked move is at 1,199,000, after the resume. Known effects of a live resume:
- a resume that splits a 1,000-step rule window may decide the safety rules differently from an uninterrupted run
  (`HANDOVER.md`, director caveats); `director.jsonl` is the record either way, and R's yoked draws depend only on
  the seed, the step and the log's budget and cooldowns, not on the vital signs;
- `Director._blocked` is not persisted, so a rule still blocked across the resume is journalled again in
  `journal.jsonl` (found by the 3D session; a fix is deferred). The evaluation reads no rule-tier journal of R.

### 1.5 Arm D

Not run. The 3D session (building `docs/3D.md`) found that no part of the 3D plan depends on it, that 2D bodies do
not transfer to 3D (gravity, bond sag), and that the goal "bodies ≥ 100" is within noise of the undirected
baseline (largest body about 90 at 64k), so even a positive result would be hard to read. A goal-directed bodies
experiment is planned in 3D after its PR 6. Owner decision (2026-10-13): deferred, to keep the GPU free; it may
still be run later, with a goal measurably above the baseline (for example largest body ≥ 200).

## 2. Metrics

All metrics are computed by `python -m alife.analysis.evaluation` (`src/alife/analysis/evaluation.py`, tests in
`tests/test_analysis_evaluation.py`) from each arm's `metrics.jsonl`, `director.jsonl`, `journal.jsonl` and event logs.

- **The director's own vital signs are recomputed for every arm**, the control included, with the director's code
  (`Tracker`, default `RuleParams`, a 1,000-step rule window), each arm's archive starting empty at the fork. This
  reproduces the vital signs the live directors logged exactly (every logged entry checked; §3).
- **Coverage:** grid cells (10 × 10 × 10 over emitters, bodies ≥ 4 and predation) ever visited; also the cells
  the control never visited.
- **Novelty:** the director's novelty (mean distance of the window's behaviour descriptor to its 5 nearest archive
  members), mean over the arm. **Excursion:** the same distance to every descriptor of the seed's control arm, which
  does not depend on the arm's own archive.
- **Persistent novelty** (§9: novelty still above the baseline 100k steps after an intervention). For each
  intervention step `s` (any applied entry but a revert) whose window ends by 1.5M:
  - *vs A* (primary): the arm's mean novelty over `(s + 100k, s + 150k]` minus the control's over the same steps.
    It is 0 for A by construction.
  - *vs before*: the same window minus the arm's own mean over `(s − 50k, s]`.
- **Turnover:** `b_shift`, the RMS distance between the descriptor now and 100k steps earlier, and archive
  additions per 100k steps.
- **Viability:** the lowest population, its ratio to a fixed per-seed reference (the base run's mean population
  over steps 70k–120k: 33,071 for seed 0, 32,164 for seed 1), **collapse** = a minimum below 20% of the reference,
  the share of rule windows "at risk" (the director's `fall` above 0.3: a fall of more than 30% within 50k steps),
  and the extinction rescues.
- **Clades:** `clades_alive`, `top_clade_share` (all immigrants share one clade id, so immigration adds at most one
  clade). **Lineages** (event logs): the fork founders' lineages still alive at 1.5M, the immigrant-descended share
  of the population, and the generations gained since the fork.
- **Interventions:** counts by action, source and rule, distinct actions, the entropy of the discretionary
  action mix, and the budget cost per 100k steps.
- **LLM:** JSON validity (replies that parsed and passed the schema), the clamp rate (clamped parameters per
  accepted proposal), rejections, fallbacks and latency.
- **Evolutionary activity:** `alife activity RUN --start 120000` (the fork step; for a fork it is the default)
  with the windows 120k–1M and 1M–1.5M, written to `analysis_cache/eval/s<seed>_<arm>/`. `alife activity` now reads
  forks: the founders are the agents alive at the fork step. On `runs/64k_s0` from step 0 the change reproduces
  `docs/ACTIVITY.md` exactly (all 820 arrays). Classes are Channon's, as in `docs/ACTIVITY.md`.

## 3. Results

Pending.

### 3.x Journal legibility (owner ratings)

The sheet is [`evaluation/legibility_sheet.md`](evaluation/legibility_sheet.md): 10 LLM journal entries of arm C, 5 per
seed, drawn uniformly without replacement (`journal_sample`, seed 0), each with the population and novelty the
director saw, and B's rule-tier lines for comparison. Rated by the owner, 1 (opaque or wrong) to 5 (clear and accurate).

| entry | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rating | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | – |

## 4. Verdict

Pending.

## 5. Reproduce

```bash
uv run alife activity runs/64k_s0 --start 120000 --out analysis_cache/eval/s0_A --windows 120000:1000000,1000000:end
```

```bash
uv run python -m alife.analysis.evaluation --arm 0:A:runs/64k_s0:120000 --arm 0:B:runs/eval_s0_B ... \
  --base 0:runs/64k_s0 --base 1:runs/eval_s1_base --activity-root analysis_cache/eval \
  --out docs/evaluation --lineages --plots --journal-sample 0:C --journal-sample 1:C
```
