# Classification per window and component definition, four runs

B = bounded, N = none, U3a / U3b / U3c = unbounded via D / via mean activity / via both (Channon's split of Bedau class 3); n/a = cohort pool still filling. Primary settings of docs/ACTIVITY.md section 3.5.

| space | window | Channon s0 | Channon s1 | Channon s0_half | Channon s0_dir | Bedau s0 | Bedau s1 | Bedau s0_half | Bedau s0_dir |
|---|---|---|---|---|---|---|---|---|---|
| cohort_h100k | 0k-100k | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| cohort_h100k | 100k-300k | B | B | B | B | B | B | U3a | U3a |
| cohort_h100k | 300k-500k | B | B | B | B | B | B | U3b | U3b |
| cohort_h100k | 0k-500k | B | B | B | B | U3a | U3a | U3a | B |
| cohort_h200k | 0k-100k | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| cohort_h200k | 100k-300k | B | B | B | B | B | U3b | B | B |
| cohort_h200k | 300k-500k | B | B | B | B | B | B | U3b | B |
| cohort_h200k | 0k-500k | B | B | B | B | U3a | B | B | B |
| cohort_h500k | 0k-100k | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| cohort_h500k | 100k-300k | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| cohort_h500k | 300k-500k | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| cohort_h500k | 0k-500k | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| marker_0.003 | 0k-100k | B | B | B | B | U3b | U3b | U3b | U3b |
| marker_0.003 | 100k-300k | B | B | B | B | U3b | U3b | U3b | U3b |
| marker_0.003 | 300k-500k | B | B | B | B | U3b | U3b | U3b | N |
| marker_0.003 | 0k-500k | B | B | B | B | U3b | U3b | U3b | U3b |
| marker_0.01 | 0k-100k | B | B | B | B | U3b | U3b | U3b | U3b |
| marker_0.01 | 100k-300k | B | B | B | B | U3b | U3b | B | U3b |
| marker_0.01 | 300k-500k | B | B | B | B | B | U3b | U3b | U3b |
| marker_0.01 | 0k-500k | B | B | B | B | B | U3b | U3b | U3b |
| marker_0.03 | 0k-100k | B | B | B | B | U3b | U3c | U3b | U3b |
| marker_0.03 | 100k-300k | B | B | B | B | B | B | B | U3b |
| marker_0.03 | 300k-500k | B | B | B | B | U3b | B | B | B |
| marker_0.03 | 0k-500k | B | B | B | B | B | U3b | B | U3b |
| marker_0.1 | 0k-100k | B | B | B | B | U3a | U3c | U3b | U3b |
| marker_0.1 | 100k-300k | B | B | B | B | B | U3b | B | B |
| marker_0.1 | 300k-500k | B | B | B | B | B | B | B | B |
| marker_0.1 | 0k-500k | B | B | B | B | B | B | B | B |

## Tallies over the valid cells

| run | valid cells | Channon primary B/N/U | Bedau primary B/N/U | Channon informative sweep n/b/u (summed) | cells with any unbounded sweep setting |
|---|---|---|---|---|---|
| s0 | 22 | 22/0/0 | 11/0/11 | 99/411/0 | 0 |
| s1 | 22 | 22/0/0 | 8/0/14 | 108/453/15 | 2 |
| s0_half | 22 | 22/0/0 | 9/0/13 | 132/378/0 | 0 |
| s0_dir | 22 | 22/0/0 | 8/1/13 | 147/435/0 | 0 |

## Agreement of the (coarse) Bedau class between runs

| pair | cells with the same coarse Bedau class | cells | fraction |
|---|---|---|---|
| s0 vs s1 | 15 | 22 | 0.68 |
| s0 vs s0_half | 14 | 22 | 0.64 |
| s0 vs s0_dir | 12 | 22 | 0.55 |
| s1 vs s0_half | 15 | 22 | 0.68 |
| s1 vs s0_dir | 15 | 22 | 0.68 |
| s0_half vs s0_dir | 16 | 22 | 0.73 |

Cells where the four runs do not all share the coarse Bedau class: 14 of 22.
