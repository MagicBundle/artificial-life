## 3D 64k3d_s0 (0-500k)

windows: [[0, 100000], [100000, 300000], [300000, 500000]]; grid 10000; analysed to 500000

### Classification (primary settings; sweep = none/bounded/unbounded counts)

| space | window | births/step | valid from | Bedau | Bedau sweep n/b/u | Channon | Channon informative sweep n/b/u | Channon full sweep n/b/u |
|---|---|---|---|---|---|---|---|---|
| cohort_h100k | 0k-500k | 43.94 | 100000 | unbounded (3a: D) | 135/49/122 | bounded | 6/24/0 | 30/48/0 |
| cohort_h100k | 0k-100k | | 100000 | n/a | | n/a | | |
| cohort_h100k | 100k-300k | 45.78 | 100000 | bounded | 135/98/73 | bounded | 3/27/0 | 15/63/0 |
| cohort_h100k | 300k-500k | 42.09 | 300000 | bounded | 135/141/12 | bounded | 0/24/0 | 48/30/0 |
| cohort_h200k | 0k-500k | 42.76 | 200000 | unbounded (3a: D) | 144/51/129 | bounded | 0/24/0 | 12/66/0 |
| cohort_h200k | 0k-100k | | 200000 | n/a | | n/a | | |
| cohort_h200k | 100k-300k | 44.11 | 200000 | bounded | 144/174/6 | bounded | 0/24/0 | 12/66/0 |
| cohort_h200k | 300k-500k | 42.09 | 300000 | bounded | 144/104/76 | bounded | 0/24/0 | 24/54/0 |
| cohort_h500k | 0k-500k | | 500000 | n/a | | n/a | | |
| cohort_h500k | 0k-100k | | 500000 | n/a | | n/a | | |
| cohort_h500k | 100k-300k | | 500000 | n/a | | n/a | | |
| cohort_h500k | 300k-500k | | 500000 | n/a | | n/a | | |
| marker_0.003 | 0k-500k | 44.99 | 0 | unbounded (3b: Ā) | 189/162/117 | bounded | 18/12/0 | 24/54/0 |
| marker_0.003 | 0k-100k | 49.21 | 0 | unbounded (3b: Ā) | 126/162/126 | bounded | 0/18/0 | 18/60/0 |
| marker_0.003 | 100k-300k | 45.78 | 100000 | unbounded (3b: Ā) | 162/162/126 | bounded | 12/12/0 | 24/54/0 |
| marker_0.003 | 300k-500k | 42.09 | 300000 | unbounded (3b: Ā) | 189/162/117 | bounded | 12/12/0 | 24/54/0 |
| marker_0.01 | 0k-500k | 44.99 | 0 | bounded | 180/252/18 | bounded | 12/18/0 | 24/54/0 |
| marker_0.01 | 0k-100k | 49.21 | 0 | unbounded (3b: Ā) | 162/144/108 | bounded | 0/18/0 | 24/54/0 |
| marker_0.01 | 100k-300k | 45.78 | 100000 | unbounded (3b: Ā) | 162/162/108 | bounded | 12/18/0 | 24/54/0 |
| marker_0.01 | 300k-500k | 42.09 | 300000 | bounded | 180/243/27 | bounded | 0/18/0 | 24/54/0 |
| marker_0.03 | 0k-500k | 44.99 | 0 | bounded | 171/243/0 | bounded | 6/18/0 | 30/48/0 |
| marker_0.03 | 0k-100k | 49.21 | 0 | unbounded (3b: Ā) | 189/88/119 | bounded | 3/21/0 | 27/51/0 |
| marker_0.03 | 100k-300k | 45.78 | 100000 | bounded | 153/243/0 | bounded | 6/18/0 | 30/48/0 |
| marker_0.03 | 300k-500k | 42.09 | 300000 | unbounded (3b: Ā) | 171/90/153 | bounded | 6/18/0 | 30/48/0 |
| marker_0.1 | 0k-500k | 44.99 | 0 | bounded | 153/207/18 | bounded | 0/18/0 | 54/24/0 |
| marker_0.1 | 0k-100k | 49.21 | 0 | unbounded (3a: D) | 153/63/126 | bounded | 0/12/0 | 54/24/0 |
| marker_0.1 | 100k-300k | 45.78 | 100000 | bounded | 144/159/57 | bounded | 0/18/0 | 48/30/0 |
| marker_0.1 | 300k-500k | 42.09 | 300000 | bounded | 135/162/81 | bounded | 3/15/0 | 57/21/0 |

### Channon component-normalised statistics (primary spaces)

theta is the primary q=5% threshold of that run; A^N_new in normalised activity per component per 10k-step grid time; rho_* in units of the birth rate (>0.01 would count as growth).

| space | window | theta | A^N_new run | neutral | permuted | positive frac | A^N_cum end | median A^N end | rho_cum | rho_med |
|---|---|---|---|---|---|---|---|---|---|---|
| cohort_h200k | 0k-500k | 1e+04 | 24.6 | 0.0494 | 0 | 1 | 4.73e+06 | -6.98 | -0.0389 | -0.016 |
| cohort_h200k | 100k-300k | 1e+04 | 26.2 | 0.148 | 0 | 1 | 5.2e+06 | -1.06 | 0.282 | -0.0234 |
| cohort_h200k | 300k-500k | 1e+04 | 23.9 | 0 | 0 | 1 | 4.73e+06 | -6.98 | 0.0549 | -0.00615 |
| marker_0.01 | 0k-500k | 3162 | 14.4 | 0.735 | 0.596 | 1 | -8.67e+05 | 3 | -0.0478 | -0.000262 |
| marker_0.01 | 0k-100k | 3162 | 19.9 | 1.42 | 0.691 | 1 | 1.67e+05 | 7 | 0.000236 | 0.00075 |
| marker_0.01 | 100k-300k | 3162 | 13.4 | 0.696 | 0.698 | 1 | -5.76e+04 | 4 | -0.041 | -0.000204 |
| marker_0.01 | 300k-500k | 3162 | 12.6 | 0.433 | 0.448 | 1 | -8.67e+05 | 3 | -0.0916 | -0.000342 |

### A^N_new at common thresholds (run / neutral / permuted)

| space | window | theta=316.228 | theta=1000 | theta=3162.28 | theta=10000 |
|---|---|---|---|---|---|
| cohort_h200k | 0k-500k | 18.4 / 18 / 14 | 18.8 / 15.7 / 12.3 | 20.3 / 6.72 / 3.99 | 24.6 / 0.0494 / 0 |
| cohort_h200k | 100k-300k | 22.4 / 19.1 / 15.7 | 22.5 / 16.1 / 13.3 | 22.3 / 7.03 / 5.07 | 26.2 / 0.148 / 0 |
| cohort_h200k | 300k-500k | 16.5 / 17.5 / 13.2 | 17 / 15.5 / 11.9 | 19.2 / 6.57 / 3.45 | 23.9 / 0 / 0 |
| marker_0.01 | 0k-500k | 17.7 / 12 / 9.74 | 15.2 / 6.02 / 4.99 | 14.4 / 0.735 / 0.596 | 14.5 / 0 / 0 |
| marker_0.01 | 0k-100k | 29.9 / 22 / 18.2 | 24.2 / 11.3 / 8.48 | 19.9 / 1.42 / 0.691 | 20.6 / 0 / 0 |
| marker_0.01 | 100k-300k | 15.6 / 10.4 / 8.65 | 13.4 / 5.26 / 4.64 | 13.4 / 0.696 / 0.698 | 11 / 0 / 0 |
| marker_0.01 | 300k-500k | 13.7 / 8.69 / 6.62 | 12.6 / 4.13 / 3.58 | 12.6 / 0.433 / 0.448 | 14.8 / 0 / 0 |

### Bedau original normalisation (primary spaces)

| space | window | class | A_new excess | rho | rho_run | rho_shadow | D growth over window | D mean |
|---|---|---|---|---|---|---|---|---|
| cohort_h200k | 0k-500k | unbounded (3a: D) | 436 | -2.7 | -3.63 | -0.924 | 0.513 | 11982 |
| cohort_h200k | 100k-300k | bounded | 322 | -1.73 | -3.1 | -1.37 | 0.125 | 9766.6 |
| cohort_h200k | 300k-500k | bounded | 493 | -1.65 | -2.13 | -0.48 | 0.271 | 13089 |
| marker_0.01 | 0k-500k | bounded | 0.031 | 0.00362 | 0.106 | 0.103 | -0.00793 | 1436.2 |
| marker_0.01 | 0k-100k | unbounded (3b: Ā) | 0.0293 | 0.0849 | 0.529 | 0.444 | -0.25 | 1511.2 |
| marker_0.01 | 100k-300k | unbounded (3b: Ā) | 0.029 | 0.183 | 0.273 | 0.0904 | -0.0248 | 1371.1 |
| marker_0.01 | 300k-500k | bounded | 0.0338 | -0.215 | -0.195 | 0.0193 | 0.0433 | 1463.8 |

### Cohort lineage sorting (surviving lineages per cohort at a given age, mean over cohorts started in the window)

| age | cohort start window | cohorts | run | neutral | heritability null | run / null | excess frac run | null |
|---|---|---|---|---|---|---|---|---|
| age_20k | 0k-100k | 10 | 934.8 | 1218 | 1631 | 0.573 | 0.423 | 0.412 |
| age_20k | 100k-300k | 20 | 1725 | 1879 | 2407 | 0.717 | 0.441 | 0.424 |
| age_20k | 300k-500k | 19 | 2421 | 2022 | 2739 | 0.884 | 0.429 | 0.42 |
| age_50k | 0k-100k | 10 | 157.1 | 526.8 | 693.8 | 0.226 | 0.169 | 0.159 |
| age_50k | 100k-300k | 20 | 396.1 | 781.9 | 958.8 | 0.413 | 0.179 | 0.172 |
| age_50k | 300k-500k | 16 | 643.1 | 837.4 | 1080 | 0.596 | 0.174 | 0.171 |
| age_100k | 0k-100k | 10 | 38.7 | 298.4 | 388.1 | 0.0997 | 0.0768 | 0.0793 |
| age_100k | 100k-300k | 20 | 126.6 | 401.2 | 484.2 | 0.261 | 0.081 | 0.0862 |
| age_100k | 300k-500k | 11 | 203.3 | 422.7 | 531.7 | 0.382 | 0.0839 | 0.0849 |
| age_200k | 0k-100k | 10 | 12 | 174.2 | 214.8 | 0.0559 | 0.0339 | 0.042 |
| age_200k | 100k-300k | 20 | 38.75 | 202.3 | 242.5 | 0.16 | 0.0429 | 0.0432 |
| age_200k | 300k-500k | 1 | 54 | 211.5 | 259.5 | 0.208 | 0.0529 | 0.041 |

## 2D 64k_s0, early windows

windows: [[0, 100000], [100000, 300000], [300000, 500000]]; grid 10000; analysed to 1500000

### Classification (primary settings; sweep = none/bounded/unbounded counts)

| space | window | births/step | valid from | Bedau | Bedau sweep n/b/u | Channon | Channon informative sweep n/b/u | Channon full sweep n/b/u |
|---|---|---|---|---|---|---|---|---|
| cohort_h100k | 0k-1500k | 28.51 | 100000 | unbounded (3b: Ā) | 126/54/126 | bounded | 6/18/0 | 54/24/0 |
| cohort_h100k | 0k-100k | | 100000 | n/a | | n/a | | |
| cohort_h100k | 100k-300k | 32.05 | 100000 | none | 144/54/108 | bounded | 3/21/0 | 57/21/0 |
| cohort_h100k | 300k-500k | 30.24 | 300000 | none | 144/54/90 | bounded | 0/18/0 | 60/18/0 |
| cohort_h200k | 0k-1500k | 28.15 | 200000 | unbounded (3b: Ā) | 135/54/135 | bounded | 12/12/0 | 60/18/0 |
| cohort_h200k | 0k-100k | | 200000 | n/a | | n/a | | |
| cohort_h200k | 100k-300k | 30.83 | 200000 | none | 153/117/54 | bounded | 3/14/7 | 51/18/9 |
| cohort_h200k | 300k-500k | 30.24 | 300000 | none | 144/54/108 | bounded | 0/18/0 | 54/24/0 |
| cohort_h500k | 0k-1500k | 27.46 | 500000 | unbounded (3b: Ā) | 171/54/135 | bounded | 12/18/0 | 48/30/0 |
| cohort_h500k | 0k-100k | | 500000 | n/a | | n/a | | |
| cohort_h500k | 100k-300k | | 500000 | n/a | | n/a | | |
| cohort_h500k | 300k-500k | | 500000 | n/a | | n/a | | |
| marker_0.003 | 0k-1500k | 28.97 | 0 | unbounded (3b: Ā) | 198/153/135 | bounded | 15/9/0 | 45/33/0 |
| marker_0.003 | 0k-100k | 35.33 | 0 | none | 153/99/108 | bounded | 3/15/0 | 63/15/0 |
| marker_0.003 | 100k-300k | 32.05 | 100000 | bounded | 144/270/0 | bounded | 3/15/0 | 57/21/0 |
| marker_0.003 | 300k-500k | 30.24 | 300000 | unbounded (3b: Ā) | 180/144/126 | bounded | 3/15/0 | 51/27/0 |
| marker_0.01 | 0k-1500k | 28.97 | 0 | unbounded (3b: Ā) | 225/159/66 | bounded | 9/15/0 | 57/21/0 |
| marker_0.01 | 0k-100k | 35.33 | 0 | unbounded (3b: Ā) | 198/72/126 | bounded | 3/15/0 | 57/21/0 |
| marker_0.01 | 100k-300k | 32.05 | 100000 | bounded | 180/234/0 | bounded | 3/15/0 | 57/21/0 |
| marker_0.01 | 300k-500k | 30.24 | 300000 | unbounded (3b: Ā) | 225/99/108 | bounded | 3/15/0 | 57/21/0 |
| marker_0.03 | 0k-1500k | 28.97 | 0 | bounded | 189/189/36 | bounded | 3/15/0 | 57/21/0 |
| marker_0.03 | 0k-100k | 35.33 | 0 | unbounded (3b: Ā) | 189/105/84 | bounded | 0/18/0 | 54/24/0 |
| marker_0.03 | 100k-300k | 32.05 | 100000 | bounded | 225/117/54 | bounded | 0/18/0 | 60/18/0 |
| marker_0.03 | 300k-500k | 30.24 | 300000 | unbounded (3b: Ā) | 234/108/72 | bounded | 6/12/0 | 66/12/0 |
| marker_0.1 | 0k-1500k | 28.97 | 0 | bounded | 189/171/0 | bounded | 6/12/0 | 66/12/0 |
| marker_0.1 | 0k-100k | 35.33 | 0 | unbounded (3b: Ā) | 180/90/90 | bounded | 3/9/0 | 63/15/0 |
| marker_0.1 | 100k-300k | 32.05 | 100000 | bounded | 189/156/15 | bounded | 0/6/0 | 66/12/0 |
| marker_0.1 | 300k-500k | 30.24 | 300000 | none | 198/135/27 | bounded | 0/6/0 | 66/12/0 |

### Channon component-normalised statistics (primary spaces)

theta is the primary q=5% threshold of that run; A^N_new in normalised activity per component per 10k-step grid time; rho_* in units of the birth rate (>0.01 would count as growth).

| space | window | theta | A^N_new run | neutral | permuted | positive frac | A^N_cum end | median A^N end | rho_cum | rho_med |
|---|---|---|---|---|---|---|---|---|---|---|
| cohort_h200k | 0k-1500k | 3162 | 11.2 | 2.81 | 0.357 | 1 | 3.62e+06 | 5.85 | -0.0122 | -0.000584 |
| cohort_h200k | 100k-300k | 3162 | 12.3 | 6.5 | 1.17 | 1 | 4.2e+06 | 9.24 | 0.214 | 0.00676 |
| cohort_h200k | 300k-500k | 3162 | 10.5 | 5.13 | 0.709 | 1 | 3.4e+06 | 14.4 | -0.0328 | 0.0144 |
| marker_0.01 | 0k-1500k | 3162 | 5.69 | 0.2 | 0.0815 | 1 | -2.84e+05 | 4.66 | -0.0129 | -1.66e-05 |
| marker_0.01 | 0k-100k | 3162 | 12.6 | 0.1 | 0.0956 | 1 | 2.25e+05 | 5 | 0.0301 | 0.00249 |
| marker_0.01 | 100k-300k | 3162 | 5.95 | 0.296 | 0.136 | 1 | 9.75e+04 | 8 | -0.0185 | 0.000417 |
| marker_0.01 | 300k-500k | 3162 | 5.1 | 0.47 | 0.0461 | 1 | -8.8e+04 | 7.93 | -0.03 | 0.000246 |

### A^N_new at common thresholds (run / neutral / permuted)

| space | window | theta=316.228 | theta=1000 | theta=3162.28 | theta=10000 |
|---|---|---|---|---|---|
| cohort_h200k | 0k-1500k | 9.58 / 13.2 / 8.07 | 9.91 / 10.9 / 5.2 | 11.2 / 2.81 / 0.357 | 12.5 / 0.00676 / 0 |
| cohort_h200k | 100k-300k | 9.54 / 17 / 10.3 | 10.2 / 15.2 / 7.8 | 12.3 / 6.5 / 1.17 | 11.3 / 0 / 0 |
| cohort_h200k | 300k-500k | 9.26 / 15.6 / 9.21 | 9.38 / 13.4 / 6.8 | 10.5 / 5.13 / 0.709 | 11.8 / 0 / 0 |
| marker_0.01 | 0k-1500k | 6.15 / 6.95 / 4.37 | 5.9 / 3.31 / 1.71 | 5.69 / 0.2 / 0.0815 | 4.49 / 0 / 0 |
| marker_0.01 | 0k-100k | 10.8 / 14.1 / 10.2 | 11.9 / 5.94 / 3.46 | 12.6 / 0.1 / 0.0956 | 6.56 / 0 / 0 |
| marker_0.01 | 100k-300k | 7.08 / 8.75 / 5.17 | 6.24 / 4.96 / 2.36 | 5.95 / 0.296 / 0.136 | 4.56 / 0 / 0 |
| marker_0.01 | 300k-500k | 6.23 / 7.5 / 4.41 | 5.85 / 3.31 / 1.78 | 5.1 / 0.47 / 0.0461 | 3.8 / 0 / 0 |

### Bedau original normalisation (primary spaces)

| space | window | class | A_new excess | rho | rho_run | rho_shadow | D growth over window | D mean |
|---|---|---|---|---|---|---|---|---|
| cohort_h200k | 0k-1500k | unbounded (3b: Ā) | 152 | 0.36 | -0.217 | -0.578 | -0.0489 | 15252 |
| cohort_h200k | 100k-300k | none | -122 | 0.297 | -1.26 | -1.56 | 0.049 | 14982 |
| cohort_h200k | 300k-500k | none | -38.1 | 1.14 | -0.189 | -1.33 | -0.052 | 15765 |
| marker_0.01 | 0k-1500k | unbounded (3b: Ā) | 139 | 0.0238 | 0.0598 | 0.036 | -0.383 | 1629.8 |
| marker_0.01 | 0k-100k | unbounded (3b: Ā) | 39.3 | 0.0923 | 0.935 | 0.842 | -0.971 | 2232 |
| marker_0.01 | 100k-300k | bounded | 306 | -0.0729 | 0.142 | 0.215 | -0.0881 | 1792.6 |
| marker_0.01 | 300k-500k | unbounded (3b: Ā) | 129 | 0.18 | 0.203 | 0.023 | -0.109 | 1781.3 |

### Cohort lineage sorting (surviving lineages per cohort at a given age, mean over cohorts started in the window)

| age | cohort start window | cohorts | run | neutral | heritability null | run / null | excess frac run | null |
|---|---|---|---|---|---|---|---|---|
| age_20k | 0k-100k | 10 | 3063 | 1392 | 2267 | 1.35 | 0.394 | 0.405 |
| age_20k | 100k-300k | 20 | 3082 | 1522 | 2480 | 1.24 | 0.408 | 0.408 |
| age_20k | 300k-500k | 20 | 3063 | 1627 | 2623 | 1.17 | 0.411 | 0.405 |
| age_50k | 0k-100k | 10 | 831.2 | 574 | 928.2 | 0.895 | 0.182 | 0.164 |
| age_50k | 100k-300k | 20 | 799.2 | 630.2 | 1021 | 0.782 | 0.19 | 0.165 |
| age_50k | 300k-500k | 20 | 794.6 | 675 | 1096 | 0.725 | 0.191 | 0.165 |
| age_100k | 0k-100k | 10 | 176.9 | 298.6 | 479.9 | 0.369 | 0.107 | 0.0848 |
| age_100k | 100k-300k | 20 | 195.7 | 317 | 513 | 0.381 | 0.103 | 0.083 |
| age_100k | 300k-500k | 20 | 202.2 | 342.9 | 557.2 | 0.363 | 0.101 | 0.0831 |
| age_200k | 0k-100k | 10 | 33.7 | 156.6 | 245.2 | 0.137 | 0.0553 | 0.0447 |
| age_200k | 100k-300k | 20 | 42.55 | 159.2 | 261.5 | 0.163 | 0.058 | 0.0413 |
| age_200k | 300k-500k | 20 | 53.7 | 178.7 | 287.8 | 0.187 | 0.0442 | 0.0419 |

## 2D 64k_s0 original windows

windows: [[0, 120000], [120000, 1000000], [1000000, None]]; grid 10000; analysed to 1500000

### Classification (primary settings; sweep = none/bounded/unbounded counts)

| space | window | births/step | valid from | Bedau | Bedau sweep n/b/u | Channon | Channon informative sweep n/b/u | Channon full sweep n/b/u |
|---|---|---|---|---|---|---|---|---|
| cohort_h100k | 0k-1500k | 28.51 | 100000 | unbounded (3b: Ā) | 126/54/126 | bounded | 6/18/0 | 54/24/0 |
| cohort_h100k | 0k-120k | | 100000 | n/a | | n/a | | |
| cohort_h100k | 120k-1000k | 29.19 | 120000 | unbounded (3b: Ā) | 126/54/126 | bounded | 6/18/0 | 54/24/0 |
| cohort_h100k | 1000k-1500k | 27.04 | 1000000 | unbounded (3b: Ā) | 126/108/54 | bounded | 0/18/0 | 54/24/0 |
| cohort_h200k | 0k-1500k | 28.15 | 200000 | unbounded (3b: Ā) | 135/54/135 | bounded | 12/12/0 | 60/18/0 |
| cohort_h200k | 0k-120k | | 200000 | n/a | | n/a | | |
| cohort_h200k | 120k-1000k | 28.84 | 200000 | unbounded (3b: Ā) | 135/54/135 | bounded | 9/15/0 | 57/21/0 |
| cohort_h200k | 1000k-1500k | 27.04 | 1000000 | unbounded (3b: Ā) | 126/66/114 | bounded | 6/18/0 | 54/24/0 |
| cohort_h500k | 0k-1500k | 27.46 | 500000 | unbounded (3b: Ā) | 171/54/135 | bounded | 12/18/0 | 48/30/0 |
| cohort_h500k | 0k-120k | | 500000 | n/a | | n/a | | |
| cohort_h500k | 120k-1000k | 27.88 | 500000 | unbounded (3b: Ā) | 180/117/63 | bounded | 9/21/0 | 51/27/0 |
| cohort_h500k | 1000k-1500k | 27.04 | 1000000 | unbounded (3b: Ā) | 171/45/126 | bounded | 6/18/0 | 30/48/0 |
| marker_0.003 | 0k-1500k | 28.97 | 0 | unbounded (3b: Ā) | 198/153/135 | bounded | 15/9/0 | 45/33/0 |
| marker_0.003 | 0k-120k | 35.32 | 0 | none | 171/99/90 | bounded | 6/12/0 | 66/12/0 |
| marker_0.003 | 120k-1000k | 29.19 | 120000 | unbounded (3b: Ā) | 189/153/126 | bounded | 15/9/0 | 51/27/0 |
| marker_0.003 | 1000k-1500k | 27.04 | 1000000 | unbounded (3b: Ā) | 216/162/108 | bounded | 9/9/0 | 27/51/0 |
| marker_0.01 | 0k-1500k | 28.97 | 0 | unbounded (3b: Ā) | 225/159/66 | bounded | 9/15/0 | 57/21/0 |
| marker_0.01 | 0k-120k | 35.32 | 0 | unbounded (3b: Ā) | 198/114/84 | bounded | 6/12/0 | 60/18/0 |
| marker_0.01 | 120k-1000k | 29.19 | 120000 | unbounded (3b: Ā) | 225/99/108 | bounded | 6/12/0 | 60/18/0 |
| marker_0.01 | 1000k-1500k | 27.04 | 1000000 | bounded | 198/252/0 | bounded | 9/15/0 | 51/27/0 |
| marker_0.03 | 0k-1500k | 28.97 | 0 | bounded | 189/189/36 | bounded | 3/15/0 | 57/21/0 |
| marker_0.03 | 0k-120k | 35.32 | 0 | unbounded (3b: Ā) | 207/111/60 | bounded | 3/15/0 | 57/21/0 |
| marker_0.03 | 120k-1000k | 29.19 | 120000 | unbounded (3b: Ā) | 225/153/36 | bounded | 0/18/0 | 60/18/0 |
| marker_0.03 | 1000k-1500k | 27.04 | 1000000 | unbounded (3b: Ā) | 180/180/54 | bounded | 3/15/0 | 57/21/0 |
| marker_0.1 | 0k-1500k | 28.97 | 0 | bounded | 189/171/0 | bounded | 6/12/0 | 66/12/0 |
| marker_0.1 | 0k-120k | 35.32 | 0 | unbounded (3b: Ā) | 180/105/75 | bounded | 6/6/0 | 66/12/0 |
| marker_0.1 | 120k-1000k | 29.19 | 120000 | bounded | 189/156/15 | bounded | 6/6/0 | 66/12/0 |
| marker_0.1 | 1000k-1500k | 27.04 | 1000000 | bounded | 180/165/15 | bounded | 6/12/0 | 66/12/0 |

### Channon component-normalised statistics (primary spaces)

theta is the primary q=5% threshold of that run; A^N_new in normalised activity per component per 10k-step grid time; rho_* in units of the birth rate (>0.01 would count as growth).

| space | window | theta | A^N_new run | neutral | permuted | positive frac | A^N_cum end | median A^N end | rho_cum | rho_med |
|---|---|---|---|---|---|---|---|---|---|---|
| cohort_h200k | 0k-1500k | 3162 | 11.2 | 2.81 | 0.357 | 1 | 3.62e+06 | 5.85 | -0.0122 | -0.000584 |
| cohort_h200k | 120k-1000k | 3162 | 10.9 | 3.27 | 0.459 | 1 | 3.1e+06 | 11.7 | -0.0442 | 0.00455 |
| cohort_h200k | 1000k-1500k | 3162 | 11.7 | 2.07 | 0.195 | 1 | 3.62e+06 | 5.85 | -0.0219 | -0.00483 |
| marker_0.01 | 0k-1500k | 3162 | 5.69 | 0.2 | 0.0815 | 1 | -2.84e+05 | 4.66 | -0.0129 | -1.66e-05 |
| marker_0.01 | 0k-120k | 3162 | 11.1 | 0.177 | 0.0796 | 1 | 1.82e+05 | 6 | 0.0209 | 0.00226 |
| marker_0.01 | 120k-1000k | 3162 | 5.22 | 0.201 | 0.096 | 1 | -1.82e+05 | 6 | -0.0155 | -4.02e-05 |
| marker_0.01 | 1000k-1500k | 3162 | 5.23 | 0.205 | 0.0565 | 1 | -2.84e+05 | 4.66 | -0.00189 | -0.000137 |

### A^N_new at common thresholds (run / neutral / permuted)

| space | window | theta=316.228 | theta=1000 | theta=3162.28 | theta=10000 |
|---|---|---|---|---|---|
| cohort_h200k | 0k-1500k | 9.58 / 13.2 / 8.07 | 9.91 / 10.9 / 5.2 | 11.2 / 2.81 / 0.357 | 12.5 / 0.00676 / 0 |
| cohort_h200k | 120k-1000k | 9.38 / 13.9 / 8.46 | 9.67 / 12 / 5.75 | 10.9 / 3.27 / 0.459 | 11.7 / 0.011 / 0 |
| cohort_h200k | 1000k-1500k | 9.91 / 12 / 7.44 | 10.3 / 9.23 / 4.32 | 11.7 / 2.07 / 0.195 | 13.8 / 0 / 0 |
| marker_0.01 | 0k-1500k | 6.15 / 6.95 / 4.37 | 5.9 / 3.31 / 1.71 | 5.69 / 0.2 / 0.0815 | 4.49 / 0 / 0 |
| marker_0.01 | 0k-120k | 10.8 / 13.5 / 9.76 | 11.3 / 5.66 / 3.3 | 11.1 / 0.177 / 0.0796 | 6.54 / 0 / 0 |
| marker_0.01 | 120k-1000k | 5.84 / 6.85 / 4.1 | 5.48 / 3.44 / 1.73 | 5.22 / 0.201 / 0.096 | 3.68 / 0 / 0 |
| marker_0.01 | 1000k-1500k | 5.6 / 5.56 / 3.57 | 5.34 / 2.53 / 1.29 | 5.23 / 0.205 / 0.0565 | 5.43 / 0 / 0 |

### Bedau original normalisation (primary spaces)

| space | window | class | A_new excess | rho | rho_run | rho_shadow | D growth over window | D mean |
|---|---|---|---|---|---|---|---|---|
| cohort_h200k | 0k-1500k | unbounded (3b: Ā) | 152 | 0.36 | -0.217 | -0.578 | -0.0489 | 15252 |
| cohort_h200k | 120k-1000k | unbounded (3b: Ā) | 96.9 | 0.562 | -0.426 | -0.988 | -0.00961 | 15379 |
| cohort_h200k | 1000k-1500k | unbounded (3b: Ā) | 240 | 0.223 | 0.12 | -0.103 | -0.0985 | 15048 |
| marker_0.01 | 0k-1500k | unbounded (3b: Ā) | 139 | 0.0238 | 0.0598 | 0.036 | -0.383 | 1629.8 |
| marker_0.01 | 0k-120k | unbounded (3b: Ā) | 49.6 | 0.082 | 0.827 | 0.745 | -0.874 | 2155.9 |
| marker_0.01 | 120k-1000k | unbounded (3b: Ā) | 175 | 0.0641 | 0.0885 | 0.0244 | -0.28 | 1636.3 |
| marker_0.01 | 1000k-1500k | bounded | 96.9 | -0.0126 | -0.00512 | 0.00743 | -0.193 | 1492 |

### Cohort lineage sorting (surviving lineages per cohort at a given age, mean over cohorts started in the window)

| age | cohort start window | cohorts | run | neutral | heritability null | run / null | excess frac run | null |
|---|---|---|---|---|---|---|---|---|
| age_20k | 0k-120k | 12 | 3094 | 1408 | 2302 | 1.34 | 0.396 | 0.406 |
| age_20k | 120k-1000k | 88 | 3049 | 1726 | 2767 | 1.1 | 0.414 | 0.405 |
| age_20k | 1000k-1500k | 49 | 2993 | 1908 | 3049 | 0.982 | 0.424 | 0.405 |
| age_50k | 0k-120k | 12 | 818.9 | 579.9 | 945 | 0.867 | 0.184 | 0.165 |
| age_50k | 120k-1000k | 88 | 743 | 714.8 | 1146 | 0.648 | 0.196 | 0.166 |
| age_50k | 1000k-1500k | 46 | 672.3 | 789.5 | 1253 | 0.537 | 0.212 | 0.168 |
| age_100k | 0k-120k | 12 | 180.1 | 300.3 | 485.8 | 0.371 | 0.105 | 0.0847 |
| age_100k | 120k-1000k | 88 | 179.9 | 360.9 | 578 | 0.311 | 0.104 | 0.0836 |
| age_100k | 1000k-1500k | 41 | 155.2 | 397.6 | 628.9 | 0.247 | 0.117 | 0.0853 |
| age_200k | 0k-120k | 12 | 32.67 | 156.9 | 246.2 | 0.133 | 0.056 | 0.0444 |
| age_200k | 120k-1000k | 88 | 45.74 | 183.1 | 292.3 | 0.156 | 0.0494 | 0.042 |
| age_200k | 1000k-1500k | 31 | 40.55 | 193.3 | 312.8 | 0.13 | 0.0501 | 0.0428 |
