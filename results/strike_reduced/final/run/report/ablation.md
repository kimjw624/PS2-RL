Seeds [0, 1, 2]; rates are mean ± std over seeds, [pooled Clopper-Pearson 95 % interval]; contact = left the safe set (obstacle, floor, chaser); kept = mean fraction of the nominal command kept; lost certificate = fraction of episodes with at least one step without a certified plan.

| variant | strike | contact | time-up | strike time [s] | kept | lost certificate | seeds x episodes |
|---|---|---|---|---|---|---|---|
| deployed filter | 0.976 ± 0.008 [0.968, 0.982] | 0.001 ± 0.001 [0.000, 0.003] | 0.024 ± 0.009 [0.017, 0.031] | 6.11 | 0.90 | 0.015 | 2 x 1000 |
| hover-LQR backup instead of the learned one | 0.820 ± 0.020 [0.802, 0.836] | 0.010 ± 0.003 [0.006, 0.015] | 0.170 ± 0.018 [0.154, 0.188] | 7.30 | 0.71 | 0.085 | 2 x 1000 |
| no escape variants (1 escape) | 0.957 ± 0.000 [0.943, 0.969] | 0.017 ± 0.000 [0.010, 0.027] | 0.026 ± 0.000 [0.017, 0.038] | 5.88 | 0.90 | 0.036 | 1 x 1000 |
| no committed chaser prediction | 0.986 ± 0.000 [0.977, 0.992] | 0.002 ± 0.000 [0.000, 0.007] | 0.012 ± 0.000 [0.006, 0.021] | 5.95 | 0.91 | 0.021 | 1 x 1000 |
| no hover latch | 0.983 ± 0.000 [0.973, 0.990] | 0.001 ± 0.000 [0.000, 0.006] | 0.016 ± 0.000 [0.009, 0.026] | 5.90 | 0.91 | 0.012 | 1 x 1000 |
| static chaser ball instead of h_c | 0.676 ± 0.000 [0.646, 0.705] | 0.263 ± 0.000 [0.236, 0.291] | 0.002 ± 0.000 [0.000, 0.007] | 5.44 | 0.75 | 0.389 | 1 x 1000 |
