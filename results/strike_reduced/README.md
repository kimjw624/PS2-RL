# Strike task (reduced arena) — results

## Decision experiments (`decision_experiments/`)

Setup:
- PS2 striker `checkpoints/strike_reduced/ps2_striker_s0`;
- 128 test scenes (seed 9001), 15 s episodes;
- the 10 s numbers are the same episodes cut at 10 s.

Each folder has the evaluation table (`strike_filter_table.json` / `.md`) and per-episode records
(`strike_filter_episodes.json`: outcome code = index in `outcomes`; steps in units of `dt`).

| folder | filter / backup family | strike 15 s | strike 10 s | contact | lost-certificate episodes | median / max strike time |
|---|---|---|---|---|---|---|
| `old_cp` | **deployed**: discrete + escape variants + committed chaser prediction, Phase-I backup `phase1_td3_a0p2_seed0` | **1.000** | 0.938 | **0** | 0.039 | 5.19 / 13.7 s |
| `v2_cp` | moving-chaser Phase I (not in this branch) alone first + variants + committed prediction | 0.977 | 0.914 | 0 | 0.031 | 5.14 / 14.0 s |
| `bcbf_cp` | bCBF QP (minimal correction), safeguard = deployed filter | 0.969 | 0.859 | 0 | 0.039 | 5.57 / 14.7 s |

Every row has 0 contacts, with a 95 % Clopper–Pearson upper bound of 2.8 %.

## Earlier sparse-arena results (10 s episodes, same 128 scenes)

| nominal | filter | strike | contact | time-up |
|---|---|---|---|---|
| pursuit 6 m/s | none | 0.445 | 0.555 | 0 |
| pursuit | static chaser ball | 0.43 | 0.49 | 0.08 |
| pursuit | guidance, 1 escape | 0.773 | 0.008 | 0.22 |
| pursuit | guidance + variants | 0.828 | 0 | 0.172 |
| pursuit | bCBF (1-escape safeguard) | 0.695 | 0.031 | 0.273 |
| residual striker | none | 0.766 | 0.234 | 0 |
| residual striker | guidance + variants | 0.828 | 0 | 0.172 |
| PS2 striker | none | 0.625 | 0.375 | 0 |
| PS2 striker | guidance, 1 escape (as trained) | 0.898 | 0.016 | 0.086 |
| PS2 striker | guidance + variants | 0.938 | 0 | 0.062 |
| PS2 striker | bCBF (1-escape safeguard) | 0.828 | 0.055 | 0.117 |

## Final 3-seed results

To be added in `final/` from `outputs/strike_final/report/` (`docker_batch/run_strike_final.sh`).
