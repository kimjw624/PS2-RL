# Filter improvements — CPU check on 16 test scenes (2026-10-31)

Seed-0 models of the report run (`../final/run`), test scenes of seed 9001, 15 s.
- Scenes: 827, 69, 601, 271, 576, 972, 845, 480, 181, 322, 934, 994, 470, 509, 752, 949. That is the 7 scenes of the paper's trajectory figures, the 5 scenes where the filter acted most in the GPU evaluation, and 4 random scenes.
- Each folder holds the evaluation table and per-episode records of one setting (`scripts/eval_strike_filter.py --only_episodes ... --save_traj 1`, CPU).
- Small sample: it shows which options are worth the full test set (`docker_batch/run_strike_improve.sh`), not final numbers.

## Outcomes and command smoothness

Columns:
- **active**: share of steps in which the filter changed the command (λ < 1, committed plan or no certificate).
- **toggles/s**: switches between kept and changed, per second.
- **Σ|Δλ|/s**: total change of the blend per second.
- **‖Δu‖** and **‖Δu_nom‖**: mean step change of the applied and the nominal command, normalised by the actuator scale.
- **‖Δu‖ active**: the same, over the steps where the filter acted.
- **variant sw/s**: backup switches per second while the filter acts.
- **near**: share of steps with 0 ≤ h_c < 0.5 m.
- **no cert**: share of steps without a certificate.

Values are per-episode means.

| run | n | strike | contact | time-up | strike t [s] | active | toggles/s | Σ|Δλ|/s | ‖Δu‖ | ‖Δu_nom‖ | ‖Δu‖ active | variant sw/s | near h_c<0.5 | no cert |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| as in the report | 16 | 14 | 1 | 1 | 7.51 | 0.290 | 1.98 | 3.18 | 0.0634 | 0.0179 | 0.1234 | 2.32 | 0.163 | 0.0145 |
| lam_rise_max 0.05 | 16 | 15 | 1 | 0 | 7.48 | 0.278 | 0.76 | 0.96 | 0.0314 | 0.0163 | 0.0710 | 0.60 | 0.107 | 0.0136 |
| lam_rise_max 0.1 | 16 | 15 | 1 | 0 | 7.85 | 0.274 | 1.00 | 1.45 | 0.0389 | 0.0168 | 0.0937 | 0.99 | 0.155 | 0.0147 |
| prefer_committed_variant | 16 | 14 | 1 | 1 | 7.22 | 0.279 | 2.02 | 3.25 | 0.0640 | 0.0183 | 0.1295 | 0.46 | 0.149 | 0.0230 |
| ramp 0.05 + prefer_committed_variant | 16 | 14 | 1 | 1 | 6.83 | 0.287 | 0.84 | 0.98 | 0.0325 | 0.0171 | 0.0708 | 0.13 | 0.087 | 0.0162 |
| lambda_grid 9 levels | 16 | 14 | 1 | 1 | 7.20 | 0.280 | 1.91 | 2.88 | 0.0621 | 0.0181 | 0.1231 | 1.32 | 0.157 | 0.0221 |
| slide_along | 16 | 15 | 1 | 0 | 8.76 | 0.325 | 2.02 | 3.34 | 0.0750 | 0.0185 | 0.1391 | 2.10 | 0.181 | 0.0449 |
| best_effort_fallback | 16 | 15 | 0 | 1 | 7.84 | 0.283 | 1.98 | 3.16 | 0.0630 | 0.0178 | 0.1236 | 2.34 | 0.163 | 0.0214 |
| escape latch 0.3/1.0 | 16 | 12 | 1 | 3 | 6.25 | 0.248 | 0.99 | 1.14 | 0.0342 | 0.0176 | 0.0762 | 0.23 | 0.061 | 0.0119 |
| escape latch + ramp 0.05 | 16 | 14 | 1 | 1 | 7.86 | 0.328 | 0.65 | 0.74 | 0.0283 | 0.0168 | 0.0557 | 0.31 | 0.034 | 0.0136 |
| v2 backup | 16 | 15 | 0 | 1 | 7.47 | 0.265 | 2.10 | 3.47 | 0.0695 | 0.0180 | 0.1457 | 1.54 | 0.162 | 0.0000 |
| v2 backup + ramp 0.05 | 16 | 15 | 0 | 1 | 8.07 | 0.322 | 0.82 | 1.17 | 0.0346 | 0.0177 | 0.0668 | 0.71 | 0.115 | 0.0000 |
| chaser 1.25x faster than assumed | 16 | 16 | 0 | 0 | 7.20 | 0.207 | 1.63 | 2.39 | 0.0500 | 0.0178 | 0.1362 | 0.53 | 0.080 | 0.0113 |
| residual, as in the report | 16 | 12 | 0 | 4 | 9.48 | 0.404 | 1.85 | 4.34 | 0.0869 | 0.0189 | 0.1446 | 5.19 | 0.296 | 0.0124 |
| residual, ramp 0.05 | 16 | 11 | 0 | 5 | 8.87 | 0.390 | 0.67 | 1.34 | 0.0426 | 0.0182 | 0.0734 | 1.39 | 0.245 | 0.0034 |
| residual, v2 backup | 16 | 11 | 0 | 5 | 8.77 | 0.404 | 1.97 | 4.90 | 0.0982 | 0.0187 | 0.1427 | 2.62 | 0.254 | 0.0121 |
| residual, v2 + ramp 0.05 | 16 | 14 | 0 | 2 | 10.06 | 0.393 | 0.80 | 1.24 | 0.0430 | 0.0194 | 0.0677 | 1.03 | 0.198 | 0.0157 |

## Outcome per scene (S = strike with time [s], C = contact, T = time-up)

| run | 827 | 69 | 601 | 271 | 576 | 972 | 845 | 480 | 181 | 322 | 934 | 994 | 470 | 509 | 752 | 949 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| as in the report | S 7.9 | S 5.1 | S 6.5 | S 5.0 | S 4.8 | S 5.5 | C-chaser | T | S 9.2 | S 13.4 | S 12.7 | S 14.6 | S 5.2 | S 4.9 | S 5.8 | S 4.6 |
| lam_rise_max 0.05 | S 8.4 | S 5.5 | S 5.5 | S 5.0 | S 4.8 | S 5.6 | C-chaser | S 14.3 | S 7.8 | S 13.6 | S 5.2 | S 14.8 | S 5.8 | S 4.9 | S 6.5 | S 4.6 |
| v2 backup | S 6.9 | S 5.1 | S 5.5 | S 5.0 | S 4.8 | S 5.7 | S 7.8 | S 14.2 | S 10.2 | S 11.7 | T | S 13.8 | S 5.2 | S 4.9 | S 6.7 | S 4.6 |
| v2 backup + ramp 0.05 | S 7.6 | S 5.6 | S 10.7 | S 5.0 | S 4.8 | S 5.8 | S 8.4 | S 14.1 | S 10.4 | S 11.8 | T | S 14.5 | S 5.5 | S 4.9 | S 7.4 | S 4.6 |
| best_effort_fallback | S 7.9 | S 5.1 | S 6.5 | S 5.0 | S 4.8 | S 5.5 | S 12.8 | T | S 9.2 | S 13.0 | S 12.7 | S 14.6 | S 5.2 | S 4.9 | S 5.8 | S 4.6 |
| slide_along | S 7.8 | C-chaser | S 10.7 | S 5.0 | S 9.6 | S 5.5 | S 9.0 | S 13.7 | S 9.9 | S 13.2 | S 11.7 | S 14.6 | S 5.2 | S 4.9 | S 5.8 | S 4.6 |
| chaser 1.25x faster | S 6.4 | S 5.0 | S 5.3 | S 4.9 | S 5.0 | S 5.4 | S 10.5 | S 12.7 | S 9.2 | S 10.8 | S 7.8 | S 11.8 | S 5.2 | S 4.9 | S 5.8 | S 4.6 |
| residual, as in the report | S 7.1 | T | S 11.4 | T | T | S 14.5 | T | S 13.8 | S 10.4 | S 11.7 | S 10.0 | S 13.0 | S 5.2 | S 5.1 | S 6.5 | S 5.0 |
| residual, v2 + ramp 0.05 | S 14.7 | T | S 11.3 | S 12.5 | S 5.0 | T | S 13.5 | S 13.9 | S 10.8 | S 12.4 | S 10.5 | S 13.2 | S 5.2 | S 5.1 | S 7.7 | S 5.0 |

## Findings

- **Chattering.** The report's filter acts in 29 % of the steps and switches between keeping and changing the command about twice a second. The applied command changes 3.5 times as much as the nominal one (residual striker: 4.6 times).
- **Chattering goes with slow strikes.** The scenes with the most chattering are the slow ones: 480 (time-up), 322, 994 and 934 (12.7–14.6 s).
- **Blend ramp `lam_rise_max` 0.05.** It halves the command change and cuts the switches by about 60 %. Outcomes are equal or better: 480 strikes, and 934 strikes in 5.2 s instead of 12.7 s.
- **Wall-aware backup (v2).** It removes the one contact (scene 845, now a strike in 7.8 s) and every lost certificate in the sample.
- **v2 + ramp.** No contact, the smoother command, one time-up (934).
- **No improvement or worse:** `prefer_committed_variant` (fewer backup switches only), a 9-level blend grid (1.8 times the compute), the escape latch (slower), and `slide_along` (adds a chaser contact in scene 69).
- **`best_effort_fallback`.** It avoids 845, but without a certificate.
- **A chaser 25 % faster than the filter assumes:** 16/16 strikes and no contact on this sample.
