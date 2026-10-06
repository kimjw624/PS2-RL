# Results for the shipped landing backup `floor_rec10_td3_seed0`

Checkpoint: `checkpoints/landing_phase1/floor_rec10_td3_seed0/` (TD3, seed 0, 5M steps, 24 min on an RTX 4050).

- **Task:** 45° approach cone over a pad of radius r0 = 0.3 m (ε = 0.015 m), safe set S = cone ∩ {z ≥ z_pad}.
- **Base set:** 9-D hover LQR ellipsoid at z_des = 1.25 m, level c_B = 12.
- **Training option:** gentle-recovery envelope κ = 10 on cone and floor (`floor_rec10`).
- **Filter setting used below:** `alpha=10, alpha_floor=20`, relative-time rows, discrete sensitivities, discrete-time safeguard.

The training numbers come from the GPU run that produced the checkpoint. Everything else was run on CPU from this repository with the commands given.

## Training (`training/`)

Fraction of held-out initial states from which the backup reaches the base set within the horizon (N = 100 steps), test split, at the checkpoint selected on the validation split:

| region | n | reaches the base set inside S and the envelope | crash | timeout |
|---|---|---|---|---|
| general (altitude-uniform over the cone) | 1024 | 0.785 | 0.215 | 0 |
| low-altitude cone edge | 1024 | 0.335 | 0.665 | 0 |
| capture shell | 512 | 1.000 | 0 | 0 |
| weighted (1/3/0.5) | | 0.509 | | |

- "Crash" here also counts leaving the gentle-recovery envelope, which is a training constraint and not part of S. Counting S only, the fractions are 0.898 (general), 0.516 (edge) and 1.000 (shell); see the self-check.
- Weighted validation recoverability rose from 0.015 (untrained actor) to 0.521; the selected checkpoint is the last one (step 5M).
- `learning_curves.png`: validation recoverability during training. `history.json`: losses and per-evaluation metrics.

## Certificate (`certificate/`, and `certificate.json` in the checkpoint folder)

Level bounds for c_B = 12 at z_des = 1.25 m:

| bound | value |
|---|---|
| input feasibility c_U | 17.88 |
| quaternion chart c_chart | 23.18 |
| cone containment c_cone | 31.83 |
| ground clearance c_ground | 18.71 |
| envelope, cone c_recovery_cone | 17.58 |
| envelope, floor c_recovery_floor | 39.67 |
| adversarial Lyapunov c_Lyap | 15.67 |

- c_B = 12 is below every bound; the adversarial one-step Lyapunov ratio at c_B is 0.967.
- Numerical checks in `base_set_certificate.json`: 200 000 uniform samples in B have min h_cone = 0.633 m, and 4096 base-controller rollouts from the level set V = c_B keep V ≤ 11.60.
- c_Lyap is a numerical search, not a proof.

```bash
JAX_PLATFORMS=cpu python scripts/certify_landing_base_set.py --cone_r0 0.3 --cone_theta_deg 45 --cone_eps 0.015 \
    --out results/landing_phase1/floor_rec10_td3_seed0/certificate/base_set_certificate.json
```

## Filter self-check (`self_check/`): PASS

| check | result |
|---|---|
| recoverability on fresh samples vs Phase I | general 0.758 (0.785), edge 0.318 (0.335), shell 1.000 (1.000); within the ±0.060 sampling band |
| cone stress test: a controller aiming 1 m outside the wall, 32 recoverable starts, 150 steps | unfiltered 100 % leave the cone; filtered 0 %, min h_cone +0.0025 m |
| floor stress test: a controller aiming 0.3 m below the pad | unfiltered min height −4.19 m; filtered +0.0004 m |
| QP health | max slack 6.0e-4, no solver fallbacks (QP in float64), filter active on 93 % of steps |

In float32 the same test passes, with solver fallbacks on 4.5 % of steps (the backup action is applied there).

```bash
JAX_PLATFORMS=cpu python scripts/check_landing_bcbf.py --backup checkpoints/landing_phase1/floor_rec10_td3_seed0 \
    --cbf_override alpha=10 --cbf_override alpha_floor=20 --out results/landing_phase1/floor_rec10_td3_seed0/self_check
```

## Reach tests (`reach_tests/`)

Can a filtered vehicle come down to the pad? Touchdown means height ≤ 2 cm over the pad disk at speed ≤ 0.3 m/s. Rollouts are 6 s long.

| test | starts | touchdown | time to touchdown [s] | min h_cone [m] | min height [m] | max slack |
|---|---|---|---|---|---|---|
| T1, LQR to the pad centre | 18 (16 random C_N starts, hover over the pad, reference start) | 100 % | median 2.54 (0.90–3.40) | +0.0000 | +0.0010 | 5.7e-4 |
| T3, LQR tracking the cone-cutting reference | 9, all in C_N | 100 % | median 2.00 (1.86–2.12) | +0.0155 | +0.0039 | 1.8e-4 |

- **No solver fallbacks** in either test.
- **T3 without the filter:** the tracker leaves the cone on every start.
- **Target below the floor** (cone-apex depth): the filter refuses it and the vehicle settles on the pad (min height +0.001 m).
- **Target at the pad edge** (0.9 r0): the vehicle stays inside the cone but is held 2.2 cm up, just above the 2 cm touchdown threshold. It lands at the centre, not at the rim.
- **Hover map** (`hover_maps.png`): holding still is allowed at 97 % of the grid points inside the cone and 95 % of those in its lower half; over the pad centre down to the lowest grid height above the floor, 5 cm.
- `setting_a10_f20.png`: T1 paths, T3 paths with and without the filter, and heights over time.

```bash
JAX_PLATFORMS=cpu python scripts/test_landing_floor_reach.py --backup checkpoints/landing_phase1/floor_rec10_td3_seed0 \
    --settings "a10_f20:alpha=10,alpha_floor=20" \
    --reference ps2rl/envs/assets/quadrotor_landing_cornercut_reference.npz \
    --tracking_reference ps2rl/envs/assets/quadrotor_landing_cornercut_reference.npz \
    --out results/landing_phase1/floor_rec10_td3_seed0/reach_tests
```

## Limits of these results

- Simulation model only: explicit Euler at 50 Hz with instantaneous body-rate tracking.
- About half of the low-altitude edge region is not recoverable, so starts there carry no guarantee.
- One seed is shipped.
