# Landing Phase I for the stock Gazebo x500

The landing defaults assume the powerloop airframe: thrust up to 4 g and body rates up to
18 rad/s. The stock x500 has 1.69 g and very little yaw torque, so the shipped backup
(`checkpoints/landing_phase1/floor_rec10_td3_seed0`) asks for more than the vehicle has and its
base set is not certified there. This batch re-derives the design for the x500 and retrains.

Nothing of the original formulation is removed. Everything below is an option that is off by
default, and the shipped 4 g backup reproduces its recorded self-check with this code.

## Run it

Inside the container (or from the host with `docker exec`, see the script header):

```bash
bash ~/ws_shared/PS2-RL/docker_batch/run_landing_x500_phase1.sh 2>&1 | tee ~/ws_shared/PS2-RL/outputs/landing_x500.out
```

Stages, each skipped when its result exists:

| stage | what | where |
|---|---|---|
| design | vehicle constants -> Phase-I config (kept if the file exists; `REDESIGN=1` derives it again) | `docker_batch/configs/landing_x500_cone45_r0p3.json` |
| certificate | full check of the base set; the batch stops if it fails | `outputs/landing_x500/certificate.json` |
| baseline | recoverability of the hover LQR alone, no training | `outputs/landing_x500/lqr_baseline.json` |
| B | Phase-I training, seeds 0 and 1, 5M steps each, two at a time on the GPU | `outputs/landing_x500/phase1/x500_rec10_td3_seed{0,1}` |
| C | self-check of each run (recoverability, cone and floor stress tests, QP health) | `outputs/landing_x500/checks/` |
| D | reach tests of each run under three filter settings | `outputs/landing_x500/tests/` |
| F | report | `outputs/landing_x500/REPORT.md` |
| regions | each trained backup against the hover LQR alone | `outputs/landing_x500/regions.json` |

The evaluations of each training run are printed as they come (`[x500_rec10_td3_seed0] step=... [eval val] ...`);
the full logs are in `outputs/landing_x500/logs/`. To stop, Ctrl-C; running the same command again
skips the finished stages and runs (an unfinished training run starts over).

Send back `outputs/landing_x500.out`, `outputs/landing_x500/REPORT.md`, `regions.json`, `logs/`,
`checks/`, `tests/` and the two run folders in `phase1/`.

## What the design stage does

`scripts/design_landing_base_set.py` reads the vehicle constants
(`docker_batch/configs/vehicles/gz_x500.json`: masses, motor constant, rotor speed range, rotor
positions, moment constant, inertia) and a reference design (the shipped backup's `configs.json`),
and derives three things together.

**1. Action box.**

| | stock x500 | rule | original |
|---|---|---|---|
| thrust | 1.99 to 14.95 m/s² | 10 % of the per-motor thrust range kept free at both ends for torque (`--thrust_reserve`) | 0 to 39.24 |
| roll, pitch rate | 3.84 rad/s | rate reached in 0.04 s at hover (`--rate_rise_time`), capped by the autopilot's 220 °/s attitude-rate limit | 18 |
| yaw rate | 0.203 rad/s | same rule; yaw angular acceleration is about 20 times lower than roll/pitch | 18 |

The rates are not hard limits of the vehicle (PX4 does not clip direct rate setpoints). The model
assumes a commanded rate is reached within a control step, so the limit is what the vehicle
reaches in about two steps. `--a_cmd_min`, `--a_cmd_max`, `--omega_max`, `--omega_z_max` set any of
them directly, e.g. after a rate step test in SITL.

**2. Hover LQR.** The input weights are rescaled to the new input ranges, `r <- r (m_ref / m)^2`
with `m` the one-sided input margin around hover. This keeps the base set as large as before.
With the original weights the x500 limits would shrink the base set to 0.17 m, and to 3 cm once the yaw limit is counted.
Yaw is set by its own rule (the yaw loop is decoupled): the base set is 45° wide in yaw
(`--base_set_yaw_deg`) and yaw is exactly as input-feasible as the most limiting other input.
A 60° wide base set certifies at the same level; at 80° it no longer does, because the slow yaw
loop cannot absorb the coupling between yaw and tilt.

**3. Base-set level.** `c_B = 0.77 x` the smallest level bound (`--safety_factor`; 0.77 is the
ratio of the original design).

**Design region.** Reset to the defaults and widened only as far as the task references need
(`TASK_REFS`, default the six `b*` descent references of the Phase-II batch): speed up to 2 m/s
instead of 4. The yaw range is 25° instead of 45°: that is what the vehicle can yaw back within
the 2 s backup horizon, and the task flies at zero yaw (Phase II starts with no yaw offset).
In a CPU trial (one seed each, 1.2M steps) the 25° region learned much faster (weighted
recoverability 0.44 against 0.05 after 0.55M steps) and both ended equal on the 25° test states
(0.506 and 0.502). To train on 45° anyway: `P1_EXTRA_ARGS="--region_yaw_max_deg 45"`.

Result for the stock x500 (`docker_batch/configs/landing_x500_cone45_r0p3.json`):

| | stock x500 | original |
|---|---|---|
| level bounds: input, Lyapunov, ground, cone, recovery cone | 28.7, 28.7, 19.7, 32.8, 19.6 | 17.9, 15.7, 18.7, 31.8, 17.6 |
| c_B at z_des = 1.25 m | 15.1 | 12 |
| base set: horizontal, vertical | 0.82 m, 0.66 m | 0.74 m, 0.60 m |
| base set: horizontal, vertical speed | 2.18, 1.25 m/s | 2.33, 1.57 m/s |
| base set: tilt, yaw | 38°, 45° | 92°, 80° |
| hover LQR yaw time constant | 5.2 s | 0.3 s |

To change the design, pass the flags through `DESIGN_ARGS` and set `REDESIGN=1`, for example

```bash
REDESIGN=1 DESIGN_ARGS="--thrust_reserve 0.15 --omega_z_max 0.5" OUT=outputs/landing_x500_b \
CONFIG=docker_batch/configs/landing_x500_b.json bash docker_batch/run_landing_x500_phase1.sh
```

For another vehicle, write its constants file next to `gz_x500.json` and set `VEHICLE`, `CONFIG`, `OUT`, `NAME`.

## The baseline to beat

The hover LQR alone (the base controller used everywhere, no learned policy) is a valid backup
for the filter and needs no training. `scripts/evaluate_landing_backup_regions.py` measures it with
Phase I's own evaluation:

| recoverability (test states) | weighted | general | edge | shell |
|---|---|---|---|---|
| original design, hover LQR alone | 0.313 | 0.532 | 0.125 | 1.000 |
| original design, shipped learned backup | 0.509 | 0.785 | 0.335 | 1.000 |
| x500 design, hover LQR alone | 0.474 | 0.744 | 0.297 | 1.000 |
| x500 design, CPU trial backup (1.2M steps, one seed) | 0.506 | 0.747 | 0.344 | 1.000 |

The two designs are measured on their own regions (the x500 one is slower, 2 m/s instead of 4), so
compare within a design, not across. For the x500 the gently tuned LQR is already a strong backup
and the trial run is only 0.03 above it; whether 5M steps add more is what the GPU run shows.
A trained backup is worth using if it beats its baseline: the batch prints both, and the reach
tests include a setting that runs the filter with the LQR as the backup (`a10_f20_lqr`).

## Checked on the CPU before sending

- The shipped 4 g backup: its self-check with this code is identical, value for value, to the one of the code
  before these changes run side by side (recoverability 0.758 / 0.318 / 1.000, cone margin +0.0025 m,
  floor +0.0004 m, slack 6.0e-4, no QP fallbacks).
- x500 certificate: c_B = 15.1 against a bound of 19.6; 200,000 samples of the base set inside the cone
  (margin 0.54 m); 4096 LQR rollouts from the boundary stay inside (largest level 14.99).
- x500 trial backup (1.2M steps): self-check PASS. With the nominal controller pushing 1 m outside the wall
  the filtered vehicle stays inside (margin +0.0008 m); pushing 0.3 m below the pad it stays above (+0.0002 m);
  largest slack 2.3e-4, no QP fallbacks.
- Reach tests of that backup, filter setting alpha 10 / alpha_floor 20: 18 of 18 LQR landings touch down
  (median 2.58 s, rest height 5 mm, cone margin never negative); 6 of 6 runs tracking the corner-cutting reference
  touch down (median 1.91 s; unfiltered they all leave the cone). With alpha 4 the vehicle stops 5 cm up, as
  with the original design. With the LQR as the backup: 18 of 18, and 7 of 7 tracking runs (median 2.36 s).
- The whole batch ran end to end with a tiny training run; Phase-II tracker and PS2 training start with an
  x500 backup and take its action box, yaw limit included.

## The options in the code

| option | where | default |
|---|---|---|
| `omega_z_max` (separate yaw-rate limit) | `QuadrotorLandingConfig`, `QuadrotorEnvConfig`; honoured by the hover LQR, the Phase-I environment, the filter, the Phase-II landing entry and its filter evaluation | 0 = yaw uses `omega_max` |
| `--config_json` and every landing field as a flag | `scripts/certify_landing_base_set.py` | the defaults |
| vehicle-driven design | `scripts/design_landing_base_set.py` (new) | |
| baseline evaluation | `scripts/evaluate_landing_backup_regions.py` (new) | |
| `--reference` (start state without a Phase-II run) | `scripts/test_landing_floor_reach.py` | off |
| `START_REF`, `FOLLOW` | `docker_batch/run_landing_floor_overnight.sh` | off |

The certificate also counts the gentle-recovery bounds when the config trains with that envelope.

## Phase II with these limits

A Phase-II policy must be trained with the same action box as the backup it is filtered with.
`scripts/train_phase2_landing.py --backup <x500 run>` takes the box, including the yaw limit, from
the backup. A policy trained with the 4 g box cannot be combined with the x500 backup; the filter
evaluation refuses the pair.
