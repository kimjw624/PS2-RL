# UE-bCBF study with the stock-x500 limits (landing inside the cone + outside-cone strike)

The disturbance-aware (UE-bCBF) procedure of the 4 g study, repeated with the limits of the stock Gazebo
x500. Same steps for both tasks:

1. **Phase I** safe-arrival backup, trained on the plain sets without a disturbance ("SA-plain", the
   method the 4 g outside-cone comparison chose). It is used blind to the disturbance estimate inside the
   UE filter.
2. **Stage V**: a vanilla tracker (SAC, no filter).
3. **Stage P**: PS2 fine-tuning through the UE-bCBF CIL, warm-started from stage V.
4. Evaluation and figures.

| | landing inside the cone | outside-cone strike |
|---|---|---|
| task file | `docker_batch/configs/landing_ue_x500.json`; deep-dip variant `landing_ue_x500_deepdip.json` | `docker_batch/configs/cone_obstacle_ue_x500.json` |
| Phase I | **reused**: `checkpoints/landing_phase1/x500_rec10_td3_seed1` (the better of the two x500 runs, `run_landing_x500_phase1.sh`) | **trained**: SA-plain, 3M steps (GPU), then compared with the LQR alone |
| reference | corner-cut slowed: 4 s from (−1.6, 0, 2) m, peak 1.53 m/s, 0.18 m outside the cone; deep dip: 4.5 s, peak 1.49 m/s, 0.56 m outside | L3-1-s: 7.6 s from (−3, 0, 2.2) m, peak 1.31 m/s |
| base set | c_B = 15.1 (x500 landing design) | c_B = 20 (was 12) |
| disturbance | δ_d 0.5 m/s² at **0.025 Hz** (δ_v 0.079 m/s³), λ 20, ē 0.02, tube ×1.2 (see check 8) | δ_d 0.5 m/s², 0.05 Hz (δ_v 0.157 m/s³), as for 4 g |
| filter | α 10 / α_floor 20 | α 10 / α_floor 20 / base 2 |

**Update (10/17): the landing's first disturbance setting could not land.** With 0.05 Hz (δ_v 0.157 m/s³) the UE
filter keeps the x500 at least about 0.2 m above the pad, so no landing policy can touch down through it (check 8).
At 0.025 Hz (same 0.5 m/s² amplitude, half the rate of change) the reference tracker lands 100 % through the
UE filter. `landing_ue_x500.json` now uses 0.025 Hz. A landing run made with the first version (0.05 Hz)
shows landed 0 % through the UE filter; keep it as the record of why, and train the landing again into a new
folder (below). The cone task is not affected (its target is recoverable at 0.05 Hz).

## Run

```bash
docker exec -d px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_x500_ue.sh > ~/ws_shared/PS2-RL/outputs/x500_ue.out 2>&1"
docker exec -it px4sitl bash -c "tail -F ~/ws_shared/PS2-RL/outputs/x500_ue.out"
```

Order (`STAGES="L_V L_PEF C_P1 C_V C_PEF"`; each sub-stage skips finished runs, so the batch can be restarted):

| stage | what | device | log (`outputs/x500_ue_logs/`) |
|---|---|---|---|
| L_V | landing vanilla trackers, reward scales 0.1 and 1 in parallel, 3M steps | GPU | `land_V.out` + `outputs/landing_phase2_ue_x500/logs/` |
| L_PEF | landing PS2 1M steps, evaluation, figures; **in the background** while the GPU runs the cone stages | CPU | `land_PEF.out` |
| C_P1 | cone Phase I SA-plain (3M steps), then `compare_phase1_cone_obstacle.py` (LQR unless SA is ≥ 2 points better) | GPU | `cone_P1.log`, `cone_P1_compare.log` |
| C_V | cone vanilla trackers, reward scales 0.01 and 1, 3M steps | GPU | `cone_V.out` |
| C_PEF | cone PS2 1M steps, evaluation, figures (after L_PEF; `CPU_PARALLEL=1` starts it at once) | CPU | `cone_PEF.out` |

The stage-P trainers honour `touch <run>/STOP`. On the 4 g runs, stage P levelled off by about 0.5M steps.

**Landing again, both references** (after the 10/17 update; a pre-check, then stage V of each on the GPU one after
the other, stages P E F of each on the CPU as soon as its stage V is done; outputs
`outputs/landing_phase2_ue_x500_v2/{cornercut,deepdip}/`):

```bash
docker exec -d px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_x500_landing_variants.sh > ~/ws_shared/PS2-RL/outputs/x500_landing_variants.out 2>&1"
docker exec -it px4sitl bash -c "tail -F ~/ws_shared/PS2-RL/outputs/x500_landing_variants.out"
```

One task at a time:
- Landing: `docker_batch/run_landing_ue_twostage.sh`. Use `STAGES=V` on the GPU, then `PS2RL_ALLOW_CPU=1 JAX_PLATFORMS=cpu STAGES="P E F"`.
- Cone: `docker_batch/run_cone_obstacle_ue.sh`, with
  ```
  TASK=docker_batch/configs/cone_obstacle_ue_x500.json
  OUT=outputs/cone_obstacle_phase2_ue_x500
  BACKUP_JSON=outputs/cone_obstacle_phase1_x500/compare/chosen_backup.json
  ```

**Outputs:**
- Landing (`outputs/landing_phase2_ue_x500/`):
  - `landing_ue_twostage.png`: tracker without the CIL / tracker + UE-CIL / Phase II.
  - `landing_ue_twostage_with_nominal.png`: adds the nominal bCBF.
  - `landing_ue_phase2_filters.png`: Phase II through UE / nominal / no filter.
  - `landing_ue_curves.png`.
- Cone (`outputs/cone_obstacle_phase2_ue_x500/`): `cone_ue_progress*.png`, the GIF/MP4 animations and `cone_ue_curves.png`, as in the 4 g study.

## Vehicle (stock x500, `docker_batch/configs/landing_x500_cone45_r0p3.json`)
- Thrust 1.99–14.95 m/s² (1.52 g, 10 % of the motor range kept in reserve).
- Roll/pitch rate ±3.84 rad/s (the autopilot's limit); yaw rate ±0.203 rad/s.
- Hover LQR re-weighted for these limits.

The 4 g box was 0–39.24 m/s² and ±18 rad/s.

## What was checked, and what changed

### 1. Is c small enough? (base set safe and inside the input box)

The base set is $B = \{e : e^\top P e \le c\}$. For a linear quantity $w^\top e$ (an input, a barrier's gradient, a position):

$$\max_{e^\top P e \le c} w^\top e = \sqrt{c\, w^\top P^{-1} w},$$

so each requirement "$w^\top e \le$ room" gives an upper bound on c.

| bound | landing (x500) | cone (x500) | cone (4 g, for reference) |
|---|---|---|---|
| input box $c_U$ | 28.66 | 28.66 | 17.88 |
| quaternion chart $c_\text{chart}$ | 103.1 | 103.1 | 23.18 |
| Lyapunov decrease $c_\text{Lyap}$ | 28.66 | ≥ 28.66 | 15.67 |
| ground / geometric | 19.72 (ground), 32.78 (cone) | **22.20** (funnel and floor) | 21.54 |
| gentle-recovery rule (κ = 10) | **19.62** (cone) | – | – |
| chosen c_B | 15.1 = 0.77 × 19.62 | **20** = 0.90 × 22.20 | 12 |

The trainer re-checks this before training: `train_phase1_cone_obstacle_ue.py` prints `[base set] c_B 20.00 c_U 28.66 c_chart 103.11 c_geometric 22.20` and stops if c_B exceeds a bound.

### 2. Is c large enough? (the UE tube must leave a non-empty tightened base set)

The filter needs each backup rollout to end in $\{V \le c_B - m_\text{base}(s_N)\}$, with

$$s_{k+1} = \lVert F_k\rVert_P\, s_k + \Delta t\,\gamma\, q(\tau_k), \qquad q(\tau) = \bar e + \min(\delta_v \tau,\, 2\delta_d),$$

$$m_\text{base} = 2\sqrt{c_B}\, s_\text{eff} + s_\text{eff}^2, \qquad m_\text{cone} = L_\text{cone}\, s_\text{eff}, \qquad m_\text{floor} = L_\text{floor}\, s_\text{eff}, \qquad s_\text{eff} = 1.2\, s.$$

Here $\gamma = \sqrt{\lambda_\max(P_{vv})}$ is how far a unit disturbance acceleration moves the LQR metric.

| | γ | $\lVert F\rVert_P$ | $s_N$ (T = 2 s) | m_base / c_B | level left | funnel / floor margin |
|---|---|---|---|---|---|---|
| 4 g (landing and cone, c_B 12) | 2.489 | 0.9814 | 0.503 | 4.55 / 12 | 7.45 (worst 3.64) | 0.17 / 0.11 m |
| x500 landing (c_B 15.1) | 3.685 | 0.9961 | 1.137 | 12.47 / 15.1 | **2.63** (worst 0.63) | 0.37 / 0.23 m |
| x500 cone (c_B 20) | 3.685 | 0.9961 | 1.137 | 14.07 / 20 | **5.93** (worst 3.70) | 0.37 / 0.23 m |
| x500 landing at 0.025 Hz (c_B 15.1, used since 10/17) | 3.685 | 0.9961 | 0.629 | 6.43 / 15.1 | **8.67** (worst 7.63) | 0.20 / 0.13 m |

("worst" = no contraction, $\lVert F\rVert_P = 1$.)

- **The x500 tube is 2.3 times wider.** Its LQR is weighted for small inputs, so its metric is more sensitive to velocity errors (γ 3.69 vs 2.49) and it hardly contracts within 2 s (0.996 per step vs 0.981).
- **Cone:** with c_B = 12 the x500 would leave only 12 − 11.3 = 0.7 (none in the worst case). Hence c_B = 20, still under the 22.2 bound.
- **Landing:** c_B cannot grow past 15.1 without lowering the safety factor (the binding 19.6 comes from the cone's gentle-recovery rule).
  - At 0.05 Hz the 2.63 left was enough for the starts (94.7 % in the tightened recoverable set), but **not for the end of the landing**: over the pad the backup cannot reach so small a tightened base set within 2 s (check 8). A positive level left is necessary, not sufficient.
  - Within the 2 s horizon $\delta_v \tau \le 0.31 < 2\delta_d$, so the tube depends on $\bar e + \delta_v \tau$ only: halving the frequency (δ_v 0.079) or halving δ_d at 0.05 Hz gives the same tube. The amplitude was kept at 0.5 m/s² and the frequency halved: 8.67 left (worst 7.63), like the 4 g study's 7.45.

### 3. Starts: are the Phase-II episodes started inside the set the filter keeps invariant?

The environments keep only starts $(x_0, \hat d_0)$ in the tube-tightened $C_N$ of the backup.

- **Landing:** the 4 g start (−2, 0, 2) m is 0.30 m inside the cone wall. With the x500 tube (0.37 m funnel margin at T), only 9–19 % of the perturbed states along a slowed reference from there were UE-recoverable, and the start itself was not recoverable even at δ_d 0.25.
  - A hover-grid scan shows the UE $C_N$ covers the cone interior except a band of about 0.3–0.5 m along the wall.
  - The start was therefore moved to (−1.6, 0, 2) m: 0.70 m inside the wall.
  - Now **94.7 %** of the perturbed starts (±0.2 m, ±0.3 m/s, 5° tilt, 5° yaw) are in the tightened $C_N$ at 0.05 Hz, **99.8 %** at 0.025 Hz (Phase-II bank; both references start at the same state).
- **Cone:** **100 %** of the L3-1-s starts are in the tightened $C_N$ with c_B 20 (LQR backup). Along the reference about 95 % of the perturbed states are UE-recoverable.

### 4. References: slower and within the x500's region

| reference | duration | peak speed | tilt | thrust | body rate |
|---|---|---|---|---|---|
| landing corner-cut (4 g study) | 2.0 s | 2.72 m/s | 23.8° | 9.2–12.9 | 1.29 rad/s |
| **landing corner-cut x500** | 4.0 s | 1.53 m/s | 8.9° | 8.9–10.8 | 0.36 rad/s |
| **landing deep dip x500** | 4.5 s | 1.49 m/s | 10.6° | 9.1–10.6 | 0.51 rad/s |
| cone L3-1-f (4 g study) | 5.6 s | 2.19 m/s | 14.3° | 8.4–11.3 | 0.59 rad/s |
| **cone L3-1-s (x500)** | 7.6 s | 1.31 m/s | 5.3° | 8.4–11.3 | 0.13 rad/s |

Why slower:
- The x500's design region (and so its base-set and Phase-I certification) stops at 2 m/s.
- Its stopping distance at 2 m/s is 25 cm toward the wall and 39 cm toward the floor (4 g: 6–7 cm).

Both references use at most 75 % of the thrust and under 10 % of the rate limits.

The landing reference:
- Generated by `ps2rl/envs/assets/generate_quadrotor_landing_reference.py --duration 4.0 --x0 -1.6 --z0 2 --vx0 0 --vz0 0 --xy_delay 1.2 --offset 0 --a_max_g 1.5235 --omega_max 3.84`.
- `--xy_delay` is new: it holds the horizontal position for the first 1.2 s, so the descent starts straight down before cutting the corner.
- The reference leaves the cone for 1.04 s (at most 0.18 m), as the 4 g corner cut does: the filter is what keeps the vehicle inside.

The deep-dip landing reference (`ps2rl/envs/assets/quadrotor_landing_deepdip_x500_reference.npz`, task `landing_ue_x500_deepdip.json`):
- Same family and start, `--duration 4.5 --xy_delay 2.0`: straight down for 2 s, out of the cone at 1.30 m, then the glide in.
- Up to **0.56 m** outside the cone (at (−1.39, 0, 0.53) m) for 1.64 s; back inside 8 cm above the pad.
- Same speed class as the corner cut (peak 1.49 m/s).
- Other depths, same start (deepest point outside the cone / peak speed / tilt):

| duration | xy_delay | outside | peak speed | tilt |
|---|---|---|---|---|
| 4.0 s | 1.2 s (corner cut) | 0.18 m | 1.53 m/s | 8.9° |
| 4.0 s | 1.6 s | 0.44 m | 1.61 m/s | 11.5° |
| 4.5 s | 2.0 s (deep dip) | 0.56 m | 1.49 m/s | 10.6° |
| 5.0 s | 2.4 s | 0.65 m | 1.41 m/s | 9.9° |
| 4.5 s | 2.4 s | 0.79 m | 1.71 m/s | 15.0° |

  Deeper dips need a longer duration to stay under 2 m/s, the x500's design region (5.5 s with 3.2 s: 0.90 m at 1.54 m/s), and come back into the cone only 1–2 cm above the pad.

### 5. Disturbance bound

The amplitude stays at δ_d 0.5 m/s², so the results compare with the 4 g study.
- For the x500 that is about 10 % of its thrust margin above hover (5.1 m/s²); for the 4 g box it was under 2 %.
- The smoke test already shows the consequence. With an untrained actor, the nominal bCBF (plain backup, no disturbance model) left the cone in 69 % of the episodes; the UE-bCBF in none.
- With the 4 g box both stayed safe at δ_d.
- Landing only: 0.025 Hz instead of 0.05 Hz (δ_v 0.079 instead of 0.157 m/s³), see check 8. A 40 s period is a slow gust: within a 5 s episode the sinusoid advances by at most 45° of phase instead of 90°.

### 6. Yaw

The x500 yaws at 0.203 rad/s at most, i.e. 23° within the 2 s backup horizon.
- The cone design region is therefore capped at |yaw| ≤ 25° (default 30°), as the x500 landing design already is.
- The references do not yaw.

### 7. Changes to the recipe (options; the 4 g defaults are unchanged)
- **Landing UE trainer (`train_phase2_landing_ue_cached.py`)** gained:
  - `--task`;
  - `--ue_config` (a plain backup needs the disturbance model from outside);
  - `--reward_scale` (as in the cone trainer).
- **Landing environment:** `terminate_pos_error` (as in the cone environment). Stage V uses 1.5 m (`VANILLA_LOST`). Without it, the unfiltered x500 tracker fell 100 m through the pad in a 5 s episode, where the floored reward is flat.
- **Evaluation:**
  - `--full_episodes` runs the "no filter" mode past the first violation.
  - The nominal mode uses the run's own plain backup.
  - The figures take the disturbance from the evaluation.

### 8. Can the filter let the vehicle land? (added 10/17)

`scripts/check_landing_ue_task.py --task <file>` (about 2 min on the CPU, no training):
1. UE recoverability of level states over the pad centre, at rest and descending at 0.3 m/s, with frozen estimates $|\hat d| = \delta_d + \bar e$ in six directions.
2. The environment's geometric reference tracker, same starts and disturbance draws, through no filter, the nominal filter and the UE filter.

| landing task | lowest fully recoverable height over the pad (at rest / descending 0.3 m/s) | tracker, no filter | tracker + nominal bCBF | tracker + UE-bCBF |
|---|---|---|---|---|
| corner cut, 0.05 Hz (first version) | 0.20 / 0.20 m | 0.20 m outside the cone | lands 100 % | **lands 0 %**: held 0.2–0.6 m up, bouncing; QP slack in 25–30 % of the steps near the pad |
| corner cut, 0.025 Hz | 0.05 / 0.10 m | 0.20 m outside | lands 100 % | **lands 100 %**, min h +0.010 m, final 2.7 cm from the centre |
| deep dip, 0.025 Hz | 0.05 / 0.10 m | **0.58 m outside** | lands 100 % | **lands 100 %**, min h +0.009 m, final 4.3 cm from the centre |

(64 episodes each; "lands" = below 0.1 m inside r0 at |v| < 0.5 m/s at the end of the episode. The same check at 0.05 Hz with 4 s more episode time still landed 0 %.)

Why 0.05 Hz fails: from the last 0.2 m above the pad, the backup has to climb back to the hover point (1.25 m) and settle inside the tightened base set within 2 s. With the x500's slow LQR and only 2.63 of 15.1 left at the end of the tube, it cannot, so those states are outside $C_N$ and the filter never lets the vehicle get there. With the 4 g box the LQR contracts faster and 7.45 is left. **Run this check for every new landing task (vehicle, reference or disturbance) before training**; `run_x500_landing_variants.sh` does it and skips a task whose tracker cannot land through the UE filter.

## Not covered yet
- **SITL on the stock x500:** the trained policies need node configs for `gz_x500` (thrust map of the stock airframe, not the powerloop one).
  - The landing-policy runtime filters with the nominal landing filter of the backup, which works with the plain x500 backup.
  - A UE-filter mode for the runtime is the next step.
- **Tarot 650:** the same procedure once its limits are measured. Steps:
  1. a vehicle file in `docker_batch/configs/vehicles/`;
  2. `run_landing_x500_phase1.sh` with `VEHICLE=...`, `CONFIG=...`;
  3. copies of the two task files with the new vehicle and backup;
  4. the checks above (c bounds, tube margin, starts, references, and check 8).
