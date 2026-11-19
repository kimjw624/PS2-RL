# Slow-track tracking policies: which to use on the vehicle, and how

This repository has two trained tracking policies and one PD controller. All three fly a slow reference with PX4 offboard **body-rate + collective-thrust** commands at 50 Hz, through the ROS 2 bridge `ps2rl_px4_bridge` (repository `dual_stage_rl`, folder `SITL_src/ps2rl_px4_bridge`).

This page answers four questions:
- which files to use;
- what the policy expects as input and output;
- what runs on the companion computer;
- what to change before flying a real vehicle.

The training side is in `docker_batch/README_slow_track.md`. The bridge is described in `dual_stage_rl/SITL_src/ps2rl_px4_bridge/README_slow_track.md`.

## 1. Which controller to use

| order | controller | what to load | reference it flies | status |
|---|---|---|---|---|
| 1 | **PD** (`controller:=pd`) | nothing | the one passed as `reference_path` | SITL: line 1.7 cm mean, 3.5 cm max; loop 1–2 cm horizontal, ±2 cm height |
| 2 | **`line_v4_s0`**, straight line | `checkpoints/slow_track/line_v4_s0/` | `ps2rl/envs/assets/slow_line/quadrotor_slow_line_reference.npz` | SITL: 2.2 cm mean, 3.5 cm max (section 8) |
| 3 | **`v4_floor_s0`**, slow loop | `checkpoints/slow_track/v4_floor_s0/` | `ps2rl/envs/assets/slow_loop/quadrotor_slow_loop_reference.npz` | SITL: 2.2 cm mean, 4.7 cm max (section 8) |

- **Fly the PD first** on every new vehicle or setup. It needs no weights and checks the whole chain:
  - frames;
  - the thrust map;
  - the rate loop;
  - the odometry.
- **Then the straight line.** It is the first policy flight on a vehicle: 2 m at up to 0.3 m/s.
- **Then the loop**, only after the straight line has flown cleanly.

**Do not use** anything else under `outputs/slow_track/`. In particular:
- `slow_loop_s0` (first recipe, 12–38 cm);
- the `v2_*`, `v3_*` and `v4_floor_lo` runs;
- any `final_weights.pkl`. Training drifts after the best checkpoint; the loop policy's final weights were 2–13 cm off.

Only the folders under `checkpoints/slow_track/` are published and checked.

## 2. What a policy folder contains

`checkpoints/slow_track/<name>/` is written by `scripts/export_slow_track_policy.py`:

| file | what | needed to fly |
|---|---|---|
| `configs.json` | the training configuration. The runtime rebuilds from its `env` block: the command box, the observation scaling, the observer gain and **the reference path** | yes |
| `best_weights.pkl` | the actor only: `{"actor_params": {"layers": [{"w", "b"}, ...]}}`, numpy arrays (34 → 256 → 256 → 8, ReLU) | yes |
| `policy_card.json` | provenance: source run, sha256 of the weights and of the reference, observation and action layout, evaluation numbers | no (check it) |
| `summary.json`, `eval/` | training summary; robust evaluation (`eval.json`, `top_view.png`, `errors.png`, `commands_nominal.png`) | no |

The reference `.npz` lives in the repository (`ps2rl/envs/assets/...`). Its sha256 must match `policy_card.json`. If the reference is regenerated with other numbers, the policy no longer fits it.

## 3. Interface contract (must match)

**Timing.** One step every $\Delta t = 0.02$ s (50 Hz). The reference is sampled at the same step. The bridge steps the policy on PX4's clock, not on wall time.

**Frames.**
- **Task frame:** ENU axes (x east, y north, z up), with its origin at the **hand-over point**, where PX4 hovers before giving control to the policy. It has no absolute position.
- **Body frame:** FLU (x forward, y left, z up).
- The bridge converts PX4's NED / FRD odometry and commands (`ps2rl_px4_bridge/frame_transforms.py`).
- Before the hand-over, PX4 turns the vehicle to yaw 0 in ENU, so the reference's +x is **local east**:
  - with GPS this is geographic east;
  - with motion capture it is whichever mocap axis is fed to PX4 as local east.

  Check which way the vehicle faces at the hold before the hand-over.

**Observation** (34 values, `ps2rl/envs/slow_track_common.py: observation_np`). Here $k$ is the reference index, $p, v, q$ are the measured state in the task frame, and $u_{prev}$ is the previous command:

| entries | name | value |
|---|---|---|
| 0–2 | `e_p` | $(p - p_{ref,k}) / 0.2\ \text{m}$ |
| 3–5 | `e_v` | $(v - v_{ref,k}) / 0.5\ \text{m/s}$ |
| 6–9 | `q` | measured attitude, body→task quaternion $[w, x, y, z]$, with $w \ge 0$ |
| 10–13 | `q_ref` | reference attitude, same convention |
| 14–16 | `v_ref` | reference velocity [m/s] |
| 17–19 | `omega_ref` | reference body rates [rad/s] |
| 20 | `a_ref_rel` | $a_{ref,k}/g - 1$ |
| 21–23 | `preview_1` | $p_{ref,k+25} - p$ (0.5 s ahead) [m] |
| 24–26 | `preview_2` | $p_{ref,k+50} - p$ (1 s ahead) [m] |
| 27–29 | `d_hat` | disturbance estimate [m/s²], see below |
| 30–33 | `u_prev` | previous command, scaled to $[-1, 1]$ by the command box |

**Action** (4 values). The actor's mean is mapped onto the box,

$$u = u_{mid} + u_{half} \odot \tanh(\mu(o)),$$

and then clipped to it:

| index | quantity | box |
|---|---|---|
| 0 | collective thrust per unit mass $c$, along body +z [m/s²] | 7.85 … 11.77 (0.8–1.2 g) |
| 1, 2 | roll and pitch rate $p, q$ (FLU) [rad/s] | ±1.0 |
| 3 | yaw rate $r$ (FLU) [rad/s] | ±0.3 |

**Disturbance observer** (as in training, `observer: commanded`, $\lambda = 5$). Here $R$ is the measured attitude and $c$ the commanded thrust:

$$\dot\xi = c\,R\,e_3 - g\,e_3 + \hat d, \qquad \hat d = \lambda\,(v - \xi).$$

- **Before the hand-over** it runs on the measured thrust (`observer_warmup`), so $\hat d$ has converged when the policy starts.
- **What it absorbs:** hover-thrust error, battery sag and wind, up to what training saw (section 9).

**Thrust to PX4.** The bridge turns $c$ into PX4's normalised `thrust_body` with a thrust map (`thrust_model.py`):
- **quadratic:** $c = k_2 T^2 + k_1 T + k_0$, inverted at run time;
- **linear:** $c = g\,T / T_{hover}$.

The map is per vehicle (section 6).

## 4. What runs on the companion computer

The policy is a small numpy MLP. No JAX, no GPU and no training code is needed on the vehicle.

| part | files |
|---|---|
| runtime (state → command, observer, reference) | `ps2rl/deploy/slow_track_runtime.py`, `ps2rl/envs/slow_track_common.py` |
| flight logic (take-off, hand-over, track, hover, land, aborts, logs) | `ps2rl/deploy/slow_track_flight.py` (extends `landing_flight.py`) |
| policy + reference | `checkpoints/slow_track/<name>/`, `ps2rl/envs/assets/slow_line/` or `slow_loop/` |
| ROS 2 node | `dual_stage_rl/SITL_src/ps2rl_px4_bridge/`: `slow_track_node.py` (extends `landing_node.py`), `px4_stamp_guard.py`, `frame_transforms.py`, `thrust_model.py`, `__init__.py` |
| parameters, launch | `config/slow_track_x500.yaml` (copy it for the vehicle), `launch/ps2rl_slow_track.launch.py` |
| dependencies | ROS 2 Jazzy, `px4_msgs` matching the PX4 firmware (developed on v1.16), Micro XRCE-DDS Agent, `python3` + `numpy` |

**Two things that cost us a day in SITL:**
- **Single-threaded BLAS.**
  - **The problem:** with older numpy (1.26), OpenBLAS spreads the policy's 256 × 256 products over every core and keeps those threads spinning between steps. On a 20-thread laptop that was 1899 % CPU, which starved PX4 and Gazebo.
  - **The fix:** `ps2rl_px4_bridge/__init__.py` sets `OPENBLAS_NUM_THREADS=1` (and the OpenMP / MKL / BLIS equivalents) before numpy loads. Keep it.
  - **Checking it:** `scripts/check_blas_threads.py` measures this on any machine. At the end of the controller phase the bridge also logs its own CPU use: "bridge process CPU during the controller phase: X % of one core". This was 78 % of a laptop core, mostly ROS 2 message handling in Python. Make sure the companion computer has that headroom.
- **Clock jumps.** `px4_stamp_guard.py` keeps the bridge on one continuous clock when the uXRCE-DDS time sync resets. Without it, a reset looked like a PX4 restart and aborted the flight.

## 5. Flight sequence, aborts, footprint

```
BOOT -> ARM -> TAKEOFF -> GOTO_START (hold settle_time) -> POLICY (reference + 2 s) -> RECOVER (hover 3 s) -> LAND
                                                              \-> abort: hover where it is (3 s) -> LAND
```

- **Before the policy:** PX4 position control takes off and holds the hand-over point (`handover_height` above the take-off point, yaw 0).
- **During the policy:** it is in control for the reference plus 2 s.
- **After the policy:** PX4 hovers and lands.

Aborts (all in the YAML):
- tracking error > `max_track_error`;
- tilt > `tilt_abort_deg`;
- speed > `max_speed`;
- height < `min_height`;
- odometry older than `odom_timeout`;
- a non-finite state or command;
- an exception;
- loss of arming or offboard mode.

| reference | duration | footprint from the hand-over point | peak speed / acceleration / tilt |
|---|---|---|---|
| straight line | 16.7 s (+ 2 s) | x 0 → 2 m, y 0, constant height | 0.3 m/s, 0.14 m/s², 0.8° |
| slow loop | 46.1 s (+ 2 s) | x 0 → 5 m, y 0 → 3 m, constant height | 0.4 m/s, 0.19 m/s², 1.1° |

Leave at least 1 m of clear space around the footprint. That covers the abort radius (0.8 m) plus the hover after an abort.

## 6. What to change for a real vehicle

Make a copy of `config/slow_track_x500.yaml` (for example `slow_track_<vehicle>.yaml`) and pass it with `config:=...`.

1. **Thrust map.** The x500 numbers are for Gazebo only.
   - The slow references need $c \approx g$ throughout (thrust 9.69–9.84 m/s² in SITL). So the simplest correct start is `thrust_model: "linear"` with `hover_thrust` = the vehicle's hover throttle: PX4's hover-thrust estimate, `MPC_THR_HOVER` once converged.
   - A quadratic fit (`thrust_calib_node.py`) is better, but is not needed for these references.
   - The policy was trained for a thrust gain of 0.9–1.1. Beyond that the observer has to absorb the rest.
2. **Rate loop.** After every flight the bridge prints the fitted roll / pitch rate response (delay + first-order lag). The policy was trained for 0–40 ms of command delay and a 20–80 ms rate lag. Fly the PD and check that the vehicle is inside this range before flying a policy. Tune PX4's rate controller if it is not.
3. **Heights and limits.**
   - `handover_height`: 2.0 in SITL; for example 1.5 indoors.
   - `min_height`: keep it at least 0.5 m below the hand-over height.
   - `max_track_error`: 0.8 m; lower it (for example to 0.5) for the first flights.
4. **State estimate.**
   - **Noise trained for:** 1 cm on position, 2 cm/s on velocity, 0.3° on attitude.
   - **Check the vehicle's:** look at its odometry noise in the PD flight's `flight.csv`. Commands jitter if it is much noisier, and a command filter would then be the next addition.
   - **Estimator check:** the summary's "estimator check" line flags a vertical velocity that disagrees with the height change.
5. **Hand-over gates.** PX4 must hold the start within `start_pos_tol` (0.15 m) and `start_speed_tol` (0.15 m/s) for `settle_time`. Otherwise the flight lands after `start_timeout` without handing over.
   - In the offline rehearsal, a velocity estimate with 5 cm/s noise never met the speed gate.
   - If the vehicle's estimate is that noisy, raise `start_speed_tol` (for example to 0.25), or improve the estimate first.
6. **Direction.** The reference goes along local +x (east). Make sure that direction is clear; the code has no option to rotate it.
7. **Logs.** Set `log_root` to a path on the companion computer. Every flight writes `flight.csv`, `summary.json` and `bridge_params.json`.

## 7. Bring-up order

| step | command (`ros2 launch ps2rl_px4_bridge ps2rl_slow_track.launch.py ...`) | pass if |
|---|---|---|
| SITL, PD, line | `reference_path:=ps2rl/envs/assets/slow_line/quadrotor_slow_line_reference.npz label:=pd_line` | complete, error a few cm, no events |
| SITL, policy, line | `controller:=policy run:=checkpoints/slow_track/line_v4_s0 label:=policy_line` | the same; commands well inside the box |
| vehicle, PD, line | as the SITL PD line, with the vehicle's `config:=` | complete; rate response inside the trained range; thrust map right (`d_hat` small at the hand-over) |
| vehicle, policy, line | as the SITL policy line, with the vehicle's `config:=` | complete; no abort; error comparable to the PD |
| vehicle, policy, loop | `controller:=policy run:=checkpoints/slow_track/v4_floor_s0 label:=policy_loop` | only after the line is clean |

After each flight, check these:
- **the flight:** `python3 scripts/plot_slow_track_flight.py <flight folder>`;
- **SITL only:** `python3 scripts/check_sim_timing.py <flight folder>`, which flags simulator timing artifacts.

## 8. Results so far (Gazebo SITL, stock x500)

| flight | controller | outcome | position error | height | commands |
|---|---|---|---|---|---|
| 20261118_064818 | PD, loop | complete | 1–2 cm horizontal | ±2 cm | ≤ 0.05 rad/s |
| 20261119_013200 | `v4_floor_s0`, loop | complete, 2384 of 2405 steps | mean 2.2 cm, p95 3.8, max 4.7, end 1.5 | mean −1.0, min −2.8 cm | thrust 9.69–9.84 m/s², ≤ 0.04 rad/s |
| 20261119_022604 | PD, line | complete, 928 of 934 steps | mean 1.7 cm, p95 2.9, max 3.5, end 1.3 | mean +0.2, min −1.2 cm | thrust 9.76–9.86 m/s², ≤ 0.03 rad/s |
| 20261119_022715 | `line_v4_s0`, line | complete, 922 of 934 steps | mean 2.2 cm, p95 3.2, max 3.5, end 1.4 | mean −0.2, min −1.9 cm | thrust 9.76–9.85 m/s², ≤ 0.05 rad/s |

What these four flights shared:
- `check_sim_timing.py` found them clean: no late IMU samples and no vertical-velocity jumps.
- The bridge used 75–78 % of one core.
- PX4 followed the rate commands to within about 0.01 rad/s (RMS).

**Robust evaluation in the training environment** (8 scenarios: delays, lags, thrust gain ±10 %, wind, offsets):
- `v4_floor_s0`: 1.3–3.0 cm mean per scenario, nothing lost.
- `line_v4_s0` (GPU, 1M steps in 3 min; best checkpoint at 0.4M steps, score 0.025; PD on the same reference 0.019), 32 episodes per scenario, nothing lost:

  | scenario | ideal | nominal | slow actuators | thrust −10 % | thrust +10 % | windy | train starts | 0.5 m offset start |
  |---|---|---|---|---|---|---|---|---|
  | mean error [cm] | 0.8 | 1.1 | 2.0 | 2.7 | 2.0 | 3.1 | 4.3 | 4.8 |

  The training starts and the offset start begin up to 0.5 m off the path, which is why their mean error is higher. The policy drifts after 0.6M steps (the final weights: 3.3–6.5 cm), so only the best checkpoint is published.
- **Offline flight rehearsal of `line_v4_s0`** (`rehearse_slow_track_flight.py --sweep`, with no ROS or PX4):
  - 9 of 10 plant variants completed, with 0.8–2.7 cm mean and 7.9 cm worst error. The variants were delay up to 60 ms, slow rates, thrust ±10 %, pushes, and all of them combined.
  - The 10th (noisy estimate) never handed over; see the hand-over gates in section 6.

**Not flown on hardware yet.**

## 9. What the policies were trained for, and their limits

**Model.**
- A rigid quadrotor with mass-normalised thrust and body-rate commands.
- First-order rate and motor lags, and a command delay.
- No aerodynamic drag model and no rotor dynamics beyond the lag.

**Randomised per episode:**
- delay 0–40 ms;
- rate lag 20–80 ms;
- motor lag 20–60 ms;
- thrust gain 0.9–1.1;
- rate gain 0.95–1.05;
- a sinusoidal push of up to 0.5 m/s² plus constant wind of up to 0.3 m/s²;
- the measurement noise of section 6.

**Starts.** Anywhere along the reference. 30 % at the hand-over, with up to ±0.5 m position offset, ±0.2 m/s, ±5° tilt and ±10° yaw.

**What the policy is.** It was trained by imitating the PD (behaviour cloning, decaying to a floor weight of 0.2) plus SAC.
- It tracks about as well as the PD, not better.
- SAC alone did not keep cm-level precision on these references.

**Not covered:**
- obstacles;
- references other than the one in `configs.json`;
- yaw other than 0;
- aggressive manoeuvres;
- wind or thrust errors beyond the ranges above.

## 10. Retraining and publishing

Train in the `px4sitl` container, on the GPU:

```bash
cd ~/ws_shared/PS2-RL
nohup bash docker_batch/run_slow_line.sh > outputs/slow_line.out 2>&1 &                        # straight line
VARIANTS=v4_floor bash docker_batch/run_slow_track_sweep.sh > outputs/slow_track_sweep.out 2>&1  # loop
```

Then publish the checked run:

```bash
python3 scripts/export_slow_track_policy.py --run outputs/slow_track/line_v4_s0 --note "straight line, v4_floor recipe"
```

The export writes `checkpoints/slow_track/<run>/`. It then checks that the copy is bit-identical and that the runtime loads it.

## 11. Files

| file | role |
|---|---|
| `ps2rl/envs/assets/generate_quadrotor_slow_loop_reference.py` | reference generator (loop; `--turns 0` for the straight line) |
| `ps2rl/envs/assets/slow_line/`, `slow_loop/` | the two references (`.npz`, figure, config) |
| `ps2rl/envs/slow_track_common.py`, `slow_track_env.py` | configuration, numpy observation / PD / command, JAX training env |
| `ps2rl/phase2_ps2/slow_track_trainer.py`, `scripts/train_slow_track.py` | SAC with PD cloning, robust evaluation |
| `scripts/eval_slow_track.py`, `rehearse_slow_track_flight.py`, `plot_slow_track_flight.py` | offline evaluation, flight rehearsal without ROS, flight figures |
| `scripts/export_slow_track_policy.py` | publish a run into `checkpoints/slow_track/` |
| `scripts/check_sim_timing.py`, `check_blas_threads.py` | SITL timing check, BLAS threading check |
| `ps2rl/deploy/slow_track_runtime.py`, `slow_track_flight.py` | what the ROS node runs |
| `docker_batch/run_slow_track.sh`, `run_slow_track_sweep.sh`, `run_slow_line.sh` | GPU batches |
| `checkpoints/slow_track/` | the published policies |
