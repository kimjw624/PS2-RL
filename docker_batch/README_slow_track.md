# Slow forward + loop tracking: PD first, then an RL policy (SITL, later hardware)

The goal is to show that an RL-trained tracking policy can be flown in PX4 SITL with body-rate and collective-thrust commands. The PD controller flies the same loop first, as a check of the whole chain. Obstacles and residual RL come later.

1. **PD in SITL.** `controller: pd` flies the reference. It needs no training.
2. **Train the policy** on the GPU (`run_slow_track.sh`). The policy is standalone RL that outputs the full command.
3. **Check the policy** offline (`eval_slow_track.py`, `rehearse_slow_track_flight.py`).
4. **Fly the policy in SITL** (`controller: policy`). The bridge side is described in `dual_stage_rl/SITL_src/ps2rl_px4_bridge/README_slow_track.md`.

## Reference

`ps2rl/envs/assets/generate_quadrotor_slow_loop_reference.py` writes `ps2rl/envs/assets/slow_loop/quadrotor_slow_loop_reference.{npz,png,_config.json}`.

| | default |
|---|---|
| hover before / after | 3 s / 3 s |
| speed | 0.4 m/s, reached in 4 s (smooth, zero jerk at both ends) |
| straight before / after the loop | 2 m / 2 m |
| loop | radius 1.5 m, to the left, one turn; curvature blended in over 1 m at each end |
| total | 46 s, 14.4 m path; x 0–5 m, y 0–3 m |
| needs | tilt ≤ 1.1°, thrust 9.81–9.812 m/s², rates ≤ 0.015 rad/s |

The task frame has its origin at the hand-over point and x pointing east. The vehicle holds yaw 0 at the hand-over, so x is also its heading. z is up.

## The PD controller

`base_action` in `ps2rl/envs/slow_track_env.py`, with a numpy twin `base_action_np` in `slow_track_common.py`, works like this:
- the desired acceleration is the reference acceleration − 3 · position error − 3 · velocity error − `d_hat`;
- that acceleration gives the thrust and a desired body z;
- the body rates are 8 · (z_body × z_desired);
- yaw is held with gain 1.

The disturbance observer is `d_hat = λ (v − ξ)` with λ = 5. ξ is advanced with the commanded thrust and the measured attitude, and the observer is started warm from take-off on.

Mean position error with the training environment's evaluation scenarios (16 episodes each; definitions below):

| ideal | nominal | train | slow actuators | thrust −10 % | thrust +10 % | windy | 0.5 m offset start |
|---|---|---|---|---|---|---|---|
| 0.1 cm | 0.6 cm | 1.6 cm | 1.3 cm | 1.8 cm | 1.3 cm | 2.3 cm | 2.3 cm |

In the offline flight rehearsal (stand-in vehicle with PX4-like modes, 10 plant variations), it took off, handed over, flew the loop and landed in all 10, with a mean error of 0.2–2.1 cm.

## The RL policy (standalone, `policy_mode: direct`)

| part | taken from | why |
|---|---|---|
| model, action `[a_cmd, p, q, r]` | `ps2rl/envs/quadrotor_env.py` | what PX4 takes through `VehicleRatesSetpoint` |
| reward (Charbonnier on position, velocity, tilt, yaw, rates, thrust) and weights | cone-obstacle stage-1 tracker (`cone_obstacle_ue_env.py`) | tracked to 2.2 cm |
| `d_hat` in the observation | the same | absorbs hover-thrust error and wind |
| lost-episode end with the discounted floor reward | the same (`VANILLA_LOST`) | without it the tracker flew hundreds of metres away |
| SAC settings, `reward_scale 0.01`, hover-initialised actor | the same (`run_cone_obstacle_ue.sh`, stage V) | the unscaled critic stalled at about −642 return / 0.16 m |

**Not used, because it failed:** the landing `tracker_v1` preset (critic targets clipped at ±1e4; five of six runs collapsed after 3–5M steps).

**One change to the SAC settings:** the temperature floor is `min_alpha 1e-4` instead of 1e-2. On this slow reference the per-step costs are small, so at 1e-2 the entropy bonus outweighs cm-level tracking:
- **with the floor at 1e-2:** α sat on the floor and the error stayed at 0.4–0.6 m after 0.8M steps (CPU check);
- **with 1e-4:** α fell to about 5e-4.

**Added for the transfer:**
- **Plant, randomised per episode:**
  - command delay of 0–40 ms;
  - rate-loop lag τ 20–80 ms;
  - motor lag τ 20–60 ms;
  - thrust gain 0.9–1.1;
  - rate gain 0.95–1.05;
  - a sinusoidal push of up to 0.5 m/s² plus constant wind of up to 0.3 m/s².
- **Measurement noise** on position (1 cm), velocity (2 cm/s) and attitude (0.3°).
- **Starts anywhere along the reference.** 30 % of episodes start at the hand-over point, with up to ±0.5 m offset, ±0.2 m/s, ±5° tilt and ±10° yaw.
- **A translation-invariant 34-D observation:**
  - tracking errors;
  - the reference;
  - a preview of the path 0.5 s and 1 s ahead;
  - `d_hat`;
  - the previous command.

  It has no absolute position and no time input.
- **A command-smoothness cost** (w 0.05).
- **Action box:** thrust 0.5–1.5 g, roll and pitch rate ±2 rad/s, yaw rate ±0.5 rad/s.
- **Timeouts are not terminal.** Only losing the reference is: more than 1 m error or more than 60° tilt.
- **The best checkpoint is chosen by robustness.** It is the lowest mean error over the 8 scenarios, each flown from the hand-over over the whole reference.

**Not yet shown:** that this converges. In short CPU checks (a smaller network and fewer updates than the GPU defaults), the policy stopped losing the reference after about 0.15M steps. But it was still 0.2–0.3 m off after 0.9M steps, and the curve was noisy. The cone tracker needed 3M steps on the GPU. The batch runs 5M steps per seed; the curves will show whether that is enough.

`policy_mode residual` (PD + a bounded correction) is kept in the code for later; it is off by default.

## Straight-line test policy (`run_slow_line.sh`)

A shorter, gentler reference for the first policy test on the vehicle:

| | |
|---|---|
| path | hover 3 s, 2 m straight along x (the heading at the hand-over), hover 3 s; 16.7 s |
| speed | up to 0.3 m/s, reached in 4 s (smooth, zero jerk at both ends) |
| needs | acceleration ≤ 0.14 m/s², tilt ≤ 0.8°, body rates ≤ 0.011 rad/s, thrust 9.81–9.811 m/s² |
| file | `ps2rl/envs/assets/slow_line/quadrotor_slow_line_reference.{npz,png,_config.json}` |

It is made by the same generator with `--turns 0 --l1 2.0 --l2 0 --speed 0.3`. The policy is trained with the recipe of the loop policy that flew in SITL (`v4_floor`):
- the smaller command box (thrust 0.8–1.2 g, roll/pitch rate ±1 rad/s, yaw rate ±0.3 rad/s);
- tracking errors scaled in the observation (0.2 m, 0.5 m/s);
- PD behaviour cloning, held for 100k actor updates, then decaying to a floor of 0.2;
- actor learning rate 1e-4.

Only the reference and the episode length (15 s) differ. In the training environment the PD scores 0.019 on this reference (0.1–4 cm mean error per scenario, nothing lost).

```bash
cd ~/ws_shared/PS2-RL && nohup bash docker_batch/run_slow_line.sh > outputs/slow_line.out 2>&1 &
tail -F outputs/slow_track/logs/line_v4_s0.log
```

One seed, 1M steps, then the evaluation and `outputs/slow_track/choice_line_v4.json`. The loop policy's best checkpoint came at 0.5M steps, while the cloning weight was still decaying.

**Publishing a policy.** `scripts/export_slow_track_policy.py --run outputs/slow_track/<run>` copies a run into `checkpoints/slow_track/<run>/`, which git tracks:
- `configs.json`;
- the actor-only `best_weights.pkl`;
- `summary.json`;
- `eval/`;
- `policy_card.json`, with the sha256 of the weights and of the reference.

It then checks that the runtime loads the copy and gives identical actions. `SLOW_TRACK_ONBOARD.md` (repository root) says which published policy to use and how.

## Where each command runs

These follow `ws_shared/PS2RL_SITL_final.md`.

| what | where | Python |
|---|---|---|
| unzip / git | host, `~/px4-ros-docker` | – |
| training, `eval_slow_track.py` | container `px4sitl`, `~/ws_shared/PS2-RL` | 3.10 GPU venv `.venv`, through `docker_batch/landing_env.sh` (the batch sources it itself) |
| bridge, `rehearse_slow_track_flight.py`, `plot_slow_track_flight.py` | container, `~/ws_shared/PS2-RL` | system `python3` (3.12: rclpy, numpy), **no venv** (`deactivate 2>/dev/null`) |

`~/ws_shared` in the container is `~/px4-ros-docker/ws_shared` on the host.

## Run (GPU, in the container)

```bash
docker exec -d px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_slow_track.sh \
    > ~/ws_shared/PS2-RL/outputs/slow_track.out 2>&1"
docker exec -it px4sitl bash -c "tail -F ~/ws_shared/PS2-RL/outputs/slow_track.out"
docker exec -it px4sitl bash -c "tail -F ~/ws_shared/PS2-RL/outputs/slow_track/logs/slow_loop_s0.log"
```

The batch runs these stages:

| stage | what it does |
|---|---|
| R | generates the reference if it is missing |
| T | trains seeds 0, 1 and 2 in parallel, 5M steps each |
| E | evaluates each run's best weights, 64 episodes per scenario, with figures |
| S | picks the best seed into `outputs/slow_track/choice.json` |

Stop a run early with `touch outputs/slow_track/slow_loop_s0/STOP`. It still finishes normally.

**What to watch in the run logs:**
- `lost` should fall to 0.
- In the `[eval]` lines, the mean position error per scenario should fall toward a few cm.
- `alpha` should fall well below 0.01.
- If the error has not dropped below about 0.1 m by 2–3M steps, stop and send me the logs.

| variable | default |
|---|---|
| `SEEDS` | `"0 1 2"` |
| `STEPS` | 5000000 |
| `EPISODES` | 64 |
| `REF`, `REF_ARGS` | default reference |
| `STAGES` | `"R T E S"` |
| `TRAIN_ARGS` | extra flags, e.g. `"--env_a_max 13.0"` or `"--env_delay_steps_max 3"` |

## Single runs, checks

Training and evaluation run in the venv, inside the container:

```bash
cd ~/ws_shared/PS2-RL && source docker_batch/landing_env.sh
python scripts/train_slow_track.py --run_tag slow_loop_s0 --seed 0
python scripts/eval_slow_track.py --run outputs/slow_track/slow_loop_s0 --weights best --episodes 64
PS2RL_ALLOW_CPU=1 JAX_PLATFORMS=cpu python scripts/train_slow_track.py --smoke_test --run_tag smoke   # 1 min check
```

**Offline flight rehearsal.** This runs the flight logic and runtime the ROS node uses, with no ROS or PX4. It uses system `python3`, not the venv:

```bash
deactivate 2>/dev/null; cd ~/ws_shared/PS2-RL
python3 scripts/rehearse_slow_track_flight.py --controller pd --sweep
python3 scripts/rehearse_slow_track_flight.py --controller policy --run outputs/slow_track/slow_loop_s0 --sweep
```

**Figure of a flight**, from SITL, hardware or a rehearsal with `--out`. Also system `python3`:

```bash
python3 ~/ws_shared/PS2-RL/scripts/plot_slow_track_flight.py --latest ~/ws_shared/experiments/slow_track
```

`eval_slow_track.py` writes the following to `<run>/eval_best/`:
- `eval.json`, with these numbers per scenario: lost rate, position error (mean, p95, max, after 5 s, final), maximum tilt, command ranges, smoothness and observer error;
- `top_view.png`;
- `errors.png`;
- `commands_nominal.png`.

**Before SITL, check:**
- the lost rate is 0 everywhere;
- the commands stay well inside the box;
- the rate commands do not chatter.

## Evaluation scenarios

Each scenario starts at the hand-over and flies the whole reference plus 2 s.

| scenario | plant | start |
|---|---|---|
| `ideal` | no delay or lag, exact thrust, no push, no noise | on the reference |
| `nominal` | 20 ms delay, 40 ms lags, exact thrust, no push, noise | hover with small errors |
| `train` | drawn as in training | training offsets |
| `slow_actuators` | 40 ms delay, rate lag 80 ms, motor lag 60 ms | hover |
| `thrust_low` / `thrust_high` | thrust gain 0.9 / 1.1 | hover |
| `windy` | 0.5 m/s² sinusoid + 0.3 m/s² wind | hover |
| `offset_start` | drawn | 0.5 m to the side, 0.2 m up or down, 10° yaw |

## Files

| file | what |
|---|---|
| `ps2rl/envs/slow_track_common.py` | config, reference loader, numpy twins of the observation / PD / command (no JAX; used by the flight runtime) |
| `ps2rl/envs/slow_track_env.py` | JAX training environment |
| `ps2rl/phase2_ps2/slow_track_trainer.py` | SAC, evaluation scenarios, selection |
| `ps2rl/deploy/slow_track_runtime.py` | PD or policy (numpy MLP) + observer, one state in, one command out |
| `ps2rl/deploy/slow_track_flight.py` | flight logic: take-off → hand-over → track → hover → land; aborts → hover where it is → land |
| `scripts/train_slow_track.py`, `eval_slow_track.py`, `rehearse_slow_track_flight.py`, `plot_slow_track_flight.py` | |
| `docker_batch/run_slow_track.sh` | GPU batch |
| `docker_batch/run_slow_track_sweep.sh` | recipe sweep (`v4_floor` is the recipe that flew) |
| `docker_batch/run_slow_line.sh` | the straight-line test policy |
| `scripts/export_slow_track_policy.py` | publish a run into `checkpoints/slow_track/` |
| `scripts/check_sim_timing.py`, `scripts/check_blas_threads.py` | SITL timing and BLAS-threading checks |
