#!/usr/bin/env bash
# Straight-line test policy for PX4 SITL and the vehicle: hover 3 s, 2 m straight ahead (x of the task frame)
# at up to 0.3 m/s, hover 3 s. 16.7 s in all; peak acceleration 0.14 m/s^2, tilt 0.8 deg, body rates 0.01 rad/s.
# Same recipe as the slow-loop policy that flew in SITL (v4_floor: small command box, scaled tracking errors in
# the observation, PD behaviour cloning decaying to a floor of 0.2, actor lr 1e-4); only the reference differs,
# and the training episodes are 15 s (the reference is 16.7 s; starts are spread along it).
#
# In the container (one seed, ~1M steps; resumable, `touch outputs/slow_track/line_v4_s0/STOP` ends it early):
#   cd ~/ws_shared/PS2-RL && nohup bash docker_batch/run_slow_line.sh > outputs/slow_line.out 2>&1 &
#   tail -F outputs/slow_track/logs/line_v4_s0.log
#
# Result: outputs/slow_track/line_v4_s0/ (best_weights.pkl, configs.json, summary.json, eval_best/) and
# outputs/slow_track/choice_line_v4.json. Knobs: SEEDS ("0"), STEPS (1000000), EPISODES (32), STAGES ("R T E S").
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BOX="--env_a_min 7.85 --env_a_max 11.77 --env_omega_xy_max 1.0 --env_omega_z_max 0.3"
V4_FLOOR="${BOX} --env_obs_pos_scale 0.2 --env_obs_vel_scale 0.5 --bc_weight 1.0 --bc_hold 100000 --bc_decay 400000 --bc_min 0.2 --actor_lr 1e-4"

TAG="${TAG:-line_v4}" \
SEEDS="${SEEDS:-0}" \
STEPS="${STEPS:-1000000}" \
EPISODES="${EPISODES:-32}" \
STAGES="${STAGES:-R T E S}" \
REF="${REF:-ps2rl/envs/assets/slow_line/quadrotor_slow_line_reference.npz}" \
REF_ARGS="${REF_ARGS:---turns 0 --l1 2.0 --l2 0 --speed 0.3}" \
TRAIN_ARGS="${V4_FLOOR} --env_episode_sec 15 ${TRAIN_ARGS:-}" \
    bash "${here}/run_slow_track.sh"
