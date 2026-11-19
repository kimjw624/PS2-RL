#!/usr/bin/env bash
# Recipe sweep for the standalone slow-track policy, one run at a time (seed 0), after the first recipe
# chattered and stalled at 0.12-0.38 m (4.9M steps, 2026-11-18).
#
#   v2_box        smaller command box: thrust 0.8-1.2 g, roll/pitch rate +-1.0, yaw rate +-0.3 rad/s
#                 (the reference needs ~1 g and 0.015 rad/s; the PD used <= 0.6 rad/s even with noisy estimates).
#                 The same tanh output resolution then covers a 2-5x smaller range.
#   v2_caps       v2_box + CAPS smoothness on the actor (temporal 0.5, spatial 0.2)
#   v2_caps_hi    v2_box + stronger CAPS (temporal 2.0, spatial 0.5)
#   -> v2_box best 0.122 (8-18 cm, unstable); both CAPS variants worse (0.42). The deterministic policy does
#      not chatter (du 0.07); it is imprecise.
#   v3_obs        v2_box + tracking errors scaled in the observation (e_p / 0.2 m, e_v / 0.5 m/s)
#   v3_bc         v3_obs + behaviour cloning of the PD as the warm start (weight 1, held for 100k actor updates,
#                 then decaying to 0 over 400k); the policy still flies alone
#                 (CPU check: 1.3-4.7 cm in every scenario after 100k steps, nothing lost)
#   -> v3_bc: best 0.024 at 0.4M steps (cloning still on); after the cloning weight reached 0, SAC alone lost
#      every episode by 3M steps.
#   v4_floor      v3_bc, but the cloning weight decays only to 0.2 and stays there (an anchor, like the residual
#                 striker's), and the actor learns 3x slower (1e-4)
#   v4_floor_lo   the same with the anchor at 0.05
#
# In the container:
#   cd ~/ws_shared/PS2-RL && nohup bash docker_batch/run_slow_track_sweep.sh > outputs/slow_track_sweep.out 2>&1 &
#   tail -F outputs/slow_track/logs/v2_box_s0.log
#
# Knobs: VARIANTS (subset, in order), STEPS (3000000 each), SEED (0), EPISODES (32). Resumable: finished runs
# are skipped. At the end it prints one table of all evaluated variants (and the first recipe, if present).
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BOX="--env_a_min 7.85 --env_a_max 11.77 --env_omega_xy_max 1.0 --env_omega_z_max 0.3"
declare -A ARGS=(
    [v2_box]="${BOX}"
    [v2_caps]="${BOX} --caps_temporal 0.5 --caps_spatial 0.2"
    [v2_caps_hi]="${BOX} --caps_temporal 2.0 --caps_spatial 0.5"
    [v3_obs]="${BOX} --env_obs_pos_scale 0.2 --env_obs_vel_scale 0.5"
    [v3_bc]="${BOX} --env_obs_pos_scale 0.2 --env_obs_vel_scale 0.5 --bc_weight 1.0 --bc_hold 100000 --bc_decay 400000"
    [v4_floor]="${BOX} --env_obs_pos_scale 0.2 --env_obs_vel_scale 0.5 --bc_weight 1.0 --bc_hold 100000 --bc_decay 400000 --bc_min 0.2 --actor_lr 1e-4"
    [v4_floor_lo]="${BOX} --env_obs_pos_scale 0.2 --env_obs_vel_scale 0.5 --bc_weight 1.0 --bc_hold 100000 --bc_decay 400000 --bc_min 0.05 --actor_lr 1e-4"
)
VARIANTS="${VARIANTS:-v3_bc v3_obs}"
STEPS="${STEPS:-3000000}"
SEED="${SEED:-0}"
EPISODES="${EPISODES:-32}"

for v in ${VARIANTS}; do
    [[ -n "${ARGS[$v]+x}" ]] || { echo "[sweep] unknown variant ${v}"; continue; }
    echo "[sweep] $(date '+%T') ${v}: ${ARGS[$v]}"
    TAG="${v}" SEEDS="${SEED}" STEPS="${STEPS}" EPISODES="${EPISODES}" STAGES="R T E" TRAIN_ARGS="${ARGS[$v]}" \
        bash "${here}/run_slow_track.sh"
done

# shellcheck disable=SC1091
source "${here}/landing_env.sh" >/dev/null 2>&1
cd "${PS2RL_ROOT}"
python - outputs/slow_track "${SEED}" slow_loop v2_box ${VARIANTS} <<'EOF'
import json, sys
from pathlib import Path
out, seed, tags = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
print(f"\n{'run':16s} {'score':>7s}  per scenario: mean position error [cm] (* = some episodes lost)")
for t in tags:
    p = out / f"{t}_s{seed}" / "eval_best" / "eval.json"
    if not p.exists():
        print(f"{t + '_s' + seed:16s}   (no evaluation)")
        continue
    ev = json.loads(p.read_text())
    sc = ev["scenarios"]
    cells = "  ".join(f"{k[:10]} {100 * v['pos_err_mean']:5.1f}{'*' if v['lost_rate'] > 0 else ' '}" for k, v in sc.items())
    du = max(v["du_mean"] for v in sc.values())
    print(f"{t + '_s' + seed:16s} {ev['score']:7.4f}  {cells}  | du max {du:.3f}")
EOF
echo "[sweep] $(date '+%F %T') done"
