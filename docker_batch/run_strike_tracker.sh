#!/usr/bin/env bash
# Strike task, Phase II stage 1: the vanilla striker (SAC, no safety filter), one run per (critic reward scale, seed),
# all on the GPU at the same time. README: docker_batch/README_strike.md.
#
#   docker exec -d px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_strike_tracker.sh > ~/ws_shared/PS2-RL/outputs/strike_tracker_v2.out 2>&1"
#   docker exec -it px4sitl bash -c "tail -F ~/ws_shared/PS2-RL/outputs/strike_tracker_v2.out"
#
# Resumable (a run with summary.json is skipped). `touch <OUT>/<run>/STOP` ends a run early (it still saves weights and
# the summary). Never edit this file while it runs.
# Knobs: TASK (docker_batch/configs/strike.json), OUT (outputs/strike_tracker_v2), SCALES ("0.1"), SEEDS ("0 1"),
#        STEPS (empty = task file, 10M), TRAIN_ARGS (extra trainer arguments), REPORT_EVERY (seconds, 900).
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${here}/landing_env.sh" || exit 1
cd "${PS2RL_ROOT}"
TASK="${TASK:-docker_batch/configs/strike.json}"
OUT="${OUT:-outputs/strike_tracker_v2}"
SCALES="${SCALES:-0.1}"
SEEDS="${SEEDS:-0 1}"
STEPS="${STEPS:-}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
REPORT_EVERY="${REPORT_EVERY:-900}"
mkdir -p "${OUT}/logs"
stamp() { date '+%F %T'; }
echo "[strike-tr] $(stamp) task ${TASK} -> ${OUT}; reward scales ${SCALES}; seeds ${SEEDS}"
ps2rl_gpu_check || exit 1
# the scene pool once (cached), so the parallel runs do not both build it
python - "${TASK}" "${OUT}" <<'PY'
import sys
from ps2rl.envs.strike_env import StrikeEnvConfig, load_or_generate_pool
from ps2rl.envs.strike_scene import StrikeConfig, load_strike_task
t = load_strike_task(sys.argv[1]); ec = StrikeEnvConfig.from_dict(t.get("env", {}))
load_or_generate_pool(ec, max(ec.obstacles_max, StrikeConfig.from_dict(t.get("strike", {})).n_obstacle_slots), cache_dir=sys.argv[2])
PY
pids=()
tags=()
for rs in ${SCALES}; do
    for seed in ${SEEDS}; do
        tag="striker_rs${rs//./p}_s${seed}"
        tags+=("${tag}")
        if [ -f "${OUT}/${tag}/summary.json" ]; then echo "[strike-tr] ${tag}: done before, skipped"; continue; fi
        steps_arg=()
        [ -n "${STEPS}" ] && steps_arg=(--total_steps "${STEPS}")
        echo "[strike-tr] $(stamp) ${tag}: training (log ${OUT}/logs/${tag}.log)"
        # shellcheck disable=SC2086
        python scripts/train_phase2_strike.py --task "${TASK}" --output_root "${OUT}" --run_tag "${tag}" --seed "${seed}" \
            --reward_scale "${rs}" "${steps_arg[@]}" ${PS2RL_GPU_FLAG} ${TRAIN_ARGS} > "${OUT}/logs/${tag}.log" 2>&1 &
        pids+=($!)
        sleep 30
    done
done
while true; do
    alive=0
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
    for tag in "${tags[@]}"; do
        echo "[strike-tr] $(date '+%T') ${tag}: $(grep -E '^\s*\[eval' "${OUT}/logs/${tag}.log" 2>/dev/null | tail -1 | cut -c1-200)"
    done
    [ "${alive}" = 0 ] && break
    sleep "${REPORT_EVERY}"
done
for tag in "${tags[@]}"; do grep -hE "^\[done\]" "${OUT}/logs/${tag}.log" | sed "s/^/[strike-tr]   /"; done
echo "[strike-tr] $(stamp) all done"
