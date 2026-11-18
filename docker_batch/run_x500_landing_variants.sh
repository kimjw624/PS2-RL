#!/usr/bin/env bash
# Landing inside the cone with the stock-x500 limits: the full two-stage procedure for several landing task
# files, pipelined over the two devices. Stage V of every task runs on the GPU one after the other; as soon as a
# task's stage V is done, its stages P E F join a CPU queue (one CPU job at a time), so the GPU never waits for
# the CPU. Each task goes through docker_batch/run_landing_ue_twostage.sh with its own output folder.
#
#   docker exec -d px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_x500_landing_variants.sh \
#       > ~/ws_shared/PS2-RL/outputs/x500_landing_variants.out 2>&1"
#   docker exec -it px4sitl bash -c "tail -F ~/ws_shared/PS2-RL/outputs/x500_landing_variants.out"
#
# Before training, each task is checked with scripts/check_landing_ue_task.py (can the filter let the vehicle
# reach the pad; does the reference tracker land through each filter), log <out>/logs/task_check.log. A task
# whose tracker does not land through the UE filter is skipped (CHECK=warn trains it anyway, CHECK=0 skips the check).
#
# Resumable (finished runs and evaluations are skipped). Never edit this file while it runs.
# Knobs (environment variables):
#   TASKS     space-separated name=task_file   ("cornercut=docker_batch/configs/landing_ue_x500.json
#                                                deepdip=docker_batch/configs/landing_ue_x500_deepdip.json")
#   OUT_ROOT  output root; task <name> goes to OUT_ROOT/<name>                     (outputs/landing_phase2_ue_x500_v2)
#   CHECK     1 = check and skip a failing task, warn = check only, 0 = no check   (1)
#   and the knobs of run_landing_ue_twostage.sh (SEED, VANILLA_STEPS, VANILLA_SCALES, VANILLA_LOST, PS2_STEPS, EPISODES, ...)
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${here}/landing_env.sh" || exit 1
cd "${PS2RL_ROOT}"

TASKS="${TASKS:-cornercut=docker_batch/configs/landing_ue_x500.json deepdip=docker_batch/configs/landing_ue_x500_deepdip.json}"
OUT_ROOT="${OUT_ROOT:-outputs/landing_phase2_ue_x500_v2}"
CHECK="${CHECK:-1}"
stamp() { date '+%F %T'; }
echo "[x500-land] $(stamp) tasks: ${TASKS} -> ${OUT_ROOT}/<name>"

cpu_prev=""
for spec in ${TASKS}; do
    name="${spec%%=*}"; task="${spec#*=}"
    out="${OUT_ROOT}/${name}"
    mkdir -p "${out}/logs"
    if [ "${CHECK}" != "0" ] && [ ! -f "${out}/logs/task_check.log" ]; then
        echo "[x500-land] $(stamp) ${name}: task check (CPU, about 2 min) -> ${out}/logs/task_check.log"
        env PS2RL_ALLOW_CPU=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python scripts/check_landing_ue_task.py --task "${task}" \
            > "${out}/logs/task_check.log" 2>&1 || echo "[x500-land] ${name}: task check failed to run"
    fi
    if [ -f "${out}/logs/task_check.log" ]; then
        grep -E "^\[ue\]|descending at|^\s*\[(none|nominal|ue) *\]" "${out}/logs/task_check.log" | sed "s/^/[x500-land]   ${name}: /" | cut -c1-230
        if [ "${CHECK}" = "1" ] && ! grep -qE "^\s*\[ue +\].* landed (0\.[5-9]|1\.)" "${out}/logs/task_check.log"; then
            echo "[x500-land] ${name}: the reference tracker does not land through the UE filter; skipped (CHECK=warn trains it anyway)"
            continue
        fi
    fi
    echo "[x500-land] $(stamp) ${name}: stage V on the GPU (log ${out}/logs/batch_V.out)"
    env OUT="${out}" TASK="${task}" STAGES=V bash "${here}/run_landing_ue_twostage.sh" > "${out}/logs/batch_V.out" 2>&1 \
        || { echo "[x500-land] ${name}: stage V failed (see ${out}/logs/batch_V.out)"; continue; }
    grep -E "^\[land-ue\] V: " "${out}/logs/batch_V.out" | tail -3 | sed "s/^/[x500-land]   /"
    echo "[x500-land] $(stamp) ${name}: stages P E F queued on the CPU (log ${out}/logs/batch_PEF.out)"
    (
        if [ -n "${cpu_prev}" ]; then
            while kill -0 "${cpu_prev}" 2>/dev/null; do sleep 30; done
        fi
        echo "[x500-land] $(stamp) ${name}: stages P E F started on the CPU"
        env OUT="${out}" TASK="${task}" PS2RL_ALLOW_CPU=1 JAX_PLATFORMS=cpu STAGES="P E F" bash "${here}/run_landing_ue_twostage.sh" \
            > "${out}/logs/batch_PEF.out" 2>&1 || echo "[x500-land] ${name}: stages P E F failed (see ${out}/logs/batch_PEF.out)"
        grep -hE "^\[(ue|nominal|none) *\]" "${out}/logs/batch_PEF.out" | cut -c1-200 | sed "s/^/[x500-land]   ${name}: /"
        echo "[x500-land] $(stamp) ${name}: done; figures ${out}/landing_ue_*.png"
    ) &
    cpu_prev=$!
done
wait
echo "[x500-land] $(stamp) all done"
