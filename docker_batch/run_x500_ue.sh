#!/usr/bin/env bash
# The whole UE-bCBF procedure again with the stock-x500 limits, for both tasks:
#   landing inside the cone (docker_batch/configs/landing_ue_x500.json) and the outside-cone strike
#   (docker_batch/configs/cone_obstacle_ue_x500.json). README: docker_batch/README_x500_ue.md.
#
#   L_V    landing  stage V: vanilla trackers (GPU)                 run_landing_ue_twostage.sh STAGES=V
#   L_PEF  landing  stages P E F: PS2 through the UE-CIL, evaluation, figures (CPU, in the background
#                   while the GPU stages of the cone task run)        run_landing_ue_twostage.sh STAGES="P E F"
#   C_P1   cone     Phase I: safe-arrival backup on the plain sets (the method chosen in the 4 g comparison),
#                   then the comparison with the LQR alone (rule: LQR unless SA is >= 2 points better) (GPU)
#   C_V    cone     stage V: vanilla trackers (GPU)                  run_cone_obstacle_ue.sh STAGES=V
#   C_PEF  cone     stages P E F (CPU, after L_PEF)                  run_cone_obstacle_ue.sh STAGES="P E F"
#
# The landing Phase I is not retrained: the task file uses checkpoints/landing_phase1/x500_rec10_td3_seed1, the
# better of the two x500 Phase-I runs (docker_batch/run_landing_x500_phase1.sh). To retrain it, run that batch and
# point "backup" in the task file at the new run.
#
#   docker exec -d px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_x500_ue.sh > ~/ws_shared/PS2-RL/outputs/x500_ue.out 2>&1"
#   docker exec -it px4sitl bash -c "tail -F ~/ws_shared/PS2-RL/outputs/x500_ue.out"
#
# Resumable (every sub-stage skips finished runs). Never edit this file while it runs (bash reads it as it goes).
# Knobs (environment variables):
#   STAGES       subset of "L_V L_PEF C_P1 C_V C_PEF"            (all)
#   LAND_TASK    landing task file                               (docker_batch/configs/landing_ue_x500.json)
#   LAND_OUT     landing Phase-II output root                    (outputs/landing_phase2_ue_x500)
#   CONE_TASK    cone task file                                  (docker_batch/configs/cone_obstacle_ue_x500.json)
#   CONE_P1_OUT  cone Phase-I output root                        (outputs/cone_obstacle_phase1_x500)
#   CONE_OUT     cone Phase-II output root                       (outputs/cone_obstacle_phase2_ue_x500)
#   SEED         seed of every stage                             (0)
#   CP1_STEPS    cone Phase-I steps                              (3000000)
#   CP1_ARGS     extra arguments of the cone Phase-I trainer
#   LOGS         driver logs                                     (outputs/x500_ue_logs)
#   CPU_PARALLEL 1: start C_PEF without waiting for L_PEF (two CPU-heavy jobs at once)   (0)
#   and the knobs of run_landing_ue_twostage.sh / run_cone_obstacle_ue.sh (VANILLA_STEPS, PS2_STEPS, EPISODES, ...);
#   VANILLA_SCALES / VANILLA_LOST are per task: LAND_VANILLA_SCALES ("0.1 1"), LAND_VANILLA_LOST (1.5),
#   CONE_VANILLA_SCALES ("0.01 1"), CONE_VANILLA_LOST (2.0)
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${here}/landing_env.sh" || exit 1
cd "${PS2RL_ROOT}"

STAGES="${STAGES:-L_V L_PEF C_P1 C_V C_PEF}"
LAND_TASK="${LAND_TASK:-docker_batch/configs/landing_ue_x500.json}"
LAND_OUT="${LAND_OUT:-outputs/landing_phase2_ue_x500}"
CONE_TASK="${CONE_TASK:-docker_batch/configs/cone_obstacle_ue_x500.json}"
CONE_P1_OUT="${CONE_P1_OUT:-outputs/cone_obstacle_phase1_x500}"
CONE_OUT="${CONE_OUT:-outputs/cone_obstacle_phase2_ue_x500}"
SEED="${SEED:-0}"
CP1_STEPS="${CP1_STEPS:-3000000}"
CP1_ARGS="${CP1_ARGS:-}"
CPU_PARALLEL="${CPU_PARALLEL:-0}"
LAND_VANILLA_SCALES="${LAND_VANILLA_SCALES:-0.1 1}"
LAND_VANILLA_LOST="${LAND_VANILLA_LOST:-1.5}"
CONE_VANILLA_SCALES="${CONE_VANILLA_SCALES:-0.01 1}"
CONE_VANILLA_LOST="${CONE_VANILLA_LOST:-2.0}"
LOGS="${LOGS:-outputs/x500_ue_logs}"
mkdir -p "${LOGS}" "${CONE_P1_OUT}"
has_stage() { [[ " ${STAGES} " == *" $1 "* ]]; }
stamp() { date '+%F %T'; }
CPU_ENV=(PS2RL_ALLOW_CPU=1 JAX_PLATFORMS=cpu)
CP1_RUN="${CONE_P1_OUT}/sa_plain_seed${SEED}"
CHOSEN="${CONE_P1_OUT}/compare/chosen_backup.json"
echo "[x500-ue] $(stamp) stages ${STAGES}"
echo "[x500-ue] landing: ${LAND_TASK} -> ${LAND_OUT}; cone: ${CONE_TASK} -> Phase I ${CONE_P1_OUT}, Phase II ${CONE_OUT}"

# land / cone VAR=value ... : run the task's batch with these settings (env, so that the arguments are assignments)
land() { env OUT="${LAND_OUT}" TASK="${LAND_TASK}" SEED="${SEED}" VANILLA_SCALES="${LAND_VANILLA_SCALES}" \
         VANILLA_LOST="${LAND_VANILLA_LOST}" "$@" bash "${here}/run_landing_ue_twostage.sh"; }
cone() { env OUT="${CONE_OUT}" TASK="${CONE_TASK}" SEED="${SEED}" BACKUP_JSON="${CHOSEN}" VANILLA_SCALES="${CONE_VANILLA_SCALES}" \
         VANILLA_LOST="${CONE_VANILLA_LOST}" "$@" bash "${here}/run_cone_obstacle_ue.sh"; }

# ------------------------------------------------------------------ landing, stage V (GPU)
if has_stage L_V; then
    echo "[x500-ue] $(stamp) L_V: landing vanilla trackers (log ${LOGS}/land_V.out)"
    land STAGES=V > "${LOGS}/land_V.out" 2>&1 || echo "[x500-ue] L_V failed (see ${LOGS}/land_V.out)"
    grep -E "^\[land-ue\] V: " "${LOGS}/land_V.out" | tail -4
fi

# ------------------------------------------------------------------ landing, stages P E F (CPU, background)
L_PID=""
if has_stage L_PEF; then
    echo "[x500-ue] $(stamp) L_PEF: landing PS2 + evaluation + figures on the CPU, in the background (log ${LOGS}/land_PEF.out)"
    land "${CPU_ENV[@]}" "STAGES=P E F" > "${LOGS}/land_PEF.out" 2>&1 &
    L_PID=$!
fi

# ------------------------------------------------------------------ cone, Phase I (GPU)
if has_stage C_P1; then
    if [ -f "${CP1_RUN}/summary.json" ] || [ -f "${CP1_RUN}/cone_obstacle_backup_policy_actor.pkl" ]; then
        echo "[x500-ue] C_P1: ${CP1_RUN} done, skipped"
    else
        echo "[x500-ue] $(stamp) C_P1: cone Phase I on the plain sets, ${CP1_STEPS} steps (log ${LOGS}/cone_P1.log)"
        # the 4 g comparison's settings (docker_batch/README_cone_obstacle_ue.md)
        # shellcheck disable=SC2086
        python scripts/train_phase1_cone_obstacle_ue.py --task "${CONE_TASK}" --tighten false --output_dir "${CONE_P1_OUT}" \
            --run_name "sa_plain_seed${SEED}" --total_steps "${CP1_STEPS}" --hidden_size 128 --start_steps 0 \
            --actor_start_update 20000 --bc_weight 0.5 --contraction_weight 0.01 --eval_every 250000 --seed "${SEED}" \
            ${CP1_ARGS} > "${LOGS}/cone_P1.log" 2>&1 || echo "[x500-ue] C_P1 training failed (see ${LOGS}/cone_P1.log)"
        grep -E "^\[(base set|ue|done)\]" "${LOGS}/cone_P1.log"
    fi
    if [ -f "${CP1_RUN}/cone_obstacle_backup_policy_actor.pkl" ] && [ ! -f "${CHOSEN}" ]; then
        echo "[x500-ue] $(stamp) C_P1: comparison with the LQR alone (log ${LOGS}/cone_P1_compare.log)"
        python scripts/compare_phase1_cone_obstacle.py --task "${CONE_TASK}" --run "sa_plain=${CP1_RUN}" \
            --out "${CONE_P1_OUT}/compare" > "${LOGS}/cone_P1_compare.log" 2>&1 \
            || echo "[x500-ue] C_P1 comparison failed (see ${LOGS}/cone_P1_compare.log)"
        grep -E "^\[(lqr|sa_|choice)" "${LOGS}/cone_P1_compare.log"
    fi
fi

# ------------------------------------------------------------------ cone, stage V (GPU)
if has_stage C_V; then
    echo "[x500-ue] $(stamp) C_V: cone vanilla trackers (log ${LOGS}/cone_V.out)"
    cone STAGES=V > "${LOGS}/cone_V.out" 2>&1 || echo "[x500-ue] C_V failed (see ${LOGS}/cone_V.out)"
    grep -E "^\[cone-ue\] V: " "${LOGS}/cone_V.out" | tail -4
fi

# ------------------------------------------------------------------ cone, stages P E F (CPU)
if has_stage C_PEF; then
    if [ -n "${L_PID}" ] && [ "${CPU_PARALLEL}" != "1" ]; then
        echo "[x500-ue] $(stamp) C_PEF: waiting for the landing CPU stages (pid ${L_PID})"
        wait "${L_PID}"
        L_PID=""
    fi
    echo "[x500-ue] $(stamp) C_PEF: cone PS2 + evaluation + figures on the CPU (log ${LOGS}/cone_PEF.out)"
    cone "${CPU_ENV[@]}" "STAGES=P E F" > "${LOGS}/cone_PEF.out" 2>&1 || echo "[x500-ue] C_PEF failed (see ${LOGS}/cone_PEF.out)"
fi
[ -n "${L_PID}" ] && wait "${L_PID}"
grep -hE "^\[(ue|nominal|none) *\]" "${LOGS}/land_PEF.out" "${LOGS}/cone_PEF.out" 2>/dev/null | cut -c1-200
echo "[x500-ue] $(stamp) done. Figures: ${LAND_OUT}/landing_ue_*.png, ${CONE_OUT}/cone_ue_*.png"
