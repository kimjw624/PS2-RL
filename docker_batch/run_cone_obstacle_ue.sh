#!/usr/bin/env bash
# Outside-cone task under a disturbance (cone-obstacle UE-bCBF), two-stage Phase II on the GPU:
#   V  vanilla tracker   (SAC, no CIL, episodes not cut at a violation; the warm-start source)
#   P  PS2               (SAC through the UE-bCBF CIL, warm-started from V; backup from chosen_backup.json)
#   E  evaluation        (vanilla: none / ue / nominal; PS2: ue / none; same starts and disturbance draws)
#   F  figures           (3-column progress figure like the UE-landing one, a 4-column one with the nominal
#                         BCBF, side-view GIF of all columns and a 3-D GIF of the final policy)
#
#   docker exec -d px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_cone_obstacle_ue.sh \
#       > ~/ws_shared/PS2-RL/outputs/cone_obstacle_ue.out 2>&1"
#   docker exec -it px4sitl bash -c "tail -F ~/ws_shared/PS2-RL/outputs/cone_obstacle_ue.out"
#
# Resumable: a training run with a summary.json is skipped, an evaluation with an eval.json is skipped.
# Knobs (environment variables):
#   OUT            output root                                   (outputs/cone_obstacle_phase2_ue)
#   TASK           task file                                     (docker_batch/configs/cone_obstacle_ue.json)
#   BACKUP_JSON    chosen Phase-I backup (compare_phase1_cone_obstacle.py)
#                                                                (outputs/cone_obstacle_phase1/compare/chosen_backup.json)
#   SEED           seed of both stages                           (0)
#   VANILLA_STEPS  stage V steps                                 (3000000)
#   VANILLA_LOST   stage V: end an episode once the position error exceeds this [m]  (2.0)
#                  (the last reward then carries the discounted floor reward of the remaining steps, so
#                  ending early never pays; without it the unfiltered tracker flies hundreds of metres away
#                  where the floored reward is flat and stops learning)
#   PS2_STEPS      stage P steps                                 (1000000)
#   EPISODES       evaluation episodes                           (128)
#   STAGES         subset of "V P E F"                           (all)
#   VANILLA_ARGS / PS2_ARGS / EVAL_ARGS / PLOT_ARGS   extra arguments
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${here}/landing_env.sh" || exit 1
cd "${PS2RL_ROOT}"

OUT="${OUT:-outputs/cone_obstacle_phase2_ue}"
TASK="${TASK:-docker_batch/configs/cone_obstacle_ue.json}"
BACKUP_JSON="${BACKUP_JSON:-outputs/cone_obstacle_phase1/compare/chosen_backup.json}"
SEED="${SEED:-0}"
VANILLA_STEPS="${VANILLA_STEPS:-3000000}"
VANILLA_LOST="${VANILLA_LOST:-2.0}"
PS2_STEPS="${PS2_STEPS:-1000000}"
EPISODES="${EPISODES:-128}"
STAGES="${STAGES:-V P E F}"
VANILLA_ARGS="${VANILLA_ARGS:-}"
PS2_ARGS="${PS2_ARGS:-}"
EVAL_ARGS="${EVAL_ARGS:-}"
PLOT_ARGS="${PLOT_ARGS:-}"
mkdir -p "${OUT}/logs"
has_stage() { [[ " ${STAGES} " == *" $1 "* ]]; }

ps2rl_gpu_check || { echo "[cone-ue] GPU check failed (set PS2RL_ALLOW_CPU=1 JAX_PLATFORMS=cpu to force CPU)"; exit 1; }
V_TAG="cone_ue_vanilla_s${SEED}"
P_TAG="cone_ue_ps2_s${SEED}"
echo "[cone-ue] $(date '+%F %T') OUT=${OUT} TASK=${TASK} SEED=${SEED} stages ${STAGES}"

if has_stage V; then
    if [ -f "${OUT}/${V_TAG}/summary.json" ]; then
        echo "[cone-ue] V: ${OUT}/${V_TAG} done, skipped"
    else
        echo "[cone-ue] V: vanilla tracker, ${VANILLA_STEPS} steps -> ${OUT}/${V_TAG}"
        # shellcheck disable=SC2086
        python scripts/train_phase2_cone_obstacle_ue.py --task "${TASK}" --output_root "${OUT}" --run_tag "${V_TAG}" \
            --seed "${SEED}" --use_projection false --env_terminate_on_unsafe false --env_terminate_pos_error "${VANILLA_LOST}" \
            --total_steps "${VANILLA_STEPS}" \
            --num_envs 64 --update_every 2 --replay_size 1000000 --eval_every 100352 ${VANILLA_ARGS} \
            2>&1 | tee "${OUT}/logs/${V_TAG}.log"
        [ -f "${OUT}/${V_TAG}/summary.json" ] || { echo "[cone-ue] V failed"; exit 1; }
    fi
fi

if has_stage P; then
    if [ -f "${OUT}/${P_TAG}/summary.json" ]; then
        echo "[cone-ue] P: ${OUT}/${P_TAG} done, skipped"
    else
        [ -f "${BACKUP_JSON}" ] || { echo "[cone-ue] P: ${BACKUP_JSON} not found (Phase-I comparison); set BACKUP_JSON"; exit 1; }
        echo "[cone-ue] P: PS2 through the UE-bCBF CIL, ${PS2_STEPS} steps, backup $(python -c "import json;d=json.load(open('${BACKUP_JSON}'));print(d['name'], d['backup_path'])")"
        # landing-ue stage-2 recipe (outputs/landing_phase2_ue/ue_p2_warm_s0/configs.json)
        # shellcheck disable=SC2086
        JAX_ENABLE_X64=1 python scripts/train_phase2_cone_obstacle_ue.py --task "${TASK}" --output_root "${OUT}" \
            --run_tag "${P_TAG}" --seed "${SEED}" --backup_from "${BACKUP_JSON}" \
            --warm_start_weights "${OUT}/${V_TAG}/best_weights.pkl" --total_steps "${PS2_STEPS}" \
            --start_steps 0 --update_after 16384 --update_every 8 --replay_size 500000 --num_envs 64 \
            --actor_lr 5e-5 --critic_lr 1e-4 --alpha_lr 5e-5 --project_target_actions true --rows_keep 32 \
            --eval_every 100352 ${PS2_ARGS} 2>&1 | tee "${OUT}/logs/${P_TAG}.log"
        [ -f "${OUT}/${P_TAG}/summary.json" ] || { echo "[cone-ue] P failed"; exit 1; }
    fi
fi

if has_stage E; then
    # the vanilla tracker goes through the same UE-CIL (same backup) as the PS2 run
    B_ARGS=""
    if [ -f "${BACKUP_JSON}" ]; then
        B_ARGS="$(python -c "import json;d=json.load(open('${BACKUP_JSON}'));print('--backup', d['backup'], *(['--backup_path', d['backup_path']] if d['backup_path'] else []))")"
    fi
    for spec in "${V_TAG}:none,ue,nominal" "${P_TAG}:ue,none"; do
        run="${spec%%:*}"; modes="${spec#*:}"
        extra=""
        [ "${run}" = "${V_TAG}" ] && extra="${B_ARGS}"
        [ -f "${OUT}/${run}/best_weights.pkl" ] || { echo "[cone-ue] E: ${run} has no weights, skipped"; continue; }
        if [ -f "${OUT}/${run}/eval/eval.json" ]; then
            echo "[cone-ue] E: ${run} evaluated, skipped"
            continue
        fi
        echo "[cone-ue] E: ${run} (${modes}), ${EPISODES} episodes"
        # shellcheck disable=SC2086
        JAX_ENABLE_X64=1 python scripts/eval_phase2_cone_obstacle_ue.py --run "${OUT}/${run}" --modes "${modes}" \
            --episodes "${EPISODES}" ${extra} ${EVAL_ARGS} 2>&1 | tee "${OUT}/logs/eval_${run}.log"
    done
fi

if has_stage F; then
    V_EV="${OUT}/${V_TAG}/eval"
    P_EV="${OUT}/${P_TAG}/eval"
    if [ -f "${V_EV}/eval.json" ] && [ -f "${P_EV}/eval.json" ]; then
        # shellcheck disable=SC2086
        python scripts/plot_cone_obstacle_ue_progress.py \
            --panel "Stage 1 tracker, no CIL=${V_EV}:none" \
            --panel "Stage 1 tracker + UE-CIL (Phase II start)=${V_EV}:ue" \
            --panel "Phase II trained through UE-CIL=${P_EV}:ue" \
            --out "${OUT}/cone_ue_progress.png" --gif --gif3d ${PLOT_ARGS} 2>&1 | tee "${OUT}/logs/plot.log"
        # shellcheck disable=SC2086
        python scripts/plot_cone_obstacle_ue_progress.py \
            --panel "Stage 1 tracker, no CIL=${V_EV}:none" \
            --panel "Stage 1 tracker + nominal BCBF=${V_EV}:nominal" \
            --panel "Stage 1 tracker + UE-CIL=${V_EV}:ue" \
            --panel "Phase II trained through UE-CIL=${P_EV}:ue" \
            --out "${OUT}/cone_ue_progress_with_nominal.png" ${PLOT_ARGS} 2>&1 | tee -a "${OUT}/logs/plot.log"
    else
        echo "[cone-ue] F: evaluations missing, skipped"
    fi
fi
echo "[cone-ue] $(date '+%F %T') done"
