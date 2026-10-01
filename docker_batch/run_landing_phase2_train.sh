#!/usr/bin/env bash
# Phase-II landing training on the GPU, for each reference:
#   B  tracker   (SAC, no safety layer; the colleague's landing-tracker settings, landing reward v2)
#   C  PS2       (SAC through the landing CIL, warm-started from the tracker, paper PS2 settings)
#   D  evaluation: tracker alone / tracker + filter / PS2, same starts, figures + GIF
#   E  REPORT.md
#
#   docker exec -d px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_landing_phase2_train.sh \
#       > ~/ws_shared/PS2-RL/outputs/landing_phase2.out 2>&1"
#   docker exec -it px4sitl bash -c "tail -F ~/ws_shared/PS2-RL/outputs/landing_phase2.out"
#
# Resumable: a run with a summary.json is skipped; an evaluation with a summary.md is skipped.
# Knobs (environment variables):
#   OUT            output root                                   (outputs/landing_phase2)
#   REFS           "name:path name:path"                         (skid + nearside references)
#   BACKUP         Phase-I checkpoint                            (checkpoints/landing_phase1/floor_rec10_td3_seed0)
#   FILTER         landing CIL settings                          (alpha=10,alpha_floor=20,control_weights=1/0.1/0.1/1)
#   SEED           training seed                                 (0)
#   TRACKER_STEPS  tracker training steps                        (8000000)
#   PS2_STEPS      PS2 training steps (upper bound)              (1500000)
#   PS2_HOURS      wall-time budget per PS2 run, 0 = none        (4)
#   MAX_PARALLEL   runs at once on the GPU                       (2)
#   STAGES         subset of "B C D E"                           (all)
#   TRACKER_ARGS / PS2_ARGS / EVAL_ARGS   extra arguments for the trainer / evaluator
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${here}/landing_env.sh" || exit 1
cd "${PS2RL_ROOT}"

OUT="${OUT:-outputs/landing_phase2}"
REFS="${REFS:-skid:ps2rl/envs/assets/quadrotor_landing_skid_reference.npz nearside:ps2rl/envs/assets/quadrotor_landing_nearside_reference.npz}"
BACKUP="${BACKUP:-checkpoints/landing_phase1/floor_rec10_td3_seed0}"
FILTER="${FILTER:-alpha=10,alpha_floor=20,control_weights=1/0.1/0.1/1}"
SEED="${SEED:-0}"
TRACKER_STEPS="${TRACKER_STEPS:-8000000}"
PS2_STEPS="${PS2_STEPS:-1500000}"
PS2_HOURS="${PS2_HOURS:-4}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
STAGES="${STAGES:-B C D E}"
mkdir -p "${OUT}/logs"

ps2rl_gpu_check || { echo "[phase2] GPU check failed (set PS2RL_ALLOW_CPU=1 JAX_PLATFORMS=cpu to force CPU)"; exit 1; }
echo "[phase2] $(date '+%F %T') OUT=${OUT} BACKUP=${BACKUP} FILTER=${FILTER} SEED=${SEED}"
echo "[phase2] tracker ${TRACKER_STEPS} steps; PS2 up to ${PS2_STEPS} steps / ${PS2_HOURS} h; ${MAX_PARALLEL} at once; stages ${STAGES}"

has_stage() { [[ " ${STAGES} " == *" $1 "* ]]; }

# run_job NAME CMD...: runs in the background, log in OUT/logs/NAME.log, lines echoed with a prefix
run_job() {
    local name="$1"; shift
    while [ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL}" ]; do sleep 5; done
    echo "[phase2] $(date '+%T') start ${name}"
    ( "$@" 2>&1 | tee "${OUT}/logs/${name}.log" | sed -u "s/^/[${name}] /"; \
      echo "[phase2] $(date '+%T') end ${name} (exit ${PIPESTATUS[0]})" ) &
}

ref_names=()
for item in ${REFS}; do
    name="${item%%:*}"; path="${item#*:}"
    [ -f "${path}" ] || { echo "[phase2] reference not found: ${path}"; exit 1; }
    ref_names+=("${name}")
    declare "REF_${name}=${path}"
done

if has_stage B; then
    for name in "${ref_names[@]}"; do
        ref_var="REF_${name}"; run="${name}_tracker"
        if [ -f "${OUT}/${run}/summary.json" ]; then echo "[phase2] skip ${run} (done)"; continue; fi
        # shellcheck disable=SC2086
        run_job "${run}" python scripts/train_phase2_landing.py --reference "${!ref_var}" --backup "${BACKUP}" \
            --filter "${FILTER}" --projection off --seed "${SEED}" --total_steps "${TRACKER_STEPS}" \
            --out_root "${OUT}" --run_name "${run}" ${TRACKER_ARGS:-}
    done
    wait
fi

if has_stage C; then
    for name in "${ref_names[@]}"; do
        ref_var="REF_${name}"; run="${name}_ps2"; warm="${OUT}/${name}_tracker/best_weights.pkl"
        if [ -f "${OUT}/${run}/summary.json" ]; then echo "[phase2] skip ${run} (done)"; continue; fi
        [ -f "${warm}" ] || { echo "[phase2] ${run}: no tracker checkpoint ${warm}; run stage B first"; continue; }
        # same reset library as the tracker, so both see the same starts
        # shellcheck disable=SC2086
        run_job "${run}" python scripts/train_phase2_landing.py --reference "${!ref_var}" --backup "${BACKUP}" \
            --filter "${FILTER}" --projection on --seed "${SEED}" --total_steps "${PS2_STEPS}" \
            --wall_time_budget_hours "${PS2_HOURS}" --warm_start_weights "${warm}" \
            --init_library "${OUT}/${name}_tracker/init_states.npz" \
            --out_root "${OUT}" --run_name "${run}" ${PS2_ARGS:-}
    done
    wait
fi

if has_stage D; then
    for name in "${ref_names[@]}"; do
        ev="${OUT}/eval_${name}"
        if [ -f "${ev}/summary.md" ]; then echo "[phase2] skip eval_${name} (done)"; continue; fi
        [ -f "${OUT}/${name}_tracker/summary.json" ] || { echo "[phase2] eval_${name}: tracker run missing"; continue; }
        ps2=""; [ -f "${OUT}/${name}_ps2/summary.json" ] && ps2="--ps2_run ${OUT}/${name}_ps2"
        # shellcheck disable=SC2086
        run_job "eval_${name}" python scripts/evaluate_landing_phase2_runs.py --tracker_run "${OUT}/${name}_tracker" \
            ${ps2} --filter "${FILTER}" --backup "${BACKUP}" --title "${name}" --out "${ev}" ${EVAL_ARGS:-}
    done
    wait
fi

if has_stage E; then
    python scripts/summarize_landing_phase2.py --root "${OUT}" --names "${ref_names[@]}" 2>&1 | tee "${OUT}/logs/report.log"
fi
echo "[phase2] $(date '+%F %T') finished; report: ${OUT}/REPORT.md"
