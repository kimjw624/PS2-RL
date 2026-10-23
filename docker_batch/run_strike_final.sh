#!/usr/bin/env bash
# Strike task, final pipeline for the report (sparse arena: 3 obstacles, 1 free chaser, 15 s): every stage for every seed on
# the GPU, one run at a time (two strikers do not fit in 6 GB together), then the evaluation and the report.
#   1. Phase I   safe-arrival backup                            (P1_TASK,  -> <ROOT>/p1/td3_a0p2_seed<s>)
#   2. Phase II  residual striker without the filter (warm-up)   (RES_TASK, -> <ROOT>/residual/striker_rs0p1_s<s>)
#   3. Phase II  PS2: the striker through the deployed filter    (PS2_TASK, -> <ROOT>/ps2/striker_rs0p1_s<s>)
#                (backup of stage 1 and warm start from stage 2 of the same seed)
#   4. evaluation on N test scenes, baselines and ablations      (run_strike_final_eval.sh -> <ROOT>/eval)
#   5. report: tables (markdown + LaTeX) and figures             (scripts/report_strike.py -> <ROOT>/report)
# README: docker_batch/README_strike.md ("Final pipeline").
#
#   docker exec -d px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_strike_final.sh > ~/ws_shared/PS2-RL/outputs/strike_final.out 2>&1"
#   docker exec -it px4sitl bash -c "tail -F ~/ws_shared/PS2-RL/outputs/strike_final.out"
#
# Resumable: every stage skips runs that already have summary.json (evaluations: strike_filter_episodes.json).
# Never edit this file while it runs.
# Knobs (environment variables):
#   STAGES     stages to run                      ("1 2 3 4")   (the report runs at the end of stage 4)
#   SEEDS      seeds (all stages)                 ("0 1 2")
#   ROOT       output root                        (outputs/strike_final)
#   P1_TASK    Phase-I task file                  (docker_batch/configs/strike_final_p1.json)
#   RES_TASK   residual striker task file         (docker_batch/configs/strike_final_residual.json)
#   PS2_TASK   PS2 task file (also the eval's)    (docker_batch/configs/strike_final_ps2.json)
#   N          test scenes per evaluation         (1000)
#   P1_ARGS    extra Phase-I trainer arguments    ()
#   EXTRA_ARGS extra striker trainer arguments (stages 2 and 3), e.g. "--total_steps 5000000"   ()
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGES="${STAGES:-1 2 3 4}"
SEEDS="${SEEDS:-0 1 2}"
ROOT="${ROOT:-outputs/strike_final}"
P1_TASK="${P1_TASK:-docker_batch/configs/strike_final_p1.json}"
RES_TASK="${RES_TASK:-docker_batch/configs/strike_final_residual.json}"
PS2_TASK="${PS2_TASK:-docker_batch/configs/strike_final_ps2.json}"
N="${N:-1000}"
P1_ARGS="${P1_ARGS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
REPO="$(cd "${here}/.." && pwd)"
stamp() { date '+%F %T'; }
echo "[strike-final] $(stamp) stages ${STAGES}; seeds ${SEEDS}; root ${ROOT}"
for st in ${STAGES}; do
    case "${st}" in
        1) echo "[strike-final] $(stamp) stage 1: Phase I (${P1_TASK})"
           TASK="${P1_TASK}" OUT="${ROOT}/p1" ANCHORS=0.2 SEEDS="${SEEDS}" TRAIN_ARGS="${P1_ARGS}" bash "${here}/run_strike_phase1.sh" \
               || { echo "[strike-final] stage 1 failed"; exit 1; } ;;
        2) for s in ${SEEDS}; do
               echo "[strike-final] $(stamp) stage 2, seed ${s}: residual striker without the filter (${RES_TASK})"
               TASK="${RES_TASK}" OUT="${ROOT}/residual" SEEDS="${s}" TRAIN_ARGS="${EXTRA_ARGS}" bash "${here}/run_strike_tracker.sh" \
                   || { echo "[strike-final] stage 2 failed"; exit 1; }
           done ;;
        3) for s in ${SEEDS}; do
               backup="${ROOT}/p1/td3_a0p2_seed${s}"
               warm="${ROOT}/residual/striker_rs0p1_s${s}/best_weights.pkl"
               for f in "${backup}/strike_backup_policy_actor.pkl" "${warm}"; do
                   case "${f}" in /*) path="${f}" ;; *) path="${REPO}/${f}" ;; esac
                   [ -f "${path}" ] || { echo "[strike-final] stage 3 needs ${f} (stages 1 and 2)"; exit 1; }
               done
               echo "[strike-final] $(stamp) stage 3, seed ${s}: PS2 (${PS2_TASK}; backup ${backup}, warm start ${warm})"
               TASK="${PS2_TASK}" OUT="${ROOT}/ps2" SEEDS="${s}" \
                   TRAIN_ARGS="--filter_backup_run ${backup} --warm_start ${warm} ${EXTRA_ARGS}" \
                   bash "${here}/run_strike_tracker.sh" || { echo "[strike-final] stage 3 failed"; exit 1; }
           done ;;
        4) echo "[strike-final] $(stamp) stage 4: evaluation on ${N} test scenes and the report"
           ROOT="${ROOT}" SEEDS="${SEEDS}" N="${N}" TASK="${PS2_TASK}" bash "${here}/run_strike_final_eval.sh" \
               || { echo "[strike-final] stage 4 failed"; exit 1; } ;;
        *) echo "[strike-final] unknown stage ${st}"; exit 1 ;;
    esac
done
echo "[strike-final] $(stamp) all done; report in ${ROOT}/report"
