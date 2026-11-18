#!/usr/bin/env bash
# Strike task, full arena (6..10 obstacles, 0..2 chasers, 15 s): the whole pipeline on the GPU, one stage after the other.
#   1. Phase I   safe-arrival backup with moving chasers and walls  (docker_batch/configs/strike_full.json)
#   2. Phase II  residual striker without the filter (warm-up)        (docker_batch/configs/strike_full_residual.json)
#   3. Phase II  PS2: the striker through the filter with the stage-1 backup, warm-started from stage 2
#                                                                    (docker_batch/configs/strike_full_ps2.json)
# README: docker_batch/README_strike.md ("Full arena").
#
#   docker exec -d px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_strike_full.sh > ~/ws_shared/PS2-RL/outputs/strike_full.out 2>&1"
#   docker exec -it px4sitl bash -c "tail -F ~/ws_shared/PS2-RL/outputs/strike_full.out"
#
# Resumable: every stage skips the runs that already have summary.json. Never edit this file while it runs.
# Knobs (environment variables):
#   STAGES      stages to run                         ("1 2 3")
#   P1_OUT      Phase-I output root                    (outputs/strike_phase1_full; run td3_a0p2_seed0)
#   RES_OUT     residual striker output root           (outputs/strike_striker_full_residual)
#   PS2_OUT     PS2 output root                        (outputs/strike_striker_full_ps2)
#   P1_ARGS     extra Phase-I trainer arguments         ()
# The stage-3 task file reads the stage-1 backup and the stage-2 weights from the default output roots.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGES="${STAGES:-1 2 3}"
P1_OUT="${P1_OUT:-outputs/strike_phase1_full}"
RES_OUT="${RES_OUT:-outputs/strike_striker_full_residual}"
PS2_OUT="${PS2_OUT:-outputs/strike_striker_full_ps2}"
P1_ARGS="${P1_ARGS:-}"
stamp() { date '+%F %T'; }
echo "[strike-full] $(stamp) stages ${STAGES}"
for st in ${STAGES}; do
    case "${st}" in
        1) echo "[strike-full] $(stamp) stage 1: Phase I (moving chasers, walls) -> ${P1_OUT}"
           TASK=docker_batch/configs/strike_full.json OUT="${P1_OUT}" ANCHORS=0.2 SEEDS=0 TRAIN_ARGS="${P1_ARGS}" \
               bash "${here}/run_strike_phase1.sh" || { echo "[strike-full] stage 1 failed"; exit 1; } ;;
        2) echo "[strike-full] $(stamp) stage 2: residual striker, no filter -> ${RES_OUT}"
           TASK=docker_batch/configs/strike_full_residual.json OUT="${RES_OUT}" SEEDS=0 \
               bash "${here}/run_strike_tracker.sh" || { echo "[strike-full] stage 2 failed"; exit 1; } ;;
        3) backup="${P1_OUT}/td3_a0p2_seed0"
           warm="${RES_OUT}/striker_rs0p1_s0/best_weights.pkl"
           for f in "${backup}/strike_backup_policy_actor.pkl" "${warm}"; do
               case "${f}" in /*) path="${f}" ;; *) path="${PS2RL_ROOT:-$(cd "${here}/.." && pwd)}/${f}" ;; esac
               [ -f "${path}" ] || { echo "[strike-full] stage 3 needs ${f} (stages 1 and 2)"; exit 1; }
           done
           echo "[strike-full] $(stamp) stage 3: PS2 through the filter (backup ${backup}, warm start ${warm}) -> ${PS2_OUT}"
           TASK=docker_batch/configs/strike_full_ps2.json OUT="${PS2_OUT}" SEEDS=0 \
               TRAIN_ARGS="--filter_backup_run ${backup} --warm_start ${warm}" \
               bash "${here}/run_strike_tracker.sh" || { echo "[strike-full] stage 3 failed"; exit 1; } ;;
        *) echo "[strike-full] unknown stage ${st}"; exit 1 ;;
    esac
done
echo "[strike-full] $(stamp) all done"
