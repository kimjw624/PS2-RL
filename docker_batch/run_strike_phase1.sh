#!/usr/bin/env bash
# Strike task, Phase I: train the safe-arrival backup (hover LQR + learned residual) for each anchor weight and seed
# (GPU, one run after the other), then compare every run with the hover-LQR backup on the held-out test scenes (CPU).
# README: docker_batch/README_strike.md.
#
#   docker exec -d px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_strike_phase1.sh > ~/ws_shared/PS2-RL/outputs/strike_phase1.out 2>&1"
#   docker exec -it px4sitl bash -c "tail -F ~/ws_shared/PS2-RL/outputs/strike_phase1.out"
#
# Resumable (a run with summary.json is skipped). Never edit this file while it runs (bash reads it as it goes).
# Knobs (environment variables):
#   TASK        task file                                                   (docker_batch/configs/strike.json)
#   OUT         output root                                                 (outputs/strike_phase1)
#   ANCHORS     weights of the anchor to the LQR (action_smoothness_weight)  ("0.5 0.2")
#   SEEDS       seeds                                                       ("0")
#   BACKBONE    td3 | sac                                                   (td3)
#   STEPS       env steps per run (empty = task file, 5M)                   ()
#   TRAIN_ARGS  extra trainer arguments                                     ()
# Run names: <backbone>_a<anchor>_seed<seed>, e.g. td3_a0p5_seed0.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${here}/landing_env.sh" || exit 1
cd "${PS2RL_ROOT}"

TASK="${TASK:-docker_batch/configs/strike.json}"
OUT="${OUT:-outputs/strike_phase1}"
ANCHORS="${ANCHORS:-0.5 0.2}"
SEEDS="${SEEDS:-0}"
BACKBONE="${BACKBONE:-td3}"
STEPS="${STEPS:-}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
mkdir -p "${OUT}/logs"
stamp() { date '+%F %T'; }
echo "[strike-p1] $(stamp) task ${TASK} -> ${OUT}; anchors ${ANCHORS}; seeds ${SEEDS}; backbone ${BACKBONE}"
ps2rl_gpu_check || exit 1

runs=()
for anchor in ${ANCHORS}; do
    for seed in ${SEEDS}; do
        name="${BACKBONE}_a${anchor//./p}_seed${seed}"
        runs+=("${OUT}/${name}")
        if [ -f "${OUT}/${name}/summary.json" ]; then
            echo "[strike-p1] ${name}: done before, skipped"
            continue
        fi
        echo "[strike-p1] $(stamp) ${name}: training (log ${OUT}/logs/${name}.log)"
        steps_arg=()
        [ -n "${STEPS}" ] && steps_arg=(--total_steps "${STEPS}")
        # shellcheck disable=SC2086
        if python scripts/train_phase1_strike.py --task "${TASK}" --output_dir "${OUT}" --run_name "${name}" --seed "${seed}" \
                --sa_backbone "${BACKBONE}" --action_smoothness_weight "${anchor}" "${steps_arg[@]}" ${PS2RL_GPU_FLAG} ${TRAIN_ARGS} \
                > "${OUT}/logs/${name}.log" 2>&1; then
            grep -E "^\[done\]" "${OUT}/logs/${name}.log" | sed "s/^/[strike-p1]   /"
        else
            echo "[strike-p1] ${name}: training failed (see ${OUT}/logs/${name}.log)"
        fi
    done
done

done_runs=()
for r in "${runs[@]}"; do [ -f "${r}/summary.json" ] && done_runs+=("${r}"); done
if [ "${#done_runs[@]}" -gt 0 ]; then
    echo "[strike-p1] $(stamp) comparison with the LQR backup (CPU, log ${OUT}/logs/compare.log)"
    env JAX_PLATFORMS=cpu python scripts/compare_phase1_strike.py --runs "${done_runs[@]}" --out "${OUT}/compare" \
        > "${OUT}/logs/compare.log" 2>&1 || echo "[strike-p1] comparison failed (see ${OUT}/logs/compare.log)"
    grep -E "^\|" "${OUT}/logs/compare.log" | sed "s/^/[strike-p1]   /"
fi
echo "[strike-p1] $(stamp) all done; figures in ${OUT}/compare"
