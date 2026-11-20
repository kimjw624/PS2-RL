#!/usr/bin/env bash
# Obstacle-only online task, Phase I: train the safe-arrival backup (hover LQR + learned residual) once per network
# width, one run after the other, then export every run to the numpy format the onboard filter loads
# (checkpoints/cyl_online/<run>/). README: docker_batch/README_cyl_online.md.
#
#   docker exec -d px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_cyl_online_phase1.sh > ~/ws_shared/PS2-RL/outputs/cyl_online_phase1.out 2>&1"
#   docker exec -it px4sitl bash -c "tail -F ~/ws_shared/PS2-RL/outputs/cyl_online_phase1.out"
#
# The runs are small (4 M steps, 48-dim observation): about 10 min (width 64) to 40 min (width 256) each on a 2-core
# CPU, faster on the GPU. Resumable (a run with summary.json is skipped). Never edit this file while it runs.
# Knobs (environment variables):
#   TASK        task file                         (docker_batch/configs/cyl_online_p1.json)
#   OUT         training output root              (outputs/cyl_online_phase1)
#   WIDTHS      hidden widths (two hidden layers) ("64 32 128 256")
#   SEED        seed                              (0)
#   STEPS       env steps per run (empty = task)  ()
#   EXPORT      export root (empty = no export)   (checkpoints/cyl_online)
#   TRAIN_ARGS  extra trainer arguments           ()
# Run names: td3_w<width>_seed<seed>, e.g. td3_w64_seed0.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${here}/landing_env.sh" || exit 1
cd "${PS2RL_ROOT}"

TASK="${TASK:-docker_batch/configs/cyl_online_p1.json}"
OUT="${OUT:-outputs/cyl_online_phase1}"
WIDTHS="${WIDTHS:-64 32 128 256}"
SEED="${SEED:-0}"
STEPS="${STEPS:-}"
EXPORT="${EXPORT-checkpoints/cyl_online}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
mkdir -p "${OUT}/logs"
stamp() { date '+%F %T'; }
echo "[cyl-p1] $(stamp) task ${TASK} -> ${OUT}; widths ${WIDTHS}; seed ${SEED}"
if [ "${PS2RL_ALLOW_CPU:-0}" != "1" ]; then
    ps2rl_gpu_check || exit 1
fi

for w in ${WIDTHS}; do
    name="td3_w${w}_seed${SEED}"
    if [ -f "${OUT}/${name}/summary.json" ]; then
        echo "[cyl-p1] ${name}: done before, skipped"
    else
        echo "[cyl-p1] $(stamp) ${name}: training (log ${OUT}/logs/${name}.log)"
        steps_arg=()
        [ -n "${STEPS}" ] && steps_arg=(--total_steps "${STEPS}")
        # shellcheck disable=SC2086
        if python scripts/train_phase1_strike.py --task "${TASK}" --output_dir "${OUT}" --run_name "${name}" --seed "${SEED}" \
                --hidden_size "${w}" "${steps_arg[@]}" ${PS2RL_GPU_FLAG:-} ${TRAIN_ARGS} > "${OUT}/logs/${name}.log" 2>&1; then
            grep -E "^\[done\]" "${OUT}/logs/${name}.log" | sed "s/^/[cyl-p1]   /"
        else
            echo "[cyl-p1] ${name}: training failed (see ${OUT}/logs/${name}.log)"
            continue
        fi
    fi
    if [ -n "${EXPORT}" ]; then
        JAX_PLATFORMS=cpu python scripts/export_cyl_backup.py --run "${OUT}/${name}" --out "${EXPORT}/${name}" \
            | sed "s/^/[cyl-p1]   /"
    fi
done
echo "[cyl-p1] $(stamp) finished"
