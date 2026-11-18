#!/usr/bin/env bash
# Landing Phase I: SAC vs TD3 over seeds on the container GPU, then summarize.
#
#     bash ~/ws_shared/PS2-RL/docker_batch/run_landing_phase1_compare.sh
#
# Knobs (environment variables):
#     SEEDS="0 1 2"          BACKBONES="sac td3"      MAX_PARALLEL=2
#     TOTAL_STEPS=5000000    OUT=outputs/landing_phase1_compare
#     EXTRA_ARGS="..."       extra flags for scripts/train_phase1_landing.py (e.g. "--num_envs 128")
#
# Detached (survives closing the terminal), from the HOST:
#     docker exec -d px4sitl bash -c "mkdir -p ~/ws_shared/PS2-RL/outputs && \
#         bash ~/ws_shared/PS2-RL/docker_batch/run_landing_phase1_compare.sh \
#         > ~/ws_shared/PS2-RL/outputs/landing_phase1_compare.out 2>&1"
#     docker exec -it px4sitl tail -f ~/ws_shared/PS2-RL/outputs/landing_phase1_compare.out
#
# Re-running resumes: runs that already have summary.json are skipped. Runs share
# one GPU; MAX_PARALLEL=2 suits a 6 GB laptop GPU (each process uses a few hundred
# MB because preallocation is off). Close Gazebo while training.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/landing_env.sh" || exit 1
cd "${PS2RL_ROOT}"
ps2rl_gpu_check || exit 1

SEEDS="${SEEDS:-0 1 2}"
BACKBONES="${BACKBONES:-sac td3}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
TOTAL_STEPS="${TOTAL_STEPS:-5000000}"
OUT="${OUT:-outputs/landing_phase1_compare}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
LOGS="${OUT}/logs"
mkdir -p "${LOGS}"

echo "[compare] seeds=(${SEEDS}) backbones=(${BACKBONES}) steps=${TOTAL_STEPS} parallel=${MAX_PARALLEL} out=${OUT}"

# 1) Certificate once (CPU, float64). Aborts the batch if c_B / z_des are not certified.
if [ ! -f "${OUT}/certificate.json" ]; then
    echo "[compare] certifying the base set -> ${OUT}/certificate.json"
    if ! JAX_PLATFORMS=cpu python scripts/certify_landing_base_set.py --out "${OUT}/certificate.json" \
            > "${LOGS}/certificate.log" 2>&1; then
        echo "[compare] certification script failed, see ${LOGS}/certificate.log" >&2
        exit 1
    fi
fi
python - "${OUT}/certificate.json" <<'PY' || exit 1
import json, sys
r = json.load(open(sys.argv[1]))
c = r["chosen"]
print(f"[compare] certificate: z_des={r['config']['z_des']} c_B={r['config']['base_set_c']} c_bar={c['c_bar']:.2f} "
      f"({c['binding']}), c_Lyap={r['c_lyap_adversarial']:.2f}, invariance max V={r['invariance_max_V']:.2f}")
if not r["chosen_ok"]:
    sys.exit("[compare] c_B is not certified at this z_des; fix the config before training")
PY

# 2) Runs, interleaved by seed so partial results are always a paired comparison.
run_one() {
    local bb="$1" seed="$2"
    local name="${bb}_seed${seed}"
    if [ -f "${OUT}/${name}/summary.json" ]; then
        echo "[compare] ${name}: already done, skipping"
        return 0
    fi
    rm -rf "${OUT:?}/${name}"
    echo "[compare] ${name}: start $(date +%H:%M:%S) (log ${LOGS}/${name}.log)"
    # shellcheck disable=SC2086
    if python scripts/train_phase1_landing.py ${PS2RL_GPU_FLAG} --skip_lyap_check \
            --backbone "${bb}" --seed "${seed}" --total_steps "${TOTAL_STEPS}" \
            --output_dir "${OUT}" --run_name "${name}" ${EXTRA_ARGS} > "${LOGS}/${name}.log" 2>&1; then
        echo "[compare] ${name}: $(grep -E '^\[done\]' "${LOGS}/${name}.log" | tail -1)"
    else
        echo "[compare] ${name}: FAILED at $(date +%H:%M:%S) - last lines of ${LOGS}/${name}.log:" >&2
        tail -n 15 "${LOGS}/${name}.log" >&2
    fi
}

for seed in ${SEEDS}; do
    for bb in ${BACKBONES}; do
        while [ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL}" ]; do
            wait -n || true
        done
        run_one "${bb}" "${seed}" &
        sleep 20  # stagger JIT compiles so the jobs do not fight for the CPU at start-up
    done
done
wait

# 3) Summary table + learning curves.
python "${HERE}/summarize_landing_phase1.py" --root "${OUT}"
