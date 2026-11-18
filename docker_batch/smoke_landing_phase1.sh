#!/usr/bin/env bash
# Quick GPU check of landing Phase I inside the container (~2-5 min, mostly JIT compile):
#   1. functional smoke test of both backbones (tiny nets, 4k steps)
#   2. throughput probe with the full-size config, printing an ETA for a full run
#
#     bash ~/ws_shared/PS2-RL/docker_batch/smoke_landing_phase1.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/landing_env.sh"
cd "${PS2RL_ROOT}"
ps2rl_gpu_check

OUT="${OUT:-outputs/landing_phase1_smoke}"
PROBE_STEPS="${PROBE_STEPS:-200000}"
FULL_STEPS="${FULL_STEPS:-5000000}"
mkdir -p "${OUT}"

for bb in sac td3; do
    echo "=== smoke: ${bb}"
    python scripts/train_phase1_landing.py --smoke_test ${PS2RL_GPU_FLAG} --backbone "${bb}" \
        --output_dir "${OUT}" --run_name "smoke_${bb}" 2>&1 | tee "${OUT}/smoke_${bb}.log" | grep -E "^\[(jax|certificate|done)\]"
done

echo "=== throughput probe: sac, full-size nets, ${PROBE_STEPS} steps (no in-loop evals)"
python scripts/train_phase1_landing.py ${PS2RL_GPU_FLAG} --skip_lyap_check --backbone sac \
    --total_steps "${PROBE_STEPS}" --eval_every 0 --log_every 25000 \
    --output_dir "${OUT}" --run_name probe_sac 2>&1 | tee "${OUT}/probe_sac.log" | grep -E "^\[(jax|done)\]|^step="

# Median of the logged rates, excluding the first and last: both include a JIT compile
# (the last chunk is shorter than steps_per_jit and gets its own compile).
rate="$(grep -oE 'eps/sec=[0-9.]+' "${OUT}/probe_sac.log" | cut -d= -f2 | python -c "
import sys, statistics
r = [float(x) for x in sys.stdin.read().split()]
r = r[1:-1] if len(r) > 2 else r
print(statistics.median(r) if r else '')")"
if [ -n "${rate}" ]; then
    python - "${rate}" "${FULL_STEPS}" <<'PY'
import sys
rate, full = float(sys.argv[1]), float(sys.argv[2])
print(f"[eta] steady state ~{rate:,.0f} env steps/s -> one {full/1e6:.0f}M-step run ~{full / rate / 60:.0f} min "
      f"(two runs in parallel on one GPU are each slower, but finish the pair sooner than running them back to back)")
PY
fi
echo "smoke outputs: ${PS2RL_ROOT}/${OUT}"
