#!/usr/bin/env bash
# Create/refresh the Python 3.10 GPU-JAX training venv at ~/ws_shared/PS2-RL/.venv
# (the same venv the SITL doc uses for training) and verify JAX sees the GPU.
#
# Run inside the container, from anywhere:
#     bash ~/ws_shared/PS2-RL/docker_batch/setup_train_venv.sh
# Idempotent: re-running only (re)installs requirements and re-checks the GPU.
# The venv lives on the ws_shared bind mount, so it survives `docker compose down`.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
VENV="${PS2RL_VENV:-${ROOT}/.venv}"

if ! command -v python3.10 >/dev/null 2>&1; then
    echo "python3.10 not found. Run this inside the px4sitl container (the image installs 3.10 for training)." >&2
    exit 1
fi

# ROS 2 Jazzy puts Python 3.12 site-packages on PYTHONPATH; keep them out of the 3.10 venv.
unset PYTHONPATH
export PYTHONNOUSERSITE=1

if [ ! -x "${VENV}/bin/python" ]; then
    echo "[setup] creating venv ${VENV}"
    python3.10 -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade pip wheel >/dev/null
echo "[setup] installing ${ROOT}/requirements.txt (jax[cuda12]==0.6.2 bundles its own CUDA 12 libraries)"
python -m pip install -r "${ROOT}/requirements.txt"

echo "[setup] checking the GPU"
# shellcheck disable=SC1091
source "${HERE}/landing_env.sh"
if ! ps2rl_gpu_check; then
    cat >&2 <<'EOF'
[setup] JAX did not get the GPU. Checklist:
  1. `nvidia-smi` works inside the container (compose: runtime: nvidia; run.sh: --gpus all).
  2. Host driver supports CUDA 12 (nvidia-smi shows "CUDA Version: 12.x" or newer; driver >= 525).
  3. The venv has the CUDA wheels:  pip list | grep -E "jax|nvidia"
     (jax-cuda12-plugin / jax-cuda12-pjrt 0.6.2 must be present; reinstall with
      pip install --force-reinstall "jax[cuda12]==0.6.2").
EOF
    exit 1
fi
python -c "import ps2rl, qpax; from ps2rl.phase1_sa.quadrotor_landing_sa_trainer import LandingSAConfig; print('[setup] imports ok')"
echo "[setup] done. Next: bash docker_batch/smoke_landing_phase1.sh"
