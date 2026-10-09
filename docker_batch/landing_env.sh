#!/usr/bin/env bash
# Shared environment for PS2-RL training inside the px4sitl container.
# Source it (do not execute):  source docker_batch/landing_env.sh
#
# What it fixes compared with a plain `docker exec -it px4sitl bash` shell:
#   * PYTHONPATH: ~/.bashrc sources ROS 2 Jazzy, which puts Python 3.12
#     site-packages on PYTHONPATH. The training venv is Python 3.10, so those
#     paths are dropped and only the PS2-RL repo is kept.
#   * GPU memory: JAX preallocates 75% of VRAM per process by default, so a
#     second parallel seed (or Gazebo) on the 6 GB RTX 4050 would OOM. The
#     networks here are tiny; allocate on demand instead.
#   * JAX_PLATFORMS=cuda: fail loudly instead of silently falling back to CPU.

_PS2RL_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PS2RL_ROOT="$(cd "${_PS2RL_ENV_DIR}/.." && pwd)"
export PS2RL_VENV="${PS2RL_VENV:-${PS2RL_ROOT}/.venv}"

if [ ! -x "${PS2RL_VENV}/bin/python" ]; then
    echo "[landing_env] venv not found at ${PS2RL_VENV}; run: bash docker_batch/setup_train_venv.sh" >&2
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "${PS2RL_VENV}/bin/activate"
export PYTHONPATH="${PS2RL_ROOT}"
export PYTHONNOUSERSITE=1
# stream python output into the per-stage log files (otherwise it arrives in 8 KB blocks)
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
# Persistent XLA compilation cache: later runs skip the (long) first-chunk compile.
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${PS2RL_ROOT}/.jax_cache}"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

# Escape hatch for debugging without a GPU: PS2RL_ALLOW_CPU=1 JAX_PLATFORMS=cpu ...
if [ "${PS2RL_ALLOW_CPU:-0}" = "1" ]; then
    export PS2RL_GPU_FLAG=""
else
    export PS2RL_GPU_FLAG="--require_gpu"
fi

ps2rl_gpu_check() {
    if [ "${PS2RL_ALLOW_CPU:-0}" = "1" ]; then
        echo "[gpu] PS2RL_ALLOW_CPU=1: skipping the GPU check (JAX_PLATFORMS=${JAX_PLATFORMS})" >&2
        return 0
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[gpu] nvidia-smi not found in the container: the NVIDIA runtime is not attached." >&2
        echo "      Start the container with 'docker compose up -d' (runtime: nvidia) or run.sh (--gpus all)." >&2
        return 1
    fi
    nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader
    python - <<'PY'
import jax
backend = jax.default_backend()
print(f"[gpu] jax {jax.__version__} backend={backend} devices={jax.devices()}")
if backend != "gpu":
    raise SystemExit("[gpu] JAX is not on the GPU; re-run docker_batch/setup_train_venv.sh")
PY
}
