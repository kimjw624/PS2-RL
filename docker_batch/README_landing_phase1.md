# Landing Phase I on the container GPU

Everything below runs **inside the `px4sitl` container**, from `~/ws_shared/PS2-RL`.
Close Gazebo while training so it does not compete for the GPU.

## 1. One-time: training venv + GPU check

```bash
docker compose up -d                  # compose already attaches the GPU (runtime: nvidia)
docker exec -it px4sitl bash
cd ~/ws_shared/PS2-RL
bash docker_batch/setup_train_venv.sh
```

- Creates or refreshes `.venv` (Python 3.10) from `requirements.txt`, which pins `jax[cuda12]==0.6.2`.
- The `jax[cuda12]` wheels bundle their own CUDA 12 libraries, so the image needs no CUDA toolkit, only the host driver.
- It must print `backend=gpu devices=[CudaDevice(id=0)]`.
- The venv lives on the bind mount, so it survives `docker compose down`.

## 2. Smoke test + ETA (a few minutes, mostly compilation)

```bash
bash docker_batch/smoke_landing_phase1.sh
```

This runs both backbones with tiny networks, then a 200k-step full-size probe. At the end it prints `[eta] ... one 5M-step run ~N min`.

## 3. SAC vs TD3 over seeds

Run it in the foreground:

```bash
bash docker_batch/run_landing_phase1_compare.sh
```

Or run it detached from the **host**, so it survives closing the terminal:

```bash
docker exec -d px4sitl bash -c "mkdir -p ~/ws_shared/PS2-RL/outputs && \
    bash ~/ws_shared/PS2-RL/docker_batch/run_landing_phase1_compare.sh \
    > ~/ws_shared/PS2-RL/outputs/landing_phase1_compare.out 2>&1"
docker exec -it px4sitl tail -f ~/ws_shared/PS2-RL/outputs/landing_phase1_compare.out
docker exec -it px4sitl nvidia-smi -l 2           # GPU utilization / memory every 2 s
```

The script does three things:
1. It certifies the base set once (CPU, float64) and aborts if c_B = 12 at z_des = 1.25 m is not certified.
2. It runs seeds 0, 1, 2 × {sac, td3}, two at a time.
3. It summarizes the results.

It is resumable: re-running skips runs that already have `summary.json`, and a failed run is re-run from scratch.

| variable | default | notes |
|---|---|---|
| `SEEDS` | `0 1 2` | |
| `BACKBONES` | `sac td3` | |
| `MAX_PARALLEL` | `2` | right for a 6 GB laptop GPU; each process takes a few hundred MB |
| `TOTAL_STEPS` | `5000000` | |
| `OUT` | `outputs/landing_phase1_compare` | |
| `EXTRA_ARGS` | *(empty)* | passed to `scripts/train_phase1_landing.py`, e.g. `"--num_envs 128"` |

To re-summarize at any time (including mid-batch, over the finished runs):

```bash
source docker_batch/landing_env.sh
python docker_batch/summarize_landing_phase1.py --root outputs/landing_phase1_compare
```

## Outputs

```
outputs/landing_phase1_compare/
  certificate.json               level bounds, z_des sweep, containment/invariance checks
  logs/<backbone>_seed<k>.log     full training log per run
  <backbone>_seed<k>/
    summary.json                 untrained / best-val / final-val / test-at-best recoverability per region
    history.json                 losses, alpha, entropy, per-eval m_general / m_edge / m_shell
    configs.json                 landing + design-region + trainer config
    best_weights.pkl             learner state at the best val checkpoint (final_weights.pkl too)
    landing_backup_policy_actor.pkl   actor checkpoint in the learned-backup format (+ certificate metadata)
  runs.csv                       one row per run
  learning_curves.png            mean over seeds with min–max band
```

**Selection and reporting.** The best checkpoint is chosen on the val split; the reported numbers are the test split at that checkpoint.

**Detecting collapse.** `best-final val` is the drop from the best to the final val score. A large value means training collapsed late, which is what TD3 did in the short test run.

## What the GPU settings in `landing_env.sh` do

- **`PYTHONPATH=<PS2-RL>`.** `~/.bashrc` sources ROS 2 Jazzy, which puts Python 3.12 packages on `PYTHONPATH`. That is wrong for the 3.10 venv.
- **`JAX_PLATFORMS=cuda`.** A missing GPU becomes an error instead of a silent CPU run. `--require_gpu` checks the backend again inside each run, and every log starts with `[jax] backend=...`.
- **`XLA_PYTHON_CLIENT_PREALLOCATE=false`.** JAX otherwise grabs 75% of VRAM per process, so a second parallel seed, or Gazebo, would run out of memory.
- **`JAX_COMPILATION_CACHE_DIR=.jax_cache`.** Runs after the first skip most of the compile time.
- **Debugging without a GPU.** `PS2RL_ALLOW_CPU=1 JAX_PLATFORMS=cpu bash docker_batch/...`

## Files

New:
- `ps2rl/base_controller/quadrotor_landing_dlqr.py`: 9-D hover LQR above the pad
- `ps2rl/envs/quadrotor_landing_config.py`: certified defaults
- `ps2rl/sets/quadrotor_cone_sets.py`, `ps2rl/sets/quadrotor_landing_safe_set.py`: smooth approach cone, and cone ∩ pad plane
- `ps2rl/sets/landing_certificate.py`: c_U, chart, cone, ground, recovery envelope, adversarial Lyapunov
- `ps2rl/phase1_sa/landing_design_region.py`: altitude-uniform / edge / shell sampler
- `ps2rl/phase1_sa/quadrotor_landing_sa_env.py`
- `ps2rl/phase1_sa/quadrotor_landing_sa_trainer.py`
- `scripts/certify_landing_base_set.py`
- `scripts/train_phase1_landing.py`
- `docker_batch/`: this folder

Modified:
- `ps2rl/phase1_sa/sa_trainer_core.py`: SAC backbone; the TD3 path is bit-identical
- `ps2rl/utils/policy.py`: optional C¹ actor activation (default `relu` unchanged)

The training uses the pip CUDA wheels in the venv, so the container image needs no CUDA toolkit.

## Troubleshooting

- **`[gpu] nvidia-smi not found`**: the container was started without the NVIDIA runtime. Use `docker compose up -d`, or `run.sh`, which passes `--gpus all`.
- **`backend=cpu` in the venv**: check that `pip list | grep -E "jax|nvidia"` shows `jax-cuda12-plugin` and `jax-cuda12-pjrt` 0.6.2. Also check that the host driver supports CUDA 12 (`nvidia-smi` shows `CUDA Version: 12.x`+).
- **Out of memory with Gazebo open**: close Gazebo or use `MAX_PARALLEL=1`.
- **A run died**: `logs/<run>.log` has the traceback. Re-running the compare script re-runs only unfinished runs.
