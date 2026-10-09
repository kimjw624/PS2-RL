"""Phase II for landing under a disturbance - fast standalone trainer with cached CIL rows (CPU path).

The repo-structured entry point is ``scripts/train_phase2_landing_ue.py`` (repo trainer +
binding/env swaps). This script runs the same SAC-through-CIL with the CIL rows cached in
the replay buffer (~10x faster updates without a GPU); it is what produced the shipped model.

    JAX_ENABLE_X64=1 python scripts/train_phase2_landing_ue_cached.py \
        --ckpt outputs/landing_phase1_ue/ue_floor_rec10_td3_h128_seed0 \
        --seed 0 --total_steps 5000000 --run_tag land_ue_p2_res_s0

    # two stages, as the repo does for the powerloop quadrotor (vanilla tracker -> PS2 warm-started from it):
    # 1) vanilla tracker: same env/observation, no CIL, episodes not cut at a violation
    python scripts/train_phase2_landing_ue_cached.py --ckpt <ckpt> --env_nominal_controller none \
        --use_projection false --env_terminate_on_unsafe false --run_tag land_ue_vanilla_s0
    # 2) PS2 through the CIL, warm-started from (1)
    JAX_ENABLE_X64=1 python scripts/train_phase2_landing_ue_cached.py --ckpt <ckpt> --env_nominal_controller none \
        --warm_start_weights outputs/landing_phase2_ue/land_ue_vanilla_s0/best_weights.pkl \
        --start_steps 0 --actor_lr 5e-5 --critic_lr 1e-4 --alpha_lr 5e-5 --run_tag land_ue_p2_warm_s0

    # CPU smoke test (untrained backup allowed)
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python scripts/train_phase2_landing_ue_cached.py --ckpt <ckpt> --smoke_test

JAX_ENABLE_X64=1 is for the float64 QP (0 % solver fallbacks vs ~2 % in float32); without it
pass --qp_float64 false.

The checkpoint must come from ``scripts/train_phase1_landing_ue.py`` (backup input (x, d_hat),
metadata with ``ue_config``). Disturbance bounds, observer gain and tube settings are read
from it; the environment's sinusoidal disturbance uses the same amplitude and frequency.

Outputs (``outputs/landing_phase2_ue/<run_tag>/``): best_weights.pkl (by eval return,
unsafe episodes penalised), final_weights.pkl, history.json, summary.json, configs.json.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402

from ps2rl.envs.quadrotor_landing_ue_env import LandingUEEnvConfig  # noqa: E402
from ps2rl.phase2_ps2.landing_ue_ps2_trainer import UEPS2Config, run_landing_ue_ps2  # noqa: E402

_SMOKE = {"total_steps": 2048, "num_envs": 8, "steps_per_jit": 8, "start_steps": 256, "update_after": 128,
          "batch_size": 32, "replay_size": 4096, "eval_every": 1024, "eval_episodes": 4, "log_every": 512,
          "hidden_size": 64, "update_every": 8}


def _add(p, cls, prefix=""):
    for f in fields(cls):
        flag = f"--{prefix}{f.name}"
        if isinstance(f.default, bool):
            p.add_argument(flag, type=lambda s: str(s).lower() in ("1", "true", "yes"), default=None)
        elif isinstance(f.default, (int, float, str)):
            p.add_argument(flag, type=type(f.default), default=None)


def _collect(a, cls, prefix=""):
    return {f.name: getattr(a, f"{prefix}{f.name}") for f in fields(cls) if getattr(a, f"{prefix}{f.name}", None) is not None}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output_root", default="outputs/landing_phase2_ue")
    p.add_argument("--run_tag", default="land_ue_p2")
    p.add_argument("--smoke_test", action="store_true")
    _add(p, UEPS2Config)
    _add(p, LandingUEEnvConfig, prefix="env_")
    a = p.parse_args(argv)
    over = _collect(a, UEPS2Config)
    env_over = _collect(a, LandingUEEnvConfig, prefix="env_")
    if a.smoke_test:
        over = {**_SMOKE, **over}
        env_over = {"require_recoverable": False, "bank_size": 256, **env_over}
    cfg = UEPS2Config.from_dict(over)
    env_cfg = LandingUEEnvConfig(**env_over)
    out = Path(a.output_root) / a.run_tag
    print(f"[jax] backend={jax.default_backend()} | run {out} | total_steps={cfg.total_steps} num_envs={cfg.num_envs} "
          f"batch={cfg.batch_size} update_every={cfg.update_every} "
          + (f"rows_keep={cfg.rows_keep}" if cfg.use_projection else "NO CIL (vanilla tracker)")
          + (f" | warm start {cfg.warm_start_weights}" if cfg.warm_start_weights else ""), flush=True)
    res = run_landing_ue_ps2(cfg, a.ckpt, env_cfg, out)
    s = res["summary"]
    print(json.dumps({"best_step": s["best_step"], "best_eval": s["best_eval"], "wall_time_min": s["wall_time_sec"] / 60},
                     indent=1), flush=True)


if __name__ == "__main__":
    main()
