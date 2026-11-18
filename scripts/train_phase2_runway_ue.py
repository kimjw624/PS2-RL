"""Phase II for the runway bird-deterrence task: SAC through the runway UE-bCBF CIL.

Two stages, as for landing (and the repo's powerloop quadrotor):

    # 1) chasing policy without the filter (warm-start source); episodes are not cut at a violation
    JAX_ENABLE_X64=1 python scripts/train_phase2_runway_ue.py --ckpt outputs/runway_phase1_ue/rwy_ue_h128_seed0 \
        --use_projection false --env_terminate_on_unsafe false --total_steps 3000000 --update_every 2 \
        --replay_size 1000000 --run_tag rwy_chase_vanilla_s0

    # 2) Phase II through the CIL, warm-started from (1)
    JAX_ENABLE_X64=1 python scripts/train_phase2_runway_ue.py --ckpt outputs/runway_phase1_ue/rwy_ue_h128_seed0 \
        --warm_start_weights outputs/runway_phase2_ue/rwy_chase_vanilla_s0/best_weights.pkl \
        --start_steps 0 --update_after 16384 --actor_lr 5e-5 --critic_lr 1e-4 --alpha_lr 5e-5 \
        --total_steps 1000000 --run_tag rwy_ps2_warm_s0

Flags: every field of ``RunwayPS2Config`` (``--name value``) and of ``RunwayBirdEnvConfig``
(``--env_name value``). Output: outputs/runway_phase2_ue/<run_tag>/ (best/final weights,
history, summary, configs).
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

from ps2rl.envs.quadrotor_runway_bird_env import RunwayBirdEnvConfig  # noqa: E402
from ps2rl.phase2_ps2.runway_ue_ps2_trainer import RunwayPS2Config, run_runway_ue_ps2  # noqa: E402

_SMOKE = {"total_steps": 2048, "num_envs": 8, "steps_per_jit": 8, "start_steps": 256, "update_after": 128,
          "batch_size": 32, "replay_size": 4096, "eval_every": 1024, "eval_episodes": 4, "log_every": 512,
          "hidden_size": 64, "update_every": 8}


def _add(p, cls, prefix=""):
    for f in fields(cls):
        if isinstance(f.default, bool):
            p.add_argument(f"--{prefix}{f.name}", type=lambda s: str(s).lower() in ("1", "true", "yes"), default=None)
        elif isinstance(f.default, (int, float, str)):
            p.add_argument(f"--{prefix}{f.name}", type=type(f.default), default=None)


def _collect(a, cls, prefix=""):
    return {f.name: getattr(a, f"{prefix}{f.name}") for f in fields(cls) if getattr(a, f"{prefix}{f.name}", None) is not None}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="runway UE Phase-I run directory (or its runway_backup_policy_actor.pkl)")
    p.add_argument("--output_root", default="outputs/runway_phase2_ue")
    p.add_argument("--run_tag", default="rwy_ps2")
    p.add_argument("--smoke_test", action="store_true")
    _add(p, RunwayPS2Config)
    _add(p, RunwayBirdEnvConfig, prefix="env_")
    a = p.parse_args(argv)
    over = _collect(a, RunwayPS2Config)
    env_over = _collect(a, RunwayBirdEnvConfig, prefix="env_")
    if a.smoke_test:
        over = {**_SMOKE, **over}
        env_over = {"require_recoverable": False, "bank_size": 256, **env_over}
    cfg = RunwayPS2Config.from_dict(over)
    env_cfg = RunwayBirdEnvConfig(**env_over)
    out = Path(a.output_root) / a.run_tag
    print(f"[jax] backend={jax.default_backend()} | run {out} | total_steps={cfg.total_steps} num_envs={cfg.num_envs} "
          f"batch={cfg.batch_size} update_every={cfg.update_every} "
          + (f"rows_keep={cfg.rows_keep}" if cfg.use_projection else "NO CIL (chasing policy alone)")
          + (f" | warm start {cfg.warm_start_weights}" if cfg.warm_start_weights else ""), flush=True)
    res = run_runway_ue_ps2(cfg, a.ckpt, env_cfg, out)
    s = res["summary"]
    print(json.dumps({"best_step": s["best_step"], "best_eval": s["best_eval"], "wall_time_min": s["wall_time_sec"] / 60},
                     indent=1), flush=True)


if __name__ == "__main__":
    main()
