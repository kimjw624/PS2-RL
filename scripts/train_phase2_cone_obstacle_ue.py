#!/usr/bin/env python
"""Phase II for the cone-obstacle task under a disturbance (two stages, as for the UE landing).

    # 1) vanilla tracker: same environment/observation, no CIL, episodes not cut at a violation
    python scripts/train_phase2_cone_obstacle_ue.py --use_projection false --env_terminate_on_unsafe false \
        --run_tag cone_ue_vanilla_s0
    # 2) PS2 through the UE-bCBF CIL, warm-started from (1)
    JAX_ENABLE_X64=1 python scripts/train_phase2_cone_obstacle_ue.py \
        --warm_start_weights outputs/cone_obstacle_phase2_ue/cone_ue_vanilla_s0/best_weights.pkl \
        --start_steps 0 --actor_lr 5e-5 --critic_lr 1e-4 --alpha_lr 5e-5 --run_tag cone_ue_ps2_s0 \
        [--backup learned --backup_path outputs/cone_obstacle_phase1/sa_plain_seed0]
    # CPU smoke test
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python scripts/train_phase2_cone_obstacle_ue.py --smoke_test

Task (funnel, floor, base set, UE bounds, filter gains, reference) from ``--task``
(default docker_batch/configs/cone_obstacle_ue.json); the backup is the base set's LQR
(``--backup lqr``) or a Phase-I safe-arrival actor (``--backup learned --backup_path <run>``),
chosen by ``scripts/compare_phase1_cone_obstacle.py`` (its ``chosen_backup.json`` can be passed
with ``--backup_from``).

Disturbance: d(t) = A n sin(2 pi f t + phi), n a random unit vector per episode, A and f drawn per
episode with A <= delta_d and 2 pi f A <= delta_v (``--env_disturbance_mode random``; ``fixed`` =
A = delta_d at the task's design frequency).

Outputs (``outputs/cone_obstacle_phase2_ue/<run_tag>/``): best/final weights, history, summary, configs.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import jax  # noqa: E402

from cone_obstacle_common import load_task  # noqa: E402
from ps2rl.envs.cone_obstacle_ue_env import ConeObstacleUEEnvConfig  # noqa: E402
from ps2rl.phase2_ps2.cone_obstacle_ue_ps2_trainer import ConeUEPS2Config, run_cone_obstacle_ue_ps2  # noqa: E402

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
    p.add_argument("--task", default="docker_batch/configs/cone_obstacle_ue.json")
    p.add_argument("--backup_from", default="", help="chosen_backup.json of compare_phase1_cone_obstacle.py "
                                                     "(sets --backup / --backup_path unless given)")
    p.add_argument("--output_root", default="outputs/cone_obstacle_phase2_ue")
    p.add_argument("--run_tag", default="cone_ue_p2")
    p.add_argument("--smoke_test", action="store_true")
    _add(p, ConeUEPS2Config)
    _add(p, ConeObstacleUEEnvConfig, prefix="env_")
    a = p.parse_args(argv)
    task = load_task(a.task)
    over = _collect(a, ConeUEPS2Config)
    env_over = _collect(a, ConeObstacleUEEnvConfig, prefix="env_")
    if a.backup_from:
        ch = json.loads(Path(a.backup_from).read_text())
        over.setdefault("backup", ch["backup"])
        over.setdefault("backup_path", ch.get("backup_path", ""))
    if "reference" in task and "reference_path" not in env_over:
        env_over["reference_path"] = task["reference"]
    if a.smoke_test:
        over = {**_SMOKE, **over}
        env_over = {"bank_size": 256, **env_over}
    cfg = ConeUEPS2Config.from_dict(over)
    env_cfg = ConeObstacleUEEnvConfig(**env_over)
    out = Path(a.output_root) / a.run_tag
    print(f"[jax] backend={jax.default_backend()} | run {out} | total_steps={cfg.total_steps} num_envs={cfg.num_envs} "
          f"batch={cfg.batch_size} update_every={cfg.update_every} | backup {cfg.backup} {cfg.backup_path} | "
          + (f"rows_keep={cfg.rows_keep}" if cfg.use_projection else "NO CIL (vanilla tracker)")
          + (f" | warm start {cfg.warm_start_weights}" if cfg.warm_start_weights else "")
          + f" | disturbance {env_cfg.disturbance_mode if env_cfg.disturbance else 'off'}", flush=True)
    res = run_cone_obstacle_ue_ps2(cfg, task, env_cfg, out, task_path=a.task)
    s = res["summary"]
    print(json.dumps({"best_step": s["best_step"], "best_eval": s["best_eval"], "wall_time_min": s["wall_time_sec"] / 60},
                     indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()
