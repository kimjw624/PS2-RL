#!/usr/bin/env python
"""Phase II stage 1 for the strike task: train the vanilla striker (SAC, no safety filter).

    python scripts/train_phase2_strike.py --run_tag striker_rs0p1_s0                 # task docker_batch/configs/strike.json
    python scripts/train_phase2_strike.py --run_tag striker_rs1_s0 --reward_scale 1.0

Task file blocks used: "strike" (vehicle, floor, chasers), "env" (arena, target, reward), "tracker" (SAC settings);
command-line flags override single values (--total_steps, --seed, --reward_scale, --env_w_progress, ...).
Output (``outputs/strike_tracker/<run_tag>/``): best_weights.pkl (highest eval return), final_weights.pkl,
history.json, summary.json, configs.json. The scene pool is cached in ``outputs/strike_tracker/``.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402

from ps2rl.envs.strike_env import StrikeEnvConfig  # noqa: E402
from ps2rl.envs.strike_scene import StrikeConfig, build_model, load_strike_task  # noqa: E402
from ps2rl.phase2_ps2.strike_tracker_trainer import StrikeTrackerConfig, run_strike_tracker  # noqa: E402


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
    p.add_argument("--task", default="docker_batch/configs/strike.json")
    p.add_argument("--output_root", default="outputs/strike_tracker")
    p.add_argument("--run_tag", default="striker_s0")
    p.add_argument("--require_gpu", action="store_true")
    _add(p, StrikeTrackerConfig)
    _add(p, StrikeEnvConfig, "env_")
    a = p.parse_args(argv)
    backend = jax.default_backend()
    print(f"[jax] backend={backend} devices={jax.devices()}", flush=True)
    if a.require_gpu and backend != "gpu":
        raise SystemExit(f"JAX is not running on the GPU (backend={backend})")
    if backend != "gpu" and os.environ.get("PS2RL_ALLOW_CPU", "0") != "1":
        raise SystemExit("refusing to train on the CPU; set PS2RL_ALLOW_CPU=1 to run anyway")
    task = load_strike_task(a.task)
    m = build_model(StrikeConfig.from_dict(task.get("strike", {})))
    ec = StrikeEnvConfig.from_dict({**task.get("env", {}), **_collect(a, StrikeEnvConfig, "env_")})
    cfg = StrikeTrackerConfig.from_dict({**task.get("tracker", {}), **_collect(a, StrikeTrackerConfig)})
    out = Path(a.output_root) / a.run_tag
    print(f"[run] {out} total_steps={cfg.total_steps} reward_scale={cfg.reward_scale} seed={cfg.seed}", flush=True)
    s = run_strike_tracker(cfg, m, ec, out, task=task, pool_cache=a.output_root)
    b = s.get("best_eval") or {}
    print(f"[done] best step {s['best_step']}: strike {b.get('strike_rate', float('nan')):.3f} crash {b.get('crash_rate', float('nan')):.3f} "
          f"return {b.get('return_mean', float('nan')):.1f}; {s['wall_time_sec'] / 3600:.2f} h; {out}", flush=True)


if __name__ == "__main__":
    main()
