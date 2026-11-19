#!/usr/bin/env python
"""Train the slow-reference tracker (forward + loop) for PX4 SITL and hardware.

    # GPU (default settings = the cone-obstacle stage-1 tracker recipe, 3M steps)
    python scripts/train_slow_track.py --run_tag slow_loop_s0 --seed 0
    # CPU smoke test (a few minutes)
    JAX_PLATFORMS=cpu python scripts/train_slow_track.py --smoke_test --run_tag smoke

Every SlowTrackSACConfig field is a flag (``--total_steps``, ``--reward_scale``, ...); every
SlowTrackEnvConfig field is a flag with the prefix ``--env_`` (``--env_reference_path``, ``--env_a_max``,
``--env_thrust_gain_min``, ...). Outputs: ``<output_root>/<run_tag>/`` with ``configs.json``,
``best_weights.pkl`` (lowest robust evaluation score), ``final_weights.pkl``, ``history.json``,
``summary.json``. ``touch <run dir>/STOP`` ends a run early (it finishes normally).
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ps2rl.envs.slow_track_env import SlowTrackEnvConfig  # noqa: E402
from ps2rl.phase2_ps2.slow_track_trainer import SlowTrackSACConfig, run_slow_track_sac  # noqa: E402

_SMOKE = {"total_steps": 40_960, "num_envs": 16, "steps_per_jit": 16, "start_steps": 2048, "update_after": 1024,
          "batch_size": 64, "replay_size": 50_000, "eval_every": 20_480, "eval_episodes": 2, "log_every": 4096,
          "hidden_size": 64, "update_every": 4}


def _add(p, cls, prefix=""):
    for f in fields(cls):
        flag = f"--{prefix}{f.name}"
        if isinstance(f.default, bool):
            p.add_argument(flag, type=lambda s: str(s).lower() in ("1", "true", "yes", "on"), default=None)
        elif isinstance(f.default, (int, float, str)):
            p.add_argument(flag, type=type(f.default), default=None)


def _collect(a, cls, prefix=""):
    return {f.name: getattr(a, f"{prefix}{f.name}") for f in fields(cls)
            if getattr(a, f"{prefix}{f.name}", None) is not None}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_root", default="outputs/slow_track")
    p.add_argument("--run_tag", default="slow_loop_s0")
    p.add_argument("--smoke_test", action="store_true")
    _add(p, SlowTrackSACConfig)
    _add(p, SlowTrackEnvConfig, prefix="env_")
    a = p.parse_args(argv)
    over = _collect(a, SlowTrackSACConfig)
    if a.smoke_test:
        over = {**_SMOKE, **over}
    cfg = SlowTrackSACConfig(**over)
    env_cfg = SlowTrackEnvConfig(**_collect(a, SlowTrackEnvConfig, prefix="env_"))
    out = Path(a.output_root) / a.run_tag
    print(f"[slow track] run {out}\n  sac {json.dumps(over)}\n  env box {env_cfg.action_low} .. {env_cfg.action_high}",
          flush=True)
    summary = run_slow_track_sac(cfg, env_cfg, out)
    print(json.dumps({k: summary[k] for k in ("total_steps", "wall_time_sec", "best_step", "best_score")}, indent=2))


if __name__ == "__main__":
    main()
