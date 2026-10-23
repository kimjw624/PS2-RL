#!/usr/bin/env python
"""Train the Phase-I safe-arrival backup for the strike task (tall obstacles, chaser balls, hover-at-current-point base set).

    python scripts/train_phase1_strike.py --run_name td3_seed0                     # task docker_batch/configs/strike.json
    python scripts/train_phase1_strike.py --run_name td3_seed1 --seed 1
    PS2RL_ALLOW_CPU=1 JAX_PLATFORMS=cpu python scripts/train_phase1_strike.py --smoke_test --run_name smoke

The task file holds {"strike": {...}, "region": {...}, "sa": {...}}; command-line flags override single values
(``--total_steps 2000000``, ``--region_blocked_std 1.0``, ``--strike_T 1.0``, ...). Before training the base set is
certified (input box, attitude chart, sampled one-step decrease with the worst moderate wind); training stops if c_B
is above a bound.

Output (``outputs/strike_phase1/<run_name>/``): ``strike_backup_policy_actor.pkl`` (best validation checkpoint),
best / final weights, ``summary.json`` (learned backup and the LQR backup per region, val and test),
``configs.json``, ``history.json``, ``certificate.json``.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import numpy as np  # noqa: E402

from ps2rl.backup_policy.backup_policy import save_learned_backup_policy  # noqa: E402
from ps2rl.envs.strike_scene import StrikeConfig, build_model, certify_base_set, load_strike_task  # noqa: E402
from ps2rl.phase1_sa.strike_sa import REGION_NAMES, StrikeRegionConfig, StrikeSAConfig, run_strike_sa_training  # noqa: E402

_SMOKE_SA = {"total_steps": 4096, "num_envs": 16, "steps_per_jit": 32, "start_steps": 512, "update_after": 256,
             "batch_size": 64, "replay_size": 20_000, "eval_every": 2048, "log_every": 1024, "hidden_size": 64,
             "curriculum_window_episodes": 20, "curriculum_min_episodes": 20}
_SMOKE_REGION = {"pool_per_region": 512, "heldout_val": 64, "heldout_test": 64}


def _add_dataclass_args(p: argparse.ArgumentParser, cls, prefix: str) -> None:
    for f in fields(cls):
        d = f.default
        if isinstance(d, bool):
            p.add_argument(f"--{prefix}{f.name}", type=lambda s: str(s).lower() in ("1", "true", "yes"), default=None)
        elif isinstance(d, (int, float, str)):
            p.add_argument(f"--{prefix}{f.name}", type=type(d), default=None)


def _collect(args, cls, prefix: str) -> dict:
    return {f.name: getattr(args, f"{prefix}{f.name}") for f in fields(cls)
            if getattr(args, f"{prefix}{f.name}", None) is not None}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="docker_batch/configs/strike.json")
    p.add_argument("--output_dir", default="outputs/strike_phase1")
    p.add_argument("--run_name", default="")
    p.add_argument("--actor_init", choices=("lqr_bc", "flee_bc", "base_bc", "random"), default=None,
                   help="actor start: behaviour-cloned hover LQR, flee-then-hover heuristic (moving chasers), du = 0 on the "
                        "base action, or random (task file 'actor_init', default lqr_bc)")
    p.add_argument("--bc_steps", type=int, default=None,
                   help="behaviour-cloning steps of the actor initialization (task file 'bc_steps', default 3000)")
    p.add_argument("--actor_start_update", type=int, default=None,
                   help="critic-only updates before the actor moves (task file 'actor_start_update', default 0)")
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--require_gpu", action="store_true")
    _add_dataclass_args(p, StrikeSAConfig, "")
    _add_dataclass_args(p, StrikeConfig, "strike_")
    _add_dataclass_args(p, StrikeRegionConfig, "region_")
    return p.parse_args(argv)


def main(argv=None) -> None:
    a = parse_args(argv)
    backend = jax.default_backend()
    print(f"[jax] backend={backend} devices={jax.devices()}", flush=True)
    if a.require_gpu and backend != "gpu":
        raise SystemExit(f"JAX is not running on the GPU (backend={backend})")
    if backend != "gpu" and not a.smoke_test and os.environ.get("PS2RL_ALLOW_CPU", "0") != "1":
        raise SystemExit("refusing a full run on the CPU; set PS2RL_ALLOW_CPU=1 to run anyway")
    task = load_strike_task(a.task)
    sa_block = dict(task.get("sa", {}))
    actor_init = a.actor_init or sa_block.pop("actor_init", "lqr_bc")
    actor_start = a.actor_start_update if a.actor_start_update is not None else int(sa_block.pop("actor_start_update", 0))
    bc_steps = a.bc_steps if a.bc_steps is not None else int(sa_block.pop("bc_steps", 3000))
    for k in ("actor_init", "actor_start_update", "bc_steps"):
        sa_block.pop(k, None)
    sa_over = {**sa_block, **_collect(a, StrikeSAConfig, "")}
    region_over = {**task.get("region", {}), **_collect(a, StrikeRegionConfig, "region_")}
    if a.smoke_test:
        sa_over, region_over = {**sa_over, **_SMOKE_SA}, {**region_over, **_SMOKE_REGION}
    ra_cfg = StrikeSAConfig.from_dict(sa_over)
    cfg = StrikeConfig.from_dict({**task.get("strike", {}), **_collect(a, StrikeConfig, "strike_")})
    rc = StrikeRegionConfig.from_dict(region_over)
    m = build_model(cfg)

    cert = certify_base_set(cfg, m.lqr)
    print(f"[base set] c_B {cert['c_B']:.1f} | c_U {cert['c_U']:.1f} c_chart {cert['c_chart']:.1f} c_Lyap {cert['c_lyap']:.1f} "
          f"(wind {cert['wind_accel_max']:.2f}: {cert['c_lyap_wind']:.1f}) | V+/V at c_B {cert['ratio_at_c_B']:.3f} "
          f"(wind {cert['ratio_at_c_B_wind']:.3f}) | hover ellipsoid {cert['rho_xy']:.2f} m wide, {cert['z_ext']:.2f} m tall",
          flush=True)
    if not cert["ok"]:
        raise SystemExit("c_B is above a certificate bound; lower strike.c_B")
    chasers = (f"moving chasers, escape margin h_c (a_e {cfg.chaser_a_evade}, delay {cfg.chaser_delay}), base-set hover "
               f"margin {cfg.chaser_base_margin} m" if cfg.moving_chasers else f"chaser ball radius {cfg.ball_radius:.2f} m")
    print(f"[task] T {cfg.T} s (N {cfg.num_steps}), {chasers}, walls {cfg.arena_walls}, floor {cfg.z_floor} m, "
          f"obs_dim {cfg.obs_dim}; regions {dict(zip(REGION_NAMES, rc.mix_high))}", flush=True)

    run = a.run_name or f"strike_p1_{ra_cfg.sa_backbone}_seed{ra_cfg.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(a.output_dir) / run
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {run_dir} backbone={ra_cfg.sa_backbone} total_steps={ra_cfg.total_steps} actor_init={actor_init} "
          f"actor_start_update={actor_start}", flush=True)
    with open(run_dir / "certificate.json", "w", encoding="utf-8") as f:
        json.dump(cert, f, indent=2)

    res = run_strike_sa_training(ra_cfg, m, rc, output_dir=str(run_dir), actor_init=actor_init, actor_start_update=actor_start,
                                 bc_steps=bc_steps)
    for name, key in (("final_weights.pkl", "final_state"), ("best_weights.pkl", "best_state")):
        with open(run_dir / name, "wb") as f:
            pickle.dump(res[key], f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(res["history"], f)
    save_learned_backup_policy(
        run_dir / "strike_backup_policy_actor.pkl", actor_params=res["best_state"]["actor_params"],
        actor_cfg=res["actor_cfg"], action_scale=np.asarray(m.scale), action_low=np.asarray(m.d_low),
        action_high=np.asarray(m.d_high),
        metadata={"task": "strike_phase1", "training_objective": "discounted_safe_arrival",
                  "sa_backbone": ra_cfg.sa_backbone,
                  "deployed_action": {"residual_lqr": "u = u_LQR(x) + tanh-mean output, clipped to the plant box",
                                      "residual_flee": "u = u_flee(x, scene) + tanh-mean output, clipped to the plant box "
                                                       "(strike_scene.flee_action)",
                                      "hover": "u = u_hover + tanh-mean output, clipped to the plant box"}[cfg.backup_action],
                  "backup_action": cfg.backup_action, "u_hover": np.asarray(m.u_hover).tolist(),
                  "observation": "ps2rl.envs.strike_scene.make_obs (scene sorted at the rollout start)",
                  "strike_config": cfg.as_dict(), "region": rc.as_dict(), "certificate": cert,
                  "best_eval_step": int(res["summary"]["best_eval_step"]), "task_file": str(a.task)})
    s = res["summary"]
    print("[done] test, learned vs LQR: " + " | ".join(
        f"{n} {s['test_at_best'][n]['m_hat']:.3f} vs {s['lqr_test'][n]['m_hat']:.3f}" for n in REGION_NAMES)
        + f" | mu_w {s['test_at_best']['mu_weighted']:.3f} vs {s['lqr_test']['mu_weighted']:.3f} "
        f"(best step {s['best_eval_step']}); {s['wall_time_sec'] / 60:.1f} min on {s['jax_backend']}; {run_dir}", flush=True)


if __name__ == "__main__":
    main()
