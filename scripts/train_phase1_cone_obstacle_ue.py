#!/usr/bin/env python
"""Train the Phase-I safe-arrival backup for the cone-obstacle task (outside of an apex-down funnel).

    # on the tube-tightened ("shrunk") sets, no disturbance in training
    python scripts/train_phase1_cone_obstacle_ue.py --tighten true  --run_name sa_shrunk_seed0
    # on the plain sets (plain BCBF Phase I)
    python scripts/train_phase1_cone_obstacle_ue.py --tighten false --run_name sa_plain_seed0
    # CPU smoke test
    JAX_PLATFORMS=cpu python scripts/train_phase1_cone_obstacle_ue.py --smoke_test

Task (geometry, floor, base set, UE bounds) from ``--task`` (default
docker_batch/configs/cone_obstacle_ue.json). Before training the base set is checked: input box
(c_U), quaternion chart, adversarial one-step Lyapunov decrease of the nominal Euler step, and that
the ellipsoid lies outside the funnel and above the floor. Training aborts if c_B exceeds a bound.

Output (``outputs/cone_obstacle_phase1/<run_name>/``): ``cone_obstacle_backup_policy_actor.pkl``
(actor input (x, d_hat), trained and used with d_hat = 0), best/final weights, summary, configs,
history, certificate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cone_obstacle_common import load_task, task_config  # noqa: E402
from ps2rl.backup_policy.backup_policy import save_learned_backup_policy  # noqa: E402
from ps2rl.envs.quadrotor_env import quadrotor_step_euler  # noqa: E402
from ps2rl.phase1_sa.cone_obstacle_design_region import ConeObstacleDesignRegionConfig  # noqa: E402
from ps2rl.phase1_sa.cone_obstacle_ue_sa_trainer import ConeObstacleUESAConfig, run_cone_obstacle_ue_sa_training  # noqa: E402
from ps2rl.phase1_sa.quadrotor_landing_sa_env import landing_action_box  # noqa: E402
from ps2rl.sets import landing_certificate as lcert  # noqa: E402
from ps2rl.sets.quadrotor_cone_obstacle_sets import obstacle_sets, position_metric  # noqa: E402
from ps2rl.uncertainty.cone_obstacle_ue_tube import UEConfig, nominal_tube, tube_constants, tube_margins  # noqa: E402
from train_phase1_landing import _add_dataclass_args, _collect  # noqa: E402

_SMOKE = {"total_steps": 4096, "num_envs": 16, "steps_per_jit": 32, "start_steps": 512, "update_after": 256,
          "batch_size": 64, "replay_size": 20_000, "eval_every": 2048, "log_every": 1024, "hidden_size": 64,
          "curriculum_window_episodes": 20, "curriculum_min_episodes": 20}
_SMOKE_REGION = {"heldout_general": 64, "heldout_edge": 64, "heldout_shell": 32}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="docker_batch/configs/cone_obstacle_ue.json")
    p.add_argument("--tighten", type=lambda s: str(s).lower() in ("1", "true", "yes"), default=True,
                   help="train on the tube-tightened sets (true) or the plain sets (false)")
    p.add_argument("--output_dir", type=str, default="outputs/cone_obstacle_phase1")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--skip_lyap_check", action="store_true")
    p.add_argument("--actor_init", choices=("lqr_bc", "random"), default="lqr_bc")
    _add_dataclass_args(p, ConeObstacleUESAConfig)
    _add_dataclass_args(p, ConeObstacleDesignRegionConfig, prefix="region_")
    return p.parse_args(argv)


def check_base_set(cfg, *, lyap: bool) -> dict:
    """Level bounds of the base set (the landing note's Definition 2, for the ellipsoid beside the funnel)."""
    safe, base = obstacle_sets(cfg)
    ctrl = base.controller
    p_inv = np.linalg.inv(ctrl.p_matrix_f64())
    c_b = float(cfg.base_set_c)
    m_h, m_v, _ = position_metric(ctrl)
    pt = cfg.hover_points[0]
    d = float(safe.m_distance(jnp.asarray(pt, jnp.float32), m_h, m_v))
    out = {"c_B": c_b, "c_U": float(ctrl.max_certified_level), "c_chart": float(lcert.c_chart_bound(p_inv)),
           "c_geometric": float(min(d * d, (pt[2] - cfg.floor_z) ** 2 * m_v)), "hover_point": list(pt)}
    if lyap:
        w = cfg.omega_clip
        w = w if isinstance(w, float) else jnp.asarray(w, jnp.float32)
        step = lambda x, u: quadrotor_step_euler(x, u, cfg.dt, cfg.gravity, cfg.a_cmd_min, cfg.a_cmd_max, w)
        hi = min(out["c_U"], out["c_chart"])
        out["c_Lyap"] = float(lcert.c_lyap_adversarial(ctrl, step, c_hi=hi, iters=12, ascent_iters=200, n_starts=2048))
    bad = [k for k in ("c_U", "c_chart", "c_geometric", "c_Lyap") if k in out and c_b > out[k] + 1e-9]
    out["ok"] = not bad
    print("[base set] " + "  ".join(f"{k} {v:.2f}" for k, v in out.items() if isinstance(v, float)), flush=True)
    if bad:
        raise SystemExit(f"base_set_c = {c_b} exceeds {bad}")
    return out


def main(argv=None) -> None:
    args = parse_args(argv)
    print(f"[jax] backend={jax.default_backend()} devices={jax.devices()}")
    task = load_task(args.task)
    cfg = task_config(task)
    sa_over = _collect(args, ConeObstacleUESAConfig)
    region_over = _collect(args, ConeObstacleDesignRegionConfig, prefix="region_")
    if args.smoke_test:
        sa_over = {**_SMOKE, **sa_over}
        region_over = {**_SMOKE_REGION, **region_over}
    ra_cfg = ConeObstacleUESAConfig.from_dict(sa_over)
    region_cfg = ConeObstacleDesignRegionConfig.from_dict({**task.get("design_region", {}), **region_over})
    ue_eval = UEConfig.from_dict(task.get("ue", {}))
    ue_train = ue_eval.replace(d_hat_max=0.0)                    # Phase I without a disturbance

    cert = check_base_set(cfg, lyap=not (args.skip_lyap_check or args.smoke_test))
    _, base = obstacle_sets(cfg)
    tc = tube_constants(cfg, base)
    s_lqr = nominal_tube(int(cfg.num_steps), 1.0, ue_eval, tc, float(cfg.dt))
    mb = [float(tube_margins(s_lqr[k], ue_eval, tc)[2]) for k in (25, 50, 75, int(cfg.num_steps))]
    mc = float(tube_margins(s_lqr[-1], ue_eval, tc)[0])
    mf = float(tube_margins(s_lqr[-1], ue_eval, tc)[1])
    print(f"[ue] delta_d={ue_eval.delta_d} delta_v={ue_eval.delta_v:.4f} e_bar={ue_eval.e_bar} tube_scale={ue_eval.tube_scale} | "
          f"gamma={tc.gamma:.3f} L_cone={tc.l_cone:.3f} L_floor={tc.l_floor:.3f} | margins at the end of the rollout "
          f"(growth 1, upper bound): funnel {mc:.3f} m, floor {mf:.3f} m, base {mb[3]:.2f} of c_B {cfg.base_set_c} "
          f"(tau 0.5/1/1.5 s: {mb[0]:.2f}/{mb[1]:.2f}/{mb[2]:.2f}) | training {'on the shrunk sets' if args.tighten else 'on the plain sets'}, "
          "no disturbance", flush=True)

    run_name = args.run_name or f"cone_p1_{'shrunk' if args.tighten else 'plain'}_seed{ra_cfg.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {run_dir} total_steps={ra_cfg.total_steps} hidden={ra_cfg.hidden_size} num_envs={ra_cfg.num_envs}", flush=True)

    result = run_cone_obstacle_ue_sa_training(ra_cfg, cfg, region_cfg, ue_train, ue_eval, tighten=bool(args.tighten),
                                              output_dir=str(run_dir), actor_init=args.actor_init)
    for name in ("final_state", "best_state"):
        with open(run_dir / f"{name.replace('_state', '')}_weights.pkl", "wb") as f:
            pickle.dump(jax.device_get(result[name]), f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(result["history"], f)
    with open(run_dir / "certificate.json", "w", encoding="utf-8") as f:
        json.dump(cert, f, indent=2, default=float)
    scale, low, high = landing_action_box(cfg)
    save_learned_backup_policy(
        run_dir / "cone_obstacle_backup_policy_actor.pkl",
        actor_params=jax.device_get(result["best_state"]["actor_params"]), actor_cfg=result["actor_cfg"],
        action_scale=np.asarray(scale), action_low=np.asarray(low), action_high=np.asarray(high),
        metadata={
            "task": "cone_obstacle_outside_funnel", "task_file": args.task,
            "training_objective": result["summary"]["training_objective"], "tighten": bool(args.tighten),
            "sa_backbone": "td3", "deployed_action": "tanh(mean) (deterministic)",
            "observation_feature_mode": "x(10) + d_hat(3); trained and deployed with d_hat = 0 (no disturbance in Phase I)",
            "beta": float(ra_cfg.beta), "cone_obstacle_config": cfg.as_dict(), "ue_config": ue_eval.as_dict(),
            "tube_constants": result["tube"].as_dict(), "design_region": region_cfg.as_dict(),
            "certificate": json.loads(json.dumps(cert, default=float)),
            "best_eval_step": int(result["summary"]["best_eval_step"]),
        },
    )
    s = result["summary"]
    print(f"[done] {'shrunk' if args.tighten else 'plain'}: tight mu_w LQR {s['lqr_test']['tight_mu_weighted']:.3f} -> "
          f"policy {s['test_at_best']['tight_mu_weighted']:.3f}; nominal LQR {s['lqr_test']['nominal_mu_weighted']:.3f} -> "
          f"policy {s['test_at_best']['nominal_mu_weighted']:.3f} (test, best step {s['best_eval_step']}); "
          f"{s['wall_time_sec'] / 60:.1f} min; {run_dir}", flush=True)


if __name__ == "__main__":
    main()
