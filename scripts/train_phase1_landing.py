"""Train the Phase-I safe-arrival policy for the approach-cone landing task.

    # full run (GPU), SAC backbone (default)
    python scripts/train_phase1_landing.py --backbone sac --seed 0

    # the paper's TD3 backbone on the same sets and samples, for comparison
    python scripts/train_phase1_landing.py --backbone td3 --seed 0

    # CPU smoke test
    JAX_PLATFORMS=cpu python scripts/train_phase1_landing.py --smoke_test

    # the shipped floor-aware backup (45 deg cone, r0 = 0.3 m; see docker_batch/README_landing_floor.md)
    python scripts/train_phase1_landing.py --config_json docker_batch/configs/landing_cone45_r0p3.json --backbone td3 \
        --floor_constraint true --region_edge_floor_prob 0.3 --recovery_rate_cone 10 --recovery_rate_floor 10

``--config_json`` takes {"landing": {...}, "region": {...}, "sa": {...}} overrides; explicit
command-line flags win over it.

Before training, the base-set level is checked against Proposition 1 of the landing
note (input feasibility, chart, cone and ground containment, and - unless
``--skip_lyap_check`` - the adversarial one-step Lyapunov decrease). Training aborts if
c_B is not certified. ``scripts/certify_landing_base_set.py`` gives the full report.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import pickle
import sys
import time

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ps2rl.backup_policy.backup_policy import save_learned_backup_policy
from ps2rl.envs.quadrotor_landing_config import QuadrotorLandingConfig
from ps2rl.phase1_sa.landing_design_region import LandingDesignRegionConfig
from ps2rl.phase1_sa.quadrotor_landing_sa_env import build_landing_sets, landing_action_box, landing_step_fn
from ps2rl.phase1_sa.quadrotor_landing_sa_trainer import LandingSAConfig, run_landing_sa_training
from ps2rl.sets import landing_certificate as lc

_SMOKE = {
    "total_steps": 4096,
    "num_envs": 16,
    "steps_per_jit": 32,
    "start_steps": 512,
    "update_after": 256,
    "batch_size": 64,
    "replay_size": 20_000,
    "eval_every": 2048,
    "log_every": 1024,
    "hidden_size": 64,
    "curriculum_window_episodes": 20,
    "curriculum_min_episodes": 20,
}
_SMOKE_REGION = {"heldout_general": 64, "heldout_edge": 64, "heldout_shell": 32}


def _add_dataclass_args(parser: argparse.ArgumentParser, cls, prefix: str = "", skip: tuple[str, ...] = ()) -> None:
    for f in fields(cls):
        if f.name in skip:
            continue
        default = f.default
        flag = f"--{prefix}{f.name}"
        if isinstance(default, bool):
            parser.add_argument(flag, type=lambda s: str(s).lower() in ("1", "true", "yes"), default=None)
        elif isinstance(default, (int, float, str)):
            parser.add_argument(flag, type=type(default), default=None)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", choices=("sac", "td3"), default="sac")
    p.add_argument("--output_dir", type=str, default="outputs/landing_phase1")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--skip_lyap_check", action="store_true")
    p.add_argument("--require_gpu", action="store_true", help="Abort unless JAX runs on a GPU backend.")
    p.add_argument("--config_json", type=str, default="",
                   help="JSON with 'landing'/'region'/'sa' override blocks (e.g. docker_batch/configs/landing_cone45_r0p3.json)")
    _add_dataclass_args(p, LandingSAConfig, skip=("sa_backbone",))
    _add_dataclass_args(p, QuadrotorLandingConfig)
    _add_dataclass_args(p, LandingDesignRegionConfig, prefix="region_")
    return p.parse_args(argv)


def _collect(args: argparse.Namespace, cls, prefix: str = "") -> dict:
    out = {}
    for f in fields(cls):
        v = getattr(args, f"{prefix}{f.name}", None)
        if v is not None:
            out[f.name] = v
    return out


def check_proposition_1(cfg: QuadrotorLandingConfig, *, lyap: bool) -> dict:
    """Raise if c_B violates the level bounds of the landing note (Definition 2)."""
    cone, base_set = build_landing_sets(cfg)
    ctrl = base_set.controller
    p_inv = np.linalg.inv(ctrl.p_matrix_f64())
    c_b = float(cfg.base_set_c)
    bounds = {
        "c_U": float(ctrl.max_certified_level),
        "c_chart": lc.c_chart_bound(p_inv),
        "c_cone": lc.c_cone_exact(p_inv, cone.cone, z_des=cfg.z_des, c_hi=4.0 * c_b),
        "c_ground": lc.c_ground_bound(p_inv, z_des=cfg.z_des, z_clear=cfg.z_clear),
    }
    if float(cfg.recovery_rate_floor) > 0.0:
        bounds["c_recovery_floor"] = lc.c_recovery_floor(p_inv, z_des=cfg.z_des, kappa=float(cfg.recovery_rate_floor))
    if float(cfg.recovery_rate_cone) > 0.0:
        bounds["c_recovery_cone"] = lc.c_recovery_cone(p_inv, cone.cone, z_des=cfg.z_des,
                                                       kappa=float(cfg.recovery_rate_cone), c_hi=4.0 * c_b)
    if lyap:
        worst, _ = lc.lyapunov_worst_adversarial(ctrl, landing_step_fn(cfg), level=c_b, n_starts=2048, iters=200)
        bounds["lyap_ratio_at_c_B"] = worst
    bad = [k for k in bounds if k.startswith("c_") and c_b > bounds[k] + 1e-9]
    if lyap and bounds["lyap_ratio_at_c_B"] >= 1.0:
        bad.append("Lyapunov")
    msg = ", ".join(f"{k}={v:.3f}" for k, v in bounds.items())
    if bad:
        raise SystemExit(f"base_set_c={c_b} is not certified ({', '.join(bad)} violated): {msg}")
    print(f"[certificate] c_B={c_b} at z_des={cfg.z_des} (safe set: cone{' + floor' if cfg.floor_constraint else ''}"
          f"{f', recovery rates cone {cfg.recovery_rate_cone:g} / floor {cfg.recovery_rate_floor:g}' if cone.recovery_enabled else ''}): "
          f"{msg} -> OK")
    return bounds


def main(argv=None) -> None:
    args = parse_args(argv)
    backend = jax.default_backend()
    print(f"[jax] backend={backend} devices={jax.devices()}")
    if args.require_gpu and backend != "gpu":
        raise SystemExit(
            "JAX is not running on the GPU (backend=%s). Inside the container, check `nvidia-smi` and that the "
            "venv has jax[cuda12]==0.6.2 (bash docker_batch/setup_train_venv.sh)." % backend
        )
    file_cfg = json.loads(Path(args.config_json).read_text(encoding="utf-8")) if args.config_json else {}
    unknown = set(file_cfg) - {"landing", "region", "sa", "provenance"}
    if unknown:
        raise SystemExit(f"{args.config_json}: unknown blocks {sorted(unknown)}")
    sa_over = {**file_cfg.get("sa", {}), **_collect(args, LandingSAConfig)}
    region_over = {**file_cfg.get("region", {}), **_collect(args, LandingDesignRegionConfig, prefix="region_")}
    if args.smoke_test:
        sa_over = {**_SMOKE, **sa_over}
        region_over = {**_SMOKE_REGION, **region_over}
    ra_cfg = LandingSAConfig.from_dict({**sa_over, "sa_backbone": args.backbone})
    cfg = QuadrotorLandingConfig().replace(**{**file_cfg.get("landing", {}), **_collect(args, QuadrotorLandingConfig)})
    region_cfg = LandingDesignRegionConfig.from_dict(region_over)

    bounds = check_proposition_1(cfg, lyap=not (args.skip_lyap_check or args.smoke_test))

    run_name = args.run_name or f"landing_p1_{args.backbone}_seed{ra_cfg.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {run_dir}  backbone={args.backbone}  total_steps={ra_cfg.total_steps}")

    result = run_landing_sa_training(ra_cfg, cfg, region_cfg, output_dir=str(run_dir))

    with open(run_dir / "final_weights.pkl", "wb") as f:
        pickle.dump(result["final_state"], f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(run_dir / "best_weights.pkl", "wb") as f:
        pickle.dump(result["best_state"], f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(result["history"], f)
    with open(run_dir / "certificate.json", "w", encoding="utf-8") as f:
        json.dump(bounds, f, indent=2)

    scale, low, high = landing_action_box(cfg)
    save_learned_backup_policy(
        run_dir / "landing_backup_policy_actor.pkl",
        actor_params=result["best_state"]["actor_params"],
        actor_cfg=result["actor_cfg"],
        action_scale=np.asarray(scale),
        action_low=np.asarray(low),
        action_high=np.asarray(high),
        metadata={
            "task": "approach_cone_landing",
            "training_objective": "discounted_safe_arrival",
            "sa_backbone": args.backbone,
            "deployed_action": "tanh(mean) (deterministic)",
            "observation_feature_mode": "raw_10d_physical_state (pad at pad_x/pad_y/pad_z)",
            "beta": float(ra_cfg.beta),
            "landing_config": cfg.as_dict(),
            "design_region": region_cfg.as_dict(),
            "certificate": bounds,
            "best_eval_step": int(result["summary"]["best_eval_step"]),
            "config_json": str(args.config_json) if args.config_json else None,
            "config_provenance": file_cfg.get("provenance"),
        },
    )
    s = result["summary"]
    print(
        f"[done] untrained mu_w={s['untrained_val']['mu_weighted']:.3f} -> best val mu_w={s['best_val']['mu_weighted']:.3f} "
        f"(step {s['best_eval_step']}), test at best mu_w={s['test_at_best']['mu_weighted']:.3f}; "
        f"{s['wall_time_sec'] / 60.0:.1f} min on {s['jax_backend']}; outputs in {run_dir}"
    )


if __name__ == "__main__":
    main()
