"""Train the disturbance-aware (UE-bCBF) Phase-I safe-arrival backup for landing.

    # full run, shipped geometry (45 deg cone, r0 0.3 m, floor, gentle-recovery envelope 10)
    python scripts/train_phase1_landing_ue.py --config_json docker_batch/configs/landing_cone45_r0p3.json \
        --floor_constraint true --region_edge_floor_prob 0.3 --recovery_rate_cone 10 --recovery_rate_floor 10 \
        --seed 0 --total_steps 5000000 --run_name ue_floor_rec10_td3_seed0

    # disturbance bounds (defaults = the quadrotor UE-bCBF experiments: A 0.5 m/s^2, 0.05 Hz, observer 20)
    ... --ue_delta_d 0.5 --ue_frequency_hz 0.05 --ue_observer_lambda 20 --ue_e_bar 0.02 --ue_tube_scale 1.2

    # CPU smoke test
    JAX_PLATFORMS=cpu python scripts/train_phase1_landing_ue.py --smoke_test

Before training: the nominal base-set certificate of the landing note (as in
``train_phase1_landing.py``) and a *robust* one-step Lyapunov check - the LQR must
decrease V on the boundary of B under the worst disturbance |d| <= delta_d, so that B is
robustly invariant after hand-off. Training aborts otherwise.

Output (``outputs/landing_phase1_ue/<run_name>/``): ``landing_backup_policy_actor.pkl``
(actor input = (x, d_hat), 13-D; metadata carries the landing config *and* the UE config,
both checked by the Phase-II UE filter), summary/configs/history/certificate json.
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
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ps2rl.backup_policy.backup_policy import save_learned_backup_policy
from ps2rl.envs.quadrotor_landing_config import QuadrotorLandingConfig
from ps2rl.phase1_sa.landing_design_region import LandingDesignRegionConfig
from ps2rl.phase1_sa.quadrotor_landing_sa_env import build_landing_sets, landing_action_box
from ps2rl.phase1_sa.quadrotor_landing_ue_sa_env import landing_ue_step_fn
from ps2rl.phase1_sa.quadrotor_landing_ue_sa_trainer import LandingUESAConfig, run_landing_ue_sa_training
from ps2rl.sets import landing_certificate as lcert
from ps2rl.uncertainty.landing_ue_tube import LandingUEConfig, nominal_tube, tube_constants
from train_phase1_landing import _add_dataclass_args, _collect, check_proposition_1  # noqa: E402

_SMOKE = {
    "total_steps": 4096, "num_envs": 16, "steps_per_jit": 32, "start_steps": 512, "update_after": 256,
    "batch_size": 64, "replay_size": 20_000, "eval_every": 2048, "log_every": 1024, "hidden_size": 64,
    "curriculum_window_episodes": 20, "curriculum_min_episodes": 20,
}
_SMOKE_REGION = {"heldout_general": 64, "heldout_edge": 64, "heldout_shell": 32}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", type=str, default="outputs/landing_phase1_ue")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--skip_lyap_check", action="store_true")
    p.add_argument("--config_json", type=str, default="")
    _add_dataclass_args(p, LandingUESAConfig)
    _add_dataclass_args(p, QuadrotorLandingConfig)
    _add_dataclass_args(p, LandingDesignRegionConfig, prefix="region_")
    _add_dataclass_args(p, LandingUEConfig, prefix="ue_")
    return p.parse_args(argv)


def robust_lyapunov_check(cfg: QuadrotorLandingConfig, ue: LandingUEConfig) -> dict:
    """Worst V(x+)/V(x) on dB under the worst disturbance |d| <= delta_d (adversarial ascent)."""
    _, base_set = build_landing_sets(cfg)
    ctrl = base_set.controller
    plant = landing_ue_step_fn(cfg)
    p = jnp.asarray(ctrl.p_matrix_f64(), dtype=jnp.float32)
    dd = float(ue.delta_d)

    def step_worst(x, u):
        xn = plant(x, u, jnp.zeros(3, x.dtype))
        g = (p @ ctrl.error_state(xn))[3:6]
        d = dd * g / jnp.maximum(jnp.linalg.norm(g), 1e-9)
        return plant(x, u, d)

    worst, _ = lcert.lyapunov_worst_adversarial(ctrl, step_worst, level=float(cfg.base_set_c), n_starts=2048, iters=200)
    out = {"robust_lyap_ratio_at_c_B": float(worst), "delta_d": dd}
    if worst >= 1.0:
        raise SystemExit(f"B is not robustly invariant under |d| <= {dd}: worst V+/V = {worst:.4f} >= 1")
    print(f"[certificate] robust one-step Lyapunov on dB with |d| <= {dd}: worst V+/V = {worst:.4f} < 1 -> OK")
    return out


def main(argv=None) -> None:
    args = parse_args(argv)
    print(f"[jax] backend={jax.default_backend()} devices={jax.devices()}")
    file_cfg = json.loads(Path(args.config_json).read_text(encoding="utf-8")) if args.config_json else {}
    sa_over = {**file_cfg.get("sa", {}), **_collect(args, LandingUESAConfig)}
    region_over = {**file_cfg.get("region", {}), **_collect(args, LandingDesignRegionConfig, prefix="region_")}
    ue_over = {**file_cfg.get("ue", {}), **_collect(args, LandingUEConfig, prefix="ue_")}
    if args.smoke_test:
        sa_over = {**_SMOKE, **sa_over}
        region_over = {**_SMOKE_REGION, **region_over}
    ra_cfg = LandingUESAConfig.from_dict(sa_over)
    cfg = QuadrotorLandingConfig().replace(**{**file_cfg.get("landing", {}), **_collect(args, QuadrotorLandingConfig)})
    region_cfg = LandingDesignRegionConfig.from_dict(region_over)
    ue = LandingUEConfig.from_dict(ue_over)

    bounds = check_proposition_1(cfg, lyap=not (args.skip_lyap_check or args.smoke_test))
    if not (args.skip_lyap_check or args.smoke_test):
        bounds.update(robust_lyapunov_check(cfg, ue))
    tc = tube_constants(cfg)
    s_lqr = nominal_tube(int(cfg.num_steps), 0.9814, ue, tc, float(cfg.dt))
    print(f"[ue] delta_d={ue.delta_d} delta_v={ue.delta_v:.4f} e_bar={ue.e_bar} (observer warm-up {ue.warmup_time():.3f} s) "
          f"tube_scale={ue.tube_scale} | gamma={tc.gamma:.3f} L_cone={tc.l_cone:.3f} L_floor={tc.l_floor:.3f} | "
          f"LQR-only tube s(T)={s_lqr[-1]:.3f} -> base margin {2*np.sqrt(tc.c_b)*ue.tube_scale*s_lqr[-1] + (ue.tube_scale*s_lqr[-1])**2:.2f} of c_B={tc.c_b}")

    run_name = args.run_name or f"landing_p1_ue_td3_seed{ra_cfg.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {run_dir} total_steps={ra_cfg.total_steps} hidden={ra_cfg.hidden_size} num_envs={ra_cfg.num_envs}", flush=True)

    result = run_landing_ue_sa_training(ra_cfg, cfg, region_cfg, ue, output_dir=str(run_dir))

    for name in ("final_state", "best_state"):
        with open(run_dir / f"{name.replace('_state', '')}_weights.pkl", "wb") as f:
            pickle.dump(result[name], f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(result["history"], f)
    with open(run_dir / "certificate.json", "w", encoding="utf-8") as f:
        json.dump(bounds, f, indent=2)
    scale, low, high = landing_action_box(cfg)
    save_learned_backup_policy(
        run_dir / "landing_backup_policy_actor.pkl",
        actor_params=result["best_state"]["actor_params"],
        actor_cfg=result["actor_cfg"],
        action_scale=np.asarray(scale), action_low=np.asarray(low), action_high=np.asarray(high),
        metadata={
            "task": "approach_cone_landing_under_disturbance",
            "training_objective": "discounted_safe_arrival_ue_tightened",
            "sa_backbone": "td3",
            "deployed_action": "tanh(mean) (deterministic)",
            "observation_feature_mode": "x(10, pad frame) + d_hat(3, world-frame acceleration estimate)",
            "beta": float(ra_cfg.beta),
            "landing_config": cfg.as_dict(),
            "ue_config": ue.as_dict(),
            "tube_constants": result["tube"].as_dict(),
            "design_region": region_cfg.as_dict(),
            "certificate": bounds,
            "contraction": {"target": ra_cfg.contraction_target, "weight": ra_cfg.contraction_weight},
            "best_eval_step": int(result["summary"]["best_eval_step"]),
        },
    )
    s = result["summary"]
    print(f"[done] UE mu_w {s['untrained_val']['mu_weighted']:.3f} -> best val {s['best_val']['mu_weighted']:.3f} "
          f"(step {s['best_eval_step']}), test at best {s['test_at_best']['mu_weighted']:.3f} "
          f"(nominal {s['test_at_best']['nominal_mu_weighted']:.3f}); {s['wall_time_sec']/60:.1f} min; {run_dir}", flush=True)


if __name__ == "__main__":
    main()
