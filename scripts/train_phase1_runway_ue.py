"""Train the disturbance-aware (UE-bCBF) Phase-I backup for the runway bird-deterrence task.

    # full run
    python scripts/train_phase1_runway_ue.py --seed 0 --total_steps 5000000 --hidden_size 128 --run_name rwy_ue_seed0

    # CPU smoke test
    JAX_PLATFORMS=cpu python scripts/train_phase1_runway_ue.py --smoke_test

Safe set S: runway keep-out (p_y <= y_edge) and ceiling (p_z <= z_max); base set B: the
retreat-at-altitude LQR ellipsoid (``ps2rl.envs.quadrotor_runway_config``). Before training,
``ps2rl.sets.runway_certificate`` checks that B n S is invariant (v_y < 0 and p_z <= z_max on B,
chart, input box) and that the LQR decreases V on the boundary of B nominally and under the
worst disturbance |d| <= delta_d. Training aborts otherwise.

Output (``outputs/runway_phase1_ue/<run_name>/``): ``runway_backup_policy_actor.pkl`` (actor input
(x, d_hat) with p_x = 0; metadata carries the runway and UE configs), summary/configs/history/
certificate json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
import time

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ps2rl.backup_policy.backup_policy import save_learned_backup_policy  # noqa: E402
from ps2rl.envs.quadrotor_runway_config import QuadrotorRunwayConfig  # noqa: E402
from ps2rl.phase1_sa.quadrotor_landing_sa_env import landing_action_box  # noqa: E402
from ps2rl.phase1_sa.runway_design_region import RunwayDesignRegionConfig  # noqa: E402
from ps2rl.phase1_sa.runway_ue_sa_trainer import RunwayUESAConfig, run_runway_ue_sa_training  # noqa: E402
from ps2rl.sets.runway_certificate import check_runway_base_set  # noqa: E402
from ps2rl.uncertainty.runway_ue_tube import UEConfig, nominal_tube, tube_constants, tube_margins  # noqa: E402
from train_phase1_landing import _add_dataclass_args, _collect  # noqa: E402

_SMOKE = {"total_steps": 4096, "num_envs": 16, "steps_per_jit": 32, "start_steps": 512, "update_after": 256,
          "batch_size": 64, "replay_size": 20_000, "eval_every": 2048, "log_every": 1024, "hidden_size": 64,
          "curriculum_window_episodes": 20, "curriculum_min_episodes": 20}
_SMOKE_REGION = {"heldout_general": 64, "heldout_edge": 64, "heldout_shell": 32}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", type=str, default="outputs/runway_phase1_ue")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--skip_lyap_check", action="store_true")
    p.add_argument("--actor_init", choices=("lqr_bc", "random"), default="lqr_bc",
                   help="start the actor from the retreat LQR (behaviour cloning) or at random")
    _add_dataclass_args(p, RunwayUESAConfig)
    _add_dataclass_args(p, QuadrotorRunwayConfig)
    _add_dataclass_args(p, RunwayDesignRegionConfig, prefix="region_")
    _add_dataclass_args(p, UEConfig, prefix="ue_")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    print(f"[jax] backend={jax.default_backend()} devices={jax.devices()}")
    sa_over = _collect(args, RunwayUESAConfig)
    region_over = _collect(args, RunwayDesignRegionConfig, prefix="region_")
    if args.smoke_test:
        sa_over = {**_SMOKE, **sa_over}
        region_over = {**_SMOKE_REGION, **region_over}
    ra_cfg = RunwayUESAConfig.from_dict(sa_over)
    cfg = QuadrotorRunwayConfig().replace(**_collect(args, QuadrotorRunwayConfig))
    region_cfg = RunwayDesignRegionConfig.from_dict(region_over)
    ue = UEConfig.from_dict(_collect(args, UEConfig, prefix="ue_"))

    cert = check_runway_base_set(cfg, ue, lyap=not (args.skip_lyap_check or args.smoke_test))
    tc = tube_constants(cfg)
    s_lqr = nominal_tube(int(cfg.num_steps), 1.0, ue, tc, float(cfg.dt))
    mb = [float(tube_margins(s_lqr[k], ue, tc)[2]) for k in (25, 50, 75, int(cfg.num_steps))]
    print(f"[ue] delta_d={ue.delta_d} delta_v={ue.delta_v:.4f} e_bar={ue.e_bar} tube_scale={ue.tube_scale} | gamma={tc.gamma:.3f} "
          f"L_y={tc.l_y:.3f} L_z={tc.l_z:.3f} kappa_B={tc.kappa_b:.3f} | base margin at tau=0.5/1/1.5/{cfg.T}s "
          f"(growth 1): {mb[0]:.2f}/{mb[1]:.2f}/{mb[2]:.2f}/{mb[3]:.2f} of c_B={cfg.base_set_c}", flush=True)

    run_name = args.run_name or f"runway_p1_ue_td3_seed{ra_cfg.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {run_dir} total_steps={ra_cfg.total_steps} hidden={ra_cfg.hidden_size} num_envs={ra_cfg.num_envs}", flush=True)

    result = run_runway_ue_sa_training(ra_cfg, cfg, region_cfg, ue, output_dir=str(run_dir), actor_init=args.actor_init)

    for name in ("final_state", "best_state"):
        with open(run_dir / f"{name.replace('_state', '')}_weights.pkl", "wb") as f:
            pickle.dump(jax.device_get(result[name]), f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(result["history"], f)
    with open(run_dir / "certificate.json", "w", encoding="utf-8") as f:
        json.dump(cert, f, indent=2, default=float)
    scale, low, high = landing_action_box(cfg)
    save_learned_backup_policy(
        run_dir / "runway_backup_policy_actor.pkl",
        actor_params=jax.device_get(result["best_state"]["actor_params"]), actor_cfg=result["actor_cfg"],
        action_scale=np.asarray(scale), action_low=np.asarray(low), action_high=np.asarray(high),
        metadata={
            "task": "runway_retreat_under_disturbance",
            "training_objective": "discounted_safe_arrival_ue_tightened",
            "sa_backbone": "td3",
            "deployed_action": "tanh(mean) (deterministic)",
            "observation_feature_mode": "x(10, p_x = 0) + d_hat(3, world-frame acceleration estimate)",
            "beta": float(ra_cfg.beta),
            "runway_config": cfg.as_dict(),
            "ue_config": ue.as_dict(),
            "tube_constants": result["tube"].as_dict(),
            "design_region": region_cfg.as_dict(),
            "certificate": json.loads(json.dumps(cert, default=float)),
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
