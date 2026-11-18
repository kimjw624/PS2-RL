#!/usr/bin/env python
"""Backup-CBF QP health check for a reference trajectory (no training required).

Runs a closed-loop rollout in the quadrotor env with a feedforward + PD stand-in
actor, projecting every action through the backup-CBF QP, and reports the flags
that actually indicate QP trouble:

    use_solver       False => the QP result was DISCARDED and the backup policy
                     action was used instead. The step is still safe, but the
                     CIL is not filtering the policy, it is replacing it.
                     Nothing in the standard training metrics surfaces this.
    q_saturated      constraint rows hit max_abs_constraint_value; solver skipped.
    slack            > 0 means a CBF row was relaxed to keep the QP feasible.
    delta_min_u_ref  how far the raw action was from the feasible set. A large
                     value means the actor is fighting the filter, which is the
                     regime that produces slack blow-ups.

The stand-in actor deliberately has two modes, because the two answer different
questions:

    --actor-z-cap inf   the actor chases a reference that runs above the hard
                        deck and never gives up. This is the worst case, and it
                        approximates an untrained policy during warm-up.
    --actor-z-cap 2.8   the actor respects the deck, approximating a trained
                        policy. Use this to check the QP itself rather than the
                        pathology of a controller that cannot learn.

Examples
--------
    python scripts/check_takeoff_qp_health.py \\
        --reference_path ps2rl/envs/assets/quadrotor_takeoff_deck_reference.npz \\
        --backup_policy_mode learned \\
        --learned_backup_policy_path checkpoints/deployed_sa/quadrotor_sa/best_weights.pkl

    # worst case, plus a state-grid sweep
    python scripts/check_takeoff_qp_health.py --actor-z-cap inf --grid
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference_path", type=str, default="ps2rl/envs/assets/quadrotor_takeoff_deck_reference.npz")
    p.add_argument("--backup_policy_mode", type=str, default="learned", choices=["analytic", "learned"])
    p.add_argument(
        "--learned_backup_policy_path",
        type=str,
        default="checkpoints/deployed_sa/quadrotor_sa/best_weights.pkl",
        help="Required when --backup_policy_mode=learned.",
    )
    p.add_argument("--env_dt", type=float, default=0.02)
    p.add_argument("--horizon_T", type=float, default=2.0)
    p.add_argument("--num_steps", type=int, default=100)
    p.add_argument("--alpha", type=float, default=4.0)
    p.add_argument("--base_alpha", type=float, default=2.0)
    p.add_argument("--base_set_c", type=float, default=8.0)
    p.add_argument("--z_max", type=float, default=3.0)
    p.add_argument("--z_des", type=float, default=2.0)
    p.add_argument("--slack_weight", type=float, default=1e6)
    p.add_argument("--solver_tol", type=float, default=5e-4)
    p.add_argument(
        "--pid_min_virtual_accel_z",
        type=float,
        default=0.0,
        help=(
            "Analytic backup only (ignored when --backup_policy_mode=learned). "
            "0.0 lets the ceiling brake drive the PID virtual acceleration to exactly zero, "
            "which makes the commanded thrust DIRECTION undefined and NaNs the CBF rows. "
            "Use ~2.0 if you run the analytic backup against a ceiling-pressing reference."
        ),
    )
    p.add_argument(
        "--actor-z-cap",
        dest="actor_z_cap",
        type=float,
        default=2.8,
        help="Altitude the stand-in actor stops asking for; 'inf' to chase the reference forever.",
    )
    p.add_argument("--kp", type=float, default=3.0, help="Stand-in actor P gain on altitude error.")
    p.add_argument("--kd", type=float, default=3.0, help="Stand-in actor D gain on vertical velocity error.")
    p.add_argument("--f64", action="store_true", default=False, help="Run the env and QP in float64.")
    p.add_argument("--grid", action="store_true", default=False, help="Also sweep a grid of states near the deck.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    if args.f64:
        os.environ["JAX_ENABLE_X64"] = "1"

    import numpy as np
    import jax
    import jax.numpy as jnp

    from ps2rl.envs.quadrotor_env import QuadrotorEnvConfig, build_quadrotor_env
    from ps2rl.cil.quadrotor_backup_cbf import QuadrotorBCBFConfig, QuadrotorBackupCBFProjector

    dtype = jnp.float64 if args.f64 else jnp.float32
    prec = "f64" if args.f64 else "f32"

    cbf = QuadrotorBCBFConfig(
        num_steps=args.num_steps,
        T=args.horizon_T,
        dt=args.env_dt,
        alpha=args.alpha,
        base_alpha=args.base_alpha,
        base_set_c=args.base_set_c,
        z_max=args.z_max,
        z_des=args.z_des,
        slack_weight=args.slack_weight,
        solver_tol=args.solver_tol,
        backup_policy_mode=args.backup_policy_mode,
        learned_backup_policy_path=(
            args.learned_backup_policy_path if args.backup_policy_mode == "learned" else ""
        ),
        pid_min_virtual_accel_z=args.pid_min_virtual_accel_z,
    )
    proj = QuadrotorBackupCBFProjector(cbf)

    env_cfg = QuadrotorEnvConfig(
        dt=args.env_dt, reference_path=args.reference_path, z_max=args.z_max, z_des=args.z_des
    )
    env = build_quadrotor_env(env_cfg, dtype=dtype)

    bundle = np.load(args.reference_path)
    ref = bundle["states"]
    a_ff = bundle["a_cmd"] if "a_cmd" in bundle else np.full(ref.shape[0], 9.81)

    print(f"reference   : {args.reference_path}")
    print(f"backup      : {args.backup_policy_mode}   precision: {prec}   actor_z_cap: {args.actor_z_cap}")
    if args.backup_policy_mode == "learned":
        print("              (analytic PID gains, including --pid_min_virtual_accel_z, are unused in this mode)")
    print(f"episode     : {env_cfg.max_steps} steps, reference {ref.shape[0]} samples")

    state, _ = env.reset(jax.random.PRNGKey(0))
    key = jax.random.PRNGKey(1)

    z_hist, slack_h, dmin_h, proj_h = [], [], [], []
    fallbacks = 0
    q_sat = 0
    for i in range(env_cfg.max_steps - 1):
        j = min(i, ref.shape[0] - 1)
        z_target = min(ref[j, 2], args.actor_z_cap)
        vz_target = ref[j, 5] if ref[j, 2] < args.actor_z_cap else 0.0
        a = a_ff[j] - args.kp * (float(state.x[2]) - z_target) - args.kd * (float(state.x[5]) - vz_target)
        u_ref = jnp.array([float(np.clip(a, 0.0, cbf.a_cmd_max)), 0.0, 0.0, 0.0], dtype=dtype)

        u_safe, slack, use_solver, info = proj.solve_single_with_info(state.x, u_ref)

        z_hist.append(float(state.x[2]))
        slack_h.append(float(slack))
        dmin_h.append(float(info["delta_min_u_ref"]))
        proj_h.append(float(info["u_ref_minus_u_safe_norm"]))
        fallbacks += int(not bool(use_solver))
        q_sat += int(bool(info["q_saturated"]))

        state, _, _, _, done, _ = env.step(state, jnp.asarray(u_safe, dtype=dtype), key)
        if bool(done):
            break

    z = np.asarray(z_hist)
    slack = np.asarray(slack_h)
    n = len(z)

    print("")
    print(f"  use_solver=False (fallbacks) : {fallbacks}/{n}" + ("   <-- QP DISCARDED" if fallbacks else "   ok"))
    print(f"  q_saturated                  : {q_sat}/{n}" + ("   <-- rows capped" if q_sat else "   ok"))
    print(f"  slack           max          : {slack.max():.3e}" + ("   <-- CBF row relaxed" if slack.max() > 1e-3 else "   ok"))
    print(f"  delta_min_u_ref max          : {max(dmin_h):.3e}" + ("   <-- actor fighting the filter" if max(dmin_h) > 1.0 else "   ok"))
    print(f"  proj_norm       max          : {max(proj_h):.3f}")
    print(f"  altitude        max          : {z.max():.4f} m   (deck {args.z_max:.2f}, margin {args.z_max - z.max():+.4f})")
    print(f"  altitude        final        : {z[-1]:.4f} m")
    violated = z.max() > args.z_max + 1e-6
    print(f"  SAFETY                       : {'VIOLATED' if violated else 'held'}")

    if args.grid:
        print("")
        print("  state-grid sweep near the deck:")
        n_g = n_fall = n_sat = 0
        max_slack = 0.0
        for zz in np.linspace(0.5, args.z_max - 0.1, 13):
            for vz in np.linspace(-1.2, 1.2, 13):
                for th in (0.0, 0.08, 0.15):
                    x = jnp.array(
                        [0.0, 0.0, zz, 0.2, 0.0, vz, np.cos(th / 2), 0.0, np.sin(th / 2), 0.0], dtype=dtype
                    )
                    for a_ref in (9.81, 14.0, 20.0):
                        _, sl, used, inf = proj.solve_single_with_info(
                            x, jnp.array([a_ref, 0.0, 0.0, 0.0], dtype=dtype)
                        )
                        n_g += 1
                        n_fall += int(not bool(used))
                        n_sat += int(bool(inf["q_saturated"]))
                        max_slack = max(max_slack, float(sl))
        print(f"    {n_g} states: fallbacks {n_fall} ({100.0*n_fall/n_g:.1f}%), q_saturated {n_sat}, max slack {max_slack:.3e}")


if __name__ == "__main__":
    main()
