"""Certify the landing base set and report the z_des sweet spot.

Computes the level bounds of the landing note (Sec. 4.5) for the configured LQR,
finds the lowest z_des at which the cone and ground stop being binding, checks the
chosen (z_des, c_B), and runs two numerical sanity checks:

  * containment: uniform samples inside B_c must satisfy h_cone >= 0 (checklist 4)
  * invariance: pi_B rollouts from the level set {V = c_B} (including the adversarial
    worst direction) must keep V <= c_B for the whole run

c_Lyap is a numerical check, not a proof (landing note, Remark 2). Report it as such.

    JAX_PLATFORMS=cpu python scripts/certify_landing_base_set.py --out outputs/landing_cert.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ps2rl.base_controller.quadrotor_landing_dlqr import QuadrotorLandingDLQR
from ps2rl.envs.quadrotor_env import quadrotor_step_euler
from ps2rl.envs.quadrotor_landing_config import QuadrotorLandingConfig
from ps2rl.sets import landing_certificate as lc
from ps2rl.sets.quadrotor_cone_sets import QuadrotorConeSafeSet


def parse_args(argv=None) -> argparse.Namespace:
    d = QuadrotorLandingConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("cone_r0", "cone_theta_deg", "cone_eps", "z_des", "base_set_c", "z_clear",
                 "lqr_q_x", "lqr_q_y", "lqr_q_z", "lqr_q_thetaz", "lqr_r_omega_z", "omega_max"):
        p.add_argument(f"--{name}", type=float, default=getattr(d, name))
    p.add_argument("--z_min", type=float, default=0.6)
    p.add_argument("--z_max", type=float, default=4.0)
    p.add_argument("--z_step", type=float, default=0.05)
    p.add_argument("--safety_factor", type=float, default=0.8,
                   help="Recommended c_B = safety_factor * min(c_U, c_chart, c_Lyap).")
    p.add_argument("--invariance_steps", type=int, default=500)
    p.add_argument("--out", type=str, default="")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    overrides = {k: v for k, v in vars(args).items() if k in QuadrotorLandingConfig.__dataclass_fields__}
    cfg = QuadrotorLandingConfig().replace(**overrides)
    cone = QuadrotorConeSafeSet.from_config(cfg)
    ctrl = QuadrotorLandingDLQR.from_config(cfg)
    p_inv = np.linalg.inv(ctrl.p_matrix_f64())

    def step_single(x, u):
        return quadrotor_step_euler(x, u, cfg.dt, cfg.gravity, cfg.a_cmd_min, cfg.a_cmd_max, cfg.omega_max)

    # ---- z-independent bounds
    c_u = float(ctrl.max_certified_level)
    c_chart = lc.c_chart_bound(p_inv)
    c_lyap = lc.c_lyap_adversarial(ctrl, step_single, c_hi=min(c_u, c_chart))
    c_rest = min(c_u, c_chart, c_lyap)
    c_rec = args.safety_factor * c_rest
    print(f"z-independent bounds: c_U = {c_u:.3f}   c_chart = {c_chart:.3f}   c_Lyap (adversarial) = {c_lyap:.3f}")
    print(f"  -> c_rest = {c_rest:.3f}; recommended c_B = {args.safety_factor} * c_rest = {c_rec:.2f}")

    # ---- z sweep
    z_grid = np.round(np.arange(args.z_min, args.z_max + 1e-9, args.z_step), 4)
    rows = []
    z_star = float("nan")
    z_star_cb = float("nan")
    for z in z_grid:
        row = lc.LevelBounds(
            z_des=float(z), c_u=c_u, c_chart=c_chart, c_lyap=c_lyap,
            c_cone_exact=lc.c_cone_exact(p_inv, cone, z_des=float(z), c_hi=c_rest),
            c_cone_note=lc.c_cone_note(p_inv, cone, z_des=float(z), c_hi=c_rest),
            c_ground=lc.c_ground_bound(p_inv, z_des=float(z), z_clear=cfg.z_clear),
        )
        rows.append(row)
        if np.isnan(z_star) and min(row.c_cone_exact, row.c_ground) >= c_rest - 1e-9:
            z_star = float(z)
        if np.isnan(z_star_cb) and min(row.c_cone_exact, row.c_ground) >= cfg.base_set_c - 1e-9:
            z_star_cb = float(z)
    print(f"\n{'z_des':>6} {'c_cone':>8} {'(note)':>8} {'c_ground':>9} {'c_bar':>7}  binding")
    for row in rows[:: max(1, int(round(0.1 / args.z_step)))]:
        print(f"{row.z_des:6.2f} {row.c_cone_exact:8.3f} {row.c_cone_note:8.3f} {row.c_ground:9.3f} {row.c_bar:7.3f}  {row.binding}")
    print(f"\nsweet spot: lowest z_des with cone+ground non-binding at c_rest: {z_star:.2f} m "
          f"(at the configured c_B = {cfg.base_set_c}: {z_star_cb:.2f} m)")

    # ---- configured point
    chosen = lc.level_bounds(ctrl, cone, z_des=cfg.z_des, z_clear=cfg.z_clear, c_lyap=c_lyap, c_hi=c_rest)
    rho = lc.support_radii(p_inv, cfg.base_set_c)
    ok = cfg.base_set_c <= chosen.c_bar + 1e-9
    print(f"\nconfigured z_des = {cfg.z_des}, c_B = {cfg.base_set_c}: c_bar(z_des) = {chosen.c_bar:.3f} "
          f"[{chosen.binding}] -> {'OK' if ok else 'VIOLATES Proposition 1'}")
    print("  extents at c_B: " + ", ".join(f"{k}={rho[k]:.3f}" for k in ("dx", "dz", "vx", "vz", "phix", "phiz")))
    tilt_deg = float(np.degrees(2.0 * np.arcsin(min(1.0, 0.5 * rho["phix"]))))
    k64 = ctrl.k_matrix_f64()
    du = np.sqrt(cfg.base_set_c * np.einsum("ij,jk,ik->i", k64, p_inv, k64))
    print(f"  max tilt in B ~ {tilt_deg:.0f} deg; lowest point {cfg.z_des - rho['dz']:.2f} m above pad; "
          f"max |pi_B - u*| on B: thrust {du[0]:.2f} m/s^2, omega_xy {du[1]:.1f} rad/s, omega_z {du[3]:.1f} rad/s")

    # ---- containment by sampling
    rng = np.random.default_rng(0)
    p = ctrl.p_matrix_f64()
    l_inv_t = np.linalg.inv(np.linalg.cholesky(p)).T
    n = 200_000
    u = rng.standard_normal((n, 9))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    radius = rng.uniform(size=(n, 1)) ** (1.0 / 9.0)
    e = np.sqrt(cfg.base_set_c) * radius * (u @ l_inv_t.T)
    x = ctrl.state_from_error(jnp.asarray(e))
    h = np.asarray(cone.value(x))
    print(f"\ncontainment: {n} uniform samples in B, min h_cone = {h.min():.4f} "
          f"({'B subset S' if h.min() >= 0 else 'B NOT in S'})")

    # ---- multi-step invariance from the boundary
    worst_ratio, e_worst = lc.lyapunov_worst_adversarial(ctrl, step_single, level=cfg.base_set_c)
    starts = np.sqrt(cfg.base_set_c) * (u[:4095] @ l_inv_t.T)
    starts = np.concatenate([starts, e_worst[None, :]], axis=0)
    p_j = jnp.asarray(p)

    @jax.jit
    def rollout(e0):
        def body(xk, _):
            xn = step_single(xk, ctrl.action(xk))
            en = ctrl.error_state(xn)
            return xn, (en @ p_j @ en, cone.value(xn))

        _, (vs, hs) = jax.lax.scan(body, ctrl.state_from_error(e0), xs=None, length=args.invariance_steps)
        return jnp.max(vs), jnp.min(hs), vs[-1]

    vmax, hmin, vend = jax.vmap(rollout)(jnp.asarray(starts))
    vmax, hmin, vend = map(np.asarray, (vmax, hmin, vend))
    print(f"invariance: {starts.shape[0]} pi_B rollouts x {args.invariance_steps} steps from V = c_B: "
          f"max V = {vmax.max():.4f} (c_B = {cfg.base_set_c}), min h_cone = {hmin.min():.4f}, "
          f"median final V = {np.median(vend):.2e}; adversarial one-step ratio at c_B = {worst_ratio:.4f}")

    report = {
        "config": cfg.as_dict(),
        "c_u": c_u,
        "c_chart": c_chart,
        "c_lyap_adversarial": c_lyap,
        "c_rest": c_rest,
        "recommended_c_B": c_rec,
        "sweet_spot_z_des_at_c_rest": z_star,
        "sweet_spot_z_des_at_c_B": z_star_cb,
        "chosen": chosen.as_dict(),
        "chosen_ok": bool(ok),
        "extents_at_c_B": rho,
        "containment_min_h": float(h.min()),
        "invariance_max_V": float(vmax.max()),
        "invariance_min_h": float(hmin.min()),
        "sweep": [r.as_dict() for r in rows],
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
