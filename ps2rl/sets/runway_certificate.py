"""Base-set certificate for the runway task (checked before Phase-I training).

B = {e^T P e <= c_B} of the retreat LQR is a valid terminal set for the backup if

  1. c_B <= c_U: the unclipped LQR stays inside the action box on B (DiscreteLQR bound);
  2. chart: ||phi|| < 2 on B (the quaternion-error chart is one-to-one), tilt < 90 deg;
  3. B n S is invariant geometrically: on B, v_y <= -v_ret + sqrt(c_B (P^-1)_vy) < 0 (p_y only
     decreases, so the runway barrier never decreases) and p_z <= z_hold + sqrt(c_B (P^-1)_zz)
     <= z_max (the ceiling holds on all of B);
  4. Lyapunov: V(x+) < V(x) on the boundary of B for the nonlinear Euler plant under the
     LQR - nominally, and robustly under the worst disturbance |d| <= delta_d (projected
     gradient ascent over the boundary; the disturbance at each point is the one that
     increases V(x+) most to first order). Then B is (robustly) invariant after hand-off.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.sets.runway_sets import base_set_extents, build_runway_sets


def lyapunov_worst_adversarial(ctrl, step_single: Callable, *, level: float, n_starts: int = 2048, iters: int = 200,
                               lr: float = 0.05, seed: int = 0) -> tuple[float, np.ndarray]:
    """max V(x+)/V(x) over {e^T P e = level} by projected gradient ascent (dimension-generic)."""
    p_np = ctrl.p_matrix_f64()
    n = p_np.shape[0]
    p = jnp.asarray(p_np)
    l_inv_t = jnp.asarray(np.linalg.inv(np.linalg.cholesky(p_np)).T)

    def ratio(u, lev):
        u = u / jnp.linalg.norm(u)
        e = jnp.sqrt(lev) * (l_inv_t @ u)
        x = ctrl.state_from_error(e)
        e_next = ctrl.error_state(step_single(x, ctrl.action(x)))
        return (e_next @ p @ e_next) / (e @ p @ e)

    vg = jax.jit(jax.vmap(jax.value_and_grad(ratio), in_axes=(0, None)))
    u = jnp.asarray(np.random.default_rng(seed).standard_normal((n_starts, n)))
    for _ in range(iters):
        _, g = vg(u, level)
        u = u + lr * g
        u = u / jnp.linalg.norm(u, axis=1, keepdims=True)
    v, _ = vg(u, level)
    i = int(jnp.argmax(v))
    return float(v[i]), np.sqrt(level) * (np.asarray(l_inv_t) @ (np.asarray(u[i]) / np.linalg.norm(u[i])))


def check_runway_base_set(cfg, ue=None, *, lyap: bool = True, raise_on_fail: bool = True) -> dict:
    from ps2rl.phase1_sa.quadrotor_landing_ue_sa_env import landing_ue_step_fn

    _, base_set = build_runway_sets(cfg)  # raises if c_B > c_U
    ctrl = base_set.controller
    ext = base_set_extents(cfg)
    c_b = float(cfg.base_set_c)
    out = {"c_B": c_b, "c_U": float(ctrl.max_certified_level), "extents": ext,
           "v_y_max_in_B": -float(cfg.v_ret) + ext["v_y"], "z_max_in_B": float(cfg.z_hold) + ext["z"]}
    problems = []
    if ext["phi_norm"] >= 2.0 or ext["phi_tilt"] >= np.pi / 2:
        problems.append(f"chart: ||phi|| up to {ext['phi_norm']:.2f} (tilt {np.rad2deg(ext['phi_tilt']):.0f} deg) on B")
    if out["v_y_max_in_B"] >= 0.0:
        problems.append(f"v_y can be >= 0 on B (max {out['v_y_max_in_B']:.3f}): raise v_ret or lower c_B")
    if out["z_max_in_B"] > float(cfg.z_max):
        problems.append(f"B reaches above the ceiling (z up to {out['z_max_in_B']:.2f} > {cfg.z_max})")
    if lyap:
        plant = landing_ue_step_fn(cfg)
        zero = jnp.zeros(3)
        worst_nom, _ = lyapunov_worst_adversarial(ctrl, lambda x, u: plant(x, u, zero.astype(x.dtype)), level=c_b)
        out["lyap_ratio_at_c_B"] = worst_nom
        if worst_nom >= 1.0:
            problems.append(f"nominal Lyapunov decrease fails on dB: worst V+/V = {worst_nom:.4f}")
        if ue is not None:
            p = jnp.asarray(ctrl.p_matrix_f64(), dtype=jnp.float32)
            dd = float(ue.delta_d)
            p_v = p[:, 1:4]  # columns of P for the velocity error

            def step_worst(x, u):
                xn = plant(x, u, jnp.zeros(3, x.dtype))
                g = (ctrl.error_state(xn) @ p_v)  # d V / d v (up to 2) at the nominal next state
                d = dd * g / jnp.maximum(jnp.linalg.norm(g), 1e-9)
                return plant(x, u, d.astype(x.dtype))

            worst_rob, _ = lyapunov_worst_adversarial(ctrl, step_worst, level=c_b)
            out["robust_lyap_ratio_at_c_B"] = worst_rob
            out["delta_d"] = dd
            if worst_rob >= 1.0:
                problems.append(f"B not robustly invariant under |d| <= {dd}: worst V+/V = {worst_rob:.4f}")
    out["ok"] = not problems
    msg = (f"[certificate] runway B: c_B={c_b} (c_U={out['c_U']:.2f}) | on B: v_y <= {out['v_y_max_in_B']:.3f} m/s, "
           f"z <= {out['z_max_in_B']:.2f} m (ceiling {cfg.z_max}), tilt <= {np.rad2deg(ext['phi_tilt']):.0f} deg"
           + (f" | V+/V nominal {out['lyap_ratio_at_c_B']:.4f}" if "lyap_ratio_at_c_B" in out else "")
           + (f", robust (|d|<={out['delta_d']}) {out['robust_lyap_ratio_at_c_B']:.4f}" if "robust_lyap_ratio_at_c_B" in out else ""))
    print(msg + (" -> OK" if not problems else " -> FAIL: " + "; ".join(problems)), flush=True)
    if problems and raise_on_fail:
        raise SystemExit("runway base-set certificate failed: " + "; ".join(problems))
    return out


__all__ = ["check_runway_base_set", "lyapunov_worst_adversarial"]
