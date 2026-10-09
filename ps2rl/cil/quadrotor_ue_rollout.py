"""Estimated-disturbance backup rollout and sensitivities for UE-bCBF.

This module adds only the rollout quantities required before constructing the
UE-bCBF inequalities.  The current disturbance estimate is frozen over the
hypothetical backup horizon, as in the UE-bCBF formulation.

For the PS2-RL quadrotor state

    x = [p(3), v(3), q(4)] in R^10,

our disturbance d_hat in R^3 is a world-frame translational acceleration.
The implemented hypothetical backup flow is the discrete map

    x+ = P(x + dt * (f_cl(x) + E_d d_hat)),

where P is the same rollout post-processing used by PS2-RL.  Sensitivities are
therefore propagated with the exact Jacobians of this implemented map,

    Phi+   = F_k Phi,
    Theta+ = F_k Theta + G_k,

where F_k = d x+ / d x and G_k = d x+ / d d_hat.  This keeps Phi and Theta
consistent with Euler propagation and quaternion normalization.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp


Array = jax.Array


def quadrotor_disturbance_injection_matrix(*, dtype=jnp.float32) -> Array:
    """Return E_d in R^(10x3) for additive translational acceleration."""

    e_d = jnp.zeros((10, 3), dtype=dtype)
    return e_d.at[3:6, :].set(jnp.eye(3, dtype=dtype))


def estimated_disturbance_backup_dynamics(
    x: Array,
    d_hat: Array,
    system: Any,
) -> Array:
    """Return f_cl(x) + E_d d_hat for a frozen current estimate d_hat."""

    x = jnp.asarray(x)
    d_hat = jnp.asarray(d_hat, dtype=x.dtype)
    if x.shape != (10,):
        raise ValueError(f"quadrotor state must have shape (10,), got {x.shape}")
    if d_hat.shape != (3,):
        raise ValueError(f"d_hat must have shape (3,), got {d_hat.shape}")

    f_cl = system.dynamics_fn(x, system.backup_policy_fn(x))
    e_d = quadrotor_disturbance_injection_matrix(dtype=x.dtype)
    return f_cl + e_d @ d_hat


def rollout_estimated_disturbance_backup_flow_and_sensitivities_with_info(
    x0: Array,
    d_hat: Array,
    cfg: Any,
    system: Any,
) -> Tuple[Array, Array, Array, Array, Dict[str, Array]]:
    """Roll out phi, Phi, Theta, and J_cl for a frozen disturbance estimate.

    Returns
    -------
    xs:
        Estimated-disturbance backup states with shape ``(N+1, 10)``.
    phis:
        State-transition sensitivities d phi / d x0 with shape
        ``(N+1, 10, 10)``.
    thetas:
        Disturbance sensitivities d phi / d d_hat with shape
        ``(N+1, 10, 3)``.
    jacobians:
        Closed-loop Jacobian at every rollout node, shape ``(N+1, 10, 10)``.
        Since E_d d_hat is constant in x, this is also the Jacobian of the
        estimated-disturbance backup dynamics.
    info:
        Simple finiteness / magnitude diagnostics for testing and later CIL
        integration.
    """

    x0 = jnp.asarray(x0)
    d_hat = jnp.asarray(d_hat, dtype=x0.dtype)
    if x0.shape != (10,):
        raise ValueError(f"x0 must have shape (10,), got {x0.shape}")
    if d_hat.shape != (3,):
        raise ValueError(f"d_hat must have shape (3,), got {d_hat.shape}")

    dt = jnp.asarray(cfg.dt, dtype=x0.dtype)
    e_d = quadrotor_disturbance_injection_matrix(dtype=x0.dtype)

    def f_hat(z: Array) -> Array:
        return estimated_disturbance_backup_dynamics(z, d_hat, system)

    jac_hat = jax.jacfwd(f_hat)

    def discrete_step(z: Array, d: Array) -> Array:
        dz = estimated_disturbance_backup_dynamics(z, d, system)
        return system.postprocess_rollout_state_fn(z + dt * dz)

    jac_step_x = jax.jacfwd(discrete_step, argnums=0)
    jac_step_d = jax.jacfwd(discrete_step, argnums=1)

    def step(carry, _):
        x, phi_sens, theta_sens = carry
        j_cl = jac_hat(x)

        # Differentiate the exact discrete rollout map, not a separate
        # continuous/frozen-J approximation.  This includes quaternion
        # normalization and every other post-processing operation used by the
        # actual hypothetical backup rollout.
        x_next = discrete_step(x, d_hat)
        f_disc = jac_step_x(x, d_hat)
        g_disc = jac_step_d(x, d_hat)
        phi_next = f_disc @ phi_sens
        theta_next = f_disc @ theta_sens + g_disc
        return (x_next, phi_next, theta_next), (x_next, phi_next, theta_next, j_cl)

    phi0 = jnp.eye(10, dtype=x0.dtype)
    theta0 = jnp.zeros((10, 3), dtype=x0.dtype)

    (x_final, phi_final, theta_final), (xs_tail, phis_tail, thetas_tail, js_head) = jax.lax.scan(
        step,
        (x0, phi0, theta0),
        jnp.arange(int(cfg.num_steps)),
    )

    # Include the final-node Jacobian so every state node has a corresponding J.
    j_final = jac_hat(x_final)

    xs = jnp.concatenate([x0[None, :], xs_tail], axis=0)
    phis = jnp.concatenate([phi0[None, :, :], phis_tail], axis=0)
    thetas = jnp.concatenate([theta0[None, :, :], thetas_tail], axis=0)
    jacobians = jnp.concatenate([js_head, j_final[None, :, :]], axis=0)

    info = {
        "all_finite": (
            jnp.all(jnp.isfinite(xs))
            & jnp.all(jnp.isfinite(phis))
            & jnp.all(jnp.isfinite(thetas))
            & jnp.all(jnp.isfinite(jacobians))
        ),
        "max_abs_phi": jnp.max(jnp.abs(phis)),
        "max_abs_theta": jnp.max(jnp.abs(thetas)),
        "max_abs_jacobian": jnp.max(jnp.abs(jacobians)),
    }
    return xs, phis, thetas, jacobians, info


def rollout_estimated_disturbance_backup_flow_and_sensitivities(
    x0: Array,
    d_hat: Array,
    cfg: Any,
    system: Any,
) -> Tuple[Array, Array, Array, Array]:
    """Convenience wrapper that omits diagnostics."""

    xs, phis, thetas, jacobians, _ = (
        rollout_estimated_disturbance_backup_flow_and_sensitivities_with_info(
            x0,
            d_hat,
            cfg,
            system,
        )
    )
    return xs, phis, thetas, jacobians
