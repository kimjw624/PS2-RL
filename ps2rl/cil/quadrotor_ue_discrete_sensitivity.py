"""Discrete UE-bCBF sensitivities for the PS2-RL quadrotor.

The PS2 backup rollout is not the exact flow of the continuous dynamics.  It is
implemented as the discrete map

    x+ = P(x + dt * f_hat(x, d_hat)),

where ``P`` is the rollout post-processing operation (notably quaternion
normalization).  UE sensitivities used by the implemented controller should be
consistent with this map.

This module provides exact first-order Jacobians of that discrete map and their
projection to the 9-D physical tangent coordinates

    dz = [dp(3), dv(3), dtheta(3)].

For the right-multiplicative quaternion perturbation convention used in the
project, ``dq = 0.5 Xi(q) dtheta``.
"""

from __future__ import annotations

from typing import Any, Tuple

import jax
import jax.numpy as jnp

from ps2rl.cil.quadrotor_ue_rollout import estimated_disturbance_backup_dynamics
from ps2rl.utils.quaternion import quaternion_rate_matrix


Array = jax.Array


def tangent_basis(x: Array) -> Array:
    """Return B(x) in R^(10x9) mapping physical tangent dz to ambient dx."""

    x = jnp.asarray(x)
    if x.shape != (10,):
        raise ValueError(f"x must have shape (10,), got {x.shape}")
    q = x[6:10]
    xi = quaternion_rate_matrix(q)
    b = jnp.zeros((10, 9), dtype=x.dtype)
    b = b.at[0:6, 0:6].set(jnp.eye(6, dtype=x.dtype))
    b = b.at[6:10, 6:9].set(0.5 * xi)
    return b


def tangent_left_inverse(x: Array) -> Array:
    """Return B^+(x) in R^(9x10) for a unit quaternion state."""

    x = jnp.asarray(x)
    if x.shape != (10,):
        raise ValueError(f"x must have shape (10,), got {x.shape}")
    q = x[6:10]
    xi = quaternion_rate_matrix(q)
    bp = jnp.zeros((9, 10), dtype=x.dtype)
    bp = bp.at[0:6, 0:6].set(jnp.eye(6, dtype=x.dtype))
    bp = bp.at[6:9, 6:10].set(2.0 * xi.T)
    return bp


def discrete_backup_step(x: Array, d_hat: Array, dt: float | Array, system: Any) -> Array:
    """Return the exact discrete frozen-d_hat backup step used by PS2-RL."""

    x = jnp.asarray(x)
    d_hat = jnp.asarray(d_hat, dtype=x.dtype)
    dt_arr = jnp.asarray(dt, dtype=x.dtype)
    dx = estimated_disturbance_backup_dynamics(x, d_hat, system)
    return system.postprocess_rollout_state_fn(x + dt_arr * dx)


def discrete_ambient_jacobians(
    x: Array,
    d_hat: Array,
    dt: float | Array,
    system: Any,
) -> Tuple[Array, Array, Array]:
    """Return ``(x_next, F10, G10)`` for the implemented discrete backup map.

    ``F10 = d x_next / d x`` and ``G10 = d x_next / d d_hat`` include both the
    Euler state update and rollout post-processing such as quaternion
    normalization.
    """

    x = jnp.asarray(x)
    d_hat = jnp.asarray(d_hat, dtype=x.dtype)

    def step_x(z: Array) -> Array:
        return discrete_backup_step(z, d_hat, dt, system)

    def step_d(d: Array) -> Array:
        return discrete_backup_step(x, d, dt, system)

    x_next = step_x(x)
    f10 = jax.jacfwd(step_x)(x)
    g10 = jax.jacfwd(step_d)(d_hat)
    return x_next, f10, g10


def discrete_tangent_jacobians(
    x: Array,
    d_hat: Array,
    dt: float | Array,
    system: Any,
) -> Tuple[Array, Array, Array]:
    """Return ``(x_next, F9, G9)`` in moving physical tangent coordinates.

    The first-order perturbation update is

        dz_next = F9 dz + G9 delta_d_hat,

    with

        F9 = B^+(x_next) F10 B(x),
        G9 = B^+(x_next) G10.

    This automatically accounts for quaternion normalization and the change of
    tangent basis between the two state nodes.
    """

    x_next, f10, g10 = discrete_ambient_jacobians(x, d_hat, dt, system)
    b0 = tangent_basis(x)
    bp1 = tangent_left_inverse(x_next)
    f9 = bp1 @ f10 @ b0
    g9 = bp1 @ g10
    return x_next, f9, g9
