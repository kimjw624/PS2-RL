"""Three-dimensional disturbance observer for quadrotor translational dynamics.

This module implements the UE-bCBF disturbance observer on the translational
velocity subsystem

    v_dot = a_nom(x, u) + d,

where d in R^3 is a world-frame acceleration disturbance.  The observer is

    d_hat = Lambda (v - xi),
    xi_dot = a_nom(x, u) + d_hat,

with Lambda = lambda I_3.  The associated estimation error e = d - d_hat
satisfies

    e_dot = d_dot - Lambda e.

For known bounds ||d|| <= delta_d and ||d_dot|| <= delta_v, the UE-bCBF
error bound becomes

    e_bar(t) = exp(-lambda t) delta_d
             + (delta_v / lambda) (1 - exp(-lambda t)).

The implementation is intentionally limited to the observer and its analytic
error bound.  It does not modify the PS2-RL control-invariant layer yet.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array


class QuadrotorDisturbanceObserverState(NamedTuple):
    """Internal state xi of the three-dimensional disturbance observer."""

    xi: Array


def _validate_positive_scalar(name: str, value: float) -> float:
    value = float(value)
    if not jnp.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite scalar, got {value}")
    return value


def initialize_disturbance_observer(
    velocity: Array,
    *,
    dtype=None,
) -> QuadrotorDisturbanceObserverState:
    """Initialize xi(0) = v(0), which gives d_hat(0) = 0."""

    velocity = jnp.asarray(velocity, dtype=dtype)
    if velocity.shape != (3,):
        raise ValueError(f"velocity must have shape (3,), got {velocity.shape}")
    return QuadrotorDisturbanceObserverState(xi=velocity)


def disturbance_estimate(
    velocity: Array,
    state: QuadrotorDisturbanceObserverState,
    *,
    lambda_gain: float = 20.0,
) -> Array:
    """Return the current disturbance estimate d_hat = lambda (v - xi)."""

    lambda_gain = _validate_positive_scalar("lambda_gain", lambda_gain)
    velocity = jnp.asarray(velocity, dtype=state.xi.dtype)
    if velocity.shape != (3,):
        raise ValueError(f"velocity must have shape (3,), got {velocity.shape}")
    return jnp.asarray(lambda_gain, dtype=state.xi.dtype) * (velocity - state.xi)


def disturbance_observer_predict(
    state: QuadrotorDisturbanceObserverState,
    nominal_acceleration: Array,
    d_hat: Array,
    *,
    dt: float,
) -> QuadrotorDisturbanceObserverState:
    """Euler-propagate xi over one controller/environment sample.

    The current estimate ``d_hat`` must be computed from the current measured
    velocity before this prediction.  At the next measurement, call
    :func:`disturbance_estimate` again using the returned state.
    """

    dt = _validate_positive_scalar("dt", dt)
    nominal_acceleration = jnp.asarray(nominal_acceleration, dtype=state.xi.dtype)
    d_hat = jnp.asarray(d_hat, dtype=state.xi.dtype)
    if nominal_acceleration.shape != (3,):
        raise ValueError(
            f"nominal_acceleration must have shape (3,), got {nominal_acceleration.shape}"
        )
    if d_hat.shape != (3,):
        raise ValueError(f"d_hat must have shape (3,), got {d_hat.shape}")

    xi_next = state.xi + jnp.asarray(dt, dtype=state.xi.dtype) * (
        nominal_acceleration + d_hat
    )
    return QuadrotorDisturbanceObserverState(xi=xi_next)


def observer_error_bound(
    t: Array | float,
    *,
    delta_d: float,
    delta_v: float,
    lambda_gain: float = 20.0,
    dtype=jnp.float32,
) -> Array:
    """Return the UE-bCBF analytic estimation-error bound e_bar(t)."""

    lambda_gain = _validate_positive_scalar("lambda_gain", lambda_gain)
    delta_d = float(delta_d)
    delta_v = float(delta_v)
    if delta_d < 0.0 or not jnp.isfinite(delta_d):
        raise ValueError(f"delta_d must be nonnegative and finite, got {delta_d}")
    if delta_v < 0.0 or not jnp.isfinite(delta_v):
        raise ValueError(f"delta_v must be nonnegative and finite, got {delta_v}")

    t_arr = jnp.asarray(t, dtype=dtype)
    lam = jnp.asarray(lambda_gain, dtype=dtype)
    dd = jnp.asarray(delta_d, dtype=dtype)
    dv = jnp.asarray(delta_v, dtype=dtype)
    decay = jnp.exp(-lam * t_arr)
    return decay * dd + (dv / lam) * (1.0 - decay)


def observer_error_bound_rate(
    t: Array | float,
    *,
    delta_d: float,
    delta_v: float,
    lambda_gain: float = 20.0,
    dtype=jnp.float32,
) -> Array:
    """Return d/dt of the analytic bound e_bar(t)."""

    lambda_gain = _validate_positive_scalar("lambda_gain", lambda_gain)
    delta_d = float(delta_d)
    delta_v = float(delta_v)
    if delta_d < 0.0 or not jnp.isfinite(delta_d):
        raise ValueError(f"delta_d must be nonnegative and finite, got {delta_d}")
    if delta_v < 0.0 or not jnp.isfinite(delta_v):
        raise ValueError(f"delta_v must be nonnegative and finite, got {delta_v}")

    t_arr = jnp.asarray(t, dtype=dtype)
    lam = jnp.asarray(lambda_gain, dtype=dtype)
    dd = jnp.asarray(delta_d, dtype=dtype)
    dv = jnp.asarray(delta_v, dtype=dtype)
    return jnp.exp(-lam * t_arr) * (dv - lam * dd)
