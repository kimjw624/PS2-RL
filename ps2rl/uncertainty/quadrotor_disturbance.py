"""Deterministic bounded translational disturbance for the quadrotor.

The disturbance is a world-frame acceleration applied to the translational
velocity dynamics:

    d(t) = A n sin(2 pi f t + phi),

where ``n`` is a unit vector.  Therefore

    ||d(t)|| <= A,
    ||d_dot(t)|| <= 2 pi f A.

This module intentionally contains no domain randomization.  A single set of
parameters is supplied by the environment configuration and is held fixed for
the episode/run.  Setting the environment's disturbance mode to ``none``
recovers the original nominal PS2-RL dynamics exactly.
"""

from __future__ import annotations

from math import pi
from typing import NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array


class SinusoidalDisturbanceParams(NamedTuple):
    amplitude: Array
    frequency_hz: Array
    phase: Array
    direction: Array


def make_sinusoidal_disturbance_params(
    *,
    amplitude: float,
    frequency_hz: float,
    phase: float,
    direction,
    dtype=jnp.float32,
) -> SinusoidalDisturbanceParams:
    """Create validated fixed sinusoidal-disturbance parameters."""

    if float(amplitude) < 0.0:
        raise ValueError("disturbance amplitude must be nonnegative")
    if float(frequency_hz) < 0.0:
        raise ValueError("disturbance frequency must be nonnegative")

    direction_arr = jnp.asarray(direction, dtype=dtype)
    if direction_arr.shape != (3,):
        raise ValueError(f"disturbance direction must have shape (3,), got {direction_arr.shape}")

    direction_norm = jnp.linalg.norm(direction_arr)
    if float(direction_norm) <= 1e-12:
        raise ValueError("disturbance direction must be nonzero")

    return SinusoidalDisturbanceParams(
        amplitude=jnp.asarray(amplitude, dtype=dtype),
        frequency_hz=jnp.asarray(frequency_hz, dtype=dtype),
        phase=jnp.asarray(phase, dtype=dtype),
        direction=direction_arr / direction_norm,
    )


def disturbance_value(t: Array | float, params: SinusoidalDisturbanceParams) -> Array:
    """Return the world-frame translational acceleration disturbance [m/s^2]."""

    dtype = params.direction.dtype
    t_arr = jnp.asarray(t, dtype=dtype)
    omega = jnp.asarray(2.0 * pi, dtype=dtype) * params.frequency_hz
    scalar = params.amplitude * jnp.sin(omega * t_arr + params.phase)
    return scalar[..., None] * params.direction


def disturbance_rate(t: Array | float, params: SinusoidalDisturbanceParams) -> Array:
    """Return the analytic disturbance derivative [m/s^3]."""

    dtype = params.direction.dtype
    t_arr = jnp.asarray(t, dtype=dtype)
    omega = jnp.asarray(2.0 * pi, dtype=dtype) * params.frequency_hz
    scalar = params.amplitude * omega * jnp.cos(omega * t_arr + params.phase)
    return scalar[..., None] * params.direction


def disturbance_bounds(amplitude: float, frequency_hz: float) -> tuple[float, float]:
    """Return ``(delta_d, delta_v)`` for the sinusoidal disturbance family."""

    amplitude = float(amplitude)
    frequency_hz = float(frequency_hz)
    if amplitude < 0.0:
        raise ValueError("disturbance amplitude must be nonnegative")
    if frequency_hz < 0.0:
        raise ValueError("disturbance frequency must be nonnegative")
    return amplitude, 2.0 * pi * frequency_hz * amplitude
