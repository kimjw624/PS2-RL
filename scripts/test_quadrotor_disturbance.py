#!/usr/bin/env python
"""Consistency tests for the deterministic quadrotor disturbance path."""

from __future__ import annotations

from math import pi
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp

from ps2rl.envs.quadrotor_env import QuadrotorEnvConfig, build_quadrotor_env
from ps2rl.uncertainty.quadrotor_disturbance import (
    disturbance_bounds,
    disturbance_rate,
    disturbance_value,
    make_sinusoidal_disturbance_params,
)


def _test_disturbance_model() -> None:
    amplitude = 1.5
    frequency_hz = 0.5
    params = make_sinusoidal_disturbance_params(
        amplitude=amplitude,
        frequency_hz=frequency_hz,
        phase=pi / 2.0,
        direction=(3.0, 0.0, 0.0),
    )

    d0 = disturbance_value(0.0, params)
    dd_quarter = disturbance_rate(0.5, params)
    delta_d, delta_v = disturbance_bounds(amplitude, frequency_hz)

    assert jnp.allclose(jnp.linalg.norm(params.direction), 1.0, atol=1e-6)
    assert jnp.allclose(d0, jnp.asarray([1.5, 0.0, 0.0]), atol=1e-6)
    assert float(jnp.linalg.norm(d0)) <= delta_d + 1e-6
    assert float(jnp.linalg.norm(dd_quarter)) <= delta_v + 1e-6
    assert abs(delta_d - 1.5) < 1e-12
    assert abs(delta_v - 1.5 * pi) < 1e-12

    print("PASS: deterministic disturbance model")
    print(f"delta_d = {delta_d:.6f} m/s^2")
    print(f"delta_v = {delta_v:.6f} m/s^3")


def _test_environment_injection() -> None:
    key = jax.random.PRNGKey(0)
    step_key = jax.random.PRNGKey(1)

    nominal = build_quadrotor_env(
        QuadrotorEnvConfig(max_steps=5, disturbance_mode="none")
    )
    disturbed = build_quadrotor_env(
        QuadrotorEnvConfig(
            max_steps=5,
            disturbance_mode="sinusoidal",
            disturbance_amplitude=1.5,
            disturbance_frequency_hz=0.5,
            disturbance_phase=pi / 2.0,
            disturbance_direction_x=1.0,
            disturbance_direction_y=0.0,
            disturbance_direction_z=0.0,
        )
    )

    state_nom, _ = nominal.reset(key)
    state_dis, _ = disturbed.reset(key)
    assert jnp.allclose(state_nom.x, state_dis.x)

    hover_action = jnp.asarray([9.81, 0.0, 0.0, 0.0], dtype=jnp.float32)
    next_nom, _, _, _, _, info_nom = nominal.step(state_nom, hover_action, step_key)
    next_dis, _, _, _, _, info_dis = disturbed.step(state_dis, hover_action, step_key)

    expected_velocity_difference = jnp.asarray([0.02 * 1.5, 0.0, 0.0], dtype=jnp.float32)
    actual_velocity_difference = next_dis.x[3:6] - next_nom.x[3:6]

    assert jnp.allclose(actual_velocity_difference, expected_velocity_difference, atol=1e-6)
    assert jnp.allclose(info_nom.disturbance_accel, jnp.zeros((3,)), atol=1e-6)
    assert jnp.allclose(
        info_dis.disturbance_accel,
        jnp.asarray([1.5, 0.0, 0.0], dtype=jnp.float32),
        atol=1e-6,
    )

    print("PASS: disturbance is injected into translational acceleration dynamics")


def main() -> None:
    _test_disturbance_model()
    _test_environment_injection()


if __name__ == "__main__":
    main()
