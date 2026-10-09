#!/usr/bin/env python
"""Tests for the 3-D UE-bCBF quadrotor disturbance observer."""

from __future__ import annotations

from math import pi
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import jax.numpy as jnp

from ps2rl.uncertainty.quadrotor_disturbance import (
    disturbance_bounds,
    disturbance_value,
    make_sinusoidal_disturbance_params,
)
from ps2rl.uncertainty.quadrotor_disturbance_observer import (
    disturbance_estimate,
    disturbance_observer_predict,
    initialize_disturbance_observer,
    observer_error_bound,
    observer_error_bound_rate,
)

DT = 0.02
LAMBDA = 20.0


def _nominal_acceleration(t: float) -> jnp.ndarray:
    """Arbitrary known nominal acceleration used to verify cancellation."""

    return jnp.asarray(
        [
            0.4 * jnp.sin(0.7 * t),
            -0.3 * jnp.cos(0.5 * t),
            0.2 * jnp.sin(1.1 * t),
        ],
        dtype=jnp.float32,
    )


def _test_initialization() -> None:
    velocity0 = jnp.asarray([1.2, -0.4, 0.7], dtype=jnp.float32)
    state = initialize_disturbance_observer(velocity0)
    d_hat0 = disturbance_estimate(velocity0, state, lambda_gain=LAMBDA)

    assert jnp.allclose(state.xi, velocity0, atol=1e-7)
    assert jnp.allclose(d_hat0, jnp.zeros((3,), dtype=jnp.float32), atol=1e-7)
    print("PASS: observer initializes with d_hat(0) = 0")


def _test_analytic_bound_and_rate() -> None:
    delta_d = 2.0
    delta_v = 2.0 * pi * 0.5 * delta_d

    e0 = float(
        observer_error_bound(
            0.0,
            delta_d=delta_d,
            delta_v=delta_v,
            lambda_gain=LAMBDA,
        )
    )
    einf = float(
        observer_error_bound(
            10.0,
            delta_d=delta_d,
            delta_v=delta_v,
            lambda_gain=LAMBDA,
        )
    )
    expected_inf = delta_v / LAMBDA

    assert abs(e0 - delta_d) < 1e-6
    assert abs(einf - expected_inf) < 1e-5

    t = 0.31
    h = 1e-4
    e_plus = float(
        observer_error_bound(
            t + h,
            delta_d=delta_d,
            delta_v=delta_v,
            lambda_gain=LAMBDA,
        )
    )
    e_minus = float(
        observer_error_bound(
            t - h,
            delta_d=delta_d,
            delta_v=delta_v,
            lambda_gain=LAMBDA,
        )
    )
    finite_difference = (e_plus - e_minus) / (2.0 * h)
    analytic = float(
        observer_error_bound_rate(
            t,
            delta_d=delta_d,
            delta_v=delta_v,
            lambda_gain=LAMBDA,
        )
    )
    assert abs(finite_difference - analytic) < 2e-3

    print("PASS: e_bar(t) and e_bar_dot(t) match the UE-bCBF formulas")
    print(f"e_bar(0)       = {e0:.6f} m/s^2")
    print(f"e_bar(infinity)~ {expected_inf:.6f} m/s^2")


def _run_tracking_case(
    *,
    amplitude: float,
    frequency_hz: float,
    phase: float,
    direction,
    duration: float = 6.0,
) -> tuple[float, float, float]:
    params = make_sinusoidal_disturbance_params(
        amplitude=amplitude,
        frequency_hz=frequency_hz,
        phase=phase,
        direction=direction,
    )
    delta_d, delta_v = disturbance_bounds(amplitude, frequency_hz)

    velocity = jnp.asarray([0.3, -0.2, 0.1], dtype=jnp.float32)
    observer_state = initialize_disturbance_observer(velocity)

    max_error = 0.0
    max_bound_violation = -float("inf")
    max_ratio = 0.0
    num_steps = int(round(duration / DT))

    for k in range(num_steps + 1):
        t = k * DT
        d_true = disturbance_value(t, params)
        d_hat = disturbance_estimate(velocity, observer_state, lambda_gain=LAMBDA)
        error = float(jnp.linalg.norm(d_true - d_hat))
        e_bar = float(
            observer_error_bound(
                t,
                delta_d=delta_d,
                delta_v=delta_v,
                lambda_gain=LAMBDA,
            )
        )

        max_error = max(max_error, error)
        max_bound_violation = max(max_bound_violation, error - e_bar)
        if e_bar > 1e-12:
            max_ratio = max(max_ratio, error / e_bar)

        assert error <= e_bar + 2e-5, (
            f"observer error bound violated at t={t:.3f}: "
            f"||e||={error:.8f}, e_bar={e_bar:.8f}"
        )

        if k == num_steps:
            break

        a_nom = _nominal_acceleration(t)
        velocity = velocity + DT * (a_nom + d_true)
        observer_state = disturbance_observer_predict(
            observer_state,
            a_nom,
            d_hat,
            dt=DT,
        )

    return max_error, max_bound_violation, max_ratio


def _test_tracking_is_bounded() -> None:
    cases = [
        (1.5, 0.1, 0.0, (1.0, 0.0, 0.0)),
        (1.5, 0.5, pi / 2.0, (0.0, 1.0, 1.0)),
        (2.0, 0.5, 0.7, (1.0, -2.0, 0.5)),
        (2.5, 0.3, 2.1, (-1.0, 0.2, 0.8)),
    ]

    worst_ratio = 0.0
    worst_violation = -float("inf")
    for amplitude, frequency_hz, phase, direction in cases:
        _, violation, ratio = _run_tracking_case(
            amplitude=amplitude,
            frequency_hz=frequency_hz,
            phase=phase,
            direction=direction,
        )
        worst_ratio = max(worst_ratio, ratio)
        worst_violation = max(worst_violation, violation)

    assert worst_violation <= 2e-5
    print("PASS: ||d - d_hat|| stays below e_bar(t) in all sinusoidal tests")
    print(f"worst ||e|| / e_bar = {worst_ratio:.6f}")


def main() -> None:
    _test_initialization()
    _test_analytic_bound_and_rate()
    _test_tracking_is_bounded()
    print("PASS: quadrotor disturbance observer tests complete")


if __name__ == "__main__":
    main()
