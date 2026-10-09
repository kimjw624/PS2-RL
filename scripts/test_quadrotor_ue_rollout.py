#!/usr/bin/env python
"""Tests for the UE-bCBF estimated-disturbance backup rollout."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp

from ps2rl.cil.quadrotor_backup_cbf import (
    QuadrotorBCBFConfig,
    make_backup_runtime,
)
from ps2rl.cil.backup_cbf import rollout_backup_flow_and_sensitivity_with_info
from ps2rl.cil.quadrotor_ue_rollout import (
    estimated_disturbance_backup_dynamics,
    quadrotor_disturbance_injection_matrix,
    rollout_estimated_disturbance_backup_flow_and_sensitivities_with_info,
)

DT = 0.02


def _make_cfg(*, num_steps: int = 10) -> QuadrotorBCBFConfig:
    return QuadrotorBCBFConfig(
        T=DT * num_steps,
        dt=DT,
        num_steps=num_steps,
        backup_policy_mode="analytic",
        use_analytic_jacobian=False,
    )


def _test_disturbance_injection_matrix() -> None:
    e_d = quadrotor_disturbance_injection_matrix(dtype=jnp.float32)
    d = jnp.asarray([0.7, -0.4, 0.2], dtype=jnp.float32)
    injected = e_d @ d

    expected = jnp.asarray(
        [0.0, 0.0, 0.0, 0.7, -0.4, 0.2, 0.0, 0.0, 0.0, 0.0],
        dtype=jnp.float32,
    )
    assert e_d.shape == (10, 3)
    assert jnp.allclose(injected, expected, atol=1e-7)
    print("PASS: E_d injects d_hat only into translational acceleration")


def _test_zero_estimate_matches_original_ps2_rollout() -> None:
    cfg = _make_cfg(num_steps=10)
    system = make_backup_runtime(cfg)
    x0 = jnp.asarray(
        [0.0, 0.0, 2.15, 0.15, -0.08, 0.04, 1.0, 0.0, 0.0, 0.0],
        dtype=jnp.float32,
    )
    d_hat = jnp.zeros((3,), dtype=jnp.float32)

    xs_nom, phis_nom, _ = rollout_backup_flow_and_sensitivity_with_info(x0, cfg, system)
    xs_ue, phis_ue, thetas_ue, js_ue, info = (
        rollout_estimated_disturbance_backup_flow_and_sensitivities_with_info(
            x0,
            d_hat,
            cfg,
            system,
        )
    )

    assert xs_ue.shape == (11, 10)
    assert phis_ue.shape == (11, 10, 10)
    assert thetas_ue.shape == (11, 10, 3)
    assert js_ue.shape == (11, 10, 10)
    assert bool(info["all_finite"])

    # With d_hat = 0 the UE state rollout is exactly the original backup rollout.
    assert jnp.allclose(xs_ue, xs_nom, atol=2e-6, rtol=2e-6)
    # Phi uses the same frozen-J matrix exponential convention as original PS2-RL.
    assert jnp.allclose(phis_ue, phis_nom, atol=2e-5, rtol=2e-5)
    assert jnp.allclose(phis_ue[0], jnp.eye(10, dtype=jnp.float32), atol=1e-7)
    assert jnp.allclose(thetas_ue[0], jnp.zeros((10, 3), dtype=jnp.float32), atol=1e-7)

    print("PASS: d_hat = 0 reproduces the original PS2-RL backup rollout and Phi")


def _test_nonzero_estimate_and_jacobian() -> None:
    cfg = _make_cfg(num_steps=1)
    system = make_backup_runtime(cfg)
    x0 = jnp.asarray(
        [0.0, 0.0, 2.10, 0.12, -0.05, 0.03, 1.0, 0.0, 0.0, 0.0],
        dtype=jnp.float32,
    )
    d_hat = jnp.asarray([0.8, -0.3, 0.25], dtype=jnp.float32)

    xs_zero, _, _, _, _ = rollout_estimated_disturbance_backup_flow_and_sensitivities_with_info(
        x0,
        jnp.zeros((3,), dtype=jnp.float32),
        cfg,
        system,
    )
    xs, phis, thetas, jacobians, info = (
        rollout_estimated_disturbance_backup_flow_and_sensitivities_with_info(
            x0,
            d_hat,
            cfg,
            system,
        )
    )

    # Euler state propagation should add exactly dt * d_hat to velocity at step 1.
    velocity_shift = xs[1, 3:6] - xs_zero[1, 3:6]
    assert jnp.allclose(velocity_shift, DT * d_hat, atol=2e-6, rtol=2e-6)

    # J_cl is computed from the same estimated-disturbance closed-loop field.
    expected_j0 = jax.jacfwd(
        lambda z: estimated_disturbance_backup_dynamics(z, d_hat, system)
    )(x0)
    assert jnp.allclose(jacobians[0], expected_j0, atol=2e-6, rtol=2e-6)

    assert jnp.all(jnp.isfinite(phis))
    assert jnp.all(jnp.isfinite(thetas))
    assert float(jnp.linalg.norm(thetas[1])) > 0.0
    assert bool(info["all_finite"])

    print("PASS: nonzero d_hat shifts the rollout correctly and produces finite Theta")
    print(f"||Theta(dt)||_F = {float(jnp.linalg.norm(thetas[1])):.6f}")


def main() -> None:
    _test_disturbance_injection_matrix()
    _test_zero_estimate_matches_original_ps2_rollout()
    _test_nonzero_estimate_and_jacobian()
    print("PASS: UE estimated-disturbance rollout tests complete")


if __name__ == "__main__":
    main()
