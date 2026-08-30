"""Experimental scaled-slack QP projectors for quadrotor evaluation.

The production backup-CBF QP uses the decision vector ``[u, s]`` and the
objective

    0.5 * w_u * ||u-u_ref||^2 + 0.5 * w_s * s^2.

For the deployed quadrotor configuration, ``w_u=1`` and ``w_s=1e6``.  That
Hessian condition ratio is unnecessarily difficult for a float32 interior-
point solve.  This module applies the exact coordinate change

    y = sqrt(w_s / w_u) * s,

so the solver sees ``[u, y]`` with an isotropic Hessian.  Constraint columns
are transformed consistently and the reported slack is converted back to
physical ``s``.  Therefore the mathematical QP is unchanged; only its
numerical parameterization differs.

An opt-in mode promotes only the assembled five-variable QP to float64 while
leaving the actor, nonlinear backup rollout, row construction, and emitted
action dtype unchanged.  Residual diagnostics are computed in the QP dtype.
This module is deliberately experimental and does not replace the production
PS2-RL projector.
"""

from __future__ import annotations

from math import isfinite, sqrt
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import qpax

from ps2rl.cil.backup_cbf import (
    _build_backup_cbf_qp_from_rows,
    build_discretized_backup_cbf_rows_with_info,
)
from ps2rl.cil.cil_policy import BCBFProjectionOps
from ps2rl.cil.quadrotor_backup_cbf import QuadrotorBCBFConfig, _resolve_runtime
from ps2rl.cil.quadrotor_ue_bcbf_experimental import (
    ExperimentalUEConfig,
    _controller_for_cfg,
    build_ue_rows_with_info,
)


Array = jax.Array


def _validated_slack_scale(cfg: QuadrotorBCBFConfig) -> float:
    control_weight = float(cfg.control_weight)
    slack_weight = float(cfg.slack_weight)
    if not isfinite(control_weight) or control_weight <= 0.0:
        raise ValueError(f"control_weight must be positive and finite, got {control_weight}")
    if not isfinite(slack_weight) or slack_weight <= 0.0:
        raise ValueError(f"slack_weight must be positive and finite, got {slack_weight}")
    return sqrt(slack_weight / control_weight)


def _validated_float64_mode(solve_float64: bool) -> bool:
    enabled = bool(solve_float64)
    if enabled and not bool(jax.config.x64_enabled):
        raise RuntimeError(
            "Float64 QP mode requires JAX_ENABLE_X64=1 to be set before Python starts"
        )
    return enabled


def _solve_scaled_rows(
    x: Array,
    u_ref: Array,
    a_rows: Array,
    b_rows: Array,
    *,
    cfg: QuadrotorBCBFConfig,
    runtime: Any,
    rows_finite: Array,
    q_saturated: Array,
    max_abs_q: Array,
    max_abs_b: Array,
    slack_scale: float,
    solve_float64: bool,
) -> Tuple[Array, Array, Array, Dict[str, Array]]:
    """Solve rows using scaled slack and optional float64 QP arithmetic."""

    output_dtype = x.dtype
    qp_dtype = jnp.float64 if solve_float64 else output_dtype
    m = int(runtime.action_dim)
    action_low_native = jnp.asarray(runtime.action_low, dtype=output_dtype)
    action_high_native = jnp.asarray(runtime.action_high, dtype=output_dtype)
    u_ref_native = jnp.asarray(u_ref, dtype=output_dtype)
    u_ref_native = jnp.nan_to_num(u_ref_native, nan=0.0, posinf=0.0, neginf=0.0)
    u_ref_native = jnp.clip(u_ref_native, action_low_native, action_high_native)

    fallback_u_native = runtime.backup_policy_fn(x)

    q_mat, q_vec, a_eq, b_eq, g, h = _build_backup_cbf_qp_from_rows(
        a_rows,
        b_rows,
        u_ref_native,
        cfg,
        runtime,
        dtype=output_dtype,
    )

    # The nonlinear backup rollout and row construction remain in the native
    # environment dtype.  Only the small QP is promoted, which isolates the
    # numerical experiment from the modeled dynamics and actor.
    a_rows_qp = jnp.asarray(a_rows, dtype=qp_dtype)
    b_rows_qp = jnp.asarray(b_rows, dtype=qp_dtype)
    u_ref_qp = jnp.asarray(u_ref_native, dtype=qp_dtype)
    fallback_u_qp = jnp.asarray(fallback_u_native, dtype=qp_dtype)
    action_low_qp = jnp.asarray(action_low_native, dtype=qp_dtype)
    action_high_qp = jnp.asarray(action_high_native, dtype=qp_dtype)
    q_mat = jnp.asarray(q_mat, dtype=qp_dtype)
    q_vec = jnp.asarray(q_vec, dtype=qp_dtype)
    a_eq = jnp.asarray(a_eq, dtype=qp_dtype)
    b_eq = jnp.asarray(b_eq, dtype=qp_dtype)
    g = jnp.asarray(g, dtype=qp_dtype)
    h = jnp.asarray(h, dtype=qp_dtype)

    fallback_z_scaled = jnp.concatenate(
        [fallback_u_qp, jnp.zeros((1,), dtype=qp_dtype)], axis=0
    )
    residual_u_ref = a_rows_qp @ u_ref_qp - b_rows_qp
    delta_min_u_ref = jnp.maximum(0.0, jnp.max(residual_u_ref))

    scale = jnp.asarray(slack_scale, dtype=qp_dtype)
    # Original variable: z=[u,s].  Scaled variable: z_tilde=[u,y], y=scale*s.
    q_mat_scaled = q_mat.at[m, m].set(jnp.asarray(cfg.control_weight, dtype=qp_dtype))
    g_scaled = g.at[:, m].set(g[:, m] / scale)

    q_mat_finite = jnp.all(jnp.isfinite(q_mat_scaled))
    q_vec_finite = jnp.all(jnp.isfinite(q_vec))
    g_finite = jnp.all(jnp.isfinite(g_scaled))
    h_finite = jnp.all(jnp.isfinite(h))
    inputs_finite = (
        jnp.asarray(rows_finite, dtype=jnp.bool_)
        & q_mat_finite
        & q_vec_finite
        & g_finite
        & h_finite
        & (~jnp.asarray(q_saturated, dtype=jnp.bool_))
    )

    def solve_qp(_: None) -> Array:
        return qpax.solve_qp_primal(
            q_mat_scaled,
            q_vec,
            a_eq,
            b_eq,
            g_scaled,
            h,
            solver_tol=cfg.solver_tol,
            target_kappa=cfg.target_kappa,
        )

    z_candidate = jax.lax.cond(inputs_finite, solve_qp, lambda _: fallback_z_scaled, operand=None)
    z_finite = inputs_finite & jnp.all(jnp.isfinite(z_candidate))
    z_scaled = jnp.where(z_finite, z_candidate, fallback_z_scaled)
    safe_u_qp = jnp.clip(z_scaled[:m], action_low_qp, action_high_qp)
    physical_slack_qp = z_scaled[m] / scale
    slack_max = cfg.max_abs_constraint_value if cfg.max_abs_constraint_value > 0.0 else 1.0e6
    physical_slack_qp = jnp.clip(
        physical_slack_qp, 0.0, jnp.asarray(slack_max, dtype=qp_dtype)
    )

    emitted_z_scaled = jnp.concatenate(
        [safe_u_qp, (physical_slack_qp * scale)[None]], axis=0
    )
    row_residual = a_rows_qp @ safe_u_qp - b_rows_qp - physical_slack_qp
    inequality_residual = g_scaled @ emitted_z_scaled - h
    safe_u = jnp.asarray(safe_u_qp, dtype=output_dtype)
    physical_slack = jnp.asarray(physical_slack_qp, dtype=output_dtype)
    info = {
        "q_mat_finite": jnp.asarray(q_mat_finite, dtype=jnp.bool_),
        "q_vec_finite": jnp.asarray(q_vec_finite, dtype=jnp.bool_),
        "g_finite": jnp.asarray(g_finite, dtype=jnp.bool_),
        "h_finite": jnp.asarray(h_finite, dtype=jnp.bool_),
        "inputs_finite": jnp.asarray(inputs_finite, dtype=jnp.bool_),
        "z_finite": jnp.asarray(z_finite, dtype=jnp.bool_),
        "q_saturated": jnp.asarray(q_saturated, dtype=jnp.bool_),
        "max_abs_q": jnp.asarray(max_abs_q, dtype=qp_dtype),
        "max_abs_b": jnp.asarray(max_abs_b, dtype=qp_dtype),
        "delta_min_u_ref": jnp.asarray(delta_min_u_ref, dtype=qp_dtype),
        "u_ref_minus_u_safe_norm": jnp.linalg.norm(u_ref_qp - safe_u_qp),
        "a_ref_minus_a_safe": u_ref_qp[0] - safe_u_qp[0],
        "r_ref_minus_r_safe": u_ref_qp[1] - safe_u_qp[1],
        "max_positive_row_residual": jnp.max(jnp.maximum(row_residual, 0.0)),
        "max_positive_inequality_residual": jnp.max(
            jnp.maximum(inequality_residual, 0.0)
        ),
        "slack_coordinate_scale": scale,
        "qp_solve_float64": jnp.asarray(solve_float64, dtype=jnp.bool_),
        "qp_solve_dtype_bits": jnp.asarray(64 if solve_float64 else 32, dtype=jnp.int32),
    }
    return safe_u, physical_slack, jnp.asarray(z_finite, dtype=jnp.bool_), info


class QuadrotorScaledSlackStandardProjector:
    """Standard bCBF projector with only the slack coordinate rescaled."""

    def __init__(
        self,
        cbf_cfg: QuadrotorBCBFConfig,
        runtime: Any | None = None,
        *,
        solve_float64: bool = False,
    ) -> None:
        self.cbf_cfg = cbf_cfg
        # Match QuadrotorBackupCBFProjector: resolving through the cache is
        # required because the runtime's terminal-set callback resolves that
        # same cached runtime while it is traced by JAX.
        self.runtime = _resolve_runtime(cbf_cfg, runtime)
        self.slack_scale = _validated_slack_scale(cbf_cfg)
        self.solve_float64 = _validated_float64_mode(solve_float64)
        self.solve_single_with_info = jax.jit(self._solve_single_with_info)
        self.solve_batch_with_info = jax.jit(jax.vmap(self._solve_single_with_info, in_axes=(0, 0)))

    def _solve_single_with_info(self, x: Array, u_ref: Array):
        x = self.runtime.sanitize_solve_state_fn(x)
        a_rows, b_rows, diag = build_discretized_backup_cbf_rows_with_info(
            x, self.cbf_cfg, self.runtime
        )
        return _solve_scaled_rows(
            x,
            u_ref,
            a_rows,
            b_rows,
            cfg=self.cbf_cfg,
            runtime=self.runtime,
            rows_finite=jnp.all(jnp.isfinite(a_rows)) & jnp.all(jnp.isfinite(b_rows)),
            q_saturated=diag["q_saturated"],
            max_abs_q=diag["max_abs_q"],
            max_abs_b=diag["max_abs_b"],
            slack_scale=self.slack_scale,
            solve_float64=self.solve_float64,
        )


class QuadrotorScaledSlackUEProjector:
    """Experimental UE-bCBF projector with the same slack rescaling."""

    def __init__(
        self,
        cbf_cfg: QuadrotorBCBFConfig,
        ue_cfg: ExperimentalUEConfig,
        runtime: Any | None = None,
        *,
        solve_float64: bool = False,
    ) -> None:
        self.cbf_cfg = cbf_cfg
        self.ue_cfg = ue_cfg
        self.runtime = _resolve_runtime(cbf_cfg, runtime)
        self.controller = _controller_for_cfg(cbf_cfg)
        self.slack_scale = _validated_slack_scale(cbf_cfg)
        self.solve_float64 = _validated_float64_mode(solve_float64)
        self.solve_single_with_info = jax.jit(self._solve_single_with_info)
        self.solve_batch_with_info = jax.jit(jax.vmap(self._solve_single_with_info, in_axes=(0, 0, 0, 0)))

    def _solve_single_with_info(self, x: Array, u_ref: Array, d_hat: Array, e_bar: Array):
        x = self.runtime.sanitize_solve_state_fn(x)
        d_hat = jnp.asarray(d_hat, dtype=x.dtype)
        e_bar = jnp.maximum(jnp.asarray(e_bar, dtype=x.dtype), 0.0)
        a_rows, b_rows, diag = build_ue_rows_with_info(
            x,
            d_hat,
            e_bar,
            self.cbf_cfg,
            self.ue_cfg,
            self.runtime,
            self.controller,
        )
        return _solve_scaled_rows(
            x,
            u_ref,
            a_rows,
            b_rows,
            cfg=self.cbf_cfg,
            runtime=self.runtime,
            rows_finite=diag["all_finite"] & jnp.all(jnp.isfinite(a_rows)) & jnp.all(jnp.isfinite(b_rows)),
            q_saturated=jnp.asarray(False),
            max_abs_q=jnp.asarray(0.0, dtype=x.dtype),
            max_abs_b=diag["max_abs_b"],
            slack_scale=self.slack_scale,
            solve_float64=self.solve_float64,
        )


def make_scaled_standard_projection_ops(
    cbf_cfg: QuadrotorBCBFConfig,
    *,
    runtime: Any | None = None,
    solve_float64: bool = False,
) -> BCBFProjectionOps:
    projector = QuadrotorScaledSlackStandardProjector(
        cbf_cfg, runtime=runtime, solve_float64=solve_float64
    )

    def project_with_info(context: Array, u_ref: Array):
        return projector.solve_batch_with_info(context[..., :10], u_ref)

    def project(context: Array, u_ref: Array):
        u, slack, _, _ = project_with_info(context, u_ref)
        return u, slack

    def backup_policy(context: Array):
        return jax.vmap(projector.runtime.backup_policy_fn)(context[..., :10])

    return BCBFProjectionOps(project_with_info=project_with_info, project=project, backup_policy=backup_policy)


def make_scaled_ue_projection_ops(
    cbf_cfg: QuadrotorBCBFConfig,
    ue_cfg: ExperimentalUEConfig,
    *,
    observer_warmup_sec: float,
    runtime: Any | None = None,
    solve_float64: bool = False,
) -> BCBFProjectionOps:
    """Use scaled standard bCBF during warm-up, then scaled UE-bCBF."""

    standard = QuadrotorScaledSlackStandardProjector(
        cbf_cfg, runtime=runtime, solve_float64=solve_float64
    )
    ue = QuadrotorScaledSlackUEProjector(
        cbf_cfg,
        ue_cfg,
        runtime=standard.runtime,
        solve_float64=solve_float64,
    )
    warmup = jnp.asarray(float(observer_warmup_sec), dtype=jnp.float32)

    def solve_single(context: Array, u_ref: Array):
        x = context[:10]
        d_hat = context[10:13]
        e_bar = context[13]
        elapsed = context[14]
        return jax.lax.cond(
            elapsed >= warmup.astype(elapsed.dtype),
            lambda _: ue._solve_single_with_info(x, u_ref, d_hat, e_bar),
            lambda _: standard._solve_single_with_info(x, u_ref),
            operand=None,
        )

    solve_batch = jax.jit(jax.vmap(solve_single, in_axes=(0, 0)))

    def project_with_info(context: Array, u_ref: Array):
        return solve_batch(context, u_ref)

    def project(context: Array, u_ref: Array):
        u, slack, _, _ = solve_batch(context, u_ref)
        return u, slack

    def backup_policy(context: Array):
        return jax.vmap(standard.runtime.backup_policy_fn)(context[..., :10])

    return BCBFProjectionOps(project_with_info=project_with_info, project=project, backup_policy=backup_policy)


__all__ = [
    "QuadrotorScaledSlackStandardProjector",
    "QuadrotorScaledSlackUEProjector",
    "make_scaled_standard_projection_ops",
    "make_scaled_ue_projection_ops",
]
