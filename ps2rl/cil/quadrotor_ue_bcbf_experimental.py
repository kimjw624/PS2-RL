"""Experimental UE-bCBF projection for Phase-2 quadrotor evaluation.

This module intentionally does not replace the production PS2 backup-CBF.
It is an evaluation-first implementation used to answer a practical question:
can the uncertainty-estimation backup-CBF idea improve Phase-2 behavior before
we finish the nonlinear tube proof?

The implementation uses

* the learned/analytic PS2 composed backup unchanged,
* the frozen current disturbance estimate ``d_hat`` over the backup rollout,
* exact discrete Phi/Theta and 9-D physical tangent Jacobians,
* the disturbance-channel generator tube used in the Step-20/21 diagnostics,
* an explicit empirical inflation factor (default 1.2),
* the analytic observer-error term in the UE derivative row, and
* the exact quadratic structure of the terminal LQR barrier.

It is deliberately marked EXPERIMENTAL.  The generator tube is still a
first-order centerline tube and the hard SA/LQR handoff is not saltation-
corrected here.  Therefore this code is for Phase-2 experiments, not yet a
formal nonlinear safety certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import qpax

from ps2rl.backup_policy.quadrotor_learned_backup import load_learned_quadrotor_backup_policy
from ps2rl.base_controller.quadrotor_dlqr import QuadrotorDLQR
from ps2rl.cil.backup_cbf import _build_backup_cbf_qp_from_rows
from ps2rl.cil.quadrotor_backup_cbf import QuadrotorBCBFConfig, make_backup_runtime
from ps2rl.cil.quadrotor_ue_discrete_sensitivity import tangent_basis, tangent_left_inverse
from ps2rl.cil.quadrotor_ue_rollout import (
    estimated_disturbance_backup_dynamics,
    quadrotor_disturbance_injection_matrix,
)

Array = jax.Array


@dataclass(frozen=True)
class ExperimentalUEConfig:
    """Extra knobs for the empirical Phase-2 UE projection."""

    delta_d: float = 2.0
    delta_v: float = 2.0 * np.pi * 0.5 * 2.0
    observer_lambda: float = 20.0
    tube_scale: float = 1.2
    rho_scale: float = 1.0
    safe_rho_scale: float = 1.0
    terminal_rho_scale: float = 1.0
    terminal_mode: str = "quadratic"
    compute_radius_diagnostics: bool = True

    def __post_init__(self) -> None:
        for name in (
            "delta_d",
            "delta_v",
            "observer_lambda",
            "tube_scale",
            "rho_scale",
            "safe_rho_scale",
            "terminal_rho_scale",
        ):
            value = float(getattr(self, name))
            if not isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
            if name == "observer_lambda" and value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
            if name in {"tube_scale", "rho_scale", "safe_rho_scale", "terminal_rho_scale"} and value < 0.0:
                raise ValueError(f"{name} must be nonnegative, got {value}")
            if name in {"delta_d", "delta_v"} and value < 0.0:
                raise ValueError(f"{name} must be nonnegative, got {value}")
        mode = str(self.terminal_mode).strip().lower()
        if mode not in {"quadratic", "nominal"}:
            raise ValueError(f"terminal_mode must be 'quadratic' or 'nominal', got {self.terminal_mode!r}")
        object.__setattr__(self, "terminal_mode", mode)
        object.__setattr__(self, "compute_radius_diagnostics", bool(self.compute_radius_diagnostics))


def _controller_for_cfg(cfg: QuadrotorBCBFConfig) -> QuadrotorDLQR:
    mode = str(cfg.backup_policy_mode).strip().lower()
    if mode == "learned":
        learned = load_learned_quadrotor_backup_policy(cfg.learned_backup_policy_path)
        payload = learned.metadata.get("lqr_config")
        if not isinstance(payload, dict):
            raise KeyError("Learned backup policy is missing metadata['lqr_config']")
        return QuadrotorDLQR.from_payload(payload, fallback=cfg)
    return QuadrotorDLQR.from_config(cfg)


def _spectral_norm_9x3_batch(mats: Array) -> Array:
    """Spectral norms of a batch (..., 9, 3), using only 3x3 eigensolves."""

    mats = jnp.asarray(mats)
    gram = jnp.einsum("...ik,...il->...kl", mats, mats)
    eigvals = jnp.linalg.eigvalsh(gram)
    return jnp.sqrt(jnp.maximum(eigvals[..., -1], jnp.asarray(0.0, dtype=mats.dtype)))


def rollout_ue_quantities(
    x0: Array,
    d_hat: Array,
    e_bar: Array,
    cbf_cfg: QuadrotorBCBFConfig,
    ue_cfg: ExperimentalUEConfig,
    runtime: Any,
) -> Tuple[Array, Array, Array, Array, Dict[str, Array]]:
    """Roll out the frozen-estimate backup and empirical channel tube.

    Returns nodes ``xs, Phi, Theta, R`` plus generator-derived barrier
    support margins. ``R`` is the same scalar first-order disturbance-channel
    bound used by the successful diagnostic, multiplied by
    ``ue_cfg.tube_scale``.  The actual UE constraints use tighter support
    functions of the same generators rather than collapsing everything to R.
    """

    x0 = jnp.asarray(x0)
    d_hat = jnp.asarray(d_hat, dtype=x0.dtype)
    e_bar = jnp.asarray(e_bar, dtype=x0.dtype)
    dt = jnp.asarray(cbf_cfg.dt, dtype=x0.dtype)
    n_steps = int(cbf_cfg.num_steps)
    delta_v = jnp.asarray(ue_cfg.delta_v, dtype=x0.dtype)
    tube_scale = jnp.asarray(ue_cfg.tube_scale, dtype=x0.dtype)

    if x0.shape != (10,):
        raise ValueError(f"x0 must have shape (10,), got {x0.shape}")
    if d_hat.shape != (3,):
        raise ValueError(f"d_hat must have shape (3,), got {d_hat.shape}")

    def discrete_step(z: Array, d: Array) -> Array:
        dz = estimated_disturbance_backup_dynamics(z, d, runtime)
        return runtime.postprocess_rollout_state_fn(z + dt * dz)

    jac_step_x = jax.jacfwd(discrete_step, argnums=0)
    jac_step_d = jax.jacfwd(discrete_step, argnums=1)

    # Fixed-size storage makes the Minkowski generator propagation compatible
    # with lax.scan.  At step k, entries [0:k] are the active disturbance
    # generators and the rest are exactly zero.
    gen0 = jnp.zeros((n_steps, 9, 3), dtype=x0.dtype)
    phi0 = jnp.eye(10, dtype=x0.dtype)
    theta0 = jnp.zeros((10, 3), dtype=x0.dtype)

    def scan_step(carry, k):
        x, phi, theta, generators = carry
        x_next = discrete_step(x, d_hat)
        f10 = jac_step_x(x, d_hat)
        g10 = jac_step_d(x, d_hat)

        b0 = tangent_basis(x)
        bp1 = tangent_left_inverse(x_next)
        f9 = bp1 @ f10 @ b0
        g9 = bp1 @ g10

        phi_next = f10 @ phi
        theta_next = f10 @ theta + g10

        generators_next = jnp.einsum("ij,njk->nik", f9, generators)
        tau_end = (k.astype(x0.dtype) + 1.0) * dt
        q_end = e_bar + delta_v * tau_end
        generators_next = generators_next.at[k].set(g9 * q_end)
        if ue_cfg.compute_radius_diagnostics:
            radius_next = tube_scale * jnp.sum(_spectral_norm_9x3_batch(generators_next))
        else:
            # The full-state spectral-norm radius is diagnostic-only. The actual
            # UE constraints below still use the exact generator supports for the
            # ceiling and terminal quadratic barrier. Skipping this during
            # training removes repeated eigensolves without changing the QP rows.
            radius_next = jnp.asarray(0.0, dtype=x0.dtype)
        # Direct support of the ceiling output delta p_z.  This is always no
        # larger than the full physical-state radius and is the margin that is
        # actually relevant for h_S = z_max - p_z.
        safe_eps_next = tube_scale * jnp.sum(jnp.linalg.norm(generators_next[:, 2, :], axis=-1))

        return (x_next, phi_next, theta_next, generators_next), (
            x_next,
            phi_next,
            theta_next,
            radius_next,
            safe_eps_next,
        )

    (_, _, _, generators_final), (xs_tail, phi_tail, theta_tail, r_tail, safe_eps_tail) = jax.lax.scan(
        scan_step,
        (x0, phi0, theta0, gen0),
        jnp.arange(n_steps, dtype=jnp.int32),
    )

    xs = jnp.concatenate([x0[None, :], xs_tail], axis=0)
    phis = jnp.concatenate([phi0[None, :, :], phi_tail], axis=0)
    thetas = jnp.concatenate([theta0[None, :, :], theta_tail], axis=0)
    radii = jnp.concatenate([jnp.zeros((1,), dtype=x0.dtype), r_tail], axis=0)
    safe_eps = jnp.concatenate([jnp.zeros((1,), dtype=x0.dtype), safe_eps_tail], axis=0)
    info = {
        "all_finite": (
            jnp.all(jnp.isfinite(xs))
            & jnp.all(jnp.isfinite(phis))
            & jnp.all(jnp.isfinite(thetas))
            & jnp.all(jnp.isfinite(radii))
        ),
        "max_radius": jnp.max(radii),
        "terminal_radius": radii[-1],
        "max_safe_output_margin": jnp.max(safe_eps),
        "terminal_safe_output_margin": safe_eps[-1],
        "safe_output_margins": safe_eps,
        "terminal_generators": generators_final,
        "max_abs_phi": jnp.max(jnp.abs(phis)),
        "max_abs_theta": jnp.max(jnp.abs(thetas)),
    }
    return xs, phis, thetas, radii, info


def build_ue_rows_with_info(
    x: Array,
    d_hat: Array,
    e_bar: Array,
    cbf_cfg: QuadrotorBCBFConfig,
    ue_cfg: ExperimentalUEConfig,
    runtime: Any,
    controller: QuadrotorDLQR,
) -> Tuple[Array, Array, Dict[str, Array]]:
    """Build the experimental UE backup-CBF rows ``A u <= b``."""

    xs, phis, thetas, radii, rollout_info = rollout_ue_quantities(
        x,
        d_hat,
        e_bar,
        cbf_cfg,
        ue_cfg,
        runtime,
    )
    f0, g0 = runtime.control_affine_terms_fn(x)
    ed = quadrotor_disturbance_injection_matrix(dtype=x.dtype)
    f0_hat = f0 + ed @ d_hat
    lam = jnp.asarray(ue_cfg.observer_lambda, dtype=x.dtype)
    lambda_mat = lam * jnp.eye(3, dtype=x.dtype)

    safe_eps = rollout_info["safe_output_margins"]

    def per_node(xi, phi_i, theta_i, safe_eps_i):
        h_c, dh_c = runtime.safe_set_values_and_grads_fn(xi)
        phi_g0 = phi_i @ g0
        phi_f0_hat = phi_i @ f0_hat
        a = -(dh_c @ phi_g0)

        f_pi_hat_i = estimated_disturbance_backup_dynamics(xi, d_hat, runtime)
        flow_term = phi_f0_hat - f_pi_hat_i if cbf_cfg.include_relative_time_term else phi_f0_hat

        # h_S = z_max - p_z.  Use the direct support of the channel
        # generators in the vertical-position output instead of the much
        # larger full-state radius.  The support values are precomputed by the
        # rollout and indexed by the corresponding node below.
        h_robust = h_c - safe_eps_i

        ue_phi = phi_i @ ed
        ue_theta = theta_i @ lambda_mat
        safe_phi_vec = dh_c @ ue_phi
        safe_theta_vec = dh_c @ ue_theta
        rho_phi_raw = e_bar * jnp.linalg.norm(safe_phi_vec, axis=-1)
        rho_theta_raw = e_bar * jnp.linalg.norm(safe_theta_vec, axis=-1)
        rho_raw = e_bar * jnp.linalg.norm(safe_phi_vec + safe_theta_vec, axis=-1)
        rho = (
            jnp.asarray(ue_cfg.rho_scale * ue_cfg.safe_rho_scale, dtype=x.dtype)
            * rho_raw
        )
        b = cbf_cfg.alpha * h_robust + dh_c @ flow_term - rho
        return a, b, h_robust, rho, rho_raw, rho_phi_raw, rho_theta_raw

    (
        a_seq,
        b_seq,
        h_rob_seq,
        rho_seq,
        rho_raw_seq,
        rho_phi_raw_seq,
        rho_theta_raw_seq,
    ) = jax.vmap(per_node, in_axes=(0, 0, 0, 0))(
        xs,
        phis,
        thetas,
        safe_eps,
    )
    a_rows = a_seq.reshape((-1, runtime.action_dim))
    b_rows = b_seq.reshape((-1,))

    x_t = xs[-1]
    phi_t = phis[-1]
    theta_t = thetas[-1]
    radius_t = radii[-1]
    h_b, dh_b = runtime.base_set_values_and_grads_fn(x_t)

    e_t = controller.error_state(x_t)
    p_mat = jnp.asarray(controller.p_matrix, dtype=x.dtype)
    pe = p_mat @ e_t

    # Map each final physical-state generator into the exact reduced LQR error
    # coordinates.  For delta_e = sum_j M_j w_j, ||w_j||<=1,
    #
    #   2 e^T P delta_e <= 2 sum_j ||M_j^T P e||,
    #   delta_e^T P delta_e <= (sum_j ||P^(1/2) M_j||_2)^2.
    #
    # This exploits the known quadratic h_B and is substantially tighter than
    # lambda_max(P) * R_9^2 when R_9 contains irrelevant state directions.
    j_e10 = jax.jacfwd(controller.error_state)(x_t)
    c_e9 = j_e10 @ tangent_basis(x_t)
    terminal_generators = rollout_info["terminal_generators"] * jnp.asarray(ue_cfg.tube_scale, dtype=x.dtype)
    e_generators = jnp.einsum("ij,njk->nik", c_e9, terminal_generators)
    linear_support = 2.0 * jnp.sum(jnp.linalg.norm(jnp.einsum("nji,j->ni", e_generators, pe), axis=-1))

    p_evals, p_evecs = jnp.linalg.eigh(p_mat)
    p_sqrt = (p_evecs * jnp.sqrt(jnp.maximum(p_evals, 0.0))[None, :]) @ p_evecs.T
    p_half_generators = jnp.einsum("ij,njk->nik", p_sqrt, e_generators)
    quadratic_support_radius = jnp.sum(_spectral_norm_9x3_batch(p_half_generators))
    quadratic_support = quadratic_support_radius * quadratic_support_radius
    eps_quad = linear_support + quadratic_support
    eps_quad_used = eps_quad if ue_cfg.terminal_mode == "quadratic" else jnp.asarray(0.0, dtype=x.dtype)
    h_b_robust = h_b - eps_quad_used

    phi_t_g0 = phi_t @ g0
    phi_t_f0_hat = phi_t @ f0_hat
    ue_phi_t = phi_t @ ed
    ue_theta_t = theta_t @ lambda_mat
    terminal_phi_vec = dh_b @ ue_phi_t
    terminal_theta_vec = dh_b @ ue_theta_t
    terminal_rho_phi_raw = e_bar * jnp.linalg.norm(terminal_phi_vec, axis=-1)
    terminal_rho_theta_raw = e_bar * jnp.linalg.norm(terminal_theta_vec, axis=-1)
    terminal_rho_raw = e_bar * jnp.linalg.norm(terminal_phi_vec + terminal_theta_vec, axis=-1)
    rho_t = (
        jnp.asarray(ue_cfg.rho_scale * ue_cfg.terminal_rho_scale, dtype=x.dtype)
        * terminal_rho_raw
    )
    terminal_alignment = jnp.sum(terminal_phi_vec * terminal_theta_vec, axis=-1) / jnp.maximum(
        jnp.linalg.norm(terminal_phi_vec, axis=-1) * jnp.linalg.norm(terminal_theta_vec, axis=-1),
        jnp.asarray(1.0e-12, dtype=x.dtype),
    )
    a_term = -(dh_b @ phi_t_g0)
    b_term = cbf_cfg.base_alpha * h_b_robust + dh_b @ phi_t_f0_hat - rho_t

    a_rows = jnp.concatenate([a_rows, a_term], axis=0)
    b_rows = jnp.concatenate([b_rows, b_term], axis=0)
    diag = {
        "all_finite": rollout_info["all_finite"],
        "max_radius": rollout_info["max_radius"],
        "terminal_radius": rollout_info["terminal_radius"],
        "max_safe_output_margin": rollout_info["max_safe_output_margin"],
        "terminal_safe_output_margin": rollout_info["terminal_safe_output_margin"],
        "max_abs_phi": rollout_info["max_abs_phi"],
        "max_abs_theta": rollout_info["max_abs_theta"],
        "min_robust_safe_h": jnp.min(h_rob_seq),
        "max_rho_safe": jnp.max(rho_seq),
        "max_rho_safe_raw": jnp.max(rho_raw_seq),
        "max_rho_safe_phi_raw": jnp.max(rho_phi_raw_seq),
        "max_rho_safe_theta_raw": jnp.max(rho_theta_raw_seq),
        "terminal_h_b": h_b[0],
        "terminal_h_b_robust": h_b_robust[0],
        "terminal_quadratic_margin": eps_quad,
        "terminal_quadratic_margin_used": eps_quad_used,
        "terminal_linear_support": linear_support,
        "terminal_quadratic_support": quadratic_support,
        "terminal_rho": rho_t[0],
        "terminal_rho_raw": terminal_rho_raw[0],
        "terminal_rho_phi_raw": terminal_rho_phi_raw[0],
        "terminal_rho_theta_raw": terminal_rho_theta_raw[0],
        "terminal_rho_phi_theta_alignment": terminal_alignment[0],
        # Row vector a_e such that the realized current observer-error
        # contribution is a_e @ (d_true - d_hat).  The robust UE bound uses
        # e_bar * ||a_e||.  Exposing this vector lets the evaluator compare
        # the realized directional effect against the worst-case norm bound
        # without leaking d_true into the controller.
        "terminal_ue_error_direction": (terminal_phi_vec + terminal_theta_vec)[0],
        "terminal_grad_h_b_norm": jnp.linalg.norm(dh_b[0]),
        "max_abs_b": jnp.max(jnp.abs(b_rows)),
    }
    return a_rows, b_rows, diag


class QuadrotorExperimentalUEProjector:
    """Standalone experimental UE projector; production PS2 is untouched."""

    def __init__(
        self,
        cbf_cfg: QuadrotorBCBFConfig,
        ue_cfg: ExperimentalUEConfig | None = None,
        runtime: Any | None = None,
    ) -> None:
        self.cbf_cfg = cbf_cfg
        self.ue_cfg = ue_cfg if ue_cfg is not None else ExperimentalUEConfig()
        self.runtime = make_backup_runtime(cbf_cfg) if runtime is None else runtime
        self.controller = _controller_for_cfg(cbf_cfg)

        self.solve_single_with_info = jax.jit(self._solve_single_with_info)
        self.solve_batch_with_info = jax.jit(jax.vmap(self._solve_single_with_info, in_axes=(0, 0, 0, 0)))

    @property
    def num_qp_inequalities(self) -> int:
        return int(self.cbf_cfg.num_qp_inequalities)

    def _solve_single_with_info(
        self,
        x: Array,
        u_ref: Array,
        d_hat: Array,
        e_bar: Array,
    ) -> Tuple[Array, Array, Array, Dict[str, Array]]:
        cfg = self.cbf_cfg
        system = self.runtime
        m = int(system.action_dim)

        x = system.sanitize_solve_state_fn(x)
        u_ref = jnp.nan_to_num(u_ref, nan=0.0, posinf=0.0, neginf=0.0)
        action_low = jnp.asarray(system.action_low, dtype=x.dtype)
        action_high = jnp.asarray(system.action_high, dtype=x.dtype)
        u_ref = jnp.clip(u_ref, action_low, action_high)
        d_hat = jnp.asarray(d_hat, dtype=x.dtype)
        e_bar = jnp.maximum(jnp.asarray(e_bar, dtype=x.dtype), 0.0)

        fallback_u = system.backup_policy_fn(x)
        fallback_z = jnp.concatenate([fallback_u, jnp.zeros((1,), dtype=x.dtype)], axis=0)

        a_rows, b_rows, ue_info = build_ue_rows_with_info(
            x,
            d_hat,
            e_bar,
            cfg,
            self.ue_cfg,
            system,
            self.controller,
        )
        residual_u_ref = a_rows @ u_ref - b_rows
        delta_min_u_ref = jnp.maximum(0.0, jnp.max(residual_u_ref))

        q_mat, q_vec, a_eq, b_eq, g, h = _build_backup_cbf_qp_from_rows(
            a_rows,
            b_rows,
            u_ref,
            cfg,
            system,
            dtype=x.dtype,
        )
        inputs_finite = (
            jnp.all(jnp.isfinite(q_mat))
            & jnp.all(jnp.isfinite(q_vec))
            & jnp.all(jnp.isfinite(g))
            & jnp.all(jnp.isfinite(h))
            & ue_info["all_finite"]
        )

        def solve_qp(_: None) -> Array:
            return qpax.solve_qp_primal(
                q_mat,
                q_vec,
                a_eq,
                b_eq,
                g,
                h,
                solver_tol=cfg.solver_tol,
                target_kappa=cfg.target_kappa,
            )

        z_candidate = jax.lax.cond(inputs_finite, solve_qp, lambda _: fallback_z, operand=None)
        z_finite = inputs_finite & jnp.all(jnp.isfinite(z_candidate))
        z_out = jnp.where(z_finite, z_candidate, fallback_z)
        safe_u = jnp.clip(z_out[:m], action_low, action_high)
        slack = jnp.maximum(z_out[m], 0.0)
        use_solver = z_finite

        row_residual = a_rows @ safe_u - b_rows - slack
        info = {
            **ue_info,
            "inputs_finite": inputs_finite,
            "z_finite": z_finite,
            "delta_min_u_ref": delta_min_u_ref,
            "max_positive_row_residual": jnp.max(jnp.maximum(row_residual, 0.0)),
            "u_ref_minus_u_safe_norm": jnp.linalg.norm(u_ref - safe_u),
            "d_hat_norm": jnp.linalg.norm(d_hat),
            "e_bar": e_bar,
        }
        return safe_u, slack, use_solver, info


__all__ = [
    "ExperimentalUEConfig",
    "QuadrotorExperimentalUEProjector",
    "build_ue_rows_with_info",
    "rollout_ue_quantities",
]
