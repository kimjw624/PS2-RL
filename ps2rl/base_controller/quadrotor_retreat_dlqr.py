"""Runway task: the retreat-at-altitude LQR (base controller) and the tube's 8-D metric chart.

``QuadrotorRetreatDLQR`` - the base-set controller. 7-D error

    e(x) = (p_z - z_hold, v_x, v_y + v_ret, v_z, phi_x, phi_y, phi_z),
    phi = 2 sgn(q_err,w) q_err,xyz,  q_err = conj(q)      (identity attitude, yaw 0),

linearized about level flight at constant velocity (0, -v_ret, 0) - the same model as the
powerloop hover LQR (``QuadrotorDLQR``) because the rigid body without drag does not care
about a constant velocity. p_x and p_y are free.

``RunwayMetricChart`` - not a controller: the DARE matrix of the 8-D system with p_y added,
e8 = (p_y - y_edge, e), defines the P-metric of the UE tube. p_y has to be in the metric
because the runway barrier depends on it and the learned backup feeds it back; p_x does not
(nothing depends on it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.base_controller.base_controller import DiscreteLQR, _solve_discrete_are_scipy, euler_discretize
from ps2rl.utils.quaternion import normalize_quaternion_batch, quaternion_conjugate_batch

_Q7 = ("lqr_q_z", "lqr_q_vx", "lqr_q_vy", "lqr_q_vz", "lqr_q_thetax", "lqr_q_thetay", "lqr_q_thetaz")
_R = ("lqr_r_a_cmd", "lqr_r_omega_x", "lqr_r_omega_y", "lqr_r_omega_z")


def _lin7(g: float) -> tuple[np.ndarray, np.ndarray]:
    a = np.zeros((7, 7))
    a[0, 3] = 1.0  # z_dot = v_z
    a[1, 5] = -g  # v_x_dot = -g phi_y
    a[2, 4] = g  # v_y_dot = +g phi_x
    b = np.zeros((7, 4))
    b[3, 0] = 1.0  # v_z_dot = delta a
    b[4, 1] = -1.0  # phi_dot = -omega
    b[5, 2] = -1.0
    b[6, 3] = -1.0
    return a, b


def _lin8(g: float) -> tuple[np.ndarray, np.ndarray]:
    a7, b7 = _lin7(g)
    a = np.zeros((8, 8))
    a[1:, 1:] = a7
    a[0, 3] = 1.0  # y_dot = v_y
    b = np.zeros((8, 4))
    b[1:] = b7
    return a, b


def _phi(x: jax.Array) -> jax.Array:
    q = normalize_quaternion_batch(x[..., 6:10])
    q_err = quaternion_conjugate_batch(q)
    sign = jnp.where(q_err[..., 0] >= 0.0, 1.0, -1.0).astype(x.dtype)
    return 2.0 * sign[..., None] * q_err[..., 1:4]


def _quat_from_phi(phi: jax.Array, floor: float = 0.0) -> jax.Array:
    half = -0.5 * phi
    qw = jnp.sqrt(jnp.clip(1.0 - jnp.sum(half * half, axis=-1, keepdims=True), floor, 1.0))
    return jnp.concatenate([qw, half], axis=-1)


@dataclass(frozen=True)
class QuadrotorRetreatDLQR(DiscreteLQR):
    z_hold: float = 6.0
    v_ret: float = 1.8

    @classmethod
    def from_config(cls, cfg: Any) -> "QuadrotorRetreatDLQR":
        g = float(cfg.gravity)
        a_d, b_d = euler_discretize(*_lin7(g), float(cfg.dt))
        w = float(cfg.omega_max)
        return cls(
            a_d=tuple(tuple(r) for r in a_d.tolist()), b_d=tuple(tuple(r) for r in b_d.tolist()),
            q_diag=tuple(float(getattr(cfg, k)) for k in _Q7), r_diag=tuple(float(getattr(cfg, k)) for k in _R),
            u_star=(g, 0.0, 0.0, 0.0), u_low=(float(cfg.a_cmd_min), -w, -w, -w),
            u_high=(float(cfg.a_cmd_max), w, w, w), z_hold=float(cfg.z_hold), v_ret=float(cfg.v_ret),
        )

    def error_state(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x)
        off = jnp.asarray([self.z_hold, 0.0, -self.v_ret, 0.0], dtype=x.dtype)
        return jnp.concatenate([x[..., 2:6] - off, _phi(x)], axis=-1)

    def state_from_error(self, e: jax.Array, p_xy: tuple[float, float] = (0.0, 0.0)) -> jax.Array:
        """Inverse chart (q_w >= 0) with the free coordinates p_x, p_y set to ``p_xy``."""
        e = jnp.asarray(e)
        off = jnp.asarray([self.z_hold, 0.0, -self.v_ret, 0.0], dtype=e.dtype)
        zv = e[..., 0:4] + off
        pxy = jnp.broadcast_to(jnp.asarray(p_xy, dtype=e.dtype), zv.shape[:-1] + (2,))
        return jnp.concatenate([pxy, zv, _quat_from_phi(e[..., 4:7])], axis=-1)

    def action(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x)
        u = jnp.asarray(self.u_star, x.dtype) - jnp.einsum("ij,...j->...i", jnp.asarray(self.k_matrix, x.dtype),
                                                           self.error_state(x))
        return jnp.clip(u, jnp.asarray(self.u_low, x.dtype), jnp.asarray(self.u_high, x.dtype))

    def p_matrix_f64(self) -> np.ndarray:
        p = _solve_discrete_are_scipy(np.asarray(self.a_d), np.asarray(self.b_d), np.diag(self.q_diag), np.diag(self.r_diag))
        return 0.5 * (p.real + p.real.T)

    def k_matrix_f64(self) -> np.ndarray:
        a, b, r = np.asarray(self.a_d), np.asarray(self.b_d), np.diag(self.r_diag)
        p = self.p_matrix_f64()
        return np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)


@dataclass(frozen=True)
class RunwayMetricChart:
    """8-D chart e8 = (p_y - y_edge, p_z - z_hold, v_x, v_y + v_ret, v_z, phi) and its DARE metric P8."""

    y_edge: float
    z_hold: float
    v_ret: float
    p: np.ndarray  # (8, 8) float64

    @classmethod
    def from_config(cls, cfg: Any) -> "RunwayMetricChart":
        a_d, b_d = euler_discretize(*_lin8(float(cfg.gravity)), float(cfg.dt))
        q = np.diag([float(cfg.metric_q_y)] + [float(getattr(cfg, k)) for k in _Q7])
        r = np.diag([float(getattr(cfg, k)) for k in _R])
        p = _solve_discrete_are_scipy(a_d, b_d, q, r)
        return cls(y_edge=float(cfg.y_edge), z_hold=float(cfg.z_hold), v_ret=float(cfg.v_ret), p=0.5 * (p.real + p.real.T))

    def error_state(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x)
        off = jnp.asarray([self.y_edge, self.z_hold, 0.0, -self.v_ret, 0.0], dtype=x.dtype)
        return jnp.concatenate([x[..., 1:6] - off, _phi(x)], axis=-1)

    def state_from_error(self, e: jax.Array) -> jax.Array:
        e = jnp.asarray(e)
        off = jnp.asarray([self.y_edge, self.z_hold, 0.0, -self.v_ret, 0.0], dtype=e.dtype)
        yzv = e[..., 0:5] + off
        # q_w floored at 1e-3: the chart is singular at 180 deg (q_w = 0), where its Jacobian - and with
        # it the tube growth and the contraction penalty's gradient - would be infinite
        return jnp.concatenate([jnp.zeros_like(yzv[..., :1]), yzv, _quat_from_phi(e[..., 5:8], 1e-6)], axis=-1)

    def __hash__(self) -> int:  # numpy field; identity hash is enough (used as a static config)
        return id(self)


__all__ = ["QuadrotorRetreatDLQR", "RunwayMetricChart"]
