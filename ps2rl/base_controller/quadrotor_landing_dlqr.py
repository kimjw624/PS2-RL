"""Quadrotor hover LQR above a landing pad (9-D error, horizontal position included).

The powerloop base set uses the 7-D error [z - z_des, v, 2 q_err] and is therefore
unbounded in (x, y). A bounded safe set such as the approach cone cannot contain a
horizontally unbounded set (landing note, Lemma 2), so the landing base controller
regulates horizontal position too:

    e(x) = (dp_x, dp_y, zeta - z_des, v_x, v_y, v_z, phi_x, phi_y, phi_z) in R^9,

with dp = p_xy - p_pad, zeta = z - z_pad and phi = 2 sgn(q_err,w) q_err,xyz, where
q_err = conj(q) (error w.r.t. the identity attitude, yaw_des = 0). The sign convention
and the Euler discretization match ``QuadrotorDLQR`` exactly; only the two
position-integrator rows are new.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.base_controller.base_controller import DiscreteLQR, euler_discretize
from ps2rl.utils.quaternion import normalize_quaternion_batch, quaternion_conjugate_batch

_LQR_Q_KEYS = (
    "lqr_q_x",
    "lqr_q_y",
    "lqr_q_z",
    "lqr_q_vx",
    "lqr_q_vy",
    "lqr_q_vz",
    "lqr_q_thetax",
    "lqr_q_thetay",
    "lqr_q_thetaz",
)
_LQR_R_KEYS = ("lqr_r_a_cmd", "lqr_r_omega_x", "lqr_r_omega_y", "lqr_r_omega_z")

# Defaults: the powerloop weights for the shared rows, and q_x = q_y = q_z.
LANDING_LQR_DEFAULTS: dict[str, float] = {
    "lqr_q_x": 1.0,
    "lqr_q_y": 1.0,
    "lqr_q_z": 1.0,
    "lqr_q_vx": 0.16,
    "lqr_q_vy": 0.16,
    "lqr_q_vz": 0.4,
    "lqr_q_thetax": 0.8,
    "lqr_q_thetay": 0.8,
    "lqr_q_thetaz": 0.16,
    "lqr_r_a_cmd": 0.02,
    "lqr_r_omega_x": 0.012,
    "lqr_r_omega_y": 0.012,
    "lqr_r_omega_z": 0.004,
}

# Error-vector index layout (used by the certificate and the samplers).
IDX_POS = slice(0, 3)
IDX_XY = slice(0, 2)
IDX_Z = 2
IDX_VEL = slice(3, 6)
IDX_ATT = slice(6, 9)


@dataclass(frozen=True)
class QuadrotorLandingDLQR(DiscreteLQR):
    """Discrete-time hover LQR at (p_pad, z_pad + z_des) in 9-D error coordinates."""

    z_des: float = 2.0
    pad_x: float = 0.0
    pad_y: float = 0.0
    pad_z: float = 0.0

    @staticmethod
    def linearization_error_coords(gravity: float) -> tuple[np.ndarray, np.ndarray]:
        """Continuous (A, B) about hover in e = (dp, zeta - z_des, v, phi).

        Four integrator chains (landing note, Lemma 3):
          x -> v_x -> phi_y <- omega_y,   y -> v_y -> phi_x <- omega_x,
          z -> v_z <- delta a,            phi_z <- omega_z.
        """
        g = float(gravity)
        a = np.zeros((9, 9), dtype=np.float64)
        a[0, 3] = 1.0  # d/dt dp_x = v_x
        a[1, 4] = 1.0  # d/dt dp_y = v_y
        a[2, 5] = 1.0  # d/dt zeta = v_z
        a[3, 7] = -g  # v_x_dot = -g phi_y   (phi is the error-quaternion angle, phi = -theta)
        a[4, 6] = g  # v_y_dot = +g phi_x
        b = np.zeros((9, 4), dtype=np.float64)
        b[5, 0] = 1.0  # v_z_dot = delta a
        b[6, 1] = -1.0  # phi_dot = -omega
        b[7, 2] = -1.0
        b[8, 3] = -1.0
        return a, b

    @classmethod
    def _build(
        cls,
        *,
        dt: float,
        gravity: float,
        z_des: float,
        pad: tuple[float, float, float],
        q_diag: tuple[float, ...],
        r_diag: tuple[float, ...],
        a_cmd_min: float,
        a_cmd_max: float,
        omega_max: float,
    ) -> "QuadrotorLandingDLQR":
        a_raw, b_raw = cls.linearization_error_coords(gravity)
        a_d, b_d = euler_discretize(a_raw, b_raw, dt)
        omega_max = float(omega_max)
        return cls(
            a_d=tuple(tuple(row) for row in a_d.tolist()),
            b_d=tuple(tuple(row) for row in b_d.tolist()),
            q_diag=tuple(float(v) for v in q_diag),
            r_diag=tuple(float(v) for v in r_diag),
            u_star=(float(gravity), 0.0, 0.0, 0.0),
            u_low=(float(a_cmd_min), -omega_max, -omega_max, -omega_max),
            u_high=(float(a_cmd_max), omega_max, omega_max, omega_max),
            z_des=float(z_des),
            pad_x=float(pad[0]),
            pad_y=float(pad[1]),
            pad_z=float(pad[2]),
        )

    @classmethod
    def from_config(cls, cfg: Any) -> "QuadrotorLandingDLQR":
        """Build from any object exposing the landing config fields.

        Missing LQR weights fall back to ``LANDING_LQR_DEFAULTS``.
        """

        def _get(key: str, default: float | None = None) -> float:
            if hasattr(cfg, key):
                return float(getattr(cfg, key))
            if default is None:
                raise AttributeError(f"landing config is missing '{key}'")
            return float(default)

        return cls._build(
            dt=_get("dt"),
            gravity=_get("gravity"),
            z_des=_get("z_des"),
            pad=(_get("pad_x", 0.0), _get("pad_y", 0.0), _get("pad_z", 0.0)),
            q_diag=tuple(_get(k, LANDING_LQR_DEFAULTS[k]) for k in _LQR_Q_KEYS),
            r_diag=tuple(_get(k, LANDING_LQR_DEFAULTS[k]) for k in _LQR_R_KEYS),
            a_cmd_min=_get("a_cmd_min"),
            a_cmd_max=_get("a_cmd_max"),
            omega_max=_get("omega_max"),
        )

    # ------------------------------------------------------------------ maps
    def error_state(self, x: jax.Array) -> jax.Array:
        """e(x) = (dp_x, dp_y, zeta - z_des, v, phi) with phi = 2 sgn(q_err,w) q_err,xyz."""
        x_arr = jnp.asarray(x)
        q = normalize_quaternion_batch(x_arr[..., 6:10])
        q_err = quaternion_conjugate_batch(q)
        sign_term = jnp.where(q_err[..., 0] >= 0.0, 1.0, -1.0).astype(x_arr.dtype)
        phi = 2.0 * sign_term[..., None] * q_err[..., 1:4]
        target = jnp.asarray(
            [self.pad_x, self.pad_y, self.pad_z + self.z_des], dtype=x_arr.dtype
        )
        return jnp.concatenate([x_arr[..., 0:3] - target, x_arr[..., 3:6], phi], axis=-1)

    def state_from_error(self, e: jax.Array) -> jax.Array:
        """Inverse chart on {||phi|| < 2}: e -> x with q_w >= 0.

        phi = 2 q_err,xyz and q_err = conj(q), so q_xyz = -phi / 2.
        """
        e_arr = jnp.asarray(e)
        half = -0.5 * e_arr[..., 6:9]
        qw = jnp.sqrt(jnp.clip(1.0 - jnp.sum(half * half, axis=-1, keepdims=True), 0.0, 1.0))
        target = jnp.asarray(
            [self.pad_x, self.pad_y, self.pad_z + self.z_des], dtype=e_arr.dtype
        )
        pos = e_arr[..., 0:3] + target
        return jnp.concatenate([pos, e_arr[..., 3:6], qw, half], axis=-1)

    def action(self, x: jax.Array) -> jax.Array:
        """pi_B(x) = u* - K e(x), clipped to the action box."""
        x_arr = jnp.asarray(x)
        u_eq = jnp.asarray(list(self.u_star), dtype=x_arr.dtype)
        k_mat = jnp.asarray(self.k_matrix, dtype=x_arr.dtype)
        u = u_eq - jnp.einsum("ij,...j->...i", k_mat, self.error_state(x_arr))
        low = jnp.asarray(list(self.u_low), dtype=x_arr.dtype)
        high = jnp.asarray(list(self.u_high), dtype=x_arr.dtype)
        return jnp.clip(u, low, high)

    def p_matrix_f64(self) -> np.ndarray:
        """Re-solve P at f64 (the stored ``p_matrix`` is f32)."""
        from ps2rl.base_controller.base_controller import _solve_discrete_are_scipy

        a_d = np.asarray(self.a_d, dtype=np.float64)
        b_d = np.asarray(self.b_d, dtype=np.float64)
        p = _solve_discrete_are_scipy(
            a_d, b_d, np.diag(np.asarray(self.q_diag)), np.diag(np.asarray(self.r_diag))
        )
        return 0.5 * (p.real + p.real.T)

    def k_matrix_f64(self) -> np.ndarray:
        a_d = np.asarray(self.a_d, dtype=np.float64)
        b_d = np.asarray(self.b_d, dtype=np.float64)
        r = np.diag(np.asarray(self.r_diag, dtype=np.float64))
        p = self.p_matrix_f64()
        return np.linalg.solve(r + b_d.T @ p @ b_d, b_d.T @ p @ a_d)


__all__ = [
    "IDX_ATT",
    "IDX_POS",
    "IDX_VEL",
    "IDX_XY",
    "IDX_Z",
    "LANDING_LQR_DEFAULTS",
    "QuadrotorLandingDLQR",
]
