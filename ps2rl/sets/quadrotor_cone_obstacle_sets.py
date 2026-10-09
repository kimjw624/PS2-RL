"""Safe set and base sets of the cone-obstacle task (``envs/quadrotor_cone_obstacle_config``).

Safe set: outside an apex-down funnel and above the ground,

    h_cone   = sqrt(r^2 + eps^2) - eps - tan(theta) (z - tip_z),   h_ground = z - ground_z,

where ``ground_z`` here is the lower bound: the config's ground plus its ``min_altitude``.

Base sets (see the config's docstring):

* ``band``     ``EllipsoidBaseSet`` of the paper's 7-D ``QuadrotorDLQR`` at the band height
* ``points``   ``PointsBaseSet``: union of 9-D hover ellipsoids at given hover points
* ``anywhere`` ``AnywhereBaseSet``: "stop wherever it is clear". The base controller is a
               6-D LQR on velocity and attitude only (``BrakeDLQR``); B holds the states it
               can stop from with the whole stopping transient clear of the funnel and the
               ground.

Every base set provides ``action(x)``, its base controller (the backup policy when no
learned safe-arrival policy is used): the 7-D LQR for the band; for a union of points the
LQR of the member with the smallest Lyapunov value; for "anywhere" the brake inside B and,
outside B, a 9-D LQR toward the nearest admissible rest point.

``m_distance`` is the distance from a point to the funnel in the metric of an ellipsoid's
position projection; it decides whether that ellipsoid fits outside the funnel (the funnel
is convex, so this is an exact convex distance, evaluated on a dense boundary sample).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import radians, sqrt, tan
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.base_controller.base_controller import DiscreteLQR, euler_discretize
from ps2rl.base_controller.quadrotor_dlqr import QuadrotorDLQR
from ps2rl.base_controller.quadrotor_landing_dlqr import QuadrotorLandingDLQR
from ps2rl.sets.base_sets import BaseSet, EllipsoidBaseSet, _softplus
from ps2rl.sets.safe_sets import SafeSet
from ps2rl.utils.quaternion import normalize_quaternion_batch, quaternion_conjugate_batch

CONSTRAINT_NAMES = ("cone", "ground")
_BOUNDARY_SAMPLES = 192      # boundary points per distance evaluation (denser near the tip)
_BOUNDARY_HEIGHT = 8.0       # how far above the tip the boundary is sampled [m]


# ------------------------------------------------------------------------------ safe set
@dataclass(frozen=True)
class ConeObstacleSafeSet(SafeSet):
    axis_x: float = 0.0
    axis_y: float = 0.0
    tip_z: float = 1.2
    theta_deg: float = 45.0
    eps: float = 0.25
    ground_z: float = 0.0
    recovery_rate_cone: float = 0.0
    recovery_rate_ground: float = 0.0

    @classmethod
    def from_config(cls, cfg: Any) -> "ConeObstacleSafeSet":
        return cls(axis_x=float(cfg.axis_x), axis_y=float(cfg.axis_y), tip_z=float(cfg.tip_z),
                   theta_deg=float(cfg.cone_theta_deg), eps=float(cfg.cone_eps), ground_z=float(cfg.floor_z),
                   recovery_rate_cone=float(cfg.recovery_rate_cone),
                   recovery_rate_ground=float(cfg.recovery_rate_ground))

    @property
    def tan_theta(self) -> float:
        return tan(radians(float(self.theta_deg)))

    @property
    def num_constraints(self) -> int:
        return 2

    @property
    def constraint_names(self) -> tuple[str, ...]:
        return CONSTRAINT_NAMES

    def cone_value(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x)
        dx, dy = x[..., 0] - self.axis_x, x[..., 1] - self.axis_y
        return jnp.sqrt(dx * dx + dy * dy + self.eps**2) - self.eps - self.tan_theta * (x[..., 2] - self.tip_z)

    def ground_value(self, x: jax.Array) -> jax.Array:
        return jnp.asarray(x)[..., 2] - self.ground_z

    def component_values(self, x: jax.Array) -> jax.Array:
        return jnp.stack([self.cone_value(x), self.ground_value(x)], axis=-1)

    def value(self, x: jax.Array) -> jax.Array:
        return jnp.min(self.component_values(x), axis=-1)

    def values_and_grads(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = jnp.asarray(x)
        dx, dy = x[0] - self.axis_x, x[1] - self.axis_y
        s = jnp.sqrt(dx * dx + dy * dy + self.eps**2)
        h_c = s - self.eps - self.tan_theta * (x[2] - self.tip_z)
        g_c = jnp.zeros((10,), x.dtype).at[0].set(dx / s).at[1].set(dy / s).at[2].set(-self.tan_theta)
        g_g = jnp.zeros((10,), x.dtype).at[2].set(1.0)
        return jnp.stack([h_c, x[2] - self.ground_z]), jnp.stack([g_c, g_g])

    def contains(self, x: jax.Array) -> jax.Array:
        return self.value(x) >= 0.0

    # --- Phase-I gentle-recovery envelope G_j = {grad_p h_j . v <= kappa_j h_j}
    @property
    def recovery_enabled(self) -> bool:
        return float(self.recovery_rate_cone) > 0.0 or float(self.recovery_rate_ground) > 0.0

    def recovery_margins(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x)
        v = x[..., 3:6]
        dx, dy = x[..., 0] - self.axis_x, x[..., 1] - self.axis_y
        s = jnp.sqrt(dx * dx + dy * dy + self.eps**2)
        dh_c = (dx * v[..., 0] + dy * v[..., 1]) / s - self.tan_theta * v[..., 2]
        inf = jnp.asarray(jnp.inf, x.dtype)
        k_c, k_g = float(self.recovery_rate_cone), float(self.recovery_rate_ground)
        m_c = jnp.where(k_c > 0.0, k_c * self.cone_value(x) - dh_c, inf)
        m_g = jnp.where(k_g > 0.0, k_g * self.ground_value(x) - v[..., 2], inf)
        return jnp.stack([m_c, m_g], axis=-1)

    def training_contains(self, x: jax.Array) -> jax.Array:
        ok = self.contains(x)
        if self.recovery_enabled:
            ok &= jnp.all(self.recovery_margins(x) >= 0.0, axis=-1)
        return ok

    # --- geometry
    def surface_z(self, r):
        """Height of the funnel's underside at horizontal distance r from the axis."""
        r = jnp.asarray(r)
        return self.tip_z + (jnp.sqrt(r * r + self.eps**2) - self.eps) / self.tan_theta

    def _boundary(self, dtype=jnp.float32):
        u = jnp.linspace(0.0, 1.0, _BOUNDARY_SAMPLES, dtype=dtype)
        s = _BOUNDARY_HEIGHT * u * u
        lin = self.eps + self.tan_theta * s
        return jnp.sqrt(jnp.maximum(lin * lin - self.eps**2, 0.0)), s

    def m_distance(self, p: jax.Array, m_h: float, m_v: float) -> jax.Array:
        """Distance from positions p (..., 3) to the funnel in the metric diag(m_h, m_h, m_v);
        0 inside. The funnel is convex and axisymmetric and the metric is isotropic
        horizontally, so the closest point lies in the vertical plane through p and the axis."""
        p = jnp.asarray(p)
        r = jnp.sqrt((p[..., 0] - self.axis_x) ** 2 + (p[..., 1] - self.axis_y) ** 2)
        s = p[..., 2] - self.tip_z
        rb, sb = self._boundary(p.dtype)
        d2 = m_h * (r[..., None] - rb) ** 2 + m_v * (s[..., None] - sb) ** 2
        inside = jnp.sqrt(r * r + self.eps**2) - self.eps - self.tan_theta * s < 0.0
        return jnp.where(inside, 0.0, jnp.sqrt(jnp.min(d2, axis=-1)))


# ------------------------------------------------------------------------------ controllers
def _box(cfg: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return tuple(cfg.action_low), tuple(cfg.action_high)


def _r_diag(cfg: Any) -> tuple[float, ...]:
    return tuple(float(getattr(cfg, k)) for k in ("lqr_r_a_cmd", "lqr_r_omega_x", "lqr_r_omega_y", "lqr_r_omega_z"))


def band_controller(cfg: Any) -> QuadrotorDLQR:
    """The PS2-RL paper's 7-D hover LQR at the band height, with the task's per-axis box."""
    ctrl = QuadrotorDLQR._build(
        dt=float(cfg.dt), design_gravity=float(cfg.gravity), z_des=float(cfg.band_height),
        q_diag=tuple(float(getattr(cfg, k)) for k in ("lqr_q_z", "lqr_q_vx", "lqr_q_vy", "lqr_q_vz",
                                                      "lqr_q_thetax", "lqr_q_thetay", "lqr_q_thetaz")),
        r_diag=_r_diag(cfg), box_gravity=float(cfg.gravity), a_cmd_min=float(cfg.a_cmd_min),
        a_cmd_max=float(cfg.a_cmd_max), omega_max=float(cfg.omega_max))
    low, high = _box(cfg)
    return replace(ctrl, u_low=low, u_high=high)


def point_controller(cfg: Any, point) -> QuadrotorLandingDLQR:
    """9-D hover LQR at ``point`` (x, y, z)."""
    px, py, pz = (float(v) for v in point)
    return QuadrotorLandingDLQR._build(
        dt=float(cfg.dt), gravity=float(cfg.gravity), z_des=pz, pad=(px, py, 0.0),
        q_diag=tuple(float(getattr(cfg, k)) for k in ("lqr_q_x", "lqr_q_y", "lqr_q_z", "lqr_q_vx", "lqr_q_vy", "lqr_q_vz",
                                                      "lqr_q_thetax", "lqr_q_thetay", "lqr_q_thetaz")),
        r_diag=_r_diag(cfg), a_cmd_min=float(cfg.a_cmd_min), a_cmd_max=float(cfg.a_cmd_max),
        omega_max=float(cfg.omega_max), omega_z_max=float(cfg.omega_z_max))


@dataclass(frozen=True)
class BrakeDLQR(DiscreteLQR):
    """6-D hover LQR on (v, phi) only: it stops and levels the vehicle wherever it is.

    For the linearised closed loop, p_k + L r_k is constant along trajectories, with
    L = dt [I 0] (I - A_cl)^-1. So the vehicle comes to rest at p_inf = p + L r, and on the
    way its positions stay inside p_inf - L {r : r^T P r <= c}, an ellipsoid around the rest
    point whose shape does not depend on where the vehicle is.
    """

    dt: float = 0.02

    @classmethod
    def from_config(cls, cfg: Any) -> "BrakeDLQR":
        a9, b9 = QuadrotorLandingDLQR.linearization_error_coords(float(cfg.gravity))
        a_d, b_d = euler_discretize(a9[3:9, 3:9], b9[3:9, :], float(cfg.dt))
        low, high = _box(cfg)
        return cls(a_d=tuple(tuple(r) for r in a_d.tolist()), b_d=tuple(tuple(r) for r in b_d.tolist()),
                   q_diag=tuple(float(getattr(cfg, k)) for k in ("lqr_q_vx", "lqr_q_vy", "lqr_q_vz",
                                                                 "lqr_q_thetax", "lqr_q_thetay", "lqr_q_thetaz")),
                   r_diag=_r_diag(cfg), u_star=(float(cfg.gravity), 0.0, 0.0, 0.0), u_low=low, u_high=high,
                   dt=float(cfg.dt))

    def error_state(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x)
        q = normalize_quaternion_batch(x[..., 6:10])
        q_err = quaternion_conjugate_batch(q)
        sign = jnp.where(q_err[..., 0] >= 0.0, 1.0, -1.0).astype(x.dtype)
        return jnp.concatenate([x[..., 3:6], 2.0 * sign[..., None] * q_err[..., 1:4]], axis=-1)

    def p_matrix_f64(self) -> np.ndarray:
        from ps2rl.base_controller.base_controller import _solve_discrete_are_scipy

        p = _solve_discrete_are_scipy(np.asarray(self.a_d), np.asarray(self.b_d),
                                      np.diag(np.asarray(self.q_diag)), np.diag(np.asarray(self.r_diag)))
        return 0.5 * (p.real + p.real.T)

    def rest_gain(self) -> np.ndarray:
        """L (3, 6): rest point p_inf = p + L r of the linearised closed loop."""
        a = np.asarray(self.a_d, np.float64)
        b = np.asarray(self.b_d, np.float64)
        p = self.p_matrix_f64()
        r = np.diag(np.asarray(self.r_diag, np.float64))
        k = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
        sel = np.zeros((3, 6))
        sel[:, 0:3] = np.eye(3)
        return float(self.dt) * sel @ np.linalg.inv(np.eye(6) - (a - b @ k))


def position_metric(ctrl) -> tuple[float, float, np.ndarray]:
    """(m_h, m_v, W) for a 9-D hover LQR: {e^T P e <= c} projects onto positions as
    {dp^T M dp <= c}, M = W^-1 with W = (P^-1)_pp = diag(1/m_h, 1/m_h, 1/m_v)."""
    return _diag_metric(np.linalg.inv(ctrl.p_matrix_f64())[0:3, 0:3])


def _diag_metric(w: np.ndarray) -> tuple[float, float, np.ndarray]:
    off = np.abs(w - np.diag(np.diag(w))).max()
    if off > 1e-8 * np.abs(w).max() or not np.isclose(w[0, 0], w[1, 1], rtol=1e-8):
        raise ValueError("expected a diagonal, horizontally isotropic position shape")
    return 1.0 / float(w[0, 0]), 1.0 / float(w[2, 2]), w


# ------------------------------------------------------------------------------ base sets
class _BandBaseSet(EllipsoidBaseSet):
    """The paper's band, with the base-controller hook the other designs expose."""

    def action(self, x: jax.Array) -> jax.Array:
        return self.controller.action(x)


@dataclass(frozen=True)
class PointsBaseSet(BaseSet):
    """Union of 9-D hover ellipsoids {V_i <= c} with a shared P and K (same LQR weights).

    Each member is invariant under its own LQR, so the union is invariant under "use the
    LQR of a member that contains x" (here: the member with the smallest V_i)."""

    controllers: tuple
    base_set_c: float
    smooth_gain: float = 20.0
    centers: jax.Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.controllers:
            raise ValueError("PointsBaseSet needs at least one controller")
        c0 = self.controllers[0]
        for c in self.controllers[1:]:
            if c.q_diag != c0.q_diag or c.r_diag != c0.r_diag:
                raise ValueError("all members must share the LQR weights")
        if float(self.base_set_c) > float(c0.max_certified_level) + 1e-12:
            raise ValueError(f"base_set_c={self.base_set_c} exceeds the input-feasibility bound {c0.max_certified_level}")
        cs = np.asarray([[c.pad_x, c.pad_y, c.pad_z + c.z_des] for c in self.controllers], dtype=np.float32)
        object.__setattr__(self, "centers", jnp.asarray(cs))

    @property
    def controller(self):
        return self.controllers[0]

    @property
    def num_constraints(self) -> int:
        return 1

    def _errors(self, x: jax.Array) -> jax.Array:
        e0 = self.controllers[0].error_state(x)
        shift = self.centers - self.centers[0]
        e = jnp.broadcast_to(e0[..., None, :], e0.shape[:-1] + (shift.shape[0], 9))
        return e.at[..., 0:3].add(-shift)

    def lyapunov_values(self, x: jax.Array) -> jax.Array:
        e = self._errors(jnp.asarray(x))
        p = jnp.asarray(self.controllers[0].p_matrix, dtype=e.dtype)
        return jnp.einsum("...ni,ij,...nj->...n", e, p, e)

    def _values_only(self, x: jax.Array) -> jax.Array:
        return jnp.asarray([self.base_set_c - jnp.min(self.lyapunov_values(x))], dtype=jnp.asarray(x).dtype)

    def values_and_grads(self, x):
        x = jnp.asarray(x)
        return self._values_only(x), jax.jacfwd(self._values_only)(x)

    def contains(self, x, atol: float = 0.0):
        return jnp.min(self.lyapunov_values(x), axis=-1) <= self.base_set_c + atol

    def margin(self, x):
        return self.base_set_c - jnp.min(self.lyapunov_values(x), axis=-1)

    def smooth_distance(self, x):
        gain = max(float(self.smooth_gain), 1e-6)
        return _softplus(gain * (jnp.min(self.lyapunov_values(x), axis=-1) - self.base_set_c)) / gain

    def action(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x)
        e = self._errors(x)
        i = jnp.argmin(self.lyapunov_values(x))
        c0 = self.controllers[0]
        u = jnp.asarray(c0.u_star, x.dtype) - jnp.asarray(c0.k_matrix, x.dtype) @ e[i]
        return jnp.clip(u, jnp.asarray(c0.u_low, x.dtype), jnp.asarray(c0.u_high, x.dtype))


@dataclass(frozen=True)
class AnywhereBaseSet(BaseSet):
    """"Stop wherever it is clear": the brake LQR (velocity and attitude only) is the base
    controller inside B, and B holds the states whose whole stopping transient clears the
    funnel and the ground.

    With r = (v, phi), V = r^T P r and the rest point p_inf = p + L r, the transient's
    positions stay inside p_inf - L {V' <= V}, an ellipsoid of shape W = L P^-1 L^T scaled by
    V. Rows (all >= 0 in B):

        c - V
        d_M(p_inf)^2 - k^2 V                         d_M: distance to the funnel, metric W^-1
        (z_inf - ground) |z_inf - ground| - k^2 W_zz V

    The clearance shrinks with the remaining energy V, not with c: p_inf is constant only
    for the linearised loop, and as V decays the shrinking clearance absorbs the drift of the
    nonlinear loop (a fixed clearance does not - the rows would drift through zero).
    k >= 1 is a safety factor on that. Squared rows keep the gradients bounded at V = 0.

    Outside B the base controller is a 9-D hover LQR toward the nearest admissible rest point
    (pushed up to the ground clearance, then radially away from the axis to the funnel
    clearance, both at level c).
    """

    brake: BrakeDLQR
    steer: QuadrotorLandingDLQR
    base_set_c: float
    safe: ConeObstacleSafeSet
    smooth_gain: float = 20.0
    clearance_factor: float = 1.0
    _l: jax.Array = field(init=False, repr=False)
    _p: jax.Array = field(init=False, repr=False)
    _m_h: float = field(init=False)
    _m_v: float = field(init=False)
    _w_zz: float = field(init=False)
    _ext_z: float = field(init=False)

    def __post_init__(self) -> None:
        if float(self.base_set_c) > float(self.brake.max_certified_level) + 1e-12:
            raise ValueError(f"base_set_c={self.base_set_c} exceeds the brake LQR's input bound {self.brake.max_certified_level}")
        p = self.brake.p_matrix_f64()
        l_gain = self.brake.rest_gain()
        m_h, m_v, w = _diag_metric(l_gain @ np.linalg.inv(p) @ l_gain.T)
        object.__setattr__(self, "_l", jnp.asarray(l_gain, jnp.float32))
        object.__setattr__(self, "_p", jnp.asarray(p, jnp.float32))
        object.__setattr__(self, "_m_h", m_h)
        object.__setattr__(self, "_m_v", m_v)
        object.__setattr__(self, "_w_zz", float(w[2, 2]))
        object.__setattr__(self, "_ext_z", float(self.clearance_factor) * float(sqrt(float(self.base_set_c) * w[2, 2])))

    @property
    def controller(self):
        return self.brake

    @property
    def num_constraints(self) -> int:
        return 3

    @property
    def clearance(self) -> float:
        return float(self.clearance_factor) * float(sqrt(float(self.base_set_c)))

    @property
    def extents(self) -> tuple[float, float]:
        """Half-extents (horizontal, vertical) of the stopping transient's positions at level c."""
        return float(self.clearance / sqrt(self._m_h)), self._ext_z

    def rest_point_and_energy(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = jnp.asarray(x)
        r = self.brake.error_state(x)
        p_inf = x[..., 0:3] + jnp.einsum("ij,...j->...i", self._l.astype(x.dtype), r)
        return p_inf, jnp.einsum("...i,ij,...j->...", r, self._p.astype(x.dtype), r)

    def _rows(self, x: jax.Array) -> jax.Array:
        p_inf, energy = self.rest_point_and_energy(x)
        k2 = float(self.clearance_factor) ** 2
        d = self.safe.m_distance(p_inf, self._m_h, self._m_v)
        dz = p_inf[..., 2] - self.safe.ground_z
        return jnp.stack([self.base_set_c - energy, d * d - k2 * energy,
                          dz * jnp.abs(dz) - k2 * self._w_zz * energy], axis=-1)

    def values_and_grads(self, x):
        x = jnp.asarray(x)
        return self._rows(x), jax.jacfwd(self._rows)(x)

    def contains(self, x, atol: float = 0.0):
        return jnp.all(self._rows(x) >= -atol, axis=-1)

    def margin(self, x):
        return jnp.min(self._rows(x), axis=-1)

    def smooth_distance(self, x):
        gain = max(float(self.smooth_gain), 1e-6)
        return jnp.sum(_softplus(-gain * self._rows(x)), axis=-1) / gain

    def admissible_point(self, x: jax.Array) -> jax.Array:
        """Nearest admissible rest point to p_inf(x): up to the ground clearance, then radially out."""
        x = jnp.asarray(x)
        p_inf, _ = self.rest_point_and_energy(x)
        z = jnp.maximum(p_inf[2], self.safe.ground_z + self._ext_z)
        dx, dy = p_inf[0] - self.safe.axis_x, p_inf[1] - self.safe.axis_y
        r0 = jnp.sqrt(dx * dx + dy * dy)
        ux = jnp.where(r0 > 1e-6, dx / jnp.maximum(r0, 1e-6), -1.0)
        uy = jnp.where(r0 > 1e-6, dy / jnp.maximum(r0, 1e-6), 0.0)

        def dist_at(r):
            return self.safe.m_distance(jnp.stack([self.safe.axis_x + r * ux, self.safe.axis_y + r * uy, z]), self._m_h, self._m_v)

        def body(_, lohi):
            lo, hi = lohi
            mid = 0.5 * (lo + hi)
            ok = dist_at(mid) >= self.clearance
            return jnp.where(ok, lo, mid), jnp.where(ok, mid, hi)

        need = dist_at(r0) < self.clearance
        _, hi = jax.lax.fori_loop(0, 18, body, (r0, r0 + 8.0))
        r = jnp.where(need, hi, r0)
        return jnp.stack([self.safe.axis_x + r * ux, self.safe.axis_y + r * uy, z])

    def action(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x)
        b = self.brake
        u_brake = jnp.asarray(b.u_star, x.dtype) - jnp.asarray(b.k_matrix, x.dtype) @ b.error_state(x)
        s = self.steer
        centre = jnp.asarray([s.pad_x, s.pad_y, s.pad_z + s.z_des], x.dtype)
        e = s.error_state(x).at[0:3].add(centre - self.admissible_point(x).astype(x.dtype))
        u_steer = jnp.asarray(s.u_star, x.dtype) - jnp.asarray(s.k_matrix, x.dtype) @ e
        u = jnp.where(self.contains(x), u_brake, u_steer)
        return jnp.clip(u, jnp.asarray(b.u_low, x.dtype), jnp.asarray(b.u_high, x.dtype))


def obstacle_sets(cfg: Any) -> tuple[ConeObstacleSafeSet, BaseSet]:
    """(S, B) for the configured base-set design; B.action is its base controller."""
    safe = ConeObstacleSafeSet.from_config(cfg)
    c = float(cfg.base_set_c)
    gain = float(cfg.base_set_smooth_gain)
    if cfg.base_set == "band":
        return safe, _BandBaseSet(band_controller(cfg), c, smooth_gain=gain)
    if cfg.base_set == "points":
        return safe, PointsBaseSet(tuple(point_controller(cfg, p) for p in cfg.hover_points), c, smooth_gain=gain)
    return safe, AnywhereBaseSet(BrakeDLQR.from_config(cfg), point_controller(cfg, (cfg.axis_x, cfg.axis_y, cfg.tip_z)),
                                 c, safe, smooth_gain=gain, clearance_factor=float(cfg.anywhere_clearance_factor))


__all__ = [
    "AnywhereBaseSet",
    "BrakeDLQR",
    "CONSTRAINT_NAMES",
    "ConeObstacleSafeSet",
    "PointsBaseSet",
    "band_controller",
    "obstacle_sets",
    "point_controller",
    "position_metric",
]
