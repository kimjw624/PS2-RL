"""Configuration of the cone-obstacle task: the *outside* of an apex-down funnel is safe.

A drone flies toward a target that lies under the funnel. The funnel's outer wall faces
outward and down, so a safety filter that removes the inward part of a command makes the
vehicle slide down the wall and under the tip. The ground is a second constraint.

    h_cone(x)   = sqrt(r^2 + eps^2) - eps - tan(theta) (z - tip_z)  >= 0   (outside the funnel)
    h_ground(x) = z - ground_z                                      >= 0

r is the horizontal distance from the funnel axis. At r = 0 the boundary is the tip,
z = tip_z; eps rounds the tip (radius of curvature about eps / tan(theta)^2).

Base set designs (``base_set``), all certified relative to a hover LQR:

* ``band``     the PS2-RL paper's construction: the 7-D hover LQR (altitude, velocity,
               attitude; horizontal position free) at height ``band_z``. Below the tip
               every horizontal position is outside the funnel, so the band is inside S
               when it fits between the ground and the tip.
* ``points``   9-D hover-LQR ellipsoids (the landing construction) at ``hover_points``;
               one point is a single base set, several are their union.
* ``anywhere`` "stop wherever it is clear": a 6-D LQR on velocity and attitude only (the
               brake) is the base controller; a state is in B when its velocity / attitude
               energy V is below ``base_set_c`` and the whole stopping transient (an
               ellipsoid around the rest point, sized by V) clears the funnel and the
               ground. A vehicle at rest anywhere in S is in B.

The target is a vertical cylinder of radius ``target_radius``, centred on the funnel axis
(or shifted by ``target_offset_x/y``, e.g. to the far side), from the ground up to
``target_height`` (default: up to the funnel's underside, i.e. the tip for a centred
cylinder): touching it is success (no need to hover inside it).

Additions only: the landing task (``quadrotor_landing_config``) is unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from math import isfinite, radians, tan
from typing import Any

BASE_SET_DESIGNS = ("band", "points", "anywhere")


@dataclass(frozen=True)
class ConeObstacleConfig:
    # dynamics / sampling (the landing / powerloop model)
    dt: float = 0.02
    T: float = 2.0
    num_steps: int = 100
    gravity: float = 9.81
    a_cmd_min: float = 0.0
    a_cmd_max: float = 39.24
    omega_max: float = 18.0
    omega_z_max: float = 0.0  # 0: yaw uses omega_max

    # funnel obstacle (apex down) and ground
    axis_x: float = 0.0
    axis_y: float = 0.0
    tip_z: float = 1.2
    cone_theta_deg: float = 45.0
    cone_eps: float = 0.25
    ground_z: float = 0.0
    # Phase-I gentle-recovery envelope G_j = {grad_p h_j . v <= kappa_j h_j} (0 = off)
    recovery_rate_cone: float = 0.0
    recovery_rate_ground: float = 0.0

    # base set
    base_set: str = "band"
    base_set_c: float = 4.0
    base_set_smooth_gain: float = 20.0
    band_z: float = -1.0                        # < 0: middle of the gap between ground and tip
    # "anywhere": safety factor (>= 1) on the stopping ellipsoid's clearance; the clearance
    # shrinks with the remaining energy, which absorbs the drift of the rest point in the
    # nonlinear loop (cone_obstacle_certify.py checks the invariance per factor)
    anywhere_clearance_factor: float = 1.0
    hover_points: tuple = ((-1.5, 0.0, 0.8),)   # for "points"

    # hover LQR weights, 9-D order (dp_x, dp_y, zeta, v, phi); the band drops q_x, q_y
    lqr_q_x: float = 1.0
    lqr_q_y: float = 1.0
    lqr_q_z: float = 1.0
    lqr_q_vx: float = 0.16
    lqr_q_vy: float = 0.16
    lqr_q_vz: float = 0.4
    lqr_q_thetax: float = 0.8
    lqr_q_thetay: float = 0.8
    lqr_q_thetaz: float = 0.5
    lqr_r_a_cmd: float = 0.02
    lqr_r_omega_x: float = 0.012
    lqr_r_omega_y: float = 0.012
    lqr_r_omega_z: float = 0.04

    # target cylinder around the funnel axis (success = touching it)
    target_radius: float = 0.3
    target_height: float = -1.0                 # < 0: up to the funnel (the tip when centred)
    target_offset_x: float = 0.0                # cylinder centre relative to the axis
    target_offset_y: float = 0.0

    def __post_init__(self) -> None:
        n = int(round(self.T / self.dt))
        if n != int(self.num_steps):
            raise ValueError(f"num_steps={self.num_steps} disagrees with T/dt={n}")
        if self.base_set not in BASE_SET_DESIGNS:
            raise ValueError(f"base_set must be one of {BASE_SET_DESIGNS}, got '{self.base_set}'")
        if not (0.0 < float(self.cone_theta_deg) < 90.0):
            raise ValueError(f"cone_theta_deg must lie in (0, 90), got {self.cone_theta_deg}")
        if float(self.cone_eps) <= 0.0:
            raise ValueError(f"cone_eps must be > 0 (it rounds the tip), got {self.cone_eps}")
        if float(self.tip_z) <= float(self.ground_z):
            raise ValueError(f"tip_z={self.tip_z} must be above ground_z={self.ground_z}")
        c = float(self.base_set_c)
        if not isfinite(c) or c <= 0.0:
            raise ValueError(f"base_set_c must be positive, got {c}")
        if float(self.anywhere_clearance_factor) < 1.0:
            raise ValueError(f"anywhere_clearance_factor must be >= 1, got {self.anywhere_clearance_factor}")
        if float(self.omega_z_max) < 0.0 or float(self.omega_z_max) > float(self.omega_max):
            raise ValueError(f"omega_z_max must lie in [0, omega_max], got {self.omega_z_max}")
        pts = tuple(tuple(float(v) for v in p) for p in self.hover_points)
        if self.base_set == "points" and (not pts or any(len(p) != 3 for p in pts)):
            raise ValueError(f"hover_points must be a non-empty list of (x, y, z), got {self.hover_points}")
        object.__setattr__(self, "hover_points", pts)
        for name in ("recovery_rate_cone", "recovery_rate_ground", "target_radius"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        h_max = self.funnel_surface_z(self.target_near_r)
        if float(self.target_height) >= 0.0 and not (float(self.ground_z) < float(self.target_height) <= h_max + 1e-9):
            raise ValueError(f"target_height={self.target_height} must lie in (ground_z, {h_max:.4f}] (the funnel above the "
                             "cylinder), or < 0 for that height")

    # ----------------------------------------------------------------- geometry
    @property
    def tan_theta(self) -> float:
        return tan(radians(float(self.cone_theta_deg)))

    @property
    def gap(self) -> float:
        """Free height between the ground and the tip."""
        return float(self.tip_z) - float(self.ground_z)

    @property
    def band_height(self) -> float:
        return float(self.band_z) if float(self.band_z) >= 0.0 else float(self.ground_z) + 0.5 * self.gap

    def funnel_surface_z(self, r: float) -> float:
        """Height of the funnel's underside at horizontal distance r from the axis."""
        e = float(self.cone_eps)
        return float(self.tip_z) + ((float(r) ** 2 + e * e) ** 0.5 - e) / self.tan_theta

    @property
    def target_center(self) -> tuple[float, float]:
        return float(self.axis_x) + float(self.target_offset_x), float(self.axis_y) + float(self.target_offset_y)

    @property
    def target_near_r(self) -> float:
        """Distance from the axis of the cylinder's point closest to it."""
        d = (float(self.target_offset_x) ** 2 + float(self.target_offset_y) ** 2) ** 0.5
        return max(0.0, d - float(self.target_radius))

    @property
    def cylinder_top(self) -> float:
        return float(self.target_height) if float(self.target_height) >= 0.0 else self.funnel_surface_z(self.target_near_r)

    def touches_target(self, p) -> bool:
        """Whether position p (x, y, z) touches the target cylinder."""
        cx, cy = self.target_center
        r2 = (float(p[0]) - cx) ** 2 + (float(p[1]) - cy) ** 2
        return r2 <= float(self.target_radius) ** 2 and float(self.ground_z) <= float(p[2]) <= self.cylinder_top

    # ----------------------------------------------------------------- action box
    @property
    def omega_limits(self) -> tuple[float, float, float]:
        w = float(self.omega_max)
        return (w, w, float(self.omega_z_max) if float(self.omega_z_max) > 0.0 else w)

    @property
    def omega_clip(self):
        return self.omega_limits if float(self.omega_z_max) > 0.0 else float(self.omega_max)

    @property
    def action_low(self) -> tuple[float, float, float, float]:
        wx, wy, wz = self.omega_limits
        return (float(self.a_cmd_min), -wx, -wy, -wz)

    @property
    def action_high(self) -> tuple[float, float, float, float]:
        wx, wy, wz = self.omega_limits
        return (float(self.a_cmd_max), wx, wy, wz)

    @property
    def action_scale(self) -> tuple[float, float, float, float]:
        wx, wy, wz = self.omega_limits
        return (float(self.a_cmd_max), wx, wy, wz)

    # ----------------------------------------------------------------- (de)serialisation
    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["hover_points"] = [list(p) for p in self.hover_points]
        return d

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConeObstacleConfig":
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in payload.items() if k in names}
        if "hover_points" in kw:
            kw["hover_points"] = tuple(tuple(float(v) for v in p) for p in kw["hover_points"])
        return cls(**kw)

    def replace(self, **kwargs: Any) -> "ConeObstacleConfig":
        d = self.as_dict()
        d.update(kwargs)
        return ConeObstacleConfig.from_dict(d)


__all__ = ["BASE_SET_DESIGNS", "ConeObstacleConfig"]
