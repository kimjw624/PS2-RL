"""Single source of truth for the landing task's Phase-I geometry and base set.

Defaults are the certified choice (see ``scripts/certify_landing_base_set.py``):

* pad at the origin, approach cone r0 = 0.5 m, theta = 30 deg, eps = 0.05 r0
* hover base set at z_des = 1.25 m above the pad, level c_B = 12
* powerloop LQR weights, plus q_x = q_y = q_z = 1, and a re-weighted yaw chain
  (q_thetaz 0.16 -> 0.5, r_omega_z 0.004 -> 0.04). The yaw chain is decoupled from
  the other three integrator chains, so this only affects yaw; with the powerloop
  yaw weights the quaternion chart caps c at 5.39 and the Lyapunov decrease fails
  near 6.7, whereas with these weights c_U = 17.9, c_chart = 23.2, c_Lyap ~ 15.5.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class QuadrotorLandingConfig:
    # dynamics / sampling (identical to the powerloop model)
    dt: float = 0.02
    T: float = 2.0
    num_steps: int = 100
    gravity: float = 9.81
    a_cmd_min: float = 0.0
    a_cmd_max: float = 39.24
    omega_max: float = 18.0

    # pad and approach cone
    pad_x: float = 0.0
    pad_y: float = 0.0
    pad_z: float = 0.0
    cone_r0: float = 0.5
    cone_theta_deg: float = 30.0
    cone_eps: float = 0.025
    # pad plane as a second safety constraint (h_floor = z - pad_z >= 0). Off by default so
    # checkpoints trained before it existed keep their meaning (cone-only safe set).
    floor_constraint: bool = False
    # Phase-I gentle-recovery envelope G_j = {grad_p h_j . v <= kappa_j h_j} (0 = off); see
    # ps2rl/sets/quadrotor_landing_safe_set.py. Training-only; not part of the safety spec.
    recovery_rate_cone: float = 0.0
    recovery_rate_floor: float = 0.0

    # base set
    z_des: float = 1.25
    base_set_c: float = 12.0
    base_set_smooth_gain: float = 20.0
    z_clear: float = 0.5  # minimum height of the base-set ellipsoid above the pad

    # hover LQR weights, error order (dp_x, dp_y, zeta, v, phi)
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

    def __post_init__(self) -> None:
        n = int(round(self.T / self.dt))
        if n != int(self.num_steps):
            raise ValueError(f"num_steps={self.num_steps} disagrees with T/dt={n}")
        if float(self.recovery_rate_floor) > 0.0 and not bool(self.floor_constraint):
            raise ValueError("recovery_rate_floor > 0 needs floor_constraint=True")
        if bool(self.floor_constraint) and float(self.z_clear) < 0.0:
            raise ValueError("with the floor in the safe set the base set must stay above it (z_clear >= 0)")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QuadrotorLandingConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in names})

    def replace(self, **kwargs: Any) -> "QuadrotorLandingConfig":
        d = self.as_dict()
        d.update(kwargs)
        return QuadrotorLandingConfig(**d)


__all__ = ["QuadrotorLandingConfig"]
