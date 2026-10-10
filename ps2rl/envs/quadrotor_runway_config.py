"""Single source of truth for the runway bird-deterrence task: geometry, base set, LQR.

Scene (world frame, z up). The runway and its protection strip occupy y > y_edge for every
x (a runway is kilometres long, the drone works within a few tens of metres of it, so the
strip is a half-space here). The drone operates on the y < y_edge side under an airspace
ceiling:

    S = { x : h_rwy(x) = y_edge - p_y >= 0,  h_ceil(x) = z_max - p_z >= 0 }.

Nothing depends on p_x (along the runway): dynamics, safe set, base set and backup are all
invariant to it, so every controller, set and checkpoint here ignores p_x.

Base set (the backup's terminal set): "retreat at altitude". The hover LQR in the 7-D error

    e(x) = (p_z - z_hold, v_x, v_y + v_ret, v_z, phi)          (phi: quaternion-error angles)

regulates altitude to z_hold, the horizontal velocity to (0, -v_ret) - moving away from the
runway at v_ret - and the attitude to level, yaw 0. B = {e^T P e <= c_B}. Inside B the
velocity error is at most sqrt(c_B (P^-1)_vy,vy) < v_ret, so v_y < 0 and p_y only decreases,
and p_z <= z_hold + sqrt(c_B (P^-1)_zz) <= z_max: B n S is forward invariant under the LQR
(checked, together with the robust Lyapunov decrease, by ``ps2rl.sets.runway_certificate``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class QuadrotorRunwayConfig:
    # dynamics / sampling (identical to the landing / powerloop model)
    dt: float = 0.02
    T: float = 2.0
    num_steps: int = 100
    gravity: float = 9.81
    a_cmd_min: float = 0.0
    a_cmd_max: float = 39.24
    omega_max: float = 18.0

    # safe set: runway keep-out half-space and airspace ceiling
    y_edge: float = 0.0
    z_max: float = 10.0

    # base set: retreat away from the runway at altitude
    z_hold: float = 6.0
    v_ret: float = 2.4
    base_set_c: float = 10.5
    base_set_smooth_gain: float = 20.0

    # retreat-LQR weights, error order (z, v_x, v_y, v_z, phi_x, phi_y, phi_z); yaw chain as in landing.
    # Chosen for a B that is wide in altitude (+-0.9 m) and velocity (+-1.8 m/s) at moderate tilt (51 deg):
    # the UE tube shrinks B by m_B(s(tau)), and B's velocity width sets how long it stays reachable
    # (m_B / c_B = 0.44 at tau = 1.5 s here vs 0.65 with the powerloop-like weights)
    lqr_q_z: float = 0.3
    lqr_q_vx: float = 0.1
    lqr_q_vy: float = 0.1
    lqr_q_vz: float = 0.2
    lqr_q_thetax: float = 4.0
    lqr_q_thetay: float = 4.0
    lqr_q_thetaz: float = 0.5
    lqr_r_a_cmd: float = 0.02
    lqr_r_omega_x: float = 0.012
    lqr_r_omega_y: float = 0.012
    lqr_r_omega_z: float = 0.04
    # tube metric only: weight on p_y in the 8-D metric LQR (y, z, v, phi); see runway_ue_tube
    metric_q_y: float = 0.3

    def __post_init__(self) -> None:
        n = int(round(float(self.T) / float(self.dt)))
        if abs(n * float(self.dt) - float(self.T)) > 1e-9 or n != int(self.num_steps):
            raise ValueError(f"T={self.T}, dt={self.dt} and num_steps={self.num_steps} are inconsistent")
        if float(self.v_ret) <= 0.0:
            raise ValueError("v_ret must be > 0 (the base set retreats away from the runway)")
        if float(self.z_hold) >= float(self.z_max):
            raise ValueError("z_hold must be below the ceiling z_max")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QuadrotorRunwayConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in names})

    def replace(self, **kw: Any) -> "QuadrotorRunwayConfig":
        d = self.as_dict()
        d.update(kw)
        return QuadrotorRunwayConfig(**d)


__all__ = ["QuadrotorRunwayConfig"]
