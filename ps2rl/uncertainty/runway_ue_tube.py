"""UE-bCBF tube for the runway task (same model and recursion as ``landing_ue_tube``).

Disturbance model, observer and frozen-estimate backup flow are those of the landing UE
tube (``LandingUEConfig`` is reused as is). The metric is the 8-D DARE matrix P8 of
``RunwayMetricChart``, e8 = (p_y - y_edge, p_z - z_hold, v, phi) (p_x is irrelevant: nothing
depends on it). With e the deviation of the true state from the frozen-estimate rollout,

    s_{k+1} = ||F_k||_P8 s_k + dt gamma q(tau_k),   gamma = ||P8^{1/2} E_d||,

bounds ||e_k||_P8 to first order, and the margins are (s_eff = tube_scale * s):

    runway  : |p_y - p_y,nom| <= L_y s_eff,  L_y = sqrt((P8^-1)_yy)
    ceiling : |p_z - p_z,nom| <= L_z s_eff,  L_z = sqrt((P8^-1)_zz)
    base    : |V7(x) - V7(x_nom)| <= 2 sqrt(c_B) kappa s_eff + (kappa s_eff)^2,
              V7 = e7^T P7 e7 (the base set's own LQR), kappa = ||P7^{1/2} Pi P8^{-1/2}||,
              Pi drops p_y (||e7||_P7 <= kappa ||e8||_P8).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.base_controller.quadrotor_retreat_dlqr import QuadrotorRetreatDLQR, RunwayMetricChart
from ps2rl.uncertainty.landing_ue_tube import LandingUEConfig, make_growth_fn, q_of_tau, tube_step

UEConfig = LandingUEConfig  # same disturbance / observer / tube settings as the landing UE task


def _sqrtm(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ev, evec = np.linalg.eigh(p)
    return evec @ np.diag(np.sqrt(ev)) @ evec.T, evec @ np.diag(1.0 / np.sqrt(ev)) @ evec.T


@dataclass(frozen=True)
class RunwayTubeConstants:
    p: np.ndarray  # (8, 8) metric
    p_half: np.ndarray
    p_mhalf: np.ndarray
    gamma: float
    l_y: float
    l_z: float
    kappa_b: float
    c_b: float

    def as_dict(self) -> dict[str, float]:
        return {"gamma": self.gamma, "l_y": self.l_y, "l_z": self.l_z, "kappa_b": self.kappa_b, "c_b": self.c_b}


def metric_chart(cfg) -> RunwayMetricChart:
    return RunwayMetricChart.from_config(cfg)


def tube_constants(cfg) -> RunwayTubeConstants:
    chart = RunwayMetricChart.from_config(cfg)
    p8 = chart.p
    ph8, pmh8 = _sqrtm(p8)
    p8_inv = np.linalg.inv(p8)
    p7 = QuadrotorRetreatDLQR.from_config(cfg).p_matrix_f64()
    ph7, _ = _sqrtm(p7)
    proj = np.zeros((7, 8))
    proj[:, 1:] = np.eye(7)
    return RunwayTubeConstants(
        p=p8, p_half=ph8, p_mhalf=pmh8,
        gamma=float(np.sqrt(np.linalg.eigvalsh(p8[2:5, 2:5]).max())),
        l_y=float(np.sqrt(p8_inv[0, 0])), l_z=float(np.sqrt(p8_inv[1, 1])),
        kappa_b=float(np.linalg.norm(ph7 @ proj @ pmh8, 2)), c_b=float(cfg.base_set_c),
    )


def tube_margins(s, ue: LandingUEConfig, tc: RunwayTubeConstants):
    """(m_runway, m_ceiling, m_base) for tube value s (any shape)."""
    se = float(ue.tube_scale) * jnp.asarray(s)
    kb = tc.kappa_b * se
    return tc.l_y * se, tc.l_z * se, 2.0 * float(np.sqrt(tc.c_b)) * kb + kb * kb


def nominal_tube(n_steps: int, growth_const: float, ue: LandingUEConfig, tc: RunwayTubeConstants, dt: float) -> np.ndarray:
    s = np.zeros(n_steps + 1)
    for k in range(n_steps):
        s[k + 1] = growth_const * s[k] + dt * tc.gamma * (ue.e_bar + min(ue.delta_v * k * dt, 2.0 * ue.delta_d))
    return s


__all__ = ["RunwayTubeConstants", "UEConfig", "make_growth_fn", "metric_chart", "nominal_tube", "q_of_tau",
           "tube_constants", "tube_margins", "tube_step"]
