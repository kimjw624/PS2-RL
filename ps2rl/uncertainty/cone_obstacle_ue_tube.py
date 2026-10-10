"""UE-bCBF tube for the cone-obstacle task (outside of an apex-down funnel is safe).

Same disturbance model, observer and recursion as ``landing_ue_tube`` (``LandingUEConfig`` is
reused as is):

    x_dot = f(x) + g(x) u + E_d d(t),  ||d|| <= delta_d,  ||d_dot|| <= delta_v,
    ||d(t + tau) - d_hat(t)|| <= q(tau) = e_bar + min(delta_v tau, 2 delta_d),
    s_{k+1} = ||F_k||_P s_k + dt gamma q(tau_k),   gamma = ||P^{1/2} E_d||,

with P the DARE matrix of the base set's 9-D hover LQR (the ellipsoid beside the funnel) and
||e||_P bounding the deviation of the true state from the frozen-estimate backup rollout.

Margins (s_eff = tube_scale * s):

    funnel : |h_cone(x) - h_cone(x_nom)| <= L_cone s_eff,
             grad_p h_cone = (dx/rho, dy/rho, -tan theta), |(dx, dy)/rho| <= 1, so
             L_cone = max_{|g_xy| <= 1} ||P^{-1/2} (g_xy, tan theta, 0..)||  (the landing formula)
    floor  : |z - z_nom| <= L_floor s_eff,  L_floor = sqrt((P^{-1})_zz)
    base   : |V(x) - V(x_nom)| <= 2 sqrt(c_B) s_eff + s_eff^2

so the tightened ("shrunk") sets at backup time tau_k are

    S_k = {h_cone >= L_cone s_eff,k} n {h_floor >= L_floor s_eff,k},  B_k = {V <= c_B - m_base(s_eff,k)}.

The constants are returned as the landing ``TubeConstants`` so ``tube_margins`` / ``tube_step``
of ``landing_ue_tube`` apply unchanged.
"""

from __future__ import annotations

from math import radians, tan

import numpy as np

from ps2rl.uncertainty.landing_ue_tube import (
    LandingUEConfig,
    TubeConstants,
    make_growth_fn,
    nominal_tube,
    q_of_tau,
    tube_margins,
    tube_step,
)

UEConfig = LandingUEConfig


def tube_constants(cfg, base_set) -> TubeConstants:
    """Tube metric and Lipschitz constants for a cone-obstacle config and its 9-D base set."""
    ctrl = base_set.controller
    p = ctrl.p_matrix_f64()
    ev, evec = np.linalg.eigh(p)
    p_half = evec @ np.diag(np.sqrt(ev)) @ evec.T
    p_mhalf = evec @ np.diag(1.0 / np.sqrt(ev)) @ evec.T
    p_inv = np.linalg.inv(p)
    tt = tan(radians(float(cfg.cone_theta_deg)))
    angs = np.linspace(0.0, 2.0 * np.pi, 1441)
    gs = np.stack([np.cos(angs), np.sin(angs), np.full_like(angs, tt)], axis=1)
    l_cone = float(np.sqrt(np.max(np.einsum("bi,ij,bj->b", gs, p_inv[0:3, 0:3], gs))))
    return TubeConstants(
        p=p, p_half=p_half, p_mhalf=p_mhalf,
        gamma=float(np.sqrt(np.linalg.eigvalsh(p[3:6, 3:6]).max())),
        l_cone=l_cone, l_floor=float(np.sqrt(p_inv[2, 2])), c_b=float(base_set.base_set_c),
    )


__all__ = ["UEConfig", "make_growth_fn", "nominal_tube", "q_of_tau", "tube_constants", "tube_margins", "tube_step"]
