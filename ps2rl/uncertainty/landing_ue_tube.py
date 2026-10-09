"""UE-bCBF tube for the landing task: margins shared by Phase I and the Phase-II filter.

Model (UE-bCBF, Assumption on the disturbance):

    x_dot = f(x) + g(x) u + E_d d(t),   ||d|| <= delta_d,  ||d_dot|| <= delta_v,

with d a world-frame translational acceleration (E_d puts it on v_dot). A disturbance
observer supplies d_hat(t) with ||d(t) - d_hat(t)|| <= e_bar after warm-up.

Backup flow used by Phase I and by the filter: the composed backup pi_b(x, d_hat) under
the *frozen* estimate,

    x_{k+1} = Post(x_k + dt (f(x_k) + g(x_k) pi_b(x_k, d_hat) + E_d d_hat)).

The true flow sees d(t + tau) instead of d_hat(t); the difference is bounded by

    ||d(t + tau) - d_hat(t)|| <= q(tau) := e_bar + min(delta_v tau, 2 delta_d).

Tube (first order, P-metric). With e = xi(x_true) - xi(x_nom) in the 9-D LQR error
coordinates xi = (dp, zeta - z_des, v, phi) and ||e||_P = sqrt(e^T P e) (P: the hover-LQR
DARE matrix that also defines the base set), the discrete recursion

    s_{k+1} = ||F_k||_P s_k + dt gamma q(tau_k),     s_0 = 0,
    F_k = d xi_{k+1} / d xi_k  of the frozen-estimate closed-loop step,
    gamma = ||P^{1/2} E_d||  (= sqrt(lambda_max(P_vv))),

bounds ||e_k||_P to first order (the same order as the experimental quadrotor UE-bCBF's
generator tube; ``tube_scale`` is the same kind of empirical inflation). The P-metric is
the natural one here: the hover LQR contracts in it (||A_cl||_P = 0.981 per step, i.e.
-0.99 /s), so the tube saturates instead of growing exponentially once the backup has
handed off.

Margins (exact for each barrier's structure, s_eff = tube_scale * s):

    cone  : |h_cone(x) - h_cone(x_nom)| <= L_cone s_eff,   L_cone = max_{|g_xy|<=1} ||P^{-1/2} (g_xy, tan th, 0..)||
    floor : |z - z_nom|                 <= L_floor s_eff,  L_floor = sqrt((P^{-1})_zz)
    base  : |V(x) - V(x_nom)|          <= 2 sqrt(c_B) s_eff + s_eff^2  (V = e^T P e, exact quadratic)

so the robust (tightened) sets at backup time tau_k are

    S_k = {h_cone >= L_cone s_eff,k} n {z >= L_floor s_eff,k},   B_k = {V <= c_B - 2 sqrt(c_B) s_eff,k - s_eff,k^2}.

Phase I learns a backup that reaches B_k without leaving S_k (first hit); the Phase-II
filter enforces the same tightened rows plus the observer-error term of UE-bCBF.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from math import isfinite, pi
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class LandingUEConfig:
    """Disturbance bounds and tube settings (single source of truth for Phase I and II)."""

    delta_d: float = 0.5  # ||d|| bound [m/s^2]
    frequency_hz: float = 0.05  # sinusoid frequency used for delta_v = 2 pi f delta_d (if delta_v < 0)
    delta_v: float = -1.0  # ||d_dot|| bound [m/s^3]; < 0 -> 2 pi f delta_d
    observer_lambda: float = 20.0  # disturbance-observer gain (UE-bCBF observer)
    e_bar: float = 0.02  # design bound on ||d - d_hat|| after observer warm-up [m/s^2]
    tube_scale: float = 1.2  # empirical inflation of the first-order tube (as in the quadrotor UE-bCBF)
    d_hat_max: float = -1.0  # Phase-I sampling radius of d_hat; < 0 -> delta_d + e_bar
    s_clip: float = 20.0  # tube values above this are clipped in observations (all sets are empty there)

    def __post_init__(self) -> None:
        for name in ("delta_d", "frequency_hz", "observer_lambda", "e_bar", "tube_scale", "s_clip"):
            v = float(getattr(self, name))
            if not isfinite(v) or v < 0.0:
                raise ValueError(f"{name} must be finite and >= 0, got {v}")
        if float(self.observer_lambda) <= 0.0:
            raise ValueError("observer_lambda must be > 0")
        if float(self.delta_v) < 0.0:
            object.__setattr__(self, "delta_v", float(2.0 * pi * float(self.frequency_hz) * float(self.delta_d)))
        if float(self.d_hat_max) < 0.0:
            object.__setattr__(self, "d_hat_max", float(self.delta_d) + float(self.e_bar))
        steady = float(self.delta_v) / float(self.observer_lambda)
        if float(self.e_bar) + 1e-12 < steady:
            raise ValueError(
                f"e_bar={self.e_bar} is below the observer's steady-state bound delta_v/lambda={steady:.4g}; "
                "the tube would not cover the estimation error"
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LandingUEConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in names})

    def replace(self, **kw: Any) -> "LandingUEConfig":
        d = self.as_dict()
        d.update(kw)
        return LandingUEConfig(**d)

    def observer_error_bound(self, t: float) -> float:
        """Analytic UE-bCBF bound e_bar(t) for an observer started with d_hat(0) = 0."""
        lam = float(self.observer_lambda)
        decay = np.exp(-lam * float(t))
        return float(decay * float(self.delta_d) + (float(self.delta_v) / lam) * (1.0 - decay))

    def warmup_time(self) -> float:
        """Smallest t with e_bar(t) <= e_bar (observer warm-up before the bound holds)."""
        lam = float(self.observer_lambda)
        steady = float(self.delta_v) / lam
        num = float(self.delta_d) - steady
        den = float(self.e_bar) - steady
        if num <= 0.0:
            return 0.0
        if den <= 0.0:
            return float("inf")
        return float(np.log(num / den) / lam)


@dataclass(frozen=True)
class TubeConstants:
    """Metric and Lipschitz constants derived from the landing config (all float64 numpy)."""

    p: np.ndarray  # (9, 9)
    p_half: np.ndarray
    p_mhalf: np.ndarray
    gamma: float  # ||P^{1/2} E_d||
    l_cone: float
    l_floor: float
    c_b: float

    def as_dict(self) -> dict[str, float]:
        return {"gamma": self.gamma, "l_cone": self.l_cone, "l_floor": self.l_floor, "c_b": self.c_b}


def tube_constants(landing_cfg: Any) -> TubeConstants:
    from ps2rl.base_controller.quadrotor_landing_dlqr import QuadrotorLandingDLQR

    ctrl = QuadrotorLandingDLQR.from_config(landing_cfg)
    p = ctrl.p_matrix_f64()
    ev, evec = np.linalg.eigh(p)
    p_half = evec @ np.diag(np.sqrt(ev)) @ evec.T
    p_mhalf = evec @ np.diag(1.0 / np.sqrt(ev)) @ evec.T
    p_inv = np.linalg.inv(p)
    gamma = float(np.sqrt(np.linalg.eigvalsh(p[3:6, 3:6]).max()))
    tt = float(np.tan(np.deg2rad(float(landing_cfg.cone_theta_deg))))
    angs = np.linspace(0.0, 2.0 * np.pi, 1441)
    gs = np.stack([np.cos(angs), np.sin(angs), np.full_like(angs, tt)], axis=1)
    # max over the unit disk of a convex quadratic is attained on the circle
    l_cone = float(np.sqrt(np.max(np.einsum("bi,ij,bj->b", gs, p_inv[0:3, 0:3], gs))))
    l_floor = float(np.sqrt(p_inv[2, 2]))
    return TubeConstants(p=p, p_half=p_half, p_mhalf=p_mhalf, gamma=gamma, l_cone=l_cone, l_floor=l_floor,
                         c_b=float(landing_cfg.base_set_c))


def q_of_tau(tau: jax.Array, ue: LandingUEConfig) -> jax.Array:
    """Bound on ||d(t + tau) - d_hat(t)|| for the frozen estimate."""
    tau = jnp.asarray(tau)
    return float(ue.e_bar) + jnp.minimum(float(ue.delta_v) * tau, 2.0 * float(ue.delta_d))


def tube_step(s: jax.Array, growth: jax.Array, tau: jax.Array, ue: LandingUEConfig, tc: TubeConstants, dt: float):
    """s_{k+1} = ||F_k||_P s_k + dt gamma q(tau_k)."""
    return growth * s + float(dt) * tc.gamma * q_of_tau(tau, ue)


def tube_margins(s: jax.Array, ue: LandingUEConfig, tc: TubeConstants) -> tuple[jax.Array, jax.Array, jax.Array]:
    """(m_cone, m_floor, m_base) for tube value s (any shape)."""
    se = float(ue.tube_scale) * jnp.asarray(s)
    return tc.l_cone * se, tc.l_floor * se, 2.0 * float(np.sqrt(tc.c_b)) * se + se * se


def make_growth_fn(step_fn: Callable[[jax.Array], jax.Array], controller: Any, tc: TubeConstants, *, dtype=jnp.float32):
    """x -> ||F||_P for the closed-loop discrete step ``step_fn`` (x -> x+), in LQR error coordinates.

    F = d xi(x+) / d xi(x), evaluated at the chart representative of x (q_w >= 0); the
    induced P-norm is sqrt(lambda_max(M^T M)) with M = P^{1/2} F P^{-1/2}.
    """
    ph = jnp.asarray(tc.p_half, dtype=dtype)
    pmh = jnp.asarray(tc.p_mhalf, dtype=dtype)

    def growth(x: jax.Array) -> jax.Array:
        xi = controller.error_state(x)
        xc = controller.state_from_error(xi)
        dx_dxi = jax.jacfwd(controller.state_from_error)(xi)
        f10 = jax.jacfwd(step_fn)(xc)
        dxi_dx = jax.jacfwd(controller.error_state)(step_fn(xc))
        f9 = dxi_dx @ f10 @ dx_dxi
        m = ph @ f9 @ pmh
        lam = jnp.linalg.eigvalsh(m.T @ m)[-1]
        return jnp.sqrt(jnp.maximum(lam, 0.0))

    return growth


def nominal_tube(n_steps: int, growth_const: float, ue: LandingUEConfig, tc: TubeConstants, dt: float) -> np.ndarray:
    """Tube s_k (k = 0..n_steps) for a constant per-step growth (reporting only)."""
    s = np.zeros(n_steps + 1)
    for k in range(n_steps):
        tau = k * dt
        s[k + 1] = growth_const * s[k] + dt * tc.gamma * (ue.e_bar + min(ue.delta_v * tau, 2.0 * ue.delta_d))
    return s


__all__ = [
    "LandingUEConfig",
    "TubeConstants",
    "make_growth_fn",
    "nominal_tube",
    "q_of_tau",
    "tube_constants",
    "tube_margins",
    "tube_step",
]
