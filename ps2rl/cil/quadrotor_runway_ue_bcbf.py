"""UE-bCBF control-invariant layer for the runway bird-deterrence task.

Same construction as ``quadrotor_landing_ue_bcbf`` (frozen-estimate backup rollout with
exact discrete sensitivities, tube-tightened rows with the observer-error term, QP, discrete
safeguard), with the runway sets:

    safe rows  (every node k):  h_rwy = y_edge - p_y  (margin m_y(s_k)),  h_ceil = z_max - p_z  (margin m_z(s_k))
    base row   (node N):        V7 <= c_B - m_B(s_N)          (retreat-at-altitude LQR ellipsoid)

The backup pi_b(x, d_hat) is the Phase-I actor (input (x with p_x = 0, d_hat)) outside B and the
retreat LQR on B; everything is invariant to p_x. Rows depend on (x, d_hat, e_bar) only, so
the Phase-II trainer can cache them exactly as for landing. Geometry, base set and disturbance
bounds come from the checkpoint (``runway_backup_policy_actor.pkl``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose, isfinite
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.backup_policy.backup_policy import BackupPolicy
from ps2rl.backup_policy.quadrotor_learned_backup import load_learned_quadrotor_backup_policy
from ps2rl.cil.backup_cbf import BackupCBFConfig
from ps2rl.cil.quadrotor_backup_cbf import QuadrotorBCBFConfig
from ps2rl.cil.backup_cbf import _build_backup_cbf_qp_from_rows
from ps2rl.cil.quadrotor_landing_ue_bcbf import UERuntime, make_recoverability_fn_ue, rollout_ue, safeguard
import qpax
from ps2rl.envs.quadrotor_env import quadrotor_control_affine_terms, quadrotor_dynamics
from ps2rl.envs.quadrotor_runway_config import QuadrotorRunwayConfig
from ps2rl.phase1_sa.runway_ue_sa_env import runway_ue_step_fn
from ps2rl.sets.runway_sets import build_runway_sets
from ps2rl.uncertainty.runway_ue_tube import UEConfig, metric_chart, tube_constants, tube_margins
from ps2rl.utils.paths import load_pickle_payload

Array = jax.Array
CKPT_NAME = "runway_backup_policy_actor.pkl"


def _qp_default(name: str) -> Any:
    return getattr(QuadrotorBCBFConfig, name)


@dataclass(frozen=True)
class RunwayUEBCBFConfig(BackupCBFConfig):
    backup_policy_mode: str = "learned"
    learned_backup_policy_path: str = ""
    runway: QuadrotorRunwayConfig = field(default_factory=QuadrotorRunwayConfig)
    ue: UEConfig = field(default_factory=UEConfig)
    rho_scale: float = 1.0
    alpha: float = 10.0
    alpha_ceiling: float | None = None  # None -> alpha
    base_alpha: float = _qp_default("base_alpha")
    slack_weight: float = _qp_default("slack_weight")
    solver_tol: float = _qp_default("solver_tol")
    use_analytic_jacobian: bool = False
    qp_solve_dtype: str = "float32"
    discrete_safeguard: bool = True
    safeguard_lambdas: tuple = (1.0, 0.75, 0.5, 0.25, 0.0)
    # QP objective weights on (a, omega_x, omega_y, omega_z): min sum_i w_i (u_i - u_ref,i)^2. With equal
    # weights the minimum-norm correction spreads into the yaw rate (when tilted, yawing turns the thrust
    # away from the runway), and the filtered chaser spins at |omega_z| ~ 4 rad/s on average.
    qp_action_weights: tuple = (1.0, 1.0, 1.0, 25.0)

    def __post_init__(self) -> None:
        rc = self.runway
        for name, given, derived in (("T", self.T, rc.T), ("dt", self.dt, rc.dt)):
            if given is not None and not isclose(float(given), float(derived), rel_tol=1e-9):
                raise ValueError(f"{name}={given} disagrees with the runway config ({derived})")
        object.__setattr__(self, "T", float(rc.T))
        object.__setattr__(self, "dt", float(rc.dt))
        object.__setattr__(self, "num_steps", int(rc.num_steps))
        object.__setattr__(self, "base_set_c", float(rc.base_set_c))
        lams = tuple(float(v) for v in self.safeguard_lambdas)
        if not lams or lams[-1] != 0.0 or any(a <= b for a, b in zip(lams, lams[1:])) or lams[0] > 1.0:
            raise ValueError(f"safeguard_lambdas must be strictly decreasing in (0, 1] and end with 0, got {lams}")
        object.__setattr__(self, "safeguard_lambdas", lams)
        if self.qp_solve_dtype not in ("float32", "float64"):
            raise ValueError("qp_solve_dtype must be 'float32' or 'float64'")
        if self.qp_solve_dtype == "float64" and not jax.config.jax_enable_x64:
            raise ValueError("qp_solve_dtype='float64' needs JAX_ENABLE_X64=1")
        for name in ("alpha", "base_alpha", "slack_weight", "solver_tol"):
            v = float(getattr(self, name))
            if not isfinite(v) or v <= 0.0:
                raise ValueError(f"{name} must be positive and finite, got {v}")
        for name in ("alpha", "alpha_ceiling", "base_alpha"):
            a = getattr(self, name)
            if a is not None and float(a) * float(rc.dt) >= 1.0:
                raise ValueError(f"{name}*dt must be < 1")

    @property
    def gravity(self) -> float:
        return float(self.runway.gravity)

    @property
    def a_cmd_min(self) -> float:
        return float(self.runway.a_cmd_min)

    @property
    def a_cmd_max(self) -> float:
        return float(self.runway.a_cmd_max)

    @property
    def omega_max(self) -> float:
        return float(self.runway.omega_max)

    @property
    def action_low(self) -> tuple[float, ...]:
        return (self.a_cmd_min, -self.omega_max, -self.omega_max, -self.omega_max)

    @property
    def action_high(self) -> tuple[float, ...]:
        return (self.a_cmd_max, self.omega_max, self.omega_max, self.omega_max)

    @property
    def action_scale(self) -> jax.Array:
        return jnp.array([self.a_cmd_max, self.omega_max, self.omega_max, self.omega_max], dtype=jnp.float32)

    @property
    def num_safe_constraints(self) -> int:
        return 2

    @property
    def alpha_per_constraint(self) -> tuple[float, float]:
        return (float(self.alpha), float(self.alpha if self.alpha_ceiling is None else self.alpha_ceiling))

    @property
    def relative_time_per_constraint(self) -> tuple[bool, bool]:
        return (bool(self.include_relative_time_term),) * 2

    @property
    def num_backup_inequalities(self) -> int:
        return 2 * (self.num_steps + 1) + 1

    @property
    def num_qp_inequalities(self) -> int:
        return self.num_backup_inequalities + 2 * 4 + 1


def read_runway_checkpoint_metadata(path: str | Path) -> dict:
    p = Path(path)
    if p.is_dir():
        p = p / CKPT_NAME
    md = dict(load_pickle_payload(p).get("metadata", {}))
    if "runway_config" not in md or "ue_config" not in md:
        raise KeyError(f"{p} is not a runway UE Phase-I checkpoint (train it with scripts/train_phase1_runway_ue.py)")
    return md


def runway_ue_bcbf_config_from_checkpoint(path: str | Path, **overrides: Any) -> RunwayUEBCBFConfig:
    p = Path(path)
    ckpt = p / CKPT_NAME if p.is_dir() else p
    md = read_runway_checkpoint_metadata(ckpt)
    rc = QuadrotorRunwayConfig.from_dict(md["runway_config"])
    ue_ck = UEConfig.from_dict(md["ue_config"])
    ue = overrides.pop("ue", ue_ck)
    for name in ("delta_d", "delta_v", "e_bar"):
        if float(getattr(ue, name)) > float(getattr(ue_ck, name)) + 1e-12:
            raise ValueError(f"runtime ue.{name}={getattr(ue, name)} exceeds the Phase-I value {getattr(ue_ck, name)}")
    if float(ue.tube_scale) < float(ue_ck.tube_scale) - 1e-12:
        raise ValueError("runtime tube_scale below the Phase-I value would loosen the tube")
    return RunwayUEBCBFConfig(learned_backup_policy_path=str(ckpt), runway=rc, ue=ue, **overrides)


def make_ue_runtime(cfg: RunwayUEBCBFConfig) -> UERuntime:
    rc = cfg.runway
    safe_set, base_set = build_runway_sets(rc)
    learned = load_learned_quadrotor_backup_policy(cfg.learned_backup_policy_path)
    trained = QuadrotorRunwayConfig.from_dict(learned.metadata["runway_config"]).as_dict()
    bad = [k for k, v in rc.as_dict().items() if not np.isclose(float(trained[k]), float(v), rtol=1e-9, atol=1e-12)]
    if bad:
        raise ValueError(f"runway config differs from the checkpoint on {bad}")
    if int(learned.actor_cfg.obs_dim) != 13:
        raise ValueError(f"runway UE backup must take (x, d_hat) (obs_dim 13), got {learned.actor_cfg.obs_dim}")
    low = jnp.asarray(cfg.action_low, dtype=jnp.float32)
    high = jnp.asarray(cfg.action_high, dtype=jnp.float32)
    plant = runway_ue_step_fn(rc)
    tc = tube_constants(rc)
    chart = metric_chart(rc)
    ue = cfg.ue

    def pi_b(x, d_hat):
        xa = x.at[0].set(0.0)  # the backup never sees p_x
        raw = learned.action_single(jnp.concatenate([xa, d_hat]).astype(jnp.float32)).astype(x.dtype)
        raw = jnp.clip(jnp.nan_to_num(raw), low, high)
        return jnp.clip(BackupPolicy.select_action(x, raw, base_set), low, high)

    def step(x, d_hat):
        return plant(x, pi_b(x, d_hat), d_hat)

    def f_cl(x, d_hat):
        xd = quadrotor_dynamics(x, pi_b(x, d_hat), rc.gravity, rc.a_cmd_min, rc.a_cmd_max, rc.omega_max)
        return xd.at[3:6].add(d_hat).at[0].set(0.0)

    ph = jnp.asarray(tc.p_half, dtype=jnp.float32)
    pmh = jnp.asarray(tc.p_mhalf, dtype=jnp.float32)

    def growth_from_f10(x, x_next, f10):
        xi = chart.error_state(x)
        dx_dxi = jax.jacfwd(chart.state_from_error)(xi)
        dxi_dx = jax.jacfwd(chart.error_state)(x_next)
        m = ph @ (dxi_dx @ f10 @ dx_dxi) @ pmh
        return jnp.sqrt(jnp.maximum(jnp.linalg.eigvalsh(m.T @ m)[-1], 0.0))

    def safe_tight(x, s):
        m_y, m_z, _ = tube_margins(s, ue, tc)
        h = safe_set.component_values(x)
        return (h[0] >= m_y) & (h[1] >= m_z)

    def base_tight(x, s):
        return base_set.margin(x) >= tube_margins(s, ue, tc)[2]

    qp_system = SimpleNamespace(action_dim=4, action_low=tuple(cfg.action_low), action_high=tuple(cfg.action_high))
    return UERuntime(pi_b=pi_b, step=step, plant=plant, f_cl=f_cl, safe_vg=safe_set.values_and_grads,
                     base_vg=base_set.values_and_grads, base_contains=base_set.contains, safe_contains_tight=safe_tight,
                     base_contains_tight=base_tight, growth_from_f10=growth_from_f10, tube=tc, qp_system=qp_system)


def build_ue_rows(x: Array, d_hat: Array, e_bar: Array, cfg: RunwayUEBCBFConfig, rt: UERuntime):
    """A u <= b rows (2 (N+1) + 1) and diagnostics; depends on (x, d_hat, e_bar) only."""
    xs, phis, thetas, ss = rollout_ue(x, d_hat, cfg, rt)
    f0, g0 = quadrotor_control_affine_terms(x, cfg.gravity)
    ed = jnp.zeros((10, 3), x.dtype).at[3:6].set(jnp.eye(3, dtype=x.dtype))
    f0_hat = f0 + ed @ d_hat
    lam = float(cfg.ue.observer_lambda)
    alphas = jnp.asarray(cfg.alpha_per_constraint, dtype=x.dtype)
    rts = jnp.asarray(cfg.relative_time_per_constraint, dtype=x.dtype)
    rho_scale = float(cfg.rho_scale)

    def per_node(xi, phi, theta, s):
        h, dh = rt.safe_vg(xi)
        m_y, m_z, _ = tube_margins(s, cfg.ue, rt.tube)
        m = jnp.stack([m_y, m_z])
        a = -(dh @ (phi @ g0))
        flow = dh @ (phi @ f0_hat) - rts * (dh @ rt.f_cl(xi, d_hat))
        rho = e_bar * jnp.linalg.norm(dh @ (phi @ ed + lam * theta), axis=-1)
        return a, alphas * (h - m) + flow - rho_scale * rho, h - m, rho

    a_seq, b_seq, h_rob, rho_seq = jax.vmap(per_node)(xs, phis, thetas, ss)
    a_rows = a_seq.reshape((-1, 4))
    b_rows = b_seq.reshape((-1,))
    hb, dhb = rt.base_vg(xs[-1])
    m_b = tube_margins(ss[-1], cfg.ue, rt.tube)[2]
    a_t = -(dhb @ (phis[-1] @ g0))
    rho_t = e_bar * jnp.linalg.norm(dhb @ (phis[-1] @ ed + lam * thetas[-1]), axis=-1)
    b_t = cfg.base_alpha * (hb - m_b) + dhb @ (phis[-1] @ f0_hat) - rho_scale * rho_t
    a_rows = jnp.concatenate([a_rows, a_t], axis=0)
    b_rows = jnp.concatenate([b_rows, b_t], axis=0)
    diag = {"tube_T": ss[-1], "min_robust_safe_h": jnp.min(h_rob), "terminal_robust_h_b": (hb - m_b)[0],
            "max_rho": jnp.maximum(jnp.max(rho_seq), rho_t[0]),
            "rows_finite": jnp.all(jnp.isfinite(a_rows)) & jnp.all(jnp.isfinite(b_rows))}
    return a_rows, b_rows, ss, diag


def solve_qp_from_rows(a_rows, b_rows, u_ref, u_backup, cfg: RunwayUEBCBFConfig, rt: UERuntime, qp_dtype=None):
    """min sum_i w_i (u_i - u_ref,i)^2 + slack  s.t. rows (with slack), box. Differentiable in u_ref.

    Solved in the scaled input u~ = W^{1/2} u (rows A W^{-1/2}, box W^{1/2} [lo, hi]); otherwise the landing
    UE solve. Returns (u, slack, used_solver); non-finite rows/solution -> u_backup (no gradient).
    """
    dtype = u_ref.dtype
    lo, hi = jnp.asarray(cfg.action_low, dtype), jnp.asarray(cfg.action_high, dtype)
    sw = jnp.sqrt(jnp.asarray(cfg.qp_action_weights, dtype))
    u_ref = jnp.clip(jnp.nan_to_num(u_ref), lo, hi)
    sys_s = SimpleNamespace(action_dim=4, action_low=tuple(np.sqrt(np.asarray(cfg.qp_action_weights)) * np.asarray(cfg.action_low)),
                            action_high=tuple(np.sqrt(np.asarray(cfg.qp_action_weights)) * np.asarray(cfg.action_high)))
    q_mat, q_vec, a_eq, b_eq, g, h = _build_backup_cbf_qp_from_rows(a_rows / sw[None, :], b_rows, u_ref * sw, cfg, sys_s,
                                                                    dtype=dtype)
    ok_in = jnp.all(jnp.isfinite(g)) & jnp.all(jnp.isfinite(h))
    g = jnp.where(ok_in, g, 0.0)
    h = jnp.where(ok_in, h, 1.0)
    cast = (lambda t: t) if qp_dtype is None else (lambda t: t.astype(qp_dtype))
    z = qpax.solve_qp_primal(cast(q_mat), cast(q_vec), cast(a_eq), cast(b_eq), cast(g), cast(h), solver_tol=cfg.solver_tol,
                             target_kappa=cfg.target_kappa).astype(dtype)
    used = ok_in & jnp.all(jnp.isfinite(z))
    z = jnp.concatenate([z[:4] / sw, z[4:]])
    fb = jnp.concatenate([u_backup.astype(dtype), jnp.zeros((1,), dtype)])
    z = jnp.where(used, jnp.nan_to_num(z), jax.lax.stop_gradient(fb))
    return jnp.clip(z[:4], lo, hi), jnp.maximum(z[4], 0.0), used


def project_full(x, d_hat, e_bar, u_ref, cfg, rt, qp_dtype=None, use_safeguard=True):
    """rows -> QP -> safeguard for one state. Returns (u_safe, aux incl. rows for caching)."""
    a_rows, b_rows, ss, diag = build_ue_rows(x, d_hat, e_bar, cfg, rt)
    ub = rt.pi_b(x, d_hat)
    u_qp, slack, used = solve_qp_from_rows(a_rows, b_rows, u_ref, ub, cfg, rt, qp_dtype)
    if use_safeguard and bool(cfg.discrete_safeguard):
        u, lam, found = safeguard(x, d_hat, u_qp, ss, cfg, rt)
    else:
        u, lam, found = u_qp, jnp.asarray(1.0, u_qp.dtype), jnp.asarray(True)
    aux = {"a_rows": a_rows, "b_rows": b_rows, "u_backup": ub, "u_qp": u_qp, "slack": slack, "used_solver": used,
           "safeguard_lambda": lam, "safeguard_found": found, **diag}
    return u, aux


make_recoverability_fn = make_recoverability_fn_ue  # (x0, d_hat) batch -> tightened first-hit C_N membership

_CACHE: Dict[tuple, UERuntime] = {}


def get_cached_runtime(cfg: RunwayUEBCBFConfig) -> UERuntime:
    key = (cfg.learned_backup_policy_path, cfg.runway, cfg.ue)
    if key not in _CACHE:
        _CACHE[key] = make_ue_runtime(cfg)
    return _CACHE[key]


__all__ = ["CKPT_NAME", "RunwayUEBCBFConfig", "build_ue_rows", "get_cached_runtime", "make_recoverability_fn",
           "make_ue_runtime", "project_full", "read_runway_checkpoint_metadata", "runway_ue_bcbf_config_from_checkpoint",
           "solve_qp_from_rows"]
