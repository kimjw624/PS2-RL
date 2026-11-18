"""Backup-CBF filter for the cone-obstacle task (outside of a funnel, above the ground).

The counterpart of ``quadrotor_landing_backup_cbf`` on the same shared engine
(``ps2rl.cil.backup_cbf``): rows for the funnel and the ground at every backup node, the
base-set rows at the end of the horizon, the QP, and the discrete safeguard. The backup
policy is the base set's own controller (``base_controller``) or, later, a learned
safe-arrival policy that hands off to it inside B.

    cfg = ConeObstacleBCBFConfig(scenario=ConeObstacleConfig(...), alpha=10, alpha_ground=20)
    u, slack, used, info = solve_backup_cbf_qp_batch_with_info(x_batch, u_ref_batch, cfg)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any

import jax
import jax.numpy as jnp

from ps2rl.cil.backup_cbf import BackupCBFConfig, BCBFSystem, make_backup_cbf_facades
from ps2rl.cil.quadrotor_backup_cbf import QuadrotorBCBFConfig
from ps2rl.envs.quadrotor_cone_obstacle_config import ConeObstacleConfig
from ps2rl.envs.quadrotor_env import quadrotor_control_affine_terms, quadrotor_dynamics, quadrotor_step_euler
from ps2rl.sets.quadrotor_cone_obstacle_sets import obstacle_sets
from ps2rl.utils.quaternion import normalize_quaternion

BACKUP_MODES = ("base_controller",)


def _qp_default(name: str) -> Any:
    return next(f.default for f in QuadrotorBCBFConfig.__dataclass_fields__.values() if f.name == name)


@dataclass(frozen=True)
class ConeObstacleBCBFConfig(BackupCBFConfig):
    backup_policy_mode: str = "base_controller"
    learned_backup_policy_path: str = ""
    scenario: ConeObstacleConfig = field(default_factory=ConeObstacleConfig)

    alpha: float = _qp_default("alpha")
    alpha_ground: float | None = None
    base_alpha: float = _qp_default("base_alpha")
    slack_weight: float = _qp_default("slack_weight")
    solver_tol: float = _qp_default("solver_tol")
    use_analytic_jacobian: bool = False
    qp_solve_dtype: str = "float32"
    sensitivity_propagation: str = "discrete"
    reduced_sensitivity: bool = True
    discrete_safeguard: bool = True
    safeguard_lambdas: tuple = (1.0, 0.75, 0.5, 0.25, 0.0)
    control_weights: tuple | None = None

    def __post_init__(self) -> None:
        if self.backup_policy_mode not in BACKUP_MODES:
            raise ValueError(f"backup_policy_mode must be one of {BACKUP_MODES} (learned backups come with Phase I)")
        sc = self.scenario
        if not isinstance(sc, ConeObstacleConfig):
            raise TypeError("scenario must be a ConeObstacleConfig")
        if self.qp_solve_dtype not in ("float32", "float64"):
            raise ValueError(f"qp_solve_dtype must be 'float32' or 'float64', got '{self.qp_solve_dtype}'")
        if self.qp_solve_dtype == "float64" and not jax.config.jax_enable_x64:
            raise ValueError("qp_solve_dtype='float64' needs JAX_ENABLE_X64=1 before JAX is imported")
        lams = tuple(float(v) for v in self.safeguard_lambdas)
        if not lams or lams[-1] != 0.0 or any(a <= b for a, b in zip(lams, lams[1:])) or lams[0] > 1.0:
            raise ValueError(f"safeguard_lambdas must be strictly decreasing in (0, 1] and end with 0, got {lams}")
        object.__setattr__(self, "safeguard_lambdas", lams)
        if self.control_weights is not None:
            w = tuple(float(v) for v in self.control_weights)
            if len(w) != 4 or any((not isfinite(v)) or v <= 0.0 for v in w):
                raise ValueError(f"control_weights must be 4 positive numbers, got {self.control_weights}")
            object.__setattr__(self, "control_weights", w)
        object.__setattr__(self, "T", float(sc.T))
        object.__setattr__(self, "dt", float(sc.dt))
        object.__setattr__(self, "num_steps", int(sc.num_steps))
        object.__setattr__(self, "base_set_c", float(sc.base_set_c))
        for name, a in (("alpha", self.alpha), ("alpha_ground", self.alpha_ground), ("base_alpha", self.base_alpha)):
            if a is not None and (float(a) <= 0.0 or float(a) * float(sc.dt) >= 1.0):
                raise ValueError(f"{name}={a} must be positive with {name}*dt < 1")

    # --- what the engine and the trainers read
    @property
    def gravity(self) -> float:
        return float(self.scenario.gravity)

    @property
    def a_cmd_min(self) -> float:
        return float(self.scenario.a_cmd_min)

    @property
    def a_cmd_max(self) -> float:
        return float(self.scenario.a_cmd_max)

    @property
    def omega_max(self) -> float:
        return float(self.scenario.omega_max)

    @property
    def action_low(self) -> tuple[float, ...]:
        return self.scenario.action_low

    @property
    def action_high(self) -> tuple[float, ...]:
        return self.scenario.action_high

    @property
    def action_scale(self) -> jax.Array:
        return jnp.array(self.scenario.action_scale, dtype=jnp.float32)

    @property
    def num_safe_constraints(self) -> int:
        return 2

    @property
    def alpha_per_constraint(self) -> tuple[float, ...]:
        return (float(self.alpha), float(self.alpha if self.alpha_ground is None else self.alpha_ground))

    @property
    def relative_time_per_constraint(self) -> tuple[bool, ...]:
        return (bool(self.include_relative_time_term),) * 2

    @property
    def num_base_set_constraints(self) -> int:
        return 3 if self.scenario.base_set == "anywhere" else 1

    @property
    def num_backup_inequalities(self) -> int:
        return self.num_safe_constraints * (self.num_steps + 1) + self.num_base_set_constraints

    @property
    def num_qp_inequalities(self) -> int:
        return self.num_backup_inequalities + 2 * len(self.action_low) + 1


def _omega_clip(sc: ConeObstacleConfig):
    w = sc.omega_clip
    return w if isinstance(w, float) else jnp.asarray(w, dtype=jnp.float32)


def obstacle_step(x: jax.Array, u: jax.Array, cfg: ConeObstacleBCBFConfig) -> jax.Array:
    sc = cfg.scenario
    return quadrotor_step_euler(x, u, sc.dt, sc.gravity, sc.a_cmd_min, sc.a_cmd_max, _omega_clip(sc))


def _sanitize_state(x: jax.Array) -> jax.Array:
    x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.at[6:10].set(normalize_quaternion(x[6:10]))


def make_obstacle_runtime(cfg: ConeObstacleBCBFConfig) -> BCBFSystem:
    sc = cfg.scenario
    safe, base = obstacle_sets(sc)
    return BCBFSystem(
        state_dim=10, action_dim=4, action_low=cfg.action_low, action_high=cfg.action_high,
        backup_policy_fn=base.action,
        safe_set_values_and_grads_fn=safe.values_and_grads,
        base_set_values_and_grads_fn=base.values_and_grads,
        dynamics_fn=lambda x, u: quadrotor_dynamics(x, u, sc.gravity, sc.a_cmd_min, sc.a_cmd_max, _omega_clip(sc)),
        control_affine_terms_fn=lambda x: quadrotor_control_affine_terms(x, sc.gravity),
        postprocess_rollout_state_fn=lambda z: z.at[6:10].set(normalize_quaternion(z[6:10])),
        sanitize_solve_state_fn=_sanitize_state,
        solve_state_slice_fn=lambda xb: xb,
        base_set_values_fn=lambda x: base.values_and_grads(x)[0],
        use_analytic_jacobian=False,
        analytic_closed_loop_and_jacobian_fn=None,
        qp_solve_dtype=None if cfg.qp_solve_dtype == "float32" else jnp.float64,
    )


_RUNTIME_CACHE: dict[tuple, BCBFSystem] = {}
_facades = make_backup_cbf_facades(make_obstacle_runtime, lambda cfg: (cfg.backup_policy_mode, cfg.scenario, cfg.qp_solve_dtype),
                                   _RUNTIME_CACHE)
get_cached_runtime = _facades["get_cached_runtime"]
_resolve_runtime = _facades["_resolve_runtime"]
_qp_batch_with_info = _facades["solve_backup_cbf_qp_batch_with_info"]


def first_hit_fn(cfg: ConeObstacleBCBFConfig, rt: BCBFSystem | None = None, *, training_envelope: bool = False):
    """x0 -> (recoverable, crashed, arrival_step): the backup reaches B within N steps without leaving S."""
    rt = _resolve_runtime(cfg, rt)
    safe, base = obstacle_sets(cfg.scenario)
    ok_fn = safe.training_contains if training_envelope else safe.contains
    n = int(cfg.num_steps)

    def one(x0):
        def body(c, k):
            x, hb, hf, tb = c
            xn = obstacle_step(x, rt.backup_policy_fn(x), cfg)
            act = ~(hb | hf)
            nf = act & ~ok_fn(xn)
            nb = act & ~nf & base.contains(xn)
            return (jnp.where(act, xn, x), hb | nb, hf | nf, jnp.where(nb, k + 1, tb)), None

        in_b = base.contains(x0)
        (_, hb, hf, tb), _ = jax.lax.scan(body, (x0, in_b, ~ok_fn(x0), jnp.where(in_b, 0, -1)), jnp.arange(n))
        return hb, hf, tb

    return one


def make_recoverability_fn(cfg: ConeObstacleBCBFConfig, rt: BCBFSystem | None = None, *, training_envelope: bool = False):
    return jax.jit(jax.vmap(first_hit_fn(cfg, rt, training_envelope=training_envelope)))


def apply_discrete_safeguard(x_b, u_b, cfg: ConeObstacleBCBFConfig, rt: BCBFSystem):
    """Largest lambda with F(x, lambda u + (1 - lambda) pi_b(x)) in C_N; returns (u, lambda, found)."""
    lams = jnp.asarray(cfg.safeguard_lambdas, dtype=u_b.dtype)
    ub = jax.vmap(rt.backup_policy_fn)(x_b).astype(u_b.dtype)
    lo, hi = jnp.asarray(cfg.action_low, u_b.dtype), jnp.asarray(cfg.action_high, u_b.dtype)
    cand = jnp.clip(lams[:, None, None] * u_b[None] + (1.0 - lams)[:, None, None] * ub[None], lo, hi)
    n_l, n_b = cand.shape[0], cand.shape[1]
    x_rep = jnp.broadcast_to(x_b[None], (n_l,) + x_b.shape).reshape((n_l * n_b,) + x_b.shape[1:])
    xn = jax.vmap(lambda x, u: obstacle_step(x, u, cfg))(x_rep, cand.reshape((n_l * n_b, -1)).astype(x_b.dtype))
    ok, _, _ = jax.vmap(first_hit_fn(cfg, rt))(xn)
    ok = ok.reshape((n_l, n_b))
    found = jnp.any(ok, axis=0)
    idx = jnp.where(found, jnp.argmax(ok, axis=0), n_l - 1)
    return cand[idx, jnp.arange(n_b)], lams[idx], found


def solve_backup_cbf_qp_batch_with_info(x_batch, u_ref_batch, cfg: ConeObstacleBCBFConfig, runtime=None):
    """Batch BCBF-QP followed by the discrete safeguard (if enabled)."""
    rt = _resolve_runtime(cfg, runtime)
    u, slack, used, info = _qp_batch_with_info(x_batch, u_ref_batch, cfg, rt)
    if bool(cfg.discrete_safeguard):
        u, lam, found = apply_discrete_safeguard(x_batch, u, cfg, rt)
        info = {**info, "safeguard_lambda": lam, "safeguard_found": found}
    return u, slack, used, info


__all__ = [
    "ConeObstacleBCBFConfig",
    "apply_discrete_safeguard",
    "first_hit_fn",
    "get_cached_runtime",
    "make_obstacle_runtime",
    "make_recoverability_fn",
    "obstacle_step",
    "solve_backup_cbf_qp_batch_with_info",
]
