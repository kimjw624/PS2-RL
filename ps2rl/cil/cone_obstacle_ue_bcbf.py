"""UE-bCBF control-invariant layer for the cone-obstacle task (outside of an apex-down funnel).

The landing UE filter (``quadrotor_landing_ue_bcbf``) with the cone-obstacle sets: per solve,
given the physical state x, the observer's estimate d_hat and its error bound e_bar,

1. roll out the backup under the frozen estimate,
       x_{k+1} = Post(x_k + dt (f(x_k) + g(x_k) pi_b(x_k) + E_d d_hat)),
   with the exact sensitivities Phi_k = dx_k/dx_0, Theta_k = dx_k/dd_hat and the P-metric tube
   s_k (``ps2rl.uncertainty.cone_obstacle_ue_tube``);
2. rows for the funnel and the floor at every node and for the base set at the end,
       -grad h_j Phi_k g0 u <= alpha_j (h_j(phi_k) - m_j(s_k)) + grad h_j (Phi_k f0_hat - rt_j f_cl)
                               - e_bar || grad h_j (Phi_k E_d + Theta_k Lambda) ||;
3. the QP (shared engine's conditioning, box, slack) and the discrete safeguard on the tightened C_N.

The row builder, QP, safeguard and recoverability are the landing UE module's functions; this
module supplies the runtime (sets, backup, tube) and the engine-style facade.

Backups (``backup``):
* ``lqr``      the base set's 9-D hover LQR everywhere (no Phase I);
* ``learned``  a Phase-I safe-arrival actor from ``scripts/train_phase1_cone_obstacle_ue.py``
               (hand-off to the LQR inside B). It was trained without a disturbance, so it is
               evaluated with d_hat = 0 in its input; the frozen d_hat still drives the rollout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.backup_policy.backup_policy import BackupPolicy
from ps2rl.backup_policy.quadrotor_learned_backup import load_learned_quadrotor_backup_policy
from ps2rl.cil.quadrotor_cone_obstacle_cbf import ConeObstacleBCBFConfig
from ps2rl.cil.quadrotor_landing_ue_bcbf import (
    UERuntime,
    build_ue_rows,
    make_recoverability_fn_ue,
    project_full,
    rollout_ue,
    safeguard,
    solve_qp_from_rows,
)
from ps2rl.envs.quadrotor_cone_obstacle_config import ConeObstacleConfig
from ps2rl.envs.quadrotor_env import quadrotor_dynamics
from ps2rl.phase1_sa.quadrotor_landing_ue_sa_env import landing_ue_step_fn
from ps2rl.sets.quadrotor_cone_obstacle_sets import obstacle_sets
from ps2rl.uncertainty.cone_obstacle_ue_tube import UEConfig, tube_constants, tube_margins

BACKUPS = ("lqr", "learned")
ACTOR_FILE = "cone_obstacle_backup_policy_actor.pkl"


@dataclass(frozen=True)
class ConeObstacleUEBCBFConfig(ConeObstacleBCBFConfig):
    ue: UEConfig = field(default_factory=UEConfig)
    rho_scale: float = 1.0          # multiplies the observer-error term (1 = the UE-bCBF bound)
    backup: str = "lqr"
    backup_path: str = ""           # Phase-I run directory or actor file (backup = "learned")

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.backup not in BACKUPS:
            raise ValueError(f"backup must be one of {BACKUPS}, got {self.backup!r}")
        if self.backup == "learned" and not self.backup_path:
            raise ValueError("backup='learned' needs backup_path")

    @property
    def num_backup_inequalities(self) -> int:
        return self.num_safe_constraints * (self.num_steps + 1) + 1

    @property
    def alpha_floor(self) -> float:          # name used by the landing UE scripts
        return float(self.alpha if self.alpha_ground is None else self.alpha_ground)

    @property
    def actor_file(self) -> str:
        p = Path(self.backup_path)
        return str(p / ACTOR_FILE if p.is_dir() else p)


def cone_ue_config(scenario: ConeObstacleConfig, ue: UEConfig, *, backup: str = "lqr", backup_path: str = "",
                   **overrides: Any) -> ConeObstacleUEBCBFConfig:
    return ConeObstacleUEBCBFConfig(scenario=scenario, ue=ue, backup=backup, backup_path=backup_path, **overrides)


def cone_ue_config_from_task(task: dict, *, backup: str = "lqr", backup_path: str = "", **overrides: Any) -> ConeObstacleUEBCBFConfig:
    """Scenario, UE bounds and filter gains from a task file (``docker_batch/configs/cone_obstacle_ue.json``)."""
    import sys
    root = Path(__file__).resolve().parents[2]
    if str(root / "scripts") not in sys.path:
        sys.path.insert(0, str(root / "scripts"))
    from cone_obstacle_common import task_config

    flt = task.get("filter", {})
    kw = {"alpha": float(flt.get("alpha", 10.0)), "alpha_ground": float(flt.get("alpha_ground", 20.0)),
          "base_alpha": float(flt.get("base_alpha", 2.0))}
    kw.update(overrides)
    return cone_ue_config(task_config(task), UEConfig.from_dict(task.get("ue", {})), backup=backup,
                          backup_path=backup_path, **kw)


def make_ue_runtime(cfg: ConeObstacleUEBCBFConfig) -> UERuntime:
    sc = cfg.scenario
    safe, base_set = obstacle_sets(sc)
    ctrl = base_set.controller
    low = jnp.asarray(cfg.action_low, dtype=jnp.float32)
    high = jnp.asarray(cfg.action_high, dtype=jnp.float32)
    plant = landing_ue_step_fn(sc)
    tc = tube_constants(sc, base_set)
    ue = cfg.ue

    if cfg.backup == "learned":
        learned = load_learned_quadrotor_backup_policy(cfg.actor_file)
        md = learned.metadata
        trained = ConeObstacleConfig.from_dict(md["cone_obstacle_config"])
        for k in ("tip_z", "cone_theta_deg", "cone_eps", "ground_z", "min_altitude", "base_set_c", "a_cmd_min",
                  "a_cmd_max", "omega_max", "dt"):
            if not np.isclose(float(getattr(trained, k)), float(getattr(sc, k))):
                raise ValueError(f"backup {cfg.actor_file} was trained with {k}={getattr(trained, k)}, the filter has {getattr(sc, k)}")
        if tuple(trained.hover_points) != tuple(sc.hover_points):
            raise ValueError("backup was trained for another base-set point")
        n_in = int(learned.actor_cfg.obs_dim)

        def raw_action(x, d_hat):
            o = jnp.concatenate([x, jnp.zeros((3,), x.dtype)]) if n_in == 13 else x
            return learned.action_single(o.astype(jnp.float32)).astype(x.dtype)
    else:
        def raw_action(x, d_hat):
            return ctrl.action(x)

    def pi_b(x, d_hat):
        raw = jnp.clip(jnp.nan_to_num(raw_action(x, d_hat)), low, high)
        return jnp.clip(BackupPolicy.select_action(x, raw, base_set), low, high)

    def step(x, d_hat):
        return plant(x, pi_b(x, d_hat), d_hat)

    def f_cl(x, d_hat):
        u = pi_b(x, d_hat)
        xd = quadrotor_dynamics(x, u, sc.gravity, sc.a_cmd_min, sc.a_cmd_max, sc.omega_max)
        return xd.at[3:6].add(d_hat)

    ph = jnp.asarray(tc.p_half, dtype=jnp.float32)
    pmh = jnp.asarray(tc.p_mhalf, dtype=jnp.float32)

    def growth_from_f10(x, x_next, f10):
        xi = ctrl.error_state(x)
        dx_dxi = jax.jacfwd(ctrl.state_from_error)(xi)
        dxi_dx = jax.jacfwd(ctrl.error_state)(x_next)
        m = ph @ (dxi_dx @ f10 @ dx_dxi) @ pmh
        return jnp.sqrt(jnp.maximum(jnp.linalg.eigvalsh(m.T @ m)[-1], 0.0))

    def safe_tight(x, s):
        m_c, m_f, _ = tube_margins(s, ue, tc)
        h = safe.component_values(x)
        return (h[0] >= m_c) & (h[1] >= m_f)

    def base_tight(x, s):
        _, _, m_b = tube_margins(s, ue, tc)
        return base_set.margin(x) >= m_b

    qp_system = SimpleNamespace(action_dim=4, action_low=tuple(cfg.action_low), action_high=tuple(cfg.action_high))
    return UERuntime(pi_b=pi_b, step=step, plant=plant, f_cl=f_cl, safe_vg=safe.values_and_grads,
                     base_vg=base_set.values_and_grads, base_contains=base_set.contains,
                     safe_contains_tight=safe_tight, base_contains_tight=base_tight,
                     growth_from_f10=growth_from_f10, tube=tc, qp_system=qp_system)


# ------------------------------------------------------------- engine-style facade (x_d = (x, d_hat))
_UE_RUNTIME_CACHE: Dict[tuple, UERuntime] = {}


def get_cached_runtime(cfg: ConeObstacleUEBCBFConfig) -> UERuntime:
    key = (cfg.backup, cfg.backup_path, cfg.scenario, cfg.ue)
    if key not in _UE_RUNTIME_CACHE:
        _UE_RUNTIME_CACHE[key] = make_ue_runtime(cfg)
    return _UE_RUNTIME_CACHE[key]


def _qp_dtype(cfg):
    return jnp.float64 if str(getattr(cfg, "qp_solve_dtype", "float32")) == "float64" else None


def solve_backup_cbf_qp_batch_with_info(xd_batch, u_ref_batch, cfg, runtime=None, qp_dtype=None):
    """Batch UE projection. Returns (u_safe, slack, used_solver, info)."""
    rt = get_cached_runtime(cfg) if runtime is None else runtime
    xd_batch = jnp.asarray(xd_batch)
    x = jnp.nan_to_num(xd_batch[..., :10]).astype(jnp.float32)
    dh = jnp.nan_to_num(xd_batch[..., 10:13]).astype(jnp.float32)
    u_ref = jnp.asarray(u_ref_batch).astype(jnp.float32)
    e_bar = jnp.asarray(float(cfg.ue.e_bar), jnp.float32)
    qd = qp_dtype if qp_dtype is not None else _qp_dtype(cfg)
    u, aux = jax.vmap(lambda xx, dd, uu: project_full(xx, dd, e_bar, uu, cfg, rt, qd))(x, dh, u_ref)
    info = {"safeguard_lambda": aux["safeguard_lambda"], "safeguard_found": aux["safeguard_found"],
            "tube_T": aux["tube_T"], "u_backup": aux["u_backup"], "min_robust_safe_h": aux["min_robust_safe_h"],
            "terminal_robust_h_b": aux["terminal_robust_h_b"]}
    return u.astype(u_ref.dtype), aux["slack"].astype(u_ref.dtype), aux["used_solver"], info


def backup_policy_batch(xd_batch, cfg, runtime=None):
    rt = get_cached_runtime(cfg) if runtime is None else runtime
    xd_batch = jnp.asarray(xd_batch)
    return jax.vmap(rt.pi_b)(xd_batch[..., :10].astype(jnp.float32), xd_batch[..., 10:13].astype(jnp.float32))


__all__ = [
    "ACTOR_FILE",
    "BACKUPS",
    "ConeObstacleUEBCBFConfig",
    "backup_policy_batch",
    "build_ue_rows",
    "cone_ue_config",
    "cone_ue_config_from_task",
    "get_cached_runtime",
    "make_recoverability_fn_ue",
    "make_ue_runtime",
    "project_full",
    "rollout_ue",
    "safeguard",
    "solve_backup_cbf_qp_batch_with_info",
    "solve_qp_from_rows",
]
