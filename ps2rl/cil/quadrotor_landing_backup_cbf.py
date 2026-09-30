"""Backup-CBF (control-invariant layer) for the approach-cone landing task.

Drop-in counterpart of ``ps2rl.cil.quadrotor_backup_cbf`` for Phase II:

    powerloop                                   landing
    ------------------------------------------  ---------------------------------------------------
    QuadrotorBCBFConfig(...)                    landing_bcbf_config_from_checkpoint(ckpt, ...)
    QuadrotorBackupCBFProjector(cfg)            QuadrotorLandingBackupCBFProjector(cfg)
    solve_backup_cbf_qp_batch_with_info(...)    solve_backup_cbf_qp_batch_with_info(...)  (this module)
    ceiling row  z_max - z                      cone row  h_cone(x) = r0 + tan(theta) zeta - sqrt(|dp|^2 + eps^2)
    7-D hover-LQR base set                      9-D hover-LQR base set (horizontal position included)

The QP itself is the shared engine in ``ps2rl.cil.backup_cbf``; only the system bundle
(safe-set rows, base-set rows, backup policy, dynamics) is landing-specific.

Single source of truth: the Phase-I safe-arrival checkpoint. Its metadata carries the
full ``QuadrotorLandingConfig`` it was trained and certified with (cone, pad, z_des,
c_B, LQR weights, dt, action box). ``landing_bcbf_config_from_checkpoint`` builds the
Phase-II config from it, and ``make_landing_backup_runtime`` refuses to run if the
config and the checkpoint disagree on anything the certificate depends on. The QP
tuning (class-K gains, slack weight, solver tolerance, row conditioning) defaults to
the powerloop Phase-II values, read from ``QuadrotorBCBFConfig`` rather than copied.

State and action contract (identical to the rest of the repo):
    x = (p_x, p_y, p_z, v_x, v_y, v_z, q_w, q_x, q_y, q_z)   world frame, q body-to-world
    u = (a, omega_x, omega_y, omega_z)                         mass-normalised thrust [m/s^2],
                                                               body rates [rad/s]
Positions are in the frame of the checkpoint's pad (``pad_x/pad_y/pad_z``). If the
Phase-II environment puts the pad elsewhere, translate positions into that frame before
calling anything here (see ``to_pad_frame``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from math import isclose, isfinite
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.backup_policy.quadrotor_learned_backup import QuadrotorLBP, load_learned_quadrotor_backup_policy
from ps2rl.base_controller.quadrotor_landing_dlqr import QuadrotorLandingDLQR
from ps2rl.cil.backup_cbf import (
    BackupCBFConfig,
    BackupCBFProjector,
    BCBFSystem,
    make_backup_cbf_facades,
)
from ps2rl.cil.quadrotor_backup_cbf import QuadrotorBCBFConfig
from ps2rl.envs.quadrotor_env import (
    quadrotor_control_affine_terms,
    quadrotor_dynamics,
    quadrotor_step_euler,
)
from ps2rl.envs.quadrotor_landing_config import QuadrotorLandingConfig
from ps2rl.sets.base_sets import EllipsoidBaseSet
from ps2rl.sets.quadrotor_cone_sets import QuadrotorConeSafeSet
from ps2rl.utils.paths import load_pickle_payload
from ps2rl.utils.quaternion import normalize_quaternion

BACKUP_MODES = ("learned", "base_controller")

# Landing fields the certificate (and the trained backup) depend on. A mismatch between
# the runtime config and the checkpoint on any of these invalidates the guarantee.
CERTIFIED_FIELDS = tuple(f.name for f in fields(QuadrotorLandingConfig) if f.name != "base_set_smooth_gain")


def _qp_default(name: str) -> Any:
    """QP tuning defaults are the powerloop Phase-II values (not duplicated here)."""
    return getattr(QuadrotorBCBFConfig, name)


@dataclass(frozen=True)
class QuadrotorLandingBCBFConfig(BackupCBFConfig):
    """Phase-II backup-CBF configuration for the landing task.

    Horizon (``T``, ``dt``, ``num_steps``) and ``base_set_c`` are derived from ``landing``
    in ``__post_init__``; pass QP tuning fields to override the powerloop defaults.
    """

    backup_policy_mode: str = "learned"
    learned_backup_policy_path: str = ""
    landing: QuadrotorLandingConfig = field(default_factory=QuadrotorLandingConfig)

    # QP tuning (defaults taken from the powerloop Phase-II config)
    alpha: float = _qp_default("alpha")
    base_alpha: float = _qp_default("base_alpha")
    slack_weight: float = _qp_default("slack_weight")
    solver_tol: float = _qp_default("solver_tol")
    use_analytic_jacobian: bool = False
    # "float64" solves only the small QP in double precision (rows stay in the state
    # dtype). Requires jax_enable_x64. In float32, qpax returns non-finite solutions on a
    # few percent of the landing QPs; the engine then applies the backup action, which is
    # safe but not minimally invasive and carries no gradient to the policy.
    qp_solve_dtype: str = "float32"

    def __post_init__(self) -> None:
        mode = str(self.backup_policy_mode).strip().lower()
        if mode not in BACKUP_MODES:
            raise ValueError(f"backup_policy_mode must be one of {BACKUP_MODES}, got '{self.backup_policy_mode}'")
        if mode == "learned" and not str(self.learned_backup_policy_path):
            raise ValueError("backup_policy_mode='learned' needs learned_backup_policy_path")
        if self.use_analytic_jacobian:
            raise ValueError("the landing system has no analytic closed-loop Jacobian; use_analytic_jacobian must be False")
        lc = self.landing
        if not isinstance(lc, QuadrotorLandingConfig):
            raise TypeError("landing must be a QuadrotorLandingConfig")
        for name, given, derived in (("T", self.T, lc.T), ("dt", self.dt, lc.dt)):
            if given is not None and not isclose(float(given), float(derived), rel_tol=1e-9):
                raise ValueError(f"{name}={given} disagrees with the landing config ({derived}); set it there instead")
        if self.qp_solve_dtype not in ("float32", "float64"):
            raise ValueError(f"qp_solve_dtype must be 'float32' or 'float64', got '{self.qp_solve_dtype}'")
        if self.qp_solve_dtype == "float64" and not jax.config.jax_enable_x64:
            raise ValueError("qp_solve_dtype='float64' needs jax_enable_x64 (set JAX_ENABLE_X64=1 before importing jax)")
        object.__setattr__(self, "backup_policy_mode", mode)
        object.__setattr__(self, "T", float(lc.T))
        object.__setattr__(self, "dt", float(lc.dt))
        object.__setattr__(self, "num_steps", int(lc.num_steps))
        object.__setattr__(self, "base_set_c", float(lc.base_set_c))
        for name in ("alpha", "base_alpha", "slack_weight", "solver_tol"):
            v = float(getattr(self, name))
            if not isfinite(v) or v <= 0.0:
                raise ValueError(f"{name} must be positive and finite, got {v}")

    # --- quantities the Phase-II trainer reads off the powerloop config ------------
    @property
    def gravity(self) -> float:
        return float(self.landing.gravity)

    @property
    def a_cmd_min(self) -> float:
        return float(self.landing.a_cmd_min)

    @property
    def a_cmd_max(self) -> float:
        return float(self.landing.a_cmd_max)

    @property
    def omega_max(self) -> float:
        return float(self.landing.omega_max)

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
        return 1

    @property
    def num_base_set_constraints(self) -> int:
        return 1

    @property
    def num_backup_inequalities(self) -> int:
        return self.num_safe_constraints * (self.num_steps + 1) + self.num_base_set_constraints

    @property
    def num_qp_inequalities(self) -> int:
        # + upper/lower bound per action + slack >= 0
        return self.num_backup_inequalities + 2 * len(self.action_low) + 1


# --------------------------------------------------------------------- checkpoint
def read_checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    """Metadata of a Phase-I landing actor checkpoint (``landing_backup_policy_actor.pkl``)."""
    p = Path(path)
    if p.is_dir():
        p = p / "landing_backup_policy_actor.pkl"
    payload = load_pickle_payload(p)
    md = dict(payload.get("metadata", {}))
    if "landing_config" not in md:
        raise KeyError(f"{p} has no metadata['landing_config']; is it a landing Phase-I checkpoint?")
    return md


def landing_config_from_checkpoint(path: str | Path) -> QuadrotorLandingConfig:
    return QuadrotorLandingConfig.from_dict(read_checkpoint_metadata(path)["landing_config"])


def landing_bcbf_config_from_checkpoint(path: str | Path, **overrides: Any) -> QuadrotorLandingBCBFConfig:
    """Phase-II BCBF config whose geometry, base set and backup all come from ``path``.

    ``overrides`` may set QP tuning fields (alpha, base_alpha, slack_weight, solver_tol,
    control_weight, ...) or ``backup_policy_mode='base_controller'`` for a baseline.
    Overriding ``landing`` is refused: re-run Phase I for a different geometry.
    """
    if "landing" in overrides:
        raise ValueError("the landing geometry comes from the checkpoint; re-run Phase I to change it")
    p = Path(path)
    ckpt = p / "landing_backup_policy_actor.pkl" if p.is_dir() else p
    return QuadrotorLandingBCBFConfig(
        backup_policy_mode=overrides.pop("backup_policy_mode", "learned"),
        learned_backup_policy_path=str(ckpt),
        landing=landing_config_from_checkpoint(ckpt),
        **overrides,
    )


def _check_against_checkpoint(cfg: QuadrotorLandingBCBFConfig, metadata: dict[str, Any]) -> None:
    trained = QuadrotorLandingConfig.from_dict(metadata["landing_config"]).as_dict()
    runtime = cfg.landing.as_dict()
    bad = [k for k in CERTIFIED_FIELDS if not np.isclose(float(trained[k]), float(runtime[k]), rtol=1e-9, atol=1e-12)]
    if bad:
        detail = ", ".join(f"{k}: checkpoint {trained[k]} vs config {runtime[k]}" for k in bad)
        raise ValueError(f"landing config does not match the safe-arrival checkpoint ({detail}). "
                         "The base-set certificate and the backup policy are only valid for the checkpoint's values.")


# ------------------------------------------------------------------------ runtime
def landing_sets(cfg: QuadrotorLandingBCBFConfig) -> tuple[QuadrotorConeSafeSet, EllipsoidBaseSet]:
    lc = cfg.landing
    cone = QuadrotorConeSafeSet.from_config(lc)
    ctrl = QuadrotorLandingDLQR.from_config(lc)
    return cone, EllipsoidBaseSet(ctrl, float(lc.base_set_c), smooth_gain=float(lc.base_set_smooth_gain))


def _sanitize_state(x: jax.Array) -> jax.Array:
    x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.at[6:10].set(normalize_quaternion(x[6:10]))


def make_landing_backup_runtime(cfg: QuadrotorLandingBCBFConfig) -> BCBFSystem:
    lc = cfg.landing
    cone, base_set = landing_sets(cfg)
    ctrl = base_set.controller
    if cfg.backup_policy_mode == "learned":
        learned = load_learned_quadrotor_backup_policy(cfg.learned_backup_policy_path)
        _check_against_checkpoint(cfg, learned.metadata)
        for name, got, want in (("action_low", learned.action_low, cfg.action_low), ("action_high", learned.action_high, cfg.action_high)):
            if not np.allclose(np.asarray(jax.device_get(got)), np.asarray(want), atol=1e-6):
                raise ValueError(f"checkpoint {name} {np.asarray(got).tolist()} != config {list(want)}")
        lbp = QuadrotorLBP(base_controller=ctrl, base_set=base_set, learned=learned,
                           a_cmd_min=cfg.a_cmd_min, a_cmd_max=cfg.a_cmd_max, omega_max=cfg.omega_max)
        backup_policy_fn = lbp.action
    else:  # base controller everywhere (baseline, no safe-arrival policy)
        backup_policy_fn = ctrl.action

    return BCBFSystem(
        state_dim=10,
        action_dim=4,
        action_low=cfg.action_low,
        action_high=cfg.action_high,
        backup_policy_fn=backup_policy_fn,
        safe_set_values_and_grads_fn=cone.values_and_grads,
        base_set_values_and_grads_fn=base_set.values_and_grads,
        dynamics_fn=lambda x, u: quadrotor_dynamics(x, u, lc.gravity, lc.a_cmd_min, lc.a_cmd_max, lc.omega_max),
        control_affine_terms_fn=lambda x: quadrotor_control_affine_terms(x, lc.gravity),
        postprocess_rollout_state_fn=lambda z: z.at[6:10].set(normalize_quaternion(z[6:10])),
        sanitize_solve_state_fn=_sanitize_state,
        solve_state_slice_fn=lambda xb: xb,
        base_set_values_fn=lambda x: base_set.values_and_grads(x)[0],
        use_analytic_jacobian=False,
        analytic_closed_loop_and_jacobian_fn=None,
        qp_solve_dtype=None if cfg.qp_solve_dtype == "float32" else jnp.float64,
    )


_RUNTIME_CACHE: dict[tuple, BCBFSystem] = {}


def _runtime_cache_key(cfg: QuadrotorLandingBCBFConfig) -> tuple:
    return (cfg.backup_policy_mode, cfg.learned_backup_policy_path, cfg.landing, cfg.qp_solve_dtype)


_facades = make_backup_cbf_facades(make_landing_backup_runtime, _runtime_cache_key, _RUNTIME_CACHE)
get_cached_runtime = _facades["get_cached_runtime"]
_resolve_runtime = _facades["_resolve_runtime"]
backup_policy = _facades["backup_policy"]
backup_policy_batch = _facades["backup_policy_batch"]
closed_loop_backup_dynamics = _facades["closed_loop_backup_dynamics"]
safe_set_values_and_grads = _facades["safe_set_values_and_grads"]
base_set_values_and_grads = _facades["base_set_values_and_grads"]
rollout_backup_flow_and_sensitivity = _facades["rollout_backup_flow_and_sensitivity"]
rollout_backup_flow_and_sensitivity_with_info = _facades["rollout_backup_flow_and_sensitivity_with_info"]
build_discretized_backup_cbf_rows = _facades["build_discretized_backup_cbf_rows"]
build_discretized_backup_cbf_rows_with_info = _facades["build_discretized_backup_cbf_rows_with_info"]
build_backup_cbf_qp = _facades["build_backup_cbf_qp"]
solve_backup_cbf_qp_single = _facades["solve_backup_cbf_qp_single"]
solve_backup_cbf_qp_single_with_info = _facades["solve_backup_cbf_qp_single_with_info"]
solve_backup_cbf_qp_batch = _facades["solve_backup_cbf_qp_batch"]
solve_backup_cbf_qp_batch_with_info = _facades["solve_backup_cbf_qp_batch_with_info"]
constraint_residuals = _facades["constraint_residuals"]


class QuadrotorLandingBackupCBFProjector(BackupCBFProjector):
    """Jitted single/batch projection u_ref -> u_safe for the landing task."""

    def __init__(self, cfg: QuadrotorLandingBCBFConfig, runtime: BCBFSystem | None = None):
        super().__init__(cfg, _resolve_runtime(cfg, runtime))


# ------------------------------------------------------------------ env helpers
def landing_step(x: jax.Array, u: jax.Array, cfg: QuadrotorLandingBCBFConfig) -> jax.Array:
    """The plant step Phase I was trained on (explicit Euler + quaternion normalisation)."""
    lc = cfg.landing
    return quadrotor_step_euler(x, u, lc.dt, lc.gravity, lc.a_cmd_min, lc.a_cmd_max, lc.omega_max)


def cone_value(x: jax.Array, cfg: QuadrotorLandingBCBFConfig) -> jax.Array:
    """h_cone(x) (>= 0 inside the cone); batched over leading axes. Use for termination/logging."""
    return QuadrotorConeSafeSet.from_config(cfg.landing).value(x)


def is_safe_state(x: jax.Array, cfg: QuadrotorLandingBCBFConfig) -> jax.Array:
    return cone_value(x, cfg) >= 0.0


def to_pad_frame(x: jax.Array, pad_world: jax.Array, cfg: QuadrotorLandingBCBFConfig) -> jax.Array:
    """Shift positions so the environment's pad ``pad_world`` lands on the checkpoint's pad."""
    lc = cfg.landing
    shift = jnp.asarray(pad_world) - jnp.asarray([lc.pad_x, lc.pad_y, lc.pad_z])
    return jnp.asarray(x).at[..., 0:3].add(-shift)


def make_recoverability_fn(cfg: QuadrotorLandingBCBFConfig, runtime: BCBFSystem | None = None):
    """Batched first-hit test x0 in C_N(pi_b): reaches B within N steps without leaving the cone.

    Returns a jitted ``fn(x_batch) -> (recoverable, crashed, arrival_step)``. Phase-II
    episodes should start from recoverable states; outside C_N the QP can need slack.
    """
    rt = _resolve_runtime(cfg, runtime)
    cone, base_set = landing_sets(cfg)
    n = int(cfg.num_steps)

    def one(x0):
        def body(c, k):
            x, hb, hf, tb = c
            xn = landing_step(x, rt.backup_policy_fn(x), cfg)
            act = ~(hb | hf)
            nf = act & ~cone.contains(xn)
            nb = act & ~nf & base_set.contains(xn)
            return (jnp.where(act, xn, x), hb | nb, hf | nf, jnp.where(nb, k + 1, tb)), None

        start_in_b = base_set.contains(x0)
        (_, hb, hf, tb), _ = jax.lax.scan(body, (x0, start_in_b, ~cone.contains(x0), jnp.where(start_in_b, 0, -1)), jnp.arange(n))
        return hb, hf, tb

    return jax.jit(jax.vmap(one))


__all__ = [
    "BACKUP_MODES",
    "CERTIFIED_FIELDS",
    "QuadrotorLandingBCBFConfig",
    "QuadrotorLandingBackupCBFProjector",
    "backup_policy",
    "backup_policy_batch",
    "build_backup_cbf_qp",
    "build_discretized_backup_cbf_rows",
    "cone_value",
    "constraint_residuals",
    "get_cached_runtime",
    "is_safe_state",
    "landing_bcbf_config_from_checkpoint",
    "landing_config_from_checkpoint",
    "landing_sets",
    "landing_step",
    "make_landing_backup_runtime",
    "make_recoverability_fn",
    "read_checkpoint_metadata",
    "rollout_backup_flow_and_sensitivity",
    "safe_set_values_and_grads",
    "solve_backup_cbf_qp_batch",
    "solve_backup_cbf_qp_batch_with_info",
    "solve_backup_cbf_qp_single",
    "solve_backup_cbf_qp_single_with_info",
    "to_pad_frame",
]
