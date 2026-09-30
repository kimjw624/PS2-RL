"""Run a trained Phase-II landing policy with and without the landing BCBF filter.

Both variants go through the Phase-II trainer's own deployment path
(``build_ps2_action_fns_for``): the actor's deterministic action, then - for the filtered
variant - the landing CIL projection (``LANDING_BINDING``), with the trainer's backup
fallback when the QP returns a non-finite solution. The environment, observation (incl.
time features) and reference are built by ``build_quadrotor_env`` from the run's own
config, so the policy sees exactly what it saw in training.

Both variants start from the same initial states: episode 0 starts exactly on the
reference's first sample ("nominal"), the others are drawn by the run's own reset
distribution. Each start is also classified with the backup rollout (x0 in C_N or not):
the forward-invariance guarantee only covers episodes that start in C_N.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.cil import quadrotor_landing_backup_cbf as lbcbf
from ps2rl.envs.quadrotor_env import build_quadrotor_env
from ps2rl.evaluation.landing_phase2_run import Phase2Run, _dataclass_from_dict
from ps2rl.phase2_ps2.landing_ps2_binding import LANDING_BINDING
from ps2rl.phase2_ps2.ps2_trainer_core import SACConfig, build_ps2_action_fns_for
from ps2rl.utils.policy import ActorConfig

PHYS_DIM = LANDING_BINDING.phys_dim
VARIANTS = ("unfiltered", "filtered")


@dataclass
class FilterEvalSetup:
    run: Phase2Run
    cbf_cfg: lbcbf.QuadrotorLandingBCBFConfig
    projector: lbcbf.QuadrotorLandingBackupCBFProjector
    env_fns: Any
    actor_cfg: ActorConfig
    eval_fns: dict[str, Any]
    max_steps: int
    dt: float


def _check_action_box(run: Phase2Run, cbf_cfg) -> None:
    env = run.env_cfg
    env_low = [env.a_cmd_min, -env.omega_max, -env.omega_max, -env.omega_max]
    env_high = [env.a_cmd_max, env.omega_max, env.omega_max, env.omega_max]
    if not (np.allclose(env_low, cbf_cfg.action_low) and np.allclose(env_high, cbf_cfg.action_high)):
        raise ValueError(
            f"action box differs: Phase-II env [{env_low}, {env_high}] vs backup [{cbf_cfg.action_low}, "
            f"{cbf_cfg.action_high}]. The actor maps tanh outputs onto the env box, so they must match."
        )


def build_setup(run: Phase2Run, backup_checkpoint: str, **qp_overrides: Any) -> FilterEvalSetup:
    cbf_cfg = lbcbf.landing_bcbf_config_from_checkpoint(backup_checkpoint, **qp_overrides)
    projector = lbcbf.QuadrotorLandingBackupCBFProjector(cbf_cfg)
    _check_action_box(run, cbf_cfg)
    if not math.isclose(float(run.env_cfg.dt), float(cbf_cfg.dt), rel_tol=1e-9):
        raise ValueError(f"env dt={run.env_cfg.dt} differs from the backup dt={cbf_cfg.dt}")

    env_fns = build_quadrotor_env(run.env_cfg)
    if run.actor_obs_dim > int(env_fns.obs_dim):
        raise ValueError(f"actor expects obs_dim={run.actor_obs_dim} > env obs_dim={env_fns.obs_dim}")
    if run.actor_head_dim != 2 * int(env_fns.action_dim):
        raise ValueError(f"actor head has {run.actor_head_dim} outputs, expected 2*action_dim={2 * env_fns.action_dim}")
    actor_cfg = ActorConfig(obs_dim=run.actor_obs_dim, action_dim=int(env_fns.action_dim),
                            hidden_sizes=run.actor_hidden_sizes)

    sac_cfg = _dataclass_from_dict(SACConfig, run.sac_dict)
    action_scale = jnp.asarray(cbf_cfg.action_scale, dtype=jnp.float32)
    variants = {
        "unfiltered": replace(sac_cfg, use_projection=False, project_actor_actions=False),
        "filtered": replace(sac_cfg, use_projection=True, project_actor_actions=True),
    }
    eval_fns = {}
    for name, cfg in variants.items():
        _, eval_action = build_ps2_action_fns_for(
            LANDING_BINDING, cfg, actor_cfg, cbf_cfg, action_scale, projector.runtime, return_solver_info=True
        )
        eval_fns[name] = eval_action
    return FilterEvalSetup(run, cbf_cfg, projector, env_fns, actor_cfg, eval_fns,
                           int(run.env_cfg.max_steps), float(run.env_cfg.dt))


def initial_states(setup: FilterEvalSetup, num_episodes: int, seed: int, *, nominal_first: bool = True):
    """Reset keys, env states and observations; episode 0 optionally starts on the reference."""
    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(num_episodes, dtype=jnp.uint32) + jnp.uint32(seed))
    state, obs = setup.env_fns.reset_batched(keys)
    if nominal_first:
        ref0 = jnp.asarray(np.load(setup.run.reference_path)["states"][0], dtype=state.x.dtype)
        x = state.x.at[0].set(ref0)
        state = state._replace(x=x)
        obs = obs.at[0, :PHYS_DIM].set(ref0)  # obs[:phys_dim] is the physical state (trainer convention)
    return keys, state, obs


def make_rollout(setup: FilterEvalSetup, variant: str):
    eval_action = setup.eval_fns[variant]
    env_fns = setup.env_fns
    obs_dim = setup.actor_cfg.obs_dim
    cfg = setup.cbf_cfg
    half = 0.5 * (jnp.asarray(cfg.action_high) - jnp.asarray(cfg.action_low))
    residual_fn = jax.vmap(setup.projector.residuals)
    batched_action = jax.vmap(eval_action, in_axes=(None, 0))
    filtered = variant == "filtered"

    @jax.jit
    def rollout(params, keys, state0, obs0):
        n = keys.shape[0]

        def body(carry, k):
            state, obs, active = carry
            safe, raw, slack, use_solver, finfo = batched_action(params, obs[:, :obs_dim])
            step_keys = jax.vmap(lambda kk: jax.random.fold_in(kk, k + 1))(keys)
            state_out, next_obs_true, obs_out, _, done, info = env_fns.step_batched(state, safe, step_keys)
            x_next = next_obs_true[:, :PHYS_DIM]
            if filtered:
                res = residual_fn(state.x, safe, slack)
                res_max = jnp.max(jnp.maximum(jnp.nan_to_num(res, nan=jnp.inf), 0.0), axis=-1)
                qp_ok = finfo["inputs_finite"] & finfo["z_finite"]
            else:
                res_max = jnp.zeros((n,), dtype=safe.dtype)
                qp_ok = jnp.ones((n,), dtype=jnp.bool_)
            rec = {
                "active": active,
                "x": state.x,
                "x_next": x_next,
                "u_raw": raw,
                "u": safe,
                "du_norm": jnp.linalg.norm((safe - raw) / half, axis=-1),
                "slack": slack,
                "qp_ok": qp_ok,
                "residual_max": res_max,
                "ref_state": info.ref_state,
                "pos_err": info.pos_error_norm,
                "t_next": info.ref_time_sec,
            }
            keep = active[:, None]
            state_next = jax.tree_util.tree_map(
                lambda a, b: jnp.where(active.reshape((-1,) + (1,) * (a.ndim - 1)), a, b), state_out, state
            )
            obs_next = jnp.where(keep, obs_out, obs)
            return (state_next, obs_next, active & ~done), rec

        _, recs = jax.lax.scan(body, (state0, obs0, jnp.ones((n,), dtype=jnp.bool_)),
                               jnp.arange(setup.max_steps, dtype=jnp.int32))
        return recs

    return rollout


def run_variant(setup: FilterEvalSetup, variant: str, keys, state0, obs0, batch_size: int) -> dict[str, np.ndarray]:
    rollout = make_rollout(setup, variant)
    params = setup.run.actor_params
    n = keys.shape[0]
    chunks = []
    for lo in range(0, n, batch_size):
        hi = min(n, lo + batch_size)
        sl = lambda a: a[lo:hi]  # noqa: E731
        rec = rollout(params, keys[lo:hi], jax.tree_util.tree_map(sl, state0), obs0[lo:hi])
        chunks.append(jax.tree_util.tree_map(lambda a: np.asarray(jax.device_get(a)), rec))
    # scan stacks time first: (T, B, ...) -> (B, T, ...)
    out = {k: np.concatenate([np.swapaxes(c[k], 0, 1) for c in chunks], axis=0) for k in chunks[0]}
    out["traj"] = np.concatenate([out["x"][:, :1], out["x_next"]], axis=1)  # (E, T+1, 10)
    return out


def classify_starts(setup: FilterEvalSetup, x0: np.ndarray) -> dict[str, np.ndarray]:
    rec_fn = lbcbf.make_recoverability_fn(setup.cbf_cfg, setup.projector.runtime)
    recoverable, crashed, arrival = rec_fn(jnp.asarray(x0))
    h0 = lbcbf.cone_value(jnp.asarray(x0), setup.cbf_cfg)
    return {"recoverable": np.asarray(recoverable), "backup_leaves_cone": np.asarray(crashed),
            "backup_arrival_step": np.asarray(arrival), "h0": np.asarray(h0)}


def cone_values(setup: FilterEvalSetup, traj: np.ndarray) -> np.ndarray:
    return np.asarray(lbcbf.cone_value(jnp.asarray(traj), setup.cbf_cfg))


def _tilt_deg(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip(1.0 - 2.0 * (q[..., 1] ** 2 + q[..., 2] ** 2), -1.0, 1.0)))


def _stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()),
            "p5": float(np.percentile(x, 5)), "median": float(np.median(x)), "p95": float(np.percentile(x, 95)),
            "max": float(x.max())}


def episode_metrics(setup: FilterEvalSetup, rec: dict[str, np.ndarray], h: np.ndarray) -> dict[str, np.ndarray]:
    """Per-episode metrics; ``h`` is h_cone on the (E, T+1) trajectory."""
    lc = setup.cbf_cfg.landing
    traj = rec["traj"]
    active = rec["active"].astype(bool)
    act_next = np.concatenate([np.ones_like(active[:, :1]), active], axis=1)  # state k valid if step k-1 active
    h_m = np.where(act_next, h, np.inf)
    viol = h_m < 0.0
    first = np.where(viol.any(axis=1), viol.argmax(axis=1), -1)
    rel = traj[..., 0:3] - np.array([lc.pad_x, lc.pad_y, lc.pad_z])
    last = act_next.sum(axis=1) - 1
    idx = np.arange(traj.shape[0])
    a_m = active.astype(np.float64)
    n_act = np.maximum(a_m.sum(axis=1), 1.0)
    thrust_sat = (np.isclose(rec["u"][..., 0], setup.cbf_cfg.action_high[0], atol=1e-4)
                  | np.isclose(rec["u"][..., 0], setup.cbf_cfg.action_low[0], atol=1e-4))
    return {
        "h_min": np.min(h_m, axis=1),
        "violated": viol.any(axis=1),
        "violation_steps": viol.sum(axis=1),
        "valid_states": act_next.sum(axis=1),
        "first_exit_time": np.where(first >= 0, first * setup.dt, np.nan),
        "final_pad_dist_xy": np.linalg.norm(rel[idx, last, 0:2], axis=-1),
        "final_height": rel[idx, last, 2],
        "min_height": np.min(np.where(act_next, rel[..., 2], np.inf), axis=1),
        "pos_err_mean": (rec["pos_err"] * a_m).sum(axis=1) / n_act,
        "pos_err_final": rec["pos_err"][idx, last - 1] if traj.shape[1] > 1 else rec["pos_err"][:, 0],
        "tilt_max_deg": np.max(np.where(act_next, _tilt_deg(traj[..., 6:10]), -np.inf), axis=1),
        "intervention_rate": ((rec["du_norm"] > 1e-3) * a_m).sum(axis=1) / n_act,
        "du_norm_mean": (rec["du_norm"] * a_m).sum(axis=1) / n_act,
        "slack_max": np.max(np.where(active, rec["slack"], 0.0), axis=1),
        "qp_fallback_steps": ((~rec["qp_ok"].astype(bool)) & active).sum(axis=1),
        "residual_max": np.max(np.where(active, rec["residual_max"], 0.0), axis=1),
        "thrust_saturation_rate": (thrust_sat * a_m).sum(axis=1) / n_act,
    }


def summarize(ep: dict[str, np.ndarray], mask: np.ndarray, r0: float) -> dict[str, Any]:
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return {"episodes": 0}
    viol = ep["violated"][m]
    exit_t = ep["first_exit_time"][m]
    out = {
        "episodes": int(m.sum()),
        "cone_violation_episode_rate": float(viol.mean()),
        "cone_violation_step_rate": float(ep["violation_steps"][m].sum() / max(1, ep["valid_states"][m].sum())),
        "h_min": _stats(ep["h_min"][m]),
        "first_exit_time": _stats(exit_t[np.isfinite(exit_t)]),
        "final_pad_dist_xy": _stats(ep["final_pad_dist_xy"][m]),
        "final_height": _stats(ep["final_height"][m]),
        "ends_over_pad_rate": float((ep["final_pad_dist_xy"][m] <= r0).mean()),
        "below_pad_rate": float((ep["min_height"][m] < 0.0).mean()),
        "min_height": _stats(ep["min_height"][m]),
        "pos_err_mean": _stats(ep["pos_err_mean"][m]),
        "tilt_max_deg": _stats(ep["tilt_max_deg"][m]),
        "intervention_rate": _stats(ep["intervention_rate"][m]),
        "du_norm_mean": _stats(ep["du_norm_mean"][m]),
        "slack_max": float(ep["slack_max"][m].max()),
        "qp_fallback_steps": int(ep["qp_fallback_steps"][m].sum()),
        "residual_max": float(ep["residual_max"][m].max()),
        "thrust_saturation_rate": _stats(ep["thrust_saturation_rate"][m]),
    }
    return out


__all__ = [
    "FilterEvalSetup",
    "VARIANTS",
    "build_setup",
    "classify_starts",
    "cone_values",
    "episode_metrics",
    "initial_states",
    "run_variant",
    "summarize",
]
