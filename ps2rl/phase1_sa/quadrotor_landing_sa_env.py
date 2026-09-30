"""Phase-I safe-arrival environment for the approach-cone landing task.

Same transition contract as ``quadrotor_sa_env`` (so the shared ``sa_trainer_core`` loop
runs unchanged): each step returns the true successor ``next_obs_true`` (used for
replay and Bellman targets) and the collector observation ``next_obs_out`` (auto-reset
if the episode ended). Differences from the powerloop env:

* safe set = smooth approach cone (optionally n the pad plane), base set = 9-D hover
  ellipsoid above the pad; with the gentle-recovery envelope enabled, leaving it counts
  as a failure too (``QuadrotorLandingSafeSet.training_contains``)
* initial states come from the analytic design-region sampler (no trace library)
* collection episodes may run longer than the backup horizon N (``episode_max_steps``;
  the note recommends k_max > N). Timeouts are not terminal in the Bellman target
  (only goal/fail flags are), so long arrivals still bootstrap correctly.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

from ps2rl.backup_policy.backup_policy import BackupPolicy
from ps2rl.base_controller.quadrotor_landing_dlqr import QuadrotorLandingDLQR
from ps2rl.envs.quadrotor_env import quadrotor_step_euler
from ps2rl.envs.quadrotor_landing_config import QuadrotorLandingConfig
from ps2rl.phase1_sa.landing_design_region import LandingDesignRegionConfig, build_landing_sampler
from ps2rl.sets.base_sets import EllipsoidBaseSet
from ps2rl.sets.quadrotor_landing_safe_set import QuadrotorLandingSafeSet
from ps2rl.utils.policy import clip_to_box
from ps2rl.utils.quaternion import normalize_quaternion_batch

Array = jax.Array


class LandingSAEnvState(NamedTuple):
    x: Array
    steps: Array
    ep_min_h: Array


class LandingSAStepInfo(NamedTuple):
    goal_next: Array
    fail_next: Array
    raw_action: Array
    applied_action: Array
    episode_done: Array
    completed_len: Array
    completed_success: Array
    completed_crash: Array
    completed_timeout: Array
    completed_success_within_n: Array
    completed_entry_step: Array
    completed_min_h: Array


class LandingSAEnvFns(NamedTuple):
    obs_dim: int
    action_dim: int
    reset: Callable
    step: Callable
    reset_batched: Callable
    step_batched: Callable
    safe_set: QuadrotorLandingSafeSet
    base_set: EllipsoidBaseSet
    sampler: dict


def build_landing_sets(cfg: QuadrotorLandingConfig) -> tuple[QuadrotorLandingSafeSet, EllipsoidBaseSet]:
    cone = QuadrotorLandingSafeSet.from_config(cfg)
    ctrl = QuadrotorLandingDLQR.from_config(cfg)
    base_set = EllipsoidBaseSet(ctrl, float(cfg.base_set_c), smooth_gain=float(cfg.base_set_smooth_gain))
    return cone, base_set


def landing_action_box(cfg: QuadrotorLandingConfig, dtype=jnp.float32) -> tuple[Array, Array, Array]:
    low = jnp.asarray([cfg.a_cmd_min, -cfg.omega_max, -cfg.omega_max, -cfg.omega_max], dtype=dtype)
    high = jnp.asarray([cfg.a_cmd_max, cfg.omega_max, cfg.omega_max, cfg.omega_max], dtype=dtype)
    scale = jnp.asarray([cfg.a_cmd_max, cfg.omega_max, cfg.omega_max, cfg.omega_max], dtype=dtype)
    return scale, low, high


def landing_step_fn(cfg: QuadrotorLandingConfig) -> Callable[[Array, Array], Array]:
    def step(x: Array, u: Array) -> Array:
        xn = quadrotor_step_euler(x, u, cfg.dt, cfg.gravity, cfg.a_cmd_min, cfg.a_cmd_max, cfg.omega_max)
        return xn.at[6:10].set(normalize_quaternion_batch(xn[6:10]))

    return step


def build_landing_sa_env(
    cfg: QuadrotorLandingConfig,
    region_cfg: LandingDesignRegionConfig,
    *,
    episode_max_steps: int | None = None,
    terminate_on_goal: bool = True,
    dtype=jnp.float32,
) -> LandingSAEnvFns:
    cone, base_set = build_landing_sets(cfg)
    sampler = build_landing_sampler(region_cfg, cone, base_set, dtype=dtype)
    _, action_low, action_high = landing_action_box(cfg, dtype)
    step_plant = landing_step_fn(cfg)
    n_backup = jnp.int32(cfg.num_steps)
    k_max = jnp.int32(int(episode_max_steps) if episode_max_steps is not None else int(cfg.num_steps))
    terminate_on_goal_j = jnp.asarray(bool(terminate_on_goal))
    inf = jnp.asarray(jnp.inf, dtype=dtype)

    def env_reset(key: Array, curriculum_scale: Array):
        x0 = sampler["sample"](key, curriculum_scale)
        state = LandingSAEnvState(x=x0, steps=jnp.int32(0), ep_min_h=inf)
        return state, state.x

    def env_step(state: LandingSAEnvState, raw_action: Array, key: Array, curriculum_scale: Array):
        raw_action = clip_to_box(jnp.asarray(raw_action, dtype=dtype), action_low, action_high)
        applied = clip_to_box(BackupPolicy.select_action(state.x, raw_action, base_set), action_low, action_high)
        x_next = step_plant(state.x, applied)

        h_next = cone.value(x_next)
        safe = cone.training_contains(x_next)
        goal = base_set.contains(x_next)
        fail = ~safe
        steps = state.steps + jnp.int32(1)
        timeout = steps >= k_max
        done = fail | (terminate_on_goal_j & goal) | timeout
        ep_min_h = jnp.minimum(state.ep_min_h, h_next)

        zero = jnp.asarray(0.0, dtype=dtype)
        success = done & goal & safe
        info = LandingSAStepInfo(
            goal_next=goal.astype(dtype),
            fail_next=fail.astype(dtype),
            raw_action=raw_action,
            applied_action=applied,
            episode_done=done,
            completed_len=jnp.where(done, steps.astype(dtype), zero),
            completed_success=success.astype(dtype),
            completed_crash=(done & fail).astype(dtype),
            completed_timeout=(done & timeout & ~goal & ~fail).astype(dtype),
            completed_success_within_n=(success & (steps <= n_backup)).astype(dtype),
            completed_entry_step=jnp.where(success, steps, jnp.int32(-1)),
            completed_min_h=jnp.where(done, ep_min_h, zero),
        )

        reset_x = sampler["sample"](key, curriculum_scale)
        next_obs_out = jnp.where(done, reset_x, x_next)
        state_out = LandingSAEnvState(
            x=next_obs_out,
            steps=jnp.where(done, jnp.int32(0), steps),
            ep_min_h=jnp.where(done, inf, ep_min_h),
        )
        return state_out, x_next, next_obs_out, done, info

    reset = jax.jit(env_reset)
    step = jax.jit(env_step)
    return LandingSAEnvFns(
        obs_dim=10,
        action_dim=4,
        reset=reset,
        step=step,
        reset_batched=jax.jit(jax.vmap(env_reset, in_axes=(0, None))),
        step_batched=jax.jit(jax.vmap(env_step, in_axes=(0, 0, 0, None))),
        safe_set=cone,
        base_set=base_set,
        sampler=sampler,
    )


__all__ = [
    "LandingSAEnvFns",
    "LandingSAEnvState",
    "LandingSAStepInfo",
    "build_landing_sa_env",
    "build_landing_sets",
    "landing_action_box",
    "landing_step_fn",
]
