"""Phase-I safe-arrival environment for the cone-obstacle task (outside of an apex-down funnel).

Same construction as ``quadrotor_landing_ue_sa_env`` / ``runway_ue_sa_env`` (episodes are backup
rollouts started at tau = 0, exactly like the filter's; critic input (x, d_hat, tau/T, s)), with
the cone-obstacle sets:

    fail  <=>  h_cone < m_cone(s)  or  h_floor < m_floor(s)
    goal  <=>  V(x) <= c_B - m_base(s)          (the hover ellipsoid beside the funnel; first hit)

``tighten=False`` drops the margins (plain BCBF indicators: the nominal Phase I).

Disturbance: the user's Phase I is trained *without* a disturbance, i.e. with
``ue.d_hat_max = 0`` the frozen estimate is always 0 and the plant is the nominal one. The tube
margins still use the full disturbance bound (delta_d, delta_v, e_bar), so the trained backup
already keeps the buffers the Phase-II UE-bCBF filter will check.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

from ps2rl.backup_policy.backup_policy import BackupPolicy
from ps2rl.phase1_sa.cone_obstacle_design_region import ConeObstacleDesignRegionConfig, build_cone_obstacle_sampler
from ps2rl.phase1_sa.quadrotor_landing_sa_env import landing_action_box
from ps2rl.phase1_sa.quadrotor_landing_ue_sa_env import (
    ACTOR_OBS_DIM,
    CRITIC_OBS_DIM,
    IDX_S,
    OBS_X,
    UELandingSAEnvState,
    UELandingSAStepInfo,
    landing_ue_step_fn,
    sample_d_hat,
)
from ps2rl.sets.quadrotor_cone_obstacle_sets import obstacle_sets
from ps2rl.uncertainty.cone_obstacle_ue_tube import UEConfig, tube_constants, tube_margins, tube_step
from ps2rl.utils.policy import clip_to_box

Array = jax.Array


class ConeObstacleUESAEnvFns(NamedTuple):
    obs_dim: int
    actor_obs_dim: int
    action_dim: int
    reset: Callable
    step: Callable
    reset_batched: Callable
    step_batched: Callable
    safe_set: Any
    base_set: Any
    sampler: dict
    tube: Any
    plant_step: Callable
    goal_fn: Callable
    fail_fn: Callable
    tighten: bool


def build_cone_obstacle_indicators(safe_set, base_set, ue: UEConfig, tc, *, tighten: bool = True):
    """(not_failed(x, s), goal(x, s)) on the (optionally) tube-tightened sets."""
    w = 1.0 if tighten else 0.0

    def not_failed(x, s):
        m_c, m_f, _ = tube_margins(s, ue, tc)
        h = safe_set.component_values(x)
        return (h[0] >= w * m_c) & (h[1] >= w * m_f)

    def goal(x, s):
        _, _, m_b = tube_margins(s, ue, tc)
        return base_set.margin(x) >= w * m_b

    return not_failed, goal


def build_cone_obstacle_ue_sa_env(cfg, region_cfg: ConeObstacleDesignRegionConfig, ue: UEConfig, *,
                                  tighten: bool = True, episode_max_steps: int | None = None,
                                  terminate_on_goal: bool = True, dtype=jnp.float32) -> ConeObstacleUESAEnvFns:
    safe_set, base_set = obstacle_sets(cfg)
    if cfg.base_set != "points" or len(cfg.hover_points) != 1:
        raise ValueError("the cone-obstacle UE Phase I expects one 9-D hover ellipsoid (base_set='points', one point)")
    tc = tube_constants(cfg, base_set)
    sampler = build_cone_obstacle_sampler(region_cfg, cfg, safe_set, base_set, dtype=dtype)
    _, action_low, action_high = landing_action_box(cfg, dtype)
    plant = landing_ue_step_fn(cfg)
    n_backup = jnp.int32(cfg.num_steps)
    k_max = jnp.int32(int(episode_max_steps) if episode_max_steps is not None else int(cfg.num_steps))
    term_goal = jnp.asarray(bool(terminate_on_goal))
    inf = jnp.asarray(jnp.inf, dtype=dtype)
    t_h = float(cfg.T)
    s_clip = float(ue.s_clip)
    not_failed_fn, goal_x_fn = build_cone_obstacle_indicators(safe_set, base_set, ue, tc, tighten=tighten)

    def make_obs(x, d_hat, tau, s):
        return jnp.concatenate([x, d_hat, jnp.minimum(tau / t_h, 1.0)[None], jnp.minimum(s, s_clip)[None]]).astype(dtype)

    def goal_obs(o):
        return goal_x_fn(o[OBS_X], o[IDX_S])

    def fail_obs(o):
        return ~not_failed_fn(o[OBS_X], o[IDX_S])

    def env_reset(key, curriculum_scale):
        k_x, k_d = jax.random.split(key)
        x0 = sampler["sample"](k_x, curriculum_scale)
        d_hat = sample_d_hat(k_d, ue, dtype)
        z = jnp.asarray(0.0, dtype=dtype)
        st = UELandingSAEnvState(x=x0, d_hat=d_hat, tau=z, s=z, steps=jnp.int32(0), ep_min_h=inf)
        return st, make_obs(x0, d_hat, z, z)

    def env_step(state, raw_action, growth, key, curriculum_scale):
        raw_action = clip_to_box(jnp.asarray(raw_action, dtype=dtype), action_low, action_high)
        applied = clip_to_box(BackupPolicy.select_action(state.x, raw_action, base_set), action_low, action_high)
        x_next = plant(state.x, applied, state.d_hat)
        s_next = jnp.minimum(tube_step(state.s, growth, state.tau, ue, tc, cfg.dt), s_clip)
        tau_next = state.tau + jnp.asarray(cfg.dt, dtype=dtype)
        h_next = safe_set.value(x_next)
        safe = not_failed_fn(x_next, s_next)
        goal = goal_x_fn(x_next, s_next)
        fail = ~safe
        steps = state.steps + jnp.int32(1)
        timeout = steps >= k_max
        done = fail | (term_goal & goal) | timeout
        ep_min_h = jnp.minimum(state.ep_min_h, h_next)
        zero = jnp.asarray(0.0, dtype=dtype)
        success = done & goal & safe
        info = UELandingSAStepInfo(
            goal_next=goal.astype(dtype), fail_next=fail.astype(dtype), raw_action=raw_action, applied_action=applied,
            episode_done=done, completed_len=jnp.where(done, steps.astype(dtype), zero),
            completed_success=success.astype(dtype), completed_crash=(done & fail).astype(dtype),
            completed_timeout=(done & timeout & ~goal & ~fail).astype(dtype),
            completed_success_within_n=(success & (steps <= n_backup)).astype(dtype),
            completed_entry_step=jnp.where(success, steps, jnp.int32(-1)),
            completed_min_h=jnp.where(done, ep_min_h, zero), completed_s=jnp.where(done, s_next, zero),
        )
        obs_true = make_obs(x_next, state.d_hat, tau_next, s_next)
        rst, robs = env_reset(key, curriculum_scale)
        nxt = UELandingSAEnvState(x=x_next, d_hat=state.d_hat, tau=tau_next, s=s_next, steps=steps, ep_min_h=ep_min_h)
        return (jax.tree_util.tree_map(lambda r, c: jnp.where(done, r, c), rst, nxt), obs_true,
                jnp.where(done, robs, obs_true), done, info)

    return ConeObstacleUESAEnvFns(
        obs_dim=CRITIC_OBS_DIM, actor_obs_dim=ACTOR_OBS_DIM, action_dim=4, reset=jax.jit(env_reset), step=jax.jit(env_step),
        reset_batched=jax.jit(jax.vmap(env_reset, in_axes=(0, None))),
        step_batched=jax.jit(jax.vmap(env_step, in_axes=(0, 0, 0, 0, None))),
        safe_set=safe_set, base_set=base_set, sampler=sampler, tube=tc, plant_step=plant, goal_fn=goal_obs,
        fail_fn=fail_obs, tighten=bool(tighten),
    )


__all__ = ["ConeObstacleUESAEnvFns", "build_cone_obstacle_indicators", "build_cone_obstacle_ue_sa_env"]
