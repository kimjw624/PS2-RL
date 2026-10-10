"""Phase-I safe-arrival environment for landing under a bounded disturbance (UE-bCBF version).

Differences from ``quadrotor_landing_sa_env`` (the nominal landing Phase I):

* dynamics: the frozen-estimate backup flow of UE-bCBF,
      x+ = Post(x + dt (f(x) + g(x) u + E_d d_hat)),
  with d_hat sampled once per episode (|d_hat| <= d_hat_max) and held fixed;
* the backup policy sees (x, d_hat): actor input = 13-D;
* safe-arrival indicators are tightened by the UE tube (``ps2rl.uncertainty.landing_ue_tube``):
      fail  <=>  h_cone < m_cone(s) or z < m_floor(s) [or the tightened gentle-recovery envelope fails]
      goal  <=>  V(x) <= c_B - m_base(s)
  where s is the P-metric tube accumulated along the episode (episodes are backup rollouts
  started at tau = 0, exactly as the filter's rollouts);
* the critic observation is the augmented state (x, d_hat, tau / T, s): the indicators
  depend on (tau, s), so the safe-arrival value is Markov only in the augmented state.

The tube growth ||F_k||_P of the *deterministic* composed backup at x is computed by the
collector (it needs the current actor) and passed to ``step``. The hand-off to the LQR uses
the untightened base set B, so the composed backup pi_b(x, d_hat) stays stationary.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.backup_policy.backup_policy import BackupPolicy
from ps2rl.envs.quadrotor_env import quadrotor_step_euler
from ps2rl.envs.quadrotor_landing_config import QuadrotorLandingConfig
from ps2rl.phase1_sa.landing_design_region import LandingDesignRegionConfig, build_landing_sampler
from ps2rl.phase1_sa.quadrotor_landing_sa_env import build_landing_sets, landing_action_box
from ps2rl.uncertainty.landing_ue_tube import LandingUEConfig, TubeConstants, tube_constants, tube_margins, tube_step
from ps2rl.utils.policy import clip_to_box
from ps2rl.utils.quaternion import normalize_quaternion_batch

Array = jax.Array

# augmented observation layout
OBS_X = slice(0, 10)
OBS_DHAT = slice(10, 13)
IDX_TAU = 13
IDX_S = 14
ACTOR_OBS_DIM = 13
CRITIC_OBS_DIM = 15


class UELandingSAEnvState(NamedTuple):
    x: Array
    d_hat: Array
    tau: Array
    s: Array
    steps: Array
    ep_min_h: Array


class UELandingSAStepInfo(NamedTuple):
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
    completed_s: Array


class UELandingSAEnvFns(NamedTuple):
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
    tube: TubeConstants
    plant_step: Callable  # (x, u, d_hat) -> x+
    goal_fn: Callable  # augmented obs -> bool
    fail_fn: Callable  # augmented obs -> bool


def landing_ue_step_fn(cfg: QuadrotorLandingConfig) -> Callable[[Array, Array, Array], Array]:
    """Frozen-estimate plant step: explicit Euler with E_d d_hat on v_dot, then |q| = 1."""

    def step(x: Array, u: Array, d_hat: Array) -> Array:
        xn = quadrotor_step_euler(x, u, cfg.dt, cfg.gravity, cfg.a_cmd_min, cfg.a_cmd_max, cfg.omega_max)
        xn = xn.at[3:6].add(cfg.dt * jnp.asarray(d_hat, dtype=xn.dtype))
        return xn.at[6:10].set(normalize_quaternion_batch(xn[6:10]))

    return step


def sample_d_hat(key: Array, ue: LandingUEConfig, dtype=jnp.float32) -> Array:
    """15 % zero, 35 % on the sphere |d_hat| = R, 50 % uniform in the ball."""
    k_m, k_d, k_r = jax.random.split(key, 3)
    u = jax.random.normal(k_d, (3,), dtype=dtype)
    u = u / jnp.maximum(jnp.linalg.norm(u), 1e-9)
    r_max = jnp.asarray(float(ue.d_hat_max), dtype=dtype)
    mode = jax.random.uniform(k_m, (), dtype=dtype)
    r_ball = r_max * jax.random.uniform(k_r, (), dtype=dtype) ** (1.0 / 3.0)
    r = jnp.where(mode < 0.15, 0.0, jnp.where(mode < 0.5, r_max, r_ball))
    return r * u


def build_ue_indicators(cone, base_set, ue: LandingUEConfig, tc: TubeConstants):
    """(safe_tight(x, s), goal_tight(x, s), envelope-aware 'not failed'(x, s)) for one state."""
    floor = bool(getattr(cone, "floor", False))
    k_c = float(getattr(cone, "recovery_rate_cone", 0.0))
    k_f = float(getattr(cone, "recovery_rate_floor", 0.0))

    def not_failed(x: Array, s: Array) -> Array:
        m_c, m_f, _ = tube_margins(s, ue, tc)
        h = cone.component_values(x)
        hc_t = h[0] - m_c
        ok = hc_t >= 0.0
        if floor:
            hf_t = h[1] - m_f
            ok &= hf_t >= 0.0
        # gentle-recovery envelope on the tightened barriers: d/dt h_j <= kappa_j (h_j - m_j)
        v = x[3:6]
        dx = x[0] - cone.pad_x
        dy = x[1] - cone.pad_y
        rr = jnp.sqrt(dx * dx + dy * dy + cone.eps**2)
        dh_c = -(dx / rr) * v[0] - (dy / rr) * v[1] + cone.tan_theta * v[2]
        if k_c > 0.0:
            ok &= k_c * hc_t - dh_c >= 0.0
        if floor and k_f > 0.0:
            ok &= k_f * hf_t - v[2] >= 0.0
        return ok

    def goal(x: Array, s: Array) -> Array:
        _, _, m_b = tube_margins(s, ue, tc)
        return base_set.margin(x) >= m_b

    return not_failed, goal


def build_landing_ue_sa_env(
    cfg: QuadrotorLandingConfig,
    region_cfg: LandingDesignRegionConfig,
    ue: LandingUEConfig,
    *,
    episode_max_steps: int | None = None,
    terminate_on_goal: bool = True,
    dtype=jnp.float32,
) -> UELandingSAEnvFns:
    cone, base_set = build_landing_sets(cfg)
    tc = tube_constants(cfg)
    sampler = build_landing_sampler(region_cfg, cone, base_set, dtype=dtype)
    _, action_low, action_high = landing_action_box(cfg, dtype)
    plant = landing_ue_step_fn(cfg)
    n_backup = jnp.int32(cfg.num_steps)
    k_max = jnp.int32(int(episode_max_steps) if episode_max_steps is not None else int(cfg.num_steps))
    terminate_on_goal_j = jnp.asarray(bool(terminate_on_goal))
    inf = jnp.asarray(jnp.inf, dtype=dtype)
    t_horizon = float(cfg.T)
    s_clip = float(ue.s_clip)
    not_failed_fn, goal_x_fn = build_ue_indicators(cone, base_set, ue, tc)

    def make_obs(x, d_hat, tau, s):
        return jnp.concatenate(
            [x, d_hat, jnp.minimum(tau / t_horizon, 1.0)[None], jnp.minimum(s, s_clip)[None]]
        ).astype(dtype)

    def goal_obs(o: Array) -> Array:
        return goal_x_fn(o[OBS_X], o[IDX_S])

    def fail_obs(o: Array) -> Array:
        return ~not_failed_fn(o[OBS_X], o[IDX_S])

    def env_reset(key: Array, curriculum_scale: Array):
        k_x, k_d = jax.random.split(key)
        x0 = sampler["sample"](k_x, curriculum_scale)
        d_hat = sample_d_hat(k_d, ue, dtype)
        z = jnp.asarray(0.0, dtype=dtype)
        state = UELandingSAEnvState(x=x0, d_hat=d_hat, tau=z, s=z, steps=jnp.int32(0), ep_min_h=inf)
        return state, make_obs(x0, d_hat, z, z)

    def env_step(state: UELandingSAEnvState, raw_action: Array, growth: Array, key: Array, curriculum_scale: Array):
        raw_action = clip_to_box(jnp.asarray(raw_action, dtype=dtype), action_low, action_high)
        applied = clip_to_box(BackupPolicy.select_action(state.x, raw_action, base_set), action_low, action_high)
        x_next = plant(state.x, applied, state.d_hat)
        s_next = jnp.minimum(tube_step(state.s, growth, state.tau, ue, tc, cfg.dt), s_clip)
        tau_next = state.tau + jnp.asarray(cfg.dt, dtype=dtype)

        h_next = cone.value(x_next)
        safe = not_failed_fn(x_next, s_next)
        goal = goal_x_fn(x_next, s_next)
        fail = ~safe
        steps = state.steps + jnp.int32(1)
        timeout = steps >= k_max
        done = fail | (terminate_on_goal_j & goal) | timeout
        ep_min_h = jnp.minimum(state.ep_min_h, h_next)

        zero = jnp.asarray(0.0, dtype=dtype)
        success = done & goal & safe
        info = UELandingSAStepInfo(
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
            completed_s=jnp.where(done, s_next, zero),
        )
        obs_true = make_obs(x_next, state.d_hat, tau_next, s_next)

        reset_state, reset_obs = env_reset(key, curriculum_scale)
        next_obs_out = jnp.where(done, reset_obs, obs_true)
        state_out = jax.tree_util.tree_map(
            lambda r, c: jnp.where(done, r, c),
            reset_state,
            UELandingSAEnvState(x=x_next, d_hat=state.d_hat, tau=tau_next, s=s_next, steps=steps, ep_min_h=ep_min_h),
        )
        return state_out, obs_true, next_obs_out, done, info

    return UELandingSAEnvFns(
        obs_dim=CRITIC_OBS_DIM,
        actor_obs_dim=ACTOR_OBS_DIM,
        action_dim=4,
        reset=jax.jit(env_reset),
        step=jax.jit(env_step),
        reset_batched=jax.jit(jax.vmap(env_reset, in_axes=(0, None))),
        step_batched=jax.jit(jax.vmap(env_step, in_axes=(0, 0, 0, 0, None))),
        safe_set=cone,
        base_set=base_set,
        sampler=sampler,
        tube=tc,
        plant_step=plant,
        goal_fn=goal_obs,
        fail_fn=fail_obs,
    )


__all__ = [
    "ACTOR_OBS_DIM",
    "CRITIC_OBS_DIM",
    "IDX_S",
    "IDX_TAU",
    "OBS_DHAT",
    "OBS_X",
    "UELandingSAEnvFns",
    "UELandingSAEnvState",
    "UELandingSAStepInfo",
    "build_landing_ue_sa_env",
    "build_ue_indicators",
    "landing_ue_step_fn",
    "sample_d_hat",
]
