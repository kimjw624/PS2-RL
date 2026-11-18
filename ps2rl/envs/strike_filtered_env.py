"""Strike environment with the safety filter in the loop (Phase II PS2 for the strike task): the residual striker's
action du is added to the pursuit controller, the sum goes through the strike filter, and the plant gets the filtered
command. The filter is part of the environment the striker learns in, so the striker learns the routes the filter
lets through (and pays the time cost of getting stuck in front of an obstacle) while every episode stays inside the
filter's recoverable set.

    u_nom = clip(u_pursuit(x) + du),   u = filter(x, u_nom),   x+ = f(x, u)
    reward = base reward (without its effort term) - w_action mean((du / scale)^2) - w_filter |(u - u_nom) / scale|^2

The filter's committed plan is carried in the state and reset with the episode.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from ps2rl.envs.strike_env import StrikeEnv, StrikeEnvConfig, StrikeEnvState, build_strike_env, residual_bound
from ps2rl.envs.strike_pursuit import make_pursuit
from ps2rl.envs.strike_scene import StrikeModel

Array = jax.Array


class FilteredState(NamedTuple):
    base: StrikeEnvState
    comm: Array            # the filter's committed plan


def build_filtered_strike_env(m: StrikeModel, ec: StrikeEnvConfig, filt: Callable, *, pool: dict,
                              w_filter: float = 0.0) -> StrikeEnv:
    """``filt``: a single-vehicle strike filter (strike_filter.make_filter / strike_bcbf.make_strike_bcbf). ``ec``
    must describe a residual striker (residual_cruise > 0)."""
    if ec.residual_cruise <= 0.0:
        raise ValueError("the filtered strike environment is for a residual striker (env residual_cruise > 0)")
    base = build_strike_env(m, ec.replace(residual_cruise=0.0, w_action=0.0), pool=pool)
    pursuit = make_pursuit(m, ec.residual_cruise)
    bound = residual_bound(m, ec)

    def fresh(st):
        return filt.fresh_view(st.x, st.obst, st.chaser_p, st.chaser_on)

    def reset(key):
        st, obs = base.reset(key)
        return FilteredState(st, fresh(st)), obs

    def step(fs: FilteredState, du, key):
        st = fs.base
        du = jnp.clip(jnp.asarray(du, jnp.float32), -bound, bound)
        u_nom = jnp.clip(pursuit(st.x, st.target_p, st.target_v) + du, m.low, m.high)
        u, info, comm = filt(st.x, u_nom, st.obst, st.chaser_p, st.chaser_on, fs.comm)
        st2, o_true, o_out, rew, done, sinfo = base.step(st, u, key)
        rew = rew - ec.w_action * jnp.mean((du / m.scale) ** 2) - w_filter * info.correction ** 2
        comm2 = jnp.where(done, fresh(st2), comm)          # st2 is the next episode's start when done
        return FilteredState(st2, comm2), o_true, o_out, rew, done, sinfo

    return StrikeEnv(obs_dim=base.obs_dim, action_dim=4, max_steps=base.max_steps,
                     reset_batched=jax.jit(jax.vmap(reset)), step_batched=jax.jit(jax.vmap(step)),
                     reset=jax.jit(reset), step=jax.jit(step), obs_fn=base.obs_fn, pool=base.pool)


__all__ = ["FilteredState", "build_filtered_strike_env"]
