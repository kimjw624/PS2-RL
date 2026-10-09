"""Phase-II environment: a quadrotor chasing a bird next to a runway, under wind (UE-bCBF).

Plant, disturbance and observer are those of the landing UE environment:
    x_dot = f(x) + g(x) u + E_d d(t),  d(t) = A n sin(2 pi f t + phi),  d_hat = lambda (v - xi).

Bird (per episode, 3 kinds): ``cross`` - flies towards and across the runway (+y), ``climb`` -
climbs through the ceiling, ``wander`` - any horizontal direction. Velocity = initial velocity
+ an Ornstein-Uhlenbeck acceleration (smooth random turns), speed clamped. The bird is not
constrained by S: following it is exactly what the safety layer must stop.

Observation (20-D): [x (10, p_x = 0: everything is invariant along the runway), d_hat (3),
p_bird - p_drone (3), v_bird (3), t / T_episode]. The first 13 entries are what the UE filter needs.

Reward = the chase only (safety is the filter's job, not a penalty):
    r = -w_d charb(max(|p_bird - p| - d_keep, 0) / s_d) - small attitude / rate / thrust terms,
floored at -floor. Termination: leaving S (runway keep-out or ceiling; optional) or the time limit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.envs.quadrotor_env import quadrotor_dynamics
from ps2rl.utils.quaternion import normalize_quaternion, quaternion_from_euler_zyx

Array = jax.Array
OBS_DIM = 20
SAFETY_OBS_DIM = 13
BIRD_KINDS = ("cross", "climb", "wander")


@dataclass(frozen=True)
class RunwayBirdEnvConfig:
    episode_sec: float = 5.0
    # drone start (filtered to the tightened C_N of the UE backup)
    init_y_min: float = -8.0
    init_y_max: float = -2.5
    init_z_min: float = 4.0
    init_z_max: float = 8.0
    init_v: float = 1.5
    init_tilt_deg: float = 10.0
    init_yaw_deg: float = 10.0
    bank_size: int = 4096
    bank_seed: int = 12345
    disturbance: bool = True
    observer_init_err_frac: float = 0.5
    require_recoverable: bool = True
    terminate_on_unsafe: bool = True
    # bird
    p_cross: float = 0.5
    p_climb: float = 0.25
    bird_speed_min: float = 2.0
    bird_speed_max: float = 4.0
    bird_ou_sigma: float = 1.5  # m/s^2
    bird_ou_tau: float = 1.0  # s
    bird_speed_clip: float = 6.0
    bird_z_min: float = 1.0
    bird_z_max: float = 16.0
    # reward
    d_keep: float = 1.0
    s_d: float = 1.0
    w_d: float = 1.0
    s_tilt: float = 0.5
    w_tilt: float = 0.05
    s_yaw: float = 0.5
    w_yaw: float = 0.1
    s_wxy: float = 4.0
    w_wxy: float = 0.01
    s_wz: float = 1.0
    w_wz: float = 0.1
    w_thrust: float = 0.01
    delta: float = 0.1
    floor: float = 20.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class RunwayBirdState(NamedTuple):
    x: Array
    xi: Array
    d_dir: Array
    d_phase: Array
    p_b: Array
    v_b: Array
    a_b: Array
    v_b0: Array
    key: Array
    steps: Array
    ep_return: Array
    ep_len: Array
    ep_dist: Array
    ep_min_h_rwy: Array
    ep_min_h_ceil: Array
    ep_unsafe: Array


class RunwayBirdInfo(NamedTuple):
    episode_done: Array
    safe: Array
    h_rwy: Array
    h_ceil: Array
    dist: Array
    d_true: Array
    d_hat: Array
    p_bird: Array
    p_true: Array
    completed_return: Array
    completed_len: Array
    completed_unsafe: Array
    completed_mean_dist: Array
    completed_min_h_rwy: Array
    completed_min_h_ceil: Array


class RunwayBirdEnvFns(NamedTuple):
    obs_dim: int
    action_dim: int
    max_steps: int
    reset: Callable
    step: Callable
    reset_batched: Callable
    step_batched: Callable
    residual: bool


def build_runway_bird_env(env_cfg: RunwayBirdEnvConfig, cbf_cfg: Any, rt: Any, recover_fn: Callable,
                          dtype=jnp.float32) -> RunwayBirdEnvFns:
    rc, ue = cbf_cfg.runway, cbf_cfg.ue
    dt, g = float(rc.dt), float(rc.gravity)
    max_steps = int(round(float(env_cfg.episode_sec) / dt))
    lam = float(ue.observer_lambda)
    amp = float(ue.delta_d) if env_cfg.disturbance else 0.0
    omega_d = 2.0 * np.pi * float(ue.frequency_hz)
    y_edge, z_max = float(rc.y_edge), float(rc.z_max)
    c = env_cfg

    def d_true(dir_, phase, t):
        return amp * jnp.sin(omega_d * t + phase) * dir_

    def charb(e):
        return jnp.sqrt(e * e + c.delta ** 2) - c.delta

    def observation(x, xi, p_b, v_b, k):
        dh = lam * (x[3:6] - xi)
        t = k.astype(dtype) * dt / float(c.episode_sec)
        return jnp.concatenate([x.at[0].set(0.0), dh, p_b - x[0:3], v_b, t[None]]).astype(dtype)

    def reward(x, u, p_b):
        dist = jnp.linalg.norm(p_b - x[0:3])
        q = normalize_quaternion(x[6:10])
        tilt = jnp.arccos(jnp.clip(1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2), -1.0, 1.0))
        yaw = jnp.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        cost = (c.w_d * charb(jnp.maximum(dist - c.d_keep, 0.0) / c.s_d) + c.w_tilt * charb(tilt / c.s_tilt)
                + c.w_yaw * charb(yaw / c.s_yaw) + c.w_wxy * ((u[1] / c.s_wxy) ** 2 + (u[2] / c.s_wxy) ** 2)
                + c.w_wz * (u[3] / c.s_wz) ** 2 + c.w_thrust * ((u[0] - g) / g) ** 2)
        return jnp.maximum(-cost, -c.floor), dist

    # ------------------------------------------------------------ initial-state bank (drone + wind)
    tilt0, yaw0 = np.deg2rad(c.init_tilt_deg), np.deg2rad(c.init_yaw_deg)

    def raw_sample(key):
        k = jax.random.split(key, 8)
        y = jax.random.uniform(k[0], (), dtype, c.init_y_min, c.init_y_max)
        z = jax.random.uniform(k[1], (), dtype, c.init_z_min, c.init_z_max)
        vd = jax.random.normal(k[2], (3,), dtype)
        v = vd / jnp.maximum(jnp.linalg.norm(vd), 1e-9) * c.init_v * jax.random.uniform(k[3], (), dtype) ** (1 / 3)
        rpy = jax.random.uniform(k[4], (3,), dtype, -1.0, 1.0) * jnp.asarray([tilt0, tilt0, yaw0], dtype)
        q = normalize_quaternion(quaternion_from_euler_zyx(rpy[0], rpy[1], rpy[2]))
        n = jax.random.normal(k[5], (3,), dtype)
        n = n / jnp.maximum(jnp.linalg.norm(n), 1e-9)
        phase = jax.random.uniform(k[6], (), dtype, 0.0, 2.0 * np.pi)
        e0 = jax.random.normal(k[7], (3,), dtype)
        e0 = e0 / jnp.maximum(jnp.linalg.norm(e0), 1e-9) * c.observer_init_err_frac * float(ue.e_bar) * 0.7
        x = jnp.concatenate([jnp.stack([jnp.zeros((), dtype), y, z]), v, q])
        return x, n, phase, d_true(n, phase, 0.0) + e0

    keys = jax.random.split(jax.random.PRNGKey(int(c.bank_seed)), int(c.bank_size))
    xs, ns, phs, dh0 = jax.jit(jax.vmap(raw_sample))(keys)
    ok = np.asarray(recover_fn(xs, dh0))
    if ok.sum() == 0:
        if c.require_recoverable:
            raise ValueError("no initial state is in the tightened C_N; shrink the init ranges")
        print("[runway bird env] WARNING: no recoverable initial state; using unfiltered starts (smoke test only)")
        ok = np.ones_like(ok)
    print(f"[runway bird env] initial-state bank: {int(ok.sum())}/{len(ok)} in the tightened C_N ({ok.mean():.1%}); "
          f"episode {max_steps} steps; wind A={amp} f={ue.frequency_hz} Hz; runway edge y={y_edge}, ceiling z={z_max}",
          flush=True)
    sel = np.nonzero(ok)[0]
    bank_x, bank_n = jnp.asarray(np.asarray(xs)[sel], dtype), jnp.asarray(np.asarray(ns)[sel], dtype)
    bank_ph, bank_dh = jnp.asarray(np.asarray(phs)[sel], dtype), jnp.asarray(np.asarray(dh0)[sel], dtype)
    nb = int(sel.size)

    def sample_bird(key, p_d):
        k = jax.random.split(key, 8)
        u = jax.random.uniform(k[0], (), dtype)
        kind = jnp.where(u < c.p_cross, 0, jnp.where(u < c.p_cross + c.p_climb, 1, 2))
        off = jnp.stack([jax.random.uniform(k[1], (), dtype, -2.0, 2.0), jax.random.uniform(k[2], (), dtype, 0.5, 3.0),
                         jax.random.uniform(k[3], (), dtype, -1.0, 1.0)])
        p_b = p_d + off
        p_b = p_b.at[1].set(jnp.minimum(p_b[1], y_edge - 0.5)).at[2].set(jnp.clip(p_b[2], 2.0, z_max - 1.0))
        speed = jax.random.uniform(k[4], (), dtype, c.bird_speed_min, c.bird_speed_max)
        ang_cross = jnp.pi / 2 + jax.random.uniform(k[5], (), dtype, -0.9, 0.9)  # around +y
        ang_any = jax.random.uniform(k[5], (), dtype, 0.0, 2.0 * jnp.pi)
        ang = jnp.where(kind == 0, ang_cross, ang_any)
        h_speed = jnp.where(kind == 1, 0.5 * speed, speed)
        vz = jnp.where(kind == 1, jax.random.uniform(k[6], (), dtype, 1.0, 2.0),
                       jax.random.uniform(k[6], (), dtype, -0.3, 0.3))
        v_b = jnp.stack([h_speed * jnp.cos(ang), h_speed * jnp.sin(ang), vz])
        return p_b, v_b, k[7]

    def env_reset(key):
        k_i, k_b = jax.random.split(key)
        i = jax.random.randint(k_i, (), 0, nb)
        x0 = bank_x[i]
        xi0 = x0[3:6] - bank_dh[i] / lam
        p_b, v_b, kb = sample_bird(k_b, x0[0:3])
        z = jnp.asarray(0.0, dtype)
        st = RunwayBirdState(x=x0, xi=xi0, d_dir=bank_n[i], d_phase=bank_ph[i], p_b=p_b, v_b=v_b, a_b=jnp.zeros(3, dtype),
                             v_b0=v_b, key=kb, steps=jnp.int32(0), ep_return=z, ep_len=jnp.int32(0), ep_dist=z,
                             ep_min_h_rwy=jnp.asarray(jnp.inf, dtype), ep_min_h_ceil=jnp.asarray(jnp.inf, dtype),
                             ep_unsafe=jnp.asarray(False))
        return st, observation(x0, xi0, p_b, v_b, jnp.int32(0))

    lo_u, hi_u = jnp.asarray(cbf_cfg.action_low, dtype), jnp.asarray(cbf_cfg.action_high, dtype)
    decay = float(np.exp(-dt / c.bird_ou_tau))
    ou_std = float(c.bird_ou_sigma * np.sqrt(1.0 - decay ** 2))

    def env_step(state: RunwayBirdState, u, key):
        u = jnp.clip(jnp.asarray(u, dtype), lo_u, hi_u)
        t = state.steps.astype(dtype) * dt
        d = d_true(state.d_dir, state.d_phase, t)
        dh = lam * (state.x[3:6] - state.xi)
        xdot = quadrotor_dynamics(state.x, u, g, rc.a_cmd_min, rc.a_cmd_max, rc.omega_max)
        x_next = state.x + dt * xdot.at[3:6].add(d)
        x_next = x_next.at[6:10].set(normalize_quaternion(x_next[6:10]))
        xi_next = state.xi + dt * (xdot[3:6] + dh)
        # bird: OU acceleration around the initial velocity, speed clamped, altitude kept in a band
        kb, kn = jax.random.split(state.key)
        a_b = decay * state.a_b + ou_std * jax.random.normal(kn, (3,), dtype) * jnp.asarray([1.0, 1.0, 0.4], dtype)
        v_b = state.v_b + dt * a_b - dt * (state.v_b - state.v_b0) / c.bird_ou_tau
        sp = jnp.linalg.norm(v_b)
        v_b = v_b * jnp.minimum(1.0, c.bird_speed_clip / jnp.maximum(sp, 1e-6))
        p_b = state.p_b + dt * v_b
        v_b = v_b.at[2].set(jnp.where((p_b[2] < c.bird_z_min) | (p_b[2] > c.bird_z_max), -v_b[2], v_b[2]))
        k1 = state.steps + 1
        h_rwy = y_edge - x_next[1]
        h_ceil = z_max - x_next[2]
        safe = (h_rwy >= 0.0) & (h_ceil >= 0.0)
        rew, dist = reward(x_next, u, p_b)
        rew = rew.astype(dtype)
        done = ((~safe) & bool(c.terminate_on_unsafe)) | (k1 >= max_steps)
        ep_return = state.ep_return + rew
        ep_len = state.ep_len + 1
        ep_dist = state.ep_dist + dist
        mh_r = jnp.minimum(state.ep_min_h_rwy, h_rwy)
        mh_c = jnp.minimum(state.ep_min_h_ceil, h_ceil)
        unsafe_ep = state.ep_unsafe | ~safe
        df = done.astype(dtype)
        info = RunwayBirdInfo(
            episode_done=done, safe=safe.astype(dtype), h_rwy=h_rwy, h_ceil=h_ceil, dist=dist, d_true=d, d_hat=dh,
            p_bird=p_b, p_true=x_next[0:3], completed_return=df * ep_return, completed_len=df * ep_len.astype(dtype),
            completed_unsafe=df * unsafe_ep.astype(dtype), completed_mean_dist=df * ep_dist / jnp.maximum(ep_len, 1),
            completed_min_h_rwy=df * mh_r, completed_min_h_ceil=df * mh_c)
        obs_true = observation(x_next, xi_next, p_b, v_b, k1)
        cont = RunwayBirdState(x=x_next, xi=xi_next, d_dir=state.d_dir, d_phase=state.d_phase, p_b=p_b, v_b=v_b, a_b=a_b,
                               v_b0=state.v_b0, key=kb, steps=k1, ep_return=ep_return, ep_len=ep_len, ep_dist=ep_dist,
                               ep_min_h_rwy=mh_r, ep_min_h_ceil=mh_c, ep_unsafe=unsafe_ep)
        rst, robs = env_reset(key)
        st_out = jax.tree_util.tree_map(lambda a, b: jnp.where(done, a, b), rst, cont)
        return st_out, obs_true, jnp.where(done, robs, obs_true), rew, done, info

    return RunwayBirdEnvFns(obs_dim=OBS_DIM, action_dim=4, max_steps=max_steps, reset=jax.jit(env_reset),
                            step=jax.jit(env_step), reset_batched=jax.jit(jax.vmap(env_reset)),
                            step_batched=jax.jit(jax.vmap(env_step)), residual=False)


__all__ = ["BIRD_KINDS", "OBS_DIM", "RunwayBirdEnvConfig", "RunwayBirdEnvFns", "RunwayBirdInfo", "RunwayBirdState",
           "SAFETY_OBS_DIM", "build_runway_bird_env"]
