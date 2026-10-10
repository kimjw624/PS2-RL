"""Phase-II environment for the cone-obstacle task with a bounded disturbance and a disturbance observer.

The landing UE environment (``quadrotor_landing_ue_env``) with the cone-obstacle geometry:

Plant (true):      x_dot = f(x) + g(x) u + E_d d(t),
                   d(t) = A n sin(2 pi f t + phi),  n a random unit vector per episode,
                   A and f drawn per episode with A <= delta_d and 2 pi f A <= delta_v, so every
                   episode satisfies the UE-bCBF assumptions ||d|| <= delta_d, ||d_dot|| <= delta_v
                   (``disturbance_mode='random'``; 'fixed' = A = delta_d at the design frequency, as in
                   the landing UE environment);
Observer:          d_hat = lambda (v - xi),  xi_dot = a_nom(x, u) + d_hat, started warm
                   (|d_hat(0) - d(0)| <= e_bar / 2).

Safe set: outside the funnel (h_cone >= 0) and above the floor (z >= ground + min_altitude).
Task: track a reference that runs into the funnel and drops onto the target cylinder under the tip
(``scripts/cone_obstacle_reference_candidates.py --save``); the CIL keeps the vehicle out of the
funnel. Touching the cylinder (r <= R, floor <= z <= top) is recorded as success.

Observation (29-D): [x (10), d_hat (3), ref_state (10), ref_omega (3), t, sin, cos] (pure policy).

Reward: the landing tracking cost (Charbonnier on position, velocity, attitude, rates, thrust)
with its own weights (horizontal and vertical errors weighted alike by default).

Initial states: (x0, disturbance) pairs around the reference start, kept only if (x0, d_hat(0)) is
in the tightened C_N of the UE backup (Phase-II episodes start in the set the filter keeps invariant).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.envs.quadrotor_env import quadrotor_dynamics
from ps2rl.utils.quaternion import (
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_from_euler_zyx,
    quaternion_multiply,
)

Array = jax.Array
_ROOT = Path(__file__).resolve().parents[2]
OBS_DIM = 29
SAFETY_OBS_DIM = 13


@dataclass(frozen=True)
class ConeObstacleUEEnvConfig:
    reference_path: str = "ps2rl/envs/assets/cone_obstacle/quadrotor_cone_obstacle_l3_1_f_reference.npz"
    extra_sec: float = 2.0       # episode = reference duration + extra_sec (the filtered vehicle may lag behind)
    init_p_range: float = 0.2
    init_v_range: float = 0.3
    init_tilt_deg: float = 5.0
    init_yaw_deg: float = 5.0
    bank_size: int = 4096
    bank_seed: int = 12345
    disturbance: bool = True
    disturbance_mode: str = "random"        # "random": A, f drawn per episode within the bounds; "fixed": A = delta_d, f = design
    amp_min_frac: float = 0.0               # A ~ U[amp_min_frac, 1] * delta_d
    freq_min_hz: float = 0.02               # f ~ log-U[freq_min, min(freq_max, delta_v / (2 pi A))]
    freq_max_hz: float = 1.0
    observer_init_err_frac: float = 0.5     # |e0| <= frac * e_bar
    require_recoverable: bool = True
    terminate_on_unsafe: bool = True        # False for the vanilla (no-CIL) warm-start stage
    # tracking cost (sum w * charb(err / s)); horizontal and vertical alike by default
    s_z: float = 0.15
    w_z: float = 1.0
    s_vz: float = 0.5
    w_vz: float = 0.5
    s_xy: float = 0.15
    w_xy: float = 1.0
    s_vxy: float = 0.5
    w_vxy: float = 0.5
    s_tilt: float = 0.3
    w_tilt: float = 0.2
    s_yaw: float = 0.5
    w_yaw: float = 0.05
    s_wz: float = 0.5
    w_wz: float = 1.0
    s_wxy: float = 3.0
    w_wxy: float = 0.02
    w_thrust: float = 0.01
    w_touch: float = 0.0                    # optional one-time bonus for touching the cylinder (0 = tracking only)
    delta: float = 0.1
    floor: float = 50.0

    def __post_init__(self) -> None:
        if self.disturbance_mode not in ("random", "fixed"):
            raise ValueError(f"disturbance_mode must be 'random' or 'fixed', got {self.disturbance_mode!r}")
        if not (0.0 <= float(self.amp_min_frac) <= 1.0):
            raise ValueError("amp_min_frac must lie in [0, 1]")
        if not (0.0 < float(self.freq_min_hz) <= float(self.freq_max_hz)):
            raise ValueError("need 0 < freq_min_hz <= freq_max_hz")

    def as_dict(self):
        return asdict(self)


class ConeObstacleUEEnvState(NamedTuple):
    x: Array
    xi: Array
    d_dir: Array
    d_phase: Array
    d_amp: Array
    d_omega: Array
    steps: Array
    touched: Array
    ep_return: Array
    ep_len: Array
    ep_min_h: Array
    ep_min_z: Array
    ep_pos_err: Array
    ep_vel_err: Array
    ep_att_err: Array
    ep_safe_sum: Array


class ConeObstacleUEStepInfo(NamedTuple):
    episode_done: Array
    safe: Array
    h_cone: Array
    z: Array                 # height above the floor (the floor barrier)
    pos_err: Array
    touch: Array
    d_true: Array
    d_hat: Array
    completed_return: Array
    completed_len: Array
    completed_min_h: Array
    completed_min_z: Array
    completed_pos_err: Array
    completed_final_dist: Array   # distance to the reference's end point
    completed_final_z: Array
    completed_final_speed: Array
    completed_unsafe: Array
    completed_touched: Array


class ConeObstacleUEEnvFns(NamedTuple):
    obs_dim: int
    action_dim: int
    max_steps: int
    reset: Callable
    step: Callable
    reset_batched: Callable
    step_batched: Callable
    d_hat_of: Callable
    ref_states: Array
    residual: bool
    bank: dict


def sample_disturbance(key, ue, env_cfg: ConeObstacleUEEnvConfig, dtype=jnp.float32):
    """(n, phase, A, omega) for one episode, within ||d|| <= delta_d and ||d_dot|| <= delta_v."""
    k_n, k_p, k_a, k_f = jax.random.split(key, 4)
    n = jax.random.normal(k_n, (3,), dtype)
    n = n / jnp.maximum(jnp.linalg.norm(n), 1e-9)
    phase = jax.random.uniform(k_p, (), dtype, 0.0, 2.0 * np.pi)
    dd, dv = float(ue.delta_d), float(ue.delta_v)
    if not env_cfg.disturbance:
        return n, phase, jnp.asarray(0.0, dtype), jnp.asarray(2.0 * np.pi * float(ue.frequency_hz), dtype)
    if env_cfg.disturbance_mode == "fixed":
        return n, phase, jnp.asarray(dd, dtype), jnp.asarray(2.0 * np.pi * float(ue.frequency_hz), dtype)
    amp = dd * jax.random.uniform(k_a, (), dtype, float(env_cfg.amp_min_frac), 1.0)
    f_cap = jnp.minimum(float(env_cfg.freq_max_hz), dv / (2.0 * np.pi * jnp.maximum(amp, 1e-9)))
    f_lo = jnp.minimum(float(env_cfg.freq_min_hz), f_cap)
    f = jnp.exp(jax.random.uniform(k_f, (), dtype, 0.0, 1.0) * (jnp.log(f_cap) - jnp.log(f_lo)) + jnp.log(f_lo))
    return n, phase, amp, 2.0 * np.pi * f


def build_cone_obstacle_ue_env(env_cfg: ConeObstacleUEEnvConfig, cbf_cfg: Any, rt: Any, recover_fn: Callable,
                               dtype=jnp.float32) -> ConeObstacleUEEnvFns:
    sc = cbf_cfg.scenario
    ue = cbf_cfg.ue
    ref_path = Path(env_cfg.reference_path)
    if not ref_path.exists() and (_ROOT / ref_path).exists():
        ref_path = _ROOT / ref_path
    ref = np.load(ref_path)
    ref_states = jnp.asarray(ref["states"], dtype=dtype)
    ref_omega = jnp.asarray(ref["omega_cmd"], dtype=dtype)
    ref_t = np.asarray(ref["t"], dtype=np.float64)
    ref_dt = float(ref_t[1] - ref_t[0])
    if abs(ref_dt - float(sc.dt)) > 1e-9:
        raise ValueError(f"reference dt {ref_dt} != scenario dt {sc.dt}")
    n_ref = int(ref_states.shape[0])
    max_steps = int(round((float(ref_t[-1]) + float(env_cfg.extra_sec)) / float(sc.dt)))
    dt = float(sc.dt)
    g = float(sc.gravity)
    lam = float(ue.observer_lambda)
    t_total = float(ref_t[-1])
    end_p = jnp.asarray(np.asarray(ref["states"])[-1, 0:3], dtype)
    ax, ay = float(sc.axis_x), float(sc.axis_y)
    tx, ty = sc.target_center
    rad, top, floor_z = float(sc.target_radius), float(sc.cylinder_top), float(sc.floor_z)
    eps, tt, tip = float(sc.cone_eps), float(sc.tan_theta), float(sc.tip_z)

    def d_true(dir_, phase, amp, omega, t):
        return amp * jnp.sin(omega * t + phase) * dir_

    def h_cone(x):
        return jnp.sqrt((x[0] - ax) ** 2 + (x[1] - ay) ** 2 + eps * eps) - eps - tt * (x[2] - tip)

    def touches(x):
        r2 = (x[0] - tx) ** 2 + (x[1] - ty) ** 2
        return (r2 <= rad * rad) & (x[2] <= top) & (x[2] >= floor_z)

    def reference(k):
        i = jnp.clip(k, 0, n_ref - 1)
        return ref_states[i], ref_omega[i]

    def d_hat_of(state):
        return lam * (state.x[3:6] - state.xi)

    lo_u = jnp.asarray(cbf_cfg.action_low, dtype)
    hi_u = jnp.asarray(cbf_cfg.action_high, dtype)

    def observation(x, xi, k):
        rs, ro = reference(k)
        t = k.astype(dtype) * dt
        ph = 2.0 * jnp.pi * jnp.clip(t / t_total, 0.0, 1.0)
        dh = lam * (x[3:6] - xi)
        return jnp.concatenate([x, dh, rs, ro, jnp.stack([t, jnp.sin(ph), jnp.cos(ph)])]).astype(dtype)

    def charb(e):
        return jnp.sqrt(e * e + env_cfg.delta ** 2) - env_cfg.delta

    def reward(x, u, rs, ro):
        pe = x[0:3] - rs[0:3]
        ve = x[3:6] - rs[3:6]
        qe = quaternion_multiply(normalize_quaternion(rs[6:10]), quaternion_conjugate(normalize_quaternion(x[6:10])))
        att = jnp.where(qe[0] >= 0.0, 1.0, -1.0) * qe[1:4]
        we = u[1:4] - ro
        c = env_cfg
        cost = (c.w_z * charb(pe[2] / c.s_z) + c.w_vz * charb(ve[2] / c.s_vz)
                + c.w_xy * charb(jnp.sqrt(pe[0] ** 2 + pe[1] ** 2 + 1e-12) / c.s_xy)
                + c.w_vxy * charb(jnp.sqrt(ve[0] ** 2 + ve[1] ** 2 + 1e-12) / c.s_vxy)
                + c.w_tilt * charb(jnp.sqrt(att[0] ** 2 + att[1] ** 2 + 1e-12) / c.s_tilt)
                + c.w_yaw * charb(att[2] / c.s_yaw)
                + c.w_wxy * ((we[0] / c.s_wxy) ** 2 + (we[1] / c.s_wxy) ** 2)
                + c.w_wz * (u[3] / c.s_wz) ** 2 + c.w_thrust * ((u[0] - g) / g) ** 2)
        return (jnp.maximum(-cost, -c.floor), jnp.linalg.norm(pe), jnp.linalg.norm(ve), jnp.linalg.norm(att),
                jnp.linalg.norm(we))

    # ---------------------------------------------------------------- initial-state bank
    tilt = np.deg2rad(env_cfg.init_tilt_deg)
    yaw = np.deg2rad(env_cfg.init_yaw_deg)

    def raw_sample(key):
        k = jax.random.split(key, 6)
        r0s = ref_states[0]
        p = r0s[0:3] + jax.random.uniform(k[0], (3,), dtype, -env_cfg.init_p_range, env_cfg.init_p_range)
        v = r0s[3:6] + jax.random.uniform(k[1], (3,), dtype, -env_cfg.init_v_range, env_cfg.init_v_range)
        rpy = jax.random.uniform(k[2], (3,), dtype, -1.0, 1.0) * jnp.asarray([tilt, tilt, yaw], dtype)
        q = normalize_quaternion(quaternion_multiply(quaternion_from_euler_zyx(rpy[0], rpy[1], rpy[2]),
                                                     normalize_quaternion(r0s[6:10])))
        n, phase, amp, omega = sample_disturbance(k[3], ue, env_cfg, dtype)
        e0 = jax.random.normal(k[4], (3,), dtype)
        e0 = e0 / jnp.maximum(jnp.linalg.norm(e0), 1e-9) * env_cfg.observer_init_err_frac * float(ue.e_bar) \
            * jax.random.uniform(k[5], (), dtype) ** (1.0 / 3.0)
        x = jnp.concatenate([p, v, q])
        return x, n, phase, amp, omega, d_true(n, phase, amp, omega, 0.0) + e0

    keys = jax.random.split(jax.random.PRNGKey(int(env_cfg.bank_seed)), int(env_cfg.bank_size))
    xs, ns, phs, amps, oms, dh0 = jax.jit(jax.vmap(raw_sample))(keys)
    ok = np.asarray(recover_fn(xs, dh0))
    if ok.sum() == 0:
        if env_cfg.require_recoverable:
            raise ValueError("no initial state is in the tightened C_N; shrink init ranges or the disturbance")
        print("[cone ue env] WARNING: no recoverable initial state; using unfiltered starts (smoke test only)")
        ok = np.ones_like(ok)
    a_np, f_np = np.asarray(amps), np.asarray(oms) / (2.0 * np.pi)
    print(f"[cone ue env] initial-state bank: {int(ok.sum())}/{len(ok)} in the tightened C_N ({ok.mean():.1%}); "
          f"episode {max_steps} steps; disturbance {env_cfg.disturbance_mode if env_cfg.disturbance else 'off'}: "
          f"A {a_np.min():.2f}-{a_np.max():.2f} m/s^2, f {f_np.min():.3f}-{f_np.max():.3f} Hz, "
          f"max 2 pi f A {float((2 * np.pi * f_np * a_np).max()):.3f} <= delta_v {float(ue.delta_v):.3f}", flush=True)
    sel = np.nonzero(ok)[0]
    bank = {"x": jnp.asarray(np.asarray(xs)[sel], dtype), "n": jnp.asarray(np.asarray(ns)[sel], dtype),
            "phase": jnp.asarray(np.asarray(phs)[sel], dtype), "amp": jnp.asarray(a_np[sel], dtype),
            "omega": jnp.asarray(np.asarray(oms)[sel], dtype), "d_hat0": jnp.asarray(np.asarray(dh0)[sel], dtype)}
    nb = int(sel.size)

    def env_reset(key):
        i = jax.random.randint(key, (), 0, nb)
        x0 = bank["x"][i]
        xi0 = x0[3:6] - bank["d_hat0"][i] / lam
        z = jnp.asarray(0.0, dtype)
        st = ConeObstacleUEEnvState(x=x0, xi=xi0, d_dir=bank["n"][i], d_phase=bank["phase"][i], d_amp=bank["amp"][i],
                                    d_omega=bank["omega"][i], steps=jnp.int32(0), touched=jnp.asarray(False), ep_return=z,
                                    ep_len=jnp.int32(0), ep_min_h=jnp.asarray(jnp.inf, dtype),
                                    ep_min_z=jnp.asarray(jnp.inf, dtype), ep_pos_err=z, ep_vel_err=z, ep_att_err=z,
                                    ep_safe_sum=z)
        return st, observation(x0, xi0, jnp.int32(0))

    def env_step(state: ConeObstacleUEEnvState, u, key):
        u = jnp.clip(jnp.asarray(u, dtype), lo_u, hi_u)
        t = state.steps.astype(dtype) * dt
        d = d_true(state.d_dir, state.d_phase, state.d_amp, state.d_omega, t)
        dh = lam * (state.x[3:6] - state.xi)
        xdot_nom = quadrotor_dynamics(state.x, u, g, sc.a_cmd_min, sc.a_cmd_max, sc.omega_max)
        x_next = state.x + dt * xdot_nom.at[3:6].add(d)
        x_next = x_next.at[6:10].set(normalize_quaternion(x_next[6:10]))
        xi_next = state.xi + dt * (xdot_nom[3:6] + dh)
        k1 = state.steps + 1
        hc = h_cone(x_next)
        zf = x_next[2] - floor_z
        safe = (hc >= 0.0) & (zf >= 0.0)
        touch_now = touches(x_next) & safe
        first_touch = touch_now & ~state.touched
        touched = state.touched | touch_now
        rs, ro = reference(k1)
        rew, pe, ve, ae, we = (v.astype(dtype) for v in reward(x_next, u, rs, ro))
        rew = rew + float(env_cfg.w_touch) * first_touch.astype(dtype)
        done = ((~safe) & bool(env_cfg.terminate_on_unsafe)) | (k1 >= max_steps)
        ep_return = state.ep_return + rew
        ep_len = state.ep_len + 1
        ep_min_h = jnp.minimum(state.ep_min_h, hc)
        ep_min_z = jnp.minimum(state.ep_min_z, zf)
        ep_pos = state.ep_pos_err + pe
        ep_vel = state.ep_vel_err + ve
        ep_att = state.ep_att_err + ae
        ep_safe = state.ep_safe_sum + safe.astype(dtype)
        df = done.astype(dtype)
        info = ConeObstacleUEStepInfo(
            episode_done=done, safe=safe.astype(dtype), h_cone=hc, z=zf, pos_err=pe, touch=touch_now.astype(dtype),
            d_true=d, d_hat=dh, completed_return=df * ep_return, completed_len=df * ep_len.astype(dtype),
            completed_min_h=df * ep_min_h, completed_min_z=df * ep_min_z,
            completed_pos_err=df * ep_pos / jnp.maximum(ep_len, 1).astype(dtype),
            completed_final_dist=df * jnp.linalg.norm(x_next[0:3] - end_p), completed_final_z=df * x_next[2],
            completed_final_speed=df * jnp.linalg.norm(x_next[3:6]),
            completed_unsafe=df * (ep_safe < ep_len.astype(dtype)).astype(dtype),
            completed_touched=df * touched.astype(dtype),
        )
        obs_true = observation(x_next, xi_next, k1)
        cont = ConeObstacleUEEnvState(x=x_next, xi=xi_next, d_dir=state.d_dir, d_phase=state.d_phase, d_amp=state.d_amp,
                                      d_omega=state.d_omega, steps=k1, touched=touched, ep_return=ep_return, ep_len=ep_len,
                                      ep_min_h=ep_min_h, ep_min_z=ep_min_z, ep_pos_err=ep_pos, ep_vel_err=ep_vel,
                                      ep_att_err=ep_att, ep_safe_sum=ep_safe)
        rst, robs = env_reset(key)
        st_out = jax.tree_util.tree_map(lambda a, b: jnp.where(done, a, b), rst, cont)
        obs_out = jnp.where(done, robs, obs_true)
        return st_out, obs_true, obs_out, rew, done, info

    return ConeObstacleUEEnvFns(
        obs_dim=OBS_DIM, action_dim=4, max_steps=max_steps, reset=jax.jit(env_reset), step=jax.jit(env_step),
        reset_batched=jax.jit(jax.vmap(env_reset)), step_batched=jax.jit(jax.vmap(env_step)), d_hat_of=d_hat_of,
        ref_states=ref_states, residual=False, bank=bank,
    )


__all__ = ["ConeObstacleUEEnvConfig", "ConeObstacleUEEnvFns", "ConeObstacleUEEnvState", "OBS_DIM", "SAFETY_OBS_DIM",
           "build_cone_obstacle_ue_env", "sample_disturbance"]
