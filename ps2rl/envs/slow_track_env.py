"""Tracking environment for a slow reference, built to transfer to PX4 SITL and to hardware.

The tracking part is the recipe of the cone-obstacle stage-1 tracker (``cone_obstacle_ue_env`` with
``use_projection=false``), which tracked to about 2 cm: the same 10-D rate-command model, the same
Charbonnier tracking cost and weights, a disturbance observer ``d_hat`` in the observation, and an
episode that ends once the vehicle is lost, with the discounted floor reward of the remaining steps as
its last reward. No safety filter is involved.

What is added for the transfer (none of it is in the cone-obstacle environment):

* Plant, drawn per episode (``PlantParams``):
    - command delay of 0..``delay_steps_max`` control steps (PX4 / ROS latency);
    - first-order lag of the body rates (the PX4 rate loop) and of the collective thrust (motors);
    - thrust gain error: the vehicle produces ``k_T`` times the commanded thrust (hover-thrust and
      thrust-curve error, battery sag); also a small rate-loop gain error;
    - external acceleration: a random-direction sinusoid plus a constant horizontal push (wind).
* Measurements: white noise on position, velocity and attitude. The policy and the observer see only
  the measured state; the reward uses the true one.
* Observer: ``d_hat = lambda (v_meas - xi)``, ``xi`` advanced with the *commanded* thrust and the
  measured attitude (the ``commanded`` mode of the deployment runtime). It absorbs the thrust gain error,
  the wind and the lag as one acceleration mismatch; started warm, as after take-off on the real vehicle.
* Starts: anywhere along the reference (``p_start_zero`` of the episodes at its start, the hand-over),
  with position, velocity, attitude and yaw offsets.
* Observation, translation-invariant (34-D, ``OBS_LAYOUT``): tracking errors and the reference instead of
  absolute positions, a preview of the reference 0.5 s and 1 s ahead, ``d_hat`` and the previous command
  (+ the base command in residual mode). The deployment runtime rebuilds exactly this vector (``observation_np``).
* Policy output (``policy_mode``): "direct" (default) = the full command, as in the cone-obstacle tracker;
  "residual" = a bounded correction added to the geometric PD base (``base_action``), kept for later.
* ``base_action`` (PD on the measured tracking error, observer as feed-forward) is also the PD controller
  flown on its own in SITL before the policy (about 3 cm in every evaluation scenario of the trainer).

Timeouts are not terminal for the critic (the observation has no time in it, so the value at a timeout
is that of any other state); only losing the reference is.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.envs.slow_track_common import (  # noqa: F401  (re-exported)
    OBS_DIM,
    OBS_LAYOUT,
    RESIDUAL_EXTRA,
    SlowTrackEnvConfig,
    base_action_np,
    command_np,
    load_reference,
    obs_layout,
    observation_np,
)
from ps2rl.utils.quaternion import (
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_from_euler_zyx,
    quaternion_multiply,
    quaternion_rate_matrix,
    rotation_matrix_from_quaternion,
    thrust_axis_world,
)

Array = jax.Array

DELAY_BUF = 4  # command buffer length (supports delays of 0..3 steps)


class PlantParams(NamedTuple):
    delay: Array          # int32, steps
    tau_rate: Array
    tau_thrust: Array
    thrust_gain: Array
    rate_gain: Array
    d_dir: Array          # (3,)
    d_amp: Array
    d_omega: Array
    d_phase: Array
    wind: Array           # (3,)
    noise_scale: Array    # 1 = the configured noise, 0 = none


class SlowTrackState(NamedTuple):
    x: Array              # true [p, v, q]
    w_act: Array          # realised body rates
    a_act: Array          # realised collective thrust (before the gain error)
    buf: Array            # (DELAY_BUF, 4) last commands, newest first
    xi: Array             # observer state
    meas: Array           # (10,) measured [p, v, q] of the current observation
    u_prev: Array
    k_ref: Array          # reference index of the current observation
    steps: Array          # steps in this episode
    max_steps: Array
    plant: PlantParams
    ep_return: Array
    ep_pos_err: Array
    ep_max_pos_err: Array
    ep_du: Array


class SlowTrackInfo(NamedTuple):
    x: Array              # true state after the step (before an automatic reset)
    u_cmd: Array          # command sent to the vehicle (base + correction in residual mode)
    u_base: Array
    episode_done: Array
    terminal: Array       # lost (not a timeout)
    pos_err: Array
    tilt_deg: Array
    d_eff: Array          # true acceleration mismatch the observer should see (gain error + push)
    d_hat: Array
    completed_return: Array
    completed_len: Array
    completed_pos_err: Array
    completed_max_pos_err: Array
    completed_lost: Array
    completed_du: Array


class SlowTrackEnvFns(NamedTuple):
    obs_dim: int
    action_dim: int
    n_ref: int
    reset: Callable
    reset_with: Callable
    step: Callable
    reset_batched: Callable
    step_batched: Callable
    sample_plant: Callable
    base_action: Callable
    ref_states: Array


def _tilt_deg(q):
    return jnp.degrees(jnp.arccos(jnp.clip(1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2), -1.0, 1.0)))


def build_slow_track_env(cfg: SlowTrackEnvConfig, dtype=jnp.float32) -> SlowTrackEnvFns:
    ref = load_reference(cfg.reference_path)
    ref_dt = float(ref["t"][1] - ref["t"][0])
    if abs(ref_dt - cfg.dt) > 1e-9:
        raise ValueError(f"reference dt {ref_dt} != env dt {cfg.dt}")
    n_ref = int(ref["states"].shape[0])
    rs_all = jnp.asarray(ref["states"], dtype)
    ro_all = jnp.asarray(ref["omega_cmd"], dtype)
    ra_all = jnp.asarray(ref["a_cmd"], dtype)
    dt, g, lam = float(cfg.dt), float(cfg.gravity), float(cfg.observer_lambda)
    h = dt / int(cfg.substeps)
    e3 = jnp.asarray([0.0, 0.0, 1.0], dtype)
    low = jnp.asarray(cfg.action_low, dtype)
    high = jnp.asarray(cfg.action_high, dtype)
    mid, half = 0.5 * (low + high), 0.5 * (high - low)
    hover = jnp.asarray([g, 0.0, 0.0, 0.0], dtype)
    ep_steps = int(round(cfg.episode_sec / dt))
    res_box = jnp.asarray(cfg.policy_high, dtype)
    p_low = jnp.asarray(cfg.policy_low, dtype)
    p_high = jnp.asarray(cfg.policy_high, dtype)

    def reference(k):
        i = jnp.clip(k, 0, n_ref - 1)
        return rs_all[i], ro_all[i], ra_all[i]

    residual = cfg.policy_mode == "residual"

    # ------------------------------------------------------------------ base controller
    def base_action(meas, k, d_hat):
        rs, _, ra = reference(k)
        q = normalize_quaternion(meas[6:10])
        a_ref = thrust_axis_world(normalize_quaternion(rs[6:10])) * ra - g * e3      # reference acceleration (feed-forward)
        acc = a_ref - cfg.base_kp * (meas[0:3] - rs[0:3]) - cfg.base_kd * (meas[3:6] - rs[3:6]) - d_hat
        f = acc + g * e3
        R = rotation_matrix_from_quaternion(q)
        zd = f / jnp.maximum(jnp.linalg.norm(f), 1e-6)
        wb = R.T @ jnp.cross(R[:, 2], zd) * cfg.base_katt
        yaw = jnp.arctan2(R[1, 0], R[0, 0])
        qr = normalize_quaternion(rs[6:10])
        yaw_ref = jnp.arctan2(2.0 * (qr[0] * qr[3] + qr[1] * qr[2]), 1.0 - 2.0 * (qr[2] ** 2 + qr[3] ** 2))
        dyaw = jnp.arctan2(jnp.sin(yaw - yaw_ref), jnp.cos(yaw - yaw_ref))
        u = jnp.stack([jnp.dot(f, R[:, 2]), wb[0], wb[1], -cfg.base_kyaw * dyaw]).astype(dtype)
        return jnp.clip(u, low, high)

    # ------------------------------------------------------------------ observation
    def observation(meas, k, d_hat, u_prev):
        rs, ro, ra = reference(k)
        p1 = rs_all[jnp.clip(k + cfg.preview_steps_1, 0, n_ref - 1), 0:3]
        p2 = rs_all[jnp.clip(k + cfg.preview_steps_2, 0, n_ref - 1), 0:3]
        p, v, q = meas[0:3], meas[3:6], normalize_quaternion(meas[6:10])
        q = jnp.where(q[0] < 0.0, -q, q)
        qr = normalize_quaternion(rs[6:10])
        qr = jnp.where(qr[0] < 0.0, -qr, qr)
        parts = [(p - rs[0:3]) / cfg.obs_pos_scale, (v - rs[3:6]) / cfg.obs_vel_scale, q, qr, rs[3:6], ro, jnp.reshape(ra / g - 1.0, (1,)),
                 p1 - p, p2 - p, d_hat, (u_prev - mid) / half]
        if residual:
            parts.append((base_action(meas, k, d_hat) - mid) / half)
        return jnp.concatenate(parts).astype(dtype)

    # ------------------------------------------------------------------ reward
    def charb(e):
        return jnp.sqrt(e * e + cfg.delta ** 2) - cfg.delta

    def reward(x, u, du, rs, ro, res):
        c = cfg
        pe = x[0:3] - rs[0:3]
        ve = x[3:6] - rs[3:6]
        qe = quaternion_multiply(normalize_quaternion(rs[6:10]), quaternion_conjugate(normalize_quaternion(x[6:10])))
        att = jnp.where(qe[0] >= 0.0, 1.0, -1.0) * qe[1:4]
        we = u[1:4] - ro
        cost = (c.w_z * charb(pe[2] / c.s_z) + c.w_vz * charb(ve[2] / c.s_vz)
                + c.w_xy * charb(jnp.sqrt(pe[0] ** 2 + pe[1] ** 2 + 1e-12) / c.s_xy)
                + c.w_vxy * charb(jnp.sqrt(ve[0] ** 2 + ve[1] ** 2 + 1e-12) / c.s_vxy)
                + c.w_tilt * charb(jnp.sqrt(att[0] ** 2 + att[1] ** 2 + 1e-12) / c.s_tilt)
                + c.w_yaw * charb(att[2] / c.s_yaw)
                + c.w_wxy * ((we[0] / c.s_wxy) ** 2 + (we[1] / c.s_wxy) ** 2)
                + c.w_wz * (u[3] / c.s_wz) ** 2 + c.w_thrust * ((u[0] - g) / g) ** 2
                + c.w_du * ((du[0] / c.s_du_a) ** 2 + jnp.sum((du[1:4] / c.s_du_w) ** 2))
                + c.w_res * jnp.sum((res / res_box) ** 2))
        return jnp.maximum(-cost, -c.floor), jnp.linalg.norm(pe)

    # ------------------------------------------------------------------ plant
    def sample_plant(key) -> PlantParams:
        k = jax.random.split(key, 10)
        u = lambda kk, lo, hi: jax.random.uniform(kk, (), dtype, lo, hi)
        n = jax.random.normal(k[0], (3,), dtype) * jnp.asarray([1.0, 1.0, cfg.dist_z_scale], dtype)
        n = n / jnp.maximum(jnp.linalg.norm(n), 1e-9)
        f = jnp.exp(u(k[1], np.log(cfg.dist_freq_min_hz), np.log(cfg.dist_freq_max_hz)))
        wdir = u(k[2], 0.0, 2.0 * np.pi)
        wind = u(k[3], 0.0, cfg.wind_max) * jnp.stack([jnp.cos(wdir), jnp.sin(wdir), 0.0]).astype(dtype)
        return PlantParams(
            delay=jax.random.randint(k[4], (), 0, int(cfg.delay_steps_max) + 1),
            tau_rate=u(k[5], cfg.tau_rate_min, cfg.tau_rate_max),
            tau_thrust=u(k[6], cfg.tau_thrust_min, cfg.tau_thrust_max),
            thrust_gain=u(k[7], cfg.thrust_gain_min, cfg.thrust_gain_max),
            rate_gain=u(k[8], cfg.rate_gain_min, cfg.rate_gain_max),
            d_dir=n, d_amp=u(k[9], 0.0, cfg.dist_amp_max), d_omega=2.0 * np.pi * f,
            d_phase=u(jax.random.fold_in(key, 7), 0.0, 2.0 * np.pi), wind=wind,
            noise_scale=jnp.asarray(1.0, dtype))

    def push(pl: PlantParams, t):
        return pl.d_amp * jnp.sin(pl.d_omega * t + pl.d_phase) * pl.d_dir + pl.wind

    def measure(x, pl: PlantParams, key):
        k1, k2, k3 = jax.random.split(key, 3)
        s = pl.noise_scale
        p = x[0:3] + s * cfg.noise_p * jax.random.normal(k1, (3,), dtype)
        v = x[3:6] + s * cfg.noise_v * jax.random.normal(k2, (3,), dtype)
        e = s * cfg.noise_att * jax.random.normal(k3, (3,), dtype)
        dq = normalize_quaternion(jnp.concatenate([jnp.ones((1,), dtype), 0.5 * e]))
        q = normalize_quaternion(quaternion_multiply(normalize_quaternion(x[6:10]), dq))
        return jnp.concatenate([p, v, q])

    def a_nom(meas, u):
        return -g * e3 + thrust_axis_world(normalize_quaternion(meas[6:10])) * u[0]

    # ------------------------------------------------------------------ reset
    def reset_with(key, k0, x0, pl: PlantParams):
        """Start at reference index k0 in true state x0 with plant pl (evaluation uses this directly)."""
        km, ko = jax.random.split(key)
        meas = measure(x0, pl, km)
        # warm observer: d_hat(0) = the true mismatch at hover + an error of observer_init_err
        d0 = (pl.thrust_gain - 1.0) * g * e3 + push(pl, 0.0)
        err = jax.random.normal(ko, (3,), dtype)
        d_hat0 = d0 + cfg.observer_init_err * err / jnp.maximum(jnp.linalg.norm(err), 1e-9) \
            * jax.random.uniform(jax.random.fold_in(ko, 1), (), dtype)
        xi0 = meas[3:6] - d_hat0 / lam
        rs, ro, _ = reference(k0)
        w0 = ro
        buf = jnp.tile(hover[None, :], (DELAY_BUF, 1))
        z = jnp.asarray(0.0, dtype)
        st = SlowTrackState(x=x0, w_act=w0, a_act=jnp.asarray(g, dtype) / pl.thrust_gain, buf=buf, xi=xi0, meas=meas,
                            u_prev=hover, k_ref=jnp.int32(k0), steps=jnp.int32(0),
                            max_steps=jnp.int32(ep_steps), plant=pl, ep_return=z, ep_pos_err=z, ep_max_pos_err=z, ep_du=z)
        return st, observation(meas, jnp.int32(k0), d_hat0, hover)

    def sample_start(key):
        k = jax.random.split(key, 6)
        at_zero = jax.random.uniform(k[0], ()) < cfg.p_start_zero
        k0 = jnp.where(at_zero, 0, jax.random.randint(k[1], (), 0, n_ref))
        rs, _, _ = reference(k0)
        pr = jnp.where(at_zero, cfg.init_p_start, cfg.init_p)
        dp = jax.random.uniform(k[2], (3,), dtype, -1.0, 1.0) * pr * jnp.asarray([1.0, 1.0, 0.5], dtype)
        dv = jax.random.uniform(k[3], (3,), dtype, -cfg.init_v, cfg.init_v)
        rpy = jax.random.uniform(k[4], (3,), dtype, -1.0, 1.0) * jnp.deg2rad(
            jnp.asarray([cfg.init_tilt_deg, cfg.init_tilt_deg, cfg.init_yaw_deg], dtype))
        q = normalize_quaternion(quaternion_multiply(quaternion_from_euler_zyx(rpy[0], rpy[1], rpy[2]),
                                                     normalize_quaternion(rs[6:10])))
        x0 = jnp.concatenate([rs[0:3] + dp, rs[3:6] + dv, q])
        return k0, x0

    def env_reset(key):
        ks, kp, kr = jax.random.split(key, 3)
        k0, x0 = sample_start(ks)
        st, obs = reset_with(kr, k0, x0, sample_plant(kp))
        w0 = jax.random.uniform(jax.random.fold_in(kr, 3), (3,), dtype, -cfg.init_rate, cfg.init_rate)
        return st._replace(w_act=st.w_act + w0 * jnp.asarray([1.0, 1.0, 0.3], dtype)), obs

    # ------------------------------------------------------------------ step
    def env_step(state: SlowTrackState, u, key):
        pl = state.plant
        a_pol = jnp.clip(jnp.asarray(u, dtype), p_low, p_high)
        d_hat = lam * (state.meas[3:6] - state.xi)
        if residual:
            ub = base_action(state.meas, state.k_ref, d_hat)
            u = jnp.clip(ub + a_pol, low, high)
            res = a_pol
        else:
            ub = jnp.zeros((4,), dtype)
            u = jnp.clip(a_pol, low, high)
            res = jnp.zeros((4,), dtype)
        buf = jnp.concatenate([u[None, :], state.buf[:-1]], axis=0)
        applied = buf[jnp.clip(pl.delay, 0, DELAY_BUF - 1)]
        t0 = state.steps.astype(dtype) * dt
        ar = 1.0 - jnp.exp(-h / pl.tau_rate)
        aa = 1.0 - jnp.exp(-h / pl.tau_thrust)

        def sub(i, c):
            x, w, a = c
            w = w + ar * (pl.rate_gain * applied[1:4] - w)
            a = a + aa * (applied[0] - a)
            q = normalize_quaternion(x[6:10])
            acc = -g * e3 + thrust_axis_world(q) * (pl.thrust_gain * a) + push(pl, t0 + i * h)
            v = x[3:6] + h * acc
            p = x[0:3] + h * v
            q = normalize_quaternion(q + h * 0.5 * quaternion_rate_matrix(q) @ w)
            return jnp.concatenate([p, v, q]), w, a

        x1, w1, a1 = jax.lax.fori_loop(0, int(cfg.substeps), sub, (state.x, state.w_act, state.a_act))
        # observer (measured quantities only, commanded thrust)
        xi1 = state.xi + dt * (a_nom(state.meas, u) + d_hat)
        km, _ = jax.random.split(key)
        meas1 = measure(x1, pl, km)
        d_hat1 = lam * (meas1[3:6] - xi1)
        k1 = state.k_ref + 1
        s1 = state.steps + 1
        rs, ro, _ = reference(k1)
        du = u - state.u_prev
        rew, pe = reward(x1, u, du, rs, ro, res)
        tilt = _tilt_deg(normalize_quaternion(x1[6:10]))
        lost = (pe > cfg.lost_pos_error) | (tilt > cfg.lost_tilt_deg) | ~jnp.all(jnp.isfinite(x1))
        timeout = s1 >= state.max_steps
        gt = float(cfg.terminal_gamma)
        rem = jnp.maximum(state.max_steps - s1, 0).astype(dtype)
        rew = jnp.where(lost, -cfg.floor - cfg.floor * gt * (1.0 - gt ** rem) / (1.0 - gt), rew).astype(dtype)
        done = lost | timeout
        du_n = jnp.linalg.norm((du / half))
        ep_return = state.ep_return + rew
        ep_pos = state.ep_pos_err + pe
        ep_max = jnp.maximum(state.ep_max_pos_err, pe)
        ep_du = state.ep_du + du_n
        df = done.astype(dtype)
        L = s1.astype(dtype)
        d_eff = (pl.thrust_gain - 1.0) * a1 * thrust_axis_world(normalize_quaternion(x1[6:10])) + push(pl, t0 + dt)
        info = SlowTrackInfo(x=x1, u_cmd=u, u_base=ub, episode_done=done, terminal=lost, pos_err=pe, tilt_deg=tilt, d_eff=d_eff, d_hat=d_hat1,
                             completed_return=df * ep_return, completed_len=df * L,
                             completed_pos_err=df * ep_pos / L, completed_max_pos_err=df * ep_max,
                             completed_lost=df * lost.astype(dtype), completed_du=df * ep_du / L)
        x1 = jnp.where(jnp.all(jnp.isfinite(x1)), x1, state.x)
        cont = state._replace(x=x1, w_act=w1, a_act=a1, buf=buf, xi=xi1, meas=meas1, u_prev=u, k_ref=k1, steps=s1,
                              ep_return=ep_return, ep_pos_err=ep_pos, ep_max_pos_err=ep_max, ep_du=ep_du)
        obs_true = observation(meas1, k1, d_hat1, u)
        rst, robs = env_reset(jax.random.fold_in(key, 11))
        st_out = jax.tree_util.tree_map(lambda a, b: jnp.where(done, a, b), rst, cont)
        obs_out = jnp.where(done, robs, obs_true)
        return st_out, obs_true, obs_out, rew, done, info

    return SlowTrackEnvFns(
        obs_dim=OBS_DIM + (4 if residual else 0), action_dim=4, n_ref=n_ref, reset=jax.jit(env_reset), reset_with=reset_with,
        step=jax.jit(env_step), reset_batched=jax.jit(jax.vmap(env_reset)), step_batched=jax.jit(jax.vmap(env_step)),
        sample_plant=sample_plant, base_action=base_action, ref_states=rs_all)


__all__ = ["OBS_DIM", "OBS_LAYOUT", "RESIDUAL_EXTRA", "obs_layout", "PlantParams", "SlowTrackEnvConfig", "SlowTrackEnvFns", "SlowTrackState",
           "base_action_np", "build_slow_track_env", "command_np", "load_reference", "observation_np"]
