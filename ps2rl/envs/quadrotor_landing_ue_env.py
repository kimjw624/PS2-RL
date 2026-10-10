"""Phase-II landing environment with a bounded disturbance and a disturbance observer.

Plant (true):      x_dot = f(x) + g(x) u + E_d d(t),
                   d(t) = A n sin(2 pi f t + phi)   (n, phi random per episode; ||d|| <= A, ||d_dot|| <= 2 pi f A),
Observer (UE-bCBF): d_hat = lambda (v - xi),  xi_dot = a_nom(x, u) + d_hat,
                   started warm (d_hat(0) = d(0) + e0, |e0| <= e_bar / 2): the landing segment begins
                   after the vehicle has flown long enough (> warm-up ~0.2 s) for the bound e_bar to hold.

Observation (29-D): [x (10), d_hat (3), ref_state (10), ref_omega (3), t, sin, cos]
(+ u_nom (4) = 33-D with ``nominal_controller='tracker'``). The first 13 entries are what
the UE filter needs (physical state + estimate).

Nominal controller (``nominal_controller='tracker'``): a geometric reference tracker (PD on
position with reference-acceleration feedforward and d_hat compensation -> thrust vector ->
body rates on the tilt error). Its command u_nom is part of the observation; the Phase-II
trainer then learns a residual, u_ref = u_nom + u_res, and the CIL filters u_ref.

Reward: the landing reward of the nominal landing Phase II (Charbonnier, altitude and
descent rate first, horizontal second so the policy slides along the wall instead of
pushing into it), tracking a landing reference that cuts the approach cone.

Termination: leaving S (cone or floor) or the time limit. Safety is evaluated on the
*true* (disturbed) state.

Initial states: (x0, disturbance) pairs drawn around the reference start and kept only if
(x0, d_hat(0)) is in the tightened C_N of the UE backup (Phase-II episodes must start in
the set the filter keeps invariant).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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

from ps2rl.utils.quaternion import rotation_matrix_from_quaternion

Array = jax.Array
_ROOT = Path(__file__).resolve().parents[2]
OBS_DIM = 29
SAFETY_OBS_DIM = 13
NOMINAL_DIM = 4


@dataclass(frozen=True)
class LandingUEEnvConfig:
    reference_path: str = "ps2rl/envs/assets/quadrotor_landing_cornercut_reference.npz"
    extra_sec: float = 0.5  # episode = reference duration + extra_sec
    init_p_range: float = 0.2
    init_v_range: float = 0.3
    init_tilt_deg: float = 5.0
    init_yaw_deg: float = 5.0
    bank_size: int = 4096
    bank_seed: int = 12345
    disturbance: bool = True
    observer_init_err_frac: float = 0.5  # |e0| <= frac * e_bar
    require_recoverable: bool = True  # False only for smoke tests with an untrained backup
    # end the episode at the first cone/floor violation. False for the vanilla (no-CIL) warm-start stage,
    # like the repo's quadrotor env: with a cost-only reward, ending early would pay off (crash to stop the cost)
    terminate_on_unsafe: bool = True
    # nominal controller whose command is appended to the observation (residual RL); "none" = pure policy
    nominal_controller: str = "tracker"
    trk_kp: float = 4.0
    trk_kd: float = 3.0
    trk_k_att: float = 8.0
    trk_k_yaw: float = 2.0
    trk_d_hat_feedforward: bool = True
    # landing reward (cost = sum w * charb(err / s)), as in the nominal landing Phase II
    s_z: float = 0.05
    w_z: float = 1.0
    s_vz: float = 0.3
    w_vz: float = 1.0
    s_xy: float = 0.3
    w_xy: float = 0.5
    s_vxy: float = 0.5
    w_vxy: float = 0.25
    s_tilt: float = 0.2
    w_tilt: float = 0.2
    s_yaw: float = 0.5
    w_yaw: float = 0.05
    s_wz: float = 0.5
    w_wz: float = 1.0
    s_wxy: float = 2.0
    w_wxy: float = 0.02
    w_thrust: float = 0.01
    delta: float = 0.1
    floor: float = 50.0

    def as_dict(self):
        return asdict(self)


class LandingUEEnvState(NamedTuple):
    x: Array
    xi: Array  # observer internal state
    d_dir: Array
    d_phase: Array
    steps: Array
    ep_return: Array
    ep_len: Array
    ep_min_h: Array
    ep_min_z: Array
    ep_pos_err: Array
    ep_vel_err: Array
    ep_att_err: Array
    ep_safe_sum: Array


class LandingUEStepInfo(NamedTuple):
    """Landing/UE fields + every field of ``quadrotor_env.QuadrotorStepInfo`` (so the repo's Phase-II
    trainer, its episode bookkeeping and ``_evaluate_policy`` work on this env unchanged)."""

    episode_done: Array
    safe: Array
    h_cone: Array
    z: Array
    pos_err: Array
    d_true: Array
    d_hat: Array
    completed_return: Array
    completed_len: Array
    completed_min_h: Array
    completed_min_z: Array
    completed_pos_err: Array
    completed_final_dist: Array
    completed_final_z: Array
    completed_final_speed: Array
    completed_unsafe: Array
    # --- QuadrotorStepInfo fields ---
    is_safe: Array
    pos_error_norm: Array
    vel_error_norm: Array
    att_error_norm: Array
    omega_ref_error_norm: Array
    hard_deck_margin: Array  # min(h_cone, z - z_pad): >= 0 is safe
    ref_progress: Array
    ref_time_sec: Array
    ref_state: Array
    ref_omega: Array
    completed_safe_rate: Array
    completed_pos_error_norm: Array
    completed_vel_error_norm: Array
    completed_att_error_norm: Array
    completed_hard_deck_margin_min: Array
    disturbance_accel: Array


class LandingUEEnvFns(NamedTuple):
    obs_dim: int
    action_dim: int
    max_steps: int
    reset: Callable
    step: Callable
    reset_batched: Callable
    step_batched: Callable
    d_hat_of: Callable  # state -> d_hat
    ref_states: Array
    residual: bool  # True: the last NOMINAL_DIM obs entries are u_nom


def build_landing_ue_env(env_cfg: LandingUEEnvConfig, cbf_cfg: Any, rt: Any, recover_fn: Callable,
                         dtype=jnp.float32) -> LandingUEEnvFns:
    lc = cbf_cfg.landing
    ue = cbf_cfg.ue
    ref_path = Path(env_cfg.reference_path)
    if not ref_path.exists() and (_ROOT / ref_path).exists():
        ref_path = _ROOT / ref_path
    ref = np.load(ref_path)
    ref_states = jnp.asarray(ref["states"], dtype=dtype)
    ref_omega = jnp.asarray(ref["omega_cmd"], dtype=dtype)
    ref_t = np.asarray(ref["t"], dtype=np.float64)
    ref_dt = float(ref_t[1] - ref_t[0])
    if abs(ref_dt - float(lc.dt)) > 1e-9:
        raise ValueError(f"reference dt {ref_dt} != checkpoint dt {lc.dt}")
    n_ref = int(ref_states.shape[0])
    max_steps = int(round((float(ref_t[-1]) + float(env_cfg.extra_sec)) / float(lc.dt)))
    dt = float(lc.dt)
    g = float(lc.gravity)
    lam = float(ue.observer_lambda)
    amp = float(ue.delta_d) if env_cfg.disturbance else 0.0
    omega_d = 2.0 * np.pi * float(ue.frequency_hz)
    r0, tt, eps2 = float(lc.cone_r0), float(np.tan(np.deg2rad(lc.cone_theta_deg))), float(lc.cone_eps) ** 2
    t_total = float(ref_t[-1])

    def d_true(dir_, phase, t):
        return amp * jnp.sin(omega_d * t + phase) * dir_

    def h_cone(x):
        return r0 + tt * (x[2] - lc.pad_z) - jnp.sqrt((x[0] - lc.pad_x) ** 2 + (x[1] - lc.pad_y) ** 2 + eps2)

    def reference(k):
        i = jnp.clip(k, 0, n_ref - 1)
        return ref_states[i], ref_omega[i]

    def d_hat_of(state):
        return lam * (state.x[3:6] - state.xi)

    # --- nominal geometric tracker (reference acceleration from the reference's thrust and attitude)
    if env_cfg.nominal_controller not in ("none", "tracker"):
        raise ValueError(f"nominal_controller must be 'none' or 'tracker', got {env_cfg.nominal_controller!r}")
    residual = env_cfg.nominal_controller == "tracker"
    q_ref_np = np.asarray(ref["states"])[:, 6:10]
    w_, x_, y_, z_ = (q_ref_np[:, i] for i in range(4))
    zb_ref = np.stack([2 * (x_ * z_ + w_ * y_), 2 * (y_ * z_ - w_ * x_), 1 - 2 * (x_ * x_ + y_ * y_)], -1)
    acc_ref = jnp.asarray(zb_ref * np.asarray(ref["a_cmd"])[:, None] - np.array([0.0, 0.0, g]), dtype=dtype)
    lo_u = jnp.asarray(cbf_cfg.action_low, dtype)
    hi_u = jnp.asarray(cbf_cfg.action_high, dtype)

    def u_nominal(x, dh, k):
        i = jnp.clip(k, 0, n_ref - 1)
        hold = (k >= n_ref - 1).astype(dtype)
        rs = ref_states[i]
        p, v, q = x[0:3], x[3:6], x[6:10]
        a_des = ((1.0 - hold) * acc_ref[i] + env_cfg.trk_kp * (rs[0:3] - p)
                 + env_cfg.trk_kd * ((1.0 - hold) * rs[3:6] - v) + jnp.asarray([0.0, 0.0, g], dtype))
        if env_cfg.trk_d_hat_feedforward:
            a_des = a_des - dh
        zd = a_des / jnp.maximum(jnp.linalg.norm(a_des), 1e-6)
        rot = rotation_matrix_from_quaternion(normalize_quaternion(q))
        zb = rot[:, 2]
        thrust = jnp.dot(a_des, zb)
        err_b = rot.T @ jnp.cross(zb, zd)
        yaw = jnp.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        rates = jnp.stack([env_cfg.trk_k_att * err_b[0], env_cfg.trk_k_att * err_b[1], -env_cfg.trk_k_yaw * yaw])
        rates = rates + (1.0 - hold) * ref_omega[i]
        return jnp.clip(jnp.concatenate([thrust[None], rates]), lo_u, hi_u).astype(dtype)

    def observation(x, xi, k):
        rs, ro = reference(k)
        t = k.astype(dtype) * dt
        ph = 2.0 * jnp.pi * jnp.clip(t / t_total, 0.0, 1.0)
        dh = lam * (x[3:6] - xi)
        parts = [x, dh, rs, ro, jnp.stack([t, jnp.sin(ph), jnp.cos(ph)])]
        if residual:
            parts.append(u_nominal(x, dh, k))
        return jnp.concatenate(parts).astype(dtype)

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
        k = jax.random.split(key, 8)
        r0s = ref_states[0]
        p = r0s[0:3] + jax.random.uniform(k[0], (3,), dtype, -env_cfg.init_p_range, env_cfg.init_p_range)
        v = r0s[3:6] + jax.random.uniform(k[1], (3,), dtype, -env_cfg.init_v_range, env_cfg.init_v_range)
        rpy = jax.random.uniform(k[2], (3,), dtype, -1.0, 1.0) * jnp.asarray([tilt, tilt, yaw], dtype)
        q = normalize_quaternion(quaternion_multiply(quaternion_from_euler_zyx(rpy[0], rpy[1], rpy[2]),
                                                     normalize_quaternion(r0s[6:10])))
        n = jax.random.normal(k[3], (3,), dtype)
        n = n / jnp.maximum(jnp.linalg.norm(n), 1e-9)
        phase = jax.random.uniform(k[4], (), dtype, 0.0, 2.0 * np.pi)
        e0 = jax.random.normal(k[5], (3,), dtype)
        e0 = e0 / jnp.maximum(jnp.linalg.norm(e0), 1e-9) * env_cfg.observer_init_err_frac * float(ue.e_bar) \
            * jax.random.uniform(k[6], (), dtype) ** (1.0 / 3.0)
        x = jnp.concatenate([p, v, q])
        d0 = d_true(n, phase, 0.0)
        return x, n, phase, d0 + e0

    keys = jax.random.split(jax.random.PRNGKey(int(env_cfg.bank_seed)), int(env_cfg.bank_size))
    xs, ns, phs, dh0 = jax.jit(jax.vmap(raw_sample))(keys)
    ok = np.asarray(recover_fn(xs, dh0))
    if ok.sum() == 0:
        if env_cfg.require_recoverable:
            raise ValueError("no initial state is in the tightened C_N; shrink init ranges or the disturbance")
        print("[landing ue env] WARNING: no recoverable initial state; using unfiltered starts (smoke test only)")
        ok = np.ones_like(ok)
    print(f"[landing ue env] initial-state bank: {int(ok.sum())}/{len(ok)} in the tightened C_N ({ok.mean():.1%}); "
          f"episode {max_steps} steps; disturbance A={amp} f={ue.frequency_hz} Hz", flush=True)
    sel = np.nonzero(ok)[0]
    bank_x = jnp.asarray(np.asarray(xs)[sel], dtype)
    bank_n = jnp.asarray(np.asarray(ns)[sel], dtype)
    bank_ph = jnp.asarray(np.asarray(phs)[sel], dtype)
    bank_dh = jnp.asarray(np.asarray(dh0)[sel], dtype)
    nb = int(sel.size)

    def env_reset(key):
        i = jax.random.randint(key, (), 0, nb)
        x0 = bank_x[i]
        xi0 = x0[3:6] - bank_dh[i] / lam  # d_hat(0) = lam (v - xi) = d(0) + e0
        z = jnp.asarray(0.0, dtype)
        st = LandingUEEnvState(x=x0, xi=xi0, d_dir=bank_n[i], d_phase=bank_ph[i], steps=jnp.int32(0), ep_return=z,
                               ep_len=jnp.int32(0), ep_min_h=jnp.asarray(jnp.inf, dtype),
                               ep_min_z=jnp.asarray(jnp.inf, dtype), ep_pos_err=z, ep_vel_err=z, ep_att_err=z,
                               ep_safe_sum=z)
        return st, observation(x0, xi0, jnp.int32(0))

    def env_step(state: LandingUEEnvState, u, key):
        u = jnp.clip(jnp.asarray(u, dtype), jnp.asarray(cbf_cfg.action_low, dtype), jnp.asarray(cbf_cfg.action_high, dtype))
        t = state.steps.astype(dtype) * dt
        d = d_true(state.d_dir, state.d_phase, t)
        dh = lam * (state.x[3:6] - state.xi)
        xdot_nom = quadrotor_dynamics(state.x, u, g, lc.a_cmd_min, lc.a_cmd_max, lc.omega_max)
        x_next = state.x + dt * xdot_nom.at[3:6].add(d)
        x_next = x_next.at[6:10].set(normalize_quaternion(x_next[6:10]))
        xi_next = state.xi + dt * (xdot_nom[3:6] + dh)
        k1 = state.steps + 1
        hc = h_cone(x_next)
        zf = x_next[2] - lc.pad_z
        safe = (hc >= 0.0) & (zf >= 0.0)
        rs, ro = reference(k1)
        rew, pe, ve, ae, we = (v.astype(dtype) for v in reward(x_next, u, rs, ro))
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
        len_f = jnp.maximum(ep_len, 1).astype(dtype)
        margin = jnp.minimum(hc, zf)
        info = LandingUEStepInfo(
            episode_done=done, safe=safe.astype(dtype), h_cone=hc, z=zf, pos_err=pe, d_true=d, d_hat=dh,
            completed_return=df * ep_return, completed_len=df * ep_len.astype(dtype), completed_min_h=df * ep_min_h,
            completed_min_z=df * ep_min_z, completed_pos_err=df * ep_pos / jnp.maximum(ep_len, 1).astype(dtype),
            completed_final_dist=df * jnp.sqrt((x_next[0] - lc.pad_x) ** 2 + (x_next[1] - lc.pad_y) ** 2),
            completed_final_z=df * zf, completed_final_speed=df * jnp.linalg.norm(x_next[3:6]),
            completed_unsafe=df * (ep_safe < ep_len.astype(dtype)).astype(dtype),  # any unsafe step in the episode
            is_safe=safe.astype(dtype), pos_error_norm=pe, vel_error_norm=ve, att_error_norm=ae,
            omega_ref_error_norm=we, hard_deck_margin=margin,
            ref_progress=jnp.clip(k1.astype(dtype) / max(n_ref - 1, 1), 0.0, 1.0), ref_time_sec=k1.astype(dtype) * dt,
            ref_state=rs, ref_omega=ro, completed_safe_rate=df * ep_safe / len_f,
            completed_pos_error_norm=df * ep_pos / len_f, completed_vel_error_norm=df * ep_vel / len_f,
            completed_att_error_norm=df * ep_att / len_f,
            completed_hard_deck_margin_min=df * jnp.minimum(ep_min_h, ep_min_z), disturbance_accel=d,
        )
        obs_true = observation(x_next, xi_next, k1)
        cont = LandingUEEnvState(x=x_next, xi=xi_next, d_dir=state.d_dir, d_phase=state.d_phase, steps=k1,
                                 ep_return=ep_return, ep_len=ep_len, ep_min_h=ep_min_h, ep_min_z=ep_min_z,
                                 ep_pos_err=ep_pos, ep_vel_err=ep_vel, ep_att_err=ep_att, ep_safe_sum=ep_safe)
        rst, robs = env_reset(key)
        st_out = jax.tree_util.tree_map(lambda a, b: jnp.where(done, a, b), rst, cont)
        obs_out = jnp.where(done, robs, obs_true)
        return st_out, obs_true, obs_out, rew, done, info

    return LandingUEEnvFns(
        obs_dim=OBS_DIM + (NOMINAL_DIM if residual else 0), action_dim=4, max_steps=max_steps, reset=jax.jit(env_reset), step=jax.jit(env_step),
        reset_batched=jax.jit(jax.vmap(env_reset)), step_batched=jax.jit(jax.vmap(env_step)), d_hat_of=d_hat_of,
        ref_states=ref_states, residual=residual,
    )


# ------------------------------------------------------------------ repo-trainer adapter
def landing_ue_env_config_from_quadrotor(env_cfg: Any, base: LandingUEEnvConfig | None = None) -> LandingUEEnvConfig:
    """Take reference, episode length and initial-state ranges from the entry's QuadrotorEnvConfig."""
    base = LandingUEEnvConfig() if base is None else base
    return replace(
        base,
        reference_path=str(env_cfg.reference_path),
        extra_sec=float(env_cfg.max_steps_extra_sec),
        init_p_range=float(max(env_cfg.init_px_range, env_cfg.init_py_range, env_cfg.init_pz_range)),
        init_v_range=float(env_cfg.init_v_range),
        init_tilt_deg=float(env_cfg.init_tilt_deg_range),
        init_yaw_deg=float(env_cfg.init_yaw_deg_range),
    )


def build_quadrotor_landing_ue_env(env_cfg: Any, cbf_cfg: Any, ue_env_cfg: LandingUEEnvConfig | None = None,
                                   dtype=jnp.float32) -> LandingUEEnvFns:
    """Same role as ``build_quadrotor_landing_env(cfg, cbf_cfg, reward_cfg)`` of the nominal landing Phase II:
    built from the entry's ``QuadrotorEnvConfig`` plus the UE CIL config; returns the
    ``QuadrotorEnvFns`` interface (obs_dim, action_dim, reset, step, reset_batched, step_batched)."""
    from ps2rl.cil.quadrotor_landing_ue_bcbf import get_cached_runtime, make_recoverability_fn_ue

    if abs(float(env_cfg.dt) - float(cbf_cfg.landing.dt)) > 1e-12:
        raise ValueError(f"env dt {env_cfg.dt} must equal the Phase-I checkpoint dt {cbf_cfg.landing.dt}")
    cfg = landing_ue_env_config_from_quadrotor(env_cfg, ue_env_cfg)
    rt = get_cached_runtime(cbf_cfg)
    return build_landing_ue_env(cfg, cbf_cfg, rt, make_recoverability_fn_ue(cbf_cfg, rt), dtype=dtype)


__all__ = [
    "LandingUEEnvConfig",
    "LandingUEEnvFns",
    "LandingUEEnvState",
    "NOMINAL_DIM",
    "OBS_DIM",
    "SAFETY_OBS_DIM",
    "build_landing_ue_env",
    "build_quadrotor_landing_ue_env",
    "landing_ue_env_config_from_quadrotor",
]
