"""Pieces of the slow-track tracker shared by training and the flight runtime, without JAX.

``SlowTrackEnvConfig`` (environment, action box, base controller, observation), the reference loader,
and numpy twins of the environment's observation, base controller and command, so the flight
runtime builds exactly what the policy saw in training. See ``ps2rl.envs.slow_track_env``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]

OBS_LAYOUT = (("e_p", 3), ("e_v", 3), ("q", 4), ("q_ref", 4), ("v_ref", 3), ("omega_ref", 3), ("a_ref_rel", 1),
              ("preview_1", 3), ("preview_2", 3), ("d_hat", 3), ("u_prev", 4))
OBS_DIM = sum(n for _, n in OBS_LAYOUT)        # direct mode
RESIDUAL_EXTRA = (("u_base", 4),)             # appended in residual mode (OBS_DIM + 4)


def obs_layout(cfg) -> tuple:
    return OBS_LAYOUT + (RESIDUAL_EXTRA if cfg.policy_mode == "residual" else ())


@dataclass(frozen=True)
class SlowTrackEnvConfig:
    reference_path: str = "ps2rl/envs/assets/slow_loop/quadrotor_slow_loop_reference.npz"
    dt: float = 0.02
    gravity: float = 9.81
    substeps: int = 2
    # "residual": command = base controller + the policy's correction (bounded by res_*); "direct": the policy's
    # action is the command. The base is a geometric PD on the measured tracking error with the observer's
    # d_hat as feed-forward (``base_action``; about 3 cm in every evaluation scenario on its own).
    policy_mode: str = "direct"
    res_a: float = 1.5            # correction box: thrust [m/s^2]
    res_w_xy: float = 0.6         # roll / pitch rate [rad/s]
    res_w_z: float = 0.2          # yaw rate [rad/s]
    w_res: float = 0.02           # cost on the correction, sum((du_res / res_box)^2)
    base_kp: float = 3.0
    base_kd: float = 3.0
    base_katt: float = 8.0
    base_kyaw: float = 1.0
    # action box [a_cmd (m/s^2), p, q, r (rad/s)]; must be inside what the vehicle can do
    a_min: float = 4.905          # 0.5 g
    a_max: float = 14.715         # 1.5 g
    omega_xy_max: float = 2.0
    omega_z_max: float = 0.5
    # episodes
    episode_sec: float = 20.0     # training episode length (starts anywhere along the reference)
    p_start_zero: float = 0.3     # fraction of episodes starting at the reference start (the hand-over)
    init_p_start: float = 0.5     # position offset at the reference start, per axis [m] (z: half)
    init_p: float = 0.25          # position offset elsewhere along the reference [m]
    init_v: float = 0.2
    init_tilt_deg: float = 5.0
    init_yaw_deg: float = 10.0
    init_rate: float = 0.2
    # plant randomisation (per episode)
    delay_steps_max: int = 2      # 0..2 steps = 0..40 ms extra
    tau_rate_min: float = 0.02
    tau_rate_max: float = 0.08
    tau_thrust_min: float = 0.02
    tau_thrust_max: float = 0.06
    thrust_gain_min: float = 0.9
    thrust_gain_max: float = 1.1
    rate_gain_min: float = 0.95
    rate_gain_max: float = 1.05
    dist_amp_max: float = 0.5     # sinusoidal push amplitude [m/s^2]
    dist_freq_min_hz: float = 0.02
    dist_freq_max_hz: float = 0.5
    dist_z_scale: float = 0.5     # vertical part of the sinusoid's direction scaled by this
    wind_max: float = 0.3         # constant horizontal push [m/s^2]
    # measurement noise (standard deviations)
    noise_p: float = 0.01
    noise_v: float = 0.02
    noise_att: float = 0.005      # rad, per axis
    # observer
    observer_lambda: float = 5.0
    observer_init_err: float = 0.1
    # observation
    preview_steps_1: int = 25
    preview_steps_2: int = 50
    # tracking errors e_p, e_v enter the observation divided by these (1 = metres, as in the first runs; 0.2 / 0.5
    # put a 2 cm error at 0.1 instead of 0.02, so the network can resolve cm-level errors)
    obs_pos_scale: float = 1.0
    obs_vel_scale: float = 1.0
    # tracking cost: sum w * charb(err / s) (the cone-obstacle tracker's weights) + command smoothness
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
    s_wxy: float = 3.0
    w_wxy: float = 0.02
    s_wz: float = 0.5
    w_wz: float = 1.0
    w_thrust: float = 0.01
    s_du_a: float = 1.0           # command change per step [m/s^2]
    s_du_w: float = 0.3           # [rad/s]
    w_du: float = 0.05
    delta: float = 0.1
    floor: float = 50.0
    # losing the reference: end the episode, last reward = discounted floor of the remaining steps
    lost_pos_error: float = 1.0
    lost_tilt_deg: float = 60.0
    terminal_gamma: float = 0.99

    def as_dict(self):
        return asdict(self)

    @property
    def action_low(self):
        return (self.a_min, -self.omega_xy_max, -self.omega_xy_max, -self.omega_z_max)

    @property
    def action_high(self):
        return (self.a_max, self.omega_xy_max, self.omega_xy_max, self.omega_z_max)

    @property
    def policy_low(self):
        if self.policy_mode == "residual":
            return (-self.res_a, -self.res_w_xy, -self.res_w_xy, -self.res_w_z)
        return self.action_low

    @property
    def policy_high(self):
        if self.policy_mode == "residual":
            return (self.res_a, self.res_w_xy, self.res_w_xy, self.res_w_z)
        return self.action_high

    def __post_init__(self):
        if self.policy_mode not in ("residual", "direct"):
            raise ValueError(f"policy_mode must be 'residual' or 'direct', got {self.policy_mode!r}")


def load_reference(path: str):
    p = Path(path)
    if not p.exists() and (_ROOT / p).exists():
        p = _ROOT / p
    ref = np.load(p)
    out = {"states": np.asarray(ref["states"], np.float64), "omega_cmd": np.asarray(ref["omega_cmd"], np.float64),
           "a_cmd": np.asarray(ref["a_cmd"], np.float64), "t": np.asarray(ref["t"], np.float64)}
    out["path"] = str(p)
    return out


# ---------------------------------------------------------------------- numpy twin for deployment
def _rotmat_np(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def base_action_np(meas_p, meas_v, meas_q, k, ref, d_hat, cfg: SlowTrackEnvConfig) -> np.ndarray:
    """The env's base controller (``base_action``) in plain numpy, for the flight runtime."""
    states = ref["states"]
    i = int(np.clip(k, 0, len(states) - 1))
    rs = states[i]
    g = cfg.gravity
    q = np.asarray(meas_q, np.float64) / np.linalg.norm(meas_q)
    a_ref = _rotmat_np(rs[6:10] / np.linalg.norm(rs[6:10]))[:, 2] * ref["a_cmd"][i] - np.array([0.0, 0.0, g])
    acc = a_ref - cfg.base_kp * (np.asarray(meas_p) - rs[0:3]) - cfg.base_kd * (np.asarray(meas_v) - rs[3:6]) - np.asarray(d_hat)
    f = acc + np.array([0.0, 0.0, g])
    R = _rotmat_np(q)
    zd = f / max(np.linalg.norm(f), 1e-6)
    wb = R.T @ np.cross(R[:, 2], zd) * cfg.base_katt
    yaw = np.arctan2(R[1, 0], R[0, 0])
    qr = rs[6:10] / np.linalg.norm(rs[6:10])
    yaw_ref = np.arctan2(2.0 * (qr[0] * qr[3] + qr[1] * qr[2]), 1.0 - 2.0 * (qr[2] ** 2 + qr[3] ** 2))
    dyaw = np.arctan2(np.sin(yaw - yaw_ref), np.cos(yaw - yaw_ref))
    u = np.array([np.dot(f, R[:, 2]), wb[0], wb[1], -cfg.base_kyaw * dyaw])
    return np.clip(u, np.asarray(cfg.action_low), np.asarray(cfg.action_high))


def observation_np(meas_p, meas_v, meas_q, k, ref, d_hat, u_prev, cfg: SlowTrackEnvConfig) -> np.ndarray:
    """The env's observation in plain numpy (one state), for the flight runtime. ``ref`` is ``load_reference``'s
    dict; positions/velocities in the task frame (z up), quaternion [w, x, y, z] body->task frame."""
    states, om, ac = ref["states"], ref["omega_cmd"], ref["a_cmd"]
    n = len(states)
    i = int(np.clip(k, 0, n - 1))
    rs = states[i]
    p1 = states[int(np.clip(k + cfg.preview_steps_1, 0, n - 1)), 0:3]
    p2 = states[int(np.clip(k + cfg.preview_steps_2, 0, n - 1)), 0:3]
    q = np.asarray(meas_q, np.float64) / np.linalg.norm(meas_q)
    q = -q if q[0] < 0 else q
    qr = rs[6:10] / np.linalg.norm(rs[6:10])
    qr = -qr if qr[0] < 0 else qr
    low, high = np.asarray(cfg.action_low), np.asarray(cfg.action_high)
    mid, half = 0.5 * (low + high), 0.5 * (high - low)
    p = np.asarray(meas_p, np.float64)
    parts = [(p - rs[0:3]) / cfg.obs_pos_scale, (np.asarray(meas_v) - rs[3:6]) / cfg.obs_vel_scale, q, qr, rs[3:6], om[i], [ac[i] / cfg.gravity - 1.0],
             p1 - p, p2 - p, np.asarray(d_hat), (np.asarray(u_prev) - mid) / half]
    if cfg.policy_mode == "residual":
        parts.append((base_action_np(meas_p, meas_v, q, k, ref, d_hat, cfg) - mid) / half)
    return np.concatenate(parts).astype(np.float32)


def command_np(policy_out, meas_p, meas_v, meas_q, k, ref, d_hat, cfg: SlowTrackEnvConfig) -> np.ndarray:
    """Command sent to PX4 from the actor's (mean) output: base + correction in residual mode."""
    a = np.clip(np.asarray(policy_out, np.float64), cfg.policy_low, cfg.policy_high)
    if cfg.policy_mode == "residual":
        a = base_action_np(meas_p, meas_v, meas_q, k, ref, d_hat, cfg) + a
    return np.clip(a, cfg.action_low, cfg.action_high)


__all__ = ["OBS_DIM", "OBS_LAYOUT", "RESIDUAL_EXTRA", "SlowTrackEnvConfig", "base_action_np", "command_np",
           "load_reference", "obs_layout", "observation_np"]
