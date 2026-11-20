"""Nominal controller: go-to-goal velocity loop with a look-ahead heading search around the known cylinders (numpy,
batched). The strike task's pursuit nominal (``ps2rl.envs.strike_pursuit``, ``detour_search``) for a fixed goal.

Velocity command: horizontal speed ``cruise`` along the searched heading (slowing down within ``arrive_dist`` of the
goal), vertical ``kz (z_goal - z)``. Heading search: candidate headings are the goal direction turned by 0, +-step,
+-2 step, ... up to +-max_turn; a heading is free when its ray (length min(lookahead, distance to the goal)) keeps
``margin`` from every known cylinder's surface. The free heading with the smallest |turn| + heading_weight x (angle to
the present velocity) is flown (the second term keeps the side once the drone goes around an obstacle); if none is free,
the heading with the most clearance. Attitude: the thrust axis is steered to the commanded acceleration (body rates from
the axis error, the tilt limited to tilt_max_deg) and the yaw to zero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ps2rl.online.model import CylModel, body_z


@dataclass
class NominalConfig:
    cruise: float = 1.0           # [m/s]
    arrive_dist: float = 1.5      # slow down linearly within this horizontal distance of the goal [m]
    kv: float = 1.5               # velocity loop [1/s]
    kz: float = 1.0               # height loop [1/s]
    katt: float = 4.0             # thrust-axis loop [1/s]
    kyaw: float = 1.0
    tilt_max_deg: float = 20.0
    search: bool = True
    lookahead: float = 4.0        # [m]
    margin: float = 1.0           # ray clearance from the cylinders' surfaces [m]
    max_turn_deg: float = 90.0
    step_deg: float = 10.0
    n_samples: int = 8
    heading_weight: float = 0.5

    def as_dict(self) -> dict:
        return asdict(self)


def rotation(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)], -1),
        np.stack([2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)], -1),
        np.stack([2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)], -1)], -2)


def velocity_command(m: CylModel, x: np.ndarray, v_des: np.ndarray, kv: float, katt: float, kyaw: float,
                     tilt_max_deg: float, d: np.ndarray | None = None) -> np.ndarray:
    """Thrust + body rates tracking a velocity (thrust-axis attitude loop, yaw to zero), clipped to the box; d: a push
    estimate, cancelled by feed-forward."""
    a = kv * (v_des - x[..., 3:6])
    if d is not None:
        a = a - d
    amax = m.gravity * np.tan(np.radians(tilt_max_deg))
    nxy = np.sqrt(np.sum(a[..., 0:2] ** 2, axis=-1, keepdims=True)) + 1e-9
    axy = a[..., 0:2] * np.minimum(1.0, amax / nxy)
    a_des = np.concatenate([axy, a[..., 2:3] + m.gravity], axis=-1)
    R = rotation(x[..., 6:10])
    zb = R[..., :, 2]
    zd = a_des / np.maximum(np.sqrt(np.sum(a_des ** 2, axis=-1, keepdims=True)), 1e-6)
    w = katt * np.einsum("...ji,...j->...i", R, np.cross(zb, zd))
    w[..., 2] -= kyaw * np.arctan2(R[..., 1, 0], R[..., 0, 0])
    thrust = np.sum(a_des * zb, axis=-1, keepdims=True)
    return np.clip(np.concatenate([thrust, w], axis=-1), m.low, m.high)


class Nominal:
    def __init__(self, m: CylModel, cfg: NominalConfig | None = None) -> None:
        self.m = m
        self.cfg = cfg or NominalConfig()
        c = self.cfg
        k = int(round(c.max_turn_deg / c.step_deg))
        self.ang = np.radians(c.step_deg) * np.asarray([0.0] + [s * i for i in range(1, k + 1) for s in (1.0, -1.0)])
        self.frac = np.arange(1, c.n_samples + 1) / c.n_samples

    def heading(self, x: np.ndarray, goal: np.ndarray, obst: np.ndarray) -> np.ndarray:
        """(..., 2) unit horizontal direction to fly (see the module)."""
        c = self.cfg
        to_g = goal[..., 0:2] - x[..., 0:2]
        d_g = np.sqrt(np.sum(to_g ** 2, axis=-1)) + 1e-9
        h0 = to_g / d_g[..., None]
        if not c.search or obst.shape[-2] == 0:
            return h0
        ca, sa = np.cos(self.ang), np.sin(self.ang)
        dirs = np.stack([ca * h0[..., None, 0] - sa * h0[..., None, 1], sa * h0[..., None, 0] + ca * h0[..., None, 1]], -1)
        L = np.minimum(c.lookahead, d_g)
        pts = x[..., None, None, 0:2] + (L[..., None, None, None] * self.frac[:, None]) * dirs[..., :, None, :]  # (..., H, S, 2)
        oc = obst[..., None, None, :, :]                                                                         # (..., 1, 1, M, 4)
        sd = np.sqrt(np.sum((pts[..., None, :] - oc[..., 0:2]) ** 2, axis=-1)) - oc[..., 2]
        sd = np.where(oc[..., 3] > 0.5, sd, 1e3)
        clear = sd.min(axis=(-1, -2)) - c.margin                                                                 # (..., H)
        v = x[..., 3:5]
        lat = np.sign(h0[..., 0] * v[..., 1] - h0[..., 1] * v[..., 0])
        cost = np.abs(self.ang)
        if c.heading_weight > 0.0:
            sp = np.sqrt(np.sum(v ** 2, axis=-1))
            vh = v / np.maximum(sp, 1e-6)[..., None]
            cosv = np.clip(np.sum(dirs * vh[..., None, :], axis=-1), -1.0, 1.0)
            cost = cost + c.heading_weight * np.clip(sp / max(c.cruise, 1e-6), 0.0, 1.0)[..., None] * np.arccos(cosv)
        key = np.where(clear >= 0.0, -cost + 1e-3 * np.sign(self.ang) * lat[..., None], -1e4 + clear)
        i = np.argmax(key, axis=-1)
        return np.take_along_axis(dirs, i[..., None, None], axis=-2)[..., 0, :]

    def __call__(self, x: np.ndarray, goal: np.ndarray, obst: np.ndarray, d: np.ndarray | None = None) -> np.ndarray:
        c = self.cfg
        h = self.heading(x, goal, obst)
        d_g = np.sqrt(np.sum((goal[..., 0:2] - x[..., 0:2]) ** 2, axis=-1))
        speed = c.cruise * np.clip(d_g / max(c.arrive_dist, 1e-6), 0.0, 1.0)
        v_des = np.concatenate([h * speed[..., None], (c.kz * (goal[..., 2] - x[..., 2]))[..., None]], axis=-1)
        return velocity_command(self.m, x, v_des, c.kv, c.katt, c.kyaw, c.tilt_max_deg, d)


__all__ = ["Nominal", "NominalConfig", "rotation", "velocity_command"]
