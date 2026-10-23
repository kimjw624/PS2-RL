"""Straight-line pursuit controller for the strike task (the nominal of the filter evaluations and the base of the
residual striker).

Velocity loop toward the target: v_des = cruise * (p_t - p) / |p_t - p| + v_t, a = kv (v_des - v) with the horizontal
part limited by the tilt; the thrust axis is steered to the commanded acceleration (body rates from the axis error) and
the yaw to zero. It ignores obstacles and chasers.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ps2rl.envs.strike_scene import StrikeModel
from ps2rl.utils.quaternion import rotation_matrix_from_quaternion


def make_pursuit(m: StrikeModel, cruise: float, kv: float = 2.5, katt: float = 10.0, kyaw: float = 3.0,
                 tilt_max_deg: float = 50.0):
    """ctrl(x, target_p, target_v) -> u (thrust, body rates) for one vehicle."""
    g = m.cfg.gravity
    amax_xy = g * np.tan(np.radians(tilt_max_deg))

    def ctrl(x, tp, tv):
        p, v, q = x[0:3], x[3:6], x[6:10]
        to_t = tp - p
        v_des = cruise * to_t / jnp.maximum(jnp.linalg.norm(to_t), 1e-6) + tv
        a = kv * (v_des - v)
        axy = a[0:2] * jnp.minimum(1.0, amax_xy / (jnp.linalg.norm(a[0:2]) + 1e-9))
        a_des = jnp.concatenate([axy, a[2:3] + g])
        R = rotation_matrix_from_quaternion(q)
        zb = R[:, 2]
        zd = a_des / jnp.maximum(jnp.linalg.norm(a_des), 1e-6)
        w = katt * (R.T @ jnp.cross(zb, zd))
        w = w.at[2].add(-kyaw * jnp.arctan2(R[1, 0], R[0, 0]))
        return jnp.clip(jnp.concatenate([jnp.dot(a_des, zb)[None], w]), m.low, m.high)

    return ctrl


__all__ = ["make_pursuit"]
