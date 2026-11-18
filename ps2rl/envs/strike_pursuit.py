"""Straight-line pursuit controller for the strike task (the nominal of the filter evaluations and the base of the
residual striker).

Velocity loop toward the target: v_des = cruise * (p_t - p) / |p_t - p| + v_t, a = kv (v_des - v) with the horizontal
part limited by the tilt; the thrust axis is steered to the commanded acceleration (body rates from the axis error) and
the yaw to zero. It ignores obstacles and chasers.

Option ``detour_radius`` > 0 (off by default): when a chaser lies ahead, inside the cone of directions that pass within
detour_radius of it, the horizontal direction to the target is turned to the nearer edge of that cone (inside the
radius: tangent to it, on the target's side), so the pursuit goes around the chaser instead of straight at it.

Option ``detour_search`` (off by default): a look-ahead heading search that also sees obstacles and the arena walls.
Candidate horizontal headings are the target direction turned by 0, +-step, +-2 step, ... up to +-max_turn; a heading is
free when its ray (length min(lookahead, horizontal distance to the target)) keeps detour_margin from every obstacle,
detour_wall_margin from the walls and, with detour_radius > 0, does not come closer to a chaser than min(detour_radius,
its present distance). The free heading with the smallest turn is flown; between +theta and -theta the side the
drone already moves to wins. With detour_heading_weight w > 0 the free heading minimises |turn| + w |angle to the
present velocity| (the velocity term fades out below 2 m/s), as in VFH+: once the drone goes around one side it keeps
to it instead of switching sides when the free set changes. If none is free, the heading with the most clearance.
With detour_final_dist > 0 the search is skipped within that horizontal distance of the target (straight final
approach; a target next to a wall or an obstacle would otherwise never have a free ray).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.envs.strike_scene import StrikeModel
from ps2rl.utils.quaternion import rotation_matrix_from_quaternion


def detour_direction(p, tp, chaser_p, chaser_on, radius: float):
    """Unit direction from p toward tp, turned around the chasers (horizontal plane) that block it (see the module)."""
    to_t = tp - p
    dirn = to_t / jnp.maximum(jnp.linalg.norm(to_t), 1e-6)
    h = dirn[0:2]
    d_t = jnp.linalg.norm(to_t[0:2])
    for c in range(chaser_p.shape[0]):
        rel = chaser_p[c, 0:2] - p[0:2]
        d_c = jnp.linalg.norm(rel) + 1e-9
        e_c = rel / d_c
        half = jnp.arcsin(jnp.clip(radius / d_c, 0.0, 1.0))           # cone half-angle (pi/2 inside the radius)
        hn = h / (jnp.linalg.norm(h) + 1e-9)
        cosang = jnp.clip(hn @ e_c, -1.0, 1.0)
        side = jnp.sign(e_c[0] * hn[1] - e_c[1] * hn[0])               # target direction left (+) / right (-) of the chaser
        side = jnp.where(side == 0.0, 1.0, side)
        ca, sa = jnp.cos(half), jnp.sin(half) * side
        edge = jnp.stack([ca * e_c[0] - sa * e_c[1], sa * e_c[0] + ca * e_c[1]])
        blocked = (chaser_on[c] > 0.5) & (jnp.arccos(cosang) < half) & (d_c < d_t)
        h = jnp.where(blocked, edge * jnp.linalg.norm(h), h)
    return jnp.concatenate([h, dirn[2:3]]) / jnp.maximum(jnp.linalg.norm(jnp.concatenate([h, dirn[2:3]])), 1e-6)


def search_direction(p, v, tp, chaser_p, chaser_on, obst, arena, radius: float, lookahead: float = 6.0,
                     margin: float = 1.0, wall_margin: float = 1.0, max_turn_deg: float = 120.0, step_deg: float = 15.0,
                     n_samples: int = 8, heading_weight: float = 0.0, final_dist: float = 0.0):
    """Unit direction from p toward tp, turned by the smallest angle whose look-ahead ray is free (see the module).

    obst (A, 7) rounded rectangles (cx, cy, bx, by, yaw, r, present) or None; arena (x_min, x_max, y_half, ...) or None;
    chasers only with radius > 0."""
    to_t = tp - p
    dirn = to_t / jnp.maximum(jnp.linalg.norm(to_t), 1e-6)
    d_t = jnp.linalg.norm(to_t[0:2])
    h0 = to_t[0:2] / jnp.maximum(d_t, 1e-6)
    k = int(round(max_turn_deg / step_deg))
    ang = np.radians(step_deg) * np.asarray([0.0] + [sg * i for i in range(1, k + 1) for sg in (1.0, -1.0)])
    ca, sa = jnp.cos(ang), jnp.sin(ang)
    dirs = jnp.stack([ca * h0[0] - sa * h0[1], sa * h0[0] + ca * h0[1]], axis=-1)            # (K, 2)
    frac = jnp.asarray(np.arange(1, n_samples + 1) / n_samples)
    L = jnp.minimum(lookahead, d_t)
    pts = p[None, None, 0:2] + L * frac[None, :, None] * dirs[:, None, :]                      # (K, S, 2)
    clear = jnp.full((len(ang),), 1e3)
    if obst is not None:
        sd = jax.vmap(lambda o: jnp.where(o[6] > 0.5, rrect_sdf_xy(pts - o[0:2], o[2], o[3], o[4], o[5]), 1e3))(obst)
        clear = jnp.minimum(clear, jnp.min(sd, axis=(0, 2)) - margin)
    if arena is not None:
        gaps = jnp.stack([pts[..., 0] - arena[0], arena[1] - pts[..., 0], pts[..., 1] + arena[2], arena[2] - pts[..., 1]])
        clear = jnp.minimum(clear, jnp.min(gaps, axis=(0, 2)) - wall_margin)
    if radius > 0.0 and chaser_p is not None:
        for c in range(chaser_p.shape[0]):
            rel = chaser_p[c, 0:2] - p[0:2]
            d_now = jnp.linalg.norm(rel)
            t_c = jnp.clip(dirs @ rel, 0.0, L)                                  # closest point of each ray (exact)
            d_ray = jnp.linalg.norm(rel[None, :] - t_c[:, None] * dirs, axis=-1)
            clear = jnp.where(chaser_on[c] > 0.5, jnp.minimum(clear, d_ray - jnp.minimum(radius, d_now) + 1e-3), clear)
    lat = jnp.sign(h0[0] * v[1] - h0[1] * v[0])                         # the drone moves to the left (+) / right (-)
    cost = jnp.abs(ang)
    if heading_weight > 0.0:
        sp = jnp.linalg.norm(v[0:2])
        vh = v[0:2] / jnp.maximum(sp, 1e-6)
        cost = cost + heading_weight * jnp.clip(sp / 2.0, 0.0, 1.0) * jnp.arccos(jnp.clip(dirs @ vh, -1.0, 1.0))
    key = jnp.where(clear >= 0.0, -cost + 1e-3 * jnp.sign(ang) * lat, -1e4 + clear)
    h = dirs[jnp.argmax(key)] * jnp.linalg.norm(dirn[0:2])
    out = jnp.concatenate([h, dirn[2:3]])
    out = out / jnp.maximum(jnp.linalg.norm(out), 1e-6)
    return jnp.where(d_t <= final_dist, dirn, out) if final_dist > 0.0 else out


def rrect_sdf_xy(p_rel, bx, by, yaw, rr):
    """Signed distance of points (..., 2) relative to the centre to a rounded rectangle (strike_scene.rrect_sdf)."""
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    lx = c * p_rel[..., 0] + s * p_rel[..., 1]
    ly = -s * p_rel[..., 0] + c * p_rel[..., 1]
    qx, qy = jnp.abs(lx) - bx, jnp.abs(ly) - by
    out = jnp.sqrt(jnp.maximum(qx, 0.0) ** 2 + jnp.maximum(qy, 0.0) ** 2 + 1e-12)
    return out + jnp.minimum(jnp.maximum(qx, qy), 0.0) - rr


def make_pursuit(m: StrikeModel, cruise: float, kv: float = 2.5, katt: float = 10.0, kyaw: float = 3.0,
                 tilt_max_deg: float = 50.0, detour_radius: float = 0.0, detour_search: bool = False,
                 detour_lookahead: float = 6.0, detour_margin: float = 1.0, detour_wall_margin: float = 1.0,
                 detour_max_turn_deg: float = 120.0, detour_step_deg: float = 15.0, detour_heading_weight: float = 0.0,
                 detour_final_dist: float = 0.0, arena=None):
    """ctrl(x, target_p, target_v[, chaser_p, chaser_on, obst]) -> u (thrust, body rates) for one vehicle."""
    g = m.cfg.gravity
    amax_xy = g * np.tan(np.radians(tilt_max_deg))

    def ctrl(x, tp, tv, chaser_p=None, chaser_on=None, obst=None):
        p, v, q = x[0:3], x[3:6], x[6:10]
        to_t = tp - p
        if detour_search:
            v_des = cruise * search_direction(p, v, tp, chaser_p, chaser_on, obst, arena, detour_radius, detour_lookahead,
                                              detour_margin, detour_wall_margin, detour_max_turn_deg, detour_step_deg,
                                              heading_weight=detour_heading_weight, final_dist=detour_final_dist) + tv
        elif detour_radius > 0.0 and chaser_p is not None:
            v_des = cruise * detour_direction(p, tp, chaser_p, chaser_on, detour_radius) + tv
        else:
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


__all__ = ["detour_direction", "make_pursuit", "search_direction"]
