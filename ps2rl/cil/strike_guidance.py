"""Guidance-law chaser constraint and evasion reflex for the strike filter (``make_filter(chaser_model="guidance")``).

Engagement variables of the drone (p, v) against a chaser at c with speed <= v_c: range R = |p - c|, line-of-sight unit
vector e = (p - c) / R, opening speed v_r = e'v. The chaser closes the range at most at v_c (heading straight at the
drone, which a pure pursuer does), so the worst-case closing speed is s = v_c - v_r. If the drone can raise v_r at a_e
after a reaction lag delta, it can still escape when

    h_c = R - (r_c + r_d) - s+ delta - s+^2 / (2 a_e) >= 0,        s+ = max(s, 0)

(at range R the required opening speed is v_c - sqrt(2 a_e (R - r_c - r_d)) without the lag). In open space {h_c >= 0}
is invariant under "accelerate away along e": dh_c/dt >= -s + a_e (delta + s / a_e) = a_e delta >= 0; the line-of-sight
rotation terms are second order (<= v_c^2 / 4R) and covered by crediting a_e below the true maximum.

Evasion reflex (part of the backup inside the filter's rollout): below ``trigger_off`` the backup's command is blended
with an escape command - a velocity loop flying away from the most critical chaser at the escape speed, horizontal
direction e deflected by obstacles and arena walls near the hover point, at least ``z_min_escape`` above the floor - with
a weight growing from 0 at trigger_off to 1 at ``trigger_on``. Above trigger_off the ordinary backup (hover LQR /
learned backup) flies alone. In the rollout the chasers are
propagated as pure pursuers at v_c toward the drone. The escape speed (``escape_speed_ratio`` v_c) is above the base
set's speed range, so a rollout may end mid-escape: the filter then asks that the straight braking point at a_evade is
clear (``stop_point``) instead of the hover ellipsoid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.envs.strike_scene import StrikeModel, anchor, chaser_pursuit_step, obstacle_distances
from ps2rl.utils.quaternion import rotation_matrix_from_quaternion

Array = jax.Array


@dataclass(frozen=True)
class GuidanceConfig:
    a_evade: float = 5.0            # [m/s^2] evasive acceleration credited to the drone (well below its maximum)
    delay: float = 0.2              # [s] reaction lag (attitude change)
    trigger_on: float = 0.5         # [m of h_c] reflex at full escape speed below this
    trigger_off: float = 2.0        # [m of h_c] reflex off above this (ordinary backup)
    escape_speed_ratio: float = 2.0 # escape speed = ratio * v_c (the reflex flies slower when h_c allows)
    # extra escape variants (speed ratio, angle [deg] from the line of sight away from the chaser), each tried as one more
    # backup: the filter applies the largest certified blend over all of them (union of their recoverable sets)
    escape_variants: tuple = ()
    horizon: float = 3.0            # [s] rollout length of the filter check (must contain an evasion)
    lqr_alternative: bool = True    # also try the hover LQR as the backup outside B (handles arena walls / ceiling, which
                                    # the learned backup never saw); the largest certified blend of either is applied
    learned_alone: bool = False     # first variant: the learned backup by itself (no reflex), certified by its Phase-I
                                    # criterion - for a backup trained with moving chasers (strike chaser_set "guidance")
    committed_prediction: bool = False  # the committed plan's controls (backup input, reflex, hover switch) see the chasers
                                    # where its certificate predicted them, so its continuation replays the certified plan
                                    # exactly; safety is still checked against the chasers as they are. Off: its controls
                                    # see the current chasers (the plan can change when they move differently)
    prefer_lqr_on_ties: bool = False  # equal blends certified with both backups: apply the hover LQR's (else the learned)
    slide_along: bool = False       # near an obstacle / wall the escape direction loses its component into it
    best_effort_fallback: bool = False  # no certificate at all: apply the candidate (or the committed plan) whose
                                    # backup rollout keeps the largest worst-case margin, instead of the committed plan
    fresh_obstacle_margin: float = 0.15  # [m] new plans also keep this much extra clearance from obstacles / walls (the
                                    # committed plan only the contact distance): slack for the reflex re-aiming
    fresh_margin: float = 0.2       # [m of h_c] new plans keep h_c >= this; the committed plan only h_c >= 0 (slack for the
                                    # chasers not moving exactly as the pure-pursuit prediction)
    k_vel: float = 2.0              # [1/s] velocity-loop gain of the reflex
    k_att: float = 10.0             # attitude gain (body rates from the thrust-axis error)
    k_yaw: float = 3.0
    tilt_max_deg: float = 45.0      # tilt limit of the reflex
    push_dist: float = 1.5          # [m] beyond the hover clearance (rho_xy + r_d): closer obstacles / walls deflect the escape
    z_min_escape: float = 1.0       # [m above the floor] the reflex climbs to at least this
    ceiling_margin: float = 3.0     # [m] the reflex descends when it is closer than this to the arena ceiling (z_max)

    @classmethod
    def from_dict(cls, d: dict | None) -> "GuidanceConfig":
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown guidance keys {sorted(unknown)}")
        if "escape_variants" in d:
            d["escape_variants"] = tuple((float(a), float(b)) for a, b in d["escape_variants"])
        return cls(**d)

    def as_dict(self) -> dict:
        return asdict(self)


def level_speed_in_base_set(m: StrikeModel) -> float:
    """Largest horizontal speed with level attitude inside the base set (V_min <= c_B)."""
    S = np.asarray(m.S)
    return float(np.sqrt(m.cfg.c_B / max(S[0, 0], S[1, 1])))


def escape_speed(m: StrikeModel, g: GuidanceConfig) -> float:
    if g.escape_speed_ratio <= 1.0:
        raise ValueError(f"escape_speed_ratio {g.escape_speed_ratio} must exceed 1 (outrun the chaser)")
    return float(g.escape_speed_ratio * m.cfg.chaser_speed)


def stop_point(g: GuidanceConfig, p: Array, v: Array) -> Array:
    """Where the drone stops braking horizontally at a_evade (straight line)."""
    vxy = v[0:2]
    return p[0:2] + vxy * jnp.linalg.norm(vxy) / (2.0 * g.a_evade)


def h_chasers(m: StrikeModel, g: GuidanceConfig, p: Array, v: Array, chaser_p: Array, chaser_on: Array) -> Array:
    """(C,) escape margin h_c of each chaser (absent chasers -> large)."""
    cfg = m.cfg
    rel = p[None, :] - chaser_p
    R = jnp.sqrt(jnp.sum(rel ** 2, axis=1) + 1e-12)
    e = rel / R[:, None]
    s = jnp.maximum(cfg.chaser_speed - e @ v, 0.0)
    h = R - (cfg.chaser_radius + cfg.r_drone) - s * g.delay - s ** 2 / (2.0 * g.a_evade)
    return jnp.where(chaser_on > 0.5, h, 1e9)


def pursue(m: StrikeModel, chaser_p: Array, p_drone: Array) -> Array:
    """One step of pure pursuit at v_c toward the drone (the worst case for the range), above the floor."""
    return chaser_pursuit_step(m.cfg, chaser_p, p_drone)


def wall_distances(arena: tuple, p_xy: Array) -> Array:
    """Distances to the four arena walls (x_min, x_max, y_half): (4,)."""
    x_min, x_max, y_half = arena[0], arena[1], arena[2]
    return jnp.stack([p_xy[0] - x_min, x_max - p_xy[0], p_xy[1] + y_half, y_half - p_xy[1]])


def velocity_track(m: StrikeModel, g: GuidanceConfig, x: Array, v_des: Array) -> Array:
    """Thrust + body rates that track a velocity (thrust-axis attitude loop, yaw to zero)."""
    cfg = m.cfg
    v, q = x[3:6], x[6:10]
    a = g.k_vel * (v_des - v)
    amax = cfg.gravity * np.tan(np.radians(g.tilt_max_deg))
    axy = a[0:2] * jnp.minimum(1.0, amax / (jnp.linalg.norm(a[0:2]) + 1e-9))
    a_des = jnp.concatenate([axy, a[2:3] + cfg.gravity])
    Rm = rotation_matrix_from_quaternion(q)
    zb = Rm[:, 2]
    zd = a_des / jnp.maximum(jnp.linalg.norm(a_des), 1e-6)
    w = g.k_att * (Rm.T @ jnp.cross(zb, zd))
    w = w.at[2].add(-g.k_yaw * jnp.arctan2(Rm[1, 0], Rm[0, 0]))
    return jnp.clip(jnp.concatenate([jnp.dot(a_des, zb)[None], w]), m.low, m.high)


def make_reflex(m: StrikeModel, g: GuidanceConfig, arena: tuple, speed_ratio: float | None = None, angle_deg: float = 0.0):
    """reflex(x, chaser_p, chaser_on, obst) -> (u, sigma): the escape command (escape speed, deflected by obstacles and
    walls) and its weight sigma in [0, 1] (0 above trigger_off, 1 below trigger_on); the filter blends
    u = (1 - sigma) u_backup + sigma u_escape, so the backup keeps handling obstacles while the threat is mild.
    speed_ratio / angle_deg: an escape variant (default: escape_speed_ratio, straight away from the chaser)."""
    cfg = m.cfg
    v_esc = escape_speed(m, g) if speed_ratio is None else float(speed_ratio * cfg.chaser_speed)
    ca, sa = float(np.cos(np.radians(angle_deg))), float(np.sin(np.radians(angle_deg)))
    clear = float(m.lqr.rho_xy + cfg.r_drone)

    def reflex(x, chaser_p, chaser_on, obst):
        p, v = x[0:3], x[3:6]
        h = h_chasers(m, g, p, v, chaser_p, chaser_on)
        j = jnp.argmin(h)
        hmin = h[j]
        sigma = jnp.clip((g.trigger_off - hmin) / (g.trigger_off - g.trigger_on), 0.0, 1.0)
        away = p[0:2] - chaser_p[j, 0:2]
        away = away / (jnp.linalg.norm(away) + 1e-6)
        away = jnp.stack([ca * away[0] - sa * away[1], sa * away[0] + ca * away[1]])   # escape variant's direction
        q_a = anchor(m, x)[0:2]                     # keep the hover point clear: push measured there
        d_o, grad_o = jax.value_and_grad(lambda q: jnp.min(obstacle_distances(q, obst)))(q_a)
        push_o = jnp.clip((g.push_dist - (d_o - clear)) / g.push_dist, 0.0, 1.0)
        d_w = wall_distances(arena, q_a)
        n_w = jnp.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        push_w = jnp.clip((g.push_dist - (d_w - clear)) / g.push_dist, 0.0, 1.0)
        # near an obstacle / wall: drop the part of the escape direction that points into it (slide along), then push out
        n_o = grad_o / (jnp.linalg.norm(grad_o) + 1e-9)
        if g.slide_along:
            away = away - jnp.where(push_o > 0.0, jnp.minimum(away @ n_o, 0.0), 0.0) * n_o
            into_w = jnp.minimum(n_w @ away, 0.0) * (push_w > 0.0)
            away = away - (into_w[:, None] * n_w).sum(axis=0)
        want = away + 2.0 * push_o * n_o + 2.0 * (push_w[:, None] * n_w).sum(axis=0)
        dirn = want / (jnp.linalg.norm(want) + 1e-6)
        z_ref = jnp.clip(p[2], cfg.z_floor + g.z_min_escape, arena[3] - g.ceiling_margin)
        v_des = jnp.concatenate([v_esc * dirn, (1.5 * (z_ref - p[2]))[None]])
        return velocity_track(m, g, x, v_des), sigma

    return reflex


__all__ = ["GuidanceConfig", "escape_speed", "stop_point", "h_chasers", "level_speed_in_base_set", "make_reflex", "pursue",
           "velocity_track", "wall_distances"]
