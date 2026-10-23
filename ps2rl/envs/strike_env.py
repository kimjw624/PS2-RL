"""Strike environment (Phase II): fly through an arena of tall obstacles, past pursuing chasers, and hit a moving target.

Arena: x in [0, arena_x], |y| <= arena_y_half, floor z_floor (from StrikeConfig), no ceiling (z above z_max = out).
Scenes come from a pool generated once with numpy (obstacle spacing, clearance around start and target, and a route
with ``route_clearance`` from start to target checked on a grid); every episode then draws its own perturbations in JAX:
start altitude / speed / heading / tilt, the target's drift velocity, the chasers' start velocity.

Target: starts at x in [target_x_min, target_x_max], drifts at a constant speed in a random horizontal direction and
reflects at the edges of its zone (x >= target_zone_x_min, inside the arena). Strike: distance < strike_radius.
Chasers: pursue the drone at <= chaser_speed with acceleration <= chaser_accel, pushed away from obstacles within
chaser_repulse_dist; they fly above the floor.

Episode ends at a strike (success), a crash (floor, obstacle within r_drone, chaser within r_c + r_drone), leaving the
arena, or the time limit. Reward per step:
    w_progress * (d_prev - d)  -  w_time  -  w_action * mean(((u - u_hover) / scale)^2)  -  envelope
    + strike_bonus [strike]  -  crash_penalty [crash or out of the arena]
(envelope: optional speed / tilt excess penalty, w_speed_excess / w_tilt_excess)
with d the distance to the target (closing on the target fast = progress per step; an early strike = less time cost).

Observation (StrikeConfig.obs_dim + 14): the backup's observation of the local scene (``strike_scene.make_obs``: the
n_obstacle_slots nearest obstacles and the chasers as balls), then target offset / 10 m, target velocity, chaser
velocities (zero for absent chasers) and the arena position (x / (arena_x / 2) - 1, y / arena_y_half). With
obs_hide_scene the obstacle / chaser parts are zero (goal-reaching striker; obstacles and chasers left to the filter).

Residual striker (residual_cruise > 0, task file strike_residual.json): the action is a correction du on the
straight-line pursuit controller; the effort term then penalises du (the anchor to pursuit).

Obstacle-free goal reaching (Phase II stage 1, task file strike_free.json): obstacles_max 0, chasers_active_max 0,
start x / target spread over the arena (start_x_max, start_target_min_dist, target_zone_x_min).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import time
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy import ndimage

from ps2rl.envs.strike_pursuit import make_pursuit
from ps2rl.envs.strike_scene import (
    EGO_FEATS,
    OBST_FIELDS,
    StrikeModel,
    arena_walls,
    join_scene,
    make_obs,
    obstacle_distances,
    plant_step,
)

Array = jax.Array


@dataclass(frozen=True)
class StrikeEnvConfig:
    # arena
    arena_x: float = 40.0
    arena_y_half: float = 10.0
    z_max: float = 10.0
    # start (perturbed per episode)
    start_x: float = 2.0
    start_x_max: float = -1.0            # > start_x: start x drawn in [start_x, start_x_max] (else fixed at start_x)
    start_y_half: float = 3.0
    start_target_min_dist: float = 0.0   # resample the target until it is at least this far from the start [m]
    start_z_min: float = 2.0
    start_z_max: float = 4.0
    start_speed_min: float = 4.0
    start_speed_max: float = 6.0
    start_heading_deg: float = 15.0
    start_tilt_deg: float = 10.0
    # target
    target_x_min: float = 32.0
    target_x_max: float = 37.0
    target_y_half: float = 5.0
    target_z_min: float = 1.6
    target_z_max: float = 2.5
    target_speed_min: float = 0.2
    target_speed_max: float = 0.8
    target_zone_x_min: float = 32.0
    strike_radius: float = 0.6
    # obstacles (tall rounded rectangles)
    obstacles_min: int = 6
    obstacles_max: int = 10
    obst_x_min: float = 7.0
    obst_x_max: float = 31.0
    cyl_fraction: float = 0.5
    cyl_radius_min: float = 0.8
    cyl_radius_max: float = 2.0
    box_half_x_min: float = 1.5
    box_half_x_max: float = 3.0
    box_half_y_min: float = 1.0
    box_half_y_max: float = 2.5
    box_yaw_deg: float = 30.0
    gap_min: float = 3.5
    wall_margin: float = 1.0
    clear_start_target: float = 4.0
    route_clearance: float = 1.5
    # chasers (chasers_max = slots; each scene draws chasers_active_min..chasers_active_max of them, -1 = chasers_max)
    chasers_max: int = 2
    chasers_active_min: int = 0
    chasers_active_max: int = -1
    chaser_side_min: float = 5.0
    chaser_side_max: float = 9.0
    chaser_along_min: float = 0.35
    chaser_along_max: float = 0.85
    chaser_accel: float = 2.0
    chaser_repulse_dist: float = 2.0
    # the chasers' actual speed = chaser_speed_scale * StrikeConfig.chaser_speed (the backup / filter keep assuming
    # chaser_speed); 0 freezes them. For tests, 1 = the task as defined.
    chaser_speed_scale: float = 1.0
    # 0 = free pursuit. > 0: guards - each chaser pursues but stays within chaser_leash [m] of its start (its post), so
    # a ball of radius r_c + r_d + chaser_leash around the post holds every point it can ever reach (with
    # chaser_leash <= v_c T that ball is inside the filter's ball R = r_c + r_d + v_c T: pass the posts to the filter).
    chaser_leash: float = 0.0
    # episode and reward
    episode_seconds: float = 10.0
    w_progress: float = 5.0
    w_time: float = 0.05
    w_action: float = 0.05
    strike_bonus: float = 50.0
    crash_penalty: float = 50.0
    # envelope penalty (0 = off): w_speed_excess * max(0, |v| - speed_soft)^2 + w_tilt_excess * max(0, tilt - tilt_soft)^2
    # [m/s, rad] per step - keeps a striker inside the speeds / tilts the Phase-I backup can recover from
    w_speed_excess: float = 0.0
    speed_soft: float = 10.0
    w_tilt_excess: float = 0.0
    tilt_soft_deg: float = 45.0
    # True: the observation keeps its layout but the obstacle and chaser features (and chaser velocities) are zero - a
    # goal-reaching striker that leaves obstacles and chasers to the safety filter
    obs_hide_scene: bool = False
    # residual striker: > 0 makes the action a residual du added to the straight-line pursuit controller at this cruise
    # speed (strike_pursuit.make_pursuit); the plant gets clip(u_pursuit + du). The policy's range is
    # +-residual_frac * action_scale (``action_bounds``) and the effort term penalises du instead of u - u_hover.
    residual_cruise: float = 0.0
    residual_frac: float = 0.25
    residual_frac_thrust: float = -1.0   # thrust channel's fraction (< 0: residual_frac)
    # True: append guidance features to the observation (19): line-of-sight direction e = (p_t - p) / d, d / 10,
    # log(d + 0.1), relative velocity e_v = v_t - v (/5), closing speed e'(v - v_t) (/5), line-of-sight rate
    # e x e_v / max(d, 0.3) (/2), e_p and e_v in the yaw frame (/10, /5), target height above the floor (/2)
    obs_guidance: bool = False
    # > 0: append the pursuit controller's command at this cruise speed, (u_pursuit - u_hover) / action_scale (4)
    obs_pursuit_cruise: float = 0.0
    # scene pool
    pool_size: int = 20_000
    pool_seed: int = 3

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def replace(self, **kw) -> "StrikeEnvConfig":
        return replace(self, **kw)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StrikeEnvConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(payload) - names
        if unknown:
            raise ValueError(f"unknown env keys {sorted(unknown)}")
        return cls(**payload)


# ----------------------------------------------------------------------------- scene pool (numpy)
def _np_sdf(pts, obs):
    """Signed distance of points (P, 2) to rounded rectangles obs (A, >=6) -> (A, P)."""
    cx, cy, bx, by, yaw, rr = (obs[:, i:i + 1] for i in range(6))
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = pts[None, :, 0] - cx, pts[None, :, 1] - cy
    qx, qy = np.abs(c * dx + s * dy) - bx, np.abs(-s * dx + c * dy) - by
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - rr


def _np_boundary(obs, n=48):
    """Boundary points (A, n, 2) of rounded rectangles (circle for a cylinder, the rectangle's edges for a box)."""
    cx, cy, bx, by, yaw, rr = (obs[:, i:i + 1] for i in range(6))
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)[None, :]
    circ = np.stack([cx + rr * np.cos(th), cy + rr * np.sin(th)], -1)
    t = np.linspace(0, 1, n // 4, endpoint=False)
    unit = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
    edges = np.concatenate([unit[k] + np.outer(t, unit[(k + 1) % 4] - unit[k]) for k in range(4)])   # (n, 2) on [-1, 1]^2
    lx, ly = edges[None, :, 0] * bx, edges[None, :, 1] * by
    c, s = np.cos(yaw), np.sin(yaw)
    box = np.stack([cx + c * lx - s * ly, cy + s * lx + c * ly], -1)
    is_cyl = (obs[:, 2] == 0) & (obs[:, 3] == 0)
    return np.where(is_cyl[:, None, None], circ, box)


def _draw_obstacles(rng, ec: StrikeEnvConfig, n: int) -> np.ndarray:
    cyl = rng.random(n) < ec.cyl_fraction
    o = np.zeros((n, OBST_FIELDS))
    o[:, 0] = rng.uniform(ec.obst_x_min, ec.obst_x_max, n)
    o[:, 1] = rng.uniform(-ec.arena_y_half, ec.arena_y_half, n)
    o[:, 2] = np.where(cyl, 0.0, rng.uniform(ec.box_half_x_min, ec.box_half_x_max, n))
    o[:, 3] = np.where(cyl, 0.0, rng.uniform(ec.box_half_y_min, ec.box_half_y_max, n))
    o[:, 4] = np.where(cyl, 0.0, np.radians(rng.uniform(-ec.box_yaw_deg, ec.box_yaw_deg, n)))
    o[:, 5] = np.where(cyl, rng.uniform(ec.cyl_radius_min, ec.cyl_radius_max, n), 0.0)
    o[:, 6] = 1.0
    return o


def generate_scene_pool(ec: StrikeEnvConfig, n_obst_slots: int, n: int | None = None, seed: int | None = None) -> dict[str, np.ndarray]:
    """n arena scenes: start xy, target xyz, obstacles (n, n_obst_slots, 7), chasers (n, chasers_max, 4: xyz, present)."""
    if ec.obstacles_max > n_obst_slots:
        raise ValueError(f"obstacles_max {ec.obstacles_max} exceeds the arena obstacle slots {n_obst_slots}")
    n = int(n if n is not None else ec.pool_size)
    rng = np.random.default_rng(ec.pool_seed if seed is None else seed)
    res = 0.5
    xs = np.arange(0.0, ec.arena_x + 1e-9, res)
    ys = np.arange(-ec.arena_y_half, ec.arena_y_half + 1e-9, res)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel()], 1)
    wall_ok = np.abs(gy) <= ec.arena_y_half - ec.route_clearance
    out = {"start": np.zeros((n, 2)), "target": np.zeros((n, 3)), "obst": np.zeros((n, n_obst_slots, OBST_FIELDS)),
           "chasers": np.zeros((n, ec.chasers_max, 4))}
    k = 0
    while k < n:
        sx = rng.uniform(ec.start_x, ec.start_x_max) if ec.start_x_max > ec.start_x else ec.start_x
        start = np.array([sx, rng.uniform(-ec.start_y_half, ec.start_y_half)])
        target = np.array([rng.uniform(ec.target_x_min, ec.target_x_max), rng.uniform(-ec.target_y_half, ec.target_y_half),
                           rng.uniform(ec.target_z_min, ec.target_z_max)])
        if np.linalg.norm(target[:2] - start) < ec.start_target_min_dist:
            continue
        n_obs = int(rng.integers(ec.obstacles_min, ec.obstacles_max + 1))
        cand = _draw_obstacles(rng, ec, 300)
        bnd = _np_boundary(cand)
        ok = np.max(np.abs(bnd[:, :, 1]), axis=1) <= ec.arena_y_half - ec.wall_margin
        ok &= np.min(_np_sdf(np.stack([start, target[:2]]), cand), axis=1) >= ec.clear_start_target
        acc = []
        for j in np.nonzero(ok)[0]:
            if len(acc) >= n_obs:
                break
            if acc and np.min(_np_sdf(bnd[j], cand[acc])) < ec.gap_min:
                continue
            acc.append(j)
        if len(acc) < ec.obstacles_min:
            continue
        obst = cand[acc]
        if len(obst):
            free = (np.min(_np_sdf(grid, obst), axis=0).reshape(gx.shape) > ec.route_clearance) & wall_ok
        else:
            free = wall_ok
        lab, _ = ndimage.label(free)
        si = (int(round(start[0] / res)), int(round((start[1] + ec.arena_y_half) / res)))
        ti = (int(round(target[0] / res)), int(round((target[1] + ec.arena_y_half) / res)))
        if lab[si] == 0 or lab[si] != lab[ti]:
            continue
        # chasers beside the straight route
        d = target[:2] - start
        L = np.linalg.norm(d)
        u = d / L
        nrm = np.array([-u[1], u[0]])
        ch = []
        n_act_max = ec.chasers_max if ec.chasers_active_max < 0 else min(ec.chasers_active_max, ec.chasers_max)
        for _ in range(int(rng.integers(min(ec.chasers_active_min, n_act_max), n_act_max + 1))):
            for _ in range(200):
                p = start + rng.uniform(ec.chaser_along_min, ec.chaser_along_max) * L * u \
                    + rng.choice([-1.0, 1.0]) * rng.uniform(ec.chaser_side_min, ec.chaser_side_max) * nrm
                if abs(p[1]) > ec.arena_y_half - 0.8 or not (0.8 < p[0] < ec.arena_x - 0.8):
                    continue
                if (len(obst) and np.min(_np_sdf(p[None], obst)) < 1.0) or any(np.linalg.norm(p - q[:2]) < 6.0 for q in ch):
                    continue
                ch.append(np.array([p[0], p[1], rng.uniform(ec.start_z_min, ec.start_z_max)]))
                break
        out["start"][k], out["target"][k] = start, target
        out["obst"][k, :len(obst)] = obst
        for j, c in enumerate(ch):
            out["chasers"][k, j] = [*c, 1.0]
        k += 1
    return out


def load_or_generate_pool(ec: StrikeEnvConfig, n_obst_slots: int, cache_dir: str | None = None) -> dict[str, np.ndarray]:
    """The scene pool, cached as an .npz keyed by the scene settings (generated once per setting)."""
    import hashlib
    import json
    from pathlib import Path
    keys = {k: v for k, v in ec.as_dict().items() if not k.startswith(("w_", "strike_bonus", "crash_penalty", "episode_",
                                                                        "chaser_accel", "chaser_repulse", "chaser_speed_scale",
                                                                        "chaser_leash", "speed_soft", "tilt_soft",
                                                                        "obs_", "residual_",
                                                                        "target_speed"))}
    tag = hashlib.sha1(json.dumps({**keys, "slots": n_obst_slots}, sort_keys=True).encode()).hexdigest()[:10]
    path = Path(cache_dir) / f"strike_scene_pool_{tag}.npz" if cache_dir else None
    if path is not None and path.exists():
        with np.load(path) as z:
            print(f"[scene pool] {path} ({z['start'].shape[0]} scenes)", flush=True)
            return {k: z[k] for k in z.files}
    t0 = time.time()
    pool = generate_scene_pool(ec, n_obst_slots)
    print(f"[scene pool] {pool['start'].shape[0]} arena scenes in {time.time() - t0:.1f} s", flush=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **pool)
        print(f"[scene pool] cached in {path}", flush=True)
    return pool


# ----------------------------------------------------------------------------- JAX environment
class StrikeEnvState(NamedTuple):
    x: Array
    obst: Array           # (K_arena, 7)
    target_p: Array
    target_v: Array
    chaser_p: Array       # (C, 3)
    chaser_v: Array
    chaser_on: Array      # (C,)
    prev_dist: Array
    steps: Array
    chaser_home: Array    # (C, 3) start positions (the posts of leashed chasers)


class StrikeStepInfo(NamedTuple):
    episode_done: Array
    terminal: Array
    strike: Array
    crash_obstacle: Array
    crash_floor: Array
    crash_chaser: Array
    out_of_arena: Array
    timeout: Array
    dist: Array
    min_obstacle_dist: Array
    min_chaser_dist: Array
    speed: Array


class StrikeEnv(NamedTuple):
    obs_dim: int
    action_dim: int
    max_steps: int
    reset_batched: Callable
    step_batched: Callable
    reset: Callable
    step: Callable
    obs_fn: Callable
    pool: dict


def action_bounds(m: StrikeModel, ec: StrikeEnvConfig):
    """(low, high) of the policy's action: the plant's box, or +-residual_frac * action_scale for a residual striker."""
    if ec.residual_cruise > 0.0:
        b = residual_bound(m, ec)
        return -b, b
    return m.low, m.high


def residual_bound(m: StrikeModel, ec: StrikeEnvConfig):
    ft = ec.residual_frac if ec.residual_frac_thrust < 0.0 else ec.residual_frac_thrust
    return jnp.asarray([ft, ec.residual_frac, ec.residual_frac, ec.residual_frac], jnp.float32) * m.scale


def build_strike_env(m: StrikeModel, ec: StrikeEnvConfig, *, pool: dict | None = None, n_arena_slots: int | None = None) -> StrikeEnv:
    cfg = m.cfg
    C = int(ec.chasers_max)
    if C != cfg.n_chaser_slots:
        raise ValueError(f"chasers_max ({C}) must equal strike.n_chaser_slots ({cfg.n_chaser_slots})")
    K = int(n_arena_slots if n_arena_slots is not None else max(ec.obstacles_max, cfg.n_obstacle_slots))
    if pool is None:
        pool = load_or_generate_pool(ec, K)
    pj = {k: jnp.asarray(v, jnp.float32) for k, v in pool.items()}
    n_pool = int(pj["start"].shape[0])
    n_steps = int(round(ec.episode_seconds / cfg.dt))
    dt = float(cfg.dt)
    deg = np.pi / 180.0
    contact_chaser = cfg.chaser_radius + cfg.r_drone
    v_ch = float(cfg.chaser_speed * ec.chaser_speed_scale)
    residual = ec.residual_cruise > 0.0
    pursuit = make_pursuit(m, ec.residual_cruise) if residual else None
    obs_pursuit = make_pursuit(m, ec.obs_pursuit_cruise) if ec.obs_pursuit_cruise > 0.0 else None
    n_obs_extra = (19 if ec.obs_guidance else 0) + (4 if ec.obs_pursuit_cruise > 0.0 else 0)
    res_bound = residual_bound(m, ec)

    walls = arena_walls((0.0, ec.arena_x, ec.arena_y_half, ec.z_max)) if cfg.arena_walls else None

    def local_scene(x, obst, chaser_p, chaser_on):
        d = obstacle_distances(x[0:2], obst)
        idx = jnp.argsort(d)[: cfg.n_obstacle_slots]
        balls = jnp.concatenate([chaser_p, chaser_on[:, None]], axis=1)
        dc = jnp.where(chaser_on > 0.5, jnp.sum((chaser_p - x[None, 0:3]) ** 2, axis=1), 1e9)
        return join_scene(obst[idx], balls[jnp.argsort(dc)], walls), jnp.argsort(dc)

    def obs_fn(st: StrikeEnvState):
        scene, corder = local_scene(st.x, st.obst, st.chaser_p, st.chaser_on)
        base = make_obs(m, st.x, scene)
        cv = jnp.where(st.chaser_on[:, None] > 0.5, st.chaser_v, 0.0)[corder]
        if ec.obs_hide_scene:                         # same layout, no obstacle / chaser information
            base = base.at[EGO_FEATS:].set(0.0)
            cv = jnp.zeros_like(cv)
        extra = jnp.concatenate([(st.target_p - st.x[0:3]) / 10.0, st.target_v, cv.reshape(-1),
                                 jnp.stack([st.x[0] / (0.5 * ec.arena_x) - 1.0, st.x[1] / ec.arena_y_half])])
        parts = [base, extra]
        if ec.obs_guidance:
            parts.append(guidance_features(st))
        if ec.obs_pursuit_cruise > 0.0:
            parts.append((obs_pursuit(st.x, st.target_p, st.target_v) - m.u_hover) / m.scale)
        return jnp.concatenate(parts).astype(jnp.float32)

    def guidance_features(st: StrikeEnvState):
        p, v, q = st.x[0:3], st.x[3:6], st.x[6:10]
        e_p = st.target_p - p
        d = jnp.sqrt(jnp.sum(e_p ** 2) + 1e-9)
        e = e_p / d
        e_v = st.target_v - v
        closing = -e @ e_v
        los_rate = jnp.cross(e, e_v) / jnp.maximum(d, 0.3)
        yaw = jnp.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))
        c, s_ = jnp.cos(yaw), jnp.sin(yaw)
        to_yaw = lambda w: jnp.stack([c * w[0] + s_ * w[1], -s_ * w[0] + c * w[1], w[2]])
        return jnp.concatenate([e, jnp.stack([d / 10.0, jnp.log(d + 0.1)]), e_v / 5.0, (closing / 5.0)[None],
                                los_rate / 2.0, to_yaw(e_p) / 10.0, to_yaw(e_v) / 5.0,
                                ((st.target_p[2] - cfg.z_floor) / 2.0)[None]])

    def reset(key):
        ks = jax.random.split(key, 9)
        i = jax.random.randint(ks[0], (), 0, n_pool)
        u = lambda k, lo, hi: jax.random.uniform(k, (), jnp.float32, lo, hi)
        start, target = pj["start"][i], pj["target"][i]
        z0 = u(ks[1], ec.start_z_min, ec.start_z_max)
        spd = u(ks[2], ec.start_speed_min, ec.start_speed_max)
        to_t = target[0:2] - start
        hd = jnp.arctan2(to_t[1], to_t[0]) + u(ks[3], -1.0, 1.0) * ec.start_heading_deg * deg
        tilt, ax = u(ks[4], 0.0, ec.start_tilt_deg * deg), u(ks[5], -np.pi, np.pi)
        rv = jnp.stack([tilt * jnp.cos(ax), tilt * jnp.sin(ax), 0.0])
        ang = jnp.sqrt(jnp.sum(rv ** 2) + 1e-12)
        q = jnp.concatenate([jnp.cos(ang / 2)[None], jnp.sin(ang / 2) * rv / ang])
        x = jnp.concatenate([start, z0[None], jnp.stack([spd * jnp.cos(hd), spd * jnp.sin(hd), 0.0]), q])
        th = u(ks[6], -np.pi, np.pi)
        tv = u(ks[7], ec.target_speed_min, ec.target_speed_max) * jnp.stack([jnp.cos(th), jnp.sin(th), 0.0])
        cp, con = pj["chasers"][i][:, 0:3], pj["chasers"][i][:, 3]
        to_d = x[None, 0:3] - cp
        cv = v_ch * to_d / jnp.sqrt(jnp.sum(to_d ** 2, axis=1, keepdims=True) + 1e-9)
        st = StrikeEnvState(x=x, obst=pj["obst"][i], target_p=target, target_v=tv, chaser_p=cp, chaser_v=cv * con[:, None],
                            chaser_on=con, prev_dist=jnp.sqrt(jnp.sum((target - x[0:3]) ** 2)), steps=jnp.int32(0),
                            chaser_home=cp)
        return st, obs_fn(st)

    def chaser_step(st: StrikeEnvState, x_new):
        def one(p, v, home):
            to_d = x_new[0:3] - p
            dirn = to_d / jnp.sqrt(jnp.sum(to_d ** 2) + 1e-9)
            sd = lambda q: jnp.min(obstacle_distances(q, st.obst))
            d, g = jax.value_and_grad(sd)(p[0:2])
            push = jnp.clip((ec.chaser_repulse_dist - d) / ec.chaser_repulse_dist, 0.0, 1.0)
            want = dirn + 2.0 * push * jnp.concatenate([g, jnp.zeros(1)])
            v_des = v_ch * want / jnp.sqrt(jnp.sum(want ** 2) + 1e-9)
            a = (v_des - v) / dt
            a = a * jnp.minimum(1.0, ec.chaser_accel / (jnp.sqrt(jnp.sum(a ** 2)) + 1e-9))
            vn = v + dt * a
            vn = vn * jnp.minimum(1.0, v_ch / (jnp.sqrt(jnp.sum(vn ** 2)) + 1e-9))
            pn = p + dt * vn
            pn = pn.at[2].set(jnp.clip(pn[2], cfg.z_floor + cfg.chaser_radius, ec.z_max))
            if ec.chaser_leash > 0.0:                     # guard: stay within the leash of the post
                off = pn - home
                d = jnp.sqrt(jnp.sum(off ** 2) + 1e-12)
                pn = home + off * jnp.minimum(1.0, ec.chaser_leash / d)
                vn = (pn - p) / dt
            return pn, vn
        pn, vn = jax.vmap(one)(st.chaser_p, st.chaser_v, st.chaser_home)
        on = st.chaser_on[:, None] > 0.5
        return jnp.where(on, pn, st.chaser_p), jnp.where(on, vn, st.chaser_v)

    def target_step(p, v):
        pn = p + dt * v
        lo = jnp.array([ec.target_zone_x_min, -ec.arena_y_half + 1.0])
        hi = jnp.array([ec.arena_x - 1.0, ec.arena_y_half - 1.0])
        flip = (pn[0:2] < lo) | (pn[0:2] > hi)
        vxy = jnp.where(flip, -v[0:2], v[0:2])
        pxy = jnp.clip(pn[0:2], lo, hi)
        return jnp.concatenate([pxy, p[2:3]]), jnp.concatenate([vxy, v[2:3]])

    def step(st: StrikeEnvState, u, key):
        u = jnp.asarray(u, jnp.float32)
        if residual:
            du = jnp.clip(u, -res_bound, res_bound)
            u = jnp.clip(pursuit(st.x, st.target_p, st.target_v) + du, m.low, m.high)
        else:
            u = jnp.clip(u, m.low, m.high)
            du = u - m.u_hover
        xn = plant_step(m, st.x, u)
        tp, tv = target_step(st.target_p, st.target_v)
        cp, cv = chaser_step(st, xn)
        dist = jnp.sqrt(jnp.sum((tp - xn[0:3]) ** 2))
        d_obs = jnp.min(obstacle_distances(xn[0:2], st.obst))
        d_ch = jnp.min(jnp.where(st.chaser_on > 0.5, jnp.sqrt(jnp.sum((cp - xn[None, 0:3]) ** 2, axis=1)), 1e3))
        c_obs = d_obs < cfg.r_drone
        c_floor = xn[2] < cfg.z_floor
        c_ch = d_ch < contact_chaser
        out = (xn[0] < -1.0) | (xn[0] > ec.arena_x + 1.0) | (jnp.abs(xn[1]) > ec.arena_y_half + 1.0) | (xn[2] > ec.z_max)
        crash = c_obs | c_floor | c_ch | out
        strike = (dist < ec.strike_radius) & ~crash
        steps = st.steps + jnp.int32(1)
        timeout = (steps >= n_steps) & ~crash & ~strike
        terminal = crash | strike
        done = terminal | timeout
        act_cost = jnp.mean((du / m.scale) ** 2)
        spd = jnp.sqrt(jnp.sum(xn[3:6] ** 2))
        tilt = jnp.arccos(jnp.clip(1.0 - 2.0 * (xn[7] ** 2 + xn[8] ** 2), -1.0, 1.0))
        envelope = (ec.w_speed_excess * jnp.maximum(spd - ec.speed_soft, 0.0) ** 2
                    + ec.w_tilt_excess * jnp.maximum(tilt - ec.tilt_soft_deg * deg, 0.0) ** 2)
        rew = (ec.w_progress * (st.prev_dist - dist) - ec.w_time - ec.w_action * act_cost - envelope
               + ec.strike_bonus * strike - ec.crash_penalty * crash)
        nxt = StrikeEnvState(x=xn, obst=st.obst, target_p=tp, target_v=tv, chaser_p=cp, chaser_v=cv, chaser_on=st.chaser_on,
                             prev_dist=dist, steps=steps, chaser_home=st.chaser_home)
        o_true = obs_fn(nxt)
        rst, robs = reset(key)
        st_out = jax.tree_util.tree_map(lambda r, c: jnp.where(done, r, c), rst, nxt)
        info = StrikeStepInfo(episode_done=done, terminal=terminal, strike=strike, crash_obstacle=c_obs, crash_floor=c_floor,
                              crash_chaser=c_ch, out_of_arena=out & ~(c_obs | c_floor | c_ch), timeout=timeout, dist=dist,
                              min_obstacle_dist=d_obs, min_chaser_dist=d_ch, speed=jnp.sqrt(jnp.sum(xn[3:6] ** 2)))
        return st_out, o_true, jnp.where(done, robs, o_true), rew, done, info

    return StrikeEnv(obs_dim=cfg.obs_dim + 14 + n_obs_extra, action_dim=4, max_steps=n_steps,
                     reset_batched=jax.jit(jax.vmap(reset)), step_batched=jax.jit(jax.vmap(step)),
                     reset=jax.jit(reset), step=jax.jit(step), obs_fn=obs_fn, pool=pool)


__all__ = ["action_bounds", "StrikeEnv", "StrikeEnvConfig", "StrikeEnvState", "StrikeStepInfo", "build_strike_env", "generate_scene_pool",
           "load_or_generate_pool"]
