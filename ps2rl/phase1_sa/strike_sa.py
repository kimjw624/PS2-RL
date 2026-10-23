"""Phase-I safe-arrival backup for the strike task: scene regions, environment, evaluator and training run.

Objective, update and loop are the shared safe-arrival ones (``sa_trainer_core``: value beta^(steps to B),
TD3 or SAC, design-region mixture): an episode is one backup rollout from a sampled (state, scene) pair that ends
when the base set is reached (success), the safe set is left (failure) or ``episode_max_steps`` pass. No disturbance.

Regions (drone at the origin of the scene, obstacles and chaser balls around it):
  general  random speed / tilt, 0..obstacles_max obstacles, 0..n_chaser_slots chaser balls;
  blocked  faster states with an obstacle where the LQR backup would end up after T;
  chaser   faster states with a chaser ball next to where the LQR would end up;
  shell    slow, near-hover states high enough for the hover ellipsoid, with the LQR end point clear (sanity region);
  escape   slow states with a chaser ball 0..escape_gap_max m beyond contact distance: the backup has to move away (what the
           filter needs when a chaser closes in on a slow or hovering vehicle). Weight 0 in a 4-entry mixture.
Episode starts are drawn from a pre-sampled pool per region (``pool_per_region`` each), held-out val / test sets
are separate draws.

Moving chasers (StrikeConfig.chaser_set "guidance"): the chasers pursue the drone at v_c during every episode and
evaluation rollout (``pursue_scene``), the safe set holds the escape margin h_c >= 0 and the base set a hover with
h_c >= chaser_base_margin (strike_scene); the regions place chasers relative to the contact distance r_c + r_d instead
of the static ball radius. Arena walls (StrikeConfig.arena_walls): per axis, with probability wall_prob, a wall at
wall_dist_min..wall_dist_max on a random side and the opposite one a span away; a ceiling at ceiling_z_min..max.

Evaluation: first-hit membership in C_N (base set reached within N = T / dt steps before leaving the safe set),
per region, for the learned backup and for the hover-LQR backup on the same held-out scenes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import time
from typing import Any, Callable, Dict, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.envs.strike_scene import (
    StrikeModel,
    backup_base_action,
    fail_from_obs,
    flee_action,
    goal_from_obs,
    in_base_set,
    is_safe,
    join_scene,
    lqr_action,
    make_obs,
    margins,
    obstacle_distances,
    plant_step,
    pursue_scene,
    sort_scene,
)
from ps2rl.phase1_sa.quadrotor_landing_sa_env import LandingSAStepInfo
from ps2rl.phase1_sa.quadrotor_landing_sa_trainer import (
    _EPISODE_HISTORY_PAIRS,
    _EPISODE_METRIC_FIELDS,
    LandingSAConfig,
    _episode_bookkeeping,
)
from ps2rl.phase1_sa.sa_critic import SafeArrivalCriticConfig
from ps2rl.phase1_sa.sa_trainer_core import (
    SALoopState,
    SASystemHooks,
    build_sa_action_fns,
    build_sa_one_vec_step,
    build_sa_update_fn,
    init_sa_state,
    run_sa_training_loop,
    sa_backbone,
    sa_replay_init,
    snapshot_sa_state,
)
from ps2rl.utils.optim import adam_init, adam_step
from ps2rl.utils.policy import ActorConfig, actor_mean_action, clip_to_box
from ps2rl.utils.seed import make_prng_key

Array = jax.Array
REGION_NAMES = ("general", "blocked", "chaser", "shell", "escape")
_FAR_WALL = 1.0e3   # an absent arena wall
StrikeSAConfig = LandingSAConfig   # same hyper-parameters (the task file sets the strike values)


@dataclass(frozen=True)
class StrikeRegionConfig:
    # start state (drone at the scene origin)
    z_min: float = 1.5
    z_max: float = 6.0
    speed_max: float = 10.0
    fast_speed_min: float = 3.0          # blocked / chaser regions
    vz_max: float = 2.0
    tilt_max_deg: float = 45.0
    yaw_max_deg: float = 30.0
    # shell region: slow, high enough for the hover ellipsoid, LQR end point clear of obstacles
    shell_speed_max: float = 2.5
    shell_vz_max: float = 0.5
    shell_tilt_max_deg: float = 17.0
    shell_z_margin: float = 0.15         # above z_floor + z_ext
    shell_clear_margin: float = 0.3      # beyond rho_xy + r_drone around the LQR end point
    shell_obstacles_max: int = 3
    shell_dist_min: float = 3.0
    # obstacles around the drone (centre distance range) and their shapes
    obstacles_max: int = 6
    dist_min: float = 2.0
    dist_max: float = 10.0
    cyl_fraction: float = 0.5
    cyl_radius_min: float = 0.8
    cyl_radius_max: float = 2.0
    box_half_x_min: float = 1.5
    box_half_x_max: float = 3.0
    box_half_y_min: float = 1.0
    box_half_y_max: float = 2.5
    blocked_std: float = 0.8             # obstacle centre around the LQR end point
    # chaser balls
    ball_gap_min: float = 0.5            # general: centre distance from R + gap ...
    ball_dist_max: float = 12.0          # ... to this
    ball_dz_max: float = 2.0
    chaser_near_extra: float = 2.0       # chaser region: centre within R + extra of the LQR end point
    escape_gap_max: float = 2.0          # escape region: ball surface 0..this far from the drone (slow, shell-like states)
    escape_gap_min: float = 0.0          # ... from this (moving chasers: hovering needs h_c >= 0, i.e. ~0.8 m beyond contact)
    # arena walls (StrikeConfig.arena_walls): per axis with probability wall_prob a wall wall_dist_min..max away on a
    # random side, the opposite wall wall_span_* (min, max) away from it; ceiling height ceiling_z_min..max
    wall_prob: float = 0.5
    wall_dist_min: float = 0.5
    wall_dist_max: float = 12.0
    wall_span_x: tuple = (30.0, 50.0)
    wall_span_y: tuple = (16.0, 24.0)
    ceiling_z_min: float = 8.0
    ceiling_z_max: float = 12.0
    # region mixture (general, blocked, chaser, shell[, escape]) at curriculum scale 0 and 1; scale 1 also weights mu.
    # A 4-entry mixture (runs before the escape region) gives escape weight 0.
    mix_low: tuple = (0.5, 0.1, 0.1, 0.3)
    mix_high: tuple = (0.4, 0.2, 0.2, 0.2)
    max_tries: int = 16
    pool_per_region: int = 50_000
    pool_seed: int = 11
    heldout_seed: int = 2026
    heldout_val: int = 1000
    heldout_test: int = 2000

    def __post_init__(self) -> None:
        pad = lambda w: tuple(float(v) for v in w) + (0.0,) * (len(REGION_NAMES) - len(w))
        if not (4 <= len(self.mix_low) <= len(REGION_NAMES) and 4 <= len(self.mix_high) <= len(REGION_NAMES)):
            raise ValueError(f"mix_low / mix_high need one weight per region {REGION_NAMES} (escape may be left out)")
        object.__setattr__(self, "mix_low", pad(self.mix_low))
        object.__setattr__(self, "mix_high", pad(self.mix_high))
        object.__setattr__(self, "wall_span_x", tuple(float(v) for v in self.wall_span_x))
        object.__setattr__(self, "wall_span_y", tuple(float(v) for v in self.wall_span_y))

    @property
    def weights(self) -> dict[str, float]:
        w = np.asarray(self.mix_high, dtype=np.float64)
        return {n: float(v) for n, v in zip(REGION_NAMES, w / w.sum())}

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mix_low"], d["mix_high"] = list(self.mix_low), list(self.mix_high)
        d["wall_span_x"], d["wall_span_y"] = list(self.wall_span_x), list(self.wall_span_y)
        return d

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StrikeRegionConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(payload) - names
        if unknown:
            raise ValueError(f"unknown region keys {sorted(unknown)}")
        return cls(**payload)


# ----------------------------------------------------------------------------- scenes
def lqr_end_point(m: StrikeModel, x: Array) -> Array:
    """Where the hover-LQR backup is after the horizon T in free space."""
    return jax.lax.fori_loop(0, m.cfg.num_steps, lambda _, z: plant_step(m, z, lqr_action(m, z)), x)[0:3]


def build_region_sampler(m: StrikeModel, rc: StrikeRegionConfig) -> Callable:
    """sample_region(idx, key) -> (start vector (10 + scene_dim,), found). Scene sorted nearest first."""
    cfg = m.cfg
    K, J = cfg.n_obstacle_slots, cfg.n_chaser_slots
    if rc.obstacles_max > K - 1 or rc.shell_obstacles_max > K - 1:
        raise ValueError(f"obstacles_max must leave one slot free for the blocked obstacle (n_obstacle_slots {K})")
    R = cfg.contact_radius if cfg.moving_chasers else cfg.ball_radius   # chaser placement: distances beyond R
    z_ch_min = cfg.z_floor + (cfg.chaser_radius if cfg.moving_chasers else 0.0)
    z_shell = cfg.z_floor + m.lqr.z_ext + rc.shell_z_margin
    clear_shell = m.lqr.rho_xy + cfg.r_drone + rc.shell_clear_margin
    deg = np.pi / 180.0

    def draw(idx, key):
        ks = jax.random.split(key, 15)
        general, blocked, chaser, shell, escape = idx == 0, idx == 1, idx == 2, idx == 3, idx == 4
        slow = shell | escape
        u = lambda k, lo, hi, shape=(): jax.random.uniform(k, shape, jnp.float32, lo, hi)
        z = jnp.where(slow, u(ks[0], z_shell, rc.z_max), u(ks[0], rc.z_min, rc.z_max))
        spd = jnp.where(slow, u(ks[1], 0.0, rc.shell_speed_max),
                        jnp.where(general, u(ks[1], 0.0, rc.speed_max), u(ks[1], rc.fast_speed_min, rc.speed_max)))
        tilt = u(ks[2], 0.0, 1.0) * deg * jnp.where(slow, rc.shell_tilt_max_deg, rc.tilt_max_deg)
        ax = u(ks[3], -np.pi, np.pi)
        yaw = u(ks[4], -1.0, 1.0) * rc.yaw_max_deg * deg
        hd = u(ks[5], -np.pi, np.pi)
        vz = jnp.clip(u(ks[6], -1.0, 1.0) * jnp.where(slow, rc.shell_vz_max, rc.vz_max), -spd, spd)
        vh = jnp.sqrt(jnp.maximum(spd ** 2 - vz ** 2, 0.0))
        rv = jnp.stack([tilt * jnp.cos(ax), tilt * jnp.sin(ax), yaw])
        ang = jnp.sqrt(jnp.sum(rv ** 2) + 1e-12)
        q = jnp.concatenate([jnp.cos(ang / 2)[None], jnp.sin(ang / 2) * rv / ang])
        x = jnp.concatenate([jnp.array([0.0, 0.0]), z[None], jnp.stack([vh * jnp.cos(hd), vh * jnp.sin(hd), vz]), q])
        p_end = lqr_end_point(m, x)

        # obstacles: slots 0..K-2 random, slot K-1 reserved for the blocked region
        n_obs = jax.random.randint(ks[7], (), 0, jnp.where(shell, rc.shell_obstacles_max, rc.obstacles_max) + 1)
        ko = jax.random.split(ks[8], 7)
        dist = u(ko[0], 0.0, 1.0, (K,)) * (rc.dist_max - jnp.where(shell, rc.shell_dist_min, rc.dist_min)) \
            + jnp.where(shell, rc.shell_dist_min, rc.dist_min)
        th = u(ko[1], -np.pi, np.pi, (K,))
        cyl = u(ko[2], 0.0, 1.0, (K,)) < rc.cyl_fraction
        bx = jnp.where(cyl, 0.0, u(ko[3], rc.box_half_x_min, rc.box_half_x_max, (K,)))
        by = jnp.where(cyl, 0.0, u(ko[4], rc.box_half_y_min, rc.box_half_y_max, (K,)))
        oy = jnp.where(cyl, 0.0, u(ko[5], -np.pi / 2, np.pi / 2, (K,)))
        rr = jnp.where(cyl, u(ko[6], rc.cyl_radius_min, rc.cyl_radius_max, (K,)), 0.0)
        cx, cy = dist * jnp.cos(th), dist * jnp.sin(th)
        present = jnp.arange(K) < n_obs
        c_blk = p_end[0:2] + rc.blocked_std * jax.random.normal(ks[9], (2,))
        cx = cx.at[K - 1].set(jnp.where(blocked, c_blk[0], cx[K - 1]))
        cy = cy.at[K - 1].set(jnp.where(blocked, c_blk[1], cy[K - 1]))
        present = present.at[K - 1].set(blocked | present[K - 1])
        obst = jnp.stack([cx, cy, bx, by, oy, rr, present.astype(jnp.float32)], axis=1)
        d_end = obstacle_distances(p_end[0:2], obst)
        obst = obst.at[:, 6].set(jnp.where(shell & (d_end < clear_shell), 0.0, obst[:, 6]))

        # chaser balls
        n_ch = jnp.where(general | blocked, jax.random.randint(ks[10], (), 0, J + 1), jnp.where(chaser | escape, 1, 0))
        kc = jax.random.split(ks[11], 4)
        bd = u(kc[0], R + rc.ball_gap_min, rc.ball_dist_max, (J,))
        bth = u(kc[1], -np.pi, np.pi, (J,))
        far = jnp.stack([bd * jnp.cos(bth), bd * jnp.sin(bth), z + u(kc[2], -rc.ball_dz_max, rc.ball_dz_max, (J,))], 1)
        dirr = jax.random.normal(kc[3], (3,))
        near = p_end + dirr / jnp.sqrt(jnp.sum(dirr ** 2) + 1e-12) * u(ks[12], 0.0, R + rc.chaser_near_extra)
        d_esc = jax.random.normal(ks[13], (3,)) * jnp.array([1.0, 1.0, 0.3])      # mostly horizontal
        esc = x[0:3] + d_esc / jnp.sqrt(jnp.sum(d_esc ** 2) + 1e-12) * (R + u(ks[12], rc.escape_gap_min, rc.escape_gap_max))
        centres = far.at[0].set(jnp.where(chaser, near, jnp.where(escape, esc, far[0])))
        centres = centres.at[:, 2].set(jnp.maximum(centres[:, 2], z_ch_min))     # chasers fly above the floor
        balls = jnp.concatenate([centres, (jnp.arange(J) < n_ch).astype(jnp.float32)[:, None]], axis=1)
        walls = draw_walls(ks[14], p_end, shell) if cfg.arena_walls else None
        scene = join_scene(obst, balls, walls)
        return x, scene, is_safe(m, x, scene)

    def draw_walls(key, p_end, shell):
        """(x_lo, x_hi, y_lo, y_hi, z_hi): per axis a near wall on a random side (probability wall_prob) and the opposite
        one a span away; shell starts drop walls too close to the LQR end point."""
        kw = jax.random.split(key, 5)
        on = jax.random.uniform(kw[0], (2,)) < rc.wall_prob
        side = jax.random.uniform(kw[1], (2,)) < 0.5                    # True: the near wall on the low side
        d = jax.random.uniform(kw[2], (2,), jnp.float32, rc.wall_dist_min, rc.wall_dist_max)
        span = jnp.stack([jax.random.uniform(kw[3], (), jnp.float32, *rc.wall_span_x),
                          jax.random.uniform(kw[4], (), jnp.float32, *rc.wall_span_y)])
        lo = jnp.where(on, jnp.where(side, -d, -(span - d)), -_FAR_WALL)     # near wall at -d (low side) or +d (high side)
        hi = jnp.where(on, jnp.where(side, span - d, d), _FAR_WALL)
        z_hi = jax.random.uniform(jax.random.fold_in(key, 7), (), jnp.float32, rc.ceiling_z_min, rc.ceiling_z_max)
        walls = jnp.stack([lo[0], hi[0], lo[1], hi[1], z_hi])
        # shell: keep the LQR end point's hover clearance from the walls
        gaps = jnp.stack([p_end[0] - walls[0], walls[1] - p_end[0], p_end[1] - walls[2], walls[3] - p_end[1]])
        far_v = jnp.array([-_FAR_WALL, _FAR_WALL, -_FAR_WALL, _FAR_WALL])
        return walls.at[0:4].set(jnp.where(shell & (gaps < clear_shell), far_v, walls[0:4]))

    def sample_region(idx, key):
        k0, key = jax.random.split(key)
        x, s, f = draw(idx, k0)

        def cond(c):
            return (~c[4]) & (c[0] < rc.max_tries)

        def body(c):
            i, k, _, _, _ = c
            k, kd = jax.random.split(k)
            x_, s_, f_ = draw(idx, kd)
            return i + 1, k, x_, s_, f_

        _, _, x, s, f = jax.lax.while_loop(cond, body, (jnp.int32(1), key, x, s, f))
        return jnp.concatenate([x, sort_scene(m, x, s)]), f

    return sample_region


_SAMPLER_CACHE: dict = {}


def _region_sampler_fn(m: StrikeModel, rc: StrikeRegionConfig) -> Callable:
    """jit(vmap(sample_region)) compiled once per (task, region config); the region index is a traced argument."""
    key = (m.cfg, rc)
    if key not in _SAMPLER_CACHE:
        _SAMPLER_CACHE[key] = jax.jit(jax.vmap(build_region_sampler(m, rc), in_axes=(None, 0)))
    return _SAMPLER_CACHE[key]


def draw_region_states(m: StrikeModel, rc: StrikeRegionConfig, idx: int, n: int, seed: int, *, chunk: int = 4096) -> np.ndarray:
    """n safe (state, scene) start vectors of region idx (i.i.d.; a draw that finds no safe start is redrawn)."""
    fn = _region_sampler_fn(m, rc)
    out, have, key = [], 0, jax.random.PRNGKey(seed)
    while have < n:
        key, k = jax.random.split(key)
        v, f = map(np.asarray, fn(jnp.int32(idx), jax.random.split(k, chunk)))   # fixed chunk: one compilation
        v = v[f]
        out.append(v)
        have += v.shape[0]
    return np.concatenate(out)[:n]


def heldout_sets(m: StrikeModel, rc: StrikeRegionConfig, split: str) -> dict[str, np.ndarray]:
    off = {"val": 0, "test": 1}[split]
    n = rc.heldout_val if split == "val" else rc.heldout_test
    return {name: draw_region_states(m, rc, i, n, rc.heldout_seed + 1000 * off + 17 * i) for i, name in enumerate(REGION_NAMES)}


# ----------------------------------------------------------------------------- environment
class StrikeSAEnvState(NamedTuple):
    x: Array
    scene: Array
    steps: Array
    ep_min_h: Array


class StrikeSAEnvFns(NamedTuple):
    obs_dim: int
    action_dim: int
    reset_batched: Callable
    step_batched: Callable
    goal_fn: Callable      # single observation -> bool
    fail_fn: Callable
    pools: Array           # (4 or 5 regions, pool_per_region, 10 + scene_dim)
    mixture: Callable      # curriculum scale -> region weights


def build_strike_sa_env(m: StrikeModel, rc: StrikeRegionConfig, *, episode_max_steps: int, terminate_on_goal: bool = True,
                        pools: np.ndarray | None = None) -> StrikeSAEnvFns:
    cfg = m.cfg
    if pools is None:
        t0 = time.time()
        n_reg = len(REGION_NAMES) if float(rc.mix_low[4]) + float(rc.mix_high[4]) > 0.0 else 4   # escape pool when used
        pools = np.stack([draw_region_states(m, rc, i, rc.pool_per_region, rc.pool_seed + 31 * i) for i in range(n_reg)])
        print(f"[pool] {rc.pool_per_region} episode starts per region in {time.time() - t0:.1f} s", flush=True)
    pools_j = jnp.asarray(pools, dtype=jnp.float32)
    n_pool = int(pools_j.shape[1])
    w_lo, w_hi = jnp.asarray(rc.mix_low, jnp.float32), jnp.asarray(rc.mix_high, jnp.float32)
    k_max = jnp.int32(int(episode_max_steps))
    n_backup = jnp.int32(cfg.num_steps)
    term_goal = jnp.asarray(bool(terminate_on_goal))
    inf = jnp.asarray(jnp.inf, jnp.float32)
    goal_fn = lambda o: goal_from_obs(m, o)
    fail_fn = lambda o: fail_from_obs(m, o)

    def mixture(scale):
        s = jnp.clip(jnp.asarray(scale, jnp.float32), 0.0, 1.0)
        w = (1.0 - s) * w_lo + s * w_hi
        return w / jnp.sum(w)

    def env_reset(key, scale):
        k1, k2 = jax.random.split(key)
        idx = jax.random.categorical(k1, jnp.log(mixture(scale) + 1e-12))
        v = pools_j[idx, jax.random.randint(k2, (), 0, n_pool)]
        x, scene = v[:10], v[10:]
        return StrikeSAEnvState(x=x, scene=scene, steps=jnp.int32(0), ep_min_h=inf), make_obs(m, x, scene)

    def env_step(state, raw_action, key, scale):
        raw = clip_to_box(jnp.asarray(raw_action, jnp.float32), m.d_low, m.d_high)      # du on top of the base action
        applied = clip_to_box(backup_base_action(m, state.x, state.scene) + raw, m.low, m.high)
        xn = plant_step(m, state.x, applied)
        scene_n = pursue_scene(m, state.scene, state.x[0:3])                          # moving chasers pursue the drone
        o_true = make_obs(m, xn, scene_n)
        goal, fail = goal_fn(o_true), fail_fn(o_true)
        h = margins(m, xn, scene_n)[0]
        steps = state.steps + jnp.int32(1)
        timeout = steps >= k_max
        done = fail | (term_goal & goal) | timeout
        ep_min_h = jnp.minimum(state.ep_min_h, h)
        zero = jnp.asarray(0.0, jnp.float32)
        success = done & goal & ~fail
        info = LandingSAStepInfo(
            goal_next=goal.astype(jnp.float32), fail_next=fail.astype(jnp.float32), raw_action=raw, applied_action=applied,
            episode_done=done, completed_len=jnp.where(done, steps.astype(jnp.float32), zero),
            completed_success=success.astype(jnp.float32), completed_crash=(done & fail).astype(jnp.float32),
            completed_timeout=(done & timeout & ~goal & ~fail).astype(jnp.float32),
            completed_success_within_n=(success & (steps <= n_backup)).astype(jnp.float32),
            completed_entry_step=jnp.where(success, steps, jnp.int32(-1)),
            completed_min_h=jnp.where(done, ep_min_h, zero),
        )
        rst, robs = env_reset(key, scale)
        nxt = StrikeSAEnvState(x=xn, scene=scene_n, steps=steps, ep_min_h=ep_min_h)
        st = jax.tree_util.tree_map(lambda r, c: jnp.where(done, r, c), rst, nxt)
        return st, o_true, jnp.where(done, robs, o_true), done, info

    return StrikeSAEnvFns(obs_dim=cfg.obs_dim, action_dim=4,
                          reset_batched=jax.jit(jax.vmap(env_reset, in_axes=(0, None))),
                          step_batched=jax.jit(jax.vmap(env_step, in_axes=(0, 0, 0, None))),
                          goal_fn=goal_fn, fail_fn=fail_fn, pools=pools_j, mixture=mixture)


# ----------------------------------------------------------------------------- backups and rollouts
def make_actor_backup(m: StrikeModel, actor_cfg: ActorConfig) -> Callable:
    """(params, x, scene) -> learned backup action: base action (LQR or hover) + du (deterministic tanh mean)."""
    def act(params, x, scene):
        o = make_obs(m, x, scene)
        du = actor_mean_action(params, o[None], m.scale, actor_cfg, action_low=m.d_low, action_high=m.d_high)[0]
        return jnp.clip(backup_base_action(m, x, scene) + jnp.nan_to_num(du), m.low, m.high)
    return act


def make_lqr_backup(m: StrikeModel) -> Callable:
    return lambda params, x, scene: lqr_action(m, x)


def make_flee_backup(m: StrikeModel) -> Callable:
    """The flee-then-hover heuristic (strike_scene.flee_action, StrikeConfig flee_*) as a backup: moving chasers only."""
    if not m.cfg.moving_chasers:
        raise ValueError("the flee backup is for moving chasers (strike chaser_set 'guidance')")
    return lambda params, x, scene: flee_action(m, x, scene)


def make_rollout(m: StrikeModel, backup: Callable, *, n_steps: int | None = None, trajectory: bool = False) -> Callable:
    """(params, x0, scene) -> (reached B, failed, first-hit step, [states (n+1, 10)]): first-hit C_N membership.

    The rollout runs all n steps (states after the first hit are kept for plotting); the verdict uses the first hit.
    Moving chasers (chaser_set "guidance") pursue the drone along the rollout."""
    n = int(n_steps if n_steps is not None else m.cfg.num_steps)

    def rollout(params, x0, scene0):
        hb0 = in_base_set(m, x0, scene0)
        hf0 = ~is_safe(m, x0, scene0)

        def body(c, _):
            x, scene, hb, hf, k_end, k = c
            xn = plant_step(m, x, backup(params, x, scene))
            sn = pursue_scene(m, scene, x[0:3])
            act = ~(hb | hf)
            nf = act & ~is_safe(m, xn, sn)
            nb = act & ~nf & in_base_set(m, xn, sn)
            k_end = jnp.where(nf | nb, k + 1, k_end)
            return (xn, sn, hb | nb, hf | nf, k_end, k + 1), xn

        init = (x0, scene0, hb0, hf0, jnp.where(hb0 | hf0, 0, n), jnp.int32(0))
        (_, _, hb, hf, k_end, _), xs = jax.lax.scan(body, init, None, length=n)
        if trajectory:
            return hb & ~hf, hf, k_end, jnp.concatenate([x0[None], xs])
        return hb & ~hf, hf, k_end

    return rollout


def build_evaluator(m: StrikeModel, rc: StrikeRegionConfig, backup: Callable, *, beta: float, hoeffding_delta: float,
                    sets: dict[str, dict[str, np.ndarray]] | None = None) -> Callable:
    """run_eval(params, split) -> per-region C_N membership of ``backup`` on the held-out scenes."""
    roll = jax.jit(jax.vmap(make_rollout(m, backup), in_axes=(None, 0, 0)))
    sets = sets or {split: heldout_sets(m, rc, split) for split in ("val", "test")}
    weights = rc.weights

    def run_eval(params, split: str) -> Dict[str, Any]:
        stats: Dict[str, Any] = {"split": split}
        mu, crash = 0.0, []
        for name in REGION_NAMES:
            v = sets[split][name]
            ok, hf, k_end = map(np.asarray, roll(params, jnp.asarray(v[:, :10]), jnp.asarray(v[:, 10:])))
            n = int(v.shape[0])
            arr = k_end[ok]
            stats[name] = {"n": n, "m_hat": float(ok.mean()), "crash_rate": float(hf.mean()),
                           "timeout_rate": float((~ok & ~hf).mean()),
                           "hoeffding_halfwidth": float(np.sqrt(np.log(2.0 / hoeffding_delta) / (2.0 * max(n, 1)))),
                           "mean_arrival_steps": float(arr.mean()) if arr.size else float("nan"),
                           "mean_discounted_score": float(np.where(ok, beta ** np.maximum(k_end, 0), 0.0).mean())}
            mu += weights[name] * stats[name]["m_hat"]
            crash.append(hf)
        stats["mu_weighted"] = mu
        stats["crash_rate"] = float(np.concatenate(crash).mean())
        return stats

    run_eval.sets = sets
    return run_eval


# ----------------------------------------------------------------------------- training
_EVAL_KEYS = tuple(f"eval_m_{r}" for r in REGION_NAMES) + ("eval_mu_weighted", "eval_crash_rate")
_HISTORY_KEYS = ("step", "episode_idx", "curriculum_scale", "critic_loss", "actor_loss", "action_penalty", "target_mean",
                 "q_pi_mean", "q1_grad_norm", "q2_grad_norm", "actor_grad_norm", "alpha", "entropy",
                 *[k for k, _ in _EPISODE_HISTORY_PAIRS], *_EVAL_KEYS)


def _eval_rank_key(stats: Dict[str, Any]) -> tuple:
    return (float(stats["mu_weighted"]), float(stats["general"]["m_hat"]))


def _append_eval_history(history: Dict[str, list], stats: Dict[str, Any]) -> None:
    for name in REGION_NAMES:
        history[f"eval_m_{name}"].append(float(stats[name]["m_hat"]))
    history["eval_mu_weighted"].append(float(stats["mu_weighted"]))
    history["eval_crash_rate"].append(float(stats["crash_rate"]))
    print(f"  [eval {stats['split']}] mu_w={stats['mu_weighted']:.3f} "
          + " ".join(f"{n}={stats[n]['m_hat']:.3f}" for n in REGION_NAMES) + f" crash={stats['crash_rate']:.3f}", flush=True)


def behaviour_clone_lqr(params, m: StrikeModel, env_fns: StrikeSAEnvFns, actor_cfg: ActorConfig, *, steps: int = 3000,
                        batch: int = 512, lr: float = 1e-3, seed: int = 5, target: Callable | None = None,
                        label: str = "the hover LQR"):
    """Fit the actor's deterministic output du so that base action + du is the hover-LQR backup (du = 0 for the
    residual backup) on pool states."""
    pools = env_fns.pools
    w = env_fns.mixture(1.0)
    n_pool = pools.shape[1]

    def loss(p, o, y):
        a = actor_mean_action(p, o, m.scale, actor_cfg, action_low=m.d_low, action_high=m.d_high)
        return jnp.mean(jnp.square((a - y) / m.scale))

    @jax.jit
    def step(p, opt, k):
        k1, k2 = jax.random.split(k)
        v = pools[jax.random.categorical(k1, jnp.log(w), shape=(batch,)), jax.random.randint(k2, (batch,), 0, n_pool)]
        x, s = v[:, :10], v[:, 10:]
        o = jax.vmap(lambda a, b: make_obs(m, a, b))(x, s)
        tgt = (lambda a, b: lqr_action(m, a)) if target is None else (lambda a, b: target(None, a, b))
        y = jnp.clip(jax.vmap(lambda a, b: tgt(a, b) - backup_base_action(m, a, b))(x, s), m.d_low, m.d_high)
        val, g = jax.value_and_grad(loss)(p, o, y)
        p, opt = adam_step(p, g, opt, lr)
        return p, opt, val

    opt, key, val = adam_init(params), jax.random.PRNGKey(seed), jnp.nan
    for _ in range(int(steps)):
        key, k = jax.random.split(key)
        params, opt, val = step(params, opt, k)
    print(f"[actor init] behaviour-cloned {label}: {steps} steps, final normalized MSE {float(val):.2e}", flush=True)
    return params


def _freeze_actor_until(update_fn: Callable, n_updates: int) -> Callable:
    """Critic-only phase: keep the actor (and its target) for the first n_updates critic updates."""
    if n_updates <= 0:
        return update_fn

    def update(state, batch, key):
        new, metrics, key_next = update_fn(state, batch, key)
        frozen = new["update_step"] <= jnp.int32(n_updates)
        keep = lambda a, b: jax.tree_util.tree_map(lambda u, o: jnp.where(frozen, o, u), a, b)
        new = dict(new)
        for k in ("actor_params", "actor_opt", "target_actor_params"):
            new[k] = keep(new[k], state[k])
        metrics = dict(metrics)
        metrics["actor_update_applied"] = jnp.where(frozen, 0.0, metrics["actor_update_applied"])
        return new, metrics, key_next

    return jax.jit(update)


def run_strike_sa_training(ra_cfg: LandingSAConfig, m: StrikeModel, rc: StrikeRegionConfig, *, output_dir: str | None = None,
                           actor_init: str = "lqr_bc", actor_start_update: int = 0, bc_steps: int = 3000) -> Dict[str, Any]:
    if not (0.0 < float(ra_cfg.beta) < 1.0):
        raise ValueError(f"beta must lie in (0, 1), got {ra_cfg.beta}")
    env_fns = build_strike_sa_env(m, rc, episode_max_steps=int(ra_cfg.episode_max_steps),
                                  terminate_on_goal=bool(ra_cfg.collector_terminate_on_goal))
    hidden = (int(ra_cfg.hidden_size), int(ra_cfg.hidden_size))
    actor_cfg = ActorConfig(obs_dim=env_fns.obs_dim, action_dim=4, hidden_sizes=hidden, log_std_min=ra_cfg.actor_log_std_min,
                            log_std_max=ra_cfg.actor_log_std_max, activation=ra_cfg.actor_activation)
    critic_cfg = SafeArrivalCriticConfig(obs_dim=env_fns.obs_dim, act_dim=4, hidden_sizes=hidden)
    goal_b, fail_b = jax.vmap(env_fns.goal_fn), jax.vmap(env_fns.fail_fn)

    key = make_prng_key(ra_cfg.seed)
    key, k_state, k_env = jax.random.split(key, 3)
    state = init_sa_state(k_state, actor_cfg, critic_cfg, ra_cfg)
    targets = {"flee_bc": (lambda: make_flee_backup(m), f"the flee-then-hover heuristic ({m.cfg.flee_speed} m/s)"),
               "base_bc": (lambda: (lambda p, x, sc: backup_base_action(m, x, sc)), f"its base action ({m.cfg.backup_action})")}
    if actor_init in targets:       # flee_bc: start from flee-then-hover; base_bc: du = 0 (the base action itself)
        make_t, label = targets[actor_init]
        state["actor_params"] = behaviour_clone_lqr(state["actor_params"], m, env_fns, actor_cfg, steps=bc_steps,
                                                    target=make_t(), label=label)
        state["target_actor_params"] = state["actor_params"]
        state["actor_opt"] = adam_init(state["actor_params"])
    elif actor_init == "lqr_bc":
        state["actor_params"] = behaviour_clone_lqr(state["actor_params"], m, env_fns, actor_cfg)
        state["target_actor_params"] = state["actor_params"]
        state["actor_opt"] = adam_init(state["actor_params"])
    elif actor_init != "random":
        raise ValueError(f"actor_init must be 'lqr_bc', 'flee_bc', 'base_bc' or 'random', got {actor_init!r}")
    replay = sa_replay_init(ra_cfg.replay_size, env_fns.obs_dim, 4)
    # the actor's action du is added to the base action (LQR or hover; box d_low..d_high)
    update_fn = _freeze_actor_until(build_sa_update_fn(ra_cfg, actor_cfg, m.scale, m.d_low, m.d_high, goal_b, fail_b, goal_b),
                                    int(actor_start_update))
    _, collect_fn = build_sa_action_fns(ra_cfg, actor_cfg, m.scale, m.d_low, m.d_high)

    env_keys = jax.random.split(k_env, int(ra_cfg.num_envs))
    s0 = jnp.asarray(float(ra_cfg.curriculum_start_scale), jnp.float32)
    env_state, obs = env_fns.reset_batched(env_keys, s0)
    window = int(max(1, ra_cfg.curriculum_window_episodes))
    loop_state = SALoopState(state=state, replay=replay, env_state=env_state, obs=obs, key=key, env_keys=env_keys,
                             global_step=jnp.int32(0), updates=jnp.int32(0), curriculum_scale=s0,
                             episode_count=jnp.int32(0), success_window=jnp.zeros((window,), jnp.float32),
                             success_window_size=jnp.int32(0), success_window_ptr=jnp.int32(0))

    beta, hd = float(ra_cfg.beta), float(ra_cfg.hoeffding_delta)
    run_eval = build_evaluator(m, rc, make_actor_backup(m, actor_cfg), beta=beta, hoeffding_delta=hd)
    lqr_eval = build_evaluator(m, rc, make_lqr_backup(m), beta=beta, hoeffding_delta=hd, sets=run_eval.sets)
    lqr_val, lqr_test = lqr_eval(None, "val"), lqr_eval(None, "test")
    initial = run_eval(state["actor_params"], "val")
    print("[eval] LQR backup (val): " + " ".join(f"{n}={lqr_val[n]['m_hat']:.3f}" for n in REGION_NAMES)
          + f" mu_w={lqr_val['mu_weighted']:.3f} | actor at start: mu_w={initial['mu_weighted']:.3f}", flush=True)
    hooks = SASystemHooks(episode_metric_fields=_EPISODE_METRIC_FIELDS, episode_bookkeeping=_episode_bookkeeping,
                          run_eval=run_eval, eval_rank_key=_eval_rank_key, append_eval_history=_append_eval_history,
                          episode_history_pairs=_EPISODE_HISTORY_PAIRS, history_keys=_HISTORY_KEYS)
    one_vec_step = build_sa_one_vec_step(env_fns=env_fns, collect_action_batch_fn=collect_fn, update_fn=update_fn,
                                         ra_cfg=ra_cfg, hooks=hooks)
    res = run_sa_training_loop(ra_cfg=ra_cfg, loop_state=loop_state, one_vec_step=one_vec_step, hooks=hooks)

    best_eval = res.best_eval_stats if res.best_eval_stats is not None else res.val_eval
    best_state = res.best_state if res.best_state is not None else snapshot_sa_state(res.loop_state.state)
    test_at_best = run_eval(best_state["actor_params"], "test")
    summary = {
        "training_objective": "discounted_safe_arrival", "task": "strike_phase1", "disturbance_in_training": False,
        "sa_backbone": sa_backbone(ra_cfg), "seed": int(ra_cfg.seed), "total_steps": int(ra_cfg.total_steps),
        "updates": int(jax.device_get(res.loop_state.updates)), "wall_time_sec": float(res.total_time),
        "jax_backend": jax.default_backend(), "actor_init": actor_init, "actor_start_update": int(actor_start_update),
        "final_curriculum_scale": float(jax.device_get(res.loop_state.curriculum_scale)),
        "lqr_val": lqr_val, "lqr_test": lqr_test, "untrained_val": initial, "final_val": res.val_eval,
        "best_val": best_eval, "best_eval_step": int(res.best_eval_step), "test_at_best": test_at_best,
        "test_at_final": res.test_eval,
    }
    configs = {"strike": m.cfg.as_dict(), "region": rc.as_dict(), "sa": asdict(ra_cfg), "actor": asdict(actor_cfg),
               "actor_init": actor_init, "actor_start_update": int(actor_start_update)}
    if output_dir is not None:
        with open(f"{output_dir}/summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=float)
        with open(f"{output_dir}/configs.json", "w", encoding="utf-8") as f:
            json.dump(configs, f, indent=2)
    return {"summary": summary, "history": res.history, "configs": configs, "actor_cfg": actor_cfg,
            "final_state": snapshot_sa_state(res.loop_state.state), "best_state": best_state}


__all__ = ["REGION_NAMES", "StrikeRegionConfig", "StrikeSAConfig", "behaviour_clone_lqr", "build_evaluator",
           "build_region_sampler", "build_strike_sa_env", "draw_region_states", "heldout_sets", "lqr_end_point",
           "make_actor_backup", "make_flee_backup", "make_lqr_backup", "make_rollout", "run_strike_sa_training"]
