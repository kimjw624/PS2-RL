"""Discrete safety filter for the strike task (backup-set check on the next state).

At every step, with the scene as it is now (all arena obstacles, each chaser as a static ball of radius
R = r_c + r_d + v_c T around its current position):

    candidates  u(lam) = lam * u_nom + (1 - lam) * u_backup(x),   lam in ``lambdas`` (1 first, 0 last)
    accept      the largest lam whose next state x+ = f(x, u(lam)) is recoverable: the backup, started at x+, reaches
                the base set within N = T / dt steps without leaving the safe set (first hit, the Phase-I criterion).
    otherwise   u_backup(x) (the certificate of the last step is lost: reported as ``no_certificate``).

Committed plan: the backup's input (the local scene, below) changes from step to step - slots re-sorted, obstacles
coming into range - and the network is not order-invariant, so the backup plan certified at the last step can change.
The filter therefore keeps the scene view of its last certificate ("committed"). Candidates are checked with the fresh
view and the full base set (chaser clearance with the hold time, StrikeConfig.chaser_hold); when none passes, the
committed plan is continued: lam = 0 with the committed view frozen as it was certified (obstacle slots and chaser balls
at their positions then), so the backup's input - and therefore its plan - is exactly the certified one. Safety and the
base set are checked against the chasers' current balls (their reach over the next T) with the true reach (hold 0).
In a static scene that continuation always passes; it fails only when a chaser's current ball actually reaches the
plan. ``no_certificate`` counts the steps where even the continuation fails (its action is applied anyway).
(Moving the ball slots of the committed view to the chasers' current positions perturbs the backup's input by a few
centimetres per step, and near obstacles at speed that alone flips its plan - the first version did that.)

The backup sees the local scene (the n_obstacle_slots nearest obstacles and the chaser balls, nearest first, frozen
for the rollout), limited to what Phase I trained it on: obstacles whose surface is within ``obstacle_range`` and balls
whose centre is within ``ball_range`` (the run's region dist_max / ball_dist_max; farther ones are left out of its input,
they cannot be reached within T anyway). Safety and base-set membership are checked against every arena obstacle.

Hover latch (default on): once a certified plan reaches the base set, the filter keeps that anchor a0 and continues with
the hover LQR at the fixed anchor, u = u* - K (p - a0, r), the controller the base-set certificate covers. (Re-anchoring
every step lets the anchor drift by millimetres; a state that entered B at zero margin then drops out of B and the
certificate was lost while hovering safely - 13 of 128 static scenes, one of which then hit an obstacle.)

Chaser model: "static" (above) or "reach" - the chaser's exact reachable ball r_c + r_d + v_c tau at rollout time tau,
and a hover point guaranteed for ``reach_hold`` seconds after arrival (see make_filter).

Backups: "lqr" (the re-anchored hover LQR) or "learned" (the Phase-I actor: hover LQR + residual, or as saved). The
backup is composed as in Phase I: the hover LQR inside the base set, the backup policy outside it (the actor was never
trained inside the base set - its episodes end at the hand-off).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp

from ps2rl.envs.strike_scene import (
    StrikeConfig,
    StrikeModel,
    anchor,
    arena_walls,
    attitude_chart,
    build_model,
    hover_chaser_margin,
    join_scene,
    lqr_action,
    margins_rel,
    obstacle_distances,
    plant_step,
    relative_scene,
    split_scene,
)
from ps2rl.cil.strike_guidance import GuidanceConfig, h_chasers, make_reflex, pursue, stop_point, wall_distances
from ps2rl.phase1_sa.strike_sa import make_actor_backup
from ps2rl.utils.policy import _coerce_actor_cfg

Array = jax.Array


class FilterInfo(NamedTuple):
    lam: Array             # chosen blend (1 = nominal kept)
    no_certificate: Array  # no candidate passed: backup action applied
    used_committed: Array  # certified only with the committed scene view
    u_backup: Array
    correction: Array      # |u - u_nom| / scale
    held: Array            # continued with the held hover (latched anchor); guidance model: the evasion reflex is active


def local_scene(m: StrikeModel, x: Array, obst: Array, chaser_p: Array, chaser_on: Array,
                obstacle_range: float = float("inf"), ball_range: float = float("inf"), return_order: bool = False):
    """The backup's scene: nearest n_obstacle_slots obstacles and the chaser balls, nearest first, within range
    (return_order: also the chaser index of each ball slot)."""
    cfg = m.cfg
    d = obstacle_distances(x[0:2], obst)
    idx = jnp.argsort(d)[: cfg.n_obstacle_slots]
    near = obst[idx]
    near = near.at[:, 6].set(jnp.where(d[idx] <= obstacle_range, near[:, 6], 0.0))
    dc = jnp.sqrt(jnp.sum((chaser_p - x[None, 0:3]) ** 2, axis=1))
    on = jnp.where(dc <= ball_range, chaser_on, 0.0)
    balls = jnp.concatenate([chaser_p, on[:, None]], axis=1)
    order = jnp.argsort(jnp.where(on > 0.5, dc, 1e9))
    view = join_scene(near, balls[order])
    return (view, order) if return_order else view


def full_margins(m: StrikeModel, x: Array, obst: Array, balls: Array, hold: float | None = None):
    o_rel, b_rel = relative_scene(x, obst, balls)
    return margins_rel(m, x[2] - m.cfg.z_floor, x[3:6], x[6:10], o_rel, b_rel, hold)


def backup_ranges(path: str) -> tuple[float, float]:
    """(obstacle_range, ball_range) a Phase-I run was trained on (its region dist_max / ball_dist_max)."""
    p = Path(path).expanduser()
    run = p if p.is_dir() else p.parent
    reg = json.loads((run / "configs.json").read_text())["region"]
    return float(reg["dist_max"]), float(reg["ball_dist_max"])


def load_backup(m: StrikeModel, kind: str, path: str | None = None):
    """(backup(params, x, scene_local) -> u, params). ``path``: Phase-I run directory or actor .pkl (kind "learned")."""
    if kind == "lqr":
        return (lambda params, x, scene: lqr_action(m, x)), None
    if kind != "learned":
        raise ValueError(f"backup must be 'lqr' or 'learned', got {kind!r}")
    p = Path(path).expanduser()
    if p.is_dir():
        p = p / "strike_backup_policy_actor.pkl"
    with open(p, "rb") as f:
        ck = pickle.load(f)
    saved = ck.get("metadata", {}).get("strike_config", {})
    if saved and StrikeConfig.from_dict(saved) != m.cfg:
        raise ValueError(f"{p}: the backup was trained with a different strike config")
    return make_actor_backup(m, _coerce_actor_cfg(ck["actor_cfg"])), ck["actor_params"]


def with_chaser(m: StrikeModel, radius: float, speed: float) -> StrikeModel:
    """The model the filter uses against a chaser of this radius and top speed (guidance margin h_c, pure-pursuit
    prediction, reflex speed, static balls). The learned backup keeps the model it was trained with (its observation)."""
    if abs(radius - m.cfg.chaser_radius) < 1e-12 and abs(speed - m.cfg.chaser_speed) < 1e-12:
        return m
    return build_model(m.cfg.replace(chaser_radius=float(radius), chaser_speed=float(speed)))


def model_from_backup(path: str) -> StrikeModel:
    """The strike model a Phase-I run was trained with (its configs.json)."""
    p = Path(path).expanduser()
    run = p if p.is_dir() else p.parent
    return build_model(StrikeConfig.from_dict(json.loads((run / "configs.json").read_text())["strike"]))


def make_filter(m: StrikeModel, backup: Callable, params, *, lambdas: Sequence[float] = (1.0, 0.75, 0.5, 0.25, 0.0),
                obstacle_range: float = float("inf"), ball_range: float = float("inf"), fresh_hold: float | None = None,
                hover_latch: bool = True, chaser_model: str = "static", reach_hold: float | None = None,
                guidance: GuidanceConfig | None = None, arena: tuple | None = None) -> Callable:
    """filt(x, u_nom, obst, chaser_p, chaser_on, committed) -> (u, FilterInfo, committed') for one vehicle (vmap it).

    ``fresh_hold``: chaser hold time [s] for certifying new plans (default StrikeConfig.chaser_hold, what the backup was
    trained with); the committed continuation always uses the true reach (hold 0). Static chaser model only.

    ``hover_latch``: once a certified plan reaches the base set, keep its anchor and continue with the hover LQR at that
    fixed anchor, u = u* - K (p - a0, r) - the controller the base-set certificate is about ({V(p - a0, r) <= c_B} is
    invariant under it). Without the latch the continuation re-anchors every step (u = u* - K~ r), the anchor drifts by
    millimetres, and a state that entered B at zero margin can drop out of it: the certificate is lost while hovering
    safely and the actor (never trained inside B) takes over.

    ``chaser_model``:
      "static" - each chaser is a ball of radius R = r_c + r_d + v_c T around its current position for the whole rollout
                 (base set: anchor clear of R + v_c hold + rho);
      "reach"  - the chaser's exact reachable ball: at rollout time tau the drone must stay out of r_c + r_d + v_c tau
                 around the chaser's current position, and the hover point must stay out of r_c + r_d + v_c (tau + H)
                 + rho, H = ``reach_hold`` (default T): the hover is guaranteed for H seconds after arrival. The check is
                 time-consistent (the tail of a certified plan stays certified at the next step); a held hover is
                 certified while its anchor is out of r_c + r_d + v_c dt + rho, i.e. its H seconds run down.
      "guidance" - the guidance-law escape set (ps2rl/cil/strike_guidance.py): the safe set holds h_c >= 0 (the drone
                 can still out-accelerate a pursuer closing at v_c) instead of the ball, the backup carries an evasion
                 reflex, the chasers are propagated as pure pursuers, and a candidate is certified when the rollout of
                 ``guidance.horizon`` s stays safe (obstacles, floor, arena walls ``arena`` = (x_min, x_max, y_half,
                 z_max), h_c >= 0) and ends with h_c >= 0 where the drone can hover clear (base set) or, mid-escape,
                 brake clear at a_evade. New plans keep extra slack (guidance.fresh_margin on h_c,
                 fresh_obstacle_margin from obstacles / walls), the committed plan only the safe set. With
                 guidance.lqr_alternative the hover LQR is tried as a second backup (walls / ceiling, which the learned
                 backup never saw). Hover latch on. A backup trained with moving chasers and walls (Phase I with
                 strike chaser_set "guidance", arena_walls) sees the chasers' current positions and the arena walls
                 in every rollout (``backup_scene_fn``); with guidance.learned_alone it is also tried by itself, first,
                 certified by its Phase-I criterion (base set with the chasers reached within N = T / dt steps inside
                 the safe set).

    ``committed``: ``fresh_view(x, obst, chaser_p, chaser_on)`` at a reset (the backup's scene view of the last
    certificate, then the held anchor and the latch flag); ``backup_view(committed)`` returns the view."""
    cfg = m.cfg
    n = int(cfg.num_steps)
    dt = float(cfg.dt)
    if chaser_model not in ("static", "reach", "guidance"):
        raise ValueError(f"chaser_model must be 'static', 'reach' or 'guidance', got {chaser_model!r}")
    reach = chaser_model == "reach"
    r_contact = float(cfg.chaser_radius + cfg.r_drone)
    v_c = float(cfg.chaser_speed)
    H = float(cfg.T if reach_hold is None else reach_hold)
    lams = jnp.asarray(sorted(lambdas, reverse=True), jnp.float32)
    if float(lams[-1]) != 0.0:
        lams = jnp.concatenate([lams, jnp.zeros(1)])
    nv = cfg.n_obstacle_slots * 7 + cfg.n_chaser_slots * 4
    K_full = jnp.asarray(m.lqr.K, jnp.float32)
    P_full = jnp.asarray(m.lqr.P, jnp.float32)
    if chaser_model == "guidance":
        return _make_guidance_filter(m, backup, params, lams, nv, K_full, P_full, obstacle_range, ball_range,
                                     guidance or GuidanceConfig(), arena, hover_latch=hover_latch)

    def chaser_dist(point, balls):
        d = jnp.sqrt(jnp.sum((balls[:, 0:3] - point[None, :]) ** 2, axis=1) + 1e-12)
        return jnp.min(jnp.where(balls[:, 3] > 0.5, d, 1e9))

    def margins_at(x, obst, balls, hold, tau):
        """(safe, base) margins of x at rollout time tau [s from now]."""
        if not reach:
            return full_margins(m, x, obst, balls, hold)
        s, b = full_margins(m, x, obst, balls.at[:, 3].set(0.0))           # obstacles and floor
        s = jnp.minimum(s, chaser_dist(x[0:3], balls) - (r_contact + v_c * tau))
        b = jnp.minimum(b, chaser_dist(anchor(m, x), balls) - (r_contact + v_c * (tau + H)) - m.lqr.rho)
        return s, b

    def composed(x, scene, obst, balls, hold, tau):
        s, b = margins_at(x, obst, balls, hold, tau)
        return jnp.where((s >= 0.0) & (b >= 0.0), lqr_action(m, x), backup(params, x, scene))

    def in_base(x, obst, balls, hold, tau):
        s, b = margins_at(x, obst, balls, hold, tau)
        return (s >= 0.0) & (b >= 0.0)

    def recoverable(x0, scene, obst, balls, hold):
        """x0 is the next state (tau = dt): the composed backup reaches B within n steps, safe until then (first hit)."""
        def ok(x, tau):
            s, b = margins_at(x, obst, balls, hold, tau)
            return s >= 0.0, (s >= 0.0) & (b >= 0.0)

        safe0, inb0 = ok(x0, dt)

        def body(c, k):
            x, hb, hf = c
            tau = (k + 1.0) * dt
            xn = plant_step(m, x, composed(x, scene, obst, balls, hold, tau))
            act = ~(hb | hf)
            s, b = ok(xn, tau + dt)
            return (xn, hb | (act & b), hf | (act & ~s)), None

        (_, hb, hf), _ = jax.lax.scan(body, (x0, inb0, ~safe0), jnp.arange(n, dtype=jnp.float32))
        return hb & ~hf

    def held(x, a0, obst, balls):
        """Hover LQR at the fixed anchor a0 and whether it is certified from x (static obstacles, chasers' current reach)."""
        e = jnp.concatenate([x[0:3] - a0, x[3:6], attitude_chart(x[6:10])])
        u = jnp.clip(m.u_hover - K_full @ e, m.low, m.high)
        xn = plant_step(m, x, u)
        en = jnp.concatenate([xn[0:3] - a0, xn[3:6], attitude_chart(xn[6:10])])
        need_c = (r_contact + v_c * dt if reach else cfg.ball_radius) + m.lqr.rho
        ok = ((en @ P_full @ en <= cfg.c_B)
              & (jnp.min(obstacle_distances(a0[0:2], obst)) - m.lqr.rho_xy - cfg.r_drone >= 0.0)
              & (a0[2] - cfg.z_floor - m.lqr.z_ext >= 0.0)
              & (chaser_dist(a0, balls) - need_c >= 0.0))
        return u, ok

    def fresh_view(x, obst, chaser_p, chaser_on):
        view = local_scene(m, x, obst, chaser_p, chaser_on, obstacle_range, ball_range)
        return jnp.concatenate([view, jnp.zeros(4, view.dtype)])

    def check(x, u_nom, scene, obst, balls, hold):
        ub = composed(x, scene, obst, balls, hold, 0.0)
        cands = jnp.clip(lams[:, None] * u_nom[None, :] + (1.0 - lams[:, None]) * ub[None, :], m.low, m.high)
        nxt = jax.vmap(lambda u: plant_step(m, x, u))(cands)
        good = jax.vmap(lambda xx: recoverable(xx, scene, obst, balls, hold))(nxt)
        i = jnp.argmax(good)
        return jnp.any(good), cands[i], lams[i], ub

    hold_f = fresh_hold                         # static model: new plans with the hold, continuation with the true reach
    hold_c = 0.0

    def filt(x, u_nom, obst, chaser_p, chaser_on, committed):
        balls = jnp.concatenate([chaser_p, chaser_on[:, None]], axis=1)       # safety: every chaser
        view, a0 = committed[:nv], committed[nv:nv + 3]
        latched = (committed[nv + 3] > 0.5) & hover_latch
        fresh = fresh_view(x, obst, chaser_p, chaser_on)[:nv]
        ok_f, u_f, lam_f, ub_f = check(x, u_nom, fresh, obst, balls, hold_f)
        # continuation: the held hover, or the committed plan with its frozen view (the backup's input as certified)
        u_h, ok_h = held(x, a0, obst, balls)
        ub_c = composed(x, view, obst, balls, hold_c, 0.0)
        ok_c = recoverable(plant_step(m, x, ub_c), view, obst, balls, hold_c)
        u_cont = jnp.where(latched, u_h, ub_c)
        ok_cont = jnp.where(latched, ok_h, ok_c)
        u = jnp.where(ok_f, u_f, u_cont)
        lam = jnp.where(ok_f, lam_f, 0.0)
        # latch: the next state is in B (the hit the certified plan starts with) -> hold its anchor from then on
        xn = plant_step(m, x, u)
        in_b = jnp.where(ok_f, in_base(xn, obst, balls, hold_f, dt), in_base(xn, obst, balls, hold_c, dt))
        keep = ~ok_f & latched
        new_latched = jnp.where(keep, True, in_b)
        new_a0 = jnp.where(keep, a0, jnp.where(in_b, anchor(m, xn), a0))
        new_view = jnp.where(ok_f, fresh, view)
        new_comm = jnp.concatenate([new_view, new_a0, new_latched.astype(view.dtype)[None]])
        info = FilterInfo(lam=lam, no_certificate=~(ok_f | ok_cont), used_committed=~ok_f & ok_cont,
                          u_backup=jnp.where(ok_f, ub_f, u_cont), correction=jnp.sqrt(jnp.sum(((u - u_nom) / m.scale) ** 2)),
                          held=~ok_f & latched)
        return u, info, new_comm

    filt.fresh_view = fresh_view
    filt.backup_view = lambda committed: committed[..., :nv]
    return filt


def backup_scene_fn(m: StrikeModel, arena: tuple | None) -> Callable:
    """scene(view, order, chaser_p) -> the learned backup's scene from a committed view (obstacle slots, chaser slots in
    ``order``). A backup trained with moving chasers (StrikeConfig.chaser_set "guidance") sees its chaser slots at the
    chasers' current positions (as in its Phase-I episodes); one trained with arena walls gets the arena's walls. Static
    balls without walls: the view as it is (the original backups)."""
    cfg = m.cfg
    if cfg.arena_walls and arena is None:
        raise ValueError("the backup was trained with arena walls: pass arena = (x_min, x_max, y_half, z_max)")
    walls = arena_walls(arena) if cfg.arena_walls else None

    def scene(view, order, chp):
        if not cfg.moving_chasers and walls is None:
            return view
        obst, balls = split_scene(cfg, view)
        if cfg.moving_chasers:
            balls = balls.at[:, 0:3].set(chp[order])
        return join_scene(obst, balls, walls)

    return scene


def _make_guidance_filter(m: StrikeModel, backup: Callable, params, lams, nv: int, K_full, P_full, obstacle_range: float,
                          ball_range: float, g: GuidanceConfig, arena: tuple | None, hover_latch: bool = True) -> Callable:
    """The filter with the guidance-law escape set (see make_filter, chaser_model "guidance"). hover_latch False (an
    ablation): the hover anchor is never held, the hover is the re-anchored LQR at every step."""
    cfg = m.cfg
    if arena is None:
        raise ValueError("chaser_model 'guidance' needs arena = (x_min, x_max, y_half, z_max)")
    arena = tuple(float(v) for v in arena)
    if cfg.moving_chasers and (abs(g.a_evade - cfg.chaser_a_evade) > 1e-9 or abs(g.delay - cfg.chaser_delay) > 1e-9):
        raise ValueError(f"guidance a_evade / delay ({g.a_evade}, {g.delay}) must equal the backup's escape margin "
                         f"(strike chaser_a_evade {cfg.chaser_a_evade}, chaser_delay {cfg.chaser_delay})")
    if g.learned_alone and not cfg.moving_chasers:
        raise ValueError("guidance learned_alone needs a backup trained with moving chasers (strike chaser_set 'guidance')")
    n = int(round(g.horizon / cfg.dt))
    n_first = int(cfg.num_steps)                # the learned-alone variant: Phase I's C_N horizon N = T / dt
    n_ch = int(cfg.n_chaser_slots)
    scene_of = backup_scene_fn(m, arena)
    reflex = make_reflex(m, g, arena)
    # backup variants (kind, hover-LQR base, escape reflex), the filter takes the largest certified blend over all of them:
    #   "alone"  the learned backup by itself, composed as in Phase I (hover in the base set with the chasers, else the
    #            backup) and certified by the Phase-I criterion (base set reached within N steps inside the safe set);
    #   "blend"  learned or hover-LQR base blended with an escape reflex, certified over guidance.horizon with the
    #            terminal condition below.
    variants = ([("alone", False, None)] if g.learned_alone else []) + [("blend", False, reflex)] + \
               ([("blend", True, reflex)] if g.lqr_alternative else []) + \
               [("blend", False, make_reflex(m, g, arena, sr, ang)) for sr, ang in g.escape_variants]
    n_var = len(variants)
    k_lqr = next((i for i, v in enumerate(variants) if v[1]), -1)
    no_balls = jnp.zeros((cfg.n_chaser_slots, 4), jnp.float32)

    def wall_margin(p, pad):
        return jnp.minimum(jnp.min(wall_distances(arena, p[0:2])) - pad, arena[3] - p[2] - pad)

    def safe_margin(x, chp, on, obst, h_min):
        """h_min = 0: the safe set; h_min > 0 (new plans): h_c >= fresh_margin and fresh_obstacle_margin of extra
        clearance from obstacles and walls."""
        o_min = jnp.where(h_min > 0.0, g.fresh_obstacle_margin, 0.0)
        so, _ = full_margins(m, x, obst, no_balls)                                 # obstacles and floor
        s = jnp.minimum(jnp.minimum(so, x[2] - cfg.z_floor) - 0.0, so - o_min)
        s = jnp.minimum(s, wall_margin(x[0:3], cfg.r_drone) - o_min)
        return jnp.minimum(s, jnp.min(h_chasers(m, g, x[0:3], x[3:6], chp, on)) - h_min)

    def in_b_obst(x, obst):
        """Base set without chasers (hover ellipsoid clear of obstacles, floor and walls)."""
        s, b = full_margins(m, x, obst, no_balls)
        b = jnp.minimum(b, wall_margin(anchor(m, x), m.lqr.rho_xy + cfg.r_drone))
        return (s >= 0.0) & (b >= 0.0)

    def err(x, a0):
        return jnp.concatenate([x[0:3] - a0, x[3:6], attitude_chart(x[6:10])])

    def held_u(x, a0):
        return jnp.clip(m.u_hover - K_full @ err(x, a0), m.low, m.high)

    def level(x, a0):
        e = err(x, a0)
        return e @ P_full @ e

    def ch_ok(a, chp, on, lv):
        return jnp.min(jnp.where(on > 0.5, hover_chaser_margin(m, chp - a[None, :], lv), 1e3)) >= 0.0

    def in_B(x, chp, on, obst, hit, a0):
        """Phase-I base set with the chasers: the latched hover while its level and chaser margin hold, else the base set
        at the state's own anchor. -> (in B, anchor)."""
        lv_h = level(x, a0)
        held_ok = hit & (lv_h <= cfg.c_B) & ch_ok(a0, chp, on, lv_h) & hover_latch
        r = jnp.concatenate([x[3:6], attitude_chart(x[6:10])])
        a_x = anchor(m, x)
        fresh_ok = in_b_obst(x, obst) & ch_ok(a_x, chp, on, r @ m.S @ r)
        return held_ok | fresh_ok, jnp.where(held_ok, a0, a_x)

    def action(x, chp, on, sv, obst, hit, a0, k):
        """Composed backup of variant k (sv = (view, chaser order)). "blend": the held hover in B (else the learned backup,
        or the hover LQR), blended with the variant's escape command by the reflex weight (the hover anchor is given up
        once the reflex acts). "alone": the hover in the Phase-I base set (with the chasers), else the learned backup."""
        kind, alt, rfx = variants[k]
        u_learned = backup(params, x, scene_of(sv[0], sv[1], chp))
        if kind == "alone":
            inb, a0_n = in_B(x, chp, on, obst, hit, a0)
            u = jnp.where(inb, held_u(x, a0_n), u_learned)
            return u, inb, jnp.where(inb, a0_n, a0), jnp.zeros((), x.dtype)
        u_r, sig = rfx(x, chp, on, obst)
        inb = hit | in_b_obst(x, obst)
        a0_n = jnp.where(hit & hover_latch, a0, anchor(m, x))
        u_base = jnp.where(inb, held_u(x, a0_n), lqr_action(m, x) if alt else u_learned)
        u = jnp.clip((1.0 - sig) * u_base + sig * u_r, m.low, m.high)
        hit_n = (sig <= 0.0) & inb
        return u, hit_n, jnp.where(hit_n, a0_n, a0), sig

    def init_carry(x0, prior_hit, prior_a0, obst):
        keep = prior_hit & (level(x0, prior_a0) <= cfg.c_B) & hover_latch
        hit0 = keep | in_b_obst(x0, obst)
        return hit0, jnp.where(keep, prior_a0, anchor(m, x0))

    def certified(x0, chc0, ch0, on, sv, obst, hit0, a00, h_min, k):
        """(certified, score): score = the worst safe margin along the rollout and h_c at its end (best effort).
        chc0: the chasers as the plan's controls see them (its own prediction), ch0: as the safety checks see them (both
        propagated as pure pursuers; equal except for a committed plan with guidance.committed_prediction)."""
        if variants[k][0] == "alone":
            return certified_first_hit(x0, chc0, ch0, on, sv, obst, hit0, a00, h_min, k)

        def body(c, _):
            x, chc, chp, hit, a0, worst = c
            u, hit, a0, _ = action(x, chc, on, sv, obst, hit, a0, k)
            xn = plant_step(m, x, u)
            chn = pursue(m, chp, x[0:3])
            return (xn, pursue(m, chc, x[0:3]), chn, hit, a0, jnp.minimum(worst, safe_margin(xn, chn, on, obst, h_min))), None

        worst0 = safe_margin(x0, ch0, on, obst, h_min)
        (xN, _, chN, hit, a0, worst), _ = jax.lax.scan(body, (x0, chc0, ch0, hit0, a00, worst0), None, length=n)
        score = jnp.minimum(worst, jnp.min(h_chasers(m, g, xN[0:3], xN[3:6], chN, on)) - h_min)
        return (worst >= 0.0) & terminal(xN, chN, on, obst, hit, a0, h_min), score

    def certified_first_hit(x0, chc0, ch0, on, sv, obst, hit0, a00, h_min, k):
        """The Phase-I criterion: the composed backup reaches the base set (with the chasers) within N steps and stays
        in the safe set until then (score: the worst margin until then, -1 if B is not reached)."""
        def body(c, _):
            x, chc, chp, hit, a0, done, worst = c
            u, hit, a0, _ = action(x, chc, on, sv, obst, hit, a0, k)
            xn = plant_step(m, x, u)
            chn = pursue(m, chp, x[0:3])
            worst = jnp.where(done, worst, jnp.minimum(worst, safe_margin(xn, chn, on, obst, h_min)))
            done = done | in_B(xn, chn, on, obst, hit, a0)[0]
            return (xn, pursue(m, chc, x[0:3]), chn, hit, a0, done, worst), None

        done0 = in_B(x0, ch0, on, obst, hit0, a00)[0]
        worst0 = safe_margin(x0, ch0, on, obst, h_min)
        (_, _, _, _, _, done, worst), _ = jax.lax.scan(body, (x0, chc0, ch0, hit0, a00, done0, worst0), None, length=n_first)
        return done & (worst >= 0.0), jnp.where(done, worst, jnp.minimum(worst, -1.0))

    def terminal(xN, chN, on, obst, hit, a0, h_min=0.0):
        """Ends where it can hover clear (held anchor or base set) or brake clear (mid-escape), with h_c >= 0."""
        hover_ok = jnp.where(hit, level(xN, a0) <= cfg.c_B, in_b_obst(xN, obst))
        ps = stop_point(g, xN[0:3], xN[3:6])
        brake_ok = (jnp.min(obstacle_distances(ps, obst)) >= cfg.r_drone) & (jnp.min(wall_distances(arena, ps)) >= cfg.r_drone)
        return (hover_ok | brake_ok) & (jnp.min(h_chasers(m, g, xN[0:3], xN[3:6], chN, on)) >= h_min)

    def fresh_parts(x, obst, chaser_p, chaser_on):
        return local_scene(m, x, obst, chaser_p, chaser_on, obstacle_range, ball_range, return_order=True)

    def fresh_view(x, obst, chaser_p, chaser_on):
        view, order = fresh_parts(x, obst, chaser_p, chaser_on)
        return jnp.concatenate([view, jnp.zeros(5, view.dtype), order.astype(view.dtype), chaser_p.reshape(-1)])

    def comm_order(committed):
        return jnp.clip(jnp.round(committed[nv + 5:nv + 5 + n_ch]).astype(jnp.int32), 0, n_ch - 1)

    def comm_chasers(committed):
        return committed[nv + 5 + n_ch:nv + 5 + 4 * n_ch].reshape(n_ch, 3)

    n_l = int(lams.shape[0])

    def filt(x, u_nom, obst, chaser_p, chaser_on, committed):
        """committed: [backup view, held anchor (3), latch flag, backup variant index, chaser slot order, the plan's chaser
        prediction (C, 3)]."""
        view, prior_a0 = committed[:nv], committed[nv:nv + 3]
        k_c = jnp.clip(jnp.round(committed[nv + 4]).astype(jnp.int32), 0, n_var - 1)
        order_c = comm_order(committed)
        # the committed plan's controls see the chasers where its certificate predicted them (exact replay), or the
        # chasers as they are now (guidance.committed_prediction off: the earlier behaviour)
        chc_c = comm_chasers(committed) if g.committed_prediction else chaser_p
        hit_x, a0_x = init_carry(x, committed[nv + 3] > 0.5, prior_a0, obst)
        fresh, order_f = fresh_parts(x, obst, chaser_p, chaser_on)
        sv_f, sv_c = (fresh, order_f), (view, order_c)
        ch1 = pursue(m, chaser_p, x[0:3])
        chc1_c = pursue(m, chc_c, x[0:3])

        def cert_from(u, sv, h_min, k, chc1=ch1):
            x1 = plant_step(m, x, u)
            h0, a00 = init_carry(x1, hit_x, a0_x, obst)
            return certified(x1, chc1, ch1, chaser_on, sv, obst, h0, a00, h_min, k)

        ubs, sigs, cands, kinds, lam_all, goods, scores = [], [], [], [], [], [], []
        for k in range(n_var):
            ub, _, _, sig = action(x, chaser_p, chaser_on, sv_f, obst, hit_x, a0_x, k)
            ck = jnp.clip(lams[:, None] * u_nom[None, :] + (1.0 - lams[:, None]) * ub[None, :], m.low, m.high)
            gk, sk = jax.vmap(lambda u: cert_from(u, sv_f, g.fresh_margin, k))(ck)
            ubs.append(ub); sigs.append(sig); cands.append(ck); goods.append(gk); scores.append(sk)
            kinds.append(jnp.full((n_l,), k, jnp.int32)); lam_all.append(lams)
        cands, kinds, lam_all = jnp.concatenate(cands), jnp.concatenate(kinds), jnp.concatenate(lam_all)
        good, scores = jnp.concatenate(goods), jnp.concatenate(scores)
        # ties (same blend certified by several variants): the earlier variant, or the hover LQR if preferred
        tie = 1e-4 * (n_var - kinds).astype(jnp.float32)
        if g.prefer_lqr_on_ties and k_lqr >= 0:
            tie = tie + jnp.where(kinds == k_lqr, 1e-3, 0.0)
        i = jnp.argmax(jnp.where(good, lam_all + tie, -1.0))                   # largest certified blend
        ok_f = jnp.any(good)
        # continuation: the committed variant with its frozen view
        conts = [action(x, chc_c, chaser_on, sv_c, obst, hit_x, a0_x, k) for k in range(n_var)]
        u_cs = jnp.stack([c[0] for c in conts])
        sig_cs = jnp.stack([c[3] for c in conts])
        okc, scc = zip(*[cert_from(u_cs[k], sv_c, 0.0, k, chc1_c) for k in range(n_var)])
        u_c, sig_c = u_cs[k_c], sig_cs[k_c]
        ok_c, score_c = jnp.stack(okc)[k_c], jnp.stack(scc)[k_c]
        # no certificate at all: the candidate (or the committed plan) with the largest worst-case margin
        j = jnp.argmax(scores)
        use_j = ~ok_f & ~ok_c & (scores[j] > score_c) & g.best_effort_fallback
        take_f = ok_f | use_j
        i = jnp.where(ok_f, i, j)
        u = jnp.where(take_f, cands[i], u_c)
        h1, a01 = init_carry(plant_step(m, x, u), hit_x, a0_x, obst)
        new_k = jnp.where(take_f, kinds[i], k_c)
        new_comm = jnp.concatenate([jnp.where(take_f, fresh, view), a01, h1.astype(view.dtype)[None],
                                    new_k.astype(view.dtype)[None], jnp.where(take_f, order_f, order_c).astype(view.dtype),
                                    jnp.where(take_f, ch1, chc1_c).reshape(-1)])
        ub_f = jnp.stack(ubs)[kinds[i]]
        info = FilterInfo(lam=jnp.where(ok_f, lam_all[i], 0.0), no_certificate=~(ok_f | ok_c), used_committed=~ok_f & ok_c,
                          u_backup=jnp.where(ok_f, ub_f, u_c), correction=jnp.sqrt(jnp.sum(((u - u_nom) / m.scale) ** 2)),
                          held=jnp.where(ok_f, jnp.stack(sigs)[kinds[i]], sig_c) > 0.0)
        return u, info, new_comm

    def explain(x, obst, chaser_p, chaser_on, committed, k: int = 0):
        """Variant k's own plan from x (lam = 0, fresh view), step by step: states, chasers, (obstacle/floor, wall,
        h_c) margins, reflex weight, held flag; and the terminal terms (terminal / base set reached, h_c at the end)."""
        hit_x, a0_x = init_carry(x, committed[nv + 3] > 0.5, committed[nv:nv + 3], obst)
        sv = fresh_parts(x, obst, chaser_p, chaser_on)
        ub, _, _, _ = action(x, chaser_p, chaser_on, sv, obst, hit_x, a0_x, k)
        x1 = plant_step(m, x, ub)
        h0, a00 = init_carry(x1, hit_x, a0_x, obst)

        def body(c, _):
            xx, chp, hit, a0 = c
            u, hit, a0, sig = action(xx, chp, chaser_on, sv, obst, hit, a0, k)
            xn = plant_step(m, xx, u)
            chn = pursue(m, chp, xx[0:3])
            so, _ = full_margins(m, xn, obst, no_balls)
            rec = (xn, chn, jnp.stack([so, wall_margin(xn[0:3], cfg.r_drone),
                                       jnp.min(h_chasers(m, g, xn[0:3], xn[3:6], chn, chaser_on))]), sig, hit)
            return (xn, chn, hit, a0), rec

        steps = n_first if variants[k][0] == "alone" else n
        (xN, chN, hit, a0), rec = jax.lax.scan(body, (x1, ch_next(x, chaser_p), h0, a00), None, length=steps)
        end_ok = (in_B(xN, chN, chaser_on, obst, hit, a0)[0] if variants[k][0] == "alone"
                  else terminal(xN, chN, chaser_on, obst, hit, a0))
        return rec, end_ok, jnp.min(h_chasers(m, g, xN[0:3], xN[3:6], chN, chaser_on))

    def ch_next(x, chaser_p):
        return pursue(m, chaser_p, x[0:3])

    def candidate(x, u_nom, obst, chaser_p, chaser_on, committed, k: int, lam, h_min=None):
        """Candidate (variant k, blend lam) exactly as the fresh check runs it (for figures): first step
        u = clip(lam u_nom + (1 - lam) pi_b^(k)(x)), then variant k's backup over the horizon, the chasers as pure
        pursuers. Returns the rollout (states x_1..x_H, chasers, safe margin with the fresh-plan slack, reflex weight,
        held flag), the path verdict, the terminal verdict and how the plan ends (hover or braking point)."""
        h_m = g.fresh_margin if h_min is None else h_min
        hit_x, a0_x = init_carry(x, committed[nv + 3] > 0.5, committed[nv:nv + 3], obst)
        sv = fresh_parts(x, obst, chaser_p, chaser_on)
        ub, _, _, _ = action(x, chaser_p, chaser_on, sv, obst, hit_x, a0_x, k)
        u = jnp.clip(lam * u_nom + (1.0 - lam) * ub, m.low, m.high)
        x1 = plant_step(m, x, u)
        ch1 = pursue(m, chaser_p, x[0:3])
        h0, a00 = init_carry(x1, hit_x, a0_x, obst)

        def body(c, _):
            xx, chp, hit, a0 = c
            uu, hit, a0, sig = action(xx, chp, chaser_on, sv, obst, hit, a0, k)
            xn = plant_step(m, xx, uu)
            chn = pursue(m, chp, xx[0:3])
            return (xn, chn, hit, a0), (xn, chn, safe_margin(xn, chn, chaser_on, obst, h_m), sig, hit)

        steps = n_first if variants[k][0] == "alone" else n
        (xN, chN, hit, a0), (xs, chs, margin, sig, held) = jax.lax.scan(body, (x1, ch1, h0, a00), None, length=steps)
        worst = jnp.minimum(safe_margin(x1, ch1, chaser_on, obst, h_m), jnp.min(margin))
        hover_ok = jnp.where(hit, level(xN, a0) <= cfg.c_B, in_b_obst(xN, obst))
        return {"u": u, "x1": x1, "ch1": ch1, "states": xs, "chasers": chs, "margin": margin, "sigma": sig, "held": held,
                "path_ok": worst >= 0.0, "terminal_ok": terminal(xN, chN, chaser_on, obst, hit, a0, h_m),
                "hover_end": hover_ok, "anchor_end": jnp.where(hit, a0, anchor(m, xN)),
                "brake_point": stop_point(g, xN[0:3], xN[3:6])}

    filt.candidate = candidate
    filt.lambdas = [float(v) for v in lams]
    filt.fresh_view = fresh_view
    filt.backup_view = lambda committed: committed[..., :nv]
    filt.horizon_steps = n
    filt.explain = explain
    filt.variants = [(kind, bool(alt)) for kind, alt, _ in variants]
    # readable names of the backup variants and where the committed vector keeps the variant of the plan (for plots)
    filt.variant_labels = ((["learned alone"] if g.learned_alone else [])
                           + [f"learned + escape ({g.escape_speed_ratio:g} v_c, 0 deg)"]
                           + ([f"hover LQR + escape ({g.escape_speed_ratio:g} v_c, 0 deg)"] if g.lqr_alternative else [])
                           + [f"learned + escape ({sr:g} v_c, {ang:+g} deg)" for sr, ang in g.escape_variants])
    filt.k_index = nv + 4
    return filt


__all__ = ["FilterInfo", "backup_ranges", "backup_scene_fn", "full_margins", "load_backup", "local_scene", "make_filter",
           "model_from_backup", "with_chaser"]
