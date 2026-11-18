"""Minimal-correction backup-CBF filter for the strike task: the PS2-RL backup-CBF QP on the strike sets, with the
discrete backup-set filter (``strike_filter.make_filter``) as its safeguard.

Each step, for the state x and the nominal command u_nom:

1. Rows (backup CBF with the sensitivity flow). The backup flow phi(tau, x) is rolled forward over the horizon with
   the discrete sensitivity Q_tau = d phi(tau, x) / d x (exact Jacobian of the Euler rollout, propagated only on
   [f(x), g(x)]; ``ps2rl.cil.backup_cbf.rollout_backup_flow_and_sensitivity_with_info``). At every ``node_stride``-th
   node and for every safe-set function h_j

       d/dt h_j(phi(tau_i, x(t))) >= -alpha h_j(phi(tau_i, x))
       <=>  -dh_j Q_i g(x) u  <=  alpha h_j + dh_j (Q_i f(x) - f_pi(phi_i))           (relative-time form)

   and at the end of the horizon the same with the base-set functions and ``base_alpha``.
2. QP (the shared engine): min sum_k w_k (u_k - u_nom,k)^2 + slack_weight s^2 subject to the rows (softened by one
   slack s >= 0) and the action box. The change is the smallest one, in the weighted metric, that keeps the backup's
   whole future safe to first order - so a command is bent only in the direction the margins are sensitive to.
3. Discrete safeguard: the discrete filter with u_QP as its nominal - the largest lambda in
   lambda u_QP + (1 - lambda) u_backup whose next state is recoverable (rollout check), else the committed plan. This
   keeps the guarantee independent of the rows' linearisation.

State of the rows: x_aug = (p, v, q, c_1 .. c_C), the drone and the chaser positions. Chaser model "guidance": the
chasers move as pure pursuers at v_c in the flow, and the chaser row is the escape margin h_c (strike_guidance);
"static": the chasers stand still and the row is |p - c| - R.

Safe-set functions h (each node): nearest obstacle sdf - r_d, height above the floor, nearest wall - r_d, ceiling
z_max - z - r_d, chaser row. Base-set functions (end): c_B - V_min, hover ellipsoid clear of obstacles / floor /
walls (guidance, reflex flying at the end: the braking point clear instead), chaser row (guidance: h_c; static:
|a - c| - R - rho). Horizon of the rows: T (the Phase-I horizon). Where u = pi_b(x) violates the rows (x outside the
rows' C_N), the QP is skipped and the safeguard filters u_nom.

Backup flow of the rows: hover LQR in the base set, else the learned backup (its scene view frozen for the step);
guidance model: blended with the evasion reflex, (1 - sigma) u_0 + sigma u_reflex.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import jax
import jax.numpy as jnp

from ps2rl.cil.backup_cbf import BackupCBFConfig, BCBFSystem, rollout_backup_flow_and_sensitivity_with_info
from ps2rl.cil.backup_cbf import solve_backup_cbf_qp_from_rows_single_with_info as _solve_from_rows
from ps2rl.cil.quadrotor_backup_cbf import QuadrotorBCBFConfig
from ps2rl.cil.strike_filter import FilterInfo, backup_scene_fn, full_margins, local_scene, make_filter
from ps2rl.cil.strike_guidance import GuidanceConfig, h_chasers, make_reflex, stop_point, wall_distances
from ps2rl.envs.quadrotor_env import quadrotor_control_affine_terms
from ps2rl.envs.strike_scene import StrikeModel, anchor, attitude_chart, hover_chaser_margin, lqr_action, obstacle_distances
from ps2rl.utils.quaternion import normalize_quaternion


@dataclass(frozen=True)
class StrikeBCBFConfig(BackupCBFConfig):
    """QP tuning; the horizon / mesh are set by ``make_strike_bcbf`` (dt of the model, horizon of the chaser model)."""

    alpha: float = QuadrotorBCBFConfig.alpha
    base_alpha: float = QuadrotorBCBFConfig.base_alpha
    slack_weight: float = QuadrotorBCBFConfig.slack_weight
    solver_tol: float = QuadrotorBCBFConfig.solver_tol
    use_analytic_jacobian: bool = False
    sensitivity_propagation: str = "discrete"
    reduced_sensitivity: bool = True
    # per-action weights (a_cmd, omega_x, omega_y, omega_z): cheaper roll / pitch rates bend the path by tilting
    # instead of changing thrust (the landing filter's finding); keep the largest at 1 (slack_weight is relative)
    control_weights: tuple | None = (1.0, 0.1, 0.1, 1.0)
    node_stride: int = 3            # rows at every node_stride-th node of the backup rollout
    horizon_s: float | None = None  # rows' horizon [s]; None: T (the Phase-I horizon the backup reaches B within)
    sensitivity_clip: float = 1e3   # Frobenius clip of the propagated sensitivity columns
    # the bCBF guarantee needs u = pi_b(x) to satisfy every row (x in C_N of the rows); where it does not, the rows
    # only distort the command, so the QP is skipped and the safeguard filters u_nom directly
    skip_if_backup_infeasible: bool = True
    backup_feasibility_tol: float = 1e-2   # on rows normalised by max(1, |a_row|)
    # keep the safeguard's slack for new plans in the rows (guidance.fresh_obstacle_margin on obstacles / walls,
    # guidance.fresh_margin on h_c), so the QP does not drive the state to where the safeguard rejects it
    row_margins: bool = False
    # per-function class-K gains (obstacle, floor, wall, ceiling, chaser); None: alpha for all. A smaller chaser gain
    # makes the QP start bending away from a chaser earlier and keep more distance
    alpha_per_row: tuple | None = None

    @classmethod
    def from_dict(cls, d: dict | None) -> "StrikeBCBFConfig":
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown bcbf keys {sorted(unknown)}")
        for k in ("control_weights", "alpha_per_row"):
            if d.get(k) is not None:
                d[k] = tuple(float(v) for v in d[k])
        return cls(**d)


def make_strike_bcbf(m: StrikeModel, backup: Callable, params, *, chaser_model: str = "guidance",
                     guidance: GuidanceConfig | None = None, arena: tuple | None = None,
                     bcbf: StrikeBCBFConfig | None = None, obstacle_range: float = float("inf"),
                     ball_range: float = float("inf"), lambdas: Sequence[float] = (1.0, 0.75, 0.5, 0.25, 0.0),
                     hover_latch: bool = True) -> Callable:
    """filt(x, u_nom, obst, chaser_p, chaser_on, committed) -> (u, FilterInfo, committed'), the interface of
    ``strike_filter.make_filter`` (vmap it). FilterInfo.lam is the safeguard's blend of u_QP; ``correction`` is
    measured against u_nom; ``held`` reports an active QP correction (|u_QP - u_nom| > 0)."""
    cfg = m.cfg
    if chaser_model not in ("guidance", "static"):
        raise ValueError(f"chaser_model must be 'guidance' or 'static', got {chaser_model!r}")
    if arena is None:
        raise ValueError("make_strike_bcbf needs arena = (x_min, x_max, y_half, z_max)")
    arena = tuple(float(v) for v in arena)
    g = guidance or GuidanceConfig()
    qp = bcbf or StrikeBCBFConfig()
    guided = chaser_model == "guidance"
    horizon = float(qp.horizon_s if qp.horizon_s is not None else cfg.T)
    n = int(round(horizon / cfg.dt))
    ecfg = StrikeBCBFConfig(**{**qp.__dict__, "T": horizon, "dt": float(cfg.dt), "num_steps": n})
    stride = max(1, int(qp.node_stride))
    alpha_vec = jnp.asarray(qp.alpha_per_row if qp.alpha_per_row is not None else (qp.alpha,) * 5, jnp.float32)
    if alpha_vec.shape != (5,):
        raise ValueError("alpha_per_row needs 5 gains (obstacle, floor, wall, ceiling, chaser)")
    C = int(cfg.n_chaser_slots)
    nx = 10 + 3 * C
    no_balls = jnp.zeros((C, 4), jnp.float32)
    reflex = make_reflex(m, g, arena)
    safeguard = make_filter(m, backup, params, lambdas=lambdas, obstacle_range=obstacle_range, ball_range=ball_range,
                            hover_latch=hover_latch, chaser_model=chaser_model, guidance=g, arena=arena)
    R_ball = float(cfg.ball_radius)
    # a backup trained with moving chasers / walls sees the chasers' flowing positions and the arena walls; with
    # guidance.learned_alone it runs without the reflex and the base set has the Phase-I hover chaser margin
    scene_of = backup_scene_fn(m, arena)
    alone = guided and bool(g.learned_alone)
    mo = float(g.fresh_obstacle_margin) if (qp.row_margins and guided) else 0.0
    mh = float(g.fresh_margin) if (qp.row_margins and guided) else 0.0

    def split(xa):
        return xa[:10], xa[10:].reshape(C, 3)

    def walls(p_xy):
        return jnp.min(wall_distances(arena, p_xy))

    def chaser_row(x, chp, on, point_offset=None):
        """Chaser margin: h_c (guidance) or |p - c| - R (static); point_offset = anchor offset for the base set."""
        if guided:
            return jnp.min(h_chasers(m, g, x[0:3], x[3:6], chp, on))
        pt = x[0:3] if point_offset is None else x[0:3] + point_offset
        d = jnp.sqrt(jnp.sum((chp - pt[None, :]) ** 2, axis=1) + 1e-12) - R_ball
        d = d - (0.0 if point_offset is None else m.lqr.rho)
        return jnp.min(jnp.where(on > 0.5, d, 1e3))

    def hover_chaser(x, chp, on):
        r = jnp.concatenate([x[3:6], attitude_chart(x[6:10])])
        return jnp.min(jnp.where(on > 0.5, hover_chaser_margin(m, chp - anchor(m, x)[None, :], r @ m.S @ r), 1e3))

    def make_system(view, order, obst, on):
        def in_base(x, chp):
            balls = jnp.concatenate([chp, on[:, None]], axis=1) if not guided else no_balls
            s, b = full_margins(m, x, obst, balls, 0.0 if not guided else None)
            b = jnp.minimum(b, walls(anchor(m, x)[0:2]) - m.lqr.rho_xy - cfg.r_drone)
            if alone:
                b = jnp.minimum(b, hover_chaser(x, chp, on))
            return (s >= 0.0) & (b >= 0.0)

        def pi_b(xa):
            x, chp = split(xa)
            u0 = jnp.where(in_base(x, chp), lqr_action(m, x), backup(params, x, scene_of(view, order, chp)))
            if guided and not alone:
                u_r, sig = reflex(x, chp, on, obst)
                u0 = jnp.clip((1.0 - sig) * u0 + sig * u_r, m.low, m.high)
            return u0

        def f_g(xa):
            x, chp = split(xa)
            f, gm = quadrotor_control_affine_terms(x, cfg.gravity)
            if guided:
                rel = x[None, 0:3] - chp
                cv = cfg.chaser_speed * rel / jnp.sqrt(jnp.sum(rel ** 2, axis=1, keepdims=True) + 1e-9) * on[:, None]
            else:
                cv = jnp.zeros((C, 3), xa.dtype)
            return jnp.concatenate([f, cv.reshape(-1)]), jnp.concatenate([gm, jnp.zeros((3 * C, 4), xa.dtype)])

        def dyn(xa, u):
            f, gm = f_g(xa)
            return f + gm @ jnp.clip(u, m.low, m.high)

        def post(z):
            x, chp = split(z)
            x = x.at[6:10].set(normalize_quaternion(x[6:10]))
            chp = chp.at[:, 2].set(jnp.maximum(chp[:, 2], cfg.z_floor + cfg.chaser_radius))
            return jnp.concatenate([x, chp.reshape(-1)])

        def safe_values(xa):
            x, chp = split(xa)
            return jnp.stack([jnp.min(obstacle_distances(x[0:2], obst)) - cfg.r_drone - mo, x[2] - cfg.z_floor,
                              walls(x[0:2]) - cfg.r_drone - mo, arena[3] - x[2] - cfg.r_drone, chaser_row(x, chp, on) - mh])

        def base_values(xa):
            """Hover ellipsoid clear (base set) - or, if the reflex is flying at the end (guidance), the braking point
            clear at a_evade (the discrete filter's terminal) - and the chaser row."""
            x, chp = split(xa)
            r = jnp.concatenate([x[3:6], attitude_chart(x[6:10])])
            off = m.D @ r
            a = x[0:3] + off
            hover = jnp.stack([cfg.c_B - r @ m.S @ r, jnp.min(obstacle_distances(a[0:2], obst)) - m.lqr.rho_xy - cfg.r_drone,
                               a[2] - m.lqr.z_ext - cfg.z_floor, walls(a[0:2]) - m.lqr.rho_xy - cfg.r_drone])
            if alone:
                return jnp.concatenate([hover, hover_chaser(x, chp, on)[None]])
            if guided:
                ps = stop_point(g, x[0:3], x[3:6])
                brake = jnp.stack([jnp.min(obstacle_distances(ps, obst)) - cfg.r_drone, x[2] - cfg.z_floor,
                                   walls(ps) - cfg.r_drone, arena[3] - x[2] - cfg.r_drone])
                hover = jnp.where(reflex(x, chp, on, obst)[1] > 0.0, brake, hover)
            return jnp.concatenate([hover, chaser_row(x, chp, on, None if guided else off)[None]])

        vg = lambda fn: (lambda xa: (fn(xa), jax.jacfwd(fn)(xa)))
        return BCBFSystem(state_dim=nx, action_dim=4, action_low=tuple(float(v) for v in cfg.action_low),
                          action_high=tuple(float(v) for v in cfg.action_high), backup_policy_fn=pi_b,
                          safe_set_values_and_grads_fn=vg(safe_values), base_set_values_and_grads_fn=vg(base_values),
                          dynamics_fn=dyn, control_affine_terms_fn=f_g, postprocess_rollout_state_fn=post,
                          sanitize_solve_state_fn=lambda z: jnp.nan_to_num(z), solve_state_slice_fn=lambda z: z)

    def rows(xa, system):
        f0, g0 = system.control_affine_terms_fn(xa)
        seed = jnp.concatenate([f0[:, None], g0], axis=1)
        xs, ms, diag = rollout_backup_flow_and_sensitivity_with_info(xa, ecfg, system, seed=seed)
        idx = jnp.arange(0, n + 1, stride)

        def node(xi, mi):
            h, dh = system.safe_set_values_and_grads_fn(xi)
            f_pi = system.dynamics_fn(xi, system.backup_policy_fn(xi))
            return -(dh @ mi[:, 1:]), alpha_vec * h + dh @ mi[:, 0] - dh @ f_pi

        a_n, b_n = jax.vmap(node)(xs[idx], ms[idx])
        h_s, dh_s = system.base_set_values_and_grads_fn(xs[-1])
        a = jnp.concatenate([a_n.reshape(-1, 4), -(dh_s @ ms[-1][:, 1:])])
        b = jnp.concatenate([b_n.reshape(-1), ecfg.base_alpha * h_s + dh_s @ ms[-1][:, 0]])
        a = jnp.nan_to_num(a)
        b = jnp.nan_to_num(b, nan=-1e6)
        return a, b, {"max_abs_q": diag["max_abs_q"], "q_saturated": diag["q_saturated"],
                      "max_abs_b": jnp.max(jnp.abs(b))}

    def filt(x, u_nom, obst, chaser_p, chaser_on, committed):
        view, order = local_scene(m, x, obst, chaser_p, chaser_on, obstacle_range, ball_range, return_order=True)
        system = make_system(view, order, obst, chaser_on)
        xa = jnp.concatenate([x, chaser_p.reshape(-1)])
        a, b, diag = rows(xa, system)
        u_qp, slack, used, _ = _solve_from_rows(xa, u_nom, a, b, diag, ecfg, system)
        if qp.skip_if_backup_infeasible:                # rows consistent only where pi_b satisfies them
            nrm = jnp.maximum(1.0, jnp.max(jnp.abs(a), axis=1))
            ok_b = jnp.max((a @ system.backup_policy_fn(xa) - b) / nrm) <= qp.backup_feasibility_tol
            used = used & ok_b
        u_qp = jnp.where(used, u_qp, u_nom)            # QP skipped / failed: the safeguard filters u_nom
        u, info, comm = safeguard(x, u_qp, obst, chaser_p, chaser_on, committed)
        info = info._replace(correction=jnp.sqrt(jnp.sum(((u - u_nom) / m.scale) ** 2)),
                             held=jnp.sum(jnp.abs(u_qp - u_nom)) > 1e-4)
        return u, info, comm

    filt.fresh_view = safeguard.fresh_view
    filt.backup_view = safeguard.backup_view
    filt.safeguard = safeguard
    filt.rows = lambda x, obst, chaser_p, chaser_on: rows(
        jnp.concatenate([x, chaser_p.reshape(-1)]),
        make_system(*local_scene(m, x, obst, chaser_p, chaser_on, obstacle_range, ball_range, return_order=True), obst,
                    chaser_on))
    return filt


__all__ = ["StrikeBCBFConfig", "make_strike_bcbf"]
