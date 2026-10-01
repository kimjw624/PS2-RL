"""How far down (and how close to the pad) does the landing BCBF filter let a vehicle go?

Three tools, all driven by the Phase-I checkpoint (cone, pad, base set, backup) and, where
a task is needed, by the Phase-II run's reference:

* ``cn_slice``      -- the backup-recoverable set C_N (first-hit: reaches B within N steps
                        without leaving the cone) on a vertical slice through the pad axis,
                        for a given velocity/attitude field. The filter keeps the state in
                        C_N, so C_N bounds where the vehicle can be, whatever the policy wants.
* ``lqr_landing``   -- a well-behaved nominal controller (the checkpoint's hover LQR,
                        re-centred on a target) run with and without the filter. If the
                        filtered LQR cannot reach a target inside C_N, the filter is too
                        conservative; if it can, a policy that does not get there has other
                        reasons.
* ``binding_rows``  -- which QP row is active at (x, u_safe): a cone row at backup node i
                        (time i*dt along the backup rollout) or the terminal base-set row.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.cil import quadrotor_landing_backup_cbf as lbcbf
from ps2rl.utils.quaternion import quaternion_from_euler_zyx


def side_direction(pad_xy: np.ndarray, ref_states: np.ndarray | None) -> np.ndarray:
    """Unit horizontal direction pad -> reference end (else -> reference start, else +x)."""
    if ref_states is not None:
        for p in (ref_states[-1, 0:2], ref_states[0, 0:2]):
            v = np.asarray(p, dtype=np.float64) - np.asarray(pad_xy, dtype=np.float64)
            if np.linalg.norm(v) > 1e-6:
                return v / np.linalg.norm(v)
    return np.array([1.0, 0.0])


def slice_positions(cfg, direction: np.ndarray, s_grid: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
    """(len(z), len(s), 3) world positions on the vertical plane through the pad axis."""
    lc = cfg.landing
    pad = np.array([lc.pad_x, lc.pad_y, lc.pad_z])
    S, Z = np.meshgrid(s_grid, z_grid)
    p = np.zeros(S.shape + (3,))
    p[..., 0] = pad[0] + S * direction[0]
    p[..., 1] = pad[1] + S * direction[1]
    p[..., 2] = pad[2] + Z
    return p


def level_quaternion(n: int, yaw: float = 0.0) -> np.ndarray:
    q = np.asarray(quaternion_from_euler_zyx(0.0, 0.0, yaw), dtype=np.float64)
    return np.tile(q, (n, 1))


def cn_slice(cfg, runtime, positions: np.ndarray, velocity: np.ndarray, quaternion: np.ndarray,
             *, chunk: int = 4096) -> dict[str, np.ndarray]:
    """First-hit C_N membership of x = [p, v, q] on a grid (any leading shape)."""
    shape = positions.shape[:-1]
    x = np.concatenate([positions.reshape(-1, 3), velocity.reshape(-1, 3), quaternion.reshape(-1, 4)], axis=1)
    rec_fn = lbcbf.make_recoverability_fn(cfg, runtime)
    outs = [rec_fn(jnp.asarray(x[i:i + chunk], dtype=jnp.float32)) for i in range(0, len(x), chunk)]
    rec, crash, arr = (np.concatenate([np.asarray(o[j]) for o in outs]) for j in range(3))
    h = np.asarray(lbcbf.cone_value(jnp.asarray(x, dtype=jnp.float32), cfg))
    return {"recoverable": rec.reshape(shape), "backup_leaves_cone": crash.reshape(shape),
            "arrival_step": arr.reshape(shape), "h": h.reshape(shape)}


def base_set_value(cfg, x: np.ndarray) -> np.ndarray:
    """Base-set value (>= 0 inside B) for states of any leading shape."""
    _, base = lbcbf.landing_sets(cfg)
    flat = jnp.asarray(x.reshape(-1, x.shape[-1]), dtype=jnp.float32)
    v = jax.vmap(lambda z: base.values_and_grads(z)[0].reshape(-1)[0])(flat)
    return np.asarray(v).reshape(x.shape[:-1])


def recentred_lqr(cfg) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """u = pi_B(x - offset): the checkpoint's hover LQR moved so that it hovers at target."""
    _, base = lbcbf.landing_sets(cfg)
    ctrl = base.controller

    def act(x, offset):
        return ctrl.action(x.at[0:3].add(-offset))

    return act


def lqr_landing(cfg, projector, x0: np.ndarray, targets: np.ndarray, steps: int) -> dict[str, np.ndarray]:
    """Unfiltered and filtered rollouts of the re-centred LQR, one target per start."""
    lc = cfg.landing
    hover = np.array([lc.pad_x, lc.pad_y, lc.pad_z + lc.z_des])
    offs = jnp.asarray(np.asarray(targets) - hover, dtype=jnp.float32)
    act = recentred_lqr(cfg)
    act_b = jax.jit(jax.vmap(act))
    step_b = jax.jit(jax.vmap(lambda x, u: lbcbf.landing_step(x, u, cfg)))
    xs_n = [jnp.asarray(x0, dtype=jnp.float32)]
    xs_f = [jnp.asarray(x0, dtype=jnp.float32)]
    u_ref_t, u_t, slack_t, used_t = [], [], [], []
    for _ in range(steps):
        xs_n.append(step_b(xs_n[-1], act_b(xs_n[-1], offs)))
        xf = xs_f[-1]
        u_ref = act_b(xf, offs)
        u_safe, slack, used, _ = projector.solve_batch_with_info(xf, u_ref)
        u_ref_t.append(np.asarray(u_ref))
        u_t.append(np.asarray(u_safe))
        slack_t.append(np.asarray(slack))
        used_t.append(np.asarray(used))
        xs_f.append(step_b(xf, u_safe.astype(jnp.float32)))
    trajn = np.stack([np.asarray(x) for x in xs_n], 1)
    trajf = np.stack([np.asarray(x) for x in xs_f], 1)
    h = lambda t: np.asarray(lbcbf.cone_value(jnp.asarray(t), cfg))  # noqa: E731
    return {"traj_unfiltered": trajn, "traj_filtered": trajf, "h_unfiltered": h(trajn), "h_filtered": h(trajf),
            "u_ref": np.stack(u_ref_t, 1), "u": np.stack(u_t, 1), "slack": np.stack(slack_t, 1),
            "qp_used": np.stack(used_t, 1)}


def binding_rows(cfg, runtime, x: np.ndarray, u: np.ndarray, slack: np.ndarray, *, tol: float = 1e-3,
                 chunk: int = 512) -> dict[str, np.ndarray]:
    """Most-binding QP row at (x, u_safe) and whether any row is active.

    Residuals are normalised like the QP rows (``constraint_row_normalize``). Returns the
    backup node of the most-binding cone row (``node``; -1 if the terminal base-set row is
    the most binding), its normalised residual, and the number of active rows.
    """
    from ps2rl.cil.backup_cbf import build_discretized_backup_cbf_rows_with_info

    n_safe = int(cfg.num_safe_constraints)
    floor = float(cfg.constraint_row_scale_floor) if cfg.constraint_row_normalize else None

    def one(xi, ui, si):
        a, b, _ = build_discretized_backup_cbf_rows_with_info(xi, cfg, runtime)
        r = a @ ui - b - si
        if floor is not None:
            scale = jnp.maximum(floor, jnp.maximum(jnp.max(jnp.abs(a), axis=1), 1.0))
            r = r / scale
        k = jnp.argmax(r)
        return k, r[k], jnp.sum(r >= -tol), r.shape[0]

    fn = jax.jit(jax.vmap(one))
    flat_x = x.reshape(-1, x.shape[-1])
    flat_u = u.reshape(-1, u.shape[-1])
    flat_s = slack.reshape(-1)
    ks, rs, na = [], [], []
    n_rows = None
    for i in range(0, len(flat_x), chunk):
        k, r, n, nr = fn(jnp.asarray(flat_x[i:i + chunk]), jnp.asarray(flat_u[i:i + chunk]),
                         jnp.asarray(flat_s[i:i + chunk]))
        ks.append(np.asarray(k))
        rs.append(np.asarray(r))
        na.append(np.asarray(n))
        n_rows = int(np.asarray(nr)[0])
    k = np.concatenate(ks)
    terminal = k == (n_rows - 1)
    node = np.where(terminal, -1, k // n_safe)
    shape = x.shape[:-1]
    return {"node": node.reshape(shape), "residual": np.concatenate(rs).reshape(shape),
            "n_active": np.concatenate(na).reshape(shape), "n_rows": n_rows}


def hover_action(cfg) -> np.ndarray:
    """u* = (g, 0, 0, 0)."""
    return np.array([cfg.landing.gravity, 0.0, 0.0, 0.0])


def row_residuals_max(cfg, runtime, x: np.ndarray, u: np.ndarray, *, chunk: int = 256) -> np.ndarray:
    """max_j (A_j u - b_j) with the QP's row normalisation, per state (no slack). <= 0: u satisfies every row."""
    from ps2rl.cil.backup_cbf import build_discretized_backup_cbf_rows_with_info

    floor = float(cfg.constraint_row_scale_floor) if cfg.constraint_row_normalize else None

    def one(xi, ui):
        a, b, _ = build_discretized_backup_cbf_rows_with_info(xi, cfg, runtime)
        r = a @ ui - b
        if floor is not None:
            r = r / jnp.maximum(floor, jnp.maximum(jnp.max(jnp.abs(a), axis=1), 1.0))
        return jnp.max(r)

    fn = jax.jit(jax.vmap(one))
    flat_x = np.asarray(x).reshape(-1, x.shape[-1])
    flat_u = np.broadcast_to(np.asarray(u), flat_x.shape[:1] + (np.asarray(u).shape[-1],))
    out = [np.asarray(fn(jnp.asarray(flat_x[i:i + chunk]), jnp.asarray(flat_u[i:i + chunk])))
           for i in range(0, len(flat_x), chunk)]
    return np.concatenate(out).reshape(np.asarray(x).shape[:-1])


def hover_feasibility(cfg, runtime, positions: np.ndarray, *, tol: float = 1e-3, chunk: int = 256) -> dict[str, np.ndarray]:
    """Is holding still (level, v = 0, u = u*) allowed by every BCBF row at these positions?"""
    shape = positions.shape[:-1]
    n = int(np.prod(shape))
    x = np.concatenate([positions.reshape(-1, 3), np.zeros((n, 3)), level_quaternion(n)], axis=1)
    res = row_residuals_max(cfg, runtime, x, hover_action(cfg), chunk=chunk)
    return {"residual": res.reshape(shape), "feasible": (res <= tol).reshape(shape)}


def tracking_lqr(cfg) -> Callable[[jax.Array, jax.Array, jax.Array], jax.Array]:
    """u = u_ref - K (e(x) - e(x_ref)): the hover LQR gain around a moving reference, with feedforward."""
    _, base = lbcbf.landing_sets(cfg)
    ctrl = base.controller
    k = jnp.asarray(ctrl.k_matrix_f64(), dtype=jnp.float32)
    low = jnp.asarray(cfg.action_low, dtype=jnp.float32)
    high = jnp.asarray(cfg.action_high, dtype=jnp.float32)

    def act(x, x_ref, u_ref):
        de = ctrl.error_state(x) - ctrl.error_state(x_ref)
        return jnp.clip(u_ref - (k @ de).astype(u_ref.dtype), low, high)

    return act


def rollout_nominal(cfg, projector, x0: np.ndarray, nominal: Callable[[jax.Array, int], jax.Array], steps: int,
                    *, filtered: bool) -> dict[str, np.ndarray]:
    """Closed loop x_{k+1} = F(x_k, u_k) with u_k = nominal(x_k, k), optionally through the filter."""
    step_b = jax.jit(jax.vmap(lambda x, u: lbcbf.landing_step(x, u, cfg)))
    xs = [jnp.asarray(x0, dtype=jnp.float32)]
    u_ref_t, u_t, slack_t, used_t, lam_t = [], [], [], [], []
    for k in range(steps):
        x = xs[-1]
        u_ref = nominal(x, k)
        lam = np.ones((x.shape[0],))
        if filtered:
            u, slack, used, info = projector.solve_batch_with_info(x, u_ref)
            if "safeguard_lambda" in info:
                lam = np.asarray(info["safeguard_lambda"])
        else:
            u, slack, used = u_ref, jnp.zeros((x.shape[0],)), jnp.ones((x.shape[0],), dtype=bool)
        lam_t.append(lam)
        u_ref_t.append(np.asarray(u_ref))
        u_t.append(np.asarray(u))
        slack_t.append(np.asarray(slack))
        used_t.append(np.asarray(used))
        xs.append(step_b(x, jnp.asarray(u, dtype=jnp.float32)))
    traj = np.stack([np.asarray(x) for x in xs], 1)
    return {"traj": traj, "u_ref": np.stack(u_ref_t, 1), "u": np.stack(u_t, 1),
            "slack": np.stack(slack_t, 1), "qp_used": np.stack(used_t, 1),
            "safeguard_lambda": np.stack(lam_t, 1)}


def touchdown_metrics(cfg, traj: np.ndarray, *, zeta_touch: float, v_touch: float) -> dict[str, np.ndarray]:
    """Per trajectory: touchdown = first state with height <= zeta_touch over the pad disk (radius r0)
    and speed <= v_touch. Also min cone margin, min height, final position."""
    lc = cfg.landing
    pad = np.array([lc.pad_x, lc.pad_y, lc.pad_z])
    rel = traj[..., 0:3] - pad
    radial = np.linalg.norm(rel[..., 0:2], axis=-1)
    speed = np.linalg.norm(traj[..., 3:6], axis=-1)
    td = (rel[..., 2] <= zeta_touch) & (radial <= lc.cone_r0) & (speed <= v_touch)
    first = np.where(td.any(axis=1), td.argmax(axis=1), -1)
    h_cone = np.asarray(lbcbf.cone_value(jnp.asarray(traj), cfg))
    return {
        "touchdown": first >= 0,
        "touchdown_time": np.where(first >= 0, first * lc.dt, np.nan),
        "min_h_cone": h_cone.min(axis=1),
        "min_height": rel[..., 2].min(axis=1),
        "final_height": rel[:, -1, 2],
        "final_radial": radial[:, -1],
        "final_speed": speed[:, -1],
        "h_cone": h_cone,
    }


def vertical_accel(u: np.ndarray, q: np.ndarray, gravity: float) -> np.ndarray:
    """World-frame vertical acceleration a_cmd * (R e3)_z - g."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    r33 = 1.0 - 2.0 * (q[..., 1] ** 2 + q[..., 2] ** 2)
    return u[..., 0] * r33 - gravity


__all__ = [
    "hover_action",
    "hover_feasibility",
    "rollout_nominal",
    "row_residuals_max",
    "touchdown_metrics",
    "tracking_lqr",
    "base_set_value",
    "binding_rows",
    "cn_slice",
    "level_quaternion",
    "lqr_landing",
    "recentred_lqr",
    "side_direction",
    "slice_positions",
    "vertical_accel",
]
