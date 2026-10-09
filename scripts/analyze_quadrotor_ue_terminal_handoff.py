#!/usr/bin/env python
"""Validate two PS2-RL-specific UE-bCBF design choices before modifying the QP.

This script does NOT change the controller.  It tests:

(1) Quadratic terminal tightening for the LQR ellipsoid

    h_B(e) = c_B - e^T P e.

    For ||Delta e|| <= r_e,

      h_B(e + Delta e)
      >= h_B(e) - [2 ||P e|| r_e + lambda_max(P) r_e^2].

    We verify the inequality on real learned-backup terminal states, compare the
    resulting margin against the previously used global Lipschitz margin
    L_hB r_e, and measure the local map from physical 9-D tangent errors to the
    7-D hover-error coordinates used by h_B.

(2) Whether the original PS2-RL learned-SA -> LQR hard handoff can be kept for
    UE-bCBF sensitivity propagation.

    We test both the action/Jacobian mismatch on h_B=0 and, more importantly,
    compare the branchwise variational sensitivity propagated by the existing
    PS2-RL convention against a finite-difference Jacobian of the actual
    discrete backup-flow map.  The comparison is reported separately for
    trajectories that cross h_B=0 and trajectories that do not cross it.

The output is diagnostic/empirical.  It does not turn a sampled result into a
formal hybrid-system proof, but it directly tells us whether the handoff is a
numerically important missing term before UE-bCBF is wired into the QP.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from analyze_quadrotor_lhb_methods import POOL_NAMES, make_dhat_samples, sample_ps2_domain
from ps2rl.backup_policy.quadrotor_learned_backup import load_learned_quadrotor_backup_policy
from ps2rl.base_controller.quadrotor_dlqr import QuadrotorDLQR
from ps2rl.cil.quadrotor_backup_cbf import make_backup_runtime
from ps2rl.cil.quadrotor_ue_rollout import estimated_disturbance_backup_dynamics
from ps2rl.evaluation.quadrotor_trace_reset_lib import QuadrotorResetLibrary
from ps2rl.utils.quaternion import (
    normalize_quaternion_np,
    quaternion_conjugate_np,
    quaternion_multiply_np,
    quaternion_rate_matrix_np,
)


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, float)):
        x = float(v)
        return x if np.isfinite(x) else None
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return v


def _stats(x: np.ndarray | list[float]) -> dict[str, Any]:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"count": 0}
    return {
        "count": int(a.size),
        "min": float(np.min(a)),
        "p50": float(np.percentile(a, 50.0)),
        "p90": float(np.percentile(a, 90.0)),
        "p95": float(np.percentile(a, 95.0)),
        "p99": float(np.percentile(a, 99.0)),
        "p99_9": float(np.percentile(a, 99.9)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
    }


def _print_stats(label: str, s: dict[str, Any]) -> None:
    if not s.get("count"):
        print(f"{label}: no samples")
        return
    print(
        f"{label}: n={s['count']} p50={s['p50']:.6g} p95={s['p95']:.6g} "
        f"p99={s['p99']:.6g} max={s['max']:.6g}"
    )


def _parse_csv(text: str) -> list[float]:
    vals = [float(v.strip()) for v in str(text).split(",") if v.strip()]
    if not vals:
        raise ValueError("expected a non-empty comma-separated list")
    return vals


def _select_starts(states: np.ndarray, labels: np.ndarray, per_region: int) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ls: list[np.ndarray] = []
    for name in POOL_NAMES:
        idx = np.flatnonzero(labels == name)
        take = min(int(per_region), int(idx.size))
        if take:
            xs.append(states[idx[:take]])
            ls.append(np.asarray([name] * take, dtype=object))
    if not xs:
        raise RuntimeError("no PS2-RL domain states selected")
    return np.concatenate(xs), np.concatenate(ls)


def _learned_controller(cfg: Any, learned_path: str) -> tuple[Any, QuadrotorDLQR]:
    learned = load_learned_quadrotor_backup_policy(learned_path)
    payload = learned.metadata.get("lqr_config")
    if not isinstance(payload, dict):
        raise KeyError("learned backup is missing metadata['lqr_config']")
    controller = QuadrotorDLQR.from_payload(payload, fallback=cfg)
    return learned, controller


def _make_state_phi_rollout(cfg: Any, runtime: Any, n_steps: int):
    dt = jnp.asarray(float(cfg.dt), dtype=jnp.float32)
    n = int(n_steps)

    def single(x0: jax.Array, d_hat: jax.Array):
        def fhat(x: jax.Array) -> jax.Array:
            return estimated_disturbance_backup_dynamics(x, d_hat, runtime)

        jac = jax.jacfwd(fhat)

        def step(carry, _):
            x, phi = carry
            j = jac(x)
            dx = fhat(x)
            x_next = runtime.postprocess_rollout_state_fn(x + dt * dx)
            phi_next = jsp.linalg.expm(dt * j) @ phi
            return (x_next, phi_next), (x_next, phi_next)

        phi0 = jnp.eye(10, dtype=x0.dtype)
        (_, _), (xt, pt) = jax.lax.scan(step, (x0, phi0), xs=None, length=n)
        xs = jnp.concatenate([x0[None, :], xt], axis=0)
        phis = jnp.concatenate([phi0[None, :, :], pt], axis=0)
        return xs, phis

    return jax.jit(jax.vmap(single, in_axes=(0, 0)))


def _make_state_rollout(cfg: Any, runtime: Any, n_steps: int):
    dt = jnp.asarray(float(cfg.dt), dtype=jnp.float32)
    n = int(n_steps)

    def single(x0: jax.Array, d_hat: jax.Array):
        def step(x, _):
            dx = estimated_disturbance_backup_dynamics(x, d_hat, runtime)
            xn = runtime.postprocess_rollout_state_fn(x + dt * dx)
            return xn, xn

        _, xt = jax.lax.scan(step, x0, xs=None, length=n)
        return jnp.concatenate([x0[None, :], xt], axis=0)

    return jax.jit(jax.vmap(single, in_axes=(0, 0)))


def _base_values(runtime: Any, xs: np.ndarray) -> np.ndarray:
    fn = jax.jit(jax.vmap(lambda x: runtime.base_set_values_fn(x)[0]))
    flat = np.asarray(xs, dtype=np.float32).reshape((-1, 10))
    out = np.asarray(jax.device_get(fn(jnp.asarray(flat))), dtype=np.float64)
    return out.reshape(xs.shape[:-1])


def _tangent_basis_np(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = normalize_quaternion_np(np.asarray(x[6:10], dtype=np.float64))
    xi = quaternion_rate_matrix_np(q)
    b = np.zeros((10, 9), dtype=np.float64)
    b[:6, :6] = np.eye(6)
    b[6:10, 6:9] = 0.5 * xi
    bp = np.zeros((9, 10), dtype=np.float64)
    bp[:6, :6] = np.eye(6)
    bp[6:9, 6:10] = 2.0 * xi.T
    return b, bp


def _quat_exp_np(dtheta: np.ndarray) -> np.ndarray:
    v = np.asarray(dtheta, dtype=np.float64)
    a = float(np.linalg.norm(v))
    if a < 1e-14:
        return normalize_quaternion_np(np.array([1.0, 0.5 * v[0], 0.5 * v[1], 0.5 * v[2]]))
    half = 0.5 * a
    return np.concatenate([[np.cos(half)], np.sin(half) * v / a])


def _perturb_tangent_np(x: np.ndarray, dz: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).copy()
    dz = np.asarray(dz, dtype=np.float64)
    out[:3] += dz[:3]
    out[3:6] += dz[3:6]
    dq = _quat_exp_np(dz[6:9])
    out[6:10] = normalize_quaternion_np(quaternion_multiply_np(out[6:10], dq))
    return out.astype(np.float32)


def _local_error9_np(x_ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    xr = np.asarray(x_ref, dtype=np.float64)
    xx = np.asarray(x, dtype=np.float64)
    q_ref = normalize_quaternion_np(xr[6:10])
    q = normalize_quaternion_np(xx[6:10])
    q_rel = normalize_quaternion_np(quaternion_multiply_np(quaternion_conjugate_np(q_ref), q))
    sign = 1.0 if q_rel[0] >= 0.0 else -1.0
    dtheta = 2.0 * sign * q_rel[1:4]
    return np.concatenate([xx[:3] - xr[:3], xx[3:6] - xr[3:6], dtheta])


def _construct_boundary_states(
    controller: QuadrotorDLQR,
    *,
    c_b: float,
    count: int,
    seed: int,
) -> np.ndarray:
    p = np.asarray(controller.p_matrix, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    out: list[np.ndarray] = []
    attempts = 0
    while len(out) < int(count) and attempts < int(count) * 100:
        attempts += 1
        d = rng.normal(size=(7,))
        d /= max(float(np.linalg.norm(d)), 1e-12)
        den = float(d @ p @ d)
        if den <= 0.0:
            continue
        e = d * np.sqrt(float(c_b) / den)
        qv = -0.5 * e[4:7]
        qv2 = float(qv @ qv)
        if qv2 >= 0.95**2:
            continue
        q = np.concatenate([[np.sqrt(max(1.0 - qv2, 0.0))], qv])
        x = np.zeros((10,), dtype=np.float64)
        x[2] = float(controller.z_des) + e[0]
        x[3:6] = e[1:4]
        x[6:10] = normalize_quaternion_np(q)
        out.append(x.astype(np.float32))
    if len(out) < int(count):
        raise RuntimeError(f"could construct only {len(out)}/{count} boundary states")
    return np.stack(out)


def _quadratic_terminal_test(
    *,
    terminal_states: np.ndarray,
    controller: QuadrotorDLQR,
    c_b: float,
    radii: list[float],
    comparison_lhb: float,
    directions_per_state: int,
    max_states: int,
    physical_radii: list[float],
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    p = np.asarray(controller.p_matrix, dtype=np.float64)
    lam_max = float(np.linalg.eigvalsh(p)[-1])
    err_fn = jax.jit(jax.vmap(controller.error_state))
    e_all = np.asarray(jax.device_get(err_fn(jnp.asarray(terminal_states))), dtype=np.float64)
    v_all = np.einsum("bi,ij,bj->b", e_all, p, e_all)
    hb_all = float(c_b) - v_all
    inside = hb_all >= 0.0
    xs = np.asarray(terminal_states[inside], dtype=np.float32)
    es = e_all[inside]
    hbs = hb_all[inside]
    if xs.shape[0] == 0:
        raise RuntimeError("no terminal states inside B; cannot test terminal tightening")
    if xs.shape[0] > int(max_states):
        idx = rng.choice(xs.shape[0], size=int(max_states), replace=False)
        xs = xs[idx]
        es = es[idx]
        hbs = hbs[idx]

    report: dict[str, Any] = {
        "terminal_inside_B_count": int(np.sum(inside)),
        "terminal_test_count": int(xs.shape[0]),
        "lambda_max_P": lam_max,
        "comparison_L_hB": float(comparison_lhb),
        "radii": {},
    }

    pe_norm = np.linalg.norm(es @ p.T, axis=1)
    for r in radii:
        rr = float(r)
        quad_margin = 2.0 * pe_norm * rr + lam_max * rr * rr
        lip_margin = float(comparison_lhb) * rr

        dirs = rng.normal(size=(xs.shape[0], int(directions_per_state), 7))
        dirs /= np.maximum(np.linalg.norm(dirs, axis=2, keepdims=True), 1e-12)
        de = rr * dirs
        e_true = es[:, None, :] + de
        v_true = np.einsum("bki,ij,bkj->bk", e_true, p, e_true)
        hb_true = float(c_b) - v_true
        lower = hbs[:, None] - quad_margin[:, None]
        slack = hb_true - lower
        actual_loss = hbs[:, None] - hb_true
        sampled_worst_loss = np.max(actual_loss, axis=1)
        ratio = quad_margin / np.maximum(sampled_worst_loss, 1e-12)

        report["radii"][f"{rr:.8g}"] = {
            "r_e": rr,
            "quadratic_margin": _stats(quad_margin),
            "lipschitz_margin": float(lip_margin),
            "sampled_bound_slack": _stats(slack),
            "sampled_violation_count": int(np.sum(slack < -1e-8)),
            "quad_over_sampled_worst_loss": _stats(ratio),
            "robust_terminal_fraction_quadratic": float(np.mean(hbs >= quad_margin)),
            "robust_terminal_fraction_lipschitz": float(np.mean(hbs >= lip_margin)),
        }

    physical: dict[str, Any] = {}
    for r in physical_radii:
        rr = float(r)
        dirs = rng.normal(size=(xs.shape[0], int(directions_per_state), 9))
        dirs /= np.maximum(np.linalg.norm(dirs, axis=2, keepdims=True), 1e-12)
        flat_states: list[np.ndarray] = []
        for i in range(xs.shape[0]):
            for j in range(int(directions_per_state)):
                flat_states.append(_perturb_tangent_np(xs[i], rr * dirs[i, j]))
        xp = np.stack(flat_states)
        ep = np.asarray(jax.device_get(err_fn(jnp.asarray(xp))), dtype=np.float64)
        e0_rep = np.repeat(es, int(directions_per_state), axis=0)
        de_norm = np.linalg.norm(ep - e0_rep, axis=1)
        ratio = de_norm / max(rr, 1e-12)
        physical[f"{rr:.8g}"] = {
            "physical_tangent_radius": rr,
            "norm_delta_e_over_norm_delta_z": _stats(ratio),
        }
    report["physical_to_reduced_error_map"] = physical
    return report


def _boundary_action_test(
    *,
    boundary_states: np.ndarray,
    learned: Any,
    controller: QuadrotorDLQR,
    cfg: Any,
) -> dict[str, Any]:
    sa_fn = jax.jit(jax.vmap(learned.action_single))
    lqr_fn = jax.jit(jax.vmap(controller.action))
    xs = jnp.asarray(boundary_states)
    usa = np.asarray(jax.device_get(sa_fn(xs)), dtype=np.float64)
    ulqr = np.asarray(jax.device_get(lqr_fn(xs)), dtype=np.float64)
    gap = np.linalg.norm(usa - ulqr, axis=1)
    span = np.asarray(
        [float(cfg.a_cmd_max) - float(cfg.a_cmd_min), 2.0 * float(cfg.omega_max), 2.0 * float(cfg.omega_max), 2.0 * float(cfg.omega_max)],
        dtype=np.float64,
    )
    gap_norm = np.linalg.norm((usa - ulqr) / span[None, :], axis=1)

    jac_sa = jax.jit(jax.vmap(jax.jacfwd(learned.action_single)))
    jac_lqr = jax.jit(jax.vmap(jax.jacfwd(controller.action)))
    jsa10 = np.asarray(jax.device_get(jac_sa(xs)), dtype=np.float64)
    jlqr10 = np.asarray(jax.device_get(jac_lqr(xs)), dtype=np.float64)
    jgap: list[float] = []
    jsa_n: list[float] = []
    jlqr_n: list[float] = []
    for i, x in enumerate(boundary_states):
        b, _ = _tangent_basis_np(x)
        a = jsa10[i] @ b
        c = jlqr10[i] @ b
        jgap.append(float(np.linalg.norm(a - c, ord=2)))
        jsa_n.append(float(np.linalg.norm(a, ord=2)))
        jlqr_n.append(float(np.linalg.norm(c, ord=2)))

    return {
        "count": int(boundary_states.shape[0]),
        "action_gap_raw_l2": _stats(gap),
        "action_gap_normalized_l2": _stats(gap_norm),
        "policy_jacobian_gap_spectral_tangent": _stats(jgap),
        "sa_policy_jacobian_spectral_tangent": _stats(jsa_n),
        "lqr_policy_jacobian_spectral_tangent": _stats(jlqr_n),
    }


def _first_crossing_indices(hb: np.ndarray) -> np.ndarray:
    before = hb[:, :-1] < 0.0
    after = hb[:, 1:] >= 0.0
    hit = before & after
    out = np.full((hb.shape[0],), -1, dtype=np.int32)
    rows = np.flatnonzero(np.any(hit, axis=1))
    if rows.size:
        out[rows] = np.argmax(hit[rows], axis=1) + 1
    return out


def _fd_group(
    *,
    name: str,
    indices: np.ndarray,
    eval_steps: np.ndarray,
    xs_nom: np.ndarray,
    phis_nom: np.ndarray,
    x0_pairs: np.ndarray,
    d_pairs: np.ndarray,
    cfg: Any,
    runtime: Any,
    max_steps: int,
    fd_eps: float,
) -> dict[str, Any]:
    if indices.size == 0:
        return {"count": 0}

    perturbed: list[np.ndarray] = []
    pd: list[np.ndarray] = []
    owner: list[tuple[int, int, int]] = []
    for local_i, pair_idx in enumerate(indices.tolist()):
        x0 = x0_pairs[pair_idx]
        for axis in range(9):
            for sign in (+1, -1):
                dz = np.zeros((9,), dtype=np.float64)
                dz[axis] = sign * float(fd_eps)
                perturbed.append(_perturb_tangent_np(x0, dz))
                pd.append(d_pairs[pair_idx])
                owner.append((local_i, axis, sign))
    xp = np.stack(perturbed).astype(np.float32)
    dp = np.stack(pd).astype(np.float32)
    roll_fn = _make_state_rollout(cfg, runtime, int(max_steps))
    xsp = np.asarray(jax.device_get(roll_fn(jnp.asarray(xp), jnp.asarray(dp))), dtype=np.float32)
    hbp = _base_values(runtime, xsp)

    fd_mats = np.zeros((indices.size, 9, 9), dtype=np.float64)
    plus_entry = np.full((indices.size, 9), -1, dtype=np.int32)
    minus_entry = np.full((indices.size, 9), -1, dtype=np.int32)

    cursor = 0
    for local_i, pair_idx in enumerate(indices.tolist()):
        k = int(eval_steps[local_i])
        x_ref = xs_nom[pair_idx, k]
        for axis in range(9):
            plus_idx = cursor
            minus_idx = cursor + 1
            cursor += 2
            zp = _local_error9_np(x_ref, xsp[plus_idx, k])
            zm = _local_error9_np(x_ref, xsp[minus_idx, k])
            fd_mats[local_i, :, axis] = (zp - zm) / (2.0 * float(fd_eps))
            cp = _first_crossing_indices(hbp[plus_idx : plus_idx + 1])[0]
            cm = _first_crossing_indices(hbp[minus_idx : minus_idx + 1])[0]
            plus_entry[local_i, axis] = cp
            minus_entry[local_i, axis] = cm

    rel_fro: list[float] = []
    abs_fro: list[float] = []
    pred_norm: list[float] = []
    fd_norm: list[float] = []
    event_shift: list[float] = []
    for local_i, pair_idx in enumerate(indices.tolist()):
        k = int(eval_steps[local_i])
        b0, _ = _tangent_basis_np(x0_pairs[pair_idx])
        _, bp_out = _tangent_basis_np(xs_nom[pair_idx, k])
        pred = bp_out @ np.asarray(phis_nom[pair_idx, k], dtype=np.float64) @ b0
        fd = fd_mats[local_i]
        err = float(np.linalg.norm(fd - pred, ord="fro"))
        den = float(np.linalg.norm(fd, ord="fro"))
        abs_fro.append(err)
        rel_fro.append(err / max(den, 1e-12))
        pred_norm.append(float(np.linalg.norm(pred, ord=2)))
        fd_norm.append(float(np.linalg.norm(fd, ord=2)))
        valid = (plus_entry[local_i] >= 0) & (minus_entry[local_i] >= 0)
        if np.any(valid):
            event_shift.extend(np.abs(plus_entry[local_i, valid] - minus_entry[local_i, valid]).astype(float).tolist())

    return {
        "name": name,
        "count": int(indices.size),
        "relative_frobenius_error": _stats(rel_fro),
        "absolute_frobenius_error": _stats(abs_fro),
        "fd_jacobian_spectral_norm": _stats(fd_norm),
        "propagated_jacobian_spectral_norm": _stats(pred_norm),
        "plus_minus_handoff_step_difference": _stats(event_shift),
    }


def _handoff_flow_test(
    *,
    x_pairs: np.ndarray,
    d_pairs: np.ndarray,
    cfg: Any,
    runtime: Any,
    horizon_steps: int,
    post_steps: int,
    fd_eps: float,
    sample_count: int,
    control_steps: int,
    seed: int,
) -> dict[str, Any]:
    phi_fn = _make_state_phi_rollout(cfg, runtime, int(horizon_steps))
    xs, phis = phi_fn(jnp.asarray(x_pairs), jnp.asarray(d_pairs))
    xs = np.asarray(jax.device_get(xs), dtype=np.float32)
    phis = np.asarray(jax.device_get(phis), dtype=np.float64)
    hb = _base_values(runtime, xs)
    crossing = _first_crossing_indices(hb)

    rng = np.random.default_rng(int(seed))
    cross_idx = np.flatnonzero(crossing >= 1)
    if cross_idx.size > int(sample_count):
        cross_idx = rng.choice(cross_idx, size=int(sample_count), replace=False)
    cross_eval = np.asarray(
        [min(int(horizon_steps), int(crossing[i]) + int(post_steps)) for i in cross_idx],
        dtype=np.int32,
    )

    # Control group: stay strictly on one side of h_B=0 over the whole tested horizon.
    no_cross_mask = (np.all(hb < -1e-3, axis=1) | np.all(hb > 1e-3, axis=1))
    control_idx = np.flatnonzero(no_cross_mask)
    if control_idx.size > int(sample_count):
        control_idx = rng.choice(control_idx, size=int(sample_count), replace=False)
    cstep = min(max(1, int(control_steps)), int(horizon_steps))
    control_eval = np.full((control_idx.size,), cstep, dtype=np.int32)

    cross_report = _fd_group(
        name="crossing",
        indices=np.asarray(cross_idx, dtype=np.int32),
        eval_steps=cross_eval,
        xs_nom=xs,
        phis_nom=phis,
        x0_pairs=x_pairs,
        d_pairs=d_pairs,
        cfg=cfg,
        runtime=runtime,
        max_steps=int(horizon_steps),
        fd_eps=float(fd_eps),
    )
    control_report = _fd_group(
        name="no_cross_control",
        indices=np.asarray(control_idx, dtype=np.int32),
        eval_steps=control_eval,
        xs_nom=xs,
        phis_nom=phis,
        x0_pairs=x_pairs,
        d_pairs=d_pairs,
        cfg=cfg,
        runtime=runtime,
        max_steps=int(horizon_steps),
        fd_eps=float(fd_eps),
    )

    cross_err = cross_report.get("relative_frobenius_error", {}).get("p50")
    ctrl_err = control_report.get("relative_frobenius_error", {}).get("p50")
    ratio = None
    if cross_err is not None and ctrl_err is not None:
        ratio = float(cross_err) / max(float(ctrl_err), 1e-12)

    return {
        "candidate_trajectory_count": int(x_pairs.shape[0]),
        "crossing_candidate_count": int(np.sum(crossing >= 1)),
        "crossing_sample": cross_report,
        "no_cross_control_sample": control_report,
        "median_crossing_error_over_control_error": ratio,
        "fd_eps": float(fd_eps),
        "post_crossing_steps": int(post_steps),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--reset-library", required=True)
    p.add_argument("--learned-backup-policy-path", required=True)
    p.add_argument("--domain-source", choices=("paper", "library"), default="paper")
    p.add_argument("--samples-per-region", type=int, default=256)
    p.add_argument("--flow-initial-per-region", type=int, default=16)
    p.add_argument("--dhat-random", type=int, default=6)
    p.add_argument("--disturbance-bound", type=float, default=2.0)
    p.add_argument("--terminal-horizon", type=float, default=1.0)
    p.add_argument("--terminal-radii", default="0.02,0.05,0.10,0.20,0.30")
    p.add_argument("--physical-radii", default="0.01,0.05,0.10")
    p.add_argument("--comparison-lhb", type=float, default=30.070784)
    p.add_argument("--terminal-test-states", type=int, default=256)
    p.add_argument("--terminal-directions", type=int, default=64)
    p.add_argument("--boundary-states", type=int, default=256)
    p.add_argument("--handoff-horizon", type=float, default=1.0)
    p.add_argument("--handoff-flow-samples", type=int, default=24)
    p.add_argument("--handoff-post-steps", type=int, default=2)
    p.add_argument("--handoff-control-time", type=float, default=0.20)
    p.add_argument("--fd-eps", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="outputs/ue_terminal_handoff_analysis.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    reset_path = Path(args.reset_library).expanduser().resolve()
    lib = QuadrotorResetLibrary.load(reset_path)
    cfg = replace(
        lib.cbf_cfg,
        backup_policy_mode="learned",
        learned_backup_policy_path=str(args.learned_backup_policy_path),
    )
    runtime = make_backup_runtime(cfg)
    learned, controller = _learned_controller(cfg, str(args.learned_backup_policy_path))

    domain, labels = sample_ps2_domain(
        lib,
        samples_per_region=int(args.samples_per_region),
        seed=int(args.seed),
        domain_source=str(args.domain_source),
    )
    starts, start_labels = _select_starts(domain, labels, int(args.flow_initial_per_region))
    dhats = make_dhat_samples(float(args.disturbance_bound), int(args.dhat_random), int(args.seed) + 2718)
    xp = np.repeat(starts, dhats.shape[0], axis=0).astype(np.float32)
    dp = np.tile(dhats, (starts.shape[0], 1)).astype(np.float32)
    pair_labels = np.repeat(start_labels, dhats.shape[0])

    dt = float(cfg.dt)
    terminal_steps = int(round(float(args.terminal_horizon) / dt))
    handoff_steps = int(round(float(args.handoff_horizon) / dt))
    if terminal_steps < 1 or terminal_steps > int(cfg.num_steps):
        raise SystemExit("--terminal-horizon must lie inside the configured backup horizon")
    if handoff_steps < 1 or handoff_steps > int(cfg.num_steps):
        raise SystemExit("--handoff-horizon must lie inside the configured backup horizon")

    print("\n=== TEST 3A: real terminal states ===")
    term_roll = _make_state_rollout(cfg, runtime, terminal_steps)
    terminal_chunks: list[np.ndarray] = []
    bs = max(1, int(args.batch_size))
    for start in range(0, xp.shape[0], bs):
        stop = min(start + bs, xp.shape[0])
        xs = term_roll(jnp.asarray(xp[start:stop]), jnp.asarray(dp[start:stop]))
        terminal_chunks.append(np.asarray(jax.device_get(xs[:, -1, :]), dtype=np.float32))
    terminals = np.concatenate(terminal_chunks, axis=0)
    hb_term = _base_values(runtime, terminals)
    print(f"terminal states: {len(terminals)}, inside B: {int(np.sum(hb_term >= 0.0))}")

    quad_report = _quadratic_terminal_test(
        terminal_states=terminals,
        controller=controller,
        c_b=float(cfg.base_set_c),
        radii=_parse_csv(args.terminal_radii),
        comparison_lhb=float(args.comparison_lhb),
        directions_per_state=int(args.terminal_directions),
        max_states=int(args.terminal_test_states),
        physical_radii=_parse_csv(args.physical_radii),
        seed=int(args.seed) + 991,
    )
    for key, rec in quad_report["radii"].items():
        print(
            f"r_e={key}: violations={rec['sampled_violation_count']} "
            f"quad feasible={rec['robust_terminal_fraction_quadratic']:.4f} "
            f"LhB feasible={rec['robust_terminal_fraction_lipschitz']:.4f}"
        )
        _print_stats("  quadratic margin", rec["quadratic_margin"])
    for key, rec in quad_report["physical_to_reduced_error_map"].items():
        _print_stats(
            f"||Delta e||/||Delta z|| at physical radius {key}",
            rec["norm_delta_e_over_norm_delta_z"],
        )

    print("\n=== TEST 4A: action/Jacobian match on h_B=0 ===")
    boundary = _construct_boundary_states(
        controller,
        c_b=float(cfg.base_set_c),
        count=int(args.boundary_states),
        seed=int(args.seed) + 404,
    )
    hb_boundary = _base_values(runtime, boundary)
    print(f"constructed boundary max |h_B| = {float(np.max(np.abs(hb_boundary))):.3e}")
    boundary_report = _boundary_action_test(
        boundary_states=boundary,
        learned=learned,
        controller=controller,
        cfg=cfg,
    )
    _print_stats("SA-LQR action gap / normalized", boundary_report["action_gap_normalized_l2"])
    _print_stats("SA-LQR tangent Jacobian spectral gap", boundary_report["policy_jacobian_gap_spectral_tangent"])

    print("\n=== TEST 4B: finite-difference backup-flow sensitivity through handoff ===")
    control_steps = int(round(float(args.handoff_control_time) / dt))
    handoff_report = _handoff_flow_test(
        x_pairs=xp,
        d_pairs=dp,
        cfg=cfg,
        runtime=runtime,
        horizon_steps=handoff_steps,
        post_steps=int(args.handoff_post_steps),
        fd_eps=float(args.fd_eps),
        sample_count=int(args.handoff_flow_samples),
        control_steps=control_steps,
        seed=int(args.seed) + 7331,
    )
    print(
        f"handoff candidates: {handoff_report['crossing_candidate_count']}/"
        f"{handoff_report['candidate_trajectory_count']}"
    )
    _print_stats(
        "crossing FD-vs-propagated relative Fro error",
        handoff_report["crossing_sample"].get("relative_frobenius_error", {"count": 0}),
    )
    _print_stats(
        "no-cross FD-vs-propagated relative Fro error",
        handoff_report["no_cross_control_sample"].get("relative_frobenius_error", {"count": 0}),
    )
    print(
        "median crossing/control sensitivity-error ratio =",
        handoff_report["median_crossing_error_over_control_error"],
    )

    report = {
        "settings": {
            "reset_library": str(reset_path),
            "learned_backup_policy_path": str(args.learned_backup_policy_path),
            "domain_source": str(args.domain_source),
            "delta_d": float(args.disturbance_bound),
            "dt": dt,
            "terminal_horizon": float(args.terminal_horizon),
            "handoff_horizon": float(args.handoff_horizon),
            "pair_count": int(xp.shape[0]),
            "pair_region_counts": {name: int(np.sum(pair_labels == name)) for name in POOL_NAMES},
        },
        "quadratic_terminal_tightening": quad_report,
        "handoff_boundary_match": boundary_report,
        "handoff_flow_sensitivity": handoff_report,
        "interpretation": {
            "quadratic_test": "sampled_violation_count should be zero; compare robust terminal fractions against L_hB*r",
            "handoff_test": "if crossing FD-vs-propagated error is much larger than the no-cross control, ordinary branchwise Phi misses important handoff sensitivity",
            "formal_status": "diagnostic only; a passing handoff finite-difference test is evidence for keeping the original PS2 handoff, not a general hybrid-system proof",
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
