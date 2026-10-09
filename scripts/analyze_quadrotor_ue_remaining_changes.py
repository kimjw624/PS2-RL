#!/usr/bin/env python
"""Run the remaining no-retraining UE-bCBF diagnostics for the PS2-RL quadrotor.

This script does not change the controller or the QP.  It tests the remaining
items that should be settled before wiring UE-bCBF into Phase II:

1. Physical 9-D tangent linearization correctness.
   A9 = B^+ (J10 B - Bdot) is checked against finite differences of the actual
   normalized one-step backup map on the quaternion manifold.

2. Three-dimensional disturbance-channel sensitivity correctness.
   The UE disturbance sensitivity Theta = d phi / d d_hat is checked against
   central finite differences of the actual frozen-d_hat backup rollout.

3. Learned-backup state sensitivity diagnosis.
   The policy Jacobian is split into position, velocity, and attitude channels,
   and compared with the physical 9-D closed-loop matrix measure.  The script
   also compares mu_2(A9) with the spectral abscissa and tests whether a single
   constant diagonal metric can reduce coordinate-induced non-normality.

4. Backup-horizon selection by PS2-RL design-region category.
   Safe/reach-B rates are reported for each requested horizon, including rates
   conditioned on states that are safe at tau=0.

5. Nonlinear sampled robust invariance of the LQR base ellipsoid at delta_d.
   Boundary states are sampled directly from e^T P e = c_B and stress-tested
   for one step against many acceleration-disturbance directions.

6. Disturbance-channel-specific linearized flow radius.
   Instead of replacing every local transition by a scalar log norm, the script
   propagates the actual LTV disturbance kernels G_k through the 9-D tangent
   dynamics and evaluates

       R_ch(k) = sum_j ||F_{k-1}...F_{j+1} G_j|| q_j,
       q_j = delta_v * tau_{j+1} + e_bar(t0).

   R_ch is compared with the actual physical 9-D separation between a true
   sinusoidally disturbed backup flow and its frozen-d_hat estimated flow.
   This is still a centerline linearization diagnostic, not yet a nonlinear
   tube certificate.

The hard learned-SA -> LQR handoff and the quadratic terminal h_B tightening
are intentionally NOT modified here; those are tested separately by the
Step-19 terminal/handoff diagnostic.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from math import pi
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy.linalg

from analyze_quadrotor_lhb_methods import POOL_NAMES, make_dhat_samples, sample_ps2_domain
from analyze_quadrotor_ue_geometry_policy_horizon import _tangent_maps
from analyze_quadrotor_ue_trajectory_tube import (
    _continuous_observer_estimate_for_sinusoid,
    make_sinusoid_scenarios,
)
from ps2rl.backup_policy.quadrotor_learned_backup import load_learned_quadrotor_backup_policy
from ps2rl.base_controller.quadrotor_dlqr import QuadrotorDLQR
from ps2rl.cil.quadrotor_backup_cbf import make_backup_runtime
from ps2rl.cil.quadrotor_ue_rollout import (
    estimated_disturbance_backup_dynamics,
    quadrotor_disturbance_injection_matrix,
    rollout_estimated_disturbance_backup_flow_and_sensitivities,
)
from ps2rl.cil.quadrotor_ue_discrete_sensitivity import (
    discrete_tangent_jacobians,
)
from ps2rl.envs.quadrotor_env import quadrotor_step_euler
from ps2rl.evaluation.quadrotor_trace_reset_lib import QuadrotorResetLibrary
from ps2rl.uncertainty.quadrotor_disturbance_observer import observer_error_bound
from ps2rl.utils.quaternion import (
    normalize_quaternion_np,
    quaternion_conjugate_np,
    quaternion_multiply_np,
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
        raise ValueError("expected a non-empty comma-separated float list")
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
        raise RuntimeError("no PS2-RL design-region states selected")
    return np.concatenate(xs), np.concatenate(ls)


def _quat_delta(dtheta: np.ndarray) -> np.ndarray:
    d = np.asarray(dtheta, dtype=np.float64)
    angle = float(np.linalg.norm(d))
    if angle < 1e-12:
        return normalize_quaternion_np(np.array([1.0, 0.5 * d[0], 0.5 * d[1], 0.5 * d[2]]))
    axis = d / angle
    half = 0.5 * angle
    return np.concatenate([[np.cos(half)], np.sin(half) * axis])


def _boxplus_np(x: np.ndarray, dz: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    dz = np.asarray(dz, dtype=np.float64)
    out = x.copy()
    out[:3] += dz[:3]
    out[3:6] += dz[3:6]
    q = normalize_quaternion_np(x[6:10])
    out[6:10] = normalize_quaternion_np(quaternion_multiply_np(q, _quat_delta(dz[6:9])))
    return out


def _boxminus_np(x_ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    xr = np.asarray(x_ref, dtype=np.float64)
    xx = np.asarray(x, dtype=np.float64)
    q0 = normalize_quaternion_np(xr[6:10])
    q1 = normalize_quaternion_np(xx[6:10])
    qe = normalize_quaternion_np(quaternion_multiply_np(quaternion_conjugate_np(q0), q1))
    if qe[0] < 0.0:
        qe = -qe
    vn = float(np.linalg.norm(qe[1:4]))
    if vn < 1e-12:
        dtheta = 2.0 * qe[1:4]
    else:
        angle = 2.0 * np.arctan2(vn, max(float(qe[0]), 0.0))
        dtheta = angle * qe[1:4] / vn
    return np.concatenate([xx[:3] - xr[:3], xx[3:6] - xr[3:6], dtheta])


def _base_values(runtime: Any, xs: np.ndarray) -> np.ndarray:
    fn = jax.jit(jax.vmap(lambda x: runtime.base_set_values_fn(x)[0]))
    return np.asarray(jax.device_get(fn(jnp.asarray(xs, dtype=jnp.float32))), dtype=np.float64)


def _make_a9_fn(runtime: Any):
    def f_cl(x: jax.Array) -> jax.Array:
        return runtime.dynamics_fn(x, runtime.backup_policy_fn(x))

    jac = jax.jacfwd(f_cl)

    def a9(x: jax.Array) -> jax.Array:
        fx = f_cl(x)
        j10 = jac(x)
        b, bp, bdot = _tangent_maps(x, fx)
        return bp @ (j10 @ b - bdot)

    return a9


def _test_tangent_fd(
    *,
    runtime: Any,
    cfg: Any,
    states: np.ndarray,
    dhats: np.ndarray,
    samples: int,
    eps: float,
    seed: int,
    handoff_exclusion: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    hb = _base_values(runtime, states)
    candidates = np.flatnonzero(np.abs(hb) >= float(handoff_exclusion))
    if candidates.size == 0:
        raise RuntimeError("no states remain after tangent-test handoff exclusion")
    take = min(int(samples), int(candidates.size))
    idx = rng.choice(candidates, size=take, replace=False)
    xs = states[idx]
    ds = dhats[rng.integers(0, len(dhats), size=take)]
    dirs = rng.normal(size=(take, 9))
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)

    a9_fn = _make_a9_fn(runtime)
    a9_batch = jax.jit(jax.vmap(a9_fn))

    def disc_one(x: jax.Array, d: jax.Array):
        return discrete_tangent_jacobians(x, d, cfg.dt, runtime)

    disc_batch = jax.jit(jax.vmap(disc_one, in_axes=(0, 0)))
    x_next_j, f9_j, _ = disc_batch(
        jnp.asarray(xs, dtype=jnp.float32), jnp.asarray(ds, dtype=jnp.float32)
    )
    x_next = np.asarray(jax.device_get(x_next_j), dtype=np.float64)
    f9s = np.asarray(jax.device_get(f9_j), dtype=np.float64)

    # Keep the old continuous approximations only as diagnostics.  The exact
    # implemented discrete tangent map F9 is the quantity that should match FD.
    a9s = np.asarray(jax.device_get(a9_batch(jnp.asarray(xs, dtype=jnp.float32))), dtype=np.float64)

    def step_one(x: jax.Array, d: jax.Array) -> jax.Array:
        dx = estimated_disturbance_backup_dynamics(x, d, runtime)
        return runtime.postprocess_rollout_state_fn(x + jnp.asarray(cfg.dt, dtype=x.dtype) * dx)

    step_batch = jax.jit(jax.vmap(step_one, in_axes=(0, 0)))
    x_pert = np.stack([_boxplus_np(x, float(eps) * z) for x, z in zip(xs, dirs)]).astype(np.float32)
    xp_next = np.asarray(
        jax.device_get(step_batch(jnp.asarray(x_pert), jnp.asarray(ds, dtype=jnp.float32))), dtype=np.float64
    )

    fd = np.stack([_boxminus_np(a, b) / float(eps) for a, b in zip(x_next, xp_next)])
    pred_discrete = np.einsum("nij,nj->ni", f9s, dirs)
    eye = np.eye(9, dtype=np.float64)
    pred_euler = np.einsum("nij,nj->ni", eye[None] + float(cfg.dt) * a9s, dirs)
    pred_exp = np.stack([scipy.linalg.expm(float(cfg.dt) * a) @ z for a, z in zip(a9s, dirs)])
    denom = np.maximum(np.linalg.norm(fd, axis=1), 1e-9)
    err_discrete = np.linalg.norm(fd - pred_discrete, axis=1) / denom
    err_euler = np.linalg.norm(fd - pred_euler, axis=1) / denom
    err_exp = np.linalg.norm(fd - pred_exp, axis=1) / denom
    return {
        "samples": int(take),
        "eps": float(eps),
        "handoff_exclusion": float(handoff_exclusion),
        "relative_error_exact_discrete": _stats(err_discrete),
        "relative_error_euler": _stats(err_euler),
        "relative_error_expm": _stats(err_exp),
    }


def _test_theta_fd(
    *,
    runtime: Any,
    cfg: Any,
    states: np.ndarray,
    dhats: np.ndarray,
    samples: int,
    horizon: float,
    eps: float,
    seed: int,
) -> dict[str, Any]:
    n_steps = int(round(float(horizon) / float(cfg.dt)))
    test_cfg = replace(cfg, T=n_steps * float(cfg.dt), num_steps=n_steps)
    rng = np.random.default_rng(seed)
    take = min(int(samples), int(states.shape[0]))
    idx = rng.choice(states.shape[0], size=take, replace=False)
    xs = states[idx].astype(np.float32)
    ds = dhats[rng.integers(0, len(dhats), size=take)].astype(np.float32)

    def sens_one(x: jax.Array, d: jax.Array):
        out = rollout_estimated_disturbance_backup_flow_and_sensitivities(x, d, test_cfg, runtime)
        return out[0], out[2]

    sens_batch = jax.jit(jax.vmap(sens_one, in_axes=(0, 0)))
    xs_roll_j, theta_j = sens_batch(jnp.asarray(xs), jnp.asarray(ds))
    xs_roll = np.asarray(jax.device_get(xs_roll_j), dtype=np.float64)
    theta = np.asarray(jax.device_get(theta_j), dtype=np.float64)[:, -1]

    def state_one(x: jax.Array, d: jax.Array):
        dt = jnp.asarray(test_cfg.dt, dtype=x.dtype)
        def step(z, _):
            dz = estimated_disturbance_backup_dynamics(z, d, runtime)
            zn = runtime.postprocess_rollout_state_fn(z + dt * dz)
            return zn, None
        zf, _ = jax.lax.scan(step, x, xs=None, length=n_steps)
        return zf

    state_batch = jax.jit(jax.vmap(state_one, in_axes=(0, 0)))
    fd_cols = []
    eye3 = np.eye(3, dtype=np.float32)
    for j in range(3):
        dp = ds + float(eps) * eye3[j][None, :]
        dm = ds - float(eps) * eye3[j][None, :]
        xp = np.asarray(jax.device_get(state_batch(jnp.asarray(xs), jnp.asarray(dp))), dtype=np.float64)
        xm = np.asarray(jax.device_get(state_batch(jnp.asarray(xs), jnp.asarray(dm))), dtype=np.float64)
        fd_cols.append((xp - xm) / (2.0 * float(eps)))
    fd = np.stack(fd_cols, axis=2)
    denom = np.maximum(np.linalg.norm(fd.reshape(take, -1), axis=1), 1e-9)
    rel = np.linalg.norm((theta - fd).reshape(take, -1), axis=1) / denom

    flat_roll = xs_roll.reshape((-1, 10)).astype(np.float32)
    hb_flat = np.asarray(
        jax.device_get(jax.jit(jax.vmap(lambda x: runtime.base_set_values_fn(x)[0]))(jnp.asarray(flat_roll))),
        dtype=np.float64,
    )
    hb = hb_flat.reshape((take, n_steps + 1))
    crossing = np.any((hb[:, :-1] < 0.0) & (hb[:, 1:] >= 0.0), axis=1)
    return {
        "samples": int(take),
        "horizon": float(n_steps * float(cfg.dt)),
        "eps": float(eps),
        "relative_fro_error_all": _stats(rel),
        "relative_fro_error_no_handoff": _stats(rel[~crossing]),
        "relative_fro_error_handoff": _stats(rel[crossing]),
        "handoff_fraction": float(np.mean(crossing)),
    }


def _make_rollout_a9_policy_fn(cfg: Any, runtime: Any):
    dt = jnp.asarray(cfg.dt, dtype=jnp.float32)
    n_steps = int(cfg.num_steps)
    a9_fn = _make_a9_fn(runtime)
    jac_u = jax.jacfwd(runtime.backup_policy_fn)

    def node(x: jax.Array):
        u = runtime.backup_policy_fn(x)
        fx = runtime.dynamics_fn(x, u)
        a9 = a9_fn(x)
        b, _, _ = _tangent_maps(x, fx)
        du = jac_u(x) @ b
        hb = runtime.base_set_values_fn(x)[0]
        hs = runtime.safe_set_values_and_grads_fn(x)[0][0]
        return a9, du, hb, hs

    def single(x0: jax.Array, d: jax.Array):
        def step(x, _):
            vals = node(x)
            dx = estimated_disturbance_backup_dynamics(x, d, runtime)
            xn = runtime.postprocess_rollout_state_fn(x + dt * dx)
            return xn, (x, *vals)
        xf, (xh, ah, duh, hbh, hsh) = jax.lax.scan(step, x0, xs=None, length=n_steps)
        af, duf, hbf, hsf = node(xf)
        xs = jnp.concatenate([xh, xf[None]], axis=0)
        aa = jnp.concatenate([ah, af[None]], axis=0)
        du = jnp.concatenate([duh, duf[None]], axis=0)
        hb = jnp.concatenate([hbh, hbf[None]], axis=0)
        hs = jnp.concatenate([hsh, hsf[None]], axis=0)
        return xs, aa, du, hb, hs

    return jax.jit(jax.vmap(single, in_axes=(0, 0)))


def _mu2_np(a: np.ndarray, scale: np.ndarray | None = None) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float64)
    if scale is not None:
        s = np.asarray(scale, dtype=np.float64)
        aa = aa * s[None, :, None] / s[None, None, :]
    sym = 0.5 * (aa + np.swapaxes(aa, 1, 2))
    return np.linalg.eigvalsh(sym)[:, -1]


def _optimize_diagonal_metric(a: np.ndarray, *, steps: int, lr: float, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(a, dtype=np.float32)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(arr))
    split = max(1, int(0.6 * len(arr)))
    train = jnp.asarray(arr[perm[:split]])
    test = arr[perm[split:]] if split < len(arr) else arr[perm[:split]]
    gain = jnp.asarray(0.15, dtype=jnp.float32)

    def loss_fn(log_s):
        log_s = log_s - jnp.mean(log_s)
        log_s = jnp.clip(log_s, -2.5, 2.5)
        s = jnp.exp(log_s)
        z = train * s[None, :, None] / s[None, None, :]
        sym = 0.5 * (z + jnp.swapaxes(z, 1, 2))
        mu = jnp.linalg.eigvalsh(sym)[:, -1]
        smooth_max = (jax.scipy.special.logsumexp(gain * mu) - jnp.log(mu.shape[0])) / gain
        return smooth_max + 1e-4 * jnp.mean(log_s * log_s)

    vg = jax.jit(jax.value_and_grad(loss_fn))
    z = jnp.zeros((9,), dtype=jnp.float32)
    m = jnp.zeros_like(z)
    v = jnp.zeros_like(z)
    b1, b2 = 0.9, 0.999
    for k in range(1, int(steps) + 1):
        _, g = vg(z)
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * g * g
        mh = m / (1.0 - b1**k)
        vh = v / (1.0 - b2**k)
        z = z - float(lr) * mh / (jnp.sqrt(vh) + 1e-8)
        z = jnp.clip(z - jnp.mean(z), -2.5, 2.5)
        z = z - jnp.mean(z)
    scale = np.asarray(jax.device_get(jnp.exp(z)), dtype=np.float64)
    scale /= np.exp(np.mean(np.log(scale)))
    return scale, {
        "train_count": int(train.shape[0]),
        "test_count": int(test.shape[0]),
        "euclidean_test": _stats(_mu2_np(test)),
        "weighted_test": _stats(_mu2_np(test, scale)),
        "scale": scale,
        "scale_min": float(np.min(scale)),
        "scale_max": float(np.max(scale)),
        "condition_number": float(np.max(scale) / np.min(scale)),
        "disturbance_input_gain_weighted": float(np.max(scale[3:6])),
        "weighted_to_physical_radius_factor": float(1.0 / np.min(scale)),
    }


def _analyze_policy_sensitivity_and_horizon(
    *, cfg: Any, runtime: Any, x0: np.ndarray, labels: np.ndarray, dhats: np.ndarray,
    horizons: list[float], batch_size: int, metric_samples: int, metric_steps: int,
    metric_lr: float, seed: int,
) -> dict[str, Any]:
    xp = np.repeat(x0, len(dhats), axis=0).astype(np.float32)
    dp = np.tile(dhats, (len(x0), 1)).astype(np.float32)
    lp = np.repeat(labels, len(dhats))
    fn = _make_rollout_a9_policy_fn(cfg, runtime)
    xs_l, a_l, du_l, hb_l, hs_l = [], [], [], [], []
    for start in range(0, len(xp), int(batch_size)):
        stop = min(start + int(batch_size), len(xp))
        out = fn(jnp.asarray(xp[start:stop]), jnp.asarray(dp[start:stop]))
        vals = [np.asarray(jax.device_get(v)) for v in out]
        xs_l.append(vals[0]); a_l.append(vals[1]); du_l.append(vals[2]); hb_l.append(vals[3]); hs_l.append(vals[4])
        if start == 0 or stop == len(xp) or ((start // int(batch_size)) + 1) % 25 == 0:
            print(f"  rollout/Jacobian {stop}/{len(xp)}")
    xs = np.concatenate(xs_l)
    aa = np.concatenate(a_l)
    du = np.concatenate(du_l)
    hb = np.concatenate(hb_l)
    hs = np.concatenate(hs_l)

    flat_a = aa.reshape((-1, 9, 9)).astype(np.float64)
    flat_du = du.reshape((-1, 4, 9)).astype(np.float64)
    flat_hb = hb.reshape(-1)
    flat_hs = hs.reshape(-1)
    mask = np.isfinite(flat_hb) & np.isfinite(flat_hs) & (flat_hb < 0.0) & (flat_hs >= 0.0)
    idx = np.flatnonzero(mask)
    rng = np.random.default_rng(seed)
    take = min(int(metric_samples), int(idx.size))
    sel = rng.choice(idx, size=take, replace=False) if take < idx.size else idx
    a_s = flat_a[sel]
    du_s = flat_du[sel]
    mu = _mu2_np(a_s)
    alpha = np.max(np.real(np.linalg.eigvals(a_s)), axis=1)
    groups = {
        "position": np.asarray([np.linalg.norm(j[:, 0:3], ord=2) for j in du_s]),
        "velocity": np.asarray([np.linalg.norm(j[:, 3:6], ord=2) for j in du_s]),
        "attitude": np.asarray([np.linalg.norm(j[:, 6:9], ord=2) for j in du_s]),
        "all": np.asarray([np.linalg.norm(j, ord=2) for j in du_s]),
    }
    corr = {}
    lm = np.log1p(np.maximum(mu, 0.0))
    for name, val in groups.items():
        lv = np.log1p(val)
        corr[name] = float(np.corrcoef(lm, lv)[0, 1]) if len(val) > 2 else np.nan
    metric_scale, metric_report = _optimize_diagonal_metric(
        a_s, steps=int(metric_steps), lr=float(metric_lr), seed=int(seed) + 909
    )

    horizon_report: dict[str, Any] = {}
    initial_safe = hs[:, 0] >= 0.0
    for h in horizons:
        k = int(round(float(h) / float(cfg.dt)))
        if k < 1 or k >= hb.shape[1]:
            continue
        safe = np.all(hs[:, : k + 1] >= 0.0, axis=1)
        reach = hb[:, k] >= 0.0
        both = safe & reach
        rec: dict[str, Any] = {
            "reach_base_fraction": float(np.mean(reach)),
            "safe_full_horizon_fraction": float(np.mean(safe)),
            "safe_and_reach_fraction": float(np.mean(both)),
            "initial_safe_fraction": float(np.mean(initial_safe)),
            "safe_and_reach_given_initially_safe": float(np.mean(both[initial_safe])) if np.any(initial_safe) else np.nan,
            "regions": {},
        }
        for name in POOL_NAMES:
            pm = lp == name
            sm = pm & initial_safe
            rec["regions"][name] = {
                "count": int(np.sum(pm)),
                "safe_and_reach_fraction": float(np.mean(both[pm])) if np.any(pm) else np.nan,
                "safe_and_reach_given_initially_safe": float(np.mean(both[sm])) if np.any(sm) else np.nan,
            }
        horizon_report[f"{h:.2f}"] = rec

    return {
        "sampled_outside_base_safe_nodes": int(len(sel)),
        "mu2": _stats(mu),
        "spectral_abscissa": _stats(alpha),
        "mu2_minus_spectral_abscissa": _stats(mu - alpha),
        "policy_jacobian_channel_spectral": {k: _stats(v) for k, v in groups.items()},
        "log1p_mu2_correlation_with_policy_channel": corr,
        "constant_diagonal_metric": metric_report,
        "horizons": horizon_report,
        "metric_scale": metric_scale,
    }


def _sample_base_boundary(controller: QuadrotorDLQR, c_b: float, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = np.asarray(controller.p_matrix, dtype=np.float64)
    vals, vecs = np.linalg.eigh(p)
    pinvhalf = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    states: list[np.ndarray] = []
    attempts = 0
    while len(states) < int(count) and attempts < int(count) * 100:
        attempts += 1
        z = rng.normal(size=7)
        z /= max(float(np.linalg.norm(z)), 1e-12)
        e = np.sqrt(float(c_b)) * (pinvhalf @ z)
        vq = 0.5 * e[4:7]
        vv = float(vq @ vq)
        if vv >= 0.999:
            continue
        qerr = np.concatenate([[np.sqrt(1.0 - vv)], vq])
        q = quaternion_conjugate_np(qerr)
        x = np.zeros((10,), dtype=np.float64)
        x[2] = float(controller.z_des) + e[0]
        x[3:6] = e[1:4]
        x[6:10] = normalize_quaternion_np(q)
        states.append(x)
    if len(states) < int(count):
        raise RuntimeError(f"could only generate {len(states)} valid base-boundary states")
    return np.asarray(states)


def _base_invariance_test(
    *, cfg: Any, controller: QuadrotorDLQR, c_b: float, delta_d: float,
    states: int, directions: int, seed: int,
) -> dict[str, Any]:
    xs = _sample_base_boundary(controller, c_b, states, seed).astype(np.float32)
    rng = np.random.default_rng(seed + 1)
    axes = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=np.float32)
    nr = max(0, int(directions) - len(axes))
    rr = rng.normal(size=(nr, 3)).astype(np.float32)
    if nr:
        rr /= np.maximum(np.linalg.norm(rr, axis=1, keepdims=True), 1e-12)
    dirs = np.concatenate([axes, rr], axis=0)

    p = np.asarray(controller.p_matrix, dtype=np.float64)
    k = np.asarray(controller.k_matrix, dtype=np.float64)
    ueq = np.asarray(controller.u_star, dtype=np.float64)
    ulo = np.asarray(controller.u_low, dtype=np.float64)
    uhi = np.asarray(controller.u_high, dtype=np.float64)

    xj = jnp.asarray(xs)
    e = np.asarray(jax.device_get(controller.error_state(xj)), dtype=np.float64)
    u_raw = ueq[None, :] - e @ k.T
    raw_sat_mask = np.any((u_raw < ulo[None, :] - 1e-9) | (u_raw > uhi[None, :] + 1e-9), axis=1)
    action_batch = jax.jit(jax.vmap(controller.action))
    step_batch = jax.jit(
        jax.vmap(
            lambda x, u: quadrotor_step_euler(
                x, u, cfg.dt, cfg.gravity, cfg.a_cmd_min, cfg.a_cmd_max, cfg.omega_max
            )
        )
    )
    u = action_batch(xj)
    xn0 = np.asarray(jax.device_get(step_batch(xj, u)), dtype=np.float32)
    cand = np.broadcast_to(xn0[:, None, :], (len(xs), len(dirs), 10)).copy()
    cand[:, :, 3:6] += float(cfg.dt) * float(delta_d) * dirs[None, :, :]
    ec = np.asarray(
        jax.device_get(controller.error_state(jnp.asarray(cand.reshape((-1, 10))))),
        dtype=np.float64,
    ).reshape((len(xs), len(dirs), 7))
    vv = np.einsum("...i,ij,...j->...", ec, p, ec)
    worst_flat = int(np.argmax(vv))
    wi, wj = np.unravel_index(worst_flat, vv.shape)
    violations = vv > float(c_b) + 1e-6
    return {
        "boundary_states": int(len(xs)),
        "directions": int(len(dirs)),
        "pairs": int(vv.size),
        "delta_d": float(delta_d),
        "c_B": float(c_b),
        "worst_V_next": float(vv[wi, wj]),
        "margin_cB_minus_worst": float(c_b - vv[wi, wj]),
        "violation_count": int(np.sum(violations)),
        "violation_fraction": float(np.mean(violations)),
        "boundary_states_with_raw_action_saturation": int(np.sum(raw_sat_mask)),
        "sampled_pass": bool((not np.any(violations)) and (not np.any(raw_sat_mask))),
        "worst_witness": {
            "x": xs[wi].astype(np.float64),
            "direction": dirs[wj].astype(np.float64),
            "V_next": float(vv[wi, wj]),
        },
    }


def _make_channel_comparison_fn(cfg: Any, runtime: Any, *, amplitude: float, frequency_hz: float, observer_time: float):
    dt = jnp.asarray(cfg.dt, dtype=jnp.float32)
    n_steps = int(cfg.num_steps)
    ed = quadrotor_disturbance_injection_matrix(dtype=jnp.float32)
    amp = jnp.asarray(float(amplitude), dtype=jnp.float32)
    omg = jnp.asarray(2.0 * pi * float(frequency_hz), dtype=jnp.float32)
    t0 = jnp.asarray(float(observer_time), dtype=jnp.float32)
    a9_fn = _make_a9_fn(runtime)

    def fcl(x):
        return runtime.dynamics_fn(x, runtime.backup_policy_fn(x))

    def single(x0, dhat, direction, phase):
        def step(carry, k):
            xh, xt = carry
            tau = dt * k.astype(xh.dtype)
            xhn, f9, g9 = discrete_tangent_jacobians(xh, dhat, dt, runtime)
            dtrue = amp * jnp.sin(omg * (t0 + tau) + phase) * direction
            dxt = fcl(xt) + ed @ dtrue
            xtn = runtime.postprocess_rollout_state_fn(xt + dt * dxt)
            return (xhn, xtn), (xh, xt, f9, g9)
        (xhf, xtf), (xhh, xth, fh, gh) = jax.lax.scan(step, (x0, x0), jnp.arange(n_steps))
        xhs = jnp.concatenate([xhh, xhf[None]], axis=0)
        xts = jnp.concatenate([xth, xtf[None]], axis=0)
        return xhs, xts, fh, gh

    return jax.jit(jax.vmap(single, in_axes=(0, 0, 0, 0)))


def _channel_kernel_radius(
    f9: np.ndarray, g9: np.ndarray, *, dt: float, delta_v: float, ebar: float
) -> np.ndarray:
    f9 = np.asarray(f9, dtype=np.float64)
    g9 = np.asarray(g9, dtype=np.float64)
    if f9.shape[0] != g9.shape[0]:
        raise ValueError("F9/G9 step counts must match")
    nodes = f9.shape[0] + 1
    r = np.zeros((nodes,), dtype=np.float64)
    generators: list[np.ndarray] = []
    for k in range(f9.shape[0]):
        f = f9[k]
        g = g9[k]
        generators = [f @ m for m in generators]
        # Lemma-4 mismatch bound grows over the backup horizon.  The upper
        # endpoint is used for the whole Euler interval, which is conservative.
        q_end = float(delta_v) * float((k + 1) * dt) + float(ebar)
        generators.append(g * q_end)
        r[k + 1] = float(sum(np.linalg.norm(m, ord=2) for m in generators))
    return r


def _channel_tube_test(
    *, cfg: Any, runtime: Any, x0: np.ndarray, labels: np.ndarray,
    delta_d: float, frequency_hz: float, scenarios: int, observer_lambda: float,
    observer_time: float, horizon: float, batch_size: int, seed: int,
) -> dict[str, Any]:
    n_steps = int(round(float(horizon) / float(cfg.dt)))
    test_cfg = replace(cfg, T=n_steps * float(cfg.dt), num_steps=n_steps)
    delta_v = 2.0 * pi * float(frequency_hz) * float(delta_d)
    ebar = float(np.asarray(observer_error_bound(
        float(observer_time), delta_d=float(delta_d), delta_v=float(delta_v),
        lambda_gain=float(observer_lambda), dtype=jnp.float64,
    )))
    directions, phases = make_sinusoid_scenarios(int(scenarios), int(seed) + 17041)
    _, dhats = _continuous_observer_estimate_for_sinusoid(
        amplitude=float(delta_d), frequency_hz=float(frequency_hz), phase=phases,
        direction=directions, observer_time=float(observer_time), lambda_gain=float(observer_lambda),
    )
    xp = np.repeat(x0, len(directions), axis=0).astype(np.float32)
    lp = np.repeat(labels, len(directions))
    dp = np.tile(dhats, (len(x0), 1)).astype(np.float32)
    rp = np.tile(directions, (len(x0), 1)).astype(np.float32)
    pp = np.tile(phases, len(x0)).astype(np.float32)
    fn = _make_channel_comparison_fn(
        test_cfg, runtime, amplitude=float(delta_d), frequency_hz=float(frequency_hz), observer_time=float(observer_time)
    )

    terminal_r, terminal_actual, max_ratio = [], [], []
    violations = 0; nodes = 0; traj_viol = 0
    worst = {"ratio": -np.inf}
    for start in range(0, len(xp), int(batch_size)):
        stop = min(start + int(batch_size), len(xp))
        xh_j, xt_j, f_j, g_j = fn(
            jnp.asarray(xp[start:stop]), jnp.asarray(dp[start:stop]),
            jnp.asarray(rp[start:stop]), jnp.asarray(pp[start:stop]),
        )
        xh = np.asarray(jax.device_get(xh_j), dtype=np.float64)
        xt = np.asarray(jax.device_get(xt_j), dtype=np.float64)
        ff = np.asarray(jax.device_get(f_j), dtype=np.float64)
        gg = np.asarray(jax.device_get(g_j), dtype=np.float64)
        for bi in range(stop - start):
            actual = np.asarray([np.linalg.norm(_boxminus_np(a, b)) for a, b in zip(xh[bi], xt[bi])])
            rr = _channel_kernel_radius(ff[bi], gg[bi], dt=float(cfg.dt), delta_v=delta_v, ebar=ebar)
            ratio = np.divide(actual[1:], rr[1:], out=np.full_like(actual[1:], np.inf), where=rr[1:] > 0)
            bad = actual[1:] > rr[1:] * (1.0 + 1e-6) + 1e-9
            violations += int(np.sum(bad)); nodes += int(len(bad)); traj_viol += int(np.any(bad))
            terminal_r.append(rr[-1]); terminal_actual.append(actual[-1]); max_ratio.append(float(np.max(ratio)))
            mi = int(np.argmax(ratio)) + 1
            if float(ratio[mi - 1]) > float(worst["ratio"]):
                gi = start + bi
                worst = {
                    "ratio": float(ratio[mi - 1]), "trajectory_index": int(gi), "region": str(lp[gi]),
                    "tau": float(mi * float(cfg.dt)), "R_channel": float(rr[mi]), "actual": float(actual[mi]),
                    "d_hat": dp[gi], "direction": rp[gi], "phase": float(pp[gi]),
                }
    return {
        "horizon": float(n_steps * float(cfg.dt)), "delta_d": float(delta_d), "frequency_hz": float(frequency_hz),
        "delta_v": float(delta_v), "e_bar": float(ebar), "trajectory_count": int(len(xp)),
        "R_channel_terminal": _stats(terminal_r), "actual_terminal_error_9d": _stats(terminal_actual),
        "max_actual_over_R_per_trajectory": _stats(max_ratio),
        "node_count": int(nodes), "violation_node_count": int(violations),
        "node_coverage_fraction": float(1.0 - violations / max(nodes, 1)),
        "trajectory_violation_count": int(traj_viol),
        "trajectory_coverage_fraction": float(1.0 - traj_viol / max(len(xp), 1)),
        "worst_ratio_witness": worst,
        "formal_status": "exact-discrete centerline first-order disturbance-channel diagnostic; not yet a nonlinear tube certificate",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reset-library", required=True)
    p.add_argument("--learned-backup-policy-path", required=True)
    p.add_argument("--domain-source", choices=("paper", "library"), default="paper")
    p.add_argument("--samples-per-region", type=int, default=512)
    p.add_argument("--flow-initial-per-region", type=int, default=16)
    p.add_argument("--dhat-random", type=int, default=6)
    p.add_argument("--disturbance-bound", type=float, default=2.0)
    p.add_argument("--horizons", default="0.4,0.6,0.8,1.0,1.2,1.5,2.0")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--tangent-fd-samples", type=int, default=256)
    p.add_argument("--tangent-fd-eps", type=float, default=1e-4)
    p.add_argument("--handoff-exclusion", type=float, default=0.10)
    p.add_argument("--theta-fd-samples", type=int, default=48)
    p.add_argument("--theta-fd-horizon", type=float, default=0.4)
    p.add_argument("--theta-fd-eps", type=float, default=1e-3)
    p.add_argument("--metric-samples", type=int, default=6000)
    p.add_argument("--metric-steps", type=int, default=250)
    p.add_argument("--metric-lr", type=float, default=0.03)
    p.add_argument("--base-boundary-states", type=int, default=4096)
    p.add_argument("--base-disturbance-directions", type=int, default=96)
    p.add_argument("--channel-flow-initial-per-region", type=int, default=4)
    p.add_argument("--channel-scenarios", type=int, default=9)
    p.add_argument("--channel-horizon", type=float, default=1.0)
    p.add_argument("--max-frequency-hz", type=float, default=0.5)
    p.add_argument("--observer-lambda", type=float, default=20.0)
    p.add_argument("--observer-time", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="outputs/ue_remaining_changes_analysis.json")
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
    learned = load_learned_quadrotor_backup_policy(str(args.learned_backup_policy_path))
    payload = learned.metadata.get("lqr_config")
    controller = QuadrotorDLQR.from_payload(payload, fallback=cfg) if isinstance(payload, dict) else QuadrotorDLQR.from_config(cfg)

    domain, domain_labels = sample_ps2_domain(
        lib, samples_per_region=int(args.samples_per_region), seed=int(args.seed), domain_source=str(args.domain_source)
    )
    x0, labels = _select_starts(domain, domain_labels, int(args.flow_initial_per_region))
    dhats = make_dhat_samples(float(args.disturbance_bound), int(args.dhat_random), int(args.seed) + 8881)

    print("\n=== Test 1: 9-D tangent finite-difference validation ===")
    tangent = _test_tangent_fd(
        runtime=runtime, cfg=cfg, states=domain, dhats=dhats, samples=int(args.tangent_fd_samples),
        eps=float(args.tangent_fd_eps), seed=int(args.seed) + 1, handoff_exclusion=float(args.handoff_exclusion),
    )
    _print_stats("relative error / exact discrete tangent step", tangent["relative_error_exact_discrete"])
    _print_stats("relative error / Euler tangent step", tangent["relative_error_euler"])
    _print_stats("relative error / expm tangent step", tangent["relative_error_expm"])

    print("\n=== Test 2: Theta finite-difference validation ===")
    theta = _test_theta_fd(
        runtime=runtime, cfg=cfg, states=domain, dhats=dhats, samples=int(args.theta_fd_samples),
        horizon=float(args.theta_fd_horizon), eps=float(args.theta_fd_eps), seed=int(args.seed) + 2,
    )
    _print_stats("Theta FD relative error / all", theta["relative_fro_error_all"])
    _print_stats("Theta FD relative error / no handoff", theta["relative_fro_error_no_handoff"])
    _print_stats("Theta FD relative error / handoff", theta["relative_fro_error_handoff"])

    print("\n=== Test 3: policy sensitivity, non-normality, metric, horizon ===")
    policy = _analyze_policy_sensitivity_and_horizon(
        cfg=cfg, runtime=runtime, x0=x0, labels=labels, dhats=dhats, horizons=_parse_csv(args.horizons),
        batch_size=int(args.batch_size), metric_samples=int(args.metric_samples), metric_steps=int(args.metric_steps),
        metric_lr=float(args.metric_lr), seed=int(args.seed) + 3,
    )
    _print_stats("mu2(A9)", policy["mu2"])
    _print_stats("spectral abscissa(A9)", policy["spectral_abscissa"])
    for name, rec in policy["policy_jacobian_channel_spectral"].items():
        _print_stats(f"||du/d{name}||2", rec)
    print("metric scale:", np.asarray(policy["constant_diagonal_metric"]["scale"]))
    _print_stats("weighted mu2 / held-out", policy["constant_diagonal_metric"]["weighted_test"])
    for h, rec in policy["horizons"].items():
        print(
            f"T={h}s: reach={100*rec['reach_base_fraction']:.2f}% "
            f"safe+reach={100*rec['safe_and_reach_fraction']:.2f}% "
            f"safe+reach | initially-safe={100*rec['safe_and_reach_given_initially_safe']:.2f}%"
        )

    print("\n=== Test 4: nonlinear sampled base-set robust invariance ===")
    base = _base_invariance_test(
        cfg=cfg, controller=controller, c_b=float(cfg.base_set_c), delta_d=float(args.disturbance_bound),
        states=int(args.base_boundary_states), directions=int(args.base_disturbance_directions), seed=int(args.seed) + 4,
    )
    print(
        f"worst V_next={base['worst_V_next']:.6f} vs c_B={base['c_B']:.6f}; "
        f"violations={base['violation_count']}/{base['pairs']}; raw-sat boundary states={base['boundary_states_with_raw_action_saturation']}"
    )

    print("\n=== Test 5: disturbance-channel-specific linearized tube ===")
    ch_x0, ch_labels = _select_starts(domain, domain_labels, int(args.channel_flow_initial_per_region))
    channel = _channel_tube_test(
        cfg=cfg, runtime=runtime, x0=ch_x0, labels=ch_labels,
        delta_d=float(args.disturbance_bound), frequency_hz=float(args.max_frequency_hz),
        scenarios=int(args.channel_scenarios), observer_lambda=float(args.observer_lambda),
        observer_time=float(args.observer_time), horizon=float(args.channel_horizon),
        batch_size=int(args.batch_size), seed=int(args.seed) + 5,
    )
    _print_stats("R_channel(T)", channel["R_channel_terminal"])
    _print_stats("actual physical-9D error(T)", channel["actual_terminal_error_9d"])
    _print_stats("max actual/R_channel", channel["max_actual_over_R_per_trajectory"])
    print(
        f"node coverage={100*channel['node_coverage_fraction']:.6f}% "
        f"trajectory coverage={100*channel['trajectory_coverage_fraction']:.6f}%"
    )

    report = {
        "settings": vars(args),
        "tangent_fd": tangent,
        "theta_fd": theta,
        "policy_metric_horizon": policy,
        "base_robust_invariance": base,
        "channel_tube": channel,
        "decision_notes": {
            "change_1": "E_d and Theta are accepted only if Theta finite differences are small away from handoffs.",
            "change_2": "The 9-D tangent model is accepted only if one-step manifold finite differences are small.",
            "policy_regularization": "Use channel Jacobian statistics/correlations to decide which state channels need regularization.",
            "metric": "A large mu2-alpha gap and a much smaller held-out weighted mu2 indicate coordinate non-normality that a constant metric may reduce.",
            "horizon": "Choose the shortest horizon with acceptable safe+reach performance, preferably conditioned on initially safe states.",
            "base": "Sampled robust-invariance pass is evidence only; it is not a formal global proof.",
            "channel_tube": "R_channel is a linearized centerline diagnostic; it must not be called a nonlinear certificate yet.",
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(report), indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
