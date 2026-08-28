#!/usr/bin/env python
"""Combined UE-bCBF diagnostics for PS2-RL quadrotor.

This script runs three no-retraining experiments in one pass:

1) Quaternion geometry:
   Compare the Euclidean log norm of the raw ambient 10-D Jacobian against a
   9-D physical tangent-space Jacobian using perturbation coordinates

       dz = [dp(3), dv(3), dtheta(3)].

   For the right-multiplicative quaternion tangent convention,

       dq = 0.5 Xi(q) dtheta,

   and the moving-frame tangent Jacobian is

       A9 = B^+ (J10 B - Bdot).

2) Backup-policy sensitivity:
   Run the exact same Omega x d_hat experiment with both the learned Phase-I
   safe-arrival backup and the existing analytic PID+LQR backup.  In addition
   to closed-loop mu_2, evaluate ||du_b/dz||_2 on a subsample of flow states.

3) Horizon and disturbance-rate sensitivity:
   Using each policy's full 2 s rollout, evaluate shorter horizons without
   rerunning the dynamics.  Report reach-B/safety rates and integrate the
   centerline differential bound

       dR/dtau = mu_2(tau) R + delta_v tau + e_bar(t0)

   for both the 10-D and 9-D log norms across a delta_v sweep.

The 9-D result removes the nonphysical radial quaternion perturbation, but the
hard SA/LQR handoff remains piecewise differentiable.  All maxima are sampled
empirical diagnostics, not formal global certificates.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from analyze_quadrotor_lhb_methods import POOL_NAMES, make_dhat_samples, sample_ps2_domain
from ps2rl.cil.quadrotor_backup_cbf import make_backup_runtime
from ps2rl.cil.quadrotor_ue_rollout import estimated_disturbance_backup_dynamics
from ps2rl.evaluation.quadrotor_trace_reset_lib import QuadrotorResetLibrary
from ps2rl.utils.quaternion import quaternion_rate_matrix


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50.0)),
        "p90": float(np.percentile(values, 90.0)),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
        "p99_9": float(np.percentile(values, 99.9)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def _print_stats(name: str, stats: dict[str, Any]) -> None:
    if int(stats.get("count", 0)) == 0:
        print(f"{name}: no samples")
        return
    print(
        f"{name}: n={stats['count']}, p50={stats['p50']:.5f}, "
        f"p95={stats['p95']:.5f}, p99={stats['p99']:.5f}, "
        f"p99.9={stats['p99_9']:.5f}, max={stats['max']:.5f}"
    )


def _select_flow_starts(
    domain_states: np.ndarray,
    domain_labels: np.ndarray,
    per_region: int,
) -> tuple[np.ndarray, np.ndarray]:
    starts: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for pool_name in POOL_NAMES:
        idx = np.flatnonzero(domain_labels == pool_name)
        if idx.size == 0:
            continue
        take = min(int(per_region), int(idx.size))
        starts.append(domain_states[idx[:take]])
        labels.append(np.asarray([pool_name] * take, dtype=object))
    if not starts:
        raise RuntimeError("No PS2-RL domain states available for flow analysis.")
    return np.concatenate(starts, axis=0), np.concatenate(labels, axis=0)


def _tangent_maps(x: jax.Array, f_cl_x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return B, B^+, Bdot for dz=[dp,dv,dtheta] with right quaternion error.

    For unit q, Xi(q)^T Xi(q)=I, so the quaternion tangent block

        Bq = 0.5 Xi(q)

    has pseudoinverse 2 Xi(q)^T.
    """

    q = x[6:10]
    qdot = f_cl_x[6:10]
    xi = quaternion_rate_matrix(q)
    xi_dot = quaternion_rate_matrix(qdot)  # Xi is linear in q.

    b = jnp.zeros((10, 9), dtype=x.dtype)
    b = b.at[0:6, 0:6].set(jnp.eye(6, dtype=x.dtype))
    b = b.at[6:10, 6:9].set(0.5 * xi)

    b_plus = jnp.zeros((9, 10), dtype=x.dtype)
    b_plus = b_plus.at[0:6, 0:6].set(jnp.eye(6, dtype=x.dtype))
    b_plus = b_plus.at[6:9, 6:10].set(2.0 * xi.T)

    b_dot = jnp.zeros((10, 9), dtype=x.dtype)
    b_dot = b_dot.at[6:10, 6:9].set(0.5 * xi_dot)
    return b, b_plus, b_dot


def _make_rollout_metric_fn(cfg: Any, runtime: Any):
    dt = jnp.asarray(cfg.dt, dtype=jnp.float32)
    num_steps = int(cfg.num_steps)

    def f_cl(x: jax.Array) -> jax.Array:
        return runtime.dynamics_fn(x, runtime.backup_policy_fn(x))

    jac_cl = jax.jacfwd(f_cl)

    def node_metrics(x: jax.Array):
        fx = f_cl(x)
        j10 = jac_cl(x)
        sym10 = 0.5 * (j10 + j10.T)
        mu10 = jnp.linalg.eigvalsh(sym10)[-1]

        b, b_plus, b_dot = _tangent_maps(x, fx)
        a9 = b_plus @ (j10 @ b - b_dot)
        sym9 = 0.5 * (a9 + a9.T)
        mu9 = jnp.linalg.eigvalsh(sym9)[-1]

        if runtime.base_set_values_fn is None:
            h_b = runtime.base_set_values_and_grads_fn(x)[0][0]
        else:
            h_b = runtime.base_set_values_fn(x)[0]
        h_s = runtime.safe_set_values_and_grads_fn(x)[0][0]
        return mu10, mu9, h_b, h_s

    def single(x0: jax.Array, d_hat: jax.Array):
        def step(x: jax.Array, _):
            mu10, mu9, h_b, h_s = node_metrics(x)
            dx = estimated_disturbance_backup_dynamics(x, d_hat, runtime)
            x_next = runtime.postprocess_rollout_state_fn(x + dt * dx)
            return x_next, (x, mu10, mu9, h_b, h_s)

        x_final, (xs_head, mu10_head, mu9_head, hb_head, hs_head) = jax.lax.scan(
            step, x0, xs=None, length=num_steps
        )
        mu10_f, mu9_f, hb_f, hs_f = node_metrics(x_final)
        xs = jnp.concatenate([xs_head, x_final[None, :]], axis=0)
        mu10 = jnp.concatenate([mu10_head, mu10_f[None]], axis=0)
        mu9 = jnp.concatenate([mu9_head, mu9_f[None]], axis=0)
        hb = jnp.concatenate([hb_head, hb_f[None]], axis=0)
        hs = jnp.concatenate([hs_head, hs_f[None]], axis=0)
        return xs, mu10, mu9, hb, hs

    return jax.jit(jax.vmap(single, in_axes=(0, 0)))


def _make_policy_jac_metric_fn(runtime: Any):
    jac_u = jax.jacfwd(runtime.backup_policy_fn)

    def single(x: jax.Array):
        u = runtime.backup_policy_fn(x)
        f_cl_x = runtime.dynamics_fn(x, u)
        b, _, _ = _tangent_maps(x, f_cl_x)
        du_dx = jac_u(x)
        du_dz = du_dx @ b
        svals = jnp.linalg.svd(du_dz, full_matrices=False, compute_uv=False)
        return svals[0], jnp.linalg.norm(du_dz), jnp.linalg.norm(du_dx)

    return jax.jit(jax.vmap(single, in_axes=0))


def _integrate_centerline_R(
    mu: np.ndarray,
    *,
    dt: float,
    n_steps: int,
    delta_v: float,
    e_bar: float,
) -> np.ndarray:
    """Piecewise-constant-mu integration using an upper-endpoint source.

    On each step, source <= delta_v*tau_{k+1}+e_bar, so using the upper endpoint
    avoids understating the monotonically increasing source term.
    """

    mu = np.asarray(mu, dtype=np.float64)
    r = np.zeros((mu.shape[0],), dtype=np.float64)
    for k in range(int(n_steps)):
        m = mu[:, k]
        source = float(delta_v) * ((k + 1) * float(dt)) + float(e_bar)
        md = np.clip(m * float(dt), -700.0, 700.0)
        exp_md = np.exp(md)
        gain = np.empty_like(m)
        small = np.abs(m) < 1e-10
        gain[small] = float(dt)
        gain[~small] = (exp_md[~small] - 1.0) / m[~small]
        r = exp_md * r + gain * source
    return r


def _parse_float_csv(text: str) -> list[float]:
    vals = [float(v.strip()) for v in str(text).split(",") if v.strip()]
    if not vals:
        raise ValueError("Expected at least one comma-separated float.")
    return vals


def _horizon_indices(horizons: list[float], dt: float, max_steps: int) -> list[tuple[float, int]]:
    out: list[tuple[float, int]] = []
    for h in horizons:
        idx = int(round(float(h) / float(dt)))
        if idx < 1 or idx > int(max_steps):
            raise ValueError(f"Horizon {h} s is outside [dt, {max_steps * dt}] s")
        actual = idx * float(dt)
        if abs(actual - float(h)) > 1e-9:
            raise ValueError(f"Horizon {h} is not an integer multiple of dt={dt}")
        out.append((actual, idx))
    return out


def _observer_bound(delta_d: float, delta_v: float, lam: float, t0: float) -> float:
    decay = np.exp(-float(lam) * float(t0))
    return float(decay * delta_d + (delta_v / lam) * (1.0 - decay))


def _region_stats(mu10: np.ndarray, mu9: np.ndarray, hb: np.ndarray, hs: np.ndarray, handoff_band: float) -> dict[str, Any]:
    finite = np.isfinite(mu10) & np.isfinite(mu9) & np.isfinite(hb) & np.isfinite(hs)
    masks = {
        "all": finite,
        "safe": finite & (hs >= 0.0),
        "outside_base": finite & (hb < 0.0),
        "inside_base": finite & (hb >= 0.0),
        "handoff_band": finite & (np.abs(hb) <= float(handoff_band)),
    }
    return {
        name: {"mu10": _stats(mu10[mask]), "mu9": _stats(mu9[mask])}
        for name, mask in masks.items()
    }


def _analyze_policy(
    *,
    mode: str,
    cfg: Any,
    runtime: Any,
    x_pairs: np.ndarray,
    d_pairs: np.ndarray,
    horizons: list[tuple[float, int]],
    delta_v_values: list[float],
    delta_d: float,
    observer_lambda: float,
    observer_time: float,
    handoff_band: float,
    batch_size: int,
    policy_jac_stride: int,
    policy_jac_max_samples: int,
    seed: int,
) -> dict[str, Any]:
    rollout_fn = _make_rollout_metric_fn(cfg, runtime)

    xs_chunks: list[np.ndarray] = []
    mu10_chunks: list[np.ndarray] = []
    mu9_chunks: list[np.ndarray] = []
    hb_chunks: list[np.ndarray] = []
    hs_chunks: list[np.ndarray] = []

    print(f"\n=== {mode.upper()} backup: rollout + Jacobian analysis ===")
    for start in range(0, x_pairs.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), x_pairs.shape[0])
        xs_j, mu10_j, mu9_j, hb_j, hs_j = rollout_fn(
            jnp.asarray(x_pairs[start:stop]),
            jnp.asarray(d_pairs[start:stop]),
        )
        xs_chunks.append(np.asarray(jax.device_get(xs_j), dtype=np.float32))
        mu10_chunks.append(np.asarray(jax.device_get(mu10_j), dtype=np.float64))
        mu9_chunks.append(np.asarray(jax.device_get(mu9_j), dtype=np.float64))
        hb_chunks.append(np.asarray(jax.device_get(hb_j), dtype=np.float64))
        hs_chunks.append(np.asarray(jax.device_get(hs_j), dtype=np.float64))
        if start == 0 or stop == x_pairs.shape[0] or ((start // int(batch_size)) + 1) % 50 == 0:
            print(f"  processed trajectories {stop}/{x_pairs.shape[0]}")

    xs = np.concatenate(xs_chunks, axis=0)
    mu10 = np.concatenate(mu10_chunks, axis=0)
    mu9 = np.concatenate(mu9_chunks, axis=0)
    hb = np.concatenate(hb_chunks, axis=0)
    hs = np.concatenate(hs_chunks, axis=0)

    region = _region_stats(mu10, mu9, hb, hs, float(handoff_band))
    _print_stats("ambient 10D mu2 / all", region["all"]["mu10"])
    _print_stats("physical 9D mu2 / all", region["all"]["mu9"])
    _print_stats("ambient 10D mu2 / outside B", region["outside_base"]["mu10"])
    _print_stats("physical 9D mu2 / outside B", region["outside_base"]["mu9"])
    _print_stats("physical 9D mu2 / inside B", region["inside_base"]["mu9"])

    # Policy Jacobian diagnostic on a deterministic subsample of flow states.
    stride = max(1, int(policy_jac_stride))
    candidates = xs[:, ::stride, :].reshape((-1, 10))
    hb_candidates = hb[:, ::stride].reshape(-1)
    # Prioritize outside-B states, because inside B both composed policies use LQR.
    outside = candidates[hb_candidates < 0.0]
    inside = candidates[hb_candidates >= 0.0]
    rng = np.random.default_rng(int(seed) + (13 if mode == "learned" else 29))

    def choose(arr: np.ndarray, max_count: int) -> np.ndarray:
        if arr.shape[0] <= max_count:
            return arr
        idx = rng.choice(arr.shape[0], size=int(max_count), replace=False)
        return arr[idx]

    max_jac = max(1, int(policy_jac_max_samples))
    outside_s = choose(outside, max_jac)
    inside_s = choose(inside, min(max_jac // 4, inside.shape[0])) if inside.shape[0] else inside
    policy_metric_fn = _make_policy_jac_metric_fn(runtime)

    def eval_policy_jac(arr: np.ndarray) -> dict[str, Any]:
        if arr.shape[0] == 0:
            return {"count": 0}
        vals: list[np.ndarray] = []
        for start in range(0, arr.shape[0], int(batch_size) * 8):
            out = policy_metric_fn(jnp.asarray(arr[start : start + int(batch_size) * 8]))
            vals.append(np.stack([np.asarray(jax.device_get(v), dtype=np.float64) for v in out], axis=1))
        mat = np.concatenate(vals, axis=0)
        return {
            "count": int(mat.shape[0]),
            "spectral_tangent": _stats(mat[:, 0]),
            "frobenius_tangent": _stats(mat[:, 1]),
            "frobenius_ambient": _stats(mat[:, 2]),
        }

    policy_jac = {
        "outside_base": eval_policy_jac(outside_s),
        "inside_base": eval_policy_jac(inside_s),
    }
    if policy_jac["outside_base"].get("count", 0):
        _print_stats(
            "||du_b/dz||_2 / outside B",
            policy_jac["outside_base"]["spectral_tangent"],
        )

    horizon_results: dict[str, Any] = {}
    for horizon, n_steps in horizons:
        terminal_hb = hb[:, n_steps]
        safe_full = np.all(hs[:, : n_steps + 1] >= 0.0, axis=1)
        reach_b = terminal_hb >= 0.0
        both = safe_full & reach_b

        h_key = f"{horizon:.2f}"
        h_result: dict[str, Any] = {
            "horizon_s": float(horizon),
            "step_index": int(n_steps),
            "reach_base_fraction": float(np.mean(reach_b)),
            "safe_full_horizon_fraction": float(np.mean(safe_full)),
            "safe_and_reach_base_fraction": float(np.mean(both)),
            "per_flow_max_mu10": _stats(np.max(mu10[:, : n_steps + 1], axis=1)),
            "per_flow_max_mu9": _stats(np.max(mu9[:, : n_steps + 1], axis=1)),
            "delta_v_sweep": {},
        }

        for delta_v in delta_v_values:
            ebar = _observer_bound(delta_d, delta_v, observer_lambda, observer_time)
            r10 = _integrate_centerline_R(
                mu10, dt=float(cfg.dt), n_steps=n_steps, delta_v=delta_v, e_bar=ebar
            )
            r9 = _integrate_centerline_R(
                mu9, dt=float(cfg.dt), n_steps=n_steps, delta_v=delta_v, e_bar=ebar
            )
            dv_key = f"{delta_v:.6g}"
            h_result["delta_v_sweep"][dv_key] = {
                "delta_v": float(delta_v),
                "e_bar_at_observer_time": float(ebar),
                "R10_terminal": _stats(r10),
                "R9_terminal": _stats(r9),
            }

        horizon_results[h_key] = h_result
        print(
            f"T={horizon:.2f}s: reach B={100*h_result['reach_base_fraction']:.2f}% | "
            f"safe={100*h_result['safe_full_horizon_fraction']:.2f}% | "
            f"safe+reach={100*h_result['safe_and_reach_base_fraction']:.2f}%"
        )
        default_dv_key = f"{delta_v_values[-1]:.6g}"
        r10s = h_result["delta_v_sweep"][default_dv_key]["R10_terminal"]
        r9s = h_result["delta_v_sweep"][default_dv_key]["R9_terminal"]
        print(
            f"  delta_v={delta_v_values[-1]:.4f}: median R10={r10s['p50']:.3e}, "
            f"median R9={r9s['p50']:.3e}"
        )

    max10_flat = int(np.nanargmax(mu10))
    max9_flat = int(np.nanargmax(mu9))
    traj10, node10 = np.unravel_index(max10_flat, mu10.shape)
    traj9, node9 = np.unravel_index(max9_flat, mu9.shape)

    return {
        "mode": mode,
        "region_stats": region,
        "policy_jacobian": policy_jac,
        "horizons": horizon_results,
        "max_witness_10d": {
            "mu2": float(mu10[traj10, node10]),
            "trajectory_index": int(traj10),
            "node_index": int(node10),
            "tau": float(node10 * float(cfg.dt)),
            "state": xs[traj10, node10],
            "h_B": float(hb[traj10, node10]),
            "h_S": float(hs[traj10, node10]),
            "d_hat": d_pairs[traj10],
        },
        "max_witness_9d": {
            "mu2": float(mu9[traj9, node9]),
            "trajectory_index": int(traj9),
            "node_index": int(node9),
            "tau": float(node9 * float(cfg.dt)),
            "state": xs[traj9, node9],
            "h_B": float(hb[traj9, node9]),
            "h_S": float(hs[traj9, node9]),
            "d_hat": d_pairs[traj9],
        },
        "handoff_sign_crossings": int(np.sum((hb[:, :-1] < 0.0) != (hb[:, 1:] < 0.0))),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reset-library", required=True)
    p.add_argument("--learned-backup-policy-path", required=True)
    p.add_argument("--domain-source", choices=("paper", "library"), default="paper")
    p.add_argument("--samples-per-region", type=int, default=512)
    p.add_argument("--flow-initial-per-region", type=int, default=64)
    p.add_argument("--disturbance-bound", type=float, default=2.0)
    p.add_argument("--dhat-random", type=int, default=12)
    p.add_argument("--horizons", default="0.4,0.6,1.0,1.5,2.0")
    p.add_argument(
        "--delta-v-values",
        default="0.15,0.5,1.0,3.14159265,6.28318531",
        help="Sensitivity sweep only; these are not all claimed to be physical wind bounds.",
    )
    p.add_argument("--observer-lambda", type=float, default=20.0)
    p.add_argument("--observer-time", type=float, default=0.2)
    p.add_argument("--handoff-band", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--policy-jac-stride", type=int, default=10)
    p.add_argument("--policy-jac-max-samples", type=int, default=12000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="outputs/ue_geometry_policy_horizon_analysis.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    reset_path = Path(args.reset_library).expanduser().resolve()
    if not reset_path.exists():
        raise FileNotFoundError(reset_path)
    learned_path = str(args.learned_backup_policy_path).strip()
    if not learned_path:
        raise SystemExit("--learned-backup-policy-path is required")
    if args.disturbance_bound < 0.0:
        raise SystemExit("--disturbance-bound must be nonnegative")
    if args.observer_lambda <= 0.0:
        raise SystemExit("--observer-lambda must be positive")

    horizons_raw = _parse_float_csv(args.horizons)
    delta_v_values = _parse_float_csv(args.delta_v_values)
    if any(v < 0.0 for v in delta_v_values):
        raise SystemExit("--delta-v-values must be nonnegative")

    library = QuadrotorResetLibrary.load(reset_path)
    base_cfg = library.cbf_cfg
    horizons = _horizon_indices(horizons_raw, float(base_cfg.dt), int(base_cfg.num_steps))

    domain_states, domain_labels = sample_ps2_domain(
        library,
        samples_per_region=int(args.samples_per_region),
        seed=int(args.seed),
        domain_source=str(args.domain_source),
    )
    x0, x0_labels = _select_flow_starts(
        domain_states, domain_labels, int(args.flow_initial_per_region)
    )
    dhats = make_dhat_samples(
        float(args.disturbance_bound), int(args.dhat_random), int(args.seed + 8881)
    )
    x_pairs = np.repeat(x0, dhats.shape[0], axis=0).astype(np.float32)
    d_pairs = np.tile(dhats, (x0.shape[0], 1)).astype(np.float32)
    pair_labels = np.repeat(x0_labels, dhats.shape[0])

    print("\n=== Combined UE-bCBF diagnostic ===")
    print(f"Omega initial states          : {x0.shape[0]}")
    print(f"d_hat samples                 : {dhats.shape[0]}")
    print(f"trajectory pairs              : {x_pairs.shape[0]}")
    print(f"base rollout dt               : {float(base_cfg.dt):.4f} s")
    print(f"base rollout T                : {float(base_cfg.T):.4f} s")
    print(f"delta_d                       : {float(args.disturbance_bound):.4f} m/s^2")
    print(f"horizon sweep                 : {[h for h, _ in horizons]}")
    print(f"delta_v sweep                 : {delta_v_values}")
    print("NOTE: 0.15 m/s^3 matches the official UE-bCBF planar-quadrotor example's time-varying disturbance rate;")
    print("      6.283... m/s^3 is the current A=2, f_max=0.5 Hz assumption.")

    results: dict[str, Any] = {
        "settings": {
            "reset_library": str(reset_path),
            "learned_backup_policy_path": learned_path,
            "domain_source": args.domain_source,
            "samples_per_region": int(args.samples_per_region),
            "flow_initial_per_region": int(args.flow_initial_per_region),
            "delta_d": float(args.disturbance_bound),
            "dhat_sample_count": int(dhats.shape[0]),
            "trajectory_count": int(x_pairs.shape[0]),
            "dt": float(base_cfg.dt),
            "T": float(base_cfg.T),
            "horizons": [float(h) for h, _ in horizons],
            "delta_v_values": delta_v_values,
            "observer_lambda": float(args.observer_lambda),
            "observer_time": float(args.observer_time),
            "pair_region_counts": {name: int(np.sum(pair_labels == name)) for name in POOL_NAMES},
        },
        "notes": {
            "tangent_convention": "right-multiplicative body-frame dtheta with dq = 0.5 Xi(q) dtheta",
            "A9_formula": "Bplus @ (J10 @ B - Bdot)",
            "formal_status": "sampled diagnostic only; hard SA/LQR switch remains piecewise differentiable",
        },
        "policies": {},
    }

    for mode in ("learned", "analytic"):
        cfg = replace(
            base_cfg,
            backup_policy_mode=mode,
            learned_backup_policy_path=learned_path if mode == "learned" else "",
        )
        runtime = make_backup_runtime(cfg)
        results["policies"][mode] = _analyze_policy(
            mode=mode,
            cfg=cfg,
            runtime=runtime,
            x_pairs=x_pairs,
            d_pairs=d_pairs,
            horizons=horizons,
            delta_v_values=delta_v_values,
            delta_d=float(args.disturbance_bound),
            observer_lambda=float(args.observer_lambda),
            observer_time=float(args.observer_time),
            handoff_band=float(args.handoff_band),
            batch_size=int(args.batch_size),
            policy_jac_stride=int(args.policy_jac_stride),
            policy_jac_max_samples=int(args.policy_jac_max_samples),
            seed=int(args.seed),
        )

    print("\n=== Direct learned vs analytic comparison ===")
    for coord in ("mu10", "mu9"):
        l = results["policies"]["learned"]["region_stats"]["outside_base"][coord]
        a = results["policies"]["analytic"]["region_stats"]["outside_base"][coord]
        print(
            f"outside-B {coord}: learned max={l.get('max', float('nan')):.5f}, "
            f"analytic max={a.get('max', float('nan')):.5f}; "
            f"learned p95={l.get('p95', float('nan')):.5f}, analytic p95={a.get('p95', float('nan')):.5f}"
        )

    lj = results["policies"]["learned"]["policy_jacobian"]["outside_base"].get("spectral_tangent", {})
    aj = results["policies"]["analytic"]["policy_jacobian"]["outside_base"].get("spectral_tangent", {})
    if lj and aj:
        print(
            f"outside-B ||du/dz||2: learned p95={lj['p95']:.5f}, max={lj['max']:.5f}; "
            f"analytic p95={aj['p95']:.5f}, max={aj['max']:.5f}"
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(_jsonable(results), f, indent=2, sort_keys=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
