#!/usr/bin/env python
"""Compare global, per-flow, and trajectory-wise UE-bCBF flow bounds.

This is a diagnostic step between the global-c experiment and integration of
uncertainty tightening into the PS2-RL CIL.

For each task-relevant initial state and sinusoidal disturbance scenario, the
script constructs two backup flows from the same initial state:

    estimated flow:  phi_hat_dot = f_cl(phi_hat) + E_d d_hat(t0)
    true flow:       phi_d_dot   = f_cl(phi_d)   + E_d d(t0 + tau)

The observer estimate d_hat(t0) is generated consistently with the scalar
observer dynamics

    d_hat_dot = lambda (d - d_hat),  d_hat(0) = 0,

for the selected sinusoidal disturbance.  It is then frozen over the backup
horizon, as required by UE-bCBF.

Three uncertainty-radius constructions are compared:

1) Global Lemma-6 bound using one global c.
2) Per-flow Lemma-6 bound using

       c_flow = max_tau mu_2(J_cl(phi_hat(tau))).

3) Trajectory-wise centerline radius R using the local matrix measure along
   the estimated flow:

       c(tau) = mu_2(J_cl(phi_hat(tau))),
       R_dot  = c(tau) R + delta_v tau + e_bar(t0),  R(0) = 0.

The third construction is intentionally a *diagnostic*, not yet a formal
certificate.  A rigorous trajectory-wise tube must upper-bound the matrix
measure over the whole uncertainty tube around phi_hat, not only on its
centerline.  This script checks empirically whether the centerline version is
useful before adding that extra robustification.
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
import numpy as np

from analyze_quadrotor_lhb_methods import POOL_NAMES, sample_ps2_domain
from ps2rl.cil.quadrotor_backup_cbf import make_backup_runtime
from ps2rl.cil.quadrotor_ue_rollout import quadrotor_disturbance_injection_matrix
from ps2rl.evaluation.quadrotor_trace_reset_lib import QuadrotorResetLibrary
from ps2rl.uncertainty.quadrotor_disturbance_observer import observer_error_bound


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
        f"{name}: n={stats['count']}, "
        f"p50={stats['p50']:.6g}, p95={stats['p95']:.6g}, "
        f"p99={stats['p99']:.6g}, p99.9={stats['p99_9']:.6g}, "
        f"max={stats['max']:.6g}"
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
        raise RuntimeError("No PS2-RL domain states were available for UE flow analysis.")
    return np.concatenate(starts, axis=0), np.concatenate(labels, axis=0)


def make_sinusoid_scenarios(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return unit directions and phases for worst-amplitude/frequency tests."""

    count = int(count)
    if count <= 0:
        raise ValueError("scenario count must be positive")

    rng = np.random.default_rng(seed)
    dirs: list[np.ndarray] = []
    phases: list[float] = []

    # Deterministic axis cases include both zero crossing (max d_dot) and peak
    # disturbance.  Sign reversal is already represented by phase + pi.
    eye = np.eye(3, dtype=np.float64)
    for axis in eye:
        for phase in (0.0, 0.5 * pi):
            if len(dirs) >= count:
                break
            dirs.append(axis.copy())
            phases.append(float(phase))
        if len(dirs) >= count:
            break

    while len(dirs) < count:
        v = rng.normal(size=3)
        v /= max(float(np.linalg.norm(v)), 1e-12)
        dirs.append(v)
        phases.append(float(rng.uniform(0.0, 2.0 * pi)))

    return np.asarray(dirs, dtype=np.float64), np.asarray(phases, dtype=np.float64)


def _continuous_observer_estimate_for_sinusoid(
    *,
    amplitude: float,
    frequency_hz: float,
    phase: np.ndarray,
    direction: np.ndarray,
    observer_time: float,
    lambda_gain: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact d(t0) and d_hat(t0) for d_hat_dot=lambda(d-d_hat), d_hat(0)=0."""

    amp = float(amplitude)
    omega = 2.0 * pi * float(frequency_hz)
    lam = float(lambda_gain)
    t = float(observer_time)
    phase = np.asarray(phase, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)

    angle = omega * t + phase
    scalar_d = amp * np.sin(angle)

    denom = lam * lam + omega * omega
    particular_t = amp * lam / denom * (lam * np.sin(angle) - omega * np.cos(angle))
    particular_0 = amp * lam / denom * (lam * np.sin(phase) - omega * np.cos(phase))
    scalar_hat = particular_t - np.exp(-lam * t) * particular_0

    return scalar_d[:, None] * direction, scalar_hat[:, None] * direction


def _lemma6_bound(tau: np.ndarray, *, c: np.ndarray | float, delta_v: float, e_bar: float) -> np.ndarray:
    """Evaluate Lemma-6 delta_max with a numerically stable c -> 0 limit."""

    tau = np.asarray(tau, dtype=np.float64)
    c_arr = np.asarray(c, dtype=np.float64)
    tau_b, c_b = np.broadcast_arrays(tau, c_arr)
    out = np.empty_like(tau_b, dtype=np.float64)
    small = np.abs(c_b) < 1e-8
    out[small] = 0.5 * float(delta_v) * tau_b[small] ** 2 + float(e_bar) * tau_b[small]
    if np.any(~small):
        cc = c_b[~small]
        tt = tau_b[~small]
        out[~small] = (
            (float(delta_v) / (cc * cc) + float(e_bar) / cc) * np.expm1(cc * tt)
            - (float(delta_v) / cc) * tt
        )
    return out


def _trajectory_radius(mu: np.ndarray, *, dt: float, delta_v: float, e_bar: float) -> np.ndarray:
    """Integrate centerline R with a conservative frozen-step update.

    For step k, use c_step=max(mu_k, mu_{k+1}) and the forcing at the end of
    the step q_end=delta_v*tau_{k+1}+e_bar.  With c_step frozen, the exact
    solution for constant forcing q_end is used.  This avoids explicit-Euler
    instability and slightly overbounds the linearly increasing forcing within
    each step.  It does *not* robustify c away from the center trajectory.
    """

    mu = np.asarray(mu, dtype=np.float64)
    if mu.ndim != 2:
        raise ValueError(f"mu must have shape (batch, nodes), got {mu.shape}")
    batch, nodes = mu.shape
    r = np.zeros((batch, nodes), dtype=np.float64)
    for k in range(nodes - 1):
        c_step = np.maximum(mu[:, k], mu[:, k + 1])
        tau_end = float(k + 1) * float(dt)
        q_end = float(delta_v) * tau_end + float(e_bar)
        z = c_step * float(dt)
        exp_z = np.exp(np.clip(z, -700.0, 700.0))
        gain = np.empty_like(c_step)
        small = np.abs(c_step) < 1e-8
        gain[small] = float(dt)
        gain[~small] = np.expm1(z[~small]) / c_step[~small]
        r[:, k + 1] = exp_z * r[:, k] + q_end * gain
    return r


def _load_global_c(path: str, fallback: float | None) -> float:
    if str(path).strip():
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        data = json.loads(p.read_text())
        return float(data["mu2"]["observed_c_candidate"])
    if fallback is None:
        raise ValueError("Provide --c-analysis or --global-c")
    return float(fallback)


def _make_batched_comparison_fn(cfg: Any, runtime: Any, *, amplitude: float, frequency_hz: float, observer_time: float):
    dt = jnp.asarray(cfg.dt, dtype=jnp.float32)
    num_steps = int(cfg.num_steps)
    e_d = quadrotor_disturbance_injection_matrix(dtype=jnp.float32)
    amp = jnp.asarray(amplitude, dtype=jnp.float32)
    omega = jnp.asarray(2.0 * pi * frequency_hz, dtype=jnp.float32)
    t0 = jnp.asarray(observer_time, dtype=jnp.float32)

    def f_cl(x: jax.Array) -> jax.Array:
        return runtime.dynamics_fn(x, runtime.backup_policy_fn(x))

    jac_cl = jax.jacfwd(f_cl)

    def mu2_at(x: jax.Array) -> jax.Array:
        j_cl = jac_cl(x)
        sym = 0.5 * (j_cl + j_cl.T)
        return jnp.linalg.eigvalsh(sym)[-1]

    def single(x0: jax.Array, d_hat: jax.Array, direction: jax.Array, phase: jax.Array):
        def step(carry, k):
            x_hat, x_true = carry
            tau = dt * k.astype(x_hat.dtype)
            mu = mu2_at(x_hat)

            dx_hat = f_cl(x_hat) + e_d @ d_hat
            d_true = amp * jnp.sin(omega * (t0 + tau) + phase) * direction
            dx_true = f_cl(x_true) + e_d @ d_true

            x_hat_next = runtime.postprocess_rollout_state_fn(x_hat + dt * dx_hat)
            x_true_next = runtime.postprocess_rollout_state_fn(x_true + dt * dx_true)
            return (x_hat_next, x_true_next), (x_hat, x_true, mu)

        (x_hat_f, x_true_f), (xh_head, xt_head, mu_head) = jax.lax.scan(
            step,
            (x0, x0),
            jnp.arange(num_steps),
        )
        mu_f = mu2_at(x_hat_f)
        xh = jnp.concatenate([xh_head, x_hat_f[None, :]], axis=0)
        xt = jnp.concatenate([xt_head, x_true_f[None, :]], axis=0)
        mu = jnp.concatenate([mu_head, mu_f[None]], axis=0)
        return xh, xt, mu

    return jax.jit(jax.vmap(single, in_axes=(0, 0, 0, 0)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset-library", required=True)
    parser.add_argument("--backup-policy-mode", choices=("learned", "analytic"), default="learned")
    parser.add_argument("--learned-backup-policy-path", default="")
    parser.add_argument("--domain-source", choices=("paper", "library"), default="paper")
    parser.add_argument("--samples-per-region", type=int, default=512)
    parser.add_argument("--flow-initial-per-region", type=int, default=64)
    parser.add_argument("--disturbance-bound", type=float, default=2.0)
    parser.add_argument("--max-frequency-hz", type=float, default=0.5)
    parser.add_argument("--disturbance-scenarios", type=int, default=19)
    parser.add_argument("--observer-lambda", type=float, default=20.0)
    parser.add_argument(
        "--observer-time",
        type=float,
        default=0.2,
        help="Physical estimator time t0 at which d_hat is frozen for the backup prediction.",
    )
    parser.add_argument(
        "--c-analysis",
        default="outputs/ue_c_bound_analysis.json",
        help="Step-16 JSON used to obtain the global sampled c.",
    )
    parser.add_argument("--global-c", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/ue_trajectory_tube_analysis.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.disturbance_bound < 0.0:
        raise SystemExit("--disturbance-bound must be nonnegative")
    if args.max_frequency_hz < 0.0:
        raise SystemExit("--max-frequency-hz must be nonnegative")
    if args.observer_lambda <= 0.0:
        raise SystemExit("--observer-lambda must be positive")
    if args.observer_time < 0.0:
        raise SystemExit("--observer-time must be nonnegative")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    reset_path = Path(args.reset_library).expanduser().resolve()
    if not reset_path.exists():
        raise FileNotFoundError(reset_path)
    if args.backup_policy_mode == "learned" and not str(args.learned_backup_policy_path).strip():
        raise SystemExit("--learned-backup-policy-path is required for learned backup mode")

    library = QuadrotorResetLibrary.load(reset_path)
    cfg = replace(
        library.cbf_cfg,
        backup_policy_mode=str(args.backup_policy_mode),
        learned_backup_policy_path=str(args.learned_backup_policy_path).strip(),
    )
    runtime = make_backup_runtime(cfg)

    global_c = _load_global_c(str(args.c_analysis), args.global_c)
    delta_d = float(args.disturbance_bound)
    delta_v = 2.0 * pi * float(args.max_frequency_hz) * delta_d
    e_bar = float(
        np.asarray(
            observer_error_bound(
                float(args.observer_time),
                delta_d=delta_d,
                delta_v=delta_v,
                lambda_gain=float(args.observer_lambda),
                dtype=jnp.float64,
            )
        )
    )

    domain_states, domain_labels = sample_ps2_domain(
        library,
        samples_per_region=int(args.samples_per_region),
        seed=int(args.seed),
        domain_source=str(args.domain_source),
    )
    x0, x0_labels = _select_flow_starts(
        domain_states,
        domain_labels,
        int(args.flow_initial_per_region),
    )

    directions, phases = make_sinusoid_scenarios(
        int(args.disturbance_scenarios),
        int(args.seed + 17041),
    )
    d_now_scen, d_hat_scen = _continuous_observer_estimate_for_sinusoid(
        amplitude=delta_d,
        frequency_hz=float(args.max_frequency_hz),
        phase=phases,
        direction=directions,
        observer_time=float(args.observer_time),
        lambda_gain=float(args.observer_lambda),
    )
    observer_errors = np.linalg.norm(d_now_scen - d_hat_scen, axis=1)

    # Pair every initial condition with every disturbance scenario.
    ns = directions.shape[0]
    x_pairs = np.repeat(x0, ns, axis=0).astype(np.float32)
    pair_labels = np.repeat(x0_labels, ns)
    d_hat_pairs = np.tile(d_hat_scen, (x0.shape[0], 1)).astype(np.float32)
    direction_pairs = np.tile(directions, (x0.shape[0], 1)).astype(np.float32)
    phase_pairs = np.tile(phases, x0.shape[0]).astype(np.float32)
    scenario_index = np.tile(np.arange(ns, dtype=np.int32), x0.shape[0])

    rollout_batch = _make_batched_comparison_fn(
        cfg,
        runtime,
        amplitude=delta_d,
        frequency_hz=float(args.max_frequency_hz),
        observer_time=float(args.observer_time),
    )

    tau = np.arange(int(cfg.num_steps) + 1, dtype=np.float64) * float(cfg.dt)
    global_curve = _lemma6_bound(tau, c=global_c, delta_v=delta_v, e_bar=e_bar)

    c_flow_all: list[np.ndarray] = []
    per_flow_terminal_all: list[np.ndarray] = []
    local_terminal_all: list[np.ndarray] = []
    actual_terminal_all: list[np.ndarray] = []
    local_max_all: list[np.ndarray] = []
    actual_max_all: list[np.ndarray] = []
    ratio_max_all: list[np.ndarray] = []
    terminal_ratio_all: list[np.ndarray] = []
    violation_count = 0
    compared_node_count = 0
    trajectory_violation_count = 0

    worst_ratio = -np.inf
    worst_payload: dict[str, Any] = {}

    print("\n=== UE trajectory-wise tube diagnostic ===")
    print(f"initial states                 : {x0.shape[0]}")
    print(f"disturbance scenarios          : {ns}")
    print(f"total trajectories             : {x_pairs.shape[0]}")
    print(f"dt / T                         : {float(cfg.dt):.4f} / {float(cfg.T):.4f} s")
    print(f"delta_d / f_max / delta_v      : {delta_d:.6f} / {float(args.max_frequency_hz):.6f} / {delta_v:.6f}")
    print(f"observer lambda / t0           : {float(args.observer_lambda):.6f} / {float(args.observer_time):.6f} s")
    print(f"e_bar(t0)                      : {e_bar:.6f}")
    print(f"max actual observer error      : {float(np.max(observer_errors)):.6f}")
    print(f"global sampled c               : {global_c:.6f}")
    print(f"global Lemma-6 bound at T      : {global_curve[-1]:.6e}")

    batch_size = int(args.batch_size)
    for start in range(0, x_pairs.shape[0], batch_size):
        stop = min(start + batch_size, x_pairs.shape[0])
        xh_j, xt_j, mu_j = rollout_batch(
            jnp.asarray(x_pairs[start:stop]),
            jnp.asarray(d_hat_pairs[start:stop]),
            jnp.asarray(direction_pairs[start:stop]),
            jnp.asarray(phase_pairs[start:stop]),
        )
        xh = np.asarray(jax.device_get(xh_j), dtype=np.float64)
        xt = np.asarray(jax.device_get(xt_j), dtype=np.float64)
        mu = np.asarray(jax.device_get(mu_j), dtype=np.float64)

        actual = np.linalg.norm(xt - xh, axis=2)
        c_flow = np.max(mu, axis=1)
        per_flow = _lemma6_bound(
            tau[None, :],
            c=c_flow[:, None],
            delta_v=delta_v,
            e_bar=e_bar,
        )
        local_r = _trajectory_radius(mu, dt=float(cfg.dt), delta_v=delta_v, e_bar=e_bar)

        # tau=0 has 0/0 by construction; evaluate ratios only after the first node.
        denom = local_r[:, 1:]
        numer = actual[:, 1:]
        ratio = np.divide(numer, denom, out=np.full_like(numer, np.inf), where=denom > 0.0)
        violations = numer > denom * (1.0 + 1e-6) + 1e-9

        violation_count += int(np.sum(violations))
        compared_node_count += int(violations.size)
        trajectory_violation_count += int(np.sum(np.any(violations, axis=1)))

        c_flow_all.append(c_flow)
        per_flow_terminal_all.append(per_flow[:, -1])
        local_terminal_all.append(local_r[:, -1])
        actual_terminal_all.append(actual[:, -1])
        local_max_all.append(np.max(local_r, axis=1))
        actual_max_all.append(np.max(actual, axis=1))
        ratio_max_batch = np.max(ratio, axis=1)
        ratio_max_all.append(ratio_max_batch)
        terminal_ratio = np.divide(
            local_r[:, -1],
            np.maximum(actual[:, -1], 1e-12),
        )
        terminal_ratio_all.append(terminal_ratio)

        flat_idx = int(np.argmax(ratio))
        b_idx, tau_idx0 = np.unravel_index(flat_idx, ratio.shape)
        tau_idx = int(tau_idx0 + 1)
        r_value = float(ratio[b_idx, tau_idx0])
        if r_value > worst_ratio:
            gi = int(start + b_idx)
            worst_ratio = r_value
            worst_payload = {
                "ratio_actual_over_R": r_value,
                "trajectory_index": gi,
                "region": str(pair_labels[gi]),
                "scenario_index": int(scenario_index[gi]),
                "tau": float(tau[tau_idx]),
                "actual_error": float(actual[b_idx, tau_idx]),
                "R_centerline": float(local_r[b_idx, tau_idx]),
                "mu2": float(mu[b_idx, tau_idx]),
                "c_flow": float(c_flow[b_idx]),
                "d_hat": d_hat_pairs[gi].astype(np.float64),
                "direction": direction_pairs[gi].astype(np.float64),
                "phase": float(phase_pairs[gi]),
                "estimated_state": xh[b_idx, tau_idx],
                "true_state": xt[b_idx, tau_idx],
            }

    c_flow_arr = np.concatenate(c_flow_all)
    per_flow_terminal = np.concatenate(per_flow_terminal_all)
    local_terminal = np.concatenate(local_terminal_all)
    actual_terminal = np.concatenate(actual_terminal_all)
    local_max = np.concatenate(local_max_all)
    actual_max = np.concatenate(actual_max_all)
    ratio_max = np.concatenate(ratio_max_all)
    terminal_ratio = np.concatenate(terminal_ratio_all)

    print("\n=== c along each estimated backup flow ===")
    _print_stats("c_flow = max_tau mu2", _stats(c_flow_arr))

    print("\n=== Terminal radius comparison ===")
    print(f"global-c Lemma-6 R(T)          : {global_curve[-1]:.6e}")
    _print_stats("per-flow-max-c Lemma-6 R(T)", _stats(per_flow_terminal))
    _print_stats("trajectory-wise centerline R(T)", _stats(local_terminal))
    _print_stats("actual ||phi_d(T)-phi_hat(T)||", _stats(actual_terminal))

    print("\n=== Whole-horizon empirical validation ===")
    _print_stats("max_tau centerline R", _stats(local_max))
    _print_stats("max_tau actual flow error", _stats(actual_max))
    _print_stats("max_tau actual/R per trajectory", _stats(ratio_max))
    _print_stats("terminal R/actual", _stats(terminal_ratio))
    node_coverage = 1.0 - float(violation_count) / max(1, compared_node_count)
    trajectory_coverage = 1.0 - float(trajectory_violation_count) / max(1, x_pairs.shape[0])
    print(f"nodes satisfying actual <= R   : {100.0 * node_coverage:.6f}%")
    print(f"trajectories fully inside R    : {100.0 * trajectory_coverage:.6f}%")
    print(f"worst actual/R                 : {worst_ratio:.6f}")

    result = {
        "settings": {
            "reset_library": reset_path,
            "backup_policy_mode": args.backup_policy_mode,
            "learned_backup_policy_path": args.learned_backup_policy_path,
            "domain_source": args.domain_source,
            "samples_per_region": args.samples_per_region,
            "flow_initial_per_region": args.flow_initial_per_region,
            "disturbance_scenarios": args.disturbance_scenarios,
            "delta_d": delta_d,
            "f_max_hz": float(args.max_frequency_hz),
            "delta_v": delta_v,
            "observer_lambda": float(args.observer_lambda),
            "observer_time": float(args.observer_time),
            "e_bar": e_bar,
            "global_c": global_c,
            "dt": float(cfg.dt),
            "T": float(cfg.T),
            "num_steps": int(cfg.num_steps),
            "seed": int(args.seed),
        },
        "observer_scenarios": {
            "count": int(ns),
            "actual_error_norm": _stats(observer_errors),
            "d_hat_norm": _stats(np.linalg.norm(d_hat_scen, axis=1)),
            "bound_satisfied": bool(np.all(observer_errors <= e_bar + 1e-8)),
        },
        "global_bound": {
            "c": global_c,
            "terminal": float(global_curve[-1]),
        },
        "per_flow_c": _stats(c_flow_arr),
        "per_flow_max_c_bound_terminal": _stats(per_flow_terminal),
        "trajectory_centerline_bound_terminal": _stats(local_terminal),
        "actual_terminal_error": _stats(actual_terminal),
        "whole_horizon": {
            "centerline_R_max_per_trajectory": _stats(local_max),
            "actual_error_max_per_trajectory": _stats(actual_max),
            "actual_over_R_max_per_trajectory": _stats(ratio_max),
            "terminal_R_over_actual": _stats(terminal_ratio),
            "node_count": int(compared_node_count),
            "violation_node_count": int(violation_count),
            "node_coverage_fraction": float(node_coverage),
            "trajectory_count": int(x_pairs.shape[0]),
            "trajectory_violation_count": int(trajectory_violation_count),
            "trajectory_coverage_fraction": float(trajectory_coverage),
            "worst_actual_over_R": float(worst_ratio),
        },
        "worst_ratio_witness": worst_payload,
        "caveat": (
            "Trajectory-wise R uses mu_2 only on the estimated-flow centerline. "
            "It is an empirical diagnostic, not yet a formal tube certificate. "
            "A rigorous version must upper-bound mu_2(J_cl) over the uncertainty "
            "tube and address the hard PS2-RL backup-policy handoff."
        ),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(result), indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
