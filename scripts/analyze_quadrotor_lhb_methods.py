#!/usr/bin/env python
"""Compare three ways of obtaining the quadrotor base-barrier Lipschitz bound.

The quantity of interest is

    L_hB >= sup_x ||grad h_B(x)||_2,

for the PS2-RL quadrotor terminal barrier

    h_B(x) = c_B - e(x)^T P e(x).

This script compares:

1) Analytic ellipsoid bound.
   In reduced hover-error coordinates e, the exact bound on B is

       L_hB,reduced = 2 sqrt(c_B lambda_max(P)).

   For the full 10-D state used by the implementation, the quaternion error
   map has local Jacobian norm <= 2 away from its 180-deg sign seam, giving
   the conservative bound

       L_hB,full <= 4 sqrt(c_B lambda_max(P)).

2) PS2-RL design-domain sampling.
   Sample states from the quadrotor trace/reset pools using either the paper
   perturbation envelope (default: 0.4 m, 1.5 m/s, 30 deg tilt, 12 deg yaw)
   or the exact settings stored in a reset_library.pkl, then evaluate the
   actual JAX gradient used by PS2-RL.

3) UE terminal reachable-domain sampling.
   Start from a subset of the same PS2-RL domain, freeze admissible d_hat
   values with ||d_hat|| <= delta_d, roll the learned/analytic backup policy
   for T seconds, keep terminal states that reach B, optionally enlarge those
   terminals by a user-supplied Euclidean tube radius, and evaluate the same
   full-state JAX gradient.

Method 2 and 3 report observed maxima; they are empirical validation values,
not formal global certificates.  Once delta_max(T,t) is available, rerun
Method 3 with --tube-radius set to that value.
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

from ps2rl.base_controller.quadrotor_dlqr import QuadrotorDLQR
from ps2rl.cil.quadrotor_backup_cbf import QuadrotorBCBFConfig, make_backup_runtime
from ps2rl.cil.quadrotor_ue_rollout import estimated_disturbance_backup_dynamics
from ps2rl.evaluation.quadrotor_trace_reset_lib import QuadrotorResetLibrary
from ps2rl.sets.base_sets import EllipsoidBaseSet
from ps2rl.utils.quaternion import (
    normalize_quaternion_np,
    quaternion_from_euler_zyx_np,
    quaternion_multiply_np,
)


POOL_NAMES = ("general_trace", "near_ceiling", "bridge", "base_shell")


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


def _normalize_states_quaternion(states: np.ndarray) -> np.ndarray:
    out = np.asarray(states, dtype=np.float64).copy()
    if out.size == 0:
        return out.reshape((-1, 10))
    out[:, 6:10] = normalize_quaternion_np(out[:, 6:10])
    return out


def _paper_perturb_state(
    anchor: np.ndarray,
    *,
    rng: np.random.Generator,
    safe_set: Any,
    max_tries: int = 100,
) -> np.ndarray:
    """Sample the perturbation envelope described for the PS2-RL quadrotor."""

    anchor = np.asarray(anchor, dtype=np.float64).reshape(10)
    for _ in range(max_tries):
        x = anchor.copy()
        x[0:3] += rng.uniform(-0.40, 0.40, size=3)
        x[3:6] += rng.uniform(-1.50, 1.50, size=3)
        q_delta = quaternion_from_euler_zyx_np(
            float(rng.uniform(-np.deg2rad(30.0), np.deg2rad(30.0))),
            float(rng.uniform(-np.deg2rad(30.0), np.deg2rad(30.0))),
            float(rng.uniform(-np.deg2rad(12.0), np.deg2rad(12.0))),
        )
        x[6:10] = normalize_quaternion_np(quaternion_multiply_np(q_delta, anchor[6:10]))
        if bool(np.asarray(safe_set.contains(x), dtype=bool)):
            return x
    x = anchor.copy()
    x[6:10] = normalize_quaternion_np(x[6:10])
    if not bool(np.asarray(safe_set.contains(x), dtype=bool)):
        raise RuntimeError("Could not sample a safe paper-domain perturbation.")
    return x


def sample_ps2_domain(
    library: QuadrotorResetLibrary,
    *,
    samples_per_region: int,
    seed: int,
    domain_source: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return task-relevant PS2-RL domain samples and region labels."""

    states_out: list[np.ndarray] = []
    labels_out: list[str] = []
    rng = np.random.default_rng(seed)

    for region_idx, pool_name in enumerate(POOL_NAMES):
        anchors = np.asarray(library.all_pools.get(pool_name, np.zeros((0, 10))), dtype=np.float64)
        if anchors.shape[0] == 0:
            continue
        if domain_source == "library":
            samples = library.sample_perturbed_region_states(
                pool_name,
                count=int(samples_per_region),
                seed=int(seed + 1009 * (region_idx + 1)),
                curriculum_scale=1.0,
                split=None,
            )
        else:
            samples_list = []
            for _ in range(int(samples_per_region)):
                idx = int(rng.integers(0, anchors.shape[0]))
                samples_list.append(
                    _paper_perturb_state(
                        anchors[idx],
                        rng=rng,
                        safe_set=library.safe_set,
                    )
                )
            samples = np.asarray(samples_list, dtype=np.float64)
        states_out.append(samples)
        labels_out.extend([pool_name] * int(samples.shape[0]))

    if not states_out:
        raise RuntimeError("No PS2-RL reset pools were available in the supplied reset library.")
    states = _normalize_states_quaternion(np.concatenate(states_out, axis=0))
    labels = np.asarray(labels_out)
    return states, labels


def make_dhat_samples(delta_d: float, random_count: int, seed: int) -> np.ndarray:
    """Zero, axis extrema, random sphere points, and random interior points."""

    delta = float(delta_d)
    if delta < 0.0:
        raise ValueError("delta_d must be nonnegative")
    samples = [np.zeros(3, dtype=np.float64)]
    if delta == 0.0:
        return np.asarray(samples)

    eye = np.eye(3, dtype=np.float64)
    for axis in eye:
        samples.append(delta * axis)
        samples.append(-delta * axis)

    rng = np.random.default_rng(seed)
    n_boundary = max(0, int(random_count) // 2)
    n_interior = max(0, int(random_count) - n_boundary)
    for _ in range(n_boundary):
        v = rng.normal(size=3)
        v /= max(np.linalg.norm(v), 1e-12)
        samples.append(delta * v)
    for _ in range(n_interior):
        v = rng.normal(size=3)
        v /= max(np.linalg.norm(v), 1e-12)
        # Uniform-in-volume radius for a 3-D ball.
        r = delta * float(rng.random() ** (1.0 / 3.0))
        samples.append(r * v)
    return np.asarray(samples, dtype=np.float64)


def _gradient_batch(base_set: EllipsoidBaseSet, states: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate h_B and ||grad_x h_B|| using the actual PS2-RL JAX function."""

    value_grad = jax.jit(jax.vmap(base_set.values_and_grads))
    values: list[np.ndarray] = []
    norms: list[np.ndarray] = []
    states = np.asarray(states, dtype=np.float32)
    for start in range(0, states.shape[0], int(batch_size)):
        xb = jnp.asarray(states[start : start + int(batch_size)])
        h, dh = value_grad(xb)
        h_np = np.asarray(jax.device_get(h), dtype=np.float64)[:, 0]
        dh_np = np.asarray(jax.device_get(dh), dtype=np.float64)[:, 0, :]
        values.append(h_np)
        norms.append(np.linalg.norm(dh_np, axis=1))
    return np.concatenate(values), np.concatenate(norms)


def _error_levels(controller: QuadrotorDLQR, states: np.ndarray, batch_size: int) -> np.ndarray:
    error_batch = jax.jit(jax.vmap(controller.error_state))
    p = np.asarray(controller.p_matrix, dtype=np.float64)
    levels: list[np.ndarray] = []
    states = np.asarray(states, dtype=np.float32)
    for start in range(0, states.shape[0], int(batch_size)):
        err = np.asarray(
            jax.device_get(error_batch(jnp.asarray(states[start : start + int(batch_size)]))),
            dtype=np.float64,
        )
        levels.append(np.einsum("bi,ij,bj->b", err, p, err))
    return np.concatenate(levels)


def _stats(values: np.ndarray, states: np.ndarray, h_values: np.ndarray, levels: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    idx = int(np.argmax(values))
    q = _normalize_states_quaternion(states)[:, 6:10]
    abs_qw = np.abs(q[:, 0])
    return {
        "count": int(values.size),
        "p50": float(np.percentile(values, 50.0)),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
        "p99_9": float(np.percentile(values, 99.9)),
        "max": float(values[idx]),
        "max_index": idx,
        "max_state": np.asarray(states[idx], dtype=np.float64),
        "max_state_hB": float(h_values[idx]),
        "max_state_V": float(levels[idx]),
        "min_abs_qw": float(np.min(abs_qw)),
        "count_abs_qw_lt_0_05": int(np.sum(abs_qw < 0.05)),
        "count_abs_qw_lt_0_10": int(np.sum(abs_qw < 0.10)),
        "max_V": float(np.max(levels)),
    }


def _rollout_terminal_batch(
    x0_batch: np.ndarray,
    d_hat_batch: np.ndarray,
    *,
    cfg: QuadrotorBCBFConfig,
    runtime: Any,
    batch_size: int,
) -> np.ndarray:
    """Lightweight UE terminal rollout; no Phi/Theta/J are needed for L_hB."""

    dt = jnp.asarray(cfg.dt, dtype=jnp.float32)
    num_steps = int(cfg.num_steps)

    def terminal_single(x0: jax.Array, d_hat: jax.Array) -> jax.Array:
        def step(x, _):
            dx = estimated_disturbance_backup_dynamics(x, d_hat, runtime)
            x_next = runtime.postprocess_rollout_state_fn(x + dt * dx)
            return x_next, None

        xf, _ = jax.lax.scan(step, x0, xs=None, length=num_steps)
        return xf

    terminal_batch = jax.jit(jax.vmap(terminal_single, in_axes=(0, 0)))
    out: list[np.ndarray] = []
    x0_batch = np.asarray(x0_batch, dtype=np.float32)
    d_hat_batch = np.asarray(d_hat_batch, dtype=np.float32)
    for start in range(0, x0_batch.shape[0], int(batch_size)):
        xf = terminal_batch(
            jnp.asarray(x0_batch[start : start + int(batch_size)]),
            jnp.asarray(d_hat_batch[start : start + int(batch_size)]),
        )
        out.append(np.asarray(jax.device_get(xf), dtype=np.float64))
    return _normalize_states_quaternion(np.concatenate(out, axis=0))


def _expand_terminal_tube(states: np.ndarray, radius: float, random_dirs: int, seed: int) -> np.ndarray:
    """Empirically sample a Euclidean shell around terminal states."""

    states = _normalize_states_quaternion(states)
    radius = float(radius)
    if radius <= 0.0 or states.shape[0] == 0:
        return states

    dirs: list[np.ndarray] = []
    eye = np.eye(10, dtype=np.float64)
    for axis in eye:
        dirs.append(axis)
        dirs.append(-axis)
    rng = np.random.default_rng(seed)
    for _ in range(max(0, int(random_dirs))):
        v = rng.normal(size=10)
        v /= max(np.linalg.norm(v), 1e-12)
        dirs.append(v)
    dmat = np.asarray(dirs, dtype=np.float64)

    perturbed = states[:, None, :] + radius * dmat[None, :, :]
    perturbed = perturbed.reshape((-1, 10))
    perturbed[:, 6:10] = normalize_quaternion_np(perturbed[:, 6:10])
    return np.concatenate([states, perturbed], axis=0)


def _print_stats(name: str, stats: dict[str, Any]) -> None:
    if int(stats.get("count", 0)) == 0:
        print(f"{name}: no samples")
        return
    print(
        f"{name}: n={stats['count']}, "
        f"p95={stats['p95']:.6f}, p99={stats['p99']:.6f}, "
        f"p99.9={stats['p99_9']:.6f}, max={stats['max']:.6f}, "
        f"max V={stats['max_V']:.6f}, min|qw|={stats['min_abs_qw']:.6f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset-library", required=True, help="Phase-I reset_library.pkl")
    parser.add_argument(
        "--backup-policy-mode",
        choices=("learned", "analytic"),
        default="learned",
        help="Use learned Phase-I backup for Method 3; analytic is useful for smoke tests.",
    )
    parser.add_argument("--learned-backup-policy-path", default="")
    parser.add_argument(
        "--domain-source",
        choices=("paper", "library"),
        default="paper",
        help="paper = 0.4m/1.5m/s/30deg/12deg; library = saved reset-library settings.",
    )
    parser.add_argument("--samples-per-region", type=int, default=512)
    parser.add_argument(
        "--flow-initial-per-region",
        type=int,
        default=64,
        help="Number of Method-2 samples per region reused as Method-3 rollout starts.",
    )
    parser.add_argument("--disturbance-bound", type=float, default=2.0)
    parser.add_argument("--dhat-random", type=int, default=12)
    parser.add_argument(
        "--tube-radius",
        type=float,
        default=0.0,
        help="Set later to delta_max(T,t). Zero gives the pre-tube Method-3 comparison.",
    )
    parser.add_argument("--tube-random-directions", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/ue_lhb_comparison.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reset_path = Path(args.reset_library).expanduser().resolve()
    if not reset_path.exists():
        raise FileNotFoundError(reset_path)
    if args.backup_policy_mode == "learned" and not str(args.learned_backup_policy_path).strip():
        raise SystemExit("--learned-backup-policy-path is required for --backup-policy-mode learned")

    library = QuadrotorResetLibrary.load(reset_path)
    cfg = replace(
        library.cbf_cfg,
        backup_policy_mode=str(args.backup_policy_mode),
        learned_backup_policy_path=str(args.learned_backup_policy_path).strip(),
    )
    controller = QuadrotorDLQR.from_config(cfg)
    base_set = EllipsoidBaseSet(
        controller,
        float(cfg.base_set_c),
        smooth_gain=float(cfg.base_set_smooth_gain),
    )

    # -------------------------- Method 1: analytic -------------------------- #
    p = np.asarray(controller.p_matrix, dtype=np.float64)
    lambda_max_p = float(np.max(np.linalg.eigvalsh(p)))
    c_b = float(cfg.base_set_c)
    l_reduced_exact = float(2.0 * np.sqrt(c_b * lambda_max_p))
    # Away from the q_w=0 shortest-quaternion seam, ||de/dx||_2 <= 2.
    l_full_conservative = float(2.0 * l_reduced_exact)

    print("\n=== Method 1: analytic ellipsoid bound ===")
    print(f"c_B                         : {c_b:.6f}")
    print(f"lambda_max(P)               : {lambda_max_p:.6f}")
    print(f"exact reduced-coordinate L  : {l_reduced_exact:.6f}")
    print(f"conservative full-state L   : {l_full_conservative:.6f}")
    print("NOTE: full-state bound is local away from the q_w=0 sign seam and is for B itself.")

    # ---------------------- Method 2: original PS2 domain ------------------ #
    domain_states, domain_labels = sample_ps2_domain(
        library,
        samples_per_region=int(args.samples_per_region),
        seed=int(args.seed),
        domain_source=str(args.domain_source),
    )
    h2, g2 = _gradient_batch(base_set, domain_states, int(args.batch_size))
    v2 = _error_levels(controller, domain_states, int(args.batch_size))
    stats2 = _stats(g2, domain_states, h2, v2)

    print("\n=== Method 2: PS2-RL design-domain empirical gradient ===")
    print(f"domain source                : {args.domain_source}")
    print(f"samples total                : {domain_states.shape[0]}")
    _print_stats("PS2 domain ||grad h_B||", stats2)
    if stats2.get("count_abs_qw_lt_0_05", 0):
        print(
            "WARNING: sampled PS2 domain approaches/crosses the q_w=0 shortest-quaternion seam; "
            "a raw global gradient-supremum argument is not sufficient across that discontinuity."
        )

    # ----------------------- Method 3: UE terminal domain ----------------- #
    flow_starts: list[np.ndarray] = []
    for pool_name in POOL_NAMES:
        idx = np.flatnonzero(domain_labels == pool_name)
        if idx.size == 0:
            continue
        take = min(int(args.flow_initial_per_region), int(idx.size))
        flow_starts.append(domain_states[idx[:take]])
    x0 = np.concatenate(flow_starts, axis=0)
    dhats = make_dhat_samples(float(args.disturbance_bound), int(args.dhat_random), int(args.seed + 8881))
    x_pairs = np.repeat(x0, dhats.shape[0], axis=0)
    d_pairs = np.tile(dhats, (x0.shape[0], 1))

    runtime = make_backup_runtime(cfg)
    terminals = _rollout_terminal_batch(
        x_pairs,
        d_pairs,
        cfg=cfg,
        runtime=runtime,
        batch_size=int(args.batch_size),
    )
    h3_all, g3_all = _gradient_batch(base_set, terminals, int(args.batch_size))
    v3_all = _error_levels(controller, terminals, int(args.batch_size))
    stats3_all = _stats(g3_all, terminals, h3_all, v3_all)

    in_base = h3_all >= 0.0
    base_terminals = terminals[in_base]
    if base_terminals.shape[0] == 0:
        raise RuntimeError("No Method-3 terminal rollout reached B; cannot form a base-relevant terminal domain.")

    tube_states = _expand_terminal_tube(
        base_terminals,
        float(args.tube_radius),
        int(args.tube_random_directions),
        int(args.seed + 9917),
    )
    h3, g3 = _gradient_batch(base_set, tube_states, int(args.batch_size))
    v3 = _error_levels(controller, tube_states, int(args.batch_size))
    stats3 = _stats(g3, tube_states, h3, v3)

    print("\n=== Method 3: UE terminal reachable-domain empirical gradient ===")
    print(f"backup mode                  : {args.backup_policy_mode}")
    print(f"initial states               : {x0.shape[0]}")
    print(f"d_hat samples                : {dhats.shape[0]}")
    print(f"terminal rollouts            : {terminals.shape[0]}")
    print(f"terminal rollouts in B       : {base_terminals.shape[0]} ({100.0 * np.mean(in_base):.2f}%)")
    print(f"tube radius                  : {float(args.tube_radius):.6f}")
    _print_stats("all terminal ||grad h_B||", stats3_all)
    _print_stats("base-relevant+tube ||grad h_B||", stats3)
    if float(args.tube_radius) <= 0.0:
        print("NOTE: Method 3 is PRE-TUBE. Rerun later with --tube-radius delta_max(T,t).")

    # Analytic envelope bounds using the largest observed V in each sampled domain.
    analytic_from_v2 = float(4.0 * np.sqrt(max(float(stats2["max_V"]), 0.0) * lambda_max_p))
    analytic_from_v3 = float(4.0 * np.sqrt(max(float(stats3["max_V"]), 0.0) * lambda_max_p))

    print("\n=== Comparison ===")
    print(f"Method 1 full-state analytic(B)        : {l_full_conservative:.6f}")
    print(f"Method 2 observed max (PS2 domain)     : {stats2['max']:.6f}")
    print(f"Method 3 observed max (UE terminal)    : {stats3['max']:.6f}")
    print(f"Analytic envelope using Method-2 max V : {analytic_from_v2:.6f}")
    print(f"Analytic envelope using Method-3 max V : {analytic_from_v3:.6f}")

    result = {
        "settings": {
            "reset_library": reset_path,
            "backup_policy_mode": args.backup_policy_mode,
            "learned_backup_policy_path": args.learned_backup_policy_path,
            "domain_source": args.domain_source,
            "samples_per_region": args.samples_per_region,
            "flow_initial_per_region": args.flow_initial_per_region,
            "delta_d": args.disturbance_bound,
            "dhat_sample_count": int(dhats.shape[0]),
            "tube_radius": args.tube_radius,
            "seed": args.seed,
        },
        "method1_analytic": {
            "base_set_c": c_b,
            "lambda_max_P": lambda_max_p,
            "L_reduced_exact_on_B": l_reduced_exact,
            "L_full_conservative_on_B": l_full_conservative,
            "caveat": "full-state bound assumes normalized quaternion and avoids q_w=0 shortest-error seam",
        },
        "method2_ps2_domain": stats2,
        "method3_all_terminals": stats3_all,
        "method3_base_relevant_tube": stats3,
        "comparison": {
            "analytic_envelope_from_method2_max_V": analytic_from_v2,
            "analytic_envelope_from_method3_max_V": analytic_from_v3,
        },
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(_jsonable(result), f, indent=2)
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
