#!/usr/bin/env python
"""Estimate the UE-bCBF logarithmic-norm bound c on quadrotor backup flows.

For the Euclidean norm, the induced logarithmic norm of the closed-loop
Jacobian is

    mu_2(J_cl) = lambda_max((J_cl + J_cl.T) / 2).

UE-bCBF Lemma 6 requires a constant c satisfying

    mu_2(J_cl(x)) <= c

on the state domain relevant to the backup flow.  This script constructs a
practical task-relevant domain by:

1) sampling initial states from the original PS2-RL quadrotor design region,
2) freezing admissible disturbance estimates ||d_hat|| <= delta_d,
3) rolling out the composed backup policy for the full backup horizon, and
4) evaluating mu_2(J_cl) at every rollout node.

The maximum reported here is an empirical sampled maximum, not a formal global
certificate.  The script also diagnoses the hard PS2-RL handoff between the
safe-arrival policy and the LQR base controller.  Because that handoff uses a
hard select at h_B = 0, the composed backup policy is generally only piecewise
differentiable; a Jacobian-based c does not by itself bound a finite policy jump
exactly on the switching surface.  We therefore report inside-B, outside-B,
and near-handoff statistics separately.
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

# This script intentionally reuses the exact design-domain sampler introduced
# by step 15 so the L_hB and c experiments use the same Omega and d_hat set.
from analyze_quadrotor_lhb_methods import POOL_NAMES, make_dhat_samples, sample_ps2_domain
from ps2rl.cil.quadrotor_backup_cbf import make_backup_runtime
from ps2rl.cil.quadrotor_ue_rollout import estimated_disturbance_backup_dynamics
from ps2rl.evaluation.quadrotor_trace_reset_lib import QuadrotorResetLibrary


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
        f"p50={stats['p50']:.6f}, p95={stats['p95']:.6f}, "
        f"p99={stats['p99']:.6f}, p99.9={stats['p99_9']:.6f}, "
        f"max={stats['max']:.6f}"
    )


def _make_batched_rollout_metric_fn(cfg: Any, runtime: Any):
    """Return a jitted batch rollout producing states and mu_2(J_cl)."""

    dt = jnp.asarray(cfg.dt, dtype=jnp.float32)
    num_steps = int(cfg.num_steps)

    def f_cl(x: jax.Array) -> jax.Array:
        return runtime.dynamics_fn(x, runtime.backup_policy_fn(x))

    jac_cl = jax.jacfwd(f_cl)

    def node_metrics(x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        j_cl = jac_cl(x)
        sym = 0.5 * (j_cl + j_cl.T)
        mu2 = jnp.linalg.eigvalsh(sym)[-1]
        if runtime.base_set_values_fn is None:
            h_b = runtime.base_set_values_and_grads_fn(x)[0][0]
        else:
            h_b = runtime.base_set_values_fn(x)[0]
        h_s = runtime.safe_set_values_and_grads_fn(x)[0][0]
        return mu2, h_b, h_s

    def single(x0: jax.Array, d_hat: jax.Array):
        def step(x: jax.Array, _):
            mu2, h_b, h_s = node_metrics(x)
            dx = estimated_disturbance_backup_dynamics(x, d_hat, runtime)
            x_next = runtime.postprocess_rollout_state_fn(x + dt * dx)
            return x_next, (x, mu2, h_b, h_s)

        x_final, (xs_head, mu_head, hb_head, hs_head) = jax.lax.scan(
            step,
            x0,
            xs=None,
            length=num_steps,
        )
        mu_f, hb_f, hs_f = node_metrics(x_final)
        xs = jnp.concatenate([xs_head, x_final[None, :]], axis=0)
        mu = jnp.concatenate([mu_head, mu_f[None]], axis=0)
        hb = jnp.concatenate([hb_head, hb_f[None]], axis=0)
        hs = jnp.concatenate([hs_head, hs_f[None]], axis=0)
        return xs, mu, hb, hs

    return jax.jit(jax.vmap(single, in_axes=(0, 0)))


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset-library", required=True, help="Phase-I reset_library.pkl")
    parser.add_argument(
        "--backup-policy-mode",
        choices=("learned", "analytic"),
        default="learned",
    )
    parser.add_argument("--learned-backup-policy-path", default="")
    parser.add_argument(
        "--domain-source",
        choices=("paper", "library"),
        default="paper",
        help="Use the same PS2-RL Omega definition as the L_hB comparison.",
    )
    parser.add_argument(
        "--samples-per-region",
        type=int,
        default=512,
        help="Candidate Omega samples generated per PS2-RL sub-region.",
    )
    parser.add_argument(
        "--flow-initial-per-region",
        type=int,
        default=64,
        help="Number of Omega samples per region used as backup-flow starts.",
    )
    parser.add_argument("--disturbance-bound", type=float, default=2.0)
    parser.add_argument("--dhat-random", type=int, default=12)
    parser.add_argument(
        "--handoff-band",
        type=float,
        default=0.05,
        help="Diagnostic |h_B| band around the hard SA/LQR handoff; does not change c.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of complete backup trajectories processed per GPU batch.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/ue_c_bound_analysis.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reset_path = Path(args.reset_library).expanduser().resolve()
    if not reset_path.exists():
        raise FileNotFoundError(reset_path)
    if args.backup_policy_mode == "learned" and not str(args.learned_backup_policy_path).strip():
        raise SystemExit("--learned-backup-policy-path is required for --backup-policy-mode learned")
    if float(args.disturbance_bound) < 0.0:
        raise SystemExit("--disturbance-bound must be nonnegative")
    if float(args.handoff_band) < 0.0:
        raise SystemExit("--handoff-band must be nonnegative")
    if int(args.batch_size) <= 0:
        raise SystemExit("--batch-size must be positive")

    library = QuadrotorResetLibrary.load(reset_path)
    cfg = replace(
        library.cbf_cfg,
        backup_policy_mode=str(args.backup_policy_mode),
        learned_backup_policy_path=str(args.learned_backup_policy_path).strip(),
    )
    runtime = make_backup_runtime(cfg)

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
    dhats = make_dhat_samples(
        float(args.disturbance_bound),
        int(args.dhat_random),
        int(args.seed + 8881),
    )

    # Pair every selected Omega initial condition with every frozen d_hat.
    x_pairs = np.repeat(x0, dhats.shape[0], axis=0).astype(np.float32)
    d_pairs = np.tile(dhats, (x0.shape[0], 1)).astype(np.float32)
    pair_labels = np.repeat(x0_labels, dhats.shape[0])

    num_nodes = int(cfg.num_steps) + 1
    total_nodes = int(x_pairs.shape[0] * num_nodes)

    print("\n=== UE-bCBF c-domain construction ===")
    print(f"domain source                 : {args.domain_source}")
    print(f"backup mode                   : {args.backup_policy_mode}")
    print(f"backup horizon T              : {float(cfg.T):.6f} s")
    print(f"backup dt                     : {float(cfg.dt):.6f} s")
    print(f"Omega initial states          : {x0.shape[0]}")
    print(f"d_hat samples                 : {dhats.shape[0]}")
    print(f"complete trajectories         : {x_pairs.shape[0]}")
    print(f"Jacobian evaluation nodes     : {total_nodes}")
    print(f"||d_hat|| upper bound         : {float(args.disturbance_bound):.6f} m/s^2")

    rollout_metrics = _make_batched_rollout_metric_fn(cfg, runtime)

    mu_chunks: list[np.ndarray] = []
    hb_chunks: list[np.ndarray] = []
    hs_chunks: list[np.ndarray] = []
    abs_qw_chunks: list[np.ndarray] = []
    transition_count = 0
    nonfinite_count = 0

    witness: dict[str, Any] | None = None
    witness_mu = -np.inf

    batch_size = int(args.batch_size)
    for start in range(0, x_pairs.shape[0], batch_size):
        stop = min(start + batch_size, x_pairs.shape[0])
        xs_j, mu_j, hb_j, hs_j = rollout_metrics(
            jnp.asarray(x_pairs[start:stop]),
            jnp.asarray(d_pairs[start:stop]),
        )
        xs = np.asarray(jax.device_get(xs_j), dtype=np.float64)
        mu = np.asarray(jax.device_get(mu_j), dtype=np.float64)
        hb = np.asarray(jax.device_get(hb_j), dtype=np.float64)
        hs = np.asarray(jax.device_get(hs_j), dtype=np.float64)

        finite = np.isfinite(mu) & np.isfinite(hb) & np.isfinite(hs) & np.all(np.isfinite(xs), axis=-1)
        nonfinite_count += int(np.size(finite) - np.sum(finite))

        mu_chunks.append(mu.reshape(-1))
        hb_chunks.append(hb.reshape(-1))
        hs_chunks.append(hs.reshape(-1))
        abs_qw_chunks.append(np.abs(xs[..., 6]).reshape(-1))

        # Count discrete crossings of the hard handoff surface h_B = 0.
        transition_count += int(np.sum((hb[:, :-1] < 0.0) != (hb[:, 1:] < 0.0)))

        # Global maximum witness, including trajectory metadata.
        finite_mu = np.where(finite, mu, -np.inf)
        local_flat = int(np.argmax(finite_mu))
        local_mu = float(finite_mu.reshape(-1)[local_flat])
        if local_mu > witness_mu:
            local_traj, node_idx = np.unravel_index(local_flat, finite_mu.shape)
            pair_idx = start + int(local_traj)
            witness_mu = local_mu
            witness = {
                "mu2": local_mu,
                "state": xs[local_traj, node_idx],
                "tau": float(node_idx * float(cfg.dt)),
                "node_index": int(node_idx),
                "pair_index": int(pair_idx),
                "initial_state": x_pairs[pair_idx].astype(np.float64),
                "region": str(pair_labels[pair_idx]),
                "d_hat": d_pairs[pair_idx].astype(np.float64),
                "d_hat_norm": float(np.linalg.norm(d_pairs[pair_idx])),
                "h_B": float(hb[local_traj, node_idx]),
                "h_S": float(hs[local_traj, node_idx]),
                "abs_qw": float(abs(xs[local_traj, node_idx, 6])),
            }

        print(f"processed trajectories {stop:5d}/{x_pairs.shape[0]:5d}", end="\r", flush=True)

    print()

    mu_all = np.concatenate(mu_chunks)
    hb_all = np.concatenate(hb_chunks)
    hs_all = np.concatenate(hs_chunks)
    abs_qw_all = np.concatenate(abs_qw_chunks)

    finite_mask = np.isfinite(mu_all) & np.isfinite(hb_all) & np.isfinite(hs_all)
    mu_finite = mu_all[finite_mask]
    hb_finite = hb_all[finite_mask]
    hs_finite = hs_all[finite_mask]
    abs_qw_finite = abs_qw_all[finite_mask]

    band = float(args.handoff_band)
    outside = hb_finite < -band
    inside = hb_finite > band
    handoff = np.abs(hb_finite) <= band
    safe = hs_finite >= 0.0

    stats_all = _stats(mu_finite)
    stats_outside = _stats(mu_finite[outside])
    stats_inside = _stats(mu_finite[inside])
    stats_handoff = _stats(mu_finite[handoff])
    stats_safe = _stats(mu_finite[safe])

    print("\n=== mu_2(J_cl) sampled statistics ===")
    _print_stats("all UE flow nodes", stats_all)
    _print_stats(f"outside B (h_B < -{band:g})", stats_outside)
    _print_stats(f"inside B  (h_B >  {band:g})", stats_inside)
    _print_stats(f"handoff band |h_B| <= {band:g}", stats_handoff)
    _print_stats("safe nodes (h_S >= 0)", stats_safe)

    print("\n=== Domain diagnostics ===")
    print(f"finite nodes                   : {mu_finite.size}/{mu_all.size}")
    print(f"non-finite nodes               : {nonfinite_count}")
    print(f"safe-node fraction             : {100.0 * np.mean(safe):.4f}%")
    print(f"minimum h_S                    : {float(np.min(hs_finite)):.6f}")
    print(f"minimum |q_w|                  : {float(np.min(abs_qw_finite)):.6f}")
    print(f"hard-handoff node fraction     : {100.0 * np.mean(handoff):.4f}%")
    print(f"discrete h_B sign crossings    : {transition_count}")

    print("\n=== Empirical c result ===")
    print(f"observed max mu_2(J_cl)        : {stats_all['max']:.6f}")
    print("This is the sampled c candidate before any certification margin.")
    print("It is NOT yet a formal global upper bound on the whole state space.")

    if witness is not None:
        print("\n=== Maximum witness ===")
        print(f"mu_2                         : {witness['mu2']:.6f}")
        print(f"tau                          : {witness['tau']:.6f} s")
        print(f"region                       : {witness['region']}")
        print(f"d_hat                        : {np.asarray(witness['d_hat']).tolist()}")
        print(f"||d_hat||                    : {witness['d_hat_norm']:.6f}")
        print(f"h_B                          : {witness['h_B']:.6f}")
        print(f"h_S                          : {witness['h_S']:.6f}")
        print(f"|q_w|                        : {witness['abs_qw']:.6f}")
        print(f"state                         : {np.asarray(witness['state']).tolist()}")

    print("\n=== Important handoff caveat ===")
    print(
        "PS2-RL uses a hard select between pi_SA and pi_B at h_B=0. "
        "The UE-bCBF paper assumes a continuously differentiable backup controller."
    )
    print(
        "The Jacobian statistics above are therefore piecewise/branchwise. "
        "They do not certify a possible finite jump exactly on the handoff surface."
    )
    print(
        "Use this run to diagnose c first; we should decide how to handle the handoff "
        "before claiming the final UE-bCBF theorem verbatim."
    )

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
            "handoff_band": args.handoff_band,
            "dt": float(cfg.dt),
            "T": float(cfg.T),
            "num_steps": int(cfg.num_steps),
            "seed": args.seed,
        },
        "domain": {
            "initial_states": int(x0.shape[0]),
            "trajectories": int(x_pairs.shape[0]),
            "nodes": int(mu_all.size),
            "finite_nodes": int(mu_finite.size),
            "nonfinite_nodes": int(nonfinite_count),
            "safe_node_fraction": float(np.mean(safe)),
            "minimum_hS": float(np.min(hs_finite)),
            "minimum_abs_qw": float(np.min(abs_qw_finite)),
            "handoff_node_fraction": float(np.mean(handoff)),
            "handoff_sign_crossings": int(transition_count),
        },
        "mu2": {
            "all": stats_all,
            "outside_base": stats_outside,
            "inside_base": stats_inside,
            "handoff_band": stats_handoff,
            "safe_nodes": stats_safe,
            "observed_c_candidate": float(stats_all["max"]),
        },
        "max_witness": witness,
        "caveat": (
            "Empirical sampled bound only. The composed PS2-RL backup uses a hard h_B=0 "
            "handoff and is generally only piecewise differentiable, whereas UE-bCBF assumes "
            "a continuously differentiable backup controller."
        ),
    }

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(result), f, indent=2)
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
