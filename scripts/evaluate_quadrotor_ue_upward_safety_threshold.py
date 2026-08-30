#!/usr/bin/env python3
"""Upward-disturbance safety-threshold test for standard versus UE-bCBF.

This add-only evaluator freezes one deployed Phase-II actor and changes only
the safety filter.  Every scenario uses a fixed world-frame +z sinusoidal
acceleration, identical initial-state keys, and identical disturbance timing.
For each amplitude/frequency pair the UE bounds are set consistently to

    delta_d = A,  delta_v = 2*pi*f*A.

The evaluator also records the disturbance estimate, observer error bound,
and empirical bound coverage.  This remains an empirical first-order UE tube
test for the nonlinear quadrotor, not a formal nonlinear certificate.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime
import importlib.metadata
import json
from math import pi
from pathlib import Path
import shlex
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.cil.quadrotor_backup_cbf import QuadrotorBCBFConfig
from ps2rl.cil.quadrotor_ue_bcbf_experimental import ExperimentalUEConfig
from ps2rl.envs.quadrotor_env import QuadrotorEnvConfig, build_quadrotor_env
from ps2rl.evaluation import quadrotor_vanilla_eval as eval_utils
from ps2rl.phase2_ps2.quadrotor_ps2_trainer import _build_ue_observer_env
from ps2rl.utils.seed import make_prng_key

from evaluate_quadrotor_ue_weak_paired import (
    _bool_scalar,
    _build_condition_runtimes,
    _git_metadata,
    _load_json,
    _load_policy,
    _physical_env_state,
    _qp_step_diagnostics,
    _resolve_config_path,
    _rms,
    _safe_mean,
    _scalar,
    _sha256,
    _source_fingerprints,
    _write_csv,
    _write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_csv_floats(raw: str, *, name: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError(f"{name} requires at least one value")
    if not all(np.isfinite(value) for value in values):
        raise ValueError(f"All {name} values must be finite")
    return values


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _runtime_metadata() -> dict[str, Any]:
    return {
        "python": sys.version,
        "jax": getattr(jax, "__version__", "unknown"),
        "jaxlib": _package_version("jaxlib"),
        "qpax": _package_version("qpax"),
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
    }


def _optional_info_scalar(info: dict[str, Any], key: str) -> float:
    return _scalar(info[key]) if key in info else float("nan")


def _finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _finite_max(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.max(finite)) if finite.size else float("nan")


def _prepare_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_root).expanduser().resolve() / f"{stamp}_upward_z_threshold"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is non-empty: {output_dir}\nUse a new path or pass --overwrite.")
    for child in ("metadata", "scenarios", "results", "comparisons", "diagnostics", "plots"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)
    return output_dir


def _rollout_episode(
    condition: Any,
    env_fns: Any,
    *,
    episode_seed: int,
    amplitude: float,
    frequency_hz: float,
    phase_deg: float,
    base_obs_dim: int,
    observer_warmup_sec: float,
    observer_tolerance: float,
    env_dt: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episode_key = make_prng_key(int(episode_seed))
    state, obs = env_fns.reset(episode_key)
    physical_state = _physical_env_state(condition.filter_name, state)
    direction = np.asarray(physical_state.disturbance_direction, dtype=np.float64)
    if not np.allclose(direction, np.asarray([0.0, 0.0, 1.0]), atol=2e-6, rtol=0.0):
        raise RuntimeError(f"Expected fixed +z disturbance, got {direction.tolist()}")

    total_return = 0.0
    safe_steps: list[float] = []
    margins: list[float] = []
    pos_errors: list[float] = []
    interventions: list[float] = []
    fallback_flags: list[float] = []
    physical_slacks: list[float] = []
    solver_physical_slacks: list[float] = []
    solver_cbf_residuals: list[float] = []
    solver_inequality_residuals: list[float] = []
    solve_dtype_bits: list[float] = []
    solve_float64_flags: list[float] = []
    observer_errors: list[float] = []
    observer_bounds: list[float] = []
    observer_coverages: list[float] = []
    post_errors: list[float] = []
    post_bounds: list[float] = []
    post_coverages: list[float] = []
    step_rows: list[dict[str, Any]] = []

    done = False
    step_idx = 0
    while not done:
        obs_np = np.asarray(obs, dtype=np.float64)
        if condition.filter_name == "ue":
            d_hat = obs_np[base_obs_dim : base_obs_dim + 3]
            e_bar = float(obs_np[base_obs_dim + 3])
            elapsed = float(obs_np[base_obs_dim + 4])
        else:
            d_hat = np.full((3,), np.nan, dtype=np.float64)
            e_bar = float("nan")
            elapsed = float(step_idx) * float(env_dt)

        safe_action, raw_action, slack, use_solver, qp_info = condition.eval_action_fn(
            condition.policy.actor_params, jnp.asarray(obs, dtype=jnp.float32)
        )
        step_key = jax.random.fold_in(episode_key, step_idx)
        state, next_obs_true, _, reward, done_j, env_info = env_fns.step(state, safe_action, step_key)
        done = _bool_scalar(done_j)

        safe_np = np.asarray(safe_action, dtype=np.float64)
        raw_np = np.asarray(raw_action, dtype=np.float64)
        disturbance = np.asarray(env_info.disturbance_accel, dtype=np.float64)
        margin = _scalar(env_info.hard_deck_margin)
        is_safe = _scalar(env_info.is_safe)
        intervention = float(np.linalg.norm(safe_np - raw_np))
        qp_diag = _qp_step_diagnostics(qp_info, use_solver)
        physical_slack = _scalar(slack)
        cbf_row_residual = _optional_info_scalar(qp_info, "max_positive_row_residual")
        inequality_residual = _optional_info_scalar(
            qp_info, "max_positive_inequality_residual"
        )
        solve_bits = _optional_info_scalar(qp_info, "qp_solve_dtype_bits")
        solve_float64 = _optional_info_scalar(qp_info, "qp_solve_float64")

        total_return += _scalar(reward)
        safe_steps.append(is_safe)
        margins.append(margin)
        pos_errors.append(_scalar(env_info.pos_error_norm))
        interventions.append(intervention)
        fallback_flags.append(qp_diag["fallback"])
        physical_slacks.append(physical_slack)
        solve_dtype_bits.append(solve_bits)
        solve_float64_flags.append(solve_float64)
        if qp_diag["use_solver"] > 0.5:
            solver_physical_slacks.append(physical_slack)
            solver_cbf_residuals.append(cbf_row_residual)
            solver_inequality_residuals.append(inequality_residual)

        observer_error = float("nan")
        observer_covered = float("nan")
        observer_active = condition.filter_name == "ue" and elapsed >= float(observer_warmup_sec) - 1e-9
        if condition.filter_name == "ue":
            observer_error = float(np.linalg.norm(disturbance - d_hat))
            observer_covered = float(observer_error <= e_bar + float(observer_tolerance))
            observer_errors.append(observer_error)
            observer_bounds.append(e_bar)
            observer_coverages.append(observer_covered)
            if observer_active:
                post_errors.append(observer_error)
                post_bounds.append(e_bar)
                post_coverages.append(observer_covered)

        step_rows.append(
            {
                "condition": condition.label,
                "filter": condition.filter_name,
                "amplitude": float(amplitude),
                "frequency_hz": float(frequency_hz),
                "phase_deg": float(phase_deg),
                "episode_seed": int(episode_seed),
                "step": int(step_idx),
                "time_sec": _scalar(env_info.ref_time_sec),
                "disturbance_x": float(disturbance[0]),
                "disturbance_y": float(disturbance[1]),
                "disturbance_z": float(disturbance[2]),
                "d_hat_x": float(d_hat[0]),
                "d_hat_y": float(d_hat[1]),
                "d_hat_z": float(d_hat[2]),
                "observer_error_norm": observer_error,
                "observer_error_bound": e_bar,
                "observer_bound_covered": observer_covered,
                "ue_filter_active": float(observer_active),
                "hard_deck_margin": margin,
                "is_safe": is_safe,
                "intervention_norm": intervention,
                "raw_action_norm": float(np.linalg.norm(raw_np)),
                "safe_action_norm": float(np.linalg.norm(safe_np)),
                "slack": physical_slack,
                "use_solver": qp_diag["use_solver"],
                "qp_fallback": qp_diag["fallback"],
                "qp_fallback_invalid_inputs": qp_diag["fallback_invalid_inputs"],
                "qp_fallback_nonfinite_solution": qp_diag["fallback_nonfinite_solution"],
                "qp_max_positive_cbf_row_residual": cbf_row_residual,
                "qp_max_positive_inequality_residual": inequality_residual,
                "qp_solve_dtype_bits": solve_bits,
                "qp_solve_float64": solve_float64,
            }
        )

        obs = next_obs_true
        step_idx += 1

    violation_steps = int(sum(value < 0.5 for value in safe_steps))
    row = {
        "condition": condition.label,
        "filter": condition.filter_name,
        "amplitude": float(amplitude),
        "frequency_hz": float(frequency_hz),
        "phase_deg": float(phase_deg),
        "episode_seed": int(episode_seed),
        "direction": "+z",
        "delta_d": float(amplitude),
        "delta_v": float(2.0 * pi * frequency_hz * amplitude),
        "steps": int(step_idx),
        "return": float(total_return),
        "violation_free": 1.0 if violation_steps == 0 else 0.0,
        "violation_steps": violation_steps,
        "violation_duration_sec": float(violation_steps) * float(env_dt),
        "hard_deck_margin_min": float(np.min(margins)),
        "hard_deck_margin_p05": float(np.percentile(margins, 5.0)),
        "pos_error_norm_rmse": _rms(pos_errors),
        "intervention_norm_mean": _safe_mean(interventions),
        "intervention_norm_max": float(np.max(interventions)),
        "qp_fallback_steps": int(sum(fallback_flags)),
        "qp_fallback_rate": _safe_mean(fallback_flags),
        "qp_physical_slack_mean": _safe_mean(physical_slacks),
        "qp_physical_slack_max": float(np.max(physical_slacks)),
        "qp_solver_physical_slack_mean": _finite_mean(solver_physical_slacks),
        "qp_solver_cbf_row_residual_max": _finite_max(solver_cbf_residuals),
        "qp_solver_inequality_residual_max": _finite_max(solver_inequality_residuals),
        "qp_solve_dtype_bits": _finite_mean(solve_dtype_bits),
        "qp_solve_float64_rate": _finite_mean(solve_float64_flags),
        "observer_error_rmse": _rms(observer_errors) if observer_errors else float("nan"),
        "observer_error_max": float(np.max(observer_errors)) if observer_errors else float("nan"),
        "observer_bound_min": float(np.min(observer_bounds)) if observer_bounds else float("nan"),
        "observer_bound_max": float(np.max(observer_bounds)) if observer_bounds else float("nan"),
        "observer_coverage_rate": _safe_mean(observer_coverages) if observer_coverages else float("nan"),
        "observer_post_warmup_error_rmse": _rms(post_errors) if post_errors else float("nan"),
        "observer_post_warmup_error_max": float(np.max(post_errors)) if post_errors else float("nan"),
        "observer_post_warmup_bound_min": float(np.min(post_bounds)) if post_bounds else float("nan"),
        "observer_post_warmup_coverage_rate": _safe_mean(post_coverages) if post_coverages else float("nan"),
    }
    return row, step_rows


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (float(row["amplitude"]), float(row["frequency_hz"]), float(row["phase_deg"]), str(row["filter"]))
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        amplitude, frequency_hz, phase_deg, filter_name = key
        group = grouped[key]

        def values(name: str) -> np.ndarray:
            return np.asarray([float(row[name]) for row in group], dtype=np.float64)

        observer_coverage = values("observer_post_warmup_coverage_rate")
        observer_error = values("observer_post_warmup_error_rmse")
        output.append(
            {
                "filter": filter_name,
                "amplitude": amplitude,
                "frequency_hz": frequency_hz,
                "phase_deg": phase_deg,
                "episodes": len(group),
                "violation_free_episode_rate": float(np.mean(values("violation_free"))),
                "total_violation_steps": int(np.sum(values("violation_steps"))),
                "violation_duration_sec_mean": float(np.mean(values("violation_duration_sec"))),
                "hard_deck_margin_min": float(np.min(values("hard_deck_margin_min"))),
                "hard_deck_margin_p05_across_episodes": float(
                    np.percentile(values("hard_deck_margin_min"), 5.0)
                ),
                "return_mean": float(np.mean(values("return"))),
                "pos_error_norm_rmse_mean": float(np.mean(values("pos_error_norm_rmse"))),
                "intervention_norm_mean": float(np.mean(values("intervention_norm_mean"))),
                "qp_fallback_rate": float(np.mean(values("qp_fallback_rate"))),
                "observer_post_warmup_coverage_rate_mean": (
                    float(np.nanmean(observer_coverage)) if np.any(np.isfinite(observer_coverage)) else float("nan")
                ),
                "observer_post_warmup_error_rmse_mean": (
                    float(np.nanmean(observer_error)) if np.any(np.isfinite(observer_error)) else float("nan")
                ),
                "observer_post_warmup_error_max": (
                    float(np.nanmax(values("observer_post_warmup_error_max")))
                    if np.any(np.isfinite(values("observer_post_warmup_error_max")))
                    else float("nan")
                ),
            }
        )
    return output


def _paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (
            float(row["amplitude"]),
            float(row["frequency_hz"]),
            float(row["phase_deg"]),
            int(row["episode_seed"]),
            str(row["filter"]),
        ): row
        for row in rows
    }
    output: list[dict[str, Any]] = []
    base_keys = sorted({key[:4] for key in index})
    metrics = (
        "violation_free",
        "violation_steps",
        "hard_deck_margin_min",
        "return",
        "pos_error_norm_rmse",
        "intervention_norm_mean",
        "qp_fallback_rate",
    )
    for amplitude, frequency_hz, phase_deg, seed in base_keys:
        standard = index.get((amplitude, frequency_hz, phase_deg, seed, "standard"))
        ue = index.get((amplitude, frequency_hz, phase_deg, seed, "ue"))
        if standard is None or ue is None:
            continue
        row: dict[str, Any] = {
            "amplitude": amplitude,
            "frequency_hz": frequency_hz,
            "phase_deg": phase_deg,
            "episode_seed": seed,
        }
        for metric in metrics:
            row[f"ue_minus_standard__{metric}"] = float(ue[metric]) - float(standard[metric])
        output.append(row)
    return output


def _critical_amplitudes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            float(row["frequency_hz"]),
            float(row["phase_deg"]),
            int(row["episode_seed"]),
            str(row["filter"]),
        )
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        frequency_hz, phase_deg, seed, filter_name = key
        group = sorted(grouped[key], key=lambda row: float(row["amplitude"]))
        unsafe = [row for row in group if float(row["violation_free"]) < 0.5]
        first_unsafe = float(unsafe[0]["amplitude"]) if unsafe else float("nan")
        safe_amplitudes = [float(row["amplitude"]) for row in group if float(row["violation_free"]) >= 0.5]
        seen_unsafe = False
        nonmonotone = False
        for row in group:
            if float(row["violation_free"]) < 0.5:
                seen_unsafe = True
            elif seen_unsafe:
                nonmonotone = True
        output.append(
            {
                "filter": filter_name,
                "frequency_hz": frequency_hz,
                "phase_deg": phase_deg,
                "episode_seed": seed,
                "first_tested_violation_amplitude": first_unsafe,
                "largest_tested_safe_amplitude": max(safe_amplitudes) if safe_amplitudes else float("nan"),
                "censored_no_violation": 1.0 if not unsafe else 0.0,
                "nonmonotone_safety": 1.0 if nonmonotone else 0.0,
            }
        )
    return output


def _plot_threshold(rows: list[dict[str, Any]], output_path: Path, *, complete: bool) -> None:
    if not rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Warning: matplotlib unavailable; skipping threshold plot: {exc}")
        return

    frequencies = sorted({float(row["frequency_hz"]) for row in rows})
    fig, axes = plt.subplots(2, len(frequencies), figsize=(6.0 * len(frequencies), 8.0), squeeze=False)
    colors = {"standard": "#4C78A8", "ue": "#F58518"}
    for column, frequency_hz in enumerate(frequencies):
        freq_rows = [row for row in rows if float(row["frequency_hz"]) == frequency_hz]
        for filter_name in ("standard", "ue"):
            filter_rows = [row for row in freq_rows if row["filter"] == filter_name]
            amplitudes = sorted({float(row["amplitude"]) for row in filter_rows})
            safe_rates = []
            p05_margins = []
            for amplitude in amplitudes:
                group = [row for row in filter_rows if float(row["amplitude"]) == amplitude]
                safe_rates.append(float(np.mean([float(row["violation_free"]) for row in group])))
                p05_margins.append(
                    float(np.percentile([float(row["hard_deck_margin_min"]) for row in group], 5.0))
                )
            axes[0, column].plot(amplitudes, safe_rates, marker="o", color=colors[filter_name], label=filter_name)
            axes[1, column].plot(amplitudes, p05_margins, marker="o", color=colors[filter_name], label=filter_name)
        axes[0, column].set_title(f"f={frequency_hz:g} Hz")
        axes[0, column].set_ylim(-0.03, 1.03)
        axes[0, column].set_ylabel("Violation-free episode rate")
        axes[1, column].axhline(0.0, color="black", linewidth=1.0)
        axes[1, column].set_xlabel("Upward disturbance amplitude A [m/s^2]")
        axes[1, column].set_ylabel("5th percentile of episode min h [m]")
        for axis in axes[:, column]:
            axis.grid(True, alpha=0.25)
            axis.legend()
    status = "complete" if complete else "in progress"
    fig.suptitle(f"Upward +z safety threshold: standard vs UE-bCBF ({status})")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_observer(step_rows: list[dict[str, Any]], output_path: Path) -> None:
    ue_rows = [row for row in step_rows if row["filter"] == "ue"]
    if not ue_rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Warning: matplotlib unavailable; skipping observer plot: {exc}")
        return

    max_amplitude = max(float(row["amplitude"]) for row in ue_rows)
    candidates = [row for row in ue_rows if float(row["amplitude"]) == max_amplitude]
    frequency_hz = min(float(row["frequency_hz"]) for row in candidates)
    candidates = [row for row in candidates if float(row["frequency_hz"]) == frequency_hz]
    phase_deg = min(float(row["phase_deg"]) for row in candidates)
    candidates = [row for row in candidates if float(row["phase_deg"]) == phase_deg]
    seed = min(int(row["episode_seed"]) for row in candidates)
    example = [row for row in candidates if int(row["episode_seed"]) == seed]
    example.sort(key=lambda row: int(row["step"]))

    time = np.asarray([float(row["time_sec"]) for row in example])
    d_z = np.asarray([float(row["disturbance_z"]) for row in example])
    d_hat_z = np.asarray([float(row["d_hat_z"]) for row in example])
    error = np.asarray([float(row["observer_error_norm"]) for row in example])
    bound = np.asarray([float(row["observer_error_bound"]) for row in example])

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(time, d_z, label="true d_z", color="#4C78A8")
    axes[0].plot(time, d_hat_z, label="estimated d_hat_z", color="#F58518")
    axes[0].set_ylabel("Acceleration [m/s^2]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(time, error, label="||d-d_hat||", color="#E45756")
    axes[1].plot(time, bound, label="e_bar", color="#54A24B")
    axes[1].fill_between(time, 0.0, bound, color="#54A24B", alpha=0.12)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Observer error / bound")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)
    fig.suptitle(f"UE observer example: A={max_amplitude:g}, f={frequency_hz:g}, phase={phase_deg:g}, seed={seed}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _save_progress(output_dir: Path, episode_rows: list[dict[str, Any]], step_rows: list[dict[str, Any]], *, complete: bool) -> None:
    summary = _summarize(episode_rows)
    _write_csv(output_dir / "results" / "episodes.csv", episode_rows)
    _write_csv(output_dir / "results" / "summary.csv", summary)
    _write_json(output_dir / "results" / "summary.json", summary)
    _write_csv(output_dir / "comparisons" / "ue_minus_standard_paired.csv", _paired_rows(episode_rows))
    _write_csv(output_dir / "results" / "critical_amplitudes.csv", _critical_amplitudes(episode_rows))
    _write_csv(output_dir / "diagnostics" / "observer_steps.csv", step_rows)
    _plot_threshold(episode_rows, output_dir / "plots" / "safety_threshold_progress.png", complete=complete)
    if complete:
        _plot_threshold(episode_rows, output_dir / "plots" / "safety_threshold.png", complete=True)
        _plot_observer(step_rows, output_dir / "plots" / "observer_validation.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal-run-dir", required=True)
    parser.add_argument("--ue-config-run-dir", required=True, help="UE run used only for env/bCBF/observer configuration")
    parser.add_argument("--nominal-checkpoint", default="best")
    parser.add_argument("--amplitudes", default="0,0.25,0.5,0.75,1.0,1.25,1.5")
    parser.add_argument("--frequencies-hz", default="0.05")
    parser.add_argument("--phases-deg", default="90")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=3_600_000)
    parser.add_argument("--observer-tolerance", type=float, default=1e-5)
    parser.add_argument("--output-root", default="outputs/ue_bcbf_evaluation/02_upward_z_safety_threshold")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    amplitudes = _parse_csv_floats(args.amplitudes, name="--amplitudes")
    frequencies = _parse_csv_floats(args.frequencies_hz, name="--frequencies-hz")
    phases_deg = _parse_csv_floats(args.phases_deg, name="--phases-deg")
    if any(value < 0.0 for value in amplitudes):
        raise ValueError("All amplitudes must be nonnegative")
    if any(value < 0.0 for value in frequencies):
        raise ValueError("All frequencies must be nonnegative")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.observer_tolerance < 0.0 or not np.isfinite(args.observer_tolerance):
        raise ValueError("--observer-tolerance must be nonnegative and finite")

    output_dir = _prepare_output_dir(args)
    nominal_policy, _ = _load_policy("nominal", args.nominal_run_dir, args.nominal_checkpoint)
    ue_config_dir = Path(args.ue_config_run_dir).expanduser().resolve()
    ue_config_path = _resolve_config_path(ue_config_dir)
    ue_json = _load_json(ue_config_path)
    if "ue" not in ue_json:
        raise KeyError(f"UE config is missing from {ue_config_path}")

    env_cfg_saved = eval_utils._dataclass_from_dict(QuadrotorEnvConfig, ue_json.get("env", {}))
    cbf_cfg = eval_utils._dataclass_from_dict(QuadrotorBCBFConfig, ue_json.get("cbf", {}))
    ue_cfg_saved = eval_utils._dataclass_from_dict(ExperimentalUEConfig, ue_json.get("ue", {}))
    observer_warmup_sec = float(ue_json.get("ue_observer_warmup_sec", 0.2))
    seeds = tuple(int(args.seed_start) + index for index in range(int(args.repeats)))

    source_fingerprints = _source_fingerprints()
    this_script = Path(__file__).resolve()
    source_fingerprints["scripts/evaluate_quadrotor_ue_upward_safety_threshold.py"] = {
        "exists": True,
        "sha256": _sha256(this_script),
        "size_bytes": this_script.stat().st_size,
    }
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "kind": "upward_z_standard_vs_ue_safety_threshold",
        "command": shlex.join([sys.executable, *sys.argv]),
        "git": _git_metadata(),
        "runtime": _runtime_metadata(),
        "source_fingerprints": source_fingerprints,
        "output_dir": str(output_dir),
        "nominal_policy": {
            "run_dir": str(nominal_policy.run_dir),
            "checkpoint": str(nominal_policy.checkpoint),
            "checkpoint_sha256": _sha256(nominal_policy.checkpoint),
            "configured_training_steps": int(nominal_policy.sac_cfg.total_steps),
        },
        "ue_config_run_dir": str(ue_config_dir),
        "ue_config_path": str(ue_config_path),
        "ue_config_sha256": _sha256(ue_config_path),
        "base_environment": asdict(env_cfg_saved),
        "common_bcbf": asdict(cbf_cfg),
        "base_ue_config": asdict(ue_cfg_saved),
        "ue_observer_warmup_sec": observer_warmup_sec,
        "disturbance_direction": [0.0, 0.0, 1.0],
        "amplitudes": list(amplitudes),
        "frequencies_hz": list(frequencies),
        "phases_deg": list(phases_deg),
        "episode_seeds": list(seeds),
        "ue_bound_rule": {"delta_d": "A", "delta_v": "2*pi*f*A"},
        "note": "Empirical first-order UE tube; not a formal nonlinear robustness certificate.",
    }
    _write_json(output_dir / "metadata" / "manifest.json", manifest)
    (output_dir / "metadata" / "command.txt").write_text(manifest["command"] + "\n", encoding="utf-8")

    scenario_rows = [
        {
            "amplitude": amplitude,
            "frequency_hz": frequency_hz,
            "phase_deg": phase_deg,
            "episode_seed": seed,
            "direction": "+z",
            "delta_d": amplitude,
            "delta_v": 2.0 * pi * frequency_hz * amplitude,
        }
        for amplitude in amplitudes
        for frequency_hz in frequencies
        for phase_deg in phases_deg
        for seed in seeds
    ]
    _write_csv(output_dir / "scenarios" / "scenario_matrix.csv", scenario_rows)

    total_episodes = len(scenario_rows) * 2
    print("Upward +z safety-threshold evaluation")
    print(f"output: {output_dir}")
    print(f"frozen actor: {nominal_policy.checkpoint}")
    print(f"amplitudes: {list(amplitudes)}")
    print(f"frequencies: {list(frequencies)}")
    print(f"phases: {list(phases_deg)}")
    print(f"repeats: {len(seeds)}")
    print(f"total episodes: {total_episodes}")

    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    completed = 0
    for amplitude in amplitudes:
        for frequency_hz in frequencies:
            delta_v = 2.0 * pi * float(frequency_hz) * float(amplitude)
            ue_cfg = replace(ue_cfg_saved, delta_d=float(amplitude), delta_v=delta_v)
            base_env_cfg = replace(
                env_cfg_saved,
                disturbance_mode="sinusoidal" if amplitude > 0.0 else "none",
                disturbance_amplitude=float(amplitude),
                disturbance_frequency_hz=float(frequency_hz),
                disturbance_direction_mode="fixed",
                disturbance_direction_x=0.0,
                disturbance_direction_y=0.0,
                disturbance_direction_z=1.0,
                terminate_on_violation=False,
            )
            conditions = _build_condition_runtimes(
                (nominal_policy,), base_env_cfg, cbf_cfg, ue_cfg, observer_warmup_sec
            )

            for phase_deg in phases_deg:
                phase_env_cfg = replace(base_env_cfg, disturbance_phase=float(phase_deg) * pi / 180.0)
                standard_env = build_quadrotor_env(phase_env_cfg)
                ue_env, base_obs_dim = _build_ue_observer_env(phase_env_cfg, ue_cfg)
                for condition in conditions:
                    env_fns = ue_env if condition.filter_name == "ue" else standard_env
                    for seed in seeds:
                        row, rows_step = _rollout_episode(
                            condition,
                            env_fns,
                            episode_seed=seed,
                            amplitude=float(amplitude),
                            frequency_hz=float(frequency_hz),
                            phase_deg=float(phase_deg),
                            base_obs_dim=base_obs_dim,
                            observer_warmup_sec=observer_warmup_sec,
                            observer_tolerance=float(args.observer_tolerance),
                            env_dt=float(phase_env_cfg.dt),
                        )
                        episode_rows.append(row)
                        step_rows.extend(rows_step)
                        completed += 1
                        print(
                            f"[{completed:04d}/{total_episodes:04d}] {condition.label} "
                            f"A={amplitude:g} f={frequency_hz:g} phase={phase_deg:g} seed={seed} "
                            f"safe={row['violation_free']:.0f} margin={row['hard_deck_margin_min']:.4f} "
                            f"fallback={row['qp_fallback_rate']:.3f}"
                        )
                _save_progress(output_dir, episode_rows, step_rows, complete=False)

    _save_progress(output_dir, episode_rows, step_rows, complete=True)
    _write_json(
        output_dir / "results" / "run_complete.json",
        {
            "completed": True,
            "episodes": len(episode_rows),
            "scenario_count": len(scenario_rows),
            "filters": 2,
        },
    )

    summary = _summarize(episode_rows)
    print("\nSafety summary")
    for row in summary:
        print(
            f"{row['filter']} A={row['amplitude']:g} f={row['frequency_hz']:g} "
            f"phase={row['phase_deg']:g}: safe={row['violation_free_episode_rate']:.3f} "
            f"min_margin={row['hard_deck_margin_min']:.4f} "
            f"p05_margin={row['hard_deck_margin_p05_across_episodes']:.4f} "
            f"observer_coverage={row['observer_post_warmup_coverage_rate_mean']:.3f}"
        )
    print(f"\nSaved results to: {output_dir}")
    print(f"Safety plot: {output_dir / 'plots' / 'safety_threshold.png'}")
    print(f"Observer plot: {output_dir / 'plots' / 'observer_validation.png'}")


if __name__ == "__main__":
    main()
