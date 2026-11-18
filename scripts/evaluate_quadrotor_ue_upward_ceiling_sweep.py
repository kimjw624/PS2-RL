#!/usr/bin/env python3
"""Paired ceiling-stress test for standard bCBF versus UE-bCBF.

This add-only evaluator freezes the nominal Phase-II actor and applies the
same fixed +z sinusoidal disturbance to the standard and UE safety filters.
It sweeps the common environment/filter ceiling ``z_max`` to find cases where
the standard filter violates the ceiling while UE remains safe.  For every
scenario, both filters use the same actor, reset key, dynamics, reference,
disturbance waveform, and ceiling.

The UE bounds follow the tested sinusoid:

    delta_d = A,  delta_v = 2*pi*f*A.

This is empirical evidence for the nonlinear 10-D quadrotor, not a formal
nonlinear robustness certificate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
from math import pi
from pathlib import Path
import shlex
import sys
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp

from ps2rl.cil.quadrotor_backup_cbf import QuadrotorBCBFConfig
from ps2rl.cil.quadrotor_qp_scaled_experimental import (
    make_scaled_standard_projection_ops,
    make_scaled_ue_projection_ops,
)
from ps2rl.cil.quadrotor_ue_bcbf_experimental import ExperimentalUEConfig
from ps2rl.envs.quadrotor_env import QuadrotorEnvConfig, build_quadrotor_env
from ps2rl.evaluation import quadrotor_vanilla_eval as eval_utils
from ps2rl.phase2_ps2.ps2_trainer_core import build_ps2_action_fns
from ps2rl.phase2_ps2.quadrotor_ps2_trainer import (
    _action_bounds_from_cbf_cfg,
    _build_ue_observer_env,
    _disable_backup_fallback,
    _ue_network_obs_fn,
    _ue_projection_obs_fn,
)

from evaluate_quadrotor_ue_upward_safety_threshold import (
    _parse_csv_floats,
    _rollout_episode,
    _runtime_metadata,
    _summarize_ue_diagnostic_steps,
)
from evaluate_quadrotor_ue_weak_paired import (
    ConditionRuntime,
    _actor_cfg,
    _build_condition_runtimes,
    _git_metadata,
    _load_json,
    _load_policy,
    _resolve_config_path,
    _sha256,
    _source_fingerprints,
    _write_csv,
    _write_json,
)


_PHYS_DIM = 10


def _build_scaled_condition_runtimes(
    policies: tuple[Any, ...],
    env_cfg: QuadrotorEnvConfig,
    cbf_cfg: QuadrotorBCBFConfig,
    ue_cfg: ExperimentalUEConfig,
    observer_warmup_sec: float,
    *,
    solve_float64: bool = False,
) -> list[ConditionRuntime]:
    """Build standard and UE conditions with the same exact slack scaling."""

    base_env = build_quadrotor_env(env_cfg)
    base_obs_dim = int(base_env.obs_dim)
    action_dim = int(base_env.action_dim)
    action_scale = jnp.asarray(
        [cbf_cfg.a_cmd_max, cbf_cfg.omega_max, cbf_cfg.omega_max, cbf_cfg.omega_max],
        dtype=jnp.float32,
    )
    action_low, action_high = _action_bounds_from_cbf_cfg(cbf_cfg)
    standard_ops = make_scaled_standard_projection_ops(
        cbf_cfg, solve_float64=solve_float64
    )
    ue_ops = make_scaled_ue_projection_ops(
        cbf_cfg,
        ue_cfg,
        observer_warmup_sec=float(observer_warmup_sec),
        solve_float64=solve_float64,
    )

    conditions: list[ConditionRuntime] = []
    for policy in policies:
        eval_sac = replace(policy.sac_cfg, use_projection=True, project_actor_actions=True)
        actor_cfg = _actor_cfg(policy, base_obs_dim, action_dim)
        _, standard_eval = build_ps2_action_fns(
            eval_sac,
            actor_cfg,
            action_scale,
            action_low,
            action_high,
            standard_ops,
            phys_dim=_PHYS_DIM,
            disable_backup_fallback=_disable_backup_fallback(eval_sac),
            return_solver_info=True,
        )
        conditions.append(
            ConditionRuntime(
                label=f"{policy.label}__standard",
                policy=policy,
                filter_name="standard",
                eval_action_fn=standard_eval,
            )
        )
        _, ue_eval = build_ps2_action_fns(
            eval_sac,
            actor_cfg,
            action_scale,
            action_low,
            action_high,
            ue_ops,
            phys_dim=_PHYS_DIM,
            disable_backup_fallback=_disable_backup_fallback(eval_sac),
            return_solver_info=True,
            network_obs_fn=_ue_network_obs_fn(base_obs_dim),
            projection_obs_fn=_ue_projection_obs_fn(base_obs_dim),
        )
        conditions.append(
            ConditionRuntime(
                label=f"{policy.label}__ue",
                policy=policy,
                filter_name="ue",
                eval_action_fn=ue_eval,
            )
        )
    return conditions


def _prepare_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_root).expanduser().resolve() / f"{stamp}_ceiling_sweep"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is non-empty: {output_dir}\nUse a new path or pass --overwrite.")
    for child in ("metadata", "scenarios", "results", "comparisons", "diagnostics", "plots"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)
    return output_dir


def _reference_z_range(reference_path: str) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(reference_path), "available": False}
    try:
        with np.load(Path(reference_path).expanduser()) as data:
            states = np.asarray(data["states"], dtype=np.float64)
        result.update(
            {
                "available": True,
                "z_min": float(np.min(states[:, 2])),
                "z_max": float(np.max(states[:, 2])),
            }
        )
    except Exception as exc:
        result["read_error"] = str(exc)
    return result


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (float(row["z_max"]), float(row["amplitude"]), str(row["filter"]))
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for (z_max, amplitude, filter_name), group in sorted(grouped.items(), reverse=True):
        def values(name: str) -> np.ndarray:
            return np.asarray([float(row[name]) for row in group], dtype=np.float64)

        def finite_mean(name: str) -> float:
            array = values(name)
            finite = array[np.isfinite(array)]
            return float(np.mean(finite)) if finite.size else float("nan")

        def finite_max(name: str) -> float:
            array = values(name)
            finite = array[np.isfinite(array)]
            return float(np.max(finite)) if finite.size else float("nan")

        coverage = values("observer_post_warmup_coverage_rate")
        output.append(
            {
                "z_max": z_max,
                "amplitude": amplitude,
                "filter": filter_name,
                "episodes": len(group),
                "violation_free_episode_rate": float(np.mean(values("violation_free"))),
                "total_violation_steps": int(np.sum(values("violation_steps"))),
                "violation_duration_sec_mean": float(np.mean(values("violation_duration_sec"))),
                "hard_deck_margin_min": float(np.min(values("hard_deck_margin_min"))),
                "hard_deck_margin_p05_across_episodes": float(
                    np.percentile(values("hard_deck_margin_min"), 5.0)
                ),
                "pos_error_norm_rmse_mean": float(np.mean(values("pos_error_norm_rmse"))),
                "intervention_norm_mean": float(np.mean(values("intervention_norm_mean"))),
                "qp_fallback_rate": float(np.mean(values("qp_fallback_rate"))),
                "qp_physical_slack_mean": finite_mean("qp_physical_slack_mean"),
                "qp_physical_slack_max": finite_max("qp_physical_slack_max"),
                "qp_solver_physical_slack_mean": finite_mean(
                    "qp_solver_physical_slack_mean"
                ),
                "qp_solver_cbf_row_residual_max": finite_max(
                    "qp_solver_cbf_row_residual_max"
                ),
                "qp_solver_inequality_residual_max": finite_max(
                    "qp_solver_inequality_residual_max"
                ),
                "qp_solve_dtype_bits": finite_mean("qp_solve_dtype_bits"),
                "qp_solve_float64_rate": finite_mean("qp_solve_float64_rate"),
                "observer_post_warmup_coverage_rate_mean": (
                    float(np.nanmean(coverage)) if np.any(np.isfinite(coverage)) else float("nan")
                ),
            }
        )
    return output


def _paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (float(row["z_max"]), float(row["amplitude"]), int(row["episode_seed"]), str(row["filter"])): row
        for row in rows
    }
    metrics = (
        "violation_free",
        "violation_steps",
        "violation_duration_sec",
        "hard_deck_margin_min",
        "pos_error_norm_rmse",
        "intervention_norm_mean",
        "qp_fallback_rate",
        "qp_physical_slack_mean",
        "qp_solver_physical_slack_mean",
        "qp_solver_cbf_row_residual_max",
        "qp_solver_inequality_residual_max",
    )
    output: list[dict[str, Any]] = []
    for z_max, amplitude, seed in sorted({key[:3] for key in index}, reverse=True):
        standard = index.get((z_max, amplitude, seed, "standard"))
        ue = index.get((z_max, amplitude, seed, "ue"))
        if standard is None or ue is None:
            continue
        paired: dict[str, Any] = {"z_max": z_max, "amplitude": amplitude, "episode_seed": seed}
        for metric in metrics:
            paired[f"ue_minus_standard__{metric}"] = float(ue[metric]) - float(standard[metric])
        paired["standard_unsafe_ue_safe"] = float(
            float(standard["violation_free"]) < 0.5 and float(ue["violation_free"]) >= 0.5
        )
        paired["standard_safe_ue_unsafe"] = float(
            float(standard["violation_free"]) >= 0.5 and float(ue["violation_free"]) < 0.5
        )
        output.append(paired)
    return output


def _critical_ceilings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (float(row["amplitude"]), int(row["episode_seed"]), str(row["filter"]))
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for (amplitude, seed, filter_name), group in sorted(grouped.items()):
        # The test becomes harder as z_max decreases.
        ordered = sorted(group, key=lambda row: float(row["z_max"]), reverse=True)
        unsafe = [row for row in ordered if float(row["violation_free"]) < 0.5]
        safe_z = [float(row["z_max"]) for row in ordered if float(row["violation_free"]) >= 0.5]
        seen_unsafe = False
        nonmonotone = False
        for row in ordered:
            if float(row["violation_free"]) < 0.5:
                seen_unsafe = True
            elif seen_unsafe:
                nonmonotone = True
        output.append(
            {
                "amplitude": amplitude,
                "episode_seed": seed,
                "filter": filter_name,
                "first_tested_violation_z_max": float(unsafe[0]["z_max"]) if unsafe else float("nan"),
                "lowest_tested_safe_z_max": min(safe_z) if safe_z else float("nan"),
                "censored_no_violation": float(not unsafe),
                "nonmonotone_safety": float(nonmonotone),
            }
        )
    return output


def _plot_sweep(rows: list[dict[str, Any]], output_path: Path, *, complete: bool) -> None:
    if not rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Warning: matplotlib unavailable; skipping ceiling plot: {exc}")
        return

    amplitudes = sorted({float(row["amplitude"]) for row in rows})
    colors = {"standard": "#4C78A8", "ue": "#F58518"}
    fig, axes = plt.subplots(2, len(amplitudes), figsize=(5.0 * len(amplitudes), 7.5), squeeze=False)
    for column, amplitude in enumerate(amplitudes):
        amplitude_rows = [row for row in rows if float(row["amplitude"]) == amplitude]
        for filter_name in ("standard", "ue"):
            filter_rows = [row for row in amplitude_rows if row["filter"] == filter_name]
            z_values = sorted({float(row["z_max"]) for row in filter_rows})
            safe_rates: list[float] = []
            margin_p05: list[float] = []
            for z_max in z_values:
                group = [row for row in filter_rows if float(row["z_max"]) == z_max]
                safe_rates.append(float(np.mean([float(row["violation_free"]) for row in group])))
                margin_p05.append(float(np.percentile([float(row["hard_deck_margin_min"]) for row in group], 5.0)))
            axes[0, column].plot(z_values, safe_rates, marker="o", color=colors[filter_name], label=filter_name)
            axes[1, column].plot(z_values, margin_p05, marker="o", color=colors[filter_name], label=filter_name)
        axes[0, column].set_title(f"A={amplitude:g} m/s²")
        axes[0, column].set_ylim(-0.03, 1.03)
        axes[0, column].set_ylabel("Violation-free episode rate")
        axes[1, column].axhline(0.0, color="black", linewidth=1.0)
        axes[1, column].set_xlabel("Common ceiling z_max [m] (stricter →)")
        axes[1, column].set_ylabel("5th percentile of episode min h [m]")
        for axis in axes[:, column]:
            axis.invert_xaxis()
            axis.grid(True, alpha=0.25)
            axis.legend()
    status = "complete" if complete else "in progress"
    fig.suptitle(f"Upward +z ceiling stress: standard vs UE-bCBF ({status})")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _select_representative_pairs(
    episode_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Choose one deterministic paired seed for every (z_max, amplitude) cell.

    A standard-unsafe/UE-safe seed is preferred when that separation exists.
    Within the selected pool, the seed closest to the median standard minimum
    ceiling margin is used, avoiding an extreme best- or worst-case example.
    """

    index = {
        (
            float(row["z_max"]),
            float(row["amplitude"]),
            int(row["episode_seed"]),
            str(row["filter"]),
        ): row
        for row in episode_rows
    }
    cells = sorted({key[:2] for key in index})
    output: list[dict[str, Any]] = []
    for z_max, amplitude in cells:
        seeds = sorted(
            {
                key[2]
                for key in index
                if key[0] == z_max and key[1] == amplitude
            }
        )
        pairs: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
        for seed in seeds:
            standard = index.get((z_max, amplitude, seed, "standard"))
            ue = index.get((z_max, amplitude, seed, "ue"))
            if standard is not None and ue is not None:
                pairs.append((seed, standard, ue))
        if not pairs:
            continue

        separated = [
            pair
            for pair in pairs
            if float(pair[1]["violation_free"]) < 0.5
            and float(pair[2]["violation_free"]) >= 0.5
        ]
        pool = separated if separated else pairs
        target_margin = float(
            np.median([float(pair[1]["hard_deck_margin_min"]) for pair in pool])
        )
        seed, standard, ue = min(
            pool,
            key=lambda pair: (
                abs(float(pair[1]["hard_deck_margin_min"]) - target_margin),
                pair[0],
            ),
        )
        stem = (
            f"representative_zmax{_plot_token(z_max)}_"
            f"A{_plot_token(amplitude)}_seed{seed}"
        )
        output.append(
            {
                "z_max": z_max,
                "amplitude": amplitude,
                "episode_seed": seed,
                "selection_rule": (
                    "median_standard_margin_among_standard_unsafe_ue_safe"
                    if separated
                    else "median_standard_margin_among_all_paired_seeds"
                ),
                "candidate_pairs": len(pool),
                "standard_violation_free": float(standard["violation_free"]),
                "ue_violation_free": float(ue["violation_free"]),
                "standard_min_ceiling_margin": float(standard["hard_deck_margin_min"]),
                "ue_min_ceiling_margin": float(ue["hard_deck_margin_min"]),
                "time_position_plot": f"plots/{stem}_position_time.png",
                "trajectory_3d_plot": f"plots/{stem}_trajectory_3d.png",
            }
        )
    return output


def _plot_representative_pairs(
    step_rows: list[dict[str, Any]],
    representatives: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Warning: matplotlib unavailable; skipping representative plots: {exc}")
        return

    colors = {"standard": "#1f77b4", "ue": "#d97706"}
    labels = {"standard": "Standard bCBF", "ue": "UE-bCBF"}
    for representative in representatives:
        z_max = float(representative["z_max"])
        amplitude = float(representative["amplitude"])
        seed = int(representative["episode_seed"])
        selected = [
            row
            for row in step_rows
            if float(row["z_max"]) == z_max
            and float(row["amplitude"]) == amplitude
            and int(row["episode_seed"]) == seed
        ]
        traces = {
            filter_name: sorted(
                [row for row in selected if row["filter"] == filter_name],
                key=lambda row: int(row["step"]),
            )
            for filter_name in ("standard", "ue")
        }
        if not all(traces.values()):
            continue

        reference_rows = traces["standard"]
        reference = np.asarray(
            [
                [row["reference_x"], row["reference_y"], row["reference_z"]]
                for row in reference_rows
            ],
            dtype=np.float64,
        )
        reference_time = np.asarray(
            [row["time_sec"] for row in reference_rows], dtype=np.float64
        )

        fig, axes = plt.subplots(3, 1, figsize=(9.2, 8.0), sharex=True)
        coordinate_labels = ("x position [m]", "y position [m]", "z position [m]")
        for coordinate, axis in enumerate(axes):
            axis.plot(
                reference_time,
                reference[:, coordinate],
                color="black",
                linestyle="--",
                linewidth=1.4,
                label="Reference",
            )
            for filter_name in ("standard", "ue"):
                rows = traces[filter_name]
                time = np.asarray([row["time_sec"] for row in rows], dtype=np.float64)
                position = np.asarray(
                    [
                        [row["position_x"], row["position_y"], row["position_z"]]
                        for row in rows
                    ],
                    dtype=np.float64,
                )
                axis.plot(
                    time,
                    position[:, coordinate],
                    color=colors[filter_name],
                    linewidth=1.7,
                    label=labels[filter_name],
                )
            axis.set_ylabel(coordinate_labels[coordinate])
            axis.grid(True, alpha=0.25)
        axes[2].axhline(z_max, color="#b91c1c", linestyle=":", linewidth=1.6, label="Ceiling")
        axes[2].set_xlabel("Time [s]")
        handles, legend_labels = axes[2].get_legend_handles_labels()
        axes[0].legend(handles, legend_labels, loc="best", ncol=2)
        fig.suptitle(
            f"Representative paired trajectory: z_max={z_max:g} m, "
            f"A={amplitude:g} m/s², seed={seed}"
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        time_path = output_dir / str(representative["time_position_plot"])
        time_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(time_path, dpi=180)
        plt.close(fig)

        fig = plt.figure(figsize=(9.0, 7.2))
        axis = fig.add_subplot(111, projection="3d")
        axis.plot(
            reference[:, 0],
            reference[:, 1],
            reference[:, 2],
            color="black",
            linestyle="--",
            linewidth=1.4,
            label="Reference",
        )
        all_positions = [reference]
        for filter_name in ("standard", "ue"):
            rows = traces[filter_name]
            position = np.asarray(
                [
                    [row["position_x"], row["position_y"], row["position_z"]]
                    for row in rows
                ],
                dtype=np.float64,
            )
            all_positions.append(position)
            axis.plot(
                position[:, 0],
                position[:, 1],
                position[:, 2],
                color=colors[filter_name],
                linewidth=1.8,
                label=labels[filter_name],
            )
            axis.scatter(
                position[0, 0], position[0, 1], position[0, 2],
                color=colors[filter_name], marker="o", s=24,
            )

        combined = np.concatenate(all_positions, axis=0)
        x_min, x_max = float(np.min(combined[:, 0])), float(np.max(combined[:, 0]))
        y_min, y_max = float(np.min(combined[:, 1])), float(np.max(combined[:, 1]))
        x_pad = max(0.15 * (x_max - x_min), 0.25)
        y_pad = max(0.15 * (y_max - y_min), 0.25)
        xx, yy = np.meshgrid(
            [x_min - x_pad, x_max + x_pad],
            [y_min - y_pad, y_max + y_pad],
        )
        axis.plot_surface(
            xx,
            yy,
            np.full_like(xx, z_max),
            color="#ef4444",
            alpha=0.12,
            linewidth=0,
        )
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_zlabel("z [m]")
        axis.set_title(
            f"3-D representative trajectory | z_max={z_max:g} m, "
            f"A={amplitude:g} m/s², seed={seed}"
        )
        axis.view_init(elev=24, azim=-55)
        axis.legend(loc="best")
        axis.grid(True, alpha=0.25)
        spans = np.ptp(combined, axis=0)
        axis.set_box_aspect(tuple(np.maximum(spans, 0.5)))
        trajectory_path = output_dir / str(representative["trajectory_3d_plot"])
        fig.tight_layout()
        fig.savefig(trajectory_path, dpi=180)
        plt.close(fig)


def _save_progress(
    output_dir: Path,
    episode_rows: list[dict[str, Any]],
    observer_rows: list[dict[str, Any]],
    *,
    complete: bool,
) -> None:
    summary = _summarize(episode_rows)
    _write_csv(output_dir / "results" / "episodes.csv", episode_rows)
    _write_csv(output_dir / "results" / "summary.csv", summary)
    _write_json(output_dir / "results" / "summary.json", summary)
    _write_csv(output_dir / "comparisons" / "ue_minus_standard_paired.csv", _paired_rows(episode_rows))
    _write_csv(output_dir / "results" / "critical_ceilings.csv", _critical_ceilings(episode_rows))
    _write_csv(output_dir / "diagnostics" / "observer_steps.csv", observer_rows)
    _write_csv(
        output_dir / "diagnostics" / "ue_qp_terms_summary.csv",
        _summarize_ue_diagnostic_steps(observer_rows),
    )
    _plot_sweep(episode_rows, output_dir / "plots" / "ceiling_safety_progress.png", complete=complete)
    if complete:
        _plot_sweep(episode_rows, output_dir / "plots" / "ceiling_safety.png", complete=True)
        representatives = _select_representative_pairs(episode_rows)
        _write_csv(
            output_dir / "diagnostics" / "representative_cases.csv",
            representatives,
        )
        _plot_representative_pairs(observer_rows, representatives, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal-run-dir", required=True)
    parser.add_argument("--ue-config-run-dir", required=True, help="UE run used only for env/bCBF/observer config")
    parser.add_argument("--nominal-checkpoint", default="best")
    parser.add_argument("--z-max-values", default="3.0,2.9,2.8,2.7")
    parser.add_argument("--amplitudes", default="0,0.5,1.0,1.5")
    parser.add_argument("--frequency-hz", type=float, default=0.05)
    parser.add_argument("--phase-deg", type=float, default=90.0)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=3_800_000)
    parser.add_argument("--observer-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--scaled-slack-qp",
        action="store_true",
        help="Use the mathematically equivalent experimental scaled-slack QP for both filters",
    )
    parser.add_argument(
        "--scaled-slack-qp-float64",
        action="store_true",
        help=(
            "Solve the scaled five-variable QP in float64; requires "
            "--scaled-slack-qp and JAX_ENABLE_X64=1"
        ),
    )
    parser.add_argument(
        "--slack-weight-override",
        type=float,
        default=None,
        help=(
            "Override the saved bCBF slack weight for both standard and UE filters; "
            "the saved and effective configurations are recorded separately"
        ),
    )
    parser.add_argument("--output-root", default="outputs/ue_bcbf_evaluation/03_upward_z_ceiling_sweep")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    z_max_values = _parse_csv_floats(args.z_max_values, name="--z-max-values")
    amplitudes = _parse_csv_floats(args.amplitudes, name="--amplitudes")
    if any(value <= 0.0 for value in z_max_values):
        raise ValueError("All --z-max-values must be positive")
    if any(value < 0.0 for value in amplitudes):
        raise ValueError("All --amplitudes must be nonnegative")
    if args.frequency_hz < 0.0 or not np.isfinite(args.frequency_hz):
        raise ValueError("--frequency-hz must be nonnegative and finite")
    if not np.isfinite(args.phase_deg):
        raise ValueError("--phase-deg must be finite")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.observer_tolerance < 0.0 or not np.isfinite(args.observer_tolerance):
        raise ValueError("--observer-tolerance must be nonnegative and finite")
    if args.scaled_slack_qp_float64 and not args.scaled_slack_qp:
        raise ValueError("--scaled-slack-qp-float64 requires --scaled-slack-qp")
    if args.scaled_slack_qp_float64 and not bool(jax.config.x64_enabled):
        raise RuntimeError(
            "--scaled-slack-qp-float64 requires JAX_ENABLE_X64=1 before Python starts"
        )
    if args.slack_weight_override is not None and (
        not np.isfinite(args.slack_weight_override) or args.slack_weight_override <= 0.0
    ):
        raise ValueError("--slack-weight-override must be positive and finite")

    output_dir = _prepare_output_dir(args)
    nominal_policy, _ = _load_policy("nominal", args.nominal_run_dir, args.nominal_checkpoint)
    ue_config_dir = Path(args.ue_config_run_dir).expanduser().resolve()
    ue_config_path = _resolve_config_path(ue_config_dir)
    ue_json = _load_json(ue_config_path)
    if "ue" not in ue_json:
        raise KeyError(f"UE config is missing from {ue_config_path}")

    env_cfg_saved = eval_utils._dataclass_from_dict(QuadrotorEnvConfig, ue_json.get("env", {}))
    cbf_cfg_loaded = eval_utils._dataclass_from_dict(
        QuadrotorBCBFConfig, ue_json.get("cbf", {})
    )
    cbf_cfg_saved = (
        replace(cbf_cfg_loaded, slack_weight=float(args.slack_weight_override))
        if args.slack_weight_override is not None
        else cbf_cfg_loaded
    )
    ue_cfg_saved = eval_utils._dataclass_from_dict(ExperimentalUEConfig, ue_json.get("ue", {}))
    observer_warmup_sec = float(ue_json.get("ue_observer_warmup_sec", 0.2))
    if any(float(cbf_cfg_saved.z_des) >= value for value in z_max_values):
        raise ValueError(
            f"Every tested z_max must exceed the saved bCBF z_des={cbf_cfg_saved.z_des}; got {z_max_values}"
        )
    seeds = tuple(int(args.seed_start) + index for index in range(int(args.repeats)))

    source_fingerprints = _source_fingerprints()
    this_script = Path(__file__).resolve()
    source_fingerprints["scripts/evaluate_quadrotor_ue_upward_ceiling_sweep.py"] = {
        "exists": True,
        "sha256": _sha256(this_script),
        "size_bytes": this_script.stat().st_size,
    }
    threshold_script = this_script.with_name(
        "evaluate_quadrotor_ue_upward_safety_threshold.py"
    )
    source_fingerprints["scripts/evaluate_quadrotor_ue_upward_safety_threshold.py"] = {
        "exists": threshold_script.exists(),
        "sha256": _sha256(threshold_script) if threshold_script.exists() else "",
        "size_bytes": threshold_script.stat().st_size if threshold_script.exists() else 0,
    }
    scaled_module = Path(__file__).resolve().parents[1] / "ps2rl" / "cil" / "quadrotor_qp_scaled_experimental.py"
    if scaled_module.exists():
        source_fingerprints["ps2rl/cil/quadrotor_qp_scaled_experimental.py"] = {
            "exists": True,
            "sha256": _sha256(scaled_module),
            "size_bytes": scaled_module.stat().st_size,
        }
    reference = _reference_z_range(env_cfg_saved.reference_path)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "kind": "paired_upward_z_common_ceiling_sweep",
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
        "saved_environment": asdict(env_cfg_saved),
        "saved_bcbf": asdict(cbf_cfg_loaded),
        "effective_bcbf": asdict(cbf_cfg_saved),
        "slack_weight_override": (
            float(args.slack_weight_override)
            if args.slack_weight_override is not None
            else None
        ),
        "effective_slack_weight": float(cbf_cfg_saved.slack_weight),
        "base_ue_config": asdict(ue_cfg_saved),
        "reference_z_range": reference,
        "z_max_values": list(z_max_values),
        "amplitudes": list(amplitudes),
        "frequency_hz": float(args.frequency_hz),
        "phase_deg": float(args.phase_deg),
        "episode_seeds": list(seeds),
        "disturbance_direction": [0.0, 0.0, 1.0],
        "ue_bound_rule": {"delta_d": "A", "delta_v": "2*pi*f*A"},
        "qp_parameterization": (
            (
                "scaled_slack_exact_coordinate_change_float64_solve"
                if args.scaled_slack_qp_float64
                else "scaled_slack_exact_coordinate_change"
            )
            if args.scaled_slack_qp
            else "original"
        ),
        "qp_solve_dtype": "float64" if args.scaled_slack_qp_float64 else "float32",
        "slack_coordinate_scale": (
            float(np.sqrt(float(cbf_cfg_saved.slack_weight) / float(cbf_cfg_saved.control_weight)))
            if args.scaled_slack_qp
            else 1.0
        ),
        "controls": "Actor, seed, dynamics, reference, disturbance, and z_max are paired across filters.",
        "interpretation_guard": (
            "Use disturbance amplitudes only at ceilings where A=0 is acceptably safe; otherwise the result "
            "is confounded by nominal constraint/reference difficulty."
        ),
        "note": "Empirical first-order UE tube; not a formal nonlinear robustness certificate.",
    }
    _write_json(output_dir / "metadata" / "manifest.json", manifest)
    (output_dir / "metadata" / "command.txt").write_text(manifest["command"] + "\n", encoding="utf-8")

    scenario_rows = [
        {
            "z_max": z_max,
            "amplitude": amplitude,
            "frequency_hz": float(args.frequency_hz),
            "phase_deg": float(args.phase_deg),
            "episode_seed": seed,
            "direction": "+z",
            "delta_d": amplitude,
            "delta_v": 2.0 * pi * float(args.frequency_hz) * amplitude,
        }
        for z_max in z_max_values
        for amplitude in amplitudes
        for seed in seeds
    ]
    _write_csv(output_dir / "scenarios" / "scenario_matrix.csv", scenario_rows)

    total_episodes = len(scenario_rows) * 2
    print("Paired upward +z common-ceiling sweep")
    print(f"output: {output_dir}")
    print(f"frozen actor: {nominal_policy.checkpoint}")
    print(f"z_max values: {list(z_max_values)}")
    print(f"amplitudes: {list(amplitudes)}")
    print(f"frequency: {args.frequency_hz:g} Hz; phase: {args.phase_deg:g} deg")
    print(f"QP parameterization: {'scaled slack' if args.scaled_slack_qp else 'original'}")
    print(f"QP solve dtype: {'float64' if args.scaled_slack_qp_float64 else 'float32'}")
    slack_source = "override" if args.slack_weight_override is not None else "saved config"
    print(f"QP slack weight: {cbf_cfg_saved.slack_weight:g} ({slack_source})")
    if reference.get("available"):
        print(f"reference z range: [{reference['z_min']:.4f}, {reference['z_max']:.4f}] m")
    print(f"paired seeds per cell: {len(seeds)}")
    print(f"total episodes: {total_episodes}")

    episode_rows: list[dict[str, Any]] = []
    observer_rows: list[dict[str, Any]] = []
    completed = 0
    for z_max in z_max_values:
        cbf_cfg = replace(cbf_cfg_saved, z_max=float(z_max))
        for amplitude in amplitudes:
            delta_v = 2.0 * pi * float(args.frequency_hz) * float(amplitude)
            ue_cfg = replace(ue_cfg_saved, delta_d=float(amplitude), delta_v=delta_v)
            env_cfg = replace(
                env_cfg_saved,
                z_max=float(z_max),
                disturbance_mode="sinusoidal" if amplitude > 0.0 else "none",
                disturbance_amplitude=float(amplitude),
                disturbance_frequency_hz=float(args.frequency_hz),
                disturbance_phase=float(args.phase_deg) * pi / 180.0,
                disturbance_direction_mode="fixed",
                disturbance_direction_x=0.0,
                disturbance_direction_y=0.0,
                disturbance_direction_z=1.0,
                terminate_on_violation=False,
            )
            if args.scaled_slack_qp:
                conditions = _build_scaled_condition_runtimes(
                    (nominal_policy,),
                    env_cfg,
                    cbf_cfg,
                    ue_cfg,
                    observer_warmup_sec,
                    solve_float64=args.scaled_slack_qp_float64,
                )
            else:
                conditions = _build_condition_runtimes(
                    (nominal_policy,), env_cfg, cbf_cfg, ue_cfg, observer_warmup_sec
                )
            standard_env = build_quadrotor_env(env_cfg)
            ue_env, base_obs_dim = _build_ue_observer_env(env_cfg, ue_cfg)
            for condition in conditions:
                env_fns = ue_env if condition.filter_name == "ue" else standard_env
                for seed in seeds:
                    row, steps = _rollout_episode(
                        condition,
                        env_fns,
                        episode_seed=seed,
                        amplitude=float(amplitude),
                        frequency_hz=float(args.frequency_hz),
                        phase_deg=float(args.phase_deg),
                        base_obs_dim=base_obs_dim,
                        observer_warmup_sec=observer_warmup_sec,
                        observer_tolerance=float(args.observer_tolerance),
                        env_dt=float(env_cfg.dt),
                    )
                    row["z_max"] = float(z_max)
                    for step in steps:
                        step["z_max"] = float(z_max)
                    episode_rows.append(row)
                    observer_rows.extend(steps)
                    completed += 1
                    print(
                        f"[{completed:04d}/{total_episodes:04d}] {condition.label} zmax={z_max:g} "
                        f"A={amplitude:g} seed={seed} safe={row['violation_free']:.0f} "
                        f"margin={row['hard_deck_margin_min']:.4f} fallback={row['qp_fallback_rate']:.3f}"
                    )
            _save_progress(output_dir, episode_rows, observer_rows, complete=False)

    _save_progress(output_dir, episode_rows, observer_rows, complete=True)
    _write_json(
        output_dir / "results" / "run_complete.json",
        {"completed": True, "episodes": len(episode_rows), "scenario_count": len(scenario_rows), "filters": 2},
    )

    print("\nSafety summary")
    for row in _summarize(episode_rows):
        print(
            f"zmax={row['z_max']:g} A={row['amplitude']:g} {row['filter']}: "
            f"safe={row['violation_free_episode_rate']:.3f} "
            f"min_margin={row['hard_deck_margin_min']:.4f} "
            f"p05_margin={row['hard_deck_margin_p05_across_episodes']:.4f} "
            f"tracking={row['pos_error_norm_rmse_mean']:.4f}"
        )
    print("\nUE-QP term diagnostics (median / p95 / max over controller steps)")
    for row in _summarize_ue_diagnostic_steps(observer_rows):
        if row["filter"] != "ue":
            continue
        print(
            f"zmax={row['z_max']:g} A={row['amplitude']:g}: "
            f"epsSmax={row['max_safe_epsilon_median']:.4g}/"
            f"{row['max_safe_epsilon_p95']:.4g}/{row['max_safe_epsilon_max']:.4g} "
            f"epsSN={row['terminal_safe_epsilon_median']:.4g}/"
            f"{row['terminal_safe_epsilon_p95']:.4g}/{row['terminal_safe_epsilon_max']:.4g}"
        )
        print(
            f"  epsB={row['terminal_base_epsilon_median']:.4g}/"
            f"{row['terminal_base_epsilon_p95']:.4g}/{row['terminal_base_epsilon_max']:.4g} "
            f"rhoSmax={row['max_safe_rho_median']:.4g}/"
            f"{row['max_safe_rho_p95']:.4g}/{row['max_safe_rho_max']:.4g} "
            f"rhoB={row['terminal_rho_median']:.4g}/"
            f"{row['terminal_rho_p95']:.4g}/{row['terminal_rho_max']:.4g}"
        )
    print(f"\nSaved results to: {output_dir}")
    print(f"Main plot: {output_dir / 'plots' / 'ceiling_safety.png'}")
    print(f"Paired rows: {output_dir / 'comparisons' / 'ue_minus_standard_paired.csv'}")
    print(f"UE-QP terms: {output_dir / 'diagnostics' / 'ue_qp_terms_summary.csv'}")
    print(f"Representative cases: {output_dir / 'diagnostics' / 'representative_cases.csv'}")
    print(f"Representative plots: {output_dir / 'plots' / 'representative_*'}")


if __name__ == "__main__":
    main()
