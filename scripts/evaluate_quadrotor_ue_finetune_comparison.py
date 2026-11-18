#!/usr/bin/env python3
"""UE-only comparison of nominal and UE-bCBF-fine-tuned quadrotor actors.

All actors are evaluated with the same UE-bCBF, environment, disturbance,
ceiling, reset seeds, and QP parameterization.  The only intended difference
is the actor checkpoint.  This isolates whether UE-bCBF Phase-II fine-tuning
improves tracking while retaining safety.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
from math import pi
from pathlib import Path
import shlex
import sys
from typing import Any

import jax
import numpy as np

from ps2rl.cil.quadrotor_backup_cbf import QuadrotorBCBFConfig
from ps2rl.cil.quadrotor_ue_bcbf_experimental import ExperimentalUEConfig
from ps2rl.envs.quadrotor_env import QuadrotorEnvConfig
from ps2rl.evaluation import quadrotor_vanilla_eval as eval_utils
from ps2rl.phase2_ps2.quadrotor_ps2_trainer import _build_ue_observer_env

from evaluate_quadrotor_ue_upward_ceiling_sweep import (
    _build_scaled_condition_runtimes,
)
from evaluate_quadrotor_ue_upward_safety_threshold import (
    _rollout_episode,
    _runtime_metadata,
    _summarize_ue_diagnostic_steps,
)
from evaluate_quadrotor_ue_weak_paired import (
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


POLICY_ORDER = ("baseline", "nominal_train", "disturbed_train")
POLICY_LABELS = {
    "baseline": "Original nominal actor",
    "nominal_train": "UE fine-tune: A_train=0",
    "disturbed_train": "UE fine-tune: A_train=2",
}
POLICY_COLORS = {
    "baseline": "#4C78A8",
    "nominal_train": "#F58518",
    "disturbed_train": "#54A24B",
}


def _prepare_output_dir(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is non-empty: {output_dir}\n"
            "Use a new path or pass --overwrite."
        )
    for child in ("metadata", "results", "comparisons", "diagnostics", "plots"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)
    return output_dir


def _finite_mean(group: list[dict[str, Any]], name: str) -> float:
    values = np.asarray([float(row[name]) for row in group], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def _finite_max(group: list[dict[str, Any]], name: str) -> float:
    values = np.asarray([float(row[name]) for row in group], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.max(values)) if values.size else float("nan")


def _summarize(episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for policy in POLICY_ORDER:
        group = [row for row in episode_rows if row["policy"] == policy]
        if not group:
            continue
        margins = np.asarray(
            [float(row["hard_deck_margin_min"]) for row in group], dtype=np.float64
        )
        output.append(
            {
                "policy": policy,
                "label": POLICY_LABELS[policy],
                "episodes": len(group),
                "violation_free_episode_rate": float(
                    np.mean([float(row["violation_free"]) for row in group])
                ),
                "total_violation_steps": int(
                    sum(int(row["violation_steps"]) for row in group)
                ),
                "hard_deck_margin_min": float(np.min(margins)),
                "hard_deck_margin_p05_across_episodes": float(
                    np.percentile(margins, 5.0)
                ),
                "pos_error_norm_rmse_mean": _finite_mean(
                    group, "pos_error_norm_rmse"
                ),
                "intervention_norm_mean": _finite_mean(
                    group, "intervention_norm_mean"
                ),
                "qp_physical_slack_mean": _finite_mean(
                    group, "qp_physical_slack_mean"
                ),
                "qp_physical_slack_max": _finite_max(
                    group, "qp_physical_slack_max"
                ),
                "qp_fallback_rate": _finite_mean(group, "qp_fallback_rate"),
                "observer_post_warmup_coverage_rate": _finite_mean(
                    group, "observer_post_warmup_coverage_rate"
                ),
            }
        )
    return output


def _paired_deltas(episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (int(row["episode_seed"]), str(row["policy"])): row
        for row in episode_rows
    }
    metrics = (
        "violation_free",
        "violation_steps",
        "hard_deck_margin_min",
        "pos_error_norm_rmse",
        "intervention_norm_mean",
        "qp_physical_slack_mean",
        "qp_fallback_rate",
    )
    output: list[dict[str, Any]] = []
    for seed in sorted({key[0] for key in index}):
        baseline = index.get((seed, "baseline"))
        if baseline is None:
            continue
        for policy in POLICY_ORDER[1:]:
            tuned = index.get((seed, policy))
            if tuned is None:
                continue
            row: dict[str, Any] = {
                "episode_seed": seed,
                "comparison": f"{policy}_minus_baseline",
            }
            for metric in metrics:
                row[f"delta__{metric}"] = float(tuned[metric]) - float(
                    baseline[metric]
                )
            output.append(row)
    return output


def _summarize_ue_terms(step_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for policy in POLICY_ORDER:
        group = [row for row in step_rows if row["policy"] == policy]
        for row in _summarize_ue_diagnostic_steps(group):
            output.append({"policy": policy, "label": POLICY_LABELS[policy], **row})
    return output


def _select_representative(episode_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for row in episode_rows:
        by_seed.setdefault(int(row["episode_seed"]), {})[str(row["policy"])] = row
    complete = [
        (seed, rows)
        for seed, rows in sorted(by_seed.items())
        if all(policy in rows for policy in POLICY_ORDER)
    ]
    if not complete:
        raise RuntimeError("No complete paired seed is available for plotting")
    all_safe = [
        pair
        for pair in complete
        if all(float(pair[1][policy]["violation_free"]) >= 0.5 for policy in POLICY_ORDER)
    ]
    pool = all_safe if all_safe else complete
    target = float(
        np.median(
            [float(rows["baseline"]["pos_error_norm_rmse"]) for _, rows in pool]
        )
    )
    seed, rows = min(
        pool,
        key=lambda pair: (
            abs(float(pair[1]["baseline"]["pos_error_norm_rmse"]) - target),
            pair[0],
        ),
    )
    result: dict[str, Any] = {
        "episode_seed": seed,
        "selection_rule": (
            "median_baseline_position_rmse_among_all_safe_paired_seeds"
            if all_safe
            else "median_baseline_position_rmse_among_all_complete_paired_seeds"
        ),
        "candidate_pairs": len(pool),
    }
    for policy in POLICY_ORDER:
        result[f"{policy}__violation_free"] = float(rows[policy]["violation_free"])
        result[f"{policy}__hard_deck_margin_min"] = float(
            rows[policy]["hard_deck_margin_min"]
        )
        result[f"{policy}__pos_error_norm_rmse"] = float(
            rows[policy]["pos_error_norm_rmse"]
        )
    return result


def _plot_summary(
    summary: list[dict[str, Any]], output_path: Path, *, amplitude: float
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Warning: matplotlib unavailable; skipping summary plot: {exc}")
        return

    labels = [row["label"] for row in summary]
    colors = [POLICY_COLORS[row["policy"]] for row in summary]
    metrics = (
        ("violation_free_episode_rate", "Violation-free episode rate"),
        ("hard_deck_margin_p05_across_episodes", "5th percentile min ceiling margin [m]"),
        ("pos_error_norm_rmse_mean", "Position-error norm RMSE [m]"),
        ("intervention_norm_mean", "Mean UE-filter intervention norm"),
        ("qp_physical_slack_mean", "Mean physical QP slack"),
        ("qp_fallback_rate", "QP fallback rate"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0))
    for axis, (key, title) in zip(axes.ravel(), metrics):
        values = [float(row[key]) for row in summary]
        positions = np.arange(len(values))
        axis.bar(positions, values, color=colors)
        axis.set_xticks(positions, labels, rotation=18, ha="right")
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.25)
        if key == "hard_deck_margin_p05_across_episodes":
            axis.axhline(0.0, color="black", linewidth=1.0)
        if key in ("violation_free_episode_rate", "qp_fallback_rate"):
            axis.set_ylim(0.0, 1.05)
    fig.suptitle(
        f"UE-bCBF actor comparison under constant upward A={amplitude:g} m/s²"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_representative(
    step_rows: list[dict[str, Any]],
    representative: dict[str, Any],
    *,
    z_max: float,
    amplitude: float,
    output_dir: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Warning: matplotlib unavailable; skipping trajectory plots: {exc}")
        return

    seed = int(representative["episode_seed"])
    traces = {
        policy: sorted(
            [
                row
                for row in step_rows
                if row["policy"] == policy and int(row["episode_seed"]) == seed
            ],
            key=lambda row: int(row["step"]),
        )
        for policy in POLICY_ORDER
    }
    if not all(traces.values()):
        raise RuntimeError(f"Missing representative trajectory rows for seed {seed}")

    base = traces["baseline"]
    reference = np.asarray(
        [[row["reference_x"], row["reference_y"], row["reference_z"]] for row in base],
        dtype=np.float64,
    )
    reference_time = np.asarray([row["time_sec"] for row in base], dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.2), sharex=True)
    axis_labels = ("x position [m]", "y position [m]", "z position [m]")
    for coordinate, axis in enumerate(axes):
        axis.plot(
            reference_time,
            reference[:, coordinate],
            color="black",
            linestyle="--",
            linewidth=1.4,
            label="Reference",
        )
        for policy in POLICY_ORDER:
            rows = traces[policy]
            time = np.asarray([row["time_sec"] for row in rows], dtype=np.float64)
            position = np.asarray(
                [[row["position_x"], row["position_y"], row["position_z"]] for row in rows],
                dtype=np.float64,
            )
            axis.plot(
                time,
                position[:, coordinate],
                color=POLICY_COLORS[policy],
                linewidth=1.7,
                label=POLICY_LABELS[policy],
            )
        axis.set_ylabel(axis_labels[coordinate])
        axis.grid(True, alpha=0.25)
    axes[2].axhline(
        z_max, color="#B91C1C", linestyle=":", linewidth=1.6, label="Ceiling"
    )
    axes[2].set_xlabel("Time [s]")
    handles, labels = axes[2].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="best", ncol=2, fontsize=8)
    fig.suptitle(
        f"Representative UE-only comparison: A={amplitude:g} m/s², seed={seed}"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_dir / "plots" / "representative_position_time.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(9.2, 7.4))
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
    for policy in POLICY_ORDER:
        rows = traces[policy]
        position = np.asarray(
            [[row["position_x"], row["position_y"], row["position_z"]] for row in rows],
            dtype=np.float64,
        )
        all_positions.append(position)
        axis.plot(
            position[:, 0],
            position[:, 1],
            position[:, 2],
            color=POLICY_COLORS[policy],
            linewidth=1.8,
            label=POLICY_LABELS[policy],
        )
        axis.scatter(
            position[0, 0], position[0, 1], position[0, 2],
            color=POLICY_COLORS[policy], marker="o", s=22,
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
        xx, yy, np.full_like(xx, z_max), color="#EF4444", alpha=0.12, linewidth=0
    )
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.set_title(
        f"3-D UE-only comparison | z_max={z_max:g} m, A={amplitude:g} m/s², seed={seed}"
    )
    axis.view_init(elev=24, azim=-55)
    axis.legend(loc="best", fontsize=8)
    spans = np.ptp(combined, axis=0)
    axis.set_box_aspect(tuple(np.maximum(spans, 0.5)))
    fig.tight_layout()
    fig.savefig(output_dir / "plots" / "representative_trajectory_3d.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run-dir", required=True)
    parser.add_argument("--nominal-finetune-run-dir", required=True)
    parser.add_argument("--disturbed-finetune-run-dir", required=True)
    parser.add_argument("--baseline-checkpoint", default="best")
    parser.add_argument("--finetune-checkpoint", default="final")
    parser.add_argument(
        "--ue-config-run-dir",
        default="",
        help="UE/env/bCBF config source; defaults to nominal-finetune-run-dir.",
    )
    parser.add_argument("--z-max", type=float, default=2.7)
    parser.add_argument("--disturbance-amplitude", type=float, default=2.0)
    parser.add_argument("--disturbance-frequency-hz", type=float, default=0.0)
    parser.add_argument("--disturbance-phase-deg", type=float, default=90.0)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=4_500_000)
    parser.add_argument("--observer-tolerance", type=float, default=1e-5)
    parser.add_argument("--slack-weight-override", type=float, default=1e6)
    parser.add_argument("--scaled-slack-qp", action="store_true")
    parser.add_argument("--scaled-slack-qp-float64", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.z_max <= 0.0 or not np.isfinite(args.z_max):
        raise ValueError("--z-max must be positive and finite")
    if args.disturbance_amplitude < 0.0 or not np.isfinite(args.disturbance_amplitude):
        raise ValueError("--disturbance-amplitude must be nonnegative and finite")
    if args.disturbance_frequency_hz < 0.0 or not np.isfinite(args.disturbance_frequency_hz):
        raise ValueError("--disturbance-frequency-hz must be nonnegative and finite")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.slack_weight_override <= 0.0 or not np.isfinite(args.slack_weight_override):
        raise ValueError("--slack-weight-override must be positive and finite")
    if args.scaled_slack_qp_float64 and not args.scaled_slack_qp:
        raise ValueError("--scaled-slack-qp-float64 requires --scaled-slack-qp")
    if args.scaled_slack_qp_float64 and not bool(jax.config.x64_enabled):
        raise RuntimeError(
            "--scaled-slack-qp-float64 requires JAX_ENABLE_X64=1 before Python starts"
        )

    output_dir = _prepare_output_dir(args)
    baseline, baseline_json = _load_policy(
        "baseline", args.baseline_run_dir, args.baseline_checkpoint
    )
    nominal_train, nominal_json = _load_policy(
        "nominal_train", args.nominal_finetune_run_dir, args.finetune_checkpoint
    )
    disturbed_train, disturbed_json = _load_policy(
        "disturbed_train", args.disturbed_finetune_run_dir, args.finetune_checkpoint
    )
    policies = (baseline, nominal_train, disturbed_train)

    ue_config_dir = Path(
        args.ue_config_run_dir or args.nominal_finetune_run_dir
    ).expanduser().resolve()
    ue_config_path = _resolve_config_path(ue_config_dir)
    ue_json = _load_json(ue_config_path)
    if "ue" not in ue_json:
        raise KeyError(f"UE config is missing from {ue_config_path}")

    env_saved = eval_utils._dataclass_from_dict(
        QuadrotorEnvConfig, ue_json.get("env", {})
    )
    cbf_saved = eval_utils._dataclass_from_dict(
        QuadrotorBCBFConfig, ue_json.get("cbf", {})
    )
    ue_saved = eval_utils._dataclass_from_dict(
        ExperimentalUEConfig, ue_json.get("ue", {})
    )
    observer_warmup_sec = float(ue_json.get("ue_observer_warmup_sec", 0.2))
    amplitude = float(args.disturbance_amplitude)
    frequency_hz = float(args.disturbance_frequency_hz)
    phase_deg = float(args.disturbance_phase_deg)
    delta_v = 2.0 * pi * frequency_hz * amplitude
    env_cfg = replace(
        env_saved,
        z_max=float(args.z_max),
        disturbance_mode="sinusoidal" if amplitude > 0.0 else "none",
        disturbance_amplitude=amplitude,
        disturbance_frequency_hz=frequency_hz,
        disturbance_phase=phase_deg * pi / 180.0,
        disturbance_direction_mode="fixed",
        disturbance_direction_x=0.0,
        disturbance_direction_y=0.0,
        disturbance_direction_z=1.0,
        terminate_on_violation=False,
    )
    cbf_cfg = replace(
        cbf_saved,
        z_max=float(args.z_max),
        slack_weight=float(args.slack_weight_override),
    )
    ue_cfg = replace(ue_saved, delta_d=amplitude, delta_v=delta_v)

    if args.scaled_slack_qp:
        conditions = _build_scaled_condition_runtimes(
            policies,
            env_cfg,
            cbf_cfg,
            ue_cfg,
            observer_warmup_sec,
            solve_float64=bool(args.scaled_slack_qp_float64),
        )
    else:
        conditions = _build_condition_runtimes(
            policies, env_cfg, cbf_cfg, ue_cfg, observer_warmup_sec
        )
    conditions = [condition for condition in conditions if condition.filter_name == "ue"]
    if len(conditions) != len(POLICY_ORDER):
        raise RuntimeError(f"Expected {len(POLICY_ORDER)} UE conditions, got {len(conditions)}")

    env_fns, base_obs_dim = _build_ue_observer_env(env_cfg, ue_cfg)
    seeds = tuple(int(args.seed_start) + index for index in range(int(args.repeats)))
    source_fingerprints = _source_fingerprints()
    this_script = Path(__file__).resolve()
    source_fingerprints["scripts/evaluate_quadrotor_ue_finetune_comparison.py"] = {
        "exists": True,
        "sha256": _sha256(this_script),
        "size_bytes": this_script.stat().st_size,
    }
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "kind": "ue_only_actor_finetune_comparison",
        "command": shlex.join([sys.executable, *sys.argv]),
        "git": _git_metadata(),
        "runtime": _runtime_metadata(),
        "source_fingerprints": source_fingerprints,
        "controls": (
            "All actors use the same UE-bCBF, constant +z disturbance, ceiling, "
            "reset seeds, reference, dynamics, and QP settings."
        ),
        "policies": {
            policy.label: {
                "run_dir": str(policy.run_dir),
                "checkpoint": str(policy.checkpoint),
                "checkpoint_sha256": _sha256(policy.checkpoint),
            }
            for policy in policies
        },
        "config_sha256": {
            "baseline": _sha256(_resolve_config_path(baseline.run_dir)),
            "nominal_train": _sha256(_resolve_config_path(nominal_train.run_dir)),
            "disturbed_train": _sha256(_resolve_config_path(disturbed_train.run_dir)),
            "ue_source": _sha256(ue_config_path),
        },
        "saved_configs": {
            "baseline_sac": baseline_json.get("sac", {}),
            "nominal_train_ue": nominal_json.get("ue", {}),
            "disturbed_train_ue": disturbed_json.get("ue", {}),
        },
        "effective_env": asdict(env_cfg),
        "effective_bcbf": asdict(cbf_cfg),
        "effective_ue": asdict(ue_cfg),
        "observer_warmup_sec": observer_warmup_sec,
        "episode_seeds": list(seeds),
        "qp_parameterization": (
            "scaled_slack_float64"
            if args.scaled_slack_qp_float64
            else "scaled_slack_float32"
            if args.scaled_slack_qp
            else "original"
        ),
        "interpretation_guard": (
            "The reference reaches above z_max, so this measures constrained tracking; "
            "perfect reference tracking is incompatible with ceiling safety."
        ),
    }
    _write_json(output_dir / "metadata" / "manifest.json", manifest)
    (output_dir / "metadata" / "command.txt").write_text(
        manifest["command"] + "\n", encoding="utf-8"
    )

    print("UE-only actor fine-tuning comparison")
    print(f"output: {output_dir}")
    print(f"conditions: {', '.join(condition.label for condition in conditions)}")
    print(
        f"z_max={args.z_max:g}, constant +z A={amplitude:g} m/s², "
        f"paired seeds={len(seeds)}"
    )
    print(f"total episodes: {len(conditions) * len(seeds)}")

    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    completed = 0
    total = len(conditions) * len(seeds)
    for condition in conditions:
        for seed in seeds:
            row, steps = _rollout_episode(
                condition,
                env_fns,
                episode_seed=seed,
                amplitude=amplitude,
                frequency_hz=frequency_hz,
                phase_deg=phase_deg,
                base_obs_dim=base_obs_dim,
                observer_warmup_sec=observer_warmup_sec,
                observer_tolerance=float(args.observer_tolerance),
                env_dt=float(env_cfg.dt),
            )
            row["policy"] = condition.policy.label
            row["z_max"] = float(args.z_max)
            for step in steps:
                step["policy"] = condition.policy.label
                step["z_max"] = float(args.z_max)
            episode_rows.append(row)
            step_rows.extend(steps)
            completed += 1
            print(
                f"[{completed:03d}/{total:03d}] {condition.policy.label} seed={seed} "
                f"safe={row['violation_free']:.0f} "
                f"margin={row['hard_deck_margin_min']:.4f} "
                f"pos_rmse={row['pos_error_norm_rmse']:.4f}"
            )

    summary = _summarize(episode_rows)
    paired = _paired_deltas(episode_rows)
    ue_terms = _summarize_ue_terms(step_rows)
    representative = _select_representative(episode_rows)
    _write_csv(output_dir / "results" / "episodes.csv", episode_rows)
    _write_csv(output_dir / "results" / "summary.csv", summary)
    _write_json(output_dir / "results" / "summary.json", summary)
    _write_csv(output_dir / "comparisons" / "paired_actor_deltas.csv", paired)
    _write_csv(output_dir / "diagnostics" / "observer_steps.csv", step_rows)
    _write_csv(output_dir / "diagnostics" / "ue_qp_terms_summary.csv", ue_terms)
    _write_json(
        output_dir / "diagnostics" / "representative_case.json", representative
    )
    _plot_summary(
        summary,
        output_dir / "plots" / "actor_comparison.png",
        amplitude=amplitude,
    )
    _plot_representative(
        step_rows,
        representative,
        z_max=float(args.z_max),
        amplitude=amplitude,
        output_dir=output_dir,
    )
    _write_json(
        output_dir / "results" / "run_complete.json",
        {"completed": True, "episodes": len(episode_rows), "policies": len(conditions)},
    )

    print("\nSummary")
    for row in summary:
        print(
            f"{row['policy']}: safe={row['violation_free_episode_rate']:.3f} "
            f"p05_margin={row['hard_deck_margin_p05_across_episodes']:.4f} "
            f"pos_rmse={row['pos_error_norm_rmse_mean']:.4f} "
            f"intervention={row['intervention_norm_mean']:.4f} "
            f"fallback={row['qp_fallback_rate']:.4f}"
        )
    print(f"\nSaved results to: {output_dir}")
    print(f"Summary plot: {output_dir / 'plots' / 'actor_comparison.png'}")
    print(f"Time-position plot: {output_dir / 'plots' / 'representative_position_time.png'}")
    print(f"3-D plot: {output_dir / 'plots' / 'representative_trajectory_3d.png'}")


if __name__ == "__main__":
    main()
