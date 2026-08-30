#!/usr/bin/env python3
"""Run the final UE-bCBF constant-upward validation in one terminal command.

The runner performs four ordered stages:

1. Recreate the trajectory-mechanism plot from an existing matched run.
2. Pilot a predeclared constant +z amplitude grid on pilot-only seeds.
3. Select challenge amplitudes using only standard-bCBF pilot safety.
4. Evaluate A=0 and the selected amplitudes on fresh paired holdout seeds.

Every simulator cell is a separate Python process so GPU allocations are
released between cells.  A zero-frequency, 90-degree-phase sinusoid is exactly
constant: d_z(t) = A.  Consequently the UE bounds are delta_d=A, delta_v=0.
The runner freezes the actor and applies the same 1e6 slack weight to both
filters.  It creates a consolidated plot, statistical records, and one archive.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
from typing import Any

import numpy as np


def _parse_floats(raw: str, *, name: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError(f"{name} requires at least one value")
    if not all(np.isfinite(value) for value in values):
        raise ValueError(f"All {name} values must be finite")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def _amplitude_tag(amplitude: float) -> str:
    return f"A{amplitude:g}".replace("-", "m").replace(".", "p")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _cell_complete(path: Path, expected_episodes: int) -> bool:
    completion_path = path / "results" / "run_complete.json"
    if not completion_path.is_file():
        return False
    completion = _read_json(completion_path)
    return bool(completion.get("completed")) and int(completion.get("episodes", -1)) == expected_episodes


def _run_and_tee(command: list[str], *, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n$ {shlex.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _evaluation_command(
    args: argparse.Namespace,
    *,
    amplitude: float,
    repeats: int,
    seed_start: int,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(args.evaluator),
        "--nominal-run-dir",
        str(args.nominal_run_dir),
        "--ue-config-run-dir",
        str(args.ue_config_run_dir),
        "--nominal-checkpoint",
        args.nominal_checkpoint,
        "--z-max-values",
        f"{args.z_max:g}",
        "--amplitudes",
        f"{amplitude:g}",
        "--frequency-hz",
        "0",
        "--phase-deg",
        "90",
        "--repeats",
        str(repeats),
        "--seed-start",
        str(seed_start),
        "--scaled-slack-qp",
        "--scaled-slack-qp-float64",
        "--slack-weight-override",
        f"{args.slack_weight:g}",
        "--output-dir",
        str(output_dir),
    ]
    if args.overwrite_incomplete:
        command.append("--overwrite")
    return command


def _run_evaluation_cell(
    args: argparse.Namespace,
    *,
    stage: str,
    amplitude: float,
    repeats: int,
    seed_start: int,
    output_dir: Path,
    child_env: dict[str, str],
) -> None:
    expected_episodes = repeats * 2
    if _cell_complete(output_dir, expected_episodes):
        print(f"Skipping completed {stage} cell: {output_dir}")
        return
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite_incomplete:
        raise FileExistsError(
            f"Incomplete non-empty cell: {output_dir}\n"
            "Inspect it, then rerun with --overwrite-incomplete or choose a new output directory."
        )
    command = _evaluation_command(
        args,
        amplitude=amplitude,
        repeats=repeats,
        seed_start=seed_start,
        output_dir=output_dir,
    )
    log_path = args.output_dir / "logs" / f"{stage}_{_amplitude_tag(amplitude)}.log"
    _run_and_tee(command, env=child_env, log_path=log_path)
    if not _cell_complete(output_dir, expected_episodes):
        raise RuntimeError(f"Cell did not produce a valid completion record: {output_dir}")


def _standard_safety_rate(cell_dir: Path) -> float:
    rows = _read_csv(cell_dir / "results" / "summary.csv")
    row = next((row for row in rows if row["filter"] == "standard"), None)
    if row is None:
        raise ValueError(f"No standard summary row in {cell_dir}")
    return float(row["violation_free_episode_rate"])


def _select_amplitudes(
    amplitudes: tuple[float, ...], standard_rates: dict[float, float]
) -> tuple[float, float]:
    """Select challenge levels without looking at UE pilot performance."""

    ordered = tuple(sorted(amplitudes))
    primary = min(ordered, key=lambda amplitude: (abs(standard_rates[amplitude] - 0.5), amplitude))
    higher = tuple(amplitude for amplitude in ordered if amplitude > primary)
    if higher:
        secondary = higher[0]
    else:
        lower = tuple(amplitude for amplitude in ordered if amplitude < primary)
        if not lower:
            raise ValueError("At least two pilot amplitudes are required")
        secondary = lower[-1]
    return primary, secondary


def _exact_mcnemar_p(ue_only: int, standard_only: int) -> float:
    discordant = ue_only + standard_only
    if discordant == 0:
        return 1.0
    smaller = min(ue_only, standard_only)
    tail = sum(math.comb(discordant, index) for index in range(smaller + 1)) / (2.0**discordant)
    return min(1.0, 2.0 * tail)


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        rate * (1.0 - rate) / total + z * z / (4.0 * total * total)
    ) / denominator
    return center - half, center + half


def _consolidate_holdout(
    args: argparse.Namespace,
    amplitudes: tuple[float, ...],
    holdout_dirs: dict[float, Path],
) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    all_episode_rows: list[dict[str, Any]] = []
    statistics: list[dict[str, Any]] = []

    for amplitude in amplitudes:
        cell_dir = holdout_dirs[amplitude]
        for row in _read_csv(cell_dir / "results" / "summary.csv"):
            output_row: dict[str, Any] = {
                "amplitude": amplitude,
                "filter": row["filter"],
            }
            for key, value in row.items():
                if key not in output_row:
                    output_row[key] = value
            summary_rows.append(output_row)

        episodes = _read_csv(cell_dir / "results" / "episodes.csv")
        for row in episodes:
            copied = dict(row)
            copied["source_cell"] = str(cell_dir)
            all_episode_rows.append(copied)
        index = {
            (int(row["episode_seed"]), row["filter"]): row
            for row in episodes
        }
        ue_only = 0
        standard_only = 0
        margin_deltas: list[float] = []
        seeds = sorted({int(row["episode_seed"]) for row in episodes})
        for seed in seeds:
            standard = index[(seed, "standard")]
            ue = index[(seed, "ue")]
            standard_safe = float(standard["violation_free"]) >= 0.5
            ue_safe = float(ue["violation_free"]) >= 0.5
            ue_only += int(ue_safe and not standard_safe)
            standard_only += int(standard_safe and not ue_safe)
            margin_deltas.append(
                float(ue["hard_deck_margin_min"]) - float(standard["hard_deck_margin_min"])
            )
        statistics.append(
            {
                "amplitude": amplitude,
                "paired_seeds": len(seeds),
                "ue_safe_standard_unsafe": ue_only,
                "standard_safe_ue_unsafe": standard_only,
                "mcnemar_two_sided_exact_p": _exact_mcnemar_p(ue_only, standard_only),
                "ue_minus_standard_margin_mean": float(np.mean(margin_deltas)),
                "ue_minus_standard_margin_median": float(np.median(margin_deltas)),
            }
        )

    results_dir = args.output_dir / "results"
    _write_csv(results_dir / "final_holdout_summary.csv", summary_rows)
    _write_csv(results_dir / "final_holdout_episodes.csv", all_episode_rows)

    indexed_summary = {
        (float(row["amplitude"]), row["filter"]): row for row in summary_rows
    }
    diagnostics = {
        "maximum_fallback_rate": max(
            float(row["qp_fallback_rate"]) for row in summary_rows
        ),
        "maximum_full_inequality_residual": max(
            float(row["qp_solver_inequality_residual_max"]) for row in summary_rows
        ),
        "minimum_float64_rate": min(
            float(row["qp_solve_float64_rate"]) for row in summary_rows
        ),
        "minimum_ue_observer_coverage": min(
            float(row["observer_post_warmup_coverage_rate_mean"])
            for row in summary_rows
            if row["filter"] == "ue"
        ),
    }
    disturbed_advantages = [
        float(indexed_summary[(amplitude, "ue")]["violation_free_episode_rate"])
        - float(indexed_summary[(amplitude, "standard")]["violation_free_episode_rate"])
        for amplitude in amplitudes
        if amplitude > 0.0
    ]
    control_standard = float(indexed_summary[(0.0, "standard")]["violation_free_episode_rate"])
    control_ue = float(indexed_summary[(0.0, "ue")]["violation_free_episode_rate"])
    stop_gate = {
        "a0_both_at_least_98_percent_safe": min(control_standard, control_ue) >= 0.98,
        "at_least_one_disturbed_ue_advantage_ge_20_points": max(disturbed_advantages) >= 0.20,
        "zero_qp_fallback": diagnostics["maximum_fallback_rate"] == 0.0,
        "full_inequality_residual_le_1e_minus_6": (
            diagnostics["maximum_full_inequality_residual"] <= 1e-6
        ),
        "ue_observer_coverage_at_least_99_percent": (
            diagnostics["minimum_ue_observer_coverage"] >= 0.99
        ),
    }
    result = {
        "disturbance_definition": "constant +z: A*sin(2*pi*0*t + pi/2) = A",
        "ue_bounds": "delta_d=A, delta_v=0",
        "selection_rule": (
            "Primary amplitude minimizes |standard pilot safety - 0.5| with lower-A tie break; "
            "secondary is the next higher grid amplitude, or next lower if primary is maximum. "
            "UE pilot outcomes are not used for selection."
        ),
        "holdout_amplitudes": list(amplitudes),
        "statistics": statistics,
        "diagnostics": diagnostics,
        "predeclared_success_checks": stop_gate,
        "all_success_checks_passed": all(stop_gate.values()),
        "testing_stop_gate_reached": True,
        "required_next_action": "Freeze experiment state and organize the complete study.",
    }
    _write_json(results_dir / "final_holdout_statistics.json", result)
    _plot_final_holdout(args, amplitudes, indexed_summary, results_dir.parent / "plots" / "final_constant_validation.png")
    return result


def _plot_final_holdout(
    args: argparse.Namespace,
    amplitudes: tuple[float, ...],
    indexed_summary: dict[tuple[float, str], dict[str, Any]],
    output_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required for the final plot: {exc}") from exc

    filters = ("standard", "ue")
    colors = {"standard": "#D55E00", "ue": "#0072B2"}
    labels = {"standard": "Standard BCBF", "ue": "UE-BCBF"}
    x = np.arange(len(amplitudes), dtype=np.float64)
    width = 0.36
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2))
    metrics = (
        ("violation_free_episode_rate", "Violation-free episode rate"),
        ("hard_deck_margin_p05_across_episodes", "5th percentile minimum margin [m]"),
        ("pos_error_norm_rmse_mean", "Position-error RMSE"),
        ("qp_physical_slack_mean", "Mean physical QP slack"),
    )

    for metric_index, (metric, ylabel) in enumerate(metrics):
        axis = axes.flat[metric_index]
        for filter_index, filter_name in enumerate(filters):
            values = np.asarray(
                [float(indexed_summary[(amplitude, filter_name)][metric]) for amplitude in amplitudes]
            )
            positions = x + (filter_index - 0.5) * width
            kwargs: dict[str, Any] = {}
            if metric == "violation_free_episode_rate":
                lower: list[float] = []
                upper: list[float] = []
                for value in values:
                    successes = int(round(value * args.final_repeats))
                    low, high = _wilson_interval(successes, args.final_repeats)
                    lower.append(value - low)
                    upper.append(high - value)
                kwargs["yerr"] = np.asarray([lower, upper])
                kwargs["capsize"] = 3
            axis.bar(
                positions,
                values,
                width,
                color=colors[filter_name],
                label=labels[filter_name],
                **kwargs,
            )
        axis.set_xticks(x, [f"A={amplitude:g}" for amplitude in amplitudes])
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", alpha=0.25)
        if metric == "violation_free_episode_rate":
            axis.set_ylim(0.0, 1.08)
        if metric == "hard_deck_margin_p05_across_episodes":
            axis.axhline(0.0, color="black", linewidth=1.0)
        if metric == "qp_physical_slack_mean":
            axis.set_yscale("log")

    handles, plot_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        plot_labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.947),
    )
    fig.suptitle(
        "Final independent validation: constant upward disturbance",
        fontsize=15,
        weight="bold",
        y=0.992,
    )
    fig.text(
        0.5,
        0.012,
        f"Frozen actor; z_max={args.z_max:g} m; common slack weight={args.slack_weight:,.0f}; "
        f"{args.final_repeats} fresh paired seeds per amplitude; scaled float64 QP.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.03, 0.045, 0.99, 0.895))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _create_archive(output_dir: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.resolve().is_relative_to(output_dir.resolve()):
        raise ValueError("--archive-path must be outside --output-dir")
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(output_dir, arcname=output_dir.name)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal-run-dir", type=Path, required=True)
    parser.add_argument("--ue-config-run-dir", type=Path, required=True)
    parser.add_argument("--nominal-checkpoint", default="best")
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=project_root / "scripts" / "evaluate_quadrotor_ue_upward_ceiling_sweep.py",
    )
    parser.add_argument(
        "--mechanism-script",
        type=Path,
        default=project_root / "scripts" / "analyze_quadrotor_ue_trajectory_mechanism.py",
    )
    parser.add_argument("--mechanism-run-dir", type=Path, required=True)
    parser.add_argument("--mechanism-seeds", default="4200000,4200002")
    parser.add_argument("--z-max", type=float, default=2.7)
    parser.add_argument("--slack-weight", type=float, default=1e6)
    parser.add_argument("--pilot-amplitudes", default="1.4,1.6,1.8,2.0,2.2")
    parser.add_argument("--pilot-repeats", type=int, default=10)
    parser.add_argument("--pilot-seed-start", type=int, default=4_300_000)
    parser.add_argument("--final-repeats", type=int, default=50)
    parser.add_argument("--final-seed-start", type=int, default=4_400_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive-path", type=Path, default=None)
    parser.add_argument("--overwrite-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.nominal_run_dir = args.nominal_run_dir.expanduser().resolve()
    args.ue_config_run_dir = args.ue_config_run_dir.expanduser().resolve()
    args.evaluator = args.evaluator.expanduser().resolve()
    args.mechanism_script = args.mechanism_script.expanduser().resolve()
    args.mechanism_run_dir = args.mechanism_run_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.archive_path = (
        args.archive_path.expanduser().resolve()
        if args.archive_path is not None
        else args.output_dir.parent / f"{args.output_dir.name}.tar.gz"
    )
    pilot_amplitudes = _parse_floats(args.pilot_amplitudes, name="--pilot-amplitudes")
    if len(pilot_amplitudes) < 2 or any(amplitude <= 0.0 for amplitude in pilot_amplitudes):
        raise ValueError("--pilot-amplitudes requires at least two positive values")
    if args.pilot_repeats <= 0 or args.final_repeats <= 0:
        raise ValueError("Repeat counts must be positive")
    if args.z_max <= 0.0 or args.slack_weight <= 0.0:
        raise ValueError("--z-max and --slack-weight must be positive")
    for required_file in (args.evaluator, args.mechanism_script):
        if not required_file.is_file():
            raise FileNotFoundError(required_file)
    if not (args.mechanism_run_dir / "diagnostics" / "observer_steps.csv").is_file():
        raise FileNotFoundError(args.mechanism_run_dir / "diagnostics" / "observer_steps.csv")

    manifest_path = args.output_dir / "metadata" / "final_validation_manifest.json"
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not manifest_path.is_file():
        raise FileExistsError(
            f"Refusing unrelated non-empty output directory: {args.output_dir}\n"
            "Use a new path."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env.update(
        {
            "JAX_ENABLE_X64": "1",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
        }
    )

    runner_path = Path(__file__).resolve()
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "kind": "final_constant_upward_validation",
        "disturbance": {
            "direction": "+z",
            "generator_mode": "sinusoidal",
            "frequency_hz": 0.0,
            "phase_deg": 90.0,
            "identity": "A*sin(2*pi*0*t + pi/2) = A",
            "ue_delta_d": "A",
            "ue_delta_v": 0.0,
        },
        "selection_rule": (
            "Choose primary amplitude using standard-bCBF pilot safety only: minimize distance to 0.5, "
            "lower-A tie break. Choose next higher amplitude, or next lower if primary is grid maximum."
        ),
        "pilot_amplitudes": list(pilot_amplitudes),
        "pilot_repeats": args.pilot_repeats,
        "pilot_seed_start": args.pilot_seed_start,
        "final_repeats": args.final_repeats,
        "final_seed_start": args.final_seed_start,
        "z_max": args.z_max,
        "common_slack_weight": args.slack_weight,
        "nominal_run_dir": str(args.nominal_run_dir),
        "ue_config_run_dir": str(args.ue_config_run_dir),
        "mechanism_run_dir": str(args.mechanism_run_dir),
        "output_dir": str(args.output_dir),
        "archive_path": str(args.archive_path),
        "source_sha256": {
            str(runner_path): _sha256(runner_path),
            str(args.evaluator): _sha256(args.evaluator),
            str(args.mechanism_script): _sha256(args.mechanism_script),
        },
        "child_environment": {
            "JAX_ENABLE_X64": "1",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
        },
    }
    _write_json(manifest_path, manifest)

    mechanism_output = args.output_dir / "mechanism"
    mechanism_command = [
        sys.executable,
        str(args.mechanism_script),
        "--run-dir",
        str(args.mechanism_run_dir),
        "--seeds",
        args.mechanism_seeds,
        "--output-dir",
        str(mechanism_output),
    ]
    if args.overwrite_incomplete:
        mechanism_command.append("--overwrite")

    print("Final constant-upward validation")
    print(f"output: {args.output_dir}")
    print(f"archive: {args.archive_path}")
    print("disturbance: constant +z via frequency=0 Hz, phase=90 deg")
    print(f"pilot amplitudes: {list(pilot_amplitudes)}")
    print(f"pilot seeds: {args.pilot_seed_start}..{args.pilot_seed_start + args.pilot_repeats - 1}")
    print(f"holdout seeds: {args.final_seed_start}..{args.final_seed_start + args.final_repeats - 1}")

    if args.dry_run:
        print("\nDRY RUN: mechanism command")
        print(shlex.join(mechanism_command))
        print("\nDRY RUN: pilot commands")
        for amplitude in pilot_amplitudes:
            cell_dir = args.output_dir / "pilot" / (
                f"constant_{_amplitude_tag(amplitude)}_seed{args.pilot_seed_start}_r{args.pilot_repeats}"
            )
            print(
                shlex.join(
                    _evaluation_command(
                        args,
                        amplitude=amplitude,
                        repeats=args.pilot_repeats,
                        seed_start=args.pilot_seed_start,
                        output_dir=cell_dir,
                    )
                )
            )
        print("Holdout amplitudes are selected after the pilot using the recorded rule.")
        return

    mechanism_complete = (
        mechanism_output / "results" / "trajectory_mechanism_summary.csv"
    ).is_file() and any((mechanism_output / "plots").glob("trajectory_mechanism_*.png"))
    if mechanism_complete:
        print(f"Skipping completed mechanism analysis: {mechanism_output}")
    else:
        if mechanism_output.exists() and any(mechanism_output.iterdir()) and not args.overwrite_incomplete:
            raise FileExistsError(
                f"Incomplete non-empty mechanism output: {mechanism_output}\n"
                "Rerun with --overwrite-incomplete or use a new main output directory."
            )
        _run_and_tee(
            mechanism_command,
            env=child_env,
            log_path=args.output_dir / "logs" / "mechanism.log",
        )

    pilot_dirs: dict[float, Path] = {}
    for amplitude in pilot_amplitudes:
        cell_dir = args.output_dir / "pilot" / (
            f"constant_{_amplitude_tag(amplitude)}_seed{args.pilot_seed_start}_r{args.pilot_repeats}"
        )
        pilot_dirs[amplitude] = cell_dir
        _run_evaluation_cell(
            args,
            stage="pilot",
            amplitude=amplitude,
            repeats=args.pilot_repeats,
            seed_start=args.pilot_seed_start,
            output_dir=cell_dir,
            child_env=child_env,
        )

    standard_rates = {
        amplitude: _standard_safety_rate(pilot_dirs[amplitude])
        for amplitude in pilot_amplitudes
    }
    primary, secondary = _select_amplitudes(pilot_amplitudes, standard_rates)
    selection = {
        "standard_pilot_safety_rates": {
            f"{amplitude:g}": standard_rates[amplitude] for amplitude in pilot_amplitudes
        },
        "primary_amplitude": primary,
        "secondary_amplitude": secondary,
        "selection_used_ue_results": False,
    }
    _write_json(args.output_dir / "metadata" / "pilot_selection.json", selection)
    print(f"\nSelected holdout amplitudes: primary={primary:g}, secondary={secondary:g}")

    holdout_amplitudes = tuple(sorted({0.0, primary, secondary}))
    holdout_dirs: dict[float, Path] = {}
    for amplitude in holdout_amplitudes:
        cell_dir = args.output_dir / "holdout" / (
            f"constant_{_amplitude_tag(amplitude)}_seed{args.final_seed_start}_r{args.final_repeats}"
        )
        holdout_dirs[amplitude] = cell_dir
        _run_evaluation_cell(
            args,
            stage="holdout",
            amplitude=amplitude,
            repeats=args.final_repeats,
            seed_start=args.final_seed_start,
            output_dir=cell_dir,
            child_env=child_env,
        )

    result = _consolidate_holdout(args, holdout_amplitudes, holdout_dirs)
    _write_json(
        args.output_dir / "results" / "run_complete.json",
        {
            "completed": True,
            "testing_stop_gate_reached": True,
            "required_next_action": "organize",
            "all_success_checks_passed": result["all_success_checks_passed"],
        },
    )
    _create_archive(args.output_dir, args.archive_path)

    print("\nFinal holdout summary")
    summary_rows = _read_csv(args.output_dir / "results" / "final_holdout_summary.csv")
    for row in summary_rows:
        print(
            f"A={float(row['amplitude']):g} {row['filter']}: "
            f"safe={float(row['violation_free_episode_rate']):.3f} "
            f"margin_p05={float(row['hard_deck_margin_p05_across_episodes']):.4f} "
            f"tracking={float(row['pos_error_norm_rmse_mean']):.4f} "
            f"fallback={float(row['qp_fallback_rate']):.3g}"
        )
    print(f"plot: {args.output_dir / 'plots' / 'final_constant_validation.png'}")
    print(f"archive: {args.archive_path}")
    print("TESTING STOP GATE REACHED: freeze the experiment state and begin organization.")


if __name__ == "__main__":
    main()
