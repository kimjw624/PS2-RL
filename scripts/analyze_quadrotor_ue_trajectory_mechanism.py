#!/usr/bin/env python3
"""Explain the trajectory-level safety separation between standard and UE-bCBF.

This add-only post-processor reads an already completed upward-ceiling sweep.
It does not run the environment, load a policy, or change any controller code.
For selected paired seeds it reconstructs altitude from the recorded ceiling
margin, estimates vertical velocity by finite differences, and compares the
filter intervention, physical QP slack, disturbance, and UE observer estimate.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np


def _parse_seeds(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not seeds:
        raise ValueError("--seeds requires at least one integer")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must not contain duplicates")
    return seeds


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _f(row: dict[str, str], name: str) -> float:
    return float(row[name])


def _first_time(time: np.ndarray, values: np.ndarray, predicate: Any) -> float:
    indices = np.flatnonzero(predicate(values))
    return float(time[indices[0]]) if indices.size else float("nan")


def _format_optional_time(value: float) -> str:
    return f"{value:.3f}s" if np.isfinite(value) else "none"


def _prepare_output_dir(run_dir: Path, raw: str, overwrite: bool) -> Path:
    output_dir = Path(raw).expanduser().resolve() if raw else run_dir / "trajectory_mechanism"
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is non-empty: {output_dir}\n"
            "Use a new path or pass --overwrite."
        )
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Completed single-cell ceiling-sweep directory")
    parser.add_argument("--seeds", default="4200000,4200002", help="Comma-separated paired episode seeds")
    parser.add_argument("--intervention-threshold", type=float, default=1.0)
    parser.add_argument("--slack-threshold", type=float, default=1e-5)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _parse_seeds(args.seeds)
    if not np.isfinite(args.intervention_threshold) or args.intervention_threshold < 0.0:
        raise ValueError("--intervention-threshold must be nonnegative and finite")
    if not np.isfinite(args.slack_threshold) or args.slack_threshold <= 0.0:
        raise ValueError("--slack-threshold must be positive and finite")

    run_dir = Path(args.run_dir).expanduser().resolve()
    input_path = run_dir / "diagnostics" / "observer_steps.csv"
    all_rows = _read_csv(input_path)
    output_dir = _prepare_output_dir(run_dir, args.output_dir, args.overwrite)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required for the mechanism plot: {exc}") from exc

    colors = {"standard": "#D55E00", "ue": "#0072B2"}
    labels = {"standard": "Standard BCBF", "ue": "UE-BCBF"}
    line_styles = {"standard": "-", "ue": "--"}
    summary_rows: list[dict[str, Any]] = []
    paired_summaries: list[dict[str, Any]] = []
    fig, axes = plt.subplots(
        5,
        len(seeds),
        figsize=(6.4 * len(seeds), 13.0),
        sharex="col",
        squeeze=False,
    )

    for column, seed in enumerate(seeds):
        by_filter: dict[str, dict[str, np.ndarray]] = {}
        for filter_name in ("standard", "ue"):
            rows = sorted(
                (
                    row
                    for row in all_rows
                    if int(row["episode_seed"]) == seed and row["filter"] == filter_name
                ),
                key=lambda row: int(row["step"]),
            )
            if not rows:
                raise ValueError(f"No {filter_name} rows found for seed {seed} in {input_path}")

            time = np.asarray([_f(row, "time_sec") for row in rows], dtype=np.float64)
            margin = np.asarray([_f(row, "hard_deck_margin") for row in rows], dtype=np.float64)
            z_max = np.asarray([_f(row, "z_max") for row in rows], dtype=np.float64)
            altitude = z_max - margin
            vertical_velocity = np.gradient(altitude, time)
            intervention = np.asarray([_f(row, "intervention_norm") for row in rows])
            slack = np.asarray([_f(row, "slack") for row in rows])
            disturbance_z = np.asarray([_f(row, "disturbance_z") for row in rows])
            d_hat_z = np.asarray([_f(row, "d_hat_z") for row in rows])
            observer_bound = np.asarray([_f(row, "observer_error_bound") for row in rows])
            fallback = np.asarray([_f(row, "qp_fallback") for row in rows])
            inequality_residual = np.asarray(
                [_f(row, "qp_max_positive_inequality_residual") for row in rows]
            )

            if np.any(np.diff(time) <= 0.0):
                raise ValueError(f"Non-increasing time grid for {filter_name}, seed {seed}")
            by_filter[filter_name] = {
                "time": time,
                "margin": margin,
                "vertical_velocity": vertical_velocity,
                "intervention": intervention,
                "slack": slack,
                "disturbance_z": disturbance_z,
                "d_hat_z": d_hat_z,
                "observer_bound": observer_bound,
            }

            first_intervention = _first_time(
                time,
                intervention,
                lambda values: values > float(args.intervention_threshold),
            )
            first_slack = _first_time(
                time,
                slack,
                lambda values: values > float(args.slack_threshold),
            )
            first_violation = _first_time(time, margin, lambda values: values < 0.0)
            finite_residual = inequality_residual[np.isfinite(inequality_residual)]
            summary_rows.append(
                {
                    "episode_seed": seed,
                    "filter": filter_name,
                    "violation_free": float(np.min(margin) >= 0.0),
                    "minimum_ceiling_margin": float(np.min(margin)),
                    "maximum_vertical_velocity": float(np.max(vertical_velocity)),
                    "first_intervention_time_sec": first_intervention,
                    "maximum_intervention_norm": float(np.max(intervention)),
                    "first_physical_slack_time_sec": first_slack,
                    "maximum_physical_slack": float(np.max(slack)),
                    "first_violation_time_sec": first_violation,
                    "qp_fallback_steps": int(np.sum(fallback > 0.5)),
                    "maximum_full_inequality_residual": (
                        float(np.max(finite_residual)) if finite_residual.size else float("nan")
                    ),
                }
            )

            plot_kwargs = {
                "color": colors[filter_name],
                "linestyle": line_styles[filter_name],
                "linewidth": 2.0,
                "label": labels[filter_name],
            }
            axes[0, column].plot(time, margin, **plot_kwargs)
            axes[1, column].plot(time, vertical_velocity, **plot_kwargs)
            axes[2, column].plot(time, intervention, **plot_kwargs)
            axes[3, column].plot(time, np.maximum(slack, 1e-12), **plot_kwargs)

        standard = by_filter["standard"]
        ue = by_filter["ue"]
        if not np.allclose(standard["time"], ue["time"], atol=1e-8, rtol=0.0):
            raise ValueError(f"Standard and UE time grids differ for seed {seed}")

        standard_summary = next(
            row for row in summary_rows if row["episode_seed"] == seed and row["filter"] == "standard"
        )
        ue_summary = next(
            row for row in summary_rows if row["episode_seed"] == seed and row["filter"] == "ue"
        )
        paired_summaries.append(
            {
                "episode_seed": seed,
                "standard_violation_free": standard_summary["violation_free"],
                "ue_violation_free": ue_summary["violation_free"],
                "ue_intervention_lead_sec": (
                    standard_summary["first_intervention_time_sec"]
                    - ue_summary["first_intervention_time_sec"]
                ),
                "standard_first_slack_time_sec": standard_summary["first_physical_slack_time_sec"],
                "standard_first_violation_time_sec": standard_summary["first_violation_time_sec"],
                "ue_minus_standard_minimum_margin": (
                    ue_summary["minimum_ceiling_margin"]
                    - standard_summary["minimum_ceiling_margin"]
                ),
            }
        )
        intervention_lead = paired_summaries[-1]["ue_intervention_lead_sec"]
        axes[2, column].text(
            0.98,
            0.94,
            f"UE threshold crossing {intervention_lead:.3f} s earlier",
            transform=axes[2, column].transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

        time = ue["time"]
        axes[4, column].plot(
            time,
            ue["disturbance_z"],
            color="black",
            linewidth=2.0,
            label=r"True $d_z$",
        )
        axes[4, column].plot(
            time,
            ue["d_hat_z"],
            color=colors["ue"],
            linestyle="--",
            linewidth=2.0,
            label=r"UE estimate $\hat d_z$",
        )
        finite_bound = np.isfinite(ue["observer_bound"]) & np.isfinite(ue["d_hat_z"])
        if np.any(finite_bound):
            axes[4, column].fill_between(
                time[finite_bound],
                ue["d_hat_z"][finite_bound] - ue["observer_bound"][finite_bound],
                ue["d_hat_z"][finite_bound] + ue["observer_bound"][finite_bound],
                color=colors["ue"],
                alpha=0.13,
                label=r"$\hat d_z \pm \bar e$",
            )

        first_slack = standard_summary["first_physical_slack_time_sec"]
        first_violation = standard_summary["first_violation_time_sec"]
        for axis in axes[:, column]:
            axis.axvline(0.2, color="#888888", linewidth=1.0, linestyle=":", alpha=0.8)
            if np.isfinite(first_slack):
                axis.axvline(first_slack, color="#A50F15", linewidth=1.1, linestyle=":")
            if np.isfinite(first_violation):
                axis.axvline(first_violation, color="#A50F15", linewidth=1.2, linestyle="--")
            axis.grid(True, alpha=0.25)

        axes[0, column].axhline(0.0, color="black", linewidth=1.0)
        axes[1, column].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axes[3, column].set_yscale("log")
        status = (
            "both safe"
            if standard_summary["violation_free"] > 0.5
            else "standard unsafe / UE safe"
        )
        axes[0, column].set_title(f"Seed {seed}: {status}", fontsize=12, weight="bold")
        axes[4, column].set_xlabel("Time [s]")

    row_labels = (
        "Ceiling margin [m]",
        r"Estimated $v_z$ [m/s]",
        "Intervention norm",
        "Physical QP slack",
        r"Upward disturbance [m/s$^2$]",
    )
    for row_index, label in enumerate(row_labels):
        axes[row_index, 0].set_ylabel(label)
        axes[row_index, 0].legend(loc="best", fontsize=8)

    fig.suptitle(
        "Trajectory mechanism: disturbance-aware intervention precedes standard-filter slack",
        fontsize=15,
        weight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.006,
        "Gray dotted: observer warm-up end. Red dotted: first standard slack event. "
        "Red dashed: first standard ceiling violation. Vertical velocity is reconstructed by finite differences.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.035, 0.028, 0.995, 0.975))

    seeds_tag = "_".join(str(seed) for seed in seeds)
    plot_path = output_dir / "plots" / f"trajectory_mechanism_seeds_{seeds_tag}.png"
    fig.savefig(plot_path, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary_path = output_dir / "results" / "trajectory_mechanism_summary.csv"
    paired_path = output_dir / "results" / "trajectory_mechanism_paired.csv"
    _write_csv(summary_path, summary_rows)
    _write_csv(paired_path, paired_summaries)

    print("Trajectory mechanism diagnostic")
    print(f"input: {input_path}")
    for row in paired_summaries:
        print(
            f"seed={row['episode_seed']} standard_safe={row['standard_violation_free']:.0f} "
            f"ue_safe={row['ue_violation_free']:.0f} "
            f"UE_intervention_lead={row['ue_intervention_lead_sec']:.3f}s "
            f"standard_slack_t={_format_optional_time(row['standard_first_slack_time_sec'])} "
            f"standard_violation_t={_format_optional_time(row['standard_first_violation_time_sec'])} "
            f"margin_delta={row['ue_minus_standard_minimum_margin']:.4f}m"
        )
    print(f"plot: {plot_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
