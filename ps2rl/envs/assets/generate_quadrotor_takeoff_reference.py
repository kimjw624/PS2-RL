#!/usr/bin/env python3
"""Generate a slow vertical-takeoff reference for QuadrotorEnv.

State ordering (10D):
    [p_x, p_y, p_z, v_x, v_y, v_z, q_w, q_x, q_y, q_z]

The trajectory is a pure vertical climb held at both ends:

    [hold at z_start] -> [quintic S-curve climb] -> [hold at z_end]

Because the reference acceleration is purely vertical, the required thrust axis
is world +z for the whole flight, so the reference attitude is level
(q = identity) and ``omega_cmd`` is identically zero. The only nontrivial
reference channel is ``a_cmd = g + z_ddot``, which stays within a few percent of
hover for a slow climb. This is the gentlest reference the quadrotor env
accepts, and it is intended as the easy counterpart to the powerloop.

The quintic profile s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5 has zero velocity AND
zero acceleration at both endpoints, so the joints with the holds are C2 and the
thrust profile has no steps.

Two intended presets:

1. Sub-ceiling (``--z-end`` below ``--z-max``): the reference never approaches
   the hard deck, so the Control-Invariant Layer should stay inactive
   (proj_norm ~ 0). Use it to validate the pipeline end to end.

2. Ceiling-pressing (``--z-end`` above ``--z-max``): the reference deliberately
   commands an altitude above the hard deck and holds it there, exactly as the
   powerloop reference does at its apex (z_top=3.5 > z_max=3.0). The CIL must
   arrest the climb at the deck and hold, so tracking error is expected and the
   safe rate must stay at 1.0.

Outputs (same bundle layout as the powerloop generator):
1) a legacy state-only trajectory array (.npy),
2) a reference bundle (.npz) with states, time, thrust, and omega_cmd,
3) a diagnostic figure of position/velocity/attitude/quaternion traces (.png),
4) a JSON file with all generation/output configuration values.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class TakeoffConfig:
    dt: float = 0.02
    z_start: float = 0.5
    z_end: float = 2.5
    climb_duration: float = 8.0
    hold_start: float = 1.0
    hold_end: float = 3.0
    x_center: float = 0.0
    y_center: float = 0.0
    gravity: float = 9.81
    a_cmd_max_g: float = 4.0
    a_cmd_min: float = 0.0
    omega_max: float = 18.0
    z_max: float = 3.0

    def validate(self) -> None:
        if self.dt <= 0.0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.climb_duration <= 0.0:
            raise ValueError(f"climb_duration must be positive, got {self.climb_duration}")
        if self.hold_start < 0.0 or self.hold_end < 0.0:
            raise ValueError(
                f"hold durations must be nonnegative, got {self.hold_start} / {self.hold_end}"
            )
        if self.gravity <= 0.0:
            raise ValueError(f"gravity must be positive, got {self.gravity}")
        if self.a_cmd_max_g <= 0.0:
            raise ValueError(f"a_cmd_max_g must be positive, got {self.a_cmd_max_g}")
        if self.omega_max <= 0.0:
            raise ValueError(f"omega_max must be positive, got {self.omega_max}")
        if self.z_start < 0.0:
            raise ValueError(f"z_start must be nonnegative, got {self.z_start}")
        if abs(self.z_end - self.z_start) < 1e-9:
            raise ValueError("z_end must differ from z_start (this is a climb reference).")
        if self.z_max <= 0.0:
            raise ValueError(f"z_max must be positive, got {self.z_max}")
        if self.z_start >= self.z_max:
            raise ValueError(
                f"z_start must start below the hard deck, got z_start={self.z_start}, z_max={self.z_max}"
            )


def _build_time_grid(total_duration: float, dt: float) -> np.ndarray:
    n_full = int(np.floor(total_duration / dt))
    t = dt * np.arange(n_full + 1, dtype=np.float64)
    if total_duration - t[-1] > 1e-12:
        t = np.append(t, total_duration)
    return t


def _quintic_profile(tau: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """s, ds/dtau, d2s/dtau2 for the C2 quintic 10t^3 - 15t^4 + 6t^5 on [0, 1]."""
    tau = np.clip(np.asarray(tau, dtype=np.float64), 0.0, 1.0)
    s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    ds = 30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4
    dds = 60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3
    return s, ds, dds


def _generate_takeoff(cfg: TakeoffConfig) -> dict[str, np.ndarray]:
    cfg.validate()

    total_duration = cfg.hold_start + cfg.climb_duration + cfg.hold_end
    t = _build_time_grid(total_duration, cfg.dt)

    dz = cfg.z_end - cfg.z_start
    t_climb_start = cfg.hold_start
    t_climb_end = cfg.hold_start + cfg.climb_duration

    tau = (t - t_climb_start) / cfg.climb_duration
    s, ds, dds = _quintic_profile(tau)

    # Outside the climb window the profile is clamped, so derivatives vanish.
    in_climb = (t >= t_climb_start) & (t <= t_climb_end)
    ds = np.where(in_climb, ds, 0.0)
    dds = np.where(in_climb, dds, 0.0)

    z = cfg.z_start + dz * s
    vz = dz * ds / cfg.climb_duration
    az = dz * dds / (cfg.climb_duration**2)

    x = np.full_like(t, cfg.x_center)
    y = np.full_like(t, cfg.y_center)
    vx = np.zeros_like(t)
    vy = np.zeros_like(t)

    # Purely vertical virtual acceleration -> thrust axis is world +z, attitude level.
    nu = np.stack([np.zeros_like(t), np.zeros_like(t), az + cfg.gravity], axis=1)
    a_cmd = np.linalg.norm(nu, axis=1)
    if np.any(a_cmd < 1e-9):
        raise RuntimeError(
            "Encountered near-zero required thrust; the climb decelerates at ~ -g. "
            "Slow the climb down (longer --climb-duration)."
        )

    qw = np.ones_like(t)
    qx = np.zeros_like(t)
    qy = np.zeros_like(t)
    qz = np.zeros_like(t)

    roll = np.zeros_like(t)
    pitch = np.zeros_like(t)
    yaw = np.zeros_like(t)

    states = np.column_stack([x, y, z, vx, vy, vz, qw, qx, qy, qz]).astype(np.float64)
    quat = states[:, 6:10]
    quat_norm = np.linalg.norm(quat, axis=1)
    quat_step_norm = np.zeros_like(t)
    if quat.shape[0] > 1:
        quat_step_norm[1:] = np.linalg.norm(np.diff(quat, axis=0), axis=1)

    omega_cmd = np.zeros((t.shape[0], 3), dtype=np.float64)

    a_cmd_limit = cfg.a_cmd_max_g * cfg.gravity
    max_a_cmd = float(np.max(a_cmd))
    min_a_cmd = float(np.min(a_cmd))
    max_abs_omega = float(np.max(np.abs(omega_cmd)))
    if max_a_cmd > a_cmd_limit + 1e-6:
        raise RuntimeError(
            f"Trajectory violates thrust limit: max a_cmd={max_a_cmd:.3f} > {a_cmd_limit:.3f} m/s^2."
        )
    if min_a_cmd < cfg.a_cmd_min - 1e-6:
        raise RuntimeError(
            f"Trajectory violates lower thrust limit: min a_cmd={min_a_cmd:.3f} < {cfg.a_cmd_min:.3f} m/s^2."
        )
    if max_abs_omega > cfg.omega_max + 1e-6:
        raise RuntimeError(
            f"Trajectory violates body-rate limit: max |omega|={max_abs_omega:.3f} > {cfg.omega_max:.3f} rad/s."
        )

    deck_excess = float(np.max(z) - cfg.z_max)
    steps_above_deck = int(np.count_nonzero(z > cfg.z_max))

    return {
        "t": t,
        "states": states,
        "a_cmd": a_cmd,
        "omega_cmd": omega_cmd,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "quat_norm": quat_norm,
        "quat_step_norm": quat_step_norm,
        "max_a_cmd": np.array([max_a_cmd], dtype=np.float64),
        "min_a_cmd": np.array([min_a_cmd], dtype=np.float64),
        "max_abs_omega": np.array([max_abs_omega], dtype=np.float64),
        "max_abs_vz": np.array([float(np.max(np.abs(vz)))], dtype=np.float64),
        "max_abs_az": np.array([float(np.max(np.abs(az)))], dtype=np.float64),
        "max_quat_step_norm": np.array([float(np.max(quat_step_norm))], dtype=np.float64),
        "max_quat_norm_error": np.array([float(np.max(np.abs(quat_norm - 1.0)))], dtype=np.float64),
        "deck_excess": np.array([deck_excess], dtype=np.float64),
        "steps_above_deck": np.array([steps_above_deck], dtype=np.float64),
    }


def _save_plot(data: dict[str, np.ndarray], output_path: Path, *, z_max: float) -> None:
    t = data["t"]
    states = data["states"]
    a_cmd = data["a_cmd"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex=True)
    series = [
        (states[:, 0], "x [m]", None),
        (states[:, 1], "y [m]", None),
        (states[:, 2], "z [m]", z_max),
        (states[:, 3], "xdot [m/s]", None),
        (states[:, 4], "ydot [m/s]", None),
        (states[:, 5], "zdot [m/s]", None),
        (a_cmd, "a_cmd [m/s^2]", None),
        (data["omega_cmd"][:, 1], "omega_y [rad/s]", None),
        (data["quat_norm"], "||q||", None),
    ]

    for ax, (y, ylabel, hline) in zip(axes.ravel(), series):
        ax.plot(t, y, linewidth=2.0)
        ax.set_ylabel(ylabel)
        if hline is not None:
            ax.axhline(hline, color="r", linestyle="--", linewidth=1.2, alpha=0.7, label="z_max")
            ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.35)

    for ax in axes[-1, :]:
        ax.set_xlabel("time [s]")

    fig.suptitle(
        "Slow Vertical-Takeoff Reference (Quadrotor 10D State)\n"
        f"max |vz| = {float(data['max_abs_vz'][0]):.3f} m/s, "
        f"a_cmd in [{float(data['min_a_cmd'][0]):.3f}, {float(data['max_a_cmd'][0]):.3f}] m/s^2, "
        f"max |omega| = {float(data['max_abs_omega'][0]):.3e} rad/s",
        fontsize=13,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dt", type=float, default=0.02, help="Reference sample timestep [s] (match env_dt).")
    parser.add_argument("--z-start", type=float, default=0.5, help="Starting altitude [m].")
    parser.add_argument("--z-end", type=float, default=2.5, help="Commanded final altitude [m].")
    parser.add_argument(
        "--climb-duration",
        type=float,
        default=None,
        help="Climb duration [s]. Mutually exclusive with --climb-speed.",
    )
    parser.add_argument(
        "--climb-speed",
        type=float,
        default=None,
        help="Peak climb speed [m/s]; sets climb_duration = 1.875 * |dz| / v_peak.",
    )
    parser.add_argument("--hold-start", type=float, default=1.0, help="Hover hold before the climb [s].")
    parser.add_argument("--hold-end", type=float, default=3.0, help="Hover hold after the climb [s].")
    parser.add_argument("--x-center", type=float, default=0.0)
    parser.add_argument("--y-center", type=float, default=0.0)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--a-cmd-max-g", type=float, default=4.0)
    parser.add_argument("--a-cmd-min", type=float, default=0.0)
    parser.add_argument("--omega-max", type=float, default=18.0)
    parser.add_argument("--z-max", type=float, default=3.0, help="Hard-deck altitude [m] (for reporting/plot only).")

    parser.add_argument("--output-bundle", type=str, default="quadrotor_takeoff_reference.npz")
    parser.add_argument("--output-npy", type=str, default="quadrotor_takeoff_reference_legacy.npy")
    parser.add_argument("--output-figure", type=str, default="quadrotor_takeoff_reference.png")
    parser.add_argument("--output-config-json", type=str, default="quadrotor_takeoff_reference_config.json")
    args = parser.parse_args(argv)

    if args.climb_duration is not None and args.climb_speed is not None:
        parser.error("--climb-duration and --climb-speed are mutually exclusive.")
    if args.climb_duration is None and args.climb_speed is None:
        args.climb_duration = 8.0
    if args.climb_speed is not None:
        if args.climb_speed <= 0.0:
            parser.error("--climb-speed must be positive.")
        # Quintic peak speed is 1.875 * dz / T.
        args.climb_duration = 1.875 * abs(args.z_end - args.z_start) / args.climb_speed
    return args


def main(argv=None) -> None:
    args = _parse_args(argv)

    cfg = TakeoffConfig(
        dt=args.dt,
        z_start=args.z_start,
        z_end=args.z_end,
        climb_duration=args.climb_duration,
        hold_start=args.hold_start,
        hold_end=args.hold_end,
        x_center=args.x_center,
        y_center=args.y_center,
        gravity=args.gravity,
        a_cmd_max_g=args.a_cmd_max_g,
        a_cmd_min=args.a_cmd_min,
        omega_max=args.omega_max,
        z_max=args.z_max,
    )

    data = _generate_takeoff(cfg)

    bundle_path = Path(args.output_bundle).expanduser().resolve()
    npy_path = Path(args.output_npy).expanduser().resolve()
    fig_path = Path(args.output_figure).expanduser().resolve()
    json_path = Path(args.output_config_json).expanduser().resolve()

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(bundle_path, **data)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, data["states"])
    _save_plot(data, fig_path, z_max=cfg.z_max)

    stats = {
        "num_points": int(data["states"].shape[0]),
        "total_duration": float(data["t"][-1]),
        "max_a_cmd": float(data["max_a_cmd"][0]),
        "min_a_cmd": float(data["min_a_cmd"][0]),
        "max_abs_omega": float(data["max_abs_omega"][0]),
        "max_abs_vz": float(data["max_abs_vz"][0]),
        "max_abs_az": float(data["max_abs_az"][0]),
        "max_quat_step_norm": float(data["max_quat_step_norm"][0]),
        "max_quat_norm_error": float(data["max_quat_norm_error"][0]),
        "z_min": float(np.min(data["states"][:, 2])),
        "z_max_reached": float(np.max(data["states"][:, 2])),
        "deck_excess": float(data["deck_excess"][0]),
        "steps_above_deck": int(data["steps_above_deck"][0]),
    }

    payload = {
        "takeoff_config": asdict(cfg),
        "takeoff_trajectory_stats": stats,
        "output_paths": {
            "reference_bundle_npz": bundle_path.name,
            "trajectory_npy": npy_path.name,
            "diagnostic_figure": fig_path.name,
            "config_json": json_path.name,
        },
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[takeoff] wrote bundle  : {bundle_path}")
    print(f"[takeoff] wrote npy     : {npy_path}")
    print(f"[takeoff] wrote figure  : {fig_path}")
    print(f"[takeoff] wrote config  : {json_path}")
    print(
        f"[takeoff] {stats['num_points']} points over {stats['total_duration']:.2f} s"
        f" | z {stats['z_min']:.2f} -> {stats['z_max_reached']:.2f} m"
        f" | max|vz| {stats['max_abs_vz']:.3f} m/s"
        f" | a_cmd [{stats['min_a_cmd']:.3f}, {stats['max_a_cmd']:.3f}] m/s^2"
        f" | max|omega| {stats['max_abs_omega']:.2e} rad/s"
    )
    if stats["steps_above_deck"] > 0:
        print(
            f"[takeoff] NOTE: reference exceeds the hard deck z_max={cfg.z_max:.2f} m by "
            f"{stats['deck_excess']:.2f} m for {stats['steps_above_deck']} samples "
            f"({stats['steps_above_deck'] * cfg.dt:.2f} s). The CIL is expected to intervene."
        )
    else:
        print(
            f"[takeoff] reference stays {cfg.z_max - stats['z_max_reached']:.2f} m below the hard deck "
            f"z_max={cfg.z_max:.2f} m; the CIL should stay inactive."
        )


if __name__ == "__main__":
    main()
