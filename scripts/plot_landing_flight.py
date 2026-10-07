#!/usr/bin/env python
"""Figures and a short report for recorded landing flights.

A flight folder holds ``flight.csv`` and ``summary.json`` as written by the flight logic
(``ps2rl/deploy/landing_flight.py``), whether the flight was flown in Gazebo through the ROS
bridge or offline with ``scripts/rehearse_landing_flight.py``.

    # one figure and one report per flight, written into the flight's folder
    python3 scripts/plot_landing_flight.py ~/ws_shared/experiments/landing/20261007_101500-landing

    # several flights, plus one figure that overlays them (filtered against unfiltered, say)
    python3 scripts/plot_landing_flight.py RUN_A RUN_B --compare compare.png

    # the newest flight under a folder
    python3 scripts/plot_landing_flight.py --latest ~/ws_shared/experiments/landing

Only numpy and matplotlib are needed; nothing is imported from ps2rl.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

NUMERIC_FROM = 2  # columns after t_flight, phase
ACTIVE_TOL = 1e-3  # the filter "acted" when it changed the command by more than this


def load(folder: Path) -> dict:
    """Read one flight folder into arrays (empty cells become NaN)."""
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    with open(folder / "flight.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{folder}: flight.csv has no rows")
    cols = list(rows[0].keys())
    data = {"phase": np.asarray([r["phase"] for r in rows])}
    for c in cols:
        if c != "phase":
            data[c] = np.asarray([float(r[c]) if r[c] not in ("", None) else math.nan for r in rows], dtype=np.float64)
    geo, fc = summary["geometry"], summary["flight_config"]
    pad = np.asarray(geo["pad"], dtype=np.float64)
    p = np.stack([data["px"], data["py"], data["pz"]], axis=1) - pad      # relative to the pad centre
    pol = data["phase"] == "POLICY"
    # horizontal axis of the side view: from the pad centre toward where the filtered descent began
    start_xy = p[pol][0, 0:2] if pol.any() else p[-1, 0:2]
    e = start_xy / np.linalg.norm(start_xy) if np.linalg.norm(start_xy) > 1e-3 else np.array([1.0, 0.0])
    data.update({
        "folder": folder, "name": folder.name, "summary": summary, "geo": geo, "fc": fc, "pol": pol,
        "p": p, "s": p[:, 0:2] @ e, "lateral": p[:, 0:2] @ np.array([-e[1], e[0]]),
        "speed": np.sqrt(data["vx"] ** 2 + data["vy"] ** 2 + data["vz"] ** 2),
        "goal": np.stack([data["goal_x"], data["goal_y"], data["goal_z"]], axis=1) - pad,
        "axis": e, "tan": math.tan(math.radians(geo["cone_theta_deg"])), "ground_z": -float(fc["pad_offset"][2]),
    })
    du = np.sqrt((data["a_cmd"] - data["a_nom"]) ** 2 + (data["wx"] - data["wx_nom"]) ** 2
                 + (data["wy"] - data["wy_nom"]) ** 2 + (data["wz"] - data["wz_nom"]) ** 2)
    data["du"] = du
    return data


def draw_cone(ax, d: dict, s_max: float, z_max: float) -> None:
    r0, tan = d["geo"]["cone_r0"], d["tan"]
    z = np.array([0.0, z_max])
    for sign in (-1.0, 1.0):
        ax.plot(sign * (r0 + tan * z), z, color="k", lw=1.2, ls="--", label="cone" if sign > 0 else None)
    ax.fill_between([-s_max, s_max], d["ground_z"] - 0.25, d["ground_z"], color="0.80", lw=0, label="ground")
    ax.axhline(0.0, color="tab:red", lw=1.0, ls=":", label="pad plane (the filter's floor)")
    ax.plot([-r0, r0], [0.0, 0.0], color="tab:red", lw=3.0, solid_capstyle="butt", label="pad")


def side_view(ax, d: dict, *, colour="tab:blue", label="filtered descent", context=True, mark_active=True) -> None:
    pol, s, z = d["pol"], d["s"], d["p"][:, 2]
    if context:
        before = np.isin(d["phase"], ("TAKEOFF", "GOTO_START"))
        after = np.isin(d["phase"], ("RECOVER", "LAND", "DONE"))
        ax.plot(s[before], z[before], color="0.6", lw=0.8, label="take-off and transit (autopilot)")
        ax.plot(s[after], z[after], color="0.6", lw=0.8, ls=":", label="after the descent (autopilot)")
    ax.plot(s[pol], z[pol], color=colour, lw=1.8, label=label)
    if pol.any():
        ax.plot(s[pol][0], z[pol][0], "o", color=colour, ms=5)
    if mark_active:
        act = pol & (d["du"] > ACTIVE_TOL)
        ax.plot(s[act], z[act], ".", color="tab:orange", ms=5, label="filter changed the command")


def report(d: dict, boundary_tol: float) -> list[str]:
    s, pol = d["summary"], d["pol"]
    ps = s.get("policy", {})
    out = [f"flight   {d['name']}", f"outcome  {s['outcome']}"]
    if not pol.any() or not ps.get("steps"):
        return out + ["no filtered descent was recorded"]
    rt = s["runtime"]
    t = d["t_policy"][pol]
    step = np.diff(t) * 1e3 if len(t) > 1 else np.array([math.nan])
    out.append(f"control  {rt['nominal']} -> target {np.round(np.asarray(rt['target']) - np.asarray(d['geo']['pad']), 3).tolist()} (rel. pad)"
               + (f", position error capped at {rt['max_position_error']:g} m" if rt["nominal"] == "lqr_target" and rt.get("max_position_error") else "")
               + f"; filter {'on ' + str(rt['filter_overrides']) if rt['use_filter'] else 'OFF'}")
    out.append(f"steps    {ps['steps']} in {ps['duration_s']:.2f} s; interval mean {np.nanmean(step):.1f} ms "
               f"(min {np.nanmin(step):.1f}, max {np.nanmax(step):.1f}) for a {rt['dt'] * 1e3:.0f} ms control period")
    h_min, above = ps["min_h_cone_m"], ps["min_height_m"] >= 0.0
    verdict = ("inside throughout" if h_min >= 0.0 else
               f"on the boundary (outside by at most {-h_min * 1e3:.0f} mm)" if h_min >= -boundary_tol else f"left the cone by {-h_min:.3f} m")
    out.append(f"cone     lowest margin {h_min:+.3f} m -> {verdict}")
    out.append(f"floor    lowest height above the pad plane {ps['min_height_m']:+.3f} m"
               + ("" if above else f" (below the plane; the ground is {-d['ground_z']:.3f} m under it)"))
    if ps.get("time_to_touch_height_s") is not None:
        out.append(f"touch    reached {d['fc']['zeta_touch'] * 100:.0f} cm after {ps['time_to_touch_height_s']:.2f} s at "
                   f"{ps['speed_at_touch_height_m_s']:.2f} m/s; ended {ps['final_radial_m']:.2f} m from the pad centre")
    else:
        out.append(f"touch    never came within {d['fc']['zeta_touch'] * 100:.0f} cm of the pad plane; ended at "
                   f"{ps['final_height_m']:+.3f} m, {ps['final_radial_m']:.2f} m from the pad centre")
    out.append(f"filter   changed the command on {100 * ps['filter_active_fraction']:.0f} % of steps; largest slack {ps['max_slack']:.1e}; "
               f"QP fallbacks {ps['qp_fallback_steps']}; safeguard blends {ps['safeguard_steps']}; thrust saturated on {ps['thrust_saturated_steps']} steps")
    lat = ps["latency_ms"]
    out.append(f"latency  median {lat['median']:.1f} ms, p95 {lat['p95']:.1f}, max {lat['max']:.1f}")
    if "rate_tracking_rms_rad_s" in ps:
        e = ps["rate_tracking_rms_rad_s"]
        out.append(f"rates    commanded - measured, RMS: roll {e[0]:.2f}, pitch {e[1]:.2f}, yaw {e[2]:.2f} rad/s")
    if "thrust_tracking_rms_m_s2" in ps:
        out.append(f"thrust   commanded - measured: mean {ps['thrust_tracking_mean_m_s2']:+.2f} m/s^2 (a steady offset is thrust-map error), "
                   f"RMS {ps['thrust_tracking_rms_m_s2']:.2f}")
    h = ps.get("handover") or {}
    if h:
        out.append(f"start    hand-over {h['position_error_m']:.2f} m from the start point at {h['speed_m_s']:.2f} m/s, tilt {h['tilt_deg']:.1f} deg; "
                   f"cone margin {h['h_cone_m']:.2f} m, height {h['height_m']:.2f} m")
    out.append(f"lateral  largest distance from the plane of the side view {np.nanmax(np.abs(d['lateral'][pol])):.3f} m")
    return out


def figure(d: dict, path: Path) -> None:
    pol, s = d["pol"], d["summary"]
    rt = s["runtime"]
    t = d["t_policy"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    fig.suptitle(f"{d['name']}  -  {s['outcome']}", fontsize=12)

    ax = axes[0, 0]
    s_max = float(np.nanmax(np.abs(d["s"]))) + 0.4
    z_max = float(np.nanmax(d["p"][:, 2])) + 0.3
    draw_cone(ax, d, s_max, z_max)
    side_view(ax, d)
    goal_s, goal_z = d["goal"][pol][:, 0:2] @ d["axis"], d["goal"][pol][:, 2]
    if pol.any():
        if np.nanmax(np.abs(goal_s - goal_s[0])) + np.nanmax(np.abs(goal_z - goal_z[0])) > 1e-6:
            ax.plot(goal_s, goal_z, color="tab:green", lw=1.0, ls="--", label="reference")
        else:
            ax.plot(goal_s[0], goal_z[0], "*", color="tab:green", ms=10, label="target of the nominal controller")
    ax.set_xlim(-s_max, s_max)
    ax.set_ylim(d["ground_z"] - 0.25, z_max)
    ax.set_aspect("equal")
    ax.set_xlabel("horizontal distance from the pad centre, toward the start [m]")
    ax.set_ylabel("height above the pad plane [m]")
    ax.set_title("side view")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)

    ax = axes[0, 1]
    ax.plot(t[pol], d["height"][pol], color="tab:blue", label="height above the pad plane")
    ax.plot(t[pol], d["h_cone"][pol], color="tab:purple", label="cone margin")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(d["ground_z"], color="0.6", lw=0.8, ls="--", label="ground")
    ax.set_xlabel("time since the hand-over [s]")
    ax.set_ylabel("[m]")
    ax.set_title("safety margins (both stay >= 0 when the filter holds)")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    ax.plot(t[pol], d["speed"][pol], color="tab:blue", label="speed")
    ax.plot(t[pol], -d["vz"][pol], color="tab:cyan", lw=1.0, label="descent rate")
    ax.axhline(d["fc"]["v_touch"], color="0.5", lw=0.8, ls=":", label="touchdown speed limit of the test")
    ax.set_xlabel("time since the hand-over [s]")
    ax.set_ylabel("[m/s]")
    ax.set_title("speed")
    ax.legend(fontsize=8)

    lo, hi = np.asarray(rt["action_low"]), np.asarray(rt["action_high"])
    ax = axes[1, 0]
    ax.plot(t[pol], d["a_nom"][pol], color="tab:green", lw=1.0, ls="--", label="nominal")
    ax.plot(t[pol], d["a_cmd"][pol], color="tab:blue", label="commanded (after the filter)")
    if np.isfinite(d["a_meas"][pol]).any():
        ax.plot(t[pol], d["a_meas"][pol], color="tab:red", lw=0.8, label="measured")
    for v in (lo[0], hi[0]):
        ax.axhline(v, color="k", lw=0.8, ls=":")
    ax.set_xlabel("time since the hand-over [s]")
    ax.set_ylabel("thrust acceleration [m/s$^2$]")
    ax.set_title("thrust (dotted: the controller's limits)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for key, colour, name in (("wx", "tab:blue", "roll"), ("wy", "tab:orange", "pitch"), ("wz", "tab:green", "yaw")):
        ax.plot(t[pol], d[key][pol], color=colour, label=f"{name} commanded")
        if np.isfinite(d[key + "_meas"][pol]).any():
            ax.plot(t[pol], d[key + "_meas"][pol], color=colour, lw=0.8, ls="--", label=f"{name} measured")
    for v in (lo[1], hi[1]):
        ax.axhline(v, color="k", lw=0.8, ls=":")
    ax.set_xlabel("time since the hand-over [s]")
    ax.set_ylabel("body rate [rad/s]")
    ax.set_title("body rates (dotted: roll / pitch limits)")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 2]
    ax.plot(t[pol], d["du"][pol], color="tab:orange", label="size of the filter's correction")
    blend = pol & (d["safeguard_lambda"] < 1.0)
    fallback = pol & (d["qp_used"] < 0.5)
    if blend.any():
        ax.plot(t[blend], d["du"][blend], "v", color="tab:purple", ms=5, label="safeguard blended toward the backup")
    if fallback.any():
        ax.plot(t[fallback], d["du"][fallback], "x", color="tab:red", ms=7, label="QP fallback (backup action)")
    ax.set_xlabel("time since the hand-over [s]")
    ax.set_ylabel("|u - u_nominal|")
    ax.set_title("filter activity and timing")
    ax2 = ax.twinx()
    if pol.sum() > 1:
        ax2.plot(t[pol][1:], np.diff(t[pol]) * 1e3, ".", color="0.65", ms=3, label="control-step interval")
    ax2.plot(t[pol], d["latency_ms"][pol], color="0.3", lw=0.7, ls=":", label="computation time")
    ax2.axhline(rt["dt"] * 1e3, color="0.65", lw=0.6)
    ax2.set_ylabel("[ms]")
    ax2.set_ylim(0.0, max(3.0 * rt["dt"] * 1e3, float(np.nanmax(np.diff(t[pol]))) * 1.1e3 if pol.sum() > 1 else 0.0))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def compare(flights: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.25, 1.0]})
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    s_max = max(float(np.nanmax(np.abs(d["s"][d["pol"]]))) for d in flights if d["pol"].any()) + 0.4
    z_max = max(float(np.nanmax(d["p"][d["pol"], 2])) for d in flights if d["pol"].any()) + 0.3
    draw_cone(axes[0], flights[0], s_max, z_max)
    for i, d in enumerate(flights):
        c = colours[i % len(colours)]
        label = f"{d['name']}: {d['summary']['outcome']}"
        side_view(axes[0], d, colour=c, label=label, context=False, mark_active=False)
        axes[1].plot(d["t_policy"][d["pol"]], d["h_cone"][d["pol"]], color=c, label=f"{d['name']}: cone margin")
        axes[1].plot(d["t_policy"][d["pol"]], d["height"][d["pol"]], color=c, ls="--", lw=1.0, label=f"{d['name']}: height")
    axes[0].set_xlim(-s_max, s_max)
    axes[0].set_ylim(flights[0]["ground_z"] - 0.25, z_max)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("horizontal distance from the pad centre, toward the start [m]")
    axes[0].set_ylabel("height above the pad plane [m]")
    axes[0].set_title("side view (each flight in the vertical plane through its own start)")
    axes[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_xlabel("time since the hand-over [s]")
    axes[1].set_ylabel("[m]")
    axes[1].set_title("safety margins")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def newest(root: Path) -> Path:
    runs = [p for p in root.iterdir() if (p / "summary.json").is_file() and (p / "flight.csv").is_file()]
    if not runs:
        raise SystemExit(f"no flight folders with flight.csv and summary.json under {root}")
    return max(runs, key=lambda p: (p / "summary.json").stat().st_mtime)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="*", help="flight folders (each with flight.csv and summary.json)")
    ap.add_argument("--latest", default="", help="use the newest flight folder under this directory")
    ap.add_argument("--compare", default="", help="also write one figure overlaying all the given flights")
    ap.add_argument("--no_figure", action="store_true", help="print the report only")
    ap.add_argument("--boundary_tol", type=float, default=0.01,
                    help="a cone margin this little below zero is reported as riding the boundary, not as leaving the cone [m]")
    args = ap.parse_args(argv)
    folders = [Path(f).expanduser() for f in args.folders]
    if args.latest:
        folders.append(newest(Path(args.latest).expanduser()))
    if not folders:
        ap.error("give at least one flight folder, or --latest ROOT")
    flights = []
    for folder in folders:
        if not ((folder / "summary.json").is_file() and (folder / "flight.csv").is_file()):
            print(f"skipped {folder}: no summary.json / flight.csv (the flight did not finish, or this is not a flight folder)\n")
            continue
        d = load(folder)
        flights.append(d)
        lines = report(d, args.boundary_tol)
        print("\n".join(lines) + "\n")
        (folder / "analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if not args.no_figure and d["pol"].any():
            figure(d, folder / "flight.png")
            print(f"wrote {folder / 'flight.png'}\n")
    if not flights:
        raise SystemExit("no flight folder with records among the given paths")
    if args.compare:
        usable = [d for d in flights if d["pol"].any()]
        if len(usable) < 2:
            raise SystemExit("--compare needs at least two flights with a filtered descent")
        compare(usable, Path(args.compare).expanduser())
        print(f"wrote {args.compare}")


if __name__ == "__main__":
    main()
