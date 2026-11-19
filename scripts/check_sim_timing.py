#!/usr/bin/env python3
"""Check a slow-track SITL flight for simulator IMU gaps and the vertical-velocity kicks they cause.

Why this exists
---------------
In Gazebo SITL, PX4's gz bridge stamps every IMU sample with PX4's clock at the moment the message
arrives (GZBridge::imuCallback uses hrt_absolute_time()). When the computer is busy, an IMU message
can arrive late, so one IMU sample spans a long interval. The IMU integrator accepts it, and EKF2
then limits the integration time of each down-sampled IMU sample to 2 x EKF2_PREDICT_US (20 ms by
default) for gravity, but keeps the full measured change in velocity. The difference,
g x (gap - 20 ms), goes into the estimated vertical velocity, always upwards, because the
accelerometer always measures about +g along the vertical.

In the flight log this looks like:
  1. the odometry position jumps ahead of its own time stamp along the velocity
     (the output predictor integrated the long sample right away);
  2. about 0.1 s later (EKF2's delayed fusion horizon) the vertical velocity steps up by
     a few cm/s while the measured thrust does not change;
  3. any controller that damps vertical velocity (the PD, the policy, PX4's own position control)
     cuts thrust and settles below the reference: e_z ~ -(kd/kp) x (velocity error).

Horizontal velocity is not affected (specific force there is ~0), which is why only height suffers.

Usage
-----
    python3 check_sim_timing.py <flight folder or flight.csv> [more ...] [--plot out.png]

The position jump can only be measured while the vehicle moves (> 0.15 m/s); kicks are detected
in the whole controller phase.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

G = 9.81
GAP_MIN_S = 0.010          # report position jumps ahead of the clock larger than this
GAP_BIG_S = 0.025          # gaps of this size were followed by kicks in the logs analysed so far
KICK_WINDOW_S = 0.16
KICK_MIN_M_S = 0.03
MOVE_MIN_M_S = 0.15


def load(path: str) -> tuple[str, dict]:
    if os.path.isdir(path):
        path = os.path.join(path, "flight.csv")
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit(f"{path}: empty")
    data = {}
    for key in rows[0]:
        if key == "phase":
            data[key] = np.array([r[key] for r in rows])
        else:
            data[key] = np.array([float(r[key]) if r[key] not in ("", None) else np.nan for r in rows])
    name = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return name, data


def controller_rows(d: dict) -> dict:
    t = d["t_flight"]
    m = d["phase"] == "POLICY"
    if "t_policy" in d:
        m &= np.nan_to_num(d["t_policy"], nan=1e9) < 1e5      # drop rows after a broken PX4 clock
    m &= t < 1e5
    return {k: v[m] for k, v in d.items()}


def clock_gaps(d: dict) -> list[tuple[float, float]]:
    """(time, size) of odometry position jumps ahead of the time stamp, while moving."""
    t = d["t_flight"]
    p = np.stack([d["px"], d["py"]], 1)
    v = np.stack([d["vx"], d["vy"]], 1)
    dt = np.diff(t)
    dp = np.diff(p, axis=0)
    vm = 0.5 * (v[1:] + v[:-1])
    sp2 = (vm ** 2).sum(1)
    ok = sp2 > MOVE_MIN_M_S ** 2
    extra = np.full(len(dt), np.nan)
    extra[ok] = (dp[ok] * vm[ok]).sum(1) / sp2[ok] - dt[ok]
    events: list[tuple[float, float]] = []
    for i in np.where(np.nan_to_num(extra) > GAP_MIN_S)[0]:
        if events and t[i + 1] - events[-1][0] < 0.3:
            events[-1] = (events[-1][0], max(events[-1][1], extra[i]))
        else:
            events.append((float(t[i + 1]), float(extra[i])))
    return events


def vz_kicks(d: dict) -> list[tuple[float, float]]:
    """(time, size) of rises in vertical velocity that the measured thrust does not explain."""
    t, vz = d["t_flight"], d["vz"]
    a = d.get("a_meas")
    have_a = a is not None and np.isfinite(a).sum() > 0.5 * len(a)
    n = max(1, int(round(KICK_WINDOW_S / max(np.median(np.diff(t)), 1e-3))))
    events: list[tuple[float, float]] = []
    last = -1e9
    for i in range(n, len(t)):
        j = i - n
        rise = vz[i] - vz[j]
        if have_a:
            seg = slice(j, i)
            acc = np.nan_to_num(a[seg] - G)
            rise -= float(np.sum(acc * np.diff(t[j:i + 1])))
        if rise > KICK_MIN_M_S and t[i] - last > 0.6:
            events.append((float(t[j]), float(rise)))
            last = t[i]
    return events


def vz_inconsistency(d: dict, window: float = 5.0) -> float:
    t, z, vz = d["t_flight"], d["pz"], d["vz"]
    worst = 0.0
    for a in np.arange(t[0], t[-1] - 0.5 * window, window):
        s = (t >= a) & (t < a + window)
        if s.sum() > 20:
            worst = max(worst, abs(float(np.mean(vz[s]) - np.polyfit(t[s], z[s], 1)[0])))
    return worst


def analyse(path: str) -> dict:
    name, raw = load(path)
    d = controller_rows(raw)
    if len(d["t_flight"]) < 50:
        raise SystemExit(f"{name}: no controller phase in the log")
    ez = d["pz"] - d["ref_z"]
    eh = np.hypot(d["px"] - d["ref_x"], d["py"] - d["ref_y"])
    gaps = clock_gaps(d)
    kicks = vz_kicks(d)
    return dict(name=name, d=d, ez=ez, eh=eh, gaps=gaps, kicks=kicks,
                duration=float(d["t_flight"][-1] - d["t_flight"][0]),
                incons=vz_inconsistency(d))


def report(r: dict) -> None:
    gaps, kicks = r["gaps"], r["kicks"]
    big = [g for g in gaps if g[1] >= GAP_BIG_S]
    print(f"== {r['name']}  ({r['duration']:.1f} s under the controller)")
    print(f"   height error: mean {100 * np.mean(r['ez']):+.1f} cm, lowest {100 * np.min(r['ez']):+.1f} cm;"
          f"  horizontal mean {100 * np.mean(r['eh']):.1f} cm")
    print(f"   PX4 vz vs height change: worst 5-s mismatch {100 * r['incons']:.1f} cm/s")
    sizes = " ".join(f"{1000 * g[1]:.0f}" for g in gaps) or "-"
    print(f"   simulator IMU gaps (position ahead of its stamp, while moving): {len(gaps)}"
          f"  [{sizes} ms]; >= {1000 * GAP_BIG_S:.0f} ms: {len(big)}")
    ks = " ".join(f"+{100 * k[1]:.1f}" for k in kicks) or "-"
    print(f"   upward vz kicks not explained by thrust: {len(kicks)}  [{ks} cm/s]")
    if kicks:
        lags = []
        for tk, _ in kicks:
            before = [g for g in gaps if -0.05 <= tk - g[0] <= 0.25]
            lags.append(f"{1000 * (tk - before[-1][0]):.0f}" if before else "n/a")
        print(f"   gap -> kick delay [ms]: {' '.join(lags)}  (n/a: vehicle too slow to see the gap)")
    if len(kicks) >= 2 and np.min(r["ez"]) < -0.05:
        print("   -> height error is the simulator's IMU timing, not the controller")
    elif not kicks and not big:
        print("   -> clean: no IMU gaps large enough to disturb PX4's vertical velocity")


def plot(results: list[dict], out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, orange, ink, muted = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
    fig, axes = plt.subplots(len(results), 1, figsize=(10, 2.6 * len(results)), squeeze=False, sharex=False)
    for ax, r in zip(axes[:, 0], results):
        t = r["d"]["t_flight"] - r["d"]["t_flight"][0]
        t0 = r["d"]["t_flight"][0]
        ax.plot(t, 100 * r["ez"], color=blue, lw=1.6, label="height error [cm]")
        for tg, sz in r["gaps"]:
            if sz >= GAP_BIG_S:
                ax.axvline(tg - t0, color=orange, lw=1.2, alpha=0.8)
        ax.plot([], [], color=orange, lw=1.2, label=f"IMU gap >= {1000 * GAP_BIG_S:.0f} ms")
        kt = [k[0] - t0 for k in r["kicks"]]
        if kt:
            yk = np.interp(kt, t, 100 * r["ez"])
            ax.scatter(kt, yk, s=36, color=ink, zorder=3, label="vz kick", edgecolor="#fcfcfb", linewidth=1.5)
        ax.axhline(0, color=muted, lw=0.8)
        ax.set_title(r["name"], loc="left", fontsize=10, color=ink)
        ax.set_ylabel("cm", color=muted)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", fontsize=8, frameon=False)
    axes[-1, 0].set_xlabel("time since hand-over [s]", color=muted)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"plot: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("flights", nargs="+", help="flight folders or flight.csv files")
    ap.add_argument("--plot", default="", help="write a figure (one row per flight)")
    args = ap.parse_args()
    results = [analyse(p) for p in args.flights]
    for r in results:
        report(r)
    if args.plot:
        plot(results, args.plot)


if __name__ == "__main__":
    sys.exit(main())
