#!/usr/bin/env python
"""Figure + short report of obstacle-only demo flights (SITL, hardware or rehearsal): numpy and matplotlib only.

    python3 scripts/plot_cyl_online_flight.py ~/ws_shared/experiments/cyl_online/20261120_150000-S1_planner
    python3 scripts/plot_cyl_online_flight.py --latest ~/ws_shared/experiments/cyl_online        # newest flight
    python3 scripts/plot_cyl_online_flight.py FLIGHT_A FLIGHT_B                                   # overlay

Writes ``<flight>/flight.png`` (or ``overlay.png`` next to the first flight): top view in the task frame (x east from
the take-off point, y north) with the virtual cylinders, the contact ring (cylinder + drone radius), the hover-set reach
and the goal circle, the path coloured by the filter's blend lam (1 = the nominal kept, 0 = the backup); speed and lam,
clearance, and height over time.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
FLIGHT_COLORS = ["#2a78d6", "#1baf7a", "#eb6834", "#8a5cd6", "#c2185b"]


def load(folder: Path) -> dict:
    rows = list(csv.DictReader(open(folder / "flight.csv", encoding="utf-8")))
    pol = [r for r in rows if r["phase"] == "POLICY" and r["t_policy"]]
    if not pol:
        raise SystemExit(f"{folder}: no controller phase in flight.csv")
    col = lambda name: np.array([float(r[name]) for r in pol])  # noqa: E731
    d = {k: col(k) for k in ("t_policy", "px", "py", "pz", "vx", "vy", "vz", "lam", "clearance", "a_cmd", "wx", "wy", "wz",
                             "dhat_x", "dhat_y", "dhat_z", "decide_ms", "latency_ms")}
    d["mode"] = [r["mode"] for r in pol]
    s = folder / "summary.json"
    d["summary"] = json.loads(s.read_text()) if s.exists() else {}
    d["name"] = folder.name
    return d


def report(d: dict) -> None:
    s = d["summary"].get("controller", {})
    sp = np.sqrt(d["vx"] ** 2 + d["vy"] ** 2 + d["vz"] ** 2)
    modes, counts = np.unique(d["mode"], return_counts=True)
    print(f"{d['name']}: {d['summary'].get('outcome', '?')}")
    ge = s.get("goal_entered_s")
    print(f"  {len(d['t_policy'])} steps, {d['t_policy'][-1]:.1f} s; goal circle " + (f"entered after {ge:.2f} s" if ge else "not reached")
          + f"; min clearance {d['clearance'].min():+.2f} m; y {d['py'].min():+.2f}..{d['py'].max():+.2f} m; "
          f"speed up to {sp.max():.2f} m/s; height {d['pz'].min():.2f}..{d['pz'].max():.2f} m")
    print(f"  filter: lam mean {d['lam'].mean():.2f}, {int(np.sum(np.abs(np.diff(d['lam'])) > 0.25))} switches; modes "
          + ", ".join(f"{m} {100 * c / len(d['mode']):.0f} %" for m, c in zip(modes, counts))
          + f"; decision median {np.median(d['decide_ms']):.2f} / max {d['decide_ms'].max():.2f} ms, step max "
          f"{d['latency_ms'].max():.2f} ms ({d['summary'].get('runtime', {}).get('backend', '?')})")
    print(f"  thrust {d['a_cmd'].min():.2f}..{d['a_cmd'].max():.2f} m/s^2; |rate cmd| max "
          f"{np.abs(np.c_[d['wx'], d['wy'], d['wz']]).max():.2f} rad/s; d_hat at the end "
          f"{np.round([d['dhat_x'][-1], d['dhat_y'][-1], d['dhat_z'][-1]], 2).tolist()} m/s^2")
    if s.get("rate_response"):
        r = s["rate_response"]
        print(f"  rate response: delay {1e3 * r['delay_s']:.0f} ms + lag {1e3 * r['tau_s']:.0f} ms")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("flights", nargs="*")
    p.add_argument("--latest", default="", help="take the newest flight folder under this root")
    a = p.parse_args(argv)
    folders = [Path(f).expanduser() for f in a.flights]
    if a.latest:
        cands = sorted([q for q in Path(a.latest).expanduser().iterdir() if (q / "flight.csv").exists()])
        if not cands:
            raise SystemExit(f"no flight under {a.latest}")
        folders.append(cands[-1])
    if not folders:
        raise SystemExit("give flight folders or --latest ROOT")
    fl = [load(f) for f in folders]
    for d in fl:
        report(d)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlesize": 9.5})
    cmap = LinearSegmentedColormap.from_list("lam", ["#e34948", "#eda100", "#2a78d6"])
    course = fl[0]["summary"].get("course", {})
    cyl = np.asarray(course.get("cylinders_xyr", []), dtype=float).reshape(-1, 3)
    goal = np.asarray(course.get("goal", [np.nan, np.nan, np.nan]), dtype=float)
    r_d, rho = float(course.get("r_drone", 0.35)), float(course.get("rho_xy", 0.40))

    fig = plt.figure(figsize=(10.5, 8.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.3, 1.0, 1.0], hspace=0.38)
    ax = fig.add_subplot(gs[0])
    for c in cyl:
        ax.add_patch(Circle(c[0:2], c[2], color="#9a9893", zorder=3))
        ax.add_patch(Circle(c[0:2], c[2] + r_d, fill=False, ls="--", lw=0.8, color=INK2, zorder=3))
        ax.add_patch(Circle(c[0:2], c[2] + r_d + rho, fill=False, ls=":", lw=0.8, color=MUTED, zorder=3))
    if np.all(np.isfinite(goal[0:2])):
        ax.plot([goal[0]], [goal[1]], marker="*", color=INK, ms=10, zorder=6)
        ax.add_patch(Circle(goal[0:2], float(course.get("goal_radius", 0.5)), fill=False, ls=":", lw=0.8, color=INK, zorder=3))
    ax.plot([0.0], [0.0], marker="o", color=INK, ms=4, zorder=6)
    handles = []
    if len(fl) == 1:
        d = fl[0]
        pts = np.c_[d["px"], d["py"]]
        lc = LineCollection(np.stack([pts[:-1], pts[1:]], axis=1), cmap=cmap, norm=plt.Normalize(0, 1), lw=2.2, zorder=4)
        lc.set_array(d["lam"][:-1])
        ax.add_collection(lc)
        fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.01, label="lam (1 = nominal kept)")
    else:
        for i, d in enumerate(fl):
            col = FLIGHT_COLORS[i % len(FLIGHT_COLORS)]
            ax.plot(d["px"], d["py"], lw=1.6, color=col, zorder=4)
            ov = d["lam"] < 0.999
            ax.plot(d["px"][ov], d["py"][ov], ls="none", marker="o", ms=2.2, mew=0, color="#e34948", zorder=5)
            handles.append(Line2D([], [], color=col, lw=1.6, label=d["name"]))
        handles.append(Line2D([], [], color="#e34948", ls="none", marker="o", ms=3, label="filter overriding"))
    handles += [Line2D([], [], color=INK2, lw=0.8, ls="--", label="contact (cylinder + drone radius)"),
                Line2D([], [], color=MUTED, lw=0.8, ls=":", label="+ hover-set reach; goal: arrival circle")]
    ax.legend(handles=handles, fontsize=7.5, loc="upper left", frameon=False)
    xs = np.concatenate([d["px"] for d in fl] + [cyl[:, 0] if len(cyl) else np.zeros(0), [0.0, goal[0] if np.isfinite(goal[0]) else 0.0]])
    ys = np.concatenate([d["py"] for d in fl] + [cyl[:, 1] - cyl[:, 2] - r_d - rho if len(cyl) else np.zeros(0),
                                                cyl[:, 1] + cyl[:, 2] + r_d + rho if len(cyl) else np.zeros(0)])
    ax.set_xlim(float(xs.min()) - 0.6, float(xs.max()) + 0.8)
    ax.set_ylim(float(ys.min()) - 0.3, float(ys.max()) + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x east of the take-off point [m]")
    ax.set_ylabel("y north [m]")
    ax.grid(color=GRID, lw=0.6)
    s0 = fl[0]["summary"]
    short = lambda text, n: text if len(text) <= n else text[: n - 1] + "\u2026"  # noqa: E731
    ax.set_title(f"{short(course.get('label', ''), 95)}\n{short(s0.get('outcome', ''), 105)}", loc="left", color=INK)

    a1 = fig.add_subplot(gs[1])
    a2 = fig.add_subplot(gs[2], sharex=a1)
    for i, d in enumerate(fl):
        col = FLIGHT_COLORS[i % len(FLIGHT_COLORS)]
        sp = np.sqrt(d["vx"] ** 2 + d["vy"] ** 2)
        lab = "" if len(fl) == 1 else f" ({d['name']})"
        a1.plot(d["t_policy"], sp, lw=1.3, color=col, label="horizontal speed [m/s]" + lab)
        a1.plot(d["t_policy"], d["lam"], lw=0.9, color=col if len(fl) > 1 else "#6c4bd1", alpha=0.8, ls="-" if len(fl) == 1 else ":",
                label="blend lam" + lab)
        a2.plot(d["t_policy"], d["clearance"], lw=1.3, color=col, label="clearance: nearest surface - drone radius [m]" + lab)
        a2.plot(d["t_policy"], d["pz"], lw=0.9, color=col, ls="--", label="height above the take-off point [m]" + lab)
    a2.axhline(0.0, color=INK, lw=0.8)
    a1.set_ylim(-0.05, 1.35)
    a2.set_xlabel("time since the hand-over [s]")
    for x in (a1, a2):
        x.grid(color=GRID, lw=0.6)
        x.legend(fontsize=7, loc="upper right", frameon=False, ncol=2)
    out = folders[0] / ("flight.png" if len(fl) == 1 else "overlay.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
