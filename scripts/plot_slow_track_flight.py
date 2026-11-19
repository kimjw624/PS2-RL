#!/usr/bin/env python
"""Figure + short report of slow-track flights (SITL, hardware or rehearsal): numpy and matplotlib only.

    python scripts/plot_slow_track_flight.py ~/ws_shared/experiments/slow_track/20261118_160000-pd
    python scripts/plot_slow_track_flight.py --latest ~/ws_shared/experiments/slow_track          # newest flight
    python scripts/plot_slow_track_flight.py FLIGHT_A FLIGHT_B                                     # overlay (PD vs policy)

Writes ``<flight>/flight.png`` (or ``overlay.png`` next to the first flight): top view of the reference and
the flown path in the task frame, height, position error over time, thrust and body-rate commands.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load(folder: Path) -> dict:
    rows = list(csv.DictReader(open(folder / "flight.csv", encoding="utf-8")))
    pol = [r for r in rows if r["phase"] == "POLICY" and r["t_policy"]]
    if not pol:
        raise SystemExit(f"{folder}: no controller phase in flight.csv")
    col = lambda name: np.array([float(r[name]) for r in pol])  # noqa: E731
    d = {k: col(k) for k in ("t_policy", "px", "py", "pz", "ref_x", "ref_y", "ref_z", "pos_err", "a_cmd", "wx", "wy", "wz",
                             "wx_meas", "wy_meas", "dhat_x", "dhat_y", "dhat_z")}
    s = folder / "summary.json"
    d["summary"] = json.loads(s.read_text()) if s.exists() else {}
    d["name"] = folder.name
    return d


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
        s = d["summary"].get("controller", {})
        e = d["pos_err"]
        print(f"{d['name']}: {d['summary'].get('outcome', '?')}")
        print(f"  {len(e)} steps, {d['t_policy'][-1]:.1f} s; position error mean {100 * e.mean():.1f} cm, p95 "
              f"{100 * np.percentile(e, 95):.1f} cm, max {100 * e.max():.1f} cm; thrust {d['a_cmd'].min():.2f}..{d['a_cmd'].max():.2f} m/s^2; "
              f"|rate cmd| max {np.abs(np.c_[d['wx'], d['wy'], d['wz']]).max():.2f} rad/s; d_hat at the end "
              f"{np.round([d['dhat_x'][-1], d['dhat_y'][-1], d['dhat_z'][-1]], 2).tolist()} m/s^2")
        if s.get("rate_response"):
            r = s["rate_response"]
            print(f"  rate response: delay {1e3 * r['delay_s']:.0f} ms + lag {1e3 * r['tau_s']:.0f} ms")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
    d0 = fl[0]
    # a straight reference has no width: an equal-aspect top view would be a flat line, so show the
    # sideways (y) error along the path instead
    straight = float(np.ptp(d0["ref_y"])) < 0.3
    if not straight:
        ax[0, 0].plot(d0["ref_x"], d0["ref_y"], "k--", lw=1.3, label="reference")
    for d in fl:
        if straight:
            ax[0, 0].plot(d["px"], 100 * (d["py"] - d["ref_y"]), lw=1.0, label=d["name"])
        else:
            ax[0, 0].plot(d["px"], d["py"], lw=1.0, label=d["name"])
        ax[0, 1].plot(d["t_policy"], d["pz"], lw=1.0, label=d["name"])
        ax[1, 0].plot(d["t_policy"], 100 * d["pos_err"], lw=1.0, label=d["name"])
    ax[0, 1].plot(d0["t_policy"], d0["ref_z"], "k--", lw=1.0, label="reference")
    if straight:
        ax[0, 0].axhline(0.0, c="k", ls="--", lw=1.0)
        ax[0, 0].set_title("straight reference: sideways error [cm] along x (east) [m]")
    else:
        ax[0, 0].set_aspect("equal"); ax[0, 0].set_title("top view, task frame (x east) [m]")
    ax[0, 1].set_title("height in the task frame [m] (0 = hand-over height)")
    ax[1, 0].set_title("position error [cm]"); ax[1, 0].set_xlabel("time since the hand-over [s]")
    for d in fl:
        ax[1, 1].plot(d["t_policy"], d["a_cmd"] - 9.81, lw=0.8, label=f"{d['name']}: thrust - g [m/s^2]")
    ax[1, 1].plot(d0["t_policy"], d0["wx"], lw=0.6, label=f"{d0['name']}: roll rate cmd [rad/s]")
    ax[1, 1].plot(d0["t_policy"], d0["wy"], lw=0.6, label=f"{d0['name']}: pitch rate cmd [rad/s]")
    ax[1, 1].set_title("commands")
    for x in ax.flat:
        x.grid(alpha=0.3)
        x.legend(fontsize=7)
    fig.tight_layout()
    out = folders[0] / ("flight.png" if len(fl) == 1 else "overlay.png")
    fig.savefig(out, dpi=110)
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
