#!/usr/bin/env python
"""Nominal bCBF vs UE-bCBF under stronger disturbances (outside-cone task).

Reads ``<root>/<case>/eval.json`` written by ``scripts/eval_phase2_cone_obstacle_ue.py`` for a grid of
disturbance cases (see ``docker_batch/run_cone_obstacle_disturbance_sweep.sh``); case folders are named
``<policy>__<mode>__<dir>__a<scale>`` (dir: rand / wall / ...). Plots, per disturbance direction, the
unsafe rate and the deepest violation (or smallest margin) against the disturbance amplitude for every
(policy, filter) pair, and writes a table (sweep.json / sweep.md).

    python scripts/plot_cone_obstacle_disturbance_sweep.py --root outputs/cone_obstacle_phase2_ue/dist_sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LABEL = {("tracker", "nominal"): "tracker + nominal bCBF", ("tracker", "ue"): "tracker + UE-bCBF",
         ("ps2", "ue"): "Phase II + UE-bCBF", ("tracker", "none"): "tracker, no filter", ("ps2", "nominal"): "Phase II + nominal bCBF"}
COLOR = {("tracker", "nominal"): "#8a5cc2", ("tracker", "ue"): "#2a78d6", ("ps2", "ue"): "#1f9e89",
         ("tracker", "none"): "#eb6834", ("ps2", "nominal"): "#c2418c"}
DIRS = {"rand": "random direction", "wall": "into the funnel wall (+x, +z)", "down": "downward (-z)"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    root = Path(a.root)
    rows = []
    for d in sorted(root.iterdir()):
        ev = d / "eval.json"
        if not ev.exists():
            continue
        policy, dmode, ddir, amp = d.name.split("__")
        meta = json.loads(ev.read_text())
        delta_d = float(meta["ue"]["delta_d"])
        for fmode, r in meta["results"].items():
            rows.append({"policy": policy, "filter": fmode, "dist_mode": dmode, "dir": ddir,
                         "amp_scale": float(amp[1:]), "amp": float(amp[1:]) * delta_d, **r})
    if not rows:
        raise SystemExit(f"no eval.json under {root}")
    (root / "sweep.json").write_text(json.dumps(rows, indent=2, default=float))
    lines = ["| direction | |d| [m/s^2] | policy + filter | unsafe | min h_cone [m] | min floor margin [m] | touched | correction | fallback |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["dir"], r["amp"], r["policy"], r["filter"])):
        lines.append(f"| {DIRS.get(r['dir'], r['dir'])} | {r['amp']:.2f} | {LABEL.get((r['policy'], r['filter']), r['policy'] + '+' + r['filter'])} "
                     f"| {r['unsafe_rate']:.0%} | {r['min_h_cone']:+.3f} | {r['min_floor_margin']:+.3f} | {r['touched_rate']:.0%} "
                     f"| {r['cil_correction_mean']:.2f} | {r['fallback_rate']:.3f} |")
    (root / "sweep.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dirs = sorted({r["dir"] for r in rows}, key=lambda s: list(DIRS).index(s) if s in DIRS else 9)
    fig, axes = plt.subplots(2, len(dirs), figsize=(6.2 * len(dirs), 7.4), squeeze=False)
    for j, dd in enumerate(dirs):
        sub = [r for r in rows if r["dir"] == dd]
        for key in sorted({(r["policy"], r["filter"]) for r in sub}):
            pts = sorted([r for r in sub if (r["policy"], r["filter"]) == key], key=lambda r: r["amp"])
            x = [r["amp"] for r in pts]
            axes[0, j].plot(x, [100 * r["unsafe_rate"] for r in pts], "o-", color=COLOR.get(key, "k"), lw=1.8,
                            label=LABEL.get(key, str(key)))
            axes[1, j].plot(x, [100 * min(r["min_h_cone"], r["min_floor_margin"]) for r in pts], "o-",
                            color=COLOR.get(key, "k"), lw=1.8, label=LABEL.get(key, str(key)))
        dd_ = float(sub[0]["amp"] / sub[0]["amp_scale"]) if sub[0]["amp_scale"] else 0.5
        for ax in axes[:, j]:
            ax.axvline(dd_, color="0.4", lw=0.9, ls="--")
            ax.grid(color="#e4e3df", lw=0.5)
            ax.set_xlabel("constant disturbance |d| [m/s^2]")
        axes[0, j].text(dd_, 102, " design bound delta_d", fontsize=8, color="0.3", va="bottom")
        axes[0, j].set_ylim(-3, 108)
        axes[0, j].set_ylabel("episodes entering the funnel / floor [%]")
        axes[0, j].set_title(f"{DIRS.get(dd, dd)}", loc="left", fontweight="bold")
        axes[1, j].axhline(0, color="#c4553f", lw=1.0)
        axes[1, j].set_ylabel("smallest barrier value over all episodes [cm]")
        axes[0, j].legend(fontsize=8)
    fig.suptitle("Nominal bCBF vs UE-bCBF beyond the design disturbance (constant push, 64 episodes per point)", x=0.01,
                 ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(a.out) if a.out else root / "sweep.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}, {root / 'sweep.md'}")


if __name__ == "__main__":
    main()
