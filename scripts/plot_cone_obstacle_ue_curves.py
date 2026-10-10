#!/usr/bin/env python
"""Training curves of the two-stage Phase II for the outside-cone task (evaluation history of each run).

    python scripts/plot_cone_obstacle_ue_curves.py --root outputs/cone_obstacle_phase2_ue \
        --out outputs/cone_obstacle_phase2_ue/cone_ue_curves.png

Stage 1 (vanilla trackers, one per critic reward scale): evaluation return and position error.
Stage 2 (PS2 through the UE-bCBF CIL): evaluation return, filter correction, time to touch, closest
approach to the funnel wall. Step 0 of stage 2 is the stage-1 tracker through the CIL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="outputs/cone_obstacle_phase2_ue")
    p.add_argument("--vanilla", default="", help="comma list of stage-1 run folders (default: cone_ue_vanilla_s*_rs*)")
    p.add_argument("--ps2", default="cone_ue_ps2_s0")
    p.add_argument("--out", default="")
    a = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(a.root)
    vruns = [root / v for v in a.vanilla.split(",") if v] if a.vanilla else sorted(root.glob("cone_ue_vanilla_s*_rs*"))
    choice = json.loads((root / "vanilla_choice.json").read_text())["tag"] if (root / "vanilla_choice.json").exists() else ""
    h2 = json.loads((root / a.ps2 / "history.json").read_text())
    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10, "axes.titleweight": "bold"})
    fig, ax = plt.subplots(1, 4, figsize=(17.5, 3.9))
    cols = ["#2a78d6", "#8a5cc2", "#1f9e89", "#c99a06"]
    for c, run in zip(cols, vruns):
        h = json.loads((run / "history.json").read_text())
        st = np.asarray(h["eval/step"]) / 1e6
        tag = run.name.split("_rs")[-1].replace("p", ".")
        lab = f"reward scale {tag}" + (" (chosen)" if run.name == choice else "")
        ax[0].plot(st, -np.asarray(h["eval/return_mean"]), color=c, lw=1.6, label=lab)
        ax[1].plot(st, np.asarray(h["eval/pos_err_mean"]), color=c, lw=1.6, label=lab)
    ax[0].set_yscale("log")
    ax[0].set_title("Stage 1 tracker (no CIL): cost = -return", loc="left")
    ax[0].set_xlabel("env steps [M]")
    ax[0].legend(fontsize=8)
    ax[1].set_title("Stage 1 tracker: mean position error [m]", loc="left")
    ax[1].set_xlabel("env steps [M]")
    ax[1].set_yscale("log")
    st2 = np.asarray(h2["eval/step"]) / 1e6
    ax[2].plot(st2, np.asarray(h2["eval/return_mean"]), "o-", color="#2a78d6", lw=1.6)
    ax[2].set_title("Stage 2 (through UE-CIL): return", loc="left")
    ax[2].set_xlabel("env steps [M]")
    ax[2].annotate("stage-1 tracker\n+ UE-CIL", (st2[0], h2["eval/return_mean"][0]), xytext=(st2[-1] * 0.25, h2["eval/return_mean"][0]),
                   fontsize=8, va="center", arrowprops={"arrowstyle": "->", "color": "0.4"})
    ax3 = ax[3]
    ax3.plot(st2, np.asarray(h2["eval/cil_correction_mean"]), "o-", color="#c4553f", lw=1.6, label="filter correction")
    ax3.set_ylabel("filter correction (normalised)", color="#c4553f")
    ax3b = ax3.twinx()
    ax3b.plot(st2, np.asarray(h2["eval/touch_time_mean"]), "s--", color="#3d3c39", lw=1.3, label="time to touch [s]")
    ax3b.set_ylabel("time to touch [s]")
    ax3.set_title("Stage 2: filter correction, time to touch", loc="left")
    unsafe = max(h2["eval/unsafe_rate"])
    ax3.set_xlabel(f"env steps [M]   (unsafe at most {unsafe:.0%}; min h_cone {min(h2['eval/min_h_cone']):+.3f} m)")
    for x in ax:
        x.grid(color="#e4e3df", lw=0.5)
    fig.tight_layout()
    out = Path(a.out) if a.out else root / "cone_ue_curves.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
