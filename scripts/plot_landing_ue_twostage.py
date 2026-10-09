#!/usr/bin/env python
"""Figure: tracker alone vs Phase II through the UE landing CIL (two-stage recipe).

Reads the ``trajectories.npz`` written by ``scripts/eval_phase2_landing_ue.py`` (same seed =
same initial states and disturbance draws in every panel) and draws, per column,

  * side view in the cone's own coordinates: horizontal distance from the pad axis r vs
    height z. The safe set is r <= r0 + tan(theta) z, z >= 0, so every point under the
    line is inside the cone - exact for any heading, unlike an x-z projection;
  * top view of the last part of the descent around the pad: the pad circle (radius r0),
    the cone's cross-section at a few heights and the touchdown points.

    JAX_ENABLE_X64=1 python scripts/eval_phase2_landing_ue.py --run outputs/landing_phase2_ue/land_ue_vanilla_s0 \
        --modes none,ue --out outputs/landing_phase2_ue/land_ue_vanilla_s0/eval
    JAX_ENABLE_X64=1 python scripts/eval_phase2_landing_ue.py --run outputs/landing_phase2_ue/land_ue_p2_warm_s0 \
        --modes ue --out outputs/landing_phase2_ue/land_ue_p2_warm_s0/eval
    python scripts/plot_landing_ue_twostage.py \
        --panel "Stage 1 tracker, no CIL=outputs/landing_phase2_ue/land_ue_vanilla_s0/eval:none" \
        --panel "Stage 1 tracker + CIL (Phase II start)=outputs/landing_phase2_ue/land_ue_vanilla_s0/eval:ue" \
        --panel "Phase II trained, through CIL=outputs/landing_phase2_ue/land_ue_p2_warm_s0/eval:ue" \
        --out landing_ue_twostage.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAFE_FILL = "#ecebe7"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d9d8d3"
SERIES = {"none": "#eb6834", "ue": "#2a78d6"}  # validated pair (categorical slots 1-2)
VIOL = "#0b0b0b"


def _geometry(eval_dir: Path):
    meta = json.loads((eval_dir / "eval.json").read_text())
    from ps2rl.cil import quadrotor_landing_ue_bcbf as uecbf
    cfg = uecbf.ue_bcbf_config_from_checkpoint(meta["ckpt"] if Path(meta["ckpt"]).exists() else str(ROOT / meta["ckpt"]))
    lc = cfg.landing
    return float(lc.cone_r0), float(np.tan(np.deg2rad(lc.cone_theta_deg))), float(lc.pad_x), float(lc.pad_y), \
        float(lc.pad_z), float(lc.dt)


def _load(eval_dir: Path, mode: str):
    d = np.load(eval_dir / "trajectories.npz")
    tr = {k[len(mode) + 1:]: d[k] for k in d.files if k.startswith(mode + "_")}
    if not tr:
        raise SystemExit(f"{eval_dir}/trajectories.npz has no mode '{mode}' (modes: "
                         f"{sorted({k.split('_')[0] for k in d.files})})")
    return tr


def _stats(tr, r0, pad):
    al = tr["alive"].astype(bool)
    h = np.where(al, tr["h"], np.inf)
    z = np.where(al, tr["z"], np.inf)
    unsafe = ((h < 0) | (z < 0)).any(0)
    last = np.maximum(al.sum(0) - 1, 0)
    fx = tr["x"][last, np.arange(al.shape[1])]
    dist = np.hypot(fx[:, 0] - pad[0], fx[:, 1] - pad[1])
    speed = np.linalg.norm(fx[:, 3:6], axis=1)
    landed = (fx[:, 2] - pad[2] < 0.10) & (dist < r0) & (speed < 0.5) & ~unsafe
    return {"n": int(al.shape[1]), "unsafe": float(unsafe.mean()), "landed": float(landed.mean()),
            "min_h": float(h.min()), "dist_mean": float(dist.mean()), "dist_max": float(dist.max()),
            "touch": fx[:, :2] - np.asarray(pad[:2]), "unsafe_mask": unsafe}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panel", action="append", required=True,
                   help='"title=EVAL_DIR:MODE" (MODE: none / ue / nominal), one per column, left to right')
    p.add_argument("--out", default="landing_ue_twostage.png")
    p.add_argument("--top_view_below", type=float, default=0.6, help="top view: the part of the descent below this height [m]")
    a = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    panels = []
    for spec in a.panel:
        title, rest = spec.split("=", 1)
        eval_dir, mode = rest.rsplit(":", 1)
        panels.append((title, Path(eval_dir), mode))
    r0, tt, px, py, pz, dt = _geometry(panels[0][1])
    pad = (px, py, pz)

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
                         "ytick.color": INK_2, "axes.titlesize": 11, "axes.titleweight": "bold"})
    nc = len(panels)
    fig, axes = plt.subplots(2, nc, figsize=(4.9 * nc, 10.0), squeeze=False,
                             gridspec_kw={"height_ratios": [1.15, 1.0]})
    fig.patch.set_facecolor("#fcfcfb")
    summary = {}
    for j, (title, eval_dir, mode) in enumerate(panels):
        tr = _load(eval_dir, mode)
        st = _stats(tr, r0, pad)
        summary[title] = {k: v for k, v in st.items() if k not in ("touch", "unsafe_mask")}
        col = SERIES.get(mode, "#2a78d6")
        al = tr["alive"].astype(bool)
        x = tr["x"]
        r = np.hypot(x[..., 0] - px, x[..., 1] - py)
        z = x[..., 2] - pz
        h = tr["h"]

        # ---- side view (r, z)
        ax = axes[0, j]
        ax.set_facecolor("#fcfcfb")
        zz = np.linspace(0.0, 2.6, 2)
        ax.fill_betweenx(zz, 0.0, r0 + tt * zz, color=SAFE_FILL, lw=0, zorder=0)
        ax.plot(r0 + tt * zz, zz, color=INK, lw=1.2, zorder=3)
        ax.axhline(0.0, color=INK, lw=1.2, zorder=3)
        ax.plot([0, r0], [0, 0], color=INK, lw=4, solid_capstyle="butt", zorder=4)
        for i in range(x.shape[1]):
            T = int(al[:, i].sum())
            ax.plot(r[:T, i], z[:T, i], color=col, lw=0.7, alpha=0.35, zorder=2)
        bad = al & ((h < 0) | (z < 0))
        if bad.any():
            ax.scatter(r[bad], z[bad], s=1.2, color=VIOL, lw=0, alpha=0.6, zorder=5)
        ax.set_xlim(0, 2.6)
        ax.set_ylim(-0.05, 2.4)
        ax.set_aspect("equal")
        ax.set_xlabel("horizontal distance from pad centre r [m]")
        if j == 0:
            ax.set_ylabel("height z [m]")
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_title(title, color=INK, loc="left")
        verdict = (f"{st['unsafe']:.0%} of episodes leave the cone/floor (deepest {-st['min_h']:.2f} m)"
                   if st["unsafe"] > 0 else f"0 of {st['n']} episodes leave the cone/floor")
        ax.text(0.02, 0.98, verdict, transform=ax.transAxes, va="top", ha="left", fontsize=9, color=INK)
        ax.text(0.08, 1.30, "inside the cone\n(safe set)", fontsize=8.5, color=INK_2, va="center")
        ax.text(1.50, 0.40, "outside the cone", fontsize=8.5, color=INK_2, va="center")

        # ---- top view around the pad
        ax = axes[1, j]
        ax.set_facecolor("#fcfcfb")
        lim = 1.0
        for zc in (0.25, 0.5):
            rc = r0 + tt * zc
            ax.add_patch(Circle((0, 0), rc, fill=False, ls=(0, (3, 3)), lw=0.8, color=INK_2, zorder=1))
            ax.text(rc * np.cos(-0.75) + 0.03, rc * np.sin(-0.75) - 0.03, f"cone at z = {zc:g} m", ha="left",
                    va="top", fontsize=7.5, color=INK_2, bbox={"fc": "#fcfcfb", "ec": "none", "pad": 0.5})
        ax.add_patch(Circle((0, 0), r0, color=SAFE_FILL, zorder=0))
        ax.add_patch(Circle((0, 0), r0, fill=False, lw=1.4, color=INK, zorder=3))
        low = al & (z < a.top_view_below)
        for i in range(x.shape[1]):
            m = low[:, i]
            if m.any():
                ax.plot(x[m, i, 0] - px, x[m, i, 1] - py, color=col, lw=0.7, alpha=0.35, zorder=2)
        tp = st["touch"]
        ok = ~st["unsafe_mask"]
        ax.scatter(tp[ok, 0], tp[ok, 1], s=14, color=col, edgecolor="#fcfcfb", lw=0.6, zorder=4)
        if (~ok).any():
            ax.scatter(tp[~ok, 0], tp[~ok, 1], s=16, marker="x", color=VIOL, lw=1.0, zorder=5)
        ax.set_xlim(-lim, lim * 0.85)
        ax.set_ylim(-0.8, 0.8)
        ax.set_aspect("equal")
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel("x [m]")
        if j == 0:
            ax.set_ylabel("y [m]")
        inside = float((np.hypot(tp[:, 0], tp[:, 1]) < r0).mean())
        ax.set_title(f"touchdown inside the pad circle {inside:.0%}, safe landings {st['landed']:.0%}", color=INK,
                     loc="left", fontsize=10)
        ax.text(0.02, 0.02, f"distance from centre: mean {st['dist_mean']:.2f} m, max {st['dist_max']:.2f} m",
                transform=ax.transAxes, fontsize=8.5, color=INK, va="bottom",
                bbox={"fc": "#fcfcfb", "ec": "none", "pad": 1.0})

    handles = [Line2D([], [], color=INK, lw=4, solid_capstyle="butt", label=f"pad (r0 = {r0:g} m)"),
               Line2D([], [], color=SERIES["none"], lw=1.5, label="trajectory, no filter"),
               Line2D([], [], color=SERIES["ue"], lw=1.5, label="trajectory, through the UE-bCBF CIL"),
               Line2D([], [], color=VIOL, marker="o", ls="", ms=3, label="outside the cone / below the pad"),
               Line2D([], [], color=INK_2, marker="o", ls="", ms=5, label="touchdown point (x = after a violation)")]
    used = {m for _, _, m in panels}
    handles = [h for h in handles if not (h.get_label().startswith("trajectory, no") and "none" not in used)
               and not (h.get_label().startswith("trajectory, through") and "ue" not in used)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    first = _load(panels[0][1], panels[0][2])
    fig.suptitle(f"Landing under a disturbance (|d| <= 0.5 m/s^2, 0.05 Hz): same {first['alive'].shape[1]} initial states "
                 f"and disturbance draws in every column", fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97), h_pad=2.5)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    for k, v in summary.items():
        print(f"{k}: unsafe {v['unsafe']:.3f} (min h {v['min_h']:+.3f}) landed {v['landed']:.3f} "
              f"touchdown dist mean {v['dist_mean']:.3f} max {v['dist_max']:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
