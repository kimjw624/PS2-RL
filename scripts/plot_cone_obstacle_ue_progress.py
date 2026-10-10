#!/usr/bin/env python
"""Figure (and GIFs): progress of the cone-obstacle task through the two-stage Phase II.

Reads the ``trajectories.npz`` written by ``scripts/eval_phase2_cone_obstacle_ue.py`` (same seed =
same initial states and disturbance draws in every panel) and draws, per column,

  * side view in the funnel's own coordinates: horizontal distance from the funnel axis r vs
    height z. The keep-out set is {z > tip + (sqrt(r^2 + eps^2) - eps) / tan(theta)} (inside the
    funnel) and {z < floor}; it is exact for any heading, unlike an x-z projection. The target
    cylinder (radius R, top at the funnel tip) stands on the axis;
  * top view around the target: the low part of every path (z below ``--top_view_below``), the
    target circle, the funnel's cross-section at a few heights, and the first-touch points
    (x = episode ended without touching: its last point).

    python scripts/plot_cone_obstacle_ue_progress.py \
        --panel "Stage 1 tracker, no CIL=outputs/cone_obstacle_phase2_ue/cone_ue_vanilla_s0/eval:none" \
        --panel "Stage 1 tracker + UE-CIL (Phase II start)=outputs/cone_obstacle_phase2_ue/cone_ue_vanilla_s0/eval:ue" \
        --panel "Phase II trained through UE-CIL=outputs/cone_obstacle_phase2_ue/cone_ue_ps2_s0/eval:ue" \
        --out outputs/cone_obstacle_phase2_ue/cone_ue_progress.png --gif --gif3d

``--gif``: animated side view, one column per panel, a few episodes (same draws).
``--gif3d``: multi-view animation, one row per panel in ``--gif_rows`` (default: first and last
column): 3-D view, exact side view (r, z), top view with the funnel's cross-section at each drone's
height, and the barrier traces h_cone(t) / floor margin(t); a drone in the keep-out set is marked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

SURF = "#fcfcfb"
KEEP_OUT = "#f3d9d3"
KEEP_OUT_EDGE = "#c4553f"
TARGET = "#3d3c39"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d9d8d3"
SERIES = {"none": "#eb6834", "ue": "#2a78d6", "nominal": "#8a5cc2"}
VIOL = "#0b0b0b"
REF = "#0b0b0b"


class Geometry:
    def __init__(self, task: dict):
        from cone_obstacle_common import task_config
        sc = task_config(task)
        self.tip, self.tt, self.eps = float(sc.tip_z), float(sc.tan_theta), float(sc.cone_eps)
        self.ax_x, self.ax_y = float(sc.axis_x), float(sc.axis_y)
        self.floor, self.ground = float(sc.floor_z), float(sc.ground_z)
        self.R, self.top = float(sc.target_radius), float(sc.cylinder_top)
        self.tx, self.ty = (float(v) for v in sc.target_center)
        self.hover = np.asarray(sc.hover_points[0], float)
        self.dt = float(sc.dt)
        self.scale = np.asarray(sc.action_scale, float)

    def surface_z(self, r):
        return self.tip + (np.sqrt(np.asarray(r) ** 2 + self.eps ** 2) - self.eps) / self.tt

    def radius_at(self, z):
        a = self.tt * (z - self.tip) + self.eps
        return np.sqrt(np.maximum(a * a - self.eps ** 2, 0.0))

    def r_of(self, x):
        return np.hypot(x[..., 0] - self.ax_x, x[..., 1] - self.ax_y)


def load_panel(eval_dir: Path, mode: str):
    d = np.load(eval_dir / "trajectories.npz")
    pre = mode + "__"
    tr = {k[len(pre):]: d[k] for k in d.files if k.startswith(pre)}
    if not tr:
        modes = sorted({k.split("__")[0] for k in d.files})
        raise SystemExit(f"{eval_dir}/trajectories.npz has no mode '{mode}' (modes: {modes})")
    meta = json.loads((eval_dir / "eval.json").read_text())
    return tr, meta


def stats(tr, geo: Geometry):
    al = tr["alive"].astype(bool)
    h = np.where(al, tr["h"], np.inf)
    z = np.where(al, tr["z"], np.inf)
    unsafe = ((h < 0) | (z < 0)).any(0)
    touch = (tr["touch"] > 0.5) & al
    touched = touch.any(0)
    k_touch = np.argmax(touch, axis=0)
    last = np.maximum(al.sum(0) - 1, 0)
    idx = np.where(touched, k_touch, last)
    pts = tr["x"][idx, np.arange(al.shape[1])]
    alf = al.astype(float)
    corr = np.linalg.norm((tr["u"] - tr["raw"]) / geo.scale, axis=-1)          # how much the filter changed the command
    return {"corr": float((corr * alf).sum() / max(alf.sum(), 1.0)), "return": float((tr["rew"] * alf).sum(0).mean()),
            "n": int(al.shape[1]), "unsafe": float(unsafe.mean()), "touched": float(touched.mean()),
            "touched_safe": float((touched & ~unsafe).mean()), "min_h": float(h.min()), "min_z": float(z.min()),
            "t_touch": float(((k_touch + 1) * geo.dt)[touched].mean()) if touched.any() else float("nan"),
            "pts": pts, "touched_mask": touched, "unsafe_mask": unsafe}


def draw_side(ax, geo: Geometry, r_max: float, z_max: float):
    rr = np.linspace(0.0, r_max, 200)
    zs = geo.surface_z(rr)
    ax.fill_between(rr, np.minimum(zs, z_max), z_max, color=KEEP_OUT, lw=0, zorder=0)
    ax.plot(rr, zs, color=KEEP_OUT_EDGE, lw=1.3, zorder=3)
    ax.fill_between([0, r_max], geo.ground, geo.floor, color=KEEP_OUT, lw=0, zorder=0)
    ax.axhline(geo.floor, color=KEEP_OUT_EDGE, lw=1.0, ls=(0, (4, 2)), zorder=3)
    ax.axhline(geo.ground, color=INK, lw=1.2, zorder=3)
    ax.fill_between([0, geo.R], geo.ground, geo.top, color=TARGET, lw=0, zorder=4)
    ax.set_xlim(0, r_max)
    ax.set_ylim(geo.ground - 0.05, z_max)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panel", action="append", required=True,
                   help='"title=EVAL_DIR:MODE" (MODE: none / ue / nominal), one per column, left to right')
    p.add_argument("--out", default="cone_ue_progress.png")
    p.add_argument("--max_paths", type=int, default=128, help="paths drawn per panel")
    p.add_argument("--top_view_below", type=float, default=1.6, help="top view: the part of the path below this height [m]")
    p.add_argument("--r_max", type=float, default=3.6)
    p.add_argument("--z_max", type=float, default=3.0)
    p.add_argument("--gif", action="store_true", help="animated side view of every panel")
    p.add_argument("--gif3d", action="store_true",
                   help="multi-view animation (3-D, side (r, z), top, barrier traces), one row per --gif_rows panel")
    p.add_argument("--gif_rows", default="first,last",
                   help="panels in the multi-view GIF: comma list of panel indices (0-based) or first/last")
    p.add_argument("--gif_elev", type=float, default=14.0, help="3-D view elevation [deg]")
    p.add_argument("--gif_azim", type=float, default=-72.0, help="3-D view azimuth [deg]")
    p.add_argument("--gif_dpi", type=int, default=72)
    p.add_argument("--gif_views_stride", type=int, default=4, help="multi-view animation: simulation steps per frame")
    p.add_argument("--mp4", action="store_true", help="also write the multi-view animation as .mp4 (needs ffmpeg)")
    p.add_argument("--gif_episodes", type=int, default=6)
    p.add_argument("--gif_stride", type=int, default=3, help="simulation steps per frame")
    p.add_argument("--gif_fps", type=int, default=20)
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
        tr, meta = load_panel(Path(eval_dir), mode)
        panels.append((title, tr, meta, mode))
    geo = Geometry(panels[0][2]["task"])
    ref = np.asarray(panels[0][2]["reference_states"])
    ue = panels[0][2]["ue"]

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
                         "ytick.color": INK_2, "axes.titlesize": 11, "axes.titleweight": "bold"})
    nc = len(panels)
    fig, axes = plt.subplots(2, nc, figsize=(5.4 * nc, 9.0), squeeze=False, gridspec_kw={"height_ratios": [1.0, 0.85]})
    fig.patch.set_facecolor(SURF)
    summary = {}
    for j, (title, tr, meta, mode) in enumerate(panels):
        st = stats(tr, geo)
        summary[title] = {k: v for k, v in st.items() if k not in ("pts", "touched_mask", "unsafe_mask")}
        col = SERIES.get(mode, "#2a78d6")
        al = tr["alive"].astype(bool)
        x = tr["x"]
        r = geo.r_of(x)
        z = x[..., 2]
        h = tr["h"]
        n_draw = min(x.shape[1], a.max_paths)

        # ---- side view (r, z)
        ax = axes[0, j]
        ax.set_facecolor(SURF)
        draw_side(ax, geo, a.r_max, a.z_max)
        ax.plot(geo.r_of(ref), ref[:, 2], color=REF, lw=1.3, ls=(0, (5, 3)), zorder=6)
        for i in range(n_draw):
            T = int(al[:, i].sum())
            ax.plot(r[:T, i], z[:T, i], color=col, lw=0.7, alpha=0.35, zorder=2)
        bad = al & ((h < 0) | (tr["z"] < 0))
        if bad.any():
            ax.scatter(r[bad], z[bad], s=1.2, color=VIOL, lw=0, alpha=0.6, zorder=5)
        ax.set_xlabel("horizontal distance from the funnel axis r [m]")
        if j == 0:
            ax.set_ylabel("height z [m]")
        ax.set_title(title, color=INK, loc="left")
        if st["unsafe"] > 0:
            verdict = f"{st['unsafe']:.0%} of episodes enter the funnel / floor (deepest {max(-st['min_h'], -st['min_z']):.2f} m)"
        else:
            verdict = f"0 of {st['n']} episodes enter the funnel / floor"
        verdict += f"\nreturn {st['return']:.0f}" + (f", filter correction {st['corr']:.2f}" if mode != "none" else "")
        ax.text(0.02, 0.98, verdict, transform=ax.transAxes, va="top", ha="left", fontsize=9, color=INK,
                bbox={"fc": SURF, "ec": "none", "pad": 1.0}, zorder=8)
        ax.text(0.08, a.z_max - 0.55, "inside the funnel\n(keep-out)", fontsize=8.5, color=KEEP_OUT_EDGE, va="top",
                bbox={"fc": KEEP_OUT, "ec": "none", "pad": 0.5}, zorder=7)
        ax.text(a.r_max - 0.1, geo.floor - 0.22, "below the floor (keep-out)", fontsize=8, color=KEEP_OUT_EDGE, ha="right",
                va="center")
        ax.text(geo.R + 0.06, 0.5 * (geo.floor + geo.top), "target", fontsize=8.5, color=TARGET, va="center")

        # ---- top view around the target
        ax = axes[1, j]
        ax.set_facecolor(SURF)
        for zc in (geo.tip + 0.3, geo.tip + 0.8):
            rc = geo.radius_at(zc)
            ax.add_patch(Circle((geo.ax_x, geo.ax_y), rc, fill=False, ls=(0, (3, 3)), lw=0.8, color=KEEP_OUT_EDGE, zorder=1))
            ax.text(geo.ax_x + rc * np.cos(-0.8) + 0.03, geo.ax_y + rc * np.sin(-0.8) - 0.03, f"funnel at z = {zc:.1f} m",
                    ha="left", va="top", fontsize=7.5, color=KEEP_OUT_EDGE, bbox={"fc": SURF, "ec": "none", "pad": 0.5})
        ax.add_patch(Circle((geo.tx, geo.ty), geo.R, color=TARGET, alpha=0.85, zorder=3))
        low = al & (z < a.top_view_below)
        for i in range(n_draw):
            m = low[:, i]
            if m.any():
                ax.plot(x[m, i, 0], x[m, i, 1], color=col, lw=0.7, alpha=0.35, zorder=2)
        pts, tm = st["pts"], st["touched_mask"]
        ax.scatter(pts[tm, 0], pts[tm, 1], s=14, color=col, edgecolor=SURF, lw=0.6, zorder=4)
        if (~tm).any():
            ax.scatter(pts[~tm, 0], pts[~tm, 1], s=16, marker="x", color=VIOL, lw=1.0, zorder=5)
        ax.set_xlim(geo.tx - 1.7, geo.tx + 0.9)
        ax.set_ylim(geo.ty - 0.9, geo.ty + 0.9)
        ax.set_aspect("equal")
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel("x [m]")
        if j == 0:
            ax.set_ylabel("y [m]")
        tt = f", at {st['t_touch']:.1f} s" if st["touched"] > 0 else ""
        ax.set_title(f"touched {st['touched']:.0%}{tt}; safe and touched {st['touched_safe']:.0%}", color=INK,
                     loc="left", fontsize=10)

    handles = [Line2D([], [], color=REF, lw=1.3, ls=(0, (5, 3)), label="reference (runs into the funnel)"),
               Line2D([], [], color=SERIES["none"], lw=1.5, label="path, no filter"),
               Line2D([], [], color=SERIES["ue"], lw=1.5, label="path, through the UE-bCBF CIL"),
               Line2D([], [], color=SERIES["nominal"], lw=1.5, label="path, through the nominal BCBF"),
               Line2D([], [], color=VIOL, marker="o", ls="", ms=3, label="inside the funnel / below the floor"),
               Line2D([], [], color=TARGET, marker="s", ls="", ms=7, label=f"target cylinder (R = {geo.R:g} m)"),
               Line2D([], [], color=INK_2, marker="o", ls="", ms=5, label="first touch (x = never touched)")]
    used = {m for *_, m in panels}
    keep = {"path, no filter": "none", "path, through the UE-bCBF CIL": "ue", "path, through the nominal BCBF": "nominal"}
    handles = [hd for hd in handles if keep.get(hd.get_label(), None) in (None, *used)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    n_ep = panels[0][1]["alive"].shape[1]
    fig.suptitle(f"Outside-cone task under a disturbance (|d| <= {ue['delta_d']:g} m/s^2, |d_dot| <= {ue['delta_v']:.3f} m/s^3, "
                 f"random direction / amplitude / frequency): same {n_ep} initial states and disturbance draws in every column",
                 fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97), h_pad=2.0)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    for k, v in summary.items():
        print(f"{k}: unsafe {v['unsafe']:.3f} (min h {v['min_h']:+.3f}, min floor {v['min_z']:+.3f}) touched {v['touched']:.3f} "
              f"(safe {v['touched_safe']:.3f}) t_touch {v['t_touch']:.2f} return {v['return']:.1f} filter correction {v['corr']:.3f}")
    print(f"wrote {out}")
    if a.gif:
        side_gif(panels, geo, ref, a, out.with_name(out.stem + "_side.gif"))
    if a.gif3d:
        sel = []
        for tok in [t.strip() for t in a.gif_rows.split(",") if t.strip()]:
            idx = {"first": 0, "last": len(panels) - 1}.get(tok, None)
            idx = int(tok) % len(panels) if idx is None else idx
            if idx not in sel:
                sel.append(idx)
        gif_views([panels[i] for i in sel], geo, ref, a, out.with_name(out.stem + "_views.gif"))


def _episodes(panels, n):
    """Episodes to animate: spread over the bank (same indices in every panel)."""
    n_ep = panels[0][1]["alive"].shape[1]
    return np.linspace(0, n_ep - 1, min(n, n_ep)).astype(int)


def side_gif(panels, geo, ref, a, path):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    eps = _episodes(panels, a.gif_episodes)
    n_t = max(int(p[1]["alive"].shape[0]) for p in panels)
    nc = len(panels)
    fig, axes = plt.subplots(1, nc, figsize=(5.0 * nc, 5.0), squeeze=False)
    fig.patch.set_facecolor(SURF)
    arts = []
    for j, (title, tr, meta, mode) in enumerate(panels):
        ax = axes[0, j]
        ax.set_facecolor(SURF)
        draw_side(ax, geo, a.r_max, a.z_max)
        ax.plot(geo.r_of(ref), ref[:, 2], color=REF, lw=1.0, ls=(0, (5, 3)), zorder=6)
        ax.set_title(title, color=INK, loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel("r [m]")
        if j == 0:
            ax.set_ylabel("z [m]")
        col = SERIES.get(mode, "#2a78d6")
        lines = [ax.plot([], [], color=col, lw=1.2, alpha=0.8, zorder=5)[0] for _ in eps]
        dots = ax.scatter(np.zeros(len(eps)), np.zeros(len(eps)), s=22, color=col, edgecolor=INK, lw=0.6, zorder=7)
        ghost = ax.scatter([0.0], [0.0], s=26, marker="o", facecolor="none", edgecolor=REF, lw=1.0, zorder=7)
        arts.append((tr, lines, dots, ghost))
    clock = fig.text(0.01, 0.97, "", fontsize=10, color=INK, va="top")
    r_ref = geo.r_of(ref)

    def frame(f):
        k = min(f * a.gif_stride, n_t - 1)
        clock.set_text(f"t = {k * geo.dt:4.1f} s   (open circle: reference)")
        for tr, lines, dots, ghost in arts:
            al = tr["alive"].astype(bool)
            xs = tr["x"]
            pos = []
            for line, i in zip(lines, eps):
                T = int(min(k + 1, al[:, i].sum()))
                rr, zz = geo.r_of(xs[:T, i]), xs[:T, i, 2]
                line.set_data(rr, zz)
                pos.append((rr[-1], zz[-1]) if T > 0 else (np.nan, np.nan))
            dots.set_offsets(np.asarray(pos))
            kr = min(k, len(ref) - 1)
            ghost.set_offsets([[r_ref[kr], ref[kr, 2]]])
        return []

    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    anim = FuncAnimation(fig, frame, frames=n_t // a.gif_stride + 1, blit=False)
    anim.save(path, writer=PillowWriter(fps=a.gif_fps), dpi=90)
    plt.close(fig)
    print(f"wrote {path}")


def _quat_rot(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


EP_COLORS = ["#2a78d6", "#1f9e89", "#8a5cc2", "#c99a06", "#c2418c", "#4d7f2a", "#3d6fb0", "#7a5a2e"]


def gif_views(panels, geo, ref, a, path):
    """Multi-view animation: one row per selected panel, four synchronised views per row.

    1. 3-D view (context). Each drone carries the funnel's cross-section ring at its own height: a drone
       inside its ring is inside the funnel. Floor shadow and a drop line give depth.
    2. Side view in the funnel's coordinates (r, z): exact - the wall is a line, a drone above it is inside.
    3. Top view: path in x-y and, per drone, the funnel's cross-section at the drone's current height
       (dashed, same colour); inside its own circle = inside the funnel.
    4. Barrier traces h_cone(t) (solid) and floor margin (dashed); below zero = in the keep-out set.
    Episodes have one colour each across the four views. A drone in the keep-out set is marked with a black x.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    eps = _episodes(panels, a.gif_episodes)
    cols = [EP_COLORS[m % len(EP_COLORS)] for m in range(len(eps))]
    n_t = max(int(p_[1]["alive"].shape[0]) for p_ in panels)
    nr = len(panels)
    fig = plt.figure(figsize=(19.0, 4.7 * nr + 0.5))
    fig.patch.set_facecolor(SURF)
    gs = fig.add_gridspec(nr, 4, width_ratios=[1.55, 1.0, 1.05, 0.9], left=0.0, right=0.99, top=1 - 0.55 / fig.get_figheight(),
                          bottom=0.07, wspace=0.24, hspace=0.25)
    half_y = 1.6
    lo = np.array([geo.ax_x - 3.6, geo.ax_y - half_y, geo.ground])
    hi = np.array([geo.ax_x + 1.4, geo.ax_y + half_y, a.z_max])
    z_cone = min(a.z_max, float(geo.surface_z(half_y)))
    th = np.linspace(0, 2 * np.pi, 60)
    t_axis = np.arange(n_t) * geo.dt
    r_ref = geo.r_of(ref)
    rows = []
    for row, (title, tr, meta, mode) in enumerate(panels):
        al = tr["alive"].astype(bool)
        xs = tr["x"]
        ax3 = fig.add_subplot(gs[row, 0], projection="3d")
        # ---- side view (r, z)
        axs = fig.add_subplot(gs[row, 1])
        axs.set_facecolor(SURF)
        draw_side(axs, geo, a.r_max, a.z_max)
        axs.plot(r_ref, ref[:, 2], color=REF, lw=1.0, ls=(0, (5, 3)), zorder=6)
        axs.set_xlabel("distance from the funnel axis r [m]", fontsize=8.5)
        axs.set_ylabel("z [m]", fontsize=8.5)
        axs.set_title("side view (r, z): above the wall = in the funnel", fontsize=9, loc="left")
        s_lines = [axs.plot([], [], color=c, lw=1.3, zorder=5)[0] for c in cols]
        s_dots = axs.scatter(np.zeros(len(eps)), np.zeros(len(eps)), s=30, c=cols, edgecolor=INK, lw=0.6, zorder=8)
        s_bad = axs.scatter([], [], s=60, marker="x", color=VIOL, lw=1.8, zorder=9)
        s_ghost = axs.scatter([0.0], [0.0], s=30, facecolor="none", edgecolor=REF, lw=1.0, zorder=8)
        s_txt = axs.text(0.98, 0.04, "", transform=axs.transAxes, ha="right", va="bottom", fontsize=9, fontweight="bold",
                         color=KEEP_OUT_EDGE, zorder=10)
        # ---- top view
        axt = fig.add_subplot(gs[row, 2])
        axt.set_facecolor(SURF)
        axt.add_patch(Circle((geo.tx, geo.ty), geo.R, color=TARGET, alpha=0.85, zorder=3))
        for zc in (geo.tip + 0.5, geo.tip + 1.0):
            axt.add_patch(Circle((geo.ax_x, geo.ax_y), geo.radius_at(zc), fill=False, lw=0.6, color=KEEP_OUT_EDGE, alpha=0.35,
                                 zorder=1))
        axt.plot(ref[:, 0], ref[:, 1], color=REF, lw=1.0, ls=(0, (5, 3)), zorder=2)
        t_lines = [axt.plot([], [], color=c, lw=1.2, zorder=4)[0] for c in cols]
        t_rings = [axt.plot([], [], color=c, lw=1.0, ls=(0, (3, 2)), zorder=4)[0] for c in cols]
        t_dots = axt.scatter(np.zeros(len(eps)), np.zeros(len(eps)), s=30, c=cols, edgecolor=INK, lw=0.6, zorder=8)
        t_bad = axt.scatter([], [], s=60, marker="x", color=VIOL, lw=1.8, zorder=9)
        axt.set_xlim(geo.ax_x - 3.4, geo.ax_x + 1.0)
        axt.set_ylim(geo.ax_y - 1.2, geo.ax_y + 1.2)
        axt.set_aspect("equal")
        axt.grid(color=GRID, lw=0.5)
        axt.set_xlabel("x [m]", fontsize=8.5)
        axt.set_ylabel("y [m]", fontsize=8.5)
        axt.set_title("top view: ring = funnel at the drone's height", fontsize=9, loc="left")
        # ---- barrier traces
        axh = fig.add_subplot(gs[row, 3])
        axh.set_facecolor(SURF)
        axh.axhspan(-5, 0, color=KEEP_OUT, lw=0, zorder=0)
        axh.axhline(0, color=KEEP_OUT_EDGE, lw=1.0, zorder=1)
        h_lines = [axh.plot([], [], color=c, lw=1.2, zorder=4)[0] for c in cols]
        z_lines = [axh.plot([], [], color=c, lw=0.9, ls=(0, (3, 2)), alpha=0.8, zorder=3)[0] for c in cols]
        cursor = axh.axvline(0, color=INK_2, lw=0.8, zorder=2)
        axh.set_xlim(0, t_axis[-1])
        h_ok = np.where(al[:, eps], tr["h"][:, eps], np.nan)
        z_ok = np.where(al[:, eps], tr["z"][:, eps], np.nan)
        lo_h = min(-0.3, float(np.nanmin([np.nanmin(h_ok), np.nanmin(z_ok)])) - 0.05)
        axh.set_ylim(max(lo_h, -1.5), 2.0)
        axh.grid(color=GRID, lw=0.5)
        axh.set_xlabel("t [s]", fontsize=8.5)
        axh.set_ylabel("barrier value [m]", fontsize=8.5)
        axh.set_title("barriers (< 0 = keep-out)", fontsize=9, loc="left")
        axh.legend(handles=[Line2D([], [], color=INK_2, lw=1.2, label="h_cone"),
                            Line2D([], [], color=INK_2, lw=0.9, ls=(0, (3, 2)), label="floor margin")],
                   fontsize=7.5, loc="upper right")
        fig.text(0.005, gs[row, 0].get_position(fig).y1 + 0.004, title, fontsize=11, fontweight="bold", color=INK,
                 ha="left", va="bottom")
        rows.append(dict(tr=tr, al=al, xs=xs, ax3=ax3, s_lines=s_lines, s_dots=s_dots, s_bad=s_bad, s_ghost=s_ghost,
                         s_txt=s_txt, t_lines=t_lines, t_rings=t_rings, t_dots=t_dots, t_bad=t_bad, h_lines=h_lines,
                         z_lines=z_lines, cursor=cursor, title=title))
    clock = fig.text(0.5, 1 - 0.12 / fig.get_figheight(), "", fontsize=11, color=INK, ha="center", va="top")

    def draw_3d(ax, R, k):
        ax.cla()
        zz = np.linspace(geo.tip, z_cone, 12)
        tt, ZZ = np.meshgrid(th, zz)
        rr = geo.radius_at(ZZ)
        ax.plot_surface(geo.ax_x + rr * np.cos(tt), geo.ax_y + rr * np.sin(tt), ZZ, color=KEEP_OUT_EDGE, alpha=0.10, lw=0,
                        shade=False)
        for zc in np.linspace(geo.tip + 0.25, z_cone, 4):        # rings and generators for depth cues
            rc = geo.radius_at(zc)
            ax.plot(geo.ax_x + rc * np.cos(th), geo.ax_y + rc * np.sin(th), np.full_like(th, zc), color=KEEP_OUT_EDGE, lw=0.5,
                    alpha=0.6)
        for ang in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            ax.plot(geo.ax_x + geo.radius_at(zz) * np.cos(ang), geo.ax_y + geo.radius_at(zz) * np.sin(ang), zz,
                    color=KEEP_OUT_EDGE, lw=0.4, alpha=0.45)
        zc_ = np.linspace(geo.ground, geo.top, 5)
        T2, Z2 = np.meshgrid(th, zc_)
        ax.plot_surface(geo.tx + geo.R * np.cos(T2), geo.ty + geo.R * np.sin(T2), Z2, color=TARGET, alpha=0.9, lw=0, shade=False)
        X, Y = np.meshgrid([lo[0], hi[0]], [lo[1], hi[1]])
        ax.plot_surface(X, Y, np.full_like(X, geo.floor), color=KEEP_OUT, alpha=0.3, lw=0, shade=False)
        ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], color=REF, lw=0.9, ls=(0, (5, 3)))
        for c, i in zip(cols, eps):
            T = int(min(k + 1, R["al"][:, i].sum()))
            if T <= 0:
                continue
            pth = R["xs"][:T, i, 0:3]
            ax.plot(*pth.T, color=c, lw=1.6)
            ax.plot(pth[:, 0], pth[:, 1], np.full(T, geo.floor), color=c, lw=0.7, alpha=0.4)      # floor shadow
            p_ = pth[-1]
            ax.plot([p_[0], p_[0]], [p_[1], p_[1]], [geo.floor, p_[2]], color=c, lw=0.6, alpha=0.6)  # drop line
            if p_[2] > geo.tip:                                                                     # funnel ring at its height
                rc = geo.radius_at(p_[2])
                ax.plot(geo.ax_x + rc * np.cos(th), geo.ax_y + rc * np.sin(th), np.full_like(th, p_[2]), color=c, lw=1.0,
                        ls=(0, (3, 2)))
            rm = _quat_rot(R["xs"][T - 1, i, 6:10])
            arms = 0.2 * np.array([[1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0]]) / np.sqrt(2)
            tips = p_ + arms @ rm.T
            for u, v in ((0, 1), (2, 3)):
                ax.plot(*np.stack([tips[u], tips[v]]).T, color=INK, lw=1.4)
            bad = R["tr"]["h"][T - 1, i] < 0 or R["tr"]["z"][T - 1, i] < 0
            if bad:
                ax.scatter(*p_, color=VIOL, s=40, marker="x", linewidths=1.8, depthshade=False)
            else:
                ax.scatter(*p_, color=c, s=26, marker="o", edgecolors=INK, depthshade=False)
        kr = min(k, len(ref) - 1)
        ax.scatter(*ref[kr, 0:3], s=26, facecolor="none", edgecolor=REF, depthshade=False)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo, zoom=1.3)
        ax.tick_params(labelsize=6.5, pad=-1)
        ax.set_xlabel("x", fontsize=8, labelpad=-4)
        ax.set_ylabel("y", fontsize=8, labelpad=-4)
        ax.set_zlabel("z", fontsize=8, labelpad=-6)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor((1, 1, 1, 0))
            pane.set_edgecolor(GRID)
        ax.view_init(elev=float(a.gif_elev), azim=float(a.gif_azim))

    stride = max(int(a.gif_views_stride), 1)
    fps = max(int(round(a.gif_fps * a.gif_stride / stride)), 1)      # same playback speed as the side-view GIF

    def frame(f):
        k = min(f * stride, n_t - 1)
        clock.set_text(f"t = {k * geo.dt:4.1f} s   (open circle: reference point; x: drone in the keep-out set)")
        for R in rows:
            draw_3d(R["ax3"], R, k)
            pos_s, pos_t, bad_s, bad_t = [], [], [], []
            n_bad = 0
            for m, i in enumerate(eps):
                T = int(min(k + 1, R["al"][:, i].sum()))
                if T <= 0:
                    pos_s.append((np.nan, np.nan))
                    pos_t.append((np.nan, np.nan))
                    continue
                x_ = R["xs"][:T, i]
                rr = geo.r_of(x_)
                R["s_lines"][m].set_data(rr, x_[:, 2])
                R["t_lines"][m].set_data(x_[:, 0], x_[:, 1])
                pos_s.append((rr[-1], x_[-1, 2]))
                pos_t.append((x_[-1, 0], x_[-1, 1]))
                rc = geo.radius_at(x_[-1, 2]) if x_[-1, 2] > geo.tip else 0.0
                R["t_rings"][m].set_data(geo.ax_x + rc * np.cos(th), geo.ax_y + rc * np.sin(th))
                R["h_lines"][m].set_data(t_axis[:T], R["tr"]["h"][:T, i])
                R["z_lines"][m].set_data(t_axis[:T], R["tr"]["z"][:T, i])
                if R["tr"]["h"][T - 1, i] < 0 or R["tr"]["z"][T - 1, i] < 0:
                    n_bad += 1
                    bad_s.append(pos_s[-1])
                    bad_t.append(pos_t[-1])
            R["s_dots"].set_offsets(np.asarray(pos_s))
            R["t_dots"].set_offsets(np.asarray(pos_t))
            R["s_bad"].set_offsets(np.asarray(bad_s) if bad_s else np.empty((0, 2)))
            R["t_bad"].set_offsets(np.asarray(bad_t) if bad_t else np.empty((0, 2)))
            kr = min(k, len(ref) - 1)
            R["s_ghost"].set_offsets([[r_ref[kr], ref[kr, 2]]])
            R["s_txt"].set_text(f"{n_bad} of {len(eps)} drones in the keep-out set now" if n_bad else "")
            R["cursor"].set_xdata([k * geo.dt, k * geo.dt])
        return []

    anim = FuncAnimation(fig, frame, frames=n_t // stride + 1, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=int(a.gif_dpi))
    print(f"wrote {path}")
    if a.mp4:  # smaller and smoother for slides; needs ffmpeg
        from matplotlib.animation import FFMpegWriter
        if FFMpegWriter.isAvailable():
            mp4 = Path(path).with_suffix(".mp4")
            anim.save(mp4, writer=FFMpegWriter(fps=fps, bitrate=2400), dpi=int(a.gif_dpi * 1.5))
            print(f"wrote {mp4}")
        else:
            print("[gif] ffmpeg not found: no .mp4 written")
    plt.close(fig)


if __name__ == "__main__":
    main()
