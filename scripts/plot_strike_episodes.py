#!/usr/bin/env python
"""Specific strike episodes under several methods, side by side: one panel per (run, mode), the same scene in every panel.
The runs are scripts/eval_strike_filter.py replays with --only_episodes <the same list> --save_traj 1 (each writes
strike_filter_traj.npz).

    python scripts/plot_strike_episodes.py --out outputs/strike_episodes \
        --panel "outputs/replay/ps2_final:none:PS2, no filter" --panel "outputs/replay/ps2_final:learned:PS2 + ours" \
        --panel "outputs/replay/pursuit_final:learned:Pursuit + ours"

Writes to --out, per episode: episode<i>.png (+ .pdf with --pdf 1) with the full paths, episode<i>.gif (--gif 1) animated;
and episodes.md (outcome of every panel in every episode).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CHASER, TARGET, OBS_FILL, OBS_EDGE = "#1baf7a", "#eb6834", "#c9c7c0", "#6b6a65"
PALETTE = ["#2a78d6", "#c4553f", "#8a5cc2", "#1b8a6b", "#b07d12", "#d1477a", "#4f6bed"]
NONE_C = "#3d3c39"
SHORT = {"strike": "strike", "crash_obstacle": "hits obstacle", "crash_floor": "hits floor", "crash_chaser": "hit by chaser",
         "out_of_arena": "leaves arena", "timeout": "time up"}


def outline(o, n=90):
    cx, cy, bx, by, yaw, rr = o[:6]
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    if bx == 0 and by == 0:
        return np.stack([cx + rr * np.cos(th), cy + rr * np.sin(th)], 1)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[-bx, -by], [bx, -by], [bx, by], [-bx, by]]) @ np.array([[c, s], [-s, c]]) + [cx, cy]


class Panel:
    def __init__(self, spec: str, color: str):
        d, mode, label = spec.split(":", 2)
        z = np.load(Path(d) / "strike_filter_traj.npz", allow_pickle=False)
        if f"{mode}/pos" not in z.files:
            raise SystemExit(f"{d}: mode {mode!r} not in the replay (modes {list(z['modes'])})")
        self.label, self.mode = label, mode
        self.color = NONE_C if mode == "none" else color
        self.idx = {int(e): j for j, e in enumerate(z["episode_index"])}
        self.obst, self.on = z["obst"], z["chaser_on"]
        self.pos, self.ch, self.tg = z[f"{mode}/pos"], z[f"{mode}/chaser"], z[f"{mode}/target"]
        self.lam, self.alive, self.vel = z[f"{mode}/lam"], z[f"{mode}/alive"], z[f"{mode}/vel"]
        self.outcome, self.end = z[f"{mode}/outcome"], z[f"{mode}/end_step"]
        self.names = [str(s) for s in z["outcomes"]]
        self.dt, self.arena = float(z["dt"]), z["arena"]
        self.r_d, self.r_s = float(z["r_drone"]), float(z["strike_radius"])
        self.r_c, self.v_c = float(z["chaser_radius"]), float(z["chaser_speed"])
        self.guidance = str(z["chaser_model"]) == "guidance"
        self.delay, self.a_e = float(z["delay"]), float(z["a_evade"])

    def ep(self, i):
        j = self.idx[i]
        k_end = int(self.alive[:, j].sum())
        return j, max(k_end, 1)

    def verdict(self, i):
        j, k_end = self.ep(i)
        return f"{SHORT[self.names[int(self.outcome[j])]]} ({k_end * self.dt:.1f} s)"


def setup(ax, p: Panel, j: int):
    ax.set_facecolor(SURF)
    for o in p.obst[j]:
        if o[6] > 0.5:
            from matplotlib.patches import Polygon
            ax.add_patch(Polygon(outline(o), closed=True, fc=OBS_FILL, ec=OBS_EDGE, lw=0.9, zorder=2))
    ax.set_xlim(-1, p.arena[0] + 1)
    ax.set_ylim(-p.arena[1] - 0.5, p.arena[1] + 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panel", action="append", required=True, help='"RUN_DIR:MODE:LABEL" (repeat; panels in this order)')
    p.add_argument("--episodes", type=int, nargs="*", default=None, help="episode indices (default: all replayed ones)")
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--gif", type=int, default=1, help="1: also an animated GIF per episode")
    p.add_argument("--pdf", type=int, default=0, help="1: also a PDF of the static figure")
    p.add_argument("--title", default="", help="figure title prefix")
    p.add_argument("--out", default="outputs/strike_episodes")
    a = p.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    panels = [Panel(s, PALETTE[k % len(PALETTE)]) for k, s in enumerate(a.panel)]
    common = set.intersection(*[set(q.idx) for q in panels])
    eps = a.episodes if a.episodes else sorted(common)
    missing = [i for i in eps if i not in common]
    if missing:
        raise SystemExit(f"episodes {missing} are not in every replay (common: {sorted(common)})")
    for i in eps:                                 # every panel must show the same scene
        o0 = panels[0].obst[panels[0].idx[i]]
        if any(not np.allclose(q.obst[q.idx[i]], o0) for q in panels[1:]):
            raise SystemExit(f"episode {i}: the replays have different scenes (same --episodes and --scene_seed?)")
    ncol = min(a.cols, len(panels))
    nrow = int(np.ceil(len(panels) / ncol))
    ref = panels[0]
    md = ["| episode | " + " | ".join(q.label for q in panels) + " |", "|---|" + "---|" * len(panels)]
    for i in eps:
        md.append(f"| {i} | " + " | ".join(q.verdict(i) for q in panels) + " |")
        head = (f"{a.title + ': ' if a.title else ''}episode {i} (chaser {ref.r_c:g} m, {ref.v_c:g} m/s)")

        # static figure: full paths
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.1 * nrow + 0.9), squeeze=False)
        fig.patch.set_facecolor(SURF)
        for ax in axes.ravel()[len(panels):]:
            ax.axis("off")
        for ax, q in zip(axes.ravel(), panels):
            j, k = q.ep(i)
            setup(ax, q, j)
            ax.plot(q.tg[:k, j, 0], q.tg[:k, j, 1], color=TARGET, lw=1.4, zorder=4)
            ax.add_patch(Circle(q.tg[k - 1, j, :2], q.r_s, fc=TARGET, ec=INK, lw=0.6, zorder=6))
            for c in range(q.ch.shape[2]):
                if q.on[j, c] > 0.5:
                    ax.plot(q.ch[:k, j, c, 0], q.ch[:k, j, c, 1], color=CHASER, lw=1.1, ls="--", zorder=4)
                    ax.add_patch(Circle(q.ch[k - 1, j, c, :2], q.r_c, fc=CHASER, ec=INK, lw=0.6, zorder=6))
            ax.plot(q.pos[:k, j, 0], q.pos[:k, j, 1], color=q.color, lw=2.0, zorder=5)
            ax.plot(*q.pos[0, j, :2], "o", color=INK, ms=5, zorder=8)
            code = q.names[int(q.outcome[j])]
            mk, mc = {"strike": ("*", "#d4a017"), "timeout": ("o", INK2)}.get(code, ("X", "#c4302b"))
            ax.plot(*q.pos[k - 1, j, :2], mk, color=mc, mec=INK, ms=12 if mk == "*" else 9, zorder=9)
            ax.set_title(f"{q.label}: {q.verdict(i)}", loc="left", fontsize=10, color=q.color, fontweight="bold")
        fig.legend(handles=[Line2D([], [], color=CHASER, lw=1.1, ls="--", label="chaser path"),
                            Line2D([], [], color=TARGET, lw=1.4, label="target path"),
                            Line2D([], [], marker="*", ls="", ms=11, color="#d4a017", mec=INK, label="strike"),
                            Line2D([], [], marker="X", ls="", ms=9, color="#c4302b", mec=INK, label="contact"),
                            Line2D([], [], marker="o", ls="", ms=8, color=INK2, mec=INK, label="time up")],
                   loc="lower center", ncol=5, frameon=False, fontsize=9)
        fig.suptitle(head, x=0.01, ha="left", fontsize=11)
        fig.tight_layout(rect=(0, 0.05, 1, 0.95))
        fig.savefig(out / f"episode{i}.png", dpi=140, facecolor=SURF)
        if a.pdf:
            fig.savefig(out / f"episode{i}.pdf", facecolor=SURF)
        plt.close(fig)

        if not a.gif:
            continue
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 2.85 * nrow + 0.7), squeeze=False)
        fig.patch.set_facecolor(SURF)
        for ax in axes.ravel()[len(panels):]:
            ax.axis("off")
        arts, k_ends = [], []
        for ax, q in zip(axes.ravel(), panels):
            j, k = q.ep(i)
            setup(ax, q, j)
            k_ends.append(k)
            ax.set_title(q.label, loc="left", fontsize=9.5, color=q.color, fontweight="bold")
            path, = ax.plot([], [], color=q.color, lw=2.0, zorder=5)
            drone = Circle((0, 0), q.r_d * 1.6, fc=q.color, ec=INK, lw=0.8, zorder=8)
            tgt = Circle((0, 0), q.r_s, fc=TARGET, ec=INK, lw=0.8, zorder=6)
            ax.add_patch(drone)
            ax.add_patch(tgt)
            rings, chs = [], []
            for c in range(q.ch.shape[2]):
                on = q.on[j, c] > 0.5
                rg = Circle((0, 0), 0.0, fc=CHASER, alpha=0.12, ec=CHASER, ls="--", lw=0.8, zorder=3)
                cc = Circle((0, 0), q.r_c if on else 0.0, fc=CHASER, ec=INK, lw=0.6, zorder=7)
                ax.add_patch(rg)
                ax.add_patch(cc)
                rings.append((rg, on))
                chs.append(cc)
            txt = ax.text(0.99, 0.03, "", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=INK2)
            arts.append((q, j, k, path, drone, tgt, rings, chs, txt))
        stride = 2
        n_frames = max(k_ends)

        def draw(f):
            kk0 = min(f * stride, n_frames - 1)
            res = []
            for q, j, k, path, drone, tgt, rings, chs, txt in arts:
                kk = min(kk0, k - 1)
                path.set_data(q.pos[:kk + 1, j, 0], q.pos[:kk + 1, j, 1])
                drone.center = tuple(q.pos[kk, j, :2])
                tgt.center = tuple(q.tg[kk, j, :2])
                for c, ((rg, on), cc) in enumerate(zip(rings, chs)):
                    cc.center = rg.center = tuple(q.ch[kk, j, c, :2])
                    if on and q.guidance:         # distance the drone needs at its current closing speed (h_c = 0)
                        rel = q.pos[kk, j] - q.ch[kk, j, c]
                        e = rel / max(np.linalg.norm(rel), 1e-6)
                        s = max(q.v_c - float(e @ q.vel[kk, j]), 0.0)
                        rg.set_radius(q.r_c + q.r_d + s * q.delay + s ** 2 / (2.0 * q.a_e))
                end = "" if kk0 < k - 1 else f"  ({q.verdict(i).split(' (')[0]})"
                lam = "" if q.mode == "none" else f"   kept {q.lam[kk, j]:.2f}"
                txt.set_text(f"t = {kk * q.dt:4.1f} s{lam}{end}")
                res += [path, drone, tgt, txt, *chs, *[r for r, _ in rings]]
            return res

        fig.suptitle(head + ("\ndashed circle: distance needed from the chaser at the current closing speed (h_c = 0)"
                             if ref.guidance else ""), x=0.01, ha="left", fontsize=9.5)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        ani = animation.FuncAnimation(fig, draw, frames=int(np.ceil(n_frames / stride)) + 12, blit=True)
        ani.save(out / f"episode{i}.gif", writer=animation.PillowWriter(fps=25), dpi=72)
        plt.close(fig)
        print(f"[episode {i}] " + "; ".join(f"{q.label}: {q.verdict(i)}" for q in panels), flush=True)
    (out / "episodes.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"wrote {out}/episode*.png" + (" and .gif" if a.gif else "") + ", episodes.md", flush=True)


if __name__ == "__main__":
    main()
