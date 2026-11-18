"""3-D views of the runway bird-deterrence episodes: a bird glyph, drone glyphs, the runway strip and ceiling.

Reads the ``trajectories.npz`` / ``eval.json`` written by ``scripts/eval_runway_ue.py`` (run it with two
cases: the policy without the filter first, then with it - the same episodes). Picks episodes of three
kinds from the bird's path (crosses the runway / climbs through the ceiling / stays on the safe side)
and draws, per episode, the bird (grey) and both drones (unfiltered / filtered) with their attitude at a
few instants.

    python scripts/plot_runway_3d.py --eval_dir outputs/runway_phase2_ue/eval_chaser_vs_cil --gif
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

INK, INK_2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
COL = ["#eb6834", "#2a78d6"]  # validated pair: unfiltered (orange), filtered (blue)
BIRD = "#3d3c39"
NOGO = "#e34948"
ASPHALT = "#8a8986"
GROUND = "#e6e4dc"


def quat_to_rot(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def draw_drone(ax, p, q, color, size=1.0, alpha=1.0, zorder=10):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    r = quat_to_rot(q)
    arms = size * np.array([[1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0]]) / np.sqrt(2)
    tips = p + arms @ r.T
    for a, b in ((0, 1), (2, 3)):
        ax.plot(*np.stack([tips[a], tips[b]]).T, color=INK, lw=1.6, alpha=alpha, zorder=zorder)
    th = np.linspace(0, 2 * np.pi, 18)
    rr = 0.36 * size
    circ = np.stack([rr * np.cos(th), rr * np.sin(th), np.zeros_like(th)], 1)
    polys = [arm_tip + circ @ r.T for arm_tip in tips]
    ax.add_collection3d(Poly3DCollection(polys, facecolor=color, edgecolor=INK, linewidths=0.6, alpha=0.85 * alpha))
    ax.scatter(*p, color=INK, s=8, alpha=alpha, depthshade=False)


def draw_bird(ax, p, v, size=1.25, alpha=1.0):
    """Gull-like glyph: body along the velocity, two wings in a shallow V."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    f = v / max(np.linalg.norm(v), 1e-6)
    side = np.cross([0.0, 0.0, 1.0], f)
    if np.linalg.norm(side) < 1e-6:
        side = np.array([0.0, 1.0, 0.0])
    side = side / np.linalg.norm(side)
    up = np.cross(f, side)
    s = size
    head, tail = p + 0.55 * s * f, p - 0.6 * s * f
    root_f, root_b = p + 0.15 * s * f, p - 0.2 * s * f
    wings = []
    for sg in (1.0, -1.0):
        elbow = p + sg * 0.55 * s * side + 0.22 * s * up + 0.05 * s * f
        tip = p + sg * 1.15 * s * side + 0.05 * s * up - 0.35 * s * f
        wings.append([root_f, elbow, tip, root_b])
    tail_fan = [tail + 0.18 * s * side, p - 0.25 * s * f, tail - 0.18 * s * side]
    ax.add_collection3d(Poly3DCollection(wings + [tail_fan], facecolor=BIRD, edgecolor=INK, linewidths=0.5, alpha=0.9 * alpha))
    ax.plot(*np.stack([tail, head]).T, color=INK, lw=2.0, alpha=alpha)
    ax.scatter(*head, color=INK, s=10, alpha=alpha, depthshade=False)


def classify(bird, z_max):
    zmax = bird[:, 2].max()
    if zmax > z_max + 1.0:
        return "climb"
    if bird[-1, 1] > 3.0:
        return "cross"
    return "safe_side"


def pick_episodes(tr0, z_max, per_kind=2):
    n = tr0["bird"].shape[1]
    kinds = [classify(tr0["bird"][:, i], z_max) for i in range(n)]
    al = tr0["alive"].astype(bool)
    inc = np.where(al, -tr0["h_rwy"], -np.inf).max(0)
    exc = np.where(al, -tr0["h_ceil"], -np.inf).max(0)
    viol = np.maximum(inc, exc)
    out = []
    for kind in ("cross", "climb"):
        idx = [i for i in range(n) if kinds[i] == kind]
        idx.sort(key=lambda i: -viol[i])
        # a clear one and a milder one (rank 1 and rank ~1/3)
        chosen = [idx[0]] + ([idx[min(len(idx) - 1, max(1, len(idx) // 3))]] if len(idx) > 1 else [])
        out += [(kind, i) for i in chosen[:per_kind]]
    safe = [i for i in range(n) if kinds[i] == "safe_side" and viol[i] < 0]
    safe.sort(key=lambda i: -np.ptp(tr0["p"][:, i, 0]) - np.ptp(tr0["p"][:, i, 2]))  # most motion
    out += [("safe_side", i) for i in safe[:per_kind]]
    return out


TITLES = {"cross": "Bird crosses the runway", "climb": "Bird climbs through the ceiling",
          "safe_side": "Bird stays on the drone's side"}


def scene_limits(trs, ep, z_max, y_edge):
    pts = np.concatenate([trs[0]["bird"][:, ep]] + [t["p"][t["alive"][:, ep].astype(bool), ep] for t in trs])
    lo, hi = pts.min(0) - 1.5, pts.max(0) + 1.5
    lo[2], hi[2] = 0.0, max(hi[2], z_max + 2.0)
    hi[1] = max(hi[1], y_edge + 6.0)
    lo[1] = min(lo[1], y_edge - 6.0)
    for i, m in ((0, 16.0), (1, 18.0)):  # minimum spans, so a short episode is not drawn as a sliver
        if hi[i] - lo[i] < m:
            c = 0.5 * (hi[i] + lo[i])
            lo[i], hi[i] = c - m / 2, c + m / 2
    return lo, hi


def scene(ax, trs, ep, z_max, y_edge, t_marks, dt, labels, lims=None):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    lo, hi = scene_limits(trs, ep, z_max, y_edge) if lims is None else lims
    # ground, runway strip with centre line, keep-out wall, ceiling over the drone's side
    ax.add_collection3d(Poly3DCollection([[(lo[0], lo[1], 0), (hi[0], lo[1], 0), (hi[0], y_edge, 0), (lo[0], y_edge, 0)]],
                                         facecolor=GROUND, edgecolor="none", alpha=0.6))
    ax.add_collection3d(Poly3DCollection([[(lo[0], y_edge, 0), (hi[0], y_edge, 0), (hi[0], hi[1], 0), (lo[0], hi[1], 0)]],
                                         facecolor=ASPHALT, edgecolor="none", alpha=0.55))
    yc = 0.5 * (y_edge + hi[1])
    xs = np.arange(lo[0], hi[0], 2.0)
    for x0 in xs:
        ax.plot([x0, min(x0 + 1.0, hi[0])], [yc, yc], [0.01, 0.01], color="white", lw=1.6)
    ax.add_collection3d(Poly3DCollection([[(lo[0], y_edge, 0), (hi[0], y_edge, 0), (hi[0], y_edge, hi[2]), (lo[0], y_edge, hi[2])]],
                                         facecolor=NOGO, edgecolor=NOGO, linewidths=0.8, alpha=0.10))
    ax.add_collection3d(Poly3DCollection([[(lo[0], lo[1], z_max), (hi[0], lo[1], z_max), (hi[0], y_edge, z_max), (lo[0], y_edge, z_max)]],
                                         facecolor=NOGO, edgecolor=NOGO, linewidths=0.8, alpha=0.08))
    ax.text(0.5 * (lo[0] + hi[0]), hi[1] - 1.0, 0.1, "runway", color="white", fontsize=8, fontweight="bold", ha="center")
    ax.text(lo[0] + 0.5, lo[1] + 0.5, z_max + 0.3, f"ceiling {z_max:g} m", color=NOGO, fontsize=7.5)
    # paths
    b = trs[0]["bird"][:, ep]
    ax.plot(*b.T, color=BIRD, lw=1.4, ls=(0, (4, 3)))
    for t, col in zip(trs, COL):
        al = t["alive"][:, ep].astype(bool)
        p = t["p"][al, ep]
        ax.plot(*p.T, color=col, lw=2.2)
        ax.plot(p[:, 0], p[:, 1], np.zeros(len(p)), color=col, lw=0.8, alpha=0.35)  # ground shadow
    ax.plot(b[:, 0], b[:, 1], np.zeros(len(b)), color=BIRD, lw=0.8, alpha=0.3, ls=(0, (4, 3)))
    # glyphs at a few instants
    n_t = b.shape[0]
    for j, k in enumerate(t_marks):
        k = min(k, n_t - 1)
        fade = 0.45 + 0.55 * (j + 1) / len(t_marks)
        vb = b[min(k + 1, n_t - 1)] - b[max(k - 1, 0)]
        draw_bird(ax, b[k], vb, alpha=fade)
        for t, col in zip(trs, COL):
            last = int(min(k, t["alive"][:, ep].sum() - 1))
            draw_drone(ax, t["p"][last, ep], t["q"][last, ep], col, alpha=fade)
        ax.text(*(b[k] + np.array([0, 0, 1.1])), f"{k * dt:.1f}s", fontsize=7, color=INK_2, ha="center")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(0, hi[2])
    ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], hi[2]), zoom=1.18)
    ax.set_xlabel("x along runway [m]", fontsize=8, labelpad=2)
    ax.set_ylabel("y toward runway [m]", fontsize=8, labelpad=2)
    ax.set_zlabel("z [m]", fontsize=8, labelpad=0)
    ax.tick_params(labelsize=7, pad=0)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((1, 1, 1, 0))
        pane.set_edgecolor("#d9d8d3")
    ax.view_init(elev=22, azim=-58)


def episode_stats(trs, ep):
    out = []
    for t in trs:
        al = t["alive"][:, ep].astype(bool)
        inc = max(0.0, -float(t["h_rwy"][al, ep].min()))
        exc = max(0.0, -float(t["h_ceil"][al, ep].min()))
        out.append((inc, exc, float(t["dist"][al, ep].mean())))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval_dir", required=True)
    p.add_argument("--z_max", type=float, default=10.0)
    p.add_argument("--y_edge", type=float, default=0.0)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--per_kind", type=int, default=2)
    p.add_argument("--gif", action="store_true")
    a = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    d = Path(a.eval_dir)
    npz = np.load(d / "trajectories.npz")
    meta = json.loads((d / "eval.json").read_text())
    labels = list(meta["results"].keys())[:2]
    trs = [{k[len(f"{i}_"):]: npz[k] for k in npz.files if k.startswith(f"{i}_")} for i in range(2)]
    if "q" not in trs[0]:
        raise SystemExit("trajectories.npz has no attitude ('q'); re-run scripts/eval_runway_ue.py")
    eps = pick_episodes(trs[0], a.z_max, a.per_kind)
    n_t = trs[0]["bird"].shape[0]
    t_marks = [0, n_t // 3, 2 * n_t // 3, n_t - 1]
    plt.rcParams.update({"font.size": 9})
    ncol = 3
    nrow = int(np.ceil(len(eps) / ncol))
    # column = kind, row = example
    by_kind = {k: [e for e in eps if e[0] == k] for k in ("cross", "climb", "safe_side")}
    nrow = max(len(v) for v in by_kind.values())
    fig = plt.figure(figsize=(6.2 * ncol, 6.4 * nrow))
    fig.patch.set_facecolor(SURF)
    for c, kind in enumerate(("cross", "climb", "safe_side")):
        for r, (_, ep) in enumerate(by_kind[kind]):
            ax = fig.add_subplot(nrow, ncol, r * ncol + c + 1, projection="3d")
            ax.set_facecolor(SURF)
            scene(ax, trs, ep, a.z_max, a.y_edge, t_marks, a.dt, labels)
            st = episode_stats(trs, ep)

            def over(x):
                parts = ([f"{x[0]:.1f} m into the runway"] if x[0] > 0 else []) + ([f"{x[1]:.1f} m above the ceiling"] if x[1] > 0 else [])
                return ", ".join(parts) if parts else "stays out"
            ax.set_title(f"{TITLES[kind]} (episode {ep})\nno filter: {over(st[0])}   |   with CIL: {over(st[1])}",
                         fontsize=9.5, color=INK, pad=0)
    handles = [Line2D([], [], color=BIRD, ls=(0, (4, 3)), lw=1.4, label="bird (glyph at 0, 1/3, 2/3, end of the episode)"),
               Line2D([], [], color=COL[0], lw=2.2, label=labels[0]),
               Line2D([], [], color=COL[1], lw=2.2, label=labels[1]),
               Line2D([], [], color=NOGO, lw=6, alpha=0.25, label="no-entry: runway strip wall (y = 0) and ceiling (z = 10 m)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10)
    r0 = meta["results"]
    fig.suptitle(f"Bird chasing next to a runway under wind, same episodes with and without the safety layer  "
                 f"(all {r0[labels[0]]['episodes']} episodes: unsafe {r0[labels[0]]['unsafe_rate']:.0%} -> "
                 f"{r0[labels[1]]['unsafe_rate']:.0%})", fontsize=12.5, color=INK, x=0.01, ha="left")
    fig.subplots_adjust(left=0.0, right=1.0, top=0.92, bottom=0.09, wspace=0.0, hspace=0.16)
    out = d / "runway_3d.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"wrote {out} | episodes {eps}")
    # one large figure per scenario (slides)
    for kind in ("cross", "climb", "safe_side"):
        for _, ep in by_kind[kind][:1]:
            f1 = plt.figure(figsize=(10, 7.6))
            f1.patch.set_facecolor(SURF)
            ax = f1.add_subplot(1, 1, 1, projection="3d")
            ax.set_facecolor(SURF)
            scene(ax, trs, ep, a.z_max, a.y_edge, t_marks, a.dt, labels)
            st = episode_stats(trs, ep)
            ax.set_title(f"{TITLES[kind]}: unfiltered chaser {', '.join(p_ for p_ in [f'{st[0][0]:.1f} m into the runway' if st[0][0] > 0 else '', f'{st[0][1]:.1f} m above the ceiling' if st[0][1] > 0 else ''] if p_) or 'stays out'}"
                         f" | with CIL: stays out", fontsize=11, color=INK)
            f1.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9.5)
            f1.subplots_adjust(left=0.0, right=1.0, top=0.95, bottom=0.08)
            f1.savefig(d / f"runway_3d_{kind}.png", dpi=170, facecolor=f1.get_facecolor())
            plt.close(f1)

    if a.gif:
        from matplotlib.animation import FuncAnimation, PillowWriter

        show = [by_kind["cross"][0], by_kind["climb"][0]]
        fig2 = plt.figure(figsize=(13, 5.8))
        fig2.patch.set_facecolor(SURF)
        axs = [fig2.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]
        lims = [scene_limits(trs, ep, a.z_max, a.y_edge) for _, ep in show]
        stride = 3

        def frame(f):
            k = min(f * stride, n_t - 1)
            for ax, (kind, ep), lim in zip(axs, show, lims):
                ax.cla()
                ax.set_facecolor(SURF)
                sub = [{key: (v[: k + 1] if v.ndim >= 2 and v.shape[0] == n_t else v) for key, v in t.items()} for t in trs]
                scene(ax, sub, ep, a.z_max, a.y_edge, [k], a.dt, labels, lims=lim)
                ax.set_title(TITLES[kind], fontsize=10, color=INK)
            fig2.suptitle(f"t = {k * a.dt:.2f} s   orange: {labels[0]}   blue: {labels[1]}", fontsize=10.5, color=INK, x=0.01,
                          ha="left")
            return []

        anim = FuncAnimation(fig2, frame, frames=n_t // stride + 1, blit=False)
        anim.save(d / "runway_3d.gif", writer=PillowWriter(fps=20), dpi=105)
        print(f"wrote {d / 'runway_3d.gif'}")


if __name__ == "__main__":
    main()
