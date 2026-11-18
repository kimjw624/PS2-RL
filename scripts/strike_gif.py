#!/usr/bin/env python
"""Presentation GIFs (and MP4s) of the strike task: Phase II episodes and Phase I backup rollouts, with simple skins.

Phase II, from eval_strike_filter.py --save_traj 1 outputs (a --traj is DIR[:MODE[:LABEL]], MODE none | lqr | learned):

    # one scene, one panel per --traj (stacked): comparisons on the same scene
    python scripts/strike_gif.py episode --traj outputs/x:none:"필터 없음" --traj outputs/x:learned:"최종 필터" \
        --episode 12 --out gifs/ep12.gif [--title ..] [--subtitle ..] [--speed 1.5] [--alt 1] [--mp4 1] \
        [--theme light|dark] [--chaser missile|delta] [--building concrete|glass] [--target vehicle|bullseye]
    # the same in 3-D (comparisons side by side with --ncols 2); camera --elev, --azim=A0,A1 (swing), --zoom
    python scripts/strike_gif.py episode --view 3d --traj outputs/x:learned:"최종 필터" --episode 12 --out gifs/ep12_3d.gif
    # several scenes of one run on a grid
    python scripts/strike_gif.py montage --traj outputs/x:learned --episodes pick:strike:0-3 --ncols 2 --out gifs/m.gif
    # episodes worth showing, by category
    python scripts/strike_gif.py pick --traj outputs/x:learned [--n 6]

--episode / --episodes take scene indices ("12", "3,7,9") or pick:CAT[:RANK] / pick:CAT:A-B, CAT one of
closest (closest chaser pass, strike), active (most filter activity), detour (widest y extent), climb (largest height
change), fastest, longest, strike (in scene order), failed, contrast (the first --traj fails, the last strikes).

Phase I (the learned safe-arrival backup vs the hover LQR on the held-out test states):

    # side by side; --index: the k-th test state where the LQR fails and the learned backup recovers (--order visual:
    # LQR failures visible from above and the largest LQR-learned split first)
    python scripts/strike_gif.py phase1 --run outputs/strike_final/p1/td3_a0p2_seed0 --region blocked --index 0 \
        --out gifs/p1_blocked.gif
    # several regions overlaid (LQR red dashed, learned blue)
    python scripts/strike_gif.py phase1grid --run ... --picks blocked:0,chaser:0,escape:0 --out gifs/p1_grid.gif
    # the whole Phase I set (one rollout pass): p1_overview.gif, p1_<region>_<k>(_3d).gif, p1_table.md
    python scripts/strike_gif.py phase1all --run ... --outdir gifs/phase1 [--n 2] [--views 2d,3d]

    python scripts/strike_gif.py skins --out gifs/skins.png                  # the skin options side by side

Skins are drawn as vector shapes (no image files), one or two colours per object: quadrotor (blue), missile or
flying-wing interceptor (red, dark red), glass or concrete buildings (fill, edge), a tracked vehicle (olive) or a
bullseye in a pulsing amber reticle (target). Defaults: light theme, missile, glass, vehicle. In 3-D the buildings are
see-through glass prisms 8 m tall, the drone tilts with its attitude and casts a ground shadow, the target point sits on
the vehicle's mast. The altitude strip (--alt 1) shows the side view x-z under the top view. The status chip shows the filter:
nominal kept, filter active (lambda < 1), committed escape plan, certificate lost. GIFs are re-encoded with ffmpeg
(one palette, rectangle diffs; --optimize 0 to skip), --mp4 1 adds an MP4 (better for slides). Driver:
docker_batch/run_strike_gifs.sh.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# one or two colours per object type: drone blue (+ dark arms), chaser red (+ dark red), buildings one fill (+ edge),
# target olive vehicle (+ dark olive) in an amber reticle
THEMES = {
    "light": dict(bg="#f3f1ea", grid="#e3e0d6", wall="#4a4945", text="#1d1d1b", sub="#6b6a65", panel="#ffffff",
                  drone="#1f5fa8", drone_arm="#1d2a3a", rotor="#8fb8e6", trail="#1f5fa8",
                  chaser="#d0342c", chaser_dk="#7e1d18", flame="#f4a29b",
                  roof="#d3cfc4", roof_edge="#8a867b", glass="#cddbe6", glass_edge="#6e8aa0", shadow=(0.0, 0.0, 0.0, 0.13),
                  target="#e8871e", vehicle="#6b7a3a", vehicle2="#47521f", ground="#ece9e0",
                  ok="#2e9e5b", warn="#e09b16", bad="#c62828", info="#4a6fa5",
                  danger=(0.85, 0.15, 0.15, 0.10), safe=(0.18, 0.62, 0.36, 0.16)),
    "dark": dict(bg="#121821", grid="#1b2430", wall="#8a96a6", text="#e8edf3", sub="#9aa6b5", panel="#0e141c",
                 drone="#4fc3f7", drone_arm="#d6e6f5", rotor="#4fc3f7", trail="#4fc3f7",
                 chaser="#ff5a4f", chaser_dk="#a3241d", flame="#ffb3ad",
                 roof="#3a4657", roof_edge="#6b7d94", glass="#2a4a66", glass_edge="#5d8fbb", shadow=(0.0, 0.0, 0.0, 0.45),
                 target="#ffb347", vehicle="#8a9a5b", vehicle2="#5d6b36", ground="#18202b",
                 ok="#5ad48a", warn="#ffc04d", bad="#ff5c5c", info="#7fa8e0",
                 danger=(1.0, 0.3, 0.3, 0.13), safe=(0.35, 0.83, 0.54, 0.18)),
}
OUTCOME_KO = {0: "타격 성공", 1: "장애물 충돌", 2: "지면 충돌", 3: "요격기 접촉", 4: "경기장 이탈", 5: "시간 초과"}


# ------------------------------------------------------------------------------------------------ geometry helpers
def rot(pts, ang):
    c, s = np.cos(ang), np.sin(ang)
    return pts @ np.array([[c, s], [-s, c]])


def quat_yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def heading(path, k, fallback=0.0, look=3):
    a, b = max(k - look, 0), min(k + look, len(path) - 1)
    d = path[b, :2] - path[a, :2]
    return float(np.arctan2(d[1], d[0])) if np.linalg.norm(d) > 1e-3 else fallback


def box_corners(o):
    cx, cy, bx, by, yaw = o[:5]
    pts = np.array([[-bx, -by], [bx, -by], [bx, by], [-bx, by]])
    return rot(pts, yaw) + [cx, cy]


# ------------------------------------------------------------------------------------------------ skins
class Skins:
    """2-D skins (top view). Defaults: light, missile, glass, vehicle."""

    def __init__(self, theme="light", chaser="missile", building="glass", target="vehicle", scale=1.0, chaser_scale=1.0):
        self.th, self.chaser_kind, self.building_kind, self.target_kind = THEMES[theme], chaser, building, target
        self.s, self.cs = scale, chaser_scale

    def colors(self):
        th = self.th
        return (th["glass"], th["glass_edge"]) if self.building_kind == "glass" else (th["roof"], th["roof_edge"])

    # static layer -------------------------------------------------------------------------
    def building(self, ax, o, shadow=(0.55, -0.55)):
        from matplotlib.patches import Circle, Polygon
        th = self.th
        fill, edge = self.colors()
        cx, cy, bx, by, yaw, rr = o[:6]
        if bx == 0 and by == 0:                                              # round tower
            ax.add_patch(Circle((cx + shadow[0], cy + shadow[1]), rr, fc=th["shadow"], ec="none", zorder=2))
            ax.add_patch(Circle((cx, cy), rr, fc=fill, ec=edge, lw=1.1, zorder=3))
            ax.add_patch(Circle((cx, cy), rr * 0.72, fill=False, ec=edge, lw=0.6, zorder=3))
            return
        c = box_corners(o)
        ax.add_patch(Polygon(c + shadow, fc=th["shadow"], ec="none", zorder=2))
        ax.add_patch(Polygon(c, fc=fill, ec=edge, lw=1.1, zorder=3))
        inner = rot(np.array([[-bx, -by], [bx, -by], [bx, by], [-bx, by]]) * 0.8, yaw) + [cx, cy]
        ax.add_patch(Polygon(inner, fill=False, ec=edge, lw=0.6, zorder=3))

    # dynamic artists (return lists so they can be removed) --------------------------------------
    def quad(self, ax, p, yaw, size=1.0, alpha=1.0, color=None):
        from matplotlib.patches import Circle, Polygon
        th, s = self.th, 0.95 * size * self.s
        col = color or th["drone"]
        arts = []
        tips = rot(np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * s * 0.62, yaw) + p[:2]
        for a, b in ((0, 2), (1, 3)):
            arts += ax.plot([tips[a, 0], tips[b, 0]], [tips[a, 1], tips[b, 1]], color=th["drone_arm"], lw=2.6,
                            solid_capstyle="round", zorder=12, alpha=alpha)
        for t in tips:
            arts.append(ax.add_patch(Circle(t, 0.40 * s, fc=th["rotor"], ec=col, lw=1.0, alpha=0.55 * alpha, zorder=12)))
        arts.append(ax.add_patch(Circle(p[:2], 0.30 * s, fc=col, ec=th["drone_arm"], lw=1.0, alpha=alpha, zorder=13)))
        nose = rot(np.array([[0.62, 0], [0.30, 0.17], [0.30, -0.17]]) * s, yaw) + p[:2]
        arts.append(ax.add_patch(Polygon(nose, fc="white", ec=th["drone_arm"], lw=0.6, alpha=alpha, zorder=14)))
        return arts

    def chaser(self, ax, p, hd, t=0.0, size=1.0, alpha=1.0):
        """Missile (red body, dark red nose and fins) or flying wing; a short flickering exhaust in a light red."""
        from matplotlib.patches import Polygon
        th, L = self.th, 1.7 * size * self.s * self.cs
        red, dk = th["chaser"], th["chaser_dk"]
        arts = []
        put = lambda pts, fc, z, ec="none", lw=0.0: arts.append(ax.add_patch(
            Polygon(rot(pts * L, hd) + p[:2], fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z)))
        if self.chaser_kind == "delta":                                      # tailless flying wing, W trailing edge
            body = np.array([[0.62, 0], [-0.30, 0.58], [-0.42, 0.50], [-0.22, 0.26], [-0.40, 0.12], [-0.28, 0],
                             [-0.40, -0.12], [-0.22, -0.26], [-0.42, -0.50], [-0.30, -0.58]])
            fl = 0.18 + 0.08 * np.sin(37 * t)
            put(np.array([[-0.30, 0.05], [-0.30 - fl, 0], [-0.30, -0.05]]), th["flame"], 10)
            put(body, red, 11, dk, 1.0)
            put(np.array([[0.55, 0], [0.05, 0.09], [-0.30, 0.05], [-0.30, -0.05], [0.05, -0.09]]), dk, 11)
            return arts
        w = 0.11
        fl = 0.28 + 0.10 * np.sin(41 * t) + 0.05 * np.sin(13 * t)
        put(np.array([[-0.5, 0.08], [-0.5 - fl, 0], [-0.5, -0.08]]), th["flame"], 9)
        for sg in (1, -1):
            put(np.array([[-0.30, w], [-0.52, w + 0.20], [-0.52, w]]) * [1, sg], dk, 10)
        put(np.array([[0.30, w], [-0.5, w], [-0.5, -w], [0.30, -w]]), red, 11)
        put(np.array([[0.5, 0], [0.30, w], [0.30, -w]]), dk, 11)
        return arts

    def target(self, ax, p, hd, t=0.0, size=1.0, hit=False):
        """Olive vehicle (or a bullseye) inside a pulsing amber reticle."""
        from matplotlib.patches import Circle, Polygon
        th, s = self.th, size * self.s
        arts = []
        put = lambda pts, fc, ec="none", lw=0.0: arts.append(ax.add_patch(
            Polygon(rot(pts * s, hd) + p[:2], fc=fc, ec=ec, lw=lw, zorder=8)))
        if self.target_kind == "bullseye":
            for r, fc in ((1.2, th["target"]), (0.85, th["panel"]), (0.5, th["target"]), (0.18, th["panel"])):
                arts.append(ax.add_patch(Circle(p[:2], r * s, fc=fc, ec=th["target"], lw=0.8, zorder=8)))
        else:
            track = np.array([[-1.0, 0.42], [1.0, 0.42], [1.0, 0.62], [-1.0, 0.62]])
            put(track, th["vehicle2"])
            put(track * [1, -1], th["vehicle2"])
            put(np.array([[-0.95, -0.42], [0.95, -0.42], [1.05, -0.25], [1.05, 0.25], [0.95, 0.42], [-0.95, 0.42]]),
                th["vehicle"], th["vehicle2"], 1.0)
            put(np.array([[0.1, 0.06], [1.45, 0.06], [1.45, -0.06], [0.1, -0.06]]), th["vehicle2"])
            arts.append(ax.add_patch(Circle(p[:2] + rot(np.array([[-0.1, 0.0]]) * s, hd)[0], 0.36 * s, fc=th["vehicle"],
                                            ec=th["vehicle2"], lw=1.0, zorder=8)))
        r = (1.75 + 0.18 * np.sin(2 * np.pi * 1.4 * t)) * s
        arts.append(ax.add_patch(Circle(p[:2], r, fill=False, ec=th["target"], lw=1.8, zorder=9)))
        for a in range(4):
            u = np.array([np.cos(a * np.pi / 2 + 0.25 * t), np.sin(a * np.pi / 2 + 0.25 * t)])
            q0, q1 = p[:2] + u * (r - 0.35 * s), p[:2] + u * (r + 0.45 * s)
            arts += ax.plot([q0[0], q1[0]], [q0[1], q1[1]], color=th["target"], lw=2.0, zorder=9)
        return arts

    def strike_flash(self, ax, p, k):
        from matplotlib.patches import Circle
        th, arts = self.th, []
        k = min(k, 6)                                                        # a short amber burst, then a ring
        for rad, al in ((0.5 + 0.25 * k, 0.9), (1.0 + 0.4 * k, 0.4)):
            arts.append(ax.add_patch(Circle(p[:2], rad, fc=th["target"], ec="none", alpha=max(al - 0.1 * k, 0.08), zorder=15)))
        arts.append(ax.add_patch(Circle(p[:2], 1.5 + 0.5 * k, fill=False, ec=th["target"], lw=1.6, alpha=0.7, zorder=15)))
        return arts


def fade_line(ax, pts, color, lw=2.0, n_fade=None, alpha_max=0.9, alpha_min=0.05, z=6, ls="-"):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgba
    if len(pts) < 2:
        return []
    seg = np.stack([pts[:-1, :2], pts[1:, :2]], axis=1)
    n = len(seg)
    a = np.linspace(alpha_min, alpha_max, n) if n_fade is None else np.clip(1 - (n - 1 - np.arange(n)) / n_fade, 0, 1) * alpha_max
    rgba = np.array([to_rgba(color, al) for al in a])
    lc = LineCollection(seg, colors=rgba, linewidths=lw, zorder=z, capstyle="round", linestyles=ls)
    ax.add_collection(lc)
    return [lc]


# ------------------------------------------------------------------------------------------------ trajectory data
class Traj:
    """One mode of an eval_strike_filter.py --save_traj output."""

    def __init__(self, spec: str):
        parts = spec.split(":", 2)                                           # the label may contain ":"
        d = Path(parts[0])
        self.dir = d
        self.mode = parts[1] if len(parts) > 1 and parts[1] else "learned"
        self.label = parts[2] if len(parts) > 2 else ""
        z = np.load(d / "strike_filter_traj.npz", allow_pickle=True)
        g = lambda k, dflt=None: z[k] if k in z.files else dflt
        m = self.mode
        self.idx = {int(e): j for j, e in enumerate(z["episode_index"])}
        self.obst, self.on = z["obst"], z["chaser_on"]
        self.pos, self.vel, self.ch, self.tg = z[f"{m}/pos"], z[f"{m}/vel"], z[f"{m}/chaser"], z[f"{m}/target"]
        self.quat, self.lam = g(f"{m}/quat"), z[f"{m}/lam"]
        self.nocert, self.committed = g(f"{m}/nocert"), g(f"{m}/committed")
        self.variant = g(f"{m}/variant")                                    # backup variant j of the applied plan
        self.outcome, self.end = z[f"{m}/outcome"], z[f"{m}/end_step"]
        self.dt, self.arena = float(z["dt"]), np.asarray(z["arena"], float)
        self.r_c, self.r_d = float(z["chaser_radius"]), float(z["r_drone"])
        self.z_floor = float(g("z_floor", 1.0))
        self.z_max = float(g("z_max", 10.0))

    def ep(self, e):
        j = self.idx[int(e)]
        return j, int(self.end[j])

    def status(self, j, k):
        """0 nominal kept, 1 corrected (lambda < 1), 2 committed plan, 3 no certificate."""
        if self.nocert is not None and self.nocert[k, j] > 0.5:
            return 3
        if self.committed is not None and self.committed[k, j] > 0.5:
            return 2
        return 1 if self.lam[k, j] < 0.999 else 0


STATUS_KO = {0: ("명목 입력 유지", "ok"), 1: ("필터 개입 λ = {lam:.2f}", "warn"), 2: ("회피 계획 이어가기", "info"),
             3: ("인증 상실", "bad")}


# ------------------------------------------------------------------------------------------------ Phase II episode
def setup_font():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "Malgun Gothic", "AppleGothic", "Noto Sans CJK JP",
                 "Noto Sans CJK SC", "Noto Sans CJK TC", "Source Han Sans K"):
        if name in have:
            plt.rcParams["font.family"] = [name, "DejaVu Sans"]
            break
    else:
        try:
            import koreanize_matplotlib  # noqa: F401
        except ImportError:
            import warnings
            warnings.warn("no Korean font: labels will not render (apt-get install fonts-nanum, or pip install "
                          "koreanize-matplotlib in the venv)")
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def render_panels(panels, out, *, ncols=1, theme="light", chaser="missile", building="glass", target="vehicle",
                  alt=True, speed=1.0, stride=3, dpi=100, hold=1.6, title="", subtitle="", mp4=False, panel_width=None,
                  scale=1.0, show_status=True, overlay=None, chaser_scale=1.0):
    """panels: [(Traj, episode, label), ...] drawn on a grid of ncols columns, all on one clock; each panel freezes at its end."""
    plt = setup_font()
    from matplotlib import animation
    th = THEMES[theme]
    sk = Skins(theme, chaser, building, target, scale, chaser_scale)
    n_p = len(panels)
    ncols = max(1, min(ncols, n_p))
    nrows = int(np.ceil(n_p / ncols))
    pw = panel_width or (8.6 if ncols == 1 else 5.6)
    fs = min(1.0, pw / 8.6) ** 0.5                                           # font scale for narrow panels
    eps = [q.ep(e) for q, e, _ in panels]
    k_max = max(k for _, k in eps)
    q0 = panels[0][0]
    ax_x, ay = q0.arena[0], q0.arena[1]
    margin = 0.035 * 8.6 / pw * 0.6
    top_h = pw * (1 - 2 * margin) * (2 * ay + 2) / (ax_x + 2)
    alt_h = 1.2 * fs if alt else 0.0
    head_h = 0.40 * fs
    title_h = (0.36 if title else 0.08) + (0.26 if subtitle else 0.0)
    cell_h = top_h + alt_h + head_h
    fig_w, fig_h = ncols * pw, nrows * cell_h + title_h
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=th["panel"])
    axes = []
    for i in range(n_p):
        r, c = divmod(i, ncols)
        x0 = (c * pw + margin * pw) / fig_w
        w = pw * (1 - 2 * margin) / fig_w
        y_top = fig_h - title_h - r * cell_h - head_h
        a_top = fig.add_axes([x0, (y_top - top_h) / fig_h, w, top_h / fig_h])
        a_alt = None
        if alt:
            a_alt = fig.add_axes([x0, (y_top - top_h - alt_h + 0.12 * fs) / fig_h, w, (alt_h - 0.22 * fs) / fig_h])
        axes.append((a_top, a_alt))
    x_t = margin * pw / fig_w
    if title:
        fig.text(x_t, 1 - 0.22 / fig_h, title, fontsize=12.5, fontweight="bold", color=th["text"], va="center")
    if subtitle:
        fig.text(x_t, 1 - (0.22 + 0.28) / fig_h, subtitle, fontsize=9.5, color=th["sub"], va="center")

    for (a_top, a_alt), (q, _, _), (j, k_end) in zip(axes, panels, eps):      # static layers
        a_top.set_facecolor(th["bg"])
        a_top.set_xlim(-1, ax_x + 1)
        a_top.set_ylim(-ay - 1, ay + 1)
        a_top.set_aspect("equal")
        a_top.set_xticks([]), a_top.set_yticks([])
        for sp in a_top.spines.values():
            sp.set_visible(False)
        for gx in np.arange(0, ax_x + 0.1, 5):
            a_top.plot([gx, gx], [-ay, ay], color=th["grid"], lw=0.6, zorder=1)
        for gy in np.arange(-ay, ay + 0.1, 5):
            a_top.plot([0, ax_x], [gy, gy], color=th["grid"], lw=0.6, zorder=1)
        a_top.plot([0, ax_x, ax_x, 0, 0], [-ay, -ay, ay, ay, -ay], color=th["wall"], lw=2.2, zorder=4)
        for o in q.obst[j]:
            if o[6] > 0.5:
                sk.building(a_top, o)
        a_top.plot(q.tg[:k_end, j, 0], q.tg[:k_end, j, 1], color=th["target"], lw=0.8, ls=":", alpha=0.5, zorder=5)
        if a_alt is not None:
            a_alt.set_facecolor(th["bg"])
            a_alt.set_xlim(-1, ax_x + 1)
            a_alt.set_ylim(0, max(q.z_max, np.nanmax(q.tg[:k_end, j, 2]) + 1.0, np.nanmax(q.pos[:k_end, j, 2]) + 1.0))
            a_alt.set_yticks([0, 5, 10])
            a_alt.tick_params(labelsize=7 * fs, colors=th["sub"], length=2)
            a_alt.set_xticks([])
            for sp in a_alt.spines.values():
                sp.set_color(th["grid"])
            a_alt.axhspan(0, q.z_floor, color=th["grid"], zorder=1)
            for o in q.obst[j]:
                if o[6] > 0.5:
                    xs = box_corners(o)[:, 0] if (o[2] > 0 or o[3] > 0) else np.array([o[0] - o[5], o[0] + o[5]])
                    a_alt.axvspan(xs.min(), xs.max(), color=sk.colors()[0], alpha=0.8, zorder=2, lw=0)
            a_alt.set_ylabel("고도 [m]", fontsize=7 * fs, color=th["sub"], labelpad=1)

    heads = []
    for (a_top, _), (q, e, label) in zip(axes, panels):
        pos = a_top.get_position()
        dy = 0.08 / fig_h
        h1 = fig.text(pos.x0, pos.y1 + dy, label, fontsize=10.5 * fs, fontweight="bold", color=th["text"], va="bottom")
        h2 = fig.text(pos.x1, pos.y1 + dy, "", fontsize=9.5 * fs, color=th["sub"], va="bottom", ha="right")
        heads.append((h1, h2))

    n_hold = int(round(hold * speed / (q0.dt * stride)))
    frames = list(range(0, k_max, stride)) + [k_max - 1] * n_hold
    dyn = []

    def draw(fi):
        for a_ in dyn:
            a_.remove()
        dyn.clear()
        f = frames[fi]
        for (a_top, a_alt), (q, _e, _l), (j, k_end), (h1, h2) in zip(axes, panels, eps, heads):
            k = min(f, k_end - 1)
            t = k * q.dt
            p, tp = q.pos[k, j], q.tg[k, j]
            ended = f >= k_end - 1
            held = (f - (k_end - 1)) // stride if ended else -1
            dyn.extend(fade_line(a_top, q.pos[max(0, k - int(4 / q.dt)):k + 1, j], th["trail"], lw=2.4, z=6))
            dyn.extend(a_top.plot(q.pos[:k + 1, j, 0], q.pos[:k + 1, j, 1], color=th["trail"], lw=0.8, alpha=0.25, zorder=5))
            dyn.extend(sk.target(a_top, tp, heading(q.tg[:, j], k), t))
            for c in range(q.ch.shape[2]):
                if q.on[j, c] > 0.5:
                    cp = q.ch[k, j, c]
                    dyn.extend(fade_line(a_top, q.ch[max(0, k - int(2.5 / q.dt)):k + 1, j, c], th["chaser"], lw=2.2,
                                         alpha_max=0.35, z=7))
                    dyn.extend(sk.chaser(a_top, cp, heading(q.ch[:, j, c], k, fallback=np.arctan2(*(p[:2] - cp[:2])[::-1])), t))
            if overlay is not None and not ended:
                dyn.extend(overlay(a_top, axes.index((a_top, a_alt)), k))
            yaw = quat_yaw(q.quat[k, j]) if q.quat is not None else heading(q.pos[:, j], k)
            dyn.extend(sk.quad(a_top, p, yaw))
            if ended and q.outcome[j] == 0:
                dyn.extend(sk.strike_flash(a_top, tp, max(held, 0)))
            if a_alt is not None:
                dyn.extend(a_alt.plot(q.pos[:k + 1, j, 0], q.pos[:k + 1, j, 2], color=th["trail"], lw=1.4, zorder=6))
                dyn.extend(a_alt.plot([p[0]], [p[2]], "o", color=th["drone"], ms=6 * fs, mec=th["drone_arm"], zorder=8))
                dyn.extend(a_alt.plot([tp[0]], [tp[2]], "D", color=th["target"], ms=6 * fs, mec="white", zorder=8))
                for c in range(q.ch.shape[2]):
                    if q.on[j, c] > 0.5:
                        dyn.extend(a_alt.plot([q.ch[k, j, c, 0]], [q.ch[k, j, c, 2]], ">", color=th["chaser"], ms=6 * fs,
                                              zorder=7))
            spd = float(np.linalg.norm(q.vel[k, j]))
            h2.set_text(f"t = {t:4.1f} s   속도 {spd:4.1f} m/s   고도 {p[2]:3.1f} m" if pw >= 7.5
                        else f"t {t:4.1f} s · {spd:3.1f} m/s · 고도 {p[2]:3.1f} m")
            if show_status and q.lam is not None:
                txt, key = STATUS_KO[q.status(j, k)]
                if q.mode == "none":
                    txt, key = "필터 없음", "sub"
                dyn.append(a_top.text(0.2, ay - 0.6, " " + txt.format(lam=float(q.lam[k, j])) + " ", fontsize=9 * fs,
                                      color="white", fontweight="bold", va="top", ha="left", zorder=20,
                                      bbox=dict(boxstyle="round,pad=0.3", fc=th[key], ec="none", alpha=0.92)))
            if ended:
                code = int(q.outcome[j])
                col = th["ok"] if code == 0 else (th["sub"] if code == 5 else th["bad"])
                mark = "✓" if code == 0 else "✗"
                dyn.append(a_top.text(ax_x / 2, -ay + 2.0, f" {mark} {OUTCOME_KO.get(code, '')}   {k_end * q.dt:.1f} s ",
                                      fontsize=15 * fs, fontweight="bold", color="white", ha="center", va="center", zorder=25,
                                      bbox=dict(boxstyle="round,pad=0.45", fc=col, ec="none", alpha=0.9)))
                if code != 0:
                    dyn.extend(a_top.plot([p[0]], [p[1]], "x", color=th["bad"], ms=16 * fs, mew=3.5, zorder=24))
        return dyn

    fps = speed / (q0.dt * stride)
    ani = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps, blit=False)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ani.save(out, writer=animation.PillowWriter(fps=fps), dpi=dpi, savefig_kwargs={"facecolor": th["panel"]})
    finish(out, mp4)
    plt.close(fig)
    return out


def render_episode(trajs, episode, out, **kw):
    """One episode, one panel per trajectory (stacked)."""
    return render_panels([(q, episode, q.label) for q in trajs], out, ncols=1, **kw)


# ------------------------------------------------------------------------------------------------ 3-D
def quat_R(q):
    """Body-to-world rotation of a quaternion (w, x, y, z)."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def dir_frame(d):
    e = np.asarray(d, float) / max(float(np.linalg.norm(d)), 1e-9)
    a = np.array([0.0, 0.0, 1.0]) if abs(e[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(e, a)
    u /= np.linalg.norm(u)
    return e, u, np.cross(e, u)


def shade(color, normal, light=(-0.45, -0.55, 0.70), lo=0.70, hi=1.0):
    """One colour, brightened or darkened by the face's angle to a fixed light (keeps one hue per object)."""
    from matplotlib.colors import to_rgb
    l = np.asarray(light, float) / np.linalg.norm(light)
    n = np.asarray(normal, float) / max(float(np.linalg.norm(normal)), 1e-9)
    return tuple(np.clip(np.asarray(to_rgb(color)) * (lo + (hi - lo) * max(float(n @ l), 0.0)), 0, 1))


def footprint(o, n=28):
    cx, cy, bx, by, yaw, rr = o[:6]
    if bx == 0 and by == 0:
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.stack([cx + rr * np.cos(a), cy + rr * np.sin(a)], 1), True
    return box_corners(o), False


def heading3(path, k, look=3):
    a, b = max(k - look, 0), min(k + look, len(path) - 1)
    d = path[b] - path[a]
    return d if np.linalg.norm(d) > 1e-3 else np.array([1.0, 0.0, 0.0])


class Skins3D(Skins):
    """3-D skins: the same colours as the top view; one hue per object, shaded by face angle."""

    def eye(self, ax):
        e, a = np.radians(ax.elev), np.radians(ax.azim)
        return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])

    def ground(self, ax, xlim, ylim, z_top, elev, azim, zoom, border=True, grid=5.0):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        th = self.th
        (x0, x1), (y0, y1) = xlim, ylim
        ax.computed_zorder = False
        ax.set_axis_off()
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_zlim(0, z_top)
        ax.set_box_aspect((x1 - x0, y1 - y0, z_top), zoom=zoom)
        ax.view_init(elev=elev, azim=azim)
        ax.add_collection3d(Poly3DCollection([[(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)]], facecolor=th["ground"],
                                             edgecolor="none", zorder=0))
        for gx in np.arange(np.ceil(x0 / grid) * grid, x1, grid):
            ax.plot([gx, gx], [y0, y1], [0, 0], color=th["grid"], lw=0.7, zorder=0.5)
        for gy in np.arange(np.ceil(y0 / grid) * grid, y1, grid):
            ax.plot([x0, x1], [gy, gy], [0, 0], color=th["grid"], lw=0.7, zorder=0.5)
        if border:
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], [0] * 5, color=th["wall"], lw=1.6, zorder=0.6)

    def buildings3d(self, ax, obst, top, z0=0.0):
        """Prisms (walls shaded per face, roof), farthest from the eye first; glass walls see-through."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        fill, edge = self.colors()
        glass = self.building_kind == "glass"
        eye = self.eye(ax)
        os_ = [o for o in obst if o[6] > 0.5]
        os_.sort(key=lambda o: float(np.array([o[0], o[1], top / 2]) @ eye))
        for i, o in enumerate(os_):
            xy, round_ = footprint(o)
            n = len(xy)
            faces, cols = [], []
            for a in range(n):
                p0, p1 = xy[a], xy[(a + 1) % n]
                faces.append([(*p0, z0), (*p1, z0), (*p1, top), (*p0, top)])
                d = p1 - p0
                cols.append(shade(fill, [d[1], -d[0], 0.0]))
            z = 2 + 0.01 * i
            ax.add_collection3d(Poly3DCollection(faces, facecolors=cols, edgecolors="none" if round_ else edge,
                                                 linewidths=0.5, alpha=0.5 if glass else 0.95, zorder=z))
            ax.add_collection3d(Poly3DCollection([[(*q, top) for q in xy]], facecolors=[shade(fill, [0, 0, 1])],
                                                 edgecolors=edge, linewidths=0.9, alpha=0.8 if glass else 1.0, zorder=z + 0.005))
            if not round_:                                                  # vertical edges
                for q in xy:
                    ax.plot([q[0], q[0]], [q[1], q[1]], [z0, top], color=edge, lw=0.6, alpha=0.8, zorder=z + 0.004)

    def quad3d(self, ax, p, quat, size=1.0, alpha=1.0, z0=0.0, shadow=True, radius=None):
        """radius [m]: draw to scale, rotor tips at this distance from the centre (else the skin's scale)."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        th, s = self.th, (radius / 1.277 if radius is not None else 1.15 * size * self.s)
        R = quat_R(quat) if quat is not None else np.eye(3)
        tips = p + (np.array([[1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0]]) * 0.62 * s) @ R.T
        arts = []
        if shadow:
            a = np.linspace(0, 2 * np.pi, 16)
            arts.append(ax.add_collection3d(Poly3DCollection([np.stack([p[0] + 0.8 * s * np.cos(a), p[1] + 0.8 * s * np.sin(a),
                                                                       np.full(16, z0 + 0.02)], 1)],
                                                             facecolor=(0, 0, 0, 0.18 * alpha), edgecolor="none", zorder=1)))
            arts += ax.plot([p[0], p[0]], [p[1], p[1]], [z0, p[2]], color=th["drone"], lw=0.8, ls=":", alpha=0.45 * alpha,
                            zorder=1.5)
        for a_, b_ in ((0, 2), (1, 3)):
            arts += ax.plot(*np.stack([tips[a_], tips[b_]], 1), color=th["drone_arm"], lw=2.4, alpha=alpha, zorder=12,
                            solid_capstyle="round")
        ang = np.linspace(0, 2 * np.pi, 14)
        ring = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], 1) * 0.40 * s
        discs = [t + ring @ R.T for t in tips]
        arts.append(ax.add_collection3d(Poly3DCollection(discs, facecolor=th["rotor"], edgecolor=th["drone"], linewidths=0.8,
                                                         alpha=0.6 * alpha, zorder=12.5)))
        arts.append(ax.scatter(*p, s=26, c=th["drone"], edgecolors=th["drone_arm"], linewidths=0.8, alpha=alpha, zorder=13,
                               depthshade=False))
        return arts

    def chaser3d(self, ax, p, d, t=0.0, size=1.0, z0=0.0, length=None):
        """length [m]: draw to scale, nose to tail (e.g. the chaser's diameter 2 r_c), else the skin's scale."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        th, L = self.th, (float(length) if length is not None else 1.7 * size * self.s * self.cs)
        e, u, v = dir_frame(d)
        tail, nb, tip = p - 0.5 * L * e, p + 0.3 * L * e, p + 0.5 * L * e
        fl = 0.28 + 0.10 * np.sin(41 * t) + 0.05 * np.sin(13 * t)
        arts = []
        arts += ax.plot(*np.stack([tail, tail - fl * L * e], 1), color=th["flame"], lw=3.2, zorder=10, solid_capstyle="round")
        if self.chaser_kind == "delta":
            wing = [tip, tail + 0.55 * L * u, tail + 0.15 * L * e, tail - 0.55 * L * u]
            arts.append(ax.add_collection3d(Poly3DCollection([wing], facecolor=th["chaser"], edgecolor=th["chaser_dk"],
                                                             linewidths=0.8, zorder=11)))
            arts += ax.plot(*np.stack([tail, tip], 1), color=th["chaser_dk"], lw=2.2, zorder=11.2)
        else:
            fins = [[tail + 0.22 * L * e, tail, tail + 0.24 * L * w] for w in (u, -u, v, -v)]
            arts.append(ax.add_collection3d(Poly3DCollection(fins, facecolor=th["chaser_dk"], edgecolor="none", zorder=10.5)))
            arts += ax.plot(*np.stack([tail, nb], 1), color=th["chaser"], lw=5.5, zorder=11, solid_capstyle="butt")
            arts += ax.plot(*np.stack([nb, tip], 1), color=th["chaser_dk"], lw=2.6, zorder=11.1, solid_capstyle="round")
        arts.append(ax.scatter(p[0], p[1], z0 + 0.02, s=22, c=[(0, 0, 0, 0.18)], edgecolors="none", zorder=1, depthshade=False))
        return arts

    def target3d(self, ax, tp, hd, t=0.0, z0=0.0, size=1.0):
        """Olive vehicle on the ground, a mast up to the strike point, an amber reticle ring around the point."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        th, s = self.th, size * self.s
        arts = []
        c, sn = np.cos(hd), np.sin(hd)
        Rz = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
        base = np.array([tp[0], tp[1], z0])
        if self.target_kind == "bullseye":
            for r, fc in ((1.3, th["target"]), (0.9, th["panel"]), (0.5, th["target"])):
                a = np.linspace(0, 2 * np.pi, 24)
                arts.append(ax.add_collection3d(Poly3DCollection([np.stack([tp[0] + r * s * np.cos(a), tp[1] + r * s * np.sin(a),
                                                                           np.full(24, z0 + 0.03)], 1)], facecolor=fc,
                                                                 edgecolor="none", zorder=1.2)))
            top = z0
        else:
            def box(cx, hx, hy, z_lo, z_hi):
                v = np.array([[x, y, z] for z in (z_lo, z_hi) for y in (-hy, hy) for x in (-hx, hx)])
                v = (v + [cx, 0, 0]) @ Rz.T + base
                f = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
                nrm = [(0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0)]
                return [[v[i] for i in ff] for ff in f], [shade(th["vehicle"], np.asarray(nn) @ Rz.T) for nn in nrm]
            f1, c1 = box(0.0, 1.0 * s, 0.6 * s, z0, z0 + 0.55 * s)
            f2, c2 = box(-0.1 * s, 0.4 * s, 0.36 * s, z0 + 0.55 * s, z0 + 0.85 * s)
            arts.append(ax.add_collection3d(Poly3DCollection(f1 + f2, facecolors=c1 + c2, edgecolors=th["vehicle2"],
                                                             linewidths=0.5, zorder=3)))
            top = z0 + 0.85 * s
        arts += ax.plot([tp[0], tp[0]], [tp[1], tp[1]], [top, tp[2]], color=th["vehicle2"], lw=1.4, zorder=3.1)
        r = (1.6 + 0.18 * np.sin(2 * np.pi * 1.4 * t)) * s
        a = np.linspace(0, 2 * np.pi, 40)
        arts += ax.plot(tp[0] + r * np.cos(a), tp[1] + r * np.sin(a), np.full(40, tp[2]), color=th["target"], lw=1.8, zorder=9)
        for k in range(4):
            ang = k * np.pi / 2 + 0.25 * t
            q0, q1 = tp[:2] + (r - 0.35 * s) * np.array([np.cos(ang), np.sin(ang)]), tp[:2] + (r + 0.45 * s) * np.array([np.cos(ang), np.sin(ang)])
            arts += ax.plot([q0[0], q1[0]], [q0[1], q1[1]], [tp[2], tp[2]], color=th["target"], lw=2.0, zorder=9)
        arts.append(ax.scatter(*tp, s=34, marker="D", c=th["target"], edgecolors=th["vehicle2"], linewidths=0.6, zorder=9.1,
                               depthshade=False))
        return arts

    def flash3d(self, ax, tp, k):
        th = self.th
        k = min(k, 6)
        a = np.linspace(0, 2 * np.pi, 40)
        r = 1.5 + 0.5 * k
        arts = [ax.scatter(*tp, s=(60 + 90 * k) * 4, c=[th["target"]], alpha=max(0.85 - 0.11 * k, 0.1), edgecolors="none",
                           zorder=15, depthshade=False)]
        arts += ax.plot(tp[0] + r * np.cos(a), tp[1] + r * np.sin(a), np.full(40, tp[2]), color=th["target"], lw=1.6, alpha=0.7,
                        zorder=15)
        return arts


def fade_line3d(ax, pts, color, lw=2.0, alpha_max=0.9, alpha_min=0.05, z=6):
    from matplotlib.colors import to_rgba
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    if len(pts) < 2:
        return []
    seg = np.stack([pts[:-1], pts[1:]], axis=1)
    a = np.linspace(alpha_min, alpha_max, len(seg))
    lc = Line3DCollection(seg, colors=[to_rgba(color, al) for al in a], linewidths=lw, zorder=z)
    ax.add_collection3d(lc)
    return [lc]


def render3d_panels(panels, out, *, ncols=1, theme="light", chaser="missile", building="glass", target="vehicle", speed=1.0,
                    stride=3, dpi=100, hold=1.6, title="", subtitle="", mp4=False, panel_width=None, scale=1.0, show_status=True,
                    elev=26.0, azim=(-122.0, -92.0), zoom=0.95, building_top=8.0, chaser_scale=1.0, obj_scale=1.4,
                    v_shift=0.0):
    """3-D version of render_panels: same clock, chips and badges; the camera swings from azim[0] to azim[1]."""
    plt = setup_font()
    from matplotlib import animation
    th = THEMES[theme]
    sk = Skins3D(theme, chaser, building, target, scale * obj_scale, chaser_scale)
    n_p = len(panels)
    ncols = max(1, min(ncols, n_p))
    nrows = int(np.ceil(n_p / ncols))
    pw = panel_width or (9.6 if ncols == 1 else 6.4)
    fs = min(1.0, pw / 9.6) ** 0.5
    eps = [q.ep(e) for q, e, _ in panels]
    k_max = max(k for _, k in eps)
    q0 = panels[0][0]
    ax_x, ay = q0.arena[0], q0.arena[1]
    z_top = max(building_top, max(float(np.nanmax(q.tg[:k, j, 2])) for (q, _, _), (j, k) in zip(panels, eps)) + 1.0,
                max(float(np.nanmax(q.pos[:k, j, 2])) for (q, _, _), (j, k) in zip(panels, eps)) + 1.0)
    cell_h = pw * 0.52
    head_h = 0.42 * fs
    title_h = (0.36 if title else 0.08) + (0.26 if subtitle else 0.0)
    fig_w, fig_h = ncols * pw, nrows * (cell_h + head_h) + title_h
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=th["panel"])
    axes, heads = [], []
    for i, ((q, e, label), (j, k_end)) in enumerate(zip(panels, eps)):
        r, c = divmod(i, ncols)
        y_top = fig_h - title_h - r * (cell_h + head_h) - head_h
        # 3-D axes draw in a square viewport: make it the panel's full width, centred on the cell, transparent
        yc = y_top - cell_h / 2 + v_shift * cell_h
        ax = fig.add_axes([c * pw / fig_w, (yc - pw / 2) / fig_h, pw / fig_w, pw / fig_h], projection="3d")
        ax.patch.set_alpha(0.0)
        sk.ground(ax, (0, ax_x), (-ay, ay), z_top, elev, azim[0], zoom)
        sk.buildings3d(ax, q.obst[j], building_top)
        ax.plot(q.tg[:k_end, j, 0], q.tg[:k_end, j, 1], np.zeros(k_end) + 0.03, color=th["target"], lw=0.8, ls=":", alpha=0.6,
                zorder=1)
        axes.append(ax)
        x0, x1 = (c * pw + 0.18) / fig_w, ((c + 1) * pw - 0.18) / fig_w
        yy = (y_top + 0.10 * fs) / fig_h
        h1 = fig.text(x0, yy, label, fontsize=10.5 * fs, fontweight="bold", color=th["text"], va="bottom")
        h2 = fig.text(x1, yy, "", fontsize=9.5 * fs, color=th["sub"], va="bottom", ha="right")
        chip = fig.text(x0, (y_top - 0.12) / fig_h, "", fontsize=9 * fs, color="white", fontweight="bold", va="top")
        badge = fig.text((c + 0.5) * pw / fig_w, (y_top - cell_h + 0.30) / fig_h, "", fontsize=15 * fs, fontweight="bold",
                         color="white", ha="center", va="center")
        heads.append((h1, h2, chip, badge))
    x_t = 0.18 / fig_w
    if title:
        fig.text(x_t, 1 - 0.22 / fig_h, title, fontsize=12.5, fontweight="bold", color=th["text"], va="center")
    if subtitle:
        fig.text(x_t, 1 - (0.22 + 0.28) / fig_h, subtitle, fontsize=9.5, color=th["sub"], va="center")

    n_hold = int(round(hold * speed / (q0.dt * stride)))
    frames = list(range(0, k_max, stride)) + [k_max - 1] * n_hold
    n_move = max(len(frames) - n_hold, 1)
    dyn = []

    def draw(fi):
        for a_ in dyn:
            a_.remove()
        dyn.clear()
        f = frames[fi]
        az = azim[0] + (azim[1] - azim[0]) * min(fi / n_move, 1.0)
        for ax, (q, _e, _l), (j, k_end), (h1, h2, chip, badge) in zip(axes, panels, eps, heads):
            ax.view_init(elev=elev, azim=az)
            k = min(f, k_end - 1)
            t = k * q.dt
            p, tp = q.pos[k, j], q.tg[k, j]
            ended = f >= k_end - 1
            held = (f - (k_end - 1)) // stride if ended else -1
            dyn.extend(fade_line3d(ax, q.pos[max(0, k - int(4 / q.dt)):k + 1, j], th["trail"], lw=2.4, z=6))
            dyn.extend(ax.plot(q.pos[:k + 1, j, 0], q.pos[:k + 1, j, 1], q.pos[:k + 1, j, 2], color=th["trail"], lw=0.8,
                               alpha=0.25, zorder=5))
            dyn.extend(ax.plot(q.pos[:k + 1, j, 0], q.pos[:k + 1, j, 1], np.full(k + 1, 0.02), color=(0, 0, 0, 0.12), lw=1.0,
                               zorder=1))
            dyn.extend(sk.target3d(ax, tp, heading(q.tg[:, j], k), t))
            for ci in range(q.ch.shape[2]):
                if q.on[j, ci] > 0.5:
                    cp = q.ch[k, j, ci]
                    dyn.extend(fade_line3d(ax, q.ch[max(0, k - int(2.5 / q.dt)):k + 1, j, ci], th["chaser"], lw=2.0,
                                           alpha_max=0.4, z=7))
                    dyn.extend(sk.chaser3d(ax, cp, heading3(q.ch[:, j, ci], k), t))
            dyn.extend(sk.quad3d(ax, p, q.quat[k, j] if q.quat is not None else None))
            if ended and q.outcome[j] == 0:
                dyn.extend(sk.flash3d(ax, tp, max(held, 0)))
            spd = float(np.linalg.norm(q.vel[k, j]))
            h2.set_text(f"t = {t:4.1f} s   속도 {spd:4.1f} m/s   고도 {p[2]:3.1f} m" if pw >= 8
                        else f"t {t:4.1f} s · {spd:3.1f} m/s · 고도 {p[2]:3.1f} m")
            if show_status and q.lam is not None:
                txt, key = STATUS_KO[q.status(j, k)]
                if q.mode == "none":
                    txt, key = "필터 없음", "sub"
                chip.set_text(" " + txt.format(lam=float(q.lam[k, j])) + " ")
                chip.set_bbox(dict(boxstyle="round,pad=0.3", fc=th[key], ec="none", alpha=0.92))
            if ended:
                code = int(q.outcome[j])
                col = th["ok"] if code == 0 else (th["sub"] if code == 5 else th["bad"])
                badge.set_text(f" {'✓' if code == 0 else '✗'} {OUTCOME_KO.get(code, '')}   {k_end * q.dt:.1f} s ")
                badge.set_bbox(dict(boxstyle="round,pad=0.45", fc=col, ec="none", alpha=0.9))
                if code != 0:
                    dyn.append(ax.scatter(*p, s=260, marker="x", c=th["bad"], linewidths=3.5, zorder=24, depthshade=False))
            else:
                badge.set_text("")
                badge.set_bbox(None)
        return dyn

    fps = speed / (q0.dt * stride)
    ani = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps, blit=False)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ani.save(out, writer=animation.PillowWriter(fps=fps), dpi=dpi, savefig_kwargs={"facecolor": th["panel"]})
    finish(out, mp4)
    plt.close(fig)
    return out


OPTIMIZE = True


def finish(gif: Path, mp4=False):
    """MP4 copy (optional), then re-encode the GIF with one ffmpeg palette and rectangle diffs (about half the size)."""
    if mp4:
        to_mp4(gif)
    if OPTIMIZE and shutil.which("ffmpeg"):
        tmp = gif.with_name(gif.stem + ".opt.gif")
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(gif), "-vf",
                            "split[s0][s1];[s0]palettegen=max_colors=256:stats_mode=full[p];"
                            "[s1][p]paletteuse=dither=none:diff_mode=rectangle", str(tmp)], check=False)
        if r.returncode == 0 and tmp.exists() and tmp.stat().st_size < gif.stat().st_size:
            tmp.replace(gif)
        elif tmp.exists():
            tmp.unlink()


def to_mp4(gif: Path):
    mp4 = gif.with_suffix(".mp4")
    if not shutil.which("ffmpeg"):
        print("[strike_gif] no ffmpeg: MP4 skipped")
        return None
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(gif), "-movflags", "faststart", "-pix_fmt", "yuv420p",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(mp4)], check=False)
    return mp4


# ------------------------------------------------------------------------------------------------ filter plans
def filter_plans(q: Traj, episode, backup_run, task, stride=3):
    """The plans the filter checks, replayed from the recorded states (approximate: fresh view, the committed state is
    not recorded): per frame step k, every backup variant's own plan (blend 0) with its verdict, and the plan of the
    recorded choice (variant, blend lambda). Returns {k: (plans (V, H, 2), ok (V,), chosen (H, 2), chaser paths (C, H, 2))}."""
    import jax
    import jax.numpy as jnp

    from ps2rl.cil.strike_filter import backup_ranges, load_backup, make_filter, model_from_backup
    from ps2rl.cil.strike_guidance import GuidanceConfig
    from ps2rl.envs.strike_env import StrikeEnvConfig
    t = json.load(open(task))
    gd = json.loads(t["tracker"]["filter_guidance"])
    m = model_from_backup(backup_run)
    o_rng, b_rng = backup_ranges(backup_run)
    bk, prm = load_backup(m, "learned", backup_run)
    ec = StrikeEnvConfig.from_dict(t.get("env", {}))
    fl = make_filter(m, bk, prm, obstacle_range=o_rng, ball_range=b_rng, hover_latch=True, chaser_model="guidance",
                     guidance=GuidanceConfig.from_dict(gd), arena=(0.0, ec.arena_x, ec.arena_y_half, ec.z_max))
    cand, fv = jax.jit(fl.candidate, static_argnums=(6,)), jax.jit(fl.fresh_view)
    z = np.load(Path(t_dir(q)) / "strike_filter_traj.npz", allow_pickle=True)
    u_nom, var = z[f"{q.mode}/u_nom"], z[f"{q.mode}/variant"]
    j, k_end = q.ep(episode)
    obst, on = jnp.asarray(q.obst[j]), jnp.asarray(q.on[j])
    nv = len(fl.variant_labels)
    out = {}
    for k in range(0, k_end, stride):
        x = jnp.asarray(np.concatenate([q.pos[k, j], q.vel[k, j], q.quat[k, j]]), jnp.float32)
        chp = jnp.asarray(q.ch[k, j])
        un = jnp.asarray(u_nom[k, j], jnp.float32)
        comm = fv(x, obst, chp, on)
        plans, ok = [], []
        for kk in range(nv):
            c = cand(x, un, obst, chp, on, comm, kk, jnp.float32(0.0))
            plans.append(np.asarray(c["states"][:, :2]))
            ok.append(bool(c["path_ok"]) and bool(c["terminal_ok"]))
        kc = int(np.clip(var[k, j], 0, nv - 1))
        c = cand(x, un, obst, chp, on, comm, kc, jnp.float32(q.lam[k, j]))
        out[k] = (np.stack(plans), np.asarray(ok), np.asarray(c["states"][:, :2]),
                  np.asarray(c["chasers"][:, :, :2]).transpose(1, 0, 2)[np.asarray(q.on[j]) > 0.5])
    return out


def t_dir(q: Traj):
    return q.dir


def plans_overlay(plans, theme="light", stride=3):
    from matplotlib.collections import LineCollection
    th = THEMES[theme]

    def draw(ax, i, k):
        kk = (k // stride) * stride
        if kk not in plans:
            return []
        pl, ok, chosen, chs = plans[kk]
        arts = []
        for good in (False, True):
            seg = [pl[v] for v in range(len(pl)) if ok[v] == good]
            if seg:
                lc = LineCollection(seg, colors=th["ok"] if good else th["bad"], linewidths=1.1 if good else 0.9,
                                    alpha=0.75 if good else 0.45, zorder=7, linestyles="-" if good else (0, (3, 2)))
                arts.append(ax.add_collection(lc))
        arts += ax.plot(chosen[:, 0], chosen[:, 1], color=th["drone"], lw=2.6, alpha=0.9, zorder=8)
        for c in chs:
            arts += ax.plot(c[:, 0], c[:, 1], color=th["chaser"], lw=1.4, ls=":", alpha=0.8, zorder=7)
        n_ok = int(ok.sum())
        arts.append(ax.text(0.975, 0.95, f" 회피 계획 {n_ok}/{len(ok)} 인증 ", fontsize=9, color="white", fontweight="bold",
                            ha="right", va="top", zorder=20, transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", ec="none", alpha=0.92,
                                                                       fc=th["ok"] if n_ok else th["bad"])))
        return arts
    return draw


# ------------------------------------------------------------------------------------------------ episode picking
PICK_CATS = ("closest", "active", "detour", "climb", "fastest", "longest", "failed", "contrast", "strike")


def episode_rows(traj: Traj):
    """(episode, outcome, time [s], closest chaser [m], filter-active fraction, y extent [m], climb [m]) per episode."""
    rows = []
    for e, j in sorted(traj.idx.items()):
        k = int(traj.end[j])
        d = np.inf
        for c in range(traj.ch.shape[2]):
            if traj.on[j, c] > 0.5:
                d = min(d, float(np.min(np.linalg.norm(traj.pos[:k, j] - traj.ch[:k, j, c], axis=1))))
        act = float(np.mean([traj.status(j, kk) > 0 for kk in range(k)])) if (traj.lam is not None and traj.mode != "none") else 0.0
        rows.append((e, int(traj.outcome[j]), k * traj.dt, d, act, float(np.ptp(traj.pos[:k, j, 1])),
                     float(abs(traj.tg[k - 1, j, 2] - traj.pos[0, j, 2]))))
    return rows


def ranked(trajs, cat):
    """Episodes of a category, best first. Most use the last trajectory; "contrast": the first fails, the last strikes."""
    last = trajs[-1]
    rows = episode_rows(last)
    st = [r for r in rows if r[1] == 0]
    if cat == "contrast":
        first = trajs[0]
        st = [r for r in st if r[0] in first.idx and int(first.outcome[first.idx[r[0]]]) != 0]
        return [r[0] for r in sorted(st, key=lambda r: r[3])]
    key = {"closest": lambda r: r[3], "active": lambda r: -r[4], "detour": lambda r: -r[5], "climb": lambda r: -r[6],
           "fastest": lambda r: r[2],
           "longest": lambda r: -r[2], "strike": lambda r: r[0]}
    if cat == "failed":
        return [r[0] for r in rows if r[1] != 0]
    common = [r for r in st if all(r[0] in q.idx for q in trajs)]
    return [r[0] for r in sorted(common, key=key[cat])]


def resolve_episodes(spec: str, trajs, n=1):
    """'12' or '3,7,9' or 'pick:CAT[:RANK]' (one) or 'pick:CAT:A-B' (a range of ranks)."""
    if not spec.startswith("pick:"):
        return [int(x) for x in spec.split(",")]
    parts = spec.split(":")
    cat = parts[1]
    if cat not in PICK_CATS:
        raise SystemExit(f"unknown pick category {cat}: {PICK_CATS}")
    lst = ranked(trajs, cat)
    if not lst:
        raise SystemExit(f"no episode in category {cat}")
    if len(parts) > 2 and "-" in parts[2]:
        a, b = map(int, parts[2].split("-"))
        return lst[a:b + 1]
    r = int(parts[2]) if len(parts) > 2 else 0
    return lst[min(r, len(lst) - 1):min(r, len(lst) - 1) + n]


def pick(traj: Traj, n=6):
    rows = {r[0]: r for r in episode_rows(traj)}
    for cat in PICK_CATS:
        if cat == "contrast":
            continue
        lst = ranked([traj], cat)[:n]
        print(f"{cat}: " + ", ".join(f"{e} ({OUTCOME_KO[rows[e][1]]} {rows[e][2]:.1f} s, chaser {rows[e][3]:.1f} m, "
                                     f"filter {rows[e][4]:.0%})" for e in lst))


# ------------------------------------------------------------------------------------------------ Phase I
def phase1_examples(run: str, split="test"):
    import jax
    import jax.numpy as jnp

    from ps2rl.envs.strike_scene import StrikeConfig, build_model, split_scene
    from ps2rl.phase1_sa.strike_sa import (REGION_NAMES, StrikeRegionConfig, heldout_sets, make_actor_backup,
                                           make_lqr_backup, make_rollout)
    from ps2rl.utils.policy import _coerce_actor_cfg
    import pickle
    run = Path(run)
    cfgs = json.loads((run / "configs.json").read_text())
    cfg, rc = StrikeConfig.from_dict(cfgs["strike"]), StrikeRegionConfig.from_dict(cfgs["region"])
    with open(run / "strike_backup_policy_actor.pkl", "rb") as f:
        ck = pickle.load(f)
    m = build_model(cfg)
    sets = heldout_sets(m, rc, split)
    out = {"cfg": cfg, "m": m, "regions": {}}
    for backup, params, tag in ((make_lqr_backup(m), None, "lqr"), (make_actor_backup(m, _coerce_actor_cfg(ck["actor_cfg"])),
                                                                     ck["actor_params"], "rl")):
        roll = jax.jit(jax.vmap(make_rollout(m, backup, trajectory=True), in_axes=(None, 0, 0)))
        for name in REGION_NAMES:
            v = sets[name]
            ok, hf, k_end, xs = map(np.asarray, roll(params, jnp.asarray(v[:, :10]), jnp.asarray(v[:, 10:])))
            r = out["regions"].setdefault(name, {"v": v})
            r[tag] = {"ok": ok, "fail": hf, "k_end": k_end, "xs": xs}
    for name, r in out["regions"].items():
        scenes = [tuple(map(np.asarray, split_scene(cfg, jnp.asarray(s)))) for s in r["v"][:, 10:]]
        r["obst"] = np.stack([s[0] for s in scenes])
        r["balls"] = np.stack([s[1] for s in scenes])
    return out


P1_NAMES = {"general": "일반 상태", "blocked": "장애물 정면", "chaser": "요격기 정면", "shell": "기준 집합 근처",
            "escape": "요격기 근접 · 저속"}


def p1_candidates(ex, region, pick_rule="lqr_fail", order="visual"):
    r = ex["regions"][region]
    lqr, rl = r["lqr"], r["rl"]
    mask = {"lqr_fail": ~lqr["ok"] & rl["ok"], "both_ok": lqr["ok"] & rl["ok"], "rl_ok": rl["ok"]}[pick_rule]
    cand = np.where(mask)[0]
    if order == "speed":                                                     # fastest initial states first
        cand = cand[np.argsort(-np.linalg.norm(r["v"][cand, 3:6], axis=1))]
    elif order == "path":                                                    # longest learned path first
        xs = rl["xs"][cand, :, :2]
        cand = cand[np.argsort(-np.sum(np.linalg.norm(np.diff(xs, axis=1), axis=2), axis=1))]
    elif order == "visual":    # LQR fails in the top view (chaser zone / obstacle) first, largest LQR-learned split first
        vis = np.array([p1_failure(ex, region, i, lqr) in ("chaser", "obstacle") for i in cand], bool)
        split = np.max(np.linalg.norm(rl["xs"][cand, :, :2] - lqr["xs"][cand, :, :2], axis=2), axis=1)
        cand = cand[np.lexsort((-split, ~vis))]
    return cand


def p1_failure(ex, region, i, d):
    """Short label of how a rollout ended: base set reached, chaser zone, obstacle, floor, or out of time."""
    if bool(d["ok"][i]):
        return "ok"
    if not bool(d["fail"][i]):
        return "time"
    r, cfg = ex["regions"][region], ex["cfg"]
    p = d["xs"][i][int(d["k_end"][i])][:3]
    for b in r["balls"][i]:
        if b[3] > 0.5 and np.linalg.norm(p - b[:3]) <= float(cfg.ball_radius) + 0.05:
            return "chaser"
    if p[2] <= float(cfg.z_floor) + 0.15:
        return "floor"
    return "obstacle"


P1_END = {"ok": ("✓ 안전 정지 (기준 집합 도달)", "ok"), "time": ("✗ 안전한 호버 지점에 못 감", "warn"),
          "chaser": ("✗ 요격 범위 진입", "bad"), "floor": ("✗ 지면 충돌", "bad"), "obstacle": ("✗ 장애물 충돌", "bad")}


def _p1_axes(ax, th, sk, r, i, R, half, show_legend_zone=True):
    from matplotlib.patches import Circle
    ax.set_facecolor(th["bg"])
    ax.set_xlim(-half, half)
    ax.set_ylim(-half * 0.95, half * 0.95)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    for g in np.arange(-20, 20.1, 2.5):
        ax.plot([g, g], [-half, half], color=th["grid"], lw=0.5, zorder=1)
        ax.plot([-half, half], [g, g], color=th["grid"], lw=0.5, zorder=1)
    for o in r["obst"][i]:
        if o[6] > 0.5:
            sk.building(ax, o)
    for b in r["balls"][i]:
        if b[3] > 0.5:
            ax.add_patch(Circle(b[:2], R, fc=th["danger"], ec=th["chaser"], lw=1.0, ls="--", zorder=4))
    v0 = r["v"][i, 3:6]
    if np.linalg.norm(v0[:2]) > 0.2:                                         # initial velocity, 0.6 s ahead
        ax.annotate("", xy=v0[:2] * 0.6, xytext=(0, 0), zorder=5,
                    arrowprops=dict(arrowstyle="-|>", color=th["sub"], lw=1.6, mutation_scale=12, alpha=0.8))


def render_phase1(ex, region, index, out, *, theme="light", chaser="missile", building="glass", speed=0.4,
                  pick_rule="lqr_fail", order="visual", dpi=100, mp4=False, hold=1.5, half=9.0):
    plt = setup_font()
    from matplotlib import animation
    from matplotlib.patches import Circle
    th = THEMES[theme]
    sk = Skins(theme, chaser, building, "vehicle")
    r = ex["regions"][region]
    cfg, m = ex["cfg"], ex["m"]
    lqr, rl = r["lqr"], r["rl"]
    cand = p1_candidates(ex, region, pick_rule, order)
    if len(cand) == 0:
        raise SystemExit(f"no {pick_rule} example in region {region}")
    i = int(cand[min(index, len(cand) - 1)])
    balls = r["balls"][i]
    R = float(cfg.ball_radius)
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 5.6), facecolor=th["panel"])
    fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.075, wspace=0.05)
    fig.text(0.02, 0.95, f"Phase I 회복 정책 — {P1_NAMES.get(region, region)}", fontsize=13, fontweight="bold",
             color=th["text"], va="center")
    fig.text(0.02, 0.895, f"같은 초기 상태, {cfg.T:g} s 안에 멈춰 떠 있어야 함: 호버 LQR(제동만) vs 학습된 회복 정책"
             f"   (속도 ×{speed:g})", fontsize=9.5, color=th["sub"], va="center")
    fig.text(0.02, 0.03, f"빨간 점선 원: 요격기가 {cfg.T:g} s 안에 닿을 수 있는 범위    초록 원: 호버 안정 영역 (기준 집합)"
             "    회색 화살표: 초기 속도", fontsize=8.5, color=th["sub"], va="center")
    data = [("호버 LQR (제동)", lqr), ("학습된 회복 정책", rl)]
    for ax, (lab, d) in zip(axs, data):
        _p1_axes(ax, th, sk, r, i, R, half)
        ax.set_title(lab, loc="left", fontsize=11, fontweight="bold", color=th["text"])
    xs_all = [d["xs"][i] for _, d in data]
    ends = [p1_failure(ex, region, i, d) for _, d in data]
    n_steps = xs_all[0].shape[0]
    n_hold = int(round(hold / cfg.dt * speed))
    frames = list(range(n_steps)) + [n_steps - 1] * n_hold
    dyn = []

    def draw(fi):
        for a_ in dyn:
            a_.remove()
        dyn.clear()
        k = frames[fi]
        for ax, (lab, d), xs, end in zip(axs, data, xs_all, ends):
            k_end = int(d["k_end"][i])
            kk = min(k, k_end) if end not in ("ok", "time") else k
            p = xs[kk]
            dyn.extend(fade_line(ax, xs[:kk + 1], th["trail"], lw=2.6, alpha_min=0.25, z=6))
            for b in balls:
                if b[3] > 0.5:
                    dyn.extend(sk.chaser(ax, b, np.arctan2(p[1] - b[1], p[0] - b[0]), k * cfg.dt))
            dyn.extend(sk.quad(ax, p, quat_yaw(p[6:10])))
            v = float(np.linalg.norm(p[3:6]))
            dyn.append(ax.text(-half + 0.4, half * 0.95 - 0.5, f" t = {kk * cfg.dt:.2f} s   속도 {v:.1f} m/s ", fontsize=9,
                               color=th["text"], va="top", zorder=20,
                               bbox=dict(boxstyle="round,pad=0.25", fc=th["panel"], ec=th["grid"], alpha=0.9)))
            if k >= k_end:
                if end == "ok":
                    dyn.append(ax.add_patch(Circle(xs[k_end][:2], float(m.lqr.rho_xy) + 0.6, fc=th["safe"], ec=th["ok"],
                                                   lw=1.6, ls="--", zorder=5)))
                elif end != "time":
                    dyn.extend(ax.plot([p[0]], [p[1]], "x", color=th["bad"], ms=18, mew=3.5, zorder=24))
                msg, key = P1_END[end]
                dyn.append(ax.text(0, -half * 0.95 + 1.0, f" {msg} ", fontsize=12, fontweight="bold", color="white",
                                   ha="center", va="center", zorder=25,
                                   bbox=dict(boxstyle="round,pad=0.4", fc=th[key], ec="none", alpha=0.92)))
        return dyn

    fps = speed / cfg.dt
    ani = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps, blit=False)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ani.save(out, writer=animation.PillowWriter(fps=fps), dpi=dpi, savefig_kwargs={"facecolor": th["panel"]})
    finish(out, mp4)
    plt.close(fig)
    return out, i


def render_phase1_3d(ex, region, index, out, *, theme="light", chaser="missile", building="glass", speed=0.4,
                     pick_rule="lqr_fail", order="visual", dpi=100, mp4=False, hold=1.5, half=9.0, elev=28.0,
                     azim=(-128.0, -100.0), zoom=0.9, building_top=8.0, obj_scale=1.2, chaser_scale=1.0):
    """Phase I side by side in 3-D: the chaser's reach (sphere), the hover LQR and the learned backup."""
    plt = setup_font()
    from matplotlib import animation
    th = THEMES[theme]
    sk = Skins3D(theme, chaser, building, "vehicle", obj_scale, chaser_scale)
    r = ex["regions"][region]
    cfg, m = ex["cfg"], ex["m"]
    cand = p1_candidates(ex, region, pick_rule, order)
    if len(cand) == 0:
        raise SystemExit(f"no {pick_rule} example in region {region}")
    i = int(cand[min(index, len(cand) - 1)])
    balls, R = r["balls"][i], float(cfg.ball_radius)
    data = [("호버 LQR (제동)", r["lqr"]), ("학습된 회복 정책", r["rl"])]
    xs_all = [d["xs"][i] for _, d in data]
    ends = [p1_failure(ex, region, i, d) for _, d in data]
    z_top = max(building_top, max(float(xs[:, 2].max()) for xs in xs_all) + 1.0)
    pw, cell_h, top_h = 6.4, 4.6, 0.95
    fig_w, fig_h = 2 * pw, cell_h + top_h + 0.35
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=th["panel"])
    fig.text(0.015, 1 - 0.25 / fig_h, f"Phase I 회복 정책 — {P1_NAMES.get(region, region)} (3D)", fontsize=13, fontweight="bold",
             color=th["text"], va="center")
    fig.text(0.015, 1 - 0.55 / fig_h, f"같은 초기 상태, {cfg.T:g} s 안에 멈춰 떠 있어야 함 · 빨간 구: 요격기가 {cfg.T:g} s 안에 닿을 수 "
             f"있는 범위 · 초록 구: 안전 정지   (속도 ×{speed:g})", fontsize=9.5, color=th["sub"], va="center")
    axes, texts = [], []
    a = np.linspace(0, 2 * np.pi, 24)
    for c, (lab, d) in enumerate(data):
        yc = (cell_h / 2 + 0.35)
        ax = fig.add_axes([c * pw / fig_w, (yc - pw / 2) / fig_h, pw / fig_w, pw / fig_h], projection="3d")
        ax.patch.set_alpha(0.0)
        sk.ground(ax, (-half, half), (-half, half), z_top, elev, azim[0], zoom, border=False, grid=2.5)
        sk.buildings3d(ax, [o for o in r["obst"][i] if max(abs(o[0]), abs(o[1])) <= half + 1.0], building_top)
        for b in balls:
            if b[3] > 0.5:
                u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:11j]
                ax.plot_surface(b[0] + R * np.cos(u) * np.sin(v), b[1] + R * np.sin(u) * np.sin(v), b[2] + R * np.cos(v),
                                color=th["chaser"], alpha=0.08, linewidth=0, shade=False, zorder=4)
                ax.plot(b[0] + R * np.cos(a), b[1] + R * np.sin(a), np.full(24, b[2]), color=th["chaser"], lw=1.0, ls="--",
                        alpha=0.7, zorder=4.1)
        v0 = r["v"][i, 3:6]
        if np.linalg.norm(v0) > 0.2:
            p0 = xs_all[0][0, :3]
            ax.plot(*np.stack([p0, p0 + 0.6 * v0], 1), color=th["sub"], lw=1.6, alpha=0.8, zorder=5)
        axes.append(ax)
        fig.text((c * pw + 0.2) / fig_w, 1 - 0.85 / fig_h, lab, fontsize=11, fontweight="bold", color=th["text"], va="center")
        hud = fig.text(((c + 1) * pw - 0.2) / fig_w, 1 - 0.85 / fig_h, "", fontsize=9, color=th["sub"], va="center", ha="right")
        badge = fig.text((c + 0.5) * pw / fig_w, 0.30 / fig_h, "", fontsize=12, fontweight="bold", color="white", ha="center",
                         va="center")
        texts.append((hud, badge))
    n_steps = xs_all[0].shape[0]
    n_hold = int(round(hold / cfg.dt * speed))
    frames = list(range(n_steps)) + [n_steps - 1] * n_hold
    dyn = []

    def draw(fi):
        for a_ in dyn:
            a_.remove()
        dyn.clear()
        k = frames[fi]
        az = azim[0] + (azim[1] - azim[0]) * min(fi / max(n_steps - 1, 1), 1.0)
        for ax, (lab, d), xs, end, (hud, badge) in zip(axes, data, xs_all, ends, texts):
            ax.view_init(elev=elev, azim=az)
            k_end = int(d["k_end"][i])
            kk = min(k, k_end) if end not in ("ok", "time") else k
            p = xs[kk]
            dyn.extend(fade_line3d(ax, xs[:kk + 1, :3], th["trail"], lw=2.6, alpha_min=0.25, z=6))
            for b in balls:
                if b[3] > 0.5:
                    dyn.extend(sk.chaser3d(ax, b[:3], p[:3] - b[:3], k * cfg.dt))
            dyn.extend(sk.quad3d(ax, p[:3], p[6:10]))
            hud.set_text(f"t = {kk * cfg.dt:.2f} s   속도 {np.linalg.norm(p[3:6]):.1f} m/s   고도 {p[2]:.1f} m")
            if k >= k_end:
                if end == "ok":
                    rr = float(m.lqr.rho_xy) + 0.6
                    u, v = np.mgrid[0:2 * np.pi:16j, 0:np.pi:9j]
                    c0 = xs[k_end][:3]
                    dyn.append(ax.plot_surface(c0[0] + rr * np.cos(u) * np.sin(v), c0[1] + rr * np.sin(u) * np.sin(v),
                                               c0[2] + rr * np.cos(v), color=th["ok"], alpha=0.15, linewidth=0, shade=False,
                                               zorder=5))
                elif end != "time":
                    dyn.append(ax.scatter(*p[:3], s=260, marker="x", c=th["bad"], linewidths=3.5, zorder=24, depthshade=False))
                msg, key = P1_END[end]
                badge.set_text(f" {msg} ")
                badge.set_bbox(dict(boxstyle="round,pad=0.4", fc=th[key], ec="none", alpha=0.92))
            else:
                badge.set_text("")
                badge.set_bbox(None)
        return dyn

    fps = speed / cfg.dt
    ani = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps, blit=False)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ani.save(out, writer=animation.PillowWriter(fps=fps), dpi=dpi, savefig_kwargs={"facecolor": th["panel"]})
    finish(out, mp4)
    plt.close(fig)
    return out, i


def render_phase1_grid(ex, picks, out, *, theme="light", chaser="missile", building="glass", speed=0.4, ncols=3,
                       dpi=90, mp4=False, hold=1.5, half=8.0, title="Phase I: 학습된 회복 정책 (파랑) vs 호버 LQR (빨강)"):
    """Several test states at once, LQR (ghost) and learned backup overlaid; picks: [(region, test index), ...]."""
    plt = setup_font()
    from matplotlib import animation
    from matplotlib.patches import Circle
    th = THEMES[theme]
    sk = Skins(theme, chaser, building, "vehicle")
    sk_lqr = Skins(theme, chaser, building, "vehicle")
    cfg, m = ex["cfg"], ex["m"]
    R = float(cfg.ball_radius)
    nrows = int(np.ceil(len(picks) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.25 * nrows + 0.75), facecolor=th["panel"], squeeze=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.62 / (3.25 * nrows + 0.75), bottom=0.25 / (3.25 * nrows + 0.75),
                        wspace=0.04, hspace=0.16)
    fig.text(0.01, 1 - 0.27 / (3.25 * nrows + 0.75), title, fontsize=12.5, fontweight="bold", color=th["text"], va="center")
    fig.text(0.01, 0.12 / (3.25 * nrows + 0.75), f"{cfg.T:g} s 안에 멈춰 떠 있어야 함 · 빨간 점선 원: 요격기 도달 범위 · "
             f"속도 ×{speed:g}", fontsize=8.5, color=th["sub"], va="center")
    lqr_col = th["bad"]
    cells = []
    for ax, (region, i) in zip(axs.ravel(), picks):
        r = ex["regions"][region]
        _p1_axes(ax, th, sk, r, i, R, half)
        ax.set_title(P1_NAMES.get(region, region), loc="left", fontsize=10, fontweight="bold", color=th["text"])
        cells.append((ax, r, i, p1_failure(ex, region, i, r["lqr"]), p1_failure(ex, region, i, r["rl"])))
    for ax in axs.ravel()[len(picks):]:
        ax.axis("off")
    n_steps = cells[0][1]["rl"]["xs"].shape[1]
    n_hold = int(round(hold / cfg.dt * speed))
    frames = list(range(n_steps)) + [n_steps - 1] * n_hold
    dyn = []

    def draw(fi):
        for a_ in dyn:
            a_.remove()
        dyn.clear()
        k = frames[fi]
        for ax, r, i, e_l, e_r in cells:
            for d, end, col, al, ghost in ((r["lqr"], e_l, lqr_col, 0.55, True), (r["rl"], e_r, th["trail"], 1.0, False)):
                xs, k_end = d["xs"][i], int(d["k_end"][i])
                kk = min(k, k_end) if end not in ("ok", "time") else k
                p = xs[kk]
                dyn.extend(fade_line(ax, xs[:kk + 1], col, lw=2.2, alpha_min=0.2, alpha_max=0.9 * al, z=6,
                                     ls="--" if ghost else "-"))
                dyn.extend((sk_lqr if ghost else sk).quad(ax, p, quat_yaw(p[6:10]), size=0.8, alpha=al,
                                                          color=lqr_col if ghost else None))
                if k >= k_end and end not in ("ok", "time"):
                    dyn.extend(ax.plot([p[0]], [p[1]], "x", color=col, ms=13, mew=3, zorder=24))
                if k >= k_end and end == "ok" and not ghost:
                    dyn.append(ax.add_patch(Circle(xs[k_end][:2], float(m.lqr.rho_xy) + 0.5, fc=th["safe"], ec=th["ok"],
                                                   lw=1.3, ls="--", zorder=5)))
            for b in r["balls"][i]:
                if b[3] > 0.5:
                    p = r["rl"]["xs"][i][min(k, n_steps - 1)]
                    dyn.extend(sk.chaser(ax, b, np.arctan2(p[1] - b[1], p[0] - b[0]), k * cfg.dt, size=0.85))
            if k >= max(int(r["rl"]["k_end"][i]), int(r["lqr"]["k_end"][i])) or fi >= len(frames) - n_hold:
                tl = "LQR ✓" if e_l == "ok" else "LQR ✗"
                tr = "학습 ✓" if e_r == "ok" else "학습 ✗"
                dyn.append(ax.text(-half + 0.4, -half * 0.95 + 0.9, f" {tl} ", fontsize=9, fontweight="bold", color="white",
                                   va="center", zorder=25, bbox=dict(boxstyle="round,pad=0.3", ec="none", alpha=0.92,
                                                                     fc=th["ok"] if e_l == "ok" else lqr_col)))
                dyn.append(ax.text(half - 0.4, -half * 0.95 + 0.9, f" {tr} ", fontsize=9, fontweight="bold", color="white",
                                   va="center", ha="right", zorder=25,
                                   bbox=dict(boxstyle="round,pad=0.3", ec="none", alpha=0.92,
                                             fc=th["ok"] if e_r == "ok" else th["bad"])))
        return dyn

    fps = speed / cfg.dt
    ani = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps, blit=False)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ani.save(out, writer=animation.PillowWriter(fps=fps), dpi=dpi, savefig_kwargs={"facecolor": th["panel"]})
    finish(out, mp4)
    plt.close(fig)
    return out


# ------------------------------------------------------------------------------------------------ skin sheet
def render_skins(out, dpi=130):
    plt = setup_font()
    fig, axs = plt.subplots(2, 4, figsize=(12, 6.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02, wspace=0.04, hspace=0.18)
    combos = [("light", "missile", "glass", "vehicle"), ("light", "delta", "glass", "bullseye"),
              ("light", "missile", "concrete", "vehicle"), ("light", "delta", "concrete", "bullseye"),
              ("dark", "missile", "concrete", "vehicle"), ("dark", "delta", "glass", "bullseye"),
              ("dark", "missile", "glass", "vehicle"), ("dark", "delta", "concrete", "bullseye")]
    box = np.array([5.0, 3.0, 2.2, 1.5, 0.3, 0.0, 1.0])
    cyl = np.array([10.5, -2.0, 0.0, 0.0, 0.0, 1.6, 1.0])
    for ax, (thn, ch, bd, tg) in zip(axs.ravel(), combos):
        th = THEMES[thn]
        sk = Skins(thn, ch, bd, tg)
        ax.set_facecolor(th["bg"])
        ax.set_xlim(0, 16)
        ax.set_ylim(-5.5, 6)
        ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
        sk.building(ax, box)
        sk.building(ax, cyl)
        fade_line(ax, np.stack([np.linspace(1, 6.5, 30), -3.6 + 0.3 * np.sin(np.linspace(0, 3, 30))], 1), th["trail"], lw=2.4)
        sk.quad(ax, np.array([7.0, -3.5]), 0.2)
        fade_line(ax, np.stack([np.linspace(15, 12.2, 20), np.linspace(-4.5, 1.5, 20)], 1), th["chaser"], lw=2.2, alpha_max=0.35)
        sk.chaser(ax, np.array([12.0, 2.0]), np.arctan2(6.0, -2.8), 0.3)
        sk.target(ax, np.array([13.5, 4.3]), 0.4, 0.2)
        ax.set_title(f"{thn} · {ch} · {bd} · {tg}", fontsize=9, loc="left")
    fig.suptitle("스킨 옵션 (--theme, --chaser, --building, --target)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def skin_args(q):
        q.add_argument("--theme", choices=tuple(THEMES), default="light")
        q.add_argument("--chaser", choices=("missile", "delta"), default="missile")
        q.add_argument("--building", choices=("concrete", "glass"), default="glass")
        q.add_argument("--mp4", type=int, default=0, help="1: also write an MP4 next to the GIF (ffmpeg)")
        q.add_argument("--dpi", type=int, default=100)
        q.add_argument("--optimize", type=int, default=1, help="1: re-encode the GIF with ffmpeg (one palette, about half the size)")
        q.add_argument("--chaser_scale", type=float, default=1.0, help="chaser size factor")

    def view_args(q):
        q.add_argument("--view", choices=("2d", "3d"), default="2d", help="top view (+ altitude strip) or 3-D")
        q.add_argument("--elev", type=float, default=None, help="3-D camera elevation [deg] (26 Phase II, 28 Phase I)")
        q.add_argument("--azim", default="", help="3-D camera azimuth sweep 'A0,A1' [deg] (-122,-92 Phase II; -128,-100 Phase I)")
        q.add_argument("--zoom", type=float, default=None, help="3-D zoom (0.95 Phase II, 0.9 Phase I)")
        q.add_argument("--obj_scale", type=float, default=None, help="3-D size of drone, chaser and target (1.4 Phase II, 1.2 Phase I)")

    def render_args(q):
        q.add_argument("--out", required=True)
        q.add_argument("--title", default="")
        q.add_argument("--subtitle", default="")
        q.add_argument("--target", choices=("vehicle", "bullseye"), default="vehicle")
        q.add_argument("--alt", type=int, default=1, help="1: altitude strip (side view) under each panel")
        q.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
        q.add_argument("--stride", type=int, default=3, help="simulation steps per frame")
        q.add_argument("--hold", type=float, default=1.6, help="[s] frozen end frame (in playback time at speed 1)")
        q.add_argument("--panel_width", type=float, default=None, help="[in] width of one panel (8.6 one column, 5.6 grid)")
        q.add_argument("--status", type=int, default=1, help="1: the filter status chip")
        skin_args(q)
        view_args(q)

    e = sub.add_parser("episode", help="one Phase II episode, one panel per --traj (same scene)")
    e.add_argument("--traj", action="append", required=True, help="DIR[:MODE[:LABEL]] (eval_strike_filter.py --save_traj output)")
    e.add_argument("--episode", required=True,
                   help="scene index, or pick:CAT[:RANK] with CAT in " + ", ".join(PICK_CATS)
                        + " (contrast: the first --traj fails, the last strikes)")
    e.add_argument("--ncols", type=int, default=1)
    render_args(e)
    mo = sub.add_parser("montage", help="several Phase II episodes of one run on a grid")
    mo.add_argument("--traj", required=True)
    mo.add_argument("--episodes", required=True, help="'3,7,9,12' or pick:CAT:A-B (ranks A..B of a category)")
    mo.add_argument("--ncols", type=int, default=2)
    mo.add_argument("--labels", default="장면 {e}", help="panel label format ({e}: scene index)")
    render_args(mo)
    pl = sub.add_parser("plans", help="one Phase II episode with the escape plans the filter checks (green certified, red not)")
    pl.add_argument("--traj", required=True)
    pl.add_argument("--episode", required=True)
    pl.add_argument("--backup_run", required=True, help="the Phase-I backup run of the replay")
    pl.add_argument("--task", default=str(REPO / "docker_batch/configs/strike_final_ps2_search.json"),
                    help="task file whose tracker.filter_guidance is the replay's filter")
    render_args(pl)
    k = sub.add_parser("pick", help="list episodes worth animating")
    k.add_argument("--traj", required=True)
    k.add_argument("--n", type=int, default=6)
    f = sub.add_parser("phase1", help="Phase I: hover LQR vs the learned backup on a held-out test state")
    f.add_argument("--run", required=True, help="Phase-I run directory (strike_backup_policy_actor.pkl, configs.json)")
    f.add_argument("--region", default="blocked", help="general | blocked | chaser | shell | escape")
    f.add_argument("--index", type=int, default=0)
    f.add_argument("--pick", choices=("lqr_fail", "both_ok", "rl_ok"), default="lqr_fail")
    f.add_argument("--order", choices=("visual", "speed", "path", "index"), default="visual",
                   help="order of the candidate test states: LQR failure visible from above and largest LQR-learned split "
                        "first, fastest first, longest learned path first, or test-set order")
    f.add_argument("--speed", type=float, default=0.4)
    f.add_argument("--out", required=True)
    skin_args(f)
    view_args(f)
    g = sub.add_parser("phase1grid", help="Phase I: several test states at once, LQR and learned backup overlaid")
    g.add_argument("--run", required=True)
    g.add_argument("--picks", default="blocked:0,chaser:0,escape:0,general:0,blocked:1,chaser:1",
                   help="REGION:K,... (K-th state where the LQR fails and the learned backup recovers, --order)")
    g.add_argument("--order", choices=("visual", "speed", "path", "index"), default="visual")
    g.add_argument("--ncols", type=int, default=3)
    g.add_argument("--speed", type=float, default=0.4)
    g.add_argument("--out", required=True)
    skin_args(g)
    pa = sub.add_parser("phase1all", help="Phase I set: the overview grid and N side-by-side GIFs per region (one rollout pass)")
    pa.add_argument("--run", required=True)
    pa.add_argument("--outdir", required=True)
    pa.add_argument("--n", type=int, default=2, help="side-by-side GIFs per region")
    pa.add_argument("--regions", default="blocked,chaser,escape,general")
    pa.add_argument("--speed", type=float, default=0.4)
    pa.add_argument("--views", default="2d", help="2d, 3d or 2d,3d (3-D files end in _3d)")
    skin_args(pa)
    view_args(pa)
    s = sub.add_parser("skins", help="the skin options on one sheet")
    s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    global OPTIMIZE
    OPTIMIZE = bool(getattr(a, "optimize", 1))
    def cam3d():
        kw = {}
        if getattr(a, "elev", None) is not None:
            kw["elev"] = a.elev
        if getattr(a, "azim", ""):
            kw["azim"] = tuple(float(x) for x in a.azim.split(","))
        if getattr(a, "zoom", None) is not None:
            kw["zoom"] = a.zoom
        if getattr(a, "obj_scale", None) is not None:
            kw["obj_scale"] = a.obj_scale
        return kw

    def rkw():
        kw = dict(theme=a.theme, chaser=a.chaser, building=a.building, target=a.target, speed=a.speed, stride=a.stride,
                  dpi=a.dpi, hold=a.hold, title=a.title, subtitle=a.subtitle, mp4=bool(a.mp4), panel_width=a.panel_width,
                  show_status=bool(a.status), chaser_scale=a.chaser_scale)
        if a.view == "3d":
            kw.update(cam3d())
        else:
            kw["alt"] = bool(a.alt)
        return kw

    def panels_out(panels, out, ncols):
        return (render3d_panels if a.view == "3d" else render_panels)(panels, out, ncols=ncols, **rkw())
    if a.cmd == "episode":
        trajs = [Traj(t) for t in a.traj]
        ep = resolve_episodes(a.episode, trajs)[0]
        out = panels_out([(q, ep, q.label) for q in trajs], a.out, a.ncols)
        print("wrote", out, "(scene", ep, ")")
    elif a.cmd == "montage":
        q = Traj(a.traj)
        eps = resolve_episodes(a.episodes, [q])
        out = panels_out([(q, e, a.labels.format(e=e)) for e in eps], a.out, a.ncols)
        print("wrote", out, "(scenes", eps, ")")
    elif a.cmd == "plans":
        q = Traj(a.traj)
        ep = resolve_episodes(a.episode, [q])[0]
        plans = filter_plans(q, ep, a.backup_run, a.task, a.stride)
        kw = rkw()
        kw["subtitle"] = kw["subtitle"] or ("초록: 인증된 회피 계획 · 빨강 점선: 실패한 계획 · 파랑: 선택된 계획 (3 s 예측) · "
                                            "빨강 점: 요격기 예측 경로")
        kw.pop("elev", None), kw.pop("azim", None), kw.pop("zoom", None), kw.pop("obj_scale", None)
        kw.setdefault("alt", bool(a.alt))
        out = render_panels([(q, ep, q.label)], a.out, ncols=1, overlay=plans_overlay(plans, a.theme, a.stride), **kw)
        print("wrote", out, "(scene", ep, ")")
    elif a.cmd == "pick":
        pick(Traj(a.traj), a.n)
    elif a.cmd == "phase1":
        ex = phase1_examples(a.run)
        f1 = render_phase1_3d if a.view == "3d" else render_phase1
        extra = dict(cam3d(), chaser_scale=a.chaser_scale) if a.view == "3d" else {}
        out, i = f1(ex, a.region, a.index, a.out, theme=a.theme, chaser=a.chaser, building=a.building,
                    speed=a.speed, pick_rule=a.pick, order=a.order, dpi=a.dpi, mp4=bool(a.mp4), **extra)
        print("wrote", out, "(test state", i, ")")
    elif a.cmd == "phase1grid":
        ex = phase1_examples(a.run)
        picks = []
        for tok in a.picks.split(","):
            reg, k = tok.split(":")
            cand = p1_candidates(ex, reg, "lqr_fail", a.order)
            picks.append((reg, int(cand[min(int(k), len(cand) - 1)])))
        print("wrote", render_phase1_grid(ex, picks, a.out, theme=a.theme, chaser=a.chaser, building=a.building,
                                          speed=a.speed, ncols=a.ncols, dpi=a.dpi, mp4=bool(a.mp4)), picks)
    elif a.cmd == "phase1all":
        ex = phase1_examples(a.run)
        od = Path(a.outdir)
        sk = dict(theme=a.theme, chaser=a.chaser, building=a.building, speed=a.speed, dpi=a.dpi, mp4=bool(a.mp4))
        regions = a.regions.split(",")
        picks = []
        for k in range(2):
            for reg in regions[:3]:
                cand = p1_candidates(ex, reg)
                if len(cand) > k:
                    picks.append((reg, int(cand[k])))
        if "2d" in a.views.split(","):
            print("wrote", render_phase1_grid(ex, picks, od / "p1_overview.gif", **sk), picks)
        views = a.views.split(",")
        for reg in regions:
            for k in range(a.n):
                if "2d" in views:
                    out, i = render_phase1(ex, reg, k, od / f"p1_{reg}_{k}.gif", **sk)
                    print("wrote", out, "(test state", i, ")")
                if "3d" in views:
                    out, i = render_phase1_3d(ex, reg, k, od / f"p1_{reg}_{k}_3d.gif", chaser_scale=a.chaser_scale, **sk,
                                              **cam3d())
                    print("wrote", out, "(test state", i, ")")
        r = ex["regions"]
        lines = ["| region | test states | hover LQR recovers | learned backup recovers | LQR fails, learned recovers |",
                 "|---|---|---|---|---|"]
        for reg, d in r.items():
            lines.append(f"| {reg} | {len(d['v'])} | {d['lqr']['ok'].mean():.3f} | {d['rl']['ok'].mean():.3f} | "
                         f"{int((~d['lqr']['ok'] & d['rl']['ok']).sum())} |")
        (od / "p1_table.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
    else:
        print("wrote", render_skins(a.out))


if __name__ == "__main__":
    main()
